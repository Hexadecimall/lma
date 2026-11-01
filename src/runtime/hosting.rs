//! A single-resident-model host. API model selection never shells out or accepts
//! arbitrary remote filesystem paths. Read leases keep streaming workers alive.
use super::server::{ApiError, BODY_LIMIT, IMAGE_BODY_LIMIT};
use crate::{
    config::ServeConfig,
    models::{
        self, DiscoveryOptions,
        modelfile::{self, Definition, DefinitionMessage, Parameters},
        registry::{self, Entry},
    },
};
use axum::{
    Json, Router,
    body::{Body, Bytes, HttpBody},
    extract::{Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};

use serde_json::{Value, json};
use std::{
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    sync::{Mutex, OwnedRwLockReadGuard, RwLock},
    task::JoinHandle,
};
use tokio_stream::Stream;
use tower::ServiceExt;

struct Loaded {
    app: Router,
    worker: JoinHandle<()>,
    config: ServeConfig,
}
#[derive(Clone)]
pub(super) struct Host {
    base: ServeConfig,
    loaded: Arc<RwLock<Option<Loaded>>>,
    operations: Arc<Mutex<()>>,
    epoch: Arc<AtomicU64>,
}
type Lease = OwnedRwLockReadGuard<Option<Loaded>>;

struct OllamaRequest {
    body: Value,
    chat: bool,
    load_only: bool,
    context: Option<u32>,
}

impl OllamaRequest {
    fn new(body: &Value, chat: bool, load_only: bool) -> Result<Self, ApiError> {
        let context = body
            .get("options")
            .map(|options| {
                let options = options
                    .as_object()
                    .ok_or_else(|| invalid("options must be an object"))?;
                options
                    .get("num_ctx")
                    .map(|value| {
                        value
                            .as_u64()
                            .filter(|&n| (2..=i32::MAX as u64).contains(&n))
                            .map(|n| n as u32)
                            .ok_or_else(|| {
                                invalid(
                                    "options.num_ctx must be an integer between 2 and 2147483647",
                                )
                            })
                    })
                    .transpose()
            })
            .transpose()?
            .flatten();
        Ok(Self {
            body: body.clone(),
            chat,
            load_only,
            context,
        })
    }

    fn apply(&self, config: &mut ServeConfig) -> Result<(), ApiError> {
        if let Some(context) = self.context {
            config.context = context;
        }
        super::ollama::validate_request(config, &self.body, self.chat, self.load_only)
    }
}

// Run on a blocking worker and retain these paths from load time, rather than
// resolving the resident's paths again after a symlink or catalog change.
fn resident_config(mut config: ServeConfig) -> Result<ServeConfig, String> {
    let canonical = |path: &std::path::Path| {
        std::fs::canonicalize(path).map_err(|error| {
            format!(
                "cannot identify resident file '{}': {error}",
                path.display()
            )
        })
    };
    // Preserve the public label if canonicalization changes a symlink's basename.
    if config.model_id.is_none() {
        config.model_id = config
            .model
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
    }
    // Startup resolution may omit the default store, while catalog selection
    // supplies it. Record the same effective store for native Ollama blob paths.
    let store = config.ollama_dir.take().or_else(|| {
        let parent = config.model.parent()?;
        (parent.file_name()? == "blobs")
            .then(|| parent.parent().map(std::path::Path::to_path_buf))
            .flatten()
    });
    config.ollama_dir = store.map(|path| std::fs::canonicalize(&path).unwrap_or(path));
    config.model = canonical(&config.model)?;
    config.projector = config.projector.as_deref().map(canonical).transpose()?;
    Ok(config)
}

fn same_resident(left: &ServeConfig, right: &ServeConfig) -> bool {
    // Exhaustive destructuring makes new settings require an explicit decision.
    // Labels and explicit-override bookkeeping may differ, but effective load
    // values and the complete prompt/default profile must match the old router.
    let ServeConfig {
        model,
        model_id: _,
        ollama_dir,
        projector,
        chat_template,
        // Terminal display/request policy does not change resident model state.
        chat_tokens: _,
        definition,
        explicit_load: _,
        prompt_cache,
        kv_cache,
        memory_fallback,
        expert_cache_mib,
        host,
        port,
        api,
        client,
        gpu_layers,
        context,
        threads,
        batch,
        micro_batch,
        flash,
        api_key,
    } = left;
    model == &right.model
        && ollama_dir == &right.ollama_dir
        && projector == &right.projector
        && chat_template == &right.chat_template
        && definition == &right.definition
        && prompt_cache == &right.prompt_cache
        && kv_cache == &right.kv_cache
        && memory_fallback == &right.memory_fallback
        && expert_cache_mib == &right.expert_cache_mib
        && host == &right.host
        && port == &right.port
        && api == &right.api
        && client == &right.client
        && gpu_layers == &right.gpu_layers
        && context == &right.context
        && threads == &right.threads
        && batch == &right.batch
        && micro_batch == &right.micro_batch
        && flash == &right.flash
        && api_key == &right.api_key
}

// The startup config has already had its alias defaults applied. Undo only
// definition-derived settings when selecting another model; explicit host CLI
// values remain authoritative and must survive every subsequent selection.
fn selection_base(base: &ServeConfig) -> ServeConfig {
    let mut config = base.clone();
    config.definition = None;
    let Some(definition) = &base.definition else {
        return config;
    };
    let defaults = ServeConfig::default();
    let implicit = |key: &str| !base.explicit_load.iter().any(|value| value == key);
    let parameters = &definition.parameters;
    if definition.template.is_some() && implicit("template") {
        config.chat_template = defaults.chat_template;
    }
    if parameters.num_ctx.is_some() && implicit("num_ctx") {
        config.context = defaults.context;
    }
    if parameters.num_gpu.is_some() && implicit("num_gpu") {
        config.gpu_layers = defaults.gpu_layers;
    }
    if parameters.num_thread.is_some() && implicit("num_thread") {
        config.threads = defaults.threads;
    }
    if parameters.num_batch.is_some() && implicit("num_batch") {
        config.batch = defaults.batch;
    }
    if (parameters.u_batch.is_some() || parameters.num_batch.is_some()) && implicit("u_batch") {
        config.micro_batch = defaults.micro_batch.min(config.batch);
    }
    config
}

impl Host {
    pub async fn new(config: ServeConfig) -> Result<(Router, Self), String> {
        super::server::validate_hosting(&config)?;
        let (config, loaded) = if config.model.as_os_str().is_empty() {
            (config, None)
        } else {
            let config = tokio::task::spawn_blocking(move || resident_config(config))
                .await
                .map_err(|error| error.to_string())??;
            let (app, worker) = super::server::build(config.clone()).await?;
            let loaded = Loaded {
                app,
                worker,
                config: config.clone(),
            };
            (config, Some(loaded))
        };
        let host = Self {
            base: config,
            loaded: Arc::new(RwLock::new(loaded)),
            operations: Arc::new(Mutex::new(())),
            epoch: Arc::new(AtomicU64::new(0)),
        };
        let mut app = Router::new();
        if matches!(host.base.api.as_str(), "all" | "openai") {
            app = app
                .route("/", get(super::web::index))
                .route("/chat.js", get(super::web::script))
                .route("/style.css", get(super::web::style));
        }
        Ok((app.fallback(dispatch).with_state(host.clone()), host))
    }
    pub async fn shutdown(&self) -> Result<(), String> {
        let _operation = self.operations.lock().await;
        let mut state = self.loaded.write().await;
        if let Some(loaded) = state.take() {
            drop(loaded.app);
            loaded.worker.await.map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn catalog(&self) -> models::ModelCatalog {
        models::discover(DiscoveryOptions {
            local_dirs: models::discovery::default_local_dirs(),
            ollama_dir: self.base.ollama_dir.clone(),
        })
    }
    async fn inventory(&self) -> Result<models::ModelCatalog, ApiError> {
        let host = self.clone();
        tokio::task::spawn_blocking(move || host.catalog())
            .await
            .map_err(|e| invalid(e.to_string()))
    }
    fn target(&self, model: &str) -> Result<ServeConfig, String> {
        let mut config = self.target_metadata(model)?;
        super::create::apply_load_defaults(&mut config)?;
        Ok(config)
    }
    fn target_metadata(&self, model: &str) -> Result<ServeConfig, String> {
        let catalog = self.catalog();
        let normalized = registry::name(model).unwrap_or_else(|_| model.into());
        let matches: Vec<_> = catalog
            .models
            .iter()
            .filter(|m| m.name == model || m.name == normalized || m.id == model)
            .collect();
        let selected = matches
            .iter()
            .find(|m| m.source == "lma")
            .copied()
            .or_else(|| matches.first().copied());
        let mut config = self.base.clone();
        if let Some(selected) = selected {
            config = selection_base(&self.base);
            config.model = if selected.ollama {
                format!("ollama:{}", selected.model).into()
            } else if selected.id == format!("lma:{}", selected.name) {
                // Keep a registry alias visible to metadata resolution so profiles
                // sharing one weights path cannot resolve as a sibling alias.
                selected.id.clone().into()
            } else {
                selected.model.clone().into()
            };
            config.ollama_dir = selected.ollama_dir.clone().or(config.ollama_dir);
            config.projector = selected.projector.clone();
            // Preserve the selected catalog alias for definition lookup even
            // when several named profiles point to the same canonical weights.
            config.model_id = Some(selected.name.clone());
            super::resolve_model_metadata(&mut config)?;
            config.model_id = Some(selected.name.clone());
        } else if self.base.model_id.as_deref() == Some(model)
            || self.base.model.file_name().is_some_and(|n| n == model)
        {
            // A startup --path model is part of this host's explicit allowlist.
        } else {
            return Err(format!(
                "installed model '{model}' not found; use lma l or pull it first"
            ));
        }
        Ok(config)
    }
    async fn lease(&self, model: Option<String>) -> Result<Lease, ApiError> {
        self.lease_with(
            model,
            |host, model| host.target(&model),
            super::server::build,
        )
        .await
    }

    async fn lease_with<
        F: Future<Output = Result<(Router, JoinHandle<()>), String>> + Send + 'static,
    >(
        &self,
        model: Option<String>,
        resolve: impl FnOnce(Host, String) -> Result<ServeConfig, String> + Send + 'static,
        build: impl FnOnce(ServeConfig) -> F + Send + 'static,
    ) -> Result<Lease, ApiError> {
        self.lease_request_with(model, None, resolve, build).await
    }

    async fn lease_ollama(
        &self,
        model: Option<String>,
        request: OllamaRequest,
    ) -> Result<Lease, ApiError> {
        self.lease_request_with(
            model,
            Some(request),
            |host, model| host.target(&model),
            super::server::build,
        )
        .await
    }

    async fn lease_request_with<
        F: Future<Output = Result<(Router, JoinHandle<()>), String>> + Send + 'static,
    >(
        &self,
        model: Option<String>,
        request: Option<OllamaRequest>,
        resolve: impl FnOnce(Host, String) -> Result<ServeConfig, String> + Send + 'static,
        build: impl FnOnce(ServeConfig) -> F + Send + 'static,
    ) -> Result<Lease, ApiError> {
        // The exact public name addresses the loaded snapshot, even if its
        // catalog entry or startup symlink has changed. Unnamed startup files
        // already use their original filename as model_id (see resident_config).
        // All unfamiliar selectors must resolve; never strip source prefixes or
        // infer equivalence from a new display name or a weights-file basename.
        if let Ok(guard) = self.loaded.clone().try_read_owned()
            && let Some(loaded) = guard.as_ref()
            && model
                .as_deref()
                .is_none_or(|name| loaded.config.model_id.as_deref() == Some(name))
            && request
                .as_ref()
                .and_then(|r| r.context)
                .is_none_or(|context| context == loaded.config.context)
        {
            // Empty load requests bypass the resident router, so validate here.
            // Ordinary generation is parsed only once on the steady-state path.
            if let Some(request) = &request
                && request.load_only
            {
                request.apply(&mut loaded.config.clone())?;
            }
            return Ok(guard);
        }
        // The owned task finishes a switch even if its HTTP caller disconnects.
        let host = self.clone();
        tokio::spawn(async move {
            let _operation = host
                .operations
                .clone()
                .try_lock_owned()
                .map_err(|_| busy())?;
            // Resizing the current public name uses its loaded snapshot, not a
            // new catalog lookup that could retarget an alias behind the client.
            let snapshot = if request.is_some() {
                host.loaded.try_read().ok().and_then(|guard| {
                    guard.as_ref().filter(|loaded| model.as_deref().is_none_or(|name| {
                        loaded.config.model_id.as_deref() == Some(name)
                    })).map(|loaded| loaded.config.clone())
                })
            } else {
                None
            };
            let mut config = if let Some(config) = snapshot {
                config
            } else {
                let model = model
                    .or_else(|| host.base.model_id.clone())
                    .or_else(|| host.base.model.file_name().map(|name| name.to_string_lossy().into_owned()))
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| invalid("model is required while no model is loaded; choose an installed name from /api/tags or /v1/models"))?;
                let resolver = host.clone();
                tokio::task::spawn_blocking(move || resident_config(resolve(resolver, model)?))
                    .await
                    .map_err(|e| invalid(e.to_string()))?
                    .map_err(invalid)?
            };
            // Validate the complete Ollama request against the proposed settings
            // before dropping a working model or allocating a replacement.
            if let Some(request) = &request {
                request.apply(&mut config)?;
            }
            // Reuse under a read lease so an equivalent selector also works while
            // another response still holds the resident model's lease.
            if let Ok(guard) = host.loaded.clone().try_read_owned()
                && let Some(loaded) = guard.as_ref()
                && same_resident(&loaded.config, &config)
            {
                return Ok(guard);
            }
            let mut guard = host.loaded.clone().try_write_owned().map_err(|_| busy())?;
            if let Some(previous) = guard.take() {
                drop(previous.app);
                previous.worker.await.map_err(|e| invalid(e.to_string()))?;
            }
            let (app, worker) = build(config.clone()).await.map_err(invalid)?;
            *guard = Some(Loaded {
                app,
                worker,
                config,
            });
            Ok(tokio::sync::OwnedRwLockWriteGuard::downgrade(guard))
        })
        .await
        .map_err(|e| invalid(e.to_string()))?
    }
    async fn stop(&self, model: Option<&str>) -> Result<Value, ApiError> {
        let host = self.clone();
        let model = model.map(str::to_owned);
        // A disconnected stop caller must not abandon native teardown and let
        // a new load start while the old worker still owns its engine.
        tokio::spawn(async move {
            let _operation = host
                .operations
                .clone()
                .try_lock_owned()
                .map_err(|_| busy())?;
            let mut guard = host.loaded.clone().try_write_owned().map_err(|_| busy())?;
            if let Some(current) = guard.as_ref()
                && model.as_deref().is_some_and(|name| {
                    current.config.model_id.as_deref() != Some(name)
                        && current.config.model.file_name().is_none_or(|n| n != name)
                })
            {
                return Err(invalid("requested model is not currently loaded"));
            }
            host.epoch.fetch_add(1, Ordering::SeqCst);
            if let Some(previous) = guard.take() {
                drop(previous.app);
                previous.worker.await.map_err(|e| invalid(e.to_string()))?;
            }
            Ok(json!({"status":"success", "loaded":false}))
        })
        .await
        .map_err(|e| invalid(e.to_string()))?
    }
    fn expire(&self, epoch: u64, keep: Option<Duration>) {
        let Some(keep) = keep else {
            return;
        };
        let host = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(keep).await;
            if host.epoch.load(Ordering::SeqCst) == epoch {
                // Timers do not interrupt active clients or race newer selections.
                let Ok(_operation) = host.operations.try_lock() else {
                    return;
                };
                let Ok(mut guard) = host.loaded.try_write() else {
                    return;
                };
                if host.epoch.load(Ordering::SeqCst) != epoch {
                    return;
                }
                if let Some(previous) = guard.take() {
                    drop(previous.app);
                    let _ = previous.worker.await;
                }
            }
        });
    }
}
fn invalid(message: impl Into<String>) -> ApiError {
    ApiError::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}
fn busy() -> ApiError {
    ApiError::new(
        StatusCode::CONFLICT,
        "model_busy",
        "model is serving a request or another model operation is active; cancel the request before switching or stopping",
    )
}
fn keep_alive(value: Option<&Value>) -> Result<Option<Duration>, ApiError> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(Some(Duration::from_secs(300)));
    };
    if value.as_i64() == Some(-1) || value.as_str() == Some("-1") {
        return Ok(None);
    }
    let seconds = if let Some(n) = value.as_f64() {
        n
    } else if let Some(text) = value.as_str() {
        let (number, factor) = if let Some(v) = text.strip_suffix("ms") {
            (v, 0.001)
        } else if let Some(v) = text.strip_suffix('s') {
            (v, 1.0)
        } else if let Some(v) = text.strip_suffix('m') {
            (v, 60.0)
        } else if let Some(v) = text.strip_suffix('h') {
            (v, 3600.0)
        } else {
            (text, 1.0)
        };
        number
            .parse::<f64>()
            .map_err(|_| invalid("invalid keep_alive duration"))?
            * factor
    } else {
        return Err(invalid(
            "keep_alive must be seconds, a duration string, or -1",
        ));
    };
    if !seconds.is_finite() || !(0.0..=31_536_000.0).contains(&seconds) {
        return Err(invalid(
            "keep_alive must be -1 or between zero and one year",
        ));
    }
    Ok(Some(Duration::from_secs_f64(seconds)))
}
fn allowed(host: &Host, request: &Request) -> bool {
    let Some(expected) = &host.base.api_key else {
        return true;
    };
    request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split_once(' '))
        .is_some_and(|(scheme, token)| scheme.eq_ignore_ascii_case("bearer") && token == expected)
        || matches!(
            request.uri().path(),
            "/v1/messages" | "/v1/messages/count_tokens"
        ) && request
            .headers()
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            == Some(expected)
}
async fn dispatch(State(host): State<Host>, request: Request) -> Response {
    let ollama = request.uri().path().starts_with("/api/");
    match handle(host, request).await {
        Ok(response) => response,
        Err(error) if ollama => error.ollama_response(),
        Err(error) => error.into_response(),
    }
}
async fn handle(host: Host, request: Request) -> Result<Response, ApiError> {
    let path = request.uri().path().to_owned();
    if path == "/health" {
        return Ok(
            Json(json!({"status":"ok", "loaded":host.loaded.read().await.is_some()}))
                .into_response(),
        );
    }
    if path == "/" && !matches!(host.base.api.as_str(), "all" | "openai") {
        if !matches!(
            *request.method(),
            axum::http::Method::GET | axum::http::Method::HEAD
        ) {
            return Err(ApiError::new(
                StatusCode::METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "method not allowed",
            ));
        }
        let response = Json(super::server::api_info(&host.base.api)).into_response();
        return Ok(if request.method() == axum::http::Method::HEAD {
            let (parts, _) = response.into_parts();
            Response::from_parts(parts, Body::empty())
        } else {
            response
        });
    }
    if !allowed(&host, &request) {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "a valid API key is required",
        ));
    }
    let family = super::server::api_family(&path);
    if !matches!(path.as_str(), "/api/stop" | "/v1/models")
        && (family.is_none() || (host.base.api != "all" && family != Some(host.base.api.as_str())))
    {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "API family is disabled",
        ));
    }
    if matches!(path.as_str(), "/props" | "/lma/v1/status") {
        if !matches!(
            *request.method(),
            axum::http::Method::GET | axum::http::Method::HEAD
        ) {
            return Err(ApiError::new(
                StatusCode::METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "method not allowed",
            ));
        }
        // Read-only metadata must not reload an unloaded model or cancel its expiry.
        let lease = host.loaded.clone().read_owned().await;
        let loaded = lease.as_ref().ok_or_else(|| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_not_loaded",
                "no model loaded",
            )
        })?;
        return loaded
            .app
            .clone()
            .oneshot(request)
            .await
            .map_err(|_| invalid("resident router failed"));
    }
    if matches!(
        path.as_str(),
        "/completion" | "/tokenize" | "/detokenize" | "/lma/v1/generate" | "/lma/v1/chat"
    ) && request.method() != axum::http::Method::POST
    {
        return Err(ApiError::new(
            StatusCode::METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "method not allowed",
        ));
    }
    if request.method() == axum::http::Method::GET {
        match path.as_str() {
            "/api/version" => {
                return Ok(Json(
                    json!({"version":concat!("lma/",env!("CARGO_PKG_VERSION")), "server":"lma"}),
                )
                .into_response());
            }
            "/api/tags" | "/v1/models" => {
                let catalog = host.inventory().await?;
                let mut entries: Vec<Value> = catalog.models.iter().map(|m| if path == "/v1/models" { json!({"id":m.name,"object":"model","owned_by":m.source}) } else { json!({"name":m.name,"model":m.name,"size":m.size_bytes,"lma":{"source":m.source,"id":m.id}}) }).collect();
                if let Some(current) = host.loaded.read().await.as_ref() {
                    let name = current.config.model_id.clone().unwrap_or_else(|| {
                        current
                            .config
                            .model
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned()
                    });
                    if !catalog.models.iter().any(|m| m.name == name) {
                        entries.push(if path == "/v1/models" {
                            json!({"id":name,"object":"model","owned_by":"lma"})
                        } else {
                            json!({"name":name,"model":name})
                        });
                    }
                }
                return Ok(Json(if path == "/v1/models" {
                    json!({"object":"list","data":entries})
                } else {
                    json!({"models":entries,"lma":{"warnings":catalog.warnings}})
                })
                .into_response());
            }
            "/api/ps" => {
                let loaded = host.loaded.read().await;
                let models: Vec<Value> = loaded.iter().map(|m| json!({"name":m.config.model_id.clone().unwrap_or_else(||m.config.model.file_name().unwrap_or_default().to_string_lossy().into_owned()),"context_length":m.config.context})).collect();
                return Ok(Json(json!({"models":models})).into_response());
            }
            _ => {
                return Err(ApiError::new(
                    StatusCode::NOT_FOUND,
                    "not_found",
                    "route not found",
                ));
            }
        }
    }
    if request.method() != axum::http::Method::POST
        && !(path == "/api/delete" && request.method() == axum::http::Method::DELETE)
    {
        return Err(ApiError::new(
            StatusCode::METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "method not allowed",
        ));
    }
    let (mut parts, body) = request.into_parts();
    let limit = if path == "/api/create" {
        modelfile::MAX_BYTES
    } else if matches!(
        path.as_str(),
        "/completion" | "/tokenize" | "/detokenize" | "/lma/v1/generate"
    ) {
        BODY_LIMIT
    } else {
        IMAGE_BODY_LIMIT
    };
    let bytes = axum::body::to_bytes(body, limit).await.map_err(|_| {
        ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
            "request exceeds body limit",
        )
    })?;
    let mut body: Value =
        serde_json::from_slice(&bytes).map_err(|_| invalid("expected a JSON object"))?;
    if !body.is_object() {
        return Err(invalid("expected a JSON object"));
    }
    match family {
        Some("llama") => super::llama_api::validate_body(&path, &body)?,
        Some("lma") => super::native_api::validate_body(&body, path == "/lma/v1/chat")?,
        _ => {}
    }
    let selector = if matches!(family, Some("llama" | "lma")) {
        body.get("model")
    } else {
        body.get("model").or_else(|| body.get("name"))
    };
    let model = selector
        .filter(|value| !value.is_null())
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("model must be a string"))
        })
        .transpose()?;
    if path == "/api/stop" {
        return Ok(Json(host.stop(model.as_deref()).await?).into_response());
    }
    if matches!(
        path.as_str(),
        "/api/pull" | "/api/copy" | "/api/create" | "/api/delete"
    ) {
        return admin(host, &path, body).await;
    }
    if path == "/api/show" {
        let name = model.ok_or_else(|| invalid("model is required"))?;
        let resolver = host.clone();
        let config = tokio::task::spawn_blocking(move || resolver.target_metadata(&name))
            .await
            .map_err(|e| invalid(e.to_string()))?
            .map_err(invalid)?;
        let value = tokio::task::spawn_blocking(move || model_show(config))
            .await
            .map_err(|e| invalid(e.to_string()))?
            .map_err(invalid)?;
        return Ok(Json(value).into_response());
    }
    let generation = matches!(
        path.as_str(),
        "/api/chat"
            | "/api/generate"
            | "/api/embed"
            | "/api/embeddings"
            | "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/responses"
            | "/v1/messages"
            | "/v1/messages/count_tokens"
            | "/v1/embeddings"
            | "/completion"
            | "/tokenize"
            | "/detokenize"
            | "/lma/v1/generate"
            | "/lma/v1/chat"
    );
    if !generation {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "route not found",
        ));
    }
    let keep = if family == Some("ollama") {
        keep_alive(body.get("keep_alive"))?
    } else {
        None
    };
    if family == Some("ollama") {
        body.as_object_mut().unwrap().remove("keep_alive");
    }
    // Ollama's empty generate request explicitly loads or unloads the chosen model.
    let load_only = path == "/api/generate"
        && body.get("prompt").is_none_or(|v| v == "")
        && body
            .get("images")
            .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty));
    let ollama_request = matches!(path.as_str(), "/api/generate" | "/api/chat")
        .then(|| OllamaRequest::new(&body, path == "/api/chat", load_only))
        .transpose()?;
    if load_only {
        if keep == Some(Duration::ZERO) {
            let mut config = host.base.clone();
            ollama_request.as_ref().unwrap().apply(&mut config)?;
            host.stop(model.as_deref()).await?;
        } else {
            let lease = host
                .lease_ollama(model.clone(), ollama_request.unwrap())
                .await?;
            drop(lease);
            let epoch = host.epoch.fetch_add(1, Ordering::SeqCst) + 1;
            host.expire(epoch, keep);
        }
        return Ok(Json(json!({"model":model,"response":"","done":true,"done_reason":if keep == Some(Duration::ZERO) { "unload" } else { "load" }})).into_response());
    }
    let tokenizer = matches!(path.as_str(), "/tokenize" | "/detokenize");
    if tokenizer && body.get("model").is_some() {
        return Err(invalid(
            "tokenizer utilities use the current model; model selection is not supported in their request bodies",
        ));
    }
    let lease = if let Some(request) = ollama_request {
        host.lease_ollama(model, request).await?
    } else {
        host.lease(model).await?
    };
    let loaded = lease.as_ref().ok_or_else(|| invalid("no model loaded"))?;
    let id = loaded.config.model_id.clone().unwrap_or_else(|| {
        loaded
            .config
            .model
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });
    if !tokenizer {
        body["model"] = json!(id);
    }
    let app = loaded.app.clone();
    let epoch = host.epoch.fetch_add(1, Ordering::SeqCst) + 1;
    parts.headers.remove(header::CONTENT_LENGTH);
    let request = Request::from_parts(
        parts,
        Body::from(serde_json::to_vec(&body).map_err(|e| invalid(e.to_string()))?),
    );
    let response = app
        .oneshot(request)
        .await
        .map_err(|e| invalid(e.to_string()))?;
    let (parts, body) = response.into_parts();
    Ok(Response::from_parts(
        parts,
        Body::from_stream(LeasedBody {
            body,
            lease: Some(lease),
            host,
            epoch,
            keep,
        }),
    ))
}

fn model_show(config: ServeConfig) -> Result<Value, String> {
    let info = models::gguf::inspect(&config.model)?;
    // Serialize only public model metadata, never the hosting configuration/key.
    Ok(
        json!({"model_info":info.model_info(), "definition":config.definition,
        "lma":{"model_switching":true,"metadata_scope":"bounded GGUF header and stored definition"}}),
    )
}

fn installed_selector(value: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    if value.is_empty()
        || value.len() > 4096
        || value.trim() != value
        || value.chars().any(char::is_control)
        || value.contains("://")
        || value.starts_with(['/', '\\', '.'])
        || value
            .split(['/', '\\'])
            .any(|part| matches!(part, "." | ".."))
        || (bytes.len() > 2
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && matches!(bytes[2], b'/' | b'\\'))
    {
        return Err("HTTP FROM/source must be an installed model name or exact catalog ID, not a URL or local/relative path; use the CLI to register local files".into());
    }
    // Catalog IDs may contain store paths. They are opaque exact selectors, not
    // paths to open; Host::target_metadata checks their installed allowlist membership.
    if value.starts_with("lma:") || value.starts_with("ollama:") {
        return Ok(());
    }
    if value.contains('\\') {
        return Err(
            "HTTP FROM/source does not accept local paths; use an installed selector".into(),
        );
    }
    registry::name(value).map(|_| ())
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ApiCreate {
    model: Option<String>,
    name: Option<String>,
    modelfile: Option<String>,
    from: Option<String>,
    system: Option<String>,
    template: Option<String>,
    parameters: Option<Parameters>,
    messages: Option<Vec<DefinitionMessage>>,
    license: Option<String>,
    #[serde(rename = "stream")]
    _stream: Option<bool>,
}

enum ProfileRequest {
    Create {
        label: String,
        definition: Definition,
    },
    Copy {
        source: String,
        destination: String,
    },
}
impl ProfileRequest {
    fn parse(path: &str, body: &Value) -> Result<Option<Self>, String> {
        if path == "/api/copy" {
            let source = body["source"]
                .as_str()
                .ok_or("source must be an installed model selector")?
                .to_owned();
            installed_selector(&source)?;
            let destination = registry::name(
                body["destination"]
                    .as_str()
                    .ok_or("destination is required")?,
            )?;
            return Ok(Some(Self::Copy {
                source,
                destination,
            }));
        }
        if path != "/api/create" {
            return Ok(None);
        }
        let request: ApiCreate = serde_json::from_value(body.clone())
            .map_err(|e| format!("invalid create request: {e}"))?;
        if body.get("model").is_some() && body.get("name").is_some() {
            return Err("specify model or name, not both".into());
        }
        let label = registry::name(&request.model.or(request.name).ok_or("model is required")?)?;
        let definition = if body.get("modelfile").is_some() {
            if [
                "from",
                "system",
                "template",
                "parameters",
                "messages",
                "license",
            ]
            .iter()
            .any(|key| body.get(key).is_some())
            {
                return Err("modelfile cannot be mixed with structured definition fields".into());
            }
            modelfile::parse(&request.modelfile.ok_or("modelfile must be a string")?)?
        } else {
            let definition = Definition {
                from: request.from.ok_or("create requires from or modelfile")?,
                system: request.system,
                template: request.template,
                parameters: request.parameters.unwrap_or_default(),
                messages: request.messages.unwrap_or_default(),
                license: request.license,
            };
            definition.validate()?;
            definition
        };
        installed_selector(&definition.from)?;
        if body.get("modelfile").is_none()
            && !["system", "template", "parameters", "messages", "license"]
                .iter()
                .any(|key| body.get(key).is_some())
        {
            // Legacy {from} creation is a profile copy, not a new empty profile.
            return Ok(Some(Self::Copy {
                source: definition.from,
                destination: label,
            }));
        }
        Ok(Some(Self::Create { label, definition }))
    }

    fn publish(
        self,
        root: &Path,
        resolve: impl FnOnce(&str) -> Result<ServeConfig, String>,
    ) -> Result<Value, String> {
        let (label, config, definition) = match self {
            Self::Copy {
                source,
                destination,
            } => {
                let config = resolve(&source)?;
                let definition = config.definition.clone();
                (destination, config, definition)
            }
            Self::Create { label, definition } => {
                let config = resolve(&definition.from)?;
                let definition =
                    super::create::inherit_definition(definition, config.definition.clone())?;
                (label, config, Some(definition))
            }
        };
        let entry = Entry {
            path: config.model,
            projector: config.projector,
        };
        let label = match definition {
            Some(definition) => registry::put_definition(root, &label, entry, definition)?,
            None => registry::put(root, &label, entry)?,
        };
        Ok(json!({"status":"success", "model":label}))
    }
}

async fn admin(host: Host, path: &str, body: Value) -> Result<Response, ApiError> {
    admin_with(
        host,
        path,
        body,
        models::discovery::default_models_dir,
        |host, selector| host.target_metadata(&selector),
    )
    .await
}

async fn admin_with(
    host: Host,
    path: &str,
    body: Value,
    root: impl FnOnce() -> Result<PathBuf, String> + Send + 'static,
    resolve: impl FnOnce(Host, String) -> Result<ServeConfig, String> + Send + 'static,
) -> Result<Response, ApiError> {
    if host.base.api_key.is_none()
        && !matches!(host.base.host.as_str(), "127.0.0.1" | "localhost" | "::1")
    {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "admin_auth_required",
            "non-loopback administration requires an API key; start the host with `lma s MODEL -ay KEY`",
        ));
    }
    let model = body
        .get("model")
        .or_else(|| body.get("name"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let source = body
        .get("source")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let stream = body
        .get("stream")
        .map(|v| v.as_bool().ok_or_else(|| invalid("stream must be boolean")))
        .transpose()?
        .unwrap_or(true);
    let allowed: &[&str] = match path {
        "/api/pull" => &["model", "name", "stream"],
        "/api/copy" => &["source", "destination", "stream"],
        "/api/delete" => &["model", "name", "source", "stream"],
        _ => &[
            "model",
            "name",
            "from",
            "modelfile",
            "system",
            "template",
            "parameters",
            "messages",
            "license",
            "stream",
        ],
    };
    if body
        .as_object()
        .unwrap()
        .keys()
        .any(|key| !allowed.contains(&key.as_str()))
    {
        return Err(invalid(
            "unsupported administration fields; refusing to silently ignore model options",
        ));
    }
    // Parse and reject HTTP path/URL inputs before any root lookup, catalog
    // scan or resolver IO. Do not use file preparation's cwd/path probing here.
    let profile = ProfileRequest::parse(path, &body).map_err(invalid)?;
    let path = path.to_owned();
    let operation = host
        .operations
        .clone()
        .try_lock_owned()
        .map_err(|_| busy())?;
    if path == "/api/delete" {
        let mut state = host.loaded.try_write().map_err(|_| busy())?;
        if let Some(previous) = state.take() {
            drop(previous.app);
            previous.worker.await.map_err(|e| invalid(e.to_string()))?;
        }
        host.epoch.fetch_add(1, Ordering::SeqCst);
    }
    let task = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let _operation = operation;
        let root = root()?;
        if let Some(profile) = profile {
            return profile.publish(&root, |selector| resolve(host, selector.to_owned()));
        }
        match path.as_str() {
            "/api/pull" => {
                let model = model.ok_or("model is required")?;
                let label = registry::name(&model)?;
                if registry::read(&root)?.contains_key(&label) {
                    return Ok(json!({"status":"success","model":label}));
                }
                let entry = crate::fetch::pull_ollama(&model, &root)?;
                let label = registry::put(&root, &model, entry)?;
                Ok(json!({"status":"success","model":label}))
            }

            "/api/delete" => models::delete::run(
                &model.ok_or("model is required")?,
                source.as_deref(),
                host.base.ollama_dir.as_deref(),
            ),
            _ => Err("unsupported administration operation".into()),
        }
    });
    if !stream {
        return Ok(Json(
            task.await
                .map_err(|e| invalid(e.to_string()))?
                .map_err(invalid)?,
        )
        .into_response());
    }
    let (sender, receiver) =
        tokio::sync::mpsc::channel::<Result<Bytes, std::convert::Infallible>>(2);
    tokio::spawn(async move {
        let result = match task.await {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => json!({"error":error}),
            Err(error) => json!({"error":error.to_string()}),
        };
        let _ = sender.send(Ok(Bytes::from(format!("{result}\n")))).await;
    });
    Ok((
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(receiver)),
    )
        .into_response())
}
struct LeasedBody {
    body: Body,
    lease: Option<Lease>,
    host: Host,
    epoch: u64,
    keep: Option<Duration>,
}
impl Drop for LeasedBody {
    fn drop(&mut self) {
        self.lease.take();
        self.host.expire(self.epoch, self.keep);
    }
}
impl Stream for LeasedBody {
    type Item = Result<Bytes, axum::Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.body).poll_frame(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(frame))) => {
                    if let Ok(data) = frame.into_data() {
                        return Poll::Ready(Some(Ok(data)));
                    }
                }
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                Poll::Ready(None) => {
                    self.lease.take();
                    return Poll::Ready(None);
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Fake {
        cached: bool,
    }
    impl super::super::inference::Engine for Fake {
        fn tokenize(
            &self,
            _: super::super::inference::TokenizeRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Vec<i32>, super::super::inference::InferenceError> {
            Ok(vec![42])
        }
        fn detokenize(
            &self,
            _: Vec<i32>,
            _: &dyn Fn() -> bool,
        ) -> Result<String, super::super::inference::InferenceError> {
            Ok("fixture".into())
        }
        fn generate(
            &mut self,
            _: super::super::inference::GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<super::super::inference::Generation, super::super::inference::InferenceError>
        {
            let text = if self.cached { "cached" } else { "hello" };
            self.cached = true;
            Ok(super::super::inference::Generation {
                text: text.into(),
                prompt_tokens: 3,
                completion_tokens: 1,
                finish_reason: "stop",
            })
        }
    }
    async fn fake_host() -> Host {
        let config = ServeConfig {
            model: "fixture.gguf".into(),
            model_id: Some("fixture:latest".into()),
            ..Default::default()
        };
        fake_host_with_config(config).await
    }
    async fn fake_app(config: ServeConfig) -> Result<(Router, JoinHandle<()>), String> {
        let (inference, worker) =
            super::super::inference::InferenceService::spawn(|| Ok(Fake::default())).await?;
        let app = super::super::server::router(super::super::server::AppState {
            inference,
            model_id: config.model_id.unwrap(),
            context: config.context,
            metadata: None,
            definition: config.definition.map(Arc::new),
            api_key: config.api_key,
            next_id: Arc::new(AtomicU64::new(0)),
            api: config.api,
        });
        Ok((app, worker))
    }
    async fn fake_host_with_config(config: ServeConfig) -> Host {
        let (app, worker) = fake_app(config.clone()).await.unwrap();
        Host {
            base: config.clone(),
            loaded: Arc::new(RwLock::new(Some(Loaded {
                app,
                worker,
                config,
            }))),
            operations: Arc::new(Mutex::new(())),
            epoch: Arc::new(AtomicU64::new(0)),
        }
    }
    fn request(path: &str, body: Value) -> Request {
        Request::builder()
            .method("POST")
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }
    async fn fixture_ollama_lease(
        host: &Host,
        body: Value,
        builds: Arc<AtomicU64>,
    ) -> Result<Lease, ApiError> {
        let model = body.get("model").and_then(Value::as_str).map(str::to_owned);
        let request = OllamaRequest::new(&body, false, false)?;
        host.lease_request_with(
            model,
            Some(request),
            |_, _| panic!("the current model must use its loaded snapshot"),
            move |config| async move {
                builds.fetch_add(1, Ordering::SeqCst);
                fake_app(config).await
            },
        )
        .await
    }

    #[tokio::test]
    async fn ollama_context_resize_reuses_the_loaded_snapshot_and_then_the_worker() {
        let host = fake_host().await;
        let builds = Arc::new(AtomicU64::new(0));
        let original = host.loaded.read().await.as_ref().unwrap().worker.id();
        for model in [Some("fixture:latest"), None] {
            let mut body = json!({"prompt":"hello", "options":{"num_ctx":8192,"num_predict":-1}});
            if let Some(model) = model {
                body["model"] = json!(model);
            }
            let lease = fixture_ollama_lease(&host, body, builds.clone())
                .await
                .unwrap();
            let loaded = lease.as_ref().unwrap();
            assert_eq!(loaded.config.context, 8192);
            assert_ne!(loaded.worker.id(), original);
            drop(lease);
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        let worker = host.loaded.read().await.as_ref().unwrap().worker.id();
        let lease = fixture_ollama_lease(&host, json!({"prompt":"follow up"}), builds.clone())
            .await
            .unwrap();
        assert_eq!(lease.as_ref().unwrap().config.context, 8192);
        assert_eq!(lease.as_ref().unwrap().worker.id(), worker);
        drop(lease);
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn ollama_idle_selection_applies_context_before_the_first_build() {
        let (_dir, config) = tokio::task::spawn_blocking(resident_fixture).await.unwrap();
        let (app, host) = Host::new(ServeConfig::default()).await.unwrap();
        let body = json!({"model":"fixture:latest","prompt":"hello","options":{"num_ctx":8192,"num_predict":-1}});
        let lease = host
            .lease_request_with(
                Some("fixture:latest".into()),
                Some(OllamaRequest::new(&body, false, false).unwrap()),
                move |_, _| Ok(config),
                |config| async move {
                    assert_eq!(config.context, 8192);
                    fake_app(config).await
                },
            )
            .await
            .unwrap();
        assert_eq!(lease.as_ref().unwrap().config.context, 8192);
        drop(lease);
        drop(app);
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn malformed_ollama_resize_never_unloads_a_working_model() {
        let host = fake_host().await;
        let builds = Arc::new(AtomicU64::new(0));
        let original = host.loaded.read().await.as_ref().unwrap().worker.id();
        for options in [
            json!({"num_ctx":0}),
            json!({"num_ctx":1}),
            json!({"num_ctx":-1}),
            json!({"num_ctx":2147483648u64}),
            json!({"num_ctx":8192.0}),
            json!({"num_ctx":null}),
            json!({"num_ctx":"8192"}),
            json!(null),
            json!({"num_ctx":8192,"num_predict":0}),
            json!({"num_ctx":8192,"num_predict":8193}),
            json!({"num_ctx":8192,"temperature":3}),
            json!({"num_ctx":8192,"top_p":0}),
            json!({"num_ctx":8192,"unsupported":true}),
        ] {
            let body = json!({"prompt":"hello", "options":options});
            let error = fixture_ollama_lease(&host, body, builds.clone())
                .await
                .err()
                .unwrap();
            assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                host.loaded.read().await.as_ref().unwrap().worker.id(),
                original
            );
        }
        for fields in [
            json!({"think":true}),
            json!({"format":"json"}),
            json!({"prompt":null}),
            json!({"stream":null}),
            json!({"unexpected":true}),
        ] {
            let mut body = json!({"prompt":"hello", "options":{"num_ctx":8192}});
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let error = fixture_ollama_lease(&host, body, builds.clone())
                .await
                .err()
                .unwrap();
            assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
        }
        assert_eq!(builds.load(Ordering::SeqCst), 0);
        assert_eq!(
            host.loaded.read().await.as_ref().unwrap().worker.id(),
            original
        );
        let lease = host.lease(None).await.unwrap();
        assert_eq!(
            fixture_generation(&lease).await["choices"][0]["text"],
            "hello"
        );
        drop(lease);
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn ollama_context_resize_rejects_active_leases_without_teardown() {
        let host = fake_host().await;
        let held = host.lease(None).await.unwrap();
        let worker = held.as_ref().unwrap().worker.id();
        let builds = Arc::new(AtomicU64::new(0));
        let error = fixture_ollama_lease(
            &host,
            json!({"prompt":"hello","options":{"num_ctx":8192}}),
            builds.clone(),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(error.into_response().status(), StatusCode::CONFLICT);
        assert_eq!(builds.load(Ordering::SeqCst), 0);
        assert_eq!(held.as_ref().unwrap().worker.id(), worker);
        assert_eq!(
            fixture_generation(&held).await["choices"][0]["text"],
            "hello"
        );
        drop(held);
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn invalid_empty_ollama_load_or_unload_does_not_touch_the_resident() {
        let host = fake_host().await;
        let worker = host.loaded.read().await.as_ref().unwrap().worker.id();
        for keep in [json!(0), json!(-1)] {
            for extra in [
                json!({"unknown":true}),
                json!({"options":{"num_ctx":1}}),
                json!({"stream":null}),
            ] {
                let mut body = json!({"model":"fixture:latest","keep_alive":keep});
                body.as_object_mut()
                    .unwrap()
                    .extend(extra.as_object().unwrap().clone());
                let response = dispatch(State(host.clone()), request("/api/generate", body)).await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                assert_eq!(
                    host.loaded.read().await.as_ref().unwrap().worker.id(),
                    worker
                );
            }
        }
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn idle_server_has_health_and_ps_without_initializing_a_model() {
        let (app, host) = Host::new(ServeConfig::default()).await.unwrap();
        assert!(host.loaded.read().await.is_none());
        for (path, expected) in [
            ("/health", json!({"status":"ok","loaded":false})),
            ("/api/ps", json!({"models":[]})),
        ] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), expected);
        }
        let response = dispatch(
            State(host.clone()),
            request("/api/generate", json!({"prompt":"hi","stream":false})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert!(
            std::str::from_utf8(&bytes)
                .unwrap()
                .contains("model is required")
        );
        assert!(host.loaded.read().await.is_none());
        drop(app);
        host.shutdown().await.unwrap();
    }
    #[tokio::test]
    async fn idle_server_preserves_auth_and_api_profile_gates() {
        let (app, host) = Host::new(ServeConfig {
            api: "openai".into(),
            api_key: Some("secret".into()),
            ..Default::default()
        })
        .await
        .unwrap();
        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        let req = Request::builder()
            .uri("/api/tags")
            .header(header::AUTHORIZATION, "Bearer secret")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::NOT_FOUND
        );
        assert!(host.loaded.read().await.is_none());
        drop(app);
        host.shutdown().await.unwrap();
    }
    #[tokio::test]
    async fn idle_first_model_selection_remains_lazy_and_retains_a_reusable_snapshot() {
        let (_directory, config) = tokio::task::spawn_blocking(resident_fixture).await.unwrap();
        let (app, host) = Host::new(ServeConfig::default()).await.unwrap();
        let builds = Arc::new(AtomicU64::new(0));
        assert!(host.loaded.read().await.is_none());
        let lease = fixture_lease(&host, "fixture:latest", config, builds.clone())
            .await
            .unwrap();
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(
            fixture_generation(&lease).await["choices"][0]["text"],
            "hello"
        );
        drop(lease);
        let lease = host.lease(None).await.unwrap();
        assert_eq!(
            fixture_generation(&lease).await["choices"][0]["text"],
            "cached"
        );
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        drop(lease);
        drop(app);
        host.shutdown().await.unwrap();
    }
    #[tokio::test]
    async fn response_lease_prevents_unload_until_client_finishes() {
        let host = fake_host().await;
        let response = handle(
            host.clone(),
            request(
                "/api/generate",
                json!({"model":"fixture:latest","prompt":"hello","raw":true,"stream":false,"keep_alive":-1}),
            ),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            host.stop(None).await.unwrap_err().into_response().status(),
            StatusCode::CONFLICT
        );
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["response"], "hello");
        assert_eq!(host.stop(None).await.unwrap()["loaded"], false);
        assert!(host.loaded.read().await.is_none());
    }
    #[tokio::test]
    async fn stop_without_model_and_invalid_admin_fields_are_safe() {
        let host = fake_host().await;
        let error = handle(
            host.clone(),
            request(
                "/api/delete",
                json!({"model":"fixture:latest","unexpected":true}),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
        assert!(host.loaded.read().await.is_some());
        let stopped = handle(host.clone(), request("/api/stop", json!({"model":null})))
            .await
            .unwrap();
        assert_eq!(stopped.status(), StatusCode::OK);
        assert!(host.loaded.read().await.is_none());
    }
    #[tokio::test]
    async fn new_families_are_forwarded_and_stream_bodies_hold_the_host_lease() {
        for api in ["all", "llama", "lma"] {
            let host = fake_host_with_config(ServeConfig {
                model: "fixture.gguf".into(),
                model_id: Some("fixture:latest".into()),
                api: api.into(),
                ..Default::default()
            })
            .await;
            for (family, path, body) in [
                (
                    "llama",
                    "/completion",
                    json!({"prompt":"hello","stream":true}),
                ),
                (
                    "lma",
                    "/lma/v1/generate",
                    json!({"prompt":"hello","stream":true,"max_tokens":null}),
                ),
                (
                    "lma",
                    "/lma/v1/chat",
                    json!({"messages":[{"role":"user","content":"hi"}],"stream":true}),
                ),
            ] {
                let response = dispatch(State(host.clone()), request(path, body)).await;
                if api != "all" && api != family {
                    assert_eq!(response.status(), StatusCode::NOT_FOUND);
                    continue;
                }
                assert_eq!(response.status(), StatusCode::OK, "{api} {path}");
                assert_eq!(
                    host.stop(None).await.unwrap_err().into_response().status(),
                    StatusCode::CONFLICT
                );
                let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
                    .await
                    .unwrap();
                let wire = std::str::from_utf8(&bytes).unwrap();
                assert!(wire.contains("hello") || wire.contains("cached"));
                assert!(wire.contains(if family == "llama" {
                    "\"stop\":true"
                } else {
                    "\"type\":\"done\""
                }));
            }
            if matches!(api, "all" | "llama") {
                for (path, body) in [
                    ("/tokenize", json!({"content":"hello"})),
                    ("/detokenize", json!({"tokens":[42]})),
                ] {
                    let response = dispatch(State(host.clone()), request(path, body)).await;
                    assert_eq!(response.status(), StatusCode::OK);
                    let bytes = axum::body::to_bytes(response.into_body(), 4096)
                        .await
                        .unwrap();
                    let value: Value = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(
                        value,
                        if path == "/tokenize" {
                            json!({"tokens":[42]})
                        } else {
                            json!({"content":"fixture"})
                        }
                    );
                }
            }
            assert!(host.loaded.read().await.is_some());
            host.shutdown().await.unwrap();
        }
    }
    #[tokio::test]
    async fn new_metadata_is_authenticated_read_only_and_does_not_reload() {
        for (api, path) in [("llama", "/props"), ("lma", "/lma/v1/status")] {
            let host = fake_host_with_config(ServeConfig {
                model: "fixture.gguf".into(),
                model_id: Some("fixture:latest".into()),
                api: api.into(),
                api_key: Some("secret".into()),
                ..Default::default()
            })
            .await;
            let get = |path: &str, auth: bool| {
                let mut req = Request::builder().method("GET").uri(path);
                if auth {
                    req = req.header(header::AUTHORIZATION, "Bearer secret");
                }
                req.body(Body::empty()).unwrap()
            };
            let response = dispatch(State(host.clone()), get("/", false)).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                dispatch(State(host.clone()), get(path, false))
                    .await
                    .status(),
                StatusCode::UNAUTHORIZED
            );
            let epoch = host.epoch.load(Ordering::SeqCst);
            assert_eq!(
                dispatch(State(host.clone()), get(path, true))
                    .await
                    .status(),
                StatusCode::OK
            );
            assert_eq!(
                host.epoch.load(Ordering::SeqCst),
                epoch,
                "metadata polling must not change idle expiry"
            );
            assert_eq!(host.stop(None).await.unwrap()["status"], "success");
            assert_eq!(
                dispatch(State(host.clone()), get(path, true))
                    .await
                    .status(),
                StatusCode::SERVICE_UNAVAILABLE
            );
            assert!(host.loaded.read().await.is_none());
        }
    }
    #[tokio::test]
    async fn new_host_routes_enforce_methods_body_limits_and_disable_other_families() {
        let host = fake_host_with_config(ServeConfig {
            model: "fixture.gguf".into(),
            model_id: Some("fixture:latest".into()),
            api: "llama".into(),
            ..Default::default()
        })
        .await;
        for path in ["/completion", "/tokenize", "/detokenize"] {
            let req = Request::builder().uri(path).body(Body::empty()).unwrap();
            assert_eq!(
                dispatch(State(host.clone()), req).await.status(),
                StatusCode::METHOD_NOT_ALLOWED
            );
            let req = Request::builder()
                .method("POST")
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(" ".repeat(BODY_LIMIT + 1)))
                .unwrap();
            assert_eq!(
                dispatch(State(host.clone()), req).await.status(),
                StatusCode::PAYLOAD_TOO_LARGE
            );
        }
        let response = dispatch(
            State(host.clone()),
            request("/tokenize", json!({"content":"hi","model":"missing"})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(host.loaded.read().await.is_some());
        assert_eq!(
            dispatch(
                State(host.clone()),
                request("/lma/v1/generate", json!({"prompt":"hi"}))
            )
            .await
            .status(),
            StatusCode::NOT_FOUND
        );
        host.shutdown().await.unwrap();
    }
    #[tokio::test]
    async fn cancelled_stop_retains_lifecycle_gates_until_worker_teardown() {
        let (finish, wait) = tokio::sync::oneshot::channel::<()>();
        let config = ServeConfig {
            model: "fixture.gguf".into(),
            ..Default::default()
        };
        let host = Host {
            base: config.clone(),
            loaded: Arc::new(RwLock::new(Some(Loaded {
                app: Router::new(),
                worker: tokio::spawn(async move {
                    let _ = wait.await;
                }),
                config,
            }))),
            operations: Arc::new(Mutex::new(())),
            epoch: Arc::new(AtomicU64::new(0)),
        };
        let stopping = host.clone();
        let caller = tokio::spawn(async move { stopping.stop(None).await });
        tokio::time::timeout(Duration::from_secs(2), async {
            while host.operations.try_lock().is_ok() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        caller.abort();
        let _ = caller.await;
        assert!(host.operations.try_lock().is_err());
        assert!(host.loaded.try_write().is_err());
        finish.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Ok(_operation) = host.operations.try_lock() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(host.loaded.read().await.is_none());
    }
    #[tokio::test]
    async fn new_host_preflight_rejects_invalid_controls_before_resident_selection() {
        for (api, path, invalids) in [
            (
                "llama",
                "/completion",
                vec![
                    json!({"prompt":"x","model":null}),
                    json!({"prompt":"x","n_predict":0}),
                    json!({"prompt":"x","name":"other"}),
                ],
            ),
            (
                "lma",
                "/lma/v1/generate",
                vec![
                    json!({"prompt":"x","model":null}),
                    json!({"prompt":"x","max_tokens":0}),
                    json!({"prompt":"x","temperature":null}),
                    json!({"prompt":"x","name":"other"}),
                ],
            ),
            (
                "llama",
                "/tokenize",
                vec![
                    json!({"content":"x","name":"other"}),
                    json!({"content":"x","model":"other"}),
                    json!({"content":"x","parse_special":null}),
                ],
            ),
        ] {
            let host = fake_host_with_config(ServeConfig {
                model: "fixture.gguf".into(),
                model_id: Some("fixture:latest".into()),
                api: api.into(),
                ..Default::default()
            })
            .await;
            let epoch = host.epoch.load(Ordering::SeqCst);
            for body in invalids {
                assert_eq!(
                    dispatch(State(host.clone()), request(path, body))
                        .await
                        .status(),
                    StatusCode::BAD_REQUEST
                );
                assert_eq!(
                    host.epoch.load(Ordering::SeqCst),
                    epoch,
                    "preflight failure must not acquire a new resident lease"
                );
                assert_eq!(
                    host.loaded
                        .read()
                        .await
                        .as_ref()
                        .unwrap()
                        .config
                        .model_id
                        .as_deref(),
                    Some("fixture:latest")
                );
            }
            host.shutdown().await.unwrap();
        }
    }
    #[tokio::test]
    async fn chat_preflight_keeps_resident_and_epoch_for_invalid_content_with_other_model() {
        let host = fake_host_with_config(ServeConfig {
            model: "fixture.gguf".into(),
            model_id: Some("fixture:latest".into()),
            api: "lma".into(),
            ..Default::default()
        })
        .await;
        let epoch = host.epoch.load(Ordering::SeqCst);
        let worker = host.loaded.read().await.as_ref().unwrap().worker.id();
        for (message, expected) in [
            (
                json!({"role":"user","content":""}),
                "message content must be nonempty",
            ),
            (
                json!({"role":"user","content":[]}),
                "message content must be nonempty",
            ),
            (
                json!({"role":"tool","content":"result"}),
                "message role must be",
            ),
            (
                json!({"role":"user","content":"bad\0text"}),
                "must not contain NUL",
            ),
            (
                json!({"role":"system","content":"x","images":["not-base64"]}),
                "only user messages",
            ),
            (
                json!({"role":"assistant","content":[{"type":"image_url","image_url":{"url":"invalid"}}]}),
                "only user messages",
            ),
            (
                json!({"role":"user","content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"invalid"}}]}),
                "image parts must precede",
            ),
            (
                json!({"role":"user","content":[{"type":"text","text":"<__"},{"type":"text","text":"media__>"}]}),
                "reserved image markers",
            ),
            (
                json!({"role":"user","content":[{"type":"text","text":"<__lma_"},{"type":"text","text":"image_0>"}]}),
                "reserved image markers",
            ),
            (
                json!({"role":"user","content":[{"type":"text","text":"<__"},{"type":"text","text":"image__>"}]}),
                "reserved image markers",
            ),
        ] {
            let response = dispatch(
                State(host.clone()),
                request(
                    "/lma/v1/chat",
                    json!({
                        "model":"other",
                        "messages":[message],
                    }),
                ),
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let error: Value = serde_json::from_slice(&bytes).unwrap();
            assert!(
                error["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(expected),
                "{error}"
            );
            assert_eq!(host.epoch.load(Ordering::SeqCst), epoch);
            let resident = host.loaded.read().await;
            let resident = resident.as_ref().unwrap();
            assert_eq!(resident.worker.id(), worker);
            assert_eq!(resident.config.model_id.as_deref(), Some("fixture:latest"));
        }
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn metadata_and_stop_routes_require_authentication() {
        let mut host = fake_host().await;
        host.base.api_key = Some("secret".into());
        let response = dispatch(State(host.clone()), request("/api/stop", json!({}))).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(host.loaded.read().await.is_some());
        host.shutdown().await.unwrap();
    }

    fn resident_fixture() -> (tempfile::TempDir, ServeConfig) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("nested")).unwrap();
        for name in [
            "weights.gguf",
            "other.gguf",
            "projector.gguf",
            "other-projector.gguf",
        ] {
            // Empty path fixtures only; no native loader reads these files.
            std::fs::write(dir.path().join(name), []).unwrap();
        }
        let config = resident_config(ServeConfig {
            model: dir.path().join("weights.gguf"),
            model_id: Some("fixture:latest".into()),
            ..Default::default()
        })
        .unwrap();
        (dir, config)
    }

    async fn fixture_lease(
        host: &Host,
        selector: &str,
        config: ServeConfig,
        builds: Arc<AtomicU64>,
    ) -> Result<Lease, ApiError> {
        let executor = std::thread::current().id();
        let expected = selector.to_owned();
        let resolutions = Arc::new(AtomicU64::new(0));
        let resolved = resolutions.clone();
        let result = host
            .lease_with(
                Some(selector.into()),
                move |_, model| {
                    resolved.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(model, expected);
                    assert_ne!(std::thread::current().id(), executor);
                    Ok(config)
                },
                move |config| async move {
                    builds.fetch_add(1, Ordering::SeqCst);
                    fake_app(config).await
                },
            )
            .await;
        assert_eq!(resolutions.load(Ordering::SeqCst), 1);
        result
    }

    async fn fixture_generation(lease: &Lease) -> Value {
        let response = lease
            .as_ref()
            .unwrap()
            .app
            .clone()
            .oneshot(request(
                "/v1/completions",
                json!({"prompt":"fixture", "max_tokens":1}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn resident_identity_uses_canonical_files_and_ignores_only_display_id() {
        let (dir, config) = resident_fixture();
        let mut equivalent = config.clone();
        equivalent.model = dir.path().join("nested/../weights.gguf");
        equivalent.model_id = Some("lma:fixture:latest".into());
        assert!(same_resident(
            &config,
            &resident_config(equivalent).unwrap()
        ));

        let differences: &[(&str, fn(&mut ServeConfig))] = &[
            ("weights/source", |c| {
                c.model = c.model.with_file_name("other.gguf")
            }),
            ("store", |c| c.ollama_dir = Some("other-store".into())),
            ("projector", |c| {
                c.projector = Some(c.model.with_file_name("projector.gguf"))
            }),
            ("chat_template", |c| c.chat_template = "plain".into()),
            ("definition", |c| {
                c.definition = Some(modelfile::parse("FROM fixture\nSYSTEM different").unwrap())
            }),
            ("prompt_cache", |c| c.prompt_cache = !c.prompt_cache),
            ("kv_cache", |c| c.kv_cache = "q8_0".into()),
            ("memory_fallback", |c| c.memory_fallback = "mmap".into()),
            ("expert_cache_mib", |c| c.expert_cache_mib = 1024),
            ("host", |c| c.host = "localhost".into()),
            ("port", |c| c.port += 1),
            ("api", |c| c.api = "openai".into()),
            ("client", |c| c.client = "changed".into()),
            ("gpu_layers", |c| {
                c.gpu_layers = c.gpu_layers.wrapping_add(1)
            }),
            ("context", |c| c.context += 1),
            ("threads", |c| c.threads += 1),
            ("batch", |c| c.batch += 1),
            ("micro_batch", |c| c.micro_batch += 1),
            ("flash", |c| c.flash = "on".into()),
            ("api_key", |c| c.api_key = Some("secret".into())),
        ];
        for (field, change) in differences {
            let mut other = config.clone();
            change(&mut other);
            assert!(
                !same_resident(&config, &resident_config(other).unwrap()),
                "{field}"
            );
        }
        let mut authenticated = config.clone();
        authenticated.api_key = Some("first-secret".into());
        let mut other = authenticated.clone();
        other.api_key = Some("second-secret".into());
        assert!(!same_resident(&authenticated, &other));
        let mut bookkeeping = config.clone();
        bookkeeping.explicit_load = vec!["num_ctx".into(), "template".into()];
        assert!(same_resident(&config, &bookkeeping));
    }

    #[test]
    fn resident_identity_normalizes_default_store_and_projector_paths() {
        let (dir, mut config) = resident_fixture();
        std::fs::create_dir(dir.path().join("blobs")).unwrap();
        config.model = dir.path().join("blobs/weights.gguf");
        std::fs::write(&config.model, []).unwrap();
        config.projector = Some(dir.path().join("projector.gguf"));
        let implicit = resident_config(config.clone()).unwrap();
        config.ollama_dir = Some(dir.path().join("nested/.."));
        config.projector = Some(dir.path().join("nested/../projector.gguf"));
        assert!(same_resident(&implicit, &resident_config(config).unwrap()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn resident_public_name_keeps_snapshot_but_unfamiliar_alias_resolves_retargeted_symlink()
    {
        let (dir, config, loaded) = tokio::task::spawn_blocking(|| {
            let (dir, mut config) = resident_fixture();
            let link = dir.path().join("friendly.gguf");
            std::os::unix::fs::symlink(&config.model, &link).unwrap();
            config.model = link;
            config.model_id = None;
            let loaded = resident_config(config.clone()).unwrap();
            (dir, config, loaded)
        })
        .await
        .unwrap();
        assert_eq!(loaded.model_id.as_deref(), Some("friendly.gguf"));
        let host = fake_host_with_config(loaded).await;
        let initial = host.lease(None).await.unwrap();
        let worker = initial.as_ref().unwrap().worker.id();
        assert_eq!(
            fixture_generation(&initial).await["choices"][0]["text"],
            "hello"
        );
        drop(initial);
        let link = config.model.clone();
        let other = dir.path().join("other.gguf");
        tokio::task::spawn_blocking(move || {
            std::fs::remove_file(&link).unwrap();
            std::os::unix::fs::symlink(other, link).unwrap();
        })
        .await
        .unwrap();
        let snapshot = host
            .lease_with(
                Some("friendly.gguf".into()),
                |_, _| panic!("exact public name must not resolve a retargeted symlink"),
                |_| async { panic!("exact public name must not replace the worker") },
            )
            .await
            .unwrap();
        assert_eq!(snapshot.as_ref().unwrap().worker.id(), worker);
        assert_eq!(
            fixture_generation(&snapshot).await["choices"][0]["text"],
            "cached"
        );
        drop(snapshot);
        let builds = Arc::new(AtomicU64::new(0));
        let alias = fixture_lease(&host, "lma:retargeted:latest", config, builds.clone())
            .await
            .unwrap();
        assert_ne!(alias.as_ref().unwrap().worker.id(), worker);
        assert_eq!(
            fixture_generation(&alias).await["choices"][0]["text"],
            "hello"
        );
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        drop(alias);
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn resident_public_name_and_startup_filename_skip_resolution_and_rebuild() {
        let (_dir, config) = tokio::task::spawn_blocking(resident_fixture).await.unwrap();
        for label in [Some("fixture:latest".to_owned()), None] {
            let mut config = config.clone();
            config.model_id = label;
            let config = tokio::task::spawn_blocking(move || resident_config(config))
                .await
                .unwrap()
                .unwrap();
            let public_name = config.model_id.clone().unwrap();
            let host = fake_host_with_config(config).await;
            let initial = host.lease(None).await.unwrap();
            let worker = initial.as_ref().unwrap().worker.id();
            assert_eq!(
                fixture_generation(&initial).await["choices"][0]["text"],
                "hello"
            );
            // A snapshot lookup needs neither catalog work nor the operations lock.
            let operation = host.operations.lock().await;
            let lease = host
                .lease_with(
                    Some(public_name),
                    |_, _| panic!("resident public name must not invoke the resolver"),
                    |_| async { panic!("resident public name must not invoke the factory") },
                )
                .await
                .unwrap();
            assert_eq!(lease.as_ref().unwrap().worker.id(), worker);
            let next = fixture_generation(&lease).await;
            assert_eq!(next["choices"][0]["text"], "cached");
            assert!(next["id"].as_str().unwrap().ends_with("-1"));
            drop(operation);
            drop(lease);
            drop(initial);
            host.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn equivalent_selectors_retain_router_worker_and_cache_with_active_lease() {
        let (dir, config) = tokio::task::spawn_blocking(resident_fixture).await.unwrap();
        let host = fake_host_with_config(config.clone()).await;
        let builds = Arc::new(AtomicU64::new(0));
        let initial = host.lease(None).await.unwrap();
        let worker = initial.as_ref().unwrap().worker.id();
        let first = fixture_generation(&initial).await;
        assert_eq!(first["choices"][0]["text"], "hello");
        assert!(first["id"].as_str().unwrap().ends_with("-0"));
        for (index, selector) in ["fixture", "lma:fixture:latest", "ollama:fixture:latest"]
            .iter()
            .enumerate()
        {
            let mut resolved = config.clone();
            resolved.model = dir.path().join("nested/../weights.gguf");
            resolved.model_id = Some((*selector).into());
            let lease = fixture_lease(&host, selector, resolved, builds.clone())
                .await
                .unwrap();
            assert_eq!(lease.as_ref().unwrap().worker.id(), worker);
            assert_eq!(lease.as_ref().unwrap().config.model_id, config.model_id);
            let next = fixture_generation(&lease).await;
            assert_eq!(next["choices"][0]["text"], "cached");
            assert!(
                next["id"]
                    .as_str()
                    .unwrap()
                    .ends_with(&format!("-{}", index + 1))
            );
        }
        assert_eq!(builds.load(Ordering::SeqCst), 0);
        drop(initial);
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn same_display_name_with_different_source_projector_or_config_switches() {
        let (dir, config) = tokio::task::spawn_blocking(resident_fixture).await.unwrap();
        let host = fake_host_with_config(config.clone()).await;
        let builds = Arc::new(AtomicU64::new(0));
        let mut source = config.clone();
        source.model = dir.path().join("other.gguf");
        let mut store = source.clone();
        store.ollama_dir = Some(dir.path().join("other-store"));
        let mut projector = store.clone();
        projector.projector = Some(dir.path().join("projector.gguf"));
        let mut other_projector = projector.clone();
        other_projector.projector = Some(dir.path().join("other-projector.gguf"));
        let mut settings = other_projector.clone();
        settings.threads += 1;
        for (index, target) in [source, store, projector, other_projector, settings]
            .into_iter()
            .enumerate()
        {
            let previous = host.lease(None).await.unwrap();
            fixture_generation(&previous).await;
            let worker = previous.as_ref().unwrap().worker.id();
            let exited = previous.as_ref().unwrap().worker.abort_handle();
            drop(previous);
            let lease = fixture_lease(&host, "lma:fixture:latest", target.clone(), builds.clone())
                .await
                .unwrap();
            assert_ne!(lease.as_ref().unwrap().worker.id(), worker);
            assert!(exited.is_finished());
            assert!(same_resident(&lease.as_ref().unwrap().config, &target));
            assert_eq!(
                fixture_generation(&lease).await["choices"][0]["text"],
                "hello"
            );
            assert_eq!(builds.load(Ordering::SeqCst), index as u64 + 1);
        }
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn different_identity_or_resolution_failure_never_disrupts_active_resident() {
        let (_dir, config) = tokio::task::spawn_blocking(resident_fixture).await.unwrap();
        let host = fake_host_with_config(config.clone()).await;
        let builds = Arc::new(AtomicU64::new(0));
        let active = host.lease(None).await.unwrap();
        let mut different = config.clone();
        different.threads += 1;
        let error = fixture_lease(&host, "lma:fixture:latest", different, builds.clone())
            .await
            .err()
            .expect("different identity must stay busy");
        assert_eq!(error.into_response().status(), StatusCode::CONFLICT);
        let mut missing = config;
        missing.model.set_file_name("missing.gguf");
        let error = fixture_lease(&host, "lma:fixture:latest", missing, builds.clone())
            .await
            .err()
            .expect("missing identity must fail before teardown");
        assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
        assert_eq!(builds.load(Ordering::SeqCst), 0);
        assert_eq!(
            fixture_generation(&active).await["choices"][0]["text"],
            "hello"
        );
        drop(active);
        host.shutdown().await.unwrap();
    }

    #[test]
    fn selection_resets_only_startup_definition_defaults_and_preserves_cli_overrides() {
        let defaults = ServeConfig::default();
        let mut base = ServeConfig {
            definition: Some(modelfile::parse("FROM fixture\nSYSTEM startup\nTEMPLATE plain\nPARAMETER num_ctx 512\nPARAMETER num_gpu 0\nPARAMETER num_thread 2\nPARAMETER num_batch 32\nPARAMETER u_batch 16").unwrap()),
            chat_template: "plain".into(), context: 512, gpu_layers: 0,
            threads: 2, batch: 32, micro_batch: 16,
            api_key: Some("host-secret".into()),
            ..Default::default()
        };
        let reset = selection_base(&base);
        assert!(reset.definition.is_none());
        assert_eq!(reset.chat_template, defaults.chat_template);
        assert_eq!(reset.context, defaults.context);
        assert_eq!(reset.gpu_layers, defaults.gpu_layers);
        assert_eq!(reset.threads, defaults.threads);
        assert_eq!(reset.batch, defaults.batch);
        assert_eq!(reset.micro_batch, defaults.micro_batch);
        assert_eq!(reset.api_key, base.api_key);
        assert!(base.definition.is_some(), "startup snapshot stays intact");
        base.explicit_load = [
            "template",
            "num_ctx",
            "num_gpu",
            "num_thread",
            "num_batch",
            "u_batch",
        ]
        .map(str::to_owned)
        .to_vec();
        let explicit = selection_base(&base);
        assert!(explicit.definition.is_none());
        assert_eq!(explicit.chat_template, base.chat_template);
        assert_eq!(explicit.context, base.context);
        assert_eq!(explicit.gpu_layers, base.gpu_layers);
        assert_eq!(explicit.threads, base.threads);
        assert_eq!(explicit.batch, base.batch);
        assert_eq!(explicit.micro_batch, base.micro_batch);
        assert_eq!(explicit.explicit_load, base.explicit_load);
        base.explicit_load = vec!["num_batch".into()];
        assert_eq!(
            selection_base(&base).micro_batch,
            defaults.micro_batch.min(base.batch)
        );
        base.definition = None;
        let configured = selection_base(&base);
        assert_eq!(configured.context, base.context);
        assert_eq!(configured.threads, base.threads);
    }

    #[tokio::test]
    async fn different_definition_on_same_weights_switches_profiles() {
        let (_dir, mut config) = tokio::task::spawn_blocking(resident_fixture).await.unwrap();
        config.definition =
            Some(modelfile::parse("FROM fixture\nSYSTEM first\nPARAMETER temperature 0").unwrap());
        let host = fake_host_with_config(config.clone()).await;
        let previous = host.lease(None).await.unwrap();
        let worker = previous.as_ref().unwrap().worker.id();
        fixture_generation(&previous).await;
        drop(previous);
        config.definition.as_mut().unwrap().system = Some("second".into());
        let builds = Arc::new(AtomicU64::new(0));
        let lease = fixture_lease(&host, "lma:second:latest", config, builds.clone())
            .await
            .unwrap();
        assert_ne!(lease.as_ref().unwrap().worker.id(), worker);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(
            lease
                .as_ref()
                .unwrap()
                .config
                .definition
                .as_ref()
                .unwrap()
                .system
                .as_deref(),
            Some("second")
        );
        drop(lease);
        host.shutdown().await.unwrap();
    }

    fn profile_fixture() -> (tempfile::TempDir, PathBuf, ServeConfig) {
        let (dir, mut config) = resident_fixture();
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        std::fs::write(&config.model, header).unwrap();
        config.definition = Some(modelfile::parse("FROM upstream:latest\nSYSTEM inherited\nTEMPLATE plain\nLICENSE public license\nPARAMETER temperature 0.5\nPARAMETER top_p 0.8\nPARAMETER seed 42\nPARAMETER num_ctx 512\nMESSAGE user history").unwrap());
        let root = dir.path().join("registry");
        (dir, root, config)
    }

    fn constrained_profile_host() -> (tempfile::TempDir, PathBuf, Host) {
        let (dir, root, mut config) = profile_fixture();
        config.definition = Some(
            modelfile::parse(
                "FROM upstream:latest\nSYSTEM inherited\nPARAMETER num_ctx 4096\nPARAMETER num_predict 1024",
            )
            .unwrap(),
        );
        config.context = 512;
        config.explicit_load = vec!["num_ctx".into()];
        let host = Host {
            base: config,
            loaded: Arc::new(RwLock::new(None)),
            operations: Arc::new(Mutex::new(())),
            epoch: Arc::new(AtomicU64::new(0)),
        };
        (dir, root, host)
    }

    async fn profile_admin(
        host: Host,
        path: &str,
        body: Value,
        root: PathBuf,
        config: ServeConfig,
    ) -> Response {
        let executor = std::thread::current().id();
        admin_with(
            host,
            path,
            body,
            move || {
                assert_ne!(std::thread::current().id(), executor);
                Ok(root)
            },
            move |_, selector| {
                assert_ne!(std::thread::current().id(), executor);
                assert_eq!(selector, "fixture:latest");
                Ok(config)
            },
        )
        .await
        .unwrap()
    }

    async fn admin_json(response: Response) -> Value {
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn api_create_parses_text_and_structured_definitions_and_inherits_source_profile() {
        let (_dir, root, config) = tokio::task::spawn_blocking(profile_fixture).await.unwrap();
        let host = fake_host().await;
        let cases = [
            (
                "text",
                json!({"model":"text","modelfile":"FROM fixture:latest\nSYSTEM child\nPARAMETER temperature 0\nMESSAGE assistant reply", "stream":false}),
            ),
            (
                "structured",
                json!({"model":"structured","from":"fixture:latest","system":"child","parameters":{"temperature":0},"messages":[{"role":"assistant","content":"reply"}],"stream":true}),
            ),
        ];
        for (label, body) in cases {
            let child = match ProfileRequest::parse("/api/create", &body)
                .unwrap()
                .unwrap()
            {
                ProfileRequest::Create { definition, .. } => definition,
                _ => panic!("expected a definition"),
            };
            let expected =
                super::super::create::inherit_definition(child, config.definition.clone()).unwrap();
            let value = admin_json(
                profile_admin(
                    host.clone(),
                    "/api/create",
                    body,
                    root.clone(),
                    config.clone(),
                )
                .await,
            )
            .await;
            assert_eq!(value["model"], format!("{label}:latest"));
            let read_root = root.clone();
            let stored = tokio::task::spawn_blocking(move || {
                registry::definition(&read_root, label).unwrap().unwrap()
            })
            .await
            .unwrap();
            assert_eq!(stored, expected);
            assert_eq!(stored.system.as_deref(), Some("child"));
            assert_eq!(stored.parameters.temperature, Some(0.0));
            assert_eq!(stored.parameters.top_p, Some(0.8));
            assert_eq!(stored.license.as_deref(), Some("public license"));
        }
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn api_create_inherits_metadata_before_validating_child_under_host_overrides() {
        let (_dir, root, host) = tokio::task::spawn_blocking(constrained_profile_host)
            .await
            .unwrap();
        for (label, body) in [
            (
                "text",
                json!({"model":"text","modelfile":"FROM fixture:latest\nPARAMETER num_predict 64","stream":false}),
            ),
            (
                "structured",
                json!({"model":"structured","from":"fixture:latest","parameters":{"num_predict":64},"stream":false}),
            ),
        ] {
            let executor = std::thread::current().id();
            let publish_root = root.clone();
            let response = admin_with(
                host.clone(),
                "/api/create",
                body,
                move || {
                    assert_ne!(std::thread::current().id(), executor);
                    Ok(publish_root)
                },
                move |host, selector| {
                    assert_ne!(std::thread::current().id(), executor);
                    host.target_metadata(&selector)
                },
            )
            .await
            .unwrap();
            assert_eq!(
                admin_json(response).await["model"],
                format!("{label}:latest")
            );
            let read_root = root.clone();
            let mut child = host.base.clone();
            let child = tokio::task::spawn_blocking(move || {
                child.definition = registry::definition(&read_root, label).unwrap();
                let definition = child.definition.as_ref().unwrap();
                assert_eq!(definition.system.as_deref(), Some("inherited"));
                assert_eq!(definition.parameters.num_ctx, Some(4096));
                assert_eq!(definition.parameters.num_predict, Some(64));
                super::super::create::apply_load_defaults(&mut child).unwrap();
                child
            })
            .await
            .unwrap();
            assert_eq!(child.context, 512, "explicit host context still wins");
        }
        assert!(host.loaded.read().await.is_none());
    }

    #[tokio::test]
    async fn api_copy_and_show_metadata_ignore_host_load_constraints_but_load_still_rejects() {
        let (_dir, root, host) = tokio::task::spawn_blocking(constrained_profile_host)
            .await
            .unwrap();
        let expected = host.base.definition.clone();
        for (path, label, body) in [
            (
                "/api/copy",
                "copied",
                json!({"source":"fixture:latest","destination":"copied","stream":false}),
            ),
            (
                "/api/create",
                "plain",
                json!({"model":"plain","from":"fixture:latest","stream":false}),
            ),
        ] {
            let publish_root = root.clone();
            let response = admin_with(
                host.clone(),
                path,
                body,
                move || Ok(publish_root),
                |host, selector| host.target_metadata(&selector),
            )
            .await
            .unwrap();
            assert_eq!(
                admin_json(response).await["model"],
                format!("{label}:latest")
            );
            let read_root = root.clone();
            let stored = tokio::task::spawn_blocking(move || {
                registry::definition(&read_root, label).unwrap()
            })
            .await
            .unwrap();
            assert_eq!(stored, expected);
        }
        let shown = admin_json(
            handle(
                host.clone(),
                request("/api/show", json!({"model":"fixture:latest"})),
            )
            .await
            .unwrap(),
        )
        .await;
        assert_eq!(shown["definition"], serde_json::to_value(expected).unwrap());
        let resolver = host.clone();
        let error = tokio::task::spawn_blocking(move || resolver.target("fixture:latest"))
            .await
            .unwrap()
            .unwrap_err();
        assert!(error.contains("num_predict exceeds the effective context"));
        let error = host
            .lease_with(
                Some("fixture:latest".into()),
                |host, selector| host.target(&selector),
                |_| async { panic!("invalid load config must not reach the model builder") },
            )
            .await
            .err()
            .expect("load selection must still reject the parent under host overrides");
        assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
        assert!(host.loaded.read().await.is_none());
    }

    #[tokio::test]
    async fn api_copy_and_plain_from_create_preserve_definition_and_do_not_overwrite() {
        let (_dir, root, config) = tokio::task::spawn_blocking(profile_fixture).await.unwrap();
        let host = fake_host().await;
        for (path, label, body) in [
            (
                "/api/copy",
                "copied",
                json!({"source":"fixture:latest","destination":"copied","stream":false}),
            ),
            (
                "/api/create",
                "plain",
                json!({"model":"plain","from":"fixture:latest","stream":false}),
            ),
        ] {
            admin_json(
                profile_admin(
                    host.clone(),
                    path,
                    body.clone(),
                    root.clone(),
                    config.clone(),
                )
                .await,
            )
            .await;
            let read_root = root.clone();
            let stored = tokio::task::spawn_blocking(move || {
                registry::definition(&read_root, label).unwrap()
            })
            .await
            .unwrap();
            assert_eq!(stored, config.definition);
            let duplicate = admin_with(
                host.clone(),
                path,
                body,
                {
                    let root = root.clone();
                    move || Ok(root)
                },
                {
                    let config = config.clone();
                    move |_, _| Ok(config)
                },
            )
            .await
            .err()
            .expect("duplicate alias must fail");
            assert_eq!(duplicate.into_response().status(), StatusCode::BAD_REQUEST);
        }
        let read_root = root.clone();
        let entries = tokio::task::spawn_blocking(move || registry::read(&read_root).unwrap())
            .await
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.values().all(|entry| entry.path == config.model));
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn api_create_rejects_mixed_invalid_and_path_inputs_before_root_or_resolver_io() {
        let host = fake_host().await;
        let mut cases = vec![
            json!({"model":"bad","modelfile":"FROM fixture","from":"fixture","stream":false}),
            json!({"model":"bad","modelfile":"FROM fixture","system":null,"stream":false}),
            json!({"model":"bad","modelfile":"FROM fixture\nADAPTER ./adapter","stream":false}),
            json!({"model":"bad","from":"fixture","template":"{{ .Prompt }}","stream":false}),
            json!({"model":"bad","from":"fixture","parameters":{"stop":"x"},"stream":false}),
            json!({"model":"bad","from":"fixture","parameters":{"num_ctx":1},"stream":false}),
            json!({"model":"bad","from":"fixture","messages":[{"role":"tool","content":"x"}],"stream":false}),
            json!({"model":"bad","from":"fixture","system":[],"stream":false}),
        ];
        for from in [
            "/tmp/private.gguf",
            "./private.gguf",
            "../private.gguf",
            "C:\\private.gguf",
            "https://example.com/model.gguf",
            "file:///tmp/model.gguf",
            "team/../private",
        ] {
            cases.push(json!({"model":"bad","from":from,"stream":false}));
            cases
                .push(json!({"model":"bad","modelfile":format!("FROM \"{from}\""),"stream":false}));
        }
        for body in cases {
            let result = admin_with(
                host.clone(),
                "/api/create",
                body,
                || panic!("invalid definition must not locate a registry root"),
                |_, _| panic!("invalid definition must not resolve or open files"),
            )
            .await
            .err()
            .expect("invalid request must fail");
            assert_eq!(result.into_response().status(), StatusCode::BAD_REQUEST);
        }
        host.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn unknown_http_source_uses_only_injected_allowlist_and_never_publishes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("registry");
        let injected = root.clone();
        let host = fake_host().await;
        let result = admin_with(
            host.clone(),
            "/api/create",
            json!({"model":"bad","from":"not-installed:latest","system":"child","stream":false}),
            move || Ok(injected),
            |_, selector| {
                assert_eq!(selector, "not-installed:latest");
                Err("installed selector not found".into())
            },
        )
        .await
        .err()
        .expect("uninstalled source must fail");
        assert_eq!(result.into_response().status(), StatusCode::BAD_REQUEST);
        assert!(!root.exists());
        host.shutdown().await.unwrap();
    }

    #[test]
    fn show_exposes_only_public_definition_metadata_not_host_credentials() {
        let (_dir, _root, mut config) = profile_fixture();
        config.api_key = Some("do-not-expose-host-key".into());
        let expected = serde_json::to_value(&config.definition).unwrap();
        let shown = model_show(config).unwrap();
        assert_eq!(shown["definition"], expected);
        assert_eq!(shown["definition"]["system"], "inherited");
        assert_eq!(shown["definition"]["license"], "public license");
        assert!(!shown.to_string().contains("do-not-expose-host-key"));
        assert!(shown.get("api_key").is_none());
        for selector in [
            "fixture:latest",
            "team/model:tag",
            "lma:fixture:latest",
            "lma:/installed/model.gguf",
            "ollama:5:/root:model:latest",
        ] {
            installed_selector(selector).unwrap();
        }
    }

    #[tokio::test]
    async fn definition_admin_and_show_routes_still_require_authentication() {
        let mut host = fake_host().await;
        host.base.api_key = Some("secret".into());
        for (path, body) in [
            (
                "/api/create",
                json!({"model":"child","from":"fixture:latest"}),
            ),
            (
                "/api/copy",
                json!({"source":"fixture:latest","destination":"child"}),
            ),
            ("/api/show", json!({"model":"fixture:latest"})),
        ] {
            let response = dispatch(State(host.clone()), request(path, body)).await;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }
        host.shutdown().await.unwrap();
    }

    #[test]
    fn lifecycle_durations_are_bounded_and_explicit() {
        assert_eq!(keep_alive(Some(&json!(-1))).unwrap(), None);
        assert_eq!(
            keep_alive(Some(&json!("5m"))).unwrap(),
            Some(Duration::from_secs(300))
        );
        assert_eq!(keep_alive(Some(&json!(0))).unwrap(), Some(Duration::ZERO));
        for value in [json!(-2), json!("forever"), json!(true), json!(1e30)] {
            assert!(keep_alive(Some(&value)).is_err());
        }
    }
}
