//! In-process access to the same API used by the CLI HTTP server.
//!
//! A host may share a session through `Arc<Session>`. Cancel requests by dropping
//! their futures and any returned streaming bodies, then join those tasks and use
//! `Arc::try_unwrap` to recover the
//! session for `shutdown`. Await shutdown before loading a replacement model:
//! dropping a session alone releases its sender but does not await backend teardown.

use std::{io::Read, path::Path};

use axum::{
    Router,
    body::{Body, Bytes},
    http::{Method, Request, header},
};
use http_body_util::{BodyExt, Limited};
use serde::Serialize;
use serde_json::Value;
use tokio::task::JoinHandle;
use tower::ServiceExt;

use crate::{
    config::ServeConfig,
    runtime::{self, server},
};

/// Resolve a local model and load its inference worker without opening a socket.
///
/// API keys are ignored for this trusted in-process bridge. The host must control
/// who can invoke it. Do not overlap model loads; await the old session's shutdown.
pub async fn start(config: ServeConfig) -> Result<Session, String> {
    prepare(config).await?.start().await
}

/// Resolve and validate model references without allocating an inference backend.
/// Hosts can do this before unloading a working session for a replacement.
pub async fn prepare(mut config: ServeConfig) -> Result<PreparedModel, String> {
    config.api_key = None;
    // Internal consumers always need the chat/completion routes, independently
    // of any public hosting profile selected by a launcher or GUI.
    config.api = "all".into();
    config.client = "default".into();
    let config = tokio::task::spawn_blocking(move || {
        runtime::resolve_model(&mut config)?;
        preflight_gguf(&config.model, "model")?;
        #[cfg(not(feature = "vision"))]
        if config.projector.is_some() {
            return Err("a projector requires a build with the vision feature enabled".into());
        }
        #[cfg(feature = "vision")]
        if let Some(projector) = &config.projector {
            preflight_gguf(projector, "projector")?;
        }
        Ok::<_, String>(config)
    })
    .await
    .map_err(|error| format!("model resolution failed: {error}"))??;
    Ok(PreparedModel { config })
}

fn preflight_gguf(path: &Path, kind: &str) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("{kind} is not a file: {}", path.display()));
    }
    let mut magic = [0; 4];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .map_err(|error| format!("cannot read {kind} header '{}': {error}", path.display()))?;
    if &magic != b"GGUF" {
        return Err(format!("{kind} is not GGUF: {}", path.display()));
    }
    Ok(())
}

/// A resolved configuration, not loaded weights. Native initialization can still
/// fail. The model and projector must remain unchanged between prepare and start.
pub struct PreparedModel {
    config: ServeConfig,
}

impl PreparedModel {
    /// Await any previous session's shutdown before calling this method.
    pub async fn start(self) -> Result<Session, String> {
        let (router, worker) = server::build(self.config).await?;
        Ok(Session::new(router, worker))
    }
}

/// Owns one model worker. This is intentionally not `Clone`.
///
/// Dropping the session naturally closes the worker's channel once all borrowed
/// request futures are gone. Use `shutdown` when teardown must finish before reload.
pub struct Session {
    router: Router,
    worker: JoinHandle<()>,
}

#[derive(Debug, Serialize)]
pub struct DesktopResponse {
    pub status: u16,
    pub body: Value,
}

/// An unbuffered in-process response. Dropping it drops the inference receiver;
/// native worker admission remains held until generation actually stops.
pub struct DesktopStream {
    pub status: u16,
    pub content_type: Option<String>,
    body: Body,
    remaining: usize,
}

impl DesktopStream {
    /// Read the next body data frame without collecting the response first.
    pub async fn next_chunk(&mut self) -> Result<Option<Bytes>, String> {
        while let Some(frame) = self.body.frame().await {
            let frame = frame.map_err(|error| format!("cannot read desktop stream: {error}"))?;
            if let Ok(bytes) = frame.into_data() {
                self.remaining = self
                    .remaining
                    .checked_sub(bytes.len())
                    .ok_or("desktop stream exceeds the response byte limit")?;
                return Ok(Some(bytes));
            }
        }
        Ok(None)
    }
}

impl Session {
    pub(crate) fn new(router: Router, worker: JoinHandle<()>) -> Self {
        Self { router, worker }
    }

    /// Dispatch directly through the HTTP router, with no socket or subprocess.
    ///
    /// Only GET /health, GET /v1/models, and POST to the three generation endpoints
    /// are accepted. Limits are checked before constructing headers or parsing JSON.
    /// The host should also bound IPC input before allocating the supplied `Vec`.
    /// Dropping this future cancels its inference reply; worker admission remains
    /// held until the engine has actually stopped processing the request.
    pub async fn request(
        &self,
        method: &str,
        path: &str,
        body: Vec<u8>,
    ) -> Result<DesktopResponse, String> {
        let response = self.dispatch(method, path, body, false).await?;
        let status = response.status().as_u16();
        let bytes = Limited::new(response.into_body(), server::IMAGE_BODY_LIMIT)
            .collect()
            .await
            .map_err(|error| format!("cannot read desktop response: {error}"))?
            .to_bytes();
        let body = serde_json::from_slice(&bytes)
            .map_err(|error| format!("desktop response is not JSON: {error}"))?;
        Ok(DesktopResponse { status, body })
    }

    /// Dispatch a streaming completion without opening a socket or spawning a task.
    /// The caller supplies `stream:true`, handles non-2xx responses, and owns the
    /// returned body until it finishes or is dropped to cancel the request.
    pub async fn stream(&self, path: &str, body: Vec<u8>) -> Result<DesktopStream, String> {
        self.stream_with_budget(path, body, false).await
    }

    /// Generate until EOS, cancellation, or the hard native context limit, without
    /// a separate reply-token cap. No context shifting, truncation, or model unload
    /// is performed. The body still requires valid API options (e.g. max_tokens:1)
    /// and stream:true; this mode is carried only by a private request extension.
    pub async fn stream_until_context(
        &self,
        path: &str,
        body: Vec<u8>,
    ) -> Result<DesktopStream, String> {
        self.stream_with_budget(path, body, true).await
    }

    async fn stream_with_budget(
        &self,
        path: &str,
        body: Vec<u8>,
        until_context: bool,
    ) -> Result<DesktopStream, String> {
        if !matches!(path, "/v1/chat/completions" | "/v1/completions") {
            return Err("desktop streaming API path is not allowed".into());
        }
        let response = self.dispatch("POST", path, body, until_context).await?;
        Ok(DesktopStream {
            status: response.status().as_u16(),
            content_type: response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
            body: response.into_body(),
            remaining: server::IMAGE_BODY_LIMIT,
        })
    }

    async fn dispatch(
        &self,
        method: &str,
        path: &str,
        body: Vec<u8>,
        until_context: bool,
    ) -> Result<axum::response::Response, String> {
        let (expected_method, limit) = match path {
            "/health" | "/v1/models" => (Method::GET, server::BODY_LIMIT),
            "/v1/completions" => (Method::POST, server::BODY_LIMIT),
            "/v1/chat/completions" | "/api/chat" => (Method::POST, server::IMAGE_BODY_LIMIT),
            _ => return Err("desktop API path is not allowed".into()),
        };
        if method != expected_method.as_str() {
            return Err("desktop API method is not allowed for this path".into());
        }
        if body.len() > limit {
            return Err(format!("request body exceeds the {limit}-byte limit"));
        }
        let mut request = Request::builder()
            .method(expected_method)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .map_err(|error| format!("cannot construct desktop request: {error}"))?;
        if until_context {
            request.extensions_mut().insert(server::UntilContext);
        }
        // No task is spawned: dropping this future drops the router's reply receiver.
        self.router
            .clone()
            .oneshot(request)
            .await
            .map_err(|error| format!("desktop request failed: {error}"))
    }

    /// Release the router and await complete worker/model/backend teardown.
    ///
    /// Call only after dropping all stream bodies and cancelling/joining all
    /// request tasks, recovering sole
    /// ownership with `Arc::try_unwrap` if necessary. Await this to completion before
    /// `start`ing another model; cancelling shutdown forfeits the teardown barrier.
    pub async fn shutdown(self) -> Result<(), String> {
        let Self { router, worker } = self;
        drop(router);
        worker
            .await
            .map_err(|error| format!("inference worker failed: {error}"))
    }
}
