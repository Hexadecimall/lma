use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use lma::{
    config::{self, ServeConfig},
    desktop::{self, DesktopResponse, Session},
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as AsyncMutex, oneshot};

use crate::update::SourceUpdater;

#[derive(Default)]
pub struct AppState {
    inner: Mutex<Inner>,
    lifecycle: AsyncMutex<()>,
    pub(crate) selection: AsyncMutex<()>,
    pub(crate) discovery: AsyncMutex<()>,
    pub(crate) store_download: AsyncMutex<()>,
    pub(crate) store_search: AsyncMutex<()>,
    pub updater: SourceUpdater,
}

#[derive(Default)]
struct Inner {
    session: Option<Arc<Session>>,
    model: Option<String>,
    backend: String,
    loading: bool,
    update_requested: bool,
    preference_warning: Option<String>,
    requests: HashMap<String, Option<oneshot::Sender<()>>>,
    early_cancellations: HashMap<String, std::time::Instant>,
}

#[derive(Serialize)]
pub struct Status {
    loaded: bool,
    model: Option<String>,
    backend: String,
    platform: &'static str,
    updating: bool,
    loading: bool,
    #[serde(rename = "preferenceWarning")]
    preference_warning: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LoadOptions {
    pub(crate) model: String,
    pub(crate) ollama: bool,
    pub(crate) ollama_dir: Option<PathBuf>,
    pub(crate) projector: Option<PathBuf>,
    pub(crate) gpu: String,
    pub(crate) context: u32,
}

impl LoadOptions {
    pub(crate) fn config(self) -> Result<ServeConfig, String> {
        if self.model.trim().is_empty() || self.model.len() > 4096 || self.model.contains('\0') {
            return Err(
                "Choose a local GGUF path, lma model alias, or installed Ollama model.".into(),
            );
        }
        if self.context < 2 {
            return Err("Context must be at least 2 tokens.".into());
        }
        let gpu_layers = match self.gpu.as_str() {
            "auto" => config::default_gpu_layers(),
            "all" => config::ALL_GPU_LAYERS,
            "cpu" => 0,
            value => value
                .parse::<u32>()
                .map_err(|_| "GPU must be auto, all, cpu, or a layer count.")?,
        };
        Ok(ServeConfig {
            model: if self.ollama {
                PathBuf::from(format!("ollama:{}", self.model.trim()))
            } else {
                PathBuf::from(self.model)
            },
            ollama_dir: self.ollama_dir,
            projector: self.projector,
            gpu_layers,
            context: self.context,
            // The frontend already applied app settings, including automatic load policy.
            explicit_load: vec!["num_ctx".into(), "num_gpu".into()],
            // Desktop failures must remain visible; CPU paging is never implicit.
            memory_fallback: "error".into(),
            ..ServeConfig::default()
        })
    }
}

impl AppState {
    pub fn status(&self) -> Status {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        Status {
            loaded: inner.session.is_some(),
            model: inner.model.clone(),
            backend: if inner.backend.is_empty() {
                "Not loaded".into()
            } else {
                inner.backend.clone()
            },
            platform: std::env::consts::OS,
            updating: inner.update_requested || self.updater.is_running(),
            loading: inner.loading,
            preference_warning: inner.preference_warning.clone(),
        }
    }

    pub fn set_preference_warning(&self, warning: Option<String>) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .preference_warning = warning;
    }

    pub fn is_updating(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .update_requested
            || self.updater.is_running()
    }

    pub async fn load(&self, options: LoadOptions) -> Result<Status, String> {
        // Resolve references before unloading a working model. Native allocation
        // still waits for the old session's complete teardown.
        let config = options.config()?;
        let gpu_layers = config.gpu_layers;
        let _operation = self
            .lifecycle
            .try_lock()
            .map_err(|_| "A model operation is already running.")?;
        let prepared = desktop::prepare(config).await?;
        self.begin_operation()?;
        let result: Result<(), String> = async {
            self.release_model().await?;
            let report = config::backend_report(gpu_layers);
            let backend = if gpu_layers == 0 {
                "CPU".into()
            } else {
                format!(
                    "GPU requested · {} layers · {}",
                    gpu_layers,
                    report.compiled_backends.join("/")
                )
            };
            let session = prepared.start().await?;
            let models = session.request("GET", "/v1/models", Vec::new()).await;
            let model = match models {
                Ok(response) => response.body["data"][0]["id"].as_str().map(str::to_owned),
                Err(error) => {
                    session.shutdown().await?;
                    return Err(error);
                }
            };
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.session = Some(Arc::new(session));
            inner.model = model;
            inner.backend = backend;
            Ok(())
        }
        .await;
        self.finish_operation();
        result?;
        Ok(self.status())
    }

    /// The blocking task owns every gate and the busy lease, even if the IPC
    /// future is dropped. Deletion never uses the teardown/cancellation path.
    pub async fn delete_model<T, F>(self: &Arc<Self>, delete: F) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, String> + Send + 'static,
    {
        let state = self.clone();
        tokio::task::spawn_blocking(move || {
            let _selection = state
                .selection
                .try_lock()
                .map_err(|_| "A model selection is already running.")?;
            let _operation = state
                .lifecycle
                .try_lock()
                .map_err(|_| "A model operation is already running.")?;
            let _download = state
                .store_download
                .try_lock()
                .map_err(|_| "A model download is already running.")?;
            let _discovery = state
                .discovery
                .try_lock()
                .map_err(|_| "Model discovery is already running.")?;
            {
                let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
                if inner.update_requested || state.updater.is_running() {
                    return Err("An update is running.".into());
                }
                if inner.loading {
                    return Err("A model operation is already running.".into());
                }
                if inner.session.is_some() || inner.model.is_some() {
                    return Err("Unload the loaded model before deleting any model.".into());
                }
                if !inner.requests.is_empty() {
                    return Err(
                        "Wait for active requests to finish before deleting a model.".into(),
                    );
                }
                inner.loading = true;
            }
            let _busy = DeletionLease(&state);
            delete()
        })
        .await
        .map_err(|e| format!("Model deletion task failed: {e}"))?
    }

    pub async fn unload(&self) -> Result<Status, String> {
        let _operation = self
            .lifecycle
            .try_lock()
            .map_err(|_| "A model operation is already running.")?;
        self.begin_operation()?;
        let result = self.release_model().await;
        self.finish_operation();
        result?;
        Ok(self.status())
    }

    pub async fn update(&self, config_dir: PathBuf) -> Result<String, String> {
        let _operation = self
            .lifecycle
            .try_lock()
            .map_err(|_| "A model operation is already running.")?;
        self.updater.preflight(&config_dir)?;
        self.begin_operation()?;
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .update_requested = true;
        let result = async {
            self.release_model().await?;
            self.updater.start(config_dir)
        }
        .await;
        self.finish_operation();
        result
    }

    fn begin_operation(&self) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if self.updater.is_running() {
            return Err("An update is running.".into());
        }
        inner.loading = true;
        for cancel in inner.requests.values_mut() {
            if let Some(cancel) = cancel.take() {
                let _ = cancel.send(());
            }
        }
        Ok(())
    }

    fn finish_operation(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.loading = false;
        inner.update_requested = false;
    }

    async fn release_model(&self) -> Result<(), String> {
        let session = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .session
            .take();
        if let Some(mut session) = session {
            // Cancelled IPC futures release their Arc before native shutdown.
            // Native inference itself is awaited by Session::shutdown below.
            let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
            loop {
                match Arc::try_unwrap(session) {
                    Ok(owned) => {
                        {
                            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                            inner.model = None;
                            inner.backend.clear();
                        }
                        owned.shutdown().await?;
                        break;
                    }
                    Err(shared) => {
                        session = shared;
                        if tokio::time::Instant::now() >= deadline {
                            self.inner.lock().unwrap_or_else(|e| e.into_inner()).session =
                                Some(session);
                            return Err(
                                "Requests are still stopping; retry the model operation.".into()
                            );
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn request(
        self: &Arc<Self>,
        id: String,
        method: String,
        path: String,
        body: Option<serde_json::Value>,
    ) -> Result<DesktopResponse, String> {
        if id.is_empty() || id.len() > 128 {
            return Err("Invalid request ID.".into());
        }
        let (session, mut cancelled) = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if inner.loading || self.updater.is_running() {
                return Err("Model is busy changing or updating.".into());
            }
            inner
                .early_cancellations
                .retain(|_, at| at.elapsed() < Duration::from_secs(30));
            if inner.early_cancellations.remove(&id).is_some() {
                return Err("Request cancelled.".into());
            }
            if inner.requests.len() >= 16 || inner.requests.contains_key(&id) {
                return Err("Too many requests or duplicate request ID.".into());
            }
            let session = inner.session.clone().ok_or("Load a model first.")?;
            let (cancel, receiver) = oneshot::channel();
            inner.requests.insert(id.clone(), Some(cancel));
            (session, receiver)
        };
        let _lease = RequestLease {
            state: self.clone(),
            id,
        };
        let limit = match path.as_str() {
            "/v1/chat/completions" | "/api/chat" => 24 * 1024 * 1024,
            _ => 1024 * 1024,
        };
        let body = encode_body(body.as_ref(), limit)?;
        tokio::select! {
            biased;
            _ = &mut cancelled => Err("Request cancelled.".into()),
            response = session.request(&method, &path, body) => response,
        }
    }

    pub fn cancel(&self, id: &str) {
        if id.is_empty() || id.len() > 128 {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        // Keep the ID reserved until its request future drops. Otherwise a new
        // request could reuse it and have its cancellation entry removed by the
        // old request's cleanup guard.
        if let Some(pending) = inner.requests.get_mut(id) {
            if let Some(cancel) = pending.take() {
                let _ = cancel.send(());
            }
        } else {
            // IPC dispatch may deliver cancellation before the async request is
            // registered. Keep a small expiring tombstone for that ordering.
            if inner.early_cancellations.len() >= 64
                && let Some(oldest) = inner
                    .early_cancellations
                    .iter()
                    .min_by_key(|(_, at)| *at)
                    .map(|(id, _)| id.clone())
            {
                inner.early_cancellations.remove(&oldest);
            }
            inner
                .early_cancellations
                .insert(id.into(), std::time::Instant::now());
        }
    }
}

fn encode_body(body: Option<&serde_json::Value>, limit: usize) -> Result<Vec<u8>, String> {
    struct Buffer {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl std::io::Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
                return Err(std::io::Error::other("request body exceeds its byte limit"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut buffer = Buffer {
        bytes: Vec::new(),
        limit,
    };
    if let Some(body) = body {
        serde_json::to_writer(&mut buffer, body).map_err(|e| e.to_string())?;
    }
    Ok(buffer.bytes)
}

struct DeletionLease<'a>(&'a AppState);
impl Drop for DeletionLease<'_> {
    fn drop(&mut self) {
        self.0
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .loading = false;
    }
}

struct RequestLease {
    state: Arc<AppState>,
    id: String,
}
impl Drop for RequestLease {
    fn drop(&mut self) {
        self.state
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .requests
            .remove(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options(gpu: &str) -> LoadOptions {
        LoadOptions {
            model: "model.gguf".into(),
            ollama: false,
            ollama_dir: None,
            projector: None,
            gpu: gpu.into(),
            context: 1024,
        }
    }
    #[test]
    fn gpu_policy_and_explicit_cpu_are_preserved() {
        assert_eq!(
            options("auto").config().unwrap().gpu_layers,
            config::default_gpu_layers()
        );
        assert_eq!(options("cpu").config().unwrap().gpu_layers, 0);
        assert_eq!(options("1").config().unwrap().gpu_layers, 1);
        assert_eq!(options("all").config().unwrap().gpu_layers, 999);
        assert_eq!(options("auto").config().unwrap().memory_fallback, "error");
        assert!(options("bogus").config().is_err());
        let mut invalid = options("auto");
        invalid.context = 1;
        assert!(invalid.config().is_err());
    }
    #[test]
    fn qualified_local_profile_preserves_identity_and_explicit_load_policy() {
        for gpu in ["auto", "cpu", "all", "7"] {
            let mut value = options(gpu);
            value.model = "lma:team/profile:latest".into();
            let config = value.config().unwrap();
            assert_eq!(config.model, PathBuf::from("lma:team/profile:latest"));
            assert_eq!(config.context, 1024);
            assert_eq!(config.explicit_load, ["num_ctx", "num_gpu"]);
            assert!(config.definition.is_none());
        }
    }

    #[test]
    fn ollama_selector_keeps_tag_and_store() {
        let mut value = options("cpu");
        value.model = "hf.co/owner/model:Q4_K_M".into();
        value.ollama = true;
        value.ollama_dir = Some("custom store".into());
        let config = value.config().unwrap();
        assert_eq!(
            config.model,
            PathBuf::from("ollama:hf.co/owner/model:Q4_K_M")
        );
        assert_eq!(config.ollama_dir, Some("custom store".into()));
    }
    #[test]
    fn ipc_serialization_stops_at_the_byte_limit() {
        let body = serde_json::json!({"prompt": "é"});
        let expected = serde_json::to_vec(&body).unwrap();
        assert_eq!(encode_body(Some(&body), expected.len()).unwrap(), expected);
        assert!(encode_body(Some(&body), expected.len() - 1).is_err());
        assert!(encode_body(None, 0).unwrap().is_empty());
    }

    #[tokio::test]
    async fn cancellation_before_dispatch_is_remembered_and_bounded() {
        let state = Arc::new(AppState::default());
        for n in 0..100 {
            state.cancel(&format!("request-{n}"));
        }
        assert_eq!(state.inner.lock().unwrap().early_cancellations.len(), 64);
        let error = state
            .request("request-99".into(), "GET".into(), "/v1/models".into(), None)
            .await
            .unwrap_err();
        assert_eq!(error, "Request cancelled.");
    }

    #[test]
    fn cancellation_reserves_id_until_request_cleanup() {
        let state = Arc::new(AppState::default());
        let (cancel, receiver) = oneshot::channel();
        state
            .inner
            .lock()
            .unwrap()
            .requests
            .insert("same-id".into(), Some(cancel));
        let lease = RequestLease {
            state: state.clone(),
            id: "same-id".into(),
        };
        state.cancel("same-id");
        assert!(receiver.blocking_recv().is_ok());
        assert!(state.inner.lock().unwrap().requests.contains_key("same-id"));
        state.cancel("same-id");
        drop(lease);
        assert!(!state.inner.lock().unwrap().requests.contains_key("same-id"));
    }

    #[tokio::test]
    async fn failed_preflight_keeps_existing_model_metadata_and_requests() {
        let state = AppState::default();
        let (cancel, mut receiver) = oneshot::channel();
        {
            let mut inner = state.inner.lock().unwrap();
            inner.model = Some("existing-model".into());
            inner.backend = "CPU".into();
            inner.requests.insert("active".into(), Some(cancel));
        }
        let mut invalid = options("cpu");
        invalid.model = std::env::temp_dir()
            .join("lma-no-such-directory/model.gguf")
            .to_string_lossy()
            .into_owned();
        assert!(state.load(invalid).await.is_err());
        assert!(
            state
                .update(PathBuf::from("missing-registration"))
                .await
                .is_err()
        );
        assert_eq!(state.status().model.as_deref(), Some("existing-model"));
        assert!(!state.status().loading);
        assert!(matches!(
            receiver.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert!(state.inner.lock().unwrap().requests["active"].is_some());
    }

    #[tokio::test]
    async fn deletion_refuses_loaded_active_loading_and_update_without_cancelling() {
        for reason in ["loaded", "requests", "loading", "update"] {
            let state = Arc::new(AppState::default());
            let (cancel, mut receiver) = oneshot::channel();
            {
                let mut inner = state.inner.lock().unwrap();
                match reason {
                    "loaded" => inner.model = Some("fixture-loaded-model".into()),
                    "loading" => inner.loading = true,
                    "update" => inner.update_requested = true,
                    _ => {}
                }
                if reason == "requests" {
                    inner.requests.insert("active".into(), Some(cancel));
                }
            }
            let error = state
                .delete_model(|| -> Result<(), String> {
                    panic!("a busy state must not run deletion")
                })
                .await
                .unwrap_err();
            let inner = state.inner.lock().unwrap();
            match reason {
                "loaded" => {
                    assert!(error.contains("Unload"));
                    assert_eq!(inner.model.as_deref(), Some("fixture-loaded-model"));
                }
                "requests" => {
                    assert!(error.contains("active requests"));
                    assert!(inner.requests["active"].is_some());
                    assert!(matches!(
                        receiver.try_recv(),
                        Err(oneshot::error::TryRecvError::Empty)
                    ));
                }
                "loading" => assert!(inner.loading),
                "update" => assert!(inner.update_requested),
                _ => unreachable!(),
            }
        }
        let state = Arc::new(AppState::default());
        let _lifecycle = state.lifecycle.lock().await;
        assert!(state.delete_model(|| Ok(())).await.is_err());
    }

    #[tokio::test]
    async fn unloaded_state_and_missing_model_requests() {
        let state = Arc::new(AppState::default());
        assert!(!state.status().loaded);
        assert!(!state.status().updating);
        assert!(
            state
                .request("id".into(), "GET".into(), "/v1/models".into(), None)
                .await
                .unwrap_err()
                .contains("Load a model")
        );
        state.cancel("unknown");
        assert!(!state.unload().await.unwrap().loaded);
    }
}
