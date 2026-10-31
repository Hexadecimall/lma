#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app_settings;
mod dictation;
mod discovery;
mod model_delete;
mod model_import;
mod preferences;
mod state;
mod update;
mod workspace;

use state::{AppState, LoadOptions, Status};
use std::sync::Arc;
use tauri::Manager;

#[tauri::command]
fn desktop_status(state: tauri::State<'_, Arc<AppState>>) -> Status {
    #[cfg(debug_assertions)]
    {
        static FRONTEND_CONNECTED: std::sync::Once = std::sync::Once::new();
        FRONTEND_CONNECTED.call_once(|| eprintln!("lma desktop: frontend IPC connected"));
    }
    state.status()
}

#[tauri::command]
async fn load_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    options: LoadOptions,
) -> Result<Status, String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let state = state.inner().clone();
    // Lifecycle tasks outlive a disconnected webview invocation. Never abandon a
    // native teardown barrier midway and allow a second backend to initialize.
    tauri::async_runtime::spawn(async move {
        let _selection = state
            .selection
            .try_lock()
            .map_err(|_| "A model selection is already running.")?;
        let settings_dir = config_dir.clone();
        let options = tauri::async_runtime::spawn_blocking(move || {
            app_settings::load(&settings_dir)?.apply(options)
        })
        .await
        .map_err(|e| format!("Load policy task failed: {e}"))??;
        state.load(options.clone()).await?;
        let saved =
            tauri::async_runtime::spawn_blocking(move || preferences::save(&config_dir, options))
                .await
                .map_err(|e| e.to_string())
                .and_then(|result| result);
        state.set_preference_warning(saved.err().map(|error| {
            format!("Model loaded, but the selection could not be remembered: {error}")
        }));
        Ok(state.status())
    })
    .await
    .map_err(|e| format!("Model task failed: {e}"))?
}

#[tauri::command]
async fn model_import(
    app: tauri::AppHandle,
    options: LoadOptions,
    name: Option<String>,
) -> Result<model_import::ImportResult, String> {
    let directory = app.path().app_config_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || model_import::import(&directory, options, name))
        .await
        .map_err(|e| format!("Model import task failed: {e}"))?
}

#[tauri::command]
async fn model_delete(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    id: String,
) -> Result<serde_json::Value, String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let cwd = std::env::current_dir().ok();
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(std::path::Path::to_path_buf);
    state
        .inner()
        .delete_model(move || {
            model_delete::run(&config_dir, &id, || {
                discovery::deletion_locations(&config_dir, data_dir, cwd, source)
            })
        })
        .await
}

#[tauri::command]
async fn app_settings_load(app: tauri::AppHandle) -> Result<app_settings::Settings, String> {
    let directory = app.path().app_config_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || app_settings::load(&directory))
        .await
        .map_err(|e| format!("Settings load failed: {e}"))?
}

#[tauri::command]
async fn app_settings_save(
    app: tauri::AppHandle,
    settings: app_settings::Settings,
) -> Result<(), String> {
    let directory = app.path().app_config_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        let _settings = model_import::SETTINGS_UPDATE
            .lock()
            .map_err(|_| "Settings update lock failed.")?;
        app_settings::save(&directory, settings)
    })
    .await
    .map_err(|e| format!("Settings save failed: {e}"))?
}

// Local disk reads only; all callers dispatch this helper to a blocking worker.
fn local_catalog(app: &tauri::AppHandle) -> Result<discovery::DiscoveryResult, String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let cwd = std::env::current_dir().ok();
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(std::path::Path::to_path_buf);
    Ok(discovery::discover(&config_dir, data_dir, cwd, source))
}

#[tauri::command]
async fn store_search(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    query: String,
    source: String,
) -> Result<lma::fetch::store::SearchResult, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _search = state
            .store_search
            .try_lock()
            .map_err(|_| "A store search is already running.")?;
        let installed = local_catalog(&app)?;
        let mut result = lma::fetch::store::search(&query, &source, &installed.models)?;
        result.warnings.extend(installed.warnings);
        result.warnings.truncate(64);
        Ok(result)
    })
    .await
    .map_err(|e| format!("Store search failed: {e}"))?
}

#[tauri::command]
async fn store_download(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    id: String,
    source: String,
    file: Option<String>,
) -> Result<lma::fetch::store::DownloadResult, String> {
    let state = state.inner().clone();
    // Keep the gate owned by the worker even if its webview invocation disappears.
    tauri::async_runtime::spawn_blocking(move || {
        let _download = state
            .store_download
            .try_lock()
            .map_err(|_| "A model download is already running.")?;
        let installed = local_catalog(&app)?;
        lma::fetch::store::download(&id, &source, file.as_deref(), &installed.models)
    })
    .await
    .map_err(|e| format!("Store download failed: {e}"))?
}

#[tauri::command]
async fn discover_models(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<discovery::DiscoveryResult, String> {
    let _scan = state
        .discovery
        .try_lock()
        .map_err(|_| "Model discovery is already running.")?;
    let found = tauri::async_runtime::spawn_blocking(move || local_catalog(&app))
        .await
        .map_err(|e| format!("Model discovery failed: {e}"))??;
    #[cfg(debug_assertions)]
    eprintln!(
        "lma desktop: model discovery complete ({} models, {} warnings)",
        found.models.len(),
        found.warnings.len()
    );
    Ok(found)
}

#[tauri::command]
async fn unload_model(state: tauri::State<'_, Arc<AppState>>) -> Result<Status, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn(async move { state.unload().await })
        .await
        .map_err(|e| format!("Model task failed: {e}"))?
}

#[tauri::command]
async fn api_request(
    state: tauri::State<'_, Arc<AppState>>,
    id: String,
    method: String,
    path: String,
    body: Option<serde_json::Value>,
) -> Result<lma::desktop::DesktopResponse, String> {
    state.inner().request(id, method, path, body).await
}

#[tauri::command]
fn cancel_request(state: tauri::State<'_, Arc<AppState>>, id: String) {
    state.cancel(&id);
}

#[tauri::command]
fn update_status(state: tauri::State<'_, Arc<AppState>>) -> update::UpdateStatus {
    state.updater.status()
}

#[tauri::command]
async fn workspace_load(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || workspace::load(&config_dir))
        .await
        .map_err(|e| format!("Workspace load failed: {e}"))?
}

#[tauri::command]
async fn workspace_save(app: tauri::AppHandle, workspace: serde_json::Value) -> Result<(), String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || workspace::save(&config_dir, workspace))
        .await
        .map_err(|e| format!("Workspace save failed: {e}"))?
}

#[tauri::command]
async fn source_update(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<serde_json::Value, String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let state = state.inner().clone();
    let message = tauri::async_runtime::spawn(async move { state.update(config_dir).await })
        .await
        .map_err(|e| format!("Update task failed: {e}"))??;
    Ok(serde_json::json!({"message": message}))
}

#[cfg(debug_assertions)]
fn native_ui_test_enabled() -> bool {
    std::env::var("LMA_DESKTOP_UI_TEST").as_deref() == Ok("1")
        && std::env::var_os("LMA_DESKTOP_TEST_DIR").is_some()
}

#[cfg(debug_assertions)]
#[tauri::command]
fn desktop_test_report(report: serde_json::Value) -> Result<(), String> {
    if !native_ui_test_enabled() {
        return Err("Native UI test mode is disabled.".into());
    }
    let report = serde_json::to_string(&report).map_err(|e| e.to_string())?;
    if report.len() > 64 * 1024 {
        return Err("Native UI report is too large.".into());
    }
    eprintln!("lma desktop: native UI report {report}");
    Ok(())
}

#[cfg(debug_assertions)]
#[tauri::command]
fn desktop_test_resize(
    window: tauri::WebviewWindow,
    width: u32,
    height: u32,
) -> Result<(), String> {
    if !native_ui_test_enabled() {
        return Err("Native UI test mode is disabled.".into());
    }
    if !(640..=1280).contains(&width) || !(480..=1000).contains(&height) {
        return Err("Invalid native test window size.".into());
    }
    window
        .set_size(tauri::LogicalSize::new(width, height))
        .map_err(|e| e.to_string())
}

fn main() {
    let mut context = tauri::generate_context!();
    if context.config().app.app_directories_override.is_none() {
        let directory = lma::models::storage::root()
            .expect("lma needs an absolute HOME/USERPROFILE for desktop storage");
        context.config_mut().app.app_directories_override = Some(
            tauri::utils::config::AppDirectoriesOverride::Root(directory),
        );
    }
    #[cfg(debug_assertions)]
    if let Some(directory) = std::env::var_os("LMA_DESKTOP_TEST_DIR") {
        let directory = std::path::PathBuf::from(directory);
        assert!(
            directory.is_absolute() && directory.is_dir(),
            "LMA_DESKTOP_TEST_DIR must be an existing absolute test directory"
        );
        // Keep launch tests away from saved model choices and user application
        // data, while still discovering the real read-only Ollama store.
        context.config_mut().app.app_directories_override = Some(
            tauri::utils::config::AppDirectoriesOverride::Root(directory),
        );
    }
    let app = tauri::Builder::default()
        .manage(Arc::new(AppState::default()))
        .manage(dictation::DictationState::default())
        .invoke_handler(tauri::generate_handler![
            desktop_status,
            app_settings_load,
            app_settings_save,
            store_search,
            store_download,
            discover_models,
            model_import,
            model_delete,
            load_model,
            unload_model,
            api_request,
            cancel_request,
            update_status,
            source_update,
            workspace_load,
            workspace_save,
            dictation::dictation_settings,
            dictation::dictation_models,
            dictation::dictation_set_model,
            dictation::dictation_install_model,
            dictation::dictation_model_plan,
            dictation::dictation_transcribe,
            #[cfg(debug_assertions)]
            desktop_test_report,
            #[cfg(debug_assertions)]
            desktop_test_resize,
        ])
        .setup(|app| {
            tauri::WebviewWindowBuilder::from_config(app.handle(), &app.config().app.windows[0])?
                .on_permission_request(|webview, kind| {
                    use tauri::webview::{PermissionKind, PermissionResponse};
                    let local = webview.url().is_ok_and(|url| {
                        matches!(
                            (url.scheme(), url.host_str()),
                            ("tauri", Some("localhost"))
                                | ("http" | "https", Some("tauri.localhost"))
                        )
                    });
                    if local && matches!(kind, PermissionKind::Microphone) {
                        // Linux's Default denies capture without a prompt. The
                        // app requests it only after an explicit microphone click.
                        #[cfg(target_os = "linux")]
                        return PermissionResponse::Allow;
                        #[cfg(not(target_os = "linux"))]
                        PermissionResponse::Default
                    } else {
                        PermissionResponse::Deny
                    }
                })
                .on_navigation(|url| {
                    matches!(
                        (url.scheme(), url.host_str()),
                        ("tauri", Some("localhost")) | ("http" | "https", Some("tauri.localhost"))
                    )
                })
                .on_page_load(|window, payload| {
                    #[cfg(debug_assertions)]
                    if payload.event() == tauri::webview::PageLoadEvent::Finished
                        && native_ui_test_enabled()
                        && let Err(error) = window.eval(include_str!("native_smoke.js"))
                    {
                        eprintln!("lma desktop: native UI probe failed: {error}");
                    }
                    #[cfg(not(debug_assertions))]
                    let _ = (window, payload);
                })
                .build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event
                && window.state::<Arc<AppState>>().is_updating()
            {
                api.prevent_close();
            }
        })
        .build(context)
        .expect("could not initialize lma desktop");
    app.run(|app, event| {
        if let tauri::RunEvent::ExitRequested { api, .. } = event
            && app.state::<Arc<AppState>>().is_updating()
        {
            // Keep progress visible and publication supervised until the updater
            // completes. No forced quit/relaunch or model process termination.
            api.prevent_exit();
        }
    });
}
