//! Exercise desktop lifecycle/update logic without compiling a native webview.
//! The Tauri executable uses these exact modules, not a separate mock backend.
#[allow(dead_code)]
#[path = "../desktop/src/app_settings.rs"]
mod app_settings;
#[allow(dead_code)]
#[path = "../desktop/src/discovery.rs"]
mod discovery;
#[allow(dead_code)]
#[path = "../desktop/src/model_delete.rs"]
mod model_delete;
#[allow(dead_code)]
#[path = "../desktop/src/model_import.rs"]
mod model_import;
#[allow(dead_code)]
#[path = "../desktop/src/preferences.rs"]
mod preferences;
#[allow(dead_code)]
#[path = "../desktop/src/state.rs"]
mod state;
#[allow(dead_code)]
#[path = "../desktop/src/update.rs"]
mod update;
#[allow(dead_code)]
#[path = "../desktop/src/workspace.rs"]
mod workspace;

mod deletion {
    use super::*;
    use lma::models::registry::{self, Entry};
    use serde_json::{Value, json};
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::Arc,
        time::Duration,
    };

    struct Fixture {
        _temp: tempfile::TempDir,
        root: PathBuf,
        config: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let temp = tempfile::tempdir().unwrap();
            let root = fs::canonicalize(temp.path()).unwrap();
            let config = root.join("config");
            for relative in ["config", "local", "studio", "ollama-a", "ollama-b"] {
                fs::create_dir(root.join(relative)).unwrap();
            }
            Self {
                _temp: temp,
                root,
                config,
            }
        }
        fn locations(&self) -> discovery::DeletionLocations {
            discovery::DeletionLocations {
                local_dirs: vec![self.root.join("local"), self.root.join("studio")],
                lm_dirs: vec![self.root.join("studio")],
                ollama_dirs: vec![
                    Some(self.root.join("ollama-a")),
                    Some(self.root.join("ollama-b")),
                ],
            }
        }
        fn model(&self, relative: &str) -> PathBuf {
            let path = self.root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, b"GGUFfixture").unwrap();
            path
        }
        fn alias(&self, label: &str, path: &Path) {
            registry::put(
                &self.root.join("local"),
                label,
                Entry {
                    path: path.to_path_buf(),
                    projector: None,
                },
            )
            .unwrap();
        }
        fn saved(&self, model: &str, store: Option<PathBuf>) {
            fs::write(
                self.config.join("model-preferences.json"),
                serde_json::to_vec(&json!({
                    "version":1, "last_model":{
                        "model":model, "ollama":store.is_some(), "ollamaDir":store,
                        "projector":null, "gpu":"cpu", "context":128
                    }
                }))
                .unwrap(),
            )
            .unwrap();
        }
        fn delete(&self, id: &str) -> Result<Value, String> {
            model_delete::run(&self.config, id, || Ok(self.locations()))
        }
        fn ollama(&self, store: &str, name: &str, digest: char) -> (PathBuf, PathBuf) {
            let root = self.root.join(store);
            let manifest = root.join(format!(
                "manifests/registry.ollama.ai/library/{name}/latest"
            ));
            let blob = root.join(format!("blobs/sha256-{}", digest.to_string().repeat(64)));
            fs::create_dir_all(manifest.parent().unwrap()).unwrap();
            fs::create_dir_all(blob.parent().unwrap()).unwrap();
            fs::write(&blob, b"GGUFfixture").unwrap();
            fs::write(
                &manifest,
                serde_json::to_vec(&json!({
                    "schemaVersion":2, "layers":[{
                        "mediaType":"application/vnd.ollama.image.model",
                        "digest":format!("sha256:{}", digest.to_string().repeat(64))
                    }]
                }))
                .unwrap(),
            )
            .unwrap();
            (manifest, blob)
        }
    }

    #[test]
    fn exact_fresh_ids_only_and_local_preference_cleanup() {
        let f = Fixture::new();
        let path = f.model("local/example.gguf");
        let other = f.model("local/other.gguf");
        let id = format!("lma:{}", path.display());
        f.saved(path.to_str().unwrap(), None);
        for invalid in [
            "example",
            "example.gguf",
            path.to_str().unwrap(),
            "lma:missing:latest",
            "",
        ] {
            assert!(f.delete(invalid).is_err(), "{invalid}");
            assert!(path.exists());
        }
        let out = f.delete(&id).unwrap();
        assert_eq!(out["source"], "lma");
        assert!(!path.exists());
        assert!(other.exists());
        assert!(preferences::load(&f.config).unwrap().is_none());
        // A stale ID must not resolve to a newly installed same-name model.
        f.model("local/nested/example.gguf");
        assert!(f.delete(&id).is_err());
        assert!(f.root.join("local/nested/example.gguf").exists());
    }

    #[test]
    fn relabeled_lm_studio_alias_uses_lma_guard_and_preserves_external_weights() {
        let f = Fixture::new();
        let path = f.model("studio/owner/model.gguf");
        f.alias("profile:latest", &path);
        let catalog = f.locations().catalog();
        let alias = catalog
            .models
            .iter()
            .find(|m| m.id == "lma:profile:latest")
            .unwrap();
        assert_eq!(alias.source, "lm-studio");
        f.saved("lma:profile:latest", None);
        let out = f.delete(&alias.id).unwrap();
        assert_eq!(out["source"], "lma");
        assert_eq!(out["aliasDeleted"], true);
        assert_eq!(out["deletedFiles"], json!([]));
        assert_eq!(out["retainedFiles"], json!([path]));
        assert!(path.exists());
        assert!(preferences::load(&f.config).unwrap().is_none());
        let out = f.delete(&format!("lma:{}", path.display())).unwrap();
        assert_eq!(out["source"], "lm-studio");
        assert!(!path.exists());
    }

    #[test]
    fn shared_alias_keeps_weights_and_unrelated_saved_choice() {
        let f = Fixture::new();
        let path = f.model("local/shared.gguf");
        f.alias("first:latest", &path);
        f.alias("second:latest", &path);
        f.saved("lma:second:latest", None);
        let out = f.delete("lma:first:latest").unwrap();
        assert_eq!(out["aliasDeleted"], true);
        assert!(path.exists());
        assert_eq!(
            preferences::load(&f.config).unwrap().unwrap().model,
            "lma:second:latest"
        );
        f.delete("lma:second:latest").unwrap();
        assert!(!path.exists());
        assert!(preferences::load(&f.config).unwrap().is_none());
    }

    #[test]
    fn identical_ollama_tags_delete_only_the_selected_trusted_store() {
        let f = Fixture::new();
        let (first, blob_a) = f.ollama("ollama-a", "example", 'a');
        let (second, blob_b) = f.ollama("ollama-b", "example", 'b');
        let (shared, _) = f.ollama("ollama-b", "shared", 'b');
        f.saved("example:latest", Some(f.root.join("ollama-a")));
        let selected = f
            .locations()
            .catalog()
            .models
            .into_iter()
            .find(|m| {
                m.ollama
                    && m.model == "example:latest"
                    && m.ollama_dir == Some(f.root.join("ollama-b"))
            })
            .unwrap();
        assert!(f.delete("ollama:example:latest").is_err());
        f.delete(&selected.id).unwrap();
        assert!(first.exists() && blob_a.exists());
        assert!(!second.exists());
        assert!(shared.exists() && blob_b.exists());
        assert!(preferences::load(&f.config).unwrap().is_some());
        let selected = f
            .locations()
            .catalog()
            .models
            .into_iter()
            .find(|m| m.ollama && m.model == "example:latest")
            .unwrap();
        f.delete(&selected.id).unwrap();
        assert!(!first.exists() && !blob_a.exists());
        assert!(preferences::load(&f.config).unwrap().is_none());
    }

    #[test]
    fn custom_ollama_blobs_cannot_be_deleted_as_local_files() {
        let f = Fixture::new();
        let (manifest, blob) = f.ollama("ollama-b", "example", 'b');
        let mut locations = f.locations();
        locations.local_dirs.push(f.root.join("ollama-b/blobs"));
        let id = format!("lma:{}", blob.display());
        f.alias("blob-alias:latest", &blob);
        // Alias deletion is allowed, but it must not turn a local search root
        // into ownership of an existing custom Ollama store's blobs.
        let out = model_delete::run(&f.config, "lma:blob-alias:latest", || Ok(locations.clone()))
            .unwrap();
        assert_eq!(out["aliasDeleted"], true);
        assert!(blob.exists() && manifest.exists());
        // The raw blob is not a runnable GGUF row (no .gguf suffix), so the
        // adapter's exact-catalog requirement rejects it before core deletion.
        assert!(model_delete::run(&f.config, &id, || Ok(locations)).is_err());
        assert!(blob.exists() && manifest.exists());
    }

    #[test]
    fn corrupt_preference_warns_after_success_and_corrupt_registry_prevents_deletion() {
        let f = Fixture::new();
        let path = f.model("local/example.gguf");
        let id = format!("lma:{}", path.display());
        let registry = f.root.join("local/.registry-v1.json");
        fs::write(&registry, "{broken").unwrap();
        assert!(f.delete(&id).is_err());
        assert!(path.exists());
        fs::remove_file(registry).unwrap();
        fs::write(f.config.join("model-preferences.json"), "{broken").unwrap();
        let out = f.delete(&id).unwrap();
        assert!(!path.exists());
        assert!(out["warnings"].as_array().unwrap().iter().any(|w| {
            w.as_str()
                .unwrap()
                .contains("saved model selection could not be cleaned up")
        }));
        assert_eq!(
            fs::read_to_string(f.config.join("model-preferences.json")).unwrap(),
            "{broken"
        );
    }

    #[test]
    fn trusted_locations_include_custom_settings_and_fail_closed_on_bad_metadata() {
        let f = Fixture::new();
        app_settings::save(
            &f.config,
            app_settings::Settings {
                model_directories: vec![f.root.join("local").to_str().unwrap().into()],
                lm_studio_directories: vec![f.root.join("studio").to_str().unwrap().into()],
                ollama_directory: Some(f.root.join("ollama-a").to_str().unwrap().into()),
                ..Default::default()
            },
        )
        .unwrap();
        f.saved("example:latest", Some(f.root.join("ollama-b")));
        let resolve = || discovery::deletion_locations(&f.config, f.root.join("data"), None, None);
        let locations = resolve().unwrap();
        assert!(locations.local_dirs.contains(&f.root.join("local")));
        assert!(locations.lm_dirs.contains(&f.root.join("studio")));
        assert!(
            locations
                .ollama_dirs
                .contains(&Some(f.root.join("ollama-a")))
        );
        assert!(
            locations
                .ollama_dirs
                .contains(&Some(f.root.join("ollama-b")))
        );
        for bytes in [
            "{broken",
            r#"{"schema_version":2,"source_dir":"/fixture"}"#,
            r#"{"schema_version":1,"source_dir":"relative"}"#,
            r#"{"schema_version":1,"source_dir":"/one","source_dir":"/two"}"#,
        ] {
            fs::write(f.config.join("source-install.json"), bytes).unwrap();
            assert!(resolve().is_err(), "{bytes}");
        }
        fs::remove_file(f.config.join("source-install.json")).unwrap();
        #[cfg(unix)]
        {
            let target = f.root.join("source-target");
            fs::write(
                &target,
                serde_json::to_vec(&json!({
                    "schema_version":1, "source_dir":f.root
                }))
                .unwrap(),
            )
            .unwrap();
            std::os::unix::fs::symlink(&target, f.config.join("source-install.json")).unwrap();
            assert!(resolve().is_err());
            fs::remove_file(f.config.join("source-install.json")).unwrap();
        }
        fs::write(f.config.join("app-settings.json"), "{broken").unwrap();
        assert!(resolve().is_err());
        #[cfg(unix)]
        {
            fs::remove_file(f.config.join("app-settings.json")).unwrap();
            let target = f.root.join("settings-target");
            fs::write(&target, "{}").unwrap();
            std::os::unix::fs::symlink(&target, f.config.join("app-settings.json")).unwrap();
            assert!(resolve().is_err());
        }
    }

    #[tokio::test]
    async fn deletion_gates_survive_cancelled_caller_and_worker_errors() {
        let app = Arc::new(state::AppState::default());
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let caller = {
            let app = app.clone();
            tokio::spawn(async move {
                app.delete_model(move || {
                    started.send(()).unwrap();
                    wait.recv_timeout(Duration::from_secs(10)).unwrap();
                    Ok(())
                })
                .await
            })
        };
        ready.await.unwrap();
        caller.abort();
        let _ = caller.await;
        assert!(app.selection.try_lock().is_err());
        assert!(app.discovery.try_lock().is_err());
        assert!(app.store_download.try_lock().is_err());
        assert!(
            serde_json::to_value(app.status()).unwrap()["loading"]
                .as_bool()
                .unwrap()
        );
        assert!(app.unload().await.is_err());
        assert!(app.delete_model(|| Ok(())).await.is_err());
        assert!(
            app.request("blocked".into(), "GET".into(), "/v1/models".into(), None)
                .await
                .unwrap_err()
                .contains("busy")
        );
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while serde_json::to_value(app.status()).unwrap()["loading"] == true {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(
            app.delete_model(|| Err::<(), _>("fixture error".into()))
                .await
                .is_err()
        );
        assert!(
            app.delete_model(|| -> Result<(), String> { panic!("fixture worker panic") })
                .await
                .is_err()
        );
        assert!(app.delete_model(|| Ok(())).await.is_ok());
        assert_eq!(
            serde_json::to_value(app.status()).unwrap()["loading"],
            false
        );
    }

    #[tokio::test]
    async fn blocking_adapter_resolves_locations_only_under_settings_update_lock() {
        let f = Fixture::new();
        let settings = model_import::SETTINGS_UPDATE.lock().unwrap();
        let (entered, ready) = tokio::sync::oneshot::channel();
        let (resolved, mut resolution) = tokio::sync::oneshot::channel();
        let worker = tokio::task::spawn_blocking(move || {
            entered.send(()).unwrap();
            model_delete::run(&f.config, "lma:fixture:latest", || {
                resolved.send(()).unwrap();
                Err("fixture resolution failure".into())
            })
        });
        ready.await.unwrap();
        assert!(matches!(
            resolution.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        drop(settings);
        assert!(
            worker
                .await
                .unwrap()
                .unwrap_err()
                .contains("fixture resolution failure")
        );
        resolution.await.unwrap();
    }

    #[tokio::test]
    async fn conflicting_public_gates_refuse_without_running_the_closure() {
        let app = Arc::new(state::AppState::default());
        for gate in [&app.selection, &app.discovery, &app.store_download] {
            let _guard = gate.lock().await;
            assert!(
                app.delete_model(|| -> Result<(), String> { panic!("deletion must not run") })
                    .await
                    .is_err()
            );
        }
        assert_eq!(
            serde_json::to_value(app.status()).unwrap()["loading"],
            false
        );
    }
}

/// Read-only check of the user's real installed store, explicitly opted in.
#[test]
#[ignore = "requires LMA_TEST_OLLAMA_MODEL in the installed Ollama store"]
fn real_installed_model_discovery() {
    let expected = std::env::var("LMA_TEST_OLLAMA_MODEL").expect("set LMA_TEST_OLLAMA_MODEL");
    let found = lma::models::discover(lma::models::DiscoveryOptions::default());
    let model = found
        .models
        .iter()
        .find(|model| model.ollama && model.model == expected)
        .unwrap_or_else(|| {
            panic!(
                "Installed model not discovered; warnings: {:?}",
                found.warnings
            )
        });
    assert!(model.size_bytes > 0);
    assert!(model.ollama_dir.as_ref().unwrap().is_absolute());
    println!(
        "Discovered {} installed models; found {} ({} bytes)",
        found.models.len(),
        model.name,
        model.size_bytes
    );
}

/// Explicit opt-in: uses installed weights only, never downloads or updates.
#[tokio::test]
#[ignore = "requires LMA_TEST_OLLAMA_MODEL and an installed compatible model"]
async fn real_installed_model_desktop_lifecycle() {
    use serde_json::json;
    use std::{sync::Arc, time::Duration};
    let model = std::env::var("LMA_TEST_OLLAMA_MODEL")
        .expect("set LMA_TEST_OLLAMA_MODEL to an installed Ollama model tag");
    let app = Arc::new(state::AppState::default());
    let options = serde_json::from_value(json!({
        "model": model, "ollama": true, "ollamaDir": null,
        "projector": null, "gpu": "1", "context": 256
    }))
    .unwrap();
    app.load(options).await.expect("load installed model");
    assert_eq!(serde_json::to_value(app.status()).unwrap()["loaded"], true);
    let models = app
        .request("models".into(), "GET".into(), "/v1/models".into(), None)
        .await
        .unwrap();
    assert_eq!(models.status, 200);
    assert_eq!(models.body["data"][0]["id"], model);

    // Predictable replacement/update errors must not discard the loaded model.
    let invalid = serde_json::from_value(json!({
        "model": "/lma-no-such-model/model.gguf", "ollama": false,
        "ollamaDir": null, "projector": null, "gpu": "cpu", "context": 256
    }))
    .unwrap();
    assert!(app.load(invalid).await.is_err());
    let directory = tempfile::tempdir().unwrap();
    assert!(app.update(directory.path().to_path_buf()).await.is_err());
    assert_eq!(serde_json::to_value(app.status()).unwrap()["loaded"], true);

    let oversized = app
        .request(
            "overflow".into(),
            "POST".into(),
            "/v1/completions".into(),
            Some(json!({
                "prompt": "hello ".repeat(1024), "max_tokens": 4
            })),
        )
        .await
        .unwrap();
    assert_eq!(oversized.status, 400);
    let response = app
        .request(
            "generate".into(),
            "POST".into(),
            "/v1/chat/completions".into(),
            Some(json!({
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 4, "temperature": 0, "seed": 42
            })),
        )
        .await
        .unwrap();
    assert_eq!(response.status, 200, "{}", response.body);
    assert!(
        !response.body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .is_empty()
    );
    println!("Desktop core response: {}", response.body);

    let worker = {
        let app = app.clone();
        tokio::spawn(async move {
            app.request(
                "cancel-me".into(),
                "POST".into(),
                "/v1/completions".into(),
                Some(json!({
                    "prompt": "Explain a linked list.", "max_tokens": 128, "temperature": 0
                })),
            )
            .await
        })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    app.cancel("cancel-me");
    let cancelled = tokio::time::timeout(Duration::from_secs(10), worker)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cancelled.unwrap_err(), "Request cancelled.");
    app.unload().await.unwrap();
    assert_eq!(serde_json::to_value(app.status()).unwrap()["loaded"], false);
    println!(
        "Desktop model identity, preserved session, error recovery, cancellation, and unload passed"
    );
}
