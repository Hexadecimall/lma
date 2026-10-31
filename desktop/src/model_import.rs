//! Register existing GGUF weights only: no copying, conversion, download, or load.
//! The command host calls `import` on a blocking worker. Tests inject only fixture
//! roots through `import_into`; the webview cannot choose a registry destination.
use crate::{app_settings, state::LoadOptions};
use lma::models::{self, registry};
use serde::Serialize;
use std::{
    fs::{self, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
};

static IMPORT: Mutex<()> = Mutex::new(());
// Also acquired by the host's settings-save command so this read/modify/write
// cannot clobber a concurrent settings save inside this desktop process.
pub(crate) static SETTINGS_UPDATE: Mutex<()> = Mutex::new(());

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ImportResult {
    pub model: String,
    pub path: String,
}

pub fn import(
    config_dir: &Path,
    options: LoadOptions,
    name: Option<String>,
) -> Result<ImportResult, String> {
    import_into(
        config_dir,
        &models::discovery::default_models_dir()?,
        options,
        name,
    )
}

/// Same implementation as the command, with host-owned roots injected for tests.
pub(crate) fn import_into(
    config_dir: &Path,
    registry_root: &Path,
    options: LoadOptions,
    name: Option<String>,
) -> Result<ImportResult, String> {
    let _import = IMPORT.lock().map_err(|_| "Model import lock failed.")?;
    let _settings = SETTINGS_UPDATE
        .lock()
        .map_err(|_| "Settings update lock failed.")?;
    // Validate explicit names before filesystem changes. Unlike generated labels,
    // user-supplied names are never sanitized or silently renamed.
    let explicit_name = name.as_deref().map(registry::name).transpose()?;
    let config = options.config()?;
    let aliases = registry::read(registry_root)?;
    // Resolve known aliases directly, including an untagged spelling. This also
    // keeps fixture imports independent of real host model directories.
    let alias = if config.model.is_file() {
        None
    } else {
        config
            .model
            .to_str()
            .and_then(|text| registry::name(text).ok())
            .and_then(|label| aliases.get(&label).map(|entry| (label, entry)))
    };
    let resolved = match alias {
        Some((label, entry)) => models::ResolvedModel {
            model: entry.path.clone(),
            projector: entry.projector.clone(),
            registry_name: Some(label.clone()),
            name: Some(label),
        },
        None => models::resolve(&config.model, config.ollama_dir.as_deref())?,
    };
    let entry = registry::Entry {
        path: canonical_gguf(&resolved.model, "Model")?,
        projector: config
            .projector
            .or(resolved.projector)
            .as_deref()
            .map(|path| canonical_gguf(path, "Projector"))
            .transpose()?,
    };
    let path = text_path(&entry.path)?;
    let label = explicit_name
        .or_else(|| {
            resolved
                .name
                .as_deref()
                .and_then(|name| registry::name(name).ok())
        })
        .or_else(|| {
            aliases
                .iter()
                .find(|(_, previous)| compatible(previous, &entry))
                .map(|(label, _)| label.clone())
        })
        .map(Ok)
        .unwrap_or_else(|| default_label(&entry.path))?;
    if let Some(previous) = aliases.get(&label) {
        require_compatible(&label, previous, &entry)?;
    }

    // Only remember an explicitly supplied store actually containing the resolved
    // weights. A local file or unrelated lma alias must not change Ollama settings.
    let custom_store = config
        .ollama_dir
        .as_deref()
        .filter(|_| {
            !config.model.is_file()
                && config
                    .model
                    .to_str()
                    .is_some_and(|text| text.starts_with("ollama:"))
        })
        .map(fs::canonicalize)
        .transpose()
        .map_err(|e| format!("Cannot resolve Ollama store: {e}"))?
        .filter(|root| entry.path.starts_with(root));

    let pending_settings = if let Some(root) = custom_store {
        // Reject corrupt/redirected settings before publishing an alias, and keep
        // the host's settings lock until both publications (or rollback) finish.
        let mut settings = app_settings::load(config_dir)?;
        let root = text_path(&root)?;
        if settings.ollama_directory.as_deref() != Some(root.as_str()) {
            settings.ollama_directory = Some(root);
            Some(settings)
        } else {
            None
        }
    } else {
        None
    };

    let registered = if aliases.contains_key(&label) {
        false
    } else {
        register_alias(registry_root, &label, &entry)?
    };
    if let Some(settings) = pending_settings
        && let Err(error) = app_settings::save(config_dir, settings)
    {
        // Settings save is atomic. Undo only this import's new alias, never an
        // idempotently reused alias or any model/projector files.
        if registered && let Err(rollback) = rollback_alias(registry_root, &label, &entry) {
            return Err(format!(
                "{error}; could not roll back newly registered alias '{label}': {rollback}. The alias may remain registered; previous settings were not changed."
            ));
        }
        return Err(error);
    }
    Ok(ImportResult { model: label, path })
}

fn register_alias(root: &Path, label: &str, entry: &registry::Entry) -> Result<bool, String> {
    match registry::put(root, label, entry.clone()) {
        Ok(_) => Ok(true),
        Err(error) => {
            // A simultaneous identical registration is idempotent, but not ours
            // to remove if a later settings save fails.
            match registry::read(root)?.get(label) {
                Some(previous) if compatible(previous, entry) => {
                    require_compatible(label, previous, entry)?;
                    Ok(false)
                }
                _ => Err(error),
            }
        }
    }
}

fn rollback_alias(root: &Path, label: &str, entry: &registry::Entry) -> Result<(), String> {
    match registry::read(root)?.get(label) {
        Some(previous) if previous == entry => registry::delete(root, label),
        None => Ok(()),
        _ => Err("alias changed after publication; refusing to remove it".into()),
    }
}

fn compatible(previous: &registry::Entry, entry: &registry::Entry) -> bool {
    fs::canonicalize(&previous.path).ok().as_ref() == Some(&entry.path)
        && entry.projector.as_ref().is_none_or(|projector| {
            previous
                .projector
                .as_deref()
                .and_then(|p| fs::canonicalize(p).ok())
                .as_ref()
                == Some(projector)
        })
}

fn require_compatible(
    label: &str,
    previous: &registry::Entry,
    entry: &registry::Entry,
) -> Result<(), String> {
    if !compatible(previous, entry) {
        return Err(format!(
            "Model alias '{label}' already refers to different weights or a different projector; choose another name. Nothing was overwritten."
        ));
    }
    if let Some(projector) = &previous.projector {
        canonical_gguf(projector, "Registered projector")?;
    }
    Ok(())
}

/// An ordinary external filename becomes an ASCII registry component, capped at
/// 255 bytes, then normalized by registry::name (an omitted tag means :latest).
/// Invalid runs become one '-', leading/trailing punctuation is removed, and an
/// empty/non-ASCII-only stem becomes 'model'. No numbered collision suffixes.
pub(crate) fn default_label(path: &Path) -> Result<String, String> {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let mut label = String::new();
    for character in stem.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
            label.push(character);
        } else if !label.ends_with('-') {
            label.push('-');
        }
    }
    let label = label.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    let label = if label.is_empty() { "model" } else { label };
    registry::name(&label[..label.len().min(255)])
}

fn text_path(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| "Imported paths must be UTF-8.".into())
}

fn canonical_gguf(path: &Path, role: &str) -> Result<PathBuf, String> {
    let path = fs::canonicalize(path).map_err(|e| format!("Cannot resolve {role} path: {e}"))?;
    let before = fs::symlink_metadata(&path).map_err(|e| format!("Cannot inspect {role}: {e}"))?;
    if !before.file_type().is_file() {
        return Err(format!("{role} must be an existing regular GGUF file."));
    }
    let mut flags = OpenOptions::new();
    flags.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        flags.custom_flags(0x20000 | 0x800); // O_NOFOLLOW | O_NONBLOCK
    }
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    ))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        flags.custom_flags(0x100 | 0x4);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        flags.custom_flags(0x00200000);
    }
    let mut file = flags
        .open(&path)
        .map_err(|e| format!("Cannot open {role}: {e}"))?;
    let after = file.metadata().map_err(|e| e.to_string())?;
    if !after.file_type().is_file() {
        return Err(format!("{role} must be a regular GGUF file."));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return Err(format!("{role} changed while opening it; retry import."));
        }
    }
    let mut magic = [0; 4];
    file.read_exact(&mut magic)
        .map_err(|e| format!("Cannot read {role} GGUF header: {e}"))?;
    if &magic != b"GGUF" {
        return Err(format!(
            "{role} is not GGUF. Import only registers existing GGUF weights; it does not convert models."
        ));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "lma-model-import-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(fs::canonicalize(root).unwrap())
        }
        fn model(&self, name: &str) -> PathBuf {
            let path = self.0.join("external").join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, b"GGUFfixture").unwrap();
            path
        }
        fn options(&self, model: &Path) -> LoadOptions {
            LoadOptions {
                model: model.to_str().unwrap().into(),
                ollama: false,
                ollama_dir: None,
                projector: None,
                gpu: "auto".into(),
                context: 4096,
            }
        }
        fn import(&self, options: LoadOptions, name: Option<&str>) -> Result<ImportResult, String> {
            import_into(
                &self.0.join("config"),
                &self.0.join("registry"),
                options,
                name.map(str::to_owned),
            )
        }
        fn entries(&self) -> std::collections::BTreeMap<String, registry::Entry> {
            registry::read(&self.0.join("registry")).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn external_file_is_persistent_idempotent_and_never_copied_or_loaded() {
        let fixture = Fixture::new();
        let model = fixture.model("My model (Q4).gguf");
        let options = fixture.options(&model);
        let imported = fixture.import(options.clone(), None).unwrap();
        assert_eq!(imported.model, "My-model-Q4:latest");
        assert_eq!(imported.path, model.to_str().unwrap());
        assert_eq!(fixture.import(options, None).unwrap(), imported);
        assert_eq!(fixture.entries()[&imported.model].path, model);
        assert_eq!(fs::read_dir(fixture.0.join("registry")).unwrap().count(), 2);
        assert_eq!(fs::read(&model).unwrap(), b"GGUFfixture");
        assert!(!fixture.0.join("config").exists());
        let catalog = models::discover(models::DiscoveryOptions {
            local_dirs: vec![fixture.0.join("registry")],
            ollama_dir: Some(fixture.0.join("missing-store")),
        });
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].name, imported.model);
        assert_eq!(catalog.models[0].model, imported.path);
        assert_eq!(
            serde_json::to_value(imported)
                .unwrap()
                .as_object()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn names_alias_reuse_and_projectors_are_preserved_without_overwrites() {
        let fixture = Fixture::new();
        let model = fixture.model("weights.gguf");
        let projector = fixture.model("mmproj.gguf");
        let mut options = fixture.options(&model);
        options.projector = Some(projector.clone());
        let first = fixture
            .import(options.clone(), Some("team/chosen:q4"))
            .unwrap();
        assert_eq!(first.model, "team/chosen:q4");
        assert_eq!(
            fixture.entries()[&first.model].projector,
            Some(projector.clone())
        );
        assert_eq!(
            fixture.import(fixture.options(&model), None).unwrap(),
            first
        );
        assert_eq!(
            fixture
                .import(fixture.options(&model), Some("team/chosen:q4"))
                .unwrap(),
            first
        );
        let mut alias = fixture.options(&model);
        alias.model = first.model.clone();
        assert_eq!(fixture.import(alias, None).unwrap(), first);
        let other_projector = fixture.model("other-projector.gguf");
        options.projector = Some(other_projector);
        assert!(fixture.import(options, Some("team/chosen:q4")).is_err());
        let other = fixture.model("other.gguf");
        assert!(
            fixture
                .import(fixture.options(&other), Some("team/chosen:q4"))
                .is_err()
        );
        assert_eq!(fixture.entries()[&first.model].projector, Some(projector));
        assert_eq!(fixture.entries()[&first.model].path, model);
        for name in ["", "../bad", "/absolute", "bad name", "ollama:bad", "name:"] {
            assert!(
                fixture.import(fixture.options(&other), Some(name)).is_err(),
                "{name}"
            );
        }
        assert_eq!(fixture.entries().len(), 1);
        assert_eq!(
            default_label(Path::new("/fixture/模型.gguf")).unwrap(),
            "model:latest"
        );
        assert_eq!(
            default_label(Path::new("/fixture/.a b_.gguf")).unwrap(),
            "a-b:latest"
        );
    }

    #[test]
    fn default_name_collisions_bad_models_and_bad_projectors_do_not_publish() {
        let fixture = Fixture::new();
        let first = fixture.model("one/same.gguf");
        let second = fixture.model("two/same.gguf");
        fixture.import(fixture.options(&first), None).unwrap();
        assert!(fixture.import(fixture.options(&second), None).is_err());
        let invalid = fixture.model("invalid.gguf");
        fs::write(&invalid, b"not GGUF").unwrap();
        assert!(fixture.import(fixture.options(&invalid), None).is_err());
        let mut options = fixture.options(&second);
        options.projector = Some(invalid);
        assert!(fixture.import(options, Some("other")).is_err());
        assert!(fixture.import(fixture.options(&fixture.0), None).is_err());
        assert!(
            fixture
                .import(fixture.options(&fixture.0.join("missing.gguf")), None)
                .is_err()
        );
        assert_eq!(fixture.entries().len(), 1);
        assert_eq!(fs::read(&first).unwrap(), b"GGUFfixture");
        assert_eq!(fs::read(&second).unwrap(), b"GGUFfixture");
    }

    #[cfg(unix)]
    #[test]
    fn source_symlinks_resolve_to_weights_but_registry_symlinks_are_not_replaced() {
        let fixture = Fixture::new();
        let weights = fixture.model("real.gguf");
        let link = fixture.0.join("linked.gguf");
        std::os::unix::fs::symlink(&weights, &link).unwrap();
        let imported = fixture.import(fixture.options(&link), None).unwrap();
        assert_eq!(imported.path, weights.to_str().unwrap());
        assert_eq!(imported.model, "real:latest");
        let root = fixture.0.join("registry");
        let saved = root.join(".registry-v1.json");
        let outside = fixture.0.join("registry-target.json");
        fs::rename(&saved, &outside).unwrap();
        let bytes = fs::read(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, &saved).unwrap();
        assert!(
            fixture
                .import(fixture.options(&weights), Some("another"))
                .is_err()
        );
        assert_eq!(fs::read(&outside).unwrap(), bytes);
        assert!(
            fs::symlink_metadata(&saved)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(&weights).unwrap(), b"GGUFfixture");
    }

    fn ollama(fixture: &Fixture) -> (LoadOptions, PathBuf, PathBuf) {
        let store = fixture.0.join("custom-ollama");
        let manifest = store.join("manifests/registry.ollama.ai/library/example/q4");
        fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        fs::create_dir_all(store.join("blobs")).unwrap();
        let weights = store.join(format!("blobs/sha256-{}", "a".repeat(64)));
        let projector = store.join(format!("blobs/sha256-{}", "b".repeat(64)));
        fs::write(&weights, b"GGUFfixture").unwrap();
        fs::write(&projector, b"GGUFprojector").unwrap();
        fs::write(manifest, serde_json::to_vec(&serde_json::json!({"schemaVersion":2,"layers":[
            {"mediaType":"application/vnd.ollama.image.model","digest":format!("sha256:{}", "a".repeat(64))},
            {"mediaType":"application/vnd.ollama.image.projector","digest":format!("sha256:{}", "b".repeat(64))}
        ]})).unwrap()).unwrap();
        (
            LoadOptions {
                model: "example:q4".into(),
                ollama: true,
                ollama_dir: Some(store),
                projector: None,
                gpu: "auto".into(),
                context: 4096,
            },
            weights,
            projector,
        )
    }

    #[test]
    fn custom_ollama_store_is_remembered_with_existing_blobs_and_projector() {
        let fixture = Fixture::new();
        let (options, weights, projector) = ollama(&fixture);
        let config = fixture.0.join("config");
        let mut settings = app_settings::Settings::default();
        settings.auto_restore = false;
        app_settings::save(&config, settings).unwrap();
        let imported = fixture.import(options.clone(), None).unwrap();
        assert_eq!(imported.model, "example:q4");
        assert_eq!(imported.path, weights.to_str().unwrap());
        assert_eq!(
            fixture.entries()[&imported.model].projector,
            Some(projector)
        );
        let settings = app_settings::load(&config).unwrap();
        assert!(!settings.auto_restore);
        assert_eq!(
            settings.ollama_directory,
            Some(options.ollama_dir.unwrap().to_str().unwrap().into())
        );
        assert_eq!(fs::read_dir(weights.parent().unwrap()).unwrap().count(), 2);
        assert_eq!(fs::read_dir(fixture.0.join("registry")).unwrap().count(), 2);
    }

    #[test]
    fn failed_alias_publication_does_not_create_or_change_settings() {
        for existing_settings in [false, true] {
            let fixture = Fixture::new();
            let (options, weights, projector) = ollama(&fixture);
            let config = fixture.0.join("config");
            let settings_path = config.join("app-settings.json");
            if existing_settings {
                app_settings::save(
                    &config,
                    app_settings::Settings {
                        auto_restore: false,
                        gpu: "cpu".into(),
                        context: 128,
                        ollama_directory: Some(
                            fixture.0.join("previous-store").to_str().unwrap().into(),
                        ),
                        model_directories: vec![fixture.0.join("local").to_str().unwrap().into()],
                        ..Default::default()
                    },
                )
                .unwrap();
            }
            let previous_settings = fs::read(&settings_path).ok();
            let root = fixture.0.join("registry");
            fs::create_dir(&root).unwrap();
            let mut entries = std::collections::BTreeMap::from([(
                "existing:latest".to_string(),
                registry::Entry {
                    path: fixture.0.join("padding"),
                    projector: None,
                },
            )]);
            // A valid, near-limit registry deterministically rejects the new
            // serialized entry; no permissions assumptions or real weights.
            let remaining = 1024 * 1024 - 1 - serde_json::to_vec_pretty(&entries).unwrap().len();
            let previous = entries.get_mut("existing:latest").unwrap();
            previous.path = PathBuf::from(format!(
                "{}{}",
                previous.path.display(),
                "x".repeat(remaining)
            ));
            let bytes = serde_json::to_vec_pretty(&entries).unwrap();
            assert_eq!(bytes.len(), 1024 * 1024 - 1);
            let registry_path = root.join(".registry-v1.json");
            fs::write(&registry_path, &bytes).unwrap();
            let error = fixture.import(options, None).unwrap_err();
            assert!(error.contains("exceeds 1 MiB"), "{error}");
            assert_eq!(fs::read(&registry_path).unwrap(), bytes);
            assert_eq!(fs::read(&settings_path).ok(), previous_settings);
            assert_eq!(config.exists(), existing_settings);
            assert_eq!(fs::read(&weights).unwrap(), b"GGUFfixture");
            assert_eq!(fs::read(&projector).unwrap(), b"GGUFprojector");
        }
    }

    #[test]
    fn settings_save_failure_rolls_back_only_a_new_alias_and_preserves_prior_settings() {
        for reuse_alias in [false, true] {
            let fixture = Fixture::new();
            let (options, weights, projector) = ollama(&fixture);
            let config = fixture.0.join("config");
            let mut directory = fixture.0.to_str().unwrap().to_owned();
            directory.push('/');
            directory.push_str(&"x".repeat(4096 - directory.len()));
            let mut settings = app_settings::Settings {
                auto_restore: false,
                gpu: "cpu".into(),
                model_directories: vec![directory; 16],
                ..Default::default()
            };
            let excess = serde_json::to_vec_pretty(&settings).unwrap().len() - (64 * 1024 - 1);
            let last = settings.model_directories.last_mut().unwrap();
            last.truncate(last.len() - excess);
            assert_eq!(
                serde_json::to_vec_pretty(&settings).unwrap().len(),
                64 * 1024 - 1
            );
            app_settings::save(&config, settings.clone()).unwrap();
            let settings_path = config.join("app-settings.json");
            let settings_bytes = fs::read(&settings_path).unwrap();
            let root = fixture.0.join("registry");
            let unrelated = fixture.model("unrelated.gguf");
            registry::put(
                &root,
                "unrelated",
                registry::Entry {
                    path: unrelated.clone(),
                    projector: None,
                },
            )
            .unwrap();
            if reuse_alias {
                registry::put(
                    &root,
                    "example:q4",
                    registry::Entry {
                        path: weights.clone(),
                        projector: Some(projector.clone()),
                    },
                )
                .unwrap();
            }
            let registry_path = root.join(".registry-v1.json");
            let registry_bytes = fs::read(&registry_path).unwrap();
            let error = fixture.import(options, None).unwrap_err();
            assert!(error.contains("exceed 64 KiB"), "{error}");
            assert_eq!(fs::read(&settings_path).unwrap(), settings_bytes);
            assert_eq!(app_settings::load(&config).unwrap(), settings);
            assert_eq!(fs::read(&registry_path).unwrap(), registry_bytes);
            assert_eq!(fixture.entries().contains_key("example:q4"), reuse_alias);
            assert_eq!(fixture.entries()["unrelated:latest"].path, unrelated);
            assert_eq!(fs::read(&weights).unwrap(), b"GGUFfixture");
            assert_eq!(fs::read(&projector).unwrap(), b"GGUFprojector");
        }
    }

    #[test]
    fn local_alias_inside_an_ollama_store_does_not_change_store_settings() {
        let fixture = Fixture::new();
        let (mut options, weights, projector) = ollama(&fixture);
        let root = fixture.0.join("registry");
        registry::put(
            &root,
            "local-alias",
            registry::Entry {
                path: weights.clone(),
                projector: Some(projector.clone()),
            },
        )
        .unwrap();
        options.model = "local-alias".into();
        options.ollama = false;
        let config = fixture.0.join("config");
        let settings = app_settings::Settings {
            auto_restore: false,
            ollama_directory: Some(fixture.0.join("previous-store").to_str().unwrap().into()),
            ..Default::default()
        };
        app_settings::save(&config, settings.clone()).unwrap();
        let settings_path = config.join("app-settings.json");
        let bytes = fs::read(&settings_path).unwrap();
        let imported = fixture.import(options, None).unwrap();
        assert_eq!(imported.model, "local-alias:latest");
        assert_eq!(imported.path, weights.to_str().unwrap());
        assert_eq!(
            fixture.entries()[&imported.model].projector,
            Some(projector)
        );
        assert_eq!(app_settings::load(&config).unwrap(), settings);
        assert_eq!(fs::read(settings_path).unwrap(), bytes);
    }

    #[test]
    fn corrupt_settings_or_registry_are_not_overwritten_and_do_not_publish() {
        let fixture = Fixture::new();
        let (options, _, _) = ollama(&fixture);
        let config = fixture.0.join("config");
        fs::create_dir(&config).unwrap();
        let settings_path = config.join("app-settings.json");
        fs::write(&settings_path, b"{corrupt").unwrap();
        assert!(fixture.import(options.clone(), None).is_err());
        assert_eq!(fs::read(&settings_path).unwrap(), b"{corrupt");
        assert!(!fixture.0.join("registry").exists());
        #[cfg(unix)]
        {
            fs::remove_file(&settings_path).unwrap();
            let target = fixture.0.join("settings-target");
            fs::write(&target, b"{}").unwrap();
            std::os::unix::fs::symlink(&target, &settings_path).unwrap();
            assert!(fixture.import(options, None).is_err());
            assert_eq!(fs::read(&target).unwrap(), b"{}");
            assert!(
                fs::symlink_metadata(&settings_path)
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
            assert!(!fixture.0.join("registry").exists());
        }
        let root = fixture.0.join("registry");
        fs::create_dir(&root).unwrap();
        let registry_path = root.join(".registry-v1.json");
        fs::write(&registry_path, b"{corrupt").unwrap();
        let model = fixture.model("weights.gguf");
        assert!(fixture.import(fixture.options(&model), None).is_err());
        assert_eq!(fs::read(registry_path).unwrap(), b"{corrupt");
    }
}
