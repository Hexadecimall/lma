//! Model selection only. Chats, prompts, images, and API keys are never persisted.
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::state::LoadOptions;
use serde::{Deserialize, Serialize};

const FILE_NAME: &str = "model-preferences.json";
const MAX_BYTES: u64 = 64 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Preferences {
    version: u32,
    last_model: LoadOptions,
}

pub fn load(config_dir: &Path) -> Result<Option<LoadOptions>, String> {
    let path = lma::models::storage::read_path(config_dir, FILE_NAME)?;
    match fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("Cannot read saved model selection: {e}")),
        Ok(meta) if !meta.file_type().is_file() => {
            return Err("Saved model selection must be a regular, non-symlink file.".into());
        }
        Ok(_) => {}
    }
    let mut bytes = Vec::new();
    File::open(&path)
        .and_then(|f| f.take(MAX_BYTES + 1).read_to_end(&mut bytes))
        .map_err(|e| format!("Cannot read saved model selection: {e}"))?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("Saved model selection exceeds 64 KiB.".into());
    }
    let saved: Preferences = serde_json::from_slice(&bytes)
        .map_err(|e| format!("Invalid saved model selection: {e}"))?;
    if saved.version != 1 {
        return Err("Unsupported saved model selection version.".into());
    }
    saved.last_model.clone().config()?;
    if (!saved.last_model.ollama
        && !Path::new(&saved.last_model.model).is_absolute()
        && qualified_local_name(&saved.last_model.model).is_none())
        || saved
            .last_model
            .ollama_dir
            .as_ref()
            .is_some_and(|p| !p.is_absolute())
        || saved
            .last_model
            .projector
            .as_ref()
            .is_some_and(|p| !p.is_absolute())
    {
        return Err("Saved model paths must be absolute or a qualified lma model selector.".into());
    }
    Ok(Some(saved.last_model))
}

fn qualified_local_name(selector: &str) -> Option<String> {
    lma::models::registry::name(selector.strip_prefix("lma:")?).ok()
}

fn normalize(options: LoadOptions) -> Result<LoadOptions, String> {
    normalize_with(options, lma::models::resolve)
}

fn normalize_with(
    mut options: LoadOptions,
    resolve: impl FnOnce(&Path, Option<&Path>) -> Result<lma::models::ResolvedModel, String>,
) -> Result<LoadOptions, String> {
    let config = options.clone().config()?;
    let input = &config.model;
    let resolved = resolve(input, config.ollama_dir.as_deref())?;
    // Names also identify local aliases/import receipts, not just Ollama tags.
    // Preserve the selected source and the resolver's existing-local-file priority.
    let ollama = !input.is_file()
        && input
            .to_str()
            .is_some_and(|text| text.starts_with("ollama:"));
    if ollama {
        let name = resolved
            .name
            .ok_or("The selected Ollama model has no tag.")?;
        let root = config
            .ollama_dir
            .as_deref()
            .or_else(|| resolved.model.parent().and_then(Path::parent))
            .ok_or("Cannot determine the selected Ollama store.")?;
        options.ollama_dir = Some(fs::canonicalize(root).map_err(|e| e.to_string())?);
        options.model = name;
        options.ollama = true;
    } else {
        let path = fs::canonicalize(&resolved.model).map_err(|e| e.to_string())?;
        // Only an explicitly qualified selection keeps alias identity. A name
        // on an import receipt must not reclassify an ordinary local filepath.
        options.model = match input
            .to_str()
            .and_then(qualified_local_name)
            .filter(|_| !input.is_file())
        {
            Some(name) => format!("lma:{name}"),
            None => path
                .to_str()
                .ok_or("The selected model path is not UTF-8.")?
                .into(),
        };
        options.ollama = false;
        options.ollama_dir = None;
    }
    options.projector = options
        .projector
        .or(resolved.projector)
        .map(fs::canonicalize)
        .transpose()
        .map_err(|e| e.to_string())?;
    Ok(options)
}

/// Call only after a successful load. A failure must not invalidate that session.
pub fn save(config_dir: &Path, options: LoadOptions) -> Result<(), String> {
    let last_model = normalize(options)?;
    // Do not overwrite an unreadable, unrecognized, or redirected preference file.
    load(config_dir)?;
    fs::create_dir_all(config_dir).map_err(|e| e.to_string())?;
    let bytes = serde_json::to_vec_pretty(&Preferences {
        version: 1,
        last_model,
    })
    .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("Model selection exceeds 64 KiB.".into());
    }
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temporary = config_dir.join(format!(
        ".model-preferences-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut flags = OpenOptions::new();
    flags.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        flags.mode(0o600);
    }
    let mut file = flags.open(&temporary).map_err(|e| e.to_string())?;
    let result = (|| {
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        fs::rename(&temporary, config_dir.join(FILE_NAME)).map_err(|e| e.to_string())
    })();
    if temporary.is_file() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "lma-model-prefs-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(fs::canonicalize(root).unwrap())
        }
        fn options(&self) -> LoadOptions {
            let model = self.0.join("model.gguf");
            fs::write(&model, b"GGUFfixture").unwrap();
            LoadOptions {
                model: model.to_str().unwrap().into(),
                ollama: false,
                ollama_dir: None,
                projector: None,
                gpu: "1".into(),
                context: 256,
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn remembers_only_model_options_and_replaces_atomically() {
        let fixture = Fixture::new();
        let config = fixture.0.join("config");
        assert_eq!(load(&config).unwrap(), None);
        assert!(!config.exists());
        let options = fixture.options();
        save(&config, options.clone()).unwrap();
        assert_eq!(load(&config).unwrap(), Some(options.clone()));
        let mut changed = options;
        changed.gpu = "cpu".into();
        changed.context = 1024;
        save(&config, changed.clone()).unwrap();
        assert_eq!(load(&config).unwrap(), Some(changed));
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(config.join(FILE_NAME)).unwrap()).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 2);
        assert_eq!(value["last_model"].as_object().unwrap().len(), 6);
        assert_eq!(fs::read_dir(&config).unwrap().count(), 1);
    }

    #[test]
    fn stale_model_is_returned_without_loading_or_dropping_the_selection() {
        let fixture = Fixture::new();
        let options = fixture.options();
        save(&fixture.0, options.clone()).unwrap();
        fs::remove_file(&options.model).unwrap();
        assert_eq!(load(&fixture.0).unwrap(), Some(options));
    }

    #[test]
    fn corrupt_oversized_unknown_and_relative_preferences_are_not_overwritten() {
        let fixture = Fixture::new();
        let options = fixture.options();
        let path = fixture.0.join(FILE_NAME);
        for bytes in [b"{broken".to_vec(), vec![b' '; MAX_BYTES as usize + 1],
            serde_json::to_vec(&serde_json::json!({"version": 2,"last_model": options})).unwrap(),
            serde_json::to_vec(&serde_json::json!({"version": 1,"last_model": {"model":"relative.gguf","ollama":false,"ollamaDir":null,"projector":null,"gpu":"cpu","context":256}})).unwrap(),
        ] {
            fs::write(&path, &bytes).unwrap();
            assert!(load(&fixture.0).is_err());
            assert!(save(&fixture.0, options.clone()).is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlink_preferences_are_never_followed_or_replaced() {
        let fixture = Fixture::new();
        let options = fixture.options();
        let target = fixture.0.join("outside.json");
        fs::write(&target, b"keep this").unwrap();
        std::os::unix::fs::symlink(&target, fixture.0.join(FILE_NAME)).unwrap();
        assert!(load(&fixture.0).is_err());
        assert!(save(&fixture.0, options).is_err());
        assert_eq!(fs::read(target).unwrap(), b"keep this");
    }

    #[test]
    fn named_local_alias_is_saved_as_a_local_path_and_restores_without_ollama() {
        let fixture = Fixture::new();
        let config = fixture.0.join("config");
        let mut options = fixture.options();
        let model = PathBuf::from(&options.model);
        let projector = fixture.0.join("mmproj.gguf");
        fs::write(&projector, b"GGUFprojector").unwrap();
        options.projector = Some(projector.clone());
        // A stale store selection must not turn a named local model into Ollama.
        options.ollama_dir = Some(fixture.0.join("missing-store"));
        let fingerprint = "a".repeat(64);
        fs::write(
            fixture.0.join(format!(".lma-import-v1-{fingerprint}.json")),
            serde_json::to_vec(&serde_json::json!({
                "version":1,"source_root":fixture.0.join("old-store"),
                "model":"team/local:q4","fingerprint":fingerprint,
                "output":model,"output_bytes":11
            }))
            .unwrap(),
        )
        .unwrap();
        let registry = fixture.0.join("registry");
        lma::models::registry::put(
            &registry,
            "team/local:q4",
            lma::models::registry::Entry {
                path: model.clone(),
                projector: Some(projector.clone()),
            },
        )
        .unwrap();
        let catalog = lma::models::discover(lma::models::DiscoveryOptions {
            local_dirs: vec![registry],
            ollama_dir: options.ollama_dir.clone(),
        });
        let selected = catalog
            .models
            .iter()
            .find(|model| model.name == "team/local:q4")
            .unwrap();
        assert!(!selected.ollama);
        options.model = selected.model.clone();
        options.projector = selected.projector.clone();
        let resolved = lma::models::resolve(&model, options.ollama_dir.as_deref()).unwrap();
        assert_eq!(resolved.name.as_deref(), Some("team/local:q4"));
        save(&config, options.clone()).unwrap();
        let saved = load(&config).unwrap().unwrap();
        options.ollama_dir = None;
        assert_eq!(saved, options);
        assert!(!saved.ollama);
        assert_eq!(saved.projector, Some(projector));
        let restored = lma::models::resolve(&saved.clone().config().unwrap().model, None).unwrap();
        assert_eq!(restored.model, model);
        save(&config, saved.clone()).unwrap();
        assert_eq!(load(&config).unwrap(), Some(saved));
        assert_eq!(fs::read(&model).unwrap(), b"GGUFfixture");
        assert!(!fixture.0.join("missing-store").exists());
    }

    #[test]
    fn qualified_local_profiles_normalize_and_restore_without_collapsing_shared_weights() {
        let fixture = Fixture::new();
        let options = fixture.options();
        let model = PathBuf::from(&options.model);
        let registry = fixture.0.join("registry");
        for (label, system) in [
            ("team/first:latest", "First profile"),
            ("team/second:latest", "Second profile"),
        ] {
            lma::models::registry::put_definition(
                &registry,
                label,
                lma::models::registry::Entry {
                    path: model.clone(),
                    projector: None,
                },
                lma::models::modelfile::Definition {
                    from: model.to_string_lossy().into_owned(),
                    system: Some(system.into()),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        lma::models::registry::put(
            &registry,
            "plain:latest",
            lma::models::registry::Entry {
                path: model.clone(),
                projector: None,
            },
        )
        .unwrap();
        let resolve = |input: &Path, _: Option<&Path>| {
            let name = qualified_local_name(input.to_str().unwrap()).unwrap();
            let entry = lma::models::registry::read(&registry)?
                .remove(&name)
                .ok_or("fixture alias missing")?;
            Ok(lma::models::ResolvedModel {
                model: entry.path,
                projector: entry.projector,
                registry_name: Some(name.clone()),
                name: Some(name),
            })
        };
        let mut saved = Vec::new();
        for label in ["team/first:latest", "team/second:latest", "plain:latest"] {
            let selected = LoadOptions {
                model: format!("lma:{label}"),
                ollama_dir: Some(fixture.0.join("missing-store")),
                ..options.clone()
            };
            let normalized = normalize_with(selected, resolve).unwrap();
            assert_eq!(normalized.model, format!("lma:{label}"));
            assert!(!normalized.ollama);
            assert!(normalized.ollama_dir.is_none());
            assert_eq!(
                normalized.clone().config().unwrap().model,
                PathBuf::from(&normalized.model)
            );
            let directory = fixture.0.join(format!("preferences-{}", saved.len()));
            fs::create_dir(&directory).unwrap();
            fs::write(
                directory.join(FILE_NAME),
                serde_json::to_vec(&Preferences {
                    version: 1,
                    last_model: normalized.clone(),
                })
                .unwrap(),
            )
            .unwrap();
            assert_eq!(load(&directory).unwrap(), Some(normalized.clone()));
            saved.push(normalized);
        }
        assert_ne!(saved[0].model, saved[1].model);
        assert_eq!(
            lma::models::registry::definition(&registry, "team/first:latest")
                .unwrap()
                .unwrap()
                .system
                .as_deref(),
            Some("First profile")
        );
        assert_eq!(
            lma::models::registry::definition(&registry, "team/second:latest")
                .unwrap()
                .unwrap()
                .system
                .as_deref(),
            Some("Second profile")
        );
        assert!(
            lma::models::registry::definition(&registry, "plain:latest")
                .unwrap()
                .is_none()
        );
        assert_eq!(fs::read(model).unwrap(), b"GGUFfixture");
        assert!(!fixture.0.join("missing-store").exists());
    }

    #[test]
    fn unqualified_alias_still_normalizes_to_a_local_weight_path() {
        let fixture = Fixture::new();
        let options = fixture.options();
        let selected = LoadOptions {
            model: "plain:latest".into(),
            ollama_dir: Some(fixture.0.join("missing-store")),
            ..options.clone()
        };
        let normalized = normalize_with(selected, |_, _| {
            Ok(lma::models::ResolvedModel {
                model: PathBuf::from(&options.model),
                projector: None,
                name: Some("plain:latest".into()),
                registry_name: Some("plain:latest".into()),
            })
        })
        .unwrap();
        assert_eq!(normalized.model, options.model);
        assert!(!normalized.ollama);
        assert!(normalized.ollama_dir.is_none());
    }

    #[test]
    fn saved_local_selectors_require_a_valid_explicit_lma_qualification() {
        let fixture = Fixture::new();
        let options = fixture.options();
        for selector in ["lma:team/profile:latest", "lma:missing-profile:latest"] {
            let selected = LoadOptions {
                model: selector.into(),
                ..options.clone()
            };
            fs::write(
                fixture.0.join(FILE_NAME),
                serde_json::to_vec(&Preferences {
                    version: 1,
                    last_model: selected.clone(),
                })
                .unwrap(),
            )
            .unwrap();
            assert_eq!(load(&fixture.0).unwrap(), Some(selected));
        }
        for selector in [
            "relative.gguf",
            "team/profile:latest",
            "lma:",
            "lma:../escape",
            "lma:/absolute",
            "lma:has space",
            "lma:ollama:profile",
        ] {
            fs::write(
                fixture.0.join(FILE_NAME),
                serde_json::to_vec(&Preferences {
                    version: 1,
                    last_model: LoadOptions {
                        model: selector.into(),
                        ..options.clone()
                    },
                })
                .unwrap(),
            )
            .unwrap();
            assert!(
                load(&fixture.0).is_err(),
                "accepted invalid saved selector: {selector}"
            );
        }
    }

    #[test]
    fn saved_ollama_tag_keeps_its_actual_store_and_projector() {
        let fixture = Fixture::new();
        let store = fixture.0.join("store");
        let manifests = store.join("manifests/hf.co/owner/model");
        fs::create_dir_all(&manifests).unwrap();
        fs::create_dir_all(store.join("blobs")).unwrap();
        let model_hash = "a".repeat(64);
        let projector_hash = "b".repeat(64);
        fs::write(
            store.join("blobs").join(format!("sha256-{model_hash}")),
            b"GGUFfixture",
        )
        .unwrap();
        let projector = store.join("blobs").join(format!("sha256-{projector_hash}"));
        fs::write(&projector, b"GGUFfixture").unwrap();
        fs::write(manifests.join("Q4_K_M"), serde_json::to_vec(&serde_json::json!({"schemaVersion":2,"layers":[
            {"mediaType":"application/vnd.ollama.image.model","digest":format!("sha256:{model_hash}")},
            {"mediaType":"application/vnd.ollama.image.projector","digest":format!("sha256:{projector_hash}")}
        ]})).unwrap()).unwrap();
        let options = LoadOptions {
            model: "hf.co/owner/model:Q4_K_M".into(),
            ollama: true,
            ollama_dir: Some(store.clone()),
            projector: None,
            gpu: "1".into(),
            context: 256,
        };
        save(&fixture.0, options).unwrap();
        let saved = load(&fixture.0).unwrap().unwrap();
        assert_eq!(saved.model, "hf.co/owner/model:Q4_K_M");
        assert_eq!(saved.ollama_dir, Some(store));
        assert_eq!(saved.projector, Some(projector));
        assert_eq!(saved.gpu, "1");
    }
}
