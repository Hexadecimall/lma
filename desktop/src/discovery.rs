//! Desktop-owned search locations; shared core owns format/store discovery.
use crate::{app_settings, preferences, state::LoadOptions};
use lma::models::{self, DiscoveryOptions};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryResult {
    pub models: Vec<models::DiscoveredModel>,
    pub warnings: Vec<String>,
    pub last_model: Option<LoadOptions>,
}

pub fn discover(
    config_dir: &Path,
    data_dir: PathBuf,
    cwd: Option<PathBuf>,
    built_source: Option<PathBuf>,
) -> DiscoveryResult {
    discover_with_override(
        config_dir,
        data_dir,
        cwd,
        built_source,
        std::env::var_os("LMA_MODELS")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from),
    )
}

/// Host-owned boundaries for destructive operations, never supplied by IPC.
#[derive(Clone)]
pub(crate) struct DeletionLocations {
    pub local_dirs: Vec<PathBuf>,
    pub lm_dirs: Vec<PathBuf>,
    pub ollama_dirs: Vec<Option<PathBuf>>,
}

impl DeletionLocations {
    pub(crate) fn catalog(&self) -> models::ModelCatalog {
        let mut stores = self.ollama_dirs.iter();
        let mut catalog = models::discover(DiscoveryOptions {
            local_dirs: self.local_dirs.clone(),
            ollama_dir: stores.next().cloned().flatten(),
        });
        for root in stores {
            let extra = models::discover(DiscoveryOptions {
                local_dirs: Vec::new(),
                ollama_dir: root.clone(),
            });
            catalog.models.extend(extra.models);
            catalog.warnings.extend(extra.warnings);
        }
        finish_catalog(&mut catalog, &self.lm_dirs);
        catalog
    }
}

pub(crate) fn deletion_locations(
    config_dir: &Path,
    data_dir: PathBuf,
    cwd: Option<PathBuf>,
    built_source: Option<PathBuf>,
) -> Result<DeletionLocations, String> {
    let settings = app_settings::load(config_dir)?;
    let mut warnings = Vec::new();
    let registered = registered_source(config_dir, &mut warnings);
    if !warnings.is_empty() {
        return Err(warnings.join(" "));
    }
    let mut roots = local_roots(
        data_dir,
        cwd,
        registered,
        built_source,
        std::env::var_os("LMA_MODELS")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from),
    );
    roots.extend(models::discovery::default_local_dirs());
    let (local_dirs, lm_dirs) =
        configured_locations(&settings, roots, models::discovery::lm_studio_dirs());
    let mut ollama_dirs = vec![None];
    if let Some(root) = settings.ollama_directory.as_deref() {
        ollama_dirs.push(Some(app_settings::path(root)?));
    }
    // A corrupt saved choice is not a trusted location. Report its cleanup error
    // after deletion rather than preventing otherwise safe deletion.
    if let Ok(Some(saved)) = preferences::load(config_dir)
        && let Some(root) = saved.ollama_dir
    {
        ollama_dirs.push(Some(root));
    }
    Ok(DeletionLocations {
        local_dirs,
        lm_dirs,
        ollama_dirs,
    })
}

fn discover_with_override(
    config_dir: &Path,
    data_dir: PathBuf,
    cwd: Option<PathBuf>,
    built_source: Option<PathBuf>,
    custom_models: Option<PathBuf>,
) -> DiscoveryResult {
    let mut warnings = Vec::new();
    let last_model = match preferences::load(config_dir) {
        Ok(saved) => saved,
        Err(error) => {
            warnings.push(error);
            None
        }
    };
    let settings = match app_settings::load(config_dir) {
        Ok(settings) => settings,
        Err(error) => {
            warnings.push(format!("Custom model locations were not applied: {error}"));
            app_settings::Settings::default()
        }
    };
    let registered = registered_source(config_dir, &mut warnings);
    let mut roots = local_roots(data_dir, cwd, registered, built_source, custom_models);
    roots.extend(models::discovery::default_local_dirs());
    let (roots, lm_roots) =
        configured_locations(&settings, roots, models::discovery::lm_studio_dirs());
    let mut catalog = models::discover(DiscoveryOptions {
        local_dirs: roots,
        ollama_dir: None,
    });
    // Preserve a custom selected store even when Finder does not inherit the
    // terminal's OLLAMA_MODELS environment on the next application launch.
    let mut seen_stores = HashSet::new();
    for root in settings
        .ollama_directory
        .as_deref()
        .and_then(|p| app_settings::path(p).ok())
        .into_iter()
        .chain(
            last_model
                .as_ref()
                .and_then(|model| model.ollama_dir.clone()),
        )
    {
        if !seen_stores.insert(fs::canonicalize(&root).unwrap_or_else(|_| root.clone())) {
            continue;
        }
        let extra = models::discover(DiscoveryOptions {
            local_dirs: Vec::new(),
            ollama_dir: Some(root),
        });
        catalog.models.extend(extra.models);
        catalog.warnings.extend(extra.warnings);
    }
    finish_catalog(&mut catalog, &lm_roots);
    warnings.extend(catalog.warnings);
    let mut seen = HashSet::new();
    warnings.retain(|warning| seen.insert(warning.clone()));
    warnings.truncate(64);
    DiscoveryResult {
        models: catalog.models,
        warnings,
        last_model,
    }
}

fn finish_catalog(catalog: &mut models::ModelCatalog, lm_roots: &[PathBuf]) {
    label_sources(catalog, lm_roots);
    let mut seen = HashSet::new();
    catalog.models.retain(|model| seen.insert(model.id.clone()));
    catalog.models.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then(a.id.cmp(&b.id))
    });
    // Shared discovery uses this prefix only for duplicate-name warnings, emitted
    // before desktop relabeling/merging. Match the prefix even if a long name caused
    // the shared warning to be truncated before its "multiple sources" suffix.
    catalog
        .warnings
        .retain(|warning| !warning.starts_with("Model '"));
    let mut names = BTreeMap::<_, (BTreeSet<_>, BTreeSet<_>)>::new();
    for model in &catalog.models {
        let (sources, stores) = names.entry(model.name.to_lowercase()).or_default();
        sources.insert(model.source.as_str());
        if model.ollama
            && let Some(root) = &model.ollama_dir
        {
            stores.insert(root);
        }
    }
    for (name, (sources, stores)) in names {
        if sources.len() > 1 {
            catalog.warnings.push(format!("Model '{name}' is available from multiple sources: {}. These may share the same weights; no files were removed.", sources.into_iter().collect::<Vec<_>>().join(", ")));
        }
        if stores.len() > 1 {
            catalog.warnings.push(format!("Model '{name}' is available from multiple Ollama stores: {}. These may share the same weights; no files were removed.", stores.into_iter().map(|root| root.display().to_string()).collect::<Vec<_>>().join(", ")));
        }
    }
    // Match the shared scanner's UTF-8-safe diagnostic length bound. The combined
    // result's warning count/deduplication is applied alongside settings errors.
    for warning in &mut catalog.warnings {
        if warning.len() > 1024 {
            let mut end = 1021;
            while !warning.is_char_boundary(end) {
                end -= 1;
            }
            warning.truncate(end);
            warning.push_str("...");
        }
    }
}

fn configured_locations(
    settings: &app_settings::Settings,
    mut roots: Vec<PathBuf>,
    mut lm_roots: Vec<PathBuf>,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    roots.extend(
        settings
            .model_directories
            .iter()
            .filter_map(|p| app_settings::path(p).ok()),
    );
    lm_roots.extend(
        settings
            .lm_studio_directories
            .iter()
            .filter_map(|p| app_settings::path(p).ok()),
    );
    roots.extend(lm_roots.iter().cloned());
    let mut seen = HashSet::new();
    roots.retain(|root| seen.insert(fs::canonicalize(root).unwrap_or_else(|_| root.clone())));
    (roots, lm_roots)
}

fn label_sources(catalog: &mut models::ModelCatalog, lm_roots: &[PathBuf]) {
    let lm_roots: Vec<_> = lm_roots
        .iter()
        .filter_map(|p| fs::canonicalize(p).ok())
        .collect();
    for model in &mut catalog.models {
        if !model.ollama
            && lm_roots
                .iter()
                .any(|root| Path::new(&model.model).starts_with(root))
        {
            // Source labels must not change the stable IDs used by preferences.
            model.source = "lm-studio".into();
        }
    }
}

fn local_roots(
    data_dir: PathBuf,
    cwd: Option<PathBuf>,
    registered: Option<PathBuf>,
    built_source: Option<PathBuf>,
    custom: Option<PathBuf>,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(custom) = custom {
        roots.push(custom);
    }
    roots.push(data_dir.join("models"));
    for source in [cwd, registered, built_source].into_iter().flatten() {
        roots.push(source.join("models"));
    }
    roots
}

fn registered_source(config_dir: &Path, warnings: &mut Vec<String>) -> Option<PathBuf> {
    let path = match lma::models::storage::read_path(config_dir, "source-install.json") {
        Ok(path) => path,
        Err(error) => {
            warnings.push(format!("Cannot read source model directory: {error}"));
            return None;
        }
    };
    let metadata = match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            warnings.push(format!("Cannot read source model directory: {error}"));
            return None;
        }
        Ok(metadata) => metadata,
    };
    if !metadata.file_type().is_file() || metadata.len() > 64 * 1024 {
        warnings
            .push("Skipping an invalid source-install registration while scanning models.".into());
        return None;
    }
    let mut bytes = Vec::new();
    let result =
        fs::File::open(&path).and_then(|file| file.take(64 * 1024 + 1).read_to_end(&mut bytes));
    if result.is_err() || bytes.len() > 64 * 1024 {
        warnings.push("Cannot read source-install registration while scanning models.".into());
        return None;
    }
    #[derive(Deserialize)]
    struct SourceLocation {
        schema_version: u32,
        source_dir: PathBuf,
    }
    let value = match serde_json::from_slice::<SourceLocation>(&bytes) {
        Ok(value) => value,
        Err(_) => {
            warnings.push("Invalid source-install registration while scanning models.".into());
            return None;
        }
    };
    if value.schema_version != 1
        || !value.source_dir.is_absolute()
        || value
            .source_dir
            .to_str()
            .is_none_or(|p| p.len() > 4096 || p.chars().any(char::is_control))
    {
        warnings.push(
            "Skipping an unsupported source-install registration while scanning models.".into(),
        );
        return None;
    }
    Some(value.source_dir)
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
                "lma-desktop-discovery-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(fs::canonicalize(root).unwrap())
        }
        fn model(&self, relative: &str) {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"GGUFfixture").unwrap();
        }
        fn ollama(&self, relative: &str) -> PathBuf {
            let store = self.0.join(relative);
            let manifest = store.join("manifests/registry.ollama.ai/library/example/latest");
            fs::create_dir_all(manifest.parent().unwrap()).unwrap();
            fs::create_dir_all(store.join("blobs")).unwrap();
            let digest = "a".repeat(64);
            fs::write(store.join(format!("blobs/sha256-{digest}")), b"GGUFfixture").unwrap();
            fs::write(&manifest, serde_json::to_vec(&serde_json::json!({
                "schemaVersion":2, "layers":[{"mediaType":"application/vnd.ollama.image.model", "digest":format!("sha256:{digest}")}]
            })).unwrap()).unwrap();
            store
        }
        fn imported_model(&self, relative: &str, store: &Path, name: &str) {
            self.model(relative);
            let output = self.0.join(relative);
            let fingerprint = "a".repeat(64);
            let receipt = output
                .parent()
                .unwrap()
                .join(format!(".lma-import-v1-{fingerprint}.json"));
            fs::write(
                receipt,
                serde_json::to_vec(&serde_json::json!({
                    "version":1, "source_root":store, "model":name,
                    "fingerprint":fingerprint, "output":output, "output_bytes":11
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn location_policy_covers_app_data_fetch_checkout_and_override() {
        let fixture = Fixture::new();
        fixture.model("data/models/managed.gguf");
        fixture.model("cwd/models/fetched.gguf");
        fixture.model("source/models/checkout.gguf");
        fixture.model("custom/extra.gguf");
        let roots = local_roots(
            fixture.0.join("data"),
            Some(fixture.0.join("cwd")),
            Some(fixture.0.join("source")),
            Some(fixture.0.join("source")),
            Some(fixture.0.join("custom")),
        );
        let found = models::discover(DiscoveryOptions {
            local_dirs: roots,
            ollama_dir: Some(fixture.0.join("missing-ollama")),
        });
        assert_eq!(found.models.len(), 4);
        assert!(found.warnings.is_empty());
    }

    #[test]
    fn configured_locations_reuse_weights_and_preserve_ids_and_preferences() {
        let fixture = Fixture::new();
        fixture.model("custom/local.gguf");
        fixture.model("studio/owner/repo/model.gguf");
        let store = fixture.0.join("ollama");
        let manifest = store.join("manifests/registry.ollama.ai/library/example/latest");
        fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        fs::create_dir_all(store.join("blobs")).unwrap();
        let digest = "a".repeat(64);
        fs::write(store.join(format!("blobs/sha256-{digest}")), b"GGUFfixture").unwrap();
        fs::write(&manifest, serde_json::to_vec(&serde_json::json!({
            "schemaVersion":2, "layers":[{"mediaType":"application/vnd.ollama.image.model", "digest":format!("sha256:{digest}")}]
        })).unwrap()).unwrap();
        let settings = app_settings::Settings {
            model_directories: vec![fixture.0.join("custom").to_str().unwrap().into()],
            lm_studio_directories: vec![fixture.0.join("studio").to_str().unwrap().into()],
            ollama_directory: Some(store.to_str().unwrap().into()),
            ..Default::default()
        };
        let config = fixture.0.join("config");
        app_settings::save(&config, settings).unwrap();
        let settings = app_settings::load(&config).unwrap();
        let (roots, lm_roots) =
            configured_locations(&settings, vec![fixture.0.join("custom")], vec![]);
        assert_eq!(roots.len(), 2);
        let mut found = models::discover(DiscoveryOptions {
            local_dirs: roots,
            ollama_dir: settings
                .ollama_directory
                .as_deref()
                .map(|p| app_settings::path(p).unwrap()),
        });
        assert_eq!(found.models.len(), 3);
        let ids: Vec<_> = found.models.iter().map(|m| m.id.clone()).collect();
        label_sources(&mut found, &lm_roots);
        assert_eq!(
            ids,
            found
                .models
                .iter()
                .map(|m| m.id.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            found
                .models
                .iter()
                .filter(|m| m.source == "lm-studio")
                .count(),
            1
        );
        assert_eq!(
            found.models.iter().filter(|m| m.source == "ollama").count(),
            1
        );
        let model = found.models.iter().find(|m| m.ollama).unwrap();
        let options = LoadOptions {
            model: model.model.clone(),
            ollama: true,
            ollama_dir: model.ollama_dir.clone(),
            projector: None,
            gpu: "auto".into(),
            context: 4096,
        };
        preferences::save(&config, options.clone()).unwrap();
        assert_eq!(preferences::load(&config).unwrap(), Some(options));
        assert_eq!(fs::read_dir(store.join("blobs")).unwrap().count(), 1);
        assert!(!config.join("models").exists());
    }

    #[test]
    fn final_warnings_replace_stale_sources_and_preserve_scan_diagnostics() {
        let fixture = Fixture::new();
        let store = fixture.ollama("store");
        fixture.imported_model("studio/model.gguf", &store, "example:latest");
        fs::write(fixture.0.join("studio/broken.gguf"), b"nope").unwrap();
        let mut catalog = models::discover(DiscoveryOptions {
            local_dirs: vec![fixture.0.join("studio")],
            ollama_dir: Some(store.clone()),
        });
        let stale = "Model 'example:latest' is available from multiple sources: lma, ollama. These may share the same weights; no files were removed.";
        assert!(catalog.warnings.iter().any(|warning| warning == stale));
        let scan_warnings: Vec<_> = catalog
            .warnings
            .iter()
            .filter(|warning| warning.as_str() != stale)
            .cloned()
            .collect();
        assert_eq!(scan_warnings.len(), 1);
        assert!(scan_warnings[0].contains("broken.gguf"));
        let before: Vec<_> = catalog
            .models
            .iter()
            .map(|model| model.id.clone())
            .collect();
        finish_catalog(&mut catalog, &[fixture.0.join("studio")]);
        let mut expected = scan_warnings;
        expected.push("Model 'example:latest' is available from multiple sources: lm-studio, ollama. These may share the same weights; no files were removed.".into());
        assert_eq!(catalog.warnings, expected);
        assert_eq!(
            catalog
                .models
                .iter()
                .map(|model| model.id.clone())
                .collect::<Vec<_>>(),
            before
        );
        assert_eq!(catalog.models.len(), 2);
        assert_eq!(fs::read_dir(store.join("blobs")).unwrap().count(), 1);
        assert_eq!(
            fs::read(fixture.0.join("studio/model.gguf")).unwrap(),
            b"GGUFfixture"
        );
    }

    #[test]
    fn final_warnings_drop_collisions_when_relabeling_leaves_only_one_source() {
        let fixture = Fixture::new();
        fixture.model("first/model.gguf");
        fixture.model("second/MODEL.gguf");
        let first = fixture.0.join("first");
        let second = fixture.0.join("second");
        let mut catalog = models::discover(DiscoveryOptions {
            local_dirs: vec![first.clone(), second.clone()],
            ollama_dir: Some(fixture.0.join("missing-store")),
        });
        finish_catalog(&mut catalog, &[first.clone()]);
        assert_eq!(
            catalog.warnings,
            [
                "Model 'model.gguf' is available from multiple sources: lm-studio, lma. These may share the same weights; no files were removed."
            ]
        );
        finish_catalog(&mut catalog, &[first, second]);
        assert!(catalog.warnings.is_empty());
        assert_eq!(catalog.models.len(), 2);
        assert!(
            catalog
                .models
                .iter()
                .all(|model| model.source == "lm-studio")
        );
    }

    #[test]
    fn merged_warnings_detect_cross_catalog_sources_and_distinct_ollama_stores() {
        let fixture = Fixture::new();
        let configured = fixture.ollama("configured-store");
        let remembered = fixture.ollama("remembered-store");
        fixture.imported_model("local/model.gguf", &configured, "EXAMPLE:latest");
        let mut catalog = models::discover(DiscoveryOptions {
            local_dirs: vec![fixture.0.join("local")],
            ollama_dir: Some(fixture.0.join("missing-default-store")),
        });
        assert!(catalog.warnings.is_empty());
        for root in [&configured, &remembered, &configured] {
            let extra = models::discover(DiscoveryOptions {
                local_dirs: Vec::new(),
                ollama_dir: Some(root.clone()),
            });
            assert!(extra.warnings.is_empty());
            catalog.models.extend(extra.models);
            catalog.warnings.extend(extra.warnings);
        }
        assert_eq!(catalog.models.len(), 4);
        finish_catalog(&mut catalog, &[]);
        assert_eq!(catalog.models.len(), 3);
        assert_eq!(catalog.warnings, [
            "Model 'example:latest' is available from multiple sources: lma, ollama. These may share the same weights; no files were removed.".to_string(),
            format!("Model 'example:latest' is available from multiple Ollama stores: {}, {}. These may share the same weights; no files were removed.", configured.display(), remembered.display()),
        ]);
        assert_eq!(
            catalog.models.iter().filter(|model| model.ollama).count(),
            2
        );
        assert!(
            catalog
                .models
                .windows(2)
                .all(
                    |pair| pair[0].name.to_lowercase() < pair[1].name.to_lowercase()
                        || (pair[0].name.to_lowercase() == pair[1].name.to_lowercase()
                            && pair[0].id < pair[1].id)
                )
        );
        let warnings = catalog.warnings.clone();
        finish_catalog(&mut catalog, &[]);
        assert_eq!(catalog.warnings, warnings);
    }

    #[test]
    fn repeated_store_scans_do_not_create_collisions_after_id_deduplication() {
        let fixture = Fixture::new();
        let store = fixture.ollama("store");
        let mut catalog = models::discover(DiscoveryOptions {
            local_dirs: Vec::new(),
            ollama_dir: Some(store),
        });
        catalog.models.extend(catalog.models.clone());
        finish_catalog(&mut catalog, &[]);
        assert_eq!(catalog.models.len(), 1);
        assert!(catalog.warnings.is_empty());
    }

    #[test]
    fn final_collision_warning_lengths_are_utf8_bounded() {
        let fixture = Fixture::new();
        fixture.model("local/model.gguf");
        let store = fixture.ollama("store");
        let mut catalog = models::discover(DiscoveryOptions {
            local_dirs: vec![fixture.0.join("local")],
            ollama_dir: Some(store),
        });
        for model in &mut catalog.models {
            model.name = "模".repeat(400);
        }
        finish_catalog(&mut catalog, &[]);
        assert_eq!(catalog.warnings.len(), 1);
        assert!(catalog.warnings[0].len() <= 1024);
        assert!(catalog.warnings[0].ends_with("..."));
        let warnings = catalog.warnings.clone();
        finish_catalog(&mut catalog, &[]);
        assert_eq!(catalog.warnings, warnings);
    }

    #[test]
    fn source_registration_is_bounded_read_only_and_optional() {
        let fixture = Fixture::new();
        let mut warnings = Vec::new();
        assert!(registered_source(&fixture.0, &mut warnings).is_none());
        assert!(warnings.is_empty());
        let path = fixture.0.join("source-install.json");
        fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({"schema_version":1,"source_dir":fixture.0}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            registered_source(&fixture.0, &mut warnings),
            Some(fixture.0.clone())
        );
        for value in [
            b"{broken".to_vec(),
            vec![b' '; 64 * 1024 + 1],
            b"{\"schema_version\":1,\"source_dir\":\"relative\"}".to_vec(),
        ] {
            fs::write(&path, &value).unwrap();
            assert!(registered_source(&fixture.0, &mut warnings).is_none());
            assert_eq!(fs::read(&path).unwrap(), value);
        }
        assert_eq!(warnings.len(), 3);
    }
}
