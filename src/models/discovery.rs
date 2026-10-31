//! Bounded, read-only discovery of host-selected local folders and one Ollama store.
//!
//! This synchronous API is intended for a caller's `spawn_blocking`. It reads only
//! GGUF headers (bounded metadata for split sets) and installed manifests, never
//! weights or hashes. Local
//! scans stop at depth 8; each scan (all local roots together, and Ollama separately)
//! examines at most 10,000 entries. At most 128 local roots, 1 MiB per manifest,
//! and 64 warnings are allowed. Missing optional roots are normal.
//!
//! Like the resolver, this assumes trusted, non-concurrently-mutated local stores:
//! regular-file/size checks precede the resolver's manifest read. Directory symlinks
//! encountered during traversal are never followed. Explicit roots are canonicalized.

use std::collections::HashSet;
use std::fs::{self, DirEntry, File};
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::{Reference, default_root, resolve, safe_component, safe_registry};

#[derive(Clone, Debug, Default)]
pub struct DiscoveryOptions {
    /// Only these local folders are searched; the core does not infer host folders.
    pub local_dirs: Vec<PathBuf>,
    /// Explicit store, otherwise OLLAMA_MODELS, otherwise HOME/.ollama/models.
    pub ollama_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredModel {
    pub id: String,
    pub name: String,
    pub source: String,
    /// Absolute local GGUF path, or normalized, tagged Ollama reference (no scheme).
    pub model: String,
    pub ollama: bool,
    pub ollama_dir: Option<PathBuf>,
    pub projector: Option<PathBuf>,
    /// Combined model/shard size, excluding any projector or external metadata.
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelCatalog {
    pub models: Vec<DiscoveredModel>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy)]
struct Limits {
    roots: usize,
    entries: usize,
    depth: usize,
    manifest_bytes: u64,
    warnings: usize,
}

const LIMITS: Limits = Limits {
    roots: 128,
    entries: 10_000,
    depth: 8,
    manifest_bytes: 1024 * 1024,
    warnings: 64,
};

/// Canonical download destination. Never falls back to the process working directory.
pub fn default_models_dir() -> Result<PathBuf, String> {
    Ok(super::storage::root()?.join("models"))
}

pub fn lm_studio_dirs() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(path) = std::env::var_os("LM_STUDIO_MODELS").filter(|v| !v.is_empty()) {
        roots.push(PathBuf::from(path));
    }
    if let Ok(home) = super::storage::home_dir() {
        roots.push(home.join(".lmstudio/models"));
        roots.push(home.join(".cache/lm-studio/models"));
    }
    roots
}

/// Standard CLI model folders. Missing folders are harmless; no folders are created.
/// Desktop hosts can continue supplying their own application-data roots.
pub fn default_local_dirs() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(path) = default_models_dir() {
        roots.push(path);
    }
    if let Some(path) = std::env::var_os("LMA_MODELS").filter(|p| !p.is_empty()) {
        roots.push(PathBuf::from(path));
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join("models"));
    }
    if let Some(data) = super::storage::legacy_data_dir() {
        roots.push(data.join("models"));
    }
    roots.extend(lm_studio_dirs());
    roots
}

/// Discover already-installed models without network access or disk writes.
///
/// Ollama templates, parameters, system prompts, and history are not applied.
/// Results are sorted by stable ID; tags remain distinct even when sharing blobs.
pub fn discover(options: DiscoveryOptions) -> ModelCatalog {
    discover_with_limits(options, LIMITS)
}

fn discover_with_limits(options: DiscoveryOptions, limits: Limits) -> ModelCatalog {
    discover_in_roots(options, limits, &lm_studio_dirs())
}

fn discover_in_roots(
    options: DiscoveryOptions,
    limits: Limits,
    lm_dirs: &[PathBuf],
) -> ModelCatalog {
    let mut scan = Scan {
        catalog: ModelCatalog::default(),
        limits,
        remaining: limits.entries,
        directories: HashSet::new(),
        files: HashSet::new(),
    };
    if options.local_dirs.len() > limits.roots {
        scan.warn(format!("local root limit ({}) reached", limits.roots));
    }
    for root in options.local_dirs.iter().take(limits.roots) {
        if scan.remaining == 0 {
            scan.warn("local entry limit reached".into());
            break;
        }
        if let Some(root) = scan.root(root) {
            scan.local(&root, 0);
        }
    }

    let mut registered_ids = HashSet::new();
    for root in options.local_dirs.iter().take(limits.roots) {
        match super::registry::read(root) {
            Ok(entries) => {
                for (name, entry) in entries {
                    let id = format!("lma:{name}");
                    if registered_ids.contains(&id) {
                        continue;
                    }
                    let path = match fs::canonicalize(&entry.path) {
                        Ok(path) => path,
                        Err(error) => {
                            scan.warn(format!("registered model '{name}' is {} at '{}': {error}; restore the target or remove the stale alias with `lma d lma:{name} -s lma`", if error.kind() == ErrorKind::NotFound { "missing" } else { "unavailable" }, entry.path.display()));
                            continue;
                        }
                    };
                    let Some(path_text) = path.to_str() else {
                        continue;
                    };
                    let size_bytes = match scan
                        .catalog
                        .models
                        .iter()
                        .find(|model| !model.ollama && model.model == path_text)
                    {
                        // Retain combined shard size already inspected by the local scan.
                        Some(existing) => existing.size_bytes,
                        None => match fs::metadata(&path) {
                            Ok(metadata) if metadata.is_file() => metadata.len(),
                            _ => {
                                scan.warn(format!(
                                    "registered model '{name}' is not a regular file"
                                ));
                                continue;
                            }
                        },
                    };
                    // Replace only the physical-file row, never a sibling alias.
                    // Registered identities and projectors belong to each logical model.
                    let path_id = format!("lma:{path_text}");
                    scan.catalog.models.retain(|model| model.id != path_id);
                    registered_ids.insert(id.clone());
                    scan.catalog.models.push(DiscoveredModel {
                        id,
                        name,
                        source: "lma".into(),
                        model: path_text.into(),
                        ollama: false,
                        ollama_dir: None,
                        projector: entry.projector,
                        size_bytes,
                    });
                }
            }
            Err(error) => scan.warn(error),
        }
    }
    // Reserve a separate budget so a large local folder cannot hide Ollama models.
    scan.remaining = limits.entries;
    match default_root(options.ollama_dir.as_deref()) {
        Ok(root) => {
            if let Some(root) = scan.root(&root) {
                let manifests = root.join("manifests");
                match fs::symlink_metadata(&manifests) {
                    Ok(metadata) if metadata.is_dir() => {
                        scan.ollama(&root, &manifests, &mut Vec::new());
                    }
                    Ok(_) => scan.warn(format!(
                        "Ollama manifests directory '{}' is not a regular directory",
                        manifests.display()
                    )),
                    Err(error) if error.kind() == ErrorKind::NotFound => {}
                    Err(error) => {
                        scan.warn(format!("cannot inspect '{}': {error}", manifests.display()))
                    }
                }
            }
        }
        Err(error) => scan.warn(error),
    }
    let lm_roots: Vec<_> = lm_dirs
        .iter()
        .filter_map(|path| fs::canonicalize(path).ok())
        .collect();
    for model in &mut scan.catalog.models {
        if !model.ollama
            && lm_roots
                .iter()
                .any(|root| Path::new(&model.model).starts_with(root))
        {
            model.source = "lm-studio".into();
        }
    }
    let mut sources =
        std::collections::BTreeMap::<String, std::collections::BTreeSet<String>>::new();
    for model in &scan.catalog.models {
        sources
            .entry(model.name.to_lowercase())
            .or_default()
            .insert(model.source.clone());
    }
    for (name, sources) in sources {
        if sources.len() > 1 {
            scan.warn(format!("Model '{name}' is available from multiple sources: {}. These may share the same weights; no files were removed.", sources.into_iter().collect::<Vec<_>>().join(", ")));
        }
    }
    scan.catalog.models.sort_by(|a, b| a.id.cmp(&b.id));
    scan.catalog
}

struct LocalFile {
    path: PathBuf,
    name: String,
    projector: bool,
    size: u64,
}

struct Scan {
    catalog: ModelCatalog,
    limits: Limits,
    remaining: usize,
    directories: HashSet<PathBuf>,
    files: HashSet<PathBuf>,
}

impl Scan {
    fn warn(&mut self, mut warning: String) {
        // Bound both the number and length of diagnostics, including OS errors.
        const MAX_WARNING_BYTES: usize = 1024;
        if warning.len() > MAX_WARNING_BYTES {
            let mut end = MAX_WARNING_BYTES - 3;
            while !warning.is_char_boundary(end) {
                end -= 1;
            }
            warning.truncate(end);
            warning.push_str("...");
        }
        let count = self.catalog.warnings.len();
        if count.saturating_add(1) < self.limits.warnings {
            self.catalog.warnings.push(warning);
        } else if count < self.limits.warnings {
            self.catalog
                .warnings
                .push("additional discovery warnings suppressed".into());
        }
    }

    fn root(&mut self, path: &Path) -> Option<PathBuf> {
        let result = fs::canonicalize(path).and_then(|path| {
            if fs::metadata(&path)?.is_dir() {
                Ok(path)
            } else {
                Err(std::io::Error::other("not a directory"))
            }
        });
        match result {
            Ok(path) => Some(path),
            Err(error) if error.kind() == ErrorKind::NotFound => None,
            Err(error) => {
                self.warn(format!("cannot scan '{}': {error}", path.display()));
                None
            }
        }
    }

    fn entries(&mut self, directory: &Path) -> (Vec<DirEntry>, bool) {
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) => {
                self.warn(format!("cannot scan '{}': {error}", directory.display()));
                return (Vec::new(), false);
            }
        };
        let mut found = Vec::new();
        let mut complete = true;
        for entry in entries {
            if self.remaining == 0 {
                self.warn(format!("entry limit reached at '{}'", directory.display()));
                complete = false;
                break;
            }
            self.remaining -= 1;
            match entry {
                Ok(entry) => found.push(entry),
                Err(error) => {
                    complete = false;
                    self.warn(format!(
                        "cannot read directory entry in '{}': {error}",
                        directory.display()
                    ));
                }
            }
        }
        // Only the bounded subset is sorted; selection at the limit is best-effort.
        found.sort_by_key(|entry| entry.file_name());
        (found, complete)
    }

    fn local(&mut self, directory: &Path, depth: usize) {
        if !self.directories.insert(directory.to_path_buf()) {
            return;
        }
        let names = crate::import::local_names(directory);
        let mut files = Vec::new();
        let mut directories = Vec::new();
        let (entries, mut complete) = self.entries(directory);
        for entry in entries {
            let path = entry.path();
            let kind = match entry.file_type() {
                Ok(kind) => kind,
                Err(error) => {
                    complete = false;
                    self.warn(format!("cannot inspect '{}': {error}", path.display()));
                    continue;
                }
            };
            if kind.is_dir() {
                if depth < self.limits.depth {
                    directories.push(path);
                } else {
                    self.warn(format!("local depth limit reached at '{}'", path.display()));
                }
                continue;
            }
            // Symlinked regular files can be deduplicated by canonical path, but
            // symlinked directories and special files are never opened.
            if !(kind.is_file() || kind.is_symlink()) {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let lower = name.to_ascii_lowercase();
            if !lower.ends_with(".gguf")
                || temporary(&lower)
                || super::gguf::shard_name(name).is_some_and(|(_, index, _)| index != 1)
            {
                continue;
            }
            match self.local_file(&path, name, is_projector(&lower)) {
                Ok(Some(mut file)) => {
                    if let Some(name) = names.get(&file.path) {
                        file.name = name.clone();
                    }
                    files.push(file);
                }
                Ok(None) => {}
                Err(error) => {
                    complete = false;
                    self.warn(error);
                }
            }
        }
        let models = files.iter().filter(|file| !file.projector).count();
        let projectors: Vec<_> = files.iter().filter(|file| file.projector).collect();
        // A truncated or corrupt listing cannot establish unique pairing.
        let projector = if complete && models == 1 && projectors.len() == 1 {
            Some(projectors[0].path.clone())
        } else {
            None
        };
        for file in files.into_iter().filter(|file| !file.projector) {
            // Non-UTF-8 paths cannot be represented losslessly in the API's String.
            let Some(model) = file.path.to_str() else {
                self.warn(format!(
                    "local model path is not UTF-8: '{}'",
                    file.path.display()
                ));
                continue;
            };
            self.catalog.models.push(DiscoveredModel {
                id: format!("lma:{model}"),
                name: file.name,
                source: "lma".into(),
                model: model.into(),
                ollama: false,
                ollama_dir: None,
                projector: projector.clone(),
                size_bytes: file.size,
            });
        }
        for directory in directories {
            if self.remaining == 0 {
                self.warn("local entry limit reached".into());
                break;
            }
            self.local(&directory, depth + 1);
        }
    }

    fn local_file(
        &mut self,
        path: &Path,
        name: &str,
        projector: bool,
    ) -> Result<Option<LocalFile>, String> {
        let inspect = || -> Result<Option<LocalFile>, std::io::Error> {
            let metadata = fs::metadata(path)?;
            if !metadata.is_file() {
                return Ok(None);
            }
            let canonical = fs::canonicalize(path)?;
            if self.files.contains(&canonical) {
                return Ok(None);
            }
            let mut magic = [0; 4];
            File::open(path)?.read_exact(&mut magic)?;
            if &magic != b"GGUF" {
                return Err(std::io::Error::other("not a GGUF file"));
            }
            Ok(Some(LocalFile {
                path: canonical,
                name: name.into(),
                projector,
                size: metadata.len(),
            }))
        };
        let mut file =
            inspect().map_err(|error| format!("cannot use '{}': {error}", path.display()))?;
        if let Some(file) = &mut file
            && let Some(set) = super::gguf::shards(path)?
        {
            file.size = set.size_bytes;
        }
        if let Some(file) = &file {
            self.files.insert(file.path.clone());
        }
        Ok(file)
    }

    fn ollama(&mut self, root: &Path, directory: &Path, parts: &mut Vec<String>) {
        // Exactly registry / namespace / model / tag. No recursive search beyond it.
        for entry in self.entries(directory).0 {
            let path = entry.path();
            let kind = match entry.file_type() {
                Ok(kind) => kind,
                Err(error) => {
                    self.warn(format!("cannot inspect '{}': {error}", path.display()));
                    continue;
                }
            };
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                self.warn(format!("non-UTF-8 Ollama entry '{}'", path.display()));
                continue;
            };
            if temporary(&name.to_ascii_lowercase()) {
                continue;
            }
            let safe = if parts.is_empty() {
                safe_registry(name) && name == name.to_ascii_lowercase()
            } else {
                safe_component(name)
            };
            if !safe {
                self.warn(format!(
                    "invalid Ollama path component '{}'",
                    path.display()
                ));
                continue;
            }
            if parts.len() < 3 {
                if kind.is_dir() {
                    if self.remaining == 0 {
                        self.warn(format!(
                            "Ollama entry limit reached at '{}'",
                            path.display()
                        ));
                        continue;
                    }
                    parts.push(name.into());
                    self.ollama(root, &path, parts);
                    parts.pop();
                }
            } else {
                let reference = Reference {
                    registry: parts[0].clone(),
                    namespace: parts[1].clone(),
                    model: parts[2].clone(),
                    tag: name.into(),
                };
                match self.ollama_model(root, &path, &reference) {
                    Ok(model) => self.catalog.models.push(model),
                    Err(error) => self.warn(error),
                }
            }
        }
    }

    fn ollama_model(
        &self,
        root: &Path,
        manifest: &Path,
        reference: &Reference,
    ) -> Result<DiscoveredModel, String> {
        let metadata = fs::metadata(manifest)
            .map_err(|error| format!("cannot inspect '{}': {error}", manifest.display()))?;
        if !metadata.is_file() || metadata.len() > self.limits.manifest_bytes {
            return Err(format!(
                "Ollama manifest '{}' must be a regular file of at most {} bytes",
                manifest.display(),
                self.limits.manifest_bytes
            ));
        }
        // Reuse the resolver's layer allowlist, blob magic checks, and single-level
        // provenance validation rather than interpreting Ollama metadata ourselves.
        let reference = reference.name();
        let resolved = resolve(Path::new(&format!("ollama:{reference}")), Some(root))
            .map_err(|error| format!("Ollama model '{reference}' in '{}': {error}; source files were left unchanged; use Ollama for unsupported formats or explicitly import a supported model with `lma it`", root.display()))?;
        if resolved.name.as_deref() != Some(&reference) {
            return Err(format!(
                "Ollama reference '{reference}' resolved as a local file"
            ));
        }
        let size_bytes = fs::metadata(&resolved.model)
            .map_err(|error| format!("cannot inspect '{}': {error}", resolved.model.display()))?
            .len();
        let root_text = root.to_str().ok_or("Ollama store path is not UTF-8")?;
        Ok(DiscoveredModel {
            // Length-prefix the store to avoid delimiter collisions in path names.
            id: format!("ollama:{}:{root_text}:{reference}", root_text.len()),
            name: reference.clone(),
            source: "ollama".into(),
            model: reference,
            ollama: true,
            ollama_dir: Some(root.to_path_buf()),
            projector: resolved.projector,
            size_bytes,
        })
    }
}

fn is_projector(name: &str) -> bool {
    name.contains("mmproj") || name.contains("projector")
}

fn temporary(name: &str) -> bool {
    name.starts_with('.')
        || [
            ".partial",
            ".part",
            ".tmp",
            ".temp",
            ".download",
            ".incomplete",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix) || name.contains(&format!("{suffix}.")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MANIFEST_MEDIA_TYPE, MODEL_MEDIA_TYPE, PROJECTOR_MEDIA_TYPE};
    use serde_json::{Value, json};
    use tempfile::{TempDir, tempdir};

    const MODEL_HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const PROJECTOR_HASH: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const PROVENANCE_HASH: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";

    struct Fixture(TempDir);

    impl Fixture {
        fn new() -> Self {
            Self(tempdir().unwrap())
        }

        fn path(&self, relative: &str) -> PathBuf {
            self.0.path().join(relative)
        }

        fn write(&self, relative: &str, bytes: &[u8]) -> PathBuf {
            let path = self.path(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, bytes).unwrap();
            path
        }

        fn options(&self, local: &[&str]) -> DiscoveryOptions {
            DiscoveryOptions {
                local_dirs: local.iter().map(|path| self.path(path)).collect(),
                ollama_dir: Some(self.path("store")),
            }
        }

        fn manifest(&self, reference: &str, layers: Value) -> PathBuf {
            let reference = Reference::parse(reference).unwrap();
            self.write(
                &format!(
                    "store/manifests/{}/{}/{}/{}",
                    reference.registry, reference.namespace, reference.model, reference.tag
                ),
                &serde_json::to_vec(&json!({"schemaVersion": 2, "layers": layers})).unwrap(),
            )
        }

        fn blob(&self, hash: &str, bytes: &[u8]) -> PathBuf {
            self.write(&format!("store/blobs/sha256-{hash}"), bytes)
        }

        fn install(&self, reference: &str) {
            self.manifest(reference, json!([layer(MODEL_MEDIA_TYPE, MODEL_HASH)]));
            self.blob(MODEL_HASH, b"GGUFmodel");
        }
    }

    fn layer(media_type: &str, hash: &str) -> Value {
        json!({"mediaType": media_type, "digest": format!("sha256:{hash}")})
    }

    #[test]
    fn custom_local_roots_are_absolute_deduplicated_and_serializable() {
        let fixture = Fixture::new();
        let first = fixture.write("appdata/models/a.gguf", b"GGUFsmall");
        fixture.write("cwd/models/nested/b.GGUF", b"GGUF");
        fixture.write("source/models/a.gguf", b"GGUFother");
        fixture.write("not-requested/invisible.gguf", b"GGUF");
        let mut options = fixture.options(&[
            "appdata/models",
            "appdata/models",
            "cwd/models/nested",
            "cwd/models",
            "source/models",
            "missing",
        ]);
        options
            .local_dirs
            .push(fixture.path("appdata/models/../models"));
        let catalog = discover(options);
        assert!(catalog.warnings.is_empty(), "{:?}", catalog.warnings);
        assert_eq!(catalog.models.len(), 3);
        let model = catalog
            .models
            .iter()
            .find(|model| model.name == "a.gguf" && model.size_bytes == 9)
            .unwrap();
        assert_eq!(
            model.model,
            fs::canonicalize(first).unwrap().to_str().unwrap()
        );
        assert_eq!(model.source, "lma");
        assert!(!model.ollama);
        assert!(model.ollama_dir.is_none());
        assert_ne!(catalog.models[0].id, catalog.models[1].id);
        let value = serde_json::to_value(&catalog).unwrap();
        assert!(value["models"][0].get("sizeBytes").is_some());
        assert!(value["models"][0].get("ollamaDir").is_some());
        assert!(value["models"][0].get("size_bytes").is_none());
        let again = discover(fixture.options(&["source/models", "cwd/models", "appdata/models"]));
        assert_eq!(
            catalog
                .models
                .iter()
                .map(|model| &model.id)
                .collect::<Vec<_>>(),
            again
                .models
                .iter()
                .map(|model| &model.id)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn missing_optional_roots_are_empty_and_explicit_store_does_not_fall_back() {
        let fixture = Fixture::new();
        fixture.install("must-not-appear");
        let mut options = fixture.options(&["missing/models"]);
        options.ollama_dir = Some(fixture.path("other-store"));
        let catalog = discover(options);
        assert!(catalog.models.is_empty());
        assert!(catalog.warnings.is_empty());
    }

    #[test]
    fn registered_alias_replaces_only_the_raw_row_and_keeps_a_stable_alias_id() {
        let fixture = Fixture::new();
        let weights = fs::canonicalize(fixture.write("local/model.gguf", b"GGUFweights")).unwrap();
        fixture.write("local/mmproj-inferred.gguf", b"GGUFinferred");
        let projector =
            fs::canonicalize(fixture.write("projectors/mmproj-selected.gguf", b"GGUFselected"))
                .unwrap();
        let label = super::super::registry::put(
            &fixture.path("local"),
            "team/plain",
            super::super::registry::Entry {
                path: weights.clone(),
                projector: Some(projector.clone()),
            },
        )
        .unwrap();
        let before = fs::read(fixture.path("local/.registry-v1.json")).unwrap();
        let catalog = discover(fixture.options(&["local", "local", "local/../local"]));
        assert!(catalog.warnings.is_empty(), "{:?}", catalog.warnings);
        assert_eq!(catalog.models.len(), 1);
        let model = &catalog.models[0];
        assert_eq!(model.id, format!("lma:{label}"));
        assert_eq!(model.name, label);
        assert_eq!(model.model, weights.to_str().unwrap());
        assert_eq!(model.projector.as_ref(), Some(&projector));
        assert_eq!(model.size_bytes, 11);
        assert_eq!(model.source, "lma");
        assert!(!model.ollama);
        assert!(model.ollama_dir.is_none());
        let again = discover(fixture.options(&["local"]));
        assert_eq!(
            serde_json::to_value(&catalog.models).unwrap(),
            serde_json::to_value(&again.models).unwrap()
        );
        assert_eq!(fs::read(weights).unwrap(), b"GGUFweights");
        assert_eq!(
            fs::read(fixture.path("local/.registry-v1.json")).unwrap(),
            before
        );
        assert_eq!(fs::read_dir(fixture.path("local")).unwrap().count(), 4);
    }

    #[test]
    fn modelfile_profiles_share_weights_but_keep_ids_systems_and_projectors_distinct() {
        let fixture = Fixture::new();
        let weights =
            fs::canonicalize(fixture.write("weights/shared.gguf", b"GGUFweights")).unwrap();
        let profiles = [
            (
                "team/first:latest",
                "First SYSTEM",
                "projectors/mmproj-first.gguf",
            ),
            (
                "team/second:latest",
                "Second SYSTEM",
                "projectors/mmproj-second.gguf",
            ),
        ];
        for (label, system, projector) in profiles {
            let projector = fixture.write(projector, b"GGUFprojector");
            super::super::registry::put_definition(
                &fixture.path("profiles"),
                label,
                super::super::registry::Entry {
                    path: weights.clone(),
                    projector: Some(projector),
                },
                super::super::modelfile::Definition {
                    from: weights.to_string_lossy().into_owned(),
                    system: Some(system.into()),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        let before = fs::read(fixture.path("profiles/.registry-v1.json")).unwrap();
        let catalog = discover(fixture.options(&["weights", "profiles", "weights", "profiles"]));
        assert!(catalog.warnings.is_empty(), "{:?}", catalog.warnings);
        assert_eq!(catalog.models.len(), 2);
        for (label, system, projector) in profiles {
            let id = format!("lma:{label}");
            let projector = fs::canonicalize(fixture.path(projector)).unwrap();
            // Exercise catalog selection by either resolver-facing key without
            // touching HOME, default registries, or an inference engine.
            for selector in [label, id.as_str()] {
                let selected: Vec<_> = catalog
                    .models
                    .iter()
                    .filter(|model| model.name == selector || model.id == selector)
                    .collect();
                assert_eq!(
                    selected.len(),
                    1,
                    "ambiguous or missing selector: {selector}"
                );
                let model = selected[0];
                assert_eq!(model.id, id);
                assert_eq!(model.name, label);
                assert_eq!(model.model, weights.to_str().unwrap());
                assert_eq!(model.projector.as_ref(), Some(&projector));
                assert_eq!(model.size_bytes, 11);
                assert_eq!(model.source, "lma");
                assert!(!model.ollama);
                assert!(model.ollama_dir.is_none());
            }
            let definition = super::super::registry::definition(&fixture.path("profiles"), label)
                .unwrap()
                .unwrap();
            assert_eq!(definition.system.as_deref(), Some(system));
        }
        for roots in [
            vec!["profiles", "weights", "profiles"],
            vec!["profiles", "profiles"],
        ] {
            let again = discover(fixture.options(&roots));
            assert!(again.warnings.is_empty(), "{:?}", again.warnings);
            assert_eq!(
                serde_json::to_value(&catalog.models).unwrap(),
                serde_json::to_value(&again.models).unwrap()
            );
        }
        assert_eq!(fs::read(&weights).unwrap(), b"GGUFweights");
        assert_eq!(fs::read_dir(fixture.path("weights")).unwrap().count(), 1);
        assert_eq!(fs::read_dir(fixture.path("profiles")).unwrap().count(), 2);
        assert_eq!(
            fs::read(fixture.path("profiles/.registry-v1.json")).unwrap(),
            before
        );
    }

    #[test]
    fn lm_studio_provenance_labels_files_and_aliases_without_changing_identity_or_storage() {
        let fixture = Fixture::new();
        let weights =
            fs::canonicalize(fixture.write("studio/weights.gguf", b"GGUFweights")).unwrap();
        fixture.write("studio-extra/local.gguf", b"GGUFlocal");
        let registry = fixture.path("profiles");
        for name in ["first", "second"] {
            super::super::registry::put(
                &registry,
                name,
                super::super::registry::Entry {
                    path: weights.clone(),
                    projector: None,
                },
            )
            .unwrap();
        }
        let before = fs::read(registry.join(".registry-v1.json")).unwrap();
        let options = fixture.options(&["studio", "studio-extra", "profiles"]);
        let catalog = discover_in_roots(options, LIMITS, &[fixture.path("studio")]);
        assert!(catalog.warnings.is_empty(), "{:?}", catalog.warnings);
        assert_eq!(catalog.models.len(), 3);
        for model in &catalog.models {
            if model.id == "lma:first:latest" || model.id == "lma:second:latest" {
                assert_eq!(model.source, "lm-studio");
                assert_eq!(Path::new(&model.model), weights);
            } else {
                assert_eq!(model.source, "lma"); // A sibling prefix is not inside studio.
            }
        }
        let physical = discover_in_roots(
            fixture.options(&["studio"]),
            LIMITS,
            &[fixture.path("studio")],
        );
        assert_eq!(physical.models[0].source, "lm-studio");
        assert_eq!(physical.models[0].id, format!("lma:{}", weights.display()));
        assert_eq!(fs::read(weights).unwrap(), b"GGUFweights");
        assert_eq!(
            fs::read(registry.join(".registry-v1.json")).unwrap(),
            before
        );
        assert_eq!(fs::read_dir(fixture.path("studio")).unwrap().count(), 1);
    }

    #[test]
    fn missing_alias_and_unsupported_phonon_remain_actionable_warnings() {
        let fixture = Fixture::new();
        let weights = fixture.write("profiles/weights.gguf", b"GGUFweights");
        super::super::registry::put(
            &fixture.path("profiles"),
            "stale",
            super::super::registry::Entry {
                path: weights.clone(),
                projector: None,
            },
        )
        .unwrap();
        fs::remove_file(&weights).unwrap();
        fixture.manifest(
            "phonon",
            json!([layer("application/vnd.ollama.image.audio", MODEL_HASH)]),
        );
        fixture.blob(MODEL_HASH, b"unsupported audio weights");
        let before = fs::read(fixture.path("profiles/.registry-v1.json")).unwrap();
        let catalog = discover_in_roots(fixture.options(&["profiles"]), LIMITS, &[]);
        assert!(catalog.models.is_empty());
        assert_eq!(catalog.warnings.len(), 2, "{:?}", catalog.warnings);
        assert!(catalog.warnings.iter().any(|warning| {
            warning.contains("registered model 'stale:latest' is missing")
                && warning.contains("lma d lma:stale:latest -s lma")
                && warning.contains("weights.gguf")
        }));
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("phonon:latest")
                    && warning.contains("unsupported")
                    && warning.contains("left unchanged"))
        );
        assert_eq!(
            fs::read(fixture.path("profiles/.registry-v1.json")).unwrap(),
            before
        );
        assert_eq!(
            fs::read(fixture.path(&format!("store/blobs/sha256-{MODEL_HASH}"))).unwrap(),
            b"unsupported audio weights"
        );
    }

    #[test]
    fn imported_labels_replace_filenames_and_duplicate_sources_warn_without_copies() {
        let fixture = Fixture::new();
        fixture.install("example:tag");
        let output =
            fs::canonicalize(fixture.write("local/converted.gguf", b"GGUFweights")).unwrap();
        let fingerprint = "a".repeat(64);
        fixture.write(&format!("local/.lma-import-v1-{fingerprint}.json"), &serde_json::to_vec(&json!({"version":1,"source_root":fs::canonicalize(fixture.path("store")).unwrap(),"model":"example:tag","fingerprint":fingerprint,"output":output,"output_bytes":11})).unwrap());
        let catalog = discover(fixture.options(&["local"]));
        assert_eq!(catalog.models.len(), 2);
        assert!(
            catalog
                .models
                .iter()
                .all(|model| model.name == "example:tag")
        );
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("multiple sources")
                    && warning.contains("lma")
                    && warning.contains("ollama"))
        );
        let local = catalog.models.iter().find(|model| !model.ollama).unwrap();
        assert_eq!(local.id, format!("lma:{}", output.to_str().unwrap()));
        assert_ne!(local.id, format!("lma:{}", local.name));
        assert!(output.exists());
        assert_eq!(fs::read_dir(fixture.path("local")).unwrap().count(), 2);
    }

    #[test]
    fn local_headers_partials_shards_and_projectors_are_filtered() {
        let fixture = Fixture::new();
        fixture.write("local/good.gguf", b"GGUFweights");
        for name in [
            "a.partial.gguf",
            "b.part.gguf",
            "c.tmp.gguf",
            ".hidden.gguf",
            "a-00001-of-00002.gguf",
            "a-00002-of-00002.gguf",
            "weights.gguf.partial",
        ] {
            fixture.write(&format!("local/{name}"), b"GGUF");
        }
        fixture.write("local/bad.gguf", b"nope");
        fixture.write("local/short.gguf", b"GG");
        fixture.write("local/weights.safetensors", b"GGUF");
        fixture.write("projectors/mmproj-only.gguf", b"GGUF");
        fixture.write("projectors/vision-projector.gguf", b"GGUF");
        let catalog = discover(fixture.options(&["local", "projectors"]));
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].name, "good.gguf");
        assert_eq!(catalog.models[0].size_bytes, 11);
        assert_eq!(catalog.warnings.len(), 3);
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("invalid GGUF shard"))
        );
    }

    #[test]
    fn projectors_require_a_unique_model_unique_projector_and_valid_header() {
        let fixture = Fixture::new();
        for directory in [
            "unique",
            "ambiguous-projectors",
            "ambiguous-models",
            "invalid",
        ] {
            fixture.write(&format!("{directory}/model.gguf"), b"GGUF");
            fixture.write(&format!("{directory}/mmproj.gguf"), b"GGUFprojector");
        }
        let projector = fs::canonicalize(fixture.path("unique/mmproj.gguf")).unwrap();
        fixture.write("ambiguous-projectors/mmproj-other.gguf", b"GGUF");
        fixture.write("ambiguous-models/other.gguf", b"GGUF");
        fixture.write("invalid/mmproj.gguf", b"not-gguf");
        let catalog = discover(fixture.options(&[
            "unique",
            "ambiguous-projectors",
            "ambiguous-models",
            "invalid",
        ]));
        assert_eq!(catalog.models.len(), 5);
        for model in &catalog.models {
            if Path::new(&model.model)
                .parent()
                .unwrap()
                .ends_with("unique")
            {
                assert_eq!(model.projector.as_ref(), Some(&projector));
            } else {
                assert!(model.projector.is_none());
            }
        }
        assert_eq!(catalog.warnings.len(), 1);
    }

    #[test]
    fn ollama_preserves_registries_tags_projectors_and_nested_hf_provenance() {
        let fixture = Fixture::new();
        let reference = "hf.co/mradermacher/Pulsar-16B-BF16-GGUF:Q4_K_M";
        let layers = json!([
            layer(MODEL_MEDIA_TYPE, MODEL_HASH),
            layer(PROJECTOR_MEDIA_TYPE, PROJECTOR_HASH)
        ]);
        fixture.blob(
            PROVENANCE_HASH,
            &serde_json::to_vec(&json!({
                "schemaVersion": 2, "mediaType": MANIFEST_MEDIA_TYPE, "layers": layers
            }))
            .unwrap(),
        );
        fixture.manifest(
            reference,
            json!([
                layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH),
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(PROJECTOR_MEDIA_TYPE, PROJECTOR_HASH),
                layer("application/vnd.ollama.image.template", MODEL_HASH)
            ]),
        );
        fixture.blob(MODEL_HASH, b"GGUFmodel");
        let projector = fixture.blob(PROJECTOR_HASH, b"GGUFprojector");
        fixture.install("team/model:Q8_0");
        fixture.install("team/model:latest");
        fixture.install("plain");
        fixture.install("localhost:5000/library/plain:v1");
        let catalog = discover(fixture.options(&[]));
        assert!(catalog.warnings.is_empty(), "{:?}", catalog.warnings);
        assert_eq!(catalog.models.len(), 5);
        let model = catalog
            .models
            .iter()
            .find(|model| model.model == reference)
            .unwrap();
        assert_eq!(model.name, reference);
        assert_eq!(model.source, "ollama");
        assert!(model.ollama);
        assert_eq!(
            model.ollama_dir,
            Some(fs::canonicalize(fixture.path("store")).unwrap())
        );
        assert_eq!(model.projector, Some(fs::canonicalize(projector).unwrap()));
        assert_eq!(model.size_bytes, 9);
        assert!(
            catalog
                .models
                .iter()
                .any(|model| model.model == "plain:latest")
        );
        let ids: HashSet<_> = catalog.models.iter().map(|model| &model.id).collect();
        assert_eq!(ids.len(), 5);
        let other = Fixture::new();
        other.install("plain");
        let other_catalog = discover(other.options(&[]));
        assert_ne!(
            other_catalog.models[0].id,
            catalog
                .models
                .iter()
                .find(|model| model.model == "plain:latest")
                .unwrap()
                .id
        );
    }

    #[test]
    fn corrupt_unsupported_and_mismatched_provenance_entries_are_skipped() {
        let fixture = Fixture::new();
        fixture.install("good");
        fixture.write(
            "store/manifests/registry.ollama.ai/library/bad/latest",
            b"{",
        );
        fixture.manifest(
            "adapter",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer("application/vnd.ollama.image.adapter", MODEL_HASH)
            ]),
        );
        fixture.manifest(
            "sharded",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(MODEL_MEDIA_TYPE, MODEL_HASH)
            ]),
        );
        fixture.manifest("missing", json!([layer(MODEL_MEDIA_TYPE, PROJECTOR_HASH)]));
        fixture.manifest(
            "invalid-digest",
            json!([{"mediaType": MODEL_MEDIA_TYPE, "digest": "../../escape"}]),
        );
        fixture.manifest(
            "provenance",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH)
            ]),
        );
        fixture.blob(
            PROVENANCE_HASH,
            &serde_json::to_vec(&json!({
                "schemaVersion": 2, "mediaType": MANIFEST_MEDIA_TYPE,
                "layers": [layer(MODEL_MEDIA_TYPE, PROJECTOR_HASH)]
            }))
            .unwrap(),
        );
        fixture.write(
            "store/manifests/registry.ollama.ai/library/unfinished/latest.partial",
            b"{",
        );
        let catalog = discover(fixture.options(&[]));
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].model, "good:latest");
        assert_eq!(catalog.warnings.len(), 6, "{:?}", catalog.warnings);
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("must match"))
        );
    }

    #[test]
    fn ollama_model_and_projector_blobs_must_be_gguf() {
        let fixture = Fixture::new();
        fixture.install("bad-model");
        fixture.blob(MODEL_HASH, b"nope");
        assert!(discover(fixture.options(&[])).models.is_empty());
        fixture.blob(MODEL_HASH, b"GGUF");
        fixture.manifest(
            "bad-projector",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(PROJECTOR_MEDIA_TYPE, PROJECTOR_HASH)
            ]),
        );
        fixture.blob(PROJECTOR_HASH, b"GG");
        let catalog = discover(fixture.options(&[]));
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].model, "bad-model:latest");
        assert_eq!(catalog.warnings.len(), 1);
    }

    #[test]
    fn traversal_root_depth_entry_and_warning_limits_are_bounded() {
        let fixture = Fixture::new();
        fixture.write("local/a.gguf", b"GGUF");
        fixture.write("local/nested/b.gguf", b"GGUF");
        fixture.write("local/nested/deeper/c.gguf", b"GGUF");
        let catalog =
            discover_with_limits(fixture.options(&["local"]), Limits { depth: 1, ..LIMITS });
        assert_eq!(catalog.models.len(), 2);
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("depth limit"))
        );
        for index in 0..20 {
            fixture.write(&format!("many/{index}.gguf"), b"GGUF");
            fixture.write(&format!("broken/{index}.gguf"), b"bad");
            fixture.install(&format!("entry-{index}"));
        }
        let catalog = discover_with_limits(
            fixture.options(&["many"]),
            Limits {
                entries: 5,
                ..LIMITS
            },
        );
        assert!(catalog.models.iter().filter(|model| !model.ollama).count() <= 5);
        assert!(catalog.models.iter().filter(|model| model.ollama).count() <= 5);
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("entry limit"))
        );
        let catalog = discover_with_limits(
            fixture.options(&["broken"]),
            Limits {
                warnings: 3,
                ..LIMITS
            },
        );
        assert_eq!(catalog.warnings.len(), 3);
        assert!(catalog.warnings.last().unwrap().contains("suppressed"));
        let catalog = discover_with_limits(
            fixture.options(&["local", "many"]),
            Limits { roots: 1, ..LIMITS },
        );
        assert!(
            !catalog
                .models
                .iter()
                .any(|model| model.model.contains("/many/"))
        );
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("root limit"))
        );
    }

    #[test]
    fn incomplete_or_corrupt_directories_do_not_establish_projector_uniqueness() {
        let fixture = Fixture::new();
        fixture.write("limited/first.gguf", b"GGUF");
        fixture.write("limited/second.gguf", b"GGUF");
        fixture.write("limited/mmproj.gguf", b"GGUF");
        let catalog = discover_with_limits(
            fixture.options(&["limited"]),
            Limits {
                entries: 2,
                ..LIMITS
            },
        );
        assert!(!catalog.models.is_empty());
        assert!(catalog.models.iter().all(|model| model.projector.is_none()));
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("entry limit"))
        );

        fixture.write("corrupt/model.gguf", b"GGUF");
        fixture.write("corrupt/mmproj.gguf", b"GGUF");
        fixture.write("corrupt/mmproj-other.gguf", b"broken");
        let catalog = discover(fixture.options(&["corrupt"]));
        assert_eq!(catalog.models.len(), 1);
        assert!(catalog.models[0].projector.is_none());
        assert_eq!(catalog.warnings.len(), 1);
    }

    #[test]
    fn ollama_warns_when_budget_ends_before_a_model_directory() {
        let fixture = Fixture::new();
        fixture.install("plain");
        let catalog = discover_with_limits(
            fixture.options(&[]),
            Limits {
                entries: 3,
                ..LIMITS
            },
        );
        assert!(catalog.models.is_empty());
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("entry limit"))
        );
    }

    #[test]
    fn oversized_outer_and_provenance_manifests_are_rejected() {
        let fixture = Fixture::new();
        fixture.install("good");
        fixture.write(
            "store/manifests/registry.ollama.ai/library/large/latest",
            &[b' '; 513],
        );
        let catalog = discover_with_limits(
            fixture.options(&[]),
            Limits {
                manifest_bytes: 512,
                ..LIMITS
            },
        );
        assert_eq!(catalog.models.len(), 1);
        assert!(catalog.warnings[0].contains("at most 512 bytes"));
        fixture.manifest(
            "provenance",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH)
            ]),
        );
        let path = fixture.blob(PROVENANCE_HASH, b"");
        File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(super::super::MAX_PROVENANCE_BYTES + 1)
            .unwrap();
        let catalog = discover(fixture.options(&[]));
        assert_eq!(catalog.models.len(), 1);
        assert!(
            catalog
                .warnings
                .iter()
                .any(|warning| warning.contains("provenance") && warning.contains("exceeds"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_directories_are_not_followed_and_file_aliases_are_deduplicated() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let model = fixture.write("local/model.gguf", b"GGUF");
        fixture.write("outside/hidden.gguf", b"GGUF");
        symlink(&model, fixture.path("local/alias.gguf")).unwrap();
        symlink(fixture.path("local"), fixture.path("local/cycle")).unwrap();
        symlink(fixture.path("outside"), fixture.path("local/outside")).unwrap();
        fs::create_dir_all(fixture.path("store/manifests")).unwrap();
        symlink(
            fixture.path("outside"),
            fixture.path("store/manifests/hf.co"),
        )
        .unwrap();
        let catalog = discover(fixture.options(&["local"]));
        assert_eq!(catalog.models.len(), 1);
        assert!(catalog.warnings.is_empty());
        assert_eq!(
            catalog.models[0].model,
            fs::canonicalize(model).unwrap().to_str().unwrap()
        );
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn fifos_are_never_opened_as_models_manifests_or_blobs() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        // Use the OS directly: discovery itself never starts a subprocess.
        #[cfg(target_os = "macos")]
        type Mode = u16;
        #[cfg(target_os = "linux")]
        type Mode = u32;
        unsafe extern "C" {
            fn mkfifo(path: *const std::ffi::c_char, mode: Mode) -> std::ffi::c_int;
        }
        fn fifo(path: &Path) {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            let path = CString::new(path.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { mkfifo(path.as_ptr(), 0o600) }, 0);
        }
        let fixture = Fixture::new();
        fifo(&fixture.path("local/trap.gguf"));
        let manifest = fixture.path("store/manifests/registry.ollama.ai/library/trap/latest");
        fifo(&manifest);
        fixture.manifest("blob", json!([layer(MODEL_MEDIA_TYPE, MODEL_HASH)]));
        fifo(&fixture.path(&format!("store/blobs/sha256-{MODEL_HASH}")));
        let catalog = discover(fixture.options(&["local"]));
        assert!(catalog.models.is_empty());
        assert_eq!(catalog.warnings.len(), 2);
    }
}
