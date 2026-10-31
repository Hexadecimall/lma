//! Read-only resolution of local files and already-installed Ollama GGUF blobs.
//!
//! This module never contacts a registry or daemon, runs a subprocess, or downloads
//! anything. Only one store is consulted: an explicit root, `OLLAMA_MODELS`, or
//! `$HOME/.ollama/models` (`USERPROFILE` on Windows, with `HOME` as a fallback).
//! System-service stores are deliberately not searched: select their models
//! directory explicitly rather than accidentally using a different installation.
//!
//! # Integration contract
//!
//! **Ollama templates, system prompts, parameters, and message history are NOT
//! applied.** Callers must communicate this limitation; this is file resolution,
//! not Ollama inference compatibility. A caller's explicit local `--mmproj` must
//! override the automatically discovered [`ResolvedModel::projector`].
//!
//! Existing local files are returned unchanged, including non-UTF-8 path bytes;
//! their format remains the loader's responsibility. Ollama blobs are checked for
//! regular-file status and GGUF magic, not fully parsed or cryptographically
//! verified. Stores are assumed to be trusted local directories (symlinks are
//! followed). Exactly one model layer and at most one projector are supported.
//! A single nested Docker v2 manifest is accepted only as provenance: its model
//! and projector digests must match the outer manifest, with no further nesting.
//!
//! I really should put more comments in my code.

use std::env;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};

use serde_json::Value;

pub mod delete;
pub mod discovery;
pub mod gguf;
pub mod modelfile;
pub mod registry;
pub mod storage;
pub use discovery::{DiscoveredModel, DiscoveryOptions, ModelCatalog, discover};

// define this crap (self explanatory)
const DEFAULT_REGISTRY: &str = "registry.ollama.ai";
const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";
const PROJECTOR_MEDIA_TYPE: &str = "application/vnd.ollama.image.projector";
const MANIFEST_MEDIA_TYPE: &str = "application/vnd.docker.distribution.manifest.v2+json";
const MAX_PROVENANCE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub model: PathBuf,
    pub projector: Option<PathBuf>,
    /// Installed model label (including imported aliases); unnamed local files use None.
    pub name: Option<String>,
    /// Registry alias selected by catalog identity, never inferred from a display name.
    pub registry_name: Option<String>,
}

impl ResolvedModel {
    pub(crate) fn registry_definition(
        &self,
        roots: &[PathBuf],
    ) -> Result<Option<modelfile::Definition>, String> {
        let Some(name) = &self.registry_name else {
            return Ok(None);
        };
        let model = fs::canonicalize(&self.model).map_err(|error| error.to_string())?;
        let projector = self
            .projector
            .as_ref()
            .map(fs::canonicalize)
            .transpose()
            .map_err(|error| error.to_string())?;
        for root in roots {
            // Discovery skips malformed registries too. Only the source whose
            // alias still points at the selected files may supply its definition.
            let Ok(mut entries) = registry::read(root) else {
                continue;
            };
            let Some(entry) = entries.remove(name) else {
                continue;
            };
            if fs::canonicalize(entry.path).ok().as_ref() == Some(&model)
                && entry.projector.map(fs::canonicalize).transpose().ok() == Some(projector.clone())
            {
                return registry::definition(root, name);
            }
        }
        Err(format!(
            "registered model '{name}' no longer matches its selected files; resolve the model again"
        ))
    }
}

/// Resolve a local file, registered alias, or installed Ollama reference without modifying it.
///
/// Local files take precedence even over `ollama:` syntax. Otherwise accepted
/// references include `llama3.2`, `ollama:llama3.2:latest`, `team/model:tag`, and
/// `registry.example:5000/team/model:tag`. A registry with no namespace uses
/// `library`. For the default registry, the registry and `library` namespace are
/// omitted from the returned name. Other registries retain their namespace to
/// keep IDs unambiguous. An omitted tag becomes `latest`.
///
/// Missing absolute paths, explicitly relative paths, and `.gguf` paths produce
/// local-path errors instead of being silently interpreted as model names.
/// Registered aliases also accept normalized bare names and `lma:` selectors.
/// An explicit `lma:` selector never falls back to Ollama. Prefix an ambiguous
/// reference with `ollama:` to explicitly request the store.
pub fn resolve(input: &Path, explicit_root: Option<&Path>) -> Result<ResolvedModel, String> {
    resolve_with_catalog(input, explicit_root, || {
        discover(DiscoveryOptions {
            local_dirs: discovery::default_local_dirs(),
            ollama_dir: explicit_root.map(Path::to_path_buf),
        })
    })
}

fn resolve_with_catalog(
    input: &Path,
    explicit_root: Option<&Path>,
    catalog: impl FnOnce() -> ModelCatalog,
) -> Result<ResolvedModel, String> {
    // here we do another match thingy for the UAHAEHHHHH I HATE WRITING COMMENTS
    // anyways we do a match on the input path to see if it is a file or not
    match fs::metadata(input) {
        Ok(metadata) if metadata.is_file() => {
            return Ok(ResolvedModel {
                model: gguf::shards(input)?.map_or_else(|| input.to_path_buf(), |set| set.first),
                projector: None,
                name: input.parent().and_then(|parent| {
                    fs::canonicalize(input)
                        .ok()
                        .and_then(|path| crate::import::local_names(parent).remove(&path))
                }),
                registry_name: None,
            });
        }
        Ok(_) => return Err(format!("local path '{}' is not a file", input.display())),
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::NotFound
                    | ErrorKind::NotADirectory
                    | ErrorKind::InvalidInput
                    | ErrorKind::InvalidFilename
            ) => {}
        Err(error) => {
            return Err(format!(
                // throw an error at the users face because they screwed up
                "cannot inspect local path '{}': {error}",
                input.display()
            ));
        }
    }

    let text = input.to_str().ok_or_else(|| {
        format!(
            "local path '{}' does not exist; Ollama references must be UTF-8",
            input.display()
        )
    })?;
    let explicit_lma = text.strip_prefix("lma:");
    if !text.starts_with("ollama:") && (explicit_lma.is_some() || !looks_like_path(input, text)) {
        let catalog = catalog();
        let normalized = registry::name(explicit_lma.unwrap_or(text)).ok();
        let mut matches = catalog.models.iter().filter(|model| {
            let registered = model.id == format!("lma:{}", model.name);
            !model.ollama
                && (model.id == text
                    || (registered && normalized.as_deref() == Some(model.name.as_str()))
                    || (explicit_lma.is_none()
                        && (model.name == text
                            || Path::new(&model.name).file_stem().and_then(|s| s.to_str())
                                == Some(text))))
        });
        if let Some(model) = matches.next() {
            if matches.next().is_some() {
                return Err(format!(
                    "installed model name '{text}' is ambiguous; choose its source in the picker or use `lma s -p FILE`"
                ));
            }
            return Ok(ResolvedModel {
                model: PathBuf::from(&model.model),
                projector: model.projector.clone(),
                name: Some(model.name.clone()),
                registry_name: (model.id == format!("lma:{}", model.name))
                    .then(|| model.name.clone()),
            });
        }
        if explicit_lma.is_some() {
            return Err(format!(
                "installed lma model '{text}' was not found; nothing is downloaded"
            ));
        }
    }
    let reference = if let Some(reference) = text.strip_prefix("ollama:") {
        reference
    } else {
        if looks_like_path(input, text) {
            return Err(format!(
                "local model path '{}' does not exist",
                input.display()
            ));
        }
        text
    };
    let reference = Reference::parse(reference)?;
    let root = default_root(explicit_root)?;
    let metadata = fs::metadata(&root).map_err(|error| {
        format!(
            "cannot access Ollama models directory '{}': {error}",
            root.display()
        )
    })?;
    if !metadata.is_dir() {
        return Err(format!(
            "Ollama models root '{}' is not a directory",
            root.display()
        ));
    }

    let manifest = root
        .join("manifests")
        .join(&reference.registry)
        .join(&reference.namespace)
        .join(&reference.model)
        .join(&reference.tag);
    let bytes = fs::read(&manifest).map_err(|error| {
        format!(
            "cannot read installed Ollama model '{}' at '{}': {error}; only this store is searched, and nothing is downloaded",
            reference.name(),
            manifest.display()
        )
    })?;
    let json: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid Ollama manifest '{}': {error}", manifest.display()))?;
    if json["format"] == "safetensors"
        && json["runner"] == "mlx"
        && let Some(model) = crate::import::cached_model(&root, &json, &reference.name())
    {
        return Ok(ResolvedModel {
            model,
            projector: None,
            name: Some(reference.name()),
            registry_name: None,
        });
    }
    let selected = selected_layers(&json)
        .and_then(|selected| {
            validate_provenance(&root, &selected)?;
            Ok(selected)
        })
        .map_err(|error| {
            format!(
                "unsupported Ollama manifest '{}': {error}",
                manifest.display()
            )
        })?;

    Ok(ResolvedModel {
        model: checked_blob(&root, selected.model, "model")?,
        projector: selected
            .projector
            .map(|digest| checked_blob(&root, digest, "projector"))
            .transpose()?,
        name: Some(reference.name()),
        registry_name: None,
    })
}

/// Locate a selected installed manifest without loading or converting weights.
/// The importer applies its own bounded, regular-file checks to this path.
pub(crate) fn import_source(
    name: &str,
    explicit_root: Option<&Path>,
) -> Result<(PathBuf, PathBuf, String), String> {
    let reference = Reference::parse(name.strip_prefix("ollama:").unwrap_or(name))?;
    let root = fs::canonicalize(default_root(explicit_root)?)
        .map_err(|error| format!("cannot access Ollama model store: {error}"))?;
    if !root.is_dir() {
        return Err("Ollama store must be a directory".into());
    }
    let manifest = root
        .join("manifests")
        .join(&reference.registry)
        .join(&reference.namespace)
        .join(&reference.model)
        .join(&reference.tag);
    Ok((root, manifest, reference.name()))
}

fn looks_like_path(path: &Path, text: &str) -> bool {
    path.is_absolute()
        || matches!(text, "." | ".." | "~")
        || text.starts_with("./")
        || text.starts_with("../")
        || text.starts_with("~/")
        || text.contains('\\')
        || (text.as_bytes().get(1) == Some(&b':') && text.as_bytes().get(2) == Some(&b'/'))
        || text.ends_with('/')
        || text.to_ascii_lowercase().ends_with(".gguf")
}

fn default_root(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let ollama = env::var_os("OLLAMA_MODELS");
    #[cfg(windows)]
    let home = env::var_os("USERPROFILE").or_else(|| env::var_os("HOME"));
    #[cfg(not(windows))]
    let home = env::var_os("HOME");
    choose_root(explicit, ollama.as_deref(), home.as_deref())
}

// Environment values are arguments so precedence can be tested without mutating
// the process environment (including when tests are run concurrently).
fn choose_root(
    explicit: Option<&Path>,
    ollama: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Result<PathBuf, String> {
    let (root, source) = if let Some(root) = explicit {
        (root.to_path_buf(), "explicit root")
    } else if let Some(root) = ollama {
        (PathBuf::from(root), "OLLAMA_MODELS")
    } else if let Some(home) = home {
        if home.is_empty() {
            return Err("HOME/USERPROFILE is empty; specify an Ollama models directory".into());
        }
        (PathBuf::from(home).join(".ollama/models"), "home directory")
    } else {
        return Err("no home directory is set; specify an explicit root or OLLAMA_MODELS".into());
    };
    if root.as_os_str().is_empty() {
        return Err(format!(
            "{source} is empty; expected an Ollama models directory"
        ));
    }
    Ok(root)
}

struct Reference {
    registry: String,
    namespace: String,
    model: String,
    tag: String,
}

impl Reference {
    fn parse(text: &str) -> Result<Self, String> {
        let invalid = || {
            format!(
                "invalid Ollama reference '{text}'; expected [registry/][namespace/]model[:tag] with safe path components"
            )
        };
        let parts: Vec<_> = text.split('/').collect();
        let (registry, namespace, model_tag) = match parts.as_slice() {
            [model] => (DEFAULT_REGISTRY, "library", *model),
            [first, model]
                if first.contains('.') || first.contains(':') || *first == "localhost" =>
            {
                (*first, "library", *model)
            }
            [namespace, model] => (DEFAULT_REGISTRY, *namespace, *model),
            [registry, namespace, model] => (*registry, *namespace, *model),
            _ => return Err(invalid()),
        };
        let (model, tag) = model_tag.rsplit_once(':').unwrap_or((model_tag, "latest"));
        if !safe_component(namespace) || !safe_component(model) || !safe_component(tag) {
            return Err(invalid());
        }
        if !safe_registry(registry) {
            return Err(invalid());
        }
        Ok(Self {
            registry: registry.to_ascii_lowercase(),
            namespace: namespace.into(),
            model: model.into(),
            tag: tag.into(),
        })
    }

    fn name(&self) -> String {
        let mut name = String::new();
        if self.registry != DEFAULT_REGISTRY {
            name.push_str(&self.registry);
            name.push('/');
        }
        if self.registry != DEFAULT_REGISTRY || self.namespace != "library" {
            name.push_str(&self.namespace);
            name.push('/');
        }
        name.push_str(&self.model);
        name.push(':');
        name.push_str(&self.tag);
        name
    }
}

fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn safe_registry(value: &str) -> bool {
    let (host, port) = value
        .split_once(':')
        .map_or((value, None), |(host, port)| (host, Some(port)));
    !host.is_empty()
        && value.len() <= 255
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.as_bytes()[0].is_ascii_alphanumeric()
                && label.as_bytes()[label.len() - 1].is_ascii_alphanumeric()
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        })
        && port.is_none_or(|port| {
            !port.is_empty()
                && port.bytes().all(|byte| byte.is_ascii_digit())
                && port.parse::<u16>().is_ok_and(|port| port != 0)
        })
}

struct SelectedLayers<'a> {
    model: &'a str,
    projector: Option<&'a str>,
    provenance: Option<&'a str>,
}

fn selected_layers(json: &Value) -> Result<SelectedLayers<'_>, String> {
    if json.get("manifests").is_some() {
        return Err("manifest indexes/multi-model or sharded layouts are not supported".into());
    }
    if json
        .get("schemaVersion")
        .is_some_and(|version| version.as_u64() != Some(2))
    {
        return Err("only schemaVersion 2 manifests are supported".into());
    }
    let layers = json
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("expected a layers array containing one GGUF model")?;
    let mut model = None;
    let mut projector = None;
    let mut provenance = None;
    for layer in layers {
        let media_type = layer
            .get("mediaType")
            .and_then(Value::as_str)
            .ok_or("a layer is missing its mediaType string")?;
        let selected = match media_type {
            MODEL_MEDIA_TYPE => &mut model,
            PROJECTOR_MEDIA_TYPE => &mut projector,
            MANIFEST_MEDIA_TYPE => &mut provenance,
            "application/vnd.ollama.image.template"
            | "application/vnd.ollama.image.system"
            | "application/vnd.ollama.image.params"
            | "application/vnd.ollama.image.messages"
            | "application/vnd.ollama.image.license" => continue,
            "application/vnd.ollama.image.tensor" => {
                return Err("per-tensor safetensors model; dense Gemma4 text models can be converted locally with `lma it -o NAME -ot models/model.gguf` (preview disk usage with -d); other architectures still require a compatible GGUF".into());
            }
            "application/vnd.ollama.image.adapter" => {
                return Err("adapter layers are not supported; use a standalone GGUF model".into());
            }
            other => {
                return Err(format!(
                    "unsupported layer mediaType '{other}'; safetensors, adapters, and sharded layouts are not supported"
                ));
            }
        };
        if selected.is_some() {
            return Err(format!(
                "multiple '{media_type}' layers; multi-model, sharded, or multiple-projector layouts are not supported"
            ));
        }
        *selected = Some(
            layer
                .get("digest")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("'{media_type}' layer is missing its digest string"))?,
        );
    }
    Ok(SelectedLayers {
        model: model.ok_or("no model layer; expected exactly one GGUF model")?,
        projector,
        provenance,
    })
}

fn validate_provenance(root: &Path, selected: &SelectedLayers<'_>) -> Result<(), String> {
    let Some(digest) = selected.provenance else {
        return Ok(());
    };
    let path = checked_blob_path(root, digest, "provenance manifest")?;
    let mut bytes = Vec::new();
    File::open(&path)
        .and_then(|file| file.take(MAX_PROVENANCE_BYTES + 1).read_to_end(&mut bytes))
        .map_err(|error| {
            format!(
                "cannot read provenance manifest '{}': {error}",
                path.display()
            )
        })?;
    if bytes.len() as u64 > MAX_PROVENANCE_BYTES {
        return Err(format!(
            "provenance manifest '{}' exceeds {MAX_PROVENANCE_BYTES} bytes",
            path.display()
        ));
    }
    let json: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid provenance manifest '{}': {error}", path.display()))?;
    if json.get("schemaVersion").and_then(Value::as_u64) != Some(2)
        || json.get("mediaType").and_then(Value::as_str) != Some(MANIFEST_MEDIA_TYPE)
    {
        return Err("provenance manifest must be a Docker schemaVersion 2 manifest".into());
    }
    // Reuse the inference-layer allowlist, but never follow nested chains or use
    // provenance to select additional weights. Ollama metadata remains ignored.
    let nested = selected_layers(&json).map_err(|error| {
        format!(
            "unsupported provenance manifest '{}': {error}",
            path.display()
        )
    })?;
    if nested.provenance.is_some() {
        return Err("provenance manifest must not contain another manifest layer".into());
    }
    if (nested.model, nested.projector) != (selected.model, selected.projector) {
        return Err(
            "provenance manifest model/projector digests must match the top-level manifest".into(),
        );
    }
    Ok(())
}

fn checked_blob_path(root: &Path, digest: &str, kind: &str) -> Result<PathBuf, String> {
    let hash = digest.strip_prefix("sha256:").filter(|hash| {
        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
    }).ok_or_else(|| format!("invalid {kind} digest '{digest}'; expected sha256: followed by 64 hexadecimal characters"))?;
    let path = root.join("blobs").join(format!("sha256-{hash}"));
    // Check before opening so a non-file (notably a FIFO) cannot block the caller.
    let metadata = fs::metadata(&path).map_err(|error| {
        format!(
            "cannot inspect Ollama {kind} blob '{}': {error}",
            path.display()
        )
    })?;
    if !metadata.is_file() {
        return Err(format!(
            "Ollama {kind} blob '{}' is not a file",
            path.display()
        ));
    }
    Ok(path)
}

fn checked_blob(root: &Path, digest: &str, kind: &str) -> Result<PathBuf, String> {
    let path = checked_blob_path(root, digest, kind)?;
    let mut file = File::open(&path).map_err(|error| {
        format!(
            "cannot open Ollama {kind} blob '{}': {error}",
            path.display()
        )
    })?;
    let mut magic = [0; 4];
    file.read_exact(&mut magic).map_err(|error| {
        format!(
            "cannot read GGUF magic from Ollama {kind} blob '{}': {error}",
            path.display()
        )
    })?;
    if &magic != b"GGUF" {
        return Err(format!(
            "Ollama {kind} blob '{}' is not GGUF; safetensors and other non-GGUF formats are not supported",
            path.display()
        ));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
    const MODEL_HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const PROJECTOR_HASH: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const PROVENANCE_HASH: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = env::temp_dir().join(format!(
                "lma-models-{}-{unique}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }

        fn manifest(&self, reference: &str, layers: Value) -> PathBuf {
            let reference = Reference::parse(reference).unwrap();
            let path = self
                .0
                .join("manifests")
                .join(reference.registry)
                .join(reference.namespace)
                .join(reference.model)
                .join(reference.tag);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(
                &path,
                serde_json::to_vec(&json!({"schemaVersion": 2, "layers": layers})).unwrap(),
            )
            .unwrap();
            path
        }

        fn blob(&self, hash: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join("blobs").join(format!("sha256-{hash}"));
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, bytes).unwrap();
            path
        }

        fn resolve(&self, input: &str) -> Result<ResolvedModel, String> {
            resolve_with_catalog(Path::new(input), Some(&self.0), ModelCatalog::default)
        }

        fn installed(&self, reference: &str) -> PathBuf {
            self.manifest(reference, json!([layer(MODEL_MEDIA_TYPE, MODEL_HASH)]));
            self.blob(MODEL_HASH, b"GGUF")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn layer(media_type: &str, hash: &str) -> Value {
        json!({"mediaType": media_type, "digest": format!("sha256:{hash}")})
    }

    fn catalog_model(id: &str, name: &str, model: &Path, ollama: bool) -> DiscoveredModel {
        DiscoveredModel {
            id: id.into(),
            name: name.into(),
            source: if ollama { "ollama" } else { "lma" }.into(),
            model: model.to_str().unwrap().into(),
            ollama,
            ollama_dir: None,
            projector: None,
            size_bytes: 0,
        }
    }

    fn resolve_catalog(
        fixture: &Fixture,
        input: &str,
        models: Vec<DiscoveredModel>,
    ) -> Result<ResolvedModel, String> {
        resolve_with_catalog(Path::new(input), Some(&fixture.0), || ModelCatalog {
            models,
            warnings: Vec::new(),
        })
    }

    #[test]
    fn registered_alias_selectors_normalize_before_ollama_fallback() {
        let fixture = Fixture::new();
        let ollama = fixture.installed("custom");
        let weights = fixture.0.join("custom.gguf");
        let projector = fixture.0.join("projector.gguf");
        let mut alias = catalog_model("lma:custom:latest", "custom:latest", &weights, false);
        alias.projector = Some(projector.clone());
        let collision = catalog_model("ollama:custom:latest", "custom:latest", &ollama, true);
        for input in [
            "custom",
            "custom:latest",
            "library/custom",
            "registry.ollama.ai/library/custom:latest",
            "lma:custom",
            "lma:custom:latest",
        ] {
            let resolved =
                resolve_catalog(&fixture, input, vec![collision.clone(), alias.clone()]).unwrap();
            assert_eq!(resolved.model, weights, "{input}");
            assert_eq!(resolved.projector, Some(projector.clone()), "{input}");
            assert_eq!(resolved.name.as_deref(), Some("custom:latest"));
            assert_eq!(resolved.registry_name.as_deref(), Some("custom:latest"));
        }
        let resolved = resolve_with_catalog(Path::new("ollama:custom"), Some(&fixture.0), || {
            panic!("explicit Ollama must not consult the local catalog")
        })
        .unwrap();
        assert_eq!(resolved.model, ollama);
        assert_eq!(resolved.registry_name, None);
    }

    #[test]
    fn normalized_alias_matching_preserves_namespaces_and_tags() {
        let fixture = Fixture::new();
        let weights = fixture.0.join("weights.gguf");
        let alias = catalog_model("lma:team/custom:v2", "team/custom:v2", &weights, false);
        for input in ["team/custom:v2", "lma:team/custom:v2"] {
            let resolved = resolve_catalog(&fixture, input, vec![alias.clone()]).unwrap();
            assert_eq!(resolved.registry_name.as_deref(), Some("team/custom:v2"));
        }
        for input in ["lma:custom:v2", "lma:team/custom", "lma:team/custom:v3"] {
            assert!(resolve_catalog(&fixture, input, vec![alias.clone()]).is_err());
        }
    }

    #[test]
    fn explicit_lma_never_falls_back_even_when_an_ollama_reference_exists() {
        let fixture = Fixture::new();
        fixture.installed("lma:missing");
        let error = resolve_catalog(&fixture, "lma:missing", Vec::new()).unwrap_err();
        assert!(error.contains("installed lma model"), "{error}");
        assert!(!error.contains("Ollama"), "{error}");
        for input in ["lma:", "lma:../missing", "lma:missing.gguf"] {
            assert!(resolve_catalog(&fixture, input, Vec::new()).is_err());
        }
        let alias = catalog_model(
            "lma:custom.gguf:latest",
            "custom.gguf:latest",
            &fixture.0.join("weights.gguf"),
            false,
        );
        assert!(resolve_catalog(&fixture, "lma:custom.gguf", vec![alias]).is_ok());
    }

    #[test]
    fn local_catalog_labels_do_not_imply_registry_provenance() {
        let fixture = Fixture::new();
        let weights = fixture.0.join("weights.gguf");
        let id = format!("lma:{}", weights.display());
        let local = catalog_model(&id, "custom:latest", &weights, false);
        let resolved = resolve_catalog(&fixture, "custom:latest", vec![local.clone()]).unwrap();
        assert_eq!(resolved.model, weights);
        assert_eq!(resolved.registry_name, None);
        let resolved = resolve_catalog(&fixture, &id, vec![local]).unwrap();
        assert_eq!(resolved.registry_name, None);
    }

    #[test]
    fn normalized_alias_collisions_remain_ambiguous() {
        let fixture = Fixture::new();
        let weights = fixture.0.join("weights.gguf");
        let alias = catalog_model("lma:custom:latest", "custom:latest", &weights, false);
        let local = catalog_model("lma:/elsewhere/custom.gguf", "custom.gguf", &weights, false);
        let error =
            resolve_catalog(&fixture, "custom", vec![alias.clone(), local.clone()]).unwrap_err();
        assert!(error.contains("ambiguous"), "{error}");
        assert_eq!(
            resolve_catalog(&fixture, "lma:custom", vec![alias, local])
                .unwrap()
                .registry_name
                .as_deref(),
            Some("custom:latest")
        );
    }

    #[test]
    fn existing_files_bypass_catalog_and_have_no_registry_provenance() {
        let fixture = Fixture::new();
        let weights = fixture.0.join("weights.gguf");
        fs::write(&weights, b"GGUF").unwrap();
        let resolved = resolve_with_catalog(&weights, None, || {
            panic!("existing files must not consult the catalog")
        })
        .unwrap();
        assert_eq!(resolved.model, weights);
        assert_eq!(resolved.registry_name, None);
    }

    #[test]
    fn discovery_registered_alias_resolves_with_isolated_roots() {
        let fixture = Fixture::new();
        let root = fixture.0.join("registry");
        let weights = fixture.0.join("weights.gguf");
        fs::write(&weights, b"GGUF").unwrap();
        registry::put(
            &root,
            "custom",
            registry::Entry {
                path: weights.clone(),
                projector: None,
            },
        )
        .unwrap();
        let resolved = resolve_with_catalog(Path::new("custom"), Some(&fixture.0), || {
            discover(DiscoveryOptions {
                local_dirs: vec![root],
                ollama_dir: Some(fixture.0.clone()),
            })
        })
        .unwrap();
        assert_eq!(resolved.model, fs::canonicalize(weights).unwrap());
        assert_eq!(resolved.registry_name.as_deref(), Some("custom:latest"));
    }

    #[test]
    fn existing_local_file_wins_without_reading_a_store_or_requiring_gguf() {
        let fixture = Fixture::new();
        let file = fixture.0.join("local.gguf");
        fs::write(&file, b"loader validates local files").unwrap();
        let original = fixture.0.join(".").join("local.gguf");
        let resolved = resolve(&original, Some(&fixture.0.join("absent"))).unwrap();
        assert_eq!(resolved.model.as_os_str(), original.as_os_str());
        assert_eq!(resolved.projector, None);
        assert_eq!(resolved.name, None);
    }

    #[cfg(unix)]
    #[test]
    fn local_files_can_have_scheme_like_names() {
        let fixture = Fixture::new();
        let path = fixture.0.join("ollama:model:latest");
        fs::write(&path, b"GGUF").unwrap();
        let resolved = resolve(&path, None).unwrap();
        assert_eq!(resolved.model.as_os_str(), path.as_os_str());
        assert_eq!(resolved.name, None);
    }

    // macOS filesystems reject these names with EILSEQ before resolution.
    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn local_files_preserve_non_utf8_path_bytes() {
        use std::os::unix::ffi::OsStringExt;
        let fixture = Fixture::new();
        let path = fixture
            .0
            .join(std::ffi::OsString::from_vec(b"non-utf8-\xff.gguf".to_vec()));
        fs::write(&path, b"GGUF").unwrap();
        let resolved = resolve(&path, None).unwrap();
        assert_eq!(resolved.model.as_os_str(), path.as_os_str());
        assert_eq!(resolved.name, None);
    }

    #[test]
    fn directories_and_obvious_missing_paths_are_not_model_references() {
        let fixture = Fixture::new();
        assert!(
            resolve(&fixture.0, None)
                .unwrap_err()
                .contains("not a file")
        );
        let absent = fixture.0.join("absent");
        assert!(
            resolve(&absent, None)
                .unwrap_err()
                .contains("does not exist")
        );
        for input in [
            "./missing",
            "../missing",
            "~/missing",
            "missing.GGUF",
            "models/missing.gguf",
            "C:\\models\\missing",
            "C:/models/missing",
            "missing/",
        ] {
            assert!(
                fixture
                    .resolve(input)
                    .unwrap_err()
                    .contains("does not exist"),
                "{input}"
            );
        }
    }

    #[test]
    fn plain_and_scheme_names_use_default_registry_namespace_and_latest() {
        let fixture = Fixture::new();
        let model = fixture.installed("llama3.2");
        for input in [
            "llama3.2",
            "llama3.2:latest",
            "ollama:llama3.2",
            "ollama:llama3.2:latest",
            "library/llama3.2",
            "registry.ollama.ai/library/llama3.2:latest",
        ] {
            let resolved = fixture.resolve(input).unwrap();
            assert_eq!(resolved.model, model);
            assert_eq!(resolved.projector, None);
            assert_eq!(resolved.name.as_deref(), Some("llama3.2:latest"));
        }
    }

    #[test]
    fn namespaced_and_explicit_registry_models_have_stable_ids() {
        let fixture = Fixture::new();
        for (input, expected) in [
            ("team/model:q4_K_M", "team/model:q4_K_M"),
            ("server/library/model", "server/library/model:latest"),
            (
                "registry.example:5000/team/model:v1",
                "registry.example:5000/team/model:v1",
            ),
            (
                "registry.example/model",
                "registry.example/library/model:latest",
            ),
            (
                "localhost:5000/library/model",
                "localhost:5000/library/model:latest",
            ),
        ] {
            let model = fixture.installed(input);
            let resolved = fixture.resolve(input).unwrap();
            assert_eq!(resolved.model, model);
            assert_eq!(resolved.name.as_deref(), Some(expected));
            let normalized = Reference::parse(expected).unwrap();
            assert_eq!(normalized.name(), expected);
        }
    }

    #[test]
    fn scheme_can_disambiguate_a_model_named_like_a_file() {
        let fixture = Fixture::new();
        fixture.installed("model.gguf");
        assert!(
            fixture
                .resolve("model.gguf")
                .unwrap_err()
                .contains("does not exist")
        );
        assert_eq!(
            fixture
                .resolve("ollama:model.gguf")
                .unwrap()
                .name
                .as_deref(),
            Some("model.gguf:latest")
        );
    }

    #[test]
    fn projector_is_resolved_alongside_model() {
        let fixture = Fixture::new();
        fixture.manifest(
            "vision",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(PROJECTOR_MEDIA_TYPE, PROJECTOR_HASH)
            ]),
        );
        let model = fixture.blob(MODEL_HASH, b"GGUFtiny fixture");
        let projector = fixture.blob(PROJECTOR_HASH, b"GGUF");
        let resolved = fixture.resolve("vision").unwrap();
        assert_eq!(resolved.model, model);
        assert_eq!(resolved.projector, Some(projector));
    }

    #[test]
    fn root_precedence_does_not_require_environment_mutation() {
        let explicit = Path::new("explicit-models");
        let env_root = OsStr::new("env-models");
        let home = OsStr::new("home");
        assert_eq!(
            choose_root(Some(explicit), Some(env_root), Some(home)).unwrap(),
            explicit
        );
        assert_eq!(
            choose_root(None, Some(env_root), Some(home)).unwrap(),
            Path::new("env-models")
        );
        assert_eq!(
            choose_root(None, None, Some(home)).unwrap(),
            Path::new("home/.ollama/models")
        );
        assert!(
            choose_root(None, None, None)
                .unwrap_err()
                .contains("no home directory")
        );
        assert!(
            choose_root(Some(Path::new("")), Some(env_root), Some(home))
                .unwrap_err()
                .contains("explicit root is empty")
        );
        assert!(
            choose_root(None, Some(OsStr::new("")), Some(home))
                .unwrap_err()
                .contains("OLLAMA_MODELS is empty")
        );
        assert!(choose_root(None, None, Some(OsStr::new(""))).is_err());
    }

    #[test]
    fn missing_or_non_directory_explicit_root_does_not_fall_back() {
        let fixture = Fixture::new();
        let absent = fixture.0.join("absent");
        assert!(
            resolve(Path::new("ollama:missing"), Some(&absent))
                .unwrap_err()
                .contains("cannot access Ollama models directory")
        );
        let file = fixture.0.join("not-a-directory");
        fs::write(&file, b"").unwrap();
        assert!(
            resolve(Path::new("ollama:missing"), Some(&file))
                .unwrap_err()
                .contains("not a directory")
        );
    }

    #[test]
    fn missing_manifest_reports_the_exact_store_without_downloading() {
        let fixture = Fixture::new();
        let error = fixture.resolve("ollama:missing:tag").unwrap_err();
        assert!(error.contains("manifests/registry.ollama.ai/library/missing/tag"));
        assert!(error.contains("nothing is downloaded"));
    }

    #[test]
    fn references_cannot_escape_manifest_directories() {
        for input in [
            "",
            ".",
            "..",
            "../model",
            "team/../model",
            "/model",
            "a/b/c/d",
            "team//model",
            "model:",
            "model:../tag",
            "model:tag:extra",
            "model\\escape",
            "model%2fescape",
            "model?query",
            "model#fragment",
            " model",
            "mødel",
            "https://host/model",
            "host:0/team/model",
            "host:65536/team/model",
            "host:abc/team/model",
            "host../team/model",
            "-host/team/model",
        ] {
            assert!(
                Reference::parse(input).is_err(),
                "accepted unsafe reference: {input}"
            );
        }
    }

    #[test]
    fn malformed_manifest_json_is_actionable() {
        let fixture = Fixture::new();
        let manifest = fixture.manifest("broken", json!([]));
        fs::write(manifest, b"{not json").unwrap();
        assert!(
            fixture
                .resolve("broken")
                .unwrap_err()
                .contains("invalid Ollama manifest")
        );
    }

    #[test]
    fn malformed_or_index_manifests_are_rejected() {
        for manifest in [
            Value::Null,
            json!({}),
            json!({"layers": {}}),
            json!({"layers": []}),
            json!({"manifests": [], "layers": []}),
            json!({"schemaVersion": 1, "layers": []}),
        ] {
            assert!(selected_layers(&manifest).is_err());
        }
    }

    #[test]
    fn unsupported_adapters_safetensors_and_shards_are_rejected() {
        let fixture = Fixture::new();
        for (media, expected) in [
            ("application/vnd.ollama.image.adapter", "adapter layers"),
            ("application/vnd.ollama.image.safetensors", "safetensors"),
            ("application/vnd.ollama.image.model.shard", "sharded"),
            ("application/octet-stream", "unsupported layer"),
        ] {
            fixture.manifest(
                "unsupported",
                json!([
                    layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                    layer(media, PROJECTOR_HASH)
                ]),
            );
            assert!(
                fixture
                    .resolve("unsupported")
                    .unwrap_err()
                    .contains(expected)
            );
        }
    }

    #[test]
    fn multiple_model_or_projector_layers_are_rejected() {
        let fixture = Fixture::new();
        for media in [MODEL_MEDIA_TYPE, PROJECTOR_MEDIA_TYPE] {
            fixture.manifest(
                "multiple",
                json!([
                    layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                    layer(media, PROJECTOR_HASH),
                    layer(media, MODEL_HASH)
                ]),
            );
            assert!(
                fixture
                    .resolve("multiple")
                    .unwrap_err()
                    .contains("multiple")
            );
        }
    }

    fn provenance_manifest(layers: Value) -> Value {
        json!({"schemaVersion": 2, "mediaType": MANIFEST_MEDIA_TYPE, "layers": layers})
    }

    fn install_provenance(fixture: &Fixture, nested: &Value) {
        fixture.manifest(
            "provenance",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH)
            ]),
        );
        fixture.blob(MODEL_HASH, b"GGUF");
        fixture.blob(PROVENANCE_HASH, &serde_json::to_vec(nested).unwrap());
    }

    #[test]
    fn pulsar_style_provenance_preserves_model_projector_and_tag() {
        let fixture = Fixture::new();
        let reference = "hf.co/mradermacher/Pulsar-16B-BF16-GGUF:Q4_K_M";
        for with_projector in [false, true] {
            let mut layers = vec![layer(MODEL_MEDIA_TYPE, MODEL_HASH)];
            if with_projector {
                layers.push(layer(PROJECTOR_MEDIA_TYPE, PROJECTOR_HASH));
                fixture.blob(PROJECTOR_HASH, b"GGUFprojector");
            }
            layers.push(layer("application/vnd.ollama.image.template", MODEL_HASH));
            layers.push(layer("application/vnd.ollama.image.params", MODEL_HASH));
            let nested = provenance_manifest(json!(layers));
            fixture.blob(PROVENANCE_HASH, &serde_json::to_vec(&nested).unwrap());
            // Provenance can precede the model; it never becomes an inference input.
            layers.insert(0, layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH));
            fixture.manifest(reference, json!(layers));
            let model = fixture.blob(MODEL_HASH, b"GGUFmodel");
            let resolved = fixture.resolve(reference).unwrap();
            assert_eq!(resolved.model, model);
            assert_eq!(resolved.name.as_deref(), Some(reference));
            assert_eq!(
                resolved.projector,
                with_projector.then(|| fixture
                    .0
                    .join("blobs")
                    .join(format!("sha256-{PROJECTOR_HASH}")))
            );
        }
    }

    #[test]
    fn provenance_model_and_projector_must_match_exactly() {
        let fixture = Fixture::new();
        for nested_layers in [
            json!([layer(MODEL_MEDIA_TYPE, PROJECTOR_HASH)]),
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(PROJECTOR_MEDIA_TYPE, PROJECTOR_HASH)
            ]),
        ] {
            install_provenance(&fixture, &provenance_manifest(nested_layers));
            assert!(
                fixture
                    .resolve("provenance")
                    .unwrap_err()
                    .contains("must match")
            );
        }
        for nested_layers in [
            json!([layer(MODEL_MEDIA_TYPE, MODEL_HASH)]),
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(PROJECTOR_MEDIA_TYPE, MODEL_HASH)
            ]),
        ] {
            install_provenance(&fixture, &provenance_manifest(nested_layers));
            fixture.manifest(
                "provenance",
                json!([
                    layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                    layer(PROJECTOR_MEDIA_TYPE, PROJECTOR_HASH),
                    layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH)
                ]),
            );
            assert!(
                fixture
                    .resolve("provenance")
                    .unwrap_err()
                    .contains("must match")
            );
        }
    }

    #[test]
    fn provenance_does_not_allow_unsupported_inference_layers_or_recursion() {
        let fixture = Fixture::new();
        for media in [
            "application/vnd.ollama.image.adapter",
            "application/vnd.ollama.image.safetensors",
            "application/vnd.ollama.image.model.shard",
            "application/vnd.docker.image.rootfs.diff.tar.gzip",
            "application/octet-stream",
            MODEL_MEDIA_TYPE,
            MANIFEST_MEDIA_TYPE,
        ] {
            install_provenance(
                &fixture,
                &provenance_manifest(json!([
                    layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                    layer(media, PROVENANCE_HASH)
                ])),
            );
            assert!(fixture.resolve("provenance").is_err(), "accepted {media}");
        }
        install_provenance(
            &fixture,
            &provenance_manifest(json!([layer(MODEL_MEDIA_TYPE, MODEL_HASH)])),
        );
        fixture.manifest(
            "provenance",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH),
                layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH)
            ]),
        );
        assert!(
            fixture
                .resolve("provenance")
                .unwrap_err()
                .contains("multiple")
        );
    }

    #[test]
    fn malformed_provenance_manifests_are_rejected() {
        let fixture = Fixture::new();
        let valid = provenance_manifest(json!([layer(MODEL_MEDIA_TYPE, MODEL_HASH)]));
        for nested in [
            Value::Null,
            json!({"layers": [layer(MODEL_MEDIA_TYPE, MODEL_HASH)]}),
            json!({"schemaVersion": 1, "mediaType": MANIFEST_MEDIA_TYPE, "layers": [layer(MODEL_MEDIA_TYPE, MODEL_HASH)]}),
            json!({"schemaVersion": 2, "mediaType": "application/octet-stream", "layers": [layer(MODEL_MEDIA_TYPE, MODEL_HASH)]}),
            json!({"schemaVersion": 2, "mediaType": MANIFEST_MEDIA_TYPE, "manifests": [], "layers": [layer(MODEL_MEDIA_TYPE, MODEL_HASH)]}),
            provenance_manifest(json!([])),
            provenance_manifest(json!([{"mediaType": MODEL_MEDIA_TYPE}])),
            provenance_manifest(json!([layer(MODEL_MEDIA_TYPE, MODEL_HASH), {}])),
        ] {
            install_provenance(&fixture, &nested);
            assert!(fixture.resolve("provenance").is_err(), "accepted {nested}");
        }
        install_provenance(&fixture, &valid);
        fixture.blob(PROVENANCE_HASH, b"{not json");
        assert!(
            fixture
                .resolve("provenance")
                .unwrap_err()
                .contains("invalid provenance manifest")
        );
        fixture.blob(
            PROVENANCE_HASH,
            &vec![b' '; MAX_PROVENANCE_BYTES as usize + 1],
        );
        assert!(
            fixture
                .resolve("provenance")
                .unwrap_err()
                .contains("exceeds")
        );
    }

    #[test]
    fn provenance_digest_and_file_are_checked_before_reading() {
        let fixture = Fixture::new();
        fixture.blob(MODEL_HASH, b"GGUF");
        for digest in [
            Value::Null,
            json!("sha256:../../escape"),
            json!("sha256:bad"),
        ] {
            fixture.manifest(
                "provenance",
                json!([
                    layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                    {"mediaType": MANIFEST_MEDIA_TYPE, "digest": digest}
                ]),
            );
            assert!(fixture.resolve("provenance").is_err());
        }
        fixture.manifest(
            "provenance",
            json!([
                layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                layer(MANIFEST_MEDIA_TYPE, PROVENANCE_HASH)
            ]),
        );
        assert!(
            fixture
                .resolve("provenance")
                .unwrap_err()
                .contains("cannot inspect")
        );
        let path = fixture
            .0
            .join("blobs")
            .join(format!("sha256-{PROVENANCE_HASH}"));
        fs::create_dir(&path).unwrap();
        assert!(
            fixture
                .resolve("provenance")
                .unwrap_err()
                .contains("not a file")
        );
    }

    #[test]
    fn metadata_layers_are_not_loaded_or_applied() {
        let fixture = Fixture::new();
        let mut layers = vec![layer(MODEL_MEDIA_TYPE, MODEL_HASH)];
        for kind in ["template", "system", "params", "messages", "license"] {
            // Deliberately absent blobs and invalid digests: metadata is not loaded.
            layers.push(json!({"mediaType": format!("application/vnd.ollama.image.{kind}"), "digest": "not-selected"}));
        }
        fixture.manifest("metadata", json!(layers));
        let model = fixture.blob(MODEL_HASH, b"GGUF");
        assert_eq!(fixture.resolve("metadata").unwrap().model, model);
    }

    #[test]
    fn selected_layers_require_string_media_type_and_digest() {
        for layers in [
            json!([{}]),
            json!([{"mediaType": 1}]),
            json!([{"mediaType": MODEL_MEDIA_TYPE}]),
            json!([{"mediaType": MODEL_MEDIA_TYPE, "digest": 12}]),
        ] {
            assert!(selected_layers(&json!({"layers": layers})).is_err());
        }
    }

    #[test]
    fn each_selected_digest_is_checked_before_use() {
        let fixture = Fixture::new();
        fixture.blob(MODEL_HASH, b"GGUF");
        for digest in [
            "",
            "sha256:..",
            "sha256:../../escape",
            "sha512:012345",
            "sha256:abc",
            &format!("sha256:{}", "g".repeat(64)),
            &format!("sha256:{}", "0".repeat(65)),
        ] {
            for media in [MODEL_MEDIA_TYPE, PROJECTOR_MEDIA_TYPE] {
                let bad = json!({"mediaType": media, "digest": digest});
                let layers = if media == MODEL_MEDIA_TYPE {
                    json!([bad])
                } else {
                    json!([layer(MODEL_MEDIA_TYPE, MODEL_HASH), bad])
                };
                fixture.manifest("bad-digest", layers);
                assert!(
                    fixture
                        .resolve("bad-digest")
                        .unwrap_err()
                        .contains("invalid"),
                    "{digest}"
                );
            }
        }
    }

    #[test]
    fn hexadecimal_digest_case_is_preserved_for_blob_lookup() {
        let fixture = Fixture::new();
        let hash = MODEL_HASH.to_ascii_uppercase();
        fixture.manifest("uppercase", json!([layer(MODEL_MEDIA_TYPE, &hash)]));
        let model = fixture.blob(&hash, b"GGUF");
        assert_eq!(fixture.resolve("uppercase").unwrap().model, model);
    }

    #[test]
    fn missing_truncated_and_non_gguf_blobs_are_rejected_for_both_roles() {
        let fixture = Fixture::new();
        for media in [MODEL_MEDIA_TYPE, PROJECTOR_MEDIA_TYPE] {
            let layers = if media == MODEL_MEDIA_TYPE {
                json!([layer(media, PROJECTOR_HASH)])
            } else {
                fixture.blob(MODEL_HASH, b"GGUF");
                json!([
                    layer(MODEL_MEDIA_TYPE, MODEL_HASH),
                    layer(media, PROJECTOR_HASH)
                ])
            };
            fixture.manifest("bad-blob", layers);
            let path = fixture
                .0
                .join("blobs")
                .join(format!("sha256-{PROJECTOR_HASH}"));
            if path.exists() {
                fs::remove_file(&path).unwrap();
            }
            assert!(
                fixture
                    .resolve("bad-blob")
                    .unwrap_err()
                    .contains("cannot inspect")
            );
            fixture.blob(PROJECTOR_HASH, b"GG");
            assert!(
                fixture
                    .resolve("bad-blob")
                    .unwrap_err()
                    .contains("GGUF magic")
            );
            fixture.blob(PROJECTOR_HASH, b"not GGUF, including safetensors");
            assert!(
                fixture
                    .resolve("bad-blob")
                    .unwrap_err()
                    .contains("not GGUF")
            );
            fs::remove_file(&path).unwrap();
            fs::create_dir_all(&path).unwrap();
            let error = fixture.resolve("bad-blob").unwrap_err();
            assert!(error.contains("not a file") || error.contains("cannot open"));
            fs::remove_dir(&path).unwrap();
        }
    }
}
