//! Source-specific, local model deletion. Callers must unload active models first.
//!
//! Store roots/parents are trusted and must not be concurrently renamed or
//! modified by another process. Within that contract, paths below the selected
//! root are checked without following symlinks. No arbitrary directory trees,
//! projectors, configs, or unrelated Ollama blobs are removed.
use crate::models::{self, DiscoveryOptions, discovery, gguf, registry};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    sync::Mutex,
};

static LOCK: Mutex<()> = Mutex::new(());
const MAX_ROOTS: usize = 128;
const MAX_ENTRIES: usize = 10_000;
const MAX_DEPTH: usize = 8;
const JSON_LIMIT: u64 = 1024 * 1024;
const SCAN_BYTES: u64 = 64 * 1024 * 1024;
const MAX_REFS: usize = 100_000;

/// Delete one installed model from exactly one source. `source`, when present,
/// must be `lma`, `ollama`, or `lm-studio`. Ambiguous names never delete anything.
/// Accepts a discovered label or ID, not an arbitrary filesystem deletion path.
/// This synchronous API is suitable for a host's `spawn_blocking`.
pub fn run(model: &str, source: Option<&str>, ollama_dir: Option<&Path>) -> Result<Value, String> {
    run_in_locations(
        model,
        source,
        DiscoveryOptions {
            local_dirs: discovery::default_local_dirs(),
            ollama_dir: ollama_dir.map(Path::to_path_buf),
        },
        discovery::lm_studio_dirs(),
    )
}

/// Delete within an application's trusted, configured discovery locations.
/// Locations must come from host settings, never an arbitrary delete request.
/// All source boundaries, reference checks and ambiguity safeguards still apply.
/// Callers must exclude active models and coordinate discovery/store mutations.
pub fn run_in_locations(
    model: &str,
    source: Option<&str>,
    options: DiscoveryOptions,
    lm_dirs: Vec<PathBuf>,
) -> Result<Value, String> {
    let _lock = LOCK.lock().map_err(|_| "model deletion lock poisoned")?;
    run_with(model, source, options, lm_dirs)
}

#[derive(Clone)]
enum Kind {
    Alias {
        root: PathBuf,
        label: String,
        entry: registry::Entry,
    },
    Local {
        path: PathBuf,
    },
    Ollama {
        root: PathBuf,
        label: String,
    },
}
#[derive(Clone)]
struct Candidate {
    label: String,
    id: String,
    source: &'static str,
    kind: Kind,
}
struct Context {
    roots: Vec<PathBuf>,
    lm_roots: Vec<PathBuf>,
    registry_roots: Vec<PathBuf>,
    ollama_root: Option<PathBuf>,
}
fn canonical_roots(paths: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    if paths.len() > MAX_ROOTS {
        return Err("model deletion root limit exceeded".into());
    }
    let mut roots = BTreeSet::new();
    for path in paths {
        match fs::canonicalize(path) {
            Ok(path) if path.is_dir() => {
                roots.insert(path);
            }
            Ok(_) => return Err("model deletion root must be a directory".into()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("cannot establish model deletion boundary: {e}")),
        }
    }
    Ok(roots.into_iter().collect())
}
fn normalized(label: &str) -> Option<String> {
    let label = label.strip_prefix("ollama:").unwrap_or(label);
    let label = gguf::shard_name(label).map_or(label, |(prefix, _, _)| prefix);
    let label = label.strip_suffix(".gguf").unwrap_or(label);
    registry::name(label).ok()
}
fn matches(candidate: &Candidate, input: &str) -> bool {
    if input.starts_with("lma:") {
        return candidate.id == input;
    }
    if candidate.id == input || candidate.label == input {
        return true;
    }
    let input_forms = [
        registry::name(input.strip_prefix("ollama:").unwrap_or(input)).ok(),
        normalized(input),
    ];
    let candidate_forms = [
        registry::name(&candidate.label).ok(),
        normalized(&candidate.label),
    ];
    input_forms
        .into_iter()
        .flatten()
        .any(|name| candidate_forms.iter().flatten().any(|other| other == &name))
}
fn boundary<'a>(path: &Path, roots: &'a [PathBuf]) -> Option<&'a Path> {
    roots
        .iter()
        .filter(|root| path.starts_with(root))
        .max_by_key(|root| root.components().count())
        .map(PathBuf::as_path)
}
fn regular(path: &Path, root: &Path) -> Result<u64, String> {
    let relative = path.strip_prefix(root).map_err(|_| {
        format!(
            "refusing deletion outside selected source root: {}",
            path.display()
        )
    })?;
    if relative.as_os_str().is_empty() {
        return Err("refusing deletion of a store directory".into());
    }
    let mut current = root.to_path_buf();
    let components: Vec<_> = relative.components().collect();
    for (i, part) in components.iter().enumerate() {
        if !matches!(part, Component::Normal(_)) {
            return Err("unsafe deletion path component".into());
        }
        current.push(part.as_os_str());
        let meta = fs::symlink_metadata(&current)
            .map_err(|e| format!("cannot validate {}: {e}", current.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!(
                "refusing deletion through symlink: {}",
                current.display()
            ));
        }
        if i + 1 == components.len() {
            if !meta.is_file() {
                return Err("deletion target must be a regular file".into());
            }
            return Ok(meta.len());
        }
        if !meta.is_dir() {
            return Err("deletion path parent must be a regular directory".into());
        }
    }
    Err("empty deletion target".into())
}
fn read_json(path: &Path, root: &Path, limit: u64) -> Result<Value, String> {
    if regular(path, root)? > limit {
        return Err(format!("JSON deletion metadata exceeds {limit} bytes"));
    }
    let mut bytes = Vec::new();
    crate::local_file::open_regular(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("JSON deletion metadata grew beyond its limit".into());
    }
    serde_json::from_slice::<StrictJson>(&bytes)
        .map(|v| v.0)
        .map_err(|e| format!("invalid deletion metadata {}: {e}", path.display()))
}

// Duplicate object keys must not hide references from the conservative GC scan.
struct StrictJson(Value);
impl<'de> serde::Deserialize<'de> for StrictJson {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = StrictJson;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON without duplicate object keys")
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::Null))
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::Bool(v)))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(StrictJson(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(StrictJson(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| StrictJson(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite JSON number"))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::String(v.into())))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::String(v)))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(v) = a.next_element::<StrictJson>()? {
                    values.push(v.0);
                }
                Ok(StrictJson(Value::Array(values)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((k, v)) = a.next_entry::<String, StrictJson>()? {
                    if values.insert(k, v.0).is_some() {
                        return Err(serde::de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(StrictJson(Value::Object(values)))
            }
        }
        d.deserialize_any(Visitor)
    }
}

fn run_with(
    model: &str,
    source: Option<&str>,
    options: DiscoveryOptions,
    lm_dirs: Vec<PathBuf>,
) -> Result<Value, String> {
    if model.is_empty() || model.len() > 4096 || model.chars().any(char::is_control) {
        return Err("invalid installed model label".into());
    }
    if source.is_some_and(|s| !["lma", "ollama", "lm-studio"].contains(&s)) {
        return Err("delete source must be lma, ollama, or lm-studio".into());
    }
    let roots = canonical_roots(&options.local_dirs)?;
    let ollama_root = models::default_root(options.ollama_dir.as_deref())
        .ok()
        .and_then(|p| fs::canonicalize(p).ok());
    let context = Context {
        roots: roots.clone(),
        lm_roots: canonical_roots(&lm_dirs)?,
        registry_roots: roots,
        ollama_root,
    };
    let catalog = models::discover(options.clone());
    // The catalog does not expose a completeness bit. Do not resolve destructive
    // intent from a truncated traversal which could hide another matching model.
    if source != Some("ollama")
        && catalog.warnings.iter().any(|w| {
            w.contains("limit reached")
                || w.contains("cannot scan")
                || w.contains("root limit")
                || w.contains("warnings suppressed")
                || w.contains("cannot read directory entry")
                || w.contains("cannot inspect")
        })
    {
        return Err("installed model discovery was incomplete; narrow the configured model roots before deleting".into());
    }
    let mut warnings = catalog.warnings;
    let mut candidates = Vec::new();
    let mut aliased = BTreeSet::new();
    for root in &context.registry_roots {
        match registry::read(root) {
            Ok(entries) => {
                for (label, entry) in entries {
                    if candidates.len() >= MAX_ENTRIES {
                        return Err("model deletion catalog limit exceeded".into());
                    }
                    if let Ok(path) = fs::canonicalize(&entry.path) {
                        aliased.insert(path);
                    }
                    candidates.push(Candidate {
                        id: format!("lma:{label}"),
                        label: label.clone(),
                        source: "lma",
                        kind: Kind::Alias {
                            root: root.clone(),
                            label,
                            entry,
                        },
                    });
                }
            }
            Err(e) => warnings.push(format!("registry {}: {e}", root.display())),
        }
    }
    let alias_ids: BTreeSet<_> = candidates
        .iter()
        .map(|candidate| candidate.id.clone())
        .collect();
    let mut local_paths = BTreeSet::new();
    for mut item in catalog.models {
        let (source, kind) = if item.ollama {
            let root = item
                .ollama_dir
                .ok_or("discovered Ollama model has no source root")?;
            (
                "ollama",
                Kind::Ollama {
                    root,
                    label: item.name.clone(),
                },
            )
        } else {
            let path = PathBuf::from(item.model);
            let source = if boundary(&path, &context.lm_roots).is_some() {
                "lm-studio"
            } else {
                "lma"
            };
            if source == "lma" && aliased.contains(&path) {
                continue;
            }
            if !local_paths.insert(path.clone()) {
                continue;
            }
            // Displayed provenance never turns a registry ID into permission to
            // delete its external weights. Keep a distinct physical-file ID.
            if alias_ids.contains(&item.id) {
                item.id = format!("lma:{}", path.display());
            }
            (source, Kind::Local { path })
        };
        candidates.push(Candidate {
            label: item.name,
            id: item.id,
            source,
            kind,
        });
        if candidates.len() > MAX_ENTRIES {
            return Err("model deletion catalog limit exceeded".into());
        }
    }
    // Discovery intentionally lists only runnable models. A selected installed
    // manifest can still be deleted when its format is unsupported for inference.
    if let Ok(label) = registry::name(model.strip_prefix("ollama:").unwrap_or(model)) {
        if let Ok((root, manifest, label)) =
            models::import_source(&label, options.ollama_dir.as_deref())
        {
            if fs::symlink_metadata(&manifest).is_ok() && !candidates.iter().any(|c| matches!(&c.kind, Kind::Ollama { root: r, label: l } if *r == root && *l == label)) {
                candidates.push(Candidate { id: format!("ollama:{label}"), label: label.clone(), source: "ollama", kind: Kind::Ollama { root, label } });
            }
        }
    }
    let found: Vec<_> = candidates
        .iter()
        .filter(|c| source.is_none_or(|s| c.source == s) && matches(c, model))
        .collect();
    if found.is_empty() {
        return Err(format!(
            "installed model '{model}' was not found in the selected source"
        ));
    }
    if found.len() != 1 {
        let sources: BTreeSet<_> = found.iter().map(|c| c.source).collect();
        return Err(format!(
            "installed model '{model}' is ambiguous ({}); specify source lma, ollama, or lm-studio, and use an exact label/ID if names still collide",
            sources.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    let selected = found[0];
    let references = registry_references(&context)?;
    match &selected.kind {
        Kind::Ollama { root, label } => delete_ollama(selected, root, label, &references, warnings),
        Kind::Local { path } => {
            let files = local_files(path, selected.source, &context)?;
            let mut paths = files.clone();
            paths.extend(receipts(&files, &mut warnings));
            let staged = Staged::new(paths)?;
            let (deleted, bytes, unlinked) = staged.commit(&mut warnings);
            let cleanup = cleanup_registry(&references, &files, &unlinked, &mut warnings);
            Ok(result(
                selected,
                deleted,
                bytes,
                Vec::new(),
                false,
                None,
                cleanup,
                warnings,
            ))
        }
        Kind::Alias { root, label, entry } => {
            let mut retained = Vec::new();
            let mut assets = Vec::new();
            let staged = registry::remove_with(root, label, entry, |_| {
                let path = match fs::canonicalize(&entry.path) {
                    Ok(path) => path,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        return Staged::new(Vec::new());
                    }
                    Err(e) => return Err(format!("cannot resolve alias target: {e}")),
                };
                // A registry alias does not confer ownership of arbitrary external
                // files, LM Studio assets, or blobs in an Ollama store.
                if boundary(&path, &context.roots).is_none()
                    || boundary(&path, &context.lm_roots).is_some()
                    || context
                        .ollama_root
                        .as_ref()
                        .is_some_and(|root| path.starts_with(root))
                {
                    retained.push(path);
                    warnings.push("external alias target retained; select its external source to delete the actual model".into());
                    return Staged::new(Vec::new());
                }
                let files = local_files(&path, "lma", &context)?;
                if shared_aliases(&context, &files, Some((root, label)), &mut warnings) {
                    retained.extend(files);
                    return Staged::new(Vec::new());
                }
                let mut paths = files.clone();
                paths.extend(receipts(&files, &mut warnings));
                assets = files;
                Staged::new(paths)
            })?;
            let (deleted, bytes, unlinked) = staged.commit(&mut warnings);
            let cleanup = cleanup_registry(&references, &assets, &unlinked, &mut warnings);
            let obsolete = format!(
                "registered model '{label}' is missing at '{}':",
                entry.path.display()
            );
            warnings.retain(|warning| !warning.starts_with(&obsolete));
            Ok(result(
                selected, deleted, bytes, retained, true, None, cleanup, warnings,
            ))
        }
    }
}
fn result(
    selected: &Candidate,
    deleted: Vec<PathBuf>,
    bytes: u64,
    retained: Vec<PathBuf>,
    alias: bool,
    gc: Option<bool>,
    mut cleanup: RegistryCleanup,
    warnings: Vec<String>,
) -> Value {
    if alias {
        if let Kind::Alias { root, label, .. } = &selected.kind {
            cleanup.deleted.push(deleted_alias(root, label));
        }
    }
    json!({"status":"success","model":selected.label,"source":selected.source,"aliasDeleted":alias,"deletedAliases":cleanup.deleted,"registryCleanupComplete":cleanup.complete,"deletedFiles":deleted,"deletedBytes":bytes,"retainedFiles":retained,"blobGcComplete":gc,"warnings":warnings})
}

type RegistryReferences = BTreeMap<PathBuf, BTreeMap<String, (registry::Entry, PathBuf)>>;

fn registry_references(context: &Context) -> Result<RegistryReferences, String> {
    let mut references = BTreeMap::new();
    let mut count = 0;
    for root in &context.registry_roots {
        let mut targets = BTreeMap::new();
        for (label, entry) in registry::read(root)? {
            count += 1;
            if count > MAX_REFS {
                return Err("registry reference snapshot limit exceeded".into());
            }
            match fs::canonicalize(&entry.path) {
                Ok(target) => {
                    targets.insert(label, (entry, target));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("cannot snapshot registry target: {e}")),
            }
        }
        references.insert(root.clone(), targets);
    }
    Ok(references)
}

struct RegistryCleanup {
    deleted: Vec<Value>,
    complete: bool,
}
fn deleted_alias(root: &Path, label: &str) -> Value {
    json!({"root":root,"alias":label,"id":format!("lma:{label}")})
}
fn cleanup_registry(
    references: &RegistryReferences,
    assets: &[PathBuf],
    unlinked: &[PathBuf],
    warnings: &mut Vec<String>,
) -> RegistryCleanup {
    let unlinked: BTreeSet<_> = unlinked.iter().collect();
    let removed: BTreeSet<_> = assets.iter().filter(|p| unlinked.contains(p)).collect();
    let mut cleanup = RegistryCleanup {
        deleted: Vec::new(),
        complete: true,
    };
    for (root, entries) in references {
        let expected = entries
            .iter()
            .filter(|(_, (_, target))| removed.contains(target))
            .map(|(label, reference)| (label.clone(), reference.clone()))
            .collect::<BTreeMap<_, _>>();
        if expected.is_empty() {
            continue;
        }
        match registry::remove_missing(root, &expected) {
            Ok(labels) => cleanup
                .deleted
                .extend(labels.iter().map(|label| deleted_alias(root, label))),
            Err(e) => {
                cleanup.complete = false;
                warnings.push(format!(
                    "model files were deleted, but registry cleanup is incomplete at {}: {e}",
                    root.display()
                ));
            }
        }
    }
    cleanup
}

fn local_files(path: &Path, source: &str, context: &Context) -> Result<Vec<PathBuf>, String> {
    let roots = if source == "lm-studio" {
        &context.lm_roots
    } else {
        &context.roots
    };
    let root = boundary(path, roots)
        .ok_or("refusing deletion of a model outside known selected source paths")?;
    if source == "lma" && boundary(path, &context.lm_roots).is_some() {
        return Err("LM Studio weights require source lm-studio".into());
    }
    if context
        .ollama_root
        .as_ref()
        .is_some_and(|root| path.starts_with(root))
    {
        return Err("Ollama source files must be deleted by manifest, not as local GGUFs".into());
    }
    regular(path, root)?;
    let mut magic = [0; 4];
    crate::local_file::open_regular(path)?
        .read_exact(&mut magic)
        .map_err(|e| e.to_string())?;
    if magic != *b"GGUF" {
        return Err("selected local model is not a GGUF file".into());
    }
    let mut files = Vec::new();
    if let Some((prefix, _, total)) = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(gguf::shard_name)
    {
        if total > 256 {
            return Err("split model exceeds 256 shards".into());
        }
        let parent = path.parent().ok_or("split model has no parent")?;
        for part in 1..=total {
            let path = parent.join(format!("{prefix}-{part:05}-of-{total:05}.gguf"));
            regular(&path, root)?;
            files.push(path);
        }
        gguf::shards(path)?.ok_or("split model validation failed")?;
    } else {
        if gguf::inspect(path).is_ok_and(|info| {
            info.split_count.is_some_and(|n| n > 1) || info.split_index.is_some_and(|n| n != 0)
        }) {
            return Err(
                "split metadata requires a complete, conventionally named shard set".into(),
            );
        }
        files.push(path.to_path_buf());
    }
    Ok(files)
}
fn shared_aliases(
    context: &Context,
    files: &[PathBuf],
    except: Option<(&Path, &str)>,
    warnings: &mut Vec<String>,
) -> bool {
    let mut shared = false;
    for root in &context.registry_roots {
        let entries = match registry::read(root) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(format!(
                    "retained weights because alias reference scan failed: {e}"
                ));
                shared = true;
                continue;
            }
        };
        for (label, entry) in entries {
            if except.is_some_and(|(r, l)| r == root && l == label) {
                continue;
            }
            for target in std::iter::once(entry.path).chain(entry.projector) {
                match fs::canonicalize(&target) {
                    Ok(path) if files.contains(&path) => {
                        warnings.push(format!("retained weights shared with lma alias '{label}'"));
                        shared = true;
                    }
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                        warnings.push(format!(
                            "retained weights because alias target could not be checked: {e}"
                        ));
                        shared = true;
                    }
                    _ => {}
                }
            }
        }
    }
    shared
}
fn receipts(files: &[PathBuf], warnings: &mut Vec<String>) -> Vec<PathBuf> {
    let parents: BTreeSet<_> = files.iter().filter_map(|p| p.parent()).collect();
    let mut found = Vec::new();
    let mut count = 0;
    for parent in parents {
        let entries = match fs::read_dir(parent) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(format!("import receipts retained: {e}"));
                return Vec::new();
            }
        };
        for entry in entries {
            count += 1;
            if count > MAX_ENTRIES {
                warnings.push("import receipts retained: directory scan limit exceeded".into());
                return Vec::new();
            }
            let Ok(entry) = entry else {
                warnings.push("unreadable import receipts retained".into());
                return Vec::new();
            };
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(fingerprint) = name
                .strip_prefix(".lma-import-v1-")
                .and_then(|s| s.strip_suffix(".json"))
            else {
                continue;
            };
            if fingerprint.len() != 64 || !fingerprint.bytes().all(|b| b.is_ascii_hexdigit()) {
                continue;
            }
            let Ok(v) = read_json(&entry.path(), parent, 65536) else {
                continue;
            };
            if v["version"] != 1 || v["fingerprint"].as_str() != Some(fingerprint) {
                continue;
            }
            let Some(output) = v["output"]
                .as_str()
                .map(Path::new)
                .filter(|p| p.is_absolute())
            else {
                continue;
            };
            if fs::canonicalize(output).is_ok_and(|p| files.contains(&p)) {
                found.push(entry.path());
            }
        }
    }
    found
}

fn digest(text: &str) -> Result<String, String> {
    let hex = text
        .strip_prefix("sha256:")
        .ok_or("unsupported Ollama digest algorithm")?;
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid Ollama SHA-256 digest".into());
    }
    Ok(hex.to_owned())
}
struct Budget {
    entries: usize,
    bytes: u64,
    nodes: usize,
}
impl Budget {
    fn new() -> Self {
        Self {
            entries: 0,
            bytes: 0,
            nodes: 0,
        }
    }
}
fn manifest(path: &Path, root: &Path, budget: &mut Budget) -> Result<Value, String> {
    let len = regular(path, root)?;
    budget.bytes = budget
        .bytes
        .checked_add(len)
        .filter(|n| *n <= SCAN_BYTES)
        .ok_or("Ollama reference scan exceeds 64 MiB")?;
    let v = read_json(path, root, JSON_LIMIT)?;
    if !v.is_object()
        || v["schemaVersion"] != 2
        || (!v["layers"].is_array() && !v["manifests"].is_array())
    {
        return Err("unrecognized Ollama manifest; blob references cannot be established".into());
    }
    Ok(v)
}
fn collect(
    v: &Value,
    refs: &mut BTreeSet<String>,
    nested: &mut BTreeSet<String>,
    budget: &mut Budget,
) -> Result<(), String> {
    budget.nodes += 1;
    if budget.nodes > 1_000_000 {
        return Err("Ollama reference node limit exceeded".into());
    }
    match v {
        Value::String(s) if s.starts_with("sha256:") => {
            refs.insert(digest(s)?);
        }
        Value::Array(a) => {
            for v in a {
                collect(v, refs, nested, budget)?;
            }
        }
        Value::Object(o) => {
            if let Some(d) = o.get("digest") {
                refs.insert(digest(d.as_str().ok_or("non-string Ollama digest")?)?);
            }
            if o.get("mediaType")
                .and_then(Value::as_str)
                .is_some_and(|m| m.contains("manifest") || m.ends_with("image.index.v1+json"))
            {
                if let Some(d) = o.get("digest") {
                    nested.insert(digest(d.as_str().ok_or("invalid nested manifest digest")?)?);
                }
            }
            for (key, v) in o {
                if key.starts_with("sha256:") {
                    refs.insert(digest(key)?);
                }
                collect(v, refs, nested, budget)?;
            }
        }
        _ => {}
    }
    if refs.len() > MAX_REFS || nested.len() > MAX_REFS {
        return Err("Ollama reference digest limit exceeded".into());
    }
    Ok(())
}
fn references(
    root: &Path,
    initial: &Value,
    budget: &mut Budget,
) -> Result<BTreeSet<String>, String> {
    let mut refs = BTreeSet::new();
    let mut pending = BTreeSet::new();
    let mut visited = BTreeSet::new();
    collect(initial, &mut refs, &mut pending, budget)?;
    while let Some(d) = pending.pop_first() {
        if !visited.insert(d.clone()) {
            continue;
        }
        if visited.len() > MAX_ENTRIES {
            return Err("nested Ollama manifest limit exceeded".into());
        }
        let v = manifest(
            &root.join("blobs").join(format!("sha256-{d}")),
            root,
            budget,
        )?;
        collect(&v, &mut refs, &mut pending, budget)?;
    }
    Ok(refs)
}
fn scan_manifests(
    directory: &Path,
    selected: &Path,
    root: &Path,
    depth: usize,
    budget: &mut Budget,
    alive: &mut BTreeSet<String>,
) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err("Ollama manifest depth limit exceeded".into());
    }
    let meta = fs::symlink_metadata(directory).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err("Ollama manifest traversal contains a symlink or non-directory".into());
    }
    for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
        budget.entries += 1;
        if budget.entries > MAX_ENTRIES {
            return Err("Ollama manifest entry limit exceeded".into());
        }
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        if kind.is_symlink() {
            return Err("Ollama manifest reference scan encountered a symlink".into());
        }
        if path == selected {
            continue;
        }
        if kind.is_dir() {
            scan_manifests(&path, selected, root, depth + 1, budget, alive)?;
        } else if kind.is_file() {
            let v = manifest(&path, root, budget)?;
            alive.extend(references(root, &v, budget)?);
            if alive.len() > MAX_REFS {
                return Err("Ollama live reference limit exceeded".into());
            }
        } else {
            return Err("Ollama manifest scan encountered a special file".into());
        }
    }
    Ok(())
}
fn delete_ollama(
    selected: &Candidate,
    root: &Path,
    label: &str,
    registry_refs: &RegistryReferences,
    mut warnings: Vec<String>,
) -> Result<Value, String> {
    let (resolved_root, path, normalized) = models::import_source(label, Some(root))?;
    if resolved_root != root || normalized != label {
        return Err("Ollama deletion source changed".into());
    }
    let mut budget = Budget::new();
    let selected_json = manifest(&path, root, &mut budget)?;
    let gc = (|| -> Result<(Vec<PathBuf>, Vec<PathBuf>), String> {
        let candidates = references(root, &selected_json, &mut budget)?;
        let mut alive = BTreeSet::new();
        scan_manifests(
            &root.join("manifests"),
            &path,
            root,
            0,
            &mut budget,
            &mut alive,
        )?;
        let mut remove = Vec::new();
        let mut retained = Vec::new();
        for d in candidates {
            let blob = root.join("blobs").join(format!("sha256-{d}"));
            match fs::symlink_metadata(&blob) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.to_string()),
                _ => {
                    regular(&blob, root)?;
                }
            }
            if alive.contains(&d) {
                retained.push(blob);
            } else {
                remove.push(blob);
            }
        }
        Ok((remove, retained))
    })();
    let (mut files, retained, complete) = match gc {
        Ok((files, retained)) => (files, retained, true),
        Err(e) => {
            warnings.push(format!("manifest will be deleted, but all source blobs are retained because reference scan was incomplete/unsafe: {e}"));
            (Vec::new(), Vec::new(), false)
        }
    };
    let assets = files.clone();
    // Hide the selected manifest before its blobs; rollback restores it last.
    files.insert(0, path);
    let staged = Staged::new(files)?;
    let (deleted, bytes, unlinked) = staged.commit(&mut warnings);
    let cleanup = cleanup_registry(registry_refs, &assets, &unlinked, &mut warnings);
    Ok(result(
        selected,
        deleted,
        bytes,
        retained,
        false,
        Some(complete),
        cleanup,
        warnings,
    ))
}

struct Staged {
    files: Vec<(PathBuf, PathBuf, u64)>,
    dirs: Vec<tempfile::TempDir>,
    committed: bool,
}
impl Staged {
    fn new(paths: Vec<PathBuf>) -> Result<Self, String> {
        let mut seen = BTreeSet::new();
        let paths: Vec<_> = paths
            .into_iter()
            .filter(|path| seen.insert(path.clone()))
            .collect();
        if paths.len() > MAX_REFS {
            return Err("model deletion file limit exceeded".into());
        }
        // Validate the entire intent before moving even the first selected file.
        for path in &paths {
            regular(path, path.parent().ok_or("deletion target has no parent")?)?;
        }
        let mut stage = Self {
            files: Vec::new(),
            dirs: Vec::new(),
            committed: false,
        };
        let mut parents = BTreeMap::new();
        for path in paths {
            let parent = path.parent().unwrap().to_path_buf();
            let index = if let Some(index) = parents.get(&parent) {
                *index
            } else {
                let index = stage.dirs.len();
                let dir = tempfile::Builder::new()
                    .prefix(".lma-delete-")
                    .tempdir_in(&parent)
                    .map_err(|e| e.to_string())?;
                stage.dirs.push(dir);
                parents.insert(parent, index);
                index
            };
            let destination = stage.dirs[index].path().join(stage.files.len().to_string());
            let bytes = fs::symlink_metadata(&path)
                .map_err(|e| e.to_string())?
                .len();
            fs::rename(&path, &destination)
                .map_err(|e| format!("cannot stage {} for deletion: {e}", path.display()))?;
            stage.files.push((path, destination, bytes));
        }
        Ok(stage)
    }
    fn commit(mut self, warnings: &mut Vec<String>) -> (Vec<PathBuf>, u64, Vec<PathBuf>) {
        self.committed = true;
        // Original paths are already gone even if purging quarantine later fails.
        let unlinked = self
            .files
            .iter()
            .map(|(original, _, _)| original.clone())
            .collect();
        let mut deleted = Vec::new();
        let mut bytes = 0u64;
        for (original, staged, size) in &self.files {
            match fs::remove_file(staged) {
                Ok(()) => {
                    deleted.push(original.clone());
                    bytes = bytes.saturating_add(*size);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    deleted.push(original.clone());
                }
                Err(e) => warnings.push(format!(
                    "model was unlinked but staged file remains at {}: {e}",
                    staged.display()
                )),
            }
        }
        for dir in self.dirs.drain(..) {
            // Never recursively clean a failed purge: retain recoverable staged data.
            if fs::read_dir(dir.path()).is_ok_and(|mut entries| entries.next().is_none()) {
                if let Err(e) = dir.close() {
                    warnings.push(format!(
                        "cannot remove empty deletion staging directory: {e}"
                    ));
                }
            } else {
                let path = dir.keep();
                warnings.push(format!(
                    "retained deletion staging directory {}",
                    path.display()
                ));
            }
        }
        (deleted, bytes, unlinked)
    }
}
impl Drop for Staged {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let mut failed = false;
        for (original, staged, _) in self.files.iter().rev() {
            let absent = matches!(fs::symlink_metadata(original), Err(e) if e.kind() == std::io::ErrorKind::NotFound);
            if !absent || fs::rename(staged, original).is_err() {
                failed = true;
                eprintln!(
                    "WARNING: deletion rollback could not restore {}; recover staged file {}",
                    original.display(),
                    staged.display()
                );
            }
        }
        if failed {
            for dir in self.dirs.drain(..) {
                let _ = dir.keep();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    struct Fixture {
        _temp: tempfile::TempDir,
        base: PathBuf,
        local: PathBuf,
        studio: PathBuf,
        store: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let temp = tempfile::tempdir().unwrap();
            let base = fs::canonicalize(temp.path()).unwrap();
            let local = base.join("managed");
            let studio = base.join("studio");
            let store = base.join("store");
            for p in [
                &local,
                &studio,
                &store.join("manifests"),
                &store.join("blobs"),
            ] {
                fs::create_dir_all(p).unwrap();
            }
            Self {
                _temp: temp,
                base,
                local,
                studio,
                store,
            }
        }
        fn options(&self) -> DiscoveryOptions {
            DiscoveryOptions {
                local_dirs: vec![self.local.clone(), self.studio.clone()],
                ollama_dir: Some(self.store.clone()),
            }
        }
        fn delete(&self, label: &str, source: Option<&str>) -> Result<Value, String> {
            run_with(label, source, self.options(), vec![self.studio.clone()])
        }
        fn blob(&self, n: u64, bytes: &[u8]) -> String {
            let digest = format!("sha256:{n:064x}");
            fs::write(
                self.store.join("blobs").join(digest.replace(':', "-")),
                bytes,
            )
            .unwrap();
            digest
        }
        fn blob_path(&self, digest: &str) -> PathBuf {
            self.store.join("blobs").join(digest.replace(':', "-"))
        }
        fn manifest(&self, label: &str, v: Value) -> PathBuf {
            let (_, path, _) = models::import_source(label, Some(&self.store)).unwrap();
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
            path
        }
        fn model(&self, label: &str, digest: &str) -> PathBuf {
            let size = fs::metadata(self.blob_path(digest)).unwrap().len();
            self.manifest(label,json!({"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":digest,"size":size}]}))
        }
        fn file(&self, root: &Path, name: &str) -> PathBuf {
            let path = root.join(name);
            fs::write(&path, b"GGUFfixture").unwrap();
            path
        }
        fn receipt(&self, path: &Path, label: &str, source: &Value) -> PathBuf {
            let fingerprint = format!("{:x}", Sha256::digest(serde_json::to_vec(source).unwrap()));
            let receipt = path
                .parent()
                .unwrap()
                .join(format!(".lma-import-v1-{fingerprint}.json"));
            fs::write(&receipt,serde_json::to_vec(&json!({"version":1,"source_root":self.store,"model":label,"fingerprint":fingerprint,"output":path,"output_bytes":fs::metadata(path).unwrap().len()})).unwrap()).unwrap();
            receipt
        }
    }
    #[test]
    fn actual_ollama_delete_preserves_shared_blobs_and_unrelated_models() {
        let f = Fixture::new();
        let shared = f.blob(1, b"GGUFshared");
        let unique = f.blob(2, b"GGUFunique");
        let unrelated = f.blob(3, b"GGUFother");
        let first=f.manifest("choice",json!({"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":unique,"size":10}],"config":{"digest":shared}}));
        let second=f.manifest("other",json!({"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":unrelated,"size":9}],"config":{"digest":shared}}));
        let out = f.delete("choice", Some("ollama")).unwrap();
        assert_eq!(out["blobGcComplete"], true);
        assert!(!first.exists());
        assert!(!f.blob_path(&unique).exists());
        assert!(f.blob_path(&shared).exists());
        assert!(f.blob_path(&unrelated).exists());
        assert!(second.exists());
        assert!(f.store.join("manifests").is_dir());
    }
    #[test]
    fn nested_mlx_strings_and_nested_manifest_blobs_keep_references_alive() {
        for nested_blob in [false, true] {
            let f = Fixture::new();
            let shared = f.blob(10, b"GGUFshared");
            let primary = f.model("choice", &shared);
            if nested_blob {
                let data = serde_json::to_vec(
                    &json!({"schemaVersion":2,"layers":[],"weights":{"tensor":shared}}),
                )
                .unwrap();
                let provenance = f.blob(11, &data);
                f.manifest("remaining",json!({"schemaVersion":2,"layers":[{"mediaType":"application/vnd.docker.distribution.manifest.v2+json","digest":provenance,"size":data.len()}]}));
            } else {
                f.manifest("remaining",json!({"schemaVersion":2,"format":"safetensors","layers":[],"assets":{"nested":[{"tensor_digest":shared}]}}));
            }
            let out = f.delete("choice", Some("ollama")).unwrap();
            assert_eq!(out["blobGcComplete"], true);
            assert!(!primary.exists());
            assert!(f.blob_path(&shared).exists());
        }
    }
    #[test]
    fn malformed_duplicate_and_overflowing_reference_scans_retain_blobs() {
        for bad in [
            b"not JSON".as_slice(),
            br#"{"schemaVersion":2,"layers":[],"same":"sha256:bad","same":null}"#.as_slice(),
            br#"{"schemaVersion":2,"layers":[],"config":{"digest":"sha256:../../unsafe"}}"#
                .as_slice(),
        ] {
            let f = Fixture::new();
            let d = f.blob(20, b"GGUFfixture");
            let chosen = f.model("choice", &d);
            let other = f.manifest("corrupt", json!({"schemaVersion":2,"layers":[]}));
            fs::write(other, bad).unwrap();
            let out = f.delete("choice", Some("ollama")).unwrap();
            assert_eq!(out["blobGcComplete"], false);
            assert!(!chosen.exists());
            assert!(f.blob_path(&d).exists());
        }
        let f = Fixture::new();
        let d = f.blob(21, b"GGUFfixture");
        let chosen = f.model("choice", &d);
        let mut budget = Budget::new();
        budget.entries = MAX_ENTRIES;
        assert!(
            scan_manifests(
                &f.store.join("manifests"),
                &chosen,
                &f.store,
                0,
                &mut budget,
                &mut BTreeSet::new()
            )
            .is_err()
        );
        assert!(chosen.exists());
        assert!(f.blob_path(&d).exists());
        let mut budget = Budget::new();
        budget.bytes = SCAN_BYTES;
        assert!(manifest(&chosen, &f.store, &mut budget).is_err());
        let mut deep = f.store.join("manifests");
        for n in 0..=MAX_DEPTH {
            deep = deep.join(format!("nested-{n}"));
        }
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("tag"), br#"{"schemaVersion":2,"layers":[]}"#).unwrap();
        let out = f.delete("choice", Some("ollama")).unwrap();
        assert_eq!(out["blobGcComplete"], false);
        assert!(!chosen.exists());
        assert!(f.blob_path(&d).exists());
    }
    #[test]
    fn source_guard_and_imported_artifact_deletion_are_independent() {
        let f = Fixture::new();
        let source_blob = f.blob(30, b"source safetensors fixture");
        let source = json!({"schemaVersion":2,"runner":"mlx","format":"safetensors","layers":[{"name":"weight","mediaType":"application/vnd.ollama.image.tensor","digest":source_blob,"size":25}]});
        let manifest = f.manifest("choice", source.clone());
        let converted = f.file(&f.local, "converted.gguf");
        let receipt = f.receipt(&converted, "choice:latest", &source);
        assert!(f.delete("choice", None).unwrap_err().contains("ambiguous"));
        assert!(manifest.exists());
        assert!(converted.exists());
        assert!(receipt.exists());
        let out = f.delete("choice", Some("ollama")).unwrap();
        assert_eq!(out["source"], "ollama");
        assert!(!manifest.exists());
        assert!(!f.blob_path(&source_blob).exists());
        assert!(converted.exists());
        assert!(receipt.exists());
        let out = f.delete("choice", Some("lma")).unwrap();
        assert_eq!(out["source"], "lma");
        assert!(!converted.exists());
        assert!(!receipt.exists());
    }
    #[test]
    fn registry_aliases_share_managed_weights_and_never_delete_projectors() {
        let f = Fixture::new();
        let weights = f.file(&f.local, "weights.gguf");
        let projector = f.file(&f.local, "mmproj.gguf");
        let entry = registry::Entry {
            path: weights.clone(),
            projector: Some(projector.clone()),
        };
        registry::put(&f.local, "first", entry.clone()).unwrap();
        registry::put(&f.local, "second", entry).unwrap();
        let out = f.delete("first", Some("lma")).unwrap();
        assert_eq!(out["aliasDeleted"], true);
        assert!(weights.exists());
        assert!(projector.exists());
        assert!(
            !registry::read(&f.local)
                .unwrap()
                .contains_key("first:latest")
        );
        f.delete("second", Some("lma")).unwrap();
        assert!(!weights.exists());
        assert!(projector.exists());
        assert!(registry::read(&f.local).unwrap().is_empty());
    }
    #[test]
    fn external_registry_alias_removal_is_distinct_from_lm_studio_file_deletion() {
        let f = Fixture::new();
        let weights = f.file(&f.studio, "choice.gguf");
        let projector = f.file(&f.studio, "mmproj.gguf");
        let config = f.studio.join("config.json");
        fs::write(&config, b"{}").unwrap();
        registry::put(
            &f.local,
            "choice",
            registry::Entry {
                path: weights.clone(),
                projector: Some(projector.clone()),
            },
        )
        .unwrap();
        assert!(f.delete("choice", None).unwrap_err().contains("ambiguous"));
        f.delete("choice", Some("lma")).unwrap();
        assert!(weights.exists());
        f.delete("choice", Some("lm-studio")).unwrap();
        assert!(!weights.exists());
        assert!(projector.exists());
        assert!(config.exists());
        assert!(f.studio.is_dir());
    }
    #[test]
    fn exact_external_alias_id_never_selects_lm_studio_weights() {
        let f = Fixture::new();
        let weights = f.file(&f.studio, "choice.gguf");
        registry::put(
            &f.local,
            "choice",
            registry::Entry {
                path: weights.clone(),
                projector: None,
            },
        )
        .unwrap();
        assert!(f.delete("lma:choice:latest", Some("lm-studio")).is_err());
        assert!(weights.exists());
        assert!(
            registry::read(&f.local)
                .unwrap()
                .contains_key("choice:latest")
        );
        let out = f.delete("lma:choice:latest", None).unwrap();
        assert_eq!(out["aliasDeleted"], true);
        assert_eq!(out["source"], "lma");
        assert_eq!(out["deletedBytes"], 0);
        assert!(weights.exists());
        assert!(registry::read(&f.local).unwrap().is_empty());
    }

    #[test]
    fn sibling_external_aliases_have_only_one_physical_deletion_candidate() {
        let f = Fixture::new();
        let weights = f.file(&f.studio, "choice.gguf");
        for name in ["first", "second"] {
            registry::put(
                &f.local,
                name,
                registry::Entry {
                    path: weights.clone(),
                    projector: None,
                },
            )
            .unwrap();
        }
        let id = format!("lma:{}", weights.display());
        let out = f.delete(&id, Some("lm-studio")).unwrap();
        assert_eq!(out["aliasDeleted"], false);
        assert!(!weights.exists());
        assert!(registry::read(&f.local).unwrap().is_empty());
        assert_eq!(out["registryCleanupComplete"], true);
        assert_eq!(out["deletedAliases"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn externally_deleted_gguf_alias_is_removed_without_obsolete_missing_warning() {
        let f = Fixture::new();
        let weights = f.file(&f.local, "gone.gguf");
        for name in ["dead", "unrelated-stale"] {
            registry::put(
                &f.local,
                name,
                registry::Entry {
                    path: weights.clone(),
                    projector: None,
                },
            )
            .unwrap();
        }
        fs::remove_file(&weights).unwrap();
        let before = models::discover(f.options());
        assert!(
            before
                .warnings
                .iter()
                .any(|w| w.contains("registered model 'dead:latest' is missing"))
        );
        // Discovery is read-only, not a global orphan sweep.
        assert_eq!(registry::read(&f.local).unwrap().len(), 2);
        let out = f.delete("lma:dead:latest", Some("lma")).unwrap();
        assert_eq!(out["aliasDeleted"], true);
        assert_eq!(out["registryCleanupComplete"], true);
        assert_eq!(out["deletedAliases"].as_array().unwrap().len(), 1);
        assert!(!out["warnings"].as_array().unwrap().iter().any(|w| {
            w.as_str()
                .unwrap()
                .contains("registered model 'dead:latest'")
        }));
        let after = models::discover(f.options());
        assert!(
            !after
                .warnings
                .iter()
                .any(|w| w.contains("registered model 'dead:latest'"))
        );
        assert!(
            after
                .warnings
                .iter()
                .any(|w| w.contains("unrelated-stale:latest"))
        );
        assert!(
            registry::read(&f.local)
                .unwrap()
                .contains_key("unrelated-stale:latest")
        );
        f.delete("lma:unrelated-stale:latest", Some("lma")).unwrap();
        assert!(models::discover(f.options()).warnings.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn physical_split_deletion_cleans_cross_root_and_symlink_primary_refs_only() {
        let f = Fixture::new();
        let first = f.studio.join("choice-00001-of-00002.gguf");
        let second = f.studio.join("choice-00002-of-00002.gguf");
        split(&first, 0, 2);
        split(&second, 1, 2);
        let other = f.file(&f.local, "other.gguf");
        let stale = f.file(&f.base, "stale.gguf");
        let definition = super::super::modelfile::parse("FROM other\nSYSTEM keep me").unwrap();
        registry::put_definition(
            &f.local,
            "unrelated",
            registry::Entry {
                path: other.clone(),
                projector: Some(first.clone()),
            },
            definition.clone(),
        )
        .unwrap();
        registry::put(
            &f.local,
            "stale",
            registry::Entry {
                path: stale.clone(),
                projector: None,
            },
        )
        .unwrap();
        fs::remove_file(stale).unwrap();
        registry::put(
            &f.studio,
            "second-shard",
            registry::Entry {
                path: second.clone(),
                projector: None,
            },
        )
        .unwrap();
        let link = f.base.join("weight-link.gguf");
        std::os::unix::fs::symlink(&first, &link).unwrap();
        // A legacy document can contain symlink paths (put canonicalizes them).
        let mut entries = registry::read(&f.local).unwrap();
        entries.insert(
            "linked:latest".into(),
            registry::Entry {
                path: link.clone(),
                projector: None,
            },
        );
        let mut document: Value =
            serde_json::from_slice(&fs::read(f.local.join(".registry-v1.json")).unwrap()).unwrap();
        document["linked:latest"] = serde_json::to_value(&entries["linked:latest"]).unwrap();
        fs::write(
            f.local.join(".registry-v1.json"),
            serde_json::to_vec(&document).unwrap(),
        )
        .unwrap();
        let out = f
            .delete(&format!("lma:{}", first.display()), Some("lm-studio"))
            .unwrap();
        assert_eq!(out["aliasDeleted"], false);
        assert_eq!(out["registryCleanupComplete"], true);
        assert_eq!(out["deletedAliases"].as_array().unwrap().len(), 2);
        assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
        let remaining = registry::read(&f.local).unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.contains_key("stale:latest"));
        assert!(remaining.contains_key("unrelated:latest"));
        assert_eq!(
            registry::definition(&f.local, "unrelated").unwrap(),
            Some(definition)
        );
        assert!(registry::read(&f.studio).unwrap().is_empty());
        assert!(other.exists());
        assert!(!first.exists());
        assert!(!second.exists());
    }

    #[test]
    fn ollama_cleanup_preserves_shared_blob_and_imported_output_aliases() {
        let f = Fixture::new();
        let unique = f.blob(70, b"GGUFunique");
        let shared = f.blob(71, b"GGUFshared");
        let source = json!({"schemaVersion":2,"layers":[
            {"mediaType":"application/vnd.ollama.image.model","digest":unique,"size":10},
            {"mediaType":"application/vnd.ollama.image.model","digest":shared,"size":10}
        ]});
        f.manifest("choice", source.clone());
        f.model("remaining", &shared);
        let imported = f.file(&f.local, "imported.gguf");
        let receipt = f.receipt(&imported, "choice:latest", &source);
        for (name, path) in [
            ("unique-a", f.blob_path(&unique)),
            ("unique-b", f.blob_path(&unique)),
            ("shared", f.blob_path(&shared)),
            ("imported", imported.clone()),
        ] {
            registry::put(
                &f.local,
                name,
                registry::Entry {
                    path,
                    projector: None,
                },
            )
            .unwrap();
        }
        let out = f.delete("choice", Some("ollama")).unwrap();
        assert_eq!(out["registryCleanupComplete"], true);
        assert_eq!(out["aliasDeleted"], false);
        assert_eq!(out["deletedAliases"].as_array().unwrap().len(), 2);
        let remaining = registry::read(&f.local).unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.contains_key("shared:latest"));
        assert!(remaining.contains_key("imported:latest"));
        assert!(imported.exists());
        assert!(receipt.exists());
        assert!(f.blob_path(&shared).exists());
        // Alias-only deletion never removes another alias to retained external weights.
        registry::put(
            &f.local,
            "shared-sibling",
            remaining["shared:latest"].clone(),
        )
        .unwrap();
        let out = f.delete("lma:shared:latest", Some("lma")).unwrap();
        assert_eq!(out["deletedAliases"].as_array().unwrap().len(), 1);
        assert!(
            registry::read(&f.local)
                .unwrap()
                .contains_key("shared-sibling:latest")
        );
        assert!(f.blob_path(&shared).exists());
    }

    #[test]
    fn physical_deletion_reports_registry_lock_failure_after_destruction() {
        let f = Fixture::new();
        let weights = f.file(&f.studio, "choice.gguf");
        registry::put(
            &f.local,
            "alias",
            registry::Entry {
                path: weights.clone(),
                projector: None,
            },
        )
        .unwrap();
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(f.local.join(".registry-v1.lock"))
            .unwrap();
        lock.try_lock().unwrap();
        let out = f
            .delete(&format!("lma:{}", weights.display()), Some("lm-studio"))
            .unwrap();
        assert!(!weights.exists());
        assert_eq!(out["registryCleanupComplete"], false);
        assert!(out["deletedAliases"].as_array().unwrap().is_empty());
        assert!(out["warnings"].as_array().unwrap().iter().any(|w| {
            w.as_str()
                .unwrap()
                .contains("registry cleanup is incomplete")
        }));
        assert!(
            registry::read(&f.local)
                .unwrap()
                .contains_key("alias:latest")
        );
    }

    #[test]
    fn cleanup_revalidates_retargeted_aliases_and_recreated_assets() {
        for recreate in [false, true] {
            let f = Fixture::new();
            let weights = f.file(&f.studio, "choice.gguf");
            let other = f.file(&f.base, "other.gguf");
            registry::put(
                &f.local,
                "alias",
                registry::Entry {
                    path: weights.clone(),
                    projector: None,
                },
            )
            .unwrap();
            let context = Context {
                roots: vec![f.local.clone()],
                lm_roots: vec![f.studio.clone()],
                registry_roots: vec![f.local.clone()],
                ollama_root: Some(f.store.clone()),
            };
            let references = registry_references(&context).unwrap();
            fs::remove_file(&weights).unwrap();
            if recreate {
                fs::write(&weights, b"GGUFrecreated").unwrap();
            } else {
                registry::delete(&f.local, "alias").unwrap();
                registry::put(
                    &f.local,
                    "alias",
                    registry::Entry {
                        path: other,
                        projector: None,
                    },
                )
                .unwrap();
            }
            let mut warnings = Vec::new();
            let cleanup =
                cleanup_registry(&references, &[weights.clone()], &[weights], &mut warnings);
            assert!(cleanup.complete);
            assert!(cleanup.deleted.is_empty());
            assert!(warnings.is_empty());
            assert!(
                registry::read(&f.local)
                    .unwrap()
                    .contains_key("alias:latest")
            );
        }
    }

    #[test]
    fn different_files_with_the_same_name_require_a_source() {
        let f = Fixture::new();
        let first = f.file(&f.local, "choice.gguf");
        let second = f.file(&f.studio, "choice.gguf");
        assert!(f.delete("choice", None).unwrap_err().contains("ambiguous"));
        assert!(first.exists());
        assert!(second.exists());
        f.delete("choice", Some("lm-studio")).unwrap();
        assert!(first.exists());
        assert!(!second.exists());
        assert!(f.delete("choice", Some("unknown")).is_err());
        assert!(f.delete("../choice", None).is_err());
        assert!(first.exists());
    }
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    }
    fn split(path: &Path, index: u64, total: u64) {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(1u64.to_le_bytes());
        out.extend(3u64.to_le_bytes());
        for (key, value) in [
            ("split.no", index),
            ("split.count", total),
            ("split.tensors.count", total),
        ] {
            string(&mut out, key);
            out.extend(10u32.to_le_bytes());
            out.extend(value.to_le_bytes());
        }
        string(&mut out, &format!("weight.{index}"));
        out.extend(1u32.to_le_bytes());
        out.extend(1u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.resize(out.len().div_ceil(32) * 32 + 4, 0);
        fs::write(path, out).unwrap();
    }
    #[test]
    fn lm_studio_complete_split_set_is_deleted_but_partial_sets_are_untouched() {
        let f = Fixture::new();
        let first = f.studio.join("choice-00001-of-00002.gguf");
        let second = f.studio.join("choice-00002-of-00002.gguf");
        split(&first, 0, 2);
        split(&second, 1, 2);
        let out = f.delete("choice", Some("lm-studio")).unwrap();
        assert_eq!(out["deletedFiles"].as_array().unwrap().len(), 2);
        assert!(!first.exists());
        assert!(!second.exists());
        split(&first, 0, 2);
        assert!(f.delete("choice", Some("lm-studio")).is_err());
        assert!(first.exists());
        split(&second, 0, 2);
        assert!(f.delete("choice", Some("lm-studio")).is_err());
        assert!(first.exists());
        assert!(second.exists());
    }
    #[cfg(unix)]
    #[test]
    fn symlinks_never_delete_external_targets_or_permit_partial_shard_deletion() {
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        let external = f.file(&f.base, "outside.gguf");
        let link = f.studio.join("linked.gguf");
        symlink(&external, &link).unwrap();
        assert!(f.delete("linked", None).is_err());
        assert!(external.exists());
        assert!(fs::symlink_metadata(&link).is_ok());
        let first = f.studio.join("choice-00001-of-00002.gguf");
        let second = f.studio.join("choice-00002-of-00002.gguf");
        let outside = f.base.join("outside-part.gguf");
        split(&first, 0, 2);
        split(&outside, 1, 2);
        symlink(&outside, &second).unwrap();
        assert!(f.delete("choice", Some("lm-studio")).is_err());
        assert!(first.exists());
        assert!(outside.exists());
        assert!(fs::symlink_metadata(&second).is_ok());
        let d = f.blob(40, b"GGUFfixture");
        let selected = f.model("selected", &d);
        let bad = f.store.join("manifests").join("linked-manifest");
        symlink(&f.base, &bad).unwrap();
        let out = f.delete("selected", Some("ollama")).unwrap();
        assert_eq!(out["blobGcComplete"], false);
        assert!(!selected.exists());
        assert!(f.blob_path(&d).exists());
        let selected = f.model("redirected", &d);
        let external_manifest = f.base.join("outside-manifest.json");
        fs::rename(&selected, &external_manifest).unwrap();
        symlink(&external_manifest, &selected).unwrap();
        assert!(f.delete("redirected", Some("ollama")).is_err());
        assert!(external_manifest.exists());
        assert!(fs::symlink_metadata(&selected).is_ok());
        assert!(f.blob_path(&d).exists());
    }
    #[test]
    fn failed_quarantine_purge_still_cleans_aliases_to_unlinked_primary_weights() {
        let f = Fixture::new();
        let model = f.file(&f.studio, "choice.gguf");
        let entry = registry::Entry {
            path: model.clone(),
            projector: None,
        };
        registry::put(&f.local, "choice", entry.clone()).unwrap();
        let references = BTreeMap::from([(
            f.local.clone(),
            BTreeMap::from([("choice:latest".into(), (entry, model.clone()))]),
        )]);
        let staged = Staged::new(vec![model.clone()]).unwrap();
        // Simulate an unpurgeable quarantine entry, without touching real stores.
        fs::remove_file(&staged.files[0].1).unwrap();
        fs::create_dir(&staged.files[0].1).unwrap();
        let mut warnings = Vec::new();
        let (deleted, bytes, unlinked) = staged.commit(&mut warnings);
        assert!(deleted.is_empty());
        assert_eq!(bytes, 0);
        assert!(!model.exists());
        let cleanup = cleanup_registry(&references, &[model], &unlinked, &mut warnings);
        assert!(cleanup.complete);
        assert_eq!(cleanup.deleted.len(), 1);
        assert!(registry::read(&f.local).unwrap().is_empty());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("staged file remains"))
        );
    }
    #[test]
    fn registry_publication_failure_rolls_staged_weights_back() {
        let f = Fixture::new();
        let model = f.file(&f.local, "choice.gguf");
        let expected = registry::Entry {
            path: model.clone(),
            projector: None,
        };
        let mut entries = BTreeMap::from([("choice:latest".to_owned(), expected.clone())]);
        // A legal compact registry can exceed the publication limit when pretty
        // printed. Trigger that failure after file staging without OS permissions.
        let filler = registry::Entry {
            path: f.base.join("x".repeat(230)),
            projector: None,
        };
        let unit = serde_json::to_vec(&BTreeMap::from([("fixture-0000:latest", filler.clone())]))
            .unwrap()
            .len()
            - 1;
        let count = (1024 * 1024 - 8192 - serde_json::to_vec(&entries).unwrap().len()) / unit;
        assert!(count < 4096);
        for n in 0..count {
            entries.insert(format!("fixture-{n:04}:latest"), filler.clone());
        }
        let compact = serde_json::to_vec(&entries).unwrap();
        assert!(compact.len() < 1024 * 1024);
        assert!(serde_json::to_vec_pretty(&entries).unwrap().len() > 1024 * 1024);
        fs::write(f.local.join(".registry-v1.json"), &compact).unwrap();
        let out = registry::remove_with(&f.local, "choice", &expected, |_| {
            Staged::new(vec![model.clone()])
        });
        assert!(out.is_err());
        assert_eq!(fs::read(&model).unwrap(), b"GGUFfixture");
        assert_eq!(
            fs::read(f.local.join(".registry-v1.json")).unwrap(),
            compact
        );
    }
    #[test]
    fn invalid_digest_and_selected_manifest_do_not_touch_unrelated_paths() {
        let f = Fixture::new();
        let outside = f.base.join("must-stay");
        fs::write(&outside, b"untouched").unwrap();
        let selected = f.manifest(
            "choice",
            json!({"schemaVersion":2,"layers":[],"config":{"digest":"sha256:../../must-stay"}}),
        );
        let out = f.delete("choice", Some("ollama")).unwrap();
        assert_eq!(out["blobGcComplete"], false);
        assert!(!selected.exists());
        assert_eq!(fs::read(outside).unwrap(), b"untouched");
        let selected = f.manifest("choice", json!({"schemaVersion":2,"layers":[]}));
        fs::write(&selected, b"malformed").unwrap();
        assert!(f.delete("choice", Some("ollama")).is_err());
        assert!(selected.exists());
    }
}
