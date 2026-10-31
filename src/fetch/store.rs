//! Typed, in-process desktop store adapter. Call only for explicit user actions,
//! on a blocking worker. The host serializes downloads and supplies its read-only
//! installed catalog; callers cannot choose a download destination.
use super::{
    catalog::{self, Candidate, Source},
    download,
};
use crate::models::{DiscoveredModel, discovery::default_models_dir, registry};
use serde::Serialize;
use std::{
    collections::HashSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreFile {
    pub name: String,
    /// Unknown upstream sizes are null, never guessed as zero.
    pub bytes: Option<u64>,
    pub installed: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreModel {
    pub id: String,
    pub source: String,
    pub name: String,
    pub exact: bool,
    pub files: Vec<StoreFile>,
    pub installed: bool,
    /// A single downloadable choice, not the sum of alternative quantizations.
    pub total_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub models: Vec<StoreModel>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DownloadResult {
    /// Existing absolute weights path; Ollama references are returned as model.
    pub path: String,
    pub model: String,
}

fn query(value: &str) -> Result<&str, String> {
    let value = value.trim();
    if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
        return Err(
            "Use a nonempty model query of at most 512 bytes without control characters.".into(),
        );
    }
    Ok(value)
}

fn sources(source: &str, search: bool) -> Result<(bool, bool), String> {
    match source {
        "ollama" => Ok((true, false)),
        "lm-studio" | "huggingface" => Ok((false, true)),
        "all" if search => Ok((true, true)),
        _ => Err("Choose ollama or lm-studio (public Hugging Face GGUF); all is supported only for search.".into()),
    }
}

fn source_name(source: Source) -> &'static str {
    match source {
        Source::Ollama => "ollama",
        Source::LmStudio => "lm-studio",
    }
}

fn files(candidate: &Candidate) -> Vec<StoreFile> {
    let Some(metadata) = &candidate.metadata else {
        return Vec::new();
    };
    match candidate.source {
        Source::Ollama => catalog::select(candidate, None)
            .map(|selection| {
                selection
                    .artifacts
                    .into_iter()
                    .map(|file| StoreFile {
                        name: file.filename,
                        bytes: Some(file.size),
                        installed: false,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        Source::LmStudio => {
            let mut seen = HashSet::new();
            let mut files: Vec<_> = metadata
                .get("siblings")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|entry| {
                    let name = entry.get("rfilename")?.as_str()?;
                    if !name.to_ascii_lowercase().ends_with(".gguf")
                        || download::safe_relative(name).is_err()
                        || !seen.insert(name)
                    {
                        return None;
                    }
                    Some(StoreFile {
                        name: name.into(),
                        installed: false,
                        bytes: entry
                            .get("lfs")
                            .and_then(|l| l.get("size"))
                            .or_else(|| entry.get("size"))
                            .and_then(|s| s.as_u64()),
                    })
                })
                .collect();
            files.sort_by(|a, b| a.name.cmp(&b.name));
            files
        }
    }
}

pub fn search(
    query_text: &str,
    source: &str,
    installed: &[DiscoveredModel],
) -> Result<SearchResult, String> {
    let query = query(query_text)?;
    let (ollama, hf) = sources(source, true)?;
    let client = download::client()?;
    let catalog = catalog::discover(&client, query, ollama, hf);
    let mut warnings = catalog.warnings;
    let root = default_models_dir().ok();
    let models = catalog
        .candidates
        .iter()
        .map(|candidate| describe(candidate, root.as_deref(), installed, &mut warnings))
        .collect();
    warnings.truncate(64);
    Ok(SearchResult { models, warnings })
}

fn describe(
    candidate: &Candidate,
    root: Option<&Path>,
    installed: &[DiscoveredModel],
    warnings: &mut Vec<String>,
) -> StoreModel {
    let mut files = files(candidate);
    for file in &mut files {
        let selected_file = (candidate.source == Source::LmStudio).then_some(file.name.as_str());
        file.installed = match is_installed(
            root,
            candidate.source,
            &candidate.id,
            selected_file,
            installed,
        ) {
            Ok(found) => found,
            Err(error) => {
                if warnings.len() < 64 {
                    warnings.push(error);
                }
                false
            }
        };
    }
    if candidate.exact && files.is_empty() {
        warnings.push(format!(
            "{}: no supported downloadable GGUF metadata; select a public standalone GGUF model.",
            candidate.id
        ));
    }
    let total_bytes = match candidate.source {
        Source::Ollama if !files.is_empty() => files
            .iter()
            .try_fold(0u64, |sum, file| sum.checked_add(file.bytes?)),
        Source::LmStudio if files.len() == 1 => files[0].bytes,
        _ => None,
    };
    StoreModel {
        id: candidate.id.clone(),
        source: source_name(candidate.source).into(),
        name: candidate.id.clone(),
        exact: candidate.exact,
        installed: files.iter().any(|file| file.installed),
        files,
        total_bytes,
    }
}

fn exact_identity(id: &str, source: Source) -> Result<String, String> {
    match source {
        Source::Ollama => {
            let id = id.strip_prefix("registry.ollama.ai/").unwrap_or(id);
            let (name, tag) = id.rsplit_once(':').unwrap_or((id, "latest"));
            let parts: Vec<_> = name.split('/').collect();
            if parts.is_empty()
                || parts.len() > 2
                || !parts.iter().all(|p| download::safe_component(p))
                || !download::safe_component(tag)
            {
                return Err("Choose an exact Ollama [namespace/]name[:tag].".into());
            }
            Ok(format!(
                "{}{name}:{tag}",
                if parts.len() == 1 { "library/" } else { "" }
            ))
        }
        Source::LmStudio => {
            if id.split('/').count() != 2 || !id.split('/').all(download::safe_component) {
                return Err(
                    "Choose an exact Hugging Face owner/repository before downloading.".into(),
                );
            }
            Ok(id.into())
        }
    }
}

/// Labels and path suffixes are not provenance. Only resolve real manifest
/// selectors; in particular an hf.co quantization tag does not identify an HF file.
fn installed_manifest(
    source: Source,
    identity: &str,
    installed: &[DiscoveredModel],
) -> Result<Option<DownloadResult>, String> {
    if source != Source::Ollama {
        return Ok(None);
    }
    for model in installed {
        if model.ollama
            && model.ollama_dir.is_some()
            && exact_identity(&model.model, source).as_deref() == Ok(identity)
        {
            return existing_result(model).map(Some);
        }
    }
    Ok(None)
}

struct LocalInstall {
    directory: PathBuf,
    selection: catalog::Selection,
}

fn managed_directory(root: &Path, source: Source, identity: &str) -> PathBuf {
    match source {
        Source::Ollama => {
            let (name, tag) = identity.rsplit_once(':').unwrap();
            root.join("ollama").join(name).join(tag)
        }
        Source::LmStudio => root.join("lm-studio").join(identity),
    }
}

/// The shared downloader already writes content-addressed origin snapshots before
/// the artifacts. They pin both identity and checksums, but are NOT completion
/// markers: every required artifact must still be present and valid for reuse.
fn local_installs(
    root: Option<&Path>,
    source: Source,
    identity: &str,
    file: Option<&str>,
    installed: &[DiscoveredModel],
) -> Result<Vec<LocalInstall>, String> {
    let filename = match source {
        Source::Ollama => "model.gguf",
        Source::LmStudio => file.ok_or(
            "Select an exact Hugging Face GGUF filename, including for single-file repositories.",
        )?,
    };
    let suffix = download::safe_relative(filename)?;
    let mut directories = Vec::new();
    if let Some(root) = root {
        directories.push(managed_directory(root, source, identity));
    }
    for model in installed.iter().filter(|model| !model.ollama) {
        let path = Path::new(&model.model);
        if path.is_absolute()
            && path.ends_with(&suffix)
            && let Some(directory) = path.ancestors().nth(suffix.components().count())
        {
            directories.push(directory.to_path_buf());
        }
    }
    let mut seen = HashSet::new();
    let mut selections = HashSet::new();
    let mut result = Vec::new();
    for directory in directories {
        let directory = fs::canonicalize(&directory).unwrap_or(directory);
        if !seen.insert(directory.clone()) {
            continue;
        }
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(format!("Cannot inspect installed provenance: {error}")),
        };
        let prefix = match source {
            Source::Ollama => "ollama-origin-",
            Source::LmStudio => "hf-origin-",
        };
        let mut receipts = 0;
        for (index, entry) in entries.enumerate() {
            if index >= 10_000 {
                return Err("Installed provenance scan limit reached; inspect the model directory before retrying.".into());
            }
            let entry = entry.map_err(|error| error.to_string())?;
            let name = entry.file_name();
            let Some(checksum) = name
                .to_str()
                .and_then(|name| name.strip_prefix(prefix))
                .and_then(|name| name.strip_suffix(".json"))
            else {
                continue;
            };
            if download::sha256(checksum).is_err() {
                continue;
            }
            receipts += 1;
            if receipts > 64 {
                return Err("Too many installed provenance snapshots; inspect the model directory before retrying.".into());
            }
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
            if !metadata.file_type().is_file() || metadata.len() > download::JSON_LIMIT {
                return Err(
                    "Installed provenance must be a regular, non-symlink file of at most 8 MiB."
                        .into(),
                );
            }
            let mut bytes = Vec::new();
            crate::local_file::open_regular(&path)?
                .take(download::JSON_LIMIT + 1)
                .read_to_end(&mut bytes)
                .map_err(|error| error.to_string())?;
            if bytes.len() as u64 > download::JSON_LIMIT || digest(&bytes) != checksum {
                return Err(
                    "Installed provenance checksum mismatch; nothing was downloaded.".into(),
                );
            }
            let snapshot: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|error| format!("Invalid installed provenance: {error}"))?;
            let candidate = Candidate {
                source,
                id: identity.into(),
                exact: true,
                metadata: snapshot.get("metadata").cloned(),
            };
            if source == Source::LmStudio
                && candidate
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("id"))
                    .and_then(|id| id.as_str())
                    != Some(identity)
            {
                continue;
            }
            let Ok(selection) = selected(&candidate, file) else {
                continue;
            };
            let expected: serde_json::Value =
                serde_json::from_slice(&selection.metadata).map_err(|error| error.to_string())?;
            if snapshot != expected {
                continue;
            }
            let key: Vec<_> = selection
                .artifacts
                .iter()
                .map(|artifact| (&artifact.filename, &artifact.sha256, artifact.size))
                .collect();
            let key = serde_json::to_vec(&key).map_err(|error| error.to_string())?;
            if selections.insert((directory.clone(), key)) {
                result.push(LocalInstall {
                    directory: directory.clone(),
                    selection,
                });
            }
        }
    }
    Ok(result)
}

fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn complete(installation: &LocalInstall, hashes: bool) -> Result<bool, String> {
    use sha2::{Digest, Sha256};
    let mut complete = true;
    for artifact in &installation.selection.artifacts {
        let path = installation
            .directory
            .join(download::safe_relative(&artifact.filename)?);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                complete = false;
                continue;
            }
            Err(error) => return Err(error.to_string()),
        };
        if !metadata.file_type().is_file() || metadata.len() != artifact.size {
            return Err(format!(
                "Installed artifact '{}' differs from its provenance; move it aside explicitly before retrying.",
                path.display()
            ));
        }
        if artifact.gguf {
            validate_gguf(&path)?;
        }
        if hashes {
            let mut file =
                crate::local_file::open_regular(&path)?.take(artifact.size.saturating_add(1));
            let mut hash = Sha256::new();
            let mut buffer = [0; 64 * 1024];
            loop {
                let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            if format!("{:x}", hash.finalize()) != artifact.sha256 {
                return Err(format!(
                    "Installed artifact '{}' failed SHA256 verification; nothing was downloaded.",
                    path.display()
                ));
            }
        }
    }
    Ok(complete)
}

fn is_installed(
    root: Option<&Path>,
    source: Source,
    id: &str,
    file: Option<&str>,
    installed: &[DiscoveredModel],
) -> Result<bool, String> {
    let identity = exact_identity(id, source)?;
    if installed_manifest(source, &identity, installed)?.is_some() {
        return Ok(true);
    }
    for installation in local_installs(root, source, &identity, file, installed)? {
        // Search stays read-only and checks layout/header/size; the explicit reuse
        // operation additionally verifies every checksum before returning success.
        if complete(&installation, false)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn existing_result(model: &DiscoveredModel) -> Result<DownloadResult, String> {
    let path = if model.ollama {
        let resolved = crate::models::resolve(
            Path::new(&format!("ollama:{}", model.model)),
            model.ollama_dir.as_deref(),
        )?;
        if let Some(projector) = &resolved.projector {
            validate_gguf(projector)?;
        }
        resolved.model
    } else {
        PathBuf::from(&model.model)
    };
    validate_gguf(&path)?;
    Ok(DownloadResult {
        path: path_text(&path)?,
        model: model.model.clone(),
    })
}

fn validate_gguf(path: &Path) -> Result<(), String> {
    let mut magic = [0; 4];
    use std::io::Read;
    crate::local_file::open_regular(path)?
        .read_exact(&mut magic)
        .map_err(|e| e.to_string())?;
    if &magic != b"GGUF" {
        return Err("Installed model is not a GGUF; refresh the library.".into());
    }
    Ok(())
}

fn path_text(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| "Model path is not UTF-8.".into())
}

fn selected(candidate: &Candidate, file: Option<&str>) -> Result<catalog::Selection, String> {
    if candidate.source == Source::LmStudio && file.is_none() {
        return Err("Select an exact Hugging Face GGUF filename from store_search, including for single-file repositories.".into());
    }
    let selection = catalog::select(candidate, file)?;
    if selection.projector_only {
        return Err("A projector is not a standalone model; choose model weights.".into());
    }
    Ok(selection)
}

fn logical_name(candidate: &Candidate, selection: &catalog::Selection) -> Result<String, String> {
    match candidate.source {
        Source::Ollama => registry::name(&candidate.id),
        Source::LmStudio => {
            let artifact = selection
                .artifacts
                .first()
                .ok_or("No model weights in selection.")?;
            let stem = Path::new(&artifact.filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("gguf");
            let mut tag: String = stem
                .chars()
                .take(160)
                .map(|c| {
                    if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                        c
                    } else {
                        '-'
                    }
                })
                .collect();
            if !tag.starts_with(|c: char| c.is_ascii_alphanumeric()) {
                tag.insert_str(0, "gguf-");
            }
            registry::name(&format!(
                "{}:{tag}-{}",
                candidate.id,
                &artifact.sha256[..12]
            ))
        }
    }
}

pub fn download(
    id: &str,
    source: &str,
    file: Option<&str>,
    installed: &[DiscoveredModel],
) -> Result<DownloadResult, String> {
    let id = query(id)?;
    let (ollama, hf) = sources(source, false)?;
    let source = if ollama {
        Source::Ollama
    } else {
        Source::LmStudio
    };
    let identity = exact_identity(id, source)?;
    if let Some(file) = file {
        download::safe_relative(file)?;
    }
    if ollama && file.is_some() {
        return Err("File selection applies only to Hugging Face repositories.".into());
    }
    // A repository/quantization tag alone never selects a Hugging Face file.
    if hf && file.is_none() {
        return Err(
            "Select an exact Hugging Face GGUF filename, including for single-file repositories."
                .into(),
        );
    }
    if let Some(result) = installed_manifest(source, &identity, installed)? {
        return Ok(result);
    }
    let root = default_models_dir()?;
    if let Some(result) = reuse_local(&root, source, &identity, file, installed)? {
        return Ok(result);
    }
    let client = download::client()?;
    let catalog = catalog::discover(&client, &identity, ollama, hf);
    let candidate = super::exact_candidate(&catalog)
        .map_err(|error| format!("{error}: {}", catalog.warnings.join("; ")))?;
    let selection = selected(candidate, file)?;
    download_selected(
        &root,
        candidate,
        &selection,
        installed,
        |directory, artifact| download::download(&client, directory, artifact).map(|_| ()),
    )
}

fn download_selected(
    root: &Path,
    candidate: &Candidate,
    selection: &catalog::Selection,
    installed: &[DiscoveredModel],
    fetch: impl FnMut(&Path, &download::Artifact) -> Result<(), String>,
) -> Result<DownloadResult, String> {
    // Only an explicit download has fetched/validated this upstream selection.
    // Ordinary external weights need no lma receipt when their bytes prove identity.
    if candidate.source == Source::LmStudio
        && let Some(result) = reuse_selected_file(selection, installed)
    {
        return Ok(result);
    }
    let label = logical_name(candidate, selection)?;
    install_selection(root, &label, selection, fetch)
}

fn reuse_selected_file(
    selection: &catalog::Selection,
    installed: &[DiscoveredModel],
) -> Option<DownloadResult> {
    let [artifact] = selection.artifacts.as_slice() else {
        return None;
    };
    if !artifact.gguf || selection.projector_only {
        return None;
    }
    let filename = Path::new(&artifact.filename).file_name()?;
    let mut seen = HashSet::new();
    for model in installed {
        let path = Path::new(&model.model);
        if model.ollama
            || !path.is_absolute()
            || path.file_name() != Some(filename)
            || model.size_bytes != artifact.size
        {
            continue;
        }
        let Ok(canonical) = fs::canonicalize(path) else {
            continue;
        };
        if !seen.insert(canonical.clone())
            || !matches_artifact(&canonical, artifact).unwrap_or(false)
            || fs::canonicalize(path).ok().as_ref() != Some(&canonical)
        {
            continue;
        }
        let Some(path) = canonical.to_str() else {
            continue;
        };
        return Some(DownloadResult {
            path: path.into(),
            model: path.into(),
        });
    }
    None
}

fn matches_artifact(path: &Path, artifact: &download::Artifact) -> Result<bool, String> {
    use sha2::{Digest, Sha256};
    let mut file = crate::local_file::open_regular(path)?;
    let before = file.metadata().map_err(|error| error.to_string())?;
    if before.len() != artifact.size || artifact.size < 4 {
        return Ok(false);
    }
    let mut magic = [0; 4];
    file.read_exact(&mut magic)
        .map_err(|error| error.to_string())?;
    if &magic != b"GGUF" {
        return Ok(false);
    }
    let mut hash = Sha256::new();
    hash.update(magic);
    let mut count = 4_u64;
    let mut buffer = [0; 64 * 1024];
    let mut reader = file.by_ref().take(artifact.size - 4 + 1);
    loop {
        let length = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if length == 0 {
            break;
        }
        count += length as u64;
        hash.update(&buffer[..length]);
    }
    let after = fs::metadata(path).map_err(|error| error.to_string())?;
    if count != artifact.size
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || format!("{:x}", hash.finalize()) != artifact.sha256
    {
        return Ok(false);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn reuse_local(
    root: &Path,
    source: Source,
    identity: &str,
    file: Option<&str>,
    installed: &[DiscoveredModel],
) -> Result<Option<DownloadResult>, String> {
    let installations = local_installs(Some(root), source, identity, file, installed)?;
    let mut integrity_error = None;
    for installation in &installations {
        match complete(installation, true) {
            Ok(true) => {
                let path = installation
                    .directory
                    .join(&installation.selection.artifacts[0].filename);
                return Ok(Some(DownloadResult {
                    model: path_text(&path)?,
                    path: path_text(&path)?,
                }));
            }
            Ok(false) => {}
            Err(error) => integrity_error = Some(error),
        }
    }
    if let Some(error) = integrity_error {
        return Err(error);
    }
    if installations.len() > 1 {
        return Err("Multiple incomplete pinned installations match; inspect their provenance before retrying. Nothing was downloaded.".into());
    }
    let Some(installation) = installations.into_iter().next() else {
        return Ok(None);
    };
    let expected = root.join(&installation.selection.relative_directory);
    if fs::canonicalize(&expected).ok().as_ref() != Some(&installation.directory) {
        return Err("An external installation is incomplete; repair it in its owning application. Nothing was downloaded.".into());
    }
    let candidate = Candidate {
        source,
        id: identity.into(),
        exact: true,
        metadata: None,
    };
    let label = logical_name(&candidate, &installation.selection)?;
    let client = download::client()?;
    // This is reached only by the explicit download command, never search/scan.
    install_selection(
        root,
        &label,
        &installation.selection,
        |directory, artifact| download::download(&client, directory, artifact).map(|_| ()),
    )
    .map(Some)
}

fn same_location(left: &Path, right: &Path) -> bool {
    left == right
        || fs::canonicalize(left)
            .ok()
            .zip(fs::canonicalize(right).ok())
            .is_some_and(|(left, right)| left == right)
}

fn check_alias(
    previous: &registry::Entry,
    entry: &registry::Entry,
    label: &str,
) -> Result<(), String> {
    if !same_location(&previous.path, &entry.path)
        || match (&previous.projector, &entry.projector) {
            (None, None) => false,
            (Some(left), Some(right)) => !same_location(left, right),
            _ => true,
        }
    {
        return Err(format!(
            "Model alias '{label}' already refers to another installation; it will not be overwritten."
        ));
    }
    Ok(())
}

fn install_selection(
    root: &Path,
    label: &str,
    selection: &catalog::Selection,
    mut fetch: impl FnMut(&Path, &download::Artifact) -> Result<(), String>,
) -> Result<DownloadResult, String> {
    let target = root.join(&selection.relative_directory);
    let entry = registry::Entry {
        path: target.join(
            &selection
                .artifacts
                .first()
                .ok_or("No model weights in selection.")?
                .filename,
        ),
        projector: selection
            .artifacts
            .iter()
            .find(|a| a.filename == "mmproj.gguf")
            .map(|a| target.join(&a.filename)),
    };
    let aliases = registry::read(&root)?;
    if let Some(previous) = aliases.get(label) {
        check_alias(previous, &entry, label)?;
    }
    let directory = download::directory(&root, &selection.relative_directory)?;
    download::save_metadata(&directory, selection.metadata_filename, &selection.metadata)?;
    for artifact in &selection.artifacts {
        fetch(&directory, artifact)?;
    }
    // No early return for an existing alias: it may have lost a projector/license.
    // Re-read before publication so alias races remain no-clobber.
    let model = match registry::read(root)?.get(label) {
        Some(previous) => {
            check_alias(previous, &entry, label)?;
            label.to_owned()
        }
        None => registry::put(root, label, entry.clone())?,
    };
    Ok(DownloadResult {
        path: path_text(&entry.path)?,
        model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn local(path: &Path, name: &str, source: &str) -> DiscoveredModel {
        DiscoveredModel {
            id: format!("lma:{}", path.display()),
            name: name.into(),
            source: source.into(),
            model: path.to_str().unwrap().into(),
            ollama: false,
            ollama_dir: None,
            projector: None,
            size_bytes: 11,
        }
    }
    const WEIGHTS: &[u8] = b"GGUFfixture";
    const PROJECTOR: &[u8] = b"GGUFprojector";
    const LICENSE: &[u8] = b"fixture license";

    fn hf(id: &str, filename: &str) -> Candidate {
        Candidate {
            source: Source::LmStudio,
            id: id.into(),
            exact: true,
            metadata: Some(serde_json::json!({
                "id": id, "sha": "a".repeat(40), "siblings": [{
                    "rfilename": filename, "lfs": {"sha256": digest(WEIGHTS), "size": WEIGHTS.len()}
                }]
            })),
        }
    }

    fn ollama() -> Candidate {
        Candidate {
            source: Source::Ollama,
            id: "library/example:latest".into(),
            exact: true,
            metadata: Some(serde_json::json!({"schemaVersion": 2, "layers": [
                {"mediaType": "application/vnd.ollama.image.model", "digest": format!("sha256:{}", digest(WEIGHTS)), "size": WEIGHTS.len()},
                {"mediaType": "application/vnd.ollama.image.projector", "digest": format!("sha256:{}", digest(PROJECTOR)), "size": PROJECTOR.len()},
                {"mediaType": "application/vnd.ollama.image.license", "digest": format!("sha256:{}", digest(LICENSE)), "size": LICENSE.len()}
            ]})),
        }
    }

    fn artifact_bytes(filename: &str) -> &'static [u8] {
        match filename {
            "mmproj.gguf" => PROJECTOR,
            "license-0.txt" => LICENSE,
            _ => WEIGHTS,
        }
    }

    fn save_fixture(directory: &Path, selection: &catalog::Selection) {
        fs::create_dir_all(directory).unwrap();
        download::save_metadata(directory, selection.metadata_filename, &selection.metadata)
            .unwrap();
        for artifact in &selection.artifacts {
            let path = directory.join(&artifact.filename);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, artifact_bytes(&artifact.filename)).unwrap();
        }
    }

    #[test]
    fn aliases_and_directory_suffixes_are_not_source_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("owner/repository/model-Q8.gguf");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, WEIGHTS).unwrap();
        for name in ["owner/repository:known-file", "example:latest"] {
            let installed = vec![local(&path, name, "lma")];
            assert!(
                !is_installed(
                    None,
                    Source::LmStudio,
                    "owner/repository",
                    Some("model-Q8.gguf"),
                    &installed
                )
                .unwrap()
            );
            assert!(
                !is_installed(None, Source::Ollama, "example:latest", None, &installed).unwrap()
            );
        }
        let path = dir.path().join("ollama/library/example/latest/model.gguf");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, WEIGHTS).unwrap();
        assert!(
            !is_installed(
                Some(dir.path()),
                Source::Ollama,
                "example",
                None,
                &[local(&path, "example:latest", "lma")]
            )
            .unwrap()
        );
    }

    #[test]
    fn receipt_reuse_is_exact_offline_and_does_not_copy_or_register_external_weights() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("managed");
        let external = dir.path().join("external");
        let candidate = hf("owner/repository", "nested/model-Q8.gguf");
        let selection = selected(&candidate, Some("nested/model-Q8.gguf")).unwrap();
        save_fixture(&external, &selection);
        let path = external.join("nested/model-Q8.gguf");
        let installed = vec![local(&path, "arbitrary-alias:latest", "lm-studio")];
        let result = reuse_local(
            &root,
            Source::LmStudio,
            "owner/repository",
            Some("nested/model-Q8.gguf"),
            &installed,
        )
        .unwrap()
        .unwrap();
        assert_eq!(Path::new(&result.path), path);
        assert!(!root.exists());
        assert!(!external.join(".registry-v1.json").exists());
        for (id, file) in [
            ("owner/other", "nested/model-Q8.gguf"),
            ("owner/repository", "nested/model-Q4.gguf"),
            ("owner/repository", "model-Q8.gguf"),
        ] {
            assert!(
                !is_installed(Some(&root), Source::LmStudio, id, Some(file), &installed).unwrap()
            );
        }
    }

    #[test]
    fn selected_upstream_hf_file_reuses_unreceipted_external_weights_without_fetch_or_writes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("managed");
        let external = dir.path().join("ordinary-lm-studio-folder");
        fs::create_dir(&external).unwrap();
        let path = external.join("model-Q8.gguf");
        fs::write(&path, WEIGHTS).unwrap();
        let candidate = hf("owner/repository", "nested/model-Q8.gguf");
        let selection = selected(&candidate, Some("nested/model-Q8.gguf")).unwrap();
        let installed = vec![local(&path, "unrelated-user-alias:latest", "lm-studio")];
        // Search does not hash unreceipted models or claim an upstream identity.
        let model = describe(&candidate, Some(&root), &installed, &mut Vec::new());
        assert!(!model.installed && !model.files[0].installed);
        let result = download_selected(&root, &candidate, &selection, &installed, |_, _| {
            panic!("an identical external file must not invoke the downloader")
        })
        .unwrap();
        let canonical = fs::canonicalize(&path).unwrap();
        assert_eq!(Path::new(&result.path), canonical);
        assert_eq!(result.path, result.model);
        assert_eq!(fs::read(&path).unwrap(), WEIGHTS);
        assert_eq!(fs::read_dir(&external).unwrap().count(), 1);
        assert!(!root.exists());
    }

    #[test]
    fn selected_upstream_hf_file_does_not_reuse_wrong_hash_variant_size_or_selector() {
        let dir = tempfile::tempdir().unwrap();
        let candidate = hf("owner/repository", "model-Q8.gguf");
        let selection = selected(&candidate, Some("model-Q8.gguf")).unwrap();
        for (case, filename, bytes, declared_size, ollama) in [
            (
                "wrong-hash",
                "model-Q8.gguf",
                b"GGUFforgery".as_slice(),
                WEIGHTS.len() as u64,
                false,
            ),
            (
                "wrong-variant",
                "model-Q4.gguf",
                WEIGHTS,
                WEIGHTS.len() as u64,
                false,
            ),
            (
                "stale-declared-size",
                "model-Q8.gguf",
                WEIGHTS,
                WEIGHTS.len() as u64 + 1,
                false,
            ),
            (
                "wrong-actual-size",
                "model-Q8.gguf",
                b"GGUFshort".as_slice(),
                WEIGHTS.len() as u64,
                false,
            ),
            (
                "ollama-selector",
                "model-Q8.gguf",
                WEIGHTS,
                WEIGHTS.len() as u64,
                true,
            ),
        ] {
            let directory = dir.path().join(case);
            fs::create_dir(&directory).unwrap();
            let path = directory.join(filename);
            fs::write(&path, bytes).unwrap();
            let mut model = local(&path, "owner/repository:pretend-Q8", "lm-studio");
            model.size_bytes = declared_size;
            model.ollama = ollama;
            let root = directory.join("managed");
            let mut fetches = 0;
            let error =
                download_selected(&root, &candidate, &selection, &[model], |_, artifact| {
                    fetches += 1;
                    assert_eq!(artifact.filename, "model-Q8.gguf");
                    Err("explicit download fallback".into())
                })
                .unwrap_err();
            assert_eq!(error, "explicit download fallback", "{case}");
            assert_eq!(fetches, 1, "{case}");
            assert_eq!(fs::read(&path).unwrap(), bytes, "{case}");
            assert!(!root.join(".registry-v1.json").exists(), "{case}");
        }
        let relative = local(
            Path::new("model-Q8.gguf"),
            "owner/repository:relative",
            "lma",
        );
        assert!(reuse_selected_file(&selection, &[relative]).is_none());
    }

    #[test]
    fn selected_upstream_hf_file_rejects_matching_hash_without_gguf_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model-Q8.gguf");
        let bytes = b"not-a-gguf!";
        assert_eq!(bytes.len(), WEIGHTS.len());
        fs::write(&path, bytes).unwrap();
        let mut candidate = hf("owner/repository", "model-Q8.gguf");
        candidate.metadata.as_mut().unwrap()["siblings"][0]["lfs"]["sha256"] = digest(bytes).into();
        let selection = selected(&candidate, Some("model-Q8.gguf")).unwrap();
        assert!(
            reuse_selected_file(&selection, &[local(&path, "model-Q8", "lm-studio")]).is_none()
        );
    }

    #[test]
    fn hf_single_file_requires_filename_and_never_substitutes_repository_or_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let old = hf("owner/repository", "model-Q4.gguf");
        let selection = selected(&old, Some("model-Q4.gguf")).unwrap();
        let directory = dir.path().join(&selection.relative_directory);
        save_fixture(&directory, &selection);
        let path = directory.join("model-Q4.gguf");
        let mut installed = vec![local(&path, "owner/repository:old-Q4", "lma")];
        let mut reference = local(&path, "hf.co/owner/repository:q8", "ollama");
        reference.ollama = true;
        reference.model = reference.name.clone();
        reference.ollama_dir = Some(dir.path().into());
        installed.push(reference);
        let current = hf("owner/repository", "model-Q8.gguf");
        assert!(selected(&current, None).is_err());
        assert!(
            download("owner/repository", "lm-studio", None, &installed)
                .unwrap_err()
                .contains("exact")
        );
        let mut warnings = Vec::new();
        let model = describe(&current, Some(dir.path()), &installed, &mut warnings);
        assert_eq!(model.files.len(), 1);
        assert!(!model.installed);
        assert!(!model.files[0].installed);
        assert!(warnings.is_empty());
        let model = describe(&old, Some(dir.path()), &installed, &mut warnings);
        assert!(model.installed && model.files[0].installed);
    }

    #[test]
    fn complete_managed_receipt_reuse_checks_projector_license_and_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let candidate = ollama();
        let selection = selected(&candidate, None).unwrap();
        let directory = dir.path().join(&selection.relative_directory);
        save_fixture(&directory, &selection);
        assert!(is_installed(Some(dir.path()), Source::Ollama, &candidate.id, None, &[]).unwrap());
        assert!(
            reuse_local(dir.path(), Source::Ollama, &candidate.id, None, &[])
                .unwrap()
                .is_some()
        );
        assert!(!dir.path().join(".registry-v1.json").exists());
        fs::remove_file(directory.join("license-0.txt")).unwrap();
        assert!(!is_installed(Some(dir.path()), Source::Ollama, &candidate.id, None, &[]).unwrap());
        fs::write(directory.join("license-0.txt"), LICENSE).unwrap();
        fs::remove_file(directory.join("mmproj.gguf")).unwrap();
        assert!(!is_installed(Some(dir.path()), Source::Ollama, &candidate.id, None, &[]).unwrap());
        fs::write(directory.join("mmproj.gguf"), PROJECTOR).unwrap();
        fs::write(directory.join("model.gguf"), b"GGUFforgery").unwrap();
        assert!(
            reuse_local(dir.path(), Source::Ollama, &candidate.id, None, &[])
                .unwrap_err()
                .contains("SHA256")
        );
    }

    #[test]
    fn corrupted_or_wrong_origin_receipts_do_not_authorize_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let selection = selected(&hf("owner/other", "model.gguf"), Some("model.gguf")).unwrap();
        save_fixture(dir.path(), &selection);
        let path = dir.path().join("model.gguf");
        let installed = vec![local(&path, "owner/repository:forged-alias", "lma")];
        assert!(
            !is_installed(
                None,
                Source::LmStudio,
                "owner/repository",
                Some("model.gguf"),
                &installed
            )
            .unwrap()
        );
        let receipt = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .unwrap();
        fs::write(receipt, b"{}").unwrap();
        assert!(
            reuse_local(
                dir.path(),
                Source::LmStudio,
                "owner/other",
                Some("model.gguf"),
                &installed
            )
            .unwrap_err()
            .contains("provenance checksum")
        );
    }

    #[test]
    fn interrupted_install_and_existing_alias_resume_without_redownloading_verified_weights() {
        let dir = tempfile::tempdir().unwrap();
        let candidate = ollama();
        let mut selection = selected(&candidate, None).unwrap();
        let label = logical_name(&candidate, &selection).unwrap();
        let directory = dir.path().join(&selection.relative_directory);
        save_fixture(&directory, &selection);
        registry::put(
            dir.path(),
            &label,
            registry::Entry {
                path: directory.join("model.gguf"),
                projector: Some(directory.join("mmproj.gguf")),
            },
        )
        .unwrap();
        fs::remove_file(directory.join("mmproj.gguf")).unwrap();
        fs::remove_file(directory.join("license-0.txt")).unwrap();
        let before = fs::read(dir.path().join(".registry-v1.json")).unwrap();
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        for artifact in &mut selection.artifacts {
            artifact.url =
                reqwest::Url::parse("http://127.0.0.1:1/must-not-download-weights").unwrap();
        }
        let mut visited = Vec::new();
        let error = install_selection(dir.path(), &label, &selection, |directory, artifact| {
            visited.push(artifact.filename.clone());
            if artifact.filename != "model.gguf" {
                return Err("simulated interruption".into());
            }
            download::download(&client, directory, artifact).map(|_| ())
        })
        .unwrap_err();
        assert_eq!(error, "simulated interruption");
        assert_eq!(visited, ["model.gguf", "mmproj.gguf"]);
        assert_eq!(fs::read(directory.join("model.gguf")).unwrap(), WEIGHTS);
        assert!(!is_installed(Some(dir.path()), Source::Ollama, &candidate.id, None, &[]).unwrap());

        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let mut requests = Vec::new();
            while requests.len() < 2 && std::time::Instant::now() < deadline {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("{error}"),
                };
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(std::time::Duration::from_secs(2)))
                    .unwrap();
                let mut request = [0; 4096];
                let count = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..count]);
                let path = request
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .trim_start_matches('/')
                    .to_owned();
                let bytes = artifact_bytes(&path);
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                )
                .unwrap();
                stream.write_all(bytes).unwrap();
                requests.push(path);
            }
            requests
        });
        for artifact in &mut selection.artifacts {
            artifact.url =
                reqwest::Url::parse(&format!("http://{address}/{}", artifact.filename)).unwrap();
        }
        let result = install_selection(dir.path(), &label, &selection, |directory, artifact| {
            download::download(&client, directory, artifact).map(|_| ())
        });
        let requests = server.join().unwrap();
        result.unwrap();
        assert_eq!(requests, ["mmproj.gguf", "license-0.txt"]);
        assert_eq!(
            fs::read(dir.path().join(".registry-v1.json")).unwrap(),
            before
        );
        assert!(
            reuse_local(dir.path(), Source::Ollama, &candidate.id, None, &[])
                .unwrap()
                .is_some()
        );
    }
    #[test]
    fn new_interrupted_install_is_not_published_or_reported_as_installed() {
        let dir = tempfile::tempdir().unwrap();
        let candidate = ollama();
        let mut selection = selected(&candidate, None).unwrap();
        let label = logical_name(&candidate, &selection).unwrap();
        let directory = dir.path().join(&selection.relative_directory);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("model.gguf"), WEIGHTS).unwrap();
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        for artifact in &mut selection.artifacts {
            artifact.url = reqwest::Url::parse("http://127.0.0.1:1/must-not-redownload").unwrap();
        }
        let error = install_selection(dir.path(), &label, &selection, |directory, artifact| {
            if artifact.filename != "model.gguf" {
                return Err("interrupted".into());
            }
            download::download(&client, directory, artifact).map(|_| ())
        })
        .unwrap_err();
        assert_eq!(error, "interrupted");
        assert!(registry::read(dir.path()).unwrap().is_empty());
        assert!(!is_installed(Some(dir.path()), Source::Ollama, &candidate.id, None, &[]).unwrap());
        let pending =
            local_installs(Some(dir.path()), Source::Ollama, &candidate.id, None, &[]).unwrap();
        assert_eq!(pending.len(), 1);
        assert!(!complete(&pending[0], true).unwrap());
        assert_eq!(fs::read(directory.join("model.gguf")).unwrap(), WEIGHTS);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_receipts_and_artifacts_are_not_authoritative_installations() {
        let dir = tempfile::tempdir().unwrap();
        let candidate = hf("owner/repository", "model.gguf");
        let selection = selected(&candidate, Some("model.gguf")).unwrap();
        let directory = dir.path().join(&selection.relative_directory);
        save_fixture(&directory, &selection);
        let receipt = fs::read_dir(&directory)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .unwrap();
        let target = dir.path().join("receipt-target");
        fs::rename(&receipt, &target).unwrap();
        std::os::unix::fs::symlink(&target, &receipt).unwrap();
        assert!(
            reuse_local(
                dir.path(),
                Source::LmStudio,
                &candidate.id,
                Some("model.gguf"),
                &[]
            )
            .unwrap_err()
            .contains("non-symlink")
        );
        fs::remove_file(&receipt).unwrap();
        fs::rename(&target, &receipt).unwrap();
        let weights = directory.join("model.gguf");
        let target = dir.path().join("weights-target");
        fs::rename(&weights, &target).unwrap();
        std::os::unix::fs::symlink(&target, &weights).unwrap();
        assert!(
            reuse_local(
                dir.path(),
                Source::LmStudio,
                &candidate.id,
                Some("model.gguf"),
                &[]
            )
            .unwrap_err()
            .contains("differs")
        );
        assert_eq!(fs::read(&target).unwrap(), WEIGHTS);
    }

    #[test]
    fn custom_ollama_installation_is_reused_without_network_or_copying() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir
            .path()
            .join("manifests/registry.ollama.ai/library/example/latest");
        fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        fs::create_dir_all(dir.path().join("blobs")).unwrap();
        let digest = "a".repeat(64);
        let weights = dir.path().join(format!("blobs/sha256-{digest}"));
        fs::write(&weights, b"GGUFfixture").unwrap();
        fs::write(&manifest, serde_json::to_vec(&serde_json::json!({
            "schemaVersion":2, "layers":[{"mediaType":"application/vnd.ollama.image.model", "digest":format!("sha256:{digest}")}]
        })).unwrap()).unwrap();
        let catalog = crate::models::discover(crate::models::DiscoveryOptions {
            local_dirs: vec![],
            ollama_dir: Some(dir.path().to_path_buf()),
        });
        assert_eq!(catalog.models.len(), 1);
        let result = download("library/example:latest", "ollama", None, &catalog.models).unwrap();
        assert_eq!(
            result.path,
            fs::canonicalize(&weights).unwrap().to_str().unwrap()
        );
        assert_eq!(result.model, "example:latest");
        assert_eq!(fs::read_dir(dir.path().join("blobs")).unwrap().count(), 1);
        assert!(!dir.path().join(".registry-v1.json").exists());
        // A stale catalog selector is not sufficient: the real manifest must
        // still resolve, and failure must not silently become a fresh download.
        fs::remove_file(&manifest).unwrap();
        assert!(download("library/example:latest", "ollama", None, &catalog.models).is_err());
        assert_eq!(fs::read_dir(dir.path().join("blobs")).unwrap().count(), 1);
    }

    #[test]
    fn store_payload_is_camel_case_and_unknown_sizes_are_null() {
        let value = serde_json::to_value(SearchResult {
            models: vec![StoreModel {
                id: "owner/repo".into(),
                source: "lm-studio".into(),
                name: "owner/repo".into(),
                exact: false,
                files: vec![StoreFile {
                    name: "model.gguf".into(),
                    bytes: None,
                    installed: false,
                }],
                installed: false,
                total_bytes: None,
            }],
            warnings: vec![],
        })
        .unwrap();
        assert_eq!(value["models"][0]["totalBytes"], serde_json::Value::Null);
        assert_eq!(
            value["models"][0]["files"][0]["bytes"],
            serde_json::Value::Null
        );
        assert_eq!(value["models"][0].as_object().unwrap().len(), 7);
        assert_eq!(
            serde_json::to_value(DownloadResult {
                path: "/weights.gguf".into(),
                model: "model:latest".into()
            })
            .unwrap(),
            serde_json::json!({"path":"/weights.gguf", "model":"model:latest"})
        );
    }

    #[test]
    fn hf_choices_are_explicit_and_typed_even_when_cli_has_a_preferred_quant() {
        let candidate = Candidate {
            source: Source::LmStudio,
            id: "owner/repo".into(),
            exact: true,
            metadata: Some(
                serde_json::json!({"id":"owner/repo", "sha":"a".repeat(40), "siblings":[
                    {"rfilename":"model-Q4_K_M.gguf","lfs":{"sha256":"b".repeat(64),"size":20}},
                    {"rfilename":"model-Q8.gguf","lfs":{"sha256":"c".repeat(64),"size":40}}
                ]}),
            ),
        };
        let choices = files(&candidate);
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0].bytes, Some(20));
        assert!(selected(&candidate, None).is_err());
        let selection = selected(&candidate, Some("model-Q8.gguf")).unwrap();
        assert!(
            logical_name(&candidate, &selection)
                .unwrap()
                .starts_with("owner/repo:model-Q8-")
        );
        assert!(selected(&candidate, Some("../bad.gguf")).is_err());
        assert!(exact_identity("search terms", Source::LmStudio).is_err());
        assert!(sources("all", false).is_err());
    }
}
