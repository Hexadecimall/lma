//! Ollama exact manifests and public Hugging Face GGUF discovery for LM Studio.
//! There is no separate proprietary LM Studio registry.

use reqwest::Url;
use reqwest::blocking::Client;
use serde_json::Value;

use super::download::{self, Artifact, safe_component, safe_relative};

const OLLAMA: &str = "https://registry.ollama.ai";
const HF: &str = "https://huggingface.co";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Source {
    Ollama,
    LmStudio,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LmStudio => "huggingface (LM Studio-compatible GGUF)",
        }
    }

    pub fn flag(self) -> &'static str {
        match self {
            Self::Ollama => "-oe",
            Self::LmStudio => "-le",
        }
    }
}

#[derive(Debug)]
pub(super) struct Candidate {
    pub source: Source,
    pub id: String,
    pub exact: bool,
    pub metadata: Option<Value>,
}

#[derive(Default, Debug)]
pub(super) struct Catalog {
    pub candidates: Vec<Candidate>,
    pub warnings: Vec<String>,
    pub succeeded: usize,
    // A failed exact lookup could conceal another matching model. Listing can
    // still show the other source, but downloading must require a source flag.
    pub failed_exact: bool,
}

pub(super) fn discover(client: &Client, query: &str, ollama: bool, lm_studio: bool) -> Catalog {
    discover_with_limit(client, query, ollama, lm_studio, 10)
}

pub(super) fn discover_with_limit(
    client: &Client,
    query: &str,
    ollama: bool,
    lm_studio: bool,
    limit: u32,
) -> Catalog {
    let mut catalog = Catalog::default();
    if ollama {
        catalog.include(
            Source::Ollama,
            Reference::parse(query).is_some(),
            ollama_candidates(client, query),
        );
    }
    if lm_studio {
        catalog.include(
            Source::LmStudio,
            hf_repo(query),
            hf_candidates(client, query, limit),
        );
    }
    catalog.sort_dedup();
    catalog
}

impl Catalog {
    fn include(&mut self, source: Source, exact: bool, result: Result<Vec<Candidate>, String>) {
        match result {
            Ok(mut candidates) => {
                self.succeeded += 1;
                self.candidates.append(&mut candidates);
            }
            Err(error) => {
                self.failed_exact |= exact;
                self.warnings.push(format!("{}: {error}", source.label()));
            }
        }
    }

    fn sort_dedup(&mut self) {
        self.candidates
            .sort_by(|a, b| (a.source, &a.id, !a.exact).cmp(&(b.source, &b.id, !b.exact)));
        self.candidates
            .dedup_by(|a, b| a.source == b.source && a.id == b.id);
    }
}

#[derive(Debug)]
struct Reference {
    namespace: String,
    model: String,
    tag: String,
}

impl Reference {
    fn parse(query: &str) -> Option<Self> {
        let (name, tag) = query.rsplit_once(':').unwrap_or((query, "latest"));
        let parts: Vec<_> = name.split('/').collect();
        let (namespace, model) = match parts.as_slice() {
            [model] => ("library", *model),
            [namespace, model] => (*namespace, *model),
            _ => return None,
        };
        if ![namespace, model, tag].iter().all(|s| safe_component(s)) {
            return None;
        }
        Some(Self {
            namespace: namespace.into(),
            model: model.into(),
            tag: tag.into(),
        })
    }

    fn id(&self) -> String {
        format!("{}/{}:{}", self.namespace, self.model, self.tag)
    }

    fn path(&self) -> String {
        format!("{}/{}/{}", self.namespace, self.model, self.tag)
    }

    fn manifest_url(&self) -> Result<Url, String> {
        url(&format!(
            "{OLLAMA}/v2/{}/{}/manifests/{}",
            self.namespace, self.model, self.tag
        ))
    }
}

fn url(text: &str) -> Result<Url, String> {
    Url::parse(text).map_err(|e| format!("cannot construct service URL: {e}"))
}

fn ollama_candidates(client: &Client, query: &str) -> Result<Vec<Candidate>, String> {
    let Some(reference) = Reference::parse(query) else {
        return Ok(vec![]);
    };
    let Some(metadata) = download::json(client, reference.manifest_url()?)? else {
        return Ok(vec![]);
    };
    // Catch error documents/malformed responses even for --list, without
    // requiring a model to be supported just to display its exact reference.
    if metadata.get("schemaVersion").and_then(Value::as_u64) != Some(2)
        || !metadata.get("layers").is_some_and(Value::is_array)
    {
        return Err("expected a Docker schema-2 manifest with layers".into());
    }
    Ok(vec![Candidate {
        source: Source::Ollama,
        id: reference.id(),
        exact: true,
        metadata: Some(metadata),
    }])
}

fn hf_repo(query: &str) -> bool {
    query.split('/').count() == 2 && query.split('/').all(safe_component)
}

fn hf_candidates(client: &Client, query: &str, limit: u32) -> Result<Vec<Candidate>, String> {
    if hf_repo(query) {
        let mut endpoint = url(&format!("{HF}/api/models/{query}"))?;
        endpoint.query_pairs_mut().append_pair("blobs", "true");
        let Some(metadata) = download::json(client, endpoint)? else {
            return Ok(vec![]);
        };
        if metadata.get("id").and_then(Value::as_str) != Some(query) {
            return Err("exact repository metadata has a different/missing id; use its canonical owner/repository explicitly".into());
        }
        return Ok(vec![Candidate {
            source: Source::LmStudio,
            id: query.into(),
            exact: true,
            metadata: Some(metadata),
        }]);
    }
    let mut endpoint = url(&format!("{HF}/api/models"))?;
    endpoint
        .query_pairs_mut()
        .append_pair("search", query)
        .append_pair("filter", "gguf")
        .append_pair("sort", "downloads")
        .append_pair("direction", "-1")
        .append_pair("limit", &limit.to_string());
    let metadata =
        download::json(client, endpoint)?.ok_or("Hugging Face search endpoint returned 404")?;
    let results = metadata
        .as_array()
        .ok_or("expected a Hugging Face search result array")?;
    results
        .iter()
        .take(limit as usize)
        .map(|entry| {
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .ok_or("search result has no repository id")?;
            if !hf_repo(id) {
                return Err("search result has an unsafe repository id".into());
            }
            Ok(Candidate {
                source: Source::LmStudio,
                id: id.into(),
                exact: false,
                metadata: None,
            })
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct PopularModel {
    pub id: String,
    pub downloads: u64,
}

fn popular_url(limit: u32) -> Result<Url, String> {
    if !(1..=1000).contains(&limit) {
        return Err("index must be between 1 and 1000".into());
    }
    let mut endpoint = url(&format!("{HF}/api/models"))?;
    endpoint
        .query_pairs_mut()
        .append_pair("filter", "gguf")
        .append_pair("sort", "downloads")
        .append_pair("direction", "-1")
        .append_pair("limit", &limit.to_string());
    Ok(endpoint)
}

pub(super) fn popular(client: &Client, limit: u32) -> Result<Vec<PopularModel>, String> {
    let metadata = download::json(client, popular_url(limit)?)?
        .ok_or("Hugging Face popularity endpoint returned 404")?;
    parse_popular(&metadata, limit)
}

fn parse_popular(metadata: &Value, limit: u32) -> Result<Vec<PopularModel>, String> {
    let entries = metadata
        .as_array()
        .ok_or("expected a Hugging Face popularity result array")?;
    let mut results: Vec<_> = entries
        .iter()
        .take(limit as usize)
        .map(|entry| {
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| hf_repo(id))
                .ok_or("popularity result has a missing or unsafe repository id")?;
            let downloads = entry
                .get("downloads")
                .and_then(Value::as_u64)
                .ok_or("popularity result has a missing or invalid download count")?;
            Ok(PopularModel {
                id: id.into(),
                downloads,
            })
        })
        .collect::<Result<_, String>>()?;
    results.sort_by(|a, b| b.downloads.cmp(&a.downloads).then_with(|| a.id.cmp(&b.id)));
    let mut seen = std::collections::HashSet::new();
    results.retain(|model| seen.insert(model.id.clone()));
    Ok(results)
}

pub(super) struct Selection {
    pub relative_directory: String,
    pub artifacts: Vec<Artifact>,
    pub metadata_filename: &'static str,
    pub metadata: Vec<u8>,
    pub projector_only: bool,
    pub notes: Vec<String>,
}

pub(super) fn select(candidate: &Candidate, file: Option<&str>) -> Result<Selection, String> {
    let metadata = candidate
        .metadata
        .as_ref()
        .ok_or("select a full repository/reference; fuzzy search results cannot be downloaded")?;
    match candidate.source {
        Source::Ollama => {
            if file.is_some() {
                return Err("-f applies only to Hugging Face GGUF repositories; use -le".into());
            }
            select_ollama(&candidate.id, metadata)
        }
        Source::LmStudio => select_hf(&candidate.id, metadata, file),
    }
}

fn snapshot(origin: &str, metadata: &Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(&serde_json::json!({ "origin": origin, "metadata": metadata }))
        .map_err(|e| format!("cannot serialize provenance: {e}"))
}

fn select_ollama(id: &str, metadata: &Value) -> Result<Selection, String> {
    let reference = Reference::parse(id).ok_or("unsafe Ollama reference")?;
    if metadata.get("schemaVersion").and_then(Value::as_u64) != Some(2) {
        return Err("only Docker schema-2 Ollama manifests are supported".into());
    }
    let layers = metadata
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("manifest has no layers")?;
    let mut models = Vec::new();
    let mut projectors = Vec::new();
    let mut licenses = Vec::new();
    for layer in layers {
        let media = layer
            .get("mediaType")
            .and_then(Value::as_str)
            .ok_or("layer has no mediaType")?;
        match media {
            "application/vnd.ollama.image.model" => models.push(layer),
            "application/vnd.ollama.image.projector" => projectors.push(layer),
            "application/vnd.ollama.image.license" => licenses.push(layer),
            "application/vnd.ollama.image.template"
            | "application/vnd.ollama.image.system"
            | "application/vnd.ollama.image.params"
            | "application/vnd.ollama.image.messages" => {}
            _ => {
                return Err(format!(
                    "unsupported Ollama layer type {media:?}; adapters and non-GGUF/multi-model bundles are not supported"
                ));
            }
        }
    }
    if models.len() != 1 || projectors.len() > 1 {
        return Err(format!(
            "expected exactly one model and at most one projector; found {} models, {} projectors (multi-model bundles unsupported)",
            models.len(),
            projectors.len()
        ));
    }
    let artifact = |layer: &Value, filename: String, gguf: bool| -> Result<Artifact, String> {
        let digest = layer
            .get("digest")
            .and_then(Value::as_str)
            .and_then(|s| s.strip_prefix("sha256:"))
            .ok_or("Ollama layer lacks a SHA256 digest; refusing unverified content")?;
        let checksum = download::sha256(digest)?;
        let size = layer
            .get("size")
            .and_then(Value::as_u64)
            .ok_or("Ollama layer lacks a valid size")?;
        if gguf && size < 4 {
            return Err("Ollama GGUF layer is too small".into());
        }
        if !gguf && size > download::JSON_LIMIT {
            return Err("Ollama license exceeds 8 MiB".into());
        }
        Ok(Artifact {
            url: url(&format!(
                "{OLLAMA}/v2/{}/{}/blobs/sha256:{checksum}",
                reference.namespace, reference.model
            ))?,
            filename,
            sha256: checksum,
            size,
            gguf,
        })
    };
    let mut artifacts = vec![artifact(models[0], "model.gguf".into(), true)?];
    if let Some(projector) = projectors.first() {
        artifacts.push(artifact(projector, "mmproj.gguf".into(), true)?);
    }
    for (index, license) in licenses.iter().enumerate() {
        artifacts.push(artifact(license, format!("license-{index}.txt"), false)?);
    }
    Ok(Selection {
        relative_directory: format!("ollama/{}", reference.path()),
        artifacts,
        metadata_filename: "ollama-origin.json",
        metadata: snapshot(reference.manifest_url()?.as_str(), metadata)?,
        projector_only: false,
        notes: vec![
            "Ollama templates, system prompts, parameters, and message history are not applied; GGUF loading is not Ollama inference compatibility.".into(),
            "Origin manifest and available license layers are preserved. Review the upstream model license before use.".into(),
        ],
    })
}

fn is_projector(filename: &str) -> bool {
    let name = filename.to_ascii_lowercase();
    name.contains("mmproj") || name.contains("projector")
}

fn is_adapter(filename: &str) -> bool {
    let name = filename.to_ascii_lowercase();
    name.contains("adapter") || name.contains("lora")
}

fn is_split(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    let stem = lower.strip_suffix(".gguf").unwrap_or(&lower);
    // Standard llama.cpp split names are *-00001-of-00002.gguf. Also reject
    // variants rather than downloading one shard and presenting it as a model.
    stem.contains("-of-") || stem.ends_with("-split")
}

fn commit_sha(metadata: &Value) -> Result<&str, String> {
    let sha = metadata
        .get("sha")
        .and_then(Value::as_str)
        .ok_or("Hugging Face metadata lacks a commit SHA; refusing a mutable download")?;
    if sha.len() != 40 || !sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid Hugging Face commit SHA; refusing a mutable download".into());
    }
    Ok(sha)
}

fn select_hf(id: &str, metadata: &Value, requested: Option<&str>) -> Result<Selection, String> {
    if !hf_repo(id) {
        return Err("unsafe Hugging Face repository id".into());
    }
    if metadata.get("private").and_then(Value::as_bool) == Some(true)
        || metadata
            .get("gated")
            .is_some_and(|g| !g.is_null() && g != &Value::Bool(false))
    {
        return Err("private/gated Hugging Face repositories are unsupported; choose a public ungated repository".into());
    }
    let commit = commit_sha(metadata)?;
    let siblings = metadata
        .get("siblings")
        .and_then(Value::as_array)
        .ok_or("Hugging Face metadata has no file list")?;
    if metadata
        .get("tags")
        .and_then(Value::as_array)
        .is_some_and(|tags| {
            tags.iter().filter_map(Value::as_str).any(|tag| {
                matches!(
                    tag.to_ascii_lowercase().as_str(),
                    "peft" | "lora" | "adapter" | "adapters"
                )
            })
        })
        || siblings
            .iter()
            .filter_map(|f| f.get("rfilename").and_then(Value::as_str))
            .any(|name| name.rsplit('/').next() == Some("adapter_config.json"))
    {
        return Err("adapter repositories are unsupported; select a complete GGUF model".into());
    }
    let mut files: Vec<(&str, &Value)> = siblings
        .iter()
        .filter_map(|entry| {
            let name = entry.get("rfilename")?.as_str()?;
            name.to_ascii_lowercase()
                .ends_with(".gguf")
                .then_some((name, entry))
        })
        .collect();
    files.sort_by_key(|(name, _)| *name);
    // Identical duplicate entries are harmless. Conflicting metadata is not.
    for pair in files.windows(2) {
        if pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1 {
            return Err(format!("conflicting metadata for {}", pair[0].0));
        }
    }
    files.dedup_by_key(|entry| entry.0);
    let choices = || {
        if files.is_empty() {
            "  (no GGUF files)".into()
        } else {
            files
                .iter()
                .map(|(name, _)| {
                    format!(
                        "  {name:?}{}{}",
                        if is_split(name) {
                            " [split shards unsupported]"
                        } else {
                            ""
                        },
                        if is_projector(name) {
                            " [projector; fetch separately]"
                        } else {
                            ""
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    };
    let selected = if let Some(requested) = requested {
        safe_relative(requested)?;
        files
            .iter()
            .find(|(name, _)| *name == requested)
            .copied()
            .ok_or_else(|| {
                format!(
                    "-f must name an exact GGUF filename in {id}. Files:\n{}",
                    choices()
                )
            })?
    } else {
        // Never treat a shard as a standalone candidate, including when only
        // one shard of a preferred quantization appears in the response.
        let models: Vec<_> = files
            .iter()
            .copied()
            .filter(|(name, _)| !is_projector(name) && !is_adapter(name))
            .collect();
        let preferred: Vec<_> = models
            .iter()
            .copied()
            .filter(|(name, _)| name.to_ascii_uppercase().contains("Q4_K_M"))
            .collect();
        if preferred.len() == 1 && !is_split(preferred[0].0) {
            preferred[0]
        } else if models.len() == 1 && !is_split(models[0].0) {
            models[0]
        } else {
            return Err(format!(
                "no unambiguous single-file GGUF in {id}; use -f with an exact filename. Split shards and adapters are unsupported. Files:\n{}",
                choices()
            ));
        }
    };
    let (filename, entry) = selected;
    safe_relative(filename)?;
    if is_split(filename) {
        return Err(
            "split GGUF shards are unsupported; select a standalone GGUF file/repository".into(),
        );
    }
    if is_adapter(filename) {
        return Err("GGUF adapters are unsupported; select a complete model".into());
    }
    let lfs = entry.get("lfs").ok_or("selected GGUF has no LFS SHA256 metadata; choose a repository exposing lfs.sha256 via ?blobs=true (Git blob IDs are not SHA256 checksums)")?;
    let checksum = download::sha256(
        lfs.get("sha256")
            .and_then(Value::as_str)
            .ok_or("selected GGUF lacks lfs.sha256; refusing an unverified download")?,
    )?;
    let size = lfs
        .get("size")
        .and_then(Value::as_u64)
        .ok_or("selected GGUF lacks a valid LFS size")?;
    if size < 4
        || entry
            .get("size")
            .and_then(Value::as_u64)
            .is_some_and(|outer| outer != size)
    {
        return Err("selected GGUF has invalid or conflicting size metadata".into());
    }
    let mut notes = vec![
        "LM Studio source means public Hugging Face GGUF discovery, not a separate LM Studio registry. Runtime architecture support is checked by the loader, not by this catalog.".into(),
        format!("Pinned Hugging Face commit: {commit}. Origin metadata (including available license tags/card data) is preserved; review https://huggingface.co/{id}/tree/{commit} for license terms."),
    ];
    let projector_only = is_projector(filename);
    if projector_only {
        notes.push("This is a projector only, not a standalone model. Pair it explicitly with its documented compatible model using `lma s MODEL -pr FILE`.".into());
    } else {
        notes.push("No Hugging Face projector is selected automatically. This download does not imply vision readiness; fetch the documented compatible projector separately with `lma f REPO -le -f FILE`, then serve with `lma s MODEL -pr PROJECTOR`.".into());
    }
    Ok(Selection {
        relative_directory: format!("lm-studio/{id}"),
        artifacts: vec![Artifact {
            url: url(&format!("{HF}/{id}/resolve/{commit}/{filename}"))?,
            filename: filename.into(),
            sha256: checksum,
            size,
            gguf: true,
        }],
        metadata_filename: "hf-origin.json",
        metadata: snapshot(&format!("{HF}/{id}/tree/{commit}"), metadata)?,
        projector_only,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hf(names: &[&str]) -> Value {
        json!({"id":"owner/model", "sha":"a".repeat(40), "siblings": names.iter().map(|name| json!({"rfilename":name, "size":10, "lfs":{"sha256":"b".repeat(64), "size":10}})).collect::<Vec<_>>()})
    }

    #[test]
    fn popularity_endpoint_is_ranked_by_downloads_and_bounded() {
        let endpoint = popular_url(50).unwrap();
        assert_eq!(endpoint.host_str(), Some("huggingface.co"));
        let pairs: std::collections::HashMap<_, _> = endpoint.query_pairs().into_owned().collect();
        assert_eq!(pairs["filter"], "gguf");
        assert_eq!(pairs["sort"], "downloads");
        assert_eq!(pairs["direction"], "-1");
        assert_eq!(pairs["limit"], "50");
        assert!(!pairs.contains_key("search"));
        assert!(popular_url(0).is_err());
        assert!(popular_url(1001).is_err());
    }

    #[test]
    fn popular_models_have_truthful_counts_and_deterministic_bounded_ranking() {
        let metadata = serde_json::json!([
            {"id":"owner/z", "downloads":10},
            {"id":"owner/b", "downloads":100},
            {"id":"owner/a", "downloads":100},
            {"id":"owner/b", "downloads":20},
            {"id":"owner/zero", "downloads":0}
        ]);
        let models = parse_popular(&metadata, 50).unwrap();
        assert_eq!(
            models
                .iter()
                .map(|m| (m.id.as_str(), m.downloads))
                .collect::<Vec<_>>(),
            vec![
                ("owner/a", 100),
                ("owner/b", 100),
                ("owner/z", 10),
                ("owner/zero", 0)
            ]
        );
        assert_eq!(parse_popular(&metadata, 2).unwrap().len(), 2);
        assert!(
            parse_popular(&serde_json::json!([]), 50)
                .unwrap()
                .is_empty()
        );
        for invalid in [
            serde_json::json!({"error":"bad"}),
            serde_json::json!([{"id":"../bad", "downloads":1}]),
            serde_json::json!([{"id":"owner/repo\n", "downloads":1}]),
            serde_json::json!([{"id":"owner/repo"}]),
            serde_json::json!([{"id":"owner/repo", "downloads":null}]),
            serde_json::json!([{"id":"owner/repo", "downloads":-1}]),
            serde_json::json!([{"id":"owner/repo", "downloads":1.5}]),
        ] {
            assert!(parse_popular(&invalid, 50).is_err(), "{invalid}");
        }
    }

    #[test]
    fn ollama_references_are_exact_and_safe() {
        assert_eq!(
            Reference::parse("llama3.2").unwrap().id(),
            "library/llama3.2:latest"
        );
        assert_eq!(
            Reference::parse("team/model:q4").unwrap().path(),
            "team/model/q4"
        );
        for query in [
            "../name",
            "a/b/c",
            "name:%2e%2e",
            "name:",
            "https://host/name",
            "a\\b",
        ] {
            assert!(Reference::parse(query).is_none(), "{query}");
        }
    }

    #[test]
    fn merged_results_deduplicate_only_source_and_exact_id() {
        let mut catalog = Catalog::default();
        for (source, id) in [
            (Source::Ollama, "a:q4"),
            (Source::Ollama, "a:q4"),
            (Source::Ollama, "a:q8"),
            (Source::LmStudio, "a:q4"),
        ] {
            catalog.candidates.push(Candidate {
                source,
                id: id.into(),
                exact: true,
                metadata: None,
            });
        }
        catalog.sort_dedup();
        assert_eq!(catalog.candidates.len(), 3);
        assert_eq!(catalog.candidates[1].id, "a:q8");
        catalog.include(Source::Ollama, true, Err("HTTP 503".into()));
        catalog.include(Source::LmStudio, false, Ok(vec![]));
        assert!(catalog.failed_exact);
        assert_eq!(catalog.succeeded, 1);
        assert_eq!(catalog.warnings.len(), 1);
    }

    #[test]
    fn hf_selection_is_pinned_and_never_equates_quantizations() {
        let metadata = hf(&["model.Q8_0.gguf", "model.Q4_K_M.gguf", "mmproj.gguf"]);
        let selection = select_hf("owner/model", &metadata, None).unwrap();
        assert_eq!(selection.artifacts[0].filename, "model.Q4_K_M.gguf");
        assert!(
            selection.artifacts[0]
                .url
                .path()
                .contains(&format!("/resolve/{}/", "a".repeat(40)))
        );
        assert_eq!(selection.artifacts.len(), 1);
        assert!(
            select_hf(
                "owner/model",
                &hf(&["a.Q4_K_M.gguf", "b.Q4_K_M.gguf"]),
                None
            )
            .is_err()
        );
        assert!(select_hf("owner/model", &hf(&["a.Q8_0.gguf", "b.Q8_0.gguf"]), None).is_err());
        assert!(select_hf("owner/model", &metadata, Some("model.Q8_0.gguf")).is_ok());
        assert!(select_hf("owner/model", &metadata, Some("missing.gguf")).is_err());
        assert!(
            select_hf("owner/model", &metadata, Some("mmproj.gguf"))
                .unwrap()
                .projector_only
        );
        assert!(select_hf("owner/model", &hf(&["mmproj.gguf"]), None).is_err());
    }

    #[test]
    fn hf_rejects_shards_adapters_weak_hashes_and_unsafe_paths() {
        for name in [
            "model-00001-of-00002.gguf",
            "adapter.gguf",
            "../model.gguf",
            "%2e%2e.gguf",
        ] {
            assert!(
                select_hf("owner/model", &hf(&[name]), Some(name)).is_err(),
                "{name}"
            );
            assert!(
                select_hf("owner/model", &hf(&[name]), None).is_err(),
                "{name}"
            );
        }
        let mut metadata = hf(&["model.gguf"]);
        metadata["siblings"][0]["lfs"] = Value::Null;
        assert!(select_hf("owner/model", &metadata, None).is_err());
        let mut metadata = hf(&["model.gguf"]);
        metadata["sha"] = json!("main");
        assert!(select_hf("owner/model", &metadata, None).is_err());
        let mut metadata = hf(&["model.gguf"]);
        metadata["tags"] = json!(["peft"]);
        assert!(select_hf("owner/model", &metadata, None).is_err());
    }

    #[test]
    fn ollama_layers_require_model_and_strong_digests() {
        let model = json!({"mediaType":"application/vnd.ollama.image.model", "digest":format!("sha256:{}", "a".repeat(64)), "size":10});
        let projector = json!({"mediaType":"application/vnd.ollama.image.projector", "digest":format!("sha256:{}", "b".repeat(64)), "size":8});
        let metadata = json!({"schemaVersion":2,"layers":[model.clone(),projector.clone()]});
        let selection = select_ollama("llama3.2", &metadata).unwrap();
        assert_eq!(selection.artifacts.len(), 2);
        assert_eq!(selection.artifacts[1].filename, "mmproj.gguf");
        for layers in [
            json!([]),
            json!([model.clone(), model.clone()]),
            json!([model.clone(), projector.clone(), projector]),
            json!([model.clone(), {"mediaType":"application/vnd.ollama.image.adapter"}]),
        ] {
            assert!(select_ollama("model", &json!({"schemaVersion":2,"layers":layers})).is_err());
        }
        let mut invalid = model;
        invalid["digest"] = json!("sha256:../bad");
        assert!(select_ollama("model", &json!({"schemaVersion":2,"layers":[invalid]})).is_err());
    }
}
