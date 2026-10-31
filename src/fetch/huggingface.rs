//! Generic, explicit Hugging Face asset installation. No built-in model selection.
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
    pub name: String,
    pub bytes: u64,
    pub sha256: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallPlan {
    pub repository: String,
    pub revision: String,
    pub files: Vec<Asset>,
    pub total_bytes: u64,
    pub directory: PathBuf,
}
fn client() -> Result<Client, String> {
    Client::builder()
        .https_only(true)
        .no_proxy()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(300))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 5
                || attempt.url().scheme() != "https"
                || !attempt.url().username().is_empty()
                || attempt.url().password().is_some()
            {
                attempt.error("invalid model download redirect")
            } else {
                attempt.follow()
            }
        }))
        .user_agent("lma-models/0.1")
        .build()
        .map_err(|e| e.to_string())
}
fn response(client: &Client, url: &str) -> Result<reqwest::blocking::Response, String> {
    client
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|e| format!("model download failed: {}", e.without_url()))
}
fn component(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 128
        && !matches!(text, "." | "..")
        && !text.starts_with('.')
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}
fn repository_parts(repository: &str) -> Result<(&str, &str), String> {
    let (owner, name) = repository
        .split_once('/')
        .ok_or("use a Hugging Face owner/repository reference")?;
    if !component(owner) || !component(name) {
        return Err("invalid Hugging Face repository reference".into());
    }
    Ok((owner, name))
}
fn asset_path(name: &str) -> Result<&Path, String> {
    let path = Path::new(name);
    if name.is_empty()
        || name.len() > 1024
        || name.contains('\\')
        || !path
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
    {
        return Err("invalid relative model asset path".into());
    }
    Ok(path)
}
fn json(client: &Client, url: &str) -> Result<Value, String> {
    let mut bytes = Vec::new();
    response(client, url)?
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err("model metadata exceeds 8 MiB".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}
/// Network metadata only; caller displays the plan before asking for installation.
pub fn inspect(
    repository: &str,
    revision: Option<&str>,
    files: Option<&[String]>,
) -> Result<InstallPlan, String> {
    let (owner, name) = repository_parts(repository)?;
    let revision = revision.unwrap_or("main");
    if !component(revision) {
        return Err("revision must be a branch, tag, or commit without path separators".into());
    }
    let client = client()?;
    let metadata = json(
        &client,
        &format!("https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"),
    )?;
    let revision = metadata["sha"]
        .as_str()
        .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or("catalog omitted immutable model revision")?
        .to_owned();
    let entries = metadata["siblings"]
        .as_array()
        .filter(|e| e.len() <= 100_000)
        .ok_or("invalid model asset catalog")?;
    let mut selected = Vec::new();
    for entry in entries {
        let Some(filename) = entry["rfilename"].as_str() else {
            continue;
        };
        let include = if let Some(files) = files {
            files.iter().any(|f| f == filename)
        } else {
            !filename.contains('/')
                && (filename.ends_with(".json")
                    || filename.ends_with(".safetensors")
                    || filename.ends_with(".model")
                    || matches!(filename, "vocab.txt" | "merges.txt"))
        };
        if !include {
            continue;
        }
        asset_path(filename)?;
        let bytes = entry["size"]
            .as_u64()
            .or_else(|| entry["lfs"]["size"].as_u64())
            .filter(|n| *n > 0 && *n <= 64 * 1024 * 1024 * 1024)
            .ok_or("missing/invalid model asset size")?;
        let sha256 = entry["lfs"]["sha256"].as_str().map(str::to_owned);
        if sha256
            .as_ref()
            .is_some_and(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err("invalid model SHA-256 digest".into());
        }
        if bytes > 32 * 1024 * 1024 && sha256.is_none() {
            return Err("large model assets require an upstream SHA-256 digest".into());
        }
        selected.push(Asset {
            name: filename.into(),
            bytes,
            sha256,
        });
    }
    if selected.is_empty() || selected.len() > 4096 {
        return Err("no selected compatible model assets, or too many assets".into());
    }
    if let Some(files) = files {
        if files.len() != selected.len() {
            return Err("requested files must all exist, without duplicates".into());
        }
    } else if !selected.iter().any(|f| f.name.ends_with(".safetensors")) {
        return Err("repository has no default safetensors assets; choose explicit files or use lma f for GGUF".into());
    }
    let total_bytes = selected
        .iter()
        .try_fold(0u64, |sum, file| sum.checked_add(file.bytes))
        .ok_or("download size overflow")?;
    let directory = crate::models::discovery::default_models_dir()?
        .join("huggingface")
        .join(owner)
        .join(name)
        .join(&revision);
    Ok(InstallPlan {
        repository: repository.into(),
        revision,
        files: selected,
        total_bytes,
        directory,
    })
}
/// Install a previously inspected immutable plan. Destination and descriptors are
/// re-derived from upstream so IPC callers cannot supply arbitrary URLs/paths.
pub fn install(plan: &InstallPlan) -> Result<PathBuf, String> {
    let files: Vec<_> = plan.files.iter().map(|f| f.name.clone()).collect();
    let checked = inspect(&plan.repository, Some(&plan.revision), Some(&files))?;
    if checked.directory != plan.directory
        || checked.total_bytes != plan.total_bytes
        || serde_json::to_value(&checked.files).map_err(|e| e.to_string())?
            != serde_json::to_value(&plan.files).map_err(|e| e.to_string())?
    {
        return Err("model installation plan changed; inspect again".into());
    }
    if fs::symlink_metadata(&checked.directory).is_ok() {
        return Err("model destination already exists; existing assets are not overwritten".into());
    }
    let parent = checked
        .directory
        .parent()
        .ok_or("model destination needs a parent")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let temporary = tempfile::Builder::new()
        .prefix(".model-install-")
        .tempdir_in(parent)
        .map_err(|e| e.to_string())?;
    let client = client()?;
    for asset in &checked.files {
        let relative = asset_path(&asset.name)?;
        let destination = temporary.path().join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .map_err(|e| e.to_string())?;
        let mut input = response(
            &client,
            &format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                checked.repository, checked.revision, asset.name
            ),
        )?;
        let mut hash = Sha256::new();
        let mut length = 0u64;
        let mut buffer = [0u8; 65536];
        loop {
            let read = input
                .read(&mut buffer)
                .map_err(|e| format!("cannot download asset: {e}"))?;
            if read == 0 {
                break;
            }
            length += read as u64;
            if length > asset.bytes {
                return Err("model asset exceeds expected size".into());
            }
            hash.update(&buffer[..read]);
            file.write_all(&buffer[..read]).map_err(|e| e.to_string())?;
        }
        if length != asset.bytes
            || asset
                .sha256
                .as_ref()
                .is_some_and(|expected| format!("{:x}", hash.finalize()) != expected.to_lowercase())
        {
            return Err("incomplete model asset or SHA-256 mismatch".into());
        }
        file.sync_all().map_err(|e| e.to_string())?;
    }
    fs::write(
        temporary.path().join(".lma-source.json"),
        serde_json::to_vec_pretty(&checked).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if checked.directory.exists() {
        return Err("model destination appeared during installation; nothing was replaced".into());
    }
    fs::rename(temporary.path(), &checked.directory).map_err(|e| e.to_string())?;
    fs::canonicalize(&checked.directory).map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn references_and_assets_cannot_escape_storage() {
        for bad in [
            "../repo",
            "owner/../repo",
            "https://example/model",
            "/repo",
            "owner/repo?token=x",
        ] {
            assert!(repository_parts(bad).is_err());
        }
        for bad in ["../x", "/tmp/x", "x/../../y", "x\\y", ""] {
            assert!(asset_path(bad).is_err());
        }
        assert!(repository_parts("owner/model-name").is_ok());
        assert!(asset_path("weights/part.safetensors").is_ok());
    }
}
