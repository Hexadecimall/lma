//! Small, local receipts reconnect an explicitly imported GGUF to its Ollama tag.
//! Receipts are trusted user configuration, not signatures or provenance proofs.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: u32,
    source_root: PathBuf,
    model: String,
    fingerprint: String,
    output: PathBuf,
    output_bytes: u64,
}

pub(super) fn fingerprint(value: &Value) -> String {
    fn canonical(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();
                Value::Object(
                    keys.into_iter()
                        .map(|k| (k.clone(), canonical(&map[k])))
                        .collect(),
                )
            }
            Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
            _ => value.clone(),
        }
    }
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&canonical(value)).expect("JSON value"))
    )
}
fn filename(fingerprint: &str) -> String {
    format!(".lma-import-v1-{fingerprint}.json")
}
fn read(path: &Path) -> Option<Receipt> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > 65536 {
        return None;
    }
    let mut bytes = Vec::new();
    File::open(path)
        .ok()?
        .take(65537)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > 65536 {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}
fn resolve_receipt(path: &Path, root: &Path, name: &str, fingerprint: &str) -> Option<PathBuf> {
    let receipt = read(path)?;
    if receipt.version != 1
        || receipt.source_root != root
        || receipt.model != name
        || receipt.fingerprint != fingerprint
        || !receipt.output.is_absolute()
        || receipt.output.starts_with(root)
    {
        return None;
    }
    let meta = fs::symlink_metadata(&receipt.output).ok()?;
    if !meta.is_file() || meta.len() != receipt.output_bytes {
        return None;
    }
    let mut magic = [0; 4];
    File::open(&receipt.output)
        .ok()?
        .read_exact(&mut magic)
        .ok()?;
    (magic == *b"GGUF").then_some(receipt.output)
}
/// Labels only; a receipt is not a signature or proof of equivalent weights.
pub(super) fn local_names(directory: &Path) -> std::collections::HashMap<PathBuf, String> {
    let mut names = std::collections::HashMap::new();
    let Ok(entries) = fs::read_dir(directory) else {
        return names;
    };
    for entry in entries.take(10_000).flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with(".lma-import-v1-") || !name.ends_with(".json") {
            continue;
        }
        let Some(receipt) = read(&entry.path()) else {
            continue;
        };
        if receipt.model.len() > 512
            || receipt.model.chars().any(char::is_control)
            || filename(&receipt.fingerprint) != name
        {
            continue;
        }
        if let Some(output) = resolve_receipt(
            &entry.path(),
            &receipt.source_root,
            &receipt.model,
            &receipt.fingerprint,
        ) && let Ok(output) = fs::canonicalize(output)
        {
            names.entry(output).or_insert(receipt.model);
        }
    }
    names
}

pub(super) fn find(root: &Path, source: &Value, name: &str) -> Option<PathBuf> {
    let root = fs::canonicalize(root).ok()?;
    let fingerprint = fingerprint(source);
    let mut folders = crate::models::discovery::default_local_dirs();
    // Finder's cwd is not normally the source checkout; this matches the desktop
    // library's existing build-checkout discovery without scanning the whole disk.
    folders.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"));
    for folder in folders {
        if let Some(path) = resolve_receipt(
            &folder.join(filename(&fingerprint)),
            &root,
            name,
            &fingerprint,
        ) {
            return Some(path);
        }
    }
    None
}
pub(super) fn register(
    root: &Path,
    name: &str,
    fingerprint: &str,
    output: &Path,
    size: u64,
) -> Result<(), String> {
    let output = fs::canonicalize(output).map_err(|e| e.to_string())?;
    let metadata = fs::symlink_metadata(&output).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() != size || output.starts_with(root) {
        return Err(
            "only a complete imported GGUF outside the source store can be registered".into(),
        );
    }
    let mut magic = [0; 4];
    File::open(&output)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map_err(|e| e.to_string())?;
    if magic != *b"GGUF" {
        return Err(
            "only a complete imported GGUF outside the source store can be registered".into(),
        );
    }
    let parent = output.parent().ok_or("import output has no parent")?;
    let target = parent.join(filename(fingerprint));
    let receipt = Receipt {
        version: 1,
        source_root: root.to_path_buf(),
        model: name.into(),
        fingerprint: fingerprint.into(),
        output: output.clone(),
        output_bytes: size,
    };
    let replace = match fs::symlink_metadata(&target) {
        Ok(_) => {
            let owned = read(&target).is_some_and(|r| {
                r.version == 1
                    && r.source_root == root
                    && r.model == name
                    && r.fingerprint == fingerprint
            });
            if !owned {
                return Err("an unrecognized import receipt was not overwritten".into());
            }
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => return Err(e.to_string()),
    };
    let mut temp = tempfile::Builder::new()
        .prefix(".lma-import-receipt-")
        .tempfile_in(parent)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut temp, &receipt).map_err(|e| e.to_string())?;
    temp.write_all(b"\n").map_err(|e| e.to_string())?;
    temp.as_file_mut().sync_all().map_err(|e| e.to_string())?;
    if replace {
        temp.persist(target).map_err(|e| e.to_string())?;
    } else {
        temp.persist_noclobber(target).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn receipt_reuse_requires_exact_source_tag_fingerprint_size_and_magic() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("store");
        fs::create_dir(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let output = temp.path().join("model.gguf");
        fs::write(&output, b"GGUFfixture").unwrap();
        let fingerprint = fingerprint(&serde_json::json!({"layers":[1]}));
        register(&root, "gemma4:12b", &fingerprint, &output, 11).unwrap();
        let target = temp.path().join(filename(&fingerprint));
        assert_eq!(
            resolve_receipt(&target, &root, "gemma4:12b", &fingerprint),
            Some(fs::canonicalize(&output).unwrap())
        );
        assert!(resolve_receipt(&target, &root, "other:latest", &fingerprint).is_none());
        assert!(resolve_receipt(&target, &root, "gemma4:12b", "changed").is_none());
        fs::write(&output, b"bad").unwrap();
        assert!(resolve_receipt(&target, &root, "gemma4:12b", &fingerprint).is_none());
        assert!(register(&root, "gemma4:12b", &fingerprint, &output, 3).is_err());
        let replacement = temp.path().join("replacement.gguf");
        fs::write(&replacement, b"GGUFsecond").unwrap();
        register(&root, "gemma4:12b", &fingerprint, &replacement, 10).unwrap();
        assert_eq!(
            resolve_receipt(&target, &root, "gemma4:12b", &fingerprint),
            Some(fs::canonicalize(replacement).unwrap())
        );
    }
    #[test]
    fn manifest_fingerprint_is_stable_but_detects_changes() {
        let a: Value = serde_json::from_str("{\"a\":1,\"nested\":{\"y\":3,\"x\":2}}").unwrap();
        let b: Value = serde_json::from_str("{\"nested\":{\"x\":2,\"y\":3},\"a\":1}").unwrap();
        assert_eq!(fingerprint(&a), fingerprint(&b));
        assert_ne!(fingerprint(&a), fingerprint(&serde_json::json!({"a":2})));
    }
}
