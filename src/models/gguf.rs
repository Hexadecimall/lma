//! Bounded, model-free GGUF metadata and split-set inspection.
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

const HEADER_LIMIT: u64 = 128 * 1024 * 1024;
const GROUP_HEADER_LIMIT: u64 = 256 * 1024 * 1024;
const MAX_PARTS: u32 = 256;

#[derive(Clone, Debug, Serialize)]
pub struct Metadata {
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub tensor_count: u64,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub expert_count: Option<u64>,
    pub experts_used: Option<u64>,
    pub file_type: Option<u64>,
    pub split_index: Option<u64>,
    pub split_count: Option<u64>,
    pub split_tensor_count: Option<u64>,
    #[serde(skip)]
    header_bytes: u64,
}
impl Metadata {
    pub fn is_moe(&self) -> bool {
        self.expert_count.is_some_and(|n| n > 0)
    }

    /// Selected on-disk values, not estimates of the active runtime or VRAM usage.
    pub fn model_info(&self) -> BTreeMap<String, Value> {
        let mut info = BTreeMap::new();
        if let Some(architecture) = &self.architecture {
            info.insert(
                "general.architecture".into(),
                Value::from(architecture.clone()),
            );
            for (suffix, value) in [
                ("context_length", self.context_length),
                ("embedding_length", self.embedding_length),
                ("expert_count", self.expert_count),
                ("expert_used_count", self.experts_used),
            ] {
                if let Some(value) = value {
                    info.insert(format!("{architecture}.{suffix}"), Value::from(value));
                }
            }
        }
        if let Some(name) = &self.name {
            info.insert("general.name".into(), Value::from(name.clone()));
        }
        for (key, value) in [
            ("general.file_type", self.file_type),
            ("split.no", self.split_index),
            ("split.count", self.split_count),
            ("split.tensors.count", self.split_tensor_count),
        ] {
            if let Some(value) = value {
                info.insert(key.into(), Value::from(value));
            }
        }
        info.insert(
            "lma.header_tensor_count".into(),
            Value::from(self.tensor_count),
        );
        info.insert("lma.moe".into(), Value::from(self.is_moe()));
        info
    }
}

struct Reader {
    file: BufReader<File>,
    position: u64,
    limit: u64,
}
impl Reader {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], String> {
        self.advance(N as u64)?;
        let mut data = [0; N];
        self.file
            .read_exact(&mut data)
            .map_err(|e| format!("truncated GGUF header: {e}"))?;
        Ok(data)
    }
    fn advance(&mut self, count: u64) -> Result<(), String> {
        self.position = self
            .position
            .checked_add(count)
            .filter(|p| *p <= self.limit)
            .ok_or("GGUF header exceeds its file/128 MiB limit")?;
        Ok(())
    }
    fn skip(&mut self, count: u64) -> Result<(), String> {
        self.advance(count)?;
        self.file
            .seek_relative(count as i64)
            .map_err(|e| e.to_string())
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self, keep: bool, max: u64) -> Result<Option<String>, String> {
        let length = self.u64()?;
        if length > max {
            return Err("GGUF string exceeds its byte limit".into());
        }
        if keep {
            self.advance(length)?;
            let mut bytes = vec![0; length as usize];
            self.file
                .read_exact(&mut bytes)
                .map_err(|e| e.to_string())?;
            Ok(Some(
                String::from_utf8(bytes).map_err(|_| "invalid UTF-8 in GGUF metadata")?,
            ))
        } else {
            self.skip(length)?;
            Ok(None)
        }
    }
    fn value(&mut self, kind: u32, keep: bool) -> Result<Option<Value>, String> {
        if kind == 8 {
            return Ok(self.string(keep, 64 * 1024 * 1024)?.map(Value::String));
        }
        if kind == 9 {
            let element = self.u32()?;
            let count = self.u64()?;
            if count > 2_000_000 || element == 9 {
                return Err("oversized or nested GGUF metadata array".into());
            }
            if element == 8 {
                for _ in 0..count {
                    self.string(false, 64 * 1024 * 1024)?;
                }
            } else {
                self.skip(
                    count
                        .checked_mul(width(element)?)
                        .ok_or("GGUF array size overflow")?,
                )?;
            }
            return Ok(None);
        }
        if !keep {
            self.skip(width(kind)?)?;
            return Ok(None);
        }
        let value = match kind {
            0 => Value::from(self.bytes::<1>()?[0]),
            1 => Value::from(i8::from_le_bytes(self.bytes()?)),
            2 => Value::from(u16::from_le_bytes(self.bytes()?)),
            3 => Value::from(i16::from_le_bytes(self.bytes()?)),
            4 => Value::from(self.u32()?),
            5 => Value::from(i32::from_le_bytes(self.bytes()?)),
            6 => Value::from(f32::from_le_bytes(self.bytes()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err("invalid GGUF boolean".into()),
            },
            10 => Value::from(self.u64()?),
            11 => Value::from(i64::from_le_bytes(self.bytes()?)),
            12 => Value::from(f64::from_le_bytes(self.bytes()?)),
            _ => return Err("unknown GGUF metadata type".into()),
        };
        Ok(Some(value))
    }
}
fn width(kind: u32) -> Result<u64, String> {
    match kind {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4..=6 => Ok(4),
        10..=12 => Ok(8),
        _ => Err("unknown GGUF metadata type".into()),
    }
}
fn selected(key: &str) -> bool {
    matches!(
        key,
        "general.architecture"
            | "general.name"
            | "general.file_type"
            | "general.alignment"
            | "split.no"
            | "split.count"
            | "split.tensors.count"
    ) || [
        ".context_length",
        ".embedding_length",
        ".expert_count",
        ".expert_used_count",
    ]
    .iter()
    .any(|suffix| key.ends_with(suffix))
}

pub fn inspect(path: &Path) -> Result<Metadata, String> {
    let file = crate::local_file::open_regular(path)?;
    let length = file.metadata().map_err(|e| e.to_string())?.len();
    let mut r = Reader {
        file: BufReader::new(file),
        position: 0,
        limit: length.min(HEADER_LIMIT),
    };
    if r.bytes::<4>()? != *b"GGUF" || !matches!(r.u32()?, 2 | 3) {
        return Err("expected a little-endian GGUF v2/v3 file".into());
    }
    let tensors = r.u64()?;
    let entries = r.u64()?;
    if tensors > 100_000 || entries > 16384 {
        return Err("GGUF tensor/metadata count exceeds inspection limits".into());
    }
    let mut values = BTreeMap::new();
    let mut keys = BTreeSet::new();
    for _ in 0..entries {
        let key = r.string(true, 256)?.unwrap();
        if !keys.insert(key.clone()) {
            return Err("duplicate GGUF metadata key".into());
        }
        let kind = r.u32()?;
        if let Some(value) = r.value(kind, selected(&key))? {
            values.insert(key, value);
        }
    }
    let mut names = BTreeSet::new();
    let mut offsets = Vec::new();
    for _ in 0..tensors {
        let name = r.string(true, 127)?.unwrap();
        if name.is_empty() || !names.insert(name) {
            return Err("invalid/duplicate GGUF tensor name".into());
        }
        let dims = r.u32()?;
        if !(1..=4).contains(&dims) {
            return Err("invalid GGUF tensor rank".into());
        }
        let mut elements = 1u64;
        for _ in 0..dims {
            let n = r.u64()?;
            elements = elements
                .checked_mul(n)
                .filter(|n| *n > 0)
                .ok_or("invalid/overflowing GGUF tensor dimensions")?;
        }
        r.u32()?;
        offsets.push(r.u64()?);
    }
    let alignment = values
        .get("general.alignment")
        .map(|v| v.as_u64().ok_or("invalid GGUF alignment"))
        .transpose()?
        .unwrap_or(32);
    if alignment == 0 || alignment > 4096 || !alignment.is_power_of_two() {
        return Err("invalid GGUF alignment".into());
    }
    let data_start = r
        .position
        .checked_add(alignment - 1)
        .ok_or("GGUF alignment overflow")?
        / alignment
        * alignment;
    for offset in offsets {
        if offset % alignment != 0 || data_start.checked_add(offset).is_none_or(|n| n >= length) {
            return Err("GGUF tensor offset is outside its file or unaligned".into());
        }
    }
    let text = |key| -> Result<Option<String>, String> {
        values
            .get(key)
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("invalid GGUF string {key}"))
            })
            .transpose()
    };
    let number = |key: &str| -> Result<Option<u64>, String> {
        values
            .get(key)
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| format!("invalid GGUF integer {key}"))
            })
            .transpose()
    };
    let architecture = text("general.architecture")?;
    let prefix = architecture.as_deref().unwrap_or("");
    Ok(Metadata {
        name: text("general.name")?,
        context_length: number(&format!("{prefix}.context_length"))?,
        embedding_length: number(&format!("{prefix}.embedding_length"))?,
        expert_count: number(&format!("{prefix}.expert_count"))?,
        experts_used: number(&format!("{prefix}.expert_used_count"))?,
        file_type: number("general.file_type")?,
        split_index: number("split.no")?,
        split_count: number("split.count")?,
        split_tensor_count: number("split.tensors.count")?,
        architecture,
        tensor_count: tensors,
        header_bytes: r.position,
    })
}

#[derive(Debug)]
pub struct ShardSet {
    pub first: PathBuf,
    pub size_bytes: u64,
    pub parts: u32,
}

pub fn shard_name(name: &str) -> Option<(&str, u32, u32)> {
    let (prefix, total) = name.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (prefix, index) = prefix.rsplit_once('-')?;
    if prefix.is_empty()
        || total.len() != 5
        || index.len() != 5
        || !total
            .bytes()
            .chain(index.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let total: u32 = total.parse().ok()?;
    let index: u32 = index.parse().ok()?;
    (total > 1 && index > 0 && index <= total).then_some((prefix, index, total))
}

/// Normalize any selected part to the first shard, validating the complete set
/// before a desktop host unloads its current working model. No files are copied.
pub fn shards(path: &Path) -> Result<Option<ShardSet>, String> {
    let Some((prefix, _, total)) = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(shard_name)
    else {
        return Ok(None);
    };
    if total > MAX_PARTS {
        return Err(format!("GGUF split count exceeds {MAX_PARTS} parts"));
    }
    let parent = path.parent().unwrap_or(Path::new("."));
    let first = parent.join(format!("{prefix}-00001-of-{total:05}.gguf"));
    let mut first_info: Option<Metadata> = None;
    let mut size = 0u64;
    let mut tensors = 0u64;
    let mut headers = 0u64;
    for index in 1..=total {
        let part = parent.join(format!("{prefix}-{index:05}-of-{total:05}.gguf"));
        let meta = fs::metadata(&part)
            .map_err(|e| format!("incomplete split GGUF: {}: {e}", part.display()))?;
        if !meta.is_file() {
            return Err(format!(
                "GGUF shard is not a regular file: {}",
                part.display()
            ));
        }
        let info =
            inspect(&part).map_err(|e| format!("invalid GGUF shard {}: {e}", part.display()))?;
        if info.split_index != Some(u64::from(index - 1))
            || info
                .split_count
                .is_some_and(|count| count != u64::from(total))
        {
            return Err(format!(
                "GGUF split metadata disagrees with filename: {}",
                part.display()
            ));
        }
        if index == 1 && info.split_count != Some(u64::from(total)) {
            return Err("first GGUF shard is missing a matching split.count".into());
        }
        if let Some(first) = &first_info
            && (info.architecture.is_some() && info.architecture != first.architecture
                || info.split_tensor_count.is_some()
                    && info.split_tensor_count != first.split_tensor_count)
        {
            return Err("GGUF shards describe different models or tensor counts".into());
        }
        tensors = tensors
            .checked_add(info.tensor_count)
            .ok_or("split tensor count overflow")?;
        size = size
            .checked_add(meta.len())
            .ok_or("split model size overflow")?;
        headers = headers
            .checked_add(info.header_bytes)
            .filter(|n| *n <= GROUP_HEADER_LIMIT)
            .ok_or("GGUF split-set header budget exceeds 256 MiB")?;
        if first_info.is_none() {
            first_info = Some(info);
        }
    }
    if first_info
        .as_ref()
        .and_then(|i| i.split_tensor_count)
        .is_some_and(|n| n != tensors)
    {
        return Err("GGUF split set is incomplete: total tensor count mismatch".into());
    }
    Ok(Some(ShardSet {
        first,
        size_bytes: size,
        parts: total,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{DiscoveryOptions, discover, resolve};

    fn string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend((text.len() as u64).to_le_bytes());
        bytes.extend(text.as_bytes());
    }
    fn fixture(path: &Path, index: u32, count: u32, moe: bool) {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend((if moe { 8u64 } else { 6u64 }).to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend(8u32.to_le_bytes());
        string(&mut bytes, "llama");
        for (key, value) in [
            ("split.no", index as u64),
            ("split.count", count as u64),
            ("split.tensors.count", count as u64),
            ("llama.context_length", 8192),
            ("llama.embedding_length", 4096),
        ] {
            string(&mut bytes, key);
            bytes.extend(10u32.to_le_bytes());
            bytes.extend(value.to_le_bytes());
        }
        if moe {
            for (key, value) in [("llama.expert_count", 8u64), ("llama.expert_used_count", 2)] {
                string(&mut bytes, key);
                bytes.extend(10u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
        }
        string(&mut bytes, &format!("blk.{index}.weight"));
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(8u64.to_le_bytes());
        bytes.extend(0u32.to_le_bytes()); // F32
        bytes.extend(0u64.to_le_bytes()); // data offset
        bytes.resize(bytes.len().div_ceil(32) * 32 + 32, 0);
        fs::write(path, bytes).unwrap();
    }
    fn parts(dir: &Path, moe: bool) -> (PathBuf, PathBuf) {
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        fixture(&first, 0, 2, moe);
        fixture(&second, 1, 2, moe);
        (first, second)
    }

    #[test]
    fn dense_and_moe_sets_normalize_without_copies_and_discover_once() {
        for moe in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let (first, second) = parts(dir.path(), moe);
            let meta = inspect(&first).unwrap();
            assert_eq!(meta.is_moe(), moe);
            assert_eq!(meta.context_length, Some(8192));
            let set = shards(&second).unwrap().unwrap();
            assert_eq!(set.first, first);
            assert_eq!(set.parts, 2);
            assert_eq!(
                set.size_bytes,
                fs::metadata(&first).unwrap().len() + fs::metadata(&second).unwrap().len()
            );
            assert_eq!(resolve(&second, None).unwrap().model, first);
            let catalog = discover(DiscoveryOptions {
                local_dirs: vec![dir.path().into()],
                ollama_dir: Some(dir.path().join("no-store")),
            });
            assert_eq!(catalog.models.len(), 1, "{:?}", catalog.warnings);
            assert_eq!(catalog.models[0].size_bytes, set.size_bytes);
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
        }
    }

    #[test]
    fn incomplete_or_mismatched_sets_fail_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        let (first, second) = parts(dir.path(), true);
        fixture(&second, 0, 2, true);
        assert!(shards(&first).unwrap_err().contains("disagrees"));
        fixture(&second, 1, 3, true);
        assert!(shards(&first).is_err());
        fs::remove_file(&second).unwrap();
        assert!(
            resolve(&first, None)
                .unwrap_err()
                .contains("incomplete split")
        );
        let catalog = discover(DiscoveryOptions {
            local_dirs: vec![dir.path().into()],
            ollama_dir: Some(dir.path().join("none")),
        });
        assert!(catalog.models.is_empty());
        assert!(
            catalog
                .warnings
                .iter()
                .any(|w| w.contains("incomplete split"))
        );
    }

    #[test]
    fn inspection_accepts_v2_and_rejects_truncation_offsets_and_count_bombs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        fixture(&path, 0, 2, false);
        let mut bytes = fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        assert!(inspect(&path).is_ok());
        let valid = bytes.clone();
        bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        assert!(inspect(&path).unwrap_err().contains("count"));
        let mut bytes = valid.clone();
        let name = b"blk.0.weight";
        let start = bytes.windows(name.len()).position(|b| b == name).unwrap();
        let offset = start + name.len() + 4 + 8 + 4;
        bytes[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        assert!(inspect(&path).unwrap_err().contains("offset"));
        for length in [0, 4, 8, 24, valid.len() / 2] {
            fs::write(&path, &valid[..length]).unwrap();
            assert!(inspect(&path).is_err());
        }
    }

    #[test]
    #[ignore = "requires LMA_TEST_GGUF and/or LMA_TEST_OLLAMA_MODEL; read-only header inspection, no weight loading"]
    fn real_installed_headers_report_architecture_and_experts() {
        let mut count = 0;
        for (variable, ollama) in [("LMA_TEST_GGUF", false), ("LMA_TEST_OLLAMA_MODEL", true)] {
            let Ok(input) = std::env::var(variable) else {
                continue;
            };
            let input = if ollama {
                format!("ollama:{input}")
            } else {
                input
            };
            let resolved = resolve(Path::new(&input), None).unwrap();
            let metadata = inspect(&resolved.model).unwrap();
            assert!(
                metadata
                    .architecture
                    .as_ref()
                    .is_some_and(|name| !name.is_empty())
            );
            assert!(metadata.tensor_count > 0);
            assert_eq!(metadata.model_info()["lma.moe"], metadata.is_moe());
            if metadata.is_moe() {
                assert!(
                    metadata
                        .experts_used
                        .is_some_and(|n| n > 0 && Some(n) <= metadata.expert_count)
                );
            }
            println!("{input}: {:?}", metadata.model_info());
            count += 1;
        }
        assert!(count > 0, "set LMA_TEST_GGUF or LMA_TEST_OLLAMA_MODEL");
    }

    #[test]
    fn split_names_are_strict_and_counts_bounded() {
        assert_eq!(
            shard_name("model-00002-of-00003.gguf"),
            Some(("model", 2, 3))
        );
        for name in [
            "model.gguf",
            "model-1-of-2.gguf",
            "model-00000-of-00002.gguf",
            "model-00003-of-00002.gguf",
        ] {
            assert!(shard_name(name).is_none());
        }
        assert!(
            shards(Path::new("model-00001-of-99999.gguf"))
                .unwrap_err()
                .contains("256")
        );
    }
}
