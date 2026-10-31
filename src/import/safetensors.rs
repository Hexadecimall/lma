//! Multi-tensor HF shards: bounded headers, exact index coverage, streamed payloads.
use super::tensors::{Source, open_regular};
use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, Visitor},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

const HEADER_LIMIT: u64 = 16 * 1024 * 1024;
const MAX_TENSORS: usize = 16384;
const CHUNK: usize = 1024 * 1024;

pub(super) fn unique_map<'de, D, T>(d: D) -> Result<BTreeMap<String, T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Unique<T>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for Unique<T> {
        type Value = BTreeMap<String, T>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a bounded map without duplicate keys")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some(key) = a.next_key::<String>()? {
                if key.is_empty()
                    || key.len() > 512
                    || out.len() >= MAX_TENSORS
                    || out.contains_key(&key)
                {
                    return Err(de::Error::custom(
                        "invalid, duplicate, or excessive map keys",
                    ));
                }
                out.insert(key, a.next_value()?);
            }
            Ok(out)
        }
    }
    d.deserialize_map(Unique(std::marker::PhantomData))
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub(super) enum Dtype {
    F32,
    F16,
    BF16,
}
impl Dtype {
    pub fn width(self) -> u64 {
        if self == Self::F32 { 4 } else { 2 }
    }
    pub fn kind(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::BF16 => 30,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    dtype: Dtype,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Entry {
    Tensor(Descriptor),
    Metadata(BTreeMap<String, String>),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
    #[serde(deserialize_with = "unique_map")]
    weight_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub(super) struct Tensor {
    pub path: PathBuf,
    source: Source,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    pub size: u64,
    offset: u64,
}
impl Tensor {
    fn open(&self) -> Result<File, String> {
        let file = open_regular(&self.path)?;
        if Source::of_file(&file)? != self.source {
            return Err("HF shard changed during import".into());
        }
        Ok(file)
    }
    pub fn check(&self) -> Result<(), String> {
        self.open().map(|_| ())
    }
    /// `permute` converts HF split-half RoPE rows to ggml adjacent pairs. Qwen2
    /// uses NEOX RoPE in the pinned loader and must NOT receive this transform.
    pub fn write(&self, output: &mut dyn Write, permute: Option<(u32, u32)>) -> Result<(), String> {
        let mut file = self.open()?;
        let vector = self.shape.len() == 1;
        let mut buffer = vec![0u8; CHUNK];
        let mut copy = |file: &mut File, offset: u64, mut size: u64| -> Result<(), String> {
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| e.to_string())?;
            while size != 0 {
                let count = size.min(CHUNK as u64) as usize;
                file.read_exact(&mut buffer[..count])
                    .map_err(|e| e.to_string())?;
                if vector && self.dtype != Dtype::F32 {
                    let mut promoted = Vec::with_capacity(count * 2);
                    for b in buffer[..count].chunks_exact(2) {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        let value = if self.dtype == Dtype::BF16 {
                            f32::from_bits(u32::from(bits) << 16)
                        } else {
                            unsafe { llama_cpp_sys_2::ggml_fp16_to_fp32(bits) }
                        };
                        if !value.is_finite() {
                            return Err("nonfinite HF vector".into());
                        }
                        promoted.extend_from_slice(&value.to_le_bytes());
                    }
                    output.write_all(&promoted).map_err(|e| e.to_string())?;
                } else {
                    output
                        .write_all(&buffer[..count])
                        .map_err(|e| e.to_string())?;
                }
                size -= count as u64;
            }
            Ok(())
        };
        if let Some((heads, dim)) = permute {
            if self.shape.len() != 2
                || self.shape[0] != u64::from(heads) * u64::from(dim)
                || dim % 2 != 0
            {
                return Err("invalid RoPE permutation shape".into());
            }
            let row_bytes = self.shape[1] * self.dtype.width();
            for head in 0..u64::from(heads) {
                for row in 0..u64::from(dim) {
                    let source_row =
                        head * u64::from(dim) + (row % 2) * u64::from(dim / 2) + row / 2;
                    copy(&mut file, self.offset + source_row * row_bytes, row_bytes)?;
                }
            }
        } else {
            copy(&mut file, self.offset, self.size)?;
        }
        if Source::of_file(&file)? != self.source {
            return Err("HF shard changed while streaming".into());
        }
        Ok(())
    }
    pub fn output_info(&self) -> (u32, u64) {
        if self.shape.len() == 1 {
            (0, self.size / self.dtype.width() * 4)
        } else {
            (self.dtype.kind(), self.size)
        }
    }
}

fn shard(path: &Path) -> Result<BTreeMap<String, Tensor>, String> {
    let mut file = open_regular(path)?;
    let source = Source::of_file(&file)?;
    let length = file.metadata().map_err(|e| e.to_string())?.len();
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix).map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0 || header_len > HEADER_LIMIT || header_len + 8 > length {
        return Err("invalid or oversized HF safetensors header (limit 16 MiB)".into());
    }
    let start = header_len + 8;
    let mut bytes = vec![0; header_len as usize];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let mut de = serde_json::Deserializer::from_slice(&bytes);
    let entries: BTreeMap<String, Entry> =
        unique_map(&mut de).map_err(|e| format!("invalid HF safetensors header: {e}"))?;
    de.end().map_err(|e| e.to_string())?;
    let mut ranges = Vec::new();
    let mut out = BTreeMap::new();
    for (name, entry) in entries {
        let d = match entry {
            Entry::Metadata(meta) if name == "__metadata__" => {
                if meta.get("format").is_some_and(|v| v != "pt") {
                    return Err("HF shard must use PyTorch tensor layout".into());
                }
                continue;
            }
            Entry::Tensor(d) if name != "__metadata__" => d,
            _ => return Err("invalid safetensors metadata entry".into()),
        };
        if d.shape.is_empty() || d.shape.len() > 3 || d.shape.contains(&0) {
            return Err(format!("unsupported HF tensor shape for {name}"));
        }
        let size = d
            .shape
            .iter()
            .try_fold(d.dtype.width(), |n, dim| n.checked_mul(*dim))
            .ok_or("HF tensor size overflow")?;
        let [begin, end] = d.data_offsets;
        if end.checked_sub(begin) != Some(size) || end > length - start {
            return Err(format!("invalid HF tensor range for {name}"));
        }
        ranges.push((begin, end));
        out.insert(
            name,
            Tensor {
                path: path.into(),
                source: source.clone(),
                shape: d.shape,
                dtype: d.dtype,
                size,
                offset: start + begin,
            },
        );
    }
    ranges.sort_unstable();
    let mut end = 0;
    for (a, b) in ranges {
        if a != end {
            return Err("HF shard contains gaps or overlapping ranges".into());
        }
        end = b;
    }
    if out.is_empty() || end != length - start {
        return Err("HF shard has empty or unindexed trailing data".into());
    }
    if Source::of_file(&file)? != source {
        return Err("HF shard changed while inspecting".into());
    }
    Ok(out)
}

pub(super) fn load(root: &Path) -> Result<BTreeMap<String, Tensor>, String> {
    let index_path = root.join("model.safetensors.index.json");
    if index_path.try_exists().map_err(|e| e.to_string())? {
        // Parse original bytes: Value would hide duplicate weight-map keys.
        let mut file = open_regular(&index_path)?;
        let before = Source::of_file(&file)?;
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take(HEADER_LIMIT + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() as u64 > HEADER_LIMIT {
            return Err("oversized HF index".into());
        }
        if Source::of_file(&file)? != before {
            return Err("HF index changed while reading".into());
        }
        let index: Index =
            serde_json::from_slice(&bytes).map_err(|e| format!("invalid HF shard index: {e}"))?;

        if index.weight_map.is_empty() {
            return Err("empty HF shard index".into());
        }
        let names: BTreeSet<_> = index.weight_map.values().cloned().collect();
        if names.len() > 1024 {
            return Err("HF shard index exceeds 1024 shards".into());
        }
        let mut tensors = BTreeMap::new();
        for name in names {
            if name.len() > 255
                || !name.ends_with(".safetensors")
                || name.contains(['/', '\\', ':'])
                || matches!(name.as_str(), "." | "..")
            {
                return Err(
                    "HF shard index must reference simple local .safetensors filenames".into(),
                );
            }
            for (key, tensor) in shard(&root.join(&name))? {
                if index.weight_map.get(&key) != Some(&name)
                    || tensors.insert(key, tensor).is_some()
                {
                    return Err("HF shard/index mismatch or duplicate tensor".into());
                }
            }
        }
        if tensors.len() != index.weight_map.len() {
            return Err("HF index references missing tensors".into());
        }
        if let Some(total) = index.metadata.get("total_size") {
            let actual = tensors
                .values()
                .try_fold(0u64, |n, t| n.checked_add(t.size))
                .ok_or("HF total size overflow")?;
            if total.as_u64() != Some(actual) {
                return Err("HF index total_size mismatch".into());
            }
        }
        Ok(tensors)
    } else {
        shard(&root.join("model.safetensors"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    pub(super) fn fixture(path: &Path, entries: Vec<(String, Vec<u64>, Vec<f32>)>) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in entries {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name,
                json!({"dtype":"F32", "shape":shape, "data_offsets":[start, data.len()]}),
            );
        }
        let mut bytes = serde_json::to_vec(&header).unwrap();
        while bytes.len() % 8 != 0 {
            bytes.push(b' ');
        }
        let mut file = File::create(path).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.write_all(&data).unwrap();
    }
    #[test]
    fn permutation_is_per_head_and_streams_exact_rows() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        fixture(
            &path,
            vec![("q".into(), vec![8, 2], (0..16).map(|n| n as f32).collect())],
        );
        let tensor = shard(&path).unwrap().remove("q").unwrap();
        let mut out = Vec::new();
        tensor.write(&mut out, Some((2, 4))).unwrap();
        let floats: Vec<_> = out
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            floats,
            vec![
                0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.
            ]
        );
        let mut out = Vec::new();
        tensor.write(&mut out, None).unwrap();
        assert_eq!(out.len(), 64);
        std::fs::write(&path, b"changed").unwrap();
        assert!(tensor.check().is_err());
    }
    #[test]
    fn index_coverage_traversal_duplicates_and_bounds() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fixture(
            &root.join("one.safetensors"),
            vec![("a".into(), vec![1], vec![1.])],
        );
        fixture(
            &root.join("two.safetensors"),
            vec![("b".into(), vec![1], vec![2.])],
        );
        let path = root.join("model.safetensors.index.json");
        std::fs::write(&path, r#"{"metadata":{"total_size":8},"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}"#).unwrap();
        assert_eq!(load(root).unwrap().len(), 2);
        for bad in [
            r#"{"weight_map":{"a":"../one.safetensors"}}"#,
            r#"{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}"#,
            r#"{"weight_map":{"missing":"one.safetensors"}}"#,
        ] {
            std::fs::write(&path, bad).unwrap();
            assert!(load(root).is_err());
        }
        let bad = root.join("bad.safetensors");
        std::fs::write(&bad, (HEADER_LIMIT + 1).to_le_bytes()).unwrap();
        assert!(shard(&bad).is_err());
    }
    fn raw(path: &Path, header: &str, data: &[u8]) {
        let mut f = File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(data).unwrap();
    }
    #[test]
    fn sixteen_bit_matrices_stay_exact_and_vectors_promote_to_f32() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        for (dtype, words) in [("F16", [0x3e00u16, 0xc000]), ("BF16", [0x3fc0u16, 0xc000])] {
            let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
            let header = format!(
                r#"{{"matrix":{{"dtype":"{dtype}","shape":[1,2],"data_offsets":[0,4]}},"norm":{{"dtype":"{dtype}","shape":[2],"data_offsets":[4,8]}}}}"#
            );
            raw(&path, &header, &[bytes.clone(), bytes.clone()].concat());
            let tensors = shard(&path).unwrap();
            let mut out = Vec::new();
            tensors["matrix"].write(&mut out, None).unwrap();
            assert_eq!(out, bytes);
            assert_eq!(
                tensors["matrix"].output_info(),
                (if dtype == "F16" { 1 } else { 30 }, 4)
            );
            let mut out = Vec::new();
            tensors["norm"].write(&mut out, None).unwrap();
            assert_eq!(out, [1.5f32.to_le_bytes(), (-2f32).to_le_bytes()].concat());
            assert_eq!(tensors["norm"].output_info(), (0, 8));
        }
    }
    #[test]
    fn malformed_shard_descriptors_duplicate_names_and_ranges_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        for header in [
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"I8","shape":[4],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"__metadata__":{"format":"np"}}"#,
        ] {
            raw(&path, header, &[0; 4]);
            assert!(shard(&path).is_err(), "accepted {header}");
        }
        raw(
            &path,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 8],
        );
        assert!(shard(&path).is_err());
    }
    #[test]
    fn sparse_multigigabyte_shard_inspection_reads_only_the_header() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        let size = 8u64 * 1024 * 1024 * 1024;
        let header = format!(
            r#"{{"embedding":{{"dtype":"F16","shape":[1048576,4096],"data_offsets":[0,{size}]}}}}"#
        );
        raw(&path, &header, &[]);
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(8 + header.len() as u64 + size).unwrap();
        let tensors = shard(&path).unwrap();
        assert_eq!(tensors["embedding"].size, size);
        assert_eq!(tensors.len(), 1);
    }
    #[cfg(unix)]
    #[test]
    fn local_shard_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let original = temp.path().join("original.safetensors");
        fixture(&original, vec![("a".into(), vec![1], vec![0.])]);
        let link = temp.path().join("model.safetensors");
        symlink(&original, &link).unwrap();
        assert!(load(temp.path()).is_err());
    }
}
