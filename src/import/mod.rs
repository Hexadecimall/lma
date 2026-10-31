//! Explicit, local-only imports. Discovery never starts a conversion.
mod aliases;
mod gemma;
mod gguf;
mod hf;
mod hf_tokenizer;
mod safetensors;
mod tensors;

use crate::config::ImportConfig;
use gguf::{Layout, TensorInfo};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
};

pub(crate) fn local_names(directory: &Path) -> std::collections::HashMap<PathBuf, String> {
    aliases::local_names(directory)
}

pub(crate) fn cached_model(root: &Path, source: &Value, name: &str) -> Option<PathBuf> {
    aliases::find(root, source, name)
}

pub(crate) fn gemma4_chat_template() -> &'static str {
    gemma::CHAT_TEMPLATE
}

struct SourceTensor {
    path: PathBuf,
    source_name: String,
    tensor: tensors::Tensor,
    embedding: bool,
}
enum Payload {
    Source(Box<SourceTensor>),
    Bytes(Vec<u8>),
    Hf(Vec<hf::Piece>),
}
struct Prepared {
    layout: Layout,
    payloads: Vec<Payload>,
    root: PathBuf,
    name: String,
    skipped: usize,
    fingerprint: Option<String>,
    notes: &'static str,
}

#[derive(Serialize)]
pub struct Preview {
    pub model: String,
    pub output: PathBuf,
    pub output_bytes: u64,
    pub tensor_count: usize,
    pub skipped_nontext_tensors: usize,
    pub notes: &'static str,
}
const NOTES: &str = "Text-only dense Gemma4 import; native NVFP4 projection weights and global scales are preserved. NVFP4 token/output embeddings are converted to Q8_0 (lossy). Vision/audio and speculative draft tensors are not imported. Source store is unchanged; no network or external tools.";

fn read_json(path: &Path, limit: u64) -> Result<Value, String> {
    let before = fs::symlink_metadata(path)
        .map_err(|e| format!("cannot inspect {}: {e}", path.display()))?;
    if !before.is_file() || before.len() > limit {
        return Err(format!(
            "{} must be a regular non-symlink JSON file of at most {limit} bytes",
            path.display()
        ));
    }
    let mut file = tensors::open_regular(path)?;
    let snapshot = tensors::Source::of_file(&file)?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() != before.len() {
        return Err("JSON source changed while opening".into());
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("JSON source exceeds its byte limit".into());
    }
    if tensors::Source::of_file(&file)? != snapshot {
        return Err("JSON source changed while reading".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON {}: {e}", path.display()))
}
fn blob(root: &Path, layer: &Value) -> Result<PathBuf, String> {
    let digest = layer["digest"]
        .as_str()
        .and_then(|s| s.strip_prefix("sha256:"))
        .ok_or("expected SHA-256 layer digest")?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("invalid layer digest".into());
    }
    let path = root.join("blobs").join(format!("sha256-{digest}"));
    let meta = fs::symlink_metadata(&path)
        .map_err(|e| format!("missing import asset {}: {e}", path.display()))?;
    if !meta.is_file() || layer["size"].as_u64() != Some(meta.len()) {
        return Err(format!(
            "import asset is not a regular file of the declared size: {}",
            path.display()
        ));
    }
    Ok(path)
}
fn prepare(config: &ImportConfig) -> Result<Prepared, String> {
    let path = Path::new(&config.model);
    if path.is_dir() {
        return hf::prepare(config, path);
    }
    if path.is_absolute() || config.model.starts_with("./") || config.model.starts_with("../") {
        return Err(format!(
            "HF input must be an existing local model directory: {}",
            path.display()
        ));
    }
    let (root, manifest, name) =
        crate::models::import_source(&config.model, config.ollama_dir.as_deref())?;
    let source = read_json(&manifest, 1024 * 1024)?;
    let fingerprint = aliases::fingerprint(&source);
    if source["schemaVersion"] != 2
        || source["format"] != "safetensors"
        || source["runner"] != "mlx"
    {
        return Err("import currently supports Ollama MLX per-tensor safetensors dense Gemma4 models only; ordinary GGUFs can be loaded directly".into());
    }
    let layers = source["layers"]
        .as_array()
        .filter(|l| l.len() <= 16384)
        .ok_or("invalid or oversized manifest layers")?;
    let mut named = BTreeMap::new();
    for layer in layers {
        let media = layer["mediaType"]
            .as_str()
            .ok_or("layer is missing mediaType")?;
        match media {
            "application/vnd.ollama.image.tensor" | "application/vnd.ollama.image.json" => {
                let label = layer["name"]
                    .as_str()
                    .filter(|s| !s.is_empty() && s.len() <= 512 && !s.chars().any(char::is_control))
                    .ok_or("invalid import layer name")?;
                if named.insert(label, layer).is_some() {
                    return Err("duplicate named import layer".into());
                }
            }
            "application/vnd.ollama.image.license"
            | "application/vnd.ollama.image.params"
            | "application/vnd.ollama.image.template"
            | "application/vnd.docker.distribution.manifest.v2+json"
            | "application/vnd.ollama.manifest.list.v2+json" => {}
            _ => return Err(format!("unsupported import layer type {media}")),
        }
    }
    let json = |key: &str, limit| -> Result<Value, String> {
        let layer = named
            .get(key)
            .ok_or_else(|| format!("missing {key} import asset"))?;
        if layer["mediaType"] != "application/vnd.ollama.image.json" {
            return Err(format!("{key} is not a JSON layer"));
        }
        read_json(&blob(&root, layer)?, limit)
    };
    let architecture = json("config.json", 1024 * 1024)?;
    // Architecture checks precede the much larger tokenizer or tensor reads.
    let expected = gemma::expected_tensors(&architecture)?;
    let tokenizer = json("tokenizer.json", 64 * 1024 * 1024)?;
    let tokenizer_config = json("tokenizer_config.json", 1024 * 1024)?;
    let generation = named
        .get("generation_config.json")
        .map(|_| json("generation_config.json", 1024 * 1024))
        .transpose()?;
    let mut metadata = gemma::metadata(
        &architecture,
        &tokenizer,
        &tokenizer_config,
        generation.as_ref(),
    )?;
    metadata.push((
        "general.source.ollama".into(),
        gguf::Value::String(name.clone()),
    ));
    metadata.push((
        "lma.import.manifest_fingerprint".into(),
        gguf::Value::String(fingerprint.clone()),
    ));
    metadata.push((
        "general.description".into(),
        gguf::Value::String(NOTES.into()),
    ));
    metadata.push(("general.quantization_version".into(), gguf::Value::U32(2)));
    let mut specs = Vec::new();
    let mut payloads = Vec::new();
    let mut seen = BTreeSet::new();
    let mut skipped = 0;
    for (label, layer) in named {
        if layer["mediaType"] != "application/vnd.ollama.image.tensor" {
            continue;
        }
        let Some(target) = gemma::tensor_name(label)? else {
            skipped += 1;
            continue;
        };
        let path = blob(&root, layer)?;
        let tensor = tensors::inspect(&path, label)?;
        let mut dimensions = tensor.shape.iter().rev().copied().collect::<Vec<_>>();
        if dimensions.is_empty() {
            dimensions.push(1);
        }
        let optional_scalar = target
            .strip_suffix(".layer_output_scale.weight")
            .is_some_and(|prefix| expected.contains_key(&format!("{prefix}.attn_norm.weight")));
        let shape = if optional_scalar {
            vec![1]
        } else if target == "output.weight" {
            expected["token_embd.weight"].clone()
        } else {
            expected
                .get(&target)
                .ok_or_else(|| format!("tensor {target} is not used by this Gemma4 configuration"))?
                .clone()
        };
        if dimensions != shape {
            return Err(format!(
                "shape mismatch for {target}: expected {shape:?}, found {dimensions:?}"
            ));
        }
        if !seen.insert(target.clone()) {
            return Err(format!("duplicate mapped tensor {target}"));
        }
        let embedding = target == "token_embd.weight" || target == "output.weight";
        // Native norm/scalar tensors conventionally use F32. Promote 16-bit
        // vectors exactly, without changing Gemma4's multiplicative norm weights.
        let promoted = promote_vector(&tensor, &path)?;
        let (kind, size) = promoted.as_ref().map_or_else(
            || tensor.output_info(embedding),
            |bytes| (llama_cpp_sys_2::GGML_TYPE_F32, bytes.len() as u64),
        );
        specs.push(TensorInfo {
            name: target.clone(),
            dimensions,
            kind,
            size,
            offset: 0,
        });
        if let Some(bytes) = promoted {
            payloads.push(Payload::Bytes(bytes));
        } else if let Some(scale) = tensor.global_scale.filter(|_| !embedding) {
            let scale_name = target
                .strip_suffix(".weight")
                .ok_or("quantized scalar is unsupported")?
                .to_owned()
                + ".scale";
            specs.push(TensorInfo {
                name: scale_name,
                dimensions: vec![1],
                kind: llama_cpp_sys_2::GGML_TYPE_F32,
                size: 4,
                offset: 0,
            });
            payloads.push(Payload::Source(Box::new(SourceTensor {
                path,
                source_name: label.into(),
                tensor,
                embedding,
            })));
            payloads.push(Payload::Bytes(scale.to_le_bytes().to_vec()));
        } else {
            payloads.push(Payload::Source(Box::new(SourceTensor {
                path,
                source_name: label.into(),
                tensor,
                embedding,
            })));
        }
    }
    for (name, values) in gemma::extra_tensors(&architecture)? {
        if !seen.insert(name.clone()) {
            return Err("duplicate generated tensor".into());
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        specs.push(TensorInfo {
            name,
            dimensions: vec![values.len() as u64],
            kind: llama_cpp_sys_2::GGML_TYPE_F32,
            size: bytes.len() as u64,
            offset: 0,
        });
        payloads.push(Payload::Bytes(bytes));
    }
    if let Some(missing) = expected.keys().find(|key| !seen.contains(*key)) {
        return Err(format!("missing required text tensor {missing}"));
    }
    let layout = Layout::new(metadata, specs)?;
    Ok(Prepared {
        layout,
        payloads,
        root,
        name,
        skipped,
        fingerprint: Some(fingerprint),
        notes: NOTES,
    })
}

fn promote_vector(tensor: &tensors::Tensor, path: &Path) -> Result<Option<Vec<u8>>, String> {
    let bf16 = tensor.ggml_type == llama_cpp_sys_2::GGML_TYPE_BF16;
    let f16 = tensor.ggml_type == llama_cpp_sys_2::GGML_TYPE_F16;
    if tensor.shape.len() > 1 || !(bf16 || f16) {
        return Ok(None);
    }
    if tensor.byte_len > 1024 * 1024 {
        return Err("oversized norm/scalar tensor".into());
    }
    let mut source = Vec::new();
    tensor.write(path, &mut source, false)?;
    let mut output = Vec::with_capacity(source.len() * 2);
    for word in source.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes([word[0], word[1]]);
        let value = if bf16 {
            f32::from_bits(u32::from(bits) << 16)
        } else {
            // SAFETY: a scalar half-precision bit pattern; no pointers or state.
            unsafe { llama_cpp_sys_2::ggml_fp16_to_fp32(bits) }
        };
        if !value.is_finite() {
            return Err("nonfinite normalization/scalar value".into());
        }
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Some(output))
}

pub fn preview(config: &ImportConfig) -> Result<Preview, String> {
    let prepared = prepare(config)?;
    Ok(summary(config, &prepared))
}
fn summary(config: &ImportConfig, prepared: &Prepared) -> Preview {
    Preview {
        model: prepared.name.clone(),
        output: config.output.clone(),
        output_bytes: prepared.layout.total_bytes,
        tensor_count: prepared.layout.tensors.len(),
        skipped_nontext_tensors: prepared.skipped,
        notes: prepared.notes,
    }
}

pub fn run(config: ImportConfig) -> Result<(), String> {
    let prepared = prepare(&config)?;
    let parent = config
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent)
        .map_err(|e| format!("output directory must already exist: {e}"))?;
    if parent.starts_with(&prepared.root) {
        return Err(
            "import output must be outside the original Ollama store or HF source directory".into(),
        );
    }
    if !config
        .output
        .extension()
        .is_some_and(|v| v.eq_ignore_ascii_case("gguf"))
    {
        return Err("import output must end in .gguf".into());
    }
    match fs::symlink_metadata(&config.output) {
        Ok(_) => return Err("import never overwrites an existing output".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.to_string()),
    }
    eprintln!(
        "Importing {}: {} tensors, {:.2} GiB new disk space. {}",
        prepared.name,
        prepared.layout.tensors.len(),
        prepared.layout.total_bytes as f64 / 1073741824.0,
        prepared.notes
    );
    let mut output = tempfile::Builder::new()
        .prefix(".lma-import-")
        .suffix(".part")
        .tempfile_in(&parent)
        .map_err(|e| e.to_string())?;
    prepared.layout.write_header(output.as_file_mut())?;
    for (index, (info, payload)) in prepared
        .layout
        .tensors
        .iter()
        .zip(&prepared.payloads)
        .enumerate()
    {
        let position = output
            .as_file_mut()
            .stream_position()
            .map_err(|e| e.to_string())?;
        let start = prepared.layout.data_start + info.offset;
        gguf::padding(
            output.as_file_mut(),
            start.checked_sub(position).ok_or("tensor output overlap")?,
        )?;
        match payload {
            Payload::Source(source) => {
                source
                    .tensor
                    .write(&source.path, output.as_file_mut(), source.embedding)?
            }
            Payload::Bytes(bytes) => output.write_all(bytes).map_err(|e| e.to_string())?,
            Payload::Hf(pieces) => {
                for piece in pieces {
                    piece.tensor.write(output.as_file_mut(), piece.permute)?;
                }
            }
        }
        if output
            .as_file_mut()
            .stream_position()
            .map_err(|e| e.to_string())?
            != start + info.size
        {
            return Err(format!("wrong output length for {}", info.name));
        }
        if index % 50 == 0 {
            eprintln!("Import: {}/{} tensors", index + 1, prepared.payloads.len());
        }
    }
    if output
        .as_file()
        .metadata()
        .map_err(|e| e.to_string())?
        .len()
        != prepared.layout.total_bytes
    {
        return Err("incorrect final GGUF size".into());
    }
    // Catch source replacements during conversion before publishing a new model.
    for payload in &prepared.payloads {
        match payload {
            Payload::Source(source)
                if tensors::inspect(&source.path, &source.source_name)? != source.tensor =>
            {
                return Err("source tensor changed during import; output not published".into());
            }
            Payload::Hf(pieces) => {
                for piece in pieces {
                    piece.tensor.check()?;
                }
            }
            _ => {}
        }
    }
    output.as_file_mut().sync_all().map_err(|e| e.to_string())?;
    output
        .persist_noclobber(&config.output)
        .map_err(|e| format!("cannot publish imported model: {e}"))?;
    if let Some(fingerprint) = &prepared.fingerprint
        && let Err(error) = aliases::register(
            &prepared.root,
            &prepared.name,
            fingerprint,
            &config.output,
            prepared.layout.total_bytes,
        )
    {
        eprintln!(
            "WARNING: imported GGUF is ready, but alias registration failed: {error}; use its local path directly"
        );
    }
    println!("Imported {}", config.output.display());
    Ok(())
}
