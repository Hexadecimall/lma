//! Validated common HF causal language models; no remote code or guessed mapping.
use super::{
    Payload, Prepared,
    gguf::{Layout, TensorInfo, Value as G},
    hf_tokenizer, read_json, safetensors,
};
use crate::config::ImportConfig;
use serde_json::Value;
use std::{fs, path::Path};

pub(super) const NOTES: &str = "Local HF text import: validated Llama, non-sliding Mistral/Mixtral, or Qwen2 profiles. F32/F16/BF16 matrix precision is preserved; vectors are promoted to F32. No network, external tools, quantization, or source writes. Unsupported architecture/tokenizer/scaling variants are rejected.";

fn error(s: impl std::fmt::Display) -> String {
    format!("HF import: {s}")
}
fn positive(v: &Value, key: &str, max: u32) -> Result<u32, String> {
    v[key]
        .as_u64()
        .filter(|n| *n > 0 && *n <= u64::from(max))
        .map(|n| n as u32)
        .ok_or_else(|| error(format!("{key} must be in 1..={max}")))
}
fn number(v: &Value, key: &str, default: f64) -> Result<f64, String> {
    let n = match v.get(key) {
        None => default,
        Some(v) => v
            .as_f64()
            .ok_or_else(|| error(format!("{key} must be numeric")))?,
    };
    if !n.is_finite() || !(n as f32).is_finite() || n <= 0. || (n as f32) <= 0. {
        return Err(error(format!("{key} must be a positive finite f32")));
    }
    Ok(n)
}
fn boolean(v: &Value, key: &str, default: bool) -> Result<bool, String> {
    match v.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| error(format!("{key} must be boolean"))),
    }
}
struct Config {
    arch: &'static str,
    hidden: u32,
    ff: u32,
    layers: u32,
    heads: u32,
    kv: u32,
    dim: u32,
    vocab: u32,
    context: u32,
    eps: f32,
    theta: f64,
    tied: bool,
    experts: u32,
    used: u32,
}
impl Config {
    fn parse(v: &Value) -> Result<Self, String> {
        let arch = v["architectures"]
            .as_array()
            .filter(|a| a.len() == 1)
            .and_then(|a| a[0].as_str())
            .ok_or_else(|| error("exactly one supported causal architecture is required"))?;
        let (kind, model_type) = match arch {
            "LlamaForCausalLM" => ("llama", "llama"),
            "MistralForCausalLM" => ("llama", "mistral"),
            "MixtralForCausalLM" => ("llama", "mixtral"),
            "Qwen2ForCausalLM" => ("qwen2", "qwen2"),
            _ => {
                return Err(error(format!(
                    "unsupported architecture {arch}; supported: LlamaForCausalLM, MistralForCausalLM, MixtralForCausalLM, Qwen2ForCausalLM"
                )));
            }
        };
        if v["model_type"] != model_type || v["hidden_act"] != "silu" {
            return Err(error("model_type mismatch or non-SiLU activation"));
        }
        for key in [
            "quantization_config",
            "compression_config",
            "text_config",
            "vision_config",
            "rope_parameters",
            "auto_map",
        ] {
            if v.get(key).is_some_and(|v| !v.is_null()) {
                return Err(error(format!("{key} variants are unsupported")));
            }
        }
        for key in [
            "mlp_bias",
            "qk_layernorm",
            "use_qk_norm",
            "use_sliding_window",
            "parallel_attn",
        ] {
            if boolean(v, key, false)? {
                return Err(error(format!("{key}=true is unsupported")));
            }
        }
        if kind != "qwen2" && boolean(v, "attention_bias", false)? {
            return Err(error("Llama/Mistral attention biases are unsupported"));
        }
        if kind == "qwen2" && !boolean(v, "attention_bias", true)? {
            return Err(error("Qwen2 requires its standard Q/K/V biases"));
        }
        if v.get("sliding_window")
            .is_some_and(|v| !v.is_null() && v.as_u64() != Some(0))
            && model_type != "qwen2"
        {
            // The pinned llama architecture does not consume sliding-window metadata.
            return Err(error(
                "sliding-window Mistral/Mixtral is not faithfully supported by the pinned native llama architecture",
            ));
        }
        for key in [
            "partial_rotary_factor",
            "attention_multiplier",
            "embedding_multiplier",
            "residual_multiplier",
            "logits_scaling",
        ] {
            if let Some(n) = v.get(key).filter(|v| !v.is_null()) {
                if n.as_f64() != Some(1.) {
                    return Err(error(format!("nonstandard {key} is unsupported")));
                }
            }
        }
        for key in ["attn_logit_softcapping", "final_logit_softcapping"] {
            if v.get(key)
                .is_some_and(|v| !v.is_null() && v.as_f64() != Some(0.))
            {
                return Err(error(format!("{key} is unsupported")));
            }
        }
        let hidden = positive(v, "hidden_size", 65536)?;
        let heads = positive(v, "num_attention_heads", 1024)?;
        let kv = if v.get("num_key_value_heads").is_some() {
            positive(v, "num_key_value_heads", heads)?
        } else {
            heads
        };
        let dim = if v.get("head_dim").is_some() {
            positive(v, "head_dim", 4096)?
        } else {
            hidden / heads
        };
        if hidden % heads != 0
            || heads % kv != 0
            || dim == 0
            || dim % 2 != 0
            || (kind == "qwen2" && dim != hidden / heads)
        {
            return Err(error("invalid attention head geometry"));
        }
        let (experts, used) = if model_type == "mixtral" {
            let n = positive(v, "num_local_experts", 256)?;
            (n, positive(v, "num_experts_per_tok", n)?)
        } else {
            for key in [
                "num_local_experts",
                "num_experts",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            ] {
                if v.get(key)
                    .is_some_and(|v| !v.is_null() && v.as_u64() != Some(0))
                {
                    return Err(error("MoE variants other than Mixtral are unsupported"));
                }
            }
            (0, 0)
        };
        let c = Self {
            arch: kind,
            hidden,
            heads,
            kv,
            dim,
            ff: positive(v, "intermediate_size", 262144)?,
            layers: positive(v, "num_hidden_layers", 256)?,
            vocab: positive(v, "vocab_size", 1_048_576)?,
            context: positive(v, "max_position_embeddings", u32::MAX)?,
            eps: number(v, "rms_norm_eps", 1e-6)? as f32,
            theta: number(v, "rope_theta", 10000.)?,
            tied: boolean(v, "tie_word_embeddings", false)?,
            experts,
            used,
        };
        if u64::from(c.layers) * (12 + u64::from(c.experts) * 3) > 16384 {
            return Err(error("configuration exceeds 16384 source tensors"));
        }
        Ok(c)
    }
    fn metadata(&self, v: &Value) -> Result<(Vec<(String, G)>, Option<Vec<f32>>), String> {
        let mut m = vec![
            ("general.architecture".into(), G::String(self.arch.into())),
            ("general.description".into(), G::String(NOTES.into())),
        ];
        for (k, n) in [
            ("context_length", self.context),
            ("embedding_length", self.hidden),
            ("feed_forward_length", self.ff),
            ("block_count", self.layers),
            ("vocab_size", self.vocab),
            ("attention.head_count", self.heads),
            ("attention.head_count_kv", self.kv),
            ("attention.key_length", self.dim),
            ("attention.value_length", self.dim),
            ("rope.dimension_count", self.dim),
        ] {
            m.push((format!("{}.{k}", self.arch), G::U32(n)));
        }
        m.push((
            format!("{}.attention.layer_norm_rms_epsilon", self.arch),
            G::F32(self.eps),
        ));
        m.push((
            format!("{}.rope.freq_base", self.arch),
            G::F32(self.theta as f32),
        ));
        if self.experts != 0 {
            m.push(("llama.expert_count".into(), G::U32(self.experts)));
            m.push(("llama.expert_used_count".into(), G::U32(self.used)));
        }
        let mut factors = None;
        let mut scaling = "none";
        if let Some(r) = v.get("rope_scaling").filter(|v| !v.is_null()) {
            let obj = r
                .as_object()
                .ok_or_else(|| error("rope_scaling must be an object"))?;
            let kind = r
                .get("rope_type")
                .or_else(|| r.get("type"))
                .and_then(Value::as_str)
                .ok_or_else(|| error("missing rope_scaling type"))?;
            if r.get("type").is_some()
                && r.get("rope_type").is_some()
                && r["type"] != r["rope_type"]
            {
                return Err(error("conflicting RoPE scaling types"));
            }
            let allowed: &[&str] = match kind {
                "default" => &["type", "rope_type"],
                "linear" => &["type", "rope_type", "factor"],
                "llama3" if self.arch == "llama" => &[
                    "type",
                    "rope_type",
                    "factor",
                    "low_freq_factor",
                    "high_freq_factor",
                    "original_max_position_embeddings",
                ],
                _ => {
                    return Err(error(format!(
                        "unsupported RoPE scaling {kind}; supported: default, linear, llama3"
                    )));
                }
            };
            if obj.keys().any(|k| !allowed.contains(&k.as_str())) {
                return Err(error("unsupported extra RoPE scaling parameters"));
            }
            if kind != "default" {
                let factor = number(r, "factor", 0.)?;
                if factor < 1. {
                    return Err(error("RoPE scaling factor must be >= 1"));
                }
                if kind == "linear" {
                    scaling = "linear";
                    m.push((
                        format!("{}.rope.scaling.factor", self.arch),
                        G::F32(factor as f32),
                    ));
                } else {
                    let low = number(r, "low_freq_factor", 0.)?;
                    let high = number(r, "high_freq_factor", 0.)?;
                    let original = positive(r, "original_max_position_embeddings", self.context)?;
                    if high <= low {
                        return Err(error("llama3 high_freq_factor must exceed low_freq_factor"));
                    }
                    let values = (0..self.dim / 2)
                        .map(|i| {
                            let frequency =
                                self.theta.powf(-2. * f64::from(i) / f64::from(self.dim));
                            let wavelength = std::f64::consts::TAU / frequency;
                            if wavelength > f64::from(original) / low {
                                factor as f32
                            } else if wavelength < f64::from(original) / high {
                                1.
                            } else {
                                let smooth =
                                    (f64::from(original) / wavelength - low) / (high - low);
                                (1. / ((1. - smooth) / factor + smooth)) as f32
                            }
                        })
                        .collect();
                    factors = Some(values);
                    // Frequency scaling lives entirely in rope_freqs.weight, not
                    // in a second global scale (which would apply it twice).
                }
            }
        }
        m.push((
            format!("{}.rope.scaling.type", self.arch),
            G::String(scaling.into()),
        ));
        Ok((m, factors))
    }
    fn specs(&self) -> Vec<Spec> {
        let h = u64::from(self.hidden);
        let ff = u64::from(self.ff);
        let vocab = u64::from(self.vocab);
        let q = u64::from(self.heads) * u64::from(self.dim);
        let k = u64::from(self.kv) * u64::from(self.dim);
        let mut out = vec![
            Spec::one(
                "token_embd.weight",
                "model.embed_tokens.weight",
                vec![vocab, h],
            ),
            Spec::one("output_norm.weight", "model.norm.weight", vec![h]),
        ];
        if !self.tied {
            out.push(Spec::one("output.weight", "lm_head.weight", vec![vocab, h]));
        }
        for layer in 0..self.layers {
            let hf = format!("model.layers.{layer}");
            let gg = format!("blk.{layer}");
            for (target, source, shape) in [
                ("attn_norm.weight", "input_layernorm.weight", vec![h]),
                (
                    "ffn_norm.weight",
                    "post_attention_layernorm.weight",
                    vec![h],
                ),
                ("attn_q.weight", "self_attn.q_proj.weight", vec![q, h]),
                ("attn_k.weight", "self_attn.k_proj.weight", vec![k, h]),
                ("attn_v.weight", "self_attn.v_proj.weight", vec![k, h]),
                ("attn_output.weight", "self_attn.o_proj.weight", vec![h, q]),
            ] {
                let mut s = Spec::one(&format!("{gg}.{target}"), &format!("{hf}.{source}"), shape);
                if self.arch == "llama" {
                    s.permute = match target {
                        "attn_q.weight" => Some((self.heads, self.dim)),
                        "attn_k.weight" => Some((self.kv, self.dim)),
                        _ => None,
                    };
                }
                out.push(s);
            }
            if self.arch == "qwen2" {
                for (part, shape) in [("q", q), ("k", k), ("v", k)] {
                    out.push(Spec::one(
                        &format!("{gg}.attn_{part}.bias"),
                        &format!("{hf}.self_attn.{part}_proj.bias"),
                        vec![shape],
                    ));
                }
            }
            if self.experts == 0 {
                for (target, source, shape) in [
                    ("ffn_gate.weight", "gate_proj.weight", vec![ff, h]),
                    ("ffn_up.weight", "up_proj.weight", vec![ff, h]),
                    ("ffn_down.weight", "down_proj.weight", vec![h, ff]),
                ] {
                    out.push(Spec::one(
                        &format!("{gg}.{target}"),
                        &format!("{hf}.mlp.{source}"),
                        shape,
                    ));
                }
            } else {
                out.push(Spec::one(
                    &format!("{gg}.ffn_gate_inp.weight"),
                    &format!("{hf}.block_sparse_moe.gate.weight"),
                    vec![u64::from(self.experts), h],
                ));
                for (target, source, shape) in [
                    ("ffn_gate_exps.weight", "w1.weight", vec![ff, h]),
                    ("ffn_down_exps.weight", "w2.weight", vec![h, ff]),
                    ("ffn_up_exps.weight", "w3.weight", vec![ff, h]),
                ] {
                    out.push(Spec {
                        target: format!("{gg}.{target}"),
                        sources: (0..self.experts)
                            .map(|expert| {
                                format!("{hf}.block_sparse_moe.experts.{expert}.{source}")
                            })
                            .collect(),
                        shape,
                        permute: None,
                    });
                }
            }
        }
        out
    }
}
struct Spec {
    target: String,
    sources: Vec<String>,
    shape: Vec<u64>,
    permute: Option<(u32, u32)>,
}
impl Spec {
    fn one(target: &str, source: &str, shape: Vec<u64>) -> Self {
        Self {
            target: target.into(),
            sources: vec![source.into()],
            shape,
            permute: None,
        }
    }
}

pub(super) struct Piece {
    pub tensor: safetensors::Tensor,
    pub permute: Option<(u32, u32)>,
}
pub(super) fn prepare(config: &ImportConfig, path: &Path) -> Result<Prepared, String> {
    if config.ollama_dir.is_some() {
        return Err(error(
            "a local HF directory cannot be combined with ollama_dir",
        ));
    }
    let root = fs::canonicalize(path).map_err(|e| error(e))?;
    let v = read_json(&root.join("config.json"), 1024 * 1024)?;
    let c = Config::parse(&v)?;
    let (mut metadata, factors) = c.metadata(&v)?;
    metadata.extend(hf_tokenizer::metadata(
        &root,
        &v,
        c.vocab,
        c.arch == "qwen2",
    )?);
    let mut tensors = safetensors::load(&root)?;
    let mut expected = c.specs();
    // When present, retain lm_head even for tied configs: some exports contain
    // independently stored head weights. When absent, the native loader ties it.
    if c.tied && tensors.contains_key("lm_head.weight") {
        expected.push(Spec::one(
            "output.weight",
            "lm_head.weight",
            vec![u64::from(c.vocab), u64::from(c.hidden)],
        ));
    }
    let mut specs = Vec::new();
    let mut payloads = Vec::new();
    for s in expected {
        let mut pieces = Vec::new();
        let mut kind = None;
        let mut size = 0u64;
        for name in &s.sources {
            let tensor = tensors
                .remove(name)
                .ok_or_else(|| error(format!("missing required tensor {name}")))?;
            if tensor.shape != s.shape {
                return Err(error(format!(
                    "shape mismatch for {name}: expected {:?}, found {:?}",
                    s.shape, tensor.shape
                )));
            }
            let (k, bytes) = tensor.output_info();
            if kind.is_some_and(|old| old != k) {
                return Err(error(
                    "all experts in a stacked tensor must have the same dtype",
                ));
            }
            kind = Some(k);
            size = size.checked_add(bytes).ok_or("HF output size overflow")?;
            pieces.push(Piece {
                tensor,
                permute: s.permute,
            });
        }
        let mut dimensions: Vec<_> = s.shape.into_iter().rev().collect();
        if s.target.ends_with("_exps.weight") {
            dimensions.push(s.sources.len() as u64);
        }
        specs.push(TensorInfo {
            name: s.target,
            dimensions,
            kind: kind.unwrap(),
            size,
            offset: 0,
        });
        payloads.push(Payload::Hf(pieces));
    }
    if let Some((name, _)) = tensors.first_key_value() {
        return Err(error(format!(
            "unrecognized tensor {name}; adapters, rotary caches, and other extra tensors are not silently ignored"
        )));
    }
    if let Some(values) = factors {
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        specs.push(TensorInfo {
            name: "rope_freqs.weight".into(),
            dimensions: vec![values.len() as u64],
            kind: 0,
            size: bytes.len() as u64,
            offset: 0,
        });
        payloads.push(Payload::Bytes(bytes));
    }
    Ok(Prepared {
        layout: Layout::new(metadata, specs)?,
        payloads,
        root,
        name: config.model.clone(),
        skipped: 0,
        fingerprint: None,
        notes: NOTES,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        collections::BTreeMap,
        io::{Read, Seek, SeekFrom, Write},
    };

    fn config(family: &str, tied: bool) -> Value {
        let (architecture, kind) = match family {
            "llama" => ("LlamaForCausalLM", "llama"),
            "mistral" => ("MistralForCausalLM", "mistral"),
            "mixtral" => ("MixtralForCausalLM", "mixtral"),
            "qwen2" => ("Qwen2ForCausalLM", "qwen2"),
            _ => panic!(),
        };
        let mut v = json!({"architectures":[architecture],"model_type":kind,"hidden_act":"silu","hidden_size":8,"intermediate_size":12,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":262,"max_position_embeddings":128,"rms_norm_eps":0.00001,"rope_theta":10000.,"tie_word_embeddings":tied,"bos_token_id":256,"eos_token_id":257});
        if family == "mixtral" {
            v["num_local_experts"] = json!(2);
            v["num_experts_per_tok"] = json!(1);
        }
        v
    }
    fn tokenizer(root: &Path, qwen: bool) {
        let mut vocab = serde_json::Map::new();
        for (id, token) in hf_tokenizer::byte_symbols().into_iter().enumerate() {
            vocab.insert(token, json!(id));
        }
        vocab.insert("ab".into(), json!(260));
        let added: Vec<_> = ["<s>","</s>",if qwen {"<|im_end|>"} else {"<|eot_id|>"},"<|eom_id|>"].iter().enumerate().map(|(i,t)| json!({"id":256+i,"content":t,"special":true,"normalized":false,"lstrip":false,"rstrip":false,"single_word":false})).collect();
        let t = json!({"version":"1.0","normalizer":null,"added_tokens":added,"model":{"type":"BPE","vocab":vocab,"merges":[["a","b"]],"ignore_merges":!qwen,"byte_fallback":false},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":if qwen {hf_tokenizer::QWEN_SPLIT} else {hf_tokenizer::LLAMA_SPLIT}},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false,"trim_offsets":true}]},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}});
        fs::write(root.join("tokenizer.json"), serde_json::to_vec(&t).unwrap()).unwrap();
        let tc = json!({"bos_token":"<s>","eos_token":"</s>","add_bos_token":!qwen,"add_eos_token":false,"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}"});
        fs::write(
            root.join("tokenizer_config.json"),
            serde_json::to_vec(&tc).unwrap(),
        )
        .unwrap();
    }
    fn shard(path: &Path, tensors: &[(String, Vec<u64>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.clone(),
                json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
            );
        }
        let mut bytes = serde_json::to_vec(&header).unwrap();
        while bytes.len() % 8 != 0 {
            bytes.push(b' ');
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
        f.write_all(&data).unwrap();
    }
    fn fixture(root: &Path, family: &str, tied: bool, sharded: bool) -> ImportConfig {
        fs::create_dir(root).unwrap();
        let v = config(family, tied);
        let c = Config::parse(&v).unwrap();
        fs::write(root.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
        tokenizer(root, c.arch == "qwen2");
        let mut tensors = Vec::new();
        for spec in c.specs() {
            for (expert, name) in spec.sources.into_iter().enumerate() {
                let size = spec.shape.iter().product::<u64>() as usize;
                let values = (0..size)
                    .map(|n| n as f32 + expert as f32 * 1000.)
                    .collect();
                tensors.push((name, spec.shape.clone(), values));
            }
        }
        if sharded {
            let midpoint = tensors.len() / 2;
            let mut map = BTreeMap::new();
            for (name, part) in [
                ("first.safetensors", &tensors[..midpoint]),
                ("second.safetensors", &tensors[midpoint..]),
            ] {
                shard(&root.join(name), part);
                for (tensor, _, _) in part {
                    map.insert(tensor, name);
                }
            }
            fs::write(
                root.join("model.safetensors.index.json"),
                serde_json::to_vec(&json!({"weight_map":map})).unwrap(),
            )
            .unwrap();
        } else {
            shard(&root.join("model.safetensors"), &tensors);
        }
        ImportConfig {
            model: root.to_str().unwrap().into(),
            ollama_dir: None,
            output: root.parent().unwrap().join(format!("{family}.gguf")),
        }
    }
    fn values(path: &Path, prepared: &Prepared, name: &str) -> Vec<f32> {
        let info = prepared
            .layout
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap();
        let mut file = fs::File::open(path).unwrap();
        file.seek(SeekFrom::Start(prepared.layout.data_start + info.offset))
            .unwrap();
        let mut bytes = vec![0; info.size as usize];
        file.read_exact(&mut bytes).unwrap();
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn assert_metadata(path: &Path, arch: &str, pre: &str) {
        let filename = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        let params = llama_cpp_sys_2::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        // Header-only native reader: no backend/model/context or inference process.
        let ctx = unsafe { llama_cpp_sys_2::gguf_init_from_file(filename.as_ptr(), params) };
        assert!(!ctx.is_null());
        for (key, expected) in [("general.architecture", arch), ("tokenizer.ggml.pre", pre)] {
            let key = std::ffi::CString::new(key).unwrap();
            let id = unsafe { llama_cpp_sys_2::gguf_find_key(ctx, key.as_ptr()) };
            assert!(id >= 0);
            let actual =
                unsafe { std::ffi::CStr::from_ptr(llama_cpp_sys_2::gguf_get_val_str(ctx, id)) }
                    .to_str()
                    .unwrap();
            assert_eq!(actual, expected);
        }
        unsafe {
            llama_cpp_sys_2::gguf_free(ctx);
        }
    }
    #[test]
    fn dense_families_shards_tied_embeddings_metadata_and_rope_rows() {
        for (family, tied, sharded) in [
            ("llama", true, false),
            ("mistral", false, true),
            ("qwen2", false, true),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let cfg = fixture(&temp.path().join("source"), family, tied, sharded);
            let prepared = prepare(&cfg, Path::new(&cfg.model)).unwrap();
            let preview = crate::import::preview(&cfg).unwrap();
            assert_eq!(preview.output_bytes, prepared.layout.total_bytes);
            assert!(!cfg.output.exists());
            let emb = prepared
                .layout
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .unwrap();
            assert_eq!(emb.dimensions, vec![8, 262]);
            assert_eq!(
                prepared
                    .layout
                    .tensors
                    .iter()
                    .any(|t| t.name == "output.weight"),
                !tied
            );
            crate::import::run(cfg.clone()).unwrap();
            assert_eq!(
                fs::metadata(&cfg.output).unwrap().len(),
                preview.output_bytes
            );
            assert_metadata(
                &cfg.output,
                if family == "qwen2" { "qwen2" } else { "llama" },
                if family == "qwen2" {
                    "qwen2"
                } else {
                    "llama-bpe"
                },
            );
            let q = values(&cfg.output, &prepared, "blk.0.attn_q.weight");
            assert_eq!(
                &q[8..16],
                if family == "qwen2" {
                    &[8., 9., 10., 11., 12., 13., 14., 15.]
                } else {
                    &[16., 17., 18., 19., 20., 21., 22., 23.]
                }
            );
            let k = values(&cfg.output, &prepared, "blk.0.attn_k.weight");
            assert_eq!(k[8], if family == "qwen2" { 8. } else { 16. });
            assert!(
                crate::import::run(cfg.clone())
                    .unwrap_err()
                    .contains("never overwrites")
            );
            let mut forbidden = cfg.clone();
            forbidden.output = Path::new(&cfg.model).join("forbidden.gguf");
            assert!(
                crate::import::run(forbidden)
                    .unwrap_err()
                    .contains("outside")
            );
            let mut ambiguous = cfg;
            ambiguous.ollama_dir = Some(temp.path().into());
            assert!(crate::import::preview(&ambiguous).is_err());
        }
    }
    #[test]
    fn mixtral_experts_stack_in_numeric_order_without_transposition() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = fixture(&temp.path().join("source"), "mixtral", false, true);
        let p = prepare(&cfg, Path::new(&cfg.model)).unwrap();
        for (name, dims) in [
            ("blk.0.ffn_gate_exps.weight", vec![8, 12, 2]),
            ("blk.0.ffn_down_exps.weight", vec![12, 8, 2]),
            ("blk.0.ffn_up_exps.weight", vec![8, 12, 2]),
        ] {
            assert_eq!(
                p.layout
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .unwrap()
                    .dimensions,
                dims
            );
        }
        crate::import::run(cfg.clone()).unwrap();
        let gate = values(&cfg.output, &p, "blk.0.ffn_gate_exps.weight");
        assert_eq!(gate.len(), 192);
        assert_eq!(&gate[94..98], &[94., 95., 1000., 1001.]);
        let (m, _) = Config::parse(&config("mixtral", false))
            .unwrap()
            .metadata(&config("mixtral", false))
            .unwrap();
        assert!(m.contains(&("llama.expert_count".into(), G::U32(2))));
    }
    #[test]
    fn tokenizer_metadata_keeps_raw_tokens_padding_merges_templates_and_stop_ids() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        tokenizer(root, false);
        let c = config("llama", true);
        fs::write(
            root.join("generation_config.json"),
            br#"{"eos_token_id":[257,258,259]}"#,
        )
        .unwrap();
        let m: BTreeMap<_, _> = hf_tokenizer::metadata(root, &c, 262, false)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(m["tokenizer.ggml.model"], G::String("gpt2".into()));
        assert_eq!(m["tokenizer.ggml.pre"], G::String("llama-bpe".into()));
        assert_eq!(m["tokenizer.ggml.bos_token_id"], G::U32(256));
        assert_eq!(m["tokenizer.ggml.eos_token_id"], G::U32(257));
        assert_eq!(m["tokenizer.ggml.eot_token_id"], G::U32(258));
        assert_eq!(m["tokenizer.ggml.eom_token_id"], G::U32(259));
        assert_eq!(m["tokenizer.ggml.merges"], G::Strings(vec!["a b".into()]));
        assert_eq!(m["tokenizer.ggml.add_bos_token"], G::Bool(true));
        assert!(m.contains_key("tokenizer.chat_template"));
        let G::Strings(tokens) = &m["tokenizer.ggml.tokens"] else {
            panic!()
        };
        assert_eq!(tokens[260], "ab");
        assert_eq!(tokens[261], "[PAD261]");
        let G::I32s(types) = &m["tokenizer.ggml.token_type"] else {
            panic!()
        };
        assert_eq!(&types[256..], &[3, 3, 3, 3, 1, 5]);
        fs::write(
            root.join("generation_config.json"),
            br#"{"eos_token_id":[257,260]}"#,
        )
        .unwrap();
        assert!(hf_tokenizer::metadata(root, &c, 262, false).is_err());
        fs::remove_file(root.join("generation_config.json")).unwrap();
        let mut invalid = c;
        invalid["eos_token_id"] = json!(261);
        invalid["bos_token_id"] = json!(261);
        assert!(hf_tokenizer::metadata(root, &invalid, 262, false).is_err());
        let mut t = read_json(&root.join("tokenizer.json"), 1024 * 1024).unwrap();
        t["normalizer"] = json!({"type":"NFC"});
        fs::write(root.join("tokenizer.json"), serde_json::to_vec(&t).unwrap()).unwrap();
        assert!(
            hf_tokenizer::metadata(root, &invalid, 262, false)
                .unwrap_err()
                .contains("normalization")
        );
    }
    #[test]
    fn scaling_is_explicit_and_llama3_factors_are_not_applied_twice() {
        let mut v = config("llama", true);
        v["rope_scaling"] = json!({"rope_type":"linear","factor":4.});
        let c = Config::parse(&v).unwrap();
        let (m, f) = c.metadata(&v).unwrap();
        assert!(f.is_none());
        assert!(m.contains(&("llama.rope.scaling.factor".into(), G::F32(4.))));
        v["rope_scaling"] = json!({"rope_type":"llama3","factor":8.,"low_freq_factor":1.,"high_freq_factor":4.,"original_max_position_embeddings":128});
        let (m, f) = c.metadata(&v).unwrap();
        assert!(!m.iter().any(|(k, _)| k == "llama.rope.scaling.factor"));
        let f = f.unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(f[0], 1.);
        assert_eq!(f[1], 8.);
        v["rope_scaling"] = json!({"rope_type":"dynamic","factor":2.});
        assert!(c.metadata(&v).is_err());
    }
    #[test]
    fn unsupported_configs_missing_weights_shapes_and_extras_fail_before_publication() {
        let base = config("llama", true);
        for (key, value) in [
            ("model_type", json!("gemma")),
            ("hidden_act", json!("gelu")),
            ("sliding_window", json!(64)),
            ("attention_bias", json!(true)),
            ("mlp_bias", json!(true)),
            ("partial_rotary_factor", json!(0.5)),
            ("quantization_config", json!({})),
            ("num_key_value_heads", json!(3)),
            ("architectures", json!(["Qwen2MoeForCausalLM"])),
        ] {
            let mut v = base.clone();
            v[key] = value;
            assert!(Config::parse(&v).is_err(), "accepted {key}");
        }
        let temp = tempfile::tempdir().unwrap();
        let cfg = fixture(&temp.path().join("source"), "llama", true, false);
        let shard_path = Path::new(&cfg.model).join("model.safetensors");
        for tensors in [
            vec![(
                "model.embed_tokens.weight".into(),
                vec![262, 7],
                vec![0.; 262 * 7],
            )],
            vec![("adapter.weight".into(), vec![1], vec![0.])],
        ] {
            shard(&shard_path, &tensors);
            assert!(crate::import::run(cfg.clone()).is_err());
            assert!(!cfg.output.exists());
        }
    }
}
