//! Dense Gemma4 text conversion for the llama.cpp bundled with llama-cpp-2 0.1.160.
//!
//! No tensor data is read here. Call `metadata` and `expected_tensors` before
//! writing weights. Shape dimensions below are GGUF/ggml order (HF reversed).
//! Gemma4 norm weights are already multiplicative: do NOT add one to them.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::gguf::Value as G;

// Text-only fallback for the native unified protocol, not Gemma2/3's template.
// A caller using llama_model_apply_chat_template must select its dedicated
// Gemma4 formatter if the pinned template recognizer falls back to plain text.
pub(super) const CHAT_TEMPLATE: &str = "{{ bos_token }}{% for message in messages %}{% if message['role'] not in ['system', 'user', 'assistant'] %}{{ raise_exception('Gemma4 text template does not support tools') }}{% endif %}{% if message['content'] is not string %}{{ raise_exception('Gemma4 text template requires string content') }}{% endif %}{{ '<|turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\\n' + message['content'] + '<turn|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|turn>model\\n' }}{% endif %}";

#[derive(Debug)]
struct Config {
    hidden: u32,
    ff: u32,
    layers: u32,
    heads: u32,
    kv: u32,
    global_kv: u32,
    head_dim: u32,
    global_dim: u32,
    vocab: u32,
    context: u32,
    window: u32,
    eps: f32,
    softcap: f32,
    full_theta: f32,
    swa_theta: f32,
    full_rot_pairs: usize,
    swa_rot: u32,
    swa: Vec<bool>,
    k_eq_v: bool,
}

fn error(message: impl std::fmt::Display) -> String {
    format!("Gemma4 import: {message}")
}

fn uint(v: &Value, key: &str) -> Result<u32, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| error(format!("{key} must be a u32")))
}

fn positive(v: &Value, key: &str, max: u32) -> Result<u32, String> {
    let n = uint(v, key)?;
    if n == 0 || n > max {
        return Err(error(format!("{key} must be in 1..={max}")));
    }
    Ok(n)
}

fn number(v: &Value, key: &str) -> Result<f32, String> {
    let n = v
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| error(format!("{key} must be numeric")))?;
    if !n.is_finite() || !(n as f32).is_finite() {
        return Err(error(format!("{key} must be finite f32")));
    }
    Ok(n as f32)
}

fn boolean(v: &Value, key: &str, default: bool) -> Result<bool, String> {
    match v.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| error(format!("{key} must be boolean"))),
    }
}

fn require(v: &Value, key: &str, expected: Value) -> Result<(), String> {
    if v.get(key) != Some(&expected) {
        return Err(error(format!("unsupported {key}: expected {expected}")));
    }
    Ok(())
}

fn absent_or_zero(v: &Value, key: &str) -> Result<(), String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(()),
        Some(n) if n.as_u64() == Some(0) => Ok(()),
        _ => Err(error(format!(
            "{key} is not supported by the dense text importer"
        ))),
    }
}

fn empty_option(v: &Value, key: &str) -> Result<(), String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(o)) if o.is_empty() => Ok(()),
        _ => Err(error(format!("{key} overrides are unsupported"))),
    }
}

fn rope(v: &Value, key: &str, kind: &str, dim: u32) -> Result<(f32, u32), String> {
    let p = v
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| error(format!("rope_parameters.{key} is required")))?;
    if p.keys().any(|k| {
        !matches!(
            k.as_str(),
            "rope_type" | "rope_theta" | "partial_rotary_factor"
        )
    }) {
        return Err(error(format!("unsupported RoPE scaling in {key}")));
    }
    let p = &v[key];
    require(p, "rope_type", json!(kind))?;
    let theta = number(p, "rope_theta")?;
    if theta <= 0.0 {
        return Err(error("rope_theta must be positive"));
    }
    let fraction = match p.get("partial_rotary_factor") {
        None if kind == "default" => 1.0,
        _ => p["partial_rotary_factor"]
            .as_f64()
            .ok_or_else(|| error("partial_rotary_factor must be numeric"))?,
    };
    let rotated = f64::from(dim) * fraction;
    if !fraction.is_finite()
        || fraction <= 0.0
        || fraction > 1.0
        || rotated.fract() != 0.0
        || rotated < 2.0
        || !(rotated as u32).is_multiple_of(2)
    {
        return Err(error(
            "partial_rotary_factor must produce a positive even rotary dimension count",
        ));
    }
    Ok((theta, rotated as u32))
}

fn validate(config: &Value) -> Result<Config, String> {
    let arch = config["architectures"]
        .as_array()
        .filter(|a| a.len() == 1)
        .and_then(|a| a[0].as_str())
        .ok_or_else(|| error("exactly one Gemma4 architecture is required"))?;
    let (model_type, text_type, nested) = match arch {
        "Gemma4UnifiedForConditionalGeneration" => ("gemma4_unified", "gemma4_unified_text", true),
        "Gemma4ForConditionalGeneration" => ("gemma4", "gemma4_text", true),
        "Gemma4ForCausalLM" => ("gemma4_text", "gemma4_text", false),
        _ => return Err(error(format!("unsupported architecture {arch}"))),
    };
    require(config, "model_type", json!(model_type))?;
    let t = if nested {
        config
            .get("text_config")
            .filter(|v| v.is_object())
            .ok_or_else(|| error("text_config must be an object"))?
    } else {
        config
    };
    require(t, "model_type", json!(text_type))?;
    require(t, "hidden_activation", json!("gelu_pytorch_tanh"))?;
    require(t, "tie_word_embeddings", json!(true))?;
    if !boolean(config, "tie_word_embeddings", true)? {
        return Err(error("untied embeddings are unsupported"));
    }
    for key in [
        "attention_bias",
        "mlp_bias",
        "enable_moe_block",
        "use_double_wide_mlp",
    ] {
        if boolean(t, key, false)? {
            return Err(error(format!("{key}=true is unsupported")));
        }
    }
    for key in [
        "hidden_size_per_layer_input",
        "num_kv_shared_layers",
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "top_k_experts",
        "moe_intermediate_size",
        "expert_intermediate_size",
        "num_nextn_predict_layers",
    ] {
        absent_or_zero(t, key)?;
    }
    for key in ["per_layer_config", "rope_scaling"] {
        empty_option(t, key)?;
    }
    if let Some(bidir) = t.get("use_bidirectional_attention")
        && bidir != "vision"
        && bidir != &json!(false)
        && !bidir.is_null()
    {
        return Err(error(
            "only causal text / vision-only bidirectional attention is supported",
        ));
    }
    if t.get("attn_logit_softcapping")
        .is_some_and(|v| !v.is_null())
    {
        return Err(error("attention logit softcapping is unsupported"));
    }
    for (key, id) in [
        ("pad_token_id", 0),
        ("eos_token_id", 1),
        ("bos_token_id", 2),
    ] {
        require(t, key, json!(id))?;
    }
    let hidden = positive(t, "hidden_size", 65536)?;
    let ff = positive(t, "intermediate_size", 1_048_576)?;
    let layers = positive(t, "num_hidden_layers", 512)?;
    let heads = positive(t, "num_attention_heads", 1024)?;
    let kv = positive(t, "num_key_value_heads", heads)?;
    let global_kv = positive(t, "num_global_key_value_heads", heads)?;
    let head_dim = positive(t, "head_dim", 4096)?;
    let global_dim = positive(t, "global_head_dim", 4096)?;
    if heads % kv != 0 || heads % global_kv != 0 || head_dim % 2 != 0 || global_dim % 2 != 0 {
        return Err(error("invalid GQA divisibility or odd head dimension"));
    }
    let vocab = positive(t, "vocab_size", 1_048_576)?;
    let context = positive(t, "max_position_embeddings", i32::MAX as u32)?;
    let window = positive(t, "sliding_window", context)?;
    let eps = number(t, "rms_norm_eps")?;
    let softcap = number(t, "final_logit_softcapping")?;
    if eps <= 0.0 || softcap < 0.0 {
        return Err(error("invalid RMS epsilon or final logit softcap"));
    }
    let types = t["layer_types"]
        .as_array()
        .filter(|a| a.len() == layers as usize)
        .ok_or_else(|| error("layer_types must have exactly num_hidden_layers entries"))?;
    let swa: Vec<bool> = types
        .iter()
        .map(|v| match v.as_str() {
            Some("sliding_attention") => Ok(true),
            Some("full_attention") => Ok(false),
            _ => Err(error("layer_types contains unsupported attention type")),
        })
        .collect::<Result<_, _>>()?;
    if !swa.contains(&true) || !swa.contains(&false) {
        return Err(error("expected both sliding and full attention layers"));
    }
    let rp = t
        .get("rope_parameters")
        .filter(|v| v.is_object())
        .ok_or_else(|| error("rope_parameters must be an object"))?;
    if rp
        .as_object()
        .unwrap()
        .keys()
        .any(|k| k != "full_attention" && k != "sliding_attention")
    {
        return Err(error("unsupported rope_parameters field"));
    }
    let (full_theta, full_rot) = rope(rp, "full_attention", "proportional", global_dim)?;
    let (swa_theta, swa_rot) = rope(rp, "sliding_attention", "default", head_dim)?;
    Ok(Config {
        hidden,
        ff,
        layers,
        heads,
        kv,
        global_kv,
        head_dim,
        global_dim,
        vocab,
        context,
        window,
        eps,
        softcap,
        full_theta,
        swa_theta,
        full_rot_pairs: (full_rot / 2) as usize,
        swa_rot,
        swa,
        k_eq_v: boolean(t, "attention_k_eq_v", false)?,
    })
}

pub(super) fn metadata(
    config: &Value,
    tokenizer: &Value,
    tokenizer_config: &Value,
    generation_config: Option<&Value>,
) -> Result<Vec<(String, G)>, String> {
    let c = validate(config)?;
    let mut entries = vec![
        ("general.architecture".into(), G::String("gemma4".into())),
        ("general.name".into(), G::String("Gemma4 dense text".into())),
    ];
    for (key, value) in [
        ("context_length", c.context),
        ("embedding_length", c.hidden),
        ("block_count", c.layers),
        ("feed_forward_length", c.ff),
        ("vocab_size", c.vocab),
        ("embedding_length_per_layer_input", 0),
        ("attention.head_count", c.heads),
        ("attention.key_length", c.global_dim),
        ("attention.value_length", c.global_dim),
        ("attention.key_length_swa", c.head_dim),
        ("attention.value_length_swa", c.head_dim),
        ("attention.shared_kv_layers", 0),
        ("attention.sliding_window", c.window),
        ("rope.dimension_count", c.global_dim),
        ("rope.dimension_count_swa", c.swa_rot),
    ] {
        entries.push((format!("gemma4.{key}"), G::U32(value)));
    }
    for (key, value) in [
        ("attention.layer_norm_rms_epsilon", c.eps),
        ("final_logit_softcapping", c.softcap),
        ("rope.freq_base", c.full_theta),
        ("rope.freq_base_swa", c.swa_theta),
    ] {
        entries.push((format!("gemma4.{key}"), G::F32(value)));
    }
    entries.push((
        "gemma4.attention.head_count_kv".into(),
        G::U32s(
            c.swa
                .iter()
                .map(|s| if *s { c.kv } else { c.global_kv })
                .collect(),
        ),
    ));
    entries.push((
        "gemma4.attention.sliding_window_pattern".into(),
        G::Bools(c.swa),
    ));
    entries.extend(tokenizer_metadata(
        tokenizer,
        tokenizer_config,
        generation_config,
        c.vocab,
    )?);
    Ok(entries)
}

/// Only one global tensor is needed: the pinned LLM_TENSOR_ROPE_FREQS name is
/// `rope_freqs`, without a block placeholder, even when requested for layer 5.
pub(super) fn extra_tensors(config: &Value) -> Result<Vec<(String, Vec<f32>)>, String> {
    let c = validate(config)?;
    let mut factors = vec![1e30; (c.global_dim / 2) as usize];
    factors[..c.full_rot_pairs].fill(1.0);
    Ok(vec![("rope_freqs.weight".into(), factors)])
}

/// Required converted tensors, in GGUF dimension order, including generated
/// RoPE. Excludes optional `output.weight` (tied), optional projection `.scale`
/// scalars, and optional layer output scales. A present layer_scalar must be
/// retained as `blk.N.layer_output_scale.weight`, shape [1]. V is required for
/// SWA, and forbidden for full attention when attention_k_eq_v=true.
pub fn expected_tensors(config: &Value) -> Result<BTreeMap<String, Vec<u64>>, String> {
    let c = validate(config)?;
    let h = u64::from(c.hidden);
    let f = u64::from(c.ff);
    let mut tensors = BTreeMap::from([
        ("token_embd.weight".into(), vec![h, u64::from(c.vocab)]),
        ("output_norm.weight".into(), vec![h]),
        (
            "rope_freqs.weight".into(),
            vec![u64::from(c.global_dim / 2)],
        ),
    ]);
    for (i, swa) in c.swa.iter().copied().enumerate() {
        let d = u64::from(if swa { c.head_dim } else { c.global_dim });
        let kv = u64::from(if swa { c.kv } else { c.global_kv });
        let q = d * u64::from(c.heads);
        for name in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            tensors.insert(format!("blk.{i}.{name}.weight"), vec![h]);
        }
        for name in ["attn_q_norm", "attn_k_norm"] {
            tensors.insert(format!("blk.{i}.{name}.weight"), vec![d]);
        }
        for (name, shape) in [
            ("attn_q", vec![h, q]),
            ("attn_k", vec![h, d * kv]),
            ("attn_output", vec![q, h]),
            ("ffn_gate", vec![h, f]),
            ("ffn_up", vec![h, f]),
            ("ffn_down", vec![f, h]),
        ] {
            tensors.insert(format!("blk.{i}.{name}.weight"), shape);
        }
        if swa || !c.k_eq_v {
            tensors.insert(format!("blk.{i}.attn_v.weight"), vec![h, d * kv]);
        }
    }
    Ok(tensors)
}

/// A closed mapping: quantization auxiliaries must be consumed by the tensor
/// decoder, not passed here as independent model weights. Unknown text names,
/// including biases, experts and train-time buffers, are errors, never skips.
pub(super) fn tensor_name(hf: &str) -> Result<Option<String>, String> {
    const NON_TEXT: &[&str] = &[
        "model.embed_vision.",
        "model.vision_embedder.",
        "model.vision_tower.",
        "model.embed_audio.",
        "model.audio_embedder.",
        "model.audio_tower.",
        "draft.",
    ];
    if NON_TEXT.iter().any(|p| hf.starts_with(p)) {
        return Ok(None);
    }
    let name = hf
        .strip_prefix("model.language_model.")
        .or_else(|| hf.strip_prefix("language_model.model."))
        .or_else(|| hf.strip_prefix("language_model."))
        .or_else(|| hf.strip_prefix("model."))
        .unwrap_or(hf);
    let global = match name {
        "embed_tokens.weight" => Some("token_embd.weight"),
        "norm.weight" => Some("output_norm.weight"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(name) = global {
        return Ok(Some(name.into()));
    }
    if let Some(rest) = name.strip_prefix("layers.")
        && let Some((index, suffix)) = rest.split_once('.')
    {
        let i = index
            .parse::<u32>()
            .map_err(|_| error(format!("invalid layer index: {hf}")))?;
        if i >= 512 || index != i.to_string() {
            return Err(error(format!("invalid layer index: {hf}")));
        }
        let target = match suffix {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_norm.weight",
            "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
            "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "layer_scalar" | "layer_scalar.weight" => "layer_output_scale.weight",
            _ => return Err(error(format!("unsupported text tensor {hf}"))),
        };
        return Ok(Some(format!("blk.{i}.{target}")));
    }
    Err(error(format!("unsupported tensor {hf}")))
}

fn pipeline(tokenizer: &Value) -> Result<(), String> {
    for (key, expected) in [
        (
            "normalizer",
            json!({"type":"Replace","pattern":{"String":" "},"content":"▁"}),
        ),
        (
            "pre_tokenizer",
            json!({"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}),
        ),
        (
            "decoder",
            json!({"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"▁"},"content":" "},
            {"type":"ByteFallback"},{"type":"Fuse"}]}),
        ),
        (
            "post_processor",
            json!({"type":"TemplateProcessing",
            "single":[{"Sequence":{"id":"A","type_id":0}}],
            "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
            "special_tokens":{}}),
        ),
    ] {
        if tokenizer.get(key) != Some(&expected) {
            return Err(error(format!(
                "unsupported tokenizer {key}; expected native Gemma4 pipeline"
            )));
        }
    }
    if !tokenizer["truncation"].is_null() || !tokenizer["padding"].is_null() {
        return Err(error("tokenizer-side padding/truncation is unsupported"));
    }
    let model = &tokenizer["model"];
    require(model, "type", json!("BPE"))?;
    require(model, "byte_fallback", json!(true))?;
    require(model, "unk_token", json!("<unk>"))?;
    if boolean(model, "ignore_merges", false)? || !boolean(model, "fuse_unk", true)? {
        return Err(error("unsupported BPE ignore_merges/fuse_unk"));
    }
    for key in ["dropout", "continuing_subword_prefix", "end_of_word_suffix"] {
        if model.get(key).is_some_and(|v| !v.is_null()) {
            return Err(error(format!("unsupported BPE {key}")));
        }
    }
    Ok(())
}

fn is_byte(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() == 6
        && b.starts_with(b"<0x")
        && b[3].is_ascii_hexdigit()
        && b[4].is_ascii_hexdigit()
        && b[5] == b'>'
}

fn is_unused(text: &str) -> bool {
    text.strip_prefix("<unused")
        .and_then(|s| s.strip_suffix('>'))
        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
}

fn token_content<'a>(config: &'a Value, key: &str) -> Result<Option<&'a str>, String> {
    match config.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(v) => v
            .get("content")
            .and_then(Value::as_str)
            .map(Some)
            .ok_or_else(|| error(format!("invalid tokenizer_config.{key}"))),
    }
}

fn ids(v: &Value, n_vocab: u32, label: &str) -> Result<Vec<u32>, String> {
    let values: Vec<&Value> = if let Some(a) = v.as_array() {
        a.iter().collect()
    } else {
        vec![v]
    };
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for v in values {
        let id = v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n < n_vocab)
            .ok_or_else(|| error(format!("invalid {label} token ID")))?;
        if !seen.insert(id) {
            return Err(error(format!("duplicate {label} token ID {id}")));
        }
        result.push(id);
    }
    Ok(result)
}

fn tokenizer_metadata(
    tokenizer: &Value,
    tc: &Value,
    generation: Option<&Value>,
    n_vocab: u32,
) -> Result<Vec<(String, G)>, String> {
    pipeline(tokenizer)?;
    if !tc.is_object() || generation.is_some_and(|g| !g.is_object()) {
        return Err(error("tokenizer/generation config must be objects"));
    }
    let vocab = tokenizer["model"]["vocab"]
        .as_object()
        .ok_or_else(|| error("BPE vocab must be a token-to-ID object"))?;
    if n_vocab == 0 || n_vocab > 1_048_576 || vocab.len() > n_vocab as usize {
        return Err(error("invalid tokenizer vocabulary size"));
    }
    let mut tokens = vec![None; n_vocab as usize];
    let mut lookup = BTreeMap::<String, u32>::new();
    for (text, value) in vocab {
        let id = value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n < n_vocab)
            .ok_or_else(|| error(format!("invalid vocab ID for {text:?}")))?;
        if text.is_empty() || text.contains('\0') || tokens[id as usize].is_some() {
            return Err(error("empty/NUL token or duplicate vocabulary ID"));
        }
        tokens[id as usize] = Some(text.clone());
        lookup.insert(text.clone(), id);
    }
    let added = tokenizer["added_tokens"]
        .as_array()
        .ok_or_else(|| error("added_tokens must be an array"))?;
    let mut specials = BTreeSet::new();
    let mut user_defined = BTreeSet::new();
    let mut added_ids = BTreeSet::new();
    for a in added {
        let id = uint(a, "id")?;
        if id >= n_vocab || !added_ids.insert(id) {
            return Err(error("out-of-range or duplicate added-token ID"));
        }
        let text = a["content"]
            .as_str()
            .filter(|s| !s.is_empty() && !s.contains('\0'))
            .ok_or_else(|| error("added token content must be nonempty text"))?;
        for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
            if boolean(a, flag, false)? {
                return Err(error(format!(
                    "added-token {flag}=true cannot be represented in GGUF"
                )));
            }
        }
        if let Some(other) = lookup.get(text)
            && *other != id
        {
            return Err(error("added token conflicts with base vocabulary ID"));
        }
        if let Some(old) = &tokens[id as usize] {
            // Some exports rename reserved slots with a human-readable alias.
            // JSON strings are already unescaped; do not unescape them again.
            if old != text && !is_unused(old) {
                return Err(error(format!("added token aliases non-reserved ID {id}")));
            }
        }
        tokens[id as usize] = Some(text.to_owned());
        lookup.insert(text.to_owned(), id);
        if boolean(a, "special", false)? {
            specials.insert(id);
        } else {
            user_defined.insert(id);
        }
    }
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.ok_or_else(|| error(format!("vocabulary has a hole at ID {i}"))))
        .collect::<Result<_, _>>()?;
    let resolve = |text: &str| -> Result<u32, String> {
        lookup
            .get(text)
            .copied()
            .filter(|id| tokens[*id as usize] == text)
            .ok_or_else(|| error(format!("special token {text:?} is missing")))
    };
    let mut entries = Vec::new();
    let mut stop_ids = BTreeSet::new();
    for (field, text, fixed) in [
        ("pad", "<pad>", Some(0)),
        ("eos", "<eos>", Some(1)),
        ("bos", "<bos>", Some(2)),
        ("unknown", "<unk>", None),
        ("mask", "<mask>", None),
        ("eot", "<turn|>", None),
        ("eom", "<|tool_response>", None),
    ] {
        let config_key = match field {
            "unknown" => "unk_token".to_owned(),
            "eom" => "str_token".to_owned(),
            _ => format!("{field}_token"),
        };
        if let Some(configured) = token_content(tc, &config_key)?
            && configured != text
        {
            return Err(error(format!("unsupported {config_key}: {configured:?}")));
        }
        let id = resolve(text)?;
        if fixed.is_some_and(|n| n != id) {
            return Err(error(format!("unexpected {field} token ID {id}")));
        }
        specials.insert(id);
        if matches!(field, "eos" | "eot" | "eom") {
            stop_ids.insert(id);
        }
        let metadata_field = if field == "pad" { "padding" } else { field };
        entries.push((
            format!("tokenizer.ggml.{metadata_field}_token_id"),
            G::U32(id),
        ));
    }
    for (key, text) in [
        ("sot_token", "<|turn>"),
        ("soc_token", "<|channel>"),
        ("eoc_token", "<channel|>"),
    ] {
        specials.insert(resolve(text)?);
        if token_content(tc, key)?.is_some_and(|s| s != text) {
            return Err(error(format!("unsupported {key}")));
        }
    }
    if !boolean(tc, "add_bos_token", true)?
        || boolean(tc, "add_eos_token", false)?
        || boolean(tc, "add_prefix_space", false)?
    {
        return Err(error(
            "native Gemma4 requires BOS, no automatic EOS, and no space prefix",
        ));
    }
    let visible = [
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|\"|>",
    ];
    for byte in 0..=255u8 {
        resolve(&format!("<0x{byte:02X}>"))?;
    }
    let types = tokens
        .iter()
        .enumerate()
        .map(|(id, text)| {
            if is_byte(text) {
                6
            } else if text == "<unk>" {
                2
            } else if is_unused(text) {
                5
            } else if visible.contains(&text.as_str()) {
                4
            } else if specials.contains(&(id as u32)) {
                3
            } else if user_defined.contains(&(id as u32)) {
                4
            } else {
                1
            }
        })
        .collect();
    let raw_merges = tokenizer["model"]["merges"]
        .as_array()
        .filter(|m| !m.is_empty() && m.len() <= 4_000_000)
        .ok_or_else(|| error("BPE merges must be a nonempty bounded array"))?;
    let mut merges = Vec::with_capacity(raw_merges.len());
    for merge in raw_merges {
        let (left, right) = if let Some(a) = merge.as_array() {
            if a.len() != 2 {
                return Err(error("BPE merge must contain exactly two strings"));
            }
            (a[0].as_str(), a[1].as_str())
        } else if let Some(s) = merge.as_str() {
            s.split_once(' ')
                .map(|(a, b)| (Some(a), Some(b)))
                .unwrap_or((None, None))
        } else {
            (None, None)
        };
        let (Some(left), Some(right)) = (left, right) else {
            return Err(error("invalid BPE merge"));
        };
        if left.is_empty()
            || right.is_empty()
            || left.contains([' ', '\0'])
            || right.contains([' ', '\0'])
            || !vocab.contains_key(left)
            || !vocab.contains_key(right)
            || !vocab.contains_key(&format!("{left}{right}"))
        {
            return Err(error(format!(
                "unrepresentable BPE merge {left:?} + {right:?}"
            )));
        }
        merges.push(format!("{left} {right}"));
    }
    if let Some(g) = generation {
        for (key, expected) in [("bos_token_id", 2), ("pad_token_id", 0)] {
            if let Some(id) = g.get(key)
                && id.as_u64() != Some(expected)
            {
                return Err(error(format!("generation {key} mismatch")));
            }
        }
        if let Some(eos) = g.get("eos_token_id") {
            let eos = ids(eos, n_vocab, "generation eos")?;
            if !eos.contains(&1) || eos.iter().any(|id| !stop_ids.contains(id)) {
                return Err(error(
                    "generation EOS IDs cannot be represented by Gemma4 EOS/EOT/EOM",
                ));
            }
        }
        if let Some(suppress) = g.get("suppress_tokens").filter(|v| !v.is_null()) {
            if !suppress.is_array() {
                return Err(error("suppress_tokens must be an array"));
            }
            let suppress = ids(suppress, n_vocab, "suppressed")?;
            // The pinned loader specifically requires GGUF_TYPE_INT32 here.
            entries.push((
                "tokenizer.ggml.suppress_tokens".into(),
                G::I32s(suppress.into_iter().map(|n| n as i32).collect()),
            ));
        }
    }
    let template = match tc.get("chat_template") {
        None | Some(Value::Null) => CHAT_TEMPLATE,
        Some(Value::String(s)) if !s.trim().is_empty() => s,
        _ => return Err(error("chat_template must be a nonempty string")),
    };
    entries.extend([
        ("tokenizer.ggml.model".into(), G::String("gemma4".into())),
        ("tokenizer.ggml.pre".into(), G::String("gemma4".into())),
        (
            "tokenizer.ggml.scores".into(),
            G::F32s(vec![-1000.0; n_vocab as usize]),
        ),
        ("tokenizer.ggml.token_type".into(), G::I32s(types)),
        ("tokenizer.ggml.tokens".into(), G::Strings(tokens)),
        ("tokenizer.ggml.merges".into(), G::Strings(merges)),
        ("tokenizer.ggml.add_space_prefix".into(), G::Bool(false)),
        ("tokenizer.ggml.add_bos_token".into(), G::Bool(true)),
        ("tokenizer.ggml.add_eos_token".into(), G::Bool(false)),
        (
            "tokenizer.chat_template".into(),
            G::String(template.to_owned()),
        ),
    ]);
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Value {
        json!({
            "architectures":["Gemma4UnifiedForConditionalGeneration"],
            "model_type":"gemma4_unified", "tie_word_embeddings":true,
            "text_config": {
                "model_type":"gemma4_unified_text", "hidden_activation":"gelu_pytorch_tanh",
                "hidden_size":3840, "intermediate_size":15360, "num_hidden_layers":48,
                "num_attention_heads":16, "num_key_value_heads":8, "num_global_key_value_heads":1,
                "head_dim":256, "global_head_dim":512, "max_position_embeddings":262144,
                "vocab_size":262144, "sliding_window":1024, "rms_norm_eps":1e-6,
                "final_logit_softcapping":30.0, "attention_k_eq_v":true,
                "attention_bias":false, "enable_moe_block":false, "num_experts":null,
                "hidden_size_per_layer_input":0, "num_kv_shared_layers":0,
                "use_double_wide_mlp":false, "use_bidirectional_attention":"vision",
                "tie_word_embeddings":true, "pad_token_id":0, "eos_token_id":1, "bos_token_id":2,
                "layer_types":(0..48).map(|i| if i%6==5 {"full_attention"} else {"sliding_attention"}).collect::<Vec<_>>(),
                "rope_parameters": {
                    "full_attention":{"rope_type":"proportional","rope_theta":1000000.0,"partial_rotary_factor":0.25},
                    "sliding_attention":{"rope_type":"default","rope_theta":10000.0}
                }
            }
        })
    }

    fn tokenizer() -> (Value, Value, u32) {
        let mut words: Vec<String> = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "<mask>",
            "<|turn>",
            "<turn|>",
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            "<|\"|>",
            "<unused42>",
            "▁",
            "x",
            "▁x",
            "\n",
            "\n\n",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        words.extend((0..=255u8).map(|b| format!("<0x{b:02X}>")));
        let vocab: serde_json::Map<String, Value> = words
            .iter()
            .enumerate()
            .map(|(id, s)| (s.clone(), json!(id)))
            .collect();
        let added: Vec<Value> = words[..14]
            .iter()
            .enumerate()
            .map(|(id, s)| {
                json!({"id":id,"content":s,"special":true,"normalized":false,
                "lstrip":false,"rstrip":false,"single_word":false})
            })
            .collect();
        (
            json!({
                "model":{"type":"BPE","byte_fallback":true,"fuse_unk":true,
                    "unk_token":"<unk>","ignore_merges":false,"vocab":vocab,
                    "merges":[["▁","x"],["\n","\n"]]},
                "added_tokens":added,
                "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
                "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
                "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
                "post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],
                    "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}
            }),
            json!({"bos_token":"<bos>","eos_token":"<eos>","eot_token":"<turn|>"}),
            words.len() as u32,
        )
    }

    fn get<'a>(m: &'a [(String, G)], key: &str) -> &'a G {
        &m.iter().find(|(k, _)| k == key).unwrap().1
    }

    #[test]
    fn small_import_previews_publishes_and_never_overwrites_or_modifies_store() {
        use crate::config::ImportConfig;
        use sha2::{Digest, Sha256};
        use std::{fs, path::Path};
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("store");
        let manifests = root.join("manifests/registry.ollama.ai/library/tiny");
        fs::create_dir_all(&manifests).unwrap();
        fs::create_dir(root.join("blobs")).unwrap();
        let mut config = config();
        let (tokenizer, tc, vocab) = tokenizer();
        let t = &mut config["text_config"];
        for (key, number) in [
            ("hidden_size", 64),
            ("intermediate_size", 128),
            ("num_hidden_layers", 2),
            ("num_attention_heads", 2),
            ("num_key_value_heads", 1),
            ("num_global_key_value_heads", 1),
            ("head_dim", 16),
            ("global_head_dim", 32),
            ("max_position_embeddings", 256),
            ("vocab_size", vocab),
            ("sliding_window", 32),
        ] {
            t[key] = json!(number);
        }
        t["layer_types"] = json!(["sliding_attention", "full_attention"]);
        fn layer(root: &Path, name: &str, media: &str, bytes: Vec<u8>) -> Value {
            let digest = format!("{:x}", Sha256::digest(&bytes));
            fs::write(root.join("blobs").join(format!("sha256-{digest}")), &bytes).unwrap();
            json!({"name":name,"mediaType":media,"digest":format!("sha256:{digest}"),"size":bytes.len()})
        }
        let mut layers = Vec::new();
        for (name, value) in [
            ("config.json", config.clone()),
            ("tokenizer.json", tokenizer),
            ("tokenizer_config.json", tc),
        ] {
            layers.push(layer(
                &root,
                name,
                "application/vnd.ollama.image.json",
                serde_json::to_vec(&value).unwrap(),
            ));
        }
        let names = BTreeMap::from([
            ("attn_norm.weight", "input_layernorm.weight"),
            (
                "post_attention_norm.weight",
                "post_attention_layernorm.weight",
            ),
            ("ffn_norm.weight", "pre_feedforward_layernorm.weight"),
            ("post_ffw_norm.weight", "post_feedforward_layernorm.weight"),
            ("attn_q_norm.weight", "self_attn.q_norm.weight"),
            ("attn_k_norm.weight", "self_attn.k_norm.weight"),
            ("attn_q.weight", "self_attn.q_proj.weight"),
            ("attn_k.weight", "self_attn.k_proj.weight"),
            ("attn_v.weight", "self_attn.v_proj.weight"),
            ("attn_output.weight", "self_attn.o_proj.weight"),
            ("ffn_gate.weight", "mlp.gate_proj.weight"),
            ("ffn_up.weight", "mlp.up_proj.weight"),
            ("ffn_down.weight", "mlp.down_proj.weight"),
        ]);
        for (name, dimensions) in expected_tensors(&config).unwrap() {
            if name == "rope_freqs.weight" {
                continue;
            }
            let hf = match name.as_str() {
                "token_embd.weight" => "model.language_model.embed_tokens.weight".to_owned(),
                "output_norm.weight" => "model.language_model.norm.weight".to_owned(),
                _ => {
                    let rest = name.strip_prefix("blk.").unwrap();
                    let (index, suffix) = rest.split_once('.').unwrap();
                    format!("model.language_model.layers.{index}.{}", names[suffix])
                }
            };
            let elements = dimensions.iter().product::<u64>() as usize;
            let shape: Vec<_> = dimensions.into_iter().rev().collect();
            let mut header = serde_json::to_vec(
                &json!({hf.clone():{"dtype":"F32","shape":shape,"data_offsets":[0,elements*4]}}),
            )
            .unwrap();
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            for _ in 0..elements {
                bytes.extend_from_slice(&0.125f32.to_le_bytes());
            }
            layers.push(layer(
                &root,
                &hf,
                "application/vnd.ollama.image.tensor",
                bytes,
            ));
        }
        let manifest = manifests.join("latest");
        let source =
            json!({"schemaVersion":2,"runner":"mlx","format":"safetensors","layers":layers});
        let original = serde_json::to_vec(&source).unwrap();
        fs::write(&manifest, &original).unwrap();
        let config = ImportConfig {
            model: "tiny:latest".into(),
            ollama_dir: Some(root.clone()),
            output: temp.path().join("tiny.gguf"),
        };
        let preview = crate::import::preview(&config).unwrap();
        assert!(!config.output.exists());
        crate::import::run(config.clone()).unwrap();
        let bytes = fs::read(&config.output).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        assert_eq!(bytes.len() as u64, preview.output_bytes);
        assert_eq!(fs::read(&manifest).unwrap(), original);
        assert!(
            crate::import::run(config.clone())
                .unwrap_err()
                .contains("never overwrites")
        );
        assert_eq!(fs::read(&config.output).unwrap(), bytes);
        let mut forbidden = config.clone();
        forbidden.output = root.join("forbidden.gguf");
        assert!(
            crate::import::run(forbidden.clone())
                .unwrap_err()
                .contains("outside")
        );
        assert!(!forbidden.output.exists());
        let mut broken = source.clone();
        broken["layers"].as_array_mut().unwrap().pop();
        fs::write(&manifest, serde_json::to_vec(&broken).unwrap()).unwrap();
        let mut missing = config.clone();
        missing.output = temp.path().join("missing.gguf");
        assert!(
            crate::import::run(missing.clone())
                .unwrap_err()
                .contains("missing required")
        );
        assert!(!missing.output.exists());
        assert!(
            !fs::read_dir(temp.path()).unwrap().any(|e| e
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "part"))
        );
    }

    #[test]
    fn installed_dense_structure_and_rope() {
        let c = config();
        let parsed = validate(&c).unwrap();
        assert_eq!(parsed.swa.iter().filter(|s| !**s).count(), 8);
        let extra = extra_tensors(&c).unwrap();
        assert_eq!(extra[0].0, "rope_freqs.weight");
        assert_eq!(extra[0].1.len(), 256);
        assert!(extra[0].1[..64].iter().all(|f| *f == 1.0));
        assert!(extra[0].1[64..].iter().all(|f| *f == 1e30));
        let shapes = expected_tensors(&c).unwrap();
        assert_eq!(shapes["blk.0.attn_q.weight"], [3840, 4096]);
        assert_eq!(shapes["blk.0.attn_v.weight"], [3840, 2048]);
        assert_eq!(shapes["blk.5.attn_q.weight"], [3840, 8192]);
        assert_eq!(shapes["blk.5.attn_k.weight"], [3840, 512]);
        assert!(!shapes.contains_key("blk.5.attn_v.weight"));
        assert_eq!(shapes["blk.5.attn_q_norm.weight"], [512]);
        assert_eq!(shapes["blk.0.ffn_down.weight"], [15360, 3840]);
        assert_eq!(shapes.len(), 619);
    }

    #[test]
    fn rejects_unsupported_architecture_before_weights() {
        for (key, value) in [
            ("attention_bias", json!(true)),
            ("enable_moe_block", json!(true)),
            ("hidden_size_per_layer_input", json!(256)),
            ("num_kv_shared_layers", json!(1)),
            ("use_double_wide_mlp", json!(true)),
            ("num_experts", json!(8)),
            ("num_key_value_heads", json!(3)),
            ("global_head_dim", json!(511)),
            ("hidden_size", json!(0)),
            ("num_hidden_layers", json!(u64::MAX)),
            ("layer_types", json!(["full_attention"])),
            ("tie_word_embeddings", json!(false)),
            ("per_layer_config", json!({"5":{"head_dim":256}})),
            ("use_bidirectional_attention", json!(true)),
            ("rms_norm_eps", json!(-1)),
        ] {
            let mut c = config();
            c["text_config"][key] = value;
            assert!(validate(&c).is_err(), "accepted {key}");
        }
        let mut c = config();
        c["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] =
            json!(0.333);
        assert!(extra_tensors(&c).is_err());
        c["architectures"] = json!(["Gemma4UnifiedAssistantForCausalLM"]);
        assert!(expected_tensors(&c).is_err());
    }

    #[test]
    fn maps_only_known_tensors_and_nontext_prefixes() {
        for (hf, gguf) in [
            (
                "model.language_model.embed_tokens.weight",
                "token_embd.weight",
            ),
            ("model.language_model.norm.weight", "output_norm.weight"),
            (
                "model.language_model.layers.5.self_attn.q_proj.weight",
                "blk.5.attn_q.weight",
            ),
            (
                "model.layers.47.layer_scalar",
                "blk.47.layer_output_scale.weight",
            ),
            (
                "model.layers.2.post_attention_layernorm.weight",
                "blk.2.post_attention_norm.weight",
            ),
            (
                "model.layers.2.pre_feedforward_layernorm.weight",
                "blk.2.ffn_norm.weight",
            ),
            ("lm_head.weight", "output.weight"),
        ] {
            assert_eq!(tensor_name(hf).unwrap().as_deref(), Some(gguf));
        }
        for hf in [
            "model.embed_vision.embedding_projection.weight",
            "model.vision_embedder.patch_dense.weight",
            "model.embed_audio.embedding_projection.weight",
            "draft.model.layers.0.self_attn.q_proj.weight",
        ] {
            assert!(tensor_name(hf).unwrap().is_none());
        }
        for hf in [
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.bias",
            "model.unknown.weight",
            "model.language_model.layers.01.self_attn.q_proj.weight",
            "my_vision.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight_scale_2",
        ] {
            assert!(tensor_name(hf).is_err(), "silently accepted {hf}");
        }
    }

    #[test]
    fn tokenizer_keeps_raw_tokens_merges_types_and_stop_ids() {
        let (t, tc, n) = tokenizer();
        let g = json!({"eos_token_id":[1,6,11],"suppress_tokens":[14]});
        let m = tokenizer_metadata(&t, &tc, Some(&g), n).unwrap();
        assert!(matches!(
            get(&m, "tokenizer.ggml.padding_token_id"),
            G::U32(0)
        ));
        assert!(matches!(get(&m, "tokenizer.ggml.eot_token_id"), G::U32(6)));
        assert!(matches!(get(&m, "tokenizer.ggml.eom_token_id"), G::U32(11)));
        assert!(matches!(get(&m,"tokenizer.ggml.suppress_tokens"),G::I32s(v) if v == &[14]));
        assert!(matches!(get(&m,"tokenizer.ggml.merges"),G::Strings(v) if v == &["▁ x", "\n \n"]));
        assert!(
            matches!(get(&m,"tokenizer.ggml.tokens"),G::Strings(v) if v[17] == "▁x" && v[13] == "<|\"|>")
        );
        let G::I32s(types) = get(&m, "tokenizer.ggml.token_type") else {
            panic!()
        };
        assert_eq!(&types[..8], &[3, 3, 3, 2, 3, 3, 3, 4]);
        assert_eq!(types[14], 5);
        assert_eq!(types[15], 1);
        assert_eq!(types[20], 6);
        assert!(CHAT_TEMPLATE.contains("<|turn>model"));
        assert!(!CHAT_TEMPLATE.contains("<start_of_turn>"));
        let mut c = config();
        c["text_config"]["vocab_size"] = json!(n);
        let m = metadata(&c, &t, &tc, Some(&g)).unwrap();
        assert!(
            matches!(get(&m,"gemma4.attention.head_count_kv"),G::U32s(v) if v[..6] == [8,8,8,8,8,1])
        );
    }

    #[test]
    fn tokenizer_rejects_holes_collisions_pipeline_and_bad_stops() {
        let (t, tc, n) = tokenizer();
        let mut bad = t.clone();
        bad["model"]["vocab"]["x"] = json!(2);
        assert!(tokenizer_metadata(&bad, &tc, None, n).is_err());
        let mut bad = t.clone();
        bad["normalizer"]["content"] = json!(" ");
        assert!(tokenizer_metadata(&bad, &tc, None, n).is_err());
        let mut bad = t.clone();
        bad["added_tokens"][0]["lstrip"] = json!(true);
        assert!(tokenizer_metadata(&bad, &tc, None, n).is_err());
        let mut bad = t.clone();
        bad["model"]["merges"] = json!([["missing", "x"]]);
        assert!(tokenizer_metadata(&bad, &tc, None, n).is_err());
        assert!(tokenizer_metadata(&t, &tc, None, n + 1).is_err());
        assert!(tokenizer_metadata(&t, &tc, Some(&json!({"eos_token_id":[1,17]})), n).is_err());
        assert!(tokenizer_metadata(&t, &tc, Some(&json!({"suppress_tokens":[n]})), n).is_err());
    }

    #[test]
    fn reserved_aliases_are_unescaped_once_and_existing_template_is_preserved() {
        let (mut t, mut tc, n) = tokenizer();
        t["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":14,"content":"<alias\"x>","special":false}));
        tc["chat_template"] = json!("{{ '<|turn>user\\n' }}");
        let m = tokenizer_metadata(&t, &tc, None, n).unwrap();
        assert!(matches!(get(&m,"tokenizer.ggml.tokens"),G::Strings(v) if v[14] == "<alias\"x>"));
        assert!(matches!(get(&m,"tokenizer.ggml.token_type"),G::I32s(v) if v[14] == 4));
        assert!(
            matches!(get(&m,"tokenizer.chat_template"),G::String(v) if v == "{{ '<|turn>user\\n' }}")
        );
    }
}
