//! Explicit tokenizer profiles, not a best-effort tokenizer.json translation.
use super::{gguf::Value as G, read_json, tensors::open_regular};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::Path,
};

pub(super) const LLAMA_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
pub(super) const QWEN_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
fn error(s: impl std::fmt::Display) -> String {
    format!("HF tokenizer: {s}")
}
fn boolean(v: &Value, key: &str, default: bool) -> Result<bool, String> {
    match v.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| error(format!("{key} must be boolean"))),
    }
}
fn optional_json(root: &Path, name: &str, limit: u64) -> Result<Option<Value>, String> {
    let path = root.join(name);
    if path.try_exists().map_err(|e| e.to_string())? {
        read_json(&path, limit).map(Some)
    } else {
        Ok(None)
    }
}

pub(super) fn metadata(
    root: &Path,
    config: &Value,
    n: u32,
    qwen: bool,
) -> Result<Vec<(String, G)>, String> {
    let tc = optional_json(root, "tokenizer_config.json", 4 * 1024 * 1024)?
        .ok_or("HF import requires tokenizer_config.json")?;
    let generation = optional_json(root, "generation_config.json", 1024 * 1024)?;
    let tokenizer = optional_json(root, "tokenizer.json", 64 * 1024 * 1024)?;
    if !tc.is_object() || generation.as_ref().is_some_and(|v| !v.is_object()) {
        return Err(error("invalid tokenizer/generation config"));
    }
    if boolean(&tc, "split_special_tokens", false)? {
        return Err(error("split_special_tokens=true is unsupported"));
    }
    if let Some(t) = &tokenizer {
        for key in ["truncation", "padding"] {
            if t.get(key).is_some_and(|v| !v.is_null()) {
                return Err(error(format!("fixed tokenizer {key} is unsupported")));
            }
        }
        if t.get("added_tokens").is_some_and(|v| !v.is_array()) {
            return Err(error("added_tokens must be an array"));
        }
    }
    let mut tokens = vec![None; n as usize];
    let mut scores = vec![-1000.; n as usize];
    let mut types = vec![5i32; n as usize];
    let mut entries = Vec::new();
    let bpe;
    if root
        .join("tokenizer.model")
        .try_exists()
        .map_err(|e| e.to_string())?
    {
        if qwen {
            return Err(error(
                "Qwen2 requires the byte-level tokenizer.json profile",
            ));
        }
        let pieces = sentencepiece(&root.join("tokenizer.model"))?;
        if pieces.len() > n as usize {
            return Err(error("SentencePiece vocab exceeds model vocab_size"));
        }
        for (id, (text, score, kind)) in pieces.into_iter().enumerate() {
            tokens[id] = Some(text);
            scores[id] = score;
            types[id] = kind;
        }
        entries.push(("tokenizer.ggml.model".into(), G::String("llama".into())));
        entries.push(("tokenizer.ggml.add_space_prefix".into(), G::Bool(true)));
        entries.push((
            "tokenizer.ggml.remove_extra_whitespaces".into(),
            G::Bool(false),
        ));
        if boolean(&tc, "add_prefix_space", true)? != true || boolean(&tc, "legacy", true)? != true
        {
            return Err(error(
                "SentencePiece requires legacy=true and add_prefix_space=true",
            ));
        }
        // A fast-tokenizer override can change normalization or token ranks. Do
        // not quietly substitute tokenizer.model for a different tokenizer.json.
        if let Some(t) = &tokenizer {
            validate_spm_json(t, &tokens)?;
        }
        bpe = false;
    } else {
        let t = tokenizer
            .as_ref()
            .ok_or("HF import requires tokenizer.json or tokenizer.model")?;
        let pre = validate_bpe(t, qwen)?;
        let vocab = t["model"]["vocab"]
            .as_object()
            .ok_or_else(|| error("BPE vocab must be a token-ID map"))?;
        for (text, id) in vocab {
            let id = token_id(id, n)? as usize;
            if text.is_empty() || text.contains('\0') || tokens[id].replace(text.clone()).is_some()
            {
                return Err(error("invalid or duplicate BPE vocabulary ID"));
            }
            types[id] = 1;
        }
        // All byte symbols are necessary for the native BPE fallback.
        for byte in byte_symbols() {
            if !vocab.contains_key(&byte) {
                return Err(error(format!("missing BPE byte symbol {byte:?}")));
            }
        }
        let raw = t["model"]["merges"]
            .as_array()
            .filter(|v| v.len() <= 4_000_000)
            .ok_or_else(|| error("missing/oversized BPE merges"))?;
        let mut merges = Vec::with_capacity(raw.len());
        let mut seen = BTreeSet::new();
        for v in raw {
            let pair = if let Some(s) = v.as_str() {
                s.split_once(' ')
            } else {
                v.as_array()
                    .filter(|a| a.len() == 2)
                    .and_then(|a| Some((a[0].as_str()?, a[1].as_str()?)))
            };
            let (a, b) = pair.ok_or_else(|| error("invalid BPE merge"))?;
            if a.is_empty()
                || b.is_empty()
                || a.contains([' ', '\0'])
                || b.contains([' ', '\0'])
                || !vocab.contains_key(a)
                || !vocab.contains_key(b)
                || !vocab.contains_key(&format!("{a}{b}"))
                || !seen.insert((a, b))
            {
                return Err(error("invalid, duplicate, or unrepresentable BPE merge"));
            }
            merges.push(format!("{a} {b}"));
        }
        entries.extend([
            ("tokenizer.ggml.model".into(), G::String("gpt2".into())),
            ("tokenizer.ggml.pre".into(), G::String(pre.into())),
            ("tokenizer.ggml.merges".into(), G::Strings(merges)),
            ("tokenizer.ggml.add_space_prefix".into(), G::Bool(false)),
        ]);
        bpe = true;
    }
    let mut added = BTreeMap::new();
    if let Some(t) = &tokenizer {
        for a in t
            .get("added_tokens")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let id = token_id(&a["id"], n)?;
            if added.insert(id, a.clone()).is_some() {
                return Err(error("duplicate added-token ID"));
            }
        }
    }
    if let Some(decoder) = tc.get("added_tokens_decoder") {
        for (key, a) in decoder
            .as_object()
            .ok_or_else(|| error("invalid added_tokens_decoder"))?
        {
            let id: u32 = key.parse().map_err(|_| error("invalid added token ID"))?;
            if id >= n {
                return Err(error("added token ID exceeds vocab_size"));
            }
            if let Some(old) = added.get(&id) {
                if old["content"] != a["content"] || old["special"] != a["special"] {
                    return Err(error("conflicting added-token definitions"));
                }
                for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
                    if boolean(old, flag, false)? != boolean(a, flag, false)? {
                        return Err(error("conflicting added-token flags"));
                    }
                }
            } else {
                added.insert(id, a.clone());
            }
        }
    }
    for (id, a) in added {
        for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
            if boolean(&a, flag, false)? {
                return Err(error(format!("added-token {flag}=true is unsupported")));
            }
        }
        let text = a["content"]
            .as_str()
            .filter(|s| !s.is_empty() && !s.contains('\0'))
            .ok_or_else(|| error("invalid added token content"))?;
        if tokens[id as usize].as_ref().is_some_and(|old| old != text) {
            return Err(error("added token conflicts with vocabulary"));
        }
        tokens[id as usize] = Some(text.into());
        types[id as usize] = if boolean(&a, "special", false)? { 3 } else { 4 };
    }
    let mut lookup = BTreeMap::new();
    for (id, text) in tokens.iter_mut().enumerate() {
        let text = text.get_or_insert_with(|| format!("[PAD{id}]"));
        if lookup.insert(text.clone(), id as u32).is_some() {
            return Err(error("duplicate token text (including padding slots)"));
        }
    }
    let mut special = BTreeMap::new();
    for (hf, gguf) in [
        ("bos", "bos"),
        ("eos", "eos"),
        ("unk", "unknown"),
        ("pad", "padding"),
    ] {
        let text = match tc.get(format!("{hf}_token")) {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.as_str()),
            Some(v) => Some(
                v["content"]
                    .as_str()
                    .ok_or_else(|| error("invalid special token content"))?,
            ),
        };
        let from_text = text
            .map(|s| {
                lookup
                    .get(s)
                    .copied()
                    .ok_or_else(|| error(format!("missing special token {s:?}")))
            })
            .transpose()?;
        let mut id = from_text;
        for source in [Some(config), generation.as_ref()].into_iter().flatten() {
            if let Some(v) = source
                .get(format!("{hf}_token_id"))
                .filter(|v| !v.is_null())
            {
                let v = if hf == "eos" && v.is_array() {
                    v.as_array()
                        .and_then(|a| a.first())
                        .ok_or_else(|| error("empty EOS list"))?
                } else {
                    v
                };
                let candidate = token_id(v, n)?;
                // generation_config can prefer a different EOS from a known EOT/EOM;
                // the complete set is checked below, without discarding extra IDs.
                if hf != "eos" && id.is_some_and(|id| id != candidate) {
                    return Err(error("conflicting special token IDs"));
                }
                id = id.or(Some(candidate));
            }
        }
        if let Some(id) = id {
            if types[id as usize] == 5 {
                return Err(error(
                    "special token ID points to an unused vocabulary slot",
                ));
            }
            special.insert(gguf, id);
            types[id as usize] = if hf == "unk" { 2 } else { 3 };
        }
    }
    for (key, candidates) in [
        ("eot", &["<|eot_id|>", "<|im_end|>"][..]),
        ("eom", &["<|eom_id|>"][..]),
    ] {
        if let Some(id) = candidates.iter().find_map(|s| lookup.get(*s)).copied() {
            special.insert(key, id);
            types[id as usize] = 3;
        }
    }
    if !special.contains_key("eos") {
        return Err(error("EOS token ID is required"));
    }
    for source in [Some(config), generation.as_ref()].into_iter().flatten() {
        if let Some(eos) = source.get("eos_token_id").filter(|v| !v.is_null()) {
            let ids: Vec<_> = if let Some(a) = eos.as_array() {
                a.iter().collect()
            } else {
                vec![eos]
            };
            if ids.is_empty() {
                return Err(error("empty EOS IDs"));
            }
            for id in ids {
                let id = token_id(id, n)?;
                if !["eos", "eot", "eom"]
                    .iter()
                    .any(|key| special.get(key) == Some(&id))
                {
                    return Err(error("multiple EOS IDs must map to native EOS/EOT/EOM"));
                }
            }
        }
    }
    for (key, id) in &special {
        entries.push((format!("tokenizer.ggml.{key}_token_id"), G::U32(*id)));
    }
    let add_bos = boolean(&tc, "add_bos_token", !qwen)?;
    let add_eos = boolean(&tc, "add_eos_token", false)?;
    if (add_bos && !special.contains_key("bos")) || (add_eos && !special.contains_key("eos")) {
        return Err(error("automatic BOS/EOS requested without token"));
    }
    if let Some(t) = &tokenizer {
        validate_postprocessor(t, add_bos, add_eos, &special, &lookup)?;
    }
    entries.extend([
        ("tokenizer.ggml.add_bos_token".into(), G::Bool(add_bos)),
        ("tokenizer.ggml.add_eos_token".into(), G::Bool(add_eos)),
        (
            "tokenizer.ggml.tokens".into(),
            G::Strings(tokens.into_iter().map(Option::unwrap).collect()),
        ),
        ("tokenizer.ggml.token_type".into(), G::I32s(types)),
    ]);
    if !bpe {
        entries.push(("tokenizer.ggml.scores".into(), G::F32s(scores)));
    }
    let configured_template = tc
        .get("chat_template")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_str()
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| error("chat_template must be a single nonempty string"))
        })
        .transpose()?;
    let template_path = root.join("chat_template.jinja");
    let file_template = if template_path.try_exists().map_err(|e| e.to_string())? {
        let mut bytes = Vec::new();
        open_regular(&template_path)?
            .take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 4 * 1024 * 1024 {
            return Err(error("chat_template.jinja exceeds 4 MiB"));
        }
        let text =
            String::from_utf8(bytes).map_err(|_| error("chat_template.jinja must be UTF-8"))?;
        if text.trim().is_empty() {
            return Err(error("chat_template.jinja must be nonempty"));
        }
        if configured_template.is_some_and(|t| t != text) {
            return Err(error(
                "conflicting tokenizer_config.json chat_template and chat_template.jinja",
            ));
        }
        Some(text)
    } else {
        None
    };
    if let Some(template) = file_template.as_deref().or(configured_template) {
        entries.push(("tokenizer.chat_template".into(), G::String(template.into())));
    }
    if let Some(g) = &generation {
        for key in [
            "suppress_tokens",
            "begin_suppress_tokens",
            "forced_bos_token_id",
            "forced_eos_token_id",
        ] {
            if g.get(key)
                .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
            {
                return Err(error(format!("generation {key} is unsupported")));
            }
        }
    }
    Ok(entries)
}
fn validate_postprocessor(
    t: &Value,
    add_bos: bool,
    add_eos: bool,
    special: &BTreeMap<&str, u32>,
    lookup: &BTreeMap<String, u32>,
) -> Result<(), String> {
    let p = &t["post_processor"];
    if p.is_null() || p["type"] == "ByteLevel" {
        return Ok(());
    }
    if p["type"] != "TemplateProcessing" {
        return Err(error("unsupported tokenizer post_processor"));
    }
    let single = p["single"]
        .as_array()
        .ok_or_else(|| error("post_processor.single must be a template array"))?;
    let mut expected = Vec::new();
    if add_bos {
        expected.push(Some(special["bos"]));
    }
    expected.push(None);
    if add_eos {
        expected.push(Some(special["eos"]));
    }
    if single.len() != expected.len() {
        return Err(error(
            "post_processor disagrees with automatic BOS/EOS flags",
        ));
    }
    for (item, expected) in single.iter().zip(expected) {
        if let Some(id) = expected {
            let part = &item["SpecialToken"];
            let text = part["id"]
                .as_str()
                .ok_or_else(|| error("invalid post_processor special token"))?;
            if lookup.get(text) != Some(&id) || part["type_id"].as_u64() != Some(0) {
                return Err(error("post_processor special token mismatch"));
            }
            let declaration = &p["special_tokens"][text];
            if declaration["ids"] != serde_json::json!([id])
                || declaration["tokens"] != serde_json::json!([text])
            {
                return Err(error(
                    "post_processor special token expansion is unsupported",
                ));
            }
        } else if item["Sequence"]["id"] != "A" || item["Sequence"]["type_id"].as_u64() != Some(0) {
            return Err(error("post_processor must contain exactly one sequence A"));
        }
    }
    Ok(())
}
fn token_id(v: &Value, n: u32) -> Result<u32, String> {
    v.as_u64()
        .filter(|id| *id < u64::from(n))
        .map(|id| id as u32)
        .ok_or_else(|| error("token ID outside model vocab_size"))
}

fn validate_bpe(t: &Value, qwen: bool) -> Result<&'static str, String> {
    if t["model"]["type"] != "BPE" || !t["normalizer"].is_null() {
        return Err(error(
            "requires BPE with no normalization; NFC and other normalization pipelines are not implemented by the pinned native BPE tokenizer",
        ));
    }
    for key in [
        "dropout",
        "unk_token",
        "continuing_subword_prefix",
        "end_of_word_suffix",
    ] {
        if t["model"]
            .get(key)
            .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
        {
            return Err(error(format!("unsupported BPE {key}")));
        }
    }
    if boolean(&t["model"], "byte_fallback", false)? || boolean(&t["model"], "fuse_unk", false)? {
        return Err(error("unsupported BPE fallback options"));
    }
    if t["decoder"]["type"] != "ByteLevel" {
        return Err(error("BPE decoder must be ByteLevel"));
    }
    let pre = &t["pre_tokenizer"];
    let sequence = pre["pretokenizers"]
        .as_array()
        .filter(|a| pre["type"] == "Sequence" && a.len() == 2)
        .ok_or_else(|| error("requires Split + ByteLevel pre-tokenizer"))?;
    let split = &sequence[0];
    let byte = &sequence[1];
    let regex = if qwen { QWEN_SPLIT } else { LLAMA_SPLIT };
    if split["type"] != "Split"
        || split["pattern"]["Regex"] != regex
        || split["behavior"] != "Isolated"
        || boolean(split, "invert", false)?
        || byte["type"] != "ByteLevel"
        || boolean(byte, "add_prefix_space", false)?
        || boolean(byte, "use_regex", true)?
    {
        return Err(error(
            "unrecognized pre-tokenizer; only exact Llama3/Qwen2 regex profiles are supported",
        ));
    }
    let ignore_merges = boolean(&t["model"], "ignore_merges", false)?;
    if ignore_merges == qwen {
        return Err(error(if qwen {
            "Qwen2 requires ignore_merges=false"
        } else {
            "the native Llama3 profile requires ignore_merges=true"
        }));
    }
    Ok(if qwen { "qwen2" } else { "llama-bpe" })
}
fn validate_spm_json(t: &Value, tokens: &[Option<String>]) -> Result<(), String> {
    if t["model"]["type"] != "BPE" || !boolean(&t["model"], "byte_fallback", false)? {
        return Err(error("tokenizer.json conflicts with SentencePiece BPE"));
    }
    let expected = serde_json::json!({"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]});
    if t["normalizer"] != expected || !t["pre_tokenizer"].is_null() {
        return Err(error("unsupported SentencePiece fast-tokenizer pipeline"));
    }
    let decoder = serde_json::json!({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]});
    if t["decoder"] != decoder {
        return Err(error("unsupported SentencePiece decoder"));
    }
    for (text, id) in t["model"]["vocab"]
        .as_object()
        .ok_or_else(|| error("missing SentencePiece fast vocabulary"))?
    {
        let id = token_id(id, tokens.len() as u32)? as usize;
        if tokens[id].as_deref() != Some(text) {
            return Err(error(
                "tokenizer.model and tokenizer.json vocabulary mismatch",
            ));
        }
    }
    Ok(())
}

pub(super) fn byte_symbols() -> Vec<String> {
    let mut extra = 0u32;
    (0..=255u32)
        .map(|b| {
            let cp = if (33..=126).contains(&b)
                || (161..=172).contains(&b)
                || (174..=255).contains(&b)
            {
                b
            } else {
                let cp = 256 + extra;
                extra += 1;
                cp
            };
            char::from_u32(cp).unwrap().to_string()
        })
        .collect()
}

// Minimal bounded protobuf wire reader. No SentencePiece dependency or external
// process: only the BPE/identity/byte-fallback profile used by Llama 1/2 and
// early Mistral/Mixtral is accepted.
fn varint(bytes: &mut &[u8]) -> Result<u64, String> {
    let mut out = 0;
    for shift in (0..70).step_by(7) {
        let b = *bytes
            .first()
            .ok_or_else(|| error("truncated SentencePiece varint"))?;
        *bytes = &bytes[1..];
        if shift == 63 && b > 1 {
            return Err(error("SentencePiece varint overflow"));
        }
        out |= u64::from(b & 127) << shift;
        if b < 128 {
            return Ok(out);
        }
    }
    Err(error("invalid SentencePiece varint"))
}
fn fields(mut bytes: &[u8]) -> Result<Vec<(u64, Vec<u8>)>, String> {
    let mut result = Vec::new();
    while !bytes.is_empty() {
        if result.len() >= 1_048_576 {
            return Err(error("excessive protobuf fields"));
        }
        let tag = varint(&mut bytes)?;
        let value = match tag & 7 {
            0 => varint(&mut bytes)?.to_le_bytes().to_vec(),
            1 | 5 => {
                let n = if tag & 7 == 1 { 8 } else { 4 };
                let v = bytes
                    .get(..n)
                    .ok_or_else(|| error("truncated fixed protobuf field"))?
                    .to_vec();
                bytes = &bytes[n..];
                v
            }
            2 => {
                let n = usize::try_from(varint(&mut bytes)?)
                    .map_err(|_| error("protobuf length overflow"))?;
                let v = bytes
                    .get(..n)
                    .ok_or_else(|| error("truncated protobuf field"))?
                    .to_vec();
                bytes = &bytes[n..];
                v
            }
            _ => return Err(error("unsupported protobuf wire type")),
        };
        if tag >> 3 == 0 {
            return Err(error("invalid protobuf tag"));
        }
        result.push((tag, value));
    }
    Ok(result)
}
fn scalar(fields: &[(u64, Vec<u8>)], tag: u64, default: u64) -> Result<u64, String> {
    if fields.iter().filter(|(t, _)| *t >> 3 == tag >> 3).count() > 1 {
        return Err(error("duplicate SentencePiece scalar field"));
    }
    match fields.iter().find(|(t, _)| *t == tag) {
        None => Ok(default),
        Some((_, b)) => b
            .as_slice()
            .try_into()
            .map(u64::from_le_bytes)
            .map_err(|_| error("invalid protobuf scalar")),
    }
}
fn sentencepiece(path: &Path) -> Result<Vec<(String, f32, i32)>, String> {
    let mut bytes = Vec::new();
    open_regular(path)?
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(error("SentencePiece model exceeds 64 MiB"));
    }
    let model = fields(&bytes)?;
    let trainer = model
        .iter()
        .find(|(tag, _)| *tag == 18)
        .ok_or_else(|| error("missing SentencePiece trainer"))?;
    let trainer = fields(&trainer.1)?;
    if scalar(&trainer, 24, 1)? != 2 || scalar(&trainer, 280, 0)? != 1 {
        return Err(error(
            "only SentencePiece BPE with byte_fallback is supported",
        ));
    }
    let normalizer = model
        .iter()
        .find(|(tag, _)| *tag == 26)
        .ok_or_else(|| error("missing SentencePiece normalizer"))?;
    let norm = fields(&normalizer.1)?;
    if norm
        .iter()
        .find(|(tag, _)| *tag == 10)
        .map(|(_, v)| v.as_slice())
        != Some(b"identity")
        || norm.iter().any(|(tag, v)| *tag == 18 && !v.is_empty())
        || scalar(&norm, 24, 1)? != 1
        || scalar(&norm, 32, 1)? != 0
        || scalar(&norm, 40, 1)? != 1
    {
        return Err(error(
            "only identity SentencePiece normalization with dummy prefix and preserved whitespace is supported",
        ));
    }
    let mut result = Vec::new();
    let mut texts = BTreeSet::new();
    for (_, piece) in model.iter().filter(|(tag, _)| *tag == 10) {
        let p = fields(piece)?;
        let text = p
            .iter()
            .find(|(tag, _)| *tag == 10)
            .ok_or_else(|| error("missing SentencePiece token"))?;
        let text =
            String::from_utf8(text.1.clone()).map_err(|_| error("non-UTF8 SentencePiece token"))?;
        let score = p
            .iter()
            .find(|(tag, _)| *tag == 21)
            .map(|(_, b)| b.as_slice().try_into().map(f32::from_le_bytes))
            .transpose()
            .map_err(|_| error("invalid SentencePiece score"))?
            .unwrap_or(0.);
        let kind = scalar(&p, 24, 1)?;
        if text.is_empty()
            || text.contains('\0')
            || !score.is_finite()
            || !(1..=6).contains(&kind)
            || !texts.insert(text.clone())
            || result.len() >= 1_048_576
        {
            return Err(error("invalid SentencePiece vocabulary"));
        }
        result.push((text, score, kind as i32));
    }
    let byte_tokens: BTreeSet<_> = result
        .iter()
        .filter(|(_, _, kind)| *kind == 6)
        .map(|(text, _, _)| text.as_str())
        .collect();
    for b in 0..=255 {
        if !byte_tokens.contains(format!("<0x{b:02X}>").as_str()) {
            return Err(error("incomplete SentencePiece byte fallback"));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    fn var(mut n: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let b = (n & 127) as u8;
            n >>= 7;
            bytes.push(if n == 0 { b } else { b | 128 });
            if n == 0 {
                break;
            }
        }
        bytes
    }
    fn field(tag: u64, data: &[u8]) -> Vec<u8> {
        let mut bytes = var(tag);
        if tag & 7 == 2 {
            bytes.extend(var(data.len() as u64));
        }
        bytes.extend(data);
        bytes
    }
    fn spm(root: &Path) -> Value {
        let mut bytes = Vec::new();
        let mut vocab = Vec::new();
        for (text, kind) in [
            ("<unk>".to_owned(), 2),
            ("<s>".into(), 3),
            ("</s>".into(), 3),
            ("▁".into(), 1),
        ]
        .into_iter()
        .chain((0..=255).map(|n| (format!("<0x{n:02X}>"), 6)))
        {
            let score = -(vocab.len() as f32);
            let mut piece = field(10, text.as_bytes());
            piece.extend(field(21, &score.to_le_bytes()));
            piece.extend(field(24, &var(kind)));
            bytes.extend(field(10, &piece));
            vocab.push(json!([text, score]));
        }
        let mut trainer = field(24, &var(2));
        trainer.extend(field(280, &var(1)));
        bytes.extend(field(18, &trainer));
        let mut norm = field(10, b"identity");
        norm.extend(field(32, &var(0)));
        bytes.extend(field(26, &norm));
        fs::write(root.join("tokenizer.model"), bytes).unwrap();
        let tc = json!({"bos_token":"<s>","eos_token":"</s>","unk_token":"<unk>","add_bos_token":true,"add_eos_token":false});
        fs::write(
            root.join("tokenizer_config.json"),
            serde_json::to_vec(&tc).unwrap(),
        )
        .unwrap();
        json!({"vocab_size":262,"bos_token_id":1,"eos_token_id":2})
    }
    #[test]
    fn sentencepiece_preserves_scores_byte_types_padding_and_special_ids() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let c = spm(root);
        let m: BTreeMap<_, _> = metadata(root, &c, 262, false)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(m["tokenizer.ggml.model"], G::String("llama".into()));
        assert_eq!(m["tokenizer.ggml.bos_token_id"], G::U32(1));
        assert_eq!(m["tokenizer.ggml.eos_token_id"], G::U32(2));
        assert_eq!(m["tokenizer.ggml.unknown_token_id"], G::U32(0));
        let G::F32s(scores) = &m["tokenizer.ggml.scores"] else {
            panic!()
        };
        assert_eq!(&scores[..4], &[0., -1., -2., -3.]);
        let G::I32s(types) = &m["tokenizer.ggml.token_type"] else {
            panic!()
        };
        assert_eq!(&types[..5], &[2, 3, 3, 1, 6]);
        assert_eq!(types[261], 5);
        let G::Strings(tokens) = &m["tokenizer.ggml.tokens"] else {
            panic!()
        };
        assert_eq!(tokens[4], "<0x00>");
        assert_eq!(tokens[259], "<0xFF>");
        assert_eq!(tokens[261], "[PAD261]");
        assert!(metadata(root, &c, 262, true).is_err());
        let path = root.join("tokenizer.model");
        let original = fs::read(&path).unwrap();
        for bad in [vec![10, 255], vec![0], vec![255; 11]] {
            fs::write(&path, bad).unwrap();
            assert!(sentencepiece(&path).is_err());
        }
        fs::write(path, original).unwrap();
        let mut tc: Value = read_json(&root.join("tokenizer_config.json"), 4096).unwrap();
        tc["legacy"] = json!(false);
        fs::write(
            root.join("tokenizer_config.json"),
            serde_json::to_vec(&tc).unwrap(),
        )
        .unwrap();
        assert!(metadata(root, &c, 262, false).is_err());
    }
    #[test]
    fn standalone_chat_template_is_preserved_and_conflicts_fail() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let c = spm(root);
        let text = "{% for m in messages %}{{ m['content'] }}{% endfor %}";
        fs::write(root.join("chat_template.jinja"), text).unwrap();
        let m: BTreeMap<_, _> = metadata(root, &c, 262, false)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(m["tokenizer.chat_template"], G::String(text.into()));
        let path = root.join("tokenizer_config.json");
        let mut tc = read_json(&path, 4096).unwrap();
        tc["chat_template"] = json!("different");
        fs::write(path, serde_json::to_vec(&tc).unwrap()).unwrap();
        assert!(metadata(root, &c, 262, false).is_err());
    }
    #[test]
    fn exact_bpe_profiles_and_postprocessor_are_required() {
        let mut t = json!({"model":{"type":"BPE","ignore_merges":true},"normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":LLAMA_SPLIT},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},"decoder":{"type":"ByteLevel"}});
        assert_eq!(validate_bpe(&t, false).unwrap(), "llama-bpe");
        assert!(validate_bpe(&t, true).is_err());
        t["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!(QWEN_SPLIT);
        t["model"]["ignore_merges"] = json!(false);
        assert_eq!(validate_bpe(&t, true).unwrap(), "qwen2");
        for (key, value) in [
            ("normalizer", json!({"type":"NFC"})),
            ("decoder", json!({"type":"WordPiece"})),
            ("pre_tokenizer", json!({"type":"Whitespace"})),
        ] {
            let mut bad = t.clone();
            bad[key] = value;
            assert!(validate_bpe(&bad, true).is_err());
        }
        let lookup = BTreeMap::from([("<s>".into(), 1), ("</s>".into(), 2)]);
        let special = BTreeMap::from([("bos", 1), ("eos", 2)]);
        t["post_processor"] = json!({"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}}});
        validate_postprocessor(&t, true, false, &special, &lookup).unwrap();
        assert!(validate_postprocessor(&t, false, false, &special, &lookup).is_err());
        t["post_processor"]["special_tokens"]["<s>"]["ids"] = json!([1, 2]);
        assert!(validate_postprocessor(&t, true, false, &special, &lookup).is_err());
    }
    #[test]
    fn unsupported_generation_flags_and_added_token_semantics_are_not_lost() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let c = spm(root);
        for g in [
            json!({"eos_token_id":[2,3]}),
            json!({"forced_bos_token_id":1}),
            json!({"suppress_tokens":[4]}),
        ] {
            fs::write(
                root.join("generation_config.json"),
                serde_json::to_vec(&g).unwrap(),
            )
            .unwrap();
            assert!(metadata(root, &c, 262, false).is_err());
        }
        fs::remove_file(root.join("generation_config.json")).unwrap();
        let path = root.join("tokenizer_config.json");
        let tc = read_json(&path, 4096).unwrap();
        for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
            let mut value = tc.clone();
            value["added_tokens_decoder"] = json!({"260":{"content":"<extra>","special":true}});
            value["added_tokens_decoder"]["260"][flag] = json!(true);
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(metadata(root, &c, 262, false).is_err());
        }
    }
}
