//! Shared validation and wire diagnostics for real local OpenAI text features.
use serde_json::{Value, json};

use super::{
    generation_controls::{GenerationControls, diagnostic_bytes},
    inference::{Generation, Options, Prompt, TokenLogprob},
    server::ApiError,
    tools::{self, bounded_text, fields, string},
};

pub(super) fn controls(body: &Value) -> Result<GenerationControls, ApiError> {
    GenerationControls::from_openai(body).map_err(Into::into)
}

// Retain the shared entry point used by the protocol adapters; it now validates
// multiple native choices rather than restricting the request to one choice.
pub(super) fn single_choice(body: &Value, chat: bool) -> Result<(), ApiError> {
    Candidates::parse(body, chat)?;
    if let Some(value) = body.get("logprobs").filter(|v| !v.is_null()) {
        let valid = if chat {
            value.is_boolean()
        } else {
            value.as_u64().is_some_and(|n| n <= 5)
        };
        if !valid {
            return Err(ApiError::invalid(if chat {
                "Chat logprobs must be boolean"
            } else {
                "Completions logprobs must be an integer between 0 and 5"
            }));
        }
    }
    controls(body)?;
    if let Some(user) = body.get("user").filter(|v| !v.is_null()) {
        bounded_text(string(user)?, 512)?;
    }
    if let Some(modalities) = body.get("modalities").filter(|v| !v.is_null())
        && modalities != &json!(["text"])
    {
        return Err(ApiError::invalid("only text output modality is supported"));
    }
    if body.get("audio").is_some_and(|v| !v.is_null()) {
        return Err(ApiError::invalid(
            "OpenAI audio output is not implemented on text generation routes",
        ));
    }
    Ok(())
}

// Bounds include every sampled candidate, including candidates hidden by best_of.
pub(super) const MAX_OUTPUT_TOKENS: usize = 262_144;
pub(super) const MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) struct Candidates {
    pub n: usize,
    pub best_of: usize,
}
impl Candidates {
    pub fn parse(body: &Value, chat: bool) -> Result<Self, ApiError> {
        let count = |key: &str, default| {
            body.get(key)
                .filter(|v| !v.is_null())
                .map_or(Ok(default), |v| {
                    v.as_u64()
                        .filter(|n| (1..=128).contains(n))
                        .map(|n| n as usize)
                        .ok_or_else(|| {
                            ApiError::invalid(format!("{key} must be an integer between 1 and 128"))
                        })
                })
        };
        let n = count("n", 1)?;
        let best_of = if chat { n } else { count("best_of", n)? };
        if best_of < n {
            return Err(ApiError::invalid("best_of must be at least n"));
        }
        Ok(Self { n, best_of })
    }
    pub fn validate(
        self,
        inputs: usize,
        options: &Options,
        diagnostics: Option<usize>,
    ) -> Result<usize, ApiError> {
        let work = inputs
            .checked_mul(self.best_of)
            .filter(|n| (1..=128).contains(n))
            .ok_or_else(|| {
                resource_error("prompt count times candidate count must be at most 128")
            })?;
        let tokens = work.saturating_mul(options.max_tokens as usize);
        if tokens > MAX_OUTPUT_TOKENS
            || tokens.saturating_mul(diagnostics.map_or(1, |n| n + 1)) > 1_048_576
        {
            return Err(resource_error(
                "aggregate candidate output exceeds 262144 tokens or 1048576 diagnostic token records; lower max_tokens, n, best_of or logprobs",
            ));
        }
        if work > 1 && options.until_context {
            return Err(resource_error(
                "until-context generation is limited to one candidate",
            ));
        }
        Ok(work)
    }
    pub fn select(self, results: &[Generation]) -> Result<Vec<usize>, ApiError> {
        if results.is_empty() || results.len() % self.best_of != 0 {
            return Err(native_error(
                "native batch returned an unexpected candidate count",
            ));
        }
        let mut selected = Vec::new();
        for start in (0..results.len()).step_by(self.best_of) {
            let mut indexes: Vec<_> = (start..start + self.best_of).collect();
            if self.best_of > self.n {
                for &index in &indexes {
                    if !results[index]
                        .mean_logprob
                        .is_some_and(|score| score.is_finite() && score <= 0.0)
                    {
                        return Err(native_error(
                            "best_of requires a finite native mean token log probability for every candidate",
                        ));
                    }
                }
                // Stable sort preserves sample order when scores tie.
                indexes.sort_by(|a, b| {
                    results[*b]
                        .mean_logprob
                        .unwrap()
                        .partial_cmp(&results[*a].mean_logprob.unwrap())
                        .unwrap()
                });
            }
            selected.extend(indexes.into_iter().take(self.n));
        }
        Ok(selected)
    }
}
pub(super) fn resource_error(message: &str) -> ApiError {
    ApiError::new(
        axum::http::StatusCode::BAD_REQUEST,
        "resource_limit_exceeded",
        message,
    )
}
pub(super) fn native_error(message: &str) -> ApiError {
    ApiError::new(
        axum::http::StatusCode::BAD_GATEWAY,
        "invalid_native_output",
        message,
    )
}

/// Candidate zero retains the supplied seed; the odd stride is bijective in u32.
pub(super) fn candidate_options(options: &Options, index: usize) -> Options {
    let mut options = options.clone();
    options.seed = options
        .seed
        .map(|seed| seed.wrapping_add((index as u32).wrapping_mul(0x9e37_79b9)));
    options
}

pub(super) fn prompts(value: &Value) -> Result<Vec<Prompt>, ApiError> {
    fn text(value: &Value) -> Result<Prompt, ApiError> {
        let text = value
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ApiError::invalid("prompt strings must be nonempty"))?;
        Ok(Prompt::Text(text.to_owned()))
    }
    fn tokens(values: &[Value]) -> Result<Prompt, ApiError> {
        if values.is_empty() || values.len() > 131_072 {
            return Err(resource_error(
                "token prompts must contain 1..131072 token IDs",
            ));
        }
        let ids = values
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| i32::try_from(n).ok())
                    .ok_or_else(|| {
                        ApiError::invalid("prompt token IDs must be nonnegative i32 integers")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Prompt::Tokens(ids))
    }
    if value.is_string() {
        return Ok(vec![text(value)?]);
    }
    let values = value.as_array().filter(|v| !v.is_empty()).ok_or_else(|| {
        ApiError::invalid(
            "prompt must be a string, string array, token-ID array or array of token-ID arrays",
        )
    })?;
    if values[0].is_number() {
        return Ok(vec![tokens(values)?]);
    }
    if values.len() > 128 {
        return Err(resource_error("at most 128 prompts are supported"));
    }
    values
        .iter()
        .map(|v| {
            if values[0].is_string() {
                text(v)
            } else {
                tokens(
                    v.as_array().ok_or_else(|| {
                        ApiError::invalid("prompt arrays must have a uniform type")
                    })?,
                )
            }
        })
        .collect()
}

pub(super) fn validate_prompt_storage(
    prompts: &[Prompt],
    copies: usize,
) -> Result<(), super::inference::InferenceError> {
    fn bytes(prompt: &Prompt) -> usize {
        match prompt {
            Prompt::Text(text) => text.len(),
            Prompt::Tokens(tokens) => tokens.len().saturating_mul(4),
            Prompt::Chat(messages) => messages.iter().fold(0usize, |total, message| {
                message.images.iter().fold(
                    total.saturating_add(message.content.len()),
                    |total, image| total.saturating_add(image.len()),
                )
            }),
            Prompt::Constrained { prompt, schema } => {
                bytes(prompt).saturating_add(schema.to_string().len())
            }
            Prompt::Controls { prompt, .. } | Prompt::RequestSystem { prompt, .. } => bytes(prompt),
        }
    }
    let total = prompts.iter().fold(0usize, |total, prompt| {
        total.saturating_add(bytes(prompt).saturating_mul(copies))
    });
    if total > MAX_OUTPUT_BYTES {
        return Err(super::inference::InferenceError::Invalid("resource limit: aggregate prepared candidate prompts exceed 64 MiB; lower n, best_of or prompt size".into()));
    }
    Ok(())
}

pub(super) fn output_bytes(result: &Generation) -> usize {
    let mut bytes = result.text.len();
    if let Some(records) = &result.logprobs {
        for token in records {
            bytes = bytes.saturating_add(diagnostic_bytes(token.token.len(), token.bytes.len()));
            for other in &token.top_logprobs {
                bytes =
                    bytes.saturating_add(diagnostic_bytes(other.token.len(), other.bytes.len()));
            }
        }
    }
    bytes
}

#[derive(Default)]
pub(super) struct Usage {
    pub prompt: usize,
    pub completion: usize,
}
impl Usage {
    pub fn add(&mut self, result: &Generation, first_for_input: bool) {
        if first_for_input {
            self.prompt += result.prompt_tokens;
        }
        self.completion += result.completion_tokens;
    }
    pub fn value(&self) -> Value {
        json!({"prompt_tokens":self.prompt,"completion_tokens":self.completion,"total_tokens":self.prompt+self.completion})
    }
}

// Native records omit suppressed/partial stop tokens. Validate alignment again at
// the public boundary: never expose a token absent from the visible text.
fn aligned(result: &Generation) -> Result<Vec<(&TokenLogprob, usize)>, ApiError> {
    let records = result
        .logprobs
        .as_ref()
        .ok_or_else(|| native_error("requested native token log probabilities are missing"))?;
    let mut cursor = 0usize;
    let mut previous_boundary = 0;
    let mut character_offset = 0;
    let mut aligned = Vec::new();
    for token in records {
        if token.bytes.is_empty()
            || !token.logprob.is_finite()
            || token.logprob > 0.0
            || token
                .top_logprobs
                .iter()
                .any(|other| !other.logprob.is_finite() || other.logprob > 0.0)
        {
            return Err(native_error("invalid native token probability record"));
        }
        let end = cursor.saturating_add(token.bytes.len());
        if result.text.as_bytes().get(cursor..end) != Some(token.bytes.as_slice()) {
            return Err(native_error(
                "native token probabilities do not align with visible output",
            ));
        }
        // Two native pieces can share one Unicode character. Legacy offsets
        // address the containing character; Chat preserves the exact raw bytes.
        let mut boundary = cursor;
        while !result.text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        character_offset += result.text[previous_boundary..boundary].chars().count();
        aligned.push((token, character_offset));
        previous_boundary = boundary;
        cursor = end;
    }
    Ok(aligned)
}
pub(super) fn chat_logprobs(result: &Generation, enabled: bool) -> Result<Value, ApiError> {
    if !enabled {
        return Ok(Value::Null);
    }
    let content = aligned(result)?.into_iter().map(|(token, _)| {
        json!({"token":token.token,"bytes":token.bytes,"logprob":token.logprob,"top_logprobs":token.top_logprobs.iter().map(|other| json!({"token":other.token,"bytes":other.bytes,"logprob":other.logprob})).collect::<Vec<_>>()})
    }).collect::<Vec<_>>();
    Ok(json!({"content":content,"refusal":null}))
}
pub(super) fn legacy_logprobs(result: &Generation, enabled: bool) -> Result<Value, ApiError> {
    if !enabled {
        return Ok(Value::Null);
    }
    let records = aligned(result)?;
    Ok(json!({
        "tokens":records.iter().map(|(t,_)| &t.token).collect::<Vec<_>>(),
        "token_logprobs":records.iter().map(|(t,_)| t.logprob).collect::<Vec<_>>(),
        "top_logprobs":records.iter().map(|(t,_)| {
            let mut map = serde_json::Map::new();
            for other in &t.top_logprobs {
                // Legacy maps cannot distinguish native IDs with identical
                // (or identical lossy UTF-8) labels. Keep the highest score.
                if map.get(&other.token).and_then(Value::as_f64).is_none_or(|existing| other.logprob > existing) {
                    map.insert(other.token.clone(), json!(other.logprob));
                }
            }
            map
        }).collect::<Vec<_>>(),
        "text_offset":records.iter().map(|(_,offset)| offset).collect::<Vec<_>>()
    }))
}

pub(super) fn stream_usage(body: &Value, stream: bool) -> Result<bool, ApiError> {
    let Some(options) = body.get("stream_options").filter(|v| !v.is_null()) else {
        return Ok(false);
    };
    fields(options, &["include_usage", "include_obfuscation"])?;
    let boolean = |key| {
        options
            .get(key)
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| ApiError::invalid("stream options must be booleans"))
            })
            .unwrap_or(Ok(false))
    };
    if !stream || boolean("include_obfuscation")? {
        return Err(ApiError::invalid(
            "stream_options requires stream:true; obfuscation is not implemented",
        ));
    }
    boolean("include_usage")
}

pub(super) fn response_schema(value: Option<&Value>) -> Result<Option<Value>, ApiError> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let kind = string(&value["type"])?;
    let schema = match kind {
        "text" => {
            fields(value, &["type"])?;
            return Ok(None);
        }
        "json_object" => {
            fields(value, &["type"])?;
            json!({"type":"object"})
        }
        "json_schema" => {
            fields(value, &["type", "json_schema"])?;
            let descriptor = &value["json_schema"];
            fields(descriptor, &["name", "description", "strict", "schema"])?;
            let name = string(&descriptor["name"])?;
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
            {
                return Err(ApiError::invalid(
                    "json_schema.name must be 1..64 letters, digits, underscores or hyphens",
                ));
            }
            if let Some(description) = descriptor.get("description") {
                bounded_text(string(description)?, 1024)?;
            }
            if descriptor.get("strict").is_some_and(|v| !v.is_boolean()) {
                return Err(ApiError::invalid("json_schema.strict must be boolean"));
            }
            descriptor
                .get("schema")
                .filter(|v| v.is_object())
                .cloned()
                .ok_or_else(|| ApiError::invalid("json_schema.schema must be an object"))?
        }
        _ => {
            return Err(ApiError::invalid(
                "response_format.type must be text, json_object or json_schema",
            ));
        }
    };
    tools::validate_output_schema(&schema)?;
    Ok(Some(schema))
}
pub(super) fn constrain(prompt: Prompt, schema: Option<Value>) -> Prompt {
    match schema {
        Some(schema) => Prompt::Constrained {
            prompt: Box::new(prompt),
            schema,
        },
        None => prompt,
    }
}
pub(super) fn validate_output(result: &Generation, schema: Option<&Value>) -> Result<(), ApiError> {
    // Length exhaustion may legitimately produce a partial JSON document.
    if result.finish_reason != "length"
        && let Some(schema) = schema
    {
        let valid = serde_json::from_str::<Value>(&result.text)
            .ok()
            .is_some_and(|value| tools::output_matches(schema, &value));
        if !valid {
            return Err(ApiError::new(
                axum::http::StatusCode::BAD_GATEWAY,
                "invalid_structured_output",
                "model output did not satisfy the requested JSON schema",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scored(text: &str, score: f64) -> Generation {
        Generation {
            text: text.into(),
            prompt_tokens: 3,
            completion_tokens: 1,
            finish_reason: "stop",
            mean_logprob: Some(score),
            logprobs: Some(vec![TokenLogprob {
                token: text.into(),
                bytes: text.as_bytes().to_vec(),
                logprob: score,
                top_logprobs: vec![super::super::inference::TokenAlternative {
                    token: text.into(),
                    bytes: text.as_bytes().to_vec(),
                    logprob: score,
                }],
            }]),
        }
    }
    #[test]
    fn candidate_ranking_is_native_stable_and_usage_counts_hidden_work() {
        let candidates = Candidates { n: 2, best_of: 3 };
        let results = [-2.0, -1.0, -1.0, -0.5, -3.0, -0.25].map(|score| scored("x", score));
        assert_eq!(candidates.select(&results).unwrap(), [1, 2, 5, 3]);
        let mut usage = Usage::default();
        for (index, result) in results.iter().enumerate() {
            usage.add(result, index % 3 == 0);
        }
        assert_eq!(
            usage.value(),
            json!({"prompt_tokens":6,"completion_tokens":6,"total_tokens":12})
        );
        for bad in [None, Some(f64::NAN), Some(f64::INFINITY), Some(0.01)] {
            let mut invalid = results.clone();
            invalid[0].mean_logprob = bad;
            assert!(candidates.select(&invalid).is_err());
        }
        assert_eq!(
            Candidates { n: 3, best_of: 3 }.select(&results).unwrap(),
            [0, 1, 2, 3, 4, 5]
        );
    }
    #[test]
    fn wire_diagnostics_preserve_native_bytes_and_legacy_character_offsets() {
        let mut result = scored("éx", -0.5);
        result.logprobs = Some(
            vec![vec![0xc3], vec![0xa9], vec![b'x']]
                .into_iter()
                .map(|bytes| TokenLogprob {
                    token: String::from_utf8_lossy(&bytes).into_owned(),
                    bytes,
                    logprob: -0.5,
                    top_logprobs: vec![],
                })
                .collect(),
        );
        let chat = chat_logprobs(&result, true).unwrap();
        assert_eq!(chat["content"][0]["bytes"], json!([195]));
        assert_eq!(chat["content"][1]["bytes"], json!([169]));
        assert_eq!(chat["content"][2]["token"], "x");
        let legacy = legacy_logprobs(&result, true).unwrap();
        assert_eq!(legacy["text_offset"], json!([0, 0, 1]));
        assert_eq!(legacy["token_logprobs"], json!([-0.5, -0.5, -0.5]));
        assert_eq!(legacy["top_logprobs"], json!([{}, {}, {}]));
        let normal = scored("visible", -0.25);
        assert_eq!(
            legacy_logprobs(&normal, true).unwrap()["top_logprobs"][0]["visible"],
            -0.25
        );
        assert_eq!(
            chat_logprobs(&normal, true).unwrap()["content"][0]["top_logprobs"][0]["bytes"],
            json!(b"visible".to_vec())
        );
        result.text = "é".into();
        assert!(
            chat_logprobs(&result, true).is_err(),
            "hidden stop record must not be published"
        );
        result.logprobs = None;
        assert!(legacy_logprobs(&result, true).is_err());
        assert_eq!(legacy_logprobs(&result, false).unwrap(), Value::Null);
    }
    #[test]
    fn prompt_shapes_candidate_limits_and_seeds_are_strict() {
        for value in [
            json!("x"),
            json!(["x", "y"]),
            json!([1, 2]),
            json!([[1, 2], [3]]),
        ] {
            assert!(prompts(&value).is_ok(), "{value}");
        }
        for value in [
            json!(null),
            json!([]),
            json!(""),
            json!([[]]),
            json!(["x", 1]),
            json!([1, "x"]),
            json!([-1]),
            json!([2147483648u64]),
            json!([[1], "x"]),
        ] {
            assert!(prompts(&value).is_err(), "{value}");
        }
        for body in [
            json!({"n":0}),
            json!({"n":129}),
            json!({"n":1.5}),
            json!({"n":2,"best_of":1}),
        ] {
            assert!(Candidates::parse(&body, false).is_err(), "{body}");
        }
        let options = Options {
            until_context: false,
            max_tokens: 4096,
            temperature: 0.8,
            top_p: 1.0,
            seed: Some(u32::MAX),
        };
        let candidates = Candidates { n: 2, best_of: 4 };
        assert_eq!(candidates.validate(2, &options, Some(5)).unwrap(), 8);
        assert!(candidates.validate(33, &options, None).is_err());
        assert!(
            Candidates {
                n: 128,
                best_of: 128
            }
            .validate(1, &options, None)
            .is_err()
        );
        assert!(
            Candidates { n: 32, best_of: 32 }
                .validate(1, &options, Some(20))
                .is_err()
        );
        assert_eq!(candidate_options(&options, 0).seed, options.seed);
        assert_ne!(candidate_options(&options, 1).seed, options.seed);
        assert_eq!(
            candidate_options(&options, 127).seed,
            candidate_options(&options, 127).seed
        );
        let mut unseeded = options;
        unseeded.seed = None;
        assert!(candidate_options(&unseeded, 5).seed.is_none());
        for (body, chat) in [
            (json!({"logprobs":true}), false),
            (json!({"logprobs":0}), true),
            (json!({"logprobs":true,"top_logprobs":21}), true),
            (json!({"logprobs":false,"top_logprobs":1}), true),
        ] {
            assert!(single_choice(&body, chat).is_err());
        }
    }
    #[test]
    fn schemas_are_real_bounded_constraints_and_not_ignored_hints() {
        assert!(
            response_schema(Some(&json!({"type":"json_object"})))
                .unwrap()
                .is_some()
        );
        let schema=response_schema(Some(&json!({"type":"json_schema","json_schema":{"name":"Edit","strict":true,"schema":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"],"additionalProperties":false}}}))).unwrap().unwrap();
        let result = |text: &str, finish_reason| Generation {
            text: text.into(),
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason,
            logprobs: None,
            mean_logprob: None,
        };
        assert!(validate_output(&result("{\"code\":\"x\"}", "stop"), Some(&schema)).is_ok());
        assert!(validate_output(&result("{\"wrong\":true}", "stop"), Some(&schema)).is_err());
        assert!(validate_output(&result("{\"code\":", "length"), Some(&schema)).is_ok());
        for value in [
            json!({"type":"json_schema","json_schema":{"name":"x","schema":{"$ref":"remote"}}}),
            json!({"type":"json_object","ignored":true}),
            json!({"type":"xml"}),
        ] {
            assert!(response_schema(Some(&value)).is_err());
        }
    }
    #[test]
    fn sdk_neutral_fields_are_accepted_without_fabricating_features() {
        single_choice(
            &json!({"n":1,"logprobs":false,"top_logprobs":0,"modalities":["text"],"audio":null}),
            true,
        )
        .unwrap();
        single_choice(&json!({"n":1,"best_of":1,"logprobs":null}), false).unwrap();
        for body in [
            json!({"n":129}),
            json!({"logprobs":1}),
            json!({"modalities":["audio"]}),
        ] {
            assert!(single_choice(&body, true).is_err());
        }
        assert!(single_choice(&json!({"logprobs":0}), false).is_ok());
        assert!(single_choice(&json!({"n":128,"logprobs":true,"top_logprobs":20}), true).is_ok());
        assert!(stream_usage(&json!({"stream_options":{"include_usage":true}}), true).unwrap());
        assert!(stream_usage(&json!({"stream_options":{"include_usage":true}}), false).is_err());
    }
}
