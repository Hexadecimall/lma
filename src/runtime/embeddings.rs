//! Native embedding wire adapters. Authentication and lifecycle belong to the router.
use std::time::Instant;

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};

use super::{
    inference::{
        EMBEDDING_MAX_INPUTS, EMBEDDING_MAX_TOKENS, EmbeddingRequest, EmbeddingResult,
        InferenceError, normalize_embedding,
    },
    server::{ApiError, AppState, ensure_available},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    OpenAi,
    Ollama,
    Legacy,
}

struct Parsed {
    request: EmbeddingRequest,
    base64: bool,
}

fn fields(value: &Value, allowed: &[&str]) -> Result<(), ApiError> {
    let object = value
        .as_object()
        .ok_or_else(|| ApiError::invalid("expected a JSON object"))?;
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        // Do not echo arbitrary keys/values, which may contain secrets.
        return Err(ApiError::invalid(format!(
            "unsupported embedding fields; supported here: {}",
            allowed.join(", ")
        )));
    }
    Ok(())
}

fn parse(body: &Value, format: Format, model_id: &str, context: u32) -> Result<Parsed, ApiError> {
    fields(
        body,
        match format {
            Format::OpenAi => &["model", "input", "encoding_format", "user", "dimensions"],
            Format::Ollama => &[
                "model",
                "input",
                "truncate",
                "options",
                "keep_alive",
                "dimensions",
            ],
            Format::Legacy => &["model", "prompt", "options", "keep_alive"],
        },
    )?;
    if let Some(model) = body.get("model") {
        let model = model
            .as_str()
            .ok_or_else(|| ApiError::invalid("model must be a string"))?;
        if model != model_id {
            return Err(ApiError::invalid(format!(
                "model must match the served model id: {model_id}"
            )));
        }
    }
    if body.get("dimensions").is_some() {
        return Err(ApiError::invalid(
            "dimensions overrides are unsupported; the model's full native embedding width is returned",
        ));
    }
    let base64 = match body.get("encoding_format") {
        None => false,
        Some(value) if value == "float" => false,
        Some(value) if value == "base64" => true,
        _ => return Err(ApiError::invalid("encoding_format must be float or base64")),
    };
    if body.get("user").is_some_and(|value| !value.is_string()) {
        return Err(ApiError::invalid("user must be a string"));
    }
    if let Some(value) = body.get("keep_alive").filter(|value| !value.is_null()) {
        // A dynamic outer router can consume lifecycle fields before forwarding
        // to this pinned-model adapter; never pretend to implement timers here.
        if value.as_i64() != Some(-1) {
            return Err(ApiError::invalid(
                "keep_alive must be handled by the lifecycle router; this pinned-model adapter accepts only -1 or omission",
            ));
        }
    }
    if let Some(options) = body.get("options").filter(|value| !value.is_null()) {
        fields(options, &["num_ctx"])?;
        if let Some(value) = options.get("num_ctx")
            && value.as_u64() != Some(u64::from(context))
        {
            return Err(ApiError::invalid(
                "options.num_ctx must equal the hosted context; per-request context changes are unsupported",
            ));
        }
    }
    let truncate = match body.get("truncate") {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| ApiError::invalid("truncate must be boolean"))?,
        None => format == Format::Ollama,
    };
    let key = if format == Format::Legacy {
        "prompt"
    } else {
        "input"
    };
    let value = body
        .get(key)
        .ok_or_else(|| ApiError::invalid(format!("{key} is required")))?;
    let input = match value {
        Value::String(text) => vec![text.clone()],
        Value::Array(items) if format != Format::Legacy => {
            if items.is_empty() || items.len() > EMBEDDING_MAX_INPUTS {
                return Err(ApiError::invalid(format!(
                    "input must contain 1-{EMBEDDING_MAX_INPUTS} strings"
                )));
            }
            items
                .iter()
                .map(|item| {
                    item.as_str().map(str::to_owned).ok_or_else(|| {
                        ApiError::invalid(
                            "input must be a string or array of strings; token-ID inputs are unsupported",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => {
            return Err(ApiError::invalid(format!(
                "{key} must be a string{}",
                if format == Format::Legacy {
                    ""
                } else {
                    " or array of strings; token-ID inputs are unsupported"
                }
            )));
        }
    };
    let request = EmbeddingRequest { input, truncate };
    request.validate().map_err(ApiError::from)?;
    Ok(Parsed { request, base64 })
}

/// `/v1/embeddings`: indexed vectors and exact native token usage.
pub(super) async fn openai(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond(state, payload, Format::OpenAi)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

/// `/api/embed`: a string or bounded batch; truncate defaults to true, as in Ollama.
pub(super) async fn embed(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond(state, payload, Format::Ollama)
        .await
        .unwrap_or_else(ApiError::ollama_response)
}

/// `/api/embeddings`: legacy single prompt; over-context input is rejected.
pub(super) async fn legacy(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond(state, payload, Format::Legacy)
        .await
        .unwrap_or_else(ApiError::ollama_response)
}

async fn respond(
    state: AppState,
    payload: Result<Json<Value>, JsonRejection>,
    format: Format,
) -> Result<Response, ApiError> {
    ensure_available(&state)?;
    let Json(body) = payload.map_err(ApiError::from_json)?;
    let parsed = parse(&body, format, &state.model_id, state.context)?;
    let count = parsed.request.input.len();
    let started = Instant::now();
    let result = state
        .inference
        .embed(parsed.request)
        .await
        .map_err(ApiError::from)?;
    let result = response_value(
        result,
        count,
        &state.model_id,
        format,
        parsed.base64,
        started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
    )?;
    Ok(Json(result).into_response())
}

fn response_value(
    mut result: EmbeddingResult,
    count: usize,
    model_id: &str,
    format: Format,
    base64: bool,
    duration: u64,
) -> Result<Value, ApiError> {
    if count == 0
        || count > EMBEDDING_MAX_INPUTS
        || result.embeddings.len() != count
        || result.prompt_tokens < count
        || result.prompt_tokens > EMBEDDING_MAX_TOKENS
    {
        return Err(InferenceError::Internal(
            "embedding engine returned invalid batch cardinality or token usage".into(),
        )
        .into());
    }
    let width = result.embeddings.first().map_or(0, Vec::len);
    for vector in &mut result.embeddings {
        if vector.len() != width {
            return Err(InferenceError::Internal(
                "embedding engine returned inconsistent vector widths".into(),
            )
            .into());
        }
        // Validate again at the wire boundary, including for injected engines.
        normalize_embedding(vector).map_err(ApiError::from)?;
    }
    Ok(match format {
        Format::OpenAi => {
            let data = result
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(index, vector)| {
                    let embedding = if base64 {
                        let bytes = vector
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>();
                        json!(STANDARD.encode(bytes))
                    } else {
                        json!(vector)
                    };
                    json!({"object":"embedding", "index":index, "embedding":embedding})
                })
                .collect::<Vec<_>>();
            json!({"object":"list", "model":model_id, "data":data,
                "usage":{"prompt_tokens":result.prompt_tokens, "total_tokens":result.prompt_tokens}})
        }
        // No fabricated load_duration: this adapter cannot observe model loading.
        Format::Ollama => json!({"model":model_id, "embeddings":result.embeddings,
            "total_duration":duration, "prompt_eval_count":result.prompt_tokens}),
        Format::Legacy => json!({"embedding":result.embeddings.remove(0)}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_requests_validate_batch_types_options_and_honest_truncation() {
        let parsed = parse(
            &json!({"model":"fixture", "input":["first", "second"], "encoding_format":"base64"}),
            Format::OpenAi,
            "fixture",
            64,
        )
        .unwrap();
        assert_eq!(parsed.request.input, ["first", "second"]);
        assert!(!parsed.request.truncate);
        assert!(parsed.base64);
        assert!(
            parse(&json!({"input":"long"}), Format::Ollama, "fixture", 64)
                .unwrap()
                .request
                .truncate
        );
        assert!(
            !parse(
                &json!({"input":"long", "truncate":false, "options":{"num_ctx":64}}),
                Format::Ollama,
                "fixture",
                64
            )
            .unwrap()
            .request
            .truncate
        );
        assert!(
            !parse(&json!({"prompt":"legacy"}), Format::Legacy, "fixture", 64)
                .unwrap()
                .request
                .truncate
        );
        for body in [
            json!({"input":[]}),
            json!({"input":""}),
            json!({"input":["ok", 7]}),
            json!({"input":[1, 2]}),
            json!({"input":[[1, 2]]}),
            json!({"input":vec!["x"; EMBEDDING_MAX_INPUTS + 1]}),
            json!({"input":"x", "model":"wrong"}),
            json!({"input":"x", "model":null}),
            json!({"input":"x", "truncate":true}),
            json!({"input":"x", "dimensions":2}),
            json!({"input":"x", "encoding_format":"other"}),
            json!({"input":"x", "encoding_format":null}),
            json!({"input":"x", "unknown":"secret"}),
        ] {
            assert!(
                parse(&body, Format::OpenAi, "fixture", 64).is_err(),
                "accepted {body}"
            );
        }
        for body in [
            json!({"input":"x", "truncate":1}),
            json!({"input":"x", "options":{"num_ctx":32}}),
            json!({"input":"x", "options":{"temperature":0}}),
            json!({"input":"x", "keep_alive":"5m"}),
        ] {
            assert!(parse(&body, Format::Ollama, "fixture", 64).is_err());
        }
        assert!(parse(&json!({"prompt":["x"]}), Format::Legacy, "fixture", 64).is_err());
        assert!(
            parse(
                &json!({"input":"x".repeat(super::super::inference::EMBEDDING_MAX_BYTES + 1)}),
                Format::OpenAi,
                "fixture",
                64
            )
            .is_err()
        );
    }

    fn fixture() -> EmbeddingResult {
        EmbeddingResult {
            embeddings: vec![vec![3.0, 4.0], vec![0.0, -2.0]],
            prompt_tokens: 7,
        }
    }

    #[test]
    fn embedding_wire_formats_preserve_order_normalize_and_report_exact_usage() {
        let value = response_value(fixture(), 2, "fixture", Format::OpenAi, false, 123).unwrap();
        assert_eq!(value["object"], "list");
        assert_eq!(value["data"][0]["index"], 0);
        assert_eq!(value["data"][1]["index"], 1);
        assert!((value["data"][0]["embedding"][0].as_f64().unwrap() - 0.6).abs() < 1e-6);
        assert_eq!(value["data"][1]["embedding"], json!([0.0, -1.0]));
        assert_eq!(value["usage"], json!({"prompt_tokens":7, "total_tokens":7}));
        let encoded = response_value(fixture(), 2, "fixture", Format::OpenAi, true, 0).unwrap();
        let bytes = STANDARD
            .decode(encoded["data"][0]["embedding"].as_str().unwrap())
            .unwrap();
        assert_eq!(bytes, [0.6f32.to_le_bytes(), 0.8f32.to_le_bytes()].concat());
        let value = response_value(fixture(), 2, "fixture", Format::Ollama, false, 123).unwrap();
        assert_eq!(value["model"], "fixture");
        assert_eq!(value["prompt_eval_count"], 7);
        assert_eq!(value["total_duration"], 123);
        assert!(value.get("load_duration").is_none());
        let value = response_value(
            EmbeddingResult {
                embeddings: vec![vec![3.0, 4.0]],
                prompt_tokens: 3,
            },
            1,
            "fixture",
            Format::Legacy,
            false,
            0,
        )
        .unwrap();
        assert_eq!(value.as_object().unwrap().len(), 1);
        assert_eq!(value["embedding"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn embedding_wire_boundary_rejects_invalid_native_output() {
        for vectors in [
            vec![],
            vec![vec![]],
            vec![vec![0.0, 0.0]],
            vec![vec![f32::NAN]],
            vec![vec![f32::INFINITY]],
            vec![vec![1.0], vec![1.0, 2.0]],
        ] {
            let count = vectors.len().max(1);
            assert!(
                response_value(
                    EmbeddingResult {
                        embeddings: vectors,
                        prompt_tokens: count
                    },
                    count,
                    "fixture",
                    Format::OpenAi,
                    false,
                    0
                )
                .is_err()
            );
        }
        assert!(response_value(fixture(), 1, "fixture", Format::OpenAi, false, 0).is_err());
        for prompt_tokens in [0, EMBEDDING_MAX_TOKENS + 1] {
            let result = EmbeddingResult {
                embeddings: vec![vec![1.0]],
                prompt_tokens,
            };
            assert!(response_value(result, 1, "fixture", Format::Ollama, false, 0).is_err());
        }
    }
}
