//! Ollama wire adapters for one statically hosted model, not an Ollama daemon.
use std::{
    convert::Infallible,
    sync::atomic::Ordering,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    body::Body,
    extract::{State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    inference::{Generation, GenerationEvent, GenerationRequest, Options, Prompt},
    server::{
        ApiError, AppState, BODY_LIMIT, ChatContent, ChatMessage, ensure_available,
        validate_chat_text, validate_model,
    },
    tools::{self, Call, Family, History, Output, Plan, bounded_text, string},
};

// Do not reflect unknown keys or values: either can contain credentials.
fn fields(value: &Value, allowed: &[&str]) -> Result<(), ApiError> {
    let object = tools::object(value)?;
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(ApiError::invalid(format!(
            "unsupported Ollama fields/options; supported here: {}; see docs/ollama-api.md",
            allowed.join(", ")
        )));
    }
    Ok(())
}
fn boolean(body: &Value, key: &str, default: bool) -> Result<bool, ApiError> {
    body.get(key)
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| ApiError::invalid(format!("{key} must be boolean")))
        })
        .unwrap_or(Ok(default))
}
fn uint(body: &Value, key: &str, default: u32) -> Result<u32, ApiError> {
    body.get(key)
        .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| {
            ApiError::invalid(format!("{key} must be an unsigned 32-bit integer; unlimited token budgets are unsupported"))
        }))
        .unwrap_or(Ok(default))
}
fn number(body: &Value, key: &str, default: f32) -> Result<f32, ApiError> {
    body.get(key)
        .map(|v| {
            v.as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| ApiError::invalid(format!("{key} must be numeric")))
        })
        .unwrap_or(Ok(default))
}
fn text(body: &Value, key: &str) -> Result<String, ApiError> {
    let text = body.get(key).map(string).transpose()?.unwrap_or("");
    bounded_text(text, BODY_LIMIT)?;
    validate_chat_text(text)?;
    Ok(text.into())
}
fn images(body: &Value) -> Result<Vec<String>, ApiError> {
    body.get("images")
        .map(|value| {
            let items = value.as_array().ok_or_else(|| {
                ApiError::invalid("images must be an array of base64 PNG/JPEG strings")
            })?;
            if items.len() > 8 {
                return Err(ApiError::invalid("at most 8 images are supported"));
            }
            items
                .iter()
                .map(|value| Ok(string(value)?.to_owned()))
                .collect()
        })
        .unwrap_or_else(|| Ok(Vec::new()))
}
fn message(role: &str, content: String, images: Vec<String>) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: ChatContent::Text(content),
        images,
    }
}

struct Parsed {
    plan: Plan,
    history: History,
    raw: Option<String>,
    options: Options,
    stream: bool,
}
impl Parsed {
    fn parse(state: &AppState, body: &Value, chat: bool) -> Result<Self, ApiError> {
        fields(
            body,
            if chat {
                &[
                    "model",
                    "messages",
                    "tools",
                    "stream",
                    "options",
                    "keep_alive",
                    "format",
                    "think",
                ]
            } else {
                &[
                    "model",
                    "prompt",
                    "system",
                    "raw",
                    "images",
                    "stream",
                    "options",
                    "keep_alive",
                    "format",
                    "think",
                ]
            },
        )?;
        validate_model(state, body.get("model").map(string).transpose()?)?;
        if let Some(value) = body.get("keep_alive").filter(|v| !v.is_null())
            && value.as_i64() != Some(-1)
        {
            return Err(ApiError::invalid(
                "keep_alive timers, loading and unloading are unsupported: this server pins one model until shutdown; omit keep_alive or use -1 as a pinned-model hint",
            ));
        }
        if body.get("format").is_some_and(|v| !v.is_null() && v != "") {
            return Err(ApiError::invalid(
                "format JSON/schema output is unsupported on this adapter; omit format (function tool arguments use validated constrained schemas)",
            ));
        }
        if let Some(value) = body.get("think")
            && value != false
        {
            return Err(ApiError::invalid(
                "separate thinking output/control is unsupported; omit think or use false (no separate thinking channel is requested; model reasoning is not suppressed)",
            ));
        }
        let empty = json!({});
        let opts = body.get("options").unwrap_or(&empty);
        fields(
            opts,
            &["num_predict", "temperature", "top_p", "seed", "num_ctx"],
        )?;
        if opts.get("num_ctx").is_some() && uint(opts, "num_ctx", state.context)? != state.context {
            return Err(ApiError::invalid(
                "options.num_ctx must equal the hosted context; restart the server to change context size",
            ));
        }
        let defaults = state.generation_defaults();
        let options = Options {
            until_context: false,
            max_tokens: uint(opts, "num_predict", defaults.max_tokens)?,
            temperature: number(opts, "temperature", defaults.temperature)?,
            top_p: number(opts, "top_p", defaults.top_p)?,
            seed: opts
                .get("seed")
                .map(|_| uint(opts, "seed", 0))
                .transpose()?
                .or(defaults.seed),
        };
        options.validate(state.context)?;
        let plan = Plan::parse(body.get("tools"), None, Family::Chat, true)?;
        let mut history = History::default();
        let mut raw = None;
        if chat {
            history = chat_history(body, &plan)?;
        } else {
            let prompt = text(body, "prompt")?;
            let system = text(body, "system")?;
            let images = images(body)?;
            if prompt.is_empty() && images.is_empty() {
                return Err(ApiError::invalid(
                    "prompt must be nonempty unless images are provided; empty-prompt model loading/unloading is unsupported",
                ));
            }
            if boolean(body, "raw", false)? {
                if !system.is_empty() || !images.is_empty() {
                    return Err(ApiError::invalid(
                        "raw:true accepts text only; omit system/images or use raw:false for chat-template and vision handling",
                    ));
                }
                raw = Some(prompt);
            } else {
                if !system.is_empty() {
                    history.message(message("system", system, Vec::new()))?;
                }
                history.message(message("user", prompt, images))?;
            }
        }
        history.validate()?;
        // Images have a separate body/decoded-byte allowance; text does not.
        let bytes: usize = history
            .messages
            .iter()
            .map(|m| match &m.content {
                ChatContent::Text(s) => s.len(),
                ChatContent::Parts(_) => unreachable!("Ollama content is text"),
            })
            .sum();
        if bytes > BODY_LIMIT {
            return Err(ApiError::invalid(
                "Ollama message text/history must total at most 1 MiB; images have separate limits",
            ));
        }
        Ok(Self {
            plan,
            history,
            raw,
            options,
            stream: boolean(body, "stream", true)?,
        })
    }
}

fn chat_history(body: &Value, plan: &Plan) -> Result<History, ApiError> {
    let messages = body["messages"]
        .as_array()
        .ok_or_else(|| ApiError::invalid("messages must be an array with 1-512 entries"))?;
    if messages.is_empty() || messages.len() > 512 {
        return Err(ApiError::invalid("messages must contain 1-512 entries"));
    }
    let mut history = History::default();
    let mut pending: Vec<(String, String)> = Vec::new();
    for (mi, item) in messages.iter().enumerate() {
        fields(
            item,
            &["role", "content", "images", "tool_calls", "tool_name"],
        )?;
        let role = string(&item["role"])?;
        let content = text(item, "content")?;
        let attachments = images(item)?;
        if role == "tool" {
            fields(item, &["role", "content", "tool_name"])?;
            let name = item.get("tool_name").map(string).transpose()?;
            // Ollama has no call IDs. Named results resolve the oldest pending
            // call of that name; unnamed results resolve in assistant call order.
            let index = pending.iter().position(|(_, n)| name.is_none_or(|name| name == n))
                .ok_or_else(|| ApiError::invalid("tool result needs a preceding unresolved assistant call; tool_name must match when provided"))?;
            let (id, _) = pending.remove(index);
            history.result(&id, content, false)?;
        } else if role == "assistant" && item.get("tool_calls").is_some() {
            fields(item, &["role", "content", "tool_calls"])?;
            let entries = match item.get("tool_calls").filter(|v| !v.is_null()) {
                Some(value) => value
                    .as_array()
                    .ok_or_else(|| ApiError::invalid("tool_calls must be an array"))?
                    .as_slice(),
                None => &[],
            };
            if entries.is_empty() {
                history.message(message(role, content, Vec::new()))?;
                continue;
            }
            if entries.len() > tools::MAX_CALLS {
                return Err(ApiError::invalid(
                    "at most 8 calls per assistant turn are supported",
                ));
            }
            let mut calls = Vec::new();
            for (ci, call) in entries.iter().enumerate() {
                fields(call, &["function"])?;
                let function = &call["function"];
                fields(function, &["name", "arguments", "index"])?;
                if function.get("index").is_some_and(|v| v.as_u64().is_none()) {
                    return Err(ApiError::invalid(
                        "function.index must be a nonnegative integer if supplied",
                    ));
                }
                let name = string(&function["name"])?;
                let arguments = &function["arguments"];
                if !arguments.is_object() {
                    return Err(ApiError::invalid(
                        "Ollama function.arguments must be a JSON object, not a JSON string",
                    ));
                }
                let id = format!("ollama_{mi}_{ci}");
                calls.push(Call {
                    id: id.clone(),
                    name: name.into(),
                    arguments: arguments.clone(),
                });
                pending.push((id, name.into()));
            }
            history.calls(plan, calls, content)?;
        } else if matches!(role, "system" | "user" | "assistant") {
            fields(item, &["role", "content", "images"])?;
            history.message(message(role, content, attachments))?;
        } else {
            return Err(ApiError::invalid(
                "message role must be system, user, assistant or tool",
            ));
        }
    }
    history.validate()?;
    Ok(history)
}

pub(super) async fn chat(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond(state, payload, true)
        .await
        .unwrap_or_else(ApiError::ollama_response)
}
pub(super) async fn generate(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond(state, payload, false)
        .await
        .unwrap_or_else(ApiError::ollama_response)
}
async fn respond(
    state: AppState,
    payload: Result<Json<Value>, JsonRejection>,
    chat: bool,
) -> Result<Response, ApiError> {
    let started = Instant::now();
    let Json(body) = payload.map_err(ApiError::from_image_json)?;
    let Parsed {
        plan,
        history,
        raw,
        options,
        stream,
    } = Parsed::parse(&state, &body, chat)?;
    let prepare_plan = plan.clone();
    let has_definition = state.definition.is_some();
    let explicit_system = history
        .messages
        .iter()
        .any(|message| message.role == "system")
        || (!chat && body.get("system").is_some());
    let prepare = move || {
        Ok(GenerationRequest {
            prompt: match raw {
                Some(text) => Prompt::Text(text),
                None => {
                    let prompt = prepare_plan.prepare(history)?;
                    if has_definition {
                        prompt.with_request_system(explicit_system)
                    } else {
                        prompt
                    }
                }
            },
            options,
        })
    };
    let id = format!("ollama_{}", state.next_id.fetch_add(1, Ordering::Relaxed));
    if !stream {
        let result = state.inference.generate_prepared(prepare).await?;
        let output = plan.decode(&result, &id)?;
        return Ok(generation_headers(
            Json(complete(&state.model_id, chat, &result, &output, started)).into_response(),
            None,
        ));
    }
    let mut native = state.inference.stream_prepared(prepare).await?;
    let (send, receive) = mpsc::channel::<Result<String, Infallible>>(8);
    let constrained = plan.constrained();
    tokio::spawn(async move {
        let mut emitted = String::new();
        let mut structured_bytes = 0usize;
        loop {
            let event =
                tokio::select! { _ = send.closed() => return, event = native.recv() => event };
            let error = match event {
                Some(GenerationEvent::Delta(text)) => {
                    if constrained {
                        structured_bytes = structured_bytes.saturating_add(text.len());
                        if structured_bytes > tools::MAX_TOOL_OUTPUT {
                            Some(ApiError::new(
                                StatusCode::BAD_GATEWAY,
                                "invalid_tool_output",
                                "constrained tool output exceeds 256 KiB; no call was issued",
                            ))
                        } else {
                            None
                        }
                    } else {
                        emitted.push_str(&text);
                        if !send_line(&send, frame(&state.model_id, chat, &text, &[], false)).await
                        {
                            return;
                        }
                        None
                    }
                }
                Some(GenerationEvent::Finished(Ok(result))) => {
                    if !constrained && emitted != result.text {
                        Some(ApiError::new(
                            StatusCode::BAD_GATEWAY,
                            "invalid_stream",
                            "native stream text did not match its final result",
                        ))
                    } else {
                        match plan.decode(&result, &id) {
                            Ok(output) => {
                                if constrained
                                    && !send_line(
                                        &send,
                                        frame(
                                            &state.model_id,
                                            chat,
                                            &output.text,
                                            &output.calls,
                                            false,
                                        ),
                                    )
                                    .await
                                {
                                    return;
                                }
                                let terminal = complete(
                                    &state.model_id,
                                    chat,
                                    &result,
                                    &Output {
                                        text: String::new(),
                                        calls: Vec::new(),
                                    },
                                    started,
                                );
                                let _ = send_line(&send, terminal).await;
                                return;
                            }
                            Err(error) => Some(error),
                        }
                    }
                }
                Some(GenerationEvent::Finished(Err(error))) => Some(error.into()),
                None => Some(ApiError::new(
                    StatusCode::BAD_GATEWAY,
                    "invalid_stream",
                    "native stream ended without a final result",
                )),
            };
            if let Some(error) = error {
                let _ = send_line(&send, error.ollama_value()).await;
                return;
            }
        }
    });
    let response = (
        [
            (header::CONTENT_TYPE, "application/x-ndjson"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        Body::from_stream(ReceiverStream::new(receive)),
    )
        .into_response();
    Ok(generation_headers(
        response,
        Some(if constrained {
            "validated-tool-buffer"
        } else {
            "native"
        }),
    ))
}
async fn send_line(send: &mpsc::Sender<Result<String, Infallible>>, value: Value) -> bool {
    send.send(Ok(format!("{value}\n"))).await.is_ok()
}
fn generation_headers(mut response: Response, streaming: Option<&'static str>) -> Response {
    response.headers_mut().insert(
        "x-lma-model-lifecycle",
        "pinned-until-shutdown".parse().unwrap(),
    );
    if let Some(mode) = streaming {
        response
            .headers_mut()
            .insert("x-lma-streaming", mode.parse().unwrap());
    }
    response
}
fn frame(model: &str, chat: bool, text: &str, calls: &[Call], done: bool) -> Value {
    let mut frame = json!({"model":model, "created_at":timestamp(SystemTime::now()), "done":done});
    if chat {
        frame["message"] = json!({"role":"assistant", "content":text});
        if !calls.is_empty() {
            frame["message"]["tool_calls"] = json!(
                calls
                    .iter()
                    .map(|c| json!({"function":{"name":c.name,"arguments":c.arguments}}))
                    .collect::<Vec<_>>()
            );
        }
    } else {
        frame["response"] = json!(text);
    }
    frame
}
fn complete(
    model: &str,
    chat: bool,
    result: &Generation,
    output: &Output,
    started: Instant,
) -> Value {
    let mut value = frame(model, chat, &output.text, &output.calls, true);
    value["done_reason"] = json!(result.finish_reason);
    value["prompt_eval_count"] = json!(result.prompt_tokens);
    value["eval_count"] = json!(result.completion_tokens);
    value["total_duration"] = json!(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    value
}

// Gregorian civil date from UTC epoch days. No local timezone or fabricated
// timestamps; the dependency set intentionally has no date/time formatting crate.
fn timestamp(time: SystemTime) -> String {
    let elapsed = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = elapsed.as_secs();
    let z = (seconds / 86400) as i64 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
        seconds / 3600 % 24,
        seconds / 60 % 60,
        seconds % 60,
        elapsed.subsec_nanos()
    )
}

pub(super) async fn version() -> Json<Value> {
    Json(json!({"version":concat!("lma/", env!("CARGO_PKG_VERSION")), "server":"lma"}))
}
pub(super) async fn tags(State(state): State<AppState>) -> Response {
    inventory(state, false)
}
pub(super) async fn ps(State(state): State<AppState>) -> Response {
    inventory(state, true)
}
fn inventory(state: AppState, active: bool) -> Response {
    if let Err(error) = ensure_available(&state) {
        return error.ollama_response();
    }
    let mut model = json!({"name":state.model_id, "model":state.model_id});
    if active {
        model["context_length"] = json!(state.context);
    }
    Json(json!({"models":[model], "lma":{"scope":"served model only", "model_switching":false, "lifecycle":"pinned-until-shutdown"}})).into_response()
}
pub(super) async fn show(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    show_inner(state, payload).unwrap_or_else(ApiError::ollama_response)
}
fn show_inner(
    state: AppState,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = payload.map_err(ApiError::from_json)?;
    fields(&body, &["model", "verbose"])?;
    validate_model(&state, Some(string(&body["model"])?))?;
    boolean(&body, "verbose", false)?;
    ensure_available(&state)?;
    let mut info = state
        .metadata
        .as_ref()
        .map(|m| m.model_info())
        .unwrap_or_default();
    info.insert("lma.model_id".into(), json!(state.model_id));
    info.insert("lma.context_length".into(), json!(state.context));
    Ok(Json(json!({
        "model_info":info,
        "capabilities":["completion"],
        "lma":{"metadata_scope":"bounded selected GGUF header fields when available; configured context is separate from model training context", "gguf_metadata_available":state.metadata.is_some(), "vision_build":cfg!(feature="vision"), "vision_projector_status":"unavailable", "function_tools":"client executed; constrained output validated before delivery", "model_switching":false}
    })).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timestamps_are_utc_rfc3339() {
        assert_eq!(timestamp(UNIX_EPOCH), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(
            timestamp(UNIX_EPOCH + std::time::Duration::from_secs(951782400)),
            "2000-02-29T00:00:00.000000000Z"
        );
        assert_eq!(
            timestamp(UNIX_EPOCH + std::time::Duration::new(1709251199, 123)),
            "2024-02-29T23:59:59.000000123Z"
        );
    }
}
