//! Stateless protocol adapters. Text SSE forwards native deltas; constrained
//! tool envelopes are buffered and validated before any call is exposed.
//! Metadata is validated but not stored; provider cache, reasoning-effort and
//! verbosity hints are advisory only (reported in response headers).
use std::{
    convert::Infallible,
    sync::atomic::Ordering,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    inference::{
        BatchGenerationEvent, Generation, GenerationEvent, GenerationRequest, InferenceError,
        Options, Prompt,
    },
    server::{ApiError, AppState, ChatContent, ChatMessage, validate_model, validate_options},
    tools::{
        self, Call, Family, History, Output, Plan, arguments, bounded_text, cache_hint, fields,
        invalid, metadata, safe_id, string,
    },
};

fn boolean(body: &Value, key: &str, default: bool) -> Result<bool, ApiError> {
    body.get(key)
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| invalid("expected a boolean request option"))
        })
        .unwrap_or(Ok(default))
}
fn uint(body: &Value, key: &str, default: u32) -> Result<u32, ApiError> {
    body.get(key)
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid("expected an unsigned 32-bit integer request option"))
        })
        .unwrap_or(Ok(default))
}
fn number(body: &Value, key: &str, default: f32) -> Result<f32, ApiError> {
    body.get(key)
        .map(|v| {
            v.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| invalid("expected a numeric sampling option"))
        })
        .unwrap_or(Ok(default))
}
fn optional_id(body: &Value, key: &str) -> Result<(), ApiError> {
    if let Some(id) = body.get(key) {
        safe_id(string(id)?)?;
    }
    Ok(())
}
fn text_message(role: &str, text: String) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: ChatContent::Text(text),
        images: Vec::new(),
    }
}
fn array<'a>(body: &'a Value, key: &str) -> Result<&'a Vec<Value>, ApiError> {
    let array = body[key]
        .as_array()
        .ok_or_else(|| invalid("messages/input must be an array"))?;
    if array.is_empty() || array.len() > 512 {
        return Err(invalid("messages/input must contain 1-512 entries"));
    }
    Ok(array)
}

struct Parsed {
    body: Value,
    plan: Plan,
    options: Options,
    stream: bool,
    include_usage: bool,
    controls: super::generation_controls::GenerationControls,
    output_schema: Option<Value>,
}
impl Parsed {
    fn parse(state: &AppState, body: Value, family: Family) -> Result<Self, ApiError> {
        fields(
            &body,
            match family {
                Family::Chat => &[
                    "model",
                    "messages",
                    "max_tokens",
                    "max_completion_tokens",
                    "temperature",
                    "top_p",
                    "seed",
                    "stream",
                    "stream_options",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "metadata",
                    "store",
                    "user",
                    "service_tier",
                    "reasoning_effort",
                    "stop",
                    "frequency_penalty",
                    "presence_penalty",
                    "logit_bias",
                    "n",
                    "logprobs",
                    "top_logprobs",
                    "response_format",
                    "modalities",
                    "audio",
                ],
                Family::Messages => &[
                    "model",
                    "messages",
                    "system",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "stream",
                    "tools",
                    "tool_choice",
                    "metadata",
                    "cache_control",
                ],
                Family::Responses => &[
                    "model",
                    "input",
                    "instructions",
                    "max_output_tokens",
                    "temperature",
                    "top_p",
                    "stream",
                    "stream_options",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "max_tool_calls",
                    "metadata",
                    "store",
                    "previous_response_id",
                    "conversation",
                    "background",
                    "include",
                    "reasoning",
                    "text",
                    "truncation",
                    "service_tier",
                    "user",
                    "safety_identifier",
                    "prompt_cache_key",
                ],
            },
        )?;
        metadata(body.get("metadata"), family == Family::Messages)?;
        cache_hint(body.get("cache_control"))?;
        if boolean(&body, "store", false)?
            || boolean(&body, "background", false)?
            || body
                .get("previous_response_id")
                .is_some_and(|v| !v.is_null())
            || body.get("conversation").is_some_and(|v| !v.is_null())
        {
            return Err(invalid(
                "cloud persistence, background jobs, conversation IDs and previous_response_id are unsupported; set store:false and include the full stateless history",
            ));
        }
        for key in ["user", "safety_identifier", "prompt_cache_key"] {
            if let Some(v) = body.get(key) {
                bounded_text(string(v)?, 512)?;
            }
        }
        if let Some(tier) = body.get("service_tier")
            && tier != "auto"
            && tier != "default"
        {
            return Err(invalid(
                "only auto/default service_tier hints are supported locally",
            ));
        }
        if let Some(truncation) = body.get("truncation")
            && truncation != "disabled"
        {
            return Err(invalid(
                "automatic history truncation is unsupported; use truncation:disabled",
            ));
        }
        if let Some(include) = body.get("include")
            && !include.as_array().is_some_and(Vec::is_empty)
        {
            return Err(invalid(
                "include expansions and encrypted cloud reasoning state are unsupported; use include:[]",
            ));
        }
        if let Some(effort) = body.get("reasoning_effort") {
            reasoning_effort(effort)?;
        }
        if let Some(reasoning) = body.get("reasoning") {
            fields(reasoning, &["effort", "summary"])?;
            if let Some(effort) = reasoning.get("effort") {
                reasoning_effort(effort)?;
            }
            if let Some(summary) = reasoning.get("summary")
                && !summary.is_null()
                && !matches!(string(summary)?, "auto" | "concise" | "detailed")
            {
                return Err(invalid("unsupported reasoning summary hint"));
            }
        }
        if family == Family::Chat {
            super::openai::single_choice(&body, true)?;
        }
        let controls = super::openai::controls(&body)?;
        let mut output_schema = if family == Family::Chat {
            super::openai::response_schema(body.get("response_format"))?
        } else {
            None
        };
        if let Some(text) = body.get("text") {
            fields(text, &["format", "verbosity"])?;
            if let Some(format) = text.get("format") {
                let normalized = if format["type"] == "json_schema" {
                    fields(format, &["type", "name", "schema", "strict", "description"])?;
                    let mut descriptor = format.clone();
                    descriptor.as_object_mut().unwrap().remove("type");
                    json!({"type":"json_schema", "json_schema":descriptor})
                } else {
                    format.clone()
                };
                output_schema = super::openai::response_schema(Some(&normalized))?;
            }
            if let Some(verbosity) = text.get("verbosity")
                && !matches!(string(verbosity)?, "low" | "medium" | "high")
            {
                return Err(invalid("unsupported verbosity hint"));
            }
        }
        let stream = boolean(&body, "stream", false)?;
        let mut include_usage = false;
        if let Some(options) = body.get("stream_options").filter(|v| !v.is_null()) {
            fields(options, &["include_usage", "include_obfuscation"])?;
            include_usage = boolean(options, "include_usage", false)?;
            if !stream || boolean(options, "include_obfuscation", false)? {
                return Err(invalid(
                    "stream_options requires stream:true; stream obfuscation is unsupported",
                ));
            }
        }
        let parallel = boolean(&body, "parallel_tool_calls", true)?;
        let mut plan = Plan::parse(body.get("tools"), body.get("tool_choice"), family, parallel)?;
        if output_schema.is_some() && plan.constrained() {
            return Err(invalid(
                "structured response_format and function-tool generation cannot be combined in one request yet",
            ));
        }
        if body.get("max_tool_calls").is_some() {
            let limit = uint(&body, "max_tool_calls", tools::MAX_CALLS as u32)? as usize;
            if limit == 0 || limit > tools::MAX_CALLS {
                return Err(invalid("max_tool_calls must be between 1 and 8"));
            }
            plan.max_calls = plan.max_calls.min(limit);
        }
        if body.get("max_tokens").is_some() && body.get("max_completion_tokens").is_some() {
            return Err(invalid("use max_tokens or max_completion_tokens, not both"));
        }
        let max_key = if family == Family::Responses {
            "max_output_tokens"
        } else if body.get("max_completion_tokens").is_some() {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        if family == Family::Messages && body.get(max_key).is_none() {
            return Err(invalid(
                "Messages generation requires max_tokens; counting does not",
            ));
        }
        let defaults = state.generation_defaults();
        let options = Options {
            until_context: false,
            max_tokens: uint(&body, max_key, defaults.max_tokens)?,
            temperature: number(&body, "temperature", defaults.temperature)?,
            top_p: number(&body, "top_p", defaults.top_p)?,
            seed: body
                .get("seed")
                .map(|_| uint(&body, "seed", 0))
                .transpose()?
                .or(defaults.seed),
        };
        let model = body.get("model").map(string).transpose()?;
        if family != Family::Chat && model.is_none() {
            return Err(invalid("model is required"));
        }
        validate_options(state, model, false, &options)?;
        Ok(Self {
            body,
            plan,
            options,
            stream,
            include_usage,
            controls,
            output_schema,
        })
    }
}
fn reasoning_effort(value: &Value) -> Result<(), ApiError> {
    if !matches!(
        string(value)?,
        "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
    ) {
        return Err(invalid("unsupported reasoning-effort hint"));
    }
    Ok(())
}

/// Legacy text completions use the same native stream, not post-hoc splitting.
pub(super) async fn completion_stream(
    state: AppState,
    request: GenerationRequest,
    include_usage: bool,
    echo: Option<String>,
) -> Result<Response, ApiError> {
    let (id, created) = identifier(&state, "cmpl");
    let response_id = id.clone();
    let mut native = state.inference.stream_prepared(move || Ok(request)).await?;
    let (send, receive) = mpsc::channel::<Result<Event, Infallible>>(8);
    tokio::spawn(async move {
        let chunk = |text: &str, finish: Option<&str>| {
            json!({
                "id":id, "object":"text_completion", "created":created, "model":state.model_id,
                "choices":[{"index":0,"text":text,"logprobs":null,"finish_reason":finish}]
            })
        };
        if let Some(echo) = echo
            && !send_events(
                &send,
                vec![Event::default().json_data(chunk(&echo, None)).unwrap()],
            )
            .await
        {
            return;
        }
        let mut emitted = String::new();
        loop {
            let event = tokio::select! { _=send.closed()=>return, event=native.recv()=>event };
            let error = match event {
                Some(GenerationEvent::Delta(text)) => {
                    emitted.push_str(&text);
                    if !send_events(
                        &send,
                        vec![Event::default().json_data(chunk(&text, None)).unwrap()],
                    )
                    .await
                    {
                        return;
                    }
                    None
                }
                Some(GenerationEvent::Finished(Ok(result))) if result.text == emitted => {
                    let mut events = vec![
                        Event::default()
                            .json_data(chunk("", Some(result.finish_reason)))
                            .unwrap(),
                    ];
                    if include_usage {
                        events.push(Event::default().json_data(json!({
                            "id":id,"object":"text_completion","created":created,"model":state.model_id,"choices":[],
                            "usage":{"prompt_tokens":result.prompt_tokens,"completion_tokens":result.completion_tokens,"total_tokens":result.prompt_tokens+result.completion_tokens}
                        })).unwrap());
                    }
                    events.push(Event::default().data("[DONE]"));
                    let _ = send_events(&send, events).await;
                    return;
                }
                Some(GenerationEvent::Finished(Ok(_))) => Some(ApiError::new(
                    axum::http::StatusCode::BAD_GATEWAY,
                    "invalid_stream",
                    "native stream text did not match its final result",
                )),
                Some(GenerationEvent::Finished(Err(error))) => Some(error.into()),
                None => Some(ApiError::new(
                    axum::http::StatusCode::BAD_GATEWAY,
                    "invalid_stream",
                    "native stream ended without a final result",
                )),
            };
            if let Some(error) = error {
                let _ = send_events(
                    &send,
                    vec![
                        Event::default()
                            .event("error")
                            .json_data(error.stream_value(&id))
                            .unwrap(),
                    ],
                )
                .await;
                return;
            }
        }
    });
    Ok(headers(
        Sse::new(ReceiverStream::new(receive))
            .keep_alive(KeepAlive::default())
            .into_response(),
        &response_id,
        Some("native"),
    ))
}

pub(super) enum BatchFormat {
    Chat { plan: Plan, schema: Option<Value> },
    Legacy { echoes: Vec<String> },
}
pub(super) struct TextBatch {
    pub candidates: super::openai::Candidates,
    pub work: usize,
    pub stream: bool,
    pub include_usage: bool,
    pub diagnostics: bool,
    pub format: BatchFormat,
}
impl TextBatch {
    fn chat(&self) -> bool {
        matches!(self.format, BatchFormat::Chat { .. })
    }
    fn buffered(&self) -> bool {
        self.diagnostics
            || matches!(&self.format, BatchFormat::Chat { plan, .. } if plan.constrained())
    }
    fn echo_bytes(&self) -> usize {
        match &self.format {
            BatchFormat::Legacy { echoes } => echoes.iter().fold(0usize, |total, text| {
                total.saturating_add(text.len().saturating_mul(self.candidates.n))
            }),
            _ => 0,
        }
    }
    fn echo(&self, index: usize) -> &str {
        match &self.format {
            BatchFormat::Legacy { echoes } => &echoes[index / self.candidates.best_of],
            _ => "",
        }
    }
    fn choice(
        &self,
        frames: &Frames,
        result: &Generation,
        sample: usize,
        index: usize,
    ) -> Result<Value, ApiError> {
        match &self.format {
            BatchFormat::Chat { plan, schema } => {
                super::openai::validate_output(result, schema.as_ref())?;
                // The public response ID is shared; tool-call IDs must still be
                // unique across independently decoded candidate envelopes.
                let output = plan.decode(result, &format!("{}_choice_{sample}", frames.id))?;
                let mut choice = frames.complete(result, &output)["choices"][0].clone();
                choice["index"] = json!(index);
                choice["logprobs"] = super::openai::chat_logprobs(result, self.diagnostics)?;
                Ok(choice)
            }
            BatchFormat::Legacy { .. } => Ok(
                json!({"index":index,"text":format!("{}{}",self.echo(sample),result.text),"finish_reason":result.finish_reason,"logprobs":super::openai::legacy_logprobs(result,self.diagnostics)?}),
            ),
        }
    }
    fn chunk(&self, frames: &Frames, choice: Value) -> Value {
        let mut value = json!({"id":frames.id,"object":if self.chat(){"chat.completion.chunk"}else{"text_completion"},"created":frames.created,"model":frames.model,"choices":[choice]});
        if self.include_usage {
            value["usage"] = Value::Null;
        }
        value
    }
    fn delta(&self, frames: &Frames, index: usize, text: &str) -> Event {
        let choice = if self.chat() {
            json!({"index":index,"delta":{"content":text},"finish_reason":null,"logprobs":null})
        } else {
            json!({"index":index,"text":text,"finish_reason":null,"logprobs":null})
        };
        Event::default()
            .json_data(self.chunk(frames, choice))
            .unwrap()
    }
    fn finish(
        &self,
        frames: &Frames,
        result: &Generation,
        index: usize,
    ) -> Result<Vec<Event>, ApiError> {
        let choice = self.choice(frames, result, index, index)?;
        let mut events = Vec::new();
        if self.buffered() {
            if self.chat() {
                let message = &choice["message"];
                let mut delta = json!({});
                if !message["content"].is_null() {
                    delta["content"] = message["content"].clone();
                }
                if let Some(calls) = message["tool_calls"].as_array() {
                    delta["tool_calls"] = json!(
                        calls
                            .iter()
                            .enumerate()
                            .map(|(i, call)| {
                                let mut call = call.clone();
                                call["index"] = json!(i);
                                call
                            })
                            .collect::<Vec<_>>()
                    );
                }
                events.push(Event::default().json_data(self.chunk(frames,json!({"index":index,"delta":delta,"finish_reason":null,"logprobs":choice["logprobs"]}))).unwrap());
            } else {
                events.push(Event::default().json_data(self.chunk(frames,json!({"index":index,"text":choice["text"],"finish_reason":null,"logprobs":choice["logprobs"]}))).unwrap());
            }
        }
        let finish = if self.chat() {
            json!({"index":index,"delta":{},"finish_reason":choice["finish_reason"],"logprobs":null})
        } else {
            json!({"index":index,"text":"","finish_reason":choice["finish_reason"],"logprobs":null})
        };
        events.push(
            Event::default()
                .json_data(self.chunk(frames, finish))
                .unwrap(),
        );
        Ok(events)
    }
}

/// A single admitted native job owns every candidate. Dropping the outgoing
/// body wakes send.closed(), drops the native receiver, and cancels the batch.
pub(super) async fn text_batch(
    state: AppState,
    batch: TextBatch,
    prepare: impl FnOnce() -> Result<Vec<GenerationRequest>, InferenceError> + Send + 'static,
) -> Result<Response, ApiError> {
    text_batch_with_limit(state, batch, prepare, super::openai::MAX_OUTPUT_BYTES).await
}

async fn text_batch_with_limit(
    state: AppState,
    batch: TextBatch,
    prepare: impl FnOnce() -> Result<Vec<GenerationRequest>, InferenceError> + Send + 'static,
    limit: usize,
) -> Result<Response, ApiError> {
    let budget = super::generation_controls::OutputBudget::new(limit);
    budget.charge(batch.echo_bytes())?;
    let prepare = move || {
        let mut requests = prepare()?;
        for request in &mut requests {
            let prompt = std::mem::replace(&mut request.prompt, Prompt::Text(String::new()));
            request.prompt = prompt.with_output_budget(budget.clone());
        }
        Ok(requests)
    };
    let (id, created) = identifier(&state, if batch.chat() { "chatcmpl" } else { "cmpl" });
    let frames = Frames {
        family: Family::Chat,
        id: id.clone(),
        created,
        model: state.model_id.clone(),
        sequence: 0,
        include_usage: false,
        parallel: false,
        metadata: json!({}),
    };
    if !batch.stream {
        let results = if batch.work == 1 {
            vec![
                state
                    .inference
                    .generate_prepared(move || {
                        let mut requests = prepare()?;
                        if requests.len() != 1 {
                            return Err(InferenceError::Internal(
                                "invalid single-candidate preparation".into(),
                            ));
                        }
                        Ok(requests.remove(0))
                    })
                    .await?,
            ]
        } else {
            state.inference.generate_batch_prepared(prepare).await?
        };
        if results.len() != batch.work {
            return Err(super::openai::native_error(
                "native batch returned an unexpected candidate count",
            ));
        }
        let mut usage = super::openai::Usage::default();
        let mut bytes = batch.echo_bytes();
        for (index, result) in results.iter().enumerate() {
            usage.add(result, index % batch.candidates.best_of == 0);
            bytes = bytes.saturating_add(super::openai::output_bytes(result));
        }
        if bytes > super::openai::MAX_OUTPUT_BYTES {
            return Err(super::openai::resource_error(
                "aggregate native output exceeds 64 MiB",
            ));
        }
        let choices = batch
            .candidates
            .select(&results)?
            .into_iter()
            .enumerate()
            .map(|(index, sample)| batch.choice(&frames, &results[sample], sample, index))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(headers(Json(json!({"id":id,"object":if batch.chat(){"chat.completion"}else{"text_completion"},"created":created,"model":state.model_id,"choices":choices,"usage":usage.value()})).into_response(), &id, None));
    }
    let mut native = state.inference.stream_batch_prepared(prepare).await?;
    let (send, receive) = mpsc::channel::<Result<Event, Infallible>>(8);
    let response_id = id.clone();
    let mode = if batch.buffered() {
        "validated-choice-buffer"
    } else {
        "native"
    };
    tokio::spawn(async move {
        let mut finished = 0;
        let mut emitted = String::new();
        let mut bytes = batch.echo_bytes();
        let mut usage = super::openai::Usage::default();
        // All lifecycle chunks use the request ID and stable flattened indexes.
        for index in 0..batch.work {
            let event = if batch.chat() {
                Event::default().json_data(batch.chunk(&frames,json!({"index":index,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}))).unwrap()
            } else {
                batch.delta(
                    &frames,
                    index,
                    if batch.buffered() {
                        ""
                    } else {
                        batch.echo(index)
                    },
                )
            };
            if !send_events(&send, vec![event]).await {
                return;
            }
        }
        loop {
            let event = tokio::select! { _=send.closed()=>return, event=native.recv()=>event };
            let outcome: Result<bool, ApiError> = match event {
                Some(BatchGenerationEvent::Delta { index, text })
                    if index == finished && index < batch.work =>
                {
                    bytes = bytes.saturating_add(text.len());
                    if bytes > super::openai::MAX_OUTPUT_BYTES {
                        Err(super::openai::resource_error(
                            "aggregate native output exceeds 64 MiB",
                        ))
                    } else if matches!(&batch.format,BatchFormat::Chat { plan, .. } if plan.constrained())
                        && emitted.len().saturating_add(text.len()) > tools::MAX_TOOL_OUTPUT
                    {
                        Err(super::openai::native_error(
                            "constrained tool output exceeds 256 KiB; no call was issued",
                        ))
                    } else {
                        emitted.push_str(&text);
                        if !batch.buffered()
                            && !send_events(&send, vec![batch.delta(&frames, index, &text)]).await
                        {
                            return;
                        }
                        Ok(false)
                    }
                }
                Some(BatchGenerationEvent::Finished {
                    index,
                    result: Ok(result),
                }) if index == finished && index < batch.work => {
                    bytes = bytes.saturating_add(
                        super::openai::output_bytes(&result).saturating_sub(result.text.len()),
                    );
                    if bytes > super::openai::MAX_OUTPUT_BYTES {
                        Err(super::openai::resource_error(
                            "aggregate native output and diagnostics exceed 64 MiB",
                        ))
                    } else if result.text != emitted {
                        Err(super::openai::native_error(
                            "native stream text did not match its final result",
                        ))
                    } else {
                        match batch.finish(&frames, &result, index) {
                            Ok(events) => {
                                usage.add(&result, index % batch.candidates.best_of == 0);
                                if !send_events(&send, events).await {
                                    return;
                                }
                                emitted.clear();
                                finished += 1;
                                Ok(finished == batch.work)
                            }
                            Err(error) => Err(error),
                        }
                    }
                }
                Some(BatchGenerationEvent::Finished {
                    result: Err(error), ..
                }) => Err(error.into()),
                Some(_) => Err(super::openai::native_error(
                    "native batch stream delivered an invalid choice index",
                )),
                None => Err(super::openai::native_error(
                    "native batch stream ended without all final results",
                )),
            };
            match outcome {
                Ok(false) => {}
                Ok(true) => {
                    let mut events = Vec::new();
                    if batch.include_usage {
                        let mut value = batch.chunk(&frames, Value::Null);
                        value["choices"] = json!([]);
                        value["usage"] = usage.value();
                        events.push(Event::default().json_data(value).unwrap());
                    }
                    events.push(Event::default().data("[DONE]"));
                    let _ = send_events(&send, events).await;
                    return;
                }
                Err(error) => {
                    // Cancel native work before a stalled client can block error delivery.
                    drop(native);
                    let _ = send_events(
                        &send,
                        vec![
                            Event::default()
                                .event("error")
                                .json_data(error.stream_value(&id))
                                .unwrap(),
                        ],
                    )
                    .await;
                    return;
                }
            }
        }
    });
    Ok(headers(
        Sse::new(ReceiverStream::new(receive))
            .keep_alive(KeepAlive::default())
            .into_response(),
        &response_id,
        Some(mode),
    ))
}

pub(super) async fn chat(
    state: AppState,
    payload: Result<Json<Value>, JsonRejection>,
    until_context: bool,
) -> Result<Response, ApiError> {
    let Json(body) = payload.map_err(ApiError::from_image_json)?;
    let mut parsed = Parsed::parse(&state, body, Family::Chat)?;
    parsed.options.until_context = until_context;
    generate(state, parsed, Family::Chat).await
}
pub(super) async fn messages(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = payload.map_err(ApiError::from_json)?;
    let parsed = Parsed::parse(&state, body, Family::Messages)?;
    generate(state, parsed, Family::Messages).await
}
pub(super) async fn responses(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = payload.map_err(ApiError::from_json)?;
    let parsed = Parsed::parse(&state, body, Family::Responses)?;
    generate(state, parsed, Family::Responses).await
}

async fn generate(state: AppState, parsed: Parsed, family: Family) -> Result<Response, ApiError> {
    let history = history(&parsed.body, family, &parsed.plan)?;
    let plan = parsed.plan;
    let prepare_plan = plan.clone();
    let has_definition = state.definition.is_some();
    let explicit_system = history
        .messages
        .iter()
        .any(|message| message.role == "system");
    let output_schema = parsed.output_schema.clone();
    let prepare_schema = output_schema.clone();
    let candidates = super::openai::Candidates::parse(&parsed.body, true)?;
    let diagnostics = parsed.controls.logprobs.is_some();
    let work = if family == Family::Chat {
        candidates.validate(1, &parsed.options, parsed.controls.logprobs)?
    } else {
        1
    };
    if family == Family::Chat && diagnostics && plan.constrained() {
        return Err(invalid(
            "Chat logprobs cannot be combined with constrained tool envelopes: native envelope tokens are not assistant content tokens",
        ));
    }
    let prepare = move || {
        let prompt = super::openai::constrain(prepare_plan.prepare(history)?, prepare_schema)
            .with_controls(parsed.controls);
        Ok(GenerationRequest {
            prompt: if has_definition {
                prompt.with_request_system(explicit_system)
            } else {
                prompt
            },
            options: parsed.options,
        })
    };
    if family == Family::Chat && (work > 1 || diagnostics) {
        return text_batch(
            state,
            TextBatch {
                candidates,
                work,
                stream: parsed.stream,
                include_usage: parsed.include_usage,
                diagnostics,
                format: BatchFormat::Chat {
                    plan,
                    schema: output_schema,
                },
            },
            move || {
                let request = prepare()?;
                super::openai::validate_prompt_storage(
                    std::slice::from_ref(&request.prompt),
                    work,
                )?;
                Ok((0..work)
                    .map(|index| GenerationRequest {
                        prompt: request.prompt.clone(),
                        options: super::openai::candidate_options(&request.options, index),
                    })
                    .collect())
            },
        )
        .await;
    }
    let prefix = match family {
        Family::Chat => "chatcmpl",
        Family::Messages => "msg",
        Family::Responses => "resp",
    };
    let (id, created) = identifier(&state, prefix);
    let metadata = parsed
        .body
        .get("metadata")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let frames = Frames {
        family,
        id: id.clone(),
        created,
        model: state.model_id.clone(),
        sequence: 0,
        include_usage: parsed.include_usage,
        parallel: plan.max_calls > 1,
        metadata,
    };
    if !parsed.stream {
        let result = state.inference.generate_prepared(prepare).await?;
        super::openai::validate_output(&result, output_schema.as_ref())?;
        let output = plan.decode(&result, &id)?;
        return Ok(headers(
            Json(frames.complete(&result, &output)).into_response(),
            &id,
            None,
        ));
    }
    let mut native = state.inference.stream_prepared(prepare).await?;
    let (send, receive) = mpsc::channel::<Result<Event, Infallible>>(8);
    let constrained = plan.constrained();
    tokio::spawn(async move {
        let mut frames = frames;
        if !send_events(&send, frames.start(constrained)).await {
            return;
        }
        let mut emitted = String::new();
        let mut structured_bytes = 0usize;
        loop {
            let event = tokio::select! { _=send.closed()=>return, event=native.recv()=>event };
            match event {
                Some(GenerationEvent::Delta(text)) => {
                    if constrained {
                        structured_bytes = structured_bytes.saturating_add(text.len());
                        if structured_bytes > tools::MAX_TOOL_OUTPUT {
                            let _ = send_events(
                                &send,
                                frames.error(ApiError::new(
                                    axum::http::StatusCode::BAD_GATEWAY,
                                    "invalid_tool_output",
                                    "constrained tool output exceeds 256 KiB; no call was issued",
                                )),
                            )
                            .await;
                            return;
                        }
                    } else {
                        emitted.push_str(&text);
                        if !send_events(&send, frames.delta(&text)).await {
                            return;
                        }
                    }
                }
                Some(GenerationEvent::Finished(result)) => {
                    let result = match result {
                        Ok(result) => result,
                        Err(error) => {
                            let _ = send_events(&send, frames.error(error.into())).await;
                            return;
                        }
                    };
                    // Never synthesize missing native deltas or silently change
                    // text already delivered to the client.
                    if !constrained && emitted != result.text {
                        let _ = send_events(
                            &send,
                            frames.error(ApiError::new(
                                axum::http::StatusCode::BAD_GATEWAY,
                                "invalid_stream",
                                "native stream text did not match its final result",
                            )),
                        )
                        .await;
                        return;
                    }
                    if let Err(error) =
                        super::openai::validate_output(&result, output_schema.as_ref())
                    {
                        let _ = send_events(&send, frames.error(error)).await;
                        return;
                    }
                    match plan.decode(&result, &frames.id) {
                        Ok(output) => {
                            let events = frames.finish(&result, &output, constrained);
                            let _ = send_events(&send, events).await;
                        }
                        Err(error) => {
                            let events = frames.error(error);
                            let _ = send_events(&send, events).await;
                        }
                    }
                    return;
                }
                None => {
                    let _ = send_events(
                        &send,
                        frames.error(ApiError::new(
                            axum::http::StatusCode::BAD_GATEWAY,
                            "invalid_stream",
                            "native stream ended without a final result",
                        )),
                    )
                    .await;
                    return;
                }
            }
        }
    });
    let response = Sse::new(ReceiverStream::new(receive))
        .keep_alive(KeepAlive::default())
        .into_response();
    Ok(headers(
        response,
        &id,
        Some(if constrained {
            "validated-tool-buffer"
        } else {
            "native"
        }),
    ))
}

fn history(body: &Value, family: Family, plan: &Plan) -> Result<History, ApiError> {
    let history = match family {
        Family::Chat => chat_history(body, plan),
        Family::Messages => anthropic_history(body, plan),
        Family::Responses => response_history(body, plan),
    }?;
    history.validate()?;
    Ok(history)
}

fn chat_history(body: &Value, plan: &Plan) -> Result<History, ApiError> {
    let mut history = History::default();
    for item in array(body, "messages")? {
        fields(
            item,
            &["role", "content", "images", "tool_calls", "tool_call_id"],
        )?;
        let role = string(&item["role"])?;
        match role {
            "tool" => {
                fields(item, &["role", "content", "tool_call_id"])?;
                history.result(
                    string(&item["tool_call_id"])?,
                    plain_text(&item["content"], false)?,
                    false,
                )?;
            }
            "assistant" if item.get("tool_calls").is_some() => {
                fields(item, &["role", "content", "tool_calls"])?;
                let items = match item.get("tool_calls").filter(|v| !v.is_null()) {
                    Some(value) => value
                        .as_array()
                        .ok_or_else(|| invalid("tool_calls must be an array"))?
                        .as_slice(),
                    None => &[],
                };
                let mut calls = Vec::new();
                for call in items {
                    fields(call, &["id", "type", "function"])?;
                    if call["type"] != "function" {
                        return Err(invalid("only function tool calls are supported"));
                    }
                    fields(&call["function"], &["name", "arguments"])?;
                    calls.push(Call {
                        id: string(&call["id"])?.into(),
                        name: string(&call["function"]["name"])?.into(),
                        arguments: arguments(&call["function"]["arguments"])?,
                    });
                }
                let text = if item["content"].is_null() {
                    String::new()
                } else {
                    plain_text(&item["content"], false)?
                };
                if calls.is_empty() {
                    history.message(text_message("assistant", text))?;
                } else {
                    history.calls(plan, calls, text)?;
                }
            }
            "user" | "assistant" | "system" | "developer" => {
                fields(item, &["role", "content", "images"])?;
                let mut item = item.clone();
                if role == "developer" {
                    item["role"] = json!("system");
                }
                let message: ChatMessage = serde_json::from_value(item).map_err(|_| {
                    invalid("invalid chat content; use text or supported PNG/JPEG image parts")
                })?;
                history.message(message)?;
            }
            _ => {
                return Err(invalid(
                    "unsupported message role; use user, assistant, system/developer or tool with a matching call ID",
                ));
            }
        }
    }
    Ok(history)
}

fn plain_text(value: &Value, anthropic: bool) -> Result<String, ApiError> {
    if let Some(text) = value.as_str() {
        bounded_text(text, tools::MAX_TOOL_OUTPUT)?;
        return Ok(text.into());
    }
    let parts = value.as_array().ok_or_else(|| {
        invalid("tool results/call text must be a string or text-only content blocks")
    })?;
    if parts.len() > 512 {
        return Err(invalid("too many text content blocks"));
    }
    let mut result = String::new();
    for part in parts {
        fields(
            part,
            if anthropic {
                &["type", "text", "cache_control"]
            } else {
                &["type", "text", "annotations"]
            },
        )?;
        if !matches!(
            string(&part["type"])?,
            "text" | "input_text" | "output_text"
        ) {
            return Err(invalid("tool results support text blocks only"));
        }
        cache_hint(part.get("cache_control"))?;
        if part
            .get("annotations")
            .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
        {
            return Err(invalid("nonempty annotations/citations are unsupported"));
        }
        result.push_str(string(&part["text"])?);
        bounded_text(&result, tools::MAX_TOOL_OUTPUT)?;
    }
    Ok(result)
}

fn anthropic_parts(content: &Value, system: bool) -> Result<Value, ApiError> {
    if content.is_string() {
        return Ok(content.clone());
    }
    let blocks = content
        .as_array()
        .ok_or_else(|| invalid("Anthropic content must be text or content blocks"))?;
    if blocks.len() > 512 {
        return Err(invalid("at most 512 content blocks are supported"));
    }
    let mut parts = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => {
                fields(block, &["type", "text", "cache_control", "citations"])?;
                cache_hint(block.get("cache_control"))?;
                if block
                    .get("citations")
                    .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
                {
                    return Err(invalid("nonempty citations are unsupported"));
                }
                parts.push(json!({"type":"text","text":string(&block["text"])?}));
            }
            Some("image") if !system => {
                fields(block, &["type", "source", "cache_control"])?;
                cache_hint(block.get("cache_control"))?;
                let source = &block["source"];
                fields(source, &["type", "media_type", "data"])?;
                if source["type"] != "base64" {
                    return Err(invalid(
                        "image sources must be base64 PNG/JPEG; remote URLs are not fetched",
                    ));
                }
                let mime = string(&source["media_type"])?;
                if !matches!(mime, "image/png" | "image/jpeg") {
                    return Err(invalid("only PNG/JPEG images are supported"));
                }
                parts.push(json!({"type":"image_url","image_url":{"url":format!("data:{mime};base64,{}",string(&source["data"])? )}}));
            }
            _ => {
                return Err(invalid(
                    "unsupported Anthropic block; use text, base64 image, tool_use or matching text tool_result blocks; cloud thinking/server-tool blocks are unavailable",
                ));
            }
        }
    }
    Ok(json!(parts))
}

fn anthropic_history(body: &Value, plan: &Plan) -> Result<History, ApiError> {
    let mut history = History::default();
    if let Some(system) = body.get("system")
        && !system.is_null()
    {
        let content = anthropic_parts(system, true)?;
        let message = serde_json::from_value(json!({"role":"system","content":content}))
            .map_err(|_| invalid("invalid Anthropic system content"))?;
        history.message(message)?;
    }
    for item in array(body, "messages")? {
        fields(item, &["role", "content"])?;
        let role = string(&item["role"])?;
        if !matches!(role, "user" | "assistant") {
            return Err(invalid(
                "Anthropic roles must be user or assistant; use system for instructions",
            ));
        }
        if let Some(blocks) = item["content"].as_array() {
            if blocks.len() > 512 {
                return Err(invalid("at most 512 content blocks are supported"));
            }
            let mut ordinary = Vec::new();
            let mut calls = Vec::new();
            let mut had_result = false;
            for block in blocks {
                match block["type"].as_str() {
                    Some("tool_use") if role == "assistant" => {
                        fields(block, &["type", "id", "name", "input", "cache_control"])?;
                        cache_hint(block.get("cache_control"))?;
                        calls.push(Call {
                            id: string(&block["id"])?.into(),
                            name: string(&block["name"])?.into(),
                            arguments: block["input"].clone(),
                        });
                    }
                    Some("tool_result") if role == "user" => {
                        if !ordinary.is_empty() {
                            return Err(invalid(
                                "tool_result blocks must precede ordinary user content",
                            ));
                        }
                        fields(
                            block,
                            &[
                                "type",
                                "tool_use_id",
                                "content",
                                "is_error",
                                "cache_control",
                            ],
                        )?;
                        cache_hint(block.get("cache_control"))?;
                        let content = block
                            .get("content")
                            .map(|v| plain_text(v, true))
                            .transpose()?
                            .unwrap_or_default();
                        history.result(
                            string(&block["tool_use_id"])?,
                            content,
                            boolean(block, "is_error", false)?,
                        )?;
                        had_result = true;
                    }
                    _ => ordinary.push(block.clone()),
                }
            }
            if !calls.is_empty() {
                let text = plain_text(&json!(ordinary), true)?;
                history.calls(plan, calls, text)?;
            } else if !ordinary.is_empty() || !had_result {
                let content = anthropic_parts(&json!(ordinary), false)?;
                let message = serde_json::from_value(json!({"role":role,"content":content}))
                    .map_err(|_| invalid("invalid Anthropic content"))?;
                history.message(message)?;
            }
        } else {
            let content = anthropic_parts(&item["content"], false)?;
            let message = serde_json::from_value(json!({"role":role,"content":content}))
                .map_err(|_| invalid("invalid Anthropic content"))?;
            history.message(message)?;
        }
    }
    Ok(history)
}

fn response_content(value: &Value, role: &str) -> Result<Value, ApiError> {
    if value.is_string() {
        return Ok(value.clone());
    }
    let parts = value
        .as_array()
        .ok_or_else(|| invalid("Responses message content must be text or content parts"))?;
    if parts.len() > 512 {
        return Err(invalid("too many content parts"));
    }
    let mut normalized = Vec::new();
    for part in parts {
        match part["type"].as_str() {
            Some("input_text") | Some("output_text") => {
                fields(part, &["type", "text", "annotations"])?;
                if part["type"] == "output_text" && role != "assistant" {
                    return Err(invalid("output_text is only valid in assistant messages"));
                }
                if part
                    .get("annotations")
                    .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
                {
                    return Err(invalid("nonempty output annotations are unsupported"));
                }
                normalized.push(json!({"type":"text","text":string(&part["text"])?}));
            }
            Some("input_image") => {
                fields(part, &["type", "image_url", "detail"])?;
                if role != "user" {
                    return Err(invalid("only user messages may contain images"));
                }
                if part.get("detail").is_some_and(|v| v != "auto") {
                    return Err(invalid(
                        "only detail:auto is supported; native vision controls image resolution",
                    ));
                }
                normalized.push(
                    json!({"type":"image_url","image_url":{"url":string(&part["image_url"])?}}),
                );
            }
            _ => {
                return Err(invalid(
                    "unsupported Responses content; use input_text/output_text or PNG/JPEG data-URL input_image",
                ));
            }
        }
    }
    Ok(json!(normalized))
}
fn response_history(body: &Value, plan: &Plan) -> Result<History, ApiError> {
    let mut history = History::default();
    if let Some(instructions) = body.get("instructions")
        && !instructions.is_null()
    {
        history.message(text_message("system", string(instructions)?.into()))?;
    }
    if let Some(input) = body["input"].as_str() {
        history.message(text_message("user", input.into()))?;
        return Ok(history);
    }
    for item in array(body, "input")? {
        match item
            .get("type")
            .map(string)
            .transpose()?
            .unwrap_or("message")
        {
            "message" => {
                fields(item, &["type", "id", "role", "content", "status"])?;
                optional_id(item, "id")?;
                if let Some(status) = item.get("status")
                    && !matches!(string(status)?, "completed" | "incomplete")
                {
                    return Err(invalid(
                        "only completed/incomplete historical message status is supported",
                    ));
                }
                let role = string(&item["role"])?;
                if !matches!(role, "user" | "assistant" | "system" | "developer") {
                    return Err(invalid(
                        "Responses tool results must use function_call_output with a matching call_id",
                    ));
                }
                let content = response_content(&item["content"], role)?;
                let message = serde_json::from_value(
                    json!({"role":if role=="developer" {"system"} else {role},"content":content}),
                )
                .map_err(|_| invalid("invalid Responses content"))?;
                history.message(message)?;
            }
            "function_call" => {
                fields(
                    item,
                    &["type", "id", "call_id", "name", "arguments", "status"],
                )?;
                optional_id(item, "id")?;
                if item.get("status").is_some_and(|v| v != "completed") {
                    return Err(invalid(
                        "incomplete historical function calls cannot be executed or continued",
                    ));
                }
                history.adjacent_call(
                    plan,
                    Call {
                        id: string(&item["call_id"])?.into(),
                        name: string(&item["name"])?.into(),
                        arguments: arguments(&item["arguments"])?,
                    },
                )?;
            }
            "function_call_output" => {
                fields(item, &["type", "id", "call_id", "output", "status"])?;
                optional_id(item, "id")?;
                if item.get("status").is_some_and(|v| v != "completed") {
                    return Err(invalid("function_call_output must be complete"));
                }
                history.result(
                    string(&item["call_id"])?,
                    plain_text(&item["output"], false)?,
                    false,
                )?;
            }
            "reasoning" => {
                fields(
                    item,
                    &["type", "id", "summary", "encrypted_content", "status"],
                )?;
                optional_id(item, "id")?;
                if item.get("encrypted_content").is_some_and(|v| !v.is_null()) {
                    return Err(invalid(
                        "encrypted cloud reasoning state is unsupported; send stateless public message/function history",
                    ));
                }
                let summary = item["summary"]
                    .as_array()
                    .ok_or_else(|| invalid("public reasoning history needs a summary array"))?;
                if summary.len() > 64 {
                    return Err(invalid("too many public reasoning summary parts"));
                }
                let mut text = String::new();
                for part in summary {
                    fields(part, &["type", "text"])?;
                    if part["type"] != "summary_text" {
                        return Err(invalid(
                            "only public summary_text reasoning history is supported",
                        ));
                    }
                    text.push_str(string(&part["text"])?);
                }
                if !text.is_empty() {
                    history.message(text_message("assistant", text))?;
                }
            }
            _ => {
                return Err(invalid(
                    "unsupported Responses item; use message, function_call, function_call_output or unencrypted public reasoning summary",
                ));
            }
        }
    }
    Ok(history)
}

/// Exact served-GGUF template count, not an Anthropic proprietary tokenizer.
pub(super) async fn count_tokens(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = payload.map_err(ApiError::from_json)?;
    fields(
        &body,
        &[
            "model",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "metadata",
            "cache_control",
        ],
    )?;
    metadata(body.get("metadata"), true)?;
    cache_hint(body.get("cache_control"))?;
    validate_model(&state, Some(string(&body["model"])?))?;
    let plan = Plan::parse(
        body.get("tools"),
        body.get("tool_choice"),
        Family::Messages,
        true,
    )?;
    let history = anthropic_history(&body, &plan)?;
    history.validate()?;
    history.reject_images()?;
    let has_definition = state.definition.is_some();
    let explicit_system = history
        .messages
        .iter()
        .any(|message| message.role == "system");
    let tokens = state
        .inference
        .count_prompt_prepared(move || {
            let prompt = plan.prepare(history)?;
            let prompt = match prompt {
                Prompt::Constrained { prompt, .. } => *prompt,
                prompt => prompt,
            };
            Ok(if has_definition {
                prompt.with_request_system(explicit_system)
            } else {
                prompt
            })
        })
        .await?;
    let (id, _) = identifier(&state, "req");
    let mut response = headers(
        Json(json!({"input_tokens":tokens})).into_response(),
        &id,
        None,
    );
    response.headers_mut().insert(
        "x-lma-tokenizer",
        "served-gguf-chat-template".parse().unwrap(),
    );
    Ok(response)
}

struct Frames {
    family: Family,
    id: String,
    created: u64,
    model: String,
    sequence: u64,
    include_usage: bool,
    parallel: bool,
    metadata: Value,
}
impl Frames {
    fn event(&mut self, kind: &str, mut value: Value) -> Event {
        if self.family == Family::Responses {
            value["type"] = json!(kind);
            value["sequence_number"] = json!(self.sequence);
            self.sequence += 1;
        }
        let event = Event::default().data(value.to_string());
        if self.family == Family::Chat {
            event
        } else {
            event.event(kind)
        }
    }
    fn chunk(&mut self, delta: Value, finish: Value) -> Event {
        let value = json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
        self.event("", value)
    }
    fn usage(&self, result: &Generation) -> Value {
        if self.family == Family::Chat {
            json!({"prompt_tokens":result.prompt_tokens,"completion_tokens":result.completion_tokens,"total_tokens":result.prompt_tokens+result.completion_tokens})
        } else {
            json!({"input_tokens":result.prompt_tokens,"output_tokens":result.completion_tokens,"total_tokens":result.prompt_tokens+result.completion_tokens})
        }
    }
    fn item_id(&self, index: usize) -> String {
        format!("{}_item_{index}", self.id)
    }
    fn text_part(text: &str) -> Value {
        json!({"type":"output_text","text":text,"annotations":[]})
    }
    fn response_items(&self, output: &Output) -> Vec<Value> {
        let mut items = Vec::new();
        if !output.text.is_empty() || output.calls.is_empty() {
            items.push(json!({"id":self.item_id(0),"type":"message","role":"assistant","status":"completed","content":[Self::text_part(&output.text)]}));
        }
        for call in &output.calls {
            items.push(json!({"id":self.item_id(items.len()),"type":"function_call","call_id":call.id,"name":call.name,"arguments":call.arguments.to_string(),"status":"completed"}));
        }
        items
    }
    fn complete(&self, result: &Generation, output: &Output) -> Value {
        let length = result.finish_reason == "length";
        match self.family {
            Family::Chat => {
                let mut message = json!({"role":"assistant","content":if output.text.is_empty()&&!output.calls.is_empty() {Value::Null}else{json!(output.text)}});
                if !output.calls.is_empty() {
                    message["tool_calls"]=json!(output.calls.iter().map(|call|json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments.to_string()}})).collect::<Vec<_>>());
                }
                json!({"id":self.id,"object":"chat.completion","created":self.created,"model":self.model,"choices":[{"index":0,"message":message,"finish_reason":if !output.calls.is_empty(){"tool_calls"}else if length{"length"}else{"stop"}}],"usage":self.usage(result)})
            }
            Family::Messages => {
                let mut content = Vec::new();
                if !output.text.is_empty() || output.calls.is_empty() {
                    content.push(json!({"type":"text","text":output.text}));
                }
                for call in &output.calls {
                    content.push(json!({"type":"tool_use","id":call.id,"name":call.name,"input":call.arguments}));
                }
                json!({"id":self.id,"type":"message","role":"assistant","model":self.model,"content":content,"stop_reason":if !output.calls.is_empty(){"tool_use"}else if length{"max_tokens"}else{"end_turn"},"stop_sequence":null,"usage":{"input_tokens":result.prompt_tokens,"output_tokens":result.completion_tokens}})
            }
            Family::Responses => {
                let mut output_items = self.response_items(output);
                if length {
                    for item in &mut output_items {
                        item["status"] = json!("incomplete");
                    }
                }
                json!({"id":self.id,"object":"response","created_at":self.created,"status":if length{"incomplete"}else{"completed"},"model":self.model,"output":output_items,"error":null,"incomplete_details":if length{json!({"reason":"max_output_tokens"})}else{Value::Null},"usage":self.usage(result),"store":false,"metadata":self.metadata,"parallel_tool_calls":self.parallel})
            }
        }
    }
    fn start(&mut self, constrained: bool) -> Vec<Event> {
        match self.family {
            Family::Chat => vec![self.chunk(json!({"role":"assistant"}), Value::Null)],
            Family::Messages => {
                let mut events=vec![self.event("message_start",json!({"type":"message_start","message":{"id":self.id,"type":"message","role":"assistant","model":self.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}))];
                if !constrained {
                    events.push(self.event("content_block_start",json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})));
                }
                events
            }
            Family::Responses => {
                let pending = json!({"id":self.id,"object":"response","created_at":self.created,"status":"in_progress","model":self.model,"output":[],"usage":null,"error":null,"incomplete_details":null,"store":false});
                let mut events = vec![
                    self.event("response.created", json!({"response":pending})),
                    self.event("response.in_progress", json!({"response":pending})),
                ];
                if !constrained {
                    events.extend(self.text_start(0));
                }
                events
            }
        }
    }
    fn text_start(&mut self, index: usize) -> Vec<Event> {
        let id = self.item_id(index);
        vec![self.event("response.output_item.added",json!({"output_index":index,"item":{"id":id,"type":"message","role":"assistant","status":"in_progress","content":[]}})),self.event("response.content_part.added",json!({"item_id":id,"output_index":index,"content_index":0,"part":Self::text_part("")}))]
    }
    fn delta(&mut self, text: &str) -> Vec<Event> {
        vec![match self.family {
            Family::Chat=>self.chunk(json!({"content":text}),Value::Null),
            Family::Messages=>self.event("content_block_delta",json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}})),
            Family::Responses=>self.event("response.output_text.delta",json!({"item_id":self.item_id(0),"output_index":0,"content_index":0,"delta":text})),
        }]
    }
    fn finish(&mut self, result: &Generation, output: &Output, constrained: bool) -> Vec<Event> {
        let complete = self.complete(result, output);
        let mut events = Vec::new();
        match self.family {
            Family::Chat => {
                if constrained {
                    if !output.text.is_empty() {
                        events.push(self.chunk(json!({"content":output.text}), Value::Null));
                    }
                    if !output.calls.is_empty() {
                        events.push(self.chunk(json!({"tool_calls":output.calls.iter().enumerate().map(|(index,call)|json!({"index":index,"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments.to_string()}})).collect::<Vec<_>>()}),Value::Null));
                    }
                }
                events.push(self.chunk(json!({}), complete["choices"][0]["finish_reason"].clone()));
                if self.include_usage {
                    events.push(self.event("",json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":[],"usage":self.usage(result)})));
                }
                events.push(Event::default().data("[DONE]"));
            }
            Family::Messages => {
                if constrained {
                    for (index, block) in complete["content"].as_array().unwrap().iter().enumerate()
                    {
                        let mut start = block.clone();
                        let delta = if block["type"] == "tool_use" {
                            start["input"] = json!({});
                            json!({"type":"input_json_delta","partial_json":block["input"].to_string()})
                        } else {
                            start["text"] = json!("");
                            json!({"type":"text_delta","text":block["text"]})
                        };
                        events.push(self.event("content_block_start",json!({"type":"content_block_start","index":index,"content_block":start})));
                        events.push(self.event(
                            "content_block_delta",
                            json!({"type":"content_block_delta","index":index,"delta":delta}),
                        ));
                        events.push(self.event(
                            "content_block_stop",
                            json!({"type":"content_block_stop","index":index}),
                        ));
                    }
                } else {
                    events.push(self.event(
                        "content_block_stop",
                        json!({"type":"content_block_stop","index":0}),
                    ));
                }
                events.push(self.event("message_delta",json!({"type":"message_delta","delta":{"stop_reason":complete["stop_reason"],"stop_sequence":null},"usage":{"input_tokens":result.prompt_tokens,"output_tokens":result.completion_tokens}})));
                events.push(self.event("message_stop", json!({"type":"message_stop"})));
            }
            Family::Responses => {
                for (index, item) in complete["output"].as_array().unwrap().iter().enumerate() {
                    let id = item["id"].clone();
                    if item["type"] == "message" {
                        if constrained {
                            events.extend(self.text_start(index));
                            events.push(self.event("response.output_text.delta",json!({"item_id":id,"output_index":index,"content_index":0,"delta":output.text})));
                        }
                        events.push(self.event("response.output_text.done",json!({"item_id":id,"output_index":index,"content_index":0,"text":output.text})));
                        events.push(self.event("response.content_part.done",json!({"item_id":id,"output_index":index,"content_index":0,"part":Self::text_part(&output.text)})));
                    } else {
                        let mut pending = item.clone();
                        pending["arguments"] = json!("");
                        pending["status"] = json!("in_progress");
                        events.push(self.event(
                            "response.output_item.added",
                            json!({"output_index":index,"item":pending}),
                        ));
                        events.push(self.event(
                            "response.function_call_arguments.delta",
                            json!({"item_id":id,"output_index":index,"delta":item["arguments"]}),
                        ));
                        events.push(self.event("response.function_call_arguments.done",json!({"item_id":id,"output_index":index,"arguments":item["arguments"]})));
                    }
                    events.push(self.event(
                        "response.output_item.done",
                        json!({"output_index":index,"item":item}),
                    ));
                }
                events.push(self.event(
                    if result.finish_reason == "length" {
                        "response.incomplete"
                    } else {
                        "response.completed"
                    },
                    json!({"response":complete}),
                ));
            }
        }
        events
    }
    fn error(&mut self, error: ApiError) -> Vec<Event> {
        vec![self.event("error", error.stream_value(&self.id))]
    }
}
async fn send_events(send: &mpsc::Sender<Result<Event, Infallible>>, events: Vec<Event>) -> bool {
    for event in events {
        if send.send(Ok(event)).await.is_err() {
            return false;
        }
    }
    true
}
fn identifier(state: &AppState, prefix: &str) -> (String, u64) {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sequence = state.next_id.fetch_add(1, Ordering::Relaxed);
    (format!("{prefix}-{created}-{sequence}"), created)
}
#[cfg(test)]
mod budget_tests {
    use super::super::{
        generation_controls::{GenerationControls, diagnostic_bytes},
        inference::{Engine, InferenceService, TokenLogprob},
        openai::Candidates,
    };
    use super::*;
    use http_body_util::BodyExt;
    use std::sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize},
    };

    struct BudgetEngine {
        calls: Arc<AtomicUsize>,
        discards: Arc<AtomicUsize>,
    }

    impl Engine for BudgetEngine {
        fn generate(
            &mut self,
            request: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let Prompt::Controls { controls, .. } = request.prompt else {
                panic!("text_batch did not attach its budget");
            };
            let budget = controls.output_budget.unwrap();
            budget.charge(2)?;
            let logprobs = if controls.logprobs.is_some() {
                let mut records = Vec::new();
                for _ in 0..2 {
                    budget.charge(diagnostic_bytes(1, 1))?;
                    records.push(TokenLogprob {
                        token: "x".into(),
                        bytes: vec![b'x'],
                        logprob: -1.0,
                        top_logprobs: vec![],
                    });
                }
                Some(records)
            } else {
                None
            };
            Ok(Generation {
                text: "xx".into(),
                prompt_tokens: 1,
                completion_tokens: 2,
                finish_reason: "length",
                logprobs,
                mean_logprob: Some(-1.0),
            })
        }

        fn discard_request_state(&mut self) {
            self.discards.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn state(inference: InferenceService) -> AppState {
        AppState {
            inference,
            model_id: "fixture".into(),
            context: 1024,
            metadata: None,
            definition: None,
            api_key: None,
            next_id: Arc::new(AtomicU64::new(1)),
            api: "all".into(),
        }
    }

    fn batch(work: usize, stream: bool, diagnostics: bool) -> TextBatch {
        TextBatch {
            candidates: Candidates {
                n: work,
                best_of: work,
            },
            work,
            stream,
            include_usage: true,
            diagnostics,
            format: BatchFormat::Legacy {
                echoes: vec![if diagnostics {
                    String::new()
                } else {
                    "echo".into()
                }],
            },
        }
    }

    fn requests(work: usize, diagnostics: bool) -> Vec<GenerationRequest> {
        (0..work)
            .map(|_| GenerationRequest {
                prompt: Prompt::Text("fixture".into()).with_controls(GenerationControls {
                    logprobs: diagnostics.then_some(0),
                    ..Default::default()
                }),
                options: Options {
                    until_context: false,
                    max_tokens: 2,
                    temperature: 0.0,
                    top_p: 1.0,
                    seed: None,
                },
            })
            .collect()
    }

    #[tokio::test]
    async fn output_budget_text_batch_reserves_echoes_and_errors_without_false_success() {
        for stream in [false, true] {
            for diagnostics in [false, true] {
                for work in [1, 3] {
                    let calls = Arc::new(AtomicUsize::new(0));
                    let discards = Arc::new(AtomicUsize::new(0));
                    let engine = BudgetEngine {
                        calls: calls.clone(),
                        discards: discards.clone(),
                    };
                    let (inference, worker) =
                        InferenceService::spawn(move || Ok(engine)).await.unwrap();
                    let per_candidate = 2 + if diagnostics {
                        2 * diagnostic_bytes(1, 1)
                    } else {
                        0
                    };
                    let echoes = if diagnostics { 0 } else { 4 * work };
                    let limit = echoes + per_candidate * work.min(2) - 1;
                    let result = text_batch_with_limit(
                        state(inference),
                        batch(work, stream, diagnostics),
                        move || Ok(requests(work, diagnostics)),
                        limit,
                    )
                    .await;
                    let response = match result {
                        Ok(response) => {
                            assert!(stream);
                            response
                        }
                        Err(error) => {
                            assert!(!stream);
                            error.into_response()
                        }
                    };
                    if !stream {
                        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
                    }
                    let body = tokio::time::timeout(
                        std::time::Duration::from_secs(2),
                        response.into_body().collect(),
                    )
                    .await
                    .unwrap()
                    .unwrap()
                    .to_bytes();
                    let body = std::str::from_utf8(&body).unwrap();
                    assert!(body.contains("resource_limit_exceeded"), "{body}");
                    if stream {
                        assert!(!body.contains("[DONE]"), "{body}");
                        assert!(
                            !body.contains("\"choices\":[]"),
                            "no successful final usage: {body}"
                        );
                    }
                    assert_eq!(calls.load(Ordering::SeqCst), work.min(2));
                    assert_eq!(discards.load(Ordering::SeqCst), 1);
                    tokio::time::timeout(std::time::Duration::from_secs(2), worker)
                        .await
                        .unwrap()
                        .unwrap();
                }
            }
        }
    }

    #[tokio::test]
    async fn output_budget_single_diagnostic_candidate_succeeds_at_exact_limit() {
        for stream in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let discards = Arc::new(AtomicUsize::new(0));
            let engine = BudgetEngine {
                calls: calls.clone(),
                discards: discards.clone(),
            };
            let (inference, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
            let response = text_batch_with_limit(
                state(inference),
                batch(1, stream, true),
                || Ok(requests(1, true)),
                2 + 2 * diagnostic_bytes(1, 1),
            )
            .await
            .unwrap();
            let body = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                response.into_body().collect(),
            )
            .await
            .unwrap()
            .unwrap()
            .to_bytes();
            let body = std::str::from_utf8(&body).unwrap();
            assert!(!body.contains("resource_limit_exceeded"), "{body}");
            assert!(body.contains("\"text\":\"xx\""), "{body}");
            assert!(body.contains("\"token_logprobs\":[-1.0,-1.0]"), "{body}");
            assert_eq!(body.contains("[DONE]"), stream);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(discards.load(Ordering::SeqCst), 0);
            tokio::time::timeout(std::time::Duration::from_secs(2), worker)
                .await
                .unwrap()
                .unwrap();
        }
    }

    #[tokio::test]
    async fn output_budget_echo_exhaustion_precedes_preparation_and_native_work() {
        let calls = Arc::new(AtomicUsize::new(0));
        let discards = Arc::new(AtomicUsize::new(0));
        let engine = BudgetEngine {
            calls: calls.clone(),
            discards: discards.clone(),
        };
        let (inference, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        let result = text_batch_with_limit(
            state(inference),
            batch(2, false, false),
            || panic!("over-budget echoes prepared"),
            7,
        )
        .await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        worker.await.unwrap();
    }
}

fn headers(mut response: Response, id: &str, streaming: Option<&str>) -> Response {
    response
        .headers_mut()
        .insert("x-request-id", id.parse().expect("server-generated ID"));
    response.headers_mut().insert("x-lma-advisory","metadata-not-stored; cache-hints-not-applied; reasoning-effort-summary-verbosity-hints-ignored".parse().unwrap());
    if let Some(mode) = streaming {
        response
            .headers_mut()
            .insert("x-lma-streaming", mode.parse().unwrap());
    }
    response
}
