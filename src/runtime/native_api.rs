//! Versioned lma wire API. All model work uses the shared admitted native worker.
use std::{convert::Infallible, sync::atomic::Ordering};

use axum::{
    Json,
    body::{Body, Bytes},
    extract::{State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    inference::{Generation, GenerationEvent, GenerationRequest, InferenceError, Options, Prompt},
    server::{self, ApiError, AppState, ChatMessage},
};

#[derive(Clone, Copy, Default)]
enum TokenLimit {
    #[default]
    Default,
    Tokens(u32),
    Context,
}
impl<'de> Deserialize<'de> for TokenLimit {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match Option::<u32>::deserialize(deserializer)? {
            Some(tokens) => Self::Tokens(tokens),
            None => Self::Context,
        })
    }
}

// Null is a deliberate budget setting only; other nullable options are rejected.
fn present<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestBody {
    #[serde(default, deserialize_with = "present")]
    model: Option<String>,
    #[serde(default, deserialize_with = "present")]
    prompt: Option<String>,
    #[serde(default, deserialize_with = "present")]
    messages: Option<Vec<ChatMessage>>,
    #[serde(default)]
    max_tokens: TokenLimit,
    #[serde(default, deserialize_with = "present")]
    temperature: Option<f32>,
    #[serde(default, deserialize_with = "present")]
    top_p: Option<f32>,
    #[serde(default, deserialize_with = "present")]
    seed: Option<u32>,
    #[serde(default)]
    stream: bool,
}

enum Input {
    Text(String),
    Chat(Vec<ChatMessage>),
}
struct Prepared {
    input: Input,
    options: Options,
}
impl Prepared {
    fn request(self) -> Result<GenerationRequest, InferenceError> {
        let prompt = match self.input {
            Input::Text(text) => Prompt::Text(text),
            Input::Chat(messages) => server::prepare_chat_prompt(messages)?,
        };
        Ok(GenerationRequest {
            prompt,
            options: self.options,
        })
    }
}
impl RequestBody {
    fn validate_input(&self, chat: bool) -> Result<(), ApiError> {
        match (chat, &self.prompt, &self.messages) {
            (false, Some(prompt), None) if !prompt.is_empty() => Ok(()),
            (true, None, Some(messages)) if !messages.is_empty() => {
                server::validate_chat_messages(messages)
            }
            _ => Err(ApiError::invalid(if chat {
                "chat requires nonempty messages and no prompt"
            } else {
                "generate requires a nonempty prompt and no messages"
            })),
        }
    }

    fn apply_options(&self, mut options: Options) -> Options {
        match self.max_tokens {
            TokenLimit::Default => {}
            TokenLimit::Tokens(tokens) => options.max_tokens = tokens,
            TokenLimit::Context => {
                options.until_context = true;
                // Real allowance is computed after formatting and native image positions.
                options.max_tokens = 1;
            }
        }
        if let Some(value) = self.temperature {
            options.temperature = value;
        }
        if let Some(value) = self.top_p {
            options.top_p = value;
        }
        if let Some(value) = self.seed {
            options.seed = Some(value);
        }
        options
    }

    fn prepare(self, state: &AppState, chat: bool) -> Result<(Prepared, bool), ApiError> {
        self.validate_input(chat)?;
        let options = self.apply_options(state.generation_defaults());
        server::validate_options(state, self.model.as_deref(), false, &options)?;
        let input = if chat {
            Input::Chat(self.messages.expect("validated chat input"))
        } else {
            Input::Text(self.prompt.expect("validated raw input"))
        };
        Ok((Prepared { input, options }, self.stream))
    }
}

fn parse(body: Value) -> Result<RequestBody, ApiError> {
    serde_json::from_value(body).map_err(|_| {
        ApiError::invalid(
            "invalid lma v1 request; use documented fields and types (only max_tokens may be null)",
        )
    })
}

/// Validate intrinsic controls before the managed host can switch resident models.
pub(super) fn validate_body(body: &Value, chat: bool) -> Result<(), ApiError> {
    let request = parse(body.clone())?;
    request.validate_input(chat)?;
    request
        .apply_options(Options {
            until_context: false,
            max_tokens: 1,
            temperature: server::default_temperature(),
            top_p: server::default_top_p(),
            seed: None,
        })
        .validate(u32::MAX)?;
    Ok(())
}

pub(super) async fn generate(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    respond(state, payload.map_err(ApiError::from_json)?.0, false).await
}

pub(super) async fn chat(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    respond(state, payload.map_err(ApiError::from_image_json)?.0, true).await
}

async fn respond(state: AppState, body: Value, chat: bool) -> Result<Response, ApiError> {
    let body = parse(body)?;
    let (prepared, stream) = body.prepare(&state, chat)?;
    server::ensure_available(&state)?;
    let until_context = prepared.options.until_context;
    let id = format!("lma-{}", state.next_id.fetch_add(1, Ordering::Relaxed));
    if stream {
        let native = state
            .inference
            .stream_prepared(move || prepared.request())
            .await?;
        Ok(stream_response(native, id, state.model_id, until_context))
    } else {
        let result = state
            .inference
            .generate_prepared(move || prepared.request())
            .await?;
        let mut value = final_value(&id, &state.model_id, &result, until_context)?;
        value["text"] = json!(result.text);
        Ok(Json(value).into_response())
    }
}

fn final_value(
    id: &str,
    model: &str,
    result: &Generation,
    until_context: bool,
) -> Result<Value, ApiError> {
    let reason = match result.finish_reason {
        "stop" => "eos",
        "length" if until_context => "context_limit",
        "length" => "token_limit",
        _ => {
            return Err(InferenceError::Internal(
                "lma generation returned an unsupported finish reason".into(),
            )
            .into());
        }
    };
    Ok(json!({
        "api_version":1, "id":id, "model":model,
        "usage":{"input_tokens":result.prompt_tokens,"output_tokens":result.completion_tokens},
        "finish_reason":reason,
    }))
}

fn record(value: Value) -> Result<Bytes, Infallible> {
    let mut bytes = serde_json::to_vec(&value).expect("JSON value serializes");
    bytes.push(b'\n');
    Ok(Bytes::from(bytes))
}

fn stream_response(
    mut native: mpsc::Receiver<GenerationEvent>,
    id: String,
    model: String,
    until_context: bool,
) -> Response {
    let (send, receive) = mpsc::channel::<Result<Bytes, Infallible>>(8);
    tokio::spawn(async move {
        if send
            .send(record(
                json!({"type":"start","api_version":1,"id":id,"model":model}),
            ))
            .await
            .is_err()
        {
            return;
        }
        let mut emitted = String::new();
        loop {
            let event = tokio::select! {
                _ = send.closed() => return,
                event = native.recv() => event,
            };
            let error = match event {
                Some(GenerationEvent::Delta(text)) => {
                    emitted.push_str(&text);
                    if send
                        .send(record(json!({"type":"delta","id":id,"text":text})))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    continue;
                }
                Some(GenerationEvent::Finished(Ok(result))) if result.text == emitted => {
                    match final_value(&id, &model, &result, until_context) {
                        Ok(mut value) => {
                            value["type"] = json!("done");
                            let _ = send.send(record(value)).await;
                            return;
                        }
                        Err(error) => error,
                    }
                }
                Some(GenerationEvent::Finished(Ok(_))) => ApiError::new(
                    StatusCode::BAD_GATEWAY,
                    "invalid_stream",
                    "native stream text did not match its final result",
                ),
                Some(GenerationEvent::Finished(Err(error))) => error.into(),
                None => ApiError::new(
                    StatusCode::BAD_GATEWAY,
                    "invalid_stream",
                    "native stream ended without a final result",
                ),
            };
            let mut value = error.stream_value(&id);
            value["api_version"] = json!(1);
            let _ = send.send(record(value)).await;
            return;
        }
    });
    (
        [
            (header::CONTENT_TYPE, "application/x-ndjson"),
            (header::CACHE_CONTROL, "no-store"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        Body::from_stream(ReceiverStream::new(receive)),
    )
        .into_response()
}

pub(super) async fn status(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    server::ensure_available(&state)?;
    let defaults = state.generation_defaults();
    Ok(Json(json!({
        "api_version":1,
        "server":{"name":"lma","version":env!("CARGO_PKG_VERSION")},
        "model":state.model_id,
        "configured_context_tokens":state.context,
        "defaults":{"max_tokens":defaults.max_tokens,"temperature":defaults.temperature,"top_p":defaults.top_p,"seed":defaults.seed},
        "stream_format":"ndjson",
    })))
}

#[cfg(test)]
mod tests {
    use super::super::inference::{Engine, InferenceService};
    use super::*;
    use axum::{Router, body::Body, http::Request};
    use http_body_util::BodyExt;
    use std::sync::{Arc, Mutex, atomic::AtomicU64};
    use tower::ServiceExt;

    struct Fixture(Arc<Mutex<Vec<GenerationRequest>>>);
    impl Engine for Fixture {
        fn generate(
            &mut self,
            request: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            self.0.lock().unwrap().push(request);
            Ok(Generation {
                text: "hello 界\nnext".into(),
                prompt_tokens: 7,
                completion_tokens: 3,
                finish_reason: "length",
            })
        }
    }
    async fn app(
        api: &str,
        key: Option<&str>,
    ) -> (
        Router,
        Arc<Mutex<Vec<GenerationRequest>>>,
        tokio::task::JoinHandle<()>,
    ) {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let engine = captured.clone();
        let (inference, worker) = InferenceService::spawn(move || Ok(Fixture(engine)))
            .await
            .unwrap();
        let app = server::router(AppState {
            inference,
            model_id: "test".into(),
            context: 512,
            metadata: None,
            definition: None,
            api_key: key.map(str::to_owned),
            next_id: Arc::new(AtomicU64::new(1)),
            api: api.into(),
        });
        (app, captured, worker)
    }
    fn request(path: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }
    async fn value(response: Response) -> Value {
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
    }
    #[tokio::test]
    async fn chat_preflight_and_standalone_route_reject_invalid_content_without_generation() {
        let (app, captured, worker) = app("lma", None).await;
        for message in [
            json!({"role":"user","content":""}),
            json!({"role":"user","content":[]}),
            json!({"role":"tool","content":"result"}),
            json!({"role":"user","content":"bad\0text"}),
            json!({"role":"system","content":"x","images":["not-base64"]}),
            json!({"role":"assistant","content":[{"type":"image_url","image_url":{"url":"invalid"}}]}),
            json!({"role":"user","content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"invalid"}}]}),
            json!({"role":"user","content":[{"type":"text","text":"<__lma_"},{"type":"text","text":"image_0>"}]}),
        ] {
            let body = json!({"messages":[message]});
            assert!(validate_body(&body, true).is_err(), "{body}");
            let response = app
                .clone()
                .oneshot(request("/lma/v1/chat", body))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        assert!(captured.lock().unwrap().is_empty());
        // Shape preflight must not decode or inspect image payloads.
        assert!(
            validate_body(
                &json!({"messages":[{"role":"user","content":"","images":["not-base64"]}]}),
                true
            )
            .is_ok()
        );
        assert!(validate_body(&json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"invalid"}}]}]}), true).is_ok());
        drop(app);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn native_generation_budget_defaults_context_and_chat_are_explicit() {
        let (app, captured, worker) = app("lma", None).await;
        for (path, body, unlimited, expected) in [
            ("/lma/v1/generate", json!({"prompt":"raw"}), false, 256),
            (
                "/lma/v1/generate",
                json!({"prompt":"raw","max_tokens":64}),
                false,
                64,
            ),
            (
                "/lma/v1/generate",
                json!({"prompt":"raw","max_tokens":null}),
                true,
                1,
            ),
            (
                "/lma/v1/chat",
                json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":null}),
                true,
                1,
            ),
        ] {
            let response = app.clone().oneshot(request(path, body)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let response = value(response).await;
            assert_eq!(response["api_version"], 1);
            assert_eq!(response["text"], "hello 界\nnext");
            assert_eq!(response["usage"]["input_tokens"], 7);
            assert_eq!(response["usage"]["output_tokens"], 3);
            assert_eq!(
                response["finish_reason"],
                if unlimited {
                    "context_limit"
                } else {
                    "token_limit"
                }
            );
            let captured = captured.lock().unwrap();
            let request = captured.last().unwrap();
            assert_eq!(request.options.until_context, unlimited);
            assert_eq!(request.options.max_tokens, expected);
            assert_eq!(
                matches!(request.prompt, Prompt::Chat(_)),
                path.ends_with("/chat")
            );
        }
        drop(app);
        worker.await.unwrap();
    }
    #[tokio::test]
    async fn native_validation_never_silently_ignores_controls_or_echoes_secrets() {
        let (app, captured, worker) = app("lma", None).await;
        for body in [
            json!({"prompt":""}),
            json!({"messages":[]}),
            json!({"prompt":"x","messages":[]}),
            json!({"prompt":"x","max_tokens":0}),
            json!({"prompt":"x","max_tokens":-1}),
            json!({"prompt":"x","max_tokens":513}),
            json!({"prompt":"x","max_tokens":"off"}),
            json!({"prompt":"x","temperature":null}),
            json!({"prompt":"x","stream":null}),
            json!({"prompt":"x","model":"wrong"}),
            json!({"prompt":"x","top_p":0}),
            json!({"prompt":"x","tools":[{"secret":"must-not-echo"}]}),
        ] {
            let response = app
                .clone()
                .oneshot(request("/lma/v1/generate", body))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(!value(response).await.to_string().contains("must-not-echo"));
        }
        assert!(captured.lock().unwrap().is_empty());
        drop(app);
        worker.await.unwrap();
    }
    #[tokio::test]
    async fn ndjson_contains_unicode_deltas_and_one_done_without_duplicate_text() {
        let (app, _, worker) = app("lma", None).await;
        let response = app
            .clone()
            .oneshot(request(
                "/lma/v1/generate",
                json!({"prompt":"x","stream":true,"max_tokens":null}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/x-ndjson"
        );
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let wire = std::str::from_utf8(&bytes).unwrap();
        let frames: Vec<Value> = wire
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            frames
                .iter()
                .map(|f| f["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["start", "delta", "done"]
        );
        assert_eq!(frames[1]["text"], "hello 界\nnext");
        assert_eq!(frames[2]["finish_reason"], "context_limit");
        assert!(frames[2].get("text").is_none());
        assert_eq!(frames[0]["id"], frames[2]["id"]);
        drop(app);
        worker.await.unwrap();
    }
    #[tokio::test]
    async fn native_auth_status_profiles_and_limits_share_server_policy() {
        let (app, captured, worker) = app("lma", Some("key")).await;
        for auth in [None, Some("Bearer wrong")] {
            let mut req = request("/lma/v1/generate", json!({"prompt":"x"}));
            if let Some(auth) = auth {
                req.headers_mut()
                    .insert(header::AUTHORIZATION, auth.parse().unwrap());
            }
            assert_eq!(
                app.clone().oneshot(req).await.unwrap().status(),
                StatusCode::UNAUTHORIZED
            );
        }
        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/lma/v1/status")
                    .header(header::AUTHORIZATION, "Bearer key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(status.status(), StatusCode::OK);
        let status = value(status).await;
        assert_eq!(status["configured_context_tokens"], 512);
        assert_eq!(status["server"]["version"], env!("CARGO_PKG_VERSION"));
        assert!(status.get("api_key").is_none());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/lma/v1/generate")
                    .header(header::AUTHORIZATION, "Bearer key")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(" ".repeat(server::BODY_LIMIT + 1)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(captured.lock().unwrap().is_empty());
        drop(app);
        worker.await.unwrap();
    }
    #[tokio::test]
    async fn stream_errors_and_disagreements_never_produce_a_successful_done() {
        for events in [
            vec![
                GenerationEvent::Delta("partial".into()),
                GenerationEvent::Finished(Err(InferenceError::Invalid("rejected".into()))),
            ],
            vec![GenerationEvent::Delta("partial".into())],
            vec![GenerationEvent::Finished(Ok(Generation {
                text: "mismatch".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: "stop",
            }))],
        ] {
            let (send, receive) = mpsc::channel(8);
            for event in events {
                send.send(event).await.unwrap();
            }
            drop(send);
            let response = stream_response(receive, "id".into(), "model".into(), false);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let frames: Vec<Value> = std::str::from_utf8(&bytes)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(frames.last().unwrap()["type"], "error");
            assert!(!frames.iter().any(|frame| frame["type"] == "done"));
        }
    }
}
