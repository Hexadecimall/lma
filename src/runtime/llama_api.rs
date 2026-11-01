//! Deliberately small llama.cpp HTTP subset, not llama-server parity.
//!
//! `/completion` accepts a string prompt, optional model, n_predict, temperature
//! (or temp), top_p, seed and stream (false by default). n_predict=-1 uses the
//! remaining native context; zero and other negative values are rejected.
//! `/tokenize` defaults add_special/parse_special to false; `/detokenize` renders
//! explicit special tokens. Both use raw native vocabulary IDs, never templates.
//! `/props` reports hosted defaults and trusted metadata. Its template information
//! describes raw completion and the stored profile selector, not an inferred or
//! fabricated effective chat template. Slots, scheduling and phase timings are
//! intentionally absent. Router authentication and the 1 MiB body limit apply.
use std::convert::Infallible;

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    inference::{
        Generation, GenerationEvent, GenerationRequest, InferenceError, Options, Prompt,
        TOKENIZER_MAX_BYTES, TOKENIZER_MAX_TOKENS, TokenizeRequest,
    },
    server::{ApiError, AppState, ensure_available, validate_options},
};

// Explicit null is not omission: it must not silently inherit a generation default.
fn present<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionRequest {
    prompt: String,
    #[serde(default, deserialize_with = "present")]
    model: Option<String>,
    #[serde(default, deserialize_with = "present")]
    n_predict: Option<i64>,
    #[serde(default, alias = "temp", deserialize_with = "present")]
    temperature: Option<f32>,
    #[serde(default, deserialize_with = "present")]
    top_p: Option<f32>,
    #[serde(default, deserialize_with = "present")]
    seed: Option<u32>,
    #[serde(default)]
    stream: bool,
}

impl CompletionRequest {
    fn apply_options(&self, mut options: Options, context: u32) -> Result<Options, ApiError> {
        if self.prompt.is_empty() {
            return Err(ApiError::invalid("prompt must be a nonempty string"));
        }
        if let Some(limit) = self.n_predict {
            if limit == -1 {
                options.until_context = true;
                // Public validation still requires a positive finite placeholder;
                // native output_allowance derives the real remaining budget.
                options.max_tokens = context;
            } else {
                options.max_tokens = u32::try_from(limit)
                    .ok()
                    .filter(|&limit| limit > 0)
                    .ok_or_else(|| {
                        ApiError::invalid("n_predict must be -1 or a positive integer")
                    })?;
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
        Ok(options)
    }

    fn prepare(self, state: &AppState) -> Result<(GenerationRequest, bool), ApiError> {
        let options = self.apply_options(state.generation_defaults(), state.context)?;
        // This adapter implements streaming itself; reuse common model/sampling validation.
        validate_options(state, self.model.as_deref(), false, &options)?;
        Ok((
            GenerationRequest {
                prompt: Prompt::Text(self.prompt),
                options,
            },
            self.stream,
        ))
    }
}

fn parse<T: DeserializeOwned>(body: &Value) -> Result<T, ApiError> {
    // Do not echo serde diagnostics: arbitrary field names/values may be secrets.
    T::deserialize(body).map_err(|_| {
        ApiError::invalid(
            "invalid llama.cpp subset request; unsupported fields, null options and non-scalar controls are rejected",
        )
    })
}

/// Model-independent checks for managed hosts, before selecting/loading a model.
/// Model identity, stored defaults, context budgets and vocabulary bounds remain
/// the selected-model handler's responsibility. Other routes are left untouched.
pub(super) fn validate_body(path: &str, body: &Value) -> Result<(), ApiError> {
    match path {
        "/completion" => {
            let request = parse::<CompletionRequest>(body)?;
            // Neutral placeholders validate only explicit controls, without
            // consulting a model or imposing the current host's context/defaults.
            let options = request.apply_options(
                Options {
                    until_context: false,
                    max_tokens: 1,
                    temperature: 0.0,
                    top_p: 1.0,
                    seed: None,
                },
                u32::MAX,
            )?;
            options.validate(u32::MAX)?;
        }
        "/tokenize" => {
            tokenize_request(body)?;
        }
        "/detokenize" => {
            detokenize_request(body)?;
        }
        _ => {}
    }
    Ok(())
}

pub(super) async fn completion(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = payload.map_err(ApiError::from_json)?;
    let (request, stream) = parse::<CompletionRequest>(&body)?.prepare(&state)?;
    ensure_available(&state)?;
    if stream {
        let native = state.inference.stream_prepared(move || Ok(request)).await?;
        Ok(stream_response(native))
    } else {
        let result = state.inference.generate(request).await?;
        Ok(Json(completion_value(&result, &result.text)?).into_response())
    }
}

fn completion_value(result: &Generation, content: &str) -> Result<Value, ApiError> {
    let stop_type = match result.finish_reason {
        "stop" => "eos",
        "length" => "limit",
        _ => {
            return Err(InferenceError::Internal(
                "llama completion returned an unsupported finish reason".into(),
            )
            .into());
        }
    };
    Ok(json!({
        "content": content,
        "stop": true,
        "tokens_predicted": result.completion_tokens,
        "tokens_evaluated": result.prompt_tokens,
        "stop_type": stop_type,
    }))
}

fn stream_response(mut native: mpsc::Receiver<GenerationEvent>) -> Response {
    let (send, receive) = mpsc::channel::<Result<Event, Infallible>>(8);
    tokio::spawn(async move {
        let mut emitted = String::new();
        loop {
            let event = tokio::select! {
                _ = send.closed() => return,
                event = native.recv() => event,
            };
            let error = match event {
                Some(GenerationEvent::Delta(content)) => {
                    emitted.push_str(&content);
                    if send
                        .send(Ok(Event::default()
                            .json_data(json!({"content":content,"stop":false}))
                            .expect("JSON value serializes")))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    continue;
                }
                Some(GenerationEvent::Finished(Ok(result))) if result.text == emitted => {
                    match completion_value(&result, "") {
                        Ok(value) => {
                            let _ = send
                                .send(Ok(Event::default()
                                    .json_data(value)
                                    .expect("JSON value serializes")))
                                .await;
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
            // Errors never fabricate a successful stop event or token usage.
            let _ = send
                .send(Ok(Event::default()
                    .event("error")
                    .json_data(error.stream_value("llama-completion"))
                    .expect("JSON value serializes")))
                .await;
            return;
        }
    });
    Sse::new(ReceiverStream::new(receive))
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizeBody {
    content: String,
    #[serde(default)]
    add_special: bool,
    #[serde(default)]
    parse_special: bool,
}

fn tokenize_request(body: &Value) -> Result<TokenizeRequest, ApiError> {
    // Reject oversized source text before allocating its typed copy.
    if body
        .get("content")
        .and_then(Value::as_str)
        .is_some_and(|content| content.len() > TOKENIZER_MAX_BYTES)
    {
        return Err(ApiError::invalid(
            "tokenizer content exceeds the 1 MiB byte limit",
        ));
    }
    let request = parse::<TokenizeBody>(body)?;
    let request = TokenizeRequest {
        content: request.content,
        add_special: request.add_special,
        parse_special: request.parse_special,
    };
    request.validate()?;
    Ok(request)
}

pub(super) async fn tokenize(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(body) = payload.map_err(ApiError::from_json)?;
    let request = tokenize_request(&body)?;
    let tokens = state.inference.tokenize(request).await?;
    Ok(Json(json!({"tokens":tokens})))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DetokenizeBody {
    tokens: Vec<i32>,
}

fn detokenize_request(body: &Value) -> Result<DetokenizeBody, ApiError> {
    // Bound the array before building a second, typed vector.
    if body
        .get("tokens")
        .and_then(Value::as_array)
        .is_some_and(|tokens| tokens.len() > TOKENIZER_MAX_TOKENS)
    {
        return Err(ApiError::invalid(
            "tokenizer input exceeds the 128k token limit",
        ));
    }
    let request = parse::<DetokenizeBody>(body)?;
    if request.tokens.iter().any(|&id| id < 0) {
        return Err(ApiError::invalid(
            "token IDs must be nonnegative native vocabulary IDs",
        ));
    }
    Ok(request)
}

pub(super) async fn detokenize(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(body) = payload.map_err(ApiError::from_json)?;
    let request = detokenize_request(&body)?;
    let content = state.inference.detokenize(request.tokens).await?;
    Ok(Json(json!({"content":content})))
}

pub(super) async fn props(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    ensure_available(&state)?;
    let defaults = state.generation_defaults();
    Ok(Json(json!({
        "model": state.model_id,
        "default_generation_settings": {
            "n_ctx": state.context,
            "n_predict": defaults.max_tokens,
            "temperature": defaults.temperature,
            "top_p": defaults.top_p,
            "seed": defaults.seed,
        },
        "model_info": state.metadata.as_deref().map(|metadata| metadata.model_info()),
        "template": {
            "completion": "raw",
            "profile_selector": state.definition.as_deref().and_then(|definition| definition.template.as_deref()),
        },
    })))
}

#[cfg(test)]
mod tests {
    use super::super::{
        inference::{Engine, InferenceService},
        server::BODY_LIMIT,
    };
    use super::*;
    use axum::{
        Router,
        body::Body,
        extract::DefaultBodyLimit,
        http::Request,
        routing::{get, post},
    };
    use http_body_util::BodyExt;
    use std::sync::{Arc, Mutex, atomic::AtomicU64};
    use tower::ServiceExt;

    #[derive(Default)]
    struct Fixture {
        generations: Arc<Mutex<Vec<GenerationRequest>>>,
        tokenizations: Arc<Mutex<Vec<TokenizeRequest>>>,
    }

    impl Engine for Fixture {
        fn generate(
            &mut self,
            request: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            self.generations.lock().unwrap().push(request);
            Ok(Generation {
                text: "hé\nllo".into(),
                prompt_tokens: 7,
                completion_tokens: 3,
                finish_reason: "length",
            })
        }
        fn tokenize(
            &self,
            request: TokenizeRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Vec<i32>, InferenceError> {
            self.tokenizations.lock().unwrap().push(request);
            Ok(vec![314, 159, 265])
        }
        fn detokenize(
            &self,
            tokens: Vec<i32>,
            _: &dyn Fn() -> bool,
        ) -> Result<String, InferenceError> {
            if tokens != [314, 159, 265] {
                return Err(InferenceError::Invalid(
                    "token ID is outside the native vocabulary".into(),
                ));
            }
            Ok("hé\nllo".into())
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

    fn app(state: AppState) -> Router {
        Router::new()
            .route("/completion", post(completion))
            .route("/tokenize", post(tokenize))
            .route("/detokenize", post(detokenize))
            .route("/props", get(props))
            .layer(DefaultBodyLimit::max(BODY_LIMIT))
            .with_state(state)
    }

    fn request(path: &str, body: Value) -> Request<Body> {
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn value(response: Response) -> Value {
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
    }

    #[test]
    fn completion_preflight_rejects_intrinsic_errors_without_a_host_or_worker() {
        for body in [
            json!(null),
            json!([]),
            json!({}),
            json!({"prompt":null}),
            json!({"prompt":[1,2]}),
            json!({"prompt":""}),
            json!({"prompt":"x","model":null}),
            json!({"prompt":"x","model":1}),
            json!({"prompt":"x","name":"other"}),
            json!({"prompt":"x","n_predict":0}),
            json!({"prompt":"x","n_predict":-2}),
            json!({"prompt":"x","n_predict":null}),
            json!({"prompt":"x","n_predict":1.5}),
            json!({"prompt":"x","n_predict":u64::from(u32::MAX) + 1}),
            json!({"prompt":"x","temperature":null}),
            json!({"prompt":"x","temperature":-0.1}),
            json!({"prompt":"x","temperature":2.1}),
            json!({"prompt":"x","temperature":1e100}),
            json!({"prompt":"x","temp":"0.5"}),
            json!({"prompt":"x","temp":null}),
            json!({"prompt":"x","temperature":0,"temp":1}),
            json!({"prompt":"x","top_p":null}),
            json!({"prompt":"x","top_p":0}),
            json!({"prompt":"x","top_p":-0.1}),
            json!({"prompt":"x","top_p":1.1}),
            json!({"prompt":"x","top_p":"0.5"}),
            json!({"prompt":"x","seed":null}),
            json!({"prompt":"x","seed":-1}),
            json!({"prompt":"x","seed":1.5}),
            json!({"prompt":"x","seed":u64::from(u32::MAX) + 1}),
            json!({"prompt":"x","stream":null}),
            json!({"prompt":"x","stream":1}),
            json!({"prompt":"x","max_tokens":1}),
            json!({"prompt":"x","stop":["x"]}),
            json!({"prompt":"x","until_context":true}),
        ] {
            let error =
                validate_body("/completion", &body).expect_err("must reject before leasing");
            assert_eq!(
                error.into_response().status(),
                StatusCode::BAD_REQUEST,
                "{body}"
            );
        }
    }

    #[test]
    fn tokenizer_preflight_rejects_strict_schema_errors_without_a_host_or_worker() {
        for body in [
            json!(null),
            json!({}),
            json!({"content":null}),
            json!({"content":[1]}),
            json!({"content":"x","add_special":null}),
            json!({"content":"x","add_special":1}),
            json!({"content":"x","parse_special":null}),
            json!({"content":"x","parse_special":"false"}),
            json!({"content":"x","name":"other"}),
            json!({"content":"x","model":null}),
            json!({"content":"x","with_pieces":true}),
        ] {
            let error = validate_body("/tokenize", &body).expect_err("must reject before leasing");
            assert_eq!(
                error.into_response().status(),
                StatusCode::BAD_REQUEST,
                "{body}"
            );
        }
        for body in [
            json!(null),
            json!({}),
            json!({"tokens":null}),
            json!({"tokens":"1"}),
            json!({"tokens":[-1]}),
            json!({"tokens":[i32::MIN]}),
            json!({"tokens":[1.5]}),
            json!({"tokens":["1"]}),
            json!({"tokens":[u64::from(i32::MAX as u32) + 1]}),
            json!({"tokens":[],"name":"other"}),
            json!({"tokens":[],"model":null}),
            json!({"tokens":[],"remove_special":true}),
        ] {
            let error =
                validate_body("/detokenize", &body).expect_err("must reject before leasing");
            assert_eq!(
                error.into_response().status(),
                StatusCode::BAD_REQUEST,
                "{body}"
            );
        }
    }

    #[test]
    fn preflight_bounds_copies_and_defers_only_model_dependent_validation() {
        for body in [
            json!({"prompt":"raw"}),
            json!({"prompt":"raw","model":"not-currently-loaded","n_predict":-1,"temp":0,"top_p":1,"seed":0,"stream":true}),
            json!({"prompt":"raw","n_predict":u32::MAX,"temperature":2,"top_p":0.5,"seed":u32::MAX,"stream":false}),
        ] {
            validate_body("/completion", &body).unwrap();
        }
        for body in [
            json!({"content":""}),
            json!({"content":"raw","add_special":true,"parse_special":true}),
            json!({"content":"raw","add_special":false,"parse_special":false}),
            json!({"content":"é".repeat(TOKENIZER_MAX_BYTES / 2)}),
        ] {
            validate_body("/tokenize", &body).unwrap();
        }
        let oversized =
            json!({"content":"é".repeat(TOKENIZER_MAX_BYTES / 2 + 1),"add_special":null});
        let error = validate_body("/tokenize", &oversized).unwrap_err();
        assert!(
            error.ollama_value()["error"]
                .as_str()
                .unwrap()
                .contains("1 MiB")
        );
        for body in [
            json!({"tokens":[]}),
            json!({"tokens":[0,i32::MAX]}), // Vocabulary upper bounds require the selected model.
            json!({"tokens":vec![0; TOKENIZER_MAX_TOKENS]}),
        ] {
            validate_body("/detokenize", &body).unwrap();
        }
        let oversized = json!({"tokens":vec!["not-an-id"; TOKENIZER_MAX_TOKENS + 1]});
        let error = validate_body("/detokenize", &oversized).unwrap_err();
        // Count rejection precedes typed item deserialization/allocation.
        assert!(
            error.ollama_value()["error"]
                .as_str()
                .unwrap()
                .contains("128k")
        );
    }

    #[tokio::test]
    async fn completion_subset_defaults_limits_and_raw_prompt() {
        let fixture = Fixture::default();
        let captured = fixture.generations.clone();
        let (service, worker) = InferenceService::spawn(move || Ok(fixture)).await.unwrap();
        let state = state(service);
        for (body, until, limit, temperature) in [
            (json!({"prompt":"raw"}), false, 256, 0.8),
            (
                json!({"prompt":"raw","model":"fixture","n_predict":-1,"temp":0,"top_p":1,"seed":0}),
                true,
                1024,
                0.0,
            ),
            (
                json!({"prompt":"raw","n_predict":8,"temperature":0.5}),
                false,
                8,
                0.5,
            ),
        ] {
            let (request, stream) = parse::<CompletionRequest>(&body)
                .unwrap()
                .prepare(&state)
                .unwrap();
            assert!(!stream);
            assert!(matches!(&request.prompt, Prompt::Text(text) if text == "raw"));
            assert_eq!(request.options.until_context, until);
            assert_eq!(request.options.max_tokens, limit);
            assert_eq!(request.options.temperature, temperature);
        }
        let router = app(state);
        let response = router
            .clone()
            .oneshot(request("/completion", json!({"prompt":"raw"})))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            value(response).await,
            json!({"content":"hé\nllo","stop":true,"tokens_predicted":3,"tokens_evaluated":7,"stop_type":"limit"})
        );
        assert_eq!(captured.lock().unwrap().len(), 1);
        drop(router);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn strict_validation_never_enters_generation() {
        let fixture = Fixture::default();
        let captured = fixture.generations.clone();
        let (service, worker) = InferenceService::spawn(move || Ok(fixture)).await.unwrap();
        let router = app(state(service));
        for body in [
            json!({"prompt":[1,2]}),
            json!({"prompt":""}),
            json!({"prompt":"x","model":"wrong"}),
            json!({"prompt":"x","model":null}),
            json!({"prompt":"x","n_predict":0}),
            json!({"prompt":"x","n_predict":-2}),
            json!({"prompt":"x","n_predict":1025}),
            json!({"prompt":"x","n_predict":1.5}),
            json!({"prompt":"x","n_predict":null}),
            json!({"prompt":"x","temperature":null}),
            json!({"prompt":"x","temperature":3}),
            json!({"prompt":"x","temperature":0,"temp":1}),
            json!({"prompt":"x","top_p":0}),
            json!({"prompt":"x","top_p":null}),
            json!({"prompt":"x","seed":-1}),
            json!({"prompt":"x","seed":null}),
            json!({"prompt":"x","stream":null}),
            json!({"prompt":"x","stream":1}),
            json!({"prompt":"x","cache_prompt":true}),
            json!({"prompt":"x","stop":["x"]}),
            json!({"prompt":"x","until_context":true}),
            json!({"prompt":"x","id_slot":0}),
            json!({"prompt":"x","grammar":"root ::= x"}),
        ] {
            let response = router
                .clone()
                .oneshot(request("/completion", body.clone()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        }
        assert!(captured.lock().unwrap().is_empty());
        drop(router);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn tokenizer_shapes_defaults_and_body_bounds() {
        let fixture = Fixture::default();
        let captured = fixture.tokenizations.clone();
        let (service, worker) = InferenceService::spawn(move || Ok(fixture)).await.unwrap();
        let router = app(state(service));
        for body in [
            json!({"content":"hé\nllo"}),
            json!({"content":"","add_special":true,"parse_special":true}),
        ] {
            let response = router
                .clone()
                .oneshot(request("/tokenize", body))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(value(response).await, json!({"tokens":[314,159,265]}));
        }
        {
            let captured = captured.lock().unwrap();
            assert_eq!(captured[0].content, "hé\nllo");
            assert!(!captured[0].add_special && !captured[0].parse_special);
            assert!(captured[1].add_special && captured[1].parse_special);
        }
        let response = router
            .clone()
            .oneshot(request("/detokenize", json!({"tokens":[314,159,265]})))
            .await
            .unwrap();
        assert_eq!(value(response).await, json!({"content":"hé\nllo"}));
        for (path, body) in [
            ("/tokenize", json!({"content":[1]})),
            ("/tokenize", json!({"content":"x","add_special":null})),
            ("/tokenize", json!({"content":"x","with_pieces":true})),
            ("/detokenize", json!({"tokens":[-1]})),
            ("/detokenize", json!({"tokens":[2147483647]})),
            ("/detokenize", json!({"tokens":[1.5]})),
            ("/detokenize", json!({"tokens":["1"]})),
            ("/detokenize", json!({"tokens":[2147483648_u64]})),
            ("/detokenize", json!({"tokens":[],"remove_special":true})),
            (
                "/detokenize",
                json!({"tokens":vec![0; TOKENIZER_MAX_TOKENS + 1]}),
            ),
        ] {
            let response = router.clone().oneshot(request(path, body)).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        }
        for path in ["/completion", "/tokenize", "/detokenize"] {
            let body = if path == "/completion" {
                json!({"prompt":"x".repeat(BODY_LIMIT)})
            } else {
                json!({"content":"x".repeat(BODY_LIMIT)})
            };
            let response = router.clone().oneshot(request(path, body)).await.unwrap();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        }
        drop(router);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn props_only_reports_known_values_and_profile_defaults() {
        let (service, worker) = InferenceService::spawn(|| Ok(Fixture::default()))
            .await
            .unwrap();
        let mut state = state(service);
        state.definition = Some(Arc::new(crate::models::modelfile::Definition {
            from: "fixture".into(),
            system: None,
            template: Some("plain".into()),
            license: None,
            messages: Vec::new(),
            parameters: crate::models::modelfile::Parameters {
                num_predict: Some(19),
                temperature: Some(0.2),
                ..Default::default()
            },
        }));
        let Json(body) = props(State(state)).await.unwrap();
        assert_eq!(body["model"], "fixture");
        assert_eq!(body["default_generation_settings"]["n_ctx"], 1024);
        assert_eq!(body["default_generation_settings"]["n_predict"], 19);
        assert_eq!(
            body["template"],
            json!({"completion":"raw","profile_selector":"plain"})
        );
        assert!(body["model_info"].is_null());
        assert!(body.get("slots").is_none());
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn sse_uses_real_deltas_and_one_final_usage_event_without_done_sentinel() {
        let (service, worker) = InferenceService::spawn(|| Ok(Fixture::default()))
            .await
            .unwrap();
        let router = app(state(service));
        let response = router
            .clone()
            .oneshot(request("/completion", json!({"prompt":"x","stream":true})))
            .await
            .unwrap();
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&bytes).unwrap();
        let records: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            records,
            [
                json!({"content":"hé\nllo","stop":false}),
                json!({"content":"","stop":true,"tokens_predicted":3,"tokens_evaluated":7,"stop_type":"limit"})
            ]
        );
        assert!(!text.contains("[DONE]") && !text.contains("timings") && !text.contains("slot"));
        drop(router);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn stream_delivers_before_native_exit_and_disconnect_keeps_worker_admission() {
        struct Gated {
            cancelled: Arc<tokio::sync::Notify>,
            release: std::sync::mpsc::Receiver<()>,
        }
        impl Engine for Gated {
            fn generate(
                &mut self,
                _: GenerationRequest,
                _: &dyn Fn() -> bool,
            ) -> Result<Generation, InferenceError> {
                panic!("stream must use the incremental engine operation");
            }
            fn generate_stream(
                &mut self,
                _: GenerationRequest,
                cancelled: &dyn Fn() -> bool,
                emit: &mut dyn FnMut(String) -> Result<(), InferenceError>,
            ) -> Result<Generation, InferenceError> {
                emit("early 🦀".into())?;
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while !cancelled() {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "stream did not cancel"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                self.cancelled.notify_one();
                self.release
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                Err(InferenceError::Invalid("cancelled".into()))
            }
        }
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let (release, receive) = std::sync::mpsc::channel();
        let engine = Gated {
            cancelled: cancelled.clone(),
            release: receive,
        };
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        let router = app(state(service.clone()));
        let response = router
            .clone()
            .oneshot(request("/completion", json!({"prompt":"x","stream":true})))
            .await
            .unwrap();
        let mut body = response.into_body();
        let frame = tokio::time::timeout(std::time::Duration::from_secs(5), body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let text = std::str::from_utf8(frame.data_ref().unwrap()).unwrap();
        assert!(text.contains("early 🦀") && text.contains("\"stop\":false"));
        assert!(matches!(
            service.detokenize(vec![]).await,
            Err(InferenceError::Busy)
        ));
        drop(body);
        tokio::time::timeout(std::time::Duration::from_secs(5), cancelled.notified())
            .await
            .unwrap();
        assert!(matches!(
            service.detokenize(vec![]).await,
            Err(InferenceError::Busy)
        ));
        release.send(()).unwrap();
        drop(router);
        drop(service);
        tokio::time::timeout(std::time::Duration::from_secs(5), worker)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn stream_errors_and_mismatched_or_missing_final_results_never_report_success() {
        for final_event in [
            Some(GenerationEvent::Finished(Err(InferenceError::Invalid(
                "failed".into(),
            )))),
            Some(GenerationEvent::Finished(Ok(Generation {
                text: "different".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: "stop",
            }))),
            None,
        ] {
            let (send, receive) = mpsc::channel(4);
            send.send(GenerationEvent::Delta("partial".into()))
                .await
                .unwrap();
            if let Some(event) = final_event {
                send.send(event).await.unwrap();
            }
            drop(send);
            let response = stream_response(receive);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(text.contains("event: error\n"));
            assert!(!text.contains("\"stop\":true") && !text.contains("tokens_predicted"));
        }
        let eos = completion_value(
            &Generation {
                text: "".into(),
                prompt_tokens: 2,
                completion_tokens: 0,
                finish_reason: "stop",
            },
            "",
        )
        .unwrap();
        assert_eq!(eos["stop_type"], "eos");
    }
}
