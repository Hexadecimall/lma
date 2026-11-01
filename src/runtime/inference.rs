use std::{
    io::Read,
    num::NonZeroU32,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use encoding_rs::{CoderResult, Decoder, UTF_8};
use llama_cpp_2::{
    context::{
        LlamaContext,
        params::{KvCacheType, LlamaContextParams, LlamaPoolingType},
    },
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::{LlamaToken, logit_bias::LlamaLogitBias},
    vocab::BoundedVocabError,
};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot},
    task::JoinHandle,
};

use super::generation_controls::{GenerationControls, StopFilter};
use crate::config::{ServeConfig, backend_report};

#[cfg(feature = "vision")]
#[path = "vision.rs"]
mod vision;

#[derive(Debug)]
pub(super) enum InferenceError {
    Invalid(String),
    Internal(String),
    Busy,
    Unavailable,
}

#[derive(Debug)]
pub(super) struct Message {
    pub role: String,
    pub content: String,
    pub images: Vec<Vec<u8>>,
}

#[derive(Debug)]
pub(super) enum Prompt {
    Text(String),
    Chat(Vec<Message>),
    Controls {
        prompt: Box<Prompt>,
        controls: GenerationControls,
    },
    Constrained {
        prompt: Box<Prompt>,
        schema: serde_json::Value,
    },
    // Protocol adapters annotate before tool preparation's synthetic system
    // message can be mistaken for client-supplied instructions.
    RequestSystem {
        prompt: Box<Prompt>,
        explicit: bool,
    },
}

impl Prompt {
    pub(super) fn with_controls(self, controls: GenerationControls) -> Self {
        if controls.is_default() {
            self
        } else {
            Self::Controls {
                prompt: Box::new(self),
                controls,
            }
        }
    }

    // The outer controls are the request contract, like the outer schema.
    fn generation_controls(&self) -> Option<&GenerationControls> {
        match self {
            Self::Controls { controls, .. } => Some(controls),
            Self::Constrained { prompt, .. } | Self::RequestSystem { prompt, .. } => {
                prompt.generation_controls()
            }
            _ => None,
        }
    }

    pub(super) fn with_request_system(self, explicit: bool) -> Self {
        Self::RequestSystem {
            prompt: Box::new(self),
            explicit,
        }
    }

    fn has_images(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Chat(messages) => messages.iter().any(|message| !message.images.is_empty()),
            Self::Constrained { prompt, .. }
            | Self::RequestSystem { prompt, .. }
            | Self::Controls { prompt, .. } => prompt.has_images(),
        }
    }

    // The outer constraint is the output contract; nested wrappers still unwrap
    // fully for formatting, image validation, and token counting.
    fn output_schema(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Constrained { schema, .. } => Some(schema),
            Self::RequestSystem { prompt, .. } | Self::Controls { prompt, .. } => {
                prompt.output_schema()
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub(super) struct Options {
    /// Trusted in-process generation only; public API limits are still validated.
    pub until_context: bool,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u32>,
}

impl Options {
    pub fn validate(&self, context: u32) -> Result<(), InferenceError> {
        if self.max_tokens == 0 || self.max_tokens > context {
            return Err(InferenceError::Invalid(format!(
                "max_tokens must be positive and at most the configured context size ({context})"
            )));
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(InferenceError::Invalid(
                "temperature must be finite and between 0 and 2".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(InferenceError::Invalid(
                "top_p must be finite, greater than 0, and at most 1".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct GenerationRequest {
    pub prompt: Prompt,
    pub options: Options,
}

#[derive(Debug)]
pub(super) struct Generation {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: &'static str,
}

/// Raw text only: embeddings never apply a chat template or generate text.
#[derive(Debug)]
pub(super) struct EmbeddingRequest {
    pub input: Vec<String>,
    pub truncate: bool,
}

#[derive(Debug)]
pub(super) struct EmbeddingResult {
    pub embeddings: Vec<Vec<f32>>,
    /// Exact number of model tokens evaluated, after any requested truncation.
    pub prompt_tokens: usize,
}

pub(super) const EMBEDDING_MAX_INPUTS: usize = 128;
pub(super) const EMBEDDING_MAX_BYTES: usize = 1024 * 1024;
pub(super) const EMBEDDING_MAX_TOKENS: usize = 32_768;

impl EmbeddingRequest {
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.input.is_empty() || self.input.len() > EMBEDDING_MAX_INPUTS {
            return Err(InferenceError::Invalid(format!(
                "embedding input must contain 1-{EMBEDDING_MAX_INPUTS} strings"
            )));
        }
        if self.input.iter().any(String::is_empty) {
            return Err(InferenceError::Invalid(
                "embedding input strings must not be empty".into(),
            ));
        }
        if self.input.iter().map(String::len).sum::<usize>() > EMBEDDING_MAX_BYTES {
            return Err(InferenceError::Invalid(format!(
                "embedding input exceeds the {EMBEDDING_MAX_BYTES} byte limit"
            )));
        }
        Ok(())
    }
}

/// Use f64 accumulation so large finite f32 values cannot overflow the norm.
pub(super) fn normalize_embedding(vector: &mut [f32]) -> Result<(), InferenceError> {
    if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
        return Err(InferenceError::Internal(
            "model returned an empty or non-finite embedding".into(),
        ));
    }
    let norm = vector
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(InferenceError::Internal(
            "model returned a zero or invalid embedding norm".into(),
        ));
    }
    for value in vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

/// Raw vocabulary operations never format a prompt or allocate an inference context.
#[derive(Debug)]
pub(super) struct TokenizeRequest {
    pub content: String,
    pub add_special: bool,
    pub parse_special: bool,
}

pub(super) const TOKENIZER_MAX_BYTES: usize = 1024 * 1024;
pub(super) const TOKENIZER_MAX_TOKENS: usize = 128 * 1024;

impl TokenizeRequest {
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.content.len() > TOKENIZER_MAX_BYTES {
            return Err(InferenceError::Invalid(
                "tokenizer content exceeds the 1 MiB byte limit".into(),
            ));
        }
        Ok(())
    }
}

fn tokenizer_native_error(error: BoundedVocabError, limit: &str) -> InferenceError {
    match error {
        BoundedVocabError::OutputTooLarge => InferenceError::Invalid(limit.into()),
        BoundedVocabError::InvalidToken | BoundedVocabError::InputTooLarge => {
            InferenceError::Invalid(error.to_string())
        }
        BoundedVocabError::AllocationFailed | BoundedVocabError::InvalidNativeResult => {
            InferenceError::Internal(error.to_string())
        }
    }
}

fn tokenizer_utf8(
    bytes: &[u8],
    limit: usize,
    cancelled: &dyn Fn() -> bool,
) -> Result<String, InferenceError> {
    let mut length = 0_usize;
    // Count replacement-expanded UTF-8 before allocating, then reserve exactly
    // that much. Decoding individual token pieces would corrupt split scalars.
    for chunk in bytes.utf8_chunks() {
        ensure_active(cancelled)?;
        length = length
            .checked_add(chunk.valid().len())
            .and_then(|n| n.checked_add(if chunk.invalid().is_empty() { 0 } else { 3 }))
            .filter(|&n| n <= limit)
            .ok_or_else(|| {
                InferenceError::Invalid("UTF-8 detokenized content exceeds the byte limit".into())
            })?;
    }
    ensure_active(cancelled)?;
    let mut content = String::new();
    content.try_reserve_exact(length).map_err(|error| {
        InferenceError::Internal(format!("cannot allocate bounded UTF-8 content: {error}"))
    })?;
    for chunk in bytes.utf8_chunks() {
        ensure_active(cancelled)?;
        content.push_str(chunk.valid());
        if !chunk.invalid().is_empty() {
            content.push('\u{fffd}');
        }
    }
    ensure_active(cancelled)?;
    Ok(content)
}

fn validate_detokenize(tokens: &[i32]) -> Result<(), InferenceError> {
    if tokens.len() > TOKENIZER_MAX_TOKENS {
        return Err(InferenceError::Invalid(
            "tokenizer input exceeds the 128k token limit".into(),
        ));
    }
    if tokens.iter().any(|&id| id < 0) {
        return Err(InferenceError::Invalid(
            "token IDs must be nonnegative native vocabulary IDs".into(),
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub(super) enum GenerationEvent {
    Delta(String),
    Finished(Result<Generation, InferenceError>),
}

// Bound both queued text and outstanding worker work, even for a stalled client.
const STREAM_CAPACITY: usize = 16;

pub(super) trait Engine: Send + 'static {
    // Model-dependent request validation must precede mutation and error cleanup.
    fn validate_generation(&self, _request: &GenerationRequest) -> Result<(), InferenceError> {
        Ok(())
    }

    fn generate(
        &mut self,
        request: GenerationRequest,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Generation, InferenceError>;

    // Fakes and non-native engines can opt into incremental output independently.
    fn generate_stream(
        &mut self,
        request: GenerationRequest,
        cancelled: &dyn Fn() -> bool,
        emit: &mut dyn FnMut(String) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        ensure_active(cancelled)?;
        let result = self.generate(request, cancelled)?;
        ensure_active(cancelled)?;
        if !result.text.is_empty() {
            emit(result.text.clone())?;
        }
        Ok(result)
    }

    /// Count without decoding, sampling, or changing generation/cache state.
    fn count_prompt(
        &self,
        _prompt: Prompt,
        _cancelled: &dyn Fn() -> bool,
    ) -> Result<usize, InferenceError> {
        Err(InferenceError::Invalid(
            "this engine does not support native prompt token counting".into(),
        ))
    }

    /// Native embedding engines opt in without changing existing generation fakes.
    fn embed(
        &mut self,
        _request: EmbeddingRequest,
        _cancelled: &dyn Fn() -> bool,
    ) -> Result<EmbeddingResult, InferenceError> {
        Err(InferenceError::Invalid(
            "this engine does not support native embeddings".into(),
        ))
    }

    fn tokenize(
        &self,
        _request: TokenizeRequest,
        _cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<i32>, InferenceError> {
        Err(InferenceError::Invalid(
            "this engine does not support native tokenization".into(),
        ))
    }

    fn detokenize(
        &self,
        _tokens: Vec<i32>,
        _cancelled: &dyn Fn() -> bool,
    ) -> Result<String, InferenceError> {
        Err(InferenceError::Invalid(
            "this engine does not support native detokenization".into(),
        ))
    }

    // A client can disappear after generate's final cancellation check but
    // before delivery. Stateless engines need no cleanup at that boundary.
    fn discard_request_state(&mut self) {}
}

enum JobRequest {
    Ready(GenerationRequest),
    Prepare(Box<dyn FnOnce() -> Result<GenerationRequest, InferenceError> + Send>),
    Count(Box<dyn FnOnce() -> Result<Prompt, InferenceError> + Send>),
    Embed(EmbeddingRequest),
    Tokenize(TokenizeRequest),
    Detokenize(Vec<i32>),
}

enum PreparedRequest {
    Generate(GenerationRequest),
    Count(Prompt),
    Embed(EmbeddingRequest),
    Tokenize(TokenizeRequest),
    Detokenize(Vec<i32>),
}

enum JobResult {
    Generated(Generation),
    Count(usize),
    Embedded(EmbeddingResult),
    Tokenized(Vec<i32>),
    Detokenized(String),
}

impl JobRequest {
    fn prepare(self) -> Result<PreparedRequest, InferenceError> {
        match self {
            Self::Ready(request) => Ok(PreparedRequest::Generate(request)),
            Self::Prepare(prepare) => prepare().map(PreparedRequest::Generate),
            Self::Count(prepare) => prepare().map(PreparedRequest::Count),
            Self::Embed(request) => Ok(PreparedRequest::Embed(request)),
            Self::Tokenize(request) => Ok(PreparedRequest::Tokenize(request)),
            Self::Detokenize(tokens) => Ok(PreparedRequest::Detokenize(tokens)),
        }
    }
}

enum JobReply {
    Single(oneshot::Sender<(Result<JobResult, InferenceError>, OwnedSemaphorePermit)>),
    Stream(mpsc::Sender<GenerationEvent>),
}

impl JobReply {
    fn is_closed(&self) -> bool {
        match self {
            Self::Single(reply) => reply.is_closed(),
            Self::Stream(reply) => reply.is_closed(),
        }
    }

    // A single-result receiver releases admission before exposing its result.
    // Failed delivery returns the permit so worker-side cleanup stays admitted.
    fn send(
        self,
        result: Result<JobResult, InferenceError>,
        permit: OwnedSemaphorePermit,
    ) -> Result<(), OwnedSemaphorePermit> {
        match self {
            Self::Single(reply) => reply.send((result, permit)).map_err(|(_, permit)| permit),
            Self::Stream(reply) => {
                let result = result.and_then(|result| match result {
                    JobResult::Generated(generation) => Ok(generation),
                    _ => Err(InferenceError::Internal(
                        "unexpected non-generation result for stream job".into(),
                    )),
                });
                if send_stream_event(&reply, GenerationEvent::Finished(result)).is_err() {
                    return Err(permit);
                }
                drop(permit);
                Ok(())
            }
        }
    }
}

// Only full queues need to park the blocking inference worker.
fn send_stream_event(
    reply: &mpsc::Sender<GenerationEvent>,
    event: GenerationEvent,
) -> Result<(), InferenceError> {
    match reply.try_send(event) {
        Ok(()) => Ok(()),
        Err(mpsc::error::TrySendError::Full(event)) => {
            reply.blocking_send(event).map_err(|_| cancelled_error())
        }
        Err(mpsc::error::TrySendError::Closed(_)) => Err(cancelled_error()),
    }
}

struct Job {
    request: JobRequest,
    reply: JobReply,
    // The worker, not the HTTP future, owns admission until the model operation ends.
    _permit: OwnedSemaphorePermit,
}

// This is not a general request queue: only a disconnected consumer can grant
// one successor a short opportunity to wait for worker-owned cleanup.
const CANCELLED_HANDOFF_TIMEOUT: Duration = Duration::from_millis(1500);

enum Consumer {
    Single(Arc<AtomicBool>),
    Stream(mpsc::WeakSender<GenerationEvent>),
    Untracked,
}

impl Consumer {
    fn is_cancelled(&self) -> bool {
        match self {
            Self::Single(cancelled) => cancelled.load(Ordering::Acquire),
            Self::Stream(reply) => reply.upgrade().is_none_or(|reply| reply.is_closed()),
            Self::Untracked => false,
        }
    }
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[derive(Clone)]
pub(super) struct InferenceService {
    sender: mpsc::Sender<Job>,
    admission: Arc<Semaphore>,
    consumer: Arc<Mutex<Consumer>>,
    handoff_waiter: Arc<Semaphore>,
}

impl InferenceService {
    pub async fn start(config: ServeConfig) -> Result<(Self, JoinHandle<()>), String> {
        Self::spawn(move || LlamaEngine::load(config)).await
    }

    pub async fn spawn<E: Engine>(
        load: impl FnOnce() -> Result<E, String> + Send + 'static,
    ) -> Result<(Self, JoinHandle<()>), String> {
        let (sender, mut receiver) = mpsc::channel::<Job>(1);
        let (ready_tx, ready_rx) = oneshot::channel();
        let worker = tokio::task::spawn_blocking(move || {
            let mut engine = match load() {
                Ok(engine) => engine,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            if ready_tx.send(Ok(())).is_err() {
                return;
            }
            while let Some(job) = receiver.blocking_recv() {
                if job.reply.is_closed() {
                    continue;
                }
                let request = match job.request.prepare() {
                    Ok(request) => request,
                    Err(error) => {
                        let _ = job.reply.send(Err(error), job._permit);
                        continue;
                    }
                };
                // Preparation can outlive a disconnected client. Do not start model
                // work (or invalidate a previous prefix) for that abandoned request.
                if job.reply.is_closed() {
                    continue;
                }
                let (result, generated) = match request {
                    PreparedRequest::Generate(request) => {
                        if let Err(error) = engine.validate_generation(&request) {
                            let _ = job.reply.send(Err(error), job._permit);
                            continue;
                        }
                        let result = match &job.reply {
                            JobReply::Single(_) => {
                                engine.generate(request, &|| job.reply.is_closed())
                            }
                            JobReply::Stream(reply) => engine.generate_stream(
                                request,
                                &|| reply.is_closed(),
                                &mut |delta| {
                                    send_stream_event(reply, GenerationEvent::Delta(delta))
                                },
                            ),
                        };
                        (result.map(JobResult::Generated), true)
                    }
                    PreparedRequest::Count(prompt) => (
                        engine
                            .count_prompt(prompt, &|| job.reply.is_closed())
                            .map(JobResult::Count),
                        false,
                    ),
                    PreparedRequest::Embed(request) => (
                        engine
                            .embed(request, &|| job.reply.is_closed())
                            .map(JobResult::Embedded),
                        false,
                    ),
                    PreparedRequest::Tokenize(request) => (
                        engine
                            .tokenize(request, &|| job.reply.is_closed())
                            .map(JobResult::Tokenized),
                        false,
                    ),
                    PreparedRequest::Detokenize(tokens) => (
                        engine
                            .detokenize(tokens, &|| job.reply.is_closed())
                            .map(JobResult::Detokenized),
                        false,
                    ),
                };
                // Embeddings own a separate context. Losing their result, like a
                // read-only count result, must not invalidate generation state.
                let discard = generated && (result.is_err() || job.reply.is_closed());
                if discard {
                    engine.discard_request_state();
                }
                if let Err(permit) = job.reply.send(result, job._permit) {
                    if generated && !discard {
                        engine.discard_request_state();
                    }
                    drop(permit);
                }
            }
        });
        ready_rx
            .await
            .map_err(|_| "inference worker stopped during model loading".to_string())??;
        Ok((
            Self {
                sender,
                admission: Arc::new(Semaphore::new(1)),
                consumer: Arc::new(Mutex::new(Consumer::Untracked)),
                handoff_waiter: Arc::new(Semaphore::new(1)),
            },
            worker,
        ))
    }

    pub async fn generate(&self, request: GenerationRequest) -> Result<Generation, InferenceError> {
        self.generate_job(JobRequest::Ready(request)).await
    }

    // Preparation shares inference admission and runs on the same blocking worker,
    // rather than building an unbounded queue of image-decoding tasks.
    pub async fn generate_prepared(
        &self,
        prepare: impl FnOnce() -> Result<GenerationRequest, InferenceError> + Send + 'static,
    ) -> Result<Generation, InferenceError> {
        self.generate_job(JobRequest::Prepare(Box::new(prepare)))
            .await
    }

    pub async fn stream_prepared(
        &self,
        prepare: impl FnOnce() -> Result<GenerationRequest, InferenceError> + Send + 'static,
    ) -> Result<mpsc::Receiver<GenerationEvent>, InferenceError> {
        let (reply, receive) = mpsc::channel(STREAM_CAPACITY);
        let consumer = Consumer::Stream(reply.downgrade());
        self.enqueue_wait(
            JobRequest::Prepare(Box::new(prepare)),
            JobReply::Stream(reply),
            consumer,
        )
        .await?;
        Ok(receive)
    }

    pub async fn count_prompt_prepared(
        &self,
        prepare: impl FnOnce() -> Result<Prompt, InferenceError> + Send + 'static,
    ) -> Result<usize, InferenceError> {
        match self.submit(JobRequest::Count(Box::new(prepare))).await? {
            JobResult::Count(tokens) => Ok(tokens),
            _ => Err(InferenceError::Internal(
                "unexpected result for count job".into(),
            )),
        }
    }

    pub async fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResult, InferenceError> {
        request.validate()?;
        match self.submit(JobRequest::Embed(request)).await? {
            JobResult::Embedded(result) => Ok(result),
            _ => Err(InferenceError::Internal(
                "unexpected result for embedding job".into(),
            )),
        }
    }

    pub async fn tokenize(&self, request: TokenizeRequest) -> Result<Vec<i32>, InferenceError> {
        request.validate()?;
        match self.submit(JobRequest::Tokenize(request)).await? {
            JobResult::Tokenized(tokens) => Ok(tokens),
            _ => Err(InferenceError::Internal(
                "unexpected result for tokenize job".into(),
            )),
        }
    }

    pub async fn detokenize(&self, tokens: Vec<i32>) -> Result<String, InferenceError> {
        validate_detokenize(&tokens)?;
        match self.submit(JobRequest::Detokenize(tokens)).await? {
            JobResult::Detokenized(content) => Ok(content),
            _ => Err(InferenceError::Internal(
                "unexpected result for detokenize job".into(),
            )),
        }
    }

    async fn generate_job(&self, request: JobRequest) -> Result<Generation, InferenceError> {
        match self.submit(request).await? {
            JobResult::Generated(generation) => Ok(generation),
            _ => Err(InferenceError::Internal(
                "unexpected result for generation job".into(),
            )),
        }
    }

    async fn submit(&self, request: JobRequest) -> Result<JobResult, InferenceError> {
        let (reply, receive) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let _cancel_on_drop = CancelOnDrop(cancelled.clone());
        self.enqueue_wait(
            request,
            JobReply::Single(reply),
            Consumer::Single(cancelled),
        )
        .await?;
        let (result, permit) = receive.await.map_err(|_| InferenceError::Unavailable)?;
        drop(permit);
        result
    }

    async fn enqueue_wait(
        &self,
        request: JobRequest,
        reply: JobReply,
        consumer: Consumer,
    ) -> Result<(), InferenceError> {
        if self.sender.is_closed() {
            return Err(InferenceError::Unavailable);
        }
        let waiter = {
            // Publish the new consumer atomically with immediate admission, so
            // another caller cannot mistake it for the previous cancelled one.
            let mut active = self.consumer.lock().unwrap();
            if let Ok(permit) = self.admission.clone().try_acquire_owned() {
                *active = consumer;
                return self.enqueue_admitted(request, reply, permit);
            }
            if !active.is_cancelled() {
                return Err(InferenceError::Busy);
            }
            self.handoff_waiter
                .clone()
                .try_acquire_owned()
                .map_err(|_| InferenceError::Busy)?
        };
        // Nothing is prepared or sent until cleanup releases admission. Dropping
        // this future removes its semaphore waiter and releases the one wait slot.
        let permit = tokio::time::timeout(
            CANCELLED_HANDOFF_TIMEOUT,
            self.admission.clone().acquire_owned(),
        )
        .await
        .map_err(|_| InferenceError::Busy)?
        .map_err(|_| InferenceError::Unavailable)?;
        *self.consumer.lock().unwrap() = consumer;
        let result = self.enqueue_admitted(request, reply, permit);
        drop(waiter);
        result
    }

    // Kept synchronous for model-free/native tests which inject abandoned jobs.
    #[cfg(test)]
    fn enqueue(&self, request: JobRequest, reply: JobReply) -> Result<(), InferenceError> {
        let mut active = self.consumer.lock().unwrap();
        let permit = self
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| InferenceError::Busy)?;
        *active = match &reply {
            JobReply::Stream(reply) => Consumer::Stream(reply.downgrade()),
            JobReply::Single(_) => Consumer::Untracked,
        };
        self.enqueue_admitted(request, reply, permit)
    }

    fn enqueue_admitted(
        &self,
        request: JobRequest,
        reply: JobReply,
        permit: OwnedSemaphorePermit,
    ) -> Result<(), InferenceError> {
        self.sender
            .try_send(Job {
                request,
                reply,
                _permit: permit,
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => InferenceError::Busy,
                mpsc::error::TrySendError::Closed(_) => InferenceError::Unavailable,
            })?;
        Ok(())
    }

    pub fn is_available(&self) -> bool {
        !self.sender.is_closed()
    }
}

// The dependent session drops before this owner; the model drops before the backend.
struct LoadedModel {
    model: LlamaModel,
    backend: LlamaBackend,
    chat_formatter: super::chat::Formatter,
    expert_history: Option<std::sync::Mutex<super::expert_history::ExpertHistory>>,
}

impl LoadedModel {
    fn checkpoint_experts(&self, force: bool) {
        if let Some(history) = &self.expert_history {
            match history.lock() {
                Ok(mut history) => {
                    if let Err(error) = history.checkpoint(&self.model, force) {
                        eprintln!("WARNING: expert history was not saved: {error}");
                    }
                }
                Err(_) => {
                    eprintln!("WARNING: expert history lock poisoned; inference remains available")
                }
            }
        }
    }
}

impl Drop for LoadedModel {
    fn drop(&mut self) {
        // Session/context is already gone; model counters and backend are live.
        self.checkpoint_experts(true);
    }
}

struct Session<'model> {
    context: LlamaContext<'model>,
    // Allocated only for embedding work; never shares generation KV/prefix state.
    embedding: Option<EmbeddingSession<'model>>,
    // `new` owns its buffers; unlike `get_one`, this batch borrows no token slice.
    batch: LlamaBatch<'static>,
    #[cfg(feature = "vision")]
    vision: Option<llama_cpp_2::mtmd::MtmdContext>,
    // Only successfully decoded text-prompt tokens are candidates for reuse.
    // Generation tokens can exist in memory but are always removed before prefill.
    cached_prompt: Vec<LlamaToken>,
    reuse_prefix: bool,
    native_state: GenerationState,
}

trait GenerationMemory {
    fn remove_all(&mut self) -> bool;
    fn is_empty(&self) -> bool;
    fn full_clear(&mut self);
}

impl GenerationMemory for LlamaContext<'_> {
    fn remove_all(&mut self) -> bool {
        matches!(self.clear_kv_cache_seq(None, None, None), Ok(true))
    }

    fn is_empty(&self) -> bool {
        self.kv_cache_seq_pos_min(0) == -1 && self.kv_cache_seq_pos_max(0) == -1
    }

    fn full_clear(&mut self) {
        self.clear_kv_cache();
    }
}

struct GenerationState {
    ordinary_attention: bool,
    dirty: bool,
    full_clear_required: bool,
}

impl GenerationState {
    fn new(recurrent: bool, hybrid: bool) -> Self {
        Self {
            ordinary_attention: !recurrent && !hybrid,
            dirty: false,
            full_clear_required: false,
        }
    }

    // Mark before native work: a failed decode may have changed memory without
    // ever publishing a cached prompt. Keep full-clear requirements sticky.
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn require_full_clear(&mut self) {
        self.mark_dirty();
        self.full_clear_required = true;
    }

    fn reset(&mut self, memory: &mut impl GenerationMemory) {
        if !self.dirty {
            return;
        }
        // Logical removal does not erase KV bytes. It is sufficient only for
        // text attention state whose native removal and empty-state checks pass.
        let removed = self.ordinary_attention
            && !self.full_clear_required
            && memory.remove_all()
            && memory.is_empty();
        if !removed {
            memory.full_clear();
        }
        self.dirty = false;
        self.full_clear_required = false;
    }
}

struct EmbeddingSession<'model> {
    context: LlamaContext<'model>,
    batch: LlamaBatch<'static>,
    encode: bool,
    limit: usize,
}

self_cell::self_cell! {
    struct ModelSession {
        owner: LoadedModel,
        #[not_covariant]
        dependent: Session,
    }
}

struct LlamaEngine {
    session: ModelSession,
    config: ServeConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryFallback {
    Error,
    Mmap,
}

impl MemoryFallback {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "error" => Ok(Self::Error),
            "mmap" => Ok(Self::Mmap),
            _ => Err("memory_fallback must be error or mmap".into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoadMode {
    Requested,
    CpuMmap,
}

#[derive(Debug)]
enum LoadFailure {
    // Backend initialization failures (including an existing singleton owner)
    // are not model/session failures and must not trigger a second init.
    Backend(String),
    Initialization(String),
}

impl LoadFailure {
    fn into_message(self) -> String {
        match self {
            Self::Backend(message) | Self::Initialization(message) => message,
        }
    }
}

fn page_memory_warning(reason: &str) -> String {
    format!(
        "WARNING: {reason}\nWARNING: CPU/pageable mmap maps file-backed WEIGHTS only, with memory locking disabled. Context/KV, compute, and projector allocations still require memory. OS-managed paging/swap may be extremely slow; mmap does not guarantee successful allocations and cannot rescue an OOM-killer termination or native abort. OS swap settings are not changed."
    )
}

fn initialize_with_fallback<T>(
    fallback: MemoryFallback,
    gpu_layers: u32,
    mut attempt: impl FnMut(LoadMode) -> Result<T, LoadFailure>,
    mut warn: impl FnMut(&str),
) -> Result<T, String> {
    if fallback == MemoryFallback::Mmap && gpu_layers == 0 {
        warn(&page_memory_warning(
            "memory_fallback=mmap explicitly requested with gpu_layers=0; using CPU/pageable mmap weights.",
        ));
        return attempt(LoadMode::CpuMmap).map_err(LoadFailure::into_message);
    }
    match attempt(LoadMode::Requested) {
        Ok(value) => Ok(value),
        Err(LoadFailure::Initialization(original))
            if fallback == MemoryFallback::Mmap && gpu_layers > 0 =>
        {
            // attempt() has returned and dropped all failed model/session/backend
            // state before another backend can be initialized. Exactly one retry.
            warn(&page_memory_warning(&format!(
                "GPU initialization failed: {original}. Retrying ONCE with CPU/pageable mmap weights (gpu_layers=0; KQV/op and projector GPU offload disabled)."
            )));
            attempt(LoadMode::CpuMmap).map_err(|retry| format!(
                "GPU initialization failed: {original}; CPU/pageable mmap retry also failed: {}",
                retry.into_message()
            ))
        }
        Err(error) => Err(error.into_message()),
    }
}

fn model_memory_policy(params: LlamaModelParams, mode: LoadMode) -> LlamaModelParams {
    match mode {
        LoadMode::Requested => params,
        LoadMode::CpuMmap => params
            .with_n_gpu_layers(0)
            .with_use_mmap(true)
            .with_use_mlock(false),
    }
}

fn expert_cache_model_policy(params: LlamaModelParams, config: &ServeConfig) -> LlamaModelParams {
    let budget_bytes = u64::from(config.expert_cache_mib) * 1_048_576;
    let params = params.with_expert_cache_bytes(budget_bytes);
    if budget_bytes > 0 {
        params.with_use_mmap(true).with_use_mlock(false)
    } else {
        params
    }
}

fn context_memory_policy(
    params: LlamaContextParams,
    mode: LoadMode,
    gpu_layers: u32,
) -> LlamaContextParams {
    if mode == LoadMode::CpuMmap || gpu_layers == 0 {
        // Zero model layers alone does not disable llama.cpp's opportunistic
        // operation offload. Explicit CPU must also keep KV/compute on CPU.
        params.with_offload_kqv(false).with_op_offload(false)
    } else {
        params
    }
}

fn validate_gguf_file(path: &std::path::Path, kind: &str) -> Result<(), String> {
    let text = path
        .to_str()
        .ok_or_else(|| format!("{kind} path must be valid UTF-8"))?;
    if text.contains('\0') {
        return Err(format!("{kind} path must not contain NUL bytes"));
    }
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("cannot open {kind} {}: {error}", path.display()))?;
    if !file
        .metadata()
        .map_err(|error| error.to_string())?
        .is_file()
    {
        return Err(format!("{kind} must be a local GGUF file"));
    }
    let mut magic = [0; 4];
    file.read_exact(&mut magic)
        .map_err(|error| format!("cannot read {kind} GGUF header: {error}"))?;
    if &magic != b"GGUF" {
        return Err(format!("{kind} is not a GGUF file (invalid header)"));
    }
    Ok(())
}

impl LlamaEngine {
    fn generate_with_output(
        &mut self,
        request: GenerationRequest,
        cancelled: &dyn Fn() -> bool,
        emit: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        self.validate_generation(&request)?;
        self.session.with_dependent_mut(|loaded, session| {
            let result = session.generate(
                &loaded.model,
                &loaded.chat_formatter,
                &self.config,
                request,
                cancelled,
                emit,
            );
            // Decode may have partially mutated native memory even when it fails.
            // Also check cancellation after the last decode/UTF-8 flush, not just before it.
            let was_cancelled = cancelled();
            if !retain_prefix(session.reuse_prefix, result.is_ok(), was_cancelled) {
                session.reset();
            }
            loaded.checkpoint_experts(false);
            if let Some(stats) = loaded.model.expert_cache_stats() {
                eprintln!(
                    "Expert cache request complete: device={}; cumulative hits={}; misses={}; evictions={}; source_bytes_copied={}",
                    if stats.gpu { "GPU" } else { "CPU" }, stats.hits, stats.misses, stats.evictions, stats.bytes_copied,
                );
            }
            if was_cancelled {
                return Err(cancelled_error());
            }
            result
        })
    }

    fn load(mut config: ServeConfig) -> Result<Self, String> {
        config.apply_expert_cache_policy()?;
        if config.context < 2 || config.context > i32::MAX as u32 {
            return Err("context must be between 2 and i32::MAX".into());
        }
        if config.threads <= 0 {
            return Err("threads must be positive".into());
        }
        if config.batch == 0
            || config.batch > i32::MAX as u32
            || config.micro_batch == 0
            || config.micro_batch > config.batch
        {
            return Err("batch must be positive and at most i32::MAX; micro_batch must be positive and at most batch".into());
        }
        if !matches!(config.flash.as_str(), "auto" | "on" | "off") {
            return Err("flash must be auto, on, or off".into());
        }
        if !crate::config::CHAT_TEMPLATES.contains(&config.chat_template.as_str()) {
            return Err(
                "unknown chat_template; use auto, native, plain, or a supported built-in format"
                    .into(),
            );
        }
        let kv_type = kv_cache_type(&config.kv_cache, &config.flash)?;
        let fallback = MemoryFallback::parse(&config.memory_fallback)?;
        #[cfg(not(feature = "vision"))]
        if config.projector.is_some() {
            return Err("a projector requires a build with the vision feature enabled".into());
        }
        validate_gguf_file(&config.model, "model")?;
        if let Some(path) = &config.projector {
            validate_gguf_file(path, "projector")?;
        }
        let report = backend_report(config.gpu_layers);
        eprintln!(
            "Backend request: compiled={}; requested={}; gpu_layers={}; memory_fallback={}. Actual tensor placement is not inferred from build features; see native device/offloaded-layer/buffer logs.",
            report.compiled_backends.join(","),
            report.requested_backend,
            report.requested_gpu_layers,
            config.memory_fallback,
        );
        initialize_with_fallback(
            fallback,
            config.gpu_layers,
            |mode| Self::load_attempt(config.clone(), kv_type, mode),
            |warning| eprintln!("{warning}"),
        )
    }

    fn load_attempt(
        config: ServeConfig,
        kv_type: KvCacheType,
        mode: LoadMode,
    ) -> Result<Self, LoadFailure> {
        Self::load_attempt_with_params(config, kv_type, mode, |params| params)
    }

    fn load_attempt_with_params(
        mut config: ServeConfig,
        kv_type: KvCacheType,
        mode: LoadMode,
        prepare_params: impl FnOnce(LlamaModelParams) -> LlamaModelParams,
    ) -> Result<Self, LoadFailure> {
        if mode == LoadMode::CpuMmap {
            // The projector uses the effective layer policy too: never leave its
            // GPU allocation enabled when retrying the text model on the CPU.
            config.gpu_layers = 0;
        }
        let backend = LlamaBackend::init().map_err(|error| {
            LoadFailure::Backend(format!("backend initialization failed: {error}"))
        })?;
        // Every failed attempt owns and drops its backend. The pinned backend's
        // Drop frees native state and releases its singleton guard for the retry.
        let session = (|| {
            let gpu_available = backend.supports_gpu_offload();
            eprintln!(
                "Backend runtime: GPU offload available={gpu_available}; effective gpu_layers={}; mode={mode:?} (availability is not actual layer placement).",
                config.gpu_layers,
            );
            if config.gpu_layers > 0 && !gpu_available {
                let compiled = backend_report(config.gpu_layers).compiled_backends.join(",");
                return Err(format!(
                    "GPU layers requested ({}) but no GPU backend is available (compiled={compiled}); check GPU/device access or build with --features metal (Apple) / --features cuda (NVIDIA). Use -g cpu to explicitly select CPU, or -mk mmap to opt into a warned CPU retry (long spellings are NSS-only)",
                    config.gpu_layers,
                ));
            }
            if config.expert_cache_mib > 0 && !backend.supports_mmap() {
                return Err("expert caching requires mmap support from the backend/platform".into());
            }
            if mode == LoadMode::CpuMmap && !backend.supports_mmap() {
                return Err("CPU/pageable mmap weights are not supported by this backend/platform".into());
            }
            let params = model_memory_policy(
                LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers),
                mode,
            );
            let params = prepare_params(expert_cache_model_policy(params, &config));
            let mut model = LlamaModel::load_from_file(&backend, &config.model, &params)
                .map_err(|error| format!("failed to load {}: {error}", config.model.display()))?;
            if let Some(stats) = model.expert_cache_stats() {
                eprintln!(
                    "Expert cache loaded: device={}; budget_bytes={}; allocated_bytes={} (actual persistent expert-slot buffers, not total RSS). Single-token expert matmuls execute on the cache device; GPU mode also streams SILU expert prefill for local batches of 2..=16 tokens through capacity-safe GPU tiles. Larger local batches use mapped CPU experts without full-weight GPU copies. Requested batching is preserved. Dense weights, mmap/OS cache, KV, expanded graph metadata and compute are outside this budget.",
                    if stats.gpu { "GPU" } else { "CPU" }, stats.budget_bytes, stats.allocated_bytes,
                );
            }
        let expert_history = if config.expert_cache_mib > 0 {
            match super::expert_history::ExpertHistory::open(&config.model, &mut model) {
                Ok(history) => history.map(std::sync::Mutex::new),
                Err(error) => {
                    eprintln!("WARNING: expert history unavailable; using demand-only expert caching: {error}");
                    None
                }
            }
        } else { None };
        let chat_formatter = super::chat::Formatter::load(&model, &config.chat_template)?;
        ModelSession::try_new(LoadedModel { model, backend, chat_formatter, expert_history }, |loaded| {
            // Recurrent/hybrid state cannot in general be rolled back to a token prefix.
            let native_state = GenerationState::new(loaded.model.is_recurrent(), loaded.model.is_hybrid());
            let reuse_prefix = config.prompt_cache && native_state.ordinary_attention;
            let params = generation_context_params(&config, kv_type);
            let params = context_memory_policy(params, mode, config.gpu_layers);
            #[cfg(test)]
            let params = native_tests::trace_context_params(params);
            let context = loaded
                .model
                .new_context(&loaded.backend, params)
                .map_err(|error| format!("context creation failed: {error}"))?;
            #[cfg(feature = "vision")]
            let vision = config
                .projector
                .as_deref()
                .map(|path| vision::load(path, &loaded.model, &config))
                .transpose()?;
            let batch_size = context.n_batch().min(config.batch).min(context.n_ctx().min(config.context));
            Ok::<_, String>(Session {
                batch: LlamaBatch::new(batch_size as usize, 1),
                context,
                embedding: None,
                #[cfg(feature = "vision")]
                vision,
                cached_prompt: Vec::new(),
                reuse_prefix,
                native_state,
            })
        })
        })().map_err(LoadFailure::Initialization)?;
        Ok(Self { session, config })
    }
}

impl Engine for LlamaEngine {
    fn validate_generation(&self, request: &GenerationRequest) -> Result<(), InferenceError> {
        if let Some(controls) = request.prompt.generation_controls() {
            controls.validate_vocab(self.session.borrow_owner().model.n_vocab())?;
        }
        Ok(())
    }

    fn tokenize(
        &self,
        request: TokenizeRequest,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<i32>, InferenceError> {
        request.validate()?;
        ensure_active(cancelled)?;
        let model = &self.session.borrow_owner().model;
        let result = model.vocab().tokenize_bounded(
            request.content.as_bytes(),
            request.add_special,
            request.parse_special,
            TOKENIZER_MAX_TOKENS,
        );
        ensure_active(cancelled)?;
        let tokens = result.map_err(|error| {
            tokenizer_native_error(error, "tokenizer output exceeds the 128k token limit")
        })?;
        Ok(tokens.into_iter().map(|token| token.0).collect())
    }

    fn detokenize(
        &self,
        tokens: Vec<i32>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String, InferenceError> {
        validate_detokenize(&tokens)?;
        ensure_active(cancelled)?;
        let model = &self.session.borrow_owner().model;
        let vocab = model.vocab();
        if tokens.iter().any(|&id| id >= vocab.n_tokens()) {
            return Err(InferenceError::Invalid(
                "token ID is outside the native vocabulary".into(),
            ));
        }
        // Decode the entire sequence, not individual pieces: a UTF-8 code point
        // can span tokens. Keep explicit special tokens visible, as llama.cpp does.
        let tokens: Vec<_> = tokens.into_iter().map(LlamaToken::new).collect();
        let result = vocab.detokenize_bounded(&tokens, false, true, TOKENIZER_MAX_BYTES);
        ensure_active(cancelled)?;
        let bytes = result.map_err(|error| {
            tokenizer_native_error(error, "detokenized content exceeds the 1 MiB byte limit")
        })?;
        tokenizer_utf8(&bytes, TOKENIZER_MAX_BYTES, cancelled)
    }

    fn embed(
        &mut self,
        request: EmbeddingRequest,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<EmbeddingResult, InferenceError> {
        request.validate()?;
        ensure_active(cancelled)?;
        self.session.with_dependent_mut(|loaded, session| {
            if session.embedding.is_none() {
                session.embedding = Some(EmbeddingSession::new(loaded, &self.config)?);
            }
            let embedding = session
                .embedding
                .as_mut()
                .expect("initialized embedding context");
            let result = embedding.embed(&loaded.model, request, cancelled);
            // Clean up on success, decode failure, and cancellation, independently
            // of the generation context (including abandoned worker replies).
            embedding.context.clear_kv_cache();
            embedding.batch.clear();
            ensure_active(cancelled)?;
            result
        })
    }

    fn count_prompt(
        &self,
        prompt: Prompt,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<usize, InferenceError> {
        ensure_active(cancelled)?;
        reject_count_images(&prompt)?;
        // Only the owner/model is borrowed; counting cannot access or mutate
        // the dependent Session, its prefix cache, batch, or native KV memory.
        let loaded = self.session.borrow_owner();
        let model = &loaded.model;
        let (formatted, _) = format_prompt(
            model,
            &loaded.chat_formatter,
            prompt,
            false,
            self.config.definition.as_ref(),
        )?;
        ensure_active(cancelled)?;
        let count = tokenize_text_prompt(model, &formatted).len();
        ensure_active(cancelled)?;
        Ok(count)
    }

    fn discard_request_state(&mut self) {
        self.session
            .with_dependent_mut(|_, session| session.reset());
    }

    fn generate(
        &mut self,
        request: GenerationRequest,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Generation, InferenceError> {
        self.generate_with_output(request, cancelled, &mut |_| Ok(()))
    }

    fn generate_stream(
        &mut self,
        request: GenerationRequest,
        cancelled: &dyn Fn() -> bool,
        emit: &mut dyn FnMut(String) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        self.generate_with_output(request, cancelled, &mut |delta| emit(delta.to_owned()))
    }
}

// The safe pinned API exposes model metadata but not has_encoder/has_decoder.
// These are the encoder architectures in its bundled llama_model_has_encoder.
fn embedding_encode_mode(architecture: &str) -> Result<bool, InferenceError> {
    match architecture {
        "t5" | "t5encoder" => Ok(true),
        "eagle3" | "dflash" => Err(InferenceError::Invalid(
            "draft/assistant architectures do not support standalone embeddings".into(),
        )),
        _ => Ok(false), // BERT-style non-causal models also use llama_decode.
    }
}

fn embedding_pooling(value: Option<&str>) -> Result<LlamaPoolingType, InferenceError> {
    match value {
        // A generative model (or an explicitly unpooled model) still supplies
        // real hidden states. Request native mean pooling, never fake vectors.
        None | Some("-1" | "0" | "1") => Ok(LlamaPoolingType::Mean),
        Some("2") => Ok(LlamaPoolingType::Cls),
        Some("3") => Ok(LlamaPoolingType::Last),
        Some("4") => Err(InferenceError::Invalid(
            "rank-pooling models return ranking scores, not embedding vectors".into(),
        )),
        Some(_) => Err(InferenceError::Invalid(
            "unsupported model embedding pooling type".into(),
        )),
    }
}

fn validate_embedding_config(config: &ServeConfig) -> Result<(), InferenceError> {
    if config.expert_cache_mib > 0 {
        return Err(InferenceError::Invalid(
            "embeddings are unavailable with expert_cache_mib enabled: the expert cache supports only one context, already used for generation; disable expert caching to use embeddings".into(),
        ));
    }
    Ok(())
}

impl<'model> EmbeddingSession<'model> {
    fn new(loaded: &'model LoadedModel, config: &ServeConfig) -> Result<Self, InferenceError> {
        validate_embedding_config(config)?;
        let architecture = loaded
            .model
            .meta_val_str("general.architecture")
            .map_err(|error| {
                InferenceError::Internal(format!(
                    "cannot read embedding model architecture: {error}"
                ))
            })?;
        let encode = embedding_encode_mode(&architecture)?;
        let pooling = loaded
            .model
            .meta_val_str(&format!("{architecture}.pooling_type"))
            .ok();
        let pooling = embedding_pooling(pooling.as_deref())?;
        // Do not extrapolate an encoder's positional embeddings beyond training.
        let limit = config
            .context
            .min(loaded.model.n_ctx_train())
            .min(EMBEDDING_MAX_TOKENS as u32);
        if limit == 0 {
            return Err(InferenceError::Invalid(
                "embedding context size must be positive".into(),
            ));
        }
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(limit))
            // Pooling and non-causal attention need the entire sequence in one
            // physical batch. Generation's smaller micro-batch is not valid here.
            .with_n_batch(limit)
            .with_n_ubatch(limit)
            .with_n_threads(config.threads)
            .with_n_threads_batch(config.threads)
            .with_embeddings(true)
            .with_pooling_type(pooling);
        // Keep native model attention defaults (causal vs non-causal), and use
        // unquantized embedding KV regardless of the generation KV policy.
        let params = context_memory_policy(params, LoadMode::Requested, config.gpu_layers);
        let context = loaded
            .model
            .new_context(&loaded.backend, params)
            .map_err(|error| {
                InferenceError::Internal(format!("embedding context creation failed: {error}"))
            })?;
        let limit = limit
            .min(context.n_ctx())
            .min(context.n_batch())
            .min(context.n_ubatch()) as usize;
        Ok(Self {
            context,
            batch: LlamaBatch::new(limit, 1),
            encode,
            limit,
        })
    }

    fn embed(
        &mut self,
        model: &LlamaModel,
        request: EmbeddingRequest,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<EmbeddingResult, InferenceError> {
        let mut inputs = Vec::with_capacity(request.input.len());
        let mut prompt_tokens = 0;
        for text in &request.input {
            ensure_active(cancelled)?;
            // Include model BOS/EOS, but user text is not a chat template: do
            // not interpret literal special-token spellings as control tokens.
            let mut tokens = model.vocab().tokenize(text.as_bytes(), true, false);
            truncate_embedding_tokens(&mut tokens, self.limit, request.truncate)?;
            prompt_tokens += tokens.len();
            if prompt_tokens > EMBEDDING_MAX_TOKENS {
                return Err(InferenceError::Invalid(format!(
                    "embedding batch exceeds {EMBEDDING_MAX_TOKENS} evaluated tokens"
                )));
            }
            inputs.push(tokens);
        }
        let mut embeddings = Vec::with_capacity(inputs.len());
        for tokens in inputs {
            ensure_active(cancelled)?;
            self.context.clear_kv_cache();
            self.batch.clear();
            for (position, token) in tokens.iter().enumerate() {
                self.batch
                    .add(*token, position as i32, &[0], true)
                    .map_err(|error| {
                        InferenceError::Internal(format!("embedding batching failed: {error}"))
                    })?;
            }
            if self.encode {
                self.context.encode(&mut self.batch).map_err(|error| {
                    InferenceError::Internal(format!("embedding encode failed: {error}"))
                })?;
            } else {
                self.context.decode(&mut self.batch).map_err(|error| {
                    InferenceError::Internal(format!("embedding decode failed: {error}"))
                })?;
            }
            ensure_active(cancelled)?;
            let mut vector = self
                .context
                .embeddings_seq_ith(0)
                .map_err(|error| {
                    InferenceError::Internal(format!(
                        "native pooled embedding unavailable: {error}"
                    ))
                })?
                .to_vec();
            normalize_embedding(&mut vector)?;
            embeddings.push(vector);
        }
        Ok(EmbeddingResult {
            embeddings,
            prompt_tokens,
        })
    }
}

fn truncate_embedding_tokens(
    tokens: &mut Vec<LlamaToken>,
    limit: usize,
    truncate: bool,
) -> Result<(), InferenceError> {
    if tokens.len() > limit {
        if !truncate {
            return Err(InferenceError::Invalid(format!(
                "embedding input has {} tokens, exceeding the embedding context limit ({limit}); enable truncate to discard trailing tokens",
                tokens.len()
            )));
        }
        tokens.truncate(limit);
    }
    if tokens.is_empty() {
        return Err(InferenceError::Invalid(
            "embedding input must tokenize to at least one token".into(),
        ));
    }
    Ok(())
}

impl Session<'_> {
    fn reset(&mut self) {
        self.cached_prompt.clear();
        self.native_state.reset(&mut self.context);
    }

    fn prefill_text(
        &mut self,
        tokens: Vec<LlamaToken>,
        batch_size: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), InferenceError> {
        let mut keep = prefix_to_reuse(&self.cached_prompt, &tokens, self.reuse_prefix);
        // Only use a prefix still present in native memory, including sliding-window
        // layers (the pinned memory API reports their minimum retained position).
        // Do not force full-size SWA caches just for reuse: preserve GPU-fit models.
        // Unsupported partial removal also falls back to a full prefill.
        if keep > 0 {
            self.native_state.mark_dirty();
            if !prefix_is_resident(
                keep,
                self.context.kv_cache_seq_pos_min(0),
                self.context.kv_cache_seq_pos_max(0),
            ) {
                keep = 0;
            } else if !matches!(
                self.context
                    .clear_kv_cache_seq(Some(0), Some(keep as u32), None),
                Ok(true)
            ) {
                // A rejected/partial mutation must not be treated as clean.
                self.native_state.require_full_clear();
                keep = 0;
            }
        }
        if keep == 0 {
            self.reset();
        }
        for (chunk_index, chunk) in tokens[keep..].chunks(batch_size).enumerate() {
            ensure_active(cancelled)?;
            self.batch.clear();
            for (offset, token) in chunk.iter().enumerate() {
                let position = keep + chunk_index * batch_size + offset;
                self.batch
                    .add(*token, position as i32, &[0], position + 1 == tokens.len())
                    .map_err(|error| {
                        InferenceError::Internal(format!("prompt batching failed: {error}"))
                    })?;
            }
            self.native_state.mark_dirty();
            self.context.decode(&mut self.batch).map_err(|error| {
                InferenceError::Internal(format!("prompt decode failed: {error}"))
            })?;
        }
        // Publish only a fully decoded prompt, taking the tokenizer's allocation
        // instead of copying every prefill chunk into a second token buffer.
        if self.reuse_prefix {
            self.cached_prompt = tokens;
        }
        Ok(())
    }

    fn generate(
        &mut self,
        model: &LlamaModel,
        chat_formatter: &super::chat::Formatter,
        config: &ServeConfig,
        request: GenerationRequest,
        cancelled: &dyn Fn() -> bool,
        emit: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        let options = request.options;
        options.validate(config.context)?;
        ensure_active(cancelled)?;
        let has_images = request.prompt.has_images();
        validate_image_support(
            has_images,
            config.projector.is_some(),
            cfg!(feature = "vision"),
        )?;
        let grammar = request
            .prompt
            .output_schema()
            .map(schema_grammar)
            .transpose()?;
        let grammar = grammar
            .as_deref()
            .map(|grammar| {
                LlamaSampler::grammar(model, grammar, "root").map_err(|error| {
                    InferenceError::Invalid(format!("cannot initialize output grammar: {error}"))
                })
            })
            .transpose()?;
        let controls = request
            .prompt
            .generation_controls()
            .cloned()
            .unwrap_or_default();
        // Reject IDs before prefill/reset or any other mutation of native state.
        controls.validate_vocab(model.n_vocab())?;
        let greedy_fast_path =
            use_controlled_greedy_fast_path(&options, grammar.is_some(), &controls);
        let mut sampler = generation_sampler_with_controls(
            grammar,
            &options,
            &controls,
            model.n_vocab(),
            config.context,
        );
        let mut stops = StopFilter::new(controls.stop);
        ensure_active(cancelled)?;
        let (prompt, images) = format_prompt(
            model,
            chat_formatter,
            request.prompt,
            has_images,
            config.definition.as_ref(),
        )?;
        let context_size = self.context.n_ctx().min(config.context);
        let batch_size = self.context.n_batch().min(config.batch).min(context_size) as usize;

        let prompt_tokens;
        let output_limit;
        let mut position;
        if has_images {
            // Multimodal memory/positions are not interchangeable with text tokens.
            self.reset();
            #[cfg(feature = "vision")]
            {
                let mtmd = self.vision.as_mut().ok_or_else(|| {
                    InferenceError::Invalid("images require a loaded vision projector".into())
                })?;
                self.native_state.require_full_clear();
                let chunks = vision::prepare(mtmd, prompt, &images, &self.context, cancelled)?;
                prompt_tokens = chunks.total_tokens();
                let positions = usize::try_from(chunks.total_positions()).map_err(|_| {
                    InferenceError::Invalid("invalid multimodal position count".into())
                })?;
                output_limit = output_allowance(prompt_tokens, positions, &options, context_size)?;
                ensure_active(cancelled)?;
                self.native_state.require_full_clear();
                position = chunks
                    .eval_chunks(mtmd, &mut self.context, 0, 0, batch_size as i32, true)
                    .map_err(|error| {
                        InferenceError::Internal(format!("image prompt evaluation failed: {error}"))
                    })?;
                if position < 0 || position as usize != positions {
                    return Err(InferenceError::Internal(
                        "multimodal evaluation returned inconsistent positions".into(),
                    ));
                }
                ensure_active(cancelled)?;
            }
            #[cfg(not(feature = "vision"))]
            {
                let _ = images;
                return Err(InferenceError::Invalid(
                    "images require a build with the vision feature enabled".into(),
                ));
            }
        } else {
            let tokens = tokenize_text_prompt(model, &prompt);
            prompt_tokens = tokens.len();
            output_limit = output_allowance(prompt_tokens, prompt_tokens, &options, context_size)?;
            self.prefill_text(tokens, batch_size, cancelled)?;
            position = prompt_tokens as i32;
        }
        let vocab = model.vocab();

        let mut decoder = UTF_8.new_decoder_without_bom_handling();
        let mut result = Generation {
            text: String::new(),
            prompt_tokens,
            completion_tokens: 0,
            finish_reason: "length",
        };
        for index in 0..output_limit {
            ensure_active(cancelled)?;
            let token = if greedy_fast_path {
                // The logits borrow ends at selection, before any context mutation.
                let token = greedy_token(self.context.get_logits_ith(-1)).ok_or_else(|| {
                    InferenceError::Internal(
                        "greedy sampling received empty or oversized logits".into(),
                    )
                })?;
                // sample() accepts internally; the shortcut must accept exactly once too.
                sampler.accept(token);
                token
            } else {
                sampler.sample(&self.context, -1)
            };
            result.completion_tokens += 1;
            if vocab.is_eog(token) {
                result.finish_reason = "stop";
                break;
            }
            if append_generation_utf8(
                &mut decoder,
                &vocab.token_to_piece(token, false, None),
                &mut result.text,
                false,
                &mut stops,
                emit,
            )? {
                result.finish_reason = "stop";
                break;
            }
            ensure_active(cancelled)?;
            if index + 1 < output_limit {
                self.batch.clear();
                self.batch
                    .add(token, position, &[0], true)
                    .map_err(|error| {
                        InferenceError::Internal(format!("token batching failed: {error}"))
                    })?;
                self.native_state.mark_dirty();
                self.context.decode(&mut self.batch).map_err(|error| {
                    InferenceError::Internal(format!("generation decode failed: {error}"))
                })?;
                position += 1;
            }
        }
        if append_generation_utf8(&mut decoder, &[], &mut result.text, true, &mut stops, emit)? {
            result.finish_reason = "stop";
        }
        ensure_active(cancelled)?;
        if has_images {
            self.reset();
        }
        Ok(result)
    }
}

fn schema_grammar(schema: &serde_json::Value) -> Result<String, InferenceError> {
    llama_cpp_2::json_schema_to_grammar(&schema.to_string()).map_err(|error| {
        InferenceError::Invalid(format!(
            "cannot convert output JSON schema to grammar: {error}"
        ))
    })
}

fn generation_context_params(config: &ServeConfig, kv_type: KvCacheType) -> LlamaContextParams {
    // llama-cpp-2 exposes the C enum: AUTO=-1, DISABLED=0, ENABLED=1.
    let flash_policy = match config.flash.as_str() {
        "on" => 1,
        "off" => 0,
        _ => -1,
    };
    // Expert matmuls have explicit cache/prefill placement; preserve the native
    // GPU operation policy. Explicit CPU selection is handled separately.
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(config.context))
        .with_n_batch(config.batch)
        .with_n_ubatch(config.micro_batch)
        .with_n_threads(config.threads)
        .with_n_threads_batch(config.threads)
        .with_flash_attention_policy(flash_policy)
        .with_type_k(kv_type)
        .with_type_v(kv_type)
        // Bound SWA memory instead of retaining full history solely for reuse.
        // Rolled-out prefixes must be prefilled again (see prefix_is_resident).
        .with_swa_full(false)
        // Pooled embedding defaults must not leak into text generation.
        .with_pooling_type(LlamaPoolingType::None)
}

fn use_greedy_fast_path(options: &Options, has_grammar: bool) -> bool {
    options.temperature == 0.0 && !has_grammar
}

fn greedy_token(logits: &[f32]) -> Option<LlamaToken> {
    logits.first()?;
    let mut selected = 0;
    for index in 1..logits.len() {
        // Match native greedy's strict comparison, including ties and NaNs.
        // max_by/total_cmp would change its first-NaN and first-maximum behavior.
        if logits[index] > logits[selected] {
            selected = index;
        }
    }
    i32::try_from(selected).ok().map(LlamaToken)
}

fn use_controlled_greedy_fast_path(
    options: &Options,
    has_grammar: bool,
    controls: &GenerationControls,
) -> bool {
    use_greedy_fast_path(options, has_grammar) && !controls.has_sampling()
}

#[cfg(test)]
fn generation_sampler(grammar: Option<LlamaSampler>, options: &Options) -> LlamaSampler {
    generation_sampler_with_controls(grammar, options, &GenerationControls::default(), 0, 0)
}

fn generation_sampler_with_controls(
    grammar: Option<LlamaSampler>,
    options: &Options,
    controls: &GenerationControls,
    n_vocab: i32,
    history_limit: u32,
) -> LlamaSampler {
    let mut chain = Vec::with_capacity(if controls.has_sampling() { 6 } else { 4 });
    if controls.frequency_penalty != 0.0 || controls.presence_penalty != 0.0 {
        // No prompt tokens are accepted. Keep the entire generated history, not
        // a sliding repetition window; a request cannot exceed its context.
        chain.push(LlamaSampler::penalties(
            n_vocab,
            history_limit.min(i32::MAX as u32) as i32,
            1.0,
            controls.frequency_penalty,
            controls.presence_penalty,
        ));
    }
    if controls.logit_bias.iter().any(|(_, bias)| *bias != 0.0) {
        let biases: Vec<_> = controls
            .logit_bias
            .iter()
            .map(|(token, bias)| LlamaLogitBias::new(LlamaToken(*token), *bias))
            .collect();
        chain.push(LlamaSampler::logit_bias(n_vocab, &biases));
    }
    // Grammar must filter the full candidate set before any selection, including
    // greedy; appending it after selection does not enforce constrained output.
    chain.extend(grammar);
    if options.temperature == 0.0 {
        chain.push(LlamaSampler::greedy());
    } else {
        // UINT32_MAX is llama.cpp's random-seed sentinel. Explicit seeds remain
        // deterministic, including an explicitly supplied maximum seed.
        let seed = options
            .seed
            .map_or(u32::MAX, |seed| if seed == u32::MAX { 0 } else { seed });
        chain.extend([
            LlamaSampler::top_p(options.top_p, 1),
            LlamaSampler::temp(options.temperature),
            LlamaSampler::dist(seed),
        ]);
    }
    LlamaSampler::chain_simple(chain)
}

fn kv_cache_type(kind: &str, flash: &str) -> Result<KvCacheType, String> {
    let kind = match kind {
        "f16" => KvCacheType::F16,
        "q8_0" => KvCacheType::Q8_0,
        "q4_0" => KvCacheType::Q4_0,
        _ => return Err("kv_cache must be f16, q8_0, or q4_0".into()),
    };
    // Quantized V requires flash attention. Require an explicit policy rather
    // than letting AUTO silently select a backend that cannot use this cache.
    if kind != KvCacheType::F16 && flash != "on" {
        return Err(
            "quantized KV cache requires flash=on; use kv_cache=f16 with flash=auto/off".into(),
        );
    }
    Ok(kind)
}

fn cancelled_error() -> InferenceError {
    InferenceError::Internal("request cancelled".into())
}

fn ensure_active(cancelled: &dyn Fn() -> bool) -> Result<(), InferenceError> {
    if cancelled() {
        Err(cancelled_error())
    } else {
        Ok(())
    }
}

fn retain_prefix(enabled: bool, succeeded: bool, cancelled: bool) -> bool {
    enabled && succeeded && !cancelled
}

fn prefix_is_resident(keep: usize, min: i32, max: i32) -> bool {
    keep > 0 && min == 0 && usize::try_from(max).is_ok_and(|max| max >= keep - 1)
}

fn prefix_to_reuse<T: Eq>(cached: &[T], next: &[T], enabled: bool) -> usize {
    if !enabled {
        return 0;
    }
    // Even identical or truncated prompts must decode their final token again
    // to refresh logits after removing the previous generation's suffix.
    cached
        .iter()
        .zip(next)
        .take(next.len().saturating_sub(1))
        .take_while(|(cached, next)| cached == next)
        .count()
}

fn validate_image_support(
    images: bool,
    projector: bool,
    vision: bool,
) -> Result<(), InferenceError> {
    if images && !vision {
        return Err(InferenceError::Invalid(
            "images require a build with the vision feature enabled".into(),
        ));
    }
    if images && !projector {
        return Err(InferenceError::Invalid(
            "images require a compatible local vision projector (--projector)".into(),
        ));
    }
    Ok(())
}

const IMAGE_PLACEHOLDER_PREFIX: &str = "<__lma_image_";

fn image_placeholder(index: usize) -> String {
    format!("{IMAGE_PLACEHOLDER_PREFIX}{index}__>")
}

fn media_marker() -> &'static str {
    #[cfg(feature = "vision")]
    {
        llama_cpp_2::mtmd::mtmd_default_marker()
    }
    #[cfg(not(feature = "vision"))]
    {
        "<__media__>"
    }
}

fn attach_images(messages: &mut [Message]) -> Result<Vec<Vec<u8>>, InferenceError> {
    let mut images = Vec::new();
    for message in messages {
        if !message.images.is_empty() && message.role != "user" {
            return Err(InferenceError::Invalid(
                "only user messages may contain images".into(),
            ));
        }
        if message.content.contains(media_marker())
            || message.content.contains(IMAGE_PLACEHOLDER_PREFIX)
        {
            return Err(InferenceError::Invalid(
                "image messages must not contain reserved media markers".into(),
            ));
        }
        if !message.images.is_empty() {
            let mut content = String::new();
            for index in images.len()..images.len() + message.images.len() {
                content.push_str(&image_placeholder(index));
                content.push('\n');
            }
            content.push_str(&message.content);
            message.content = content;
            images.append(&mut message.images);
        }
    }
    Ok(images)
}

fn restore_image_markers(prompt: String, count: usize) -> Result<String, InferenceError> {
    if prompt.contains(media_marker()) {
        return Err(InferenceError::Invalid(
            "chat template inserted an unbound media marker".into(),
        ));
    }
    let mut restored = String::with_capacity(prompt.len());
    let mut remaining = prompt.as_str();
    for index in 0..count {
        let placeholder = image_placeholder(index);
        let Some(offset) = remaining.find(IMAGE_PLACEHOLDER_PREFIX) else {
            return Err(InferenceError::Invalid(
                "the model's chat template must preserve every image marker exactly once and in message order".into(),
            ));
        };
        let marker = &remaining[offset..];
        if !marker.starts_with(&placeholder) {
            return Err(InferenceError::Invalid(
                "the model's chat template must preserve every image marker exactly once and in message order".into(),
            ));
        }
        restored.push_str(&remaining[..offset]);
        restored.push_str(media_marker());
        remaining = &marker[placeholder.len()..];
    }
    if remaining.contains(IMAGE_PLACEHOLDER_PREFIX) {
        return Err(InferenceError::Invalid(
            "chat template contains an unknown or duplicate image marker".into(),
        ));
    }
    restored.push_str(remaining);
    Ok(restored)
}

fn format_prompt(
    model: &LlamaModel,
    chat_formatter: &super::chat::Formatter,
    prompt: Prompt,
    has_images: bool,
    definition: Option<&crate::models::modelfile::Definition>,
) -> Result<(String, Vec<Vec<u8>>), InferenceError> {
    let prompt = super::chat::profile_prompt(prompt, definition);
    format_prompt_with(prompt, has_images, &mut |messages| {
        chat_formatter.format(model, messages)
    })
}

fn format_prompt_with(
    prompt: Prompt,
    has_images: bool,
    render: &mut dyn FnMut(&[Message]) -> Result<String, InferenceError>,
) -> Result<(String, Vec<Vec<u8>>), InferenceError> {
    match prompt {
        Prompt::Text(text) => Ok((text, Vec::new())),
        Prompt::Constrained { prompt, .. }
        | Prompt::RequestSystem { prompt, .. }
        | Prompt::Controls { prompt, .. } => format_prompt_with(*prompt, has_images, render),
        Prompt::Chat(mut messages) => {
            let images = if has_images {
                attach_images(&mut messages)?
            } else {
                Vec::new()
            };
            let prompt = render(&messages)?;
            let prompt = if has_images {
                restore_image_markers(prompt, images.len())?
            } else {
                prompt
            };
            Ok((prompt, images))
        }
    }
}

fn reject_count_images(prompt: &Prompt) -> Result<(), InferenceError> {
    if prompt.has_images() {
        return Err(InferenceError::Invalid(
            "prompt token counting supports text only; image inputs are unsupported".into(),
        ));
    }
    Ok(())
}

fn tokenize_text_prompt(model: &LlamaModel, prompt: &str) -> Vec<LlamaToken> {
    // Pinned llama-cpp-2 uses the vocab API instead of str_to_token/AddBos.
    // Include model-configured BOS/EOS and parse template special tokens in
    // exactly the same way for counting and text-generation prefill.
    model.vocab().tokenize(prompt.as_bytes(), true, true)
}

fn output_allowance(
    prompt_tokens: usize,
    positions: usize,
    options: &Options,
    context: u32,
) -> Result<u32, InferenceError> {
    let requested = if options.until_context {
        0
    } else {
        options.max_tokens
    };
    check_budget(prompt_tokens, requested, context)?;
    check_budget(positions, requested, context)?;
    if options.until_context {
        // Both counts were checked against the effective native context before
        // narrowing. A completely full prompt yields an immediate length finish.
        context
            .checked_sub(prompt_tokens.max(positions) as u32)
            .ok_or_else(|| InferenceError::Invalid("prompt exceeds context size".into()))
    } else {
        Ok(options.max_tokens)
    }
}

fn check_budget(prompt: usize, generated: u32, context: u32) -> Result<(), InferenceError> {
    if prompt == 0 {
        return Err(InferenceError::Invalid("prompt produced no tokens".into()));
    }
    if prompt > context as usize || generated as usize > context as usize - prompt {
        return Err(InferenceError::Invalid(format!(
            "prompt ({prompt} tokens) plus max_tokens ({generated}) exceeds context size ({context})"
        )));
    }
    Ok(())
}

fn append_generation_utf8(
    decoder: &mut Decoder,
    bytes: &[u8],
    text: &mut String,
    last: bool,
    stops: &mut Option<StopFilter>,
    emit: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
) -> Result<bool, InferenceError> {
    let Some(stops) = stops else {
        append_stream_utf8(decoder, bytes, text, last, emit)?;
        return Ok(false);
    };
    let mut decoded = String::new();
    append_utf8(decoder, bytes, &mut decoded, last)?;
    stops.push(&decoded, last, &mut |delta| {
        let start = text.len();
        text.push_str(delta);
        emit(&text[start..])
    })
}

fn append_stream_utf8(
    decoder: &mut Decoder,
    bytes: &[u8],
    text: &mut String,
    last: bool,
    emit: &mut dyn FnMut(&str) -> Result<(), InferenceError>,
) -> Result<(), InferenceError> {
    let start = text.len();
    append_utf8(decoder, bytes, text, last)?;
    if text.len() != start {
        emit(&text[start..])?;
    }
    Ok(())
}

fn append_utf8(
    decoder: &mut Decoder,
    bytes: &[u8],
    text: &mut String,
    last: bool,
) -> Result<(), InferenceError> {
    let capacity = decoder
        .max_utf8_buffer_length(bytes.len())
        .ok_or_else(|| InferenceError::Internal("decoded text is too large".into()))?;
    text.reserve(capacity);
    let (result, read, _) = decoder.decode_to_string(bytes, text, last);
    if result != CoderResult::InputEmpty || read != bytes.len() {
        return Err(InferenceError::Internal(
            "UTF-8 decoder could not consume token bytes".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "inference_native_tests.rs"]
mod native_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn controls_unwrap_without_losing_schema_images_or_request_system() {
        assert!(matches!(
            Prompt::Text("plain".into()).with_controls(GenerationControls::default()),
            Prompt::Text(_)
        ));
        let schema = serde_json::json!({"type": "object"});
        let controls = GenerationControls {
            stop: vec!["END".into()],
            ..Default::default()
        };
        let prompt = Prompt::Constrained {
            prompt: Box::new(Prompt::Chat(vec![message("user", "hello world", vec![])])),
            schema: schema.clone(),
        }
        .with_request_system(false)
        .with_controls(controls.clone());
        assert_eq!(prompt.output_schema(), Some(&schema));
        assert_eq!(prompt.generation_controls().unwrap().stop, ["END"]);
        assert_eq!(fixture_token_count(&prompt), 7);
        assert!(!prompt.has_images());
        let definition = crate::models::modelfile::parse("FROM base\nSYSTEM stored").unwrap();
        let profiled = super::super::chat::profile_prompt(prompt, Some(&definition));
        assert_eq!(profiled.output_schema(), Some(&schema));
        let (text, _) = format_prompt_with(profiled, false, &mut |messages| {
            assert_eq!(messages[0].role, "system");
            assert_eq!(messages[0].content, "stored");
            Ok(messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("|"))
        })
        .unwrap();
        assert_eq!(text, "stored|hello world");
        let image = Prompt::Chat(vec![message("user", "look", vec![vec![1]])])
            .with_controls(controls)
            .with_request_system(true);
        assert!(image.has_images());
        assert!(reject_count_images(&image).is_err());
    }

    fn candidates(logits: &[f32]) -> llama_cpp_2::token::data_array::LlamaTokenDataArray {
        use llama_cpp_2::token::{data::LlamaTokenData, data_array::LlamaTokenDataArray};
        LlamaTokenDataArray::new(
            logits
                .iter()
                .enumerate()
                .map(|(id, logit)| LlamaTokenData::new(LlamaToken(id as i32), *logit, 0.0))
                .collect(),
            false,
        )
    }

    #[test]
    fn native_controls_apply_openai_math_to_generated_history_only() {
        for (frequency, presence) in [(0.5, 1.0), (-0.5, -1.0)] {
            let controls = GenerationControls {
                frequency_penalty: frequency,
                presence_penalty: presence,
                logit_bias: vec![(0, 0.25), (2, -2.0)],
                ..Default::default()
            };
            let options = generation_request().options;
            let mut sampler = generation_sampler_with_controls(None, &options, &controls, 3, 128);
            let mut before = candidates(&[4.0, 3.0, 2.0]);
            sampler.apply(&mut before);
            assert_eq!(
                before.data.iter().map(|t| t.logit()).collect::<Vec<_>>(),
                [4.25, 3.0, 0.0]
            );
            // This is all generated history; prompt tokens never enter accept().
            sampler.accept_many([LlamaToken(0), LlamaToken(0), LlamaToken(1)]);
            let mut after = candidates(&[4.0, 3.0, 2.0]);
            sampler.apply(&mut after);
            assert_eq!(
                after.data.iter().map(|t| t.logit()).collect::<Vec<_>>(),
                [
                    4.25 - 2.0 * frequency - presence,
                    3.0 - frequency - presence,
                    0.0,
                ]
            );
        }
        let controls = GenerationControls {
            frequency_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = generation_sampler_with_controls(
            None,
            &generation_request().options,
            &controls,
            2,
            128,
        );
        for expected in [0, 1, 0] {
            let mut logits = candidates(&[2.0, 1.25]);
            sampler.apply(&mut logits);
            let token = logits.selected_token().unwrap();
            assert_eq!(token, LlamaToken(expected));
            sampler.accept(token); // Exactly once per selected token.
        }
        let mut sampler = generation_sampler_with_controls(
            None,
            &generation_request().options,
            &controls,
            2,
            128,
        );
        sampler.accept(LlamaToken(0));
        sampler.accept_many(std::iter::repeat_n(LlamaToken(1), 70));
        let mut logits = candidates(&[2.0, 1.25]);
        sampler.apply(&mut logits);
        assert_eq!(
            logits.data[0].logit(),
            1.0,
            "history must not use a 64-token repetition window"
        );
    }

    #[test]
    fn native_bias_changes_selection_but_never_bypasses_output_filter() {
        for temperature in [0.0, 0.8] {
            let mut options = generation_request().options;
            options.temperature = temperature;
            options.seed = Some(7);
            let controls = GenerationControls {
                logit_bias: vec![(0, 100.0)],
                ..Default::default()
            };
            let mut sampler = generation_sampler_with_controls(None, &options, &controls, 2, 128);
            let mut logits = candidates(&[-10.0, 1.0]);
            sampler.apply(&mut logits);
            assert_eq!(logits.selected_token(), Some(LlamaToken(0)));
            // A model-free hard candidate mask stands in for the grammar filter.
            let filter = LlamaSampler::logit_bias(
                2,
                &[LlamaLogitBias::new(LlamaToken(0), f32::NEG_INFINITY)],
            );
            let mut sampler =
                generation_sampler_with_controls(Some(filter), &options, &controls, 2, 128);
            let mut logits = candidates(&[-10.0, 1.0]);
            sampler.apply(&mut logits);
            assert_eq!(logits.selected_token(), Some(LlamaToken(1)));
        }
    }

    #[test]
    fn unused_controls_preserve_greedy_shortcut_and_native_chain() {
        let options = generation_request().options;
        for controls in [
            GenerationControls::default(),
            GenerationControls {
                stop: vec!["END".into()],
                ..Default::default()
            },
            GenerationControls {
                logit_bias: vec![(0, 0.0)],
                ..Default::default()
            },
        ] {
            assert!(use_controlled_greedy_fast_path(&options, false, &controls));
            assert!(!use_controlled_greedy_fast_path(&options, true, &controls));
            let baseline = format!("{:?}", generation_sampler(None, &options));
            let controlled = format!(
                "{:?}",
                generation_sampler_with_controls(None, &options, &controls, 2, 128)
            );
            assert_eq!(baseline, controlled);
        }
        for controls in [
            GenerationControls {
                frequency_penalty: -1.0,
                ..Default::default()
            },
            GenerationControls {
                presence_penalty: 1.0,
                ..Default::default()
            },
            GenerationControls {
                logit_bias: vec![(0, -1.0)],
                ..Default::default()
            },
        ] {
            assert!(!use_controlled_greedy_fast_path(&options, false, &controls));
        }
    }

    struct StopFixture {
        pieces: Vec<Vec<u8>>,
        eos: bool,
        sampled: Arc<AtomicUsize>,
    }

    impl Engine for StopFixture {
        fn generate(
            &mut self,
            request: GenerationRequest,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            self.generate_stream(request, cancelled, &mut |_| Ok(()))
        }

        fn generate_stream(
            &mut self,
            request: GenerationRequest,
            cancelled: &dyn Fn() -> bool,
            emit: &mut dyn FnMut(String) -> Result<(), InferenceError>,
        ) -> Result<Generation, InferenceError> {
            let controls = request
                .prompt
                .generation_controls()
                .cloned()
                .unwrap_or_default();
            let mut stops = StopFilter::new(controls.stop);
            let mut decoder = UTF_8.new_decoder_without_bom_handling();
            let mut result = Generation {
                text: String::new(),
                prompt_tokens: 5,
                completion_tokens: 0,
                finish_reason: "length",
            };
            for piece in self.pieces.iter().take(request.options.max_tokens as usize) {
                ensure_active(cancelled)?;
                self.sampled.fetch_add(1, Ordering::SeqCst);
                result.completion_tokens += 1;
                if append_generation_utf8(
                    &mut decoder,
                    piece,
                    &mut result.text,
                    false,
                    &mut stops,
                    &mut |delta| emit(delta.to_owned()),
                )? {
                    result.finish_reason = "stop";
                    break;
                }
            }
            if self.eos
                && result.finish_reason != "stop"
                && result.completion_tokens < request.options.max_tokens as usize
            {
                self.sampled.fetch_add(1, Ordering::SeqCst);
                result.completion_tokens += 1;
                result.finish_reason = "stop";
            }
            if append_generation_utf8(
                &mut decoder,
                &[],
                &mut result.text,
                true,
                &mut stops,
                &mut |delta| emit(delta.to_owned()),
            )? {
                result.finish_reason = "stop";
            }
            Ok(result)
        }
    }

    #[tokio::test]
    async fn stop_fixture_stream_and_single_keep_actual_usage_without_leaking_split_utf8() {
        let cases = [
            (
                vec![
                    b"hello ".to_vec(),
                    vec![0xf0, 0x9f],
                    vec![0xa6, 0x80],
                    "終later".as_bytes().to_vec(),
                    b"NEVER".to_vec(),
                ],
                vec!["🦀終"],
                false,
                "hello ",
                4,
                "stop",
            ),
            (
                vec![b"ok END ignored".to_vec(), b"NEVER".to_vec()],
                vec!["ignored", "END"],
                false,
                "ok ",
                1,
                "stop",
            ),
            (
                vec![b"hello EN".to_vec()],
                vec!["END"],
                true,
                "hello EN",
                2,
                "stop",
            ),
            (
                vec![b"hello EN".to_vec()],
                vec!["END"],
                false,
                "hello EN",
                1,
                "length",
            ),
        ];
        for (pieces, stops, eos, text, usage, reason) in cases {
            let sampled = Arc::new(AtomicUsize::new(0));
            let engine = StopFixture {
                pieces,
                eos,
                sampled: sampled.clone(),
            };
            let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
            let controls = GenerationControls {
                stop: stops.into_iter().map(str::to_owned).collect(),
                ..Default::default()
            };
            let request = move || {
                let mut request = generation_request();
                request.options.max_tokens = 100;
                request.prompt = request.prompt.with_controls(controls.clone());
                request
            };
            let single = service.generate(request()).await.unwrap();
            assert_eq!(single.text, text);
            assert_eq!(single.completion_tokens, usage);
            assert_eq!(single.finish_reason, reason);
            let mut stream = service
                .stream_prepared(move || Ok(request()))
                .await
                .unwrap();
            let mut deltas = String::new();
            let mut finished = false;
            while let Some(event) = stream.recv().await {
                match event {
                    GenerationEvent::Delta(delta) => deltas.push_str(&delta),
                    GenerationEvent::Finished(result) => {
                        let result = result.unwrap();
                        assert_eq!(result.text, text);
                        assert_eq!(result.completion_tokens, usage);
                        assert_eq!(result.finish_reason, reason);
                        assert_eq!(result.prompt_tokens, 5);
                        finished = true;
                    }
                }
            }
            assert!(finished);
            assert_eq!(deltas, text);
            assert_eq!(sampled.load(Ordering::SeqCst), usage * 2);
            drop(service);
            worker.await.unwrap();
        }
    }

    #[test]
    fn stop_filter_is_safe_at_every_byte_split_and_flushes_unmatched_utf8_suffixes() {
        let bytes = "café 🦀終hidden".as_bytes();
        for first in 0..=bytes.len() {
            for second in first..=bytes.len() {
                let mut decoder = UTF_8.new_decoder_without_bom_handling();
                let mut stops = StopFilter::new(vec!["🦀終".into(), "hidden".into()]);
                let mut text = String::new();
                let mut deltas = String::new();
                for chunk in [&bytes[..first], &bytes[first..second], &bytes[second..]] {
                    if append_generation_utf8(
                        &mut decoder,
                        chunk,
                        &mut text,
                        false,
                        &mut stops,
                        &mut |delta| {
                            deltas.push_str(delta);
                            Ok(())
                        },
                    )
                    .unwrap()
                    {
                        break;
                    }
                }
                assert!(
                    append_generation_utf8(
                        &mut decoder,
                        &[],
                        &mut text,
                        true,
                        &mut stops,
                        &mut |delta| {
                            deltas.push_str(delta);
                            Ok(())
                        }
                    )
                    .unwrap()
                );
                assert_eq!(text, "café ", "splits {first}, {second}");
                assert_eq!(deltas, text);
            }
        }
        for tail in ["🦀".as_bytes(), b"\xf0\x9f".as_slice()] {
            let mut decoder = UTF_8.new_decoder_without_bom_handling();
            let mut stops = StopFilter::new(vec!["🦀終".into()]);
            let mut text = String::new();
            append_generation_utf8(
                &mut decoder,
                tail,
                &mut text,
                false,
                &mut stops,
                &mut |_| Ok(()),
            )
            .unwrap();
            assert!(text.is_empty());
            assert!(
                !append_generation_utf8(
                    &mut decoder,
                    &[],
                    &mut text,
                    true,
                    &mut stops,
                    &mut |_| Ok(())
                )
                .unwrap()
            );
            assert_eq!(text, if tail.len() == 4 { "🦀" } else { "\u{fffd}" });
        }
    }

    struct VocabValidationFixture {
        calls: Arc<AtomicUsize>,
        discards: Arc<AtomicUsize>,
    }

    impl Engine for VocabValidationFixture {
        fn validate_generation(&self, request: &GenerationRequest) -> Result<(), InferenceError> {
            if let Some(controls) = request.prompt.generation_controls() {
                controls.validate_vocab(3)?;
            }
            Ok(())
        }

        fn generate(
            &mut self,
            _: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Generation {
                text: "ok".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: "stop",
            })
        }

        fn discard_request_state(&mut self) {
            self.discards.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn invalid_bias_id_is_rejected_before_generation_or_discard_including_zero_bias() {
        let calls = Arc::new(AtomicUsize::new(0));
        let discards = Arc::new(AtomicUsize::new(0));
        let engine = VocabValidationFixture {
            calls: calls.clone(),
            discards: discards.clone(),
        };
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        let invalid = || {
            let mut request = generation_request();
            request.prompt = request.prompt.with_controls(GenerationControls {
                logit_bias: vec![(3, 0.0)],
                ..Default::default()
            });
            request
        };
        assert!(matches!(
            service.generate(invalid()).await,
            Err(InferenceError::Invalid(_))
        ));
        let mut stream = service
            .stream_prepared(move || Ok(invalid()))
            .await
            .unwrap();
        assert!(matches!(
            stream.recv().await,
            Some(GenerationEvent::Finished(Err(InferenceError::Invalid(_))))
        ));
        assert!(stream.recv().await.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        service.generate(generation_request()).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        drop(service);
        worker.await.unwrap();
    }

    struct SlowCleanupFixture {
        entered: Arc<tokio::sync::Notify>,
        cleanup: Arc<tokio::sync::Notify>,
        release: std::sync::mpsc::Receiver<()>,
        calls: Arc<AtomicUsize>,
        cleaned: Arc<AtomicBool>,
    }

    impl Engine for SlowCleanupFixture {
        fn generate(
            &mut self,
            _: GenerationRequest,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.entered.notify_one();
                while !cancelled() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                return Err(cancelled_error());
            }
            assert!(
                self.cleaned.load(Ordering::SeqCst),
                "successor overlapped native cleanup"
            );
            Ok(Generation {
                text: "next".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: "stop",
            })
        }

        fn discard_request_state(&mut self) {
            self.cleanup.notify_one();
            self.release.recv_timeout(Duration::from_secs(5)).unwrap();
            self.cleaned.store(true, Ordering::SeqCst);
        }
    }

    async fn slow_cleanup_service() -> (
        InferenceService,
        JoinHandle<()>,
        Arc<tokio::sync::Notify>,
        Arc<tokio::sync::Notify>,
        std::sync::mpsc::Sender<()>,
        Arc<AtomicUsize>,
    ) {
        let entered = Arc::new(tokio::sync::Notify::new());
        let cleanup = Arc::new(tokio::sync::Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let (release, wait) = std::sync::mpsc::channel();
        let engine = SlowCleanupFixture {
            entered: entered.clone(),
            cleanup: cleanup.clone(),
            release: wait,
            calls: calls.clone(),
            cleaned: Arc::new(AtomicBool::new(false)),
        };
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        (service, worker, entered, cleanup, release, calls)
    }

    async fn wait_for_handoff(service: &InferenceService) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.handoff_waiter.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn immediate_followup_after_disconnect_waits_for_cleanup_without_overlap() {
        for streaming in [false, true] {
            let (service, worker, entered, cleanup, release, calls) = slow_cleanup_service().await;
            let mut stream = None;
            let mut single = None;
            if streaming {
                stream = Some(
                    service
                        .stream_prepared(|| Ok(generation_request()))
                        .await
                        .unwrap(),
                );
            } else {
                single = Some(tokio::spawn({
                    let service = service.clone();
                    async move { service.generate(generation_request()).await }
                }));
            }
            entered.notified().await;
            assert!(matches!(
                service
                    .generate_prepared(|| panic!("active overlap prepared"))
                    .await,
                Err(InferenceError::Busy)
            ));
            drop(stream);
            if let Some(single) = single {
                single.abort();
                let _ = single.await;
            }
            // Submit immediately, before waiting for native cancellation/cleanup.
            let prepared = Arc::new(AtomicUsize::new(0));
            let next = tokio::spawn({
                let service = service.clone();
                let prepared = prepared.clone();
                async move {
                    service
                        .generate_prepared(move || {
                            prepared.fetch_add(1, Ordering::SeqCst);
                            Ok(generation_request())
                        })
                        .await
                }
            });
            wait_for_handoff(&service).await;
            cleanup.notified().await;
            assert_eq!(service.admission.available_permits(), 0);
            assert_eq!(prepared.load(Ordering::SeqCst), 0);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(matches!(
                service
                    .stream_prepared(|| panic!("second waiter prepared"))
                    .await,
                Err(InferenceError::Busy)
            ));
            release.send(()).unwrap();
            let result = tokio::time::timeout(Duration::from_secs(2), next)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(result.text, "next");
            assert_eq!(prepared.load(Ordering::SeqCst), 1);
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            drop(service);
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    async fn dropped_handoff_waiter_never_prepares_and_releases_its_wait_slot() {
        let (service, worker, entered, cleanup, release, calls) = slow_cleanup_service().await;
        let stream = service
            .stream_prepared(|| Ok(generation_request()))
            .await
            .unwrap();
        entered.notified().await;
        drop(stream);
        cleanup.notified().await;
        for streaming in [false, true] {
            let waiter = tokio::spawn({
                let service = service.clone();
                async move {
                    if streaming {
                        service
                            .stream_prepared(|| panic!("aborted stream waiter prepared"))
                            .await
                            .map(|_| ())
                    } else {
                        service
                            .generate_prepared(|| panic!("aborted waiter prepared"))
                            .await
                            .map(|_| ())
                    }
                }
            });
            wait_for_handoff(&service).await;
            waiter.abort();
            assert!(waiter.await.unwrap_err().is_cancelled());
            assert_eq!(service.handoff_waiter.available_permits(), 1);
            assert_eq!(service.admission.available_permits(), 0);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
        release.send(()).unwrap();
        let permit = tokio::time::timeout(Duration::from_secs(2), service.admission.acquire())
            .await
            .unwrap()
            .unwrap();
        drop(permit);
        service.generate(generation_request()).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        drop(service);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn handoff_wait_timeout_is_bounded_and_does_not_release_native_admission() {
        let (service, worker, entered, cleanup, release, calls) = slow_cleanup_service().await;
        let stream = service
            .stream_prepared(|| Ok(generation_request()))
            .await
            .unwrap();
        entered.notified().await;
        drop(stream);
        cleanup.notified().await;
        let start = std::time::Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            service.generate_prepared(|| panic!("timed out waiter prepared")),
        )
        .await
        .unwrap();
        assert!(matches!(result, Err(InferenceError::Busy)));
        assert!(start.elapsed() >= CANCELLED_HANDOFF_TIMEOUT);
        assert_eq!(service.admission.available_permits(), 0);
        assert_eq!(service.handoff_waiter.available_permits(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        release.send(()).unwrap();
        drop(service);
        worker.await.unwrap();
    }

    fn embedding_request() -> EmbeddingRequest {
        EmbeddingRequest {
            input: vec!["first input".into(), "second".into()],
            truncate: false,
        }
    }

    #[test]
    fn embedding_normalization_handles_extremes_and_rejects_invalid_vectors() {
        for mut vector in [
            vec![3.0, 4.0],
            vec![f32::MAX, f32::MAX],
            vec![f32::MIN_POSITIVE, 0.0],
        ] {
            normalize_embedding(&mut vector).unwrap();
            assert!(vector.iter().all(|value| value.is_finite()));
            assert!(
                (vector
                    .iter()
                    .map(|&value| f64::from(value).powi(2))
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1e-6
            );
        }
        for mut vector in [vec![], vec![0.0], vec![f32::NAN], vec![f32::INFINITY]] {
            assert!(normalize_embedding(&mut vector).is_err());
        }
    }

    #[test]
    fn embedding_pooling_and_encoder_selection_preserve_model_policy() {
        for value in [None, Some("-1"), Some("0"), Some("1")] {
            assert_eq!(embedding_pooling(value).unwrap(), LlamaPoolingType::Mean);
        }
        assert_eq!(embedding_pooling(Some("2")).unwrap(), LlamaPoolingType::Cls);
        assert_eq!(
            embedding_pooling(Some("3")).unwrap(),
            LlamaPoolingType::Last
        );
        assert!(embedding_pooling(Some("4")).is_err());
        assert!(embedding_pooling(Some("99")).is_err());
        assert!(embedding_encode_mode("t5encoder").unwrap());
        assert!(!embedding_encode_mode("bert").unwrap());
        assert!(!embedding_encode_mode("llama").unwrap());
        assert!(embedding_encode_mode("eagle3").is_err());
    }

    #[test]
    fn embedding_token_truncation_is_explicit_and_counts_evaluated_tokens() {
        let mut tokens = vec![LlamaToken(1), LlamaToken(2), LlamaToken(3)];
        assert!(truncate_embedding_tokens(&mut tokens, 2, false).is_err());
        assert_eq!(tokens.len(), 3);
        truncate_embedding_tokens(&mut tokens, 2, true).unwrap();
        assert_eq!(tokens, [LlamaToken(1), LlamaToken(2)]);
        assert!(truncate_embedding_tokens(&mut Vec::new(), 2, true).is_err());
        assert!(truncate_embedding_tokens(&mut tokens, 0, true).is_err());
    }

    #[test]
    fn embedding_request_limits_bound_batch_and_total_text() {
        assert!(embedding_request().validate().is_ok());
        for input in [
            vec!["x".into(); EMBEDDING_MAX_INPUTS + 1],
            vec!["".into()],
            vec!["x".repeat(EMBEDDING_MAX_BYTES / 2 + 1); 2],
        ] {
            assert!(
                EmbeddingRequest {
                    input,
                    truncate: true
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            EmbeddingRequest {
                input: vec![],
                truncate: false
            }
            .validate()
            .is_err()
        );
    }

    struct EmbeddingFixture {
        calls: Arc<AtomicUsize>,
        discards: Arc<AtomicUsize>,
        gate: Option<(Arc<tokio::sync::Notify>, std::sync::mpsc::Receiver<()>)>,
    }

    impl Engine for EmbeddingFixture {
        fn generate(
            &mut self,
            _: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            panic!("embedding requests must never generate text");
        }

        fn embed(
            &mut self,
            request: EmbeddingRequest,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<EmbeddingResult, InferenceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some((entered, release)) = self.gate.take() {
                entered.notify_one();
                release
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
            ensure_active(cancelled)?;
            if request.input[0] == "fail" {
                return Err(InferenceError::Invalid("fixture embedding error".into()));
            }
            assert_eq!(request.input, ["first input", "second"]);
            assert!(!request.truncate);
            Ok(EmbeddingResult {
                embeddings: vec![vec![0.6, 0.8], vec![0.0, 1.0]],
                prompt_tokens: 7,
            })
        }

        fn discard_request_state(&mut self) {
            self.discards.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn embedding_worker_uses_native_method_and_preserves_generation_state_on_errors() {
        let calls = Arc::new(AtomicUsize::new(0));
        let discards = Arc::new(AtomicUsize::new(0));
        let engine = EmbeddingFixture {
            calls: calls.clone(),
            discards: discards.clone(),
            gate: None,
        };
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        let result = service.embed(embedding_request()).await.unwrap();
        assert_eq!(result.prompt_tokens, 7);
        assert_eq!(result.embeddings, [vec![0.6, 0.8], vec![0.0, 1.0]]);
        assert!(
            service
                .embed(EmbeddingRequest {
                    input: vec!["fail".into()],
                    truncate: false
                })
                .await
                .is_err()
        );
        assert!(
            service
                .embed(EmbeddingRequest {
                    input: vec![],
                    truncate: false
                })
                .await
                .is_err()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        drop(service);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn embedding_default_is_unsupported_without_generating() {
        let (service, worker, calls, discards) = counting_service().await;
        assert!(
            matches!(service.embed(embedding_request()).await, Err(InferenceError::Invalid(message)) if message.contains("native embeddings"))
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        drop(service);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn embedding_cancellation_retains_shared_admission_until_native_exit() {
        let calls = Arc::new(AtomicUsize::new(0));
        let discards = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(tokio::sync::Notify::new());
        let (release, receive) = std::sync::mpsc::channel();
        let engine = EmbeddingFixture {
            calls: calls.clone(),
            discards: discards.clone(),
            gate: Some((entered.clone(), receive)),
        };
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        let pending = tokio::spawn({
            let service = service.clone();
            async move { service.embed(embedding_request()).await }
        });
        entered.notified().await;
        assert!(matches!(
            service.generate(generation_request()).await,
            Err(InferenceError::Busy)
        ));
        pending.abort();
        let _ = pending.await;
        assert!(matches!(
            service.embed(embedding_request()).await,
            Err(InferenceError::Busy)
        ));
        release.send(()).unwrap();
        let permit = service.admission.acquire().await.unwrap();
        drop(permit);
        service.embed(embedding_request()).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        drop(service);
        worker.await.unwrap();
    }

    #[test]
    #[ignore = "requires LMA_TEST_EMBED_GGUF pointing to installed embedding GGUF; loads real weights"]
    fn real_native_embedding_vectors_and_exact_token_usage() {
        let path = std::env::var_os("LMA_TEST_EMBED_GGUF").expect("set LMA_TEST_EMBED_GGUF");
        let mut engine = LlamaEngine::load(ServeConfig {
            model: path.into(),
            gpu_layers: 0,
            context: 256,
            batch: 32,
            micro_batch: 32,
            threads: 2,
            ..ServeConfig::default()
        })
        .unwrap();
        let input = vec![
            "The cat sat on the mat.".to_owned(),
            "A distant galaxy contains many stars.".to_owned(),
        ];
        let expected = input
            .iter()
            .map(|text| {
                engine
                    .session
                    .borrow_owner()
                    .model
                    .vocab()
                    .tokenize(text.as_bytes(), true, false)
                    .len()
            })
            .sum::<usize>();
        let result = engine
            .embed(
                EmbeddingRequest {
                    input: input.clone(),
                    truncate: false,
                },
                &|| false,
            )
            .unwrap();
        assert_eq!(result.prompt_tokens, expected);
        assert_eq!(result.embeddings.len(), 2);
        assert!(result.embeddings[0].len() > 1);
        assert_ne!(result.embeddings[0], result.embeddings[1]);
        for vector in &result.embeddings {
            assert!(vector.iter().all(|value| value.is_finite()));
            assert!(
                (vector
                    .iter()
                    .map(|&value| f64::from(value).powi(2))
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1e-5
            );
        }
        let again = engine
            .embed(
                EmbeddingRequest {
                    input,
                    truncate: false,
                },
                &|| false,
            )
            .unwrap();
        for (a, b) in result
            .embeddings
            .iter()
            .flatten()
            .zip(again.embeddings.iter().flatten())
        {
            assert!((a - b).abs() < 1e-5);
        }
        let long = "hello world ".repeat(512);
        assert!(matches!(
            engine.embed(
                EmbeddingRequest {
                    input: vec![long.clone()],
                    truncate: false
                },
                &|| false
            ),
            Err(InferenceError::Invalid(_))
        ));
        let truncated = engine
            .embed(
                EmbeddingRequest {
                    input: vec![long],
                    truncate: true,
                },
                &|| false,
            )
            .unwrap();
        assert_eq!(
            truncated.prompt_tokens,
            engine
                .session
                .with_dependent(|_, session| session.embedding.as_ref().unwrap().limit)
        );
    }

    #[test]
    fn output_schemas_convert_natively_for_tool_arguments_and_json_values() {
        for schema in [
            serde_json::json!({"type": "object", "properties": {
                "tool": {"enum": ["weather"]},
                "arguments": {"type": "object", "properties": {
                    "city": {"type": "string"}, "days": {"type": "integer"}
                }, "required": ["city"], "additionalProperties": false}
            }, "required": ["tool", "arguments"], "additionalProperties": false}),
            serde_json::json!({"type": "object"}),
            serde_json::json!({"type": "array", "items": {"type": "string"}}),
            serde_json::json!({"enum": ["café", "🦀"]}),
            serde_json::json!({"$defs": {"args": {"type": "integer"}}, "$ref": "#/$defs/args"}),
        ] {
            let grammar = schema_grammar(&schema).unwrap();
            assert!(grammar.contains("root ::="), "{grammar}");
        }
        assert!(matches!(
            schema_grammar(&serde_json::json!({"type": "not-a-json-type"})),
            Err(InferenceError::Invalid(_))
        ));
    }

    #[test]
    fn profile_formatting_is_shared_by_generation_and_counting_without_native_work() {
        let definition = crate::models::modelfile::Definition {
            from: "base".into(),
            system: Some("Profile system".into()),
            messages: vec![crate::models::modelfile::DefinitionMessage {
                role: "assistant".into(),
                content: "Saved reply".into(),
            }],
            ..Default::default()
        };
        let prompt = || Prompt::Chat(vec![message("user", "Current question", vec![])]);
        let mut render = |messages: &[Message]| {
            Ok(messages
                .iter()
                .map(|message| format!("{}:{}\n", message.role, message.content))
                .collect::<String>())
        };
        let generated = format_prompt_with(
            super::super::chat::profile_prompt(prompt(), Some(&definition)),
            false,
            &mut render,
        )
        .unwrap();
        let schema = serde_json::json!({"type":"object"});
        let counted = Prompt::Constrained {
            prompt: Box::new(prompt()),
            schema: schema.clone(),
        }
        .with_request_system(false);
        assert_eq!(counted.output_schema(), Some(&schema));
        assert!(!counted.has_images());
        let counted = format_prompt_with(
            super::super::chat::profile_prompt(counted, Some(&definition)),
            false,
            &mut render,
        )
        .unwrap();
        assert_eq!(generated, counted);
        assert_eq!(
            generated.0,
            "system:Profile system\nassistant:Saved reply\nuser:Current question\n"
        );
        let raw = format_prompt_with(
            super::super::chat::profile_prompt(
                Prompt::Text("raw bytes 🦀".into()),
                Some(&definition),
            ),
            false,
            &mut |_| panic!("raw prompt must bypass templates"),
        )
        .unwrap();
        assert_eq!(raw.0, "raw bytes 🦀");
        assert!(raw.1.is_empty());
        let image =
            Prompt::Chat(vec![message("user", "image", vec![vec![1]])]).with_request_system(false);
        assert!(image.has_images());
        assert!(reject_count_images(&image).is_err());
    }

    #[test]
    fn constraints_unwrap_recursively_for_images_and_prompt_counts() {
        let wrap = |prompt| Prompt::Constrained {
            prompt: Box::new(Prompt::Constrained {
                prompt: Box::new(prompt),
                schema: serde_json::json!({"type": "string"}),
            }),
            schema: serde_json::json!({"type": "object"}),
        };
        let text = wrap(Prompt::Text("hello café 🦀".into()));
        assert!(!text.has_images());
        assert!(reject_count_images(&text).is_ok());
        assert_eq!(fixture_token_count(&text), 5);
        assert_eq!(text.output_schema().unwrap()["type"], "object");
        let image = wrap(Prompt::Chat(vec![message("user", "look", vec![vec![1]])]));
        assert!(image.has_images());
        assert!(matches!(
            reject_count_images(&image),
            Err(InferenceError::Invalid(_))
        ));
    }

    fn assert_greedy_matches_native(logits: &[f32]) {
        use llama_cpp_2::token::{data::LlamaTokenData, data_array::LlamaTokenDataArray};

        let original: Vec<_> = logits.iter().map(|value| value.to_bits()).collect();
        let selected = greedy_token(logits).unwrap();
        let mut options = generation_request().options;
        options.temperature = 0.0;
        for mut sampler in [LlamaSampler::greedy(), generation_sampler(None, &options)] {
            let mut candidates = LlamaTokenDataArray::new(
                logits
                    .iter()
                    .enumerate()
                    .map(|(index, logit)| {
                        LlamaTokenData::new(LlamaToken(index as i32), *logit, 0.0)
                    })
                    .collect(),
                false,
            );
            sampler.apply(&mut candidates);
            assert_eq!(
                Some(selected),
                candidates.selected_token(),
                "logits={logits:?}"
            );
            sampler.accept(selected);
        }
        assert_eq!(
            logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original
        );
    }

    #[test]
    fn borrowed_greedy_matches_native_for_ties_negatives_infinities_and_nans() {
        assert!(greedy_token(&[]).is_none());
        for (logits, expected) in [
            (vec![4.0, 4.0, 1.0], 0),
            (vec![-3.0, -1.0, -1.0], 1),
            (vec![-0.0, 0.0], 0),
            (vec![f32::NEG_INFINITY; 3], 0),
            (vec![0.0, f32::INFINITY, f32::INFINITY], 1),
            (vec![f32::NAN, f32::INFINITY, 9.0], 0),
            (vec![-3.0, f32::NAN, -1.0], 2),
            (vec![f32::NAN; 3], 0),
            (vec![1.0, 2.0, 3.0, 2.0], 2),
        ] {
            assert_eq!(greedy_token(&logits), Some(LlamaToken(expected)));
            assert_greedy_matches_native(&logits);
        }
        let values = [
            f32::NEG_INFINITY,
            -3.0,
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(0xffc0_0123),
        ];
        for a in values {
            assert_greedy_matches_native(&[a]);
            for b in values {
                assert_greedy_matches_native(&[a, b]);
                for c in values {
                    assert_greedy_matches_native(&[a, b, c]);
                }
            }
        }
    }

    #[test]
    fn greedy_shortcut_never_bypasses_grammar_or_stochastic_sampling() {
        let mut options = generation_request().options;
        for temperature in [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            0.8,
            1.0,
            2.0,
            f32::INFINITY,
            f32::NAN,
        ] {
            options.temperature = temperature;
            assert_eq!(use_greedy_fast_path(&options, false), temperature == 0.0);
            assert!(!use_greedy_fast_path(&options, true));
        }
    }

    #[derive(Default)]
    struct MemoryFixture {
        removal_succeeds: bool,
        empty: bool,
        removals: usize,
        checks: std::cell::Cell<usize>,
        clears: usize,
    }

    impl GenerationMemory for MemoryFixture {
        fn remove_all(&mut self) -> bool {
            self.removals += 1;
            self.removal_succeeds
        }

        fn is_empty(&self) -> bool {
            self.checks.set(self.checks.get() + 1);
            self.empty
        }

        fn full_clear(&mut self) {
            self.clears += 1;
            self.empty = true;
        }
    }

    #[test]
    fn attention_reset_is_logical_checked_and_idempotent() {
        let mut state = GenerationState::new(false, false);
        let mut memory = MemoryFixture {
            removal_succeeds: true,
            empty: true,
            ..Default::default()
        };
        state.reset(&mut memory);
        assert_eq!(
            (memory.removals, memory.checks.get(), memory.clears),
            (0, 0, 0)
        );
        state.mark_dirty();
        assert!(state.dirty);
        state.reset(&mut memory);
        assert!(!state.dirty);
        assert_eq!(
            (memory.removals, memory.checks.get(), memory.clears),
            (1, 1, 0)
        );
        state.reset(&mut memory);
        state.reset(&mut memory);
        assert_eq!(
            (memory.removals, memory.checks.get(), memory.clears),
            (1, 1, 0)
        );
        state.mark_dirty();
        state.reset(&mut memory);
        assert_eq!(
            (memory.removals, memory.checks.get(), memory.clears),
            (2, 2, 0)
        );
    }

    #[test]
    fn unsupported_or_incomplete_logical_removal_falls_back_to_one_full_clear() {
        for (removal_succeeds, empty) in [(false, false), (false, true), (true, false)] {
            let mut state = GenerationState::new(false, false);
            let mut memory = MemoryFixture {
                removal_succeeds,
                empty,
                ..Default::default()
            };
            state.mark_dirty();
            state.reset(&mut memory);
            assert_eq!(memory.removals, 1);
            assert_eq!(memory.checks.get(), usize::from(removal_succeeds));
            assert_eq!(memory.clears, 1);
            assert!(!state.dirty);
            state.reset(&mut memory);
            assert_eq!((memory.removals, memory.clears), (1, 1));
        }
    }

    #[test]
    fn recurrent_hybrid_and_vision_state_keep_full_clear_semantics() {
        for (recurrent, hybrid) in [(true, false), (false, true), (true, true)] {
            let mut state = GenerationState::new(recurrent, hybrid);
            assert!(!state.ordinary_attention);
            let mut memory = MemoryFixture::default();
            state.mark_dirty();
            state.reset(&mut memory);
            state.reset(&mut memory);
            assert_eq!(
                (memory.removals, memory.checks.get(), memory.clears),
                (0, 0, 1)
            );
        }
        let mut state = GenerationState::new(false, false);
        let mut memory = MemoryFixture {
            removal_succeeds: true,
            empty: true,
            ..Default::default()
        };
        // Vision preparation/evaluation and failed prefix mutations require a
        // full clear even if subsequent ordinary token decodes also mark dirty.
        state.require_full_clear();
        state.mark_dirty();
        assert!(state.dirty && state.full_clear_required);
        state.reset(&mut memory);
        state.reset(&mut memory);
        assert_eq!(
            (memory.removals, memory.checks.get(), memory.clears),
            (0, 0, 1)
        );
        assert!(!state.full_clear_required);
        state.mark_dirty();
        state.reset(&mut memory);
        assert_eq!(
            (memory.removals, memory.checks.get(), memory.clears),
            (1, 1, 1)
        );
    }

    #[test]
    fn partial_decode_failure_and_cancellation_reset_without_a_published_prefix() {
        for (succeeded, cancelled) in [(false, false), (false, true), (true, true)] {
            let mut state = GenerationState::new(false, false);
            let mut memory = MemoryFixture {
                removal_succeeds: true,
                empty: true,
                ..Default::default()
            };
            state.mark_dirty();
            // No cached prompt is needed to know that native work must be invalidated.
            assert!(!retain_prefix(true, succeeded, cancelled));
            state.reset(&mut memory);
            state.reset(&mut memory);
            assert_eq!(
                (memory.removals, memory.checks.get(), memory.clears),
                (1, 1, 0)
            );
            assert!(!state.dirty);
        }
    }

    #[test]
    fn bounded_swa_trades_full_history_reuse_for_memory_without_changing_other_defaults() {
        let config = ServeConfig::default();
        let params = generation_context_params(&config, KvCacheType::F16);
        assert!(!params.swa_full());
        assert_eq!(params.n_ctx(), NonZeroU32::new(config.context));
        assert_eq!(params.n_batch(), config.batch);
        assert_eq!(params.n_ubatch(), config.micro_batch);
        assert_eq!(params.n_threads(), config.threads);
        assert_eq!(params.n_threads_batch(), config.threads);
        assert_eq!(params.type_k(), KvCacheType::F16);
        assert_eq!(params.type_v(), KvCacheType::F16);
        assert_eq!(params.pooling_type(), LlamaPoolingType::None);
        assert_eq!(params.flash_attention_policy(), -1);

        let cached = [1, 2, 3, 4];
        let keep = prefix_to_reuse(&cached, &[1, 2, 3, 4, 5], true);
        assert_eq!(keep, 4);
        assert!(prefix_is_resident(keep, 0, 3));
        // A bounded SWA cache may roll out the prefix despite exact token matches.
        assert!(!prefix_is_resident(keep, 1, 4));
        assert!(!prefix_is_resident(keep, 0, 2));
        assert!(!prefix_is_resident(keep, -1, -1));
        assert!(!prefix_is_resident(0, 0, 3));
    }

    #[test]
    fn sampler_filter_precedes_greedy_and_stochastic_selection() {
        // Use a model-free candidate filter in the grammar slot, then inspect
        // the native chain: grammar itself requires a model vocabulary.
        for temperature in [0.0, 0.8] {
            let mut options = generation_request().options;
            options.temperature = temperature;
            let sampler = generation_sampler(Some(LlamaSampler::top_k(7)), &options);
            let debug = format!("{sampler:?}");
            let filter = debug.find("top-k").unwrap();
            let selector = debug
                .find(if temperature == 0.0 {
                    "greedy"
                } else {
                    "top-p"
                })
                .unwrap();
            assert!(filter < selector, "{debug}");
        }
    }

    #[test]
    fn streaming_utf8_emits_only_complete_new_text_and_flushes_once() {
        let mut decoder = UTF_8.new_decoder_without_bom_handling();
        let mut text = String::new();
        let mut deltas = Vec::new();
        let expected = "Hi 🦀 café";
        for byte in expected.as_bytes() {
            append_stream_utf8(&mut decoder, &[*byte], &mut text, false, &mut |delta| {
                assert!(!delta.is_empty());
                deltas.push(delta.to_owned());
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(deltas.concat(), expected);
        assert_eq!(
            deltas.iter().filter(|delta| delta.as_str() == "🦀").count(),
            1
        );
        append_stream_utf8(&mut decoder, &[0xe2], &mut text, false, &mut |delta| {
            deltas.push(delta.to_owned());
            Ok(())
        })
        .unwrap();
        assert_eq!(deltas.concat(), expected);
        append_stream_utf8(&mut decoder, &[], &mut text, true, &mut |delta| {
            deltas.push(delta.to_owned());
            Ok(())
        })
        .unwrap();
        assert_eq!(deltas.concat(), format!("{expected}\u{fffd}"));
        assert_eq!(deltas.concat(), text);
    }

    #[test]
    fn native_output_callback_borrows_only_the_new_accumulated_text() {
        let mut decoder = UTF_8.new_decoder_without_bom_handling();
        let mut text = String::with_capacity(128);
        text.push_str("prefix ");
        let buffer = text.as_ptr();
        let delta_start = buffer.wrapping_add(text.len());
        let mut calls = 0;
        append_stream_utf8(
            &mut decoder,
            "🦀 café".as_bytes(),
            &mut text,
            false,
            &mut |delta| {
                assert_eq!(delta, "🦀 café");
                assert_eq!(delta.as_ptr(), delta_start);
                calls += 1;
                Ok(())
            },
        )
        .unwrap();
        append_stream_utf8(&mut decoder, &[], &mut text, true, &mut |_| {
            panic!("flush must not emit an empty delta")
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(text.as_ptr(), buffer);
        assert_eq!(text, "prefix 🦀 café");
    }

    #[test]
    fn native_output_noop_callback_retains_text_and_callback_errors_propagate() {
        let mut decoder = UTF_8.new_decoder_without_bom_handling();
        let mut text = String::new();
        for byte in "Hi 🦀 café".as_bytes() {
            append_stream_utf8(&mut decoder, &[*byte], &mut text, false, &mut |_| Ok(())).unwrap();
        }
        append_stream_utf8(&mut decoder, &[], &mut text, true, &mut |_| Ok(())).unwrap();
        assert_eq!(text, "Hi 🦀 café");

        for (bytes, last, expected) in [
            (b"hello".as_slice(), false, "hello"),
            (b"\xe2".as_slice(), true, "\u{fffd}"),
        ] {
            let mut decoder = UTF_8.new_decoder_without_bom_handling();
            let mut text = String::new();
            let mut calls = 0;
            let result = append_stream_utf8(&mut decoder, bytes, &mut text, last, &mut |delta| {
                assert_eq!(delta, expected);
                calls += 1;
                Err(cancelled_error())
            });
            assert!(
                matches!(result, Err(InferenceError::Internal(error)) if error == "request cancelled")
            );
            assert_eq!(calls, 1);
            assert_eq!(text, expected);
        }
    }

    #[tokio::test]
    async fn stream_queue_writable_and_closed_do_not_park_the_runtime() {
        let (reply, mut stream) = mpsc::channel(2);
        // blocking_send would panic in this runtime, even with free capacity.
        send_stream_event(&reply, GenerationEvent::Delta("first".into())).unwrap();
        send_stream_event(
            &reply,
            GenerationEvent::Finished(Err(InferenceError::Invalid("done".into()))),
        )
        .unwrap();
        assert_eq!(stream.len(), 2);
        stream.close();
        assert!(matches!(
            send_stream_event(&reply, GenerationEvent::Delta("closed".into())),
            Err(InferenceError::Internal(error)) if error == "request cancelled"
        ));
        assert!(
            matches!(stream.recv().await, Some(GenerationEvent::Delta(text)) if text == "first")
        );
        assert!(matches!(stream.recv().await,
            Some(GenerationEvent::Finished(Err(InferenceError::Invalid(error)))) if error == "done"));
        assert!(stream.recv().await.is_none());
        drop(stream);
        assert!(matches!(
            send_stream_event(&reply, GenerationEvent::Delta("dropped".into())),
            Err(InferenceError::Internal(error)) if error == "request cancelled"
        ));
    }

    #[tokio::test]
    async fn stream_queue_full_preserves_order_and_wakes_on_disconnect() {
        use std::time::Duration;
        for disconnect in [false, true] {
            let (reply, mut stream) = mpsc::channel(1);
            send_stream_event(&reply, GenerationEvent::Delta("first".into())).unwrap();
            let (entered, wait) = oneshot::channel();
            let mut blocked = tokio::task::spawn_blocking(move || {
                entered.send(()).unwrap();
                send_stream_event(&reply, GenerationEvent::Delta("second".into()))
            });
            tokio::time::timeout(Duration::from_secs(5), wait)
                .await
                .unwrap()
                .unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(25), &mut blocked)
                    .await
                    .is_err()
            );
            if disconnect {
                drop(stream);
                let result = tokio::time::timeout(Duration::from_secs(5), blocked)
                    .await
                    .unwrap()
                    .unwrap();
                assert!(
                    matches!(result, Err(InferenceError::Internal(error)) if error == "request cancelled")
                );
            } else {
                assert!(
                    matches!(stream.recv().await, Some(GenerationEvent::Delta(text)) if text == "first")
                );
                tokio::time::timeout(Duration::from_secs(5), blocked)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(
                    matches!(stream.recv().await, Some(GenerationEvent::Delta(text)) if text == "second")
                );
                assert!(stream.recv().await.is_none());
            }
        }
    }

    struct GatedStreamEngine {
        entered: Option<oneshot::Sender<()>>,
        release: std::sync::mpsc::Receiver<()>,
        fail: bool,
    }

    impl Engine for GatedStreamEngine {
        fn generate(
            &mut self,
            _: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            panic!("stream must use the incremental engine method")
        }

        fn generate_stream(
            &mut self,
            _: GenerationRequest,
            cancelled: &dyn Fn() -> bool,
            emit: &mut dyn FnMut(String) -> Result<(), InferenceError>,
        ) -> Result<Generation, InferenceError> {
            emit("first 🦀".into())?;
            self.entered.take().unwrap().send(()).unwrap();
            self.release
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            ensure_active(cancelled)?;
            if self.fail {
                return Err(InferenceError::Internal("decode fixture failed".into()));
            }
            emit(" second".into())?;
            Ok(Generation {
                text: "first 🦀 second".into(),
                prompt_tokens: 3,
                completion_tokens: 2,
                finish_reason: "stop",
            })
        }
    }

    #[tokio::test]
    async fn stream_delivers_during_generation_and_finishes_with_usage_or_error() {
        for fail in [false, true] {
            let (entered, wait) = oneshot::channel();
            let (release, gate) = std::sync::mpsc::channel();
            let (service, worker) = InferenceService::spawn(move || {
                Ok(GatedStreamEngine {
                    entered: Some(entered),
                    release: gate,
                    fail,
                })
            })
            .await
            .unwrap();
            let executor = std::thread::current().id();
            let mut stream = service
                .stream_prepared(move || {
                    assert_ne!(std::thread::current().id(), executor);
                    Ok(generation_request())
                })
                .await
                .unwrap();
            wait.await.unwrap();
            assert!(
                matches!(stream.recv().await, Some(GenerationEvent::Delta(text)) if text == "first 🦀")
            );
            assert!(matches!(
                service
                    .stream_prepared(|| panic!("busy stream prepared"))
                    .await,
                Err(InferenceError::Busy)
            ));
            release.send(()).unwrap();
            if !fail {
                assert!(
                    matches!(stream.recv().await, Some(GenerationEvent::Delta(text)) if text == " second")
                );
            }
            match stream.recv().await.unwrap() {
                GenerationEvent::Finished(Ok(generation)) if !fail => {
                    assert_eq!(generation.text, "first 🦀 second");
                    assert_eq!(generation.prompt_tokens, 3);
                    assert_eq!(generation.completion_tokens, 2);
                    assert_eq!(generation.finish_reason, "stop");
                }
                GenerationEvent::Finished(Err(InferenceError::Internal(error))) if fail => {
                    assert_eq!(error, "decode fixture failed");
                }
                event => panic!("unexpected event: {event:?}"),
            }
            assert!(stream.recv().await.is_none());
            drop(service);
            worker.await.unwrap();
        }
    }

    struct BackpressureEngine {
        progress: Arc<AtomicUsize>,
        full: Arc<tokio::sync::Notify>,
        cancelled: Arc<tokio::sync::Notify>,
        release: std::sync::mpsc::Receiver<()>,
        discards: Arc<AtomicUsize>,
    }

    impl Engine for BackpressureEngine {
        fn generate(
            &mut self,
            _: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            panic!("unexpected non-stream generation")
        }

        fn generate_stream(
            &mut self,
            _: GenerationRequest,
            cancelled: &dyn Fn() -> bool,
            emit: &mut dyn FnMut(String) -> Result<(), InferenceError>,
        ) -> Result<Generation, InferenceError> {
            for index in 0..STREAM_CAPACITY + 1 {
                if index == STREAM_CAPACITY {
                    self.full.notify_one();
                }
                if let Err(error) = emit("x".into()) {
                    assert!(cancelled());
                    self.cancelled.notify_one();
                    // Simulate native cancellation/cleanup that has not exited yet.
                    self.release
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                    return Err(error);
                }
                self.progress.fetch_add(1, Ordering::SeqCst);
            }
            panic!("bounded stream should block before reaching here")
        }

        fn discard_request_state(&mut self) {
            self.discards.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn stream_backpressure_and_disconnect_retain_admission_until_native_exit() {
        let progress = Arc::new(AtomicUsize::new(0));
        let full = Arc::new(tokio::sync::Notify::new());
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let discards = Arc::new(AtomicUsize::new(0));
        let (release, gate) = std::sync::mpsc::channel();
        let engine = BackpressureEngine {
            progress: progress.clone(),
            full: full.clone(),
            cancelled: cancelled.clone(),
            release: gate,
            discards: discards.clone(),
        };
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        let stream = service
            .stream_prepared(|| Ok(generation_request()))
            .await
            .unwrap();
        full.notified().await;
        assert_eq!(progress.load(Ordering::SeqCst), STREAM_CAPACITY);
        assert_eq!(stream.len(), STREAM_CAPACITY);
        assert!(matches!(
            service
                .generate_prepared(|| panic!("busy request prepared"))
                .await,
            Err(InferenceError::Busy)
        ));
        drop(stream);
        cancelled.notified().await;
        assert_eq!(progress.load(Ordering::SeqCst), STREAM_CAPACITY);
        assert!(matches!(
            service
                .stream_prepared(|| panic!("cancelled native work still owns admission"))
                .await,
            Err(InferenceError::Busy)
        ));
        let admission = service.admission.clone();
        release.send(()).unwrap();
        drop(service);
        worker.await.unwrap();
        assert_eq!(discards.load(Ordering::SeqCst), 1);
        assert_eq!(admission.available_permits(), 1);
    }

    #[tokio::test]
    async fn stream_preparation_errors_and_default_engine_results_are_terminal_events() {
        let (service, worker, calls, discards) = counting_service().await;
        let mut stream = service
            .stream_prepared(|| Err(InferenceError::Invalid("bad schema".into())))
            .await
            .unwrap();
        assert!(matches!(stream.recv().await,
            Some(GenerationEvent::Finished(Err(InferenceError::Invalid(message)))) if message == "bad schema"));
        assert!(stream.recv().await.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let mut stream = service
            .stream_prepared(|| Ok(generation_request()))
            .await
            .unwrap();
        assert!(matches!(stream.recv().await, Some(GenerationEvent::Delta(text)) if text == "ok"));
        assert!(matches!(
            stream.recv().await,
            Some(GenerationEvent::Finished(Ok(_)))
        ));
        assert!(stream.recv().await.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        drop(service);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn dropped_stream_during_preparation_skips_engine_and_retains_admission() {
        let (service, worker, calls, discards) = counting_service().await;
        let (entered, wait) = oneshot::channel();
        let (release, gate) = std::sync::mpsc::channel();
        let stream = service
            .stream_prepared(move || {
                entered.send(()).unwrap();
                gate.recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                Ok(generation_request())
            })
            .await
            .unwrap();
        wait.await.unwrap();
        drop(stream);
        assert!(matches!(
            service.stream_prepared(|| panic!("still preparing")).await,
            Err(InferenceError::Busy)
        ));
        release.send(()).unwrap();
        drop(service);
        worker.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
    }

    struct CountingEngine {
        calls: Arc<AtomicUsize>,
        discards: Arc<AtomicUsize>,
    }

    impl Engine for CountingEngine {
        fn generate(
            &mut self,
            _: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Generation {
                text: "ok".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: "stop",
            })
        }

        fn discard_request_state(&mut self) {
            self.discards.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn generation_request() -> GenerationRequest {
        GenerationRequest {
            prompt: Prompt::Text("hello".into()),
            options: Options {
                until_context: false,
                max_tokens: 1,
                temperature: 0.0,
                top_p: 1.0,
                seed: None,
            },
        }
    }

    async fn counting_service() -> (
        InferenceService,
        JoinHandle<()>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let discards = Arc::new(AtomicUsize::new(0));
        let engine = CountingEngine {
            calls: calls.clone(),
            discards: discards.clone(),
        };
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        (service, worker, calls, discards)
    }

    #[tokio::test]
    async fn preparation_runs_off_executor_and_errors_leave_engine_state_alone() {
        let (service, worker, calls, discards) = counting_service().await;
        let executor = std::thread::current().id();
        service.generate(generation_request()).await.unwrap();
        let error = service
            .generate_prepared(move || {
                assert_ne!(std::thread::current().id(), executor);
                Err(InferenceError::Invalid("bad image".into()))
            })
            .await;
        assert!(matches!(error, Err(InferenceError::Invalid(message)) if message == "bad image"));
        service
            .generate_prepared(move || {
                assert_ne!(std::thread::current().id(), executor);
                Ok(generation_request())
            })
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        drop(service);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn cancelled_preparation_keeps_admission_until_it_finishes_and_skips_model_work() {
        let (service, worker, calls, discards) = counting_service().await;
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let first_service = service.clone();
        let first = tokio::spawn(async move {
            first_service
                .generate_prepared(move || {
                    entered_tx.send(()).unwrap();
                    wait.recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                    Ok(generation_request())
                })
                .await
        });
        entered_rx.await.unwrap();
        for disconnected in [false, true] {
            if disconnected {
                first.abort();
                // Wait for receiver cancellation without relying on scheduler timing.
                while !first.is_finished() {
                    tokio::task::yield_now().await;
                }
            }
            assert!(matches!(
                service
                    .generate_prepared(|| { panic!("busy requests must not run preparation") })
                    .await,
                Err(InferenceError::Busy)
            ));
        }
        assert!(first.await.unwrap_err().is_cancelled());
        release.send(()).unwrap();
        drop(service);
        worker.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn cancelled_queued_job_never_starts_preparation() {
        let (service, worker, calls, discards) = counting_service().await;
        let (reply, receive) = oneshot::channel();
        drop(receive);
        let permit = service.admission.clone().try_acquire_owned().unwrap();
        assert!(
            service
                .sender
                .try_send(Job {
                    request: JobRequest::Prepare(Box::new(|| panic!("cancelled job was prepared"))),
                    reply: JobReply::Single(reply),
                    _permit: permit,
                })
                .is_ok()
        );
        drop(service);
        worker.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
    }

    struct PromptEngine {
        generations: Arc<AtomicUsize>,
        counts: Arc<AtomicUsize>,
        discards: Arc<AtomicUsize>,
        cache: Vec<String>,
        count_gate: Option<(Arc<tokio::sync::Notify>, std::sync::mpsc::Receiver<()>)>,
    }

    fn fixture_token_count(prompt: &Prompt) -> usize {
        match prompt {
            Prompt::Constrained { prompt, .. }
            | Prompt::RequestSystem { prompt, .. }
            | Prompt::Controls { prompt, .. } => fixture_token_count(prompt),
            Prompt::Text(text) => 2 + text.split_whitespace().count(),
            Prompt::Chat(messages) => {
                2 + messages
                    .iter()
                    .map(|m| 3 + m.content.split_whitespace().count())
                    .sum::<usize>()
            }
        }
    }

    impl Engine for PromptEngine {
        fn generate(
            &mut self,
            request: GenerationRequest,
            _: &dyn Fn() -> bool,
        ) -> Result<Generation, InferenceError> {
            self.generations.fetch_add(1, Ordering::SeqCst);
            self.cache = vec!["reusable generation prefix".into()];
            Ok(Generation {
                text: "ok".into(),
                prompt_tokens: fixture_token_count(&request.prompt),
                completion_tokens: 1,
                finish_reason: "stop",
            })
        }

        fn count_prompt(
            &self,
            prompt: Prompt,
            _: &dyn Fn() -> bool,
        ) -> Result<usize, InferenceError> {
            assert_eq!(self.cache, vec!["reusable generation prefix"]);
            self.counts.fetch_add(1, Ordering::SeqCst);
            if let Some((entered, release)) = &self.count_gate {
                entered.notify_one();
                release
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
            if matches!(&prompt, Prompt::Text(text) if text == "fail count") {
                return Err(InferenceError::Invalid("fixture count error".into()));
            }
            Ok(fixture_token_count(&prompt))
        }

        fn discard_request_state(&mut self) {
            self.discards.fetch_add(1, Ordering::SeqCst);
            self.cache.clear();
        }
    }

    impl Drop for PromptEngine {
        fn drop(&mut self) {
            assert_eq!(self.cache, vec!["reusable generation prefix"]);
            assert_eq!(self.discards.load(Ordering::SeqCst), 0);
        }
    }

    fn prompt_engine() -> PromptEngine {
        PromptEngine {
            generations: Arc::new(AtomicUsize::new(0)),
            counts: Arc::new(AtomicUsize::new(0)),
            discards: Arc::new(AtomicUsize::new(0)),
            cache: Vec::new(),
            count_gate: None,
        }
    }

    #[tokio::test]
    async fn count_jobs_use_tokenizer_and_blocking_preparation_without_generation_or_cache_mutation()
     {
        let engine = prompt_engine();
        let generations = engine.generations.clone();
        let counts = engine.counts.clone();
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        let generated = service.generate(generation_request()).await.unwrap();
        let executor = std::thread::current().id();
        let tokens = service
            .count_prompt_prepared(move || {
                assert_ne!(std::thread::current().id(), executor);
                Ok(Prompt::Text("hello".into()))
            })
            .await
            .unwrap();
        assert_eq!(tokens, fixture_token_count(&Prompt::Text("hello".into())));
        assert_eq!(tokens, generated.prompt_tokens);
        assert!(matches!(
            service
                .count_prompt_prepared(|| Err(InferenceError::Invalid("preparation error".into())))
                .await,
            Err(InferenceError::Invalid(_))
        ));
        assert!(matches!(
            service
                .count_prompt_prepared(|| Ok(Prompt::Text("fail count".into())))
                .await,
            Err(InferenceError::Invalid(_))
        ));
        assert_eq!(
            service
                .count_prompt_prepared(|| Ok(Prompt::Text("hello 🦀 again".into())))
                .await
                .unwrap(),
            5
        );
        assert_eq!(
            service
                .count_prompt_prepared(|| Ok(Prompt::Constrained {
                    prompt: Box::new(Prompt::Constrained {
                        prompt: Box::new(Prompt::Text("hello 🦀 again".into())),
                        schema: serde_json::Value::Null,
                    }),
                    // Counting ignores output grammar, including unusable schemas.
                    schema: serde_json::json!({"type": "not-a-json-type"}),
                }))
                .await
                .unwrap(),
            5
        );
        assert_eq!(generations.load(Ordering::SeqCst), 1);
        assert_eq!(counts.load(Ordering::SeqCst), 4);
        drop(service);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn count_default_is_unsupported_without_generating_or_discarding_cache() {
        let (service, worker, calls, discards) = counting_service().await;
        let result = service
            .count_prompt_prepared(|| Ok(Prompt::Text("hello".into())))
            .await;
        assert!(
            matches!(result, Err(InferenceError::Invalid(message)) if message.contains("does not support"))
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(discards.load(Ordering::SeqCst), 0);
        drop(service);
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn cancelled_counts_keep_admission_and_do_not_discard_generation_cache() {
        for during_preparation in [false, true] {
            let mut engine = prompt_engine();
            let generations = engine.generations.clone();
            let counts = engine.counts.clone();
            let entered = Arc::new(tokio::sync::Notify::new());
            let (release, wait) = std::sync::mpsc::channel();
            let mut preparation_gate = if during_preparation {
                Some(wait)
            } else {
                engine.count_gate = Some((entered.clone(), wait));
                None
            };
            let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
            service.generate(generation_request()).await.unwrap();
            let first_service = service.clone();
            let prepared = entered.clone();
            let first = tokio::spawn(async move {
                first_service
                    .count_prompt_prepared(move || {
                        if let Some(wait) = preparation_gate.take() {
                            prepared.notify_one();
                            wait.recv_timeout(std::time::Duration::from_secs(5))
                                .unwrap();
                        }
                        Ok(Prompt::Text("hello".into()))
                    })
                    .await
            });
            entered.notified().await;
            for disconnect in [false, true] {
                if disconnect {
                    first.abort();
                    while !first.is_finished() {
                        tokio::task::yield_now().await;
                    }
                }
                assert!(matches!(
                    service
                        .count_prompt_prepared(|| panic!("busy count must not prepare"))
                        .await,
                    Err(InferenceError::Busy)
                ));
                assert!(matches!(
                    service
                        .generate_prepared(|| panic!("busy generation must not prepare"))
                        .await,
                    Err(InferenceError::Busy)
                ));
            }
            assert!(first.await.unwrap_err().is_cancelled());
            release.send(()).unwrap();
            drop(service);
            worker.await.unwrap();
            assert_eq!(generations.load(Ordering::SeqCst), 1);
            assert_eq!(
                counts.load(Ordering::SeqCst),
                usize::from(!during_preparation)
            );
        }
    }

    #[tokio::test]
    async fn cancelled_queued_count_skips_preparation_and_tokenizer() {
        let engine = prompt_engine();
        let counts = engine.counts.clone();
        let (service, worker) = InferenceService::spawn(move || Ok(engine)).await.unwrap();
        service.generate(generation_request()).await.unwrap();
        let (reply, receive) = oneshot::channel();
        drop(receive);
        let permit = service.admission.clone().try_acquire_owned().unwrap();
        assert!(
            service
                .sender
                .try_send(Job {
                    request: JobRequest::Count(Box::new(|| panic!(
                        "cancelled count must not prepare"
                    ))),
                    reply: JobReply::Single(reply),
                    _permit: permit,
                })
                .is_ok()
        );
        drop(service);
        worker.await.unwrap();
        assert_eq!(counts.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn native_count_rejects_images_before_template_or_tokenizer_work() {
        assert!(reject_count_images(&Prompt::Text("hello".into())).is_ok());
        assert!(reject_count_images(&Prompt::Chat(vec![message("user", "hello", vec![])])).is_ok());
        let result = reject_count_images(&Prompt::Chat(vec![message(
            "user",
            "hello",
            vec![vec![1, 2, 3]],
        )]));
        assert!(
            matches!(result, Err(InferenceError::Invalid(message)) if message.contains("text only"))
        );
    }

    #[test]
    #[ignore = "requires LMA_TEST_OLLAMA_MODEL naming an installed GGUF without an embedded template; loads real weights"]
    fn real_missing_template_model_counts_and_streams() {
        let name = std::env::var("LMA_TEST_OLLAMA_MODEL").expect("set LMA_TEST_OLLAMA_MODEL");
        let resolved =
            crate::models::resolve(std::path::Path::new(&format!("ollama:{name}")), None).unwrap();
        let config = ServeConfig {
            model: resolved.model,
            model_id: resolved.name,
            gpu_layers: 1,
            context: 256,
            batch: 32,
            micro_batch: 32,
            threads: 6,
            ..ServeConfig::default()
        };
        let mut engine = LlamaEngine::load(config).unwrap();
        assert!(
            engine
                .session
                .borrow_owner()
                .model
                .chat_template(None)
                .is_err(),
            "choose a model missing an embedded template to exercise the regression"
        );
        let prompt = || Prompt::Chat(vec![message("user", "Say hello.", vec![])]);
        let count = engine.count_prompt(prompt(), &|| false).unwrap();
        let mut streamed = String::new();
        let result = engine
            .generate_stream(
                GenerationRequest {
                    prompt: prompt(),
                    options: Options {
                        until_context: false,
                        max_tokens: 4,
                        temperature: 0.0,
                        top_p: 0.95,
                        seed: Some(42),
                    },
                },
                &|| false,
                &mut |delta| {
                    streamed.push_str(&delta);
                    Ok(())
                },
            )
            .unwrap();
        assert!(count > 0);
        assert_eq!(count, result.prompt_tokens);
        assert_eq!(streamed, result.text);
        assert!(!streamed.trim().is_empty());
        assert!((1..=4).contains(&result.completion_tokens));
        println!(
            "Real missing-template chat: count={count}, generated={}, text={streamed:?}",
            result.completion_tokens
        );
    }

    #[test]
    fn owned_batch_can_be_cleared_and_reused_without_a_model() {
        let mut batch: LlamaBatch<'static> = LlamaBatch::new(4, 1);
        for count in [4, 1, 3, 4] {
            batch.clear();
            assert_eq!(batch.n_tokens(), 0);
            for position in 0..count {
                batch
                    .add(LlamaToken(position), position, &[0], position + 1 == count)
                    .unwrap();
            }
            assert_eq!(batch.n_tokens(), count);
        }
        assert!(batch.add(LlamaToken(9), 4, &[0], true).is_err());
    }

    #[test]
    fn utf8_survives_split_tokens_and_flushes_incomplete_sequences() {
        let mut decoder = UTF_8.new_decoder_without_bom_handling();
        let mut text = String::new();
        for byte in "Hi 🦀 café".as_bytes() {
            append_utf8(&mut decoder, &[*byte], &mut text, false).unwrap();
        }
        append_utf8(&mut decoder, &[0xe2], &mut text, false).unwrap();
        append_utf8(&mut decoder, &[], &mut text, true).unwrap();
        assert_eq!(text, "Hi 🦀 café\u{fffd}");
    }

    #[test]
    fn context_budget_accounts_for_prompt_and_generation() {
        assert!(check_budget(10, 6, 16).is_ok());
        assert!(check_budget(10, 7, 16).is_err());
        assert!(check_budget(17, 1, 16).is_err());
        assert!(check_budget(0, 1, 16).is_err());
        assert!(check_budget(usize::MAX, u32::MAX, 16).is_err());
    }

    #[test]
    fn until_context_budget_uses_actual_prompt_and_native_image_positions() {
        let mut options = generation_request().options;
        options.until_context = true;
        assert_eq!(output_allowance(10, 10, &options, 16).unwrap(), 6);
        assert_eq!(output_allowance(10, 14, &options, 16).unwrap(), 2);
        assert_eq!(output_allowance(14, 10, &options, 16).unwrap(), 2);
        assert_eq!(output_allowance(15, 15, &options, 16).unwrap(), 1);
        assert_eq!(output_allowance(16, 16, &options, 16).unwrap(), 0);
        // The effective context is supplied after native initialization, not the
        // requested/configured size, and the valid placeholder is not a reply cap.
        options.max_tokens = 8;
        assert_eq!(output_allowance(6, 6, &options, 8).unwrap(), 2);
        for (tokens, positions) in [
            (0, 1),
            (1, 0),
            (17, 1),
            (1, 17),
            (usize::MAX, 1),
            (1, usize::MAX),
        ] {
            assert!(output_allowance(tokens, positions, &options, 16).is_err());
        }
        assert!(output_allowance(1, 1, &options, 0).is_err());
        assert_eq!(
            output_allowance(1, 1, &options, u32::MAX).unwrap(),
            u32::MAX - 1
        );
    }

    #[test]
    fn finite_budget_remains_strict_for_tokens_and_image_positions() {
        let mut options = generation_request().options;
        options.max_tokens = 6;
        assert_eq!(output_allowance(10, 10, &options, 16).unwrap(), 6);
        assert_eq!(output_allowance(2, 3, &options, 16).unwrap(), 6);
        for (tokens, positions) in [
            (11, 10),
            (10, 11),
            (16, 16),
            (0, 1),
            (1, 0),
            (usize::MAX, 1),
        ] {
            assert!(output_allowance(tokens, positions, &options, 16).is_err());
        }
    }

    #[test]
    fn until_context_never_bypasses_public_option_validation() {
        for until_context in [false, true] {
            let mut options = generation_request().options;
            options.until_context = until_context;
            for max_tokens in [0, 17, u32::MAX] {
                options.max_tokens = max_tokens;
                assert!(options.validate(16).is_err());
            }
            for max_tokens in [1, 16] {
                options.max_tokens = max_tokens;
                assert!(options.validate(16).is_ok());
            }
            options.temperature = f32::NAN;
            assert!(options.validate(16).is_err());
            options.temperature = 0.0;
            options.top_p = 0.0;
            assert!(options.validate(16).is_err());
        }
    }

    #[test]
    fn prefix_plan_reuses_exact_common_prefix_and_refreshes_logits() {
        assert_eq!(prefix_to_reuse(&[1, 2, 3], &[1, 2, 3], true), 2);
        assert_eq!(prefix_to_reuse(&[1, 2, 3], &[1, 2, 3, 4], true), 3);
        assert_eq!(prefix_to_reuse(&[1, 2, 3], &[1, 2, 9], true), 2);
        assert_eq!(prefix_to_reuse(&[1, 2, 3], &[1, 2], true), 1);
        assert_eq!(prefix_to_reuse(&[1, 2, 3], &[1, 2, 3], false), 0);
        assert_eq!(prefix_to_reuse(&[], &[1, 2], true), 0);
        assert_eq!(prefix_to_reuse(&[1], &[1], true), 0);
        assert_eq!(prefix_to_reuse(&[1], &[1, 2], true), 1);
        assert_eq!(prefix_to_reuse::<u8>(&[], &[], true), 0);
    }

    #[test]
    fn prefix_plan_changed_suffix_and_truncation_match_exact_tokens() {
        let cached = [LlamaToken(1), LlamaToken(2), LlamaToken(3), LlamaToken(4)];
        for (next, expected) in [
            (vec![1, 2, 9, 4], 2),
            (vec![1, 9, 3, 4], 1),
            (vec![9, 2, 3, 4], 0),
            (vec![1, 2, 3], 2),
            (vec![1, 2], 1),
            (vec![1], 0),
            (vec![], 0),
            (vec![1, 2, 3, 4], 3),
            (vec![1, 2, 3, 4, 5], 4),
        ] {
            let next: Vec<_> = next.into_iter().map(LlamaToken).collect();
            let keep = prefix_to_reuse(&cached, &next, true);
            assert_eq!(keep, expected, "next={next:?}");
            assert_eq!(&cached[..keep], &next[..keep]);
            assert!(keep < next.len() || next.is_empty());
            assert_eq!(prefix_to_reuse(&cached, &next, false), 0);
        }
    }

    #[test]
    fn failed_or_cancelled_requests_never_retain_reusable_state() {
        assert!(retain_prefix(true, true, false));
        assert!(!retain_prefix(true, false, false));
        assert!(!retain_prefix(true, true, true));
        assert!(!retain_prefix(true, false, true));
        assert!(!retain_prefix(false, true, false));
        let cancelled = std::cell::Cell::new(false);
        assert!(ensure_active(&|| cancelled.get()).is_ok());
        cancelled.set(true);
        assert!(ensure_active(&|| cancelled.get()).is_err());
        // Cancellation at the final boundary invalidates even a successful decode.
        assert!(!retain_prefix(true, true, cancelled.get()));
    }

    #[test]
    fn images_need_both_a_vision_build_and_a_projector() {
        assert!(validate_image_support(true, false, true).is_err());
        assert!(validate_image_support(true, true, false).is_err());
        assert!(validate_image_support(true, false, false).is_err());
        assert!(validate_image_support(true, true, true).is_ok());
        assert!(validate_image_support(false, false, false).is_ok());
        assert!(
            matches!(validate_image_support(true, false, true), Err(InferenceError::Invalid(message)) if message.contains("projector"))
        );
        assert!(
            matches!(validate_image_support(true, true, false), Err(InferenceError::Invalid(message)) if message.contains("vision"))
        );
    }

    fn message(role: &str, content: &str, images: Vec<Vec<u8>>) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            images,
        }
    }

    #[test]
    fn image_markers_stay_inside_their_messages_in_attachment_order() {
        let mut messages = vec![
            message("system", "Be helpful", vec![]),
            message("user", "Compare these", vec![vec![1], vec![2]]),
            message("assistant", "Different", vec![]),
            message("user", "And this?", vec![vec![3]]),
        ];
        let images = attach_images(&mut messages).unwrap();
        assert_eq!(images, vec![vec![1], vec![2], vec![3]]);
        assert_eq!(messages[0].content, "Be helpful");
        assert_eq!(
            messages[1].content,
            "<__lma_image_0__>\n<__lma_image_1__>\nCompare these"
        );
        assert_eq!(messages[2].content, "Different");
        assert_eq!(messages[3].content, "<__lma_image_2__>\nAnd this?");
        let formatted = messages
            .iter()
            .map(|message| format!("{}:{}\n", message.role, message.content))
            .collect::<String>();
        let restored = restore_image_markers(formatted, 3).unwrap();
        assert_eq!(restored.matches(media_marker()).count(), 3);
        assert!(!restored.contains(IMAGE_PLACEHOLDER_PREFIX));
    }

    #[test]
    fn marker_restoration_preserves_unicode_and_multi_digit_indices() {
        let mut prompt = String::from("🦀 header");
        let mut expected = prompt.clone();
        for index in 0..12 {
            prompt.push_str(&image_placeholder(index));
            prompt.push_str(" café\n");
            expected.push_str(media_marker());
            expected.push_str(" café\n");
        }
        assert_eq!(restore_image_markers(prompt, 12).unwrap(), expected);
        assert_eq!(
            restore_image_markers("plain 🦀".into(), 0).unwrap(),
            "plain 🦀"
        );
    }

    #[test]
    fn image_template_cannot_drop_duplicate_or_reorder_attachments() {
        for prompt in [
            "<__lma_image_0__>",
            "<__lma_image_0__><__lma_image_0__><__lma_image_1__>",
            "<__lma_image_1__><__lma_image_0__>",
            "<__lma_image_0__><__lma_image_1__><__lma_image_9__>",
            "<__lma_image_0__><__lma_image_1__><__lma_image_0__>",
            "<__lma_image_unknown__><__lma_image_0__><__lma_image_1__>",
        ] {
            assert!(restore_image_markers(prompt.into(), 2).is_err());
        }
        let injected = format!(
            "{}{}{}",
            image_placeholder(0),
            image_placeholder(1),
            media_marker()
        );
        assert!(restore_image_markers(injected, 2).is_err());
        assert!(attach_images(&mut [message("assistant", "No", vec![vec![1]])]).is_err());
        assert!(attach_images(&mut [message("user", media_marker(), vec![vec![1]])]).is_err());
    }

    #[test]
    fn expert_cache_model_parameters_preserve_dense_gpu_policy() {
        for budget_mib in [0, 1, 64, 1_048_576] {
            let mut config = ServeConfig {
                expert_cache_mib: budget_mib,
                gpu_layers: 42,
                ..ServeConfig::default()
            };
            config.apply_expert_cache_policy().unwrap();
            let params = expert_cache_model_policy(
                model_memory_policy(
                    LlamaModelParams::default()
                        .with_n_gpu_layers(config.gpu_layers)
                        .with_use_mmap(false)
                        .with_use_mlock(true),
                    LoadMode::Requested,
                ),
                &config,
            );
            assert_eq!(
                params.expert_cache_bytes(),
                u64::from(budget_mib) * 1_048_576
            );
            assert_eq!(params.n_gpu_layers(), 42);
            assert_eq!(params.use_mmap(), budget_mib > 0);
            assert_eq!(params.use_mlock(), budget_mib == 0);
            assert_eq!(config.micro_batch, 128);

            let cpu_params = model_memory_policy(params, LoadMode::CpuMmap);
            assert_eq!(
                cpu_params.expert_cache_bytes(),
                u64::from(budget_mib) * 1_048_576
            );
            assert_eq!(cpu_params.n_gpu_layers(), 0);
            assert!(cpu_params.use_mmap());
            assert!(!cpu_params.use_mlock());
        }
    }

    #[test]
    fn expert_cache_generation_context_preserves_batching_and_normal_backend_policy() {
        for gpu_layers in [0, 42] {
            let mut config = ServeConfig {
                expert_cache_mib: 64,
                gpu_layers,
                ..ServeConfig::default()
            };
            config.apply_expert_cache_policy().unwrap();
            for mode in [LoadMode::Requested, LoadMode::CpuMmap] {
                let params = context_memory_policy(
                    generation_context_params(&config, KvCacheType::F16),
                    mode,
                    config.gpu_layers,
                );
                assert_eq!(params.n_ubatch(), config.micro_batch);
                assert_eq!(params.n_batch(), config.batch);
                assert_eq!(
                    params.op_offload(),
                    mode == LoadMode::Requested
                        && gpu_layers > 0
                        && LlamaContextParams::default().op_offload()
                );
                if mode == LoadMode::Requested && gpu_layers > 0 {
                    assert_eq!(
                        params.offload_kqv(),
                        LlamaContextParams::default().offload_kqv()
                    );
                }
            }
        }
        let config = ServeConfig::default();
        let params = generation_context_params(&config, KvCacheType::F16);
        assert_eq!(params.n_ubatch(), config.micro_batch);
        assert_eq!(
            params.op_offload(),
            LlamaContextParams::default().op_offload()
        );
    }

    #[test]
    fn expert_cache_runtime_rejects_incompatible_options_before_validation_or_file_io() {
        let base = ServeConfig {
            expert_cache_mib: 64,
            // Policy errors must win even over ordinary runtime validation.
            context: 0,
            ..ServeConfig::default()
        };
        let configs = [
            ServeConfig {
                projector: Some("unused-projector.gguf".into()),
                ..base.clone()
            },
            ServeConfig {
                expert_cache_mib: 1_048_577,
                ..base
            },
        ];
        for config in configs {
            let expected = config.clone().apply_expert_cache_policy().unwrap_err();
            // Leave the model path empty: a file open or backend init is a bug.
            let error = LlamaEngine::load(config)
                .err()
                .expect("invalid cache policy");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn expert_cache_embedding_rejection_needs_no_model_or_second_context() {
        assert!(validate_embedding_config(&ServeConfig::default()).is_ok());
        let config = ServeConfig {
            expert_cache_mib: 1,
            ..ServeConfig::default()
        };
        match validate_embedding_config(&config) {
            Err(InferenceError::Invalid(message)) => {
                assert!(message.contains("embeddings"));
                assert!(message.contains("only one context"));
                assert!(message.contains("disable expert caching"));
            }
            other => panic!("expected an invalid embedding configuration, got {other:?}"),
        }
    }

    #[test]
    fn memory_fallback_policy_is_explicit() {
        assert_eq!(
            MemoryFallback::parse("error").unwrap(),
            MemoryFallback::Error
        );
        assert_eq!(MemoryFallback::parse("mmap").unwrap(), MemoryFallback::Mmap);
        for invalid in ["", "auto", "swap", "MMAP"] {
            assert!(MemoryFallback::parse(invalid).is_err());
        }
    }

    #[test]
    fn gpu_success_never_warns_or_tries_cpu_even_when_fallback_is_enabled() {
        for policy in [MemoryFallback::Error, MemoryFallback::Mmap] {
            let mut modes = Vec::new();
            let result = initialize_with_fallback(
                policy,
                42,
                |mode| {
                    modes.push(mode);
                    Ok("GPU session")
                },
                |_| panic!("successful GPU initialization must not warn"),
            );
            assert_eq!(result.unwrap(), "GPU session");
            assert_eq!(modes, [LoadMode::Requested]);
        }
    }

    #[test]
    fn mmap_retries_once_after_failed_attempt_state_has_dropped() {
        use std::cell::Cell;
        struct FailedAttempt<'a>(&'a Cell<bool>);
        impl Drop for FailedAttempt<'_> {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }
        let dropped = Cell::new(false);
        let mut modes = Vec::new();
        let mut warnings = Vec::new();
        let result = initialize_with_fallback(
            MemoryFallback::Mmap,
            42,
            |mode| {
                modes.push(mode);
                if mode == LoadMode::Requested {
                    let _state = FailedAttempt(&dropped);
                    return Err(LoadFailure::Initialization(
                        "native model/context/projector failure".into(),
                    ));
                }
                assert!(
                    dropped.get(),
                    "retry must not overlap failed attempt resources"
                );
                Ok("CPU session")
            },
            |warning| warnings.push(warning.to_owned()),
        );
        assert_eq!(result.unwrap(), "CPU session");
        assert_eq!(modes, [LoadMode::Requested, LoadMode::CpuMmap]);
        assert_eq!(warnings.len(), 1);
        let warning = &warnings[0];
        for expected in [
            "WARNING",
            "native model/context/projector failure",
            "ONCE",
            "WEIGHTS",
            "extremely slow",
            "does not guarantee",
            "OOM-killer",
            "native abort",
        ] {
            assert!(warning.contains(expected));
        }
        assert!(!warning.contains("out of memory"));
    }

    #[test]
    fn failed_retry_preserves_both_initialization_errors() {
        let mut attempts = 0;
        let result = initialize_with_fallback::<()>(
            MemoryFallback::Mmap,
            1,
            |mode| {
                attempts += 1;
                Err(LoadFailure::Initialization(match mode {
                    LoadMode::Requested => "first GPU error".into(),
                    LoadMode::CpuMmap => "second CPU error".into(),
                }))
            },
            |_| {},
        );
        let error = result.unwrap_err();
        assert_eq!(attempts, 2);
        assert!(error.contains("first GPU error"));
        assert!(error.contains("second CPU error"));
    }

    #[test]
    fn default_policy_and_backend_init_failures_never_retry() {
        for gpu_layers in [0, 42] {
            let mut attempts = 0;
            let error = initialize_with_fallback::<()>(
                MemoryFallback::Error,
                gpu_layers,
                |mode| {
                    assert_eq!(mode, LoadMode::Requested);
                    attempts += 1;
                    Err(LoadFailure::Initialization("original failure".into()))
                },
                |_| panic!("fallback was not enabled"),
            )
            .unwrap_err();
            assert_eq!(error, "original failure");
            assert_eq!(attempts, 1);
        }
        let mut attempts = 0;
        let error = initialize_with_fallback::<()>(
            MemoryFallback::Mmap,
            42,
            |_| {
                attempts += 1;
                Err(LoadFailure::Backend("backend already initialized".into()))
            },
            |_| panic!("backend singleton failures must not trigger fallback"),
        )
        .unwrap_err();
        assert_eq!(error, "backend already initialized");
        assert_eq!(attempts, 1);
    }

    #[test]
    fn explicit_cpu_mmap_warns_and_attempts_only_once() {
        let mut modes = Vec::new();
        let mut warnings = Vec::new();
        let error = initialize_with_fallback::<()>(
            MemoryFallback::Mmap,
            0,
            |mode| {
                modes.push(mode);
                Err(LoadFailure::Initialization(
                    "CPU initialization failure".into(),
                ))
            },
            |warning| warnings.push(warning.to_owned()),
        )
        .unwrap_err();
        assert_eq!(error, "CPU initialization failure");
        assert_eq!(modes, [LoadMode::CpuMmap]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("explicitly requested with gpu_layers=0"));
    }

    #[test]
    fn memory_parameter_overrides_apply_only_to_cpu_mmap() {
        let requested_model = model_memory_policy(
            LlamaModelParams::default()
                .with_n_gpu_layers(42)
                .with_use_mmap(false)
                .with_use_mlock(true),
            LoadMode::Requested,
        );
        assert_eq!(requested_model.n_gpu_layers(), 42);
        assert!(!requested_model.use_mmap());
        assert!(requested_model.use_mlock());
        let cpu_model = model_memory_policy(requested_model, LoadMode::CpuMmap);
        assert_eq!(cpu_model.n_gpu_layers(), 0);
        assert!(cpu_model.use_mmap());
        assert!(!cpu_model.use_mlock());
        let requested_context = context_memory_policy(
            LlamaContextParams::default()
                .with_offload_kqv(true)
                .with_op_offload(true),
            LoadMode::Requested,
            42,
        );
        assert!(requested_context.offload_kqv());
        assert!(requested_context.op_offload());
        let cpu_context = context_memory_policy(requested_context, LoadMode::CpuMmap, 42);
        assert!(!cpu_context.offload_kqv());
        assert!(!cpu_context.op_offload());
    }

    #[test]
    fn explicit_cpu_disables_context_offload_without_enabling_mmap_fallback() {
        let params = context_memory_policy(
            LlamaContextParams::default()
                .with_offload_kqv(true)
                .with_op_offload(true),
            LoadMode::Requested,
            0,
        );
        assert!(!params.offload_kqv());
        assert!(!params.op_offload());
        let model = model_memory_policy(
            LlamaModelParams::default()
                .with_n_gpu_layers(0)
                .with_use_mmap(false)
                .with_use_mlock(true),
            LoadMode::Requested,
        );
        assert_eq!(model.n_gpu_layers(), 0);
        assert!(!model.use_mmap());
        assert!(model.use_mlock());
    }

    #[test]
    fn kv_quantization_is_opt_in_and_requires_explicit_flash() {
        assert_eq!(kv_cache_type("f16", "auto").unwrap(), KvCacheType::F16);
        assert_eq!(kv_cache_type("f16", "off").unwrap(), KvCacheType::F16);
        assert_eq!(kv_cache_type("q8_0", "on").unwrap(), KvCacheType::Q8_0);
        assert_eq!(kv_cache_type("q4_0", "on").unwrap(), KvCacheType::Q4_0);
        assert!(kv_cache_type("q8_0", "auto").is_err());
        assert!(kv_cache_type("q4_0", "off").is_err());
        assert!(kv_cache_type("q3_k", "on").is_err());
    }

    #[test]
    fn rejects_nonfinite_sampling_parameters() {
        let mut options = Options {
            until_context: false,
            max_tokens: 1,
            temperature: f32::NAN,
            top_p: 0.95,
            seed: None,
        };
        assert!(options.validate(16).is_err());
        options.temperature = 0.0;
        options.top_p = f32::INFINITY;
        assert!(options.validate(16).is_err());
        options.top_p = 1.0;
        assert!(options.validate(16).is_ok());
    }
}
