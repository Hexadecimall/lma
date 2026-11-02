# OpenAI text compatibility implementation notes (0.3.4)

These notes audit the current text WIP across `src/runtime/server.rs`, `compat.rs`, `openai.rs`, `generation_controls.rs`, and `inference.rs`. The [public API contract](openai-api.md) describes supported requests. Files storage safety, mounting, and its dedicated documentation are owned separately; see [Files integration](#files-and-cloud-batch-boundary). This is an implementation audit, not a claim of full OpenAI compatibility, real-model validation, or performance parity.

## Native contracts used

- `Prompt::Tokens(Vec<i32>)` sends supplied token IDs directly to native inference; it does not turn the IDs into a textual prompt.
- `Generation` supplies visible text, native usage and finish reason, optional `Vec<TokenLogprob>`, and optional `mean_logprob`.
- `GenerationControls.logprobs`: `None` disables diagnostics, `Some(0)` requests chosen-token scores, `Some(k)` requests up to `k` alternatives.
- `generate_batch_prepared` / `stream_batch_prepared` admit a multi-candidate request as **one job with one worker permit**. Native execution is sequential and non-interleaved, not continuous GPU batching, parallel slots, or the cloud Batch API. Single-candidate nonstreaming uses `generate_prepared`; existing non-diagnostic single-choice streaming fast paths remain. Dropping the native receiver cancels the remaining candidates; cleanup still owns admission until it completes.
- Token-ID echo uses native `detokenize` utility jobs before generation. These do not sample candidates. All candidate generation is still admitted together in one batch.

## Requests and choices

Chat accepts integer `n:1..128`, boolean `logprobs`, and integer `top_logprobs:0..20`. A positive `top_logprobs` requires `logprobs:true`; omission/null/false disables public Chat diagnostics. Chat does not accept `best_of`. The existing single-choice, non-diagnostic Chat path is retained. Responses does not expose `n`, `best_of`, token-ID prompt arrays, or output logprobs.

Legacy Completions accepts a nonempty string, a nonempty array of nonempty strings, a nonempty nonnegative i32 token-ID array, or a nonempty array of nonempty token-ID arrays. At most 128 input prompts are accepted, subject to the tighter total-candidate bound below. Each token prompt is limited to 131072 IDs and checked against the native vocabulary/context during inference; supplied IDs bypass text formatting/retokenization. Mixed and empty prompt arrays are rejected. `n` and `best_of` are integers in `1..128`; omitted/null `n` defaults to one, omitted/null `best_of` defaults to `n`, and explicit `best_of<n` is rejected. Completions `logprobs` accepts integer `0..5` (zero still enables chosen-token scores); omission/null disables public diagnostics, and booleans are rejected.

Candidates are prepared in prompt-major order. Each prompt returns `n` choices, with flattened indexes `prompt_index * n + returned_candidate_index`. With `best_of>n`, candidates are stably ranked by descending **finite, nonpositive native mean sampled-token log probability**, including native EOG/stop sampling in that mean. Missing, nonfinite, or positive scores cause `invalid_native_output`, not a synthetic ranking. Equal scores preserve sampling order. With `best_of==n`, choices retain sampling order without ranking. Hidden candidates are still sampled and charged in usage. `best_of>n` cannot stream; `best_of==n` can.

When a seed is configured, sample `i` receives `seed.wrapping_add(i * 0x9e3779b9)` in wrapping u32 arithmetic. Sample zero retains the original seed. Unseeded requests stay unseeded. This makes the candidate seed mapping deterministic, not a promise of cross-hardware/model reproducibility.

Usage counts logical prompt tokens once per input prompt and completion tokens for **all actually sampled candidates**, including candidates hidden by `best_of`.

Structured JSON and validated tool plans support multiple Chat choices. Tool envelopes are decoded independently, with unique tool-call ID prefixes per choice, and never executed by the server. Existing schema validation, sampling, native admission, and tool validation remain in place.

## Diagnostics and precision

Chat returns `logprobs.content` entries with native `token`, `bytes`, `logprob`, and `top_logprobs` entries containing the same three fields. Legacy returns aligned `tokens`, `token_logprobs`, `top_logprobs` dictionaries, and `text_offset` arrays. Legacy offsets count Unicode scalar values in the visible output, not UTF-8 bytes or JavaScript UTF-16 units. If native pieces split one UTF-8 character, their offsets address that same containing character; Chat retains their exact raw byte vectors. Piece labels can therefore contain native lossy UTF-8 replacement characters even when their combined bytes form valid output.

Probabilities are natural-log numerical estimates normalized from native logits **after grammar/penalty/bias preprocessing and before temperature/top-p selection**, not fabricated values, prompt likelihoods, or a guarantee of matching an upstream sampling distribution. Collection adds native/CPU bookkeeping; no unchanged-throughput claim is made. Top lists may contain fewer entries than requested: EOG alternatives are filtered and native pieces may be unavailable. Legacy dictionaries cannot distinguish alternatives with identical token labels (including lossy labels); a colliding label retains the greatest actual native score. Chat's list and byte vectors preserve more information.

The native collector omits partially visible tokens at stop boundaries and records that cannot be aligned after lossy UTF-8 decoding. The adapters require supplied records to match a prefix of visible output exactly. They neither relabel partial tokens nor invent missing probabilities. Thus diagnostic tokens need not reconstruct all visible output, and their count can be smaller than usage's sampled completion-token count. Suppressed sampled stop/EOG records are not exposed as output tokens. Alternative candidates remain native vocabulary alternatives, not claims about emitted output.

`echo:true` returns the actual supplied text, or the native detokenization of supplied token IDs. **`echo:true` with public logprobs is rejected** because native prompt probabilities are unavailable. Internal scoring for `best_of` is still allowed with echo when public logprobs are omitted. Chat logprobs with constrained tool envelopes are explicitly rejected: envelope-token likelihoods are not assistant-content likelihoods. Structured JSON without tool envelopes can use logprobs.

## Streaming

Every choice shares one request ID and response creation timestamp. Choice indexes are stable; each choice in a successful stream receives one terminal finish chunk. Successful requests emit exactly one final empty-choices usage chunk with `stream_options.include_usage:true`, none otherwise, followed by exactly one `[DONE]`. Errors emit an error event without a success sentinel or fabricated final usage; unfinished choices need not receive a success finish chunk.

Native ordinary-text deltas are forwarded as they arrive. Diagnostic-enabled choices are buffered until their final native result, because the native event contract supplies diagnostic records only at completion. Each buffered choice emits its text and matching diagnostics together, then its finish chunk. Constrained tool choices are also buffered until validation succeeds. The `x-lma-streaming` header identifies `validated-choice-buffer` for buffered batch/diagnostic paths; the retained single-choice non-diagnostic tool path uses `validated-tool-buffer`, and unbuffered paths use `native`. Structured JSON without tools remains incremental unless diagnostics enable buffering. Disconnecting the HTTP body drops the receiver and cancels the native batch; no extra generation admissions are created for subsequent choices.

## Resource bounds

- At most 128 sampled candidates total: `prompt_count * best_of <= 128`.
- At most 262144 requested completion tokens across all sampled candidates.
- At most 1048576 requested diagnostic slots (chosen token plus requested alternatives); internal hidden-candidate scoring participates in preflight accounting.
- Candidate/batch paths cap aggregate cloned prepared prompt payloads at 64 MiB, including image bytes and schema text.
- Candidate/batch paths cap aggregate visible output plus a conservative diagnostic-payload estimate and echoed prompts at 64 MiB. A shared native budget reserves echoes and charges text before retention/emission and diagnostic records before retention. Exhaustion stops the remaining candidates in nonstreaming and streaming work, rather than rejecting only after the batch has accumulated. Hidden candidates and subsequently suppressed diagnostic pieces are not refunded. This is not exact serialized JSON size or total RSS: native sampling/vocabulary buffers and transient token pieces remain outside it. Existing single-choice non-diagnostic fast paths retain their own limits.
- Existing per-choice 256 KiB constrained-tool limits remain enforced.
- Trusted until-context generation remains single-candidate only.

Preflight work/output limits use `resource_limit_exceeded`. Prepared-prompt storage checks occur on the admitted worker and return a native invalid-request error with an explicit resource-limit message. Limits reject oversized work rather than silently reducing `n`, `best_of`, token limits, or diagnostics.

HTTP body limits remain 1 MiB for Completions/Responses and 24 MiB for Chat. Native vocabulary and per-candidate context validation still apply. Accepting token-ID arrays does not bypass these checks.

## Files and cloud Batch boundary

`src/runtime/openai_files.rs` implements an independent local Files router mounted by `hosting.rs` on the CLI managed host's `all`/`openai` profiles. Storage operations are authenticated and model-independent. Temporary-root tests cover recovery journals, conflicting metadata and rollback failure; host tests cover route/profile/auth gates without loading a model. See the [Files notes](openai-files-034.md) for platform, storage and durability limits.

Files upload/list/metadata/delete/content operations are storage, not inference or hosted processing. Accepting purposes such as `batch`, `fine-tune`, or `assistants` does not implement `/v1/batches`, fine-tuning, Assistants, or generation from file IDs. Request-level native batching above is synchronous/sequential, not an asynchronous cloud job service.

## Remaining unsupported surface

Generic suffix infill, prompt-token likelihoods, echo with public logprobs, streaming `best_of>n`, and Chat logprobs for constrained tool envelopes remain rejected. Responses remains stateless and single-output without these new diagnostic fields. Response/session persistence/retrieval, background generation jobs, Assistants/vector stores, hosted tools, fine-tuning/account services, Responses WebSockets, OpenAI audio/video/image-generation APIs, compressed request bodies, and arbitrary unknown fields are not implemented. Local Files storage does not close those gaps. There is no universal SDK/client certification or speed-parity claim.

## Focused test locations and validation status

- `openai.rs` unit tests cover strict input shapes/bounds, candidate seeds, stable native-score ranking, missing-score errors, hidden-candidate usage, wire diagnostics, and Unicode offsets.
- `server.rs` HTTP fixtures cover token-ID echo, multiple Completions/Chat choices, actual supplied diagnostic records, SSE identity/index/finish/usage behavior, independently validated tool calls with unique IDs, and structured JSON choices.
- `inference.rs` tests cover token-ID validation, numerical logprob collection, sequential atomic batch jobs, bounded backpressure, errors, cancellation, and worker-owned cleanup/admission.

Integration validation runs the no-default-features library suite, the Metal-enabled library/CLI/desktop-state suites, and a locked Metal release CLI build. `python3 tests/openai_files_http.py target/release/lma` exercises authenticated Files upload/list/metadata/content/HEAD, restart persistence, and deletion through isolated child HTTP servers and a temporary HOME, asserting no model is loaded. Storage regressions also cover parser framing bounds, stalled-upload timeouts and atomic no-replace publication. These checks do not download/load a real model or certify real-model output quality, installed-client interoperability, Windows/FreeBSD Files storage, or performance parity.
