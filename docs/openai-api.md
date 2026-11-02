# OpenAI-compatible local text API

Enable `lma s -a openai` or the default `all` profile. The base URL is
`http://127.0.0.1:8080/v1`; supply the configured Bearer key, or a nonsecret dummy
key only for SDKs that require one when server authentication is disabled.
Installed model IDs come from `GET /v1/models`. No model download is implicit.

This is a **local inference API, not all OpenAI cloud services or every parameter
combination**. Unsupported functionality returns an error rather than a simulated
success. Generation uses the in-process native engine, including multiple choices,
token-ID prompts, and output-token diagnostics. This does not certify every
Zed/OpenAI SDK version or imply cloud-service or performance parity.

## Text generation capabilities

| Feature                                     | `/completions`                                                | `/chat/completions`                           | `/responses`                        |
| ------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- | ----------------------------------- |
| Native incremental SSE                      | Yes; diagnostics buffer each choice                           | Yes; diagnostics/tools can buffer each choice | Yes, Responses event subset         |
| Nonstreaming output and actual token usage  | Yes, including hidden sampled candidates                      | Yes, across all choices                       | Yes                                 |
| Multiple returned choices (`n`)             | `1..128` per prompt, subject to aggregate limits              | `1..128`, subject to aggregate limits         | Not exposed                         |
| Best-of candidate selection                 | `best_of:1..128`, at least `n`; ranking requires nonstreaming | No                                            | No                                  |
| Batched prompts / exact token-ID prompts    | String arrays, token-ID arrays, arrays of token-ID arrays     | No; one message history per request           | No equivalent text-generation batch |
| Native output logprobs                      | Integer `0..5`                                                | `logprobs:true`, `top_logprobs:0..20`         | Not exposed                         |
| Temperature, top-p, seed                    | Yes                                                           | Yes                                           | Temperature/top-p; existing subset  |
| Stop strings                                | Yes                                                           | Yes                                           | Not a Responses request field       |
| Frequency/presence penalties and logit bias | Yes                                                           | Yes                                           | Not exposed                         |
| Function declarations/calls/results         | No                                                            | Existing client-executed subset               | Existing client-executed subset     |
| JSON object and bounded JSON-schema output  | No                                                            | `response_format`                             | `text.format`                       |
| Prompt echo                                 | `echo:true`                                                   | No                                            | No                                  |
| Optional SSE usage record                   | `stream_options.include_usage`                                | Same                                          | Existing Responses usage events     |
| Images                                      | No                                                            | Existing PNG/JPEG data-URL/projector support  | Existing `input_image` subset       |

### Completions and Zed edit prompts

```json
{
  "model": "INSTALLED_MODEL_ID",
  "prompt": "The exact edit/FIM prompt, including its native markers",
  "max_tokens": 128,
  "temperature": 0,
  "stop": ["<|endoftext|>"],
  "stream": true,
  "stream_options": { "include_usage": true }
}
```

Raw text `prompt` is not chat-formatted or stripped of edit/FIM markers. Accepted
shapes are:

- A nonempty string, or an array of nonempty strings.
- A nonempty array of nonnegative i32 token IDs, or an array of such arrays.
  IDs go directly to native inference, without detokenizing and retokenizing them.
  Each token prompt is limited to 131072 IDs; IDs must belong to the loaded model's
  vocabulary, and each prompt plus its requested output must fit the context.

Empty or mixed-type prompt arrays are rejected. There are at most 128 input
prompts, and the candidate limit below can reduce that number. Optional bounded
`user`, `echo`, and `suffix:null`/`""` are accepted. There is no generic suffix
infill implementation: provide the model-specific full FIM prompt instead.

`echo:true` prepends the supplied text, or native detokenization of supplied token
IDs, to each returned choice, without counting that echo as generated tokens.
Streaming emits it before generated deltas. **Echo with public `logprobs` is
rejected, including `logprobs:0`: prompt-token probabilities are unavailable.**
Echo can still be combined with internal `best_of` scoring when public logprobs
are omitted/null.

### Multiple choices, best-of, and request batching

Both Completions and Chat accept integer `n:1..128`; omission/null defaults to one.
Completions also accepts integer `best_of:1..128`, defaulting to `n` on omission/null.
Explicit `best_of<n` is rejected. Chat does not accept `best_of`.

For each Completions prompt, the engine samples `best_of` candidates and returns
`n` choices. When `best_of>n`, it ranks candidates by descending native mean
sampled-token log probability, including sampled EOG/suppressed stop tokens.
Scores must be finite and nonpositive; missing or invalid scores are errors,
not synthetic rankings. Ties preserve sampling order. With `best_of==n`, all
candidates are returned in sampling order. **`best_of>n` requires `stream:false`;**
`best_of==n` can stream.

Prompts and candidates run in prompt-major order. Returned Completions indexes
are flattened as `prompt_index * n + returned_candidate_index`; Chat indexes are
`0..n-1`. Usage counts prompt tokens once per logical input prompt and completion
tokens for **all sampled candidates**, including those hidden by `best_of`.

If a seed is configured, global sample index `i` uses
`seed.wrapping_add(i * 0x9e3779b9)` in wrapping u32 arithmetic. Sample zero keeps
the configured seed; an unseeded request remains unseeded. This is a deterministic
seed mapping, not a cross-model/hardware reproducibility guarantee or a promise
that greedy candidates differ.

All sampled candidates share one native worker admission and execute sequentially,
without interleaving other jobs. This is **request-level batching, not continuous
GPU batching, parallel inference slots, or the asynchronous OpenAI `/v1/batches`
API**. See [resource bounds](#batch-and-diagnostic-resource-bounds).

### Stop and sampling semantics

- `stop` accepts null/omission, one string, or an array of up to four strings.
  Empty arrays disable it; each sequence must contain 1–4096 UTF-8 bytes and no NUL.
- Matching happens during native generation, not by truncating a completed reply.
  The first match ends generation with `finish_reason:"stop"`; neither the stop
  text nor subsequent text is returned. Partial prefixes are withheld across
  native/UTF-8 chunk boundaries and flushed if unmatched at EOS/output exhaustion.
- Usage still counts generated tokens, including the token that triggered a stop.
  Native model EOG tokens also stop normally; an end-token spelling is not a shim
  that is silently discarded.
- `frequency_penalty` and `presence_penalty` are finite values in `[-2,2]`, applied
  to **generated-token history**, not the prompt. Null/omission means zero.
- `logit_bias` accepts a null/omitted map or up to 1024 entries. Keys are canonical
  nonnegative decimal token IDs from the **loaded model's** vocabulary; values
  are finite `[-100,100]`. IDs are validated before generation-state mutation.
- Real native samplers apply these controls, including temperature-zero requests.
  Default controls retain the existing greedy shortcut. No GPU/CPU fallback or
  alternate inference process is introduced.

Chat additionally accepts `max_completion_tokens` instead of `max_tokens`,
`modalities:["text"]`, and `audio:null`; audio output is not implemented.
Existing metadata, store:false, tools, history, and advisory reasoning-field
rules remain documented in [client setup](clients.md).

## Output-token log probabilities

Completions accepts integer `logprobs:0..5`: `0` requests chosen-token scores
without alternatives; omission/null disables public diagnostics. Chat accepts
boolean `logprobs` (omission/null/false disables it), with integer
`top_logprobs:0..20` (omission/null defaults to zero). A positive `top_logprobs`
requires `logprobs:true`. Responses does not expose these fields.

- Chat returns `choices[].logprobs.content` entries with `token`, raw `bytes`,
  `logprob`, and `top_logprobs` entries with the same three fields.
- Completions returns aligned `tokens`, `token_logprobs`, `top_logprobs` maps,
  and `text_offset` arrays. Offsets count Unicode scalar values in visible output,
  not UTF-8 bytes or UTF-16 code units. Split UTF-8 pieces can share a character
  offset; token labels may contain lossy replacement characters even when their
  combined bytes form valid text.
- Scores are natural-log probabilities normalized from native logits **after
  grammar/penalty/bias processing but before temperature/top-p selection**. They
  are numerical estimates, not fabricated values, prompt likelihoods, or an exact
  upstream sampling-distribution contract. Enabling diagnostics adds native/CPU
  bookkeeping; no equal-throughput claim is made.
- Top lists can be shorter than requested: EOG alternatives are omitted and native
  pieces may be unavailable. Legacy maps collapse identical token labels to the
  greatest actual native score; Chat lists/raw bytes preserve more information.
- Only complete native token pieces aligned to a prefix of visible output are
  reported. Partial stop-boundary tokens, suppressed stop/EOG output, and records
  that cannot align after lossy UTF-8 decoding are omitted, not relabeled.
  Diagnostic tokens therefore need not reconstruct all visible output and can be
  fewer than usage's sampled completion tokens.

Chat logprobs cannot be combined with constrained function-tool envelopes:
envelope-token likelihoods are not assistant-content likelihoods. Structured JSON
without a tool envelope can use diagnostics. Neither endpoint provides prompt-token
probabilities; Completions `echo:true` with public logprobs is rejected.

## Streaming choices and diagnostics

Every choice shares one request ID and creation timestamp and has a stable index.
Successful Completions/Chat streams send one terminal finish chunk per choice,
then one separate `choices:[]` usage chunk only if
`stream_options.include_usage:true`, followed by exactly one `data: [DONE]`.
Nonstreaming results contain usage. Errors after SSE begins emit an error event,
without a fabricated final usage record or success `[DONE]`; unfinished choices
need not receive a success finish chunk.

Ordinary text deltas are forwarded as native generation produces them.
**Diagnostic-enabled choices are buffered until each candidate finishes**: text
and its diagnostics arrive together, followed by that choice's finish chunk.
Constrained tool choices are likewise withheld until validated. Multi-choice or
diagnostic buffered paths use `x-lma-streaming: validated-choice-buffer`; the
retained single-choice non-diagnostic tool path uses `validated-tool-buffer`.
Unbuffered paths use `native`. Structured JSON without tools streams incrementally
unless diagnostics request per-choice buffering.

Disconnecting the HTTP body cancels remaining candidates; it does not submit
separate follow-up generation jobs. Worker cleanup still owns admission.

## Batch and diagnostic resource bounds

Limits reject oversized work rather than silently reducing candidates or budgets:

- At most **128 sampled candidates total**: `prompt_count * best_of <= 128`
  for Completions, or `n <= 128` for Chat.
- At most **262144 requested completion tokens** across all sampled candidates.
- At most **1048576 diagnostic slots**, counting one chosen-token score plus
  requested alternatives per requested output token. Hidden `best_of` scoring
  also participates in preflight accounting.
- Candidate/batch paths cap aggregate cloned prepared prompt payloads at **64 MiB**
  (including images/schema text), and aggregate output/echo plus a conservative
  diagnostic-payload estimate at **64 MiB**. One shared native byte budget reserves
  echoes, charges text before retention/emission, and charges diagnostic records
  before retention; exhaustion stops the remaining batch, including nonstreaming
  work. Charges for hidden candidates and suppressed diagnostic pieces are not
  refunded. This is not an exact JSON wire-size or process-RSS guarantee; native
  sampling/vocabulary buffers and transient token pieces are outside it. Existing
  single-choice non-diagnostic fast paths retain their own limits.
- Constrained tool output remains limited to **256 KiB per choice**.
- Trusted internal until-context generation is single-candidate only; it is not a
  wire option to bypass these bounds.

Preflight work/output-limit errors use `resource_limit_exceeded`. Prepared-prompt
storage checks run on the admitted worker and return an invalid-request error
with an explicit resource-limit message. Existing HTTP body limits still apply:
1 MiB for Completions/Responses and 24 MiB for Chat; accepting token IDs or prompt
arrays does not bypass body, vocabulary, or context validation.

## Structured JSON output

Chat accepts `response_format:{"type":"json_object"}` or:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "EditResult",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": { "ok": { "const": true } },
      "required": ["ok"],
      "additionalProperties": false
    }
  }
}
```

Responses uses the same schema with its flattened shape:
`text:{"format":{"type":"json_schema","name":"EditResult","strict":true,"schema":...}}`.
`type:"text"` selects ordinary text. Schemas use the existing bounded local tool
schema subset: depth 8, 256 nodes, at most 64 properties per object; unsupported
keywords such as external references are rejected. Native grammar constrains
sampling, and complete non-length-limited output is also checked against the
schema. A violation is an error, never claimed structured success. Output limits
can legitimately return partial JSON with a length finish reason. Streaming JSON
is incremental unless diagnostics enable per-choice buffering; clients must
inspect the final finish/error, not assume every intermediate prefix is complete.
Multiple Chat choices retain the same native schema constraints independently.

Combining a structured response format with function-tool generation in one
request is not yet supported. Tool arguments retain their separate constrained,
validated/buffered envelope and are not executed by lma.

## Cancellation and admission

One native operation runs at a time. Connected overlap still receives HTTP 429
with `Retry-After: 1`. After a client disconnects, **one** successor may wait up to
**1.5 seconds** for the old worker's native operation and discard cleanup to exit.
It is not an unbounded request queue or permission to run two operations together.
A dropped waiting request never prepares/enqueues model work. Cleanup beyond the
bound, an additional waiter, or ordinary concurrent generation can still return 429. Model switching/unloading while active response leases exist remains 409.

This fixes the brief post-cancellation busy window server-side; clients should
still handle standard overload responses rather than retry indefinitely.

## Local Files integration

The CLI managed host mounts local Files routes for its `all`/`openai` profiles:
multipart upload, list, metadata retrieval, deletion, and binary content retrieval
under `/v1/files`. No model is required. Data stays in `~/.lma/openai/files`;
non-loopback binds require an administrator API key. The initial secure storage
implementation requires Linux/macOS/Android and atomic no-replace rename support;
Windows and FreeBSD Files storage are not implemented. Embedded
in-process desktop/TUI routers do not expose Files. See the
[Files notes](openai-files-034.md) for quotas, recovery and durability boundaries.

Files are local single-administrator storage, not hosted processing. Accepting an
upload purpose such as `batch`, `fine-tune`, or `assistants` does not implement
those services or make file IDs usable as text/vision generation inputs. In
particular, **`/v1/batches` remains unsupported**. Files integration does not add
response/session persistence or an asynchronous generation queue.

## Explicit remaining limits

Not implemented: generic suffix infill, prompt-token likelihoods, streaming
`best_of>n`, echo with public logprobs, Chat logprobs for constrained tool envelopes,
Responses multiple choices/output logprobs, server-side response/session
persistence and retrieval, background jobs, hosted web search/code interpreter,
Assistants/vector-store/Batch/fine-tuning/account APIs, Responses WebSockets,
OpenAI audio/video/image-generation APIs, compressed request bodies, or arbitrary
unknown provider fields. Request-level batching remains sequential, not parallel
slots or continuous GPU batching. Local image/speech features and local Files
storage are separate workflows, not proof of compatibility with those cloud
contracts. See [implementation notes](openai-text-034.md) for code/test locations;
this document is not an end-to-end client or real-model validation report.
