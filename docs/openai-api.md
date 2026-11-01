# OpenAI-compatible local text API

Enable `lma s -a openai` or the default `all` profile. The base URL is
`http://127.0.0.1:8080/v1`; supply the configured Bearer key, or a nonsecret dummy
key only for SDKs that require one when server authentication is disabled.
Installed model IDs come from `GET /v1/models`. No model download is implicit.

This is a **local inference API, not all OpenAI cloud services or every parameter
combination**. Unsupported functionality returns an error rather than a simulated
success. The supported surface below replaces the old blanket rejection of stop
sequences and sampling controls. It does not certify every Zed/OpenAI SDK version.

## Text generation capabilities

| Feature | `/completions` | `/chat/completions` | `/responses` |
| --- | --- | --- | --- |
| Native incremental SSE | Yes | Yes | Yes, Responses event subset |
| Nonstreaming output and actual token usage | Yes | Yes | Yes |
| Temperature, top-p, seed | Yes | Yes | Temperature/top-p; existing subset |
| Stop strings | Yes | Yes | Not a Responses request field |
| Frequency/presence penalties and logit bias | Yes | Yes | Not exposed |
| Function declarations/calls/results | No | Existing client-executed subset | Existing client-executed subset |
| JSON object and bounded JSON-schema output | No | `response_format` | `text.format` |
| Prompt echo | `echo:true` | No | No |
| Optional SSE usage record | `stream_options.include_usage` | Same | Existing Responses usage events |
| Images | No | Existing PNG/JPEG data-URL/projector support | Existing `input_image` subset |

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

Raw `prompt` is not chat-formatted or stripped of edit/FIM markers. A string or
one-element string array is accepted. `n:1`, `best_of:1`, `logprobs:null`, optional
bounded `user`, `echo`, and `suffix:null`/`""` are accepted. There is no generic
suffix infill implementation: provide the model-specific full FIM prompt instead.
Token-ID prompts, multiple prompts/candidates, `n>1`, `best_of>1`, and non-null
Completions `logprobs` (including `0`) are explicitly unsupported, not ignored.

`echo:true` prepends the exact prompt to output, without counting it as generated
tokens. Streaming emits the echo before generated deltas. A successful stream
has a finish chunk, an optional separate `choices:[]` usage chunk when requested,
and `data: [DONE]`. Without `include_usage`, usage is not fabricated in SSE chunks.
Nonstreaming results always contain usage. Errors after SSE has begun emit an
error event, not a fabricated success or `[DONE]`.

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
`n:1`, `logprobs:false`/null, `top_logprobs:0`/null, `modalities:["text"]`, and
`audio:null`. These neutral values do not assert support for probabilities/audio.
Existing metadata, store:false, tools, history, and advisory reasoning-field
rules remain documented in [client setup](clients.md).

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
is incremental; clients must inspect the final finish/error, not assume every
intermediate prefix is a complete document.

Combining a structured response format with function-tool generation in one
request is not yet supported. Tool arguments retain their separate constrained,
validated/buffered envelope and are not executed by lma.

## Cancellation and admission

One native operation runs at a time. Connected overlap still receives HTTP 429
with `Retry-After: 1`. After a client disconnects, **one** successor may wait up to
**1.5 seconds** for the old worker's native operation and discard cleanup to exit.
It is not an unbounded request queue or permission to run two operations together.
A dropped waiting request never prepares/enqueues model work. Cleanup beyond the
bound, an additional waiter, or ordinary concurrent generation can still return
429. Model switching/unloading while active response leases exist remains 409.

This fixes the brief post-cancellation busy window server-side; clients should
still handle standard overload responses rather than retry indefinitely.

## Explicit remaining limits

Not implemented: token log probabilities/top-logprobs, multiple candidate or
batched-prompt generation, generic suffix infill, server-side response/session
persistence and retrieval, background jobs, hosted web search/code interpreter,
OpenAI file/vector-store/batch/fine-tuning/account APIs, Responses WebSockets,
OpenAI audio/video/image-generation APIs, compressed request bodies, or arbitrary
unknown provider fields. Local image/speech features are separate APIs/workflows,
not proof of compatibility with those cloud contracts.
