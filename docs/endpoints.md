# HTTP endpoint profiles

Profiles select HTTP contracts around the **same in-process inference engine**. They do not launch `llama-server`, an Ollama daemon, or cloud services, and make no performance claim. These are supported subsets, **not full llama-server, cloud-provider, or agent parity**. Selecting a profile does not add model capabilities.

## Selecting a profile

```sh
# Canonical root syntax: short flags only.
lma s                     # Idle, all API families; no model loaded
lma s -a ollama            # Idle Ollama-compatible host
lma s MODEL -a llama
lma s MODEL -a lma
lma s MODEL -a all

# Long options belong only in nss.
lma nss serve MODEL --api llama
lma nss serve MODEL --api lma
```

Accepted values are `all|openai|ollama|llama|lma|anthropic`. `all` (the default) enables every family below, including existing Anthropic endpoints. No new root long aliases are introduced; `lma s MODEL --api llama` is not root syntax.

| Profile     | Base URL at the default address         | Family routes                                                                |
| ----------- | --------------------------------------- | ---------------------------------------------------------------------------- |
| `openai`    | `http://127.0.0.1:8080/v1`              | `/v1/completions`, `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings` |
| `ollama`    | `http://127.0.0.1:8080`                 | Ollama `/api/*` discovery, generation, embeddings, and administration routes |
| `llama`     | `http://127.0.0.1:8080` (**not** `/v1`) | `GET /props`, `POST /completion`, `POST /tokenize`, `POST /detokenize`       |
| `lma`       | `http://127.0.0.1:8080/lma/v1`          | `GET /lma/v1/status`, `POST /lma/v1/generate`, `POST /lma/v1/chat`           |
| `anthropic` | `http://127.0.0.1:8080`                 | `/v1/messages`, `/v1/messages/count_tokens`                                  |
| `all`       | Use the family-specific base URLs above | Every listed family                                                          |

A startup model is optional. `lma s` starts idle, without inference initialization,
a TTY, downloads, or arbitrary model selection. `/api/tags` works immediately with
`all`/`ollama`; `/api/ps` is empty until a model is loaded. The first generation
request must select an installed model. Explicit startup `MODEL` still preloads.

`GET /health` and `GET /v1/models` are shared across profiles. `POST /api/stop` is shared host control, **not** an Ollama-only route; `lma sp` / `lma nss stop` use it to unload a managed host's model. Browser chat at `/` is available only with `all` or `openai`. Other profiles return API information JSON there instead of a chat UI.

`-ct PRESET` (NSS `--client PRESET`) prints connection hints only. Without an explicit profile, `claude-code` selects `anthropic`; `codex`, `zed`, `vscode` (Continue), and `opencode` select `openai`; `default` and `custom` select `all`. Explicit `-a` / NSS `--api` wins in either argument order. If it disables a preset's required family, lma prints a warning and **no usable client preset**. llama/native-lma base URLs are endpoint hints, not claims that OpenAI or Anthropic client presets support those protocols. Nothing is installed, executed, written, or exported; keys are never printed. See [client setup and compatibility limits](clients.md).

## Shared safety and lifecycle

- With root `-ay KEY` (NSS `--api-key KEY`), send `Authorization: Bearer KEY` on **every protected API route**, including shared discovery/control, llama routes, and native lma routes. Only the existing Anthropic Messages/count routes additionally accept `x-api-key`. `/health` stays public. The public UI/information page is not a substitute for API authentication. Keep loopback binding unless external TLS/firewall protection is configured.
- New llama/native routes have a **1 MiB** request-body limit except native `/lma/v1/chat`, which allows **24 MiB** for image-bearing chat. Existing OpenAI/Ollama body limits and contracts are unchanged; see the [README](../README.md#http-api) and [Ollama contract](ollama-api.md).
- HTTP generation can select **installed models only**. Omission of `model` uses the current model; while idle, generation without a model returns HTTP 400. Unknown model names and filesystem paths do not trigger implicit downloads or arbitrary file loads. An explicitly selected startup GGUF is still served under its model ID. Explicit downloads remain separate operations (`lma f` or enabled Ollama `/api/pull`).
- The host keeps one resident model. Active requests prevent switching/unloading; the **streaming host lease persists until the response body is done**, not merely until headers are sent or a handler returns. Do not unload a model underneath an active stream. One model operation runs at a time; overlapping operations receive HTTP 429, and unsafe host switching/stopping receives HTTP 409.
- All protocols use real model tokenization and inference. Prompt/chat/image formatting consumes context. Send full desired chat history each time; there is no server-managed conversation history. Model/template/vision compatibility remains necessary.

## llama profile: strict llama-server subset

`GET /props` exposes served-model/context properties. It is not the full upstream server configuration or slot-management API.

### Raw completion

`POST /completion` accepts only:

| Field         | Contract                                                                                                                                                                                                             |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`      | Required nonempty string; raw prompt, not chat messages.                                                                                                                                                             |
| `model`       | Optional installed-model selector; omission uses the current model.                                                                                                                                                  |
| `n_predict`   | Positive integer or `-1`. Omission inherits the selected definition's stored `num_predict`, otherwise defaults to 256. `-1` uses remaining context without a reply cap. Zero and other negative values are rejected. |
| `temperature` | Optional sampling temperature, 0–2; zero is greedy. `temp` is its alias, not a second setting.                                                                                                                       |
| `top_p`       | Optional, greater than 0 and at most 1.                                                                                                                                                                              |
| `seed`        | Optional unsigned 32-bit integer.                                                                                                                                                                                    |
| `stream`      | Boolean, default `false`; `true` enables SSE on `/completion`.                                                                                                                                                       |

Sampling defaults inherit the selected definition where supplied; otherwise temperature is 0.8 and top-p is 0.95. Request values override stored defaults. A finite requested output budget must fit alongside the prompt within context.

```sh
curl http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","n_predict":64,"stream":false}'

curl -N http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","n_predict":64,"stream":true}'
```

This is **not full llama-server compatibility**: no slots/slot selection, FIM/infill, `cache_prompt`, stop sequences/stop controls, or arbitrary upstream flags/fields. Unsupported fields are strictly rejected, not ignored. The absence of request-level `cache_prompt` does not change the host's conservative prefix-reuse policy. Existing OpenAI SSE and Ollama NDJSON contracts are unchanged.

Nonstream completion returns `content`, `stop:true`, `tokens_predicted`,
`tokens_evaluated`, and `stop_type` (`eos` or `limit`). SSE sends
`data: {"content":"...","stop":false}` deltas followed by a final event with
`content:""`, `stop:true`, usage counts and `stop_type`. There is no OpenAI
`[DONE]` sentinel. An `event:error` terminates a failed stream without a fabricated
successful stop. No phase timings or slot IDs are invented.

### Real token IDs and vocabulary

`POST /tokenize` takes required `content:string` and optional boolean `add_special` and `parse_special` (both default to false). `true` requests native special-token insertion or recognition, respectively. The result uses **actual served-model token IDs**, not character codes, byte offsets, or estimates.

```sh
curl http://127.0.0.1:8080/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"content":"Hello!","add_special":false,"parse_special":false}'
```

`POST /detokenize` takes `{"tokens":[i32]}`: an array of signed 32-bit token IDs interpreted using the **actual model vocabulary**. IDs must be valid for that vocabulary. Tokenizer operations are limited to 128k
IDs and 1 MiB of input/output text, with bounded native buffers before allocation
of expanded output. Special tokens are rendered explicitly when detokenizing;
invalid UTF-8 token sequences use replacement characters. Both utilities use the
current model, do not accept `model` in their bodies, and do not mutate generation
history/KV state. `/props` and native status are authenticated, read-only snapshots;
when unloaded they return HTTP 503 rather than reloading the model. Use IDs returned by `/tokenize` for the same model, not IDs from a different tokenizer. Unsupported fields are rejected on both routes.

## Native lma v1 profile

`GET /lma/v1/status` provides native API status. The native contract is versioned with `api_version:1`; it is not an OpenAI, Ollama, or Anthropic response wrapper.

Status reports the configured context size and effective sampling defaults,
not an estimate of free GPU memory or a guarantee about actual tensor placement.

### Generation and chat requests

`POST /lma/v1/generate` takes required `prompt:string`, kept raw. `POST /lma/v1/chat` instead takes required `messages`, using the existing OpenAI role/content/images schema: `system`, `user`, and `assistant` text messages; user images may be raw-base64 `images` attachments or ordered image/text content parts. No tools, tool calls/tool-result history, or remote image fetching are supported. Native chat does not inherit the OpenAI tools surface merely by sharing its message/image representation.

Both accept the same optional generation fields:

| Field         | Contract                                                                                                                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model`       | String selecting an installed model; omission uses the current model. No filesystem paths or implicit downloads.                                                                                                         |
| `max_tokens`  | Positive integer **or explicit `null`**. Omission inherits stored `num_predict`, otherwise defaults to 256. **`null` means remaining context/no reply cap**, not the default 256. Zero and negative values are rejected. |
| `temperature` | 0–2; zero is greedy. Inherits stored sampling defaults, otherwise 0.8.                                                                                                                                                   |
| `top_p`       | Greater than 0 and at most 1; inherits stored defaults, otherwise 0.95.                                                                                                                                                  |
| `seed`        | Unsigned 32-bit integer.                                                                                                                                                                                                 |
| `stream`      | Boolean, default `false`; `true` enables NDJSON.                                                                                                                                                                         |

Unsupported fields are rejected. Syntax, intrinsic sampling controls and
nullable-field rules are checked before managed model selection; `name` is not an
alternate selector in these new protocols. Context/vocabulary checks still use
the selected model. Unlike `max_tokens`, `null` is not a general-purpose substitute
for omitting other options. A finite reply cap must fit with the formatted input in context. Remaining-context generation is still bounded by context and end-of-sequence.

```sh
curl http://127.0.0.1:8080/lma/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":64}'

# Explicitly use remaining context, rather than the stored/default reply cap.
curl http://127.0.0.1:8080/lma/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":null,"stream":false}'
```

For images, use a vision-enabled build and compatible model/projector. Only user PNG/JPEG images are supported; HTTP(S), remote, and file URLs are never fetched. Images must precede nonempty text in content parts. Existing limits apply: 8 images/request, 8 MiB decoded/image, 16 MiB decoded total; native chat's request body is limited to 24 MiB. Raw native generation is text-only.

### Nonstream response

A successful nonstream response has exactly this native envelope (placeholder ID/model/text and illustrative counts):

```json
{
  "api_version": 1,
  "id": "<request-id>",
  "model": "<served-model-id>",
  "text": "Hello!",
  "usage": { "input_tokens": 12, "output_tokens": 3 },
  "finish_reason": "eos"
}
```

`usage` reports actual input/output model tokens. `finish_reason` is one of:

- `eos`: generation reached end-of-sequence.
- `token_limit`: generation reached its finite reply cap.
- `context_limit`: generation reached the available context limit.

### Native streaming

With `stream:true`, `/lma/v1/generate` and `/lma/v1/chat` return newline-delimited JSON (`application/x-ndjson`), **not SSE**. Records are distinguished by `type`:

| Type    | Meaning                                                                  |
| ------- | ------------------------------------------------------------------------ |
| `start` | Request/model lifecycle metadata before text.                            |
| `delta` | Incremental `text`; concatenate these records to assemble the answer.    |
| `done`  | Final `usage` and `finish_reason`, **without duplicate aggregate text**. |
| `error` | A stream failure; not a successful `done`.                               |

There is no OpenAI `choices` envelope or `[DONE]` sentinel. The stream does not replay the completed answer in its final record. Client disconnects cancel generation. The host lease is retained until body
completion or disconnect; the worker retains admission until native cleanup
finishes. A disconnected stop request also retains its lifecycle gates through
worker teardown. Native NDJSON does not change existing OpenAI/Anthropic SSE or Ollama NDJSON behavior.
