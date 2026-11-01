# Ollama-compatible API

`lma s` starts the CLI HTTP host **without requiring or loading a model**.
It discovers installed models and loads the model named by a request. It keeps **one resident model at a time**, not one process per model.
Desktop/TUI in-process sessions remain explicitly selected single-model sessions.
This is an implemented protocol subset, not complete Ollama parity.

## Routes

Enable root hosting flag `-a all` or `-a ollama` (NSS: `--api all` / `--api ollama`). A configured API key is required as
`Authorization: Bearer KEY` for API discovery, generation, and administration.
`/health` is public. Mutation on a non-loopback bind requires an API key.

| Method      | Route             | Behavior                                                                 |
| ----------- | ----------------- | ------------------------------------------------------------------------ |
| POST        | `/api/chat`       | Text/images/function tools; native NDJSON streaming by default           |
| POST        | `/api/generate`   | Raw or template-formatted prompts; native NDJSON streaming by default    |
| POST        | `/api/embed`      | Native normalized vectors for a string or string array                   |
| POST        | `/api/embeddings` | Legacy single-prompt native embedding response                           |
| GET         | `/api/tags`       | Installed lma, Ollama, and standard LM Studio models; source information |
| GET         | `/api/ps`         | Currently resident model and configured context length                   |
| POST        | `/api/show`       | Bounded header metadata and stored definition, without loading weights   |
| GET         | `/api/version`    | Actual `lma/<package-version>` version, not an invented Ollama release   |
| POST        | `/api/stop`       | Unload the resident model; used by `lma sp` / `lma nss stop`             |
| POST        | `/api/pull`       | Download a public compatible GGUF from Ollama into `~/.lma/models/`      |
| POST        | `/api/copy`       | Create another lma model alias without duplicating weights               |
| POST        | `/api/create`     | Register a text definition/alias over an installed model selector        |
| POST/DELETE | `/api/delete`     | Source-specific deletion of lma, Ollama, or LM Studio models             |

Shared `/v1/models` lists installed models. OpenAI `/v1/embeddings` is enabled
with `all`/`openai`. Other OpenAI and Anthropic routes retain their documented
request subsets; see [client connections](clients.md).

`create` supports the [bounded text Modelfile subset](modelfile.md), with built-in
template names and typed parameters only. **Not implemented:** push, blob uploads,
arbitrary Ollama Go templates, training, adapters, unsupported parameters, remote
credentials, or cloud model execution. `create` does not pretend to convert arbitrary
safetensors: use `lma it -hr DIR -ot FILE.gguf` for supported profiles.
Unsupported creation fields fail instead of being silently ignored.

## Using lma in place of a local Ollama server

Once port 11434 is free, start an idle native host:

```sh
lma s -a ollama -pt 11434
curl http://127.0.0.1:11434/api/tags
```

Point an Ollama-compatible client at `http://127.0.0.1:11434` (not `/v1`).
The client selects an installed model by name. Existing compatible Ollama GGUF
blobs are read in place; no copy, download or Ollama daemon is needed. Use
`-a all` instead if that same listener should also expose the other APIs.
The installer does not stop Ollama, claim its port, or change client settings.

The baseline uses normal model residency: expert caching is **off unless `-e` is
explicitly supplied**, GPU offload is requested by default in GPU-enabled builds,
and memory fallback is `error`, not automatic CPU retry. This is a supported API
subset, not full Ollama replacement certification: unsupported model formats,
custom Go templates, ignored Ollama profile layers, missing options and client
requirements can still block a particular workflow. See [performance evidence](performance.md)
for the scoped measurements; universal throughput parity has not been established.

## Loading, residency and stopping

```sh
lma s                         # Idle host on 127.0.0.1:8080; no inference initialization
curl http://127.0.0.1:8080/api/tags  # Installed catalog without loading a model
# Optional: preload a model instead of starting idle
lma s -p /models/model.gguf -pt 8080
lma sp                      # unload from the local host on port 8080
lma nss stop --port 8080
```

API model selection is restricted to the installed catalog and the host's
explicit startup model. It is not an arbitrary filesystem-open endpoint. Duplicate
names prefer the lma catalog entry for generation. Source-specific catalog IDs
can disambiguate selection. Without a startup model, `/health` reports
`loaded:false`, `/api/ps` returns `{"models":[]}`, and `/api/tags` is available
immediately (with `all`/`ollama`). The first generation or explicit load request
must include `model`; omission while idle returns HTTP 400 and never picks a
model arbitrarily. With an explicit startup model, initial native loading happens
before the host is ready. Switching preflights the next model before unloading the old worker, but
a native allocation/load failure after teardown can leave the host unloaded.
The listener remains available for subsequent selections.

A generation's response owns a lifecycle lease through body completion or
client disconnect. Switching/stopping during an active request returns **409**;
cancel that request first. Models are torn down before loading their replacement,
so large native model processes do not overlap. HTTP stop unloads models from the
HTTP host, not unrelated desktop/TUI sessions or an Ollama daemon.

Ollama generation accepts `keep_alive` as seconds, a duration string (`30s`,
`5m`, `1h`, `100ms`), or `-1` for indefinite residency. Omission defaults to five
minutes. Expiration begins after response completion; newer requests invalidate
older expiration timers. `0` unloads after the response. An empty `/api/generate`
request with `keep_alive:0` explicitly unloads; an empty generate request with a
positive duration or `-1` loads the named model without generating. OpenAI and
Anthropic requests do not schedule automatic unloading.

## Administration

```json
{ "model": "MODEL:TAG", "stream": false }
```

Use that body with `/api/pull`. Public GGUFs only; multi-GB network/disk usage is
possible and this request explicitly starts the download. It uses the same Rust
checksum/size-verified downloader as `lma f`. `stream` defaults true for pull and
alias mutation: NDJSON carries a final success or error record, **not synthetic
byte-progress percentages**. Explicit `stream:false` returns one JSON response.

```json
{ "source": "existing:tag", "destination": "new:tag", "stream": false }
```

Use with `/api/copy`; it preserves a source lma definition as well as its weight
references. `/api/create` also accepts
`{"model":"new:tag","from":"existing:tag","stream":false}` as a profile copy.
For a new/overridden definition, use structured fields:

```json
{
  "model": "concise:latest",
  "from": "existing:tag",
  "system": "Be concise.",
  "template": "auto",
  "parameters": { "temperature": 0.7, "num_predict": 128, "num_ctx": 4096 },
  "messages": [{ "role": "user", "content": "Answer briefly." }],
  "license": "Local license metadata",
  "stream": false
}
```

Or supply inline text, not a server-side Modelfile path:

```json
{
  "model": "concise:latest",
  "modelfile": "FROM existing:tag\nSYSTEM Be concise.\nPARAMETER temperature 0.7",
  "stream": false
}
```

Use `model` or its alias `name`, not both. Inline `modelfile` cannot be mixed with
`from`, `system`, `template`, `parameters`, `messages`, or `license`. The create
body is limited to 1 MiB. **HTTP FROM/source accepts installed model names or exact
catalog IDs only** (including the explicitly allowlisted startup model), not
local/relative file paths or URLs. IDs containing store paths are opaque catalog
selectors, not arbitrary file-open requests. Use `lma ce NAME -f Modelfile`
for local files and Modelfile-relative `FROM` resolution.

Structured parameters accept `temperature`, `top_p`, `seed`, `num_predict`,
`num_ctx`, `num_gpu`, `num_thread`, `num_batch`, and `u_batch`; these are not all
per-request generation options. `template` selects an lma built-in name only.
Omitted metadata/parameters inherit from a source lma definition; child messages
append after inherited messages. See [the guide](modelfile.md) for bounds and
precedence. Unsupported fields, custom Go templates and adapters fail explicitly.

Aliases/definitions live in the lma registry; copy/create do not download, copy,
or load weights or modify external source stores. Existing aliases are not
overwritten. `/api/show` includes the selected model's stored `definition`, if any.

Deletion is real and source-specific:

```json
{ "model": "MODEL:TAG", "source": "ollama", "stream": false }
```

`source` can be `lma`, `ollama`, or `lm-studio`; ambiguous names require a source.
The host unloads its current model before deletion and rejects active requests.
Ollama deletion removes the original manifest, not an imported GGUF. Referenced
blobs are collected only after a complete bounded scan proves they are unshared;
malformed metadata, limits, or symlinks cause conservative retention. LM Studio
GGUF/split weights can be deleted, but projectors/config directories are not
recursively removed. lma aliases retain shared/external weights when appropriate.
Deletion results report deleted/retained files and warnings. Keep store parents
trusted and non-racing; this is not a crash-atomic filesystem transaction.

## Generation and streaming

`/api/chat` accepts `messages` with role/string content and optional base64 images.
`/api/generate` accepts `prompt`, optional `system`/`images`, and `raw:true` for an
exact prompt. Chat tools are client-executed; return function results as history.
No command, shell, browser, MCP, or code execution is performed by lma.

`stream` defaults **true**. Each NDJSON record ends with a newline. Text deltas
come from native generation, not splitting a completed answer. Chat uses
`message:{role,content}` and generate uses `response`. A successful terminal
record includes `done:true`, `done_reason`, engine token counts and measured
`total_duration`. Tool output is constrained/buffered until validated. Client
disconnect cancels native delivery; admission remains held until decoding exits.
Explicit `stream:false` returns one JSON object.

Supported options: `num_predict` as a positive integer or **`-1`**,
`temperature` (0–2), `top_p` (0–1, exclusive of zero), unsigned `seed`, and
`num_ctx` (integer 2–2147483647). `num_predict:-1` removes the reply cap and uses
remaining native context; EOS, cancellation and context exhaustion still stop
execution. When omitted, `num_predict` inherits the selected lma definition's
explicit limit, otherwise uses remaining context instead of the former 256-token
fallback. Zero, null, fractional limits and historical `-2` fill-context mode fail.
Positive limits remain bounded by the configured context.

The **managed HTTP host** honors `options.num_ctx` on `/api/chat` and
`/api/generate`, including empty-prompt load requests. A changed context reloads
the selected model through the same exclusive lifecycle as a model switch; the
old worker exits before the new one starts. This can incur model-load latency and
a larger context can fail memory allocation. Native load failure can leave the
host idle; no silent CPU fallback is enabled. Request context overrides the
startup/default context for that resident; subsequent requests that omit it keep
the resident context. Repeating the same context reuses the worker and prefix
cache. Malformed resize requests are rejected before unloading the current model;
active response leases return 409 rather than being interrupted. Pinned
in-process/desktop sessions cannot resize and still require matching context.

Unknown options fail. JSON/schema `format` and separate thinking control/output
remain unsupported. `think:false` does not suppress a model's internal reasoning.

For a selected lma definition, omitted sampling/output options inherit its
`temperature`, `top_p`, `seed`, and `num_predict`; explicit request values win
(including zero temperature/seed). Invalid/null fields are still errors. Stored
`SYSTEM` and `MESSAGE` history apply to chat-formatted prompts; explicit request
system instructions override `SYSTEM` (`system:""` clears it in generate), not
stored messages. `raw:true` applies no stored system/history/template formatting,
but retains sampling defaults. Explicit host load flags override stored load and
template defaults, also when switching models. The managed Ollama request's
`num_ctx` is an explicit per-resident context override as described above.
See [runtime precedence](modelfile.md#runtime-precedence).

Images require a compatible GGUF and projector. PNG/JPEG only, at most eight
images, 8 MiB per image and 16 MiB total; remote image URLs are not fetched.

## Embeddings

Native embedding requests share the worker/admission path. Inputs are tokenized
and evaluated in a separate lazy embedding context; generation-prefix cache is
not reused. Vectors are finite and L2-normalized; counts are actual evaluated
model tokens. Native model/backend/pooling compatibility is required.

`/api/embed` accepts a string or 1–128 strings and defaults `truncate:true`.
Truncation removes trailing tokens to fit both context and physical embedding
batch limits. `truncate:false` rejects over-limit input. `/api/embeddings`
accepts one `prompt`, with no truncation by default. `/v1/embeddings` supports
string batches and float or little-endian float32 base64 output. Token-ID input,
dimension overrides and rank pooling are rejected. These are not proprietary
OpenAI embeddings and not every generative model is useful for retrieval.

## Validation scope

Adapters, lifecycle leases/timers/authentication, deletion safety and parser
behavior have fixture tests. Earlier real Gemma4/Mellum2 checks established basic
native chat/tool compatibility, not complete client or model coverage. Actual
embedding-model inference, source-format-specific safetensors imports, all-GPU
performance and microphone capture require model/device-specific validation.
