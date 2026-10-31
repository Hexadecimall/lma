# Client connection presets

`lma s NAME -ct PRESET` prints connection settings after the server starts. Use `-p FILE` for a local GGUF. Hosting uses root `s`; stopping/unloading uses `sp`. Long commands/flags and legacy aliases are NSS-only, for example `lma nss serve NAME --client PRESET` or `lma nss server`; the removed mode flag is not accepted. The tables and long-option discussion below use **NSS flags only** (`--client`, `--api`, `--api-key`, `--dry-run`); root equivalents are `-ct`, `-a`, `-ay`, and `-d`. These client presets are **print-only recipes**: they do not install or execute clients, export environment variables, read their credentials, or write their configuration files. Separately, the terminal launcher implements owned local-provider sessions for supported OpenCode, Codex, and Claude Code **CLI** targets. External desktop-client targets fail closed with setup-required; they are not opened/configured, and CLI overrides do not establish desktop support. `--dry-run` prints the redacted lma plan only, without loading a model or starting clients.

The formats below were checked against the official documentation linked at the end on **2026-10-25**. They have **not** been verified end-to-end with installed Zed, Continue, OpenCode, Claude Code, or Codex releases. A protocol adapter or a valid configuration does not imply full coding-agent compatibility.

## Select a preset and API family

| `--client`    | Connection recipe                                                   | API when `--api` is omitted |
| ------------- | ------------------------------------------------------------------- | --------------------------- |
| `default`     | Concise addresses, model ID, authentication, one compatibility line | `all`                       |
| `zed`         | Zed's OpenAI-compatible **model provider**, not ACP                 | `openai`                    |
| `vscode`      | **Continue extension** in VS Code; no universal VS Code/Copilot API | `openai`                    |
| `opencode`    | Custom provider using `@ai-sdk/openai-compatible`                   | `openai`                    |
| `codex`       | User-level HTTP Responses provider                                  | `openai`                    |
| `claude-code` | Anthropic-format gateway environment values                         | `anthropic`                 |
| `custom`      | Generic addresses without a client-specific config                  | `all`                       |

There is no `continue` alias: use `--client vscode` for the Continue recipe. API choices remain `all`, `openai`, `anthropic`, and `ollama`. **Explicit `--api` wins in either argument order**, even when it disables the selected client's protocol. A disabled family produces a warning instead of a misleading client configuration. For example, `--client zed --api ollama` does not emit a usable Zed preset. Select `openai` or `all` to enable it.

NSS rejects root short commands (except legacy `lh`) but retains legacy option shortcuts: `lma nss serve --path model.gguf --client zed` works. Terminal chat uses `lma c NAME` / `lma nss chat NAME`, setup uses `lma lh` / `lma nss launch`, and help uses `lma h` / `lma nss help`.

## Credentials, model IDs, and network scope

- Use the exact served model ID shown at startup or in authenticated `GET /v1/models`. Ollama references may resolve to a normalized tag rather than a filename. Never rename a local model to a provider model ID to imply capabilities it does not have.
- Printed snippets **never contain the configured server API key**. When `--api-key` is set, `REPLACE_WITH_YOUR_LMA_API_KEY` means supply your actual key privately in the client. It is not a working credential. Environment references such as `{env:LMA_API_KEY}` are resolved by the client, not by lma.
- Without server authentication, `local-not-required` is a nonsecret dummy value for clients that require an API key field. It does not turn on server authentication.
- Avoid reusing a cloud-provider key for a local server. Do not commit real credentials or copy them into screenshots, issue reports, or shared project settings. Use the client's secret/keychain features where available.
- The examples use `http://127.0.0.1:8080`. Startup recipes use the actual bound port and map wildcard binds to loopback for local connection hints, not for changing the bind itself. Remote clients need a deliberately reachable address and appropriate authentication/TLS/firewall. A client's `localhost` is its own machine; SSH/container networking may need forwarding.
- Local inference does not make the entire client offline. Clients/extensions can contact their own telemetry, update, plugin, account, indexing, or tool services. Review those separately. Client tool calls can read code, change files, run commands, or access networks subject to client permissions; lma does not provide that sandbox.

The following examples assume the served ID is `model.gguf`, the server context is 8192, and the conservative output budget is 256. Generated recipes use your actual model/context. Agent instructions, history, schemas, and output must all fit the context. Large coding-agent prompts may not fit the default 4096-token server context; configure an appropriate supported context before use.

## Zed: OpenAI-compatible model provider

Start with `lma s model.gguf -c 8k -ct zed`. Merge the generated `language_models` entry into **user** settings using Zed's settings UI:

```json
{
  "language_models": {
    "openai_compatible": {
      "lma": {
        "api_url": "http://127.0.0.1:8080/v1",
        "available_models": [
          {
            "name": "model.gguf",
            "display_name": "lma (local)",
            "max_tokens": 8192,
            "max_output_tokens": 256,
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true,
              "interleaved_reasoning": false,
              "max_tokens_parameter": true
            }
          }
        ]
      }
    }
  }
}
```

Run `agent: open settings`, supply the `lma` provider key in the system keychain, and select the model. Alternatively, Zed's provider ID `lma` maps to **`LMA_API_KEY` in the local Zed process environment**. Do not put keys into `settings.json`; environment values take precedence over keychain values. Dock-launched Zed may not inherit terminal environment changes.

`max_tokens` is the context size in Zed's model definition. `max_tokens_parameter: true` tells Zed to send the generation limit as `max_tokens`, rather than `max_completion_tokens`. `chat_completions: true` selects Chat Completions, not Responses. The conservative recipe disables images, parallel tool calls, prompt-cache keys, and interleaved reasoning. `tools: true` enables the client path; set it to `false` for a chat-only model that cannot reliably call functions.

This powers Zed-owned AI features. It is **not an ACP provider or external-agent launcher**, and it does not configure external agents, Terminal Threads, or edit prediction. Those paths have separate configuration and compatibility requirements.

## VS Code: Continue extension

Start with `lma s model.gguf -c 8k -ct vscode`. Install Continue yourself and merge its generated YAML into your local `~/.continue/config.yaml`, not VS Code's `settings.json`:

```yaml
name: lma
version: 1.0.0
schema: v1
models:
  - name: lma
    provider: openai
    model: "model.gguf"
    apiBase: "http://127.0.0.1:8080/v1"
    apiKey: "REPLACE_WITH_YOUR_LMA_API_KEY"
    useResponsesApi: false
    roles: [chat]
    capabilities: [tool_use]
    defaultCompletionOptions:
      contextLength: 8192
      maxTokens: 256
```

Supply the key privately, or use Continue's supported secrets syntax, for example `apiKey: ${{ secrets.LMA_API_KEY }}` after configuring that secret through Continue. With no lma authentication use the printed dummy key instead. Select this model in Continue.

`useResponsesApi: false` selects Chat Completions even for model IDs that Continue might otherwise route to Responses. Only the `chat` role is configured: this does not provide embeddings, reranking, autocomplete, or a separate apply/edit model. `tool_use` enables the extension's tool path; remove it for chat-only models. It does not prove a particular model can operate Continue Agent mode successfully. This recipe does not configure Copilot or other VS Code extensions.

## OpenCode: custom Chat Completions provider

Start with `lma s model.gguf -c 8k -ct opencode`. Merge into `opencode.json` (or your existing user OpenCode config):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "lma/model.gguf",
  "small_model": "lma/model.gguf",
  "provider": {
    "lma": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "lma (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "{env:LMA_API_KEY}"
      },
      "models": {
        "model.gguf": {
          "name": "lma (local)",
          "limit": { "context": 8192, "output": 256 }
        }
      }
    }
  }
}
```

With authentication, set `LMA_API_KEY` in the environment where you separately start OpenCode. Without authentication, the printed recipe uses the dummy key instead of an environment reference. Alternatively, store a credential through OpenCode's `/connect` → Other with provider ID `lma`, and omit the explicit `options.apiKey` override. Select the model with `/models`.

The `npm` field identifies OpenCode's SDK adapter; lma installs nothing. `@ai-sdk/openai-compatible` uses `/v1/chat/completions`, whereas OpenCode documents `@ai-sdk/openai` for Responses. Both `model` and `small_model` are pinned to the local provider to avoid an implicit cloud helper model. The model map key must match the served ID, even when it contains slashes. Do not infer hosted web search, embeddings, reasoning transport, or successful full-agent operation from this entry.

## Codex: user-level HTTP Responses provider

Start with `lma s model.gguf -c 8k -ct codex`. Merge into **user-level `~/.codex/config.toml`** (or the file under your configured `CODEX_HOME`), preserving existing TOML tables rather than duplicating them:

```toml
model = "model.gguf"
model_provider = "lma"
model_context_window = 8192
model_supports_reasoning_summaries = false
web_search = "disabled"

[features]
enable_request_compression = false

[model_providers.lma]
name = "lma"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "LMA_API_KEY"
requires_openai_auth = false
supports_websockets = false

[model_providers.lma.capabilities]
remote_compaction = "unsupported"
external_web_access = false
```

Set `LMA_API_KEY` privately in the environment of the separately started Codex CLI; use the dummy key only if lma has no API key configured. `requires_openai_auth: false` does not disable lma authentication: it avoids requiring an OpenAI account for this provider.

Current official Codex docs say project-local `.codex/config.toml` cannot configure `model_provider`/`model_providers`; use user-level configuration. This is HTTP Responses/SSE, **not WebSockets**. Keep `supports_websockets = false` and `features.enable_request_compression = false` unless the server explicitly implements those transports. It does not implement hosted search, remote compaction, OpenAI account services, cloud tasks, or all Codex-specific tool/schema/reasoning fields. Client-managed local compaction and tools still depend on model/context and API subset compatibility. No sandbox/approval policy is weakened by this recipe.

## Claude Code: gateway environment values

Start with `lma s model.gguf -c 8k -ct claude-code`. The printed JSON is a map of environment values, **not a shell script or complete settings file**. Set them privately for the separately launched Claude Code CLI, or merge them under `env` in user `~/.claude/settings.json`:

```json
{
  "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
  "ANTHROPIC_AUTH_TOKEN": "REPLACE_WITH_YOUR_LMA_API_KEY",
  "ANTHROPIC_MODEL": "model.gguf",
  "ANTHROPIC_DEFAULT_OPUS_MODEL": "model.gguf",
  "ANTHROPIC_DEFAULT_SONNET_MODEL": "model.gguf",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL": "model.gguf",
  "ANTHROPIC_DEFAULT_FABLE_MODEL": "model.gguf",
  "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
  "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
  "CLAUDE_CODE_DISABLE_THINKING": "1",
  "DISABLE_PROMPT_CACHING": "1",
  "ENABLE_TOOL_SEARCH": "false",
  "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "8192",
  "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "256"
}
```

`ANTHROPIC_AUTH_TOKEN` sends `Authorization: Bearer`; `ANTHROPIC_API_KEY` is the alternative for `x-api-key`. lma accepts both on its Anthropic routes. Do not set conflicting credentials/provider overrides. For an unauthenticated server the emitted dummy value is nonsecret. Gateway credentials can override a saved claude.ai login; check the actual credential source in Claude Code's `/status`.

All family aliases above name **the same local model**, not actual Opus/Sonnet/Haiku/Fable models or their capabilities. The root gateway URL has **no `/v1` suffix**. Prompt caching, experimental betas, thinking parameters, and deferred tool search are disabled to reduce unsupported API behavior; this does not remove every unsupported request a client release may send. The context override applies only to model IDs allowed by that client's documented rules; a custom ID may otherwise inherit an incorrect large context assumption. A tiny local context may still be unusable with Claude Code's prompts/output requirements.

`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` suppresses updates and other background traffic, so arrange a separate update path. It is **not a complete outbound-network sandbox**: tools, plugins, and some checks can still access networks. No cloud sessions, Remote Control, hosted tools, adaptive/proprietary reasoning, or every advanced agent field is promised. Anthropic's gateway docs explicitly do not endorse/support routing Claude Code to non-Claude models. This is a configuration recipe for experimentation, not a certification that lma works as a full Claude Code backend.

## Ollama-compatible clients and raw Completions

There is no dedicated Ollama `--client` preset. Start `lma s model.gguf -a ollama` (or `-a all`; NSS: `lma nss serve model.gguf --api ollama`) and configure a client's Ollama base URL as `http://127.0.0.1:8080`, **without `/v1`**. Use the exact hosted model ID and, if configured, an `Authorization: Bearer` credential; `x-api-key` does not authenticate Ollama routes. Clients that cannot send the required credential need an appropriate trusted authentication gateway, not a real key embedded in a shared URL/config.

- `POST /api/chat` and `/api/generate` default to **`stream:true`**, using newline-delimited JSON (`application/x-ndjson`), not SSE. Set `stream:false` for a single JSON object. This changes the old lma missing-stream default; clients must not assume a whole JSON response when omitting it.
- Chat supports client-executed function tools with object-valued arguments and string-content tool results. Assistant history with null/empty `tool_calls` is an ordinary text turn, not a tool invocation. Real calls still require matching results and valid supplied definitions. Tool-constrained output is buffered until fully validated, not incremental argument streaming.
- Generate uses chat-template/system/user formatting by default. `raw:true` sends exact prompt text and rejects nonempty system text/images. Empty generate requests support explicit model loading/unloading. Infill suffixes, custom Ollama templates and continued `context` arrays remain unsupported.
- In the CLI HTTP host, `GET /api/tags` lists installed models, `/api/ps` lists the resident model, and `/api/show` inspects a selected installed model header. Unavailable digest/VRAM statistics are not invented. `GET /api/version` identifies lma, not a fabricated Ollama version.
- The HTTP host supports single-resident dynamic selection, keep_alive/unload, public GGUF pull, alias create/copy with [bounded text Modelfile definitions](modelfile.md), and source-specific deletion. `/api/create` accepts structured fields or inline Modelfile text, but `FROM` must be an installed selector, not a local/relative path or URL. Push, blob uploads, custom Go templates, adapters and unsupported parameters are rejected. Native embedding routes require model/backend support. Context cannot be resized per request. Desktop/TUI sessions remain explicitly selected, in-process sessions.
- Named definitions provide default system/history, built-in template selection, and sampling/load defaults. Explicit host load flags and supported request sampling/output fields win; clients that always send sampling settings override those defaults. Raw requests do not apply stored prompt formatting/history. See [precedence](modelfile.md#runtime-precedence); Ollama store template/system/parameter layers are still not applied.
- Check each NDJSON record for `error`, even after HTTP 200. Success ends with `done:true`; failures do not fabricate that terminal record. There is no SSE `[DONE]` marker in this protocol.

See [the complete Ollama subset contract](ollama-api.md) for supported options, images, limits, discovery omissions, authentication and error shapes. This is a wire-protocol adapter, not certification of any external Ollama client or every dense/MoE model.

For OpenAI **raw text** clients, `/v1/completions` accepts a string `prompt`, defaults to nonstreaming, and supports explicit `stream:true` as native SSE in `all`/`openai` mode. Successful streams end with a finish/usage chunk and `data: [DONE]`; stream errors emit an error event instead. This route has a 1 MiB body limit and does not add tools, chat history, images, stop sequences, logprobs, or multiple candidates. The client presets above use Chat Completions, Messages, or Responses rather than this legacy route.

## Protocol limits and verification

Function declarations, calls, and tool-result turns are protocol data. **The client executes tools**, checks permissions, and returns results; lma does not execute shell commands, browse, run an MCP server, or provide hosted web search/code interpreter. Use a local model and chat template capable of reliable schema-following. Forced function choices use an empty narration field so native tool/thought markers cannot leak through as assistant prose. Text streaming uses native generation events on supported routes; **tool arguments remain constrained/buffered**, not arbitrary incremental tool-argument streaming. This is not provider-identical event coverage or proof of compatibility with every client version.

Unsupported modes include ACP hosting, a universal VS Code API, hosted tool types/services, provider account/OAuth services, compressed request bodies, Responses WebSockets, server-managed response/session persistence, and arbitrary provider fields. Ollama remains a separate supported API subset, not a full daemon replacement. Messages accepts user base64 PNG/JPEG image blocks and Responses accepts user input_image data URLs (detail:auto), subject to native projector and image-ordering requirements; their bodies are limited to 1 MiB. Combining images with tool history has not been verified with a real model; token counting rejects images. Token counts use the served GGUF's tokenizer/applied chat template/generation special tokens, not Anthropic's proprietary tokenizer.

Start with plain text, then a single harmless function/tool round trip using the client's normal approval flow. Check authentication, model ID, context usage, streaming completion/errors, and tool-result resubmission. Record client version, local model/template, enabled API family, and observed results before claiming compatibility. Unit and mock protocol tests are not an installed-client end-to-end test.

## Official references

- [Zed model providers](https://zed.dev/docs/ai/llm-providers) and [current OpenAI-compatible configuration and key handling](https://zed.dev/docs/ai/use-api-access#openai-compatible).
- [Continue OpenAI-compatible setup and `useResponsesApi`](https://docs.continue.dev/customize/model-providers/top-level/openai) and [YAML reference: roles, capabilities, limits, secrets examples](https://docs.continue.dev/reference).
- [OpenCode custom providers, SDK choice, credentials, and environment references](https://opencode.ai/docs/providers/#custom-provider).
- [Codex config reference: user-level provider scope, Responses, WebSockets, compression, provider capabilities](https://developers.openai.com/codex/config-reference/).
- [Claude Code gateway connection/authentication/traffic controls](https://code.claude.com/docs/en/llm-gateway-connect), [model aliases and context behavior](https://code.claude.com/docs/en/model-config), and [environment variables](https://code.claude.com/docs/en/env-vars).
