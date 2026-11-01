# lma

Local models with a **native desktop app and built-in browser chat**, GGUF serving/quantization, public GGUF downloads, and local Stable Diffusion 1.5 image generation. The CLI and desktop share the Rust library: text and vision inference use **llama-cpp-2 in-process**; image generation uses Candle. No inference subprocess, `llama-server`, `llama-quantize`, or Ollama daemon is needed.

Load local GGUF files or compatible models already installed by Ollama in the desktop app, or run `lma s NAME` and visit **http://127.0.0.1:8080/**. Desktop uses Tauri IPC and `lma::desktop::Session` directly, without an HTTP listening socket. Only the explicit `lma f` / `lma nss fetch` command downloads model files; installers never download models. This is a small embedded runner, not a full llama.cpp, Ollama, or Diffusers replacement.

## Build and install

### Install CLI from source (desktop opt-in)

From a complete checkout, with prerequisites installed:

```sh
# macOS / Linux: preview, then build/install only the CLI for the current user
sh install/install.sh --dry-run
sh install/install.sh
# Optional: build/install CLI + native desktop
sh install/install.sh --desktop
# Updates also default to CLI only; add --desktop to update both
sh install/install.sh --update
```

```powershell
# Windows (PowerShell)
powershell -File install/install.ps1 --dry-run
powershell -File install/install.ps1
# Optional CLI + desktop installation or update
powershell -File install/install.ps1 --desktop
powershell -File install/install.ps1 --update --desktop
```

These wrappers use the Python installer to build **only the CLI by default**, including `--update`, with locked release dependencies, a shared cached `target` directory, and **two Cargo jobs** by default. `--desktop` explicitly opts into building/installing both components for that invocation; saved or legacy desktop registrations do not opt future runs in. CLI-only runs leave existing desktop apps, bundles and launchers unchanged and need no Tauri/webview dependencies. No elevation, `sudo`, toolchain installation, PATH edits, or model downloads. macOS defaults to Metal; CLI-only `--cpu` omits explicit GPU features but may retain implicit llama Metal on Apple Silicon. Linux/Windows default to CPU, with optional `--cuda` requiring a compatible toolkit. With `--desktop`, macOS requires Metal and installs `~/Applications/lma.app`, ad-hoc signed when `codesign` is available—not Developer ID signed or notarized.

See **[installation, prerequisites, updates, and recovery](install/INSTALL.md)** for platform dependencies, install locations, build controls, and Windows staging. Rust must support **edition 2024**; native builds need a C/C++ compiler, CMake, and **libclang** for bindgen. Desktop additionally needs the platform webview dependencies and, on macOS, a working Metal compiler. Initial dependency downloads and native compilation can take a while.

### Direct builds

For an optimized **CLI-only** build, use the standalone helper from the repository root:

```sh
rustc --edition=2024 build.rs -o lma-build && ./lma-build -j 2
./lma-build --cpu --dry-run               # Preview without building the app
./lma-build --portable -j 2              # Omit automatic host-CPU tuning
./lma-build --cuda -j 2                  # NVIDIA on non-macOS; CUDA toolkit required
```

`build.rs` is **not a Cargo build hook** (`package.build = false`). It launches build-time compiler/Cargo subprocesses, not inference subprocesses. The helper explicitly enables `vision,image-gen,speech`, plus `metal` on macOS or no explicit GPU feature elsewhere; `--cuda` opts into CUDA. Its `--cpu` omits explicit GPU features but **does not remove implicit Metal on Apple Silicon**; use runtime `lma s NAME -g cpu` to disable offload. Fetch, quantization, and browser chat are always included. It builds for the active Rust compiler's host, with output under `target/<host>/release/lma` (or the configured Cargo target directory; `.exe` on Windows).

Host-CPU tuning (`target-cpu=native`) is the default and may produce binaries incompatible with older CPUs. Use `--portable` to omit that tuning; it preserves supplied flags rather than guaranteeing portability. `CARGO_ENCODED_RUSTFLAGS` takes precedence over `RUSTFLAGS`; existing flags are retained, and custom `target-cpu` settings require `--portable`. User `CARGO_PROFILE_RELEASE_*` environment overrides still apply. The helper sets native build profiles (`LLAMA_LIB_PROFILE` and `CMAKE_BUILD_TYPE`) to `Release`.

The release profile uses optimization level 3, fat **Rust LTO** (not cross-language Rust/C++ LTO), one codegen unit, no debug info or incremental compilation, and symbol stripping. Stripping reduces binary size, not execution time. LTO increases build time and peak memory; `-j 2` limits Cargo jobs, not all native compiler parallelism. These settings do not guarantee the fastest result for every model or machine.

Direct Cargo builds and installation remain available:

```sh
cargo build --locked --release             # vision + image-gen + speech; implicit llama Metal on Apple Silicon
cargo install --locked --path .            # Optional: install CLI as `lma`
cargo build --locked --release --no-default-features # Text-only, not necessarily CPU-only
cargo build --locked --release --no-default-features --features vision # Vision without image-gen
cargo build --locked --release --features metal    # Explicit Metal, including Candle
cargo build --locked --release --features cuda     # NVIDIA; CUDA toolkit required
cargo build --locked --release --manifest-path desktop/Cargo.toml # Native desktop
```

Direct Cargo release builds (without an explicit target) produce `target/release/lma`; use the appropriate binary path below if not installed. Add the same GPU feature to `cargo install --path .` for a GPU installation. Optional, default-enabled features are `vision` (`llama-cpp-2/mtmd`), `image-gen` (Candle, tokenizers, and PNG output), and `speech` (Candle Whisper and WAV decoding/resampling). With defaults disabled, enable the required features explicitly; image/transcription help and dry-run remain available without their backends.

The explicit `metal` and `cuda` features enable the corresponding llama.cpp backend and, when `image-gen` or `speech` is enabled, Candle backend. **On Apple Silicon, llama-cpp-2's target dependency enables Metal even with root `--no-default-features`.** That flag removes vision/image generation/speech, not necessarily GPU support. Implicit llama.cpp Metal does not enable Candle Metal; use the explicit `metal` feature for that.

**Serving defaults to all GPU layers when Metal or CUDA is compiled in, otherwise CPU.** `config::default_gpu_layers` also recognizes implicit Apple Silicon Metal. This is a compiled-backend policy, not a hardware/memory-fit guarantee: unavailable hardware can cause loading to fail. Root hosting flag `-g cpu` explicitly disables all model, KV/compute, and projector offload; `-g all` or a count requests GPU layers. Image generation has its own `-de auto` policy: try compiled CUDA, then Metal, then CPU; use `-de cpu` to force CPU there.

Desktop is a **standalone Cargo package**, using `desktop/Cargo.toml` and `desktop/Cargo.lock`, not a root workspace member. It enables vision/image generation/speech and adds Metal on macOS; other platforms use CPU unless built with `--features cuda`. Its default `custom-protocol` feature enables `tauri/custom-protocol` so a plain Cargo build embeds the UI. If disabling desktop defaults, retain `--features tauri/custom-protocol`. Direct desktop builds normally output to `desktop/target/release/`; set `CARGO_TARGET_DIR` to share artifacts with root builds, as the installer and CI do.

## Quick start

```sh
lma h                                    # Help (long form: lma nss help)
lma l                                    # Installed models (lma nss list)
lma lh                                   # Model picker + chat (lma nss launch)
lma c llama3.2                           # Terminal chat (lma nss chat llama3.2)
lma t -m models/whisper-tiny audio.wav    # lma nss transcribe -m ... audio.wav
lma s                                    # Idle server; /api/tags lists installed models
lma s llama3.2 -c 8k                     # Optional preload; open localhost:8080
lma s -p model.gguf -g cpu               # Explicit local file and CPU inference
lma s -p vision.gguf -pr mmproj.gguf
lma sp                                   # Stop/unload the active model (lma nss stop)
lma sp -m llama3.2 -d                    # Preview stop selection; no HTTP request
lma q model.gguf                          # Writes model.Q4_K_M.gguf
lma f llama3.2 -l                        # Browse both download sources
lma i models/sd15 "a watercolor fox" -o fox.png # Requires local SD1.5 assets
```

| Root command | NSS command          | Purpose                               |
| ------------ | -------------------- | ------------------------------------- |
| `lma h`      | `lma nss help`       | Local help                            |
| `lma l`      | `lma nss list`       | Installed models                      |
| `lma lh`     | `lma nss launch`     | Model picker and terminal chat        |
| `lma c`      | `lma nss chat`       | Terminal chat                         |
| `lma ce`     | `lma nss create`     | Register a local Modelfile definition |
| `lma d`      | `lma nss delete`     | Delete an installed model             |
| `lma t`      | `lma nss transcribe` | Local Whisper transcription           |
| `lma s`      | `lma nss serve`      | Host browser chat and HTTP APIs       |
| `lma sp`     | `lma nss stop`       | Stop/unload on the local managed host |
| `lma q`      | `lma nss quantize`   | Quantize a local GGUF                 |
| `lma f`      | `lma nss fetch`      | Explicit model downloads              |
| `lma i`      | `lma nss image`      | Local SD1.5 image generation          |
| `lma it`     | `lma nss import`     | Convert supported local safetensors   |

Root commands and flags have **one canonical spelling**: the first letter, or first + last letter to resolve a collision (`lh`, `ce`, `sp`, `it`). There are no long command aliases or long flags at root. Use `lma -h` / `lma -v` for help/version, not `--help`, `--version`, or `-V`. Command flags reserve `-h` for help and `-d` for dry-run. Detailed help uses `lma h s -a` or `lma nss help serve advanced`.

Long commands, long flags, and legacy aliases are **NSS-only**, for example `lma nss server`, `lma nss quant`, `lma nss imagen`, and `lma nss img`. Bare `--` at root is an error, not an escape into NSS; start with `lma nss COMMAND` explicitly. Multi-letter root flags are exact names, not bundles; see the command tables below. Cargo, Python, installer, and build-helper flags are separate and unchanged.

### Stop or unload

`lma sp` (or `lma nss stop`) requests `POST /api/stop` from an already running local managed host, defaulting to `127.0.0.1:8080`. Omit `-m` to stop the current model, or use `-m NAME` for an installed model. Stop flags `-ht HOST`, `-p PORT`, and `-a KEY` select/authenticate the managed endpoint; the API key is redacted in previews. `-d` prints configuration without a request. Stop unloads inference state; it does not find or kill arbitrary OS processes. **`lma s` now serves; migrate old stop invocations to `lma sp`.** Hosting uses `lma s NAME` or `lma s -p model.gguf`; note that hosting's `-p` is a path and `-pt` is its port. NSS equivalents use `--model`, `--host`, `--port`, `--api-key`, and `--dry-run`.

## Installed models

```sh
lma l                                    # Names, sources, sizes, model references
lma l pulsar                             # Case-insensitive name/reference filter
lma l -j                                 # {models, warnings}; full paths and byte sizes
lma l -dr /path/to/models                 # Additional GGUF folder; repeatable
lma l -o /path/to/ollama/models           # Explicit Ollama store
```

Read-only discovery scans `~/.lma/models`, `./models`, `LMA_MODELS` (one directory), the desktop app-data `models` folder, LM Studio folders (`LM_STUDIO_MODELS`, `~/.lmstudio/models`, `~/.cache/lm-studio/models`), and one Ollama store. Ollama precedence is root `-o` (NSS `--ollama-dir`), then `OLLAMA_MODELS`, then the user's `.ollama/models`. No daemon, network requests, model loading, downloads, or filesystem writes. Scans use the existing bounded GGUF/manifest discovery and skip projectors and incomplete downloads. Empty results succeed; warnings appear on stderr, or in `warnings` with `-j` (NSS `--json`). `-d` (NSS `--dry-run`) previews options without scanning.

On a terminal, model listings adapt to its width: wide windows show a table with shortened long cells; narrow windows show wrapped entries. Use `-j` for full names, paths, and exact byte sizes. Piped listings retain the stable, untruncated `NAME`, `SOURCE`, `SIZE`, `MODEL` text layout.

Lma-owned persistent settings, caches and models now default to **`~/.lma/`**; legacy desktop state remains readable without automatic moves. See the [storage policy](docs/storage.md). `~/.lma/models` is lma's default download and model-registry location. Ollama and LM Studio keep their own stores; listing or registering an external model does not copy it into `~/.lma` or transfer ownership. A registry alias can point at external/shared weights, so removing an alias is not a promise that its weights are deleted. Explicit output/store options still select their documented locations.

## Delete installed models

```sh
lma d MODEL -d                 # Argument preview only; no resolution or deletion
lma d MODEL                    # Interactive terminal confirmation
lma d MODEL -s lma -y          # Explicit source guard; skip confirmation
```

Deletion can permanently remove model files. **Unload the selected model first**
with `lma sp -m MODEL` or the desktop app, and stop its use in other processes.
Deletion never unloads models or contacts a host for you. Non-terminal stdin
requires `-y`; piping `yes` is not confirmation. `-s lma|ollama|lm-studio` guards
the source; `-o DIR` selects an Ollama store, not a source. `-d` does **not** resolve
a file list or prove that the model exists or is deletable. See the
[deletion guide](docs/deletion.md) for source-specific retention and safety rules.
Long syntax is `lma nss delete MODEL --source lma --yes`.

## Optional MoE expert caching

```sh
lma s -p existing-moe.gguf -e 1024 -g all -mh 16  # Explicit bounded GPU prefill
lma s -p existing-moe.gguf -e 1024 -g cpu  # Explicit CPU cache mode
```

A native, bounded **expert-slot cache** retains hot routed experts while reading
cold experts from the existing mmap GGUF. GPU requests place compact slots and
single-token expert matmuls on GPU; **SILU prefill with 2..16 local tokens** now
uses capacity-safe sequential GPU expert tiles. Larger (>16) and empty local
views use original mmap CPU expert matmuls; explicit `-g cpu` caching is unchanged.
Requested batch/micro-batch settings are preserved, not forced to 1. Explicitly
choose root `-mh 16` or NSS `--ubatch-size 16` to keep each local prefill at most
16 tokens; the cache does not set this automatically. Longer prompts within the
context limit are split across multiple native 16-token micro-batches, not
limited to 16 tokens overall.

Cache mode saves model-specific routing counts, not text, under
`~/.lma/expert-cache/` (optionally `LMA_EXPERT_CACHE_DIR`) and warms **at most
64 MiB** on later loads. It stores no prompts, generated text, or duplicate
on-disk weights. The feature is **off by default**: cache-off inference leaves
history untouched and performs no expert warming or placement/batching changes.

Initial support is limited to compatible `qwen2moe`, `qwen3moe`, and `mellum`
layouts. The budget bounds compact device slot allocations including alignment,
not total RSS; Apple unified memory remains shared system memory. **Graph metadata
and workspaces can increase**: reservation includes worst-case 16-token stream
expansion, outside the slot budget. Routing/top-k preparation stays on CPU and
uploads are synchronous, with no predictive/asynchronous prefetch. Counts now
include bounded GPU prefill, but not >16-token CPU fallback operations.

Sequential Mellum2 GPU checks with a **1 GiB cache** pass decode/history,
micro-batch-32 short prompts (all local chunks <=16), and **micro-batch-16 long
prefill across multiple micro-batches**, at unchanged logit tolerances with exact
text/usage/finish. After the view-placement fix, full-feature GPU micro-batch-16
long-prompt and history reruns also **pass**; history warms `66,834,432` bytes.
This does **not** establish 32-local-token GPU expert batches.
**A 2 GiB cache with normal fusion fails short prompt B before reaching the long
case**; a fusion-disabled diagnostic is bitwise exact in all cases, including
the long prompt. This larger-cache, fusion-related drift remains **unresolved**;
tests are not loosened and production fusion is not globally disabled.

The final full native suite **passes all 26 CTests**: **22 GPU configurations and
four other native tests**, with **no skips or timeouts in 271.9 seconds**. Four
added fused-weight cases cover >16-token CPU fallback and empty outputs with
fusion on/off after the backend view-placement fix. These passes do not close
the separate 2 GiB real-model validation gap; the view fix is not a resolution
of that fusion-related drift. See the [expert-cache guide](docs/expert-cache.md)
for scoped evidence. No universal GPU-prefill/offload parity, throughput, or
SSD-speed guarantee is claimed. Final model-free checks pass **508 Rust library,
56 CLI and 56 desktop-state tests**. Latest full-feature CLI and desktop release
builds both **pass**; installed binaries are unchanged.

CLI `lma f` downloads now show a throttled progress bar on stderr, with byte
counts, percentage, and speed. Completion is marked verified only after checksum
verification and publication; reused files are not downloaded again.

## Local Modelfile definitions

```sh
lma ce concise -f recipes/Modelfile -d
lma ce concise -f recipes/Modelfile
lma nss create concise --file recipes/Modelfile --dry-run
```

Create a named text-runtime profile over an installed model or existing local weights, without copying/downloading weights or loading inference. CLI relative `FROM` files resolve from the Modelfile's directory. Definitions support inheritance, `SYSTEM`, initial `MESSAGE` history, `LICENSE` metadata, built-in `TEMPLATE` names, and a bounded parameter subset; custom Go templates, `ADAPTER`, and unsupported parameters are rejected. Explicit load flags and supported request sampling fields override stored defaults. Only explicit `lma ce` / `lma nss create` reads the default `./Modelfile`; serving does not scan for one. See **[the Modelfile guide](docs/modelfile.md)** for syntax, precedence, dry-run behavior, and HTTP installed-selector restrictions.

## Import installed Gemma4 safetensors

Dense Gemma4 models installed in Ollama's **MLX per-tensor safetensors** format can now be converted locally, entirely in Rust:

```sh
mkdir -p models
lma it -o gemma4:12b -ot models/gemma4-12b.gguf -d
lma it -o gemma4:12b -ot models/gemma4-12b.gguf
lma c -o gemma4:12b
```

The dry run validates architecture/configuration, tokenizer, tensor shapes, quantization scales, and offsets, and reports the exact output size without writing a model or loading inference weights. The installed `gemma4:12b` used during validation produced a **6.72 GiB** GGUF. Full conversion streams bounded chunks, leaves the original Ollama store untouched, never overwrites an existing output, and publishes only after success. The output parent must already exist. No network, Python, Ollama process, or external converter is used.

**This first import path is text-only.** It preserves native NVFP4 projection data/global scales, promotes norm/scalar values to F32 where needed, and converts NVFP4 token/output embeddings to **Q8_0 (lossy)**. It skips vision/audio encoders and speculative draft tensors. This Ollama MLX profile rejects other architectures, MoE/shared-KV variants, and unsupported quantization schemes rather than silently guessing them. Existing compatible GGUF+projector vision support is unchanged.

A small `.lma-import-v1-*.json` receipt beside the GGUF reconnects the original Ollama tag to the imported file. If this folder is `./models`, `LMA_MODELS`, the app-data model folder, or the build checkout's `models`, normal `-o gemma4:12b` resolution reuses it. It checks the source store/tag, manifest fingerprint, output size, and GGUF header. Changed manifests or missing/modified-size outputs invalidate reuse; it never starts a conversion automatically. Other output folders remain usable by their explicit GGUF path. Receipts are trusted local configuration, not cryptographic certification of the weights; keep stores, receipts, and model files private and immutable during use.

Import root flags are `-m MODEL`, `-o OLLAMA_TAG`, `-hr HF_DIR`, `-or OLLAMA_DIR`, and `-ot OUTPUT`, plus `-h` / `-d`. Long equivalents belong only to `lma nss import`.

The shared formatter recognizes the imported Gemma4 unified turn protocol and prefills an empty thought channel for ordinary chat, avoiding visible thought/protocol markers. Root hosting flag `-ce gemma4` (NSS `--chat-template gemma4`) is an explicit override. Real local tests exercised native chat, token-count consistency, and a constrained function-call/result round trip; they do not certify every external coding client or multimodal capability.

### Explicit local Hugging Face profiles

A local HF directory can also be imported with `lma it -hr ./local-hf-model -ot models/local.gguf -d`, followed by the same command without `-d`. The output parent must exist. Current architecture profiles are **`LlamaForCausalLM`**, non-sliding **`MistralForCausalLM` / `MixtralForCausalLM`**, and standard **`Qwen2ForCausalLM`**, with strict configuration and tokenizer validation. F32/F16/BF16 matrices retain their precision; vectors are promoted to F32. Supported single-file/sharded safetensors are streamed locally; source files are not modified, remote model code is not executed, and nothing is downloaded. Unsupported quantized, tokenizer, scaling, or architecture variants fail explicitly. Synthetic fixtures do not establish compatibility with arbitrary models or replace real-model validation.

Implementation: [`src/import/`](src/import/mod.rs). Large local `models/` artifacts are ignored by source control.

## Terminal chat and client hosting

`lma lh` opens model setup; `lma c NAME` goes directly to terminal chat. Use positional `NAME` or `-m NAME` for an installed model, and `-p FILE` for an explicit local file. The removed mode flag is not accepted by chat, launch, or serve; use `lma s` for HTTP hosting. Both use the shared in-process engine. Enter sends; **Shift+Enter** inserts a newline when reported distinctly, with **Alt+Enter** as the fallback. Keyboard enhancement is requested only on terminals supporting it; legacy terminals may send Shift+Enter as plain Enter. Tab switches composer/history. The mouse wheel and **PgUp/PgDn** scroll history; when history is focused, **Up/Down** scroll and **Home/End** select oldest/latest (composer arrows still edit input lines). **Ctrl+Home/End** jump to oldest/latest from either focus. Older messages stay in place while replies stream or finish; Ctrl+End resumes following live output. A scrollbar shows your position. Esc cancels, and Ctrl+Q quits. `/help` lists commands including `/models`, `/system`, `/tokens`, `/image`, `/logs`, and explicit `/save FILE`. Terminal conversations stay in memory unless exported.

Assistant replies render Markdown in the ratatui terminal chat: headings, **bold**, _italic_, inline/fenced code, lists, quotes, and links. This is safe display formatting, not an execution environment: code is never run, links are not opened automatically, and HTML is not executed. Model text cannot supply terminal control sequences. Use `lma h c` for the full keyboard, slash-command, and attachment limits. Use `/plain` to show literal Markdown and `/markdown` to restore formatted display. Requests and exports retain the original text; this is a bounded display subset, not full CommonMark. Formatting does not change the model's capabilities or promise a speedup.

### Reply length (not model lifetime)

```sh
lma c MODEL -ts 2048 -c 8k      # Configurable initial reply cap
lma c MODEL -ts off -c 8k       # No separate reply cap; use remaining context
lma nss chat MODEL --tokens off --ctx-size 8k
```

Inside chat, `/tokens N` changes the cap, `/tokens off` (or `/tokens 0`) removes it, and `/tokens` shows the setting. The default remains `min(256, context - 1)` unless `-ts` overrides it. The native engine determines no-cap allowance after actual prompt/template tokenization and image-position preparation. **No cap does not mean infinite context**: EOS, Esc/Ctrl+C cancellation, and the loaded context window still stop generation. Limits never unload the model. Status messages distinguish a reply cap, full context, and model EOS; use `/save` then `/new` when history fills the window, or restart with a larger `-c`. History is never silently dropped or shifted. HTTP `max_tokens` validation is unchanged; no-cap terminal requests use a private in-process policy, not a public API override.

```sh
lma s model.gguf -ct zed -c 8k
lma s model.gguf -ct vscode -c 8k    # Continue extension
lma s model.gguf -ct opencode -c 8k
lma s model.gguf -ct codex -c 8k
lma s model.gguf -ct claude-code -c 8k
lma s -p model.gguf -a all
```

Root `-a all|openai|anthropic|ollama` (NSS `--api`) controls routes. `lma s -ct PRESET` (NSS `serve --client`) prints a recipe, without launching clients or editing configuration/credentials. Separately, the terminal launcher's supported external CLI targets can start owned local-provider sessions; external desktop-client targets fail closed with setup-required and are not opened/configured. When API selection is omitted, Claude Code selects Anthropic; Codex/Zed/VS Code/OpenCode select OpenAI; default/custom select all. Explicit API selection wins. See **[client setup and limits](docs/clients.md)** for exact configuration, authentication, and version caveats. Function calls/results are supported protocol data; **clients execute tools and enforce permissions, never lma**. Local model quality/context still determines whether coding-agent workflows are useful; no installed-client end-to-end certification is claimed.

## Desktop and browser chat

Launch the installed desktop app, or `cargo run --locked --manifest-path desktop/Cargo.toml`. The desktop has its own chat-first interface in `desktop/ui/`: saved conversations in the sidebar, a full-page **Models** library, and an inline conversation inspector for generation/settings/export. There are no mandatory model or settings dialogs at startup. It automatically scans lma model folders and the installed Ollama store, showing a searchable model list with source labels and sizes. Select a model once; after a successful load, the app remembers and restores that selection on the next launch, including its projector, Ollama store, GPU layers, and context size. A failed restore leaves the model library available with an error, without an automatic retry loop. With no saved selection, the app never arbitrarily loads a large model just because it was discovered.

The model library has **Rescan**, GPU/context controls, and optional manual file/Ollama import. The header's quick switch changes models without leaving chat. `auto` follows the shared compiled-backend default. Discovery reads bounded directory/manifest data and four-byte GGUF headers; it does not load weights, copy files, download models, or require an Ollama daemon. Corrupt/unsupported entries are skipped with scan warnings. Discovery identifies GGUF files, not guaranteed architecture compatibility.

Local search locations are `LMA_MODELS` when set, the app-data `models/` directory, and `models/` under the current working directory, registered source checkout, and build checkout. Discovery also includes `~/.lma/models/`, the fixed CLI fetch destination. Downloads never default to the working directory; use `lma f QUERY -o .` to request it explicitly. The normal Ollama store follows `OLLAMA_MODELS`, then the home `.ollama/models` directory; a remembered custom store is scanned too. Custom paths outside scanned folders remain available as the previous selection. No whole-disk scan is performed.

The desktop bridge calls the same Rust inference, validation, limits, and model/file-resolution code as the CLI; it is not a separate inference engine or a shell around `lma s`.

Browser mode remains available: start `lma s model.gguf`, then open **http://127.0.0.1:8080/** (or your configured host/port). The CLI owns model selection in browser mode; desktop model loading and source updates are not exposed as public HTTP routes. The browser keeps its existing, separate UI in `src/runtime/ui/`; this desktop redesign does not change the CLI, inference behavior, or browser interface. Both frontends are embedded: no frontend build, npm install, CDN, or separate UI server. Desktop uses Enter to send and Shift+Enter for a newline; Ctrl/Cmd+Enter also sends. Browser mode keeps its existing Ctrl/Cmd+Enter shortcut. Replies arrive whole, not streamed.

- **Quiet interface:** concise controls and status, with no tutorial panels or routine success notifications. Detailed usage lives here rather than alongside every interaction.
- **Coding-oriented chat:** optional coding-instructions preset and custom system prompt; Markdown rendering with fenced code blocks, language labels, and code copy/download controls. The preset is prompt text only, not an execution agent.
- **Conversations:** search, rename, and delete saved desktop chats in the sidebar; edit a user turn or regenerate a reply as a new branch without changing the original. Drafts and chat history survive desktop restarts. Export Markdown or JSON; copy/download individual responses. Typing uses a lightweight update path without repeatedly cloning/decoding unrelated image history.
- **Controls:** max output tokens (UI default **128**, API default 256), temperature, top-p, and optional seed; response token counts, elapsed time, and output-limit notices. Stop cancels the browser request; the server may still be finishing work. Failed/stopped requests preserve the draft and attachments for retry.
- **Chat or raw completion:** chat sends history through `/v1/chat/completions`, preferring the model's embedded template. Missing/unusable templates automatically fall back to a recognized special-token format or plain role-labeled text, so compatible base/edit GGUFs no longer fail solely because they lack a chat template. Raw mode still sends only the exact composer text to `/v1/completions`—no history, system/coding prompt, or images. Desktop dispatches through IPC to the shared core; browser mode uses HTTP.
- **Image input:** attach PNG/JPEG files for a compatible vision model/projector. Request limits include images in conversation history; this is image understanding, not image generation.
- **Browser API keys:** if started with `-ay KEY` (NSS `--api-key KEY`), enter the key in Settings. It is sent only to the same server and excluded from exports. The page and static assets remain public; API calls remain authenticated. Desktop's trusted in-process bridge does not require an HTTP API key.

**Desktop storage:** conversations, drafts, attached images, system instructions, and generation settings are autosaved locally in `desktop-workspace.json` under the app-config directory (on macOS, `~/Library/Application Support/org.lma.app/`). Saves are bounded to 64 MiB, validated, and published atomically; Unix files use private permissions. Save errors remain visible and keep the working data in the window rather than discarding it. Autosave is debounced; closing/termination before a pending save completes can lose the latest edits. Model selection remains in the separate `model-preferences.json`. These are local plaintext files, not encrypted storage; remove conversations or the workspace file if you do not want them retained. No API key is stored in the desktop workspace.

**Browser storage is unchanged:** conversations, drafts, attachments, generation settings, and API keys remain in the page's memory; reloading loses them. Export explicitly to keep a copy. Exports can contain sensitive prompts/images. Neither interface provides streaming, tool calling, code/test execution, or an image-generation UI/HTTP endpoint. Model output is unverified; review code before running it yourself.

### Explicit source updates

For a source-installed app, the desktop update button starts a **user-confirmed source rebuild**, not an unattended “latest version” check or a signed binary updater. It requires the registered checkout, build tools, and `ghx`. The updater refuses dirty tracked/untracked files and any origin other than `https://github.com/Hexadecimall/lma.git`, then uses `ghx pull origin` (fast-forward only) on the current branch. Both locked release builds and packaging must succeed before installed files are replaced. A successful pull is not undone if compilation fails; existing installed binaries remain intact on build failure.

The app checks update registration first, then cancels active requests and unloads the model before starting the source build; restart manually after success. On Windows, close the app/CLI and use the external PowerShell installer; locked executables may require a staged update via `--apply-staged`. There is no automatic download, background update, release-signature verification, or automatic relaunch. See [the installer guide](install/INSTALL.md) for exact commands, trust assumptions, rollback limits, and recovery. Installation and explicitly requested updates run build/tool subprocesses; **inference does not**.

## Serve and installed Ollama models

`lma s [NAME] [OPTIONS]` (or `lma nss serve [NAME]`) starts a server. With no model,
it starts **idle**, without a picker or inference initialization. `GET /api/tags`
lists installed models immediately with the default `all` profile (or `-a ollama`);
`GET /v1/models` works with every profile, and `/api/ps` is empty until a model is
loaded. The first generation request must include an installed `model` name;
later requests can use the resident model. Nothing is downloaded implicitly.
An explicit `NAME` preloads an installed model; root `-m NAME` also selects it. Use root `-p FILE` to explicitly select a local GGUF, including a file whose name collides with an installed model. Positional/`-m` file paths remain accepted; existing local files win. On actual startup, installed names (e.g. `llama3.2`) go through the shared installed-model resolver. Missing absolute, explicitly relative, or `.gguf` paths remain path errors; use `ollama:NAME` to disambiguate.

- Select explicitly with root `-o NAME` or positional `ollama:llama3.2:latest`. `lma nss serve --ollama NAME` also works; NSS retains its legacy option shortcuts.
- Store precedence: **root `-or PATH` (NSS `--ollama-dir PATH` / `-od PATH`) > `OLLAMA_MODELS` > `HOME/.ollama/models`** (Windows uses `USERPROFILE`, then `HOME`). Only that store is searched, not broad system locations.
- Read existing manifests and GGUF blobs in place: one model and an optional projector. No daemon, downloads, or weight copies. Ollama multi-model-layer shards and adapters remain unsupported; supported Gemma4 safetensors need explicit import first.
- Local complete named split-GGUF sets load directly without merging: select any `name-00001-of-000NN.gguf` part. Discovery lists the validated set once with combined size. This does not add split downloads or arbitrary Ollama blob merging. See [model support](docs/model-support.md) for dense/MoE, formats, and tested limits.
- Installed Hugging Face imports may include a Docker v2 provenance manifest. One such local manifest is accepted when its model/projector digests match the main manifest; nested chains and additional inference layers are rejected.
- Model IDs use the normalized Ollama name/tag, such as `llama3.2:latest`, not a blob digest. An omitted tag becomes `latest`; local files use their filename.
- Ollama template, system, and parameter layers are **ignored with a startup warning**. Chat prefers the GGUF template, with automatic compatibility formatting when it is missing/unusable; it does not evaluate Ollama's Go templates/settings.
- Root `-pr PATH` (NSS `--projector PATH` / `--mmproj PATH`) overrides a discovered projector.
- Root `-d` (NSS `--dry-run`) with an explicit `ollama:` reference (including `-o`) or a store override with a supplied model (`-or` at root, `--ollama-dir` in NSS) performs resolution, reading manifests and selected blob headers, but **does not load weights**. Explicit `lma:NAME:TAG` previews also resolve the registered definition and its load defaults without loading weights. Ordinary local-file previews remain no-read; plain names without those explicit selectors are not resolved during preview. API keys are redacted.

### Server options

These are the canonical root flags for `lma s`, `lma c`, and `lma lh`. The backend recipes and long-option discussion below use **`lma nss serve`**; those long flags are never accepted at root.

| Root flag                  | Meaning / default                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `-m NAME`, `-p FILE`       | Installed model or explicit local GGUF; positional names also work.                                                 |
| `-o NAME`, `-or PATH`      | Installed Ollama selector and store override.                                                                       |
| `-c N`, `-t auto\|N`       | Context tokens (4096; `8k` = 8192; minimum 2) and CPU threads (auto).                                               |
| `-g all\|cpu\|N`           | Default `all` with compiled Metal/CUDA, otherwise `cpu`; `all` requests 999.                                        |
| `-b N`, `-mh N`            | Batch (512) and micro-batch (`min(128, batch)`); micro-batch cannot exceed batch.                                   |
| `-f auto\|on\|off`         | Flash attention; default `auto`.                                                                                    |
| `-pr PATH`, `-ce NAME`     | Compatible vision projector and chat-template override.                                                             |
| `-pe`, `-n`                | Enable (default) / disable conservative text-prefix reuse.                                                          |
| `-k f16\|q8_0\|q4_0`       | KV cache; default `f16`; quantized types require `-f on`.                                                           |
| `-mk error\|mmap`          | Default `error`; opt-in CPU/pageable-weight fallback below.                                                         |
| `-e N`                     | Compact MoE slot budget in MiB; GPU when requested, CPU with `-g cpu`; 1..1048576; omitted = off. Not an RSS limit. |
| `-a PROFILE`, `-ct PRESET` | `all\|openai\|ollama\|llama\|lma\|anthropic` and printed client connection recipe.                                  |
| `-ht HOST`, `-pt PORT`     | Bind address; defaults `127.0.0.1` / 8080.                                                                          |
| `-ay KEY`                  | Bearer authentication for all protected API routes, including llama/native lma and discovery.                       |
| `-tt TARGET`, `-i`         | Launcher target and interactive setup.                                                                              |
| `-h`, `-d`                 | Help / JSON config preview; dry-run does not load weights or serve.                                                 |

Keep loopback binding for local use. Before binding to `0.0.0.0` or another non-loopback address, configure `-ay KEY` (NSS `--api-key KEY`) and external TLS/firewall protection. `/`, `/chat.js`, `/style.css`, and `/health` remain public.

### Speed, GPU fit, and memory

The model, inference context, and owned decode batch persist across requests. Text prefill and token generation clear/reuse that batch rather than allocate one per decode; the cache takes ownership of the tokenized prompt instead of copying each chunk. Chat/image request preparation runs on the blocking inference worker under the same single-request admission limit, rather than blocking the async executor or queuing unbounded image-decoding work.

Default `--prompt-cache` conservatively reuses the **exact common token prefix of text-only prompts**, including changed suffixes and shortened prompts, when native cache state permits. This is token-identical prefix reuse, not semantic-anchor caching or recurrent/hybrid-state rollback. Identical prompts re-evaluate the final token to refresh logits; extensions decode the uncached suffix after removing stale generation tokens. If native cache retention/removal cannot safely support reuse, lma does a full prefill. Failed/cancelled generation invalidates reusable state. Image requests reset state; there is **no image caching**. Send the desired chat history each time; this is not server-managed conversation history.

Quantized KV (`--kv-cache q8_0` or `q4_0`, with `--flash on`) is an opt-in memory/quality/performance tradeoff, not a guaranteed speedup. Keep models that fit on the GPU resident: the normal GPU path is unchanged, and prefix reuse does not force larger caches or CPU offload.

```sh
lma nss serve model.gguf --gpu all --flash on --kv-cache q8_0
lma nss serve model.gguf --gpu all --memory-fallback mmap
```

**Page-memory fallback is opt-in and may be extremely slow.** With `--memory-fallback mmap`, the requested GPU setup is tried first unchanged. If GPU availability, model, context, or projector initialization **returns an error** (not necessarily OOM), lma warns with the original error and retries **once** on CPU: memory-mapped weights, `use_mlock=false`, CPU KV/compute, and no projector GPU offload. A successful GPU initialization never triggers fallback. Explicit `--gpu cpu --memory-fallback mmap` starts in that mode and warns immediately.

Bad configuration, preflight model/projector-header validation failures, and backend initialization failures do not retry. Native aborts and OS OOM-killer terminations **cannot be rescued**; this is an initialization fallback, not recovery from later generation failures.

File-backed weights are demand-paged by the OS. Anonymous context/KV, compute, and projector allocations still need RAM or OS-configured swap; mmap does not guarantee successful allocation. lma does not change system swap settings, turn NVMe into VRAM, or promise usable speed when paging.

### Opt-in MoE expert cache

NSS `--expert-cache-mib N` (root `-e N`) enables a **persistent, bounded compact expert cache** in the vendored native runtime, not merely mmap or routing statistics. It is off by default; accepted explicit budgets are **1..1048576 MiB** (1 MiB = 1,048,576 bytes). Original GGUF expert weights remain memory-mapped on CPU. Routed misses in decode and bounded GPU prefill synchronously upload complete per-layer bundles into compact slots on the requested GPU, or copy into CPU slots with explicit `--gpu cpu`. Frequency-biased LRU eviction reuses slots. Native expert kernels consume compact tensors with remapped slot IDs; routing probabilities retain original expert IDs. No downloads or new on-disk weight copies are needed.

```sh
# Use an existing compatible text MoE GGUF; this does not fetch a model.
lma nss serve --path existing.gguf --expert-cache-mib 1024 --gpu cpu
# GPU-enabled build: explicitly bound local prefill to <=16 tokens.
lma nss serve --path existing.gguf --expert-cache-mib 1024 --gpu all --ubatch-size 16
```

The initial architecture allowlist is `qwen2moe`, `qwen3moe`, and `mellum`, with ordinary contiguous separate up/gate/down or fused gate_up/down expert weights and no expert biases or sidecar scales. Only **single-sequence causal text generation with one live context per model** is supported; embeddings, vision projectors, and adapters are unsupported. Requested batch/micro-batch settings are preserved. **Single-token decode uses resident slots; SILU prefill of 2..16 local tokens uses capacity-safe sequential GPU expert tiles**. Tile capacity is `min(16, n_slots / top_k)`, or 16 when the whole expert set fits. Larger (>16) and empty local views use original mmap CPU expert matmuls, without full expert-weight GPU copies. The choice is local to each expert operation: a final-layer view can have fewer tokens than the requested micro-batch. Explicitly set root `-mh 16` or NSS `--ubatch-size 16` to keep local prefill within the GPU bound; no default is changed. Explicit CPU caching is unchanged.

The budget covers actual compact expert **device slot allocations**, including alignment/padding (and Metal page rounding), **not process RSS**. Apple unified memory is shared with the CPU and other applications, not an additional independent VRAM pool. Source mmap/OS page cache, dense weights, KV, graph metadata/workspaces, and other overhead are outside the budget. Graph reservation includes worst-case 16-token stream expansion, so metadata and workspace allocations can increase. Source pages can remain resident alongside compact slots; there is no `DONTNEED` reclamation, predictive/asynchronous prefetch, or SSD-speed guarantee. Too little budget fails with the minimum required aligned byte count in native logs rather than silently increasing allocation. Startup logs show cache device, budget, and allocation; request logs report cumulative bundle hits, misses, evictions, and source bytes copied. These include bounded GPU prefill slot traffic and routing counts, but not >16-token CPU fallback operations or all prefill expert use. CPU routing/top-k preparation and demand uploads remain synchronous; GPU expert math is real, not asynchronous prefetch. See the [expert-cache guide](docs/expert-cache.md).

NSS `--memory-fallback mmap` remains a **separate CPU/pageable initialization retry**, not a switch for this cache. The routing-order fix completes original-ID routing weights before cache admission, preserving normal Metal fusion. Sequential Mellum2 Base Q4_K_M checks with GPU request 999 and a 1 GiB cache pass micro-batch-1 decode (exact raw text/usage/finish, unchanged logit bounds, not bitwise under normal fusion) and GPU history reload. CPU micro-batch-32 full-model logits pass exactly. With that 1 GiB cache, GPU micro-batch-32 short prompts pass unchanged tolerances (all local chunks <=16), and micro-batch-16 long prefill passes across multiple native micro-batches with exact text/usage/finish. The 16-token limit is local, not a total prompt-length limit. **At 2 GiB, normal-fusion micro-batch-16 testing fails short prompt B before the long case; a fusion-disabled diagnostic is bitwise exact throughout.** That fusion-related larger-cache drift is unresolved; thresholds are unchanged and production fusion remains enabled. The final 26 passing native CTests (22 GPU configurations plus four other tests, no skips/timeouts in 271.9 seconds) and the backend view-placement fix do not resolve it. Full-feature 1 GiB GPU micro-batch-16 long-prompt and history reruns pass after that fix, with `66,834,432` history bytes warmed. This is not evidence for 32-local-token GPU expert batches; >16 local tokens still fall back to CPU. See [current validation evidence](docs/expert-cache.md#latest-reported-real-model-evidence); these are scoped correctness checks, not universal GPU-prefill/offload parity, memory-fit, or throughput guarantees.

See [performance measurement and the explicit, model-reusing benchmark harness](docs/performance.md) for matched lma/Ollama comparisons. No speed-parity result is claimed.

[FreeToken's README](https://github.com/FlashML-org/FreeToken/blob/main/README.md) describes oversized-MoE serving with expert caching, CPU/GPU co-execution, semantic-anchor caching, elastic memory, and its custom FTW format. lma uses GGUF/llama.cpp with the narrower GPU/CPU expert-slot cache above, bounded GPU prefill, and >16-token mmap CPU fallback, not that architecture. It promises neither FreeToken performance nor a paging capability absent from FreeToken; that comparison is unverified.

## HTTP API

Choose `all|openai|ollama|llama|lma|anthropic` with root **`lma s MODEL -a PROFILE`** or **`lma nss serve MODEL --api PROFILE`**. Long flags stay NSS-only; there are no new root long aliases. Default `all` enables every family, including the existing Anthropic routes. `/health` and `/v1/models` are shared; `/api/stop` remains shared host control in every profile. Browser chat is available only with `all`/`openai`; other profiles return API information JSON at `/`.

Profiles wrap the **same in-process engine**, not separate servers or faster inference paths. They are supported subsets, **not full llama-server or cloud parity**; no performance claim is made. Existing OpenAI/Ollama contracts are unchanged. See the **[endpoint profiles and native API contract](docs/endpoints.md)** for exact fields, streaming, limits, and base URLs.

- **llama** base URL: `http://127.0.0.1:8080` (not `/v1`). Strict `/completion` supports raw `prompt`, `n_predict` (positive or `-1`, inherited `num_predict`/default 256), `temperature` (alias `temp`), `top_p`, `seed`, and `stream` (default false; true enables SSE). No slots, FIM, `cache_prompt`, or stop controls; unsupported fields are rejected. `/tokenize` uses `content` with `add_special:false` / `parse_special:false`; `/detokenize` uses `tokens:[i32]`. Both use actual model token IDs/vocabulary.
- **Native lma** base URL: `http://127.0.0.1:8080/lma/v1`. `/generate` takes raw `prompt`; `/chat` uses the existing OpenAI role/content/images schema, without tools or remote image fetching. Both accept optional `model`, `max_tokens`, `temperature`, `top_p`, `seed`, and `stream` (default false). Omitted `max_tokens` inherits stored `num_predict`/default 256; **explicit `null` means remaining context/no reply cap**; zero/negative values are rejected. Nonstream JSON contains `api_version:1`, `id`, `model`, `text`, `usage:{input_tokens,output_tokens}`, and `finish_reason:eos|token_limit|context_limit`. `stream:true` emits NDJSON `start`/`delta`/`done`/`error`; deltas carry text, `done` carries usage/finish without repeating aggregate text. New routes have 1 MiB bodies except native `/lma/v1/chat` (24 MiB).

Printed client presets require their API family to be enabled: an explicit incompatible profile prints a warning, not a usable preset. llama/native-lma URLs are endpoint hints, not OpenAI/Anthropic client compatibility claims. See [client setup](docs/clients.md).

| Endpoint                             | Purpose                                                                |
| ------------------------------------ | ---------------------------------------------------------------------- |
| `GET /`                              | Public browser chat with all/openai; API info JSON otherwise.          |
| `GET /health`                        | Ready status; public.                                                  |
| `GET /v1/models`                     | Installed models plus the explicit startup model.                      |
| `POST /v1/completions`               | String `prompt` → text; native SSE with `stream:true`.                 |
| `POST /v1/chat/completions`          | Text/image chat, function calls/results, native text SSE.              |
| `POST /v1/messages`                  | Anthropic-format messages, function calls/results, SSE.                |
| `POST /v1/messages/count_tokens`     | Served GGUF prompt count, including function definitions/history.      |
| `POST /v1/embeddings`                | Native embeddings, float/base64 output, exact token usage.             |
| `POST /v1/responses`                 | Stateless Responses, function calls/results, SSE.                      |
| `POST /api/chat`                     | Ollama-compatible chat/tools; NDJSON by default (`stream:true`).       |
| `POST /api/generate`                 | Ollama-compatible prompt/system/images or raw text; NDJSON by default. |
| `GET /api/tags`, `GET /api/ps`       | Installed catalog / currently resident model.                          |
| `POST /api/show`                     | Selected installed-model GGUF header information.                      |
| `GET /api/version`                   | Actual lma version, not an Ollama release number.                      |
| `POST /api/embed`, `/api/embeddings` | Native normalized vectors; batched or legacy single prompt.            |
| `GET /props`                         | llama-profile served-model/context properties.                         |
| `POST /completion`                   | llama-profile raw text completion; optional SSE.                       |
| `POST /tokenize`, `/detokenize`      | llama-profile actual model token IDs / vocabulary text.                |
| `GET /lma/v1/status`                 | Native lma API status.                                                 |
| `POST /lma/v1/generate`              | Native raw text generation; JSON or NDJSON.                            |
| `POST /lma/v1/chat`                  | Native text/image chat; JSON or NDJSON, no tools.                      |
| `POST /api/stop`                     | Shared managed-host unload control, in every profile.                  |

```sh
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":64}'
curl http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":false,"options":{"num_predict":64}}'
```

With `-ay KEY` (NSS `--api-key KEY`), add `-H 'Authorization: Bearer YOUR_KEY'` to **all protected API routes**, including `/v1/*`, `/api/*`, llama routes, native `/lma/v1/*`, discovery, and shared `/api/stop`. Only Anthropic Messages/count routes also accept `x-api-key`; `/health` stays public. Route availability follows `-a` (NSS `--api`): `/v1/models` and public `/health` remain shared; Ollama `/api/*` routes require `all` or `ollama` **except shared `/api/stop`**. The CLI HTTP host lists installed models and supports single-resident model selection, keep-alive/unload, public GGUF pull, alias copy/create with stored text definitions, and source-specific deletion, including Ollama and LM Studio. `lma sp` / `lma nss stop` unload from this host. `/api/create` supports the [bounded Modelfile subset](docs/modelfile.md), but HTTP `FROM` accepts installed selectors only, never arbitrary filesystem paths. Push, blob uploads, custom Go templates, adapters and training are not implemented. See [the API contract](docs/ollama-api.md).

HTTP generation selects only installed models (or the explicit startup model's ID); unknown model names and filesystem paths never cause implicit downloads or arbitrary file loads. Streaming host leases persist until the response body is done, preventing unsafe model switches/unloads during a stream.

OpenAI chat accepts `system`, `developer` (mapped to system), `user`, and `assistant` roles, plus matched `tool` results. **A missing chat template no longer blocks chat.** The default root `-ce auto` (NSS `--chat-template auto`) prefers a usable embedded GGUF template, otherwise detects Llama 3, Gemma/Gemma4, ChatML/Phi 4, or Phi 3 signatures using exact special/control tokens, and finally falls back to plain `System:`, `User:`, `Assistant:` conversation formatting. Formatting is shared by desktop, TUI, hosted APIs, and token counting; raw completions remain unchanged. A warning is logged once per model when compatibility formatting is used.

Override with root `-ce plain` (NSS `--chat-template plain`), `native` (strict embedded-template-only), the lma `gemma4` unified formatter, or a pinned llama.cpp built-in name such as `chatml`, `llama3`, `gemma`, `mistral-v7`, or `gpt-oss`. For example: `lma c -o MODEL -ce plain`. Explicit choices fail visibly rather than silently switching formats. This removes a metadata limitation, **not model training or architecture limitations**: base/edit models may echo prompts, continue conversations, or ignore instructions/tools; incompatible weight formats are still unsupported.

### OpenAI-style request fields

Raw Completions and Chat Completions share these options. Sampling/output defaults below apply when no selected Modelfile definition supplies them; explicit request values win:

| Field         | Behavior                                                               |
| ------------- | ---------------------------------------------------------------------- |
| `model`       | Optional; CLI HTTP host selects an installed model, otherwise current. |
| `max_tokens`  | Positive output limit; default 256.                                    |
| `temperature` | 0–2; default 0.8; zero selects greedy sampling.                        |
| `top_p`       | Greater than 0 and at most 1; default 0.95.                            |
| `seed`        | Optional unsigned 32-bit integer.                                      |
| `stream`      | Default false; true enables native SSE for Chat and raw Completions.   |

These endpoints return `choices`, `usage`, and `finish_reason`. Chat also accepts `tools`, `tool_choice`, `parallel_tool_calls`, assistant `tool_calls`, matched `role:tool` results, and `stream_options.include_usage`. `max_completion_tokens` is an alternative to `max_tokens`, not an additional budget. Unsupported fields such as stop sequences and penalties are rejected. Prompt tokens (including injected tool definitions and chat/image formatting) plus the **requested** output limit must fit the configured context.

Messages uses `tool_use` / `tool_result` content blocks; Responses uses `function_call` / `function_call_output` items. Include full conversation history on every request: no server-side response storage or `previous_response_id`. The shared tool bridge constrains output with native JSON-schema grammar, validates calls/arguments against supplied definitions, and rejects truncated or invalid output without issuing a call. Limits include 32 tools, 8 calls/turn, and a bounded schema subset—not arbitrary JSON Schema. Tools are executed by clients only.

Text SSE forwards actual native deltas. Raw Completions ends with a finish/usage chunk and `data: [DONE]`; Chat sends a separate usage chunk only with `stream_options.include_usage:true`. With tools enabled, lifecycle events start early, but the structured envelope (including any answer text) is buffered until validated; `x-lma-streaming` distinguishes `native` from `validated-tool-buffer`. Provider cache/reasoning/verbosity hints are advisory, not implemented provider services; accepted hints are disclosed through `x-lma-advisory`. See `lma nss help serve advanced` and [client documentation](docs/clients.md).

### Images and Ollama-compatible generation

Images require a vision-enabled build and a **compatible GGUF model plus projector**; an arbitrary text model does not gain vision by attaching a projector. Only **PNG/JPEG** are accepted. HTTP(S), remote, and `file:` image URLs are never fetched.

For `/v1/chat/completions`, each user message may have string `content` plus `"images":["RAW_BASE64"]`, or `content` parts such as:

```json
[
  { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } },
  { "type": "text", "text": "Describe this image." }
]
```

`data:image/jpeg;base64,...` also works. Image parts must precede all nonempty text parts; interleaved images after text are **rejected, not reordered**. The `images` attachments precede content parts. Only `user` messages may contain images.

`/api/chat` accepts stateless `messages` (string `content`, optional raw-base64 user `images`) and client-executed function tools with object-valued arguments. `/api/generate` accepts `prompt`, optional `system`/`images`, and `raw:false` (chat-template formatting) or `raw:true` (exact text, no system/images). In the CLI HTTP host, `model` can select another installed model; omission uses the current model. In-process desktop/TUI sessions remain explicitly selected.

**Both Ollama generation routes default to `stream:true`**, returning newline-delimited JSON (`application/x-ndjson`), not SSE. Send explicit `stream:false` for one JSON response; this is required for clients relying on the older lma default. Records carry `model`, UTC `created_at`, `message` (chat) or `response` (generate), and `done`; the successful terminal record adds `done_reason`, real token counts, and measured `total_duration`. Unavailable phase timings are omitted. Ollama handler/auth errors use `{"error":"message"}`; errors after streaming headers arrive as NDJSON error records without a successful terminal record. There is no `[DONE]` sentinel.

Supported `options` are `num_predict`, `temperature`, `top_p`, `seed`, and `num_ctx` only when equal to the hosted context. Sampling/output defaults match the `/v1` fields above unless supplied by the selected definition; explicit request options win. Non-raw requests also apply its default system prompt and initial messages; explicit system instructions override `SYSTEM`, and `system:""` clears it in `/api/generate`. The HTTP host accepts `keep_alive` seconds/duration strings, `0` for unloading after completion, or `-1` for indefinite residency. It switches installed models by request name; active response leases prevent unsafe unloads. In-process desktop/TUI sessions remain explicitly selected. JSON/schema `format`, separate thinking output/control, unknown fields, and other Ollama options are rejected rather than silently applied (`think:false` only requests no separate thinking channel). Tool output is constrained and buffered until validation. See **[the Ollama API contract](docs/ollama-api.md)** for wire examples, discovery omissions, lifecycle, tools, and errors. This is a compatibility subset, not a full Ollama daemon or universal dense/MoE model-support claim.

With a compatible vision server running, encode a local file on the client and send it without a giant inline payload:

```sh
python3 - <<'PY' | curl http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' --data-binary @-
import base64, json
from pathlib import Path
image = base64.b64encode(Path("photo.png").read_bytes()).decode("ascii")
print(json.dumps({"messages": [{"role": "user", "content": "Describe this image.",
                               "images": [image]}], "stream": False,
                  "options": {"num_predict": 128}}))
PY
```

Limits per request: **8 images**, **8 MiB decoded per image**, **16 MiB decoded total**; **24 MiB** bodies for `/v1/chat/completions`, `/api/chat`, and `/api/generate`. Ollama combined message text/history is limited to **1 MiB**. `/v1/completions` retains its **1 MiB** body limit.

Messages and Responses accept user PNG/JPEG base64/data-URL image inputs with the same image ordering and vision requirements, but have **1 MiB** bodies. Combining images with tool history has not been verified with a real model; token counting rejects images. No endpoint fetches remote image URLs.

Only **one model operation runs at a time**; overlapping requests receive HTTP **429**. Disconnecting an SSE client cancels native generation, with bounded queues and admission retained until cleanup. Native embeddings share that worker through `/v1/embeddings`, `/api/embed`, and `/api/embeddings`; vectors are normalized and usage counts evaluated model tokens, not generated text. Automatic downloads during serving, LoRA adapters, parallel slots, hosted cloud tools, Responses WebSockets, compressed request bodies, and arbitrary upstream flags remain unsupported. GGUF downloads are explicit through `lma f` or `/api/pull`; image generation remains CLI-only. Performance depends on model, build, hardware, and settings; no benchmark claim is made.

## Fetch GGUF models

`lma f QUERY [OPTIONS]` (NSS `lma nss fetch`) combines **Ollama's registry and public Hugging Face GGUF repositories by default**, with source-labelled results. The latter is the LM Studio-compatible download source, not a separate proprietary LM Studio registry. Combining catalogs **does not merge model weights**.

```sh
lma f llama3.2 -l
lma f llama3.2 -oe
lma f lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF -le
lma f owner/repository -le -f model.Q4_K_M.gguf -o models
lma f llama3.2 -d                        # No network or writes
```

The table and long-option discussion in this section apply only to **`lma nss fetch`**. Root `lma f` uses `-l` (list), `-f NAME` (file), `-o DIR` (output), `-oe` (Ollama database), `-le` (LM Studio-compatible database), `-h` (help), and `-d` (dry-run).

| NSS option                    | Behavior                                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `--ollama-database`           | Only Ollama; exact `name[:tag]` or `namespace/name[:tag]` lookup (default tag `latest`).                   |
| `--lm-studio-database`        | Only public Hugging Face GGUF repositories; searches names, requires exact `owner/repository` to download. |
| Both source flags, or neither | Query both sources.                                                                                        |
| `--list`                      | Discover/list candidates; network access, but no weight downloads.                                         |
| `--file NAME`                 | Exact Hugging Face GGUF filename; otherwise prefer a unique Q4_K_M file, then a unique standalone GGUF.    |
| `--output`, `--out`, `-o DIR` | Download root; default `~/.lma/models/`. Use `--out .` explicitly for the current directory.               |
| `--dry-run`                   | Config preview only; no network or writes.                                                                 |

The default download root is the absolute home-based `~/.lma/models/`, independent of the working directory. Missing or relative `HOME` (`USERPROFILE` first on Windows) is an error unless `--out DIR` is explicit. `--out .` deliberately opts into the current directory; no implicit current-directory fallback occurs.

Downloading requires one unambiguous exact source match; fuzzy search results are never downloaded automatically. A failed exact lookup blocks implicit selection of another source—retry or restrict the source. Ollama projectors are downloaded with the model; Hugging Face projectors require separate selection. The command prints local paths and a suggested serve command, but does not start a server or install into Ollama/LM Studio stores.

Downloads stream to temporary sibling files and are checked against upstream SHA256/size metadata and GGUF magic before publication. Existing verified files are reused; mismatches are refused. These checks do **not** guarantee architecture compatibility or validate the full tensor structure. Use a trusted output directory.

**GGUF only:** no diffusion/image-generation weights, private/gated repositories, split GGUFs, adapters, or resume. Downloads can be many GB and contact Ollama/Hugging Face and their HTTPS storage redirects. No daemon or subprocess is needed.

## Local speech and dictation

```sh
lma t -m models/whisper-tiny recording.wav
lma t -m models/whisper-tiny recording.wav -l en -j
```

Root transcription flags are `-m MODEL`, `-a AUDIO`, `-l LANGUAGE`, `-t THREADS`, `-g GPU`, and `-j` JSON, plus `-h` / `-d`; audio may also be positional. Long flags belong only to `lma nss transcribe`.

The default-enabled `speech` feature runs Candle Whisper in process. Supply a local Hugging Face Whisper directory with `config.json`, `tokenizer.json`, and `model.safetensors`—not a whisper.cpp `.bin` or chat GGUF. No automatic downloads. WAV input supports PCM16/float32 mono/stereo, 8–192 kHz, up to 30 minutes/512 MiB. Use root `-g off` (NSS `--gpu off`) for CPU; explicit GPU requests fail instead of silently falling back.

Desktop dictation uses the microphone button or **Ctrl/Cmd+Shift+D**. Select an installed compatible speech model, or enter a Hugging Face repository, inspect its files/revision/size, then explicitly install it. There is no fixed speech-model download preset. The app saves the selection, not recordings. Audio stays local, is capped at 120 seconds, and transcription is appended to the draft without sending. Microphone permission is required. **Real speech transcription and native microphone capture are not yet end-to-end verified.** See [model support](docs/model-support.md) for limits and build requirements.

## Local image generation

`lma i MODEL_DIR "PROMPT" [OPTIONS]` (NSS `lma nss image`, with NSS-only aliases `imagen`, `image-gen`, and `img`) generates a PNG in-process with the optional, **default-enabled `image-gen` Candle backend**. It accepts only a **local standard Stable Diffusion 1.5 Diffusers directory with safetensors weights**, not a chat model. This is **CLI only**: no image-generation HTTP route or browser UI, and no automatic downloads (including via `lma f`).

Required files beneath `MODEL_DIR`:

```text
model_index.json
text_encoder/config.json
text_encoder/model.safetensors
unet/config.json
unet/diffusion_pytorch_model.safetensors
vae/config.json
vae/diffusion_pytorch_model.safetensors
scheduler/scheduler_config.json
tokenizer/tokenizer.json
```

**Tokenizer caveat:** original Diffusers SD1.5 exports often lack `tokenizer.json`; vocabulary/merges files alone are insufficient. Supply a separately obtained local **`openai/clip-vit-large-patch14` tokenizer JSON** via root `-t FILE` (NSS `--tokenizer FILE`) if needed, not an LLM tokenizer. Architecture/scheduler configs are validated; SD1.5 PNDM or DDIM configs are accepted, but inference always uses DDIM (`eta=0`). Each positive/negative prompt is limited to **75 CLIP content tokens** (77 with BOS/EOS); overlong prompts are rejected, not silently truncated.

```sh
lma i models/sd15 "a watercolor fox" -o fox.png
lma i -m models/sd15 -p "a tiny cabin" -n "blurry" \
  -t models/clip-tokenizer.json -de metal -s 30 -o cabin.png
lma i models/sd15 "a red fox" -d          # Config only; no asset reads or generation
```

The table and long-option discussion below use **`lma nss image`** only. Root `lma i` flags are `-m` model, `-p` prompt, `-n` negative prompt, `-o` output, `-t` tokenizer, `-w` width, `-ht` height, `-s` steps, `-sd` seed, `-g` guidance, `-de` device, and `-a` attention slicing (currently rejected by the backend), plus `-h` / `-d`.

| NSS option                         | Meaning / default                                                                             |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `--model-dir`, `--model`, `-m DIR` | Local SD1.5 directory; alternatively first positional argument.                               |
| `--prompt`, `-p TEXT`              | Required prompt; alternatively second positional argument.                                    |
| `--negative-prompt`, `-n TEXT`     | Default empty.                                                                                |
| `--out`, `--output`, `-o FILE`     | New `.png` file; default `image.png`; never overwrites.                                       |
| `--tokenizer FILE`                 | Override local `tokenizer/tokenizer.json`.                                                    |
| `--width N`, `--height N`          | Each 64–1024, divisible by 8; default 512 × 512.                                              |
| `--steps N`                        | 1–150 DDIM steps; default 20.                                                                 |
| `--seed N`                         | Unsigned 64-bit initial-noise seed; default 42.                                               |
| `--guidance N`                     | CFG scale 1–20; default 7.5.                                                                  |
| `--device auto\|cpu\|metal\|cuda`  | Default `auto`; GPU devices need matching build features.                                     |
| `--attention-slicing N`            | Unavailable: Candle 0.11 tensor-rank bug; any value is rejected before weights load. Omit it. |
| `--dry-run`                        | Preview configuration without reading/loading assets or writing output.                       |

The output parent must already exist, outside the model tree; output cannot intersect any input asset. A temporary PNG is published without replacing an existing destination. Keep memory-mapped safetensors **immutable throughout generation**, including against other processes. CPU generation is slow; seeds reproduce initial noise, not guaranteed bit-identical GPU results.

Safetensors preflight rejects invalid/LFS-pointer files and incompatible key shapes before GPU allocation. With `--device auto`, an accelerator inference failure warns before a single F32 CPU retry; explicit `metal`/`cuda` does not silently switch devices.

Unsupported: Google Imagen, GGUF diffusion models, SDXL, Flux, single-file checkpoints (including pickle checkpoints), and nonstandard SD1.5 architectures. Supply the complete compatible local asset set yourself. Component tests do not prove end-to-end generation: a real image test still requires compatible local weights; no multi-gigabyte downloads are performed automatically.

## Quantize

`lma q INPUT [OUTPUT] [OPTIONS]` quantizes in-process. Without an output path, `models/model.gguf` becomes `models/model.Q4_K_M.gguf` (or the selected type).

```sh
lma q model.gguf
lma q model.gguf smaller.gguf -t Q8_0 -ts 8
lma q model.gguf -o smaller.gguf -t q4_k_m -l
lma q model.gguf -d
```

The table and long-option discussion below use **`lma nss quantize`** only; its legacy `-q` (type) and `-t` (threads) differ from root. Root `lma q` uses `-t TYPE`, `-ts THREADS`, `-o FILE`, `-p` pure, `-l` leave output tensor, `-a` allow requantize, `-f` force, and `-h` / `-d`.

| NSS option                | Meaning / default                                      |
| ------------------------- | ------------------------------------------------------ |
| `--type`, `-q TYPE`       | Default `Q4_K_M`; normalized to uppercase.             |
| `--out`, `-o FILE`        | Output path instead of positional `OUTPUT`.            |
| `--threads`, `-t auto\|N` | Default `auto` uses available logical CPUs.            |
| `--pure`                  | Do not mix quantization types between tensors.         |
| `--leave-output`          | Leave the output tensor unquantized.                   |
| `--allow-requantize`      | Permit already-quantized input; quality may degrade.   |
| `--force`                 | Permit replacing existing output, **never the input**. |
| `--dry-run`               | Print JSON without reading or writing models.          |

Supported types: `F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K`.

Existing output is refused unless `--force` is supplied. A temporary sibling file is published only after success; use a **trusted output directory**. On non-Unix systems, forced replacement is unsupported. Safeguards also apply in NSS mode, including after its quantize `--` separator. Root bare `--` is rejected. Importance matrices (`imatrix`), split outputs, and tensor overrides are unsupported. Quantization trades quality for memory; not every type suits every model.

## No short syntax (`nss`)

NSS is the **only namespace for long commands, long flags, and legacy aliases**. Root shortcuts `h`, `l`, `c`, `ce`, `d`, `t`, `s`, `sp`, `q`, `f`, `i`, and `it` are errors after `lma nss`. Use `help`, `list`, `launch` / `lh` (the legacy NSS exception), `chat`, `transcribe`, `serve` / `server`, `stop`, `quantize` / `quant`, `fetch`, `import`, `create`, `delete`, or `image` / `imagen` / `image-gen` / `img`. NSS retains legacy option shortcuts such as `-m`, `-c`, `-t`, `-q`, `-p`, `-o`, and `-h`, plus positional models and friendly options like `--gpu all`. Those shortcuts are not necessarily the root flag table: for example, quantization uses root `-t TYPE -ts THREADS`, versus NSS `-q TYPE -t THREADS`. Both namespaces use the same embedded engine and safety checks, not upstream passthrough.

```sh
lma nss serve --path model.gguf -c 8k --gpu cpu
lma nss quantize in.gguf out.gguf Q8_0 8
lma nss --dry-run serve --model llama3.2
lma nss serve --path model.gguf --dry-run
lma nss fetch llama3.2 --ollama-database --list
lma nss img -m models/sd15 -p "a fox" -o fox.png --dry-run
lma nss serve model.gguf -- --ctx-size 8192 --n-gpu-layers 0
```

Serve long-form options: `--path`, `--model`, `--ollama`, `--ollama-dir`, `--projector` / `--mmproj`, `--n-gpu-layers`, `--ctx-size`, `--threads`, `--batch-size`, `--ubatch-size`, `--flash-attn`, `--prompt-cache`, `--no-prompt-cache`, `--kv-cache`, `--memory-fallback`, `--expert-cache-mib`, `--host`, `--port`, `--api-key`.

Quantize long-form options: `--type`, `--out`, `--threads`, `--pure`, `--leave-output-tensor`, `--allow-requantize`, `--force`. NSS also accepts `INPUT OUTPUT TYPE [THREADS]` as above.

Inside NSS, commands retain their legacy option shortcuts and accept `--dry-run`, `--help`, and `--help-all`. Fetch and image use the NSS options documented above. `--dry-run` works before or after the NSS command and redacts the API key. NSS serve/quantize support a `--` separator for implemented long-form syntax; it **does not forward arbitrary arguments**. Root bare `--` cannot enter NSS. Unknown options are errors.

## Help and migration

```sh
lma -h                       # Root quick start; also: lma h
lma -v                       # Version (NSS: lma nss --version)
lma h s                      # Hosting options; also: lma s -h
lma h sp                     # Stop/unload options; also: lma sp -h
lma nss help serve advanced   # HTTP API guide (root: lma h s -a)
lma h q -a                   # Quantization help and types
lma h l                      # Installed model discovery
lma h lh                     # Interactive model picker and terminal chat
lma h ce                     # Local Modelfile definitions
lma h d                      # Installed-model deletion and confirmation rules
lma h it                     # Explicit local safetensors import
lma h f                      # Combined catalogs and download selection
lma h i                      # Local SD1.5 assets and generation options
lma h t                      # Speech transcription (NSS: lma nss help transcribe)
lma h nss                    # Long syntax and legacy alias rules
```

### Terminal color and machine output

Help, version, main CLI errors, and human-readable model listings use restrained color only when their own output stream is a terminal. A **nonempty** `NO_COLOR` (including `NO_COLOR=0`) or `TERM=dumb` disables color; an empty `NO_COLOR` does not. Stdout and stderr are detected independently, so redirecting one does not change the other's behavior. Pipes and redirected output never receive styling ANSI codes; JSON and dry-run plans bypass styling entirely. Names, paths, and diagnostic text escape control/directional-formatting characters before terminal display; JSON retains the original data using JSON escaping.

```sh
NO_COLOR=1 lma h             # Plain help even on a terminal
TERM=dumb lma l              # No list colors
lma l > models.txt           # Stable plain-text table, not the compact TTY layout
lma l -j > models.json       # Full machine-readable catalog, never ANSI styled
```

Help is local, never upstream help. The old executable wrapper's full feature parity is not retained: `--backend` and `--backend-help` are rejected; `LMA_SERVER` and `LMA_QUANTIZE` are ignored. Remove executable-selection settings and unsupported native flags. NSS does not bypass output safeguards.

## Development

`llama-cpp-2` and `llama-cpp-sys-2` are pinned to **0.1.160** and locally patched through both root and desktop Cargo manifests for the native expert-cache ABI extension; see [vendor provenance and update notes](vendor/README.md). Candle core/transformers are pinned to **0.11.0**, with Axum/Tokio for HTTP and Serde for configuration/request data. Native build prerequisites apply to Rust tests too. UI tests use Node's built-in test runner, with no npm dependencies. Use `ghx` for source-control commands, including the explicit updater's clean-check/fast-forward flow.

```sh
cargo check --locked --no-default-features --all-targets
cargo test --locked --no-default-features
cargo check --locked --manifest-path desktop/Cargo.toml --all-targets
node --test tests/chat_ui.mjs tests/desktop_ui.mjs
python3 -B -m unittest discover -s install -p 'test_*.py' -v
# Optional broader local validation (default vision/image-gen features):
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

For a bounded **native macOS launch check**, build the debug app into the shared target directory and run the smoke test:

```sh
cargo build --locked --manifest-path desktop/Cargo.toml --target-dir target --bin lma-desktop -j 2
cargo test --locked --manifest-path desktop/Cargo.toml --target-dir target --bin lma-desktop -j 2
python3 -B tests/desktop_launch.py
python3 -B tests/desktop_launch.py --exercise # Two launches: real WebKit UI + local save/restore
```

The launch test briefly opens the actual app, checks for its on-screen WindowServer window, frontend-to-Rust IPC readiness, and completed model discovery, then terminates its test process. It uses the debug-only `LMA_DESKTOP_TEST_DIR` override to isolate app data/preferences, so it cannot auto-load a user's remembered model. It needs a logged-in macOS graphical session and Swift, but no Accessibility or Screen Recording permission. Logs go to `target/desktop-launch.log`. The optional `--exercise` mode checks actual WebKit navigation, model search, safe code/HTML rendering, normal/narrow-window geometry, and conversation/draft/settings persistence across two native launches. Debug-only probe commands are disabled unless explicitly opted into the isolated test mode, and excluded from release builds. This is not pixel inspection or a real-model test; it does not install, update, or download anything.

[Cross-platform CI](.github/workflows/ci.yml) defines no-default core checks/tests and a locked desktop check (including test targets) on `macos-latest`, `windows-latest`, and `ubuntu-24.04`. Linux/Windows use CPU; macOS desktop enables Metal and Apple Silicon core retains implicit Metal. Root and desktop share a cached Cargo target directory, keyed on both lockfiles. CI also runs Node UI tests on all three platforms and isolated Python installer tests on Linux/macOS; the POSIX fake-tool fixtures simulate Windows paths/replacement failures rather than performing a native Windows installation.

This matrix does **not** exercise CUDA, real-model inference, GPU memory fit/performance, default-feature core runtime tests, native webview interaction, desktop test execution, release packaging/signing, or a live source update. A desktop `cargo check` is not a linked release build or an app-launch test. Workflow definitions are intended coverage, not evidence that CI has already passed.

Implementation references: [shared Rust library](src/lib.rs), [in-process desktop session](src/desktop.rs), [desktop host](desktop/src/main.rs), [source installer](install/INSTALL.md), [desktop UI](desktop/ui/README.md), [desktop storage](desktop/src/workspace.rs), [browser UI](src/runtime/ui/chat.js), [public web assets](src/runtime/web.rs), [fetch](src/fetch/mod.rs), [image generation](src/runtime/imagegen.rs), [model resolution](src/models.rs), [dry-run dispatch](src/runtime/mod.rs), [inference/cache/fallback](src/runtime/inference.rs), [vision](src/runtime/vision.rs), and [HTTP schemas/limits](src/runtime/server.rs). CLI/configuration lives in `src/cli/` and `src/config.rs`; quantization safeguards in `src/runtime/quantize.rs`.

Model-free tests cover parsing, fixtures, policies, mocked HTTP inference, browser behavior, and image-generation validation/output safeguards; they are not a substitute for real GGUF/projector, SD1.5, and GPU checks on your hardware. These examples are usage instructions, not a claim that real-model performance, paging recovery, or GPU behavior was validated.
