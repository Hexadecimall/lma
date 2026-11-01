use super::{Action, Command, Syntax, args::Args, command_named, nss_command_named};

pub(super) fn parse(mut args: Args, syntax: Syntax) -> Result<Action, String> {
    let Some(topic) = args.next() else {
        return Ok(Action::Help(ROOT));
    };
    if topic == "nss" && args.is_empty() {
        return Ok(Action::Help(NSS));
    }
    if (topic == "-h"
        || (syntax == Syntax::Nss && matches!(topic.to_str(), Some("--help" | "--help-all"))))
        && args.is_empty()
    {
        return Ok(Action::Help(ROOT));
    }
    let command = match syntax {
        Syntax::Friendly => command_named(&topic),
        Syntax::Nss => nss_command_named(&topic).or_else(|| command_named(&topic)),
    }.ok_or("unknown help topic; root help uses short topics (`lma h s`); long topics belong in `lma nss help COMMAND`")?;
    let detail = args.next();
    if !args.is_empty() {
        return Err("help accepts COMMAND and optional advanced or --all".into());
    }
    match detail.as_deref().and_then(|s| s.to_str()) {
        None if detail.is_none() => {}
        Some("--all") if syntax == Syntax::Nss => {}
        Some("-a") if syntax == Syntax::Friendly => {
            return Ok(Action::Help(match command {
                Command::Serve => API,
                Command::Stop => STOP,
                Command::Launch => LAUNCH,
                Command::List => LIST,
                Command::Chat => CHAT,
                Command::Quantize => QUANTIZE,
                Command::Fetch => FETCH,
                Command::Image => IMAGE,
                Command::Import => IMPORT,
                Command::Create => CREATE,
                Command::Delete => super::delete::HELP,
                Command::Transcribe => super::transcribe::HELP,
            }));
        }
        Some("advanced") if syntax == Syntax::Nss => {
            return Ok(Action::Help(match command {
                Command::Serve => API,
                Command::Stop => STOP,
                Command::Launch => LAUNCH,
                Command::List => LIST,
                Command::Chat => CHAT,
                Command::Quantize => QUANTIZE,
                Command::Fetch => FETCH,
                Command::Image => IMAGE,
                Command::Import => IMPORT,
                Command::Create => CREATE,
                Command::Delete => super::delete::HELP,
                Command::Transcribe => super::transcribe::HELP,
            }));
        }
        _ => {
            return Err("use `lma h SHORT [-a]` or `lma nss help COMMAND [advanced|--all]`".into());
        }
    }
    Ok(Action::Help(match command {
        Command::Serve => SERVE,
        Command::Stop => STOP,
        Command::Launch => LAUNCH,
        Command::List => LIST,
        Command::Chat => CHAT,
        Command::Quantize => QUANTIZE,
        Command::Fetch => FETCH,
        Command::Image => IMAGE,
        Command::Import => IMPORT,
        Command::Create => CREATE,
        Command::Delete => super::delete::HELP,
        Command::Transcribe => super::transcribe::HELP,
    }))
}

pub(super) const ROOT: &str = concat!(
    "lma ",
    env!("CARGO_PKG_VERSION"),
    " — local models, simple commands.\n\n",
    "\
USAGE  lma COMMAND [OPTIONS]

COMMANDS
├─ h [COMMAND]           Help                 / nss help [COMMAND]
├─ l [FILTER]            Installed models     / nss list [FILTER]
├─ lh [NAME]             Pick model + target  / nss launch [NAME]
├─ c [NAME]              Terminal chat        / nss chat [NAME]
├─ t [OPTIONS] WAV       Whisper transcription / nss transcribe
├─ s [NAME]              Serve browser chat and HTTP APIs / nss serve [NAME]
├─ sp [-m NAME]          Stop/unload a model  / nss stop
│  ├─ s -p FILE          Select an explicit local GGUF
│  └─ s -ct PRESET       Print connection settings; -a selects API routes
├─ q INPUT [OUTPUT]      Quantize GGUF        / nss quantize
├─ f [QUERY]             Browse/download GGUF / nss fetch
├─ i DIR PROMPT          Local SD1.5 image    / nss image (alias: imagen)
├─ it                    Import supported safetensors / nss import
├─ ce NAME               Create a Modelfile profile / nss create NAME
├─ d NAME                Delete an installed model / nss delete NAME
└─ nss COMMAND           Long commands, long flags, and legacy aliases

EXAMPLES
  lma h                       / lma nss help
  lma l                       / lma nss list
  lma lh -tt codex            / lma nss launch --target codex
  lma c llama3.2              / lma nss chat llama3.2
  lma c llama3.2 -ts off -c 8k # No reply cap; context remains bounded
  lma t -m whisper audio.wav  / lma nss transcribe -m whisper audio.wav
  lma s                       / lma nss serve (idle; models chosen by requests)
  lma s llama3.2              / lma nss serve llama3.2
  lma sp                     / lma nss stop
  lma s -p model.gguf -ct claude-code
  lma ce my-local -f ./Modelfile -d
  lma f -l -i 50              / lma nss fetch --list --index 50

Root commands and flags use the first letter; collisions use first + last.
No root aliases or long flags. Use `lma COMMAND -h`, `lma h s -a` for HTTP
requests, or `lma nss help serve advanced`. Bare -- cannot switch syntax.
Launch targets start clients; -ct presets only print settings.
Downloads are explicit: `f`, API pull, or inspected desktop model installation.
~/.lma/ holds lma-owned settings, caches and models; external weights stay in place.
Color is terminal-only; NO_COLOR (nonempty) or TERM=dumb disables it. JSON stays plain.
"
);

pub(super) const CREATE: &str = r#"lma ce — register a named local Modelfile definition.

USAGE
  lma ce NAME [-f PATH] [OPTIONS]
  lma nss create NAME [--file PATH] [OPTIONS]

OPTIONS
  -f PATH       Modelfile to read (default: ./Modelfile in the current directory)
  -o DIR        Explicit Ollama store for resolving an installed FROM model
  -d            Parse and resolve FROM, then preview without registry writes,
                       downloads, weight loading, inference, or client execution
  -h            This help (long help flags are NSS-only)

Only an explicit create command reads the default current-directory Modelfile.
Other commands do not automatically discover or execute a nearby Modelfile.
NAME must be a model name, not a filesystem path or ollama: reference.
Only one NAME and one -f may be supplied. Unknown options are errors.
NSS accepts the same option shortcuts, including -f; use the long command create.

SUPPORTED TEXT-ONLY SUBSET
  FROM MODEL           Required installed model name or existing local model file
                       Relative files resolve from the Modelfile's directory
  SYSTEM TEXT          Default system prompt
  MESSAGE ROLE TEXT    Initial system, user, or assistant message
  LICENSE TEXT         Stored license metadata; not model training or enforcement
  TEMPLATE NAME        lma template selection: auto, native, plain, or a supported
                       built-in such as llama3, chatml, gemma4; see lma h s
  PARAMETER KEY VALUE  temperature, top_p, seed, num_predict, num_ctx, num_gpu,
                       num_thread, num_batch, u_batch

Quoted text and literal triple-quoted multiline text are supported; definitions
are bounded to 1 MiB. Custom Ollama Go TEMPLATE programs are unsupported and
rejected, not silently accepted. Unknown directives/parameters, adapters,
training, shell commands, and automatic model downloads are unsupported.
CLI load options override stored num_ctx/num_gpu/num_thread/num_batch/u_batch
and TEMPLATE defaults only when explicitly supplied. Serving does not scan CWD.

EXAMPLE MODELFILE CONTENT (save as Modelfile)
  FROM ./base.gguf
  SYSTEM "Be concise."
  TEMPLATE llama3
  PARAMETER temperature 0.7
  PARAMETER num_ctx 4096

EXAMPLES
  lma ce my-local -f ./Modelfile -d
  lma nss create my-local --file ./recipes/Modelfile --dry-run
  lma ce my-local

A dry run still validates the selected file and resolves its FROM; it is not
merely an argument dump. Real creation publishes a local registry definition;
it does not copy, quantize, or modify the source weights. The registry defaults
to ~/.lma/models; an alias to external weights does not transfer their ownership.
"#;

pub(super) const STOP: &str = "\
lma sp — stop/unload a model on the local managed host.

USAGE
  lma sp [OPTIONS]
  lma nss stop [OPTIONS]
  lma sp -m llama3.2 -p 8080

OPTIONS
  -m NAME               Optional model to stop; omit for the current model
  -ht HOST              Managed host (default: 127.0.0.1)
  -p PORT               Port (default: 8080; 1..65535)
  -a KEY                API key, redacted in previews
  -d                    Dry run; no HTTP request or model changes
  -h                    Show this help
  NSS-only long flags: --model, --host, --port, --api-key, --dry-run, --help

Requests POST /api/stop from an already running local managed host.
This unloads inference weights; it does not kill arbitrary OS processes or
start a server. Use `lma s NAME` or `lma s -p FILE` for hosting.
";

pub(super) const SERVE: &str = "\
lma s — local model server, browser chat, and HTTP APIs.

With -a all/openai, open http://127.0.0.1:8080/ (or your configured port).
Browser chat: Ctrl/Cmd+Enter sends; keys stay in memory, not browser storage.
Ollama/llama/lma/Anthropic-only profiles show API info JSON at /, not browser chat.

USAGE
  lma s [NAME] [OPTIONS]
  lma s -m NAME [OPTIONS]
  lma s -p FILE [OPTIONS]
  lma nss serve [NAME] [OPTIONS]

IDLE STARTUP AND DISCOVERY
  No model is required: `lma s` starts without loading weights or opening setup.
  GET /api/tags lists installed models with all/ollama (default: all).
  GET /v1/models lists installed models for every profile; /api/ps is initially empty.
  The first generation/load request must name an installed model. Later requests
  may omit model while one is resident. An explicit startup NAME/-m/-p still
  preloads that model. No discovery request loads or downloads weights.
  Example: lma s -a ollama -pt 11434 (requires that port to be available).

MODEL & SPEED
  -m NAME       Installed model name (or positional NAME)
  -p FILE       Explicit local GGUF path (legacy path arguments work)
  -o NAME       Select installed Ollama model; also ollama:NAME
  -or PATH      Store: this > OLLAMA_MODELS > HOME/.ollama/models
  -pr PATH      Vision projector GGUF
                        Overrides the projector discovered in an Ollama store
      -ce NAME
                        auto (default), native, plain, or a built-in format
                        Examples: chatml, llama3, gemma, gemma4, phi3, mistral-v7
      -g all|cpu|N    GPU layers (default: all in Metal/CUDA builds, else cpu)
                        cpu/0 disables model, KV/compute, and projector offload
  -c N       Context size (default: 4096; 8k = 8192; minimum: 2)
  -t auto|N  CPU threads (default: available logical CPUs)
      -b N         Prompt batch size (default: 512)
      -mh N   Physical batch (default: min(128, batch))
      -f on|off|auto
                        Flash attention (default: auto)
      -pe    Conservative exact text-prefix reuse (default: on)
      -n  Disable reuse; context still persists
      -k f16|q8_0|q4_0
                        Default: f16; quantized KV requires -f on
      -mk error|mmap
                        Default: error; opt-in CPU/pageable-weight fallback
      -e N
                        GPU-resident expert-slot budget (MiB); -g cpu selects CPU
                        Opt-in: 1..1048576; omitted = disabled (internal 0)

MODEL RESOLUTION
  Positional NAME or -m NAME resolves through the shared installed-model
  resolver on startup. -p FILE preserves explicit file intent; legacy path
  arguments still work. Missing explicit relative/absolute/.gguf paths are errors.
  Example: lma s ollama:llama3.2:latest -or /models/ollama
  Only the selected store is searched; no system-store scan, daemon, download,
  or weight copies. Dense/MoE GGUF and complete named split-GGUF sets load
  directly; select any part. Ollama multi-layer shards/adapters are unsupported.
  Supported Gemma4 safetensors need explicit `lma it` first; cached imports
  resolve by original tag. IDs are normalized tags, not digests.
  Ollama template/system/parameter layers are ignored with a warning; chat
  prefers a usable GGUF template. If missing/unusable, auto detects known
  special-token formats or uses plain role-labeled text. Base/edit models
  can generate but may not obey chat/tool instructions. -ce native
  requires an embedded template; plain forces role-labeled text. This does not
  add support for incompatible weights/architectures. Images need a compatible
  model/projector and the vision feature (llama-cpp-2/mtmd).

BACKEND SELECTION
  The pinned llama.cpp wrapper also compiles Metal on Apple Silicon without
  the root metal feature; --no-default-features is not CPU-only there.
  Metal/CUDA builds prefer GPU; CPU-only builds default to CPU. GPU requests
  fail if no GPU is available, unless -mk mmap explicitly allows
  a warned CPU retry. Build features and requested layers are not proof of
  actual placement: inspect native device, offloaded-layer, and buffer logs.
  -g all is not automatic memory fitting. On memory-constrained machines,
  start with -g 1 and a small context/batch, then increase as memory allows.

CACHE & MEMORY
  -e budgets compact expert slots, not total RSS. Dense weights, KV/context,
  graph metadata/workspaces, and OS file-cache pages need additional memory.
  Graph reservation covers worst-case 16-token stream expansion and can grow.
  With GPU requested, slots serve single-token decode and SILU prefill of
  2..16 local tokens via capacity-safe sequential GPU tiles; routing prep stays
  synchronous on CPU. Larger (>16) and empty local views use mapped CPU experts.
  Explicit -g cpu retains CPU caching; ordinary cache-off inference is unchanged.
  Uses the existing mmap GGUF backing: no on-disk weight copies or downloads.
  Requires no projector; requested batch/micro-batch settings are preserved.
  For bounded GPU prefill, explicitly choose -mh 16 (NSS --ubatch-size 16);
  this keeps local prefill <=16, not an automatic default or force-to-1 policy.
  Slot/history counts include bounded prefill, not >16 CPU fallback operations.
  No asynchronous/predictive prefetch or guaranteed speedup.
  Unsupported architectures error at load time; dry-run does not check support.
  This is a host knob, not a Modelfile PARAMETER.
  Persistent context; longest common text token prefixes are reused when safe,
  including changed suffixes. No semantic-anchor, recurrent/hybrid reuse, or
  image caching; image requests reset state. Quantized KV trades memory,
  quality, and performance. GPU-fit models need not be moved to CPU.
  With mmap fallback, requested GPU settings are tried first unchanged.
  Returned GPU availability/model/context/projector init errors (not only OOM)
  warn with the original error, then retry ONCE on CPU with mmap weights,
  use_mlock=false, CPU KV/compute, and projector GPU offload disabled.
  Explicit -g cpu -mk mmap warns and uses that mode directly.
  Bad config/header validation and backend init errors do not retry. Native
  aborts/OOM kills cannot be rescued; later generation failures do not retry.
  OS-demand-paged weights may be EXTREMELY slow. Anonymous context/KV, compute,
  and projector allocations still need RAM/OS-configured swap. No system swap
  changes, allocation guarantee, or NVMe-as-VRAM promise; GPU success is untouched.
  This is GGUF/llama.cpp, not FreeToken's FTW/MoE runtime or a performance promise.

HTTP & CONNECTION PRESETS
      -a PROFILE     all|openai|ollama|llama|lma|anthropic (default: all)
      -ct PRESET   default|claude-code|codex|zed|vscode|opencode|custom
                        Default: default. vscode means the Continue extension.
                        Without -a: claude-code -> anthropic;
                        codex/zed/vscode/opencode -> openai; default/custom -> all.
                        Explicit -a wins in either argument order.
      -ht HOST       Bind address (default: 127.0.0.1)
      -pt PORT       Port (default: 8080)
      -ay KEY     Require API authentication (see advanced HTTP help)

OTHER
      -d         Preview settings; no weights loaded or server started
                        Explicit ollama: refs/-o (or -or) resolve
                        manifests/blob headers. Local-file previews: no reads.
                        Plain names alone are not resolved in previews.
  -h            This help (long help flags are NSS-only)

NSS-ONLY LONG-FORM OPTIONS
  --model, --path, --ollama, --ollama-dir, --projector, --mmproj, --n-gpu-layers,
  --ctx-size, --threads, --batch-size, --ubatch-size, --flash-attn,
  --prompt-cache, --no-prompt-cache, --kv-cache, --memory-fallback,
  --expert-cache-mib,
  --host, --port, --api, --client, --api-key, --dry-run, --help, --help-all
  Example: lma nss serve --path model.gguf --ctx-size 8192
  NSS preserves legacy long commands and option aliases. Root accepts only
  canonical short spellings; root -- cannot enable long options.
  NSS -- supports only documented options, never upstream passthrough.

SUPPORTED FEATURES & LIMITS
  OpenAI, Ollama, llama, native lma, and Anthropic expose request subsets,
  not full llama-server, cloud API, or agent parity. All profiles use the
  same in-process engine; no performance claim. See docs/endpoints.md.
  Client presets print connection setup;
  they never spawn Claude Code, Codex, or other external agent processes.
  Function tools are client-executed; lma never runs commands or hosted tools.
  Native text streaming is supported; tool arguments are constrained/buffered,
  not unrestricted incremental tool-argument streaming. Local model/template
  capability determines tool reliability, not the selected client preset.
  No installed-client end-to-end compatibility is claimed; see docs/clients.md.
  Use `lma h s -a` for HTTP request fields and image support.
  One generation at a time; concurrent generation requests get HTTP 429.
  Native embeddings use /v1/embeddings, /api/embed, and /api/embeddings.
  No automatic downloads, LoRA adapters, or upstream passthrough.
  Non-loopback binding needs authentication and external TLS/firewall.
";

pub(super) const IMPORT: &str = "\
lma it — explicit local-only safetensors to GGUF conversion.

USAGE
  lma it -o gemma4:12b -ot models/gemma4.gguf -d
  lma it -o gemma4:12b -ot models/gemma4.gguf
  lma it -hr ./local-hf-model -ot models/local.gguf -d

OPTIONS
  -o NAME      Installed Ollama model reference (or positional NAME)
  -m NAME       Same source selector; positional local directory works
      -hr DIR       Explicit existing local Hugging Face directory
  -or DIR  Explicit store, else OLLAMA_MODELS / ~/.ollama/models
      -ot FILE.gguf    Required new output; parent directory must exist
      -d          Validate local assets and estimate output disk usage
  -h             Show this help

Ollama MLX profile: dense Gemma4 text models in per-tensor safetensors format.
Local HF profiles: LlamaForCausalLM, non-sliding MistralForCausalLM and
MixtralForCausalLM, and standard Qwen2ForCausalLM; only validated configuration
and tokenizer variants. HF F32/F16/BF16 matrix precision is preserved and vectors
are promoted to F32. This is not arbitrary safetensors or architecture support.
Architecture, tokenizer, shapes, offsets and scales are validated first.
Native NVFP4 projection weights/global scales are preserved; quantized token and
output embeddings become Q8_0 (lossy). BF16/F16/F32 inputs are also supported.
Vision/audio and speculative draft tensors are omitted: output is TEXT ONLY.
No network, Python, Ollama process, or external converter. Never modifies the
source store or overwrites output. Uses bounded streaming buffers and publishes
only a complete GGUF. Keep input assets immutable until conversion finishes.
Dry-run reads JSON, tensor headers and quantization scales; no weights loaded
for inference, no output written. Full imports may need several GiB of new disk.
After import use `lma c -p models/gemma4.gguf` or select it in the library.
A small local receipt reuses the imported file for the original -o NAME when its
folder is a standard lma model folder. Manifest changes invalidate that reuse.
No automatic conversion. Receipts are trusted local settings, not signatures.
Unsupported layouts/architectures/quantization schemes remain explicit errors.
";

pub(super) const LIST: &str = "\
lma l — installed models, without loading weights or contacting a server.

USAGE
  lma l [FILTER] [OPTIONS]
  lma l
  lma l pulsar -j
  lma l -dr /path/to/models -o /path/to/ollama/models

OPTIONS
  -dr DIR        Add a local GGUF directory (repeatable)
  -o DIR         Select the Ollama store instead of its default
  -j             Print {models, warnings}, with full paths and byte sizes
  -d             Preview options without scanning files
  -h             Show this help
  NSS-only long flags: --dir, --ollama-dir, --json, --dry-run, --help

Scans ~/.lma/models, ./models, LMA_MODELS, the desktop app-data folder, installed
LM Studio folders, and Ollama models. Ollama store: -o, OLLAMA_MODELS, then
~/.ollama/models. LM Studio: LM_STUDIO_MODELS, ~/.lmstudio/models, and
~/.cache/lm-studio/models. Discovery does not move or take ownership of files.
FILTER matches names or model references, case-insensitively. No filter lists all.
Missing optional folders are fine; scan warnings go to stderr (or JSON warnings).
Scans are bounded, skip partial downloads/projectors, and never download or load
weights. This is installed-model discovery, not `lma nss fetch MODEL --list` search.
Terminals use a width-aware table, or compact wrapped entries in narrow windows.
Long table cells are shortened with an ellipsis; -j preserves full names/paths.
Piped text keeps the stable, full-width NAME/SOURCE/SIZE/MODEL table without ANSI.
Nonempty NO_COLOR or TERM=dumb disables color; stderr is detected independently.
`l` means list; use `lh` / `lma nss launch` for the interactive launcher.
NSS accepts `list`, not `l`; option shortcuts still work. Only NSS accepts --
before a filter starting with '-': lma nss list -- -name.
";

pub(crate) const LAUNCH: &str = "\
lma lh — pick a local model and launch target (nss launch).

USAGE
  lma lh [NAME] [OPTIONS]
  lma nss launch [NAME] [OPTIONS]
  lma lh -p model.gguf
  lma lh llama3.2 -tt opencode
  lma nss launch llama3.2 --target codex
  lma nss launch llama3.2 --target claude
  lma nss launch llama3.2 --target opencode-desktop --dry-run
  lma nss launch llama3.2 --target codex-desktop --dry-run
  lma nss launch llama3.2 --target claude-desktop --dry-run

TARGET
      -tt CHOICE    chat (default), opencode, codex, claude,
                         opencode-desktop, codex-desktop, claude-desktop
                         chat uses in-process terminal chat. CLI targets require
                         an installed client. Desktop targets are gated by local
                         app/platform support; selecting one does not guarantee
                         a configurable local-model endpoint.
                         Only launch/lh accepts -tt; c/chat and serve do not.
                         -ct remains a print-only hosting preset, not a target.

SETUP

      -i      Force the setup picker even when MODEL is supplied
                         No model also enables interactive setup

SETUP KEYS
  Up/Down or Tab/Shift+Tab select a row; Left/Right or Space cycle values.
  Enter edits Model/GPU/context, changes Target, or launches on the Start row.
  Target sits between Model and GPU; Left/Right cycles all seven choices.
  Esc cancels an edit or quits setup; Ctrl+C/Ctrl+Q quits.
  -ts N|off sets the initial terminal reply cap (NSS --tokens); off/0 uses
  remaining context with no separate reply cap. Change it live with /tokens.
  Chat keys and slash commands: `lma h c`.

MODEL & SPEED (same flags and validation as serve; model is optional here)
  -m NAME       Installed model name (or positional NAME)
      -p FILE        Explicit local GGUF path
  -o NAME      Select an installed Ollama model
  -or DIR  Select the local Ollama model store
      -pr FILE   Vision projector; -pr is an alias
      -ce NAME
                         auto (default), native, plain, or chatml/llama3/gemma4
      -g all|cpu|N    Default: all in Metal/CUDA builds, else cpu
  -c N        Context tokens (default: 4096; accepts 8k)
  -t auto|N   CPU threads
      -b N, -mh N, -f on|off|auto
      -pe, -n, -k f16|q8_0|q4_0
      -mk error|mmap
      -e N
                         GPU-resident expert slots, not total RSS; 1..1048576 MiB
                         Omitted = disabled; existing mmap GGUF, no copies/downloads
                         GPU slots: decode + SILU prefill of 2..16 local tokens
                         Larger/empty views use mapped CPU experts; CPU mode unchanged
                         Requires no projector; requested batch/micro-batch preserved
                         Explicit -mh 16 (NSS --ubatch-size 16) keeps local prefill <=16
                         Graph metadata/workspaces can grow outside the slot budget
                         CPU routing prep/uploads stay synchronous; no speed guarantee
                         Unsupported architectures error; host-only, not Modelfile
                         Full model/cache/backend limits: `lma h s`

CONNECTION SETTINGS (retained for shared config; use serve for HTTP hosting)
      -a PROFILE      all|openai|ollama|llama|lma|anthropic (default: all)
      -ct PRESET    default|claude-code|codex|zed|vscode|opencode|custom
                         vscode selects Continue, not a universal VS Code API.
                         Omitted -a: claude-code -> anthropic;
                         codex/zed/vscode/opencode -> openai; default/custom -> all.
                         Explicit -a wins regardless of argument order.
      -ht HOST        Bind address (default: 127.0.0.1)
      -pt PORT        Port (default: 8080)
      -ay KEY      Require API authentication

OTHER
      -d          Print config only, including the typed target: no TTY,
                         discovery, model resolution, weights loaded, server or
                         client started, desktop app opened, or files written.
                         Desktop targets can be previewed without executing them;
                         runtime app/platform gates still apply to a real launch.
                         An omitted model is serialized as an empty string.
  -h             This help (long help flags are NSS-only)

SUPPORTED FEATURES & LIMITS
  The default chat target runs in-process; use `lma s` for HTTP hosting.
  --mode is no longer accepted; select clients explicitly with -tt.
  API/client profiles remain connection presets, not full Ollama, Claude Code,
  or Codex parity. Function tools are client-executed, never server-executed.
  Native text streams; tool arguments are constrained/buffered. Local model
  and template capabilities matter; no hosted tools or blanket client support.
  Printed setup never contains the server key. See docs/clients.md.
  No automatic model downloads. Image generation is a separate CLI command
  using local SD1.5 assets, not an agent tool or automatic chat skill.
  For direct terminal chat use `lma c`; HTTP details: `lma h s -a`.
  The chat target renders Markdown safely; formatting/keys: `lma h c`.
";

pub(crate) const CHAT: &str = "\
lma c — open local terminal chat directly (nss chat).

USAGE
  lma c [NAME] [OPTIONS]
  lma nss chat [NAME] [OPTIONS]
  lma c -p model.gguf -g cpu -c 8k
  lma c llama3.2 -ts off -c 8k
  lma nss chat llama3.2 --tokens 2048 --ctx-size 8k

SETUP
  Opens model/GPU/context setup without launch targets; no model enables the picker.
  -i forces setup with a model. --mode is no longer accepted.
  -tt is launch-only; use `lma lh -tt CHOICE` to select an external client.
  Use `lma s` for HTTP hosting, not chat/launch.
  Setup keys: Up/Down or Tab/Shift+Tab select rows; Left/Right change values;
  Enter edits/starts; Esc cancels an edit or quits setup.

REPLY LENGTH
  -ts N|off              Initial terminal reply cap (NSS: --tokens N|off)
                        Default: min(256, context - 1); off/0 removes the reply cap
  Change it live with /tokens N or /tokens off. No-cap mode uses remaining native
  context after the exact prompt/template/image positions, not an estimate.
  EOS, cancellation and the context window still stop generation. There is
  no automatic history truncation or context shifting. At a limit the MODEL STAYS
  LOADED; send another message, /save then /new, or restart with a larger -c.
  The status distinguishes reply/context limits from the model ending its reply.

CHAT KEYS
  Enter / Ctrl+Enter     Send
  Shift+Enter            Newline, when reported distinctly by the terminal
  Alt+Enter              Newline fallback for terminals without Shift+Enter
  Tab                    Change focus between composer and history
  Esc                    Close/cancel a panel or cancel generation
  Ctrl+C                 Cancel while generating; quit when idle
  Ctrl+Q                 Quit
  PageUp / PageDown      Scroll history (PgUp/PgDn)
  Mouse wheel            Scroll history
  Up / Down              Scroll when history is focused; edit lines in composer
  Home / End             Oldest/latest when history is focused
  Ctrl+Home / Ctrl+End   Oldest/latest from either focus; Ctrl+End follows live output
  Reading older messages keeps your position while replies stream or finish.
  Keyboard enhancement is requested only on terminals supporting it. Legacy
  terminals may send Shift+Enter as plain Enter; use Alt+Enter as the fallback.

SLASH COMMANDS
  /help                  Show terminal help
  /new                   Start a new conversation
  /models                Unload model, clear conversation, and reopen setup
  /system TEXT           Set system instructions; omit TEXT to clear them
  /tokens N|off          Reply cap: 0 < N < context size; off/0 = remaining context
  /tokens                Show current cap (default: 256, or -ts setting)
  /temperature N         Sampling temperature (0..2; 0 = greedy)
  /raw                   Raw prompt mode (without chat-template formatting)
  /chat                  Return to chat-template mode
  /markdown              Render assistant Markdown (default)
  /plain                 Show literal assistant text (does not change prompts)
  /image FILE            Attach a local PNG/JPEG for a compatible vision model
                         Requires a projector; max 8 images, 8 MiB/image,
                         16 MiB total, and 24 MiB serialized JSON request
  /save FILE             Save completed text and attachment names to a NEW file
                         Existing files are never overwritten
  /logs                  Open logs
  /quit                  Quit

MARKDOWN DISPLAY
  Assistant replies render Markdown in the terminal: headings, bold, italic,
  inline/fenced code, lists, quotes, and links. Code and links are display-only:
  no commands are executed, no URLs are opened, and HTML is not executed.
  Model text is treated as untrusted display data, not terminal control codes.
  Formatting is display-only; requests and saved Markdown retain the original text.
  Unsupported syntax stays literal; this is not a full CommonMark/HTML renderer.
  Nonempty NO_COLOR or TERM=dumb disables colors, including in chat/setup.

SHARED FLAGS
  -m NAME       Installed model name (optional; or positional NAME)
      -p FILE        Explicit local GGUF path
  -o NAME      Installed Ollama model
  -or DIR  Local Ollama store
      -pr FILE
      -g all|cpu|N, -c N, -t auto|N
      -b N, -mh N, -f on|off|auto
      -pe, -n, -k f16|q8_0|q4_0
      -mk error|mmap
      -e N
                         GPU-resident expert slots, not total RSS; 1..1048576 MiB
                         Omitted = disabled; existing mmap GGUF, no copies/downloads
                         GPU slots: decode + SILU prefill of 2..16 local tokens
                         Larger/empty views use mapped CPU experts; CPU mode unchanged
                         Requires no projector; requested batch/micro-batch preserved
                         Explicit -mh 16 (NSS --ubatch-size 16) keeps local prefill <=16
                         Graph metadata/workspaces can grow outside the slot budget
                         CPU routing prep/uploads stay synchronous; no speed guarantee
                         Unsupported architectures error; host-only, not Modelfile
      -a all|openai|ollama|llama|lma|anthropic
      -ct default|claude-code|codex|zed|vscode|opencode|custom
      -ht HOST, -pt PORT, -ay KEY
                         Hosting flags are retained in config, not a promise
                         that chat starts an HTTP listener or external client.
                         See `lma h lh` for profile defaults.

OTHER
      -d          Config only; no TTY/model loading/discovery or writes
                         Missing model stays empty; API keys are redacted.
  -h             This help (long help flags are NSS-only)

SUPPORTED FEATURES & LIMITS
  In-process local chat, not a Claude Code/Codex agent or shell executor.
  No tool calling or automatic model downloads. Image generation is the
  separate image/imagen command using local SD1.5 assets only.
  Full shared flag/backend limits: `lma h s`.
";

pub(super) const QUANTIZE: &str = "\
lma q — smaller GGUF models, in-process.

USAGE
  lma q INPUT [OUTPUT] [OPTIONS]
  lma q model.gguf -t q4_k_m

OPTIONS
  -t TYPE        Type (default: Q4_K_M; case-insensitive)
  -o FILE         Output (or use positional OUTPUT)
  -ts auto|N  CPU threads (default: available logical CPUs)
      -p            Don't mix quantization types between tensors
      -l    Leave output tensor unquantized
      -a Allow already-quantized input (quality may degrade)
      -f           Replace existing output, never the input
      -d         Print settings without reading/writing models
  -h            This help (long help flags are NSS-only)

SUPPORTED TYPES
  F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K_S Q3_K_M Q3_K_L
  Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K

Output defaults to INPUT_STEM.TYPE.gguf. Writes a temporary sibling file
and publishes only after success. Use a trusted output directory. On
non-Unix systems, -f cannot replace an existing file.

NSS EXAMPLE
  lma nss quantize input.gguf output.gguf Q8_0 8
  Long options: --type, --out, --threads, --pure, --leave-output-tensor,
  --allow-requantize, --force, --dry-run. File safeguards also apply to nss.

No importance matrices, split outputs, or tensor overrides yet.
Quantization can reduce quality; not every type suits every model.
";

pub(super) const NSS: &str = "\
lma nss — no short syntax.

Use the SAME embedded engine with long command names and flags.
Use serve/server, stop, launch/lh, list, chat, transcribe, help,
quantize/quant, fetch, import, create, delete, or image/imagen/image-gen/img.
All legacy long aliases live HERE, not at root.
Root commands/flags use first letters, then first + last on collisions.
Root t pairs with nss transcribe; s serves and sp stops.
Option shortcuts (-o, -od, -m, -c, -t, -q, -h) and positional models still work.
For create, -f is the file option, not the disabled fetch command alias.
This does not launch llama.cpp executables or accept every upstream flag.

EXAMPLES
  lma l                  / lma nss list
  lma lh                 / lma nss launch
  lma c llama3.2         / lma nss chat llama3.2
  lma t -m whisper a.wav / lma nss transcribe -m whisper a.wav
  lma s llama3.2        / lma nss serve llama3.2
  lma sp -m llama3.2     / lma nss stop -m llama3.2
  lma h s               / lma nss help serve
  lma nss serve --path model.gguf -c 8k --gpu cpu
  lma nss --dry-run serve --model llama3.2
  lma nss create my-local --file ./Modelfile --dry-run

Use `lma h s` or `lma h q` for supported options.
NSS --dry-run works before or after the command; --help shows local help.
Root uses -d, -h, and -v. Long global options belong in NSS.
Output naming and overwrite safeguards remain active in nss mode.
Unknown options are errors, including after --. --backend and LMA_SERVER /
LMA_QUANTIZE are no longer used; external clients require an explicit launch target.
";

const API: &str = "\
lma HTTP API — supported request subsets, not full provider/agent parity.

ENDPOINTS
  GET  /health                Ready status (no API key required)
  GET  /v1/models             ID: GGUF filename or normalized Ollama name:tag
  POST /v1/completions        Text prompt -> text response, optional SSE
  POST /v1/chat/completions   Text/image messages -> assistant response, SSE
  POST /api/chat              Ollama chat + function tools, NDJSON by default
  POST /api/generate          Ollama raw/formatted generation, NDJSON by default
  GET  /api/version          lma s version (not an Ollama release)
  GET  /api/tags             Installed models across lma/Ollama/LM Studio
  GET  /api/ps               Currently loaded model and configured context
  POST /api/show              Selected model header information
  POST /api/pull              Download a public GGUF into ~/.lma/models/
  POST /api/create, /api/copy Create lma aliases from installed models
  POST /api/delete           Delete a model; choose source if ambiguous
  POST /api/stop              Unload; also `lma sp` / `lma nss stop`
  POST /v1/embeddings, /api/embed, /api/embeddings
                              Native normalized vectors and exact token usage
  POST /v1/messages           Anthropic-format messages + function tools
  POST /v1/messages/count_tokens
                              HTTP 200: {\"input_tokens\":N} (served GGUF tokens)
  POST /v1/responses          Stateless Responses + function tools
  GET  /props                llama model/context properties
  POST /completion           llama raw text completion, optional SSE
  POST /tokenize             Actual model token IDs from content
  POST /detokenize           Text from actual model vocabulary token IDs
  GET  /lma/v1/status        Native lma status (api_version:1)
  POST /lma/v1/generate      Native raw text generation, optional NDJSON
  POST /lma/v1/chat          Native text/image chat, optional NDJSON; no tools

API PROFILES
  Root: lma s MODEL -a PROFILE. NSS: lma nss serve MODEL --api PROFILE.
  all|openai|ollama|llama|lma|anthropic; all enables every family (default).
  openai enables completions/chat/responses/embeddings; ollama enables /api
  routes; anthropic preserves /v1/messages and /v1/messages/count_tokens.
  llama enables /props, /completion, /tokenize, /detokenize only.
  lma enables /lma/v1/status, /lma/v1/generate, /lma/v1/chat only.
  /health and /v1/models remain shared; /api/stop is shared host control.
  Browser chat is available with all/openai; others show API info JSON at /.
  These are subsets, not full llama-server or cloud parity. All profiles use
  the same in-process engine; no performance claim. See docs/endpoints.md.
  --client default|claude-code|codex|zed|vscode|opencode|custom prints presets.
  Without --api: claude-code selects anthropic; codex/zed/vscode/opencode select
  openai; default/custom select all. Explicit --api wins in either order,
  even if it disables a preset's API; no usable preset is printed in that case.
  No environment mutation, config-file writes, external agent spawning,
  or secrets in printed connection settings.
  zed configures the OpenAI-compatible model provider, NOT ACP/external agents.
  vscode configures the Continue extension, not a universal VS Code API.
  opencode selects @ai-sdk/openai-compatible for Chat Completions; codex uses
  a user-level Responses provider with WebSockets and compression disabled.
  No installed-client end-to-end compatibility is claimed; see docs/clients.md.

LLAMA SUBSET
  Base URL: http://127.0.0.1:8080 (NOT /v1).
  POST /completion: prompt:string, n_predict:positive|-1, temperature (alias
  temp), top_p, seed, stream (default false; true enables SSE).
  Omitted n_predict inherits stored num_predict, otherwise 256; -1 means
  remaining context. Slots, FIM, cache_prompt, and stop controls are rejected.
  POST /tokenize: content:string, add_special/parse_special booleans (default false).
  POST /detokenize: tokens:[i32]. Uses actual token IDs/model vocabulary,
  not character/byte stand-ins. Unsupported fields are rejected.

NATIVE LMA V1
  Base URL: http://127.0.0.1:8080/lma/v1.
  /generate takes prompt:string; /chat takes messages using the existing
  OpenAI role/content/images schema, but no tools or remote image fetching.
  Both accept model?:string, max_tokens?:positive|null, temperature, top_p,
  seed, stream (default false). Omitted max_tokens inherits stored num_predict
  or defaults to 256; explicit null means remaining context/no reply cap.
  Zero/negative max_tokens are rejected. Nonstream JSON: api_version:1, id,
  model, text, usage:{input_tokens,output_tokens}, finish_reason:
  eos|token_limit|context_limit. stream:true emits NDJSON start/delta/done/error
  records; delta carries text, done carries usage/finish without aggregate text.
  New routes have 1 MiB bodies except native /lma/v1/chat (24 MiB).
  HTTP model selection accepts installed models only; unknown names/filepaths
  never trigger implicit downloads. Streaming host leases last until body done.

TEXT REQUEST
  {\"prompt\":\"Once upon a time\",\"max_tokens\":64}

CHAT REQUEST
  {\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":64}
  Roles: system, user, assistant. Chat prefers a usable GGUF template;
  missing/unsupported templates use a known special-token format or plain
  role-labeled text automatically. --chat-template overrides this policy.
  Ollama template/system/parameter layers are still ignored with a warning.
  Base/edit model chat quality is not guaranteed. Raw completions stay exact.

OPTIONAL COMPLETIONS / CHAT COMPLETIONS FIELDS
  model         Served model ID (must match if provided)
  max_tokens    Positive token limit (default: 256)
  temperature   0..2 (default: 0.8; 0 = greedy)
  top_p         Greater than 0, at most 1 (default: 0.95)
  seed          Unsigned 32-bit seed (optional)
  stream        Chat and raw completions support native SSE (default: false)

IMAGES
  Requires a vision build (default) and compatible GGUF model + projector.
  Use --projector/--mmproj PATH or an installed Ollama model with a projector.
  PNG/JPEG only; only user messages may contain images. No HTTP(S), remote,
  or file URLs are ever fetched. /v1/chat/completions accepts either:
    - String content plus message images: [raw-base64 strings].
    - Content parts: {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,...\"}}
      followed by {\"type\":\"text\",\"text\":\"Describe this image.\"}.
  data:image/jpeg;base64,... also works. Image parts must precede all nonempty
  text parts; images after text are rejected, NOT reordered. Message images
  attachments precede content parts. No image caching; image requests reset.
  Limits: 8 images/request, 8 MiB decoded/image, 16 MiB decoded total.
  Body limits: 24 MiB for both chat routes; 1 MiB for /v1/completions.
  See README for a local-file Python/base64 + curl example.

OLLAMA SUBSET
  /api/chat: messages with role, string content, optional raw-base64 images;
  function definitions/calls/results use client-executed tools.
  /api/generate: prompt, optional system/images; raw:true keeps the exact prompt.
  stream defaults to TRUE (NDJSON); explicit false returns one JSON response.
  options: num_predict, temperature, top_p, seed, num_ctx.
  num_predict:-1 uses remaining context; omission inherits a model limit if set,
  otherwise has no reply cap. EOS/cancellation/context exhaustion still stop it.
  Managed HTTP num_ctx resizes via a safe reload; unchanged context reuses the model.
  Pinned desktop/in-process sessions still require num_ctx to match their context.
  Example: {\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":false,\"options\":{\"num_predict\":64}}
  Returns model, created_at, content/tool calls, done/done_reason, engine token
  counts, and measured total_duration; no invented phase timings or VRAM stats.
  The HTTP host keeps one model resident and loads installed models by name.
  keep_alive accepts seconds, duration strings, 0 (unload), or -1 (keep loaded).
  Switching/stopping during active requests returns 409; cancel the request first.
  Pull/copy/create/delete administration is supported; create is alias-only.
  No push, blob upload, arbitrary Modelfiles, or training. Unsupported fields fail.
  See docs/ollama-api.md for exact formats, authentication, and lifecycle limits.

ANTHROPIC MESSAGES
  Required: model, messages:[{role:user|assistant,content:TEXT}], max_tokens.
  content accepts a string or [{type:text,text:TEXT}]; optional top-level system
  accepts the same text forms. Optional: temperature, top_p, stream (false).
  Example: {\"model\":\"model.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}
  Function tools and tool_use/tool_result turns are client-executed: send tool
  definitions and return their results in the supported protocol forms.
  Tool arguments use constrained/buffered generation, not a hosted executor.
  User image blocks: base64 PNG/JPEG source; no URL fetches. Same vision and
  image ordering limits as chat, with a smaller 1 MiB body limit. Combining
  vision and tool history needs real-model validation. Hosted tools unsupported.

MESSAGES TOKEN COUNTING (TEXT ONLY)
  POST /v1/messages/count_tokens accepts model, messages, and optional system
  in the same text forms as Messages; omit generation options/max_tokens.
  Success: HTTP 200 {\"input_tokens\":N}. Counts with the served GGUF tokenizer,
  applied chat template, and generation special tokens, without generation.
  This is not Anthropic's proprietary token count. Function tools/history are
  included in the count; images and unsupported fields are rejected.

OPENAI RESPONSES (STATELESS)
  Required: model, input (string or text-message array).
  Message fields: optional type:message, role, content (string or input_text/
  output_text parts). Optional: instructions, max_output_tokens (default: 256),
  temperature, top_p, stream (false), store (false; true is rejected).
  Example: {\"model\":\"model.gguf\",\"input\":\"Hello\",\"max_output_tokens\":64}
  Include the full conversation, function calls, and function-call outputs
  each time. Function execution and approvals belong to the client, not lma.
  User input_image parts accept PNG/JPEG data URLs; no URL fetches, detail:auto
  only, 1 MiB body limit. Vision with tools needs real-model validation.
  No server-managed sessions, previous_response_id, persistence, hosted tools,
  agent execution, or arbitrary provider fields.

STREAMING & FUNCTION TOOLS
  Text generation supports native streaming: SSE (OpenAI/Anthropic), NDJSON (Ollama).
  Tool arguments use constrained/buffered generation; do not assume unrestricted
  live tool-argument deltas or provider-identical events/timing.
  Tools are client-executed: the client validates requests, obtains permission,
  executes functions, and sends results back. lma has no hosted shell, browser,
  web search, code interpreter, or MCP execution service.
  A local model and compatible chat template must reliably follow tool schemas.
  A connection preset does not add model capabilities or guarantee every client
  release, agent mode, schema, thinking mode, or advanced workflow works.

Prompt (including chat/image formatting) + requested output must fit context.
Unsupported fields and hosted tool types are rejected, not silently executed.
Completions/Chat Completions responses contain choices, usage, finish_reason.
Only one generation runs at once; busy requests receive HTTP 429.
Context persists; default prompt caching only reuses safe exact prefixes of
identical/extended text prompts, not edits, semantic anchors, or recurrent/
hybrid state. Send desired chat history each time; no server-managed history.
With --api-key, send `Authorization: Bearer KEY` on every protected API route,
including llama/native lma, discovery, and shared /api/stop control.
Only /v1/messages and /v1/messages/count_tokens also accept `x-api-key: KEY`.
/health stays public; non-loopback use needs external TLS/firewall protection.
";

pub(super) const FETCH: &str = "\
lma f — find and download public GGUF models.

USAGE
  lma f [QUERY] [OPTIONS]
  lma f -l                   # Top 50 by Hugging Face downloads; no weights fetched
  lma f -l -i 20             # Show up to 20 repositories
  lma f llama3.2 -l
  lma f llama3.2 -oe
  lma f lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF -le

SOURCES
  With a query and no source flags: combine both catalogs, with source labels.
  Without a query, -l lists popular public Hugging Face GGUF repositories by
  reported downloads (default 50). This is not a merged Ollama/HF ranking.
  -oe      Only Ollama's registry (exact model[:tag] lookup)
  -le   Only public Hugging Face GGUF repositories,
                        the source used by LM Studio's downloader
  Both flags: combine both sources. Model weights are never merged.

OPTIONS
      -l            List candidates without downloading weights; QUERY optional
      -i NUM        Maximum results to display (index/count; default 50; 1..1000)
                    Does not select a model. NSS: --index NUM.
      -f NAME       Exact Hugging Face GGUF filename
  -o DIR                Download root (default: ~/.lma/models)
                        Independent of working directory. Use -o . explicitly.
                        Missing/relative HOME/USERPROFILE needs -o DIR.
      -d         Preview settings; no network or writes
  -h            This help (long help flags are NSS-only)

A query is required to download. Catalog popularity does not prove architecture
support, quality, or memory fit. -oe alone needs a query (no Ollama popularity API
is used). Listing contacts catalog metadata only and never creates model files.

Downloads require one unambiguous exact model/repository match; fuzzy search
results are not downloaded automatically. On Hugging Face, -f wins;
otherwise prefer a unique Q4_K_M file, then a unique standalone GGUF.
Ollama projectors are downloaded too. HF projectors need separate selection.

Files stream to temporary siblings and are SHA256/size/GGUF-header checked
before publication. Existing verified files are reused; mismatches refused.
Private/gated repositories, split GGUFs, adapters, resume, and image-generation
weights are not supported. No daemon or subprocess; downloads may be many GB.
Requests contact Ollama/Hugging Face and their HTTPS storage redirects.
The completed command prints local paths and how to serve them.
~/.lma/models is lma's own download/registry location, separate from Ollama and
LM Studio stores. Explicit -o DIR changes the destination; discovery/aliases do
not transfer ownership of external files or copy them into ~/.lma.
";

pub(super) const IMAGE: &str = "\
lma i — local Stable Diffusion 1.5 text-to-image.

USAGE
  lma i MODEL_DIR \"a watercolor fox\" [OPTIONS]
  lma i -m models/sd15 -p \"a tiny cabin\" -o cabin.png
  NSS-only aliases: image, imagen, image-gen, img; root uses i only

OPTIONS
  -m DIR    Local SD1.5 Diffusers directory (required)
  -p TEXT      Image prompt (required; or second positional)
  -n TEXT
                        Things to avoid (default: empty)
  -o FILE        New PNG output (default: image.png; never overwritten)
      -t FILE  Override local CLIP tokenizer JSON
      -w N         64..1024, divisible by 8 (default: 512)
      -ht N        64..1024, divisible by 8 (default: 512)
      -s N         DDIM steps, 1..150 (default: 20)
      -sd N          Initial-noise seed (default: 42)
      -g N      CFG scale, 1..20 (default: 7.5)
      -de NAME     auto|cpu|metal|cuda (default: auto)
      -a N
                        Unavailable: Candle 0.11 tensor-rank bug. Runtime
                        rejects any value before loading weights, even a valid
                        head-count divisor. -d can still preview config;
                        omit this flag for generation. Reduce dimensions to
                        lower memory use instead.
      -d         Preview config; no local assets/backend required,
                        weights loaded, or files written
  -h            This help (long help flags are NSS-only)

REQUIRED LOCAL ASSETS
  model_index.json
  text_encoder/config.json + model.safetensors
  unet/config.json + diffusion_pytorch_model.safetensors
  vae/config.json + diffusion_pytorch_model.safetensors
  scheduler/scheduler_config.json
  tokenizer/tokenizer.json (or -t FILE)

Use the SD1.5 CLIP-vit-large-patch14 tokenizer, not an LLM tokenizer.
Only standard SD1.5 is supported, not GGUF, SDXL, Flux, or pickle checkpoints.
Runs Candle in-process; source builds with the image-gen feature only.
No automatic model downloads. Keep mmap'd weights immutable.
CPU is slow; Metal/CUDA require the matching build feature. GPU results are
not guaranteed bit-identical. Image-gen is a CLI command, not a chat-model skill.
";

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn help_is_always_local_and_needs_no_model() {
        for argv in [
            vec!["h"],
            vec!["h", "s", "-a"],
            vec!["h", "q"],
            vec!["s", "-h"],
            vec!["q", "-h"],
            vec!["h", "lh", "-a"],
            vec!["h", "c"],
            vec!["l", "-h"],
            vec!["c", "-h"],
            vec!["i", "-h"],
            vec!["nss", "server", "--help-all"],
            vec!["nss", "help", "chat", "--all"],
            vec!["nss", "imagen", "--help"],
            vec!["h", "nss"],
        ] {
            assert!(matches!(
                crate::cli::parse(argv.into_iter().map(OsString::from)).unwrap(),
                Action::Help(_)
            ));
        }
    }

    #[test]
    fn no_obsolete_executable_lookup_instructions() {
        assert!(!ROOT.contains("on PATH"));
        assert!(NSS.contains("no short syntax"));
    }
}
