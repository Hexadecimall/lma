# Native settings and model-store IPC

These commands are registered in `main.rs`. No UI files are owned by this implementation.
All command errors reject with a string. Neither settings nor store commands load a model.

## Settings

- `app_settings_load()` → `Settings`
- `app_settings_save({ settings: Settings })` → `null`

The complete camelCase payload, with defaults:

```json
{
  "loadPolicy": "automatic",
  "defaultsPath": null,
  "autoRestore": true,
  "gpu": "auto",
  "context": 4096,
  "modelDirectories": [],
  "ollamaDirectory": null,
  "lmStudioDirectories": []
}
```

`loadPolicy` is `automatic`, `defaults`, or `manual`. `gpu` is a string:
`auto`, `all`, `cpu`, or an unsigned 32-bit decimal layer count. Context is a
32-bit integer >= 2. Directory arrays contain strings (64 custom directories
total). Paths are absolute or start with `~/`, at most 4096 bytes, without control
characters. Tilde expansion happens only in the native path adapter; shell or
environment substitution is never performed.

Settings live in the host-selected app-config directory as `app-settings.json`.
Missing settings return defaults without writes. Reads/writes are serialized;
JSON is capped at 64 KiB and unknown fields are rejected. Writes use a private
0600 temporary file on Unix, sync it, and rename atomically. Existing malformed,
oversized, unknown-field, or symlink files are not repaired or overwritten.
Parent directories are trusted, as with the existing desktop storage modules.

`load_model({ options })` keeps the existing LoadOptions payload:
`{model, ollama, ollamaDir, projector, gpu, context}`. It reads settings and applies
policy before touching a working session:

- **automatic**: ignores supplied GPU/context, using the shared compiled GPU
  default and `ServeConfig::default().context` (4096). No silent CPU fallback.
- **defaults**: reads `defaultsPath`, or `~/.lma/config.json` when null; applies
  only the GPU/context whitelist below. Missing or invalid config is an error.
- **manual**: uses GPU/context from the supplied LoadOptions. The frontend should
  initialize its controls from the settings' `gpu` and `context` fields.

`autoRestore` is only a persisted frontend preference. The frontend decides whether
to restore `discover_models().lastModel`; native setup does not auto-load.

### Defaults-config whitelist

Flat JSON object only. Every field is optional; omitted fields use the same
shared automatic defaults:

```json
{"gpu":"auto", "context":4096}
```

- `gpu`: `"auto"`, `"all"`, `"cpu"`, an unsigned decimal string, or a u32 integer.
- `gpu_layers`: CLI-serialized u32 integer, **instead of** `gpu` (0 means CPU).
- `context`: u32 integer >= 2.

Specifying both GPU fields, explicit nulls, invalid types/counts, duplicate fields,
unknown fields, or non-object JSON is rejected. Full ServeConfig/CLI plan objects
are **not** supported. Model/projector paths, API keys, host/port, tools, commands,
threads, cache configuration, fallback policy, and all other fields are rejected;
they are never applied or executed. The config is read-only, regular/non-symlink,
and bounded to 64 KiB.

## Discovery

`discover_models()` retains its `{models, warnings, lastModel}` shape and stable
model IDs. It combines shared `default_local_dirs()`/`lm_studio_dirs()` with app
model data, registered/built source checkout, cwd, and the configured roots. It
also scans default Ollama storage plus the configured and remembered Ollama stores.
Custom LM Studio roots are labelled `lm-studio`; Ollama remains `ollama`, and other
local/managed models remain `lma`. Canonical duplicate roots and stable IDs are
deduplicated. Discovery is read-only and does not copy weights or use the network.

## Store

- `store_search({ query: string, source: string })` →
  `{models: StoreModel[], warnings: string[]}`
- `store_download({ id: string, source: string, file?: string | null })` →
  `{path: string, model: string}`

```ts
type StoreModel = {
  id: string;
  source: 'ollama' | 'lm-studio';
  name: string;
  exact: boolean;
  files: {name: string; bytes: number | null}[];
  installed: boolean;
  totalBytes: number | null;
};
```

Inputs accept source `ollama`, `lm-studio`, or `huggingface` (alias for
`lm-studio`); search additionally accepts `all`. LM Studio is the existing public
Hugging Face GGUF catalog, not a proprietary registry. Ollama supports exact
`[namespace/]name[:tag]` lookup, not a hardcoded/fuzzy model listing.

Fuzzy HF results have `exact:false`, empty files, and null size. The frontend must
select an exact `owner/repository` and search that ID to get its GGUF file list.
Download never chooses a fuzzy result. If an exact repository exposes multiple
GGUFs, supply its exact relative filename in `file`; the desktop adapter does not
silently pick the CLI's preferred quantization. The shared selector still rejects
unsupported shards, adapters, gated repositories, or projector-only selections.
Unknown file sizes are null. `totalBytes` includes Ollama's selected artifacts, or
one HF file; it is null for multiple alternative HF files.

Search/download are explicit user actions and use only the existing Rust
catalog/downloader. They send public queries/IDs to the selected upstream services;
no credentials, chats, or other workspace data are sent. No network runs on startup.

Downloads are serialized by managed AppState and remain supervised when the
webview invocation disconnects. Before network access, known normalized installed
identities are checked across the same library catalog (Ollama references, managed
source paths, repo paths, and logical registry names). Reuse returns the existing
path/reference without copying, registry mutation, or downloading. No fuzzy
basename match is used to equate different repositories or quantizations.

New weights always go under `models::discovery::default_models_dir()`
(`~/.lma/models`), irrespective of configured scan paths or cwd. The existing
SHA256/size-verified, no-clobber downloader is reused. New standalone GGUFs are
registered without overwriting aliases: Ollama uses its normalized tagged name;
HF uses `owner/repository:<filename-stem>-<sha256-prefix>` to distinguish files.
An alias collision is an error, never a silent replacement.

For a reused model, `path` is the absolute local GGUF path and `model` is its
existing library load selector/reference. For a new download, `model` is the
registered logical name. After success, the parent frontend must immediately call
`discover_models()` and select the matching path/reference. There is no native
refresh loop and no automatic inference load.
