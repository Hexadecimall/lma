# Desktop workspace

Isolated desktop assets. Tauri `build.frontendDist` points to `ui` relative to
`desktop/tauri.conf.json`. These assets do not import or modify the browser UI.

- `core.js`: attributed desktop-local safe helpers, request validation, image
  normalization, text-only Markdown renderer, clipboard and downloads.
- `workspace.js`: version-1 host schema allowlists, atomic mutations, UUID IDs,
  non-destructive branching, hydration and a 400 ms serialized autosave queue.
- `app.js`: DOM/controller and only the documented Tauri IPC commands.
- `preferences.js`: frontend application-settings validation and source labels.
- `notifications.js`: transient desktop notices.
- `index.html`, `style.css`: local, dependency-free shell and desktop layout.

## Native launch harness

The entry point is `desktop/ui/index.html`. The module auto-starts and exposes
`globalThis.__LMA_DESKTOP__`. Await `__LMA_DESKTOP__.ready` for the initial status,
workspace hydration, model scan and any one-shot restore attempt to settle.
Workspace editing becomes available as soon as hydration settles, without waiting
for a slow model restore. Check `state.error` and `store.persistence` separately;
read/save failures remain visible and do not reject the normal startup promise.

Useful controller API:

```js
const app = globalThis.__LMA_DESKTOP__;
await app.ready;
app.setView("chat"); // or "models", "settings"
app.setModelsTab("installed"); // or "store"; does not start a network search
app.openInspector(true);
app.newChat();
app.selectConversation(id);
await app.send(); // whole nonstream response; state.active while pending
app.stop();
await app.retry();
await app.branchAt(messageIndex, false); // edit in a new conversation
await app.branchAt(assistantIndex, true); // regenerate in a new conversation
await app.discover(); // rescan, does not auto-restore again
app.store.updateDraft(app.store.currentId, "Working draft"); // lightweight typing path
await app.store.flush(); // resolves false on persistence failure; data retained
const snapshot = app.store.workspace;
const persistence = app.store.persistence;
```

`store.workspace` and `store.current` return defensive working snapshots.
`store.composer` is a lightweight defensive summary (draft, settings, attachment
count; no history or image data). `store.listConversations(query)` returns only
matching IDs/titles. `store.contentRevision` changes on structural mutations and
hydration, not draft edits; the controller caches its defensive rendering snapshot
against that revision and the current ID. Draft events update only the composer
and notices; persistence notifications update only notices. Neither path clones,
serializes, decodes, or rebuilds message/image history.

`store.updateDraft(id, text)` validates the draft's type and UTF-8 size without
walking unrelated data. Full workspace schema/byte/image validation still happens
before every serialized save. Drafts over the 1 MiB text limit remain local and
recoverable through navigation and JSON/Markdown export; they display an actionable
error and block sending/saving until shortened or the conversation is removed.
Working snapshots include that recoverable text, so use `store.flush()` rather
than passing working snapshots straight to persistence IPC. No partial invalid
workspace is saved, and failed saves leave dirty working data intact.

Native operation guards apply to inference, regeneration, and model changes, not
navigation, drafts, settings, rename/delete of unrelated chats, or edit-as-branch.
Ephemeral active requests/errors/model state are never in persisted snapshots. Source updates are
available through the explicit UI button and always require confirmation. No
startup code invokes `source_update`. `beforeunload` flushes best-effort; native
close coordination can await `store.flush()` if a guaranteed final write is needed.

For injected DOM/IPC tests use exported `createDesktopApp({ document, window,
invoke, setTimer, clearTimer })` and call `.start()`. Suppress auto-start before
loading the module with `globalThis.__LMA_NO_AUTOSTART__ = true` when necessary.

### Stable DOM IDs

| Area            | IDs                                                                                                                                                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Navigation      | `new-chat`, `nav-chat`, `nav-models`, `nav-settings`, `conversation-search`, `conversation-list`                                                                                                                                 |
| Header          | `conversation-title`, `model-quick-switch`, `conversation-options`                                                                                                                                                               |
| Chat            | `chat-view`, `message-scroll`, `messages`, `composer-form`, `draft`, `send`, `stop`, `retry-request`                                                                                                                             |
| Images          | `attach-image`, `image-input`, `attachments`                                                                                                                                                                                     |
| Library         | `models-view`, `model-search`, `installed-models`, `rescan-models`, `load-gpu`, `load-context`, `load-model`, `unload-model`, `delete-model`, `delete-model-note`                                                                |
| Model tabs      | `models-installed-tab`, `models-store-tab`, `installed-models-panel`, `store-models-panel`                                                                                                                                       |
| Store           | `store-search-form`, `store-query`, `store-source`, `store-search`, `store-results`, `store-status`                                                                                                                              |
| App settings    | `settings-view`, `app-settings-form`, `app-load-policy`, `app-defaults-path`, `app-gpu`, `app-context`, `app-auto-restore`, `app-model-dirs`, `app-ollama-dir`, `app-lm-studio-dirs`, `save-app-settings`, `app-settings-status` |
| Optional import | `toggle-manual-model`, `manual-toggle`, `manual-import`, `manual-source`, `manual-model`, `manual-name`, `manual-ollama-dir`, `manual-projector`, `manual-load`                                                                  |
| Inspector       | `inspector`, `close-inspector`, `rename-title`, `rename-conversation`, `delete-conversation`, `settings-form`                                                                                                                    |
| Generation      | `setting-mode`, `setting-max-tokens`, `setting-temperature`, `setting-top-p`, `setting-seed`, `setting-system`, `setting-coding`                                                                                                 |
| Export/update   | `export-json`, `export-markdown`, `source-update`, `update-state`, `update-log`                                                                                                                                                  |
| Errors          | `persistence-error`, `persistence-error-text`, `persistence-retry`, `app-error`, `request-error`                                                                                                                                 |

Conversation rows carry `data-conversation-id`; model buttons carry
`data-model-id` and `aria-pressed`; message articles carry `data-message-index`.
Sidebar selection uses `.conversation-select`; message branch buttons have
accessible labels `Edit as a new branch` and `Regenerate as a new branch`.
`model-quick-switch` values use exported `modelIdentity(model)` (not display names).

## Models and application settings

**Installed** is the local library. Its search filters discovered models and source
labels without contacting a registry. Discovery reads existing local GGUFs,
lma aliases and supported installed Ollama manifests; it does not copy, convert,
download, or load weights. Selecting a library row selects it for the separate
**Load model** action. The chat quick-switch loads the chosen model. Scans are
bounded and can report warnings; an absent entry is not proof that no installation
exists. Application startup may make one attempt to restore the last loaded model
when the saved **Restore the last model when the app opens** setting is enabled
(the default). Later scans do not trigger another restore.

**Delete model** permanently deletes the selected discovered Installed entry after
confirmation naming the model and source. Unload first: deletion is disabled while
**any** model is loaded, status is unknown, a scan is running, or the app is busy.
It never automatically unloads a model. The GUI sends only the selected catalog ID
with `invoke("model_delete", { id: model.id })`; the native adapter revalidates the
catalog, source boundaries and idle state. Shared aliases may lose only their
entry, and shared/external weights may be retained. Result notifications report
retained files and backend warnings. Success rescans the library, clears stale
frontend choices, and preserves another selection where possible, without loading
a model or modifying conversations.

**Store** searches only after an explicit Search or View files action. Searches
send the query/repository ID to the selected public services: Ollama exact
`[namespace/]name[:tag]` lookup, or public Hugging Face GGUF search. “Hugging Face /
LM Studio” is not a separate LM Studio download service. View files resolves a
search result to an exact HF repository; repositories listing multiple GGUFs
require choosing a filename. Single-file results select that filename explicitly.
Unknown sizes are not download-size estimates.

**Download** is explicit and separate from loading. The native adapter checks the
local catalog for an authoritative installed identity before downloading; recognized
Ollama manifests and complete pinned lma installations return existing weights
rather than copying them. Display aliases are not proof of upstream identity.
For an ordinary external HF GGUF without lma provenance, an explicit Download
can still reuse a discovered file with the exact basename and size after its full
SHA256 matches the selected upstream artifact. This fetches upstream metadata,
not weights. Such external files are not marked Installed in Store until source
identity is known; they remain directly usable from Installed without a Store
action. Search and periodic discovery do not hash large model files. New downloads
go under `~/.lma/models`, not
into custom scan folders. Successful downloads refresh Installed but do not start
inference. The current progress indicator is indeterminate, not byte progress.
Neither opening Store nor startup initiates model downloads.

Discovery, registration and Store listing are not compatibility certification.
The loader must support the model architecture and quantization. The Store
selector rejects unsupported split/adapter and projector-only downloads; this is
not a promise that every listed file is downloadable or runnable. Ollama templates,
parameters, system prompts and history are not applied. HF downloads do not
choose a vision projector automatically; vision needs a compatible projector and
a vision-enabled build.

Named lma definitions created with `lma ce NAME -f Modelfile` use the shared
text runtime; there is no Modelfile creation/editor UI here. See the
[Modelfile guide](../../docs/modelfile.md) for inheritance, built-in templates,
parameters and rejection rules. Desktop load policy supplies explicit GPU/context
values, and conversation requests always send max output tokens, temperature and
top-p, overriding those stored defaults. A nonblank seed is explicit too. Chat
uses stored system/history unless a conversation system prompt overrides `SYSTEM`;
raw mode skips stored prompt formatting/history, not sampling defaults.

This lma desktop app is separate from external desktop-client launcher targets.
The terminal launcher can run supported external CLI clients; its external
desktop targets fail closed with setup-required and are not opened/configured.
CLI local-provider overrides are not actual external desktop-client support.

### Settings

Application Settings are separate from per-conversation generation settings.
**Save settings** persists `app-settings.json` in the host-selected app-config
folder and rescans the library; changing a load policy does not reload an active
model. Native `load_model` applies the saved policy:

- **Automatic** uses the compiled GPU default and the shared default context
  (currently 4096), not adaptive hardware tuning or a silent CPU fallback.
- **Defaults file** reads the supplied path, or `~/.lma/config.json` when blank.
  It accepts only `gpu` or `gpu_layers` (not both), and `context`. All are optional;
  omitted fields use automatic defaults. GPU accepts `auto`, `all`, `cpu`, or an
  unsigned layer count; context must be an integer of at least 2. Unknown fields,
  null values, missing files and invalid JSON fail the load before replacement.
  This is not a general CLI/ServeConfig configuration-file loader.
- **Manual** uses the GPU/context controls supplied for the load. The Settings
  values initialize the library's advanced Load options.

Model folders and LM Studio folders are additional scan roots, one per line, with
at most 64 entries combined. The optional Ollama directory adds a custom installed
store; default locations are still searched. These fields do not move weights or
change the download destination. Settings paths must be absolute or begin with
`~/`; native code expands only that home prefix, not environment variables or
shell expressions. Existing malformed or symlinked settings files are not silently
repaired or overwritten. See `desktop/src/SETTINGS_STORE.md` for the IPC schema.

### Import existing weights (no copy)

**Import model** accepts an existing GGUF path or supported installed Ollama name,
an optional alias, optional Ollama store path, and optional projector path. It
validates the existing files' GGUF headers and records canonical paths in the lma
registry; it does not copy, convert, download, or load model assets. Keep the
original files available after import. Import does not establish architecture or
model/projector compatibility. Use absolute file/store paths in this form; its
inputs do not use the Settings path-expansion adapter.

An omitted name reuses a known label where possible, otherwise derives one from
the filename. Conflicting aliases are errors rather than overwrites; an identical
registration is reusable. An explicit custom Ollama store containing the resolved
weights can also be remembered in application settings. After success the UI
rescans and selects the imported entry; loading remains a separate action.

## Local dictation setup

The microphone and `Ctrl/Cmd+Shift+D` add transcription text to the current draft;
results are never sent automatically. The first microphone permission request of
a session requires clicking the microphone. Audio remains in memory on this device.

Opening dictation settings calls `dictation_models` for a read-only local scan.
The installed-model select contains only complete directories with `config.json`,
`tokenizer.json`, a single `model.safetensors`, and `model_type: whisper` (the
currently supported backend). Labels come from `.lma-source.json` repository
metadata, a config name, or the actual directory path. Discovery uses the shared
`default_local_dirs()` roots plus the remembered selected directory, with limits
of 128 roots, 10,000 traversal entries, depth 8, 512 results, and 1 MiB per metadata
file. Nested symlinks and hidden staging directories are not traversed. Scan
warnings are visible; Refresh rescans, and a manual absolute directory remains
available for models outside the scan. Selecting an option alone changes nothing:
**Use selected** validates and remembers the directory through `dictation_set_model`.

There is no bundled repository, model-size estimate, or speech-specific download
location. A user-entered `owner/repository` is sent to `dictation_model_plan` only
after **Inspect download** is clicked. The returned immutable revision, files,
exact byte total, and destination are displayed before a second explicit
**Install** click invokes `dictation_install_model` with that plan. Installation
uses the generic Hugging Face installer, which revalidates the plan upstream.
Changing the repository, closing setup, or an installation error clears consent
and requires inspection again. Inspection checks the required asset layout;
architecture is checked from the installed config before selecting the model.
No model assets are downloaded on startup, discovery, inspection, or capture.

Stable setup IDs include `dictation-installed-models`, `dictation-use-installed`,
`dictation-rescan-models`, `dictation-repository`, `dictation-install-model`,
`dictation-plan`, `dictation-plan-summary`, `dictation-plan-files`,
`dictation-plan-directory`, `dictation-model-path`, and `dictation-save-model`.

## Validation

From the repository root:

```sh
node --test tests/desktop_ui.mjs
```

Tests use a strict mocked DOM and IPC, not a browser server or native build. They
exercise schema/safety, autosave ordering/failure, hydration, restoration, model
selection, whole-response inference, cancellation, branching, images, export and
explicit update confirmation. Regression tests spy on image decoding,
serialization, defensive snapshot reads and DOM replacement during typing; they
also cover draft-only save races, oversized-draft recovery, and unrelated editing
during each native operation. Native layout and platform clipboard/download
behavior still need the launch harness on each supported OS.
