# Storage policy

## Default lma-owned state

New application-managed persistent state belongs under **`~/.lma/`**, independent
of the working directory. `HOME` must be absolute; on Windows, `USERPROFILE` takes
precedence, with `HOME` as a fallback. Missing or relative home values cause an
error rather than silently writing into the checkout or working directory.
Resolving a default path does not create it.

| Data                                                                    | Default location                                                                                 |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Downloaded models and their source metadata                             | `~/.lma/models/`                                                                                 |
| Model aliases and Modelfile definitions                                 | `~/.lma/models/.registry-v1.json`                                                                |
| Registry mutation lock                                                  | `~/.lma/models/.registry-v1.lock`                                                                |
| Learned expert-routing frequencies and locks                            | `~/.lma/expert-cache/`                                                                           |
| Optional defaults configuration                                         | `~/.lma/config.json`                                                                             |
| Source installation registration                                        | `~/.lma/source-install.json`                                                                     |
| Desktop settings, saved model selection, dictation selection, workspace | `~/.lma/app-settings.json`, `model-preferences.json`, `dictation.json`, `desktop-workspace.json` |
| Desktop app config/data/local-data APIs                                 | `~/.lma/`                                                                                        |
| Desktop app cache/log APIs                                              | `~/.lma/caches/`, `~/.lma/logs/`                                                                 |

Expert history stores routing counts, not prompts or duplicate model weights.
The defaults configuration is read only when requested by the desktop load policy;
the CLI does not automatically apply this JSON configuration. The desktop directory
initialization changes storage locations only, not GUI features or load behavior.

The shared path helper is `lma::models::storage`. No new top-level library export
is required.

## Explicit paths and output artifacts

Explicit caller paths remain authoritative. This includes fetch output roots,
import/quantization/image outputs, `LMA_EXPERT_CACHE_DIR`, host-selected desktop
app-directory overrides, and the debug-only `LMA_DESKTOP_TEST_DIR` isolation
setting. An explicit output may be relative; it is not silently rebased under
`~/.lma`. Import receipts, download metadata and atomic-publication temporary
files stay beside the selected output/store. Registry APIs likewise retain their
explicit root argument.

Model scan overrides such as `LMA_MODELS` select **read/discovery locations**, not
a replacement default download destination. CLI image generation's existing
`image.png` output is a user-facing export artifact, not hidden application state;
its behavior is unchanged. Import and quantization require caller-selected output
paths. No existing exports are relocated.

## Existing state stays readable, without migration

The default desktop directory reads a missing individual state/registration file
from its former OS-specific config location:

- macOS: `~/Library/Application Support/org.lma.app/`
- Linux: `$XDG_CONFIG_HOME/org.lma.app/`, or `~/.config/org.lma.app/`
- Windows: `%APPDATA%/org.lma.app/`

This fallback covers settings, preferences, dictation selection, workspace and
`source-install.json`. Each reader retains its existing size, regular-file and
format checks. An existing new-location file wins even when invalid or a dangling
symlink: fallback must not conceal a file requiring repair. Explicit host/test
directories do not fall back to real user settings. Subsequent saves target the
selected new directory, never the legacy file. There are **no automatic copies,
moves, deletions, recursive migrations, or merges**. Legacy files remain present
and may describe models that are no longer installed.

Model discovery still searches the legacy desktop data directory's `models/`
subdirectory, as well as `~/.lma/models`, `./models`, explicit scan roots and
external stores. Linux's legacy data directory is `$XDG_DATA_HOME/org.lma.app/`
or `~/.local/share/org.lma.app/`. Discovery does not create these directories.

## External weights and source identity

Installed Ollama and LM Studio weights are used **in place**:

- Ollama: explicit store, then `OLLAMA_MODELS`, then `~/.ollama/models`.
- LM Studio: `LM_STUDIO_MODELS`, `~/.lmstudio/models`, and the legacy
  `~/.cache/lm-studio/models` root.

The core canonicalizes known LM Studio roots and labels local files inside them
`lm-studio`, including files reached through registered lma aliases. This label
describes the weights' location, not ownership of the alias. IDs such as
`lma:my-profile:latest`, projector choices and registry definitions remain intact.
A similarly named sibling directory is not inside a root.

Deletion must preserve that distinction. An exact external alias ID selects the
**registry alias**, never the LM Studio weights. `-s lm-studio` combined with that
alias ID is refused; deleting actual external weights requires a distinct physical
file ID or unambiguous installed label and the external source. Existing managed
lma alias deletion rules are unchanged: unshared managed weights may also be
removed. See [deletion](deletion.md) before using any destructive command.
Source labeling and normal discovery never move, copy or delete external weights.

Missing registered targets and unsupported Ollama entries remain real discovery
warnings, not hidden rows presented as runnable models. Missing-target warnings
identify the path and offer restoration or explicit stale-alias removal. Unsupported
entries (for example an unsupported `phonon` manifest) identify the Ollama model,
store and original failure; use its source application or a supported explicit
conversion rather than expecting discovery to download/repair it.

## Boundaries and remaining exceptions

This policy covers core application state and desktop app-path initialization,
not a relocation of the software installation or operating-system infrastructure:

- Executables, application bundles, launcher entries, source checkouts, Cargo build
  outputs and package-manager caches retain their installation/build locations.
  Source installation registration defaults to `~/.lma/source-install.json`;
  explicit `--config-dir` overrides remain supported. Legacy registrations remain
  readable by the application.
- Browser chat storage, browser profiles and macOS WebKit data, OS permission
  records, crash reports, swap, GPU/Metal/CUDA driver caches and filesystem caches
  are controlled by the browser, OS or driver. App cache/data overrides are not a
  guarantee that those systems write nothing elsewhere.
- Short-lived OS temporary files/directories (including isolated external-client
  runtime state) remain temporary, not a persistent lma model store. Atomic model
  publication/deletion staging stays on the target filesystem for safety.
- Explicitly launched external clients own their own configuration and storage.
  Their files are not moved into `~/.lma`.

No real user files need to be modified to validate these rules: regression tests
use temporary fixtures for default path validation, read-only legacy fallback,
source labeling, missing/unsupported warnings and alias deletion safety.
