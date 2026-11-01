# Delete installed models

`lma d MODEL` deletes one installed model through the local deletion API. It can
permanently remove model files; it is not an unload command. Root hosting is
`lma s`, stop/unload is `lma sp`, and deletion is `lma d`. Root accepts **only
canonical short commands and flags**. Long syntax belongs under `lma nss ...`.

## Before deleting

1. Inspect installed names, IDs, and sources with `lma l` or `lma l -j`.
2. Unload the selected model with `lma sp -m MODEL` on its running local managed
   host, or unload it in the desktop app. Stop its use in other processes too.
3. Choose an exact label/ID and, when appropriate, an explicit source guard.

**Deletion does not automatically unload anything**, contact an HTTP host, or
stop another process. The CLI cannot prove whether some other process still has
the weights loaded. `lma sp` only unloads from the selected managed host; it is
not a global process check. Deletion itself is local and performs no network
queries or downloads.

## Commands and flags

```sh
# Preview the supplied arguments only; no model resolution or filesystem changes.
lma d MODEL -d

# Interactive deletion: requires confirmation on terminal stdin.
lma d MODEL

# Explicit source and confirmation bypass (also suitable for non-interactive use).
lma d MODEL -s lma -y

# Select an Ollama source and a specific store.
lma d MODEL -s ollama -o /path/to/ollama/models -y

# LM Studio source guard.
lma d MODEL -s lm-studio -y

# Local help.
lma h d
```

| Root option           | Meaning                                                                     |
| --------------------- | --------------------------------------------------------------------------- |
| `MODEL` or `-m MODEL` | Exactly one installed model label/ID; do not supply both.                   |
| `-s SOURCE`           | Source guard: `lma`, `ollama`, or `lm-studio` (exact lowercase values).     |
| `-o DIR`              | Ollama store directory; **not** a source selector.                          |
| `-y`                  | Skip confirmation; required for real deletion when stdin is not a terminal. |
| `-d`                  | Argument-only JSON preview; no resolution or deletion.                      |
| `-h`                  | Help.                                                                       |

Multi-letter root flags are exact names, not bundles. Duplicate options, multiple
models, unknown flags, and bare `--` are rejected. For a value starting with `-`,
use an inline form such as `-m=-model`. Labels and discovered IDs, including
source-prefixed IDs, are passed unchanged to the deletion API. Ambiguous matches
are refused: use `-s` to distinguish sources and an exact label/ID if names still
collide within one source. A store override alone does not restrict selection to
Ollama.

Long-form equivalents are **NSS-only**:

```sh
lma nss delete MODEL --source lma --dry-run
lma nss delete --model=MODEL --source=ollama --ollama-dir=/path/to/store --yes
lma nss delete MODEL --source lm-studio --yes
lma nss help delete
```

NSS deletion accepts `--model` / `-m`, `--source`, `--ollama-dir`, `--yes`,
`--dry-run`, `--help` / `-h`, and `--help-all`. Do not assume root `-s`, `-o`,
`-y`, or `-d` are NSS option aliases; use the documented long options there.

## Confirmation and preview are not a deletion plan

Without `-y`, real deletion requires terminal stdin. Only `y` or `yes`
(case-insensitive, with surrounding whitespace ignored) confirms; any other
response or EOF cancels. A pipe containing `yes` does not satisfy terminal
confirmation. For scripts, explicitly pass `-y` (NSS `--yes`) after checking the
selector/source yourself.

**`lma d MODEL -d` prints parsed arguments, not a resolved deletion plan.** It
neither checks whether the model exists or is deletable nor lists the files that
would be removed or retained. It does not contact a host, load weights, perform
source discovery, or ask for confirmation. Adding `-y` to a preview still does
not delete anything. Real confirmation also precedes model resolution: confirming
a label is not approval of an already-enumerated file list.

## What can be removed or retained

The existing deletion API applies source boundaries, regular-file checks, and
bounded discovery/reference scans. It is not an arbitrary-path removal tool.

- **lma local GGUFs:** known managed model paths can be removed, including a
  validated complete conventionally named split-GGUF set. Valid matching import
  receipts can be removed with imported output files. Partial/unsafe split sets
  and unsafe paths are refused.
- **lma registry aliases:** the alias is removed; its weights can also be removed
  if managed and not shared with another alias. Shared weights and external
  targets (including Ollama or LM Studio assets) are retained. An alias does not
  confer ownership of arbitrary external files. Deleting a Modelfile profile is
  therefore **not always alias-only**, nor does it always remove weights.
  Projector targets of aliases are not automatically deleted with the alias.
- **Ollama:** the selected manifest is removed. Source blobs are removed only when
  a bounded reference scan determines they are not shared with other manifests,
  including nested manifest references. If that scan is incomplete or unsafe,
  the manifest can still be removed while all source blobs are retained with a
  warning. This is not a whole-store orphan sweep. Original Ollama assets and
  separately imported GGUF outputs are independent deletion targets.
- **LM Studio:** deletion is restricted to known LM Studio model roots, including
  complete validated split sets. Removing an lma alias pointing there retains
  those external weights; selecting `lm-studio` is a distinct destructive action.

When physical model files are deleted, lma also removes aliases in its trusted
registry roots whose **primary weight targets** were removed by that operation.
This includes aliases to any removed split shard and to unshared Ollama blobs.
References are snapshotted with canonical targets before unlinking, so legacy
symlink references can be matched afterward. Aliases to retained shared blobs or
separately imported GGUF outputs are kept, as are unrelated stale aliases and
aliases that only reference a deleted asset as a projector. Projector files and
unrelated model definitions are not deleted by registry cleanup.

Alias-only deletion removes just the selected registration when weights are
shared, external, or already missing. It does not prune sibling aliases. To remove
an alias after deleting its GGUF outside lma, use `lma d lma:NAME:TAG -s lma -y`.
Discovery itself never performs a global stale-alias/orphan sweep.

The command is not an expert-history purge. Do not assume model deletion erases
routing-count snapshots under `~/.lma/expert-cache/`; see the
[expert-cache guide](expert-cache.md#learned-history-under-lma) for history removal
with the model unloaded.

## Result and limitations

Success prints pretty JSON with `status`, `model`, `source`, `aliasDeleted`,
`deletedFiles`, `deletedBytes`, `retainedFiles`, `blobGcComplete`, and `warnings`,
plus `deletedAliases` and `registryCleanupComplete`.

- `aliasDeleted` means the **selected target was an alias and was removed**. It
  stays `false` for physical LM Studio/local/Ollama deletion even when that
  operation cleans up other aliases.
- `deletedAliases` lists every registration removed by the operation as an object
  with `root`, `alias` (normalized name/tag), and `id` (`lma:NAME:TAG`). The root
  distinguishes identical names in different registries.
- `registryCleanupComplete` is `false`, with an explicit warning, if files were
  deleted but registry cleanup could not finish (for example, a registry is busy
  or cannot be published). File deletion is not undone at this point. Inspect the
  warnings and remove remaining dead registrations explicitly.

Registry cleanup re-reads entries under the existing cross-process mutation
locks and preserves unrelated definitions. Repointed aliases and recreated/live
targets are retained; changed missing entries are not blindly removed. Only
`NotFound` is treated as absence, never permission or other inspection errors.
Inspect the paths and warnings: successful alias/manifest removal can retain
weights, and incomplete blob garbage collection does not imply all bytes were
reclaimed. On Ollama reference-scan failure, retained blobs may be described in
warnings rather than fully enumerated in `retainedFiles`.

The implementation stages selected files and attempts rollback on failures before
commit (including selected-alias publication failures); subsequent physical-file
registry cleanup is a separately locked metadata update. Neither step provides
an atomic transaction across all registry roots, a crash-proof transaction, or
an undo facility. Keep backups if needed. Files still mapped by another process may remain in use or consume
storage, and platform file-locking rules can cause deletion to fail. No automatic
unload, secure erasure, immediate free-space guarantee, or cross-process safety
certification is provided.
