# Local Modelfile definitions

`lma ce` (NSS `lma nss create`) registers a named, text-only model definition over **existing weights**.
It does not download, copy, convert, train, or load the model. Keep the original
weights (and any resolved projector) available after registration. This is a
bounded Ollama-style subset, not arbitrary Ollama Modelfile compatibility.

## Create and preview

For example, with `base.gguf` beside `recipes/Modelfile`:

```text
FROM ./base.gguf
SYSTEM "Be concise and accurate."
TEMPLATE auto
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER seed 42
PARAMETER num_predict 128
PARAMETER num_ctx 4096
MESSAGE user "How should you answer?"
MESSAGE assistant "Concisely, with uncertainty stated."
LICENSE "Local license metadata"
```

```sh
lma ce concise -f recipes/Modelfile -d
lma ce concise -f recipes/Modelfile
lma nss create concise-nss --file recipes/Modelfile --dry-run
```

Root `-f` (NSS `--file`) defaults to `./Modelfile` in the current directory, **only for an
explicit create command**. Other commands do not discover a nearby Modelfile.
Root `lma ce` uses `-o DIR` for the Ollama store, `-d` for dry-run, and `-h` for help.
Long commands and flags are NSS-only: `lma nss create` retains legacy shortcuts
such as `-f` and `-od`; NSS `--ollama-dir DIR` / `-od DIR` selects the store.

A dry run parses and validates the file, resolves `FROM`, merges inherited
metadata, and prints JSON with the resolved weights/projector and definition.
It performs metadata/header reads, but no registry writes, downloads, weight
loading, inference, or client execution. Real creation atomically publishes the
alias and definition in `~/.lma/models/.registry-v1.json`. Names normalize to a tag
(`concise:latest`); existing aliases are errors, not overwrites. Mutations use a
persistent `.registry-v1.lock` sidecar to protect concurrent CLI/server writes. A
busy registry returns a retryable error; reads and dry runs do not create locks.

## FROM and inheritance

Exactly one `FROM` is required. The CLI accepts an installed model name (including
an lma definition or `ollama:NAME`) or an existing local model file. Relative
files resolve from the **Modelfile's directory**, not the command's working
directory. URLs and automatic downloads are unsupported; install/import weights
first. Use an explicit relative path such as `./base.gguf` for local files.

When `FROM` selects another registered lma definition, omitted `SYSTEM`,
`TEMPLATE`, `LICENSE`, and individual parameters inherit from that definition;
child values override them. Parent `MESSAGE` records precede child records.
Inheritance is merged at creation, not a live link to later parent changes. An
installed Ollama model supplies weights/projector, not its Ollama Go template,
system prompt, parameters, or message layers.

Select the registered name to use its defaults, for example
`lma s lma:concise:latest`. Opening the weights by root `-p` (NSS `--path`) is not selecting
that named definition. Registration validates references/headers, not native
architecture compatibility or model quality.

## Supported directives

| Directive             | Meaning                                                                     |
| --------------------- | --------------------------------------------------------------------------- |
| `FROM MODEL`          | Exactly one installed selector or existing local file (CLI only for files). |
| `SYSTEM TEXT`         | Default system prompt for chat-formatted requests.                          |
| `MESSAGE ROLE TEXT`   | Initial text history; role must be `system`, `user`, or `assistant`.        |
| `LICENSE TEXT`        | Stored metadata only; no license enforcement or training.                   |
| `TEMPLATE NAME`       | Select an allowlisted lma template name, not template source code.          |
| `PARAMETER KEY VALUE` | One of the typed parameters below.                                          |

`TEMPLATE` accepts `auto`, `native`, `plain`, or supported built-in names such as
`llama3`, `chatml`, and `gemma4`; see `lma h s` (NSS `lma nss help serve`) for the supported names.
Custom Ollama **Go template programs are rejected**, including `{{ ... }}` text.

Directives are case-insensitive; parameter keys and message roles use the
lowercase spellings shown. Repeated scalar directives/parameters use the last
value (every occurrence must validate); `MESSAGE` appends. Unquoted `#` starts a comment. Single/double-quoted
values and literal triple-double-quoted multiline `SYSTEM`, `MESSAGE`, `LICENSE`,
and `TEMPLATE` values are supported (TEMPLATE must still resolve to an allowed
name). Files must be UTF-8, contain no NUL, and fit 1 MiB; definitions are also
bounded to 1 MiB serialized and 4096 messages, including inherited content.

| Parameter     | Accepted value / use                                                  |
| ------------- | --------------------------------------------------------------------- |
| `temperature` | Finite 0–2; 0 selects greedy sampling.                                |
| `top_p`       | Finite, greater than 0 and at most 1.                                 |
| `seed`        | Unsigned 32-bit integer.                                              |
| `num_predict` | Positive unsigned 32-bit output-token limit.                          |
| `num_ctx`     | Context tokens, 2–2147483647.                                         |
| `num_gpu`     | Unsigned 32-bit GPU-layer count; 0 means CPU.                         |
| `num_thread`  | CPU threads, 1–2147483647.                                            |
| `num_batch`   | Logical batch tokens, 1–2147483647.                                   |
| `u_batch`     | Physical micro-batch tokens, 1–2147483647; cannot exceed `num_batch`. |

Unknown directives, `ADAPTER`, unsupported parameters (including `stop`, `top_k`,
and `repeat_penalty`), training, and shell commands fail rather than being ignored.
This does not add new weight formats or multimodal directives.

## Runtime precedence

- Explicit root hosting flags win over stored defaults: `-ce`, `-c`, `-g`,
  `-t`, `-b`, and `-mh` override `TEMPLATE`, `num_ctx`, `num_gpu`, `num_thread`,
  `num_batch`, and `u_batch`, respectively. NSS equivalents are `--chat-template`,
  `--context`, `--gpu`, `--threads`, `--batch`, and `--micro-batch`. Even an explicitly supplied normal default wins.
  Otherwise stored values take precedence over lma defaults. Effective batch and
  context/output constraints are validated; stored `num_predict` must fit context.
- For HTTP generation, stored `temperature`, `top_p`, `seed`, and `num_predict`
  fill **omitted** request fields. Explicit supported request sampling/output
  fields win, including zero temperature/seed. Ollama uses `options.num_predict`;
  OpenAI-style routes use `max_tokens` (Chat also accepts `max_completion_tokens`).
  Invalid/null fields do not inherit silently, and Messages still requires its
  request `max_tokens`. Clients that always send sampling fields override those
  defaults; frontend controls are not automatically populated from the definition.
- A request's explicit system instructions replace the default `SYSTEM`, but do
  not remove stored `MESSAGE` history. `/api/generate` with `system:""` explicitly
  clears the default. Stored messages precede request history after any leading
  request system instructions. Prompt token counting uses the same stored text.
- Raw completions and Ollama `raw:true` keep the exact prompt: no stored system,
  messages, or chat-template formatting. Sampling defaults still apply.
- The CLI HTTP host applies the selected alias's definition when switching models;
  explicit host load flags remain authoritative. Request `options.num_ctx` must
  match the loaded context, not resize it.

## HTTP creation and frontend boundaries

`POST /api/create` accepts either structured `from`, `system`, `template`,
`parameters`, `messages`, and `license` fields, or inline `modelfile` text, never
both forms together. **HTTP FROM accepts installed selectors only**, not local or
relative paths or URLs. Use the CLI to register local files. See the
[Ollama API contract](ollama-api.md#administration) for examples and copy behavior.

The lma desktop uses the shared text runtime for named definitions, but has no
Modelfile creation/editor UI. Its load and generation controls can override stored
defaults; see [desktop notes](../desktop/ui/README.md#models-and-application-settings).
Separately, launcher support for external **CLI** clients does not imply external
desktop-client support: those desktop targets fail closed with setup-required,
without opening/configuring them. See [client connections](clients.md).
