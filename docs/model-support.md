# Model support

lma uses the pinned `llama-cpp-2 = 0.1.160` backend for GGUF language models.
“Dense” and “MoE” are model architecture categories, not interchangeable file
formats. An architecture/quantization must be supported by that backend, and
weights plus runtime buffers must fit the selected memory configuration. This
is not a claim that all models or all Ollama artifacts are supported.

| Model/artifact                            | Current support                                                                                                                                                         | Validation                                                                                                      |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Native GGUF dense language models         | Load directly; embedded template preferred, known-format/plain fallback when absent                                                                                     | Real imported Gemma4 12B chat, tools, token counting, Ollama chat/generate                                      |
| Native GGUF MoE language models           | Native expert routing; same loader/API as dense                                                                                                                         | Real Mellum2 12B A2.5B Q4_K_M counting and streamed chat without an embedded template                           |
| Named split GGUF                          | Complete `name-00001-of-000NN.gguf` sets; any selected part normalizes to first, no merge/copy                                                                          | Dense/MoE fixtures, discovery, missing/mismatched shards and corrupt header checks; no real split-model run yet |
| Installed Ollama GGUF                     | Read manifest and blobs in place; optional projector                                                                                                                    | Local resolver fixtures and real installed tags; Ollama template/system/parameters are not applied              |
| Ollama MLX Gemma4 NVFP4 safetensors       | Explicit `lma it` (NSS `lma nss import`); dense text subset only; original tag reuses import receipt                                                                    | Real local `gemma4:12b` import and inference; source store unchanged                                            |
| Local HF safetensors text profiles        | Explicit import of validated LlamaForCausalLM, non-sliding MistralForCausalLM/MixtralForCausalLM, standard Qwen2ForCausalLM configurations and tokenizers; F32/F16/BF16 | Synthetic architecture/tokenizer/shard fixtures; no blanket real-model claim                                    |
| Other safetensors/PyTorch language models | Not directly loadable; no arbitrary architecture/quantization import                                                                                                    | Unsupported profiles are rejected with format guidance                                                          |
| MoE/shared-KV Gemma4 safetensors import   | Unsupported by current importer; this does not exclude native compatible GGUF MoE                                                                                       | Import preflight rejects unsupported layouts                                                                    |
| Vision input                              | Compatible GGUF plus projector, `vision` feature, PNG/JPEG                                                                                                              | Adapter/validation fixtures; model/projector-specific E2E still required                                        |
| Speech recognition                        | Local Hugging Face Whisper safetensors directory, `speech` feature                                                                                                      | WAV/resampling/mel/decoder-cache and validation tests; real speech transcription not yet verified               |
| Image generation                          | Local SD1.5 Diffusers safetensors, CLI only, `image-gen` feature                                                                                                        | Validation/output tests; real generated image not yet verified                                                  |

## Split GGUF

Keep all parts in one directory, with their original five-digit split names:

```sh
lma s -p /models/model-00001-of-00003.gguf
# Selecting another part resolves to the same first part:
lma c -p /models/model-00002-of-00003.gguf
```

Discovery lists a complete set once and reports the combined part size. It warns
and excludes invalid/incomplete sets. Preparation checks split indices, counts,
architecture agreement when present, tensor totals and basic header/offset bounds
before a desktop host unloads its current model. The native loader performs the
actual weight-format validation and loads the parts directly.

Inspection accepts little-endian GGUF v2/v3. Limits: 256 parts, 128 MiB per header,
256 MiB combined headers, 100,000 tensor descriptors per part, 16,384 metadata
entries per part. These bounds can exclude unusually large valid artifacts.
Ollama manifests containing multiple model layers are still unsupported: this
support is for local named split sets, not arbitrary blob-layer merging. Fetch
and split-output quantization have not gained split-set support.

## Memory and GPU

GPU builds keep the requested offload settings; there is no silent CPU retry.
Root hosting flag `-mk mmap` (NSS `--memory-fallback mmap`) explicitly permits one warned CPU/pageable-weight retry
on eligible initialization failures. It does not make NVMe into VRAM, configure
swap, rescue native aborts/OOM kills, or guarantee model fit. Root hosting flag `-g all` (NSS `--gpu all`) is not
automatic memory fitting. Actual placement is visible in native buffer/offload
logs; compiled features alone are not proof.

Real checks on a 16 GiB Apple Silicon host used **one GPU layer**, small context
and batches, and one native model process at a time. They establish basic
compatibility, not all-GPU performance or production load capacity.

## APIs

See [Ollama protocol support](ollama-api.md) and [client connections](clients.md).
Ollama `/api/chat` and `/api/generate` stream NDJSON by default. OpenAI and
Anthropic subsets are also available. Native embeddings use `/v1/embeddings`,
`/api/embed`, and legacy `/api/embeddings` with normalized vectors and exact
model-token usage, through the same single-admission worker. Model/backend
capabilities still matter; there is no arbitrary embedding-model guarantee.
The CLI HTTP host also supports single-resident dynamic selection, keep-alive,
stop/unload, public GGUF pull, alias copy/create, and source-specific deletion.
Push, arbitrary Modelfiles, training, and blob uploads remain unsupported.
Client connection recipes are not installed-client end-to-end certification.

## Local speech and dictation

```text
models/SPEECH_MODEL/
  config.json
  tokenizer.json
  model.safetensors
```

Use standard Hugging Face Whisper weights (for example `openai/whisper-tiny`),
not whisper.cpp GGML `.bin` or chat GGUF. Assets are local and single-file
safetensors; lma does not download them automatically.

```sh
lma t -m models/SPEECH_MODEL recording.wav
lma t -m models/SPEECH_MODEL recording.wav -l en -j
lma t -m models/SPEECH_MODEL recording.wav -g off
# Long commands/flags are NSS-only:
lma nss transcribe -m models/SPEECH_MODEL recording.wav
```

WAV: PCM16/float32, mono/stereo, 8–192 kHz, at most 30 minutes and 512 MiB.
Audio is resampled in process to mono 16 kHz. Decoding is greedy transcription,
not translation; JSON segment times are chunk boundaries, not word alignment.
Root transcription flag `-t` (NSS `--threads`) controls spectral preprocessing, not Candle's process-wide inference
pool. GPU defaults on only with an explicitly compiled speech GPU backend;
`-g on` (NSS `--gpu on`) errors if unavailable rather than silently using CPU.

Desktop: click the microphone or press **Ctrl/Cmd+Shift+D** to start/stop.
Choose an installed compatible speech model, or inspect a Hugging Face
owner/repository and explicitly install the displayed files/size/revision. There
is no compiled-in model choice. Downloads go beneath `~/.lma/models/huggingface/`;
only the selected local directory is saved. Captured audio is
bounded to 120 seconds, sent to the Rust backend in process, and not saved or
uploaded. Recognized text is appended to the draft, never automatically sent.
Microphone permission must be granted. Native microphone capture and real
speech transcription still require end-to-end validation; UI/component tests
are not a substitute. Concurrent LLM and Whisper allocations also need memory.

macOS source/Tauri bundles include a microphone usage description. Sandboxed
redistributions additionally require the appropriate audio-input entitlement;
the source installer does not produce an App Store sandboxed application.
