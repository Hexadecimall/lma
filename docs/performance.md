# Runtime performance

The current engineering target is **2–3× Ollama throughput** on representative
text workloads, with identical GGUF bytes, context, GPU placement and sampling.
This replaces the earlier 1.4× target; **neither target is achieved or guaranteed**.
Expert-cache priorities should improve reuse on repeated workloads, but learned
frequency alone does not prove faster inference or monotonic improvement. Measure
cold history, warmed history, and workload changes separately, including startup,
transfer bytes, misses/evictions, latency and throughput. Keep cache-off inference
unchanged and test unrelated workloads for regression. Separate model loading,
prompt processing, first-text latency, and decode speed. Faster HTTP responses are
not proof of a faster engine; repeatedly testing an identical cached prompt is not
proof of faster conversational inference.

## Current improvements

- Text-only requests reuse the **longest common token prefix (LCP)** in KV,
  including changed suffixes and shortened prompts. The final prompt token is
  evaluated again when
  needed to refresh logits. Native residency and suffix-removal checks still gate
  reuse; images and recurrent/hybrid models do not enter this rollback path.
- Equivalent installed selectors resolve to a stable resident identity before a
  switch. Matching weights, projector, store, and configuration keep the existing
  worker/router/cache. The exact resident public name retains its fast path without
  a catalog scan. That name addresses the loaded snapshot until unloading/switching;
  an unfamiliar selector is resolved and checked, not blindly trusted as an alias.
- Native generation callbacks borrow UTF-8 stream deltas. Nonstream generation
  avoids allocating an owned string for each discarded callback; streaming owns
  the delta at the channel boundary, tries the writable queue first, and blocks
  only on backpressure.
- Temperature-zero generation **without a grammar** selects directly from borrowed
  logits, avoiding a per-token candidate-array copy. Strict comparisons preserve
  native greedy tie/NaN behavior, and the sampler accepts the selection exactly
  once. Grammar-constrained and non-greedy requests retain the sampler path.
- For ordinary attention, reset first attempts **logical KV invalidation** and
  verifies native state is empty; failed/unsupported removal or recurrent/hybrid
  state uses the full-clear fallback. Clean state skips redundant clearing.
  Logical invalidation does **not** erase old KV bytes, free the context's KV
  allocation, reduce its reserved memory, or provide secure data erasure.
- Generation uses `with_swa_full(false)` rather than retaining full sliding-window
  attention (SWA) history solely to improve cache reuse. A prefix that has rolled
  out of native memory must be prefilled again. This preserves the bounded SWA
  memory policy; it does not promise every warm-prefix request a cache hit.

These remove unnecessary work. Model-free regressions establish the policies and
lifecycle behavior, **not a measured tokens/second improvement or native cache-hit
rate**. Separately, an opt-in native regression without the compact expert cache also passed on the already-installed
Mellum2-12B-A2.5B Base Q4_K_M GGUF using one CPU/pageable-mmap instance, context
512, batch/micro-batch 32, four threads, F16 KV, and flash attention off. It checked
exact greedy text/usage equality with caching on/off for identical, extended,
changed-suffix, and shortened prompts; real native prefix residency/removal;
logical versus full reset; shortcut versus native greedy selection; and
cancellation recovery. This is a short correctness test, not a representative
throughput benchmark, SWA rollover test, or bounded-RSS guarantee. The scoped
HTTP experiment below measures lma against the installed Ollama build; it does
not establish general engine parity or FreeToken parity.

## Local Ollama comparison — 2026-10-31

Apple M5, the same installed Mellum2-12B-A2.5B Base Q4_K_M blob
(`f30c7d928a39f5278b01f9c4a7d74ee2a88a24f887bb196a3033abe625a8381b`),
Ollama 0.40.12 and the current full-feature lma release build. Both used context
1024, four threads, batch/micro-batch 16, all GPU layers, F16 KV and flash off.
Native logs verified the same weight path, 29/29 offloaded layers, context,
batching and flash setting. Expert-cache mode kept original mmap expert sources
and used 1 GiB of compact GPU slots; it is not the same memory placement as full
residency. No total-RSS comparison was made.

The raw-text coding workload independently primed identical, extended and
changed-suffix prompt transitions. Three repetitions per engine produced 64
output tokens each. Loading/warming requests were excluded. These are **client
end-to-end token rates for warm-prefix requests**, not isolated decode speeds.
Engines ran sequentially on temporary isolated loopback ports, with no downloads.

Repeat run, reverse engine order and Ollama debug logging off:

| Workload         | Ollama tokens/s | lma, expert cache off | lma, 1 GiB expert cache |
| ---------------- | --------------: | --------------------: | ----------------------: |
| Identical prompt |           49.30 |                 51.43 |                   32.09 |
| Extended prompt  |           47.69 |                 49.68 |                   30.33 |
| Changed suffix   |           48.90 |                 50.66 |                   31.46 |

Cache-off lma was **3.6–4.3% faster** in this repeat; the 1 GiB expert-cache path
was **34.9–36.4% slower** than Ollama. Cache-off median first-text latency was
19.68 / 61.64 / 37.28 ms versus Ollama's 63.93 / 110.96 / 71.90 ms respectively.
Every measured workload produced identical output hashes and token counts across
engines and repetitions: **54 measured responses across both runs**.

The first run (Ollama first, debug logging on) measured cache-off lma at
69.61–71.22 tokens/s against 46.74–48.70 for Ollama. That larger advantage did
**not** hold in the reverse-order repeat. Timing variation and the logging/order
change prevent attributing the gap to one cause or claiming a stable 46–49%
speedup. Neither run covers cold prompts, long-context workloads, an optimized
flash-on configuration, oversized-model paging, or other models/hardware.

The cache-enabled workload uses local batches no larger than 16, so its slowdown
is present **without larger-batch CPU expert fallback**. Its final counters were
174,100 bundle hits, 123,556 misses, 123,296 evictions and 512,223,473,664 raw
source bytes copied. Those copies are not physical SSD-read measurements. The
small-cache transfer/admission path needs optimization; it is not currently a
resident-model speed upgrade. Profiling is needed to attribute its costs.

Local experiment scripts, logs and raw reports are in
`target/ollama-speed-check/` and `target/ollama-speed-check-repeat/` (untracked
artifacts). The bounded orchestration scripts start and clean up only their own
server process groups; `tests/benchmark.py` itself still never starts processes.

### Follow-up after Metal readback/upload changes

A later sequential run with the same model/settings, including borrowed Metal
source buffers and coalesced demand uploads, measured:

| Workload         | Ollama tokens/s | lma, cache off | lma, 1 GiB cache |
| ---------------- | --------------: | -------------: | ---------------: |
| Identical prompt |          66.755 |         70.922 |           33.297 |
| Extended prompt  |          64.485 |         68.790 |           31.332 |
| Changed suffix   |          66.748 |         69.983 |           32.662 |

All **27 measured responses** match output hashes, token counts and finish reasons
by workload. Cache-off ratios are **1.048–1.067×**, and cache-enabled ratios are
**0.486–0.499×**. Both engines varied substantially from the earlier run, so these
numbers do not isolate the gain from individual changes. They do **not** meet the
2–3× goal, and the cache is still a memory/reuse tradeoff, not a proven faster mode.
Artifacts: `target/ollama-speed-optimized/comparison.json` and adjacent logs/script.
These remain warm-prefix, short-output HTTP measurements, not general decode or
cold/oversized-model benchmarks.

The Metal changes memoize duplicate fusion checks within one encoder invocation,
avoid an extra small shared-buffer readback blit after producer synchronization,
and avoid empty command-buffer submissions. Expert uploads borrow existing model
buffer registrations rather than registering the same large mapping again. The
observed Mellum binding covers 7,324,434,432 source bytes; **1 of 28 layers retains
generic uploads**. Transfer planning batches available misses and publishes cache
identities only after completion; CPU routing and synchronization remain.

The native test suite, including shared readback reuse/offset coverage, passes
**26/26**. This does not resolve the documented 2 GiB fusion-normalized real-model
logit mismatch. The n-gram candidate-history helper is currently a **test-only
prototype**, not integrated speculative decoding or a delivered speedup.

## Reproducible HTTP comparison

`tests/benchmark.py` uses Python's standard library and the raw streaming
`POST /api/generate` contract implemented by lma and Ollama. It does not start
processes, invoke inference subprocesses, download weights, unload models, or send
`keep_alive` overrides. Ordinary server expiry policies still apply.

Prepare one local text file and use that exact file for both engines. Use the same
GGUF file hash/quantization and verify context, batch, micro-batch, threads, flash
attention, GPU placement, KV type, and build features independently. Record these
in `--notes`; the script cannot infer or enforce all backend settings. Do not run
two large engines concurrently on a memory-limited machine.

Example command shape (replace `MODEL`, `PROMPT_FILE`, and result filenames):

```sh
# Network-free plan. No model/server access without --run.
python3 -B tests/benchmark.py \
  --base-url http://127.0.0.1:8080 --model MODEL --label lma \
  --prompt-file PROMPT_FILE --context 4096 --max-tokens 128 --runs 3

# Run only against a model already resident on this server.
python3 -B tests/benchmark.py \
  --base-url http://127.0.0.1:8080 --model MODEL --label lma \
  --prompt-file PROMPT_FILE --context 4096 --max-tokens 128 --runs 3 \
  --run --output lma-result.json

# After manually arranging the same model/settings in Ollama, run sequentially.
python3 -B tests/benchmark.py \
  --base-url http://127.0.0.1:11434 --model MODEL --label ollama \
  --prompt-file PROMPT_FILE --context 4096 --max-tokens 128 --runs 3 \
  --run --output ollama-result.json
```

Without `--allow-load`, the exact model name must appear in `/api/ps` **and its
`context_length` must be a known positive JSON integer equal to `--context`**.
Missing, null, malformed, or mismatched context fails before any generation request;
otherwise sending `num_ctx` could silently cause a reload/reconfiguration. The
script does not guess aliases. Every supplied `model` field in generation frames,
including priming and terminal frames, must match the requested exact name. Omitted
response identities are accepted, not treated as independently verified identity.

`--allow-load` explicitly permits requests despite unknown/different resident
context, or for a nonresident name confirmed in `/api/tags`. Loading/reconfiguring
can evict another resident model according to the server's policy, so only use that
option when you intend the lifecycle change. This opt-in does not prove the server
honors `num_ctx`; verify effective settings separately. Preflight is a snapshot,
not a reservation: keep other clients and model/context changes out of the run.
Reports are created exclusively and cannot overwrite an existing result. `--api-key-env NAME` reads a
key from that environment variable without saving it. Remote URLs require
`--allow-remote` because generation sends prompt text to the server; redirects and
environment proxies are disabled.

Each repetition independently primes the base before **each** measured transition:
base → identical, base → extended, and base → changed suffix. The request order is
`prime(base), measure(identical), prime(base), measure(extended), prime(base),
measure(changed_suffix)`: three unmeasured priming requests and three measured
requests per repetition. Priming uses the same generation options and can leave
generated suffix tokens in native state; it is not a prefill-only operation. Only
the measured requests enter samples and summaries. Reports retain prompt/output
hashes rather than their text. These are **warm-prefix workloads**, not guaranteed
cold-cache runs or verified cache hits. No cache flush is performed. The
deterministic raw request uses temperature 0, top-p 1 and seed 42, avoiding
chat-template differences. Check output hashes and
counts: different output lengths, tokenizers, EOG behavior, model artifacts, or
backend defaults can invalidate a simple speed comparison.

### Metric definitions

- **TTFT:** client request start to the first nonempty `response` text frame—not
  headers, empty lifecycle frames, or thinking metadata. No text means `null`.
- **End-to-end tokens/s:** server-reported output token count divided by client
  request-to-terminal-frame elapsed time. This includes prefill, transport and any
  model loading. It is **not isolated decode throughput**; token counts may include
  an EOG sample that is not visible text.
- **Reported prompt tokens:** `reported_prompt_tokens` preserves the server's
  `prompt_eval_count`; it is not a measured cache-miss count. Current lma reports
  the full **logical** prompt length even when a prefix is reused, not the number
  of tokens actually evaluated in prefill.
- **Server phase rates:** reported count/duration ratios, calculated only when the
  server supplies positive `prompt_eval_duration` / `eval_duration` values (in
  nanoseconds). Missing durations remain `null`; zero durations yield no rate.
  Current lma supplies neither phase duration, so its phase rates remain `null`.
  A supplied duration alone does not establish that a prompt count represents
  evaluated/cache-miss tokens. Verify each backend's count and timing semantics
  before interpreting these ratios as prefill/decode throughput. Do not divide
  lma's logical prompt count by TTFT or total elapsed time and call it prefill speed.
- **Distinct outputs:** number of output hashes within each workload, useful for
  spotting cache/output instability. Numerical backend differences can also change
  greedy output; hashes alone do not identify a bug.

Timeouts bound socket I/O and check elapsed time between frames; they are not a
hard process watchdog against a peer that trickles an unfinished frame. Use an
external process timeout if benchmarking untrusted servers. Generation streams
and metadata have byte limits; incomplete/error streams fail instead of producing
successful-looking benchmark records. Reports include resident metadata and notes,
which can contain local paths; inspect them before sharing.

## SSD and mmap limitations

The opt-in NSS flag `--memory-fallback mmap` (root hosting: `-mk mmap`) is a **CPU/pageable weight-mapping fallback**,
not SSD-native inference. With requested GPU layers, a recoverable initialization
failure permits one retry with model, KQV/op, and projector GPU offload disabled;
with zero GPU layers, it selects CPU/pageable mmap directly. Weight locking is
disabled. The OS controls file-backed page residency and eviction. This fallback
alone does not enable the compact expert cache described below. There is no
application-managed SSD KV cache, asynchronous expert prefetch scheduler, or
guaranteed RAM/VRAM fit. Context/KV, compute, and projector allocations still need
memory.

File-backed weight paging is distinct from anonymous-memory swap. lma does not
configure OS swap, turn SSD capacity into GPU memory, or guarantee that an oversized
model will load or run. Heavy page faults can make generation extremely slow even
on a fast SSD; an OOM-killer termination or native abort cannot be rescued by the
retry. Distinguish resident steady-state measurements from loading and page-fault
costs rather than presenting mmap as a throughput optimization.

## Compact GPU/CPU expert cache: memory/performance tradeoff

NSS flag `--expert-cache-mib N` (root hosting: `-e N`) is a separate opt-in feature, **disabled by default**, with
an explicit range of **1..1048576 MiB**. It uses native vendored llama.cpp changes:
persistent compact expert tensors per layer, synchronous demand uploads/copies
from the original mmap GGUF, and frequency-biased LRU slot replacement.
When GPU is requested, single-token expert `ggml_mul_mat_id` kernels read compact
GPU weights with remapped slot IDs; explicit `--gpu cpu` uses CPU slots/kernels.
Routing probability lookup retains original expert IDs. Complete routing weights
and normalization are expanded before cache preparation, preserving normal Metal
routing fusion; production lookup does not use prep-validated IDs. External
expert selections are rejected. GPU SILU prefill with **2..16 local tokens** now
uses batched routing and capacity-safe sequential GPU expert tiles. Tile capacity
is `min(16, n_slots / top_k)`, or 16 when the full expert set fits. Each admission
depends on the previous completed GPU output before slots can be overwritten;
outputs use balanced concatenation, with coherent projection-bundle evictions
and the same frequency bias. Larger (>16) and empty local views still use mmap
CPU expert matmuls. This is not an mmap-only fallback and does not download or
create converted/on-disk weights.

```sh
lma nss serve --path existing.gguf --expert-cache-mib 1024 --gpu cpu
# GPU-enabled build: explicitly keep local prefill <=16 for GPU expert tiles:
lma nss serve --path existing.gguf --expert-cache-mib 1024 --gpu all --ubatch-size 16
```

The initial allowlist is `qwen2moe`, `qwen3moe`, and `mellum`, using ordinary
contiguous separate up/gate/down or fused gate_up/down expert projections without
expert biases or sidecar scales. Causal text generation only: one live context per
model, one sequence, no embeddings, projectors, or adapters. Requested batch and
micro-batch sizes are preserved, not forced to 1. Placement is local to each
expert operation: single-token views retain the slot/decode path, SILU views of
2..16 tokens use bounded GPU tiles, and >16/empty views use CPU/mmap experts.
A larger requested micro-batch may still have smaller local views. Explicit CPU
cache mode is unchanged. To keep local prefill <=16, **explicitly choose root
`-mh 16` or NSS `--ubatch-size 16`**; this is not an automatic override. Longer
prompts within the context limit span multiple native 16-token micro-batches;
the bound is local, not a total prompt-length limit.
`--gpu all` is not proof of all-GPU execution: routing/top-k preparation remains
synchronous CPU work, even with real GPU expert math. With a 1 GiB cache, GPU
micro-batch-32 short prompts (all local chunks <=16) and micro-batch-16 long
prefill pass. **A 2 GiB normal-fusion run fails short prompt B before the long
case**, while its fusion-disabled diagnostic is bitwise exact throughout. This
larger-cache fusion-related drift remains unresolved. Neither bounded batching
nor 26 passing native CTests establish universal GPU-prefill/offload parity,
32-local-token GPU expert batches, or measured speed parity.

### What to measure

- **Budget versus allocation:** startup reports `budget_bytes` and
  `allocated_bytes`. The latter is the actual persistent compact expert backend
  device slot allocation, including backend alignment/tail padding and Metal
  buffer page rounding, and cannot exceed the
  budget. At least top-k bundles must fit in every layer. A smaller budget fails
  with `minimum aligned bundle bytes` in native logs rather than silently
  increasing the allocation. The CLI's numeric range is not a model-fit promise.
- **Memory outside the budget:** source mmap and OS page cache, dense/shared
  weights, KV/context, graph metadata/workspaces, routing metadata, and allocator
  overhead. **Graph reservation includes worst-case 16-token stream expansion;
  metadata and workspace allocations can increase outside the GPU slot budget.**
  On Apple Silicon, GPU allocations share unified memory with CPU/system use;
  they do not provide independent extra RAM. Measure total RSS, device allocations,
  system memory pressure, and file-backed residency separately. Copying another source
  expert may fault in more pages; those pages can remain OS-resident alongside
  compact copies. There is no `DONTNEED` reclamation or hard RSS ceiling.
- **Cache traffic:** after each generation request, logs report cumulative bundle
  hits, misses, evictions, and `source_bytes_copied`. Use before/after differences
  for a request or interval. These are bundle events, not token counts, projection
  counts, physical disk reads, or prompt-prefix hits. Bounded GPU prefill now
  contributes slot counters and routing-history counts, not just decode.
  > 16-token CPU fallback operations still bypass slot preparation and those
  > counts. A miss can copy from pages
  > already in RAM; a cache hit is not an SSD-throughput measurement.
- **Latency and throughput:** separate load, prefill/TTFT, and decode. More slots
  can reduce copies and eviction at the cost of device memory, but they do not
  remove >16-token CPU fallback, synchronous CPU routing/top-k preparation,
  CPU/GPU transfers, or synchronous upload costs. There is no predictive or
  asynchronous prefetch, measured GPU-prefill speedup, or SSD-speed guarantee.

### Controlled cache-on/cache-off comparison

Use one **already-existing compatible GGUF** and run engines sequentially, not
concurrently on a memory-limited machine. No new downloads or weight copies are
needed. Keep GGUF bytes, prompt/tokenization, output limit, greedy sampling,
context, threads, batch, micro-batch, flash attention, KV type, and verified dense
placement identical. Compare cache off (omit the flag) versus cache on with the
same requested batch/micro-batch sizes; there is no micro-batch-1 requirement.
For a token-at-a-time diagnostic, explicitly set NSS `--ubatch-size 1` (root
`-mh 1`) in **both** runs and report it separately from ordinary batched prefill.
For bounded GPU-prefill measurements, explicitly set `--ubatch-size 16` (root
`-mh 16`) in both runs; the runtime does not apply it automatically. Record
local expert-operation sizes and CPU fallback for >16-token views rather than
attributing every timing difference solely to slot reuse.

Disable prompt-prefix reuse in both runs with `--no-prompt-cache` when isolating
expert reuse. For conversational measurements, keep prefix reuse configured the
same and report it independently. Record the first request after model load and
repeated warm requests separately, including per-request counter deltas, output
hashes/counts, TTFT, end-to-end time, memory, and page-fault/storage observations.
Reloading restores frequency history and may warm up to 64 MiB of hot bundles;
it does **not** reset the OS file cache. Isolate `LMA_EXPERT_CACHE_DIR` for a fresh
history when comparing cold application caches; do not
call such a run cold-SSD without independent evidence. Increasing budgets also
requires a model reload, so repeat the same workload and warmup for each budget.

The HTTP harness above can record client timing/output evidence, but does not
collect expert counters, RSS, page faults, or isolated lma prefill/decode timing;
retain server logs and independent observations alongside it. See the
[expert-cache guide](expert-cache.md) for lifecycle and exact counter semantics.

**Latest reported validation boundary:** sequential Mellum2 Base Q4_K_M runs
with all-GPU request `999` and a 1 GiB cache passed **GPU micro-batch-1 decode**:
exact raw text/usage/finish, final-logit bounds unchanged at
`1e-2 + 1e-3*abs(reference)` and normalized RMSE `<= 1e-3`. Normal fusion is
still enabled, so logits are not bitwise equal (max abs `0.003628254` and
`0.003098965`; normalized RMSE `0.000228704` and `0.000150378`). A diagnostic
with genuine `GGML_METAL_FUSION_DISABLE=1` on both paths made every final logit
bitwise identical; it is not a production workaround. The root cause was cache
admission interposing in routing, corrected by completing routing before prep.

**1 GiB bounded-prefill short-prompt regression passes:** GPU micro-batch 32 versus
the all-GPU reference passes unchanged tolerances after bounded GPU expert
prefill. Run A: max abs `0.0026488304`, normalized RMSE `0.000126679`; B:
`0.003186226`, `0.000188306`. All local prompt chunks were <=16 tokens, so this
is **not** support for entire 32-local-token GPU expert batches. >16 local tokens
still use CPU fallback. The historical CPU-prefill comparison failed (max abs
`1.2669526`, normalized RMSE `0.0689628`); CPU micro-batch-32 full-model checks
passed exact logits.

**1 GiB micro-batch-16 long prefill also passes:** enabling
`LMA_TEST_EXPERT_CACHE_LONG_PROMPT=1` adds a sentence repeated four times,
producing >32 tokens across multiple native 16-token micro-batches. Raw text,
usage and finish reason are exact; unchanged logit bounds pass. The long case
had max abs `0.002730608` and normalized RMSE `0.000216652`.

**2 GiB normal-fusion validation fails:** the additional GPU micro-batch-16
short + long run fails on **short prompt B before reaching the long case**, with
max abs `0.096595764` and normalized RMSE `0.00848751`. The same 2 GiB suite with
`GGML_METAL_FUSION_DISABLE=1` passes **bitwise exactly in every case, including
the long prompt**. This is a diagnostic, not a production workaround. The
larger-cache fusion-related drift is **unresolved**; no test threshold was
relaxed and fusion is not globally disabled. The final 26 passing native CTests
and the backend view-placement fix do not close this separate real-model
conformance gap. Verified 1 GiB cases are not universal
GPU-prefill/offload parity.

GPU slots allocated `1,073,709,056` of `1,073,741,824` budget bytes. Repeating a
one-token request added 224 hits without new copies/misses/evictions. GPU history
reload restored exact counts, warmed `66,834,432` bytes (<64 MiB), and preserved
output. After the backend view-placement fix, **full-feature 1 GiB GPU
micro-batch-16 long-prompt and history reruns pass**; GPU history again warms
`66,834,432` bytes. See [detailed evidence](expert-cache.md#latest-reported-real-model-evidence).
These are correctness/reuse checks, **not speed measurements**. Latest full-feature
release **CLI and desktop builds both pass**; installed binaries are unchanged.
No global GPU parity, GPU-prefill speedup, bounded RSS, SSD speed, or cross-engine
speed parity is established.

## FreeToken is a separate target

[FlashML FreeToken](https://github.com/FlashML-org/FreeToken) documents an
NVIDIA-oriented oversized-MoE runtime with expert caching, CPU/GPU co-execution,
semantic-anchor checkpoints and its own FTW format. lma currently uses
GGUF/llama.cpp, including Metal on Apple Silicon. Whole-model CPU mmap fallback
is **not expert caching**. The separate compact GPU/CPU expert-slot cache is
implemented, with bounded GPU SILU prefill through 16 local tokens and CPU/mmap
fallback for >16-token or empty local views. It is not
semantic-anchor checkpointing, FTW support, or FreeToken feature/performance parity.

The HTTP harness currently targets the shared lma/Ollama raw endpoint; it does not
claim FreeToken protocol support. A FreeToken comparison needs supported hardware,
a compatible model/quantization and an explicitly matched workload. Its published
performance claims are not measurements of this repository.

## Model-free regression checks

```sh
cargo test --offline --no-default-features --lib
python3 -B -m unittest discover -s tests -p 'test_benchmark.py'
```

The native cache's [model-free test sources and build instructions](../vendor/llama-cpp-sys-2/llama.cpp/docs/expert-cache.md#model-free-native-tests)
cover synthetic separate/fused expert graphs, original-ID weighting, reuse/LRU,
aligned budgets, context leases, malformed routing, and Q4_0 byte/stride
preservation. Rust tests cover CLI/config policy and the binding plumbing. These
also cover bounded frequency warmup and empty routing. The updated Metal routing
regression requires actual fusion on both paths (reported counts **32/32**, zero
weight/weighted-output error). The **final full native suite passes all 26 CTests**:
all **22 GPU graph configurations** plus CPU cache, Metal, C API and CPU graph,
with **no skips or timeouts in 271.9 seconds**.
Expanded synthetic all-GPU-reference coverage through 16 local tokens includes
multi-token tile capacity (>1), fused weights, full-cache residency, the 16-token
boundary, and 17-token CPU fallback. Observed maximum absolute difference was
**0**, with the `1e-6` bound unchanged. The older CPU-prefill-reference cases are
historical, not the scope of the new bounded-prefill all-GPU comparisons. Four
added fused-weight cases cover >16-token CPU fallback and empty outputs with
fusion on/off. GPU-op assignment now skips `VIEW`, `RESHAPE`, `PERMUTE` and
`TRANSPOSE`, allowing CPU-backed views to transfer before GPU consumers. These
synthetic passes and the backend view-placement fix do **not** resolve the
separate 2 GiB normal-fusion real-model failure reported above. A byte comparison
against the original cached crate confirmed that upstream cache-off routing
ordering remains unchanged; this is not a broad performance/parity result.

Used empty ggml chunks now get at least one byte of backing, avoiding a
double-allocation assertion caused by NULL data with a non-NULL buffer. CPU
regressions cover shared empty inputs/views and repeated allocation/compute.
The previous intermittent graph-allocation timeout **did not recur** in the final
26-test run, but the empty-chunk bug is **not its proven root cause**. GPU cases
run in fresh processes.

Final model-free validation passes **508 Rust library, 56 CLI and 56 desktop-state
tests**. Latest full-feature release **CLI and desktop builds both pass**;
installed binaries are unchanged. None of these checks establishes a throughput
improvement.

The ignored native prefix/reset regression below is separate from expert-cache
validation. It explicitly loads one installed model and never
contacts Ollama's daemon or downloads weights. Run it alone when memory permits:

```sh
LMA_TEST_OLLAMA_MODEL=YOUR_INSTALLED_TAG cargo test --offline --no-default-features \
  --lib runtime::inference::native_tests::real_native_text_cache_reset_and_greedy_parity \
  -- --ignored --exact --test-threads=1 --nocapture
```

Alternatively set `LMA_TEST_TEXT_GGUF` to an existing text GGUF, not both variables.
It uses a cooperative 120-second deadline; native loading/decode cannot be
preempted by that closure. Tests on other architectures, GPU paths and rolled-out
SWA prefixes, plus controlled phase/throughput measurements, remain necessary.
Batch, KV and thread tuning should follow those measurements—not larger
allocations or changed sampling defaults chosen solely to improve a benchmark
number.
