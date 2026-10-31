# Native expert cache and learned hot experts

This opt-in feature is a persistent, compact **device-slot cache of routed MoE
expert bundles**, backed by the original mmap GGUF. When GPU is requested,
compact slots and single-token expert matmuls reside on GPU; **SILU prefill of
2..16 local tokens** uses capacity-safe sequential GPU expert tiles. Larger
(>16) and empty local views use original mmap CPU expert matmuls. Explicit CPU
mode (`-g cpu`) is unchanged. This is real GPU expert computation, not just mmap
hints, but not FreeToken compatibility, SSD-as-VRAM, or a guaranteed speedup.
With a **1 GiB cache**, real-model GPU decode/history, micro-batch-32 short
prompts (all local chunks <=16), and micro-batch-16 long prefill across multiple
micro-batches pass unchanged tolerances. **At 2 GiB, normal-fusion testing fails
short prompt B before the long case; a fusion-disabled diagnostic is bitwise
exact for all cases, including long prefill.** The larger-cache fusion-related
drift remains unresolved. These results do not establish entire 32-local-token
GPU expert batches, universal GPU-prefill/offload parity, or speed parity.

```sh
# Supported GPU build: explicitly keep local prefill <=16 for GPU expert tiles.
lma s -p existing-moe.gguf -e 1024 -g all -mh 16
# Explicit CPU cache mode; terminal chat accepts the same root flags.
lma s -p existing-moe.gguf -e 1024 -g cpu
lma c -p existing-moe.gguf -e 1024 -g cpu
# Long commands and flags are NSS-only.
lma nss serve --path existing-moe.gguf --expert-cache-mib 1024 --gpu all --ubatch-size 16
```

Root accepts **only canonical short commands and flags**. Use `lma h s` for
hosting flags; root memory fallback is `-mk mmap`. Long flags below are
**NSS-only**, for example `lma nss serve ... --memory-fallback mmap`.

`N` is 1..1048576 MiB; omitting the flag disables the feature. There is no new
model download, conversion, or redundant on-disk expert-weight copy. Keep the
original model available and immutable while loaded. This cache is separate from
`--memory-fallback mmap`, which controls an explicit CPU initialization retry.

## What happens

- Model loading validates supported expert layouts before loading weight data.
  Source expert tensors stay in original CPU file mappings; dense CPU/GPU kernel
  choices remain independent. Whole-file startup prefetch is suppressed in cache
  mode, and the source mapping is not locked into RAM.
- Each layer receives compact slots for coherent up/gate/down or fused gate-up/down
  bundles on the requested GPU, or CPU in explicit CPU mode. The router keeps
  original expert IDs for probability lookup. The complete original-ID routing
  weight chain (including normalization) is expanded **before** cache admission;
  an explicit routing-ready dependency keeps preparation after it. CPU
  preparation/remap nodes supply distinct slot IDs only to expert matmuls.
  Production probability lookup does not consume prep-validated IDs: IDs come
  from internal top-k routing, and external expert selections are rejected.
  Normal Metal fusion remains enabled; there is no global production disable.
- Single-token local expert operations retain the existing slot/decode path.
  With GPU caching, SILU operations with 2..16 local tokens use batched routing
  and sequential GPU expert tiles. A tile has at most
  `min(16, n_slots / top_k)` tokens, or 16 when the entire expert set fits.
  This bounds each tile's active set to available slots. A dependency on the
  previous completed GPU output serializes admission before any slot overwrite;
  tile outputs are joined with balanced concatenation. All projections remain a
  coherent bundle through eviction, with the same frequency-biased policy.
- Larger (>16) and empty local views use original mmap CPU expert matmuls without
  uploading full expert tensors. The decision is per operation: a final-layer
  view can be smaller than the requested micro-batch. Explicit CPU cache mode is
  unchanged. Batch/micro-batch settings are preserved, never forced to 1.
- Demand misses plan coherent projection uploads before publishing bundle
  identities. The Metal path borrows already-registered source buffers and
  coalesces supported misses into a synchronous blit batch. Uncovered source
  ranges retain generic uploads; production does not create duplicate large
  NoCopy mappings/residency registrations. The observed Mellum binding covers
  7,324,434,432 source bytes, with 1 of 28 layers still using generic uploads.
  The complete active top-k set is protected from eviction. Hits reuse slots
  across tokens and requests, independently of prompt/KV prefix reuse.
- Eviction is LRU with a bounded logarithmic frequency bonus. Frequent experts
  get preference, but history cannot permanently pin obsolete experts after the
  workload changes. There is no speculative or asynchronous expert prefetch.

## Learned history under `~/.lma`

When cache mode is enabled, lma records per-layer/original-expert selection counts
observed by the slot path in `~/.lma/expert-cache/<model-key>.json`. Counts now
include bounded GPU prefill as well as decode; >16-token CPU fallback operations
still bypass slot preparation and do not add counts. If coding workloads repeatedly route to
particular experts, those experts naturally gain priority. This is learned routing
frequency, **not semantic labeling of a universal set of “coding experts.”**

History contains only format version, a model identity key, expert IDs, and
saturating counts. No prompts, generated text, credentials, or weight copies are
stored. The local key hashes canonical source path, size, and modification time;
it is an invalidation hint, not a complete weight-file SHA256 certification.
Different aliases over the same source share history. Layout/identity mismatches
are rejected; a changed file identity starts new history. Model stores and parent
configuration directories are trusted local locations.

On subsequent loads, the most frequent nonzero expert bundles are warmed first,
subject to available slots and an independent **64 MiB raw-copy startup cap**.
An expert bundle larger than the remaining cap is skipped. This avoids reading a
whole large cache before any request. Warmup adds to copied bytes, not demand
hits/misses or routing counts. A new history performs no warmup copies.

Writes happen after the first observed request, then at most once every 30 seconds,
and when the model is dropped normally. A crash can lose observations since the
last checkpoint. Files are atomically replaced; per-model nonblocking file locks
protect merges of session deltas across processes. Lock sidecars remain in place.
Unix cache-directory/file permissions are 0700/0600. Symlink/special-file targets,
corrupt snapshots, oversized snapshots (1 MiB), and invalid layouts are rejected.
History errors warn and do not prevent inference; retryable save errors back off.

`LMA_EXPERT_CACHE_DIR` can explicitly select another history directory, including
an isolated test directory. With the model unloaded, remove that model's JSON
file to forget its learned priorities. Do not remove lock sidecars while another
process may be using them.

The performance target is now **2–3× Ollama**, not an achieved result. Repeated
workloads can improve expert priority/reuse, but history is not a guarantee of
monotonically increasing speed: changing workloads, transfers, capacity pressure
and synchronization still matter. Benchmark cold/warm history and workload shifts;
see [measured results and acceptance policy](performance.md). The latest local
1 GiB cache run remains slower than Ollama on the tested resident-model workload.

**Cache-off mode does not load/save history, count routed expert frequencies,
perform startup warming, or change normal batching/placement settings.** No
universal zero-regression benchmark is claimed; the default native inference path
is retained rather than automatically enabling this tradeoff for every model.

## Restrictions and memory

Initial architectures: `qwen2moe`, `qwen3moe`, and `mellum`, with ordinary contiguous
separate up/gate/down or fused gate-up/down expert matrices and supported CPU
quantizations. Expert biases/scales, exotic layouts and other architectures fail
closed. Model discovery alone does not certify compatibility.

Only single-sequence causal text generation is supported: one live context per
model, no embedding contexts, vision/projectors, adapters, training, or MTP.
Token counting remains available without a second inference context.

Cache mode **preserves requested logical batch and physical micro-batch sizes**,
including profile defaults and explicit overrides (root `-b` / `-mh`; NSS
`--batch-size` / `--ubatch-size`). To keep each local prefill within the GPU
path's 16-token bound, **explicitly set root `-mh 16` or NSS `--ubatch-size 16`**;
this is a recommendation, not an automatic override. Longer prompts within the
context limit are processed through multiple native micro-batches of at most
16 tokens; total prompt length is not limited to 16. This batching policy does
not resolve the 2 GiB normal-fusion validation failure below. A requested micro-batch of
32 is not support for 32-local-token GPU expert operations: views >16 still use
CPU/mmap. Native routing/top-k preparation remains synchronous CPU work even
when expert math executes on GPU. Synchronous uploads and CPU/GPU transfers can
add latency. Use this for its memory/reuse tradeoff, not as a blanket fastest-mode
switch or a measured GPU-prefill speedup.

The budget covers actual persistent compact expert **device slot allocations**,
including backend alignment/padding and Metal buffer page rounding. At least the active top-k must fit in every layer; an insufficient
budget fails with the minimum aligned bytes in native logs instead of increasing
the budget silently. Unused budget is normal because slots are indivisible.

**Not included:** original mmap/source pages, OS file cache, dense/shared weights,
KV, graph metadata/workspaces, routing tables, allocator overhead, and other
applications. **Graph reservation includes worst-case 16-token stream expansion,
so graph metadata and workspace allocations can increase outside the GPU slot
budget.** On Apple Silicon, GPU buffers use **shared unified memory**, not a separate pool
of extra RAM. Source pages can stay resident alongside compact copies. There is no hard RSS cap,
OS swap configuration, source-page eviction guarantee, or promise that a model
larger than RAM will load or run quickly. Paging can be extremely slow.

## Counters and validation

Startup logs show cache device, budget and actual compact allocation. Each
generation request logs cumulative expert-bundle hits, misses, evictions and
source bytes copied. Copied bytes include bounded startup warming. Bounded GPU
prefill now contributes slot counters and routing-frequency counts; >16-token
CPU fallback operations still do not. Counters are not a record of all prefill
routing or total expert compute. Nor are they token counts, physical
disk reads, or proof that bytes came from SSD rather than OS cache.

### Latest reported real-model evidence

Authorized runs loaded **Mellum2-12B-A2.5B Base Q4_K_M sequentially**, using
all-GPU request `999`. Passing GPU checks below use a **1 GiB** expert-cache
budget unless stated otherwise; the additional **2 GiB** normal-fusion run fails.
The original decode-logit failure was traced to cache admission interposing in
routing. Expanding complete original-ID routing weights before admission fixed
that decode failure without disabling normal fusion or relaxing test thresholds.

| Check                                                                     | Result / boundary                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU micro-batch 1, normal fusion                                          | **PASS:** exact raw text, usage and finish reason. Final logits pass unchanged bounds `abs <= 1e-2 + 1e-3*abs(reference)` and normalized RMSE `<= 1e-3`, not bitwise equality. Run A: max abs `0.003628254`, normalized RMSE `0.000228704`; B: `0.003098965`, `0.000150378`. |
| GPU micro-batch 1, diagnostic `GGML_METAL_FUSION_DISABLE=1` on both paths | Every final logit bitwise identical. This is a genuine fusion-off diagnostic, **not** a global production setting.                                                                                                                                                           |
| GPU slots / repeated one-token request                                    | `1,073,709,056` allocated bytes within `1,073,741,824` budget bytes; repeat adds **224 hits**, no new copies, misses or evictions.                                                                                                                                           |
| GPU history reload                                                        | **PASS:** exact restored counts, `66,834,432` bytes warmed (below 64 MiB), output unchanged. History/warm API semantics are unchanged.                                                                                                                                       |
| CPU micro-batch 32, full model                                            | **PASS:** exact final logits in all tested cases.                                                                                                                                                                                                                            |
| GPU micro-batch 32, 1 GiB, short prompts vs all-GPU reference             | **PASS after bounded GPU prefill:** unchanged logit bounds. Run A: max abs `0.0026488304`, normalized RMSE `0.000126679`; B: `0.003186226`, `0.000188306`. All local prompt chunks were <=16 tokens; this does not validate 32-local-token GPU expert batches.               |
| GPU micro-batch 16, 1 GiB, long-prompt fixture, normal fusion             | **PASS:** exact raw text, usage and finish reason, with unchanged logit bounds. Long case: max abs `0.002730608`, normalized RMSE `0.000216652`. A sentence repeated four times yields >32 tokens, spanning multiple native 16-token micro-batches.                          |
| GPU micro-batch 16, 2 GiB, short + long suite, normal fusion              | **FAIL on short prompt B before the long case:** max abs `0.096595764`, normalized RMSE `0.00848751`. Larger-cache fusion-related drift remains unresolved; the normal-fusion long case was not reached.                                                                     |
| GPU micro-batch 16, 2 GiB, diagnostic `GGML_METAL_FUSION_DISABLE=1`       | **PASS, bitwise exact:** all cases, including the long prompt. Diagnostic only, not a global production setting or resolution of the normal-fusion failure.                                                                                                                  |

Historically, the GPU micro-batch-32 comparison failed when all multi-token
prefill used CPU expert kernels (max abs `1.2669526`, normalized RMSE `0.0689628`).
The 1 GiB short-prompt regression and micro-batch-16 long-prompt check now pass
with bounded GPU expert prefill. Local expert operations above 16 tokens still
use CPU fallback and remain outside that GPU-prefill evidence.

**Real-model conformance is verified for the reported 1 GiB cases, not arbitrary
cache budgets.** The 2 GiB normal-fusion failure remains an explicit validation
gap despite all 26 native CTests passing. Fusion-off bitwise equality points
to fusion-related drift but does not establish its root cause or fix it. No
thresholds were loosened, and normal fusion remains globally enabled in
production. These results do not establish universal GPU-prefill/offload parity,
throughput, bounded RSS, or cross-engine speed parity.
After the backend view-placement fix, **full-feature 1 GiB GPU micro-batch-16
long-prompt and GPU history reruns pass**; history reload warms `66,834,432` bytes.
These reruns do not resolve the separate 2 GiB normal-fusion failure. Latest
full-feature release **CLI and desktop builds both pass**. Final model-free
validation passes **508 Rust library, 56 CLI and 56 desktop-state tests**.
Installed binaries are unchanged, so these results do not certify installed
executables.

Opt-in real-model checks (loads one model at a time; use an isolated history dir):

```sh
LMA_EXPERT_CACHE_DIR=/path/to/test-history LMA_TEST_OLLAMA_MODEL=INSTALLED_TAG \
  cargo test --offline --no-default-features --lib \
  runtime::inference::native_tests::real_native_expert_ \
  -- --ignored --test-threads=1 --nocapture
```

Use `LMA_TEST_TEXT_GGUF` instead of `LMA_TEST_OLLAMA_MODEL` for a file.
`LMA_TEST_EXPERT_CACHE_MIB` overrides the test's 1024 MiB budget. The reference
comparison and history reload default to CPU;
`LMA_TEST_EXPERT_CACHE_GPU_LAYERS` explicitly opts either into a GPU request with
no silent CPU retry. Match the micro-batch and fusion settings to the scoped
results above; this is not a blanket assertion that GPU real-model parity passes.
The comparison defaults to micro-batch 1 to isolate slots;
`LMA_TEST_EXPERT_CACHE_MICRO_BATCH=16` explicitly exercises bounded prefill.
Add **`LMA_TEST_EXPERT_CACHE_LONG_PROMPT=1`** to the conformance check to append
a >32-token fixture (one sentence repeated four times), spanning multiple native
16-token micro-batches when micro-batch 16 is selected. The reported normal-fusion
long-prompt pass used `LMA_TEST_EXPERT_CACHE_MIB=1024` and
`LMA_TEST_EXPERT_CACHE_GPU_LAYERS=999`; increasing the budget to 2048 exposed the
failure above, before reaching the long case. The long fixture is optional and
does not change runtime defaults.
Values 2..32 enable multi-token checks, but GPU placement depends on each local
operation: >16 local tokens compare CPU fallback against the all-GPU reference,
not just compact storage. A short prompt with requested micro-batch 32 can still
have every local chunk <=16. The history check keeps micro-batch 1.
These settings are **not** runtime batching requirements. Cooperative
240-second test deadlines cannot preempt an individual native call. Native
synthetic tests cover byte preservation, separate/fused projections, budgets,
LRU/frequency warmup, original-ID weighting, failures, and empty routing; see the
[vendor test instructions](../vendor/llama-cpp-sys-2/llama.cpp/docs/expert-cache.md).

### Native synthetic evidence and final suite

The **final full native run passes all 26 CTests**: all **22 GPU graph
configurations** plus CPU cache, Metal, C API and CPU graph tests, with **no skips
or timeouts in 271.9 seconds**. Expanded synthetic coverage uses
an all-GPU expert reference for bounded prefill up to 16 local tokens, including
multi-token tile capacity (>1), fused weights, full-cache residency, the 16-token
boundary, and 17-token CPU fallback. Observed per-layer maximum absolute
difference was **0**; the `1e-6` bound is unchanged. The Metal routing regression
requires actual fusion on both paths; its reported **32/32** fusion counts and
zero routing-weight/weighted-output error remain separate diagnostic evidence.
Four added fused-weight cases cover >16-token CPU fallback and empty outputs,
each with fusion on/off. Review fixed backend placement: GPU-op assignment now
skips `VIEW`, `RESHAPE`, `PERMUTE` and `TRANSPOSE`, allowing CPU-backed views to
transfer before their GPU consumers instead of forcing the views onto GPU.
These synthetic passes and the view-placement fix do **not** resolve the
separate 2 GiB normal-fusion real-model failure above.

A byte comparison against the original cached crate also confirmed that the
upstream **cache-off routing ordering is unchanged**. This source check is not
a universal offload-parity or performance result.

An empty-chunk allocation bug was also fixed: used empty ggml chunks receive at
least one byte of backing. Previously, NULL data with a non-NULL buffer could
trigger a double-allocation assertion. CPU regressions cover shared empty inputs,
views, and repeated allocation/compute. The earlier intermittent graph-allocation
timeout **did not recur in the final 26-test run**, but this allocation bug is
**not a proven root cause of the earlier timeout**. GPU cases run in fresh
processes.

For speed/memory comparisons, see [performance methodology](performance.md).
No measured speedup/parity, GPU-prefill speedup, hard resident-memory limit,
FreeToken/Ollama parity, or reliable SSD throughput is established.
