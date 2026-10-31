# Native GPU/CPU MoE expert-slot cache

This is an opt-in, demand-filled cache of persistent, compact device expert
bundles. GPU requests select GPU slots; explicit CPU mode retains CPU slots.
It is not an mmap hint or a routing-statistics-only feature. For single-token
local expert operations, `build_moe_ffn` replaces expert weights with compact
tensors and uses slot IDs with `ggml_mul_mat_id` kernels on the cache device.
For GPU caching, SILU prefill of **2..16 local tokens** now uses batched routing
and capacity-safe sequential GPU expert tiles. Larger (>16) and empty local
views use original CPU/mmap expert matmuls; explicit CPU caching is unchanged.
Requested batching is preserved, not forced to 1. With a **1 GiB cache**,
real-model GPU decode/history, micro-batch-32 short prompts (all local chunks
<=16), and micro-batch-16 long prefill pass unchanged tolerances. **A 2 GiB
normal-fusion run fails short prompt B before the long case; its fusion-disabled
diagnostic is bitwise exact throughout, including long prefill.** The
larger-cache fusion-related drift is unresolved. This is not support for entire
32-local-token GPU expert batches, universal GPU-prefill/offload parity, or
measured speed parity.

## API and setup

```c
struct llama_model_params mp = llama_model_default_params();
mp.expert_cache_bytes = 512ULL * 1024 * 1024; // 0 (default) disables the cache
mp.n_gpu_layers = 1; // GPU slots in a supported GPU build; use 0 for CPU slots
struct llama_model * model = llama_model_load_from_file(path, mp);
// Check model != NULL: too small a budget or unsupported layouts fail loading.

struct llama_context_params cp = llama_context_default_params();
cp.n_seq_max = 1;
cp.embeddings = false;
cp.n_ubatch = 16; // Explicit choice for bounded GPU prefill; not forced by cache setup
struct llama_context * ctx = llama_init_from_model(model, cp);

struct llama_expert_cache_stats stats;
if (llama_model_get_expert_cache_stats(model, &stats)) {
    // budget_bytes, allocated_bytes, hits, misses, evictions, bytes_copied, gpu
}
```

The model owns the cache through an opaque `unique_ptr`, independent of graph
reuse and context recreation. Only one context may be alive for a cached model;
failed context construction releases its lease. As with ordinary llama models,
the model must outlive its context. Counters persist for the model lifetime.
Snapshots are safe during decode, but individual atomic counters are not a
transactionally consistent snapshot of an in-progress bundle copy.

Native model setup forces mmap without mlock and forces routed experts to
unrepacked buffers. Dense weights retain their normal extra/repacked kernel
choices. All routed expert projections (including `ffn_gate_up_exps`) use the
plain CPU buffer type for their original mmap sources, overriding tensor
placement rules. Compact slot buffers use the selected GPU's buffer type when
`n_gpu_layers > 0`, otherwise the CPU buffer type. Nonexpert weights retain their
requested CPU/GPU placement. Check the selected buffer/device in logs and
`stats.gpu`; unsupported operations on the required device fail rather than
silently moving compact matmuls elsewhere. Whole-file startup prefetch is disabled. Tensor validation
(`check_tensors`), metadata-only/no-allocation loading, user-provided tensor data,
MTP, and tensor-parallel splitting are rejected in this mode.

Native context setup preserves requested `n_batch`/`n_ubatch`; it no longer
forces `n_ubatch=1`. It disables opportunistic `op_offload` only for CPU cache mode;
GPU cache mode preserves the requested GPU operation policy. Pipeline parallelism
and all-expert warmup remain disabled. Expert matmul placement is explicit:
single-token local operations use the existing cache-device decode path. With
GPU caching, SILU operations with 2..16 local tokens use sequential GPU expert
tiles; >16 and empty local views use CPU/mmap experts without full expert-weight
GPU copies. Explicit CPU caching is unchanged. Selection is per local expert
operation, not per HTTP request, requested `n_ubatch`, or logical prompt batch;
a final-layer view can have fewer tokens. Explicitly choose `n_ubatch=16` (lma
root `-mh 16`, NSS `--ubatch-size 16`) to keep each local prefill <=16; the cache
does not set that automatically. Longer prompts within the context limit span
multiple native 16-token micro-batches; the bound is per local operation, not
total prompt length. This does not resolve the 2 GiB normal-fusion validation
failure below.
Only single-sequence causal token generation is initially supported
(`n_seq_max <= 1`). Embedding contexts, embedding-input/vision batches, encoder
calls, training, LoRA/embedded adapters, and control vectors are unsupported.
Do not change the context into an embedding/noncausal/training context after
creation; decode rejects these configurations.

## Supported layouts

The initial architecture allowlist is `qwen2moe`, `qwen3moe`, and `mellum`.
This vendor revision has a distinct `mellum` architecture; a Mellum2 GGUF using
`qwen2moe` is also covered. Sequential Mellum2-12B-A2.5B Base Q4_K_M checks now
include passing 1 GiB GPU micro-batch-1 decode/history reload, GPU micro-batch-32
short prompts (all local chunks <=16), GPU micro-batch-16 long prefill, and CPU
micro-batch-32 full-model logits. The earlier CPU-prefill cross-backend failure
is historical, but the additional 2 GiB normal-fusion short-prompt failure remains
unresolved. See the
[application evidence](../../../../docs/expert-cache.md#latest-reported-real-model-evidence)
for exact scope, numbers, and unchanged thresholds.

Every layer must have either:

- separate contiguous `[input, intermediate, expert]` up and gate weights,
  plus `[intermediate, output, expert]` down weights; or
- a contiguous `[input, 2*intermediate, expert]` fused gate/up weight (gate first),
  plus the ordinary down weights.

All four ggml strides must be the original ordinary block-contiguous strides;
views, exotic layouts, inconsistent dimensions/counts, expert biases, sidecar
scales, and NVFP4 are rejected before loading weight data. Floating F32/F16/BF16
and the explicitly allowlisted ordinary CPU vec-dot quantized formats are
supported. Quantized bytes are copied verbatim, without dequantization,
repacking, or changes to block/row/expert strides. Per-weight activation precision
policy is carried over to compact tensors.

## Budget, routing, and execution

- Every layer gets at least its top-k number of slots. Insufficient budgets fail
  with the minimum required aligned byte count, before weight data loading.
- Remaining capacity is distributed across layers in deterministic round-robin
  order, up to the model's full expert count per layer. Allocation sizes include
  selected-device backend alignment/tail padding and physical buffer granularity
  (including Metal page rounding). Actual buffer sizes are checked:
  `allocated_bytes <= budget_bytes`.
- All projections share one expert-to-slot mapping per layer. On a miss, the
  prep callback synchronously uploads/copies **all** projections via
  `ggml_backend_tensor_set` before publishing the bundle. It supports private GPU
  buffers without assuming that compact `data` is host-dereferenceable.
- Existing members of the complete active set are pinned before any victim is
  selected. Frequency-biased LRU victims exclude the entire active set, including
  hits occurring later in routing order. All projections share coherent bundle
  admission/eviction and frequency accounting. Slots remain valid through the
  down projection; other layers have disjoint bundles.
- GPU SILU prefill with 2..16 local tokens uses batched routing and sequential
  tiles of at most `min(16, n_slots / top_k)` tokens. When the entire expert set
  fits, the tile limit is 16. This keeps the worst-case active expert set within
  available slots. Each tile's admission depends on the previous completed GPU
  output, serializing preparation before any slot overwrite. Tile outputs are
  joined using balanced concatenation. The single-token decode path is retained.
- Production probability lookup uses **original internal top-k IDs**, not
  prep-validated IDs; external expert selections are rejected in cache mode.
  `build_moe_ffn` expands the complete original-ID routing-weight chain, including
  normalization/scaling, **before** calling the cache builder. Interposing CPU
  admission before `GET_ROWS` had changed routing arithmetic/fusion and caused
  the original GPU decode-logit failure.
- The cache builder accepts optional `routing_ready`, making routing completion
  a dependency of CPU admission. Bounded prefill also depends on the prior tile's
  completed GPU output before reusing slots. Prep validates IDs for slot safety
  and remaps them to slots. Only expert matmuls consume slot IDs; the production
  probability path does not wait on prep's checked-ID output. Normal Metal
  fusion remains enabled; there is no global production disable.
- Compact matmuls are explicitly assigned to the cache device. Routing/top-k
  preparation and remapping remain synchronous CPU work, completing before
  dependent GPU expert matmuls consume slots. >16-token and empty local views use
  CPU with original mmap weights; the scheduler must not upload full expert
  tensors. No predictive/asynchronous upload pipeline is implemented.
- Graph construction/reservation never copies experts or updates hit counters.
  Sources can be unset when the compact metadata/buffers are initialized. Before
  returning a loaded model, sources are checked against the actual file mapping
  address ranges and must be host-backed.
- Worker callbacks do not allocate or throw. Invalid/duplicate IDs or unavailable
  sources set a persistent cache failure flag and produce safe IDs. The native
  abort callback observes the flag (while preserving the application's abort
  callback), and graph synchronization makes the current decode return an error.
  A model whose cache has failed must be freed/reloaded, not reused for inference.

Hits/misses/evictions count **expert bundles**, not projections. `bytes_copied`
counts raw bytes in copied/uploaded projections, not slot padding, including
explicit frequency-based startup warming. Bounded GPU prefill now updates slot
counters and usage frequencies, not just decode. >16-token CPU fallback
operations still bypass slot preparation and do not contribute those counts.
These statistics do not measure all prefill routing, disk reads, or transfer
latency. There is no predictive or asynchronous expert prefetch.

`llama_model_get_expert_usage` snapshots saturating per-expert routing counts in
deterministic layer/expert order. `llama_model_set_expert_usage` validates the full
table before importing it. `llama_model_warm_expert_cache(model, max_copy_bytes)`
warms nonzero frequencies in descending order, filling only free slots under the
raw-copy cap. Import/warm operations require no live context. Warmup does not
increment demand misses, hits, or usage counts. The application persists these
counts (not prompts or generated text) under `~/.lma/expert-cache/`, optionally
overridden by `LMA_EXPERT_CACHE_DIR`, and caps startup copies/uploads at **64 MiB**.
Cache-off application inference does not read, write, or warm from this history.

Eviction adds a logarithmic frequency bonus (at most 63) to recency, so frequent
experts get preference without permanently pinning obsolete workload history.
Empty final-layer routing views skip cache preparation entirely. Public decode
catches native graph exceptions rather than unwinding through the Rust C ABI.

## What is not bounded

The byte budget bounds persistent compact expert **device slot allocations**,
including alignment/padding, not whole-process RSS. On Apple Silicon, GPU buffers
share unified memory with CPU/system use; they are not an independent extra-RAM
pool. Graph metadata/workspaces, routing metadata, KV cache, dense/shared
weights, allocator overhead, mapped source pages and the OS filesystem cache are
outside it. **Graph reservation now includes worst-case 16-token stream expansion;
graph metadata and workspace allocations can increase outside the GPU slot
budget.** Copying a source expert faults in file-backed pages. The OS may keep
those pages resident after the copy. This implementation deliberately does not
issue `madvise(DONTNEED)` on source or heap addresses, and does not claim a hard
RSS limit or evict adjacent dense pages.

## Model-free native tests

From the llama.cpp root, independently of Rust/Cargo:

```sh
cmake -S tests/expert-cache -B .expert-cache-build \
  -DGIT_EXE=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build .expert-cache-build --target test-expert-cache -j2
ctest --test-dir .expert-cache-build --output-on-failure
```

Alternatively enable `LLAMA_BUILD_EXPERT_CACHE_TESTS=ON` in a normal native build.
The tests compare two-layer separate/fused F32 inference with uncached execution,
check original-ID weighting, top-k routing views, graph reuse, late-hit pinning,
hits/evictions/copy counts, aligned budgets, context leases, malformed-ID aborts,
and Q4_0 byte/block-stride preservation plus exact matmul equality. The integrated
build also has a C API/defaults/link smoke test. The standalone build above is
CPU-only; source coverage or CPU passes alone do not certify GPU execution.

Full native builds with a `ggml-metal` target also add `test-expert-cache-metal`
(CTest name `expert-cache-metal`, skip code 77 if Metal is unavailable). It is a
model-free **diagnostic**, covering Q4_K original-versus-compact indexed matvecs,
slot byte uploads/readback, reuse/evictions, aligned allocation accounting, and
routing/normalization with fusion diagnostics. Its production routing regression
requires actual fusion on both paths; the latest reported counts were **32/32**,
with zero routing-weight and weighted-output error. Legacy interposed topology
remains diagnostic only. The standalone CMake project forces Metal off, so the
CPU commands above do not exercise it. Static full
builds also provide `test-expert-cache-graph` for CPU production-graph coverage and
`test-expert-cache-gpu-graph` for the actual common MoE builder on Metal. The latter
checks four Q4_K layers, GPU-slot decode and bounded GPU prefill with operation
offload enabled, last-only/all/empty outputs, repeated execution, and explicit
expert placement. Expanded coverage compares bounded prefill through 16 local
tokens against an **all-GPU expert reference**, including tile capacity >1,
fused weights, full-cache residency, the 16-token boundary, and 17-token CPU/mmap
fallback. The >16 fallback is not claimed as GPU expert compute. Per-layer
maximum absolute error must stay within the unchanged `1e-6` bound; observed
maximum difference was **0**.

The **final full native suite passes all 26 CTests**: all **22 GPU graph
configurations** plus CPU cache, Metal, C API and CPU graph, with **no skips or
timeouts in 271.9 seconds**. GPU cases run in fresh processes. Four added
fused-weight configurations cover >16-token CPU fallback and empty outputs,
each with fusion on/off. Review fixed backend placement: GPU-op assignment now
skips `VIEW`, `RESHAPE`, `PERMUTE` and `TRANSPOSE`, so CPU-backed views can
transfer before their GPU consumers rather than being forced onto GPU.
These synthetic passes and the view-placement fix do **not** resolve the
separate 2 GiB normal-fusion real-model validation failure below.

A byte comparison against the original cached crate confirmed that upstream
**cache-off routing ordering remains unchanged**. This source check does not
establish universal offload parity or performance.

Used empty ggml chunks now receive **at least one byte of backing**: NULL data
with a non-NULL buffer could previously trigger a double-allocation assertion.
CPU regressions exercise shared empty inputs/views and repeated allocation and
compute. The previous intermittent decode/fusion-on graph-allocation timeout
**did not recur in the final 26-test run**, but the empty-chunk bug is **not a
proven root cause of that timeout**.

With a **1 GiB cache**, real-model GPU micro-batch-1 tests pass unchanged logit bounds
`abs <= 1e-2 + 1e-3*abs(reference)` and normalized RMSE `<= 1e-3`, with exact raw
text/usage/finish but not bitwise logits under normal fusion. A genuine
`GGML_METAL_FUSION_DISABLE=1` diagnostic on both paths was bitwise exact; production
does **not** globally disable fusion. After bounded GPU prefill, the GPU
micro-batch-32 short-prompt regression versus the all-GPU reference now passes
those unchanged bounds: run A max abs `0.0026488304`, normalized RMSE `0.000126679`;
B `0.003186226`, `0.000188306`. All local prompt chunks were <=16 tokens, not
entire 32-local-token GPU expert batches. The old failure (max abs `1.2669526`,
normalized RMSE `0.0689628`) occurred with CPU expert prefill. >16-local-token
fallback remains CPU.

The **1 GiB GPU micro-batch-16 long-prompt check passes**, with exact raw
text/usage/finish and unchanged logit bounds: long max abs `0.002730608`, normalized
RMSE `0.000216652`. Optional `LMA_TEST_EXPERT_CACHE_LONG_PROMPT=1` appends a

> 32-token fixture (one sentence repeated four times) to the application conformance
> test; native micro-batch 16 splits it across multiple micro-batches. Use
> `LMA_TEST_EXPERT_CACHE_MICRO_BATCH=16`, `LMA_TEST_EXPERT_CACHE_MIB=1024` and
> `LMA_TEST_EXPERT_CACHE_GPU_LAYERS=999` for that reported configuration; see the
> [application test instructions](../../../../docs/expert-cache.md#latest-reported-real-model-evidence).

**At 2 GiB, the normal-fusion GPU micro-batch-16 short + long run fails on short
prompt B before reaching the long case:** max abs `0.096595764`, normalized RMSE
`0.00848751`. The same 2 GiB suite with diagnostic
`GGML_METAL_FUSION_DISABLE=1` passes **bitwise exactly for all cases, including
the long prompt**. This establishes fusion-related drift, not a proven root
cause or fix. Larger-cache real-model conformance remains **unresolved** despite
26 passing native CTests and the separate backend view-placement fix. No thresholds were loosened, and production fusion
is not globally disabled. Verified 1 GiB cases do not establish universal
GPU-prefill/offload parity, throughput, or a GPU-prefill speedup.
After the backend view-placement fix, **full-feature 1 GiB GPU micro-batch-16
long-prompt and GPU history reruns pass**; history warms `66,834,432` bytes.
Latest full-feature release **CLI and desktop builds both pass**. Final model-free
validation passes **508 Rust library, 56 CLI and 56 desktop-state tests**.
Installed binaries are unchanged.

For the integrated tests, from the project root:

```sh
cmake -S vendor/llama-cpp-sys-2/llama.cpp -B target/native-expert-gpu-tests \
  -DGIT_EXE=OFF -DGIT_EXECUTABLE=/usr/bin/false -DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE \
  -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DLLAMA_BUILD_EXPERT_CACHE_TESTS=ON -DLLAMA_BUILD_COMMON=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF \
  -DGGML_METAL=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF
cmake --build target/native-expert-gpu-tests --target test-expert-cache \
  test-expert-cache-api test-expert-cache-graph test-expert-cache-metal \
  test-expert-cache-gpu-graph -j2
ctest --test-dir target/native-expert-gpu-tests --output-on-failure
```
