# Native inference vendor patches

`llama-cpp-2` and `llama-cpp-sys-2` are local source copies of the exact cached
0.1.160 crates previously pinned by this project. Upstream license files and
notices are retained in each crate and the bundled llama.cpp/ggml tree. No
FreeToken source or proprietary model weights are included.

Both root `Cargo.toml` and standalone `desktop/Cargo.toml` patch crates.io to these
sources. Keep both patches and lockfiles in sync: native model parameters and C
APIs have ABI extensions, so mixing unpatched native code/bindings is unsupported.

Local changes implement an opt-in bounded GPU/CPU expert-slot cache,
routing-frequency import/export, warm-start hints, safe decode exception
reporting, and Rust API wrappers. GPU requests select compact GPU slots and
single-token expert matmuls; SILU prefill of 2..16 local tokens now uses batched
routing and capacity-safe sequential GPU tiles. Larger (>16) and empty local
views use original mmap CPU expert matmuls, without full expert GPU copies.
Explicit CPU caching is unchanged. Requested batch/micro-batch sizes are
preserved, not forced to 1; explicitly choose root `-mh 16` or NSS
`--ubatch-size 16` to keep local prefill <=16. Longer prompts within the context
limit span multiple native 16-token micro-batches; this is not a total
prompt-length limit. Core native changes are in
`llama.cpp/src/llama-graph.cpp`, `llama.cpp/src/llama-expert-cache.{h,cpp}`, the
model loader/owner, graph/context integration, `include/llama.h`, and CMake/test
wiring. The safe wrapper adds model-param controls and statistics/history APIs.
Ordinary cache-off behavior and default dense CPU repacking remain enabled;
cache-off inference leaves expert history untouched.

The budget bounds compact device slot allocations, including alignment/padding
and Metal page rounding, not total RSS. Graph reservation covers worst-case
16-token stream expansion; graph metadata/workspaces can increase outside the
GPU slot budget. Apple unified memory is shared system memory, not extra VRAM.
Routing/top-k preparation stays synchronous on CPU; expert math is on GPU for
the bounded path. Demand uploads are synchronous, with no predictive or
asynchronous expert prefetch. Metal demand uploads batch supported misses and
borrow existing mapped model buffers; production does not duplicate large NoCopy
residency registrations. Uncovered source ranges retain generic uploads. The
observed Mellum binding covers 7,324,434,432 source bytes, with 1 of 28 layers
using generic uploads. Standalone unbound synthetic mappings have a separate
64 MiB registration cap outside the slot budget. Small shared-buffer Metal
readbacks synchronize producers then copy coherent contents directly, avoiding
an extra destination buffer/blit. Slot/history counts include bounded GPU prefill,
but not >16-token CPU fallback operations. The application saves counts, not text, under
`~/.lma/expert-cache/` (optional `LMA_EXPERT_CACHE_DIR`) and warms at most 64 MiB.

User documentation: [expert caching](../docs/expert-cache.md). Native design and
test instructions: [backend notes](llama-cpp-sys-2/llama.cpp/docs/expert-cache.md).

When updating upstream:

1. Review and rebase each local patch rather than replacing only one crate.
2. Regenerate bindings and both Cargo lockfiles together.
3. Recheck routing-ID layouts, graph ordering, empty microbatches, buffer ownership,
   CPU/GPU kernel/quantization support, synchronous upload ordering, local
   prefill/decode placement, and device allocation accounting.
4. Run native synthetic tests, all model-free Rust/CLI/frontend tests, full-feature
   CLI/desktop builds, and controlled sequential real-model parity/history tests.
5. Preserve upstream license/provenance notices and update this version record.

Production routing completes original-ID probability lookup and normalization
before cache admission; an optional `routing_ready` / `MAP_CUSTOM2` dependency
preserves this ordering. Lookup no longer uses prep-validated IDs, and external
expert selections are rejected. Normal Metal fusion remains enabled.

Latest reported sequential Mellum2 Base Q4_K_M checks (GPU request 999, 1 GiB
cache) pass GPU micro-batch-1 decode conformance and GPU history reload. Raw
text/usage/finish are exact; normal-fusion logits satisfy unchanged
`1e-2 + 1e-3*abs(reference)` and normalized RMSE `<= 1e-3` bounds, not bitwise
equality. CPU micro-batch-32 full-model logits pass exactly. The formerly failing
GPU micro-batch-32 **short-prompt regression now passes**: run A max abs
`0.0026488304`, normalized RMSE `0.000126679`; B `0.003186226`, `0.000188306`.
All local prompt chunks were <=16; this does not validate entire 32-local-token
GPU expert batches. >16-token views retain CPU fallback. **1 GiB GPU
micro-batch-16 long prefill also passes**, with exact text/usage/finish and
unchanged logit bounds: long max abs `0.002730608`, normalized RMSE `0.000216652`.
Optional `LMA_TEST_EXPERT_CACHE_LONG_PROMPT=1` adds a sentence repeated four times
(>32 tokens), spanning multiple native 16-token micro-batches. After the backend
view-placement fix, **full-feature 1 GiB GPU micro-batch-16 long-prompt and history
reruns pass**; GPU history warms `66,834,432` bytes.

**2 GiB normal-fusion GPU micro-batch-16 testing fails short prompt B before the
long case:** max abs `0.096595764`, normalized RMSE `0.00848751`. The same 2 GiB
suite with diagnostic `GGML_METAL_FUSION_DISABLE=1` is **bitwise exact in all
cases, including the long prompt**. Larger-cache fusion-related drift remains
**unresolved**: thresholds are unchanged and production fusion is not globally
disabled. Reported 1 GiB conformance is not universal GPU-prefill/offload parity
or a speed result; see the [detailed evidence](../docs/expert-cache.md#latest-reported-real-model-evidence).

The **final full native suite passes all 26 CTests**: all **22 GPU graph
configurations** plus CPU cache, Metal, C API and CPU graph, with **no skips or
timeouts in 526.29 seconds on the latest rerun**. Expanded synthetic
all-GPU-reference coverage through 16 local tokens includes multi-token tiles,
fused weights, full-cache residency, the 16 boundary and 17-token CPU fallback;
observed maximum difference is zero at the unchanged `1e-6` bound. Four added
fused-weight cases cover >16-token CPU fallback and empty outputs with fusion
on/off. GPU-op assignment now skips `VIEW`, `RESHAPE`, `PERMUTE` and `TRANSPOSE`,
allowing CPU-backed views to transfer before GPU consumers. These synthetic
passes and the backend view-placement fix do **not** resolve the separate 2 GiB
normal-fusion real-model failure. A byte comparison against the original cached
crate confirmed unchanged upstream cache-off routing ordering. Used empty
ggml chunks now receive at least one byte of backing, avoiding a double-allocation
assertion from NULL data with a non-NULL buffer. CPU regressions cover shared
empty inputs/views and repeated allocation/compute. The earlier intermittent
graph-allocation timeout did not recur in this final run; it is **not proven** to
have been caused by that empty-chunk bug.

Full-feature release **CLI and desktop builds** and model-free regression suites
are required after changes; see the current validation report rather than treating
historical test counts as a version guarantee. Installed binaries are not replaced
by a source build.

This is bounded GPU expert prefill, not asynchronous/predictive expert prefetch,
FreeToken's FTW format, or semantic-anchor checkpointing. It makes no universal
speed claim. The current 2–3× Ollama engineering target is not achieved; the latest
scoped comparison is documented in [performance](../docs/performance.md).
