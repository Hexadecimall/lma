#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <memory>
#include <utility>
#include <vector>

// Persistent CPU or GPU bundles. Sources are borrowed from the owning model and
// need not have data yet at construction (graph reservation never populates slots).
class llama_expert_cache {
public:
    struct layer_config {
        ggml_tensor * up;
        ggml_tensor * gate;
        ggml_tensor * down;
        ggml_tensor * gate_up;
        int64_t n_used;
    };

    llama_expert_cache(uint64_t budget, const std::vector<layer_config> & layers,
                       ggml_backend_buffer_type_t cache_buft = nullptr);
    ~llama_expert_cache();

    // Prove that production sources really are CPU file mappings, not full heap copies.
    void validate_sources(const std::vector<std::pair<const void *, size_t>> & mappings) const;
    // Call after source validation/weight loading, before creating a context.
    // Borrow existing same-device model mappings; NEVER own/free/retain them or
    // register additional mappings, even if some projections are uncovered.
    // Returns true if every layer can blit from borrowed buffers. Otherwise logs
    // the uncovered/unsupported layers and uses generic uploads for those layers.
    // Buffers and CPU sources must outlive every context lease, NOT the cache:
    // model pimpl (which owns the buffers) is destroyed before expert_cache.
    bool bind_upload_buffers(const std::vector<ggml_backend_buffer_t> & buffers);
    // Borrowed sources must outlive this lease. Release only after the context's
    // scheduler is synchronized; release drops NoCopy aliases, not cached slots.
    std::shared_ptr<void> acquire_context();
    bool failed() const noexcept;
    llama_expert_cache_stats stats() const noexcept;
    // Diagnostics separate from the stable public llama stats ABI. Context-scoped
    // metadata aliases of bound buffers add NO GPU residency registration. Without
    // binding, standalone NoCopy maps are allowed only when a conservative total
    // page-rounded registration bound is <= 64 MiB (and no overlapping views are
    // needed). That separate residency cost is OUTSIDE the compact-slot budget.
    struct upload_stats {
        uint64_t aliased_bytes; // live logical source bytes; page/view residency can be larger
        uint64_t submissions;
        uint64_t borrowed_bytes; // configured logical source coverage, survives contexts
        uint64_t mapped_bytes;   // live logical bytes in cache-owned standalone mappings
        uint64_t map_calls;      // cumulative new mapping calls, including failed attempts
    };
    upload_stats uploads() const noexcept;
    size_t usage(llama_expert_usage * entries, size_t capacity) const noexcept;
    bool set_usage(const llama_expert_usage * entries, size_t count);
    bool warm(uint64_t max_copy_bytes);
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> tensor_pairs() const;

    // Bound graph expansion independently of the requested attention micro-batch.
    static constexpr int64_t max_stream_tokens = 16;
    // A worst-case tile's entire active union must fit before any matmul starts.
    int64_t max_batch_tokens(int il) const;

    // Cached matmuls must execute on the cache device; full-layout prefill and
    // empty ops must stay CPU-side, without a scheduler copy of source weights.
    void assign_matmul(ggml_backend_sched_t sched, ggml_backend_t cpu,
                       ggml_tensor * node, bool cached) const;

    // Returns checked ORIGINAL ids for probability gathering and separate slot ids
    // for matmuls, replacing only the latter's weight tensors with compact tensors.
    ggml_tensor * build(ggml_context * ctx, int il, ggml_tensor * ids,
                       ggml_tensor *& slots, ggml_tensor *& up, ggml_tensor *& gate,
                       ggml_tensor *& down, ggml_tensor *& gate_up,
                       ggml_backend_sched_t sched, ggml_backend_t cpu,
                       ggml_tensor * routing_ready = nullptr);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
