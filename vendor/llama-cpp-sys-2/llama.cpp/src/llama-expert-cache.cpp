#include "llama-expert-cache.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-metal.h" // optional proc-address ABI only; no Metal linkage/types


#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef __APPLE__
#include <unistd.h>
#endif

namespace {
// Deliberately mirror only weight types with a CPU vec_dot in this vendor
// revision. Do not link CPU-private traits: backends can be loaded dynamically.
bool supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0: case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_TQ1_0: case GGML_TYPE_TQ2_0: case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

void upload_log(ggml_log_level level, const char * message) {
    ggml_log_callback callback = nullptr;
    void * data = nullptr;
    ggml_log_get(&callback, &data);
    if (callback) { callback(level, message, data); }
    else { std::fputs(message, stderr); }
}

uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) {
        throw std::runtime_error("expert cache size overflow");
    }
    return a + b;
}
}

struct llama_expert_cache::impl {
    struct projection {
        ggml_tensor * source;
        ggml_tensor * compact;
        ggml_tensor * alias;
        ggml_backend_buffer_ptr alias_buffer; // owned ONLY for bounded standalone maps
        ggml_backend_buffer_t borrowed_buffer = nullptr; // model owns; never freed/dereferenced at cache teardown
    };
    struct layer {
        impl * owner;
        layer_config config;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        std::vector<projection> projections;
        std::vector<int32_t> expert_slot;
        std::vector<int32_t> slot_expert;
        std::vector<uint64_t> age;
        std::vector<uint8_t> pinned;
        std::unique_ptr<std::atomic<uint64_t>[]> frequency;
        std::vector<int32_t> pending_slot;
        std::vector<std::pair<int32_t, int32_t>> pending;
        std::vector<ggml_backend_metal_copy> ranges;
        bool borrowed_complete = false;

        bool use_blit() const {
            // Unleased standalone users, private/discrete/other backends and
            // non-blittable quantized strides retain the generic tensor_set path.
            if (!owner->copy_batch || !owner->context_in_use.load()) { return false; }
            for (const auto & p : projections) {
                if ((uintptr_t(p.source->data) % 4) || (p.source->nb[2] % 4)) { return false; }
            }
            if (owner->upload_buffers_bound) { return borrowed_complete; }
            if (!owner->standalone_maps_allowed) {
                if (!owner->unbound_warning_logged) {
                    owner->unbound_warning_logged = true;
                    upload_log(GGML_LOG_LEVEL_WARN,
                            "expert cache: unbound NoCopy sources exceed the standalone 64 MiB registration bound "
                            "or need overlapping views; using generic uploads. Bind existing model buffers to enable blits without new residency.\n");
                }
                return false;
            }
            return true;
        }

        bool map_sources() {
            for (auto & p : projections) {
                if (p.alias->buffer) { continue; }
                ggml_backend_buffer_t source_buffer = p.borrowed_buffer;
                if (!owner->upload_buffers_bound) {
                    // Only bounded standalone users create mappings. Production
                    // borrows model buffers and NEVER registers projection views.
                    owner->map_calls.fetch_add(1);
                    p.alias_buffer.reset(ggml_backend_dev_buffer_from_host_ptr(owner->device,
                            p.source->data, ggml_nbytes(p.source), p.source->nb[2]));
                    if (!p.alias_buffer) { return false; }
                    source_buffer = p.alias_buffer.get();
                    owner->mapped_bytes.fetch_add(ggml_nbytes(p.source));
                }
                if (!source_buffer) { return false; }
                if (ggml_backend_tensor_alloc(source_buffer, p.alias, p.source->data) != GGML_STATUS_SUCCESS) {
                    p.alias->buffer = nullptr;
                    p.alias->data = nullptr;
                    return false;
                }
                owner->aliased_bytes.fetch_add(ggml_nbytes(p.source));
            }
            return true;
        }

        uint64_t score(int32_t slot) const noexcept {
            uint64_t count = frequency[slot_expert[slot]].load(std::memory_order_relaxed);
            uint64_t bonus = 0;
            while (count > 1) { count >>= 1; ++bonus; }
            // Frequency is a bounded recency bonus, not permanent pinning: old
            // coding habits cannot prevent admission after the workload changes.
            return age[slot] > UINT64_MAX - bonus ? UINT64_MAX : age[slot] + bonus;
        }

        uint64_t clock = 0;
        int64_t n_expert;
        int64_t n_slots;

        void resize(int64_t count) {
            n_slots = count;
            for (auto & p : projections) {
                p.compact->ne[2] = count;
                p.compact->nb[3] = p.compact->nb[2] * count;
            }
        }

        uint64_t size() const {
            // Include the backend's per-tensor allocation padding and the
            // physical buffer granularity (Metal rounds every buffer to pages).
            uint64_t result = 0;
            const auto buft = owner->buft;
            const size_t alignment = ggml_backend_buft_get_alignment(buft);
            for (const auto & p : projections) {
                const uint64_t bytes = ggml_backend_buft_get_alloc_size(buft, p.compact);
                result = checked_add(result, checked_add(bytes, alignment - 1) / alignment * alignment);
            }
            return checked_add(result, owner->buffer_alignment - 1) / owner->buffer_alignment * owner->buffer_alignment;
        }

        // Only the single-task prep callback writes bundles. Pin ALL existing
        // active experts before selecting any victims (including hits later in top-k).
        bool prepare(const int32_t * ids, int64_t n_tokens) {
            const int64_t n_ids = config.n_used * n_tokens;
            if (owner->failure.load()) {
                return false;
            }
            for (int64_t i = 0; i < n_ids; ++i) {
                if (ids[i] < 0 || ids[i] >= n_expert) {
                    return false;
                }
                // The same expert across tokens is valid; duplicates within
                // one token's top-k indicate malformed routing.
                for (int64_t j = i / config.n_used * config.n_used; j < i; ++j) {
                    if (ids[j] == ids[i]) {
                        return false;
                    }
                }
            }
            for (const auto & p : projections) {
                if (!p.source->data || !p.source->buffer || !ggml_backend_buffer_is_host(p.source->buffer)) {
                    return false;
                }
            }
            std::fill(pinned.begin(), pinned.end(), 0);
            for (int64_t i = 0; i < n_ids; ++i) {
                const int32_t slot = expert_slot[ids[i]];
                if (slot >= 0) {
                    pinned[slot] = 1;
                }
            }
            pending.clear();
            ranges.clear();
            std::fill(pending_slot.begin(), pending_slot.end(), -1);
            for (int64_t i = 0; i < n_ids; ++i) {
                const int32_t id = ids[i];
                int32_t slot = expert_slot[id] >= 0 ? expert_slot[id] : pending_slot[id];
                if (slot < 0) {
                    for (int32_t s = 0; s < n_slots; ++s) {
                        if (!pinned[s] && (slot < 0 || slot_expert[s] < 0 || score(s) < score(slot))) {
                            slot = s;
                            if (slot_expert[s] < 0) {
                                break;
                            }
                        }
                    }
                    if (slot < 0) {
                        return false;
                    }
                    pending_slot[id] = slot;
                    pending.emplace_back(id, slot);
                }
                pinned[slot] = 1;
            }
            uint64_t copied = 0;
            if (!pending.empty()) {
                if (use_blit()) {
                    if (!map_sources()) { return false; }
                    // Projection-major ordering also coalesces adjacent expert /
                    // slot ranges in the backend, without widening copied bytes.
                    for (const auto & p : projections) {
                        for (const auto & entry : pending) {
                            ranges.push_back({p.alias, p.compact, size_t(entry.first) * p.source->nb[2],
                                    size_t(entry.second) * p.compact->nb[2], p.source->nb[2]});
                            copied += p.source->nb[2];
                        }
                    }
                    uint64_t submitted = 0;
                    const bool ok = owner->copy_batch(ranges.data(), ranges.size(), &submitted);
                    owner->submissions.fetch_add(submitted);
                    if (!ok) { return false; } // never publish or retry a failed GPU batch
                    owner->bytes_copied.fetch_add(copied);
                } else {
                    for (const auto & entry : pending) {
                        for (const auto & p : projections) {
                            ggml_backend_tensor_set(p.compact,
                                    static_cast<const char *>(p.source->data) + entry.first * p.source->nb[2],
                                    entry.second * p.compact->nb[2], p.source->nb[2]);
                            owner->bytes_copied.fetch_add(p.source->nb[2]);
                        }
                    }
                }
            }
            // Publish the ENTIRE active union only after every range completed.
            for (const auto & entry : pending) {
                const int32_t id = entry.first, slot = entry.second;
                if (slot_expert[slot] >= 0) {
                    expert_slot[slot_expert[slot]] = -1;
                    owner->evictions.fetch_add(1);
                }
                slot_expert[slot] = id;
                expert_slot[id] = slot;
            }
            owner->misses.fetch_add(pending.size());
            owner->hits.fetch_add(n_ids - pending.size());
            for (int64_t i = 0; i < n_ids; ++i) {
                const int32_t id = ids[i];
                age[expert_slot[id]] = ++clock;
                const uint64_t count = frequency[id].load(std::memory_order_relaxed);
                if (count != UINT64_MAX) { frequency[id].store(count + 1, std::memory_order_relaxed); }
            }
            return true;
        }

        static void prep(ggml_tensor * dst, const ggml_tensor * src, int ith, int, void * data) noexcept {
            // ggml's CPU executor can invoke custom ops on every worker even
            // with n_tasks=1; the callback itself must select the writer.
            if (ith != 0) { return; }
            auto & self = *static_cast<layer *>(data);
            // Upload implementations may throw internally. Never unwind into
            // ggml's C/worker executor; native decode observes the failure flag.
            try {
                if (self.prepare(static_cast<const int32_t *>(src->data), src->ne[1])) {
                    std::memcpy(dst->data, src->data, ggml_nbytes(dst));
                    return;
                }
            } catch (...) {
            }
            self.owner->failure.store(true);
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }

        static void prep_after_routing(ggml_tensor * dst, const ggml_tensor * src,
                                       const ggml_tensor *, int ith, int nth, void * data) noexcept {
            prep(dst, src, ith, nth, data);
        }

        static void remap(ggml_tensor * dst, const ggml_tensor * src, int ith, int, void * data) noexcept {
            if (ith != 0) { return; }
            auto & self = *static_cast<layer *>(data);
            auto * out = static_cast<int32_t *>(dst->data);
            const auto * ids = static_cast<const int32_t *>(src->data);
            for (int64_t i = 0; i < ggml_nelements(src); ++i) {
                const int32_t id = ids[i];
                const int32_t slot = id >= 0 && id < self.n_expert ? self.expert_slot[id] : -1;
                if (slot < 0 || self.owner->failure.load()) {
                    self.owner->failure.store(true);
                    out[i] = 0;
                } else {
                    out[i] = slot;
                }
            }
        }
    };

    ggml_backend_buffer_type_t buft;
    ggml_backend_dev_t device;
    bool gpu;
    size_t buffer_alignment;
    uint64_t budget;
    uint64_t allocated = 0;
    std::atomic<uint64_t> hits{0}, misses{0}, evictions{0}, bytes_copied{0};
    ggml_backend_metal_copy_batch_t copy_batch = nullptr;
    std::atomic<uint64_t> aliased_bytes{0}, submissions{0};
    std::atomic<uint64_t> borrowed_bytes{0}, mapped_bytes{0}, map_calls{0};
    bool upload_buffers_bound = false;
    bool standalone_maps_allowed = true;
    bool unbound_warning_logged = false;
    uint64_t standalone_registration_bound = 0;

    void release_aliases() noexcept {
        // The synchronized context lease must release these BEFORE model mmap
        // teardown. Compact payloads and history intentionally survive contexts.
        for (auto & l : layers) {
            for (auto & p : l->projections) {
                p.alias_buffer.reset();
                p.alias->buffer = nullptr;
                p.alias->data = nullptr;
            }
        }
        aliased_bytes.store(0);
        mapped_bytes.store(0);
    }
    std::atomic<bool> failure{false}, context_in_use{false};
    std::vector<std::unique_ptr<layer>> layers;

    impl(uint64_t budget, const std::vector<layer_config> & configs, ggml_backend_buffer_type_t cache_buft) :
        buft(cache_buft ? cache_buft : ggml_backend_cpu_buffer_type()),
        device(ggml_backend_buft_get_device(buft)),
        gpu(device && (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                       ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU)),
        buffer_alignment(ggml_backend_buft_get_alignment(buft)), budget(budget) {
#ifdef __APPLE__
        if (device && (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "MTL") == 0 ||
                               std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "Metal") == 0)) {
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0) { throw std::runtime_error("cannot determine Metal allocation page size"); }
            buffer_alignment = std::max(buffer_alignment, size_t(page_size));
        }
#endif
        if (gpu) {
            const auto reg = ggml_backend_dev_backend_reg(device);
            const auto supports = reinterpret_cast<bool (*)(ggml_backend_buffer_type_t)>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_copy_batch_supports_buft"));
            if (supports && supports(buft)) {
                copy_batch = reinterpret_cast<ggml_backend_metal_copy_batch_t>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_copy_batch"));
            }
        }
        if (!budget || configs.empty()) {
            throw std::runtime_error("expert cache requires a nonzero budget and expert layers");
        }
        uint64_t total = 0;
        for (const auto & c : configs) {
            if (!c.down || (c.gate_up ? (c.up || c.gate) : (!c.up || !c.gate))) {
                throw std::runtime_error("expert cache requires up/gate/down or fused gate_up/down");
            }
            auto l = std::make_unique<layer>();
            l->owner = this;
            l->config = c;
            l->n_expert = c.down->ne[2];
            if (l->n_expert <= 0 || l->n_expert > INT32_MAX || c.n_used <= 0 || c.n_used > l->n_expert) {
                throw std::runtime_error("expert cache has invalid expert/top-k count");
            }
            l->ctx.reset(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
            if (!l->ctx) {
                throw std::runtime_error("expert cache metadata allocation failed");
            }
            for (auto * t : {c.up, c.gate, c.down, c.gate_up}) {
                if (!t) {
                    continue;
                }
                if (t->view_src || !ggml_is_contiguous(t) || t->ne[3] != 1 || t->ne[2] != l->n_expert ||
                        !supported_type(t->type) || t->nb[0] != ggml_type_size(t->type) ||
                        t->nb[1] != ggml_row_size(t->type, t->ne[0]) ||
                        t->nb[2] != t->nb[1] * t->ne[1] || t->nb[3] != t->nb[2] * t->ne[2]) {
                    throw std::runtime_error(std::string("unsupported expert cache tensor: ") + t->name);
                }
                auto * compact = ggml_new_tensor_3d(l->ctx.get(), t->type, t->ne[0], t->ne[1], c.n_used);
                ggml_set_name(compact, t->name);
                auto * alias = ggml_dup_tensor(l->ctx.get(), t);
                l->projections.push_back({t, compact, alias, {}});
                // Include an extra page for an arbitrary source offset, and
                // reject overlapping Metal views in the unbound standalone path.
                const uint64_t registration = checked_add(
                        checked_add(ggml_nbytes(t), buffer_alignment - 1) / buffer_alignment * buffer_alignment,
                        buffer_alignment);
                standalone_registration_bound = checked_add(standalone_registration_bound, registration);
                if (registration > ggml_backend_buft_get_max_size(buft) ||
                        standalone_registration_bound > 64 * 1024 * 1024) {
                    standalone_maps_allowed = false;
                }
            }
            const auto * input = c.gate_up ? c.gate_up : c.up;
            const int64_t n_ff = c.down->ne[0];
            if (input->ne[0] != c.down->ne[1] || input->ne[1] != n_ff * (c.gate_up ? 2 : 1) ||
                    (c.gate && (c.gate->ne[0] != input->ne[0] || c.gate->ne[1] != n_ff))) {
                throw std::runtime_error("expert cache projection dimensions disagree");
            }
            l->resize(c.n_used);
            if (l->size() > ggml_backend_buft_get_max_size(buft)) {
                throw std::runtime_error("expert cache layer exceeds device buffer size limit");
            }
            total = checked_add(total, l->size());
            layers.push_back(std::move(l));
        }
        if (total > budget) {
            throw std::runtime_error("expert cache budget too small; minimum aligned bundle bytes: " + std::to_string(total));
        }
        // Deterministic round-robin distribution of the remaining budget. Slots
        // are layer-private so an active bundle remains pinned through down-proj.
        bool added;
        do {
            added = false;
            for (auto & l : layers) {
                if (l->n_slots == l->n_expert) {
                    continue;
                }
                const auto before = l->size();
                l->resize(l->n_slots + 1);
                const auto delta = l->size() - before;
                if (delta <= budget - total && l->size() <= ggml_backend_buft_get_max_size(buft)) {
                    total += delta;
                    added = true;
                } else {
                    l->resize(l->n_slots - 1);
                }
            }
        } while (added);
        for (auto & l : layers) {
            l->expert_slot.assign(l->n_expert, -1);
            l->slot_expert.assign(l->n_slots, -1);
            l->age.assign(l->n_slots, 0);
            l->pinned.assign(l->n_slots, 0);
            l->pending_slot.assign(l->n_expert, -1);
            l->pending.reserve(l->n_slots);
            l->ranges.reserve(size_t(l->n_slots) * l->projections.size());
            l->frequency.reset(new std::atomic<uint64_t>[l->n_expert]);
            for (int64_t i = 0; i < l->n_expert; ++i) { l->frequency[i].store(0); }
            // One explicitly sized buffer per layer: account page rounding and
            // every projection, without hidden context-allocation sub-buffers.
            l->buffer.reset(ggml_backend_buft_alloc_buffer(buft, l->size()));
            if (!l->buffer) {
                throw std::runtime_error("expert cache device allocation failed");
            }
            auto allocator = ggml_tallocr_new(l->buffer.get());
            for (const auto & p : l->projections) {
                if (ggml_tallocr_alloc(&allocator, p.compact) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("expert cache tensor allocation failed");
                }
            }
            allocated = checked_add(allocated, ggml_backend_buffer_get_size(l->buffer.get()));
            if (allocated > budget) {
                throw std::runtime_error("expert cache allocator exceeded budget");
            }
            ggml_backend_buffer_set_usage(l->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(l->buffer.get(), 0);
        }
    }
};

llama_expert_cache::llama_expert_cache(uint64_t budget, const std::vector<layer_config> & layers,
                                     ggml_backend_buffer_type_t cache_buft) :
    pimpl(std::make_unique<impl>(budget, layers, cache_buft)) {}
llama_expert_cache::~llama_expert_cache() = default;

bool llama_expert_cache::failed() const noexcept { return pimpl->failure.load(); }

int64_t llama_expert_cache::max_batch_tokens(int il) const {
    if (il < 0 || size_t(il) >= pimpl->layers.size()) {
        throw std::runtime_error("expert cache tile layer out of range");
    }
    const auto & l = *pimpl->layers[il];
    return l.n_slots == l.n_expert ? max_stream_tokens
                                  : std::min(max_stream_tokens, l.n_slots / l.config.n_used);
}

llama_expert_cache_stats llama_expert_cache::stats() const noexcept {
    return {pimpl->budget, pimpl->allocated, pimpl->hits.load(), pimpl->misses.load(),
            pimpl->evictions.load(), pimpl->bytes_copied.load(), pimpl->gpu};
}

llama_expert_cache::upload_stats llama_expert_cache::uploads() const noexcept {
    return {pimpl->aliased_bytes.load(), pimpl->submissions.load(), pimpl->borrowed_bytes.load(),
            pimpl->mapped_bytes.load(), pimpl->map_calls.load()};
}

size_t llama_expert_cache::usage(llama_expert_usage * entries, size_t capacity) const noexcept {
    size_t required = 0;
    for (const auto & l : pimpl->layers) { required += size_t(l->n_expert); }
    if (entries && capacity >= required) {
        size_t index = 0;
        for (size_t il = 0; il < pimpl->layers.size(); ++il) {
            const auto & l = pimpl->layers[il];
            for (int64_t id = 0; id < l->n_expert; ++id) {
                entries[index++] = {uint32_t(il), uint32_t(id), l->frequency[id].load(std::memory_order_relaxed)};
            }
        }
    }
    return required;
}

bool llama_expert_cache::set_usage(const llama_expert_usage * entries, size_t count) {
    const auto lease = acquire_context();
    if (failed() || !entries || count != usage(nullptr, 0)) { return false; }
    size_t index = 0;
    for (size_t il = 0; il < pimpl->layers.size(); ++il) {
        for (int64_t id = 0; id < pimpl->layers[il]->n_expert; ++id) {
            const auto & entry = entries[index++];
            if (entry.layer != il || entry.expert != id) { return false; }
        }
    }
    index = 0;
    for (auto & l : pimpl->layers) {
        for (int64_t id = 0; id < l->n_expert; ++id) {
            l->frequency[id].store(entries[index++].count, std::memory_order_relaxed);
        }
    }
    return true;
}

bool llama_expert_cache::warm(uint64_t max_copy_bytes) {
    const auto lease = acquire_context();
    if (failed()) { return false; }
    struct candidate { size_t layer; int32_t id; uint64_t count; };
    std::vector<candidate> ranked;
    for (size_t il = 0; il < pimpl->layers.size(); ++il) {
        const auto & l = pimpl->layers[il];
        for (int32_t id = 0; id < l->n_expert; ++id) {
            const auto count = l->frequency[id].load(std::memory_order_relaxed);
            if (count && l->expert_slot[id] < 0) { ranked.push_back({il, id, count}); }
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const candidate & a, const candidate & b) {
        return a.count > b.count;
    });
    for (const auto & entry : ranked) {
        auto & l = *pimpl->layers[entry.layer];
        const auto empty = std::find(l.slot_expert.begin(), l.slot_expert.end(), -1);
        if (empty == l.slot_expert.end()) { continue; }
        uint64_t bytes = 0;
        for (const auto & p : l.projections) {
            if (!p.source->data || !p.source->buffer || !ggml_backend_buffer_is_host(p.source->buffer)) { return false; }
            bytes = checked_add(bytes, p.source->nb[2]);
        }
        if (bytes > max_copy_bytes) { continue; }
        const auto slot = int32_t(empty - l.slot_expert.begin());
        for (const auto & p : l.projections) {
            ggml_backend_tensor_set(p.compact,
                    static_cast<const char *>(p.source->data) + entry.id * p.source->nb[2],
                    slot * p.compact->nb[2], p.source->nb[2]);
        }
        l.slot_expert[slot] = entry.id;
        l.expert_slot[entry.id] = slot;
        l.age[slot] = ++l.clock;
        pimpl->bytes_copied.fetch_add(bytes);
        max_copy_bytes -= bytes;
    }
    return true;
}

std::vector<std::pair<ggml_tensor *, ggml_tensor *>> llama_expert_cache::tensor_pairs() const {
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> result;
    for (const auto & l : pimpl->layers) {
        for (const auto & p : l->projections) {
            result.emplace_back(p.source, p.compact);
        }
    }
    return result;
}

std::shared_ptr<void> llama_expert_cache::acquire_context() {
    if (pimpl->context_in_use.exchange(true)) {
        throw std::runtime_error("expert cache supports only one live context per model");
    }
    return std::shared_ptr<void>(pimpl.get(), [](void * p) {
        auto & self = *static_cast<impl *>(p);
        self.release_aliases();
        self.context_in_use.store(false);
    });
}

void llama_expert_cache::validate_sources(const std::vector<std::pair<const void *, size_t>> & mappings) const {
    for (const auto & l : pimpl->layers) {
        for (const auto & p : l->projections) {
            const auto * t = p.source;
            bool mapped = false;
            for (const auto & m : mappings) {
                const uintptr_t begin = reinterpret_cast<uintptr_t>(m.first);
                const uintptr_t addr = reinterpret_cast<uintptr_t>(t->data);
                if (addr >= begin && addr - begin <= m.second && ggml_nbytes(t) <= m.second - (addr - begin)) {
                    mapped = true;
                    break;
                }
            }
            if (!mapped || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
                throw std::runtime_error(std::string("expert cache source must be original CPU mmap: ") + t->name);
            }
        }
    }
}

bool llama_expert_cache::bind_upload_buffers(const std::vector<ggml_backend_buffer_t> & buffers) {
    // Same exclusion as warm/history updates: binding must never race decode or
    // a live context. The lease also detaches any temporary alias metadata.
    const auto lease = acquire_context();
    pimpl->upload_buffers_bound = true; // even an empty/partial binding forbids new maps
    uint64_t borrowed = 0;
    size_t generic_layers = 0;
    for (auto & l : pimpl->layers) {
        l->borrowed_complete = pimpl->copy_batch != nullptr;
        for (auto & p : l->projections) {
            p.borrowed_buffer = nullptr;
            if (pimpl->copy_batch && p.source->data && p.source->buffer &&
                    ggml_backend_buffer_is_host(p.source->buffer)) {
                const uintptr_t addr = reinterpret_cast<uintptr_t>(p.source->data);
                for (auto * buffer : buffers) {
                    if (!buffer || ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) != pimpl->device) {
                        continue;
                    }
                    // Same-device Metal buffers expose either a real shared base
                    // or a private virtual address. Only a buffer whose address
                    // range contains the ORIGINAL CPU source can serve as alias.
                    // Multi-buffers expose no base and are not alias candidates.
                    const auto * base_ptr = ggml_backend_buffer_get_base(buffer);
                    if (!base_ptr) { continue; }
                    const uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr);
                    const size_t size = ggml_backend_buffer_get_size(buffer);
                    if (addr >= base && addr - base <= size &&
                            ggml_nbytes(p.source) <= size - (addr - base)) {
                        p.borrowed_buffer = buffer;
                        borrowed = checked_add(borrowed, ggml_nbytes(p.source));
                        break;
                    }
                }
            }
            if (!p.borrowed_buffer || (uintptr_t(p.source->data) % 4) || (p.source->nb[2] % 4)) {
                l->borrowed_complete = false;
            }
        }
        if (!l->borrowed_complete) { ++generic_layers; }
    }
    pimpl->borrowed_bytes.store(borrowed);
    char message[256];
    std::snprintf(message, sizeof(message),
            "expert cache: borrowed upload source coverage = %llu bytes; %zu/%zu layers use generic uploads; "
            "new source mappings/residency registrations disabled\n",
            (unsigned long long) borrowed, generic_layers, pimpl->layers.size());
    upload_log(generic_layers ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO, message);
    return generic_layers == 0;
}

void llama_expert_cache::assign_matmul(ggml_backend_sched_t sched, ggml_backend_t cpu,
                                      ggml_tensor * node, bool cached) const {
    ggml_backend_t target = cpu;
    if (cached && pimpl->gpu) {
        target = nullptr;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto * backend = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(backend) == pimpl->device) {
                target = backend;
                break;
            }
        }
        if (!target || !ggml_backend_supports_buft(target, pimpl->buft)) {
            throw std::runtime_error("expert cache GPU is unavailable in this scheduler");
        }
    }
    if (!target || !ggml_backend_supports_op(target, node)) {
        throw std::runtime_error("expert cache matmul is unsupported on its required device");
    }
    ggml_backend_sched_set_tensor_backend(sched, node, target);
}

ggml_tensor * llama_expert_cache::build(ggml_context * ctx, int il, ggml_tensor * ids,
                                      ggml_tensor *& slots, ggml_tensor *& up, ggml_tensor *& gate,
                                      ggml_tensor *& down, ggml_tensor *& gate_up,
                                      ggml_backend_sched_t sched, ggml_backend_t cpu,
                                      ggml_tensor * routing_ready) {
    if (il < 0 || size_t(il) >= pimpl->layers.size()) {
        throw std::runtime_error("expert cache graph layer out of range");
    }
    auto & l = *pimpl->layers[il];
    const auto & c = l.config;
    // Empty final-layer routing views can have nonstandard strides/shapes.
    // No expert data is read or cached for them.
    if (ggml_nelements(ids) == 0) {
        slots = ids;
        return ids;
    }
    if (up != c.up || gate != c.gate || down != c.down || gate_up != c.gate_up ||
            ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids) ||
            ids->ne[0] != c.n_used || ids->ne[1] < 1 || ids->ne[1] > max_batch_tokens(il) ||
            ids->ne[2] != 1 || ids->ne[3] != 1) {
        throw std::runtime_error("expert cache graph/layout mismatch or tile exceeds slot capacity");
    }

    for (const auto & p : l.projections) {
        if (up == p.source) { up = p.compact; }
        if (gate == p.source) { gate = p.compact; }
        if (down == p.source) { down = p.compact; }
        if (gate_up == p.source) { gate_up = p.compact; }
    }
    // Keep normalization in the preceding GPU split and previous tile reads
    // complete before overwriting any compact expert weights.
    auto * original = routing_ready
        ? ggml_map_custom2(ctx, ids, routing_ready, impl::layer::prep_after_routing, 1, &l)
        : ggml_map_custom1(ctx, ids, impl::layer::prep, 1, &l);
    slots = ggml_map_custom1(ctx, original, impl::layer::remap, 1, &l);
    ggml_format_name(original, "expert_cache_prep-%d", il);
    ggml_format_name(slots, "expert_cache_slots-%d", il);
    if (sched) {
        ggml_backend_sched_set_tensor_backend(sched, original, cpu);
        ggml_backend_sched_set_tensor_backend(sched, slots, cpu);
    }
    return original;
}
