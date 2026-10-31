// Model-free Metal numerical-parity diagnostic. Do not relax direct matmul
// equality here: an expert-axis resize must not change this matvec dispatch.
#include "llama-expert-cache.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

namespace {
constexpr int n_expert = 64, n_slots = 9, top_k = 8, n_input = 1024, n_ff = 256;
using config = llama_expert_cache::layer_config;

struct error_stats {
    double squares = 0;
    double max_abs = 0;
    size_t samples = 0, unequal = 0;
    void add(float a, float b) {
        CHECK(std::isfinite(a) && std::isfinite(b));
        const double d = double(a) - b;
        squares += d * d;
        max_abs = std::max(max_abs, std::fabs(d));
        unequal += a != b;
        ++samples;
    }
    void print(const char * label) const {
        std::printf("%s: max_abs=%.12g RMSE=%.12g unequal=%zu/%zu\n", label,
                    max_abs, samples ? std::sqrt(squares / samples) : 0, unequal, samples);
    }
};

// The backend already exposes these ad-hoc diagnostic hooks. No kernel or
// production graph changes are needed to measure fusion-on versus fusion-off.
struct fusion_api {
    void * handle;
    void (*reset)(void *);
    int (*get)(void *, const char **, uint64_t *, int);
    void (*enable)(void *, bool);
    explicit fusion_api(ggml_backend_dev_t device) {
        auto reg = ggml_backend_dev_backend_reg(device);
        auto resolve = [&](const char * name) {
            auto fn = ggml_backend_reg_get_proc_address(reg, name);
            CHECK(fn);
            return fn;
        };
        auto obtain = reinterpret_cast<void * (*)(ggml_backend_dev_t)>(resolve("ggml_backend_fusion_get"));
        auto init = reinterpret_cast<void (*)(void *)>(resolve("ggml_backend_fusion_stats_init"));
        reset = reinterpret_cast<void (*)(void *)>(resolve("ggml_backend_fusion_stats_reset"));
        get = reinterpret_cast<int (*)(void *, const char **, uint64_t *, int)>(resolve("ggml_backend_fusion_stats_get"));
        enable = reinterpret_cast<void (*)(void *, bool)>(resolve("ggml_backend_fusion_set_enabled"));
        handle = obtain(device);
        CHECK(handle);
        init(handle); // before backend init: serialize encoding for reliable counters
    }
    uint64_t topk_count() const {
        const char * labels[128];
        uint64_t counts[128];
        CHECK(get(handle, nullptr, nullptr, 0) <= 128);
        const int n = get(handle, labels, counts, 128);
        uint64_t result = 0;
        for (int i = 0; i < n; ++i) {
            if (std::strstr(labels[i], "SOFT_MAX") && std::strstr(labels[i], "ARGSORT") &&
                    std::strstr(labels[i], "GET_ROWS")) { result += counts[i]; }
        }
        return result;
    }
};

struct readonly_mapping {
    void * data;
    size_t size;
    explicit readonly_mapping(size_t bytes, bool sparse = false) : size(bytes) {
        std::unique_ptr<FILE, int (*)(FILE *)> file(std::tmpfile(), std::fclose);
        CHECK(file);
        if (sparse) {
            CHECK(ftruncate(fileno(file.get()), size) == 0);
        } else {
            std::vector<unsigned char> contents(size);
            for (size_t i = 0; i < size; ++i) { contents[i] = (i * 37 + (i >> 8) * 11) % 251; }
            CHECK(std::fwrite(contents.data(), 1, size, file.get()) == size);
            CHECK(std::fflush(file.get()) == 0);
        }
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fileno(file.get()), 0);
        CHECK(data != MAP_FAILED);
    }
    ~readonly_mapping() { munmap(data, size); }
};

enum class upload_binding { standalone, full, split, partial, empty, limited_standalone };

void mapped_uploads(ggml_backend_t gpu, ggml_backend_t cpu, bool fused,
                    upload_binding binding = upload_binding::standalone) {
    const bool limited = binding == upload_binding::limited_standalone;
    const bool standalone = binding == upload_binding::standalone || limited;
    const int experts = limited ? 8192 : 8; // sparse 96 MiB source; no GPU map is allowed
    constexpr int slots_count = 4;
    const auto buft = ggml_backend_get_default_buffer_type(gpu);
    ggml_context_ptr sources(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}));
    config c{};
    c.n_used = 2;
    c.down = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 32, 32, experts);
    if (fused) {
        c.gate_up = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 32, 64, experts);
    } else {
        c.up = ggml_dup_tensor(sources.get(), c.down);
        c.gate = ggml_dup_tensor(sources.get(), c.down);
    }
    std::vector<ggml_tensor *> projections;
    uint64_t total = 0, bundle = 0, budget = 0;
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    for (auto * t : {c.up, c.gate, c.down, c.gate_up}) {
        if (!t) { continue; }
        projections.push_back(t);
        total += ggml_nbytes(t);
        bundle += t->nb[2];
        auto * compact = ggml_new_tensor_3d(sources.get(), t->type, t->ne[0], t->ne[1], slots_count);
        budget += (ggml_backend_buft_get_alloc_size(buft, compact) + alignment - 1) / alignment * alignment;
    }
    const size_t page = sysconf(_SC_PAGESIZE);
    CHECK(page > 64);
    budget = (budget + page - 1) / page * page;
    readonly_mapping mapping((total + 64 + page - 1) / page * page, limited);
    ggml_backend_buffer_ptr host(ggml_backend_cpu_buffer_from_ptr(mapping.data, mapping.size));
    size_t offset = 64; // deliberately not page aligned, including each expert slice
    for (auto * t : projections) {
        CHECK(ggml_backend_tensor_alloc(host.get(), t, static_cast<char *>(mapping.data) + offset) == GGML_STATUS_SUCCESS);
        offset += ggml_nbytes(t);
    }
    // Model-owned mappings are created BEFORE the cache binds them. The cache
    // must not register duplicate views or ever free these borrowed buffers.
    std::vector<ggml_backend_buffer_ptr> model_buffers;
    if (binding == upload_binding::full) {
        model_buffers.emplace_back(ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(gpu),
                static_cast<char *>(mapping.data) + 32, mapping.size - 32, bundle));
    } else if (binding == upload_binding::split || binding == upload_binding::partial) {
        for (size_t i = 0; i < projections.size() - size_t(binding == upload_binding::partial); ++i) {
            auto * t = projections[i];
            model_buffers.emplace_back(ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(gpu),
                    t->data, ggml_nbytes(t), t->nb[2]));
        }
    }
    for (const auto & buffer : model_buffers) { CHECK(buffer); }
    auto cache_owner = std::make_unique<llama_expert_cache>(budget, std::vector<config>{c}, buft);
    auto & cache = *cache_owner;
    cache.validate_sources({{mapping.data, mapping.size}});
    const bool blit = binding != upload_binding::partial && binding != upload_binding::empty && !limited;
    if (!standalone) {
        // Include invalid candidates as the real ctxs_bufs list can contain CPU,
        // unrelated GPU, and null buffers. Only containing same-device buffers win.
        std::vector<ggml_backend_buffer_t> buffers = {nullptr, host.get(), cache.tensor_pairs()[0].second->buffer};
        for (const auto & buffer : model_buffers) { buffers.push_back(buffer.get()); }
        CHECK(cache.bind_upload_buffers(buffers) == blit);
        const uint64_t covered = binding == upload_binding::empty ? 0 :
                total - (binding == upload_binding::partial ? ggml_nbytes(projections.back()) : 0);
        CHECK(cache.uploads().borrowed_bytes == covered);
        CHECK(cache.uploads().map_calls == 0 && cache.uploads().mapped_bytes == 0);
    }
    CHECK(cache.max_batch_tokens(0) == 2);
    CHECK(cache.stats().allocated_bytes == budget);
    CHECK(cache.uploads().aliased_bytes == 0 && cache.uploads().submissions == 0);
    std::vector<llama_expert_usage> history(experts);
    for (int e = 0; e < experts; ++e) { history[e] = {0, uint32_t(e), uint64_t(e == 7 ? 100 : 0)}; }
    CHECK(cache.set_usage(history.data(), experts));
    CHECK(cache.warm(bundle));
    CHECK(cache.stats().bytes_copied == bundle && cache.stats().misses == 0);
    CHECK(cache.uploads().aliased_bytes == 0 && cache.uploads().submissions == 0);

    const int32_t selected[][4] = {{7, 1, 7, 2}, {7, 1, 7, 2}, {3, 7, 4, 2}, {3, 7, 4, 2}, {5, 7, 4, 2}};
    const uint64_t misses[] = {2, 0, 2, 0, 1};
    for (int context = 0; context < 3; ++context) {
        auto lease = cache.acquire_context();
        CHECK(cache.uploads().aliased_bytes == 0);
        if (context == 1) {
            // A live (even zero-decode) context must prevent rebinding/races.
            const auto covered = cache.uploads().borrowed_bytes;
            bool rejected = false;
            try { cache.bind_upload_buffers({}); }
            catch (const std::runtime_error &) { rejected = true; }
            CHECK(rejected && cache.uploads().borrowed_bytes == covered);
            continue; // a zero-decode context must not map/upload
        }
        ggml_context_ptr inputs(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        auto * ids = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, 2, 2);
        ggml_set_input(ids);
        ggml_backend_buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), cpu));
        ggml_context_ptr ctx(ggml_init({ggml_graph_overhead_custom(32, false) + 32 * ggml_tensor_overhead(), nullptr, true}));
        ggml_backend_t backends[] = {gpu, cpu};
        ggml_backend_buffer_type_t types[] = {buft, ggml_backend_cpu_buffer_type()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, types, 2, 32, false, false));
        auto * up = c.up; auto * gate = c.gate; auto * down = c.down; auto * gate_up = c.gate_up;
        ggml_tensor * slot_ids;
        cache.build(ctx.get(), 0, ids, slot_ids, up, gate, down, gate_up, sched.get(), cpu);
        ggml_set_output(slot_ids);
        auto * gf = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(gf, slot_ids);
        CHECK(ggml_backend_sched_alloc_graph(sched.get(), gf));
        CHECK(cache.uploads().aliased_bytes == 0); // graph reservation is metadata only
        for (int step = context == 0 ? 0 : 3; step < (context == 0 ? 3 : 5); ++step) {
            const auto before = cache.stats();
            const auto uploads = cache.uploads();
            ggml_backend_tensor_set(ids, selected[step], 0, sizeof(selected[step]));
            CHECK(ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS);
            CHECK(!cache.failed());
            const auto after = cache.stats();
            CHECK(after.misses - before.misses == misses[step]);
            CHECK(after.hits - before.hits == 4 - misses[step]);
            CHECK(after.bytes_copied - before.bytes_copied == misses[step] * bundle);
            CHECK(cache.uploads().submissions - uploads.submissions == uint64_t(blit && misses[step] != 0));
            if (misses[step] && blit) { CHECK(cache.uploads().aliased_bytes == total); }
            if (!blit) { CHECK(cache.uploads().aliased_bytes == 0); }
            if (!standalone || limited) {
                CHECK(cache.uploads().mapped_bytes == 0 && cache.uploads().map_calls == 0);
            } else if (misses[step]) {
                CHECK(cache.uploads().mapped_bytes == total);
            }
            if (step == 3) { CHECK(cache.uploads().aliased_bytes == 0); } // new context, resident hits only
            int32_t actual_slots[4];
            ggml_backend_tensor_get(slot_ids, actual_slots, 0, sizeof(actual_slots));
            for (const auto & pair : cache.tensor_pairs()) {
                CHECK(pair.first->buffer == host.get());
                CHECK(pair.second->ne[2] == slots_count);
                std::vector<unsigned char> bytes(pair.first->nb[2]);
                for (int i = 0; i < 4; ++i) {
                    CHECK(actual_slots[i] >= 0 && actual_slots[i] < slots_count);
                    ggml_backend_tensor_get(pair.second, bytes.data(), actual_slots[i] * bytes.size(), bytes.size());
                    CHECK(std::memcmp(bytes.data(), static_cast<char *>(pair.first->data) + selected[step][i] * bytes.size(), bytes.size()) == 0);
                }
            }
        }
    }
    CHECK(cache.uploads().aliased_bytes == 0 && cache.uploads().submissions == (blit ? 3 : 0));
    CHECK(cache.uploads().mapped_bytes == 0);
    CHECK(cache.uploads().map_calls == (binding == upload_binding::standalone ? 2 * projections.size() : 0));
    CHECK(cache.stats().misses == 5 && cache.stats().hits == 15 && cache.stats().evictions == 2);
    CHECK(cache.stats().bytes_copied == 6 * bundle && cache.stats().allocated_bytes == budget);
    CHECK(cache.usage(history.data(), experts) == size_t(experts) && history[7].count == 107);
    for (size_t i = 0; i < (limited ? 64 : mapping.size); ++i) {
        CHECK(static_cast<unsigned char *>(mapping.data)[i] == (limited ? 0 : (i * 37 + (i >> 8) * 11) % 251));
    }
    std::printf("Metal readonly mmap fused=%d binding=%d: 5 misses/15 hits, %llu actual blit submissions, "
                "raw_bytes=%llu compact_bytes=%llu borrowed_bytes=%llu new_map_calls=%llu; aliases detached across contexts\n",
                int(fused), int(binding), (unsigned long long) cache.uploads().submissions,
                (unsigned long long) cache.stats().bytes_copied, (unsigned long long) budget,
                (unsigned long long) cache.uploads().borrowed_bytes, (unsigned long long) cache.uploads().map_calls);
    if (!model_buffers.empty()) {
        // The first borrowed buffer must still be usable after all context leases
        // released; the cache neither reset nor freed the model's Metal mapping.
        auto probe = *projections.front();
        probe.buffer = model_buffers.front().get();
        std::array<unsigned char, 64> bytes;
        ggml_backend_tensor_get(&probe, bytes.data(), 0, bytes.size());
        CHECK(std::memcmp(bytes.data(), probe.data, bytes.size()) == 0);
    }
    // Reproduce production destruction order: model buffers can die BEFORE the
    // cache. Its stored raw borrowed references must not be touched at teardown.
    model_buffers.clear();
    cache_owner.reset();
}

struct fixture {
    ggml_backend_t gpu, cpu;
    ggml_backend_buffer_type_t buft;
    ggml_context_ptr host, device;
    ggml_backend_buffer_ptr host_buffer, device_buffer;
    config c{};
    ggml_tensor * ids;
    ggml_tensor * logits;
    ggml_tensor * x;
    ggml_tensor * down_x;
    std::array<ggml_tensor *, 3> full;
    std::unique_ptr<llama_expert_cache> cache;
    std::shared_ptr<void> lease;

    fixture(ggml_backend_t gpu, ggml_backend_t cpu) : gpu(gpu), cpu(cpu), buft(ggml_backend_get_default_buffer_type(gpu)) {
        host.reset(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}));
        device.reset(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}));
        c.n_used = top_k;
        c.up = ggml_new_tensor_3d(host.get(), GGML_TYPE_Q4_K, n_input, n_ff, n_expert);
        c.gate = ggml_dup_tensor(host.get(), c.up);
        c.down = ggml_new_tensor_3d(host.get(), GGML_TYPE_Q4_K, n_ff, n_input, n_expert);
        ids = ggml_new_tensor_2d(host.get(), GGML_TYPE_I32, top_k, 1);
        ggml_set_input(ids);
        int p = 0;
        for (auto * source : {c.up, c.gate, c.down}) { full[p++] = ggml_dup_tensor(device.get(), source); }
        x = ggml_new_tensor_3d(device.get(), GGML_TYPE_F32, n_input, 1, 1);
        down_x = ggml_new_tensor_3d(device.get(), GGML_TYPE_F32, n_ff, top_k, 1);
        logits = ggml_new_tensor_2d(device.get(), GGML_TYPE_F32, n_expert, 1);
        for (auto * t : {x, down_x, logits}) { ggml_set_input(t); }
        host_buffer.reset(ggml_backend_alloc_ctx_tensors(host.get(), cpu));
        device_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(device.get(), buft));
        CHECK(host_buffer && device_buffer);
        std::vector<float> values(n_input * n_ff);
        p = 0;
        for (auto * source : {c.up, c.gate, c.down}) {
            for (int e = 0; e < n_expert; ++e) {
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = 0.2f * std::sin(float((i * 37 + e * 101 + p * 13) % 65521) * 0.003f);
                }
                auto * out = static_cast<char *>(source->data) + e * source->nb[2];
                CHECK(ggml_quantize_chunk(source->type, values.data(), out, 0, source->ne[1], source->ne[0], nullptr) == source->nb[2]);
            }
            ggml_backend_tensor_set(full[p++], source->data, 0, ggml_nbytes(source));
        }
        for (auto * t : {x, down_x}) {
            values.resize(ggml_nelements(t));
            for (size_t i = 0; i < values.size(); ++i) { values[i] = std::sin(float(i * 17 + 3) * 0.071f); }
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        }
        // Exactly nine slots, including Metal's physical page rounding.
        ggml_context_ptr meta(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t bytes = 0;
        for (auto * source : {c.up, c.gate, c.down}) {
            auto * t = ggml_new_tensor_3d(meta.get(), source->type, source->ne[0], source->ne[1], n_slots);
            bytes += (ggml_backend_buft_get_alloc_size(buft, t) + alignment - 1) / alignment * alignment;
        }
        const long page = sysconf(_SC_PAGESIZE);
        CHECK(page > 0);
        bytes = (bytes + size_t(page) - 1) / size_t(page) * size_t(page);
        cache = std::make_unique<llama_expert_cache>(bytes, std::vector<config>{c}, buft);
        CHECK(cache->stats().gpu && cache->stats().allocated_bytes == bytes);
        for (const auto & pair : cache->tensor_pairs()) { CHECK(pair.second->ne[2] == n_slots); }
    }
    ggml_backend_sched_ptr scheduler() const {
        ggml_backend_t backends[] = {gpu, cpu};
        ggml_backend_buffer_type_t types[] = {buft, ggml_backend_cpu_buffer_type()};
        return ggml_backend_sched_ptr(ggml_backend_sched_new(backends, types, 2, 256, false, false));
    }
    uint64_t bundle_bytes() const { return c.up->nb[2] + c.gate->nb[2] + c.down->nb[2]; }
};

void indexed_matmuls(fixture & f) {
    const int32_t selections[][top_k] = {
        {63, 17, 42, 3, 55, 8, 31, 0},
        {63, 17, 42, 3, 55, 8, 31, 0},
        {4, 17, 62, 3, 55, 27, 31, 1},
        {5, 60, 2, 61, 7, 59, 9, 58},
    };
    std::array<llama_expert_usage, n_expert> usage{};
    for (int e = 0; e < n_expert; ++e) { usage[e] = {0, uint32_t(e), 0}; }
    for (int i = 0; i < top_k; ++i) { usage[selections[0][i]].count = 100 - i; }
    CHECK(f.cache->set_usage(usage.data(), usage.size()));
    CHECK(f.cache->warm(top_k * f.bundle_bytes()));
    CHECK(f.cache->stats().hits == 0 && f.cache->stats().misses == 0);
    CHECK(f.cache->stats().bytes_copied == top_k * f.bundle_bytes());
    CHECK(f.cache->uploads().aliased_bytes == 0 && f.cache->uploads().submissions == 0);
    f.lease = f.cache->acquire_context();
    auto batch = reinterpret_cast<ggml_backend_metal_copy_batch_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(f.gpu)), "ggml_backend_metal_copy_batch"));
    CHECK(batch);
    uint64_t submitted = 99;
    CHECK(batch(nullptr, 0, &submitted) && submitted == 0);
    auto * compact = f.cache->tensor_pairs()[0].second;
    std::array<unsigned char, 64> old_bytes{}, new_bytes{};
    ggml_backend_tensor_get(compact, old_bytes.data(), 0, old_bytes.size());
    ggml_backend_metal_copy invalid[] = {{f.full[0], compact, 0, 0, 64},
                                        {f.full[0], compact, 0, ggml_nbytes(compact), 64}};
    CHECK(!batch(invalid, 2, &submitted) && submitted == 0);
    ggml_backend_tensor_get(compact, new_bytes.data(), 0, new_bytes.size());
    CHECK(old_bytes == new_bytes); // invalid later range must not partially upload
    invalid[1] = {f.c.up, compact, 0, 0, 64}; // CPU buffers must not be cast to Metal contexts
    CHECK(!batch(invalid, 2, &submitted) && submitted == 0);

    ggml_context_ptr ctx(ggml_init({ggml_graph_overhead_custom(256, false) + 256 * ggml_tensor_overhead(), nullptr, true}));
    auto sched = f.scheduler();
    auto * up = f.c.up; auto * gate = f.c.gate; auto * down = f.c.down;
    ggml_tensor * fused = nullptr; ggml_tensor * slots;
    f.cache->build(ctx.get(), 0, f.ids, slots, up, gate, down, fused, sched.get(), f.cpu);
    ggml_set_output(slots);
    auto * gf = ggml_new_graph_custom(ctx.get(), 256, false);
    std::array<ggml_tensor *, 3> cached{}, reference{};
    int p = 0;
    for (auto * weight : {up, gate, down}) {
        auto * input = p == 2 ? f.down_x : f.x;
        cached[p] = ggml_mul_mat_id(ctx.get(), weight, input, slots);
        reference[p] = ggml_mul_mat_id(ctx.get(), f.full[p], input, f.ids);
        f.cache->assign_matmul(sched.get(), f.cpu, cached[p], true);
        CHECK(ggml_backend_supports_op(f.gpu, reference[p]));
        ggml_backend_sched_set_tensor_backend(sched.get(), reference[p], f.gpu);
        ggml_set_output(cached[p]); ggml_set_output(reference[p]);
        ggml_build_forward_expand(gf, cached[p]);
        ggml_build_forward_expand(gf, reference[p]);
        ++p;
    }
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), gf));
    error_stats errors;
    for (const auto & selection : selections) {
        const auto before = f.cache->stats();
        const auto uploads = f.cache->uploads();
        ggml_backend_tensor_set(f.ids, selection, 0, sizeof(selection));
        CHECK(ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS);
        CHECK(!f.cache->failed());
        const auto after = f.cache->stats();
        CHECK(after.hits + after.misses == before.hits + before.misses + top_k);
        CHECK(after.bytes_copied - before.bytes_copied == (after.misses - before.misses) * f.bundle_bytes());
        CHECK(f.cache->uploads().submissions - uploads.submissions == uint64_t(after.misses != before.misses));
        if (after.misses != before.misses) {
            CHECK(f.cache->uploads().aliased_bytes == n_expert * f.bundle_bytes());
        }
        int32_t slot_ids[top_k];
        ggml_backend_tensor_get(slots, slot_ids, 0, sizeof(slot_ids));
        for (const auto & pair : f.cache->tensor_pairs()) {
            std::vector<unsigned char> bytes(pair.first->nb[2]);
            for (int i = 0; i < top_k; ++i) {
                CHECK(slot_ids[i] >= 0 && slot_ids[i] < n_slots);
                ggml_backend_tensor_get(pair.second, bytes.data(), slot_ids[i] * pair.second->nb[2], bytes.size());
                CHECK(std::memcmp(bytes.data(), static_cast<const char *>(pair.first->data) + selection[i] * pair.first->nb[2], bytes.size()) == 0);
            }
        }
        for (p = 0; p < 3; ++p) {
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), cached[p]) == f.gpu);
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), reference[p]) == f.gpu);
            std::vector<float> a(ggml_nelements(cached[p])), b(a.size());
            ggml_backend_tensor_get(cached[p], a.data(), 0, ggml_nbytes(cached[p]));
            ggml_backend_tensor_get(reference[p], b.data(), 0, ggml_nbytes(reference[p]));
            for (size_t i = 0; i < a.size(); ++i) { errors.add(a[i], b[i]); }
        }
    }
    errors.print("Metal Q4_K indexed matvec original64 vs compact9, up/gate/down, identical inputs");
    CHECK(errors.unequal == 0); // shape-only change must not be excused as rounding
    const auto s = f.cache->stats();
    std::printf("cache gpu=%d allocated=%llu budget=%llu hits=%llu misses=%llu evictions=%llu raw_upload_bytes=%llu\n",
                int(s.gpu), (unsigned long long) s.allocated_bytes, (unsigned long long) s.budget_bytes,
                (unsigned long long) s.hits, (unsigned long long) s.misses, (unsigned long long) s.evictions,
                (unsigned long long) s.bytes_copied);
}

struct routing_graph {
    ggml_context_ptr ctx;
    ggml_backend_sched_ptr sched;
    ggml_cgraph * gf;
    ggml_tensor * ids;
    ggml_tensor * ids_output;
    ggml_tensor * weights;
    ggml_tensor * weights_output;
    ggml_tensor * slots = nullptr;
    ggml_tensor * weighted_output = nullptr;
    routing_graph(fixture & f, bool cached, bool norm, bool force_gpu = true, bool gpu_producer = false,
                  bool production = false) {
        ctx.reset(ggml_init({ggml_graph_overhead_custom(256, false) + 256 * ggml_tensor_overhead(), nullptr, true}));
        sched = f.scheduler();
        gf = ggml_new_graph_custom(ctx.get(), 256, false);
        auto * logits = f.logits;
        if (gpu_producer) {
            // A GPU-produced router output seeds scheduler propagation, unlike
            // a preallocated GPU leaf. Scale by one preserves the exact input.
            logits = ggml_scale(ctx.get(), logits, 1.0f);
            ggml_backend_sched_set_tensor_backend(sched.get(), logits, f.gpu);
        }
        auto * probs = ggml_soft_max(ctx.get(), logits);
        ids = ggml_argsort_top_k(ctx.get(), probs, top_k);
        auto * selected = ids;
        auto * up = f.c.up; auto * gate = f.c.gate; auto * down = f.c.down;
        ggml_tensor * fused = nullptr;
        if (cached && !production) {
            // Retain the old topology as a separately labeled negative control.
            selected = f.cache->build(ctx.get(), 0, ids, slots, up, gate, down, fused, sched.get(), f.cpu);
        }
        weights = ggml_get_rows(ctx.get(), ggml_reshape_3d(ctx.get(), probs, 1, n_expert, 1), selected);
        if (norm) {
            auto * w = ggml_reshape_2d(ctx.get(), weights, top_k, 1);
            auto * sum = ggml_clamp(ctx.get(), ggml_sum_rows(ctx.get(), w), 6.103515625e-5f, INFINITY);
            weights = ggml_reshape_3d(ctx.get(), ggml_div(ctx.get(), w, sum), 1, top_k, 1);
        }
        // Output flags propagate through views: flagging IDs would pin ARGSORT,
        // and flagging normalized weights would pin DIV. Copies preserve reads
        // without changing which routing intermediates fusion may elide.
        ids_output = ggml_dup(ctx.get(), ids);
        weights_output = ggml_dup(ctx.get(), weights);
        ggml_set_output(ids_output);
        ggml_set_output(weights_output);
        ggml_build_forward_expand(gf, weights);
        if (cached && production) {
            // Match build_moe_ffn: finish original-ID routing before admission,
            // with a dependency on final weights that survives scheduler splitting.
            auto * original = f.cache->build(ctx.get(), 0, ids, slots, up, gate, down, fused,
                                             sched.get(), f.cpu, weights);
            CHECK(original->op == GGML_OP_MAP_CUSTOM2 && original->src[1] == weights);
            ggml_build_forward_expand(gf, slots);
        }
        auto * sink = ggml_mul_mat_id(ctx.get(), cached ? up : f.full[0], f.x, cached ? slots : ids);
        ggml_backend_sched_set_tensor_backend(sched.get(), sink, f.gpu);
        if (production) {
            weighted_output = ggml_mul(ctx.get(), sink, weights);
            ggml_set_output(weighted_output);
            ggml_backend_sched_set_tensor_backend(sched.get(), weighted_output, f.gpu);
        }
        ggml_build_forward_expand(gf, production ? weighted_output : sink);
        ggml_build_forward_expand(gf, ids_output);
        ggml_build_forward_expand(gf, weights_output);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            auto * node = ggml_graph_node(gf, i);
            if (force_gpu && node->op != GGML_OP_MAP_CUSTOM1 && node->op != GGML_OP_MAP_CUSTOM2 &&
                    node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE) {
                CHECK(ggml_backend_supports_op(f.gpu, node));
                ggml_backend_sched_set_tensor_backend(sched.get(), node, f.gpu);
            }
        }
        CHECK(ggml_backend_sched_alloc_graph(sched.get(), gf));
        std::printf("routing placement topology=%s cached=%d norm=%d forced_gpu=%d gpu_producer=%d:",
                    production ? "production" : "legacy-interposed", int(cached), int(norm), int(force_gpu), int(gpu_producer));
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            auto * node = ggml_graph_node(gf, i);
            auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), node);
            std::printf(" %s=%s", ggml_op_name(node->op), backend ? ggml_backend_name(backend) : "view");
        }
        std::puts("");
    }
    void compute(std::array<float, top_k> & w, std::array<int32_t, top_k> & selection,
                 std::vector<float> * weighted = nullptr) {
        CHECK(ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(weights_output, w.data(), 0, sizeof(w));
        ggml_backend_tensor_get(ids_output, selection.data(), 0, sizeof(selection));
        if (weighted) {
            CHECK(weighted_output);
            weighted->resize(ggml_nelements(weighted_output));
            ggml_backend_tensor_get(weighted_output, weighted->data(), 0, ggml_nbytes(weighted_output));
        }
    }
};

void routing(fixture & f, fusion_api & api, bool enabled, bool norm, bool force_gpu = true,
             bool gpu_producer = false, bool production = false) {
    api.enable(api.handle, enabled);
    routing_graph reference(f, false, norm, true, gpu_producer, production),
                  cached(f, true, norm, force_gpu, gpu_producer, production);
    error_stats errors, weighted_errors, oracle_errors;
    bool remapped_ids = false;
    uint64_t full_fusions = 0, cached_fusions = 0;
    for (int trial = 0; trial < 32; ++trial) {
        float logits[n_expert];
        for (int i = 0; i < n_expert; ++i) {
            logits[i] = 4.0f * std::sin(float(i * 197 + trial * 101 + 13) * 0.017f) + 0.003f * i;
        }
        ggml_backend_tensor_set(f.logits, logits, 0, sizeof(logits));
        std::array<float, top_k> a, b;
        std::array<int32_t, top_k> ia, ib;
        std::vector<float> wa, wb;
        api.reset(api.handle);
        reference.compute(a, ia, production ? &wa : nullptr);
        const auto reference_count = api.topk_count();
        full_fusions += reference_count;
        api.reset(api.handle);
        const auto before = f.cache->stats();
        const auto uploads = f.cache->uploads();
        cached.compute(b, ib, production ? &wb : nullptr);
        const auto after = f.cache->stats();
        CHECK(after.bytes_copied - before.bytes_copied == (after.misses - before.misses) * f.bundle_bytes());
        CHECK(f.cache->uploads().submissions - uploads.submissions == uint64_t(after.misses != before.misses));
        const auto cached_count = api.topk_count();
        cached_fusions += cached_count;
        CHECK(ia == ib);
        CHECK(!f.cache->failed());
        for (int i = 0; i < top_k; ++i) { errors.add(a[i], b[i]); }
        if (production) {
            if (reference_count != uint64_t(enabled) || cached_count != uint64_t(enabled)) {
                std::printf("production fusion mismatch trial=%d norm=%d enabled=%d reference=%llu cached=%llu\n",
                            trial, int(norm), int(enabled), (unsigned long long) reference_count, (unsigned long long) cached_count);
            }
            CHECK(reference_count == uint64_t(enabled) && cached_count == uint64_t(enabled));
            std::array<int32_t, n_expert> order;
            for (int i = 0; i < n_expert; ++i) { order[i] = i; }
            std::sort(order.begin(), order.end(), [&](int32_t x, int32_t y) {
                return logits[x] > logits[y] || (logits[x] == logits[y] && x < y);
            });
            const double max_logit = logits[order[0]];
            double denominator = 0;
            for (int i = 0; i < (norm ? top_k : n_expert); ++i) {
                denominator += std::exp(double(logits[order[i]]) - max_logit);
            }
            std::array<int32_t, top_k> slot_ids;
            ggml_backend_tensor_get(cached.slots, slot_ids.data(), 0, sizeof(slot_ids));
            for (int i = 0; i < top_k; ++i) {
                CHECK(ia[i] == order[i]);
                CHECK(slot_ids[i] >= 0 && slot_ids[i] < n_slots);
                remapped_ids |= slot_ids[i] != ia[i];
                const float expected = float(std::exp(double(logits[ia[i]]) - max_logit) / denominator);
                oracle_errors.add(expected, a[i]);
                oracle_errors.add(expected, b[i]);
            }
            CHECK(wa.size() == wb.size());
            for (size_t i = 0; i < wa.size(); ++i) { weighted_errors.add(wa[i], wb[i]); }
        }
    }
    char label[256];
    std::snprintf(label, sizeof(label), "routing topology=%s fusion=%s norm=%s cached_placement=%s gpu_producer=%d reference_topk_fusions=%llu cached_topk_fusions=%llu",
                  production ? "production" : "legacy-interposed", enabled ? "on" : "off", norm ? "on" : "off",
                  force_gpu ? "GPU" : "scheduler", int(gpu_producer),
                  (unsigned long long) full_fusions, (unsigned long long) cached_fusions);
    errors.print(label);
    if (production) {
        CHECK(remapped_ids);
        oracle_errors.print("production routing original-ID softmax oracle");
        weighted_errors.print("production routing weighted expert output");
        CHECK(errors.max_abs <= 1e-6 && oracle_errors.max_abs <= 1e-6 && weighted_errors.max_abs <= 1e-6);
        if (!enabled) { CHECK(weighted_errors.unequal == 0); }
    } else {
        CHECK(cached_fusions == 0);
    }
    if (!enabled) {
        CHECK(full_fusions == 0);
        if (force_gpu) { CHECK(errors.unequal == 0); }
    }
    // Legacy fusion-on differences remain diagnostic only; production topology
    // requires live fusion on both sides and strict checks independent of slots.
}

void shared_readback(ggml_backend_t gpu) {
    constexpr int n = 4096;
    ggml_context_ptr ctx(ggml_init({ggml_graph_overhead_custom(64, false) + 64 * ggml_tensor_overhead(), nullptr, true}));
    CHECK(ctx);
    auto * x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    auto * y = ggml_scale(ctx.get(), x, 2.0f);
    auto * gf = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), gpu));
    CHECK(buffer);
    std::vector<float> values(n), expected(n);
    std::vector<unsigned char> bytes(n * sizeof(float) + 8, 0);
    for (int trial = 0; trial < 3; ++trial) {
        for (int i = 0; i < n; ++i) {
            values[i] = float(i + trial);
            expected[i] = values[i] * 2.0f;
        }
        ggml_backend_tensor_set(x, values.data(), 0, ggml_nbytes(x));
        CHECK(ggml_backend_graph_compute_async(gpu, gf) == GGML_STATUS_SUCCESS);
        // Deliberately unaligned CPU destination and byte offset: shared reads
        // need no page-aligned destination MTLBuffer wrapper.
        ggml_backend_tensor_get_async(gpu, y, bytes.data() + 1, 3, ggml_nbytes(y) - 7);
        ggml_backend_synchronize(gpu);
        CHECK(std::memcmp(bytes.data() + 1, reinterpret_cast<const char *>(expected.data()) + 3, ggml_nbytes(y) - 7) == 0);
    }
    std::puts("shared async readback: producer completion, byte offsets, unaligned destination and reuse OK");
}
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        ggml_backend_load_all();
        auto * reg = ggml_backend_metal_reg();
        auto * metal = reg && ggml_backend_reg_dev_count(reg) ? ggml_backend_reg_dev_get(reg, 0) : nullptr;
        if (!metal) { std::puts("SKIP: no Metal device available"); return 77; }
        fusion_api api(metal);
        ggml_backend_ptr gpu(ggml_backend_dev_init(metal, nullptr)), cpu(ggml_backend_cpu_init());
        CHECK(gpu && cpu);
        ggml_backend_cpu_set_n_threads(cpu.get(), 2);
        shared_readback(gpu.get());
        for (auto binding : {upload_binding::standalone, upload_binding::full, upload_binding::split,
                             upload_binding::partial, upload_binding::empty, upload_binding::limited_standalone}) {
            mapped_uploads(gpu.get(), cpu.get(), false, binding);
            mapped_uploads(gpu.get(), cpu.get(), true, binding);
        }
        fixture f(gpu.get(), cpu.get());
        indexed_matmuls(f);
        routing(f, api, true, false);
        routing(f, api, true, true);
        routing(f, api, false, false);
        routing(f, api, false, true);
        routing(f, api, true, false, false);
        routing(f, api, true, true, false);
        routing(f, api, true, false, false, true);
        routing(f, api, true, true, false, true);
        for (bool enabled : {true, false}) {
            for (bool norm : {false, true}) {
                routing(f, api, enabled, norm, true, true, true);
            }
        }
        api.enable(api.handle, true);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
