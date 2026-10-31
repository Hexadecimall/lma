#include "llama-expert-cache.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

namespace {
using config = llama_expert_cache::layer_config;

uint64_t minimum_budget(const std::vector<config> & configs) {
    const auto alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    uint64_t bytes = 0;
    for (const auto & c : configs) {
        for (const auto * t : {c.up, c.gate, c.down, c.gate_up}) {
            if (t) {
                bytes += (t->nb[2] * c.n_used + alignment - 1) / alignment * alignment;
            }
        }
    }
    return bytes;
}

struct graph {
    ggml_context_ptr ctx;
    ggml_backend_sched_ptr sched;
    ggml_cgraph * gf;
    ggml_tensor * out;
    std::vector<ggml_tensor *> compact, originals, slots;

    graph(ggml_backend_t cpu, llama_expert_cache * cache, const std::vector<config> & configs,
          ggml_tensor * input, ggml_tensor * probabilities, ggml_tensor * ids) {
        ctx.reset(ggml_init({ggml_graph_overhead_custom(256, false) + 256 * ggml_tensor_overhead(), nullptr, true}));
        CHECK(ctx);
        auto buft = ggml_backend_cpu_buffer_type();
        sched.reset(ggml_backend_sched_new(&cpu, &buft, 1, 256, false, false));
        gf = ggml_new_graph_custom(ctx.get(), 256, false);
        if (ids->type == GGML_TYPE_F32) {
            ids = ggml_argsort_top_k(ctx.get(), ids, 2);
        }
        auto * cur = ggml_reshape_3d(ctx.get(), input, 4, 1, 1);
        for (size_t il = 0; il < configs.size(); ++il) {
            const auto & c = configs[il];
            auto * up = c.up;
            auto * gate = c.gate;
            auto * down = c.down;
            auto * gate_up = c.gate_up;
            auto * slot_ids = ids;
            auto * original_ids = cache ? cache->build(ctx.get(), il, ids, slot_ids, up, gate, down, gate_up, sched.get(), cpu) : ids;
            originals.push_back(original_ids);
            slots.push_back(slot_ids);
            ggml_set_output(original_ids);
            ggml_set_output(slot_ids);
            ggml_tensor * u;
            ggml_tensor * g;
            if (gate_up) {
                compact.push_back(gate_up);
                auto * gu = ggml_mul_mat_id(ctx.get(), gate_up, cur, slot_ids);
                g = ggml_view_3d(ctx.get(), gu, 4, 2, 1, gu->nb[1], gu->nb[2], 0);
                u = ggml_view_3d(ctx.get(), gu, 4, 2, 1, gu->nb[1], gu->nb[2], 4 * gu->nb[0]);
            } else {
                compact.push_back(up);
                compact.push_back(gate);
                u = ggml_mul_mat_id(ctx.get(), up, cur, slot_ids);
                g = ggml_mul_mat_id(ctx.get(), gate, cur, slot_ids);
            }
            compact.push_back(down);
            auto * activated = ggml_swiglu_split(ctx.get(), g, u);
            auto * d = ggml_mul_mat_id(ctx.get(), down, activated, slot_ids);
            // Deliberately non-uniform probabilities: using slot IDs here is wrong.
            auto * weights = ggml_get_rows(ctx.get(), ggml_reshape_3d(ctx.get(), probabilities, 1, 4, 1), original_ids);
            auto * weighted = ggml_mul(ctx.get(), d, weights);
            cur = ggml_add(ctx.get(), ggml_view_1d(ctx.get(), weighted, 4, 0),
                                     ggml_view_1d(ctx.get(), weighted, 4, weighted->nb[1]));
            cur = ggml_reshape_3d(ctx.get(), cur, 4, 1, 1);
        }
        out = cur;
        ggml_build_forward_expand(gf, out);
        CHECK(ggml_backend_sched_alloc_graph(sched.get(), gf));
    }

    std::array<float, 4> compute() {
        CHECK(ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS);
        std::array<float, 4> result;
        ggml_backend_tensor_get(out, result.data(), 0, sizeof(result));
        return result;
    }
};

void quantized_bundles() {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    ggml_context_ptr sources(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}));
    config c{};
    c.n_used = 2;
    c.up = ggml_new_tensor_3d(sources.get(), GGML_TYPE_Q4_0, 32, 32, 4);
    c.gate = ggml_dup_tensor(sources.get(), c.up);
    c.down = ggml_dup_tensor(sources.get(), c.up);
    auto * ids = ggml_new_tensor_2d(sources.get(), GGML_TYPE_I32, 2, 1);
    auto * x = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 32, 1, 1);
    ggml_set_input(ids);
    ggml_set_input(x);
    ggml_backend_buffer_ptr source_buffer(ggml_backend_alloc_ctx_tensors(sources.get(), cpu.get()));
    CHECK(source_buffer);
    std::vector<float> data(32 * 32 * 4);
    for (size_t i = 0; i < data.size(); ++i) { data[i] = 0.1f * std::sin(float(i)); }
    for (auto * t : {c.up, c.gate, c.down}) {
        CHECK(ggml_quantize_chunk(t->type, data.data(), t->data, 0, 32 * 4, 32, nullptr) == ggml_nbytes(t));
    }
    for (int i = 0; i < 32; ++i) { static_cast<float *>(x->data)[i] = 0.01f * i; }
    llama_expert_cache cache(minimum_budget({c}), {c});
    ggml_context_ptr ctx(ggml_init({ggml_graph_overhead_custom(64, false) + 64 * ggml_tensor_overhead(), nullptr, true}));
    auto buft = ggml_backend_cpu_buffer_type();
    auto * backend = cpu.get();
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(&backend, &buft, 1, 64, false, false));
    auto * up = c.up;
    auto * gate = c.gate;
    auto * down = c.down;
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * slots;
    cache.build(ctx.get(), 0, ids, slots, up, gate, down, gate_up, sched.get(), cpu.get());
    ggml_set_output(slots);
    auto * actual = ggml_mul_mat_id(ctx.get(), up, x, slots);
    auto * expected = ggml_mul_mat_id(ctx.get(), c.up, x, ids);
    ggml_set_output(actual);
    ggml_set_output(expected);
    auto * gf = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(gf, actual);
    ggml_build_forward_expand(gf, expected);
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), gf));
    const int32_t selections[][2] = {{3, 1}, {0, 3}, {2, 1}};
    for (const auto & selection : selections) {
        ggml_backend_tensor_set(ids, selection, 0, sizeof(selection));
        CHECK(ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS);
        float a[64], b[64];
        ggml_backend_tensor_get(actual, a, 0, sizeof(a));
        ggml_backend_tensor_get(expected, b, 0, sizeof(b));
        CHECK(std::memcmp(a, b, sizeof(a)) == 0);
        int32_t mapped[2];
        ggml_backend_tensor_get(slots, mapped, 0, sizeof(mapped));
        for (const auto & pair : cache.tensor_pairs()) {
            CHECK(pair.first->type == pair.second->type);
            CHECK(pair.first->nb[2] == pair.second->nb[2]);
            for (int i = 0; i < 2; ++i) {
                CHECK(std::memcmp(static_cast<const char *>(pair.first->data) + selection[i] * pair.first->nb[2],
                                  static_cast<const char *>(pair.second->data) + mapped[i] * pair.second->nb[2],
                                  pair.first->nb[2]) == 0);
            }
        }
    }
    bool rejected = false;
    try { cache.validate_sources({}); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected); // heap weights cannot masquerade as production mapped sources
    std::puts("quantized bundles: byte/block-stride preservation and mul_mat_id equality OK");
}

void empty_graph_allocation() {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    CHECK(cpu);
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    ggml_context_ptr ctx(ggml_init({ggml_graph_overhead_custom(32, false) + 32 * ggml_tensor_overhead(), nullptr, true}));
    CHECK(ctx);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
    CHECK(alloc);
    auto * empty = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 0);
    ggml_set_input(empty);
    auto * sum = ggml_add(ctx.get(), empty, empty);
    auto * view = ggml_view_2d(ctx.get(), sum, 4, 0, sum->nb[1], 0);
    auto * reshaped = ggml_reshape_2d(ctx.get(), view, 2, 0);
    auto * out = ggml_add(ctx.get(), view, empty);
    ggml_set_output(out);
    ggml_set_output(reshaped);
    auto * gf = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, reshaped);
    ggml_tensor * tensors[] = {empty, sum, view, reshaped, out};
    std::array<void *, 5> addresses{};
    for (int trial = 0; trial < 3; ++trial) {
        // No non-empty tensor can accidentally supply backing for this chunk.
        // Repeated references formerly retried allocation with buffer != NULL.
        CHECK(ggml_gallocr_alloc_graph(alloc.get(), gf));
        CHECK(ggml_gallocr_get_buffer_size(alloc.get(), 0) > 0);
        for (size_t i = 0; i < addresses.size(); ++i) {
            auto * tensor = tensors[i];
            CHECK(ggml_nbytes(tensor) == 0);
            CHECK(tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer));
            CHECK(tensor->data != nullptr);
            if (trial == 0) { addresses[i] = tensor->data; }
            else { CHECK(tensor->data == addresses[i]); }
        }
        CHECK(view->data == sum->data && reshaped->data == sum->data);
        CHECK(ggml_backend_graph_compute(cpu.get(), gf) == GGML_STATUS_SUCCESS);
    }
    std::puts("empty CPU graph: shared inputs, empty views and repeated allocation/compute OK");
}

void history_and_empty_graphs() {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    ggml_context_ptr sources(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}));
    config c{};
    c.n_used = 2;
    c.up = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 4, 4, 4);
    c.gate = ggml_dup_tensor(sources.get(), c.up);
    c.down = ggml_dup_tensor(sources.get(), c.up);
    auto * input = ggml_new_tensor_1d(sources.get(), GGML_TYPE_F32, 4);
    auto * probabilities = ggml_new_tensor_1d(sources.get(), GGML_TYPE_F32, 4);
    auto * ids = ggml_new_tensor_2d(sources.get(), GGML_TYPE_I32, 2, 1);
    for (auto * t : {input, probabilities, ids}) { ggml_set_input(t); }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(sources.get(), cpu.get()));
    for (auto * t : {c.up, c.gate, c.down}) {
        for (size_t i = 0; i < ggml_nbytes(t) / sizeof(float); ++i) {
            static_cast<float *>(t->data)[i] = 0.001f * (1 + i);
        }
    }
    const float values[] = {0.1f, 0.2f, 0.3f, 0.4f};
    const int32_t selected[] = {2, 3};
    ggml_backend_tensor_set(input, values, 0, sizeof(values));
    ggml_backend_tensor_set(probabilities, values, 0, sizeof(values));
    ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
    llama_expert_cache cache(minimum_budget({c}), {c});
    CHECK(!cache.bind_upload_buffers({nullptr, buffer.get()})); // CPU buffers cannot become Metal aliases
    CHECK(cache.uploads().borrowed_bytes == 0 && cache.uploads().map_calls == 0);
    llama_expert_usage usage[] = {{0,0,0}, {0,1,20}, {0,2,1000}, {0,3,0}};
    CHECK(cache.set_usage(usage, 4));
    usage[1].expert = 0;
    CHECK(!cache.set_usage(usage, 4));
    usage[1].expert = 1;
    const uint64_t bundle_bytes = 3 * c.up->nb[2];
    CHECK(cache.warm(bundle_bytes - 1));
    CHECK(cache.stats().bytes_copied == 0);
    CHECK(cache.warm(bundle_bytes));
    CHECK(cache.stats().bytes_copied == bundle_bytes);
    CHECK(cache.stats().hits == 0 && cache.stats().misses == 0);
    CHECK(cache.uploads().aliased_bytes == 0 && cache.uploads().submissions == 0);
    graph cached(cpu.get(), &cache, {c}, input, probabilities, ids);
    graph reference(cpu.get(), nullptr, {c}, input, probabilities, ids);
    CHECK(cached.compute() == reference.compute());
    CHECK(cache.stats().hits == 1 && cache.stats().misses == 1);
    CHECK(cache.stats().bytes_copied == 2 * bundle_bytes);
    CHECK(cache.uploads().aliased_bytes == 0 && cache.uploads().submissions == 0);
    CHECK(cache.usage(usage, 4) == 4);
    CHECK(usage[2].count == 1001 && usage[3].count == 1);
    const auto before = cache.stats();
    cached.compute();
    CHECK(cache.stats().hits == before.hits + 2);
    CHECK(cache.stats().bytes_copied == before.bytes_copied);

    ggml_context_ptr empty_ctx(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
    auto * empty = ggml_new_tensor_2d(empty_ctx.get(), GGML_TYPE_I32, 2, 0);
    auto * up = c.up; auto * gate = c.gate; auto * down = c.down;
    ggml_tensor * gate_up = nullptr; ggml_tensor * slots = nullptr;
    CHECK(cache.build(empty_ctx.get(), 0, empty, slots, up, gate, down, gate_up, nullptr, cpu.get()) == empty);
    CHECK(slots == empty && up == c.up && down == c.down);
    CHECK(cache.stats().bytes_copied == before.bytes_copied);
    std::puts("history warmup: ranked first hit, cap, validation, frequencies and empty routing OK");
}

void run(bool fused, bool routed = false) {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    CHECK(cpu);
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    ggml_context_ptr sources(ggml_init({32 * ggml_tensor_overhead(), nullptr, true}));
    std::vector<config> configs;
    for (int il = 0; il < 2; ++il) {
        config c{};
        c.n_used = 2;
        if (fused) {
            c.gate_up = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 4, 8, 4);
        } else {
            c.up = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 4, 4, 4);
            c.gate = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 4, 4, 4);
        }
        c.down = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 4, 4, 4);
        configs.push_back(c);
    }
    auto * input = ggml_new_tensor_1d(sources.get(), GGML_TYPE_F32, 4);
    auto * probs = ggml_new_tensor_1d(sources.get(), GGML_TYPE_F32, 4);
    auto * ids = ggml_new_tensor_2d(sources.get(), routed ? GGML_TYPE_F32 : GGML_TYPE_I32, routed ? 4 : 2, 1);
    ggml_set_input(input);
    ggml_set_input(probs);
    ggml_set_input(ids);
    ggml_backend_buffer_ptr source_buffer(ggml_backend_alloc_ctx_tensors(sources.get(), cpu.get()));
    CHECK(source_buffer);
    for (size_t il = 0; il < configs.size(); ++il) {
        const auto & c = configs[il];
        int projection = 0;
        for (auto * t : {c.up, c.gate, c.down, c.gate_up}) {
            if (!t) { continue; }
            ++projection;
            auto * data = static_cast<float *>(t->data);
            for (int64_t e = 0; e < 4; ++e) {
                for (size_t j = 0; j < t->nb[2] / sizeof(float); ++j) {
                    data[e * t->nb[2] / sizeof(float) + j] = 0.03f * (1 + il + projection + e) + 0.001f * j;
                }
            }
        }
    }
    const float x[] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float p[] = {0.07f, 0.13f, 0.31f, 0.49f};
    ggml_backend_tensor_set(input, x, 0, sizeof(x));
    ggml_backend_tensor_set(probs, p, 0, sizeof(p));
    const uint64_t budget = minimum_budget(configs);
    bool rejected = false;
    try { llama_expert_cache too_small(budget - 1, configs); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected);
    llama_expert_cache cache(budget, configs);
    auto lease = cache.acquire_context();
    rejected = false;
    try { auto second = cache.acquire_context(); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected);
    lease.reset();
    lease = cache.acquire_context();
    graph cached(cpu.get(), &cache, configs, input, probs, ids);
    graph reference(cpu.get(), nullptr, configs, input, probs, ids);
    CHECK(cache.stats().allocated_bytes == budget);
    CHECK(!cache.stats().gpu);
    CHECK(cache.stats().bytes_copied == 0); // reservation is metadata-only
    for (auto * t : cached.compact) { CHECK(t->ne[2] == 2); }
    const int32_t selections[][2] = {{0, 1}, {2, 0}, {2, 3}, {0, 1}};
    for (const auto & selection : selections) {
        if (routed) {
            float logits[] = {0, 0, 0, 0};
            logits[selection[0]] = 2;
            logits[selection[1]] = 1;
            ggml_backend_tensor_set(ids, logits, 0, sizeof(logits));
        } else {
            ggml_backend_tensor_set(ids, selection, 0, sizeof(selection));
        }
        const auto expected = reference.compute();
        const auto actual = cached.compute();
        for (size_t i = 0; i < 4; ++i) {
            CHECK(std::isfinite(actual[i]));
            CHECK(std::fabs(actual[i] - expected[i]) < 1e-7f);
        }
        for (auto * original : cached.originals) {
            int32_t actual_ids[2];
            ggml_backend_tensor_get(original, actual_ids, 0, sizeof(actual_ids));
            CHECK(std::memcmp(actual_ids, selection, sizeof(selection)) == 0);
        }
    }
    auto stats = cache.stats();
    std::printf("cache counters: hits=%llu misses=%llu evictions=%llu copied=%llu\n",
                static_cast<unsigned long long>(stats.hits), static_cast<unsigned long long>(stats.misses),
                static_cast<unsigned long long>(stats.evictions), static_cast<unsigned long long>(stats.bytes_copied));
    CHECK(stats.hits == 4);
    CHECK(stats.misses == 12);
    CHECK(stats.evictions == 8);
    CHECK(stats.bytes_copied == 12 * 3 * 4 * 4 * sizeof(float));
    CHECK(stats.allocated_bytes <= stats.budget_bytes);
    // A completely hot active set performs no copying, in either projection layout.
    cached.compute();
    CHECK(cache.stats().hits == stats.hits + 4);
    CHECK(cache.stats().bytes_copied == stats.bytes_copied);

    // Invalid IDs must set a recoverable failure and never reach GET_ROWS or
    // MUL_MAT_ID unchecked, even if an application has no user abort callback.
    if (!routed) {
        ggml_backend_cpu_set_abort_callback(cpu.get(), [](void * data) {
            return static_cast<llama_expert_cache *>(data)->failed();
        }, &cache);
        const int32_t invalid[] = {-1, 4};
        ggml_backend_tensor_set(ids, invalid, 0, sizeof(invalid));
        const auto status = ggml_backend_sched_graph_compute(cached.sched.get(), cached.gf);
        CHECK(cache.failed());
        CHECK(status == GGML_STATUS_ABORTED);
    }
    std::printf("two-layer %s cache: budget=%llu hits/evictions/weighting/failure OK\n",
                fused ? "fused" : "separate", static_cast<unsigned long long>(budget));
}
}

int main() {
    try {
        ggml_backend_load_all();
        run(false);
        run(true);
        run(false, true);
        quantized_bundles();
        empty_graph_allocation();
        history_and_empty_graphs();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
