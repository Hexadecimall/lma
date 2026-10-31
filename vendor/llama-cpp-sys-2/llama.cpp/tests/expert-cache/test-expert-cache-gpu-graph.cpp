// Exercise production MoE graph construction without loading a GGUF. Bounded
// prefill/decode use full GPU experts as reference; larger/empty layers use CPU.
#include "llama-expert-cache.h"
#include "llama-graph.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

namespace {
constexpr int width = 1024, hidden = 256, experts = 16, used = 4, depth = 4;
using config = llama_expert_cache::layer_config;

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
                    std::strstr(labels[i], "GET_ROWS")) {
                result += counts[i];
            }
        }
        return result;
    }
};

struct fixture {
    ggml_backend_t gpu, cpu;
    ggml_context_ptr host, device;
    ggml_backend_buffer_ptr host_buffer, device_buffer;
    std::vector<config> sources, full;
    std::vector<ggml_tensor *> routers;
    ggml_tensor * input;
    ggml_tensor * outputs;

    fixture(ggml_backend_t gpu, ggml_backend_t cpu, int tokens, int n_outputs, bool fused) : gpu(gpu), cpu(cpu) {
        host.reset(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
        device.reset(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
        for (int il = 0; il < depth; ++il) {
            config c{};
            c.n_used = used;
            if (fused) {
                c.gate_up = ggml_new_tensor_3d(host.get(), GGML_TYPE_Q4_K, width, 2 * hidden, experts);
            } else {
                c.up = ggml_new_tensor_3d(host.get(), GGML_TYPE_Q4_K, width, hidden, experts);
                c.gate = ggml_dup_tensor(host.get(), c.up);
            }
            c.down = ggml_new_tensor_3d(host.get(), GGML_TYPE_Q4_K, hidden, width, experts);
            sources.push_back(c);
            config d{};
            d.n_used = used;
            d.up = c.up ? ggml_dup_tensor(device.get(), c.up) : nullptr;
            d.gate = c.gate ? ggml_dup_tensor(device.get(), c.gate) : nullptr;
            d.gate_up = c.gate_up ? ggml_dup_tensor(device.get(), c.gate_up) : nullptr;
            d.down = ggml_dup_tensor(device.get(), c.down);
            full.push_back(d);
            routers.push_back(ggml_new_tensor_2d(device.get(), GGML_TYPE_F32, width, experts));
        }
        input = ggml_new_tensor_2d(device.get(), GGML_TYPE_F32, width, tokens);
        outputs = ggml_new_tensor_1d(host.get(), GGML_TYPE_I32, n_outputs);
        ggml_set_input(input);
        ggml_set_input(outputs);
        host_buffer.reset(ggml_backend_alloc_ctx_tensors(host.get(), cpu));
        device_buffer.reset(ggml_backend_alloc_ctx_tensors(device.get(), gpu));
        CHECK(host_buffer && device_buffer);
        std::vector<float> values;
        for (int il = 0; il < depth; ++il) {
            const auto & c = sources[il];
            const auto & d = full[il];
            ggml_tensor * src[] = {c.up, c.gate, c.down, c.gate_up};
            ggml_tensor * dst[] = {d.up, d.gate, d.down, d.gate_up};
            for (int p = 0; p < 4; ++p) {
                if (!src[p]) { continue; }
                values.resize(src[p]->ne[0] * src[p]->ne[1]);
                for (int e = 0; e < experts; ++e) {
                    for (size_t i = 0; i < values.size(); ++i) {
                        const unsigned v = unsigned(i * 1664525 + (e + 1) * 1013904223u + il * 971 + p * 137);
                        values[i] = (float((v ^ (v >> 13)) % 65521) / 65521.0f - 0.5f) * 0.04f;
                    }
                    auto * bytes = static_cast<char *>(src[p]->data) + e * src[p]->nb[2];
                    CHECK(ggml_quantize_chunk(src[p]->type, values.data(), bytes, 0, src[p]->ne[1], src[p]->ne[0], nullptr) == src[p]->nb[2]);
                }
                ggml_backend_tensor_set(dst[p], src[p]->data, 0, ggml_nbytes(src[p]));
            }
            values.resize(width * experts);
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.1f * std::sin(float(i * 37 + il * 101 + 7) * 0.017f);
            }
            ggml_backend_tensor_set(routers[il], values.data(), 0, ggml_nbytes(routers[il]));
        }
        std::vector<int32_t> selected(n_outputs);
        for (int i = 0; i < n_outputs; ++i) { selected[i] = n_outputs == 1 ? tokens - 1 : i; }
        if (!selected.empty()) { ggml_backend_tensor_set(outputs, selected.data(), 0, selected.size() * sizeof(int32_t)); }
    }
};

struct graph {
    llama_adapter_loras loras; // result retains params.loras after construction
    llm_graph_result result{2048};
    ggml_backend_sched_ptr sched;
    ggml_tensor * output;
    std::vector<ggml_tensor *> layers;
    size_t gpu_matmuls = 0, cpu_matmuls = 0;

    graph(fixture & f, llama_expert_cache * cache, int tokens, int n_outputs, int slots, bool fused) {
        const char * name = cache ? "cache" : "reference";
        std::printf("GPU graph %s: build begin\n", name);
        std::vector<std::pair<ggml_tensor *, bool>> matmuls;
        const auto pairs = cache ? cache->tensor_pairs() : std::vector<std::pair<ggml_tensor *, ggml_tensor *>>{};
        ggml_backend_t backends[] = {f.gpu, f.cpu};
        ggml_backend_buffer_type_t bufts[] = {ggml_backend_get_default_buffer_type(f.gpu), ggml_backend_cpu_buffer_type()};
        sched.reset(ggml_backend_sched_new(backends, bufts, 2, 2048, false, true));
        llm_graph_params params{};
        params.arch = LLM_ARCH_QWEN3MOE;
        params.hparams.n_embd = width;
        params.hparams.n_layer_all = depth;
        params.hparams.n_expert = experts;
        params.hparams.n_head_arr.fill(1);
        params.hparams.n_head_kv_arr.fill(1);
        params.hparams.n_expert_used_arr.fill(used);
        params.hparams.n_embd_head_k_full = width;
        params.hparams.n_embd_head_v_full = width;
        params.ubatch.n_tokens = tokens;
        params.n_outputs = n_outputs;
        params.sched = sched.get();
        params.backend_cpu = f.cpu;
        params.loras = &loras;
        params.res = &result;
        params.expert_cache = cache;
        llm_graph_context builder(params);
        auto * ctx = result.get_ctx();
        auto * cur = f.input;
        for (int il = 0; il < depth; ++il) {
            if (il == depth - 1) { cur = ggml_get_rows(ctx, cur, f.outputs); }
            auto * residual = cur;
            cur = ggml_rms_norm(ctx, cur, 1e-6f);
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, f.gpu);
            const int64_t local_tokens = cur->ne[1];
            const bool on_gpu = local_tokens > 0 && local_tokens <= llama_expert_cache::max_stream_tokens;
            const auto & weights = cache || !on_gpu ? f.sources[il] : f.full[il];
            const int first_node = ggml_graph_n_nodes(result.get_gf());
            cur = builder.build_moe_ffn(cur, f.routers[il], weights.up, weights.gate, weights.down, nullptr,
                                        experts, used, LLM_FFN_SILU, true, 1.0f,
                                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il, nullptr, weights.gate_up);
            ggml_build_forward_expand(result.get_gf(), cur);
            size_t layer_matmuls = 0;
            for (int i = first_node; i < ggml_graph_n_nodes(result.get_gf()); ++i) {
                auto * node = ggml_graph_node(result.get_gf(), i);
                if (node->op == GGML_OP_MUL_MAT_ID) {
                    matmuls.emplace_back(node, on_gpu);
                    ++layer_matmuls;
                    if (cache && on_gpu) {
                        CHECK(node->src[2]->ne[1] <= cache->max_batch_tokens(il));
                        CHECK(node->src[0]->ne[2] == slots);
                        CHECK(slots == experts || node->src[0]->ne[2] < experts);
                        CHECK(std::any_of(pairs.begin(), pairs.end(), [&](const auto & pair) {
                            return pair.second == node->src[0];
                        }));
                    }
                }
                if (!cache) {
                    auto * target = node->op == GGML_OP_MUL_MAT_ID && !on_gpu ? f.cpu : f.gpu;
                    if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) { continue; }
                    CHECK(ggml_backend_supports_op(target, node));
                    ggml_backend_sched_set_tensor_backend(sched.get(), node, target);
                }
            }
            const int64_t tile = cache && on_gpu ? cache->max_batch_tokens(il) : std::max<int64_t>(1, local_tokens);
            const size_t tiles = cache && on_gpu ? (local_tokens + tile - 1) / tile : 1;
            CHECK(layer_matmuls == (fused ? 2 : 3) * tiles);
            std::printf("GPU graph %s: layer=%d local_tokens=%lld tile_tokens=%lld expert_matmuls=%zu backend=%s\n",
                        name, il, (long long) local_tokens, (long long) tile, layer_matmuls, on_gpu ? "GPU" : "CPU");
            cur = ggml_add(ctx, cur, residual);
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, f.gpu);
            ggml_set_output(cur);
            layers.push_back(cur);
            ggml_build_forward_expand(result.get_gf(), cur);
        }
        output = cur;
        std::printf("GPU graph %s: allocation begin\n", name);
        CHECK(ggml_backend_sched_alloc_graph(sched.get(), result.get_gf()));
        std::printf("GPU graph %s: allocation complete\n", name);
        for (const auto & entry : matmuls) {
            auto * node = entry.first;
            const bool on_gpu = entry.second;
            auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), node);
            CHECK(backend == (on_gpu ? f.gpu : f.cpu));
            CHECK(node->src[0]->buffer);
            if (on_gpu) {
                ++gpu_matmuls;
                CHECK(ggml_backend_supports_buft(f.gpu, ggml_backend_buffer_get_type(node->src[0]->buffer)));
                if (cache) {
                    // Identity after splitting rejects scheduler-created full-source GPU copies.
                    CHECK(std::any_of(pairs.begin(), pairs.end(), [&](const auto & pair) {
                        return pair.second == node->src[0] && pair.second->ne[2] == slots &&
                               pair.first->buffer && ggml_backend_buffer_is_host(pair.first->buffer);
                    }));
                }
            } else {
                ++cpu_matmuls;
                CHECK(ggml_backend_buffer_is_host(node->src[0]->buffer));
                CHECK(node->src[0]->ne[2] == experts);
            }
        }
    }

    std::vector<std::vector<float>> compute() {
        CHECK(ggml_backend_sched_graph_compute(sched.get(), result.get_gf()) == GGML_STATUS_SUCCESS);
        std::vector<std::vector<float>> values;
        for (auto * layer : layers) {
            values.emplace_back(ggml_nelements(layer));
            if (!values.back().empty()) { ggml_backend_tensor_get(layer, values.back().data(), 0, ggml_nbytes(layer)); }
        }
        return values;
    }
};

void run(ggml_backend_t gpu, ggml_backend_t cpu, fusion_api & api, bool fusion,
         int tokens, int n_outputs, int slots, bool fused) {
    api.enable(api.handle, fusion);
    fixture f(gpu, cpu, tokens, n_outputs, fused);
    // --slots controls the byte budget without increasing full fixture allocations.
    const uint64_t budget = depth * 3 * ggml_row_size(GGML_TYPE_Q4_K, width) * hidden * slots;
    llama_expert_cache cache(budget, f.sources, ggml_backend_get_default_buffer_type(gpu));
    CHECK(cache.stats().gpu && cache.stats().allocated_bytes <= budget);
    const int64_t tile = slots == experts ? llama_expert_cache::max_stream_tokens : slots / used;
    for (int il = 0; il < depth; ++il) { CHECK(cache.max_batch_tokens(il) == tile); }
    std::printf("GPU cache budget=%llu allocated=%llu slots_per_layer=%d tile_tokens=%lld projections=%s\n",
                (unsigned long long) budget, (unsigned long long) cache.stats().allocated_bytes,
                slots, (long long) tile, fused ? "fused" : "separate");
    graph reference(f, nullptr, tokens, n_outputs, slots, fused), cached(f, &cache, tokens, n_outputs, slots, fused);
    uint64_t reference_fusions = 0, cached_fusions = 0;
    for (int trial = 0; trial < 4; ++trial) {
        std::vector<float> input(width * tokens);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = std::sin(float(i * 17 + trial * 101 + 3) * 0.071f); }
        ggml_backend_tensor_set(f.input, input.data(), 0, input.size() * sizeof(float));
        api.reset(api.handle);
        const auto a = reference.compute();
        const auto reference_count = api.topk_count();
        reference_fusions += reference_count;
        api.reset(api.handle);
        const auto b = cached.compute();
        const auto cached_count = api.topk_count();
        cached_fusions += cached_count;
        // Empty final-layer routing does not dispatch a kernel. Every other
        // layer must exercise at least the base softmax/top-k/gather fusion.
        const uint64_t expected = fusion ? depth - (n_outputs == 0 ? 1 : 0) : 0;
        CHECK(reference_count == expected && cached_count == expected);
        CHECK(!cache.failed());
        for (size_t il = 0; il < a.size(); ++il) {
            CHECK(a[il].size() == b[il].size());
            float max_abs = 0;
            for (size_t i = 0; i < a[il].size(); ++i) {
                CHECK(std::isfinite(a[il][i]) && std::isfinite(b[il][i]));
                if (!fusion) { CHECK(a[il][i] == b[il][i]); }
                max_abs = std::max(max_abs, std::fabs(a[il][i] - b[il][i]));
            }
            std::printf("GPU production MoE fusion=%s tokens=%d outputs=%d trial=%d layer=%zu max_abs=%.9g\n",
                        fusion ? "on" : "off", tokens, n_outputs, trial, il, max_abs);
            CHECK(max_abs <= 1e-6f);
        }
    }
    const auto admissions = cache.stats().hits + cache.stats().misses;
    if (cached.gpu_matmuls > 0) { CHECK(admissions > 0); }
    else { CHECK(admissions == 0); }
    std::printf("production GPU graph fusion=%s tokens=%d outputs=%d slots=%d projections=%s: reference_GPU_matmuls=%zu cached_GPU_matmuls=%zu CPU_fallback_matmuls=%zu reference_topk_fusions=%llu cached_topk_fusions=%llu\n",
                fusion ? "on" : "off", tokens, n_outputs, slots, fused ? "fused" : "separate",
                reference.gpu_matmuls, cached.gpu_matmuls, cached.cpu_matmuls,
                (unsigned long long) reference_fusions, (unsigned long long) cached_fusions);
}
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 5 || argc % 2 != 1 || std::strcmp(argv[1], "--fusion") != 0 || std::strcmp(argv[3], "--case") != 0 ||
            (std::strcmp(argv[2], "on") != 0 && std::strcmp(argv[2], "off") != 0)) {
        std::fprintf(stderr, "usage: %s --fusion on|off --case decode|prefill-last|prefill-all|prefill-empty|prefill-boundary|prefill-fallback [--tokens 1..17] [--slots 4..16] [--projections separate|fused]\n", argv[0]);
        return 1;
    }
    try {
        const bool fusion = std::strcmp(argv[2], "on") == 0;
        int tokens = 3, slots = used + 1;
        bool fused = false;
        const bool decode = std::strcmp(argv[4], "decode") == 0;
        const bool boundary = std::strcmp(argv[4], "prefill-boundary") == 0;
        const bool fallback = std::strcmp(argv[4], "prefill-fallback") == 0;
        if (decode) { tokens = 1; }
        if (boundary || fallback) { tokens = llama_expert_cache::max_stream_tokens + (fallback ? 1 : 0); }
        auto number = [](const char * value, int minimum, int maximum) {
            char * end = nullptr;
            const long n = std::strtol(value, &end, 10);
            if (!*value || *end || n < minimum || n > maximum) {
                throw std::runtime_error("numeric argument outside supported fixture range");
            }
            return int(n);
        };
        for (int i = 5; i < argc; i += 2) {
            if (std::strcmp(argv[i], "--tokens") == 0) {
                tokens = number(argv[i + 1], 1, llama_expert_cache::max_stream_tokens + 1);
            } else if (std::strcmp(argv[i], "--slots") == 0) {
                slots = number(argv[i + 1], used, experts);
            } else if (std::strcmp(argv[i], "--projections") == 0) {
                CHECK(std::strcmp(argv[i + 1], "fused") == 0 || std::strcmp(argv[i + 1], "separate") == 0);
                fused = std::strcmp(argv[i + 1], "fused") == 0;
            } else {
                throw std::runtime_error("unknown option");
            }
        }
        int n_outputs = 1;
        if (decode) {
            CHECK(tokens == 1);
        } else if (std::strcmp(argv[4], "prefill-last") == 0) {
            n_outputs = 1;
        } else if (std::strcmp(argv[4], "prefill-all") == 0 || boundary || fallback) {
            n_outputs = tokens;
        } else if (std::strcmp(argv[4], "prefill-empty") == 0) {
            n_outputs = 0;
        } else {
            throw std::runtime_error("unknown prefill/decode case");
        }
        std::printf("GPU production MoE isolated case=%s fusion=%s tokens=%d slots=%d projections=%s\n",
                    argv[4], argv[2], tokens, slots, fused ? "fused" : "separate");
        ggml_backend_load_all();
        auto * reg = ggml_backend_metal_reg();
        auto * dev = reg && ggml_backend_reg_dev_count(reg) ? ggml_backend_reg_dev_get(reg, 0) : nullptr;
        if (!dev) { std::puts("SKIP: no Metal device"); return 77; }
        fusion_api api(dev);
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev, nullptr)), cpu(ggml_backend_cpu_init());
        CHECK(gpu && cpu);
        ggml_backend_cpu_set_n_threads(cpu.get(), 2);
        run(gpu.get(), cpu.get(), api, fusion, tokens, n_outputs, slots, fused);
        api.enable(api.handle, true);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
