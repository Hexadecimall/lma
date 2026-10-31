#include "llama-expert-cache.h"
#include "llama-graph.h"
#include "llama-cparams.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

namespace {
using config = llama_expert_cache::layer_config;

struct result {
    std::vector<float> values;
    size_t prep_nodes = 0;
    size_t source_matmuls = 0;
    size_t compact_matmuls = 0;
};

result compute(ggml_backend_t cpu, llama_expert_cache * cache, const std::vector<config> & layers,
               ggml_tensor * input, ggml_tensor * router, ggml_tensor * outputs, int n_outputs) {
    auto buft = ggml_backend_cpu_buffer_type();
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(&cpu, &buft, 1, 512, false, false));
    llm_graph_result graph_result(512);
    llm_graph_params params{};
    params.arch = LLM_ARCH_QWEN3MOE;
    params.hparams.n_embd = 4;
    params.hparams.n_layer_all = 2;
    params.hparams.n_expert = 4;
    params.hparams.n_head_arr.fill(1);
    params.hparams.n_head_kv_arr.fill(1);
    params.hparams.n_expert_used_arr.fill(2);
    params.hparams.n_embd_head_k_full = 4;
    params.hparams.n_embd_head_v_full = 4;
    params.ubatch.n_tokens = 3;
    params.n_outputs = n_outputs;
    params.sched = sched.get();
    params.backend_cpu = cpu;
    llama_adapter_loras loras;
    params.loras = &loras;
    params.res = &graph_result;
    params.expert_cache = cache;
    llm_graph_context graph(params);
    auto * cur = input;
    for (int il = 0; il < 2; ++il) {
        // Match ordinary last-layer output selection, including zero-output
        // prompt microbatches followed by the final last-only-logits microbatch.
        if (il == 1) { cur = ggml_get_rows(graph_result.get_ctx(), cur, outputs); }
        const auto & c = layers[il];
        cur = graph.build_moe_ffn(cur, router, c.up, c.gate, c.down, nullptr,
                                 4, 2, LLM_FFN_SILU, true, 1.0f,
                                 LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
    }
    ggml_set_output(cur);
    auto * gf = graph_result.get_gf();
    ggml_build_forward_expand(gf, cur);
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), gf));
    result r;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_MAP_CUSTOM1 || node->op == GGML_OP_MAP_CUSTOM2) { ++r.prep_nodes; }
        if (node->op == GGML_OP_MUL_MAT_ID) {
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), node) == cpu);
            CHECK(node->src[0]->buffer != nullptr);
            if (node->src[0]->ne[2] == 4) { ++r.source_matmuls; }
            else { CHECK(node->src[0]->ne[2] == 2); ++r.compact_matmuls; }
        }
    }
    CHECK(ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS);
    r.values.resize(4 * n_outputs);
    if (!r.values.empty()) { ggml_backend_tensor_get(cur, r.values.data(), 0, r.values.size() * sizeof(float)); }
    return r;
}

void run(int n_outputs) {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    ggml_context_ptr sources(ggml_init({32 * ggml_tensor_overhead(), nullptr, true}));
    std::vector<config> layers;
    for (int il = 0; il < 2; ++il) {
        config c{};
        c.n_used = 2;
        c.up = ggml_new_tensor_3d(sources.get(), GGML_TYPE_F32, 4, 4, 4);
        c.gate = ggml_dup_tensor(sources.get(), c.up);
        c.down = ggml_dup_tensor(sources.get(), c.up);
        layers.push_back(c);
    }
    auto * input = ggml_new_tensor_2d(sources.get(), GGML_TYPE_F32, 4, 3);
    auto * router = ggml_new_tensor_2d(sources.get(), GGML_TYPE_F32, 4, 4);
    auto * outputs = ggml_new_tensor_1d(sources.get(), GGML_TYPE_I32, n_outputs);
    ggml_set_input(input);
    ggml_set_input(outputs);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(sources.get(), cpu.get()));
    CHECK(buffer);
    for (auto * t = ggml_get_first_tensor(sources.get()); t; t = ggml_get_next_tensor(sources.get(), t)) {
        if (t->type != GGML_TYPE_F32) { continue; }
        auto * data = static_cast<float *>(t->data);
        for (size_t i = 0; i < ggml_nbytes(t) / sizeof(float); ++i) { data[i] = 0.01f * (1 + i % 17); }
    }
    for (int i = 0; i < n_outputs; ++i) { static_cast<int32_t *>(outputs->data)[i] = n_outputs == 1 ? 2 : i; }
    llama_expert_cache cache(768, layers);
    CHECK(!cache.stats().gpu);
    const auto expected = compute(cpu.get(), nullptr, layers, input, router, outputs, n_outputs);
    const auto actual = compute(cpu.get(), &cache, layers, input, router, outputs, n_outputs);
    CHECK(actual.values.size() == expected.values.size());
    for (size_t i = 0; i < actual.values.size(); ++i) {
        CHECK(std::isfinite(actual.values[i]));
        CHECK(std::fabs(actual.values[i] - expected.values[i]) < 1e-7f);
    }
    CHECK(expected.prep_nodes == 0 && expected.compact_matmuls == 0);
    CHECK(actual.source_matmuls == (n_outputs == 1 ? 3 : 6));
    CHECK(actual.compact_matmuls == (n_outputs == 1 ? 3 : 0));
    CHECK(actual.prep_nodes == (n_outputs == 1 ? 2 : 0));
    CHECK(cache.stats().hits + cache.stats().misses == (n_outputs == 1 ? 2 : 0));
    std::printf("common MoE graph: batch=3 outputs=%d CPU prefill/empty/last-only logits OK\n", n_outputs);
}
}

int main() {
    try {
        ggml_backend_load_all();
        run(0);
        run(1);
        run(3);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
