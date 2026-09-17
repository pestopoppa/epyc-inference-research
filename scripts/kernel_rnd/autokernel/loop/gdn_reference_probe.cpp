// Independent fixed-input GDN probe. The expected output is computed in
// gdn_reference.py, not by ggml's CPU implementation.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <vector>

static constexpr int D = 64;
static constexpr int H = 2;
static constexpr int T = 4;
static constexpr int K = 2;
static constexpr int q_rows[T] = {0, 15, 32, 63};
static constexpr int k_rows[T] = {1, 16, 31, 48};

int main() {
    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 2;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, H, 1);
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 3;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 4;

    std::vector<float> qv(D*H*T, 0), kv(D*H*T, 0), vv(D*H*T);
    std::vector<float> gv(H*T, 0), bv(H*T, 0.5f), sv(D*D*H);
    for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
        qv[D*(h + H*t) + (q_rows[t] + 3*h)%D] = 1.0f;
        kv[D*(h + H*t) + (k_rows[t] + 3*h)%D] = 1.0f;
        for (int j = 0; j < D; ++j)
            vv[D*(h + H*t) + j] = (j%8 - 4)/16.0f + h/32.0f + t/64.0f;
    }
    for (int h = 0; h < H; ++h) for (int j = 0; j < D; ++j)
        for (int i = 0; i < D; ++i)
            sv[i + D*(j + D*h)] = (i%8 - 4)/128.0f + (j%8 - 4)/1024.0f + h/64.0f;
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size()*sizeof(float));
    ggml_backend_tensor_set(k, kv.data(), 0, kv.size()*sizeof(float));
    ggml_backend_tensor_set(v, vv.data(), 0, vv.size()*sizeof(float));
    ggml_backend_tensor_set(g, gv.data(), 0, gv.size()*sizeof(float));
    ggml_backend_tensor_set(beta, bv.data(), 0, bv.size()*sizeof(float));
    ggml_backend_tensor_set(state, sv.data(), 0, sv.size()*sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 5;
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
    std::printf("AK_GDN_REFERENCE_V1 %zu\n", result.size());
    for (float value : result) std::printf("%a\n", value);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
