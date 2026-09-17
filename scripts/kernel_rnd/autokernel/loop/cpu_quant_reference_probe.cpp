// Fixed-input CPU quant probe. Python independently decodes the quantized A
// bytes and computes the dot products; no ggml CPU reference is used.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int K = 256;
static constexpr int ROWS = 40;  // Crosses the Q8_0 32-row dispatch boundary.
static constexpr int TOKENS = 2;

static float activation(int token, int column) {
    for (int slot = 0; slot < 4; ++slot) {
        if (column == 3 + token * 11 + slot * 64) {
            return (slot & 1) ? -0.5f : 1.0f;
        }
    }
    return 0.0f;
}

static float weight(int expert, int row, int column) {
    return ((column * 13 + row * 7 + expert * 17) % 31 - 15) / 16.0f;
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    ggml_type type;
    if (std::strcmp(argv[1], "Q4_K") == 0) type = GGML_TYPE_Q4_K;
    else if (std::strcmp(argv[1], "Q5_K") == 0) type = GGML_TYPE_Q5_K;
    else if (std::strcmp(argv[1], "Q8_0") == 0) type = GGML_TYPE_Q8_0;
    else return 2;
    const bool fused = std::strcmp(argv[2], "FUSED_UP_GATE") == 0;
    const bool id_op = fused || std::strcmp(argv[2], "MUL_MAT_ID") == 0;
    if (!id_op && std::strcmp(argv[2], "MUL_MAT") != 0) return 2;
    const int experts = id_op ? 2 : 1;
    const size_t row_bytes = ggml_row_size(type, K);
    std::vector<float> source(K * ROWS * experts);
    for (int expert = 0; expert < experts; ++expert)
        for (int row = 0; row < ROWS; ++row)
            for (int column = 0; column < K; ++column)
                source[column + K * (row + ROWS * expert)] = weight(expert, row, column);
    std::vector<unsigned char> quantized(row_bytes * ROWS * experts);
    if (ggml_quantize_chunk(type, source.data(), quantized.data(), 0,
                            ROWS * experts, K, nullptr) != quantized.size()) return 3;
    // A distinct gate matrix is essential: reusing the up weights could hide
    // operand swaps or a missing half of the fused computation.
    std::vector<float> gate_source;
    std::vector<unsigned char> gate_quantized;
    if (fused) {
        gate_source.resize(source.size());
        gate_quantized.resize(quantized.size());
        for (int expert = 0; expert < experts; ++expert)
            for (int row = 0; row < ROWS; ++row)
                for (int column = 0; column < K; ++column)
                    gate_source[column + K * (row + ROWS * expert)] =
                        weight(expert + 3, row + 5, column);
        if (ggml_quantize_chunk(type, gate_source.data(), gate_quantized.data(), 0,
                                ROWS * experts, K, nullptr) != gate_quantized.size()) return 3;
    }

    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 4;
    ggml_tensor * a = id_op ? ggml_new_tensor_3d(ctx, type, K, ROWS, experts)
                            : ggml_new_tensor_2d(ctx, type, K, ROWS);
    ggml_tensor * gate_a = fused ? ggml_new_tensor_3d(ctx, type, K, ROWS, experts) : nullptr;
    ggml_tensor * b = id_op ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, TOKENS, 1)
                            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, TOKENS);
    ggml_tensor * ids = id_op ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, TOKENS, 1) : nullptr;
    ggml_tensor * up = id_op ? ggml_mul_mat_id(ctx, a, b, ids)
                             : ggml_mul_mat(ctx, a, b);
    ggml_tensor * gate = fused ? ggml_mul_mat_id(ctx, gate_a, b, ids) : nullptr;
    ggml_tensor * out = fused ? ggml_swiglu_split(ctx, gate, up) : up;
    if (fused) {
        gate->flags = (ggml_tensor_flag) (gate->flags | GGML_TENSOR_FLAG_COMPUTE);
        out->flags = (ggml_tensor_flag) (out->flags | GGML_TENSOR_FLAG_COMPUTE);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    if (fused) ggml_build_forward_expand(graph, up);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 5;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 6;
    std::vector<float> activations(K * TOKENS);
    for (int token = 0; token < TOKENS; ++token)
        for (int column = 0; column < K; ++column)
            activations[column + K * token] = activation(token, column);
    const int32_t expert_ids[TOKENS] = {1, 0};
    ggml_backend_tensor_set(a, quantized.data(), 0, quantized.size());
    if (fused) ggml_backend_tensor_set(gate_a, gate_quantized.data(), 0,
                                       gate_quantized.size());
    ggml_backend_tensor_set(b, activations.data(), 0, activations.size() * sizeof(float));
    if (ids) ggml_backend_tensor_set(ids, expert_ids, 0, sizeof(expert_ids));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 7;
    std::vector<float> output(ggml_nelements(out));
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    std::printf("AK_CPU_QUANT_REFERENCE_V1 %s %s %d %d %d %zu\n",
                argv[1], argv[2], K, ROWS, TOKENS, row_bytes);
    for (int expert = 0; expert < experts; ++expert)
        for (int row = 0; row < ROWS; ++row) {
            std::printf("A %d %d ", expert, row);
            const unsigned char * bytes = quantized.data() + row_bytes * (row + ROWS * expert);
            for (size_t index = 0; index < row_bytes; ++index) std::printf("%02x", bytes[index]);
            std::putchar('\n');
        }
    if (fused)
        for (int expert = 0; expert < experts; ++expert)
            for (int row = 0; row < ROWS; ++row) {
                std::printf("G %d %d ", expert, row);
                const unsigned char * bytes = gate_quantized.data() +
                    row_bytes * (row + ROWS * expert);
                for (size_t index = 0; index < row_bytes; ++index) std::printf("%02x", bytes[index]);
                std::putchar('\n');
            }
    for (int token = 0; token < TOKENS; ++token)
        for (int row = 0; row < ROWS; ++row)
            std::printf("O %d %d %a\n", token, row, output[row + ROWS * token]);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
