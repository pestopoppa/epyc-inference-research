// Fixed-input CPU probe for the DS41 hc_mixes chain: reshape -> RMS_NORM (no weight)
// -> MUL_MAT(F16 W, normed x), the graph `build_hc_mixes` emits (src/models/deepseek4.cpp).
// It runs the candidate's ggml graph (so a candidate fusion in ggml_cpu_try_fuse_ops is
// exercised exactly as in serving) on a pinned thread team and prints the raw float32
// output bits. Python (cpu_fusion_reference.py) regenerates the same inputs and judges
// the outputs against a float64 reference; no ggml reference is used.
//
//   probe <signed|positive> <K> <nt> <M> <eps_bits_hex> <threads> <reps> <seed>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint64_t mix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Exact float with a `bits`-bit significand times 2^(e - bits + 1), e in [-lo, span - lo).
// Bit-identical to cpu_fusion_reference._value.
static float value(uint64_t seed, uint64_t index, int bits, int exp_lo, int exp_span,
                   bool positive) {
    const uint64_t h = mix(seed * 0x100000001B3ull + index);
    const uint64_t top = 1ull << (bits - 1);
    const int64_t mantissa = (int64_t) ((h & (top - 1)) | top);
    const int exponent = (int) ((h >> 24) % (uint64_t) exp_span) - exp_lo;
    const float magnitude = std::ldexp((float) mantissa, exponent - (bits - 1));
    return (!positive && ((h >> 40) & 1)) ? -magnitude : magnitude;
}

static uint64_t fnv1a(const void * data, size_t size, uint64_t h = 0xCBF29CE484222325ull) {
    const unsigned char * bytes = (const unsigned char *) data;
    for (size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

static bool parse_i64(const char * text, int64_t & out) {
    char * end = nullptr;
    const long long v = std::strtoll(text, &end, 10);
    if (!text[0] || *end) return false;
    out = v;
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 9) return 2;
    const char * mode = argv[1];
    const bool positive = std::strcmp(mode, "positive") == 0;
    if (!positive && std::strcmp(mode, "signed") != 0) return 2;
    int64_t K, nt, M, threads, reps, seed;
    if (!parse_i64(argv[2], K) || K < 4 || K > 65536 || K % 4) return 2;
    if (!parse_i64(argv[3], nt) || nt < 1 || nt > 64) return 2;
    if (!parse_i64(argv[4], M) || M < 1 || M > 256) return 2;
    char * end = nullptr;
    const unsigned long eps_bits = std::strtoul(argv[5], &end, 16);
    if (!argv[5][0] || *end || eps_bits > 0xFFFFFFFFul) return 2;
    float eps;
    const uint32_t eps_u32 = (uint32_t) eps_bits;
    std::memcpy(&eps, &eps_u32, sizeof(eps));
    if (!parse_i64(argv[6], threads) || threads < 1 || threads > 256) return 2;
    if (!parse_i64(argv[7], reps) || reps < 1 || reps > 64) return 2;
    if (!parse_i64(argv[8], seed) || seed < 0) return 2;

    std::vector<float> x(K * nt);
    for (int64_t i = 0; i < K * nt; ++i)
        x[i] = value((uint64_t) seed, (uint64_t) i, 24, 12, 25, positive);
    std::vector<ggml_fp16_t> w(K * M);
    for (int64_t i = 0; i < K * M; ++i)
        w[i] = ggml_fp32_to_fp16(value((uint64_t) seed + 1, (uint64_t) i, 11, 6, 8, positive));

    ggml_init_params params = {64 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 4;
    // [K/4, 4, nt] reshaped to [K, nt]: the hc layout before `ggml_reshape_2d`.
    ggml_tensor * x3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K / 4, 4, nt);
    ggml_tensor * flat = ggml_reshape_2d(ctx, x3, K, nt);
    ggml_tensor * norm = ggml_rms_norm(ctx, flat, eps);
    ggml_tensor * wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
    ggml_tensor * out = ggml_mul_mat(ctx, wt, norm);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 5;
    ggml_backend_cpu_set_n_threads(backend, (int) threads);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 6;
    if (!ggml_is_contiguous(out) || out->ne[0] != M || out->ne[1] != nt) return 8;

    uint64_t input_hash = fnv1a(x.data(), x.size() * sizeof(float));
    input_hash = fnv1a(w.data(), w.size() * sizeof(ggml_fp16_t), input_hash);
    ggml_backend_tensor_set(wt, w.data(), 0, w.size() * sizeof(ggml_fp16_t));
    std::vector<float> output(ggml_nelements(out));
    std::vector<float> first;
    std::printf("AK_CPU_FUSION_REFERENCE_V1 %s %" PRId64 " %" PRId64 " %" PRId64
                " %08lx %" PRId64 " %" PRId64 " %" PRId64 "\n", mode, K, nt, M, eps_bits,
                threads, reps, seed);
    std::printf("I %016" PRIx64 "\n", input_hash);
    for (int64_t rep = 0; rep < reps; ++rep) {
        ggml_backend_tensor_set(x3, x.data(), 0, x.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 7;
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
        std::printf("D %" PRId64 " %016" PRIx64 "\n", rep,
                    fnv1a(output.data(), output.size() * sizeof(float)));
        if (rep == 0) first = output;
    }
    for (int64_t t = 0; t < nt; ++t) {
        std::printf("O %" PRId64 " ", t);
        const unsigned char * bytes = (const unsigned char *) (first.data() + t * M);
        for (int64_t i = 0; i < M * (int64_t) sizeof(float); ++i) std::printf("%02x", bytes[i]);
        std::putchar('\n');
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
