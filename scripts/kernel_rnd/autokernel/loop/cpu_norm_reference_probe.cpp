// Fixed-input CPU RMS_NORM probe for the cpu_norm_rowsplit route. It runs the candidate
// ggml graph on a pinned thread team and prints the raw float32 output bits. Python
// (cpu_norm_reference.py) regenerates the same inputs, emulates HEAD's arithmetic
// bit-for-bit, and computes a float64 reference; no ggml reference is used.
//
//   probe <mode> <ne0> <ne1> <ne2> <ne3> <eps_bits_hex> <threads> <reps> <seed>
//
// mode: plain | inplace | view | fused | fused_bcast
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

// Must stay identical to cpu_norm_reference: VIEW_PAD extra columns, VIEW_OFFSET floats of
// leading offset (so the view's rows start off a 64-byte boundary).
static constexpr int64_t VIEW_PAD = 13;
static constexpr int64_t VIEW_OFFSET = 5;

static uint64_t mix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Exact float: a 24-bit significand times 2^(e-23), e in [-lo, hi). Bit-identical to
// cpu_norm_reference._value.
static float value(uint64_t seed, uint64_t index, int exp_lo, int exp_span) {
    const uint64_t h = mix(seed * 0x100000001B3ull + index);
    const int64_t mantissa = (int64_t) ((h & 0x7FFFFFull) | 0x800000ull);
    const int exponent = (int) ((h >> 24) % (uint64_t) exp_span) - exp_lo;
    const float magnitude = std::ldexp((float) mantissa, exponent - 23);
    return ((h >> 40) & 1) ? -magnitude : magnitude;
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
    if (argc != 10) return 2;
    const char * mode = argv[1];
    const bool inplace = std::strcmp(mode, "inplace") == 0;
    const bool view = std::strcmp(mode, "view") == 0;
    const bool fused = std::strcmp(mode, "fused") == 0;
    const bool fused_bcast = std::strcmp(mode, "fused_bcast") == 0;
    if (!inplace && !view && !fused && !fused_bcast && std::strcmp(mode, "plain") != 0) return 2;
    int64_t ne[4], threads, reps, seed;
    for (int i = 0; i < 4; ++i)
        if (!parse_i64(argv[2 + i], ne[i]) || ne[i] < 1 || ne[i] > 65536) return 2;
    if (ne[0] * ne[1] * ne[2] * ne[3] > (int64_t) 1 << 22) return 2;
    char * end = nullptr;
    const unsigned long eps_bits = std::strtoul(argv[6], &end, 16);
    if (!argv[6][0] || *end || eps_bits > 0xFFFFFFFFul) return 2;
    float eps;
    const uint32_t eps_u32 = (uint32_t) eps_bits;
    std::memcpy(&eps, &eps_u32, sizeof(eps));
    if (!parse_i64(argv[7], threads) || threads < 1 || threads > 256) return 2;
    if (!parse_i64(argv[8], reps) || reps < 1 || reps > 64) return 2;
    if (!parse_i64(argv[9], seed) || seed < 0) return 2;

    // Source tensor (the wider parent for a view), then the norm weight.
    const int64_t src_ne0 = view ? ne[0] + VIEW_PAD : ne[0];
    const int64_t src_count = src_ne0 * ne[1] * ne[2] * ne[3];
    std::vector<float> src(src_count);
    for (int64_t i = 0; i < src_count; ++i) src[i] = value((uint64_t) seed, (uint64_t) i, 12, 25);
    const bool weighted = fused || fused_bcast;
    const int64_t w_count = !weighted ? 0 : fused_bcast ? ne[0] : ne[0] * ne[1] * ne[2] * ne[3];
    std::vector<float> weight(w_count);
    for (int64_t i = 0; i < w_count; ++i)
        weight[i] = value((uint64_t) seed + 1, (uint64_t) i, 4, 9);

    ggml_init_params params = {64 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 4;
    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, src_ne0, ne[1], ne[2], ne[3]);
    ggml_tensor * x = view ? ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1],
                                          a->nb[2], a->nb[3], VIEW_OFFSET * sizeof(float))
                           : a;
    ggml_tensor * w = !weighted ? nullptr
                    : fused_bcast ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], 1, 1, 1)
                                  : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    ggml_tensor * norm = inplace ? ggml_rms_norm_inplace(ctx, x, eps) : ggml_rms_norm(ctx, x, eps);
    // RMS_NORM then MUL with the norm result as src0: the shape ggml_cpu_try_fuse_ops
    // fuses into ggml_compute_forward_rms_norm_mul_fused.
    ggml_tensor * out = weighted ? ggml_mul(ctx, norm, w) : norm;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 5;
    // An 8-thread team with fewer rows than threads is the narrow-row regime the route
    // exists to split; the suite's wide shapes keep the row-split path covered.
    ggml_backend_cpu_set_n_threads(backend, (int) threads);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 6;
    if (!ggml_is_contiguous(out)) return 8;

    uint64_t input_hash = fnv1a(src.data(), src.size() * sizeof(float));
    input_hash = fnv1a(weight.data(), weight.size() * sizeof(float), input_hash);
    if (w) ggml_backend_tensor_set(w, weight.data(), 0, weight.size() * sizeof(float));
    std::vector<float> output(ggml_nelements(out));
    std::vector<float> first;
    std::printf("AK_CPU_NORM_REFERENCE_V1 %s %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64
                " %08lx %" PRId64 " %" PRId64 " %" PRId64 "\n", mode, ne[0], ne[1], ne[2], ne[3],
                eps_bits, threads, reps, seed);
    std::printf("I %016" PRIx64 "\n", input_hash);
    for (int64_t rep = 0; rep < reps; ++rep) {
        // In-place overwrites the input: every repetition starts from the same bytes.
        ggml_backend_tensor_set(a, src.data(), 0, src.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 7;
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
        std::printf("D %" PRId64 " %016" PRIx64 "\n", rep,
                    fnv1a(output.data(), output.size() * sizeof(float)));
        if (rep == 0) first = output;
    }
    const int64_t rows = ne[1] * ne[2] * ne[3];
    for (int64_t row = 0; row < rows; ++row) {
        std::printf("O %" PRId64 " ", row);
        const unsigned char * bytes = (const unsigned char *) (first.data() + row * ne[0]);
        for (int64_t i = 0; i < ne[0] * (int64_t) sizeof(float); ++i) std::printf("%02x", bytes[i]);
        std::putchar('\n');
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
