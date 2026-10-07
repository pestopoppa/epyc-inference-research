// Fixed-input CPU FLASH_ATTN_EXT probe for the cpu_fa_schedule route. Compiled once per
// arm (anchor, candidate) and linked against that arm's ggml; cpu_fa_reference.py runs
// both on identical inputs and requires the output bits to agree. It prints digests,
// never judges: the anchor is the reference (the route admits only bit-exact changes).
//
//   probe <hsk> <hsv> <n_kv_heads> <gqa> <kv> <nb> <sinks 0|1> <mask causal|sparse|captured>
//         <layout cache|plain> <threads> <reps> <seed> [--mask-file <path>]
//
// layout cache: Q is [D, n_q_heads, nb] and K/V are [D, n_kv_heads, kv], each permuted
// (0, 2, 1, 3) as llama-graph does, so a KV cell's heads are interleaved exactly like the
// llama KV cache view. layout plain: contiguous [D, nb, n_q_heads] / [D, kv, n_kv_heads].
//
// Output lines: the header echo, `L <libggml-cpu path that served ggml_backend_cpu_init>`,
// `I <input digest>`, `D <rep> <output digest>` per repetition, and `R <row> <digest>`
// for every output row (row = token * n_q_heads + head) of repetition 0.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <dlfcn.h>

#include <algorithm>
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

// Uniform in [lo, hi), from 24 hash bits (exact in float).
static float uniform(uint64_t stream, uint64_t index, float lo, float hi) {
    const uint64_t h = mix(stream * 0x100000001B3ull + index);
    return lo + (hi - lo) * (float) (h >> 40) * (1.0f / 16777216.0f);
}

static uint64_t fnv1a(const void * data, size_t size, uint64_t h = 0xCBF29CE484222325ull) {
    const unsigned char * bytes = (const unsigned char *) data;
    for (size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

static bool parse_i64(const char * text, int64_t & out, int64_t lo, int64_t hi) {
    char * end = nullptr;
    const long long v = std::strtoll(text, &end, 10);
    if (!text[0] || *end || v < lo || v > hi) return false;
    out = v;
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 13 && argc != 15) return 2;
    int64_t hsk, hsv, nkvh, gqa, kv, nb, sinks, threads, reps, seed;
    if (!parse_i64(argv[1], hsk, 1, 1024) || !parse_i64(argv[2], hsv, 1, 1024) ||
        !parse_i64(argv[3], nkvh, 1, 64) || !parse_i64(argv[4], gqa, 1, 256) ||
        !parse_i64(argv[5], kv, 1, 1 << 20) || !parse_i64(argv[6], nb, 1, 512) ||
        !parse_i64(argv[7], sinks, 0, 1) || !parse_i64(argv[10], threads, 1, 512) ||
        !parse_i64(argv[11], reps, 1, 64) || !parse_i64(argv[12], seed, 0, INT64_MAX)) {
        return 2;
    }
    const char * mask_mode = argv[8];
    const char * layout = argv[9];
    const bool sparse = std::strcmp(mask_mode, "sparse") == 0;
    const bool captured = std::strcmp(mask_mode, "captured") == 0;
    if (!sparse && !captured && std::strcmp(mask_mode, "causal") != 0) return 2;
    if (captured != (argc == 15) || (captured && std::strcmp(argv[13], "--mask-file") != 0)) return 2;
    const bool cache = std::strcmp(layout, "cache") == 0;
    if (!cache && std::strcmp(layout, "plain") != 0) return 2;
    const int64_t nqh = nkvh * gqa;
    if (nb > kv) return 2;
    std::vector<ggml_fp16_t> captured_mask;
    if (captured) {
        FILE * file = std::fopen(argv[14], "rb");
        if (!file) return 10;
        const size_t count = (size_t) kv * nb;
        std::vector<unsigned char> bytes(count * sizeof(ggml_fp16_t));
        const bool valid = std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size()
                        && std::fgetc(file) == EOF && !std::ferror(file);
        const bool closed = std::fclose(file) == 0;
        if (!valid || !closed) return 10;
        captured_mask.resize(count);
        for (size_t i = 0; i < count; ++i) {
            captured_mask[i] = (ggml_fp16_t) (bytes[2*i] | (uint16_t) bytes[2*i + 1] << 8);
        }
    }

    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 4;
    // Storage tensors (data is written to these) and the FA operands (views of them).
    ggml_tensor * qs = cache ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hsk, nqh, nb)
                             : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hsk, nb, nqh);
    ggml_tensor * ks = cache ? ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hsk, nkvh, kv)
                             : ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hsk, kv, nkvh);
    ggml_tensor * vs = cache ? ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hsv, nkvh, kv)
                             : ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hsv, kv, nkvh);
    ggml_tensor * q = cache ? ggml_permute(ctx, qs, 0, 2, 1, 3) : qs;
    ggml_tensor * k = cache ? ggml_permute(ctx, ks, 0, 2, 1, 3) : ks;
    ggml_tensor * v = cache ? ggml_permute(ctx, vs, 0, 2, 1, 3) : vs;
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nb);
    ggml_tensor * s = sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nqh) : nullptr;
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / std::sqrt((float) hsk),
                                            0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, s);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 5;
    ggml_backend_cpu_set_n_threads(backend, (int) threads);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 6;
    if (!ggml_is_contiguous(out)) return 8;

    // Inputs in storage order. Q is scaled up so the online softmax meets new maxima at
    // irregular cells; sinks span [-10, 10) like test-backend-ops.
    uint64_t input_hash = 0xCBF29CE484222325ull;
    {
        std::vector<float> qv(ggml_nelements(qs));
        for (size_t i = 0; i < qv.size(); ++i) qv[i] = uniform((uint64_t) seed, i, -4.0f, 4.0f);
        ggml_backend_tensor_set(qs, qv.data(), 0, qv.size() * sizeof(float));
        input_hash = fnv1a(qv.data(), qv.size() * sizeof(float), input_hash);
    }
    for (int which = 0; which < 2; ++which) {
        ggml_tensor * t = which == 0 ? ks : vs;
        std::vector<float> f(ggml_nelements(t));
        for (size_t i = 0; i < f.size(); ++i)
            f[i] = uniform((uint64_t) seed + 1 + which, i, -1.0f, 1.0f);
        std::vector<ggml_fp16_t> h(f.size());
        ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) f.size());
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
        input_hash = fnv1a(h.data(), h.size() * sizeof(ggml_fp16_t), input_hash);
    }
    {
        if (captured) {
            ggml_backend_tensor_set(m, captured_mask.data(), 0,
                                    captured_mask.size() * sizeof(ggml_fp16_t));
            input_hash = fnv1a(captured_mask.data(), captured_mask.size() * sizeof(ggml_fp16_t), input_hash);
        } else {
            // Causal: row t sees cells [0, kv - nb + t]. Sparse (a DS41 top-k stand-in): also
            // drop ~1/4 of the visible cells, never the row's last one.
            std::vector<float> f((size_t) kv * nb);
            for (int64_t t = 0; t < nb; ++t) {
                const int64_t last = kv - nb + t;
                for (int64_t c = 0; c < kv; ++c) {
                    bool hidden = c > last;
                    if (sparse && c != last) hidden = hidden || (mix((uint64_t) seed * 31 + c) & 3) == 0;
                    f[t * kv + c] = hidden ? -INFINITY : 0.0f;
                }
            }
            std::vector<ggml_fp16_t> h(f.size());
            ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) f.size());
            ggml_backend_tensor_set(m, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
            input_hash = fnv1a(h.data(), h.size() * sizeof(ggml_fp16_t), input_hash);
        }
    }
    if (s) {
        std::vector<float> f(nqh);
        for (int64_t i = 0; i < nqh; ++i) f[i] = uniform((uint64_t) seed + 7, i, -10.0f, 10.0f);
        ggml_backend_tensor_set(s, f.data(), 0, f.size() * sizeof(float));
        input_hash = fnv1a(f.data(), f.size() * sizeof(float), input_hash);
    }

    // Which libggml-cpu actually serves this process (the arm's own, or the probe is void).
    Dl_info info;
    void * entry = dlsym(RTLD_DEFAULT, "ggml_backend_cpu_init");
    if (!entry || !dladdr(entry, &info) || !info.dli_fname) return 9;

    std::printf("AK_CPU_FA_REFERENCE_V1 %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64
                " %" PRId64 " %" PRId64 " %s %s %" PRId64 " %" PRId64 " %" PRId64 "\n",
                hsk, hsv, nkvh, gqa, kv, nb, sinks, mask_mode, layout, threads, reps, seed);
    std::printf("L %s\n", info.dli_fname);
    std::printf("I %016" PRIx64 "\n", input_hash);
    std::vector<float> output(ggml_nelements(out));
    std::vector<float> first;
    for (int64_t rep = 0; rep < reps; ++rep) {
        std::fill(output.begin(), output.end(), 0.0f);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 7;
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
        std::printf("D %" PRId64 " %016" PRIx64 "\n", rep,
                    fnv1a(output.data(), output.size() * sizeof(float)));
        if (rep == 0) first = output;
    }
    // out is [hsv, n_q_heads, nb]: row = token * n_q_heads + head.
    for (int64_t row = 0; row < nqh * nb; ++row) {
        std::printf("R %" PRId64 " %016" PRIx64 "\n", row,
                    fnv1a(first.data() + row * hsv, hsv * sizeof(float)));
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
