// FA masked-KV-skip harness: bit-exactness (dump outputs) and kernel micro-benchmark.
// Links against whichever libggml it is run with (LD_LIBRARY_PATH / rpath), so the same binary compares the
// unpatched champion libs with the patched libs.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct fa_case {
    int64_t hsk, hsv, nh, nr2, kv, nb, n_seq, run, n_free;
    ggml_type type;
    float max_bias, softcap;
    int active_seq; // >=0: all queries belong to this sequence (perf shapes); <0: queries split over sequences
};

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// Save exactly the host bytes submitted to each backend tensor. Caller gives
// each process its own pre-created directory; a missing write is fatal.
static void save_input(const void * data, size_t len) {
    static unsigned index = 0;
    const char * dir = getenv("FA_INPUT_DIR");
    if (!dir || !*dir) { fprintf(stderr, "FA_INPUT_DIR is required\n"); exit(2); }
    char name[512];
    if (snprintf(name, sizeof(name), "%s/input_%02u.bin", dir, index++) >= (int) sizeof(name)) exit(2);
    FILE * f = fopen(name, "wbx");
    if (!f) { perror("input capture open"); exit(2); }
    if (fwrite(data, 1, len, f) != len || fclose(f) != 0) { fprintf(stderr, "input capture write failed\n"); exit(2); }
}

static void set_tensor_data(ggml_tensor * t, std::mt19937 & gen, int64_t n_f32) {
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> f(n_f32);
    for (auto & x : f) x = dis(gen);
    if (t->type == GGML_TYPE_F32) {
        save_input(f.data(), n_f32*sizeof(float));
        ggml_backend_tensor_set(t, f.data(), 0, n_f32*sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(n_f32);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n_f32);
        save_input(h.data(), n_f32*sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t, h.data(), 0, n_f32*sizeof(ggml_fp16_t));
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> h(n_f32);
        ggml_fp32_to_bf16_row(f.data(), h.data(), n_f32);
        save_input(h.data(), n_f32*sizeof(ggml_bf16_t));
        ggml_backend_tensor_set(t, h.data(), 0, n_f32*sizeof(ggml_bf16_t));
    } else {
        const int64_t ne0 = t->ne[0];
        std::vector<uint8_t> q(ggml_row_size(t->type, ne0) * (n_f32/ne0));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n_f32/ne0, ne0, nullptr);
        save_input(q.data(), q.size());
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

// returns output floats; if reps > 0 also measures mean time per compute in us
static std::vector<float> run_case(ggml_backend_t backend, const fa_case & c, int reps, double * us) {
    const int64_t kv_size = c.kv + 256;
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hsk, c.nb, c.nh*c.nr2, 1);
    ggml_tensor * k_cache = ggml_new_tensor_2d(ctx, c.type, c.hsk*c.nh, kv_size);
    ggml_tensor * k = ggml_view_4d(ctx, k_cache, c.hsk, c.nh, c.kv, 1,
            ggml_row_size(c.type, c.hsk), ggml_row_size(c.type, c.hsk*c.nh), ggml_row_size(c.type, c.hsk*c.nh*kv_size), 0);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * v_cache = nullptr;
    ggml_tensor * v = nullptr;
    if (c.hsk == 576) {
        v = ggml_view_4d(ctx, k_cache, c.hsv, c.nh, c.kv, 1,
                ggml_row_size(c.type, c.hsk), ggml_row_size(c.type, c.hsk*c.nh), ggml_row_size(c.type, c.hsk*c.nh*kv_size), 0);
    } else {
        v_cache = ggml_new_tensor_2d(ctx, c.type, c.hsv*c.nh, kv_size);
        v = ggml_view_4d(ctx, v_cache, c.hsv, c.nh, c.kv, 1,
                ggml_row_size(c.type, c.hsv), ggml_row_size(c.type, c.hsv*c.nh), ggml_row_size(c.type, c.hsv*c.nh*kv_size), 0);
    }
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.kv, c.nb, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(c.hsk), c.max_bias, c.softcap);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return {};
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 gen(1234);
    set_tensor_data(q, gen, ggml_nelements(q));
    set_tensor_data(k_cache, gen, ggml_nelements(k_cache));
    if (v_cache) set_tensor_data(v_cache, gen, ggml_nelements(v_cache));

    // unified-KV mask
    std::vector<int> owner(c.kv, -1);
    std::vector<int64_t> pos(c.kv, -1), n_cells(c.n_seq, 0);
    for (int64_t i = 0; i < c.kv - c.n_free; ++i) {
        const int s = (i / c.run) % c.n_seq;
        owner[i] = s; pos[i] = n_cells[s]++;
    }
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> mf(c.kv*c.nb);
    const int64_t group = (c.nb + c.n_seq - 1)/c.n_seq;
    for (int64_t i = 0; i < c.nb; ++i) {
        int s; int64_t n_grp, off;
        if (c.active_seq >= 0) { s = c.active_seq; n_grp = c.nb; off = i; }
        else { s = std::min<int64_t>(i/group, c.n_seq - 1); n_grp = std::min<int64_t>(group, c.nb - s*group); off = i - s*group; }
        const int64_t pos_q = std::max<int64_t>(0, n_cells[s] - n_grp + off);
        for (int64_t j = 0; j < c.kv; ++j) {
            mf[i*c.kv + j] = (owner[j] == s && pos[j] <= pos_q) ? dis(gen) : -INFINITY;
        }
    }
    std::vector<ggml_fp16_t> mh(mf.size());
    ggml_fp32_to_fp16_row(mf.data(), mh.data(), mf.size());
    save_input(mh.data(), mh.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m, mh.data(), 0, mh.size()*sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));

    if (reps > 0) {
        for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const auto t1 = std::chrono::high_resolution_clock::now();
        *us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <backend: ROCm0|CPU> <exact|perf> [threads]\n", argv[0]); return 1; }
    const std::string bname = argv[1], mode = argv[2];
    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    if (bname == "CPU") {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, argc > 3 ? atoi(argv[3]) : 8);
    } else {
        backend = ggml_backend_init_by_name(bname.c_str(), nullptr);
    }
    if (!backend) { fprintf(stderr, "no backend %s\n", bname.c_str()); return 1; }

    if (mode == "exact") {
        std::vector<fa_case> cases;
        for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_BF16}) {
            for (int64_t nb : {1, 2, 4, 8, 32, 75, 512}) {
                cases.push_back({128, 128, 4, 4, 8192, nb, 4, 1024, 1024, t, 0.0f, 0.0f, -1});
            }
        }
        for (int64_t hs : {64, 256}) for (int64_t nb : {1, 2, 8, 64, 512}) for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
            cases.push_back({hs, hs, 4, 6, 12288, nb, 3, 2048, 2560, t, 0.0f, 0.0f, -1});
        }
        cases.push_back({128, 128, 1, 8, 8192, 1, 8, 300, 1792, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({128, 128, 1, 8, 8192, 16, 8, 300, 1792, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({128, 128, 2, 4, 8192, 64, 16, 64, 512, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({128, 128, 4, 1, 8192, 4, 4, 1024, 1024, GGML_TYPE_F16, 8.0f, 0.0f, -1});
        cases.push_back({128, 128, 4, 4, 8192, 4, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 10.0f, -1});
        cases.push_back({80, 80, 4, 4, 8192, 8, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({40, 40, 4, 4, 8192, 3, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({192, 128, 2, 8, 8192, 4, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({192, 128, 2, 8, 8192, 512, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({576, 512, 1, 16, 8192, 4, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({576, 512, 1, 16, 8192, 256, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({512, 512, 1, 8, 8192, 4, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({320, 256, 1, 32, 8192, 1, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({320, 256, 1, 32, 8192, 3, 4, 1024, 1024, GGML_TYPE_F16, 0.0f, 0.0f, -1});
        cases.push_back({256, 256, 4, 6, 65536, 1, 4, 16384, 0, GGML_TYPE_Q8_0, 0.0f, 0.0f, 0});
        cases.push_back({256, 256, 4, 6, 65536, 8, 4, 16384, 0, GGML_TYPE_Q8_0, 0.0f, 0.0f, 0});
        cases.push_back({128, 128, 4, 4, 65536, 1, 4, 16384, 0, GGML_TYPE_F16, 0.0f, 0.0f, 0});
        uint64_t all = 1469598103934665603ull;
        for (size_t i = 0; i < cases.size(); ++i) {
            const fa_case & c = cases[i];
            if (getenv("FA_CASE") && atoi(getenv("FA_CASE")) != (int) i) continue;
            std::vector<float> r = run_case(backend, c, 0, nullptr);
            if (r.empty()) { printf("case %2zu unsupported\n", i); continue; }
            int nan = 0; for (float x : r) nan += std::isnan(x);
            const uint64_t h = fnv1a(r.data(), r.size()*sizeof(float));
            all ^= h; all *= 1099511628211ull;
            printf("case %2zu hs=%lld/%lld nh=%lld nr2=%lld kv=%lld nb=%lld nseq=%lld run=%lld free=%lld type=%s bias=%g cap=%g hash=%016llx nan=%d\n",
                i, (long long)c.hsk, (long long)c.hsv, (long long)c.nh, (long long)c.nr2, (long long)c.kv, (long long)c.nb, (long long)c.n_seq,
                (long long)c.run, (long long)c.n_free, ggml_type_name(c.type), c.max_bias, c.softcap, (unsigned long long) h, nan);
            char fn[256]; snprintf(fn, sizeof(fn), "out_%02zu.bin", i);
            if (getenv("FA_DUMP_DIR")) {
                std::string p = std::string(getenv("FA_DUMP_DIR")) + "/" + fn;
                FILE * f = fopen(p.c_str(), "wb"); fwrite(r.data(), sizeof(float), r.size(), f); fclose(f);
            }
        }
        printf("ALL %016llx\n", (unsigned long long) all);
    } else {
        // P3-like decode shapes (27B: hs 256, 4 KV heads, GQA 6, q8_0 KV): one active sequence owning `own` cells at
        // the front of the pool, `n_foreign` idle neighbours each owning `own_foreign` cells after it.
        const int reps = argc > 4 ? atoi(argv[4]) : 100;
        const int64_t own = 16384;
        for (int64_t nb : {1, 8}) {
            for (int64_t foreign : {0, 16384, 49152, 114688}) {
                // seq 0 owns cells [0, own); the idle neighbours own the following runs of `own` cells each
                fa_case c = {256, 256, 4, 6, own + foreign, nb, 1 + foreign/own, own, 0, GGML_TYPE_Q8_0, 0.0f, 0.0f, 0};
                double us = 0.0;
                run_case(backend, c, reps, &us);
                printf("perf nb=%lld own=%lld foreign=%lld n_kv=%lld : %.1f us/op\n", (long long)nb, (long long)own, (long long)foreign, (long long)c.kv, us);
                fflush(stdout);
            }
        }
    }
    ggml_backend_free(backend);
    return 0;
}
