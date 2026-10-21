// DS41-W5a part 2: served-path NMSE of the CPU MXFP4 expert kernel.
//
// Why this harness exists: the AK served-shape calibration (test-backend-ops, use_ref) reports
// NMSE = 0.0 for every MXFP4 case, but test-backend-ops allocates weights in the plain CPU buffer,
// so it never exercises the CPU_REPACK buffer the 1017 server actually uses for the experts
// (mxfp4_8x8_q8_0 repack; ggml_gemv_mxfp4_8x8_q8_0 in forward_mul_mat_id). Both of its arms run the
// same generic kernel, so 0.0 is vacuous for the served path.
//
// This harness takes REAL expert slices from the 1017 GGUF (8 experts of one tensor), uploads them
// twice -- into the CPU_REPACK buffer (served path) and a plain CPU buffer (generic path) -- runs
// GGML_OP_MUL_MAT_ID at served widths (n_used = 6 of 8, tokens n = 1..512), and compares both
// against an f32-dequant reference computed in double: ref = sum_k dequant(W)[r,k] * x[k].
// NMSE = sum((y-ref)^2) / sum(ref^2). Cap (W5a gate): 5e-4.
//
// usage: mxfp4_repack_nmse <nthreads> <out.jsonl> <case>...
//   case = name:gguf_path:tensor_offset:k:m:n_exp_total:bcast(0|1)
//   bcast=1: src1 is [k,1,n] (gate/up, one row shared by all used experts); 0: [k,n_used,n] (down)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> out;
    size_t a = 0, b;
    while ((b = s.find(c, a)) != std::string::npos) { out.push_back(s.substr(a, b - a)); a = b + 1; }
    out.push_back(s.substr(a));
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <nthreads> <out.jsonl> <case>...\n", argv[0]); return 2; }
    const int nt = atoi(argv[1]);
    FILE * fo = fopen(argv[2], "w");
    if (!fo) { perror("out"); return 2; }

    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t buft_rep = nullptr;
    if (get_extra) {
        for (ggml_backend_buffer_type_t * p = get_extra(dev); p && *p; ++p) {
            if (strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) buft_rep = *p;
        }
    }
    if (!buft_rep) { fprintf(stderr, "no CPU_REPACK buffer type in this build\n"); return 3; }
    ggml_backend_buffer_type_t buft_plain = ggml_backend_cpu_buffer_type();
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nt);

    const int n_as = 8, n_used = 6;
    const int expert_ids[n_as] = {0, 47, 95, 143, 191, 239, 287, 383};
    const int ns[] = {1, 2, 3, 4, 5, 8, 32, 128, 512};
    const auto * tt = ggml_get_type_traits(GGML_TYPE_MXFP4);
    int n_fail = 0;

    for (int ci = 3; ci < argc; ++ci) {
        auto f = split(argv[ci], ':');
        if (f.size() != 7) { fprintf(stderr, "bad case %s\n", argv[ci]); return 2; }
        const std::string name = f[0], path = f[1];
        const long long toff = atoll(f[2].c_str());
        const int64_t k = atoll(f[3].c_str()), m = atoll(f[4].c_str());
        const int n_exp_total = atoi(f[5].c_str());
        const bool bcast = f[6] == "1";
        const size_t row_bytes = ggml_row_size(GGML_TYPE_MXFP4, k);
        const size_t exp_bytes = row_bytes * m;

        // read 8 real experts
        std::vector<uint8_t> raw(exp_bytes * n_as);
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { perror(path.c_str()); return 2; }
        for (int e = 0; e < n_as; ++e) {
            if (expert_ids[e] >= n_exp_total) { fprintf(stderr, "expert id out of range\n"); return 2; }
            const off_t off = (off_t) toff + (off_t) expert_ids[e] * (off_t) exp_bytes;
            size_t got = 0;
            while (got < exp_bytes) {
                ssize_t r = pread(fd, raw.data() + e * exp_bytes + got, exp_bytes - got, off + got);
                if (r <= 0) { perror("pread"); return 2; }
                got += r;
            }
        }
        close(fd);

        // f32 dequant of the same bytes (reference weights)
        std::vector<float> wf((size_t) n_as * m * k);
        for (size_t r = 0; r < (size_t) n_as * m; ++r) tt->to_float(raw.data() + r * row_bytes, wf.data() + r * k, k);

        // weights: repack + plain copies
        ggml_init_params wp = { 2 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * cw_rep = ggml_init(wp);
        ggml_context * cw_pl  = ggml_init(wp);
        ggml_tensor * a_rep = ggml_new_tensor_3d(cw_rep, GGML_TYPE_MXFP4, k, m, n_as);
        ggml_tensor * a_pl  = ggml_new_tensor_3d(cw_pl,  GGML_TYPE_MXFP4, k, m, n_as);
        ggml_backend_buffer_t bw_rep = ggml_backend_alloc_ctx_tensors_from_buft(cw_rep, buft_rep);
        ggml_backend_buffer_t bw_pl  = ggml_backend_alloc_ctx_tensors_from_buft(cw_pl,  buft_plain);
        ggml_backend_buffer_set_usage(bw_rep, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(bw_pl,  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(a_rep, raw.data(), 0, raw.size());
        ggml_backend_tensor_set(a_pl,  raw.data(), 0, raw.size());
        const bool repacked = a_rep->extra != nullptr;

        for (int n : ns) {
            const int ne11 = bcast ? 1 : n_used;
            std::mt19937_64 rng(0x5735A0000ULL + (uint64_t) ci * 1000003ULL + (uint64_t) n);
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<float> x((size_t) k * ne11 * n);
            for (auto & v : x) v = nd(rng);
            std::vector<int32_t> ids((size_t) n_used * n);
            for (int t = 0; t < n; ++t) {
                int perm[n_as];
                for (int i = 0; i < n_as; ++i) perm[i] = i;
                std::shuffle(perm, perm + n_as, rng);
                for (int u = 0; u < n_used; ++u) ids[(size_t) t * n_used + u] = perm[u];
            }
            // reference in double
            std::vector<double> ref((size_t) m * n_used * n);
            #pragma omp parallel for collapse(2) num_threads(nt) schedule(static)
            for (int t = 0; t < n; ++t) {
                for (int u = 0; u < n_used; ++u) {
                    const int e = ids[(size_t) t * n_used + u];
                    const float * xr = x.data() + ((size_t) t * ne11 + (bcast ? 0 : u)) * k;
                    for (int64_t r = 0; r < m; ++r) {
                        const float * wr = wf.data() + ((size_t) e * m + r) * k;
                        double acc = 0.0;
                        for (int64_t kk = 0; kk < k; ++kk) acc += (double) wr[kk] * (double) xr[kk];
                        ref[((size_t) t * n_used + u) * m + r] = acc;
                    }
                }
            }
            double res_nmse[2] = {0, 0}, res_maxabs[2] = {0, 0};
            std::vector<float> outs[2];
            for (int arm = 0; arm < 2; ++arm) {
                ggml_init_params cp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                ggml_context * cc = ggml_init(cp);
                ggml_tensor * b  = ggml_new_tensor_3d(cc, GGML_TYPE_F32, k, ne11, n);
                ggml_tensor * id = ggml_new_tensor_2d(cc, GGML_TYPE_I32, n_used, n);
                ggml_tensor * y  = ggml_mul_mat_id(cc, arm == 0 ? a_rep : a_pl, b, id);
                ggml_cgraph * gf = ggml_new_graph(cc);
                ggml_build_forward_expand(gf, y);
                ggml_backend_buffer_t bc = ggml_backend_alloc_ctx_tensors(cc, be);
                ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
                ggml_backend_tensor_set(id, ids.data(), 0, ids.size() * sizeof(int32_t));
                if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 4; }
                outs[arm].resize(ggml_nelements(y));
                ggml_backend_tensor_get(y, outs[arm].data(), 0, outs[arm].size() * sizeof(float));
                double num = 0, den = 0, mx = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const double d = (double) outs[arm][i] - ref[i];
                    num += d * d; den += ref[i] * ref[i];
                    mx = std::max(mx, std::fabs(d));
                }
                res_nmse[arm] = num / den; res_maxabs[arm] = mx;
                ggml_backend_buffer_free(bc);
                ggml_free(cc);
            }
            double num = 0, den = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = (double) outs[0][i] - (double) outs[1][i];
                num += d * d; den += (double) outs[1][i] * (double) outs[1][i];
            }
            const bool pass = res_nmse[0] <= 5e-4 && std::isfinite(res_nmse[0]);
            if (!pass) n_fail++;
            fprintf(fo, "{\"case\":\"%s\",\"k\":%lld,\"m\":%lld,\"n_as\":%d,\"n_used\":%d,\"n\":%d,\"bcast\":%d,"
                        "\"repack_buft\":\"%s\",\"repacked\":%s,\"threads\":%d,"
                        "\"nmse_repack_vs_f32ref\":%.6e,\"maxabs_repack\":%.6e,"
                        "\"nmse_plain_vs_f32ref\":%.6e,\"maxabs_plain\":%.6e,"
                        "\"nmse_repack_vs_plain\":%.6e,\"cap\":5e-4,\"pass\":%s}\n",
                    name.c_str(), (long long) k, (long long) m, n_as, n_used, n, bcast ? 1 : 0,
                    ggml_backend_buffer_name(bw_rep), repacked ? "true" : "false", nt,
                    res_nmse[0], res_maxabs[0], res_nmse[1], res_maxabs[1], num / den, pass ? "true" : "false");
            fflush(fo);
            fprintf(stderr, "%s n=%d repacked=%d nmse_rep=%.3e nmse_plain=%.3e rep_vs_plain=%.3e %s\n", name.c_str(), n,
                    (int) repacked, res_nmse[0], res_nmse[1], num / den, pass ? "PASS" : "FAIL");
        }
        ggml_backend_buffer_free(bw_rep);
        ggml_backend_buffer_free(bw_pl);
        ggml_free(cw_rep);
        ggml_free(cw_pl);
    }
    fclose(fo);
    ggml_backend_free(be);
    return n_fail == 0 ? 0 : 1;
}
