// AutoKernel CPU_REPACK MUL_MAT_ID probe (`cpu_repack_cases.py`, gate `check_cpu_repack_case_set`).
//
// WHY. test-backend-ops allocates every weight in the plain CPU buffer, so no served-shape case reaches
// the CPU_REPACK extra buffer that the DS41-1017 server uses for every MXFP4 expert (mxfp4_8x8_q8_0:
// repack.cpp forward_mul_mat_id -> ggml_gemv_mxfp4_8x8_q8_0 -> gemv_q4_b32_8x8_q8_0_lut_avx). Its MXFP4
// NMSE therefore reads exactly 0.0 by construction. This probe (generalised from the DS41-W5a harness
// `data/ds41-1017-w5-20261021/w5a/mxfp4_repack_nmse.cpp`, research 73aab47cc) uploads the same expert
// bytes twice -- into CPU_REPACK (the served path) and a plain CPU buffer -- runs GGML_OP_MUL_MAT_ID at
// each requested width and compares both against a double-precision reference that decodes the stored
// bytes WITHOUT ggml (an independent scalar decoder for MXFP4 / Q4_0 / IQ4_NL, cross-checked against
// ggml's own to_float).
//
// The probe MEASURES; it never judges. Bounds live in the calibrated manifest and are applied in Python.
// Engagement is proven per case, not assumed:
//   * extra:    tensor->extra is set by the CPU_REPACK buffer (checked BEFORE set_tensor, which would
//               dereference a null traits pointer for a type the repack does not claim);
//   * layout:   the repacked bytes differ from the source bytes;
//   * witness:  the repack arm is re-run with one stored scale perturbed IN THE REPACKED BUFFER; exactly
//               output row 0 of expert 0 must move (and only for slots routed to expert 0), and restoring
//               the byte must reproduce the clean output bit for bit. This proves the measured arm reads
//               the repacked layout through the repack kernel.
//
// TEST-ONLY FIXTURE (never in the gate's binary): compiled with -DAK_REPACK_FIXTURE=1|2 the probe leaves
// a scale perturbed in the repacked buffer for the measured runs, i.e. it behaves like a repack GEMV that
// reads a wrong scale. 1 = lane-0 scale of EVERY repacked block (a kernel misreading one scale lane);
// 2 = one block of one row, x4 (the minimal localized fault; aggregate NMSE cannot see it, max_rel can).
// The fixture literal only exists under the macro, and the gate refuses any binary carrying it.
//
// usage: cpu_repack_probe <nthreads> <cases.tsv> <out.jsonl>
//   cases.tsv, one case per line, TAB separated (lines starting with '#' ignored):
//     name type k m n_as n_used bcast widths src path offset n_exp_total expert_ids seed
//     type   = MXFP4 | Q4_0 | IQ4_NL
//     bcast  = 1: src1 [k,1,n] (gate/up: one row shared by the used experts); 0: [k,n_used,n] (down)
//     widths = comma list of token counts
//     src    = gguf (path/offset/n_exp_total/expert_ids name n_as real experts) | synth (seeded
//              Gaussian weights quantized with ggml_quantize_chunk; path '-', offset 0, ids '-')
// exit: 0 every case measured (engaged or not -- the gate judges); 2 usage/input error; 3 no CPU_REPACK;
//       4 compute failure.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#define AK_CPU_REPACK_PROBE_MARKER "AK_CPU_REPACK_PROBE_V1"
#define AK_CPU_REPACK_CASE_SET "cpu_repack_mmid_v1"
#define AK_CPU_REPACK_TYPES "MXFP4,Q4_0,IQ4_NL"

#if defined(AK_REPACK_FIXTURE)
#if AK_REPACK_FIXTURE == 1
#define AK_FIXTURE_NAME "AK_REPACK_FIXTURE_LANE0_ALL_BLOCKS"
#elif AK_REPACK_FIXTURE == 2
#define AK_FIXTURE_NAME "AK_REPACK_FIXTURE_ONE_BLOCK_X4"
#else
#error "AK_REPACK_FIXTURE must be 1 or 2"
#endif
#endif

static const char * const k_marker_literals[] = {AK_CPU_REPACK_PROBE_MARKER, AK_CPU_REPACK_CASE_SET,
                                                 AK_CPU_REPACK_TYPES};

// ---------------------------------------------------------------- deterministic inputs (no libstdc++ RNG)
static uint64_t splitmix64(uint64_t & s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double uniform01(uint64_t & s) { return ((splitmix64(s) >> 11) + 0.5) * 0x1.0p-53; }
static void gaussians(uint64_t seed, float * out, size_t n) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; i += 2) {
        const double u1 = uniform01(s), u2 = uniform01(s);
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = (float) (r * std::cos(6.283185307179586 * u2));
        if (i + 1 < n) out[i + 1] = (float) (r * std::sin(6.283185307179586 * u2));
    }
}
static uint64_t fnv1a(const void * p, size_t n, uint64_t h = 0xCBF29CE484222325ULL) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001B3ULL; }
    return h;
}

// ---------------------------------------------------------------- independent scalar decoders (no ggml)
static double half_to_double(uint16_t h) {
    const int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    double v;
    if (exp == 0) v = std::ldexp((double) man, -24);
    else if (exp == 31) v = man ? NAN : INFINITY;
    else v = std::ldexp((double) (man | 0x400), exp - 25);
    return sign ? -v : v;
}
static const int8_t k_mxfp4_values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static const int8_t k_iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

struct TypeInfo { ggml_type type; size_t block_bytes; };
static bool type_info(const std::string & name, TypeInfo & out) {
    if (name == "MXFP4")  { out = {GGML_TYPE_MXFP4, 17};  return true; }   // u8 e (E8M0) + 16 nibble bytes
    if (name == "Q4_0")   { out = {GGML_TYPE_Q4_0, 18};   return true; }   // f16 d + 16 nibble bytes
    if (name == "IQ4_NL") { out = {GGML_TYPE_IQ4_NL, 18}; return true; }   // f16 d + 16 nibble bytes
    return false;
}
// Decode one 32-element block into doubles. Element j (0..15) is the low nibble of qs[j], j+16 the high.
static void decode_block(ggml_type t, const uint8_t * blk, double * y) {
    double d;
    const uint8_t * qs;
    if (t == GGML_TYPE_MXFP4) {
        d = std::ldexp(1.0, (int) blk[0] - 128);   // E8M0 "half" scale: the stored values are 2x E2M1
        qs = blk + 1;
        for (int j = 0; j < 16; ++j) { y[j] = d * k_mxfp4_values[qs[j] & 0xF]; y[j + 16] = d * k_mxfp4_values[qs[j] >> 4]; }
        return;
    }
    uint16_t h;
    memcpy(&h, blk, 2);
    d = half_to_double(h);
    qs = blk + 2;
    if (t == GGML_TYPE_Q4_0) {
        for (int j = 0; j < 16; ++j) { y[j] = d * ((qs[j] & 0xF) - 8); y[j + 16] = d * ((qs[j] >> 4) - 8); }
    } else {
        for (int j = 0; j < 16; ++j) { y[j] = d * k_iq4nl_values[qs[j] & 0xF]; y[j + 16] = d * k_iq4nl_values[qs[j] >> 4]; }
    }
}

// ---------------------------------------------------------------- repacked-buffer scale perturbation
// Every claimed type repacks to an x8 block whose first bytes are the 8 per-row scales of 8 interleaved
// rows (block_mxfp4x8: u8 e[8]; block_q4_0x8 / block_iq4_nlx8: ggml_half d[8]), then 128 qs bytes.
// The repack walks row groups of 8, then k blocks: block (g, b) sits at index g * (k/32) + b, and
// lane i of it is row 8g + i. `factor_log2` scales that lane's scale by 2^factor_log2.
static size_t x8_block_bytes(ggml_type t) { return t == GGML_TYPE_MXFP4 ? 8 + 128 : 16 + 128; }
static void perturb_lane(uint8_t * base, ggml_type t, size_t block_index, int lane, int factor_log2) {
    uint8_t * blk = base + block_index * x8_block_bytes(t);
    if (t == GGML_TYPE_MXFP4) {
        int e = blk[lane] + factor_log2;
        if (e > 254) e = blk[lane] - factor_log2;  // stay finite; any change is a perturbation
        blk[lane] = (uint8_t) e;
        return;
    }
    uint16_t h;
    memcpy(&h, blk + 2 * lane, 2);
    int exp = (h >> 10) & 0x1F;
    int ne = exp + factor_log2;
    if (exp == 0 || ne >= 31) ne = exp > factor_log2 ? exp - factor_log2 : exp + 1;  // keep it a normal f16
    h = (uint16_t) ((h & ~(0x1F << 10)) | ((ne & 0x1F) << 10));
    memcpy(blk + 2 * lane, &h, 2);
}

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> out;
    size_t a = 0, b;
    while ((b = s.find(c, a)) != std::string::npos) { out.push_back(s.substr(a, b - a)); a = b + 1; }
    out.push_back(s.substr(a));
    return out;
}

struct Case {
    std::string name, type_name, src, path;
    TypeInfo ti{};
    int64_t k = 0, m = 0;
    int n_as = 0, n_used = 0;
    bool bcast = false;
    std::vector<int> widths, expert_ids;
    long long offset = 0;
    int n_exp_total = 0;
    uint64_t seed = 0;
};

static bool parse_case(const std::string & line, Case & c, std::string & err) {
    auto f = split(line, '\t');
    if (f.size() != 14) { err = "expected 14 TAB-separated fields"; return false; }
    c.name = f[0]; c.type_name = f[1];
    if (!type_info(c.type_name, c.ti)) { err = "unsupported type " + c.type_name; return false; }
    c.k = atoll(f[2].c_str()); c.m = atoll(f[3].c_str());
    c.n_as = atoi(f[4].c_str()); c.n_used = atoi(f[5].c_str());
    c.bcast = f[6] == "1";
    for (auto & w : split(f[7], ',')) c.widths.push_back(atoi(w.c_str()));
    c.src = f[8]; c.path = f[9]; c.offset = atoll(f[10].c_str()); c.n_exp_total = atoi(f[11].c_str());
    if (c.src == "gguf") for (auto & e : split(f[12], ',')) c.expert_ids.push_back(atoi(e.c_str()));
    c.seed = strtoull(f[13].c_str(), nullptr, 10);
    if (c.k <= 0 || c.k % 32 || c.m <= 0 || c.m % 8 || c.n_as < 1 || c.n_used < 1 || c.n_used > c.n_as) {
        err = "bad dims"; return false;
    }
    for (int w : c.widths) if (w < 1 || w > 4096) { err = "bad width"; return false; }
    if (c.src == "gguf" && ((int) c.expert_ids.size() != c.n_as || c.n_exp_total < c.n_as)) { err = "bad gguf ids"; return false; }
    for (int e : c.expert_ids) if (e < 0 || e >= c.n_exp_total) { err = "expert id out of range"; return false; }
    if (c.src != "gguf" && c.src != "synth") { err = "src must be gguf or synth"; return false; }
    return true;
}

static void json_str(FILE * f, const char * k, const std::string & v) { fprintf(f, "\"%s\":\"%s\"", k, v.c_str()); }

// One MUL_MAT_ID run of `a` on (x, ids); returns false on compute failure.
static bool run_mmid(ggml_backend_t be, ggml_tensor * a, int64_t k, int ne11, int n, int n_used,
                     const std::vector<float> & x, const std::vector<int32_t> & ids, std::vector<float> & out) {
    ggml_init_params cp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * cc = ggml_init(cp);
    ggml_tensor * b  = ggml_new_tensor_3d(cc, GGML_TYPE_F32, k, ne11, n);
    ggml_tensor * id = ggml_new_tensor_2d(cc, GGML_TYPE_I32, n_used, n);
    ggml_tensor * y  = ggml_mul_mat_id(cc, a, b, id);
    ggml_cgraph * gf = ggml_new_graph(cc);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t bc = ggml_backend_alloc_ctx_tensors(cc, be);
    ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(id, ids.data(), 0, ids.size() * sizeof(int32_t));
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.resize(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(bc);
    ggml_free(cc);
    return ok;
}

// Inputs are keyed by the case NAME (not its position in the case file), so a case measures the same
// activations and routing whichever subset of cases a run carries.
static void make_inputs(const Case & c, int n, bool force_expert0, std::vector<float> & x, std::vector<int32_t> & ids) {
    const uint64_t key = fnv1a(c.name.data(), c.name.size());
    const int ne11 = c.bcast ? 1 : c.n_used;
    x.resize((size_t) c.k * ne11 * n);
    gaussians(0x414B52455041434BULL ^ key ^ ((uint64_t) n << 48) ^ (force_expert0 ? 0x5A5AULL : 0), x.data(), x.size());
    ids.resize((size_t) c.n_used * n);
    uint64_t s = 0x72657061636B6964ULL ^ (key * 31) ^ ((uint64_t) n << 48) ^ (force_expert0 ? 0xA5A5ULL : 0);
    std::vector<int> perm(c.n_as);
    for (int t = 0; t < n; ++t) {
        for (int i = 0; i < c.n_as; ++i) perm[i] = i;
        for (int i = c.n_as - 1; i > 0; --i) { int j = (int) (splitmix64(s) % (uint64_t) (i + 1)); std::swap(perm[i], perm[j]); }
        if (force_expert0) { auto it = std::find(perm.begin(), perm.end(), 0); std::swap(*it, perm[0]); }
        for (int u = 0; u < c.n_used; ++u) ids[(size_t) t * c.n_used + u] = perm[u];
    }
}

int main(int argc, char ** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <nthreads> <cases.tsv> <out.jsonl>\n", argv[0]); return 2; }
    const int nt = atoi(argv[1]);
    if (nt < 1) { fprintf(stderr, "bad nthreads\n"); return 2; }
    FILE * fc = fopen(argv[2], "r");
    if (!fc) { perror("cases"); return 2; }
    std::vector<Case> cases;
    {
        char buf[8192];
        int lineno = 0;
        while (fgets(buf, sizeof(buf), fc)) {
            ++lineno;
            std::string line(buf);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            Case c; std::string err;
            if (!parse_case(line, c, err)) { fprintf(stderr, "cases line %d: %s\n", lineno, err.c_str()); return 2; }
            cases.push_back(c);
        }
        fclose(fc);
    }
    if (cases.empty()) { fprintf(stderr, "no cases\n"); return 2; }
    FILE * fo = fopen(argv[3], "w");
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
    const char * iqk_env = getenv("GGML_IQK");
#if defined(AK_REPACK_FIXTURE)
    const char * fixture = AK_FIXTURE_NAME;
#else
    const char * fixture = "none";
#endif
    fprintf(fo, "{\"kind\":\"header\",\"marker\":\"%s\",\"case_set\":\"%s\",\"types\":\"%s\",\"fixture\":\"%s\","
                "\"threads\":%d,\"ggml_iqk\":\"%s\",\"repack_buft\":%s,\"n_cases\":%zu}\n",
            k_marker_literals[0], k_marker_literals[1], k_marker_literals[2], fixture, nt,
            iqk_env ? iqk_env : "", buft_rep ? "true" : "false", cases.size());
    fflush(fo);
    if (!buft_rep) { fprintf(stderr, "no CPU_REPACK buffer type in this build\n"); fclose(fo); return 3; }
    ggml_backend_buffer_type_t buft_plain = ggml_backend_cpu_buffer_type();
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nt);

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case & c = cases[ci];
        const ggml_type t = c.ti.type;
        const int64_t nblk = c.k / 32;
        const size_t row_bytes = (size_t) nblk * c.ti.block_bytes;
        if (row_bytes != ggml_row_size(t, c.k)) { fprintf(stderr, "%s: block size mismatch\n", c.name.c_str()); return 2; }
        const size_t exp_bytes = row_bytes * c.m;

        // ---- stored weight bytes (real experts or seeded synthetic) ----
        std::vector<uint8_t> raw(exp_bytes * c.n_as);
        if (c.src == "gguf") {
            int fd = open(c.path.c_str(), O_RDONLY);
            if (fd < 0) { perror(c.path.c_str()); return 2; }
            for (int e = 0; e < c.n_as; ++e) {
                const off_t off = (off_t) c.offset + (off_t) c.expert_ids[e] * (off_t) exp_bytes;
                size_t got = 0;
                while (got < exp_bytes) {
                    ssize_t r = pread(fd, raw.data() + e * exp_bytes + got, exp_bytes - got, off + got);
                    if (r <= 0) { perror("pread"); close(fd); return 2; }
                    got += r;
                }
            }
            close(fd);
        } else {
            std::vector<float> wsrc((size_t) c.n_as * c.m * c.k);
            gaussians(c.seed, wsrc.data(), wsrc.size());
            const size_t wrote = ggml_quantize_chunk(t, wsrc.data(), raw.data(), 0, (int64_t) c.n_as * c.m, c.k, nullptr);
            if (wrote != raw.size()) { fprintf(stderr, "%s: quantize wrote %zu of %zu bytes\n", c.name.c_str(), wrote, raw.size()); return 2; }
        }
        const uint64_t raw_hash = fnv1a(raw.data(), raw.size());

        // ---- independent decode (exact in f32 for these types: a power-of-two or f16 scale times a
        //      small integer), cross-checked against ggml's to_float ----
        std::vector<float> wd((size_t) c.n_as * c.m * c.k);
        long long decode_inexact = 0;
        for (size_t r = 0; r < (size_t) c.n_as * c.m; ++r)
            for (int64_t b = 0; b < nblk; ++b) {
                double tmp[32];
                decode_block(t, raw.data() + r * row_bytes + b * c.ti.block_bytes, tmp);
                float * dst = wd.data() + r * c.k + b * 32;
                for (int j = 0; j < 32; ++j) { dst[j] = (float) tmp[j]; if ((double) dst[j] != tmp[j]) decode_inexact++; }
            }
        double decode_maxdiff = 0;
        {
            const auto * tt = ggml_get_type_traits(t);
            std::vector<float> rowf(c.k);
            for (size_t r = 0; r < (size_t) c.n_as * c.m; r += std::max<size_t>(1, ((size_t) c.n_as * c.m) / 97)) {
                tt->to_float(raw.data() + r * row_bytes, rowf.data(), c.k);
                for (int64_t kk = 0; kk < c.k; ++kk) decode_maxdiff = std::max(decode_maxdiff, std::fabs((double) rowf[kk] - (double) wd[r * c.k + kk]));
            }
        }
        // the witness / fixture block: the first k-block of expert 0, row 0 with a non-zero weight
        int64_t wblk = 0;
        while (wblk + 1 < nblk) {
            const float * v = wd.data() + wblk * 32;
            bool nz = false;
            for (int j = 0; j < 32; ++j) nz = nz || v[j] != 0.0f;
            if (nz) break;
            ++wblk;
        }

        // ---- weights: CPU_REPACK copy (served path) and plain copy ----
        ggml_init_params wp = { 2 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * cw_rep = ggml_init(wp);
        ggml_context * cw_pl  = ggml_init(wp);
        ggml_tensor * a_rep = ggml_new_tensor_3d(cw_rep, t, c.k, c.m, c.n_as);
        ggml_tensor * a_pl  = ggml_new_tensor_3d(cw_pl,  t, c.k, c.m, c.n_as);
        ggml_backend_buffer_t bw_rep = ggml_backend_alloc_ctx_tensors_from_buft(cw_rep, buft_rep);
        ggml_backend_buffer_t bw_pl  = ggml_backend_alloc_ctx_tensors_from_buft(cw_pl,  buft_plain);
        ggml_backend_buffer_set_usage(bw_rep, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(bw_pl,  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // init_tensor has run: a null extra means the repack does not claim this tensor (type, shape or
        // GGML_IQK diversion), and set_tensor would dereference it. Report it unengaged and measure nothing.
        const bool extra = a_rep->extra != nullptr;
        bool layout_changed = false, engaged = false, restored = false;
        int witness_changed = 0, witness_unexpected = 0;
        if (extra) {
            ggml_backend_tensor_set(a_rep, raw.data(), 0, raw.size());
            layout_changed = memcmp(a_rep->data, raw.data(), std::min(raw.size(), ggml_nbytes(a_rep))) != 0;
        }
        ggml_backend_tensor_set(a_pl, raw.data(), 0, raw.size());

        if (extra) {
            // ---- engagement witness: one repacked scale moves exactly expert 0 / row 0 ----
            const int wn = 4, ne11 = c.bcast ? 1 : c.n_used;
            std::vector<float> wx; std::vector<int32_t> wids;
            make_inputs(c, wn, true, wx, wids);
            std::vector<float> y0, y1, y2;
            uint8_t * base = (uint8_t *) a_rep->data;
            uint8_t * wb = base + (size_t) wblk * x8_block_bytes(t);   // row group 0 = rows 0..7 of expert 0
            std::vector<uint8_t> saved(wb, wb + x8_block_bytes(t));
            bool ok = run_mmid(be, a_rep, c.k, ne11, wn, c.n_used, wx, wids, y0);
            perturb_lane(base, t, (size_t) wblk, 0, 2);
            ok = ok && run_mmid(be, a_rep, c.k, ne11, wn, c.n_used, wx, wids, y1);
            memcpy(wb, saved.data(), saved.size());
            ok = ok && run_mmid(be, a_rep, c.k, ne11, wn, c.n_used, wx, wids, y2);
            if (!ok) { fprintf(stderr, "%s: witness compute failed\n", c.name.c_str()); return 4; }
            for (int tk = 0; tk < wn; ++tk)
                for (int u = 0; u < c.n_used; ++u)
                    for (int64_t r = 0; r < c.m; ++r) {
                        const size_t i = ((size_t) tk * c.n_used + u) * c.m + r;
                        if (y1[i] != y0[i]) {
                            const bool expected = r == 0 && wids[(size_t) tk * c.n_used + u] == 0;
                            if (expected) witness_changed++; else witness_unexpected++;
                        }
                    }
            restored = memcmp(y0.data(), y2.data(), y0.size() * sizeof(float)) == 0;
            engaged = layout_changed && witness_changed > 0 && witness_unexpected == 0 && restored;
#if defined(AK_REPACK_FIXTURE)
#if AK_REPACK_FIXTURE == 1
            for (size_t bi = 0; bi < ggml_nbytes(a_rep) / x8_block_bytes(t); ++bi) perturb_lane(base, t, bi, 0, 1);
#else
            perturb_lane(base, t, (size_t) wblk, 0, 2);
#endif
#endif
        }
        fprintf(fo, "{\"kind\":\"case\",");
        json_str(fo, "case", c.name); fprintf(fo, ",");
        json_str(fo, "type", c.type_name);
        fprintf(fo, ",\"k\":%lld,\"m\":%lld,\"n_as\":%d,\"n_used\":%d,\"bcast\":%d,\"repack_buft\":\"%s\","
                    "\"extra\":%s,\"layout_changed\":%s,\"witness_changed\":%d,\"witness_unexpected\":%d,"
                    "\"witness_restored\":%s,\"engaged\":%s,\"raw_fnv64\":\"%016llx\",\"decode_maxdiff\":%.6e,\"decode_inexact\":%lld,\"witness_block\":%lld}\n",
                (long long) c.k, (long long) c.m, c.n_as, c.n_used, c.bcast ? 1 : 0, ggml_backend_buffer_name(bw_rep),
                extra ? "true" : "false", layout_changed ? "true" : "false", witness_changed, witness_unexpected,
                restored ? "true" : "false", engaged ? "true" : "false", (unsigned long long) raw_hash, decode_maxdiff,
                decode_inexact, (long long) wblk);
        fflush(fo);

        if (extra) {
            const int ne11 = c.bcast ? 1 : c.n_used;
            for (int n : c.widths) {
                std::vector<float> x; std::vector<int32_t> ids;
                make_inputs(c, n, false, x, ids);
                const uint64_t input_hash = fnv1a(ids.data(), ids.size() * sizeof(int32_t), fnv1a(x.data(), x.size() * sizeof(float)));
                // reference in double over the independently decoded weights
                std::vector<double> ref((size_t) c.m * c.n_used * n);
                #pragma omp parallel for collapse(2) num_threads(nt) schedule(static)
                for (int tk = 0; tk < n; ++tk) {
                    for (int u = 0; u < c.n_used; ++u) {
                        const int e = ids[(size_t) tk * c.n_used + u];
                        const float * xr = x.data() + ((size_t) tk * ne11 + (c.bcast ? 0 : u)) * c.k;
                        for (int64_t r = 0; r < c.m; ++r) {
                            const float * wr = wd.data() + ((size_t) e * c.m + r) * c.k;
                            double acc[4] = {0, 0, 0, 0};
                            for (int64_t kk = 0; kk < c.k; kk += 4)
                                for (int l = 0; l < 4; ++l) acc[l] += (double) wr[kk + l] * (double) xr[kk + l];
                            ref[((size_t) tk * c.n_used + u) * c.m + r] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                        }
                    }
                }
                double den = 0;
                for (double v : ref) den += v * v;
                const double rms = std::sqrt(den / (double) ref.size());
                double nmse[2], maxrel[2];
                bool finite[2];
                std::vector<float> outs[2];
                for (int arm = 0; arm < 2; ++arm) {
                    if (!run_mmid(be, arm == 0 ? a_rep : a_pl, c.k, ne11, n, c.n_used, x, ids, outs[arm])) {
                        fprintf(stderr, "%s n=%d: compute failed\n", c.name.c_str(), n); return 4;
                    }
                    double num = 0, mx = 0;
                    bool fin = outs[arm].size() == ref.size();
                    for (size_t i = 0; fin && i < ref.size(); ++i) {
                        if (!std::isfinite(outs[arm][i])) { fin = false; break; }
                        const double d = (double) outs[arm][i] - ref[i];
                        num += d * d;
                        mx = std::max(mx, std::fabs(d));
                    }
                    finite[arm] = fin;
                    nmse[arm] = fin && den > 0 ? num / den : NAN;
                    maxrel[arm] = fin && rms > 0 ? mx / rms : NAN;
                }
                double num = 0, d2 = 0;
                for (size_t i = 0; finite[0] && finite[1] && i < ref.size(); ++i) {
                    const double d = (double) outs[0][i] - (double) outs[1][i];
                    num += d * d; d2 += (double) outs[1][i] * (double) outs[1][i];
                }
                fprintf(fo, "{\"kind\":\"row\",");
                json_str(fo, "case", c.name);
                fprintf(fo, ",\"n\":%d,\"input_fnv64\":\"%016llx\",\"finite_repack\":%s,\"finite_plain\":%s,"
                            "\"nmse_repack\":%.9e,\"maxrel_repack\":%.9e,\"nmse_plain\":%.9e,\"maxrel_plain\":%.9e,"
                            "\"nmse_repack_vs_plain\":%.9e,\"ref_rms\":%.9e}\n",
                        n, (unsigned long long) input_hash, finite[0] ? "true" : "false", finite[1] ? "true" : "false",
                        std::isfinite(nmse[0]) ? nmse[0] : -1.0, std::isfinite(maxrel[0]) ? maxrel[0] : -1.0,
                        std::isfinite(nmse[1]) ? nmse[1] : -1.0, std::isfinite(maxrel[1]) ? maxrel[1] : -1.0,
                        d2 > 0 ? num / d2 : -1.0, rms);
                fflush(fo);
                fprintf(stderr, "%s n=%d nmse_rep=%.3e maxrel_rep=%.3e nmse_plain=%.3e\n", c.name.c_str(), n, nmse[0], maxrel[0], nmse[1]);
            }
        }
        ggml_backend_buffer_free(bw_rep);
        ggml_backend_buffer_free(bw_pl);
        ggml_free(cw_rep);
        ggml_free(cw_pl);
    }
    fprintf(fo, "{\"kind\":\"end\",\"n_cases\":%zu}\n", cases.size());
    fclose(fo);
    ggml_backend_free(be);
    return 0;
}
