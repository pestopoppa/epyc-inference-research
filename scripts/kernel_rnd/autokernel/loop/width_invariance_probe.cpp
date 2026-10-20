// AKX-ALL-26 op-level width-invariance probe.
//
// Contract under test: ONE graph holds a batched op over k columns/queries/tokens AND k serial single-column
// ops on the SAME operands. Their output bytes are compared with memcmp (tolerance 0) and the whole graph is
// computed `reps` times (outputs poisoned with 0xFF before every run) so run-to-run nondeterminism (A/A) is
// separated from width variance. One JSONL record is written per (route, nth, op, type, shape, mode, k) cell.
//
// Nothing here is a numerical reference: the verdict is only "batched bytes == serial bytes" or not.
// Written backend-agnostically (tensor data moves through ggml_backend_tensor_set/get) so the HIP half can run the
// same source with --backend ROCm0; this CPU half only exercised --backend cpu. The `ctl` op is CPU-only.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// ------------------------------------------------------------------ small utilities
static uint64_t fnv1a(const void * p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL) { next(); next(); }
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uni() { return ((next() >> 11) + 0.5) * 0x1p-53; }
    double gauss() {
        double u1 = uni(), u2 = uni();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

static std::string hex64(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}

static std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o + "\"";
}

struct Cfg {
    std::string backend = "cpu";
    std::string route = "cpu";
    std::string op;
    std::string type = "-";
    std::string shape;         // optional filter
    std::string mode;          // optional ids-mode filter (mmid)
    int nth = 24;
    int reps = 3;
    int kmax = 9;
    uint64_t seed = 1;
    std::string out;
    FILE * f = nullptr;
    std::string nkv;           // optional comma list override for fa
} G;

static ggml_backend_t g_be = nullptr;

static void emit_line(const std::string & s) {
    std::fwrite(s.data(), 1, s.size(), G.f);
    std::fputc('\n', G.f);
    std::fflush(G.f);
}

// ------------------------------------------------------------------ comparison
struct Seg { const float * b; const float * s; int64_t n; };

struct RepOut {
    uint64_t bdig = 0, sdig = 0;
    bool equal = true;
    int64_t n_diff = 0, first_seg = -1, first_elem = -1;
    double max_abs = 0;
    uint32_t b_bits = 0, s_bits = 0;
    bool poison_left = false;
};

static RepOut compare_segs(const std::vector<Seg> & segs) {
    RepOut r;
    r.bdig = 1469598103934665603ULL;
    r.sdig = 1469598103934665603ULL;
    for (size_t j = 0; j < segs.size(); ++j) {
        const Seg & sg = segs[j];
        r.bdig = fnv1a(sg.b, sg.n * 4, r.bdig);
        r.sdig = fnv1a(sg.s, sg.n * 4, r.sdig);
        const uint32_t * bb = (const uint32_t *) sg.b;
        const uint32_t * ss = (const uint32_t *) sg.s;
        for (int64_t i = 0; i < sg.n; ++i) {
            if (bb[i] == 0xFFFFFFFFu || ss[i] == 0xFFFFFFFFu) r.poison_left = true;
            if (bb[i] != ss[i]) {
                if (r.first_seg < 0) { r.first_seg = (int64_t) j; r.first_elem = i; r.b_bits = bb[i]; r.s_bits = ss[i]; }
                r.n_diff++;
                double d = std::fabs((double) sg.b[i] - (double) sg.s[i]);
                if (std::isfinite(d) && d > r.max_abs) r.max_abs = d;
            }
        }
    }
    r.equal = (r.n_diff == 0);
    return r;
}

// One compute of the cell's graph with all output tensors poisoned first; returns the read-back buffers.
using Bufs = std::vector<std::vector<float>>;
using SegFn = std::function<std::vector<Seg>(const Bufs &)>;

struct Cell {
    ggml_cgraph * gf = nullptr;
    std::vector<ggml_tensor *> outs;
    SegFn segs;
    std::function<void(Bufs &)> tamper;   // positive-control hook: mutate the host copy of the batched output
};

static bool run_cell(Cell & c, int reps, std::vector<RepOut> & res, std::string & err) {
    for (int r = 0; r < reps; ++r) {
        for (ggml_tensor * t : c.outs) {
            std::vector<uint8_t> p(ggml_nbytes(t), 0xFF);
            ggml_backend_tensor_set(t, p.data(), 0, p.size());
        }
        if (ggml_backend_graph_compute(g_be, c.gf) != GGML_STATUS_SUCCESS) { err = "graph_compute failed"; return false; }
        Bufs bufs(c.outs.size());
        for (size_t i = 0; i < c.outs.size(); ++i) {
            bufs[i].resize(ggml_nbytes(c.outs[i]) / 4);
            ggml_backend_tensor_get(c.outs[i], bufs[i].data(), 0, ggml_nbytes(c.outs[i]));
        }
        if (c.tamper) c.tamper(bufs);
        res.push_back(compare_segs(c.segs(bufs)));
    }
    return true;
}

// ------------------------------------------------------------------ record emission
struct Meta {
    std::string op, type, shape, mode;
    std::string dims;      // preformatted extra json fields (no braces)
};

static void emit_cell(const Meta & m, int k, const std::vector<RepOut> & reps, const std::string & forced, const std::string & reason,
                      const std::string & extra = "") {
    std::string v;
    bool aa = true;
    std::string reps_json = "[";
    if (!forced.empty()) {
        v = forced;
    } else {
        bool any_poison = false, all_eq = true, any_eq = false;
        for (size_t i = 0; i < reps.size(); ++i) {
            if (reps[i].bdig != reps[0].bdig || reps[i].sdig != reps[0].sdig) aa = false;
            any_poison |= reps[i].poison_left;
            all_eq &= reps[i].equal;
            any_eq |= reps[i].equal;
            char b[640];
            std::snprintf(b, sizeof b,
                "%s{\"batched_digest\":\"%s\",\"serial_digest\":\"%s\",\"equal\":%s,\"n_diff\":%lld,\"first_diff\":%s,\"max_abs_diff\":%.9g}",
                i ? "," : "", hex64(reps[i].bdig).c_str(), hex64(reps[i].sdig).c_str(), reps[i].equal ? "true" : "false",
                (long long) reps[i].n_diff,
                reps[i].first_seg < 0 ? "null" :
                    (std::string("{\"seg\":") + std::to_string(reps[i].first_seg) + ",\"elem\":" + std::to_string(reps[i].first_elem) +
                     ",\"batched_bits\":\"" + hex64(reps[i].b_bits).substr(8) + "\",\"serial_bits\":\"" + hex64(reps[i].s_bits).substr(8) + "\"}").c_str(),
                reps[i].max_abs);
            reps_json += b;
        }
        if (any_poison) v = "ERR";
        else if (!aa || (any_eq && !all_eq)) v = "NONDET";
        else v = all_eq ? "PASS" : "FAIL";
        if (v == "ERR") { aa = false; }
    }
    reps_json += "]";
    std::string line = "{\"schema\":\"akx-all-26.cell.v1\",\"backend\":" + jstr(G.backend) + ",\"route\":" + jstr(G.route) +
        ",\"nth\":" + std::to_string(G.nth) + ",\"op\":" + jstr(m.op) + ",\"type\":" + jstr(m.type) + ",\"shape\":" + jstr(m.shape) +
        ",\"mode\":" + jstr(m.mode) + ",\"k\":" + std::to_string(k) + ",\"seed\":" + std::to_string(G.seed) + "," + m.dims +
        ",\"n_reps\":" + std::to_string(reps.size()) + ",\"aa_clean\":" + (forced.empty() ? (aa ? "true" : "false") : "null") +
        ",\"verdict\":" + jstr(v) + ",\"reason\":" + jstr(reason) + ",\"reps\":" + reps_json + (extra.empty() ? "" : "," + extra) + "}";
    emit_line(line);
}

static void emit_unsupported(const Meta & m, const std::string & verdict, const std::string & why) {
    for (int k = 1; k <= G.kmax; ++k) emit_cell(m, k, {}, verdict, why);
}

// ------------------------------------------------------------------ data helpers
static const float * g_imat_ones = nullptr;

static void parallel_for(int n_items, int n_workers, const std::function<void(int, int)> & fn) {
    n_workers = std::max(1, std::min(n_workers, n_items));
    std::vector<std::thread> th;
    for (int w = 0; w < n_workers; ++w) {
        int lo = (int) ((int64_t) n_items * w / n_workers), hi = (int) ((int64_t) n_items * (w + 1) / n_workers);
        th.emplace_back([=] { fn(lo, hi); });
    }
    for (auto & t : th) t.join();
}

static std::vector<float> gauss_vec(size_t n, uint64_t seed, float scale) {
    std::vector<float> v(n);
    int nw = 16;
    parallel_for(nw, nw, [&](int lo, int hi) {
        for (int w = lo; w < hi; ++w) {
            size_t a = n * w / nw, b = n * (w + 1) / nw;
            Rng r(seed * 1000003ULL + w);
            for (size_t i = a; i < b; ++i) v[i] = (float) (r.gauss() * scale);
        }
    });
    return v;
}

// n_rows x ne00 floats -> quantized bytes of `type` (rows contiguous).
static std::vector<uint8_t> quantize_rows(ggml_type type, const std::vector<float> & src, int64_t n_rows, int64_t ne00) {
    size_t rs = ggml_row_size(type, ne00);
    std::vector<uint8_t> dst(rs * n_rows);
    if (type == GGML_TYPE_F32) { std::memcpy(dst.data(), src.data(), dst.size()); return dst; }
    if (type == GGML_TYPE_F16) { ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) dst.data(), n_rows * ne00); return dst; }
    ggml_quantize_init(type);
    std::vector<float> ones;
    const float * imat = nullptr;
    if (ggml_quantize_requires_imatrix(type)) { ones.assign(ne00, 1.0f); imat = ones.data(); }
    parallel_for((int) n_rows, std::max(1, G.nth), [&](int lo, int hi) {
        ggml_quantize_chunk(type, src.data(), dst.data(), (int64_t) lo * ne00, hi - lo, ne00, imat);
    });
    return dst;
}

static ggml_type type_from_name(const std::string & n) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * nm = ggml_type_name((ggml_type) i);
        if (!nm) continue;
        std::string a = nm, b = n;
        for (auto & c : a) c = (char) std::toupper((unsigned char) c);
        for (auto & c : b) c = (char) std::toupper((unsigned char) c);
        if (a == b) return (ggml_type) i;
    }
    return GGML_TYPE_COUNT;
}

static ggml_context * new_ctx(size_t n_tensors) {
    ggml_init_params p = { ggml_tensor_overhead() * (n_tensors + 16) + ggml_graph_overhead_custom(4096, false) + 65536, nullptr, true };
    return ggml_init(p);
}

static void set_f32(ggml_tensor * t, const float * p, size_t n) { ggml_backend_tensor_set(t, p, 0, n * 4); }

// Persistent weight-like tensor [ne00, ne01, n_exp] of `type`, rows tiled from a pool of distinct quantized rows.
struct Weights {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t = nullptr;
    void free_all() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); buf = nullptr; ctx = nullptr; }
};

static bool make_weights(Weights & w, ggml_type type, int64_t ne00, int64_t ne01, int64_t n_exp, uint64_t seed) {
    const int64_t POOL = 1024;
    w.ctx = new_ctx(4);
    w.t = n_exp > 1 ? ggml_new_tensor_3d(w.ctx, type, ne00, ne01, n_exp) : ggml_new_tensor_2d(w.ctx, type, ne00, ne01);
    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, g_be);
    if (!w.buf) return false;
    std::vector<float> src = gauss_vec((size_t) POOL * ne00, seed, 0.05f);
    std::vector<uint8_t> pool = quantize_rows(type, src, POOL, ne00);
    size_t rs = ggml_row_size(type, ne00);
    int64_t total_rows = ne01 * n_exp;
    // fill in slabs to bound the host staging buffer
    const int64_t SLAB = 4096;
    std::vector<uint8_t> stage((size_t) std::min(SLAB, total_rows) * rs);
    for (int64_t r0 = 0; r0 < total_rows; r0 += SLAB) {
        int64_t n = std::min(SLAB, total_rows - r0);
        for (int64_t i = 0; i < n; ++i) {
            int64_t r = r0 + i;
            int64_t e = r / ne01, row = r % ne01;
            uint64_t pick = ((uint64_t) row * 2654435761ULL + (uint64_t) e * 40503ULL + (uint64_t) e * (uint64_t) row * 7ULL) % POOL;
            std::memcpy(stage.data() + (size_t) i * rs, pool.data() + pick * rs, rs);
        }
        ggml_backend_tensor_set(w.t, stage.data(), (size_t) r0 * rs, (size_t) n * rs);
    }
    return true;
}

// ------------------------------------------------------------------ MUL_MAT
struct MMShape { const char * tag; int64_t ne00, ne01; bool small; };
static const MMShape MM_SHAPES[] = {
    {"D27-5120", 5120, 17408, false}, {"D27-17408", 17408, 5120, false}, {"D27-6144", 6144, 5120, false},
    {"M2048", 2048, 512, true}, {"M512", 512, 2048, true}, {"F2560", 2560, 640, false}, {"F640", 640, 2560, false},
};

static void run_mm(ggml_type type) {
    const std::string tn = ggml_type_name(type);
    for (const MMShape & sh : MM_SHAPES) {
        if (!G.shape.empty() && G.shape != sh.tag) continue;
        if (G.nth == 1 && !sh.small) continue;
        Meta m{"MUL_MAT", tn, sh.tag, "-", "\"ne00\":" + std::to_string(sh.ne00) + ",\"ne01\":" + std::to_string(sh.ne01)};
        if (sh.ne00 % ggml_blck_size(type) != 0) {
            emit_unsupported(m, "UNSUPPORTED", "ne00 not a multiple of block size " + std::to_string(ggml_blck_size(type)));
            continue;
        }
        Weights w;
        if (!make_weights(w, type, sh.ne00, sh.ne01, 1, G.seed * 7 + 1)) { emit_unsupported(m, "ERR", "weight alloc failed"); continue; }
        std::vector<float> X9 = gauss_vec((size_t) sh.ne00 * 9, G.seed * 13 + 5, 1.0f);
        for (int k = 1; k <= G.kmax; ++k) {
            ggml_context * ctx = new_ctx(4 * k + 8);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.ne00, k);
            ggml_tensor * Y = ggml_mul_mat(ctx, w.t, X);
            std::vector<ggml_tensor *> Xj(k), Yj(k);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
            ggml_build_forward_expand(gf, Y);
            for (int j = 0; j < k; ++j) {
                Xj[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.ne00, 1);
                Yj[j] = ggml_mul_mat(ctx, w.t, Xj[j]);
                ggml_build_forward_expand(gf, Yj[j]);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_be);
            if (!buf) { emit_cell(m, k, {}, "ERR", "activation alloc failed"); ggml_free(ctx); continue; }
            set_f32(X, X9.data(), (size_t) sh.ne00 * k);
            for (int j = 0; j < k; ++j) set_f32(Xj[j], X9.data() + (size_t) sh.ne00 * j, sh.ne00);
            Cell c;
            c.gf = gf;
            c.outs.push_back(Y);
            for (int j = 0; j < k; ++j) c.outs.push_back(Yj[j]);
            int64_t rows = sh.ne01;
            c.segs = [=](const Bufs & b) {
                std::vector<Seg> s;
                for (int j = 0; j < k; ++j) s.push_back({b[0].data() + (size_t) rows * j, b[1 + j].data(), rows});
                return s;
            };
            std::vector<RepOut> reps;
            std::string err;
            if (!run_cell(c, G.reps, reps, err)) emit_cell(m, k, {}, "ERR", err);
            else emit_cell(m, k, reps, "", "");
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        w.free_all();
    }
}

// ------------------------------------------------------------------ MUL_MAT_ID
struct MMIDShape { const char * tag; int64_t ne00, ne01, n_exp, top; bool bcast; bool small; };
static const MMIDShape MMID_SHAPES[] = {
    {"M2048", 2048, 512, 256, 8, true, true}, {"M512", 512, 2048, 256, 8, false, true},
    {"F2560", 2560, 640, 512, 10, true, false}, {"F640", 640, 2560, 512, 10, false, false},
};

static std::vector<int32_t> make_ids(const std::string & mode, int64_t n_exp, int top, int n_tok, uint64_t seed) {
    std::vector<int32_t> ids((size_t) top * n_tok);
    Rng r(seed * 99991ULL + 17);
    auto pick = [&](std::vector<int32_t> & out, int cnt) {
        std::vector<int32_t> pool(n_exp);
        for (int64_t i = 0; i < n_exp; ++i) pool[i] = (int32_t) i;
        for (int i = 0; i < cnt; ++i) {
            int64_t j = i + (int64_t) (r.next() % (uint64_t) (n_exp - i));
            std::swap(pool[i], pool[j]);
        }
        out.assign(pool.begin(), pool.begin() + cnt);
    };
    if (mode == "shared") {
        std::vector<int32_t> s;
        pick(s, top);
        for (int t = 0; t < n_tok; ++t) for (int i = 0; i < top; ++i) ids[(size_t) t * top + i] = s[i];
    } else if (mode == "disjoint") {
        std::vector<int32_t> s;
        pick(s, top * n_tok);
        for (int t = 0; t < n_tok; ++t) for (int i = 0; i < top; ++i) ids[(size_t) t * top + i] = s[(size_t) t * top + i];
    } else {
        for (int t = 0; t < n_tok; ++t) {
            std::vector<int32_t> s;
            pick(s, top);
            for (int i = 0; i < top; ++i) ids[(size_t) t * top + i] = s[i];
        }
    }
    return ids;
}

// smallest k such that some expert is routed from >= 2 (token, slot) pairs among the first k tokens; 0 = never
static int collision_first_k(const std::vector<int32_t> & ids, int top, int n_tok, int64_t n_exp) {
    std::vector<int> cnt(n_exp, 0);
    for (int t = 0; t < n_tok; ++t) {
        for (int i = 0; i < top; ++i) if (++cnt[ids[(size_t) t * top + i]] >= 2) return t + 1;
    }
    return 0;
}

static void run_mmid(ggml_type type) {
    const std::string tn = ggml_type_name(type);
    const char * modes[] = {"rand", "shared", "disjoint"};
    for (const MMIDShape & sh : MMID_SHAPES) {
        if (!G.shape.empty() && G.shape != sh.tag) continue;
        if (G.nth == 1 && !sh.small) continue;
        Meta m0{"MUL_MAT_ID", tn, sh.tag, "-", "\"ne00\":" + std::to_string(sh.ne00) + ",\"ne01\":" + std::to_string(sh.ne01) +
            ",\"n_expert\":" + std::to_string(sh.n_exp) + ",\"n_used\":" + std::to_string(sh.top) + ",\"bcast_src1\":" + (sh.bcast ? "true" : "false")};
        if (sh.ne00 % ggml_blck_size(type) != 0) {
            for (const char * md : modes) { Meta m = m0; m.mode = md; emit_unsupported(m, "UNSUPPORTED", "ne00 not a multiple of block size " + std::to_string(ggml_blck_size(type))); }
            continue;
        }
        Weights w;
        if (!make_weights(w, type, sh.ne00, sh.ne01, sh.n_exp, G.seed * 11 + 3)) {
            for (const char * md : modes) { Meta m = m0; m.mode = md; emit_unsupported(m, "ERR", "weight alloc failed"); }
            continue;
        }
        int rows_per_tok = sh.bcast ? 1 : sh.top;
        std::vector<float> B9 = gauss_vec((size_t) sh.ne00 * rows_per_tok * 9, G.seed * 17 + 9, 1.0f);
        for (const char * md : modes) {
            if (!G.mode.empty() && G.mode != md) continue;
            Meta m = m0;
            m.mode = md;
            std::vector<int32_t> ids9 = make_ids(md, sh.n_exp, sh.top, 9, G.seed);
            for (int k = 1; k <= G.kmax; ++k) {
                ggml_context * ctx = new_ctx(6 * k + 8);
                ggml_tensor * B = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.ne00, rows_per_tok, k);
                ggml_tensor * I = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, sh.top, k);
                ggml_tensor * Y = ggml_mul_mat_id(ctx, w.t, B, I);
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
                ggml_build_forward_expand(gf, Y);
                std::vector<ggml_tensor *> Bj(k), Ij(k), Yj(k);
                for (int j = 0; j < k; ++j) {
                    Bj[j] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.ne00, rows_per_tok, 1);
                    Ij[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, sh.top, 1);
                    Yj[j] = ggml_mul_mat_id(ctx, w.t, Bj[j], Ij[j]);
                    ggml_build_forward_expand(gf, Yj[j]);
                }
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_be);
                if (!buf) { emit_cell(m, k, {}, "ERR", "activation alloc failed"); ggml_free(ctx); continue; }
                size_t bt = (size_t) sh.ne00 * rows_per_tok;
                set_f32(B, B9.data(), bt * k);
                ggml_backend_tensor_set(I, ids9.data(), 0, (size_t) sh.top * k * 4);
                for (int j = 0; j < k; ++j) {
                    set_f32(Bj[j], B9.data() + bt * j, bt);
                    ggml_backend_tensor_set(Ij[j], ids9.data() + (size_t) sh.top * j, 0, (size_t) sh.top * 4);
                }
                Cell c;
                c.gf = gf;
                c.outs.push_back(Y);
                for (int j = 0; j < k; ++j) c.outs.push_back(Yj[j]);
                int64_t blk = sh.ne01 * sh.top;
                c.segs = [=](const Bufs & b) {
                    std::vector<Seg> s;
                    for (int j = 0; j < k; ++j) s.push_back({b[0].data() + (size_t) blk * j, b[1 + j].data(), blk});
                    return s;
                };
                std::vector<RepOut> reps;
                std::string err;
                std::string extra = "\"collision_first_k\":" + std::to_string(collision_first_k(ids9, sh.top, 9, sh.n_exp)) +
                                    ",\"ids_digest\":\"" + hex64(fnv1a(ids9.data(), (size_t) sh.top * k * 4)) + "\"";
                if (!run_cell(c, G.reps, reps, err)) emit_cell(m, k, {}, "ERR", err, extra);
                else emit_cell(m, k, reps, "", "", extra);
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
        }
        w.free_all();
    }
}

// ------------------------------------------------------------------ FLASH_ATTN_EXT
static void run_fa(ggml_type kvt) {
    const std::string tn = ggml_type_name(kvt);
    const int D = 256, HQ = 24, HKV = 4, QPAD = 64, KM = G.kmax;
    std::vector<int64_t> nkvs = {2048, 7168, 16384, 65536};
    if (!G.nkv.empty()) {
        nkvs.clear();
        size_t p = 0;
        while (p < G.nkv.size()) {
            size_t q = G.nkv.find(',', p);
            if (q == std::string::npos) q = G.nkv.size();
            nkvs.push_back(std::atoll(G.nkv.substr(p, q - p).c_str()));
            p = q + 1;
        }
    }
    std::vector<float> Q9 = gauss_vec((size_t) D * HQ * KM, G.seed * 31 + 1, 1.0f);   // [d, h, j]
    for (int64_t nkv : nkvs) {
        std::string tag = std::to_string(nkv);
        if (!G.shape.empty() && G.shape != tag) continue;
        Meta m{"FLASH_ATTN_EXT", tn, tag, "-", "\"head_dim\":256,\"n_head\":24,\"n_head_kv\":4,\"n_kv\":" + tag};
        // KV: F32 source -> type; same K and V type
        ggml_context * kctx = new_ctx(4);
        ggml_tensor * K = ggml_new_tensor_4d(kctx, kvt, D, nkv, HKV, 1);
        ggml_tensor * V = ggml_new_tensor_4d(kctx, kvt, D, nkv, HKV, 1);
        ggml_backend_buffer_t kbuf = ggml_backend_alloc_ctx_tensors(kctx, g_be);
        if (!kbuf) { emit_unsupported(m, "ERR", "kv alloc failed"); ggml_free(kctx); continue; }
        for (int which = 0; which < 2; ++which) {
            std::vector<float> src = gauss_vec((size_t) D * nkv * HKV, G.seed * 41 + which, 1.0f);
            std::vector<uint8_t> q = quantize_rows(kvt, src, nkv * HKV, D);
            ggml_backend_tensor_set(which ? V : K, q.data(), 0, q.size());
        }
        for (int k = 1; k <= KM; ++k) {
            ggml_context * ctx = new_ctx(6 * k + 8);
            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, k, HQ, 1);
            ggml_tensor * mk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, QPAD, 1, 1);
            ggml_tensor * Y = ggml_flash_attn_ext(ctx, q, K, V, mk, 1.0f / 16.0f, 0.0f, 0.0f);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
            ggml_build_forward_expand(gf, Y);
            std::vector<ggml_tensor *> qj(k), mj(k), Yj(k);
            for (int j = 0; j < k; ++j) {
                qj[j] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, HQ, 1);
                mj[j] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, QPAD, 1, 1);
                Yj[j] = ggml_flash_attn_ext(ctx, qj[j], K, V, mj[j], 1.0f / 16.0f, 0.0f, 0.0f);
                ggml_build_forward_expand(gf, Yj[j]);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_be);
            if (!buf) { emit_cell(m, k, {}, "ERR", "activation alloc failed"); ggml_free(ctx); continue; }
            // query j sees L_j = n_kv - (KM-1) + j positions (independent of k): width never changes an operand
            auto fill_mask = [&](ggml_tensor * t, const std::vector<int>& lens) {
                std::vector<uint16_t> mv((size_t) nkv * QPAD, 0xFC00);
                for (size_t r = 0; r < lens.size(); ++r)
                    for (int64_t p = 0; p < lens[r]; ++p) mv[r * nkv + p] = 0;
                ggml_backend_tensor_set(t, mv.data(), 0, mv.size() * 2);
            };
            std::vector<int> lens(k);
            for (int j = 0; j < k; ++j) lens[j] = (int) (nkv - (KM - 1) + j);
            fill_mask(mk, lens);
            std::vector<float> qb((size_t) D * k * HQ);
            for (int h = 0; h < HQ; ++h) for (int j = 0; j < k; ++j)
                std::memcpy(&qb[((size_t) h * k + j) * D], &Q9[((size_t) j * HQ + h) * D], D * 4);
            set_f32(q, qb.data(), qb.size());
            for (int j = 0; j < k; ++j) {
                fill_mask(mj[j], {lens[j]});
                std::vector<float> qs((size_t) D * HQ);
                for (int h = 0; h < HQ; ++h) std::memcpy(&qs[(size_t) h * D], &Q9[((size_t) j * HQ + h) * D], D * 4);
                set_f32(qj[j], qs.data(), qs.size());
            }
            Cell c;
            c.gf = gf;
            c.outs.push_back(Y);
            for (int j = 0; j < k; ++j) c.outs.push_back(Yj[j]);
            int64_t blk = (int64_t) D * HQ;
            c.segs = [=](const Bufs & b) {
                std::vector<Seg> s;
                for (int j = 0; j < k; ++j) s.push_back({b[0].data() + (size_t) blk * j, b[1 + j].data(), blk});
                return s;
            };
            std::vector<RepOut> reps;
            std::string err;
            if (!run_cell(c, G.reps, reps, err)) emit_cell(m, k, {}, "ERR", err);
            else emit_cell(m, k, reps, "", "");
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        ggml_backend_buffer_free(kbuf);
        ggml_free(kctx);
    }
}

// ------------------------------------------------------------------ GATED_DELTA_NET (state threaded serially)
static void unit_norm_heads(std::vector<float> & v, int S, int H, int T) {
    for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
        float * p = &v[((size_t) t * H + h) * S];
        double n = 0;
        for (int i = 0; i < S; ++i) n += (double) p[i] * p[i];
        float s = (float) (1.0 / std::sqrt(n + 1e-12));
        for (int i = 0; i < S; ++i) p[i] *= s;
    }
}

static void run_gdn() {
    struct Sh { const char * tag; int Sv, Hk, Hv, gdim; };      // gdim: 1 scalar gate, 0 -> S_v (KDA)
    const Sh shapes[] = {
        {"Q38-27B-Hv48-scalar", 128, 16, 48, 1}, {"A3B-Hv32-scalar", 128, 16, 32, 1}, {"Q38-27B-Hv48-kda", 128, 16, 48, 0},
    };
    for (const Sh & sh : shapes) {
        if (!G.shape.empty() && G.shape != sh.tag) continue;
        int gd = sh.gdim ? 1 : sh.Sv;
        Meta m{"GATED_DELTA_NET", "F32", sh.tag, "-", "\"S_v\":" + std::to_string(sh.Sv) + ",\"H_k\":" + std::to_string(sh.Hk) + ",\"H_v\":" + std::to_string(sh.Hv) + ",\"gate_dim\":" + std::to_string(gd)};
        const int KM = G.kmax;
        std::vector<float> q9 = gauss_vec((size_t) sh.Sv * sh.Hk * KM, G.seed * 51 + 1, 1.0f);
        std::vector<float> k9 = gauss_vec((size_t) sh.Sv * sh.Hk * KM, G.seed * 51 + 2, 1.0f);
        unit_norm_heads(q9, sh.Sv, sh.Hk, KM);
        unit_norm_heads(k9, sh.Sv, sh.Hk, KM);
        std::vector<float> v9 = gauss_vec((size_t) sh.Sv * sh.Hv * KM, G.seed * 51 + 3, 1.0f);
        std::vector<float> g9((size_t) gd * sh.Hv * KM), b9((size_t) sh.Hv * KM);
        { Rng r(G.seed * 51 + 4); for (auto & x : g9) x = (float) (-0.02 - 1.5 * r.uni()); }
        { Rng r(G.seed * 51 + 5); for (auto & x : b9) x = (float) (0.1 + 0.8 * r.uni()); }
        std::vector<float> s0 = gauss_vec((size_t) sh.Sv * sh.Sv * sh.Hv, G.seed * 51 + 6, 0.05f);
        for (int k = 1; k <= KM; ++k) {
            ggml_context * ctx = new_ctx(12 * k + 24);
            const int Sv = sh.Sv, Hk = sh.Hk, Hv = sh.Hv;
            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Sv, Hk, k, 1);
            ggml_tensor * kk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Sv, Hk, k, 1);
            ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Sv, Hv, k, 1);
            ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, gd, Hv, k, 1);
            ggml_tensor * be = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, k, 1);
            ggml_tensor * st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Sv, Sv, Hv, 1);
            ggml_tensor * Y = ggml_gated_delta_net(ctx, q, kk, v, g, be, st, 1);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
            ggml_build_forward_expand(gf, Y);
            std::vector<ggml_tensor *> qj(k), kj(k), vj(k), gj(k), bj(k), Yj(k);
            ggml_tensor * st_prev = nullptr;
            for (int j = 0; j < k; ++j) {
                qj[j] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Sv, Hk, 1, 1);
                kj[j] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Sv, Hk, 1, 1);
                vj[j] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Sv, Hv, 1, 1);
                gj[j] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, gd, Hv, 1, 1);
                bj[j] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, 1, 1);
                ggml_tensor * sin_;
                if (j == 0) {
                    sin_ = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Sv, Sv, Hv, 1);   // separate copy of s0
                    st_prev = sin_;
                } else {
                    sin_ = ggml_view_4d(ctx, Yj[j - 1], Sv, Sv, Hv, 1, (size_t) Sv * 4, (size_t) Sv * Sv * 4, (size_t) Sv * Sv * Hv * 4,
                                        (size_t) Sv * Hv * 1 * 4);   // new state of the previous serial step
                }
                Yj[j] = ggml_gated_delta_net(ctx, qj[j], kj[j], vj[j], gj[j], bj[j], sin_, 1);
                ggml_build_forward_expand(gf, Yj[j]);
            }
            ggml_tensor * s0_serial = st_prev;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_be);
            if (!buf) { emit_cell(m, k, {}, "ERR", "alloc failed"); ggml_free(ctx); continue; }
            set_f32(q, q9.data(), (size_t) Sv * Hk * k);
            set_f32(kk, k9.data(), (size_t) Sv * Hk * k);
            set_f32(v, v9.data(), (size_t) Sv * Hv * k);
            set_f32(g, g9.data(), (size_t) gd * Hv * k);
            set_f32(be, b9.data(), (size_t) Hv * k);
            set_f32(st, s0.data(), s0.size());
            set_f32(s0_serial, s0.data(), s0.size());
            for (int j = 0; j < k; ++j) {
                set_f32(qj[j], q9.data() + (size_t) Sv * Hk * j, (size_t) Sv * Hk);
                set_f32(kj[j], k9.data() + (size_t) Sv * Hk * j, (size_t) Sv * Hk);
                set_f32(vj[j], v9.data() + (size_t) Sv * Hv * j, (size_t) Sv * Hv);
                set_f32(gj[j], g9.data() + (size_t) gd * Hv * j, (size_t) gd * Hv);
                set_f32(bj[j], b9.data() + (size_t) Hv * j, (size_t) Hv);
            }
            Cell c;
            c.gf = gf;
            c.outs.push_back(Y);
            for (int j = 0; j < k; ++j) c.outs.push_back(Yj[j]);
            int64_t att = (int64_t) Sv * Hv, stn = (int64_t) Sv * Sv * Hv;
            c.segs = [=](const Bufs & b) {
                std::vector<Seg> s;
                for (int j = 0; j < k; ++j) s.push_back({b[0].data() + (size_t) att * j, b[1 + j].data(), att});          // token j output
                s.push_back({b[0].data() + (size_t) att * k, b[k].data() + att, stn});                                     // final state
                return s;
            };
            std::vector<RepOut> reps;
            std::string err;
            if (!run_cell(c, G.reps, reps, err)) emit_cell(m, k, {}, "ERR", err);
            else emit_cell(m, k, reps, "", "");
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
}

// ------------------------------------------------------------------ positive / negative / harness controls
struct CtlUD { int mode; };      // 0 = width-dependent partial sums (positive), 1 = fixed single chain (negative)

static void ctl_dot(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    const ggml_tensor * W = dst->src[0];
    const ggml_tensor * X = dst->src[1];
    const int64_t K = W->ne[0], rows = W->ne[1], ncol = X->ne[1];
    const int mode = ((CtlUD *) ud)->mode;
    const int P = mode == 0 ? (int) ncol : 1;           // number of interleaved partial accumulators = batch width
    std::vector<float> acc(P);
    for (int64_t c = 0; c < ncol; ++c) {
        const float * x = (const float *) ((const char *) X->data + c * X->nb[1]);
        for (int64_t r = 0; r < rows; ++r) {
            const float * w = (const float *) ((const char *) W->data + r * W->nb[1]);
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int64_t i = 0; i < K; ++i) acc[i % P] += w[i] * x[i];
            float s = 0.0f;
            for (int p = 0; p < P; ++p) s += acc[p];
            ((float *) ((char *) dst->data + c * dst->nb[1]))[r] = s;
        }
    }
}

static void run_ctl() {
    const int64_t K = 4096, rows = 64;
    struct C { const char * name; int mode; int flip; };
    const C cs[] = {
        {"ctl-positive-width-dependent", 0, 0},       // MUST FAIL for k >= 2
        {"ctl-negative-width-independent", 1, 0},     // MUST PASS for every k
        {"ctl-bitflip-injection", 1, 1},              // single bit flipped in host copy of batched col (k-1): must FAIL at every k >= 1 at the injected index
    };
    std::vector<float> W = gauss_vec((size_t) K * rows, G.seed * 61 + 1, 0.05f);
    std::vector<float> X9 = gauss_vec((size_t) K * 9, G.seed * 61 + 2, 1.0f);
    for (const C & cc : cs) {
        if (!G.shape.empty() && G.shape != cc.name) continue;
        Meta m{"CONTROL", "F32", cc.name, "-", "\"K\":4096,\"rows\":64"};
        CtlUD ud{cc.mode};
        for (int k = 1; k <= G.kmax; ++k) {
            ggml_context * ctx = new_ctx(4 * k + 8);
            ggml_tensor * Wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, rows);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, k);
            ggml_tensor * args[2] = {Wt, X};
            ggml_tensor * Y = ggml_custom_4d(ctx, GGML_TYPE_F32, rows, k, 1, 1, args, 2, ctl_dot, 1, &ud);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
            ggml_build_forward_expand(gf, Y);
            std::vector<ggml_tensor *> Xj(k), Yj(k);
            for (int j = 0; j < k; ++j) {
                Xj[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
                ggml_tensor * a2[2] = {Wt, Xj[j]};
                Yj[j] = ggml_custom_4d(ctx, GGML_TYPE_F32, rows, 1, 1, 1, a2, 2, ctl_dot, 1, &ud);
                ggml_build_forward_expand(gf, Yj[j]);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_be);
            set_f32(Wt, W.data(), W.size());
            set_f32(X, X9.data(), (size_t) K * k);
            for (int j = 0; j < k; ++j) set_f32(Xj[j], X9.data() + (size_t) K * j, K);
            Cell c;
            c.gf = gf;
            c.outs.push_back(Y);
            for (int j = 0; j < k; ++j) c.outs.push_back(Yj[j]);
            c.segs = [=](const Bufs & b) {
                std::vector<Seg> s;
                for (int j = 0; j < k; ++j) s.push_back({b[0].data() + (size_t) rows * j, b[1 + j].data(), rows});
                return s;
            };
            if (cc.flip) c.tamper = [=](Bufs & b) { uint32_t u; std::memcpy(&u, &b[0][(size_t) rows * (k - 1) + 5], 4); u ^= 1u; std::memcpy(&b[0][(size_t) rows * (k - 1) + 5], &u, 4); };
            std::vector<RepOut> reps;
            std::string err;
            std::string extra = cc.flip ? "\"injected\":{\"seg\":" + std::to_string(k - 1) + ",\"elem\":5}" : "";
            if (!run_cell(c, G.reps, reps, err)) emit_cell(m, k, {}, "ERR", err, extra);
            else emit_cell(m, k, reps, "", "", extra);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
}

// ------------------------------------------------------------------ main
static std::string env_or(const char * n) { const char * v = std::getenv(n); return v ? v : "<unset>"; }

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--backend") G.backend = nxt();
        else if (a == "--route") G.route = nxt();
        else if (a == "--op") G.op = nxt();
        else if (a == "--type") G.type = nxt();
        else if (a == "--shape") G.shape = nxt();
        else if (a == "--mode") G.mode = nxt();
        else if (a == "--nth") G.nth = std::atoi(nxt().c_str());
        else if (a == "--reps") G.reps = std::atoi(nxt().c_str());
        else if (a == "--kmax") G.kmax = std::atoi(nxt().c_str());
        else if (a == "--seed") G.seed = std::strtoull(nxt().c_str(), nullptr, 10);
        else if (a == "--out") G.out = nxt();
        else if (a == "--nkv") G.nkv = nxt();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (G.out.empty() || G.op.empty()) { std::fprintf(stderr, "usage: --op mm|mmid|fa|gdn|ctl --type T --out FILE [--route R --nth N --reps 3 --kmax 9 --seed S --shape S --mode M --nkv a,b]\n"); return 2; }
    G.f = std::fopen(G.out.c_str(), "a");
    if (!G.f) return 3;
    if (G.backend == "cpu") {
        g_be = ggml_backend_cpu_init();
        if (g_be) ggml_backend_cpu_set_n_threads(g_be, G.nth);
    } else {
        g_be = ggml_backend_init_by_name(G.backend.c_str(), nullptr);
    }
    if (!g_be) { std::fprintf(stderr, "backend init failed: %s\n", G.backend.c_str()); return 4; }
    {
        std::string h = "{\"schema\":\"akx-all-26.group.v1\",\"backend\":" + jstr(G.backend) + ",\"route\":" + jstr(G.route) + ",\"op\":" + jstr(G.op) +
            ",\"type\":" + jstr(G.type) + ",\"nth\":" + std::to_string(G.nth) + ",\"reps\":" + std::to_string(G.reps) + ",\"kmax\":" + std::to_string(G.kmax) +
            ",\"seed\":" + std::to_string(G.seed) + ",\"env\":{";
        const char * names[] = {"GGML_IQK", "GGML_IQK_Q8_0", "GGML_IQK_DEQUANT", "GGML_ROWEXACT_N", "GGML_FA_SPLIT_KV", "GGML_QSPLIT", "GGML_QSPLIT_MIN", "OMP_NUM_THREADS", "LD_LIBRARY_PATH"};
        for (size_t i = 0; i < sizeof names / sizeof *names; ++i) h += std::string(i ? "," : "") + jstr(names[i]) + ":" + jstr(env_or(names[i]));
        h += "},\"cpu\":{\"avx512\":" + std::to_string(ggml_cpu_has_avx512()) + ",\"avx512_vnni\":" + std::to_string(ggml_cpu_has_avx512_vnni()) +
             "},\"backend_name\":" + jstr(ggml_backend_name(g_be)) + "}";
        emit_line(h);
    }
    if (G.op == "gdn") run_gdn();
    else if (G.op == "ctl") run_ctl();
    else {
        ggml_type t = type_from_name(G.type);
        if (t == GGML_TYPE_COUNT) { std::fprintf(stderr, "bad type %s\n", G.type.c_str()); return 5; }
        if (G.op == "mm") run_mm(t);
        else if (G.op == "mmid") run_mmid(t);
        else if (G.op == "fa") run_fa(t);
        else { std::fprintf(stderr, "bad op %s\n", G.op.c_str()); return 6; }
    }
    ggml_backend_free(g_be);
    std::fclose(G.f);
    return 0;
}
