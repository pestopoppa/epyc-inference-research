// AKX-ALL-26 width-invariance op probe.
//
// For every predicted cell (row of the frozen predicted-fail list) and every width k = 1..kmax this
// program builds ONE ggml graph that holds
//   * the BATCHED op over k columns / tokens / queries, and
//   * k SERIAL single-column ops that read column/token i through views of the very same operand
//     tensors (GATED_DELTA_NET threads the recurrent state serially: state_out(i) -> state_in(i+1)),
// computes that graph on the requested backend (ggml_backend_graph_compute) `repeats` times, and
// compares batched column i with serial result i byte-for-byte (memcmp, tolerance 0).
//
// Public ggml API only (ggml.h, ggml-alloc.h, ggml-backend.h, ggml-cpu.h). Built OUT OF TREE:
//   g++ -O2 -std=c++17 -I<tree>/ggml/include width_probe.cpp -L<build>/bin
//       -lggml -lggml-base -lggml-cpu -lpthread -Wl,-rpath,<build>/bin -o width_probe
//
// Backend binding is explicit and never falls back:
//   --backend cpu    -> ggml_backend_cpu_init() (does NOT touch the global registry, so a HIP-linked
//                       binary would not initialise ROCm for a CPU run; still: use the CPU build).
//   --backend rocm0  -> ggml_backend_dev_by_name("ROCm0") from the statically-registered backends
//                       (GGML_BACKEND_DL=OFF builds); --backend-dl-dir <dir> additionally calls
//                       ggml_backend_load_all_from_path(<dir>) for GGML_BACKEND_DL=ON builds.
//   The bound device name/description is printed and recorded; a mismatch ABORTS (exit 4).
//
// Exit codes: 0 ok; 2 PC-1 ran and did NOT fail; 3 some repeat was unstable (A/A not clean);
//             4 usage / backend binding / IO error. ggml asserts abort the process after the abort
//             callback writes an ERR record for the current cell (the driver runs one id per process).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ------------------------------------------------------------------------------------------------
// minimal JSON (parse the predicted rows; write records)
// ------------------------------------------------------------------------------------------------
struct JVal {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<JVal> a;
    std::vector<std::pair<std::string, JVal>> o;
    const JVal * get(const std::string & k) const {
        for (auto & kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    std::string str(const std::string & k, const std::string & def = "") const {
        const JVal * v = get(k);
        return (v && v->t == STR) ? v->s : def;
    }
};

struct JParser {
    const char * p; const char * e; std::string err;
    explicit JParser(const std::string & s) : p(s.data()), e(s.data() + s.size()) {}
    void ws() { while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++; }
    bool parse(JVal & v) {
        ws(); if (p >= e) { err = "eof"; return false; }
        if (*p == '{') { v.t = JVal::OBJ; p++; ws(); if (p < e && *p == '}') { p++; return true; }
            for (;;) { ws(); JVal k; if (!pstr(k.s)) return false; ws(); if (p >= e || *p != ':') { err = "':'"; return false; } p++;
                JVal val; if (!parse(val)) return false; v.o.emplace_back(k.s, std::move(val)); ws();
                if (p < e && *p == ',') { p++; continue; } if (p < e && *p == '}') { p++; return true; } err = "obj"; return false; } }
        if (*p == '[') { v.t = JVal::ARR; p++; ws(); if (p < e && *p == ']') { p++; return true; }
            for (;;) { JVal val; if (!parse(val)) return false; v.a.push_back(std::move(val)); ws();
                if (p < e && *p == ',') { p++; continue; } if (p < e && *p == ']') { p++; return true; } err = "arr"; return false; } }
        if (*p == '"') { v.t = JVal::STR; return pstr(v.s); }
        if (!strncmp(p, "true", 4))  { v.t = JVal::BOOL; v.b = true;  p += 4; return true; }
        if (!strncmp(p, "false", 5)) { v.t = JVal::BOOL; v.b = false; p += 5; return true; }
        if (!strncmp(p, "null", 4))  { v.t = JVal::NUL; p += 4; return true; }
        char * end = nullptr; v.t = JVal::NUM; v.n = strtod(p, &end);
        if (end == p) { err = "value"; return false; } p = end; return true;
    }
    bool pstr(std::string & out) {
        if (p >= e || *p != '"') { err = "string"; return false; } p++;
        while (p < e && *p != '"') {
            if (*p == '\\') { p++; if (p >= e) return false;
                switch (*p) { case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break; case 'f': out += '\f'; break;
                    case 'u': { if (e - p < 5) return false; unsigned cp = strtoul(std::string(p + 1, p + 5).c_str(), nullptr, 16); p += 4;
                        if (cp < 0x80) out += (char) cp; else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
                        else { out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); } break; }
                    default: out += *p; }
                p++; continue; }
            out += *p++;
        }
        if (p >= e) { err = "unterminated string"; return false; } p++; return true;
    }
};

static std::string jesc(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) { case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break; case '\r': o += "\\r"; break;
            default: if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char) c; }
    }
    return o + "\"";
}
static std::string jnum(double v) {
    if (!std::isfinite(v)) return std::isnan(v) ? "\"nan\"" : (v > 0 ? "\"inf\"" : "\"-inf\"");
    char b[64]; snprintf(b, sizeof b, "%.9g", v); return b;
}
struct JW { // tiny object writer
    std::string s = "{"; bool first = true;
    JW & k(const std::string & key) { if (!first) s += ","; first = false; s += jesc(key) + ":"; return *this; }
    JW & str(const std::string & key, const std::string & v) { k(key); s += jesc(v); return *this; }
    JW & raw(const std::string & key, const std::string & v) { k(key); s += v; return *this; }
    JW & num(const std::string & key, double v) { k(key); s += jnum(v); return *this; }
    JW & i64(const std::string & key, long long v) { k(key); s += std::to_string(v); return *this; }
    JW & bl(const std::string & key, bool v) { k(key); s += v ? "true" : "false"; return *this; }
    std::string done() { return s + "}"; }
};
template <class T, class F> static std::string jarr(const std::vector<T> & v, F f) {
    std::string s = "["; for (size_t i = 0; i < v.size(); i++) { if (i) s += ","; s += f(v[i]); } return s + "]";
}

// ------------------------------------------------------------------------------------------------
// SHA-256 (digests of the compared byte ranges)
// ------------------------------------------------------------------------------------------------
struct Sha256 {
    uint32_t h[8]; uint64_t len = 0; uint8_t buf[64]; size_t bl = 0;
    static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    Sha256() { const uint32_t i[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}; memcpy(h, i, sizeof h); }
    void block(const uint8_t * p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
            0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
            0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
            0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i+1] << 16 | (uint32_t) p[4*i+2] << 8 | p[4*i+3];
        for (int i = 16; i < 64; i++) { uint32_t s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3); uint32_t s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10); w[i] = w[i-16] + s0 + w[i-7] + s1; }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) { uint32_t S1 = ror(e,6)^ror(e,11)^ror(e,25); uint32_t ch = (e&f)^(~e&g); uint32_t t1 = hh+S1+ch+K[i]+w[i];
            uint32_t S0 = ror(a,2)^ror(a,13)^ror(a,22); uint32_t mj = (a&b)^(a&c)^(b&c); uint32_t t2 = S0+mj; hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2; }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const void * data, size_t n) { const uint8_t * p = (const uint8_t *) data; len += n;
        while (n) { size_t c = std::min(n, 64 - bl); memcpy(buf + bl, p, c); bl += c; p += c; n -= c; if (bl == 64) { block(buf); bl = 0; } } }
    std::string hex() { uint64_t bits = len * 8; uint8_t pad = 0x80; update(&pad, 1); uint8_t z = 0; while (bl != 56) update(&z, 1);
        uint8_t L[8]; for (int i = 0; i < 8; i++) L[i] = (uint8_t) (bits >> (56 - 8*i)); update(L, 8);
        char o[65]; for (int i = 0; i < 8; i++) snprintf(o + 8*i, 9, "%08x", h[i]); return std::string(o, 64); }
};
static std::string sha256_hex(const std::vector<uint8_t> & v) { Sha256 s; s.update(v.data(), v.size()); return s.hex(); }

// ------------------------------------------------------------------------------------------------
// deterministic operands
// ------------------------------------------------------------------------------------------------
static uint64_t fnv1a(const std::string & s) { uint64_t h = 0xcbf29ce484222325ull; for (unsigned char c : s) { h ^= c; h *= 0x100000001b3ull; } return h; }
struct Rng {
    uint64_t x;
    explicit Rng(uint64_t seed) : x(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() { uint64_t z = (x += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    double uni() { return (next() >> 11) * 0x1.0p-53; }
    float normal() { double u1 = uni(), u2 = uni(); if (u1 < 1e-300) u1 = 1e-300; return (float) (std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::acos(-1.0) * u2)); }
};
// operand stream is keyed by (seed, cell key, operand name) so sharding / subsetting never changes operands
static Rng rng_for(uint64_t seed, const std::string & key, const std::string & name) { return Rng(seed ^ fnv1a(key + "|" + name)); }
static void fill_normal(std::vector<float> & v, Rng r, float scale) { for (auto & x : v) x = r.normal() * scale; }

static int g_quant_threads = 8;
static void quantize_rows(ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    if (type == GGML_TYPE_F32) { memcpy(dst, src, sizeof(float) * nrows * n_per_row); return; }
    ggml_quantize_init(type);
    int nt = std::max<int>(1, std::min<int64_t>(g_quant_threads, nrows));
    std::vector<std::thread> th;
    int64_t per = (nrows + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        int64_t r0 = t * per, r1 = std::min(nrows, r0 + per);
        if (r0 >= r1) break;
        th.emplace_back([=]() { ggml_quantize_chunk(type, src, dst, r0 * n_per_row, r1 - r0, n_per_row, imatrix); });
    }
    for (auto & x : th) x.join();
}

// ------------------------------------------------------------------------------------------------
// cell definitions
// ------------------------------------------------------------------------------------------------
enum Kind { K_MM, K_MMID, K_FA, K_GDN, K_PC1, K_NOTPROBED };

struct Sub {
    Kind kind = K_MM;
    std::string label;          // sub-shape label (e.g. "D27:ne00=5120,ne01=17408")
    ggml_type type = GGML_TYPE_F32;     // weight type (MM/MMID), K type (FA), state type (GDN)
    ggml_type vtype = GGML_TYPE_F32;    // FA V type
    int64_t K = 0, M = 0, n_expert = 0, n_used = 0, x_ne1 = 1;      // MM / MMID
    int64_t D = 0, H = 0, Hkv = 0, nkv = 0;                          // FA
    int64_t S = 0, Hv = 0, Hk = 0;                                   // GDN
    std::string unsupported_reason;     // pre-construction UNSUPPORTED (e.g. ne00 % block size)
    std::string note;
};

struct Row {
    std::string id, backend, op, type, shape;
    std::vector<Sub> subs;
    std::string resolve_error;
};

static ggml_type type_by_name(const std::string & n) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * nm = ggml_type_name((ggml_type) t);
        if (nm && strcasecmp(nm, n.c_str()) == 0) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

struct ShapeMM { const char * tag; int64_t K, M; };
// real served shapes (predicted list assumption 4): 27B dense qwen35 hidden 5120, ffn 17408, attn/ssm out 6144;
// qwen35moe hidden 2048, expert ff 512, 256 exp top-8; qwen4exp/Q38FN hidden 2560, expert ff 640, 512 exp top-10.
static const ShapeMM DENSE_D27[3] = { {"D27", 5120, 17408}, {"D27", 6144, 5120}, {"D27", 17408, 5120} };
struct ShapeMoE { const char * tag; int64_t K, M, n_expert, n_used, x_ne1; };
static const ShapeMoE MOE_SHAPES[4] = {
    {"M2048", 2048, 512, 256, 8, 1},     // qwen35moe gate/up experts: X [2048, 1, T] broadcast over the 8 used slots
    {"M512", 512, 2048, 256, 8, 8},      // qwen35moe down experts:    X [512, 8, T]
    {"F2560", 2560, 640, 512, 10, 1},    // qwen4exp gate/up experts:  X [2560, 1, T]
    {"F640", 640, 2560, 512, 10, 10},    // qwen4exp down experts:     X [640, 10, T]
};

static void check_block(Sub & s, ggml_type t, int64_t row) {
    if (t == GGML_TYPE_COUNT) { s.unsupported_reason = "unknown ggml type"; return; }
    int64_t bs = ggml_blck_size(t);
    if (row % bs != 0) {
        char b[256]; snprintf(b, sizeof b, "row length ne00=%" PRId64 " is not a multiple of the %s block size %" PRId64 " (ggml_quantize_chunk / row_size cannot represent it)",
                              row, ggml_type_name(t), bs);
        s.unsupported_reason = b;
    }
}

static Sub mm_sub(ggml_type t, const char * tag, int64_t K, int64_t M) {
    Sub s; s.kind = K_MM; s.type = t; s.K = K; s.M = M;
    char b[128]; snprintf(b, sizeof b, "%s:W[ne00=%" PRId64 ",ne01=%" PRId64 "]", tag, K, M); s.label = b;
    check_block(s, t, K); return s;
}
static Sub mmid_sub(ggml_type t, const ShapeMoE & m) {
    Sub s; s.kind = K_MMID; s.type = t; s.K = m.K; s.M = m.M; s.n_expert = m.n_expert; s.n_used = m.n_used; s.x_ne1 = m.x_ne1;
    char b[160]; snprintf(b, sizeof b, "%s:as[%" PRId64 ",%" PRId64 ",%" PRId64 "],top-%" PRId64 ",X_ne1=%" PRId64, m.tag, m.K, m.M, m.n_expert, m.n_used, m.x_ne1);
    s.label = b; check_block(s, t, m.K); return s;
}

static std::string suffix_after_last_dash(const std::string & id) { auto p = id.rfind('-'); return p == std::string::npos ? "" : id.substr(p + 1); }

static void resolve_row(Row & r) {
    const std::string sfx = suffix_after_last_dash(r.id);
    if (r.backend == "graph") {
        Sub s; s.kind = K_NOTPROBED; s.label = "graph-level";
        s.note = "graph-builder gate (llama build_delta_net fused/unfused selection); not expressible as a single ggml op probe";
        r.subs.push_back(s); return;
    }
    if (r.op == "MUL_MAT") {
        ggml_type t = type_by_name(r.type);
        if (sfx == "ALL") {
            for (auto & d : DENSE_D27) r.subs.push_back(mm_sub(t, d.tag, d.K, d.M));
            for (auto & m : MOE_SHAPES) r.subs.push_back(mm_sub(t, m.tag, m.K, m.M));
        } else if (sfx == "D27") {
            for (auto & d : DENSE_D27) r.subs.push_back(mm_sub(t, d.tag, d.K, d.M));
        } else {
            bool ok = false;
            for (auto & m : MOE_SHAPES) if (sfx == m.tag) { r.subs.push_back(mm_sub(t, m.tag, m.K, m.M)); ok = true; }
            if (!ok) r.resolve_error = "cannot resolve MUL_MAT shape suffix '" + sfx + "'";
        }
        return;
    }
    if (r.op == "MUL_MAT_ID") {
        ggml_type t = type_by_name(r.type);
        if (sfx == "ALL") { for (auto & m : MOE_SHAPES) r.subs.push_back(mmid_sub(t, m)); }
        else {
            bool ok = false;
            for (auto & m : MOE_SHAPES) if (sfx == m.tag) { r.subs.push_back(mmid_sub(t, m)); ok = true; }
            if (!ok) r.resolve_error = "cannot resolve MUL_MAT_ID shape suffix '" + sfx + "'";
        }
        return;
    }
    if (r.op == "FLASH_ATTN_EXT") {
        Sub s; s.kind = K_FA; s.D = 256; s.H = 24; s.Hkv = 4;
        s.nkv = atoll(sfx.c_str());
        if (s.nkv <= 0) { r.resolve_error = "cannot parse n_kv from id suffix '" + sfx + "'"; return; }
        if (r.type == "F16K/Q8_0V") { s.type = GGML_TYPE_F16; s.vtype = GGML_TYPE_Q8_0; }
        else { s.type = type_by_name(r.type); s.vtype = s.type; }
        char b[160]; snprintf(b, sizeof b, "27B-attn:D=%" PRId64 ",H=%" PRId64 ",H_kv=%" PRId64 ",n_kv=%" PRId64 ",K=%s,V=%s", s.D, s.H, s.Hkv, s.nkv,
                              s.type == GGML_TYPE_COUNT ? "?" : ggml_type_name(s.type), s.vtype == GGML_TYPE_COUNT ? "?" : ggml_type_name(s.vtype));
        s.label = b; check_block(s, s.type, s.D); if (s.unsupported_reason.empty()) check_block(s, s.vtype, s.D);
        r.subs.push_back(s); return;
    }
    if (r.op == "GATED_DELTA_NET") {
        ggml_type st = (r.type.find("BF16") != std::string::npos) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        std::vector<int64_t> hvs = {48};
        // CPU-GDN-F32 and GPU-GDN-F32state name both H_v=48 (27B/Q38FN) and H_v=32 (qwen35moe)
        if (st == GGML_TYPE_F32 && r.shape.find("32") != std::string::npos) hvs.push_back(32);
        for (int64_t hv : hvs) {
            Sub s; s.kind = K_GDN; s.type = st; s.S = 128; s.Hv = hv; s.Hk = 16;
            char b[128]; snprintf(b, sizeof b, "S_v=128,H_v=%" PRId64 ",H_k=16,state=%s,g=scalar", hv, ggml_type_name(st)); s.label = b;
            r.subs.push_back(s);
        }
        return;
    }
    r.resolve_error = "unknown op '" + r.op + "'";
}

static Row pc1_row(const std::string & backend) {
    Row r; r.id = "PC-1"; r.backend = backend; r.op = "MUL_MAT(PC-1 split-K positive control)"; r.type = "F32";
    r.shape = "W F32 [4096, 512]; batched mul_mat(W, X) vs serial add(mul_mat(W[:, :K/2], X_i[:K/2]), mul_mat(W[:, K/2:], X_i[K/2:]))";
    Sub s; s.kind = K_PC1; s.type = GGML_TYPE_F32; s.K = 4096; s.M = 512; s.label = "PC-1:W[4096,512]F32 split-K"; r.subs.push_back(s);
    return r;
}

// ------------------------------------------------------------------------------------------------
// globals / abort hook
// ------------------------------------------------------------------------------------------------
static FILE * g_out = nullptr;
static std::string g_cur_id, g_cur_sub, g_cur_backend;
static int g_cur_k = 0;
static void emit(const std::string & line) { if (g_out) { fputs(line.c_str(), g_out); fputc('\n', g_out); fflush(g_out); } }
static void on_ggml_abort(const char * msg) {
    JW w; w.str("record", "cell").str("id", g_cur_id).str("sub_label", g_cur_sub).str("backend_row", g_cur_backend)
        .str("status", "ERR").i64("k_at_abort", g_cur_k).str("error", std::string("ggml abort: ") + (msg ? msg : "?"));
    emit(w.done());
    fprintf(stderr, "[width_probe] ggml abort in %s/%s k=%d: %s\n", g_cur_id.c_str(), g_cur_sub.c_str(), g_cur_k, msg ? msg : "?");
}

static const char * ENV_KEYS[] = {"GGML_IQK", "GGML_IQK_Q8_0", "GGML_ROWEXACT_N", "GGML_FA_SPLIT_KV", "GGML_QSPLIT", "GGML_QSPLIT_MIN",
    "GGML_MMID_SLAB", "GGML_TINY_SOLO", "GGML_CPU_DISABLE_FUSION", "GGML_CUDA_Q8_PREFETCH", "GGML_CUDA_FA_SEQ_ROWS", "GGML_CUDA_FA_MASK_SKIP",
    "GGML_CUDA_GDN_STATE_BF16", "GGML_CUDA_DISABLE_GRAPHS", "OMP_NUM_THREADS", "LD_LIBRARY_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"};
static std::string env_json() {
    JW w; for (auto k : ENV_KEYS) { const char * v = getenv(k); if (v) w.str(k, v); else w.raw(k, "null"); } return w.done();
}
static bool env_on(const char * k) { const char * v = getenv(k); return v && *v && atoi(v) != 0; }

// row backend string -> required process env (the driver sets it; we verify instead of trusting it)
static std::string env_mismatch(const std::string & row_backend, const std::string & run_backend) {
    if (row_backend == "rocm0") return run_backend == "rocm0" ? "" : "row backend rocm0 requires --backend rocm0";
    if (row_backend.rfind("cpu", 0) == 0 && run_backend != "cpu") return "row backend " + row_backend + " requires --backend cpu";
    if (row_backend == "cpu") {
        if (!env_on("GGML_IQK")) return "row backend 'cpu' (production env) requires GGML_IQK=1";
        if (env_on("GGML_IQK_Q8_0")) return "row backend 'cpu' requires GGML_IQK_Q8_0 unset";
        return "";
    }
    if (row_backend == "cpu-native") {
        if (env_on("GGML_IQK")) return "row backend 'cpu-native' requires GGML_IQK unset/0";
        return "";
    }
    if (row_backend == "cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]") {
        if (!env_on("GGML_IQK") || !env_on("GGML_IQK_Q8_0")) return "row backend requires GGML_IQK=1 and GGML_IQK_Q8_0=1";
        return "";
    }
    return "unknown row backend '" + row_backend + "'";
}

// ------------------------------------------------------------------------------------------------
// per-k graph
// ------------------------------------------------------------------------------------------------
struct Seg { ggml_tensor * t; size_t off, size; };

struct KGraph {
    ggml_context * ctx = nullptr;          // op/view tensors (no_alloc)
    ggml_context * in_ctx = nullptr;       // k-specific inputs (FA mask)
    ggml_backend_buffer_t in_buf = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<Seg> bseg, sseg;           // batched / serial byte segments (same layout)
    std::vector<int64_t> col_elems;        // elements per comparison column (in layout order)
    std::vector<std::string> col_kind;     // "col"/"token"/"query"/"final_state"
    std::vector<ggml_tensor *> outputs;    // poisoned before each compute
    std::vector<ggml_tensor *> main_ops;   // nodes checked with supports_op (batched first)
    size_t elem_size = 4; ggml_type out_type = GGML_TYPE_F32;
    ~KGraph() {
        if (galloc) ggml_gallocr_free(galloc);
        if (in_buf) ggml_backend_buffer_free(in_buf);
        if (in_ctx) ggml_free(in_ctx);
        if (ctx) ggml_free(ctx);
    }
};

struct Operands {
    ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> t;
    std::string info_json = "{}";
    ~Operands() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
};

static ggml_context * new_meta_ctx(size_t n_tensors) {
    ggml_init_params p = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(n_tensors, false) + 1024 * 1024, nullptr, true };
    return ggml_init(p);
}

static std::vector<float> imatrix_for(int64_t n, Rng r) { std::vector<float> m(n); for (auto & x : m) x = 0.25f + (float) r.uni(); return m; }

// returns "" on success, else error
static std::string build_operands(ggml_backend_t be, const Sub & s, const std::string & key, uint64_t seed, int kmax, Operands & op, int router_mode) {
    op.ctx = new_meta_ctx(32);
    auto mk = [&](const char * name, ggml_type t, int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1) {
        ggml_tensor * x = ggml_new_tensor_4d(op.ctx, t, a, b, c, d); ggml_set_name(x, name); op.t[name] = x; return x; };
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> uploads;
    auto put_f32 = [&](ggml_tensor * x, const std::vector<float> & v) { std::vector<uint8_t> b(v.size() * 4); memcpy(b.data(), v.data(), b.size()); uploads.emplace_back(x, std::move(b)); };
    std::vector<float> imat;
    // i-quants get a deterministic positive imatrix (IQ2_XXS/IQ2_XS/IQ1_* cannot be quantized without one); other types: none
    const bool use_imat_type = s.type == GGML_TYPE_IQ2_XXS || s.type == GGML_TYPE_IQ2_XS || s.type == GGML_TYPE_IQ3_XXS || s.type == GGML_TYPE_IQ1_S ||
                               s.type == GGML_TYPE_IQ4_NL || s.type == GGML_TYPE_IQ3_S || s.type == GGML_TYPE_IQ2_S || s.type == GGML_TYPE_IQ4_XS ||
                               s.type == GGML_TYPE_IQ1_M || (s.type < GGML_TYPE_COUNT && ggml_quantize_requires_imatrix(s.type));
    JW info; info.i64("seed", (long long) seed).str("operand_key", key);

    if (s.kind == K_MM || s.kind == K_PC1) {
        ggml_tensor * W = mk("W", s.type, s.K, s.M);
        ggml_tensor * X = mk("X", GGML_TYPE_F32, s.K, kmax);
        std::vector<float> wf(s.K * s.M); fill_normal(wf, rng_for(seed, key, "W"), 0.05f);
        std::vector<float> xf(s.K * kmax); fill_normal(xf, rng_for(seed, key, "X"), 1.0f);
        const float * im = nullptr;
        if (s.kind == K_MM && use_imat_type) { imat = imatrix_for(s.K, rng_for(seed, key, "imatrix")); im = imat.data(); }
        std::vector<uint8_t> wq(ggml_nbytes(W)); quantize_rows(s.type, wf.data(), wq.data(), s.M, s.K, im);
        uploads.emplace_back(W, std::move(wq)); put_f32(X, xf);
        info.bl("imatrix", im != nullptr);
        if (s.kind == K_PC1) { // contiguous copies of the two K-halves of W (same values as W)
            ggml_tensor * Wlo = mk("Wlo", GGML_TYPE_F32, s.K / 2, s.M); ggml_tensor * Whi = mk("Whi", GGML_TYPE_F32, s.K / 2, s.M);
            std::vector<float> lo(s.K / 2 * s.M), hi(s.K / 2 * s.M);
            for (int64_t m = 0; m < s.M; m++) { memcpy(&lo[m * s.K / 2], &wf[m * s.K], 4 * s.K / 2); memcpy(&hi[m * s.K / 2], &wf[m * s.K + s.K / 2], 4 * s.K / 2); }
            put_f32(Wlo, lo); put_f32(Whi, hi);
        }
    } else if (s.kind == K_MMID) {
        // router: ids [n_used, kmax]. random (default): n_used distinct experts per token, seeded.
        std::vector<int32_t> ids(s.n_used * kmax);
        Rng rr = rng_for(seed, key, "router");
        std::vector<int32_t> perm(s.n_expert);
        for (int t = 0; t < kmax; t++) {
            for (int64_t e = 0; e < s.n_expert; e++) perm[e] = (int32_t) e;
            if (router_mode == 1) { // disjoint: token t uses experts [t*n_used, (t+1)*n_used)
                for (int64_t u = 0; u < s.n_used; u++) ids[t * s.n_used + u] = (int32_t) ((t * s.n_used + u) % s.n_expert);
                continue;
            }
            if (router_mode == 2) { // collide: every token uses the same expert set (token 0's)
                if (t > 0) { for (int64_t u = 0; u < s.n_used; u++) ids[t * s.n_used + u] = ids[u]; continue; }
            }
            for (int64_t u = 0; u < s.n_used; u++) { int64_t j = u + (int64_t) (rr.next() % (uint64_t) (s.n_expert - u)); std::swap(perm[u], perm[j]); ids[t * s.n_used + u] = perm[u]; }
        }
        // first width at which some expert is shared by two different tokens (CPU iqk MoE predictions depend on it)
        int first_collision_k = 0; { std::map<int32_t, int> owner;
            for (int t = 0; t < kmax && !first_collision_k; t++) for (int64_t u = 0; u < s.n_used; u++) {
                auto it = owner.find(ids[t * s.n_used + u]); if (it != owner.end() && it->second != t) { first_collision_k = t + 1; break; } owner[ids[t * s.n_used + u]] = t; } }
        std::set<int32_t> used(ids.begin(), ids.end());
        ggml_tensor * E = mk("E", s.type, s.K, s.M, s.n_expert);
        ggml_tensor * X = mk("X", GGML_TYPE_F32, s.K, s.x_ne1, kmax);
        ggml_tensor * I = mk("ids", GGML_TYPE_I32, s.n_used, kmax);
        const float * im = nullptr;
        if (use_imat_type) { imat = imatrix_for(s.K, rng_for(seed, key, "imatrix")); im = imat.data(); }
        const size_t esz = ggml_row_size(s.type, s.K) * s.M;
        std::vector<uint8_t> eq(esz * s.n_expert);
        std::vector<float> wf(s.K * s.M);
        int32_t first_used = *used.begin();
        // every expert referenced by ids gets its own seeded weights; unreferenced experts (never read) copy one referenced expert
        for (int32_t e : used) { fill_normal(wf, rng_for(seed, key, "E" + std::to_string(e)), 0.05f); quantize_rows(s.type, wf.data(), eq.data() + e * esz, s.M, s.K, im); }
        for (int64_t e = 0; e < s.n_expert; e++) if (!used.count((int32_t) e)) memcpy(eq.data() + e * esz, eq.data() + first_used * esz, esz);
        uploads.emplace_back(E, std::move(eq));
        std::vector<float> xf(s.K * s.x_ne1 * kmax); fill_normal(xf, rng_for(seed, key, "X"), 1.0f); put_f32(X, xf);
        { std::vector<uint8_t> b(ids.size() * 4); memcpy(b.data(), ids.data(), b.size()); uploads.emplace_back(I, std::move(b)); }
        info.str("router", router_mode == 0 ? "random-distinct-per-token" : router_mode == 1 ? "disjoint" : "collide-all-same-set")
            .i64("n_distinct_experts_referenced", (long long) used.size()).raw("first_collision_k", first_collision_k ? std::to_string(first_collision_k) : "null")
            .raw("ids_tokens_major", jarr(ids, [](int32_t v) { return std::to_string(v); })).bl("imatrix", im != nullptr);
    } else if (s.kind == K_FA) {
        // Q stored as llama does: [D, H, T] (per-token heads contiguous), permuted to [D, T, H] for the op.
        // K/V stored like the KV cache: [D, H_kv, n_kv] (per-position heads contiguous), permuted to [D, n_kv, H_kv].
        ggml_tensor * Q = mk("Q", GGML_TYPE_F32, s.D, s.H, kmax);
        ggml_tensor * Kt = mk("Kc", s.type, s.D, s.Hkv, s.nkv);
        ggml_tensor * Vt = mk("Vc", s.vtype, s.D, s.Hkv, s.nkv);
        std::vector<float> qf(s.D * s.H * kmax); fill_normal(qf, rng_for(seed, key, "Q"), 1.0f); put_f32(Q, qf);
        std::vector<float> kf(s.D * s.Hkv * s.nkv);
        fill_normal(kf, rng_for(seed, key, "K"), 1.0f);
        { std::vector<uint8_t> b(ggml_nbytes(Kt)); quantize_rows(s.type, kf.data(), b.data(), s.Hkv * s.nkv, s.D, nullptr); uploads.emplace_back(Kt, std::move(b)); }
        fill_normal(kf, rng_for(seed, key, "V"), 1.0f);
        { std::vector<uint8_t> b(ggml_nbytes(Vt)); quantize_rows(s.vtype, kf.data(), b.data(), s.Hkv * s.nkv, s.D, nullptr); uploads.emplace_back(Vt, std::move(b)); }
        info.num("scale", 1.0 / std::sqrt((double) s.D)).str("prec", "GGML_PREC_F32").num("max_bias", 0).num("softcap", 0).str("sinks", "none")
            .str("mask", "f16 [n_kv, k]: query i (0-based) sees positions j < n_kv-k+i+1 (last k positions = verify rows), -inf elsewhere");
    } else if (s.kind == K_GDN) {
        ggml_tensor * q = mk("q", GGML_TYPE_F32, s.S, s.Hk, kmax, 1);
        ggml_tensor * k = mk("k", GGML_TYPE_F32, s.S, s.Hk, kmax, 1);
        ggml_tensor * v = mk("v", GGML_TYPE_F32, s.S, s.Hv, kmax, 1);
        ggml_tensor * g = mk("g", GGML_TYPE_F32, 1, s.Hv, kmax, 1);
        ggml_tensor * b = mk("beta", GGML_TYPE_F32, 1, s.Hv, kmax, 1);
        ggml_tensor * s0 = mk("s0", s.type, s.S, s.S, s.Hv, 1);
        auto l2rows = [&](std::vector<float> & x, int64_t row) { for (size_t r0 = 0; r0 < x.size(); r0 += row) { double ss = 0; for (int64_t i = 0; i < row; i++) ss += (double) x[r0 + i] * x[r0 + i];
            float inv = (float) (1.0 / std::sqrt(std::max(ss, 1e-12))); for (int64_t i = 0; i < row; i++) x[r0 + i] *= inv; } };
        std::vector<float> qf(s.S * s.Hk * kmax), kf(s.S * s.Hk * kmax), vf(s.S * s.Hv * kmax), gfv(s.Hv * kmax), bfv(s.Hv * kmax), sf(s.S * s.S * s.Hv);
        fill_normal(qf, rng_for(seed, key, "q"), 1.0f); l2rows(qf, s.S);   // qwen35: q,k are L2-normalised (ggml_l2_norm)
        fill_normal(kf, rng_for(seed, key, "k"), 1.0f); l2rows(kf, s.S);
        fill_normal(vf, rng_for(seed, key, "v"), 1.0f);
        { Rng ra = rng_for(seed, key, "alpha"), rA = rng_for(seed, key, "A");
          std::vector<float> A(s.Hv); for (auto & a : A) a = std::exp(0.5f * rA.normal());           // exp(A_log) > 0
          for (int64_t t = 0; t < kmax; t++) for (int64_t h = 0; h < s.Hv; h++) { float x = ra.normal(); float sp = x > 20 ? x : std::log1p(std::exp(x));
              gfv[t * s.Hv + h] = -A[h] * sp; } }                                                          // g = -exp(A_log) * softplus(alpha + dt_bias)
        { Rng rb = rng_for(seed, key, "beta"); for (auto & x : bfv) x = 1.0f / (1.0f + std::exp(-rb.normal())); } // beta = sigmoid(.)
        fill_normal(sf, rng_for(seed, key, "s0"), 0.1f);
        put_f32(q, qf); put_f32(k, kf); put_f32(v, vf); put_f32(g, gfv); put_f32(b, bfv);
        if (s.type == GGML_TYPE_F32) put_f32(s0, sf);
        else { std::vector<uint8_t> bb(ggml_nbytes(s0)); quantize_rows(s.type, sf.data(), bb.data(), s.S * s.Hv, s.S, nullptr); uploads.emplace_back(s0, std::move(bb)); }
        info.str("gate_form", "scalar g [1,H_v,T,1] = -exp(A_log)*softplus(alpha) (qwen35 build); beta=sigmoid; q,k L2-normalised; K snapshots=1")
            .str("state_threading", "serial op i+1 reads state view of serial op i's result (offset S_v*H_v*1 elements); batched final state vs last serial state");
    }
    op.info_json = info.done();
    op.buf = ggml_backend_alloc_ctx_tensors(op.ctx, be);
    if (!op.buf) return "ggml_backend_alloc_ctx_tensors failed for operands";
    for (auto & u : uploads) {
        if (u.second.size() != ggml_nbytes(u.first)) return std::string("operand size mismatch for ") + ggml_get_name(u.first);
        ggml_backend_tensor_set(u.first, u.second.data(), 0, u.second.size());
    }
    return "";
}

static ggml_tensor * out_mark(KGraph & kg, ggml_tensor * t) { ggml_set_output(t); kg.outputs.push_back(t); ggml_build_forward_expand(kg.gf, t); return t; }

static std::string build_kgraph(ggml_backend_t be, const Sub & s, Operands & op, int k, KGraph & kg) {
    const int max_nodes = 64 + 16 * k;
    kg.ctx = new_meta_ctx(max_nodes * 4);
    kg.gf = ggml_new_graph_custom(kg.ctx, max_nodes * 4, false);
    ggml_context * c = kg.ctx;
    if (s.kind == K_MM || s.kind == K_PC1) {
        ggml_tensor * W = op.t["W"]; ggml_tensor * X = op.t["X"];
        ggml_tensor * Xb = ggml_view_2d(c, X, s.K, k, X->nb[1], 0);
        ggml_tensor * Yb = out_mark(kg, ggml_mul_mat(c, W, Xb)); kg.main_ops.push_back(Yb);
        kg.bseg.push_back({Yb, 0, ggml_nbytes(Yb)});
        for (int i = 0; i < k; i++) {
            ggml_tensor * Yi;
            if (s.kind == K_MM) {
                ggml_tensor * Xi = ggml_view_2d(c, X, s.K, 1, X->nb[1], (size_t) i * X->nb[1]);
                Yi = ggml_mul_mat(c, W, Xi);
                kg.main_ops.push_back(Yi);
            } else {
                ggml_tensor * Xlo = ggml_view_2d(c, X, s.K / 2, 1, X->nb[1], (size_t) i * X->nb[1]);
                ggml_tensor * Xhi = ggml_view_2d(c, X, s.K / 2, 1, X->nb[1], (size_t) i * X->nb[1] + (size_t) (s.K / 2) * sizeof(float));
                ggml_tensor * a = ggml_mul_mat(c, op.t["Wlo"], Xlo); ggml_tensor * b = ggml_mul_mat(c, op.t["Whi"], Xhi);
                Yi = ggml_add(c, a, b);
                kg.main_ops.push_back(a); kg.main_ops.push_back(b); kg.main_ops.push_back(Yi);
            }
            out_mark(kg, Yi);
            kg.sseg.push_back({Yi, 0, ggml_nbytes(Yi)});
            kg.col_elems.push_back(s.M); kg.col_kind.push_back("col");
        }
    } else if (s.kind == K_MMID) {
        ggml_tensor * E = op.t["E"]; ggml_tensor * X = op.t["X"]; ggml_tensor * I = op.t["ids"];
        ggml_tensor * Xb = ggml_view_3d(c, X, s.K, s.x_ne1, k, X->nb[1], X->nb[2], 0);
        ggml_tensor * Ib = ggml_view_2d(c, I, s.n_used, k, I->nb[1], 0);
        ggml_tensor * Yb = out_mark(kg, ggml_mul_mat_id(c, E, Xb, Ib)); kg.main_ops.push_back(Yb);
        kg.bseg.push_back({Yb, 0, ggml_nbytes(Yb)});
        for (int i = 0; i < k; i++) {
            ggml_tensor * Xi = ggml_view_3d(c, X, s.K, s.x_ne1, 1, X->nb[1], X->nb[2], (size_t) i * X->nb[2]);
            ggml_tensor * Ii = ggml_view_2d(c, I, s.n_used, 1, I->nb[1], (size_t) i * I->nb[1]);
            ggml_tensor * Yi = out_mark(kg, ggml_mul_mat_id(c, E, Xi, Ii)); kg.main_ops.push_back(Yi);
            kg.sseg.push_back({Yi, 0, ggml_nbytes(Yi)});
            kg.col_elems.push_back(s.M * s.n_used); kg.col_kind.push_back("token");
        }
    } else if (s.kind == K_FA) {
        ggml_tensor * Q = op.t["Q"]; ggml_tensor * Kc = op.t["Kc"]; ggml_tensor * Vc = op.t["Vc"];
        // k-specific mask input
        kg.in_ctx = new_meta_ctx(4);
        ggml_tensor * mask = ggml_new_tensor_2d(kg.in_ctx, GGML_TYPE_F16, s.nkv, k);
        ggml_set_name(mask, "mask");
        kg.in_buf = ggml_backend_alloc_ctx_tensors(kg.in_ctx, be);
        if (!kg.in_buf) return "mask buffer allocation failed";
        {
            std::vector<ggml_fp16_t> m((size_t) s.nkv * k);
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int i = 0; i < k; i++) for (int64_t j = 0; j < s.nkv; j++) m[(size_t) i * s.nkv + j] = (j < s.nkv - k + i + 1) ? zero : ninf;
            ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
        }
        ggml_tensor * Kp = ggml_permute(c, Kc, 0, 2, 1, 3);   // [D, n_kv, H_kv]
        ggml_tensor * Vp = ggml_permute(c, Vc, 0, 2, 1, 3);
        const float scale = 1.0f / std::sqrt((float) s.D);
        ggml_tensor * Qb = ggml_permute(c, ggml_view_3d(c, Q, s.D, s.H, k, Q->nb[1], Q->nb[2], 0), 0, 2, 1, 3);
        ggml_tensor * Ob = ggml_flash_attn_ext(c, Qb, Kp, Vp, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(Ob, GGML_PREC_F32);
        out_mark(kg, Ob); kg.main_ops.push_back(Ob);
        kg.bseg.push_back({Ob, 0, ggml_nbytes(Ob)});            // [D, H, k]
        for (int i = 0; i < k; i++) {
            ggml_tensor * Qi = ggml_permute(c, ggml_view_3d(c, Q, s.D, s.H, 1, Q->nb[1], Q->nb[2], (size_t) i * Q->nb[2]), 0, 2, 1, 3);
            ggml_tensor * Mi = ggml_view_2d(c, mask, s.nkv, 1, mask->nb[1], (size_t) i * mask->nb[1]);
            ggml_tensor * Oi = ggml_flash_attn_ext(c, Qi, Kp, Vp, Mi, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(Oi, GGML_PREC_F32);
            out_mark(kg, Oi); kg.main_ops.push_back(Oi);
            kg.sseg.push_back({Oi, 0, ggml_nbytes(Oi)});
            kg.col_elems.push_back(s.D * s.H); kg.col_kind.push_back("query");
        }
    } else if (s.kind == K_GDN) {
        ggml_tensor * q = op.t["q"]; ggml_tensor * kk = op.t["k"]; ggml_tensor * v = op.t["v"]; ggml_tensor * g = op.t["g"]; ggml_tensor * b = op.t["beta"];
        ggml_tensor * s0 = op.t["s0"];
        auto tok = [&](ggml_tensor * t, int64_t t0, int64_t n) { return ggml_view_4d(c, t, t->ne[0], t->ne[1], n, 1, t->nb[1], t->nb[2], t->nb[3], (size_t) t0 * t->nb[2]); };
        const size_t es = ggml_type_size(s.type);
        kg.elem_size = es; kg.out_type = s.type;
        ggml_tensor * Rb = ggml_gated_delta_net(c, tok(q, 0, k), tok(kk, 0, k), tok(v, 0, k), tok(g, 0, k), tok(b, 0, k), s0, 1);
        out_mark(kg, Rb); kg.main_ops.push_back(Rb);
        const size_t attn_tok = (size_t) s.S * s.Hv * es, state_bytes = (size_t) s.S * s.S * s.Hv * es;
        kg.bseg.push_back({Rb, 0, attn_tok * k});
        kg.bseg.push_back({Rb, attn_tok * k, state_bytes});
        ggml_tensor * st = s0; ggml_tensor * Rlast = nullptr;
        for (int i = 0; i < k; i++) {
            ggml_tensor * Ri = ggml_gated_delta_net(c, tok(q, i, 1), tok(kk, i, 1), tok(v, i, 1), tok(g, i, 1), tok(b, i, 1), st, 1);
            out_mark(kg, Ri); kg.main_ops.push_back(Ri);
            kg.sseg.push_back({Ri, 0, attn_tok});
            kg.col_elems.push_back(s.S * s.Hv); kg.col_kind.push_back("token");
            st = ggml_view_4d(c, Ri, s.S, s.S, s.Hv, 1, ggml_row_size(s.type, s.S), ggml_row_size(s.type, s.S * s.S),
                              ggml_row_size(s.type, s.S * s.S * s.Hv), ggml_row_size(s.type, s.S * s.Hv * 1));
            Rlast = Ri;
        }
        kg.sseg.push_back({Rlast, attn_tok, state_bytes});
        kg.col_elems.push_back(s.S * s.S * s.Hv); kg.col_kind.push_back("final_state");
    }
    if (kg.col_kind.size() && kg.col_kind[0] != "token" && kg.col_kind[0] != "final_state") {} // layout already per column
    for (ggml_tensor * n : kg.main_ops) if (!ggml_backend_supports_op(be, n)) {
        bool batched = (n == kg.main_ops[0]);
        return std::string("UNSUPPORTED:ggml_backend_supports_op==false for ") + ggml_op_desc(n) + (batched ? " (batched node)" : " (serial node)");
    }
    kg.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(kg.galloc, kg.gf)) return "ggml_gallocr_alloc_graph failed";
    return "";
}

static void read_segs(const std::vector<Seg> & segs, std::vector<uint8_t> & out) {
    size_t tot = 0; for (auto & s : segs) tot += s.size;
    out.resize(tot); size_t o = 0;
    for (auto & s : segs) { ggml_backend_tensor_get(s.t, out.data() + o, s.off, s.size); o += s.size; }
}

static double elem_value(const uint8_t * p, ggml_type t) {
    if (t == GGML_TYPE_BF16) { uint16_t h; memcpy(&h, p, 2); uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
    float f; memcpy(&f, p, 4); return f;
}
static std::string elem_hex(const uint8_t * p, size_t es) {
    uint32_t u = 0; memcpy(&u, p, es); char b[16]; snprintf(b, sizeof b, es == 2 ? "0x%04x" : "0x%08x", u); return b;
}

struct CellSummary { std::string status; int first_fail_k = 0; int first_fail_k_stable = 0; std::vector<int> unstable_ks, unsupported_ks, err_ks, k1_violation; std::string err; };

static CellSummary run_sub(ggml_backend_t be, const std::string & be_name, const std::string & be_desc, const Row & r, int sub_idx, const Sub & s,
                           uint64_t seed, int kmax, int repeats, int threads, int router_mode) {
    CellSummary cs;
    g_cur_id = r.id; g_cur_sub = s.label; g_cur_backend = r.backend; g_cur_k = 0;
    const std::string key = r.id + "#" + s.label;
    auto base = [&](JW & w, const char * rec) {
        w.str("record", rec).str("id", r.id).i64("sub", sub_idx).str("sub_label", s.label).str("backend_row", r.backend)
         .str("backend_bound", be_name).str("device_description", be_desc).str("op", r.op).str("type", r.type).i64("threads", threads); };
    auto summary = [&](const std::string & status, const std::string & reason, const std::string & opinfo, const std::vector<std::string> & digests_note) {
        (void) digests_note;
        JW w; base(w, "cell");
        w.str("status", status);
        if (!reason.empty()) w.str("reason", reason);
        w.i64("kmax", kmax).i64("repeats", repeats).raw("env", env_json()).raw("operands", opinfo.empty() ? "null" : opinfo);
        if (status == "OK" || status == "UNSTABLE") {
            w.raw("first_fail_k", cs.first_fail_k ? std::to_string(cs.first_fail_k) : "null");
            w.raw("first_fail_k_stable_repeats_only", cs.first_fail_k_stable ? std::to_string(cs.first_fail_k_stable) : "null");
            w.str("verdict", status == "UNSTABLE" ? "UNSTABLE" : (cs.first_fail_k ? "FAIL" : "PASS"));
        } else {
            w.raw("first_fail_k", "null").str("verdict", status);
        }
        auto ia = [](const std::vector<int> & v) { return jarr(v, [](int x) { return std::to_string(x); }); };
        w.raw("unstable_ks", ia(cs.unstable_ks)).raw("unsupported_ks", ia(cs.unsupported_ks)).raw("err_ks", ia(cs.err_ks)).raw("k1_identity_violation", ia(cs.k1_violation));
        if (r.id == "PC-1") w.str("expected_verdict", "FAIL");
        emit(w.done());
    };

    if (s.kind == K_NOTPROBED) { cs.status = "NOT_PROBED"; summary("NOT_PROBED", s.note, "", {}); return cs; }
    if (!s.unsupported_reason.empty()) { cs.status = "UNSUPPORTED"; summary("UNSUPPORTED", s.unsupported_reason, "", {}); return cs; }
    const std::string em = r.id == "PC-1" ? "" : env_mismatch(r.backend, be_name == "CPU" ? "cpu" : "rocm0");
    if (!em.empty()) { cs.status = "ERR"; summary("ERR", "env/backend mismatch: " + em, "", {}); return cs; }

    Operands op;
    auto t0 = std::chrono::steady_clock::now();
    std::string e = build_operands(be, s, key, seed, kmax, op, router_mode);
    double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[width_probe] %s %s operands ready in %.0f ms\n", r.id.c_str(), s.label.c_str(), gen_ms);
    if (!e.empty()) { cs.status = "ERR"; summary("ERR", e, op.info_json, {}); return cs; }

    int n_ok_k = 0;
    for (int k = 1; k <= kmax; k++) {
        g_cur_k = k;
        KGraph kg;
        std::string be_err = build_kgraph(be, s, op, k, kg);
        JW w; base(w, "k"); w.i64("k", k);
        if (!be_err.empty()) {
            bool uns = be_err.rfind("UNSUPPORTED:", 0) == 0;
            (uns ? cs.unsupported_ks : cs.err_ks).push_back(k);
            w.str("status", uns ? "UNSUPPORTED" : "ERR").str("reason", uns ? be_err.substr(12) : be_err);
            emit(w.done()); continue;
        }
        std::vector<std::string> db, ds; std::vector<bool> differs; std::vector<double> ms;
        std::string fde = "null"; long long n_diff = 0; double max_abs = 0; bool any_diff = false; std::string cerr;
        std::vector<uint8_t> bb, sb;
        for (int rep = 0; rep < repeats; rep++) {
            for (ggml_tensor * t : kg.outputs) ggml_backend_tensor_memset(t, 0xA5, 0, ggml_nbytes(t)); // poison: unwritten bytes cannot alias a previous repeat
            auto c0 = std::chrono::steady_clock::now();
            ggml_status st = ggml_backend_graph_compute(be, kg.gf);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count());
            if (st != GGML_STATUS_SUCCESS) { cerr = std::string("ggml_backend_graph_compute status ") + ggml_status_to_string(st); break; }
            read_segs(kg.bseg, bb); read_segs(kg.sseg, sb);
            if (bb.size() != sb.size()) { cerr = "layout size mismatch batched vs serial"; break; }
            db.push_back(sha256_hex(bb)); ds.push_back(sha256_hex(sb));
            bool d = memcmp(bb.data(), sb.data(), bb.size()) != 0;
            differs.push_back(d);
            if (d && !any_diff) {
                any_diff = true;
                const size_t es = kg.elem_size; const size_t ne = bb.size() / es;
                size_t first = SIZE_MAX;
                for (size_t i = 0; i < ne; i++) if (memcmp(&bb[i * es], &sb[i * es], es) != 0) {
                    if (first == SIZE_MAX) first = i;
                    n_diff++;
                    double a = elem_value(&bb[i * es], kg.out_type), b2 = elem_value(&sb[i * es], kg.out_type);
                    double ad = std::fabs(a - b2); if (std::isfinite(ad)) max_abs = std::max(max_abs, ad); else max_abs = INFINITY;
                }
                size_t col = 0, rem = first; while (col < kg.col_elems.size() && rem >= (size_t) kg.col_elems[col]) { rem -= kg.col_elems[col]; col++; }
                JW f; f.i64("repeat", rep).i64("flat_index", (long long) first).i64("column", (long long) col)
                     .str("column_kind", col < kg.col_kind.size() ? kg.col_kind[col] : "?").i64("element_in_column", (long long) rem)
                     .str("batched_hex", elem_hex(&bb[first * es], es)).str("serial_hex", elem_hex(&sb[first * es], es))
                     .num("batched_value", elem_value(&bb[first * es], kg.out_type)).num("serial_value", elem_value(&sb[first * es], kg.out_type));
                fde = f.done();
            }
        }
        if (!cerr.empty()) { cs.err_ks.push_back(k); w.str("status", "ERR").str("reason", cerr); emit(w.done()); continue; }
        auto all_eq = [](const std::vector<std::string> & v) { for (auto & x : v) if (x != v[0]) return false; return true; };
        const bool bst = all_eq(db), sst = all_eq(ds);
        const bool wd = any_diff;
        const bool stable = bst && sst;
        if (!stable) cs.unstable_ks.push_back(k);
        if (wd && !cs.first_fail_k) cs.first_fail_k = k;
        if (wd && stable && !cs.first_fail_k_stable) cs.first_fail_k_stable = k;
        if (k == 1 && wd) cs.k1_violation.push_back(1);
        n_ok_k++;
        auto sa = [](const std::vector<std::string> & v) { return jarr(v, [](const std::string & x) { return jesc(x); }); };
        w.str("status", stable ? "OK" : "UNSTABLE").str("digest_algo", "sha256")
         .raw("digest_batched", sa(db)).raw("digest_serial", sa(ds))
         .bl("batched_repeat_stable", bst).bl("serial_repeat_stable", sst).bl("width_differs", wd)
         .raw("width_differs_by_repeat", jarr(differs, [](bool x) { return std::string(x ? "true" : "false"); }))
         .raw("first_differing_element", fde).i64("n_diff_elements_first_differing_repeat", n_diff).num("max_abs_diff_first_differing_repeat", max_abs)
         .i64("compared_bytes", (long long) bb.size()).str("elem_type", ggml_type_name(kg.out_type)).i64("graph_nodes", ggml_graph_n_nodes(kg.gf))
         .raw("compute_ms", jarr(ms, [](double x) { return jnum(x); }));
        emit(w.done());
        fprintf(stderr, "[width_probe] %s %s k=%d %s%s\n", r.id.c_str(), s.label.c_str(), k, wd ? "DIFF" : "same", stable ? "" : " UNSTABLE");
    }
    g_cur_k = 0;
    if (n_ok_k == 0) {
        if (!cs.unsupported_ks.empty() && cs.err_ks.empty()) { cs.status = "UNSUPPORTED"; summary("UNSUPPORTED", "every width unsupported by the backend (supports_op)", op.info_json, {}); }
        else { cs.status = "ERR"; summary("ERR", "no width computed", op.info_json, {}); }
        return cs;
    }
    cs.status = cs.unstable_ks.empty() ? "OK" : "UNSTABLE";
    summary(cs.status, "", op.info_json, {});
    return cs;
}

// ------------------------------------------------------------------------------------------------
static void usage() {
    fprintf(stderr,
        "usage: width_probe --backend cpu|rocm0 --cases <predicted.jsonl> [--ids ID1,ID2|all] [--pc1] [--kmax 9] [--repeats 3]\n"
        "                   [--threads N] [--quant-threads N] [--seed S] [--router random|disjoint|collide] [--backend-dl-dir DIR]\n"
        "                   --out <records.jsonl>\n"
        "  rows are filtered to the backend family (cpu*: --backend cpu; rocm0: --backend rocm0; graph rows: NOT_PROBED record).\n"
        "  --pc1 adds the positive control PC-1 for the bound backend; '--ids PC-1' runs only it.\n");
}

int main(int argc, char ** argv) {
    std::string backend, cases, out, ids_arg = "all", dl_dir, router = "random";
    int kmax = 9, repeats = 3, threads = 16, qthreads = -1; uint64_t seed = 0xA4C26ull; bool pc1 = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(4); } return argv[++i]; };
        if (a == "--backend") backend = nxt(); else if (a == "--cases") cases = nxt(); else if (a == "--out") out = nxt();
        else if (a == "--ids") ids_arg = nxt(); else if (a == "--kmax") kmax = atoi(nxt().c_str()); else if (a == "--repeats") repeats = atoi(nxt().c_str());
        else if (a == "--threads") threads = atoi(nxt().c_str()); else if (a == "--quant-threads") qthreads = atoi(nxt().c_str());
        else if (a == "--seed") seed = strtoull(nxt().c_str(), nullptr, 0); else if (a == "--pc1") pc1 = true;
        else if (a == "--router") router = nxt(); else if (a == "--backend-dl-dir") dl_dir = nxt();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 4; }
    }
    if ((backend != "cpu" && backend != "rocm0") || out.empty() || kmax < 1 || repeats < 1 || threads < 1) { usage(); return 4; }
    int router_mode = router == "random" ? 0 : router == "disjoint" ? 1 : router == "collide" ? 2 : -1;
    if (router_mode < 0) { usage(); return 4; }
    g_quant_threads = qthreads > 0 ? qthreads : threads;

    std::set<std::string> want; bool want_all = ids_arg == "all";
    { std::stringstream ss(ids_arg); std::string x; while (std::getline(ss, x, ',')) if (!x.empty()) want.insert(x); }
    if (want.count("PC-1")) pc1 = true;

    std::vector<Row> rows;
    if (!cases.empty()) {
        FILE * f = fopen(cases.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", cases.c_str()); return 4; }
        std::string line; char buf[1 << 16];
        std::string acc;
        while (fgets(buf, sizeof buf, f)) { acc += buf; if (acc.empty() || acc.back() != '\n') continue;
            line.swap(acc); acc.clear();
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            JVal v; JParser p(line); if (!p.parse(v) || v.t != JVal::OBJ) { fprintf(stderr, "bad JSON line in %s: %s\n", cases.c_str(), p.err.c_str()); fclose(f); return 4; }
            Row r; r.id = v.str("id"); r.backend = v.str("backend"); r.op = v.str("op"); r.type = v.str("type"); r.shape = v.str("shape");
            if (!want_all && !want.count(r.id)) continue;
            const bool fam = (backend == "cpu") ? (r.backend.rfind("cpu", 0) == 0 || r.backend == "graph") : (r.backend == "rocm0");
            if (!fam) continue;
            if (r.backend == "graph" && backend != "cpu") continue; // graph rows reported once (by the CPU half)
            resolve_row(r); rows.push_back(r);
        }
        if (!acc.empty()) { fprintf(stderr, "truncated last line in %s\n", cases.c_str()); fclose(f); return 4; }
        fclose(f);
    }
    if (pc1) rows.push_back(pc1_row(backend));
    if (rows.empty()) { fprintf(stderr, "no rows selected\n"); return 4; }

    g_out = fopen(out.c_str(), "a");
    if (!g_out) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 4; }
    ggml_set_abort_callback(on_ggml_abort);

    // ---- explicit backend binding; never fall back
    ggml_backend_t be = nullptr; std::string be_name, be_desc;
    if (backend == "cpu") {
        be = ggml_backend_cpu_init();
        if (!be) { fprintf(stderr, "FATAL: ggml_backend_cpu_init failed\n"); return 4; }
        ggml_backend_cpu_set_n_threads(be, threads);
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        be_name = ggml_backend_name(be);
        be_desc = dev ? ggml_backend_dev_description(dev) : "?";
        if (be_name != "CPU" || (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU)) {
            fprintf(stderr, "FATAL: requested cpu but bound '%s' -- aborting (no fallback)\n", be_name.c_str()); return 4; }
    } else {
        if (!dl_dir.empty()) ggml_backend_load_all_from_path(dl_dir.c_str());
        ggml_backend_dev_t dev = ggml_backend_dev_by_name("ROCm0");
        if (!dev) { fprintf(stderr, "FATAL: requested rocm0 but no device named 'ROCm0' is registered (is this the HIP build's libggml?) -- aborting, no fallback\n"); return 4; }
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { fprintf(stderr, "FATAL: device ROCm0 is not a GPU device -- aborting\n"); return 4; }
        be = ggml_backend_dev_init(dev, nullptr);
        if (!be) { fprintf(stderr, "FATAL: ggml_backend_dev_init(ROCm0) failed -- aborting\n"); return 4; }
        be_name = ggml_backend_name(be); be_desc = ggml_backend_dev_description(dev);
        if (be_name.rfind("ROCm", 0) != 0) { fprintf(stderr, "FATAL: requested rocm0 but bound backend '%s' -- aborting\n", be_name.c_str()); return 4; }
    }
    fprintf(stderr, "[width_probe] bound backend: name='%s' description='%s' threads=%d kmax=%d repeats=%d seed=0x%" PRIx64 "\n",
            be_name.c_str(), be_desc.c_str(), threads, kmax, repeats, seed);
    { JW w; w.str("record", "run").str("backend_requested", backend).str("backend_bound", be_name).str("device_description", be_desc)
        .i64("threads", threads).i64("kmax", kmax).i64("repeats", repeats).str("seed", std::to_string(seed)).str("router", router)
        .raw("env", env_json()).str("ggml_version", ggml_version()).str("ggml_commit", ggml_commit());
      emit(w.done()); }

    int rc = 0;
    for (auto & r : rows) {
        if (!r.resolve_error.empty()) {
            JW w; w.str("record", "cell").str("id", r.id).i64("sub", 0).str("sub_label", "?").str("backend_row", r.backend).str("op", r.op).str("type", r.type)
                .str("status", "ERR").str("reason", r.resolve_error).raw("first_fail_k", "null").str("verdict", "ERR"); emit(w.done()); continue; }
        for (size_t si = 0; si < r.subs.size(); si++) {
            CellSummary cs = run_sub(be, be_name, be_desc, r, (int) si, r.subs[si], seed, kmax, repeats, threads, router_mode);
            if (cs.status == "UNSTABLE") rc = rc ? rc : 3;
            if (r.id == "PC-1" && !(cs.status == "OK" && cs.first_fail_k)) { fprintf(stderr, "[width_probe] PC-1 did NOT fail (status %s)\n", cs.status.c_str()); rc = 2; }
        }
        JW w; w.str("record", "row_done").str("id", r.id).str("backend_row", r.backend).i64("n_sub", (long long) r.subs.size()); emit(w.done());
    }
    ggml_backend_free(be);
    ggml_quantize_free();
    fclose(g_out);
    return rc;
}
