// act_dump: full-tensor activation dumper over the public libllama cb_eval hook (AK-PROF-SCLK-CAPT).
//
// Loads a GGUF, runs ONE prompt batch of exactly N tokens, and writes the full contents of the operands of
//   * every MUL_MAT whose weight (src0) is named "blk.<L>.*"   -> act_<proj>.bin  (src1, the activation input)
//                                                              -> w_<proj>.bin    (src0, raw weight blocks; --weights)
//   * the FLASH_ATTN_EXT node of layer <L>                     -> fa_q.bin / fa_k.bin / fa_v.bin
// Device tensors are copied to host with ggml_backend_tensor_get. Non-contiguous views are gathered into a
// contiguous ne0-fastest layout before writing. Each .bin gets a .json sidecar (name, ggml type, ne, source nb,
// bytes, basic stats); manifest.json lists all of them.
//
// FA is matched by node name "*-<L>" or by its K operand being a view of "cache_k_l<L>".
//
// --head mode (DRAFT-HEAD-APPROX-RECALL): instead of layer L, match every MUL_MAT whose weight is the LM head
// ("output.weight", or any "*shared_head_head*" / "*shared_head.head*" tensor, or --head-name NAME). The batch requests
// logits for EVERY token so the head runs on all N rows. Per matched head node it writes
//   head_hidden.bin   f32 [n_embd, N]   the pre-head hidden state (src1)
//   head_w.f32.bin    f32 [n_embd, V]   the head weight dequantized with ggml to_float (raw blocks too with --weights)
//   head_topk_ids.bin i32 [K, N]        top-K vocab ids of the model's own head logits, descending (K = --head-topk, 16)
//   head_topk_val.bin f32 [K, N]        their logits
//   head_logits.bin   f32 [V, N]        only with --head-logits-full
// Usage: act_dump -m model.gguf -o outdir [--layer L] [-n 512] [-ngl 0] [-t 8] [--prompt-file f] [--weights]
//                 [--head [--head-name NAME] [--head-topk K] [--head-logits-full]]
// Exit: 0 ok | 1 usage | 2 load/decode failure | 3 nothing matched (no MUL_MAT or no FA at layer L; --head: no head MUL_MAT)
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct dump_state {
    std::string out;
    int layer = 3;
    bool weights = false;
    bool list = false;  // --list: print candidate nodes (debug)
    bool head = false;  // --head: dump the LM-head MUL_MAT instead of layer L
    std::string head_name;
    int head_topk = 16;
    bool head_logits_full = false;
    int n_head = 0;
    std::map<std::string, int> seen;
    std::vector<std::string> manifest;  // JSON objects
    int n_mm = 0, n_fa = 0;
};

static bool ends_with(const std::string & s, const std::string & suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// The FA node itself is usually unnamed; its K operand is a view of the layer's cache tensor "cache_k_l<L>".
static bool kv_layer(const ggml_tensor * k, int layer) {
    if (!k) return false;
    const std::string tag = "cache_k_l" + std::to_string(layer);
    for (const char * p = strstr(k->name, tag.c_str()); p; p = strstr(p + 1, tag.c_str())) {
        const char c = p[tag.size()];
        if (c < '0' || c > '9') return true;
    }
    return false;
}

// Read a (possibly strided) tensor into a contiguous, ne0-fastest host buffer.
static bool fetch_contig(const ggml_tensor * t, std::vector<uint8_t> & dst) {
    if (!t->buffer) {
        return false;
    }
    const size_t span = ggml_nbytes(t);
    std::vector<uint8_t> raw(span);
    ggml_backend_tensor_get(t, raw.data(), 0, span);
    if (ggml_is_contiguous(t)) {
        dst.swap(raw);
        return true;
    }
    const size_t ts = ggml_type_size(t->type);
    const int64_t bs = ggml_blck_size(t->type);
    if (t->nb[0] == ts) {  // rows contiguous: gather row by row
        const size_t rb = ggml_row_size(t->type, t->ne[0]);
        dst.resize(rb * t->ne[1] * t->ne[2] * t->ne[3]);
        size_t o = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
            for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                    memcpy(dst.data() + o, raw.data() + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], rb);
                    o += rb;
                }
        return true;
    }
    if (bs != 1) {
        return false;  // transposed quantized view: not supported
    }
    dst.resize(ts * ggml_nelements(t));
    size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    memcpy(dst.data() + o, raw.data() + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], ts);
                    o += ts;
                }
    return true;
}

static std::string stats_json(const ggml_tensor * t, const std::vector<uint8_t> & b) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        return "null";
    }
    const int64_t n = ggml_nelements(t);
    double sum = 0, asum = 0, mn = INFINITY, mx = -INFINITY;
    int64_t nonfinite = 0, zeros = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = t->type == GGML_TYPE_F32 ? ((const float *) b.data())[i] : ggml_fp16_to_fp32(((const ggml_fp16_t *) b.data())[i]);
        if (!std::isfinite(v)) { nonfinite++; continue; }
        if (v == 0.0f) zeros++;
        sum += v; asum += std::fabs(v); mn = std::min(mn, (double) v); mx = std::max(mx, (double) v);
    }
    const double k = n - nonfinite > 0 ? (double) (n - nonfinite) : 1.0;
    std::ostringstream s;
    s.precision(9);
    s << "{\"mean\":" << sum / k << ",\"abs_mean\":" << asum / k << ",\"min\":" << (std::isfinite(mn) ? mn : 0)
      << ",\"max\":" << (std::isfinite(mx) ? mx : 0) << ",\"nonfinite\":" << nonfinite << ",\"zeros\":" << zeros << "}";
    return s.str();
}

static void dump(dump_state & st, const ggml_tensor * t, std::string key, const ggml_tensor * node) {
    int & c = st.seen[key];
    if (c++ > 0) {
        key += "_" + std::to_string(c - 1);
    }
    std::vector<uint8_t> b;
    if (!fetch_contig(t, b)) {
        fprintf(stderr, "act_dump: cannot fetch %s (%s)\n", key.c_str(), t->name);
        return;
    }
    const std::string bin = st.out + "/" + key + ".bin";
    std::ofstream(bin, std::ios::binary).write((const char *) b.data(), (std::streamsize) b.size());
    std::ostringstream j;
    j << "{\"key\":\"" << key << "\",\"file\":\"" << key << ".bin\",\"tensor\":\"" << t->name << "\",\"node\":\"" << node->name
      << "\",\"op\":\"" << ggml_op_name(node->op) << "\",\"type\":\"" << ggml_type_name(t->type) << "\",\"ne\":[" << t->ne[0] << ","
      << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "],\"src_nb\":[" << t->nb[0] << "," << t->nb[1] << "," << t->nb[2] << ","
      << t->nb[3] << "],\"contiguous_src\":" << (ggml_is_contiguous(t) ? "true" : "false") << ",\"bytes\":" << b.size()
      << ",\"backend\":\"" << (t->buffer ? ggml_backend_buffer_name(t->buffer) : "none") << "\",\"stats\":" << stats_json(t, b) << "}";
    std::ofstream(st.out + "/" + key + ".json") << j.str() << "\n";
    st.manifest.push_back(j.str());
    fprintf(stderr, "act_dump: %-14s %-6s ne=[%lld,%lld,%lld,%lld] %zu B\n", key.c_str(), ggml_type_name(t->type), (long long) t->ne[0],
            (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], b.size());
}

static bool is_head_weight(const dump_state & st, const ggml_tensor * w) {
    if (!w) return false;
    const std::string n = w->name;
    if (!st.head_name.empty()) return n == st.head_name;
    return n == "output.weight" || n.find("shared_head_head") != std::string::npos || n.find("shared_head.head") != std::string::npos;
}

static void write_bin(dump_state & st, const std::string & key, const void * p, size_t bytes, const std::string & type,
                      const std::vector<int64_t> & ne, const ggml_tensor * node) {
    std::ofstream(st.out + "/" + key + ".bin", std::ios::binary).write((const char *) p, (std::streamsize) bytes);
    std::ostringstream j;
    j << "{\"key\":\"" << key << "\",\"file\":\"" << key << ".bin\",\"tensor\":\"" << node->src[0]->name << "\",\"node\":\"" << node->name
      << "\",\"op\":\"" << ggml_op_name(node->op) << "\",\"type\":\"" << type << "\",\"ne\":[";
    for (size_t i = 0; i < 4; i++) j << (i ? "," : "") << (i < ne.size() ? ne[i] : 1);
    j << "],\"contiguous_src\":true,\"bytes\":" << bytes << ",\"stats\":null}";
    std::ofstream(st.out + "/" + key + ".json") << j.str() << "\n";
    st.manifest.push_back(j.str());
    fprintf(stderr, "act_dump: %-18s %s %zu B\n", key.c_str(), type.c_str(), bytes);
}

// LM-head node: hidden (src1), dequantized weight (src0), and top-K of the node's own f32 output.
static void dump_head(dump_state & st, ggml_tensor * t) {
    std::string sfx = st.n_head ? "_" + std::to_string(st.n_head) : "";
    const ggml_tensor * w = t->src[0];
    dump(st, t->src[1], "head_hidden" + sfx, t);
    if (st.weights) dump(st, w, "head_w_raw" + sfx, t);
    std::vector<uint8_t> wb;
    if (!fetch_contig(w, wb)) { fprintf(stderr, "act_dump: cannot fetch head weight %s\n", w->name); return; }
    const int64_t d = w->ne[0], V = w->ne[1];
    {
        std::vector<float> wf((size_t) d * V);
        const size_t rb = ggml_row_size(w->type, d);
        if (w->type == GGML_TYPE_F32) memcpy(wf.data(), wb.data(), wf.size() * 4);
        else {
            const auto * tr = ggml_get_type_traits(w->type);
            if (!tr->to_float) { fprintf(stderr, "act_dump: no to_float for %s\n", ggml_type_name(w->type)); return; }
            for (int64_t r = 0; r < V; r++) tr->to_float(wb.data() + r * rb, wf.data() + r * d, d);
        }
        write_bin(st, "head_w.f32" + sfx, wf.data(), wf.size() * 4, "f32", {d, V}, t);
    }
    std::vector<uint8_t> lb;
    if (t->type != GGML_TYPE_F32 || !fetch_contig(t, lb)) { fprintf(stderr, "act_dump: cannot fetch head logits\n"); return; }
    const int64_t N = t->ne[1], K = std::min<int64_t>(st.head_topk, V);
    const float * L = (const float *) lb.data();
    std::vector<int32_t> ids((size_t) K * N);
    std::vector<float> vals((size_t) K * N);
    std::vector<int32_t> idx(V);
    for (int64_t c = 0; c < N; c++) {
        const float * col = L + c * V;
        for (int64_t i = 0; i < V; i++) idx[i] = (int32_t) i;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) { return col[a] > col[b] || (col[a] == col[b] && a < b); });
        for (int64_t k = 0; k < K; k++) { ids[c * K + k] = idx[k]; vals[c * K + k] = col[idx[k]]; }
    }
    write_bin(st, "head_topk_ids" + sfx, ids.data(), ids.size() * 4, "i32", {K, N}, t);
    write_bin(st, "head_topk_val" + sfx, vals.data(), vals.size() * 4, "f32", {K, N}, t);
    if (st.head_logits_full) write_bin(st, "head_logits" + sfx, lb.data(), lb.size(), "f32", {V, N}, t);
    st.n_head++;
}

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto & st = *(dump_state *) ud;
    if (st.head) {
        const bool hm = t->op == GGML_OP_MUL_MAT && is_head_weight(st, t->src[0]);
        if (ask) {
            if (st.list && t->op == GGML_OP_MUL_MAT) fprintf(stderr, "act_dump: node %s src0=%s\n", t->name, t->src[0] ? t->src[0]->name : "-");
            return hm;
        }
        if (hm) dump_head(st, t);
        return true;
    }
    const std::string pre = "blk." + std::to_string(st.layer) + ".";
    bool mm = t->op == GGML_OP_MUL_MAT && t->src[0] && strncmp(t->src[0]->name, pre.c_str(), pre.size()) == 0;
    bool fa = t->op == GGML_OP_FLASH_ATTN_EXT && (ends_with(t->name, "-" + std::to_string(st.layer)) || kv_layer(t->src[1], st.layer));
    if (ask) {
        if (st.list && (t->op == GGML_OP_FLASH_ATTN_EXT || t->op == GGML_OP_SOFT_MAX || t->op == GGML_OP_MUL_MAT)) {
            fprintf(stderr, "act_dump: node %s op=%s src0=%s src1=%s\n", t->name, ggml_op_name(t->op), t->src[0] ? t->src[0]->name : "-",
                    t->src[1] ? t->src[1]->name : "-");
        }
        return mm || fa;
    }
    if (mm) {
        std::string w = t->src[0]->name + pre.size();  // e.g. ffn_up.weight
        w = w.substr(0, w.find('.'));
        dump(st, t->src[1], "act_" + w, t);
        if (st.weights) {
            dump(st, t->src[0], "w_" + w, t);
        }
        st.n_mm++;
    }
    if (fa) {
        dump(st, t->src[0], "fa_q", t);
        dump(st, t->src[1], "fa_k", t);
        dump(st, t->src[2], "fa_v", t);
        st.n_fa++;
    }
    return true;
}

static const char * kDefaultText =
    "The measurement plan states its question, the expected effect against the sample-size error, the checkpoints, and the rules "
    "for stopping on an answer or on futility. Reuse valid evidence first, save incremental results, and stop when the scoped "
    "question is answered. A build path or directory name does not identify the binary it contains. ";

int main(int argc, char ** argv) {
    std::string model, out, prompt_file;
    int n_tok = 512, ngl = 0, threads = 8;
    dump_state st;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "-o") out = next();
        else if (a == "--layer") st.layer = atoi(next());
        else if (a == "-n") n_tok = atoi(next());
        else if (a == "-ngl") ngl = atoi(next());
        else if (a == "-t") threads = atoi(next());
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--weights") st.weights = true;
        else if (a == "--list") st.list = true;
        else if (a == "--head") st.head = true;
        else if (a == "--head-name") { st.head_name = next(); st.head = true; }
        else if (a == "--head-topk") st.head_topk = atoi(next());
        else if (a == "--head-logits-full") st.head_logits_full = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model.empty() || out.empty() || n_tok < 1) {
        fprintf(stderr, "usage: act_dump -m model.gguf -o outdir [--layer L] [-n 512] [-ngl 0] [-t 8] [--prompt-file f] [--weights]\n");
        return 1;
    }
    st.out = out;
    std::filesystem::create_directories(out);
    std::string text = kDefaultText;
    if (!prompt_file.empty()) {
        std::ifstream f(prompt_file);
        std::stringstream ss; ss << f.rdbuf();
        if (!ss.str().empty()) text = ss.str();
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "act_dump: model load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(m);

    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n <= 0) { fprintf(stderr, "act_dump: tokenize failed\n"); return 2; }
    toks.resize(n);
    std::vector<llama_token> batch;  // repeat the text to exactly n_tok tokens (BOS only at the start)
    while ((int) batch.size() < n_tok) {
        for (size_t i = batch.empty() ? 0 : 1; i < toks.size() && (int) batch.size() < n_tok; i++) batch.push_back(toks[i]);
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((n_tok + 255) / 256 * 256);
    cp.n_batch = cp.n_ubatch = (uint32_t) n_tok;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "act_dump: context init failed\n"); return 2; }

    llama_batch lb = llama_batch_init((int32_t) batch.size(), 0, 1);
    for (size_t i = 0; i < batch.size(); i++) {  // --head: logits for every row so the head MUL_MAT sees all N tokens
        lb.token[i] = batch[i]; lb.pos[i] = (llama_pos) i; lb.n_seq_id[i] = 1; lb.seq_id[i][0] = 0;
        lb.logits[i] = st.head || i + 1 == batch.size();
    }
    lb.n_tokens = (int32_t) batch.size();
    const int drc = llama_decode(ctx, lb);
    llama_batch_free(lb);
    if (drc != 0) {
        fprintf(stderr, "act_dump: decode failed\n");
        return 2;
    }
    std::ofstream mf(out + "/manifest.json");
    mf << "{\"tool\":\"act_dump\",\"model\":\"" << model << "\",\"layer\":" << st.layer << ",\"n_tokens\":" << batch.size()
       << ",\"n_gpu_layers\":" << ngl << ",\"head_mode\":" << (st.head ? "true" : "false") << ",\"head_nodes\":" << st.n_head
       << ",\"vocab\":" << llama_vocab_n_tokens(vocab) << ",\"mul_mat_nodes\":" << st.n_mm << ",\"fa_nodes\":" << st.n_fa << ",\"tensors\":[";
    for (size_t i = 0; i < st.manifest.size(); i++) mf << (i ? "," : "") << "\n " << st.manifest[i];
    mf << "\n]}\n";
    mf.close();
    fprintf(stderr, "act_dump: layer %d, %zu tokens, %d mul_mat, %d fa, %zu tensors -> %s\n", st.layer, batch.size(), st.n_mm, st.n_fa,
            st.manifest.size(), out.c_str());
    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    if (st.head) return st.n_head > 0 ? 0 : 3;
    return (st.n_mm > 0 && st.n_fa > 0) ? 0 : 3;
}
