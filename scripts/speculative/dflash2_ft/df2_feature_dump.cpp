// df2_feature_dump -- on-policy target feature dump for DFlash2 drafter training (DF2-FT-1 / DF2-BLOCK16).
//
// For each request it (a) prefills the prompt on the target, (b) either samples a completion on-policy with
// the given sampler (--generate) or teacher-forces a supplied completion (replay), and (c) writes, for every
// token in the capture window [max(0, n_prompt - prefix_window), n_prompt + n_completion), the target's
// layer-INPUT hidden states at the DFlash2 tap ids plus (optionally) the post-norm final hidden state.
//
// No llama.cpp change is needed. The champion already exposes the taps the DFlash speculative path reads:
//   llama_set_embeddings_layer_inp(ctx, lid, true) / llama_get_embeddings_layer_inp(ctx, lid)   (src/llama-ext.h)
// They are exported (C++-mangled) from the C14 store libllama.so. lid is the layer-INPUT id, i.e. the HF
// target_layer_ids + 1 that convert_hf_to_gguf.py writes as dflash.target_layers ([6,20,34,48,62] for the
// z-lab Qwen3.8-27B drafter, HF ids [5,19,33,47,61]). qwen35 captures lid in [0, n_layer); the final hidden
// comes from llama_get_embeddings_ith with pooling NONE (result_norm), not from lid = n_layer.
//
// Output format (df2-feat-v1, read/validated by df2_dump_format.py; all little-endian, row-major):
//   <out>/shard-NNNN/tokens.i32     int32  [N]
//   <out>/shard-NNNN/feats.bf16     bf16   [N, n_taps * n_embd]   (taps concatenated in --taps order)
//   <out>/shard-NNNN/final.bf16     bf16   [N, n_embd]            (only with --final)
//   <out>/shard-NNNN/records.jsonl  one record per request: offsets into the arrays + loss_start
//   <out>/shard-NNNN/manifest.json  written at shard close; `df2_dump_format.py seal` adds sha256s
//
// Input (--requests): one request per line, TAB-separated:  <id> \t <prompt token ids, comma-sep> [\t <completion ids>]
// (df2_dump_format.py prep-requests builds it from JSONL; tokenization/chat template happen there, once).
//
// Build (out-of-tree, against the champion store; nothing in any llama.cpp tree is modified):
//   see CMakeLists.txt in this directory.
#include "llama.h"
#include "llama-ext.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

struct args_t {
    std::string model, requests, out;
    std::vector<int> taps = {6, 20, 34, 48, 62};
    int prefix_window = 2048;
    int max_new = 2048;
    int n_ctx = 65536, n_batch = 512, n_ubatch = 512, ngl = 999, threads = 8;
    bool generate = false, final_hidden = false;
    float temp = 1.0f, top_p = 0.95f, min_p = 0.0f;
    int top_k = 20;
    uint32_t seed = 1234;
    double shard_max_gb = 60.0;
    std::string kv_type = "q8_0";
};

static void die(const char * m) { fprintf(stderr, "df2_feature_dump: %s\n", m); exit(2); }

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
    return v;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;           // NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                // round to nearest even
    return (uint16_t) (u >> 16);
}

struct shard_writer {
    std::string root; int idx = -1; std::string dir;
    FILE * ftok = nullptr, * ffeat = nullptr, * ffin = nullptr, * frec = nullptr;
    int64_t n_tok = 0, n_rec = 0; double bytes = 0;
    const args_t * a = nullptr; int n_embd = 0;

    void open_next() {
        close();
        ++idx; n_tok = 0; n_rec = 0; bytes = 0;
        char b[64]; snprintf(b, sizeof b, "/shard-%04d", idx); dir = root + b;
        mkdir(root.c_str(), 0755); mkdir(dir.c_str(), 0755);
        ftok = fopen((dir + "/tokens.i32").c_str(), "wb");
        ffeat = fopen((dir + "/feats.bf16").c_str(), "wb");
        if (a->final_hidden) ffin = fopen((dir + "/final.bf16").c_str(), "wb");
        frec = fopen((dir + "/records.jsonl").c_str(), "w");
        if (!ftok || !ffeat || !frec || (a->final_hidden && !ffin)) die("cannot open shard files");
    }
    void close() {
        if (!ftok) return;
        fclose(ftok); fclose(ffeat); if (ffin) fclose(ffin); fclose(frec);
        ftok = ffeat = ffin = frec = nullptr;
        FILE * m = fopen((dir + "/manifest.json").c_str(), "w");
        fprintf(m, "{\"format\": \"df2-feat-v1\", \"n_embd\": %d, \"taps\": [", n_embd);
        for (size_t i = 0; i < a->taps.size(); ++i) fprintf(m, "%s%d", i ? ", " : "", a->taps[i]);
        fprintf(m, "], \"has_final\": %s, \"n_tokens\": %" PRId64 ", \"n_records\": %" PRId64
                   ", \"prefix_window\": %d, \"feat_dtype\": \"bf16\", \"model\": \"%s\"}\n",
                a->final_hidden ? "true" : "false", n_tok, n_rec, a->prefix_window, a->model.c_str());
        fclose(m);
    }
};

int main(int argc, char ** argv) {
    args_t a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() -> std::string { if (i + 1 >= argc) die(("missing value for " + k).c_str()); return argv[++i]; };
        if (k == "-m") a.model = nx();
        else if (k == "--requests") a.requests = nx();
        else if (k == "--out") a.out = nx();
        else if (k == "--taps") a.taps = parse_ints(nx());
        else if (k == "--prefix-window") a.prefix_window = std::atoi(nx().c_str());
        else if (k == "--max-new") a.max_new = std::atoi(nx().c_str());
        else if (k == "-c") a.n_ctx = std::atoi(nx().c_str());
        else if (k == "-b") a.n_batch = std::atoi(nx().c_str());
        else if (k == "-ub") a.n_ubatch = std::atoi(nx().c_str());
        else if (k == "-ngl") a.ngl = std::atoi(nx().c_str());
        else if (k == "-t") a.threads = std::atoi(nx().c_str());
        else if (k == "--generate") a.generate = true;
        else if (k == "--final") a.final_hidden = true;
        else if (k == "--temp") a.temp = std::atof(nx().c_str());
        else if (k == "--top-k") a.top_k = std::atoi(nx().c_str());
        else if (k == "--top-p") a.top_p = std::atof(nx().c_str());
        else if (k == "--min-p") a.min_p = std::atof(nx().c_str());
        else if (k == "--seed") a.seed = (uint32_t) std::strtoul(nx().c_str(), nullptr, 10);
        else if (k == "--shard-max-gb") a.shard_max_gb = std::atof(nx().c_str());
        else if (k == "--kv-type") a.kv_type = nx();
        else die(("unknown arg " + k).c_str());
    }
    if (a.model.empty() || a.requests.empty() || a.out.empty()) die("need -m, --requests, --out");

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = a.ngl;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model), n_layer = llama_model_n_layer(model);
    for (int t : a.taps) if (t < 0 || t >= n_layer) die("tap id out of range [0, n_layer) -- qwen35 captures layer inputs only");

    auto cp = llama_context_default_params();
    cp.n_ctx = a.n_ctx; cp.n_batch = a.n_batch; cp.n_ubatch = a.n_ubatch; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = a.threads;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.embeddings = a.final_hidden;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (a.kv_type == "q8_0") { cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; }
    else if (a.kv_type == "f16") { cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16; }
    else die("--kv-type must be q8_0 or f16");
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");

    auto set_capture = [&](bool on) { for (int t : a.taps) llama_set_embeddings_layer_inp(ctx, (uint32_t) t, on); };

    shard_writer w; w.root = a.out; w.a = &a; w.n_embd = n_embd; w.open_next();
    std::vector<uint16_t> rowbuf;
    llama_batch batch = llama_batch_init(a.n_batch, 0, 1);

    std::ifstream in(a.requests);
    if (!in) die("cannot read --requests");
    std::string line;
    int64_t n_req = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; { std::stringstream ss(line); std::string c; while (std::getline(ss, c, '\t')) f.push_back(c); }
        if (f.size() < 2) die("request line needs id<TAB>prompt_ids");
        const std::string id = f[0];
        std::vector<int> prompt = parse_ints(f[1]);
        std::vector<int> given = f.size() > 2 ? parse_ints(f[2]) : std::vector<int>{};
        if (!a.generate && given.empty()) die("replay mode needs completion ids (or pass --generate)");
        const int n_prompt = (int) prompt.size();
        const int ws = std::max(0, n_prompt - a.prefix_window);           // capture window start

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler * smpl = nullptr;
        if (a.generate) {
            smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (a.top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(a.top_k));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(a.top_p, 1));
            if (a.min_p > 0) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(a.min_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(a.temp));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(a.seed + (uint32_t) n_req));
        }

        const int64_t tok_off = w.n_tok;
        std::vector<int> seq = prompt;                                      // grows with the completion
        int pos = 0;
        // decode seq[pos, end) in batches; capture rows whose position >= ws; request logits at the last token
        auto feed = [&](int end, bool want_last_logits) {
            while (pos < end) {
                const int n = std::min(a.n_batch, end - pos);
                const bool cap = pos + n > ws;
                set_capture(cap);
                batch.n_tokens = n;
                for (int j = 0; j < n; ++j) {
                    batch.token[j] = seq[pos + j]; batch.pos[j] = pos + j;
                    batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                    const bool in_win = pos + j >= ws;
                    batch.logits[j] = (a.final_hidden && in_win) || (want_last_logits && pos + j == end - 1);
                }
                if (llama_decode(ctx, batch) != 0) die("llama_decode failed");
                if (cap) {
                    std::vector<const float *> L(a.taps.size());
                    for (size_t k = 0; k < a.taps.size(); ++k) L[k] = llama_get_embeddings_layer_inp(ctx, (uint32_t) a.taps[k]);
                    int out_i = 0;
                    for (int j = 0; j < n; ++j) {
                        const bool out_j = batch.logits[j];
                        if (pos + j >= ws) {
                            rowbuf.resize(a.taps.size() * (size_t) n_embd);
                            for (size_t k = 0; k < a.taps.size(); ++k)
                                for (int e = 0; e < n_embd; ++e)
                                    rowbuf[k * n_embd + e] = f32_to_bf16(L[k][(size_t) j * n_embd + e]);
                            fwrite(rowbuf.data(), 2, rowbuf.size(), w.ffeat);
                            if (a.final_hidden) {
                                const float * h = llama_get_embeddings_ith(ctx, out_i);
                                if (!h) die("final hidden missing (embeddings disabled?)");
                                for (int e = 0; e < n_embd; ++e) rowbuf[e] = f32_to_bf16(h[e]);
                                fwrite(rowbuf.data(), 2, n_embd, w.ffin);
                            }
                            const int32_t t = seq[pos + j]; fwrite(&t, 4, 1, w.ftok);
                            ++w.n_tok; w.bytes += 4 + 2.0 * a.taps.size() * n_embd + (a.final_hidden ? 2.0 * n_embd : 0);
                        }
                        if (out_j) ++out_i;
                    }
                }
                pos += n;
            }
        };

        feed(n_prompt, a.generate);
        int n_comp = 0;
        if (a.generate) {
            for (; n_comp < a.max_new && (int) seq.size() < a.n_ctx - 1; ++n_comp) {
                // the logits row for the last fed token is the last output of the previous decode
                const llama_token t = llama_sampler_sample(smpl, ctx, -1);
                seq.push_back(t);
                feed((int) seq.size(), true);      // feeding t also captures its features
                if (llama_vocab_is_eog(vocab, t)) { ++n_comp; break; }
            }
            llama_sampler_free(smpl);
        } else {
            seq.insert(seq.end(), given.begin(), given.end());
            n_comp = (int) given.size();
            feed((int) seq.size(), false);
        }

        const int64_t n_rec_tok = w.n_tok - tok_off;
        fprintf(w.frec, "{\"rec\": %" PRId64 ", \"id\": \"%s\", \"tok_off\": %" PRId64 ", \"n_tok\": %" PRId64
                        ", \"loss_start\": %d, \"n_prompt\": %d, \"n_completion\": %d, \"mode\": \"%s\", "
                        "\"sampler\": {\"temp\": %g, \"top_k\": %d, \"top_p\": %g, \"min_p\": %g, \"seed\": %u}}\n",
                w.n_rec, id.c_str(), tok_off, n_rec_tok, n_prompt - ws, n_prompt, n_comp,
                a.generate ? "generate" : "replay", a.temp, a.top_k, a.top_p, a.min_p, a.seed + (uint32_t) n_req);
        fflush(w.frec);
        ++w.n_rec; ++n_req;
        if (w.bytes > a.shard_max_gb * 1e9) w.open_next();
    }
    w.close();
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "df2_feature_dump: %" PRId64 " requests written under %s\n", n_req, a.out.c_str());
    return 0;
}
