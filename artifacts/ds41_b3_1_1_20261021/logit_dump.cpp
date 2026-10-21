// DS41-B3.1.1: greedy decode + per-step full-vocab logits hash/top-k dump. Links only libllama.
// modes: tok  <model> <textfile>              -> one line of comma-separated token ids per input line (parse_special, no auto BOS)
//        detok <model> <ids-csv>              -> prints the text
//        run  <model> <idsfile> <outdir> <n_threads> <n_predict> <topk>
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

static void quiet(ggml_log_level, const char *, void *) {}

static uint64_t fnv64(const void * p, size_t n) {
    const unsigned char * b = (const unsigned char *) p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static llama_model * load(const char * path, bool vocab_only) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = vocab_only; mp.use_mmap = true; mp.n_gpu_layers = 0;
    return llama_model_load_from_file(path, mp);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage\n"); return 2; }
    std::string mode = argv[1];
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    if (mode == "tok" || mode == "detok") {
        llama_model * m = load(argv[2], true); if (!m) return 3;
        const llama_vocab * v = llama_model_get_vocab(m);
        if (mode == "tok") {
            std::ifstream f(argv[3]); std::string line;
            while (std::getline(f, line)) {
                // literal "\n" sequences in the file are turned into newlines
                std::string t; for (size_t i = 0; i < line.size(); i++) { if (line[i]=='\\' && i+1<line.size() && line[i+1]=='n') { t += '\n'; i++; } else t += line[i]; }
                std::vector<llama_token> ids(t.size() + 16);
                int n = llama_tokenize(v, t.data(), (int) t.size(), ids.data(), (int) ids.size(), false, true);
                if (n < 0) { fprintf(stderr, "tokenize fail\n"); return 4; }
                for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
                printf("\n");
            }
        } else {
            std::stringstream ss(argv[3]); std::string tk; std::string out;
            while (std::getline(ss, tk, ',')) { char buf[256]; int n = llama_token_to_piece(v, atoi(tk.c_str()), buf, sizeof buf, 0, true); if (n > 0) out.append(buf, n); }
            printf("%s\n", out.c_str());
        }
        return 0;
    }
    if (mode != "run" || argc < 8) return 2;
    const char * mpath = argv[2]; const char * idsfile = argv[3]; std::string outdir = argv[4];
    int nthr = atoi(argv[5]), npred = atoi(argv[6]), topk = atoi(argv[7]);
    llama_model * model = load(mpath, false); if (!model) { fprintf(stderr, "load fail\n"); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.n_threads = nthr; cp.n_threads_batch = nthr;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.kv_unified = false;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16; cp.offload_kqv = false; cp.op_offload = false;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) { fprintf(stderr, "ctx fail\n"); return 3; }

    std::ifstream f(idsfile); std::string line; int pi = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<llama_token> prompt; { std::stringstream ss(line); std::string tk; while (std::getline(ss, tk, ',')) prompt.push_back(atoi(tk.c_str())); }
        llama_memory_clear(llama_get_memory(ctx), true);
        char name[64]; snprintf(name, sizeof name, "p%02d", pi);
        FILE * fl = fopen((outdir + "/" + name + ".logits.bin").c_str(), "wb");
        FILE * fj = fopen((outdir + "/" + name + ".steps.jsonl").c_str(), "w");
        std::vector<llama_token> gen; std::vector<llama_token> cur = prompt; int pos = 0;
        for (int step = 0; step <= npred; step++) {
            llama_batch b = llama_batch_get_one(cur.data(), (int) cur.size());
            // decode of >n_batch tokens is not needed (prompts are short)
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode fail step %d\n", step); return 5; }
            pos += (int) cur.size();
            const float * lg = llama_get_logits_ith(ctx, -1);
            if (!lg) { fprintf(stderr, "no logits\n"); return 5; }
            fwrite(lg, sizeof(float), nv, fl);
            uint64_t h = fnv64(lg, sizeof(float) * (size_t) nv);
            int nnan = 0; for (int i = 0; i < nv; i++) if (std::isnan(lg[i])) nnan++;
            std::vector<int> idx(nv); for (int i = 0; i < nv; i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int c) { return lg[a] > lg[c] || (lg[a] == lg[c] && a < c); });
            int am = idx[0];
            fprintf(fj, "{\"step\":%d,\"pos\":%d,\"fnv64\":\"%016llx\",\"nan\":%d,\"argmax\":%d,\"top\":[", step, pos, (unsigned long long) h, nnan, am);
            for (int k = 0; k < topk; k++) { uint32_t bits; memcpy(&bits, &lg[idx[k]], 4); fprintf(fj, "%s[%d,\"%08x\",%.9g]", k ? "," : "", idx[k], bits, lg[idx[k]]); }
            fprintf(fj, "]}\n");
            gen.push_back(am); cur.assign(1, am);
        }
        fclose(fl); fclose(fj);
        std::string toks; for (size_t i = 0; i < gen.size(); i++) toks += (i ? "," : "") + std::to_string(gen[i]);
        FILE * ft = fopen((outdir + "/" + name + ".tokens.txt").c_str(), "w"); fprintf(ft, "%s\n", toks.c_str()); fclose(ft);
        fprintf(stderr, "done %s prompt_n=%zu gen=%zu\n", name, prompt.size(), gen.size());
        pi++;
    }
    llama_free(ctx); llama_model_free(model);
    return 0;
}
