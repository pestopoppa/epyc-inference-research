// test-akx-qdec.cpp: [AKX-ALL-31.QDEC] native producer for the exhaustive code-space dequant check.
//
// Usage:
//   test-akx-qdec decode <type> <blocks.bin> <out-prefix> [backend (default CPU)] [onehot (default 127)] [rows-per-graph (default 4096)] [ncols (default 256)]
//     ncols < 256 (AK-Q6K-QDEC-ARTEFACT row overlay): only the mul_mat leg runs, as leg "mul_mat_n<ncols>", with at most <ncols>
//     activation columns per MUL_MAT call (a divisor of 256), so a column-threshold requantising path cannot engage.
//   test-akx-qdec random <type> <n-blocks> <seed-u64> <out.bin> [tensor-blocks (default 64)]
//
// decode: <blocks.bin> is a raw concatenation of ggml blocks of <type> (a ggml type name such as q4_K, iq2_xxs,
// mxfp4), written by the research-repo loop/quant_codespace.py enumerator; its size must be a whole number of
// 256-element rows. Three legs, each writing raw little-endian float32 in block/element order:
//   to_float   <out-prefix>.to_float.f32             ggml_get_type_traits(type)->to_float (the CPU reference path)
//   get_rows   <out-prefix>.get_rows.<dev>.f32       GGML_OP_GET_ROWS graph on <backend> (bit-exact leg)
//   mul_mat    <out-prefix>.mul_mat.<dev>.f32        GGML_OP_MUL_MAT(W, onehot * I) on <backend>, transposed back to
//                                                    element order; the backend's served matmul decode path
//                                                    (MMQ / vec_dot), judged against a bound, not bit-exact
// A leg the backend does not support (ggml_backend_supports_op false, e.g. HIP GET_ROWS for K/IQ types at C12)
// is reported as status "unsupported" and writes no file: it never silently falls back to another device.
//
// random: emulates tests/test-backend-ops.cpp init_tensor_uniform (deterministic path) for one quant type:
// uniform [-1,1] floats from suite_seed_mix, quantized by ggml_quantize_chunk with one block per row and the
// dummy all-ones importance matrix (dropped when optional and the tensor's first value > 0, as in the suite),
// in tensors of <tensor-blocks> blocks whose seeds are mix(seed ^ mix(t)). The blocks are written raw, so the
// host can report exactly how much of the code space the random suite reaches.
//
// Every leg prints one `AK_QDEC_V1 {json}` receipt line. Judging lives in the research repo
// (scripts/kernel_rnd/autokernel/loop/quant_codespace.py judge). Exit 0 when every requested leg produced a
// receipt (unsupported counts as a receipt); nonzero on bad input or a failed computation.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int64_t QDEC_ROW = 256;   // elements per row; QK=32 types carry 8 blocks per row

static bool same_name(const char * a, const char * b) {
    for (; *a && *b; ++a, ++b) {
        if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
            return false;
        }
    }
    return *a == *b;
}

static ggml_type type_by_name(const char * name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = (ggml_type) t;
        const char * tn = ggml_type_name(type);
        if (tn != nullptr && ggml_blck_size(type) > 0 && same_name(tn, name)) {
            return type;
        }
    }
    return GGML_TYPE_COUNT;
}

static bool read_file(const char * path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    out.clear();
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) {
        out.insert(out.end(), buf, buf + n);
    }
    const bool ok = ferror(f) == 0;
    fclose(f);
    return ok;
}

static bool write_file(const std::string & path, const void * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const bool ok = fwrite(data, 1, n, f) == n;
    return fclose(f) == 0 && ok;
}

static void receipt(const char * leg, const char * dev, const char * type, const char * status,
                    int64_t nrows, const std::string & path, const std::string & detail) {
    std::string s = std::string("AK_QDEC_V1 {\"schema\": \"epyc.autokernel.quant_codespace.receipt.v1\", \"leg\": \"") +
                    leg + "\", \"backend\": \"" + dev + "\", \"type\": \"" + type + "\", \"status\": \"" + status +
                    "\", \"rows\": " + std::to_string(nrows) + ", \"elements\": " + std::to_string(nrows * QDEC_ROW) +
                    ", \"out\": \"" + path + "\", \"detail\": \"" + detail + "\"}";
    puts(s.c_str());
    fflush(stdout);
}

// one graph per chunk of rows (and, for MUL_MAT, per group of <ncols> activation columns); op = 0 GET_ROWS,
// 1 MUL_MAT(W, onehot*I). ncols < QDEC_ROW keeps every MUL_MAT call's activation at <= ncols columns (iqk routes a type to a
// weight-requantising path above a column threshold, e.g. Q6_K -> Q8_0_R8 at >= 64; AK-Q6K-QDEC-ARTEFACT).
// returns 1 ok, 0 unsupported, -1 failure
static int run_graph(ggml_backend_t be, ggml_type type, const uint8_t * blocks, int64_t nrows, int64_t chunk,
                     int op, float onehot, std::vector<float> & out, int64_t ncols = QDEC_ROW) {
    const size_t row_bytes = ggml_row_size(type, QDEC_ROW);
    out.assign((size_t) (nrows * QDEC_ROW), 0.0f);
    std::vector<float> ident;
    if (op == 1) {
        ident.assign((size_t) (QDEC_ROW * QDEC_ROW), 0.0f);
        for (int64_t k = 0; k < QDEC_ROW; ++k) {
            ident[(size_t) (k * QDEC_ROW + k)] = onehot;
        }
    }
    const int64_t C = op == 1 ? ncols : QDEC_ROW;   // columns per MUL_MAT graph (GET_ROWS: one group)
    std::vector<float> res;
    std::vector<int32_t> idx;
    for (int64_t r0 = 0; r0 < nrows; r0 += chunk) {
        const int64_t R = nrows - r0 < chunk ? nrows - r0 : chunk;
        for (int64_t j0 = 0; j0 < (op == 1 ? QDEC_ROW : 1); j0 += C) {
            ggml_init_params ip = { /*mem_size*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, /*no_alloc*/ true };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return -1;
            }
            ggml_tensor * W = ggml_new_tensor_2d(ctx, type, QDEC_ROW, R);
            ggml_tensor * B = nullptr;
            ggml_tensor * Y = nullptr;
            if (op == 0) {
                B = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, R);
                Y = ggml_get_rows(ctx, W, B);
            } else {
                B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QDEC_ROW, C);
                Y = ggml_mul_mat(ctx, W, B);   // Y[j*R + i] = sum_k W[i][k] * B[j][k] = onehot * W[i][j0 + j]
            }
            ggml_set_output(Y);
            if (!ggml_backend_supports_op(be, Y)) {
                ggml_free(ctx);
                return 0;
            }
            ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, Y);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            if (!buf) {
                ggml_free(ctx);
                return -1;
            }
            ggml_backend_tensor_set(W, blocks + (size_t) r0 * row_bytes, 0, (size_t) R * row_bytes);
            if (op == 0) {
                idx.resize((size_t) R);
                for (int64_t i = 0; i < R; ++i) {
                    idx[(size_t) i] = (int32_t) i;
                }
                ggml_backend_tensor_set(B, idx.data(), 0, idx.size() * sizeof(int32_t));
            } else {
                ggml_backend_tensor_set(B, ident.data() + (size_t) (j0 * QDEC_ROW), 0, (size_t) (C * QDEC_ROW) * sizeof(float));
            }
            const bool ok = ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS;
            if (ok) {
                res.resize((size_t) (R * (op == 1 ? C : QDEC_ROW)));
                ggml_backend_tensor_get(Y, res.data(), 0, res.size() * sizeof(float));
                float * dst = out.data() + (size_t) (r0 * QDEC_ROW);
                if (op == 0) {
                    memcpy(dst, res.data(), res.size() * sizeof(float));
                } else {
                    for (int64_t j = 0; j < C; ++j) {
                        for (int64_t i = 0; i < R; ++i) {
                            dst[(size_t) (i * QDEC_ROW + j0 + j)] = res[(size_t) (j * R + i)];
                        }
                    }
                }
            }
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            if (!ok) {
                return -1;
            }
        }
    }
    return 1;
}

static int cmd_decode(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s decode <type> <blocks.bin> <out-prefix> [backend] [onehot] [rows-per-graph] [ncols]\n", argv[0]);
        return 2;
    }
    const ggml_type type = type_by_name(argv[2]);
    if (type == GGML_TYPE_COUNT || !ggml_is_quantized(type)) {
        fprintf(stderr, "unknown or non-quantized type %s\n", argv[2]);
        return 2;
    }
    const char * tname = ggml_type_name(type);
    std::vector<uint8_t> blocks;
    if (!read_file(argv[3], blocks)) {
        fprintf(stderr, "cannot read %s\n", argv[3]);
        return 2;
    }
    const std::string prefix = argv[4];
    const char * want = argc > 5 ? argv[5] : "CPU";
    const float onehot = argc > 6 ? (float) atof(argv[6]) : 127.0f;
    const int64_t chunk = argc > 7 ? atoll(argv[7]) : 4096;
    const int64_t ncols = argc > 8 ? atoll(argv[8]) : QDEC_ROW;
    if (ncols <= 0 || ncols > QDEC_ROW || QDEC_ROW % ncols != 0) {
        fprintf(stderr, "ncols %lld must divide %lld\n", (long long) ncols, (long long) QDEC_ROW);
        return 2;
    }
    const size_t row_bytes = ggml_row_size(type, QDEC_ROW);
    if (blocks.empty() || blocks.size() % row_bytes != 0 || chunk <= 0) {
        fprintf(stderr, "%s: %zu bytes is not a whole number of %zu-byte rows\n", argv[3], blocks.size(), row_bytes);
        return 2;
    }
    const int64_t nrows = (int64_t) (blocks.size() / row_bytes);

    // tiny context first, so any one-time ggml initialisation has happened before to_float runs
    {
        ggml_init_params ip = { 1024, nullptr, true };
        ggml_context * c0 = ggml_init(ip);
        ggml_free(c0);
    }

    // leg 1: CPU type-traits to_float
    {
        const ggml_type_traits * tr = ggml_get_type_traits(type);
        const std::string path = prefix + ".to_float.f32";
        if (tr == nullptr || tr->to_float == nullptr) {
            receipt("to_float", "CPU", tname, "unsupported", nrows, "", "no to_float");
        } else {
            std::vector<float> y((size_t) (nrows * QDEC_ROW));
            tr->to_float(blocks.data(), y.data(), nrows * QDEC_ROW);
            if (!write_file(path, y.data(), y.size() * sizeof(float))) {
                fprintf(stderr, "cannot write %s\n", path.c_str());
                return 4;
            }
            receipt("to_float", "CPU", tname, "ok", nrows, path, "ggml_get_type_traits()->to_float");
        }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(want);
    if (!dev) {
        fprintf(stderr, "backend %s not found\n", want);
        return 3;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) {
        fprintf(stderr, "backend %s init failed\n", want);
        return 3;
    }
    const char * dname = ggml_backend_dev_name(dev);
    int rc = 0;
    char mm_leg[32];
    if (ncols == QDEC_ROW) {
        snprintf(mm_leg, sizeof mm_leg, "mul_mat");
    } else {
        snprintf(mm_leg, sizeof mm_leg, "mul_mat_n%d", (int) ncols);
    }
    const char * legs[2] = { "get_rows", mm_leg };
    for (int op = 0; op < 2; ++op) {
        if (op == 0 && ncols != QDEC_ROW) {
            continue;   // the narrow run is a MUL_MAT-only leg; GET_ROWS is judged by the default run
        }
        std::vector<float> y;
        const int st = run_graph(be, type, blocks.data(), nrows, chunk, op, onehot, y, ncols);
        const std::string path = prefix + "." + legs[op] + "." + dname + ".f32";
        char detail[96];
        if (op == 0) {
            snprintf(detail, sizeof detail, "GGML_OP_GET_ROWS");
        } else {
            snprintf(detail, sizeof detail, "GGML_OP_MUL_MAT onehot=%.9g ncols=%d", (double) onehot, (int) ncols);
        }
        if (st == 0) {
            receipt(legs[op], dname, tname, "unsupported", nrows, "", "ggml_backend_supports_op false");
        } else if (st < 0) {
            receipt(legs[op], dname, tname, "failed", nrows, "", "graph alloc or compute failed");
            rc = 4;
        } else if (!write_file(path, y.data(), y.size() * sizeof(float))) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            rc = 4;
        } else {
            receipt(legs[op], dname, tname, "ok", nrows, path, detail);
        }
    }
    ggml_backend_free(be);
    return rc;
}

// identical to tests/test-backend-ops.cpp suite_seed_mix at C12
static uint64_t qdec_seed_mix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static int cmd_random(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s random <type> <n-blocks> <seed-u64> <out.bin> [tensor-blocks]\n", argv[0]);
        return 2;
    }
    const ggml_type type = type_by_name(argv[2]);
    if (type == GGML_TYPE_COUNT || !ggml_is_quantized(type)) {
        fprintf(stderr, "unknown or non-quantized type %s\n", argv[2]);
        return 2;
    }
    const int64_t nblocks = atoll(argv[3]);
    const uint64_t seed = strtoull(argv[4], nullptr, 0);
    const int64_t tblocks = argc > 6 ? atoll(argv[6]) : 64;
    if (nblocks <= 0 || tblocks <= 0) {
        fprintf(stderr, "n-blocks and tensor-blocks must be positive\n");
        return 2;
    }
    const int64_t blck = ggml_blck_size(type);
    const size_t bsize = ggml_type_size(type);
    std::vector<uint8_t> out((size_t) nblocks * bsize);
    std::vector<float> data;
    std::vector<float> imatrix((size_t) blck, 1.0f);
    for (int64_t t = 0, b0 = 0; b0 < nblocks; ++t, b0 += tblocks) {
        const int64_t nb = nblocks - b0 < tblocks ? nblocks - b0 : tblocks;
        const uint64_t tseed = qdec_seed_mix(seed ^ qdec_seed_mix((uint64_t) t));
        data.resize((size_t) (nb * blck));
        for (size_t i = 0; i < data.size(); ++i) {
            const uint64_t bits = qdec_seed_mix(tseed ^ qdec_seed_mix((uint64_t) i));
            const double unit = (double) (bits >> 11) * 0x1.0p-53;
            data[i] = (float) (-1.0 + 2.0 * unit);
        }
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(type) && data[0] > 0.0f) {
            im = nullptr;
        }
        ggml_quantize_chunk(type, data.data(), out.data() + (size_t) b0 * bsize, 0, nb, blck, im);
    }
    ggml_quantize_free();
    if (!write_file(argv[5], out.data(), out.size())) {
        fprintf(stderr, "cannot write %s\n", argv[5]);
        return 4;
    }
    char detail[128];
    snprintf(detail, sizeof detail, "seed=%" PRIu64 " tensor_blocks=%" PRId64 " blocks=%" PRId64, seed, tblocks, nblocks);
    receipt("random", "CPU", ggml_type_name(type), "ok", (nblocks * blck) / QDEC_ROW, argv[5], detail);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "decode") == 0) {
        return cmd_decode(argc, argv);
    }
    if (argc >= 2 && strcmp(argv[1], "random") == 0) {
        return cmd_random(argc, argv);
    }
    fprintf(stderr, "usage: %s decode|random ...\n", argc > 0 ? argv[0] : "test-akx-qdec");
    return 2;
}
