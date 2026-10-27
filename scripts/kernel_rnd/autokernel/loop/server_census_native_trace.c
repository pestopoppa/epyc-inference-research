#define _GNU_SOURCE
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "llama.h"
#include "ggml-backend.h"

/* Private C5 correctness instrument. Never reads tensor data or changes graphs. */
static _Thread_local struct llama_context * active_ctx;
static _Thread_local unsigned long active_decode;
static unsigned long serial;
static unsigned long graph_serial;

static void * symbol(const char * name) {
    void * result = dlsym(RTLD_NEXT, name);
    if (!result) {
        dprintf(STDERR_FILENO, "CENSUS_ABI_REFUSED symbol=%s\n", name);
        _exit(71);
    }
    return result;
}

int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch) {
    int32_t (*next)(struct llama_context *, struct llama_batch) = symbol("llama_decode");
    struct llama_context * previous_ctx = active_ctx;
    unsigned long previous_decode = active_decode;
    active_ctx = ctx;
    active_decode = __atomic_add_fetch(&serial, 1, __ATOMIC_RELAXED);
    flockfile(stderr);
    fprintf(stderr, "CENSUS_API_BEGIN id=%lu ctx=%p tokens=%d scope=api_batch seq_ids=",
            active_decode, (void *)ctx, batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (!batch.n_seq_id || !batch.seq_id) {
            fprintf(stderr, "unavailable");
            break;
        }
        for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
            fprintf(stderr, "%d,", batch.seq_id[i][j]);
        }
    }
    fputc('\n', stderr);
    funlockfile(stderr);
    int32_t rc = next(ctx, batch); /* Exactly one original invocation. */
    flockfile(stderr);
    fprintf(stderr, "CENSUS_API_END id=%lu ctx=%p rc=%d\n", active_decode, (void *)ctx, rc);
    funlockfile(stderr);
    active_ctx = previous_ctx;
    active_decode = previous_decode;
    return rc;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched,
                                                       struct ggml_cgraph * graph) {
    enum ggml_status (*next)(ggml_backend_sched_t, struct ggml_cgraph *) =
        symbol("ggml_backend_sched_graph_compute_async");
    int (*n_nodes)(struct ggml_cgraph *) = symbol("ggml_graph_n_nodes");
    struct ggml_tensor * (*node_at)(struct ggml_cgraph *, int) = symbol("ggml_graph_node");
    const char * (*op_name)(enum ggml_op) = symbol("ggml_op_name");
    ggml_backend_t (*assigned)(ggml_backend_sched_t, struct ggml_tensor *) =
        symbol("ggml_backend_sched_get_tensor_backend");
    const char * (*backend_name)(ggml_backend_t) = symbol("ggml_backend_name");
    const struct llama_model * (*get_model)(const struct llama_context *) = symbol("llama_get_model");
    enum ggml_status rc = next(sched, graph); /* No callback, allocation or second execution. */
    unsigned long gid = __atomic_add_fetch(&graph_serial, 1, __ATOMIC_RELAXED);
    const struct llama_model * model = active_ctx ? get_model(active_ctx) : NULL;
    const int count = n_nodes(graph);
    flockfile(stderr);
    fprintf(stderr, "CENSUS_GRAPH gid=%lu id=%lu ctx=%p model=%p nodes=%d rc=%d\n",
            gid, active_decode, (void *)active_ctx, (const void *)model, count, (int)rc);
    for (int i = 0; i < count; ++i) {
        const struct ggml_tensor * node = node_at(graph, i);
        ggml_backend_t backend = assigned(sched, (struct ggml_tensor *)node);
        fprintf(stderr, "CENSUS_NODE gid=%lu id=%lu ctx=%p model=%p node=%d op=%s backend=%s dst_type=%d ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64,
                gid, active_decode, (void *)active_ctx, (const void *)model, i, op_name(node->op),
                backend ? backend_name(backend) : "NULL", (int)node->type,
                node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
        for (int j = 0; j < 2; ++j) {
            const struct ggml_tensor * src = node->src[j];
            if (src) fprintf(stderr, " src%d_type=%d src%d_ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64,
                             j, (int)src->type, j, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        }
        fputc('\n', stderr);
    }
    funlockfile(stderr);
    return rc;
}

struct llama_model * llama_model_load_from_file(const char * path, struct llama_model_params params) {
    struct llama_model * (*next)(const char *, struct llama_model_params) =
        symbol("llama_model_load_from_file");
    struct llama_model * model = next(path, params);
    flockfile(stderr);
    fprintf(stderr, "CENSUS_MODEL model=%p path=%s\n", (void *)model, path);
    funlockfile(stderr);
    return model;
}

struct llama_model * llama_model_load_from_splits(const char ** paths, size_t n_paths,
                                                struct llama_model_params params) {
    struct llama_model * (*next)(const char **, size_t, struct llama_model_params) =
        symbol("llama_model_load_from_splits");
    struct llama_model * model = next(paths, n_paths, params);
    flockfile(stderr);
    for (size_t i = 0; i < n_paths; ++i) {
        fprintf(stderr, "CENSUS_MODEL model=%p path=%s\n", (void *)model, paths[i]);
    }
    funlockfile(stderr);
    return model;
}
