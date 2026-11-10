#define _GNU_SOURCE
#include "mtmd.h"
#include <time.h>
#include <pthread.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "llama.h"
#include "ggml-backend.h"

static pthread_once_t trace_once = PTHREAD_ONCE_INIT;
static FILE * native_trace;
static void trace_init(void) {
    const char * path = getenv("CENSUS_NATIVE_TRACE");
    if (!path || path[0] != '/') _exit(71);
    int fd = open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0 || !(native_trace = fdopen(fd, "a"))) _exit(71);
    setvbuf(native_trace, NULL, _IONBF, 0);
}
static FILE * trace_stream(void) {
    if (pthread_once(&trace_once, trace_init) != 0) _exit(71);
    return native_trace;
}

/* Private C5 correctness instrument. Never reads tensor data or changes graphs. */
static unsigned long long mm_clock_ns(clockid_t clock) {
    struct timespec t;
    if (clock_gettime(clock, &t) != 0) return 0;
    return (unsigned long long)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
static _Thread_local mtmd_batch * observed_batch;
static _Thread_local mtmd_context * observed_batch_ctx;
static _Thread_local const mtmd_input_chunk * observed_chunk;
static _Thread_local unsigned int observed_chunks;
static _Thread_local mtmd_context * active_mm_ctx;
static _Thread_local unsigned long active_mm_encode;
static unsigned long mm_serial;
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
    flockfile(trace_stream());
    fprintf(trace_stream(), "CENSUS_API_BEGIN id=%lu ctx=%p tokens=%d scope=api_batch seq_ids=",
            active_decode, (void *)ctx, batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (!batch.n_seq_id || !batch.seq_id) {
            fprintf(trace_stream(), "unavailable");
            break;
        }
        for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
            fprintf(trace_stream(), "%d,", batch.seq_id[i][j]);
        }
    }
    fputc('\n', trace_stream());
    funlockfile(trace_stream());
    int32_t rc = next(ctx, batch); /* Exactly one original invocation. */
    flockfile(trace_stream());
    fprintf(trace_stream(), "CENSUS_API_END id=%lu ctx=%p rc=%d\n", active_decode, (void *)ctx, rc);
    funlockfile(trace_stream());
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
    const unsigned long long graph_start_utc = mm_clock_ns(CLOCK_REALTIME);
    const unsigned long long graph_start_mono = mm_clock_ns(CLOCK_MONOTONIC);
    enum ggml_status rc = next(sched, graph); /* No callback, allocation or second execution. */
    unsigned long gid = __atomic_add_fetch(&graph_serial, 1, __ATOMIC_RELAXED);
    const struct llama_model * model = active_ctx ? get_model(active_ctx) : NULL;
    const int count = n_nodes(graph);
    flockfile(trace_stream());
    if (active_mm_ctx && active_mm_encode) {
        fprintf(trace_stream(), "CENSUS_MM_GRAPH gid=%lu encode=%lu mmctx=%p utc_ns=%llu mono_ns=%llu\n",
                gid, active_mm_encode, (void *)active_mm_ctx, mm_clock_ns(CLOCK_REALTIME), mm_clock_ns(CLOCK_MONOTONIC));
    }
    fprintf(trace_stream(), "CENSUS_GRAPH gid=%lu id=%lu ctx=%p model=%p nodes=%d rc=%d\n",
            gid, active_decode, (void *)active_ctx, (const void *)model, count, (int)rc);
    for (int i = 0; i < count; ++i) {
        const struct ggml_tensor * node = node_at(graph, i);
        ggml_backend_t backend = assigned(sched, (struct ggml_tensor *)node);
        fprintf(trace_stream(), "CENSUS_NODE gid=%lu id=%lu ctx=%p model=%p node=%d op=%s backend=%s dst_type=%d ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64,
                gid, active_decode, (void *)active_ctx, (const void *)model, i, op_name(node->op),
                backend ? backend_name(backend) : "NULL", (int)node->type,
                node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
        for (int j = 0; j < 2; ++j) {
            const struct ggml_tensor * src = node->src[j];
            if (src) fprintf(trace_stream(), " src%d_type=%d src%d_ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64,
                             j, (int)src->type, j, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        }
        fputc('\n', trace_stream());
    }
    fprintf(trace_stream(), "CENSUS_GRAPH_WINDOW gid=%lu id=%lu ctx=%p model=%p start_utc_ns=%llu end_utc_ns=%llu start_mono_ns=%llu end_mono_ns=%llu\n",
            gid, active_decode, (void *)active_ctx, (const void *)model,
            graph_start_utc, mm_clock_ns(CLOCK_REALTIME), graph_start_mono, mm_clock_ns(CLOCK_MONOTONIC));
    funlockfile(trace_stream());
    return rc;
}

struct llama_model * llama_model_load_from_file(const char * path, struct llama_model_params params) {
    struct llama_model * (*next)(const char *, struct llama_model_params) =
        symbol("llama_model_load_from_file");
    struct llama_model * model = next(path, params);
    flockfile(trace_stream());
    fprintf(trace_stream(), "CENSUS_MODEL model=%p path=%s\n", (void *)model, path);
    funlockfile(trace_stream());
    return model;
}

struct llama_model * llama_model_load_from_splits(const char ** paths, size_t n_paths,
                                                struct llama_model_params params) {
    struct llama_model * (*next)(const char **, size_t, struct llama_model_params) =
        symbol("llama_model_load_from_splits");
    struct llama_model * model = next(paths, n_paths, params);
    flockfile(trace_stream());
    for (size_t i = 0; i < n_paths; ++i) {
        fprintf(trace_stream(), "CENSUS_MODEL model=%p path=%s\n", (void *)model, paths[i]);
    }
    funlockfile(trace_stream());
    return model;
}

mtmd_context * mtmd_init_from_file(const char * path, const struct llama_model * model,
                                   const struct mtmd_context_params params) {
    mtmd_context * (*next)(const char *, const struct llama_model *, const struct mtmd_context_params) =
        symbol("mtmd_init_from_file");
    mtmd_context * result = next(path, model, params);
    flockfile(trace_stream());
    fprintf(trace_stream(), "CENSUS_MM_INIT mmctx=%p model=%p path=%s\n", (void *)result, (const void *)model, path);
    funlockfile(trace_stream());
    return result;
}

int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    int32_t (*next)(mtmd_context *, const mtmd_input_chunk *) = symbol("mtmd_encode_chunk");
    size_t (*n_tokens)(const mtmd_input_chunk *) = symbol("mtmd_input_chunk_get_n_tokens");
    const char * (*chunk_id)(const mtmd_input_chunk *) = symbol("mtmd_input_chunk_get_id");
    enum mtmd_input_chunk_type (*chunk_type)(const mtmd_input_chunk *) = symbol("mtmd_input_chunk_get_type");
    mtmd_context * previous = active_mm_ctx;
    unsigned long previous_encode = active_mm_encode;
    active_mm_ctx = ctx;
    active_mm_encode = __atomic_add_fetch(&mm_serial, 1, __ATOMIC_RELAXED);
    const char * id = chunk_id(chunk);
    flockfile(trace_stream());
    fprintf(trace_stream(), "CENSUS_MM_BEGIN encode=%lu mmctx=%p type=%d tokens=%zu chunk=%s utc_ns=%llu mono_ns=%llu\n",
            active_mm_encode, (void *)ctx, (int)chunk_type(chunk), n_tokens(chunk), id ? id : "<null>", mm_clock_ns(CLOCK_REALTIME), mm_clock_ns(CLOCK_MONOTONIC));
    funlockfile(trace_stream());
    int32_t rc = next(ctx, chunk);
    flockfile(trace_stream());
    fprintf(trace_stream(), "CENSUS_MM_END encode=%lu mmctx=%p rc=%d utc_ns=%llu mono_ns=%llu\n", active_mm_encode, (void *)ctx, (int)rc, mm_clock_ns(CLOCK_REALTIME), mm_clock_ns(CLOCK_MONOTONIC));
    funlockfile(trace_stream());
    active_mm_ctx = previous;
    active_mm_encode = previous_encode;
    return rc;
}

mtmd_batch * mtmd_batch_init(mtmd_context * ctx) {
    mtmd_batch * (*next)(mtmd_context *) = symbol("mtmd_batch_init");
    mtmd_batch * batch = next(ctx);
    observed_batch = batch;
    observed_batch_ctx = batch ? ctx : NULL;
    observed_chunk = NULL;
    observed_chunks = 0;
    return batch;
}

int32_t mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    int32_t (*next)(mtmd_batch *, const mtmd_input_chunk *) = symbol("mtmd_batch_add_chunk");
    int32_t rc = next(batch, chunk);
    if (batch != observed_batch || rc != 0) {
        observed_batch = NULL; observed_batch_ctx = NULL; observed_chunk = NULL; observed_chunks = 0;
    } else {
        observed_chunks++;
        observed_chunk = observed_chunks == 1 ? chunk : NULL;
    }
    return rc;
}

void mtmd_batch_free(mtmd_batch * batch) {
    void (*next)(mtmd_batch *) = symbol("mtmd_batch_free");
    if (batch == observed_batch) {
        observed_batch = NULL; observed_batch_ctx = NULL; observed_chunk = NULL; observed_chunks = 0;
    }
    next(batch);
}

int32_t mtmd_batch_encode(mtmd_batch * batch) {
    int32_t (*next)(mtmd_batch *) = symbol("mtmd_batch_encode");
    if (batch != observed_batch || !observed_batch_ctx || observed_chunks != 1 || !observed_chunk) {
        flockfile(trace_stream());
        fprintf(trace_stream(), "CENSUS_MM_REFUSAL batch=%p reason=untracked-or-nonsingle-image-batch\n", (void *)batch);
        funlockfile(trace_stream());
        return next(batch); /* Observe-only refusal: never change the native execution. */
    }
    size_t (*n_tokens)(const mtmd_input_chunk *) = symbol("mtmd_input_chunk_get_n_tokens");
    const char * (*chunk_id)(const mtmd_input_chunk *) = symbol("mtmd_input_chunk_get_id");
    enum mtmd_input_chunk_type (*chunk_type)(const mtmd_input_chunk *) = symbol("mtmd_input_chunk_get_type");
    mtmd_context * previous = active_mm_ctx;
    unsigned long previous_encode = active_mm_encode;
    active_mm_ctx = observed_batch_ctx;
    active_mm_encode = __atomic_add_fetch(&mm_serial, 1, __ATOMIC_RELAXED);
    const char * id = chunk_id(observed_chunk);
    flockfile(trace_stream());
    fprintf(trace_stream(), "CENSUS_MM_BEGIN encode=%lu mmctx=%p type=%d tokens=%zu chunk=%s utc_ns=%llu mono_ns=%llu\n",
            active_mm_encode, (void *)active_mm_ctx, (int)chunk_type(observed_chunk), n_tokens(observed_chunk),
            id ? id : "<null>", mm_clock_ns(CLOCK_REALTIME), mm_clock_ns(CLOCK_MONOTONIC));
    funlockfile(trace_stream());
    int32_t rc = next(batch);
    flockfile(trace_stream());
    fprintf(trace_stream(), "CENSUS_MM_END encode=%lu mmctx=%p rc=%d utc_ns=%llu mono_ns=%llu\n",
            active_mm_encode, (void *)active_mm_ctx, (int)rc, mm_clock_ns(CLOCK_REALTIME), mm_clock_ns(CLOCK_MONOTONIC));
    funlockfile(trace_stream());
    active_mm_ctx = previous;
    active_mm_encode = previous_encode;
    observed_batch = NULL; observed_batch_ctx = NULL; observed_chunk = NULL; observed_chunks = 0;
    return rc;
}
