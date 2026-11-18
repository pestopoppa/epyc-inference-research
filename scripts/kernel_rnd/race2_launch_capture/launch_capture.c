/* RVP-RACE-2 launch-capture shim (LD_PRELOAD). Records, for every hipLaunchKernel (the path HIP-clang <<<>>> stubs
 * take), the kernel name, grid (in workgroups), workgroup, dynamic LDS and the EXPLICIT kernarg bytes packed at their
 * metadata offsets. Hidden kernargs are runtime-filled and not captured. Consumers: race2_executor (index inputs) and
 * C5-ISA-2 GCN-Gym (hipModuleLaunchKernel replay of the served-shape launches).
 *
 * Build (no GPU needed):  /opt/rocm/llvm/bin/clang -O2 -shared -fPIC -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include \
 *                          -o librace2_capture.so launch_capture.c -ldl -lpthread
 * Env:  RACE2_CAPTURE_ARGTABLE  file from `race2_launch_capture.py argtable` (one line per kernel:
 *                               "<name> <kernarg_segment_size> <n> <off>:<size> ...")      required
 *       RACE2_CAPTURE_OUT       JSONL output path (appended)                                  required
 *       RACE2_CAPTURE_MAX       max records per kernel (default 64; identical launches are still counted once each)
 * A kernel missing from the table is recorded with "kernarg": null (geometry only). Never alters the launch. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>

#define MAXK 4096
#define MAXA 64
typedef struct { char *name; unsigned seg; int n; unsigned off[MAXA], sz[MAXA]; unsigned seen; } kent;
static kent tab[MAXK]; static int ntab = -1; static FILE *out; static unsigned maxrec = 64;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
typedef hipError_t (*launch_fn)(const void *, dim3, dim3, void **, size_t, hipStream_t);
static launch_fn real_launch;

static void load_table(void) {
  ntab = 0;
  const char *t = getenv("RACE2_CAPTURE_ARGTABLE"), *o = getenv("RACE2_CAPTURE_OUT"), *m = getenv("RACE2_CAPTURE_MAX");
  if (m) maxrec = (unsigned)strtoul(m, NULL, 10);
  if (o) out = fopen(o, "a");
  FILE *f = t ? fopen(t, "r") : NULL;
  if (!f) return;
  char *line = NULL; size_t cap = 0;
  while (ntab < MAXK && getline(&line, &cap, f) > 0) {
    kent *e = &tab[ntab]; char *save = NULL, *tok = strtok_r(line, " \n", &save);
    if (!tok) continue;
    e->name = strdup(tok);
    tok = strtok_r(NULL, " \n", &save); e->seg = tok ? (unsigned)strtoul(tok, NULL, 10) : 0;
    tok = strtok_r(NULL, " \n", &save); e->n = tok ? atoi(tok) : 0;
    if (e->n > MAXA) e->n = MAXA;
    for (int i = 0; i < e->n && (tok = strtok_r(NULL, " \n", &save)); i++) sscanf(tok, "%u:%u", &e->off[i], &e->sz[i]);
    ntab++;
  }
  free(line); fclose(f);
}

static kent *find(const char *name) {
  for (int i = 0; i < ntab; i++) if (!strcmp(tab[i].name, name)) return &tab[i];
  return NULL;
}

hipError_t hipLaunchKernel(const void *fn, dim3 grid, dim3 block, void **args, size_t shmem, hipStream_t stream) {
  if (!real_launch) real_launch = (launch_fn)dlsym(RTLD_NEXT, "hipLaunchKernel");
  pthread_mutex_lock(&mu);
  if (ntab < 0) load_table();
  if (out) {
    const char *name = hipKernelNameRefByPtr(fn, stream);
    kent *e = name ? find(name) : NULL;
    if (!e || e->seen < maxrec) {
      if (e) e->seen++;
      fprintf(out, "{\"k\":\"%s\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shmem\":%zu,\"kernarg\":",
              name ? name : "?", grid.x, grid.y, grid.z, block.x, block.y, block.z, shmem);
      if (e && e->seg && e->seg <= 65536) {
        unsigned char *buf = calloc(1, e->seg);
        for (int i = 0; i < e->n; i++)
          if (args && args[i] && e->off[i] + e->sz[i] <= e->seg) memcpy(buf + e->off[i], args[i], e->sz[i]);
        fputc('"', out);
        for (unsigned i = 0; i < e->seg; i++) fprintf(out, "%02x", buf[i]);
        fputc('"', out); free(buf);
      } else fputs("null", out);
      fputs("}\n", out); fflush(out);
    }
  }
  pthread_mutex_unlock(&mu);
  return real_launch(fn, grid, block, args, shmem, stream);
}
