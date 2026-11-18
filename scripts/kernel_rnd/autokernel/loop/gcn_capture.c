/* C5-ISA-2 GCN-Gym replay-capture shim (LD_PRELOAD). Extends the RVP-RACE-2 launch-capture shim
 * (scripts/kernel_rnd/race2_launch_capture/launch_capture.c, which it does not modify) with what a hipModuleLaunchKernel
 * replay needs beyond geometry and kernarg bytes:
 *   - every 8-byte explicit arg that resolves through hipMemGetAddressRange is recorded as (kernarg offset, allocation,
 *     delta into the allocation), so the replay can re-point it at its own buffer;
 *   - for the first GCN_CAPTURE_DUMPS launches of each (kernel, grid, block, lds) key, every such allocation is copied to
 *     disk immediately before (".pre") and after (".post") the launch (stream synchronised around it): the replay's
 *     inputs and the ORIGINAL kernel's in-situ output;
 *   - the next GCN_CAPTURE_TIMED launches of the key are timed in situ with a hipEvent pair on the launch stream
 *     (record, launch, record, synchronize) -- the same per-launch measurement gcn_replay_driver.cpp makes.
 * Only kernels whose (mangled) name matches GCN_CAPTURE_MATCH (POSIX ERE) are touched; every other launch, and every launch
 * on a stream that is being graph-captured, passes straight through. The launch itself is never altered.
 *
 * Build (CPU only, inside the row):  /opt/rocm/llvm/bin/clang -O2 -shared -fPIC -D__HIP_PLATFORM_AMD__ \
 *     -I/opt/rocm/include -o libgcn_capture.so gcn_capture.c -L/opt/rocm/lib -lamdhip64 -ldl -lpthread
 * Env:  GCN_CAPTURE_ARGTABLE  arg table (race2_launch_capture.py argtable format)                       required
 *       GCN_CAPTURE_OUT       JSONL record path (appended)                                              required
 *       GCN_CAPTURE_DIR       directory for the .pre/.post blobs                                        required
 *       GCN_CAPTURE_MATCH     ERE on the mangled kernel name                                            required
 *       GCN_CAPTURE_DUMPS     dumped launches per key (default 2)
 *       GCN_CAPTURE_TIMED     timed launches per key (default 200)
 *       GCN_CAPTURE_MAXBYTES  largest allocation dumped (default 1 GiB; a larger one marks the record "dump_skipped") */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <regex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>

#define MAXK 4096
#define MAXA 64
#define MAXKEYS 4096
typedef struct { char *name; unsigned seg; int n; unsigned off[MAXA], sz[MAXA]; } kent;
typedef struct { const char *name; unsigned g[3], b[3]; size_t sh; unsigned dumps, timed; } keyent;
static kent tab[MAXK]; static int ntab = -1; static keyent keys[MAXKEYS]; static int nkeys;
static FILE *out; static const char *dir; static regex_t re; static int have_re;
static unsigned maxdumps = 2, maxtimed = 200; static size_t maxbytes = (size_t)1 << 30; static unsigned long recid;
static hipEvent_t ev0, ev1; static int have_ev;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
typedef hipError_t (*launch_fn)(const void *, dim3, dim3, void **, size_t, hipStream_t);
static launch_fn real_launch;

static void init(void) {
  ntab = 0;
  const char *t = getenv("GCN_CAPTURE_ARGTABLE"), *o = getenv("GCN_CAPTURE_OUT"), *m = getenv("GCN_CAPTURE_MATCH");
  const char *d = getenv("GCN_CAPTURE_DUMPS"), *ti = getenv("GCN_CAPTURE_TIMED"), *mb = getenv("GCN_CAPTURE_MAXBYTES");
  dir = getenv("GCN_CAPTURE_DIR");
  if (d) maxdumps = (unsigned)strtoul(d, NULL, 10);
  if (ti) maxtimed = (unsigned)strtoul(ti, NULL, 10);
  if (mb) maxbytes = (size_t)strtoull(mb, NULL, 10);
  if (m && !regcomp(&re, m, REG_EXTENDED | REG_NOSUB)) have_re = 1;
  if (o && dir && have_re) out = fopen(o, "a");
  FILE *f = t ? fopen(t, "r") : NULL;
  if (!f) { out = NULL; return; }
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

static keyent *key(const char *name, dim3 g, dim3 b, size_t sh) {
  for (int i = 0; i < nkeys; i++) {
    keyent *k = &keys[i];
    if (k->sh == sh && k->g[0] == g.x && k->g[1] == g.y && k->g[2] == g.z && k->b[0] == b.x && k->b[1] == b.y &&
        k->b[2] == b.z && !strcmp(k->name, name)) return k;
  }
  if (nkeys >= MAXKEYS) return NULL;
  keyent *k = &keys[nkeys++];
  k->name = strdup(name); k->g[0] = g.x; k->g[1] = g.y; k->g[2] = g.z; k->b[0] = b.x; k->b[1] = b.y; k->b[2] = b.z;
  k->sh = sh; k->dumps = k->timed = 0;
  return k;
}

static void head(const char *name, dim3 g, dim3 b, size_t sh) {
  fprintf(out, "{\"id\":%lu,\"k\":\"%s\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shmem\":%zu", recid, name, g.x, g.y,
          g.z, b.x, b.y, b.z, sh);
}

/* returns 0 on success; writes "<dir>/L<id>.a<j>.<tag>" */
static int dump(void *base, size_t sz, unsigned long id, int j, const char *tag) {
  char path[4096]; snprintf(path, sizeof path, "%s/L%lu.a%d.%s", dir, id, j, tag);
  void *h = malloc(sz); if (!h) return -1;
  int rc = hipMemcpy(h, base, sz, hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
  FILE *f = rc ? NULL : fopen(path, "wb");
  if (f) { if (fwrite(h, 1, sz, f) != sz) rc = -1; fclose(f); } else rc = -1;
  free(h); return rc;
}

static hipError_t dumped_launch(const void *fn, dim3 g, dim3 b, void **args, size_t sh, hipStream_t s, const char *name,
                                kent *e) {
  void *bases[MAXA]; size_t sizes[MAXA]; int na = 0; int pa[MAXA]; unsigned long long pd[MAXA]; unsigned po[MAXA]; int np = 0;
  for (int i = 0; i < e->n; i++) {
    if (e->sz[i] != 8 || !args || !args[i]) continue;
    void *p = *(void **)args[i], *base = NULL; size_t sz = 0;
    if (!p || hipMemGetAddressRange(&base, &sz, p) != hipSuccess || !base) { (void)hipGetLastError(); continue; }
    int j = 0; while (j < na && bases[j] != base) j++;
    if (j == na) { bases[na] = base; sizes[na] = sz; na++; }
    pa[np] = j; pd[np] = (unsigned long long)((char *)p - (char *)base); po[np] = e->off[i]; np++;
  }
  int skip = 0; for (int j = 0; j < na; j++) if (sizes[j] > maxbytes) skip = 1;
  unsigned long id = recid++;
  int ok = 1;
  if (!skip) { (void)hipStreamSynchronize(s); for (int j = 0; j < na; j++) if (dump(bases[j], sizes[j], id, j, "pre")) ok = 0; }
  hipError_t r = real_launch(fn, g, b, args, sh, s);
  if (!skip) { (void)hipStreamSynchronize(s); for (int j = 0; j < na; j++) if (dump(bases[j], sizes[j], id, j, "post")) ok = 0; }
  unsigned long save = recid; recid = id; head(name, g, b, sh); recid = save;
  fputs(",\"kernarg\":\"", out);
  unsigned char *buf = calloc(1, e->seg ? e->seg : 1);
  for (int i = 0; i < e->n; i++)
    if (args && args[i] && e->off[i] + e->sz[i] <= e->seg) memcpy(buf + e->off[i], args[i], e->sz[i]);
  for (unsigned i = 0; i < e->seg; i++) fprintf(out, "%02x", buf[i]);
  free(buf);
  fprintf(out, "\",\"dumped\":%s,\"dump_skipped\":%s,\"launch_rc\":%d,\"allocs\":[", ok && !skip ? "true" : "false",
          skip ? "true" : "false", (int)r);
  for (int j = 0; j < na; j++) fprintf(out, "%s{\"size\":%zu,\"pre\":\"L%lu.a%d.pre\",\"post\":\"L%lu.a%d.post\"}",
                                       j ? "," : "", sizes[j], id, j, id, j);
  fputs("],\"ptrs\":[", out);
  for (int i = 0; i < np; i++) fprintf(out, "%s[%u,%d,%llu]", i ? "," : "", po[i], pa[i], pd[i]);
  fputs("]}\n", out); fflush(out);
  return r;
}

hipError_t hipLaunchKernel(const void *fn, dim3 grid, dim3 block, void **args, size_t shmem, hipStream_t stream) {
  if (!real_launch) real_launch = (launch_fn)dlsym(RTLD_NEXT, "hipLaunchKernel");
  pthread_mutex_lock(&mu);
  if (ntab < 0) init();
  const char *name = out ? hipKernelNameRefByPtr(fn, stream) : NULL;
  hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
  if (name && (hipStreamIsCapturing(stream, &cs) != hipSuccess || cs != hipStreamCaptureStatusNone)) name = NULL;
  kent *e = (name && !regexec(&re, name, 0, NULL, 0)) ? find(name) : NULL;
  keyent *k = e ? key(name, grid, block, shmem) : NULL;
  hipError_t r;
  if (k && k->dumps < maxdumps) {
    k->dumps++;
    r = dumped_launch(fn, grid, block, args, shmem, stream, name, e);
  } else if (k && k->timed < maxtimed) {
    k->timed++;
    if (!have_ev) have_ev = hipEventCreate(&ev0) == hipSuccess && hipEventCreate(&ev1) == hipSuccess;
    float ms = -1.f;
    (void)hipEventRecord(ev0, stream);
    r = real_launch(fn, grid, block, args, shmem, stream);
    (void)hipEventRecord(ev1, stream);
    if (hipEventSynchronize(ev1) != hipSuccess || hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) ms = -1.f;
    head(name, grid, block, shmem); recid++;
    fprintf(out, ",\"timed\":true,\"ms\":%.6f,\"launch_rc\":%d}\n", ms, (int)r); fflush(out);
  } else {
    r = real_launch(fn, grid, block, args, shmem, stream);
  }
  pthread_mutex_unlock(&mu);
  return r;
}
