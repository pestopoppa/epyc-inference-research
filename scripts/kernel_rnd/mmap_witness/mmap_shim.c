/* LD_PRELOAD shim: log every fd-backed mmap/mmap64 (flags, length, fd target) to $MMAP_SHIM_LOG. Observation only. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>

static void logmap(const char *fn, size_t len, int prot, int flags, int fd, long long off, void *ret) {
    const char *path = getenv("MMAP_SHIM_LOG");
    if (!path || fd < 0) return;
    char lnk[64], tgt[PATH_MAX]; ssize_t n;
    snprintf(lnk, sizeof lnk, "/proc/self/fd/%d", fd);
    n = readlink(lnk, tgt, sizeof tgt - 1);
    if (n < 0) n = 0;
    tgt[n] = 0;
    char buf[PATH_MAX + 256];
    int m = snprintf(buf, sizeof buf, "{\"fn\":\"%s\",\"pid\":%d,\"len\":%zu,\"prot\":%d,\"flags\":%d,\"populate\":%d,\"off\":%lld,\"ok\":%d,\"target\":\"%s\"}\n",
                     fn, (int)getpid(), len, prot, flags, (flags & MAP_POPULATE) ? 1 : 0, off, ret != MAP_FAILED, tgt);
    int lf = open(path, O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (lf >= 0) { if (write(lf, buf, m) < 0) {} close(lf); }
}
void *mmap(void *a, size_t l, int p, int f, int fd, off_t o) {
    static void *(*real)(void *, size_t, int, int, int, off_t);
    if (!real) real = dlsym(RTLD_NEXT, "mmap");
    void *r = real(a, l, p, f, fd, o); logmap("mmap", l, p, f, fd, (long long)o, r); return r;
}
void *mmap64(void *a, size_t l, int p, int f, int fd, off_t o) {
    static void *(*real)(void *, size_t, int, int, int, off_t);
    if (!real) real = dlsym(RTLD_NEXT, "mmap64");
    void *r = real(a, l, p, f, fd, o); logmap("mmap64", l, p, f, fd, (long long)o, r); return r;
}
