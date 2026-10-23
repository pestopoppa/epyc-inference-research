#!/usr/bin/env python3
"""Drop the page cache of exactly the given files (posix_fadvise DONTNEED).

What a --no-mmap load needs evicted is the read cache of the GGUFs it is about to re-read; dropping
exactly those takes seconds, where allocation pressure (numa_evict.py) takes minutes. fadvise is
ADVISORY and evicts only CLEAN pages, so this is the fast path only: the caller still runs
numa_evict.py (which forces and verifies the per-node free target) and, after the load, proves the
placement with numastat. Prints one line per file and exits 0 unless a file cannot be opened.

  usage: pagecache_drop.py FILE...
"""
import os
import sys


def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return size
    finally:
        os.close(fd)


def main(argv):
    rc = 0
    for p in argv[1:]:
        try:
            print(f"DROP {drop(p) / 2**30:8.3f} GiB  {p}")
        except OSError as e:
            print(f"CANNOT DROP {p}: {e}", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
