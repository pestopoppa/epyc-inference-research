#!/usr/bin/env python3
"""R21-10: classify differing byte offsets of two ELF files by section."""
import subprocess, sys
from collections import Counter

def sections(path):
    out = subprocess.run(["readelf", "-S", "-W", path], capture_output=True,
                         text=True).stdout
    secs = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        parts = line.split("]", 1)[1].split()
        if len(parts) < 6 or parts[1] in ("NULL", "Type"):
            continue
        try:
            name, off, size = parts[0], int(parts[3], 16), int(parts[4], 16)
        except ValueError:
            continue
        secs.append((off, off + size, name))
    return sorted(secs)

def classify(a, b):
    secs = sections(a)
    counts = Counter()
    da, db = open(a, "rb").read(), open(b, "rb").read()
    if len(da) != len(db):
        print(f"SIZE DIFFERS {len(da)} vs {len(db)}")
    n = min(len(da), len(db))
    CHUNK = 1 << 20
    diffs = []
    for base in range(0, n, CHUNK):
        ca, cb = da[base:base+CHUNK], db[base:base+CHUNK]
        if ca == cb:
            continue
        for i in range(len(ca)):
            if ca[i] != cb[i]:
                diffs.append(base + i)
    for off in diffs:
        name = "<header/unmapped>"
        for lo, hi, nm in secs:
            if lo <= off < hi:
                name = nm
                break
        counts[name] += 1
    total = len(diffs)
    print(f"{a}\n  vs {b}\n  differing bytes: {total}")
    for name, c in counts.most_common():
        print(f"  {name:24s} {c}")

if __name__ == "__main__":
    classify(sys.argv[1], sys.argv[2])
