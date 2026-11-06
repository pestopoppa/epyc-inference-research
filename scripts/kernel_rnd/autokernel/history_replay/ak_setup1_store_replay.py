#!/usr/bin/env python3
"""AK-SETUP-1 replay: how many historical AK builds would a kernel-store-first lookup skip?

Reads every `build-cache/entries/*/intent.json` under the deployments root (read-only),
orders them by intent mtime, and walks them as a store would have seen them: an entry is
a HIT when an earlier entry produced the same content-addressed key.

Two keys per entry:
  candidate store key  -> `controller.build_store_lookup.store_key` (the live key)
  anchor store key     -> same fields minus the patch (what an anchor tree depends on)
plus the key the harness used BEFORE reuse landed (`build_key`, which hashes patch,
paths and deployment identity) as the no-store baseline.

Skipped seconds are ESTIMATES from build-log mtimes (anchor.log - intent.json for the
anchor tree, candidate.log - anchor.log for the candidate tree); entries without logs
use `FALLBACK_TREE_SECONDS` (the 1,012 s the harness's own comment measured).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from ..controller import build_store_lookup as bsl
from .. import schemas

FALLBACK_TREE_SECONDS = 1012.0


def anchor_store_key(contract):
    fields = [f for f in bsl._KEY_FIELDS if f not in ("patch_sha256", "selected_gpu_base_blobs")]
    return schemas.content_hash({"schema": "epyc.autokernel.anchor_store_key.v1",
                                 **{k: contract[k] for k in fields}})


def load(root):
    out = []
    for p in glob.glob(str(Path(root) / "*" / "operations" / "build-cache" / "entries" / "*" / "intent.json")):
        try:
            c = json.loads(Path(p).read_text("utf-8"))["build_contract"]
            ckey, akey = bsl.store_key(c), anchor_store_key(c)
        except (OSError, KeyError, ValueError):
            continue
        d = Path(p).parent
        logs = d / "logs"
        a, k = (logs / "akc-anchor.log"), next(iter(sorted(logs.glob("akc-discovery-*.log"))), None) if logs.is_dir() else None
        t0 = os.stat(p).st_mtime
        a_s = max(os.stat(a).st_mtime - t0, 0.0) if a.is_file() else None
        k_s = max(os.stat(k).st_mtime - os.stat(a).st_mtime, 0.0) if (k and a.is_file()) else None
        out.append({"entry": d.name, "deployment": d.parents[2].name, "t0": t0, "ckey": ckey, "akey": akey,
                    "build_key": c.get("build_key"), "jobs": (c.get("parallelism") or {}).get("jobs"),
                    "anchor_s": a_s, "cand_s": k_s, "has_logs": a_s is not None})
    out.sort(key=lambda r: (r["t0"], r["entry"]))
    return out


def replay(entries):
    seen_c, seen_a, seen_b = set(), set(), set()
    hits_c = hits_a = hits_b = 0
    skipped = 0.0
    for e in entries:
        a_hit, c_hit = e["akey"] in seen_a, e["ckey"] in seen_c
        b_hit = e["build_key"] in seen_b
        hits_a += a_hit; hits_c += c_hit; hits_b += b_hit
        if a_hit:
            skipped += e["anchor_s"] if e["anchor_s"] is not None else FALLBACK_TREE_SECONDS
        if c_hit:
            skipped += e["cand_s"] if e["cand_s"] is not None else FALLBACK_TREE_SECONDS
        seen_a.add(e["akey"]); seen_c.add(e["ckey"]); seen_b.add(e["build_key"])
    n = len(entries)
    return {"schema": "epyc.autokernel.setup1_store_replay.v1", "n_entries": n,
            "anchor_tree_hits": hits_a, "candidate_tree_hits": hits_c, "baseline_build_key_hits": hits_b,
            "anchor_hit_rate": hits_a / n if n else None, "candidate_hit_rate": hits_c / n if n else None,
            "skipped_seconds_estimate": round(skipped, 1), "entries_with_logs": sum(e["has_logs"] for e in entries),
            "jobs_seen": sorted({e["jobs"] for e in entries if e["jobs"] is not None})}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/raid0/llm/autokernel/deployments")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = replay(load(a.root))
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
