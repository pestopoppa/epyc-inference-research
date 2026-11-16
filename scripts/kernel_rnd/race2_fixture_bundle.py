#!/usr/bin/env python3
"""RVP-RACE-2 fixture bundle: launch geometry + per-kernel gfx90a ISA for the race/deadlock executor.

The RVP-RACE-2 executor (handoffs/active/rocm-kernel-authoring.md, INF48-RACE-2-FIXTURES) takes, per kernel instance,
the disassembled code object plus the launch grid / workgroup size of a correctness-only served-shape run. This module
is the CPU half of that capture:

* ``read_kernel_trace``  folds a rocprofv3 ``--kernel-trace -M`` CSV (mangled names) into unique launch-geometry
  tuples per kernel: grid and workgroup in work-items, workgroup count per axis, LDS and scratch bytes, dispatch count.
* ``kernel_isa``         extracts the gfx90a code objects of a ``libggml-hip.so`` (or a bare ``.co``), reads the kernel
  metadata notes and disassembles the wanted kernels, with the sync-relevant opcode counts the executor models
  (s_barrier, s_waitcnt, ds_*, buffer_*...lds, global atomics) and the distinct opcode set.
* ``geometry_delta``     compares a mutant's geometry with BASELINE's for the same kernels (a mutant edits a barrier,
  never the launch, so any difference means the fixture is not the kernel instance the executor will be checked on).

Kernarg bytes are not captured: rocprofv3 6.2 does not record them. The executor feeds index inputs concretely from the
test-backend-ops case parameters (served_shape case set), so this bundle records the case set and seed instead.
Stdlib + gfx90a_isa_audit only; no GPU access.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx90a_isa_audit as ia  # noqa: E402

SCHEMA = "epyc.race2.fixture_bundle.v1"
SYNC_PATTERNS = {
    "s_barrier": re.compile(r"^s_barrier$"),
    "s_waitcnt": re.compile(r"^s_waitcnt"),
    "ds": re.compile(r"^ds_"),
    "lds_dma": re.compile(r"^buffer_load_\w+$"),   # counted only when the operand list carries `lds`
    "global_atomic": re.compile(r"^(global|flat|buffer)_atomic"),
    "s_sleep_or_sethalt": re.compile(r"^s_(sleep|sethalt)$"),
}


def _wg_count(grid: int, wg: int) -> int:
    return -(-grid // wg) if wg else 0


def kernel_key(name: str) -> str:
    """Join key between a trace and code-object metadata: rocprofv3 6.2 -M names the kernel descriptor symbol
    (``<mangled>.kd``); the amdhsa metadata ``.name`` omits the suffix (inf48-race2-fixtures r1: 0/7 joined)."""
    name = name.strip()
    return name[:-3] if name.endswith(".kd") else name


def read_kernel_trace(path: str | Path, wanted: set[str] | None = None) -> dict[str, list[dict]]:
    """Unique launch-geometry tuples per kernel name (mangled when traced with -M), with dispatch counts."""
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = kernel_key(row.get("Kernel_Name", ""))
            if wanted is not None and name not in wanted:
                continue
            g = tuple(int(row["Grid_Size_" + a]) for a in "XYZ")
            w = tuple(int(row["Workgroup_Size_" + a]) for a in "XYZ")
            key = (g, w, int(row.get("Group_Segment_Size") or 0), int(row.get("Private_Segment_Size") or 0))
            agg[name][key] += 1
    out = {}
    for name, c in sorted(agg.items()):
        rows = []
        for (g, w, lds, priv), n in sorted(c.items()):
            wgs = [_wg_count(a, b) for a, b in zip(g, w)]
            rows.append({"grid": list(g), "workgroup": list(w), "n_workgroups": wgs,
                         "n_workgroups_total": wgs[0] * wgs[1] * wgs[2],
                         "waves_per_wg": -(-(w[0] * w[1] * w[2]) // 64),
                         "lds_bytes": lds, "private_bytes": priv, "dispatches": n})
        out[name] = rows
    return out


def geometry_delta(base: dict[str, list[dict]], mut: dict[str, list[dict]], kernels: list[str]) -> dict:
    """Per kernel: same unique (grid, workgroup, lds) set in BASELINE and mutant? Dispatch counts are not compared."""
    def keyset(rows):
        return sorted((tuple(r["grid"]), tuple(r["workgroup"]), r["lds_bytes"]) for r in rows)
    res = {}
    for k in kernels:
        b, m = base.get(k), mut.get(k)
        res[k] = {"in_baseline": b is not None, "in_mutant": m is not None,
                  "same_geometry": b is not None and m is not None and keyset(b) == keyset(m)}
    return res


def _sync_counts(insts) -> dict:
    c = collections.Counter()
    for i in insts:
        for tag, rx in SYNC_PATTERNS.items():
            if rx.match(i.op) and (tag != "lds_dma" or re.search(r"\blds\b", i.args)):
                c[tag] += 1
    return dict(c)


def kernel_isa(lib: str | Path, wanted: set[str], out_dir: str | Path | None = None) -> dict[str, dict]:
    """Metadata, sync-opcode counts, distinct opcodes and (optionally written) disassembly for each wanted kernel.

    ``wanted`` holds kernel names without the ``.kd`` suffix. A kernel present in several code objects is reported
    from the first one that holds it."""
    found: dict[str, dict] = {}
    for idx, co in enumerate(ia.load_code_objects(Path(lib))):
        todo = wanted - set(found)
        if not todo:
            break
        with tempfile.NamedTemporaryFile(suffix=".co") as tmp:
            tmp.write(co)
            tmp.flush()
            meta = ia.parse_notes_text(ia._run("llvm-readelf", "--notes", tmp.name))
            here = todo & set(meta)
            if not here:
                continue
            text = ia._run("llvm-objdump", "-d", "--no-show-raw-insn", tmp.name)
        dis = ia.split_disassembly(text, here)
        co_sha = hashlib.sha256(co).hexdigest()
        for name in sorted(here):
            insts = dis.get(name, [])
            rec = {"code_object_index": idx, "code_object_sha256": co_sha, "meta": meta[name],
                   "n_insts": len(insts), "sync": _sync_counts(insts),
                   "opcodes": sorted({i.op for i in insts})}
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                p = Path(out_dir) / (hashlib.sha256(name.encode()).hexdigest()[:16] + ".s")
                p.write_text(f"; {name}\n" + "\n".join(f"{i.addr:x}\t{i.op}\t{i.args}" for i in insts) + "\n")
                rec["disassembly"] = str(p)
                rec["disassembly_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            found[name] = rec
    return found


def bundle_entry(fixture_id: str, build: dict, geometry: dict[str, list[dict]], isa: dict[str, dict],
                 base_isa: dict[str, dict], delta: dict) -> dict:
    """One fixture: the changed kernels, their launch tuples, and the structural mutation proof (s_barrier drop)."""
    kernels = sorted((build.get("isa_proof") or {}).get("kernels") or {})
    per = {}
    for k in kernels:
        # _sync_counts omits zero counts: a kernel whose ISA was read but has no s_barrier counts 0, not unknown
        mb = isa[k]["sync"].get("s_barrier", 0) if k in isa else None
        bb = base_isa[k]["sync"].get("s_barrier", 0) if k in base_isa else None
        per[k] = {"launches": geometry.get(k, []), "isa": isa.get(k), "s_barrier_base": bb, "s_barrier_mutant": mb,
                  "barrier_dropped": bb is not None and mb is not None and mb < bb, "geometry": delta.get(k)}
    return {"fixture_id": fixture_id, "family": build.get("family"), "source_sha": build.get("source_sha"),
            "libggml_hip_sha256": build.get("libggml_hip_sha256"), "kernels": per,
            "launched": all(per[k]["launches"] for k in kernels) if kernels else False}
