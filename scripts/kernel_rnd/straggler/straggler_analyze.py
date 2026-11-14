#!/usr/bin/env python3
"""Straggler attribution for C11 STRAGGLER-PROF thread traces (GGML_THREAD_TRACE=1, record GTT2).

Branch llama.cpp-experimental-c11-straggler-prof-20261114 writes, per traced graph eval:
  u32 magic "GTT2", u32 n_threads, u32 n_rec, u32 batch, u64 eval_idx, u64 t0_ns, i32 cpu[n_threads],
  n_rec x {i32 node_idx, i32 op, i16 kind, i16 n_extra, i32 dst_ne[4], char opname[28], char name[64]},
  n_threads x n_rec x u32[start, end, arrive, leave]  (ns from t0).
kind: 0 node, 1 fused group (n_extra extra nodes), 2 solo run (thread 0 only).

Per node record: straggler wait = max(thread end) - mean(thread end)  [ns, wall];
thread-wait = sum over threads of (max end - end) = n_threads * straggler wait  [thread-ns].
Reports per op type, per node name (layer index stripped), per category, the habitually-last
threads and their CPU / core / L3 (CCD) / NUMA node / SMT position, and the LOCATED/DIFFUSE
verdict: LOCATED iff the top-3 op types hold >= 60 % of the total straggler wait.

Pure stdlib (no numpy on the host python).  Usage:
  straggler_analyze.py TRACE.bin [TRACE.bin ...] [--topology lscpu-p.csv] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

MAGIC = 0x32545447
HDR = struct.Struct("<4I2Q")
NODE = struct.Struct("<iihh4i28s64s")
LOCATED_SHARE = 0.60
FEW_ROWS_PER_THREAD = 32

GDN_RE = re.compile(r"ssm|gdn|delta|conv|linear_attn|recurrent|beta|alpha", re.I)
GDN_OPS = {"SSM_CONV", "SSM_SCAN", "GATED_DELTA_NET", "RWKV_WKV6", "RWKV_WKV7", "GATED_LINEAR_ATTN", "SOLVE_TRI",
           "CUMSUM", "TRI"}
ATTN_OPS = {"FLASH_ATTN_EXT", "SOFT_MAX", "ROPE", "SET_ROWS"}
ATTN_RE = re.compile(r"attn|\bkq|kqv|\bq_|\bk_|\bv_|Qcur|Kcur|Vcur|cache_[kv]", re.I)
ROUTE_RE = re.compile(r"ffn_moe_(probs|logits|argsort|topk|weights|weights_norm|weights_sum|weights_scaled|group)|"
                      r"moe_(probs|argsort|topk)|router|gate_inp", re.I)
NORM_OPS = {"RMS_NORM", "NORM", "L2_NORM", "GROUP_NORM", "RMS_NORM_BACK"}


def read_trace(path):
    """Yield eval dicts from one trace file."""
    data = Path(path).read_bytes()
    off, n = 0, len(data)
    while off + HDR.size <= n:
        magic, nth, nrec, batch, eidx, t0 = HDR.unpack_from(data, off)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic at {off}")
        need = HDR.size + 4 * nth + NODE.size * nrec + 16 * nth * nrec
        if off + need > n:      # truncated tail (process killed mid-write)
            break
        off += HDR.size
        cpus = list(struct.unpack_from(f"<{nth}i", data, off))
        off += 4 * nth
        nodes = []
        for _ in range(nrec):
            idx, op, kind, nx, a, b, c, d, opn, nm = NODE.unpack_from(data, off)
            off += NODE.size
            nodes.append({"idx": idx, "op": op, "kind": kind, "n_extra": nx, "ne": (a, b, c, d),
                          "opname": opn.split(b"\0", 1)[0].decode(errors="replace"),
                          "name": nm.split(b"\0", 1)[0].decode(errors="replace")})
        times = []
        for _ in range(nth):
            t = array("I")
            t.frombytes(data[off:off + 16 * nrec])
            if sys.byteorder != "little":
                t.byteswap()
            times.append(t)
            off += 16 * nrec
        yield {"file": str(path), "n_threads": nth, "batch": batch, "eval_idx": eidx, "t0": t0, "cpus": cpus,
               "nodes": nodes, "times": times}


def load_topology(path):
    """lscpu -p=CPU,CORE,SOCKET,NODE,CACHE -> {cpu: {core, socket, node, l3, smt}}"""
    topo = {}
    if not path:
        return topo
    rows = []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split(",")
        cache = f[4].split(":") if len(f) > 4 else []
        rows.append((int(f[0]), int(f[1]), int(f[2] or 0), int(f[3] or 0), int(cache[-1]) if cache and cache[-1] else -1))
    first = {}
    for cpu, core, sock, node, l3 in sorted(rows):
        smt = 0 if (sock, core) not in first else 1
        first.setdefault((sock, core), cpu)
        topo[cpu] = {"core": core, "socket": sock, "node": node, "l3": l3, "smt": smt}
    return topo


def strip_layer(name):
    return re.sub(r"-\d+\b", "-N", re.sub(r"\.\d+\.", ".N.", name))


def category(nd, nth):
    op, name = nd["opname"].upper(), nd["name"]
    if nd["kind"] == 2:
        return "solo_run"
    if op in GDN_OPS or (op != "MUL_MAT_ID" and GDN_RE.search(name)):
        return "gdn_ssm"
    if ROUTE_RE.search(name) or op in ("ARGSORT", "TOP_K"):
        return "moe_routing"
    if op in NORM_OPS:
        return "norm"
    if op in ATTN_OPS or ATTN_RE.search(name):
        if op not in ("MUL_MAT", "MUL_MAT_ID"):
            return "attention"
    if op in ("MUL_MAT", "MUL_MAT_ID"):
        rows = nd["ne"][0]
        tag = "mul_mat_id" if op == "MUL_MAT_ID" else "mul_mat"
        if rows / max(nth, 1) <= FEW_ROWS_PER_THREAD:
            return tag + "_few_rows"
        return "attention_proj" if ATTN_RE.search(name) else tag
    return "other"


def analyse(paths, topology=None, max_batch=None):
    topo = load_topology(topology)
    by_op = defaultdict(lambda: {"wait_ns": 0, "thread_wait_ns": 0, "count": 0})
    by_name = defaultdict(lambda: {"wait_ns": 0, "count": 0, "op": None})
    by_cat = defaultdict(lambda: {"wait_ns": 0, "count": 0})
    last_tid, last_cpu = Counter(), Counter()
    n_evals, eval_wall, barrier_thread_ns, thread_ns_total, nth_seen = 0, 0, 0, 0, set()
    for p in paths:
        for ev in read_trace(p):
            if max_batch is not None and ev["batch"] > max_batch:
                continue
            nth, T = ev["n_threads"], ev["times"]
            nth_seen.add(nth)
            n_evals += 1
            nrec = len(ev["nodes"])
            if nrec == 0:
                continue
            eval_wall += max(T[t][4 * (nrec - 1) + 3] for t in range(nth)) - min(T[t][0] for t in range(nth))
            for k, nd in enumerate(ev["nodes"]):
                ends = [T[t][4 * k + 1] for t in range(nth)]
                mx = max(ends)
                w = mx - sum(ends) / nth
                tw = sum(mx - e for e in ends)
                for t in range(nth):
                    barrier_thread_ns += T[t][4 * k + 3] - T[t][4 * k + 1]
                    thread_ns_total += T[t][4 * k + 3] - T[t][4 * k]
                lt = ends.index(mx)
                if nd["kind"] != 2:     # a solo run is last by construction (thread 0)
                    last_tid[lt] += 1
                    last_cpu[ev["cpus"][lt]] += 1
                o = by_op[nd["opname"]]
                o["wait_ns"] += w; o["thread_wait_ns"] += tw; o["count"] += 1
                key = strip_layer(nd["name"]) or f"<{nd['opname']}>"
                nm = by_name[key]
                nm["wait_ns"] += w; nm["count"] += 1; nm["op"] = nd["opname"]
                c = by_cat[category(nd, nth)]
                c["wait_ns"] += w; c["count"] += 1
    total = sum(o["wait_ns"] for o in by_op.values())

    def ranked(d, top=None):
        items = sorted(d.items(), key=lambda kv: -kv[1]["wait_ns"])[:top]
        return [dict(key=k, **{kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()},
                     share=(v["wait_ns"] / total if total else 0.0)) for k, v in items]

    ops = ranked(by_op)
    top3 = sum(o["share"] for o in ops[:3])
    n_last = sum(last_tid.values())
    nth = max(nth_seen) if nth_seen else 0

    def group(fn):
        g = Counter()
        for cpu, c in last_cpu.items():
            g[fn(cpu)] += c
        return {str(k): {"last": v, "share": v / n_last if n_last else 0.0} for k, v in sorted(g.items(), key=lambda x: str(x[0]))}

    tinfo = lambda cpu, f: topo.get(cpu, {}).get(f, "?")  # noqa: E731
    return {
        "schema": "epyc.straggler-analysis.v1",
        "n_evals": n_evals, "n_threads": sorted(nth_seen),
        "total_straggler_wait_ns": round(total, 1),
        "eval_wall_ns": eval_wall,
        "straggler_wait_share_of_wall": (total / eval_wall) if eval_wall else None,
        "barrier_share_of_thread_time": (barrier_thread_ns / thread_ns_total) if thread_ns_total else None,
        "by_op": ops, "by_name_top": ranked(by_name, 30), "by_category": ranked(by_cat),
        "top3_ops": [o["key"] for o in ops[:3]], "top3_share": top3,
        "verdict": "LOCATED" if total > 0 and top3 >= LOCATED_SHARE else "DIFFUSE",
        "rule": f"LOCATED iff top-3 op types hold >= {LOCATED_SHARE:.0%} of total straggler wait (max end - mean end)",
        "last_thread": {
            "uniform_share": (1 / nth) if nth else None,
            "top_tids": [{"tid": t, "last": c, "share": c / n_last} for t, c in last_tid.most_common(10)],
            "top_cpus": [{"cpu": cpu, "last": c, "share": c / n_last, "core": tinfo(cpu, "core"), "l3": tinfo(cpu, "l3"),
                          "numa": tinfo(cpu, "node"), "smt": tinfo(cpu, "smt")} for cpu, c in last_cpu.most_common(10)],
            "by_l3_ccd": group(lambda c: tinfo(c, "l3")) if topo else None,
            "by_numa": group(lambda c: tinfo(c, "node")) if topo else None,
            "by_smt": group(lambda c: tinfo(c, "smt")) if topo else None,
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--topology")
    ap.add_argument("--max-batch", type=int)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    r = analyse(a.traces, a.topology, a.max_batch)
    s = json.dumps(r, indent=1)
    if a.json:
        Path(a.json).write_text(s)
    print(json.dumps({k: r[k] for k in ("n_evals", "verdict", "top3_ops", "top3_share", "straggler_wait_share_of_wall")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
