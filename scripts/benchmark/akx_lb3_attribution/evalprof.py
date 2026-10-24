"""M arm (measurement build, GGML_CPU_PROF + the LB3.2a per-evaluation record): critical-path attribution
(AKX-ALL-LB3.2 item (c)).

Per graph node, thread 0's wall from node start to AFTER the graph barrier (INF-70 D0-c: compute + straggler wait +
barrier) is the node's share of the critical path; summed per class it charges FA K/V (with its in-kernel mask scan),
the in-graph mask build, the indexer and top-k, and every barrier wait to the op that holds it. Host-side per-ubatch
main-thread work comes from the build's existing phase lines: `[phase_prof]` (mctx_apply = apply_ubatch,
set_inputs = graph build/reuse + alloc + set_input, graph_compute) and the LB3.2a `[input_prof]` lines (set_input
per input class: the KQ mask, the PLE n-gram input that calls get_prev_tokens, ...).

Gating is by stderr ORDER, which is exact: a `[cpu_prof] graph_eval idx=K` line, a `[phase_prof]` line and an
`[input_prof]` line belong to the timed repetition whose `prompt run` line precedes them.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .phases import KIND_PROMPT, Segment

GRAPH_EVAL_RE = re.compile(r"\[cpu_prof\] graph_eval idx=(\d+) n_nodes=(\d+) nth=(\d+) acc=(\d)")
PHASE_RE = re.compile(r"\[phase_prof\] n_tokens=(\d+) mctx_apply=([0-9.]+) ms set_inputs=([0-9.]+) ms "
                      r"graph_compute=([0-9.]+) ms")
INPUT_RE = re.compile(r"\[input_prof\] n_tokens=(\d+) us=(-?\d+) class=(\S+)")


@dataclass
class Node:
    idx: int
    op: str
    name: str
    src0: str
    src1: str
    ne: Tuple[int, int, int, int]
    label: str = ""
    cls: str = ""


@dataclass
class Eval:
    gi: int
    acc: int
    key: str
    n_nodes: int
    nth: int
    t_open_us: int
    t_last_us: int
    tot_wall_us: int
    tot_comp_us: int
    nodes: Dict[int, Tuple[int, int]] = field(default_factory=dict)     # idx -> (wall_us, compute_us)


@dataclass
class EvalFile:
    shapes: Dict[str, List[Node]]
    evals: Dict[int, Eval]
    truncated: bool
    n_evals_trailer: Optional[int]


def parse_eval_file(lines: Iterable[str]) -> EvalFile:
    shapes: Dict[str, List[Node]] = {}
    evals: Dict[int, Eval] = {}
    cur_shape: Optional[List[Node]] = None
    last: Optional[Eval] = None
    truncated, n_tr = False, None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        tag = f[0]
        if tag == "S":
            cur_shape = []
            shapes[f[1]] = cur_shape
        elif tag == "N" and cur_shape is not None:
            cur_shape.append(Node(idx=int(f[1]), op=f[2], name=f[3], src0=f[4], src1=f[5],
                                  ne=tuple(int(x) for x in f[6:10])))
        elif tag == "E":
            last = Eval(gi=int(f[1]), acc=int(f[2]), key=f[3], n_nodes=int(f[4]), nth=int(f[5]), t_open_us=int(f[6]),
                        t_last_us=int(f[7]), tot_wall_us=int(f[8]), tot_comp_us=int(f[9]))
            evals[last.gi] = last
        elif tag == "W" and last is not None:
            for tok in f[1:]:
                i, w, c = tok.split(":")
                last.nodes[int(i)] = (int(w), int(c))
        elif tag == "T":
            truncated, n_tr = f[1] != "0", int(f[2])
    for nodes in shapes.values():
        label_nodes(nodes)
        classify_nodes(nodes)
    return EvalFile(shapes=shapes, evals=evals, truncated=truncated, n_evals_trailer=n_tr)


def _opaque(name: str) -> bool:
    """INF-70 DS41-C4 rule: empty, or a ggml-derived name ending in a parenthesised suffix."""
    return not name or name.endswith(")")


def label_nodes(nodes: List[Node]) -> None:
    """Own name when usable, else 'after:<nearest preceding usable name>+<distance>' (same rule as the profiler's
    snapshot labels, so the two artefacts agree)."""
    anchor = None
    for i, n in enumerate(nodes):
        if not _opaque(n.name):
            n.label, anchor = n.name, i
        elif anchor is not None:
            n.label = f"after:{nodes[anchor].name}+{i - anchor}"
        else:
            n.label = f"anon:{n.op}@{i}"


_MASK_RX = re.compile(r"kq_mask|attn_mask|self_kq_mask")
_IDX_RX = re.compile(r"^indexer_|^(after:)indexer_")
_KV_RX = re.compile(r"cache_[kv]|[kv]_cache|cache_idx|_cache_l")
NODE_CLASSES = ["fa", "topk", "indexer", "mask_build", "kv_store", "ssm_gdn", "matmul", "get_rows", "norm", "other"]


def _anchor_dist(label: str) -> int:
    m = re.search(r"\+(\d+)$", label)
    return int(m.group(1)) if m and label.startswith("after:") else 0


def classify_node(n: Node, mask_names: set) -> str:
    """Own op and names first; the after:<anchor> label (which can run across a stage boundary) only for the
    indexer, and only within 12 nodes of its anchor and after the mask test."""
    op = n.op
    if op == "FLASH_ATTN_EXT":
        return "fa"
    if op in ("TOP_K", "ARGSORT"):
        return "topk"
    if _MASK_RX.search(n.name) or (op in ("FILL", "SET_ROWS", "ADD", "CPY", "CAST", "CONT", "SCALE")
                                   and (_MASK_RX.search(n.src0) or _MASK_RX.search(n.src1)
                                        or n.src0 in mask_names or n.src1 in mask_names)):
        return "mask_build"
    if (n.name.startswith("indexer_") or (_IDX_RX.search(n.label) and _anchor_dist(n.label) <= 12)) \
            and op != "MUL_MAT_ID":
        return "indexer"
    if op in ("SET_ROWS", "CPY") and (_KV_RX.search(n.name) or _KV_RX.search(n.src0)):
        return "kv_store"
    if op in ("GATED_DELTA_NET", "SSM_CONV", "SSM_SCAN", "RWKV_WKV7", "GATED_LINEAR_ATTN"):
        return "ssm_gdn"
    if op in ("MUL_MAT", "MUL_MAT_ID", "OUT_PROD"):
        return "matmul"
    if op == "GET_ROWS":
        return "get_rows"
    if op in ("RMS_NORM", "NORM", "L2_NORM", "GROUP_NORM"):
        return "norm"
    return "other"


def classify_nodes(nodes: List[Node]) -> None:
    """Nodes in graph order; a node that consumes a mask-build node (by name) is mask build too."""
    mask_names: set = set()
    for n in nodes:
        n.cls = classify_node(n, mask_names)
        if n.cls == "mask_build" and n.name:
            mask_names.add(n.name)


# ---------------------------------------------------------------------------------------------- gating + rollup
@dataclass
class RepAttribution:
    test: int
    rep: int
    t_step_us: Optional[float]
    evals: List[int]
    node_wall_us: Dict[str, float]
    node_comp_us: Dict[str, float]
    solo_wall_us: float
    eval_span_us: float
    host_us: Dict[str, float]
    receipt_lag_ms: List[float]
    unattributed_us: Optional[float]
    excluded: Optional[str] = None


def attribute(segs: Sequence[Segment], ef: EvalFile, samples_by_test: dict) -> List[RepAttribution]:
    out: List[RepAttribution] = []
    for s in segs:
        if s.kind != KIND_PROMPT or s.rep is None:
            continue
        sn = samples_by_test.get(s.test)
        t_step = sn[s.rep - 1] / 1e3 if sn is not None and len(sn) >= s.rep else None
        gis, lags = [], []
        host: Dict[str, float] = defaultdict(float)
        for t, text in s.lines:
            m = GRAPH_EVAL_RE.search(text)
            if m:
                gi = int(m.group(1))
                gis.append(gi)
                e = ef.evals.get(gi)
                if e is not None:
                    lags.append(round((t - e.t_open_us / 1e6) * 1e3, 3))
                continue
            m = PHASE_RE.search(text)
            if m:
                host["mctx_apply"] += float(m.group(2)) * 1e3
                host["set_inputs"] += float(m.group(3)) * 1e3
                host["graph_compute"] += float(m.group(4)) * 1e3
                host["n_ubatch"] += 1
                continue
            m = INPUT_RE.search(text)
            if m:
                host["input:" + m.group(3)] += float(m.group(2))
        nw: Dict[str, float] = defaultdict(float)
        nc: Dict[str, float] = defaultdict(float)
        solo = span = 0.0
        why = None
        if not gis:
            why = "no graph evaluation inside the repetition"
        for gi in gis:
            e = ef.evals.get(gi)
            if e is None:
                why = f"graph_eval idx={gi} missing from the eval file"
                continue
            if not e.acc:
                why = f"graph_eval idx={gi} was not accumulated (acc=0)"
                continue
            shape = ef.shapes.get(e.key)
            if shape is None:
                why = f"shape {e.key} missing"
                continue
            by_idx = {n.idx: n for n in shape}
            sw = 0
            for i, (w, c) in e.nodes.items():
                n = by_idx.get(i)
                cls = n.cls if n is not None else "other"
                nw[cls] += w
                nc[cls] += c
                sw += w
            solo += max(0, e.tot_wall_us - sw)
            span += max(0, e.t_last_us - e.t_open_us)
        known = host.get("mctx_apply", 0.0) + host.get("set_inputs", 0.0) + host.get("graph_compute", 0.0)
        unattr = (t_step - known) if (t_step is not None and host.get("n_ubatch")) else None
        if t_step is None:
            why = why or "no samples_ns for this repetition"
        if ef.truncated:
            why = why or "eval file truncated"
        out.append(RepAttribution(test=s.test, rep=s.rep, t_step_us=t_step, evals=gis, node_wall_us=dict(nw),
                                  node_comp_us=dict(nc), solo_wall_us=solo, eval_span_us=span, host_us=dict(host),
                                  receipt_lag_ms=lags, unattributed_us=unattr, excluded=why))
    return out


def rollup(reps: Sequence[RepAttribution]) -> Dict[int, dict]:
    """Per test: per-class means over the usable repetitions, in us per timed step, and as shares of the step."""
    by: Dict[int, List[RepAttribution]] = defaultdict(list)
    for r in reps:
        by[r.test].append(r)
    out: Dict[int, dict] = {}
    for test, rs in sorted(by.items()):
        ok = [r for r in rs if r.excluded is None]
        res = dict(reps_used=len(ok), reps_excluded=[dict(rep=r.rep, why=r.excluded) for r in rs if r.excluded])
        if ok:
            n = len(ok)
            T = sum(r.t_step_us for r in ok) / n
            wall = {c: sum(r.node_wall_us.get(c, 0.0) for r in ok) / n for c in NODE_CLASSES}
            comp = {c: sum(r.node_comp_us.get(c, 0.0) for r in ok) / n for c in NODE_CLASSES}
            hosts = sorted({k for r in ok for k in r.host_us if k != "n_ubatch"})
            host = {k: sum(r.host_us.get(k, 0.0) for r in ok) / n for k in hosts}
            solo = sum(r.solo_wall_us for r in ok) / n
            span = sum(r.eval_span_us for r in ok) / n
            un = [r.unattributed_us for r in ok if r.unattributed_us is not None]
            lags = [x for r in ok for x in r.receipt_lag_ms]
            res.update(
                t_step_us=round(T, 1),
                node_wall_us={c: round(v, 1) for c, v in wall.items()},
                node_dead_us={c: round(wall[c] - comp[c], 1) for c in NODE_CLASSES},
                solo_wall_us=round(solo, 1), eval_span_us=round(span, 1),
                host_us={k: round(v, 1) for k, v in host.items()},
                unattributed_us=round(sum(un) / len(un), 1) if un else None,
                share={c: round(wall[c] / T, 5) for c in NODE_CLASSES} if T > 0 else {},
                share_solo=round(solo / T, 5) if T > 0 else None,
                share_host={k: round(v / T, 5) for k, v in host.items() if k != "graph_compute"} if T > 0 else {},
                coverage_defect=round(abs(sum(un) / len(un)) / T, 5) if (un and T > 0) else None,
                receipt_lag_ms=dict(p50=statistics.median(lags), max=max(lags)) if lags else None,
                evals_per_rep=sorted({len(r.evals) for r in ok}),
            )
        out[test] = res
    return out


def top_nodes(ef: EvalFile, reps_hi: Sequence[RepAttribution], reps_lo: Sequence[RepAttribution], k: int = 25):
    """Nodes with the largest mean wall increase from the low-depth to the high-depth cell of the same N (same graph
    topology, so the same node index). Reporting only: names the depth-growing nodes the class table sums."""
    def per_node(reps):
        acc: Dict[Tuple[str, int], float] = defaultdict(float)
        ok = [r for r in reps if r.excluded is None]
        for r in ok:
            for gi in r.evals:
                e = ef.evals.get(gi)
                if e is None:
                    continue
                for i, (w, _c) in e.nodes.items():
                    acc[(e.key, i)] += w
        return {kk: v / max(1, len(ok)) for kk, v in acc.items()}, ok
    hi, ok_hi = per_node(reps_hi)
    lo, _ = per_node(reps_lo)
    lo_by_idx: Dict[int, float] = defaultdict(float)
    for (key, i), v in lo.items():
        lo_by_idx[i] += v
    rows = []
    for (key, i), v in hi.items():
        n = next((x for x in ef.shapes.get(key, []) if x.idx == i), None)
        rows.append(dict(idx=i, op=n.op if n else "?", label=n.label if n else "?", cls=n.cls if n else "?",
                         wall_hi_us=round(v, 1), wall_lo_us=round(lo_by_idx.get(i, 0.0), 1),
                         delta_us=round(v - lo_by_idx.get(i, 0.0), 1)))
    rows.sort(key=lambda r: -r["delta_us"])
    return rows[:k]
