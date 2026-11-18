"""RVP-RACE-2 executor (M1-lite): static barrier-epoch LDS race checker over gfx90a disassembly.

Input is the per-kernel ``.s`` written by ``race2_fixture_bundle.kernel_isa`` (``<hex addr>\\t<op>\\t<args>`` lines,
first line ``; <kernel name>``). No GPU.

Model (deliberately smaller than the handoff's full executor; what it does NOT do is listed below):

* CFG from ``s_branch`` / ``s_cbranch_*`` (target = pc + 4 + 4*simm16) and ``s_endpgm``. Both edges of every
  conditional branch are feasible (exec/scc/vcc are not evaluated). Indirect control flow (``s_setpc``,
  ``s_swappc``, ``s_cbranch_join``) or a branch into the middle of nowhere is UNKNOWN (fail-closed).
* LDS accesses: ``ds_read*`` = R, ``ds_write*`` = W, ``ds_<atomic>`` = A (atomic; A-A never conflicts),
  ``buffer_/global_load* ... lds`` = W issued asynchronously. Cross-lane ops that do not touch LDS memory
  (``ds_bpermute``, ``ds_permute``, ``ds_swizzle``, ``ds_nop``) are ignored. Any other ``ds_*`` opcode is UNKNOWN.
* Completion: ds ops count on lgkmcnt and return in order among themselves; scalar memory ops also count on
  lgkmcnt but may return out of order, so ``lgkmcnt(N)`` proves only the oldest ``len-N-n_smem`` ds ops complete.
  Async LDS writes complete only at ``vmcnt(0)`` (conservative, as the handoff specifies).
* ``s_barrier`` retires every access in the current window that has completed; an access still in flight
  crosses the barrier and stays in the window.
* A FINDING is a pair (Y, X) of distinct LDS instructions, at least one writing and not both atomic, where Y is
  in the window when X issues: two waves of one workgroup can execute Y and X with no barrier ordering them.
  Addresses are not evaluated (every pair may alias), so a wave-private LDS region shows up as a finding.
* Kernels whose every launch is single-wave are not raceable across waves and are reported ``N/A``.

Not modelled (later milestones): concrete address lattice, exec-mask evaluation, intra-instruction lane
write-write, cross-workgroup global races, barrier-epoch deadlock, wait-state hazards.

Verdict per kernel: ``FINDING`` | ``CLEAN`` | ``UNKNOWN`` (with reason) | ``N/A``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA = "epyc.race2.executor.v1"
MAX_VISITS_PER_INST = 64
DEFAULT_TIME_BUDGET_S = 600.0     # M1: <=10 min per kernel instance on one core

_IGNORED_DS = re.compile(r"^ds_(bpermute|permute|swizzle|nop)")
_DS_READ = re.compile(r"^ds_read")
_DS_WRITE = re.compile(r"^ds_(write|wrxchg|wrap)")
_DS_ATOMIC = re.compile(r"^ds_(add|sub|rsub|inc|dec|min|max|and|or|xor|mskor|cmpst|cmpswap|condxchg|append|consume)")
_SMEM = re.compile(r"^s_(load|buffer_load|store|buffer_store|memtime|memrealtime|dcache|atc_probe|scratch_load|scratch_store)")
_VMEM_LDS = re.compile(r"^(buffer|global)_load")
_INDIRECT = re.compile(r"^s_(setpc|swappc|cbranch_join|rfe)")
_WAIT = re.compile(r"(vmcnt|lgkmcnt|expcnt)\((\d+)\)")


class ExecutorError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Inst:
    idx: int
    addr: int
    op: str
    args: str


def parse_s(text: str) -> tuple[str, list[Inst]]:
    name, insts = "", []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith(";"):
            name = name or line[1:].strip()
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ExecutorError(f"unparsable line: {line!r}")
        insts.append(Inst(len(insts), int(parts[0], 16), parts[1].strip(), parts[2].strip() if len(parts) > 2 else ""))
    return name, insts


def lds_kind(i: Inst) -> str | None:
    """'R' | 'W' | 'A' | 'AW' (async LDS write) | None; raises on an unclassified ds_ opcode."""
    op = i.op
    if op.startswith("ds_"):
        if _IGNORED_DS.match(op):
            return None
        if _DS_READ.match(op):
            return "R"
        if _DS_WRITE.match(op):
            return "W"
        if _DS_ATOMIC.match(op):
            return "A"
        raise ExecutorError(f"unclassified ds opcode {op}")
    if _VMEM_LDS.match(op) and re.search(r"(^|\s)lds(\s|$)", i.args):
        return "AW"
    return None


def _conflict(a: str, b: str) -> bool:
    """Two LDS accesses may race unless both read or both are (non-async) atomics."""
    return not ((a == "R" and b == "R") or (a == "A" and b == "A"))


def _simm16(args: str) -> int:
    tok = args.split(",")[0].split()[0]
    v = int(tok, 0)
    return v - 0x10000 if v >= 0x8000 else v


def successors(insts: Sequence[Inst]) -> list[list[int]]:
    by_addr = {i.addr: i.idx for i in insts}
    out = []
    for i in insts:
        nxt = [i.idx + 1] if i.idx + 1 < len(insts) else []
        if i.op == "s_endpgm":
            out.append([])
        elif _INDIRECT.match(i.op):
            raise ExecutorError(f"indirect control flow at {i.addr:x}: {i.op}")
        elif i.op == "s_branch" or i.op.startswith("s_cbranch"):
            tgt = i.addr + 4 + 4 * _simm16(i.args)
            if tgt not in by_addr:
                raise ExecutorError(f"branch target {tgt:x} not an instruction (at {i.addr:x})")
            out.append([by_addr[tgt]] if i.op == "s_branch" else nxt + [by_addr[tgt]])
        else:
            out.append(nxt)
    return out


# state = (window, ds_queue, n_smem, queue_exact, async_pending)
#   window: frozenset of (idx, kind, done)
#   ds_queue: tuple of ds idx in issue order (only those touching LDS memory or not, all ds count on lgkm)
def _wait(state, args: str):
    window, q, nsm, exact, apend = state
    w = {k: int(v) for k, v in _WAIT.findall(args)}
    if not w and args.strip() not in ("", "0"):
        raise ExecutorError(f"unparsable s_waitcnt {args!r}")
    if args.strip() == "0":
        w = {"vmcnt": 0, "lgkmcnt": 0}
    done_ds: set[int] = set()
    if "lgkmcnt" in w:
        n = w["lgkmcnt"]
        if n == 0:
            done_ds, q, nsm, exact = set(q), (), 0, True
        elif exact:
            k = max(0, (len(q) + nsm - n) - nsm)
            done_ds, q = set(q[:k]), q[k:]
    if w.get("vmcnt") == 0:
        apend = False
    still_q = set(q)
    nw = set()
    for (idx, kind, done) in window:
        if not done and ((kind != "AW" and idx in done_ds and idx not in still_q) or (kind == "AW" and w.get("vmcnt") == 0)):
            done = True
        nw.add((idx, kind, done))
    return (frozenset(nw), q, nsm, exact, apend)


def _join(a, b):
    if a is None:
        return b
    wa, qa, na, ea, pa = a
    wb, qb, nb, eb, pb = b
    if qa == qb:
        q, e = qa, ea and eb
    else:
        q, e = tuple(sorted(set(qa) | set(qb))), False
    return (wa | wb, q, max(na, nb), e, pa or pb)


def analyze(insts: Sequence[Inst], *, time_budget_s: float = DEFAULT_TIME_BUDGET_S) -> dict:
    """Worklist may-analysis to a fixpoint. Returns {'verdict', 'findings', 'reason', 'seconds', ...}."""
    t0 = time.time()
    try:
        succ = successors(insts)
        kinds = [lds_kind(i) for i in insts]
    except ExecutorError as e:
        return {"verdict": "UNKNOWN", "reason": str(e), "findings": [], "seconds": round(time.time() - t0, 3)}
    if not insts:
        return {"verdict": "UNKNOWN", "reason": "empty disassembly", "findings": [], "seconds": 0.0}
    IN: list = [None] * len(insts)
    IN[0] = (frozenset(), (), 0, True, False)
    visits = [0] * len(insts)
    work = [0]
    inwork = {0}
    findings: set[tuple[int, int, str, str]] = set()
    while work:
        if time.time() - t0 > time_budget_s:
            return {"verdict": "UNKNOWN", "reason": f"time budget {time_budget_s}s exceeded", "findings": [],
                    "seconds": round(time.time() - t0, 3)}
        n = work.pop()
        inwork.discard(n)
        visits[n] += 1
        if visits[n] > MAX_VISITS_PER_INST:
            return {"verdict": "UNKNOWN", "reason": f"no fixpoint at {insts[n].addr:x}", "findings": [],
                    "seconds": round(time.time() - t0, 3)}
        i, st = insts[n], IN[n]
        window, q, nsm, exact, apend = st
        k = kinds[n]
        try:
            if i.op == "s_waitcnt":
                st = _wait(st, i.args)
            elif i.op == "s_barrier":
                st = (frozenset(e for e in window if not e[2]), q, nsm, exact, apend)
            elif _SMEM.match(i.op):
                st = (window, q, nsm + 1, exact, apend)
            elif i.op.startswith("ds_"):
                if k is not None:
                    for (y, ky, _d) in window:
                        if y != n and _conflict(ky, k):
                            findings.add((y, n, ky, k))
                    window = window | {(n, k, False)}
                st = (window, q + (n,), nsm, exact, apend)
            elif k == "AW":
                for (y, ky, _d) in window:
                    if y != n and _conflict(ky, k):
                        findings.add((y, n, ky, k))
                st = (window | {(n, k, False)}, q, nsm, exact, True)
        except ExecutorError as e:
            return {"verdict": "UNKNOWN", "reason": str(e), "findings": [], "seconds": round(time.time() - t0, 3)}
        for s in succ[n]:
            new = _join(IN[s], st)
            if new != IN[s]:
                IN[s] = new
                if s not in inwork:
                    work.append(s)
                    inwork.add(s)
    fl = [{"first": f"{insts[y].addr:x} {insts[y].op}", "second": f"{insts[x].addr:x} {insts[x].op}", "kinds": f"{ky}->{kx}"}
          for (y, x, ky, kx) in sorted(findings)]
    return {"verdict": "FINDING" if fl else "CLEAN", "reason": None, "findings": fl, "n_findings": len(fl),
            "n_insts": len(insts), "n_lds": sum(1 for k in kinds if k), "seconds": round(time.time() - t0, 3)}


def analyze_file(path: str | Path, *, waves_per_wg: Iterable[int] | None = None,
                 time_budget_s: float = DEFAULT_TIME_BUDGET_S) -> dict:
    p = Path(path)
    data = p.read_bytes()
    name, insts = parse_s(data.decode())
    w = sorted(set(waves_per_wg or []))
    rec = {"kernel": name, "disassembly": str(p), "disassembly_sha256": hashlib.sha256(data).hexdigest(), "waves_per_wg": w}
    if w and max(w) <= 1:
        return dict(rec, verdict="N/A", reason="single-wave workgroups only", findings=[], seconds=0.0)
    return dict(rec, **analyze(insts, time_budget_s=time_budget_s))


# ------------------------------------------------------------------ verdict over the fixture bundle
def decide(fixtures: Mapping[str, Sequence[dict]], controls: Sequence[dict], *, deadlock: Mapping[str, object] | None = None,
           max_seconds: float = DEFAULT_TIME_BUDGET_S) -> dict:
    """fixtures[id] = analyze_file records of that mutant's barrier-dropped launched kernels;
    controls = analyze_file records of the distinct BASELINE kernels (expected CLEAN).

    Pre-registered bar (= race_eqv.race2_m1_accept): sensitivity 7/7 (a fixture is DETECTED iff >=1 of its kernels is
    FINDING), specificity 100% (every control CLEAN; FINDING or UNKNOWN fails it), every kernel <= max_seconds."""
    per_fixture = {}
    for fid, recs in sorted(fixtures.items()):
        v = [r["verdict"] for r in recs]
        per_fixture[fid] = {"expected": "positive", "detected": "FINDING" in v, "n_kernels": len(v),
                            "kernels_finding": v.count("FINDING"), "kernels_unknown": v.count("UNKNOWN"),
                            "kernels_clean": v.count("CLEAN"), "kernels_na": v.count("N/A"),
                            "dynamic_deadlock": (deadlock or {}).get(fid)}
    ctl = [{"kernel": c["kernel"], "disassembly_sha256": c["disassembly_sha256"], "verdict": c["verdict"],
            "passed": c["verdict"] == "CLEAN", "n_findings": c.get("n_findings", 0), "reason": c.get("reason")} for c in controls]
    ctl_eval = [c for c in ctl if c["verdict"] != "N/A"]
    tp = sum(1 for x in per_fixture.values() if x["detected"])
    tn = sum(1 for c in ctl_eval if c["passed"])
    slow = max([r.get("seconds", 0) for rs in fixtures.values() for r in rs] + [c.get("seconds", 0) for c in controls] + [0])
    sens = tp / len(per_fixture) if per_fixture else 0.0
    spec = tn / len(ctl_eval) if ctl_eval else 0.0
    checks = {"sensitivity_7_of_7": bool(per_fixture) and tp == len(per_fixture) == 7,
              "specificity_100pct": bool(ctl_eval) and tn == len(ctl_eval),
              "within_time_budget": slow <= max_seconds}
    return {"verdict": "EXECUTOR_M1_BAR_MET" if all(checks.values()) else "EXECUTOR_M1_BAR_MISSED",
            "checks": checks, "sensitivity": f"{tp}/{len(per_fixture)}", "sensitivity_frac": round(sens, 4),
            "specificity": f"{tn}/{len(ctl_eval)}", "specificity_frac": round(spec, 4),
            "undetected": sorted(f for f, x in per_fixture.items() if not x["detected"]),
            "controls_failed": sorted({c["kernel"] for c in ctl_eval if not c["passed"]}),
            "max_seconds_per_kernel": slow, "per_fixture": per_fixture, "controls": ctl}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("s_files", nargs="+")
    a = ap.parse_args(argv)
    for f in a.s_files:
        r = analyze_file(f)
        print(json.dumps({k: r[k] for k in ("kernel", "verdict", "reason", "seconds")} | {"n_findings": len(r["findings"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
