"""G arm (champion store image under perf): CPU-sample class shares inside the timed intervals only.

Reported, not gating: a CPU-sample share cannot charge barrier wait to the op that holds the barrier (that is the
M arm's job), so F_cpu here is a LOWER bound on an op's share of the step. What (a) and (b) fix is that the
denominator now holds only timed work: restores are outside every interval by construction, a straddling restore
run is trimmed or the repetition dropped, `state_restore` samples can never count, and templated symbols keep their
class instead of landing in `other`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List

from . import perfscript as P
from .phases import Rep, assign, trim_restore


def aggregate(sample_source: Callable[[], Iterable[P.Sample]], reps: List[Rep], main_tid: int | None = None,
              max_trim_s: float = 0.005) -> Dict[int, dict]:
    """sample_source: zero-argument callable returning an iterator over parsed perf samples (it is called twice, so
    a multi-GB perf script output can be streamed from a file). reps: from phases.timed_reps (perf-uncovered reps
    are dropped here). Returns per test idx: class counts, shares, active tids per class, repetition bookkeeping."""
    pids: Counter = Counter()
    main_all: List[P.Sample] = []
    for s in sample_source():                      # pass 1: the main thread only (restore trimming needs it)
        if s.pid is not None:
            pids[s.pid] += 1
            if s.pid == s.tid:
                main_all.append(s)
    if main_tid is None:
        main_tid = pids.most_common(1)[0][0] if pids else None
    main = sorted((s for s in main_all if s.tid == main_tid), key=lambda s: s.t)
    for r in reps:
        if r.interval is not None and r.excluded is None:
            if not r.perf_covered:
                r.excluded = "perf events not enabled before the repetition started"
                continue
            trim_restore(r, main, P.is_restore_family, max_trim_s)
    usable = sorted([r for r in reps if r.interval is not None and r.excluded is None], key=lambda r: r.interval[0])
    starts = [r.interval[0] for r in usable]
    per: Dict[tuple, Counter] = defaultdict(Counter)
    tids: Dict[tuple, set] = defaultdict(set)
    for s in sample_source():                      # pass 2: classify inside the intervals
        r = assign(s.t, starts, usable)
        if r is None:
            continue
        c = P.classify(s.sym, s.dso)
        per[(r.test, r.rep)][c] += 1
        tids[(r.test, r.rep, c)].add(s.tid)
    # a repetition with ANY state_restore sample inside its interval is not a clean timed interval: drop it whole
    for r in usable:
        if per.get((r.test, r.rep), Counter()).get("state_restore", 0) > 0:
            r.excluded = f"{per[(r.test, r.rep)]['state_restore']} state_restore samples inside the interval"
    out: Dict[int, dict] = {}
    for r in reps:
        t = out.setdefault(r.test, dict(counts=Counter(), reps_used=0, reps_excluded=[], restore_trim_ms=[],
                                        active_tids=defaultdict(set)))
        if r.interval is None or r.excluded:
            t["reps_excluded"].append(dict(rep=r.rep, why=r.excluded or "no interval"))
            continue
        t["reps_used"] += 1
        t["restore_trim_ms"].append(round(r.restore_trim_s * 1e3, 3))
        t["counts"].update(per.get((r.test, r.rep), Counter()))
        for c in P.CLASSES:
            t["active_tids"][c] |= tids.get((r.test, r.rep, c), set())
    for idx, t in out.items():
        cnt = t["counts"]
        den = sum(v for k, v in cnt.items() if k not in P.NEVER_TIMED)
        t["samples"] = den
        t["share"] = {k: round(v / den, 5) for k, v in sorted(cnt.items()) if den > 0 and k not in P.NEVER_TIMED}
        t["active_tids"] = {k: len(v) for k, v in t["active_tids"].items() if v}
        t["counts"] = dict(cnt)
    return out


def iter_perf_script(lines: Iterable[str]) -> Iterable[P.Sample]:
    for line in lines:
        s = P.parse_line(line)
        if s is not None:
            yield s


def file_source(path: str) -> Callable[[], Iterable[P.Sample]]:
    """A re-iterable sample source over a saved `perf script -F pid,tid,time,ip,sym,dso` output file."""
    def src():
        with open(path, errors="replace") as f:
            yield from iter_perf_script(f)
    return src
