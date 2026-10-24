"""llama-bench phase tracking, perf enable/disable policy and timed-interval construction (AKX-ALL-LB3.2 item (a)).

What llama-bench (champion tree, tools/llama-bench/llama-bench.cpp) does per repetition of a depth test:

    llama_memory_clear; [cached: llama_state_seq_set_data (THE RESTORE)]; print "depth run k/R[ (cached)]";
    [not cached: prefill n_depth tokens]; t_start = now; print "prompt run k/R"; test_prompt(n_prompt); t_ns = now - t_start

So the restore of repetition k+1 runs BEFORE its "depth run (cached)" line: a perf window opened at "prompt run k" and
closed at the next progress line (the 2026-10-22 instrument) always contains the restore. The timed interval is
instead taken from llama-bench's own clock: [t_line(prompt run k), t_line + samples_ns[k-1]], where t_line is the
wrapper's CLOCK_MONOTONIC receipt time of that line (taken before anything else is done with it). A guard trims the
end, a main-thread restore run that still straddles the end is cut off (`trim_restore`), and a repetition whose
interval holds any `state_restore` sample is dropped.

perf policy (`control_action`): events stay ENABLED from a test's "starting" line, through warmup, cached restores and
every timed repetition, and are DISABLED only on a non-cached "depth run" line (the multi-minute prefill). A timed
repetition is usable for perf only if events were already enabled (acknowledged) before its line: the first
repetition after a non-cached depth run is therefore reported as uncovered, never silently half-sampled.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

PROG_RE = re.compile(r"llama-bench: benchmark (\d+)/(\d+): (.*?)\s*$")
REP_RE = re.compile(r"(\d+)/(\d+)")

KIND_STARTING, KIND_WARMUP, KIND_DEPTH, KIND_DEPTH_CACHED, KIND_PROMPT, KIND_OTHER = (
    "starting", "warmup", "depth", "depth_cached", "prompt", "other")


def parse_progress(text: str) -> Optional[Tuple[int, int, str, Optional[int]]]:
    """-> (test_idx, n_tests, kind, rep) for a llama-bench --progress line, else None."""
    m = PROG_RE.search(text)
    if not m:
        return None
    idx, n, what = int(m.group(1)), int(m.group(2)), m.group(3)
    rep = None
    r = REP_RE.search(what)
    if what.startswith("starting"):
        kind = KIND_STARTING
    elif what.startswith("warmup"):
        kind = KIND_WARMUP
    elif what.startswith("depth run"):
        kind = KIND_DEPTH_CACHED if "(cached)" in what else KIND_DEPTH
        rep = int(r.group(1)) if r else None
    elif what.startswith("prompt run"):
        kind = KIND_PROMPT
        rep = int(r.group(1)) if r else None
    else:
        kind = KIND_OTHER
    return idx, n, kind, rep


def control_action(kind: str) -> Optional[str]:
    """perf --control command to send on a progress line of this kind (None = leave the state as it is)."""
    if kind in (KIND_STARTING, KIND_PROMPT):
        return "enable"
    if kind == KIND_DEPTH:
        return "disable"
    return None


@dataclass
class Segment:
    """The stretch of stderr between one progress line and the next."""
    test: int
    kind: str
    rep: Optional[int]
    t_start: float              # receipt time of the progress line (s, CLOCK_MONOTONIC)
    t_end: Optional[float]      # receipt time of the next progress line (None for the last segment)
    lines: List[Tuple[float, str]] = field(default_factory=list)   # non-progress lines inside the segment


def segments(lines: Iterable[Tuple[float, str]]) -> List[Segment]:
    """Split (t_recv, text) stderr lines into progress segments. Lines before the first progress line are dropped."""
    out: List[Segment] = []
    for t, text in lines:
        p = parse_progress(text)
        if p is not None:
            if out:
                out[-1].t_end = t
            idx, _n, kind, rep = p
            out.append(Segment(test=idx, kind=kind, rep=rep, t_start=t, t_end=None))
        elif out:
            out[-1].lines.append((t, text))
    return out


@dataclass
class Rep:
    test: int
    rep: int                    # 1-based
    t_line: float
    samples_ns: Optional[int]
    t_next: Optional[float]     # next progress line (window end)
    perf_covered: bool = True   # events enabled and acknowledged before t_line
    interval: Optional[Tuple[float, float]] = None
    excluded: Optional[str] = None
    restore_trim_s: float = 0.0


def perf_enabled_at(control_log: Sequence[Tuple[float, str, Optional[float]]], t: float) -> bool:
    """control_log: (t_sent, cmd, t_ack or None). True when the last command acknowledged before t is `enable` and no
    command was still in flight at t. Without a control log (perf ran enabled throughout) everything is covered."""
    if not control_log:
        return True
    state = False
    for t_sent, cmd, t_ack in control_log:
        if t_sent > t:
            break
        if t_ack is None or t_ack > t:
            return False        # a command was in flight at t: coverage unknown -> not covered
        state = cmd == "enable"
    return state


def timed_reps(segs: Sequence[Segment], samples_by_test: dict, guard_s: float = 0.002,
               control_log: Sequence[Tuple[float, str, Optional[float]]] = ()) -> List[Rep]:
    """Timed repetitions with their intervals. samples_by_test: test idx (1-based, llama-bench order) -> samples_ns."""
    reps: List[Rep] = []
    for s in segs:
        if s.kind != KIND_PROMPT or s.rep is None:
            continue
        sn = samples_by_test.get(s.test)
        ns = sn[s.rep - 1] if sn is not None and len(sn) >= s.rep else None
        r = Rep(test=s.test, rep=s.rep, t_line=s.t_start, samples_ns=ns, t_next=s.t_end,
                perf_covered=perf_enabled_at(control_log, s.t_start))
        if ns is None:
            r.excluded = "no samples_ns for this repetition"
        else:
            end = s.t_start + ns / 1e9 - guard_s
            if s.t_end is not None:
                end = min(end, s.t_end)
            if end <= s.t_start:
                r.excluded = "interval empty after guard"
            else:
                r.interval = (s.t_start, end)
        reps.append(r)
    return reps


def trim_restore(rep: Rep, main_samples: Sequence, is_restore_family, max_trim_s: float = 0.005) -> None:
    """Cut a main-thread restore run that straddles the interval end (wrapper receipt latency shifted the interval
    late). main_samples: the main thread's samples (objects with .t), sorted by time, covering at least the interval
    and what follows it up to the next progress line. A run longer than max_trim_s inside the interval excludes the
    repetition instead (the interval is then not trustworthy)."""
    if rep.interval is None:
        return
    a, b = rep.interval
    ts = [s.t for s in main_samples]
    j = bisect.bisect_right(ts, b)          # first main-thread sample after the interval end
    if j >= len(main_samples) or not is_restore_family(main_samples[j]):
        return
    k = j
    while k - 1 >= 0 and main_samples[k - 1].t > a and is_restore_family(main_samples[k - 1]):
        k -= 1
    if k == j:
        return                              # the run starts after the interval: nothing to trim
    run_start = main_samples[k].t
    trim = b - run_start
    if trim > max_trim_s:
        rep.excluded = f"restore run straddles the interval end by {trim * 1e3:.1f} ms (> {max_trim_s * 1e3:.1f} ms)"
        return
    # end just before the first restore-family sample of the run: the last non-restore sample, or the start
    prev = main_samples[k - 1].t if k - 1 >= 0 and main_samples[k - 1].t > a else a
    rep.interval = (a, max(prev, a + 1e-9))
    rep.restore_trim_s = b - rep.interval[1]


def assign(t: float, starts: Sequence[float], reps: Sequence[Rep]) -> Optional[Rep]:
    """The usable repetition whose interval contains t (reps sorted by interval start, `starts` their starts)."""
    k = bisect.bisect_right(starts, t) - 1
    if k < 0:
        return None
    r = reps[k]
    if r.interval is None or r.excluded:
        return None
    return r if r.interval[0] <= t <= r.interval[1] else None
