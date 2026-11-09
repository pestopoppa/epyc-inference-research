"""C4 prompt-feed pieces: two-slot hardware context (RVP-C4-8), per-rule hit-rate
telemetry (RVP-C4-9) and estimated-vs-achieved speedup logging (RVP-C4-11)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .counters import SignalError

CONTEXT_SCHEMA = "epyc.autokernel.c4_hw_context.v1"
MAX_CONTEXT_BYTES = 1200          # curated: a 10k-token dump measured no better than none (intake-1221)
MAX_COUNTERS_IN_FEED = 12         # a raw ~50-counter dump measured WORSE than none (intake-1222)

GFX90A_ARCH = {"arch": "gfx90a", "device": "MI210", "cus": 104, "l2_mib": 8, "l2_scope": "unified",
               "dies": 1, "xcds": 0, "wave_size": 64, "lds_kib_per_cu": 64, "simds_per_cu": 4}
GFX90A_SCHEDULING = {"dispatch": "linear", "xcd_remap": "identity", "workgroup_swizzle_levers": [
    "l2_grouped_order(group_size)"]}   # grouped ordering is die-count independent and measured +9.823% (G17)


def hardware_context(counters: Mapping[str, float] | None = None) -> dict:
    """Slot 1 arch constants, slot 2 block-scheduling policy; optional curated counters."""
    ctx = {"schema": CONTEXT_SCHEMA, "arch_constants": dict(GFX90A_ARCH),
           "block_scheduling_policy": dict(GFX90A_SCHEDULING),
           "suppressed": ["xcd_aware_pid_swizzle: provably the identity map at xcds=0"]}
    if counters:
        if len(counters) > MAX_COUNTERS_IN_FEED:
            raise SignalError(f"feed carries at most {MAX_COUNTERS_IN_FEED} counters, got {len(counters)}")
        ctx["counters"] = {k: v for k, v in sorted(counters.items())}
    size = len(json.dumps(ctx, sort_keys=True).encode())
    if size > MAX_CONTEXT_BYTES:
        raise SignalError(f"hardware context is {size} bytes > curated cap {MAX_CONTEXT_BYTES}")
    return ctx


def validate_context(ctx: Mapping) -> None:
    arch, sched = ctx.get("arch_constants"), ctx.get("block_scheduling_policy")
    if not isinstance(arch, Mapping) or not isinstance(sched, Mapping):
        raise SignalError("both slots (arch_constants, block_scheduling_policy) are required")
    if arch.get("xcds") == 0 and sched.get("xcd_remap") != "identity":
        raise SignalError("xcds=0 requires xcd_remap=identity")


# --------------------------------------------------------------------- C4-9 / C4-11
@dataclass(frozen=True)
class RuleEvent:
    task_id: str
    rule_id: str
    fired: bool
    speedup: float | None = None         # achieved downstream, None when the run produced none
    estimated_speedup: float | None = None


def _pos(v: float | None, label: str) -> None:
    if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool)
                          or not math.isfinite(v) or v <= 0):
        raise SignalError(f"{label} must be a positive finite number or None")


def make_event(task_id: str, rule_id: str, fired: bool, speedup=None, estimated_speedup=None) -> RuleEvent:
    _pos(speedup, "speedup")
    _pos(estimated_speedup, "estimated_speedup")
    if fired and estimated_speedup is None:
        raise SignalError(f"rule {rule_id} fired without an estimated speedup (RVP-C4-11)")
    if not fired and (speedup is not None or estimated_speedup is not None):
        raise SignalError("a rule that did not fire carries no speedup")
    return RuleEvent(task_id, rule_id, bool(fired), speedup, estimated_speedup)


def append_event(path: Path, ev: RuleEvent) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev.__dict__, sort_keys=True) + "\n")


def load_events(path: Path) -> list[RuleEvent]:
    return [make_event(**json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def fold(events: Sequence[RuleEvent], *, hit_threshold: float = 1.5, min_fires: int = 5) -> dict:
    """Per-rule tasks-covered / fires / hit-rate at >threshold / avg speedup, plus prune list.

    The base rate is the hit-rate over ALL measured fired events; a rule is a prune
    candidate only with >= min_fires measured fires and hit-rate <= base rate.
    """
    tasks = {e.task_id for e in events}
    by: dict[str, list[RuleEvent]] = {}
    for e in events:
        by.setdefault(e.rule_id, []).append(e)
    measured_all = [e for e in events if e.fired and e.speedup is not None]
    base = (sum(e.speedup > hit_threshold for e in measured_all) / len(measured_all)) if measured_all else None
    rules = {}
    for rid, evs in sorted(by.items()):
        fired = [e for e in evs if e.fired]
        meas = [e for e in fired if e.speedup is not None]
        hits = sum(e.speedup > hit_threshold for e in meas)
        rules[rid] = {
            "tasks_covered": len({e.task_id for e in fired}), "fires": len(fired), "measured": len(meas),
            "hit_rate": (hits / len(meas)) if meas else None,
            "avg_speedup": (sum(e.speedup for e in meas) / len(meas)) if meas else None,
            "estimate_error": estimate_error(fired),
        }
    prune = [r for r, v in rules.items() if base is not None and v["measured"] >= min_fires
             and v["hit_rate"] is not None and v["hit_rate"] <= base]
    return {"tasks": len(tasks), "base_hit_rate": base, "hit_threshold": hit_threshold,
            "rules": rules, "prune_candidates": prune}


def estimate_error(fired: Iterable[RuleEvent]) -> dict | None:
    """|estimated - achieved| / achieved over fired events that have both; mean AND worst
    (GPA: mean 4.1% but worst rows 42% and 26% - the mean is not any single prediction's accuracy)."""
    errs = [abs(e.estimated_speedup - e.speedup) / e.speedup
            for e in fired if e.speedup is not None and e.estimated_speedup is not None]
    if not errs:
        return None
    return {"n": len(errs), "mean": sum(errs) / len(errs), "worst": max(errs)}
