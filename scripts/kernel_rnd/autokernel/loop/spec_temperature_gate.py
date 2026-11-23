"""A speculative-decoding / acceptance lever is not KEEP or DROP on greedy evidence alone.

Pure decision (AK-TEMP-AXIS-4). Greedy evidence parked tree drafting while sampled trees accept
7-11% more tokens per verify at the served temperature (DICE replay, research 03c100462), and
at T > 0 the correctness criterion is a DISTRIBUTION, not token identity. So a lever's evidence
must carry

  1. a measured arm at the production-weighted served temperature (`production_temperature`), and
  2. for every sampled (T > 0) arm, a SPEC-PIT.1 distribution result against no-draft.

Greedy identity (T = 0) stays what it was: a reproducibility check that is neither required nor
sufficient here. SPEC-PIT.1 (handoffs/active/speculative-decoding.md, INF-50) is NOT implemented
in code yet, so the gate is a HOOK: the producer supplies results in `PIT_FIELDS` shape and an
absent or malformed result is a refusal, never a pass.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from . import sampling_axis

GATE_ID = "spec.served_temperature"
PIT_GATE_ID = "SPEC-PIT.1"
#: SPEC-PIT.1 recipe (speculative-decoding.md): fail an arm on KS p <= 1e-5.
PIT_P_MIN = 1e-5
PIT_FIELDS = ("gate", "temperature", "against", "test", "statistic", "p_value", "null_bound")
#: True once SPEC-PIT.1 exists as an instrument; until then results arrive from the hook only.
PIT_IMPLEMENTED = False


def production_temperature(served: Iterable[float], weights: Mapping[float, float] | None = None) -> float:
    """The production-weighted served temperature: the heaviest served T. With no measured
    weights that is :8083 direct traffic (30 of 32 /slots snapshots), i.e. 1.0."""
    served = tuple(served)
    if weights:
        usable = {t: w for t, w in weights.items() if t in served and w > 0}
        if usable:
            return max(sorted(usable), key=lambda t: usable[t])
    return sampling_axis.DIRECT_8083_TEMPERATURE


def applies(result: Mapping[str, Any]) -> bool:
    """A RESULT is a spec-decode/acceptance lever when it says so or its own text does."""
    return bool(result.get("spec_decode") is True or sampling_axis.is_spec_decode(result))


def _pit_check(arm: Mapping[str, Any], results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    t = arm["temperature"]
    base = {"gate": PIT_GATE_ID, "temperature": t}
    mine = [r for r in results if isinstance(r, Mapping) and r.get("gate") == PIT_GATE_ID
            and isinstance(r.get("temperature"), (int, float)) and math.isclose(r["temperature"], t)]
    if not mine:
        return {**base, "passed": False, "why": f"no {PIT_GATE_ID} distribution result at T={t:g}"}
    r = mine[-1]
    if r.get("against") != "no-draft":
        return {**base, "passed": False, "why": f"{PIT_GATE_ID} result at T={t:g} is not against no-draft"}
    test, stat = r.get("test"), r.get("statistic")
    if test == "KS":
        p = r.get("p_value")
        if not isinstance(p, (int, float)) or not math.isfinite(p):
            return {**base, "passed": False, "why": "KS result carries no finite p_value"}
        return {**base, "passed": p > PIT_P_MIN, "why": f"KS p={p:g} vs floor {PIT_P_MIN:g}"}
    if test in ("TV", "KL"):
        bound = r.get("null_bound")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (stat, bound)):
            return {**base, "passed": False, "why": f"{test} result needs finite statistic and A/A null_bound"}
        return {**base, "passed": stat <= bound, "why": f"{test}={stat:g} vs A/A null bound {bound:g}"}
    return {**base, "passed": False, "why": f"unknown distribution test {test!r}"}


def decide(sampling: Any, pit_results: Iterable[Mapping[str, Any]] | None, served: Iterable[float], *,
           weights: Mapping[float, float] | None = None) -> dict[str, Any]:
    """{ok, gate, reason, checks}: ok False means the lever cannot be KEEP or DROP yet."""
    served = tuple(served)
    # An offline projection is not evidence of a run: only measured arms satisfy this gate.
    arms = [a for a in sampling_axis.arms(sampling) if a.get("kind") != "projection"]
    checks: list[dict[str, Any]] = []
    target = production_temperature(served, weights)
    if not arms:
        return {"ok": False, "gate": GATE_ID, "checks": checks,
                "reason": "evidence sampling is unknown: no served-temperature arm can be shown"}
    if not any(math.isclose(a["temperature"], target) for a in arms):
        have = sorted({a["temperature"] for a in arms})
        return {"ok": False, "gate": GATE_ID, "checks": checks,
                "reason": f"no arm at the production-weighted served temperature T={target:g} (evidence at {have})"}
    checks.append({"gate": GATE_ID, "temperature": target, "passed": True, "why": "served-T arm present"})
    results = list(pit_results or ())
    for a in arms:
        if a["temperature"] > 0:
            checks.append(_pit_check(a, results))
    failed = [c for c in checks if not c["passed"]]
    if failed:
        return {"ok": False, "gate": PIT_GATE_ID, "checks": checks, "reason": "; ".join(c["why"] for c in failed)}
    return {"ok": True, "gate": GATE_ID, "checks": checks, "reason": "served-T arm and distribution gate passed"}
