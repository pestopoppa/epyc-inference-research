"""Decision logic and fixtures for the GPU-AK signal boxes.

* RVP-ATT-1 : gfx90a MFMA pass table (LLVM SISchedule.td, SIDPFullSpeedModel) and the
  per-iteration MFMA-cycle share that an ATT hot-loop time would divide into.
* RVP-RACE-1: does the correctness gate see a barrier-deletion race?  (decision rule)
* RVP-RACE-2: M1 acceptance predicate for the AMDGCN race/deadlock executor.
* RVP-EQV-1 : do real-valued sub-tolerance algebraic mutants survive the gate?

Everything here consumes RAW per-run verdicts; the GPU builds and runs that produce
them are keeper rows.  A verdict list shorter than the declared run count is refused,
never padded.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from .counters import SignalError

MANIFEST = Path(__file__).with_name("fixtures") / "race_eqv_mutants.json"
CYCLES_PER_PASS = 4   # CDNA2 ISA convention (pass = 4 clocks); NOT re-measured here


def load_manifest(path: Path = MANIFEST) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [x["id"] for x in m["race1"]["mutants"] + m["race1"]["isa_only_mutants"] + m["eqv1"]["mutants"]]
    if len(ids) != len(set(ids)):
        raise SignalError("duplicate mutant id")
    if sum(x["kind"] == "control" for x in m["eqv1"]["mutants"]) != 2:
        raise SignalError("EQV-1 needs exactly two controls")
    return m


# --------------------------------------------------------------------- ATT-1
# SISchedule.td@rocm-6.2.0 lines 258-262 (gfx90a model): 4x4 -> 2, 16x16 -> 8, 32x32 -> 16 passes;
# f64 (DGEMM): 4x4 -> 4, 16x16 -> 8.
_MFMA = re.compile(r"^v_mfma_([a-z]+)(\d+)_(\d+)x(\d+)x(\d+)")


def mfma_passes(op: str) -> int:
    m = _MFMA.match(op)
    if not m:
        raise SignalError(f"not an MFMA mnemonic: {op}")
    dtype, bits, mn = m.group(1), int(m.group(2)), int(m.group(3))
    table = {4: 4, 16: 8} if (dtype == "f" and bits == 64) else {4: 2, 16: 8, 32: 16}
    if mn not in table:
        raise SignalError(f"no pass count for {op}")
    return table[mn]


def mfma_cycles(ops: Sequence[str]) -> int:
    return sum(mfma_passes(o) for o in ops if o.startswith("v_mfma")) * CYCLES_PER_PASS


def mfma_efficiency(loop_ops: Sequence[str], iteration_cycles: float) -> float:
    """MFMA cycles of one hot-loop iteration over its measured cycles (per SIMD)."""
    if iteration_cycles <= 0:
        raise SignalError("iteration_cycles must be positive")
    return mfma_cycles(loop_ops) / iteration_cycles


# --------------------------------------------------------------------- shared
def _verdicts(v: Sequence[bool], n: int, label: str) -> list[bool]:
    if len(v) != n or not all(isinstance(x, bool) for x in v):
        raise SignalError(f"{label}: need exactly {n} boolean verdicts, got {len(v)}")
    return list(v)


# --------------------------------------------------------------------- RACE-1
def race1_decide(mutants: Mapping[str, Mapping], unmutated: Sequence[bool], *, runs: int = 20) -> dict:
    """mutants[id] = {"gate_pass": [bool]*runs, "barriers_base": int, "barriers_mut": int,
    "waves_per_wg": int}.  gate_pass True means the correctness gate PASSED the mutant (a miss)."""
    base = _verdicts(unmutated, runs, "unmutated")
    base_ok = all(base)                  # the unmutated build must pass 20/20
    rows, missed = {}, []
    for mid, r in sorted(mutants.items()):
        proof = r["barriers_mut"] < r["barriers_base"]
        if r["waves_per_wg"] <= 1:
            rows[mid] = {"status": "excluded_single_wave"}
            continue
        passes = sum(_verdicts(r["gate_pass"], runs, mid))
        status = "invalid_no_structural_proof" if not proof else ("miss" if passes >= 1 else "caught")
        rows[mid] = {"status": status, "gate_passes": passes, "runs": runs}
        if status == "miss":
            missed.append(mid)
    return {"unmutated_pass": f"{sum(base)}/{runs}", "unmutated_ok": base_ok, "mutants": rows,
            "measured_race_misses": missed,
            "decision": ("INVALID_UNMUTATED_FAILS" if not base_ok else
                         "gate_blind_to_races" if missed else "gate_sees_all_tested_races"),
            "fixture_set_for_race2": sorted(rows)}


# --------------------------------------------------------------------- RACE-2 M1
def race2_m1_accept(*, prod_findings: int, mutant_flagged: Mapping[str, bool],
                    required_mutants: Sequence[str], isa_only_flagged: Mapping[str, bool],
                    seconds_per_kernel_instance: float) -> dict:
    """M1: 0 findings on v10 MMVQ(nw4,nw8,q8pf)+FA-vec, 100% of RACE-1 MMVQ/FA mutants and both
    ISA-only mutants flagged, <=10 min per kernel instance on one core."""
    missing = [m for m in required_mutants if not mutant_flagged.get(m)]
    isa_missing = [m for m, v in isa_only_flagged.items() if not v]
    checks = {"zero_prod_findings": prod_findings == 0, "all_mutants_flagged": not missing,
              "two_isa_only_mutants_flagged": len(isa_only_flagged) == 2 and not isa_missing,
              "within_10_min": seconds_per_kernel_instance <= 600}
    return {"accepted": all(checks.values()), "checks": checks, "unflagged": missing + isa_missing}


# --------------------------------------------------------------------- EQV-1
def eqv1_decide(survival: Mapping[str, Sequence[bool]], controls: Sequence[str],
                unmutated: Sequence[bool], *, seeds: int = 5) -> dict:
    """survival[id][seed] = True when the mutant passed EVERY gate on that seed."""
    base = _verdicts(unmutated, seeds, "unmutated")
    ctl_bad = [c for c in controls if any(_verdicts(survival[c], seeds, c))]
    survivors = sorted(m for m, v in survival.items()
                       if m not in controls and any(_verdicts(v, seeds, m)))
    if not all(base) or ctl_bad:
        decision = "INVALID_HARNESS"
    elif survivors:
        decision = "open_equivalence_milestone_for_subclass"
    else:
        decision = "close_equivalence_port_with_catch_table"
    return {"unmutated_pass": f"{sum(base)}/{seeds}", "controls_not_caught": ctl_bad,
            "survivors": survivors, "decision": decision,
            "catch_table": {m: sum(not x for x in v) for m, v in sorted(survival.items())}}
