"""AK-integrity pack: evaluator hardening for the C6 verify path (intake-1454, intake-1447).

Pure functions, no I/O, no compute, no subprocess. Three parts:

1. ``plan_fresh_seeds`` / ``fresh_input_recheck`` -- SimpleTES-style *fresh input per timed run*
   recheck. Every timed run gets its own input seed, derived from the pre-declared suite seed
   but distinct from it and from every other run; the recheck then proves, from the receipts
   alone, that (a) no two timed runs shared a seed, (b) no timed run reused the seed the
   candidate could have seen during correctness, and (c) distinct inputs produced distinct
   outputs (identical output digests across different input digests is the signature of a
   cached or constant result).
2. ``static_blocklist`` -- the C6 pre-build refusal set as ONE ordered list: the RVP-C6-9
   detectors (``reward_hack_scan``) plus the SOL/wave64 detectors (``reward_hack_scan_solx``).
   ``integrity.screen_candidate`` refuses on it (pre-build); parity is test-enforced.
3. ``DIVERGENCE_SEARCH_SCOPE`` -- the scoped (not implemented) AIChilles-style evaluator-derived
   shape grammar + divergence-guided regression search against the SEALED anchor for the
   section 9.5 cheap suites, with the section V-D tradeoff recorded as a pre-registered cost.

Nothing here grants keep/promotion authority; verdicts are report-only until a gate imports them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..execution import reward_hack_scan, reward_hack_scan_solx

SCHEMA = "epyc.autokernel.integrity_pack.v1"
SEED_DOMAIN = b"epyc.autokernel.fresh_timed_run_seed.v1"
PASS, FAIL, INDETERMINATE = "pass", "fail", "indeterminate"


def plan_fresh_seeds(suite_seed: int, n_runs: int) -> tuple[int, ...]:
    """Deterministic, pairwise-distinct uint64 seeds, none equal to ``suite_seed``."""
    if isinstance(suite_seed, bool) or not isinstance(suite_seed, int) or not 0 <= suite_seed < 2**64:
        raise ValueError("suite_seed must be a uint64 fixed before the arm")
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs < 1:
        raise ValueError("n_runs must be a positive integer")
    seeds: list[int] = []
    counter = 0
    while len(seeds) < n_runs:
        digest = hashlib.sha256(SEED_DOMAIN + suite_seed.to_bytes(8, "big")
                                + counter.to_bytes(8, "big")).digest()
        counter += 1
        seed = int.from_bytes(digest[:8], "big")
        if seed != suite_seed and seed not in seeds:
            seeds.append(seed)
    return tuple(seeds)


@dataclass(frozen=True)
class RecheckVerdict:
    status: str
    findings: tuple[str, ...] = ()
    runs: int = 0
    schema: str = SCHEMA
    authority: str = "report_only"

    @property
    def refuses(self) -> bool:
        """Fail closed: only an explicit pass lets the caller continue."""
        return self.status != PASS


def fresh_input_recheck(runs: Sequence[Mapping[str, object]], *, suite_seed: int,
                        min_runs: int = 2) -> RecheckVerdict:
    """Recheck the receipts of the timed runs of ONE arm.

    Each run mapping needs ``run_id``, ``input_seed`` (int), ``input_digest`` and
    ``output_digest`` (non-empty str). Missing or malformed facts, or fewer than ``min_runs``
    runs, are ``indeterminate`` (never ``pass``).
    """
    if len(runs) < min_runs:
        return RecheckVerdict(INDETERMINATE, (f"need >= {min_runs} timed runs, got {len(runs)}",),
                              len(runs))
    findings: list[str] = []
    missing: list[str] = []
    seen_seed: dict[int, str] = {}
    by_input: dict[str, set[str]] = {}
    by_output: dict[str, set[str]] = {}
    for index, run in enumerate(runs):
        rid = str(run.get("run_id", f"#{index}"))
        seed, inp, out = run.get("input_seed"), run.get("input_digest"), run.get("output_digest")
        if isinstance(seed, bool) or not isinstance(seed, int):
            missing.append(f"{rid}: input_seed missing")
            continue
        if not (isinstance(inp, str) and inp and isinstance(out, str) and out):
            missing.append(f"{rid}: input_digest/output_digest missing")
            continue
        if seed == suite_seed:
            findings.append(f"{rid}: timed run reused the correctness suite seed")
        if seed in seen_seed:
            findings.append(f"{rid}: input seed shared with {seen_seed[seed]}")
        seen_seed.setdefault(seed, rid)
        by_input.setdefault(inp, set()).add(out)
        by_output.setdefault(out, set()).add(inp)
    if findings:
        return RecheckVerdict(FAIL, tuple(findings), len(runs))
    if missing:
        return RecheckVerdict(INDETERMINATE, tuple(missing), len(runs))
    for inp, outs in by_input.items():
        if len(outs) > 1:
            findings.append(f"same input {inp[:12]} gave {len(outs)} different outputs (nondeterminism)")
    for out, inps in by_output.items():
        if len(inps) > 1:
            findings.append(f"output {out[:12]} identical across {len(inps)} different inputs "
                            "(cached or constant result)")
    return RecheckVerdict(FAIL if findings else PASS, tuple(findings), len(runs))


STATIC_BLOCKLIST_FIELDS = (
    "environment_probe_findings", "timing_dependent_branch_findings", "stream_creation_findings",
    "async_escape_findings", "instrument_frame_findings", "pointer_memoization_findings",
    "structured_short_circuit_findings", "phase_detection_findings", "capture_replay_findings",
    "content_specialization_findings",
)


def static_blocklist(diff_text: str, *, environment_probe=None) -> dict[str, tuple[str, ...]]:
    """Every blocklist class -> findings (empty classes omitted) for a unified diff.

    ``environment_probe`` lets the caller substitute its own pre-filtered environment-probe
    findings (integrity.py drops reads of variables the anchor already reads)."""
    base = reward_hack_scan.scan_unified_diff(diff_text)
    out: dict[str, tuple[str, ...]] = {}
    for name in STATIC_BLOCKLIST_FIELDS:
        rows = getattr(base, name)
        if name == "environment_probe_findings" and environment_probe is not None:
            rows = tuple(environment_probe)
        if name == "phase_detection_findings":
            # counter-derived phase state can be legitimate hot-path state (integrity.py keeps
            # those for held-out confirmation); only explicit phase probes are blocklisted
            rows = tuple(r for r in rows if ":phase_detection:" in r)
        if rows:
            out[name.removesuffix("_findings")] = rows
    for name, rows in reward_hack_scan_solx.scan_unified_diff(diff_text).findings.items():
        if rows:
            out[f"solx.{name}"] = rows
    return out


def static_blocklist_refusal(diff_text: str, **kw) -> str | None:
    """One refusal string (like ``IntegrityRefused('reward_hack_scan', ...)``) or ``None``."""
    hits = static_blocklist(diff_text, **kw)
    return "; ".join(f for rows in hits.values() for f in rows) or None


@dataclass(frozen=True)
class PreRegisteredCost:
    """The section V-D tradeoff, fixed BEFORE any divergence search runs."""
    statement: str
    source: str
    observed_in_source: Mapping[str, str]
    accepted_loss_policy: str
    stop_rule: str


DIVERGENCE_SEARCH_SCOPE = {
    "schema": "epyc.autokernel.divergence_search_scope.v1",
    "status": "scoped_not_implemented",
    "source": "intake-1447 (AIChilles, arXiv:2606.15834v2 section V-D)",
    "applies_to": "section 9.5 change-class cheap suites, vs the SEALED anchor (never the live champion)",
    "oracle": "the sealed anchor kernel is the differential oracle for each candidate",
    "shape_grammar": {
        "derivation": "deterministic extraction of op shape parameters from the evaluator's own "
                      "test cases (test-backend-ops case lists and the c3 oracle cases), NOT "
                      "free-form LLM invention; each parameter keeps its observed min/max, "
                      "alignment and divisibility constraints",
        "expansion": "grammar-bounded neighbours: boundary +/-1, alignment-breaking, above/below "
                     "every size threshold present in the candidate diff (links "
                     "AK-HACKSCAN-SOLX size_threshold_selector), non-power-of-two",
        "never": "shapes outside the evaluator's declared legal domain",
    },
    "divergence_search": {
        "objectives": ["output mismatch vs anchor beyond the op tolerance",
                       "runtime ratio candidate/anchor above the pre-registered bound",
                       "peak memory ratio above the pre-registered bound"],
        "coverage": "MAP-Elites-style archive over (op, shape-bucket, objective) cells",
        "budget": "bounded CPU-only for correctness; any timing/memory cell is a separate "
                  "inference row booked through the normal grant path, never run ad hoc",
    },
    "inputs_needed_before_implementation": [
        "sealed-anchor build identity", "per-change_class case list", "tolerance table per op"],
}

PRE_REGISTERED_COST = PreRegisteredCost(
    statement=("Robust divergence gating may remove a champion's headline gain; that outcome is "
               "an accepted, pre-registered cost of the gate, not evidence the gate is wrong."),
    source="intake-1447 section V-D: robust gating reverted Prism to 0% and TXN-Sched from 49% to 3.9%",
    observed_in_source={"Prism": "improvement reverted to 0%", "TXN-Sched": "49% -> 3.9%"},
    accepted_loss_policy=("A candidate whose gain disappears under the divergence gate is recorded "
                          "as divergence_failed with the offending shape, never silently rescued by "
                          "loosening the grammar or the bound after the fact."),
    stop_rule=("Stop on the first confirmed divergence cell (answer: not robust). Stop on "
               "futility when the budgeted grammar cells are exhausted with none found."),
)
