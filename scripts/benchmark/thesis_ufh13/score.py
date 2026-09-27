"""UFH-13 scorer: the pre-registered decision rule, and a ClaimTuple-projectable sidecar.

Rule (epyc-root ``handoffs/active/thesis-experiment-orchestrator-vs-strongest-model.md``,
"Decision rule"; X and Y await the operator at TE-0, so they are parameters, recorded):

* ``G = (Q_A2 - Q_A1) / (Q_A0 - Q_A1)`` on pooled accuracy; ``d = DS_A2 / DS_A0`` in consultant
  device-seconds (sums over all items).
* SUPPORTED: ``G >= X`` and ``d <= Y`` and the paired-bootstrap 95% lower bound of G ``> d``
  (by item, 10,000 resamples, suites stratified).
* REFUTED: the 95% upper bound of G ``< d``; or A2 quality-inferior to A1
  (``Q_A2 - Q_A1 < -1`` pooled quantum, paired).
* NO GAP: the paired 95% CI of ``Q_A0 - Q_A1`` includes 0.
* INCONCLUSIVE: anything else (write a BOUNDED-NULL-1 statement).

Evaluation order, stated because the prose does not fix it: A2-inferior REFUTED first (it holds
whatever the gap), then NO GAP (G is uninformative without a gap), then SUPPORTED, then the
G-upper-bound REFUTED, else INCONCLUSIVE.

Accounting: every item is in every arm's denominator; a failed, timed-out or unparseable item is
wrong. A MISSING record (never run) is refused unless ``allow_incomplete``, and then scored wrong
and flagged. d needs a measured consultant cost on every A0 and A2 record; otherwise d is None
and the verdict is INCONCLUSIVE with the reason.

Bootstrap resamples where ``Q_A0 == Q_A1`` leave G undefined; they are dropped and counted.
Percentile intervals; the seed is recorded.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .records import MANIFEST_NAME, RECORDS_NAME, read_records
from .suite import SUITES, Item

SCORE_SCHEMA = "ufh13-thesis-score/v1"
BELIEF_SCHEMA = "ufh13-thesis-belief/v1"
BELIEF_SIDECAR = "belief_measurements.jsonl"
ARM_NAMES = ("A0", "A1", "A2")
DEFAULT_X = 0.75
DEFAULT_Y = 0.50
DEFAULT_RESAMPLES = 10_000
DEFAULT_BOOT_SEED = 20260927

_BENCH_DIR = Path(__file__).resolve().parents[1]
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
from answer_scoring import extract_letter_answer  # noqa: E402

SCORER_PATH = _BENCH_DIR / "answer_scoring.py"


def is_correct(text: str, expected: str) -> bool:
    """The pre-registered scorer, identical for every arm."""
    return bool(text) and extract_letter_answer(text).upper() == expected.upper()


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


class IncompleteRun(RuntimeError):
    pass


def build_table(items: list[Item], records: list[dict[str, Any]], *,
                allow_incomplete: bool = False) -> dict[str, Any]:
    """Per arm: correctness and consultant cost per item, in suite order."""
    by_key = {}
    for row in records:
        if row.get("pilot"):
            continue
        by_key[(row["arm"], row["item_id"])] = row
    missing = [(arm, item.item_id) for arm in ARM_NAMES for item in items
               if (arm, item.item_id) not in by_key]
    if missing and not allow_incomplete:
        raise IncompleteRun(f"{len(missing)} (arm, item) records missing, e.g. {missing[:3]}")
    table: dict[str, Any] = {"missing": missing, "arms": {}}
    for arm in ARM_NAMES:
        correct, cost, statuses, problems = [], [], {}, []
        for item in items:
            row = by_key.get((arm, item.item_id))
            if row is None:
                correct.append(0)
                cost.append(None)
                statuses["missing"] = statuses.get("missing", 0) + 1
                continue
            status = row.get("status", "?")
            statuses[status] = statuses.get(status, 0) + 1
            ok = status == "ok" and is_correct(row.get("answer_text") or "", item.expected)
            correct.append(1 if ok else 0)
            cost.append(row.get("consultant_device_seconds"))
            problems.extend(row.get("cost_problems") or [])
        table["arms"][arm] = {"correct": correct, "cost": cost, "statuses": statuses,
                              "cost_problems": sorted(set(problems))}
    return table


def _accuracy(correct: list[int], idx: list[int]) -> float:
    return sum(correct[i] for i in idx) / len(idx) if idx else float("nan")


def score(items: list[Item], records: list[dict[str, Any]], *, x: float = DEFAULT_X,
          y: float = DEFAULT_Y, resamples: int = DEFAULT_RESAMPLES,
          boot_seed: int = DEFAULT_BOOT_SEED, allow_incomplete: bool = False) -> dict[str, Any]:
    table = build_table(items, records, allow_incomplete=allow_incomplete)
    arms = table["arms"]
    n = len(items)
    all_idx = list(range(n))
    strata = {suite: [i for i, it in enumerate(items) if it.suite == suite] for suite in SUITES}
    strata = {k: v for k, v in strata.items() if v}

    q = {arm: _accuracy(arms[arm]["correct"], all_idx) for arm in ARM_NAMES}
    q_suite = {arm: {s: _accuracy(arms[arm]["correct"], idx) for s, idx in strata.items()}
               for arm in ARM_NAMES}
    gap = q["A0"] - q["A1"]
    g = (q["A2"] - q["A1"]) / gap if gap != 0 else None
    quantum = 1.0 / n

    def cost_total(arm: str) -> float | None:
        values = arms[arm]["cost"]
        if any(v is None for v in values):
            return None
        return float(sum(values))

    ds = {arm: cost_total(arm) for arm in ARM_NAMES}
    d = ds["A2"] / ds["A0"] if ds["A2"] is not None and ds["A0"] else None

    rng = random.Random(boot_seed)
    g_samples: list[float] = []
    gap_samples: list[float] = []
    d_samples: list[float] = []
    undefined = 0
    c0, c1, c2 = (arms[a]["correct"] for a in ARM_NAMES)
    k0, k2 = arms["A0"]["cost"], arms["A2"]["cost"]
    for _ in range(resamples):
        idx = [idx_s[rng.randrange(len(idx_s))] for idx_s in strata.values() for _ in idx_s]
        s0 = sum(c0[i] for i in idx)
        s1 = sum(c1[i] for i in idx)
        s2 = sum(c2[i] for i in idx)
        gap_samples.append((s0 - s1) / n)
        if s0 == s1:
            undefined += 1
        else:
            g_samples.append((s2 - s1) / (s0 - s1))
        if d is not None:
            den = sum(k0[i] for i in idx)
            if den:
                d_samples.append(sum(k2[i] for i in idx) / den)
    g_samples.sort()
    gap_samples.sort()
    d_samples.sort()
    g_ci = ([_percentile(g_samples, 0.025), _percentile(g_samples, 0.975)]
            if g_samples else [None, None])
    gap_ci = [_percentile(gap_samples, 0.025), _percentile(gap_samples, 0.975)]
    d_ci = ([_percentile(d_samples, 0.025), _percentile(d_samples, 0.975)]
            if d_samples else [None, None])

    a2_minus_a1 = q["A2"] - q["A1"]
    reasons: list[str] = []
    if a2_minus_a1 < -quantum:
        verdict = "REFUTED"
        reasons.append(f"A2 quality-inferior to A1: Q_A2-Q_A1={a2_minus_a1:.4f} < -{quantum:.4f}")
    elif gap_ci[0] <= 0.0 <= gap_ci[1]:
        verdict = "NO GAP"
        reasons.append(f"paired 95% CI of Q_A0-Q_A1 {gap_ci} includes 0 (a saturated suite "
                       "can hide a real model gap)")
    elif d is None:
        verdict = "INCONCLUSIVE"
        reasons.append("d unmeasured: a consultant cost is missing on some A0/A2 record")
    elif g is not None and g >= x and d <= y and g_ci[0] is not None and g_ci[0] > d:
        verdict = "SUPPORTED"
        reasons.append(f"G={g:.4f}>=X={x}, d={d:.4f}<=Y={y}, G lower bound {g_ci[0]:.4f} > d")
    elif g_ci[1] is not None and g_ci[1] < d:
        verdict = "REFUTED"
        reasons.append(f"G upper bound {g_ci[1]:.4f} < d={d:.4f}: no better than random escalation")
    else:
        verdict = "INCONCLUSIVE"
        reasons.append("no rule fired; write a BOUNDED-NULL-1 statement")
    if table["missing"]:
        reasons.append(f"{len(table['missing'])} missing records scored wrong (allow_incomplete)")

    return {
        "schema": SCORE_SCHEMA,
        "n_items": n,
        "n_by_suite": {s: len(idx) for s, idx in strata.items()},
        "pooled_quantum": quantum,
        "accuracy": q,
        "accuracy_by_suite": q_suite,
        "consultant_device_seconds": ds,
        "G": g,
        "d": d,
        "G_ci95": g_ci,
        "d_ci95": d_ci,
        "gap_A0_minus_A1": gap,
        "gap_ci95": gap_ci,
        "A2_minus_A1": a2_minus_a1,
        "bootstrap": {"resamples": resamples, "seed": boot_seed, "stratified_by": sorted(strata),
                      "g_undefined_resamples": undefined, "interval": "percentile"},
        "rule": {"X": x, "Y": y, "X_Y_status": "PROPOSED (TE-0 pending) unless the caller says otherwise"},
        "verdict": verdict,
        "reasons": reasons,
        "statuses": {arm: arms[arm]["statuses"] for arm in ARM_NAMES},
        "cost_problems": {arm: arms[arm]["cost_problems"] for arm in ARM_NAMES},
        "missing_records": len(table["missing"]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def belief_rows(result: dict[str, Any], *, run_id: str, records_path: Path,
                scored_at: str, suite_sha256: str, preregistration_sha256: str | None = None
                ) -> list[dict[str, Any]]:
    """ClaimTuple-projectable rows (``scripts/vidya/claim_tuple.py::ClaimTuple`` field names).

    The producer writes; an adapter (not yet registered) projects. ``protocol_id`` is EMPTY on
    purpose: no UFH-13 protocol is codified, so every row grades as an observation until one is.
    Nothing here grades.
    """
    records_sha = _sha256(records_path)
    locator = f"run_id={run_id};suite_sha256={suite_sha256};records={records_path.name}"
    common = {
        "schema": BELIEF_SCHEMA,
        "date": scored_at,
        "protocol_id": "",
        "reps_basis": "scored",
        "attestation_path": str(records_path.resolve()),
        "attestation_sha256": records_sha,
        "attestation_locator": locator,
        "source_kind": "ufh13-thesis-measurement",
    }
    extra_common = {"run_id": run_id, "suite_sha256": suite_sha256,
                    "preregistration_sha256": preregistration_sha256,
                    "verdict": result["verdict"], "rule": result["rule"]}
    category = {"A0": "BASELINE", "A1": "BASELINE", "A2": "CANDIDATE"}
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, *, arm: str | None, direction: str, unit: str,
            claim: str, reps: int, extra: dict[str, Any] | None = None, cat: str = "CANDIDATE"):
        if value is None:
            return  # absence is recorded by omission, never filled
        mid = f"ufh13:{run_id}:{arm or 'rule'}:{metric}"
        rows.append({**common, "measurement_id": mid, "metric": metric, "value": value,
                     "category": cat, "claim": claim, "metric_direction": direction,
                     "reps": reps, "unit": unit,
                     "extra": {**extra_common, "arm": arm, **(extra or {})}})

    n = result["n_items"]
    for arm in ARM_NAMES:
        add("ufh13.accuracy.pooled", result["accuracy"][arm], arm=arm, direction="higher_better",
            unit="fraction", reps=n, cat=category[arm],
            claim=f"UFH-13 {arm} pooled accuracy on the frozen MMLU-Pro 200 + GPQA 195 suite",
            extra={"by_suite": result["accuracy_by_suite"][arm]})
        add("ufh13.consultant_device_seconds", result["consultant_device_seconds"][arm], arm=arm,
            direction="lower_better", unit="s", reps=n, cat=category[arm],
            claim=f"UFH-13 {arm} total consultant llama-server prompt+decode seconds")
    add("ufh13.gap_closure_G", result["G"], arm=None, direction="higher_better",
        unit="fraction", reps=n,
        claim="UFH-13 gap-closure fraction G=(Q_A2-Q_A1)/(Q_A0-Q_A1), pooled",
        extra={"ci95": result["G_ci95"], "bootstrap": result["bootstrap"]})
    add("ufh13.gap_closure_G_lower95", result["G_ci95"][0], arm=None,
        direction="higher_better", unit="fraction", reps=n,
        claim="UFH-13 paired-bootstrap 95% lower bound of G (suites stratified)",
        extra={"bootstrap": result["bootstrap"]})
    add("ufh13.consultant_cost_fraction_d", result["d"], arm=None, direction="lower_better",
        unit="fraction", reps=n, claim="UFH-13 consultant-cost fraction d=DS_A2/DS_A0",
        extra={"ci95": result["d_ci95"]})
    return rows


def score_run(run_dir: Path, items: list[Item], **kwargs: Any) -> dict[str, Any]:
    """Score a run directory; write ``score.json`` and the belief sidecar next to the records."""
    records_path = run_dir / RECORDS_NAME
    manifest = json.loads((run_dir / MANIFEST_NAME).read_text())
    if manifest.get("pilot"):
        raise ValueError(f"{run_dir} is a pilot run; pilots are never scored against the rule")
    result = score(items, read_records(records_path), **kwargs)
    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result.update({"run_id": manifest["run_id"], "scored_at": scored_at,
                   "records_sha256": _sha256(records_path),
                   "scorer_sha256": _sha256(SCORER_PATH)})
    (run_dir / "score.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    rows = belief_rows(result, run_id=manifest["run_id"], records_path=records_path,
                       scored_at=scored_at, suite_sha256=manifest["suite_sha256"],
                       preregistration_sha256=manifest.get("preregistration_sha256"))
    tmp = run_dir / (BELIEF_SIDECAR + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    tmp.replace(run_dir / BELIEF_SIDECAR)
    return result
