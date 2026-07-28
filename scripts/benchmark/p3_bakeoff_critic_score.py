#!/usr/bin/env python3
"""Deterministic replay scorer for the P3 co-critic duty (zero-inference).

Consumes a pinned critic task file (p3_bakeoff_critic_build.py) plus a
``v7_quality_gate_capture.v4`` per-question capture produced in an operator
window, parses each captured response as a typed ReviewDecision verdict,
and scores verdicts against the executable-oracle gold labels.

Calibration vocabulary follows the reviewer control plane (H4):
- FA (false-accept) rate, lower-better: accept-class verdict on a
  ``known_wrong`` candidate, over committed gold-wrong rows.
- FR (false-reject) rate, lower-better: reject-class verdict on a
  ``known_correct`` candidate, over committed gold-correct rows.
- FA/FR ratio is a first-class column.
- Abstention estimand DECLARED: non-committal verdicts
  (request_evidence/abstain/escalate) and parse failures are reported as
  their own rates, never silently dropped; for the paired primary metric
  (``verdict_correct``) they count as incorrect (conservative, stated).
- Cohen's kappa over committed accept/reject verdicts + prevalence
  disclosure (intake-876: raw rates overstate quality on skewed marginals).

Fail-closed (capture contract): rows must be current-schema v4 with intact
response fingerprints; every pinned task id must be present exactly once.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_bakeoff_common import (  # noqa: E402
    CRITIC_SCORE_SCHEMA_VERSION,
    CRITIC_SUITE,
    GOLD_CORRECT,
    GOLD_WRONG,
    cohens_kappa,
    load_jsonl,
    parse_typed_verdict,
    sha256_file,
    sha256_text,
    write_json,
)

CAPTURE_SCHEMA_VERSION = "v7_quality_gate_capture.v4"


def validate_capture_row(row: dict) -> str | None:
    """Return a fail-closed reason, or None if the row is score-eligible."""
    if row.get("suite") != CRITIC_SUITE:
        return "wrong_suite"
    if row.get("capture_schema_version") != CAPTURE_SCHEMA_VERSION:
        return "wrong_capture_schema"
    response = row.get("response")
    if not isinstance(response, str):
        return "missing_response"
    fp = row.get("response_fingerprint") or {}
    if fp.get("sha256") != sha256_text(response):
        return "response_fingerprint_mismatch"
    if row.get("request_error") or row.get("finish_reason") == "request_error":
        return "request_error"
    return None


def score_rows(tasks: list[dict], capture_rows: list[dict]) -> dict:
    tasks_by_id = {t["id"]: t for t in tasks}
    rows_by_id: dict[str, dict] = {}
    duplicates: list[str] = []
    for row in capture_rows:
        rid = row.get("id")
        if rid in rows_by_id:
            duplicates.append(rid)
        rows_by_id[rid] = row

    missing = sorted(set(tasks_by_id) - set(rows_by_id))
    extra = sorted(set(rows_by_id) - set(tasks_by_id))
    quarantined: list[dict] = []
    per_row: list[dict] = []

    # Confusion over committed verdicts (accept/reject only).
    tp = fp = fn = tn = 0  # accept-on-correct, accept-on-wrong, reject-on-correct, reject-on-wrong
    n_noncommittal = n_parse_fail = 0
    truncated_rows = 0
    conf_correct: list[float] = []
    conf_incorrect: list[float] = []

    for tid, task in sorted(tasks_by_id.items()):
        row = rows_by_id.get(tid)
        if row is None:
            continue
        reason = validate_capture_row(row)
        if reason is not None:
            quarantined.append({"id": tid, "reason": reason})
            continue
        gold = task["scoring_config"]["gold_label"]
        verdict = parse_typed_verdict(row["response"])
        if row.get("truncated"):
            truncated_rows += 1
        committed = verdict["decision_class"] in ("accept", "reject")
        if verdict["parse_status"] != "ok":
            n_parse_fail += 1
            verdict_correct = False
        elif not committed:
            n_noncommittal += 1
            verdict_correct = False
        else:
            accept = verdict["decision_class"] == "accept"
            if gold == GOLD_CORRECT:
                tp += accept
                fn += not accept
                verdict_correct = accept
            else:
                fp += accept
                tn += not accept
                verdict_correct = not accept
        if verdict.get("confidence") is not None and verdict["parse_status"] == "ok":
            (conf_correct if verdict_correct else conf_incorrect).append(
                verdict["confidence"]
            )
        per_row.append({
            "id": tid,
            "gold_label": gold,
            "parse_status": verdict["parse_status"],
            "decision": verdict["decision"],
            "decision_class": verdict["decision_class"],
            "confidence": verdict["confidence"],
            "tripwire": verdict["tripwire"],
            "verdict_correct": bool(verdict_correct),
            "truncated": bool(row.get("truncated")),
            "completion_tokens": row.get("completion_tokens", 0),
        })

    n_scored = len(per_row)
    n_committed = tp + fp + fn + tn
    gold_correct_committed = tp + fn
    gold_wrong_committed = fp + tn
    fa_rate = fp / gold_wrong_committed if gold_wrong_committed else None
    fr_rate = fn / gold_correct_committed if gold_correct_committed else None
    prevalence_correct = (
        sum(1 for r in per_row if r["gold_label"] == GOLD_CORRECT) / n_scored
        if n_scored else None
    )
    summary = {
        "n_tasks": len(tasks_by_id),
        "n_scored": n_scored,
        "n_missing": len(missing),
        "n_extra_capture_rows": len(extra),
        "n_quarantined": len(quarantined),
        "n_duplicate_capture_ids": len(duplicates),
        "n_committed": n_committed,
        "committed_rate": n_committed / n_scored if n_scored else None,
        "noncommittal_rate": n_noncommittal / n_scored if n_scored else None,
        "parse_failure_rate": n_parse_fail / n_scored if n_scored else None,
        "truncated_rows": truncated_rows,
        "confusion_committed": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "fa_rate": fa_rate,
        "fr_rate": fr_rate,
        "fa_fr_ratio": (
            fa_rate / fr_rate if fa_rate is not None and fr_rate else None
        ),
        "kappa_committed": cohens_kappa(tp, fp, fn, tn),
        "prevalence_gold_correct": prevalence_correct,
        "verdict_accuracy_all": (
            sum(1 for r in per_row if r["verdict_correct"]) / n_scored
            if n_scored else None
        ),
        "verdict_accuracy_committed": (
            (tp + tn) / n_committed if n_committed else None
        ),
        "mean_confidence_when_correct": (
            sum(conf_correct) / len(conf_correct) if conf_correct else None
        ),
        "mean_confidence_when_incorrect": (
            sum(conf_incorrect) / len(conf_incorrect) if conf_incorrect else None
        ),
        "estimand_note": (
            "verdict_correct counts noncommittal + parse-fail as incorrect "
            "(declared abstention estimand); FA/FR/kappa are committed-only. "
            "Directions: FA lower-better, FR lower-better, kappa higher-better."
        ),
    }
    return {
        "summary": summary,
        "missing_ids": missing,
        "extra_ids": extra,
        "duplicate_ids": sorted(set(duplicates)),
        "quarantined": quarantined,
        "per_row": per_row,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", type=Path, required=True,
                   help="Pinned critic tasks JSON (p3_bakeoff_critic_build.py)")
    p.add_argument("--capture", type=Path, required=True,
                   help="Per-question capture JSONL from the operator window")
    p.add_argument("--arm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expect-tasks-sha256", default=None,
                   help="Fail closed unless the tasks file hashes to this "
                        "(cross-check against the bake-off manifest pin)")
    p.add_argument("--allow-partial", action="store_true",
                   help="Score an incomplete capture (rates still computed; "
                        "exit 0 despite missing rows)")
    args = p.parse_args(argv)

    tasks_sha = sha256_file(args.tasks)
    if args.expect_tasks_sha256 and tasks_sha != args.expect_tasks_sha256:
        print(f"[critic-score] FATAL tasks sha256 {tasks_sha[:12]} != "
              f"expected {args.expect_tasks_sha256[:12]}", file=sys.stderr)
        return 1
    payload = json.loads(args.tasks.read_text())
    tasks = payload["suites"][CRITIC_SUITE]
    capture_rows = load_jsonl(args.capture)
    result = score_rows(tasks, capture_rows)
    result_doc = {
        "schema_version": CRITIC_SCORE_SCHEMA_VERSION,
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "tasks_file": {"path": str(args.tasks), "sha256": tasks_sha},
        "capture_file": {"path": str(args.capture),
                         "sha256": sha256_file(args.capture)},
        "scorer": {"path": str(Path(__file__)),
                   "sha256": sha256_file(Path(__file__))},
        **result,
    }
    write_json(args.output, result_doc, sort_keys=False)
    s = result["summary"]
    print(f"[critic-score] {args.arm}: n={s['n_scored']}/{s['n_tasks']} "
          f"FA={s['fa_rate']} FR={s['fr_rate']} kappa={s['kappa_committed']} "
          f"noncommittal={s['noncommittal_rate']} parse_fail={s['parse_failure_rate']}")
    incomplete = s["n_missing"] or s["n_quarantined"] or s["n_duplicate_capture_ids"]
    if incomplete and not args.allow_partial:
        print("[critic-score] FAIL-CLOSED: capture incomplete or quarantined "
              "rows present (see output; use --allow-partial for an "
              "observation-grade partial read)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
