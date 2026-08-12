#!/usr/bin/env python
"""Incremental collector: watch data/evaluation/canonical_*/ and append one JSONL
row per completed problem-arm to the evidence dir. Crash-safe: upstream already
persists per-problem JSON; this file is the gate's incremental per-problem record.
"""
import json
import time
import sys
from pathlib import Path

EVAL_DIR = Path("/workspace/tmp/effibench-gate/data/evaluation")
OUT = Path("/workspace/repos/epyc-inference-research/artifacts/effibench-x-acceptance-gate-20260812/results.jsonl")

seen = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        try:
            row = json.loads(line)
            seen.add((row["arm"], row["problem"]))
        except Exception:
            pass


def classify(records):
    """Classify a problem-arm result by reason."""
    n = len(records)
    n_passed = sum(1 for r in records if r.get("passed"))
    statuses = {}
    for r in records:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    zero_runtime_done = sum(
        1 for r in records if r["status"] == "done" and (r.get("runtime") in (0, 0.0))
    )
    none_runtime_done = sum(
        1 for r in records if r["status"] == "done" and r.get("runtime") is None
    )
    measurement_errors = sum(
        1 for r in records
        if r["status"] == "error" and "MeasurementError" in (r.get("text") or "")
    )
    stats_parse_errors = sum(
        1 for r in records
        if r["status"] == "error" and "Failed to parse execution statistics" in (r.get("text") or "")
    )
    if n_passed == n:
        reason = "canonical-pass"
    elif measurement_errors or stats_parse_errors:
        reason = "fail-closed-measurement-error"
    elif statuses.get("timeout"):
        reason = "timeout"
    elif statuses.get("oom"):
        reason = "oom"
    elif statuses.get("error"):
        reason = "runtime-error"
    else:
        reason = "wrong-answer"
    runtimes = [r["runtime"] for r in records if r.get("runtime") is not None]
    return {
        "n_tests": n,
        "n_passed": n_passed,
        "passed": n_passed == n,
        "reason": reason,
        "statuses": statuses,
        "zero_runtime_done": zero_runtime_done,
        "none_runtime_done": none_runtime_done,
        "measurement_errors": measurement_errors,
        "stats_parse_errors": stats_parse_errors,
        "runtime_sum_ns": sum(runtimes) if runtimes else None,
        "runtime_max_ns": max(runtimes) if runtimes else None,
    }


def scan_once():
    new = 0
    for arm_dir in sorted(EVAL_DIR.glob("canonical_*")):
        if not arm_dir.is_dir() or arm_dir.name == "cache":
            continue
        arm = arm_dir.name
        for f in sorted(arm_dir.glob("*_python3.json")):
            problem = f.stem[: -len("_python3")]
            key = (arm, problem)
            if key in seen:
                continue
            try:
                records = json.loads(f.read_text())
            except Exception:
                continue  # mid-write; next pass
            row = {"arm": arm, "problem": problem, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            row.update(classify(records))
            with open(OUT, "a") as out:
                out.write(json.dumps(row) + "\n")
            seen.add(key)
            new += 1
    return new


if __name__ == "__main__":
    watch = len(sys.argv) > 1 and sys.argv[1] == "--watch"
    while True:
        n = scan_once()
        if n:
            print(f"{time.strftime('%H:%M:%S')} collected {n} new rows (total {len(seen)})", flush=True)
        if not watch:
            break
        time.sleep(20)
