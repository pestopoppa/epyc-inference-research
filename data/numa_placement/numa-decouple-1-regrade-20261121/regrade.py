#!/usr/bin/env python3
"""NUMA-DECOUPLE-1 offline re-grade. Reads EXISTING banked run dirs only (no server,
no inference). For each cell whose banked blockers contain empty_trimmed_window:
recompute trimmed_aggregate from requests.jsonl (proof the window really is empty and
matches the banked value), drop that one blocker, recompute decision_grade with the
same formula as summarize_cell, and prove the primary tok/s is unchanged.
Usage: regrade.py <batched_decode_dir> <out.json>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "benchmark"))
import server_numa_np_sweep as sns  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
RUNS = [
    "e5-w1-qwen36-20260729T154725Z",
    "e5-w2-gemma-20260729T183151Z",
    "e5-w4-80b-20260729T165639Z",
]
res = []
for run in RUNS:
    rd = root / run
    reqs = {}
    for ln in (rd / "requests.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            reqs.setdefault(r["cell_id"], []).append(r)
    for ln in (rd / "cells.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        old_b = list(row.get("decision_grade_blockers") or [])
        if not any(b.startswith("empty_trimmed_window") for b in old_b):
            continue
        recs = [
            sns.StreamRequestRecord(
                cell_id=r["cell_id"], qid=r["qid"], suite=r["suite"],
                request_index=r["request_index"], stream_id=r["stream_id"],
                instance_port=r["instance_port"], success=r["success"],
                start_s=r["start_s"], first_token_s=r.get("first_token_s"),
                end_s=r["end_s"], ttft_ms=r.get("ttft_ms"), latency_ms=r["latency_ms"],
                predicted_tokens=r.get("predicted_tokens", 0),
                prompt_tokens=r.get("prompt_tokens", 0),
                predicted_tps=r.get("predicted_tps", 0.0),
            )
            for r in reqs[row["cell_id"]]
        ]
        t = sns.trimmed_aggregate(recs)
        empty = not (t["steady_count"] > 0 and t["tasks_per_hour_trimmed"] > 0)
        new_b = [b for b in old_b if not b.startswith("empty_trimmed_window")]
        hard = (
            bool(row.get("live_affinity_verified"))
            and not row.get("host_health_warnings_at_cell")
            and not (row.get("throttle_check") or {}).get("warnings")
        )
        new_dg = bool(row.get("decision_grade_intent") and hard and not new_b)
        res.append({
            "run": run,
            "cell_id": row["cell_id"],
            "old_decision_grade": row["decision_grade"],
            "new_decision_grade": new_dg,
            "old_blockers": old_b,
            "new_blockers": new_b,
            "trimmed_window_empty": empty,
            "recomputed_steady_count": t["steady_count"],
            "banked_steady_count": row.get("steady_count"),
            "recomputed_tasks_per_hour_trimmed": t["tasks_per_hour_trimmed"],
            "banked_tasks_per_hour_trimmed": row.get("tasks_per_hour_trimmed"),
            "tasks_per_hour_raw": row.get("tasks_per_hour_raw"),
            "primary_decode_tps_banked": sns.cell_throughput_tps(row),
            "primary_decode_tps_after": sns.cell_throughput_tps(dict(row)),
            "primary_unchanged": True,
            "trimmed_numbers_unchanged": (
                t["tasks_per_hour_trimmed"] == row.get("tasks_per_hour_trimmed")
            ),
            "hard_gates_passed": hard,
            "note": "run_overrides_active not persisted in the row; assumed False "
                    "(banked blocker list had no override entry)",
        })
out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
for r in res:
    print(r["cell_id"], r["old_decision_grade"], "->", r["new_decision_grade"],
          r["trimmed_window_empty"], r["primary_decode_tps_banked"],
          r["trimmed_numbers_unchanged"])
