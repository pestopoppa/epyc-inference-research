#!/usr/bin/env python
"""Acceptance-gate analysis over the canonical-arm evaluation outputs.

1. Per-arm counts by reason (from results.jsonl rows written by the collector).
2. Fail-open scan over EVERY raw result record: any status=="done" record with
   runtime None or 0.0 is a first-class finding (must be zero post-patch).
3. Cross-arm scoring consistency: score canonical_memory AS IF it were a model
   arm against canonical_runtime as the baseline (and vice versa) using the
   harness's own compute_model_stats/compute_model_metrics. All scores must be
   real, nonzero, in (0, 1]. A fabricated-zero denominator would zero them.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/workspace/tmp/effibench-x-upstream")

EVAL_DIR = Path("/workspace/tmp/effibench-gate/data/evaluation")
RESULTS = Path("/workspace/repos/epyc-inference-research/artifacts/effibench-x-acceptance-gate-20260812/results.jsonl")

# --- 1. counts by reason -----------------------------------------------------
rows = [json.loads(l) for l in RESULTS.read_text().splitlines()]
by_arm = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_arm[r["arm"]][r["reason"]].append(r["problem"])

print("=== Counts by reason ===")
for arm in sorted(by_arm):
    total = sum(len(v) for v in by_arm[arm].values())
    print(f"{arm}: {total} problems")
    for reason, probs in sorted(by_arm[arm].items()):
        print(f"  {reason}: {len(probs)}")
        if reason != "canonical-pass":
            for p in probs[:20]:
                print(f"    - {p}")

# --- 2. fail-open scan over raw records --------------------------------------
print("\n=== Fail-open scan (status=done with runtime None/0.0) ===")
fail_open_hits = []
n_records = 0
min_done_runtime = None
for arm_dir in sorted(EVAL_DIR.glob("canonical_*")):
    if not arm_dir.is_dir() or arm_dir.name == "cache":
        continue
    for f in arm_dir.glob("*_python3.json"):
        recs = json.loads(f.read_text())
        for i, r in enumerate(recs):
            n_records += 1
            if r["status"] == "done":
                rt = r.get("runtime")
                if rt is None or rt == 0.0:
                    fail_open_hits.append((arm_dir.name, f.stem, i, rt))
                else:
                    if min_done_runtime is None or rt < min_done_runtime:
                        min_done_runtime = rt
print(f"records scanned: {n_records}")
print(f"fail-open hits: {len(fail_open_hits)}")
for h in fail_open_hits[:20]:
    print(f"  {h}")
print(f"min runtime among done records: {min_done_runtime} ns"
      f" ({(min_done_runtime or 0)/1e6:.3f} ms)")

# --- 3. cross-arm scoring consistency ----------------------------------------
print("\n=== Cross-arm scoring (harness's own stats/metrics code) ===")
import importlib
es = importlib.import_module("evaluate_solution")

arms = [d.name for d in EVAL_DIR.glob("canonical_*") if d.is_dir() and d.name != "cache"]
langs = ["python3"]
stats = {}
for arm in arms:
    stats_file = EVAL_DIR / f"gatestats_{arm}.json"
    stats_file.unlink(missing_ok=True)  # always recompute
    problem_names = sorted(f.stem[:-len("_python3")] for f in (EVAL_DIR / arm).glob("*_python3.json"))
    stats[arm] = es.compute_model_stats(problem_names, langs, EVAL_DIR / arm, stats_file)

if "canonical_runtime" in stats and "canonical_memory" in stats:
    common = sorted(set(stats["canonical_runtime"]) & set(stats["canonical_memory"]))
    both_passed = [p for p in common
                   if (stats["canonical_runtime"][p]["python3"] or {}).get("passed")
                   and (stats["canonical_memory"][p]["python3"] or {}).get("passed")]
    print(f"problems evaluated in both arms: {len(common)}; both passed: {len(both_passed)}")

    for target, baseline in (("canonical_memory", "canonical_runtime"),
                             ("canonical_runtime", "canonical_memory")):
        scores = []
        zero_scores = []
        for p in both_passed:
            base = stats[baseline][p]["python3"]
            tgt = stats[target][p]["python3"]
            b, t = base.get("runtime_sum"), tgt.get("runtime_sum")
            if b is None or t is None or t == 0:
                zero_scores.append((p, "missing/zero", b, t))
                continue
            s = min(1.0, max(0.0, b / t))
            scores.append(s)
            if s == 0.0:
                zero_scores.append((p, "zero-score", b, t))
        if scores:
            scores.sort()
            n = len(scores)
            print(f"{target} scored against {baseline} (runtime_sum ratio, clipped):")
            print(f"  n={n} min={scores[0]:.4f} p25={scores[n//4]:.4f} "
                  f"median={scores[n//2]:.4f} p75={scores[3*n//4]:.4f} max={scores[-1]:.4f}")
        print(f"  degenerate/zero cases: {len(zero_scores)}")
        for z in zero_scores[:10]:
            print(f"    {z}")
