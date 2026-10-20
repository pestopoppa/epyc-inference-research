#!/usr/bin/env python3
"""Summarize the Q38FN thread x depth sweep into SUMMARY.md + summary.json."""
import json, statistics as st
from collections import defaultdict

R = "/mnt/raid0/llm/tmp/q38fn-threads-depth-20261020"
rows = [json.loads(l) for l in open(f"{R}/results.jsonl")]
rows += [json.loads(l) for l in open(f"{R}/followup_128k.jsonl")]

pre, dec, shas, acc = defaultdict(list), defaultdict(list), defaultdict(set), defaultdict(list)
for r in rows:
    k = r.get("kind")
    arm = {"B0": "B", "B1": "B"}.get(r.get("arm"), r.get("arm"))
    if k == "prefill_at_depth":
        pre[(arm, r["depth"])].append(r["prompt_tps"])
    elif k in ("decode_at_depth", "decode_at_depth_noeos") and r["predicted_n"] > 1:
        dec[(arm, r["depth"])].append(r["pred_tps"])
        shas[(arm, r["depth"])].add(r["content_sha"])
        if r.get("draft_n"):
            acc[(arm, r["depth"])].append(r["draft_accepted"] / r["draft_n"])
# the B0 vs B1 drift control, kept separate
drift = defaultdict(dict)
for r in rows:
    if r.get("kind") == "decode_at_depth" and r.get("arm") in ("B0", "B1") and r["predicted_n"] > 1:
        drift[r["depth"]].setdefault(r["arm"], []).append(r["pred_tps"])

depths = [512, 8192, 32768, 65536, 131072]
arms = [("A", "-t48 -tb48 (current)"), ("B", "-t48 -tb96"), ("C", "-t96 -tb96")]


def m(v):
    return f"{st.mean(v):.1f}" if v else "—"


out = ["# Q38FN served thread x depth sweep (2026-10-20)", "",
       "Binary cpu-20261004-b0ba1d427, UD-IQ4_XS + shared MTP head (n-max 4, p-min 0.5), "
       "production env/OMP stack, taskset 0-95 + interleave=all, f16 KV, greedy. "
       "Context: wikitext2_test via slot restore. Higher t/s = better.", "",
       "## Prefill at depth (4096-token tail appended at depth, t/s)", "",
       "| depth | " + " | ".join(a[1] for a in arms) + " |", "|---|---|---|---|"]
for d in depths:
    out.append(f"| {d} | " + " | ".join(m(pre[(a, d)]) for a, _ in arms) + " |")
out += ["", "## Decode at depth with MTP (256 greedy tokens, t/s, mean of reps)", "",
        "| depth | " + " | ".join(a[1] for a in arms) + " |", "|---|---|---|---|"]
for d in depths:
    out.append(f"| {d} | " + " | ".join(m(dec[(a, d)]) for a, _ in arms) + " |")
out += ["", "## Draft acceptance (accepted/drafted) and greedy-output identity", "",
        "| depth | " + " | ".join(a[1] for a in arms) + " |", "|---|---|---|---|"]
for d in depths:
    out.append(f"| {d} | " + " | ".join(
        (f"{st.mean(acc[(a, d)]):.2f} / {len(shas[(a, d)])} distinct" if acc[(a, d)] else "—")
        for a, _ in arms) + " |")
out += ["", "## Drift control (arm B run first and last)", "",
        "| depth | B0 | B1 |", "|---|---|---|"]
for d in depths:
    out.append(f"| {d} | {m(drift[d].get('B0', []))} | {m(drift[d].get('B1', []))} |")
open(f"{R}/SUMMARY.md", "w").write("\n".join(out) + "\n")
json.dump({"prefill": {f"{a}@{d}": pre[(a, d)] for a, _ in arms for d in depths},
           "decode": {f"{a}@{d}": dec[(a, d)] for a, _ in arms for d in depths},
           "acceptance": {f"{a}@{d}": acc[(a, d)] for a, _ in arms for d in depths},
           "distinct_outputs": {f"{a}@{d}": sorted(shas[(a, d)]) for a, _ in arms for d in depths}},
          open(f"{R}/summary.json", "w"), indent=1)
print("\n".join(out))
