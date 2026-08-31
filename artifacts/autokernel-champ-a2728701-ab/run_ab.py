#!/usr/bin/env python3
"""A/B: reconciled champion a2728701 vs frozen production v9 build.

Reuses the AutoKernel loop's own modules (bench/claim/residency) from the lane
worktree -- statistics are NOT reimplemented here. 20 pairs + 1 discarded warmup,
alternating across processes, both surfaces (tg128 then pp512), claim held for
the whole window. Results persisted incrementally per surface.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/raid0/llm/worktrees/mains/ak-rebuild-research/scripts/kernel_rnd")
from autokernel.loop import bench, claim  # noqa: E402

OUT = Path("/mnt/raid0/llm/tmp/champ-a2728701-ab/champion-a2728701-vs-v9.json")
MODEL = Path("/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf")
ANCHOR = bench.Arm("v9-0db32c06", Path("/mnt/raid0/llm/tmp/v9v-build-base/bin/llama-bench"))
CAND = bench.Arm("champ-a2728701", Path("/mnt/raid0/llm/tmp/build-champ-a2728701/bin/llama-bench"))
PAIRS = 20

record = {
    "schema": "epyc.autokernel.ab_result.v1",
    "generated_at": None,
    "anchor": {"name": ANCHOR.name, "binary": str(ANCHOR.binary),
               "commit": "0db32c06e3e550065b78311a6031ef3dd2c4f27c",
               "resolved_from": "git -C /mnt/raid0/llm/llama.cpp rev-parse HEAD (branch production-consolidated-v9)"},
    "candidate": {"name": CAND.name, "binary": str(CAND.binary),
                  "commit": "a2728701530d2b76a71939509afbeb2386e53751",
                  "branch": "ak/champion/llama-cpp-0db32c06e3e5"},
    "model": str(MODEL),
    "pairs": PAIRS,
    "warmup_pairs": bench.WARMUP_PAIRS,
    "cpu_list": bench.CPU_LIST,
    "estimator": "median_over_median",
    "surfaces": {},
}


def persist() -> None:
    record["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.rename(OUT)


with claim.hold() as receipt:
    record["claim"] = receipt
    print(f"claim held: {receipt}", flush=True)

    for surface, pp, tg, floor, floor_note in (
        ("tg128", 0, 128, bench.MEASURED_FLOOR_PCT["tg128"][PAIRS],
         "calibrated: p95 |median effect| over all C(20,k) A/A subsets"),
        ("pp512", 512, 0, None,
         "UNCALIBRATED: no A/A floor established for pp512 at this pair count; "
         "effect reported without a decisiveness verdict"),
    ):
        print(f"=== {surface}: {PAIRS} pairs + {bench.WARMUP_PAIRS} warmup ===", flush=True)
        started = time.time()
        cmp_ = bench.compare(ANCHOR, CAND, MODEL, pp=pp, tg=tg,
                             pairs=PAIRS, noise_floor_pct=floor)
        out = cmp_.to_dict()
        out["floor_note"] = floor_note
        out["wall_seconds"] = time.time() - started
        record["surfaces"][surface] = out
        persist()
        print(json.dumps({k: out[k] for k in
                          ("surface", "effect_pct", "pairs", "noise_floor_pct",
                           "decisive", "drifting", "anchor_drift_pct",
                           "candidate_drift_pct", "residency")}, indent=1), flush=True)

print("DONE", flush=True)
