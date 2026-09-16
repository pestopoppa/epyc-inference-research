#!/usr/bin/env python3
"""INF-62 SL-5 -- A/A control on the DF2-6 greedy-parity gate.

Handoff: `handoffs/active/dflash2-block-drafter-experimental-build.md` -> SL-5.
Runs the unchanged DF2-6 runner (`df2_greedy_parity.py`, 12 prompts, fresh process per arm,
f16 KV, temp 0 / top_k 1 / seed 42) TWICE on the identical build, under one GPU claim, and
reports the A/A distribution the 7/12-vs-5/12 reading needs:

  * baseline_A vs baseline_B      -- is the non-speculative reference itself reproducible
                                     across processes? If not, no parity verdict means anything.
  * <arm>_A vs <arm>_B             -- does a speculative arm reproduce itself?
  * <arm>_X vs baseline_X          -- the DF2-6 verdict, once per run (the pass counts to compare)
  * <arm>_X vs baseline_Y (X != Y) -- the verdict against the other run's reference

Every comparison is per prompt (first differing generation-token index), never aggregate-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts/kernel_rnd"))
sys.path.insert(0, str(HERE.parent))
from autokernel.loop import claim  # noqa: E402
from df2_greedy_parity import first_diff_index  # noqa: E402

BUILD_BIN = Path("/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin")
ARMS = ("baseline", "dflash2", "draft_simple")
PIN = "184-191"


def load(run_dir: Path, arm: str) -> dict:
    return {r["id"]: r for r in json.loads((run_dir / arm / "records.json").read_text())}


def compare(a: dict, b: dict) -> dict:
    rows = []
    for pid in a:
        if pid not in b:
            continue
        same = a[pid]["tokens"] == b[pid]["tokens"]
        rows.append({"id": pid, "verdict": "PASS" if same else "FAIL",
                     "first_diff": None if same else first_diff_index(a[pid]["tokens"], b[pid]["tokens"])})
    return {"n_pass": sum(r["verdict"] == "PASS" for r in rows), "n": len(rows), "per_prompt": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", default="A,B")
    args = ap.parse_args()
    runs = args.runs.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    with claim.hold() as receipt:
        print(f"GPU claim held: {dict(receipt)}", flush=True)
        for run in runs:
            run_dir = args.out / f"run{run}"
            if (run_dir / "parity_report.json").exists():
                print(f"run {run}: already complete", flush=True)
                continue
            argv = [sys.executable, str(HERE.parent / "df2_greedy_parity.py"),
                    "--build-bin", str(BUILD_BIN), "--out", str(run_dir),
                    "--pin-host-cores", PIN]
            for arm in ARMS:
                argv += ["--only-arm", arm]
            print(f"[{time.strftime('%H:%M:%S')}] run {run}: {' '.join(argv)}", flush=True)
            rc = subprocess.run(argv).returncode
            if rc != 0:
                print(f"run {run} exited {rc}", flush=True)
                return rc
    report = {"schema": "epyc.inf62.sl5_df26_aa.v1", "build_bin": str(BUILD_BIN),
              "arms": ARMS, "runs": runs, "pin_host_cores": PIN, "comparisons": {}}
    data = {run: {arm: load(args.out / f"run{run}", arm) for arm in ARMS} for run in runs}
    for arm in ARMS:
        for i, x in enumerate(runs):
            for y in runs[i + 1:]:
                report["comparisons"][f"{arm}_{x}~{arm}_{y}"] = compare(data[x][arm], data[y][arm])
    for arm in ARMS[1:]:
        for x in runs:
            for y in runs:
                report["comparisons"][f"{arm}_{x}~baseline_{y}"] = compare(data[x][arm], data[y]["baseline"])
    for run in runs:
        rep = json.loads((args.out / f"run{run}" / "parity_report.json").read_text())
        report[f"run{run}_negative_controls"] = {
            "baseline_drafted_nothing": rep["baseline_negative_control_ok"],
            **{arm: rep["arms"][arm]["negative_control_arm_drafted"] for arm in rep["arms"]}}
        report[f"run{run}_draft_volume"] = {
            arm: {"draft_n": sum((r["draft_n"] or 0) for r in data[run][arm].values()),
                  "accepted": sum((r["draft_n_accepted"] or 0) for r in data[run][arm].values())}
            for arm in ARMS}
    (args.out / "aa_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== SL-5 A/A (PASS/n, per-prompt detail in aa_report.json) ===")
    for key, c in report["comparisons"].items():
        fails = [(r["id"], r["first_diff"]) for r in c["per_prompt"] if r["verdict"] == "FAIL"]
        print(f"  {key:32s} {c['n_pass']}/{c['n']}  fails={fails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
