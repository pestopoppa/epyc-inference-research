#!/usr/bin/env python3
"""Serving-path evidence refresh for the current AutoKernel champion (CH-13 rerun).

WHAT THIS RE-PRODUCES
---------------------
The operator-gated bundle at `/mnt/raid0/llm/autokernel/surface/operator_gate_bundle.json`
was sealed 2026-08-28 for champion `270b48ed` from three manual gate artifacts:

  1. `champion_anchor_validation.py`  -> champion_anchor_<date>/champion_anchor_validation.json
  2. `g2_df25_concurrency_grid.py`    -> dflash2_concurrency_<date>/cells.json
  3. `df2_greedy_parity.py`           -> dflash2_greedy_parity_<date>/parity_report.json
  4. `emit_operator_gate_bundle.py`   -> the sealed bundle

The champion has since advanced (`aba5a815` at preparation time, +16.18%
single-stream vs production), so the serving number is stale-good, not
stale-wrong. This driver re-runs the same four steps against the CURRENT
champion tip, mechanically, at a run boundary. It changes no protocol except
one declared delta: llama-server host threads are pinned to the codified GPU
list (`evaluator/recipes.py:gpu_host_cpu_list()`, 184-191), which the
2026-08-27/28 originals did not do. Both arms of every comparison share the
pinning, so within-bundle deltas remain claim-grade; absolute numbers are not
directly comparable to the unpinned originals and the runbook says so.

INTERLOCKS (all refusals, none observational)
---------------------------------------------
* The mi210_0 claim (`autokernel.loop.claim.hold`) is ACQUIRED for the whole
  window and re-verified at close -- never observed via rocm-smi.
* If `--loop-pid` names a live process whose cmdline contains
  `autokernel.loop.run`, the driver refuses outright: the champion worktree and
  its build dir belong to the loop while a run lives.
* The frozen production anchor tree is verified (branch + commit) and NEVER
  built. Only the champion build dir is (incrementally) built, and only after
  its CMakeCache proves the house flags (incl. GGML_HIP_ROCWMMA_FATTN=ON,
  the CH-8 flag).
* GPU residency inside each harness is the harness's own VRAM-floor refusal;
  this driver adds nothing softer on top.

PUBLISH CONTRACT
----------------
The dashboard reader (`dashboard/server.py:_read_operator_gate_bundle`) reads
ONE canonical path; it does not glob for the newest file. So the bundle is
written to a NEW dated file (the archival record) and then atomically copied
over the canonical path via os.replace. The emitter now writes a body
`generated_at`, which that reader prefers over file mtime -- the fix for the
2026-08-31 false-STALE.

Usage (at the run-21->22 boundary, after run 21 has stopped):

    python3 scripts/benchmark/serving_evidence_refresh.py \
        --date $(date -u +%Y%m%d) [--minimal] [--stages anchor,grid,parity,emit,publish]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kernel_rnd"))

from autokernel.evaluator import recipes                            # noqa: E402
from autokernel.loop import claim                                   # noqa: E402

CHAMPION_TREE = Path("/mnt/raid0/llm/tmp/champ2")
CHAMPION_BUILD = CHAMPION_TREE / "build-hip"
CHAMPION_BRANCH = "ak/champion/llama-cpp-0db32c06e3e5"
ANCHOR_TREE = Path("/mnt/raid0/llm/llama.cpp")
ANCHOR_BRANCH = "production-consolidated-v9"
ANCHOR_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
ANCHOR_BIN = ANCHOR_TREE / "build-hip" / "bin"
ARTIFACTS = Path("/mnt/raid0/llm/artifacts-df25")
SURFACE = Path("/mnt/raid0/llm/autokernel/surface")
CANONICAL_BUNDLE = SURFACE / "operator_gate_bundle.json"

#: CH-8: the house flags every measured GPU binary must carry. Checked against
#: the champion build's CMakeCache before building; the frozen anchor build is
#: taken as-is (it IS production) and never reconfigured.
REQUIRED_CACHE_LINES = (
    "GGML_HIP:BOOL=ON",
    "GGML_HIP_ROCWMMA_FATTN:BOOL=ON",
)

STAGES = ("anchor", "grid", "parity", "emit", "publish")


class Refused(RuntimeError):
    pass


def _git(tree: Path, *args: str) -> str:
    out = subprocess.run(("git", "-C", str(tree), *args), check=False,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise Refused(f"git -C {tree} {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


def _run(argv: list[str], label: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {label}\n  $ {' '.join(argv)}", flush=True)
    rc = subprocess.run(argv).returncode
    if rc != 0:
        raise Refused(f"{label} exited rc={rc}")


def _loop_alive(pid: int) -> bool:
    """True only if `pid` is alive AND is the autokernel loop (guards pid reuse).

    Read-only /proc inspection of one explicit pid -- never a name-pattern sweep.
    """
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return False
    try:
        argv = cmdline.read_bytes().split(b"\0")
    except OSError:
        return True  # alive but unreadable: refuse rather than assume dead
    return any(b"autokernel.loop" in part for part in argv)


def preflight(args: argparse.Namespace) -> dict:
    if args.loop_pid and _loop_alive(args.loop_pid):
        raise Refused(
            f"pid {args.loop_pid} is a live autokernel loop; the champion worktree "
            "and build dir are its property while a run lives. Stop the run first "
            "(STOP file in the store, or SIGTERM) and re-invoke.")

    champion_commit = _git(ANCHOR_TREE, "rev-parse", args.champion_branch)
    head = _git(CHAMPION_TREE, "rev-parse", "HEAD")
    if head != champion_commit:
        raise Refused(
            f"{CHAMPION_TREE} HEAD {head[:12]} != {args.champion_branch} tip "
            f"{champion_commit[:12]}; refresh must measure the branch tip")
    branch_here = _git(CHAMPION_TREE, "branch", "--show-current")
    if branch_here != args.champion_branch:
        raise Refused(f"{CHAMPION_TREE} is on {branch_here!r}, "
                      f"not {args.champion_branch!r}")

    anchor_head = _git(ANCHOR_TREE, "rev-parse", "HEAD")
    anchor_branch = _git(ANCHOR_TREE, "branch", "--show-current")
    if anchor_head != ANCHOR_COMMIT or anchor_branch != ANCHOR_BRANCH:
        raise Refused(
            f"frozen anchor tree is {anchor_branch}@{anchor_head[:12]}, expected "
            f"{ANCHOR_BRANCH}@{ANCHOR_COMMIT[:12]} -- do NOT touch it; escalate")
    if not (ANCHOR_BIN / "llama-bench").is_file():
        raise Refused(f"no llama-bench under {ANCHOR_BIN}; the frozen production "
                      "build is missing and this driver will never build it")

    cache = CHAMPION_BUILD / "CMakeCache.txt"
    if not cache.is_file():
        raise Refused(f"{cache} missing -- champion build dir was never "
                      "configured; configure it with the house recipe first "
                      "(build_recipe.HOUSE_GPU_RECIPE), then re-run")
    text = cache.read_text()
    for line in REQUIRED_CACHE_LINES:
        if line not in text:
            raise Refused(f"{cache} lacks {line!r}; a binary built here would "
                          "not carry the house flags (CH-8)")

    pin = recipes.gpu_host_cpu_list()   # raises SourcedConstantUnavailable

    print(f"preflight champion {args.champion_branch} @ {champion_commit[:12]}")
    print(f"preflight anchor   {ANCHOR_BRANCH} @ {anchor_head[:12]} (frozen, untouched)")
    print(f"preflight pin      host threads {pin} (codified GPU recipe)")
    return {"champion_commit": champion_commit, "pin": pin}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True,
                    help="UTC date tag for artifact dirs and the bundle, YYYYMMDD")
    ap.add_argument("--champion-branch", default=CHAMPION_BRANCH)
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma list from {STAGES}; earlier artifacts must "
                         "already exist for later stages")
    ap.add_argument("--minimal", action="store_true",
                    help="grid runs kv_unified=0 only (the half the bundle "
                         "consumes): ~1.3h instead of ~2.6h, at the cost of the "
                         "G2 paired control")
    ap.add_argument("--reps", type=int, default=6,
                    help="anchor-validation alternating pairs (original: 6)")
    ap.add_argument("--loop-pid", type=int, default=None,
                    help="pid of the loop run that must be dead before this "
                         "may run (run 21: 2767457)")
    ap.add_argument("--build-jobs", type=int, default=64)
    args = ap.parse_args()

    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print(f"REFUSED: unknown stages {unknown}", file=sys.stderr)
        return 2

    try:
        pf = preflight(args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    champion_commit, pin = pf["champion_commit"], pf["pin"]
    anchor_out = ARTIFACTS / f"champion_anchor_{args.date}"
    grid_out = ARTIFACTS / f"dflash2_concurrency_{args.date}"
    parity_out = ARTIFACTS / f"dflash2_greedy_parity_{args.date}"
    dated_bundle = SURFACE / f"operator_gate_bundle_{args.date}.json"

    try:
        # Everything below holds the device claim, INCLUDING the incremental
        # build: unlike the funsafe harness (scratch worktrees), this build
        # writes into the loop's own build dir, so the claim doubles as the
        # interlock proving no loop measurement window is open. At a boundary
        # the build is a no-op or a short incremental, so run 9's
        # idle-while-claimed concern does not bite.
        with claim.hold() as receipt:
            print(f"claim held on {receipt['device_id']}")

            _run(["taskset", "-c", "96-183", "cmake", "--build",
                  str(CHAMPION_BUILD), "-j", str(args.build_jobs)],
                 "champion incremental build (no-op when current)")
            served = CHAMPION_BUILD / "bin" / "llama-server"
            if not served.is_file():
                raise Refused(f"{served} missing after build")

            if "anchor" in stages:
                _run([sys.executable, str(HERE / "champion_anchor_validation.py"),
                      "--anchor-bin", str(ANCHOR_BIN),
                      "--champion-bin", str(CHAMPION_BUILD / "bin"),
                      "--reps", str(args.reps),
                      "--out", str(anchor_out)],
                     "gate 1/3: champion vs frozen anchor (T1/T2-equivalent)")

            if "grid" in stages:
                grid = [sys.executable, str(HERE / "g2_df25_concurrency_grid.py"),
                        "--build-bin", str(CHAMPION_BUILD / "bin"),
                        "--out", str(grid_out),
                        "--run-id", f"df25-refresh-{args.date}",
                        "--pin-host-cores", pin]
                if args.minimal:
                    grid += ["--only-kvu", "0"]
                _run(grid, "gate 2/3: DF2-5 serving concurrency grid")

            if "parity" in stages:
                _run([sys.executable, str(HERE / "df2_greedy_parity.py"),
                      "--build-bin", str(CHAMPION_BUILD / "bin"),
                      "--out", str(parity_out),
                      "--pin-host-cores", pin],
                     "gate 3/3: DF2-6 greedy parity with controls")

        # Sealing and publishing need no device; the claim is released first.
        if "emit" in stages:
            _run([sys.executable, str(HERE / "emit_operator_gate_bundle.py"),
                  "--champion-branch", args.champion_branch,
                  "--champion-commit", champion_commit,
                  "--anchor-artifact",
                  str(anchor_out / "champion_anchor_validation.json"),
                  "--concurrency-artifact", str(grid_out / "cells.json"),
                  "--parity-artifact", str(parity_out / "parity_report.json"),
                  "--out", str(dated_bundle)],
                 "seal the operator-gate bundle (dated)")
            sealed = json.loads(dated_bundle.read_text())
            if sealed.get("gates_missing"):
                raise Refused(
                    f"bundle records missing gates {sealed['gates_missing']} -- "
                    "publishing a bundle with holes needs an explicit operator "
                    "decision, not a driver default")
            if not sealed.get("generated_at"):
                raise Refused("bundle carries no generated_at; the emitter "
                              "regressed and the false-STALE would return")

        if "publish" in stages:
            if not dated_bundle.is_file():
                raise Refused(f"nothing to publish: {dated_bundle} missing")
            tmp = CANONICAL_BUNDLE.with_suffix(".json.tmp")
            shutil.copyfile(dated_bundle, tmp)
            tmp.replace(CANONICAL_BUNDLE)   # atomic on the same filesystem
            print(f"published {dated_bundle.name} -> {CANONICAL_BUNDLE}")
            print("verify on /kernel: champion_commit "
                  f"{champion_commit[:12]}, freshness source body_generated_at")
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except claim.ClaimRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
