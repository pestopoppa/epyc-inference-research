#!/usr/bin/env python3
"""Calibrate the ppl_contract layer (a) served-shape case set against the ANCHOR.

    python3 -m autokernel.loop.calibrate_served_shapes --store <store> --tree <tree> \
        --stage-calibration-patch
    python3 -m autokernel.loop.calibrate_served_shapes --anchor-build <calib build> \
        --store <store> --cpu-list <served list> --threads <served -t> [--execute] \
        [--apply --tree <tree>]

WHY. Layer (a)'s served-shape cases are bound per (shape, type, width) by
`served_shape_cases.tightened_nmse_bound(anchor NMSE)`; nothing measured the anchor's
NMSE, so the manifest never existed and every ppl_contract candidate (and every fold
touching iqk/repack) refused. This tool closes that gap in three steps:

1. `--stage-calibration-patch` writes the CALIBRATION block (same shapes/types/widths,
   a non-failing bound, a per-case `AK_SERVED_NMSE` print) into the EXPERIMENTAL
   tree's tests/test-backend-ops.cpp and prints the build step. The frozen production
   tree is refused.
2. `--execute` runs that build's test-backend-ops under the CPU region lock
   (`region-lock run --cpu-list <served list> --role bench`), parses every case's
   NMSE, and writes `<store>/served_shape/calibration-<utc>.json` with provenance
   (anchor commit, every bin/ file's sha256, cpu list, threads, argv).
3. `--apply` turns the measurements into `case_set(anchor_nmse)`, writes
   `<store>/served_shape/patch.cpp` and `<store>/served_shape/manifest.json`, stages
   the final block into `--tree` (replacing the calibration block), and prints the
   commit + rebuild step. It never commits or builds itself.

Dry by default: without `--execute`/`--apply`/`--stage-calibration-patch` it only
prints the plan. Every mutating step refuses while the loop owning `--store` is alive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from . import served_shape_cases as ssc
from . import status

REGION_LOCK = "/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock"
FROZEN_TREE = "/mnt/raid0/llm/llama.cpp"
TERMINAL_STATES = {"complete", "failed", "stopped"}
CALIBRATION_TIMEOUT_S = 3600


class Refused(RuntimeError):
    """A precondition failed; nothing was changed."""


def loop_alive_refusal(store: Path) -> "str | None":
    """Refuse while the loop owning `store` is (or may be) alive: a fresh heartbeat in a
    non-terminal state, or an unreadable/malformed one."""
    body = status.read(store)
    fresh = status.freshness(body)
    if fresh["state"] == "absent" or fresh["state"] == "stale":
        return None
    if fresh["state"] == "malformed":
        return f"loop-status.json in {store} is malformed; cannot prove the loop is down"
    if str((body or {}).get("state", "")).lower() in TERMINAL_STATES:
        return None
    return (f"the loop owning {store} is alive (state {body.get('state')!r}, "
            f"{fresh.get('detail')}); stop it first")


def frozen_tree_refusal(tree: Path) -> "str | None":
    real = os.path.realpath(tree)
    if real == os.path.realpath(FROZEN_TREE):
        return f"{tree} is the frozen production tree; stage into an experimental tree"
    inside = subprocess.run(["git", "-C", str(tree), "rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return f"{tree} is not a git work tree: {inside.stderr.strip()[:200]}"
    done = subprocess.run(["git", "-C", str(tree), "branch", "--show-current"],
                          capture_output=True, text=True)
    if done.stdout.strip().startswith("production-"):
        return f"{tree} is on a production branch ({done.stdout.strip()}); refusing"
    if not (Path(tree) / "tests" / "test-backend-ops.cpp").is_file():
        return f"{tree} has no tests/test-backend-ops.cpp"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(build: Path, cpu_list: str, threads: int, argv: list) -> dict:
    bin_dir = Path(build) / "bin"
    record = {"anchor_build": str(Path(build).resolve()), "cpu_list": cpu_list,
              "threads": threads, "argv": argv,
              "measured_at": datetime.now(timezone.utc).isoformat(),
              "binary_digests": {p.name: _sha256(p) for p in sorted(bin_dir.iterdir())
                                 if p.is_file()}}
    prov = Path(build) / "provenance.json"
    if prov.is_file():
        try:
            record["anchor_commit"] = json.loads(prov.read_text(encoding="utf-8")).get(
                "champion_commit")
        except (OSError, ValueError):
            record["anchor_commit"] = None
    return record


def calibration_argv(build: Path, cpu_list: str, region_lock: str) -> list:
    binary = Path(build) / "bin" / "test-backend-ops"
    return [region_lock, "run", "--cpu-list", cpu_list, "--role", "bench", "--",
            "taskset", "-c", cpu_list, str(binary), "test", "-o", "MUL_MAT,MUL_MAT_ID",
            "-b", "CPU", "-p", ssc.calibration_regex()]


def calibration_env(build: Path) -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(build) / "bin")
    env[ssc.CASE_SET_ENV] = ssc.CALIBRATION_CASE_SET_ID
    env.pop("LD_PRELOAD", None)
    return env


def execute(build: Path, store: Path, cpu_list: str, threads: int, region_lock: str,
            out=sys.stdout) -> Path:
    if not ssc.binary_has_calibration(build):
        raise Refused(f"{build}/bin/test-backend-ops does not carry the calibration block "
                      f"({ssc.CALIBRATION_CASE_SET_ID}); stage it with "
                      "--stage-calibration-patch and rebuild test-backend-ops first")
    argv = calibration_argv(build, cpu_list, region_lock)
    print(f"execute   {' '.join(argv[:12])} ... -p <{len(ssc.canonical_triples())} cases>",
          file=out)
    done = subprocess.run(argv, capture_output=True, text=True, env=calibration_env(build),
                          stdin=subprocess.DEVNULL, timeout=CALIBRATION_TIMEOUT_S)
    if done.returncode != 0:
        raise Refused(f"calibration run exited {done.returncode}: "
                      f"{(done.stderr or done.stdout)[-600:]}")
    measurements = ssc.parse_calibration(done.stdout)
    folder = Path(store) / "served_shape"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"calibration-{stamp}.json"
    body = {"schema": "epyc.autokernel.served_shape_calibration.v1",
            "case_set_id": ssc.CASE_SET_ID,
            "provenance": provenance(build, cpu_list, threads, argv),
            "measurements": [{"shape_name": k[0], "type_a": k[1], "n": k[2], "nmse": v}
                             for k, v in sorted(measurements.items())]}
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    print(f"measured  {len(measurements)} cases -> {path}", file=out)
    return path


def load_measurements(path: Path) -> dict:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("schema") != "epyc.autokernel.served_shape_calibration.v1":
        raise Refused(f"{path} is not a served-shape calibration record")
    return {(row["shape_name"], row["type_a"], int(row["n"])): float(row["nmse"])
            for row in body["measurements"]}


def apply(measurements: dict, store: Path, tree: "Path | None", out=sys.stdout) -> None:
    try:
        cases = ssc.case_set(measurements)
    except (KeyError, ValueError) as exc:
        raise Refused(f"calibration cannot be baked: {exc}") from exc
    folder = Path(store) / "served_shape"
    folder.mkdir(parents=True, exist_ok=True)
    block = ssc.backend_ops_patch_block(cases)
    (folder / "patch.cpp").write_text(block, encoding="utf-8")
    ssc.write_manifest(folder / "manifest.json", cases)
    print(f"apply     manifest {folder / 'manifest.json'} ({len(cases)} cases), "
          f"patch {folder / 'patch.cpp'}", file=out)
    if tree is not None:
        ssc.apply_patch_block(Path(tree) / "tests" / "test-backend-ops.cpp", block)
        print(f"staged    final served-shape block into {tree}/tests/test-backend-ops.cpp",
              file=out)
        print("next      commit it on the experimental champion branch "
              f"(git -C {tree} commit -m 'tests: AK served-shape case set' -- "
              "tests/test-backend-ops.cpp); the loop's next anchor promotion rebuilds "
              "test-backend-ops with it (or restart the lane at a maintenance boundary)",
              file=out)


def main(argv: "list[str] | None" = None, out=sys.stdout) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--anchor-build", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--cpu-list")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--region-lock", default=REGION_LOCK)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--stage-calibration-patch", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        mutating = args.stage_calibration_patch or args.execute or args.apply
        if mutating:
            alive = loop_alive_refusal(args.store)
            if alive:
                raise Refused(alive)
        if args.tree is not None and (args.stage_calibration_patch or args.apply):
            frozen = frozen_tree_refusal(args.tree)
            if frozen:
                raise Refused(frozen)
        if args.stage_calibration_patch:
            if args.tree is None:
                raise Refused("--stage-calibration-patch needs --tree")
            ssc.apply_patch_block(args.tree / "tests" / "test-backend-ops.cpp",
                                  ssc.calibration_patch_block())
            print(f"staged    calibration block into {args.tree}/tests/test-backend-ops.cpp",
                  file=out)
            print("next      build test-backend-ops of this tree with the lane's CPU recipe "
                  "into a scratch build dir under the region lock, e.g. "
                  f"`{args.region_lock} run --cpu-list <served list> --role bench -- "
                  f"cmake --build <calib build> --target test-backend-ops -j 64`, then "
                  "rerun with --anchor-build <calib build> --execute --apply", file=out)
            return 0
        measurements = None
        if args.execute:
            if args.anchor_build is None or not args.cpu_list or not args.threads:
                raise Refused("--execute needs --anchor-build, --cpu-list and --threads")
            path = execute(args.anchor_build, args.store, args.cpu_list, args.threads,
                           args.region_lock, out=out)
            measurements = load_measurements(path)
        elif args.measurements is not None:
            measurements = load_measurements(args.measurements)
        if args.apply:
            if measurements is None:
                raise Refused("--apply needs --execute or --measurements")
            apply(measurements, args.store, args.tree, out=out)
            return 0
        if not mutating:
            ready = (args.anchor_build is not None
                     and ssc.binary_has_calibration(args.anchor_build))
            print(f"DRY RUN   store {args.store}; anchor build {args.anchor_build} "
                  f"{'carries' if ready else 'does NOT carry'} the calibration block; "
                  f"{len(ssc.canonical_triples())} cases; region-lock {args.region_lock} "
                  f"--cpu-list {args.cpu_list or '<served list>'} --role bench. "
                  "Pass --stage-calibration-patch / --execute / --apply.", file=out)
        return 0
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
