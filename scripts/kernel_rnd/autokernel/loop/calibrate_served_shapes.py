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
import shlex
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
BUILD_TIMEOUT_S = 7200
#: The lanes' exact anchor recipe (tmp/ak-lanes-relaunch-20261005/build_anchors.sh and
#: build_anchor_q38fn_802bf9.sh): Release, gcc-15, GGML_NATIVE, GGML_OPENMP, HIP off.
ANCHOR_RECIPE_DEFINES = ("-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=OFF", "-DGGML_NATIVE=ON",
                         "-DGGML_OPENMP=ON", "-DCMAKE_C_COMPILER=/usr/bin/gcc-15",
                         "-DCMAKE_CXX_COMPILER=/usr/bin/g++-15")
CALIBRATION_BUILD_DIRNAME = "build-ak-calib"


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


def provenance(build: Path, cpu_list: str, threads: int, argv: list,
               timeout_s: int = CALIBRATION_TIMEOUT_S) -> dict:
    bin_dir = Path(build) / "bin"
    record = {"anchor_build": str(Path(build).resolve()), "cpu_list": cpu_list,
              "threads": threads, "timeout_s": timeout_s, "argv": argv,
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


def build_calibration_argv(tree: Path, build: Path, cpu_list: str, region_lock: str,
                           jobs: int) -> list:
    """`region-lock run --role build` around configure + build of test-backend-ops with
    the lanes' exact anchor recipe, pinned to `cpu_list`."""
    script = (f"set -euo pipefail; taskset -c {shlex.quote(cpu_list)} cmake -S "
              f"{shlex.quote(str(tree))} -B {shlex.quote(str(build))} "
              + " ".join(ANCHOR_RECIPE_DEFINES)
              + f" && taskset -c {shlex.quote(cpu_list)} nice -n 10 cmake --build "
              f"{shlex.quote(str(build))} -j {int(jobs)} --target test-backend-ops")
    return [region_lock, "run", "--cpu-list", cpu_list, "--role", "build",
            "--timeout-s", str(BUILD_TIMEOUT_S), "--tag", "ak-served-shape-calibration-build",
            "--", "bash", "-c", script]


def build_calibration(tree: Path, cpu_list: str, region_lock: str, jobs: int,
                      out=sys.stdout) -> Path:
    """Build test-backend-ops of `tree` (with the staged calibration block) into
    `<tree>/build-ak-calib`. Refuses unless the tree's ONLY change is the staged
    tests/test-backend-ops.cpp carrying the calibration block -- the build must be the
    anchor commit's code plus the test patch, nothing else."""
    test_file = Path(tree) / "tests" / "test-backend-ops.cpp"
    if ssc.CALIBRATION_CASE_SET_ID not in test_file.read_text(encoding="utf-8"):
        raise Refused(f"{test_file} carries no calibration block; run "
                      "--stage-calibration-patch first")
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True)
    changed = {line[3:] for line in dirty.stdout.splitlines() if line.strip()}
    if dirty.returncode != 0 or changed - {"tests/test-backend-ops.cpp"}:
        raise Refused(f"{tree} has changes besides tests/test-backend-ops.cpp: "
                      f"{sorted(changed - {'tests/test-backend-ops.cpp'})}")
    build = Path(tree) / CALIBRATION_BUILD_DIRNAME
    argv = build_calibration_argv(tree, build, cpu_list, region_lock, jobs)
    print(f"build     {' '.join(argv[:11])} -- <configure + build test-backend-ops>",
          file=out)
    done = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          timeout=BUILD_TIMEOUT_S + 600)
    if done.returncode != 0:
        raise Refused(f"calibration build exited {done.returncode}: "
                      f"{(done.stderr or done.stdout)[-600:]}")
    if not ssc.binary_has_calibration(build):
        raise Refused(f"{build}/bin/test-backend-ops was built without the calibration block")
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    (build / "provenance.json").write_text(json.dumps(
        {"champion_commit": head, "built_for": "served-shape calibration",
         "recipe": list(ANCHOR_RECIPE_DEFINES), "cpu_list": cpu_list}), encoding="utf-8")
    print(f"built     {build} (tree HEAD {head[:12]}); next: --anchor-build {build} "
          "--execute --apply", file=out)
    return build


def served_recipe(launch_path: Path, build: Path, *, cpu_list: "str | None",
                  threads: "int | None") -> dict:
    """Round-12: the SERVED launch the calibration must reproduce, from the lane's
    resolved launch JSON (`inputs-*/<target>.launch.json`): its launch environment
    (GGML_IQK, OMP_*, ...) with LD_LIBRARY_PATH's build entry rebound to `build`, its
    topology prefix, its cpu list and its -t. Refuses when any of these is missing,
    when --cpu-list / --threads disagree with it, or when a loader variable other than
    LD_LIBRARY_PATH is set (the served environment could not be reproduced)."""
    try:
        launch = json.loads(Path(launch_path).read_text(encoding="utf-8"))
        env = dict(launch["launch_env"])
        prefix = list(launch["topology_prefix"])
        argv = list(launch["command_argv"])
        served_build = str(Path(launch["build_dir"]) / "bin")
        served_cpus = str(launch["template"]["cpu_list"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused(f"{launch_path} is not a resolved launch record: {exc}") from exc
    served_threads = next((int(argv[i + 1]) for i, tok in enumerate(argv[:-1])
                           if tok in ("-t", "--threads") and str(argv[i + 1]).isdigit()), None)
    if not served_threads or not prefix:
        raise Refused(f"{launch_path} carries no -t or no topology prefix")
    if threads is not None and threads != served_threads:
        raise Refused(f"--threads {threads} differs from the served -t {served_threads}")
    if cpu_list is not None and cpu_list != served_cpus:
        raise Refused(f"--cpu-list {cpu_list} differs from the served cpu list {served_cpus}")
    loader = sorted(k for k in env if k.startswith("LD_") and k != "LD_LIBRARY_PATH")
    if loader:
        raise Refused(f"served env sets loader variables {loader}; cannot reproduce it")
    parts = str(env.get("LD_LIBRARY_PATH", "")).split(":")
    hits = [i for i, part in enumerate(parts)
            if part and os.path.realpath(part) == os.path.realpath(served_build)]
    if not hits or any(not part for part in parts):
        raise Refused(f"served LD_LIBRARY_PATH {env.get('LD_LIBRARY_PATH')!r} does not name "
                      f"the served build bin {served_build}")
    for i in hits:
        parts[i] = str(Path(build) / "bin")
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    env[ssc.CASE_SET_ENV] = ssc.CALIBRATION_CASE_SET_ID
    env[ssc.BACKEND_THREADS_ENV] = str(served_threads)
    return {"env": env, "prefix": prefix, "cpu_list": served_cpus,
            "threads": served_threads, "launch": str(Path(launch_path).resolve()),
            "launch_sha256": _sha256(Path(launch_path))}


def calibration_argv(build: Path, recipe: dict, region_lock: str, lane: str,
                     timeout_s: int = CALIBRATION_TIMEOUT_S) -> list:
    binary = Path(build) / "bin" / "test-backend-ops"
    return [region_lock, "run", "--cpu-list", recipe["cpu_list"], "--role", "bench",
            "--timeout-s", str(timeout_s), "--",
            *recipe["prefix"], str(binary), "test", "-o", "MUL_MAT,MUL_MAT_ID",
            "-b", "CPU", "-p", ssc.calibration_regex(lane)]


def lane_profile_refusal(launch_path: Path, lane: str) -> "str | None":
    """Round-13: the served model in the launch record must BE the lane's profiled
    GGUF, and re-deriving its MoE profile from the header must give the recorded one."""
    try:
        model = Path(json.loads(Path(launch_path).read_text(encoding="utf-8"))["model"]["path"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{launch_path} names no served model: {exc}"
    if ssc.lane_for_model(model) != lane:
        return f"served model {model} is not lane {lane}'s profiled GGUF"
    shards = sorted(model.parent.glob(model.name.replace("00001-of", "*-of"))) or [model]
    try:
        derived = ssc.moe_profile_from_gguf(shards, lane)
    except Exception as exc:  # noqa: BLE001 -- an unreadable header is a refusal
        return f"cannot read {model}'s MoE profile: {type(exc).__name__}: {exc}"
    recorded = ssc.LANE_PROFILES[lane]
    fields = ("expert_count", "expert_used", "hidden", "expert_ff", "shexp_ff")
    diff = {f: (getattr(derived, f), getattr(recorded, f)) for f in fields
            if getattr(derived, f) != getattr(recorded, f)}
    if diff or set(derived.gate_up_types) != set(recorded.gate_up_types) \
            or set(derived.down_types) != set(recorded.down_types):
        return (f"{lane}'s GGUF disagrees with LANE_PROFILES: "
                f"{diff or (derived.gate_up_types, derived.down_types)}")
    return None


def execute(build: Path, store: Path, recipe: dict, region_lock: str, lane: str,
            timeout_s: int = CALIBRATION_TIMEOUT_S, out=sys.stdout) -> Path:
    if not ssc.binary_has_calibration(build):
        raise Refused(f"{build}/bin/test-backend-ops does not carry the calibration block "
                      f"({ssc.CALIBRATION_CASE_SET_ID}) with the backend-thread control; "
                      "stage it with --stage-calibration-patch and rebuild first")
    cpu_list, threads = recipe["cpu_list"], recipe["threads"]
    argv = calibration_argv(build, recipe, region_lock, lane, timeout_s=timeout_s)
    print(f"execute   {' '.join(argv[:12])} ... -p <{len(ssc.calibration_triples(lane))} cases> "
          f"({ssc.BACKEND_THREADS_ENV}={threads})", file=out)
    done = subprocess.run(argv, capture_output=True, text=True, env=recipe["env"],
                          stdin=subprocess.DEVNULL, timeout=timeout_s)
    if done.returncode != 0:
        raise Refused(f"calibration run exited {done.returncode}: "
                      f"{(done.stderr or done.stdout)[-600:]}")
    measurements = ssc.parse_calibration(done.stdout, lane)
    folder = Path(store) / "served_shape"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"calibration-{stamp}.json"
    body = {"schema": "epyc.autokernel.served_shape_calibration.v1",
            "case_set_id": ssc.CASE_SET_ID, "lane": lane,
            "provenance": {**provenance(build, cpu_list, threads, argv, timeout_s=timeout_s),
                           "launch": recipe["launch"], "launch_sha256": recipe["launch_sha256"],
                           "served_env": {k: v for k, v in sorted(recipe["env"].items())
                                          if k != "PATH"}},
            "measurements": [{"shape_name": k[0], "type_a": k[1], "n": k[2], "nmse": v}
                             for k, v in sorted(measurements.items())]}
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    print(f"measured  {len(measurements)} cases -> {path}", file=out)
    return path


def measurement_record_refusal(path: Path, launch: Path, lane: str,
                               region_lock: str) -> "str | None":
    """Round-14: an IMPORTED calibration record (--apply --measurements) may bake bounds
    only if it was measured for THIS lane under THIS served recipe: same lane, same
    launch record (sha256), the same served env, cpu list, threads, topology and case
    argv as the recipe would produce now, the lane's served GGUF, and the calibration
    binary it names still byte-identical. Any mismatch refuses."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        prov = body["provenance"]
        build = Path(prov["anchor_build"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{path} is not a calibration record with provenance: {exc}"
    if body.get("schema") != "epyc.autokernel.served_shape_calibration.v1":
        return f"{path} is not a served-shape calibration record"
    if body.get("lane") != lane:
        return f"{path} was measured for lane {body.get('lane')!r}, not {lane!r}"
    try:
        recipe = served_recipe(launch, build, cpu_list=None, threads=None)
    except Refused as exc:
        return str(exc)
    checks = {
        "launch_sha256": (prov.get("launch_sha256"), recipe["launch_sha256"]),
        "served_env": (prov.get("served_env"),
                       {k: v for k, v in sorted(recipe["env"].items()) if k != "PATH"}),
        "cpu_list": (prov.get("cpu_list"), recipe["cpu_list"]),
        "threads": (prov.get("threads"), recipe["threads"]),
    }
    argv = prov.get("argv") or []
    expected = calibration_argv(build, recipe, region_lock, lane)
    tail = lambda items: list(items[items.index("--") + 1:]) if "--" in items else None
    checks["argv"] = (tail(list(argv)), tail(expected))
    bad = [name for name, (got, want) in checks.items() if got != want]
    if bad:
        return f"{path} does not match the intended served recipe: {', '.join(bad)}"
    binary = build / "bin" / "test-backend-ops"
    if not binary.is_file() or _sha256(binary) != prov.get("binary_digests", {}).get(
            "test-backend-ops"):
        return f"{binary} is not the calibration binary the record measured"
    return lane_profile_refusal(launch, lane)


def load_measurements(path: Path) -> dict:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("schema") != "epyc.autokernel.served_shape_calibration.v1":
        raise Refused(f"{path} is not a served-shape calibration record")
    return {(row["shape_name"], row["type_a"], int(row["n"])): float(row["nmse"])
            for row in body["measurements"]}


def apply(measurements: dict, store: Path, tree: "Path | None", lane: str,
          out=sys.stdout) -> None:
    try:
        cases = ssc.case_set(measurements)
        routed = ssc.case_set(measurements, routed=True, lane=lane)
    except (KeyError, ValueError) as exc:
        raise Refused(f"calibration cannot be baked: {exc}") from exc
    folder = Path(store) / "served_shape"
    folder.mkdir(parents=True, exist_ok=True)
    block = ssc.backend_ops_patch_block(cases, routed)
    (folder / "patch.cpp").write_text(block, encoding="utf-8")
    ssc.write_manifest(folder / "manifest.json", cases)
    ssc.write_manifest(folder / "manifest-routed.json", routed, routed=True, lane=lane)
    print(f"apply     manifest {folder / 'manifest.json'} ({len(cases)} cases), routed "
          f"manifest ({len(routed)} cases), patch {folder / 'patch.cpp'}", file=out)
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
    parser.add_argument("--lane", choices=sorted(ssc.LANE_PROFILES),
                        help="the lane whose GGUF MoE profile the routed corpus derives from")
    parser.add_argument("--launch", type=Path,
                        help="the lane's resolved launch JSON (served env, prefix, -t)")
    parser.add_argument("--timeout-s", type=int, default=CALIBRATION_TIMEOUT_S,
                        help="calibration timeout in seconds (includes region-lock wait time; "
                        "default 3600)")
    parser.add_argument("--stage-calibration-patch", action="store_true")
    parser.add_argument("--build-calibration", action="store_true")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        mutating = (args.stage_calibration_patch or args.build_calibration or args.execute
                    or args.apply)
        if mutating:
            alive = loop_alive_refusal(args.store)
            if alive:
                raise Refused(alive)
        if args.tree is not None and (args.stage_calibration_patch or args.apply
                                      or args.build_calibration):
            frozen = frozen_tree_refusal(args.tree)
            if frozen:
                raise Refused(frozen)
        if (args.stage_calibration_patch or args.execute or args.apply) and not args.lane:
            raise Refused("--lane is required: the routed corpus derives from that lane's "
                          "GGUF expert_count / expert_used_count")
        if args.stage_calibration_patch:
            if args.tree is None:
                raise Refused("--stage-calibration-patch needs --tree")
            try:
                ssc.apply_patch_block(args.tree / "tests" / "test-backend-ops.cpp",
                                      ssc.calibration_patch_block(args.lane))
            except ValueError as exc:
                raise Refused(f"cannot stage the calibration block: {exc}") from exc
            print(f"staged    calibration block into {args.tree}/tests/test-backend-ops.cpp",
                  file=out)
            print("next      --build-calibration --cpu-list <served list> (builds "
                  f"{args.tree}/{CALIBRATION_BUILD_DIRNAME} with the anchor recipe under the "
                  "region lock), then --anchor-build <that dir> --execute --apply", file=out)
            return 0
        if args.build_calibration:
            if args.tree is None or not args.cpu_list:
                raise Refused("--build-calibration needs --tree and --cpu-list")
            build_calibration(args.tree, args.cpu_list, args.region_lock, args.jobs, out=out)
            return 0
        measurements = None
        if args.execute:
            if args.anchor_build is None or args.launch is None:
                raise Refused("--execute needs --anchor-build and --launch (the served "
                              "recipe it must reproduce)")
            recipe = served_recipe(args.launch, args.anchor_build, cpu_list=args.cpu_list,
                                   threads=args.threads)
            profile_refusal = lane_profile_refusal(args.launch, args.lane)
            if profile_refusal:
                raise Refused(profile_refusal)
            path = execute(args.anchor_build, args.store, recipe, args.region_lock,
                           args.lane, timeout_s=args.timeout_s, out=out)
            measurements = load_measurements(path)
        elif args.measurements is not None:
            if args.launch is None or not args.lane:
                raise Refused("--measurements needs --launch and --lane: an imported "
                              "record is validated against the intended served recipe")
            refusal = measurement_record_refusal(args.measurements, args.launch, args.lane,
                                                 args.region_lock)
            if refusal:
                raise Refused(refusal)
            measurements = load_measurements(args.measurements)
        if args.apply:
            if measurements is None:
                raise Refused("--apply needs --execute or --measurements")
            apply(measurements, args.store, args.tree, args.lane, out=out)
            return 0
        if not mutating:
            ready = (args.anchor_build is not None
                     and ssc.binary_has_calibration(args.anchor_build))
            print(f"DRY RUN   store {args.store}; anchor build {args.anchor_build} "
                  f"{'carries' if ready else 'does NOT carry'} the calibration block; "
                  f"{len(ssc.canonical_triples())} candidate cases; region-lock {args.region_lock} "
                  f"--cpu-list {args.cpu_list or '<served list>'} --role bench. "
                  "Pass --stage-calibration-patch / --execute / --apply.", file=out)
        return 0
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
