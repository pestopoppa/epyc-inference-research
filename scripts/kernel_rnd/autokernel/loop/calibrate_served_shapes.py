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
from concurrent.futures import ThreadPoolExecutor
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
#: Correctness mode (NMSE / input-identity / layer-(a) op tests) measures CORRECTNESS,
#: not speed -- it needs the served thread count and deterministic per-case-key seeded
#: inputs (ac97318f), not a quiet or exclusive host. Default shard count for --execute:
#: one test-backend-ops process per shard, run concurrently under oversubscription.
DEFAULT_SHARDS = 16


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
    topology prefix, its served cpu list and its -t. Refuses when any of these is
    missing, when --threads disagrees with the served -t, or when a loader variable
    other than LD_LIBRARY_PATH is set (the served environment could not be reproduced).

    2026-10-06 correctness-mode lock: `--threads` must still equal the served -t (the
    one hard requirement a correctness oracle has on CPU occupancy), but `--cpu-list`
    no longer has to equal the served topology's cpu list -- NMSE / input-identity /
    layer-(a) op tests measure correctness, not speed, so the host region they CLAIM
    (`recipe["lock_cpu_list"]`, region-lock's `--cpu-list` under `--role build`) may be
    narrower than, or disjoint from, where the model is actually served
    (`recipe["served_cpu_list"]`, still used for provenance and import replay).
    Oversubscription of the served cpu list is fine; exclusivity is not required."""
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
    lock_cpu_list = cpu_list if cpu_list is not None else served_cpus
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
    return {"env": env, "prefix": prefix, "served_cpu_list": served_cpus,
            "lock_cpu_list": lock_cpu_list, "cpu_list": served_cpus,
            "threads": served_threads, "launch": str(Path(launch_path).resolve()),
            "launch_sha256": _sha256(Path(launch_path))}


def resolve_shard_count(requested: "int | None", n_cases: int) -> int:
    """`--shards N` (default `min(DEFAULT_SHARDS, cases)`), clamped to `[1, n_cases]`."""
    if requested is not None and requested < 1:
        raise Refused(f"--shards must be >= 1, got {requested}")
    n = requested if requested is not None else min(DEFAULT_SHARDS, n_cases)
    return max(1, min(n, n_cases))


def shard_argv(build: Path, recipe: dict, region_lock: str, params_filter: str, *,
              shard_index: int, timeout_s: int = CALIBRATION_TIMEOUT_S) -> list:
    """One shard's `region-lock run --role build` wrapping its own `test-backend-ops -p
    <subset regex>`. Correctness mode (2026-10-06): the region-lock claim is `--role
    build`, not the served topology's `--role bench` -- NMSE/input-identity/layer-(a)
    op tests measure correctness, not speed, so exclusivity is not required; this still
    claims load other sessions' TIMING measurements must treat as contention (the
    build-role claim they already account for), so the claim is made honestly rather
    than omitted. `recipe["lock_cpu_list"]` may differ from the served topology's cpu
    list (narrowed via `--cpu-list`); the served thread count
    (`AUTOKERNEL_BACKEND_THREADS`, already in `recipe["env"]`) is unaffected and must
    still equal the served `-t` -- oversubscription of the lock's cpu list is fine for
    a correctness-only run."""
    binary = Path(build) / "bin" / "test-backend-ops"
    return [region_lock, "run", "--cpu-list", recipe["lock_cpu_list"], "--role", "build",
            "--timeout-s", str(timeout_s), "--tag",
            f"ak-served-shape-calibration-shard{shard_index}", "--",
            *recipe["prefix"], str(binary), "test", "-o", "MUL_MAT,MUL_MAT_ID",
            "-b", "CPU", "-p", params_filter]


def shard_argvs(build: Path, recipe: dict, region_lock: str, lane: str, n_shards: int,
                timeout_s: int = CALIBRATION_TIMEOUT_S) -> list:
    """One argv per shard of `ssc.calibration_triples(lane)`, in shard order."""
    groups = ssc.shard_sequence(ssc.calibration_triples(lane), n_shards)
    return [shard_argv(build, recipe, region_lock, ssc.calibration_regex_for(group),
                       shard_index=i, timeout_s=timeout_s)
            for i, group in enumerate(groups)]


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


def _run_one_shard(argv: list, env: dict, timeout_s: int) -> tuple:
    """Run one shard's argv; returns (returncode_or_None, stdout, stderr, error) where
    `error` is a timeout/OS message (argv never ran to completion) or None."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, env=env,
                              stdin=subprocess.DEVNULL, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, "", "", f"timed out after {timeout_s}s"
    except OSError as exc:
        return None, "", "", f"could not launch: {exc}"
    return done.returncode, done.stdout, done.stderr, None


def execute(build: Path, store: Path, recipe: dict, region_lock: str, lane: str,
            timeout_s: int = CALIBRATION_TIMEOUT_S, out=sys.stdout,
            shards: "int | None" = None) -> Path:
    """Sharded, correctness-mode --execute (2026-10-06): split `lane`'s calibration
    corpus into `shards` (default `min(DEFAULT_SHARDS, cases)`) disjoint case sets, run
    one `test-backend-ops` process per shard CONCURRENTLY under the correctness-mode
    region-lock claim (`shard_argv`, `--role build`), then merge. Every case must
    appear exactly once in the merge and every shard must announce the seed-scheme
    marker; ANY shard failing (non-zero exit, timeout, no seed marker, unparseable or
    out-of-assignment output) fails the WHOLE execute and writes nothing -- a partial
    merge would silently understate the corpus a later --apply bakes bounds from."""
    if not ssc.binary_has_seed_scheme(build):
        raise Refused(f"{build}/bin/test-backend-ops {ssc.SEED_REBUILD_HINT}; re-stage with "
                      "--stage-calibration-patch and rebuild with --build-calibration")
    if not ssc.binary_has_calibration(build):
        raise Refused(f"{build}/bin/test-backend-ops does not carry the calibration block "
                      f"({ssc.CALIBRATION_CASE_SET_ID}) with the backend-thread control; "
                      "stage it with --stage-calibration-patch and rebuild first")
    lock_cpu_list, threads = recipe["lock_cpu_list"], recipe["threads"]
    binary = Path(build) / "bin" / "test-backend-ops"
    digest_before = _sha256(binary)
    triples = ssc.calibration_triples(lane)
    n_shards = resolve_shard_count(shards, len(triples))
    groups = ssc.shard_sequence(triples, n_shards)
    argvs = [shard_argv(build, recipe, region_lock, ssc.calibration_regex_for(group),
                        shard_index=i, timeout_s=timeout_s)
            for i, group in enumerate(groups)]
    print(f"execute   {n_shards} shard(s) over {len(triples)} case(s), region-lock "
          f"--cpu-list {lock_cpu_list} --role build, {ssc.BACKEND_THREADS_ENV}={threads}",
          file=out)
    with ThreadPoolExecutor(max_workers=n_shards) as pool:
        raw = list(pool.map(
            lambda i: (i, *_run_one_shard(argvs[i], recipe["env"], timeout_s)),
            range(n_shards)))
    errors = []
    per_shard: dict = {}
    for i, returncode, stdout, stderr, error in raw:
        if error is not None:
            errors.append(f"shard {i}: {error}")
            continue
        if returncode != 0:
            errors.append(f"shard {i} exited {returncode}: {(stderr or stdout)[-400:]}")
            continue
        if ssc.SEED_MARKER not in (stderr or "").splitlines():
            errors.append(f"shard {i} did not announce {ssc.SEED_MARKER}: the cases did "
                          "not run through the seeded subclasses")
            continue
        try:
            per_shard[i] = ssc.parse_calibration(stdout, lane, triples=groups[i])
        except ValueError as exc:
            errors.append(f"shard {i}: {exc}")
    digest_after = _sha256(binary)
    if digest_after != digest_before:
        raise Refused(f"{binary} changed digest while shards were running "
                      f"({digest_before[:12]} -> {digest_after[:12]}); the shards did not "
                      "all measure the SAME binary, recording nothing")
    if errors:
        raise Refused(f"{len(errors)}/{n_shards} calibration shard(s) failed, recording "
                      "nothing (rebuild with --build-calibration if this is a stale "
                      "binary): " + "; ".join(errors))
    measurements: dict = {}
    for i in range(n_shards):
        overlap = set(per_shard[i]) & set(measurements)
        if overlap:
            raise Refused(f"shard {i} measured case(s) another shard already measured "
                          f"(non-disjoint shard assignment): {sorted(overlap)[:3]}")
        measurements.update(per_shard[i])
    expected = {(t[0].name, t[1], t[2]) for t in triples}
    if set(measurements) != expected:
        missing = sorted(expected - set(measurements))[:3]
        raise Refused(f"merged shards cover {len(measurements)}/{len(expected)} case(s); "
                      f"missing e.g. {missing}")
    folder = Path(store) / "served_shape"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"calibration-{stamp}.json"
    body = {"schema": "epyc.autokernel.served_shape_calibration.v2",
            "case_set_id": ssc.CASE_SET_ID, "lane": lane, "seed_scheme": ssc.SEED_SCHEME,
            "partition": [shape.name for shape in ssc.lane_served_shapes(lane)],
            "provenance": {**provenance(build, recipe["served_cpu_list"], threads, argvs,
                                        timeout_s=timeout_s),
                           "launch": recipe["launch"], "launch_sha256": recipe["launch_sha256"],
                           "served_env": {k: v for k, v in sorted(recipe["env"].items())
                                          if k != "PATH"},
                           "lock_cpu_list": lock_cpu_list, "shards": n_shards,
                           "shard_assignment": {str(i): [ssc.case_key(*t) for t in group]
                                                for i, group in enumerate(groups)}},
            "measurements": [{"shape_name": k[0], "type_a": k[1], "n": k[2], "nmse": v}
                             for k, v in sorted(measurements.items())]}
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    print(f"measured  {len(measurements)} case(s) across {n_shards} shard(s) -> {path}",
          file=out)
    return path


def measurement_record_refusal(path: Path, launch: Path, lane: str,
                               region_lock: str) -> "str | None":
    """Round-14: an IMPORTED calibration record (--apply --measurements) may bake bounds
    only if it was measured for THIS lane under THIS served recipe: same lane, same
    launch record (sha256), the same served env, served cpu list, threads, topology and
    (per-shard) case argv as the recipe would produce now, the lane's served GGUF, and
    the calibration binary it names still byte-identical. Any mismatch refuses.

    2026-10-06: the schema moved to v2 (sharded --execute, `shards`/`shard_assignment`
    provenance, `argv` as one list per shard); a pre-sharding v1 record is refused with
    a distinct, clear message rather than silently misread (no valid seeded v1 record
    exists to migrate)."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        prov = body["provenance"]
        build = Path(prov["anchor_build"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{path} is not a calibration record with provenance: {exc}"
    schema = body.get("schema")
    if schema == "epyc.autokernel.served_shape_calibration.v1":
        return (f"{path} is schema v1 (pre-sharding, single-process --execute); the "
                "sharded v2 format is required and no valid seeded v1 record exists to "
                "migrate -- re-calibrate")
    if schema != "epyc.autokernel.served_shape_calibration.v2":
        return f"{path} is not a served-shape calibration record"
    if body.get("lane") != lane:
        return f"{path} was measured for lane {body.get('lane')!r}, not {lane!r}"
    if body.get("seed_scheme") != ssc.SEED_SCHEME:
        return (f"{path} was measured with input-seed scheme {body.get('seed_scheme')!r}, "
                f"not {ssc.SEED_SCHEME!r} (random-input records cannot bound seeded runs; "
                "re-calibrate)")
    if body.get("partition") != [shape.name for shape in ssc.lane_served_shapes(lane)]:
        return f"{path} does not carry lane {lane}'s candidate partition"
    try:
        recipe = served_recipe(launch, build, cpu_list=None, threads=None)
    except Refused as exc:
        return str(exc)
    n_shards = prov.get("shards")
    if not isinstance(n_shards, int) or n_shards < 1:
        return f"{path} carries no valid 'shards' count in provenance"
    checks = {
        "launch_sha256": (prov.get("launch_sha256"), recipe["launch_sha256"]),
        "served_env": (prov.get("served_env"),
                       {k: v for k, v in sorted(recipe["env"].items()) if k != "PATH"}),
        "cpu_list": (prov.get("cpu_list"), recipe["served_cpu_list"]),
        "threads": (prov.get("threads"), recipe["threads"]),
    }
    argv = prov.get("argv") or []
    try:
        expected = shard_argvs(build, recipe, region_lock, lane, n_shards)
    except Refused as exc:
        return str(exc)
    tail = lambda items: list(items[items.index("--") + 1:]) if "--" in items else None
    got_tails = [tail(list(a)) for a in argv] if isinstance(argv, list) else None
    want_tails = [tail(a) for a in expected]
    checks["argv"] = (got_tails, want_tails)
    bad = [name for name, (got, want) in checks.items() if got != want]
    if bad:
        return f"{path} does not match the intended served recipe: {', '.join(bad)}"
    binary = build / "bin" / "test-backend-ops"
    if not binary.is_file() or _sha256(binary) != prov.get("binary_digests", {}).get(
            "test-backend-ops"):
        return f"{binary} is not the calibration binary the record measured"
    if not ssc.binary_has_seed_scheme(build):
        # Round-17: the record's own binary predates seeding (an execute against a stale
        # build-ak-calib) -- its seed_scheme label is not evidence.
        return f"{binary} {ssc.SEED_REBUILD_HINT}; rebuild with --build-calibration"
    return lane_profile_refusal(launch, lane)


def load_measurements(path: Path) -> dict:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("schema") != "epyc.autokernel.served_shape_calibration.v2":
        raise Refused(f"{path} is not a v2 (sharded) served-shape calibration record")
    return {(row["shape_name"], row["type_a"], int(row["n"])): float(row["nmse"])
            for row in body["measurements"]}


def apply(measurements: dict, store: Path, tree: "Path | None", lane: str,
          out=sys.stdout, anchor_exceeds_generic: str = "refuse") -> None:
    over = ssc.anchor_exceeds_generic(measurements)
    relative = over if anchor_exceeds_generic == "anchor-relative" else frozenset()
    if over:
        print(f"anchor    {len(over)} case(s) where the ANCHOR's NMSE is >= the generic "
              f"bound {ssc.SERVED_SHAPE_NMSE_CAP}: "
              + ", ".join(f"{k[0]}/{k[1]}/n={k[2]}={measurements[k]:.3g}"
                          for k in sorted(over))
              + (" -> held anchor-relative (factor x anchor)" if relative
                 else " -> refusing (pass --anchor-exceeds-generic anchor-relative to "
                      "hold them to factor x the anchor instead)"), file=out)
    try:
        cases = ssc.case_set(measurements, lane=lane, anchor_relative_keys=relative)
        routed = ssc.case_set(measurements, routed=True, lane=lane,
                              anchor_relative_keys=relative)
    except (KeyError, ValueError) as exc:
        raise Refused(f"calibration cannot be baked: {exc}") from exc
    folder = Path(store) / "served_shape"
    folder.mkdir(parents=True, exist_ok=True)
    block = ssc.backend_ops_patch_block(cases, routed)
    (folder / "patch.cpp").write_text(block, encoding="utf-8")
    ssc.write_manifest(folder / "manifest.json", cases, lane=lane,
                       anchor_relative_keys=relative)
    ssc.write_manifest(folder / "manifest-routed.json", routed, routed=True, lane=lane,
                       anchor_relative_keys=relative)
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
    parser.add_argument("--anchor-exceeds-generic", choices=("refuse", "anchor-relative"),
                        default="refuse",
                        help="cases where the anchor's own NMSE is >= the generic 5e-4 bound: "
                             "refuse to bake (default) or hold them to factor x the anchor")
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
    parser.add_argument("--shards", type=int,
                        help=f"--execute: disjoint test-backend-ops processes run "
                        f"concurrently under correctness mode (default "
                        f"min({DEFAULT_SHARDS}, cases)); correctness measures NMSE, not "
                        "speed, so a quiet/exclusive host is not required")
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
                           args.lane, timeout_s=args.timeout_s, out=out, shards=args.shards)
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
            apply(measurements, args.store, args.tree, args.lane, out=out,
                  anchor_exceeds_generic=args.anchor_exceeds_generic)
            return 0
        if not mutating:
            ready = (args.anchor_build is not None
                     and ssc.binary_has_calibration(args.anchor_build))
            print(f"DRY RUN   store {args.store}; anchor build {args.anchor_build} "
                  f"{'carries' if ready else 'does NOT carry'} the calibration block; "
                  f"{len(ssc.canonical_triples())} candidate cases; correctness-mode "
                  f"region-lock {args.region_lock} --cpu-list "
                  f"{args.cpu_list or '<served list>'} --role build, "
                  f"{args.shards or DEFAULT_SHARDS} shard(s) (speed is NOT measured here). "
                  "Pass --stage-calibration-patch / --execute / --apply.", file=out)
        return 0
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
