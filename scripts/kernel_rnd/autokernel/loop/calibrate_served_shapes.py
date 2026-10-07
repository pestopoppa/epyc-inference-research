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
2. `--execute` runs that build's test-backend-ops under ONE correctness-mode CPU
   region-lock claim (`region-lock run --cpu-list <L> --role build` -- region-lock
   always takes an exclusive flock regardless of `--role`, so N per-shard claims would
   SERIALIZE; `-L` defaults to the served list but may be narrowed, since correctness
   work needs no exclusivity, only the served thread count). Inside that ONE claim,
   `--shards N` (default `min(16, cases, max(1, lock cpus // 4))` -- ~4 cores/shard,
   since the single-process run used only ~400% CPU) runs N disjoint
   `test-backend-ops` invocations CONCURRENTLY as background jobs, each confined by
   affinity to the lock's own cpu list (never the full served topology), and merges
   them, parses every case's NMSE, and writes
   `<store>/served_shape/calibration-<utc>.json` with provenance (anchor commit, every
   bin/ file's sha256, cpu list, threads, the one lock argv, the effective
   (affinity-confined) prefix, the per-shard filters and case assignment).
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
import re
import shlex
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from . import served_shape_cases as ssc
from . import status, scratch

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
#: 2026-10-06 coordinator review: region-lock's CLI ALWAYS takes an exclusive flock
#: (fcntl.LOCK_EX) per claimed CPU region -- `--role` is an attribution label, not a
#: shared/exclusive switch (src/runtime/cpu_region_lock.py, region_lock_cli.py never
#: passes `shared=`). N shards each wrapped in their OWN `region-lock run --cpu-list
#: <same list>` therefore SERIALIZE on the same flock and sharding buys nothing. The
#: fix: exactly ONE region-lock claim wraps every shard; the shards run as background
#: jobs of one `bash -c` script executed under that single claim.
CALIBRATION_LOCK_TAG = "ak-served-shape-calibration"
#: taskset/region-lock cpu-list syntax: comma-separated singles or ranges (e.g.
#: "0-95" or "0,2-4,7"). Used to validate a RECORDED lock_cpu_list before trusting it
#: to rebuild the expected lock-claim argv for import replay (Codex Astra review).
CPU_LIST_RE = re.compile(r"^\d+(-\d+)?(,\d+(-\d+)?)*$")


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


@contextmanager
def _calibration_build_scope(tree: Path):
    """A staged build is scratch retained between the tool's separate CLI phases.

    In a loop it belongs to the existing run. Standalone success retains a native
    marker/journal until its owner explicitly retires it; failed staging releases
    immediately. Unmarked or live-owner collisions are refused by the registry.
    """
    if scratch.ambient() is not None:
        yield scratch.current("run") or scratch.active_scope()
        return
    registry = scratch.ScratchRegistry(Path(tree) / ".ak-calib-scratch",
        owner={"campaign": "served-shape-calibration", "state_dir": str(tree)},
        min_free_bytes=max(scratch.DEFAULT_MIN_FREE_BYTES, 100 * scratch.GB), keep="all")
    if not registry.ensure_free():
        raise Refused("calibration staging would violate the native 100 GiB disk safety floor")
    try:
        with registry.scope("run", name="calibration-staging") as scope:
            try:
                yield scope
            except BaseException as exc:
                # A child that could not be verified dead still owns these paths.
                if not isinstance(exc, scratch.ScratchRefused):
                    registry.keep = "none"
                raise
    finally:
        registry.close()


def build_calibration(tree: Path, cpu_list: str, region_lock: str, jobs: int,
                      out=sys.stdout) -> Path:
    """Build test-backend-ops of `tree` (with the staged calibration block) into
    a unique native-owned `<tree>/build-ak-calib-<owner>` slot. Refuses unless the tree's ONLY change is the staged
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
    with _calibration_build_scope(tree) as scope:
        # Previous phases can have readers; a new staging invocation never retires
        # their builds or reuses a foreign fixed build-ak-calib directory.
        target = Path(tree) / (f"{CALIBRATION_BUILD_DIRNAME}-"
                               f"{scope.registry.instance}-{scope.id}")
        if os.path.lexists(target):
            raise Refused(f"{target} already exists; refusing to replace a staged/foreign build")
        build = scope.dir("calibration-build", "staged", at=target)
        argv = build_calibration_argv(tree, build, cpu_list, region_lock, jobs)
        print(f"build     {' '.join(argv[:11])} -- <configure + build test-backend-ops>",
              file=out)
        try:
            with scope.scope("call", name="calibration-build-command") as command_scope:
                temporary_env = command_scope.tmp_env()
                try:
                    with scratch.owned_child_env(command_scope, temporary_env, protect=((scope, (build,)),)) as child_env:
                        done = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                              timeout=BUILD_TIMEOUT_S + 600, env=child_env)
                finally:
                    if not command_scope.retention_reason:
                        command_scope.release(Path(temporary_env["TMPDIR"]))
        except scratch.ScratchRefused:
            scope.retain("calibration command child cleanup could not be verified")
            raise
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


def cpu_list_count(cpu_list: str) -> int:
    """Number of distinct CPUs a taskset/region-lock cpu-list string names (e.g. "0-95"
    -> 96, "0,2-4,7" -> 5). Caller must have already validated the syntax
    (`CPU_LIST_RE`)."""
    total = 0
    for part in cpu_list.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            total += int(hi) - int(lo) + 1
        else:
            total += 1
    return total


def rewrite_prefix_for_lock(prefix: list, lock_cpu_list: str) -> list:
    """Correctness-mode CPU affinity fix (2026-10-06, live-defect follow-up): replace
    the topology prefix's AFFINITY element -- `taskset -c <list>` and/or any
    `numactl --physcpubind=<list>` token -- with `lock_cpu_list`, so shards actually
    run on the cpus the region-lock claims, not the full served topology. Narrowing
    only the lock's claim while leaving the served `taskset -c 0-95` untouched let
    every shard run on all 96 served cores regardless of the lock, trampling quarters
    other sessions hold (the live Q38FN defect: 16 shards x ~100 threads, load avg
    ~1135). The memory POLICY (`--interleave=...`, `--membind=...`) is kept as-is --
    numerics do not depend on NUMA placement, only on the served thread count. Fails
    closed (refuses) when no affinity token is found: a prefix this tool cannot
    confine is not safe to shard."""
    out = list(prefix)
    rewrote = False
    for i, tok in enumerate(out):
        if tok == "taskset" and i + 2 < len(out) and out[i + 1] == "-c":
            out[i + 2] = lock_cpu_list
            rewrote = True
        elif isinstance(tok, str) and tok.startswith("--physcpubind="):
            out[i] = f"--physcpubind={lock_cpu_list}"
            rewrote = True
    if not rewrote:
        raise Refused(f"cannot confine topology prefix {prefix} to the correctness-mode "
                      "lock: no taskset -c <list> or numactl --physcpubind=<list> token "
                      "found (refusing rather than running unconfined)")
    return out


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
    # 2026-10-06 live-defect follow-up: `prefix` (what shard_inner_argv actually runs
    # under) is the AFFINITY-CONFINED prefix -- rewritten to the lock's own cpu list,
    # never the full served topology -- while `served_prefix` keeps the untouched
    # original for provenance/replay. Idempotent when the lock list equals the served
    # one (the common, non-narrowed case): the rewrite is then a no-op.
    confined_prefix = rewrite_prefix_for_lock(prefix, lock_cpu_list)
    return {"env": env, "prefix": confined_prefix, "served_prefix": prefix,
            "served_cpu_list": served_cpus, "lock_cpu_list": lock_cpu_list,
            "cpu_list": served_cpus, "threads": served_threads,
            "launch": str(Path(launch_path).resolve()),
            "launch_sha256": _sha256(Path(launch_path))}


def resolve_shard_count(requested: "int | None", n_cases: int,
                        lock_cpu_list: "str | None" = None) -> int:
    """`--shards N` always overrides (clamped to `[1, n_cases]`). The DEFAULT (no
    --shards) is `min(DEFAULT_SHARDS, cases, max(1, lock_cpu_count // 4))`: the
    single-process run used only ~400% CPU, so ~4 cores per shard is enough --
    defaulting to one shard per core (or even DEFAULT_SHARDS regardless of the lock's
    size) would oversubscribe far past where more shards buys any wall-clock win and
    just adds scheduling noise. `lock_cpu_list` is validated by the caller
    (`CPU_LIST_RE`); omitted (e.g. a caller with no lock cpu list in hand yet) skips
    the cpu-based cap and falls back to the plain cases-based default."""
    if requested is not None and requested < 1:
        raise Refused(f"--shards must be >= 1, got {requested}")
    if requested is not None:
        n = requested
    else:
        n = DEFAULT_SHARDS
        if lock_cpu_list:
            n = min(n, max(1, cpu_list_count(lock_cpu_list) // 4))
    return max(1, min(n, n_cases))


def lock_claim_argv(recipe: dict, region_lock: str, *,
                    timeout_s: int = CALIBRATION_TIMEOUT_S,
                    tag: str = CALIBRATION_LOCK_TAG) -> list:
    """The ONE correctness-mode region-lock claim that wraps EVERY shard (2026-10-06
    coordinator review). `--role build`, not the served topology's `--role bench` --
    NMSE/input-identity/layer-(a) op tests measure correctness, not speed, so
    exclusivity is not required; this still claims load other sessions' TIMING
    measurements must treat as contention (the build-role claim they already account
    for), so the claim is made honestly rather than omitted. `recipe["lock_cpu_list"]`
    may differ from the served topology's cpu list (narrowed via `--cpu-list`); the
    served thread count (`AUTOKERNEL_BACKEND_THREADS`, already in `recipe["env"]`) is
    unaffected and must still equal the served `-t`. Does NOT include the `--`
    terminator or the wrapped command -- the caller appends those."""
    return [region_lock, "run", "--cpu-list", recipe["lock_cpu_list"], "--role", "build",
            "--timeout-s", str(timeout_s), "--tag", tag]


def shard_inner_argv(build: Path, recipe: dict, params_filter: str) -> list:
    """One shard's bare `test-backend-ops -p <subset regex>` invocation, with the
    served topology prefix but NO region-lock wrapper of its own -- region-lock always
    takes an exclusive flock regardless of `--role`, so every shard runs inside the
    SAME single outer claim (`lock_claim_argv`) as a background job, never its own."""
    binary = Path(build) / "bin" / "test-backend-ops"
    return [*recipe["prefix"], str(binary), "test", "-o", "MUL_MAT,MUL_MAT_ID",
            "-b", "CPU", "-p", params_filter]


def shard_inner_argvs(build: Path, recipe: dict, lane: str, n_shards: int) -> list:
    """One inner argv per shard of `ssc.calibration_triples(lane)`, in shard order."""
    groups = ssc.shard_sequence(ssc.calibration_triples(lane), n_shards)
    return [shard_inner_argv(build, recipe, ssc.calibration_regex_for(group))
            for group in groups]


def fan_out_script(inner_argvs: list, out_paths: list, err_paths: list,
                   rc_paths: list) -> str:
    """A `bash -c` script that launches every shard's inner argv as a background job
    (stdout/stderr redirected to its own file), then waits on each job BY PID and
    records its individual exit code -- `wait $pid` returns that job's own status, so
    one slow or crashing shard never corrupts another's result. Runs entirely inside
    ONE region-lock claim; no shard takes a lock of its own."""
    lines = ["set -u"]
    for i, argv in enumerate(inner_argvs):
        cmd = " ".join(shlex.quote(a) for a in argv)
        lines.append(f"({cmd}) >{shlex.quote(str(out_paths[i]))} "
                     f"2>{shlex.quote(str(err_paths[i]))} &")
        lines.append(f"PID_{i}=$!")
    for i in range(len(inner_argvs)):
        lines.append(f'wait "$PID_{i}"')
        lines.append(f"echo $? > {shlex.quote(str(rc_paths[i]))}")
    return "\n".join(lines) + "\n"


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
            timeout_s: int = CALIBRATION_TIMEOUT_S, out=sys.stdout,
            shards: "int | None" = None) -> Path:
    """Sharded, correctness-mode --execute (2026-10-06, revised after coordinator
    review): split `lane`'s calibration corpus into `shards` (default
    `resolve_shard_count`'s `min(DEFAULT_SHARDS, cases, max(1, lock cpus // 4))`)
    disjoint case sets and run them CONCURRENTLY, each confined by AFFINITY to the
    lock's own cpu list (`recipe["prefix"]`, already rewritten by `served_recipe` --
    never the full served topology), as background jobs of ONE `bash -c` script,
    executed under exactly ONE region-lock claim (`lock_claim_argv`, `--role build`)
    -- region-lock always takes an exclusive flock per CPU region regardless of
    `--role`, so N per-shard claims on the same region would SERIALIZE and erase the
    sharding speedup entirely. Every case must appear exactly once in the merge and
    every shard must announce the seed-scheme marker; ANY shard failing (non-zero
    exit, the claim itself failing/timing out, no seed marker, unparseable or
    out-of-assignment output) fails the WHOLE execute and writes nothing -- a partial
    merge would silently understate the corpus a later
    --apply bakes bounds from."""
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
    n_shards = resolve_shard_count(shards, len(triples), lock_cpu_list)
    groups = ssc.shard_sequence(triples, n_shards)
    inner_argvs = [shard_inner_argv(build, recipe, ssc.calibration_regex_for(group))
                  for group in groups]
    lock_argv = lock_claim_argv(recipe, region_lock, timeout_s=timeout_s)
    print(f"execute   ONE region-lock claim ({' '.join(lock_argv)}) wraps {n_shards} "
          f"concurrent shard(s) over {len(triples)} case(s), affinity confined to "
          f"{recipe['prefix']}, {ssc.BACKEND_THREADS_ENV}={threads}", file=out)
    with scratch.registry_for(Path(store) / "scratch").scope("call", name="calibration-shards") as scope:
        tmp = scope.dir("calibration-shards", f"{scope.registry.instance}-{scope.id}")
        out_paths = [tmp / f"shard{i}.out" for i in range(n_shards)]
        err_paths = [tmp / f"shard{i}.err" for i in range(n_shards)]
        rc_paths = [tmp / f"shard{i}.rc" for i in range(n_shards)]
        script = fan_out_script(inner_argvs, out_paths, err_paths, rc_paths)
        full_argv = [*lock_argv, "--", "bash", "-c", script]
        lifecycle_environment = None
        try:
            with scratch.owned_child_env(scope, recipe["env"]) as child_env:
                from .procguard import ENV_SCOPE
                lifecycle_environment = {
                    "schema": "epyc.autokernel.lifecycle_environment.v1",
                    "actual_env_sha256": hashlib.sha256(json.dumps(child_env,
                        sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "injected": {k: v for k, v in child_env.items()
                                 if recipe["env"].get(k) != v}}
                if set(lifecycle_environment["injected"]) - {"TMPDIR", "TMP", "TEMP", ENV_SCOPE}:
                    raise Refused("calibration lifecycle overlay changed a served treatment variable")
                claim = subprocess.run(full_argv, capture_output=True, text=True,
                                       env=child_env, stdin=subprocess.DEVNULL,
                                       timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise Refused(f"the correctness-mode region-lock claim timed out after "
                          f"{timeout_s}s (wraps all {n_shards} shards): {exc}") from exc
        if claim.returncode != 0:
            # Codex Astra review (2026-10-06): a non-zero outer exit must refuse even
            # when every shard happened to record an rc file -- e.g. region-lock itself
            # failing AFTER the wrapped script ran (a late acquisition fault, a
            # region-lock bug, or anything else) must never be read as a passing run.
            raise Refused(
                f"the region-lock claim exited {claim.returncode} (wraps all {n_shards} "
                f"shards); refusing even though shard exit codes were recorded -- an "
                f"outer claim failure is never evidence of a passing run: "
                f"{(claim.stderr or claim.stdout)[-600:]}")
        missing = [i for i in range(n_shards) if not rc_paths[i].is_file()]
        if missing:
            raise Refused(
                f"the region-lock claim exited {claim.returncode} before "
                f"{len(missing)}/{n_shards} shard(s) recorded an exit code (lock not "
                f"acquired, or the fan-out script crashed before launching every "
                f"shard): {(claim.stderr or claim.stdout)[-600:]}")
        errors = []
        per_shard: dict = {}
        for i in range(n_shards):
            rc_text = rc_paths[i].read_text(encoding="utf-8").strip()
            stdout_i = out_paths[i].read_text(encoding="utf-8", errors="replace")
            stderr_i = err_paths[i].read_text(encoding="utf-8", errors="replace")
            try:
                returncode = int(rc_text)
            except ValueError:
                errors.append(f"shard {i}: unreadable exit code {rc_text!r}")
                continue
            if returncode != 0:
                errors.append(f"shard {i} exited {returncode}: "
                              f"{(stderr_i or stdout_i)[-400:]}")
                continue
            if ssc.SEED_MARKER not in stderr_i.splitlines():
                errors.append(f"shard {i} did not announce {ssc.SEED_MARKER}: the cases did "
                              "not run through the seeded subclasses")
                continue
            try:
                per_shard[i] = ssc.parse_calibration(stdout_i, lane, triples=groups[i])
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
        missing_cases = sorted(expected - set(measurements))[:3]
        raise Refused(f"merged shards cover {len(measurements)}/{len(expected)} case(s); "
                      f"missing e.g. {missing_cases}")
    folder = Path(store) / "served_shape"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"calibration-{stamp}.json"
    body = {"schema": "epyc.autokernel.served_shape_calibration.v2",
            "case_set_id": ssc.CASE_SET_ID, "lane": lane, "seed_scheme": ssc.SEED_SCHEME,
            "partition": [shape.name for shape in ssc.lane_served_shapes(lane)],
            "provenance": {**provenance(build, recipe["served_cpu_list"], threads, lock_argv,
                                        timeout_s=timeout_s),
                           "launch": recipe["launch"], "launch_sha256": recipe["launch_sha256"],
                           "served_env": {k: v for k, v in sorted(recipe["env"].items())
                                          if k != "PATH"},
                           "lifecycle_environment": lifecycle_environment,
                           "lock_cpu_list": lock_cpu_list, "lock_role": "build",
                           "effective_prefix": recipe["prefix"],
                           "shards": n_shards,
                           "shard_params_filters": [a[-1] for a in inner_argvs],
                           "shard_assignment": {str(i): [ssc.case_key(*t) for t in group]
                                                for i, group in enumerate(groups)}},
            "measurements": [{"shape_name": k[0], "type_a": k[1], "n": k[2], "nmse": v}
                             for k, v in sorted(measurements.items())]}
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    print(f"measured  {len(measurements)} case(s) across {n_shards} shard(s) under ONE "
          f"region-lock claim -> {path}", file=out)
    return path


def measurement_record_refusal(path: Path, launch: Path, lane: str,
                               region_lock: str) -> "str | None":
    """Round-14: an IMPORTED calibration record (--apply --measurements) may bake bounds
    only if it was measured for THIS lane under THIS served recipe: same lane, same
    launch record (sha256), the same served env, served cpu list, threads, the ONE
    region-lock claim argv and the per-shard filter list the recipe would produce now,
    the lane's served GGUF, and the calibration binary it names still byte-identical.
    Any mismatch refuses.

    2026-10-06: the schema moved to v2 (sharded --execute under exactly ONE region-lock
    claim -- `shards`/`shard_params_filters`/`shard_assignment` provenance, `argv` the
    single lock-claim argv); a pre-sharding v1 record is refused with a distinct, clear
    message rather than silently misread (no valid seeded v1 record exists to
    migrate)."""
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
    # Codex Astra review (2026-10-06): rebuild the recipe's LOCK cpu list from the
    # RECORD's own recorded lock_cpu_list (which may be narrowed or disjoint from the
    # served topology, by design -- correctness mode needs no exclusivity), not from
    # the served topology's default. Using `cpu_list=None` here would silently default
    # back to the served list and refuse every honestly-narrowed-lock record forever.
    # Still validated as a cpu-list before being trusted.
    lock_cpu_list = prov.get("lock_cpu_list")
    if not isinstance(lock_cpu_list, str) or not CPU_LIST_RE.fullmatch(lock_cpu_list):
        return f"{path} carries no valid lock_cpu_list in provenance: {lock_cpu_list!r}"
    try:
        recipe = served_recipe(launch, build, cpu_list=lock_cpu_list, threads=None)
    except Refused as exc:
        return str(exc)
    n_shards = prov.get("shards")
    if not isinstance(n_shards, int) or n_shards < 1:
        return f"{path} carries no valid 'shards' count in provenance"
    if prov.get("lock_role") != "build":
        return f"{path} was not measured under the correctness-mode --role build claim"
    expected_lock_argv = lock_claim_argv(recipe, region_lock, timeout_s=prov.get(
        "timeout_s", CALIBRATION_TIMEOUT_S))
    expected_filters = [a[-1] for a in shard_inner_argvs(build, recipe, lane, n_shards)]
    checks = {
        "launch_sha256": (prov.get("launch_sha256"), recipe["launch_sha256"]),
        "served_env": (prov.get("served_env"),
                       {k: v for k, v in sorted(recipe["env"].items()) if k != "PATH"}),
        "cpu_list": (prov.get("cpu_list"), recipe["served_cpu_list"]),
        "threads": (prov.get("threads"), recipe["threads"]),
        "argv": (prov.get("argv"), expected_lock_argv),
        "shard_params_filters": (prov.get("shard_params_filters"), expected_filters),
    }
    bad = [name for name, (got, want) in checks.items() if got != want]
    # Live-defect follow-up (2026-10-06): records from BEFORE the affinity fix carry no
    # "effective_prefix" at all (they ran unconfined, on the served prefix) and must
    # stay importable -- affinity never affects numerics. Accept either the served
    # prefix or the lock-confined one; default an absent field to the served prefix.
    effective_prefix = prov.get("effective_prefix")
    if effective_prefix is None:
        effective_prefix = recipe["served_prefix"]
    if list(effective_prefix) not in (list(recipe["served_prefix"]), list(recipe["prefix"])):
        bad.append("effective_prefix")
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
                        help=f"--execute: disjoint test-backend-ops processes, each "
                        f"affinity-confined to the lock's own cpu list, run concurrently "
                        f"under correctness mode (default min({DEFAULT_SHARDS}, cases, "
                        "max(1, lock cpus // 4)) -- ~4 cores/shard); correctness "
                        "measures NMSE, not speed, so a quiet/exclusive host is not "
                        "required")
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
                  f"a unique {args.tree}/{CALIBRATION_BUILD_DIRNAME}-<owner> with the anchor recipe under the "
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
            shard_note = (str(args.shards) if args.shards
                         else f"min({DEFAULT_SHARDS}, cases, lock cpus // 4), each "
                         "affinity-confined to the lock cpu list")
            print(f"DRY RUN   store {args.store}; anchor build {args.anchor_build} "
                  f"{'carries' if ready else 'does NOT carry'} the calibration block; "
                  f"{len(ssc.canonical_triples())} candidate cases; correctness-mode "
                  f"region-lock {args.region_lock} --cpu-list "
                  f"{args.cpu_list or '<served list>'} --role build, "
                  f"{shard_note} shard(s) (speed is NOT measured here). "
                  "Pass --stage-calibration-patch / --execute / --apply.", file=out)
        return 0
    except (Refused, scratch.ScratchRefused) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
