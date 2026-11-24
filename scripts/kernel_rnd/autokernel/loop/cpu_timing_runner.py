"""RSL-R6 live CPU timing runner: llama-bench tg128 for a CPU path on that host's SERVED topology.

Placement is never typed here: cpus, threads, numactl policy and the instance list come from the registry path spec
(`spec["serving"]["numa_instances"|"numa_ports"]`, host name) joined with `stack_numa` (`NUMA_CONFIG`, `_numa_prefix`).
Q38FN :8074 is one NUMA_FULL instance; A3B :8070 is two NUMA-half instances, launched CONCURRENTLY (the served
condition); a sample is the AGGREGATE tok/s over the instances, one process (pair) per sample.

Floors: a `cpu_aa_calibration` record keyed per model AND topology, never borrowed from the GPU `bench.floor_rows` file
(the same GGUF has a different floor on a different device). `calibrate` writes it from an A/A run (a row runs it).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import bench, cpu_aa_calibration as aa, serving
from .floor_units import UNIT_PROCESS
from .op_perf_runner import RunnerRefused, STORE, base_parser, emit, prove_binaries

STACK_NUMA = Path("/workspace/repos/epyc-orchestrator/scripts/server/stack_numa.py")
SURFACE = "TG128"
TIMEOUT_S = 3600


def _stack_numa():
    sys.path.insert(0, str(STACK_NUMA.parent))
    return __import__("stack_numa")


def topology(spec: Mapping, sn: Any = None) -> list[dict]:
    """One dict per served instance of the path's host: idx, port, cpus, threads, numactl+taskset prefix."""
    serving_ = spec.get("serving")
    if not serving_ or not spec.get("host"):
        raise RunnerRefused("FUTILITY_HARNESS_MISSING_INPUT", f"{spec.get('key')} has no served host (planned path): no topology")
    sn = sn or _stack_numa()
    insts = (sn.NUMA_CONFIG.get(spec["host"]) or {}).get("instances") or []
    ports = list(serving_.get("numa_ports") or [serving_.get("port")])
    idx = [next((i for i, x in enumerate(insts) if x[1] == p), None) for p in ports]
    if None in idx or len(ports) != int(serving_.get("numa_instances") or len(ports)):
        raise RunnerRefused("FUTILITY_HARNESS_MISSING_INPUT", f"{spec['host']}: registry ports {ports} not all in stack_numa instances")
    return [{"idx": i, "port": insts[i][1], "cpus": insts[i][0], "threads": insts[i][2], "prefix": sn._numa_prefix(spec["host"], i)}
            for i in idx]


def recipe_env(spec: Mapping) -> dict[str, str]:
    """The recipe's env: the registry recipe block, else the two-instance recipe for this GGUF; empty is stated, not guessed."""
    env = ((spec.get("serving") or {}).get("recipe") or {}).get("env")
    if env is None:
        from . import two_instance_floor as tif
        env = next((dict(r.env) for r in tif.RECIPES.values() if r.model == spec["gguf"]), {})
    return {str(k): str(v) for k, v in env.items()}


def bench_argv(binary: Path, spec: Mapping, inst: Mapping, *, reps: int, seed: int) -> list[str]:
    kv = (((spec["serving"].get("serving_shape") or {}).get("kv_quant") or {}).get("k")) or bench.KV_DEFAULT
    return [*inst["prefix"], str(binary), "-m", spec["gguf"], "-p", "0", "-n", "128", "-r", str(reps), "-t", str(inst["threads"]),
            "-ngl", "0", "-fa", "1", "-o", "json", "--autokernel-harden", str(seed), "-ctk", kv, "-ctv", kv]


def recipe_name(spec: Mapping, topo: Sequence[Mapping]) -> str:
    return f"cpu-tg128.{Path(spec['gguf']).stem}.{len(topo)}inst-" + "-".join(f"{t['threads']}t" for t in topo)


def floor_pct(spec: Mapping, store: Path, pairs: int) -> float | None:
    """The quiet A/A floor (percent) at `pairs` for this model+topology, or None (ungated). Never borrowed."""
    try:
        rec = json.loads((Path(store) / "calibration" / f"cpu-aa-calibration.{serving.floor_key(recipe_name(spec, topology(spec)))}.quiet.json").read_text())
        s = rec["surfaces"][SURFACE]
        hit = [f["floor_pct"] for f in (s["screen_floor"], s["serving_floor"]) if f["pairs"] == pairs]
        ok = rec["schema"] == aa.SCHEMA and rec["unit"] == UNIT_PROCESS and s["n"] >= aa.MIN_PAIRS and s["gating"]
        return float(hit[0]) if hit and ok else None
    except (OSError, ValueError, KeyError, RunnerRefused):
        return None


def sample(binary: Path, spec: Mapping, topo: Sequence[Mapping], *, reps: int, seed: int, runner=subprocess.run) -> float:
    """One launch of every instance, concurrently; aggregate tok/s. Linkage is proven by `prove` before the first launch."""
    env = {**os.environ, "LD_LIBRARY_PATH": f"{binary.parent}", **recipe_env(spec)}

    def one(inst):
        done = runner(bench_argv(binary, spec, inst, reps=reps, seed=seed), capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
        if done.returncode:
            raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", f"llama-bench rc={done.returncode}: {done.stderr[-200:]}")
        try:
            return bench.hardened_row(done.stdout, pp=0, tg=128, reps=reps).avg_ts
        except bench.BenchFailed as e:
            raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", str(e))
    with cf.ThreadPoolExecutor(max_workers=len(topo)) as ex:
        return float(sum(f.result() for f in [ex.submit(one, i) for i in topo]))


def collect(a_bin: Path, c_bin: Path, spec, topo, *, pairs: int, reps: int = 9, runner=subprocess.run, live=True) -> dict:
    """Warm-up pair(s) discarded, then ABBA: the arm order alternates per pair. Returns samples + pre-launch busy readings."""
    from ..controller import host_idle
    if live:
        prove_binaries([a_bin, c_bin], literal=None, ld_tail="")
    out: dict[str, list] = {"anchor": [], "candidate": [], "busy": []}
    bins, cpus = {"anchor": a_bin, "candidate": c_bin}, ",".join(t["cpus"] for t in topo)
    for i in range(-bench.WARMUP_PAIRS, pairs):
        for role in (("anchor", "candidate") if i % 2 == 0 else ("candidate", "anchor")):
            idle = host_idle.observe("pre", cpus) if live else {"busy_pct": 0.0}
            v = sample(bins[role], spec, topo, reps=reps, seed=1, runner=runner)
            if i >= 0:
                out[role].append(v)
                out["busy"].append(idle.get("busy_pct"))
    return out


def time_path(cand: Mapping, a_build: Path, c_build: Path, spec: Mapping, *, store: Path = STORE, runner=subprocess.run,
              live=True, floor: float | None = None) -> dict:
    """The `Primitives.timing` result for a CPU path (effect in PERCENT, higher tok/s = positive)."""
    pairs = max(int(cand.get("pairs") or bench.MIN_PAIRS), bench.MIN_PAIRS)
    topo = topology(spec) if live else cand["test_topology"]
    nf = floor_pct(spec, store, pairs) if floor is None else floor
    if nf is None:
        raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", f"no quiet A/A floor at {pairs} pairs for {recipe_name(spec, topo)}")
    s = collect(Path(a_build) / "bin" / "llama-bench", Path(c_build) / "bin" / "llama-bench", spec, topo, pairs=pairs,
                reps=int(cand.get("reps") or 9), runner=runner, live=live)
    c = bench.Comparison(surface=spec["surface"], anchor_samples=s["anchor"], candidate_samples=s["candidate"],
                         effect=st.median(s["candidate"]) / st.median(s["anchor"]) - 1.0, estimator="median_over_median",
                         pairs=pairs, noise_floor_pct=nf, residency={}, model=spec["gguf"])
    return {"effect_pct": c.effect * 100.0, "pairs": pairs, "decisive": c.decisive, "noise_floor_pct": nf, "device_seconds": 0.0,
            "anchor_samples": s["anchor"], "candidate_samples": s["candidate"], "model": spec["gguf"],
            "topology": recipe_name(spec, topo), "foreign_busy_pct_max": max(b or 0.0 for b in s["busy"])}


def calibrate(build: Path, spec: Mapping, *, source_row: str, pairs: int = aa.MIN_PAIRS, store: Path = STORE, runner=subprocess.run,
              live=True, label: str = "quiet", topo: Sequence[Mapping] | None = None, reps: int = 9) -> Path:
    """A/A (both arms the same llama-bench): writes cpu-aa-calibration.<recipe key>.<label>.json under <store>/calibration."""
    topo = topo or topology(spec)
    b = Path(build) / "bin" / "llama-bench"
    s = collect(b, b, spec, topo, pairs=pairs, reps=reps, runner=runner, live=live)
    if any(v is None for v in s["busy"]):
        raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", "an unreadable foreign-load reading: the contention label cannot be proven")
    env = json.dumps(dict(sorted(recipe_env(spec).items())))
    digest = hashlib.sha256(b.read_bytes()).hexdigest() if live else "test-mode"
    state = {"store_digest": digest, "cpu_list": ";".join(t["cpus"] for t in topo), "threads": sum(t["threads"] for t in topo),
             "numa": ";".join(" ".join(t["prefix"][:2]) for t in topo), "recipe_env_digest": hashlib.sha256(env.encode()).hexdigest()[:16]}
    try:
        rec = aa.build_record(recipe=recipe_name(spec, topo), recipe_hash=digest, store_digest=digest, unit=UNIT_PROCESS, host_state=state,
                              contention_label=label, foreign_busy_p95=s["busy"], source_row=source_row, serving_pairs=bench.MIN_PAIRS,
                              surfaces={SURFACE: {"anchor": s["anchor"], "candidate": s["candidate"], "gating": True}})
        return aa.write(Path(store) / "calibration", rec)
    except aa.CpuAACalibrationError as e:
        raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", str(e))


def synth_runner(outcome: str = "pass", base: float = 20.0):
    """--test-mode: the RECORDED real llama-bench receipt (execution/testdata, hardened fields as test_microbench adds them),
    avg_ts rescaled; NO binary runs. The candidate arm's path must contain `cand`. outcome: pass | flat | harness | bad-rc."""
    from ..execution.test_microbench import CANONICAL, read_fixture
    calls = [0]

    def runner(argv, **kw):
        calls[0] += 1
        cand = "cand" in str(next(a for a in argv if a.endswith("llama-bench")))
        rows = json.loads(read_fixture(CANONICAL))
        rows[0]["avg_ts"] = base * (1.0 + 0.004 * ((calls[0] * 7) % 5 - 2)) * (1.05 if cand and outcome == "pass" else 1.0)
        return subprocess.CompletedProcess(argv, 3 if outcome == "bad-rc" else 0, "{}" if outcome == "harness" else json.dumps(rows), "boom")
    return runner


def main(argv: list[str] | None = None) -> int:
    a = base_parser(__doc__, 'served CPU path key, e.g. "q38fn-ud-iq4xs-cpu|cpu"').parse_args(argv)
    from . import census_coverage as cc, stage_bodies as sb
    spec = sb.path_specs(cc.load_registry() or {})[a.path]
    runner, live = (synth_runner(a.outcome), False) if a.test_mode else (subprocess.run, True)
    fake = [{"idx": 0, "port": 0, "cpus": "0-95", "threads": 96, "prefix": ["numactl", "--interleave=all", "--", "taskset", "-c", "0-95"]}]
    try:
        if a.calibrate:
            return emit(a, {"floor_file": str(calibrate(a.anchor_build, spec, source_row=a.row or "", pairs=a.pairs or aa.MIN_PAIRS, store=a.store,
                                                        runner=runner, live=live, topo=None if live else fake, reps=9 if live else 10))})
        return emit(a, time_path({"pairs": a.pairs, "reps": 9 if live else 10, "test_topology": fake}, a.anchor_build, a.candidate_build or a.anchor_build,
                                 spec, store=a.store, runner=runner, live=live, floor=None if live else 1.0))
    except RunnerRefused as e:
        return emit(a, {"verdict": f"{e.token} ({e})"})


if __name__ == "__main__":
    sys.exit(main())
