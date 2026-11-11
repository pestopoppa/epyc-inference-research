#!/usr/bin/env python3
"""INF-45 artifact (3): cross-role interference coefficients, small and SPARSE.

Box: epyc-root numa-placement-and-contention.md -> "(3) Cross-role interference coefficients".
Only pairs that are feasible under artifact (1) (cpusets derived from the orchestrator's
stack_numa.py, never retyped) and that genuinely share a resource:

  (i)   cross-node memory bandwidth: victim on half A (NPS4 nodes 0+1) vs an aggressor
        saturating half B (nodes 2+3), victim at np 1 and np 4;
  (ii)  quarter vs quarter on the same socket: victim on node 0 vs aggressor on node 1;
  (iii) the GPU host lane vs CPU tenants: ALREADY MEASURED 2026-10-22 (U4-NOISE-1,
        research b65284539 scripts/benchmark/gpu_cpu_noise/): CPU decode -0.7..-1.05 %,
        prefill -4.3..-4.8 %, SMT channel (88-95 vs 184-191) prefill -9.6 %. Reused as a
        recorded coefficient, never re-run.

Per measured pair: the victim's aggregate decode tok/s ALONE (aggressor loaded, idle) vs
WITH the aggressor saturated (every aggressor slot busy with long generations), ABAB
alternation, 5 rounds. Coefficient = median over rounds of with/alone. Weights are private
(--no-mmap) and bound to the instance's own nodes, so victim and aggressor never share a
page cache. Expected effect 1-10 %; the 1.96 % serving floor sets n = 5 rounds.

Stop rules (per pair): stop-on-answer when the 5-round 95 % CI of (ratio - 1) excludes 0 or
lies within +/-2 %; futility after 5 rounds when the round spread (max-min ratio) > 10 %.

Subcommands:
  pairs   [--output P]                       write the pair list (design; zero inference)
  run     --pair ID --out DIR [--execute --i-have-operator-grant]
          default is DRY: prints the launch argv and exits 0, nothing spawned.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STACK_NUMA = Path("/mnt/raid0/llm/epyc-orchestrator/scripts/server/stack_numa.py")
DEFAULT_PAIRS = REPO / "data" / "batched_decode" / "interference_pairs_v1.json"
LLAMA_SERVER = Path("/mnt/raid0/llm/kernels/production/cpu/llama-server")
ROUNDS = 5
FLOOR = 0.02
SPREAD_FUTILITY = 0.10

VICTIM = {"model_key": "qwen36_q8_0", "model_path": "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
          "spec": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
          "chat_template_kwargs": {"enable_thinking": False}}
AGGRESSOR = {"model_key": "qwen38_flash_next_ud_iq4xs",
             "model_path": "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/"
                           "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
             "spec": [], "chat_template_kwargs": {"enable_thinking": False}}
GPU_LANE_MEASURED = {
    "id": "P3-gpu-lane-vs-cpu", "status": "measured-reuse",
    "source": "U4-NOISE-1 2026-10-22, research b65284539 scripts/benchmark/gpu_cpu_noise/, "
              "run dir /mnt/raid0/llm/tmp/gpu-cpu-noise-20261021/",
    "coefficients": {"cpu_decode_pct": [-0.7, -1.05], "cpu_prefill_pct": [-4.3, -4.8],
                     "smt_channel_prefill_pct": -9.6},
}


def _stack_numa():
    sys.path.insert(0, str(STACK_NUMA.parent))
    import stack_numa  # noqa: E402
    return stack_numa


def build_pairs() -> dict:
    sn = _stack_numa()
    nodes = sn._NPS4_NODES
    half_a, half_b = sn.NUMA_HALF_A, sn.NUMA_HALF_B
    commit = subprocess.run(["git", "-C", str(STACK_NUMA.parent), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    def inst(role, model, cpus, threads, policy, port, np_):
        return {"role": role, "model": model, "cpu_list": cpus, "threads": threads,
                "numactl": policy, "port": port, "np": np_}

    pairs = []
    for np_ in (1, 4):
        pairs.append({"id": f"P1-halfA-vs-halfB-np{np_}", "resource": "cross-node memory bandwidth",
                      "victim": inst("victim", VICTIM, half_a[0], half_a[1], "--interleave=0,1", 19600, np_),
                      "aggressor": inst("aggressor", AGGRESSOR, half_b[0], half_b[1], "--interleave=2,3", 19601, 4)})
    pairs.append({"id": "P2-q0-vs-q1-np1", "resource": "same-socket quarter vs quarter",
                  "victim": inst("victim", VICTIM, nodes[0], 24, "--membind=0", 19602, 1),
                  "aggressor": inst("aggressor", AGGRESSOR, nodes[1], 24, "--membind=1", 19603, 4)})
    return {"schema": "interference-pairs/1", "generated_at": datetime.now(timezone.utc).isoformat(),
            "cpuset_source": f"epyc-orchestrator scripts/server/stack_numa.py @ {commit} "
                             "(NUMA_HALF_A/B, _NPS4_NODES)",
            "rounds": ROUNDS, "floor": FLOOR, "spread_futility": SPREAD_FUTILITY,
            "measured": pairs, "reused": [GPU_LANE_MEASURED]}


# ---------------------------------------------------------------------------
# launch + drive
# ---------------------------------------------------------------------------

def launch_argv(i: dict) -> list[str]:
    m = i["model"]
    return (["taskset", "-c", i["cpu_list"], "numactl", i["numactl"], str(LLAMA_SERVER), "-m", m["model_path"],
             "--host", "127.0.0.1", "--port", str(i["port"]), "-np", str(i["np"]), "-c", str(4096 * i["np"]),
             "-t", str(i["threads"]), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "--no-mmap",
             "--reasoning", "off", "--device", "none", "--log-colors", "off"] + m["spec"]
            + (["--device-draft", "none"] if m["spec"] else []))


def _post(port: int, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_health(port: int, proc, limit: float = 1800.0) -> None:
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"server on {port} exited rc={proc.returncode} during load")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise RuntimeError(f"server on {port} not healthy after {limit}s")


VICTIM_PROMPTS = [
    "Explain, step by step, how a hash map handles collisions.",
    "Write a Python function that merges two sorted lists, then explain its complexity.",
    "Summarise the causes of the French Revolution in five bullet points.",
    "Describe how TCP congestion control works.",
]


def victim_phase(i: dict, reps: int = 3) -> float:
    """Aggregate decode tok/s: np concurrent streams x reps requests each, llama.cpp timings."""
    rates: list[float] = []
    lock = threading.Lock()

    def stream(k):
        for r in range(reps):
            out = _post(i["port"], {"messages": [{"role": "user", "content": VICTIM_PROMPTS[(k + r) % 4]}],
                                    "max_tokens": 128, "temperature": 0.0, "seed": 42,
                                    "chat_template_kwargs": i["model"]["chat_template_kwargs"]})
            pps = (out.get("timings") or {}).get("predicted_per_second")
            if pps:
                with lock:
                    rates.append(float(pps))
    ts = [threading.Thread(target=stream, args=(k,)) for k in range(i["np"])]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if not rates:
        raise RuntimeError("victim phase produced no timings")
    return statistics.median(rates) * i["np"]


class Saturator:
    """Keeps every aggressor slot busy with long generations until stopped."""

    def __init__(self, i: dict):
        self.i, self.stop, self.threads, self.errors = i, threading.Event(), [], []

    def _loop(self, k):
        while not self.stop.is_set():
            try:
                _post(self.i["port"], {"messages": [{"role": "user", "content":
                      f"Write a long, detailed essay (part {k}) on the history of computing."}],
                      "max_tokens": 512, "temperature": 0.7, "seed": k,
                      "chat_template_kwargs": self.i["model"]["chat_template_kwargs"]})
            except Exception as exc:  # noqa: BLE001
                self.errors.append(str(exc))
                time.sleep(1)

    def __enter__(self):
        self.threads = [threading.Thread(target=self._loop, args=(k,), daemon=True) for k in range(self.i["np"])]
        [t.start() for t in self.threads]
        time.sleep(20)  # let every slot reach decode
        return self

    def __exit__(self, *a):
        self.stop.set()
        [t.join(timeout=900) for t in self.threads]


def verdict(ratios: list[float]) -> dict:
    d = [r - 1.0 for r in ratios]
    n = len(d)
    mean = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else float("inf")
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.0)
    lo, hi = mean - t * se, mean + t * se
    spread = max(ratios) - min(ratios)
    if n >= ROUNDS and spread > SPREAD_FUTILITY:
        v = "futility"
    elif lo > 0 or hi < 0:
        v = "answered-nonzero"
    elif lo >= -FLOOR and hi <= FLOOR:
        v = "answered-within-floor"
    else:
        v = "unanswered" if n >= ROUNDS else "continue"
    return {"coefficient_median_ratio": statistics.median(ratios), "ci95_delta": [lo, hi],
            "spread": spread, "n_rounds": n, "verdict": v}


def cmd_run(a) -> int:
    doc = json.loads(Path(a.pairs).read_text())
    pair = next((p for p in doc["measured"] if p["id"] == a.pair), None)
    if pair is None:
        raise SystemExit(f"unknown pair {a.pair}")
    argvs = [launch_argv(pair["victim"]), launch_argv(pair["aggressor"])]
    if not a.execute:
        for v in argvs:
            print("[dry-run] " + " ".join(v))
        print(f"[dry-run] {pair['id']}: {ROUNDS} ABAB rounds; nothing spawned")
        return 0
    if not a.i_have_operator_grant:
        raise SystemExit("--execute requires --i-have-operator-grant")
    for p in (pair["victim"]["model"]["model_path"], pair["aggressor"]["model"]["model_path"]):
        if not os.path.exists(p):
            raise SystemExit(f"model missing: {p}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "scripts" / "lib"))
    from canonical_recipe import build_canonical_env  # same env composition as the E5 harness
    env = build_canonical_env(extra_vars={"KMP_BLOCKTIME": "10"})
    env["LD_LIBRARY_PATH"] = ":".join([str(LLAMA_SERVER.parent)] + [
        x for x in env.get("LD_LIBRARY_PATH", "").split(":") if x and x != str(LLAMA_SERVER.parent)])
    procs = []
    pidfile = os.environ.get("FILLER_PIDFILE")
    try:
        for i, argv in zip((pair["victim"], pair["aggressor"]), argvs):
            log = open(out / f"{pair['id']}-{i['role']}.log", "w")
            p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            procs.append(p)
            if pidfile:
                with open(pidfile, "a") as fh:
                    fh.write(f"{p.pid}\n")
            _wait_health(i["port"], p)
        victim_phase(pair["victim"], reps=1)  # warm
        rounds, ratios = [], []
        for r in range(ROUNDS):
            if os.environ.get("FILLER_STOPFILE") and os.path.exists(os.environ["FILLER_STOPFILE"]):
                raise SystemExit("STOP file present; pair abandoned (re-runs whole next grant)")
            alone = victim_phase(pair["victim"])
            with Saturator(pair["aggressor"]) as sat:
                loaded = victim_phase(pair["victim"])
            rounds.append({"round": r + 1, "alone_tps": alone, "with_aggressor_tps": loaded,
                           "ratio": loaded / alone, "aggressor_errors": len(sat.errors)})
            ratios.append(loaded / alone)
        rec = {"pair": pair["id"], "resource": pair["resource"], "rounds": rounds, **verdict(ratios),
               "victim": {k: pair["victim"][k] for k in ("cpu_list", "numactl", "np", "threads")},
               "aggressor": {k: pair["aggressor"][k] for k in ("cpu_list", "numactl", "np", "threads")},
               "metric": "victim aggregate decode tok/s (median per-request llama.cpp predicted_per_second x np)",
               "kernel": "production-consolidated-v10 (kernels/production/cpu)",
               "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
        with open(out / "results.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        (out / f"{pair['id']}.done").write_text(rec["verdict"] + "\n")
        print(json.dumps({k: rec[k] for k in ("pair", "coefficient_median_ratio", "ci95_delta", "verdict")}))
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, 15)
                    p.wait(timeout=60)
                except Exception:  # noqa: BLE001
                    os.killpg(p.pid, 9)
                    p.wait(timeout=30)
            if subprocess.run(["ps", "-p", str(p.pid)], capture_output=True).returncode == 0 and p.poll() is None:
                raise RuntimeError(f"server pid {p.pid} survived teardown")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--output", default=str(DEFAULT_PAIRS))
    p.set_defaults(func=lambda a: (Path(a.output).write_text(json.dumps(build_pairs(), indent=2) + "\n"),
                                   print(f"wrote {a.output}"), 0)[-1])
    r = sub.add_parser("run")
    r.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    r.add_argument("--pair", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--execute", action="store_true")
    r.add_argument("--i-have-operator-grant", action="store_true")
    r.set_defaults(func=cmd_run)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
