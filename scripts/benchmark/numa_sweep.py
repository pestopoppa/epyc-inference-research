#!/usr/bin/env python3
"""
numa_sweep.py — Deterministic NUMA throughput sweep for GGUF models.

HARDWARE: AMD EPYC 9655, 96 physical cores (192 logical), 2 NUMA nodes, 1.1TB RAM

NUMA LAUNCH RULES:
  1. ALWAYS use --mlock (forces private resident pages)
  2. Single-instance: numactl --interleave=all (spread across both nodes)
  3. Per-node instance: numactl --cpunodebind=N --membind=N (pin CPU AND memory)
  4. Quarter-machine: numactl --membind=N + taskset -c <cpus> (membind to parent node)
  5. NEVER use bare taskset — it pins CPU but not memory, causing cross-NUMA thrash
  6. Always use 96 threads for full-node, 48 for quarter (physical cores only)
  7. Multi-instance: load SEQUENTIALLY (concurrent mlock crashes)

CONFIGS:
  A: 1×96t, numactl --interleave=all     (baseline, best single-request latency)
  B: 1×96t, numactl --cpunodebind=0      (single NUMA node)
  C: 2×96t, one per NUMA node            (2x throughput)
  D: 4×48t, quarter-machine              (4x throughput)
  E: 8×24t, eighth-machine               (8x throughput for small models)

Usage:
  numa_sweep.py <model_path> [options]
  numa_sweep.py <model_path> --configs A,B,C
  numa_sweep.py <model_path> --draft-max 64 --configs B,C,D
  numa_sweep.py <model_path> --extra-args "--kv-unified" --draft-model /path/to/draft.gguf

Options:
  --name <label>          Model label (default: derived from filename)
  --draft-max <n>         Fixed draft-max (0=no speculation, omit=sweep)
  --draft-model <path>    Draft model for speculative decoding
  --configs <A,B,C,D>     Configs to run (default: all that fit)
  --n-predict <n>         Tokens per prompt (default: 256)
  --port <port>           Base port (default: 8190)
  --max-instances <n>     Override auto-detected max instances
  --extra-args "<args>"   Additional llama-server args
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ============================================================
# Constants
# ============================================================

LLAMA_SERVER = "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
DATA_DIR = Path("/mnt/raid0/llm/epyc-inference-research/data/numa_sweeps")
TOTAL_RAM_GB = 1100

NUMA = {
    "node0": "0-47,96-143",
    "node1": "48-95,144-191",
    # Quarters (48 logical threads each)
    "n0a": "0-23,96-119",
    "n0b": "24-47,120-143",
    "n1a": "48-71,144-167",
    "n1b": "72-95,168-191",
    # Eighths (24 logical threads each)
    "n0a0": "0-11,96-107",
    "n0a1": "12-23,108-119",
    "n0b0": "24-35,120-131",
    "n0b1": "36-47,132-143",
    "n1a0": "48-59,144-155",
    "n1a1": "60-71,156-167",
    "n1b0": "72-83,168-179",
    "n1b1": "84-95,180-191",
}
QUARTER_MEMBIND = {"n0a": 0, "n0b": 0, "n1a": 1, "n1b": 1}
EIGHTH_MEMBIND = {
    "n0a0": 0, "n0a1": 0, "n0b0": 0, "n0b1": 0,
    "n1a0": 1, "n1a1": 1, "n1b0": 1, "n1b1": 1,
}

PROMPTS = [
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:",
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:",
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:",
]

DRAFT_MAX_SWEEP = [0, 16, 32, 48, 64, 96, 128]


# ============================================================
# Server management
# ============================================================

class Server:
    """Manages a single llama-server process."""

    def __init__(self, model_path, port, threads, numa_cmd, extra_args="",
                 draft_model=None, spec_args="", log_path=None):
        self.port = port
        self.proc = None
        self.log_path = log_path

        cmd_parts = numa_cmd + [
            LLAMA_SERVER, "-m", model_path,
            "-t", str(threads), "-np", "1",
            "--port", str(port), "-ngl", "0",
            "--mlock", "--metrics",
        ]
        if draft_model:
            cmd_parts.extend(["-md", draft_model])
        if spec_args:
            cmd_parts.extend(spec_args.split())
        if extra_args:
            cmd_parts.extend(extra_args.split())

        log_fh = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd_parts, stdout=log_fh, stderr=subprocess.STDOUT)
        self._log_fh = log_fh

    def wait_healthy(self, timeout=600):
        """Block until /health returns ok."""
        url = f"http://localhost:{self.port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = urllib.request.urlopen(url, timeout=5)
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    elapsed = int(time.time() - start)
                    print(f"    port {self.port} ready ({elapsed}s)")
                    return True
            except Exception:
                pass
            time.sleep(5)
        print(f"    ERROR: port {self.port} did not start within {timeout}s")
        return False

    def warmup(self):
        """Send a short warmup request."""
        self._complete("Hello", max_tokens=32)

    def complete(self, prompt, max_tokens=256):
        """Send a completion request, return (tokens, elapsed_ms, tps)."""
        start = time.monotonic()
        result = self._complete(prompt, max_tokens)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        tokens = 0
        if result:
            tokens = result.get("usage", {}).get("completion_tokens", 0)

        if tokens > 0 and elapsed_ms > 0:
            tps = round(tokens / (elapsed_ms / 1000), 2)
        else:
            tps = 0.0

        return tokens, elapsed_ms, tps

    def _complete(self, prompt, max_tokens):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        payload = json.dumps({
            "model": "test",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            return json.loads(resp.read())
        except Exception as e:
            print(f"    WARNING: request failed on port {self.port}: {e}")
            return None

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        if hasattr(self, '_log_fh') and self._log_fh != subprocess.DEVNULL:
            self._log_fh.close()


def kill_all_servers(servers):
    for s in servers:
        s.kill()
    time.sleep(2)


# ============================================================
# Results writer
# ============================================================

class ResultsWriter:
    FIELDS = ["model", "config", "instance", "threads", "cpu_binding",
              "spec", "prompt_idx", "tokens_generated", "time_ms", "tokens_per_sec"]

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.fh.flush()

    def write(self, **kwargs):
        self.writer.writerow(kwargs)
        self.fh.flush()  # incremental — safe to kill

    def close(self):
        self.fh.close()


# ============================================================
# Benchmark configs
# ============================================================

def run_prompts(server, model_name, config, instance, threads, cpu_binding,
                spec_label, n_predict, results):
    for i, prompt in enumerate(PROMPTS):
        tokens, elapsed_ms, tps = server.complete(prompt, n_predict)
        results.write(
            model=model_name, config=config, instance=instance,
            threads=threads, cpu_binding=cpu_binding, spec=spec_label,
            prompt_idx=i, tokens_generated=tokens, time_ms=elapsed_ms,
            tokens_per_sec=tps,
        )
        print(f"    prompt {i}: {tps} t/s ({tokens} tokens)")


def bench_config_A(model_name, model_path, spec_args, spec_label, n_predict,
                   results, log_dir, draft_model=None, extra_args=""):
    """1×96t, numactl --interleave=all"""
    print("  --- Config A: 1×96t interleave ---")
    s = Server(model_path, 8190, 96,
               ["numactl", "--interleave=all"],
               spec_args=spec_args, draft_model=draft_model,
               extra_args=extra_args,
               log_path=str(log_dir / f"{model_name}_A.log"))
    if not s.wait_healthy():
        s.kill()
        return
    s.warmup()
    run_prompts(s, model_name, "A_1x96t_interleave", "1", 96, "interleave",
                spec_label, n_predict, results)
    s.kill()


def bench_config_B(model_name, model_path, spec_args, spec_label, n_predict,
                   results, log_dir, draft_model=None, extra_args=""):
    """1×96t, pinned to NUMA node 0"""
    print("  --- Config B: 1×96t node0 ---")
    s = Server(model_path, 8190, 96,
               ["numactl", "--cpunodebind=0", "--membind=0"],
               spec_args=spec_args, draft_model=draft_model,
               extra_args=extra_args,
               log_path=str(log_dir / f"{model_name}_B.log"))
    if not s.wait_healthy():
        s.kill()
        return
    s.warmup()
    run_prompts(s, model_name, "B_1x96t_node0", "1", 96, "node0",
                spec_label, n_predict, results)
    s.kill()


def bench_config_C(model_name, model_path, spec_args, spec_label, n_predict,
                   results, log_dir, draft_model=None, extra_args=""):
    """2×96t, one per NUMA node, sequential load"""
    print("  --- Config C: 2×96t dual-node ---")
    servers = []

    # Instance 1: node0
    s1 = Server(model_path, 8190, 96,
                ["numactl", "--cpunodebind=0", "--membind=0"],
                spec_args=spec_args, draft_model=draft_model,
                extra_args=extra_args,
                log_path=str(log_dir / f"{model_name}_C_n0.log"))
    servers.append(s1)
    print("    loading instance 1 (node0)...")
    if not s1.wait_healthy():
        kill_all_servers(servers)
        return

    # Instance 2: node1 (sequential)
    s2 = Server(model_path, 8191, 96,
                ["numactl", "--cpunodebind=1", "--membind=1"],
                spec_args=spec_args, draft_model=draft_model,
                extra_args=extra_args,
                log_path=str(log_dir / f"{model_name}_C_n1.log"))
    servers.append(s2)
    print("    loading instance 2 (node1)...")
    if not s2.wait_healthy():
        kill_all_servers(servers)
        return

    s1.warmup()
    s2.warmup()
    print("    both ready")

    names = ["n0", "n1"]
    server_list = [s1, s2]

    for i, prompt in enumerate(PROMPTS):
        with ThreadPoolExecutor(max_workers=len(server_list)) as pool:
            futures = {pool.submit(s.complete, prompt, n_predict): name
                       for s, name in zip(server_list, names)}
            instance_results = {}
            for f in as_completed(futures):
                name = futures[f]
                instance_results[name] = f.result()

        parts = []
        for name in names:
            t, ms, tps = instance_results[name]
            binding = "node0" if name == "n0" else "node1"
            results.write(model=model_name, config="C_2x96t", instance=name,
                          threads=96, cpu_binding=binding, spec=spec_label,
                          prompt_idx=i, tokens_generated=t, time_ms=ms,
                          tokens_per_sec=tps)
            parts.append(f"{name}={tps}")
        print(f"    prompt {i}: {', '.join(parts)}")

    kill_all_servers(servers)


def bench_config_D(model_name, model_path, spec_args, spec_label, n_predict,
                   results, log_dir, draft_model=None, extra_args=""):
    """4×48t, quarter-machine, sequential load"""
    print("  --- Config D: 4×48t quarter-machine ---")

    quarters = ["n0a", "n0b", "n1a", "n1b"]
    ports = [8190, 8191, 8192, 8193]
    servers = []

    for q, (name, port) in enumerate(zip(quarters, ports)):
        membind = QUARTER_MEMBIND[name]
        cpus = NUMA[name]
        s = Server(model_path, port, 48,
                   ["numactl", f"--membind={membind}", "taskset", "-c", cpus],
                   spec_args=spec_args, draft_model=draft_model,
                   extra_args=extra_args,
                   log_path=str(log_dir / f"{model_name}_D_{name}.log"))
        servers.append(s)
        print(f"    loading instance {q+1} ({name})...")
        if not s.wait_healthy():
            kill_all_servers(servers)
            return

    for s in servers:
        s.warmup()
    print("    all ready")

    for i, prompt in enumerate(PROMPTS):
        with ThreadPoolExecutor(max_workers=len(servers)) as pool:
            futures = {pool.submit(s.complete, prompt, n_predict): name
                       for s, name in zip(servers, quarters)}
            instance_results = {}
            for f in as_completed(futures):
                name = futures[f]
                instance_results[name] = f.result()

        parts = []
        for name in quarters:
            t, ms, tps = instance_results[name]
            results.write(model=model_name, config="D_4x48t", instance=name,
                          threads=48, cpu_binding=name, spec=spec_label,
                          prompt_idx=i, tokens_generated=t, time_ms=ms,
                          tokens_per_sec=tps)
            parts.append(f"{name}={tps}")
        print(f"    prompt {i}: {', '.join(parts)}")

    kill_all_servers(servers)


def bench_config_E(model_name, model_path, spec_args, spec_label, n_predict,
                   results, log_dir, draft_model=None, extra_args=""):
    """8×24t, eighth-machine, sequential load"""
    print("  --- Config E: 8×24t eighth-machine ---")

    eighths = ["n0a0", "n0a1", "n0b0", "n0b1", "n1a0", "n1a1", "n1b0", "n1b1"]
    ports = [8190 + i for i in range(8)]
    servers = []

    for q, (name, port) in enumerate(zip(eighths, ports)):
        membind = EIGHTH_MEMBIND[name]
        cpus = NUMA[name]
        s = Server(model_path, port, 24,
                   ["numactl", f"--membind={membind}", "taskset", "-c", cpus],
                   spec_args=spec_args, draft_model=draft_model,
                   extra_args=extra_args,
                   log_path=str(log_dir / f"{model_name}_E_{name}.log"))
        servers.append(s)
        print(f"    loading instance {q+1} ({name})...")
        if not s.wait_healthy():
            kill_all_servers(servers)
            return

    for s in servers:
        s.warmup()
    print("    all ready")

    for i, prompt in enumerate(PROMPTS):
        with ThreadPoolExecutor(max_workers=len(servers)) as pool:
            futures = {pool.submit(s.complete, prompt, n_predict): name
                       for s, name in zip(servers, eighths)}
            instance_results = {}
            for f in as_completed(futures):
                name = futures[f]
                instance_results[name] = f.result()

        parts = []
        for name in eighths:
            t, ms, tps = instance_results[name]
            results.write(model=model_name, config="E_8x24t", instance=name,
                          threads=24, cpu_binding=name, spec=spec_label,
                          prompt_idx=i, tokens_generated=t, time_ms=ms,
                          tokens_per_sec=tps)
            parts.append(f"{name}={tps}")
        print(f"    prompt {i}: {', '.join(parts)}")

    kill_all_servers(servers)


CONFIG_FUNCS = {
    "A": bench_config_A,
    "B": bench_config_B,
    "C": bench_config_C,
    "D": bench_config_D,
    "E": bench_config_E,
}


# ============================================================
# Draft-max sweep
# ============================================================

def draft_max_sweep(model_name, model_path, n_predict, results, log_dir,
                    draft_model=None, extra_args=""):
    """Sweep draft-max at 1×96t interleave, return best value."""
    print("=" * 64)
    print("  Phase 1: draft-max sweep (1×96t interleave)")
    print("=" * 64)
    print()

    best_dm = 0
    best_tps = 0.0

    for dm in DRAFT_MAX_SWEEP:
        if dm == 0:
            spec_args = ""
            label = "baseline"
        else:
            spec_args = f"--spec-type ngram-simple --draft-max {dm}"
            label = f"ngram_dm{dm}"

        print(f"  --- draft-max={dm} ---")
        s = Server(model_path, 8190, 96,
                   ["numactl", "--interleave=all"],
                   spec_args=spec_args, draft_model=draft_model,
                   extra_args=extra_args,
                   log_path=str(log_dir / f"{model_name}_dmsweep_{dm}.log"))
        if not s.wait_healthy():
            s.kill()
            continue

        s.warmup()
        total_tps = 0.0
        for i, prompt in enumerate(PROMPTS):
            tokens, elapsed_ms, tps = s.complete(prompt, n_predict)
            results.write(
                model=model_name, config=f"dmsweep_dm{dm}", instance="1",
                threads=96, cpu_binding="interleave", spec=label,
                prompt_idx=i, tokens_generated=tokens, time_ms=elapsed_ms,
                tokens_per_sec=tps,
            )
            print(f"    prompt {i}: {tps} t/s")
            total_tps += tps

        avg_tps = total_tps / len(PROMPTS)
        print(f"    avg: {avg_tps:.2f} t/s")

        if avg_tps > best_tps:
            best_tps = avg_tps
            best_dm = dm

        s.kill()
        print()

    print("=" * 64)
    print(f"  Phase 1 result: best draft-max={best_dm} ({best_tps:.2f} t/s)")
    print("=" * 64)
    print()
    return best_dm


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NUMA throughput sweep")
    parser.add_argument("model_path", help="Path to GGUF model file")
    parser.add_argument("--name", help="Model label")
    parser.add_argument("--draft-max", type=int, default=None,
                        help="Fixed draft-max (0=no spec, omit=sweep)")
    parser.add_argument("--draft-model", help="Draft model path")
    parser.add_argument("--configs", default="A,C,D,E",
                        help="Configs to run (default: A,B,C,D)")
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--extra-args", default="", help="Extra llama-server args")

    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"ERROR: {args.model_path} not found", file=sys.stderr)
        sys.exit(1)

    # Derive model name
    model_name = args.name or Path(args.model_path).stem.replace("-00001-of-", "_")

    # Estimate model size — for split models, sum shards matching the same prefix
    model_file = Path(args.model_path)
    base_name = model_file.stem
    # Strip shard suffix like "-00001-of-00006" to find all shards
    import re
    shard_match = re.match(r"(.+)-\d{5}-of-\d{5}", base_name)
    if shard_match:
        prefix = shard_match.group(1)
        model_size_gb = sum(
            f.stat().st_size for f in model_file.parent.iterdir()
            if f.name.startswith(prefix) and f.suffix == ".gguf"
        ) / (1024**3)
    else:
        model_size_gb = os.path.getsize(args.model_path) / (1024**3)

    # Auto-detect max instances
    max_inst = args.max_instances
    if max_inst is None:
        usable = TOTAL_RAM_GB - 50
        max_inst = min(8, max(1, int(usable / max(model_size_gb, 1))))

    # Filter configs by what fits
    requested = [c.strip() for c in args.configs.split(",")]
    configs = []
    for c in requested:
        if c == "C" and max_inst < 2:
            print(f"  Config C skipped: model too large for 2 instances ({model_size_gb:.0f}GB × 2)")
            continue
        if c == "D" and max_inst < 4:
            print(f"  Config D skipped: model too large for 4 instances ({model_size_gb:.0f}GB × 4)")
            continue
        if c == "E" and max_inst < 8:
            print(f"  Config E skipped: model too large for 8 instances ({model_size_gb:.0f}GB × 8)")
            continue
        configs.append(c)

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = DATA_DIR / f"logs_{model_name}_{timestamp}"
    log_dir.mkdir(exist_ok=True)
    results_path = DATA_DIR / f"{model_name}_numa_{timestamp}.csv"
    results = ResultsWriter(str(results_path))

    # Banner
    print("=" * 64)
    print(f"  NUMA Sweep: {model_name}")
    print("=" * 64)
    print(f"  Model:       {args.model_path}")
    print(f"  Size:        ~{model_size_gb:.0f} GB")
    print(f"  Max inst:    {max_inst}")
    print(f"  Draft-max:   {args.draft_max if args.draft_max is not None else 'sweep'}")
    print(f"  Draft model: {args.draft_model or 'none'}")
    print(f"  Extra args:  {args.extra_args or 'none'}")
    print(f"  Configs:     {','.join(configs)}")
    print(f"  n_predict:   {args.n_predict}")
    print(f"  Results:     {results_path}")
    print(f"  Logs:        {log_dir}")
    print("=" * 64)
    print()

    # Phase 1: draft-max sweep (if not fixed)
    dm = args.draft_max
    phase1_ran = False
    if dm is None:
        dm = draft_max_sweep(model_name, args.model_path, args.n_predict,
                             results, log_dir, args.draft_model, args.extra_args)
        phase1_ran = True

    # Build spec args
    if dm == 0:
        spec_args = ""
        spec_label = "baseline"
    else:
        spec_args = f"--spec-type ngram-simple --draft-max {dm}"
        spec_label = f"ngram_dm{dm}"

    # Phase 2: NUMA sweep
    print("=" * 64)
    print(f"  Phase 2: NUMA sweep (draft-max={dm})")
    print("=" * 64)
    print()

    common = dict(
        model_name=model_name, model_path=args.model_path,
        spec_args=spec_args, spec_label=spec_label,
        n_predict=args.n_predict, results=results, log_dir=log_dir,
        draft_model=args.draft_model, extra_args=args.extra_args,
    )

    for c in configs:
        if c == "A" and phase1_ran:
            print("  --- Config A: 1×96t interleave — already measured in Phase 1 ---")
            print()
            continue
        CONFIG_FUNCS[c](**common)
        print()

    # Summary
    results.close()
    print("=" * 64)
    print(f"  SWEEP COMPLETE: {model_name}")
    print(f"  Best draft-max: {dm}")
    print("=" * 64)
    print()
    print(f"Results: {results_path}")
    print(f"Logs:    {log_dir}")


if __name__ == "__main__":
    main()
