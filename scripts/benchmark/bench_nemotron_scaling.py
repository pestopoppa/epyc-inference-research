#!/usr/bin/env python3
"""NUMA scaling test for Nemotron-Cascade 2: 1×48t, 2×48t, 2×96t, 4×48t."""
import subprocess, time, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import urllib.request

MODEL = "/mnt/raid0/llm/models/nemotron-cascade-2/nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf"
SERVER = "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
PROMPT = {
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Write a detailed Python implementation of a red-black tree with insert, delete, and search operations. Include type hints and docstrings."}],
    "max_tokens": 512,
    "temperature": 0.2
}
WARMUP = 2
ROUNDS = 8

CONFIGS = [
    {
        "label": "1x48t (baseline)",
        "instances": [{"cpus": "0-47", "threads": 48, "port": 8199}],
    },
    {
        "label": "2x48t (same NUMA node 0)",
        "instances": [
            {"cpus": "0-47",  "threads": 48, "port": 8199},
            {"cpus": "48-95", "threads": 48, "port": 8198},
        ],
    },
    {
        "label": "2x96t (cross-NUMA)",
        "instances": [
            {"cpus": "0-95",   "threads": 96, "port": 8199},
            {"cpus": "96-191", "threads": 96, "port": 8198},
        ],
    },
    {
        "label": "4x48t (all quarters)",
        "instances": [
            {"cpus": "0-47",    "threads": 48, "port": 8199},
            {"cpus": "48-95",   "threads": 48, "port": 8198},
            {"cpus": "96-143",  "threads": 48, "port": 8197},
            {"cpus": "144-191", "threads": 48, "port": 8196},
        ],
    },
]

def start_instances(instances):
    procs = []
    for inst in instances:
        cmd = (f"taskset -c {inst['cpus']} {SERVER} -m {MODEL} "
               f"-t {inst['threads']} --port {inst['port']} -c 4096 -np 1 --no-warmup")
        p = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)

    for inst in instances:
        for attempt in range(90):
            try:
                req = urllib.request.Request(f"http://localhost:{inst['port']}/health")
                resp = urllib.request.urlopen(req, timeout=2)
                if b"ok" in resp.read():
                    break
            except Exception:
                pass
            time.sleep(1)
    return procs

def send_request(port):
    data = json.dumps(PROMPT).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    resp = urllib.request.urlopen(req, timeout=120)
    raw = resp.read().decode()
    elapsed = time.monotonic() - start
    r = json.loads(raw)
    tokens = r["usage"]["completion_tokens"]
    tps = tokens / elapsed if elapsed > 0 else 0
    return tokens, elapsed, tps

def stop_instances(procs):
    for p in procs:
        p.kill()
    for p in procs:
        p.wait()
    time.sleep(3)

def bench_config(config):
    label = config["label"]
    instances = config["instances"]
    ports = [i["port"] for i in instances]
    n = len(instances)

    print(f"\n{'='*60}")
    print(f"CONFIG: {label}")
    print(f"{'='*60}")

    procs = start_instances(instances)
    print(f"  {n} instance(s) ready")

    try:
        # Warmup
        for w in range(WARMUP):
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = [ex.submit(send_request, p) for p in ports]
                for f in as_completed(futs):
                    f.result()
        print(f"  Warmup done ({WARMUP} rounds)")

        # Benchmark
        all_results = []
        round_aggs = []
        for rnd in range(1, ROUNDS + 1):
            round_results = {}
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {ex.submit(send_request, p): (i, p) for i, p in enumerate(ports)}
                for f in as_completed(futs):
                    inst_idx, port = futs[f]
                    tokens, wall, tps = f.result()
                    round_results[inst_idx] = tps
                    all_results.append((rnd, inst_idx, tps))

            agg = sum(round_results.values())
            round_aggs.append(agg)
            per = " | ".join(f"i{k}={v:.1f}" for k, v in sorted(round_results.items()))
            print(f"  round {rnd}: agg={agg:.1f} t/s  [{per}]")

        avg_agg = sum(round_aggs) / len(round_aggs)
        per_inst_avg = avg_agg / n
        print(f"\n  >> {label}: avg_agg={avg_agg:.1f} t/s, per_inst={per_inst_avg:.1f} t/s")
        return label, avg_agg, per_inst_avg, n

    finally:
        stop_instances(procs)

def main():
    print("Nemotron-Cascade 2 NUMA Scaling Test")
    print("=" * 60)

    results = []
    for config in CONFIGS:
        label, avg_agg, per_inst, n = bench_config(config)
        results.append((label, n, avg_agg, per_inst))

    print(f"\n{'='*60}")
    print("SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'N':>3} {'Agg t/s':>10} {'Per-inst':>10} {'Efficiency':>10}")
    baseline_per = results[0][3]
    for label, n, agg, per in results:
        eff = per / baseline_per * 100
        print(f"{label:<30} {n:>3} {agg:>10.1f} {per:>10.1f} {eff:>9.0f}%")

if __name__ == "__main__":
    main()
