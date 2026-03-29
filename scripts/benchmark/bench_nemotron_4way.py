#!/usr/bin/env python3
"""NUMA 4-way concurrent benchmark for Nemotron-Cascade 2."""
import subprocess, time, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import urllib.request

MODEL = "/mnt/raid0/llm/models/nemotron-cascade-2/nvidia_Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf"
SERVER = "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
PORTS = [8199, 8198, 8197, 8196]
CPUS = ["0-47", "48-95", "96-143", "144-191"]
PROMPT = {
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Write a detailed Python implementation of a red-black tree with insert, delete, and search operations. Include type hints and docstrings."}],
    "max_tokens": 512,
    "temperature": 0.2
}
WARMUP = 2
ROUNDS = 8

def start_servers():
    procs = []
    for i, (cpus, port) in enumerate(zip(CPUS, PORTS)):
        cmd = f"taskset -c {cpus} {SERVER} -m {MODEL} -t 48 --port {port} -c 4096 -np 1 --no-warmup"
        p = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        print(f"  Instance {i}: CPUs {cpus}, port {port}, PID {p.pid}")

    for i, port in enumerate(PORTS):
        for attempt in range(90):
            try:
                req = urllib.request.Request(f"http://localhost:{port}/health")
                resp = urllib.request.urlopen(req, timeout=2)
                if b"ok" in resp.read():
                    print(f"  Instance {i} ready after {attempt+1}s")
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

def main():
    print("Starting 4 NUMA instances...")
    procs = start_servers()

    try:
        print(f"\nWarmup ({WARMUP} rounds)...")
        for w in range(WARMUP):
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = [ex.submit(send_request, p) for p in PORTS]
                for f in as_completed(futs):
                    f.result()
            print(f"  warmup {w+1} done")

        print(f"\nBenchmark ({ROUNDS} rounds x 4 instances)...")
        all_results = []
        round_aggs = []

        for rnd in range(1, ROUNDS + 1):
            round_results = {}
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = {ex.submit(send_request, p): (i, p) for i, p in enumerate(PORTS)}
                for f in as_completed(futs):
                    inst, port = futs[f]
                    tokens, wall, tps = f.result()
                    round_results[inst] = (tokens, wall, tps)
                    all_results.append((rnd, inst, port, tokens, wall, tps))

            agg = sum(r[2] for r in round_results.values())
            round_aggs.append(agg)
            per_inst = " | ".join(f"i{k}={v[2]:.1f}" for k, v in sorted(round_results.items()))
            print(f"  round {rnd}: agg={agg:.1f} t/s  [{per_inst}]")

        print("\n" + "=" * 60)
        print("NUMA 4-WAY SUMMARY")
        print("=" * 60)
        print(f"Aggregate: avg={sum(round_aggs)/len(round_aggs):.2f} t/s, "
              f"min={min(round_aggs):.2f}, max={max(round_aggs):.2f}")

        per_inst = defaultdict(list)
        for rnd, inst, port, tokens, wall, tps in all_results:
            per_inst[inst].append(tps)
        for inst, vals in sorted(per_inst.items()):
            print(f"  Instance {inst}: avg={sum(vals)/len(vals):.2f} t/s")

        out = "/mnt/raid0/llm/epyc-inference-research/data/nemotron_cascade2/4way_results.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write("round,instance,port,completion_tokens,wall_seconds,tokens_per_sec\n")
            for rnd, inst, port, tokens, wall, tps in all_results:
                f.write(f"{rnd},{inst},{port},{tokens},{wall:.3f},{tps:.2f}\n")
        print(f"\nResults saved to {out}")

    finally:
        print("\nStopping servers...")
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()
        print("All stopped.")

if __name__ == "__main__":
    main()
