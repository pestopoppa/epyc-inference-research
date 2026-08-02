#!/usr/bin/env python3
"""Measure 27B decode tok/s via llama.cpp /completion. One rep per distinct prompt."""
import json
import sys
import time
import urllib.request

PORT = 8801
N_PREDICT = 256

# 9 distinct prompts + spares. Never reuse a prompt against a warm server.
PROMPTS = {
    "p1": "Explain, in careful detail, how a modern CPU's branch predictor works and why mispredictions are expensive.",
    "p2": "Describe the lifecycle of an HTTP request from DNS resolution through TLS handshake to the first byte of the response body.",
    "p3": "Write a thorough comparison of B-trees and LSM-trees as storage engine index structures, covering write amplification.",
    "p4": "Explain the mathematics of singular value decomposition and give three practical applications in engineering.",
    "p5": "Describe how a distributed consensus protocol such as Raft elects a leader and replicates a log entry safely.",
    "p6": "Discuss the tradeoffs between arena allocators, reference counting and tracing garbage collection in systems languages.",
    "p7": "Explain how photosynthesis converts light energy into chemical energy, tracing the path of an electron in detail.",
    "p8": "Give a detailed account of how a jet engine's compressor, combustor and turbine stages interact thermodynamically.",
    "p9": "Explain the design of a modern filesystem journal and how it guarantees crash consistency across a power loss.",
    "p10": "Describe how error-correcting codes such as Reed-Solomon recover data from erasures, with a worked intuition.",
    "p11": "Explain the biological mechanism of long-term potentiation and its relationship to memory formation in mammals.",
    "p12": "Discuss how ocean thermohaline circulation transports heat and what would happen if it slowed substantially.",
}


def one_rep(prompt_key):
    body = json.dumps({
        "prompt": PROMPTS[prompt_key],
        "n_predict": N_PREDICT,
        "ignore_eos": True,       # force exactly N_PREDICT decode steps
        "cache_prompt": False,    # no cross-rep prompt-cache reuse
        "temperature": 0.7,
        "top_p": 0.8,
        "seed": 42,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    t = out["timings"]
    return {
        "prompt_key": prompt_key,
        "decode_tps": t["predicted_per_second"],
        "predicted_n": t["predicted_n"],
        "predicted_ms": t["predicted_ms"],
        "prompt_n": t["prompt_n"],
        "prompt_per_second": t.get("prompt_per_second"),
        "wall_s": round(wall, 3),
    }


if __name__ == "__main__":
    keys = sys.argv[1:]
    results = []
    for k in keys:
        r = one_rep(k)
        results.append(r)
        print(json.dumps(r), flush=True)
    tps = sorted(x["decode_tps"] for x in results)
    n = len(tps)
    median = tps[n // 2] if n % 2 else (tps[n // 2 - 1] + tps[n // 2]) / 2
    print(json.dumps({"summary": {"n": n, "min": tps[0], "median": median, "max": tps[-1]}}), flush=True)
