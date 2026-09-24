"""M-4: aggregate decode on the LIVE :8083 (np4 / 196608 / --kv-unified / depth 4) under concurrent load.

Fixed-length generation (ignore_eos, n_predict fixed) so every request decodes the same number of tokens:
this removes the answer-length artifact that made single-wave aggregates misleading in the study.
Concurrency 1, 2, 4; 2 waves each; per-request decode tok/s from server timings.
"""
import json, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8083/completion"
N_PREDICT = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
PROMPTS = [
    "Write a detailed technical explanation of how paged attention works in LLM serving.",
    "Explain the difference between tensor parallelism and pipeline parallelism in depth.",
    "Describe, step by step, how a CPU cache hierarchy affects matrix multiplication performance.",
    "Give a thorough overview of speculative decoding and how draft acceptance affects speed.",
]

def one(prompt: str) -> dict:
    body = {"prompt": prompt, "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    tm = out.get("timings", {})
    return {"wall": time.time() - t, "n": tm.get("predicted_n", 0), "tps": tm.get("predicted_per_second", 0.0)}

results = []
for conc in (1, 2, 4):
    for wave in (1, 2):
        t0 = time.time()
        with ThreadPoolExecutor(conc) as ex:
            rs = list(ex.map(one, PROMPTS[:conc]))
        wall = time.time() - t0
        agg = sum(r["n"] for r in rs) / wall
        row = {"conc": conc, "wave": wave, "agg_tps": round(agg, 1),
               "perreq_med_tps": round(statistics.median(r["tps"] for r in rs), 1),
               "tokens": [r["n"] for r in rs], "wall_s": round(wall, 1)}
        results.append(row)
        print(json.dumps(row), flush=True)
json.dump(results, open("/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_kvu_study_20260924/m4_np4_concurrent_live.json", "w"), indent=2)
