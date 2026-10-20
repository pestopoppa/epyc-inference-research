#!/usr/bin/env python3
"""128k follow-up: the main sweep's 128k decode stopped at token 1 (EOS) in every arm.
(1) Diagnose: first-token top-10 probabilities at depth 131072 vs 65536 (same probe).
(2) Time: decode at 131072 with ignore_eos, every arm, 2 reps.
Reuses driver.py's launch/restore/recipe; appends to followup_128k.jsonl."""
import hashlib, json, sys
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/mnt/raid0/llm/tmp/q38fn-threads-depth-20261020")
import driver as D

D.OUT = f"{D.R}/followup_128k.jsonl"
tok, probe = json.load(open(f"{D.R}/corpus_tokens.json"))
ARMS = [("A", 48, 48), ("B", 48, 96), ("C", 96, 96)]


def req(prompt, n, ignore_eos=False, n_probs=0):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1,
            "cache_prompt": True, "stream": False, "samplers": ["top_k"],
            "ignore_eos": ignore_eos, "n_probs": n_probs}
    return D.post("/completion", body)


for arm, t, tb in ARMS:
    p = D.start(f"f128-{arm}", t, tb)
    try:
        if arm == "A":
            for d in (65536, 131072):
                D.restore(d)
                r = req(tok[:d] + probe, 1, n_probs=10)
                probs = r.get("completion_probabilities", [{}])[0]
                D.log({"kind": "first_token_probs", "arm": arm, "depth": d,
                       "content": r.get("content"), "stop_type": r.get("stop_type"),
                       "top": probs.get("top_logprobs") or probs.get("probs")})
        for rep in range(2):
            D.restore(131072)
            r = req(tok[:131072] + probe, D.N_PRED, ignore_eos=True)
            tm = r["timings"]
            D.log({"kind": "decode_at_depth_noeos", "arm": arm, "t": t, "tb": tb,
                   "depth": 131072, "rep": rep, "prompt_n": tm["prompt_n"],
                   "predicted_n": tm["predicted_n"], "pred_tps": tm["predicted_per_second"],
                   "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
                   "content_sha": hashlib.sha256(r["content"].encode()).hexdigest()[:16],
                   "content": r["content"]})
    finally:
        D.stop(p)
        D.log({"kind": "server_down", "arm": f"f128-{arm}"})
