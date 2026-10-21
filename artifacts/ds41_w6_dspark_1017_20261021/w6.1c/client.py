"""DS41-W6.1 card-sampling client (extends w6.1/client.py; prompt set, n_predict and seeds are IDENTICAL to W6/W6.2).

usage: client.py <port> <outdir> <seedbase> <kind dspark|plain> <modes csv of card,stoch,greedy> <nmax csv, e.g. 2,3> [deadline_epoch]

Sampling arms (the SAMPLING TRIPLE is recorded on every row and in sampling.json, never inferred):
  card   temperature=1.0 top_p=1.0, plus top_k=0 min_p=0.0 repeat_penalty=1.0 sent EXPLICITLY (llama-server's defaults are
         top_k=40, min_p=0.05, which would silently turn "T=1.0 top_p=1.0" into a truncated sampler)
  stoch  the W6/W6.2 arm, body UNCHANGED (temperature=0.7 top_p=1.0 top_k=0; min_p is the server default, read back via /props)
  greedy temperature=0 top_k=1 (the W6 greedy arm)
kind=plain sends no speculative field and is the speedup denominator (a target-only server). kind=dspark sends
speculative.n_max per block width.
Every row carries `valid`; invalid = HTTP/parse error, no timings, a dspark row that drafted nothing, or fewer than
n_predict tokens WITHOUT an EOS stop. Invalid rows are kept and counted, never dropped.
"""
import json, sys, time, hashlib, urllib.request

port, out, seedbase, kind, modes, nmaxes = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5].split(","), sys.argv[6]
deadline = float(sys.argv[7]) if len(sys.argv) > 7 else None
NMAX = [int(x) for x in nmaxes.split(",") if x]
assert kind in ("dspark", "plain")
P = [
 "The history of the printing press begins in the fifteenth century, when",
 "def quicksort(arr):\n    \"\"\"Sort a list using quicksort.\"\"\"\n",
 "Photosynthesis is the process by which plants convert light energy into chemical energy. The process has two main stages:",
 "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,",
 "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA: Let's think step by step.",
 "| Country | Capital | Population (millions) |\n|---|---|---|\n| France | Paris | 68 |\n",
 "The French Revolution was a period of political and societal change in France that began in 1789 and",
 "import numpy as np\n\ndef softmax(x):\n",
 "Dear hiring manager,\n\nI am writing to apply for the position of",
 "In mathematics, a prime number is a natural number greater than 1 that",
 "The recipe for a classic margherita pizza: first, prepare the dough by",
 "SELECT customer_id, COUNT(*) AS orders\nFROM orders\n",
 "Once upon a time, in a small village at the edge of a great forest,",
 "The main differences between TCP and UDP are",
 "Monday, Tuesday, Wednesday, Thursday,",
 "#include <stdio.h>\n\nint main(void) {\n",
]
NPRED = 128
SAMPLING = {
    "card": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
    "stoch": {"temperature": 0.7, "top_p": 1.0, "top_k": 0},                       # == w6.1/client.py, unchanged
    "greedy": {"temperature": 0.0, "top_k": 1},
}


def get(path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30))


try:
    props = get("/props")
    dgs = (props.get("default_generation_settings") or {}).get("params") or props.get("default_generation_settings") or {}
except Exception as e:  # noqa: BLE001
    dgs = {"error": str(e)}
json.dump({"modes": {m: SAMPLING[m] for m in modes}, "server_defaults_readback": dgs, "n_predict": NPRED, "kind": kind,
           "seedbase": seedbase, "note": "stoch/greedy leave unlisted samplers at the server default (see server_defaults_readback)"},
          open(out + "/sampling.json", "w"), indent=1)

rows = []


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def run(nmax, mode):
    tot_d = tot_a = 0
    for i, p in enumerate(P):
        if deadline and time.time() > deadline - 60:
            print("DEADLINE: stopping before the next request", flush=True)
            json.dump(rows, open(out + "/rows.json", "w"), indent=1)
            sys.exit(75)
        s = SAMPLING[mode]
        seed = (seedbase + i) if mode != "greedy" else 1234 + i
        body = {"prompt": p, "n_predict": NPRED, "cache_prompt": False, **s, "seed": seed}
        if kind == "dspark":
            body["speculative.n_max"] = nmax
        row = {"kind": kind, "mode": mode, "n_max": nmax if kind == "dspark" else None, "prompt": i, "seed": seed,
               "sampling": {k: body[k] for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty") if k in body}}
        try:
            t0 = time.time()
            r = post(body)
            row["wall_s"] = time.time() - t0
            t = r.get("timings", {})
            d, a = t.get("draft_n", 0) or 0, t.get("draft_n_accepted", 0) or 0
            tot_d += d; tot_a += a
            txt = r.get("content", "")
            pn = t.get("predicted_n")
            row.update({"draft_n": d, "draft_n_accepted": a, "predicted_n": pn, "predicted_per_second": t.get("predicted_per_second"),
                        "predicted_ms": t.get("predicted_ms"), "stop_type": r.get("stop_type"),
                        "text_sha": hashlib.sha256(txt.encode()).hexdigest()[:16], "text": txt if mode == "greedy" else txt[:300]})
            bad = []
            if not t or pn is None or not t.get("predicted_per_second"): bad.append("no_timings")
            if kind == "dspark" and d == 0: bad.append("no_draft")
            if pn is not None and pn < NPRED and r.get("stop_type") != "eos": bad.append("short_without_eos")
            row["valid"] = not bad; row["invalid_reasons"] = bad
        except Exception as e:  # noqa: BLE001
            row.update({"valid": False, "invalid_reasons": ["exception:" + repr(e)[:200]]})
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
    print(f"SUMMARY kind={kind} {mode} n_max={nmax if kind == 'dspark' else '-'} draft_n={tot_d} accepted={tot_a} alpha={tot_a / max(1, tot_d):.4f}", flush=True)


for mode in modes:
    for nmax in (NMAX if kind == "dspark" and mode != "greedy" else ([3] if kind == "dspark" else [None])):
        run(nmax, mode)
        json.dump(rows, open(out + "/rows.json", "w"), indent=1)
json.dump(rows, open(out + "/rows.json", "w"), indent=1)
