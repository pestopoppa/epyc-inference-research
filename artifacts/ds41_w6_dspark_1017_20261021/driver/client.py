import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
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
rows = []
for nmax in (2, 3):
    tot_d = tot_a = 0
    for i, p in enumerate(P):
        body = {"prompt": p, "n_predict": 128, "temperature": 0.7, "top_p": 1.0, "top_k": 0, "seed": 1234 + i,
                "speculative.n_max": nmax, "cache_prompt": False}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        t = r.get("timings", {})
        d, a = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        tot_d += d; tot_a += a
        rows.append({"n_max": nmax, "prompt": i, "draft_n": d, "draft_n_accepted": a, "predicted_n": t.get("predicted_n"),
                     "predicted_per_second": t.get("predicted_per_second"), "text": r.get("content", "")[:300]})
        print(json.dumps(rows[-1])[:400], flush=True)
    print(f"SUMMARY n_max={nmax} draft_n={tot_d} accepted={tot_a} alpha={tot_a/max(1,tot_d):.4f}", flush=True)
json.dump(rows, open(out + "/rows.json", "w"), indent=1)
