#!/usr/bin/env python3
"""AGG-1.9 DONE-WHEN 2: re-run the 14 saved AGG-1.1 store-gap SQLs offline (no model) on the corrected v2 canonical store and
score them with the frozen AGG-1.8 scorer (agg18_miss_table.py sql_scores, tolerance 0.005). Stores are opened mode=ro.
Also re-runs them on the AGG-1.1 store (must reproduce 0 of 14) and reads the gold-corrected upper bound from agg18_miss_table.json.
Usage: agg19_sql_rescore.py [--store PATH]"""
import argparse, hashlib, json, sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL = 0.005
AGG11 = Path("/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out")
DEFAULT_STORE = "/mnt/raid0/llm/tmp/agg112-arms-gpu-20261112-r2/out/store.sqlite"  # /mnt/raid0/llm/tmp/agg114-scratch/store.sqlite is gone

def sql_scores(rows, q):  # verbatim from agg18_miss_table.py
    if not rows: return False
    op, gold = q["op"], q["gold"]
    try:
        if op in ("argmax", "argmin"): return str(rows[0][0]).lower() == str(gold).lower()
        if op == "top3": return [str(r[0]).lower() for r in rows[:3]] == [str(g).lower() for g in gold]
        v = float(rows[0][0]); g = float(gold)
        return abs(v - g) <= (0 if op.startswith("count") else TOL * abs(g)) + 1e-9
    except Exception: return False

def run(path, sql):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try: return con.execute(sql).fetchall()
    except Exception as e: return f"ERR {type(e).__name__}: {e}"
    finally: con.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--store", default=DEFAULT_STORE); a = ap.parse_args()
    miss = json.load(open(HERE / "agg18_miss_table.json"))
    golds = {}
    for l in open(HERE / "golds.jsonl"):
        d = json.loads(l); golds[d["id"]] = d
    fin = {}
    for l in open(AGG11 / "results.jsonl"):
        x = json.loads(l)
        if x.get("kind") == "answer" and x.get("arm") == "P-large" and x.get("final"): fin[x["qid"]] = x
    sg = [r for r in miss["rows"] if r["category"] == "store_gap_m1"]
    assert len(sg) == 14, len(sg)
    sha = hashlib.sha256(open(a.store, "rb").read()).hexdigest()
    out = []
    for r in sg:
        q = golds[r["qid"]]; sql = fin[r["qid"]]["sql"]
        old = run(AGG11 / "store.sqlite", sql); new = run(a.store, sql)
        ok_old = sql_scores(old, q) if isinstance(old, list) else False
        ok_new = sql_scores(new, q) if isinstance(new, list) else False
        out.append(dict(qid=r["qid"], op=q["op"], expr=q["expr"], gold=q["gold"],
                        agg11_rows=str(old)[:200], v2_rows=str(new)[:200],
                        agg11_correct=ok_old, v2_correct=ok_new,
                        gold_corrected_upper_bound=r["rescore_on_gold_cells_correct"]))
    res = dict(schema="agg19_sql_rescore.v1", task="AGG-1.9 DONE-WHEN 2", predeclared="PREDECLARED_AGG19.md sha256 4eed298ac95022de532f0a30844ab6d7af684e261e7e1a3e4109ae749ad7917c",
               store=a.store, store_sha256=sha, canonical_rule="canonical.v2:v1+attributable_net_income+thousands_unit (facts_canonical)",
               scorer="agg18_miss_table.py sql_scores, tolerance 0.005", model_calls=0,
               n=14, v2_correct=sum(o["v2_correct"] for o in out), agg11_correct=sum(o["agg11_correct"] for o in out),
               upper_bound=sum(o["gold_corrected_upper_bound"] for o in out), per_question=out)
    json.dump(res, open(HERE / "agg19_sql_rescore.json", "w"), indent=1)
    for o in out: print(o["qid"], o["op"], o["expr"], "agg11", o["agg11_correct"], "v2", o["v2_correct"], "| gold", o["gold"], "| v2 rows", o["v2_rows"][:80])
    print("v2", res["v2_correct"], "/14  agg11", res["agg11_correct"], " upper", res["upper_bound"], " sha", sha)
main()
