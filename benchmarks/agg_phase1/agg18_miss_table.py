#!/usr/bin/env python3
"""AGG-1.8: classify every AGG-1.1 P-large miss by reason. No model call; store opened read-only (mode=ro).

Inputs (AGG-1.1 arms row agg1-arms-gpu-20261111): results.jsonl (final P-large answer records), store.sqlite
(the 28-doc store the arms read), golds.jsonl + facts_sheet.csv (verified cell golds). Output: agg18_miss_table.json.
Also a no-model re-score: saved SQL re-executed on a COPY of facts_canonical whose needed cells are corrected to
the verified golds (upper bound of the M1-gap fix on single-field questions)."""
import csv, json, re, sqlite3, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out")
TOL = 0.005

golds = {}
for l in open(HERE / "golds.jsonl"):
    d = json.loads(l); golds[d["id"]] = d
sheet = {}
for r in csv.DictReader(open(HERE / "facts_sheet.csv")):
    if r["status"] == "filled" and r["value_usd_m"].strip():
        sheet[(r["doc_name"], r["metric"])] = (r["company"], float(r["value_usd_m"].replace(",", "")))
sheet_na = {(r["doc_name"], r["metric"]) for r in csv.DictReader(open(HERE / "facts_sheet.csv")) if r["status"] == "na"}
policy_excl = {d for g in golds.values() for d in (g.get("excluded_by_rule") or [])}
ans = [json.loads(l) for l in open(SRC / "results.jsonl")]
fin = {x["qid"]: x for x in ans if x.get("kind") == "answer" and x.get("arm") == "P-large" and x.get("final")}
db = sqlite3.connect(f"file:{SRC}/store.sqlite?mode=ro", uri=True)
canon = {(r[0], r[1]): r for r in db.execute(
    "select doc_id, field, entity, value, unit from facts_canonical where period='2022'")}

def fields_of(expr):
    return [f for f in re.split(r"[/-]", expr)]

def cell_state(doc, field, gold):
    r = canon.get((doc, field))
    if r is None: return "missing"
    v = r[3]
    if v is None: return "missing"
    if abs(abs(v) - abs(gold)) <= TOL * abs(gold) + 1e-9: return "ok"
    if gold and (abs(abs(v) / abs(gold) - 1000) < 5 or abs(abs(v) / abs(gold) - 0.001) < 1e-4): return "unit1000"
    return "wrong"

def needed_cells(q):
    fl = set(fields_of(q["expr"]))
    if q["where"]: fl.add(q["where"]["expr"])
    return sorted((d, f) for (d, f) in sheet if f in fl)

def spurious_cells(q):
    """Sheet says NA (no such line) but the store holds a value, excluding documents a question excludes by rule."""
    fl = set(fields_of(q["expr"]))
    if q["where"]: fl.add(q["where"]["expr"])
    return sorted((d, f) for (d, f) in sheet_na if f in fl and d not in policy_excl and (d, f) in canon
                  and canon[(d, f)][3] is not None)

def sql_defect(x):
    s = x["sql"]
    if not x["sql_valid"]: return "sql_invalid_flag"
    try: db.execute(s).fetchall()
    except Exception as e: return f"exec_error:{type(e).__name__}"
    if re.search(r"SUM\((field|quote)\)", s): return "degenerate:SUM(field|quote)"
    if len(re.findall(r"entity\s*=", s)) >= 3: return "degenerate:entity-enumeration"
    if re.search(r"(value,\s*){5,}", s): return "degenerate:repeat"
    return None

def run_sql(con, s):
    try: return con.execute(s).fetchall()
    except Exception: return None

def sql_scores(rows, q):
    if not rows: return False
    op, gold = q["op"], q["gold"]
    try:
        if op in ("argmax", "argmin"): return str(rows[0][0]).lower() == str(gold).lower()
        if op == "top3": return [str(r[0]).lower() for r in rows[:3]] == [str(g).lower() for g in gold]
        v = float(rows[0][0]); g = float(gold)
        return abs(v - g) <= (0 if op.startswith("count") else TOL * abs(g)) + 1e-9
    except Exception: return False

# corrected copy: needed cells replaced by verified golds
mem = sqlite3.connect(":memory:")
db.backup(mem)
def corrected_con(q):
    c = sqlite3.connect(":memory:"); mem.backup(c)
    for d, f in needed_cells(q):
        comp, g = sheet[(d, f)]
        c.execute("delete from facts_canonical where doc_id=? and field=? and period='2022'", (d, f))
        c.execute("insert into facts_canonical(corpus_id,doc_id,field,period,entity,value,unit) values('x',?,?,'2022',?,?, 'USD_millions')", (d, f, comp, g))
    for d, f in spurious_cells(q):
        c.execute("delete from facts_canonical where doc_id=? and field=? and period='2022'", (d, f))
    return c

out = []
for qid, x in sorted(fin.items()):
    if x["correct"] is True: continue
    q = golds[qid]
    derived = bool(re.search(r"[/-]", q["expr"]))
    filtered = bool(q["where"])
    cells = needed_cells(q)
    states = {f"{d}|{f}": cell_state(d, f, sheet[(d, f)][1]) for d, f in cells}
    for d, f in spurious_cells(q): states[f"{d}|{f}"] = "spurious"
    bad = {k: v for k, v in states.items() if v != "ok"}
    defect = sql_defect(x)
    trunc = x["finish_reason"] == "length"
    plan_wrong = (derived or filtered) and not defect
    if defect: cat = "sql_malformed_or_degenerate"
    elif plan_wrong: cat = "wrong_plan_derived_or_filtered"
    elif bad: cat = "store_gap_m1"
    else:
        rows = run_sql(db, x["sql"])
        cat = "scorer_or_format" if sql_scores(rows, q) else "other"
    # no-model re-score on corrected store
    rows_fix = run_sql(corrected_con(q), x["sql"]) if not defect else None
    flips = bool(sql_scores(rows_fix, q)) if rows_fix is not None else False
    n_by = {s: sum(1 for v in bad.values() if v == s) for s in ("missing", "wrong", "unit1000", "spurious")}
    out.append(dict(qid=qid, family=q["family"], op=q["op"], expr=q["expr"], where=bool(q["where"]),
                    continuity=q["continuity"], category=cat, sql_defect=defect, answer_truncated=trunc,
                    n_cells=len(cells), n_bad_cells=len(bad), bad_by_state=n_by,
                    bad_cells=sorted(bad)[:12], rescore_on_gold_cells_correct=flips))
# distinct-cell decomposition over every verified cell of the fields the 40 misses need
raw = {}
for d, f, v, u in db.execute("select doc_id, field, value, unit from facts where period like '%2022%' and value is not null"):
    raw.setdefault((d, f), []).append(v)
need_all = sorted({c for qid in {r["qid"] for r in out} for c in needed_cells(golds[qid])})
cell_decomp = Counter()
cell_rows = []
for d, f in need_all:
    g = sheet[(d, f)][1]
    st = cell_state(d, f, g)
    in_raw = any(abs(abs(v) - abs(g)) <= TOL * abs(g) + 1e-9 for v in raw.get((d, f), []))
    kind = "ok" if st == "ok" else ("canonical_chose_wrong_or_dropped_but_candidate_correct" if in_raw else "never_extracted_correctly")
    cell_decomp[(st, kind)] += 1
    if st != "ok": cell_rows.append(dict(doc=d, field=f, gold=g, canonical_state=st, correct_candidate_in_facts=in_raw))
sp_cells = sorted({c for r in out for c in spurious_cells(golds[r["qid"]])})
for d, f in sp_cells:
    cell_decomp[("spurious", "na_in_gold_but_value_in_store")] += 1
    cell_rows.append(dict(doc=d, field=f, gold="NA", canonical_state="spurious", correct_candidate_in_facts=False))
json.dump(dict(schema="agg18_miss_table.v1", source=str(SRC), n_p_large=len(fin), n_miss=len(out), cell_decomposition={f"{a}/{b}":n for (a,b),n in cell_decomp.items()}, bad_cells=cell_rows, rows=out),
          open(HERE / "agg18_miss_table.json", "w"), indent=1)
print(len(fin), "answered;", len(out), "misses")
print(Counter(r["category"] for r in out))
print("truncated:", sum(r["answer_truncated"] for r in out), " rescore flips:", sum(r["rescore_on_gold_cells_correct"] for r in out))
for c in sorted({r["category"] for r in out}):
    print(c, [r["qid"] for r in out if r["category"] == c])
print("misses with >=1 bad needed cell:", sum(1 for r in out if r["n_bad_cells"]),
      " bad cell states:", Counter(k for r in out for k, v in r["bad_by_state"].items() for _ in range(v)))
print("flips:", [r["qid"] for r in out if r["rescore_on_gold_cells_correct"]])

print("distinct needed cells:", len(need_all), dict(cell_decomp))

# No-model probe of the canonical-tier fix: prefer an l4_llm candidate over the l3_textrow one (everything else equal).
def ok(v, g): return v is not None and abs(abs(v) - abs(g)) <= TOL * abs(g) + 1e-9
r1 = r2 = 0
fixed, broken = [], []
for d, f in need_all:
    g = sheet[(d, f)][1]
    c = canon.get((d, f)); v1 = c[3] if c else None
    l4 = [v for (v,) in db.execute("select value from facts where doc_id=? and field=? and period like '%2022%' and extractor like 'l4_llm%' and value is not null order by (confidence='high') desc, abs(value) desc", (d, f))]
    v2 = l4[0] if l4 else v1
    r1 += ok(v1, g); r2 += ok(v2, g)
    if ok(v2, g) and not ok(v1, g): fixed.append(f"{d}|{f}")
    if ok(v1, g) and not ok(v2, g): broken.append(f"{d}|{f}")
print(f"tier probe over {len(need_all)} cells: canonical ok {r1}, l4-preferred ok {r2}; fixed {len(fixed)} broken {len(broken)}", broken)
probe = dict(cells=len(need_all), canonical_ok=r1, l4_preferred_ok=r2, fixed=fixed, broken=broken,
             fixed_by_field=dict(Counter(x.split("|")[1] for x in fixed)), broken_by_field=dict(Counter(x.split("|")[1] for x in broken)))
j = json.load(open(HERE / "agg18_miss_table.json")); j["tier_probe"] = probe
json.dump(j, open(HERE / "agg18_miss_table.json", "w"), indent=1)
