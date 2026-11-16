#!/usr/bin/env python3
"""AGG-1.15 step 1 (PREDECLARED_AGG115.md): the 6 FY2022 long_term_debt cells ``agg_pipeline.v2`` lost vs AGG-1.4 r3, OFFLINE.

No model, no server, no process signalling. Reads the saved AGG-1.12 ingest ``t_window`` records and the AGG-1.11 step-2 records,
replays the typed L4 finish (``keep_slot`` -> ``drop_unquoted`` -> ``reconcile``) under the current rule and under the opt-in
``keep_slot`` harvest (``facts_typed.KEEP_SLOT_HARVEST``), writes the DELTA into COPIES of the stores, re-canonicalises and scores
the 25 backfill, 211 FY2022 and 49 AGG-1.11 cells. Retrieval ranks of the gold windows are computed over the stored chunks.

    python3 agg115.py --orch-root <orchestrator worktree> --scratch <dir> --out agg115_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

JOB = Path("/mnt/raid0/llm/tmp/agg111s2-cpu-20261116")
SEED = JOB / "seed"
LOST = {"RETRIEVAL": ("LOCKHEEDMARTIN_2022_10K", "AMD_2022_10K", "MGMRESORTS_2022_10K"),
        "CANONICAL": ("PEPSICO_2022_10K", "COSTCO_2022_10K", "JOHNSON_JOHNSON_2022_10K")}
FIELD = "long_term_debt"
FY = "2022"
LTD_QUERY = "long-term debt net of current portion excluding current portion total long-term debt"


def sha256(p: Path) -> str:
    import hashlib  # noqa: PLC0415
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.open() if line.strip()]


# --------------------------------------------------------------------------- replay of the typed finish
def replay(ft: Any, recs: list[dict[str, Any]], planned: set[str], harvest: bool) -> list[dict[str, Any]]:
    """One document's L4 facts from its window records: keep_slot (or the harvest), the quote rule, reconcile."""
    from src.aggregation import facts_llm  # noqa: PLC0415
    kept_in = []
    for r in recs:
        chosen = set(r["chosen_by"])
        for f in r["facts"]:
            ok = ft.keep_slot_harvest(f, chosen, planned) if harvest else ft.keep_slot(f, chosen)
            if ok:
                kept_in.append(f)
    kept, _drop = ft.drop_unquoted(kept_in)
    return facts_llm.reconcile(list(kept))[0]


def delta_store(ft: Any, src: Path, dst: Path, plans: dict[str, dict[str, Any]], corpus_id: str) -> dict[str, Any]:
    """Copy ``src`` to ``dst`` and insert, per plan, the harvest facts the current rule does not produce. Facts the current rule
    produces and the harvest does not are counted (must be 0: the harvest only adds rows of unplanned slots)."""
    import phase0_arms as P  # noqa: PLC0415
    if dst.exists():
        raise SystemExit(f"{dst} exists; use a fresh --scratch (this script never removes files)")
    P.copy_store(src, dst)
    st = P.open_store(dst)
    added: list[dict[str, Any]] = []
    missing = 0
    try:
        for pid, p in sorted(plans.items()):
            base = replay(ft, p["recs"], p["planned"], False)
            harv = replay(ft, p["recs"], p["planned"], True)
            bids = {f["fact_id"] for f in base}
            hids = {f["fact_id"] for f in harv}
            missing += len(bids - hids)
            new = [f for f in harv if f["fact_id"] not in bids]
            have = {r[0] for r in st.conn.execute("SELECT fact_id FROM facts WHERE doc_id = ?", [p["doc"]])}
            new = [f for f in new if f["fact_id"] not in have]
            st.write_facts(new)
            added += [dict(f, plan=pid) for f in new]
        st.conn.commit()
    finally:
        st.close()
    by_field: dict[str, int] = {}
    for f in added:
        by_field[f["field"]] = by_field.get(f["field"], 0) + 1
    return {"added": len(added), "added_by_field": by_field, "base_not_in_harvest": missing,
            "ltd_added": [{k: f.get(k) for k in ("plan", "doc_id", "period", "value", "unit", "quote", "chunk_id")}
                          for f in added if f["field"] == FIELD]}


def ingest_plans(results: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(results):
        if r.get("kind") == "t_window" and r.get("final"):
            out.setdefault(r["doc"], {"doc": r["doc"], "recs": []})["recs"].append(r)
    import agg112 as G112  # noqa: PLC0415
    planned = set(G112.orch_symbols()["ft"].SCHEMA_FIELDS)
    for p in out.values():
        p["planned"] = planned
    return out


def s2_plans(results: Path, store: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(results):
        if r.get("kind") == "t_window" and r.get("final") and r["plan"].startswith(store + ":"):
            out.setdefault(r["plan"], {"doc": r["doc"], "recs": []})["recs"].append(r)
    for p in out.values():
        p["planned"] = {f for r in p["recs"] for f in r["chosen_by"]}
    return out


# --------------------------------------------------------------------------- retrieval
def gold_regex(gold: float) -> list[re.Pattern[str]]:
    from agg111_triage import num_re  # noqa: PLC0415
    pats = [num_re(gold)]
    th = gold * 1000.0
    if abs(th - round(th)) < 1e-6:     # the thousands-printed form (MGM: 7,432,817)
        pats.append(num_re(round(th)))
    return pats


def has_gold(text: str, pats: list[re.Pattern[str]]) -> bool:
    from agg111_triage import squash  # noqa: PLC0415
    s = squash(text)
    return any(p.search(s) for p in pats)


def full_rank(wins: list[dict[str, Any]], query: str) -> list[str]:
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    idx = BM25Index([w["text"] for w in wins])
    return [wins[i]["chunk_id"] for i, _s in idx.scores(query)]


def v2_rank(wins: list[dict[str, Any]], query: str) -> list[str]:
    from src.aggregation.passage_rank import numeric_frac  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index, CANDIDATES_PER_RANKER  # noqa: PLC0415
    idx = BM25Index([w["text"] for w in wins])
    pool = idx.scores(query)[:CANDIDATES_PER_RANKER]
    key = sorted((-(s * (1.0 + numeric_frac(wins[i]["text"]))), i) for i, s in pool)
    return [wins[i]["chunk_id"] for _k, i in key]


def rank_of(order: list[str], gold: set[str]) -> int | None:
    for i, c in enumerate(order, 1):
        if c in gold:
            return i
    return None


def retrieval(ft: Any, r2: Path, ingest: dict[str, dict[str, Any]], golds: dict[str, float]) -> dict[str, Any]:
    import agg112 as G112  # noqa: PLC0415
    import agg16 as G16  # noqa: PLC0415
    import agg111s2 as S2  # noqa: PLC0415
    descs = G112.descs_for(FY)
    q_desc = f"{FIELD.replace('_', ' ')} {descs[FIELD]}"
    variants = {
        "current_union": None,
        "union+ltd_query_k3": (LTD_QUERY, 3),
        "union+ltd_query_k5": (LTD_QUERY, 5),
        "union_field_query_k5": (ft.FIELD_QUERY[FIELD], 5),
    }
    cells = {}
    for doc in LOST["RETRIEVAL"] + LOST["CANONICAL"]:
        st = G16.mem_store(r2, [doc])
        try:
            wins = S2.doc_windows(st).get(doc, [])
        finally:
            st.close()
        pats = gold_regex(golds[doc])
        gw = {w["chunk_id"] for w in wins if has_gold(w["text"], pats)}
        gw_strict = {w["chunk_id"] for w in wins if has_gold(w["text"], pats[:1])}
        called = {r["chunk_id"]: r for r in ingest[doc]["recs"]}
        ltd_rows = {cid: [f for f in r["facts"] if f["field"] == FIELD] for cid, r in called.items()}
        gold_called = {cid: [{k: f.get(k) for k in ("period", "value", "unit", "quote")} for f in ltd_rows[cid]]
                       for cid in sorted(gw & set(called))}
        union = [w["chunk_id"] for w in ft.union_candidates(wins, FIELD, descs[FIELD])]
        per_var = {}
        for name, v in variants.items():
            ids = list(union)
            if v is not None:
                q, k = v
                ids += [c for c in full_rank(wins, q)[:k] if c not in ids]
            per_var[name] = {"n_windows": len(ids), "gold_in": sorted(gw & set(ids)), "new_uncalled": sorted(set(ids) - set(called))}
        ranks = {"bm25_desc_query": rank_of(full_rank(wins, q_desc), gw),
                 "bm25_field_query": rank_of(full_rank(wins, ft.FIELD_QUERY[FIELD]), gw),
                 "bm25_ltd_query": rank_of(full_rank(wins, LTD_QUERY), gw),
                 "v2_desc_query": rank_of(v2_rank(wins, q_desc), gw),
                 "v2_field_query": rank_of(v2_rank(wins, ft.FIELD_QUERY[FIELD]), gw),
                 "v2_ltd_query": rank_of(v2_rank(wins, LTD_QUERY), gw)}
        cells[doc] = {"gold": golds[doc], "n_doc_windows": len(wins), "gold_windows": sorted(gw),
                      "gold_windows_strict_regex": sorted(gw_strict), "gold_windows_called_in_ingest": gold_called,
                      "n_called_in_ingest": len(called), "ranks": ranks, "variants": per_var}
    return cells


# --------------------------------------------------------------------------- scoring
def score_211(store_path: Path, spec: Any, cells: list[Any], corpus_id: str) -> dict[str, dict[str, bool]]:
    import agg112 as G112  # noqa: PLC0415
    import agg19 as G19  # noqa: PLC0415
    import phase0_arms as P  # noqa: PLC0415
    st = P.open_store(store_path)
    try:
        spec.refresh_canonical(st, corpus_id)
        st.conn.commit()
        return G112.readings(G19.fact_rows(st.conn, corpus_id), st.canonical_facts(corpus_id), cells)
    finally:
        st.close()


def canon_fy(path: Path, cells: list[dict[str, Any]], rule: str) -> dict[str, bool]:
    import agg16 as G16  # noqa: PLC0415
    from src.aggregation.store import select_canonical  # noqa: PLC0415
    kw = {"rule": rule}
    return G16.canon_ok_fy(select_canonical(G16.saved_rows(path), **kw)[0], cells)


def diff(before: dict[str, bool], after: dict[str, bool]) -> dict[str, Any]:
    return {"n": len(before), "before": sum(before.values()), "after": sum(after.values()),
            "fixed": sorted(k for k in after if after[k] and not before.get(k)),
            "regressed": sorted(k for k in before if before[k] and not after.get(k))}


def hits(d: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {k: v["verdict"] == "HIT" for k, v in d.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orch-root", required=True, type=Path)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.orch_root.resolve()))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    import agg111s2 as S2  # noqa: PLC0415
    import agg112 as G112  # noqa: PLC0415
    import phase1_arms as X  # noqa: PLC0415
    X.configure(G112.CORPUS_ID)
    ft = G112.orch_symbols()["ft"]
    spec = G112.orch_symbols()["pl"].pipeline_spec(G112.PIPELINE)
    from src.aggregation.store import CANONICAL_RULE, CANONICAL_RULE_V2  # noqa: PLC0415
    a.scratch.mkdir(parents=True, exist_ok=False)
    cid = G112.CORPUS_ID
    r2 = SEED / "store_agg112r2.sqlite"
    ingest_res = SEED / "agg112ingest_results.jsonl"
    s2_res = JOB / "out" / "results.jsonl"
    inputs = {str(p): sha256(p) for p in (r2, ingest_res, s2_res, JOB / "out/store_agg13.sqlite", JOB / "out/store_agg14.sqlite",
                                          SEED / "store_agg14r3.sqlite", HERE / "PREDECLARED_AGG115.md")}
    cells49 = S2.load_cells(HERE)
    inp112 = G112.load_inputs(HERE)
    cells211 = inp112["cells"]
    golds = {c["doc"]: c["gold"] for c in cells49["agg14"] if c["field"] == FIELD and c["doc"] in sum(LOST.values(), ())}

    # parity: replaying the current rule must reproduce the r2 store's L4 facts
    ingest = ingest_plans(ingest_res)
    import sqlite3  # noqa: PLC0415
    c = sqlite3.connect(f"file:{r2}?immutable=1", uri=True)
    stored = {r[0] for r in c.execute("SELECT fact_id FROM facts WHERE extractor = 'l4_llm.v1'")}
    c.close()
    replayed = {f["fact_id"] for p in ingest.values() for f in replay(ft, p["recs"], p["planned"], False)}
    parity = {"stored_l4": len(stored), "replayed_l4": len(replayed), "stored_not_replayed": len(stored - replayed),
              "replayed_not_stored": len(replayed - stored)}

    # 211 + 25 on a copy of the r2 store
    base211 = a.scratch / "r2_base.sqlite"
    shutil.copyfile(r2, base211)
    before211 = score_211(base211, spec, cells211, cid)
    h211 = a.scratch / "r2_h1.sqlite"
    d211 = delta_store(ft, r2, h211, ingest, cid)
    after211 = score_211(h211, spec, cells211, cid)
    b25, a25 = S2.score_store(r2, cells49["agg14"]), S2.score_store(h211, cells49["agg14"])
    c25 = {r: diff(canon_fy(r2, cells49["agg14"], r), canon_fy(h211, cells49["agg14"], r)) for r in (CANONICAL_RULE, CANONICAL_RULE_V2)}
    six = {}
    for step, docs in LOST.items():
        for doc in docs:
            k = S2.key(doc, FIELD, FY)
            six[k] = {"step_agg111s2": step, "before": b25[k]["verdict"], "after": a25[k]["verdict"],
                      "after_candidate": a25[k]["candidate"],
                      "canonical_v2_after": canon_fy(h211, cells49["agg14"], CANONICAL_RULE_V2)[k]}

    # 49 on copies of the AGG-1.11 step-2 work stores
    r49 = {}
    for name in ("agg13", "agg14"):
        src = JOB / "out" / f"store_{name}.sqlite"
        dst = a.scratch / f"s2_{name}_h1.sqlite"
        d = delta_store(ft, src, dst, s2_plans(s2_res, name), cid)
        cs = cells49[name]
        r49[name] = {"delta": {k: v for k, v in d.items() if k != "ltd_added"}, "m1": diff(hits(S2.score_store(src, cs)), hits(S2.score_store(dst, cs))),
                     "canonical_v1": diff(canon_fy(src, cs, CANONICAL_RULE), canon_fy(dst, cs, CANONICAL_RULE)),
                     "canonical_v2": diff(canon_fy(src, cs, CANONICAL_RULE_V2), canon_fy(dst, cs, CANONICAL_RULE_V2))}

    # canonical.v3 (opt-in) vs canonical.v2 on every store, the harvested ones included
    import agg19 as G19  # noqa: PLC0415
    from src.aggregation.store import CANONICAL_RULE_V3, select_canonical  # noqa: PLC0415
    v3: dict[str, Any] = {}
    for name, path in (("r2", r2), ("r2_h1", h211)):
        import sqlite3 as _s  # noqa: PLC0415
        conn = _s.connect(f"file:{path}?mode=ro", uri=True)
        rows = G19.fact_rows(conn, cid)
        conn.close()
        v3[name] = {"fy2022_211": diff(G19.canon_ok(select_canonical(rows, rule=CANONICAL_RULE_V2)[0], cells211),
                                       G19.canon_ok(select_canonical(rows, rule=CANONICAL_RULE_V3)[0], cells211)),
                    "backfill25": diff(canon_fy(path, cells49["agg14"], CANONICAL_RULE_V2), canon_fy(path, cells49["agg14"], CANONICAL_RULE_V3))}
    for name in ("agg13", "agg14"):
        for tag, path in (("work", JOB / "out" / f"store_{name}.sqlite"), ("work_h1", a.scratch / f"s2_{name}_h1.sqlite")):
            v3[f"{name}_{tag}"] = diff(canon_fy(path, cells49[name], CANONICAL_RULE_V2), canon_fy(path, cells49[name], CANONICAL_RULE_V3))
    for k in six:
        six[k]["canonical_v3_after"] = canon_fy(h211, cells49["agg14"], CANONICAL_RULE_V3)[k]

    res = {
        "task": "AGG-1.15 step 1 (offline, no model)",
        "orch_head": __import__("subprocess").run(["git", "-C", str(a.orch_root), "rev-parse", "HEAD"], capture_output=True,
                                                  text=True).stdout.strip(), "predeclared": "PREDECLARED_AGG115.md", "inputs_sha256": inputs,
        "label": "IN-SAMPLE, POST-HOC for the 6 cells; the 211 / 49 are a regression check",
        "parity_current_rule_vs_r2_store": parity,
        "retrieval": retrieval(ft, r2, ingest, golds),
        "h1_keep_slot_harvest": {
            "rule": ft.KEEP_SLOT_HARVEST, "delta_r2": {k: v for k, v in d211.items()},
            "six_cells": six, "backfill25_m1": diff(hits(b25), hits(a25)), "backfill25_canonical": c25,
            "fy2022_211": {"raw": diff(before211["raw"], after211["raw"]), "canonical_v2": diff(before211["canonical"], after211["canonical"])},
            "agg111_49": r49,
            "extra_calls": 0},
        "c_canonical_v3_vs_v2": v3,
    }
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"v3": v3, "parity": parity, "six": six, "211": res["h1_keep_slot_harvest"]["fy2022_211"],
                      "25": res["h1_keep_slot_harvest"]["backfill25_m1"], "49": {k: {kk: vv for kk, vv in v.items() if kk != 'delta'} for k, v in r49.items()}},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
