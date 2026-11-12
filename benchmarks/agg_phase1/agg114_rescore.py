#!/usr/bin/env python3
"""AGG-1.14: enforce the value-in-quote rule in the typed L4 stage (``facts_typed.drop_unquoted``), re-canonicalise the saved
AGG-1.12 store and re-score the 58 saved P-large answers, OFFLINE (no model, no server).

Inputs are the saved AGG-1.12 artifacts: the CPU row's ``results.jsonl`` (one ``t_window`` record per typed call, carrying every
row the model returned) and ``store.sqlite``; the GPU row's ``results.jsonl`` (58 answers x 2 planners, each with its SQL) and
``store.sqlite``. Saved stores are never opened for writing: every store used here is a copy under ``--scratch``.

Part A (ingest replay). The typed windows are replayed exactly as ``agg112.finish_doc`` does (keep_slot gate, then the typed
stage's document finish, which now applies the quote rule): per document the ``l4_llm.v1`` facts are deleted from a store copy and
the document's kept rows go through ``TypedBackfill.finish_doc``. A BASELINE replay with the rule off must reproduce the saved
store's L4 facts and the recorded M1 (208 raw / 202 canonical) before the rule-on result is trusted.

Part B (P-large re-score). For each saved ``agg110.v1`` / ``agg11.v0`` answer whose path was ``sql``: the saved SQL is re-executed
on the saved GPU store (must reproduce the saved row count) and on the corrected store, with the pipeline's own execution code
(pivot view + null guard for ``agg110.v1``, the statement verbatim for ``agg11.v0``). Identical rows => the model would see the
same prompt, so the saved answer and its saved score stand. Changed rows => the saved answer is void; the rows are read by a
deterministic renderer (validated on the unchanged questions against the saved correctness) and scored by the frozen scorer
``phase0_arms.score_agg``. That score is a MECHANICAL reading of the rows, not a model answer, and is labelled as such. Questions
on another path (single_doc / whole_doc / agent) never read the fact table; their saved answers stand.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg110 as G110  # noqa: E402
import agg112 as G112  # noqa: E402
import agg19 as G19  # noqa: E402
import agg_phase0 as A  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402

CORPUS_ID = G112.CORPUS_ID
SQL_ROW_CAP = P.CFG.sql_row_cap
QUOTE_CHARS = P.CFG.quote_chars
OPS_NUMERIC = ("sum", "count_gt", "count_lt")


# --------------------------------------------------------------------------- Part A: replay the typed ingest
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def window_records(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """doc -> its final ``t_window`` records in file order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        if r.get("kind") == "t_window" and r.get("final"):
            out.setdefault(r["doc"], []).append(r)
    return out


def kept_rows(ft: Any, wrecs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The document's rows after the field gate: what ``agg112.finish_doc`` hands the typed stage."""
    out: list[dict[str, Any]] = []
    for r in wrecs:
        chosen = set(r["chosen_by"])
        out.extend(f for f in r["facts"] if ft.keep_slot(f, chosen))
    return out


def replay(store_path: Path, wins: dict[str, list[dict[str, Any]]], *, rule: bool, ft: Any, fl: Any, spec: Any
           ) -> dict[str, Any]:
    """Replay the L4 stage on a store copy. ``rule=False`` is the AGG-1.12 stage (reconcile only); ``rule=True`` runs the typed
    stage's own ``finish_doc`` (value-in-quote rule, then reconcile). Returns the open store and the dropped-row notes."""
    store = P.open_store(store_path)
    dropped: list[dict[str, Any]] = []
    n_in = 0
    for doc in sorted(wins):
        kept = kept_rows(ft, wins[doc])
        n_in += len(kept)
        store.conn.execute("DELETE FROM facts WHERE doc_id = ? AND extractor = ?", (doc, fl.EXTRACTOR_L4))
        store.conn.commit()
        if rule:
            out = ft.TypedBackfill(store, None).finish_doc(CORPUS_ID, kept, ft.SCHEMA_FIELDS)
            dropped.extend(out["quote_dropped"])
        else:
            facts, _conf = fl.reconcile(list(kept))
            store.write_facts(facts)
    store.conn.execute("DELETE FROM facts_canonical")
    store.conn.commit()
    canon_stats = spec.refresh_canonical(store, CORPUS_ID)
    return {"store": store, "dropped": dropped, "n_in": n_in, "canon_stats": canon_stats}


def readings_of(store: Any, cells: list[tuple[str, str, float]]) -> dict[str, dict[str, bool]]:
    rows = G19.fact_rows(store.conn, CORPUS_ID)
    return G112.readings(rows, store.canonical_facts(CORPUS_ID), cells)


def l4_signature(store: Any) -> list[tuple[Any, ...]]:
    cur = store.conn.execute("SELECT fact_id, doc_id, field, period, round(value, 9), unit, quote FROM facts "
                             "WHERE extractor = 'l4_llm.v1' ORDER BY fact_id")
    return [tuple(r) for r in cur.fetchall()]


def violation_report(ft: Any, wins: dict[str, list[dict[str, Any]]], stored_l4_ids: set[str]) -> dict[str, Any]:
    """Violating rows over every typed row: all rows the model returned, and the rows that enter the stage (after keep_slot)."""
    def tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
        bad = [f for f in rows if ft.value_in_quote(f) is None]
        basis: dict[str, int] = {}
        for f in rows:
            b = ft.value_in_quote(f) or "violation"
            basis[b] = basis.get(b, 0) + 1
        by_field: dict[str, int] = {}
        for f in bad:
            by_field[f["field"]] = by_field.get(f["field"], 0) + 1
        by_doc: dict[str, int] = {}
        for f in bad:
            by_doc[f["doc_id"]] = by_doc.get(f["doc_id"], 0) + 1
        return {"n_rows": len(rows), "n_violating": len(bad), "by_basis": dict(sorted(basis.items())),
                "by_field": dict(sorted(by_field.items())), "by_doc": dict(sorted(by_doc.items()))}
    all_rows = [f for ws in wins.values() for r in ws for f in r["facts"]]
    entering = [f for d in sorted(wins) for f in kept_rows(ft, wins[d])]
    bad_entering = [f for f in entering if ft.value_in_quote(f) is None]
    return {"all_window_rows": tally(all_rows), "rows_entering_stage": tally(entering),
            "violating_rows_entering_stage": [
                {"doc": f["doc_id"], "field": f["field"], "period": f["period"], "value": f["value"], "unit": f["unit"],
                 "declared_scale": f.get("declared_scale"), "quote": f["quote"],
                 "stored_in_agg112_store": f["fact_id"] in stored_l4_ids} for f in bad_entering]}


# --------------------------------------------------------------------------- Part B: re-execute the saved SQL
def exec_saved_sql(store: Any, rec: dict[str, Any], sg: Any) -> dict[str, Any]:
    """The saved statement on ``store`` with the same code the pipeline ran: ``agg110.v1`` builds the pivot view over the store's
    canonical rows and runs the null-guarded statement; ``agg11.v0`` runs the statement verbatim and truncates quotes."""
    out: dict[str, Any] = {"rows": [], "error": None}
    try:
        if rec["planner_version"] == G110.V1:
            canon = store.canonical_facts(CORPUS_ID)
            ents = sorted({r["entity"] for r in canon if r.get("entity")})
            cat = sg.pivot_catalog(canon, names=ents, fields=tuple(A.METRICS))
            sg.create_pivot_view(store.conn, CORPUS_ID, cat)
            sql = sg.null_guard(rec["sql"], cat)
        else:
            sql = rec["sql"]
        cur = store.conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(SQL_ROW_CAP)]
        if rec["planner_version"] != G110.V1:
            for r in rows:
                if isinstance(r.get("quote"), str):
                    r["quote"] = r["quote"][:QUOTE_CHARS]
        out["rows"], out["sql"] = rows, sql
    except Exception as e:  # noqa: BLE001 - recorded
        out["error"] = f"{type(e).__name__}:{str(e)[:160]}"
    return out


def render(op: str, rows: list[dict[str, Any]]) -> str:
    """Deterministic reading of the rows as a FINAL line (a MECHANICAL stand-in for the answer call; validated against the saved
    correctness on the questions whose rows did not change)."""
    if not rows:
        return "FINAL: no data"
    if op in ("argmax", "argmin"):
        r = rows[0]
        for k, v in r.items():
            if isinstance(v, str):
                return f"FINAL: {v}"
        return "FINAL: no data"
    if op == "top3":
        names = [next((v for v in r.values() if isinstance(v, str)), "") for r in rows[:3]]
        return "FINAL: " + "; ".join(names)
    for v in rows[0].values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f"FINAL: {v:,.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else f"FINAL: {v}"
    return "FINAL: no data"


def rows_key(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, sort_keys=True, default=str)


def rescore(gpu_results: Path, gpu_store: Path, new_store: Any, agg: list[dict[str, Any]], sg: Any, scratch: Path) -> dict[str, Any]:
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    old_copy = scratch / "gpu_store_copy.sqlite"
    if not old_copy.exists():
        P.copy_store(gpu_store, old_copy)
    old_store = CorpusStore(old_copy)
    recs = [r for r in read_jsonl(gpu_results) if r.get("kind") == "answer" and r.get("final")]
    gold = {q["id"]: q for q in agg}
    ref = json.loads((HERE / G112.REFERENCE_FILE).read_text())
    agg11 = {q: v["correct"] for q, v in ref["agg11"]["p_large"]["per_question"].items()}
    ids = [q["id"] for q in agg]
    per: dict[str, list[dict[str, Any]]] = {G110.V1: [], G110.V0: []}
    for r in sorted(recs, key=lambda r: (r["planner_version"], ids.index(r["qid"]))):
        q = gold[r["qid"]]
        row: dict[str, Any] = {"qid": r["qid"], "planner": r["planner_version"], "path": r.get("path"), "op": q["op"],
                               "saved_correct": r.get("correct") is True, "saved_status": r.get("status"), "agg11_correct": agg11.get(r["qid"])}
        if r.get("path") != "sql" or not r.get("sql_valid") or r.get("status") != "ok":
            row.update(kind="not_sql", rescored_correct=row["saved_correct"], changed=False,
                       note="answer did not read the fact table (or no valid SQL): saved answer and score stand")
            per[r["planner_version"]].append(row)
            continue
        o = exec_saved_sql(old_store, r, sg)
        n = exec_saved_sql(new_store, r, sg)
        row["saved_n_rows"], row["old_n_rows"], row["new_n_rows"] = r.get("n_rows"), len(o["rows"]), len(n["rows"])
        row["old_reproduces_saved_n_rows"] = (len(o["rows"]) == r.get("n_rows") and o["error"] is None) or \
            (o["error"] is not None and bool(r.get("sql_error")))
        row["sql"] = r["sql"]
        row["old_rows"], row["new_rows"] = o["rows"][:6], n["rows"][:6]
        row["new_error"] = n["error"]
        old_mech = P.score_agg(gold[r["qid"]], render(q["op"], o["rows"]))
        row["mechanical_old_correct"] = old_mech is True
        row["mechanical_agrees_with_saved_on_old_rows"] = (old_mech is True) == row["saved_correct"]
        if rows_key(o["rows"]) == rows_key(n["rows"]) and o["error"] == n["error"]:
            row.update(kind="sql_rows_unchanged", changed=False, rescored_correct=row["saved_correct"],
                       note="identical SQL result => identical answer prompt => saved answer and score stand")
        else:
            new_mech = P.score_agg(gold[r["qid"]], render(q["op"], n["rows"]))
            row.update(kind="sql_rows_changed", changed=True, mechanical_new_text=render(q["op"], n["rows"]),
                       mechanical_new_correct=new_mech is True, rescored_correct=new_mech is True,
                       note="saved answer void; rescored from the rows by the deterministic renderer + frozen scorer "
                            "(MECHANICAL, not a model answer)")
        per[r["planner_version"]].append(row)
    old_store.close()
    out: dict[str, Any] = {"planners": {}}
    for pv, rows in per.items():
        saved = {x["qid"]: x["saved_correct"] for x in rows}
        new = {x["qid"]: x["rescored_correct"] for x in rows}
        pr = G112.paired(new, agg11, ids)
        out["planners"][pv] = {
            "n": len(rows), "saved_correct": sum(saved.values()), "rescored_correct": sum(new.values()),
            "saved_accuracy": sum(saved.values()) / len(rows), "rescored_accuracy": sum(new.values()) / len(rows),
            "vs_agg11_saved": G112.paired(saved, agg11, ids), "vs_agg11_rescored": pr,
            "changed_questions": [x["qid"] for x in rows if x["changed"]],
            "answer_changed_questions": [x["qid"] for x in rows if x["saved_correct"] != x["rescored_correct"]],
            "regressed_vs_agg11": sorted(q for q in ids if agg11.get(q) and not new.get(q)),
            "fixed_vs_agg11": sorted(q for q in ids if new.get(q) and not agg11.get(q)),
            "n_sql_questions": sum(x["kind"] != "not_sql" for x in rows),
            "old_rows_not_reproducing_saved": [x["qid"] for x in rows if x.get("old_reproduces_saved_n_rows") is False],
            "renderer_agrees_on_old_rows": [sum(x.get("mechanical_agrees_with_saved_on_old_rows", False) for x in rows if x["kind"] != "not_sql"),
                                            sum(x["kind"] != "not_sql" for x in rows)],
            "rows": rows}
    return out


# --------------------------------------------------------------------------- driver
def run(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(Path(args.orch_root).resolve()))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    from src.aggregation import facts_llm as fl, facts_typed as ft, pipeline as pl, sql_grammar as sg  # noqa: PLC0415
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    inp = G112.load_inputs()
    ref = inp["reference"]
    cells = inp["cells"]
    agg = X.verified_agg(inp)
    spec = pl.pipeline_spec(G112.PIPELINE)
    cpu_recs = read_jsonl(Path(args.cpu_results))
    wins = window_records(cpu_recs)

    # the saved AGG-1.12 store, as recorded (copy; never written)
    orig = scratch / "orig_agg112_store.sqlite"
    base = scratch / "baseline_replay.sqlite"
    fixed = scratch / "fixed_replay.sqlite"
    for dst in (orig, base, fixed):
        if dst.exists():
            dst.unlink()
        P.copy_store(Path(args.cpu_store), dst)
    s_orig = P.open_store(orig)
    now_orig = readings_of(s_orig, cells)
    sig_orig = l4_signature(s_orig)
    stored_ids = {r[0] for r in sig_orig}
    s_orig.close()

    rb = replay(base, wins, rule=False, ft=ft, fl=fl, spec=spec)
    rf = replay(fixed, wins, rule=True, ft=ft, fl=fl, spec=spec)
    now_base, now_fix = readings_of(rb["store"], cells), readings_of(rf["store"], cells)
    held = set(ref["partition"]["held_out"])
    import subprocess  # noqa: PLC0415
    pin = subprocess.run(["git", "-C", str(args.orch_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out: dict[str, Any] = {"task": "AGG-1.14", "orch_pin": pin, "pipeline": G112.PIPELINE,
                           "inputs": {"cpu_results": str(args.cpu_results), "cpu_store": str(args.cpu_store),
                                      "gpu_results": str(args.gpu_results), "gpu_store": str(args.gpu_store)}}
    out["baseline_replay"] = {
        "l4_facts_match_saved_store": l4_signature(rb["store"]) == sig_orig,
        "n_l4_facts": len(sig_orig), "raw": sum(now_base["raw"].values()), "canonical": sum(now_base["canonical"].values()),
        "raw_cells_equal_saved": now_base["raw"] == now_orig["raw"], "canonical_cells_equal_saved": now_base["canonical"] == now_orig["canonical"],
        "note": "rule OFF; must reproduce the saved store before the rule-on numbers are trusted"}
    out["saved_agg112"] = {"raw": sum(now_orig["raw"].values()), "canonical": sum(now_orig["canonical"].values())}
    out["violations"] = violation_report(ft, wins, stored_ids)
    out["violations"]["dropped_notes_from_finish_doc"] = len(rf["dropped"])
    out["n_rows_entering_stage"] = rf["n_in"]
    out["n_l4_facts_after_rule"] = len(l4_signature(rf["store"]))
    reg = G112.regressions(ref, now_fix, held)
    reg_old = G112.regressions(ref, now_orig, held)
    out["corrected"] = {
        "raw": sum(now_fix["raw"].values()), "canonical": sum(now_fix["canonical"].values()), "n_cells": len(cells),
        "agg112_recorded": {"raw": 208, "canonical": 202}, "agg11": {"raw": ref["agg11"]["raw_found"], "canonical": ref["agg11"]["canonical_v1_ok"]},
        "raw_fixed_vs_agg11": reg["raw_fixed"], "raw_broken_vs_agg11": reg["raw_broken"],
        "canonical_fixed_vs_agg11": reg["canonical_fixed"], "canonical_broken_vs_agg11": reg["canonical_broken"],
        "n_raw_fixed": len(reg["raw_fixed"]), "n_raw_broken": len(reg["raw_broken"]),
        "n_canonical_fixed": len(reg["canonical_fixed"]), "n_canonical_broken": len(reg["canonical_broken"]),
        "agg112_saved": {"n_raw_fixed": len(reg_old["raw_fixed"]), "n_raw_broken": len(reg_old["raw_broken"]),
                         "n_canonical_fixed": len(reg_old["canonical_fixed"]), "n_canonical_broken": len(reg_old["canonical_broken"]),
                         "canonical_broken": reg_old["canonical_broken"]},
        "cells_lost_by_the_rule_vs_agg112": sorted(k for k, v in now_orig["raw"].items() if v and not now_fix["raw"][k]),
        "canonical_cells_lost_by_the_rule_vs_agg112": sorted(k for k, v in now_orig["canonical"].items() if v and not now_fix["canonical"][k]),
        "canonical_cells_gained_by_the_rule_vs_agg112": sorted(k for k, v in now_fix["canonical"].items() if v and not now_orig["canonical"][k]),
        "strata": G112.strata_report(ref, now_fix, held)}
    jj = "JOHNSON_JOHNSON_2022_10K|dividends_paid"
    cf = [r for r in rf["store"].canonical_facts(CORPUS_ID) if r.get("doc_id") == "JOHNSON_JOHNSON_2022_10K" and r.get("field") == "dividends_paid"]
    out["johnson_johnson_dividends_paid"] = {
        "canonical_ok_now": bool(now_fix["canonical"].get(jj)), "canonical_ok_agg112": bool(now_orig["canonical"].get(jj)),
        "canonical_ok_agg11": jj in set(ref["agg11"]["canonical_v1_ok_cells"]),
        "raw_ok_now": bool(now_fix["raw"].get(jj)),
        "canonical_rows_now": [{"period": r.get("period"), "value": r.get("value"), "unit": r.get("unit"), "quote": r.get("quote")} for r in cf]}
    if args.gpu_results:
        out["p_large"] = rescore(Path(args.gpu_results), Path(args.gpu_store), rf["store"], agg, sg, scratch)
    for r in (rb, rf):
        r["store"].close()
    # the amendment condition: no previously correct (AGG-1.1) cell regresses, raw or canonical
    out["no_previously_correct_cell_regresses"] = not out["corrected"]["raw_broken_vs_agg11"] and not out["corrected"]["canonical_broken_vs_agg11"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orch-root", required=True)
    ap.add_argument("--cpu-results", required=True)
    ap.add_argument("--cpu-store", required=True)
    ap.add_argument("--gpu-results")
    ap.add_argument("--gpu-store")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    c = res["corrected"]
    print(f"raw {c['raw']}/{c['n_cells']} canonical {c['canonical']} (AGG-1.12 {res['saved_agg112']}, AGG-1.1 {c['agg11']}); "
          f"violating rows {res['violations']['rows_entering_stage']['n_violating']}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
