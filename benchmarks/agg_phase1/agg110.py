#!/usr/bin/env python3
"""AGG-1.10: planner plan/SQL fixes for derived questions, GPU re-score of the 26 affected questions (library;
``run_agg110.py`` is the CLI).

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an injected
backend (``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in tests).

Implements ``benchmarks/agg_phase1/PREDECLARED_AGG110.md`` literally. Nothing under ``agg_phase0/`` and none of the pinned
Phase 1 files is edited: ``planner_version == "agg11.v0"`` runs ``phase1_arms.p_large_answer`` itself (Phase 0 / AGG-1.1
behaviour, byte for byte); ``"agg110.v1"`` replaces ONLY the SQL answer step (pivot view, bounded plan grammar, entity-conjunct
guard with one re-ask, null guard, SQL ``n_predict`` 256). Every answer record carries ``planner_version``.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import sys

HERE = Path(__file__).resolve().parent
P0DIR = HERE.parent / "agg_phase0"
for _p in (str(P0DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg_phase0 as A  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402

# --------------------------------------------------------------------------- pins
PREDECLARED_FILE = "PREDECLARED_AGG110.md"
PREDECLARED_SHA = "8fa6f293571d733f49f4fbdaa6f0ec84d96a854fdd63fea4f730e40863a60498"
AMENDMENT1_FILE = "PREDECLARED_AGG110_AMENDMENT1_20261112.md"
MISS_TABLE_SHA = "97ebad89a70e2802d4d1c12a5418389ee1471ae24709c8e747c42ba05dafa193"
V0, V1 = "agg11.v0", "agg110.v1"
SQL_N_PREDICT = {V0: 160, V1: 256}
LONGEST_STATEMENT_FRACTION = 0.60      # PREDECLARED: the longest legitimate reference statement <= 60% of the cap
CHARS_PER_TOKEN = 2.5                  # conservative (SQL tokenises to about 3 characters per token)
M4_CALLS_MAX, M4_PREFILL_MAX, M4_KILL_CALLS = 3, 2510, 8
PILOT_N, PILOT_WALL_S, BOOKED_S = 6, 600, 1800.0
PILOT_STRATA = ("continuity-malformed", "continuity-plan", "margin/top3", "where-filter sum", "current-ratio count",
                "dividends where")
AGG11_REF = {"median_calls": 2.0, "median_prefill": 644.5}      # AGG-1.1 over the same 26 (computed from results.jsonl)
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_PARTIAL = (X.RC_OK, X.RC_PREFLIGHT, X.RC_GATE, X.RC_PRECOND,
                                                                   X.RC_BUDGET, X.RC_PARTIAL)
SHA256 = X.sha256_file


# --------------------------------------------------------------------------- the 26
def affected(miss: dict[str, Any]) -> list[dict[str, Any]]:
    """The 26 rows of ``agg18_miss_table.json`` that are wrong-plan or malformed-SQL misses, in table order."""
    return [r for r in miss["rows"] if r["category"] in ("wrong_plan_derived_or_filtered", "sql_malformed_or_degenerate")]


def stratum(row: dict[str, Any]) -> str:
    """The pilot stratum of one affected question (PREDECLARED order); one stratum per question, first match."""
    q, fam = row["qid"], row["family"]
    mal = row["category"] == "sql_malformed_or_degenerate"
    if row["continuity"]:
        return "continuity-malformed" if mal else "continuity-plan"
    if row["where"] and fam == "dividends":
        return "dividends where"
    if fam in ("margin",) or row["op"] == "top3":
        return "margin/top3"
    if row["where"]:
        return "where-filter sum"
    if fam == "balance_sheet" and row["op"].startswith("count"):
        return "current-ratio count"
    return "other"


# --------------------------------------------------------------------------- reference plan (compiled from the golds)
_FLD = re.compile(r"^[a-z_]+$")


def _expr(e: str) -> str:
    parts = re.split(r"([/-])", e)
    return " ".join(parts) if len(parts) == 3 else e


def compile_reference_sql(g: dict[str, Any], corpus: dict[str, Any], *, period: str = "2022") -> str:
    """The ``agg110.v1`` statement a perfect plan writes for gold ``g`` (op / expr / where / threshold / exclusions)."""
    company = {d["doc_name"]: d["company"] for d in corpus["docs"]}
    conds = [f"period = '{period}'"]
    if g.get("where"):
        w = g["where"]
        conds.append(f"{_expr(w['expr'])} {w['cmp']} {w['t']}")
    for d in g.get("excluded_by_rule") or []:
        conds.append(f"entity != '{company[d]}'")
    ex = _expr(g["expr"])
    op = g["op"]
    head = f" FROM facts_pivot WHERE {conds[0]}"
    rest = "".join(f" AND {c}" for c in conds[1:])
    if op in ("argmax", "argmin"):
        return f"SELECT entity{head}{rest} ORDER BY {ex} {'DESC' if op == 'argmax' else 'ASC'} LIMIT 1"
    if op == "top3":
        return f"SELECT entity, {ex}{head}{rest} ORDER BY {ex} DESC LIMIT 3"
    if op in ("count_gt", "count_lt"):
        return f"SELECT COUNT(*){head}{rest} AND {ex} {'>' if op == 'count_gt' else '<'} {g['threshold']}"
    if op == "sum":
        return f"SELECT SUM({ex}){head}{rest}"
    raise ValueError(f"unsupported op {op!r}")


def rows_to_final(g: dict[str, Any], rows: list[tuple]) -> str:
    """A ``FINAL:`` answer line from the executed reference rows, for the frozen scorer (``phase0_arms.score_agg``)."""
    if not rows:
        return "FINAL: no rows"
    if g["op"] in ("argmax", "argmin"):
        return f"FINAL: {rows[0][0]}"
    if g["op"] == "top3":
        return "FINAL: " + ", ".join(str(r[0]) for r in rows[:3])
    v = rows[0][0]
    return f"FINAL: {int(v) if g['op'].startswith('count') else v}"


def pivot_names(store_path: Path, corpus_id: str) -> list[str]:
    import sqlite3  # noqa: PLC0415
    c = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        return sorted(r[0] for r in c.execute("SELECT DISTINCT entity FROM facts_canonical WHERE corpus_id = ?", [corpus_id]))
    finally:
        c.close()


def ceiling_table(miss: dict[str, Any], golds: dict[str, dict[str, Any]], corpus: dict[str, Any], store_path: Path,
                  corpus_id: str) -> dict[str, Any]:
    """Zero-model: execute each reference statement on an in-memory copy of the AGG-1.1 store, score it with the frozen
    scorer. The ceiling is what a correct plan achieves on THIS store (its bad cells included)."""
    import sqlite3  # noqa: PLC0415
    from src.aggregation import sql_grammar as G  # noqa: PLC0415
    src = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    mem = sqlite3.connect(":memory:")
    src.backup(mem)
    src.close()
    canon = [dict(zip(("field", "entity", "corpus_id"), r)) for r in mem.execute(
        "SELECT DISTINCT field, entity, corpus_id FROM facts_canonical WHERE corpus_id = ?", [corpus_id])]
    cat = G.pivot_catalog(canon, names=pivot_names(store_path, corpus_id), fields=tuple(A.METRICS))
    G.create_pivot_view(mem, corpus_id, cat)
    out = []
    for r in affected(miss):
        g = golds[r["qid"]]
        sql = compile_reference_sql(g, corpus)
        ok, why = G.validate_pivot_sql(sql, cat)
        guarded = G.null_guard(sql, cat) if ok else sql
        rows = mem.execute(guarded).fetchall() if ok else []
        text = rows_to_final(g, rows)
        out.append({"qid": r["qid"], "category": r["category"], "stratum": stratum(r), "reference_sql": sql,
                    "valid": ok, "reason": why, "executed_sql": guarded, "answer": text, "gold": g["gold"],
                    "ceiling": bool(ok and P.score_agg(g, text)), "statement_chars": len(sql),
                    "statement_tokens_est": round(len(sql) / CHARS_PER_TOKEN, 1)})
    longest = max(o["statement_chars"] for o in out)
    return {"schema": "agg110_ceiling.v1", "planner_version": V1, "miss_table_sha256": MISS_TABLE_SHA,
            "n": len(out), "n_ceiling_true": sum(o["ceiling"] for o in out), "n_valid": sum(o["valid"] for o in out),
            "longest_statement_chars": longest,
            "longest_statement_tokens_est": round(longest / CHARS_PER_TOKEN, 1),
            "sql_n_predict": SQL_N_PREDICT[V1],
            "longest_within_60pct_of_cap": longest / CHARS_PER_TOKEN <= LONGEST_STATEMENT_FRACTION * SQL_N_PREDICT[V1],
            "agg11_reference_m4_over_26": AGG11_REF, "rows": out}


def gold_cell_check(miss: dict[str, Any], golds: dict[str, dict[str, Any]], corpus: dict[str, Any], store_path: Path,
                    corpus_id: str, sheet_csv: Path) -> dict[str, Any]:
    """Compile sanity: the 26 reference statements on a copy of the store whose FY2022 cells for the 8 metrics are replaced by
    the verified golds (NA cells deleted) must all score correct; a failure is a compile or scorer bug, not a store defect."""
    import csv  # noqa: PLC0415
    import sqlite3  # noqa: PLC0415
    from src.aggregation import sql_grammar as G  # noqa: PLC0415
    src = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    mem = sqlite3.connect(":memory:")
    src.backup(mem)
    src.close()
    ents = {r[0]: r[1] for r in mem.execute("SELECT DISTINCT doc_id, entity FROM facts_canonical WHERE corpus_id = ?", [corpus_id])}
    for r in csv.DictReader(sheet_csv.open(newline="")):
        d, f = r["doc_name"], r["metric"]
        if f not in A.METRICS or r["status"] not in ("filled", "na"):
            continue
        mem.execute("DELETE FROM facts_canonical WHERE doc_id = ? AND field = ? AND period = '2022'", [d, f])
        if r["status"] == "filled" and r["value_usd_m"].strip():
            mem.execute("INSERT INTO facts_canonical (corpus_id, doc_id, field, period, entity, value, unit) VALUES (?,?,?,?,?,?,?)",
                        [corpus_id, d, f, "2022", ents[d], float(r["value_usd_m"].replace(",", "")), "USD_millions"])
    canon = [dict(field=r[0], entity=r[1]) for r in mem.execute("SELECT DISTINCT field, entity FROM facts_canonical")]
    cat = G.pivot_catalog(canon, names=sorted(set(ents.values())), fields=tuple(A.METRICS))
    G.create_pivot_view(mem, corpus_id, cat)
    wrong = []
    for r in affected(miss):
        g = golds[r["qid"]]
        text = rows_to_final(g, mem.execute(G.null_guard(compile_reference_sql(g, corpus), cat)).fetchall())
        if not P.score_agg(g, text):
            wrong.append({"qid": r["qid"], "answer": text, "gold": g["gold"]})
    return {"n": len(affected(miss)), "n_correct": len(affected(miss)) - len(wrong), "wrong": wrong}


def pilot_selection(ceiling: dict[str, Any]) -> dict[str, Any]:
    """The first 6 ceiling-true questions in the fixed stratum order; when a stratum has none, the next stratum's second
    ceiling-true question stands in (PREDECLARED), and strata after the last are used in order as a final fill."""
    by: dict[str, list[str]] = {}
    for r in ceiling["rows"]:
        if r["ceiling"]:
            by.setdefault(r["stratum"], []).append(r["qid"])
    picked: list[tuple[str, str]] = []
    spare: list[tuple[str, str]] = []
    for i, s in enumerate(PILOT_STRATA):
        qs = by.get(s, [])
        if qs:
            picked.append((s, qs[0]))
            spare += [(s, q) for q in qs[1:]]
        else:
            nxt = by.get(PILOT_STRATA[(i + 1) % len(PILOT_STRATA)], [])
            if len(nxt) >= 2:
                picked.append((PILOT_STRATA[(i + 1) % len(PILOT_STRATA)], nxt[1]))
    ids = [q for _s, q in picked]
    for _s, q in spare + [("other", q) for q in by.get("other", [])]:
        if len(ids) >= PILOT_N:
            break
        if q not in ids:
            ids.append(q)
    return {"schema": "agg110_pilot.v1", "rule": "first 6 ceiling-true questions in stratum order; PREDECLARED_AGG110.md",
            "strata": list(PILOT_STRATA), "picked": [{"stratum": s, "qid": q} for s, q in picked], "qids": ids[:PILOT_N]}


# --------------------------------------------------------------------------- the agg110.v1 SQL answer
def sql_prompt_v1(question: str, fdesc: str, names: list[str], fields: list[str]) -> str:
    return ("Write ONE SQLite SELECT over the view facts_pivot(entity, doc_id, period, " + ", ".join(fields) + "). It has one "
            "row per company and fiscal year; every field is a numeric column in USD millions (NULL when the filing has no "
            "such line); period is the 4-digit fiscal year as text, e.g. '2022'. Always start with WHERE period = 'YYYY'. "
            "A ratio or a difference of two fields is ONE expression, for example revenue / total_assets or "
            "total_assets - revenue; a percentage is a fraction (10% is 0.1). Filter companies by a condition on a field "
            "(for example net_income < 0), never by listing companies; leave a named company out with "
            "entity != 'Name'. For 'which company' select entity ordered by the expression, for 'top N' select entity and "
            "the expression ordered DESC with LIMIT N, for 'how many' use COUNT(*) with the condition, for a combined "
            "amount use SUM(expression).\n"
            f"Fields:\n{fdesc}\nCompany names (entity): {'; '.join(names)}\nQuestion: {question}\nSQL:")


def reask_prompt(prompt: str, bad_sql: str, why: str) -> str:
    return (prompt + " " + bad_sql[:300] + f"\nThat statement was rejected ({why}). Write ONE corrected statement that "
            "follows the rules above.\nSQL:")


def sql_answer_v1(backend, store, question: str, names, cfg: P.Cfg, *, corpus_id: str | None = None
                  ) -> tuple[str, str, dict[str, Any]]:
    """``phase0_arms.sql_answer`` for ``agg110.v1``: pivot view, bounded grammar, one re-ask on a rejected statement, null
    guard, ``n_predict`` 256. The answer call and its prompt are ``phase0_arms``' unchanged."""
    from src.aggregation import sql_grammar as G  # noqa: PLC0415
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    corpus_id = corpus_id or P.CORPUS_ID
    if not store.canonical_facts(corpus_id):
        store.refresh_canonical(corpus_id)
    canon = store.canonical_facts(corpus_id)
    ents = sorted({r["entity"] for r in canon if r.get("entity")})
    cat = G.pivot_catalog(canon, names=ents, fields=tuple(A.METRICS))
    G.create_pivot_view(store.conn, corpus_id, cat)
    gbnf = G.build_pivot_gbnf(cat)
    fdesc = "\n".join(f"- {f}: {A.METRICS[f]}" for f in cat.fields)
    prompt = sql_prompt_v1(question, fdesc, list(names), list(cat.fields))
    info: dict[str, Any] = {"planner_version": V1, "sql_table": cat.table, "n_canonical": len(canon),
                            "sql_n_predict": SQL_N_PREDICT[V1], "sql_reask": False}
    sql, why, ok = "", "", False
    try:
        for attempt in (0, 1):
            reply = backend.chat(prompt if attempt == 0 else reask_prompt(prompt, sql, why),
                                 n_predict=SQL_N_PREDICT[V1], grammar=gbnf, id_slot=0)
            sql = str(reply["text"]).strip()
            if reply.get("finish_reason") == "length":
                info["sql_truncated"] = True
                ok, why = False, "truncated at n_predict"
            else:
                ok, why = G.validate_pivot_sql(sql, cat)
            if attempt == 0:
                info.update(sql_first=sql, sql_first_reason=why)
            if ok:
                break
            info["sql_reask"] = True
    except OutOfWindow:
        return "", "out_of_window", info
    info.update(sql=sql, sql_valid=ok, sql_reason=why)
    rows: list[Any] = []
    if ok:
        guarded = G.null_guard(sql, cat)
        info["sql_executed"] = guarded
        try:
            cur = store.conn.execute(guarded)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchmany(cfg.sql_row_cap)]
        except Exception as e:  # noqa: BLE001 - recorded, answered from no rows
            info["sql_error"] = str(e)[:200]
    info["n_rows"] = len(rows)
    info["backfill_needed"] = ok and not rows
    shown = info.get("sql_executed", sql)
    ctx = f"SQL: {shown}\nRows ({len(rows)}):\n" + "\n".join(json.dumps(r, default=str) for r in rows)
    try:
        ans = backend.chat(P._ans_prompt(question, ctx, list(names)), n_predict=cfg.answer_n_predict, id_slot=0)
    except OutOfWindow:
        return "", "out_of_window", info
    info["finish_reason"] = ans.get("finish_reason")
    return ans["text"], "ok", info


def p_large_answer_v(backend, store, q: dict[str, Any], *, scope, planner, docs_meta, built, sql_present, names, cfg: P.Cfg,
                     corpus_chunks, corpus_bm25, doc_chunks: Callable[[str], list[dict[str, Any]]],
                     planner_version: str = V0, corpus_id: str | None = None) -> tuple[str, str, dict[str, Any]]:
    """``phase1_arms.p_large_answer`` with a planner version. ``agg11.v0`` calls it unchanged; ``agg110.v1`` takes the same
    planning step and replaces only the ``sql`` branch's SQL answer. Every record gets ``planner_version`` and ``sql_shape``."""
    if planner_version == V0:
        text, st, extra = X.p_large_answer(backend, store, q, scope=scope, planner=planner, docs_meta=docs_meta, built=built,
                                           sql_present=sql_present, names=names, cfg=cfg, corpus_chunks=corpus_chunks,
                                           corpus_bm25=corpus_bm25, doc_chunks=doc_chunks)
        extra["planner_version"] = V0
        return text, st, extra
    if planner_version != V1:
        raise ValueError(f"unknown planner version {planner_version!r}")
    scope_doc, scope_why, qsf = scope
    pf, _ = planner.scoped_facts(q["question"], docs_meta, questions_so_far=qsf, available_layers=tuple(built),
                                 sql_fields_present=sql_present)
    dec = planner.plan(q["question"], pf)
    extra: dict[str, Any] = {"path": dec.path, "decided_by": dec.decided_by, "receipt_stub": dec.receipt.get("stub", False),
                             "scope_doc": scope_doc, "scope_reason": scope_why, "planner_questions_so_far": qsf,
                             "planner_corpus_tokens": pf.corpus_tokens, "planner_version": V1,
                             "sql_shape": planner.sql_shape(q["question"])}
    if dec.path == "sql":
        text, st, info = sql_answer_v1(backend, store, q["question"], names, cfg, corpus_id=corpus_id)
        extra.update(info)
    elif scope_doc:
        text, st = P.single_doc_answer(backend, store, scope_doc, q["question"], names, cfg, dec.path,
                                       doc_chunks(scope_doc), extra)
    else:
        text, st = P.agent_answer(backend, corpus_chunks, q["question"], names, cfg, bm25=corpus_bm25)
    return text, st, extra


# --------------------------------------------------------------------------- pure evaluations
def question_order(ids_26: list[str], pilot: list[str]) -> list[str]:
    """Pilot questions first (frozen order), then the rest of the 26 in table order."""
    return list(pilot) + [q for q in ids_26 if q not in pilot]


def pilot_eval(answers: dict[str, dict[str, Any]], pilot: list[str]) -> dict[str, Any]:
    done = [answers[q] for q in pilot if q in answers]
    fixed = sum(1 for r in done if r.get("correct") is True)
    wall = sum(float(r.get("wall_s") or 0) for r in done)
    return {"n": len(done), "fixed": fixed, "gates_tripped": ["futility:pilot_fixes_0_of_6"] if len(done) == len(pilot) and fixed == 0 else [],
            "wall_s": wall, "budget_stops": ["pilot_wall_over_600s"] if wall > PILOT_WALL_S else []}


def m4(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    recs = list(records)
    if not recs:
        return {"n": 0, "median_calls": None, "median_prefill": None, "within_bounds": None}
    mc = statistics.median(r["calls"] for r in recs)
    mp = statistics.median(r["prefill_tokens"] for r in recs)
    return {"n": len(recs), "median_calls": mc, "median_prefill": mp,
            "within_bounds": mc <= M4_CALLS_MAX and mp <= M4_PREFILL_MAX and mc <= M4_KILL_CALLS,
            "bounds": {"median_calls_max": M4_CALLS_MAX, "median_prefill_max": M4_PREFILL_MAX, "kill_median_calls": M4_KILL_CALLS},
            "agg11_reference_same_26": AGG11_REF}


def verdict110(answers: dict[str, dict[str, Any]], ceiling: dict[str, Any], ids_26: list[str]) -> dict[str, Any]:
    """Before/after accuracy on the 26, split by class and by ceiling, M4 over every question that ran, planner versions."""
    cmap = {r["qid"]: r for r in ceiling["rows"]}
    ran = [answers[q] for q in ids_26 if q in answers]
    n_correct = sum(1 for r in ran if r.get("correct") is True)

    def split(key: Callable[[str], str]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in ran:
            b = out.setdefault(key(r["qid"]), {"n": 0, "correct": 0})
            b["n"] += 1
            b["correct"] += int(r.get("correct") is True)
        return dict(sorted(out.items()))

    complete = len(ran) == len(ids_26)
    mm = m4(ran)
    trips = ["M4"] if complete and mm["within_bounds"] is False else []
    return {"n_questions": len(ids_26), "n_scored": len(ran), "complete": complete,
            "before": {"correct": 0, "n": len(ids_26), "note": "all 26 were wrong in AGG-1.1 by construction"},
            "after": {"correct": n_correct, "n": len(ran), "accuracy": n_correct / len(ran) if ran else None},
            "by_class": split(lambda q: cmap[q]["category"]),
            "by_ceiling": split(lambda q: "ceiling_true" if cmap[q]["ceiling"] else "store_limited"),
            "ceiling": {"n_true": ceiling["n_ceiling_true"], "n": ceiling["n"]},
            "correct_qids": sorted(r["qid"] for r in ran if r.get("correct") is True),
            "m4": mm, "planner_versions": sorted({r.get("planner_version") for r in ran}),
            "n_reask": sum(1 for r in ran if r.get("sql_reask")),
            "n_sql_invalid": sum(1 for r in ran if r.get("sql_valid") is False),
            "n_answer_truncated": sum(1 for r in ran if r.get("finish_reason") == "length"),
            "gates_tripped": trips,
            "verdict": ("phase1_gate_tripped" if trips else "agg_1_10_done") if complete else "incomplete"}


def routing_report(questions: list[dict[str, Any]], planner, saved: dict[str, dict[str, Any]], ids_26: list[str]
                   ) -> dict[str, Any]:
    """Zero model: ``planner.sql_shape`` over all 58 questions, with the AGG-1.1 correctness; the exposure is how many
    currently-correct questions would also take the pivot plan."""
    rows = []
    for q in questions:
        sh = planner.sql_shape(q["question"])
        rows.append({"qid": q["id"], "shape": sh["shape"], "in_26": q["id"] in ids_26,
                     "agg11_correct": (saved.get(q["id"]) or {}).get("correct") is True})
    return {"n": len(rows), "pivot": sum(r["shape"] == "pivot" for r in rows),
            "pivot_in_26": sum(r["shape"] == "pivot" and r["in_26"] for r in rows),
            "single_in_26": [r["qid"] for r in rows if r["shape"] == "single" and r["in_26"]],
            "exposure_currently_correct_routed_pivot": [r["qid"] for r in rows if r["shape"] == "pivot" and r["agg11_correct"]],
            "rows": rows}


# --------------------------------------------------------------------------- frozen files and preflight
FROZEN = {"agg18_miss_table.json": MISS_TABLE_SHA}
# computed by ``agg110_ceiling.py`` (zero model) BEFORE any model call and pinned here, in the job's precond.sh and in the row
FROZEN_ALL = {**FROZEN,
              "agg110_ceiling.json": "716e3a2d02c9ab47a830ba9dfeeb5e73846649badfbcdf0e23e3daede359ce8b",
              "agg110_pilot.json": "b75a57681a3f1db5299c14bc764591aac815345899f8aacf5c0d0536b84bb7e6"}


def pin_problems(set_dir: Path = HERE, *, predeclared_sha: str = PREDECLARED_SHA, frozen: dict[str, str] | None = None
                 ) -> list[str]:
    problems = []
    pd = set_dir / PREDECLARED_FILE
    got = SHA256(pd) if pd.exists() else None
    if got != predeclared_sha:
        problems.append(f"{PREDECLARED_FILE} sha256 {got} != recorded {predeclared_sha}")
    for name, want in (FROZEN if frozen is None else frozen).items():
        p = set_dir / name
        got = SHA256(p) if p.exists() else None
        if got != want:
            problems.append(f"frozen input {name} sha256 {got} != pinned {want}")
    return problems


def phase0_pin_problems(set_dir: Path = HERE) -> list[str]:
    """The Phase 0 files ``agg11.v0`` runs through must keep their pinned sha256 (``X.PHASE0_PINS``)."""
    out = []
    for name, want in X.PHASE0_PINS.items():
        p = (set_dir.parent / "agg_phase0" / name)
        got = SHA256(p) if p.exists() else None
        if got != want:
            out.append(f"agg_phase0/{name} sha256 {got} != pinned {want}")
    return out


def load_frozen(set_dir: Path = HERE) -> dict[str, Any]:
    return {"miss": json.loads((set_dir / "agg18_miss_table.json").read_text()),
            "ceiling": json.loads((set_dir / "agg110_ceiling.json").read_text()),
            "pilot": json.loads((set_dir / "agg110_pilot.json").read_text())}


def frozen_problems(fz: dict[str, Any], golds: dict[str, dict[str, Any]]) -> list[str]:
    out = []
    ids = [r["qid"] for r in affected(fz["miss"])]
    if len(ids) != 26:
        out.append(f"{len(ids)} affected questions in the miss table, PREDECLARED has 26")
    if sorted(r["qid"] for r in fz["ceiling"]["rows"]) != sorted(ids):
        out.append("agg110_ceiling.json does not cover exactly the 26 affected questions")
    if not fz["ceiling"].get("longest_within_60pct_of_cap"):
        out.append(f"longest reference statement {fz['ceiling'].get('longest_statement_tokens_est')} tokens exceeds 60% of the "
                   f"{SQL_N_PREDICT[V1]}-token cap")
    if any(not r["valid"] for r in fz["ceiling"]["rows"]):
        out.append("a reference statement is rejected by the pivot grammar validator")
    pq = fz["pilot"]["qids"]
    cm = {r["qid"]: r["ceiling"] for r in fz["ceiling"]["rows"]}
    if len(pq) != PILOT_N or any(q not in cm for q in pq):
        out.append(f"pilot is not {PILOT_N} of the 26: {pq}")
    elif not all(cm[q] for q in pq):
        out.append("a pilot question is not ceiling-true")
    for q in ids:
        if q not in golds:
            out.append(f"{q}: no gold")
    return out
