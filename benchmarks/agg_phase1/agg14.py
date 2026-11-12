#!/usr/bin/env python3
"""AGG-1.4: one lazy backfill field (library; ``run_agg14.py`` is the CLI).

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an injected
backend (``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in tests).

Implements ``benchmarks/agg_phase1/PREDECLARED_AGG14.md`` literally. The L4 unit is ``phase1_arms.l4_unit_fy`` (amendment 2,
2026-11-12: the AGG-1.1 ``l4_unit`` with the field description rendered for the document's own fiscal year, FY2022 here, so
byte-identical): grammar-constrained GBNF rows with the verbatim-quote check and the ``n_predict`` cap, unchanged. The only new input is the field description, supplied to that function
through ``field_description`` (it reads ``phase0 METRICS``; the context manager adds and removes the key, no file edited).

Phase ``backfill`` (``run_agg14.py``): copy the AGG-1.1 store; seed the 224 AGG-1.1 L4 units as cache hits; snapshot the
earlier L4 facts; build the planner receipt (``sql+backfill`` trigger, no model); pilot = ONE document, then the other
missing documents; G1 calls <= 100, G2 field recall M1 >= 0.75, G3 earlier units untouched, G4 receipt;
``result.cpu.json``.

Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped, 4 precondition, 5 budget stop, 75 partial.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

HERE = Path(__file__).resolve().parent
P0DIR = HERE.parent / "agg_phase0"
for _p in (str(P0DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg_phase0 as A  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
from agg1314_zero_cell_diag import fy_contract_problems, scale_signature  # noqa: E402

# --------------------------------------------------------------------------- pins
AMENDMENT1_SHA = "8a632c67748acadbe7a525fd1ff5a5f0b094da585e97e37915f3987b8ab46b49"
AMENDMENT2_FILE = "PREDECLARED_AGG14_AMENDMENT2_20261112.md"
AMENDMENT2_SHA = "fd506c07e489c56d072d1ffbbaf6dccd8c80791a257c23a36399541a408abc6b"   # 2026-11-12: L4d unit-label cross-check; fiscal year taken from the manifest
PREDECLARED_SHA = "e27b4dcb212dd2692f69e9247a4ae923a1d43a8741eb724ae287df4bd63c038b"
FROZEN = {
    "agg14_field_pick.json": "007e39070665863792713cf2ee63f2b0d486f30afe1fbbc99c785879aa3806c9",
    "agg14_cells.csv": "1c699a6a2f5e6cbb46ac4ee1d6acdf8d428974244f10e49008a5260a7be879b7",
    "agg14_xcheck.json": "940dbbb399e2720f10e21049d47c6f6d89ca96f4ce028f0db4dd429a54799369",
}
FIELD = "long_term_debt"
FIELD_DESC = ("Long-term debt (non-current portion on the balance sheet, excluding current maturities), "
              "FY2022 year end (USD millions, positive)")
QUESTION = ("Which company in the corpus reported the highest FY2022 long-term debt (non-current portion), "
            "in USD millions?")
PERIOD = "2022"
EXPECTED_MISSING = ("ACTIVISIONBLIZZARD_2022_10K", "ADOBE_2022_10K", "AES_2022_10K", "AMD_2022_10K", "APPLE_2022_10K",
                    "BOEING_2022_10K", "COCACOLA_2022_10K", "COSTCO_2022_10K", "JOHNSON_JOHNSON_2022_10K",
                    "LOCKHEEDMARTIN_2022_10K", "MGMRESORTS_2022_10K", "PEPSICO_2022_10K")
PILOT_DOC = "COSTCO_2022_10K"          # the smallest of the 12 by extracted size (239,294 characters)
OLD_FIELDS = tuple(A.METRICS)          # the 8 fields with L4 units; captured at import, before any injection
CALL_GATE = 100
M1_GATE = X.M1_GATE
PILOT_WALL_S = 600
BOOKED_S = 1000.0
BUDGET_X = 2.0
EXPECTED_NA = ("ADOBE_2022_10K", "AES_2022_10K", "APPLE_2022_10K")
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_PARTIAL = (X.RC_OK, X.RC_PREFLIGHT, X.RC_GATE, X.RC_PRECOND,
                                                                   X.RC_BUDGET, X.RC_PARTIAL)
DEFAULT_AGG11_JOB = Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111")
SHA256 = X.sha256_file


# --------------------------------------------------------------------------- inputs and pins
def load_inputs(set_dir: Path = HERE, *, corpus: Path | None = None, cells: Path | None = None) -> dict[str, Any]:
    return {"corpus": json.loads((corpus or set_dir / "corpus.json").read_text()),
            "cells": list(csv.DictReader((cells or set_dir / "agg14_cells.csv").open(newline="")))}


def gold_cells(inp: dict[str, Any]) -> list[tuple[str, str, float]]:
    """(doc, field, value) for every FILLED cell (NA excluded, counted separately)."""
    return [(r["doc_name"], FIELD, float(r["value_usd_m"].replace(",", ""))) for r in inp["cells"]
            if r["status"] == "filled" and r["value_usd_m"].strip()]


def na_docs(inp: dict[str, Any]) -> list[str]:
    return sorted(r["doc_name"] for r in inp["cells"] if r["status"] != "filled")


def pin_problems(set_dir: Path = HERE, *, predeclared_sha: str = PREDECLARED_SHA,
                 frozen: dict[str, str] | None = None) -> list[str]:
    problems = []
    pd = set_dir / "PREDECLARED_AGG14.md"
    got = SHA256(pd) if pd.exists() else None
    if got != predeclared_sha:
        problems.append(f"PREDECLARED_AGG14.md sha256 {got} != recorded {predeclared_sha}")
    am = set_dir / "PREDECLARED_AGG14_AMENDMENT1_20261111.md"
    got = SHA256(am) if am.exists() else None
    if got != AMENDMENT1_SHA:
        problems.append(f"PREDECLARED_AGG14_AMENDMENT1_20261111.md sha256 {got} != recorded {AMENDMENT1_SHA}")
    am2 = set_dir / AMENDMENT2_FILE
    got = SHA256(am2) if am2.exists() else None
    if got != AMENDMENT2_SHA:
        problems.append(f"{AMENDMENT2_FILE} sha256 {got} != recorded {AMENDMENT2_SHA}")
    for name, want in (FROZEN if frozen is None else frozen).items():
        p = set_dir / name
        got = SHA256(p) if p.exists() else None
        if got != want:
            problems.append(f"frozen input {name} sha256 {got} != pinned {want}")
    return problems


def l4_reuse_check14(agg11_job: Path, orch_root: Path, research_arms: Path | None = None) -> dict[str, Any]:
    """The AGG-1.1 reuse rule against the AGG-1.1 job's pin, with the one amended L4 stage file (``facts_llm.py``,
    amendment 2, 2026-11-12) expected at ``X.AMENDED_L4`` for the 12 backfill units; the 224 earlier units stay the
    AGG-1.1 cache hits."""
    res = X.l4_reuse_check(agg11_job, orch_root, research_arms, amended=X.AMENDED_L4)
    res["against"] = ("AGG-1.1 job pin (" + str(agg11_job) + "), L4 stage as amended 2026-11-12 "
                      "(PREDECLARED_AGG14_AMENDMENT2_20261112.md) for the backfill units")
    return res


def doc_fy(inp: dict[str, Any], doc: str) -> str:
    """The document's own fiscal year from the corpus manifest (the AGG-1.1 manifest: 2022)."""
    entry = next((d for d in inp["corpus"]["docs"] if d["doc_name"] == doc), {})
    return X.doc_fiscal_year(entry, inp["corpus"])


def fy_problems(inp: dict[str, Any]) -> list[str]:
    """Amendment 2 hard preflight check: the field description rendered for each document's own fiscal year (corpus
    manifest) must not name a fiscal year other than the one its gold cell is for."""
    out = []
    for r in inp["cells"]:
        fy_doc = doc_fy(inp, r["doc_name"])
        descs = {FIELD: X.l4_description(FIELD_DESC, fy_doc)}
        out += [f"{r['doc_name']}: {p}" for p in fy_contract_problems(descs, [{"field": r["metric"], "fy": r["fiscal_year"]}])]
    return out


def cells_problems(inp: dict[str, Any], *, expect: bool = True) -> list[str]:
    problems = []
    docs = [d["doc_name"] for d in inp["corpus"]["docs"]]
    rows = inp["cells"]
    if sorted(r["doc_name"] for r in rows) != sorted(docs):
        problems.append("agg14 cells are not exactly one per corpus document")
    for r in rows:
        if r["metric"] != FIELD or r["verified"] != "true" or str(r["fiscal_year"]) != "2022":
            problems.append(f"{r['doc_name']}: cell is not a verified FY2022 {FIELD} cell")
    if expect and (len(gold_cells(inp)), na_docs(inp)) != (25, sorted(EXPECTED_NA)):
        problems.append(f"{len(gold_cells(inp))} filled cells / NA {na_docs(inp)}, PREDECLARED expects 25 / {EXPECTED_NA}")
    return problems


@contextlib.contextmanager
def field_description(field: str = FIELD, desc: str = FIELD_DESC) -> Iterator[None]:
    """Make ``phase1_arms.l4_unit`` (which reads ``A.METRICS[field]``) know the new field for the duration of a
    unit, then restore the table exactly: no frozen file is edited and the 8-metric table is never left changed."""
    had = field in A.METRICS
    old = A.METRICS.get(field)
    A.METRICS[field] = desc
    try:
        yield
    finally:
        if had:
            A.METRICS[field] = old
        else:
            A.METRICS.pop(field, None)


# --------------------------------------------------------------------------- snapshot, trigger
def l4_snapshot(conn, fields: tuple[str, ...] = OLD_FIELDS) -> dict[str, Any]:
    """sha256 over the sorted rows of every ``l4_llm.v1`` fact of the given fields, plus table counts that a
    backfill must not move (documents, chunks, L3 facts)."""
    marks = ",".join("?" for _ in fields)
    rows = conn.execute(
        "SELECT fact_id, doc_hash, field, value, unit, quote, audit_status, period, page FROM facts "
        f"WHERE extractor = 'l4_llm.v1' AND field IN ({marks}) ORDER BY fact_id, doc_hash, field", list(fields)).fetchall()
    h = hashlib.sha256(json.dumps([list(r) for r in rows], sort_keys=True, default=str).encode()).hexdigest()
    cnt = {"documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
           "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
           "l3_facts": conn.execute("SELECT COUNT(*) FROM facts WHERE extractor LIKE 'l3%'").fetchone()[0],
           "l4_old_fields": len(rows)}
    return {"sha256": h, "n_l4_facts": len(rows), "counts": cnt}


def canonical_in_memory(store, corpus_id: str) -> list[dict[str, Any]]:
    """``select_canonical`` over the live facts, nothing written (the trigger must not touch the store)."""
    from src.aggregation.store import FACT_COLUMNS, select_canonical  # noqa: PLC0415
    rows = [dict(zip(("corpus_id",) + FACT_COLUMNS, r)) for r in store.conn.execute(
        f"SELECT d.corpus_id, {', '.join('f.' + c for c in FACT_COLUMNS)} FROM facts f JOIN documents d "
        "ON f.doc_hash = d.doc_hash WHERE d.corpus_id = ?", [corpus_id]).fetchall()]
    return select_canonical(rows)[0]


def build_receipt(store, corpus_id: str, question: str = QUESTION) -> tuple[Any, dict[str, Any]]:
    """The planner decision for ``question`` over the store: ``(PlanDecision, backfill_plan)``. No model, no write."""
    from src.aggregation import planner  # noqa: PLC0415
    docs = store.corpus_docs(corpus_id)
    canon = canonical_in_memory(store, corpus_id)
    bp = planner.backfill_plan(question, canon, docs, period=planner.question_period(question))
    facts, why = planner.scoped_facts(question, docs, available_layers=tuple(sorted(store.built_layers(corpus_id))),
                                      sql_fields_present=bool(canon))
    dec = planner.plan(question, facts, backfill=bp)
    dec.receipt["scope_reason"] = why
    return dec, bp


def g4_eval(dec: Any, bp: dict[str, Any], expected: tuple[str, ...] = EXPECTED_MISSING) -> dict[str, Any]:
    r = dec.receipt
    bf = r.get("backfill") or {}
    checks = {"path_sql": dec.path == "sql", "no_model_call": dec.llm_called is False,
              "trigger": bf.get("trigger") is True, "fields": bf.get("fields") == [FIELD],
              "missing_docs": sorted(bf.get("missing_docs") or []) == sorted(expected),
              "path_label": r.get("path_label") == "sql+backfill", "schema": r.get("schema") == "decision_receipt.v1"}
    return {"ok": all(checks.values()), "checks": checks, "path": dec.path, "path_label": r.get("path_label"),
            "decided_by": dec.decided_by, "llm_called": dec.llm_called, "backfill": bf}


# --------------------------------------------------------------------------- gates (pure)
def g1_eval(fresh: dict[str, Any], n_units_done: int, n_units: int) -> dict[str, Any]:
    return {"ok": fresh["llm_calls"] <= CALL_GATE, "calls": fresh["llm_calls"], "gate": CALL_GATE,
            "units_done": n_units_done, "units": n_units, "calls_per_unit": (fresh["llm_calls"] / n_units_done)
            if n_units_done else None, "prefill_tokens": fresh["prefill_tokens"], "wall_s": fresh["wall_s"]}


def pilot_eval(unit: dict[str, Any], n_units: int) -> dict[str, Any]:
    """Pilot futility (G1 projection) and budget stops, pure arithmetic over the pilot unit."""
    proj_calls = unit["llm_calls"] * n_units
    proj_wall = unit["wall_s"] * n_units
    gates = ["G1"] if proj_calls > CALL_GATE else []
    budget = []
    if unit["wall_s"] > PILOT_WALL_S:
        budget.append("pilot_wall_over_600s")
    if proj_wall > BUDGET_X * BOOKED_S:
        budget.append("projected_wall_over_2x_booked")
    return {"calls": unit["llm_calls"], "prefill_tokens": unit["prefill_tokens"], "wall_s": unit["wall_s"],
            "projected_calls": proj_calls, "projected_wall_s": proj_wall, "gates_tripped": gates, "budget_stops": budget}


def m1_field(facts: list[dict[str, Any]], cells: list[tuple[str, str, float]], backfilled: set[str],
             *, scale: dict[str, float] | None = None) -> dict[str, Any]:
    hits = X.m1_hits(facts, cells, unit_scale=scale or X.M1R_UNIT_SCALE)
    bf = [(c, h) for c, h in zip(cells, hits) if c[0] in backfilled]
    per = {c[0]: bool(h) for c, h in zip(cells, hits)}
    return {"gold": len(cells), "found": sum(hits), "m1": (sum(hits) / len(cells)) if cells else None,
            "backfilled_docs": {"gold": len(bf), "found": sum(h for _c, h in bf),
                                "m1": (sum(h for _c, h in bf) / len(bf)) if bf else None},
            "per_doc": dict(sorted(per.items()))}


# --------------------------------------------------------------------------- CPU phase
def seed_units(rec: P.Recorder, agg11_results: Path) -> dict[str, Any]:
    recs = X.read_jsonl(agg11_results)
    ok = X.l4_ok(recs)
    n = 0
    for unit, r in sorted(ok.items()):
        if unit in rec.done:
            continue
        rec.emit("l4_unit", **{k: v for k, v in r.items() if k not in ("kind", "ts", "reused", "reused_from")}, reused=True,
                 reused_from=str(agg11_results))
        n += 1
    seeded = [r for r in rec.records if r["kind"] == "l4_unit" and r.get("reused")]
    return {"seeded": n, "units": len(seeded), **X.tally(seeded)}


def _result(out: Path, **res: Any) -> dict[str, Any]:
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = PREDECLARED_SHA
    X._atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    return res


def _gate_stop(rec: P.Recorder, out: Path, gates: list[str], where: str, **detail: Any) -> dict[str, Any]:
    rec.emit("gate_tripped", unit=f"gate_tripped:{where}", gates=gates, where=where, **detail)
    rec.emit("stop", reason="phase1_gate_tripped", gates=gates, where=where, final=False)
    res = _result(out, complete=True, futility_stop="+".join(gates), gate_tripped=gates,
                  verdict="phase1_gate_tripped", where=where, m1=detail.get("m1"), rc=RC_GATE)
    res["rc"] = RC_GATE
    return res


def _fresh(rec: P.Recorder) -> dict[str, Any]:
    return X.tally(r for r in X.l4_ok(rec.records).values() if not r.get("reused"))


def run_backfill(rec: P.Recorder, out: Path, inp: dict[str, Any], orch_root: Path, backend, *, base_store: Path,
                 agg11_results: Path, guard: Callable[[], None], smoke: bool = False, corpus_id: str | None = None,
                 reuse_ok: bool = True, expected_missing: tuple[str, ...] = EXPECTED_MISSING,
                 pilot_doc: str = PILOT_DOC, cfg: P.Cfg = P.CFG) -> dict[str, Any]:
    corpus_id = corpus_id or X.CORPUS_ID
    if not reuse_ok:
        rec.emit("precondition_failed", reason="L4 reuse test failed: the earlier units are not cache hits")
        return {"complete": False, "rc": RC_PRECOND}
    store_path = out / "store.sqlite"
    if not store_path.exists():
        if not base_store.exists():
            rec.emit("precondition_failed", reason=f"no base store {base_store}")
            return {"complete": False, "rc": RC_PRECOND}
        P.copy_store(base_store, store_path)
        rec.emit("store_copied", src=str(base_store), sha256=SHA256(store_path), final=False)
    store = P.open_store(store_path)
    try:
        if "reuse_seeded" not in rec.done:
            rec.emit("reuse_seeded", unit="reuse_seeded", **seed_units(rec, agg11_results))
        if "before_snapshot" not in rec.done:
            rec.emit("before_snapshot", unit="before_snapshot", **l4_snapshot(store.conn))
        before = next(r for r in rec.records if r.get("unit") == "before_snapshot")
        # ---- the planner receipt (deterministic, no model, no store write)
        if "receipt" not in rec.done:
            dec, bp = build_receipt(store, corpus_id)
            g4 = g4_eval(dec, bp, expected_missing)
            rec.emit("receipt", unit="receipt", receipt=dec.receipt, backfill_plan=bp, g4=g4)
        rcp = next(r for r in rec.records if r.get("unit") == "receipt")
        if not rcp["g4"]["ok"]:
            return _gate_stop(rec, out, ["G4"], "receipt", g4=rcp["g4"])
        missing = list(rcp["backfill_plan"]["per_field"][FIELD]["missing_docs"])
        order = [pilot_doc] + sorted(d for d in missing if d != pilot_doc)
        if pilot_doc not in missing:
            rec.emit("precondition_failed", reason=f"pilot document {pilot_doc} is not in the backfill set")
            return {"complete": False, "rc": RC_PRECOND}
        units = P.windows(store, corpus_id, cfg.window_chars)
        for i, doc in enumerate(order):
            unit = f"l4:{FIELD}:{doc}"
            if unit in rec.done:
                continue
            with field_description():
                X.l4_unit_fy(rec, store, backend, units, FIELD, doc, guard, cfg, fy=doc_fy(inp, doc))
            if unit not in rec.done:
                return {"complete": False, "reason": f"{unit} errored; re-grant retries it", "rc": RC_PARTIAL}
            if smoke:
                return {"complete": False, "smoke": True, "rc": RC_OK}
            if i == 0 and "pilot_done" not in rec.done:
                u = X.l4_ok(rec.records)[unit]
                ev = pilot_eval(u, len(order))
                rec.emit("pilot_done", unit="pilot_done", doc=doc, **ev)
                if ev["gates_tripped"]:
                    return _gate_stop(rec, out, ev["gates_tripped"], "pilot", pilot=ev)
                if ev["budget_stops"]:
                    rec.emit("stop", reason="budget_stop:" + "+".join(ev["budget_stops"]), final=False, pilot=ev)
                    return {"complete": False, "budget_stop": "+".join(ev["budget_stops"]), "rc": RC_BUDGET, **ev}
            fresh = _fresh(rec)
            if fresh["llm_calls"] > CALL_GATE:
                return _gate_stop(rec, out, ["G1"], "running", fresh=fresh)
        return _finish(rec, out, store, inp, corpus_id, before, rcp, order)
    finally:
        store.close()


def _finish(rec, out, store, inp, corpus_id, before, rcp, order) -> dict[str, Any]:
    ok = X.l4_ok(rec.records)
    fresh_units = [r for r in ok.values() if not r.get("reused")]
    fresh = X.tally(fresh_units)
    g1 = g1_eval(fresh, len(fresh_units), len(order))
    canon = store.refresh_canonical(corpus_id)
    cells = gold_cells(inp)
    facts = store.facts(corpus_id)
    m1 = m1_field(facts, cells, set(order))
    l3_only = X.m1_hits([f for f in facts if str(f.get("extractor") or "").startswith("l3")], cells,
                        unit_scale=X.M1R_UNIT_SCALE)
    m1["l3_only"] = {"found": sum(l3_only), "m1": sum(l3_only) / len(cells) if cells else None}
    m1["lift"] = m1["found"] - sum(l3_only)
    g2 = {"ok": (m1["m1"] or 0) >= M1_GATE, "m1": m1["m1"], "gate": M1_GATE, "na_cells": na_docs(inp), **m1}
    after = l4_snapshot(store.conn)
    old_units = [u for u in ok if u.split(":", 2)[1] in OLD_FIELDS]
    fresh_old = [u for u in old_units if not ok[u].get("reused")]
    g3 = {"ok": after["sha256"] == before["sha256"] and after["counts"] == before["counts"] and not fresh_old
          and len(old_units) == len(OLD_FIELDS) * len(inp["corpus"]["docs"]),
          "l4_snapshot_equal": after["sha256"] == before["sha256"], "counts_before": before["counts"],
          "counts_after": after["counts"], "previously_built_units": len(old_units), "previously_built_rerun": len(fresh_old),
          "fresh_calls_on_old_fields": sum(ok[u]["llm_calls"] for u in fresh_old)}
    g4 = rcp["g4"]
    # informational: the canonical table now answers the question deterministically (no model)
    gold_top = max(cells, key=lambda c: c[2]) if cells else None
    row = store.conn.execute(
        "SELECT doc_id, value FROM facts_canonical WHERE corpus_id = ? AND field = ? AND period = ? "
        "ORDER BY value DESC LIMIT 1", [corpus_id, FIELD, PERIOD]).fetchone()
    sig = scale_signature(store.conn, FIELD)   # informational: facts stored |value| < 1 USD_millions (the r2 mis-scale)
    ovr = [o for r in fresh_units for o in (r.get("unit_overrides") or [])]
    det = {"informational": True, "canonical": canon, "gold_argmax": gold_top[0] if gold_top else None,
           "table_argmax": row[0] if row else None, "table_argmax_value": row[1] if row else None,
           "table_rows_for_field": store.conn.execute(
               "SELECT COUNT(*) FROM facts_canonical WHERE corpus_id = ? AND field = ? AND period = ?",
               [corpus_id, FIELD, PERIOD]).fetchone()[0]}
    rec.emit("gates_cpu", unit="gates_cpu", g1=g1, g2=g2, g3=g3, g4=g4, deterministic_check=det, scale_signature=sig)
    tripped = [g for g, v in (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)) if not v["ok"]]
    return _result(out, complete=True, futility_stop="+".join(tripped) if tripped else None, gate_tripped=tripped,
                   verdict="phase1_gate_tripped" if tripped else "agg_1_4_done", m1=g2["m1"], g1=g1, g2=g2, g3=g3, g4=g4,
                   fresh=fresh, deterministic_check=det, scale_signature=sig,
                   n_unit_overrides=sum(int(r.get("n_unit_overrides") or 0) for r in fresh_units), unit_overrides=ovr,
                   order=order, rc=RC_GATE if tripped else RC_OK)
