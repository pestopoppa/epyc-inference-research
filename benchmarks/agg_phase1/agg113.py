#!/usr/bin/env python3
"""AGG-1.13 step B: out-of-sample validation of ``agg_pipeline.v2`` on 10 new FinanceBench filings (library; ``run_agg113.py`` is the CLI).

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an injected backend
(``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in tests).

Implements ``agg113/PREDECLARED_AGG113.md`` literally and changes none of: the questions, the golds, the cells, the frozen scorer
(``phase0_arms.score_agg``), the M1 reading (``phase1_arms.m1_hits``), the metrics, the arms, the gates. Everything pipeline-specific is
the orchestrator's (``src.aggregation.pipeline``: flag ``ORCHESTRATOR_AGG_PIPELINE``); this module imports the AGG-1.12 library
(``agg112``) for the typed L4 window machinery and the Phase 0 / AGG-1.1 helpers for the v1 stage, and edits none of them.

Phases (``run_agg113.py --phase``): ``preflight`` (0, zero compute), ``l3`` (1, shared L0-L3 seed, NO model), ``ingest`` (2-4, one arm:
``--pipeline v1|v2``; the 3 pilot documents first, then the other 7), ``gpu-audit`` / ``gpu-answers`` (5: L4c audit of both stores,
then P-large for arms A, B, C), ``score`` (verdict only, no model).

HONESTY (PREDECLARED rule 1): this module prints and records COUNTS ONLY for pilot and interim looks; the per-cell hit vectors needed for
the paired reading live in machine files (``cells.json``, ``checkpoint.*.json``) that nobody reads before ``score`` has run.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
P0DIR = HERE.parent / "agg_phase0"
D113 = HERE / "agg113"
for _p in (str(P0DIR), str(HERE), str(D113)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg110 as G110  # noqa: E402
import agg112 as G  # noqa: E402
import agg16 as G16  # noqa: E402
import agg19 as G19  # noqa: E402
import agg1314_zero_cell_diag as D  # noqa: E402
import agg_phase0 as A  # noqa: E402
import l3_load as L3  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
import design113 as D113_DESIGN  # noqa: E402

# --------------------------------------------------------------------------- pins
PREDECLARED_FILE = "PREDECLARED_AGG113.md"
PREDECLARED_SHA = "a8b715254051062d6bfe184066916a01ae4c9f7a7b7a7db67203e26a1fff7ecf"
AMENDMENT_FILE = "PREDECLARED_AGG113_AMENDMENT1_20261114.md"    # the fiscal-year-end rule (operator decision 2026-11-14), written before any data
AMENDMENT_SHA = "279645ad90b74f479a3decd397c8ffc8b26750491332a478985750d9aae2daaa"
FY_END_MONTHS = (1, 2, 3)                                        # the fy_end rule's months (orchestrator facts_det.FY_END_MONTHS)
FROZEN = {   # sha256 of the step A inputs, exactly the table in PREDECLARED_AGG113.md (set dir = benchmarks/agg_phase1/agg113)
    "financebench_document_information.jsonl": "1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575",
    "selection.json": "52e11d3b93b321cb1feac4dbbaa429bdfda0328dcb6ef1c1cbe85c02dd80191d",
    "corpus.json": "f45ee6a4080f0eddf8f410184295cad8154ca78b84ae09be778427cbd42846ec",
    "facts_sheet.csv": "fd37e254ff6d6d2766b3681884348d9feabe7fd0144f4aca9b58a1ba72021328",
    "facts_sheet_xcheck.json": "321c95bd2e33b95354f3b5df6f286dbaab0b1cf8703b89815aa2732ce7a3d5d7",
    "aggregation_questions.json": "c73986a4ed89333484c5156a3b10a4adddac2f98e67c5ffe43b184a2813d4d18",
    "golds.jsonl": "c5c2325c9d915a58f13051b63d398ffb76ca7ff02ea333c33d10740af5203ca3",
    "verification_record.json": "e55ae142c8dc05b95162a6de6112e7f72e6d3deb9575e8535d51df8b9005bf79",
}
VERIFY_CELLS_SHA = "ce74aa5ab65d460009768274fa3d2e4d569ec511ee4700dee693c5b5056fad78"    # ../verify_phase1_cells.py, reused unchanged
MANIFEST_DIGEST = "94c8497392bd304685b1b48aac553af0b18bbb97ea29a41286b3257a5ca5dc97"
ORCH_PIN_MIN = "8ac97157557ca45a4b86c8c9d2874c4b13977b6b"      # 737bf0535 (PREDECLARED *Arms*) plus the opt-in fy_end rule (amendment 1) or a descendant
SHA256 = X.sha256_file
CORPUS_ID = X.CORPUS_ID                                         # one corpus id per store file; the stores are separate files
V1, V2 = "v1", "v2"
PIPELINE_OF = {V1: "agg_pipeline.v1", V2: "agg_pipeline.v2"}

# --------------------------------------------------------------------------- the design (PREDECLARED_AGG113.md)
N_DOCS, N_CELLS_FILLED, N_QUESTIONS = 10, 76, 30
PILOT_DOCS = tuple(D113_DESIGN.PILOT_DOCS)   # the first 3 of the salted order (corpus.json pilot_docs; names live in agg113/design113.py)
PILOT_CELLS = 22
PILOT_WALL_S = 1800
INTERIM_DOCS = 6                    # futility-only look after 6 of the 10 documents, pilot included
L3_FUTILITY_FAILS = 2               # >= 2 of 10 fail to parse: infrastructure stop (fix under rule 2, rerun)
BOOKED_L4_S = {V1: 4100.0, V2: 3300.0}      # ESTIMATE (PREDECLARED offline plan); the pilot replaces them
BUDGET_X, HARD_STOP_X = 2.0, 1.5
ARMS = {"A": (V1, G110.V0), "B": (V2, G110.V0), "C": (V2, G110.V1)}      # arm -> (ingest pipeline, planner); run order A, B, C
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_L3, RC_PARTIAL = X.RC_OK, X.RC_PREFLIGHT, X.RC_GATE, X.RC_PRECOND, X.RC_BUDGET, X.RC_L3, X.RC_PARTIAL
M4_CALLS_SOFT, M4_PREFILL_MAX = X.M4_MEDIAN_CALLS_SOFT, X.M4_MEDIAN_PREFILL_MAX
STRATA = D113_DESIGN.STRATA      # PREDECLARED *Metrics*: declared strata (a stratum is reported, not tested)
STATEMENT = {"revenue": "income", "net_income": "income", "capex": "cash_flow", "operating_cash_flow": "cash_flow",
             "dividends_paid": "cash_flow", "total_assets": "balance_sheet", "total_current_assets": "balance_sheet",
             "total_current_liabilities": "balance_sheet"}
LABEL = {"revenue": "Total revenues", "net_income": "Net income", "capex": "Capital expenditures", "total_assets": "Total assets",
         "operating_cash_flow": "Net cash provided by operating activities", "total_current_assets": "Total current assets",
         "total_current_liabilities": "Total current liabilities", "dividends_paid": "Dividends paid"}
PRINTED_HEADERS = D113_DESIGN.PRINTED_HEADERS


# --------------------------------------------------------------------------- inputs and pins
def load_inputs(set_dir: Path = D113) -> dict[str, Any]:
    inp = X.load_inputs(corpus=set_dir / "corpus.json", golds=set_dir / "golds.jsonl",
                        verification=set_dir / "verification_record.json", facts_sheet=set_dir / "facts_sheet.csv")
    inp["cells"] = X.m1_cells(set_dir / "facts_sheet.csv", X.verified_agg(inp), [d["doc_name"] for d in inp["corpus"]["docs"]])
    inp["set_dir"] = set_dir
    return inp


def pin_problems(set_dir: Path = D113, *, predeclared_sha: str | None = None, frozen: dict[str, str] | None = None,
                 amendment_sha: str | None = None) -> list[str]:
    out = []
    want_am = AMENDMENT_SHA if amendment_sha is None else amendment_sha
    q_am = set_dir / AMENDMENT_FILE
    got_am = SHA256(q_am) if q_am.exists() else None
    if got_am != want_am:
        out.append(f"{AMENDMENT_FILE} sha256 {got_am} != recorded {want_am}")
    want_pd = PREDECLARED_SHA if predeclared_sha is None else predeclared_sha
    p = set_dir / PREDECLARED_FILE
    got = SHA256(p) if p.exists() else None
    if got != want_pd:
        out.append(f"{PREDECLARED_FILE} sha256 {got} != recorded {want_pd}")
    side = set_dir / "PREDECLARED_AGG113.sha256"
    if side.exists() and side.read_text().split()[0] != want_pd:
        out.append("PREDECLARED_AGG113.sha256 disagrees with the runner pin")
    for name, want in (FROZEN if frozen is None else frozen).items():
        q = set_dir / name
        got = SHA256(q) if q.exists() else None
        if got != want:
            out.append(f"frozen input {name} sha256 {got} != recorded {want}")
    vc = set_dir.parent / "verify_phase1_cells.py"
    if not vc.exists() or SHA256(vc) != VERIFY_CELLS_SHA:
        out.append("verify_phase1_cells.py is not the reused-unchanged AGG-1.1 file")
    return out


def corpus_problems(inp: dict[str, Any], *, check_pdfs: bool = True) -> list[str]:
    c = inp["corpus"]
    out = []
    if X.digest_docs(c["docs"]) != c.get("manifest_digest_sha256") or c.get("manifest_digest_sha256") != MANIFEST_DIGEST:
        out.append("corpus manifest digest differs from the PREDECLARED pin")
    if len(c["docs"]) != N_DOCS:
        out.append(f"{len(c['docs'])} documents, PREDECLARED has {N_DOCS}")
    if tuple(c.get("pilot_docs") or ()) != PILOT_DOCS:
        out.append("corpus.json pilot_docs differ from the PREDECLARED pilot")
    if len(inp["cells"]) != N_CELLS_FILLED:
        out.append(f"{len(inp['cells'])} M1 cells, PREDECLARED has {N_CELLS_FILLED}")
    pc = [x for x in inp["cells"] if x[0] in PILOT_DOCS]
    if len(pc) != PILOT_CELLS:
        out.append(f"{len(pc)} pilot cells, PREDECLARED has {PILOT_CELLS}")
    agg = X.verified_agg(inp)
    if len(agg) != N_QUESTIONS:
        out.append(f"{len(agg)} verified questions, PREDECLARED has {N_QUESTIONS}")
    if check_pdfs:
        bad = [d["doc_name"] for d in c["docs"] if not Path(d["pdf"]).exists() or SHA256(Path(d["pdf"])) != d["sha256"]]
        if bad:
            out.append(f"pdf missing or sha256 mismatch: {bad}")
    return out


def build_check(set_dir: Path = D113) -> dict[str, Any]:
    """``build_agg113.py check`` under the system python (CPython 3.12 sums floats more exactly than the 3.11 orchestrator venv)."""
    py = "/usr/bin/python3" if sys.version_info < (3, 12) and Path("/usr/bin/python3").exists() else sys.executable
    cp = subprocess.run([py, "-I", "-B", str(set_dir / "build_agg113.py"), "check", "--out", str(set_dir)], capture_output=True, text=True,
                        timeout=300)
    return {"rc": cp.returncode, "python": py, "out": (cp.stdout + cp.stderr)[-300:]}


def fiscal_years(corpus: dict[str, Any]) -> dict[str, str]:
    return G.fiscal_years(corpus)


def doc_order(corpus: dict[str, Any]) -> list[str]:
    """Pilot documents first (declared order), then the other 7 by ascending PDF size (ties by name): known without a parse."""
    rest = sorted((d for d in corpus["docs"] if d["doc_name"] not in PILOT_DOCS), key=lambda d: (d["bytes"], d["doc_name"]))
    return list(PILOT_DOCS) + [d["doc_name"] for d in rest]


def january_end_docs(corpus: dict[str, Any]) -> list[str]:
    return [d["doc_name"] for d in corpus["docs"] if str(d.get("fiscal_year_end", ""))[5:7] == "01"]


# --------------------------------------------------------------------------- the fiscal-year contract (preflight hard check)
def fy_contract(corpus: dict[str, Any], cells: list[tuple[str, str, float]], *, orch: Any = None, fy_end: bool = False) -> dict[str, Any]:
    """PREDECLARED checkpoint 0: the January-end filers (Best Buy, Walmart: fiscal 2022 ends 2022-01; Foot Locker, Ulta: ends 2023-01) must be
    reachable as fiscal ``2022`` by the L3 period logic and by ``m1_hits``. No model, no PDF, no store: the L3 table logic of the pinned
    orchestrator is run on one fixture table per (document, statement) whose header is the one the filing PRINTS; the fact it makes for the
    fiscal-2022 (first) column is then read by the frozen ``m1_hits`` exactly as a real fact would be. A period that does not contain 2022 is
    a problem: the runner's own period reading, not the pipeline, would then decide M1 (rule 2: needs a dated amendment BEFORE any data).

    Also the description rule of ``agg1314_zero_cell_diag.fy_contract_problems`` (no L4 field description may name another fiscal year).

    ``fy_end=True`` (PREDECLARED_AGG113_AMENDMENT1_20261114.md) runs the same fixtures with the orchestrator's opt-in fiscal-year-end column
    labelling on, for BOTH L3 versions; ``False`` is the original, unamended check (it fails on the real corpus)."""
    from src.aggregation import facts_det  # noqa: PLC0415
    problems: list[str] = []
    descs = {f: G16.field_desc(f, "2022") for f in A.METRICS}
    problems += D.fy_contract_problems(descs, [{"field": f, "fy": "2022"} for f in A.METRICS])
    jan = january_end_docs(corpus)
    by_doc: dict[str, dict[str, Any]] = {}
    versions = (("v1_legacy", None), ("v2", facts_det.L3_PERIOD_V2))
    for doc in jan:
        hdrs = PRINTED_HEADERS.get(doc)
        if hdrs is None:
            problems.append(f"{doc}: January-end filer with no declared printed-header fixture")
            continue
        fye = next(d["fiscal_year_end"] for d in corpus["docs"] if d["doc_name"] == doc)
        rec = {"fiscal_year_end": fye, "cells": {}}
        for field in sorted({f for d, f, _g in cells if d == doc}):
            hdr = hdrs[STATEMENT[field]]
            sentinel = 1000.0 + len(rec["cells"])
            nums = [sentinel] + [sentinel + 10 * (i + 1) for i in range(len(hdr) - 1)]
            table = {"table_id": f"fx:{doc}:{field}", "doc_hash": "fx", "doc_id": doc, "entity": doc, "chunk_id": "fx#0", "page": 1,
                     "caption": "($ in millions)", "rows": [[""] + list(hdr), [LABEL[field]] + [f"{v:,.0f}" for v in nums]]}
            seen = {}
            for vname, pv in versions:
                kw = {} if pv is None else {"period_version": pv, "fiscal_year": "2022"}
                if fy_end:
                    kw = {**kw, "fiscal_year": "2022", "fy_end": FY_END_MONTHS}
                facts = facts_det.facts_from_table(table, "", **kw)
                f0 = next((f for f in facts if f["value"] == sentinel), None)
                period = f0["period"] if f0 else None
                hit = bool(X.m1_hits([{"doc_id": doc, "field": field, "value": sentinel, "unit": "USD_millions", "period": period}],
                                     [(doc, field, sentinel)], unit_scale=X.M1R_UNIT_SCALE)[0]) if f0 else False
                seen[vname] = {"period": period, "m1_hits_sees_it": hit}
                if not hit:
                    problems.append(f"{doc}/{field} ({STATEMENT[field]}, printed first column {hdr[0]!r}): L3 {vname} gives the fiscal-2022 "
                                    f"column the period {period!r}; m1_hits needs '2022' in the period")
            rec["cells"][field] = {"statement": STATEMENT[field], "printed_first_column": hdr[0], **seen}
        by_doc[doc] = rec
    return {"ok": not problems, "problems": problems, "january_end_docs": jan, "by_doc": by_doc, "fy_end": bool(fy_end),
            "n_cells_checked": sum(len(v["cells"]) for v in by_doc.values())}


def store_fy_check(store: Any, corpus: dict[str, Any], cells: list[tuple[str, str, float]], corpus_id: str = CORPUS_ID) -> dict[str, Any]:
    """The same contract on the REAL L3 facts of the shared seed (counts only): per January-end document, how many gold cells have an L3 fact
    carrying the gold value (0.5%) under a period containing 2022, under another period only, or no L3 fact at all."""
    rows = [f for f in store.facts(corpus_id) if str(f.get("extractor") or "").startswith("l3")]
    out: dict[str, dict[str, int]] = {}
    for doc in january_end_docs(corpus):
        c = {"cells": 0, "l3_value_with_2022": 0, "l3_value_other_period_only": 0, "no_l3_value": 0}
        for d, f, g in cells:
            if d != doc:
                continue
            c["cells"] += 1
            vals = [(x, float(x["value"]) * X.M1R_UNIT_SCALE.get(x.get("unit"), 1.0)) for x in rows
                    if x.get("doc_id") == d and x.get("field") == f and x.get("value") is not None]
            match = [x for x, v in vals if abs(abs(v) - abs(g)) <= P.CFG.rel_tol * abs(g) + 1e-9]
            if any("2022" in str(x.get("period") or "") for x in match):
                c["l3_value_with_2022"] += 1
            elif match:
                c["l3_value_other_period_only"] += 1
            else:
                c["no_l3_value"] += 1
        out[doc] = c
    return out


# --------------------------------------------------------------------------- the offline plan / estimates
def estimates(plans: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """ESTIMATE rows from PREDECLARED_AGG113.md's offline plan table; with a store-derived v2 window plan the v2 figure is re-derived."""
    out = {"label": "ESTIMATE", "l3_s": 1700, "v1_cpu_s": 4100, "v2_cpu_s": 3300, "gpu_s": 2400,
           "basis": "PREDECLARED *Offline plan and estimates*: v1 6.07 calls/unit x 8.2 s, v2 4.03 calls/unit x 9.68 s, 80 units each"}
    if plans:
        tot = G.plan_totals(plans)
        out["v2_plan_from_store"] = {"calls_exact": tot["calls"], "wall_est_s": round(tot["wall_est_s"] + G.SERVER_START_S + G.L3_ESTIMATE_S)}
    return out


# --------------------------------------------------------------------------- store: copies, wipe, arm L3
def seed_ready(store: Any, corpus: dict[str, Any], corpus_id: str = CORPUS_ID) -> list[str]:
    docs = {d["doc_id"] for d in store.corpus_docs(corpus_id)}
    return sorted(set(d["doc_name"] for d in corpus["docs"]) - docs)


def failed_docs(store: Any, corpus: dict[str, Any], corpus_id: str = CORPUS_ID) -> list[str]:
    """Documents the shared L3 load could not parse (absent from the store): their cells count as misses, never silently dropped."""
    return seed_ready(store, corpus, corpus_id)


def doc_l3_facts_113(store: Any, corpus_id: str, doc_id: str, fiscal_year: str | None, fy_end: bool) -> tuple[list[dict[str, Any]], int, int]:
    """``l3_load.doc_l3_facts`` (the table and regex facts of one document) with the orchestrator's fiscal-year-end column labelling when
    ``fy_end``; with it off this IS ``l3_load.doc_l3_facts``. The corpus-level rebuild uses the same rule, so one fact is never written
    twice under two period labels (the period is part of the table fact's id)."""
    if not fy_end:
        return L3.doc_l3_facts(store, corpus_id, doc_id)
    from src.aggregation import facts_det  # noqa: PLC0415
    chunks = [c for c in store.chunks(corpus_id) if c["doc_id"] == doc_id]
    by_id = {c["chunk_id"]: c for c in chunks}
    ent = {d["doc_hash"]: d["entity"] for d in store.corpus_docs(corpus_id)}
    facts: list[dict[str, Any]] = []
    n_tab = 0
    for t in (t for t in store.tables(corpus_id) if t["doc_id"] == doc_id):
        got = facts_det.facts_from_table(t, (by_id.get(t["chunk_id"]) or {}).get("text", ""), fiscal_year=fiscal_year, fy_end=FY_END_MONTHS)
        n_tab += len(got)
        facts.extend(got)
    n_rx = 0
    for c in chunks:
        got = facts_det.facts_from_text(c, ent.get(c["doc_hash"]))
        n_rx += len(got)
        facts.extend(got)
    return facts, n_tab, n_rx


def seed_fy_end_problem(seed_store: Path) -> str | None:
    """The l3 row writes ``result.l3.json`` next to its store; an arm that reads a seed built WITHOUT the fiscal-year-end rule would silently
    lose the amendment's cells, so the seed must say ``fy_end`` true and the amendment sha it was built under."""
    f = seed_store.parent / "result.l3.json"
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        return f"{f} missing or unreadable: the seed is not a recorded l3 row output"
    if r.get("fy_end") is not True or r.get("amendment_sha256") != AMENDMENT_SHA or r.get("complete") is not True:
        return f"{f}: the seed was not built complete under the fy_end rule and amendment {AMENDMENT_SHA[:12]}"
    return None


def arm_l3(rec: P.Recorder, store: Any, inp: dict[str, Any], pipeline: str, corpus_id: str = CORPUS_ID, fy_end: bool = False) -> None:
    """v1 keeps the seed's legacy L3 facts exactly (the Phase 0 / AGG-1.1 path); v2 wipes the COPY's facts and re-extracts L3 under
    ``l3_period.v2`` (no model, no PDF) as AGG-1.12 did."""
    unit = "l3v1" if pipeline == V1 else "l3v2"
    if unit in rec.done:
        return
    sym = G.orch_symbols()
    spec = sym["pl"].pipeline_spec(PIPELINE_OF[pipeline])
    t0 = time.time()
    n_before = len(store.facts(corpus_id))
    if pipeline == V2:
        G._wipe_facts(store)
        facts = spec.extract_l3(store, corpus_id, fiscal_years(inp["corpus"]), FY_END_MONTHS if fy_end else None)
        store.write_facts(facts)
        h = store.corpus_content_hash(corpus_id)
        store.record_index(corpus_id, h, "l3_facts_det", "", layer="L3", n_items=len(facts))
    n_l3 = len(store.facts(corpus_id))
    raw = G19.raw_hits(G19.fact_rows(store.conn, corpus_id), inp["cells"])
    rec.emit(unit, unit=unit, pipeline=spec.as_record(), fy_end=bool(fy_end), n_facts_seed=n_before, n_l3_facts=n_l3, m1_l3_only_raw=sum(raw.values()),
             n_cells=len(raw), llm_calls=0, prefill_tokens=0, wall_s=time.time() - t0, final=True)


# --------------------------------------------------------------------------- shared L3 load (phase l3): no model
def run_l3(rec: P.Recorder, out: Path, inp: dict[str, Any], *, orch_root: Path, guard: Callable[[], None],
           parse_fn: Callable[[Path], Any] | None = None, now: Callable[[], float] = time.time, smoke: bool = False,
           corpus_id: str = CORPUS_ID, orch_pin: str = "", research_pin: str = "", fy_end: bool = False) -> dict[str, Any]:
    """Checkpoint 1 (CPU, no model): each of the 10 documents is parsed with the existing document flow (local ODL, no OCR server), stored,
    L3-extracted (legacy path) and recorded with the ``l3_load.py`` record kinds. A fresh store (no base store). >= 2 failed parses
    stop the run (rc 6, infrastructure); fewer are ``doc_fail`` (final): their cells count as misses."""
    parse_fn = parse_fn or L3.real_parse
    for k, v in L3.JOB_ENV.items():
        os.environ[k] = v
    corpus = inp["corpus"]
    order = sorted(corpus["docs"], key=lambda d: (d["bytes"], d["doc_name"]))
    order = [d["doc_name"] for d in order]
    if smoke:
        order = order[:1]
    by_name = {d["doc_name"]: d for d in corpus["docs"]}
    from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    from src.models.document import DocumentPreprocessResult  # noqa: PLC0415
    store = CorpusStore(out / "store.sqlite")
    fyears = fiscal_years(corpus)
    rec.emit("meta", phase="l3", smoke=smoke, schema="agg113_l3.v1", fy_end=bool(fy_end), amendment_sha256=AMENDMENT_SHA, model_calls=0, orch_pin=orch_pin, research_pin=research_pin,
             n_planned=len(order), corpus_id=corpus_id, final=False)
    failed = {r["doc"] for r in rec.records if r["kind"] == "doc_fail"} - {r["doc"] for r in rec.records if r["kind"] == "doc_done"}
    cache_dir = out / "cache"
    try:
        for name in order:
            d = by_name[name]
            unit = f"l3:{name}"
            if unit in rec.done:
                continue
            guard()
            if not Path(d["pdf"]).exists() or SHA256(Path(d["pdf"])) != d["sha256"]:
                rec.emit("doc_fail", unit=unit, doc=name, error="pdf missing or sha256 mismatch", wall_s=0.0, final=True)
                failed.add(name)
            else:
                t0 = now()
                cache = cache_dir / f"{name}.{d['sha256'][:12]}.json"
                try:
                    if cache.exists():
                        blob = json.loads(cache.read_text())
                        result = DocumentPreprocessResult.from_cache_dict(blob)
                        parse_s, cached = blob.get("_parse_s"), True
                    else:
                        tp = now()
                        result = parse_fn(Path(d["pdf"]))
                        parse_s, cached = now() - tp, False
                        payload = result.to_cache_dict()
                        payload["_parse_s"] = parse_s
                        X._atomic_write(cache, json.dumps(payload, default=str))
                    rec.emit("parse", doc=name, pdf_bytes=d["bytes"], parse_s=parse_s, cached=cached, total_pages=result.total_pages,
                             pages_seen=max((s.page_end for s in result.sections), default=0), n_sections=len(result.sections),
                             n_tables=len(result.tables), n_chars=sum(len(s.content) for s in result.sections),
                             failed_pages=len(result.failed_pages))
                    rc = store.ingest(result, corpus_id=corpus_id, doc_id=name, entity=d["company"])
                    rec.emit("store", doc=name, status=rc.status, doc_hash=rc.doc_hash, n_chunks=rc.n_chunks, n_tables=rc.n_tables,
                             n_tokens_est=rc.n_tokens_est)
                    facts, n_tab, n_rx = doc_l3_facts_113(store, corpus_id, name, fyears.get(name), fy_end)
                    store.write_facts(facts)
                    units: dict[str, int] = {}
                    for f in facts:
                        units[f["unit"]] = units.get(f["unit"], 0) + 1
                    rec.emit("l3_doc", doc=name, n_facts=len(facts), n_table_facts=n_tab, n_regex_facts=n_rx,
                             fields=len({f["field"] for f in facts}), units=units, llm_calls=0, prefill_tokens=0)
                    rec.emit("doc_done", unit=unit, doc=name, pdf_sha256=d["sha256"], doc_hash=rc.doc_hash, wall_s=now() - t0, final=True)
                except Exception as e:  # noqa: BLE001 - recorded per document, the run continues until the futility rule
                    rec.emit("doc_fail", unit=unit, doc=name, error=f"{type(e).__name__}: {e}"[:500], wall_s=now() - t0, final=True)
                    failed.add(name)
            if len(failed) >= L3_FUTILITY_FAILS:
                rec.emit("stop", reason="futility_l3_parse", failed=sorted(failed), final=False)
                return {"complete": True, "infrastructure_stop": "l3_parse_futility", "failed": sorted(failed), "rc": RC_L3}
        if smoke:
            return {"complete": False, "smoke": True, "failed": sorted(failed), "rc": RC_OK}
        reg = IndexRegistry(store, fiscal_years=fyears, fy_end=FY_END_MONTHS) if fy_end else IndexRegistry(store)
        reg.get_or_build(corpus_id, "l1_bm25")
        facts = reg.get_or_build(corpus_id, "l3_facts_det")
        rec.emit("corpus", unit="corpus", n_docs=len(store.corpus_docs(corpus_id)), corpus_tokens=store.corpus_tokens(corpus_id),
                 n_chunks=len(store.chunks(corpus_id)), n_tables=len(store.tables(corpus_id)), n_facts=len(facts),
                 built_layers=sorted(store.built_layers(corpus_id)), corpus_content_hash=store.corpus_content_hash(corpus_id),
                 failed=sorted(failed), llm_calls=0, prefill_tokens=0)
        l3f = [f for f in store.facts(corpus_id) if str(f.get("extractor") or "").startswith("l3")]
        hits = X.m1_hits(l3f, inp["cells"], unit_scale=X.M1R_UNIT_SCALE)
        fy = store_fy_check(store, corpus, inp["cells"], corpus_id)
        rec.emit("m1_l3", unit="m1_l3", kind_note="L3-only, informational counts (the gate reads M1 after L4d)", gold_facts=len(inp["cells"]),
                 found=sum(hits), store_fy_check=fy, final=True)
        res = {"complete": True, "phase": "l3", "fy_end": bool(fy_end), "amendment_sha256": AMENDMENT_SHA, "n_docs": len(order),
               "failed": sorted(failed), "m1_l3_found": sum(hits),
               "m1_l3_n": len(hits), "store_fy_check": fy, "model_calls": 0, "rc": RC_OK}
        return res
    finally:
        store.close()


# --------------------------------------------------------------------------- M1 readings, M3 and the paired statistic (pure)
def hits_of(store: Any, cells: list[tuple[str, str, float]], corpus_id: str = CORPUS_ID) -> dict[str, bool]:
    return G19.raw_hits(G19.fact_rows(store.conn, corpus_id), cells)


def cell_keys(cells: Iterable[tuple[str, str, float]], docs: Iterable[str] | None = None) -> list[str]:
    ds = None if docs is None else set(docs)
    return [G.cell_key(d, f) for d, f, _g in cells if ds is None or d in ds]


def binom_two_sided(b: int, c: int) -> float | None:
    """Exact two-sided binomial p of b of (b + c) at 0.5; None when b + c = 0."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_113(first: dict[str, bool], second: dict[str, bool], keys: list[str]) -> dict[str, Any]:
    """PREDECLARED *Paired statistic*: b = only ``first`` right, c = only ``second``; z = (b - c) / sqrt(b + c); exact two-sided binomial p;
    d = (b - c) / n with SE sqrt(b + c - (b - c)^2 / n) / n and a 95% interval; the mechanical reading (not a gate)."""
    n = len(keys)
    b = sum(1 for k in keys if first.get(k) and not second.get(k))
    c = sum(1 for k in keys if second.get(k) and not first.get(k))
    both = sum(1 for k in keys if first.get(k) and second.get(k))
    out: dict[str, Any] = {"n": n, "b_only_first": b, "c_only_second": c, "both": both, "neither": n - both - b - c,
                           "first_correct": both + b, "second_correct": both + c}
    if b + c == 0:
        out.update(z=None, p=None, d=0.0, se=None, ci95=None, reading="inconclusive", meets_planned_z=False,
                   note="b + c = 0: the statistic is undefined (no discordant pairs)")
        return out
    z = (b - c) / math.sqrt(b + c)
    d = (b - c) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    out.update(z=z, p=binom_two_sided(b, c), d=d, se=se, ci95=[d - 1.96 * se, d + 1.96 * se],
               reading="v2_better" if (b > c and z >= 2.0) else "v2_worse" if (c > b and z <= -2.0) else "inconclusive",
               meets_planned_z=z >= 2.5)
    return out


def strata_113(per_arm_hits: dict[str, dict[str, bool]], cells: list[tuple[str, str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, docs in STRATA.items():
        keys = cell_keys(cells, docs)
        out[name] = {"docs": list(docs), "n": len(keys), **{arm: sum(bool(h.get(k)) for k in keys) for arm, h in per_arm_hits.items()}}
    return out


def cross_futility(v2_hits: dict[str, bool], v1_hits: dict[str, bool], keys: list[str]) -> dict[str, Any]:
    """PREDECLARED pilot / interim futility on M1: the v2 arm's raw hits on the SAME cells strictly below v1's."""
    h2, h1 = sum(bool(v2_hits.get(k)) for k in keys), sum(bool(v1_hits.get(k)) for k in keys)
    return {"n_cells": len(keys), "v2_found": h2, "v1_found": h1, "tripped": h2 < h1}


# --------------------------------------------------------------------------- ingest bookkeeping shared by both arms
def unit_v1(field: str, doc: str) -> str:
    return f"l4:{field}:{doc}"


def doc_complete(rec: P.Recorder, pipeline: str, doc: str, plan: dict[str, Any] | None) -> bool:
    if pipeline == V1:
        return all(unit_v1(f, doc) in rec.done for f in A.METRICS)
    return f"doc_l4:{doc}" in rec.done


def doc_tally(rec: P.Recorder, pipeline: str, docs: Iterable[str]) -> dict[str, Any]:
    docs = set(docs)
    if pipeline == V2:
        return G.tally_docs(r for d, r in G.doc_records(rec).items() if d in docs)
    ok = X.l4_ok(rec.records)
    sel = [r for u, r in ok.items() if u.split(":", 2)[2] in docs]
    t = X.tally(sel)
    return {"n_docs": len(docs), "llm_calls": t["llm_calls"], "prefill_tokens": t["prefill_tokens"], "wall_s": t["wall_s"],
            "bad_replies": 0, "chars": 0, "cache_n": 0}


def total_cells_units(corpus: dict[str, Any], failed: Iterable[str] = ()) -> int:
    return (len(corpus["docs"]) - len(set(failed))) * len(A.METRICS)


def pilot_eval(*, rec: P.Recorder, pipeline: str, plans: dict[str, dict[str, Any]] | None, stats: dict[str, dict[str, int]],
               hits: dict[str, bool], inp: dict[str, Any], failed: set[str], booked_s: float, pilot_wall_s: float = PILOT_WALL_S
               ) -> dict[str, Any]:
    """Checkpoint 2, arithmetic over the pilot records of ONE arm. Gate (futility): projected M3 over a kill bound. The M1 comparison with the
    other arm needs the other arm's pilot hits, so it is made by ``cross_check`` (online when the peer file exists, always again in ``score``).
    Budget stops (re-book, not gates): pilot wall > 1800 s; projected L4 wall > 2x booked. Counts only (rule 1)."""
    pdocs = [d for d in PILOT_DOCS if d not in failed]
    pil = doc_tally(rec, pipeline, pdocs)
    pcells = cell_keys(inp["cells"], PILOT_DOCS)
    found = sum(bool(hits.get(k)) for k in pcells)
    n_units = total_cells_units(inp["corpus"], failed)
    tokens, chunks = sum(v["tokens"] for v in stats.values()), sum(v["chunks"] for v in stats.values())
    pil_units = len(pdocs) * len(A.METRICS)
    if pipeline == V2 and plans:
        tot = G.plan_totals({d: plans[d] for d in plans if d not in failed})
        proj_calls = tot["calls"]
        per_char = pil["prefill_tokens"] / pil["chars"] if pil["chars"] else 0.0
        proj_prefill = per_char * tot["chars"]
        per_call_wall = pil["wall_s"] / pil["llm_calls"] if pil["llm_calls"] else 0.0
        proj_wall = per_call_wall * proj_calls
    else:
        k = (n_units / pil_units) if pil_units else 0.0
        proj_calls, proj_prefill, proj_wall = pil["llm_calls"] * k, pil["prefill_tokens"] * k, pil["wall_s"] * k
    proj_m3 = X.m3_values(proj_prefill, proj_calls, tokens, chunks)
    gates, budget = [], []
    if X.m3_trips(proj_m3):
        gates.append("M3")
    if pil["wall_s"] > pilot_wall_s:
        budget.append("pilot_wall_over_1800s")
    if proj_wall > BUDGET_X * booked_s:
        budget.append("projected_l4_wall_over_2x_booked")
    return {"pipeline": pipeline, "pilot": pil, "pilot_m1": {"cells": len(pcells), "found": found},
            "projected_calls": proj_calls, "projected_prefill": proj_prefill, "projected_m3": proj_m3, "projected_l4_wall_s": proj_wall,
            "booked_l4_s": booked_s, "hard_stop_s": HARD_STOP_X * proj_wall, "gates_tripped": gates, "budget_stops": budget}


def _write_cpu_result(out: Path, pipeline: str, **res: Any) -> dict[str, Any]:
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = PREDECLARED_SHA
    res["pipeline"] = PIPELINE_OF[pipeline]
    res["arm_pipeline"] = pipeline
    X._atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    return res


def _gate_stop(rec: P.Recorder, out: Path, pipeline: str, gates: list[str], where: str, **detail: Any) -> dict[str, Any]:
    """A predeclared futility / gate stop: filed as ``agg113_futility`` (M1 vs the other arm) or ``phase1_gate_tripped`` (M3)."""
    verdict = "agg113_futility" if any(g.startswith("M1") for g in gates) else "phase1_gate_tripped"
    rec.emit("gate_tripped", unit=f"gate_tripped:{where}", gates=gates, where=where, verdict=verdict, **detail)
    rec.emit("stop", reason=verdict, gates=gates, where=where, final=False)
    res = _write_cpu_result(out, pipeline, complete=True, futility_stop="+".join(gates), gate_tripped=gates, verdict=verdict, where=where,
                            m3_passed=False if "M3" in gates else None, rc=RC_GATE, **detail)
    res["rc"] = RC_GATE
    return res


def _budget_stop(rec: P.Recorder, reason: str, **detail: Any) -> dict[str, Any]:
    rec.emit("stop", reason=f"budget_stop:{reason}", final=False, **detail)
    return {"complete": False, "budget_stop": reason, "rc": RC_BUDGET, **detail}


def write_checkpoint(out: Path, name: str, hits: dict[str, bool], docs: Iterable[str], cells: list[tuple[str, str, float]]) -> None:
    """Machine file (never read by a person before ``score``): the raw per-cell hits over the checkpoint's documents."""
    keys = cell_keys(cells, docs)
    X._atomic_write(out / f"checkpoint.{name}.json", json.dumps({"docs": sorted(set(docs)), "hits": {k: bool(hits.get(k)) for k in keys}},
                                                                 indent=0, sort_keys=True) + "\n")


def cross_check(pipeline: str, name: str, hits: dict[str, bool], docs: Iterable[str], cells: list[tuple[str, str, float]],
                peer_out: Path | None) -> dict[str, Any] | None:
    """v2 only: compare against the v1 arm's checkpoint file when it exists (None while the peer has not reached that checkpoint)."""
    if pipeline != V2 or peer_out is None:
        return None
    f = peer_out / f"checkpoint.{name}.json"
    if not f.exists():
        return None
    try:
        peer = json.loads(f.read_text())["hits"]
    except (OSError, ValueError, KeyError):
        return None
    keys = [k for k in cell_keys(cells, docs) if k in peer]
    return cross_futility(hits, peer, keys)


def interim_look(rec: P.Recorder, out: Path, pipeline: str, store: Any, inp: dict[str, Any], plans: dict[str, Any] | None,
                 stats: dict[str, dict[str, int]], failed: set[str], peer_out: Path | None) -> dict[str, Any] | None:
    """Checkpoint 3, futility only (never an early GO): once 6 documents are complete, running M3 over a bound, or v2's raw M1 on those
    documents' cells below v1's on the same cells."""
    done = [d for d in doc_order(inp["corpus"]) if d not in failed and doc_complete(rec, pipeline, d, plans and plans.get(d))]
    if len(done) < INTERIM_DOCS or "interim6" in rec.done:
        return None
    hits = hits_of(store, inp["cells"])
    docs6 = done[:INTERIM_DOCS]
    keys = cell_keys(inp["cells"], docs6)
    t = doc_tally(rec, pipeline, docs6)
    tok, chunks = X.sum_stats(stats, docs6)
    m3 = X.m3_values(t["prefill_tokens"], t["llm_calls"], tok, chunks)
    write_checkpoint(out, "interim6", hits, docs6, inp["cells"])
    cx = cross_check(pipeline, "interim6", hits, docs6, inp["cells"], peer_out)
    gates = (["M1_interim_below_v1"] if cx and cx["tripped"] else []) + (["M3"] if X.m3_trips(m3) else [])
    rec.emit("interim6", unit="interim6", n_docs_done=len(done), found=sum(bool(hits.get(k)) for k in keys), cells=len(keys), running_m3=m3,
             cross_check=cx, cross_check_pending=bool(pipeline == V2 and cx is None), gates_tripped=gates)
    return _gate_stop(rec, out, pipeline, gates, "interim", running_m3=m3, cross_check=cx) if gates else None


def doc_plans_v2(store: Any, corpus: dict[str, Any], failed: set[str]) -> dict[str, dict[str, Any]]:
    plans = G.doc_plans(store, corpus, CORPUS_ID)
    return {d: v for d, v in plans.items() if d not in failed}


# --------------------------------------------------------------------------- ingest (one arm)
def run_ingest(rec: P.Recorder, out: Path, inp: dict[str, Any], backend: Any, *, pipeline: str, guard: Callable[[], None],
               seed_store: Path, stage: str = "all", smoke: bool = False, peer_out: Path | None = None,
               booked_l4_s: float | None = None, pilot_wall_s: float = PILOT_WALL_S, fy_end: bool = False) -> dict[str, Any]:
    """Checkpoints 2-4 for ONE arm (``pipeline`` v1 or v2). Copies the shared seed once (never mutated), prepares the arm's L3, runs the 3 pilot
    documents (futility on projected M3, budget stops), then ``stage`` ``all`` the other 7 with the interim look after 6, then the final look
    and ``result.cpu.json``. ``stage`` = ``pilot`` stops after the pilot. Smoke: ONE real call on the smallest document, then return."""
    sym = G.orch_symbols()
    corpus = inp["corpus"]
    store_path = out / "store.sqlite"
    if not store_path.exists():
        if not seed_store.exists():
            rec.emit("precondition_failed", reason=f"no seed store {seed_store} (run the l3 row first)")
            return {"complete": False, "rc": RC_PRECOND}
        why = seed_fy_end_problem(seed_store) if fy_end else None
        if why:
            rec.emit("precondition_failed", reason=why)
            return {"complete": False, "rc": RC_PRECOND}
        P.copy_store(seed_store, store_path)
        rec.emit("store_copied", src=str(seed_store), sha256=SHA256(store_path), final=False)
    store = P.open_store(store_path)
    booked = booked_l4_s or BOOKED_L4_S[pipeline]
    try:
        failed = set(failed_docs(store, corpus))
        if len(failed) >= L3_FUTILITY_FAILS:
            rec.emit("precondition_failed", reason=f"the shared seed lacks {sorted(failed)}")
            return {"complete": False, "rc": RC_PRECOND}
        arm_l3(rec, store, inp, pipeline, fy_end=fy_end)
        order = [d for d in doc_order(corpus) if d not in failed]
        stats = X.doc_stats(store, CORPUS_ID)
        fy = fiscal_years(corpus)
        plans = doc_plans_v2(store, corpus, failed) if pipeline == V2 else None
        state = {r["unit"]: r for r in rec.records if r.get("unit") in ("pilot_done", "interim6")}
        if "pilot_done" in state and not smoke:
            pd = state["pilot_done"]
            if pd.get("gates_tripped"):
                return {"complete": True, "phase": "ingest", "gate_tripped": pd["gates_tripped"], "rc": RC_GATE}
            if stage == "pilot":
                return {"complete": not pd.get("budget_stops"), "phase": "pilot", "already_done": True,
                        **({"budget_stop": "+".join(pd["budget_stops"]), "rc": RC_BUDGET} if pd.get("budget_stops") else {"rc": RC_OK})}
        todo = [d for d in order if d in PILOT_DOCS] if stage == "pilot" else order
        if smoke:
            todo = [min(order, key=lambda d: next(x["bytes"] for x in corpus["docs"] if x["doc_name"] == d))]
        backend_rb = G16.RecordingBackend(backend) if pipeline == V2 else backend
        tb = sym["ft"].TypedBackfill(store, backend_rb) if pipeline == V2 else None
        units_v1 = P.windows(store, CORPUS_ID, P.CFG.window_chars) if pipeline == V1 else None
        if "interim6" not in rec.done and not smoke and stage != "pilot" and "pilot_done" in state:
            early = interim_look(rec, out, pipeline, store, inp, plans, stats, failed, peer_out)
            if early:
                return early
        for doc in todo:
            if pipeline == V2:
                plan = plans[doc]
                descs = G.descs_for(fy[doc])
                for w in plan["windows"]:
                    if G.window_unit(doc, w["chunk_id"]) in rec.done:
                        continue
                    if doc in PILOT_DOCS and not smoke and "pilot_done" not in rec.done:
                        used = sum(r["wall_s"] for r in rec.records if r.get("kind") == "t_window" and r.get("doc") in PILOT_DOCS and r.get("final"))
                        if used > pilot_wall_s:
                            rec.emit("pilot_bound_hit", unit="pilot_bound_hit", wall_s=used, bound_s=pilot_wall_s)
                            return _budget_stop(rec, "pilot_wall_over_1800s", wall_s=used)
                    G.run_window(rec, tb, backend_rb, doc, w, plan["chosen"][w["chunk_id"]], descs, guard)
                    if smoke:
                        return {"complete": False, "smoke": True, "rc": RC_OK}
                    st = _ingest_budget(rec, state)
                    if st and doc not in PILOT_DOCS:
                        return st
                if not G.doc_windows_done(rec, doc, plan):
                    return {"complete": False, "reason": f"{doc}: windows errored; re-grant retries them", "rc": RC_PARTIAL}
                if f"doc_l4:{doc}" not in rec.done:
                    G.finish_doc(rec, tb, store, doc, plan, CORPUS_ID)
            else:
                for f in A.METRICS:
                    if unit_v1(f, doc) in rec.done:
                        continue
                    if doc in PILOT_DOCS and not smoke and "pilot_done" not in rec.done:
                        ok_now = X.l4_ok(rec.records)
                        used = X.tally(r for u, r in ok_now.items() if u.split(":", 2)[2] in PILOT_DOCS)["wall_s"]
                        if used > pilot_wall_s:
                            rec.emit("pilot_bound_hit", unit="pilot_bound_hit", wall_s=used, bound_s=pilot_wall_s)
                            return _budget_stop(rec, "pilot_wall_over_1800s", wall_s=used)
                    X.l4_unit_fy(rec, store, backend, units_v1, f, doc, guard, P.CFG, fy=fy[doc])
                    if smoke:
                        return {"complete": False, "smoke": True, "rc": RC_OK}
                    st = _ingest_budget(rec, state)
                    if st and doc not in PILOT_DOCS:
                        return st
                if not all(unit_v1(f, doc) in rec.done for f in A.METRICS):
                    return {"complete": False, "reason": f"{doc}: units errored; re-grant retries them", "rc": RC_PARTIAL}
            # pilot evaluation the moment the last pilot document is complete
            if "pilot_done" not in rec.done and all(doc_complete(rec, pipeline, d, plans and plans.get(d)) for d in PILOT_DOCS if d not in failed):
                hits = hits_of(store, inp["cells"])
                ev = pilot_eval(rec=rec, pipeline=pipeline, plans=plans, stats=stats, hits=hits, inp=inp, failed=failed, booked_s=booked,
                                pilot_wall_s=pilot_wall_s)
                write_checkpoint(out, "pilot", hits, PILOT_DOCS, inp["cells"])
                cx = cross_check(pipeline, "pilot", hits, PILOT_DOCS, inp["cells"], peer_out)
                ev["cross_check"] = cx
                ev["cross_check_pending"] = bool(pipeline == V2 and cx is None)
                if cx and cx["tripped"]:
                    ev["gates_tripped"] = ev["gates_tripped"] + ["M1_pilot_below_v1"]
                rec.emit("pilot_done", unit="pilot_done", **ev)
                X._atomic_write(out / "result.pilot.json", json.dumps({"phase": "pilot", "complete": True, **ev, "predeclared_sha256": PREDECLARED_SHA},
                                                                       indent=1, default=str) + "\n")
                state["pilot_done"] = {"hard_stop_s": ev["hard_stop_s"], "gates_tripped": ev["gates_tripped"]}
                if ev["gates_tripped"]:
                    return _gate_stop(rec, out, pipeline, ev["gates_tripped"], "pilot", pilot=ev)
                if ev["budget_stops"]:
                    return _budget_stop(rec, "+".join(ev["budget_stops"]), pilot=ev)
                if stage == "pilot":
                    return {"complete": True, "phase": "pilot", "rc": RC_OK, **ev}
            if stage != "pilot" and "pilot_done" in rec.done:
                early = interim_look(rec, out, pipeline, store, inp, plans, stats, failed, peer_out)
                if early:
                    return early
        if stage == "pilot":
            return {"complete": False, "reason": "pilot documents incomplete", "rc": RC_PARTIAL}
        return finish_ingest(rec, out, pipeline, store, inp, plans, stats, failed)
    finally:
        store.close()


def _ingest_budget(rec: P.Recorder, state: dict[str, Any]) -> dict[str, Any] | None:
    pd = state.get("pilot_done")
    if not pd:
        return None
    fresh = sum((r.get("wall_s") or 0.0) for r in rec.records if r.get("kind") in ("t_window", "l4_unit") and r.get("final")
                and r.get("doc") not in PILOT_DOCS)
    if 0 < pd.get("hard_stop_s", 0) < fresh:
        return _budget_stop(rec, "l4_wall_over_1.5x_pilot_projection", wall_s=fresh, hard_stop_s=pd["hard_stop_s"])
    return None


def finish_ingest(rec: P.Recorder, out: Path, pipeline: str, store: Any, inp: dict[str, Any], plans: dict[str, Any] | None,
                  stats: dict[str, dict[str, int]], failed: set[str]) -> dict[str, Any]:
    """Checkpoint 4 for one arm: M1 (raw and canonical), M3, per-document and per-stratum counts; ``result.cpu.json`` and the machine file
    ``cells.json`` (per-cell vectors; read only by ``score``). M3 is a gate here; M1 has no gate at the final look (futility is pilot / interim)."""
    sym = G.orch_symbols()
    spec = sym["pl"].pipeline_spec(PIPELINE_OF[pipeline])
    docs = [d for d in doc_order(inp["corpus"]) if d not in failed]
    missing = [d for d in docs if not doc_complete(rec, pipeline, d, plans and plans.get(d))]
    if missing:
        return {"complete": False, "reason": f"documents incomplete: {missing}; re-grant retries them", "rc": RC_PARTIAL}
    t = doc_tally(rec, pipeline, docs)
    tokens, chunks = sum(v["tokens"] for v in stats.values()), sum(v["chunks"] for v in stats.values())
    m3 = X.m3_values(t["prefill_tokens"], t["llm_calls"], tokens, chunks)
    canon_stats = spec.refresh_canonical(store, CORPUS_ID)
    rows = G19.fact_rows(store.conn, CORPUS_ID)
    now = G.readings(rows, store.canonical_facts(CORPUS_ID), inp["cells"])
    keys = cell_keys(inp["cells"])
    found = sum(bool(now["raw"][k]) for k in keys)
    summ = X.m1_summary(inp["cells"], [now["raw"][G.cell_key(d, f)] for d, f, _g in inp["cells"]], set())
    n_units = len(docs) * len(A.METRICS)
    rec.emit("ingest", unit="ingest", pipeline=PIPELINE_OF[pipeline], prefill_tokens=t["prefill_tokens"], corpus_tokens=tokens, llm_calls=t["llm_calls"],
             n_chunks=chunks, calls_per_unit=t["llm_calls"] / n_units if n_units else None, wall_s_fresh=t["wall_s"], bad_replies=t["bad_replies"])
    rec.emit("m1", unit="m1", gold_facts=len(keys), found=found, m1=found / len(keys), per_doc=summ["per_doc"],
             canonical_ok=sum(bool(now["canonical"][k]) for k in keys), canonical=canon_stats, doc_fail=sorted(failed))
    X._atomic_write(out / "cells.json", json.dumps({"pipeline": pipeline, "raw": {k: bool(v) for k, v in sorted(now["raw"].items())},
                                                     "canonical": {k: bool(v) for k, v in sorted(now["canonical"].items())}}, indent=0) + "\n")
    gates = ["M3"] if X.m3_trips(m3) else []
    detail = {"m1_found": found, "m1_n": len(keys), "m1": found / len(keys), "canonical_ok": sum(bool(now["canonical"][k]) for k in keys),
              "m3": m3, "m3_passed": not X.m3_trips(m3), "per_doc": summ["per_doc"], "doc_fail": sorted(failed),
              "strata_counts": strata_113({pipeline: now["raw"]}, inp["cells"]), "canonical": {"rule": canon_stats["rule"], "guards": canon_stats.get("guards")},
              "calls_per_unit": t["llm_calls"] / n_units if n_units else None, "bad_reply_rate": (t["bad_replies"] / t["llm_calls"]) if t["llm_calls"] else None}
    rec.emit("gates_cpu", unit="gates_cpu", gates_tripped=gates, **{k: v for k, v in detail.items() if k != "strata_counts"})
    if gates:
        return _gate_stop(rec, out, pipeline, gates, "final", **detail)
    return _write_cpu_result(out, pipeline, complete=True, futility_stop=None, gate_tripped=[], rc=RC_OK, **detail)


# --------------------------------------------------------------------------- GPU: audit and answers
def cpu_ready(result: dict[str, Any], pipeline: str) -> tuple[bool, str]:
    """GPU precondition on ONE arm's ``result.cpu.json``: complete, no futility stop, no gate tripped, M3 passed (PREDECLARED checkpoint 4)."""
    ok = (result.get("complete") is True and result.get("phase") == "cpu" and result.get("pipeline") == PIPELINE_OF[pipeline]
          and not result.get("futility_stop") and not result.get("gate_tripped") and result.get("m3_passed") is True)
    return ok, (f"complete={result.get('complete')} pipeline={result.get('pipeline')} futility={result.get('futility_stop')} "
                f"gates={result.get('gate_tripped')} m3_passed={result.get('m3_passed')}")


class ScopedRec:
    """A view of one ``P.Recorder`` for one store: units are prefixed ``<tag>:`` and every record carries ``store=<tag>``; ``done`` and ``records``
    are filtered to the tag, so ``phase0_arms.run_audit`` (written for one store) runs unchanged on each of two stores in one results file."""

    def __init__(self, rec: P.Recorder, tag: str) -> None:
        self.rec, self.tag, self._pre = rec, tag, f"{tag}:"

    @property
    def done(self) -> set[str]:
        return {u[len(self._pre):] for u in self.rec.done if u.startswith(self._pre)}

    @property
    def records(self) -> list[dict[str, Any]]:
        return [r for r in self.rec.records if r.get("store") == self.tag]

    def emit(self, kind: str, **kw: Any) -> dict[str, Any]:
        if kw.get("unit"):
            kw["unit"] = self._pre + kw["unit"]
        return self.rec.emit(kind, store=self.tag, **kw)


def run_audit(rec: P.Recorder, stores: dict[str, Any], backend: Any, *, guard: Callable[[], None], smoke: bool = False) -> dict[str, Any]:
    """Checkpoint 5a: the L4c audit sample (the AGG-1.1 recipe, unmodified) on each store, ``v1`` then ``v2``."""
    res: dict[str, Any] = {}
    for tag in (V1, V2):
        sr = ScopedRec(rec, tag)
        res[tag] = P.run_audit(sr, stores[tag], backend, guard=guard, smoke=smoke)
        if smoke and res[tag].get("n_queue") == 0 and "audit_done" not in sr.done:
            sr.emit("audit_done", unit="audit_done", n_queue=0, disputed=0)
        if smoke:
            break
    done = all("audit_done" in ScopedRec(rec, t).done for t in (V1, V2))
    return {"complete": done, "stores": res}


def answer_unit(arm: str, qid: str) -> str:
    return f"answer:{arm}:{qid}"


def run_answers(rec: P.Recorder, stores: dict[str, Any], backend: Any, *, agg: list[dict[str, Any]], names: list[str],
                guard: Callable[[], None], smoke: bool = False, cfg: P.Cfg = P.CFG) -> dict[str, Any]:
    """Checkpoint 5b: P-large over the 30 verified questions for arm A (v1 store, agg11.v0), B (v2 store, agg11.v0), C (v2 store, agg110.v1);
    run order A, B, C. Each store's canonical table is built once with ITS pipeline's rule (canonical.v1 / canonical.v2)."""
    from src.aggregation import planner  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    sym = G.orch_symbols()
    ctx: dict[str, Any] = {}
    for tag, store in stores.items():
        spec = sym["pl"].pipeline_spec(PIPELINE_OF[tag])
        chunks = store.chunks(CORPUS_ID)
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(c)
        canon = spec.refresh_canonical(store, CORPUS_ID)
        if f"canonical:{tag}" not in rec.done:
            rec.emit("canonical", unit=f"canonical:{tag}", store=tag, pipeline=spec.as_record(), **canon)
        ctx[tag] = {"chunks": chunks, "bm25": BM25Index([c["text"] for c in chunks]), "docs_meta": store.corpus_docs(CORPUS_ID),
                    "sql_present": bool(store.facts(CORPUS_ID)), "built": sorted(store.built_layers(CORPUS_ID)), "by_doc": by_doc}
    scopes = P._scope_plan(planner, ctx[V1]["docs_meta"], [("P-large", "agg", q) for q in agg])
    plan = [("C", agg[:1])] if smoke else [(arm, agg) for arm in ("A", "B", "C")]
    for arm, qs in plan:
        tag, pv = ARMS[arm]
        c = ctx[tag]
        for q in qs:
            unit = answer_unit(arm, q["id"])
            if unit in rec.done:
                continue
            guard()
            t, t0 = P.Tally(backend), time.time()
            text, st, extra = G110.p_large_answer_v(
                backend, stores[tag], q, scope=scopes[f"answer:P-large:{q['id']}"], planner=planner, docs_meta=c["docs_meta"], built=c["built"],
                sql_present=c["sql_present"], names=names, cfg=cfg, corpus_chunks=c["chunks"], corpus_bm25=c["bm25"],
                doc_chunks=lambda d, _b=c["by_doc"]: _b.get(d, []), planner_version=pv, corpus_id=CORPUS_ID)
            ok = P.score_agg(q, text) if st == "ok" else None
            rec.emit("answer", unit=unit, arm=arm, qid=q["id"], qclass="agg", status=st, correct=ok, calls=t.calls, prefill_tokens=t.prefill,
                     answer=text[-600:], final=st in ("ok", "out_of_window", "budget_exhausted"), wall_s=time.time() - t0,
                     pipeline=PIPELINE_OF[tag], store=tag, **extra)
            if smoke:
                return {"complete": False, "smoke": True}
    return {"complete": all(answer_unit(a, q["id"]) in rec.done for a in ARMS for q in agg)}


# --------------------------------------------------------------------------- score (no model)
def m4_arm(records: list[dict[str, Any]], arm: str, ids: list[str]) -> dict[str, Any]:
    mine = {r["qid"]: r for r in records if r.get("kind") == "answer" and r.get("final") and r.get("arm") == arm}
    m4 = G.m4_of([mine[q] for q in ids if q in mine])
    return {**m4, "complete": all(q in mine for q in ids), "trips": G.m4_trips(m4, all(q in mine for q in ids))}


def answers_of(records: list[dict[str, Any]], arm: str) -> dict[str, bool]:
    return {r["qid"]: r.get("correct") is True for r in records if r.get("kind") == "answer" and r.get("final") and r.get("arm") == arm}


def score(v1: dict[str, Any], v2: dict[str, Any], cells1: dict[str, Any], cells2: dict[str, Any], records: list[dict[str, Any]],
          agg: list[dict[str, Any]], cells: list[tuple[str, str, float]], *, checkpoints: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The mechanical verdict (PREDECLARED *Verdict*): ``agg_1_13_filed`` always, with M1 and P-large for A, B, C, the paired statistics, strata,
    M3 and M4 per arm, the regressions and fixes (listed here, after both are scored); plus ``agg113_futility`` / ``phase1_gate_tripped``. No
    ``adopt`` recommendation is computed from these data alone."""
    keys = cell_keys(cells)
    ids = [q["id"] for q in agg]
    r1, r2 = cells1["raw"], cells2["raw"]
    c1, c2 = cells1["canonical"], cells2["canonical"]
    out: dict[str, Any] = {"verdict": "agg_1_13_filed", "also": [], "n_cells": len(keys), "n_questions": len(ids)}
    # futility, evaluated from the final cell vectors (document-local, so equal to the online pilot / interim looks)
    cp = checkpoints or {}
    fut = {}
    for name in ("pilot", "interim6"):
        docs = (cp.get(name) or {}).get("docs")
        if docs:
            fut[name] = cross_futility(r2, r1, cell_keys(cells, docs))
    out["futility_checks"] = fut
    if any(x["tripped"] for x in fut.values()):
        out["also"].append("agg113_futility")
    out["m1"] = {"raw": {"v1": sum(bool(r1.get(k)) for k in keys), "v2": sum(bool(r2.get(k)) for k in keys), "n": len(keys),
                         "paired_v2_vs_v1": paired_113(r2, r1, keys)},
                 "canonical_secondary": {"v1": sum(bool(c1.get(k)) for k in keys), "v2": sum(bool(c2.get(k)) for k in keys),
                                         "paired_v2_vs_v1": paired_113(c2, c1, keys)},
                 "strata": strata_113({"v1": r1, "v2": r2}, cells),
                 "regressions_v1_found_v2_not": sorted(k for k in keys if r1.get(k) and not r2.get(k)),
                 "fixes_v2_found_v1_not": sorted(k for k in keys if r2.get(k) and not r1.get(k))}
    out["m3"] = {"v1": v1.get("m3"), "v2": v2.get("m3"), "bounds": {"prefill_ratio_max": A.KILL["m3_prefill_ratio_max"],
                                                                      "calls_per_100_chunks_max": A.KILL["m3_calls_per_100_chunks_max"]}}
    trips = [f"M3:{t}" for t, r in ((V1, v1), (V2, v2)) if r.get("m3_passed") is not True]
    corr = {a: answers_of(records, a) for a in ARMS}
    m4 = {a: m4_arm(records, a, ids) for a in ARMS}
    trips += [f"M4:{a}" for a, m in m4.items() if m["trips"]]
    complete = all(m["complete"] for m in m4.values())
    out["p_large"] = {"complete": complete, "arms": {a: {"n_correct": sum(corr[a].get(q, False) for q in ids), "n": len(ids), "m4": m4[a]} for a in ARMS},
                      "total_C_vs_A": paired_113(corr["C"], corr["A"], ids), "pipeline_B_vs_A": paired_113(corr["B"], corr["A"], ids),
                      "planner_C_vs_B": paired_113(corr["C"], corr["B"], ids)}
    out["gates_tripped"] = trips
    if trips:
        out["also"].append("phase1_gate_tripped")
    out["complete"] = complete
    out["adopt_recommendation"] = None
    return out


# --------------------------------------------------------------------------- checks for the CLI
def orch_checks(orch_root: Path, orch_pin: str) -> list[str]:
    out = G.orch_problems(orch_root, orch_pin)
    try:
        from src.aggregation import facts_det, facts_typed, pipeline as _pl  # noqa: PLC0415
        if not hasattr(facts_det, "fy_end_months") or _pl.pipeline_spec(_pl.PIPELINE_V1).l3_kwargs({"x": "2022"}, fy_end=True).get("fy_end") != FY_END_MONTHS:
            out.append("orchestrator lacks the opt-in fy_end rule (facts_det.fy_end_months, PipelineSpec.l3_kwargs(fy_end=)): amendment 1 would not apply")
        if _pl.pipeline_spec(_pl.PIPELINE_V1).l3_kwargs({"x": "2022"}) != {}:
            out.append("orchestrator agg_pipeline.v1 default is no longer the legacy call")
        if not hasattr(facts_typed, "drop_unquoted"):
            out.append("orchestrator lacks facts_typed.drop_unquoted (AGG-1.14 value-in-quote rule): the run would not count")
    except Exception as e:  # noqa: BLE001
        out.append(f"orchestrator facts_typed import failed: {type(e).__name__}: {e}")
    return out


def readme_sha_table() -> dict[str, str]:
    return {"predeclared": PREDECLARED_SHA, **FROZEN}
