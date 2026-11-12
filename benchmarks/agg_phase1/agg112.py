#!/usr/bin/env python3
"""AGG-1.12: adopt ``agg_pipeline.v2`` (typed L4 over union windows + ``l3_period.v2`` + ``canonical.v2`` attributable-net-income)
and re-ingest the 28-document corpus, then re-answer the 58 questions (library; ``run_agg112.py`` is the CLI).

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an injected backend
(``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in tests).

Implements ``PREDECLARED_AGG112.md`` literally. Phase 0 / AGG-1.1 behaviour is the orchestrator's default (``agg_pipeline.v1``,
``agg11.v0``) and is not touched; nothing under ``agg_phase0/`` and none of the pinned Phase 1 files is edited.

Phases (``run_agg112.py --phase``): ``preflight`` (0, zero compute), ``l3`` (1, no model: copy of the AGG-1.1 store, facts wiped,
L3 re-extracted under ``l3_period.v2``), ``pilot`` (2, typed L4 on MICROSOFT / JOHNSON_JOHNSON / 3M, <= 1800 s, futility), ``ingest``
(3-4, the other 25 documents, an interim look after 14, the final look), ``gpu-audit`` (5a, L4c, the AGG-1.1 recipe), ``gpu-answers`` (5b,
P-large on the 58 under ``agg110.v1`` then ``agg11.v0``), ``score``. Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate
tripped (``phase1_gate_tripped``), 4 precondition, 5 budget stop (re-book, not a gate), 75 partial (resumable).

Every record is appended with flush + fsync as soon as it exists (``phase0_arms.Recorder``); a re-run skips every ``unit`` already
recorded. One window call is one unit (``tw:<doc>:<window id>``), so a kill loses at most one call.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
P0DIR = HERE.parent / "agg_phase0"
for _p in (str(P0DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg110 as G110  # noqa: E402
import agg16 as G16  # noqa: E402
import agg19 as G19  # noqa: E402
import agg_phase0 as A  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402

# --------------------------------------------------------------------------- pins
PREDECLARED_FILE = "PREDECLARED_AGG112.md"
PREDECLARED_SHA = "22479709afbe5110115f6b2d1a951e799b8bb8752fabc5c51681c4aa30f59c1c"
REFERENCE_FILE = "agg112_reference.json"
REFERENCE_SHA = "f23bcbb33bb7d71718d4e0d68ce57aad8813226a5668500b5a8ff1104ead36ae"
SHA256 = X.sha256_file
PIPELINE = "agg_pipeline.v2"
CORPUS_ID = X.CORPUS_ID

# --------------------------------------------------------------------------- the design (PREDECLARED_AGG112.md)
PILOT_DOCS = X.PILOT_DOCS
PILOT_WALL_S = 1800
INTERIM_DOCS = 14                     # futility-only look after 14 of the 28 documents
BOOKED_L4_S = 9000.0                  # ESTIMATE: 903 windows x 9.68 s (AGG-1.6 arm T, measured) + server load + L3
BUDGET_X = 2.0                        # projected L4 wall > 2x booked -> budget stop (re-book, not a gate)
HARD_STOP_X = 1.5                     # fresh wall > 1.5x the pilot projection -> budget stop
RATE_S_PER_CALL = 1946.074385881424 / 201     # MEASURED, AGG-1.6 r2 arm T: 201 calls in 1946.1 s
PREFILL_PER_CALL, PREFILL_PER_CHAR = 338.1, 0.2226   # MEASURED fit over the 154 AGG-1.6 T windows on the AGG-1.1 store
SERVER_START_S = 120.0                # ESTIMATE (AGG-1.6)
L3_ESTIMATE_S = 60.0                  # measured 10 s for the extraction on one core, plus the store copy
M1_PILOT = "pilot M1 below AGG-1.1 on the same cells"
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_PARTIAL = (X.RC_OK, X.RC_PREFLIGHT, X.RC_GATE, X.RC_PRECOND, X.RC_BUDGET,
                                                                   X.RC_PARTIAL)
M4_CALLS_SOFT, M4_PREFILL_MAX = X.M4_MEDIAN_CALLS_SOFT, X.M4_MEDIAN_PREFILL_MAX
PLANNERS = (G110.V1, G110.V0)         # the answer passes, in order: agg110.v1 (primary) then agg11.v0
N_QUESTIONS = 58
N_CELLS = 211
AGG11_P_LARGE = {"correct": 18, "n": 58, "accuracy": 18 / 58}


# --------------------------------------------------------------------------- inputs and pins
def load_inputs(set_dir: Path = HERE) -> dict[str, Any]:
    inp = X.load_inputs(corpus=set_dir / "corpus.json", golds=set_dir / "golds.jsonl",
                        verification=set_dir / "verification_record.json", facts_sheet=set_dir / "facts_sheet.csv")
    corpus = inp["corpus"]
    inp["cells"] = X.m1_cells(set_dir / "facts_sheet.csv", X.verified_agg(inp), [d["doc_name"] for d in corpus["docs"]])
    inp["reference"] = json.loads((set_dir / REFERENCE_FILE).read_text()) if (set_dir / REFERENCE_FILE).exists() else None
    inp["sizes"] = X.load_sizes(set_dir / "extracted_sizes.json")
    inp["set_dir"] = set_dir
    return inp


def pin_problems(set_dir: Path = HERE, *, predeclared_sha: str | None = None, reference_sha: str | None = None) -> list[str]:
    out = []
    for name, want in ((PREDECLARED_FILE, PREDECLARED_SHA if predeclared_sha is None else predeclared_sha),
                       (REFERENCE_FILE, REFERENCE_SHA if reference_sha is None else reference_sha)):
        p = set_dir / name
        got = SHA256(p) if p.exists() else None
        if got != want:
            out.append(f"{name} sha256 {got} != recorded {want}")
    return out


def fiscal_years(corpus: dict[str, Any]) -> dict[str, str]:
    """doc_id -> the document's own fiscal year (the manifest's; every AGG-1.1 filing is fiscal 2022)."""
    return {d["doc_name"]: X.doc_fiscal_year(d, corpus) for d in corpus["docs"]}


def descs_for(fy: str) -> dict[str, str]:
    """One description per typed slot, rendered for the document's fiscal year (AGG-1.3 amendment 2; FY2022 is the identity)."""
    return {s: G16.field_desc(s, fy) for s in orch_symbols()["ft"].SLOTS}


def orch_symbols() -> dict[str, Any]:
    """The pinned orchestrator's pipeline modules (the caller put ``--orch-root`` on ``sys.path`` and the layer flag on)."""
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    from src.aggregation import facts_typed as ft, pipeline as pl  # noqa: PLC0415
    return {"ft": ft, "pl": pl}


def doc_order(corpus: dict[str, Any], sizes: dict[str, int]) -> list[str]:
    """Pilot documents first (declared order), then the others by ascending extracted size (ties by name)."""
    rest = sorted((d["doc_name"] for d in corpus["docs"] if d["doc_name"] not in PILOT_DOCS),
                  key=lambda n: (sizes[n], n))
    return list(PILOT_DOCS) + rest


# --------------------------------------------------------------------------- the offline plan (zero model)
def doc_plans(store: Any, corpus: dict[str, Any], corpus_id: str = CORPUS_ID) -> dict[str, dict[str, Any]]:
    """Per document: the distinct windows to call (one typed call each), which fields chose each, and the characters."""
    ft = orch_symbols()["ft"]
    fy = fiscal_years(corpus)
    wins: dict[str, list[dict[str, Any]]] = {}
    for w in ft.split_windows(store, corpus_id, P.CFG.window_chars):
        wins.setdefault(w["doc_id"], []).append(w)
    out = {}
    for d in corpus["docs"]:
        name = d["doc_name"]
        windows, chosen = ft.doc_call_plan(wins.get(name, []), ft.SCHEMA_FIELDS, descs_for(fy[name]))
        out[name] = {"windows": windows, "chosen": {k: sorted(v) for k, v in chosen.items()}, "n_windows": len(windows),
                     "chars": sum(len(w["text"]) for w in windows), "n_doc_windows": len(wins.get(name, []))}
    return out


def plan_totals(plans: dict[str, dict[str, Any]], docs: Iterable[str] | None = None) -> dict[str, Any]:
    names = list(plans) if docs is None else list(docs)
    calls = sum(plans[n]["n_windows"] for n in names)
    chars = sum(plans[n]["chars"] for n in names)
    return {"docs": len(names), "calls": calls, "chars": chars, "prefill_est": round(calls * PREFILL_PER_CALL + chars * PREFILL_PER_CHAR),
            "wall_est_s": round(calls * RATE_S_PER_CALL, 1)}


def plan_m3(plans: dict[str, dict[str, Any]], stats: dict[str, dict[str, int]]) -> dict[str, Any]:
    """The projected M3 of the whole ingest from the offline plan, against the Phase 0 kill bounds and the AGG-1.1 figures. The
    call count is exact (every union window is called, no admission gate); the prefill uses the measured per-call / per-character fit."""
    tot = plan_totals(plans)
    tokens = sum(v["tokens"] for v in stats.values())
    chunks = sum(v["chunks"] for v in stats.values())
    m3 = X.m3_values(tot["prefill_est"], tot["calls"], tokens, chunks)
    return {"label": "ESTIMATE (calls exact; prefill from the AGG-1.6 per-call / per-character fit)", **tot, "m3": m3,
            "calls_per_unit": tot["calls"] / (len(plans) * len(A.METRICS)),
            "inside_kill_bounds": not X.m3_trips(m3),
            "bounds": {"prefill_ratio_max": A.KILL["m3_prefill_ratio_max"],
                       "calls_per_100_chunks_max": A.KILL["m3_calls_per_100_chunks_max"]},
            "agg11": {"prefill_ratio": 0.3783642241455476, "calls_per_100_chunks": 22.30062356416147, "llm_calls": 1359,
                      "prefill_tokens": 1712985, "calls_per_unit": 1359 / 224}}


def estimates(plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tot = plan_totals(plans)
    cpu = tot["wall_est_s"] + SERVER_START_S + L3_ESTIMATE_S
    gpu_answers = 804.0 + 900.0       # measured P-large over the 58 (AGG-1.1, 804 s) plus agg110.v1 (about the same plus re-asks)
    gpu = gpu_answers + 250.0 + 200.0 # + L4c audit (queue about 8% of the kept facts) + two cold 27B server loads
    return {"label": "ESTIMATE", "rates": {"s_per_call": round(RATE_S_PER_CALL, 3), "source": "AGG-1.6 r2 arm T: 201 calls / 1946.1 s"},
            "cpu_s": round(cpu), "cpu_booked_s": BOOKED_L4_S, "gpu_s": round(gpu),
            "gpu_parts": {"p_large_two_planners_s": gpu_answers, "audit_s": 250.0, "server_loads_s": 200.0}}


# --------------------------------------------------------------------------- parity: the pipeline stage IS the measured arm T
def parity_problems(store: Any, corpus: dict[str, Any], docs: tuple[str, ...] = ("MICROSOFT_2022_10K",)) -> list[str]:
    """Fail-closed zero-compute check that the orchestrator stage equals the AGG-1.6 arm T code the A/B measured: same field queries,
    grammar, prompt and union windows."""
    ft = orch_symbols()["ft"]
    from agg111_triage import FIELD_QUERY  # noqa: PLC0415
    out = []
    if dict(ft.FIELD_QUERY) != dict(FIELD_QUERY):
        out.append("facts_typed.FIELD_QUERY != agg111_triage.FIELD_QUERY")
    if tuple(ft.SLOTS) != tuple(G16.T_SLOTS):
        out.append("facts_typed.SLOTS != agg16.T_SLOTS")
    if ft.typed_grammar() != G16.t_grammar():
        out.append("facts_typed.typed_grammar() != agg16.t_grammar()")
    if (ft.N_PREDICT, ft.MAX_ROWS, ft.QUOTE_CHARS, ft.PERIOD_CHARS, ft.K_FIELD, ft.K_CURRENT, ft.WINDOW_CHARS) != (
            G16.N_PREDICT_T, G16.T_MAX_ROWS, G16.T_QUOTE_CHARS, G16.T_PERIOD_CHARS, G16.K_FIELD, P.CFG.k_per_doc, P.CFG.window_chars):
        out.append("facts_typed design constants != agg16 / Phase 0 constants")
    fy = fiscal_years(corpus)
    wins16 = G16.doc_windows(store, CORPUS_ID, P.CFG.window_chars)
    wins_ft: dict[str, list[dict[str, Any]]] = {}
    for w in ft.split_windows(store, CORPUS_ID, P.CFG.window_chars):
        wins_ft.setdefault(w["doc_id"], []).append(w)
    for d in docs:
        if [w["chunk_id"] for w in wins_ft.get(d, [])] != [w["chunk_id"] for w in wins16.get(d, [])]:
            out.append(f"{d}: facts_typed.split_windows != phase0_arms.windows")
            continue
        sample = (wins16[d][0] if wins16.get(d) else {"text": "x"})["text"]
        if ft.typed_prompt(sample, descs_for(fy[d])) != G16.t_prompt(sample, fy[d]):
            out.append(f"{d}: facts_typed.typed_prompt != agg16.t_prompt")
        for f in A.METRICS:
            a = [w["chunk_id"] for w in G16.union_candidates(wins16[d], f, fy[d])]
            b = [w["chunk_id"] for w in ft.union_candidates(wins_ft[d], f, G16.field_desc(f, fy[d]))]
            if a != b:
                out.append(f"{d}/{f}: facts_typed.union_candidates != agg16.union_candidates")
    return out


def orch_problems(orch_root: Path, orch_pin: str = "") -> list[str]:
    out = []
    f = orch_root / ".pin"
    if orch_pin and f.exists() and f.read_text().strip() != orch_pin:
        out.append(f"{f} = {f.read_text().strip()} != expected orchestrator pin {orch_pin}")
    try:
        sym = orch_symbols()
        spec = sym["pl"].pipeline_spec(PIPELINE)
        assert spec.l3_period_version and spec.canonical_rule.startswith("canonical.v2") and spec.l4_stage == "l4_typed.v1"
        assert sym["pl"].pipeline_spec(sym["pl"].PIPELINE_V1).l3_kwargs({"x": "1"}) == {}
        from src.aggregation import facts_llm, planner, sql_grammar  # noqa: PLC0415
        assert hasattr(planner, "sql_shape") and sql_grammar.PLANNER_V1 == G110.V1 and sql_grammar.PLANNER_V0 == G110.V0
    except Exception as e:  # noqa: BLE001
        return out + [f"orchestrator tree {orch_root} lacks the agg_pipeline.v2 stage: {type(e).__name__}: {e}"]
    got = SHA256(Path(facts_llm.__file__))
    if got != G16.FACTS_LLM_SHA:
        out.append(f"facts_llm.py sha256 {got} != the amended v1 stage {G16.FACTS_LLM_SHA}")
    return out


# --------------------------------------------------------------------------- M1 readings and regressions (pure over rows)
def cell_key(d: str, f: str) -> str:
    return f"{d}|{f}"


def readings(rows: list[dict[str, Any]], canon: list[dict[str, Any]], cells: list[tuple[str, str, float]]) -> dict[str, dict[str, bool]]:
    return {"raw": G19.raw_hits(rows, cells), "canonical": G19.canon_ok(canon, cells)}


def regressions(ref: dict[str, Any], now: dict[str, dict[str, bool]], held_out: set[str]) -> dict[str, Any]:
    """A cell that was right in AGG-1.1 and is not right now, on either measure (raw M1, canonical): the AGG-1.9 rule. Broken and
    fixed cells are listed with their stratum (held-out = never named in a design artifact; touched = named)."""
    ref_raw, ref_can = set(ref["agg11"]["raw_hit_cells"]), set(ref["agg11"]["canonical_v1_ok_cells"])
    broken_raw = sorted(k for k in ref_raw if not now["raw"].get(k))
    broken_can = sorted(k for k in ref_can if not now["canonical"].get(k))
    fixed_raw = sorted(k for k, v in now["raw"].items() if v and k not in ref_raw)
    fixed_can = sorted(k for k, v in now["canonical"].items() if v and k not in ref_can)

    def split(keys: list[str]) -> dict[str, list[str]]:
        return {"held_out": [k for k in keys if k in held_out], "touched": [k for k in keys if k not in held_out]}

    return {"ok": not broken_raw and not broken_can, "raw_broken": broken_raw, "canonical_broken": broken_can,
            "raw_fixed": fixed_raw, "canonical_fixed": fixed_can, "n_retained_raw": len(ref_raw), "n_retained_canonical": len(ref_can),
            "raw_broken_by_stratum": split(broken_raw), "canonical_broken_by_stratum": split(broken_can),
            "raw_fixed_by_stratum": split(fixed_raw), "canonical_fixed_by_stratum": split(fixed_can)}


def strata_report(ref: dict[str, Any], now: dict[str, dict[str, bool]], held_out: set[str]) -> dict[str, Any]:
    """M1 (raw and canonical) per stratum, with the AGG-1.1 figure beside it. Labels are part of the numbers."""
    keys = sorted(now["raw"])
    out: dict[str, Any] = {}
    for name, sel in (("held_out", [k for k in keys if k in held_out]), ("touched", [k for k in keys if k not in held_out]),
                      ("all", keys)):
        r_raw, r_can = set(ref["agg11"]["raw_hit_cells"]), set(ref["agg11"]["canonical_v1_ok_cells"])
        out[name] = {"n": len(sel), "raw": {"found": sum(now["raw"][k] for k in sel), "agg11": sum(k in r_raw for k in sel)},
                     "canonical": {"ok": sum(now["canonical"][k] for k in sel), "agg11_v1": sum(k in r_can for k in sel)},
                     "label": {"held_out": "HELD-OUT (never named in a design artifact): a regression test only, AGG-1.1 was already "
                                           "100% on these, so no gain is possible",
                               "touched": "IN-SAMPLE (named in the miss tables / triage the design came from)",
                               "all": "IN-SAMPLE for any gain (the design saw every cell in aggregate)"}[name]}
    return out


def agg19_report(set_dir: Path, now: dict[str, dict[str, bool]]) -> dict[str, Any]:
    """The 26 AGG-1.9 cells (never extracted correctly under AGG-1.1) under the v2 pipeline: filled on raw M1 and on the canonical
    reading, with the AGG-1.8 state each had. IN-SAMPLE (they are the cells the design came from)."""
    import csv  # noqa: PLC0415
    rows = []
    with (set_dir / "agg19_cells.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            k = cell_key(r["doc_name"], r["metric"])
            rows.append({"cell": k, "state_agg11": r["agg18_state"], "raw_found": bool(now["raw"].get(k)),
                         "canonical_ok": bool(now["canonical"].get(k))})
    return {"n": len(rows), "raw_filled": sum(r["raw_found"] for r in rows), "canonical_ok": sum(r["canonical_ok"] for r in rows),
            "unfilled": [r["cell"] for r in rows if not r["raw_found"]], "cells": rows,
            "label": "IN-SAMPLE: the 26 cells the AGG-1.8 / AGG-1.11 / AGG-1.6 designs came from"}


def paired(new: dict[str, bool], old: dict[str, bool], keys: list[str]) -> dict[str, Any]:
    """b = only new right, c = only old right; z = (b - c) / sqrt(b + c)."""
    b = sum(1 for k in keys if new.get(k) and not old.get(k))
    c = sum(1 for k in keys if old.get(k) and not new.get(k))
    both = sum(1 for k in keys if new.get(k) and old.get(k))
    d = b + c
    return {"n": len(keys), "both": both, "only_new": b, "only_old": c, "neither": len(keys) - both - d,
            "z": ((b - c) / d ** 0.5) if d else 0.0}


# --------------------------------------------------------------------------- store wipe and L3 (CPU, no model)
def _wipe_facts(store: Any) -> None:
    """Remove every fact and everything derived from facts from a COPY of the AGG-1.1 store; the parse (documents, chunks, tables,
    pages) and the L1 index rows are kept. Never call this on a seed."""
    c = store.conn
    for t in ("facts", "facts_canonical", "fact_conflicts", "facts_superseded", "documents_superseded"):
        c.execute(f"DELETE FROM {t}")
    c.execute("DELETE FROM index_manifest WHERE layer IN ('L3', 'L4')")
    c.commit()


def run_l3(rec: P.Recorder, out: Path, inp: dict[str, Any], *, seed_store: Path, guard: Callable[[], None], smoke: bool = False,
           corpus_id: str = CORPUS_ID) -> dict[str, Any]:
    """Checkpoint 1: copy the AGG-1.1 store, wipe its facts, re-extract L3 under ``l3_period.v2``. No model, no PDF. The document
    parse is not repeated: it does not depend on the pipeline version."""
    sym = orch_symbols()
    spec = sym["pl"].pipeline_spec(PIPELINE)
    store_path = out / "store.sqlite"
    if not store_path.exists():
        if not seed_store.exists():
            rec.emit("precondition_failed", reason=f"no seed store {seed_store}")
            return {"complete": False, "rc": RC_PRECOND}
        P.copy_store(seed_store, store_path)
        rec.emit("store_copied", src=str(seed_store), sha256=SHA256(store_path), final=False)
    store = P.open_store(store_path)
    try:
        if "l3v2" not in rec.done:
            guard()
            t0 = time.time()
            n_before = len(store.facts(corpus_id))
            _wipe_facts(store)
            facts = spec.extract_l3(store, corpus_id, fiscal_years(inp["corpus"]))
            store.write_facts(facts)
            h = store.corpus_content_hash(corpus_id)
            store.record_index(corpus_id, h, "l3_facts_det", "", layer="L3", n_items=len(facts))
            rows = G19.fact_rows(store.conn, corpus_id)
            raw = G19.raw_hits(rows, inp["cells"])
            rec.emit("l3v2", unit="l3v2", pipeline=spec.as_record(), n_facts_before_wipe=n_before, n_l3_facts=len(facts),
                     n_period_none=sum(1 for f in facts if f.get("period") is None), m1_l3_only_raw=sum(raw.values()),
                     n_cells=len(raw), llm_calls=0, prefill_tokens=0, wall_s=time.time() - t0, final=True)
        if smoke:
            return {"complete": False, "smoke": True, "rc": RC_OK}
        if "corpus" not in rec.done:
            rec.emit("corpus", unit="corpus", n_docs=len(store.corpus_docs(corpus_id)), corpus_tokens=store.corpus_tokens(corpus_id),
                     n_chunks=len(store.chunks(corpus_id)), built_layers=sorted(store.built_layers(corpus_id)),
                     corpus_content_hash=store.corpus_content_hash(corpus_id), llm_calls=0, prefill_tokens=0)
        return {"complete": True, "phase": "l3", "model_calls": 0, "rc": RC_OK}
    finally:
        store.close()


# --------------------------------------------------------------------------- L4: typed windows, one unit per call
def window_unit(doc: str, chunk_id: str) -> str:
    return f"tw:{doc}:{chunk_id}"


def full_fact(f: dict[str, Any]) -> dict[str, Any]:
    """A fact as the record keeps it: every store column plus the declared-scale fields ``reconcile`` reads (JSON-safe)."""
    from src.aggregation.store import FACT_COLUMNS  # noqa: PLC0415
    keys = tuple(FACT_COLUMNS) + ("declared_scale", "declared_scale_basis")
    return {k: f.get(k) for k in keys}


def doc_windows_done(rec: P.Recorder, doc: str, plan: dict[str, Any]) -> bool:
    return all(window_unit(doc, w["chunk_id"]) in rec.done for w in plan["windows"])


def windows_records(rec: P.Recorder) -> dict[str, dict[str, Any]]:
    return {r["unit"]: r for r in rec.records if r.get("kind") == "t_window" and r.get("final")}


def run_window(rec: P.Recorder, tb: Any, rb: Any, doc: str, w: dict[str, Any], chosen: list[str], descs: dict[str, str],
               guard: Callable[[], None]) -> dict[str, Any]:
    """ONE typed call for one window; the record carries every slot's facts (gated at the document's finish)."""
    unit = window_unit(doc, w["chunk_id"])
    guard()
    t0 = time.time()
    mark = rb.mark()
    status, res = "ok", None
    try:
        res = tb.call_window(w, descs)
    except Exception as e:  # noqa: BLE001 - recorded per window, the run continues
        status = f"error:{type(e).__name__}:{str(e)[:200]}"
    cs = G16.calls_stats(rb.since(mark))
    bad = bool(res.bad) if res is not None else False
    return rec.emit("t_window", unit=unit, doc=doc, chunk_id=w["chunk_id"], status=status, ok=status == "ok",
                    parsed=bool(res and res.parsed), bad=bad and status == "ok", chosen_by=list(chosen),
                    chars=len(w["text"]), **cs, wall_s=time.time() - t0, n_facts=len(res.facts) if res else 0,
                    facts=[full_fact(f) for f in res.facts] if res else [], final=status == "ok")


def finish_doc(rec: P.Recorder, tb: Any, store: Any, doc: str, plan: dict[str, Any], corpus_id: str = CORPUS_ID) -> dict[str, Any]:
    """L4d for one document from its window records: gate each window's rows to the fields that chose it, reconcile, write."""
    ft = orch_symbols()["ft"]
    byw = windows_records(rec)
    kept_in: list[dict[str, Any]] = []
    recs = []
    for w in plan["windows"]:
        r = byw[window_unit(doc, w["chunk_id"])]
        recs.append(r)
        chosen = set(plan["chosen"][w["chunk_id"]])
        kept_in.extend(f for f in r["facts"] if ft.keep_slot(f, chosen))
    out = tb.finish_doc(corpus_id, kept_in, ft.SCHEMA_FIELDS)
    bad = sum(1 for r in recs if r.get("bad"))
    return rec.emit("doc_l4", unit=f"doc_l4:{doc}", doc=doc, n_windows=len(recs), llm_calls=sum(r["calls"] for r in recs),
                    prefill_tokens=sum(r["prefill_tokens"] for r in recs), cache_n=sum(r["cache_n"] for r in recs),
                    decode_tokens=sum(r["decode_tokens"] for r in recs), wall_s=sum(r["wall_s"] for r in recs),
                    bad_replies=bad, cut=sum(1 for r in recs if r.get("cut")), unparseable=sum(1 for r in recs if not r.get("parsed")),
                    n_facts_in=len(kept_in), n_facts_kept=len(out["facts"]), n_conflicts=len(out["conflicts"]),
                    n_quote_dropped=len(out.get("quote_dropped", [])),    # AGG-1.14 value-in-quote rule (absent before it)
                    n_unit_overrides=len(out["unit_overrides"]), unit_overrides=out["unit_overrides"][:20],
                    by_field={f: sum(1 for k in out["facts"] if k.get("field") == f) for f in ft.SCHEMA_FIELDS},
                    chars=plan["chars"], final=True)


def done_docs(rec: P.Recorder, plans: dict[str, dict[str, Any]]) -> list[str]:
    return [d for d in plans if f"doc_l4:{d}" in rec.done]


def doc_records(rec: P.Recorder) -> dict[str, dict[str, Any]]:
    return {r["doc"]: r for r in rec.records if r.get("kind") == "doc_l4"}


def tally_docs(recs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    recs = list(recs)
    return {"n_docs": len(recs), "llm_calls": sum(r["llm_calls"] for r in recs), "prefill_tokens": sum(r["prefill_tokens"] for r in recs),
            "wall_s": sum(r["wall_s"] for r in recs), "chars": sum(r["chars"] for r in recs),
            "bad_replies": sum(r["bad_replies"] for r in recs), "cache_n": sum(r.get("cache_n", 0) for r in recs)}


def ref_found(ref: dict[str, Any], docs: Iterable[str]) -> int:
    docs = set(docs)
    return sum(1 for k in ref["agg11"]["raw_hit_cells"] if k.split("|")[0] in docs)


def cells_of(inp: dict[str, Any], docs: Iterable[str]) -> list[str]:
    docs = set(docs)
    return [cell_key(d, f) for d, f, _g in inp["cells"] if d in docs]


def pilot_eval(*, rec: P.Recorder, plans: dict[str, dict[str, Any]], stats: dict[str, dict[str, int]], raw: dict[str, bool],
               inp: dict[str, Any], booked_s: float = BOOKED_L4_S, pilot_wall_s: float = PILOT_WALL_S) -> dict[str, Any]:
    """Checkpoint 2, pure arithmetic over the pilot records. Futility: projected M3 over a kill bound, or the pilot M1 (raw, the pilot
    documents' cells) below AGG-1.1's on the SAME cells. Budget (re-book, not a gate): pilot wall > 1800 s, or the projected L4 wall
    > 2x the booked estimate."""
    ref = inp["reference"]
    drs = doc_records(rec)
    pilot = tally_docs(drs[d] for d in PILOT_DOCS if d in drs)
    pcells = cells_of(inp, PILOT_DOCS)
    found = sum(1 for k in pcells if raw.get(k))
    ref_n = ref_found(ref, PILOT_DOCS)
    tot = plan_totals(plans)
    pil_plan = plan_totals(plans, PILOT_DOCS)
    per_call_wall = pilot["wall_s"] / pilot["llm_calls"] if pilot["llm_calls"] else 0.0
    per_char_prefill = pilot["prefill_tokens"] / pilot["chars"] if pilot["chars"] else 0.0
    proj_calls = tot["calls"]
    proj_prefill = per_char_prefill * tot["chars"]
    tokens, chunks = (sum(v["tokens"] for v in stats.values()), sum(v["chunks"] for v in stats.values()))
    proj_m3 = X.m3_values(proj_prefill, proj_calls, tokens, chunks)
    proj_wall = per_call_wall * proj_calls
    gates, budget = [], []
    if X.m3_trips(proj_m3):
        gates.append("M3")
    if found < ref_n:
        gates.append("M1_pilot_below_agg11")
    if pilot["wall_s"] > pilot_wall_s:
        budget.append("pilot_wall_over_1800s")
    if proj_wall > BUDGET_X * booked_s:
        budget.append("projected_l4_wall_over_2x_booked")
    return {"pilot": pilot, "pilot_plan": pil_plan, "pilot_m1": {"cells": len(pcells), "found": found, "agg11_found": ref_n,
                                                                  "m1": found / len(pcells) if pcells else None},
            "per_call_wall_s": per_call_wall, "projected_calls": proj_calls, "projected_prefill": proj_prefill, "projected_m3": proj_m3,
            "projected_l4_wall_s": proj_wall, "booked_l4_s": booked_s, "hard_stop_s": HARD_STOP_X * proj_wall,
            "gates_tripped": gates, "budget_stops": budget,
            "bad_reply_rate": (pilot["bad_replies"] / pilot["llm_calls"]) if pilot["llm_calls"] else None}


def _write_cpu_result(out: Path, **res: Any) -> dict[str, Any]:
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = PREDECLARED_SHA
    res["pipeline"] = PIPELINE
    X._atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    return res


def _gate_stop(rec: P.Recorder, out: Path, gates: list[str], where: str, **detail: Any) -> dict[str, Any]:
    rec.emit("gate_tripped", unit=f"gate_tripped:{where}", gates=gates, where=where, **detail)
    rec.emit("stop", reason="phase1_gate_tripped", gates=gates, where=where, final=False)
    res = _write_cpu_result(out, complete=True, futility_stop="+".join(gates), gate_tripped=gates, verdict="phase1_gate_tripped",
                            where=where, m3_passed=False if "M3" in gates else None, rc=RC_GATE, **detail)
    res["rc"] = RC_GATE
    return res


def _budget_stop(rec: P.Recorder, reason: str, **detail: Any) -> dict[str, Any]:
    rec.emit("stop", reason=f"budget_stop:{reason}", final=False, **detail)
    return {"complete": False, "budget_stop": reason, "rc": RC_BUDGET, **detail}


def interim_look(rec: P.Recorder, out: Path, store: Any, inp: dict[str, Any], plans: dict[str, dict[str, Any]],
                 stats: dict[str, dict[str, int]]) -> dict[str, Any] | None:
    """Checkpoint 3, futility only (never an early GO): once INTERIM_DOCS documents are complete, running M3 over a kill bound, or raw
    M1 on those documents' cells below AGG-1.1's on the same cells."""
    done = done_docs(rec, plans)
    if len(done) < INTERIM_DOCS or "interim14" in rec.done:
        return None
    raw = G19.raw_hits(G19.fact_rows(store.conn, CORPUS_ID), inp["cells"])
    cells = cells_of(inp, done)
    found, ref_n = sum(1 for k in cells if raw.get(k)), ref_found(inp["reference"], done)
    drs = doc_records(rec)
    t = tally_docs(drs[d] for d in done)
    tok, chunks = X.sum_stats(stats, done)
    m3 = X.m3_values(t["prefill_tokens"], t["llm_calls"], tok, chunks)
    gates = (["M1_interim_below_agg11"] if found < ref_n else []) + (["M3"] if X.m3_trips(m3) else [])
    rec.emit("interim14", unit="interim14", n_docs_done=len(done), found=found, agg11_found=ref_n, cells=len(cells), running_m3=m3,
             gates_tripped=gates)
    return _gate_stop(rec, out, gates, "interim", running_m3=m3, found=found, agg11_found=ref_n) if gates else None


def run_ingest(rec: P.Recorder, out: Path, inp: dict[str, Any], backend: Any, *, stage: str, guard: Callable[[], None],
               smoke: bool = False, corpus_id: str = CORPUS_ID, booked_l4_s: float | None = None,
               pilot_wall_s: float = PILOT_WALL_S) -> dict[str, Any]:
    """Checkpoints 2-4. ``stage`` = ``pilot`` (the 3 pilot documents, then the pilot evaluation) or ``ingest`` (the other 25
    documents, interim look, final look). Smoke: ONE real typed call on the smallest document, then return."""
    sym = orch_symbols()
    ft = sym["ft"]
    store_path = out / "store.sqlite"
    if "l3v2" not in rec.done or not store_path.exists():
        rec.emit("precondition_failed", reason="run --phase l3 first (no l3v2 record / store)")
        return {"complete": False, "rc": RC_PRECOND}
    booked = booked_l4_s or BOOKED_L4_S
    store = P.open_store(store_path)
    try:
        corpus = inp["corpus"]
        plans = doc_plans(store, corpus, corpus_id)
        stats = X.doc_stats(store, corpus_id)
        order = doc_order(corpus, inp["sizes"])
        fy = fiscal_years(corpus)
        state = {r["unit"]: r for r in rec.records if r.get("unit") in ("pilot_done", "interim14")}
        if stage == "ingest" and "pilot_done" not in state and not smoke:
            rec.emit("precondition_failed", reason="the pilot has not completed (run --phase pilot first)")
            return {"complete": False, "rc": RC_PRECOND}
        if "pilot_done" in state and not smoke:
            pd = state["pilot_done"]
            if pd.get("gates_tripped"):
                return {"complete": True, "phase": stage, "gate_tripped": pd["gates_tripped"], "futility_stop": "+".join(pd["gates_tripped"]),
                        "rc": RC_GATE}
            if stage == "pilot":
                if pd.get("budget_stops"):
                    return {"complete": False, "budget_stop": "+".join(pd["budget_stops"]), "rc": RC_BUDGET}
                return {"complete": True, "phase": "pilot", "already_done": True, "rc": RC_OK}
        backend_rb = G16.RecordingBackend(backend)
        tb = ft.TypedBackfill(store, backend_rb)
        todo = list(PILOT_DOCS) if stage == "pilot" else [d for d in order if d not in PILOT_DOCS]
        if smoke:
            todo = [min(plans, key=lambda d: (inp["sizes"][d], d))]
        if stage == "ingest" and not smoke:
            early = interim_look(rec, out, store, inp, plans, stats)
            if early:
                return early
        for doc in todo:
            plan = plans[doc]
            descs = descs_for(fy[doc])
            for w in plan["windows"]:
                unit = window_unit(doc, w["chunk_id"])
                if unit in rec.done:
                    continue
                if stage == "pilot" and not smoke:
                    used = sum(r["wall_s"] for r in rec.records if r.get("kind") == "t_window" and r.get("doc") in PILOT_DOCS and r.get("final"))
                    if used > pilot_wall_s:
                        rec.emit("pilot_bound_hit", unit="pilot_bound_hit", wall_s=used, bound_s=pilot_wall_s)
                        return _budget_stop(rec, "pilot_wall_over_1800s", wall_s=used)
                run_window(rec, tb, backend_rb, doc, w, plan["chosen"][w["chunk_id"]], descs, guard)
                if smoke:
                    return {"complete": False, "smoke": True, "rc": RC_OK}
                if stage == "ingest":
                    st = _ingest_budget(rec, state)
                    if st:
                        return st
            if not doc_windows_done(rec, doc, plan):
                return {"complete": False, "reason": f"{doc}: windows errored; re-grant retries them", "rc": RC_PARTIAL}
            if f"doc_l4:{doc}" not in rec.done:
                finish_doc(rec, tb, store, doc, plan, corpus_id)
            if stage == "ingest":
                early = interim_look(rec, out, store, inp, plans, stats)
                if early:
                    return early
        if stage == "pilot":
            rows = G19.fact_rows(store.conn, corpus_id)
            raw = G19.raw_hits(rows, inp["cells"])
            ev = pilot_eval(rec=rec, plans=plans, stats=stats, raw=raw, inp=inp, booked_s=booked, pilot_wall_s=pilot_wall_s)
            rec.emit("pilot_done", unit="pilot_done", **ev)
            X._atomic_write(out / "result.pilot.json", json.dumps({"phase": "pilot", "complete": True, **ev,
                                                                    "predeclared_sha256": PREDECLARED_SHA}, indent=1, default=str) + "\n")
            if ev["gates_tripped"]:
                return _gate_stop(rec, out, ev["gates_tripped"], "pilot", m1=ev["pilot_m1"]["m1"], pilot=ev)
            if ev["budget_stops"]:
                return _budget_stop(rec, "+".join(ev["budget_stops"]), pilot=ev)
            return {"complete": True, "phase": "pilot", "rc": RC_OK, **ev}
        return finish_ingest(rec, out, store, inp, plans, stats, corpus_id=corpus_id)
    finally:
        store.close()


def _ingest_budget(rec: P.Recorder, state: dict[str, Any]) -> dict[str, Any] | None:
    pd = state.get("pilot_done")
    if not pd:
        return None
    fresh = sum(r["wall_s"] for r in rec.records if r.get("kind") == "t_window" and r.get("final") and r.get("doc") not in PILOT_DOCS)
    if 0 < pd["hard_stop_s"] < fresh:
        return _budget_stop(rec, "l4_wall_over_1.5x_pilot_projection", wall_s=fresh, hard_stop_s=pd["hard_stop_s"])
    return None


def finish_ingest(rec: P.Recorder, out: Path, store: Any, inp: dict[str, Any], plans: dict[str, dict[str, Any]],
                  stats: dict[str, dict[str, int]], *, corpus_id: str = CORPUS_ID) -> dict[str, Any]:
    """Checkpoint 4: M1 (raw and canonical), M3, regressions and strata; the gates M3 and M1 >= AGG-1.1; ``result.cpu.json``."""
    sym = orch_symbols()
    spec = sym["pl"].pipeline_spec(PIPELINE)
    ref = inp["reference"]
    missing = [d for d in plans if f"doc_l4:{d}" not in rec.done]
    if missing:
        return {"complete": False, "reason": f"documents without an L4 record: {missing}; re-grant retries them", "rc": RC_PARTIAL}
    drs = doc_records(rec)
    allr = tally_docs(drs[d] for d in plans)
    tokens, chunks = sum(v["tokens"] for v in stats.values()), sum(v["chunks"] for v in stats.values())
    m3 = X.m3_values(allr["prefill_tokens"], allr["llm_calls"], tokens, chunks)
    canon_stats = spec.refresh_canonical(store, corpus_id)
    rows = G19.fact_rows(store.conn, corpus_id)
    now = readings(rows, store.canonical_facts(corpus_id), inp["cells"])
    held = set(ref["partition"]["held_out"])
    reg = regressions(ref, now, held)
    strata = strata_report(ref, now, held)
    from src.aggregation.store import CANONICAL_RULE, select_canonical  # noqa: PLC0415
    v1_on_v2 = G19.canon_ok(select_canonical(rows, rule=CANONICAL_RULE)[0], inp["cells"])
    raw_found = sum(now["raw"].values())
    m1 = raw_found / len(inp["cells"])
    p0n = {d["doc_name"] for d in X.p0_docs(inp["corpus"])}
    summ = X.m1_summary(inp["cells"], [now["raw"][cell_key(d, f)] for d, f, _g in inp["cells"]], p0n)
    n_windows = sum(r["n_windows"] for r in drs.values())
    rec.emit("ingest", unit="ingest", prefill_tokens=allr["prefill_tokens"], corpus_tokens=tokens, llm_calls=allr["llm_calls"],
             n_chunks=chunks, n_windows=n_windows, cache_n=allr["cache_n"], wall_s_fresh=allr["wall_s"],
             calls_per_unit=allr["llm_calls"] / (len(plans) * len(A.METRICS)),
             prefill_per_unit=allr["prefill_tokens"] / (len(plans) * len(A.METRICS)), bad_replies=allr["bad_replies"],
             bad_reply_rate=allr["bad_replies"] / allr["llm_calls"] if allr["llm_calls"] else None, reused_units=0)
    rec.emit("m1", unit="m1", gold_facts=len(inp["cells"]), found=raw_found, m1=m1, phase0_docs=summ["phase0_docs"],
             new_docs=summ["new_docs"], per_doc=summ["per_doc"], canonical_ok=sum(now["canonical"].values()),
             canonical_v1_rule_on_v2_facts=sum(v1_on_v2.values()), canonical=canon_stats)
    gates = (["M1"] if raw_found < ref["agg11"]["raw_found"] else []) + (["M3"] if X.m3_trips(m3) else [])
    detail = {"m1": m1, "m1_found": raw_found, "m1_n": len(inp["cells"]), "m1_agg11": ref["agg11"]["raw_m1"],
              "m1_agg11_found": ref["agg11"]["raw_found"], "m3": m3, "m3_agg11": ref["agg11"]["m3"], "m3_passed": not X.m3_trips(m3),
              "canonical": {"rule": canon_stats["rule"], "guards": canon_stats["guards"], "ok": sum(now["canonical"].values()),
                            "agg11_v1_ok": ref["agg11"]["canonical_v1_ok"], "v1_rule_on_v2_facts_ok": sum(v1_on_v2.values())},
              "regressions": reg, "strata": strata, "agg19_cells": agg19_report(inp["set_dir"], now), "m1_phase0_docs": summ["phase0_docs"], "m1_new_docs": summ["new_docs"],
              "calls_per_unit": allr["llm_calls"] / (len(plans) * len(A.METRICS)), "calls_per_unit_agg11": 1359 / 224,
              "bad_reply_rate": allr["bad_replies"] / allr["llm_calls"] if allr["llm_calls"] else None,
              "l4_reused": False, "l3_parse_reused_from_agg11_store": True}
    rec.emit("gates_cpu", unit="gates_cpu", gates_tripped=gates, **detail)
    if gates:
        return _gate_stop(rec, out, gates, "final", **detail)
    return _write_cpu_result(out, complete=True, futility_stop=None, gate_tripped=[], rc=RC_OK, **detail)


# --------------------------------------------------------------------------- GPU: audit and answers
def cpu_ready(result: dict[str, Any], ref: dict[str, Any]) -> tuple[bool, str]:
    """GPU precondition on the CPU row's ``result.cpu.json``: complete, no gate tripped, M1 found >= AGG-1.1's, M3 passed."""
    ok = (result.get("complete") is True and result.get("phase") == "cpu" and not result.get("futility_stop") and not result.get("gate_tripped")
          and (result.get("m1_found") or 0) >= ref["agg11"]["raw_found"] and result.get("m3_passed") is True)
    return ok, (f"complete={result.get('complete')} futility={result.get('futility_stop')} m1_found={result.get('m1_found')} "
                f"(AGG-1.1 {ref['agg11']['raw_found']}) m3_passed={result.get('m3_passed')}")


def answer_unit(planner: str, qid: str) -> str:
    return f"answer:{planner}:{qid}"


def run_answers(rec: P.Recorder, store: Any, backend: Any, *, agg: list[dict[str, Any]], names: list[str],
                guard: Callable[[], None], smoke: bool = False, cfg: P.Cfg = P.CFG, corpus_id: str = CORPUS_ID) -> dict[str, Any]:
    """Checkpoint 5b: P-large on every verified question under each planner in ``PLANNERS`` (agg110.v1 first, then agg11.v0), against the
    SAME store (canonical.v2 attributable-net-income, built once). No B1, P-small, B-today or B2 (not in AGG-1.12)."""
    from src.aggregation import planner  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    spec = orch_symbols()["pl"].pipeline_spec(PIPELINE)
    corpus_chunks = store.chunks(corpus_id)
    corpus_bm25 = BM25Index([c["text"] for c in corpus_chunks])
    docs_meta = store.corpus_docs(corpus_id)
    sql_present = bool(store.facts(corpus_id))
    built = sorted(store.built_layers(corpus_id))
    canon = spec.refresh_canonical(store, corpus_id)
    if "canonical" not in rec.done:
        rec.emit("canonical", unit="canonical", pipeline=spec.as_record(), **canon)
    scopes = P._scope_plan(planner, docs_meta, [("P-large", "agg", q) for q in agg])
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for c in corpus_chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    for pv in PLANNERS:
        for q in agg:
            unit = answer_unit(pv, q["id"])
            if unit in rec.done:
                continue
            guard()
            t, t0 = P.Tally(backend), time.time()
            text, st, extra = G110.p_large_answer_v(
                backend, store, q, scope=scopes[f"answer:P-large:{q['id']}"], planner=planner, docs_meta=docs_meta, built=built,
                sql_present=sql_present, names=names, cfg=cfg, corpus_chunks=corpus_chunks, corpus_bm25=corpus_bm25,
                doc_chunks=lambda d: by_doc.get(d, []), planner_version=pv, corpus_id=corpus_id)
            ok = P.score_agg(q, text) if st == "ok" else None
            rec.emit("answer", unit=unit, arm="P-large", qid=q["id"], qclass="agg", status=st, correct=ok, calls=t.calls,
                     prefill_tokens=t.prefill, answer=text[-600:], final=st in ("ok", "out_of_window", "budget_exhausted"),
                     wall_s=time.time() - t0, pipeline=PIPELINE, **extra)
            if smoke:
                return {"complete": False, "smoke": True}
    done = all(answer_unit(pv, q["id"]) in rec.done for pv in PLANNERS for q in agg)
    return {"complete": done}


def m4_of(recs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(recs), "median_calls": X._median([r["calls"] for r in recs]) if recs else None,
            "median_prefill": X._median([r["prefill_tokens"] for r in recs]) if recs else None}


def m4_trips(m4: dict[str, Any], complete: bool) -> bool:
    return bool(complete and (m4["median_calls"] > A.KILL["median_calls_max"] or m4["median_calls"] > M4_CALLS_SOFT
                              or m4["median_prefill"] > M4_PREFILL_MAX))


def verdict(cpu_result: dict[str, Any], records: list[dict[str, Any]], agg: list[dict[str, Any]], ref: dict[str, Any], *,
            store: Any = None) -> dict[str, Any]:
    """Gates (M1 and M3 from the CPU result, M4 per planner here), then the mechanical verdict:

    * ``phase1_gate_tripped``  a gate failed (M1 below AGG-1.1, M3 over a kill bound, M4 over its bound for either planner);
    * ``agg_1_12_not_met``     gates hold, but a cell that was right in AGG-1.1 is not right now (raw M1 or canonical);
    * ``agg_1_12_done``        gates hold and no previously correct cell regressed.
    P-large accuracy is FILED against AGG-1.1 (0.310) in every case, with its direction; it is not a gate (PREDECLARED)."""
    ids = [q["id"] for q in agg]
    ans = [r for r in records if r.get("kind") == "answer" and r.get("final")]
    ref_pl = ref["agg11"]["p_large"]["per_question"]
    old = {q: v["correct"] for q, v in ref_pl.items()}
    out: dict[str, Any] = {"planners": {}, "pipeline": PIPELINE}
    trips: list[str] = []
    if not (cpu_result.get("m1_found") is not None and cpu_result["m1_found"] >= ref["agg11"]["raw_found"]):
        trips.append("M1")
    if cpu_result.get("m3_passed") is not True:
        trips.append("M3")
    complete = True
    cont = {g["id"] for g in agg if g.get("continuity", not g["id"].startswith("p1_"))}
    per_pv: dict[str, dict[str, bool]] = {}
    for pv in PLANNERS:
        mine = {r["qid"]: r for r in ans if r.get("planner_version") == pv}
        done = all(q in mine for q in ids)
        complete = complete and done
        correct = {q: r.get("correct") is True for q, r in mine.items()}
        per_pv[pv] = correct
        m4 = m4_of([mine[q] for q in ids if q in mine])
        if m4_trips(m4, done):
            trips.append(f"M4:{pv}")
        n_ok = sum(correct.get(q, False) for q in ids)
        out["planners"][pv] = {
            "n_answered": len(mine), "n_correct": n_ok, "accuracy": n_ok / len(ids) if ids else None, "m4": m4,
            "vs_agg11": paired(correct, old, ids),
            "continuity": {"n_correct": sum(correct.get(q, False) for q in ids if q in cont), "n": sum(1 for q in ids if q in cont),
                           "agg11_correct": sum(old.get(q, False) for q in ids if q in cont)},
            "new": {"n_correct": sum(correct.get(q, False) for q in ids if q not in cont), "n": sum(1 for q in ids if q not in cont),
                    "agg11_correct": sum(old.get(q, False) for q in ids if q not in cont)},
            "regressed_questions": sorted(q for q in ids if old.get(q) and not correct.get(q)),
            "fixed_questions": sorted(q for q in ids if correct.get(q) and not old.get(q))}
    out["agg11"] = {"n_correct": ref["agg11"]["p_large"]["n_correct"], "n": ref["agg11"]["p_large"]["n"],
                    "accuracy": ref["agg11"]["p_large"]["accuracy"], "pipeline": "agg_pipeline.v1", "planner": G110.V0}
    if all(pv in per_pv for pv in (G110.V1, G110.V0)):
        out["planner_effect_same_store"] = paired(per_pv[G110.V1], per_pv[G110.V0], ids)
        out["pipeline_effect_same_planner_v0"] = paired(per_pv[G110.V0], old, ids)
        out["note"] = ("pipeline effect = v2 store with agg11.v0 vs AGG-1.1 (v1 store, agg11.v0); planner effect = agg110.v1 vs agg11.v0 on "
                       "the v2 store. IN-SAMPLE: all 58 questions were seen while the design was made; none reads only untouched cells.")
    reg = cpu_result.get("regressions") or {}
    out["regressions"] = {k: reg.get(k) for k in ("ok", "raw_broken", "canonical_broken", "n_retained_raw", "n_retained_canonical")}
    out["gates_tripped"] = trips
    out["complete"] = complete
    if trips:
        out["verdict"] = "phase1_gate_tripped"
    elif not complete:
        out["verdict"] = "incomplete"
    elif reg.get("ok") is False:
        out["verdict"] = "agg_1_12_not_met"
    else:
        out["verdict"] = "agg_1_12_done"
    prim = out["planners"].get(G110.V1)
    if prim and complete:
        out["direction"] = {"m1": "up" if cpu_result.get("m1_found", 0) > ref["agg11"]["raw_found"] else
                            "same" if cpu_result.get("m1_found") == ref["agg11"]["raw_found"] else "down",
                            "p_large_primary": "up" if prim["n_correct"] > out["agg11"]["n_correct"] else
                            "same" if prim["n_correct"] == out["agg11"]["n_correct"] else "down"}
        out["adopt_agg_pipeline_v2_as_default"] = bool(
            not trips and reg.get("ok") is True and out["direction"]["m1"] != "down" and out["direction"]["p_large_primary"] != "down")
    return out
