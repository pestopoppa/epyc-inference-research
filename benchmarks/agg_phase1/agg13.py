#!/usr/bin/env python3
"""AGG-1.3: incremental re-ingest of 3 changed documents (library; ``run_agg13.py`` is the CLI).

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an injected
backend (``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in tests).

Implements ``benchmarks/agg_phase1/PREDECLARED_AGG13.md`` literally. It imports the AGG-1.1 runner (``phase1_arms``),
the Phase 0 arms and the Phase 0 L3 loader and edits none of them (their sha256 pins are asserted by ``preflight``);
the L4 unit is ``phase1_arms.l4_unit`` itself, so the AGG-1.1 cache hits are valid by construction.

Phases (``run_agg13.py --phase``):

* ``preflight``    zero compute: pins, 28 PDFs, the AGG-1.1 L4 reuse test, the base store, the hash check of the 25
                   retained documents against the base store.
* ``reingest``     CPU: copy the base store; seed the 200 retained L4 units as cache hits; send all 28 documents through
                   ``CorpusStore.ingest`` (retained: parse cache -> ``unchanged``; changed: real parse -> ``replaced``);
                   L3 + 8 L4 units for the changed ones; G1 manifest, G2 supersession (structural), G3 cost, G4 M1;
                   ``result.cpu.json``.
* ``gpu-audit``    the Phase 0 L4c audit over the changed store (27B np4).
* ``gpu-answers``  P-large on the 58 AGG-1.1 questions (no B1; not scored); G5 (M4) and G2 (behavioural half).
* ``score``        verdict only (no model).

Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped (``phase1_gate_tripped``), 4 precondition,
5 budget stop (re-book), 75 partial (resume).
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
P0DIR = HERE.parent / "agg_phase0"
for _p in (str(P0DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg_phase0 as A  # noqa: E402
import l3_load as L3  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402

# --------------------------------------------------------------------------- pins
AMENDMENT1_SHA = "07831066b7549c1bb2b97b92e97ec960ec415ee794724189d120be537031d84c"
PREDECLARED_SHA = "fde0cb0d4bea0fc816e1393b38533c82c766853330d1218d2b0b37c3c5607183"
FROZEN = {  # asserted by preflight and by the tests (the gold sheet is read by one reader + the cross-extractor check)
    "corpus_agg13.json": "933b4d2f47f313cb903127a27f12f6f348b6887bdf813d8bc2fc8fe424926e43",
    "agg13_cells.csv": "5b7a4ea4b12375ae5d4e98b488f6c1287f9232b6775784a9620b18e295fb7b7e",
    "agg13_xcheck.json": "400a946172cf41e901b3cce71c147421e6f8c7bf7c59d3dc421ad30a7953f3bb",
}
REPLACED = ("MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K", "3M_2022_10K")  # declared order (= the AGG-1.1 pilot docs)
N_DOCS, N_REPLACED = 28, 3
FY_NEW = "2021"
FY_RETAINED = "2022"

# AGG-1.1 reference numbers (results.jsonl / verdict.json of agg1-ingest-cpu-20261111), written into PREDECLARED_AGG13.md
AGG11_FULL = {"llm_calls": 1359, "prefill_tokens": 1_712_985}
AGG11_PILOT = {"MICROSOFT_2022_10K": {"llm_calls": 48, "prefill_tokens": 57_618, "wall_s": 416.2},
               "JOHNSON_JOHNSON_2022_10K": {"llm_calls": 48, "prefill_tokens": 68_847, "wall_s": 421.2},
               "3M_2022_10K": {"llm_calls": 50, "prefill_tokens": 59_511, "wall_s": 449.4}}
G3_SHARE = 3 / 28
G3_SLACK = 1.10
FIRST_DOC_X = 2.0                      # budget stop: the first changed document over 2x its AGG-1.1 pilot rate
M4_MEDIAN_CALLS_MAX = X.M4_MEDIAN_CALLS_SOFT      # 3
M4_MEDIAN_PREFILL_MAX = X.M4_MEDIAN_PREFILL_MAX   # 2,510
M1_GATE = X.M1_GATE                    # 0.75
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_PARTIAL = (X.RC_OK, X.RC_PREFLIGHT, X.RC_GATE, X.RC_PRECOND,
                                                                   X.RC_BUDGET, X.RC_PARTIAL)

DEFAULT_AGG11_JOB = Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111")
SHA256 = X.sha256_file


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def g3_bounds(full: dict[str, float] | None = None) -> dict[str, float]:
    """3/28 x the full AGG-1.1 ingest x 1.10, for LLM calls and prefill tokens (the box's bound)."""
    full = full or AGG11_FULL
    f = G3_SHARE * G3_SLACK
    return {"llm_calls": full["llm_calls"] * f, "prefill_tokens": full["prefill_tokens"] * f,
            "plain_calls": full["llm_calls"] * G3_SHARE, "plain_prefill": full["prefill_tokens"] * G3_SHARE}


# --------------------------------------------------------------------------- inputs
def load_inputs(set_dir: Path = HERE, *, corpus: Path | None = None, cells: Path | None = None) -> dict[str, Any]:
    """The AGG-1.1 inputs (golds, verification, sheet) plus the changed corpus and the 3 new filings' cells."""
    inp = X.load_inputs(corpus=set_dir / "corpus.json", golds=set_dir / "golds.jsonl",
                        verification=set_dir / "verification_record.json", facts_sheet=set_dir / "facts_sheet.csv")
    inp["corpus11"] = inp["corpus"]
    inp["corpus"] = json.loads((corpus or set_dir / "corpus_agg13.json").read_text())
    inp["cells13"] = list(csv.DictReader((cells or set_dir / "agg13_cells.csv").open(newline="")))
    return inp


def doc_names(corpus: dict[str, Any]) -> list[str]:
    return [d["doc_name"] for d in corpus["docs"]]


def retained(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for d in corpus["docs"] if d["doc_name"] not in REPLACED]


def replaced_docs(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    by = {d["doc_name"]: d for d in corpus["docs"]}
    return [by[n] for n in REPLACED]


def m1_cells13(inp: dict[str, Any]) -> list[tuple[str, str, float, str]]:
    """(doc, metric, value, fiscal-year string): the AGG-1.1 verified cells of the 25 retained documents (FY2022)
    plus every filled cell of ``agg13_cells.csv`` (FY2021 for the 3 replaced slots)."""
    keep = {d["doc_name"] for d in retained(inp["corpus"])}
    cells = [(d, m, v, FY_RETAINED) for d, m, v in X.m1_cells(inp["sheet"], X.verified_agg(inp), sorted(keep))]
    for r in inp["cells13"]:
        if r["status"] == "filled" and r["value_usd_m"].strip():
            cells.append((r["doc_name"], r["metric"], float(r["value_usd_m"].replace(",", "")), str(r["fiscal_year"])))
    return cells


def m1_hits_fy(facts: Iterable[dict[str, Any]], cells: list[tuple[str, str, float, str]], *,
               unit_scale: dict[str, float] | None = None, rel_tol: float = P.CFG.rel_tol) -> list[bool]:
    """``phase1_arms.m1_hits`` with the fiscal year per cell: a fact for the same doc and field whose period
    contains the cell's year, value within ``rel_tol`` after unit scale, no leniency for a 1000x unit error."""
    if unit_scale is None:
        unit_scale = X.M1R_UNIT_SCALE
    by: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for f in facts:
        if f.get("value") is None or not f.get("period"):
            continue
        v = float(f["value"])
        if f.get("unit") in unit_scale:
            v *= unit_scale[f["unit"]]
        by.setdefault((f.get("doc_id"), f.get("field")), []).append((str(f["period"]), v))
    return [any(fy in per and abs(abs(v) - abs(gv)) <= rel_tol * abs(gv) + 1e-9 for per, v in by.get((d, m), []))
            for d, m, gv, fy in cells]


def m1_summary13(cells: list[tuple[str, str, float, str]], hits: list[bool]) -> dict[str, Any]:
    def agg(sel: Callable[[str], bool]) -> dict[str, Any]:
        idx = [i for i, c in enumerate(cells) if sel(c[0])]
        g, f = len(idx), sum(hits[i] for i in idx)
        return {"gold": g, "found": f, "m1": (f / g) if g else None}
    per: dict[str, dict[str, int]] = {}
    for (d, _m, _v, _fy), h in zip(cells, hits):
        b = per.setdefault(d, {"gold": 0, "found": 0})
        b["gold"] += 1
        b["found"] += int(h)
    return {"overall": agg(lambda d: True), "retained": agg(lambda d: d not in REPLACED),
            "changed": agg(lambda d: d in REPLACED), "per_doc": dict(sorted(per.items()))}


# --------------------------------------------------------------------------- preflight
def pin_problems(set_dir: Path = HERE, *, predeclared_sha: str = PREDECLARED_SHA,
                 frozen: dict[str, str] | None = None) -> list[str]:
    """Frozen-input and PREDECLARED pins; the AGG-1.1 pins are checked by ``X.preflight``."""
    problems = []
    pd = set_dir / "PREDECLARED_AGG13.md"
    got = SHA256(pd) if pd.exists() else None
    if got != predeclared_sha:
        problems.append(f"PREDECLARED_AGG13.md sha256 {got} != recorded {predeclared_sha}")
    am = set_dir / "PREDECLARED_AGG13_AMENDMENT1_20261111.md"
    got = SHA256(am) if am.exists() else None
    if got != AMENDMENT1_SHA:
        problems.append(f"PREDECLARED_AGG13_AMENDMENT1_20261111.md sha256 {got} != recorded {AMENDMENT1_SHA}")
    for name, want in (FROZEN if frozen is None else frozen).items():
        p = set_dir / name
        got = SHA256(p) if p.exists() else None
        if got != want:
            problems.append(f"frozen input {name} sha256 {got} != pinned {want}")
    return problems


def corpus_problems(corpus: dict[str, Any], corpus11: dict[str, Any], *, check_pdfs: bool = True) -> list[str]:
    """28 docs; the 25 retained entries byte-identical to AGG-1.1; the 3 replaced ones new files with new hashes."""
    problems = []
    docs = corpus["docs"]
    if len(docs) != N_DOCS:
        problems.append(f"{len(docs)} documents, PREDECLARED has {N_DOCS}")
    if X.digest_docs(docs) != corpus.get("manifest_digest_sha256"):
        problems.append("corpus_agg13 manifest digest differs from its docs array")
    old = {d["doc_name"]: d for d in corpus11["docs"]}
    for d in docs:
        n = d["doc_name"]
        if n in REPLACED:
            r = d.get("replaces") or {}
            if r.get("sha256") != old[n]["sha256"] or d["sha256"] == old[n]["sha256"]:
                problems.append(f"{n}: replacement must be a new file replacing the AGG-1.1 one")
            if not str(d.get("file_name", "")).endswith("_2021_10K"):
                problems.append(f"{n}: replacement is not a FY2021 10-K ({d.get('file_name')})")
        elif d != old.get(n):
            problems.append(f"{n}: retained entry differs from AGG-1.1 corpus.json")
    if sum(1 for d in docs if d["doc_name"] in REPLACED) != N_REPLACED:
        problems.append("replaced slots are not the 3 declared documents")
    if check_pdfs:
        bad = [d["doc_name"] for d in docs if not Path(d["pdf"]).exists() or SHA256(Path(d["pdf"])) != d["sha256"]]
        if bad:
            problems.append(f"pdf missing or sha256 mismatch: {bad}")
    return problems


def cells_problems(inp: dict[str, Any]) -> list[str]:
    """The 3 new filings' cell sheet: 8 metrics per replaced slot, FY2021, verified, cross-extractor clean."""
    problems = []
    rows = inp["cells13"]
    by: dict[str, set[str]] = {}
    for r in rows:
        by.setdefault(r["doc_name"], set()).add(r["metric"])
        if r["doc_name"] not in REPLACED:
            problems.append(f"agg13 cell for a non-replaced slot: {r['doc_name']}")
        if str(r["fiscal_year"]) != FY_NEW:
            problems.append(f"{r['doc_name']} {r['metric']}: fiscal_year {r['fiscal_year']} != {FY_NEW}")
        if r["verified"] != "true":
            problems.append(f"{r['doc_name']} {r['metric']}: not verified")
    for n in REPLACED:
        if by.get(n) != set(A.METRICS):
            problems.append(f"{n}: cells for {sorted(by.get(n, ()))}, need the 8 metrics")
    return problems


def l4_reuse_check13(agg11_job: Path, orch_root: Path, research_arms: Path | None = None) -> dict[str, Any]:
    """The AGG-1.1 reuse rule, against the AGG-1.1 job's own pin: the L4 stage sources (``facts_llm.py``,
    ``llm_backend.py``, ``phase0_arms.py``) equal their AGG-1.1 sha256 and the ``l4_unit`` statements are AST-equal to
    ``run_cpu``'s L4 loop. Empty diff -> the 200 retained L4 units are cache hits; otherwise fail closed."""
    res = X.l4_reuse_check(agg11_job, orch_root, research_arms)
    res["against"] = "AGG-1.1 job pin (" + str(agg11_job) + ")"
    return res


def read_doc_hashes(store_path: Path, corpus_id: str) -> dict[str, str]:
    """``doc_id -> doc_hash`` of a store, opened ``immutable=1`` (read-only, never creates a WAL or a table): the
    AGG-1.1 stores are never opened through ``CorpusStore`` (its constructor writes DDL and a meta row)."""
    import sqlite3  # noqa: PLC0415
    c = sqlite3.connect(f"file:{store_path}?immutable=1", uri=True)
    try:
        return {r[0]: r[1] for r in c.execute("SELECT doc_id, doc_hash FROM documents WHERE corpus_id = ?", [corpus_id])}
    finally:
        c.close()


def base_hash_check(corpus: dict[str, Any], have: dict[str, str], parse_for: Callable[[dict[str, Any]], Any]
                    ) -> dict[str, Any]:
    """Zero-compute check that each retained document's cached parse hashes to the store's content hash for its
    slot (so ``ingest`` will answer ``unchanged``), and that the 3 slots hold the OLD documents."""
    from src.aggregation.store import document_content_hash  # noqa: PLC0415
    out: dict[str, Any] = {"retained_equal": 0, "retained_unequal": [], "retained_no_cache": [], "base_hashes": have}
    for d in retained(corpus):
        n = d["doc_name"]
        result = parse_for(d)
        if result is None:
            out["retained_no_cache"].append(n)
        elif document_content_hash(result) == have.get(n):
            out["retained_equal"] += 1
        else:
            out["retained_unequal"].append(n)
    out["slots_present"] = all(n in have for n in REPLACED)
    out["n_base_docs"] = len(have)
    return out


# --------------------------------------------------------------------------- parse cache
def cache_name(d: dict[str, Any]) -> str:
    return f"{d['doc_name']}.{d['sha256'][:12]}.json"


def find_cached(d: dict[str, Any], dirs: Iterable[Path]) -> Path | None:
    for c in dirs:
        p = Path(c) / cache_name(d)
        if p.exists():
            return p
    return None


def load_cached(d: dict[str, Any], dirs: Iterable[Path]) -> tuple[Any, float | None] | None:
    from src.models.document import DocumentPreprocessResult  # noqa: PLC0415
    p = find_cached(d, dirs)
    if p is None:
        return None
    blob = json.loads(p.read_text())
    return DocumentPreprocessResult.from_cache_dict(blob), blob.get("_parse_s")


# --------------------------------------------------------------------------- store snapshot + supersession checks
def superseded_check(store, corpus_id: str, base: dict[str, Any], new_hashes: dict[str, str]) -> dict[str, Any]:
    """G2 structural half (PREDECLARED_AGG13.md): per replaced slot, the old hash has no live rows, its facts are in the
    ledger (count equal to the base store's), ``superseded_by`` is the new hash; after ``refresh_canonical`` no
    canonical row carries an old hash or a superseded ``fact_id``; probe SQL returns only new-hash rows."""
    store.refresh_canonical(corpus_id)
    ledger_docs = {d["doc_hash"]: d for d in store.superseded_documents(corpus_id)}
    sup_ids = {f["fact_id"] for f in store.superseded_facts(corpus_id)}
    live_ids = {r[0] for r in store.conn.execute("SELECT fact_id FROM facts").fetchall()}
    canon_ids = {r[0] for r in store.conn.execute(
        "SELECT fact_id FROM facts_canonical WHERE corpus_id = ?", [corpus_id]).fetchall()}
    per: dict[str, Any] = {}
    problems: list[str] = []
    for slot in REPLACED:
        old = base["old_hashes"][slot]
        new = new_hashes.get(slot)
        live = {t: store.conn.execute(f"SELECT COUNT(*) FROM {t} WHERE doc_hash = ?", [old]).fetchone()[0]
                for t in ("documents", "chunks", "tables_", "facts", "facts_canonical")}
        led = store.superseded_facts(corpus_id, doc_hash=old)
        doc_row = ledger_docs.get(old)
        probe = store.conn.execute(
            "SELECT DISTINCT doc_hash FROM facts_canonical WHERE corpus_id = ? AND doc_id = ?", [corpus_id, slot]
        ).fetchall()
        row = {"old_hash": old, "new_hash": new, "live_rows_of_old_hash": live, "ledger_facts": len(led),
               "base_facts": base["old_fact_counts"][slot],
               "ledger_superseded_by_new": all(f["superseded_by"] == new for f in led),
               "documents_superseded_row": bool(doc_row) and doc_row["superseded_by"] == new,
               "canonical_hashes_for_slot": sorted(h[0] for h in probe)}
        if any(live.values()):
            problems.append(f"{slot}: old hash still live {live}")
        if len(led) != base["old_fact_counts"][slot] or not led:
            problems.append(f"{slot}: ledger holds {len(led)} facts, base store held {base['old_fact_counts'][slot]}")
        if not row["ledger_superseded_by_new"] or not row["documents_superseded_row"]:
            problems.append(f"{slot}: superseded_by is not the new hash")
        if any(h != new for h in row["canonical_hashes_for_slot"]):
            problems.append(f"{slot}: a canonical row of the slot carries a hash other than the new one")
        per[slot] = row
    leaked = sorted((sup_ids & live_ids) | (sup_ids & canon_ids))
    if leaked:
        problems.append(f"{len(leaked)} superseded fact_ids are live or canonical, e.g. {leaked[:3]}")
    return {"ok": not problems, "problems": problems, "per_slot": per, "n_ledger_facts": len(sup_ids),
            "n_documents_superseded": len(ledger_docs)}


def sql_absence(store, corpus_id: str, sqls: list[str], *, row_cap: int = 200) -> dict[str, Any]:
    """G2 behavioural half: re-execute each emitted SQL read-only; no returned row may equal a superseded fact by
    (doc_id, field, period, value, quote) unless the live canonical table independently carries the identical row."""
    sup = {(f["doc_id"], f["field"], str(f["period"]), round(float(f["value"]), 6), (f["quote"] or "")[:160])
           for f in store.superseded_facts(corpus_id) if f.get("value") is not None}
    live = {(r["doc_id"], r["field"], str(r["period"]), round(float(r["value"]), 6), (r["quote"] or "")[:160])
            for r in store.canonical_facts(corpus_id) if r.get("value") is not None}
    hits, n_rows, n_run, n_fail = [], 0, 0, 0
    for sql in sqls:
        try:
            cur = store.conn.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchmany(row_cap)]
        except Exception:  # noqa: BLE001 - a failing statement returns no rows; counted, not hidden
            n_fail += 1
            continue
        n_run += 1
        for r in rows:
            n_rows += 1
            key = (r.get("doc_id"), r.get("field"), str(r.get("period")),
                   round(float(r["value"]), 6) if isinstance(r.get("value"), (int, float)) else None,
                   (r.get("quote") or "")[:160])
            if key in sup and key not in live:
                hits.append({"sql": sql[:200], "row": {k: r[k] for k in ("doc_id", "field", "period", "value") if k in r}})
    return {"n_sql": len(sqls), "n_run": n_run, "n_failed": n_fail, "n_rows": n_rows, "superseded_hits": hits,
            "ok": not hits}


# --------------------------------------------------------------------------- gates (pure)
def g1_eval(manifest: list[dict[str, Any]], failed: list[str]) -> dict[str, Any]:
    """Exactly 3 re-parsed and re-extracted, 25 skipped by hash."""
    chg = [m for m in manifest if m["name"] in REPLACED]
    ret = [m for m in manifest if m["name"] not in REPLACED]
    ok_chg = [m for m in chg if m["status"] == "replaced" and m["parse"] == "real" and m["l3_ran"]
              and m["l4_units_run"] == len(A.METRICS)]
    ok_ret = [m for m in ret if m["status"] == "unchanged" and m["parse"] == "cached" and not m["l3_ran"]
              and m["l3_facts"] == 0 and m["l4_units_run"] == 0 and m["llm_calls"] == 0]
    problems = []
    if len(ok_chg) != N_REPLACED:
        problems.append(f"{len(ok_chg)} of {N_REPLACED} changed documents re-parsed and re-extracted")
    if len(ok_ret) != N_DOCS - N_REPLACED:
        problems.append(f"{len(ok_ret)} of {N_DOCS - N_REPLACED} retained documents skipped by hash")
    if failed:
        problems.append(f"doc_fail {sorted(failed)}")
    return {"ok": not problems, "problems": problems, "n_reparsed_reextracted": len(ok_chg),
            "n_skipped_by_hash": len(ok_ret), "n_manifest": len(manifest)}


def g3_eval(fresh: dict[str, Any], bounds: dict[str, float] | None = None) -> dict[str, Any]:
    b = bounds or g3_bounds()
    over = [k for k in ("llm_calls", "prefill_tokens") if fresh[k] > b[k]]
    return {"ok": not over, "over": over, "fresh_calls": fresh["llm_calls"], "fresh_prefill": fresh["prefill_tokens"],
            "bound_calls": b["llm_calls"], "bound_prefill": b["prefill_tokens"],
            "calls_share_of_bound": fresh["llm_calls"] / b["llm_calls"],
            "prefill_share_of_bound": fresh["prefill_tokens"] / b["prefill_tokens"],
            "full_ingest_reference": dict(AGG11_FULL), "plain_3_28": {"calls": b["plain_calls"],
                                                                       "prefill": b["plain_prefill"]}}


def first_doc_over(units: list[dict[str, Any]], slot: str) -> dict[str, Any] | None:
    """Budget stop: the first changed document costs more than 2x its AGG-1.1 pilot rate (calls, prefill or wall)."""
    ref = AGG11_PILOT[slot]
    t = X.tally(units)
    over = [k for k in ("llm_calls", "prefill_tokens", "wall_s") if t[k] > FIRST_DOC_X * ref[k]]
    return {"doc": slot, "over": over, "tally": t, "limit": {k: FIRST_DOC_X * ref[k] for k in ref}} if over else None


def g5_eval(answers: list[dict[str, Any]], n_expected: int) -> dict[str, Any]:
    """M4 unchanged: median P-large calls <= 3 and median prefill <= 2,510 over the 58."""
    done = [r for r in answers if r.get("final")]
    m = {"n": len(done), "median_calls": X._median([r["calls"] for r in done]) if done else None,
         "median_prefill": X._median([r["prefill_tokens"] for r in done]) if done else None}
    complete = len(done) == n_expected
    trip = complete and (m["median_calls"] > M4_MEDIAN_CALLS_MAX or m["median_prefill"] > M4_MEDIAN_PREFILL_MAX
                         or m["median_calls"] > A.KILL["median_calls_max"])
    return {"complete": complete, "ok": complete and not trip, **m,
            "agg11": {"median_calls": 2.0, "median_prefill": 606.5},
            "bounds": {"median_calls": M4_MEDIAN_CALLS_MAX, "median_prefill": M4_MEDIAN_PREFILL_MAX}}


# --------------------------------------------------------------------------- CPU phase
def seed_units(rec: P.Recorder, agg11_results: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    """The AGG-1.1 ok L4 units of the 25 retained documents as cache hits (``reused: true``); the 3 replaced slots'
    old units are NOT seeded (their documents changed). Also records the AGG-1.1 full-ingest numbers beside them."""
    recs = X.read_jsonl(agg11_results)
    ok = X.l4_ok(recs)
    keep = {d["doc_name"] for d in retained(corpus)}
    n = 0
    for unit, r in sorted(ok.items()):
        if r.get("doc") not in keep or unit in rec.done:
            continue
        rec.emit("l4_unit", **{k: v for k, v in r.items() if k not in ("kind", "ts")}, reused=True,
                 reused_from=str(agg11_results))
        n += 1
    ing = next((r for r in recs if r.get("kind") == "ingest"), None)
    if ing and "agg11_ingest" not in rec.done:
        rec.emit("agg11_ingest", unit="agg11_ingest", **{k: ing[k] for k in (
            "prefill_tokens", "corpus_tokens", "llm_calls", "n_chunks") if k in ing})
    m1 = next((r for r in recs if r.get("kind") == "m1"), None)
    if m1 and "agg11_m1" not in rec.done:
        rec.emit("agg11_m1", unit="agg11_m1", per_doc=m1.get("per_doc"), m1=m1.get("m1"))
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
                  verdict="phase1_gate_tripped", where=where, m1=detail.get("m1"), m3_passed=False, rc=RC_GATE)
    res["rc"] = RC_GATE
    return res


def _doc_units(rec: P.Recorder, slot: str) -> list[dict[str, Any]]:
    return [r for r in X.l4_ok(rec.records).values() if r.get("doc") == slot and not r.get("reused")]


def run_reingest(rec: P.Recorder, out: Path, inp: dict[str, Any], orch_root: Path, backend, *,
                 base_store: Path, agg11_results: Path, cache_dirs: list[Path], guard: Callable[[], None],
                 parse_fn: Callable[[Path], Any] | None = None, smoke: bool = False,
                 corpus_id: str | None = None, reuse_ok: bool = True, now: Callable[[], float] = time.time,
                 bounds: dict[str, float] | None = None, cfg: P.Cfg = P.CFG) -> dict[str, Any]:
    """The CPU row. Resumable: every record with a ``unit`` is skipped on re-entry."""
    corpus_id = corpus_id or X.CORPUS_ID
    parse_fn = parse_fn or L3.real_parse
    for k, v in L3.JOB_ENV.items():
        os.environ[k] = v
    corpus = inp["corpus"]
    names = doc_names(corpus)
    by_name = {d["doc_name"]: d for d in corpus["docs"]}
    if not reuse_ok:
        rec.emit("precondition_failed", reason="L4 reuse test failed: the 200 retained units are not cache hits")
        return {"complete": False, "rc": RC_PRECOND}
    store_path = out / "store.sqlite"
    if not store_path.exists():
        if not base_store.exists():
            rec.emit("precondition_failed", reason=f"no base store {base_store}")
            return {"complete": False, "rc": RC_PRECOND}
        P.copy_store(base_store, store_path)
        rec.emit("store_copied", src=str(base_store), sha256=SHA256(store_path), final=False)
    from src.aggregation.registry import IndexRegistry  # noqa: PLC0415
    from src.aggregation.store import CorpusStore, document_content_hash  # noqa: PLC0415
    store = CorpusStore(store_path)
    try:
        if "reuse_seeded" not in rec.done:
            rec.emit("reuse_seeded", unit="reuse_seeded", **seed_units(rec, agg11_results, corpus))
        # ---- the base snapshot, written once BEFORE the first ingest touches the copy
        if "base_snapshot" not in rec.done:
            have = {d["doc_id"]: d["doc_hash"] for d in store.corpus_docs(corpus_id)}
            missing = sorted(set(names) - set(have))
            if missing:
                rec.emit("precondition_failed", reason=f"base store lacks documents {missing}")
                return {"complete": False, "rc": RC_PRECOND}
            counts = {s: store.conn.execute("SELECT COUNT(*) FROM facts WHERE doc_hash = ?", [have[s]]).fetchone()[0]
                      for s in REPLACED}
            rec.emit("base_snapshot", unit="base_snapshot", n_docs=len(have), old_hashes={s: have[s] for s in REPLACED},
                     old_fact_counts=counts, base_hashes=have,
                     base_facts=store.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        base = next(r for r in rec.records if r.get("unit") == "base_snapshot")
        rec.emit("meta", phase="reingest", smoke=smoke, schema="agg13.v1", corpus_id=corpus_id, final=False,
                 n_docs=len(names), n_replaced=len(REPLACED), predeclared_sha256=PREDECLARED_SHA)
        failed = {r["doc"] for r in rec.records if r["kind"] == "doc_fail"}
        if failed:  # a recorded document failure is final: G1 cannot be met (rerun only after a dated amendment)
            return _gate_stop(rec, out, ["G1"], "ingest", doc_fail=sorted(failed))
        cache_out = out / "cache"
        order = [d for d in names if d not in REPLACED] + list(REPLACED)
        for name in order:
            d = by_name[name]
            unit = f"doc:{name}"
            if unit not in rec.done:  # ingest once; a re-entry still finishes the changed document's L4 units
                guard()
                t0 = now()
                if not Path(d["pdf"]).exists() or SHA256(Path(d["pdf"])) != d["sha256"]:
                    rec.emit("doc_fail", unit=unit, doc=name, error="pdf missing or sha256 mismatch", final=True)
                    return _gate_stop(rec, out, ["G1"], "ingest", doc_fail=[name])
                try:
                    got = load_cached(d, [cache_out, *cache_dirs])
                    if got is None and name not in REPLACED:
                        # a retained document without a cached parse would be a real re-parse: that is a G1 failure
                        rec.emit("doc_fail", unit=unit, doc=name, error="no cached parse for a retained document",
                                 final=True)
                        return _gate_stop(rec, out, ["G1"], "ingest", doc_fail=[name])
                    if got is None:
                        tp = now()
                        result = parse_fn(Path(d["pdf"]))
                        parse_s = now() - tp
                        payload = result.to_cache_dict()
                        payload["_parse_s"] = parse_s
                        X._atomic_write(cache_out / cache_name(d), json.dumps(payload, default=str))
                        parse_kind = "real"
                    else:
                        result, parse_s = got
                        # a blob in OUT/cache was written by this job's own real parse (a resumed entry), a blob in
                        # a seed cache dir is a cache hit
                        parse_kind = "real" if find_cached(d, [cache_out]) is not None else "cached"
                    new_hash = document_content_hash(result)
                    if name not in REPLACED and new_hash != base["base_hashes"][name]:
                        # ingest would REPLACE a retained document and drop its facts: never let it
                        rec.emit("doc_fail", unit=unit, doc=name, final=True,
                                 error=f"cached parse hashes to {new_hash[:12]}, store holds {base['base_hashes'][name][:12]}")
                        return _gate_stop(rec, out, ["G1"], "ingest", doc_fail=[name])
                    rc = store.ingest(result, corpus_id=corpus_id, doc_id=name, entity=d["company"])
                    status, superseded = rc.status, list(rc.superseded)
                    if name in REPLACED and status == "unchanged" and new_hash != base["old_hashes"][name]:
                        # a previous entry was killed between the store commit and its ``doc_ingest`` record:
                        # the slot already holds the new document, so it IS the replaced one (ledger has the old hash)
                        status, superseded = "replaced", [base["old_hashes"][name]]
                    have_l3 = store.conn.execute("SELECT COUNT(*) FROM facts WHERE doc_hash = ? AND extractor LIKE 'l3%'",
                                                 [new_hash]).fetchone()[0]
                    n_l3, l3_ran = 0, False
                    if status in ("replaced", "ingested") and not have_l3:
                        l3_ran = True
                        facts, n_tab, n_rx = L3.doc_l3_facts(store, corpus_id, name)
                        store.write_facts(facts)
                        n_l3 = len(facts)
                        rec.emit("l3_doc", doc=name, n_facts=n_l3, n_table_facts=n_tab, n_regex_facts=n_rx,
                                 fields=len({f["field"] for f in facts}), llm_calls=0, prefill_tokens=0)
                    rec.emit("doc_ingest", unit=unit, doc=name, status=status, doc_hash=rc.doc_hash,
                             superseded=superseded, parse=parse_kind, parse_s=parse_s,
                             pdf_sha256=d["sha256"], l3_facts=n_l3, l3_ran=l3_ran, wall_s=now() - t0, final=True)
                except P.Stop:
                    raise
                except Exception as e:  # noqa: BLE001 - recorded; a failed document trips G1 at once
                    rec.emit("doc_fail", unit=unit, doc=name, error=f"{type(e).__name__}: {e}"[:500], final=True)
                    return _gate_stop(rec, out, ["G1"], "ingest", doc_fail=[name])
            if name not in REPLACED:
                continue
            # ---- the changed document's L4 units (the AGG-1.1 l4_unit, unchanged)
            units = P.windows(store, corpus_id, cfg.window_chars)
            for f in A.METRICS:
                if f"l4:{f}:{name}" in rec.done:
                    continue
                X.l4_unit(rec, store, backend, units, f, name, guard, cfg)
                if smoke:
                    return {"complete": False, "smoke": True, "rc": RC_OK}
                gate = _running_g3(rec, bounds)
                if gate:
                    return _gate_stop(rec, out, ["G3"], "running", g3=gate)
            if not all(f"l4:{f}:{name}" in rec.done for f in A.METRICS):
                return {"complete": False, "reason": "l4 units with errors; re-grant retries them", "rc": RC_PARTIAL}
            if not any(r.get("unit") == "first_doc_check" for r in rec.records):
                over = first_doc_over(_doc_units(rec, name), name)
                rec.emit("first_doc_check", unit="first_doc_check", doc=name, over=over)
                if over:
                    rec.emit("stop", reason="budget_stop:first_changed_doc_over_2x_pilot_rate", final=False, **over)
                    return {"complete": False, "budget_stop": "first_changed_doc_over_2x_pilot_rate", "rc": RC_BUDGET,
                            **over}
        # ---- finalise: indexes, manifest, gates
        return _finish(rec, out, store, inp, corpus_id, base, failed, bounds, reg_cls=IndexRegistry)
    finally:
        store.close()


def _running_g3(rec: P.Recorder, bounds: dict[str, float] | None) -> dict[str, Any] | None:
    fresh = X.tally(r for r in X.l4_ok(rec.records).values() if not r.get("reused"))
    g = g3_eval(fresh, bounds)
    return None if g["ok"] else g


def _finish(rec, out, store, inp, corpus_id, base, failed, bounds, *, reg_cls) -> dict[str, Any]:
    corpus = inp["corpus"]
    done = {r["doc"]: r for r in rec.records if r["kind"] == "doc_ingest"}
    ok_units = X.l4_ok(rec.records)
    manifest = []
    for n in doc_names(corpus):
        r = done.get(n, {})
        mine = [u for u in ok_units.values() if u.get("doc") == n and not u.get("reused")]
        manifest.append({"name": n, "pdf_sha256": by_sha(corpus, n), "status": r.get("status"), "parse": r.get("parse"),
                         "doc_hash": r.get("doc_hash"), "superseded": r.get("superseded", []),
                         "l3_facts": r.get("l3_facts", 0), "l3_ran": bool(r.get("l3_ran")),
                         "l4_units_run": len(mine),
                         "llm_calls": sum(u["llm_calls"] for u in mine),
                         "prefill_tokens": sum(u["prefill_tokens"] for u in mine),
                         "wall_s": sum(u.get("wall_s") or 0.0 for u in mine)})
    g1 = g1_eval(manifest, sorted(failed))
    new_hashes = {n: done[n]["doc_hash"] for n in REPLACED if n in done}
    g2 = superseded_check(store, corpus_id, base, new_hashes)
    fresh = X.tally(r for r in ok_units.values() if not r.get("reused"))
    g3 = g3_eval(fresh, bounds)
    cells = m1_cells13(inp)
    hits = m1_hits_fy(store.facts(corpus_id), cells)
    m1s = m1_summary13(cells, hits)
    # AGG-1.1 comparator for the 25 retained docs: their hits must equal the AGG-1.1 hits for those docs (their facts
    # were not touched); skipped, and reported as None, only when the AGG-1.1 results carry no per-document M1
    a11 = next((r for r in rec.records if r.get("unit") == "agg11_m1"), None)
    same = None
    if a11 and a11.get("per_doc"):
        same = all(m1s["per_doc"].get(d, {}).get("found") == a11["per_doc"].get(d, {}).get("found")
                   for d in a11["per_doc"] if d not in REPLACED)
    g4 = {"ok": (m1s["overall"]["m1"] or 0) >= M1_GATE and same is not False, "m1": m1s["overall"]["m1"],
          "gate": M1_GATE, "retained_equals_agg11": same, **m1s}
    reg = reg_cls(store)
    reg.get_or_build(corpus_id, "l1_bm25")
    n_l3 = store.conn.execute("SELECT COUNT(*) FROM facts WHERE extractor LIKE 'l3%'").fetchone()[0]
    store.record_index(corpus_id, store.corpus_content_hash(corpus_id), "l3_facts_det", "", layer="L3", n_items=n_l3)
    manifest_doc = {"schema": "agg13_reingest_manifest.v1", "n_docs": len(manifest),
                    "n_reparsed_reextracted": g1["n_reparsed_reextracted"], "n_skipped_by_hash": g1["n_skipped_by_hash"],
                    "docs": manifest, "corpus_content_hash": store.corpus_content_hash(corpus_id)}
    X._atomic_write(out / "reingest_manifest.json", json.dumps(manifest_doc, indent=1, default=str) + "\n")
    store.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                       ["agg13_reingest_manifest", json.dumps({k: manifest_doc[k] for k in (
                           "n_docs", "n_reparsed_reextracted", "n_skipped_by_hash")})])
    store.conn.commit()
    rec.emit("manifest", unit="manifest", **{k: manifest_doc[k] for k in (
        "n_reparsed_reextracted", "n_skipped_by_hash", "corpus_content_hash")})
    rec.emit("gates_cpu", unit="gates_cpu", g1=g1, g2=g2, g3=g3, g4=g4)
    tripped = [g for g, v in (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)) if not v["ok"]]
    res = _result(out, complete=True, futility_stop=("+".join(tripped) if tripped else None), gate_tripped=tripped,
                  verdict="phase1_gate_tripped" if tripped else "agg_1_3_cpu_pass", m1=g4["m1"], m3_passed=not tripped,
                  g1=g1, g2=g2, g3=g3, g4=g4, m1_retained=m1s["retained"], m1_changed=m1s["changed"],
                  fresh=fresh, doc_fail=sorted(failed), rc=RC_GATE if tripped else RC_OK)
    return res


def by_sha(corpus: dict[str, Any], name: str) -> str:
    return next(d["sha256"] for d in corpus["docs"] if d["doc_name"] == name)


def cpu_ready(result: dict[str, Any]) -> tuple[bool, str]:
    """GPU precondition on the CPU row's ``result.cpu.json``: complete, no gate tripped, G1-G4 ok."""
    ok = (result.get("complete") is True and not result.get("futility_stop") and not result.get("gate_tripped")
          and all((result.get(g) or {}).get("ok") is True for g in ("g1", "g2", "g3", "g4"))
          and result.get("verdict") == "agg_1_3_cpu_pass")
    return ok, (f"complete={result.get('complete')} gate_tripped={result.get('gate_tripped')} "
                f"verdict={result.get('verdict')}")


# --------------------------------------------------------------------------- GPU phases
def run_p_large(rec: P.Recorder, store, backend, *, agg: list[dict[str, Any]], names: list[str],
                guard: Callable[[], None], smoke: bool = False, cfg: P.Cfg = P.CFG,
                corpus_id: str | None = None) -> dict[str, Any]:
    """P-large on the 58 questions over the changed store (``phase1_arms.run_answers``' P-large loop, no B1,
    answers recorded but not scored: ``correct`` is null by design)."""
    from src.aggregation import planner  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    corpus_id = corpus_id or X.CORPUS_ID
    corpus_chunks = store.chunks(corpus_id)
    corpus_bm25 = BM25Index([c["text"] for c in corpus_chunks])
    docs_meta = store.corpus_docs(corpus_id)
    sql_present = bool(store.facts(corpus_id))
    built = sorted(store.built_layers(corpus_id))
    canon = store.refresh_canonical(corpus_id)
    if "canonical" not in rec.done:
        rec.emit("canonical", unit="canonical", **canon)
    scopes = P._scope_plan(planner, docs_meta, [("P-large", "agg", q) for q in agg])
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for c in corpus_chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    for q in agg:
        unit = f"answer:P-large:{q['id']}"
        if unit in rec.done:
            continue
        guard()
        t, t0 = P.Tally(backend), time.time()
        text, st, extra = X.p_large_answer(backend, store, q, scope=scopes[unit], planner=planner, docs_meta=docs_meta,
                                           built=built, sql_present=sql_present, names=names, cfg=cfg,
                                           corpus_chunks=corpus_chunks, corpus_bm25=corpus_bm25,
                                           doc_chunks=lambda d: by_doc.get(d, []))
        rec.emit("answer", unit=unit, arm="P-large", qid=q["id"], qclass="agg", status=st, correct=None, calls=t.calls,
                 prefill_tokens=t.prefill, answer=text[-600:], final=st in ("ok", "out_of_window", "budget_exhausted"),
                 wall_s=time.time() - t0, **extra)
        if smoke:
            return {"complete": False, "smoke": True}
    return {"complete": all(f"answer:P-large:{q['id']}" in rec.done for q in agg)}


def verdict13(cpu_result: dict[str, Any], own_records: list[dict[str, Any]], agg: list[dict[str, Any]], store=None,
              corpus_id: str | None = None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """G5 (M4) and the behavioural half of G2 over the P-large answers; G1-G4 are read from the CPU row."""
    corpus_id = corpus_id or X.CORPUS_ID
    ans = [r for r in own_records if r.get("kind") == "answer" and r.get("arm") == "P-large" and r.get("final")]
    g5 = g5_eval(ans, len(agg))
    absence = None
    if store is not None:
        absence = sql_absence(store, corpus_id, [r["sql"] for r in ans if r.get("sql") and r.get("sql_valid")])
    trips = [g for g, ok in (("G1", (cpu_result.get("g1") or {}).get("ok")), ("G2", (cpu_result.get("g2") or {}).get("ok")),
                             ("G3", (cpu_result.get("g3") or {}).get("ok")), ("G4", (cpu_result.get("g4") or {}).get("ok")))
             if ok is not True]
    if g5["complete"] and not g5["ok"]:
        trips.append("G5")
    if absence is not None and not absence["ok"]:
        trips.append("G2")
    n_dep = sum(1 for q in agg if any(e.get("doc") in REPLACED for e in (q.get("evidence") or [])))
    complete = g5["complete"]
    return {"G5": g5, "G2_behavioural": absence, "gates_tripped": sorted(set(trips)), "complete": complete,
            "scored": False, "note": "answers are not scored: golds describe the pre-replacement corpus",
            "n_questions": len(agg), "n_questions_touching_replaced": n_dep,
            "verdict": ("phase1_gate_tripped" if trips else "agg_1_3_done") if complete else "incomplete"}


def ast_l4_unit_unchanged() -> bool:
    """``phase1_arms.l4_unit`` is the AGG-1.1 function (the reuse test compares it with run_cpu's loop)."""
    return bool(X.l4_unit_ast())
