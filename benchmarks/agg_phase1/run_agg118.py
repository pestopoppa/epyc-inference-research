#!/usr/bin/env python3
"""AGG-1.18 step 2 runner: held-out verification of ``agg_pipeline.v4`` against ``agg_pipeline.v3`` (PREDECLARED_AGG118.md).

ONE ingest under v3; v4 is DERIVED OFFLINE from the same saved typed window replies (the agg118.py method): v4's L4 calls are identical
to v3's, its two changes are deterministic (the L3 text-row components ``pipeline.V4_L3_TEXTROW`` and the post-reply quote rule
``value_in_quote.rounded.v1``).

  --phase preflight   zero model: pins, PREDECLARED sha256, the 10 PDFs by sha256, the facts sheet, the pinned orchestrator's v3/v4 specs.
  --phase ingest      parse (local ODL, no OCR server) + store + L3 (v3 kwargs, fy_end) of the 10 documents (no model); then typed L4
                      under v3 (one call per union window, every reply saved in results.jsonl), pilot documents first. After the 3 pilot
                      documents: derive v4 offline on a COPY, score v3 and v4 on the pilot cells, futility look. After all 10: the
                      final derive and score -> result.cpu.json (v3 vs v4, raw + canonical, paired, long_term_debt subset).
  --dry-run           validate every input and print the plan: no model, no server, no writes.

    python -s -B run_agg118.py --phase ingest --orch-root <orch @ 88bba4fd> --out <dir> --url http://127.0.0.1:PORT

rc 0 complete (or smoke ok), 2 preflight, 3 futility / regression gate (predeclared stop), 4 precondition, 5 budget stop (re-book),
6 two or more documents failed to parse (infrastructure), 75 partial (resumable).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
D118 = HERE / "agg118"
for _p in (str(HERE.parent / "agg_phase0"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402

PREDECLARED_FILE = "PREDECLARED_AGG118.md"
PREDECLARED_SHA = "78a0a25feca4be2ed7e1dc7fb096801419756436def98ecc530b7102e20e733e"
CORPUS_ID = "agg118_heldout10"
MANIFEST_DIGEST = "3c8d2a7ec78abd6547187fde4bb557e0cc6b1a11e186b2286f21e5e6daaa3e41"
ORCH_PIN = "88bba4fd"                       # agg_pipeline.v4 (opt-in), default v3
PILOT_DOCS = ("NETFLIX_2019_10K", "SALESFORCE_2023_10K", "BOEING_2019_10K")
PILOT_WALL_S = 1800.0
BOOKED_L4_S = 3300.0
L3_FAILS_STOP = 2
FY_END_MONTHS = (1, 2, 3)
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_L3, RC_PARTIAL = 0, 2, 3, 4, 5, 6, 75
V3, V4 = "agg_pipeline.v3", "agg_pipeline.v4"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- inputs
def load_inputs(set_dir: Path = D118) -> dict[str, Any]:
    corpus = json.loads((set_dir / "corpus.json").read_text())
    rows = list(csv.DictReader((set_dir / "facts_sheet.csv").open(newline="")))
    cells = [{"doc": r["doc_name"], "field": r["metric"], "gold": float(r["value_usd_m"]), "fy": r["fiscal_year"]}
             for r in rows if r["status"] == "filled" and r["verified"] == "true"]
    return {"corpus": corpus, "rows": rows, "cells": cells, "set_dir": set_dir}


def fiscal_years(corpus: dict[str, Any]) -> dict[str, str]:
    return {d["doc_name"]: str(d["fiscal_year"]) for d in corpus["docs"]}


def doc_order(corpus: dict[str, Any]) -> list[str]:
    rest = sorted((d for d in corpus["docs"] if d["doc_name"] not in PILOT_DOCS), key=lambda d: (d["bytes"], d["doc_name"]))
    return list(PILOT_DOCS) + [d["doc_name"] for d in rest]


def ckey(c: dict[str, Any]) -> str:
    return f"{c['doc']}|{c['field']}|{c['fy']}"


def problems_of(a: argparse.Namespace, inp: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    out: list[str] = []
    sd = a.set_dir
    pd = sd / PREDECLARED_FILE
    if not pd.exists() or sha256(pd) != PREDECLARED_SHA:
        out.append(f"{PREDECLARED_FILE} missing or sha256 != {PREDECLARED_SHA[:12]}")
    c = inp["corpus"]
    if c.get("corpus_id") != CORPUS_ID or len(c["docs"]) != 10:
        out.append("corpus.json is not the agg118_heldout10 manifest of 10 documents")
    if c.get("manifest_digest_sha256") != MANIFEST_DIGEST:
        out.append("corpus manifest digest differs from the PREDECLARED one")
    if tuple(c.get("pilot_docs") or ()) != PILOT_DOCS:
        out.append("corpus pilot_docs differ from the runner's")
    for d in c["docs"]:
        p = Path(d["pdf"])
        if not p.exists() or sha256(p) != d["sha256"]:
            out.append(f"{d['doc_name']}: pdf missing or sha256 mismatch")
    st = [r["status"] for r in inp["rows"]]
    if len(inp["rows"]) != 180 or len(inp["cells"]) != st.count("filled"):
        out.append("facts sheet is not 180 rows with every filled cell verified")
    info: dict[str, Any] = {"n_cells_scored": len(inp["cells"]), "n_rows": len(inp["rows"]), "na": st.count("na"),
                            "excluded": st.count("EXCLUDED"), "ltd_cells": sum(c2["field"] == "long_term_debt" for c2 in inp["cells"]),
                            "pilot_docs": list(PILOT_DOCS), "doc_order": doc_order(c),
                            "pilot_cells": sum(c2["doc"] in PILOT_DOCS for c2 in inp["cells"])}
    if a.orch_root is None or not (a.orch_root / "src" / "aggregation" / "pipeline.py").exists():
        out.append("--orch-root is not an orchestrator tree")
        return out, info
    sys.path.insert(0, str(a.orch_root.resolve()))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    try:
        from src.aggregation import facts_det as fd, facts_typed as ft, pipeline as pl  # noqa: PLC0415
        s3, s4 = pl.pipeline_spec(V3), pl.pipeline_spec(V4)
        assert s3.l4_kwargs() == {"harvest": True} and s4.l4_kwargs() == {"harvest": True, "quote_rule": ft.VALUE_IN_QUOTE_ROUNDED}
        assert tuple(s4.l3_textrow) == tuple(pl.V4_L3_TEXTROW) and not s3.l3_textrow
        assert s3.canonical_rule == s4.canonical_rule and hasattr(fd, "fy_end_months") and hasattr(ft, "keep_slot_harvest")
        info["specs"] = {"v3": s3.as_record(), "v4": s4.as_record()}
    except Exception as e:  # noqa: BLE001
        out.append(f"pinned orchestrator lacks agg_pipeline.v3/v4: {type(e).__name__}: {e}")
    return out, info


# --------------------------------------------------------------------------- parse + store + L3 (no model)
def load_docs(rec: P.Recorder, out: Path, inp: dict[str, Any], guard: Callable[[], None], cache_dir: Path, smoke: bool) -> dict[str, Any]:
    import l3_load as L3  # noqa: PLC0415
    from src.aggregation import pipeline as pl  # noqa: PLC0415
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    from src.models.document import DocumentPreprocessResult  # noqa: PLC0415
    for k, v in L3.JOB_ENV.items():
        os.environ[k] = v
    corpus = inp["corpus"]
    by = {d["doc_name"]: d for d in corpus["docs"]}
    order = doc_order(corpus)
    if smoke:
        order = [min(order, key=lambda n: by[n]["bytes"])]
    failed = {r["doc"] for r in rec.records if r["kind"] == "doc_fail"} - {r["doc"] for r in rec.records if r["kind"] == "doc_done"}
    store = CorpusStore(out / "store.sqlite")
    try:
        for name in order:
            if f"l3:{name}" in rec.done:
                continue
            guard()
            d, t0 = by[name], time.time()
            try:
                if sha256(Path(d["pdf"])) != d["sha256"]:
                    raise ValueError("pdf sha256 mismatch")
                cache = cache_dir / f"{name}.{d['sha256'][:12]}.json"
                if cache.exists():
                    result, cached = DocumentPreprocessResult.from_cache_dict(json.loads(cache.read_text())), True
                else:
                    result, cached = L3.real_parse(Path(d["pdf"])), False
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    X._atomic_write(cache, json.dumps(result.to_cache_dict(), default=str))
                rc = store.ingest(result, corpus_id=CORPUS_ID, doc_id=name, entity=d["company"])
                rec.emit("parse", doc=name, cached=cached, parse_s=time.time() - t0, total_pages=result.total_pages,
                         n_sections=len(result.sections), n_tables=len(result.tables), failed_pages=len(result.failed_pages),
                         n_chunks=rc.n_chunks, llm_calls=0)
                rec.emit("doc_done", unit=f"l3:{name}", doc=name, pdf_sha256=d["sha256"], wall_s=time.time() - t0, final=True)
                failed.discard(name)
            except P.Stop:
                raise
            except Exception as e:  # noqa: BLE001 - recorded per document
                rec.emit("doc_fail", unit=f"l3:{name}", doc=name, error=f"{type(e).__name__}: {e}"[:500], wall_s=time.time() - t0, final=True)
                failed.add(name)
            if len(failed) >= L3_FAILS_STOP:
                rec.emit("stop", reason="l3_parse_futility", failed=sorted(failed), final=False)
                return {"stop": RC_L3, "failed": sorted(failed)}
        if "l3v3" not in rec.done:
            spec = pl.pipeline_spec(V3)
            facts = spec.extract_l3(store, CORPUS_ID, fiscal_years(corpus), FY_END_MONTHS)
            store.write_facts(facts)
            store.record_index(CORPUS_ID, store.corpus_content_hash(CORPUS_ID), "l3_facts_det", "", layer="L3", n_items=len(facts))
            store.conn.commit()
            rec.emit("l3v3", unit="l3v3", pipeline=spec.as_record(), fy_end=list(FY_END_MONTHS), n_l3_facts=len(facts), llm_calls=0,
                     final=True)
        return {"failed": sorted(failed), "order": [n for n in order if n not in failed]}
    finally:
        store.close()


# --------------------------------------------------------------------------- v3 L4 finish and the offline v4 derivation
def finish_v3(rec: P.Recorder, tb: Any, doc: str, plan: dict[str, Any]) -> None:
    import agg112 as G  # noqa: PLC0415
    from src.aggregation import facts_typed as ft  # noqa: PLC0415
    byw = G.windows_records(rec)
    recs = [byw[G.window_unit(doc, w["chunk_id"])] for w in plan["windows"]]
    kept = [f for r, w in zip(recs, plan["windows"]) for f in r["facts"]
            if ft.keep_slot_harvest(f, set(plan["chosen"][w["chunk_id"]]), ft.SCHEMA_FIELDS)]
    res = tb.finish_doc(CORPUS_ID, kept, ft.SCHEMA_FIELDS)        # v3: quote rule value_in_quote.v1 (default)
    rec.emit("doc_l4", unit=f"doc_l4:{doc}", doc=doc, n_windows=len(recs), llm_calls=sum(r["calls"] for r in recs),
             prefill_tokens=sum(r["prefill_tokens"] for r in recs), wall_s=sum(r["wall_s"] for r in recs),
             bad_replies=sum(1 for r in recs if r.get("bad")), n_facts_in=len(kept), n_facts_kept=len(res["facts"]),
             n_quote_dropped=len(res.get("quote_dropped", [])), final=True)


def score(path: Path, cells: list[dict[str, Any]], spec: Any) -> dict[str, dict[str, bool]]:
    import agg116b as B  # noqa: PLC0415
    st = P.open_store(path)
    try:
        spec.refresh_canonical(st, CORPUS_ID)
        st.conn.commit()
    finally:
        st.close()
    s = B.score(path, cells)
    return {"raw": {k: bool(v["raw_hit"]) for k, v in s.items()}, "canonical": {k: bool(v["canonical_ok"]) for k, v in s.items()}}


def derive_v4(out: Path, rec: P.Recorder, docs: list[str], plans: dict[str, Any], tag: str) -> dict[str, Any]:
    """Copy the v3 store twice (v3 scoring copy, v4 copy); on the v4 copy apply the v4 L3 text-row delta and the v4 quote-rule replay of the
    saved replies (agg118.apply_quote). Parity: the v3 L3 kwargs must reproduce the stored L3 rows and the v1-rule replay only stored fact ids."""
    import agg117 as A117  # noqa: PLC0415
    import agg118 as A118  # noqa: PLC0415
    from src.aggregation import facts_det as fd, facts_typed as ft, pipeline as pl  # noqa: PLC0415
    s3, s4 = pl.pipeline_spec(V3), pl.pipeline_spec(V4)
    fy = fiscal_years(json.loads((D118 / "corpus.json").read_text()))
    p3, p4 = out / f"store.{tag}.v3.sqlite", out / f"store.{tag}.v4.sqlite"
    for p in (p3, p4):
        if p.exists():
            p.unlink()
        P.copy_store(out / "store.sqlite", p)
    st = P.open_store(p4)
    try:
        stored = A117.stored_l3(st)
        base = {A117.sig(f): f for f in fd.extract_corpus_facts(st, CORPUS_ID, **s3.l3_kwargs(fy, FY_END_MONTHS))}
        new = {A117.sig(f): f for f in fd.extract_corpus_facts(st, CORPUS_ID, **s4.l3_kwargs(fy, FY_END_MONTHS))}
        l3_parity = set(base) == stored
        gone = sorted({base[k]["fact_id"] for k in set(base) - set(new)})
        for i in range(0, len(gone), 500):
            part = gone[i:i + 500]
            st.conn.execute(f"DELETE FROM facts WHERE fact_id IN ({','.join('?' * len(part))})", part)
        st.write_facts([new[k] for k in sorted(set(new) - set(base))])
        st.conn.commit()
        textrow = {"removed": len(gone), "added": len(set(new) - set(base))}
    finally:
        st.close()
    byw = {}
    for r in rec.records:
        if r.get("kind") == "t_window" and r.get("final") and r.get("doc") in docs:
            byw.setdefault(r["doc"], {"doc": r["doc"], "recs": [], "planned": set(ft.SCHEMA_FIELDS)})["recs"].append(r)
    quote = A118.apply_quote(ft, P, p4, byw)
    return {"p3": p3, "p4": p4, "l3_parity": l3_parity, "textrow": textrow, "quote": {k: v for k, v in quote.items() if k != "added"},
            "replayed_not_stored": quote["replayed_not_stored"], "s3": s3, "s4": s4}


def sign_test(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(cells: list[dict[str, Any]], d: dict[str, Any]) -> dict[str, Any]:
    r3, r4 = score(d["p3"], cells, d["s3"]), score(d["p4"], cells, d["s4"])
    out: dict[str, Any] = {}
    for m in ("raw", "canonical"):
        for sub, cs in (("all", cells), ("long_term_debt", [c for c in cells if c["field"] == "long_term_debt"])):
            ks = [ckey(c) for c in cs]
            fixed = sorted(k for k in ks if r4[m][k] and not r3[m][k])
            reg = sorted(k for k in ks if r3[m][k] and not r4[m][k])
            out[f"{m}_{sub}"] = {"n": len(ks), "v3": sum(r3[m][k] for k in ks), "v4": sum(r4[m][k] for k in ks),
                                 "fixed": fixed, "regressed": reg, "sign_test_p": sign_test(len(fixed), len(reg))}
    return out


# --------------------------------------------------------------------------- ingest
def run_ingest(rec: P.Recorder, out: Path, inp: dict[str, Any], backend: Any, guard: Callable[[], None], *, smoke: bool,
               cache_dir: Path, booked_s: float) -> dict[str, Any]:
    import agg112 as G  # noqa: PLC0415
    import agg16 as G16  # noqa: PLC0415
    from src.aggregation import facts_typed as ft  # noqa: PLC0415
    ld = load_docs(rec, out, inp, guard, cache_dir, smoke)
    if "stop" in ld:
        return {"complete": True, "infrastructure_stop": "l3_parse_futility", "failed": ld["failed"], "rc": ld["stop"]}
    order, failed = ld["order"], set(ld["failed"])
    cells = [c for c in inp["cells"] if c["doc"] not in failed]
    store = P.open_store(out / "store.sqlite")
    try:
        corpus = inp["corpus"]
        plans = {k: v for k, v in G.doc_plans(store, {**corpus, "docs": [d for d in corpus["docs"] if d["doc_name"] in order]},
                                                CORPUS_ID).items()}
        if "plan" not in rec.done:
            rec.emit("plan", unit="plan", n_windows={k: v["n_windows"] for k, v in plans.items()},
                     llm_calls_planned=sum(v["n_windows"] for v in plans.values()), final=True)
        rb = G16.RecordingBackend(backend)
        tb = ft.TypedBackfill(store, rb)
        fy = fiscal_years(corpus)
        pilot = [d for d in PILOT_DOCS if d in order]
        for doc in order:
            plan, descs = plans[doc], G.descs_for(fy[doc])
            for w in plan["windows"]:
                if G.window_unit(doc, w["chunk_id"]) in rec.done:
                    continue
                if doc in pilot and "pilot_done" not in rec.done and not smoke:
                    used = sum(r["wall_s"] for r in rec.records if r.get("kind") == "t_window" and r.get("final") and r.get("doc") in pilot)
                    if used > PILOT_WALL_S:
                        rec.emit("budget_stop", reason="pilot_wall_over_1800s", wall_s=used, final=False)
                        return {"complete": False, "budget_stop": "pilot_wall_over_1800s", "rc": RC_BUDGET}
                G.run_window(rec, tb, rb, doc, w, plan["chosen"][w["chunk_id"]], descs, guard)
                if smoke:
                    return {"complete": False, "smoke": True, "rc": RC_OK}
                pd = next((r for r in rec.records if r.get("unit") == "pilot_done"), None)
                if pd and doc not in pilot:
                    fresh = sum(r["wall_s"] for r in rec.records if r.get("kind") == "t_window" and r.get("final") and r.get("doc") not in pilot)
                    if fresh > pd["hard_stop_s"]:
                        rec.emit("budget_stop", reason="l4_wall_over_projection", wall_s=fresh, hard_stop_s=pd["hard_stop_s"], final=False)
                        return {"complete": False, "budget_stop": "l4_wall_over_projection", "rc": RC_BUDGET}
            if not G.doc_windows_done(rec, doc, plan):
                return {"complete": False, "reason": f"{doc}: windows errored; a re-grant retries them", "rc": RC_PARTIAL}
            if f"doc_l4:{doc}" not in rec.done:
                finish_v3(rec, tb, doc, plan)
                store.conn.commit()
            if "pilot_done" not in rec.done and all(f"doc_l4:{d}" in rec.done for d in pilot):
                res = pilot_look(rec, out, cells, pilot, plans, booked_s)
                if res:
                    return res
        return final_look(rec, out, cells, order, plans, failed)
    finally:
        store.close()


def pilot_look(rec: P.Recorder, out: Path, cells: list[dict[str, Any]], pilot: list[str], plans: dict[str, Any],
               booked_s: float) -> dict[str, Any] | None:
    pc = [c for c in cells if c["doc"] in pilot]
    d = derive_v4(out, rec, pilot, plans, "pilot")
    cmp = compare(pc, d)
    wall = sum(r["wall_s"] for r in rec.records if r.get("kind") == "t_window" and r.get("final") and r.get("doc") in pilot)
    win_pilot = sum(plans[x]["n_windows"] for x in pilot)
    win_rest = sum(v["n_windows"] for k, v in plans.items() if k not in pilot)
    projected = wall / win_pilot * win_rest if win_pilot else 0.0
    gates = []
    if not d["l3_parity"] or d["replayed_not_stored"]:
        gates.append("derivation_parity_failed")
    if any(cmp[k]["regressed"] for k in cmp):
        gates.append("v4_regression_in_pilot")
    if not any(cmp[k]["fixed"] or cmp[k]["regressed"] for k in cmp):
        gates.append("v4_equals_v3_on_every_pilot_cell")
    budget = ["projected_l4_over_2x_booked"] if wall + projected > 2 * booked_s else []
    counts = {k: {"n": v["n"], "v3": v["v3"], "v4": v["v4"], "fixed": len(v["fixed"]), "regressed": len(v["regressed"])} for k, v in cmp.items()}
    ev = {"pilot_docs": pilot, "pilot_cells": len(pc), "counts": counts, "l3_parity": d["l3_parity"],
          "replayed_not_stored": d["replayed_not_stored"], "pilot_wall_s": wall, "projected_rest_s": projected,
          "hard_stop_s": 1.5 * projected if projected else 0.0, "gates_tripped": gates, "budget_stops": budget}
    rec.emit("pilot_done", unit="pilot_done", **ev)
    X._atomic_write(out / "result.pilot.json", json.dumps({"phase": "pilot", "complete": True, "predeclared_sha256": PREDECLARED_SHA, **ev},
                                                           indent=1) + "\n")
    if gates:
        X._atomic_write(out / "result.cpu.json", json.dumps({"phase": "cpu", "complete": True, "gate_stop": "pilot", "gates_tripped": gates,
                                                             "pilot": ev, "pilot_detail": cmp, "predeclared_sha256": PREDECLARED_SHA},
                                                            indent=1) + "\n")
        return {"complete": True, "gate_tripped": gates, "rc": RC_GATE}
    if budget:
        return {"complete": False, "budget_stop": budget, "rc": RC_BUDGET}
    return None


def final_look(rec: P.Recorder, out: Path, cells: list[dict[str, Any]], order: list[str], plans: dict[str, Any],
               failed: set[str]) -> dict[str, Any]:
    missing = [d for d in order if f"doc_l4:{d}" not in rec.done]
    if missing:
        return {"complete": False, "reason": f"documents incomplete: {missing}", "rc": RC_PARTIAL}
    d = derive_v4(out, rec, order, plans, "final")
    cmp = compare(cells, d)
    l4 = [r for r in rec.records if r.get("kind") == "doc_l4"]
    res = {"phase": "cpu", "complete": True, "predeclared_sha256": PREDECLARED_SHA, "corpus_id": CORPUS_ID, "docs": order,
           "doc_fail": sorted(failed), "n_cells": len(cells), "llm_calls": sum(r["llm_calls"] for r in l4),
           "prefill_tokens": sum(r["prefill_tokens"] for r in l4), "l4_wall_s": sum(r["wall_s"] for r in l4),
           "l3_parity": d["l3_parity"], "replayed_not_stored": d["replayed_not_stored"], "textrow": d["textrow"], "quote": d["quote"],
           "v3_vs_v4": cmp, "regressions": sorted({k for v in cmp.values() for k in v["regressed"]}),
           "label": "HELD-OUT (10 FinanceBench 10-Ks never used in AGG-1.x design); v4 derived offline from the v3 replies"}
    res["verdict"] = "REGRESSIONS" if res["regressions"] else "0_regressions"
    X._atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    rec.emit("final", unit="final", **{k: res[k] for k in ("n_cells", "llm_calls", "verdict", "regressions")},
             totals={k: {"v3": v["v3"], "v4": v["v4"], "n": v["n"]} for k, v in cmp.items()})
    return {"complete": True, "rc": RC_GATE if res["regressions"] else RC_OK, "verdict": res["verdict"]}


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("preflight", "ingest"), required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-dir", type=Path, default=D118)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--orch-root", type=Path)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--parse-cache", type=Path)
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--booked-l4-s", type=float, default=BOOKED_L4_S)
    ap.add_argument("--smoke", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    X.configure(CORPUS_ID)
    try:
        inp = load_inputs(a.set_dir)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load inputs: {type(e).__name__}: {e}"]}))
        return RC_PREFLIGHT
    problems, info = problems_of(a, inp)
    if a.orch_pin and not a.orch_pin.startswith(ORCH_PIN):
        problems.append(f"--orch-pin {a.orch_pin} is not {ORCH_PIN}")
    if a.dry_run or a.phase == "preflight":
        print(json.dumps({"dry_run": a.dry_run, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
                          "predeclared_sha256": PREDECLARED_SHA, "corpus_id": CORPUS_ID, "orch_pin": a.orch_pin,
                          "research_pin": a.research_pin, "problems": problems, "ok": not problems, **info}, indent=1, default=str))
        return RC_PREFLIGHT if problems else RC_OK
    if a.out is None or not a.url:
        build_parser().error("--out and --url are required for ingest")
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    if problems:
        rec.emit("preflight", ok=False, problems=problems, final=False)
        return RC_PREFLIGHT
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=PREDECLARED_SHA, orch_pin=a.orch_pin, research_pin=a.research_pin,
             corpus_id=CORPUS_ID, url=a.url, model_id=a.model_id, arm=V3, derived=V4, final=False)
    from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
    try:
        res = run_ingest(rec, a.out, inp, LlamaServerBackend(a.url, model_id=a.model_id), guard, smoke=smoke,
                         cache_dir=a.parse_cache or (a.out / "cache"), booked_s=a.booked_l4_s)
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return RC_PARTIAL
    rc = res.pop("rc", RC_PARTIAL)
    print(json.dumps(res, indent=1, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
