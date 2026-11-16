#!/usr/bin/env python3
"""AGG-1.19 step 2 runner (agg118/PREDECLARED_AGG119.md): held-out check of ``agg_pipeline.v5`` (and v4) against ``agg_pipeline.v3`` on the
7 UNSEEN AGG-1.18 held-out 10-Ks (MGM, American Water, Corning, CVS, PepsiCo, Verizon, FedEx; golds in agg118/facts_sheet.csv).

ONE ingest under v3 (parse + store + L3 with v3 kwargs and fy_end, typed L4, every reply saved); v4 and v5 are DERIVED OFFLINE from the same
replies: each spec's L3 text-row delta (re-extraction with that spec's kwargs) plus the ``value_in_quote.rounded.v1`` replay
(agg118.apply_quote). v4 and v5 make no extra model calls. Parse, L4 windows, recorder and budget logic are run_agg118's, reused with
this run's corpus id, documents and pilot.

  pilot   VERIZON_2019_10K, CORNING_2019_10K (the 2 smallest of the 7). STOP-ON-FUTILITY (rc 3, result.cpu.json filed): v5 fixes 0 pilot
          cells vs v3 (raw and canonical), or any v5 regression vs v3, or a derivation parity failure.
  final   v3 / v4 / v5 table, raw + canonical, paired (v4 vs v3, v5 vs v3, v5 vs v4) -> result.cpu.json.

    python -s -B run_agg119.py --phase ingest --orch-root <orch @ e1ccd216> --out <dir> --url http://127.0.0.1:PORT

rc 0 complete (or smoke ok), 2 preflight, 3 gate, 5 budget stop, 6 parse infrastructure, 75 partial.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_agg118 as R  # noqa: E402

PREDECLARED_AGG119 = R.D118 / "PREDECLARED_AGG119.md"
PREDECLARED_AGG119_SHA = "d41a9d284e10935e27f2d930847d63cbf89710e1d618569864b280208cb670a4"
CORPUS_ID = "agg119_heldout7"
DOCS7 = ("MGMRESORTS_2019_10K", "AMERICANWATERWORKS_2019_10K", "CORNING_2019_10K", "CVSHEALTH_2019_10K", "PEPSICO_2019_10K",
         "VERIZON_2019_10K", "FEDEX_2023_10K")
PILOT_DOCS = ("VERIZON_2019_10K", "CORNING_2019_10K")
ORCH_PIN = "e1ccd216"
BOOKED_L4_S = 2000.0
PILOT_WALL_S = 900.0
V3, V4, V5 = "agg_pipeline.v3", "agg_pipeline.v4", "agg_pipeline.v5"

# run_agg118's ingest machinery reads these module globals at call time
R.CORPUS_ID, R.PILOT_DOCS, R.PILOT_WALL_S, R.BOOKED_L4_S, R.ORCH_PIN = CORPUS_ID, PILOT_DOCS, PILOT_WALL_S, BOOKED_L4_S, ORCH_PIN
_load_inputs_118, _problems_118 = R.load_inputs, R.problems_of


def load_inputs(set_dir: Path = R.D118) -> dict[str, Any]:
    inp = _load_inputs_118(set_dir)
    c = dict(inp["corpus"])
    c["docs"] = [d for d in c["docs"] if d["doc_name"] in DOCS7]
    inp["corpus118"], inp["corpus"] = inp["corpus"], c
    inp["cells"] = [x for x in inp["cells"] if x["doc"] in DOCS7]
    return inp


def problems_of(a: Any, inp: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    full = dict(inp, corpus=inp["corpus118"], cells=_load_inputs_118(a.set_dir)["cells"])
    saved = R.CORPUS_ID, R.PILOT_DOCS
    R.CORPUS_ID, R.PILOT_DOCS = "agg118_heldout10", ("NETFLIX_2019_10K", "SALESFORCE_2023_10K", "BOEING_2019_10K")
    try:
        out, info = _problems_118(a, full)          # the 10-doc manifest, PDFs, facts sheet, AGG118 predeclaration
    finally:
        R.CORPUS_ID, R.PILOT_DOCS = saved
    out = [p for p in out if "agg_pipeline.v3/v4" not in p]
    if not PREDECLARED_AGG119.exists() or R.sha256(PREDECLARED_AGG119) != PREDECLARED_AGG119_SHA:
        out.append(f"PREDECLARED_AGG119.md missing or sha256 != {PREDECLARED_AGG119_SHA[:12]}")
    if sorted(d["doc_name"] for d in inp["corpus"]["docs"]) != sorted(DOCS7):
        out.append("the 7 held-out documents are not all in the manifest")
    try:
        from src.aggregation import facts_det as fd, facts_typed as ft, pipeline as pl  # noqa: PLC0415
        s3, s4, s5 = (pl.pipeline_spec(v) for v in (V3, V4, V5))
        assert s5.l3_textrow == s4.l3_textrow + tuple(fd.L3_AGG119_OPTIONAL) and s5.l4_kwargs() == s4.l4_kwargs()
        assert s4.l4_kwargs() == {"harvest": True, "quote_rule": ft.VALUE_IN_QUOTE_ROUNDED} and pl.DEFAULT == V3
        info["specs"] = {"v3": s3.as_record(), "v4": s4.as_record(), "v5": s5.as_record()}
    except Exception as e:  # noqa: BLE001
        out.append(f"pinned orchestrator lacks agg_pipeline.v5: {type(e).__name__}: {e}")
    info.update({"corpus_id": CORPUS_ID, "docs": list(DOCS7), "pilot_docs": list(PILOT_DOCS), "doc_order": R.doc_order(inp["corpus"]),
                 "n_cells_scored": len(inp["cells"]), "pilot_cells": sum(c["doc"] in PILOT_DOCS for c in inp["cells"]),
                 "ltd_cells": sum(c["field"] == "long_term_debt" for c in inp["cells"])})
    return out, info


def derive(out: Path, rec: Any, docs: list[str], tag: str) -> dict[str, Any]:
    """v3 scoring copy + one derived copy per spec (v4, v5): L3 delta + quote-rule replay. Parity as run_agg118.derive_v4."""
    import agg117 as A117  # noqa: PLC0415
    import agg118 as A118  # noqa: PLC0415
    from src.aggregation import facts_det as fd, facts_typed as ft, pipeline as pl  # noqa: PLC0415
    fy = R.fiscal_years(json.loads((R.D118 / "corpus.json").read_text()))
    specs = {v: pl.pipeline_spec(v) for v in (V3, V4, V5)}
    paths = {v: out / f"store.{tag}.{v.rsplit('.', 1)[1]}.sqlite" for v in specs}
    for p in paths.values():
        if p.exists():
            p.unlink()
        R.P.copy_store(out / "store.sqlite", p)
    byw: dict[str, Any] = {}
    for r in rec.records:
        if r.get("kind") == "t_window" and r.get("final") and r.get("doc") in docs:
            byw.setdefault(r["doc"], {"doc": r["doc"], "recs": [], "planned": set(ft.SCHEMA_FIELDS)})["recs"].append(r)
    res: dict[str, Any] = {"paths": paths, "specs": specs, "textrow": {}, "quote": {}, "l3_parity": True, "replayed_not_stored": 0}
    for v in (V4, V5):
        st = R.P.open_store(paths[v])
        try:
            stored = A117.stored_l3(st)
            base = {A117.sig(f): f for f in fd.extract_corpus_facts(st, CORPUS_ID, **specs[V3].l3_kwargs(fy, R.FY_END_MONTHS))}
            new = {A117.sig(f): f for f in fd.extract_corpus_facts(st, CORPUS_ID, **specs[v].l3_kwargs(fy, R.FY_END_MONTHS))}
            res["l3_parity"] = res["l3_parity"] and set(base) == stored
            gone = sorted({base[k]["fact_id"] for k in set(base) - set(new)})
            for i in range(0, len(gone), 500):
                part = gone[i:i + 500]
                st.conn.execute(f"DELETE FROM facts WHERE fact_id IN ({','.join('?' * len(part))})", part)
            st.write_facts([new[k] for k in sorted(set(new) - set(base))])
            st.conn.commit()
            res["textrow"][v] = {"removed": len(gone), "added": len(set(new) - set(base))}
        finally:
            st.close()
        q = A118.apply_quote(ft, R.P, paths[v], byw)
        res["quote"][v] = {k: x for k, x in q.items() if k != "added"}
        res["replayed_not_stored"] += q["replayed_not_stored"]
    return res


def compare(cells: list[dict[str, Any]], d: dict[str, Any]) -> dict[str, Any]:
    sc = {v: R.score(d["paths"][v], cells, d["specs"][v]) for v in (V3, V4, V5)}
    out: dict[str, Any] = {}
    for a, b in ((V3, V4), (V3, V5), (V4, V5)):
        pair = f"{b.rsplit('.', 1)[1]}_vs_{a.rsplit('.', 1)[1]}"
        for m in ("raw", "canonical"):
            for sub, cs in (("all", cells), ("long_term_debt", [c for c in cells if c["field"] == "long_term_debt"])):
                ks = [R.ckey(c) for c in cs]
                fixed = sorted(k for k in ks if sc[b][m][k] and not sc[a][m][k])
                reg = sorted(k for k in ks if sc[a][m][k] and not sc[b][m][k])
                out[f"{pair}:{m}_{sub}"] = {"n": len(ks), "before": sum(sc[a][m][k] for k in ks), "after": sum(sc[b][m][k] for k in ks),
                                            "fixed": fixed, "regressed": reg, "sign_test_p": R.sign_test(len(fixed), len(reg))}
    return out


def pilot_look(rec: Any, out: Path, cells: list[dict[str, Any]], pilot: list[str], plans: dict[str, Any],
               booked_s: float) -> dict[str, Any] | None:
    pc = [c for c in cells if c["doc"] in pilot]
    d = derive(out, rec, pilot, "pilot")
    cmp = compare(pc, d)
    wall = sum(r["wall_s"] for r in rec.records if r.get("kind") == "t_window" and r.get("final") and r.get("doc") in pilot)
    win_pilot = sum(plans[x]["n_windows"] for x in pilot)
    win_rest = sum(v["n_windows"] for k, v in plans.items() if k not in pilot)
    projected = wall / win_pilot * win_rest if win_pilot else 0.0
    v5 = {k: v for k, v in cmp.items() if k.startswith("v5_vs_v3:")}
    gates = []
    if not d["l3_parity"] or d["replayed_not_stored"]:
        gates.append("derivation_parity_failed")
    if any(v["regressed"] for v in v5.values()):
        gates.append("v5_regression_in_pilot")
    if not any(v["fixed"] for v in v5.values()):
        gates.append("v5_fixes_0_pilot_cells_vs_v3")
    budget = ["projected_l4_over_2x_booked"] if wall + projected > 2 * booked_s else []
    counts = {k: {"n": v["n"], "before": v["before"], "after": v["after"], "fixed": len(v["fixed"]), "regressed": len(v["regressed"])}
              for k, v in cmp.items()}
    ev = {"pilot_docs": pilot, "pilot_cells": len(pc), "counts": counts, "l3_parity": d["l3_parity"],
          "replayed_not_stored": d["replayed_not_stored"], "pilot_wall_s": wall, "projected_rest_s": projected,
          "hard_stop_s": 1.5 * projected if projected else 0.0, "gates_tripped": gates, "budget_stops": budget}
    rec.emit("pilot_done", unit="pilot_done", **ev)
    R.X._atomic_write(out / "result.pilot.json", json.dumps({"phase": "pilot", "complete": True,
                                                             "predeclared_sha256": PREDECLARED_AGG119_SHA, **ev}, indent=1) + "\n")
    if gates:
        R.X._atomic_write(out / "result.cpu.json", json.dumps({"phase": "cpu", "complete": True, "gate_stop": "pilot", "gates_tripped": gates,
                                                               "pilot": ev, "pilot_detail": cmp,
                                                               "predeclared_sha256": PREDECLARED_AGG119_SHA}, indent=1) + "\n")
        return {"complete": True, "gate_tripped": gates, "rc": R.RC_GATE}
    if budget:
        return {"complete": False, "budget_stop": budget, "rc": R.RC_BUDGET}
    return None


def final_look(rec: Any, out: Path, cells: list[dict[str, Any]], order: list[str], plans: dict[str, Any],
               failed: set[str]) -> dict[str, Any]:
    missing = [d for d in order if f"doc_l4:{d}" not in rec.done]
    if missing:
        return {"complete": False, "reason": f"documents incomplete: {missing}", "rc": R.RC_PARTIAL}
    d = derive(out, rec, order, "final")
    cmp = compare(cells, d)
    l4 = [r for r in rec.records if r.get("kind") == "doc_l4"]
    regs = sorted({k for p, v in cmp.items() if p.startswith("v5_vs_v3:") for k in v["regressed"]})
    res = {"phase": "cpu", "complete": True, "predeclared_sha256": PREDECLARED_AGG119_SHA, "corpus_id": CORPUS_ID, "docs": order,
           "doc_fail": sorted(failed), "n_cells": len(cells), "llm_calls": sum(r["llm_calls"] for r in l4),
           "prefill_tokens": sum(r["prefill_tokens"] for r in l4), "l4_wall_s": sum(r["wall_s"] for r in l4),
           "l3_parity": d["l3_parity"], "replayed_not_stored": d["replayed_not_stored"], "textrow": d["textrow"], "quote": d["quote"],
           "pairs": cmp, "v5_regressions_vs_v3": regs,
           "label": "HELD-OUT (7 AGG-1.18 held-out 10-Ks, unseen by AGG-1.18 and AGG-1.19 design); v4 and v5 derived offline from the v3 replies"}
    res["verdict"] = "REGRESSIONS" if regs else "0_regressions"
    R.X._atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    rec.emit("final", unit="final", **{k: res[k] for k in ("n_cells", "llm_calls", "verdict")}, regressions=regs,
             totals={k: {"before": v["before"], "after": v["after"], "n": v["n"]} for k, v in cmp.items()})
    return {"complete": True, "rc": R.RC_GATE if regs else R.RC_OK, "verdict": res["verdict"]}


R.load_inputs, R.problems_of, R.pilot_look, R.final_look = load_inputs, problems_of, pilot_look, final_look

if __name__ == "__main__":
    sys.exit(R.main())
