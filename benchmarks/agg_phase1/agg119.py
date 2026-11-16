#!/usr/bin/env python3
"""AGG-1.19 (agg118/PREDECLARED_AGG119.md): the opt-in L3 components ``fy_end_lone_monthday`` and ``fy1_captions``, replayed OFFLINE.
No model, no server, no compute row.

The agg117 method, extended with the AGG-1.18 pilot store: for every store (AGG-1.13 = 76 + 14 LTD, AGG-1.12 r2 = 211, AGG-1.11 s2
agg13 / agg14 = 49 x 2, AGG-1.18 pilot = 48), a COPY at the ``agg_pipeline.v3`` state is scored (BEFORE); its L3 facts are re-extracted
from the stored pages with the kwargs that reproduce the stored L3 rows (parity), then with a variant's components added; the
fact-signature delta is applied to a fresh copy, re-canonicalised under canonical.v3 and scored (AFTER). Raw and canonical, every set.
A cell right BEFORE and wrong AFTER is a regression. On the pilot store only the 3 pilot documents are re-extracted and scored; the 7
held-out documents stored alongside them are excluded from every delta and never scored (they stay unseen for step 2).

Variants: each component alone, both (the proposed bundle), the v4 text-row components (V4_L3_TEXTROW), and v4 + both (``agg_pipeline.v5``
L3). The v5 rows are compared against v3 AND against v4. The quote rule of v4 is an L4 replay and is not re-run here.

``fy_end_stress``: on every store, the extra fact-period changes from adding ``fy_end_lone_monthday`` on top of the parity kwargs WITH fy_end
on (the stores ingested without fy_end are otherwise a no-op for the component), listed per document.

    python3 agg119.py --orch-root <orchestrator worktree with the components> --scratch <NEW dir> --out agg119_result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE), str(HERE / "agg113")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PREDECLARED = HERE / "agg118" / "PREDECLARED_AGG119.md"
PILOT_STORE = Path("/mnt/raid0/llm/tmp/agg118-ingest-cpu-20261116/out/store.pilot.v3.sqlite")
PILOT_DOCS = ("NETFLIX_2019_10K", "SALESFORCE_2023_10K", "BOEING_2019_10K")
TARGET12 = ("SALESFORCE_2023_10K|capex|2021", "SALESFORCE_2023_10K|capex|2022", "SALESFORCE_2023_10K|net_income|2021",
            "SALESFORCE_2023_10K|net_income|2022", "SALESFORCE_2023_10K|operating_cash_flow|2021",
            "SALESFORCE_2023_10K|operating_cash_flow|2022", "SALESFORCE_2023_10K|revenue|2021", "SALESFORCE_2023_10K|revenue|2022",
            "BOEING_2019_10K|capex|2018", "BOEING_2019_10K|net_income|2018", "BOEING_2019_10K|operating_cash_flow|2018",
            "NETFLIX_2019_10K|operating_cash_flow|2018")
BOEING_NI19 = "BOEING_2019_10K|net_income|2019"
LONE, FY1 = "fy_end_lone_monthday", "fy1_captions"
FUTILITY_MIN_FIXED = 6


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
    import agg113 as H  # noqa: PLC0415
    import agg115 as A115  # noqa: PLC0415
    import agg116b as B  # noqa: PLC0415
    import agg117 as A117  # noqa: PLC0415
    import agg_l3period as L3P  # noqa: PLC0415
    import csv  # noqa: PLC0415
    import phase0_arms as P  # noqa: PLC0415
    import phase1_arms as X  # noqa: PLC0415
    import run_agg118 as R  # noqa: PLC0415
    X.configure(G112.CORPUS_ID)
    from src.aggregation import facts_det as fd, pipeline as pl  # noqa: PLC0415
    from src.aggregation.store import CANONICAL_RULE_V3  # noqa: PLC0415
    assert set(fd.L3_AGG119_OPTIONAL) == {LONE, FY1}
    ft = G112.orch_symbols()["ft"]
    a.scratch.mkdir(parents=True, exist_ok=False)
    cid = G112.CORPUS_ID
    spec3 = pl.pipeline_spec(pl.PIPELINE_V3)
    inp113, inp112 = H.load_inputs(H.D113), G112.load_inputs(HERE)
    cells49 = S2.load_cells(HERE)
    rows = list(csv.DictReader(A117.GOLDS14.open(newline="")))
    cells14 = [{"doc": r["doc_name"], "field": r["metric"], "gold": float(r["value_usd_m"]), "fy": r["fiscal_year"]}
               for r in rows if r["status"] == "filled"]
    inp118 = R.load_inputs()
    cells48 = [c for c in inp118["cells"] if c["doc"] in PILOT_DOCS]
    assert len(cells48) == 48, len(cells48)
    ctx = {"cid": cid, "spec": spec3, "v3rule": CANONICAL_RULE_V3, "cells14": cells14,
           "cells": {"agg113": inp113["cells"], "r2": inp112["cells"], "agg13": cells49["agg13"], "agg14": cells49["agg14"]}}
    fys = {"agg113": L3P.fiscal_years(inp113["corpus"]), "r2": L3P.fiscal_years(inp112["corpus"]),
           "pilot": R.fiscal_years(inp118["corpus"])}
    fys["agg13"] = fys["agg14"] = fys["r2"]
    cids = {n: cid for n in ("agg113", "r2", "agg13", "agg14")} | {"pilot": R.CORPUS_ID}
    all_pilot_store_docs = {d["doc_name"] for d in inp118["corpus"]["docs"]}
    unseen = sorted(all_pilot_store_docs - set(PILOT_DOCS))

    def score(n: str, path: Path) -> dict[str, dict[str, bool]]:
        if n != "pilot":
            return A117.score_set(n, path, ctx)
        st = P.open_store(path)
        try:
            spec3.refresh_canonical(st, R.CORPUS_ID)
            st.conn.commit()
        finally:
            st.close()
        s = B.score(path, cells48)
        return {"raw": {k: bool(v["raw_hit"]) for k, v in s.items()}, "canonical": {k: bool(v["canonical_ok"]) for k, v in s.items()},
                "_detail": {k: {"raw_verdict": v["raw_verdict"], "canonical_value": (v["canonical_row"] or {}).get("value"),
                                "canonical_quote": ((v["canonical_row"] or {}).get("quote") or "")[:100]} for k, v in s.items()}}

    # 0. v3-state base copies (same construction as agg117) + the pilot store copy
    base: dict[str, Path] = {}
    base["agg113"] = a.scratch / "agg113_base.sqlite"
    P.copy_store(A117.STORE113, base["agg113"])
    base["r2"] = a.scratch / "r2_base.sqlite"
    A115.delta_store(ft, A117.R2, base["r2"], A115.ingest_plans(A117.R2_INGEST), cid)
    for name in ("agg13", "agg14"):
        base[name] = a.scratch / f"{name}_base.sqlite"
        A115.delta_store(ft, A117.JOB / "out" / f"store_{name}.sqlite", base[name], A115.s2_plans(A117.S2_RES, name), cid)
    base["pilot"] = a.scratch / "pilot_base.sqlite"
    P.copy_store(PILOT_STORE, base["pilot"])
    before = {n: score(n, p) for n, p in base.items()}

    # 1. parity
    parity: dict[str, Any] = {}
    for n, p in base.items():
        st = P.open_store(p)
        try:
            got = A117.find_parity(fd, st, cids[n], fys[n])
        finally:
            st.close()
        parity[n] = None if got is None else {"kwargs": got[0], **got[2]}
    res: dict[str, Any] = {
        "task": "AGG-1.19 opt-in L3 components fy_end_lone_monthday + fy1_captions, offline replay (no model)", "model_calls": 0,
        "label": "POST-HOC for the 12 target cells (the AGG-1.18 diagnosis saw them); every other cell is a regression check. "
                 "The 7 held-out docs are excluded and unscored.",
        "predeclared_sha256": sha(PREDECLARED),
        "orch_head": subprocess.run(["git", "-C", str(a.orch_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "orch_facts_det_sha256": sha(a.orch_root / "src/aggregation/facts_det.py"),
        "inputs_sha256": {str(p): sha(p) for p in (A117.STORE113, A117.R2, A117.R2_INGEST, A117.S2_RES, PILOT_STORE,
                                                   A117.JOB / "out/store_agg13.sqlite", A117.JOB / "out/store_agg14.sqlite")},
        "l3_parity": parity, "unseen_docs_excluded": unseen,
        "before_v3": {n: {m: sum(v.values()) for m, v in s.items() if not m.startswith("_")} for n, s in before.items()}}
    if any(v is None for v in parity.values()):
        res["verdict"] = "l3_parity_failed"
        a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(json.dumps({k: res[k] for k in ("l3_parity", "before_v3")}, indent=1))
        return 2
    cand = {n: dict(A117.l3_kwargs_candidates(fd, fys[n])) for n in base}
    kw_of = {n: cand[n][parity[n]["kwargs"]] for n in base}
    skip = {n: set(parity[n]["excluded_docs"]) | (set(unseen) if n == "pilot" else set()) for n in base}

    # 2. fy_end stress: lone month-day on top of the parity kwargs with fy_end forced on
    stress: dict[str, Any] = {}
    for n, p in base.items():
        st = P.open_store(p)
        try:
            kw = dict(kw_of[n], fy_end=fd.FY_END_MONTHS)
            if kw.get("period_version") is None:
                kw.pop("components", None)
            b = {f["fact_id"] + f"|{f['field']}": f for f in fd.extract_corpus_facts(st, cids[n], **kw) if f["doc_id"] not in skip[n]}
            kw2 = dict(kw, components=list(kw.get("components") or []) + [LONE])
            w = {f["fact_id"] + f"|{f['field']}": f for f in fd.extract_corpus_facts(st, cids[n], **kw2) if f["doc_id"] not in skip[n]}
        finally:
            st.close()
        changed: dict[str, int] = {}
        for k, f in w.items():
            if k in b and str(b[k]["period"]) != str(f["period"]):
                changed[f["doc_id"]] = changed.get(f["doc_id"], 0) + 1
        stress[n] = {"period_changes_by_doc": changed, "facts_only_in_one": len(set(b) ^ set(w))}
    res["fy_end_stress"] = stress

    # 3. variants
    v4 = list(pl.V4_L3_TEXTROW)
    variants = {LONE: [LONE], FY1: [FY1], "both": [LONE, FY1], "v4_textrow": v4, "v5_l3": v4 + [LONE, FY1]}
    after: dict[str, dict[str, Any]] = {}
    deltas: dict[str, Any] = {}
    for vname, comps in variants.items():
        after[vname], deltas[vname] = {}, {}
        for n, p in base.items():
            dst = a.scratch / f"{n}__{vname}.sqlite"
            P.copy_store(p, dst)
            st = P.open_store(dst)
            try:
                d = A117.apply_variant(fd, st, cids[n], kw_of[n], comps, skip[n])
            finally:
                st.close()
            deltas[vname][n] = {k: v for k, v in d.items() if k != "ltd_added"}
            after[vname][n] = score(n, dst)

    label = {"agg113": "76", "r2": "211", "agg13": "49:agg13", "agg14": "49:agg14", "pilot": "48"}

    def table(b_of: dict[str, Any], a_of: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for n in base:
            for m in ("raw", "canonical", "raw14", "canonical14"):
                if m in b_of[n]:
                    key = ("14" if m.endswith("14") else label[n]) + ":" + m.replace("14", "")
                    out[key] = A117.diff(b_of[n][m], a_of[n][m])
        return out

    out_v: dict[str, Any] = {}
    for vname in variants:
        sets = table(before, after[vname])
        regs = sorted({k for s in sets.values() for k in s["regressed"]})
        fixed12 = sorted(k for k in sets["48:raw"]["fixed"] if k in TARGET12)
        out_v[vname] = {"components": variants[vname], "delta": deltas[vname], "sets": sets, "regressions_vs_v3": regs,
                        "target12_fixed_raw": fixed12, "n_target12_fixed_raw": len(fixed12),
                        "pilot_fixed_raw_nontarget": sorted(k for k in sets["48:raw"]["fixed"] if k not in TARGET12),
                        "boeing_ni_2019_canonical": after[vname]["pilot"]["_detail"][BOEING_NI19]}
    v5_vs_v4 = table(after["v4_textrow"], after["v5_l3"])
    out_v["v5_l3"]["sets_vs_v4"] = v5_vs_v4
    out_v["v5_l3"]["regressions_vs_v4"] = sorted({k for s in v5_vs_v4.values() for k in s["regressed"]})
    res["variants"] = out_v
    res["boeing_ni_2019_canonical_before"] = before["pilot"]["_detail"][BOEING_NI19]
    res["pilot_before_detail_target12"] = {k: before["pilot"]["_detail"][k] for k in TARGET12}
    bundle = out_v["both"]
    futile = bundle["n_target12_fixed_raw"] < FUTILITY_MIN_FIXED or bool(bundle["regressions_vs_v3"]) or \
        bool(out_v["v5_l3"]["regressions_vs_v3"]) or bool(out_v["v5_l3"]["regressions_vs_v4"])
    res["futility_rule"] = f"bundle fixes < {FUTILITY_MIN_FIXED} of 12 pilot raw misses, or any regression (raw or canonical) on any set"
    res["verdict"] = "futile" if futile else "pass"
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"parity": parity, "before": res["before_v3"], "stress": stress, "variants": {k: {
        "fixed12": v["n_target12_fixed_raw"], "regs_v3": v["regressions_vs_v3"], "regs_v4": v.get("regressions_vs_v4"),
        "boeing_ni19": v["boeing_ni_2019_canonical"]["canonical_value"],
        "sets": {s: (d["before"], d["after"]) for s, d in v["sets"].items()}} for k, v in out_v.items()}, "verdict": res["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
