#!/usr/bin/env python3
"""AGG-1.18 offline replay: the opt-in quote rule ``value_in_quote.rounded.v1`` and the bundle ``agg_pipeline.v4``. No model, no server.

Same stores and cell sets as agg117 (AGG-1.13 store = 76 cells + the 14 long_term_debt golds; AGG-1.12 r2 = 211 FY2022 cells; AGG-1.11
step 2 agg13 / agg14 = 49 cells each), each copy first brought to the ``agg_pipeline.v3`` state exactly as agg117 does (BEFORE).

* Q   (Task 1)  the saved typed window replies are replayed through the v3 typed finish (``keep_slot_harvest`` -> ``drop_unquoted`` ->
                ``reconcile``) under ``value_in_quote.v1`` and under ``value_in_quote.rounded.v1``; the fact-id difference is applied
                to a copy, re-canonicalised under canonical.v3 and scored.
* V4  (Task 2)  the AGG-1.17 text-row delta (``pipeline.V4_L3_TEXTROW`` on top of the parity L3 kwargs, agg117.apply_variant) and
                then the Q delta, on one copy.

Any cell right BEFORE and wrong AFTER (raw or canonical, any set) is a regression. The v1-rule replay must reproduce only fact ids the
store holds (``replayed_not_stored`` = 0 per set), else the set is reported as a parity failure.

    python3 agg118.py --orch-root <orchestrator worktree with agg_pipeline.v4> --scratch <NEW dir> --out agg118_result.json
"""

from __future__ import annotations

import argparse
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

V2_INGEST_RES = Path("/mnt/raid0/llm/tmp/agg113-ingest-v2-cpu-20261114-r2/out/results.jsonl")


def replay(ft: Any, recs: list[dict[str, Any]], planned: set[str], rule: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(reconciled facts, rows the rule kept) of one plan under the v3 typed finish with quote rule ``rule``."""
    from src.aggregation import facts_llm  # noqa: PLC0415
    kept_in = [f for r in recs for f in r["facts"] if ft.keep_slot_harvest(f, set(r["chosen_by"]), planned)]
    kept, _drop = ft.drop_unquoted([dict(f) for f in kept_in], rule=rule)
    return facts_llm.reconcile([dict(f) for f in kept])[0], kept


def plans_for(name: str, ft: Any) -> dict[str, dict[str, Any]]:
    import agg115 as A115  # noqa: PLC0415
    import agg117 as A117  # noqa: PLC0415
    if name == "agg113":
        out: dict[str, dict[str, Any]] = {}
        for line in V2_INGEST_RES.open():
            r = json.loads(line)
            if r.get("kind") == "t_window" and r.get("final"):
                out.setdefault(r["doc"], {"doc": r["doc"], "recs": [], "planned": set(ft.SCHEMA_FIELDS)})["recs"].append(r)
        return out
    if name == "r2":
        return A115.ingest_plans(A117.R2_INGEST)
    return A115.s2_plans(A117.S2_RES, name)


def apply_quote(ft: Any, P: Any, path: Path, plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    st = P.open_store(path)
    try:
        stored = {r[0] for r in st.conn.execute("SELECT fact_id FROM facts WHERE extractor = 'l4_llm.v1'")}
        not_stored, removed, added, restored = 0, set(), [], []
        for pid, p in sorted(plans.items()):
            base, kept1 = replay(ft, p["recs"], p["planned"], ft.VALUE_IN_QUOTE_V1)
            new, kept2 = replay(ft, p["recs"], p["planned"], ft.VALUE_IN_QUOTE_ROUNDED)
            bids = {f["fact_id"] for f in base}
            not_stored += len(bids - stored)
            k1 = {f["fact_id"] for f in kept1}
            restored += [{"plan": pid, **{k: f.get(k) for k in ("doc_id", "field", "period", "value", "unit", "declared_scale", "quote")}}
                         for f in kept2 if f["fact_id"] not in k1]
            removed |= bids - {f["fact_id"] for f in new}
            added += [f for f in new if f["fact_id"] not in bids and f["fact_id"] not in stored]
        gone = sorted(removed & stored)
        for i in range(0, len(gone), 500):
            part = gone[i:i + 500]
            st.conn.execute(f"DELETE FROM facts WHERE fact_id IN ({','.join('?' * len(part))})", part)
        uniq = {f["fact_id"]: f for f in added}
        st.write_facts(list(uniq.values()))
        st.conn.commit()
    finally:
        st.close()
    return {"replayed_not_stored": not_stored, "rows_restored_by_rule": len(restored), "restored": restored,
            "facts_removed": len(gone), "facts_added": len(uniq),
            "added": [{k: f.get(k) for k in ("doc_id", "field", "period", "value", "unit", "quote")} for f in uniq.values()]}


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
    import agg117 as A117  # noqa: PLC0415
    import agg_l3period as L3P  # noqa: PLC0415
    import csv  # noqa: PLC0415
    import phase0_arms as P  # noqa: PLC0415
    import phase1_arms as X  # noqa: PLC0415
    X.configure(G112.CORPUS_ID)
    from src.aggregation import facts_det as fd, facts_typed as ft, pipeline as pl  # noqa: PLC0415
    from src.aggregation.store import CANONICAL_RULE_V3  # noqa: PLC0415
    assert pl.pipeline_spec(pl.PIPELINE_V4).canonical_rule == CANONICAL_RULE_V3
    a.scratch.mkdir(parents=True, exist_ok=False)
    cid = G112.CORPUS_ID
    inp113, inp112 = H.load_inputs(H.D113), G112.load_inputs(HERE)
    cells49 = S2.load_cells(HERE)
    rows = list(csv.DictReader(A117.GOLDS14.open(newline="")))
    cells14 = [{"doc": r["doc_name"], "field": r["metric"], "gold": float(r["value_usd_m"]), "fy": r["fiscal_year"]}
               for r in rows if r["status"] == "filled"]
    ctx = {"cid": cid, "spec": pl.pipeline_spec(pl.PIPELINE_V3), "v3rule": CANONICAL_RULE_V3, "cells14": cells14,
           "cells": {"agg113": inp113["cells"], "r2": inp112["cells"], "agg13": cells49["agg13"], "agg14": cells49["agg14"]}}
    fys = {"agg113": L3P.fiscal_years(inp113["corpus"]), "r2": L3P.fiscal_years(inp112["corpus"])}
    fys["agg13"] = fys["agg14"] = fys["r2"]

    base: dict[str, Path] = {"agg113": a.scratch / "agg113_base.sqlite", "r2": a.scratch / "r2_base.sqlite"}
    P.copy_store(A117.STORE113, base["agg113"])
    A115.delta_store(ft, A117.R2, base["r2"], A115.ingest_plans(A117.R2_INGEST), cid)
    for name in ("agg13", "agg14"):
        base[name] = a.scratch / f"{name}_base.sqlite"
        A115.delta_store(ft, A117.JOB / "out" / f"store_{name}.sqlite", base[name], A115.s2_plans(A117.S2_RES, name), cid)
    before = {n: A117.score_set(n, p, ctx) for n, p in base.items()}

    parity: dict[str, Any] = {}
    for n, p in base.items():
        st = P.open_store(p)
        try:
            got = A117.find_parity(fd, st, cid, fys[n])
        finally:
            st.close()
        parity[n] = None if got is None else {"kwargs": got[0], **got[2]}
    if any(v is None for v in parity.values()):
        a.out.write_text(json.dumps({"verdict": "l3_parity_failed", "l3_parity": parity}, indent=1) + "\n")
        return 2
    kw_of = {n: dict(A117.l3_kwargs_candidates(fd, fys[n]))[parity[n]["kwargs"]] for n in base}
    plans = {n: plans_for(n, ft) for n in base}
    label = {"agg113": "76", "r2": "211", "agg13": "49:agg13", "agg14": "49:agg14"}

    arms: dict[str, Any] = {}
    for arm in ("Q", "V4"):
        per: dict[str, Any] = {"sets": {}, "quote": {}, "textrow": {}}
        for n, p in base.items():
            dst = a.scratch / f"{n}__{arm}.sqlite"
            P.copy_store(p, dst)
            if arm == "V4":
                st = P.open_store(dst)
                try:
                    per["textrow"][n] = A117.apply_variant(fd, st, cid, kw_of[n], list(pl.V4_L3_TEXTROW),
                                                           set(parity[n]["excluded_docs"]))
                finally:
                    st.close()
            per["quote"][n] = apply_quote(ft, P, dst, plans[n])
            after = A117.score_set(n, dst, ctx)
            for m in ("raw", "canonical", "raw14", "canonical14"):
                if m in before[n]:
                    lab = "14" if m.endswith("14") else label[n]
                    per["sets"][f"{lab}:{m.replace('14', '')}"] = A117.diff(before[n][m], after[m])
            if n == "agg113":
                per["after_14_detail"] = after["_detail14"]
        per["regressions"] = sorted({k for s in per["sets"].values() for k in s["regressed"]})
        per["fixed"] = sorted({k for s in per["sets"].values() for k in s["fixed"]})
        per["rows_restored_by_rule"] = sum(q["rows_restored_by_rule"] for q in per["quote"].values())
        per["replayed_not_stored"] = {n: q["replayed_not_stored"] for n, q in per["quote"].items()}
        arms[arm] = per

    res = {"task": "AGG-1.18 value_in_quote.rounded.v1 (Q) and agg_pipeline.v4 (V4), offline replay (no model)", "model_calls": 0,
           "label": "IN-SAMPLE, POST-HOC (the 14 long_term_debt golds tuned AGG-1.17; BLOCK motivated the rule); regression check on all sets",
           "orch_head": subprocess.run(["git", "-C", str(a.orch_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "orch_facts_typed_sha256": A117.sha(a.orch_root / "src/aggregation/facts_typed.py"),
           "orch_pipeline_sha256": A117.sha(a.orch_root / "src/aggregation/pipeline.py"),
           "inputs_sha256": {str(p): A117.sha(p) for p in (A117.STORE113, A117.R2, A117.R2_INGEST, A117.S2_RES, V2_INGEST_RES,
                                                            A117.JOB / "out/store_agg13.sqlite", A117.JOB / "out/store_agg14.sqlite")},
           "l3_parity": parity,
           "before_v3": {n: {m: sum(v.values()) for m, v in s.items() if not m.startswith("_")} for n, s in before.items()},
           "arms": arms}
    res["verdict"] = {k: ("0_regressions" if not v["regressions"] else "REGRESSIONS") for k, v in arms.items()}
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"before": res["before_v3"], "arms": {k: {"sets": {s: (d["before"], d["after"]) for s, d in v["sets"].items()},
          "fixed": v["fixed"], "regressions": v["regressions"], "rows_restored_by_rule": v["rows_restored_by_rule"],
          "replayed_not_stored": v["replayed_not_stored"]} for k, v in arms.items()}, "verdict": res["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
