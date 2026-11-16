#!/usr/bin/env python3
"""AGG-1.17 (agg113/PREDECLARED_AGG117.md): the opt-in ``l3_textrow`` label components, replayed OFFLINE. No model, no server.

For each of the four cell sets' stores (AGG-1.13 store = the 76 cells + the 14 AGG-1.16b long_term_debt golds; the AGG-1.12 r2 seed = the
211 FY2022 cells; the AGG-1.11 step-2 work stores agg13 / agg14 = the 49 cells), a COPY is brought to the ``agg_pipeline.v3`` state
(the keep_slot harvest delta for the stores that predate it), canonicalised under canonical.v3 and scored (BEFORE). The copy's L3
facts are then re-extracted from its stored pages / tables / chunks: first with the kwargs that REPRODUCE the stored L3 fact ids
exactly (parity, else the store is not scored), then with those kwargs plus a variant's ``L3_TEXTROW_OPTIONAL`` components; the
fact-id difference is applied to a fresh copy, which is re-canonicalised under canonical.v3 and scored (AFTER). Any cell right
BEFORE and wrong AFTER, raw or canonical, on any set, is a regression.

    python3 agg117.py --orch-root <orchestrator worktree> --scratch <NEW dir> --out agg117_result.json
"""

from __future__ import annotations

import argparse
import csv
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

PREDECLARED = HERE / "agg113" / "PREDECLARED_AGG117.md"
GOLDS14 = HERE / "agg113" / "ltd_golds_agg116b.csv"
STORE113 = Path("/mnt/raid0/llm/tmp/agg116b-replay-20261116/store.v3.sqlite")   # v2 ingest + harvest delta, canonical.v3
JOB = Path("/mnt/raid0/llm/tmp/agg111s2-cpu-20261116")
R2 = JOB / "seed" / "store_agg112r2.sqlite"
R2_INGEST = JOB / "seed" / "agg112ingest_results.jsonl"
S2_RES = JOB / "out" / "results.jsonl"
TARGET8 = ("BLOCK_2022_10K|long_term_debt|2022", "BLOCK_2022_10K|long_term_debt|2021", "FOOTLOCKER_2023_10K|long_term_debt|2022",
           "FOOTLOCKER_2023_10K|long_term_debt|2021", "AMCOR_2022_10K|long_term_debt|2021", "NIKE_2022_10K|long_term_debt|2021",
           "PG_E_2022_10K|long_term_debt|2022", "PG_E_2022_10K|long_term_debt|2021")
BUNDLE3, BUNDLE4 = "ltd_qualifier+note_ref+tail_label", "ltd_qualifier+note_ref+tail_label+pipe_cells_ltd"
PILOT3 = ("BLOCK_2022_10K|long_term_debt|2021", "FOOTLOCKER_2023_10K|long_term_debt|2022", "PG_E_2022_10K|long_term_debt|2022")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def variants(fd: Any) -> dict[str, list[str]]:
    out = {c: [c] for c in fd.L3_TEXTROW_OPTIONAL}
    out[BUNDLE3] = ["ltd_qualifier", "note_ref", "tail_label"]
    out[BUNDLE4] = ["ltd_qualifier", "note_ref", "tail_label", "pipe_cells_ltd"]
    out["all_with_pipe_cells"] = ["ltd_qualifier", "note_ref", "tail_label", "pipe_cells"]
    return out


def l3_kwargs_candidates(fd: Any, fy: dict[str, str]) -> list[tuple[str, dict[str, Any]]]:
    v2 = list(fd.L3_PERIOD_COMPONENTS)
    return [("v1", {"period_version": None, "fiscal_years": fy}),
            ("v2", {"period_version": fd.L3_PERIOD_V2, "fiscal_years": fy, "components": v2}),
            ("v2+fy_end", {"period_version": fd.L3_PERIOD_V2, "fiscal_years": fy, "components": v2, "fy_end": fd.FY_END_MONTHS}),
            ("v1+fy_end", {"period_version": None, "fiscal_years": fy, "fy_end": fd.FY_END_MONTHS})]


SIG = ("fact_id", "field", "period", "value", "unit", "confidence")


def sig(f: dict[str, Any]) -> tuple[str, ...]:
    """A text-row fact id does not include the field or the period, so parity and the variant delta compare full signatures."""
    return tuple(str(f.get(k)) for k in SIG)


def stored_l3(st: Any) -> set[tuple[str, ...]]:
    cur = st.conn.execute(f"SELECT {', '.join(SIG)} FROM facts WHERE extractor LIKE 'l3%'")
    return {sig(dict(zip(SIG, r))) for r in cur.fetchall()}


def find_parity(fd: Any, st: Any, cid: str, fy: dict[str, str]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """The kwargs whose re-extraction reproduces the stored L3 fact ids: exactly, else per document (no stored fact missed, and the
    documents whose replay has EXTRA facts -- re-ingested without every L3 path -- are excluded from the variant delta)."""
    have = stored_l3(st)
    best = None
    for name, kw in l3_kwargs_candidates(fd, fy):
        fs = fd.extract_corpus_facts(st, cid, **kw)
        ids = {sig(f) for f in fs}
        miss, extra = have - ids, ids - have
        if miss:
            continue
        bad_docs = sorted({f["doc_id"] for f in fs if sig(f) in extra})
        rec = {"stored_not_replayed": 0, "replayed_not_stored": len(extra), "excluded_docs": bad_docs}
        if best is None or len(bad_docs) < len(best[2]["excluded_docs"]):
            best = (name, kw, rec)
        if not extra:
            break
    return best


def apply_variant(fd: Any, st: Any, cid: str, kw: dict[str, Any], comps: list[str], skip: set[str]) -> dict[str, Any]:
    base = {sig(f): f for f in fd.extract_corpus_facts(st, cid, **kw) if f["doc_id"] not in skip}
    kw2 = dict(kw, components=list(kw.get("components") or []) + comps)
    new = {sig(f): f for f in fd.extract_corpus_facts(st, cid, **kw2) if f["doc_id"] not in skip}
    gone = sorted({base[k]["fact_id"] for k in set(base) - set(new)})
    add = [new[k] for k in sorted(set(new) - set(base))]
    for i in range(0, len(gone), 500):
        part = gone[i:i + 500]
        st.conn.execute(f"DELETE FROM facts WHERE fact_id IN ({','.join('?' * len(part))})", part)
    st.write_facts(add)
    st.conn.commit()
    ltd = [{k: f.get(k) for k in ("doc_id", "period", "value", "unit", "page", "quote")} for f in add if f["field"] == "long_term_debt"]
    return {"removed": len(gone), "added": len(add), "added_by_field_top": _top(add), "ltd_added": ltd}


def _top(fs: list[dict[str, Any]], n: int = 12) -> dict[str, int]:
    c: dict[str, int] = {}
    for f in fs:
        c[f["field"]] = c.get(f["field"], 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1])[:n])


# --------------------------------------------------------------------------- scoring per set
def score_set(name: str, path: Path, ctx: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """{"raw": {key: bool}, "canonical": {key: bool}} (+ the 14 for agg113) after canonical.v3 is refreshed on ``path``."""
    import agg19 as G19  # noqa: PLC0415
    import agg112 as G112  # noqa: PLC0415
    import agg111s2 as S2  # noqa: PLC0415
    import agg116b as B  # noqa: PLC0415
    import phase0_arms as P  # noqa: PLC0415
    from agg115 import canon_fy  # noqa: PLC0415
    st = P.open_store(path)
    try:
        ctx["spec"].refresh_canonical(st, ctx["cid"])
        st.conn.commit()
        if name in ("agg113", "r2"):
            r = G112.readings(G19.fact_rows(st.conn, ctx["cid"]), st.canonical_facts(ctx["cid"]), ctx["cells"][name])
        else:
            r = None
    finally:
        st.close()
    out: dict[str, dict[str, bool]] = {}
    if r is not None:
        out["raw"], out["canonical"] = {k: bool(v) for k, v in r["raw"].items()}, {k: bool(v) for k, v in r["canonical"].items()}
    else:
        cs = ctx["cells"][name]
        out["raw"] = {k: v["verdict"] == "HIT" for k, v in S2.score_store(path, cs).items()}
        out["canonical"] = canon_fy(path, cs, ctx["v3rule"])
    if name == "agg113":
        s14 = B.score(path, ctx["cells14"])
        out["raw14"] = {k: v["raw_hit"] for k, v in s14.items()}
        out["canonical14"] = {k: v["canonical_ok"] for k, v in s14.items()}
        out["_detail14"] = {k: {"raw_verdict": v["raw_verdict"], "canonical_value": (v["canonical_row"] or {}).get("value"),
                               "canonical_quote": ((v["canonical_row"] or {}).get("quote") or "")[:90],
                               "canonical_page": (v["canonical_row"] or {}).get("page")} for k, v in s14.items()}
    return out


def diff(b: dict[str, bool], a: dict[str, bool]) -> dict[str, Any]:
    return {"n": len(b), "before": sum(b.values()), "after": sum(a.get(k, False) for k in b),
            "fixed": sorted(k for k in b if a.get(k) and not b[k]), "regressed": sorted(k for k in b if b[k] and not a.get(k))}


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
    import agg_l3period as L3P  # noqa: PLC0415
    import phase0_arms as P  # noqa: PLC0415
    import phase1_arms as X  # noqa: PLC0415
    X.configure(G112.CORPUS_ID)
    from src.aggregation import facts_det as fd, pipeline as pl  # noqa: PLC0415
    from src.aggregation.store import CANONICAL_RULE_V3  # noqa: PLC0415
    ft = G112.orch_symbols()["ft"]
    a.scratch.mkdir(parents=True, exist_ok=False)
    cid = G112.CORPUS_ID
    inp113, inp112 = H.load_inputs(H.D113), G112.load_inputs(HERE)
    cells49 = S2.load_cells(HERE)
    rows = list(csv.DictReader(GOLDS14.open(newline="")))
    cells14 = [{"doc": r["doc_name"], "field": r["metric"], "gold": float(r["value_usd_m"]), "fy": r["fiscal_year"]}
               for r in rows if r["status"] == "filled"]
    ctx = {"cid": cid, "spec": pl.pipeline_spec(pl.PIPELINE_V3), "v3rule": CANONICAL_RULE_V3, "cells14": cells14,
           "cells": {"agg113": inp113["cells"], "r2": inp112["cells"], "agg13": cells49["agg13"], "agg14": cells49["agg14"]}}
    fys = {"agg113": L3P.fiscal_years(inp113["corpus"]), "r2": L3P.fiscal_years(inp112["corpus"])}
    fys["agg13"] = fys["agg14"] = fys["r2"]

    # 0. the v3-state base copy of every store
    base: dict[str, Path] = {}
    harvest: dict[str, Any] = {}
    base["agg113"] = a.scratch / "agg113_base.sqlite"
    P.copy_store(STORE113, base["agg113"])
    base["r2"] = a.scratch / "r2_base.sqlite"
    harvest["r2"] = {k: v for k, v in A115.delta_store(ft, R2, base["r2"], A115.ingest_plans(R2_INGEST), cid).items() if k != "ltd_added"}
    for name in ("agg13", "agg14"):
        base[name] = a.scratch / f"{name}_base.sqlite"
        harvest[name] = {k: v for k, v in A115.delta_store(ft, JOB / "out" / f"store_{name}.sqlite", base[name],
                                                           A115.s2_plans(S2_RES, name), cid).items() if k != "ltd_added"}
    before = {n: score_set(n, p, ctx) for n, p in base.items()}

    # 1. L3 parity per store
    parity: dict[str, Any] = {}
    for n, p in base.items():
        st = P.open_store(p)
        try:
            got = find_parity(fd, st, cid, fys[n])
        finally:
            st.close()
        parity[n] = None if got is None else {"kwargs": got[0], **got[2]}
    res: dict[str, Any] = {"task": "AGG-1.17 opt-in l3_textrow label components, offline replay (no model)", "model_calls": 0,
                           "label": "IN-SAMPLE, POST-HOC for the 8 target cells; the other sets are a regression check",
                           "predeclared_sha256": sha(PREDECLARED), "golds_sha256": sha(GOLDS14),
                           "orch_head": subprocess.run(["git", "-C", str(a.orch_root), "rev-parse", "HEAD"], capture_output=True,
                                                       text=True).stdout.strip(),
                           "orch_facts_det_sha256": sha(a.orch_root / "src/aggregation/facts_det.py"),
                           "inputs_sha256": {str(p): sha(p) for p in (STORE113, R2, R2_INGEST, S2_RES, JOB / "out/store_agg13.sqlite",
                                                                      JOB / "out/store_agg14.sqlite")},
                           "harvest_delta": harvest, "l3_parity": parity,
                           "before_v3": {n: {m: sum(v.values()) for m, v in s.items() if not m.startswith("_")} for n, s in before.items()},
                           "before_14_detail": before["agg113"]["_detail14"]}
    if any(v is None for v in parity.values()):
        res["verdict"] = "l3_parity_failed"
        a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(json.dumps({k: res[k] for k in ("l3_parity", "before_v3")}, indent=1))
        return 2

    # 2. variants
    kw_of = {n: dict(l3_kwargs_candidates(fd, fys[n]))[parity[n]["kwargs"]] for n in base}
    out_v: dict[str, Any] = {}
    for vname, comps in variants(fd).items():
        per: dict[str, Any] = {"components": comps, "sets": {}, "delta": {}}
        for n, p in base.items():
            dst = a.scratch / f"{n}__{vname.replace('+', '_')}.sqlite"
            P.copy_store(p, dst)
            st = P.open_store(dst)
            try:
                per["delta"][n] = apply_variant(fd, st, cid, kw_of[n], comps, set(parity[n]["excluded_docs"]))
            finally:
                st.close()
            after = score_set(n, dst, ctx)
            for m in ("raw", "canonical", "raw14", "canonical14"):
                if m in before[n]:
                    label = {"agg113": "76", "r2": "211", "agg13": "49:agg13", "agg14": "49:agg14"}[n] if not m.endswith("14") else "14"
                    per["sets"][f"{label}:{m.replace('14', '')}"] = diff(before[n][m], after[m])
            if n == "agg113":
                per["after_14_detail"] = after["_detail14"]
        regs = sorted({k for s in per["sets"].values() for k in s["regressed"]})
        fixed8 = sorted({k for m in ("14:raw", "14:canonical") for k in per["sets"][m]["fixed"] if k in TARGET8})
        per["regressions"] = regs
        per["target8_fixed_raw"] = sorted(k for k in per["sets"]["14:raw"]["fixed"] if k in TARGET8)
        per["target8_fixed_canonical"] = sorted(k for k in per["sets"]["14:canonical"]["fixed"] if k in TARGET8)
        per["pilot3_fixed"] = sorted(k for k in fixed8 if k in PILOT3)
        per["admissible"] = not regs and bool(fixed8)
        out_v[vname] = per
    res["variants"] = out_v
    # predeclared futility: the 3-cell pilot fixes 0 cells or regresses any hit -- judged on the proposed bundle
    ok = [v for v in (BUNDLE4, BUNDLE3) if out_v[v]["admissible"] and out_v[v]["pilot3_fixed"]]
    res["recommended"] = ok[0] if ok else None
    res["verdict"] = f"admissible:{ok[0]}" if ok else "pilot_futile"
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"parity": parity, "before": res["before_v3"], "variants": {k: {kk: v[kk] for kk in (
        "regressions", "target8_fixed_raw", "target8_fixed_canonical", "pilot3_fixed", "admissible")} | {
        "sets": {s: (d["before"], d["after"]) for s, d in v["sets"].items()}} for k, v in out_v.items()}, "verdict": res["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
