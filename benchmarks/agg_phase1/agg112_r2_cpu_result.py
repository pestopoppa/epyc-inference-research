#!/usr/bin/env python3
"""AGG-1.12 r2 (PREDECLARED_AGG112_AMENDMENT1_20261113.md): the CPU row's ``result.cpu.json`` re-evaluated on the AGG-1.14 corrected store.

OFFLINE, no model, no server. The AGG-1.12 verdict reads M1 / M3 / regressions from a CPU result file (``agg112.verdict``). The saved
CPU row's file carries the regressions of the UNCORRECTED store (JOHNSON_JOHNSON dividends_paid broken). The amendment re-runs the GPU leg
on the corrected store, so the no-regression rule must be evaluated over the corrected store's cells: this script recomputes, with the
runner's own functions (``agg112.readings``, ``regressions``, ``strata_report``, ``agg19_report``, ``phase1_arms.m1_summary``), every
field of the saved file that is a reading of the fact table (m1, m1_found, canonical, regressions, strata, agg19_cells, m1_*_docs) and
carries every other field (M3 and the ingest cost: no model call changed, the quote rule only drops rows) over unchanged.
The output is deterministic (no timestamps) and carries a ``corrected_from`` provenance block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg112 as G  # noqa: E402
import agg19 as G19  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orch-root", required=True, type=Path)
    ap.add_argument("--cpu-result", required=True, type=Path, help="the saved AGG-1.12 CPU row's result.cpu.json")
    ap.add_argument("--store", required=True, type=Path, help="the AGG-1.14 corrected store (copied, never opened for writing)")
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.orch_root.resolve()))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    X.configure(G.CORPUS_ID)
    inp = G.load_inputs(HERE)
    ref, cells = inp["reference"], inp["cells"]
    saved = json.loads(a.cpu_result.read_text())
    a.scratch.mkdir(parents=True, exist_ok=True)
    work = a.scratch / "r2_cpu_result_store_copy.sqlite"
    if work.exists():
        work.unlink()
    P.copy_store(a.store, work)
    store = P.open_store(work)
    sym = G.orch_symbols()
    spec = sym["pl"].pipeline_spec(G.PIPELINE)
    canon_stats = spec.refresh_canonical(store, G.CORPUS_ID)
    rows = G19.fact_rows(store.conn, G.CORPUS_ID)
    now = G.readings(rows, store.canonical_facts(G.CORPUS_ID), cells)
    held = set(ref["partition"]["held_out"])
    from src.aggregation.store import CANONICAL_RULE, select_canonical  # noqa: PLC0415
    v1_on_v2 = G19.canon_ok(select_canonical(rows, rule=CANONICAL_RULE)[0], cells)
    p0n = {d["doc_name"] for d in X.p0_docs(inp["corpus"])}
    summ = X.m1_summary(cells, [now["raw"][G.cell_key(d, f)] for d, f, _g in cells], p0n)
    raw_found = sum(now["raw"].values())
    out = dict(saved)
    out.update({
        "m1": raw_found / len(cells), "m1_found": raw_found, "m1_n": len(cells),
        "canonical": {"rule": canon_stats["rule"], "guards": canon_stats["guards"], "ok": sum(now["canonical"].values()),
                      "agg11_v1_ok": ref["agg11"]["canonical_v1_ok"], "v1_rule_on_v2_facts_ok": sum(v1_on_v2.values())},
        "regressions": G.regressions(ref, now, held), "strata": G.strata_report(ref, now, held),
        "agg19_cells": G.agg19_report(HERE, now), "m1_phase0_docs": summ["phase0_docs"], "m1_new_docs": summ["new_docs"],
        "corrected_from": {
            "task": "AGG-1.12 r2 (PREDECLARED_AGG112_AMENDMENT1_20261113.md)", "saved_cpu_result": str(a.cpu_result),
            "saved_cpu_result_sha256": sha256(a.cpu_result), "corrected_store": str(a.store), "corrected_store_sha256": sha256(a.store),
            "note": "M1, canonical, regressions, strata and the AGG-1.9 report are re-read from the AGG-1.14 corrected store (value-in-quote "
                    "rule on the typed L4 stage, offline replay of the saved t_window records); M3 and every ingest-cost field are the saved "
                    "CPU row's, unchanged (the rule drops rows, it makes no model call)."}})
    store.close()
    a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    reg = out["regressions"]
    print(f"raw {out['m1_found']}/{out['m1_n']} canonical {out['canonical']['ok']} regressions ok={reg['ok']} raw_broken={len(reg['raw_broken'])} "
          f"canonical_broken={len(reg['canonical_broken'])}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
