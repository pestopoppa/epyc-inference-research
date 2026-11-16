#!/usr/bin/env python3
"""AGG-1.11 step 2 runner CLI (keeper jobs run it under normal grants; logic in ``agg111s2.py``). Code only: no model, server or
process is started here.

``--phase``:
  preflight  zero compute: pins, the 13 targets and 24 + 25 cells, the seed stores, the orchestrator stage, the offline window plan
  run        CPU: [smoke = ONE typed window call] or the pilot (5 cells, futility at 0 of 5), the rest, per-document finish on the
             working store copies, M1 before/after, the backfill-gap diagnosis -> ``result.cpu.json``
  diag       zero model: the backfill-gap diagnosis alone -> ``result.diag.json``
``--dry-run`` validates every input and prints the plan; zero model calls, no server, nothing written.
Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 futility stop, 4 precondition, 5 budget stop, 75 partial (resumable).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
import agg111s2 as G  # noqa: E402
import agg112 as G112  # noqa: E402
import agg16 as G16  # noqa: E402
import phase1_arms as X  # noqa: E402
from phase1_arms import P  # noqa: E402

PHASES = ("preflight", "run", "diag")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, required=True)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--seed-agg13", type=Path, required=True, help="sealed copy of the AGG-1.3 r3 store (never written)")
    ap.add_argument("--seed-agg14", type=Path, required=True, help="sealed copy of the AGG-1.4 r3 store (never written)")
    ap.add_argument("--r2-store", type=Path, required=True, help="AGG-1.12 r2 store (read-only, diagnosis)")
    ap.add_argument("--ingest-store", type=Path, required=True, help="AGG-1.12 ingest store (read-only, diagnosis)")
    ap.add_argument("--ingest-results", type=Path, required=True, help="AGG-1.12 ingest results.jsonl (read-only, diagnosis)")
    ap.add_argument("--booked-s", type=float, default=800.0)
    ap.add_argument("--no-pins", action="store_true", help="tests only")
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def checks(a: argparse.Namespace, inp: dict) -> tuple[list[str], dict]:
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    problems = G.input_problems(inp, a.set_dir, check_pin=not a.no_pins)
    problems += G112.orch_problems(a.orch_root, a.orch_pin) if a.orch_pin else []
    for n in ("seed_agg13", "seed_agg14", "r2_store", "ingest_store", "ingest_results"):
        if not getattr(a, n).exists():
            problems.append(f"{n} missing: {getattr(a, n)}")
    info: dict = {}
    if problems:
        return problems, info
    seeds = {"agg13": a.seed_agg13, "agg14": a.seed_agg14}
    wins = {}
    for name, path in seeds.items():
        docs = sorted({t["doc"] for t in inp["targets"] if t["store"] == name})
        st = G16.mem_store(path, docs)
        try:
            wins[name] = G.doc_windows(st)
        finally:
            st.close()
    plans = G.doc_plans(inp["targets"], wins)
    est = G.estimates(plans)
    if est["total_s"] > G.BUDGET_X * a.booked_s:
        problems.append(f"plan {est['total_s']} s > {G.BUDGET_X}x booked {a.booked_s} s")
    info["plan"] = {"targets": [t["key"] for t in inp["targets"]], "pilot": ["|".join(k) for k in G.PILOT_CELLS],
                    "per_doc": {k: {"windows": p["n_windows"], "fields": p["fields"], "chars": p["chars"]} for k, p in plans.items()},
                    "estimate": est, "booked_s": a.booked_s, "cells": {k: len(v) for k, v in inp["cells"].items()},
                    "gates": {"pilot_futility": "0 of 5 filled -> rc 3", "budget": "pilot wall > 600 s or projected > 2x booked -> rc 5",
                              "done_when": "M1 before/after over 24 + 25; any HIT -> not-HIT regression -> agg111s2_regression"}}
    return problems, info


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    X.configure(G.CORPUS_ID)
    try:
        inp = G.load_inputs(a.set_dir)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load inputs: {type(e).__name__}: {e}"]}))
        return G.RC_PREFLIGHT
    problems, info = checks(a, inp)
    if a.dry_run or a.phase == "preflight":
        print(json.dumps({"dry_run": a.dry_run, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
                          "predeclared_sha256": G.PREDECLARED_SHA, "problems": problems, "ok": not problems, **info}, indent=1, default=str))
        if a.phase == "preflight" and not a.dry_run:
            if a.out is None:
                build_parser().error("--out is required")
            a.out.mkdir(parents=True, exist_ok=True)
            X._atomic_write(a.out / "result.preflight.json", json.dumps(
                {"phase": "preflight", "complete": not problems, "problems": problems, "predeclared_sha256": G.PREDECLARED_SHA,
                 **info}, indent=1, default=str) + "\n")
        return G.RC_PREFLIGHT if problems else 0
    if a.out is None:
        build_parser().error("--out is required")
    if problems:
        print(json.dumps({"problems": problems}))
        return G.RC_PREFLIGHT
    a.out.mkdir(parents=True, exist_ok=True)
    diag_inputs = {"r2_store": a.r2_store, "ingest_store": a.ingest_store, "ingest_results": a.ingest_results}
    if a.phase == "diag":
        d = G.gap_diagnosis(inp["cells"]["agg14"], agg14_store=a.seed_agg14, **diag_inputs)
        X._atomic_write(a.out / "result.diag.json", json.dumps({"phase": "diag", "complete": True, "model_calls": 0, **d,
                                                                 "predeclared_sha256": G.PREDECLARED_SHA}, indent=1, default=str) + "\n")
        print(json.dumps({"phase": "diag", "lost": d["lost"], "by_step": d["by_step"]}))
        return 0
    if not a.url:
        build_parser().error("--url is required for --phase run")
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    grant_guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    row_stop = os.environ.get("AGG111S2_ROW_STOP")      # the row's own STOP file, checked before every model call

    def guard() -> None:
        grant_guard()
        if row_stop and os.path.exists(row_stop):
            raise P.Stop("row-stopfile")
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=G.PREDECLARED_SHA, orch_pin=a.orch_pin, url=a.url,
             model_id=a.model_id, booked_s=a.booked_s, final=False)
    from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
    backend = LlamaServerBackend(a.url, model_id=a.model_id)
    try:
        res = G.run(rec, a.out, inp, backend, seeds={"agg13": a.seed_agg13, "agg14": a.seed_agg14}, diag_inputs=diag_inputs,
                    guard=guard, booked_s=a.booked_s, smoke=smoke)
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        print(json.dumps({"stopped": str(e), "rc": G.RC_PARTIAL}))
        return G.RC_PARTIAL
    print(json.dumps({k: v for k, v in res.items() if k in ("complete", "verdict", "rc", "smoke", "budget_stop", "futility_stop", "reason")},
                     default=str))
    return int(res.get("rc", G.RC_PARTIAL))


if __name__ == "__main__":
    sys.exit(main())
