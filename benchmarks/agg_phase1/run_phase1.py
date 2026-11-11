#!/usr/bin/env python3
"""AGG-1.1 step B: Phase 1 runner (keeper jobs run it under normal grants; logic in ``phase1_arms.py``).

Code only: no model, server or process is started here. The corpus manifest, the question / gold set and the
output directory are parameters (``--corpus``, ``--golds``, ``--verification``, ``--facts-sheet``, ``--out``);
the defaults are the frozen ``benchmarks/agg_phase1`` files. The Phase 0 runner and arms are imported, never
edited (``run_phase0.py`` keeps its 10-document set, so Phase 0 stays byte-reproducible).

``--phase`` (PREDECLARED checkpoints in parentheses):

  preflight     (0)   zero-compute checks and the L4 reuse test
  l3            (1)   L3 load of the 18 new documents into a copy of the Phase 0 store (no model)
  pilot         (2)   L4 on MICROSOFT, JOHNSON_JOHNSON, 3M + its futility checks (<= 1,800 s)
  ingest        (3,4) remaining L4 units, interim look, final look; writes result.cpu.json
  gpu-audit     (5)   L4c audit sample
  gpu-answers   (5)   P-large then B1 on the 58 (B1 stop-on-answer after 30), then the verdict
  score               verdict only

``--dry-run`` validates every input, prints the plan (units, order, reuse, budgets) and exits: zero model calls,
no server, no store, nothing written.

Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped (phase1_gate_tripped: the GPU window
is never requested), 4 precondition, 5 budget stop (re-book), 6 L3 parse futility, 75 partial (resume).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
import phase1_arms as X  # noqa: E402
from phase1_arms import A, P  # noqa: E402

PHASES = ("preflight", "l3", "pilot", "ingest", "gpu-audit", "gpu-answers", "score")
DEFAULT_P0_JOB = Path("/mnt/raid0/llm/tmp/agg0-arms-cpu-20261110-r2")
DEFAULT_P0_L3_STORE = Path("/mnt/raid0/llm/tmp/agg0-l3-load-20261110-r3/out/store.sqlite")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; no model, no writes")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--corpus", type=Path, default=None, help="corpus manifest (default SET/corpus.json)")
    ap.add_argument("--golds", type=Path, default=None, help="question + gold set (default SET/golds.jsonl)")
    ap.add_argument("--verification", type=Path, default=None, help="default SET/verification_record.json")
    ap.add_argument("--facts-sheet", type=Path, default=None, help="default SET/facts_sheet.csv")
    ap.add_argument("--sizes", type=Path, default=None, help="extracted sizes (default SET/extracted_sizes.json)")
    ap.add_argument("--out", type=Path, default=None, help="results dir (required to run)")
    ap.add_argument("--pins", choices=("phase1", "none"), default="phase1",
                    help="check the frozen-input sha256 pins and the 58 / 211 / ranks counts (default); "
                         "'none' is for fixture sets only")
    ap.add_argument("--predeclared-sha256", default=X.PREDECLARED_SHA)
    ap.add_argument("--corpus-id", default=P.CORPUS_ID, help="store corpus id (the Phase 0 store's by default)")
    ap.add_argument("--orch-root", type=Path, default=None, help="pinned orchestrator tree; prepended to sys.path")
    ap.add_argument("--orch-pin", default=X.ORCH_PIN, help="expected <orch-root>/.pin content, when that file exists")
    ap.add_argument("--research-pin", default="", help="recorded in meta (the commit that added this runner)")
    ap.add_argument("--p0-job", type=Path, default=DEFAULT_P0_JOB, help="Phase 0 CPU job dir (L4 reuse source)")
    ap.add_argument("--p0-results", type=Path, default=None, help="default P0-JOB/out/results.jsonl")
    ap.add_argument("--store", type=Path, default=None, help="this job's store (default OUT/store.sqlite)")
    ap.add_argument("--store-src", type=Path, default=None,
                    help="l3: base store when L4 units are reused (default P0-JOB/out/store.sqlite); "
                         "gpu phases: the CPU job's store to copy once")
    ap.add_argument("--l3-store-src", type=Path, default=DEFAULT_P0_L3_STORE,
                    help="l3: base store (L3 only) when the L4 reuse test fails and all units run fresh")
    ap.add_argument("--cpu-results", type=Path, help="the CPU job's results.jsonl (gpu phases)")
    ap.add_argument("--cpu-result-json", type=Path, help="the CPU job's result.cpu.json (default beside results)")
    ap.add_argument("--url", help="private llama-server base URL")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--booked-l4-s", type=float, default=None, help="override the booked L4 estimate (s)")
    return ap


def resolve(a: argparse.Namespace) -> dict[str, Path]:
    s = a.set_dir
    return {"corpus": a.corpus or s / "corpus.json", "golds": a.golds or s / "golds.jsonl",
            "verification": a.verification or s / "verification_record.json",
            "facts_sheet": a.facts_sheet or s / "facts_sheet.csv", "sizes": a.sizes or s / "extracted_sizes.json"}


def pins_for(a: argparse.Namespace) -> dict[str, Any]:
    if a.pins == "none":
        return {"pins": None, "predeclared_sha": None, "expect_cells": None}
    return {"pins": X.FROZEN_P1, "predeclared_sha": a.predeclared_sha256, "expect_cells": X.N_CELLS}


def orch_pin_problem(a: argparse.Namespace) -> str | None:
    if a.orch_root is None:
        return None
    f = a.orch_root / ".pin"
    if f.exists() and f.read_text().strip() != a.orch_pin:
        return f"{f} = {f.read_text().strip()} != expected orchestrator pin {a.orch_pin}"
    return None


def dry_run(a: argparse.Namespace, inp: dict[str, Any], paths: dict[str, Path]) -> tuple[int, dict[str, Any]]:
    """Validate inputs and plan units. Reads files and imports the orchestrator package; calls no model."""
    sizes = X.load_sizes(paths["sizes"])
    do_p0 = a.phase in ("preflight", "l3", "pilot", "ingest")
    ok, info = X.preflight(inp, a.set_dir, orch_root=a.orch_root if do_p0 else None,
                           p0_job=a.p0_job if do_p0 and a.p0_job.exists() else None, sizes=sizes,
                           run_build_check=True, check_pdfs=a.phase not in ("gpu-audit", "gpu-answers", "score"),
                           **pins_for(a))
    problems = list(info["problems"])
    p = orch_pin_problem(a)
    if p:
        problems.append(p)
    if a.orch_root is not None:  # the orchestrator tree must carry the symbols the phases call
        sys.path.insert(0, str(a.orch_root))
        os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
        try:
            import importlib  # noqa: PLC0415
            for m in ("src.aggregation.llm_backend", "src.aggregation.facts_llm", "src.aggregation.planner",
                      "src.aggregation.store", "src.aggregation.sql_grammar", "src.aggregation.registry",
                      "src.repl_environment.context_search"):
                importlib.import_module(m)
            from src.aggregation import planner, sql_grammar, store as st  # noqa: PLC0415
            assert hasattr(planner, "scoped_facts") and hasattr(planner, "resolve_scope")
            assert hasattr(st.CorpusStore, "refresh_canonical") and hasattr(sql_grammar, "catalog_from_canonical")
        except Exception as e:  # noqa: BLE001
            problems.append(f"orchestrator tree {a.orch_root} cannot serve the phases: {type(e).__name__}: {e}")
    corpus = inp["corpus"]
    agg = X.verified_agg(inp)
    reuse_ok = bool((info.get("l4_reuse") or {}).get("reuse_ok"))
    plan = X.plan_units(corpus, sizes, reuse_ok)
    stage_n = {s: sum(1 for p_ in plan if p_["stage"] == s) for s in ("reused", "pilot", "ingest")}
    order = X.new_doc_order(corpus, sizes)
    out: dict[str, Any] = {
        "dry_run": True, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
        "preflight": {k: v for k, v in info.items() if k != "problems"}, "problems": problems,
        "predeclared_sha256": info.get("predeclared_sha256"),
        "inputs": {"corpus": str(paths["corpus"]), "golds": str(paths["golds"]),
                   "verification": str(paths["verification"]), "facts_sheet": str(paths["facts_sheet"]),
                   "out": str(a.out) if a.out else None}}
    if a.phase in ("preflight", "l3", "pilot", "ingest"):
        p0_results = a.p0_results or a.p0_job / "out" / "results.jsonl"
        out["l4_reuse"] = {"reuse_ok": reuse_ok, "p0_results_present": p0_results.exists(),
                           "base_store": str((a.store_src or a.p0_job / "out" / "store.sqlite") if reuse_ok
                                             else a.l3_store_src)}
        if reuse_ok and p0_results.exists():
            ok_units = X.l4_ok(X.read_jsonl(p0_results))
            out["l4_reuse"]["p0_reused_units"] = X.tally(ok_units.values())
        elif reuse_ok:
            problems.append(f"L4 reuse selected but Phase 0 results missing: {p0_results}")
        out["plan"] = {"l3_new_docs": order, "l3_docs_to_parse": len(order),
                       "pilot": {"docs": list(X.PILOT_DOCS), "units": stage_n["pilot"],
                                 "bound_s": X.PILOT_WALL_S,
                                 "futility": "projected M3 over a kill bound, pilot M1 < 0.60 on its 24 cells, "
                                             "projected L4 wall > 2x booked, calls/unit > 12"},
                       "ingest": {"units_reused": stage_n["reused"], "units_to_run_after_pilot": stage_n["ingest"],
                                  "total_units": len(plan), "interim_look_after_new_docs": X.INTERIM_NEW_DOCS,
                                  "order_new_docs": order,
                                  "booked_l4_s": a.booked_l4_s or X.BOOKED_L4_S["reuse" if reuse_ok else "fresh"],
                                  "hard_stop": "1.5x pilot-based projection"},
                       "gates": {"M1_min": X.M1_GATE, "M3_prefill_ratio_max": A.KILL["m3_prefill_ratio_max"],
                                 "M3_calls_per_100_chunks_max": A.KILL["m3_calls_per_100_chunks_max"],
                                 "M4": {"median_calls_kill": A.KILL["median_calls_max"],
                                        "median_calls_soft": X.M4_MEDIAN_CALLS_SOFT,
                                        "median_prefill": X.M4_MEDIAN_PREFILL_MAX}}}
    if a.phase in ("gpu-audit", "gpu-answers", "score"):
        ready = None
        if a.cpu_results:
            rj = a.cpu_result_json or a.cpu_results.parent / "result.cpu.json"
            ready = X.cpu_ready(json.loads(rj.read_text())) if rj.exists() else (False, f"{rj} not present yet")
        out["precondition_cpu"] = {"ready": ready[0] if ready else False,
                                   "detail": ready[1] if ready else "no --cpu-results given",
                                   "note": "informational in a dry run: the CPU row has not run yet"}
        out["plan"] = {"audit": "L4c sample computed at run time from the CPU store",
                       "p_large": {"n": len(agg), "order": [q["id"] for q in agg]},
                       "b1": {"n": len(agg), "order": [q["id"] for q in agg],
                              "early_stop": f"after {X.B1_LOOK_AT} answers if paired z >= {X.B1_STOP_Z} "
                                            "(b - c) / sqrt(b + c)"},
                       "arms_not_run": ["P-small", "B-today", "B2"],
                       "n_excluded_unverified": len(inp["golds"]) - len(agg)}
    out["ok"] = not problems
    return (0 if not problems else X.RC_PREFLIGHT), out


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    paths = resolve(a)
    X.configure(a.corpus_id)
    try:
        inp = X.load_inputs(corpus=paths["corpus"], golds=paths["golds"], verification=paths["verification"],
                            facts_sheet=paths["facts_sheet"])
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load inputs: {type(e).__name__}: {e}"]}))
        return X.RC_PREFLIGHT
    if a.dry_run:
        rc, out = dry_run(a, inp, paths)
        print(json.dumps(out, indent=1, default=str))
        return rc
    if a.out is None:
        build_parser().error("--out is required to run (use --dry-run to only validate)")
    if a.phase in ("l3", "pilot", "ingest", "gpu-audit", "gpu-answers", "score") and a.orch_root is None:
        build_parser().error("--orch-root (the pinned orchestrator tree) is required to run this phase")
    sizes = X.load_sizes(paths["sizes"])
    gpu = a.phase in ("gpu-audit", "gpu-answers", "score")
    do_p0 = not gpu and a.phase != "score"
    ok, info = X.preflight(inp, a.set_dir, orch_root=a.orch_root if do_p0 else None,
                           p0_job=a.p0_job if do_p0 and a.p0_job.exists() else None, sizes=sizes,
                           check_pdfs=not gpu, **pins_for(a))
    pp = orch_pin_problem(a)
    if pp:
        info["problems"].append(pp)
        ok = False
    print(json.dumps({k: v for k, v in info.items() if k != "l4_reuse"}, indent=1, default=str))
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    if not ok or a.phase == "preflight":
        rec.emit("preflight", unit=None, ok=ok, final=False, **{k: v for k, v in info.items()})
        res = {"phase": "preflight", "complete": ok, "problems": info["problems"],
               "predeclared_sha256": info.get("predeclared_sha256"),
               "l4_reuse": (info.get("l4_reuse") or {}).get("reuse_ok")}
        if a.phase == "preflight" or not ok:
            X._atomic_write(a.out / "result.preflight.json", json.dumps(res, indent=1, default=str) + "\n")
        if not ok:
            return X.RC_PREFLIGHT
        return 0
    if a.orch_root:
        sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    pred_sha = info.get("predeclared_sha256") or X.sha256_file(a.set_dir / "PREDECLARED.md")
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=pred_sha, orch_pin=a.orch_pin,
             research_pin=a.research_pin, pins=a.pins, corpus_id=a.corpus_id, url=a.url, model_id=a.model_id,
             cfg=dict(vars(P.CFG)), n_agg_verified=info["n_agg_verified"], n_agg_excluded=info["n_agg_excluded"],
             m1_cells=info["m1_cells"], final=False)
    agg = X.verified_agg(inp)
    store_path = a.store or a.out / "store.sqlite"
    try:
        if a.phase == "l3":
            reuse = (info.get("l4_reuse") or {}).get("reuse_ok", False)
            dec = next((r for r in rec.records if r.get("unit") == "reuse_decision"), None)
            if dec is None:
                dec = rec.emit("reuse_decision", unit="reuse_decision", reuse_ok=reuse,
                               detail=info.get("l4_reuse"), predeclared_rule="L4 stage code diff vs the Phase 0 "
                               "commit must be empty, else all 224 units run fresh")
            reuse = dec["reuse_ok"]
            p0_results = a.p0_results or a.p0_job / "out" / "results.jsonl"
            if reuse:
                if not p0_results.exists():
                    rec.emit("precondition_failed", reason=f"Phase 0 results missing: {p0_results}")
                    return X.RC_PRECOND
                if "reuse_seeded" not in rec.done:
                    rec.emit("reuse_seeded", unit="reuse_seeded", **X.seed_reuse(rec, p0_results))
            src = (a.store_src or a.p0_job / "out" / "store.sqlite") if reuse else a.l3_store_src
            res = X.run_l3(rec, a.out, inp, sizes, a.orch_root, store_src=src, guard=guard, smoke=smoke,
                           corpus_id=a.corpus_id, orch_pin=a.orch_pin, research_pin=a.research_pin)
        elif a.phase in ("pilot", "ingest"):
            dec = next((r for r in rec.records if r.get("unit") == "reuse_decision"), None)
            if dec is None or not store_path.exists():
                rec.emit("precondition_failed", reason="run --phase l3 first (no reuse decision / store)")
                return X.RC_PRECOND
            from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
            backend = LlamaServerBackend(a.url, model_id=a.model_id)
            store = P.open_store(store_path)
            res = X.run_l4_stage(rec, a.out, store, backend, inp, sizes, stage=a.phase, reuse_ok=dec["reuse_ok"],
                                 guard=guard, smoke=smoke, corpus_id=a.corpus_id, booked_l4_s=a.booked_l4_s)
        else:  # gpu-audit | gpu-answers | score
            cpu_results = a.cpu_results
            rj = a.cpu_result_json or (cpu_results.parent / "result.cpu.json" if cpu_results else None)
            if not cpu_results or not cpu_results.exists() or not rj or not rj.exists():
                rec.emit("precondition_failed", reason="--cpu-results / result.cpu.json missing")
                return X.RC_PRECOND
            ready, why = X.cpu_ready(json.loads(rj.read_text()))
            if not ready:
                rec.emit("precondition_failed", reason=f"CPU result not ready: {why}")
                return X.RC_PRECOND
            if a.phase != "score" and not store_path.exists():
                if not a.store_src or not a.store_src.exists():
                    rec.emit("precondition_failed", reason=f"no store at {store_path} and no --store-src")
                    return X.RC_PRECOND
                P.copy_store(a.store_src, store_path)
                rec.emit("store_copied", src=str(a.store_src), sha256=X.sha256_file(store_path), final=False)
            store = P.open_store(store_path) if store_path.exists() else None
            names = sorted({d["company"] for d in inp["corpus"]["docs"]})
            cpu_recs = X.read_jsonl(cpu_results)
            if a.phase == "score":
                res = {"complete": True, "verdict": X.verdict(cpu_recs, rec.records, agg,
                                                              n_excluded=info["n_agg_excluded"], store=store)}
            else:
                from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
                backend = LlamaServerBackend(a.url, model_id=a.model_id)
                if a.phase == "gpu-audit":
                    res = P.run_audit(rec, store, backend, guard=guard, smoke=smoke)
                else:
                    res = X.run_answers(rec, store, backend, agg=agg, names=names, guard=guard, smoke=smoke,
                                        corpus_id=a.corpus_id)
                    if res.get("complete"):
                        res["verdict"] = X.verdict(cpu_recs, rec.records, agg, n_excluded=info["n_agg_excluded"],
                                                   store=store)
                        rec.emit("verdict", unit="verdict", **res["verdict"])
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return X.RC_PARTIAL
    rc = res.pop("rc", None)
    res.setdefault("phase", a.phase)
    res["predeclared_sha256"] = pred_sha
    if not smoke and a.phase not in ("pilot", "ingest") and res.get("complete") is not None:
        X._atomic_write(a.out / f"result.{a.phase}.json", json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    if rc is not None:
        return rc
    return 0 if (res.get("complete") or smoke) else X.RC_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
