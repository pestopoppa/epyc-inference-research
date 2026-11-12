#!/usr/bin/env python3
"""AGG-1.12 runner CLI (keeper jobs run it under normal grants; logic in ``agg112.py``). Code only: no model, server or process is
started here.

``--phase`` (PREDECLARED_AGG112.md checkpoints in parentheses):

  preflight    (0)  zero-compute checks and the offline plan (pins, frozen inputs, the pipeline stage equals arm T, the window plan, the M3
                    projection against the Phase 0 kill bounds)
  l3           (1)  copy of the AGG-1.1 store, facts wiped, L3 re-extracted under ``l3_period.v2`` (no model, no PDF)
  pilot        (2)  typed L4 on MICROSOFT, JOHNSON_JOHNSON, 3M (<= 1800 s) and its futility checks
  ingest       (3,4) the other 25 documents, interim look after 14, final look (M1, M3, regressions); writes ``result.cpu.json``
  gpu-audit    (5a) L4c audit sample (the AGG-1.1 recipe, unmodified)
  gpu-answers  (5b) P-large on the 58 under ``agg110.v1`` then ``agg11.v0`` on the canonical.v2 store, then the verdict
  score             verdict only (no model)

``--dry-run`` validates every input, prints the plan (calls, M3 projection, estimates, gates, partition counts) and exits: zero model
calls, no server, no store copy, nothing written. Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped, 4
precondition, 5 budget stop (re-book, not a gate), 75 partial (resumable).
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
import agg112 as G  # noqa: E402
import phase1_arms as X  # noqa: E402
from phase1_arms import A, P  # noqa: E402

PHASES = ("preflight", "l3", "pilot", "ingest", "gpu-audit", "gpu-answers", "score")
DEFAULT_SEED = Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/store.sqlite")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; no model, no writes")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, default=None)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--seed-store", type=Path, default=DEFAULT_SEED, help="CPU: the AGG-1.1 store (read-only seed; copied once)")
    ap.add_argument("--store-src", type=Path, default=None, help="GPU: the CPU row's store (copied once to OUT/store.sqlite)")
    ap.add_argument("--cpu-result-json", type=Path, default=None, help="GPU: the CPU row's result.cpu.json")
    ap.add_argument("--cpu-results", type=Path, default=None, help="GPU: the CPU row's results.jsonl (informational)")
    ap.add_argument("--pins", choices=("agg112", "none"), default="agg112")
    ap.add_argument("--booked-l4-s", type=float, default=None)
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def orch_import(a: argparse.Namespace) -> list[str]:
    if a.orch_root is None:
        return ["--orch-root is required"]
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    return G.orch_problems(a.orch_root, a.orch_pin)


def checks(a: argparse.Namespace, inp: dict[str, Any], *, deep: bool) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    info: dict[str, Any] = {}
    if a.pins == "agg112":
        problems += G.pin_problems(a.set_dir)
    ref = inp["reference"]
    if ref is None:
        problems.append(f"{G.REFERENCE_FILE} missing")
        return problems, info
    ok11, info11 = X.preflight(inp, a.set_dir, check_pdfs=False, run_build_check=True, sizes=inp["sizes"])
    problems += [f"AGG-1.1 inputs: {p}" for p in info11["problems"]]
    problems += orch_import(a)
    part = ref["partition"]
    info["reference"] = {"agg11_raw_found": ref["agg11"]["raw_found"], "agg11_canonical_v1_ok": ref["agg11"]["canonical_v1_ok"],
                         "agg11_p_large": {k: v for k, v in ref["agg11"]["p_large"].items() if k != "per_question"},
                         "n_touched": part["n_touched"], "n_held_out": part["n_held_out"],
                         "questions_with_only_untouched_evidence": ref["questions"]["n_with_only_untouched_evidence"],
                         "agg11_hit_rate_on_held_out": ref["partition_hit_rates_agg11"]["held_out"]}
    if a.phase in ("preflight", "l3", "pilot", "ingest"):
        if not a.seed_store.exists():
            problems.append(f"seed store missing: {a.seed_store}")
        elif deep and not problems:
            from src.aggregation.store import CorpusStore  # noqa: PLC0415
            import agg16 as G16  # noqa: PLC0415
            docs = [d["doc_name"] for d in inp["corpus"]["docs"]]
            store = G16.mem_store(a.seed_store, docs)
            try:
                n_docs, n_chunks = len(store.corpus_docs(G.CORPUS_ID)), len(store.chunks(G.CORPUS_ID))
                if (n_docs, n_chunks) != (28, 6094):
                    problems.append(f"seed store has {n_docs} documents / {n_chunks} chunks, expected 28 / 6094")
                problems += G.parity_problems(store, inp["corpus"], ("MICROSOFT_2022_10K", "COSTCO_2022_10K"))
                plans = G.doc_plans(store, inp["corpus"])
                stats = X.doc_stats(store, G.CORPUS_ID)
                plan = G.plan_m3(plans, stats)
                info["plan"] = {
                    "pipeline": G.PIPELINE, "l4_reuse": "none: every one of the 28 documents gets a fresh typed L4 (the v2 stage is not "
                    "identical to any AGG-1.1 / Phase 0 unit); the L0 parse (documents, chunks, tables, pages) is reused from the AGG-1.1 "
                    "store because it does not depend on the pipeline version; L3 is re-extracted under l3_period.v2 (no model)",
                    "order": G.doc_order(inp["corpus"], inp["sizes"]), "pilot": {"docs": list(G.PILOT_DOCS), "bound_s": G.PILOT_WALL_S},
                    "per_doc": {d: {"windows_called": v["n_windows"], "of": v["n_doc_windows"], "chars": v["chars"]} for d, v in plans.items()},
                    "m3_projection": plan, "estimate": G.estimates(plans),
                    "pilot_projection": G.plan_totals(plans, G.PILOT_DOCS),
                    "gates": {"pilot_futility": [G.M1_PILOT, "projected M3 over a kill bound"],
                              "interim": f"after {G.INTERIM_DOCS} documents: futility only (M1 below AGG-1.1 on the same cells, or M3 over a kill bound)",
                              "final_cpu": ["M1 found >= AGG-1.1 (185 of 211)", "M3 inside the Phase 0 kill bounds (prefill <= 2.0x, calls <= 150 per 100 chunks)"],
                              "reported_not_gated_for_gpu": "regressions of previously correct cells (raw or canonical): verdict agg_1_12_not_met",
                              "gpu": "M4 median calls <= 3 and median prefill <= 2510 for BOTH planners"},
                    "n_cells": len(inp["cells"]), "n_questions": len(X.verified_agg(inp))}
                if not plan["inside_kill_bounds"]:
                    problems.append("the offline plan projects M3 over a kill bound")
                if plan["m3"]["prefill_ratio"] >= plan["agg11"]["prefill_ratio"] or plan["m3"]["calls_per_100_chunks"] >= plan["agg11"]["calls_per_100_chunks"]:
                    info["plan"]["note_m3"] = "projected M3 is NOT below AGG-1.1 on both axes"
            finally:
                store.close()
    if a.phase in ("gpu-audit", "gpu-answers", "score"):
        ready = None
        if a.cpu_result_json is not None and a.cpu_result_json.exists():
            ready = G.cpu_ready(json.loads(a.cpu_result_json.read_text()), ref)
        info["precondition_cpu"] = {"ready": ready[0] if ready else False, "detail": ready[1] if ready else "no result.cpu.json yet",
                                    "note": "informational in a dry run: the CPU row has not run yet"}
        agg = X.verified_agg(inp)
        info["gpu_plan"] = {"audit": "L4c sample computed at run time from the CPU store (confidence <= medium plus a 5% hash sample)",
                            "p_large": {"n": len(agg), "planners": list(G.PLANNERS), "passes": len(G.PLANNERS),
                                        "calls_per_question": "2 (SQL + answer), 3 with one agg110.v1 re-ask; agent path up to 30"},
                            "arms_not_run": ["B1", "P-small", "B-today", "B2"], "store": "the CPU row's store, canonical.v2 attributable_net_income"}
    return problems, info


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    X.configure(G.CORPUS_ID)
    try:
        inp = G.load_inputs(a.set_dir)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load inputs: {type(e).__name__}: {e}"]}))
        return G.RC_PREFLIGHT
    if a.dry_run or a.phase == "preflight":
        problems, info = checks(a, inp, deep=True)
        out = {"dry_run": a.dry_run, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
               "predeclared_sha256": G.SHA256(a.set_dir / G.PREDECLARED_FILE) if (a.set_dir / G.PREDECLARED_FILE).exists() else None,
               "problems": problems, "ok": not problems, **info}
        print(json.dumps(out, indent=1, default=str))
        if a.phase == "preflight" and not a.dry_run:
            if a.out is None:
                build_parser().error("--out is required to run (use --dry-run to only validate)")
            a.out.mkdir(parents=True, exist_ok=True)
            rec = P.Recorder(a.out / "results.jsonl")
            rec.emit("preflight", unit=None, ok=not problems, final=False, problems=problems)
            X._atomic_write(a.out / "result.preflight.json", json.dumps(
                {"phase": "preflight", "complete": not problems, "problems": problems, "predeclared_sha256": G.PREDECLARED_SHA},
                indent=1, default=str) + "\n")
        return G.RC_PREFLIGHT if problems else 0
    if a.out is None:
        build_parser().error("--out is required to run (use --dry-run to only validate)")
    problems, info = checks(a, inp, deep=False)
    print(json.dumps({"problems": problems, **{k: v for k, v in info.items() if k != "plan"}}, indent=1, default=str))
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    if problems:
        rec.emit("preflight", unit=None, ok=False, final=False, problems=problems)
        return G.RC_PREFLIGHT
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=G.PREDECLARED_SHA, orch_pin=a.orch_pin, research_pin=a.research_pin,
             corpus_id=G.CORPUS_ID, pipeline=G.PIPELINE, url=a.url, model_id=a.model_id, cfg=dict(vars(P.CFG)), final=False)
    ref = inp["reference"]
    try:
        if a.phase == "l3":
            res = G.run_l3(rec, a.out, inp, seed_store=a.seed_store, guard=guard, smoke=smoke)
        elif a.phase in ("pilot", "ingest"):
            if not a.url:
                build_parser().error("--url is required for this phase")
            from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
            backend = LlamaServerBackend(a.url, model_id=a.model_id)
            res = G.run_ingest(rec, a.out, inp, backend, stage=a.phase, guard=guard, smoke=smoke, booked_l4_s=a.booked_l4_s)
        else:  # gpu-audit | gpu-answers | score
            rj = a.cpu_result_json
            if rj is None or not rj.exists():
                rec.emit("precondition_failed", reason="--cpu-result-json missing or not present yet")
                return G.RC_PRECOND
            cpu = json.loads(rj.read_text())
            ready, why = G.cpu_ready(cpu, ref)
            if not ready:
                rec.emit("precondition_failed", reason=f"CPU result not ready: {why}")
                return G.RC_PRECOND
            store_path = a.out / "store.sqlite"
            if a.phase != "score" and not store_path.exists():
                if not a.store_src or not a.store_src.exists():
                    rec.emit("precondition_failed", reason=f"no store at {store_path} and no --store-src")
                    return G.RC_PRECOND
                P.copy_store(a.store_src, store_path)
                rec.emit("store_copied", src=str(a.store_src), sha256=G.SHA256(store_path), final=False)
            store = P.open_store(store_path) if store_path.exists() else None
            agg = X.verified_agg(inp)
            names = sorted({d["company"] for d in inp["corpus"]["docs"]})
            if a.phase == "score":
                res = {"complete": True, "verdict": G.verdict(cpu, rec.records, agg, ref, store=store)}
            else:
                from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
                if not a.url:
                    build_parser().error("--url is required for this phase")
                backend = LlamaServerBackend(a.url, model_id=a.model_id)
                if a.phase == "gpu-audit":
                    res = P.run_audit(rec, store, backend, guard=guard, smoke=smoke)
                    if smoke and res.get("n_queue") == 0 and "audit_done" not in rec.done:
                        rec.emit("audit_done", unit="audit_done", n_queue=0, disputed=0)
                else:
                    res = G.run_answers(rec, store, backend, agg=agg, names=names, guard=guard, smoke=smoke)
                    if res.get("complete"):
                        v = G.verdict(cpu, rec.records, agg, ref, store=store)
                        res["verdict"] = v
                        rec.emit("verdict", unit="verdict", **v)
                        res["rc"] = G.RC_GATE if v.get("gates_tripped") else G.RC_OK
            if store is not None:
                store.close()
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return G.RC_PARTIAL
    rc = res.pop("rc", None)
    res.setdefault("phase", a.phase)
    res["predeclared_sha256"] = G.PREDECLARED_SHA
    if not smoke and a.phase in ("l3", "gpu-audit", "gpu-answers", "score") and res.get("complete") is not None:
        X._atomic_write(a.out / f"result.{a.phase}.json", json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "summaries"}, indent=1, default=str))
    if rc is not None:
        return rc
    return 0 if (res.get("complete") or smoke) else G.RC_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
