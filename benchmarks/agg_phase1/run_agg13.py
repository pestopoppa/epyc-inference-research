#!/usr/bin/env python3
"""AGG-1.3 runner CLI (keeper jobs run it under normal grants; logic in ``agg13.py``). Code only: no model, server or
process is started here.

``--phase``:

  preflight     zero-compute checks (pins, 28 PDFs, L4 reuse test, base store + parse-cache hash check)
  reingest      CPU: ingest the 28 documents (25 skipped by hash, 3 replaced), L3 + L4 for the 3, G1-G4, result.cpu.json
  gpu-audit     the Phase 0 L4c audit over the changed store (27B np4)
  gpu-answers   P-large on the 58 questions (no B1, not scored), G5 and the behavioural half of G2, the verdict
  score         verdict only

``--dry-run`` validates every input, prints the plan and exits: zero model calls, no server, no store, nothing written.
Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped, 4 precondition, 5 budget stop, 75 partial.
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
import agg13 as G  # noqa: E402
import phase1_arms as X  # noqa: E402
from phase1_arms import A, P  # noqa: E402

PHASES = ("preflight", "reingest", "gpu-audit", "gpu-answers", "score")
DEFAULT_CACHE_DIRS = (Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/cache"),
                      Path("/mnt/raid0/llm/tmp/agg0-l3-load-20261110-r3/out/cache"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, default=None, help="pinned orchestrator tree; prepended to sys.path")
    ap.add_argument("--orch-pin", default="", help="expected <orch-root>/.pin content, when that file exists")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--agg11-job", type=Path, default=G.DEFAULT_AGG11_JOB, help="AGG-1.1 CPU job dir (L4 reuse pin)")
    ap.add_argument("--agg11-results", type=Path, default=None, help="default AGG11-JOB/out/results.jsonl")
    ap.add_argument("--base-store", type=Path, default=None,
                    help="the AGG-1.1 store copy this job starts from (copied to OUT/store.sqlite once)")
    ap.add_argument("--cache-dir", type=Path, action="append", default=None,
                    help="content-keyed parse caches of the retained documents (repeatable)")
    ap.add_argument("--pins", choices=("agg13", "none"), default="agg13", help="'none' is for fixture sets only")
    ap.add_argument("--corpus-id", default=P.CORPUS_ID)
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--store-src", type=Path, help="gpu phases: the CPU job's store, copied once")
    ap.add_argument("--cpu-results", type=Path, help="gpu phases: the CPU job's results.jsonl")
    ap.add_argument("--cpu-result-json", type=Path, help="default beside --cpu-results")
    return ap


def orch_pin_problem(a: argparse.Namespace) -> str | None:
    if a.orch_root is None or not a.orch_pin:
        return None
    f = a.orch_root / ".pin"
    if f.exists() and f.read_text().strip() != a.orch_pin:
        return f"{f} = {f.read_text().strip()} != expected orchestrator pin {a.orch_pin}"
    return None


def checks(a: argparse.Namespace, inp: dict[str, Any], *, deep: bool) -> tuple[list[str], dict[str, Any]]:
    """Every zero-compute check. ``deep`` also imports the orchestrator and hashes the retained parse caches."""
    problems: list[str] = []
    info: dict[str, Any] = {}
    if a.pins == "agg13":
        problems += G.pin_problems(a.set_dir)
    problems += G.corpus_problems(inp["corpus"], inp["corpus11"], check_pdfs=a.phase not in ("gpu-audit", "gpu-answers",
                                                                                           "score"))
    problems += G.cells_problems(inp)
    if a.pins == "agg13":
        ok11, info11 = X.preflight(inp | {"corpus": inp["corpus11"]}, a.set_dir, orch_root=None, p0_job=None,
                                   check_pdfs=False, run_build_check=True,
                                   sizes=X.load_sizes(a.set_dir / "extracted_sizes.json"))
    else:  # fixture sets: no frozen pins, no build check
        ok11, info11 = X.preflight(inp | {"corpus": inp["corpus11"]}, a.set_dir, pins=None, predeclared_sha=None,
                                   check_pdfs=False, run_build_check=False, expect_cells=None, strict_p0_pins=False)
    problems += [f"AGG-1.1 inputs: {p}" for p in info11["problems"]]
    info["agg11_inputs"] = {k: v for k, v in info11.items() if k != "problems"}
    cells = G.m1_cells13(inp)
    n_new = sum(1 for c in cells if c[0] in G.REPLACED)
    info["m1_cells"] = {"total": len(cells), "retained": len(cells) - n_new, "changed": n_new}
    if a.pins == "agg13" and (len(cells) - n_new, n_new) != (187, 24):
        problems.append(f"M1 cells {len(cells) - n_new} retained + {n_new} changed, PREDECLARED has 187 + 24")
    b = G.g3_bounds()
    info["g3_bounds"] = {k: round(v, 2) for k, v in b.items()}
    p = orch_pin_problem(a)
    if p:
        problems.append(p)
    if a.orch_root is not None and a.agg11_job.exists():
        info["l4_reuse"] = G.l4_reuse_check13(a.agg11_job, a.orch_root)
        if not info["l4_reuse"].get("reuse_ok"):
            problems.append("L4 reuse test failed: the retained units are not cache hits")
    if deep and a.orch_root is not None:
        sys.path.insert(0, str(a.orch_root))
        os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
        try:
            from src.aggregation import planner, store as st  # noqa: PLC0415
            assert hasattr(st.CorpusStore, "superseded_facts") and hasattr(st.CorpusStore, "refresh_canonical")
            assert hasattr(planner, "scoped_facts")
        except Exception as e:  # noqa: BLE001
            problems.append(f"orchestrator tree {a.orch_root} cannot serve the phases: {type(e).__name__}: {e}")
        else:
            base = a.base_store
            dirs = list(a.cache_dir or DEFAULT_CACHE_DIRS)
            if base is not None and base.exists():
                hc = G.base_hash_check(inp["corpus"], G.read_doc_hashes(base, a.corpus_id),
                                       lambda d: (G.load_cached(d, dirs) or (None, None))[0])
                info["base_hash_check"] = {k: v for k, v in hc.items() if k != "base_hashes"}
                if hc["retained_unequal"] or hc["retained_no_cache"] or not hc["slots_present"] \
                        or hc["n_base_docs"] != G.N_DOCS:
                    problems.append(f"base store / parse cache check failed: {info['base_hash_check']}")
            else:
                problems.append(f"base store missing: {base}")
    return problems, info


def dry_run(a: argparse.Namespace, inp: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    problems, info = checks(a, inp, deep=True)
    corpus = inp["corpus"]
    order = [d for d in G.doc_names(corpus) if d not in G.REPLACED] + list(G.REPLACED)
    out: dict[str, Any] = {
        "dry_run": True, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
        "predeclared_sha256": G.SHA256(a.set_dir / "PREDECLARED_AGG13.md") if (a.set_dir / "PREDECLARED_AGG13.md").exists() else None,
        "problems": problems, "checks": info,
        "plan": {"documents": len(order), "retained_skipped_by_hash_expected": G.N_DOCS - G.N_REPLACED,
                 "changed_reparsed_reextracted_expected": G.N_REPLACED, "replaced_slots": list(G.REPLACED),
                 "replacement_files": [d["file_name"] for d in G.replaced_docs(corpus)],
                 "l4_units_to_run": G.N_REPLACED * len(A.METRICS), "l4_units_seeded_as_cache_hits":
                 (G.N_DOCS - G.N_REPLACED) * len(A.METRICS), "order": order,
                 "gates": {"G1": "exactly 3 re-parsed and re-extracted, 25 skipped by hash",
                           "G2": "old hashes superseded (ledger) and absent from live tables / canonical / SQL answers",
                           "G3": f"fresh calls <= {info['g3_bounds']['llm_calls']:.1f} and prefill <= "
                                 f"{info['g3_bounds']['prefill_tokens']:.0f} (3/28 x full AGG-1.1 ingest x 1.10)",
                           "G4": f"M1 >= {G.M1_GATE} over {info['m1_cells']['total']} cells (25 retained FY2022 + 3 new FY2021)",
                           "G5": f"GPU row: median P-large calls <= {G.M4_MEDIAN_CALLS_MAX}, prefill <= {G.M4_MEDIAN_PREFILL_MAX}"},
                 "budget_stop": {"first_changed_doc": G.REPLACED[0], "limit": {k: G.FIRST_DOC_X * v
                                                                               for k, v in G.AGG11_PILOT[G.REPLACED[0]].items()}}}}
    if a.phase in ("gpu-audit", "gpu-answers", "score"):
        ready = None
        if a.cpu_results:
            rj = a.cpu_result_json or a.cpu_results.parent / "result.cpu.json"
            ready = G.cpu_ready(json.loads(rj.read_text())) if rj.exists() else (False, f"{rj} not present yet")
        out["precondition_cpu"] = {"ready": ready[0] if ready else False, "detail": ready[1] if ready else "no --cpu-results",
                                   "note": "informational in a dry run: the CPU row has not run yet"}
        out["plan"]["p_large"] = {"n": len(X.verified_agg(inp)), "scored": False}
    out["ok"] = not problems
    return (0 if not problems else X.RC_PREFLIGHT), out


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    X.configure(a.corpus_id)
    try:
        inp = G.load_inputs(a.set_dir)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load inputs: {type(e).__name__}: {e}"]}))
        return X.RC_PREFLIGHT
    if a.base_store is None:
        a.base_store = (a.agg11_job / "out" / "store.sqlite")
    if a.dry_run:
        rc, out = dry_run(a, inp)
        print(json.dumps(out, indent=1, default=str))
        return rc
    if a.out is None:
        build_parser().error("--out is required to run (use --dry-run to only validate)")
    if a.phase != "preflight" and a.orch_root is None:
        build_parser().error("--orch-root (the pinned orchestrator tree) is required to run this phase")
    gpu = a.phase in ("gpu-audit", "gpu-answers", "score")
    problems, info = checks(a, inp, deep=not gpu)
    print(json.dumps({"problems": problems, **info}, indent=1, default=str))
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    if problems or a.phase == "preflight":
        rec.emit("preflight", unit=None, ok=not problems, final=False, problems=problems,
                 **{k: v for k, v in info.items() if k != "l4_reuse"}, l4_reuse_ok=(info.get("l4_reuse") or {}).get("reuse_ok"))
        X._atomic_write(a.out / "result.preflight.json", json.dumps(
            {"phase": "preflight", "complete": not problems, "problems": problems,
             "predeclared_sha256": G.PREDECLARED_SHA}, indent=1, default=str) + "\n")
        return X.RC_PREFLIGHT if problems else 0
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=G.PREDECLARED_SHA, orch_pin=a.orch_pin,
             research_pin=a.research_pin, corpus_id=a.corpus_id, url=a.url, model_id=a.model_id, cfg=dict(vars(P.CFG)),
             m1_cells=info["m1_cells"], g3_bounds=info["g3_bounds"], final=False)
    agg = X.verified_agg(inp)
    store_path = a.out / "store.sqlite"
    try:
        if a.phase == "reingest":
            from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
            backend = LlamaServerBackend(a.url, model_id=a.model_id)
            res = G.run_reingest(rec, a.out, inp, a.orch_root, backend, base_store=a.base_store,
                                 agg11_results=a.agg11_results or a.agg11_job / "out" / "results.jsonl",
                                 cache_dirs=list(a.cache_dir or DEFAULT_CACHE_DIRS), guard=guard, smoke=smoke,
                                 corpus_id=a.corpus_id, reuse_ok=bool(info["l4_reuse"]["reuse_ok"]))
        else:
            cpu_results = a.cpu_results
            rj = a.cpu_result_json or (cpu_results.parent / "result.cpu.json" if cpu_results else None)
            if not cpu_results or not cpu_results.exists() or not rj or not rj.exists():
                rec.emit("precondition_failed", reason="--cpu-results / result.cpu.json missing")
                return X.RC_PRECOND
            cpu_json = json.loads(rj.read_text())
            ready, why = G.cpu_ready(cpu_json)
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
            if a.phase == "score":
                res = {"complete": True, "verdict": G.verdict13(cpu_json, rec.records, agg, store, a.corpus_id)}
            else:
                from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
                backend = LlamaServerBackend(a.url, model_id=a.model_id)
                if a.phase == "gpu-audit":
                    res = P.run_audit(rec, store, backend, guard=guard, smoke=smoke)
                else:
                    res = G.run_p_large(rec, store, backend, agg=agg, names=names, guard=guard, smoke=smoke,
                                        corpus_id=a.corpus_id)
                    if res.get("complete"):
                        res["verdict"] = G.verdict13(cpu_json, rec.records, agg, store, a.corpus_id)
                        rec.emit("verdict", unit="verdict", **res["verdict"])
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return X.RC_PARTIAL
    rc = res.pop("rc", None)
    res.setdefault("phase", a.phase)
    res["predeclared_sha256"] = G.PREDECLARED_SHA
    if not smoke and a.phase not in ("reingest",) and res.get("complete") is not None:
        X._atomic_write(a.out / f"result.{a.phase}.json", json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    if rc is not None:
        return rc
    return 0 if (res.get("complete") or smoke) else X.RC_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
