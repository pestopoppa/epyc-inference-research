#!/usr/bin/env python3
"""AGG-1.10 runner CLI (keeper jobs run it under normal grants; logic in ``agg110.py``). Code only: no model, server or
process is started here.

``--phase``:

  preflight  zero-compute checks (pins, frozen ceiling/pilot, AGG-1.1 inputs, orchestrator planner, routing report, store)
  answers    GPU: the 26 questions under planner ``agg110.v1``, pilot (6) first, futility stop, then the rest; M4 and the
             before/after verdict -> ``result.gpu.json``
  score      verdict only (no model), from the records already written

``--dry-run`` validates every input, prints the plan (question order, pilot, ceiling, routing, gates, planner versions) and
exits: zero model calls, no server, no store copy, nothing written. The source store is read ``immutable=1`` only.
Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped, 4 precondition, 5 budget stop, 75 partial.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
import agg110 as G  # noqa: E402
import phase1_arms as X  # noqa: E402
from phase1_arms import A, P  # noqa: E402

PHASES = ("preflight", "answers", "score")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, default=None)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--store-src", type=Path, default=None, help="the AGG-1.1 store (read only); copied once to OUT/store.sqlite")
    ap.add_argument("--agg11-results", type=Path, default=None, help="AGG-1.1 results.jsonl (reference M4, informational)")
    ap.add_argument("--pins", choices=("agg110", "none"), default="agg110")
    ap.add_argument("--planner-version", choices=(G.V0, G.V1), default=G.V1)
    ap.add_argument("--corpus-id", default=P.CORPUS_ID)
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def load(set_dir: Path) -> dict[str, Any]:
    inp = X.load_inputs(corpus=set_dir / "corpus.json", golds=set_dir / "golds.jsonl",
                        verification=set_dir / "verification_record.json", facts_sheet=set_dir / "facts_sheet.csv")
    inp["frozen"] = G.load_frozen(set_dir)
    return inp


def orch_import(a: argparse.Namespace) -> tuple[Any, list[str]]:
    problems: list[str] = []
    if a.orch_root is None:
        return None, ["--orch-root is required"]
    f = a.orch_root / ".pin"
    if a.orch_pin and f.exists() and f.read_text().strip() != a.orch_pin:
        problems.append(f"{f} = {f.read_text().strip()} != expected orchestrator pin {a.orch_pin}")
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    try:
        from src.aggregation import planner, sql_grammar  # noqa: PLC0415
        assert hasattr(planner, "sql_shape") and hasattr(sql_grammar, "build_pivot_gbnf") \
            and hasattr(sql_grammar, "create_pivot_view") and hasattr(sql_grammar, "null_guard")
        assert sql_grammar.PLANNER_V1 == G.V1 and sql_grammar.SQL_N_PREDICT_V1 == G.SQL_N_PREDICT[G.V1]
    except Exception as e:  # noqa: BLE001
        return None, problems + [f"orchestrator tree {a.orch_root} lacks the agg110.v1 plan: {type(e).__name__}: {e}"]
    return planner, problems


def store_info(path: Path, corpus_id: str) -> dict[str, Any]:
    c = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    try:
        return {"path": str(path), "sha256": X.sha256_file(path),
                "documents": c.execute("SELECT COUNT(*) FROM documents WHERE corpus_id = ?", [corpus_id]).fetchone()[0],
                "facts_canonical": c.execute("SELECT COUNT(*) FROM facts_canonical WHERE corpus_id = ?", [corpus_id]).fetchone()[0]}
    finally:
        c.close()


def saved_p_large(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {r["qid"]: r for r in X.read_jsonl(path)
            if r.get("kind") == "answer" and r.get("arm") == "P-large" and r.get("final")}


def checks(a: argparse.Namespace, inp: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    info: dict[str, Any] = {}
    if a.pins == "agg110":
        problems += G.pin_problems(a.set_dir, frozen=G.FROZEN_ALL)
        ok11, info11 = X.preflight(inp, a.set_dir, check_pdfs=False, run_build_check=True,
                                   sizes=X.load_sizes(a.set_dir / "extracted_sizes.json"))
        problems += [f"AGG-1.1 inputs: {p}" for p in info11["problems"]]
    problems += G.phase0_pin_problems(a.set_dir)
    golds = {g["id"]: g for g in inp["golds"]}
    fz = inp["frozen"]
    problems += G.frozen_problems(fz, golds)
    ids = [r["qid"] for r in G.affected(fz["miss"])]
    info["questions"] = {"n": len(ids), "ids": ids, "pilot": fz["pilot"]["qids"],
                         "ceiling_true": fz["ceiling"]["n_ceiling_true"],
                         "longest_statement_tokens_est": fz["ceiling"]["longest_statement_tokens_est"]}
    info["planner_versions"] = {"default": G.V0, "this_run": a.planner_version, "sql_n_predict": G.SQL_N_PREDICT}
    planner, p = orch_import(a)
    problems += p
    if planner is not None:
        agg = X.verified_agg(inp)
        info["routing"] = {k: v for k, v in G.routing_report(agg, planner, saved_p_large(a.agg11_results), ids).items()
                           if k != "rows"}
        if info["routing"]["pivot_in_26"] != 26:
            problems.append(f"{info['routing']['pivot_in_26']} of 26 route to the pivot plan, PREDECLARED expects all of them to "
                            "be recorded as routed; check sql_shape")
        from src.aggregation import sql_grammar as SG  # noqa: PLC0415
        v0_ok = SG.PLANNER_V0 == G.V0
        info["v0_default"] = v0_ok
        if not v0_ok:
            problems.append("orchestrator PLANNER_V0 is not agg11.v0")
    if a.store_src is not None:
        if not a.store_src.exists():
            problems.append(f"source store missing: {a.store_src}")
        else:
            info["store"] = store_info(a.store_src, a.corpus_id)
            if info["store"]["documents"] != 28:
                problems.append(f"source store has {info['store']['documents']} documents, expected 28")
    return problems, info


def dry_run(a: argparse.Namespace, inp: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    problems, info = checks(a, inp)
    fz = inp["frozen"]
    ids = [r["qid"] for r in G.affected(fz["miss"])]
    order = G.question_order(ids, fz["pilot"]["qids"])
    pd = a.set_dir / G.PREDECLARED_FILE
    out = {"dry_run": True, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
           "predeclared_sha256": G.SHA256(pd) if pd.exists() else None, "problems": problems, "checks": info,
           "plan": {"planner_version": a.planner_version, "n": len(order), "order": order,
                    "pilot": {"qids": fz["pilot"]["qids"], "bound_s": G.PILOT_WALL_S},
                    "expected_calls_per_question": "2 (SQL + answer), 3 with one re-ask",
                    "gates": {"futility": "pilot fixes 0 of 6 -> phase1_gate_tripped, the other 20 not run",
                              "budget": f"pilot wall > {G.PILOT_WALL_S} s -> budget stop (re-book), not a gate",
                              "M4": f"median calls <= {G.M4_CALLS_MAX}, median prefill <= {G.M4_PREFILL_MAX} over every question run"},
                    "estimate": f"{G.BOOKED_S:.0f} s ESTIMATE (box value)"}}
    out["ok"] = not problems
    return (0 if not problems else X.RC_PREFLIGHT), out


def run_answers(rec: P.Recorder, out: Path, inp: dict[str, Any], planner, backend, *, store_src: Path, guard,
                smoke: bool, corpus_id: str, planner_version: str) -> dict[str, Any]:
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    fz = inp["frozen"]
    agg_all = X.verified_agg(inp)
    ids = [r["qid"] for r in G.affected(fz["miss"])]
    byid = {g["id"]: g for g in agg_all}
    order = G.question_order(ids, fz["pilot"]["qids"])
    store_path = out / "store.sqlite"
    if not store_path.exists():
        P.copy_store(store_src, store_path)
        rec.emit("store_copied", src=str(store_src), sha256=X.sha256_file(store_path), final=False)
    store = P.open_store(store_path)
    try:
        corpus_chunks = store.chunks(corpus_id)
        corpus_bm25 = BM25Index([c["text"] for c in corpus_chunks])
        docs_meta = store.corpus_docs(corpus_id)
        sql_present = bool(store.facts(corpus_id))
        built = sorted(store.built_layers(corpus_id))
        canon = store.refresh_canonical(corpus_id)
        if "canonical" not in rec.done:
            rec.emit("canonical", unit="canonical", planner_version=planner_version, **canon)
        scopes = P._scope_plan(planner, docs_meta, [("P-large", "agg", q) for q in agg_all])
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for c in corpus_chunks:
            by_doc.setdefault(c["doc_id"], []).append(c)
        names = sorted({d["company"] for d in inp["corpus"]["docs"]})
        for qid in order:
            q = byid[qid]
            unit = f"answer:P-large:{qid}"
            if unit not in rec.done:
                guard()
                t, t0 = P.Tally(backend), time.time()
                text, st, extra = G.p_large_answer_v(
                    backend, store, q, scope=scopes[unit], planner=planner, docs_meta=docs_meta, built=built,
                    sql_present=sql_present, names=names, cfg=P.CFG, corpus_chunks=corpus_chunks, corpus_bm25=corpus_bm25,
                    doc_chunks=lambda d: by_doc.get(d, []), planner_version=planner_version, corpus_id=corpus_id)
                ok = P.score_agg(q, text) if st == "ok" else None
                rec.emit("answer", unit=unit, arm="P-large", qid=qid, qclass="agg", status=st, correct=ok, calls=t.calls,
                         prefill_tokens=t.prefill, answer=text[-600:],
                         final=st in ("ok", "out_of_window", "budget_exhausted"), wall_s=time.time() - t0, **extra)
                if smoke:
                    return {"complete": False, "smoke": True, "rc": X.RC_OK}
            if "pilot_done" not in rec.done and all(f"answer:P-large:{p}" in rec.done for p in fz["pilot"]["qids"]):
                ans = {r["qid"]: r for r in rec.records if r.get("kind") == "answer" and r.get("final")}
                ev = G.pilot_eval(ans, fz["pilot"]["qids"])
                rec.emit("pilot_done", unit="pilot_done", **ev)
                if ev["gates_tripped"]:
                    rec.emit("gate_tripped", unit="gate_tripped:pilot", gates=ev["gates_tripped"], where="pilot", pilot=ev)
                    rec.emit("stop", reason="phase1_gate_tripped", gates=ev["gates_tripped"], where="pilot", final=False)
                    res = {"complete": True, "phase": "gpu", "futility_stop": "+".join(ev["gates_tripped"]),
                           "gate_tripped": ev["gates_tripped"], "verdict": "phase1_gate_tripped", "where": "pilot",
                           "pilot": ev, "planner_version": planner_version, "rc": X.RC_GATE}
                    return res
                if ev["budget_stops"]:
                    rec.emit("stop", reason="budget_stop:" + "+".join(ev["budget_stops"]), final=False, pilot=ev)
                    return {"complete": False, "budget_stop": "+".join(ev["budget_stops"]), "rc": X.RC_BUDGET, **ev}
    finally:
        store.close()
    return finish(rec, inp)


def finish(rec: P.Recorder, inp: dict[str, Any]) -> dict[str, Any]:
    fz = inp["frozen"]
    ids = [r["qid"] for r in G.affected(fz["miss"])]
    ans = {r["qid"]: r for r in rec.records if r.get("kind") == "answer" and r.get("final")}
    v = G.verdict110(ans, fz["ceiling"], ids)
    pilot_ev = next((r for r in rec.records if r.get("unit") == "pilot_done"), None)
    res = {"complete": v["complete"], "phase": "gpu", "verdict": v["verdict"], "result": v, "pilot": pilot_ev,
           "gate_tripped": v["gates_tripped"], "futility_stop": "+".join(v["gates_tripped"]) or None,
           "rc": X.RC_GATE if v["gates_tripped"] else (X.RC_OK if v["complete"] else X.RC_PARTIAL)}
    if v["complete"]:
        rec.emit("verdict", unit="verdict", **v)
    return res


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    X.configure(a.corpus_id)
    try:
        inp = load(a.set_dir)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load inputs: {type(e).__name__}: {e}"]}))
        return X.RC_PREFLIGHT
    if a.dry_run:
        rc, out = dry_run(a, inp)
        print(json.dumps(out, indent=1, default=str))
        return rc
    if a.out is None:
        build_parser().error("--out is required to run (use --dry-run to only validate)")
    problems, info = checks(a, inp)
    print(json.dumps({"problems": problems, **info}, indent=1, default=str))
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    if problems or a.phase == "preflight":
        rec.emit("preflight", unit=None, ok=not problems, final=False, problems=problems, **info)
        X._atomic_write(a.out / "result.preflight.json", json.dumps(
            {"phase": "preflight", "complete": not problems, "problems": problems,
             "predeclared_sha256": G.PREDECLARED_SHA}, indent=1, default=str) + "\n")
        return X.RC_PREFLIGHT if problems else 0
    planner, _ = orch_import(a)
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=G.PREDECLARED_SHA, orch_pin=a.orch_pin,
             research_pin=a.research_pin, corpus_id=a.corpus_id, url=a.url, model_id=a.model_id, cfg=dict(vars(P.CFG)),
             planner_version=a.planner_version, sql_n_predict=G.SQL_N_PREDICT[a.planner_version], final=False)
    try:
        if a.phase == "score":
            res = finish(rec, inp)
        else:
            if a.store_src is None or not a.store_src.exists():
                rec.emit("precondition_failed", reason=f"no --store-src {a.store_src}")
                return X.RC_PRECOND
            from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
            backend = LlamaServerBackend(a.url, model_id=a.model_id)
            res = run_answers(rec, a.out, inp, planner, backend, store_src=a.store_src, guard=guard, smoke=smoke,
                              corpus_id=a.corpus_id, planner_version=a.planner_version)
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return X.RC_PARTIAL
    rc = res.pop("rc", None)
    res["predeclared_sha256"] = G.PREDECLARED_SHA
    if not smoke and res.get("complete") is not None and (res.get("complete") or res.get("verdict")):
        X._atomic_write(a.out / "result.gpu.json", json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    if rc is not None:
        return rc
    return 0 if (res.get("complete") or smoke) else X.RC_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
