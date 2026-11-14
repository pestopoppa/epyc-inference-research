#!/usr/bin/env python3
"""AGG-1.13 runner CLI (keeper jobs run it under normal grants; logic in ``agg113.py``). Code only: no model, server or process is started here.

``--phase`` (PREDECLARED_AGG113.md checkpoints in parentheses):

  preflight     (0)   zero-compute checks: pins and sha256 of every frozen input, the 10 PDFs, ``build_agg113.py check``, the orchestrator pin and the
                      pipeline stages, and the HARD fiscal-year contract of the January-end filers (L3 period logic and ``m1_hits`` must see
                      fiscal 2022). Any problem exits 2 and nothing runs; a contract failure needs a dated amendment BEFORE any data.
  l3            (1)   the shared L0-L3 seed: parse + store + L3 of the 10 documents, NO model (a fresh store; copied for each arm, never mutated)
  ingest        (2-4) ONE arm, ``--pipeline v1|v2``: the 3 pilot documents (WALMART, ULTABEAUTY_2023, BLOCK), then the other 7; futility-only looks
  gpu-audit     (5a)  the L4c audit sample on the v1 store, then the v2 store (the AGG-1.1 recipe, unmodified)
  gpu-answers   (5b)  P-large over the 30 questions for arm A (v1 + agg11.v0), B (v2 + agg11.v0), C (v2 + agg110.v1), in that order
  gpu                 gpu-audit then gpu-answers (``--url`` for the audit server, ``--answers-url`` for the np1 server)
  score               verdict only (no model): paired M1 and P-large, strata, M3, M4, regressions and fixes

``--dry-run`` validates every input, prints the plan (estimates labelled ESTIMATE) and exits: zero model calls, no server, no store copy, nothing
written. Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped (agg113_futility / phase1_gate_tripped), 4 precondition,
5 budget stop (re-book, not a gate), 6 shared L3 load failed (>= 2 of 10), 75 partial (resumable).
Pilot and interim output is COUNTS ONLY (PREDECLARED rule 1).
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
import agg113 as H  # noqa: E402
import agg112 as G  # noqa: E402
import agg16 as G16  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402

PHASES = ("preflight", "l3", "ingest", "gpu-audit", "gpu-answers", "gpu", "score")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; no model, no writes")
    ap.add_argument("--set-dir", type=Path, default=H.D113)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, default=None)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--pipeline", choices=(H.V1, H.V2), default=None, help="ingest: the arm's pipeline version")
    ap.add_argument("--stage", choices=("all", "pilot"), default="all", help="ingest: stop after the pilot")
    ap.add_argument("--seed-store", type=Path, default=None, help="ingest: the l3 row's store.sqlite (read-only seed; copied once)")
    ap.add_argument("--peer-out", type=Path, default=None, help="ingest v2: the v1 arm's out dir (futility looks read its checkpoint files)")
    ap.add_argument("--v1-dir", type=Path, default=None, help="gpu/score: the v1 ingest row's out dir (store.sqlite, result.cpu.json, cells.json)")
    ap.add_argument("--v2-dir", type=Path, default=None, help="gpu/score: the v2 ingest row's out dir")
    ap.add_argument("--gpu-dir", type=Path, default=None, help="score: the GPU row's out dir (results.jsonl); default --out")
    ap.add_argument("--booked-l4-s", type=float, default=None)
    ap.add_argument("--url")
    ap.add_argument("--answers-url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def orch_import(a: argparse.Namespace) -> list[str]:
    if a.orch_root is None:
        return ["--orch-root is required"]
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    return H.orch_checks(a.orch_root, a.orch_pin)


def load_arm(d: Path | None, pipeline: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    if d is None or not (d / "result.cpu.json").exists():
        return None, None, "no result.cpu.json yet"
    res = json.loads((d / "result.cpu.json").read_text())
    ok, why = H.cpu_ready(res, pipeline)
    cells = json.loads((d / "cells.json").read_text()) if (d / "cells.json").exists() else None
    return (res if ok else None), cells, why


def checks(a: argparse.Namespace, inp: dict[str, Any], *, deep: bool) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    info: dict[str, Any] = {}
    problems += H.pin_problems(a.set_dir)
    problems += H.corpus_problems(inp, check_pdfs=deep or a.phase in ("preflight", "l3"))
    if deep:
        bc = H.build_check(a.set_dir)
        info["build_check"] = bc
        if bc["rc"] != 0:
            problems.append("build_agg113.py check exited non-zero")
    problems += orch_import(a)
    corpus = inp["corpus"]
    info["plan"] = {"documents": len(corpus["docs"]), "cells_filled": len(inp["cells"]), "questions": len(X.verified_agg(inp)),
                    "pilot_docs": list(H.PILOT_DOCS), "pilot_cells": sum(1 for c in inp["cells"] if c[0] in H.PILOT_DOCS),
                    "order": H.doc_order(corpus), "interim_after_docs": H.INTERIM_DOCS,
                    "arms": {k: {"ingest": v[0], "planner": v[1]} for k, v in H.ARMS.items()}, "estimate": H.estimates(),
                    "gates": {"pilot_futility": ["projected M3 over a kill bound (each arm)", "v2 raw M1 strictly below v1's on the SAME pilot cells"],
                              "interim": f"after {H.INTERIM_DOCS} of 10 documents: the same two tests, futility only (never an early GO)",
                              "final_cpu": "M3 inside the Phase 0 kill bounds (prefill <= 2.0x, calls <= 150 per 100 chunks); M1 has no final gate",
                              "gpu": "M4 median calls <= 3 and median prefill <= 2510 for each of A, B, C; P-large accuracy is not a gate"}}
    if not problems:
        try:
            fc = H.fy_contract(corpus, inp["cells"])
            info["fy_contract"] = {"ok": fc["ok"], "january_end_docs": fc["january_end_docs"], "n_cells_checked": fc["n_cells_checked"]}
            problems += [f"fiscal-year contract: {p}" for p in fc["problems"]]
        except Exception as e:  # noqa: BLE001
            problems.append(f"fiscal-year contract check could not run: {type(e).__name__}: {e}")
    if a.phase == "ingest":
        if a.pipeline is None:
            problems.append("--pipeline v1|v2 is required for ingest")
        if a.seed_store is None or not a.seed_store.exists():
            info["seed_store"] = "missing (the l3 row has not produced it yet): informational in a dry run"
            if not a.dry_run:
                problems.append(f"seed store missing: {a.seed_store}")
        elif deep and not problems:
            docs = [d["doc_name"] for d in corpus["docs"]]
            store = G16.mem_store(a.seed_store, docs)
            try:
                have = {d["doc_id"] for d in store.corpus_docs(H.CORPUS_ID)}
                failed = sorted(set(docs) - have)
                info["seed"] = {"documents": len(have), "chunks": len(store.chunks(H.CORPUS_ID)), "doc_fail": failed}
                if len(failed) >= H.L3_FUTILITY_FAILS:
                    problems.append(f"seed lacks {failed}")
                keep = tuple(d for d in (H.PILOT_DOCS[0], H.PILOT_DOCS[2]) if d in have)
                problems += G.parity_problems(store, corpus, keep)
                plans = H.doc_plans_v2(store, corpus, set(failed))
                stats = X.doc_stats(store, H.CORPUS_ID)
                plan = G.plan_m3(plans, stats)
                info["plan"]["v2_window_plan"] = {"per_doc": {d: {"windows_called": v["n_windows"], "of": v["n_doc_windows"], "chars": v["chars"]}
                                                              for d, v in plans.items()},
                                                  "m3_projection": plan, "estimate": H.estimates(plans)}
                if not plan["inside_kill_bounds"]:
                    problems.append("the offline plan projects M3 over a kill bound (v2)")
            finally:
                store.close()
    if a.phase in ("gpu-audit", "gpu-answers", "gpu", "score"):
        st = {}
        for tag, d in ((H.V1, a.v1_dir), (H.V2, a.v2_dir)):
            res, _cells, why = load_arm(d, tag)
            st[tag] = {"ready": res is not None, "detail": why}
        info["precondition_cpu"] = {**st, "note": "informational in a dry run: the CPU rows have not run yet"}
        info["gpu_plan"] = {"audit": "L4c sample computed at run time from each store (confidence <= medium plus a 5% hash sample)",
                            "p_large": {"n": len(X.verified_agg(inp)), "arms": list(H.ARMS), "calls_per_question": "2 (SQL + answer), 3 with one agg110.v1 re-ask"},
                            "stores": "v1 for A; v2 for B and C (copied once from the CPU rows)"}
    return problems, info


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    X.configure(H.CORPUS_ID)
    try:
        inp = H.load_inputs(a.set_dir)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load inputs: {type(e).__name__}: {e}"]}))
        return H.RC_PREFLIGHT
    if a.dry_run or a.phase == "preflight":
        problems, info = checks(a, inp, deep=True)
        out = {"dry_run": a.dry_run, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
               "predeclared_sha256": H.SHA256(a.set_dir / H.PREDECLARED_FILE) if (a.set_dir / H.PREDECLARED_FILE).exists() else None,
               "problems": problems, "ok": not problems, **info}
        print(json.dumps(out, indent=1, default=str))
        if a.phase == "preflight" and not a.dry_run:
            if a.out is None:
                build_parser().error("--out is required to run (use --dry-run to only validate)")
            a.out.mkdir(parents=True, exist_ok=True)
            rec = P.Recorder(a.out / "results.jsonl")
            rec.emit("preflight", unit=None, ok=not problems, final=False, problems=problems)
            X._atomic_write(a.out / "result.preflight.json", json.dumps(
                {"phase": "preflight", "complete": not problems, "problems": problems, "predeclared_sha256": H.PREDECLARED_SHA},
                indent=1, default=str) + "\n")
        return H.RC_PREFLIGHT if problems else 0
    if a.out is None:
        build_parser().error("--out is required to run (use --dry-run to only validate)")
    problems, info = checks(a, inp, deep=False)
    print(json.dumps({"problems": problems, **{k: v for k, v in info.items() if k not in ("plan",)}}, indent=1, default=str))
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    if problems:
        rec.emit("preflight", unit=None, ok=False, final=False, problems=problems)
        return H.RC_PREFLIGHT
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec.emit("meta", phase=a.phase, pipeline=a.pipeline, smoke=smoke, predeclared_sha256=H.PREDECLARED_SHA, orch_pin=a.orch_pin,
             research_pin=a.research_pin, corpus_id=H.CORPUS_ID, url=a.url, model_id=a.model_id, cfg=dict(vars(P.CFG)), final=False)
    try:
        if a.phase == "l3":
            res = H.run_l3(rec, a.out, inp, orch_root=a.orch_root, guard=guard, smoke=smoke, orch_pin=a.orch_pin, research_pin=a.research_pin)
        elif a.phase == "ingest":
            if not a.url:
                build_parser().error("--url is required for this phase")
            from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
            backend = LlamaServerBackend(a.url, model_id=a.model_id)
            res = H.run_ingest(rec, a.out, inp, backend, pipeline=a.pipeline, guard=guard, seed_store=a.seed_store, stage=a.stage, smoke=smoke,
                               peer_out=a.peer_out, booked_l4_s=a.booked_l4_s)
        else:  # gpu-audit | gpu-answers | gpu | score
            for t, d in ((H.V1, a.v1_dir), (H.V2, a.v2_dir)):
                r, _c, why = load_arm(d, t)
                missing_files = d is None or not (d / "result.cpu.json").exists() or not (d / "cells.json").exists()
                if missing_files or (a.phase != "score" and r is None):
                    rec.emit("precondition_failed", reason=f"{t} ingest not ready: {why}" if not missing_files else f"{t} ingest files missing")
                    return H.RC_PRECOND
            stores = {}
            if a.phase != "score":
                for t, d in ((H.V1, a.v1_dir), (H.V2, a.v2_dir)):
                    sp = a.out / f"store.{t}.sqlite"
                    if not sp.exists():
                        P.copy_store(d / "store.sqlite", sp)
                        rec.emit("store_copied", src=str(d / "store.sqlite"), tag=t, sha256=H.SHA256(sp), final=False)
                    stores[t] = P.open_store(sp)
            agg = X.verified_agg(inp)
            names = sorted({d["company"] for d in inp["corpus"]["docs"]})
            try:
                if a.phase == "score":
                    res = _score(a, rec, inp, agg)
                else:
                    from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
                    res = {}
                    if a.phase in ("gpu-audit", "gpu"):
                        if not a.url:
                            build_parser().error("--url is required for this phase")
                        res = H.run_audit(rec, stores, LlamaServerBackend(a.url, model_id=a.model_id), guard=guard, smoke=smoke)
                    if a.phase in ("gpu-answers", "gpu") and (a.phase == "gpu-answers" or res.get("complete")):
                        url = a.answers_url if a.phase == "gpu" else a.url
                        if not url:
                            build_parser().error("--url (or --answers-url for phase gpu) is required for this phase")
                        res = H.run_answers(rec, stores, LlamaServerBackend(url, model_id=a.model_id), agg=agg, names=names, guard=guard, smoke=smoke)
                        if res.get("complete") and not smoke:
                            sc = _score(a, rec, inp, agg)
                            res["verdict"] = sc
                            rec.emit("verdict", unit="verdict", **{k: v for k, v in sc.items() if k != "m1"})
                            res["rc"] = H.RC_GATE if sc.get("gates_tripped") else H.RC_OK
            finally:
                for s in stores.values():
                    s.close()
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return H.RC_PARTIAL
    rc = res.pop("rc", None)
    res.setdefault("phase", a.phase)
    res["predeclared_sha256"] = H.PREDECLARED_SHA
    if not smoke and a.phase in ("l3", "gpu-audit", "gpu-answers", "gpu", "score") and res.get("complete") is not None:
        X._atomic_write(a.out / f"result.{a.phase}.json", json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("summaries", "verdict")}, indent=1, default=str))
    if rc is not None:
        return rc
    return 0 if (res.get("complete") or smoke) else H.RC_PARTIAL


def _score(a: argparse.Namespace, rec: P.Recorder, inp: dict[str, Any], agg: list[dict[str, Any]]) -> dict[str, Any]:
    r1, c1 = json.loads((a.v1_dir / "result.cpu.json").read_text()), json.loads((a.v1_dir / "cells.json").read_text())
    r2, c2 = json.loads((a.v2_dir / "result.cpu.json").read_text()), json.loads((a.v2_dir / "cells.json").read_text())
    cps = {}
    for n in ("pilot", "interim6"):
        f = a.v2_dir / f"checkpoint.{n}.json"
        if f.exists():
            cps[n] = json.loads(f.read_text())
    gdir = a.gpu_dir or a.out
    records = P.Recorder(gdir / "results.jsonl").records if gdir != a.out else rec.records
    return H.score(r1, r2, c1, c2, records, agg, inp["cells"], checkpoints=cps)


if __name__ == "__main__":
    sys.exit(main())
