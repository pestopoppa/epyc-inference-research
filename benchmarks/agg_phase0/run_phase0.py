#!/usr/bin/env python3
"""AGG-0.4 Phase-0 runner (keeper jobs run it under normal grants; arms in phase0_arms.py).

* ``--preflight``: zero-compute checks — corpus PDFs present with pinned sha256,
  every question to run is verified in ``verification_record.json``, golds
  complete, PREDECLARED.md sha256 captured. Exit 2 on any failure.
* ``--phase cpu``          M5 manifest (futility 1), lazy L4 on worker_general, M1 (futility 2).
* ``--phase gpu-audit``    L4c 27B audit sample (np4 server).
* ``--phase gpu-answers``  P-small, B-today, P-large, B1, B2 on the 27B np1 196k server, then the
  verdict via ``agg_phase0.score`` (M1-M5 + kill gate) over the CPU job's records plus its own.
* ``--phase score``        verdict only (no model).

Exit codes: 0 phase complete (or smoke ok), 3 futility stop, 75 partial (deadline / stopfile),
2 preflight failure, 4 precondition (missing input store / CPU result).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agg_phase0 as A  # noqa: E402

ARMS = ("P-large", "P-small", "B1", "B2", "B-today")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_set(d: Path) -> dict[str, Any]:
    golds = [json.loads(l) for l in (d / "golds.jsonl").read_text().splitlines() if l.strip()]
    single = [json.loads(l) for l in (d / "single_fact.jsonl").read_text().splitlines() if l.strip()]
    ver = json.loads((d / "verification_record.json").read_text())["questions"]
    return {"golds": golds, "single": single, "verified": ver,
            "corpus": json.loads((d / "corpus.json").read_text())}


def preflight(d: Path) -> tuple[bool, dict[str, Any]]:
    s = load_set(d)
    problems: list[str] = []
    for doc in s["corpus"]["docs"]:
        p = Path(doc["pdf"])
        if doc["sha256"] is None:
            problems.append(f"unpinned/missing pdf {doc['doc_name']}")
        elif not p.exists() or A.sha256_file(p) != doc["sha256"]:
            problems.append(f"pdf sha256 mismatch {doc['doc_name']}")
    run_agg = [g for g in s["golds"] if s["verified"].get(g["id"], {}).get("verified")]
    run_single = [q for q in s["single"] if s["verified"].get(q["id"], {}).get("verified")]
    for g in run_agg:
        if g["gold"] is None:
            problems.append(f"{g['id']} verified but gold incomplete")
    if not run_agg:
        problems.append("no verified aggregation questions")
    if not run_single:
        problems.append("no verified single-fact questions")
    info = {"predeclared_sha256": sha(d / "PREDECLARED.md"),
            "n_agg_verified": len(run_agg), "n_agg_excluded": len(s["golds"]) - len(run_agg),
            "n_single_verified": len(run_single),
            "n_single_excluded": len(s["single"]) - len(run_single), "problems": problems}
    return not problems, info


def _verified(s):
    v = lambda qid: s["verified"].get(qid, {}).get("verified")  # noqa: E731
    return [g for g in s["golds"] if v(g["id"])], [q for q in s["single"] if v(q["id"])]


def verdict(cpu_results: Path, own: "P.Recorder", store) -> dict[str, Any]:
    recs = [json.loads(l) for l in cpu_results.read_text().splitlines() if l.strip()]
    recs += own.records
    keep = [r for r in recs if r["kind"] in ("ingest", "m1", "answer")]
    # M5 manifest: the run-time P-small manifests (the CPU job's plan-time ones are the futility-1 check)
    keep += [r for r in recs if r["kind"] == "manifest" and r.get("unit", "").startswith("manifest_run:")]
    return A.score(keep)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None, help="results dir (required to run)")
    ap.add_argument("--preflight", action="store_true", help="checks only, zero compute")
    ap.add_argument("--phase", choices=("cpu", "gpu-audit", "gpu-answers", "score"))
    ap.add_argument("--store", type=Path, help="this job's store.sqlite (copied once from --store-src)")
    ap.add_argument("--store-src", type=Path, help="input store to copy on first run (read-only)")
    ap.add_argument("--cpu-results", type=Path, help="the CPU job's results.jsonl (gpu phases)")
    ap.add_argument("--url", help="private llama-server base URL")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--orch-root", type=Path, default=None,
                    help="pinned orchestrator tree (src/aggregation); prepended to sys.path")
    a = ap.parse_args(argv)
    ok, info = preflight(a.set_dir)
    print(json.dumps(info, indent=1))
    if a.preflight or not ok:
        return 0 if ok else 2
    if a.out is None or a.phase is None:
        ap.error("--out and --phase are required to run")
    if a.orch_root:
        sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    import phase0_arms as P  # noqa: PLC0415
    s = load_set(a.set_dir)
    agg, single = _verified(s)
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec = P.Recorder(a.out / "results.jsonl")
    rec.emit("meta", phase=a.phase, smoke=smoke, preflight=info, url=a.url, model_id=a.model_id,
             cfg=dict(vars(P.CFG)), final=False)
    if a.store and not a.store.exists():
        if not a.store_src or not a.store_src.exists():
            rec.emit("precondition_failed", reason=f"no store at {a.store} and no --store-src")
            return 4
        P.copy_store(a.store_src, a.store)
        rec.emit("store_copied", src=str(a.store_src), sha256=A.sha256_file(a.store), final=False)
    store = P.open_store(a.store) if a.store else None
    if a.phase in ("gpu-audit", "gpu-answers", "score") and (not a.cpu_results or not a.cpu_results.exists()):
        rec.emit("precondition_failed", reason="--cpu-results missing")
        return 4
    backend = None
    if a.phase != "score":
        from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
        backend = LlamaServerBackend(a.url, model_id=a.model_id)
    names = sorted({d["company"] for d in s["corpus"]["docs"]})
    keys = json.loads((a.set_dir / "single_fact_keys.json").read_text())["keys"]
    try:
        if a.phase == "cpu":
            res = P.run_cpu(rec, store, a.set_dir, backend, agg=agg, single=single, guard=guard, smoke=smoke)
        elif a.phase == "gpu-audit":
            res = P.run_audit(rec, store, backend, guard=guard, smoke=smoke)
        elif a.phase == "gpu-answers":
            res = P.run_answers(rec, store, backend, agg=agg, single=single, keys=keys, names=names,
                                guard=guard, smoke=smoke)
            if res.get("complete"):
                res["verdict"] = verdict(a.cpu_results, rec, store)
                rec.emit("verdict", unit="verdict", **res["verdict"])
        else:
            res = {"complete": True, "verdict": verdict(a.cpu_results, rec, store)}
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return 75
    res.update(phase=a.phase, predeclared_sha256=info["predeclared_sha256"])
    if not smoke:
        (a.out / f"result.{a.phase}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    if res.get("futility_stop"):
        return 3
    return 0 if (res.get("complete") or smoke) else 75


if __name__ == "__main__":
    sys.exit(main())
