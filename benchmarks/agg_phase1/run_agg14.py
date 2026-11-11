#!/usr/bin/env python3
"""AGG-1.4 runner CLI (keeper jobs run it under normal grants; logic in ``agg14.py``). Code only: no model, server or
process is started here.

``--phase``:

  preflight   zero-compute checks (pins, gold cells, L4 reuse test, the backfill set recomputed from the base store)
  backfill    CPU: planner receipt, pilot on one document, the other missing documents, G1-G4, result.cpu.json

``--dry-run`` validates every input, prints the plan (the 12 units, order, receipt, gates) and exits: zero model calls,
no server, no store, nothing written. The base store is read ``immutable=1`` and never opened through ``CorpusStore``.
Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped, 4 precondition, 5 budget stop, 75 partial.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
import agg14 as G  # noqa: E402
import phase1_arms as X  # noqa: E402
from phase1_arms import A, P  # noqa: E402

PHASES = ("preflight", "backfill")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, default=None)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--agg11-job", type=Path, default=G.DEFAULT_AGG11_JOB)
    ap.add_argument("--agg11-results", type=Path, default=None)
    ap.add_argument("--base-store", type=Path, default=None)
    ap.add_argument("--pins", choices=("agg14", "none"), default="agg14")
    ap.add_argument("--corpus-id", default=P.CORPUS_ID)
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def read_only_missing(base: Path, corpus_id: str) -> list[str]:
    """The backfill set from the base store WITHOUT opening it through ``CorpusStore`` (read-only, immutable=1):
    documents with no canonical (field, FY2022) row after L3 + L4, via the pure ``select_canonical``."""
    from src.aggregation.store import FACT_COLUMNS, select_canonical  # noqa: PLC0415
    c = sqlite3.connect(f"file:{base}?immutable=1", uri=True)
    try:
        rows = [dict(zip(("corpus_id",) + FACT_COLUMNS, r)) for r in c.execute(
            f"SELECT d.corpus_id, {', '.join('f.' + x for x in FACT_COLUMNS)} FROM facts f JOIN documents d "
            "ON f.doc_hash = d.doc_hash WHERE d.corpus_id = ?", [corpus_id]).fetchall()]
        docs = [r[0] for r in c.execute("SELECT doc_id FROM documents WHERE corpus_id = ? ORDER BY doc_id", [corpus_id])]
    finally:
        c.close()
    have = {r["doc_id"] for r in select_canonical(rows)[0] if r["field"] == G.FIELD and r["period"] == G.PERIOD}
    return [d for d in docs if d not in have]


def checks(a: argparse.Namespace, inp: dict[str, Any], *, deep: bool) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    info: dict[str, Any] = {}
    if a.pins == "agg14":
        problems += G.pin_problems(a.set_dir)
    problems += G.cells_problems(inp, expect=a.pins == "agg14")
    if a.pins == "agg14":
        inp11 = X.load_inputs(corpus=a.set_dir / "corpus.json", golds=a.set_dir / "golds.jsonl",
                              verification=a.set_dir / "verification_record.json",
                              facts_sheet=a.set_dir / "facts_sheet.csv")
        ok11, info11 = X.preflight(inp11, a.set_dir, check_pdfs=False, run_build_check=True,
                                   sizes=X.load_sizes(a.set_dir / "extracted_sizes.json"))
        problems += [f"AGG-1.1 inputs: {p}" for p in info11["problems"]]
    info["gold"] = {"filled": len(G.gold_cells(inp)), "na": G.na_docs(inp)}
    info["field"] = {"name": G.FIELD, "period": G.PERIOD, "question": G.QUESTION, "pilot_doc": G.PILOT_DOC,
                     "expected_missing": list(G.EXPECTED_MISSING)}
    if a.orch_root is not None:
        f = a.orch_root / ".pin"
        if a.orch_pin and f.exists() and f.read_text().strip() != a.orch_pin:
            problems.append(f"{f} = {f.read_text().strip()} != expected orchestrator pin {a.orch_pin}")
        if a.agg11_job.exists():
            import agg13 as G13  # noqa: PLC0415  (the shared AGG-1.1 reuse rule)
            info["l4_reuse"] = G13.l4_reuse_check13(a.agg11_job, a.orch_root)
            if not info["l4_reuse"].get("reuse_ok"):
                problems.append("L4 reuse test failed: the earlier units are not cache hits")
    if deep and a.orch_root is not None:
        sys.path.insert(0, str(a.orch_root))
        os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
        try:
            from src.aggregation import planner  # noqa: PLC0415
            assert hasattr(planner, "backfill_plan") and hasattr(planner, "question_fields")
        except Exception as e:  # noqa: BLE001
            problems.append(f"orchestrator tree {a.orch_root} lacks the planner backfill trigger: {type(e).__name__}: {e}")
        else:
            if a.base_store is None or not a.base_store.exists():
                problems.append(f"base store missing: {a.base_store}")
            else:
                missing = read_only_missing(a.base_store, a.corpus_id)
                info["backfill_set"] = {"n": len(missing), "docs": missing,
                                        "expected_equal": tuple(missing) == tuple(sorted(G.EXPECTED_MISSING))}
                if a.pins == "agg14" and not info["backfill_set"]["expected_equal"]:
                    problems.append(f"backfill set {missing} != PREDECLARED {sorted(G.EXPECTED_MISSING)}")
    return problems, info


def dry_run(a: argparse.Namespace, inp: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    problems, info = checks(a, inp, deep=True)
    order = [G.PILOT_DOC] + sorted(d for d in G.EXPECTED_MISSING if d != G.PILOT_DOC)
    pd = a.set_dir / "PREDECLARED_AGG14.md"
    out = {"dry_run": True, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
           "predeclared_sha256": G.SHA256(pd) if pd.exists() else None, "problems": problems, "checks": info,
           "plan": {"field": G.FIELD, "units": len(order), "order": order, "pilot": {"doc": G.PILOT_DOC, "bound_s": G.PILOT_WALL_S},
                    "l4_units_seeded_as_cache_hits": len(A.METRICS) * 28,
                    "expected_calls": round(len(order) * 6.08, 1), "expected_wall_s": round(len(order) * 54.5, 1),
                    "gates": {"G1": f"fresh calls <= {G.CALL_GATE}", "G2": f"field M1 >= {G.M1_GATE} over "
                              f"{len(G.gold_cells(inp))} filled cells", "G3": "earlier L4 units untouched (snapshot sha256, "
                              "0 reruns, 0 calls)", "G4": "planner receipt: sql path, no model, backfill trigger, field, 12 docs"},
                    "stops": {"futility": "pilot calls x units > 100 or running calls > 100",
                              "budget": "pilot wall > 600 s or projected wall > 2x booked 1000 s"}}}
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
        a.base_store = a.agg11_job / "out" / "store.sqlite"
    if a.dry_run:
        rc, out = dry_run(a, inp)
        print(json.dumps(out, indent=1, default=str))
        return rc
    if a.out is None:
        build_parser().error("--out is required to run (use --dry-run to only validate)")
    if a.phase != "preflight" and a.orch_root is None:
        build_parser().error("--orch-root (the pinned orchestrator tree) is required to run this phase")
    problems, info = checks(a, inp, deep=True)
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
             field=G.FIELD, final=False)
    try:
        from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
        backend = LlamaServerBackend(a.url, model_id=a.model_id)
        res = G.run_backfill(rec, a.out, inp, a.orch_root, backend, base_store=a.base_store,
                             agg11_results=a.agg11_results or a.agg11_job / "out" / "results.jsonl", guard=guard,
                             smoke=smoke, corpus_id=a.corpus_id, reuse_ok=bool(info["l4_reuse"]["reuse_ok"]))
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return X.RC_PARTIAL
    rc = res.pop("rc", None)
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = G.PREDECLARED_SHA
    print(json.dumps(res, indent=1, default=str))
    if rc is not None:
        return rc
    return 0 if (res.get("complete") or smoke) else X.RC_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
