#!/usr/bin/env python3
"""AGG-1.9 runner CLI (keeper jobs run it under normal grants; logic in ``agg19.py``). Code only: no model, server or process
is started here.

``--phase``:

  probe       zero model: the canonical guard variants and the 14 saved store-gap SQLs on a store (writes --probe-out)
  preflight   zero-compute checks (pins, the 26 cells, L4 reuse test, the probe recomputed and compared with the frozen one)
  fill        CPU: pilot 5 cells, the other 21, M1 before/after, canonical rule, the 14 SQL re-runs, regression gate

``--dry-run`` validates every input and prints the plan and exits: zero model calls, no server, no store, nothing written.
The base store is read ``immutable=1``. Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 gate tripped,
4 precondition, 5 budget stop, 75 partial.
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
import agg19 as G  # noqa: E402
import phase1_arms as X  # noqa: E402
from phase1_arms import A, P  # noqa: E402

PHASES = ("probe", "preflight", "fill")
DEFAULT_AGG11_JOB = Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111")
DEFAULT_BASE = Path("/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out/store.sqlite")
DEFAULT_ANSWERS = Path("/mnt/raid0/llm/tmp/agg1-arms-gpu-20261111/out/results.jsonl")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, default=None)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--agg11-job", type=Path, default=DEFAULT_AGG11_JOB)
    ap.add_argument("--agg11-results", type=Path, default=None, help="AGG-1.1 CPU results.jsonl (the L4 units to seed)")
    ap.add_argument("--agg11-answers", type=Path, default=DEFAULT_ANSWERS, help="AGG-1.1 GPU results.jsonl (saved SQL)")
    ap.add_argument("--base-store", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--probe-out", type=Path, default=None)
    ap.add_argument("--pins", choices=("agg19", "none"), default="agg19")
    ap.add_argument("--corpus-id", default=P.CORPUS_ID)
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def _import_orch(orch_root: Path) -> None:
    sys.path.insert(0, str(orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")


def checks(a: argparse.Namespace, inp: dict[str, Any], *, deep: bool) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    info: dict[str, Any] = {}
    if a.pins == "agg19":
        problems += G.pin_problems(a.set_dir)
        inp11 = X.load_inputs(corpus=a.set_dir / "corpus.json", golds=a.set_dir / "golds.jsonl",
                              verification=a.set_dir / "verification_record.json", facts_sheet=a.set_dir / "facts_sheet.csv")
        ok11, info11 = X.preflight(inp11, a.set_dir, check_pdfs=False, run_build_check=True,
                                   sizes=X.load_sizes(a.set_dir / "extracted_sizes.json"))
        problems += [f"AGG-1.1 inputs: {p}" for p in info11["problems"]]
    problems += G.cells_problems(inp, expect=a.pins == "agg19")
    info["cells"] = {"n": len(inp["cells"]), "pilot": [G.unit_of(c) for c in G.pilot_cells(inp)],
                     "units": [G.unit_of(c) for c in inp["cells"]], "n_verified_cells": len(inp["cells211"]),
                     "store_gap_questions": inp["store_gap_qids"]}
    if a.orch_root is not None:
        f = a.orch_root / ".pin"
        if a.orch_pin and f.exists() and f.read_text().strip() != a.orch_pin:
            problems.append(f"{f} = {f.read_text().strip()} != expected orchestrator pin {a.orch_pin}")
        if a.agg11_job.exists():
            info["l4_reuse"] = G.l4_reuse_check19(a.agg11_job, a.orch_root)
            if not info["l4_reuse"].get("reuse_ok"):
                problems.append("L4 reuse test failed: the earlier units are not cache hits")
    if deep and a.orch_root is not None:
        _import_orch(a.orch_root)
        try:
            from src.aggregation import store as S  # noqa: PLC0415
            assert hasattr(S, "CANONICAL_RULE_V2") and hasattr(S, "guard_attributable_net_income")
        except Exception as e:  # noqa: BLE001
            problems.append(f"orchestrator tree {a.orch_root} lacks canonical.v2: {type(e).__name__}: {e}")
        else:
            if not a.base_store.exists():
                problems.append(f"base store missing: {a.base_store}")
            else:
                live = G.probe_base(a.base_store, inp, a.agg11_answers, a.corpus_id)
                info["probe"] = {"adopted_guards": live["adopted_guards"], "raw_hits": live["raw_hits"],
                                 "canonical_ok_v1": live["control"]["n_ok"],
                                 "variants": {k: {"n_ok": v["n_ok"], "fixed": len(v["fixed"]), "broken": len(v["broken"])}
                                              for k, v in live["variants"].items()},
                                 "saved_sql_correct_before": live.get("saved_sql_on_agg11_store", {}).get("n_correct")}
                if a.pins == "agg19":
                    frozen = json.loads((a.set_dir / G.PROBE_FILE).read_text())
                    if json.loads(json.dumps(live, default=str)) != frozen:
                        problems.append("probe recomputed from the base store != frozen agg19_guard_probe.json")
                    if live["raw_hits"]["found"] != G.EXPECTED_BEFORE["raw_hits"] or \
                            live["control"]["n_ok"] != G.EXPECTED_BEFORE["canonical_ok_v1"]:
                        problems.append(f"before counts {live['raw_hits']['found']}/{live['control']['n_ok']} != "
                                        f"{G.EXPECTED_BEFORE}")
    return problems, info


def dry_run(a: argparse.Namespace, inp: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    problems, info = checks(a, inp, deep=True)
    n = len(inp["cells"])
    pd = a.set_dir / "PREDECLARED_AGG19.md"
    out = {"dry_run": True, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0,
           "predeclared_sha256": G.SHA256(pd) if pd.exists() else None, "problems": problems, "checks": info,
           "plan": {"units": n, "pilot": info["cells"]["pilot"], "pilot_bound_s": G.PILOT_WALL_S,
                    "l4_units_seeded_as_cache_hits": len(A.METRICS) * 28 - n,
                    "expected_calls": round(n * 6.08, 1), "expected_wall_s": round(n * 54.5, 1),
                    "booked_s_for_budget_stop_ESTIMATE": G.BOOKED_S,
                    "gates": {"futility": "pilot fills < 2 of 5", "retained": "0 retained cells regress (canonical and raw)",
                              "done": "M1 over 211 before vs after, the 14 saved store-gap SQLs re-run"},
                    "stops": {"budget": "pilot wall > 300 s or projected wall > 2x 1400 s (amendment 1)"}}}
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
    if a.phase == "probe":
        if a.orch_root is None:
            build_parser().error("--orch-root is required for --phase probe")
        _import_orch(a.orch_root)
        res = G.probe_base(a.base_store, inp, a.agg11_answers, a.corpus_id)
        text = json.dumps(res, indent=1, sort_keys=True, default=str) + "\n"
        if a.probe_out:
            a.probe_out.write_text(text)
        else:
            print(text)
        return 0
    if a.agg11_results is None:
        a.agg11_results = a.agg11_job / "out" / "results.jsonl"
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
    _import_orch(a.orch_root)
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=G.PREDECLARED_SHA, orch_pin=a.orch_pin,
             research_pin=a.research_pin, corpus_id=a.corpus_id, url=a.url, model_id=a.model_id, cfg=dict(vars(P.CFG)),
             adopted_guards=list(G.adopted_guards(json.loads((a.set_dir / G.PROBE_FILE).read_text()))), final=False)
    try:
        from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
        backend = LlamaServerBackend(a.url, model_id=a.model_id)
        probe = json.loads((a.set_dir / G.PROBE_FILE).read_text())
        res = G.run_fill(rec, a.out, inp, a.orch_root, backend, base_store=a.base_store, agg11_results=a.agg11_results,
                         agg11_answers=a.agg11_answers, guard=guard, probe=probe, smoke=smoke, corpus_id=a.corpus_id,
                         reuse_ok=bool(info["l4_reuse"]["reuse_ok"]))
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
