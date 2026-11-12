#!/usr/bin/env python3
"""AGG-1.6 runner CLI (keeper jobs run it under normal grants; logic in ``agg16.py``). Code only: no model, server or process is
started here.

``--phase``:

  plan        zero model: pins, the 38 targets, the offline window plan and the measured-rate ESTIMATE (reads the seed stores
              ``immutable=1``; writes nothing unless --plan-out is given)
  preflight   zero-compute checks (pins, targets, the amended L4 stage, seeds, saved C0 results) and the plan
  run         CPU: smoke (one C1 cell) or pilot on 5 cells, then the rest, the metrics and the verdict

``--dry-run`` validates every input, prints the plan and exits: zero model calls, no server, nothing written.
Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 futility stop (pilot), 4 precondition, 5 budget stop, 75 partial.
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
import agg16 as G  # noqa: E402
import phase1_arms as X  # noqa: E402
from phase1_arms import P  # noqa: E402

PHASES = ("plan", "preflight", "run")
DEFAULT_SEEDS = {"agg11": Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/store.sqlite"),
                 "agg13": Path("/mnt/raid0/llm/tmp/agg13-reingest-cpu-20261111-r3/out/store.sqlite")}
DEFAULT_C0 = {"c": Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/results.jsonl"),
              "a": Path("/mnt/raid0/llm/tmp/agg13-reingest-cpu-20261111-r3/out/results.jsonl"),
              "b": Path("/mnt/raid0/llm/tmp/agg14-backfill-cpu-20261111-r3/out/results.jsonl")}
BOOKED_S_DEFAULT = 4200.0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, default=None)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--seed-agg11", type=Path, default=DEFAULT_SEEDS["agg11"], help="AGG-1.1 store (read-only seed)")
    ap.add_argument("--seed-agg13", type=Path, default=DEFAULT_SEEDS["agg13"], help="AGG-1.3 r3 store (read-only seed)")
    ap.add_argument("--c0-agg11", type=Path, default=DEFAULT_C0["c"], help="saved AGG-1.1 results.jsonl (C0, groups c)")
    ap.add_argument("--c0-agg13", type=Path, default=DEFAULT_C0["a"], help="saved AGG-1.3 r3 results.jsonl (C0, group a)")
    ap.add_argument("--c0-agg14", type=Path, default=DEFAULT_C0["b"], help="saved AGG-1.4 r3 results.jsonl (C0, group b)")
    ap.add_argument("--booked-s", type=float, default=BOOKED_S_DEFAULT)
    ap.add_argument("--pins", choices=("agg16", "none"), default="agg16")
    ap.add_argument("--plan-out", type=Path, default=None)
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def seeds_of(a: argparse.Namespace) -> dict[str, Path]:
    return {"agg11": a.seed_agg11, "agg13": a.seed_agg13}


def c0_of(a: argparse.Namespace) -> dict[str, Path]:
    return {"c": a.c0_agg11, "a": a.c0_agg13, "b": a.c0_agg14}


def build_plan(a: argparse.Namespace, inp: dict[str, Any]) -> dict[str, Any]:
    """The offline window plan over read-only seeds (needs the orchestrator tree for BM25 and the store schema)."""
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    wins: dict[str, Any] = {}
    for name, path in seeds_of(a).items():
        docs = {t["doc"] for t in inp["targets"] if t["store"] == name}
        wins[name] = G.doc_windows(G.mem_store(path, docs))
    plan = G.plan_cells(inp["targets"], wins)
    plan["estimate"] = G.estimates(plan, len(inp["targets"]))
    return plan


def checks(a: argparse.Namespace, inp: dict[str, Any], *, deep: bool) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    info: dict[str, Any] = {}
    if a.pins == "agg16":
        problems += G.pin_problems(a.set_dir)
    problems += G.targets_problems(inp["targets"], expect=a.pins == "agg16")
    t = inp["targets"]
    info["targets"] = {"n": len(t), "l4_format": sum(x["l4_class"] == "FORMAT" for x in t),
                       "l4_retrieval": sum(x["l4_class"] == "RETRIEVAL" for x in t),
                       "primary_format": sum(x["primary"] == "FORMAT" for x in t),
                       "by_store": {s: sum(x["store"] == s for x in t) for s in G.STORE_OF_GROUP.values()},
                       "pilot": ["|".join(k) for k in G.PILOT_CELLS]}
    info["design"] = {"k_field": G.K_FIELD, "n_predict_c1": G.N_PREDICT_C1, "n_predict_t": G.N_PREDICT_T,
                      "t_max_rows": G.T_MAX_ROWS, "t_quote_chars": G.T_QUOTE_CHARS, "t_slots": list(G.T_SLOTS)}
    for name, p in seeds_of(a).items():
        if not p.exists():
            problems.append(f"seed store {name} missing: {p}")
    for g, p in c0_of(a).items():
        if not p.exists():
            problems.append(f"saved C0 results {g} missing: {p}")
    if a.orch_root is not None:
        f = a.orch_root / ".pin"
        if a.orch_pin and f.exists() and f.read_text().strip() != a.orch_pin:
            problems.append(f"{f} = {f.read_text().strip()} != expected orchestrator pin {a.orch_pin}")
        problems += G.stage_problems(a.orch_root)
    if deep and a.orch_root is not None and not problems:
        info["plan"] = build_plan(a, inp)
        c0 = G.c0_summary(t, G.read_results(c0_of(a)))
        info["c0"] = {k: v for k, v in c0.items() if k != "per_cell"}
        if c0["missing"]:
            problems.append(f"saved C0 units missing for {c0['missing']}")
    return problems, info


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        inp = G.load_inputs(a.set_dir)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load inputs: {type(e).__name__}: {e}"]}))
        return G.RC_PREFLIGHT
    if a.dry_run or a.phase in ("plan", "preflight"):
        if a.orch_root is None:
            build_parser().error("--orch-root is required (the plan reads BM25 and the store schema from the pinned tree)")
        problems, info = checks(a, inp, deep=True)
        out = {"dry_run": a.dry_run, "phase": a.phase, "model_calls": 0, "servers_started": 0,
               "wrote_files": 0 if not a.plan_out else 1, "predeclared_sha256": G.SHA256(a.set_dir / G.PREDECLARED_FILE),
               "problems": problems, "ok": not problems, **info}
        print(json.dumps(out, indent=1, default=str))
        if a.plan_out and not a.dry_run and "plan" in info:
            X._atomic_write(a.plan_out, json.dumps(info["plan"], indent=1, sort_keys=True, default=str) + "\n")
        if a.phase == "preflight" and not a.dry_run and a.out is not None:
            a.out.mkdir(parents=True, exist_ok=True)
            X._atomic_write(a.out / "result.preflight.json", json.dumps(
                {"phase": "preflight", "complete": not problems, "problems": problems,
                 "predeclared_sha256": G.PREDECLARED_SHA}, indent=1, default=str) + "\n")
        return G.RC_PREFLIGHT if problems else 0
    if a.out is None or a.orch_root is None or not a.url:
        build_parser().error("--out, --orch-root and --url are required to run (use --dry-run to only validate)")
    problems, info = checks(a, inp, deep=True)
    print(json.dumps({"problems": problems, **{k: v for k, v in info.items() if k != "plan"}}, indent=1, default=str))
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    if problems:
        rec.emit("preflight", unit=None, ok=False, final=False, problems=problems)
        return G.RC_PREFLIGHT
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=G.PREDECLARED_SHA, orch_pin=a.orch_pin,
             research_pin=a.research_pin, url=a.url, model_id=a.model_id, cfg=dict(vars(P.CFG)), final=False,
             design=info["design"])
    try:
        from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
        backend = LlamaServerBackend(a.url, model_id=a.model_id)
        res = G.run_ab(rec, a.out, inp, backend, seeds=seeds_of(a), c0_results=c0_of(a), guard=guard, booked_s=a.booked_s,
                       smoke=smoke)
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return G.RC_PARTIAL
    rc = res.pop("rc", None)
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = G.PREDECLARED_SHA
    print(json.dumps({k: v for k, v in res.items() if k != "summaries"}, indent=1, default=str))
    if rc is not None:
        return rc
    return 0 if (res.get("complete") or smoke) else G.RC_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
