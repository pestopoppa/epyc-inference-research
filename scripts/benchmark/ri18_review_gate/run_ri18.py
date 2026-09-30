#!/usr/bin/env python3
"""RI-18 driver: does the ``_should_review`` MemRL Q gate earn its keep?

Design + pre-registration: ``/mnt/raid0/llm/tmp/ri18/DESIGN.md`` (§5 is the rule). Operator
answers 2026-09-30: Q1 cost cap X = <=60 GPU and <=120 CPU device-seconds per net fix and <=+20%
added p50 latency on reviewed requests; Q2 workload = S1 re-render (``ri18-brief-justify-v1``)
+ S2 ``olympiadbench_hard``; Q3 = the V-full rule may land ``question_cap=1500``.

Run with the ORCHESTRATOR venv from the research repo root, ``--code-root`` at the orchestrator
checkout the in-process calls must use (the served commit)::

    PY=/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python
    $PY -m scripts.benchmark.ri18_review_gate.run_ri18 <command> [options]

Commands (each segment is resumable per item; rerun the same command to continue):

  plan           remaining work per segment + window estimates (no network)
  snapshot       freeze the MemRL store for the gate (run right BEFORE the first answer segment)
  answer         stage 1+2, --suite s1|s2: A1 via the API (CPU window)
  verdict        stage 3+5: production verdict (+ V-full on S1), GPU :8083 (no CPU window)
  revise         stage 4: production revision on WRONG, CPU :8070 in-process (CPU window)
  noise-verdict  control: verdict re-run on the seeded 50 (GPU)
  noise-revise   control: revision re-run on the first 30 WRONG (CPU window)
  gate           stage 6: offline gate against the snapshot (CPU embedders; CPU window)
  score          all metrics + the pre-registered rule -> score.json + belief sidecar (offline)

``--allow-announced-pause`` (or env ``WS8D_ALLOW_ANNOUNCED_PAUSE=1``): the CPU window gate also admits
an announced DS41 pause (``window.evaluate_pause``); each admitting mode is logged and written to
segments.jsonl as a ``window_admitted`` event. Default off = the gate is unchanged. The GPU segments
(verdict, noise-verdict) send one request at a time to :8083 (the canary pair and every item are
sequential calls), with or without the opt-in.

``--dry-run``: fake transport/primitives/retriever (the REAL orchestrator review functions when
``--code-root`` imports; ``--stub-orch`` for none), no window check unless ``--window-file`` is
given, output to a scratch dir. Exit codes: 0 done, 2 usage, 3 CPU window refused / region
claim denied (resume next window), 4 budget or --limit reached (resume), 5 VOID segment, 6
instrument fault, 7 :8083 all slots busy, 8 prerequisite missing, 9 manifest drift (refused).
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

# Shared host: the offline scorer's numpy must not fan out across every core.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.benchmark.ri18_review_gate"

from ..thesis_ufh13.arms import GENERATION  # noqa: E402
from ..thesis_ufh13.transports import V1Transport  # noqa: E402
from . import fakes, pipeline, render, score, snapshot, window  # noqa: E402
from .bridge import (  # noqa: E402
    CPU_URL,
    GPU_URL,
    REVIEWER_ROLE,
    VFULL_QUESTION_CAP,
    Bridge,
    InstrumentFault,
    git_state,
)
from .store import ManifestDrift, RunDir, StageBusy  # noqa: E402

RUNNER = "ri18-runner/v1"
DEFAULT_CODE_ROOT = "/mnt/raid0/llm/epyc-orchestrator"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SCRATCH = Path("/mnt/raid0/llm/tmp/ri18")
DEFAULT_BUDGET_S = {"answer": 5400.0, "verdict": 3600.0, "revise": 5400.0,
                    "noise-verdict": 900.0, "noise-revise": 1800.0, "gate": 1800.0}
STAGE_OF = {"answer": "answer", "verdict": "verdict", "revise": "revise",
            "noise-verdict": "noise_verdict", "noise-revise": "noise_revise", "gate": "gate"}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mode(args: argparse.Namespace) -> str:
    if args.stub_orch:
        return "stub"
    return "dry" if args.dry_run else "real"


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _served_identity(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    if mode != "real":
        return fakes.fake_served_identity()
    try:
        ident = pipeline.probe_served(args.base_url)
    except Exception as exc:  # noqa: BLE001 - any failure means "no probe"
        if not args.served_commit:
            raise SystemExit(f"cannot read {args.base_url}/dashboard/api/version ({exc}); "
                             "pass --served-commit") from exc
        return {"server_launch_git_sha": args.served_commit, "server_started_at": None,
                "git_sha": None, "probe": f"unavailable: {exc}"}
    launch = ident.get("server_launch_git_sha") or ""
    if args.served_commit and not (launch and (launch.startswith(args.served_commit)
                                               or args.served_commit.startswith(launch))):
        raise SystemExit(f"--served-commit {args.served_commit} != API launch sha {launch!r}")
    return ident


def build_pin(args: argparse.Namespace, mode: str, kind: str, run: RunDir,
              workload: dict[str, Any]) -> dict[str, Any]:
    existing = run.manifest() if run.manifest_path.exists() else None
    if existing is None and kind != "answer":
        raise pipeline.Stop(pipeline.RC_PREREQ, f"{run.out} has no run manifest: the first "
                                                "segment must be `answer` (after `snapshot`)")
    served = ({"server_launch_git_sha": _served_identity(args, mode)["server_launch_git_sha"]}
              if kind == "answer" or existing is None else existing["served"])
    if mode == "real":
        snap = snapshot.snapshot_pin(Path(args.snapshot_dir))
        code = git_state(args.code_root)
    elif mode == "dry":
        snap = {"dir": None, "files": {}, "dry_run": True}
        code = git_state(args.code_root)
    else:
        snap = {"dir": None, "files": {}, "dry_run": True}
        code = {"commit": "stub", "dirty": False}
    return {
        "runner": RUNNER,
        "run_id": args.run_id or (existing or {}).get("run_id"),
        "dry_run": mode,
        "items_sha256": workload["items_sha256"],
        "workload_manifest_sha256": snapshot.sha256_file(render.MANIFEST_PATH),
        "render": workload["render"],
        "split_sha256": _sha_json({"split": workload["split"], "noise": workload["noise_subset"]}),
        "served": served,
        "code_root": args.code_root,
        "code_root_commit": code["commit"],
        "code_root_dirty": code.get("dirty"),
        "snapshot": snap,
        "generation": GENERATION,
        "body_keys": pipeline.v1_body_keys(),
        "user_id": pipeline.USER_ID,
        "reviewer_role": REVIEWER_ROLE,
        "verdict_caps": {"production_question_cap": 300, "production_answer_cap": 1500,
                         "vfull_question_cap": VFULL_QUESTION_CAP},
        "window_gate": {"vendored_from": window.SOURCE_PATH, "source_sha256": window.SOURCE_SHA256,
                        "window_file": args.window_file or window.WINDOW},
        "workload_paths": {"items": str(render.ITEMS_PATH), "manifest": str(render.MANIFEST_PATH)},
        "research_commit": git_state(Path(__file__).resolve().parent)["commit"],
        "scorer_sha256": score.scorer_sha256(),
    }


def run_segment(args: argparse.Namespace) -> int:
    kind = args.cmd
    mode = _mode(args)
    for attr in ("out", "snapshot_dir", "window_file"):
        if getattr(args, attr, None):
            setattr(args, attr, str(Path(getattr(args, attr)).resolve()))
    if args.out is None:
        if mode == "real":
            _log("--out is required for a real run")
            return pipeline.RC_USAGE
        args.out = str(SCRATCH / f"dryrun-{time.strftime('%Y%m%dT%H%M%S')}")
    if args.run_id is None and mode != "real":
        args.run_id = "dryrun"
    items, workload = render.load_workload()
    run = RunDir(Path(args.out))
    try:
        with run.stage_lock(STAGE_OF[kind] + (f"-{args.suite}" if kind == "answer" else "")):
            pin = build_pin(args, mode, kind, run, workload)
            if not pin["run_id"]:
                _log("--run-id is required when opening a new run")
                return pipeline.RC_USAGE
            run.open(pin)
            allow_pause = bool(args.allow_announced_pause
                               or os.environ.get(window.PAUSE_ENV) == "1")
            ctx = pipeline.Ctx(
                run=run, items=items, workload=workload, mode=mode, limit=args.limit,
                budget_s=args.budget_s or DEFAULT_BUDGET_S[kind],
                window_path=args.window_file or window.WINDOW, margin_s=args.margin_s,
                cpuset=args.cpuset, skip_window=(mode != "real" and args.window_file is None),
                allow_announced_pause=allow_pause, pause_file=args.pause_file, log=_log)
            device = {"verdict": "gpu", "noise-verdict": "gpu", "revise": "cpu",
                      "noise-revise": "cpu"}.get(kind, "none")
            if kind in ("revise", "noise-revise", "gate", "answer") and not ctx.skip_window:
                pre = window.check(pipeline.planning(
                    "answer" if kind == "answer" else kind,
                    {"s1": "S1", "s2": "S2"}.get(getattr(args, "suite", None)))[1] + args.margin_s,
                    window_path=ctx.window_path, cpuset=ctx.cpuset,
                    **({"allow_announced_pause": True, "pause_path": ctx.pause_file}
                       if allow_pause else {}))
                if not pre["ok"]:
                    _log(f"[{kind}] CPU window refused: {pre['reasons']}")
                    return pipeline.RC_WINDOW
                if allow_pause:
                    _log(f"[{kind}] CPU window admitted (mode={pre.get('mode')})")
            bridge = Bridge(mode=mode, device=device, code_root=args.code_root, run_dir=run.out,
                            segment_tag=f"{kind}-{time.strftime('%Y%m%dT%H%M%S')}",
                            gpu_url=args.gpu_url, cpu_url=args.cpu_url,
                            lock_timeout_s=args.lock_timeout_s)
            _log(f"[{kind}] mode={mode} out={run.out}")
            _log(f"[{kind}] header: {json.dumps(bridge.header, default=str, sort_keys=True)}")
            if kind == "answer":
                transport = (V1Transport(args.base_url, timeout_s=args.timeout_s,
                                         user_id=pipeline.USER_ID)
                             if mode == "real" else fakes.FakeTransport(items))
                return pipeline.seg_answer(ctx, bridge, transport, suite=args.suite,
                                           served_probe=lambda: _served_identity(args, mode))
            preflight = ((lambda: pipeline.preflight_8083(args.gpu_url)) if mode == "real"
                         else (lambda: {"n_slots": 1, "busy_slots": [], "all_busy": False,
                                        "dry_run": True}))
            if kind == "verdict":
                return pipeline.seg_verdict(ctx, bridge, preflight=preflight)
            if kind == "noise-verdict":
                return pipeline.seg_noise_verdict(ctx, bridge, preflight=preflight)
            if kind == "revise":
                return pipeline.seg_revise(ctx, bridge)
            if kind == "noise-revise":
                return pipeline.seg_noise_revise(ctx, bridge)
            if kind == "gate":
                return pipeline.seg_gate(ctx, bridge, snapshot_dir=Path(args.snapshot_dir)
                                         if mode == "real" else None,
                                         retriever_kind=args.retriever)
            raise AssertionError(kind)
    except pipeline.Stop as stop:
        _log(f"[{kind}] stopped: {stop.reason}")
        return stop.rc
    except ManifestDrift as exc:
        _log(str(exc))
        return pipeline.RC_DRIFT
    except StageBusy as exc:
        _log(str(exc))
        return pipeline.RC_USAGE
    except InstrumentFault as exc:
        _log(f"[{kind}] INSTRUMENT FAULT: {exc}")
        return pipeline.RC_INSTRUMENT


def cmd_plan(args: argparse.Namespace) -> int:
    items, workload = render.load_workload()
    out = Path(args.out) if args.out else SCRATCH / "plan-empty"
    ctx = pipeline.Ctx(run=RunDir(out), items=items, workload=workload, mode="plan")
    counts: dict[str, int] = {}
    for row in items:
        counts[f"{row['stratum']}/{row['suite']}"] = counts.get(f"{row['stratum']}/{row['suite']}", 0) + 1
    report = {"workload": {"items_sha256": workload["items_sha256"], "counts": counts,
                           "split_counts": workload["split_counts"],
                           "noise_subset": len(workload["noise_subset"])},
              "run_dir": str(out) if args.out else None,
              "segments": pipeline.plan(ctx)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    retrieval, threshold = snapshot.resolve_config(args.code_root)
    manifest = snapshot.take_snapshot(Path(args.source_root), Path(args.dest),
                                      code_root=args.code_root, retrieval=retrieval,
                                      threshold=threshold, retriever_kind=args.retriever,
                                      force=args.force)
    print(json.dumps({k: manifest[k] for k in ("files", "retrieval_config",
                                               "review_low_q_threshold", "retriever")},
                     indent=2, sort_keys=True))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    result = score.score_run(Path(args.out), allow_incomplete=args.allow_incomplete,
                             resamples=args.resamples, boot_seed=args.boot_seed)
    print(json.dumps(score.summary(result), indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", default=None)
    p = sub.add_parser("snapshot")
    p.add_argument("--code-root", default=DEFAULT_CODE_ROOT)
    p.add_argument("--source-root", default=DEFAULT_CODE_ROOT,
                   help="orchestrator checkout whose live store is copied")
    p.add_argument("--dest", default=str(snapshot.SNAPSHOT_DIR))
    p.add_argument("--retriever", choices=("graph", "two_phase"), default="graph")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--resamples", type=int, default=score.DEFAULT_RESAMPLES)
    p.add_argument("--boot-seed", type=int, default=score.DEFAULT_BOOT_SEED)
    for name in ("answer", "verdict", "revise", "noise-verdict", "noise-revise", "gate"):
        p = sub.add_parser(name)
        if name == "answer":
            p.add_argument("--suite", choices=("s1", "s2"), required=True)
        p.add_argument("--out", default=None, help="run directory (required unless --dry-run)")
        p.add_argument("--run-id", default=None, help="stable id; required when opening a run")
        p.add_argument("--code-root", default=DEFAULT_CODE_ROOT)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--stub-orch", action="store_true",
                       help="dry-run without importing the orchestrator at all")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--budget-s", type=float, default=None)
        p.add_argument("--window-file", default=None)
        p.add_argument("--margin-s", type=float, default=120.0)
        p.add_argument("--cpuset", default="",
                       help="restore the sidecar gate's loop-reserved-cpu overlap test")
        p.add_argument("--allow-announced-pause", action="store_true",
                       help=f"also admit an announced DS41 pause (or env {window.PAUSE_ENV}=1)")
        p.add_argument("--pause-file", default=window.PAUSE_FILE)
        p.add_argument("--served-commit", default=None)
        p.add_argument("--base-url", default=DEFAULT_BASE_URL)
        p.add_argument("--gpu-url", default=GPU_URL)
        p.add_argument("--cpu-url", default=CPU_URL)
        p.add_argument("--timeout-s", type=float, default=3600.0)
        p.add_argument("--lock-timeout-s", type=float, default=30.0)
        p.add_argument("--snapshot-dir", default=str(snapshot.SNAPSHOT_DIR))
        p.add_argument("--retriever", choices=("graph", "two_phase"), default="graph")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "snapshot":
        return cmd_snapshot(args)
    if args.cmd == "score":
        return cmd_score(args)
    if args.stub_orch:
        args.dry_run = True
    return run_segment(args)


if __name__ == "__main__":
    sys.exit(main())
