#!/usr/bin/env python3
"""UFH-13 thesis runner: does the orchestrator beat the strongest model alone?

SENDS INFERENCE (``run``, ``pilot``). ``plan``, ``score`` and ``pilot-report`` never do.
Run from the research repo root. The main session runs the inference commands inside its
bus-granted window, with AutoPilot quiesced and the orchestrator flag ``v1_escalation`` ON.

  plan          print the arms, the suite fingerprint and the checks; no network.
  pilot N       A2 ONLY on N items from the PILOT POOL (outside the frozen 395; see
                pilot_pool.py), allocated by the frozen suite's subject mix -> escalation rate
                by trigger and the review verdict distribution. Answers "does A2 escalate at
                all?" before the full window. Any pilot item whose question hash is in the
                frozen suite is refused. Pilot records can never be scored against the rule.
  run           A0/A1/A2 over the frozen 395 items, interleaved per item in seeded random order
                (blocks of 3). Resumable: rerun the same command to continue.
  score         apply the pre-registered rule to a finished run; writes score.json and the
                ClaimTuple-projectable belief_measurements.jsonl.
  pilot-report  re-print a pilot's report from its records.

Every question is persisted as soon as it finishes (``records.jsonl``, fsynced). Rerunning
``run``/``pilot`` with the same ``--out`` skips what is on disk. The run manifest pins the arms,
the transport, the suite sha and the orchestrator commit; a resume with different settings is
refused.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.benchmark.thesis_ufh13"

from .arms import ARMS, ENABLE_THINKING, GENERATION  # noqa: E402
from .records import (  # noqa: E402
    MANIFEST_NAME,
    RECORD_SCHEMA,
    RECORDS_NAME,
    append_record,
    done_keys,
    read_records,
    summarize_receipts,
)
from .score import SCORER_PATH, is_correct, score_run  # noqa: E402
from .pilot_pool import (  # noqa: E402
    composition,
    frozen_hashes,
    load_pool,
    mix_matched_sample,
    refuse_frozen,
)
from .suite import SUITE_PATH, SUITE_SHA256, Item, file_sha256, load_suite  # noqa: E402
from .transports import OpenCodeTransport, Transport, V1Transport, session_id_for  # noqa: E402

RUNNER_VERSION = "ufh13-runner/v1"
DEFAULT_PREREG = Path("/mnt/raid0/llm/epyc-root/handoffs/active/"
                      "thesis-experiment-orchestrator-vs-strongest-model.md")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def schedule(items: list[Item], arms: list[str], seed: int) -> list[tuple[str, Item]]:
    """Items in seeded random order; per item the arms in a seeded random order (block of 3)."""
    rng = random.Random(seed)
    order = rng.sample(items, len(items))
    plan: list[tuple[str, Item]] = []
    for item in order:
        block = rng.sample(arms, len(arms))
        plan.extend((arm, item) for arm in block)
    return plan


def build_manifest(args: argparse.Namespace, *, pilot: bool, arms: list[str],
                   items: list[Item], transport: Transport,
                   pilot_pool_sha256: str | None = None) -> dict[str, Any]:
    prereg = Path(args.preregistration) if args.preregistration else None
    return {
        "pilot_pool_sha256": pilot_pool_sha256,
        "pilot_items_composition": composition(items) if pilot else None,
        "runner": RUNNER_VERSION,
        "run_id": args.run_id,
        "pilot": pilot,
        "arms": {arm: {"description": ARMS[arm].description, "body_keys": ARMS[arm].body_keys}
                 for arm in arms},
        "generation": GENERATION,
        "enable_thinking": ENABLE_THINKING,
        "seed": args.seed,
        "items": [item.item_id for item in items],
        "suite_path": str(SUITE_PATH),
        "suite_sha256": SUITE_SHA256,
        "scorer_path": str(SCORER_PATH),
        "scorer_sha256": file_sha256(SCORER_PATH),
        "preregistration_path": str(prereg) if prereg else None,
        "preregistration_sha256": file_sha256(prereg) if prereg and prereg.is_file() else None,
        "orchestrator_commit": args.orchestrator_commit,
        "research_commit": _git_head(Path(__file__).resolve().parent),
        "transport": transport.describe(),
    }


# Fields that must match on resume (a changed one is a different experiment).
_RESUME_KEYS = ("runner", "run_id", "pilot", "arms", "generation", "seed", "items",
                "suite_sha256", "scorer_sha256", "transport", "pilot_pool_sha256")


def open_run(out: Path, manifest: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST_NAME
    if path.exists():
        old = json.loads(path.read_text())
        diffs = [k for k in _RESUME_KEYS if old.get(k) != manifest.get(k)]
        if diffs:
            raise SystemExit(f"refusing to resume {out}: manifest differs on {diffs}")
        return
    path.write_text(json.dumps({**manifest, "created_at": _now()}, indent=2, sort_keys=True) + "\n")


def run_items(out: Path, plan: list[tuple[str, Item]], transport: Transport, *, run_id: str,
              pilot: bool, log: Callable[[str], None] = print) -> int:
    """Run every (arm, item) not yet on disk; returns how many were run now."""
    records_path = out / RECORDS_NAME
    done = done_keys(read_records(records_path))
    todo = [(arm, item) for arm, item in plan if (arm, item.item_id) not in done]
    log(f"{len(done)} on disk, {len(todo)} to run -> {records_path}")
    for position, (arm, item) in enumerate(todo, 1):
        session_id = session_id_for(run_id, arm, item.item_id)
        started_at = _now()
        t0 = time.perf_counter()
        result = transport.ask(arm, item.item_id, item.prompt, session_id)
        wall_s = round(time.perf_counter() - t0, 3)
        costs = summarize_receipts(arm, result.receipts)
        record = {
            "schema": RECORD_SCHEMA,
            "run_id": run_id,
            "pilot": pilot,
            "arm": arm,
            "item_id": item.item_id,
            "suite": item.suite,
            "expected": item.expected,
            "status": result.status,
            "answer_text": result.text,
            "correct": result.status == "ok" and is_correct(result.text, item.expected),
            "finish_reason": result.finish_reason,
            "http_status": result.http_status,
            "error": result.error,
            "session_id": result.session_id or session_id,
            "served_role": result.served_role,
            "usage": result.usage,
            "wall_s": wall_s,
            "started_at": started_at,
            "transport": transport.name,
            "receipts": result.receipts,
            **costs,
            "transport_extra": result.extra,
        }
        append_record(records_path, record)
        log(f"[{position}/{len(todo)}] {arm} {item.item_id} status={result.status} "
            f"correct={record['correct']} fired={costs['escalation_fired']} "
            f"consultant_s={costs['consultant_device_seconds']} wall={wall_s}s")
    return len(todo)


def pilot_report(records: list[dict[str, Any]],
                 manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Escalation rate by trigger, review verdicts, costs. Answers "does A2 escalate at all?"."""
    manifest = manifest or {}
    rows = [r for r in records if r.get("arm") == "A2"]
    n = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]

    def rate(count: int) -> float | None:
        return round(count / n, 4) if n else None

    fired = sum(1 for r in rows if r.get("escalation_fired"))
    by_trigger: Counter[str] = Counter()
    items_by_trigger: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    to_roles: Counter[str] = Counter()
    models: Counter[str] = Counter()
    for r in rows:
        triggers = [t for t in r.get("escalation_triggers") or [] if t]
        by_trigger.update(triggers)
        items_by_trigger.update(set(triggers))
        verdicts.update(v for v in r.get("review_verdicts") or [] if v)
        to_roles.update(t for t in r.get("escalation_to_roles") or [] if t)
        models.update(m for m in r.get("escalation_models") or [] if m)
    consultant = [r["consultant_device_seconds"] for r in rows
                  if r.get("consultant_device_seconds") is not None]
    disabled = Counter(reason for r in rows for reason in r.get("escalation_disabled_reasons") or [])
    return {
        "pilot_pool_sha256": manifest.get("pilot_pool_sha256"),
        "pilot_items_composition": manifest.get("pilot_items_composition"),
        "n_items": n,
        "statuses": dict(Counter(r.get("status") for r in rows)),
        "accuracy": round(sum(1 for r in ok if r.get("correct")) / n, 4) if n else None,
        "escalation_rate": rate(fired),
        "escalated_items": fired,
        "items_by_trigger": {t: {"items": c, "rate": rate(c)} for t, c in items_by_trigger.items()},
        "steps_by_trigger": dict(by_trigger),
        "review_verdicts": dict(verdicts),
        "escalation_to_roles": dict(to_roles),
        "escalation_models": dict(models),
        "escalation_not_enabled": dict(disabled),
        "consultant_device_seconds": {
            "total": round(sum(consultant), 3) if consultant else 0.0,
            "mean_per_item": round(sum(consultant) / len(consultant), 3) if consultant else None,
            "measured_items": len(consultant),
        },
        "cost_problems": dict(Counter(p for r in rows for p in r.get("cost_problems") or [])),
        "wall_s_total": round(sum(r.get("wall_s") or 0 for r in rows), 1),
    }


def make_transport(args: argparse.Namespace, out: Path) -> Transport:
    if args.transport == "v1":
        return V1Transport(args.base_url, timeout_s=args.timeout_s)
    return OpenCodeTransport(epyc_root=Path(args.epyc_root), workdir=out,
                             tap_events=Path(args.tap_events), timeout_s=args.timeout_s)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, help="run directory (records.jsonl, manifest)")
    p.add_argument("--run-id", required=True, help="stable id; part of every x_session_id")
    p.add_argument("--seed", type=int, default=42, help="order / pilot sampling seed")
    p.add_argument("--transport", choices=("v1", "opencode"), default="v1")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--epyc-root", default="/mnt/raid0/llm/epyc-root")
    p.add_argument("--tap-events", default="/mnt/raid0/llm/tmp/inference_tap_events.jsonl")
    p.add_argument("--orchestrator-commit", required=True,
                   help="the commit the live API serves (verify the uvicorn start time)")
    p.add_argument("--preregistration", default=str(DEFAULT_PREREG))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p_pilot = sub.add_parser("pilot")
    p_pilot.add_argument("n", type=int)
    _common(p_pilot)
    p_run = sub.add_parser("run")
    _common(p_run)
    p_score = sub.add_parser("score")
    p_score.add_argument("--out", required=True)
    p_score.add_argument("--x", type=float, default=0.75)
    p_score.add_argument("--y", type=float, default=0.50)
    p_score.add_argument("--resamples", type=int, default=10_000)
    p_score.add_argument("--allow-incomplete", action="store_true")
    p_report = sub.add_parser("pilot-report")
    p_report.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "plan":
        items = load_suite()
        print(f"suite {SUITE_PATH}\n  sha256 {SUITE_SHA256} (verified), {len(items)} items")
        for arm, spec in ARMS.items():
            print(f"  {arm}: {spec.description}; body keys {spec.body_keys}")
        print(f"generation {GENERATION}; enable_thinking {ENABLE_THINKING}")
        print("checks the operator of the run confirms: v1_escalation flag ON; API restarted "
              "after the TE-1 commits; role swap applied and serving proved (TE-2); tap on.")
        return 0
    if args.cmd == "score":
        result = score_run(Path(args.out), load_suite(), x=args.x, y=args.y,
                           resamples=args.resamples, allow_incomplete=args.allow_incomplete)
        print(json.dumps({k: result[k] for k in ("verdict", "reasons", "accuracy", "G", "d",
                                                 "G_ci95", "gap_ci95")}, indent=2))
        return 0
    if args.cmd == "pilot-report":
        report = pilot_report(read_records(Path(args.out) / RECORDS_NAME),
                              json.loads((Path(args.out) / MANIFEST_NAME).read_text()))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    out = Path(args.out)
    frozen = load_suite()
    pilot = args.cmd == "pilot"
    pool_sha = None
    if pilot:
        pool, pool_sha = load_pool()
        items = mix_matched_sample(pool, frozen, args.n, args.seed)
        refuse_frozen(items, frozen_hashes(frozen))  # by construction, and checked again here
        arms = ["A2"]
    else:
        items = frozen
        arms = list(ARMS)
    transport = make_transport(args, out)
    manifest = build_manifest(args, pilot=pilot, arms=arms, items=items, transport=transport,
                              pilot_pool_sha256=pool_sha)
    open_run(out, manifest)
    run_items(out, schedule(items, arms, args.seed), transport, run_id=args.run_id, pilot=pilot)
    if pilot:
        report = pilot_report(read_records(out / RECORDS_NAME), manifest)
        (out / "pilot_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
