#!/usr/bin/env python3
"""Run the no-inference candidate gate for research repo changes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic


DEFAULT_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("docs-check", ("make", "docs-check")),
    ("analysis-check", ("make", "analysis-check")),
    ("security-check", ("make", "security-check")),
    ("health", ("make", "health")),
    ("test", ("make", "test")),
)


@dataclass(frozen=True)
class GateStep:
    name: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class StepResult:
    name: str
    command: tuple[str, ...]
    status: str
    returncode: int | None = None
    elapsed_s: float | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_step_filter(raw: str | None) -> set[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def selected_steps(step_filter: set[str] | None) -> list[GateStep]:
    steps = [GateStep(name, command) for name, command in DEFAULT_STEPS]
    if step_filter is None:
        return steps

    known = {step.name for step in steps}
    unknown = sorted(step_filter - known)
    if unknown:
        raise SystemExit(f"unknown gate step(s): {', '.join(unknown)}")
    return [step for step in steps if step.name in step_filter]


def run_step(step: GateStep, *, root: Path, timeout_s: float) -> StepResult:
    started = monotonic()
    try:
        completed = subprocess.run(
            step.command,
            cwd=root,
            check=False,
            timeout=timeout_s,
        )
        elapsed = monotonic() - started
        status = "pass" if completed.returncode == 0 else "fail"
        return StepResult(
            name=step.name,
            command=step.command,
            status=status,
            returncode=completed.returncode,
            elapsed_s=round(elapsed, 3),
        )
    except subprocess.TimeoutExpired:
        elapsed = monotonic() - started
        return StepResult(
            name=step.name,
            command=step.command,
            status="timeout",
            returncode=None,
            elapsed_s=round(elapsed, 3),
        )


def render_report(results: list[StepResult], *, mode: str, root: Path) -> dict[str, object]:
    ok = all(result.status in {"pass", "planned"} for result in results)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "mode": mode,
        "ok": ok,
        "steps": [
            {
                "name": result.name,
                "command": list(result.command),
                "status": result.status,
                "returncode": result.returncode,
                "elapsed_s": result.elapsed_s,
            }
            for result in results
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        help="Comma-separated subset of gate steps. Defaults to all steps.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="Timeout per step when executing.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the gate. Without this flag, print the planned gate only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    steps = selected_steps(parse_step_filter(args.steps))

    if args.execute:
        results = [run_step(step, root=root, timeout_s=args.timeout_s) for step in steps]
        mode = "execute"
    else:
        results = [
            StepResult(name=step.name, command=step.command, status="planned")
            for step in steps
        ]
        mode = "plan"

    report = render_report(results, mode=mode, root=root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"candidate gate mode={mode} ok={str(report['ok']).lower()}")
        for result in results:
            command = " ".join(result.command)
            print(f"- {result.name}: {result.status} :: {command}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
