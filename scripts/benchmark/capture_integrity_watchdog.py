#!/usr/bin/env python3
"""Read-only watchdog for canonical benchmark ``*.live-status.json`` sidecars.

The watchdog distinguishes harness/capture defects from model outcomes.  In
particular, length caps and prompt-contract misses are reported as warnings;
they never cause this process to abort a benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


CANONICAL_CAPTURE_SCHEMA = "v7_quality_gate_capture.v4"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
READ_ATTEMPTS = 3


@dataclass(frozen=True)
class CheckResult:
    """The result of validating one immutable status snapshot."""

    completed_draws: int
    complete: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def read_status_snapshot(path: Path) -> dict[str, Any]:
    """Read a stable JSON snapshot, tolerating a writer's atomic replacement."""
    last_error: Exception | None = None
    for attempt in range(READ_ATTEMPTS):
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_ino, before.st_mtime_ns, before.st_size) != (
                after.st_ino,
                after.st_mtime_ns,
                after.st_size,
            ):
                raise OSError("status changed during read")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("status root must be a JSON object")
            return parsed
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < READ_ATTEMPTS:
                time.sleep(0.02)
    assert last_error is not None
    raise ValueError(str(last_error)) from last_error


def validate_status(status: dict[str, Any], *, request_error_threshold: int) -> CheckResult:
    """Classify one canonical status without judging model-side outcomes."""
    errors: list[str] = []
    warnings: list[str] = []

    if request_error_threshold < 1:
        errors.append("watchdog configuration request-error threshold must be at least 1")
    if status.get("schema_version") != CANONICAL_CAPTURE_SCHEMA:
        errors.append("unsupported or missing capture schema version")
    source_hash = status.get("runner_source_sha256")
    if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash):
        errors.append("missing or invalid runner provenance hash")
    if not isinstance(status.get("suite"), str) or not status["suite"]:
        errors.append("missing suite identity")

    completed = status.get("completed_draws")
    expected = status.get("expected_draws")
    complete = status.get("complete")
    if not _is_nonnegative_int(completed) or not _is_nonnegative_int(expected):
        errors.append("invalid completed/expected draw counters")
        completed_value = 0
    else:
        completed_value = completed
        if completed > expected:
            errors.append("completed draws exceed expected draws")
    if not isinstance(complete, bool):
        errors.append("missing or invalid completion flag")
    elif _is_nonnegative_int(completed) and _is_nonnegative_int(expected) and complete != (completed >= expected):
        errors.append("completion flag disagrees with draw counters")

    request_errors = status.get("request_error_rows")
    if not _is_nonnegative_int(request_errors):
        errors.append("invalid request-error counter")
    elif request_errors >= request_error_threshold:
        errors.append(
            f"request-error threshold reached ({request_errors}/{request_error_threshold})"
        )

    integrity = status.get("artifact_integrity_fail_closed")
    if not isinstance(integrity, bool):
        errors.append("missing or invalid artifact-integrity flag")
    elif integrity:
        errors.append("artifact_integrity_fail_closed")

    length_caps = status.get("length_cap_rows")
    if not _is_nonnegative_int(length_caps):
        errors.append("invalid length-cap counter")
    elif length_caps:
        warnings.append(f"model length caps observed ({length_caps})")

    swe = status.get("swebench_search_replace")
    if not isinstance(swe, dict):
        errors.append("missing SWE capture status")
    else:
        states = swe.get("state_counts")
        if not isinstance(states, dict):
            errors.append("invalid SWE capture state counts")
        else:
            for state in (
                "prompt_contract_candidate",
                "model_truncation_no_patch",
                "model_truncation_partial_patch",
            ):
                count = states.get(state, 0)
                if not _is_nonnegative_int(count):
                    errors.append(f"invalid SWE state count for {state}")
                elif count:
                    warnings.append(f"model outcome {state} ({count})")

    return CheckResult(
        completed_draws=completed_value,
        complete=bool(complete) if isinstance(complete, bool) else False,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _emit(path: Path, result: CheckResult) -> None:
    progress = "complete" if result.complete else f"draws={result.completed_draws}"
    print(f"[capture-watchdog] {path}: {progress}", file=sys.stderr)
    for warning in result.warnings:
        print(f"[capture-watchdog] WARNING {path}: {warning}", file=sys.stderr)


def watch_paths(
    paths: list[Path],
    *,
    watch: bool,
    poll_interval_s: float,
    startup_grace_s: float,
    stale_timeout_s: float,
    request_error_threshold: int,
    require_complete: bool = True,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Watch paths until completion, or observe each once when explicitly asked.

    No status write is attempted.  Invalid or unavailable status files are
    tolerated only through startup grace; a valid status must make draw-count
    progress before the stale timeout while watch mode is active.
    """
    if poll_interval_s <= 0 or startup_grace_s < 0 or stale_timeout_s <= 0:
        print("[capture-watchdog] FAIL invalid timing configuration", file=sys.stderr)
        return 2
    if request_error_threshold < 1:
        print("[capture-watchdog] FAIL request-error threshold must be at least 1", file=sys.stderr)
        return 2

    started = monotonic()
    last_progress: dict[Path, tuple[int, float]] = {}
    last_emitted: dict[Path, tuple[int, bool, tuple[str, ...]]] = {}
    pending = set(paths)
    while pending:
        now = monotonic()
        for path in tuple(pending):
            try:
                result = validate_status(
                    read_status_snapshot(path),
                    request_error_threshold=request_error_threshold,
                )
            except ValueError as exc:
                if now - started >= startup_grace_s:
                    print(f"[capture-watchdog] FAIL {path}: missing or malformed status: {exc}", file=sys.stderr)
                    return 1
                continue

            if result.errors:
                print(f"[capture-watchdog] FAIL {path}: {'; '.join(result.errors)}", file=sys.stderr)
                return 1
            emitted = (result.completed_draws, result.complete, result.warnings)
            if last_emitted.get(path) != emitted:
                _emit(path, result)
                last_emitted[path] = emitted
            previous = last_progress.get(path)
            if previous is None or result.completed_draws > previous[0]:
                last_progress[path] = (result.completed_draws, now)
            elif watch and not result.complete and now - previous[1] >= stale_timeout_s:
                print(f"[capture-watchdog] FAIL {path}: no draw-count progress for {stale_timeout_s:g}s", file=sys.stderr)
                return 1
            if result.complete:
                pending.remove(path)

        if not watch:
            # A one-shot invocation is a completion gate, not a passive health
            # probe.  Callers that intentionally inspect live work use
            # --observe-once, which maps to ``watch=False`` plus this flag.
            if require_complete and pending:
                print(f"[capture-watchdog] FAIL {next(iter(pending))}: capture incomplete", file=sys.stderr)
                return 1
            return 0
        if pending:
            sleep(poll_interval_s)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status_paths", metavar="STATUS", nargs="+", type=Path)
    parser.add_argument("--watch", action="store_true", help="Poll until every status is complete")
    parser.add_argument("--observe-once", action="store_true",
                        help="Validate a live snapshot once without requiring completion")
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--startup-grace-s", type=float, default=30.0)
    parser.add_argument("--stale-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-error-threshold", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.watch and args.observe_once:
        raise SystemExit("--watch and --observe-once are mutually exclusive")
    if args.observe_once:
        return watch_paths(
            args.status_paths,
            watch=False,
            poll_interval_s=args.poll_interval_s,
            startup_grace_s=args.startup_grace_s,
            stale_timeout_s=args.stale_timeout_s,
            request_error_threshold=args.request_error_threshold,
            require_complete=False,
        )
    return watch_paths(
        args.status_paths,
        watch=args.watch,
        poll_interval_s=args.poll_interval_s,
        startup_grace_s=args.startup_grace_s,
        stale_timeout_s=args.stale_timeout_s,
        request_error_threshold=args.request_error_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
