#!/usr/bin/env python3
"""Capture P-BENCH-1 host evidence without assigning measurement grade."""
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import server_np_sweep
from server_np_sweep import collect_attestation, host_health_warnings


def _grade(status, known, warnings):
    """S-11 fail-closed grade: decision_grade is True only with a present
    attestation, a known uptime and no host-health warning (incl. uptime >= 1 week)."""
    reasons = []
    if status != "present":
        reasons.append("attestation_missing")
    elif not known:
        reasons.append("uptime_unknown")
    reasons.extend(f"host_health_warning: {w}" for w in warnings)
    return (not reasons), reasons


def missing_attestation(claim_mode, error):
    """Degraded artifact when collection itself failed: still emitted, never decision-grade."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema": "canonical-host-attestation-v1",
        "claim_mode": claim_mode,
        "capture_started_at": now,
        "capture_ended_at": now,
        "attestation": {"uptime_seconds": None},
        "attestation_status": "missing",
        "attestation_error": repr(error),
        "uptime_status": "unknown",
        "host_health_warnings": [],
        "decision_grade": False,
        "decision_grade_blockers": ["attestation_missing"],
        "coverage_note": "Host attestation could not be produced; run is NOT decision-grade (S-11 fail-closed).",
    }


def capture(claim_mode):
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic_ns = time.monotonic_ns()
    attestation = collect_attestation()
    ended_monotonic_ns = time.monotonic_ns()
    ended_at = datetime.now(timezone.utc).isoformat()
    warnings = host_health_warnings(attestation)
    uptime = attestation.get("uptime_seconds")
    try:
        known = (isinstance(uptime, (int, float)) and not isinstance(uptime, bool)
                 and math.isfinite(uptime) and uptime >= 0)
    except OverflowError:
        known = False
    if not known:
        # JSON has no portable representation for NaN or infinity.
        attestation["uptime_seconds"] = None
    decision_grade, blockers = _grade("present", known, warnings)
    return {
        "schema": "canonical-host-attestation-v1",
        "attestation_status": "present",
        "decision_grade": decision_grade,
        "decision_grade_blockers": blockers,
        "source": "scripts/benchmark/server_np_sweep.py:collect_attestation,host_health_warnings",
        "claim_mode": claim_mode,
        "capture_started_at": started_at,
        "capture_ended_at": ended_at,
        "capture_started_monotonic_ns": started_monotonic_ns,
        "capture_ended_monotonic_ns": ended_monotonic_ns,
        "capture_scope": "after CPU region claim admission when enabled, immediately before launch; unlocked if CANONICAL_SKIP_REGION_LOCK=1",
        "source_sha256": hashlib.sha256(Path(server_np_sweep.__file__).read_bytes()).hexdigest(),
        "attestation": attestation,
        "uptime_status": "known" if known else "unknown",
        "uptime_raw_if_unknown": None if known else repr(uptime),
        "host_health_warnings": warnings,
        "coverage_note": "Partial host evidence only; does not attest cache drop, NUMA re-warm, or THP state, and does not determine compliance or grade.",
        "warning_boundary_note": "Warning check is uptime >= 1 week (P-BENCH-1 Annex B: >= 1 week requires reboot). Unknown uptime is not evidence of compliance and is never decision-grade.",
    }


def main():
    if len(sys.argv) < 6 or sys.argv[4] != "--" or sys.argv[3] not in ("cpu_region_claim", "explicit_unlocked_override"):
        raise SystemExit("usage: canonical_host_attestation.py OUTPUT BENCH_LOG CLAIM_MODE -- COMMAND [ARGS...]")
    try:
        try:
            artifact = capture(sys.argv[3])
        except Exception as exc:  # S-11: still emit an artifact, marked missing
            artifact = missing_attestation(sys.argv[3], exc)
        payload = json.dumps(artifact, sort_keys=True, allow_nan=False)
        # Exclusive creation prevents stale evidence from being silently reused.
        with Path(sys.argv[1]).open("x", encoding="utf-8") as out:
            out.write(payload + "\n")
        with Path(sys.argv[2]).open("a", encoding="utf-8") as log:
            log.write(payload + "\n")
            log.flush()
            os.fsync(log.fileno())
    except Exception as exc:
        raise SystemExit(f"ERROR: host-health attestation failed; refusing benchmark: {exc}") from exc
    os.execvp(sys.argv[5], sys.argv[5:])


if __name__ == "__main__":
    main()
