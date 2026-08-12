"""Prospective GEAK/AgentKernelArena round-trip receipt producer.

The producer records two independent measured rates for Vidya: correctness and
timing-harness validity.  Preflight facts are deliberately retained only as
dependency evidence; source pins, licences, hardware identity and registry
shape have no honest ordinal metric and must never be coerced into one.

This module does not launch an agent, compile a kernel, profile, or score speed.
The governed arena runner calls it at write time with the counts it observed.
Older receipts are never retrofitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from . import arena_adapter


SCHEMA = "epyc.autokernel.geak_arena_roundtrip.v1"
PRODUCER_ID = "autokernel.controller.arena_roundtrip/v1"
PREFLIGHT_CLASSIFICATION = "dependency_evidence_only"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RoundTripReceiptError(ValueError):
    """A round-trip result cannot support a prospective governed receipt."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RoundTripReceiptError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    value = _text(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise RoundTripReceiptError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ScoredCount:
    passed: int
    total: int
    reps_basis: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.passed, bool)
            or not isinstance(self.passed, int)
            or isinstance(self.total, bool)
            or not isinstance(self.total, int)
            or self.total < 1
            or not 0 <= self.passed <= self.total
        ):
            raise RoundTripReceiptError(
                "scored counts require integers with 0 <= passed <= total and total >= 1"
            )
        _text(self.reps_basis, "reps_basis")

    @property
    def rate(self) -> float:
        value = self.passed / self.total
        if not math.isfinite(value):  # defensive; integer inputs make this unreachable
            raise RoundTripReceiptError("scored rate must be finite")
        return value


def _measurement(
    *, measurement_id: str, metric: str, measurement_role: str, claim: str,
    count: ScoredCount, extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "measurement_id": measurement_id,
        "metric": metric,
        "value": count.rate,
        "unit": "fraction",
        "metric_direction": "higher_better",
        "category": "CANDIDATE",
        "claim": claim,
        "reps": count.total,
        "reps_basis": count.reps_basis,
        "extra": {
            "measurement_role": measurement_role,
            "passed": count.passed,
            "total": count.total,
            **dict(extra),
        },
    }


def build_receipt(
    *, campaign_id: str, task_id: str, controller_id: str,
    attempt_id: str | None = None, claim_campaign_id: str | None = None,
    started_at: str, ended_at: str,
    correctness: ScoredCount, timing_validity: ScoredCount,
    preflight_locator: str, preflight_sha256: str,
    source: Mapping[str, Any], artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Build one successful producer receipt from observed round-trip counts.

    ``status=pass`` means the producer captured a complete, internally valid
    record.  It does not mean the authored kernel was correct or fast; those
    outcomes remain the two explicit rates below.
    """
    campaign = _text(campaign_id, "campaign_id")
    task = _text(task_id, "task_id")
    controller = _text(controller_id, "controller_id")
    attempt = (_text(attempt_id, "attempt_id")
               if attempt_id is not None else None)
    claim_scope = (_text(claim_campaign_id, "claim_campaign_id")
                   if claim_campaign_id is not None else None)
    if (attempt is None) != (claim_scope is None):
        raise RoundTripReceiptError(
            "attempt_id and claim_campaign_id must be supplied together")
    if attempt is not None and claim_scope != attempt:
        raise RoundTripReceiptError(
            "claim_campaign_id must equal the campaign attempt")
    if controller not in arena_adapter.CONTROLLERS:
        raise RoundTripReceiptError(
            f"controller_id must be registered; observed {controller!r}"
        )
    start = _text(started_at, "started_at")
    end = _text(ended_at, "ended_at")
    locator = _text(preflight_locator, "preflight_locator")
    preflight_digest = _sha256(preflight_sha256, "preflight_sha256")
    if not isinstance(source, Mapping) or not source:
        raise RoundTripReceiptError("source must be a non-empty object")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RoundTripReceiptError("artifacts must be a non-empty path-to-SHA object")
    artifact_rows = []
    for path, digest in sorted(artifacts.items()):
        artifact_rows.append({
            "path": _text(path, "artifact path"),
            "sha256": _sha256(digest, f"artifact {path!r} SHA-256"),
        })
    shared = {"task_id": task, "controller_id": controller}
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "producer_id": PRODUCER_ID,
        "status": "pass",
        "authority": "diagnostic_only",
        "campaign_id": campaign,
        **({"attempt_id": attempt, "claim_campaign_id": claim_scope}
           if attempt is not None else {}),
        "started_at": start,
        "ended_at": end,
        "task": {"task_id": task, "controller_id": controller},
        "source": dict(source),
        "artifacts": artifact_rows,
        "dependencies": {
            "preflight": {
                "schema": arena_adapter.PREFLIGHT_SCHEMA,
                "locator": locator,
                "sha256": preflight_digest,
                "classification": PREFLIGHT_CLASSIFICATION,
                "belief_measurement_emitted": False,
            },
        },
        "belief_measurements": [
            _measurement(
                measurement_id="arena_correctness_pass_rate",
                metric="geak_arena_correctness_pass_rate",
                measurement_role="kernel_authoring_correctness",
                claim="Fraction of scored GEAK/Arena correctness cases that passed",
                count=correctness,
                extra=shared,
            ),
            _measurement(
                measurement_id="arena_timing_harness_validity_rate",
                metric="geak_arena_timing_harness_validity_rate",
                measurement_role="kernel_authoring_timing_validity",
                claim="Fraction of scored GEAK/Arena timing repetitions admitted as valid",
                count=timing_validity,
                extra=shared,
            ),
        ],
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["receipt_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return receipt


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically write a receipt already built by :func:`build_receipt`."""
    if not isinstance(receipt, Mapping) or receipt.get("producer_id") != PRODUCER_ID:
        raise RoundTripReceiptError("receipt was not built by this producer")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


__all__ = [
    "PREFLIGHT_CLASSIFICATION", "PRODUCER_ID", "SCHEMA", "RoundTripReceiptError",
    "ScoredCount", "build_receipt", "write_receipt",
]
