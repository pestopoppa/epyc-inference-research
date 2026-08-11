"""Prospective write-side receipts for real AutoKernel MMQ WGM profiles.

This module is deliberately a pure producer: it does not launch inference, acquire a
device, or parse historical evidence.  A successor real-MMQ launch-order experiment
must call :func:`build_receipt` while its raw wall-time and counter observations still
have producer-owned provenance.  The admitted 2026-08-11 r2 negative predates this
hook and must never be reconstructed through it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence


SCHEMA = "epyc.autokernel.mmq_wgm_profile.v1"
PRODUCER_ID = "scripts.benchmark.autokernel_mmq_wgm_receipt/v1"
AUTHORITY = "diagnostic_only"
_CLAIM_SCHEMA = "epyc.autokernel.device_claim_receipt.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_CATEGORIES = frozenset({"BASELINE", "CANDIDATE", "OPTIMUM"})


class WgmReceiptError(ValueError):
    """Observed inputs cannot support a governed prospective WGM receipt."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WgmReceiptError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    value = _text(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise WgmReceiptError(f"{label} must be a lowercase SHA-256")
    return value


def _scored_basis(value: object, label: str) -> str:
    value = _text(value, label)
    if not value.startswith("scored:"):
        raise WgmReceiptError(f"{label} must begin with 'scored:'")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WgmReceiptError(f"{label} must be a finite number")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < 0 or (positive and rendered <= 0):
        qualifier = "positive finite" if positive else "non-negative finite"
        raise WgmReceiptError(f"{label} must be a {qualifier} number")
    return rendered


def _identity(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise WgmReceiptError("source must be an object")
    commit = _text(source.get("base_commit"), "source.base_commit")
    if not _COMMIT_RE.fullmatch(commit):
        raise WgmReceiptError("source.base_commit must be a lowercase 40-character commit")
    identity = {
        "repo": _text(source.get("repo"), "source.repo"),
        "base_commit": commit,
        "state": _text(source.get("state"), "source.state"),
        "source_path": _text(source.get("source_path"), "source.source_path"),
        "source_sha256": _sha256(source.get("source_sha256"), "source.source_sha256"),
    }
    diff_digest = source.get("source_diff_sha256")
    if diff_digest is not None:
        identity["source_diff_sha256"] = _sha256(
            diff_digest, "source.source_diff_sha256"
        )
    return identity


def _evidence_ref(value: Mapping[str, Any], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise WgmReceiptError(f"{label} must be an object")
    return {
        "locator": _text(value.get("locator"), f"{label}.locator"),
        "sha256": _sha256(value.get("sha256"), f"{label}.sha256"),
    }


def _device_claim(value: Mapping[str, Any], *, campaign_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WgmReceiptError("device_claim must be an object")
    opened = value.get("opened")
    released = value.get("released")
    if not isinstance(opened, Mapping) or not isinstance(released, Mapping):
        raise WgmReceiptError("device_claim requires opened and released receipts")
    for label, receipt in (("opened", opened), ("released", released)):
        if receipt.get("schema") != _CLAIM_SCHEMA:
            raise WgmReceiptError(f"device_claim.{label} has the wrong schema")
        _text(receipt.get("claim_id"), f"device_claim.{label}.claim_id")
        _text(receipt.get("device_id"), f"device_claim.{label}.device_id")
        if receipt.get("campaign_id") != campaign_id:
            raise WgmReceiptError(
                f"device_claim.{label}.campaign_id must match the WGM campaign"
            )
    for field in ("claim_id", "device_id", "acquired_at"):
        if opened.get(field) != released.get(field):
            raise WgmReceiptError(f"device claim {field} changed across release")
    _text(released.get("released_at"), "device_claim.released.released_at")
    return {"opened": dict(opened), "released": dict(released)}


def _surface(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise WgmReceiptError("surface must be a non-empty object")
    result = dict(value)
    _text(result.get("surface_id"), "surface.surface_id")
    return result


@dataclass(frozen=True)
class CounterSample:
    """All-MMQ totals from one scored profiler repetition."""

    tcc_hits: float
    tcc_misses: float
    read_requests: float
    dispatches: int

    def __post_init__(self) -> None:
        _number(self.tcc_hits, "counter sample tcc_hits")
        _number(self.tcc_misses, "counter sample tcc_misses")
        _number(self.read_requests, "counter sample read_requests")
        if isinstance(self.dispatches, bool) or not isinstance(self.dispatches, int):
            raise WgmReceiptError("counter sample dispatches must be a positive integer")
        if self.dispatches < 1:
            raise WgmReceiptError("counter sample dispatches must be a positive integer")
        if self.tcc_hits + self.tcc_misses <= 0:
            raise WgmReceiptError("counter sample must contain at least one TCC lookup")


def _producer_identity() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "producer_id": PRODUCER_ID,
        "path": "scripts/benchmark/autokernel_mmq_wgm_receipt.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _measurement(
    *, measurement_id: str, metric: str, value: float, unit: str,
    direction: str, category: str, reps: int, reps_basis: str,
    claim: str, extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "measurement_id": measurement_id,
        "metric": metric,
        "value": value,
        "unit": unit,
        "metric_direction": direction,
        "category": category,
        "reps": reps,
        "reps_basis": reps_basis,
        "claim": claim,
        "extra": dict(extra),
    }


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash the canonical payload, excluding its self-identifying digest field."""
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_receipt(
    *, campaign_id: str, started_at: str, ended_at: str,
    wgm_arm: int, category: str,
    wall_time_samples_ms: Sequence[float], wall_time_reps_basis: str,
    counter_samples: Sequence[CounterSample], counter_reps_basis: str,
    surface: Mapping[str, Any], source: Mapping[str, Any],
    device_claim: Mapping[str, Any], wall_time_evidence: Mapping[str, Any],
    counter_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one successful per-arm receipt from current producer observations."""
    campaign = _text(campaign_id, "campaign_id")
    start = _text(started_at, "started_at")
    end = _text(ended_at, "ended_at")
    if isinstance(wgm_arm, bool) or not isinstance(wgm_arm, int) or wgm_arm < 0:
        raise WgmReceiptError("wgm_arm must be a non-negative integer; 0 means none")
    if category not in _CATEGORIES:
        raise WgmReceiptError(f"category must be one of {sorted(_CATEGORIES)}")
    wall_basis = _scored_basis(wall_time_reps_basis, "wall_time_reps_basis")
    counter_basis = _scored_basis(counter_reps_basis, "counter_reps_basis")
    if isinstance(wall_time_samples_ms, (str, bytes)) or not isinstance(
        wall_time_samples_ms, Sequence
    ):
        raise WgmReceiptError("wall_time_samples_ms must be a sequence")
    if isinstance(counter_samples, (str, bytes)) or not isinstance(
        counter_samples, Sequence
    ):
        raise WgmReceiptError("counter_samples must be a sequence")
    walls = tuple(
        _number(value, f"wall_time_samples_ms[{index}]", positive=True)
        for index, value in enumerate(wall_time_samples_ms)
    )
    counters = tuple(counter_samples)
    if not walls:
        raise WgmReceiptError("wall_time_samples_ms must contain scored repetitions")
    if not counters or any(not isinstance(row, CounterSample) for row in counters):
        raise WgmReceiptError("counter_samples must contain CounterSample repetitions")

    source_identity = _identity(source)
    surface_identity = _surface(surface)
    claim = _device_claim(device_claim, campaign_id=campaign)
    wall_evidence = _evidence_ref(wall_time_evidence, "wall_time_evidence")
    counter_evidence_ref = _evidence_ref(counter_evidence, "counter_evidence")

    total_hits = sum(float(row.tcc_hits) for row in counters)
    total_misses = sum(float(row.tcc_misses) for row in counters)
    total_read_requests = sum(float(row.read_requests) for row in counters)
    total_dispatches = sum(row.dispatches for row in counters)
    hit_rate = total_hits / (total_hits + total_misses)
    read_requests_per_rep = total_read_requests / len(counters)
    arm_label = "none" if wgm_arm == 0 else str(wgm_arm)
    shared_extra = {
        "measurement_surface": "real_stream_k_mmq",
        "surface_id": surface_identity["surface_id"],
        "wgm_arm": wgm_arm,
        "wgm_arm_label": arm_label,
        "device_id": claim["opened"]["device_id"],
        "device_claim_id": claim["opened"]["claim_id"],
    }
    prefix = f"mmq_wgm_arm_{wgm_arm}"
    measurements = [
        _measurement(
            measurement_id=f"{prefix}_end_to_end_wall_time_ms",
            metric="mmq_wgm_end_to_end_wall_time_ms",
            value=statistics.median(walls),
            unit="ms",
            direction="lower_better",
            category=category,
            reps=len(walls),
            reps_basis=wall_basis,
            claim=f"Median end-to-end wall time for real MMQ WGM arm {arm_label}",
            extra={
                **shared_extra,
                "measurement_role": "end_to_end_wall_time",
                "aggregation": "median",
                "minimum_ms": min(walls),
                "maximum_ms": max(walls),
                "evidence_sha256": wall_evidence["sha256"],
            },
        ),
        _measurement(
            measurement_id=f"{prefix}_all_mmq_tcc_hit_rate",
            metric="mmq_wgm_all_mmq_tcc_hit_rate",
            value=hit_rate,
            unit="fraction",
            direction="higher_better",
            category=category,
            reps=len(counters),
            reps_basis=counter_basis,
            claim=f"Pooled all-MMQ TCC hit rate for real MMQ WGM arm {arm_label}",
            extra={
                **shared_extra,
                "measurement_role": "all_mmq_tcc_hit_rate",
                "aggregation": "sum(hits)/(sum(hits)+sum(misses))",
                "tcc_hit_sum": total_hits,
                "tcc_miss_sum": total_misses,
                "all_mmq_dispatches": total_dispatches,
                "evidence_sha256": counter_evidence_ref["sha256"],
            },
        ),
        _measurement(
            measurement_id=f"{prefix}_all_mmq_read_requests_per_rep",
            metric="mmq_wgm_all_mmq_read_request_volume_per_rep",
            value=read_requests_per_rep,
            unit="requests/repetition",
            direction="lower_better",
            category=category,
            reps=len(counters),
            reps_basis=counter_basis,
            claim=(
                "Mean all-MMQ read-request volume per scored counter repetition "
                f"for real MMQ WGM arm {arm_label}"
            ),
            extra={
                **shared_extra,
                "measurement_role": "all_mmq_read_request_volume",
                "aggregation": "sum(read_requests)/scored_counter_repetitions",
                "read_request_sum": total_read_requests,
                "all_mmq_dispatches": total_dispatches,
                "evidence_sha256": counter_evidence_ref["sha256"],
            },
        ),
    ]
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "authority": AUTHORITY,
        "campaign_id": campaign,
        "started_at": start,
        "ended_at": end,
        "surface": surface_identity,
        "wgm_arm": {"value": wgm_arm, "label": arm_label},
        "source": source_identity,
        "producer": _producer_identity(),
        "device_claim": claim,
        "evidence": {
            "wall_time": wall_evidence,
            "all_mmq_counters": counter_evidence_ref,
        },
        "observations": {
            "wall_time_samples_ms": list(walls),
            "all_mmq_counter_samples": [
                {
                    "tcc_hits": float(row.tcc_hits),
                    "tcc_misses": float(row.tcc_misses),
                    "read_requests": float(row.read_requests),
                    "dispatches": row.dispatches,
                }
                for row in counters
            ],
        },
        "belief_measurements": measurements,
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    return receipt


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Validate and atomically admit a receipt built by this producer."""
    if not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA:
        raise WgmReceiptError("receipt has the wrong producer schema")
    producer = receipt.get("producer")
    if not isinstance(producer, Mapping) or dict(producer) != _producer_identity():
        raise WgmReceiptError("receipt was not built by this producer")
    expected = receipt_sha256(receipt)
    if receipt.get("receipt_sha256") != expected:
        raise WgmReceiptError("receipt_sha256 does not bind the admitted payload")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    rendered = json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n"
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
    "AUTHORITY", "PRODUCER_ID", "SCHEMA", "CounterSample", "WgmReceiptError",
    "build_receipt", "receipt_sha256", "write_receipt",
]
