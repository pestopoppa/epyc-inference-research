"""Prospective ClaimTuple-shaped rows for AutoKernel ROCm diagnostics.

The 2026-08-12 RVP-T0-1 and AK-BH-1 receipts predate this module and remain
unprojected.  The two live runners call :func:`attach_beliefs` before their
atomic write, while the source material and device-claim receipts are still
available.  Every row is self-hashed and the enclosing receipt is self-hashed;
the root read side independently re-derives both rather than trusting them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.kernel_rnd.autokernel import schemas


SATURATION_SCHEMA = "epyc.rvp_t0_1_saturation_probe.v1"
VENDOR_SCHEMA = "epyc.ak_bh_1_gemm_baseline_compare.v1"
SATURATION_PRODUCER_ID = "scripts.benchmark.run_rocm_saturation_probe/v2"
VENDOR_PRODUCER_ID = "scripts.benchmark.run_rocm_gemm_baseline_compare/v2"
SATURATION_PRODUCER_PATH = "scripts/benchmark/run_rocm_saturation_probe.py"
VENDOR_PRODUCER_PATH = "scripts/benchmark/run_rocm_gemm_baseline_compare.py"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{label} must be a {qualifier}finite number")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _claim_identity(receipt: Mapping[str, Any]) -> tuple[dict, str]:
    opened = receipt.get("device_claim_open")
    released = receipt.get("device_claim_released")
    if not isinstance(opened, dict) or not isinstance(released, dict):
        raise ValueError("diagnostic receipt needs opened and released device claims")
    for key in ("claim_id", "device_id", "campaign_id", "acquired_at"):
        if not isinstance(opened.get(key), str) or not opened[key]:
            raise ValueError(f"device_claim_open.{key} must be non-empty text")
        if released.get(key) != opened[key]:
            raise ValueError(f"device claim {key} changed across release")
    if released.get("state") != "released" or not released.get("released_at"):
        raise ValueError("device claim must be durably released before belief capture")
    identity = {"opened": opened, "released": released}
    return identity, schemas.content_hash(identity)


def _producer(*, producer_id: str, producer_path: str, path: Path) -> dict:
    return {
        "producer_id": producer_id,
        "path": producer_path,
        "sha256": _sha256_file(path),
    }


def _measurement(*, measurement_id: str, metric: str, value: float, unit: str,
                 direction: str, reps: int, reps_basis: str, claim: str,
                 extra: Mapping[str, Any]) -> dict:
    row = {
        "measurement_id": measurement_id,
        "metric": metric,
        "value": value,
        "unit": unit,
        "metric_direction": direction,
        "category": "BASELINE",
        "protocol_id": "",
        "reps": reps,
        "reps_basis": reps_basis,
        "claim": claim,
        "extra": dict(extra),
    }
    row["measurement_sha256"] = schemas.content_hash(row)
    return row


def _base(receipt: Mapping[str, Any], *, producer_id: str,
          producer_path: str, path: Path) -> tuple[dict, dict, str]:
    claim_identity, claim_sha = _claim_identity(receipt)
    producer = _producer(
        producer_id=producer_id, producer_path=producer_path, path=path)
    return producer, claim_identity, claim_sha


def _attach_saturation(receipt: dict, *, producer_path: Path) -> dict:
    workload = receipt.get("workload")
    sampling = receipt.get("device_sampling")
    if not isinstance(workload, dict) or workload.get("schema") != "epyc.rocm_gemm_saturation.v1":
        raise ValueError("saturation receipt lacks its admitted workload result")
    if not isinstance(sampling, dict) \
            or sampling.get("schema") != "epyc.autokernel.device_sampling_receipt.v1":
        raise ValueError("saturation receipt lacks its admitted device trace")
    iterations = _positive_int(workload.get("iterations"), "workload.iterations")
    sample_count = _positive_int(sampling.get("sample_count"), "device_sampling.sample_count")
    samples = sampling.get("samples")
    if not isinstance(samples, list) or len(samples) != sample_count:
        raise ValueError("device sampling count does not bind its sample vector")
    unsigned_sampling = dict(sampling)
    sampling_sha = unsigned_sampling.pop("sha256", None)
    if _sha(sampling_sha, "device_sampling.sha256") != schemas.content_hash(unsigned_sampling):
        raise ValueError("device sampling sha256 does not bind its trace")
    throughput = _finite(workload.get("throughput_tflops"), "throughput_tflops", positive=True)
    nominal_sclk = _finite(receipt.get("nominal_sclk_mhz"), "nominal_sclk_mhz", positive=True)
    sample_sclks = [
        _finite(item.get("sclk_mhz") if isinstance(item, dict) else None,
                f"device_sampling.samples[{index}].sclk_mhz", positive=True)
        for index, item in enumerate(samples)]
    sample_powers = [
        _finite(item.get("power_w") if isinstance(item, dict) else None,
                f"device_sampling.samples[{index}].power_w", positive=True)
        for index, item in enumerate(samples)]
    nominal_fraction = sum(value >= nominal_sclk for value in sample_sclks) / sample_count
    if not math.isclose(
            nominal_fraction,
            _finite(receipt.get("nominal_sclk_sample_fraction"),
                    "nominal_sclk_sample_fraction"), rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("nominal clock fraction does not re-derive from device samples")
    if nominal_fraction < 0 or nominal_fraction > 1:
        raise ValueError("nominal_sclk_sample_fraction must be in [0, 1]")
    max_power = max(sample_powers)
    if max_power != _finite(receipt.get("max_power_w"), "max_power_w", positive=True):
        raise ValueError("max_power_w does not re-derive from device samples")
    power_cap = _finite(receipt.get("power_cap_w"), "power_cap_w", positive=True)
    headroom = power_cap - max_power
    source = {
        "path": _text(receipt.get("workload_source"), "workload_source"),
        "sha256": _sha(receipt.get("workload_source_sha256"), "workload_source_sha256"),
    }
    binary = {
        "path": _text(receipt.get("workload_binary"), "workload_binary"),
        "sha256": _sha(receipt.get("workload_binary_sha256"), "workload_binary_sha256"),
    }
    producer, claim_identity, claim_sha = _base(
        receipt, producer_id=SATURATION_PRODUCER_ID,
        producer_path=SATURATION_PRODUCER_PATH, path=producer_path)
    evidence = {
        "workload": workload,
        "device_sampling_sha256": sampling_sha,
        "source_identity": source,
        "binary_identity": binary,
        "device_claim_identity_sha256": claim_sha,
        "power_cap_w": power_cap,
        "nominal_sclk_mhz": nominal_sclk,
        "producer_sha256": producer["sha256"],
    }
    evidence_sha = schemas.content_hash(evidence)
    common = {
        "campaign_id": receipt.get("campaign_id"),
        "source_identity": source,
        "binary_identity": binary,
        "device_claim_identity": claim_identity,
        "device_claim_identity_sha256": claim_sha,
        "producer_id": producer["producer_id"],
        "producer_sha256": producer["sha256"],
        "evidence_basis": evidence,
        "evidence_sha256": evidence_sha,
        "diagnostic_only": True,
        "grants_campaign_authority": False,
    }
    rows = [
        _measurement(
            measurement_id="rvp_t0_1_sustained_gemm_throughput_tflops",
            metric="gfx90a_sustained_gemm_throughput_tflops", value=throughput,
            unit="TFLOP/s", direction="higher_better", reps=iterations,
            reps_basis="scored:completed GEMM iterations",
            claim=f"RVP-T0-1 observed sustained GEMM throughput {throughput:.9g} TFLOP/s",
            extra=common),
        _measurement(
            measurement_id="rvp_t0_1_nominal_sclk_hold_fraction",
            metric="gfx90a_nominal_sclk_hold_fraction", value=nominal_fraction,
            unit="fraction", direction="higher_better", reps=sample_count,
            reps_basis="scored:in-window device-state samples",
            claim=f"RVP-T0-1 observed nominal-clock hold fraction {nominal_fraction:.9g}",
            extra=common),
        _measurement(
            measurement_id="rvp_t0_1_peak_power_w",
            metric="gfx90a_peak_power_w", value=max_power, unit="W",
            direction="lower_better", reps=sample_count,
            reps_basis="scored:in-window device-state samples",
            claim=f"RVP-T0-1 observed peak board power {max_power:.9g} W",
            extra=common),
        _measurement(
            measurement_id="rvp_t0_1_power_headroom_w",
            metric="gfx90a_power_cap_headroom_w", value=headroom, unit="W",
            direction="higher_better", reps=sample_count,
            reps_basis="scored:in-window device-state samples",
            claim=f"RVP-T0-1 observed {headroom:.9g} W headroom to the declared cap",
            extra=common),
    ]
    for row in rows:
        row["protocol_id"] = SATURATION_SCHEMA
        row["measurement_sha256"] = schemas.content_hash(
            {key: value for key, value in row.items() if key != "measurement_sha256"})
    result = dict(receipt)
    result["status"] = "complete"
    result["producer"] = producer
    result["source_identity"] = source
    result["binary_identity"] = binary
    result["device_claim_identity_sha256"] = claim_sha
    result["belief_measurements"] = rows
    result["receipt_sha256"] = schemas.content_hash(result)
    return result


def _attach_vendor(receipt: dict, *, producer_path: Path) -> dict:
    metadata = receipt.get("metadata")
    comparisons = receipt.get("comparisons")
    raw = receipt.get("raw_results")
    if not isinstance(metadata, dict) or metadata.get("schema") != "epyc.rocm.gemm_baseline.meta.v1":
        raise ValueError("vendor receipt lacks comparator metadata")
    if not isinstance(comparisons, list) or not comparisons \
            or not isinstance(raw, list) or len(raw) != 2 * len(comparisons):
        raise ValueError("vendor receipt lacks exact paired shape evidence")
    if metadata.get("shape_count") != len(comparisons):
        raise ValueError("vendor metadata shape_count does not bind comparisons")
    source = {
        "path": _text(receipt.get("comparator_source"), "comparator_source"),
        "sha256": _sha(receipt.get("comparator_source_sha256"), "comparator_source_sha256"),
    }
    binary = {
        "path": _text(receipt.get("comparator_binary"), "comparator_binary"),
        "sha256": _sha(receipt.get("comparator_binary_sha256"), "comparator_binary_sha256"),
    }
    producer, claim_identity, claim_sha = _base(
        receipt, producer_id=VENDOR_PRODUCER_ID,
        producer_path=VENDOR_PRODUCER_PATH, path=producer_path)
    raw_by_shape: dict[tuple[int, int, int], dict[str, dict]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("vendor raw result must be an object")
        key = (item.get("m"), item.get("n"), item.get("k"))
        raw_by_shape.setdefault(key, {})[item.get("library")] = item
    rows = []
    for item in sorted(comparisons, key=lambda row: (row["m"], row["n"], row["k"])):
        m, n, k = (_positive_int(item.get(axis), f"comparison.{axis}")
                   for axis in ("m", "n", "k"))
        pair = raw_by_shape.get((m, n, k))
        if not isinstance(pair, dict) or set(pair) != {"rocblas", "hipblaslt"}:
            raise ValueError("vendor comparison does not bind one row per provider")
        repetitions = _positive_int(pair["rocblas"].get("repetitions"), "repetitions")
        if pair["hipblaslt"].get("repetitions") != repetitions:
            raise ValueError("vendor provider repetitions differ within an exact shape")
        rocblas = _finite(pair["rocblas"].get("tflops"), "rocblas.tflops", positive=True)
        hipblaslt = _finite(pair["hipblaslt"].get("tflops"), "hipblaslt.tflops", positive=True)
        if (rocblas != _finite(item.get("rocblas_tflops"), "rocblas_tflops", positive=True)
                or hipblaslt != _finite(
                    item.get("hipblaslt_tflops"), "hipblaslt_tflops", positive=True)):
            raise ValueError("vendor comparison throughput does not re-derive from provider rows")
        ratio = hipblaslt / rocblas
        if not math.isclose(
                ratio, _finite(item.get("hipblaslt_over_rocblas"), "provider ratio"),
                rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("vendor provider ratio does not re-derive from exact-shape arms")
        best = "hipblaslt" if ratio > 1 else "rocblas"
        evidence = {
            "shape": {"m": m, "n": n, "k": k, "dtype": pair["rocblas"].get("dtype")},
            "provider_rows": pair,
            "source_identity": source,
            "binary_identity": binary,
            "device_claim_identity_sha256": claim_sha,
            "producer_sha256": producer["sha256"],
        }
        evidence_sha = schemas.content_hash(evidence)
        rows.append(_measurement(
            measurement_id=f"ak_bh_1_m{m}_n{n}_k{k}_hipblaslt_over_rocblas",
            metric="hipblaslt_over_rocblas_exact_shape_throughput_ratio",
            value=ratio, unit="ratio", direction="higher_better",
            reps=repetitions,
            reps_basis="scored:timed repetitions per provider at exact shape",
            claim=(f"AK-BH-1 shape m={m},n={n},k={k} observed hipBLASLt/rocBLAS "
                   f"throughput ratio {ratio:.9g}; stronger provider {best}"),
            extra={
                "campaign_id": receipt.get("campaign_id"),
                "shape": evidence["shape"],
                "rocblas_tflops": rocblas,
                "hipblaslt_tflops": hipblaslt,
                "stronger_provider": best,
                "source_identity": source,
                "binary_identity": binary,
                "device_claim_identity": claim_identity,
                "device_claim_identity_sha256": claim_sha,
                "producer_id": producer["producer_id"],
                "producer_sha256": producer["sha256"],
                "evidence_basis": evidence,
                "evidence_sha256": evidence_sha,
                "exact_shape_only": True,
                "global_provider_selection": False,
                "grants_campaign_authority": False,
            }))
    for row in rows:
        row["protocol_id"] = VENDOR_SCHEMA
        row["measurement_sha256"] = schemas.content_hash(
            {key: value for key, value in row.items() if key != "measurement_sha256"})
    result = dict(receipt)
    result["status"] = "complete"
    result["producer"] = producer
    result["source_identity"] = source
    result["binary_identity"] = binary
    result["device_claim_identity_sha256"] = claim_sha
    result["belief_measurements"] = rows
    result["receipt_sha256"] = schemas.content_hash(result)
    return result


def attach_beliefs(receipt: Mapping[str, Any], *, producer_path: Path) -> dict:
    """Attach prospective rows exactly once to one admitted diagnostic receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError("diagnostic receipt must be a mapping")
    if "belief_measurements" in receipt or "receipt_sha256" in receipt:
        raise ValueError("diagnostic belief capture is write-once")
    value = json.loads(schemas.canonical_json(dict(receipt)))
    if value.get("schema") == SATURATION_SCHEMA:
        return _attach_saturation(value, producer_path=producer_path)
    if value.get("schema") == VENDOR_SCHEMA:
        return _attach_vendor(value, producer_path=producer_path)
    raise ValueError(f"unsupported diagnostic receipt schema {value.get('schema')!r}")


__all__ = [
    "SATURATION_PRODUCER_ID", "SATURATION_PRODUCER_PATH", "SATURATION_SCHEMA",
    "VENDOR_PRODUCER_ID", "VENDOR_PRODUCER_PATH", "VENDOR_SCHEMA",
    "attach_beliefs",
]
