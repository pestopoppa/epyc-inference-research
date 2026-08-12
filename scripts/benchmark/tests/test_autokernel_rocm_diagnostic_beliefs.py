"""SC28 prospective ROCm diagnostic belief-capture contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.benchmark import autokernel_rocm_diagnostic_beliefs as beliefs
from scripts.kernel_rnd.autokernel import schemas


def _claim(*, released: bool) -> dict:
    return {
        "schema": "epyc.autokernel.device_claim_receipt.v1",
        "claim_id": "akd-fixture", "device_id": "mi210_0",
        "campaign_id": "ak-fixture", "acquired_at": "2026-08-12T06:00:00Z",
        # ClaimReceipt retains the pre-release state; released_at is the
        # canonical proof that Claim.release() completed and journaled.
        "state": "held",
        "released_at": "2026-08-12T06:01:00Z" if released else None,
    }


def _sampling() -> dict:
    rows = [
        {"offset_s": 0.0, "power_w": 190.0, "sclk_mhz": 1700.0},
        {"offset_s": 0.25, "power_w": 196.0, "sclk_mhz": 1700.0},
        {"offset_s": 0.50, "power_w": 194.0, "sclk_mhz": 1600.0},
    ]
    value = {
        "schema": "epyc.autokernel.device_sampling_receipt.v1",
        "sample_count": len(rows), "samples": rows,
    }
    value["sha256"] = schemas.content_hash(value)
    return value


def saturation_receipt() -> dict:
    return {
        "schema": beliefs.SATURATION_SCHEMA, "campaign_id": "ak-fixture",
        "started_at": "2026-08-12T06:00:00Z", "ended_at": "2026-08-12T06:01:00Z",
        "workload": {
            "schema": "epyc.rocm_gemm_saturation.v1", "arch": "gfx90a:sramecc+",
            "m": 8192, "n": 8192, "k": 8192, "iterations": 64,
            "throughput_tflops": 41.75,
        },
        "workload_source": "/source/rocm_gemm_saturation.cpp",
        "workload_source_sha256": "a" * 64,
        "workload_binary": "/bin/rocm_gemm_saturation",
        "workload_binary_sha256": "b" * 64,
        "device_claim_open": _claim(released=False),
        "device_claim_released": _claim(released=True),
        "device_sampling": _sampling(),
        "power_cap_w": 300.0, "max_power_w": 196.0,
        "nominal_sclk_mhz": 1700.0, "nominal_sclk_sample_fraction": 2 / 3,
    }


def vendor_receipt() -> dict:
    raw = []
    comparisons = []
    for m, n, k, roc, hip in (
            (896, 128, 896, 10.0, 12.0), (4864, 128, 896, 20.0, 18.0)):
        raw.extend((
            {"schema": "epyc.rocm.gemm_baseline.v1", "library": "rocblas",
             "dtype": "fp16_compute_fp32", "m": m, "n": n, "k": k,
             "repetitions": 30, "tflops": roc},
            {"schema": "epyc.rocm.gemm_baseline.v1", "library": "hipblaslt",
             "dtype": "fp16_compute_fp32", "m": m, "n": n, "k": k,
             "repetitions": 30, "tflops": hip},
        ))
        comparisons.append({
            "m": m, "n": n, "k": k, "rocblas_tflops": roc,
            "hipblaslt_tflops": hip, "hipblaslt_over_rocblas": hip / roc,
        })
    return {
        "schema": beliefs.VENDOR_SCHEMA, "campaign_id": "ak-fixture",
        "started_at": "2026-08-12T06:00:00Z", "ended_at": "2026-08-12T06:01:00Z",
        "comparator_source": "/source/rocm_gemm_baseline_compare.cpp",
        "comparator_source_sha256": "c" * 64,
        "comparator_binary": "/bin/rocm_gemm_baseline_compare",
        "comparator_binary_sha256": "d" * 64,
        "device_claim_open": _claim(released=False),
        "device_claim_released": _claim(released=True),
        "device_sampling": _sampling(),
        "metadata": {"schema": "epyc.rocm.gemm_baseline.meta.v1", "shape_count": 2},
        "raw_results": raw, "comparisons": comparisons,
    }


def test_saturation_writes_four_directional_self_hashed_rows() -> None:
    value = beliefs.attach_beliefs(
        saturation_receipt(), producer_path=Path(beliefs.__file__).with_name(
            "run_rocm_saturation_probe.py"))
    rows = value["belief_measurements"]
    assert [row["metric_direction"] for row in rows] == [
        "higher_better", "higher_better", "lower_better", "higher_better"]
    assert [row["reps"] for row in rows] == [64, 3, 3, 3]
    assert rows[-1]["value"] == 104.0
    assert all(row["extra"]["grants_campaign_authority"] is False for row in rows)
    for row in rows:
        unsigned = dict(row)
        stored = unsigned.pop("measurement_sha256")
        assert stored == schemas.content_hash(unsigned)
    unsigned_receipt = dict(value)
    stored_receipt = unsigned_receipt.pop("receipt_sha256")
    assert stored_receipt == schemas.content_hash(unsigned_receipt)


def test_vendor_rows_are_exact_shape_only_and_use_scored_provider_repetitions() -> None:
    value = beliefs.attach_beliefs(
        vendor_receipt(), producer_path=Path(beliefs.__file__).with_name(
            "run_rocm_gemm_baseline_compare.py"))
    rows = value["belief_measurements"]
    assert len(rows) == 2
    assert {row["value"] for row in rows} == {1.2, 0.9}
    assert all(row["reps"] == 30 for row in rows)
    assert {row["extra"]["stronger_provider"] for row in rows} == {
        "hipblaslt", "rocblas"}
    assert all(row["extra"]["exact_shape_only"] is True for row in rows)
    assert all(row["extra"]["global_provider_selection"] is False for row in rows)


def test_pre_hook_receipt_is_not_mutated_and_capture_is_write_once() -> None:
    source = saturation_receipt()
    original = copy.deepcopy(source)
    value = beliefs.attach_beliefs(
        source, producer_path=Path(beliefs.__file__).with_name(
            "run_rocm_saturation_probe.py"))
    assert source == original
    with pytest.raises(ValueError, match="write-once"):
        beliefs.attach_beliefs(
            value, producer_path=Path(beliefs.__file__).with_name(
                "run_rocm_saturation_probe.py"))


def test_claim_without_release_timestamp_refuses() -> None:
    value = saturation_receipt()
    value["device_claim_released"]["released_at"] = None
    with pytest.raises(ValueError, match="durably released"):
        beliefs.attach_beliefs(
            value, producer_path=Path(beliefs.__file__).with_name(
                "run_rocm_saturation_probe.py"))


@pytest.mark.parametrize("defect", ["claim", "sampling", "ratio", "reps"])
def test_incomplete_or_non_rederivable_inputs_refuse(defect: str) -> None:
    if defect in {"claim", "sampling"}:
        value = saturation_receipt()
        if defect == "claim":
            value["device_claim_released"]["claim_id"] = "other"
        else:
            value["device_sampling"]["sample_count"] += 1
        producer = "run_rocm_saturation_probe.py"
    else:
        value = vendor_receipt()
        if defect == "ratio":
            value["comparisons"][0]["hipblaslt_over_rocblas"] = 99.0
        else:
            value["raw_results"][1]["repetitions"] = 29
        producer = "run_rocm_gemm_baseline_compare.py"
    with pytest.raises(ValueError):
        beliefs.attach_beliefs(
            value, producer_path=Path(beliefs.__file__).with_name(producer))
