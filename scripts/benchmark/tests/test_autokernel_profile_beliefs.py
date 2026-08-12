from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.benchmark import autokernel_profile_beliefs as P


SHA = "1" * 64


def claim(campaign: str) -> tuple[dict, dict]:
    opened = {
        "schema": P.CLAIM_SCHEMA, "claim_id": "akd-1234", "device_id": "mi210_0",
        "campaign_id": campaign,
    }
    released = {**opened, "released_at": "2026-08-11T20:00:01Z"}
    return opened, released


def g15() -> dict:
    campaign = "g15-test"
    opened, released = claim(campaign)
    profile = {
        "parallel": 64,
        "bench": {"speed_tg": 168.25},
        "attribution": {"elementwise_norm_target_share": 0.018},
        "hypothesis": {"observed_target_share": 0.018,
                       "verdict": "FALSIFIED_PROFILE_TARGET"},
    }
    return {
        "schema": P.G15_SCHEMA, "status": "passed", "campaign_id": campaign,
        "started_at": "2026-08-11T20:00:00Z", "ended_at": "2026-08-11T20:00:02Z",
        "device_claim_open": opened, "device_claim_released": released,
        "profiles": [profile],
    }


def c4() -> tuple[dict, dict, str]:
    campaign = "c4-test"
    opened, released = claim(campaign)
    report = {
        "schema": "epyc.autokernel.c4_profile_report.v1",
        "kernel_table": [
            {"kernel_family": "mul_mat_vec_q", "duration_ns": 500.0,
             "gpu_time_share": 0.5},
            {"kernel_family": "quantize_q8_1", "duration_ns": 250.0,
             "gpu_time_share": 0.25},
        ],
    }
    digest = hashlib.sha256(
        (json.dumps(report, sort_keys=True) + "\n").encode()).hexdigest()
    receipt = {
        "schema": P.C4_SCHEMA, "campaign_id": campaign,
        "started_at": "2026-08-11T20:00:00Z", "ended_at": "2026-08-11T20:00:02Z",
        "device_claim_open": opened, "device_claim_released": released,
        "report_sha256": digest,
        "formal": {"active_steps": 5, "receipt": {"workload_id": "q4-k"}},
    }
    return receipt, report, digest


def wgm() -> dict:
    campaign = "wgm-test"
    opened, released = claim(campaign)
    return {
        "schema": P.WGM_SCHEMA, "status": "pass", "campaign_id": campaign,
        "started_at": "2026-08-11T20:00:00Z", "ended_at": "2026-08-11T20:00:02Z",
        "surface": "standalone_l2_tile_reuse_proxy_not_mmq",
        "device_claim": {"opened": opened, "released": released},
        "result": {"factors": {
            "0": {"median_ms": 1.2, "sample_count": 48},
            "16": {"median_ms": 1.1, "sample_count": 48},
        }},
    }


def test_g15_emits_performance_and_target_selection_rows() -> None:
    value = P.finalize_g15(g15(), source_locator="/e/g15.json", source_sha256=SHA)
    assert value["status"] == "passed"
    assert [row["metric_direction"] for row in value["belief_measurements"]] == [
        "higher_better", "higher_better"]
    assert value["belief_measurements"][1]["extra"]["target_selection_only"] is True
    assert value["receipt_sha256"] == P.receipt_sha256(value)


def test_c4_emits_formal_per_suite_duration_rows() -> None:
    receipt, report, digest = c4()
    value = P.finalize_c4(
        receipt, report, source_locator="/e/c4.json", source_sha256=SHA,
        report_sha256=digest)
    rows = value["belief_measurements"]
    assert [row["value"] for row in rows] == [100.0, 50.0]
    assert all(row["metric_direction"] == "lower_better" for row in rows)
    assert rows[0]["extra"]["formal_gpu_time_share"] == 0.5


def test_wgm_proxy_preserves_nontransfer_boundary() -> None:
    value = P.finalize_wgm_proxy(
        wgm(), source_locator="/e/wgm.json", source_sha256=SHA)
    assert len(value["belief_measurements"]) == 2
    assert value["belief_measurements"][1]["extra"]["does_not_transfer_to_real_mmq"] is True
    assert value["belief_measurements"][0]["reps"] == 48


def test_failed_source_is_refused() -> None:
    source = g15()
    source["status"] = "failed"
    with pytest.raises(P.ProfileBeliefError, match="failed or incomplete"):
        P.finalize_g15(source, source_locator="/e/g15.json", source_sha256=SHA)


def test_claim_mismatch_is_refused() -> None:
    source = g15()
    source["device_claim_released"]["claim_id"] = "akd-other"
    with pytest.raises(P.ProfileBeliefError, match="claim_id differs"):
        P.finalize_g15(source, source_locator="/e/g15.json", source_sha256=SHA)


def test_c4_report_hash_mismatch_is_refused() -> None:
    receipt, report, digest = c4()
    with pytest.raises(P.ProfileBeliefError, match="report hash differs"):
        P.finalize_c4(
            receipt, report, source_locator="/e/c4.json", source_sha256=SHA,
            report_sha256="2" * 64)


def test_wgm_proxy_cannot_be_relabelled_as_real_mmq() -> None:
    source = wgm()
    source["surface"] = "real_stream_k_mmq"
    with pytest.raises(P.ProfileBeliefError, match="standalone proxy boundary"):
        P.finalize_wgm_proxy(source, source_locator="/e/wgm.json", source_sha256=SHA)


def test_write_receipt_refuses_overwrite_and_tamper(tmp_path) -> None:
    value = P.finalize_g15(g15(), source_locator="/e/g15.json", source_sha256=SHA)
    target = tmp_path / "receipt.json"
    P.write_receipt(target, value)
    with pytest.raises(P.ProfileBeliefError, match="overwrite"):
        P.write_receipt(target, value)
    tampered = copy.deepcopy(value)
    tampered["belief_measurements"][0]["value"] += 1
    with pytest.raises(P.ProfileBeliefError, match="self-hash"):
        P.write_receipt(tmp_path / "other.json", tampered)
