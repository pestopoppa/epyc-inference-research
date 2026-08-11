from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.benchmark import autokernel_p2_5j_receipt as P


def _artifact(tmp_path, name: str, payload: dict) -> dict[str, str]:
    path = tmp_path / name
    raw = json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(raw, encoding="utf-8")
    return {"locator": name, "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def _claim(schema: str, campaign: str, resource: str, claim_id: str) -> tuple[dict, dict]:
    base = {
        "schema": schema,
        "claim_id": claim_id,
        "campaign_id": campaign,
        "acquired_at": "2026-08-11T20:00:00Z",
    }
    if schema == P.CPU_CLAIM_SCHEMA:
        base |= {
            "cpu_list": resource,
            "regions": [next(spec["cpu_region"] for spec in P.ARM_SPECS.values()
                             if spec["cpu_list"] == resource)],
        }
    else:
        base["device_id"] = resource
    return base, {**base, "released_at": "2026-08-11T20:01:00Z"}


def campaign(tmp_path, *, local_factor: float = 1.03) -> dict:
    campaign_id = "p2-5j-test"
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    binary.write_bytes(b"binary")
    model.write_bytes(b"model")
    blocks = []
    for block in range(P.REQUIRED_BLOCKS):
        order = list(P.ARM_SPECS)
        order = order[block % 4:] + order[:block % 4]
        samples = []
        for arm in order:
            factor = {"I": 1.0, "H": 1.01, "Lp": local_factor, "Ls": 1.0}[arm]
            cpu_open, cpu_released = _claim(
                P.CPU_CLAIM_SCHEMA, campaign_id, P.ARM_SPECS[arm]["cpu_list"],
                f"cpu-{block}-{arm}")
            gpu_open, gpu_released = _claim(
                P.DEVICE_CLAIM_SCHEMA, campaign_id, "mi210_0", f"gpu-{block}-{arm}")
            throughput = (100.0 + block) * factor
            affinity_path = tmp_path / f"aff-{block}-{arm}.json"
            affinity = {
                "live_affinity_verified": True,
                "foreign_llama_overlaps": [],
                "foreign_allowed_overlaps": [],
                "instances": [{
                    "expected_cpus": P.ARM_SPECS[arm]["cpu_list"],
                    "observed_thread_union": P.ARM_SPECS[arm]["cpu_list"],
                    "match": True,
                }],
            }
            affinity_artifact = _artifact(
                tmp_path, affinity_path.name, affinity)
            run_row = {
                "protocol_id": "P-BENCH-3",
                "decision_grade": True,
                "live_affinity_verified": True,
                "cell_error": None,
                "error_rate": 0.0,
                "np": 8,
                "total_streams": 8,
                "ctx": 65536,
                "per_stream_ctx": 8192,
                "instances": [{"cpu_list": P.ARM_SPECS[arm]["cpu_list"],
                               "threads": 8}],
                "aggregate_decode_tps": throughput,
                "p50_latency_ms": 1000.0,
                "p95_latency_ms": 1200.0,
                "affinity_artifact": str(affinity_path.resolve()),
            }
            samples.append({
                "sample_id": f"sample-{block}-{arm}",
                "arm": arm,
                "attempt": 1,
                "valid": True,
                "decision_grade": True,
                "measurement_protocol": P.MEASUREMENT_PROTOCOL,
                "cpu_list": P.ARM_SPECS[arm]["cpu_list"],
                "aggregate_decode_tps": throughput,
                "p50_latency_ms": 1000.0,
                "p95_latency_ms": 1200.0,
                "artifacts": {
                    "run_receipt": _artifact(
                        tmp_path, f"run-{block}-{arm}.json", run_row),
                    "affinity_receipt": affinity_artifact,
                },
                "cpu_claim_open": cpu_open,
                "cpu_claim_released": cpu_released,
                "device_claim_open": gpu_open,
                "device_claim_released": gpu_released,
            })
        blocks.append({"block": block, "order": order, "samples": samples})
    return {
        "schema": P.INPUT_SCHEMA,
        "status": "completed",
        "protocol_id": P.PROTOCOL_ID,
        "campaign_id": campaign_id,
        "started_at": "2026-08-11T20:00:00Z",
        "ended_at": "2026-08-11T22:00:00Z",
        "identity": {
            "binary": {"path": str(binary), "version": "10125",
                       "sha256": hashlib.sha256(b"binary").hexdigest()},
            "model": {"path": str(model), "sha256": hashlib.sha256(b"model").hexdigest(),
                      "size_bytes": len(b"model")},
            "device": {"device_id": "mi210_0", "pci_bdf": "0000:43:00.0",
                       "numa_node": 1},
        },
        "shape": {"np_slots": 8, "slot_context_tokens": 8192,
                  "total_context_tokens": 65536, "mtp": False},
        "blocks": blocks,
    }


def test_full_campaign_emits_observation_signal_and_all_arms(tmp_path) -> None:
    value = P.finalize_campaign(campaign(tmp_path), base_dir=tmp_path)
    assert value["verdict"]["status"] == "observation_only_device_local_signal"
    assert value["verdict"]["observed_leader_arm"] == "Lp"
    assert value["verdict"]["selected_arm"] == "I"
    assert value["verdict"]["device_local_move_authorized"] is False
    assert value["verdict"]["kernel_speedup_claim"] is False
    assert set(value["arm_summaries"]) == set(P.ARM_SPECS)
    assert value["arm_summaries"]["Lp"]["n"] == 10
    assert value["receipt_sha256"] == P.receipt_sha256(value)


def test_below_threshold_is_no_demonstrated_win(tmp_path) -> None:
    value = P.finalize_campaign(campaign(tmp_path, local_factor=1.019), base_dir=tmp_path)
    assert value["verdict"]["status"] == "no_demonstrated_device_local_signal"
    assert value["verdict"]["selected_arm"] == "I"
    assert value["verdict"]["device_local_move_authorized"] is False


def test_missing_block_is_refused(tmp_path) -> None:
    source = campaign(tmp_path)
    source["blocks"].pop()
    with pytest.raises(P.PlacementReceiptError, match="exactly 10"):
        P.finalize_campaign(source, base_dir=tmp_path)


def test_wrong_cpu_list_is_refused(tmp_path) -> None:
    source = campaign(tmp_path)
    source["blocks"][0]["samples"][0]["cpu_list"] = "0-7"
    with pytest.raises(P.PlacementReceiptError, match="cpu_list must be"):
        P.finalize_campaign(source, base_dir=tmp_path)


def test_unreleased_claim_is_refused(tmp_path) -> None:
    source = campaign(tmp_path)
    source["blocks"][0]["samples"][0]["device_claim_released"].pop("released_at")
    with pytest.raises(P.PlacementReceiptError, match="lacks released_at"):
        P.finalize_campaign(source, base_dir=tmp_path)


def test_artifact_hash_mismatch_is_refused(tmp_path) -> None:
    source = campaign(tmp_path)
    source["blocks"][0]["samples"][0]["artifacts"]["run_receipt"]["sha256"] = "9" * 64
    with pytest.raises(P.PlacementReceiptError, match="hash mismatch"):
        P.finalize_campaign(source, base_dir=tmp_path)


def test_write_receipt_refuses_overwrite_and_tamper(tmp_path) -> None:
    value = P.finalize_campaign(campaign(tmp_path), base_dir=tmp_path)
    target = tmp_path / "receipt.json"
    P.write_receipt(target, value)
    with pytest.raises(P.PlacementReceiptError, match="overwrite"):
        P.write_receipt(target, value)
    tampered = copy.deepcopy(value)
    tampered["verdict"]["selected_arm"] = "Ls"
    with pytest.raises(P.PlacementReceiptError, match="self-hash"):
        P.write_receipt(tmp_path / "other.json", tampered)
