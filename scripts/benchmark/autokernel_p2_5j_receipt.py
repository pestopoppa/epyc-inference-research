#!/usr/bin/env python3
"""Finalize a four-arm P2-5j placement campaign into immutable evidence.

The server-native sweep owns raw request and affinity artifacts.  This module
does not run inference.  It verifies the campaign record, recomputes the paired
statistics and emits the narrow receipt that AutoKernel may consume as host
topology context.  A placement result is never a kernel speedup, carve grant or
production activation grant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping


INPUT_SCHEMA = "epyc.autokernel.p2_5j_campaign_record.v1"
SCHEMA = "epyc.autokernel.p2_5j_placement_receipt.v1"
PRODUCER_ID = "scripts.benchmark.autokernel_p2_5j_receipt/v1"
PRODUCER_PATH = "scripts/benchmark/autokernel_p2_5j_receipt.py"
CPU_CLAIM_SCHEMA = "epyc.autokernel.cpu_region_claim_receipt.v1"
DEVICE_CLAIM_SCHEMA = "epyc.autokernel.device_claim_receipt.v1"
PROTOCOL_ID = "P2-5j"
MEASUREMENT_PROTOCOL = "P-GPU-1"
AUTHORITY = "observation_only_placement_context_no_selection_speedup_carve_or_activation"
REQUIRED_BLOCKS = 10
PRACTICAL_THRESHOLD = 0.02
ARM_SPECS = {
    "I": {"cpu_list": "184-191", "cpu_region": "q3", "numa_node": 3,
          "relation": "cross_node", "role": "incumbent"},
    "H": {"cpu_list": "88-95", "cpu_region": "q3", "numa_node": 3,
          "relation": "cross_node", "role": "historical_physical"},
    "Lp": {"cpu_list": "40-47", "cpu_region": "q1", "numa_node": 1,
           "relation": "device_local", "role": "local_physical"},
    "Ls": {"cpu_list": "136-143", "cpu_region": "q1", "numa_node": 1,
           "relation": "device_local", "role": "local_smt"},
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PlacementReceiptError(ValueError):
    """The campaign cannot safely become a P2-5j placement receipt."""


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def receipt_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _producer_identity() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "producer_id": PRODUCER_ID,
        "path": PRODUCER_PATH,
        "sha256": sha256_file(path),
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlacementReceiptError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlacementReceiptError(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    rendered = _text(value, label)
    if not _SHA256_RE.fullmatch(rendered):
        raise PlacementReceiptError(f"{label} must be a lowercase SHA-256")
    return rendered


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlacementReceiptError(f"{label} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or (positive and rendered <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise PlacementReceiptError(f"{label} must be {qualifier}")
    return rendered


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlacementReceiptError(f"{label} must be an integer >= {minimum}")
    return value


def _verify_artifact(value: Any, label: str, *, base_dir: Path) -> dict[str, str]:
    record = _mapping(value, label)
    locator = _text(record.get("locator"), f"{label}.locator")
    expected = _sha(record.get("sha256"), f"{label}.sha256")
    path = Path(locator)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise PlacementReceiptError(f"{label} does not exist: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise PlacementReceiptError(
            f"{label} hash mismatch: expected {expected}, observed {observed}")
    return {"locator": str(path), "sha256": observed}


def _load_json_artifact(artifact: Mapping[str, str], label: str) -> Mapping[str, Any]:
    path = Path(artifact["locator"])
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if len(lines) != 1:
            raise PlacementReceiptError(f"{label} must contain exactly one JSON row")
        return _mapping(json.loads(lines[0]), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlacementReceiptError(f"{label} is not valid UTF-8 JSON") from exc


def _same_number(left: Any, right: Any, label: str) -> float:
    observed = _number(left, f"{label}.artifact", positive=True)
    declared = _number(right, f"{label}.campaign_record", positive=True)
    if observed != declared:
        raise PlacementReceiptError(
            f"{label} differs between the run artifact and campaign record")
    return observed


def _verify_sample_artifacts(
    artifacts_value: Any, *, label: str, base_dir: Path, expected_cpu_list: str,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = _mapping(artifacts_value, f"{label}.artifacts")
    run_artifact = _verify_artifact(
        artifacts.get("run_receipt"), f"{label}.artifacts.run_receipt",
        base_dir=base_dir)
    affinity_artifact = _verify_artifact(
        artifacts.get("affinity_receipt"), f"{label}.artifacts.affinity_receipt",
        base_dir=base_dir)
    row = _load_json_artifact(run_artifact, f"{label}.run_receipt")
    affinity = _load_json_artifact(affinity_artifact, f"{label}.affinity_receipt")
    if row.get("protocol_id") != "P-BENCH-3":
        raise PlacementReceiptError(f"{label} run artifact must be P-BENCH-3")
    if row.get("decision_grade") is not True or row.get("live_affinity_verified") is not True:
        raise PlacementReceiptError(
            f"{label} run artifact must be decision-grade with live affinity verified")
    if row.get("cell_error") is not None or _number(
            row.get("error_rate"), f"{label}.run_receipt.error_rate") != 0.0:
        raise PlacementReceiptError(f"{label} run artifact contains a failed request/cell")
    expected_shape = {
        "np": 8, "total_streams": 8, "ctx": 65536, "per_stream_ctx": 8192,
    }
    for field, expected in expected_shape.items():
        if row.get(field) != expected:
            raise PlacementReceiptError(
                f"{label} run artifact {field} must be {expected}")
    instances = row.get("instances")
    if not isinstance(instances, list) or len(instances) != 1:
        raise PlacementReceiptError(f"{label} run artifact must have one server instance")
    instance = _mapping(instances[0], f"{label}.run_receipt.instances[0]")
    if instance.get("cpu_list") != expected_cpu_list or instance.get("threads") != 8:
        raise PlacementReceiptError(
            f"{label} run artifact must pin eight threads to {expected_cpu_list}")
    if affinity.get("live_affinity_verified") is not True:
        raise PlacementReceiptError(f"{label} affinity artifact is not live-verified")
    if affinity.get("foreign_llama_overlaps") or affinity.get("foreign_allowed_overlaps"):
        raise PlacementReceiptError(f"{label} affinity artifact reports foreign overlap")
    affinity_instances = affinity.get("instances")
    if not isinstance(affinity_instances, list) or len(affinity_instances) != 1:
        raise PlacementReceiptError(f"{label} affinity artifact must have one instance")
    affinity_instance = _mapping(
        affinity_instances[0], f"{label}.affinity_receipt.instances[0]")
    for field in ("expected_cpus", "observed_thread_union"):
        if affinity_instance.get(field) != expected_cpu_list:
            raise PlacementReceiptError(
                f"{label} affinity {field} must be {expected_cpu_list}")
    if affinity_instance.get("match") is not True:
        raise PlacementReceiptError(f"{label} affinity instance does not match")
    recorded_affinity = row.get("affinity_artifact")
    if not isinstance(recorded_affinity, str) or Path(recorded_affinity).resolve() != Path(
            affinity_artifact["locator"]):
        raise PlacementReceiptError(
            f"{label} run artifact does not point at the hash-bound affinity artifact")
    return {
        "run_receipt": run_artifact,
        "affinity_receipt": affinity_artifact,
        "aggregate_decode_tps": _same_number(
            row.get("aggregate_decode_tps"), sample.get("aggregate_decode_tps"),
            f"{label}.aggregate_decode_tps"),
        "p50_latency_ms": _same_number(
            row.get("p50_latency_ms"), sample.get("p50_latency_ms"),
            f"{label}.p50_latency_ms"),
        "p95_latency_ms": _same_number(
            row.get("p95_latency_ms"), sample.get("p95_latency_ms"),
            f"{label}.p95_latency_ms"),
    }


def _verify_claim_pair(
    opened_value: Any, released_value: Any, *, schema: str, campaign_id: str,
    label: str, expected_resource: str,
) -> dict[str, Any]:
    opened = _mapping(opened_value, f"{label}.opened")
    released = _mapping(released_value, f"{label}.released")
    for state, record in (("opened", opened), ("released", released)):
        if record.get("schema") != schema:
            raise PlacementReceiptError(f"{label}.{state} has the wrong schema")
        if record.get("campaign_id") != campaign_id:
            raise PlacementReceiptError(f"{label}.{state} names another campaign")
    for field in ("claim_id", "campaign_id", "acquired_at"):
        if opened.get(field) != released.get(field):
            raise PlacementReceiptError(f"{label} {field} changed across release")
    if not released.get("released_at"):
        raise PlacementReceiptError(f"{label}.released lacks released_at")
    resource_field = "cpu_list" if schema == CPU_CLAIM_SCHEMA else "device_id"
    if opened.get(resource_field) != expected_resource:
        raise PlacementReceiptError(
            f"{label}.opened {resource_field} must be {expected_resource!r}")
    if released.get(resource_field) != expected_resource:
        raise PlacementReceiptError(
            f"{label}.released {resource_field} must be {expected_resource!r}")
    if schema == CPU_CLAIM_SCHEMA:
        expected_region = next(
            spec["cpu_region"] for spec in ARM_SPECS.values()
            if spec["cpu_list"] == expected_resource)
        if list(opened.get("regions") or []) != [expected_region]:
            raise PlacementReceiptError(
                f"{label}.opened regions must be [{expected_region!r}]")
        if list(released.get("regions") or []) != [expected_region]:
            raise PlacementReceiptError(
                f"{label}.released regions must be [{expected_region!r}]")
    return {"opened": dict(opened), "released": dict(released)}


def _median_absolute_deviation(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def _belief_measurement(
    *, arm: str, metric_suffix: str, metric: str, value: float, unit: str,
    direction: str, values: list[float], claim: str, extra: Mapping[str, Any],
) -> dict[str, Any]:
    if direction not in {"higher_better", "lower_better"}:
        raise PlacementReceiptError("belief measurement direction is unsupported")
    if len(values) != REQUIRED_BLOCKS or any(
            not math.isfinite(item) for item in values):
        raise PlacementReceiptError(
            "belief measurement must bind one finite value per scored block")
    row = {
        "measurement_id": f"p2_5j_{arm.lower()}_{metric_suffix}",
        "metric": metric,
        "value": value,
        "unit": unit,
        "metric_direction": direction,
        "category": "BASELINE" if arm in {"I", "H"} else "CANDIDATE",
        "reps": REQUIRED_BLOCKS,
        "reps_basis": "scored:ten randomized complete four-arm placement blocks",
        "claim": claim,
        "extra": {
            **dict(extra),
            "block_values": values,
            "aggregation": "median",
        },
    }
    row["measurement_sha256"] = canonical_sha256(row)
    return row


def finalize_campaign(source: Mapping[str, Any], *, base_dir: str | Path) -> dict[str, Any]:
    """Verify and summarize one complete four-arm observation campaign.

    P2-5j varies CPU affinity, so the ratified P-BENCH-PLACEMENT-1 composite is
    controlling.  Its five arms, np=1 anchor and measured-locality requirements
    are absent from the historical P2-5j design.  This receipt therefore cannot
    select a placement even when the four-arm signal clears 2%.
    """
    if source.get("schema") != INPUT_SCHEMA:
        raise PlacementReceiptError(f"source.schema must be {INPUT_SCHEMA}")
    if source.get("status") != "completed":
        raise PlacementReceiptError("source campaign must be completed")
    if source.get("protocol_id") != PROTOCOL_ID:
        raise PlacementReceiptError(f"protocol_id must be {PROTOCOL_ID}")
    campaign_id = _text(source.get("campaign_id"), "campaign_id")
    started_at = _text(source.get("started_at"), "started_at")
    ended_at = _text(source.get("ended_at"), "ended_at")
    base = Path(base_dir).resolve()

    identity = _mapping(source.get("identity"), "identity")
    binary = _mapping(identity.get("binary"), "identity.binary")
    model = _mapping(identity.get("model"), "identity.model")
    device = _mapping(identity.get("device"), "identity.device")
    normalized_identity = {
        "binary": {
            "path": _text(binary.get("path"), "identity.binary.path"),
            "version": _text(binary.get("version"), "identity.binary.version"),
            "sha256": _sha(binary.get("sha256"), "identity.binary.sha256"),
        },
        "model": {
            "path": _text(model.get("path"), "identity.model.path"),
            "sha256": _sha(model.get("sha256"), "identity.model.sha256"),
            "size_bytes": _integer(model.get("size_bytes"),
                                   "identity.model.size_bytes", minimum=1),
        },
        "device": {
            "device_id": _text(device.get("device_id"), "identity.device.device_id"),
            "pci_bdf": _text(device.get("pci_bdf"), "identity.device.pci_bdf"),
            "numa_node": _integer(device.get("numa_node"),
                                  "identity.device.numa_node"),
        },
    }
    if normalized_identity["device"]["numa_node"] != 1:
        raise PlacementReceiptError("P2-5j device NUMA node must be 1")
    for kind in ("binary", "model"):
        path = Path(normalized_identity[kind]["path"]).resolve()
        if not path.is_file():
            raise PlacementReceiptError(f"identity.{kind}.path does not exist: {path}")
        observed = sha256_file(path)
        if observed != normalized_identity[kind]["sha256"]:
            raise PlacementReceiptError(
                f"identity.{kind}.sha256 differs from the bytes at {path}")
        normalized_identity[kind]["path"] = str(path)
    if Path(normalized_identity["model"]["path"]).stat().st_size != normalized_identity[
            "model"]["size_bytes"]:
        raise PlacementReceiptError("identity.model.size_bytes differs from the model bytes")

    shape = _mapping(source.get("shape"), "shape")
    normalized_shape = {
        "np_slots": _integer(shape.get("np_slots"), "shape.np_slots", minimum=1),
        "slot_context_tokens": _integer(
            shape.get("slot_context_tokens"), "shape.slot_context_tokens", minimum=1),
        "total_context_tokens": _integer(
            shape.get("total_context_tokens"), "shape.total_context_tokens", minimum=1),
        "mtp": shape.get("mtp"),
    }
    if normalized_shape != {
        "np_slots": 8, "slot_context_tokens": 8192,
        "total_context_tokens": 65536, "mtp": False,
    }:
        raise PlacementReceiptError("P2-5j shape must be np=8 x 8192 with MTP off")

    blocks = source.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != REQUIRED_BLOCKS:
        raise PlacementReceiptError(f"blocks must contain exactly {REQUIRED_BLOCKS} rows")
    seen_blocks: set[int] = set()
    arm_samples: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_SPECS}
    normalized_blocks = []
    for block_index, block_value in enumerate(blocks):
        block = _mapping(block_value, f"blocks[{block_index}]")
        number = _integer(block.get("block"), f"blocks[{block_index}].block")
        if number in seen_blocks:
            raise PlacementReceiptError("block numbers must be unique")
        seen_blocks.add(number)
        order = block.get("order")
        if not isinstance(order, list) or set(order) != set(ARM_SPECS) or len(order) != 4:
            raise PlacementReceiptError(
                f"blocks[{block_index}].order must be a permutation of {list(ARM_SPECS)}")
        samples = block.get("samples")
        if not isinstance(samples, list) or len(samples) != 4:
            raise PlacementReceiptError(f"blocks[{block_index}] must contain four samples")
        by_arm: dict[str, dict[str, Any]] = {}
        for sample_index, sample_value in enumerate(samples):
            label = f"blocks[{block_index}].samples[{sample_index}]"
            sample = _mapping(sample_value, label)
            arm = _text(sample.get("arm"), f"{label}.arm")
            if arm not in ARM_SPECS or arm in by_arm:
                raise PlacementReceiptError(f"{label}.arm is unknown or duplicated")
            if sample.get("valid") is not True or sample.get("decision_grade") is not True:
                raise PlacementReceiptError(f"{label} must be valid and decision-grade")
            if sample.get("measurement_protocol") != MEASUREMENT_PROTOCOL:
                raise PlacementReceiptError(
                    f"{label}.measurement_protocol must be {MEASUREMENT_PROTOCOL}")
            spec = ARM_SPECS[arm]
            if sample.get("cpu_list") != spec["cpu_list"]:
                raise PlacementReceiptError(
                    f"{label}.cpu_list must be {spec['cpu_list']!r}")
            verified_artifacts = _verify_sample_artifacts(
                sample.get("artifacts"), label=label, base_dir=base,
                expected_cpu_list=spec["cpu_list"], sample=sample)
            normalized = {
                "sample_id": _text(sample.get("sample_id"), f"{label}.sample_id"),
                "block": number,
                "arm": arm,
                "order_index": order.index(arm),
                "attempt": _integer(sample.get("attempt"), f"{label}.attempt", minimum=1),
                "cpu_list": spec["cpu_list"],
                "cpu_region": spec["cpu_region"],
                "numa_node": spec["numa_node"],
                "relation": spec["relation"],
                "aggregate_decode_tps": verified_artifacts["aggregate_decode_tps"],
                "p50_latency_ms": verified_artifacts["p50_latency_ms"],
                "p95_latency_ms": verified_artifacts["p95_latency_ms"],
                "artifacts": {
                    "run_receipt": verified_artifacts["run_receipt"],
                    "affinity_receipt": verified_artifacts["affinity_receipt"],
                },
                "cpu_claim": _verify_claim_pair(
                    sample.get("cpu_claim_open"), sample.get("cpu_claim_released"),
                    schema=CPU_CLAIM_SCHEMA, campaign_id=campaign_id,
                    label=f"{label}.cpu_claim", expected_resource=spec["cpu_list"]),
                "device_claim": _verify_claim_pair(
                    sample.get("device_claim_open"), sample.get("device_claim_released"),
                    schema=DEVICE_CLAIM_SCHEMA, campaign_id=campaign_id,
                    label=f"{label}.device_claim",
                    expected_resource=normalized_identity["device"]["device_id"]),
            }
            by_arm[arm] = normalized
            arm_samples[arm].append(normalized)
        normalized_blocks.append({
            "block": number,
            "order": list(order),
            "samples": [by_arm[arm] for arm in order],
        })
    if seen_blocks != set(range(REQUIRED_BLOCKS)):
        raise PlacementReceiptError(
            f"block numbers must be exactly 0..{REQUIRED_BLOCKS - 1}")
    sample_ids = [sample["sample_id"] for values in arm_samples.values() for sample in values]
    if len(sample_ids) != len(set(sample_ids)):
        raise PlacementReceiptError("sample_id values must be unique")

    arm_summaries: dict[str, dict[str, Any]] = {}
    by_block = {
        block["block"]: {sample["arm"]: sample for sample in block["samples"]}
        for block in normalized_blocks
    }
    incumbent = [by_block[index]["I"]["aggregate_decode_tps"]
                 for index in range(REQUIRED_BLOCKS)]
    for arm, spec in ARM_SPECS.items():
        samples = sorted(arm_samples[arm], key=lambda row: row["block"])
        values = [sample["aggregate_decode_tps"] for sample in samples]
        p50_values = [sample["p50_latency_ms"] for sample in samples]
        p95_values = [sample["p95_latency_ms"] for sample in samples]
        ratios = [values[index] / incumbent[index] for index in range(REQUIRED_BLOCKS)]
        arm_summaries[arm] = {
            **spec,
            "n": len(values),
            "median_decode_tps": statistics.median(values),
            "mad_decode_tps": _median_absolute_deviation(values),
            "median_p50_latency_ms": statistics.median(p50_values),
            "median_p95_latency_ms": statistics.median(p95_values),
            "paired_ratios_to_incumbent": ratios,
            "median_paired_ratio_to_incumbent": statistics.median(ratios),
            "samples": samples,
        }

    comparisons: dict[str, dict[str, Any]] = {}
    signals: list[str] = []
    historical_median = arm_summaries["H"]["median_decode_tps"]
    for arm in ("Lp", "Ls"):
        ratio = arm_summaries[arm]["median_paired_ratio_to_incumbent"]
        threshold_met = ratio >= 1.0 + PRACTICAL_THRESHOLD
        noninferior_to_h = arm_summaries[arm]["median_decode_tps"] >= historical_median
        qualifies = threshold_met and noninferior_to_h
        comparisons[f"{arm}_vs_I"] = {
            "local_arm": arm,
            "incumbent_arm": "I",
            "median_paired_ratio": ratio,
            "median_improvement_fraction": ratio - 1.0,
            "practical_threshold_fraction": PRACTICAL_THRESHOLD,
            "practical_threshold_met": threshold_met,
            "noninferior_to_H": noninferior_to_h,
            "qualifies_as_observation_only_device_local_signal": qualifies,
            "placement_selection_authority": False,
        }
        if qualifies:
            signals.append(arm)
    selected = max(
        signals,
        key=lambda arm: arm_summaries[arm]["median_paired_ratio_to_incumbent"],
        default=None,
    )
    verdict_status = (
        "observation_only_device_local_signal"
        if selected else "no_demonstrated_device_local_signal")
    belief_measurements: list[dict[str, Any]] = []
    for arm, summary in arm_summaries.items():
        samples = summary["samples"]
        common = {
            "measurement_surface": "p2_5j_four_arm_host_thread_placement",
            "arm": arm,
            "arm_role": summary["role"],
            "cpu_list": summary["cpu_list"],
            "cpu_region": summary["cpu_region"],
            "numa_node": summary["numa_node"],
            "relation": summary["relation"],
            "device_id": normalized_identity["device"]["device_id"],
            "shape": normalized_shape,
            "authority": AUTHORITY,
            "placement_selection_authority": False,
            "kernel_speedup_authority": False,
            "carve_authority": False,
            "production_activation_authority": False,
            "sample_ids": [sample["sample_id"] for sample in samples],
            "cpu_claim_ids": [sample["cpu_claim"]["opened"]["claim_id"]
                              for sample in samples],
            "device_claim_ids": [sample["device_claim"]["opened"]["claim_id"]
                                 for sample in samples],
        }
        decode_values = [sample["aggregate_decode_tps"] for sample in samples]
        p50_values = [sample["p50_latency_ms"] for sample in samples]
        p95_values = [sample["p95_latency_ms"] for sample in samples]
        ratio_values = list(summary["paired_ratios_to_incumbent"])
        belief_measurements.extend((
            _belief_measurement(
                arm=arm, metric_suffix="decode_tps",
                metric="aggregate_decode_tokens_per_second",
                value=summary["median_decode_tps"], unit="tokens/s",
                direction="higher_better", values=decode_values,
                claim=(f"P2-5j arm {arm} observed median aggregate decode throughput "
                       f"{summary['median_decode_tps']:.9g} tokens/s; observation only"),
                extra={**common, "measurement_role": "placement_observation"},
            ),
            _belief_measurement(
                arm=arm, metric_suffix="p50_latency_ms",
                metric="request_latency_p50_ms",
                value=summary["median_p50_latency_ms"], unit="ms",
                direction="lower_better", values=p50_values,
                claim=(f"P2-5j arm {arm} observed median p50 request latency "
                       f"{summary['median_p50_latency_ms']:.9g} ms; observation only"),
                extra={**common, "measurement_role": "placement_observation"},
            ),
            _belief_measurement(
                arm=arm, metric_suffix="p95_latency_ms",
                metric="request_latency_p95_ms",
                value=summary["median_p95_latency_ms"], unit="ms",
                direction="lower_better", values=p95_values,
                claim=(f"P2-5j arm {arm} observed median p95 request latency "
                       f"{summary['median_p95_latency_ms']:.9g} ms; observation only"),
                extra={**common, "measurement_role": "placement_observation"},
            ),
            _belief_measurement(
                arm=arm, metric_suffix="paired_ratio_to_incumbent",
                metric="paired_decode_ratio_to_incumbent",
                value=summary["median_paired_ratio_to_incumbent"], unit="ratio",
                direction="higher_better", values=ratio_values,
                claim=(f"P2-5j arm {arm} observed median paired decode ratio "
                       f"{summary['median_paired_ratio_to_incumbent']:.9g} versus I; "
                       "observation only"),
                extra={**common, "measurement_role": "placement_comparison",
                       "incumbent_arm": "I"},
            ),
        ))
    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "authority": AUTHORITY,
        "campaign_id": campaign_id,
        "protocol_id": PROTOCOL_ID,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "started_at": started_at,
        "ended_at": ended_at,
        "producer": _producer_identity(),
        "identity": normalized_identity,
        "shape": normalized_shape,
        "arm_definitions": {arm: dict(spec) for arm, spec in ARM_SPECS.items()},
        "blocks": sorted(normalized_blocks, key=lambda row: row["block"]),
        "arm_summaries": arm_summaries,
        "comparisons": comparisons,
        "belief_measurements": belief_measurements,
        "verdict": {
            "status": verdict_status,
            "observed_leader_arm": selected or "I",
            "selected_arm": "I",
            "device_local_move_authorized": False,
            "requires_np_context_ceiling_rederivation": False,
            "would_require_np_context_ceiling_rederivation_after_selection": bool(selected),
            "kernel_speedup_claim": False,
            "carve_authorized": False,
            "production_activation_authorized": False,
            "measurement_constitution_gap": (
                "P2-5j varies CPU affinity, so P-BENCH-PLACEMENT-1 controls. "
                "The four-arm design lacks that protocol's A0-A4 composite, np=1 "
                "anchor, cold/warm cache pairing, and measured-locality gates; this "
                "receipt is observation-only until a human-ratified compliant "
                "successor protocol is executed."),
        },
    }
    payload["receipt_sha256"] = receipt_sha256(payload)
    return payload


def write_receipt(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise PlacementReceiptError(f"refusing to overwrite existing receipt: {target}")
    if payload.get("receipt_sha256") != receipt_sha256(payload):
        raise PlacementReceiptError("receipt self-hash is absent or invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-record", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    source_path = args.campaign_record.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise PlacementReceiptError("campaign record must be a JSON object")
    payload = finalize_campaign(source, base_dir=source_path.parent)
    write_receipt(args.output, payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "receipt_sha256": payload["receipt_sha256"],
        "verdict": payload["verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
