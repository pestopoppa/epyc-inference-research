#!/usr/bin/env python3
"""Finalize ROCm profile receipts into prospective AutoKernel belief rows.

The original profiler receipts are immutable evidence.  This module verifies one
of those receipts and emits a separate, producer-bound receipt containing only
measurements that can enter Vidya's shared measurement ladder.  It never infers
rows from prose and never mutates the source receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "epyc.autokernel.profile_beliefs.v1"
PRODUCER_ID = "scripts.benchmark.autokernel_profile_beliefs/v1"
G15_SCHEMA = "epyc.autokernel.g15_profile.v1"
C4_SCHEMA = "epyc.autokernel.c4_profile_capture.v1"
WGM_SCHEMA = "epyc.autokernel.wgm_proxy_sweep.v1"
CLAIM_SCHEMA = "epyc.autokernel.device_claim_receipt.v1"


class ProfileBeliefError(ValueError):
    """The source receipt cannot support prospective belief measurements."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileBeliefError(f"{path} must be a non-empty string")
    return value.strip()


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileBeliefError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ProfileBeliefError(f"{path} must be finite" + (" and positive" if positive else ""))
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProfileBeliefError(f"{path} must be a positive integer")
    return value


def _producer_identity() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "producer_id": PRODUCER_ID,
        "path": "scripts/benchmark/autokernel_profile_beliefs.py",
        "sha256": sha256_file(path),
    }


def _claim_pair(opened: Any, released: Any, *, campaign_id: str) -> dict[str, Any]:
    if not isinstance(opened, Mapping) or not isinstance(released, Mapping):
        raise ProfileBeliefError("source receipt must contain opened and released device claims")
    for name, receipt in (("opened", opened), ("released", released)):
        if receipt.get("schema") != CLAIM_SCHEMA:
            raise ProfileBeliefError(f"device_claim.{name} has the wrong schema")
        if receipt.get("campaign_id") != campaign_id:
            raise ProfileBeliefError(f"device_claim.{name} campaign differs from the source")
        _text(receipt.get("claim_id"), f"device_claim.{name}.claim_id")
        _text(receipt.get("device_id"), f"device_claim.{name}.device_id")
    for field in ("claim_id", "device_id"):
        if opened.get(field) != released.get(field):
            raise ProfileBeliefError(f"device claim {field} differs across acquisition/release")
    _text(released.get("released_at"), "device_claim.released.released_at")
    return {"opened": dict(opened), "released": dict(released)}


def _measurement(
    *, measurement_id: str, metric: str, value: float, unit: str,
    direction: str, reps: int, reps_basis: str, claim: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "measurement_id": _text(measurement_id, "measurement_id"),
        "metric": _text(metric, "metric"),
        "value": _number(value, "value"),
        "unit": _text(unit, "unit"),
        "metric_direction": direction,
        "category": "BASELINE",
        "reps": _positive_int(reps, "reps"),
        "reps_basis": _text(reps_basis, "reps_basis"),
        "claim": _text(claim, "claim"),
        "extra": dict(extra),
    }
    if direction not in {"higher_better", "lower_better"}:
        raise ProfileBeliefError("metric direction is unsupported")
    row["measurement_sha256"] = canonical_sha256(row)
    return row


def _base(
    receipt: Mapping[str, Any], *, expected_schema: str,
    source_locator: str, source_sha256: str,
    claim: Mapping[str, Any], measurements: list[dict[str, Any]],
    allow_status_absent: bool = False,
) -> dict[str, Any]:
    if receipt.get("schema") != expected_schema:
        raise ProfileBeliefError(f"source schema is not {expected_schema}")
    status = receipt.get("status")
    if (status not in {"pass", "passed", "complete"}
            and not (allow_status_absent and status is None)):
        raise ProfileBeliefError("failed or incomplete source receipt cannot emit belief rows")
    campaign_id = _text(receipt.get("campaign_id"), "campaign_id")
    digest = _text(source_sha256, "source_sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ProfileBeliefError("source_sha256 must be a lowercase SHA-256 digest")
    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "authority": "prospective_profile_measurements_only",
        "campaign_id": campaign_id,
        "started_at": _text(receipt.get("started_at"), "started_at"),
        "ended_at": _text(receipt.get("ended_at"), "ended_at"),
        "producer": _producer_identity(),
        "source_receipt": {
            "schema": expected_schema,
            "locator": _text(source_locator, "source_locator"),
            "sha256": digest,
        },
        "device_claim": dict(claim),
        "belief_measurements": measurements,
    }
    if not measurements:
        raise ProfileBeliefError("a successful finalizer must emit at least one measurement")
    payload["receipt_sha256"] = receipt_sha256(payload)
    return payload


def finalize_g15(
    receipt: Mapping[str, Any], *, source_locator: str, source_sha256: str,
) -> dict[str, Any]:
    if receipt.get("schema") != G15_SCHEMA:
        raise ProfileBeliefError("source is not a G15 receipt")
    campaign_id = _text(receipt.get("campaign_id"), "campaign_id")
    claim = _claim_pair(
        receipt.get("device_claim_open"), receipt.get("device_claim_released"),
        campaign_id=campaign_id,
    )
    profiles = receipt.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ProfileBeliefError("G15 receipt has no scored profiles")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            raise ProfileBeliefError(f"profiles[{index}] must be an object")
        parallel = _positive_int(profile.get("parallel"), f"profiles[{index}].parallel")
        if parallel in seen:
            raise ProfileBeliefError("G15 parallel cells must be unique")
        seen.add(parallel)
        bench = profile.get("bench")
        attribution = profile.get("attribution")
        hypothesis = profile.get("hypothesis")
        if not all(isinstance(value, Mapping) for value in (bench, attribution, hypothesis)):
            raise ProfileBeliefError(f"profiles[{index}] lacks bench/attribution/hypothesis")
        speed = _number(bench.get("speed_tg"), f"profiles[{index}].bench.speed_tg", positive=True)
        share = _number(
            attribution.get("elementwise_norm_target_share"),
            f"profiles[{index}].attribution.elementwise_norm_target_share",
        )
        if not 0 <= share <= 1:
            raise ProfileBeliefError("G15 target share must be in [0,1]")
        if _number(hypothesis.get("observed_target_share"), "hypothesis.observed_target_share") != share:
            raise ProfileBeliefError("G15 hypothesis and attribution target shares differ")
        common = {
            "measurement_surface": "gfx90a_batched_decode_g15",
            "parallel": parallel,
            "device_id": claim["opened"]["device_id"],
            "device_claim_id": claim["opened"]["claim_id"],
            "source_receipt_sha256": source_sha256,
            "promotion_authority": False,
        }
        rows.append(_measurement(
            measurement_id=f"g15_b{parallel}_decode_tokens_per_second",
            metric="batched_decode_tokens_per_second", value=speed,
            unit="tokens/s", direction="higher_better", reps=1,
            reps_basis="scored:one hash-bound profiled llama-bench cell",
            claim=f"gfx90a B={parallel} profiled decode throughput is {speed:.9g} tokens/s",
            extra={**common, "measurement_role": "performance_baseline"},
        ))
        rows.append(_measurement(
            measurement_id=f"g15_b{parallel}_elementwise_norm_target_share",
            metric="elementwise_norm_summed_kernel_time_share", value=share,
            unit="fraction", direction="higher_better", reps=1,
            reps_basis="scored:one complete rocprof-v1 timestamp cell",
            claim=(f"gfx90a B={parallel} elementwise+norm target share is {share:.12g}; "
                   "higher means more authoring opportunity, not higher runtime performance"),
            extra={**common, "measurement_role": "target_selection_signal",
                   "target_selection_only": True,
                   "verdict": _text(hypothesis.get("verdict"), "hypothesis.verdict")},
        ))
    return _base(
        receipt, expected_schema=G15_SCHEMA, source_locator=source_locator,
        source_sha256=source_sha256, claim=claim, measurements=rows,
    )


def finalize_c4(
    receipt: Mapping[str, Any], report: Mapping[str, Any], *,
    source_locator: str, source_sha256: str, report_sha256: str,
) -> dict[str, Any]:
    if receipt.get("schema") != C4_SCHEMA:
        raise ProfileBeliefError("source is not a C4 paired-capture receipt")
    campaign_id = _text(receipt.get("campaign_id"), "campaign_id")
    claim = _claim_pair(
        receipt.get("device_claim_open"), receipt.get("device_claim_released"),
        campaign_id=campaign_id,
    )
    if _text(receipt.get("report_sha256"), "report_sha256") != report_sha256:
        raise ProfileBeliefError("C4 report hash differs from the source receipt")
    if report.get("schema") != "epyc.autokernel.c4_profile_report.v1":
        raise ProfileBeliefError("C4 report schema is unsupported")
    formal = receipt.get("formal")
    if not isinstance(formal, Mapping):
        raise ProfileBeliefError("C4 receipt lacks the formal capture")
    steps = _positive_int(formal.get("active_steps"), "formal.active_steps")
    formal_receipt = formal.get("receipt")
    if not isinstance(formal_receipt, Mapping):
        raise ProfileBeliefError("C4 formal capture lacks its receipt")
    workload_id = _text(formal_receipt.get("workload_id"), "formal.receipt.workload_id")
    table = report.get("kernel_table")
    if not isinstance(table, list) or not table:
        raise ProfileBeliefError("C4 report has no kernel table")
    rows = []
    for index, item in enumerate(table):
        if not isinstance(item, Mapping):
            raise ProfileBeliefError(f"kernel_table[{index}] must be an object")
        family = _text(item.get("kernel_family"), f"kernel_table[{index}].kernel_family")
        duration = _number(item.get("duration_ns"), f"kernel_table[{index}].duration_ns")
        share = _number(item.get("gpu_time_share"), f"kernel_table[{index}].gpu_time_share")
        if duration < 0 or not 0 <= share <= 1:
            raise ProfileBeliefError("C4 kernel duration/share is outside its physical range")
        per_suite = duration / steps
        safe_family = "".join(char if char.isalnum() else "_" for char in family).strip("_").lower()
        rows.append(_measurement(
            measurement_id=f"c4_{workload_id}_{safe_family}_device_ns_per_suite",
            metric="profiled_kernel_family_device_duration_ns_per_suite",
            value=per_suite, unit="ns", direction="lower_better", reps=steps,
            reps_basis="scored:formal production-optimization profiler suites",
            claim=(f"C4 {workload_id} {family} mean formal device time is "
                   f"{per_suite:.9g} ns per suite"),
            extra={
                "measurement_surface": "c4_formal_profile",
                "measurement_role": "kernel_family_performance_baseline",
                "workload_id": workload_id,
                "kernel_family": family,
                "formal_gpu_time_share": share,
                "formal_active_steps": steps,
                "report_sha256": report_sha256,
                "source_receipt_sha256": source_sha256,
                "device_id": claim["opened"]["device_id"],
                "device_claim_id": claim["opened"]["claim_id"],
                "promotion_authority": False,
            },
        ))
    return _base(
        receipt, expected_schema=C4_SCHEMA, source_locator=source_locator,
        source_sha256=source_sha256, claim=claim, measurements=rows,
        allow_status_absent=True,
    )


def finalize_wgm_proxy(
    receipt: Mapping[str, Any], *, source_locator: str, source_sha256: str,
) -> dict[str, Any]:
    if receipt.get("schema") != WGM_SCHEMA:
        raise ProfileBeliefError("source is not a WGM proxy receipt")
    campaign_id = _text(receipt.get("campaign_id"), "campaign_id")
    device_claim = receipt.get("device_claim")
    if not isinstance(device_claim, Mapping):
        raise ProfileBeliefError("WGM receipt lacks its device claim")
    claim = _claim_pair(
        device_claim.get("opened"), device_claim.get("released"),
        campaign_id=campaign_id,
    )
    if receipt.get("surface") != "standalone_l2_tile_reuse_proxy_not_mmq":
        raise ProfileBeliefError("WGM receipt does not declare the standalone proxy boundary")
    result = receipt.get("result")
    factors = result.get("factors") if isinstance(result, Mapping) else None
    if not isinstance(factors, Mapping) or not factors:
        raise ProfileBeliefError("WGM receipt has no scored factors")
    rows = []
    for label, item in sorted(factors.items(), key=lambda pair: int(pair[0])):
        if not isinstance(item, Mapping):
            raise ProfileBeliefError(f"WGM factor {label} must be an object")
        factor = int(label)
        median = _number(item.get("median_ms"), f"factors.{label}.median_ms", positive=True)
        reps = _positive_int(item.get("sample_count"), f"factors.{label}.sample_count")
        rows.append(_measurement(
            measurement_id=f"wgm_proxy_factor_{factor}_elapsed_ms",
            metric="wgm_l2_proxy_elapsed_ms", value=median, unit="ms",
            direction="lower_better", reps=reps,
            reps_basis="scored:balanced standalone WGM proxy rounds",
            claim=f"Standalone gfx90a WGM proxy factor {factor} median is {median:.9g} ms",
            extra={
                "measurement_surface": "standalone_l2_tile_reuse_proxy_not_mmq",
                "measurement_role": "design_prior_only",
                "wgm_factor": factor,
                "does_not_transfer_to_real_mmq": True,
                "source_receipt_sha256": source_sha256,
                "device_id": claim["opened"]["device_id"],
                "device_claim_id": claim["opened"]["claim_id"],
                "promotion_authority": False,
            },
        ))
    return _base(
        receipt, expected_schema=WGM_SCHEMA, source_locator=source_locator,
        source_sha256=source_sha256, claim=claim, measurements=rows,
    )


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    if receipt.get("schema") != SCHEMA or receipt.get("receipt_sha256") != receipt_sha256(receipt):
        raise ProfileBeliefError("finalized receipt failed its self-hash contract")
    target = Path(path)
    if target.exists():
        raise ProfileBeliefError(f"refusing to overwrite existing receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("g15", "c4", "wgm-proxy"), required=True)
    parser.add_argument("--input-receipt", required=True)
    parser.add_argument("--report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_path = Path(args.input_receipt).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    common = {"source_locator": str(source_path), "source_sha256": sha256_file(source_path)}
    if args.kind == "g15":
        if args.report:
            raise ProfileBeliefError("G15 finalization does not accept --report")
        finalized = finalize_g15(source, **common)
    elif args.kind == "wgm-proxy":
        if args.report:
            raise ProfileBeliefError("WGM finalization does not accept --report")
        finalized = finalize_wgm_proxy(source, **common)
    else:
        if not args.report:
            raise ProfileBeliefError("C4 finalization requires --report")
        report_path = Path(args.report).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        finalized = finalize_c4(
            source, report, report_sha256=sha256_file(report_path), **common)
    target = write_receipt(args.output, finalized)
    print(json.dumps({"receipt": str(target), "receipt_sha256": finalized["receipt_sha256"],
                      "measurements": len(finalized["belief_measurements"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
