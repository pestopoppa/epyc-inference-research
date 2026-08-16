"""Prospective ClaimTuple-shaped rows for non-promotable GPU discovery.

The first GPU discovery receipts (s1/s2) predate this hook and deliberately
remain unprojected.  Successor runs call these functions before their atomic
writes so the native receipt, rather than a later reader, records what was
measured and the strict boundary on what that measurement can authorize.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from scripts.kernel_rnd.autokernel import schemas


BANK_SCHEMA = "epyc.autokernel.gpu_screening_baseline.v2"
RESULT_SCHEMA = "epyc.autokernel.gpu_candidate_only_screen.v2"
PRODUCER_ID = "scripts.benchmark.run_autokernel_gpu_discovery/v4"
PRODUCER_PATH = "scripts/benchmark/run_autokernel_gpu_discovery.py"
AUTHORITY = "nonpromotable_candidate_only_discovery"
ALLOWED_DISCOVERY_REPS = {3, 5, 9}


class BeliefRefused(ValueError):
    """The native GPU discovery evidence cannot honestly produce a row."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BeliefRefused(f"{label} must be a mapping")
    return value


def _samples(value: Any, label: str, *, expected: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) \
            or len(value) != expected:
        raise BeliefRefused(f"{label} must contain exactly {expected} samples")
    samples = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) \
                or not math.isfinite(item) or item <= 0:
            raise BeliefRefused(f"{label}[{index}] must be positive and finite")
        samples.append(float(item))
    return samples


def _producer(path: Path) -> dict:
    if path.name != "run_autokernel_gpu_discovery.py" or not path.is_file():
        raise BeliefRefused("producer path must name run_autokernel_gpu_discovery.py")
    return {"producer_id": PRODUCER_ID, "path": PRODUCER_PATH,
            "sha256": _sha256_file(path)}


def _row(*, measurement_id: str, metric: str, value: float, unit: str,
         category: str, claim: str, reps_basis: str, extra: Mapping[str, Any],
         protocol_id: str, reps: int) -> dict:
    row = {
        "measurement_id": measurement_id,
        "metric": metric,
        "value": value,
        "unit": unit,
        "metric_direction": "higher_better",
        "category": category,
        "protocol_id": protocol_id,
        "reps": reps,
        "reps_basis": reps_basis,
        "claim": claim,
        "extra": dict(extra),
    }
    row["measurement_sha256"] = schemas.content_hash(row)
    return row


def _common(receipt: Mapping[str, Any], *, producer: Mapping[str, Any],
            samples: list[float]) -> dict:
    if receipt.get("authority") != AUTHORITY:
        raise BeliefRefused("GPU discovery authority is not non-promotable")
    frame = _mapping(receipt.get("frame"), "frame")
    recipe_metric = {
        "pp512-ngl99": "prefill_tokens_per_s",
        "tg128-ngl99": "decode_tokens_per_s",
    }
    if (frame.get("backend") != "llama_gpu"
            or recipe_metric.get(frame.get("recipe")) != frame.get("metric")
            or frame.get("metric_direction") != "higher_better"
            or frame.get("device") != "AMD Instinct MI210"
            or frame.get("architecture") != "gfx90a"):
        raise BeliefRefused("GPU discovery frame is not a sealed MI210 discovery frame")
    factor = _mapping(receipt.get("sole_factor"), "sole_factor")
    if set(factor) != {"name", "anchor", "candidate"}:
        raise BeliefRefused("sole_factor must have exact name/anchor/candidate identity")
    evidence = {
        "campaign_id": receipt.get("campaign_id"),
        "authority": AUTHORITY,
        "frame": dict(frame),
        "sole_factor": dict(factor),
        "samples": samples,
        "producer_sha256": producer["sha256"],
    }
    return {
        "authority": AUTHORITY,
        "non_promotable": True,
        "top_k_discovery_only": True,
        "promotion_authority": False,
        "production_tree_touched": False,
        "frame": dict(frame),
        "sole_factor": dict(factor),
        "producer_id": producer["producer_id"],
        "producer_sha256": producer["sha256"],
        "evidence_basis": evidence,
        "evidence_sha256": schemas.content_hash(evidence),
    }


def _baseline_center(receipt: Mapping[str, Any], *, samples: list[float],
                     run: Mapping[str, Any]) -> float:
    frame = _mapping(receipt.get("frame"), "frame")
    contract = frame.get("metric_contract")
    if (isinstance(contract, Mapping)
            and contract.get("schema") ==
            "epyc.autokernel.serialized_pair_max_metric.v1"):
        metric = run.get("metric")
        if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric)) or float(metric) <= 0):
            raise BeliefRefused("serialized pair-max run lacks tokens/mean-latency metric")
        return float(metric)
    return sum(samples) / len(samples)


def attach_baseline_beliefs(receipt: Mapping[str, Any], *, producer_path: Path) -> dict:
    """Seal the three-anchor bank and its prospective baseline ClaimTuple row."""
    if receipt.get("schema") != BANK_SCHEMA or receipt.get("status") != "complete":
        raise BeliefRefused("baseline receipt must be a complete v2 bank")
    if "belief_measurements" in receipt or "baseline_sha256" in receipt:
        raise BeliefRefused("baseline belief attachment is write-once")
    reps = receipt.get("anchor_invocations")
    if reps not in ALLOWED_DISCOVERY_REPS:
        raise BeliefRefused("anchor_invocations must be one of 3, 5, or 9")
    samples = _samples(receipt.get("anchor_samples"), "anchor_samples", expected=reps)
    runs = receipt.get("anchor_runs")
    if (not isinstance(runs, list) or len(runs) != 1
            or not isinstance(runs[0], dict)
            or runs[0].get("samples") != samples
            or runs[0].get("sample_count") != reps
            or runs[0].get("hip_residency_proved") is not True):
        raise BeliefRefused("anchor process does not bind the native raw sample vector")
    producer = _producer(producer_path)
    common = _common(receipt, producer=producer, samples=samples)
    identity = _mapping(receipt.get("anchor_identity"), "anchor_identity")
    common.update({"arm": "anchor", "build_identity": dict(identity)})
    center = _baseline_center(receipt, samples=samples, run=runs[0])
    recipe = receipt["frame"]["recipe"]
    label = "pp512" if recipe == "pp512-ngl99" else "tg128"
    metric = receipt["frame"]["metric"]
    result = dict(receipt)
    result["producer"] = producer
    result["belief_measurements"] = [_row(
        measurement_id=f"gpu_discovery_anchor_{label}_median_tokens_per_s",
        metric=f"gpu_{metric}", value=median(samples), unit="tokens/s",
        category="BASELINE",
        claim=(f"Non-promotable GPU discovery anchor observed median {label} throughput "
               f"{median(samples):.9g} tokens/s"),
        reps_basis=f"scored:{reps} anchor-bank MI210 llama-bench native repetitions",
        extra={**common, "sealed_baseline_center": center,
               "baseline_center_method": (
                   "tokens_per_mean_protected_latency"
                   if receipt["frame"].get("metric_contract", {}).get("schema") ==
                   "epyc.autokernel.serialized_pair_max_metric.v1"
                   else "arithmetic_mean_native_samples")},
        protocol_id=BANK_SCHEMA, reps=reps)]
    result["baseline_sha256"] = schemas.content_hash(result)
    return result


def attach_result_beliefs(receipt: Mapping[str, Any], *, bank: Mapping[str, Any],
                          producer_path: Path) -> dict:
    """Seal candidate throughput/effect rows against an already sealed anchor bank."""
    if receipt.get("schema") != RESULT_SCHEMA or receipt.get("status") != "complete":
        raise BeliefRefused("candidate receipt must be a complete v2 result")
    if "belief_measurements" in receipt or "result_sha256" in receipt:
        raise BeliefRefused("candidate belief attachment is write-once")
    if bank.get("schema") != BANK_SCHEMA or bank.get("status") != "complete":
        raise BeliefRefused("candidate result requires a complete v2 anchor bank")
    bank_sha = bank.get("baseline_sha256")
    unsigned_bank = {key: value for key, value in bank.items() if key != "baseline_sha256"}
    if not isinstance(bank_sha, str) or bank_sha != schemas.content_hash(unsigned_bank) \
            or receipt.get("baseline_sha256") != bank_sha:
        raise BeliefRefused("candidate result does not bind the sealed anchor bank")
    reps = receipt.get("candidate_invocations")
    if reps not in ALLOWED_DISCOVERY_REPS or receipt.get("anchor_invocations") != reps:
        raise BeliefRefused("matched invocation counts must both be one of 3, 5, or 9")
    samples = _samples(receipt.get("candidate_samples"), "candidate_samples",
                       expected=reps)
    runs = receipt.get("candidate_runs")
    if (not isinstance(runs, list) or len(runs) != 1
            or not isinstance(runs[0], dict)
            or runs[0].get("samples") != samples
            or runs[0].get("sample_count") != reps
            or runs[0].get("hip_residency_proved") is not True):
        raise BeliefRefused("candidate process does not bind the native raw sample vector")
    baseline_center = receipt.get("baseline_center")
    if isinstance(baseline_center, bool) or not isinstance(baseline_center, (int, float)) \
            or not math.isfinite(baseline_center) or baseline_center <= 0:
        raise BeliefRefused("baseline_center must be positive and finite")
    if bank.get("anchor_invocations") != reps:
        raise BeliefRefused("candidate invocation count differs from sealed bank")
    bank_samples = _samples(bank.get("anchor_samples"), "bank.anchor_samples",
                            expected=reps)
    bank_runs = bank.get("anchor_runs")
    if (not isinstance(bank_runs, list) or len(bank_runs) != 1
            or not isinstance(bank_runs[0], Mapping)):
        raise BeliefRefused("bank anchor run is malformed")
    expected_center = _baseline_center(bank, samples=bank_samples, run=bank_runs[0])
    if not math.isclose(float(baseline_center), expected_center,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise BeliefRefused("baseline_center does not rederive from the sealed bank")
    effects = [(sample - float(baseline_center)) / float(baseline_center)
               for sample in samples]
    declared_effects = receipt.get("relative_effects")
    if not isinstance(declared_effects, list) or len(declared_effects) != reps \
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12)
                   for value, expected in zip(declared_effects, effects)):
        raise BeliefRefused("relative effects do not rederive from candidate and bank samples")
    if not math.isclose(float(receipt.get("median_relative")), median(effects),
                        rel_tol=1e-12, abs_tol=1e-12):
        raise BeliefRefused("median_relative does not rederive from the effect vector")
    if (receipt.get("state") != "decided" or receipt.get("ok") is not True
            or receipt.get("non_promotable") is not True
            or receipt.get("nomination") != "top_k_candidate_only_not_a_keep"
            or receipt.get("hip_residency_proved") is not True):
        raise BeliefRefused("candidate result lacks its successful non-promotable boundary")
    producer = _producer(producer_path)
    common = _common(receipt, producer=producer, samples=samples)
    common.update({
        "arm": "candidate",
        "build_identity": dict(_mapping(receipt.get("candidate_identity"),
                                        "candidate_identity")),
        "baseline_sha256": bank_sha,
        "baseline_anchor_samples": bank_samples,
        "baseline_center": float(baseline_center),
        "hip_residency_proved": True,
    })
    recipe = receipt["frame"]["recipe"]
    label = "pp512" if recipe == "pp512-ngl99" else "tg128"
    metric = receipt["frame"]["metric"]
    rows = [
        _row(
            measurement_id=f"gpu_discovery_candidate_{label}_median_tokens_per_s",
            metric=f"gpu_{metric}", value=median(samples), unit="tokens/s",
            category="CANDIDATE",
            claim=(f"Non-promotable GPU candidate discovery observed median {label} throughput "
                   f"{median(samples):.9g} tokens/s"),
            reps_basis=f"scored:{reps} candidate-only MI210 llama-bench invocations",
            extra=common, protocol_id=RESULT_SCHEMA, reps=reps),
        _row(
            measurement_id=f"gpu_discovery_candidate_{label}_median_relative_effect",
            metric=f"gpu_{metric.removesuffix('_tokens_per_s')}_relative_effect_vs_sealed_anchor",
            value=median(effects), unit="fraction", category="CANDIDATE",
            claim=("Non-promotable GPU candidate discovery observed median relative effect "
                   f"{median(effects):.9g} versus its sealed anchor bank"),
            reps_basis=f"scored:{reps} candidate-only MI210 llama-bench invocations",
            extra={**common, "relative_effects": effects}, protocol_id=RESULT_SCHEMA,
            reps=reps),
    ]
    result = dict(receipt)
    result["baseline_anchor_samples"] = bank_samples
    result["producer"] = producer
    result["belief_measurements"] = rows
    result["result_sha256"] = schemas.content_hash(result)
    return result
