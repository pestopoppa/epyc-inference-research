#!/usr/bin/env python3
"""Finalize prospective IQ2_XXS model-confirmation receipts with belief rows.

This is a write-side finalizer, not a historical adapter.  It accepts only the
new ``epyc.autokernel.iq2_xxs_model_confirmation.v1`` receipt shape and refuses
an input which already carries ``belief_measurements``.  A valid input binds:

* an admitted T1/T1a event for the one-row kernel result;
* an admitted T2 event for each of decode (TG) and prefill (PP);
* the exact ``microbench_raw_vector.v1`` llama-bench material behind each T2
  event;
* a current ``candidate.v1`` build record; and
* a released ``cpu_region_claim_receipt.v1``.

Only after those records agree does the finalizer emit arm-specific median TG
and PP rows.  Raw, incomplete, mixed-identity, unranked, or unreleased evidence
cannot become a model-level belief.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics as std_statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_rnd.autokernel import schemas
from scripts.kernel_rnd.autokernel.evaluator import statistics as ak_statistics
from scripts.kernel_rnd.autokernel.execution import cpu_region_claim


SCHEMA = "epyc.autokernel.iq2_xxs_model_confirmation.v1"
PRODUCER_ID = "autokernel.iq2_xxs_model_beliefs/v1"
RAW_SCHEMA = "epyc.autokernel.microbench_raw_vector.v1"
EVENT_SCHEMA = schemas.SCHEMA_EVALUATION_EVENT_V5
QUANTIZATION = "IQ2_XXS"
LANES = {
    "tg": {
        "recipe_id": "t1b.llama_cpu.llama_bench_decode.v1",
        "metric": "decode_tokens_per_s",
    },
    "pp": {
        "recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1",
        "metric": "prefill_tokens_per_s",
    },
}
_T1_TIERS = frozenset({"T1", "T1a", "T1b", "T1c"})
_SHA256_LEN = 64


class ReceiptRefused(ValueError):
    """The supplied material cannot support prospective model beliefs."""


def _refuse(message: str) -> None:
    raise ReceiptRefused(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _refuse(f"{path} must be a mapping")
    return value


def _list(value: Any, path: str, *, nonempty: bool = False) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        suffix = " and non-empty" if nonempty else ""
        _refuse(f"{path} must be a list{suffix}")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _refuse(f"{path} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, path: str) -> str:
    value = _text(value, path)
    if len(value) != _SHA256_LEN or any(c not in "0123456789abcdef" for c in value):
        _refuse(f"{path} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        _refuse(f"{path} must be a finite {qualifier}number")
    return result


def _instant(value: Any, path: str) -> datetime:
    value = _text(value, path)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        _refuse(f"{path} must be an ISO-8601 timestamp: {exc}")
    if result.tzinfo is None:
        _refuse(f"{path} must carry a timezone offset")
    return result


def _content_sha256(value: Any) -> str:
    return schemas.content_hash(value)


def _producer_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _event_discipline(event: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    performance = _mapping(event.get("performance"), f"{path}.performance")
    return _mapping(
        performance.get("search_discipline"),
        f"{path}.performance.search_discipline",
    )


def _admitted_event(
    event: Any,
    *,
    path: str,
    campaign_id: str,
    candidate_id: str,
    candidate_record: Mapping[str, Any],
    anchor: Mapping[str, Any],
    claim_id: str,
    allowed_tiers: frozenset[str],
) -> tuple[Mapping[str, Any], float]:
    event = _mapping(event, path)
    if event.get("schema") != EVENT_SCHEMA:
        _refuse(f"{path}.schema must be the current {EVENT_SCHEMA!r}")
    violations = schemas.validate_evaluation_event(event)
    if violations:
        _refuse(f"{path} is not a valid evaluation event: {'; '.join(violations)}")
    if event.get("campaign_id") != campaign_id or event.get("candidate_id") != candidate_id:
        _refuse(f"{path} belongs to a different campaign or candidate")
    if event.get("tier") not in allowed_tiers:
        _refuse(f"{path}.tier must be one of {sorted(allowed_tiers)}")
    if event.get("backend") != "llama_cpu" or event.get("device_state") is not None:
        _refuse(f"{path} must be a CPU event with explicit null device_state")
    if event.get("co_residency") != "single":
        _refuse(f"{path}.co_residency must be 'single'")
    if event.get("status") != "pass" or event.get("integrity_flags") != []:
        _refuse(f"{path} must be a clean PASS with no integrity flags")
    if event.get("resource_claim_receipt") != claim_id:
        _refuse(f"{path}.resource_claim_receipt does not name the released claim")

    candidate_receipts = _mapping(candidate_record.get("receipts"), "candidate_record.receipts")
    if event.get("host_receipt") != candidate_receipts.get("host_receipt"):
        _refuse(f"{path}.host_receipt does not match candidate_record.receipts")
    if event.get("artifact") != {
        "source_sha256": candidate_record["source_snapshot"]["snapshot_sha256"],
        "binary_sha256": candidate_record["artifacts"]["binary_sha256"],
        "linkage_sha256": candidate_record["artifacts"]["linkage_sha256"],
    }:
        _refuse(f"{path}.artifact does not match candidate source/build identity")
    if event.get("anchor") != anchor:
        _refuse(f"{path}.anchor does not match the receipt anchor")

    discipline = _event_discipline(event, path)
    search_grade = _mapping(discipline.get("search_grade"), f"{path}.search_grade")
    if search_grade.get("satisfied") is not True or search_grade.get("failed") not in ([], ()):
        _refuse(f"{path} is not search-grade")
    if discipline.get("void_findings") not in ([], ()):
        _refuse(f"{path} carries void findings")
    if discipline.get("effect_resolution") != "improvement":
        _refuse(f"{path} did not resolve to an improvement")
    if discipline.get("speed_rank_admissible") is not True:
        _refuse(f"{path} has no admitted speed rank")

    performance = _mapping(event["performance"], f"{path}.performance")
    estimate = _finite(performance.get("estimate"), f"{path}.performance.estimate")
    return event, estimate


def _candidate_record(receipt: Mapping[str, Any], *, campaign_id: str,
                      candidate_id: str, claim_id: str) -> Mapping[str, Any]:
    candidate = _mapping(receipt.get("candidate_record"), "candidate_record")
    violations = schemas.validate_candidate(candidate)
    if violations:
        _refuse(f"candidate_record is invalid: {'; '.join(violations)}")
    if candidate.get("campaign_id") != campaign_id or candidate.get("candidate_id") != candidate_id:
        _refuse("candidate_record belongs to a different campaign or candidate")
    if candidate["receipts"].get("resource_claim_receipt") != claim_id:
        _refuse("candidate_record does not name the released CPU claim")
    if candidate.get("status") not in {"evaluating", "banked"}:
        _refuse("candidate_record must still be evaluating or already banked")
    return candidate


def _released_claim(receipt: Mapping[str, Any], *, campaign_id: str) -> Mapping[str, Any]:
    value = _mapping(receipt.get("resource_claim_receipt"), "resource_claim_receipt")
    try:
        parsed = cpu_region_claim.RegionClaimReceipt.from_dict(value)
    except (TypeError, ValueError) as exc:
        _refuse(f"resource_claim_receipt is invalid: {exc}")
    if parsed.campaign_id != campaign_id:
        _refuse("resource_claim_receipt belongs to a different campaign")
    if parsed.role != "autokernel" or parsed.roles != ("autokernel",):
        _refuse("resource_claim_receipt is not an exclusive AutoKernel CPU claim")
    if not parsed.released_at:
        _refuse("resource_claim_receipt has not been released")
    return value


def _model_identity(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    model = _mapping(receipt.get("model_identity"), "model_identity")
    _text(model.get("model_id"), "model_identity.model_id")
    _text(model.get("path"), "model_identity.path")
    _sha256(model.get("sha256"), "model_identity.sha256")
    if model.get("quantization") != QUANTIZATION:
        _refuse(f"model_identity.quantization must be exactly {QUANTIZATION!r}")
    return model


def _parse_paired_block(value: Any, path: str) -> ak_statistics.PairedBlock:
    if not isinstance(value, list) or len(value) != 9:
        _refuse(f"{path} must be a canonical nine-field PairedBlock list")
    try:
        return ak_statistics.PairedBlock(
            block_index=value[0], unit_id=value[1], stratum=value[2], order=value[3],
            segment=value[4], extension_round=value[5], measured_at=value[6],
            anchor_samples=tuple(value[7]), candidate_samples=tuple(value[8]),
        )
    except (TypeError, ValueError) as exc:
        _refuse(f"{path} is not a valid PairedBlock: {exc}")


def _receipt_identity(receipt: Mapping[str, Any]) -> dict:
    return {
        key: receipt.get(key)
        for key in (
            "runner_id", "registry_id", "arm", "binary_path", "binary_sha256",
            "binary_size", "source_root", "library_path",
        )
    }


def _verify_execution_receipt(
    value: Any,
    *,
    path: str,
    arm: str,
    recipe_id: str,
    model: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _mapping(value, path)
    if value.get("runner_id") != "autokernel.execution.microbench/v1":
        _refuse(f"{path}.runner_id is not the formal microbench runner")
    if value.get("arm") != arm or value.get("recipe_id") != recipe_id:
        _refuse(f"{path} names the wrong arm or recipe")
    for key in ("constructor_sha256", "argv_sha256", "env_sha256", "binary_sha256"):
        _sha256(value.get(key), f"{path}.{key}")
    params = _mapping(value.get("params"), f"{path}.params")
    if params.get("model") != model["path"]:
        _refuse(f"{path}.params.model does not match model_identity.path")
    recipe_env = _mapping(value.get("recipe_env"), f"{path}.recipe_env")
    if recipe_env.get("GGML_IQK") != "1":
        _refuse(f"{path} must run with GGML_IQK=1")

    if arm == "candidate":
        expected_binary = candidate_record["artifacts"]["binary_sha256"]
        if value.get("source_root") != candidate_record["worktree"]["path"]:
            _refuse(f"{path}.source_root does not match candidate_record.worktree")
        build_dir = os.path.normpath(candidate_record["build"]["build_dir"])
        binary_path = os.path.normpath(_text(value.get("binary_path"), f"{path}.binary_path"))
        try:
            inside = os.path.commonpath((build_dir, binary_path)) == build_dir
        except ValueError:
            inside = False
        if not inside:
            _refuse(f"{path}.binary_path is outside candidate_record.build.build_dir")
    else:
        expected_binary = anchor["binary_sha256"]
    if value.get("binary_sha256") != expected_binary:
        _refuse(f"{path}.binary_sha256 does not match the bound {arm} identity")
    return value


def _invocation_samples(block: Mapping[str, Any], *, arm: str, model_path: str,
                        receipt_identity: Mapping[str, Any], path: str) -> tuple[float, ...]:
    invocations = _list(block.get("invocations"), f"{path}.invocations", nonempty=True)
    arm_rows = [row for row in invocations if isinstance(row, Mapping) and row.get("arm") == arm]
    if len(arm_rows) != 1:
        _refuse(f"{path} must carry exactly one {arm} invocation")
    invocation = arm_rows[0]
    if _receipt_identity(_mapping(invocation.get("receipt"), f"{path}.{arm}.receipt")) \
            != dict(receipt_identity):
        _refuse(f"{path}.{arm}.receipt changes the run-level execution identity")
    row = _mapping(invocation.get("row"), f"{path}.{arm}.row")
    if row.get("model_filename") != model_path:
        _refuse(f"{path}.{arm}.row names a different model")
    samples = tuple(
        _finite(value, f"{path}.{arm}.samples[{index}]", positive=True)
        for index, value in enumerate(
            _list(invocation.get("samples"), f"{path}.{arm}.samples", nonempty=True)
        )
    )
    row_samples = row.get("samples_ts")
    if row_samples is not None and tuple(float(v) for v in row_samples) != samples:
        _refuse(f"{path}.{arm}.row.samples_ts disagrees with invocation.samples")
    for name, check in _list(invocation.get("checks"), f"{path}.{arm}.checks"):
        if not isinstance(check, Mapping) or check.get("outcome") != schemas.PASS:
            _refuse(f"{path}.{arm} carries a non-PASS invocation check {name!r}")
    spawn = _mapping(invocation.get("spawn"), f"{path}.{arm}.spawn")
    if spawn.get("returncode") != 0 or spawn.get("timed_out") is not False:
        _refuse(f"{path}.{arm} did not exit cleanly")
    return samples


def _lane_samples(
    lane: str,
    lane_value: Any,
    *,
    campaign_id: str,
    candidate_id: str,
    candidate_record: Mapping[str, Any],
    model: Mapping[str, Any],
    anchor: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict:
    path = f"lanes.{lane}"
    value = _mapping(lane_value, path)
    claim_id = claim["claim_id"]
    t1_event, t1_effect = _admitted_event(
        value.get("t1_event"), path=f"{path}.t1_event", campaign_id=campaign_id,
        candidate_id=candidate_id, candidate_record=candidate_record, anchor=anchor,
        claim_id=claim_id, allowed_tiers=_T1_TIERS,
    )
    t2_event, t2_effect = _admitted_event(
        value.get("t2_event"), path=f"{path}.t2_event", campaign_id=campaign_id,
        candidate_id=candidate_id, candidate_record=candidate_record, anchor=anchor,
        claim_id=claim_id, allowed_tiers=frozenset({"T2"}),
    )
    grammar = _mapping(t2_event.get("claim_grammar"), f"{path}.t2_event.claim_grammar")
    expected = LANES[lane]
    if grammar.get("metric") != expected["metric"] or grammar.get("metric_direction") != "higher_better":
        _refuse(f"{path}.t2_event is not the expected higher-is-better {expected['metric']} cell")
    if grammar.get("category") != "CANDIDATE":
        _refuse(f"{path}.t2_event.claim_grammar.category must be CANDIDATE")

    transfers = _list(t2_event.get("transfer_ratio_to"), f"{path}.t2_event.transfer_ratio_to")
    matches = [row for row in transfers if isinstance(row, Mapping)
               and row.get("event_id") == t1_event.get("event_id")]
    if len(matches) != 1 or matches[0].get("tier") != t1_event.get("tier"):
        _refuse(f"{path}.t2_event must carry exactly one transfer binding to its T1 event")
    transfer = matches[0]
    if not math.isclose(float(transfer.get("source_effect")), t2_effect,
                        rel_tol=1e-12, abs_tol=1e-15):
        _refuse(f"{path}.t2_event transfer source_effect does not match its estimate")
    if not math.isclose(float(transfer.get("target_effect")), t1_effect,
                        rel_tol=1e-12, abs_tol=1e-15):
        _refuse(f"{path}.t2_event transfer target_effect does not match the T1 estimate")

    vectors = _list(value.get("raw_vectors"), f"{path}.raw_vectors", nonempty=True)
    blocks: list[ak_statistics.PairedBlock] = []
    vector_digests: list[str] = []
    seen_indexes: set[int] = set()
    candidate_identity = None
    anchor_identity = None
    unit_ids: set[str] = set()
    for vector_index, raw_value in enumerate(vectors):
        raw_path = f"{path}.raw_vectors[{vector_index}]"
        raw = _mapping(raw_value, raw_path)
        if raw.get("schema") != RAW_SCHEMA or raw.get("runner_id") != "autokernel.execution.microbench/v1":
            _refuse(f"{raw_path} is not a formal microbench raw vector")
        if raw.get("recipe_id") != expected["recipe_id"] or raw.get("candidate_id") != candidate_id:
            _refuse(f"{raw_path} names a different recipe or candidate")
        if raw.get("complete") is not True or raw.get("refusals") != []:
            _refuse(f"{raw_path} is incomplete or refused")
        order = _mapping(raw.get("order_control"), f"{raw_path}.order_control")
        if order.get("outcome") != schemas.PASS:
            _refuse(f"{raw_path} failed order control")
        if raw.get("scope_denominator") != t2_event.get("scope_denominator"):
            _refuse(f"{raw_path}.scope_denominator does not match the T2 event")
        if raw.get("anchor_identity") != anchor:
            _refuse(f"{raw_path}.anchor_identity does not match the event anchor")
        raw_started = _instant(raw.get("started_at"), f"{raw_path}.started_at")
        raw_ended = _instant(raw.get("ended_at"), f"{raw_path}.ended_at")
        acquired = _instant(claim.get("acquired_at"), "resource_claim_receipt.acquired_at")
        released = _instant(claim.get("released_at"), "resource_claim_receipt.released_at")
        if not acquired <= raw_started <= raw_ended <= released:
            _refuse(f"{raw_path} measurement interval is outside the released CPU claim")

        cand_receipt = _verify_execution_receipt(
            raw.get("candidate_receipt"), path=f"{raw_path}.candidate_receipt",
            arm="candidate", recipe_id=expected["recipe_id"], model=model,
            candidate_record=candidate_record, anchor=anchor,
        )
        anch_receipt = _verify_execution_receipt(
            raw.get("anchor_receipt"), path=f"{raw_path}.anchor_receipt",
            arm="anchor", recipe_id=expected["recipe_id"], model=model,
            candidate_record=candidate_record, anchor=anchor,
        )
        this_candidate = _receipt_identity(cand_receipt)
        this_anchor = _receipt_identity(anch_receipt)
        if candidate_identity is None:
            candidate_identity, anchor_identity = this_candidate, this_anchor
        elif candidate_identity != this_candidate or anchor_identity != this_anchor:
            _refuse(f"{raw_path} mixes candidate or anchor execution identities")

        attestations = _list(raw.get("claim_attestations"),
                             f"{raw_path}.claim_attestations", nonempty=True)
        if any(not isinstance(row, Mapping) or row.get("claim_id") != claim_id
               or row.get("outcome") != schemas.PASS for row in attestations):
            _refuse(f"{raw_path} contains a missing, foreign, or non-PASS claim attestation")

        raw_blocks = _list(raw.get("blocks"), f"{raw_path}.blocks", nonempty=True)
        for local_index, block_value in enumerate(raw_blocks):
            block_path = f"{raw_path}.blocks[{local_index}]"
            block_record = _mapping(block_value, block_path)
            if block_record.get("complete") is not True or block_record.get("refusals") != []:
                _refuse(f"{block_path} is incomplete or refused")
            paired = _parse_paired_block(block_record.get("paired_block"),
                                         f"{block_path}.paired_block")
            if paired.block_index in seen_indexes:
                _refuse(f"{block_path} repeats scored block index {paired.block_index}")
            seen_indexes.add(paired.block_index)
            unit_ids.add(paired.unit_id)
            plan = _mapping(block_record.get("plan"), f"{block_path}.plan")
            if plan.get("block_index") != paired.block_index or plan.get("unit_id") != paired.unit_id:
                _refuse(f"{block_path}.plan does not match its paired block")
            observed_anchor = _invocation_samples(
                block_record, arm="anchor", model_path=model["path"],
                receipt_identity=this_anchor, path=block_path,
            )
            observed_candidate = _invocation_samples(
                block_record, arm="candidate", model_path=model["path"],
                receipt_identity=this_candidate, path=block_path,
            )
            if observed_anchor != paired.anchor_samples or observed_candidate != paired.candidate_samples:
                _refuse(f"{block_path} invocation samples do not match paired_block")
            blocks.append(paired)
        vector_digests.append(_content_sha256(raw))

    blocks.sort(key=lambda row: row.block_index)
    if len(unit_ids) != 1:
        _refuse(f"{path} mixes {len(unit_ids)} model/recipe unit ids")
    canonical_blocks = [block.to_list() for block in blocks]
    performance = _mapping(t2_event.get("performance"), f"{path}.t2_event.performance")
    if performance.get("raw_samples") != canonical_blocks:
        _refuse(f"{path}.t2_event.raw_samples do not exactly match the raw vectors")
    if performance.get("paired_blocks") != len(blocks) or grammar.get("reps") != len(blocks):
        _refuse(f"{path}.t2_event scored-block denominator does not match the raw vectors")

    return {
        "anchor_samples": tuple(sample for block in blocks for sample in block.anchor_samples),
        "candidate_samples": tuple(sample for block in blocks for sample in block.candidate_samples),
        "block_count": len(blocks),
        "samples_per_block": [
            {"block_index": block.block_index,
             "anchor": len(block.anchor_samples), "candidate": len(block.candidate_samples)}
            for block in blocks
        ],
        "unit_id": next(iter(unit_ids)),
        "recipe_id": expected["recipe_id"],
        "protocol_id": grammar["protocol_id"],
        "t1_event_id": t1_event["event_id"],
        "t1_event_sha256": _content_sha256(t1_event),
        "t2_event_id": t2_event["event_id"],
        "t2_event_sha256": _content_sha256(t2_event),
        "raw_vector_sha256s": vector_digests,
        "candidate_execution": candidate_identity,
        "anchor_execution": anchor_identity,
    }


def _measurement_row(
    *,
    lane: str,
    arm: str,
    samples: Sequence[float],
    lane_result: Mapping[str, Any],
    model: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    anchor: Mapping[str, Any],
    claim: Mapping[str, Any],
    source_receipt_sha256: str,
    producer_sha256: str,
) -> dict:
    median = float(std_statistics.median(samples))
    metric = LANES[lane]["metric"]
    row = {
        "measurement_id": f"iq2_xxs_model_{lane}_{arm}_median_tokens_per_s",
        "metric": f"iq2_xxs_model_{metric}",
        "value": median,
        "unit": "tokens/s",
        "metric_direction": "higher_better",
        "category": "BASELINE" if arm == "anchor" else "CANDIDATE",
        "reps": len(samples),
        "reps_basis": (
            f"scored:{len(samples)} llama-bench samples from "
            f"{lane_result['block_count']} admitted matched paired blocks"
        ),
        "claim": (
            f"{model['model_id']} {QUANTIZATION} {arm} median "
            f"{metric} is {median:.9g} tokens/s"
        ),
        "extra": {
            "measurement_role": "model_level_confirmation",
            "lane": lane,
            "arm": arm,
            "reduction": "median_of_all_scored_arm_samples",
            "model_identity": dict(model),
            "candidate_identity": {
                "candidate_id": candidate_record["candidate_id"],
                "candidate_record_sha256": _content_sha256(candidate_record),
                "source_commit": candidate_record["worktree"]["source_commit"],
                "source_snapshot_sha256": candidate_record["source_snapshot"]["snapshot_sha256"],
                "patch_bundle_sha256": candidate_record["source_snapshot"]["patch_bundle_sha256"],
                "build_identity_sha256": _content_sha256(candidate_record["build"]),
                "build_log_sha256": candidate_record["build"]["log_sha256"],
                "binary_sha256": candidate_record["artifacts"]["binary_sha256"],
                "linkage_sha256": candidate_record["artifacts"]["linkage_sha256"],
            },
            "anchor_identity": dict(anchor),
            "candidate_execution": dict(lane_result["candidate_execution"]),
            "anchor_execution": dict(lane_result["anchor_execution"]),
            "recipe_id": lane_result["recipe_id"],
            "evaluation_protocol_id": lane_result["protocol_id"],
            "unit_id": lane_result["unit_id"],
            "scored_blocks": lane_result["block_count"],
            "samples_per_block": lane_result["samples_per_block"],
            "resource_claim_receipt": claim["claim_id"],
            "resource_claim_receipt_sha256": _content_sha256(claim),
            "cpu_list": claim["cpu_list"],
            "claim_released_at": claim["released_at"],
            "t1_event_id": lane_result["t1_event_id"],
            "t1_event_sha256": lane_result["t1_event_sha256"],
            "t2_event_id": lane_result["t2_event_id"],
            "t2_event_sha256": lane_result["t2_event_sha256"],
            "raw_vector_sha256s": list(lane_result["raw_vector_sha256s"]),
            "source_receipt_sha256": source_receipt_sha256,
            "producer_id": PRODUCER_ID,
            "producer_sha256": producer_sha256,
        },
    }
    row["extra"]["self_sha256"] = _content_sha256(row)
    return row


def finalize_receipt(receipt: Any, *, producer_sha256: str | None = None) -> dict:
    """Return a finalized copy or raise :class:`ReceiptRefused`.

    The caller writes the returned object atomically.  ``receipt`` is never
    mutated, which lets a runner retain the exact pre-finalization source digest.
    """
    receipt = _mapping(receipt, "receipt")
    if receipt.get("schema") != SCHEMA:
        _refuse(f"receipt.schema must be {SCHEMA!r}")
    if receipt.get("status") != "complete":
        _refuse("receipt.status must be 'complete'")
    if "belief_measurements" in receipt:
        _refuse("receipt already carries belief_measurements; finalization is write-once")
    for forbidden in ("source_receipt_sha256", "producer_id", "producer_sha256", "self_sha256"):
        if forbidden in receipt:
            _refuse(f"unfinalized receipt must not predeclare {forbidden}")

    campaign_id = _text(receipt.get("campaign_id"), "campaign_id")
    candidate_id = _text(receipt.get("candidate_id"), "candidate_id")
    claim = _released_claim(receipt, campaign_id=campaign_id)
    candidate = _candidate_record(
        receipt, campaign_id=campaign_id, candidate_id=candidate_id,
        claim_id=claim["claim_id"],
    )
    model = _model_identity(receipt)
    anchor = _mapping(receipt.get("anchor_identity"), "anchor_identity")
    for key in ("source_commit", "binary_sha256", "linkage_sha256"):
        if key == "source_commit":
            value = _text(anchor.get(key), f"anchor_identity.{key}")
            if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
                _refuse("anchor_identity.source_commit must be a full lowercase commit")
        else:
            _sha256(anchor.get(key), f"anchor_identity.{key}")
    _list(anchor.get("measurement_event_ids"), "anchor_identity.measurement_event_ids",
          nonempty=True)

    lanes = _mapping(receipt.get("lanes"), "lanes")
    if set(lanes) != set(LANES):
        _refuse(f"lanes must contain exactly {sorted(LANES)}")
    lane_results = {
        lane: _lane_samples(
            lane, lanes[lane], campaign_id=campaign_id, candidate_id=candidate_id,
            candidate_record=candidate, model=model, anchor=anchor,
            claim=claim,
        )
        for lane in LANES
    }

    event_ids = {
        result[key] for result in lane_results.values()
        for key in ("t1_event_id", "t2_event_id")
    }
    if not event_ids.issubset(set(candidate["evaluation_event_ids"])):
        _refuse("candidate_record.evaluation_event_ids omits evidence consumed by the producer")
    if lane_results["tg"]["candidate_execution"] \
            != lane_results["pp"]["candidate_execution"]:
        _refuse("TG and PP used different candidate execution identities")
    if lane_results["tg"]["anchor_execution"] != lane_results["pp"]["anchor_execution"]:
        _refuse("TG and PP used different anchor execution identities")

    source_digest = _content_sha256(receipt)
    producer_digest = _sha256(
        producer_sha256 if producer_sha256 is not None else _producer_sha256(),
        "producer_sha256",
    )
    measurements = []
    for lane, result in lane_results.items():
        for arm, key in (("anchor", "anchor_samples"), ("candidate", "candidate_samples")):
            measurements.append(_measurement_row(
                lane=lane, arm=arm, samples=result[key], lane_result=result,
                model=model, candidate_record=candidate, anchor=anchor, claim=claim,
                source_receipt_sha256=source_digest, producer_sha256=producer_digest,
            ))

    finalized = copy.deepcopy(dict(receipt))
    finalized.update({
        "source_receipt_sha256": source_digest,
        "producer_id": PRODUCER_ID,
        "producer_sha256": producer_digest,
        "belief_measurements": measurements,
    })
    finalized["self_sha256"] = _content_sha256(finalized)
    return finalized


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", required=True, type=Path,
                        help="unfinalized prospective model-confirmation receipt")
    result.add_argument("--output", required=True, type=Path,
                        help="new finalized receipt; an existing path is refused")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        finalized = finalize_receipt(source)
        write_json_atomic(args.output, finalized)
    except (OSError, json.JSONDecodeError, ReceiptRefused, ValueError) as exc:
        print(f"IQ2 model-belief finalization REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "receipt": str(args.output),
        "self_sha256": finalized["self_sha256"],
        "measurements": len(finalized["belief_measurements"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
