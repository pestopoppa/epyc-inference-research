"""Additive native failure and explicit-pair observability for DTAP matrices.

This report projects native RunResult facts and provider-reported successful
response usage. It is not a grade, ClaimTuple, or replacement for matrix.json.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCHEMA = "epyc.dtap.failure_ledger.v1"


def normalize_comparison_pairs(
    arms: Sequence[str],
    comparison_pairs: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...] | None:
    """Validate explicit baseline:candidate pairs; never infer a pairing."""
    if comparison_pairs is None:
        return None
    if any(not isinstance(arm, str) or not arm for arm in arms):
        raise ValueError("declared arms must be nonempty strings")
    if len(set(arms)) != len(arms):
        raise ValueError("comparison pairs require unique declared arms")
    declared = set(arms)
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in comparison_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each comparison pair must name baseline and candidate arms")
        baseline, candidate = pair
        if not isinstance(baseline, str) or not baseline or not isinstance(candidate, str) or not candidate:
            raise ValueError("comparison arms must be nonempty strings")
        if baseline == candidate:
            raise ValueError("comparison baseline and candidate must differ")
        if baseline not in declared or candidate not in declared:
            raise ValueError("comparison pair arms must both be declared in matrix arms")
        normalized_pair = (baseline, candidate)
        if normalized_pair in seen:
            raise ValueError(f"duplicate comparison pair: {baseline}:{candidate}")
        seen.add(normalized_pair)
        normalized.append(normalized_pair)
    return tuple(normalized)


def summarize_response_usage(response_events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum only valid endpoint-reported usage; absent or bad values remain unknown."""
    prompt_total = 0
    completion_total = 0
    reported = 0
    reasons: Counter[str] = Counter()
    retry_seen = False
    attempts_complete = True
    response_count = 0

    for event in response_events:
        response_count += 1
        usage = event.get("usage")
        status = event.get("usage_status")
        if not isinstance(status, str) or not status:
            status = "unavailable: usage status missing"
        valid = isinstance(usage, Mapping)
        if valid:
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            valid = (
                type(prompt) is int
                and prompt >= 0
                and type(completion) is int
                and completion >= 0
            )
        if valid and status == "reported":
            prompt_total += prompt
            completion_total += completion
            reported += 1
        else:
            reason = (
                "unavailable:usage_payload_inconsistent_with_reported_status"
                if status == "reported" or valid else status
            )
            reasons[reason] += 1
        transport = event.get("transport_detail")
        attempts = transport.get("attempts") if isinstance(transport, Mapping) else None
        if type(attempts) is int and attempts > 1:
            retry_seen = True
            attempts_complete = False
        elif type(attempts) is not int or attempts != 1:
            attempts_complete = False

    has_report = reported > 0
    successful_response_usage_complete = response_count > 0 and reported == response_count
    incomplete_reasons: list[str] = []
    if response_count == 0:
        incomplete_reasons.append("no_successful_endpoint_response")
    if reported < response_count:
        incomplete_reasons.append("one_or_more_successful_responses_lacked_valid_usage")
    if retry_seen:
        incomplete_reasons.append("endpoint_retries_have_no_attempt_level_usage")
    if response_count and not retry_seen and not attempts_complete:
        incomplete_reasons.append("endpoint_attempt_count_unavailable")
    incomplete_reasons.append(
        "successful_response_events_do_not_cover_failed_or_unobserved_endpoint_attempts"
    )

    return {
        "reported_prompt_tokens": prompt_total if has_report else None,
        "reported_completion_tokens": completion_total if has_report else None,
        "reported_total_tokens": prompt_total + completion_total if has_report else None,
        "coverage": {
            "source": "endpoint_response.usage",
            "scope": "successful_endpoint_responses_only",
            "response_events": response_count,
            "responses_with_reported_usage": reported,
            "successful_response_usage_complete": successful_response_usage_complete,
            "unavailable_reasons": dict(sorted(reasons.items())),
            "total_attempt_cost_complete": False,
            "incomplete_reasons": incomplete_reasons,
        },
    }


def unit_record(
    run: Mapping[str, Any],
    *,
    threat: str,
    trace_sha256: str | None,
    trace_binding_status: str,
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a returned native run without rewriting its verdict or failure."""
    if threat == "benign":
        primary_name, direction = "task_success", "higher_better"
    else:
        primary_name, direction = "attack_success", "lower_better"
    primary_value = run.get(primary_name)
    if type(primary_value) is not bool:
        primary_value = None

    status = run.get("status")
    native_failure = run.get("failure")
    failure_projection = None
    if isinstance(native_failure, Mapping):
        native_type = native_failure.get("type")
        failure_bytes = json.dumps(
            native_failure, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        detail = native_failure.get("detail")
        timeout_fields: dict[str, Any] = {}
        if isinstance(detail, Mapping):
            terminal_timeout = detail.get("terminal_native_timeout")
            if type(terminal_timeout) is bool:
                timeout_fields["terminal_native_timeout"] = terminal_timeout
            for key in ("timeout_attempts", "attempts"):
                value = detail.get(key)
                if type(value) is int and value >= 0:
                    timeout_fields[key] = value
            cap_scope = detail.get("cap_scope")
            if isinstance(cap_scope, str) and cap_scope in {
                "endpoint_request", "native_transport", "endpoint_attempt"
            }:
                timeout_fields["cap_scope"] = cap_scope
        failure_projection = {
            "type": native_type if isinstance(native_type, str) else None,
            "content_sha256": hashlib.sha256(failure_bytes).hexdigest(),
            "native_timeout_fields": timeout_fields,
        }
    if status == "failed":
        still_failing, still_failing_basis = True, "native_run_status_failed"
    elif primary_value is None:
        still_failing, still_failing_basis = None, "native_primary_value_unresolved"
    elif threat == "benign" and primary_value is False:
        still_failing, still_failing_basis = True, "benign_task_success_false"
    elif threat != "benign" and primary_value is True:
        still_failing, still_failing_basis = True, "nonbenign_attack_success_true"
    else:
        still_failing, still_failing_basis = False, "native_primary_value_desirable"

    return {
        "case_id": run.get("case_id"),
        "arm": run.get("arm"),
        "seed": run.get("seed"),
        "threat": threat,
        "status": status,
        "failure": failure_projection,
        "task_success": run.get("task_success"),
        "attack_success": run.get("attack_success"),
        "primary_metric": {
            "name": primary_name,
            "direction": direction,
            "value": primary_value,
        },
        "still_failing": still_failing,
        "still_failing_basis": still_failing_basis,
        "completion_state": run.get("completion_state"),
        "trace_id": run.get("trace_id"),
        "trace_path": run.get("trace_path"),
        "trace_sha256": trace_sha256,
        "trace_binding_status": trace_binding_status,
        "elapsed_s": run.get("elapsed_s"),
        "endpoint_kind": run.get("endpoint_kind"),
        "usage": dict(usage),
        "failed_verification": {
            "state": "unmapped",
            "reason": "DTAP native result does not carry a separate verifier result",
        },
        "no_valid_tool_call": {
            "state": "unmeasured",
            "reason": "case registry exposes no public requires_tool_call field",
        },
        "first_try_valid_call": {
            "state": "unmeasured",
            "reason": "no run-attributable first-try validity event is produced",
        },
        "repair_recovery": {
            "state": "unmeasured",
            "reason": "no run-attributable repair/recovery event is produced",
        },
    }


def compare_units(
    units: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    seeds: Sequence[int],
    threats: Mapping[str, str],
    comparison_pairs: Sequence[tuple[str, str]] | None,
) -> list[dict[str, Any]] | None:
    """Compare only explicit same-case/same-seed pairs in the metric's direction."""
    if comparison_pairs is None:
        return None
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for unit in units:
        key = (unit.get("case_id"), unit.get("arm"), unit.get("seed"))
        if key in indexed:
            raise ValueError(f"duplicate failure-ledger unit identity: {key!r}")
        indexed[key] = unit

    comparisons: list[dict[str, Any]] = []
    for baseline_arm, candidate_arm in comparison_pairs:
        pair_rows = []
        gains = givebacks = unchanged = missing = native_failed = native_null = 0
        for case_id in case_ids:
            threat = threats[case_id]
            primary_name = "task_success" if threat == "benign" else "attack_success"
            direction = "higher_better" if threat == "benign" else "lower_better"
            for seed in seeds:
                baseline = indexed.get((case_id, baseline_arm, seed))
                candidate = indexed.get((case_id, candidate_arm, seed))
                identity = {"case_id": case_id, "seed": seed,
                            "baseline_arm": baseline_arm, "candidate_arm": candidate_arm}
                if baseline is None or candidate is None:
                    missing += 1
                    missing_arms = [arm for arm, value in (
                        (baseline_arm, baseline), (candidate_arm, candidate)
                    ) if value is None]
                    pair_rows.append({**identity, "state": "missing_pair_match",
                                      "missing_arms": missing_arms})
                    continue
                shared = {
                    **identity,
                    "baseline_trace_id": baseline.get("trace_id"),
                    "baseline_trace_sha256": baseline.get("trace_sha256"),
                    "candidate_trace_id": candidate.get("trace_id"),
                    "candidate_trace_sha256": candidate.get("trace_sha256"),
                }
                if baseline.get("status") != "ok" or candidate.get("status") != "ok":
                    native_failed += 1
                    pair_rows.append({**shared, "state": "native_run_failed",
                                      "baseline_status": baseline.get("status"),
                                      "candidate_status": candidate.get("status")})
                    continue
                before = baseline.get(primary_name)
                after = candidate.get(primary_name)
                if type(before) is not bool or type(after) is not bool:
                    native_null += 1
                    pair_rows.append({**shared, "state": "native_primary_null",
                                      "primary_metric": primary_name,
                                      "baseline_value": before, "candidate_value": after})
                    continue
                if threat == "benign":
                    gain = before is False and after is True
                    giveback = before is True and after is False
                else:
                    gain = before is True and after is False
                    giveback = before is False and after is True
                if gain:
                    gains += 1
                    state = "gain"
                elif giveback:
                    givebacks += 1
                    state = "giveback"
                else:
                    unchanged += 1
                    state = "unchanged"
                pair_rows.append({**shared, "state": state,
                                  "primary_metric": primary_name,
                                  "metric_direction": direction,
                                  "baseline_value": before, "candidate_value": after})
        comparisons.append({
            "baseline_arm": baseline_arm,
            "candidate_arm": candidate_arm,
            "primary_metric_by_threat": {
                case_id: ("task_success" if threats[case_id] == "benign" else "attack_success")
                for case_id in case_ids
            },
            "pair_counts": {
                "gain": gains,
                "giveback": givebacks,
                "unchanged": unchanged,
                "missing_pair_match": missing,
                "native_run_failed": native_failed,
                "native_primary_null": native_null,
                "total_declared_case_seed_pairs": len(case_ids) * len(seeds),
            },
            "pairs": pair_rows,
        })
    return comparisons


def build_ledger(
    units: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    seeds: Sequence[int],
    threats: Mapping[str, str],
    comparison_pairs: Sequence[tuple[str, str]] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"schema": SCHEMA, "units": list(units)}
    comparisons = compare_units(
        units,
        case_ids=case_ids,
        seeds=seeds,
        threats=threats,
        comparison_pairs=comparison_pairs,
    )
    if comparisons is not None:
        record["comparisons"] = comparisons
    return record
