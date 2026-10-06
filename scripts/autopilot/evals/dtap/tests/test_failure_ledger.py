"""Synthetic controls for the additive DTAP native failure ledger; no endpoint calls."""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from harness import cli, runner
from harness.endpoint import ChatEndpoint
from harness.failure_ledger import (
    build_ledger,
    normalize_comparison_pairs,
    summarize_response_usage,
    unit_record,
)
from harness.outcomes import EndpointFailure
from harness.runner import RunResult

DTAP_DIR = pathlib.Path(__file__).resolve().parent.parent
BENIGN_FIXTURE = DTAP_DIR / "fixtures" / "finance-benign-trade-execution-001.done.json"


def test_endpoint_usage_requires_complete_nonnegative_integer_pair():
    parse = ChatEndpoint("http://fixture")._parse
    def response(usage_marker=...):
        raw = {"choices": [{"message": {"content": "ok"}}]}
        if usage_marker is not ...:
            raw["usage"] = usage_marker
        return parse(raw)

    valid = response({"prompt_tokens": 4, "completion_tokens": 7, "total_tokens": 11})
    assert valid.usage == {"prompt_tokens": 4, "completion_tokens": 7}
    assert valid.usage_status == "reported"
    assert response().usage is None and response().usage_status == "unavailable:usage_missing"
    assert response({"prompt_tokens": 4}).usage is None
    assert "completion_tokens" in response({"prompt_tokens": 4}).usage_status
    for invalid in (
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": False},
        {"prompt_tokens": -1, "completion_tokens": 1},
        {"prompt_tokens": 1.0, "completion_tokens": 1},
        {"prompt_tokens": "1", "completion_tokens": 1},
        [],
        None,
    ):
        result = response(invalid)
        assert result.usage is None
        assert result.usage_status.startswith("unavailable:")


def _run(case_id, arm, seed, *, status="ok", task=None, attack=None, failure=None,
         completion="judged", trace_id=None, trace_path=None, endpoint="native_transport"):
    return RunResult(
        case_id=case_id,
        arm=arm,
        seed=seed,
        status=status,
        failure=failure,
        task_success=task,
        attack_success=attack,
        trace_id=trace_id,
        trace_path=trace_path,
        elapsed_s=0.25,
        completion_state=completion,
        endpoint_kind=endpoint,
    )


def _unit(run, threat):
    return unit_record(
        run.to_dict(),
        threat=threat,
        trace_sha256="a" * 64,
        trace_binding_status="verified_closed_trace",
        usage=summarize_response_usage(()),
    )


def test_still_failing_uses_threat_specific_primary_metric_and_null_is_unresolved():
    assert _unit(_run("b", "done", 0, task=False), "benign")["still_failing"] is True
    assert _unit(_run("b", "done", 1, task=True), "benign")["still_failing"] is False
    assert _unit(_run("a", "compliant", 0, attack=True), "direct")["still_failing"] is True
    assert _unit(_run("a", "compliant", 1, attack=False), "indirect")["still_failing"] is False
    unresolved = _unit(_run("b", "done", 2, task=None), "benign")
    assert unresolved["still_failing"] is None
    assert unresolved["still_failing_basis"] == "native_primary_value_unresolved"
    failed = _unit(_run("b", "done", 3, status="failed", task=None,
                        failure={"type": "endpoint", "message": "x", "detail": {}}), "benign")
    assert failed["still_failing"] is True
    assert failed["still_failing_basis"] == "native_run_status_failed"


@pytest.mark.parametrize("native_type", ["endpoint", "overflow", "model", "parser", "harness"])
def test_native_failure_type_is_preserved_not_regraded(native_type):
    failure = {"type": native_type, "message": "native fixture failure", "detail": {"fixture": True}}
    unit = _unit(_run("a", "compliant", 0, status="failed", failure=failure,
                      completion="other_error"), "direct")
    assert unit["status"] == "failed"
    assert unit["failure"] == failure
    assert unit["completion_state"] == "other_error"


def test_usage_totals_keep_unknown_and_retry_coverage_explicit():
    no_usage = summarize_response_usage([
        {"usage": None, "usage_status": "unavailable:usage_missing"},
    ])
    assert no_usage["reported_prompt_tokens"] is None
    assert no_usage["reported_completion_tokens"] is None
    assert no_usage["reported_total_tokens"] is None
    assert no_usage["coverage"]["unavailable_reasons"] == {
        "unavailable:usage_missing": 1,
    }
    inconsistent = summarize_response_usage([
        {"usage": {"prompt_tokens": 8, "completion_tokens": 3},
         "usage_status": "unavailable:usage_invalid_fields:prompt_tokens"},
    ])
    assert inconsistent["reported_total_tokens"] is None
    assert inconsistent["coverage"]["unavailable_reasons"] == {
        "unavailable:usage_payload_inconsistent_with_reported_status": 1,
    }

    partial = summarize_response_usage([
        {"usage": {"prompt_tokens": 5, "completion_tokens": 2},
         "usage_status": "reported", "transport_detail": {"attempts": 2}},
        {"usage": None, "usage_status": "unavailable:usage_missing"},
    ])
    assert partial["reported_prompt_tokens"] == 5
    assert partial["reported_completion_tokens"] == 2
    assert partial["reported_total_tokens"] == 7
    assert partial["coverage"]["scope"] == "successful_endpoint_responses_only"
    assert partial["coverage"]["total_attempt_cost_complete"] is False
    assert "endpoint_retries_have_no_attempt_level_usage" in partial["coverage"]["incomplete_reasons"]

    complete = summarize_response_usage([
        {"usage": {"prompt_tokens": 0, "completion_tokens": 0},
         "usage_status": "reported", "transport_detail": {"attempts": 1}},
    ])
    assert complete["reported_total_tokens"] == 0
    assert complete["coverage"]["total_attempt_cost_complete"] is True


def test_directional_pairs_cover_gain_giveback_missing_and_native_null():
    benign = [
        _unit(_run("b", "done", 0, task=False), "benign"),
        _unit(_run("b", "none", 0, task=True), "benign"),
        _unit(_run("b", "done", 1, task=True), "benign"),
        _unit(_run("b", "none", 1, task=False), "benign"),
        _unit(_run("b", "done", 2, task=None), "benign"),
        _unit(_run("b", "none", 2, task=True), "benign"),
        _unit(_run("b", "done", 3, task=False), "benign"),
    ]
    result = build_ledger(
        benign,
        case_ids=["b"],
        seeds=[0, 1, 2, 3],
        threats={"b": "benign"},
        comparison_pairs=[("done", "none")],
    )
    counts = result["comparisons"][0]["pair_counts"]
    assert counts["gain"] == 1
    assert counts["giveback"] == 1
    assert counts["native_primary_null"] == 1
    assert counts["missing_pair_match"] == 1
    assert result["comparisons"][0]["pairs"][0]["state"] == "gain"

    attack = [
        _unit(_run("a", "base", 0, attack=True), "direct"),
        _unit(_run("a", "candidate", 0, attack=False), "direct"),
        _unit(_run("a", "base", 1, attack=False), "indirect"),
        _unit(_run("a", "candidate", 1, attack=True), "indirect"),
    ]
    result = build_ledger(
        attack,
        case_ids=["a"],
        seeds=[0, 1],
        threats={"a": "direct"},
        comparison_pairs=[("base", "candidate")],
    )
    assert result["comparisons"][0]["pair_counts"]["gain"] == 1
    assert result["comparisons"][0]["pair_counts"]["giveback"] == 1
    assert result["comparisons"][0]["pairs"][0]["metric_direction"] == "lower_better"


def test_pairs_are_explicit_unique_and_declared():
    assert normalize_comparison_pairs(["base", "candidate"], None) is None
    assert normalize_comparison_pairs(["base", "candidate"], [("base", "candidate")]) == (
        ("base", "candidate"),
    )
    for arms, pairs in (
        (["base"], [("base", "missing")]),
        (["base", "candidate"], [("base", "candidate"), ("base", "candidate")]),
        (["same"], [("same", "same")]),
    ):
        with pytest.raises(ValueError):
            normalize_comparison_pairs(arms, pairs)
    with pytest.raises(ValueError):
        normalize_comparison_pairs([[]], [("base", "candidate")])
    with pytest.raises(SystemExit):
        cli._comparison_pairs_from(["base", "candidate"])
    with pytest.raises(SystemExit):
        cli._comparison_pairs_from(["base:candidate", "base:candidate"])


def test_no_producer_fields_stay_explicitly_unmeasured():
    unit = _unit(_run("b", "done", 0, task=False), "benign")
    assert unit["failed_verification"]["state"] == "unmapped"
    assert unit["no_valid_tool_call"]["state"] == "unmeasured"
    assert unit["first_try_valid_call"]["state"] == "unmeasured"
    assert unit["repair_recovery"]["state"] == "unmeasured"


def test_matrix_keeps_primary_summary_and_native_timeout_receipt_semantics(
    tmp_path,
):
    case_id = "finance-benign-trade-execution-001"

    class Endpoint:
        def complete(self, *args, **kwargs):
            raise EndpointFailure(
                "fixture endpoint timeout",
                detail={"terminal_native_timeout": True, "timeout_attempts": 1,
                        "attempts": 1, "cap_scope": "endpoint_request", "request_timeout_s": 0.1},
            )

    rows = runner.run_matrix(
        [case_id], ["done"], [0], lambda *_: Endpoint(),
        results_dir=tmp_path / "runs",
        native_capture_root=tmp_path / "private-native",
        capture_applicability={"scope": "synthetic fault endpoint; no network", "mode": "synthetic"},
        capture_fixture_paths=[BENIGN_FIXTURE],
    )
    original = json.loads((tmp_path / "runs" / "matrix.json").read_text())
    assert original == rows
    row = rows[f"{case_id}::done"]
    assert set(row) == {
        "case_id", "arm", "threat", "seeds", "n", "successes", "failed", "rate",
        "ci95", "typed_failures", "timeout_reporting",
    }
    assert row["timeout_reporting"]["total"] == 1
    assert row["timeout_reporting"]["terminal_timeouts"] == 1
    ledger = json.loads((tmp_path / "runs" / "failure_ledger.json").read_text())
    unit = ledger["units"][0]
    assert unit["status"] == "failed"
    assert unit["completion_state"] == "terminal_timeout"
    assert unit["failure"]["type"] == "endpoint"
    assert unit["trace_binding_status"] == "verified_closed_trace"
    assert unit["trace_sha256"] == hashlib.sha256(pathlib.Path(unit["trace_path"]).read_bytes()).hexdigest()
    assert unit["usage"]["reported_total_tokens"] is None
    assert unit["still_failing"] is True

    receipt_paths = list((tmp_path / "private-native").glob("*/receipt.json"))
    assert len(receipt_paths) == 1
    receipt = json.loads(receipt_paths[0].read_text())
    assert receipt["timeout_reporting_integrity"] is True
    assert receipt["decided_proposition"].endswith("true.")
