from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import yaml

MODULE_PATH = Path(__file__).with_name("xmas_function_axis_sweep.py")
SPEC = importlib.util.spec_from_file_location("xmas_function_axis_sweep", MODULE_PATH)
assert SPEC is not None
xmas_sweep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(xmas_sweep)


def _manifest() -> dict:
    domain_task_sets = {
        f"{domain}_v1": [f"{domain}_task_0", f"{domain}_task_1"]
        for domain in xmas_sweep.XMAS_DOMAINS
    }
    cells = {}
    for domain in xmas_sweep.XMAS_DOMAINS:
        cells[domain] = {}
        for function in xmas_sweep.XMAS_FUNCTIONS:
            cells[domain][function] = {
                "task_ids_ref": f"{domain}_v1",
                "prompt_wrapper": "verify_answer"
                if function == "verify"
                else "solve_direct",
                "scoring_family": "binary_validity"
                if function == "verify"
                else "source_auto",
                "failure_policy": "answer_tag_valid_invalid"
                if function == "verify"
                else "source_parse_then_score",
            }
    return {
        "version": "test",
        "capture_profiles": {
            "default": {},
            "qwen_nothink": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "models": {
            "frontdoor": {
                "url": "http://127.0.0.1:8070/v1",
                "capture_profile": "qwen_nothink",
            },
            "worker_general": {
                "url": "http://127.0.0.1:8072/v1",
                "capture_profile": "default",
            },
        },
        "domain_task_sets": domain_task_sets,
        "cells": cells,
    }


def _question_pool() -> dict[str, dict]:
    out = {}
    for domain in xmas_sweep.XMAS_DOMAINS:
        for idx in range(2):
            task_id = f"{domain}_task_{idx}"
            out[task_id] = {
                "id": task_id,
                "suite": domain,
                "prompt": f"Prompt {task_id}",
                "expected": f"expected-{idx}",
                "scoring_method": "exact_match",
            }
    return out


def test_manifest_validation_requires_all_cells() -> None:
    manifest = _manifest()
    manifest["cells"]["math"].pop("solve")

    try:
        xmas_sweep.validate_manifest(manifest)
    except ValueError as exc:
        assert "manifest missing cells.math.solve" in str(exc)
    else:
        raise AssertionError("expected missing cell failure")


def test_build_requests_emits_25_cells() -> None:
    requests = xmas_sweep.build_requests(_manifest(), _question_pool())

    assert len(requests) == 5 * 5 * 2
    assert {row["cell"] for row in requests} == {
        f"{domain}:{function}"
        for domain in xmas_sweep.XMAS_DOMAINS
        for function in xmas_sweep.XMAS_FUNCTIONS
    }
    verify = next(row for row in requests if row["cell"] == "math:verify")
    assert verify["expected"] == "valid"
    assert "Expected answer:" in verify["prompt"]
    assert verify["model_capture_profiles"]["frontdoor"]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }


def test_filter_requests_selects_precise_slices() -> None:
    requests = xmas_sweep.build_requests(_manifest(), _question_pool())

    assert len(xmas_sweep.filter_requests(requests, cell="math:verify")) == 2
    assert {
        row["cell"]
        for row in xmas_sweep.filter_requests(requests, domain="code")
    } == {f"code:{function}" for function in xmas_sweep.XMAS_FUNCTIONS}
    assert {
        row["cell"]
        for row in xmas_sweep.filter_requests(requests, function="extract")
    } == {f"{domain}:extract" for domain in xmas_sweep.XMAS_DOMAINS}
    assert len(
        xmas_sweep.filter_requests(
            requests,
            cell="math:solve",
            source_task_id="math_task_1",
        )
    ) == 1


def test_filter_requests_rejects_invalid_cell() -> None:
    requests = xmas_sweep.build_requests(_manifest(), _question_pool())

    try:
        xmas_sweep.filter_requests(requests, cell="math")
    except ValueError as exc:
        assert "domain:function" in str(exc)
    else:
        raise AssertionError("expected invalid cell failure")


def test_summarize_results_outputs_function_axis_table() -> None:
    rows = []
    for domain in xmas_sweep.XMAS_DOMAINS:
        for function in xmas_sweep.XMAS_FUNCTIONS:
            for model_id, correct, wall_s in [
                ("frontdoor", False, 1.0),
                ("worker_general", True, 2.0),
            ]:
                rows.append({
                    "request_id": f"{domain}:{function}:sample",
                    "domain": domain,
                    "function": function,
                    "model_id": model_id,
                    "correct": correct,
                    "ok": True,
                    "wall_s": wall_s,
                })

    summary = xmas_sweep.summarize_results(rows)

    assert summary["derivation_mode"] == "function_axis_sweep"
    assert summary["cell_winners"]["math:solve"] == "worker_general"
    assert summary["table"]["math"]["solve"]["worker_general"]["correct"] == 1
    assert summary["table"]["math"]["solve"]["worker_general"]["total"] == 1


def test_summarize_results_allows_partial_smoke_rows() -> None:
    rows = [
        {
            "request_id": "math:solve:sample",
            "domain": "math",
            "function": "solve",
            "model_id": "frontdoor",
            "correct": True,
            "ok": True,
            "wall_s": 1.0,
        }
    ]

    summary = xmas_sweep.summarize_results(rows, require_complete=False)

    assert summary["cell_winners"] == {"math:solve": "frontdoor"}
    assert summary["table"]["math"]["solve"]["frontdoor"]["correct"] == 1
    assert summary["table"]["math"].get("verify") is None


def test_score_response_handles_source_binary_and_plan_cells() -> None:
    source_request = {
        "scoring_family": "source_auto",
        "source_scoring_method": "exact_match",
        "expected": "42",
    }
    assert xmas_sweep.score_response(source_request, "<answer>42</answer>") == (
        True,
        "",
    )

    multiple_choice_request = {
        "scoring_family": "source_auto",
        "source_scoring_method": "multiple_choice",
        "expected": "C",
    }
    assert xmas_sweep.score_response(
        multiple_choice_request,
        "I considered B while reasoning.\n<answer>C</answer>",
    ) == (True, "")

    verify_request = {
        "scoring_family": "binary_validity",
        "expected": "valid",
    }
    assert xmas_sweep.score_response(
        verify_request,
        "<answer>valid</answer> because it matches.",
    ) == (True, "")
    assert xmas_sweep.score_response(verify_request, "maybe") == (
        False,
        "parse_failure",
    )

    plan_request = {"scoring_family": "rubric", "expected": ""}
    assert xmas_sweep.score_response(
        plan_request,
        "1. Read the task\n2. Compute the result\n3. Format the answer",
    ) == (True, "")
    assert xmas_sweep.score_response(plan_request, "do it") == (
        False,
        "rubric_unscored",
    )


def test_run_requests_rows_are_summarizable(monkeypatch) -> None:
    async def fake_query_model(client, request, model_id, model_cfg, *, timeout_s=600.0):
        return {
            "ok": True,
            "wall_s": 1.0 if model_id == "frontdoor" else 2.0,
            "answer": "<answer>expected-0</answer>",
            "prompt_tokens": 10,
            "completion_tokens": 3,
        }

    monkeypatch.setattr(xmas_sweep, "query_model", fake_query_model)
    request = xmas_sweep.build_requests(_manifest(), _question_pool())[0]

    rows = __import__("asyncio").run(xmas_sweep.run_requests([request]))

    assert len(rows) == 2
    assert {row["model_id"] for row in rows} == {"frontdoor", "worker_general"}
    assert rows[0]["domain"] == "math"
    summary = xmas_sweep.summarize_results(rows)
    assert summary["cell_winners"]["math:solve"] == "frontdoor"


def test_completed_result_keys_reads_existing_rows(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps({"request_id": "math:solve:a", "model_id": "frontdoor"}) + "\n",
        encoding="utf-8",
    )

    assert xmas_sweep.completed_result_keys(results) == {
        ("math:solve:a", "frontdoor")
    }


def test_post_json_uses_curl_wall_timeout(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps({"ok": True}))

    monkeypatch.setattr(xmas_sweep.subprocess, "run", fake_run)

    assert xmas_sweep._post_json("http://example.test/v1/chat/completions", {"x": 1}, 7.5) == {
        "ok": True,
    }
    cmd, kwargs = calls[0]
    assert cmd[:3] == ["curl", "--silent", "--show-error"]
    assert "--max-time" in cmd
    assert "7.500" in cmd
    assert kwargs["timeout"] == 9.5


def test_cli_emit_requests_and_summary(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest()), encoding="utf-8")
    pool_path = tmp_path / "pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as fh:
        for row in _question_pool().values():
            fh.write(json.dumps(row) + "\n")
    requests_path = tmp_path / "requests.jsonl"

    assert xmas_sweep.main.__module__ == "xmas_function_axis_sweep"
    requests = xmas_sweep.build_requests(
        xmas_sweep.load_manifest(manifest_path),
        xmas_sweep.load_question_pool(pool_path),
    )
    xmas_sweep.write_jsonl(requests_path, requests)
    assert len(requests_path.read_text(encoding="utf-8").splitlines()) == 50
