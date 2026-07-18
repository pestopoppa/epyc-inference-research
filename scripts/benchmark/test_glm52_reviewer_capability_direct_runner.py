#!/usr/bin/env python3
"""Tests for glm52_reviewer_capability_direct_runner.py.

All tests are inference-free. They cover dry-run planning, channel extraction,
prompt shaping, and reuse of the orchestrator GC-1 scorer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent / "glm52_reviewer_capability_direct_runner.py"
_SPEC = importlib.util.spec_from_file_location("glm52_reviewer_capability_direct_runner", _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_reviewer_capability_direct_runner"] = runner
_SPEC.loader.exec_module(runner)


def test_parse_args_defaults_to_small_strict_if_smoke(tmp_path):
    args = runner.parse_args(["--probe", "strict_if", "--output-dir", str(tmp_path)])
    assert args.execute is False
    assert args.m == runner.DEFAULT_SMOKE_M
    assert args.k == runner.DEFAULT_SMOKE_K
    assert args.lanes == "grammar,free"
    assert args.band == "p2168_tk4096"
    assert args.max_tokens == 256
    assert args.trace_logs is True
    assert args.prompt_style == "context_fill"


def test_rubric_default_keeps_schema_and_free_lanes(tmp_path):
    args = runner.parse_args(["--probe", "rubric_authoring", "--output-dir", str(tmp_path)])
    assert args.lanes == "grammar,free"
    assert args.max_tokens == 256


def test_why_default_lane_is_free(tmp_path):
    args = runner.parse_args(["--probe", "why_diagnosis", "--output-dir", str(tmp_path)])
    assert args.lanes == "free"
    assert args.max_tokens == 192


def test_prompt_for_strict_if_names_review_decision_contract():
    prompt, answer = runner.prompt_for_task(
        "strict_if",
        {"task_id": "si-0", "prompt": "candidate has missing tests"},
        grammar_constrained=True,
    )
    assert "ReviewDecision JSON object" in prompt
    assert "blocking.tripwire" in prompt
    assert "confidence must be a JSON number" in prompt
    assert "never null" in prompt
    assert "candidate has missing tests" in prompt
    assert "Emit only the JSON object" in answer


def test_build_natural_prompt_does_not_inject_dsa_filler():
    prompt_info = runner.build_natural_prompt(
        task_line="Review candidate X.",
        context_length=4096,
        max_completion_tokens=256,
        prompt_context_guard_tokens=128,
        token_counter=lambda prompt: len(prompt.split()),
        answer_instruction="Emit JSON only.",
    )
    assert prompt_info["prompt_style"] == "natural"
    assert prompt_info["prompt_token_min"] == 0
    assert "Review candidate X." in prompt_info["prompt"]
    assert "The GLM DSA probe keeps the context deterministic" not in prompt_info["prompt"]


def test_response_text_prefers_message_content_over_reasoning():
    response = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "thinking text",
                    "content": '{"decision":"approve","confidence":0.9,"blocking":{"tripwire":false}}',
                }
            }
        ]
    }
    assert runner.response_text_for_scoring(response).startswith('{"decision":"approve"')


def test_server_extra_args_add_schema_only_for_strict_grammar_lane():
    grammar_args = runner.server_extra_args_for_lane("strict_if", "grammar")
    assert "--json-schema" in grammar_args
    schema = grammar_args[grammar_args.index("--json-schema") + 1]
    assert '"decision"' in schema
    free_args = runner.server_extra_args_for_lane("strict_if", "free")
    assert "--json-schema" not in free_args
    rubric_schema_args = runner.server_extra_args_for_lane("rubric_authoring", "grammar")
    assert "--json-schema" in rubric_schema_args
    assert '"rubric_id"' in rubric_schema_args[rubric_schema_args.index("--json-schema") + 1]
    rubric_free_args = runner.server_extra_args_for_lane("rubric_authoring", "free")
    assert "--json-schema" not in rubric_free_args
    assert "--reasoning" in rubric_free_args


def test_score_lane_strict_if_reuses_orchestrator_model_indexed_scorer():
    tasks = [
        {"task_id": "si-0", "prompt": "review 0"},
        {"task_id": "si-1", "prompt": "review 1"},
        {"task_id": "si-2", "prompt": "review 2"},
    ]
    task_results = [
        {
            "task_id": "si-0",
            "scoring_text": '{"decision":"approve","confidence":0.9,"blocking":{"tripwire":false}}',
        },
        {
            "task_id": "si-1",
            "scoring_text": '{"decision":"reject","confidence":0.2,"blocking":{"tripwire":true}}',
        },
        {"task_id": "si-2", "scoring_text": "APPROVE"},
    ]
    score = runner.score_lane("strict_if", tasks, task_results, lane="grammar", k=2, m=3)
    assert score["model_key"] == "glm_52_ud_iq2m"
    assert score["quant"] == "UD-IQ2_M"
    assert score["n_valid"] == 2
    assert score["emission_rate"] == pytest.approx(2 / 3)
    assert score["passed"] is True


def test_build_plan_records_topk_schedule_without_inference(tmp_path, monkeypatch):
    args = runner.parse_args(
        [
            "--probe",
            "strict_if",
            "--output-dir",
            str(tmp_path),
            "--m",
            "2",
            "--k",
            "1",
            "--prompt-style",
            "natural",
        ]
    )

    monkeypatch.setattr(runner.base, "resolve_binary", lambda path: Path("/tmp/llama-server"))
    monkeypatch.setattr(runner.base, "resolve_library_path", lambda binary, library_path: Path("/tmp"))
    monkeypatch.setattr(
        runner.base,
        "collect_inventory",
        lambda model_dir: {
            "status": "ready",
            "primary_shard": "/tmp/glm.gguf",
            "refusal_reasons": [],
        },
    )
    monkeypatch.setattr(runner, "pgrep", lambda pattern: [])
    plan = runner.build_plan(args)
    assert plan["execution_allowed"] is True
    assert plan["n_tasks"] == 2
    assert [lane["lane"] for lane in plan["lanes"]] == ["grammar", "free"]
    assert plan["lanes"][0]["band"]["indexer_top_k"] == 4096
    assert plan["lanes"][0]["server"]["context_length"] == 4096
    assert "--json-schema" in plan["lanes"][0]["server"]["server_command"]
    assert "--json-schema" not in plan["lanes"][1]["server"]["server_command"]
    assert plan["lanes"][0]["request"]["prompt_style"] == "natural"
