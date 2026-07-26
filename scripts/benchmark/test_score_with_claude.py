"""Lossless-input tests for the LLM judge scorer."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("score_with_claude.py")
SPEC = importlib.util.spec_from_file_location("score_with_claude", MODULE_PATH)
assert SPEC and SPEC.loader
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


class FakeResponse:
    def read(self):
        return b'{"choices":[{"message":{"content":"{\\"score\\":3,\\"reason\\":\\"ok\\"}"}}]}'


def capture_judge_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(scorer.urllib.request, "urlopen", fake_urlopen)
    return captured


def current_capture(prompt, response, reasoning=""):
    return {
        "capture_schema_version": scorer.CURRENT_CAPTURE_SCHEMA,
        "runner_source_sha256": hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
        "prompt": prompt,
        "response": response,
        "reasoning": reasoning,
        "request_error": "",
        "prompt_fingerprint": scorer.producer_fingerprint(prompt),
        "response_fingerprint": scorer.producer_fingerprint(response),
        "reasoning_fingerprint": scorer.producer_fingerprint(reasoning),
    }


def write_pinned_questions(tmp_path, suite="suite", question_id="q", prompt="prompt"):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"suites": {suite: [{"id": question_id, "prompt": prompt}]}}))
    return path


def test_score_response_sends_response_text_after_4000_chars(monkeypatch):
    captured = capture_judge_payload(monkeypatch)
    response = "a" * 4_000 + " decisive-response-tail"

    assert scorer.score_response("q", "suite", "prompt", response, "http://judge") == (3, "ok")

    assert "decisive-response-tail" in captured["payload"]["messages"][1]["content"]


def test_score_response_sends_prompt_text_after_2000_chars(monkeypatch):
    captured = capture_judge_payload(monkeypatch)
    prompt = "a" * 2_000 + " required-prompt-detail"

    assert scorer.score_response("q", "suite", prompt, "response", "http://judge") == (3, "ok")

    assert "required-prompt-detail" in captured["payload"]["messages"][1]["content"]


def test_main_persists_full_input_identities_and_source_generation_metadata(tmp_path, monkeypatch):
    prompt = "hello \u03c0"
    response = "answer \u00e9"
    result_json = tmp_path / "result.json"
    output_csv = tmp_path / "review.csv"
    result_json.write_text(json.dumps({
        "model_role": "role",
        "config_name": "config",
        "results": {"suite": {"q": {
            **current_capture(prompt, response),
            "tokens_per_second": 12.34,
            "finish_reason": "stop",
            "usage": {"completion_tokens": 7},
        }}},
    }), encoding="utf-8")
    expected_input = scorer.build_judge_input("q", "suite", prompt, response)
    pinned = write_pinned_questions(tmp_path, prompt=prompt)
    capture_judge_payload(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "score_with_claude.py", "--result-json", str(result_json), "--output", str(output_csv),
        "--producer-source", str(MODULE_PATH), "--pinned-questions", str(pinned),
    ])

    assert scorer.main() == 0

    with output_csv.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["prompt_utf8_chars"] == str(len(prompt))
    assert row["prompt_utf8_bytes"] == str(len(prompt.encode("utf-8")))
    assert row["prompt_sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert row["response_utf8_chars"] == str(len(response))
    assert row["response_utf8_bytes"] == str(len(response.encode("utf-8")))
    assert row["response_sha256"] == hashlib.sha256(response.encode("utf-8")).hexdigest()
    assert row["finish_reason"] == "stop"
    assert json.loads(row["usage"]) == {"completion_tokens": 7}
    assert row["scorer_input_utf8_bytes"] == str(expected_input["scorer_input_utf8_bytes"])
    assert row["scorer_input_sha256"] == expected_input["scorer_input_sha256"]


def test_score_response_marks_explicit_over_budget_ineligible_without_judge_call(monkeypatch):
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("over-budget rows must not call the judge")

    monkeypatch.setattr(scorer.urllib.request, "urlopen", fail_urlopen)
    score, reason = scorer.score_response(
        "q", "suite", "prompt", "response", "http://judge", judge_input_budget_bytes=1
    )

    assert score == -1
    assert reason.startswith("provisional_input_over_budget:")


def test_main_fails_closed_without_partial_aggregate(tmp_path, monkeypatch, capsys):
    result_json = tmp_path / "result.json"
    output_csv = tmp_path / "review.csv"
    result_json.write_text(json.dumps({
        "model_role": "role",
        "config_name": "config",
        "results": {"suite": {"q": {
            **current_capture("prompt", "response"),
        }}},
    }))
    monkeypatch.setattr(
        scorer.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("over-budget rows must not call the judge")
        ),
    )
    pinned = write_pinned_questions(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "score_with_claude.py",
        "--result-json", str(result_json),
        "--output", str(output_csv),
        "--judge-input-budget-bytes", "1",
        "--producer-source", str(MODULE_PATH), "--pinned-questions", str(pinned),
    ])

    assert scorer.main() == 2
    output = capsys.readouterr().out
    assert "SCORING PROVISIONAL" in output
    assert "no aggregate score is decision-grade" in output
    with output_csv.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["score_eligibility"] == "provisional_input_over_budget"


def test_main_rejects_mismatched_producer_fingerprint_before_judge(tmp_path, monkeypatch):
    result_json = tmp_path / "result.json"
    output_csv = tmp_path / "review.csv"
    row = current_capture("prompt", "response")
    row["response_fingerprint"] = scorer.producer_fingerprint("tampered")
    result_json.write_text(json.dumps({
        "model_role": "role", "config_name": "config", "results": {"suite": {"q": row}},
    }))
    monkeypatch.setattr(scorer.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("judge must not run")))
    pinned = write_pinned_questions(tmp_path)
    monkeypatch.setattr(sys, "argv", ["score_with_claude.py", "--result-json", str(result_json), "--output", str(output_csv), "--producer-source", str(MODULE_PATH), "--pinned-questions", str(pinned)])

    assert scorer.main() == 2
    with output_csv.open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["score_eligibility"] == "response_fingerprint_mismatch"


def test_legacy_requires_explicit_provisional_flag(tmp_path, monkeypatch):
    result_json = tmp_path / "result.json"
    output_csv = tmp_path / "review.csv"
    result_json.write_text(json.dumps({
        "model_role": "role", "config_name": "config",
        "results": {"suite": {"q": {"prompt": "prompt", "response": "response"}}},
    }))
    monkeypatch.setattr(scorer.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy must not judge")))
    monkeypatch.setattr(sys, "argv", ["score_with_claude.py", "--result-json", str(result_json), "--output", str(output_csv), "--allow-provisional-legacy"])

    assert scorer.main() == 2
    with output_csv.open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["score_eligibility"] == "provisional_legacy_capture"
