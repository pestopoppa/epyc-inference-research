"""Lossless-input tests for the LLM judge scorer."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


# --- Suite retirement guard (2026-08-12) ------------------------------------
# Property under test, per test: see each docstring. Origin: 2026-08-02
# judge-suite head-to-head ceiling measurement — general 10/10 both-perfect,
# thinking 9/10, math 8/9 at the >=27B tier.


def write_run(tmp_path, suites, model_path="Qwen3.6-27B-Q8_0.gguf"):
    """Write a result JSON + matching pinned questions for ``suites``.

    ``suites`` maps suite name -> (prompt, response).
    """
    result_json = tmp_path / "result.json"
    pinned = tmp_path / "questions.json"
    results = {}
    pinned_suites = {}
    for suite, (prompt, response) in suites.items():
        results[suite] = {"q": {
            **current_capture(prompt, response),
            "tokens_per_second": 1.0,
            "finish_reason": "stop",
            "usage": None,
        }}
        pinned_suites[suite] = [{"id": "q", "prompt": prompt}]
    result_json.write_text(json.dumps({
        "model_role": "role",
        "config_name": "config",
        "model_path": model_path,
        "results": results,
    }), encoding="utf-8")
    pinned.write_text(json.dumps({"suites": pinned_suites}))
    return result_json, pinned


def scorer_argv(result_json, output_csv, pinned):
    return [
        "score_with_claude.py", "--result-json", str(result_json),
        "--output", str(output_csv),
        "--producer-source", str(MODULE_PATH), "--pinned-questions", str(pinned),
    ]


def read_rows(output_csv):
    with output_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_repo_retirement_sidecar_binds_the_three_saturated_suites():
    """The shipped metadata names general/thinking/math as retired at 27B+
    with the measured both-perfect rates and the 2026-08-02 evidence pointer.
    Deleting or gutting any of the three entries fails HERE, loudly."""
    retired = scorer.load_suite_retirements()
    assert {"general", "thinking", "math"} <= set(retired)
    expected = {"general": ("10/10", 1.0), "thinking": ("9/10", 0.9),
                "math": ("8/9", 0.89)}
    for suite, (both_perfect, rate) in expected.items():
        entry = retired[suite]
        assert entry["both_perfect"] == both_perfect
        assert entry["both_perfect_rate"] == rate
        assert entry["min_params_b"] == 27
        assert entry["measured"] == "2026-08-02"
        assert any("judge_suite_headtohead_20260802" in e for e in entry["evidence"])


def test_missing_retirement_sidecar_refuses_loudly_before_scoring(
        tmp_path, monkeypatch, capsys):
    """Deleting the retirement metadata must be a loud refusal, never an
    implicit return to 'every suite is discriminating'. No judge call, no CSV."""
    monkeypatch.setattr(scorer, "SUITE_RETIREMENTS_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(
        scorer.urllib.request, "urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("judge must not run without the retirement sidecar")))
    result_json, pinned = write_run(tmp_path, {"coder": ("prompt", "response")})
    output_csv = tmp_path / "review.csv"
    monkeypatch.setattr(sys, "argv", scorer_argv(result_json, output_csv, pinned))

    with pytest.raises(SystemExit) as excinfo:
        scorer.main()

    assert excinfo.value.code == 2
    assert "FAIL-CLOSED" in capsys.readouterr().err
    assert not output_csv.exists()


def test_invalid_retirement_sidecar_refuses_loudly(tmp_path, monkeypatch, capsys):
    """A sidecar with the wrong schema is indistinguishable from tampering:
    refuse, do not score."""
    bad = tmp_path / "suite_retirements.json"
    bad.write_text(json.dumps({"schema": "wrong", "retired_for_discrimination": {}}))
    monkeypatch.setattr(scorer, "SUITE_RETIREMENTS_PATH", bad)
    result_json, pinned = write_run(tmp_path, {"coder": ("prompt", "response")})
    output_csv = tmp_path / "review.csv"
    monkeypatch.setattr(sys, "argv", scorer_argv(result_json, output_csv, pinned))

    with pytest.raises(SystemExit) as excinfo:
        scorer.main()

    assert excinfo.value.code == 2
    assert "FAIL-CLOSED" in capsys.readouterr().err
    assert not output_csv.exists()


def test_retired_suite_is_recorded_but_supports_no_comparative_read(
        tmp_path, monkeypatch, capsys):
    """An all-retired run still records scores (retirement is forward-looking,
    not data destruction) but must not emit a clean comparative aggregate:
    banner + stamp + distinct exit code."""
    capture_judge_payload(monkeypatch)
    result_json, pinned = write_run(tmp_path, {"general": ("prompt", "response")})
    output_csv = tmp_path / "review.csv"
    monkeypatch.setattr(sys, "argv", scorer_argv(result_json, output_csv, pinned))

    assert scorer.main() == 3

    out = capsys.readouterr().out
    assert "NON-DISCRIMINATING" in out
    assert "NO DISCRIMINATING SUITES IN THIS RUN" in out
    assert "Total:" not in out
    row = read_rows(output_csv)[0]
    assert row["claude_score"] == "3"
    assert row["suite_retirement"].startswith("RETIRED_NON_DISCRIMINATING")
    assert "2026-08-02" in row["suite_retirement"]


def test_mixed_run_total_covers_only_discriminating_suites(
        tmp_path, monkeypatch, capsys):
    """When retired and live suites are scored together, the headline aggregate
    is computed over the live suites only and says so; the retired suite is
    bannered inline."""
    capture_judge_payload(monkeypatch)
    result_json, pinned = write_run(tmp_path, {
        "general": ("prompt-g", "response-g"),
        "coder": ("prompt-c", "response-c"),
    })
    output_csv = tmp_path / "review.csv"
    monkeypatch.setattr(sys, "argv", scorer_argv(result_json, output_csv, pinned))

    assert scorer.main() == 0

    out = capsys.readouterr().out
    assert "Total (discriminating suites only): 3/3" in out
    assert "NON-DISCRIMINATING" in out
    rows = {r["suite"]: r for r in read_rows(output_csv)}
    assert rows["coder"]["suite_retirement"] == ""
    assert rows["general"]["suite_retirement"].startswith("RETIRED_NON_DISCRIMINATING")


def test_non_retired_suite_is_unaffected_at_tier(tmp_path, monkeypatch, capsys):
    """The compliant path: a live suite (coder) at the 27B tier scores exactly
    as before — plain total, no banner, empty stamp column, exit 0."""
    capture_judge_payload(monkeypatch)
    result_json, pinned = write_run(tmp_path, {"coder": ("prompt", "response")})
    output_csv = tmp_path / "review.csv"
    monkeypatch.setattr(sys, "argv", scorer_argv(result_json, output_csv, pinned))

    assert scorer.main() == 0

    out = capsys.readouterr().out
    assert "Total: 3/3" in out
    assert "RETIRED" not in out
    assert "NON-DISCRIMINATING" not in out
    assert read_rows(output_csv)[0]["suite_retirement"] == ""


def test_sub_tier_model_is_not_stamped(tmp_path, monkeypatch, capsys):
    """The other compliant path: retirement is tier-scoped. general still
    discriminates below 27B, so a 7B run is untouched."""
    capture_judge_payload(monkeypatch)
    result_json, pinned = write_run(
        tmp_path, {"general": ("prompt", "response")},
        model_path="Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    output_csv = tmp_path / "review.csv"
    monkeypatch.setattr(sys, "argv", scorer_argv(result_json, output_csv, pinned))

    assert scorer.main() == 0

    out = capsys.readouterr().out
    assert "Total: 3/3" in out
    assert "NON-DISCRIMINATING" not in out
    assert read_rows(output_csv)[0]["suite_retirement"] == ""


def test_unresolved_model_tier_fails_closed_to_stamped(
        tmp_path, monkeypatch, capsys):
    """A model whose size cannot be parsed cannot be certified sub-tier, so a
    retired suite is stamped (fail-closed), with the reason in the stamp."""
    capture_judge_payload(monkeypatch)
    result_json, pinned = write_run(
        tmp_path, {"general": ("prompt", "response")},
        model_path="mystery-model.gguf")
    output_csv = tmp_path / "review.csv"
    monkeypatch.setattr(sys, "argv", scorer_argv(result_json, output_csv, pinned))

    assert scorer.main() == 3

    row = read_rows(output_csv)[0]
    assert row["suite_retirement"].startswith("RETIRED_NON_DISCRIMINATING")
    assert "model-tier-unresolved" in row["suite_retirement"]


def test_parse_model_params_b_naming_conventions():
    """Total params come from the largest bare <num>B token; active-expert
    suffixes like -A10B are ignored; no token means unresolved (None)."""
    assert scorer.parse_model_params_b("Qwen3.5-122B-A10B-UD-Q4_K_M.gguf") == 122
    assert scorer.parse_model_params_b("gemma4-26B-A4B-Q4_K_M.gguf") == 26
    assert scorer.parse_model_params_b("Qwen2.5-7B-Instruct-Q4_K_M.gguf") == 7
    assert scorer.parse_model_params_b("Qwen3.6-27B-Q8_0.gguf") == 27
    assert scorer.parse_model_params_b("qwen3-30b-a3b-instruct.gguf") == 30
    assert scorer.parse_model_params_b("mystery-model-Q8_0.gguf") is None
    assert scorer.parse_model_params_b("") is None


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
