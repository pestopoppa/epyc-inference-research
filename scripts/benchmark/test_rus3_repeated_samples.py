"""Prepared controls; not executed in the source-only preparation window."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("rus3_producer", Path(__file__).with_name("rus3_repeated_samples.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def response():
    return {"model": "model", "usage": {"completion_tokens": 40}, "choices": [{
        "finish_reason": "stop", "message": {"content": "42", "reasoning_content": "one\nWait\nthree\nfour"}}]}


def test_decoder_preserves_native_fields_and_no_token_fallback():
    data = response()
    assert m.decode_response(data)["tokens_generated"] == 40
    data["usage"] = {}
    with pytest.raises(ValueError, match="completion_tokens"):
        m.decode_response(data)


def test_decoder_does_not_invent_finish_or_trace():
    data = response()
    data["choices"][0]["message"].pop("reasoning_content")
    assert m.decode_response(data)["reasoning_content"] is None
    assert m.decode_response(data)["trace_source"] == "unavailable"
    data["choices"][0].pop("finish_reason")
    with pytest.raises(ValueError, match="finish_reason"):
        m.decode_response(data)


def test_window_and_source_enrollment_are_separate_refusals(monkeypatch):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    monkeypatch.delenv("RUS3_SOURCE_ENROLLMENT_ACCEPTED", raising=False)
    with pytest.raises(ValueError, match="WINDOW_CONFIRMED"):
        m.transport_allowed()
    monkeypatch.setenv("WINDOW_CONFIRMED", "1")
    with pytest.raises(ValueError, match="enrollment"):
        m.transport_allowed()


def test_native_length_is_not_rewritten_to_stop():
    data = response()
    data["choices"][0]["finish_reason"] = "length"
    assert m.decode_response(data)["finish_reason"] == "length"


def config():
    return {"protocol_id": "", "k": 4, "temperature": 0.7, "seed": 42, "max_tokens": 50,
            "endpoint": "http://127.0.0.1:8083/v1/chat/completions", "run_id": "r", "role": "r",
            "model": "m", "build_commit": "commit", "model_sha256": "a" * 64,
            "binary_sha256": "b" * 64, "scorer_sha256": "c" * 64, "owner_guard_sha256": "d" * 64}


def question():
    return {"id": "q", "suite": "s", "prompt": "what", "expected": "42", "scoring_method": "exact_match"}


@pytest.mark.parametrize("seed", [-1, 4294967295, 4294967293, 2**64])
def test_sentinel_negative_and_seed_range_overflow_refuse(seed):
    cfg = config()
    cfg["seed"] = seed
    with pytest.raises(ValueError, match="seeds"):
        m.validate_config(cfg, [question()])


def test_judge_transport_is_not_a_supported_scoring_mode():
    q = question()
    q["scoring_method"] = "llm_judge"
    with pytest.raises(ValueError, match="non-transport"):
        m.validate_config(config(), [q])
