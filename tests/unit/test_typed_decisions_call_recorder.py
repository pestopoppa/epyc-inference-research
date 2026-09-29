"""Unit tests for the TD-29.M0 per-call recorder (call_recorder.py). No server is touched."""

from __future__ import annotations

from typing import Any

import pytest

from src.typed_decisions.call_recorder import CallLog, read_last_meta, record_llm_calls
from src.typed_decisions.measure import parse_server_url_overrides


class _Primitives:
    """Each call publishes the next canned meta dict as ``_last_inference_meta``."""

    def __init__(self, metas: list[dict[str, Any] | None], *, fail_on: int | None = None):
        self.metas = list(metas)
        self.fail_on = fail_on
        self.calls = 0
        self._last_inference_meta: Any = None

    def llm_call(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        if self.fail_on == self.calls:
            raise RuntimeError("transport down")
        self._last_inference_meta = self.metas[self.calls - 1]
        return "ok"


class _PerCallPrimitives(_Primitives):
    """Exposes the per-call-safe getter, which must win over the plain attribute."""

    def get_last_inference_meta(self) -> dict[str, Any] | None:
        return {"tokens": 99}


def _meta(tokens: int, prompt: int | None = None, cached: int | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "tokens": tokens,
        "prompt_ms": 10.0,
        "gen_ms": 20.0,
        "elapsed_ms": 31.0,
        "completion_reason": "stop",
    }
    if prompt is not None:
        meta["prompt_tokens"] = prompt
    if cached is not None:
        meta["cached_prompt_tokens"] = cached
    return meta


class TestRecordLlmCalls:
    def test_every_call_is_logged_and_summed(self):
        primitives = _Primitives([_meta(100, 900, 0), _meta(40, 950, 880)])

        with record_llm_calls(primitives) as log:
            primitives.llm_call("a")
            primitives.llm_call("b")

        summary = log.summary()
        assert summary["call_count"] == 2
        assert summary["tokens_total"] == 140.0
        assert summary["tokens_last_call"] == 40.0
        assert summary["prompt_tokens_total"] == 1850.0
        assert summary["cache_n_total"] == 880.0
        assert summary["prompt_n_total"] == 900.0 + 70.0
        assert summary["prompt_ms_total"] == 20.0
        assert summary["gen_ms_total"] == 40.0
        assert summary["tokens_missing"] == 0
        assert [call["prompt_n"] for call in log.calls] == [900.0, 70.0]
        assert all(call["error"] is False for call in log.calls)

    def test_absent_telemetry_is_none_not_zero(self):
        primitives = _Primitives([{"tokens": 5}, None])

        with record_llm_calls(primitives) as log:
            primitives.llm_call("a")
            primitives.llm_call("b")

        summary = log.summary()
        assert summary["tokens_total"] == 5.0
        assert summary["tokens_last_call"] is None
        assert summary["tokens_missing"] == 1
        assert summary["prompt_tokens_total"] is None
        assert summary["prompt_n_total"] is None
        assert summary["prompt_n_missing"] == 2
        assert log.calls[1]["meta_present"] is False

    def test_prompt_n_needs_both_prompt_and_cache(self):
        primitives = _Primitives([_meta(1, prompt=500)])

        with record_llm_calls(primitives) as log:
            primitives.llm_call("a")

        assert log.calls[0]["prompt_tokens"] == 500.0
        assert log.calls[0]["cache_n"] is None
        assert log.calls[0]["prompt_n"] is None

    def test_wrapper_is_removed_and_class_method_restored(self):
        primitives = _Primitives([_meta(1), _meta(2)])

        with record_llm_calls(primitives):
            assert "llm_call" in vars(primitives)
            primitives.llm_call("a")
        assert "llm_call" not in vars(primitives)

        with record_llm_calls(primitives) as second:
            primitives.llm_call("b")
        assert second.summary()["call_count"] == 1

    def test_preexisting_instance_attribute_is_restored(self):
        primitives = _Primitives([_meta(1)])
        sentinel = primitives.llm_call
        primitives.llm_call = sentinel

        with record_llm_calls(primitives):
            primitives.llm_call("a")

        assert vars(primitives)["llm_call"] is sentinel

    def test_raising_call_is_logged_and_propagates(self):
        primitives = _Primitives([_meta(3)], fail_on=2)

        with pytest.raises(RuntimeError, match="transport down"):
            with record_llm_calls(primitives) as log:
                primitives.llm_call("a")
                primitives.llm_call("b")

        assert log.errors == 1
        assert [call["error"] for call in log.calls] == [False, True]
        assert log.calls[1]["tokens"] is None
        assert "llm_call" not in vars(primitives)

    def test_per_call_safe_getter_wins(self):
        primitives = _PerCallPrimitives([_meta(1)])

        with record_llm_calls(primitives) as log:
            primitives.llm_call("a")

        assert log.calls[0]["tokens"] == 99.0
        assert read_last_meta(primitives) == {"tokens": 99}

    def test_object_without_llm_call_yields_empty_log(self):
        with record_llm_calls(object()) as log:
            pass
        assert isinstance(log, CallLog)
        assert log.summary()["call_count"] == 0
        assert log.summary()["tokens_last_call"] is None


class TestServerUrlOverrides:
    def test_parses_role_url_pairs(self):
        assert parse_server_url_overrides(
            ["frontdoor=http://127.0.0.1:8199/", "worker=https://h:1"]
        ) == {"frontdoor": "http://127.0.0.1:8199", "worker": "https://h:1"}

    def test_empty_is_empty(self):
        assert parse_server_url_overrides([]) == {}

    @pytest.mark.parametrize(
        "item",
        ["frontdoor", "=http://x", "frontdoor=", "frontdoor=127.0.0.1:8199"],
    )
    def test_malformed_items_are_rejected(self, item: str):
        with pytest.raises(ValueError):
            parse_server_url_overrides([item])

    def test_conflicting_duplicate_is_rejected(self):
        with pytest.raises(ValueError, match="two URLs"):
            parse_server_url_overrides(["f=http://a", "f=http://b"])
