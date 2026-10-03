"""UFH14-B4: serving records carry what the prefix-cache metric needs.

* ``result.cached_prompt_tokens`` falls back to llama's ``timings.cache_n`` (the
  ``/completion`` lane never sets it on the result);
* ``request.slot_id_sent`` says whether a requested slot actually went on the wire;
* ``notes.server_slot`` is the slot the server reports it used;
* ``request.prefix_fp`` fingerprints the wire prompt at fixed character depths.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest

from src.backends import serving_calls as sc
from src.model_server import InferenceRequest, InferenceResult

TIMINGS = {"cache_n": 1200, "prompt_n": 340, "prompt_ms": 812.5, "predicted_n": 8, "predicted_ms": 90.0}


@pytest.fixture
def log_file(monkeypatch, tmp_path) -> Path:
    path = tmp_path / "serving_calls" / "serving_calls.jsonl"
    monkeypatch.setenv(sc.LOG_ENV, str(path))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    sc._SIDECAR_CACHE.clear()
    sc.clear_staged()
    yield path
    sc.clear_staged()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


class _Backend:
    def __init__(self, behaviour):
        self.config = types.SimpleNamespace(base_url="http://localhost:8083")
        self._behaviour = behaviour

    @sc.recorded_call("infer")
    def infer(self, role_config, request):
        return self._behaviour(request)


def _rc():
    return types.SimpleNamespace(name="architect_critic", model=types.SimpleNamespace(name="Qwen3.8-27B-Q8_0"))


def _result(**kw) -> InferenceResult:
    base = dict(role="architect_critic", output="ok", tokens_generated=8, generation_speed=1.0,
                elapsed_time=1.0, success=True)
    base.update(kw)
    return InferenceResult(**base)


def _completion_lane(request):
    sc.note_timings(TIMINGS, endpoint="/completion", stream=False)
    sc.note_server_slot(2)
    return _result()  # cached_prompt_tokens left at its default (None), as on /completion


def _chat_lane(request):
    sc.note_timings(TIMINGS, endpoint="/v1/chat/completions", stream=False)
    return _result(cached_prompt_tokens=1199)


def test_completion_lane_cached_tokens_fall_back_to_cache_n(log_file):
    _Backend(_completion_lane).infer(_rc(), InferenceRequest(role="architect_critic", prompt="x"))
    [rec] = _records(log_file)
    assert rec["result"]["cached_prompt_tokens"] == 1200
    assert rec["notes"]["server_slot"] == 2


def test_chat_lane_result_value_wins_over_cache_n(log_file):
    _Backend(_chat_lane).infer(_rc(), InferenceRequest(role="architect_critic", prompt="x"))
    [rec] = _records(log_file)
    assert rec["result"]["cached_prompt_tokens"] == 1199


def test_no_timings_leaves_cached_tokens_null(log_file):
    _Backend(lambda req: _result()).infer(_rc(), InferenceRequest(role="r", prompt="x"))
    [rec] = _records(log_file)
    assert rec["result"]["cached_prompt_tokens"] is None


@pytest.mark.parametrize("behaviour,slot,sent", [
    (_completion_lane, 1, True),
    (_completion_lane, None, False),
    (_chat_lane, 1, False),  # the chat lane never puts id_slot on the wire
])
def test_slot_id_sent_reflects_the_wire(log_file, behaviour, slot, sent):
    _Backend(behaviour).infer(_rc(), InferenceRequest(role="r", prompt="x", slot_id=slot))
    [rec] = _records(log_file)
    assert rec["request"]["slot_id"] == slot
    assert rec["request"]["slot_id_sent"] is sent


@pytest.mark.parametrize("bad", [None, -1, True, "2", 2.0])
def test_server_slot_ignores_non_slots(log_file, bad):
    def beh(request):
        sc.note_timings(TIMINGS, endpoint="/completion")
        sc.note_server_slot(bad)
        return _result()

    _Backend(beh).infer(_rc(), InferenceRequest(role="r", prompt="x"))
    [rec] = _records(log_file)
    assert "server_slot" not in rec["notes"]


def test_fingerprints_cover_only_reached_depths():
    text = "S" * 9000
    fp = sc.prefix_fingerprints(InferenceRequest(role="r", prompt=text))
    assert fp["chars"] == 9000
    assert fp["c2048"] == hashlib.sha256(text[:2048].encode()).hexdigest()[:16]
    assert "c8192" in fp and "c32768" not in fp


def test_shared_prefix_shares_shallow_fingerprints_only():
    head = "system+tools " * 1000  # 13k chars
    a = sc.prefix_fingerprints(InferenceRequest(role="r", prompt=head + "task A " * 4000))
    b = sc.prefix_fingerprints(InferenceRequest(role="r", prompt=head + "task B " * 4000))
    assert a["c2048"] == b["c2048"] and a["c8192"] == b["c8192"]
    assert a["c32768"] != b["c32768"]


def test_tool_order_is_part_of_the_prefix():
    msgs = [{"role": "user", "content": "u" * 3000}]
    t1, t2 = {"name": "a"}, {"name": "b"}
    fa = sc.prefix_fingerprints(InferenceRequest(role="r", prompt="", chat_payload={"tools": [t1, t2], "messages": msgs}))
    fb = sc.prefix_fingerprints(InferenceRequest(role="r", prompt="", chat_payload={"tools": [t2, t1], "messages": msgs}))
    assert fa["c2048"] != fb["c2048"]


def test_empty_or_odd_prompts_yield_none_not_errors():
    assert sc.prefix_fingerprints(InferenceRequest(role="r", prompt="")) is None
    assert sc.prefix_fingerprints(types.SimpleNamespace(prompt=None)) is None
    assert sc.prefix_fingerprints(types.SimpleNamespace(prompt="\ud800" * 3000))["chars"] == 3000


def test_record_carries_prefix_fp(log_file):
    _Backend(_completion_lane).infer(_rc(), InferenceRequest(role="r", prompt="p" * 2500))
    [rec] = _records(log_file)
    assert rec["request"]["prefix_fp"]["chars"] == 2500
    assert set(rec["request"]["prefix_fp"]) == {"chars", "c2048"}
