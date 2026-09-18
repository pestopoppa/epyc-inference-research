#!/usr/bin/env python3
"""M-12 review fixes: truncation is surfaced, /completion records its stop reason,
registry overrides cannot move the M-12 pins, and the smoke refuses production ports.

Offline: fake HTTP session, fake registry, in-process fake server.
"""

from __future__ import annotations

import json

import pytest

import judge_beam_run
import run_benchmark
import score_beam_run
import smoke_prefix_reuse as smoke
from lib.executor import InferenceResult, ServerManager, completion_finish_reason
from score_tulving_run import GoldIndex, render_markdown, score_result_payload
from suites import Question
from test_beam_adapter import _judged_payload
from test_smoke_prefix_reuse import FakeLlamaServer


# ── 1. finish_reason=length is surfaced ───────────────────────────────────────

def _tulving_prompt(qid):
    return {"id": qid, "prompt": "Q", "metadata": {
        "ground_truth_items": ["Jan 1"], "retrieval_type": "Times", "get_style": "all",
        "nb_events": 1}}


def test_tulving_summary_counts_truncated_rows(tmp_path):
    prov = {"chapters": 20, "variant": "Udefault_Sdefault_seed0"}
    payload = {"results": {"tulving_episodic": {
        "a": {"response": "- Jan 1", "finish_reason": "stop", "provenance": prov},
        "b": {"response": "- Jan 1\n- Jan", "finish_reason": "length", "provenance": prov},
        "c": {"response": "- Jan 1", "provenance": prov},
    }}}
    gold = GoldIndex({q: _tulving_prompt(q) for q in "abc"}, chapters=20,
                     variant="Udefault_Sdefault_seed0", book_chapters=19)
    scored = score_result_payload(payload, gold)
    s = scored["summary"]
    assert s["finish_reason_by_row"] == {"length": 1, "stop": 1, "unrecorded": 1}
    assert s["truncated_rows"] == 1 and scored["truncated_ids"] == ["b"]
    assert {r["question_id"]: r["finish_reason"] for r in scored["per_question"]}["b"] == "length"
    assert "truncated at max_tokens (finish_reason=length): 1" in render_markdown(scored, tmp_path)


def test_beam_judged_payload_and_score_carry_truncation():
    result = {"run_id": "r", "results": {"beam": {
        "q1": {"response": "x", "finish_reason": "length", "provenance": {"context_mode": "rag"}},
        "q2": {"response": "x", "finish_reason": "stop", "provenance": {"context_mode": "rag"}},
    }}}
    judged = judge_beam_run.build_judged_payload(
        result, {"records": [], "unjudged": []}, split="100K", judge_model="gemma")
    assert judged["finish_reason_by_row"] == {"length": 1, "stop": 1}
    assert judged["truncated_rows"] == 1 and judged["truncated_question_ids"] == ["q1"]
    assert judged["context_mode_by_row"] == {"rag": 2}

    payload = {**_judged_payload(), **{k: judged[k] for k in (
        "finish_reason_by_row", "truncated_rows", "truncated_question_ids")}}
    summary = score_beam_run.score_judged_payload(payload)["summary"]
    assert summary["truncated_rows"] == 1 and summary["truncated_question_ids"] == ["q1"]
    assert summary["finish_reason_by_row"] == {"length": 1, "stop": 1}


def test_beam_score_without_recorded_finish_reasons_says_so():
    summary = score_beam_run.score_judged_payload(_judged_payload())["summary"]
    assert summary["finish_reason_by_row"] == {"unrecorded": 0}
    assert summary["truncated_rows"] is None


# ── 2. /completion streaming records its stop reason ──────────────────────────

@pytest.mark.parametrize("chunk, expected", [
    ({"stop": True, "stop_type": "limit"}, "length"),
    ({"stop": True, "stop_type": "eos"}, "stop"),
    ({"stop": True, "stop_type": "word"}, "stop"),
    ({"stop": True, "stop_type": "none"}, None),
    ({"stop": True}, None),
])
def test_completion_finish_reason_mapping(chunk, expected):
    assert completion_finish_reason(chunk) == expected


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


class _FakeSession:
    def __init__(self, lines):
        self.lines, self.payloads = lines, []

    def post(self, url, json=None, timeout=None, stream=False):
        self.payloads.append(json)
        return _FakeResponse(self.lines)


def test_streaming_completion_captures_the_stop_type(monkeypatch):
    final = {"content": "", "stop": True, "stop_type": "limit", "truncated": False,
             "timings": {"predicted_per_second": 10.0, "prompt_n": 5, "predicted_n": 4,
                         "predicted_ms": 400.0}}
    session = _FakeSession(["data: " + json.dumps({"content": "- A"}),
                            "data: " + json.dumps(final)])
    server = ServerManager(port=1)
    monkeypatch.setattr(server, "_get_http_session", lambda: session)
    result = server.run_inference(prompt="p", max_tokens=4, cache_prompt=True)
    assert result.finish_reason == "length" and result.success
    assert session.payloads[0]["cache_prompt"] is True


# ── 3. registry overrides cannot move the M-12 pins ───────────────────────────

class _Registry:
    def get_temperature_override(self, role, suite):
        return 0.7

    def get_thinking_disable_trick(self, role, suite):
        return " /no_think"

    def get_role_config(self, role):
        return {"model": {"max_tokens_multiplier": 4, "disable_thinking": True,
                          "sampling": {}}}


class _Server:
    mmproj_path = None
    use_chat_api = False

    def __init__(self):
        self.calls = []

    def is_running(self):
        return True

    def run_inference(self, **kw):
        self.calls.append(kw)
        return InferenceResult(raw_output="- A\n\n", exit_code=0, command="fake",
                               finish_reason="stop")


class _Store:
    def __init__(self):
        self.rows = []

    def add_question_result(self, **kw):
        self.rows.append(kw["question_result"])


def _run(monkeypatch, suite_name, params):
    from lib.executor import Config

    monkeypatch.setattr(run_benchmark, "result_exists", lambda *a, **k: False)
    ss = run_benchmark._ServerState()
    ss.server = _Server()
    store = _Store()
    stats = {"passed": 0, "errors": 0, "skipped": 0}
    question = Question(id="q", tier=1, name="q", prompt="PROMPT", expected="", scoring=[])
    run_benchmark._run_quality_question(
        executor=None, results_manager=store, ss=ss,
        config=Config(name="baseline", config_type="baseline"), model_path="/m.gguf",
        mmproj_path=None, role="frontdoor", run_id="r", suite_name=suite_name,
        question=question, params=params, stats=stats, force=True, registry=_Registry())
    assert stats["passed"] == 1, stats
    return ss.server.calls[0], store.rows[0]


PINNED = {"temperature": 0.0, "max_tokens": 1024, "timeout": 1800,
          "enable_thinking": False, "cache_prompt": True}


@pytest.mark.parametrize("suite", sorted(run_benchmark.PINNED_GENERATION_SUITES))
def test_pinned_suites_ignore_registry_overrides_and_record_it(monkeypatch, capsys, suite):
    call, row = _run(monkeypatch, suite, dict(PINNED))
    assert call["temperature"] == 0.0 and call["max_tokens"] == 1024
    assert call["timeout"] == 1800 and call["prompt"] == "PROMPT"
    assert call["enable_thinking"] is False
    assert row.inference["ignored_registry_overrides"] == {
        "temperature_override": 0.7, "thinking_disable_trick": " /no_think",
        "max_tokens_multiplier": 4}
    assert row.inference["pinned_suite"] is True and row.inference["temperature"] == 0.0
    assert capsys.readouterr().out.count("[PINNED]") == 3


def test_other_suites_still_take_registry_overrides(monkeypatch):
    params = {"temperature": 0.6, "max_tokens": 512, "timeout": 180,
              "enable_thinking": None, "cache_prompt": None}
    call, row = _run(monkeypatch, "general", params)
    assert call["temperature"] == 0.7 and call["max_tokens"] == 2048
    assert call["prompt"] == "PROMPT /no_think"
    assert row.inference["ignored_registry_overrides"] == {}


# ── 4. the smoke refuses production ports ─────────────────────────────────────

def test_declared_ports_are_collected_from_every_port_key(tmp_path):
    manifest = tmp_path / "launch_manifest.yaml"
    manifest.write_text("port_map:\n  frontdoor: 8070\n  nested: {a: 8071}\n"
                        "roles:\n  x: {port: 8072, numa_ports: [8180, 8280], threads: 96}\n")
    registry = tmp_path / "registry.yaml"
    registry.write_text("server:\n  port: 8080\n  context_length: 16384\n")
    assert smoke.declared_ports([manifest, registry]) == {8070, 8071, 8072, 8180, 8280, 8080}


def test_smoke_refuses_a_production_port_before_any_request(tmp_path, monkeypatch):
    with FakeLlamaServer() as server:
        manifest = tmp_path / "m.yaml"
        manifest.write_text(f"port_map:\n  frontdoor: {server.port}\n")
        out = tmp_path / "r.json"
        monkeypatch.setattr("sys.argv", ["smoke_prefix_reuse.py", "--port", str(server.port),
                                         "--out", str(out), "--port-manifests", str(manifest)])
        assert smoke.main() == 1
        assert server.requests == []
    assert "declared by the production" in json.loads(out.read_text())["fails"][0]


def test_smoke_fails_closed_when_the_declarations_cannot_be_read(tmp_path, monkeypatch):
    out = tmp_path / "r.json"
    monkeypatch.setattr("sys.argv", ["smoke_prefix_reuse.py", "--port", "8199", "--out", str(out),
                                     "--port-manifests", str(tmp_path / "missing.yaml")])
    assert smoke.main() == 1
    assert "cannot read the production port" in json.loads(out.read_text())["fails"][0]


def test_the_real_declarations_include_frontdoor_and_not_the_m12_port():
    try:
        ports = smoke.declared_ports()
    except OSError:
        pytest.skip("orchestrator declarations not on this host")
    assert 8070 in ports and 8199 not in ports


def test_unrecorded_finish_reasons_do_not_claim_zero_truncation():
    prov = {"chapters": 20, "variant": "Udefault_Sdefault_seed0"}
    payload = {"results": {"tulving_episodic": {"a": {"response": "- Jan 1", "provenance": prov}}}}
    gold = GoldIndex({"a": _tulving_prompt("a")}, chapters=20,
                     variant="Udefault_Sdefault_seed0", book_chapters=19)
    s = score_result_payload(payload, gold)["summary"]
    assert s["finish_reason_by_row"] == {"unrecorded": 1} and s["truncated_rows"] is None
    judged = judge_beam_run.finish_reason_summary({"results": {"beam": {"q": {"response": "x"}}}})
    assert judged["truncated_rows"] is None
