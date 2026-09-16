"""Offline tests for eval_tale_budget.py (PRB-T4 harness fixes).

All HTTP goes through the injectable ``poster``/``fetcher`` seams; no server
is contacted. Run: `python -m pytest scripts/benchmark/test_eval_tale_budget.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import eval_tale_budget as tb  # noqa: E402


class FakeServer:
    """Records requests; answers the estimator and answer calls differently."""

    def __init__(self, budget_reply="Budget: [[150]]", est_tokens=7, ans_tokens=40):
        self.calls: list[tuple[str, dict]] = []
        self.budget_reply = budget_reply
        self.est_tokens = est_tokens
        self.ans_tokens = ans_tokens

    def __call__(self, url, body, timeout):
        self.calls.append((url, body))
        content = body["messages"][0]["content"]
        is_est = body["max_tokens"] == tb.ESTIMATOR_MAX_TOKENS
        return {
            "model": "Qwen3.8-27B-Q8_0.gguf",
            "choices": [{"message": {
                "content": self.budget_reply if is_est else f"ans:{len(content)}"}}],
            "usage": {"completion_tokens": self.est_tokens if is_est else self.ans_tokens},
        }


@pytest.fixture(autouse=True)
def _fake_scorer(monkeypatch):
    monkeypatch.setattr(tb, "score_question", lambda text, q: q.get("suite") == "math")


QUESTIONS = [
    {"id": "m1", "suite": "math", "prompt": "What is 2+2?"},
    {"id": "k1", "suite": "mmlu_pro", "prompt": "Capital of France?"},
]


def _cfg(server, **kw):
    return tb.GenConfig(base_url="http://127.0.0.1:9999", poster=server, **kw)


def test_resolve_base_url():
    assert tb.resolve_base_url(None, "h", 8083) == "http://h:8083"
    assert tb.resolve_base_url("http://127.0.0.1:8083/v1/") == "http://127.0.0.1:8083"
    assert tb.resolve_base_url("127.0.0.1:8083") == "http://127.0.0.1:8083"


def test_parse_budget_prefers_bracket_format_and_clamps():
    assert tb.parse_budget("Budget: [[12]] maybe 900", "tokens") == 12
    assert tb.parse_budget("about 99999", "tokens") == 4096
    assert tb.parse_budget("3", "words") == 10
    assert tb.parse_budget("<think>7</think>none", "tokens") is None


def test_token_prompts_follow_tale_ep():
    p = tb.build_tale_prompt("Q?", 150, "tokens")
    assert p == "Q?\n\nLet's think step by step and use less than 150 tokens."
    assert tb.build_tale_prompt("Q?", 45, "words") == "Answer in under 45 words.\n\nQ?"
    assert "tokens" in tb.TALE_PREPASS_PROMPT_TOKENS
    assert "[[budget]]" in tb.TALE_PREPASS_PROMPT_TOKENS


def test_estimator_cost_is_charged_on_tale_rows_only():
    server = FakeServer()
    results = tb.run_evaluation(QUESTIONS, ["baseline", "static", "tale"], _cfg(server))
    assert len(results) == 6
    # 2 estimator calls + 6 answer calls
    assert len(server.calls) == 8
    for r in results:
        assert r.total_tokens == 40  # answer-only unchanged meaning
        if r.condition == "tale":
            assert r.estimator_tokens == 7
            assert r.total_tokens_incl_estimator == 47
            assert r.tale_budget == 150 and r.budget_unit == "tokens"
            assert r.estimator_response == "Budget: [[150]]"
            assert r.prompt.endswith("use less than 150 tokens.")
            assert r.elapsed_s_incl_estimator >= r.elapsed_s
        else:
            assert r.estimator_tokens == 0 and r.estimator_s == 0.0
            assert r.total_tokens_incl_estimator == 40
            assert r.tale_budget is None and r.budget_unit is None
        assert r.served_model == "Qwen3.8-27B-Q8_0.gguf"


def test_estimator_fallback_when_no_number():
    server = FakeServer(budget_reply="no idea")
    est = tb.estimate_tale_budget_full("Q?", _cfg(server))
    assert est.budget == tb.BUDGET_CLAMPS["tokens"][2]
    assert est.tokens == 7


def test_words_unit_uses_legacy_prompts():
    server = FakeServer(budget_reply="45")
    results = tb.run_evaluation(QUESTIONS[:1], ["tale"], _cfg(server, budget_unit="words"))
    est_body = server.calls[0][1]
    assert "Words needed:" in est_body["messages"][0]["content"]
    assert results[0].prompt.startswith("Answer in under 45 words.")
    assert results[0].budget_unit == "words"


def test_request_body_carries_temperature_seed_model_and_kwargs():
    server = FakeServer()
    cfg = _cfg(server, temperature=0.2, seed=42, model="m-id",
               chat_template_kwargs={"enable_thinking": False})
    results = tb.run_evaluation(QUESTIONS[:1], ["baseline", "tale"], cfg)
    for url, body in server.calls:
        assert url == "http://127.0.0.1:9999/v1/chat/completions"
        assert body["temperature"] == 0.2
        assert body["seed"] == 42
        assert body["model"] == "m-id"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert all(r.temperature == 0.2 and r.seed == 42 for r in results)


def test_seed_omitted_when_none():
    server = FakeServer()
    tb.run_evaluation(QUESTIONS[:1], ["baseline"], _cfg(server, seed=None))
    assert "seed" not in server.calls[0][1]
    assert "model" not in server.calls[0][1]


def test_summarize_reports_answer_only_and_net_totals():
    server = FakeServer()
    results = tb.run_evaluation(QUESTIONS, ["baseline", "tale"], _cfg(server))
    summ = tb.summarize(results)
    tale = summ["math"]["tale"]
    assert tale["mean_tokens_answer_only"] == 40
    assert tale["mean_tokens_incl_estimator"] == 47
    assert tale["answer_token_change_vs_baseline"] == 0.0
    assert tale["net_token_change_vs_baseline"] == pytest.approx(7 / 40)
    assert tale["budget"]["unit"] == "tokens"
    assert summ["math"]["baseline"]["accuracy"] == 1.0
    assert summ["mmlu_pro"]["tale"]["accuracy_delta_pp_vs_baseline"] == 0.0
    assert summ["__all__"]["tale"]["n"] == 2


def test_fetch_serving_identity_from_v1_models_and_props(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 12)

    def fetcher(url, timeout):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "qwen38-27b"}]}
        if url.endswith("/props"):
            return {"model_path": str(gguf), "build_info": "b10125-0db32c06",
                    "total_slots": 1, "default_generation_settings": {"n_ctx": 32768}}
        raise AssertionError(url)

    ident = tb.fetch_serving_identity("http://x:1", fetcher)
    assert ident["model_id"] == "qwen38-27b" and ident["model_id_source"] == "v1_models"
    assert ident["gguf_path"] == str(gguf) and ident["gguf_path_source"] == "props"
    assert ident["gguf"]["size_bytes"] == 16
    assert "sha256" not in ident["gguf"]
    assert ident["props"]["build_info"] == "b10125-0db32c06"
    assert ident["props"]["n_ctx"] == 32768
    assert "errors" not in ident


def test_fetch_serving_identity_overrides_hash_and_errors(tmp_path):
    gguf = tmp_path / "o.gguf"
    gguf.write_bytes(b"abc")

    def fetcher(url, timeout):
        raise ConnectionError("down")

    ident = tb.fetch_serving_identity("http://x:1", fetcher, model_id="ovr",
                                      gguf_path=str(gguf), hash_gguf=True)
    assert ident["model_id"] == "ovr" and ident["model_id_source"] == "override"
    assert ident["gguf_path_source"] == "override"
    assert ident["gguf"]["sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert set(ident["errors"]) == {"v1_models", "props"}
    missing = tb.stat_gguf(str(tmp_path / "nope.gguf"))
    assert "error" in missing


def test_main_end_to_end_writes_rows_meta_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "load_questions", lambda suites, n: list(QUESTIONS))
    server = FakeServer()
    fetched = []

    def fetcher(url, timeout):
        fetched.append(url)
        return {"data": [{"id": "served"}]} if url.endswith("/v1/models") else {
            "model_path": "/nonexistent/m.gguf"}

    out = tmp_path / "run.jsonl"
    tb.main([
        "--endpoint", "http://127.0.0.1:8083/v1", "--suites", "math", "mmlu_pro",
        "--temperature", "0.2", "--chat-template-kwargs", '{"enable_thinking": false}',
        "--output", str(out),
    ], poster=server, fetcher=fetcher)

    assert fetched == ["http://127.0.0.1:8083/v1/models", "http://127.0.0.1:8083/props"]
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 6
    assert {"estimator_tokens", "estimator_s", "total_tokens_incl_estimator",
            "elapsed_s_incl_estimator", "budget_unit", "temperature", "seed"} <= set(rows[0])
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["budget_unit"] == "tokens"
    assert meta["temperature"] == 0.2 and meta["seed"] == 42
    assert meta["serving"]["model_id"] == "served"
    assert meta["serving"]["gguf_path"] == "/nonexistent/m.gguf"
    assert meta["served_models_seen"] == ["Qwen3.8-27B-Q8_0.gguf"]
    assert meta["chat_template_kwargs"] == {"enable_thinking": False}
    summ = json.loads(out.with_suffix(".summary.json").read_text())
    assert summ["math"]["tale"]["mean_tokens_incl_estimator"] == 47


def test_main_dry_run_makes_no_requests(monkeypatch, capsys):
    monkeypatch.setattr(tb, "load_questions", lambda suites, n: list(QUESTIONS))

    def boom(*a, **k):
        raise AssertionError("network used in dry run")

    tb.main(["--dry-run", "--endpoint", "http://127.0.0.1:8083"], poster=boom, fetcher=boom)
    assert "use less than 200 tokens" in capsys.readouterr().out


def test_seed_negative_disables_and_bad_kwargs_rejected():
    args = tb.build_parser().parse_args(["--seed", "-1"])
    assert tb.build_config(args).seed is None
    args = tb.build_parser().parse_args(["--chat-template-kwargs", "[1]"])
    with pytest.raises(SystemExit):
        tb.build_config(args)
