#!/usr/bin/env python3
"""HS-PROF-1: the anthropic judge backend of judge_beam_run.py. Offline; the HTTP layer is mocked."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import judge_beam_run as jbr  # noqa: E402
from beam_scoring import build_nugget_judge_prompt  # noqa: E402
from test_judge_beam_run import _index, _payload  # noqa: E402

FAKE_KEY = "sk-ant-TESTKEY-do-not-leak-0123456789"


def _ok(text='{"score": 1.0, "reason": "ok"}', model="claude-sonnet-5-5-20261001", i=10, o=5):
    return 200, json.dumps({"model": model, "content": [{"type": "text", "text": text}],
                            "usage": {"input_tokens": i, "output_tokens": o}}).encode()


class FakePost:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, dict(headers), json.loads(body)))
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv(jbr.ANTHROPIC_KEY_ENV, FAKE_KEY)


def test_request_shape_and_prompt_parity_with_served_path(tmp_path):
    index = _index(tmp_path)
    post = FakePost([_ok()])
    usage = jbr.AnthropicUsage()
    judge = jbr.anthropic_judge("claude-sonnet-5-5", usage, post=post)
    jbr.judge_run(_payload(index), index, judge_for=lambda cfg: judge)
    # served path: judge_question builds prompts via build_nugget_judge_prompt and hands the SAME string to the judge
    seen = {c[2]["messages"][0]["content"] for c in post.calls}
    expected = set()
    for qid, p in index.items():
        cfg = p["scoring_config"]
        for item in cfg["nuggets"]:
            expected.add(build_nugget_judge_prompt(cfg.get("probing_question", ""), item, "A responsive answer."))
    assert seen == expected
    url, headers, body = post.calls[0]
    assert url == jbr.ANTHROPIC_URL and body["model"] == "claude-sonnet-5-5"
    assert body["temperature"] == 0 and body["max_tokens"] == jbr.NUGGET_MAX_TOKENS
    assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
    assert usage.totals["calls"] == len(post.calls) and usage.response_models == {"claude-sonnet-5-5-20261001"}


def test_parsing_fenced_and_garbage(tmp_path):
    index = _index(tmp_path)
    fenced = FakePost([_ok('```json\n{"score": 0.5}\n```')])
    out = jbr.judge_run(_payload(index), index, judge_for=lambda c: jbr.anthropic_judge("m", jbr.AnthropicUsage(), post=fenced))
    assert not out["unjudged"] and all(v == 0.5 for r in out["records"] for v in r["nugget_verdicts"])
    bad = FakePost([_ok("no json here")])
    out = jbr.judge_run(_payload(index), index, judge_for=lambda c: jbr.anthropic_judge("m", jbr.AnthropicUsage(), post=bad))
    assert out["unjudged"] and not out["records"]
    assert {u["reason"] for u in out["unjudged"]} == {"unparseable_verdict"}


def test_retry_429_5xx_then_success_and_giveup(tmp_path):
    sleeps = []
    post = FakePost([(429, b"slow down"), (503, b"x"), _ok()])
    judge = jbr.anthropic_judge("m", jbr.AnthropicUsage(), post=post, sleep=sleeps.append)
    assert "score" in judge("p") and len(post.calls) == 3 and sleeps == [2.0, 4.0]
    post = FakePost([(500, b"boom")])
    judge = jbr.anthropic_judge("m", jbr.AnthropicUsage(), post=post, sleep=lambda s: None)
    with pytest.raises(jbr.JudgeUnavailable):
        judge("p")
    assert len(post.calls) == jbr.ANTHROPIC_MAX_RETRIES + 1
    post = FakePost([(400, b"bad request")])  # non-retryable
    with pytest.raises(jbr.JudgeUnavailable):
        jbr.anthropic_judge("m", jbr.AnthropicUsage(), post=post, sleep=lambda s: None)("p")
    assert len(post.calls) == 1


def test_resume_does_not_rejudge_and_usage_totals_survive(tmp_path):
    index = _index(tmp_path)
    partial = tmp_path / "o.json.partial.jsonl"
    u1 = jbr.AnthropicUsage()
    p1 = FakePost([_ok()])
    j1 = jbr.anthropic_judge("m", u1, post=p1)
    jbr.judge_run(_payload(index), index, judge_for=lambda c: j1, partial_path=partial, usage_probe=u1.snapshot)
    first_calls = len(p1.calls)
    u2 = jbr.AnthropicUsage()
    p2 = FakePost([_ok()])
    j2 = jbr.anthropic_judge("m", u2, post=p2)
    judged = jbr.judge_run(_payload(index), index, judge_for=lambda c: j2, partial_path=partial, usage_probe=u2.snapshot)
    assert p2.calls == [] and len(judged["records"]) == len(index)
    out = jbr.build_judged_payload(_payload(index), judged, split="100K", judge_model="m",
                                   judge_backend="anthropic", judge_model_id="m", response_models=["r"])
    assert out["judge_usage_totals"]["calls"] == first_calls and out["judge_usage_totals"]["input_tokens"] == 10 * first_calls
    assert out["judge_backend"] == "anthropic" and out["judge_model_id"] == "m" and out["judge_response_models"] == ["r"]


def test_served_payload_unchanged():
    out = jbr.build_judged_payload({"results": {"beam": {}}}, {"records": [], "unjudged": []}, split="100K", judge_model="x")
    assert out["judge_backend"] == "served" and "judge_usage_totals" not in out and "judge_model_id" not in out


def test_missing_key_refuses(monkeypatch):
    monkeypatch.delenv(jbr.ANTHROPIC_KEY_ENV)
    with pytest.raises(jbr.MissingApiKey):
        jbr.anthropic_judge("m", jbr.AnthropicUsage(), post=FakePost([_ok()]))
    assert jbr.main(["x.json", "--out-json", "y.json", "--judge-backend", "anthropic"]) == 5


def test_estimate_before_calls(tmp_path):
    index = _index(tmp_path)
    est = jbr.estimate_judge_calls(_payload(index), index, {})
    n = sum(len(p["scoring_config"]["nuggets"]) for p in index.values())
    assert est["judge_calls"] == n and est["avg_prompt_chars"] > 0
    first = sorted(index)[0]
    assert jbr.estimate_judge_calls(_payload(index), index, {first: {}})["judge_calls"] < n


def test_main_end_to_end_no_key_in_any_output(tmp_path, monkeypatch, capsys):
    index = _index(tmp_path)
    res = tmp_path / "r.json"
    res.write_text(json.dumps(_payload(index)))
    post = FakePost([_ok()])
    monkeypatch.setattr(jbr, "_http_post", post)
    import long_context_adapters
    monkeypatch.setattr(long_context_adapters, "BEAMAdapter",
                        lambda **kw: types.SimpleNamespace(extract_all=lambda: list(index.values())))
    out = tmp_path / "j.json"
    rc = jbr.main([str(res), "--out-json", str(out), "--judge-backend", "anthropic"])
    cap = capsys.readouterr()
    assert rc == 0
    blob = cap.out + cap.err + out.read_text() + (tmp_path / "j.json.partial.jsonl").read_text()
    assert FAKE_KEY not in blob and "TESTKEY" not in blob
    d = json.loads(out.read_text())
    assert d["judge_model_id"] == "claude-sonnet-5-5" and d["judge_backend"] == "anthropic"
    assert "judge estimate (before any call)" in cap.err
    assert post.calls[0][1]["x-api-key"] == FAKE_KEY  # key reaches only the header
