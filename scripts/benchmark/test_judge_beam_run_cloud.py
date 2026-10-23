#!/usr/bin/env python3
"""HS-PROF-1: the cloud (claude_cli) judge backend of judge_beam_run.py. Offline: ``run_fn`` is injected, the CLI is never spawned."""

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import judge_beam_run as jbr  # noqa: E402
from beam_scoring import build_nugget_judge_prompt  # noqa: E402
from test_judge_beam_run import _index, _payload  # noqa: E402

cj = jbr.load_cloud_judges()
SPEC = cj.load_registry()["sonnet-low"]


def envelope(score=1.0, model="claude-sonnet-5-5", i=10, o=5, cost=0.01, is_error=False):
    return json.dumps({"type": "result", "is_error": is_error, "structured_output": {"score": score, "reason": "r"},
                       "usage": {"input_tokens": i, "output_tokens": o, "cache_read_input_tokens": 3},
                       "total_cost_usd": cost, "modelUsage": {model: {}}})


class FakeRun:
    def __init__(self, outs):
        self.outs, self.calls, self.lock = list(outs), [], threading.Lock()

    def __call__(self, argv, stdin_text, timeout_s, cwd):
        with self.lock:
            self.calls.append((argv, stdin_text, timeout_s, cwd))
            out = self.outs.pop(0) if len(self.outs) > 1 else self.outs[0]
        return (0, out, "")


def _judge(run):
    return lambda cfg: jbr.CloudJudge(cj, SPEC, None, run_fn=run)


def test_prompt_parity_and_cli_shape(tmp_path):
    index = _index(tmp_path)
    run = FakeRun([envelope()])
    jbr.judge_run(_payload(index), index, judge_for=_judge(run))
    seen = {c[1] for c in run.calls}
    expected = {build_nugget_judge_prompt(p["scoring_config"].get("probing_question", ""), item, "A responsive answer.")
                for p in index.values() for item in p["scoring_config"]["nuggets"]}
    assert seen == expected  # stdin prompt == served-path prompt, byte for byte
    argv = run.calls[0][0]
    for flag in ("-p", "--restricted", "--strict-mcp-config", "--json-schema", "--model"):
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--model") + 1] == SPEC.model
    assert json.loads(argv[argv.index("--json-schema") + 1]) == jbr.NUGGET_OUTPUT_SCHEMA


def test_verdict_parse_and_error_paths(tmp_path):
    index = _index(tmp_path)
    out = jbr.judge_run(_payload(index), index, judge_for=_judge(FakeRun([envelope(0.5)])))
    assert not out["unjudged"] and all(v == 0.5 for r in out["records"] for v in r["nugget_verdicts"])
    out = jbr.judge_run(_payload(index), index, judge_for=_judge(FakeRun([envelope(is_error=True)])))
    assert out["unjudged"] and {u["reason"] for u in out["unjudged"]} == {"judge_unavailable"}
    out = jbr.judge_run(_payload(index), index, judge_for=_judge(FakeRun([envelope(0.7)])))  # off-scale
    assert {u["reason"] for u in out["unjudged"]} == {"unparseable_verdict"}


def test_usage_cost_model_recorded_and_resume(tmp_path):
    index = _index(tmp_path)
    partial = tmp_path / "o.json.partial.jsonl"
    r1 = FakeRun([envelope()])
    first = jbr.judge_run(_payload(index), index, judge_for=_judge(r1), partial_path=partial)
    n = len(r1.calls)
    r2 = FakeRun([envelope()])
    judged = jbr.judge_run(_payload(index), index, judge_for=_judge(r2), partial_path=partial)
    assert r2.calls == [] and len(judged["records"]) == len(first["records"]) == len(index)
    out = jbr.build_judged_payload(_payload(index), judged, split="100K", judge_model="m",
                                   judge_backend="cloud", cloud_judge="sonnet-low")
    t = out["judge_usage_totals"]
    assert t["calls"] == n and t["input_tokens"] == 10 * n and t["output_tokens"] == 5 * n
    assert t["total_cost_usd"] == pytest.approx(0.01 * n) and out["judge_response_models"] == ["claude-sonnet-5-5"]
    assert out["judge_backend"] == "cloud" and out["cloud_judge"] == "sonnet-low"


def test_concurrent_workers_same_result_as_serial(tmp_path):
    index = _index(tmp_path)
    a = jbr.judge_run(_payload(index), index, judge_for=_judge(FakeRun([envelope()])), workers=1)
    b = jbr.judge_run(_payload(index), index, judge_for=_judge(FakeRun([envelope()])), workers=2,
                      partial_path=tmp_path / "p.jsonl")
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("judge_")}  # noqa: E731
    assert [strip(r) for r in a["records"]] == [strip(r) for r in b["records"]]
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == len(index)


def test_served_payload_unchanged():
    out = jbr.build_judged_payload({"results": {"beam": {}}}, {"records": [], "unjudged": []}, split="100K", judge_model="x")
    assert out["judge_backend"] == "served" and "judge_usage_totals" not in out


def test_preflight_ok_and_failure():
    jbr.cloud_preflight(cj, SPEC, run_fn=FakeRun([json.dumps({"structured_output": {"ok": True}})]))
    with pytest.raises(jbr.JudgeUnavailable, match="not authenticated"):
        jbr.cloud_preflight(cj, SPEC, run_fn=lambda *a: (1, "", "Not logged in"))


def test_estimate_before_calls(tmp_path):
    index = _index(tmp_path)
    est = jbr.estimate_judge_calls(_payload(index), index, {})
    assert est["judge_calls"] == sum(len(p["scoring_config"]["nuggets"]) for p in index.values()) and est["avg_prompt_chars"] > 0


def test_main_cloud_end_to_end_and_blocked(tmp_path, monkeypatch, capsys):
    index = _index(tmp_path)
    res = tmp_path / "r.json"
    res.write_text(json.dumps(_payload(index)))
    import long_context_adapters
    monkeypatch.setattr(long_context_adapters, "BEAMAdapter",
                        lambda **kw: types.SimpleNamespace(extract_all=lambda: list(index.values())))
    run = FakeRun([json.dumps({"structured_output": {"ok": True}})] + [envelope()])
    monkeypatch.setattr(cj, "_default_run", run)
    out = tmp_path / "j.json"
    assert jbr.main([str(res), "--out-json", str(out), "--judge-backend", "cloud"]) == 0
    err = capsys.readouterr().err
    assert "judge estimate (before any call)" in err
    d = json.loads(out.read_text())
    assert d["judge_backend"] == "cloud" and d["judge_model_id"] == SPEC.model and not d["unjudged"]
    # not authenticated -> BLOCKED rc 5, no judging
    out2 = tmp_path / "j2.json"
    monkeypatch.setattr(cj, "_default_run", lambda *a: (1, "", "Not logged in"))
    assert jbr.main([str(res), "--out-json", str(out2), "--judge-backend", "cloud"]) == 5
    assert "BLOCKED" in capsys.readouterr().err and not out2.exists()
