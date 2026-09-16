#!/usr/bin/env python3
"""CME-1: judge_beam_run.py, the per-nugget judging step between a stored run and the fold.

Offline only. The judge is a stub, the served transport is a fake ``debug_scorer``
module, and the dataset is ``test_beam_adapter``'s repo-tree fixture (no pyarrow
needed).
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import judge_beam_run as jbr  # noqa: E402
import score_beam_run  # noqa: E402
from beam_scoring import BEAM_ABILITIES, BEAMFoldError  # noqa: E402
from long_context_adapters import BEAMAdapter  # noqa: E402
from test_beam_adapter import write_repo_fixture  # noqa: E402


def _index(tmp_path):
    adapter = BEAMAdapter(data_dir=write_repo_fixture(tmp_path / "data"))
    return {p["id"]: p for p in adapter.extract_all()}


def _payload(index, response="A responsive answer."):
    return {"run_id": "r1", "model_role": "ingest_long_context", "config_name": "baseline",
            "results": {"beam": {qid: {"question_id": qid, "prompt": "...",
                                       "response": response} for qid in index}}}


def _rubric(prompt: str) -> str:
    return re.search(r"RUBRIC CRITERION \(what to check\): (.*)\n", prompt).group(1)


class StubJudge:
    """Scores every nugget 0.5 and records each prompt it saw."""

    def __init__(self, reply='{"score": 0.5, "reason": "partial"}'):
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_one_judge_call_per_nugget_and_the_question_is_in_every_prompt(tmp_path):
    index = _index(tmp_path)
    judge = StubJudge()
    judged = jbr.judge_run(_payload(index), index, judge_for=lambda cfg: judge)
    n_nuggets = sum(len(p["scoring_config"]["nuggets"]) for p in index.values())
    assert len(judge.prompts) == n_nuggets
    assert judged["unjudged"] == []
    by_id = {r["question_id"]: r for r in judged["records"]}
    for qid, prompt in index.items():
        cfg = prompt["scoring_config"]
        assert by_id[qid]["nugget_verdicts"] == [0.5] * len(cfg["nuggets"])
        assert by_id[qid]["ability"] == cfg["ability"]
    assert all("QUESTION (what the user asked): " in p for p in judge.prompts)
    assert {_rubric(p) for p in judge.prompts} == {
        n for p in index.values() for n in p["scoring_config"]["nuggets"]}


def test_all_half_run_folds_to_0500_end_to_end(tmp_path):
    index = _index(tmp_path)
    judged = jbr.judge_run(_payload(index), index, judge_for=lambda cfg: StubJudge())
    payload = jbr.build_judged_payload(_payload(index), judged, split="100K",
                                       judge_model="stub-judge")
    folded = score_beam_run.score_judged_payload(payload, index)
    assert folded["summary"]["headline"] == pytest.approx(0.5)
    assert folded["summary"]["secondary_diagnostics"]["binarised_pass_rate"] == 1.0
    assert folded["summary"]["judge_model"] == "stub-judge"
    assert set(folded["summary"]["abilities_reported"]) == set(BEAM_ABILITIES)


def test_unparseable_verdict_is_unjudged_never_zero(tmp_path):
    index = _index(tmp_path)
    judged = jbr.judge_run(_payload(index), index,
                           judge_for=lambda cfg: StubJudge(reply="I think it is fine"))
    assert judged["records"] == []
    assert {u["reason"] for u in judged["unjudged"]} == {"unparseable_verdict"}
    payload = jbr.build_judged_payload(_payload(index), judged, split="100K", judge_model="j")
    with pytest.raises(BEAMFoldError):
        score_beam_run.score_judged_payload(payload, index)


def test_judge_unavailable_and_empty_response_are_unjudged(tmp_path):
    index = _index(tmp_path)
    payload = _payload(index)
    first, second = sorted(index)[:2]
    payload["results"]["beam"][second]["response"] = "   "

    def flaky(cfg):
        def judge(prompt):
            if cfg["probing_question"] == index[first]["scoring_config"]["probing_question"]:
                raise jbr.JudgeUnavailable("llm_judge_transport_error: down")
            return '{"score": 1.0}'
        return judge

    judged = jbr.judge_run(payload, index, judge_for=flaky)
    reasons = {u["question_id"]: u["reason"] for u in judged["unjudged"]}
    assert reasons[first] == "judge_unavailable"
    assert reasons[second] == "empty_response"
    assert len(judged["records"]) == len(index) - 2


def test_fold_refuses_a_payload_with_unjudged_questions(tmp_path):
    index = _index(tmp_path)
    judged = jbr.judge_run(_payload(index), index, judge_for=lambda cfg: StubJudge())
    judged["unjudged"].append({"question_id": "x", "reason": "judge_unavailable"})
    payload = jbr.build_judged_payload(_payload(index), judged, split="100K", judge_model="j")
    with pytest.raises(BEAMFoldError, match="never judged"):
        score_beam_run.score_judged_payload(payload, index)


def test_unknown_question_id_is_unjudged(tmp_path):
    index = _index(tmp_path)
    payload = _payload(index)
    payload["results"]["beam"]["beam_100K_999_abstention_0"] = {"response": "x"}
    judged = jbr.judge_run(payload, index, judge_for=lambda cfg: StubJudge())
    assert {"question_id": "beam_100K_999_abstention_0",
            "reason": "unknown_question_id"} in judged["unjudged"]


def test_partial_file_resumes_without_rejudging(tmp_path):
    index = _index(tmp_path)
    partial = tmp_path / "out.json.partial.jsonl"
    first = StubJudge()
    jbr.judge_run(_payload(index), index, judge_for=lambda cfg: first, partial_path=partial)
    assert first.prompts
    second = StubJudge(reply="must not be called")
    again = jbr.judge_run(_payload(index), index, judge_for=lambda cfg: second,
                          partial_path=partial)
    assert second.prompts == []
    assert len(again["records"]) == len(index)


def test_rerun_retries_unjudged_rows_from_the_partial_file(tmp_path):
    """Regression: unjudged rows in the partial file must be retried, not carried forward."""
    index = _index(tmp_path)
    partial = tmp_path / "out.json.partial.jsonl"
    first_id = sorted(index)[0]

    def down_for_first(cfg):
        def judge(prompt):
            if cfg["probing_question"] == index[first_id]["scoring_config"]["probing_question"]:
                raise jbr.JudgeUnavailable("llm_judge_transport_error: down")
            return '{"score": 1.0}'
        return judge

    before = jbr.judge_run(_payload(index), index, judge_for=down_for_first,
                           partial_path=partial)
    assert [u["question_id"] for u in before["unjudged"]] == [first_id]

    recovered = StubJudge()
    after = jbr.judge_run(_payload(index), index, judge_for=lambda cfg: recovered,
                          partial_path=partial)
    assert after["unjudged"] == []
    assert len(after["records"]) == len(index)
    # Only the previously unjudged question was re-judged.
    assert len(recovered.prompts) == len(index[first_id]["scoring_config"]["nuggets"])
    by_id = {r["question_id"]: r for r in after["records"]}
    assert set(by_id[first_id]["nugget_verdicts"]) == {0.5}

    # A third pass is a pure resume: nothing left to judge.
    third = StubJudge(reply="must not be called")
    final = jbr.judge_run(_payload(index), index, judge_for=lambda cfg: third,
                          partial_path=partial)
    assert third.prompts == [] and final["unjudged"] == []


def test_served_judge_binds_the_orchestrator_transport(monkeypatch):
    calls = []

    class Unavailable(RuntimeError):
        pass

    def request_llm_judge_text(prompt, config, *, max_tokens, output_schema):
        calls.append((prompt, config, max_tokens, output_schema))
        if prompt == "boom":
            raise Unavailable("llm_judge_transport_error")
        return '{"score": 1.0}'

    fake = types.ModuleType("debug_scorer")
    fake.request_llm_judge_text = request_llm_judge_text
    fake.ScoringUnavailableError = Unavailable
    monkeypatch.setitem(sys.modules, "debug_scorer", fake)

    judge = jbr.served_judge({"judge_port": 8082, "per_nugget": True},
                             {"judge_role": "architect_general", "timeout": None})
    assert judge("P") == '{"score": 1.0}'
    prompt, config, max_tokens, schema = calls[0]
    assert config["judge_role"] == "architect_general" and "timeout" not in config
    assert max_tokens == jbr.NUGGET_MAX_TOKENS and schema == jbr.NUGGET_OUTPUT_SCHEMA
    with pytest.raises(jbr.JudgeUnavailable):
        judge("boom")


def test_main_writes_the_judged_payload_and_signals_gaps(tmp_path, monkeypatch):
    index = _index(tmp_path)
    result = tmp_path / "run.json"
    result.write_text(json.dumps(_payload(index)))
    monkeypatch.setattr(jbr, "served_judge", lambda cfg, overrides: StubJudge())
    out = tmp_path / "judged.json"
    rc = jbr.main([str(result), "--out-json", str(out), "--judge-model", "stub",
                   "--data-dir", str(tmp_path / "data")])
    assert rc == 0
    body = json.loads(out.read_text())
    assert body["schema"] == jbr.JUDGED_SCHEMA and body["judge_model"] == "stub"
    assert body["question_in_judge_prompt"] is True and body["unjudged"] == []
    assert len(body["records"]) == len(index)

    monkeypatch.setattr(jbr, "served_judge", lambda cfg, overrides: StubJudge(reply="??"))
    out2 = tmp_path / "judged2.json"
    assert jbr.main([str(result), "--out-json", str(out2), "--judge-model", "stub",
                     "--data-dir", str(tmp_path / "data")]) == 3
