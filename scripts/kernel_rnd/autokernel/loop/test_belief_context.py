"""The AutoKernel planner's MANDATORY belief context -- no provider call, no real alarm.

The planner runs ROOT's shared reader (`scripts/vidya/loop_beliefs.py`) as a child process.
Every test but one mocks that child with canned answers (`runner=`); the one integration test
runs the REAL child against a tmp copy of ROOT's `scripts/vidya` (EPYC_ROOT_REPO, the root dev
worktree) with a fixture ledger written through ROOT's `ClaimTuple -> to_frames -> Ledger`, a
tiny fake `sync.py` that reports fresh, and tmp --known-good/--alarm-state/--alarm-config.
Neither the canonical ledger nor the shared root worktree is written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from unittest import mock

import pytest

from autokernel.conftest import belief_ready_answer
from autokernel.loop import actors, belief_context as bc, bench, gates, loop, pipeline, resume
from autokernel.loop.loop import Abstain, Hypothesis, Outcome

GPU_27B_REGIME = {"model": {"path": "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf"},
                  "quant": "Q8_0", "backend": "gpu", "measurement_surface": "pp512"}
DS41_CPU_REGIME = {"model": {"path": "/mnt/raid0/llm/models/DeepSeek-V4.1-Flash-Q4.gguf"},
                   "quant": "Q4_K", "backend": "cpu", "measurement_surface": "pp512"}
HYPOTHESIS = {"mechanism_id": "akm-x", "statement": "s", "falsifier": "f",
              "target_surface": "ggml/src/ggml-cuda/fattn.cu", "target_symbol": "k"}
LAST_KNOWN_GOOD = {"path": "/test/kg.json", "as_of": "2026-10-20T00:00:00Z",
                   "ledger_state": {"frame_count": 2}, "ledger_state_digest": "sha256:" + "1" * 64}
KNOWN_GOOD = {"ledger_state": {"frame_count": 2}, "ledger_state_digest": "sha256:" + "1" * 64,
              "withdrawn": ["clm_dead"], "contested": ["clm_disputed"]}
OPERATOR = {"kind": "operator", "author": "daniele", "at": "2026-10-21T00:00:00Z"}


# ------------------------------------------------------------------------ harness


class Child:
    """A canned `loop_beliefs.py` child: records every argv, answers per subcommand."""

    def __init__(self, prepare=None, *, prepare_rc=0, known_good=None):
        self.prepare, self.prepare_rc = prepare, prepare_rc
        #: None = no known-good snapshot (check-seeds admits nothing).
        self.known_good = known_good
        self.calls: list[list[str]] = []

    def commands(self) -> list[str]:
        return [argv[2] for argv in self.calls]

    def argv(self, command: str) -> list[str]:
        return next(argv for argv in self.calls if argv[2] == command)

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        command = argv[2]
        if command == "prepare":
            if isinstance(self.prepare, BaseException):
                raise self.prepare
            body, rc = self.prepare, self.prepare_rc
        elif command == "check-seeds":
            body, rc = self._check(json.loads(argv[argv.index("--items-json") + 1])), 0
        else:
            body, rc = {"key": "belief-ledger-not-current.autokernel.planner",
                        "raised": "test-double"}, 0
        stdout = body if isinstance(body, str) else json.dumps(body)
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr="")

    def _check(self, items):
        """Root's `check_seeds` rule, over the canned snapshot."""
        kg = self.known_good
        rows = []
        for item in items:
            reasons, flagged = [], []
            if kg is None:
                reasons.append("no_known_good_state")
            if not item.get("seeded_against"):
                reasons.append("no_seeded_against_state")
            for premise in item.get("premises") or []:
                for status in ("withdrawn", "contested"):
                    if premise in (kg or {}).get(status, ()):
                        flagged.append({"claim_id": premise, "status": status})
                        break
            if flagged:
                reasons.append("premise_flagged")
            rows.append({**item, "admissible": not reasons, "reasons": reasons,
                         "flagged_premises": flagged})
        return {"schema": bc.SEED_CHECK_SCHEMA,
                "checked_against": (kg or {}).get("ledger_state"),
                "checked_against_digest": (kg or {}).get("ledger_state_digest"),
                "items": rows, "admissible_ids": [r["id"] for r in rows if r["admissible"]],
                "flagged_ids": [r["id"] for r in rows if r["flagged_premises"]]}


def _refused(reason="chain_broken", detail="frame 7: hash mismatch"):
    return {"schema": bc.ANSWER_SCHEMA, "consumer": bc.CONSUMER, "status": "refused",
            "reason": reason, "detail": detail, "ledger_state": None, "evidence": None,
            "alarm": {"key": "belief-ledger-not-current.autokernel.planner",
                      "raised": "raised"}, "last_known_good": LAST_KNOWN_GOOD}


def _config(tmp_path, **kw):
    return bc.BeliefConfig(root=tmp_path / "root", known_good=tmp_path / "kg.json",
                           alarm_state=tmp_path / "alarm-state.json",
                           alarm_config=tmp_path / "alarm.yaml", **kw)


def _planner(tmp_path, child, **kw):
    workspace = tmp_path / "workers" / "lane0"
    workspace.mkdir(parents=True, exist_ok=True)
    return actors.AgentPlanner(workspace=workspace, belief=_config(tmp_path),
                               belief_runner=child, **kw)


def _log_dir(tmp_path) -> Path:
    return tmp_path / "workers" / actors.ACTOR_REPLY_DIR


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _receipts(tmp_path) -> list[dict]:
    return _rows(_log_dir(tmp_path) / bc.RECEIPT_LOG)


# ------------------------------------------------- (i) ready -> planner called, receipt


def test_stale_ledger_ingested_then_ready_calls_planner_and_receipt_names_the_state(tmp_path):
    section = (f"{bc.SECTION_HEADER}\n### Contested\n- clm_c3 [abstain] kvq:...\n"
               "### Usable\n- clm_c1 [Witnessed/Attested] kvq:...\n- clm_c2 [...] kvq:...")
    answer = belief_ready_answer(sync_action="ingested", evidence={
        "status": "presented", "reasons": [], "claim_ids": ["clm_c1", "clm_c2"],
        "contested_claim_ids": ["clm_c3"], "withdrawn_claim_ids": [],
        "runs": [{"source_kind": "kvq", "run": "r1", "status": "presented"}],
        "section": section, "section_sha256": hashlib.sha256(section.encode()).hexdigest(),
        "section_bytes": len(section)})
    child = Child(answer)
    reply = json.dumps({**HYPOTHESIS, "relies_on_claims": ["clm_c1", "clm_c3"]})
    with mock.patch.object(actors, "_run_agent", return_value=reply) as run:
        hyp = _planner(tmp_path, child).propose({"current_regime": GPU_27B_REGIME})
    prompt, env = run.call_args.args[0], run.call_args.kwargs["env"]
    assert section in prompt and child.commands() == ["prepare"]
    argv = child.argv("prepare")
    assert argv[1] == str(tmp_path / "root" / "scripts" / "vidya" / "loop_beliefs.py")
    assert argv[argv.index("--consumer") + 1] == "autokernel.planner"
    assert json.loads(argv[argv.index("--target-json") + 1])["model_path"].endswith(
        "Qwen3.8-27B-Q8_0.gguf")
    assert argv[argv.index("--known-good") + 1] == str(tmp_path / "kg.json")
    assert argv[argv.index("--alarm-state") + 1] == str(tmp_path / "alarm-state.json")
    assert json.loads(env[bc.ENV_KEY])["ledger_state_digest"] == answer["ledger_state_digest"]
    [receipt] = _receipts(tmp_path)
    assert receipt["schema"] == bc.RECEIPT_SCHEMA and receipt["status"] == "ready"
    assert receipt["ledger_state"]["freshness"]["sync_action"] == "ingested"
    assert receipt["ledger_state_digest"] == answer["ledger_state_digest"]
    assert receipt["claim_ids"] == ["clm_c1", "clm_c2"]
    assert receipt["contested_claim_ids"] == ["clm_c3"]
    assert receipt["outcome"] == {"kind": "hypothesis", "mechanism_id": "akm-x"}
    # Contested is not evidence: only a presented USABLE id is accepted as reliance.
    assert receipt["reliance"]["accepted"] == ["clm_c1"]
    assert receipt["reliance"]["rejected"] == [{"claim_id": "clm_c3", "reason": "not_presented"}]
    assert receipt["call"]["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert isinstance(hyp, Hypothesis) and hyp.relies_on_claims == ("clm_c1",)
    assert hyp.belief_receipt_id == receipt["receipt_id"]
    row = Outcome(status="keep", hypothesis=hyp).to_attempt()
    assert row["relies_on_claims"] == ["clm_c1"] and "belief_mode" not in row


def test_an_inapplicable_target_is_still_a_plan_with_verified_ledger_state(tmp_path):
    with mock.patch.object(actors, "_run_agent",
                           return_value='{"abstain": "no feasible hypothesis"}') as run:
        out = _planner(tmp_path, Child(belief_ready_answer())).propose(
            {"current_regime": DS41_CPU_REGIME})
    assert isinstance(out, Abstain) and bc.SECTION_HEADER not in run.call_args.args[0]
    [receipt] = _receipts(tmp_path)
    assert receipt["status"] == "ready" and receipt["ledger_state"]["fresh"] is True
    assert receipt["evidence_reasons"] == ["inapplicable_target"] and receipt["claim_ids"] == []
    assert receipt["outcome"] == {"kind": "abstain", "abstain_reason": "no feasible hypothesis"}


def test_a_failed_call_still_leaves_an_error_receipt(tmp_path):
    with mock.patch.object(actors, "_run_agent", return_value='{"mechanism_id": "akm-x"}'), \
         mock.patch.object(actors, "_parse_reply", return_value={"mechanism_id": "akm-x"}):
        with pytest.raises(actors.ProviderTransient):
            _planner(tmp_path, Child(belief_ready_answer())).propose(
                {"current_regime": GPU_27B_REGIME})
    assert _receipts(tmp_path)[0]["outcome"]["kind"] == "error"


# ---------------------------------------------- (ii) refused -> NO planner call, ever


def test_refused_chain_broken_never_calls_the_planner_and_leaves_a_refusal_receipt(tmp_path):
    child = Child(_refused(), prepare_rc=3)
    planner = _planner(tmp_path, child)
    with mock.patch.object(actors, "_run_agent", side_effect=AssertionError("planner called")) \
            as run, mock.patch.object(actors.AgentPlanner, "_context_block",
                                      side_effect=AssertionError("prompt built")):
        with pytest.raises(bc.BeliefContextRefused) as caught:
            planner.propose({"current_regime": GPU_27B_REGIME})
    assert run.call_count == 0
    refused = caught.value
    assert (refused.reason, refused.last_known_good) == ("chain_broken", LAST_KNOWN_GOOD)
    assert isinstance(refused.degraded, bc.DegradedWork)
    # The child decided the refusal and raised the alarm itself: no consumer-side alarm.
    assert child.commands() == ["prepare"]
    [receipt] = _receipts(tmp_path)
    assert receipt["status"] == "refused" and receipt["ledger_state"] is None
    assert receipt["outcome"]["kind"] == "refused"
    assert receipt["outcome"]["reason"] == "chain_broken"
    assert receipt["refusal"]["last_known_good"] == LAST_KNOWN_GOOD
    assert receipt["call"]["prompt_sha256"] is None
    assert refused.receipt_id == receipt["receipt_id"]
    assert refused.summary()["last_known_good_digest"] == LAST_KNOWN_GOOD["ledger_state_digest"]


@pytest.mark.parametrize("failure,reason", [
    (subprocess.TimeoutExpired(["loop_beliefs"], 30), "reader_timeout"),
    (FileNotFoundError("no interpreter"), "reader_crash"),
    ("Traceback (most recent call last):\n  boom", "reader_crash"),
    ("", "reader_crash"),
])
def test_timeout_crash_or_garbage_refuses_and_raises_the_alarm(tmp_path, failure, reason):
    child = Child(failure, prepare_rc=1)
    with mock.patch.object(actors, "_run_agent", side_effect=AssertionError("planner called")) \
            as run:
        with pytest.raises(bc.BeliefContextRefused) as caught:
            _planner(tmp_path, child).propose({"current_regime": GPU_27B_REGIME})
    assert run.call_count == 0 and caught.value.reason == reason
    assert child.commands() == ["prepare", "alarm"]
    alarm = child.argv("alarm")
    assert alarm[alarm.index("--reason") + 1] == reason
    assert alarm[alarm.index("--alarm-state") + 1] == str(tmp_path / "alarm-state.json")
    assert alarm[alarm.index("--alarm-config") + 1] == str(tmp_path / "alarm.yaml")
    assert alarm[alarm.index("--escalate-after-s") + 1] == "3600.0"
    assert caught.value.alarm["invoked"] is True
    assert _receipts(tmp_path)[0]["status"] == "refused"


def test_the_reader_deadlines_are_argv_configured():
    seen = {}

    def runner(argv, **kw):
        seen.setdefault(argv[2], kw["timeout"])
        if argv[2] == "prepare":
            raise subprocess.TimeoutExpired(argv, kw["timeout"])
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    config = bc.BeliefConfig(reader_timeout_s=7.5, alarm_timeout_s=2.0, escalate_after_s=60)
    with pytest.raises(bc.BeliefContextRefused):
        bc.prepare({"current_regime": GPU_27B_REGIME}, config, runner=runner)
    assert seen == {"prepare": 7.5, "alarm": 2.0}


# ------------------------------------------- (v) construction: no unvalidated plan


@pytest.mark.parametrize("forge", [
    lambda a: {k: v for k, v in a.items() if k != "ledger_state"},
    lambda a: {**a, "ledger_state": {**a["ledger_state"], "fresh": False}},
    lambda a: {**a, "ledger_state": {**a["ledger_state"], "head_hash": ""}},
    lambda a: {**a, "ledger_state": {**a["ledger_state"], "frame_count": "3"}},
    lambda a: {**a, "evidence": None},
    lambda a: {**a, "schema": "epyc.vidya.something_else.v1"},
])
def test_a_forged_ready_answer_is_refused_before_any_planner_call(tmp_path, forge):
    child = Child(forge(belief_ready_answer()))
    with mock.patch.object(actors, "_run_agent", side_effect=AssertionError("planner called")) \
            as run:
        with pytest.raises(bc.BeliefContextRefused) as caught:
            _planner(tmp_path, child).propose({"current_regime": GPU_27B_REGIME})
    assert run.call_count == 0 and caught.value.reason == "invalid_answer"
    assert child.commands() == ["prepare", "alarm"]


def test_a_ready_answer_with_a_failing_exit_code_is_not_ready(tmp_path):
    with pytest.raises(bc.BeliefContextRefused) as caught:
        bc.prepare({"current_regime": GPU_27B_REGIME}, _config(tmp_path),
                   runner=Child(belief_ready_answer(), prepare_rc=3))
    assert caught.value.reason == "invalid_answer"


def test_the_backend_call_itself_refuses_anything_but_a_sealed_ready_answer(tmp_path):
    planner = _planner(tmp_path, Child(belief_ready_answer()))
    with mock.patch.object(actors, "_run_agent", side_effect=AssertionError("planner called")):
        for forged in (None, belief_ready_answer(), {"answer": belief_ready_answer()}):
            with pytest.raises(bc.BeliefContextRefused) as caught:
                planner._proposal_from_agent("p", planner.backend, None, {}, belief=forged)
            assert caught.value.reason == "unvalidated_answer"
    with pytest.raises(TypeError):
        bc.ReadyAnswer(belief_ready_answer(), elapsed_s=0.0, target={})
    ready = bc.prepare({"current_regime": GPU_27B_REGIME}, _config(tmp_path),
                       runner=Child(belief_ready_answer()))
    ready.answer["ledger_state"] = None          # tampered after validation
    with pytest.raises(bc.BeliefContextRefused) as caught:
        bc.require_ready(ready)
    assert caught.value.reason == "invalid_answer"


def test_there_is_no_off_mode(tmp_path):
    assert actors.BELIEF_CONTEXT_MODES == ("required",)
    for mode in ("off", "on", ""):
        with pytest.raises(ValueError, match="no planning without belief context"):
            actors.AgentPlanner(workspace=tmp_path, belief_context=mode)
    assert actors.AgentPlanner(workspace=tmp_path).belief_context == "required"


def test_run_argv_has_no_off_and_builds_the_reader_config(tmp_path):
    from autokernel.loop import run
    assert run._belief_seconds("2.5") == 2.5
    with pytest.raises(Exception):
        run._belief_seconds("0")
    args = mock.Mock(belief_root_repo=tmp_path, belief_ledger=tmp_path / "l.jsonl",
                     belief_known_good=tmp_path / "kg.json", belief_escalate_after_s=60.0,
                     belief_reader_timeout_s=45.0)
    config = run._belief_config(args)
    assert (config.root, config.ledger, config.known_good, config.escalate_after_s,
            config.reader_timeout_s) == (tmp_path, tmp_path / "l.jsonl", tmp_path / "kg.json",
                                         60.0, 45.0)


# --------------------------------------------- (iii) degraded mode in the formation loop


class _RefusingPlanner:
    """`AgentPlanner.propose` when the ledger cannot be made current."""

    def __init__(self, degraded):
        self.degraded, self.proposals, self.authored = degraded, 0, []

    def propose(self, context):
        self.proposals += 1
        refused = bc.BeliefContextRefused("chain_broken", "frame 7",
                                          last_known_good=LAST_KNOWN_GOOD)
        refused.degraded = self.degraded
        raise refused

    def author(self, hypothesis, context):
        self.authored.append(hypothesis.mechanism_id)
        return ("ggml/src/ggml-cuda/fattn.cu",)


class _Critic:
    """Degraded-mode work is still REVIEWED (only new hypothesis generation is refused)."""

    def __init__(self):
        self.reviewed, self.patches = [], []

    def review_hypothesis(self, hypothesis, context):
        self.reviewed.append(hypothesis.mechanism_id)
        return loop.Review(True)

    def review_patch(self, hypothesis, paths, context):
        self.patches.append(hypothesis.mechanism_id)
        return loop.Review(True)


def _null(hypothesis, paths):
    return bench.Comparison(surface="tg128", anchor_samples=[100.0], candidate_samples=[100.0],
                            effect=0.0, estimator="median_over_median", pairs=5,
                            noise_floor_pct=1.0, residency={"invocations": 10, "resident": 10})


def _iterate(planner, **kw):
    critic = _Critic()
    outcome = loop.iterate(planner=planner, critic=critic, context={}, measure=_null,
                           gate=lambda h, p: (True, [gates.Verdict("compile", True)]),
                           commit=lambda h, p, c: None, **kw)
    return outcome, critic


def _seed(seed_id, premises, seeded_against="sha256:" + "1" * 64):
    return {"id": seed_id, **HYPOTHESIS, "mechanism_id": f"akm-{seed_id}",
            "premises": premises, "seeded_against": seeded_against}


def _degraded(tmp_path, child, seeds=None):
    path = None
    if seeds is not None:
        path = tmp_path / "preseeded.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in seeds) + "not json\n")
    return bc.DegradedWork(log_dir=_log_dir(tmp_path), config=_config(tmp_path),
                           preseeded=path, runner=child)


def _point(mechanism, relies=(), checkpoint_id=None):
    return resume.ResumePoint(
        checkpoint_id=checkpoint_id or (mechanism[-1] * 64 + "#0"), stage="critic1",
        hypothesis=Hypothesis(**{**HYPOTHESIS, "mechanism_id": mechanism},
                              relies_on_claims=tuple(relies)),
        checkpoint={}, provenance=(), prior_patch_rejections=(), patch_rounds=1)


def test_degraded_mode_runs_an_already_planned_resume_point_reviewed_never_planned(tmp_path):
    point = _point("akm-resumed", relies=("clm_old",))
    queue = [point]
    planner = _RefusingPlanner(_degraded(tmp_path, Child(known_good=KNOWN_GOOD)))
    outcome, critic = _iterate(planner, next_resume=lambda: queue.pop() if queue else None)
    assert planner.proposals == 1                      # asked once, refused, never again
    assert critic.reviewed == ["akm-resumed"] and critic.patches == ["akm-resumed"]
    assert outcome.status == "measured_null" and outcome.hypothesis.mechanism_id == "akm-resumed"
    assert outcome.resumed_from == point.checkpoint_id
    row = outcome.to_attempt()
    assert row["belief_mode"] == "degraded"
    assert row["belief_refusal"]["reason"] == "chain_broken"
    assert row["belief_refusal"]["last_known_good_digest"] == LAST_KNOWN_GOOD["ledger_state_digest"]
    [logged] = _rows(_log_dir(tmp_path) / bc.DEGRADED_LOG)
    assert (logged["event"], logged["kind"], logged["id"], logged["premises"]) == (
        "run", "resume", point.checkpoint_id, ["clm_old"])
    # Planned under no v2 receipt: recorded, not blocking (it was already planned).
    assert logged["reasons"] == ["no_seeded_against_state"]


def test_degraded_mode_skips_a_resume_point_whose_premise_is_flagged_before_it_runs(tmp_path):
    flagged, clean = _point("akm-flagged", relies=("clm_dead",)), _point("akm-clean")
    queue = [clean, flagged]                           # pop(): flagged first
    disposed = []
    planner = _RefusingPlanner(_degraded(tmp_path, Child(known_good=KNOWN_GOOD)))
    outcome, critic = _iterate(planner, next_resume=lambda: queue.pop() if queue else None,
                               record_abandoned=disposed.append)
    assert critic.reviewed == ["akm-clean"] and outcome.hypothesis.mechanism_id == "akm-clean"
    [skip] = disposed
    assert (skip.status, skip.refusal_gate) == (loop.RESUME_REJECTED,
                                                 "resume:belief_premise_flagged")
    assert "clm_dead (withdrawn)" in skip.reasons[0] and skip.belief_mode == "degraded"
    log = _rows(_log_dir(tmp_path) / bc.DEGRADED_LOG)
    assert [(r["event"], r["id"]) for r in log] == [("skipped", flagged.checkpoint_id),
                                                    ("run", clean.checkpoint_id)]
    assert log[0]["flagged_premises"] == [{"claim_id": "clm_dead", "status": "withdrawn"}]


def test_degraded_mode_runs_an_admissible_seed_and_skips_the_rest_with_reasons(tmp_path):
    child = Child(known_good=KNOWN_GOOD)
    seeds = [_seed("flagged", ["clm_dead"]), _seed("blind", ["clm_ok"], seeded_against=None),
             _seed("good", ["clm_ok"]), _seed("operator", [], seeded_against=OPERATOR)]
    planner = _RefusingPlanner(_degraded(tmp_path, child, seeds))
    outcome, critic = _iterate(planner)
    assert planner.proposals == 1 and critic.reviewed == ["akm-good"]    # still reviewed
    assert outcome.status == "measured_null" and outcome.hypothesis.preseed_id == "good"
    row = outcome.to_attempt()
    assert (row["belief_mode"], row["preseed_id"], row["belief_premises"], row["seeded_against"]) \
        == ("degraded", "good", ["clm_ok"], "sha256:" + "1" * 64)
    log = _rows(_log_dir(tmp_path) / bc.DEGRADED_LOG)
    skipped = {r["id"]: r for r in log if r["event"] == "skipped"}
    assert skipped["flagged"]["reasons"] == ["premise_flagged"]
    assert skipped["flagged"]["flagged_premises"] == [{"claim_id": "clm_dead",
                                                       "status": "withdrawn"}]
    assert skipped["blind"]["reasons"] == ["no_seeded_against_state"]
    assert skipped[None]["reasons"] == ["invalid_seed_row"]
    assert [r["id"] for r in log if r["event"] == "run"] == ["good"]
    # The queue never re-runs a taken seed; an operator-provenance seed is admissible.
    outcome, _ = _iterate(_RefusingPlanner(_degraded(tmp_path, child, seeds)))
    assert outcome.hypothesis.preseed_id == "operator"
    assert outcome.to_attempt()["seeded_against"] == OPERATOR
    outcome, _ = _iterate(_RefusingPlanner(_degraded(tmp_path, child, seeds)))
    assert outcome.status == loop.BELIEF_REFUSED and outcome.hypothesis is None
    assert outcome.to_attempt()["belief_mode"] == "refused"
    # A checkpoint of a seeded hypothesis round-trips its marks (Hypothesis(**to_dict())).
    assert Hypothesis(**{k: v for k, v in row.items() if k in Hypothesis.__dataclass_fields__}) \
        .belief_mode == "degraded"


def test_without_a_known_good_snapshot_no_seed_runs(tmp_path):
    planner = _RefusingPlanner(_degraded(tmp_path, Child(known_good=None), [_seed("good", [])]))
    outcome, critic = _iterate(planner)
    assert outcome.status == loop.BELIEF_REFUSED and critic.reviewed == []
    [skip] = [r for r in _rows(_log_dir(tmp_path) / bc.DEGRADED_LOG) if r["id"] == "good"]
    assert "no_known_good_state" in skip["reasons"]


def test_a_failed_seed_check_runs_nothing(tmp_path):
    def broken(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    outcome, _ = _iterate(_RefusingPlanner(_degraded(tmp_path, broken, [_seed("good", [])])))
    assert outcome.status == loop.BELIEF_REFUSED
    skip = _rows(_log_dir(tmp_path) / bc.DEGRADED_LOG)[-1]
    assert skip["reasons"] == ["check_seeds_failed"] and "TimeoutExpired" in skip["check_error"]


def test_the_pool_backs_off_on_belief_refused_instead_of_spinning(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(pipeline, "_belief_wait", lambda s, stop: waits.append(s))
    planner = _RefusingPlanner(None)
    worker = pipeline.Worker("lane0", tmp_path / "lane0", tmp_path / "b0")
    outcomes = pipeline.run_pool(
        workers=[worker], make_planner=lambda w: planner, make_critic=lambda w: _Critic(),
        build_context=dict, make_gate=lambda w: (lambda h, p: (True, [])),
        make_measure=lambda w: _null, commit=lambda w, h, p, c: None,
        champion_head=lambda: "c0", reset_to_champion=lambda w: "c0",
        record=lambda o: None, iterations=4)
    assert [o.status for o in outcomes] == [loop.BELIEF_REFUSED] * 4
    assert waits == [30.0, 60.0, 120.0, 240.0] and planner.proposals == 4
    assert pipeline.belief_backoff_s(20) == pipeline.BELIEF_BACKOFF_MAX_S


def test_the_backoff_wait_honours_a_stop():
    import time
    started = time.monotonic()
    pipeline._belief_wait(120.0, lambda: True)
    assert time.monotonic() - started < 1.0


def test_belief_refused_is_a_classified_serial_outcome():
    from autokernel.loop import serial_scheduling
    assert serial_scheduling._iteration_class(loop.BELIEF_REFUSED) == "failed"


# -------------------------------------------------------- (iv) recovery re-check


def test_recovery_recheck_flags_a_degraded_result_whose_premise_is_now_withdrawn(tmp_path):
    degraded = _degraded(tmp_path, Child(known_good=KNOWN_GOOD),
                         [_seed("good", ["clm_ok"]), _seed("plain", ["clm_x"])])
    assert degraded.next_seed({"reason": "chain_broken"})["id"] == "good"
    assert degraded.next_seed({"reason": "chain_broken"})["id"] == "plain"
    # The ledger recovers, and the fresh snapshot now withdraws clm_ok.
    fresh = Child(belief_ready_answer(), known_good={
        **KNOWN_GOOD, "withdrawn": ["clm_ok"], "ledger_state_digest": "sha256:" + "2" * 64})
    with mock.patch.object(actors, "_run_agent", return_value=json.dumps(HYPOTHESIS)):
        hyp = _planner(tmp_path, fresh).propose({"current_regime": GPU_27B_REGIME})
    assert isinstance(hyp, Hypothesis)                    # recovery plans normally
    flags = {row["id"]: row for row in _rows(_log_dir(tmp_path) / bc.RECHECK_LOG)}
    assert flags["good"]["review_flag"] is True
    assert flags["good"]["flagged_premises"] == [{"claim_id": "clm_ok", "status": "withdrawn"}]
    assert flags["good"]["rechecked_against_digest"] == "sha256:" + "2" * 64
    assert flags["plain"]["review_flag"] is False
    # Re-checked once: the next ready spawns no check-seeds; nothing was reversed.
    before = len(fresh.calls)
    with mock.patch.object(actors, "_run_agent", return_value=json.dumps(HYPOTHESIS)):
        _planner(tmp_path, fresh).propose({"current_regime": GPU_27B_REGIME})
    assert [argv[2] for argv in fresh.calls[before:]] == ["prepare"]
    assert [r["id"] for r in _rows(_log_dir(tmp_path) / bc.DEGRADED_LOG)
            if r["event"] == "run"] == ["good", "plain"]


def test_a_recheck_without_a_fresh_snapshot_waits_for_the_next_ready(tmp_path):
    _degraded(tmp_path, Child(known_good=KNOWN_GOOD), [_seed("good", [])]).next_seed({})
    blind = bc.DegradedWork(log_dir=_log_dir(tmp_path), config=_config(tmp_path),
                            runner=Child(known_good=None))
    assert blind.recheck() == [] and not (_log_dir(tmp_path) / bc.RECHECK_LOG).exists()


# ------------------------------------------------------------------------ reliance


def test_shown_is_not_relied_on_and_only_presented_ids_count():
    assert bc.reliance(None, ["a", "b"]) == {"declared": [], "accepted": [], "rejected": []}
    got = bc.reliance(["a", "zz", "a", 7], ["a", "b"])
    assert got["accepted"] == ["a"]
    assert [r["claim_id"] for r in got["rejected"]] == ["zz", "7"]
    assert bc.reliance(["a"], [])["accepted"] == []


def test_metrics_row_carries_the_presented_half(tmp_path):
    ready = bc.prepare({"current_regime": GPU_27B_REGIME}, _config(tmp_path), runner=Child(
        belief_ready_answer(evidence={"status": "presented", "claim_ids": ["c1"],
                                      "frontier": 36})))
    workspace = tmp_path / "lane"
    workspace.mkdir()
    actors._record_metrics(workspace, actors.PLANNER_DEFAULT, role="planner", returncode=0,
                           wall_s=1.0, timed_out=False, before_ids=set(), collect_metrics=False,
                           schema=None, final_text=None, salvaged=False,
                           env={bc.ENV_KEY: bc.metrics_summary(ready)})
    row = json.loads((workspace.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG)
                     .read_text().splitlines()[-1])
    assert row["belief_context"]["claim_ids"] == ["c1"] and row["belief_context"]["frontier"] == 36
    assert row["belief_context"]["ledger_state_digest"] == ready.ledger_state_digest


def test_hypothesis_schema_accepts_optional_reliance_only():
    assert actors._schema_valid({**HYPOTHESIS, "relies_on_claims": ["c"]},
                                actors.HYPOTHESIS_SCHEMA)
    assert actors._schema_valid(HYPOTHESIS, actors.HYPOTHESIS_SCHEMA)
    assert not actors._schema_valid({**HYPOTHESIS, "relies_on_claims": "c"},
                                    actors.HYPOTHESIS_SCHEMA)
    assert "relies_on_claims" not in actors.HYPOTHESIS_SCHEMA["required"]


# ------------------------------------------------------------- ROOT resolution


def _roots(tmp_path, monkeypatch, *, workspace_has, view_has, env=None):
    workspace, view = tmp_path / "workspace", tmp_path / "view"
    for root, has in ((workspace, workspace_has), (view, view_has)):
        (root / "scripts" / "vidya").mkdir(parents=True)
        if has:
            (root / bc.READER_RELPATH).write_text("# reader\n")
    monkeypatch.setattr(bc, "DEFAULT_ROOT", workspace)
    monkeypatch.setattr(bc, "FALLBACK_ROOT", view)
    monkeypatch.delenv(bc.ROOT_REPO_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(bc.ROOT_REPO_ENV, str(env))
    return workspace, view


def test_root_prefers_workspace_when_it_has_the_reader(tmp_path, monkeypatch):
    workspace, _ = _roots(tmp_path, monkeypatch, workspace_has=True, view_has=True)
    assert bc.resolve_root_repo() == workspace == bc.root_repo()
    assert bc.BeliefConfig().reader() == workspace / bc.READER_RELPATH


def test_root_falls_back_to_the_hub_view_when_workspace_lacks_the_reader(tmp_path, monkeypatch):
    _, view = _roots(tmp_path, monkeypatch, workspace_has=False, view_has=True)
    assert bc.resolve_root_repo() == view
    assert bc.BeliefConfig().reader().is_file()


def test_explicit_root_wins_over_env_and_both_are_verbatim(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"          # no reader there: still taken verbatim
    _roots(tmp_path, monkeypatch, workspace_has=True, view_has=True, env=tmp_path / "env-root")
    assert bc.resolve_root_repo(elsewhere) == elsewhere
    assert bc.resolve_root_repo() == tmp_path / "env-root"
    assert bc.BeliefConfig(root=elsewhere).reader() == elsewhere / bc.READER_RELPATH


def test_neither_root_has_the_reader_stays_missing_refused_and_alarmed(tmp_path, monkeypatch):
    workspace, _ = _roots(tmp_path, monkeypatch, workspace_has=False, view_has=False)
    assert bc.resolve_root_repo() == workspace          # today's behaviour, unchanged
    config = bc.BeliefConfig()
    assert not config.reader().is_file()

    def runner(argv, **kw):
        raise FileNotFoundError(argv[1])
    with pytest.raises(bc.BeliefContextRefused) as refused:
        bc.prepare({"current_regime": GPU_27B_REGIME}, config, runner=runner)
    assert refused.value.reason == "reader_crash"
    assert refused.value.alarm["fallback"]["error"].startswith("reader and alarm channel missing")


def test_serial_run_resolves_the_same_root(tmp_path, monkeypatch):
    from . import serial_run
    assert serial_run.belief_context.resolve_root_repo is bc.resolve_root_repo
    _, view = _roots(tmp_path, monkeypatch, workspace_has=False, view_has=True)
    assert serial_run.belief_context.resolve_root_repo(None) == view


# -------------------------------------------- integration: the REAL ROOT child process

ROOT = bc.root_repo()
VIDYA = ROOT / "scripts" / "vidya"
KVQ = "kv-quant-27b-v10-measurement"
WRITTEN = "2026-09-24T10:00:00+00:00"
SHA = "a" * 64
FAKE_SYNC = '''"""Test double: the plumbing sync module, reporting a fresh ledger."""
def status(ledger_path, index_path):
    return {"fresh": True, "index_max_id": "intake-1", "synced_max_id": "intake-1",
            "last_sync": "2026-10-21T00:00:00Z", "lag_entries": 0}


def sync(ledger_path, index_path):
    return {"action": "noop", "frames_emitted": 0, "seconds": 0.0}
'''


def _root_modules():
    sys.path.insert(0, str(VIDYA))
    try:
        import claim_tuple
        import frames
        import ledger
    finally:
        sys.path.remove(str(VIDYA))
    return claim_tuple, frames, ledger


def _kvq_run(run="run-20260924T100000Z"):
    claim_tuple, _, _ = _root_modules()
    out = []
    for arm in ("A_f16_kv", "B_q8_0_kv", "C_q4_0_kv"):
        for depth in ("d2k", "d32k"):
            for metric in ("gpu_decode_tps", "gpu_prefill_tps"):
                out.append(claim_tuple.ClaimTuple(
                    measurement_id=f"kvq-{run}-{arm}-{depth}-{metric}", metric=metric,
                    value=100.0, date="2026-09-24",
                    category="BASELINE" if arm == "A_f16_kv" else "CANDIDATE",
                    claim=f"{arm} {metric} at {depth}: 100.0 tokens/s (fixture).",
                    protocol_id="P-GPU-1", reps=5, reps_basis="scored", unit="tokens/s",
                    attestation_sha256=SHA,
                    attestation_locator=(f"kvq:{run}:{arm}:{depth}:{metric}|data/run/"
                                         f"summary.json#sha256={SHA}"),
                    attestation_present=True, attestation_verified=True, source_kind=KVQ))
    return out


@pytest.mark.skipif(not (VIDYA / "loop_beliefs.py").is_file(),
                    reason=f"ROOT shared reader absent under {VIDYA}")
def test_integration_real_root_child_ready_then_recheck_after_a_retraction(tmp_path):
    claim_tuple, frames, ledger_mod = _root_modules()
    fake_root = tmp_path / "root"
    vidya = fake_root / "scripts" / "vidya"
    vidya.mkdir(parents=True)
    for source in VIDYA.glob("*.py"):
        shutil.copyfile(source, vidya / source.name)
    (vidya / "sync.py").write_text(FAKE_SYNC)         # never the real plumbing sync
    ledger_path = tmp_path / "ledger.jsonl"
    led = ledger_mod.Ledger(ledger_path)
    tuples = _kvq_run()
    for tup in tuples:
        for frame in claim_tuple.to_frames(tup, as_of=WRITTEN, adapter_id="fixture/v1"):
            led.append(frame)
    config = bc.BeliefConfig(root=fake_root, ledger=ledger_path, known_good=tmp_path / "kg.json",
                             alarm_state=tmp_path / "alarm-state.json",
                             alarm_config=tmp_path / "alarm.yaml", reader_timeout_s=60.0)
    workspace = tmp_path / "workers" / "lane0"
    workspace.mkdir(parents=True)
    planner = actors.AgentPlanner(workspace=workspace, belief=config,
                                  belief_runner=subprocess.run)
    seen = {}

    def reply(prompt, **kw):
        seen["prompt"] = prompt
        relied = re.findall(r"^- (clm_\S+) \[", prompt, flags=re.MULTILINE)[0]
        return json.dumps({**HYPOTHESIS, "relies_on_claims": [relied]})

    with mock.patch.object(actors, "_run_agent", side_effect=reply):
        hyp = planner.propose({"current_regime": GPU_27B_REGIME})
    assert bc.SECTION_HEADER in seen["prompt"]
    [receipt] = _receipts(tmp_path)
    assert receipt["status"] == "ready" and len(receipt["claim_ids"]) == 12
    assert receipt["ledger_state"]["frame_count"] == len(ledger_path.read_text().splitlines())
    assert receipt["ledger_state"]["freshness"]["sync_action"] == "already-fresh"
    assert hyp.relies_on_claims and hyp.relies_on_claims[0] in receipt["claim_ids"]
    assert (tmp_path / "kg.json").is_file()

    # Degraded-mode work relying on a claim (admitted by the REAL check-seeds against the
    # snapshot just written), then that claim's support is retracted.
    target_claim = f"clm_{tuples[3].measurement_id}"
    work = bc.DegradedWork(log_dir=_log_dir(tmp_path), config=config, runner=subprocess.run)
    verdict = work.admit_resume(
        checkpoint_id="b" * 64 + "#0",
        hypothesis={**HYPOTHESIS, "relies_on_claims": [target_claim],
                    "belief_receipt_id": receipt["receipt_id"]},
        refusal={"reason": "chain_broken"})
    assert verdict["admissible"] is True and verdict["reasons"] == []
    assert verdict["seeded_against"] == receipt["ledger_state_digest"]
    evidence = next(f for f in claim_tuple.to_frames(tuples[3], as_of=WRITTEN,
                                                     adapter_id="fixture/v1")
                    if f["frame_type"].endswith("evidence_supports_claim/v1"))
    led.append(frames.make_frame(
        frame_type="epyc.vidya/frame/retraction/v1",
        assertion={"retracts": evidence["frame_id"], "reason": "fixture"},
        provenance={"method": "fixture/v1"}, actor="fixture/v1",
        authority_scope="measurement", created_at="2026-09-24T11:00:00+00:00"))
    with mock.patch.object(actors, "_run_agent", side_effect=reply):
        planner.propose({"current_regime": GPU_27B_REGIME})
    [flag] = _rows(_log_dir(tmp_path) / bc.RECHECK_LOG)
    assert flag["review_flag"] is True
    assert flag["flagged_premises"] == [{"claim_id": target_claim, "status": "withdrawn"}]


def test_missing_reader_alarm_falls_back_to_the_root_alarm_channel(tmp_path):
    channel = tmp_path / "root" / "scripts" / "coordination" / "alarm_channel.py"
    channel.parent.mkdir(parents=True)
    channel.write_text("# stand-in; the runner below never executes it\n")
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        if argv[1].endswith("loop_beliefs.py"):        # the missing reader: python exits 2
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="No such file")
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    config = bc.BeliefConfig(root=tmp_path / "root", alarm_state=tmp_path / "s.json",
                             alarm_config=tmp_path / "c.yaml")
    out = bc.raise_alarm(config, "reader_crash", "loop_beliefs.py missing", runner=runner)
    fb = out["fallback"]
    assert fb["fallback"] is True and fb["invoked"] is True and len(calls) == 2
    argv = calls[1]
    assert argv[1] == str(channel) and "raise" in argv
    assert argv[argv.index("--severity") + 1] == "critical"
    assert argv[argv.index("--key") + 1] == f"belief-ledger-not-current-escalated.{config.consumer}"
    assert argv[argv.index("--state") + 1] == str(tmp_path / "s.json")


def test_missing_reader_and_channel_is_recorded_not_raised(tmp_path):
    out = bc.raise_alarm(bc.BeliefConfig(root=tmp_path / "nowhere"), "reader_crash", "x",
                         runner=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("x")))
    assert out["invoked"] is False
    assert out["fallback"]["invoked"] is False and "missing" in out["fallback"]["error"]


def test_the_loop_store_is_ingested_by_the_reader_before_planning(tmp_path):
    seen = []

    def runner(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(belief_ready_answer()),
                                           stderr="")

    config = bc.BeliefConfig(root=tmp_path, source_roots=(tmp_path / "store",))
    bc.prepare({"current_regime": GPU_27B_REGIME}, config, runner=runner)
    argv = seen[0]
    assert argv[argv.index("--source-root") + 1] == str(tmp_path / "store")


# ------------------------------------------- Phase 1 P1-G: DESIGN §7.4/§7.5 consumer changes

V2_SECTION = (f"{bc.SECTION_HEADER}\n### Target\n- clm_t1 [Judged] DS41 REGRESSED -8.22% (floor 3.41%)\n"
              "### Host\n- clm_h1 [Witnessed] host ceiling, not about this model")


def _v2_answer():
    return belief_ready_answer(evidence={
        "status": "presented", "reasons": [], "claim_ids": ["clm_t1", "clm_h1"],
        "contested_claim_ids": [], "withdrawn_claim_ids": [], "runs": [],
        "tiers": {"target": 1, "sibling": 0, "host": "1"},
        "tiers_shown": {"target": 1, "sibling": 0, "host": 1}, "banner": None,
        "tried_index_count": "2", "not_evidence_claim_ids": ["clm_h1"],
        "refuted_claim_ids": ["clm_r1"],
        "nonempty_target": True, "tried_index": [{"mechanism_id": "akm-a", "outcome": "null"},
                                                 {"mechanism_id": "akm-b", "outcome": "kept"}],
        "claims": [{"claim_id": "clm_t1", "tier": "target", "status": "Judged",
                    "outcome": "regressed", "mechanism_id": "akm-a", "value": -8.22,
                    "unit": "%", "effect_rel": -0.0822},
                   {"claim_id": "clm_h1", "tier": "host"}],
        "section": V2_SECTION, "section_sha256": hashlib.sha256(V2_SECTION.encode()).hexdigest(),
        "section_bytes": len(V2_SECTION.encode())})


def test_hypothesis_template_asks_for_relies_on_claims():
    assert '"relies_on_claims"' in actors._HYPOTHESIS_TASK
    assert actors.HYPOTHESIS_SCHEMA["properties"]["relies_on_claims"]["type"] == "array"
    assert "relies_on_claims" not in actors.HYPOTHESIS_SCHEMA["required"]


def test_v2_answer_keys_are_accepted_recorded_and_host_ids_are_not_reliance(tmp_path):
    reply = json.dumps({**HYPOTHESIS, "relies_on_claims": ["clm_t1", "clm_h1"]})
    with mock.patch.object(actors, "_run_agent", return_value=reply) as run:
        hyp = _planner(tmp_path, Child(_v2_answer())).propose({"current_regime": DS41_CPU_REGIME})
    prompt, env = run.call_args.args[0], run.call_args.kwargs["env"]
    assert V2_SECTION in prompt
    [receipt] = _receipts(tmp_path)
    assert receipt["section_bytes"] == len(V2_SECTION.encode())
    assert receipt["evidence_v2"] == {
        "tiers": {"target": 1, "sibling": 0, "host": 1},
        "tiers_shown": {"target": 1, "sibling": 0, "host": 1},
        "nonempty_target": True, "tried_index_count": 2, "banner": None,
        "host_claim_ids": ["clm_h1"], "refuted_claim_ids": ["clm_r1"]}
    assert receipt["reliance"]["accepted"] == ["clm_t1"]
    assert receipt["reliance"]["rejected"] == [{"claim_id": "clm_h1", "reason": "host_tier"}]
    assert hyp.relies_on_claims == ("clm_t1",)
    assert json.loads(env[bc.ENV_KEY])["v2"]["host_claim_ids"] == ["clm_h1"]


def test_an_older_answer_without_v2_keys_still_works(tmp_path):
    with mock.patch.object(actors, "_run_agent",
                           return_value=json.dumps({**HYPOTHESIS, "relies_on_claims": ["clm_c1"]})):
        answer = belief_ready_answer(evidence={
            "status": "presented", "reasons": [], "claim_ids": ["clm_c1"], "runs": [],
            "section": "S", "section_sha256": "x", "section_bytes": 1})
        hyp = _planner(tmp_path, Child(answer)).propose({"current_regime": GPU_27B_REGIME})
    [receipt] = _receipts(tmp_path)
    assert receipt["evidence_v2"] == {"tiers": None, "tiers_shown": None,
                                      "nonempty_target": None, "tried_index_count": None,
                                      "banner": None, "host_claim_ids": [],
                                      "refuted_claim_ids": []}
    assert hyp.relies_on_claims == ("clm_c1",)


def test_reliable_claim_ids_drops_host_tier():
    ev = _v2_answer()["evidence"]
    assert bc.reliable_claim_ids(ev) == ["clm_t1"]
    assert bc.reliance(["clm_h1"], ev["claim_ids"], ["clm_h1"])["accepted"] == []


def test_belief_section_sits_directly_after_already_tried_not_at_the_tail(tmp_path):
    context = {"current_regime": DS41_CPU_REGIME,
               "prior_experiments": [{"mechanism_id": "akm-old", "status": "measured_null"}],
               "inbox": ["operator note"]}
    with mock.patch.object(actors, "_run_agent",
                           return_value='{"abstain": "none"}') as run:
        _planner(tmp_path, Child(_v2_answer())).propose(context)
    prompt = run.call_args.args[0]
    assert prompt.count(V2_SECTION) == 1
    tried, belief, inbox = (prompt.index("## Already tried"), prompt.index(V2_SECTION),
                            prompt.index("## Operator suggestions"))
    assert tried < belief < inbox
