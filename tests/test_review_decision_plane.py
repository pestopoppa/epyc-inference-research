"""Tests for the reviewer decision-plane core (H3 RD-1/3/5/6/8/9 + H1 TM-3).

All model calls go through a STUB completion callable — zero real inference. Trace
emission is exercised both via an injected capturing sink and via the real
write-through path against a temp SQLite DB.

Coverage map:
  RD-1  resolve_reviewer_role binding + service default-unchanged
  RD-3  apply_verifier_precedence (fa/fr candidates, three-valued, inconclusive defer)
  RD-5  review_decision_shadow/enforce flags + warn_only shadow downgrade
  RD-6  review_candidate framing-neutral pointwise on sanitized package
  RD-8  reject-admissibility + escalate stub
  RD-9  plan rubric, reject_to_empty fallback, plan-reminder, compliance-trend hook
  TM-3  always-on shadow emission (latency_ms + tokens) on review()/review_plan()
  Regression: flags-off default path is byte-identical.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from src.features import Features, get_features, reset_features
from src.proactive_delegation.review_service import (
    FA_CANDIDATE,
    FR_CANDIDATE,
    ArchitectReviewService,
)
from src.proactive_delegation.types import ArchitectReview, PlanReviewResult, ReviewDecision
from src.roles import DEFAULT_REVIEWER_ROLE, Role, resolve_reviewer_role


# ─── Stubs / fixtures ────────────────────────────────────────────────────────


class StubPrimitives:
    """Fake LLMPrimitives: llm_call returns a canned string (records calls)."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list[dict] = []

    def llm_call(self, prompt, role=None, n_tokens=None, **kwargs):
        self.calls.append({"prompt": prompt, "role": role, "n_tokens": n_tokens})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture(autouse=True)
def _reset_features():
    yield
    reset_features()


def _service(response="", **kwargs):
    """Service with a null trace sink by default (no DB writes) unless overridden."""
    kwargs.setdefault("trace_sink", lambda ev: None)
    return ArchitectReviewService(StubPrimitives(response), **kwargs)


def _capturing_service(response="", **kwargs):
    events: list = []
    kwargs["trace_sink"] = events.append
    return ArchitectReviewService(StubPrimitives(response), **kwargs), events


def _review(decision=ReviewDecision.REJECT, evidence=None, tripwire=False, **kw):
    return ArchitectReview(
        subtask_id="S1",
        decision=decision,
        evidence=evidence or [],
        tripwire=tripwire,
        **kw,
    )


# ═══ RD-1: reviewer role binding ═════════════════════════════════════════════


class TestReviewerRoleBinding:
    def test_default_resolves_to_architect_critic(self, monkeypatch):
        # ARCHSWAP-20260927: reviews stay on the MI210 27B (operator, 2026-09-27).
        monkeypatch.delenv("ORCHESTRATOR_REVIEWER_ROLE", raising=False)
        assert resolve_reviewer_role() is DEFAULT_REVIEWER_ROLE
        assert resolve_reviewer_role() is Role.ARCHITECT_CRITIC

    def test_env_override_targets_different_model(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_REVIEWER_ROLE", "coder_escalation")
        assert resolve_reviewer_role() is Role.CODER_ESCALATION

    def test_unknown_binding_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_REVIEWER_ROLE", "no_such_role")
        assert resolve_reviewer_role() is DEFAULT_REVIEWER_ROLE

    def test_explicit_override_wins(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_REVIEWER_ROLE", "coder_escalation")
        assert resolve_reviewer_role(override="worker_math") is Role.WORKER_MATH

    def test_service_default_role_unchanged(self, monkeypatch):
        """Default service role is architect_critic — the 27B that reviewed before ARCHSWAP-20260927."""
        monkeypatch.delenv("ORCHESTRATOR_REVIEWER_ROLE", raising=False)
        svc = _service()
        assert svc.architect_role == "architect_critic"

    def test_service_reviewer_role_override(self):
        svc = _service(reviewer_role="coder_escalation")
        assert svc.architect_role == "coder_escalation"

    def test_reviewer_alias_still_resolves(self):
        # Role("reviewer") remains a stable enum-resolution fallback.
        assert Role("reviewer") is DEFAULT_REVIEWER_ROLE


# ═══ RD-5 flags ══════════════════════════════════════════════════════════════


class TestDecisionPlaneFlags:
    def test_flags_default_off(self):
        f = Features()
        assert f.review_decision_shadow is False
        assert f.review_decision_enforce is False

    def test_flags_off_in_get_features(self):
        f = get_features(production=True)
        assert f.review_decision_shadow is False
        assert f.review_decision_enforce is False

    def test_shadow_flag_has_no_memrl_dependency(self):
        """Decoupled path: enabling shadow does NOT require memrl (unlike plan_review)."""
        f = Features(review_decision_shadow=True, memrl=False)
        assert f.validate() == []

    def test_enforce_flag_no_dependency_errors(self):
        f = Features(review_decision_enforce=True, memrl=False)
        assert f.validate() == []


# ═══ Regression: flags-off default path byte-identical ═══════════════════════


class TestDefaultPathUnchanged:
    def test_review_return_byte_identical(self):
        """review() returns exactly the legacy ArchitectReview for a given emission."""
        svc = _service('{"d":"approve","s":0.9,"f":"looks good","c":["x"]}')
        out = "the output body"
        r = svc.review(spec={"objective": "obj"}, subtask={"id": "S1", "action": "a"}, output=out)
        assert r.subtask_id == "S1"
        assert r.decision == ReviewDecision.APPROVE
        assert r.score == 0.9
        assert r.feedback == "looks good"
        assert r.suggested_changes == ["x"]
        assert r.approved_output == out  # approve → carries the output
        # New RA-6 fields keep their defaults (additive, unchanged).
        assert r.confidence == 0.0
        assert r.tripwire is False

    def test_review_changes_normalization_unchanged(self):
        svc = _service('{"d":"changes","s":0.4,"f":"fix"}')
        r = svc.review(spec={}, subtask={"id": "S2", "action": "a"}, output="o")
        assert r.decision == ReviewDecision.REQUEST_CHANGES
        assert r.approved_output is None

    def test_review_exception_fallback_unchanged(self):
        svc = _service(TimeoutError("boom"))
        r = svc.review(spec={}, subtask={"id": "S3", "action": "a"}, output="o")
        assert r.decision == ReviewDecision.REQUEST_CHANGES
        assert r.score == 0.3
        assert "Review failed" in r.feedback

    def test_review_plan_return_unchanged(self):
        svc = _service('{"d":"reroute","s":0.7,"f":"fix","p":[{"step":"S1","op":"reroute","v":"architect_general"}]}')
        res = svc.review_plan(objective="o", task_type="code", plan_steps=[{"id": "S1", "actor": "coder", "action": "x"}])
        assert isinstance(res, PlanReviewResult)
        assert res.decision == "reroute"
        assert res.score == 0.7
        assert res.patches[0]["v"] == "architect_general"

    def test_review_plan_none_on_exception_unchanged(self):
        svc = _service(TimeoutError("t"))
        res = svc.review_plan(objective="o", task_type="code", plan_steps=[{"id": "S1", "actor": "coder", "action": "x"}])
        assert res is None


# ═══ TM-3: always-on shadow emission ═════════════════════════════════════════


class TestAlwaysOnEmission:
    def test_review_emits_with_latency_and_tokens(self):
        svc, events = _capturing_service('{"d":"approve","s":0.9,"f":"ok"}')
        svc.review(spec={}, subtask={"id": "S1", "action": "a"}, output="hello world output")
        assert len(events) == 1
        ev = events[0]
        assert ev.category == "review_decision"
        import json

        detail = json.loads(ev.detail_json)
        assert detail["mode"] == "review"
        assert detail["decision"] == "approve"
        assert "latency_ms" in detail and detail["latency_ms"] >= 0
        assert detail["tokens"]["tokens_out"] >= 1

    def test_review_plan_emits_on_success(self):
        svc, events = _capturing_service('{"d":"ok","s":0.9,"f":"good"}')
        svc.review_plan(objective="o", task_type="code", plan_steps=[{"id": "S1", "actor": "coder", "action": "x"}], session_id="task-42")
        assert len(events) == 1
        assert events[0].category == "review_decision"
        assert events[0].session_id == "task-42"

    def test_review_plan_emits_even_on_exception(self):
        """TM-3: emission happens regardless of whether plan_review acts (returns None)."""
        svc, events = _capturing_service(RuntimeError("x"))
        res = svc.review_plan(objective="o", task_type="code", plan_steps=[{"id": "S1", "actor": "coder", "action": "x"}])
        assert res is None
        assert len(events) == 1
        import json

        assert json.loads(events[0].detail_json)["parse_ok"] is False

    def test_review_event_tags_verifier_assigned_role(self):
        """RD-7: a review-decision turn IS a Verifier turn — the emitted review event
        carries the Trinity ``assigned_role=verifier`` axis (orthogonal to the model
        role) so tri-role shadow telemetry can capture review dispatches."""
        import json

        svc, events = _capturing_service('{"d":"approve","s":0.9,"f":"ok"}')
        svc.review(spec={}, subtask={"id": "S1", "action": "a"}, output="hello world")
        assert len(events) == 1
        detail = json.loads(events[0].detail_json)
        assert detail["assigned_role"] == "verifier"

    def test_all_review_dispatches_tag_verifier_role(self):
        """RD-7: every review-plane dispatch (review / candidate / escalate) tags the
        Trinity Verifier role, keeping the axis distinct from the model ``Event.role``."""
        import json

        svc, events = _capturing_service(
            '{"decision":"approve","confidence":0.8,"blocking":{"tripwire":false},"advisory":{"score":0.9,"feedback":"ok"}}'
        )
        svc.review_candidate({"task_ref": "T1", "objective": "x", "outputs": []}, subtask_id="C1")
        svc.escalate(_review(decision=ReviewDecision.ESCALATE))
        assert events
        for ev in events:
            assert json.loads(ev.detail_json)["assigned_role"] == "verifier"
            # Orthogonality: the model role stays the reviewer's model binding.
            assert ev.role == "architect_critic"  # ARCHSWAP-20260927 reviewer default (the 27B)

    def test_write_through_to_temp_db(self, tmp_path):
        """Real emit.py write-through lands a durable row (no injected sink)."""
        db = tmp_path / "trace.sqlite"
        svc = ArchitectReviewService(StubPrimitives('{"d":"approve","s":0.8}'), trace_db_path=str(db))
        svc.review(spec={}, subtask={"id": "S9", "action": "a"}, output="body")
        assert db.exists()
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT category, source FROM event WHERE category='review_decision'"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "review_plane"


# ═══ RD-6: framing-neutral pointwise review ══════════════════════════════════


class TestFramingNeutralReview:
    def test_prompt_has_no_competence_priming(self):
        p = ArchitectReviewService.FRAMING_NEUTRAL_REVIEW_PROMPT.lower()
        for banned in ("expert", "competent", "assume", "refined", "final answer", "senior"):
            assert banned not in p, f"framing-leaking token {banned!r} in prompt"
        # verdict-first: 'decide first' before any fix instruction
        assert "decide first" in p
        assert p.index("decide first") < p.index("fix")

    def test_valid_decision_maps_to_review(self):
        svc = _service('{"decision":"approve","confidence":0.8,"blocking":{"tripwire":false},"advisory":{"score":0.9,"feedback":"ok"}}')
        view = {"task_ref": "T1", "objective": "do X", "outputs": [{"type": "answer", "ref": "42"}], "acceptance_checks": []}
        r = svc.review_candidate(view, subtask_id="C1")
        assert r.decision == ReviewDecision.APPROVE
        assert r.confidence == 0.8
        assert r.score == 0.9
        assert r.tripwire is False

    def test_parse_failure_withholds_never_rejects(self):
        svc, events = _capturing_service("not json at all")
        r = svc.review_candidate({"task_ref": "T", "outputs": []})
        # Admissibility: a parse failure must NOT become a reject.
        assert r.decision == ReviewDecision.REQUEST_EVIDENCE
        import json

        assert json.loads(events[0].detail_json)["parse_ok"] is False

    def test_unsanitized_package_is_tolerated(self, caplog):
        svc = _service('{"decision":"approve","confidence":0.5,"blocking":{"tripwire":false}}')
        view = {
            "task_ref": "T",
            "outputs": [],
            "author_self_assessment": "I am very confident this is perfect",
        }
        # Should not crash; framing-leaking field is ignored + logged.
        r = svc.review_candidate(view)
        assert r.decision == ReviewDecision.APPROVE

    def test_consumes_only_sanitized_fields(self):
        svc = _service('{"decision":"approve","confidence":0.5,"blocking":{"tripwire":false}}')
        view = {"task_ref": "T", "objective": "OBJ-TOKEN", "outputs": [{"type": "text", "ref": "OUT-TOKEN"}], "acceptance_checks": [{"id": "A1", "statement": "CHK-TOKEN"}]}
        svc.review_candidate(view)
        prompt = svc.primitives.calls[0]["prompt"]
        assert "OBJ-TOKEN" in prompt and "OUT-TOKEN" in prompt and "CHK-TOKEN" in prompt


# ═══ RD-3: verifier precedence ═══════════════════════════════════════════════


def _report(verdict=None, checks=None, report_id="R1"):
    r = {"report_id": report_id, "checks": checks or [{"check_id": "c1", "kind": "gate", "outcome": "pass"}]}
    if verdict is not None:
        r["summary"] = {"conclusive_verdict": verdict}
    return r


class TestVerifierPrecedence:
    def test_conclusive_verdict_from_checks(self):
        cv = ArchitectReviewService._conclusive_verdict
        assert cv({"checks": [{"outcome": "pass"}, {"outcome": "pass"}]}) == "pass"
        assert cv({"checks": [{"outcome": "pass"}, {"outcome": "fail"}]}) == "fail"
        assert cv({"checks": [{"outcome": "pass"}, {"outcome": "inconclusive"}]}) == "inconclusive"
        assert cv({"checks": []}) == "inconclusive"

    def test_approve_plus_fail_emits_fa_candidate(self):
        svc, events = _capturing_service()
        review = _review(decision=ReviewDecision.APPROVE)
        report = _report(checks=[{"check_id": "c1", "kind": "gate", "outcome": "fail", "certificate": {"type": "failing_assertion", "payload": "x"}}])
        out = svc.apply_verifier_precedence(review, report)
        # objective FAIL overrides reviewer approve → withhold via request_evidence
        assert out.decision == ReviewDecision.REQUEST_EVIDENCE
        assert any(e.category == FA_CANDIDATE for e in events)
        # certificate carried into evidence
        assert any(ev.get("kind") == "gate_result" for ev in out.evidence)

    def test_reject_plus_pass_emits_fr_candidate(self):
        svc, events = _capturing_service()
        review = _review(decision=ReviewDecision.REJECT)
        out = svc.apply_verifier_precedence(review, _report(verdict="pass"))
        # objective PASS: never keep a reject → downgrade to request_evidence
        assert out.decision == ReviewDecision.REQUEST_EVIDENCE
        assert any(e.category == FR_CANDIDATE for e in events)

    def test_reject_to_empty_plus_pass_downgrades(self):
        svc, events = _capturing_service()
        out = svc.apply_verifier_precedence(_review(decision=ReviewDecision.REJECT_TO_EMPTY), _report(verdict="pass"))
        assert out.decision == ReviewDecision.REQUEST_EVIDENCE
        assert any(e.category == FR_CANDIDATE for e in events)

    def test_inconclusive_defers_to_reviewer(self):
        svc, events = _capturing_service()
        review = _review(decision=ReviewDecision.APPROVE)
        out = svc.apply_verifier_precedence(review, _report(verdict="inconclusive"))
        assert out is review  # unchanged
        assert not any(e.category in (FA_CANDIDATE, FR_CANDIDATE) for e in events)

    def test_agreement_no_override(self):
        svc, events = _capturing_service()
        # reviewer approve + objective pass → agreement, no candidate event, unchanged
        review = _review(decision=ReviewDecision.APPROVE)
        out = svc.apply_verifier_precedence(review, _report(verdict="pass"))
        assert out is review
        assert events == []


# ═══ RD-5: warn_only shadow downgrade ════════════════════════════════════════


class TestWarnOnly:
    def test_env_default_on(self, monkeypatch):
        monkeypatch.delenv("REVIEW_DECISION_WARN_ONLY", raising=False)
        assert _service().warn_only is True

    def test_env_off(self, monkeypatch):
        monkeypatch.setenv("REVIEW_DECISION_WARN_ONLY", "0")
        assert _service().warn_only is False

    def test_reject_downgraded_when_warn_only(self):
        svc, events = _capturing_service(warn_only=True)
        out = svc.apply_warn_only(_review(decision=ReviewDecision.REJECT))
        assert out.decision == ReviewDecision.REQUEST_CHANGES
        assert any(e.status == "warn_only" for e in events)

    def test_tripwire_downgraded_when_warn_only(self):
        svc, _ = _capturing_service(warn_only=True)
        out = svc.apply_warn_only(_review(decision=ReviewDecision.APPROVE, tripwire=True))
        assert out.decision == ReviewDecision.REQUEST_CHANGES
        assert out.tripwire is False

    def test_non_blocking_unchanged(self):
        svc, _ = _capturing_service(warn_only=True)
        review = _review(decision=ReviewDecision.APPROVE)
        assert svc.apply_warn_only(review) is review

    def test_warn_only_off_leaves_reject(self):
        svc, _ = _capturing_service(warn_only=False)
        review = _review(decision=ReviewDecision.REJECT)
        assert svc.apply_warn_only(review) is review


# ═══ RD-8: reject-admissibility + escalate ═══════════════════════════════════


class TestRejectAdmissibility:
    def test_reject_without_evidence_inadmissible(self):
        assert ArchitectReviewService.check_reject_admissibility(_review(decision=ReviewDecision.REJECT)) is False

    def test_reject_with_objective_evidence_admissible(self):
        r = _review(decision=ReviewDecision.REJECT, evidence=[{"kind": "gate_result", "ref": "g1"}])
        assert ArchitectReviewService.check_reject_admissibility(r) is True

    def test_reject_with_only_softevidence_inadmissible(self):
        r = _review(decision=ReviewDecision.REJECT, evidence=[{"kind": "answer_span"}])
        assert ArchitectReviewService.check_reject_admissibility(r) is False

    def test_non_reject_admissible(self):
        assert ArchitectReviewService.check_reject_admissibility(_review(decision=ReviewDecision.APPROVE)) is True

    def test_mark_flags_unverified_and_emits(self):
        svc, events = _capturing_service()
        art = svc.mark_reject_admissibility(_review(decision=ReviewDecision.REJECT))
        assert art["unverified_rejection"] is True
        assert art["decision"] == "reject"  # to_dict superset preserved
        assert any(e.status == "unverified_rejection" for e in events)

    def test_mark_admissible_reject_not_flagged(self):
        svc, events = _capturing_service()
        r = _review(decision=ReviewDecision.REJECT, evidence=[{"kind": "test_result"}])
        art = svc.mark_reject_admissibility(r)
        assert art["unverified_rejection"] is False
        assert events == []

    def test_escalate_emits_escalation_event(self):
        svc, events = _capturing_service()
        r = _review(decision=ReviewDecision.ESCALATE)
        out = svc.escalate(r, reason="needs human")
        assert out is r
        assert any(e.category == "review_escalation" for e in events)


# ═══ shadow_decide integration ═══════════════════════════════════════════════


class TestShadowDecide:
    def test_pipeline_precedence_then_admissible(self):
        svc, events = _capturing_service(warn_only=True)
        review = _review(decision=ReviewDecision.APPROVE)
        report = _report(checks=[{"check_id": "c", "kind": "gate", "outcome": "fail", "certificate": {"type": "diff", "payload": "d"}}])
        art = svc.shadow_decide(review, verification_report=report)
        assert art["shadow"] is True
        assert art["decision"] == "request_evidence"  # precedence overrode approve
        assert art["unverified_rejection"] is False

    def test_pipeline_unverified_reject_recorded_not_acted(self):
        svc, events = _capturing_service(warn_only=False)
        art = svc.shadow_decide(_review(decision=ReviewDecision.REJECT))
        assert art["decision"] == "reject"  # nothing enforced/mutated
        assert art["unverified_rejection"] is True
        assert art["shadow"] is True


# ═══ RD-9: plan-review specifics ═════════════════════════════════════════════


class TestPlanReviewSpecifics:
    def test_plan_rubric_prompt_ignores_prose(self):
        p = ArchitectReviewService.PLAN_REVIEW_RUBRIC_PROMPT.lower()
        assert "phase_coverage" in p and "order" in p and "executor_alignment" in p
        assert "prose" in p  # explicitly says ignore prose
        assert "over-specification" in p

    def test_plan_rubric_structured_output(self):
        svc, events = _capturing_service('{"decision":"approve","confidence":0.7,"phase_coverage":true,"order":true,"executor_alignment":false,"advisory":{"score":0.6,"feedback":"x"}}')
        res = svc.review_plan_rubric("obj", "code", [{"id": "S1", "actor": "coder", "action": "x"}])
        assert res["decision"] == "approve"
        assert res["executor_alignment"] is False
        assert res["score"] == 0.6
        assert any(e.status == "approve" for e in events)

    def test_plan_rubric_invalid_decision_is_a_terminal_parse_failure(self):
        """TD-21.6: an out-of-enum `decision` is a SCHEMA failure. The stub
        returns the same invalid payload for the repair turn too, so this is a
        terminal failure -> PARSE_FAILURE_SENTINEL + axes=None, never the old
        silent "approve" normalization."""
        svc = _service('{"decision":"nonsense","confidence":0.5}')
        res = svc.review_plan_rubric("o", "code", [{"id": "S1", "actor": "coder", "action": "x"}])
        assert res["decision"] == "parse_failure"
        assert res["phase_coverage"] is None

    def test_reject_to_empty_fallback(self):
        f = ArchitectReviewService.plan_review_reject_to_empty_fallback
        assert f(ReviewDecision.REJECT_TO_EMPTY) is True
        assert f("reject_to_empty") is True
        assert f(ReviewDecision.APPROVE) is False
        assert f("ok") is False

    def test_plan_reminder_fires_on_cadence(self):
        svc = _service()
        plan = [{"id": "S1", "actor": "coder", "action": "x"}]
        assert svc.build_plan_reminder(plan, cadence_n=5, step_index=5) is not None
        assert svc.build_plan_reminder(plan, cadence_n=5, step_index=3) is None
        assert svc.build_plan_reminder(plan, cadence_n=5, step_index=0) is None
        assert svc.build_plan_reminder([], cadence_n=5, step_index=5) is None

    def test_plan_reminder_emits_when_requested(self):
        svc, events = _capturing_service()
        msg = svc.build_plan_reminder([{"id": "S1", "actor": "coder", "action": "x"}], cadence_n=5, step_index=10, emit=True)
        assert msg is not None
        assert any(e.category == "plan_reminder" for e in events)

    def test_compliance_trend_hook(self):
        h = ArchitectReviewService.iteration_bound_by_compliance_trend
        assert h([0.9]) == "ok"
        assert h([0.4]) == "reminder"
        assert h([0.1]) == "replan"
        assert h([]) == "ok"
