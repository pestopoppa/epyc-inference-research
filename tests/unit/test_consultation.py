"""Unit tests for one-shot internal consultation scaffolding."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.orchestration.consultation import (
    ConsultationDenied,
    build_consult_prompt,
    consult,
    _maybe_dcp_consult_context,
    load_interaction_skill,
)
from src.scheduling.contention_gate import ContentionDenied as GateContentionDenied


class _FakePrimitives:
    def __init__(self, response: str | None = None, raise_contention: bool = False):
        self.response = response or json.dumps(
            {
                "risks": ["syntax drift"],
                "blocking_issues": [],
                "confidence": 0.7,
                "recommended_delta": "keep the draft",
            }
        )
        self.raise_contention = raise_contention
        self.calls: list[dict] = []
        self.contexts: list[dict] = []

    def request_context(self, **kwargs):
        self.contexts.append(kwargs)

        class _Ctx:
            def __enter__(_self):
                return None

            def __exit__(_self, exc_type, exc, tb):
                return False

        return _Ctx()

    def llm_call(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.raise_contention:
            raise GateContentionDenied("busy")
        return self.response


def test_load_review_before_commit_skill() -> None:
    skill = load_interaction_skill("architect_critic", "review_before_commit")

    assert skill.kind == "consult"
    assert skill.max_output_tokens == 400
    assert skill.scheduler_defaults["priority"] == "background"
    assert skill.output_schema["required"] == [
        "risks",
        "blocking_issues",
        "confidence",
        "recommended_delta",
    ]
    assert len(skill.schema_hash) == 16


def test_build_consult_prompt_names_roles_and_schema() -> None:
    skill = load_interaction_skill("architect_critic", "review_before_commit")

    prompt = build_consult_prompt(
        requester_role="coder_escalation",
        consultant_role="architect_critic",
        skill=skill,
        context="draft",
    )

    assert "coder_escalation" in prompt
    assert "architect_critic" in prompt
    assert "review_before_commit" in prompt
    assert "JSON schema" in prompt


def test_consult_calls_llm_with_schema_and_scheduler_context() -> None:
    primitives = _FakePrimitives()

    advisory, stats = consult(
        "architect_critic",
        "coder_escalation",
        "review_before_commit",
        "draft context",
        primitives,
    )

    assert advisory["recommended_delta"] == "keep the draft"
    assert stats["interaction_type"] == "consult"
    assert stats["skill"] == "review_before_commit"
    assert primitives.contexts == [
        {
            "priority": "background",
            "workload_class": "consult",
            "max_queue_wait_ms": 2000,
        }
    ]
    assert primitives.calls[0]["role"] == "architect_critic"
    assert primitives.calls[0]["n_tokens"] == 400
    assert primitives.calls[0]["json_schema"]["type"] == "object"


def test_consult_rejects_schema_violations() -> None:
    primitives = _FakePrimitives(response=json.dumps({"risks": []}))

    with pytest.raises(ConsultationDenied) as exc_info:
        consult(
            "architect_critic",
            "coder_escalation",
            "review_before_commit",
            "draft context",
            primitives,
        )

    assert exc_info.value.reason == "schema_violation"


def test_consult_translates_contention_to_consultation_denied() -> None:
    primitives = _FakePrimitives(raise_contention=True)

    with pytest.raises(ConsultationDenied) as exc_info:
        consult(
            "architect_critic",
            "coder_escalation",
            "review_before_commit",
            "draft context",
            primitives,
        )

    assert exc_info.value.reason == "contention_skip"


def test_dcp_consult_context_is_default_inert(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.features.features",
        lambda: SimpleNamespace(dcp_for_consult=False, dcp_pre_assembly=True),
    )

    out = _maybe_dcp_consult_context("base", code_search_fn=lambda _q: ["x.py"])

    assert out == "base"


def test_dcp_consult_context_requires_dcp_pre_assembly(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.features.features",
        lambda: SimpleNamespace(dcp_for_consult=True, dcp_pre_assembly=False),
    )

    out = _maybe_dcp_consult_context("base", code_search_fn=lambda _q: ["x.py"])

    assert out == "base"


def test_dcp_consult_context_reuses_seed_helper(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.features.features",
        lambda: SimpleNamespace(dcp_for_consult=True, dcp_pre_assembly=True),
    )
    calls: list[dict] = []

    def fake_seed_context(query, *, code_search_fn, base_ctx, budget):
        calls.append(
            {
                "query": query,
                "code_search_fn": code_search_fn,
                "base_ctx": base_ctx,
                "budget": budget,
            }
        )
        return f"{base_ctx}\n[DCP] seeded"

    monkeypatch.setattr(
        "src.api.routes.chat_delegation._maybe_dcp_seed_context",
        fake_seed_context,
    )
    def search(_query: str) -> list[str]:
        return ["x.py"]

    out = _maybe_dcp_consult_context("base", code_search_fn=search, budget=123)

    assert out == "base\n[DCP] seeded"
    assert calls == [
        {
            "query": "base",
            "code_search_fn": search,
            "base_ctx": "base",
            "budget": 123,
        }
    ]


def test_consult_injects_dcp_augmented_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.orchestration.consultation._maybe_dcp_consult_context",
        lambda context, *, code_search_fn, budget: f"{context}\n[DCP] seeded",
    )
    primitives = _FakePrimitives()

    consult(
        "architect_critic",
        "coder_escalation",
        "review_before_commit",
        "draft context",
        primitives,
        code_search_fn=lambda _q: ["x.py"],
        dcp_budget=321,
    )

    assert "[DCP] seeded" in primitives.calls[0]["prompt"]
