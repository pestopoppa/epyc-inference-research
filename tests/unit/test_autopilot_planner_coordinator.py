"""Tests for AutoPilot provider coordination."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

planner_coordinator = importlib.import_module("planner_coordinator")
planner_providers = importlib.import_module("planner_providers")

PlannerProviderResult = planner_providers.PlannerProviderResult
PlannerSettings = planner_coordinator.PlannerSettings
MEMRL_NUMERIC_ACTION = {
    "type": "numeric_trial",
    "surface": "memrl_retrieval",
    "params": {"memrl_retrieval.semantic_k": 15},
}
REPL_NUMERIC_ACTION = {
    "type": "numeric_trial",
    "surface": "repl_executor",
    "params": {"repl.turn_token_cap": 1024},
}


def _unreachable_repair_complete(messages: Any, schema: Any) -> str:
    """Default TD-21.2/.3 repair completer for this test module: raises
    immediately instead of ever reaching a real server. `action_repair_completer()`
    (unpatched) would build an `http_chat_completer` pointed at the live
    orchestrator API (127.0.0.1:8000 by default) -- this repo's tests must
    NEVER make a live model/network request, so every test in this module
    gets this safe default via the autouse fixture below. Tests that want to
    exercise the "repaired" path monkeypatch `planner_coordinator.action_repair_completer`
    themselves to return a fake that returns canned JSON instead."""
    raise ConnectionError("no local planner reachable in unit tests")


@pytest.fixture(autouse=True)
def _no_planner_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner_coordinator, "_append_planner_archive", lambda _record: None)
    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "0")
    monkeypatch.setattr(
        planner_coordinator, "action_repair_completer", lambda: _unreachable_repair_complete
    )


class FakeProvider:
    def __init__(
        self,
        name: str,
        responses: list[PlannerProviderResult],
        *,
        supports_resume: bool = False,
    ) -> None:
        self.name = name
        self.supports_resume = supports_resume
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        prompt: str,
        *,
        role: str,
        session_id: str | None = None,
        timeout: int = 300,
        cwd: Path | str | None = None,
    ) -> PlannerProviderResult:
        self.calls.append(
            {
                "prompt": prompt,
                "role": role,
                "session_id": session_id,
                "timeout": timeout,
                "cwd": cwd,
            }
        )
        result = self.responses.pop(0)
        result.provider = self.name
        result.role = role
        return result


def _factory(providers: dict[str, FakeProvider]):
    def _get(name: str) -> FakeProvider:
        return providers[name]

    return _get


def _action_text(action: dict[str, Any]) -> str:
    import json

    return (
        "```json:autopilot_actions\n"
        f"{json.dumps(action)}\n"
        "```\n\n"
        "```json:autopilot_rationale\n"
        '{"falsifier":"x","rubric_scores":{"info_gain":3}}\n'
        "```\n"
    )


def _leading_action_text(action: dict[str, Any]) -> str:
    import json

    return (
        f"{json.dumps(action)}\n\n"
        "```json:autopilot_rationale\n"
        '{"falsifier":"x","rubric_scores":{"info_gain":3}}\n'
        "```\n"
    )


def _critique_text(payload: dict[str, Any]) -> str:
    import json

    return f"```json:autopilot_critique\n{json.dumps(payload)}\n```\n"


def test_primary_failure_falls_back_to_secondary() -> None:
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json")],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text({"type": "seed_batch", "n_questions": 10}),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state={},
        settings=PlannerSettings(mode="fallback"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == {"type": "seed_batch", "n_questions": 10}
    assert decision.draft_provider == "codex"
    assert "claude draft failed" in decision.fallback_reason
    assert len(claude.calls) == 1
    assert len(codex.calls) == 1


def test_codex_primary_can_use_distinct_codex_critic_alias() -> None:
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text(
                    {
                        "type": "structural_prune",
                        "file": "frontdoor.md",
                        "block": "## Examples",
                    }
                ),
            )
        ],
    )
    codex_critic = FakeProvider(
        "codex_critic",
        [
            PlannerProviderResult(
                provider="codex_critic",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "approve",
                        "confidence": 0.8,
                        "issues": [],
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(
            primary="codex",
            critic="codex_critic",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory({"codex": codex, "codex_critic": codex_critic}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == "codex_critic"
    assert decision.degraded is False
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == ""


def test_active_roster_treats_sol_and_terra_as_distinct_codex_models(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AUTOPILOT_PLANNER_ROSTER_POLICY_ACTIVE", "1")
    monkeypatch.setenv("AUTOPILOT_CODEX_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("AUTOPILOT_CODEX_CRITIC_MODEL", "gpt-5.6-terra")
    assert planner_coordinator._model_of("codex") == "codex:gpt-5.6-sol"
    assert planner_coordinator._model_of("codex_critic") == "codex:gpt-5.6-terra"
    assert (
        planner_coordinator._draft_fallback_provider_name(
            "codex", "codex_critic", spend_breaker_active=False
        )
        == "codex_critic"
    )


def test_fallback_draft_gets_independent_primary_critique() -> None:
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json"),
            PlannerProviderResult(
                provider="claude",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "approve",
                        "confidence": 0.86,
                        "issues": ["fallback draft is independently reviewed"],
                    }
                ),
            ),
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text(MEMRL_NUMERIC_ACTION),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == "claude"
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert [call["role"] for call in claude.calls] == ["draft", "critique"]
    assert [call["role"] for call in codex.calls] == ["draft"]
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == ""


def test_currently_unavailable_primary_draft_falls_back_before_critique() -> None:
    primary = FakeProvider(
        "local_frontdoor",
        [
            PlannerProviderResult(
                provider="local_frontdoor",
                role="draft",
                ok=True,
                text=_action_text({"type": "deep_eval", "tier": 3}),
            ),
            PlannerProviderResult(
                provider="local_frontdoor",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "approve",
                        "confidence": 0.91,
                        "issues": [],
                    }
                ),
            ),
        ],
    )
    fallback = FakeProvider(
        "local_worker",
        [
            PlannerProviderResult(
                provider="local_worker",
                role="draft",
                ok=True,
                text=_action_text(MEMRL_NUMERIC_ACTION),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(
            primary="local_frontdoor",
            critic="local_worker",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory({"local_frontdoor": primary, "local_worker": fallback}),
        allowed_action_types=["numeric_trial", "structural_experiment"],
    )

    assert decision.action == MEMRL_NUMERIC_ACTION
    assert decision.draft_provider == "local_worker"
    assert decision.critic_provider == "local_frontdoor"
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert [call["role"] for call in primary.calls] == ["draft", "critique"]
    assert [call["role"] for call in fallback.calls] == ["draft"]
    assert "local_frontdoor draft failed: action type currently unavailable: deep_eval" in (
        decision.fallback_reason
    )
    assert decision.provider_trace[0]["stage"] == "draft_primary"
    assert decision.provider_trace[0]["action_type"] == "deep_eval"
    assert (
        decision.provider_trace[0]["unusable_reason"]
        == "action type currently unavailable: deep_eval"
    )


def test_selectable_deep_eval_tier3_local_draft_and_critic_is_accepted() -> None:
    local_frontdoor = FakeProvider(
        "local_frontdoor",
        [
            PlannerProviderResult(
                provider="local_frontdoor",
                role="draft",
                ok=True,
                text=_action_text({"type": "deep_eval", "tier": 3}),
            )
        ],
    )
    local_worker = FakeProvider(
        "local_worker",
        [
            PlannerProviderResult(
                provider="local_worker",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "approve",
                        "confidence": 0.92,
                        "issues": [],
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(
            primary="local_frontdoor",
            critic="local_worker",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory(
            {
                "local_frontdoor": local_frontdoor,
                "local_worker": local_worker,
            }
        ),
        allowed_action_types=["numeric_trial", "structural_experiment", "deep_eval"],
    )

    assert decision.action == {"type": "deep_eval", "tier": 3}
    assert decision.draft_provider == "local_frontdoor"
    assert decision.critic_provider == "local_worker"
    assert decision.fallback_reason == ""
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert len(decision.provider_trace) == 2
    assert decision.provider_trace[0]["action_type"] == "deep_eval"
    assert decision.provider_trace[0]["unusable_reason"] == ""
    assert "currently unavailable" not in str(decision.provider_trace)
    assert [call["role"] for call in local_frontdoor.calls] == ["draft"]
    assert [call["role"] for call in local_worker.calls] == ["critique"]
    critique_prompt = local_worker.calls[0]["prompt"]
    assert "parsed_action_type: `deep_eval`" in critique_prompt
    assert (
        "currently_selectable_action_types: `deep_eval, numeric_trial, structural_experiment`"
        in critique_prompt
    )
    assert "`seed_batch` and `deep_eval` are valid known action types" in critique_prompt


def test_trial_planning_banner_only_writes_for_live_trial_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []

    class Tap:
        def write(self, text: str) -> None:
            writes.append(text)

        def flush(self) -> None:
            writes.append("<flush>")

        def close(self) -> None:
            writes.append("<close>")

    monkeypatch.setattr(planner_coordinator, "_open_planner_tap", lambda: Tap())

    planner_coordinator._write_trial_planning_banner(None)
    assert writes == []

    planner_coordinator._write_trial_planning_banner(1234)
    assert any("TRIAL 1234" in item for item in writes)
    assert "<flush>" in writes
    assert "<close>" in writes


def test_critique_prompt_distinguishes_known_action_from_availability() -> None:
    prompt = planner_coordinator.build_critique_prompt(
        "### Action Availability\nDo not emit seed_batch for W8 replay pressure.\n",
        _action_text({"type": "seed_batch", "n_questions": 12}),
        {"type": "seed_batch", "n_questions": 12},
        {"falsifier": "x"},
        allowed_action_types=["numeric_trial", "structural_experiment"],
    )

    assert "parsed_action_type: `seed_batch`" in prompt
    assert "seed_batch" in prompt.split("known_action_types:", 1)[1]
    assert "currently_selectable_action_types: `numeric_trial, structural_experiment`" in prompt
    assert "`seed_batch` and `deep_eval` are valid known action types" in prompt


def test_planner_archive_records_provider_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(planner_coordinator, "_append_planner_archive", records.append)

    draft_text = _action_text(MEMRL_NUMERIC_ACTION)
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=draft_text,
            )
        ],
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.9}),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        trial_id=33,
        vidya_kvq_context={"status": "unavailable", "reason": "no ingested ledger",
                           "claim_ids": [], "run": None, "frontier": None},
        settings=PlannerSettings(
            primary="claude",
            critic="codex",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.provider_trace == records[-1]["provider_trace"]
    assert records[-1]["trial_id"] == 33
    assert records[-1]["vidya_kvq"]["status"] == "unavailable"
    assert records[-1]["draft_action"] == MEMRL_NUMERIC_ACTION
    assert records[-1]["final_action"] == decision.action
    assert records[-1]["provider_trace"][0] == {
        "stage": "draft_primary",
        "role": "draft",
        "provider": "claude",
        "ok": True,
        "text_chars": len(draft_text),
        "error": "",
        "parse_ok": True,
        "action_type": "numeric_trial",
        "unusable_reason": "",
        "parse_status": "parsed",
        "repair_calls": 0,
    }
    assert records[-1]["action_parse_status"] == "parsed"
    assert records[-1]["action_repair_calls"] == 0
    assert records[-1]["rationale_parse_status"] == "parsed"
    assert records[-1]["rationale_repair_calls"] == 0
    assert records[-1]["provider_trace"][1]["stage"] == "critique_primary"
    assert records[-1]["provider_trace"][1]["parse_ok"] is True
    assert records[-1]["provider_trace"][1]["critique_decision"] == "approve"


def test_vidya_receipt_separates_exposure_from_declared_reliance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(planner_coordinator, "_append_planner_archive", records.append)
    context = {"status": "available", "reason": "", "run": "run-1",
               "frontier": 42, "state_hash": "fold-hash",
               "claim_ids": ["claim-a", "claim-b"]}
    decision = planner_coordinator.PlannerDecision(
        action=MEMRL_NUMERIC_ACTION, rationale={"vidya_claim_ids": ["claim-b", "foreign"]},
        session_id=None, canonical_text="", draft_text="", draft_provider="claude",
        mode="single", draft_action=MEMRL_NUMERIC_ACTION,
    )
    planner_coordinator._archive_decision(decision, {}, trial_id=17,
                                          vidya_kvq_context=context)
    receipt = records[-1]["vidya_kvq"]
    assert records[-1]["trial_id"] == 17
    assert receipt["exposed_claim_ids"] == ["claim-a", "claim-b"]
    assert receipt["declared_reliance_claim_ids"] == ["claim-b"]
    assert receipt["reliance_status"] == "planner_declared"
    assert receipt["run"] == "run-1" and receipt["frontier"] == 42

    decision.action = None
    planner_coordinator._archive_decision(decision, {}, trial_id=18,
                                          vidya_kvq_context=context)
    assert records[-1]["vidya_kvq"]["declared_reliance_claim_ids"] == []

    decision.action = {"type": "seed_batch"}
    planner_coordinator._archive_decision(decision, {}, trial_id=19,
                                          vidya_kvq_context=context)
    assert records[-1]["vidya_kvq"]["declared_reliance_claim_ids"] == []


def test_empty_numeric_planner_action_is_deterministically_blocked() -> None:
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(
                    {"type": "numeric_trial", "surface": "memrl_retrieval", "params": {}}
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(primary="claude", mode="single"),
        provider_factory=_factory({"claude": claude}),
    )

    assert decision.action is None
    assert "exactly one explicit params entry" in decision.deterministic_block_reason
    assert "claude draft failed" in decision.fallback_reason


def test_prior_critic_rejected_signature_is_deterministically_blocked() -> None:
    signature = planner_coordinator.action_signature(MEMRL_NUMERIC_ACTION)
    state = {
        "critic_rejected_signatures": {
            signature: {
                "trial_id": 1326,
                "reason": "critic rejected: stale exact proposal",
            }
        }
    }
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(MEMRL_NUMERIC_ACTION),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(primary="claude", mode="single"),
        provider_factory=_factory({"claude": claude}),
        action_feedback_state=state,
    )

    assert decision.action is None
    assert "previously critic-rejected at trial 1326" in decision.deterministic_block_reason
    assert "claude draft failed" in decision.fallback_reason


def test_fallback_draft_gets_primary_critique_even_when_draft_failure_opens_circuit() -> None:
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json"),
            PlannerProviderResult(
                provider="claude",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "approve",
                        "confidence": 0.86,
                        "issues": [],
                    }
                ),
            ),
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text(
                    {"type": "structural_experiment", "flags": {"user_modeling": True}}
                ),
            )
        ],
    )

    state = {"claude": {"failures": 1, "circuit_open_until": 0.0}}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state=state,
        settings=PlannerSettings(
            mode="draft_critique",
            critique_policy="always",
            circuit_failures=2,
        ),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == "claude"
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert [call["role"] for call in claude.calls] == ["draft", "critique"]
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == ""


def test_fallback_draft_pauses_when_independent_primary_critique_fails() -> None:
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(provider="claude", role="draft", ok=True, text="no json"),
            PlannerProviderResult(
                provider="claude",
                role="critique",
                ok=False,
                text="",
                error="empty response",
            ),
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text(MEMRL_NUMERIC_ACTION),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == "claude"
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == "critic_unavailable"


def test_local_failure_codex_fallback_can_dispatch_without_external_critic() -> None:
    local_ingest = FakeProvider(
        "local_ingest",
        [
            PlannerProviderResult(
                provider="local_ingest",
                role="draft",
                ok=False,
                text="",
                error="local stack unavailable",
            )
        ],
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text(MEMRL_NUMERIC_ACTION),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(
            primary="local_ingest",
            critic="codex",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory({"local_ingest": local_ingest, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.critic_provider == ""
    assert decision.critique is None
    assert decision.degraded is True
    assert "local_ingest draft failed" in decision.fallback_reason
    assert planner_coordinator._model_of("local_ingest") == "local"
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == ""
    assert [call["role"] for call in local_ingest.calls] == ["draft"]
    assert [call["role"] for call in codex.calls] == ["draft"]


def _fake_economics_ledger(*, triggered: bool) -> SimpleNamespace:
    return SimpleNamespace(
        days=7,
        planner=SimpleNamespace(total_usd=124.5),
        review=SimpleNamespace(
            planner_spend_triggered=triggered,
            projected_monthly_planner_spend_usd=541.49 if triggered else 100.0,
        ),
        rules=SimpleNamespace(planner_monthly_spend_threshold_usd=250.0),
    )


def test_spend_breaker_defaults_off_when_env_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.economics import ledger as ledger_mod

    monkeypatch.delenv("AUTOPILOT_PLANNER_SPEND_BREAKER", raising=False)

    def _unexpected_ledger(*, days: int = 7) -> SimpleNamespace:
        raise AssertionError("spend breaker should stay off unless explicitly enabled")

    monkeypatch.setattr(ledger_mod, "summarize_economics", _unexpected_ledger)
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(REPL_NUMERIC_ACTION),
            )
        ],
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.8}),
            )
        ],
    )
    state: dict[str, Any] = {"_spend_breaker": {"active": True}}

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(
            primary="claude",
            critic="codex",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory(
            {
                "claude": claude,
                "codex": codex,
            }
        ),
    )

    assert decision.draft_provider == "claude"
    assert decision.critic_provider == "codex"
    assert "_spend_breaker" not in state


def test_spend_breaker_forces_cloud_primary_to_local_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.economics import ledger as ledger_mod

    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "1")
    monkeypatch.setattr(
        ledger_mod,
        "summarize_economics",
        lambda *, days=7: _fake_economics_ledger(triggered=True),
    )
    local_ingest = FakeProvider(
        "local_ingest",
        [
            PlannerProviderResult(
                provider="local_ingest",
                role="draft",
                ok=True,
                text=_action_text(REPL_NUMERIC_ACTION),
            )
        ],
    )
    local_frontdoor = FakeProvider(
        "local_frontdoor",
        [
            PlannerProviderResult(
                provider="local_frontdoor",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.8}),
            )
        ],
    )
    state: dict[str, Any] = {}

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(
            primary="claude",
            critic="codex",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory(
            {
                "local_ingest": local_ingest,
                "local_frontdoor": local_frontdoor,
            }
        ),
    )

    assert decision.draft_provider == "local_ingest"
    assert decision.critic_provider == "local_frontdoor"
    assert decision.action == REPL_NUMERIC_ACTION
    assert state["_spend_breaker"]["active"] is True
    assert state["_spend_breaker"]["local_primary"] == "local_ingest"
    assert state["_spend_breaker"]["local_critic"] == "local_frontdoor"
    assert state["_spend_breaker"]["previous_primary"] == "claude"
    assert state["_spend_breaker"]["previous_critic"] == "codex"
    assert [call["role"] for call in local_ingest.calls] == ["draft"]
    assert [call["role"] for call in local_frontdoor.calls] == ["critique"]


def test_spend_breaker_accepts_local_ingest_leading_json_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.economics import ledger as ledger_mod

    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "1")
    monkeypatch.setattr(
        ledger_mod,
        "summarize_economics",
        lambda *, days=7: _fake_economics_ledger(triggered=True),
    )
    local_ingest = FakeProvider(
        "local_ingest",
        [
            PlannerProviderResult(
                provider="local_ingest",
                role="draft",
                ok=True,
                text=_leading_action_text({"type": "deep_eval", "tier": 3}),
            )
        ],
    )
    local_frontdoor = FakeProvider(
        "local_frontdoor",
        [
            PlannerProviderResult(
                provider="local_frontdoor",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.8}),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(
            primary="claude",
            critic="codex",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory(
            {
                "local_ingest": local_ingest,
                "local_frontdoor": local_frontdoor,
            }
        ),
    )

    assert decision.draft_provider == "local_ingest"
    assert decision.critic_provider == "local_frontdoor"
    assert decision.action == {"type": "deep_eval", "tier": 3}
    assert decision.fallback_reason == ""
    assert [call["role"] for call in local_ingest.calls] == ["draft"]
    assert [call["role"] for call in local_frontdoor.calls] == ["critique"]


def test_spend_breaker_replaces_cloud_critic_for_local_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.economics import ledger as ledger_mod

    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "1")
    monkeypatch.setattr(
        ledger_mod,
        "summarize_economics",
        lambda *, days=7: _fake_economics_ledger(triggered=True),
    )
    local_ingest = FakeProvider(
        "local_ingest",
        [
            PlannerProviderResult(
                provider="local_ingest",
                role="draft",
                ok=True,
                text=_action_text(REPL_NUMERIC_ACTION),
            )
        ],
    )
    local_frontdoor = FakeProvider(
        "local_frontdoor",
        [
            PlannerProviderResult(
                provider="local_frontdoor",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.8}),
            )
        ],
    )
    state: dict[str, Any] = {}

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(
            primary="local_chat",
            critic="codex",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory(
            {
                "local_ingest": local_ingest,
                "local_frontdoor": local_frontdoor,
            }
        ),
    )

    assert decision.draft_provider == "local_ingest"
    assert decision.critic_provider == "local_frontdoor"
    assert state["_spend_breaker"]["previous_primary"] == "local_chat"
    assert state["_spend_breaker"]["previous_critic"] == "codex"


def test_spend_breaker_preserves_configured_local_critic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.economics import ledger as ledger_mod

    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "1")
    monkeypatch.setattr(
        ledger_mod,
        "summarize_economics",
        lambda *, days=7: _fake_economics_ledger(triggered=True),
    )
    local_ingest = FakeProvider(
        "local_ingest",
        [
            PlannerProviderResult(
                provider="local_ingest",
                role="draft",
                ok=True,
                text=_action_text(REPL_NUMERIC_ACTION),
            )
        ],
    )
    local_worker = FakeProvider(
        "local_worker",
        [
            PlannerProviderResult(
                provider="local_worker",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.8}),
            )
        ],
    )
    state: dict[str, Any] = {}

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(
            primary="claude",
            critic="local_worker",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory(
            {
                "local_ingest": local_ingest,
                "local_worker": local_worker,
            }
        ),
    )

    assert decision.draft_provider == "local_ingest"
    assert decision.critic_provider == "local_worker"
    assert state["_spend_breaker"]["local_critic"] == "local_worker"
    assert state["_spend_breaker"]["previous_critic"] == "local_worker"


def test_local_role_primary_falls_back_to_local_critic_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "0")
    action = REPL_NUMERIC_ACTION
    local_frontdoor = FakeProvider(
        "local_frontdoor",
        [
            PlannerProviderResult(
                provider="local_frontdoor",
                role="draft",
                ok=True,
                text="I cannot decide.",
            ),
            PlannerProviderResult(
                provider="local_frontdoor",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.82}),
            ),
        ],
    )
    local_worker = FakeProvider(
        "local_worker",
        [
            PlannerProviderResult(
                provider="local_worker",
                role="draft",
                ok=True,
                text=_action_text(action),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(
            primary="local_frontdoor",
            critic="local_worker",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory(
            {
                "local_frontdoor": local_frontdoor,
                "local_worker": local_worker,
            }
        ),
    )

    assert decision.action == action
    assert decision.draft_provider == "local_worker"
    assert decision.critic_provider == "local_frontdoor"
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert "local_frontdoor draft failed" in decision.fallback_reason
    assert [call["role"] for call in local_frontdoor.calls] == ["draft", "critique"]
    assert [call["role"] for call in local_worker.calls] == ["draft"]


def test_spend_breaker_keeps_config_when_under_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.economics import ledger as ledger_mod

    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "1")
    monkeypatch.setattr(
        ledger_mod,
        "summarize_economics",
        lambda *, days=7: _fake_economics_ledger(triggered=False),
    )
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(REPL_NUMERIC_ACTION),
            )
        ],
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.8}),
            )
        ],
    )
    state: dict[str, Any] = {}

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(
            primary="claude",
            critic="codex",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "claude"
    assert decision.critic_provider == "codex"
    assert state["_spend_breaker"]["active"] is False


def test_nonresumable_primary_clears_persisted_session_id() -> None:
    action = {"type": "seed_batch", "n_questions": 10}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(action),
                session_id="new-session",
            )
        ],
        supports_resume=False,
    )
    codex = FakeProvider("codex", [])

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old-session",
        planner_state={},
        settings=PlannerSettings(mode="single"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == action
    assert decision.session_id is None
    assert claude.calls[0]["session_id"] is None


def test_shadow_critique_does_not_apply_revision() -> None:
    original = {"type": "code_mutation", "file": "src/escalation.py"}
    revised = {"type": "seed_batch", "n_questions": 10}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "revise",
                        "confidence": 0.8,
                        "issues": ["too risky"],
                        "revised_action": revised,
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="shadow_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == original
    assert decision.critique is not None
    assert decision.critique.decision == "revise"


def test_active_critique_applies_valid_revision() -> None:
    original = {"type": "structural_experiment", "flags": {"a": True}}
    revised = {"type": "seed_batch", "n_questions": 10}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "revise",
                        "confidence": 0.9,
                        "issues": ["missing validation"],
                        "revised_action": revised,
                        "revised_rationale": {"falsifier": "x", "rubric_scores": {}},
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == revised
    assert decision.canonical_text.startswith("```json:autopilot_actions")


def test_active_reject_without_revision_uses_numeric_fallback() -> None:
    original = {"type": "rollback", "to_checkpoint": "production_best"}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text(
                    {
                        "decision": "reject",
                        "confidence": 0.95,
                        "issues": ["unsupported rollback"],
                    }
                ),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action is not None
    assert decision.action == {
        "type": "numeric_trial",
        "surface": planner_coordinator.SAFE_FALLBACK_NUMERIC_SURFACE,
        "params": planner_coordinator.SAFE_FALLBACK_NUMERIC_PARAMS,
    }
    assert decision.rationale["critic_reject_numeric_fallback"] is True
    assert decision.rationale["critic_reject_issues"] == ["unsupported rollback"]


def test_unparseable_critique_fails_closed_not_open() -> None:
    """Regression: a critic invoke that 'succeeds' (ok=True) but returns text
    that is NOT a valid json:autopilot_critique block (e.g. Codex emitting a
    file-read error or prose) must NOT silently auto-approve the risky draft.
    It must be treated as a FAILED REVIEW: verdict "unavailable", the trusted
    primary draft KEPT (not swapped for seed_batch), critic marked degraded for
    the circuit breaker. Because the draft is HIGH-risk, the dispatch gate then
    fails CLOSED by PAUSING (critic_unavailable) — the unsafe draft is still not
    admitted unreviewed. (Behavior refined 2026-06-10.)"""
    original = {"type": "structural_experiment", "flags": {"a": True}}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            # ok=True but garbage payload — exactly the file-read-error case:
            # "Unable to read /mnt/raid0/llm/tmp/tmp....txt".
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text="Unable to read /mnt/raid0/llm/tmp/tmpXXXX.txt: file not found",
            )
        ],
    )

    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    # New contract (2026-06-10): a FAILED review does NOT substitute seed_batch —
    # the trusted-primary draft is KEPT and the verdict is "unavailable".
    assert decision.action is not None
    assert decision.action["type"] == "structural_experiment"
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert decision.critique.parse_error  # recorded, not a clean approve
    assert decision.degraded is True
    # Critic marked failed (feeds the circuit breaker).
    assert state.get("codex", {}).get("failures", 0) >= 1
    # Fail-closed for HIGH-risk: the unsafe uncritiqued draft is NOT admitted —
    # the dispatch gate pauses for operator review.
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == "critic_unavailable"


def test_extract_critique_recovers_trailing_bracket_noise() -> None:
    text = """```json:autopilot_critique
{"decision": "reject", "confidence": 0.95, "issues": ["bad draft"]}
}
```"""
    critique = planner_coordinator.extract_critique(text)
    assert critique.decision == "reject"
    assert critique.confidence == 0.95
    assert critique.issues == ["bad draft"]
    assert critique.parse_error == ""


def test_extract_critique_requires_named_fence_not_incidental_json() -> None:
    text = """This draft seems acceptable.

Draft action mentioned in prose:
{"type": "numeric_trial", "surface": "chat_pipeline", "params": {"x": 1}}
"""

    critique = planner_coordinator.extract_critique(text)

    assert critique.parse_error == "autopilot_critique fence missing"
    assert critique.decision == "approve"  # dataclass default, not a parsed approval


def test_extract_critique_requires_explicit_valid_decision() -> None:
    missing = planner_coordinator.extract_critique(
        """```json:autopilot_critique
{"confidence": 0.9, "issues": []}
```"""
    )
    invalid = planner_coordinator.extract_critique(
        """```json:autopilot_critique
{"decision": "maybe", "confidence": 0.9, "issues": []}
```"""
    )

    assert missing.parse_error == "critique decision missing"
    assert invalid.parse_error == "invalid critique decision: maybe"


def test_failed_critique_invoke_fails_closed_not_open() -> None:
    """A critic process failure (timeout, nonzero exit, empty response) on a
    HIGH-risk draft must fail closed: keep the trusted draft + verdict
    "unavailable", and the dispatch gate PAUSES (critic_unavailable) rather than
    admit it unreviewed or substitute a stale seed_batch. (Refined 2026-06-10.)"""
    original = {"type": "structural_experiment", "flags": {"a": True}}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=False,
                text="",
                error="timeout after 300s",
            )
        ],
    )

    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action is not None
    # Trusted draft KEPT (not seed_batch); verdict "unavailable".
    assert decision.action["type"] == "structural_experiment"
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert decision.critique.parse_error == "timeout after 300s"
    assert decision.degraded is True
    assert state.get("codex", {}).get("failures", 0) >= 1
    # HIGH-risk + critic unavailable => gate pauses (fails closed).
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == "critic_unavailable"


def test_codex_critic_failure_falls_back_to_claude_for_local_draft() -> None:
    original = MEMRL_NUMERIC_ACTION
    local_ingest = FakeProvider(
        "local_ingest",
        [
            PlannerProviderResult(
                provider="local_ingest",
                role="draft",
                ok=True,
                text=_action_text(original),
            )
        ],
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=False,
                text="",
                error="quota exhausted",
            )
        ],
    )
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.83}),
            )
        ],
    )
    state: dict[str, Any] = {}

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(
            primary="local_ingest",
            critic="codex",
            mode="draft_critique",
            critique_policy="always",
        ),
        provider_factory=_factory(
            {
                "local_ingest": local_ingest,
                "codex": codex,
                "claude": claude,
            }
        ),
    )

    assert decision.action == original
    assert decision.draft_provider == "local_ingest"
    assert decision.critic_provider == "claude"
    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert decision.degraded is False
    assert state.get("codex", {}).get("failures") == 1
    assert state.get("claude", {}).get("failures") == 0
    assert [call["role"] for call in codex.calls] == ["critique"]
    assert [call["role"] for call in claude.calls] == ["critique"]


def test_critique_prompt_rejects_empty_numeric_planner_request() -> None:
    prompt = "### Evidence Power and Sequential Candidate Status\nW8 replay pressure"
    text = planner_coordinator.build_critique_prompt(
        prompt,
        _action_text({"type": "numeric_trial", "surface": "think_harder", "params": {}}),
        {"type": "numeric_trial", "surface": "think_harder", "params": {}},
        {"falsifier": "x"},
    )

    assert 'Reject or revise a `numeric_trial` with missing or empty `"params"`' in text
    assert "exactly one explicit parameter" in text
    assert "internal deterministic fallbacks" in text


def test_planner_hard_block_rejects_unknown_numeric_param() -> None:
    action = {
        "type": "numeric_trial",
        "surface": "repl_executor",
        "params": {"repl_executor.tool_activation_threshold": 0.35},
    }

    reason = planner_coordinator._planner_action_hard_block_reason(action)

    assert reason is not None
    assert "not valid for surface 'repl_executor'" in reason
    assert "repl.turn_token_cap" in reason
    assert "tool_activation_threshold" in reason
    assert "detect_tool_requirement" in reason


def test_planner_hard_block_rejects_numeric_param_type_and_range() -> None:
    wrong_type = {
        "type": "numeric_trial",
        "surface": "repl_executor",
        "params": {"repl.turn_token_cap": 0.35},
    }
    too_large = {
        "type": "numeric_trial",
        "surface": "repl_executor",
        "params": {"repl.turn_token_cap": 5000},
    }

    assert "must be int" in (
        planner_coordinator._planner_action_hard_block_reason(wrong_type) or ""
    )
    assert "must be <= 4096" in (
        planner_coordinator._planner_action_hard_block_reason(too_large) or ""
    )


def test_planner_hard_block_allows_review_skip_threshold() -> None:
    action = {
        "type": "numeric_trial",
        "surface": "chat_review_skip",
        "params": {"chat.review_skip_q_threshold": 0.7},
    }

    assert planner_coordinator._planner_action_hard_block_reason(action) is None


def test_unparseable_critique_shadow_mode_keeps_draft() -> None:
    """In shadow_critique (non-binding) mode, an unparseable critique still must
    not crash or fabricate approval — the draft stands (shadow never revises),
    but the critic is still marked degraded/failed."""
    original = {"type": "seed_batch", "n_questions": 10}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude", role="draft", ok=True, text=_action_text(original)
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex", role="critique", ok=True, text="garbage not-json"
            )
        ],
    )
    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(mode="shadow_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )
    assert decision.action == original  # shadow never revises
    assert decision.critique is not None and decision.critique.parse_error
    assert decision.degraded is True


def test_default_planner_mode_is_active_draft_critique() -> None:
    """The shipped default must be the BINDING critic (the shadow→active flip)."""
    assert planner_coordinator.PlannerSettings().mode == "draft_critique"
    import os as _os

    saved = _os.environ.pop("AUTOPILOT_PLANNER_MODE", None)
    try:
        assert planner_coordinator.load_planner_settings_from_env().mode == "draft_critique"
    finally:
        if saved is not None:
            _os.environ["AUTOPILOT_PLANNER_MODE"] = saved


# ----- _reconcile matrix (gates the active-mode default, req #1) -----

PlannerCritique = planner_coordinator.PlannerCritique
_reconcile = planner_coordinator._reconcile


def test_reconcile_inactive_is_passthrough_even_on_reject() -> None:
    action = {"type": "structural_experiment", "flags": {"a": True}}
    crit = PlannerCritique(decision="reject", issues=["x"])
    out_action, out_rat, out_text = _reconcile(action, {"r": 1}, "draft", crit, active=False)
    assert out_action == action
    assert out_rat == {"r": 1}


def test_reconcile_active_approve_is_passthrough() -> None:
    action = {"type": "seed_batch", "n_questions": 10}
    crit = PlannerCritique(decision="approve")
    out_action, _, _ = _reconcile(action, {}, "draft", crit, active=True)
    assert out_action == action


def test_reconcile_active_revise_applies_valid_revision() -> None:
    action = {"type": "structural_experiment", "flags": {"a": True}}
    revised = {"type": "seed_batch", "n_questions": 12}
    crit = PlannerCritique(
        decision="revise",
        revised_action=revised,
        revised_rationale={"falsifier": "y", "rubric_scores": {}},
    )
    out_action, out_rat, out_text = _reconcile(action, {}, "draft", crit, active=True)
    assert out_action == revised
    assert out_rat == {"falsifier": "y", "rubric_scores": {}}
    assert out_text.startswith("```json:autopilot_actions")


def test_reconcile_active_revise_with_invalid_revision_uses_fallback() -> None:
    """A revise whose revised_action fails validation must not leave the rejected
    draft unchanged."""
    action = {"type": "seed_batch", "n_questions": 10}
    bad_revision = {"type": "not_a_real_action"}
    crit = PlannerCritique(decision="revise", revised_action=bad_revision)
    out_action, out_rat, _ = _reconcile(action, {}, "draft", crit, active=True)
    assert out_action == {
        "type": "numeric_trial",
        "surface": planner_coordinator.SAFE_FALLBACK_NUMERIC_SURFACE,
        "params": planner_coordinator.SAFE_FALLBACK_NUMERIC_PARAMS,
    }
    assert out_rat["critic_reject_numeric_fallback"] is True


def test_reconcile_active_reject_without_revision_is_numeric_fallback() -> None:
    action = {"type": "rollback", "to_checkpoint": "production_best"}
    crit = PlannerCritique(decision="reject", issues=["unsupported"])
    out_action, out_rat, _ = _reconcile(action, {}, "draft", crit, active=True)
    assert out_action == {
        "type": "numeric_trial",
        "surface": planner_coordinator.SAFE_FALLBACK_NUMERIC_SURFACE,
        "params": planner_coordinator.SAFE_FALLBACK_NUMERIC_PARAMS,
    }
    assert out_rat["critic_reject_numeric_fallback"] is True
    assert out_rat["critic_reject_issues"] == ["unsupported"]
    assert out_rat["critic_reject_original_action"] == action


def test_reconcile_reject_ignores_noop_structural_revision() -> None:
    action = {"type": "structural_experiment", "flags": {"graph_router": True}}
    noop_revision = {"type": "structural_experiment", "flags": {"specialist_routing": True}}
    crit = PlannerCritique(
        decision="reject",
        issues=["dependency first"],
        revised_action=noop_revision,
    )
    prompt = (
        "### Feature Flags (live state + dependency rules)\n"
        "  - specialist_routing (currently ON)\n"
        "  - graph_router (currently OFF) requires [specialist_routing=ON]\n"
    )

    out_action, out_rat, _ = _reconcile(
        action,
        {},
        "draft",
        crit,
        active=True,
        allowed_action_types=["numeric_trial", "structural_experiment"],
        planner_prompt=prompt,
    )

    assert out_action == {
        "type": "numeric_trial",
        "surface": planner_coordinator.SAFE_FALLBACK_NUMERIC_SURFACE,
        "params": planner_coordinator.SAFE_FALLBACK_NUMERIC_PARAMS,
    }
    assert out_rat["critic_reject_numeric_fallback"] is True
    assert out_rat["critic_reject_original_action"] == action


def test_decision_carries_original_draft_action_after_substitution() -> None:
    """draft_action must preserve the planner's ORIGINAL action even when the
    binding critic substitutes it (so the loop can record the rejected draft)."""
    original = {"type": "structural_experiment", "flags": {"graph_router": True}}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude", role="draft", ok=True, text=_action_text(original)
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "reject", "issues": ["deps unmet"]}),
            )
        ],
    )
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )
    assert decision.action == {
        "type": "numeric_trial",
        "surface": planner_coordinator.SAFE_FALLBACK_NUMERIC_SURFACE,
        "params": planner_coordinator.SAFE_FALLBACK_NUMERIC_PARAMS,
    }  # substituted
    assert decision.draft_action == original  # original preserved


def test_critique_prompt_surfaces_flag_and_feedback_context() -> None:
    """The critic sees the measurement/constraint view, not the full prompt."""
    planner_prompt = (
        "## Original Planning Instructions\n"
        "FULL_PLANNER_MARKER_UNIQUE should never reach the critic.\n"
        "### Evidence Power and Sequential Candidate Status\n"
        "  - quality MDE: 0.070; seq verdict: accumulating\n"
        "### System Health\n"
        "  - nominal; no host contention detected\n"
        "### Action Availability\n"
        "  - graph_router unavailable until specialist_routing is OFF\n"
        "### Blacklisted Configurations\n  some-entry\n"
        "### Feature Flags (live state + dependency rules)\n"
        "  - graph_router (currently OFF) requires [specialist_routing=OFF]\n"
        "### Last Non-Executing Action (validator/dispatch feedback)\n"
        "  reason: graph_router feature requires specialist_routing feature\n"
        "### Experiment Journal\n"
        "  SHOULD_NOT_REACH_CRITIC\n"
    )
    out = planner_coordinator.build_critique_prompt(
        planner_prompt,
        "draft text",
        {"type": "structural_experiment", "flags": {"graph_router": True}},
        {},
    )
    # Selected context embedded:
    assert "Selected Measurement and Constraint Context" in out
    assert "Evidence Power and Sequential Candidate Status" in out
    assert "quality MDE: 0.070" in out
    assert "System Health" in out
    assert "Action Availability" in out
    assert "Feature Flags" in out
    assert "Last Non-Executing Action" in out
    assert "Blacklisted Configurations" in out
    assert "specialist_routing" in out
    # Full planner context not embedded:
    assert "Original Planner Context" not in out
    assert "FULL_PLANNER_MARKER_UNIQUE" not in out
    assert "SHOULD_NOT_REACH_CRITIC" not in out
    # Instructions tell the critic to use them + that the verdict is binding:
    assert "BINDING" in out
    assert "dependencies are not all currently ON" in out
    assert "below-MDE" in out
    assert "unrelated dirty files" in out
    assert "target-path dirty fence" in out


def test_critique_prompt_caps_selected_sections() -> None:
    long_feature_section = "x" * 2000
    planner_prompt = f"### Feature Flags\n{long_feature_section}\nTAIL_MARKER_AFTER_LIMIT\n"
    out = planner_coordinator.build_critique_prompt(
        planner_prompt,
        "draft text",
        {"type": "seed_batch"},
        {},
    )
    assert "... [truncated for critic context]" in out
    assert "TAIL_MARKER_AFTER_LIMIT" not in out


def test_critique_prompt_uses_safe_fallback_when_selected_sections_absent() -> None:
    out = planner_coordinator.build_critique_prompt(
        "## Planner internals\nFULL_PLANNER_MARKER_UNIQUE\n",
        "draft text",
        {"type": "seed_batch"},
        {},
    )
    assert "selected planner context unavailable" in out
    assert "FULL_PLANNER_MARKER_UNIQUE" not in out


def test_open_primary_circuit_routes_directly_to_fallback() -> None:
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text="unused")],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text({"type": "seed_batch", "n_questions": 10}),
            )
        ],
    )
    state = {"claude": {"circuit_open_until": time.time() + 60}}

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id="old",
        planner_state=state,
        settings=PlannerSettings(mode="fallback"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.draft_provider == "codex"
    assert decision.fallback_reason == "claude circuit open"
    assert claude.calls == []
    assert len(codex.calls) == 1


def _uncritiqued_decision(
    action: Any,
    *,
    degraded: bool,
    critique: Any,
    rationale: Any = None,
    mode: str = "draft_critique",
):
    return planner_coordinator.PlannerDecision(
        action=action,
        rationale=rationale or {},
        session_id=None,
        canonical_text="",
        draft_text="",
        draft_provider="codex",
        mode=mode,
        degraded=degraded,
        critique=critique,
    )


def test_uncritiqued_degraded_nonobservational_action_pauses() -> None:
    """Degraded with NO critic verdict + non-observational action => critic_unavailable.
    seed_batch is explicitly NOT safe (the @708 critic_reject_loop failure mode)."""
    ubr = planner_coordinator.uncritiqued_dispatch_block_reason
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "seed_batch", "n_questions": 12}, degraded=True, critique=None
            )
        )
        == "critic_unavailable"
    )
    # OBSERVATIONAL_ACTIONS is empty => every non-observational action blocks.
    assert (
        ubr(_uncritiqued_decision({"type": "structural_experiment"}, degraded=True, critique=None))
        == "critic_unavailable"
    )


def test_uncritiqued_gate_allows_when_critiqued_or_not_degraded() -> None:
    ubr = planner_coordinator.uncritiqued_dispatch_block_reason
    crit = planner_coordinator.PlannerCritique(decision="approve", provider="codex")
    # real critic verdict => not uncritiqued => no pause
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "seed_batch", "n_questions": 12}, degraded=True, critique=crit
            )
        )
        == ""
    )
    # not degraded => no pause
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "seed_batch", "n_questions": 12}, degraded=False, critique=None
            )
        )
        == ""
    )
    # no dict action => handled by the separate no-action path => no pause
    assert ubr(_uncritiqued_decision(None, degraded=True, critique=None)) == ""


def test_draft_validation_error_is_surfaced_not_opaque() -> None:
    """#776 root cause (2026-06-11): a draft that PARSES but violates a schema cap
    (train_routing_models min_memories > 100000) must surface the EXACT validator
    error in fallback_reason, not an opaque 'invalid action' — so the next trial's
    feedback can self-correct."""
    bad = {"type": "train_routing_models", "min_memories": 250000}
    good = {"type": "train_routing_models", "min_memories": 500}
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="draft", ok=True, text=_action_text(bad))],
    )
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text=_action_text(good))],
        supports_resume=True,
    )
    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(primary="codex", critic="claude", mode="fallback"),
        provider_factory=_factory({"codex": codex, "claude": claude}),
    )
    assert decision.fallback_reason
    assert "invalid action" not in decision.fallback_reason
    assert "min_memories" in decision.fallback_reason
    assert "100000" in decision.fallback_reason
    assert decision.action == good  # the valid fallback draft was adopted


def test_uncritiqued_unavailable_dispatch_rule() -> None:
    """Tightened Case-B rule (2026-06-12): verdict 'unavailable' on a TRUSTED
    primary draft. Risk class alone is NOT the guard (the @708 failure was
    'low-risk' seed looping):
      - HIGH risk => pause.
      - seed_batch / passive low-risk => pause (loop-prone class).
      - MEDIUM experiment => proceed ONLY IF novel + non-looping (not blacklisted,
        not repeated, carries a falsifier); else pause.
      - shadow (non-binding) mode => never blocks."""
    ubr = planner_coordinator.uncritiqued_dispatch_block_reason
    unavail = planner_coordinator.PlannerCritique(
        decision="unavailable", provider="claude", parse_error="timeout"
    )
    falsifier = {"falsifier": "tps drops >5% if X regresses"}

    # HIGH risk => pause
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "structural_experiment"},
                degraded=True,
                critique=unavail,
                rationale=falsifier,
            )
        )
        == "critic_unavailable"
    )
    # seed_batch (loop-prone low-risk) => pause even with a falsifier
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "seed_batch", "n_questions": 12},
                degraded=True,
                critique=unavail,
                rationale=falsifier,
            )
        )
        == "critic_unavailable"
    )
    # passive low-risk (deep_eval) => pause (not in OBSERVATIONAL_ACTIONS)
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "deep_eval", "tier": 2},
                degraded=True,
                critique=unavail,
                rationale=falsifier,
            )
        )
        == "critic_unavailable"
    )

    # MEDIUM experiment, novel + falsifier => PROCEED
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "numeric_trial"}, degraded=True, critique=unavail, rationale=falsifier
            )
        )
        == ""
    )
    # MEDIUM but no falsifier (loop-keeping) => pause
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "numeric_trial"}, degraded=True, critique=unavail, rationale={}
            )
        )
        == "critic_unavailable"
    )
    # MEDIUM + blacklisted => pause
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "numeric_trial"}, degraded=True, critique=unavail, rationale=falsifier
            ),
            is_blacklisted=True,
        )
        == "critic_unavailable"
    )
    # MEDIUM + repeated (recurring invalid signature) => pause
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "numeric_trial"}, degraded=True, critique=unavail, rationale=falsifier
            ),
            is_repeated=True,
        )
        == "critic_unavailable"
    )

    # shadow (non-binding) mode never blocks, even for HIGH risk
    assert (
        ubr(
            _uncritiqued_decision(
                {"type": "structural_experiment"},
                degraded=True,
                critique=unavail,
                rationale=falsifier,
                mode="shadow_critique",
            )
        )
        == ""
    )


def test_failed_critique_keeps_draft_seed_batch_still_pauses() -> None:
    """End-to-end (2026-06-12): a critic failure on a seed_batch draft KEEPS the
    draft (verdict 'unavailable', not a stale seed_batch *substitution*) — but the
    tightened gate still PAUSES it, because seed looping is the exact @708 failure
    the critic exists to catch. Risk class alone must not let it through."""
    original = {"type": "seed_batch", "n_questions": 12}  # loop-prone low-risk
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude", role="draft", ok=True, text=_action_text(original)
            )
        ],
        supports_resume=True,
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex", role="critique", ok=False, text="", error="timeout after 600s"
            )
        ],
    )
    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )
    # Draft KEPT (not the seed_batch-fallback substitution), verdict 'unavailable'.
    assert decision.action == original
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert decision.degraded is True
    assert state.get("codex", {}).get("failures", 0) >= 1
    # seed_batch is loop-prone => the gate pauses for operator review.
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == "critic_unavailable"


def test_cross_model_failover_codex_offline_crosses_to_claude() -> None:
    # PRIMARY=codex + CRITIC=codex_critic both resolve to the codex binary. When
    # codex is offline the failover draft MUST cross to a DIFFERENT model (claude),
    # not re-hit codex. (cross-model failover, 2026-06-12)
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex", role="draft", ok=False, text="", error="timeout after 600s"
            )
        ],
    )
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text({"type": "seed_batch", "n_questions": 10}),
            )
        ],
    )

    def factory(name: str) -> FakeProvider:
        # Both configured codex roles map to the codex stub; claude to claude.
        if name in ("codex", "codex_critic"):
            return codex
        return claude

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(primary="codex", critic="codex_critic", mode="draft_critique"),
        provider_factory=factory,
    )

    assert decision.action == {"type": "seed_batch", "n_questions": 10}
    assert decision.draft_provider == "claude"
    assert decision.providers_unavailable is False


def test_both_models_unavailable_sets_providers_unavailable() -> None:
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="draft", ok=False, text="", error="timeout")],
    )
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude", role="draft", ok=False, text="", error="empty response"
            )
        ],
    )

    def factory(name: str) -> FakeProvider:
        if name in ("codex", "codex_critic"):
            return codex
        return claude

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(primary="codex", critic="codex_critic", mode="draft_critique"),
        provider_factory=factory,
    )

    assert decision.action is None
    assert decision.providers_unavailable is True
    assert "both planner models unavailable" in decision.fallback_reason


def test_content_failure_is_not_providers_unavailable() -> None:
    # Both models RESPOND ok but neither emits a parseable action block. This is a
    # CONTENT failure, not an availability failure → providers_unavailable False.
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="draft", ok=True, text="no json")],
    )
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text="also no json")],
    )

    def factory(name: str) -> FakeProvider:
        if name in ("codex", "codex_critic"):
            return codex
        return claude

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(primary="codex", critic="codex_critic", mode="draft_critique"),
        provider_factory=factory,
    )

    assert decision.action is None
    assert decision.providers_unavailable is False


def test_open_primary_circuit_routes_to_other_model() -> None:
    # A pre-open primary (codex) circuit must route the draft to the OTHER model
    # (claude), even though the configured critic (codex_critic) is also codex.
    import time as _time

    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude",
                role="draft",
                ok=True,
                text=_action_text({"type": "seed_batch", "n_questions": 10}),
            )
        ],
    )
    codex = FakeProvider("codex", [])  # must NOT be invoked

    def factory(name: str) -> FakeProvider:
        if name in ("codex", "codex_critic"):
            return codex
        return claude

    state = {"codex": {"circuit_open_until": _time.time() + 600.0, "failures": 2}}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(primary="codex", critic="codex_critic", mode="draft_critique"),
        provider_factory=factory,
    )

    assert decision.draft_provider == "claude"
    assert decision.action == {"type": "seed_batch", "n_questions": 10}
    assert len(codex.calls) == 0


def test_local_briefed_aliases_are_classified_as_local_model() -> None:
    assert planner_coordinator._model_of("local_brief_frontdoor") == "local"
    assert planner_coordinator._model_of("local_ingest_frontdoor") == "local"
    assert planner_coordinator._model_of("local_brief_worker") == "local"


# ----- TD-21.2/.3/.17: repair-aware draft-action extraction -----


def test_recovered_parse_repairs_action_without_fallback_or_circuit_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TD-21.17: a malformed-but-repairable ACTION block must NOT trip the
    per-provider circuit breaker and must NOT trigger a from-zero fallback
    re-prompt. `mode="fallback"` deliberately leaves the fallback provider
    ELIGIBLE (`allow_fallback=True`) so this proves it is skipped because the
    repaired action makes the draft usable, not because fallback was
    unavailable."""
    import json

    draft_text = "I will run a seed batch of 10 questions next, type seed_batch."
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text=draft_text)],
    )
    codex = FakeProvider("codex", [])  # must never be invoked

    def fake_complete(messages, schema):
        return json.dumps({"type": "seed_batch", "n_questions": 10})

    monkeypatch.setattr(planner_coordinator, "action_repair_completer", lambda: fake_complete)

    planner_state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=planner_state,
        settings=PlannerSettings(mode="fallback"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == {"type": "seed_batch", "n_questions": 10}
    assert decision.action_parse_status == "repaired"
    assert decision.action_repair_calls == 1
    assert len(codex.calls) == 0  # no fallback re-prompt from zero
    assert planner_state["claude"]["failures"] == 0  # success, not a circuit-breaker failure
    assert planner_coordinator._circuit_is_open(planner_state, "claude") is False


def test_unrepairable_action_still_falls_back_and_trips_circuit_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contrast case: when repair ALSO fails, behavior is unchanged from
    before this conversion -- fallback IS invoked and the circuit breaker
    DOES accumulate failures."""
    draft_text = "no usable action in this reply at all"
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text=draft_text)],
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="draft",
                ok=True,
                text=_action_text({"type": "seed_batch", "n_questions": 10}),
            )
        ],
    )

    def failing_complete(messages, schema):
        return "still not json"

    monkeypatch.setattr(planner_coordinator, "action_repair_completer", lambda: failing_complete)

    planner_state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=planner_state,
        settings=PlannerSettings(mode="fallback"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action == {"type": "seed_batch", "n_questions": 10}
    assert decision.draft_provider == "codex"
    assert len(codex.calls) == 1  # fallback WAS invoked
    assert planner_state["claude"]["failures"] == 1  # circuit breaker DID accumulate


def test_rationale_repair_recovers_malformed_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    draft_text = (
        "```json:autopilot_actions\n"
        f"{json.dumps(MEMRL_NUMERIC_ACTION)}\n"
        "```\n\n"
        "```json:autopilot_rationale\n"
        '{"falsifier": "x", "rubric_scores": {"info_gain": 4}\n'  # missing closing brace
        "```\n"
    )
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text=draft_text)],
    )
    codex = FakeProvider("codex", [])

    def fake_complete(messages, schema):
        return json.dumps({"falsifier": "x", "rubric_scores": {"info_gain": 4}})

    monkeypatch.setattr(planner_coordinator, "action_repair_completer", lambda: fake_complete)

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="single"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    # The action fence parses clean (0 repair calls); only the rationale
    # sidecar needed the repair turn.
    assert decision.action == MEMRL_NUMERIC_ACTION
    assert decision.action_parse_status == "parsed"
    assert decision.rationale == {"falsifier": "x", "rubric_scores": {"info_gain": 4}}
    assert decision.rationale_parse_status == "repaired"
    assert decision.rationale_repair_calls == 1


def test_rationale_repair_unrepairable_persists_empty_default_flagged_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    draft_text = (
        "```json:autopilot_actions\n"
        f"{json.dumps(MEMRL_NUMERIC_ACTION)}\n"
        "```\n\n"
        "```json:autopilot_rationale\nnot json at all\n```\n"
    )
    claude = FakeProvider(
        "claude",
        [PlannerProviderResult(provider="claude", role="draft", ok=True, text=draft_text)],
    )
    codex = FakeProvider("codex", [])

    def failing_complete(messages, schema):
        return "still garbage"

    monkeypatch.setattr(planner_coordinator, "action_repair_completer", lambda: failing_complete)

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="single"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.rationale == {"falsifier": "", "rubric_scores": {}}
    assert decision.rationale_parse_status == "failed"


# ----- TD-21.16: repair-aware critic-verdict extraction -----


def test_recovered_critic_verdict_repairs_without_fallback_or_circuit_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed-but-repairable critic verdict must NOT trip the critic's
    per-provider circuit breaker and must NOT spend a fallback-critic call.
    `codex` (critic) and `claude` (draft) each queue exactly ONE response --
    an unexpected second call to either raises IndexError, proving no
    fallback/re-review was attempted."""
    import json

    original = {"type": "structural_experiment", "flags": {"a": True}}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude", role="draft", ok=True, text=_action_text(original)
            )
        ],
    )
    malformed = (
        "The draft looks fine overall. My decision is to approve it. "
        "confidence 0.8. No real issues here."
    )
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="critique", ok=True, text=malformed)],
    )

    complete_calls: list[Any] = []

    def fake_complete(messages, schema):
        complete_calls.append((messages, schema))
        return json.dumps({"decision": "approve", "confidence": 0.8, "issues": []})

    monkeypatch.setattr(planner_coordinator, "action_repair_completer", lambda: fake_complete)

    planner_state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=planner_state,
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert decision.critique.confidence == 0.8
    assert decision.critique.parse_error == ""
    assert decision.critique.parse_status == "repaired"
    assert decision.critique.repair_calls == 1
    assert len(complete_calls) == 1  # exactly one repair turn
    assert len(codex.calls) == 1  # no fallback re-review from the critic
    assert planner_state["codex"]["failures"] == 0  # success, not a circuit-breaker failure
    assert planner_coordinator._circuit_is_open(planner_state, "codex") is False
    assert decision.degraded is False


def test_unrepairable_critic_verdict_is_typed_failure_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contrast case: when the TD-21.16 repair turn ALSO fails, the existing
    fallback-critic-provider / circuit-breaker path fires exactly as it did
    before this conversion (the default codex-critic fallback is `claude`,
    same model as the draft provider here, so `_critic_fallback_provider_name`
    is disabled -- same precondition as every pre-existing fail-closed test in
    this module). The unrecoverable verdict is a typed failure
    (`parse_status="failed"`), never a default verdict, and is counted in the
    shared `STRUCTURED_OUTPUT_REPAIR_COUNTS` telemetry."""
    from src.structured_output.repair import STRUCTURED_OUTPUT_REPAIR_COUNTS

    original = {"type": "structural_experiment", "flags": {"a": True}}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude", role="draft", ok=True, text=_action_text(original)
            )
        ],
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text="Unable to read /mnt/raid0/llm/tmp/tmpXXXX.txt: file not found",
            )
        ],
    )

    def failing_complete(messages, schema):
        return "still not json"

    monkeypatch.setattr(planner_coordinator, "action_repair_completer", lambda: failing_complete)

    key = (planner_coordinator._CRITIC_REPAIR_SITE, "failed")
    before = STRUCTURED_OUTPUT_REPAIR_COUNTS.get(key, 0)

    state: dict[str, Any] = {}
    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state=state,
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.action is not None
    assert decision.action["type"] == "structural_experiment"  # trusted draft KEPT
    assert decision.critique is not None
    assert decision.critique.decision == "unavailable"
    assert decision.critique.parse_status == "failed"
    assert decision.critique.repair_calls == 1
    assert decision.degraded is True
    assert state.get("codex", {}).get("failures", 0) >= 1  # circuit breaker DID accumulate
    assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(key, 0) == before + 1  # counted
    assert planner_coordinator.uncritiqued_dispatch_block_reason(decision) == "critic_unavailable"


def test_clean_critic_verdict_never_invokes_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: a clean, cleanly-fenced critic verdict costs 0 repair
    calls and never touches the repair completer at all."""

    def boom(messages, schema):
        raise AssertionError("repair completer must not be called on a clean verdict")

    monkeypatch.setattr(planner_coordinator, "action_repair_completer", lambda: boom)

    original = MEMRL_NUMERIC_ACTION
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude", role="draft", ok=True, text=_action_text(original)
            )
        ],
    )
    codex = FakeProvider(
        "codex",
        [
            PlannerProviderResult(
                provider="codex",
                role="critique",
                ok=True,
                text=_critique_text({"decision": "approve", "confidence": 0.9, "issues": []}),
            )
        ],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.critique is not None
    assert decision.critique.decision == "approve"
    assert decision.critique.parse_status == "parsed"
    assert decision.critique.repair_calls == 0


def test_external_cli_critic_trailing_noise_recovered_by_fish_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An external-CLI critic (codex) commonly appends trailing bracket noise
    after its closing fence. The shared deterministic `fish_json` balancer
    already recovers this WITHOUT a repair turn (0 calls) -- proving the
    TD-21.16 conversion does not spend a completion turn recovering noise the
    existing fish step already handles, regardless of which CLI/transport
    produced the critique."""

    def boom(messages, schema):
        raise AssertionError("repair completer must not be called: fish alone recovers this")

    monkeypatch.setattr(planner_coordinator, "action_repair_completer", lambda: boom)

    original = {"type": "structural_experiment", "flags": {"a": True}}
    claude = FakeProvider(
        "claude",
        [
            PlannerProviderResult(
                provider="claude", role="draft", ok=True, text=_action_text(original)
            )
        ],
    )
    noisy = (
        "```json:autopilot_critique\n"
        '{"decision": "reject", "confidence": 0.95, "issues": ["bad draft"]}\n'
        "}\n"
        "```"
    )
    codex = FakeProvider(
        "codex",
        [PlannerProviderResult(provider="codex", role="critique", ok=True, text=noisy)],
    )

    decision = planner_coordinator.plan_with_providers(
        "prompt",
        session_id=None,
        planner_state={},
        settings=PlannerSettings(mode="draft_critique", critique_policy="always"),
        provider_factory=_factory({"claude": claude, "codex": codex}),
    )

    assert decision.critique is not None
    assert decision.critique.decision == "reject"
    assert decision.critique.parse_status == "parsed"
    assert decision.critique.repair_calls == 0
    assert len(codex.calls) == 1
