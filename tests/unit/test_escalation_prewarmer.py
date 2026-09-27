"""Tests for architect escalation prewarming stack-prior integration."""

from __future__ import annotations

import asyncio

from src.services import escalation_prewarmer


def _write_stack_priors(path, *, port: int = 9103, display_name: str = "TestArchitect") -> None:
    path.write_text(
        f"""
roles:
  architect_general:
    role: architect_general
    deployment_status: live_stack
    display_name: {display_name}
    model_id: test-architect-model
    serving:
      endpoint: http://localhost:{port}
  old_architect:
    role: old_architect
    deployment_status: benchmark_only
    display_name: OldArchitect
    model_id: old-architect
    serving:
      endpoint: http://localhost:9999
  frontdoor:
    role: frontdoor
    deployment_status: live_stack
    display_name: Frontdoor
    model_id: frontdoor-model
    serving:
      endpoint: http://localhost:9070
""".lstrip(),
        encoding="utf-8",
    )


def test_architect_ports_derive_from_live_stack_priors(tmp_path):
    priors = tmp_path / "stack_priors.yaml"
    _write_stack_priors(priors, port=9103)

    assert escalation_prewarmer.architect_ports_from_stack_priors(priors) == {
        "architect_general": 9103,
    }
    assert escalation_prewarmer.architect_port_for_role("architect_general", priors) == 9103


def test_architect_model_hints_derive_from_stack_priors(tmp_path):
    priors = tmp_path / "stack_priors.yaml"
    _write_stack_priors(priors, port=9103, display_name="QwenTest-Architect")

    assert escalation_prewarmer.architect_port_model_hints_from_stack_priors(priors) == {
        9103: "QwenTest-Architect",
    }
    assert escalation_prewarmer.architect_model_hint_for_port(9103, priors) == "QwenTest-Architect"


def test_architect_helpers_keep_degraded_fallback_when_priors_missing(tmp_path):
    missing = tmp_path / "missing_stack_priors.yaml"

    assert escalation_prewarmer.architect_ports_from_stack_priors(missing) == {}
    # 2026-09-27 ARCHITECT SWAP: architect_general = Flash-Next CPU :8074,
    # architect_critic = the MI210 27B :8083.
    assert escalation_prewarmer.architect_port_for_role("architect_general", missing) == 8074
    assert escalation_prewarmer.architect_port_for_role("architect_critic", missing) == 8083
    assert escalation_prewarmer.architect_model_hint_for_port(8074, missing) == "Qwen3.8-Flash-Next-UD-IQ4_XS"
    assert escalation_prewarmer.architect_model_hint_for_port(8083, missing) == "Qwen3.8-27B-Q8_0"
    assert escalation_prewarmer.ARCHITECT_PORTS == {
        "architect_general": 8074,
        "architect_critic": 8083,
    }


def test_prewarm_if_complex_uses_stack_prior_architect_port(tmp_path, monkeypatch):
    priors = tmp_path / "stack_priors.yaml"
    _write_stack_priors(priors, port=9103)
    prewarmer = escalation_prewarmer.EscalationPrewarmer(
        timeout=0.1,
        stack_priors_path=priors,
    )
    sent: list[tuple[int, str]] = []

    async def fake_check(port: int) -> bool:
        assert port == 9103
        return True

    async def fake_send(port: int, objective: str) -> bool:
        sent.append((port, objective))
        return True

    monkeypatch.setattr(prewarmer, "_check_slot_available", fake_check)
    monkeypatch.setattr(prewarmer, "_send_prewarm", fake_send)

    assert asyncio.run(prewarmer.prewarm_if_complex("build the thing", "COMPLEX")) is True
    assert sent == [(9103, "build the thing")]

