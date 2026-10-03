"""Tests for autopilot config application contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.autopilot import config_applicator as applicator
from scripts.autopilot import kv_compress


def test_try_cheap_first_q_threshold_maps_to_live_env_alias() -> None:
    env_changes = applicator.EnvRestartApplicator(restart=False).env_changes_for(
        {"chat.try_cheap_first_q_threshold": 0.78}
    )

    assert env_changes == {"ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_Q_THRESHOLD": "0.78"}


def test_legacy_try_cheap_first_quality_threshold_maps_to_live_env_alias() -> None:
    env_changes = applicator.EnvRestartApplicator(restart=False).env_changes_for(
        {"chat.try_cheap_first_quality_threshold": 0.78}
    )

    assert env_changes == {"ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_Q_THRESHOLD": "0.78"}


def test_approved_chat_threshold_params_map_to_env() -> None:
    env_changes = applicator.EnvRestartApplicator(restart=False).env_changes_for(
        {
            "chat.long_context_threshold_chars": 64000,
            "chat.summarization_threshold_tokens": 24000,
            "chat.review_skip_q_threshold": 0.72,
        }
    )

    assert env_changes == {
        "ORCHESTRATOR_CHAT_LONG_CONTEXT_THRESHOLD_CHARS": "64000",
        "ORCHESTRATOR_CHAT_SUMMARIZATION_THRESHOLD_TOKENS": "24000",
        "ORCHESTRATOR_CHAT_REVIEW_SKIP_Q_THRESHOLD": "0.72",
    }


def test_removed_review_low_threshold_maps_to_no_env() -> None:
    """RI-18c removed chat.review_low_q_threshold; it is no longer a tunable surface."""
    assert "review_low_q_threshold" not in applicator.ENV_PARAMS["chat"]
    env_changes = applicator.EnvRestartApplicator(restart=False).env_changes_for(
        {"chat.review_low_q_threshold": 0.55}
    )
    assert env_changes == {}


def test_review_skip_threshold_is_staged() -> None:
    result = applicator.EnvRestartApplicator(restart=False).apply(
        {"chat.review_skip_q_threshold": 0.72}
    )

    assert result.status == "staged"
    assert result.payload["env_changes"] == {
        "ORCHESTRATOR_CHAT_REVIEW_SKIP_Q_THRESHOLD": "0.72"
    }


def test_restart_api_with_env_uses_stack_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        applicator,
        "health_check",
        lambda _url: applicator.HealthCheckResult(ok=True),
    )

    result = applicator.restart_api(
        env_overrides={"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.05"},
        url="http://testserver",
    )

    assert result["status"] == "ok"
    assert result["method"] == "stack_reload"
    assert calls
    assert calls[0]["cmd"][-2:] == ["reload", "orchestrator"]
    assert calls[0]["cwd"] == str(ROOT)
    assert calls[0]["env"]["ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI"] == "0.05"


def test_restart_role_success_uses_stack_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "frontdoor",
        env_overrides={"ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS": "768"},
    )

    assert result["status"] == "ok"
    assert result["method"] == "stack_reload"
    assert result["role"] == "frontdoor"
    assert calls[0]["cmd"][-2:] == ["reload", "frontdoor"]
    assert calls[0]["cwd"] == str(ROOT)
    assert calls[0]["timeout"] == 180
    assert calls[0]["env"]["ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS"] == "768"


def test_stack_reload_python_prefers_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override = tmp_path / "reload-python"
    override.write_text("#!/bin/sh\n", encoding="utf-8")
    override.chmod(0o755)
    monkeypatch.setenv("AUTOPILOT_STACK_RELOAD_PYTHON", str(override))
    monkeypatch.setattr(applicator.sys, "_base_executable", "/missing/base/python")
    monkeypatch.setattr(applicator.sys, "executable", "/missing/venv/python")
    monkeypatch.setattr(applicator, "_stack_reload_python_usable", lambda path: path == override)

    assert applicator._stack_reload_python() == str(override)


def test_stack_reload_python_preserves_live_venv_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.chmod(0o755)
    monkeypatch.delenv("AUTOPILOT_STACK_RELOAD_PYTHON", raising=False)
    monkeypatch.setattr(applicator, "ORCH_ROOT", tmp_path)
    monkeypatch.setattr(applicator.sys, "_base_executable", str(tmp_path / "base-python"))
    monkeypatch.setattr(applicator.sys, "executable", str(tmp_path / "missing-venv-python"))
    monkeypatch.setattr(applicator, "_stack_reload_python_usable", lambda path: path == venv_python)

    assert applicator._stack_reload_python() == str(venv_python)


def test_stack_reload_python_survives_stale_sys_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path / "base-python"
    base.write_text("#!/bin/sh\n", encoding="utf-8")
    base.chmod(0o755)
    monkeypatch.delenv("AUTOPILOT_STACK_RELOAD_PYTHON", raising=False)
    monkeypatch.setattr(applicator, "ORCH_ROOT", tmp_path / "missing-orch-root")
    monkeypatch.setattr(applicator.sys, "_base_executable", str(base))
    monkeypatch.setattr(applicator.sys, "executable", str(tmp_path / "missing-venv-python"))
    monkeypatch.setattr(applicator, "_stack_reload_python_usable", lambda path: path == base.resolve())

    assert applicator._stack_reload_python() == str(base.resolve())


def test_stack_reload_python_skips_broken_venv_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    broken_venv = tmp_path / ".venv" / "bin" / "python"
    base = tmp_path / "base-python"
    broken_venv.parent.mkdir(parents=True)
    broken_venv.symlink_to(tmp_path / "missing-python")
    base.write_text("#!/bin/sh\n", encoding="utf-8")
    base.chmod(0o755)
    monkeypatch.delenv("AUTOPILOT_STACK_RELOAD_PYTHON", raising=False)
    monkeypatch.setattr(applicator, "ORCH_ROOT", tmp_path)
    monkeypatch.setattr(applicator.sys, "_base_executable", str(base))
    monkeypatch.setattr(applicator.sys, "executable", str(broken_venv))
    monkeypatch.setattr(applicator, "_stack_reload_python_usable", lambda path: path == base.resolve())

    assert applicator._stack_reload_python() == str(base.resolve())


def test_restart_role_success_journals_restart_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    monkeypatch.setattr(applicator.subprocess, "run", lambda *a, **kw: None)

    result = applicator.restart_role(
        "frontdoor",
        env_overrides={"ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS": "768"},
        journal=FakeJournal(),
        affected_roles=["frontdoor", "coder_escalation"],
        trial_id=27,
        boundary_reason="unit-test restart",
        actor="unit-test",
    )

    assert result["status"] == "ok"
    assert result["restart_boundary_event"]["type"] == "role_restart_boundary"
    assert events == [
        {
            "role": "frontdoor",
            "affected_roles": ["frontdoor", "coder_escalation"],
            "env_keys": ["ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS"],
            "registry_override_keys": [],
            "status": "ok",
            "rollback_status": "",
            "reason": "unit-test restart",
            "actor": "unit-test",
            "boundary_trial_id": 27,
            "command": "orchestrator_stack.py reload frontdoor",
        }
    ]


def test_restart_role_pause_dispatch_restores_after_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    state_path = tmp_path / "autopilot_state.json"
    state_path.write_text(json.dumps({"paused": False}) + "\n", encoding="utf-8")

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })
        assert json.loads(state_path.read_text(encoding="utf-8"))["paused"] is True

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "frontdoor",
        pause_dispatch=True,
        autopilot_state_path=state_path,
        dispatch_pause_grace_s=0,
    )

    assert result["status"] == "ok"
    assert len(calls) == 1
    assert result["dispatch_pause"]["paused_pre"] is False
    assert result["dispatch_pause"]["restore"] == {
        "status": "ok",
        "state_path": str(state_path),
        "restored": True,
    }
    assert json.loads(state_path.read_text(encoding="utf-8"))["paused"] is False


def test_restart_role_pause_dispatch_restores_after_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    state_path = tmp_path / "autopilot_state.json"
    state_path.write_text(json.dumps({"paused": False}) + "\n", encoding="utf-8")

    def fake_run(cmd, *, cwd, env, timeout, check):
        nonlocal calls
        calls += 1
        assert json.loads(state_path.read_text(encoding="utf-8"))["paused"] is True
        if calls == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "worker_general",
        env_overrides={"ORCHESTRATOR_WORKER_CALL_BUDGET_CAP": "12"},
        pause_dispatch=True,
        autopilot_state_path=state_path,
        dispatch_pause_grace_s=0,
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert calls == 2
    assert result["dispatch_pause"]["restore"]["status"] == "ok"
    assert json.loads(state_path.read_text(encoding="utf-8"))["paused"] is False


def test_restart_role_pause_dispatch_missing_state_fails_before_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role(
        "frontdoor",
        pause_dispatch=True,
        autopilot_state_path=tmp_path / "missing_state.json",
        dispatch_pause_grace_s=0,
    )

    assert result["status"] == "error"
    assert result["error"] == "failed to pause autopilot dispatch"
    assert result["dispatch_pause"]["status"] == "error"
    assert calls == []


def test_restart_role_strict_requires_explicit_affected_roles_before_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role(
        "frontdoor",
        require_explicit_affected_roles=True,
        smoke_check=lambda _role, _affected_roles: True,
    )

    assert result["status"] == "error"
    assert result["reason"] == "affected_roles_required"
    assert result["error"] == "affected_roles required for strict role restart"
    assert calls == []


def test_restart_role_strict_requires_smoke_check_before_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role(
        "frontdoor",
        affected_roles=["frontdoor", "coder_escalation"],
        require_smoke_check=True,
    )

    assert result["status"] == "error"
    assert result["reason"] == "smoke_check_required"
    assert result["error"] == "smoke_check required for strict role restart"
    assert result["affected_roles"] == ["frontdoor", "coder_escalation"]
    assert calls == []


def test_restart_role_strict_accepts_explicit_scope_and_smoke_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)

    result = applicator.restart_role(
        "frontdoor",
        affected_roles=["coder_escalation", "frontdoor"],
        require_explicit_affected_roles=True,
        require_smoke_check=True,
        smoke_check=lambda _role, _affected_roles: True,
    )

    assert result["status"] == "ok"
    assert result["affected_roles"] == ["frontdoor", "coder_escalation"]
    assert result["smoke_check"]["status"] == "ok"
    assert calls[0]["cmd"][-2:] == ["reload", "frontdoor"]


def test_restart_role_resolves_stack_affinity_from_priors(
    tmp_path: Path,
) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        """
stack_priors_version: 4
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      launch:
        entries:
          - port: 8070
            primary_role: frontdoor
  coder_escalation:
    deployment_status: live_stack
    serving:
      launch:
        entries:
          - port: 8070
            primary_role: frontdoor
  worker_general:
    deployment_status: live_stack
    serving:
      launch:
        entries:
          - port: 8072
            primary_role: worker_general
""",
        encoding="utf-8",
    )

    assert applicator.resolve_restart_affected_roles(
        "coder_escalation",
        stack_priors_path=priors,
    ) == ["coder_escalation", "frontdoor"]


def test_restart_role_rolls_back_to_prior_env_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", "8")

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "worker_general",
        env_overrides={"ORCHESTRATOR_WORKER_CALL_BUDGET_CAP": "12"},
    )

    assert result["status"] == "error"
    assert result["role"] == "worker_general"
    assert result["rollback"] == {
        "attempted": True,
        "status": "ok",
        "env_keys": ["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"],
    }
    assert calls[0]["env"]["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"] == "12"
    assert calls[1]["env"]["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"] == "8"


def test_restart_role_rollback_journals_restart_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    monkeypatch.setenv("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", "8")

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    calls = 0

    def fake_run(cmd, *, cwd, env, timeout, check):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "worker_general",
        env_overrides={"ORCHESTRATOR_WORKER_CALL_BUDGET_CAP": "12"},
        journal=FakeJournal(),
        trial_id=28,
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert events == [
        {
            "role": "worker_general",
            "affected_roles": ["worker_general"],
            "env_keys": ["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"],
            "registry_override_keys": [],
            "status": "error",
            "rollback_status": "ok",
            "reason": "intentional role restart",
            "actor": "config_applicator.restart_role",
            "boundary_trial_id": 28,
            "command": "orchestrator_stack.py reload worker_general",
        }
    ]


def test_restart_role_rolls_back_by_unsetting_new_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.delenv("ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K", raising=False)

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "ingest_long_context",
        env_overrides={"ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K": "12"},
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert calls[0]["env"]["ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K"] == "12"
    assert "ORCHESTRATOR_MEMRL_RETRIEVAL_SEMANTIC_K" not in calls[1]["env"]


def test_restart_role_smoke_failure_rolls_back_without_success_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "check": check,
        })

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(
        applicator,
        "resolve_restart_affected_roles",
        lambda _role: ["frontdoor", "coder_escalation"],
    )

    result = applicator.restart_role(
        "frontdoor",
        env_overrides={"ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS": "768"},
        journal=FakeJournal(),
        smoke_check=lambda _role, _affected_roles: {
            "status": "error",
            "error": "role probe failed",
        },
    )

    assert result["status"] == "error"
    assert result["error"] == "role probe failed"
    assert result["rollback"] == {
        "attempted": True,
        "status": "ok",
        "env_keys": ["ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS"],
    }
    assert len(calls) == 2
    assert events == [
        {
            "role": "frontdoor",
            "affected_roles": ["frontdoor", "coder_escalation"],
            "env_keys": ["ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS"],
            "registry_override_keys": [],
            "status": "error",
            "rollback_status": "ok",
            "reason": "intentional role restart",
            "actor": "config_applicator.restart_role",
            "boundary_trial_id": None,
            "command": "orchestrator_stack.py reload frontdoor",
        }
    ]


# ---------------------------------------------------------------------------
# W3 health-gate coverage (capability-registry-and-promotion.md).
#
# The W3 row's acceptance reads "a deliberately failed health gate rolls back",
# and until 2026-08-12 nothing exercised it. Measured with
# `coverage run --include='*config_applicator.py' -m pytest tests/unit/test_config_applicator.py`:
# the whole `role == "orchestrator"` branch of `restart_role` (the ONLY place the
# API health gate is consulted) was missing — lines 2221-2223, 2228-2230, 2232 and
# 2246-2248 — because every existing restart_role test passes a MODEL role, which
# evaluates `if role == "orchestrator"` False and skips the gate entirely. The
# rollback tests that do exist induce a *reload* failure or a *smoke* failure, which
# are different failure sources reaching the same rollback block, so a health-gate
# regression would have been invisible.
#
# These tests induce the failure rather than assume it: `health_check` is replaced
# with one that returns `HealthCheckResult(ok=False, ...)`. `subprocess.run` is
# stubbed throughout, so no reload is ever executed against the live stack —
# reload ownership belongs to the session that owns inference
# (INC-20260728-reload-preemption), and these must stay pure unit tests.


def test_restart_role_health_gate_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead API after reload must roll back env and journal an error boundary."""
    events: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI", "0.02")

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({"cmd": cmd, "env": env})

    smoke_calls: list[str] = []

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])
    monkeypatch.setattr(
        applicator,
        "health_check",
        lambda _url: applicator.HealthCheckResult(
            ok=False,
            failure_reason="connection_refused",
            failure_detail="All connection attempts failed",
        ),
    )

    result = applicator.restart_role(
        "orchestrator",
        env_overrides={"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.05"},
        journal=FakeJournal(),
        smoke_check=lambda role, _affected_roles: smoke_calls.append(role),
        url="http://testserver",
    )

    assert result["status"] == "error"
    # The diagnostics from HealthCheckResult must survive onto the result, or the
    # caller cannot tell a dead API from a failed smoke completion.
    assert result["error"] == "connection_refused"
    assert result["detail"] == "All connection attempts failed"
    # The gate is ordered BEFORE the canned smoke completion: an unreachable API
    # must never be asked to serve one.
    assert smoke_calls == []
    assert "smoke_check" not in result
    assert result["rollback"] == {
        "attempted": True,
        "status": "ok",
        "env_keys": ["ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI"],
    }
    # Two reloads: the attempt, then the rollback carrying the PRIOR value.
    assert len(calls) == 2
    assert calls[0]["env"]["ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI"] == "0.05"
    assert calls[1]["env"]["ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI"] == "0.02"
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert events[0]["rollback_status"] == "ok"
    assert events[0]["role"] == "orchestrator"


def test_restart_role_health_ok_smoke_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy /health but a failing canned completion still rolls back."""
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI", "0.02")

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({"cmd": cmd, "env": env})

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])
    monkeypatch.setattr(
        applicator,
        "health_check",
        lambda _url: applicator.HealthCheckResult(ok=True),
    )

    result = applicator.restart_role(
        "orchestrator",
        env_overrides={"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.05"},
        smoke_check=lambda _role, _affected_roles: {
            "status": "error",
            "error": "canned completion returned no tokens",
        },
        url="http://testserver",
    )

    assert result["status"] == "error"
    assert result["error"] == "canned completion returned no tokens"
    # /health passing must NOT overwrite the smoke verdict — a transport-only probe
    # is green over a server that cannot generate.
    assert result["smoke_check"]["status"] == "error"
    assert result["rollback"] == {
        "attempted": True,
        "status": "ok",
        "env_keys": ["ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI"],
    }
    assert len(calls) == 2
    assert calls[1]["env"]["ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI"] == "0.02"


def test_restart_role_health_and_smoke_ok_journals_success_without_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both gates green: one reload, a success boundary, and no rollback."""
    events: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({"cmd": cmd, "env": env})

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])
    monkeypatch.setattr(
        applicator,
        "health_check",
        lambda _url: applicator.HealthCheckResult(ok=True),
    )

    result = applicator.restart_role(
        "orchestrator",
        env_overrides={"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.05"},
        journal=FakeJournal(),
        smoke_check=lambda _role, _affected_roles: True,
        url="http://testserver",
    )

    assert result["status"] == "ok"
    assert result["smoke_check"]["status"] == "ok"
    assert "rollback" not in result
    assert len(calls) == 1
    assert len(events) == 1
    assert events[0]["status"] == "ok"
    # `_attach_restart_boundary_event` stringifies, so "no rollback" is "" not None.
    assert events[0]["rollback_status"] == ""


def test_restart_role_rejects_empty_role_before_any_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty role must fail closed, not reload whatever `reload ''` resolves to."""
    calls: list[object] = []
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role("   ")

    assert result == {
        "status": "error",
        "error": "role is required",
        "method": "stack_reload",
    }
    assert calls == []


def test_restart_role_records_registry_restore_failure_and_still_rolls_back_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed registry restore is reported, and must not abort the env rollback."""
    events: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"server_mode": {"worker_general": {"compaction_profile": "default"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", "8")

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    def fake_run(cmd, *, cwd, env, timeout, check):
        calls.append({"cmd": cmd, "env": env})
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    def exploding_restore(**_kwargs):
        raise OSError("registry file vanished under us")

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])
    monkeypatch.setattr(applicator, "_restore_registry_overrides", exploding_restore)

    result = applicator.restart_role(
        "worker_general",
        env_overrides={"ORCHESTRATOR_WORKER_CALL_BUDGET_CAP": "12"},
        registry_overrides={"server_mode.worker_general.compaction_profile": "S8"},
        registry_path=registry_path,
        journal=FakeJournal(),
    )

    assert result["status"] == "error"
    assert result["rollback"]["registry"] == {
        "status": "error",
        "registry_path": str(registry_path),
        "error": "registry file vanished under us",
    }
    # The env rollback is the point: a registry-restore exception must not swallow it.
    assert result["rollback"]["attempted"] is True
    assert result["rollback"]["status"] == "ok"
    assert len(calls) == 2
    assert calls[1]["env"]["ORCHESTRATOR_WORKER_CALL_BUDGET_CAP"] == "8"
    assert len(events) == 1
    assert events[0]["status"] == "error"


def test_restart_role_applies_registry_overrides_and_journals_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[dict[str, object]] = []
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "server_mode": {
                    "frontdoor": {
                        "draft_max": 16,
                        "p_split": 0.0,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeJournal:
        def append_role_restart_boundary_event(self, **kwargs):
            events.append(kwargs)
            return {"type": "role_restart_boundary", **kwargs}

    monkeypatch.setattr(applicator.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "frontdoor",
        registry_overrides={
            "server_mode.frontdoor.draft_max": 24,
            "server_mode.frontdoor.p_split": 0.1,
        },
        registry_path=registry_path,
        journal=FakeJournal(),
    )

    assert result["status"] == "ok"
    assert result["registry_overrides"] == {
        "status": "ok",
        "registry_path": str(registry_path),
        "override_keys": [
            "server_mode.frontdoor.draft_max",
            "server_mode.frontdoor.p_split",
        ],
    }
    reloaded = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert reloaded["server_mode"]["frontdoor"]["draft_max"] == 24
    assert reloaded["server_mode"]["frontdoor"]["p_split"] == 0.1
    assert events[0]["registry_override_keys"] == [
        "server_mode.frontdoor.draft_max",
        "server_mode.frontdoor.p_split",
    ]


def test_restart_role_restores_registry_overrides_on_reload_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    registry_path = tmp_path / "model_registry.yaml"
    original_registry = {
        "server_mode": {
            "worker_general": {
                "compaction_profile": "default",
            }
        }
    }
    registry_path.write_text(
        yaml.safe_dump(original_registry, sort_keys=False),
        encoding="utf-8",
    )

    def fake_run(cmd, *, cwd, env, timeout, check):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "worker_general",
        registry_overrides={
            "server_mode.worker_general.compaction_profile": "S8",
        },
        registry_path=registry_path,
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert result["rollback"]["registry"] == {
        "status": "ok",
        "registry_path": str(registry_path),
        "restored_keys": ["server_mode.worker_general.compaction_profile"],
    }
    assert calls == 2
    assert yaml.safe_load(registry_path.read_text(encoding="utf-8")) == original_registry


def test_restart_role_restores_registry_overrides_on_smoke_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    registry_path = tmp_path / "model_registry.yaml"
    original_registry = {
        "server_mode": {
            "frontdoor": {
                "draft_max": 16,
            }
        }
    }
    registry_path.write_text(
        yaml.safe_dump(original_registry, sort_keys=False),
        encoding="utf-8",
    )

    def fake_run(cmd, *, cwd, env, timeout, check):
        nonlocal calls
        calls += 1
        if calls == 2:
            restored = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            assert restored == original_registry

    monkeypatch.setattr(applicator.subprocess, "run", fake_run)
    monkeypatch.setattr(applicator, "resolve_restart_affected_roles", lambda role: [role])

    result = applicator.restart_role(
        "frontdoor",
        registry_overrides={"server_mode.frontdoor.draft_max": 24},
        registry_path=registry_path,
        smoke_check=lambda _role, _affected_roles: False,
    )

    assert result["status"] == "error"
    assert result["rollback"]["status"] == "ok"
    assert result["rollback"]["registry"]["status"] == "ok"
    assert calls == 2
    assert yaml.safe_load(registry_path.read_text(encoding="utf-8")) == original_registry


def test_restart_role_rejects_registry_override_missing_parent_before_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    registry_path = tmp_path / "model_registry.yaml"
    registry_path.write_text(yaml.safe_dump({"server_mode": {}}), encoding="utf-8")
    monkeypatch.setattr(
        applicator.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = applicator.restart_role(
        "frontdoor",
        registry_overrides={"server_mode.frontdoor.draft_max": 24},
        registry_path=registry_path,
    )

    assert result["status"] == "error"
    assert result["error"].startswith("failed to apply registry_overrides:")
    assert result["registry_override_keys"] == ["server_mode.frontdoor.draft_max"]
    assert calls == []


def test_apply_params_marks_env_restart_failure() -> None:
    def fail_env_params(_params, restart=True, url=applicator.ORCHESTRATOR_URL):
        return {"status": "error", "error": "reload failed"}

    original = applicator.apply_env_params
    applicator.apply_env_params = fail_env_params
    try:
        result = applicator.apply_params({"think_harder.min_expected_roi": 0.05})
    finally:
        applicator.apply_env_params = original

    assert result["status"] == "error"
    assert result["errors"] == ["env_restart: reload failed"]


def test_apply_params_marks_unknown_params_as_error() -> None:
    result = applicator.apply_params({"not_a_surface.value": 1})

    assert result["status"] == "error"
    assert result["unknown_params"] == ["not_a_surface.value"]
    assert "unknown_params: not_a_surface.value" in result["errors"]


def test_apply_params_marks_partial_kv_failure() -> None:
    def partial_kv_failure(_params, roles=None):
        return {
            "per_role": {
                "frontdoor": {"success": True, "error": None},
                "worker_general": {"success": False, "error": "slot busy"},
            }
        }

    original = applicator.apply_kv_compact
    applicator.apply_kv_compact = partial_kv_failure
    try:
        result = applicator.apply_params({"kv.keep_ratio": 0.5})
    finally:
        applicator.apply_kv_compact = original

    assert result["status"] == "error"
    assert result["errors"] == ["kv_compact:worker_general: slot busy"]


def test_kv_compaction_applicator_defaults_to_physical_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, float]] = []

    def fake_ports(*, include_aliases: bool = False) -> dict[str, int]:
        if include_aliases:
            return {"frontdoor": 8070, "coder_escalation": 8070, "worker_general": 8072}
        return {"frontdoor": 8070, "worker_general": 8072}

    def fake_compress_slot(*, port: int, keep_ratio: float, **_kwargs):
        calls.append((port, keep_ratio))
        return kv_compress.CompressResult(success=True, port=port)

    monkeypatch.setattr(kv_compress, "production_ports", fake_ports)
    monkeypatch.setattr(kv_compress, "compress_slot", fake_compress_slot)

    result = applicator.KvCompactionApplicator().apply({"kv.keep_ratio": 0.4})

    assert result.status == "ok"
    assert set(result.payload["per_role"]) == {"frontdoor", "worker_general"}
    assert calls == [(8070, 0.4), (8072, 0.4)]


def test_kv_compaction_applicator_honors_explicit_alias_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_ports(*, include_aliases: bool = False) -> dict[str, int]:
        if include_aliases:
            return {"frontdoor": 8070, "coder_escalation": 8070}
        return {"frontdoor": 8070}

    def fake_compress_slot(*, port: int, **_kwargs):
        calls.append(port)
        return kv_compress.CompressResult(success=True, port=port)

    monkeypatch.setattr(kv_compress, "production_ports", fake_ports)
    monkeypatch.setattr(kv_compress, "compress_slot", fake_compress_slot)

    result = applicator.KvCompactionApplicator(roles=["coder_escalation"]).apply(
        {"kv.keep_ratio": 0.4}
    )

    assert result.status == "ok"
    assert set(result.payload["per_role"]) == {"coder_escalation"}
    assert calls == [8070]


# ----- 2026-07-16 kv_compaction 500 thrash (resume-precondition) --------------
# "Expected Attention compression failed" is llama-server's blanket 500 for a
# scoring PRECONDITION failure (empty slot / unsupported memory layout) — a
# per-role capability condition, not a trial failure. It must skip the role,
# and an all-skip apply must report no_changes so the trial journals as a
# benign no-change skip instead of a hard apply error.

_EA_500_BODY = (
    '{"error":{"code":500,"message":"Expected Attention compression failed",'
    '"type":"server_error"}}'
)


def test_kv_compaction_uncompactable_500_is_per_role_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trial-1422/1427 case: ingest_long_context 500s with the EA
    precondition error while other roles compact fine — the role is skipped
    (annotated) and the apply succeeds on the remaining roles."""
    def fake_ports(*, include_aliases: bool = False) -> dict[str, int]:
        return {"frontdoor": 8070, "ingest_long_context": 8074}

    def fake_compress_slot(*, port: int, **_kwargs):
        if port == 8074:
            return kv_compress.CompressResult(success=False, port=port, error=_EA_500_BODY)
        return kv_compress.CompressResult(success=True, port=port, n_evicted=100)

    monkeypatch.setattr(kv_compress, "production_ports", fake_ports)
    monkeypatch.setattr(kv_compress, "compress_slot", fake_compress_slot)

    result = applicator.KvCompactionApplicator().apply({"kv.n_future": 256})

    assert result.status == "ok"
    assert result.errors == []
    assert result.payload["per_role"]["frontdoor"]["success"] is True
    skipped = result.payload["per_role"]["ingest_long_context"]
    assert skipped["status"] == "skipped"
    assert "not compactable" in skipped["reason"]
    assert "Expected Attention compression failed" in skipped["detail"]


def test_kv_compaction_all_roles_uncompactable_is_no_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If NO role was actually compacted, nothing changed live — report
    no_changes so actions._numeric_apply_no_changes journals a benign skip
    rather than evaluating a baseline masquerading as the trial params."""
    def fake_ports(*, include_aliases: bool = False) -> dict[str, int]:
        return {"ingest_long_context": 8074}

    def fake_compress_slot(*, port: int, **_kwargs):
        return kv_compress.CompressResult(success=False, port=port, error=_EA_500_BODY)

    monkeypatch.setattr(kv_compress, "production_ports", fake_ports)
    monkeypatch.setattr(kv_compress, "compress_slot", fake_compress_slot)

    result = applicator.KvCompactionApplicator().apply({"kv.keep_ratio": 0.5})

    assert result.status == "no_changes"
    assert result.errors == []
    assert not result.failed


def test_kv_compaction_other_failures_still_hard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the EA precondition 500 is a skip — any other failure (timeout,
    connection refused, different 500) remains a hard apply error."""
    def fake_ports(*, include_aliases: bool = False) -> dict[str, int]:
        return {"frontdoor": 8070}

    def fake_compress_slot(*, port: int, **_kwargs):
        return kv_compress.CompressResult(success=False, port=port, error="Connection refused")

    monkeypatch.setattr(kv_compress, "production_ports", fake_ports)
    monkeypatch.setattr(kv_compress, "compress_slot", fake_compress_slot)

    result = applicator.KvCompactionApplicator().apply({"kv.keep_ratio": 0.5})

    assert result.status == "error"
    assert result.errors == ["frontdoor: Connection refused"]


# ----- no-op API restart guard (env already live → skip the restart) ---------


def test_env_restart_skips_noop_when_live_env_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reverting to a baseline the API already runs must not bounce the API —
    env-carrying trials restarted it twice (apply + revert) even when the
    boundary revert was a no-op."""
    restarts: list[dict] = []
    monkeypatch.setattr(
        applicator, "restart_api",
        lambda env_overrides=None, url="": restarts.append(env_overrides) or {"status": "ok"},
    )
    monkeypatch.setattr(
        applicator, "_live_api_env",
        lambda keys: {"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.05"},
    )

    result = applicator.apply_env_params({"think_harder.min_expected_roi": 0.05})

    assert result["status"] == "skipped_noop"
    assert result["api_restart"] == "skipped_noop"
    assert result["env_changes"] == {"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.05"}
    assert restarts == []


def test_env_restart_performs_when_live_env_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    restarts: list[dict] = []
    monkeypatch.setattr(
        applicator, "restart_api",
        lambda env_overrides=None, url="": restarts.append(env_overrides) or {"status": "ok"},
    )
    monkeypatch.setattr(
        applicator, "_live_api_env",
        lambda keys: {"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.02"},
    )

    result = applicator.apply_env_params({"think_harder.min_expected_roi": 0.05})

    assert result["status"] == "ok"
    assert result["api_restart"] == "performed"
    assert restarts == [{"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": "0.05"}]


def test_env_restart_performs_when_live_env_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown live state must fail safe toward restarting, never skipping."""
    restarts: list[dict] = []
    monkeypatch.setattr(
        applicator, "restart_api",
        lambda env_overrides=None, url="": restarts.append(env_overrides) or {"status": "ok"},
    )
    monkeypatch.setattr(applicator, "_live_api_env", lambda keys: None)

    result = applicator.apply_env_params({"think_harder.min_expected_roi": 0.05})

    assert result["status"] == "ok"
    assert result["api_restart"] == "performed"
    assert len(restarts) == 1


def test_env_restart_performs_when_key_missing_from_live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    restarts: list[dict] = []
    monkeypatch.setattr(
        applicator, "restart_api",
        lambda env_overrides=None, url="": restarts.append(env_overrides) or {"status": "ok"},
    )
    # Key absent from the live process env → live.get(k) is None → restart.
    monkeypatch.setattr(
        applicator, "_live_api_env",
        lambda keys: {"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": None},
    )

    result = applicator.apply_env_params({"think_harder.min_expected_roi": 0.05})

    assert result["api_restart"] == "performed"
    assert len(restarts) == 1


def test_live_api_env_reads_proc_environ(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeCompleted:
        stdout = "4242\n"

    monkeypatch.setattr(
        applicator.subprocess, "run", lambda *a, **kw: FakeCompleted(),
    )
    environ_file = tmp_path / "environ"
    environ_file.write_bytes(b"FOO=1\0ORCHESTRATOR_X=abc\0NOEQ\0")
    real_read_bytes = Path.read_bytes

    def fake_read_bytes(self):
        if str(self) == "/proc/4242/environ":
            return real_read_bytes(environ_file)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    out = applicator._live_api_env(["ORCHESTRATOR_X", "MISSING"])
    assert out == {"ORCHESTRATOR_X": "abc", "MISSING": None}


def test_live_api_env_none_when_no_process(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCompleted:
        stdout = ""

    monkeypatch.setattr(applicator.subprocess, "run", lambda *a, **kw: FakeCompleted())
    assert applicator._live_api_env(["ANY"]) is None


def test_capture_reversible_param_preimage_reads_attested_worker_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"pid": 4242, "flags": {"memrl": True}}

    environ_file = tmp_path / "environ"
    environ_file.write_bytes(
        b"ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI=0.025\0OTHER=x\0"
    )
    real_read_bytes = Path.read_bytes

    def fake_read_bytes(self):
        if str(self) == "/proc/4242/environ":
            return real_read_bytes(environ_file)
        return real_read_bytes(self)

    monkeypatch.setattr(applicator.httpx, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    result = applicator.capture_reversible_param_preimage(
        {"think_harder.min_expected_roi": 0.05, "memrl": False}
    )

    assert result["status"] == "ok"
    assert result["reversible"] is True
    assert result["hot_swap"] == {"memrl": True}
    assert result["env"] == {
        "ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": {
            "present": True,
            "value": "0.025",
        }
    }


def test_capture_preimage_rejects_nonreversible_surface() -> None:
    result = applicator.capture_reversible_param_preimage({"kv.keep_ratio": 0.5})

    assert result["status"] == "unsupported"
    assert result["reversible"] is False
    assert result["unsupported"] == {"kv_compact": ["kv.keep_ratio"]}


def test_restore_reversible_param_preimage_preserves_absent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preimage = {
        "schema_version": applicator.PARAM_PREIMAGE_SCHEMA,
        "status": "ok",
        "reversible": True,
        "params": ["think_harder.min_expected_roi"],
        "hot_swap": {},
        "env": {
            "ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI": {
                "present": False,
                "value": None,
            }
        },
    }
    reload_calls: list[dict] = []
    monkeypatch.setattr(
        applicator,
        "_reload_api_via_stack",
        lambda **kwargs: reload_calls.append(kwargs) or {"status": "ok"},
    )
    monkeypatch.setattr(
        applicator,
        "capture_reversible_param_preimage",
        lambda params, url="": {**preimage, "source_pid": 5252},
    )

    result = applicator.restore_reversible_param_preimage(preimage, url="http://test")

    assert result["status"] == "ok"
    assert reload_calls == [
        {
            "env_overrides": {},
            "env_unset": ["ORCHESTRATOR_THINK_HARDER_MIN_EXPECTED_ROI"],
            "url": "http://test",
        }
    ]
