"""OAB-32: default per-call wall budgets, role x kind, with a fake that never finishes."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from autokernel.loop import actors, call_budget

NEVER = r"""
import os, sys, time
open(sys.argv[1], "w").write(str(os.getpid()))
time.sleep(120)
"""
SCHEMAS = {"planner": actors.HYPOTHESIS_SCHEMA, "author": actors.PATHS_SCHEMA,
           "critic": actors.REVIEW_SCHEMA}


class _Never(actors.Backend):
    marker: str = ""

    def argv(self, prompt, workspace, *, read_only=False, **_kw):
        return [sys.executable, "-c", NEVER, self.marker]


def _kill(marker: Path) -> None:
    if marker.exists() and marker.read_text().strip():
        try:
            os.kill(int(marker.read_text()), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass


@pytest.mark.parametrize("kind", call_budget.KINDS)
@pytest.mark.parametrize("role", call_budget.ROLES)
def test_never_finishing_backend_is_cut_at_the_default_budget_once(tmp_path, role, kind):
    ws = tmp_path / "lane"
    ws.mkdir()
    marker = tmp_path / "pid"
    backend = _Never(kind=kind, model="test/never", effort="high", binary=sys.executable)
    object.__setattr__(backend, "marker", str(marker))
    # The role's own class and entry point: planner -> AgentPlanner._run proposal call,
    # author -> AgentPlanner author call, critic -> AgentCritic._review.
    kw = actors._default_budget_kw(backend, role, 6, 0)
    assert kw == {"budget_s": 3.0}
    started = time.monotonic()
    try:
        with mock.patch.object(actors, "STOP_POLL_S", 0.05), \
                mock.patch.object(actors, "STOP_GRACE_S", 3.0):
            with pytest.raises(actors.ActorBudgetExhausted) as info:
                actors._run_agent("task", workspace=ws, timeout_s=6, backend=backend,
                                  schema=SCHEMAS[role], read_only=role == "critic", **kw)
        assert time.monotonic() - started < 6, "cut by the budget, before the hard timeout"
        assert "budget_exhausted" in str(info.value)
    finally:
        _kill(marker)


@pytest.mark.parametrize("kind", call_budget.KINDS)
def test_critic_class_applies_the_default_only_when_enabled(tmp_path, kind):
    backend = _Never(kind=kind, model="m", effort="high", binary=sys.executable)
    off = actors.AgentCritic(workspace=tmp_path, backend=backend, timeout_s=100)
    on = actors.AgentCritic(workspace=tmp_path, backend=backend, timeout_s=100,
                            default_budgets=True)
    assert off.default_budgets is False and on.default_budgets is True


@pytest.mark.parametrize("role", ("planner", "author"))
def test_planner_budget_kw_historical_off_default_on(tmp_path, role):
    backend = _Never(kind="claude", model="m", effort="high", binary=sys.executable)
    off = actors.AgentPlanner(workspace=tmp_path, backend=backend, timeout_s=100)
    on = actors.AgentPlanner(workspace=tmp_path, backend=backend, timeout_s=100,
                             default_budgets=True)
    assert off._budget_kw(role) == {}                       # historical: byte for byte
    assert on._budget_kw(role) == {"budget_s": 90.0}


def test_unconfigured_default_sits_under_the_timeout_and_configured_wins():
    assert call_budget.default_wall_s(1800) == 1620
    assert call_budget.default_wall_s(20) == 10 and call_budget.default_wall_s(0) == 0
    assert call_budget.effective_wall_s(None, 1800) == 1620
    assert call_budget.effective_wall_s(100, 1800) == 100          # never loosened or replaced
    assert call_budget.effective_wall_s(None, 0) is None


def test_off_switch_restores_the_historical_path():
    assert call_budget.effective_wall_s(None, 1800, env={call_budget.DISABLE_ENV: "off"}) is None
    assert call_budget.enabled({}) and not call_budget.enabled({call_budget.DISABLE_ENV: "0"})


@pytest.mark.parametrize("kind", sorted(call_budget.LOCAL_KINDS))
def test_local_backend_without_a_budget_refuses(kind):
    with pytest.raises(call_budget.UnboundedBackendRefused):
        call_budget.require_bounded(kind, "planner", None)
    call_budget.require_bounded("claude", "critic", None)        # cloud CLI: wall only at timeout
    call_budget.require_bounded(kind, "planner", 30)


def test_local_backend_with_no_timeout_refuses_to_run(tmp_path):
    backend = _Never(kind="opencode", model="m", effort="high", binary=sys.executable)
    planner = actors.AgentPlanner(workspace=tmp_path, backend=backend, timeout_s=0,
                                  default_budgets=True)
    with pytest.raises(call_budget.UnboundedBackendRefused):
        planner._budget_kw("planner")


def test_summary_reports_cap_hit_rate_per_role():
    from autokernel.loop import actor_metrics as am
    rows = [{"role": "planner", "failure_class": am.OUTPUT_CAPPED_EMPTY},
            {"role": "planner", "failure_class": None},
            {"role": "planner", "failure_class": None, "opencode": {"totals": {"output_capped_steps": 2}}},
            {"role": "planner", "failure_class": None}]
    out = am._summary(rows)
    assert out["output_capped_empty"] == 1 and out["output_capped_empty_rate"] == 0.25
    assert out["output_capped_steps_total"] == 2
    assert am._summary([])["output_capped_empty_rate"] is None
