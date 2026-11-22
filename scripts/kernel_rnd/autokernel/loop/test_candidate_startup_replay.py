"""AKU-09g4: a restarted controller rebuilds the trusted candidate cache at startup."""
from __future__ import annotations

import pytest

from . import campaign_control as control
from . import candidate_startup_replay as startup
from .test_campaign_control import _resolved
from .test_candidate_manifest import _base, _next, _state
from .test_candidate_transactions import FakeGitBackend, _manager


def _verified(controller):  # the exact predicate the v4 candidate projection uses
    latest = next(e for e in reversed(controller._candidate_entries)
                  if e.payload["phase"] == "COMMITTED")
    return latest.seq <= getattr(controller._candidate_projection_cache, "last_seq", -1)


def test_restart_reports_replay_verified_from_startup_rebuilt_cache(tmp_path):
    backend = FakeGitBackend()
    controller, manager = _manager(tmp_path, backend=backend)
    base = _base()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        manager.integrate(request_id="integrate-1", previous=base, candidate=_next(base, 0))
    finally:
        controller.close()
    with control.CampaignController(_resolved(), tmp_path / "service") as restarted:
        assert restarted._candidate_projection_cache is not None  # no inspect() call
        assert restarted._candidate_projection_position == len(restarted._candidate_entries) == 6
        assert _verified(restarted) is True
        assert startup.rebuild_cache(restarted) is False  # idempotent: already built


def test_fresh_controller_has_nothing_to_rebuild(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as fresh:
        assert fresh._candidate_projection_cache is None
        assert startup.rebuild_cache(fresh) is False


def test_intent_without_completion_infers_no_committed_state(tmp_path):
    def fault(point):
        if point == "after_intent":
            raise RuntimeError("crash after intent")
    controller, manager = _manager(tmp_path, fault_hook=fault)
    try:
        with pytest.raises(RuntimeError, match="after intent"):
            manager.initialize(request_id="init", state=_state(_base()), manifest=_base())
    finally:
        controller.close()
    with control.CampaignController(_resolved(), tmp_path / "service") as restarted:
        cache = restarted._candidate_projection_cache  # replayer records the pending intent
        assert cache.state is None and cache.pending["transaction_id"] == "init"
        assert not any(e.payload["phase"] == "COMMITTED" for e in restarted._candidate_entries)
