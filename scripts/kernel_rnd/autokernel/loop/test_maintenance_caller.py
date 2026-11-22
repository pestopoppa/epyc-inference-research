"""AKU-10d2: the controller maintenance caller runs view -> bind -> execute."""
from __future__ import annotations

from dataclasses import replace

import pytest

from . import candidate_transactions as ct
from . import maintenance_caller as mcall
from . import maintenance_execution as me
from . import native_retention_catalog as catalog
from . import retention_consumer as rc
from .test_candidate_manifest import _base, _state
from .test_candidate_transactions import FakeGitBackend
from .test_maintenance_broker import InMemoryMaintenanceBroker, _prepared, root  # noqa: F401
from .test_native_retention_catalog import _controller, _preparations, _seed
from .test_retention_consumer import NOW, _policy, _view


def _setup(path):
    artifact_root = path / "artifacts"
    resolved, anchors, seed = _seed(artifact_root=artifact_root)
    store = path / "service"
    store.mkdir(mode=0o700)
    store.chmod(0o700)
    controller = _controller(resolved, store)
    controller.__enter__()
    controller.install_native_retention_catalog(
        seed, runtime_anchors=anchors, model_preparations=_preparations(seed),
        artifact_root=artifact_root)
    manager = ct.CandidateTransactions(controller, git_backend=FakeGitBackend())
    manager.initialize(request_id="init", state=_state(_base()), manifest=_base())
    frontier = controller.native_retention_catalog_capture()["frontier"]
    view, old = _view(path)
    node = view.nodes[-1]
    node = replace(node, expiry=replace(node.expiry, campaign_id=resolved.campaign_id))
    view = replace(view, snapshot_id=seed.seed_digest,
                   generation=max(1, int(frontier[2]) + 1), nodes=view.nodes[:-1] + (node,))
    return controller, manager, view, old


def test_caller_runs_view_bind_execute_end_to_end_with_injected_provider(root, monkeypatch):
    controller, manager, view, old = _setup(root)
    try:  # the fixture view carries one expirable node; the frontier is the live capture's
        monkeypatch.setattr(catalog, "build_native_view", lambda capture: _prepared(
            replace(view, generation=max(1, int(capture["frontier"][2]) + 1)), capture["frontier"]))
        broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
        run = mcall.run_maintenance(manager, _policy(root), provider=broker, limit=1, now=NOW)
        assert run.status == "executed" and run.result is not None
        assert not old.exists() and broker.violations == []
        assert controller._maintenance_state.phase == "COMPLETED"
        assert broker.calls.count("acquire") == 1 and broker.calls.count("settle") == 1
    finally:
        controller.close()


@pytest.mark.parametrize("provider", [None, me.MaintenanceHoldProvider(), me.BrokerHoldProvider(), object()])
def test_caller_refuses_without_a_real_provider_before_collecting(root, monkeypatch, provider):
    controller, manager, _prepared_view, old = _setup(root)
    try:
        monkeypatch.setattr(manager, "prepare_retention",
                            lambda *a, **k: pytest.fail("collected without a provider"))
        with pytest.raises(mcall.MaintenanceCallerRefused, match="unavailable"):
            mcall.run_maintenance(manager, _policy(root), provider=provider, now=NOW)
        assert old.exists() and not controller._maintenance_state.owned
    finally:
        controller.close()


def test_real_native_view_with_nothing_expirable_is_a_noop(tmp_path, monkeypatch):
    from .test_native_retention_catalog import _complete_controller
    _r, _a, _s, controller, manager = _complete_controller(tmp_path, monkeypatch)
    try:
        manager.initialize(request_id="init", state=_state(_base()), manifest=_base())
        run = mcall.run_maintenance(manager, _policy(tmp_path),
                                    provider=InMemoryMaintenanceBroker(), now=NOW)
        assert run.status == "nothing_to_expire" and run.result is None
        assert not controller._maintenance_state.owned
    finally:
        controller.close()
