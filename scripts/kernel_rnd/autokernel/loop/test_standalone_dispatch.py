"""AKU-07h4: standalone runtime dispatches actor and continuous-feed work via owners."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from . import evidence_feed, feed_runtime as F, standalone_dispatch as D
from . import standalone_runtime as sr, unified_driver
from .test_evidence_feed import _native_event, _root_repo
from .test_feed_runtime import binding, config
from .test_standalone_runtime import _reopen_runtime, _runtime
from .test_unified_driver import runtime_driver


class _Receipt:
    def __init__(self, label):
        self.label = label

    def to_dict(self):
        return {"fixture_receipt": self.label}


def _stub_selected(transition_id):
    """Labelled fixture: typed SelectedActorWork shell carrying only the transition."""
    stub = object.__new__(unified_driver.SelectedActorWork)
    object.__setattr__(stub, "transition_id", transition_id)
    return stub


def _actor_owner(calls):
    return D.InstalledActorDispatch(
        run=lambda selected: calls.append(("run", selected.transition_id)) or _Receipt("run"),
        recover=lambda selected: (calls.append(("recover", selected.transition_id))
                                  or _Receipt("recover")))


def _as_actor(monkeypatch, runtime):
    driver = runtime.driver
    monkeypatch.setattr(driver, "issued_work_kind", lambda outcome: D.ACTOR_KIND)
    monkeypatch.setattr(driver, "materialize_actor",
                        lambda outcome: _stub_selected(outcome.transition_id))
    monkeypatch.setattr(runtime.executor, "execute",
                        lambda *_: pytest.fail("actor work reached the runtime executor"))
    monkeypatch.setattr(runtime.executor, "recover_issued",
                        lambda *_: pytest.fail("actor recovery reached the runtime executor"))


def test_compose_installs_actor_owner_and_dispatches_actor_selection(tmp_path, monkeypatch):
    calls = []
    original = sr.StandaloneRuntimeInputs.__init__

    def with_actor(self, *args, **kwargs):
        original(self, *args, **kwargs)
        object.__setattr__(self, "actor_dispatch", _actor_owner(calls))
    monkeypatch.setattr(sr.StandaloneRuntimeInputs, "__init__", with_actor)
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    try:
        assert D.ACTOR_KIND in runtime.driver._executable_work_kinds
        assert D.ACTOR_KIND in D.installed_kinds(runtime)
        runtime.recover()
        _as_actor(monkeypatch, runtime)
        result = runtime.tick()
        assert result.status == "settled", result.reason
        assert result.execution_receipt == {"fixture_receipt": "run"}
        assert calls == [("run", result.driver_outcome["transition_id"])]
    finally:
        runtime.close()
        controller.close()


def test_actor_selection_without_installed_owner_is_refused(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    try:
        assert runtime.actor_executor is None
        assert D.ACTOR_KIND not in D.installed_kinds(runtime)
        with pytest.raises(D.ActorDispatchRefused, match="not installed"):
            D.dispatch(runtime, D.ACTOR_KIND, object())
        with pytest.raises(sr.StandaloneRuntimeRefused, match="installed concrete"):
            sr.StandaloneRuntime.compose(controller=controller, inputs=replace(
                _inputs_of(runtime), actor_dispatch=object()))
    finally:
        runtime.close()
        controller.close()


def _inputs_of(runtime):
    d = runtime.driver
    return sr.StandaloneRuntimeInputs(
        d.resolved, d.scheduler, d.profiles, d.evidence, d.runtime_anchors,
        d.runtime_dimensions, d.experiment_plans, d.profile_requests,
        d.actor_identities, d.execution_inputs, d.sink_ref)


def test_restart_recovers_restored_actor_intent_through_actor_owner_once(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    transition_id = runtime.driver.tick(now=1.0).transition_id
    reopened_runtime, reopened, _new, _replay = _reopen_runtime(
        runtime, controller, engine, monkeypatch)
    calls = []
    reopened_runtime.actor_executor = _actor_owner(calls)
    _as_actor(monkeypatch, reopened_runtime)
    try:
        recovered = reopened_runtime.recover()
        assert recovered.status == "settled", recovered.reason
        assert recovered.driver_outcome["transition_id"] == transition_id
        assert recovered.execution_receipt == {"fixture_receipt": "recover"}
        assert calls == [("recover", transition_id)]
    finally:
        reopened_runtime.close()
        reopened.close()


def test_actor_owner_refuses_non_current_selection():
    class Driver:
        def materialize_actor(self, outcome):
            raise unified_driver.DriverRefused("selected work is not actor preparation")
    owner = _actor_owner([])
    with pytest.raises(D.ActorDispatchRefused, match="not current"):
        owner.execute(Driver(), object())
    with pytest.raises(D.ActorDispatchRefused, match="run and recover"):
        D.InstalledActorDispatch(run=None, recover=lambda _: None)


def test_continuous_feed_drains_in_driver_tick_and_restart_resumes_without_reprojection(
        tmp_path):
    cfg = config(tmp_path)
    Path(cfg.corpus_root).mkdir()
    journal, _ = _native_event(Path(cfg.source_root), Path(cfg.corpus_root), _root_repo())
    owner = F.FeedRuntimeOwner(cfg, binding())
    try:
        driver, _, _, _, _ = runtime_driver()
        driver.feed_owner, driver.evidence = owner, owner.view
        assert driver.tick(now=1).status == "intent_recorded"
        first = dict(owner.last_snapshot)
        ledger = len(owner.current_feed().ledger.read_all())
        cursor = journal.cursor(cfg.reader_id).last_seq
    finally:
        owner.close()
    restarted = F.FeedRuntimeOwner(cfg, binding())  # process restart: new owner, same paths
    try:
        driver, _, _, _, _ = runtime_driver()
        driver.feed_owner, driver.evidence = restarted, restarted.view
        assert driver.tick(now=2).status == "intent_recorded"
        assert restarted.last_snapshot["projection_frontier"] == first["projection_frontier"]
        assert len(restarted.current_feed().ledger.read_all()) == ledger
        assert journal.cursor(cfg.reader_id).last_seq == cursor
        journal.append(evidence_feed.journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": "after-restart", "reason": "fixture"})
        restarted.drain()
        assert restarted.last_snapshot["source_frontier"] == first["source_frontier"] + 1
    finally:
        restarted.close()
