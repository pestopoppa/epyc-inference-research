"""Installed owner caches are bounded diagnostics, not new readiness authority."""
from __future__ import annotations

import copy
import sys
import time

import pytest

from . import campaign_control as cc, runtime_aggregates as A, serving_preparation as prep
from . import standalone_runtime as sr
from .test_standalone_runtime import _runtime
from .test_profile_preparation_runtime import _fixture, _start
from .test_feed_runtime import opened_owner
from . import test_serving_preparation_execution as calibration_cases


def _fail(*_args, **_kwargs):
    raise OSError("deliberate diagnostic fault")


def _rows(controller):
    unified = controller.snapshot()["unified"]
    return {"evidence": unified["evidence"],
            **{kind: unified["actors"][kind] for kind in ("actor", "profile", "calibration")}}


@pytest.mark.parametrize("point", ["record_runtime_aggregates", "_observe_aggregates"])
def test_diagnostic_publication_failure_cannot_change_recovery_or_settlement(tmp_path, monkeypatch, point):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    try:
        target = controller if point == "record_runtime_aggregates" else runtime
        monkeypatch.setattr(target, point, _fail)
        assert runtime.recover().status == "recovered"
        calls, original = [], lifecycle.run_stage
        monkeypatch.setattr(lifecycle, "run_stage", lambda *a, **k: (calls.append(1) or original(*a, **k)))
        assert runtime.tick().status == "settled"
        assert calls == [1] and len(engine.export_state().receipts) == 1
        assert runtime._uncertain is None and runtime._pending_outcome is None
        assert _rows(controller)["evidence"]["observed_at"] is None
    finally:
        runtime.close()
        controller.close()


def test_future_diagnostic_report_cannot_poison_snapshot_or_settlement(tmp_path, monkeypatch):
    runtime, controller, _, engine = _runtime(tmp_path, monkeypatch)
    try:
        runtime.recover()
        before = _rows(controller)
        bad = copy.deepcopy(before)
        bad["actor"]["attempted_at"] = "2999-01-01T00:00:00Z"
        with pytest.raises(cc.ControlRefused, match="newer than its publication attempt"):
            controller.record_runtime_aggregates(runtime, bad)
        assert _rows(controller) == before
        monkeypatch.setattr(controller, "actor_preparation_observation", lambda: bad["actor"])
        assert runtime.tick().status == "settled"
        assert len(engine.export_state().receipts) == 1
        assert runtime._uncertain is None and _rows(controller) == before
    finally:
        runtime.close()
        controller.close()


def test_profile_capture_fault_retains_original_dated_reduction_and_debt(tmp_path, monkeypatch):
    materialized, registry, target, _, counter, _ = _fixture(tmp_path, selected_provider=True)
    controller, runtime = _start(materialized, registry)
    try:
        assert runtime.tick().status == "settled"
        owner = runtime.profile_executor
        runtime.driver.refresh_installed_profiles(owner, now=time.monotonic())
        before = owner.observation_snapshot()
        assert before["status"] == "available" and before["data"]["usable_count"] == 1
        assert before["data"]["items"][0]["settled"]
        assert before["data"]["items"][0]["consumed_request_debt"] is False
        with monkeypatch.context() as fault:
            fault.setattr(owner, "_profile_observation_row", _fail)
            runtime.driver.refresh_installed_profiles(owner, now=time.monotonic())
            assert target in runtime.driver.profiles  # owning reduction still succeeds
        after = owner.observation_snapshot()
        assert after["status"] == "unknown" and "diagnostic fault" in after["error"]
        assert after["data"] == before["data"] and after["observed_at"] == before["observed_at"]
        assert after["attempted_at"] >= before["attempted_at"]
        row = controller.current_verified_profile_result(target)["profile_event"]
        runtime.driver.refresh_installed_profiles(owner, now=row["valid_until"])
        expired = owner.observation_snapshot()
        assert expired["data"]["usable_count"] == 0 and expired["data"]["debt_count"] == 1
        assert expired["data"]["items"][0]["consumed_request_debt"] is True
        assert expired["data"]["items"][0]["remaining_seconds"] == 0
        assert counter.read_text() == "x"
        # Fault inside debt-cache copying must also retain the already dated data.
        exact_profiles = owner.planner_profiles(row["valid_until"])
        before = owner.observation_snapshot()
        with monkeypatch.context() as fault:
            fault.setattr(owner, "_record_profile_debt", _fail)
            assert owner.consumed_request_debt(runtime.driver.profile_requests, exact_profiles)
        after = owner.observation_snapshot()
        assert after["data"] == before["data"] and after["observed_at"] == before["observed_at"]
        assert after["error"] is not None
    finally:
        runtime.close()
        controller.close()


def test_unsettled_profile_never_enters_usable_cache(tmp_path):
    materialized, registry, target, _, counter, _ = _fixture(tmp_path)
    controller, runtime = _start(materialized, registry)
    try:
        with pytest.raises(sr.StandaloneRuntimeUncertain):
            runtime.tick()
        assert controller.current_verified_profile_result(target)["settlement"] is None
        usable = runtime.profile_executor.planner_profiles(time.monotonic())
        runtime.profile_executor.consumed_request_debt(runtime.driver.profile_requests, usable)
        row = runtime.profile_executor.observation_snapshot()
        assert row["data"]["usable_count"] == 0
        assert row["data"]["items"][0]["settled"] is False
        assert row["data"]["items"][0]["available_at_planning"] is False
        assert counter.read_text() == "x"
    finally:
        runtime.close()
        controller.close()


def test_calibration_diagnostic_fault_retains_original_empty_settled_pool(tmp_path, monkeypatch):
    from . import serving
    runtime, controller, _, engine = _runtime(tmp_path, monkeypatch)
    owner = None
    try:
        requests = calibration_cases.make_requests(tmp_path, runtime.driver, engine, serving._measure_once)
        owner = prep.InstalledServingPreparationOwner(controller=controller, requests=requests)
        owner.recover()
        before = owner.observation_snapshot()
        assert before["data"]["collected_count"] == 0
        with monkeypatch.context() as fault:
            fault.setattr(A, "observation", _fail)
            owner.refresh_settled()
        after = owner.observation_snapshot()
        assert after["status"] == "unknown" and "diagnostic fault" in after["error"]
        assert after["data"] == before["data"] and after["observed_at"] == before["observed_at"]
        assert owner.pending_requests() == requests
    finally:
        if owner is not None:
            owner.close()
        runtime.close()
        controller.close()


@pytest.mark.parametrize("contamination", [False, True])
def test_actual_calibration_caches_follow_settlement_not_orphan_artifacts(tmp_path, monkeypatch, contamination):
    # Observe code-object returns without replacing source-pinned owners. Only
    # the existing fixture provider/raw observation boundary is synthetic.
    records = []
    watched = {prep.InstalledServingPreparationOwner.recover.__code__: "recover",
               prep.InstalledServingPreparationOwner.accept.__code__: "accept"}
    def observe(frame, event, _value):
        if event == "return" and frame.f_code in watched:
            owner = frame.f_locals["self"]
            snapshot = owner.observation_snapshot()
            prepared = frame.f_locals.get("prepared")
            records.append((watched[frame.f_code], id(owner), A.plain(snapshot),
                            None if prepared is None else prepared.plan.digest))
    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        case = (calibration_cases.test_actual_observed_placement_failure_uses_only_predeclared_pair_retry
                if contamination else calibration_cases.test_actual_public_v3_child_collection_settlement_and_numeric_pool)
        case(tmp_path, monkeypatch)
    finally:
        sys.setprofile(previous)
    rows = [row for _, _, row, _ in records if row["data"] is not None]
    assert rows and all(row["data"]["qualification"] == "unavailable" and
                        row["data"]["ranking_authorized"] is False for row in rows)
    if contamination:
        assert any(row["data"]["contaminated_count"] == 1 and row["data"]["pending_count"] > 0 for row in rows)
    else:
        # Second accept is still an orphan until the actual settlement commits.
        accepted = [(plan, row) for event, _, row, plan in records if event == "accept"]
        assert [row["data"]["collected_count"] for _, row in accepted] == [0, 1, 1]
        assert accepted[1][0] == accepted[2][0] != accepted[0][0]  # original exact retry, not a new chunk
        assert len({owner for _, owner, row, _ in records if row["data"] and row["data"]["collected_count"] == 2}) >= 2


def test_feed_outage_close_and_diagnostic_fault_do_not_redate_projection(tmp_path, monkeypatch):
    owner = opened_owner(tmp_path)
    try:
        before = owner.observation_snapshot()
        with monkeypatch.context() as fault:
            fault.setattr(owner.current_feed(), "drain_once", lambda *_: (_ for _ in ()).throw(TimeoutError("busy")))
            with pytest.raises(Exception, match="busy"):
                owner.drain()
        after = owner.observation_snapshot()
        assert after["status"] == "unknown" and after["error"] == "busy"
        assert after["observed_at"] == before["observed_at"]
        assert after["data"]["projection_frontier"] == before["data"]["projection_frontier"]
        assert after["attempted_at"] >= before["attempted_at"]
        with monkeypatch.context() as fault:
            fault.setattr(A, "utc_now", _fail)
            owner.drain()  # diagnostic clock is never an admission prerequisite
        failed = owner.observation_snapshot()
        assert failed["status"] == "unknown" and failed["observed_at"] == before["observed_at"]
        assert failed["data"] == before["data"]
        owner.drain()
        before = owner.observation_snapshot()
        owner.close()
        closed = owner.observation_snapshot()
        assert closed["data"]["owner_state"] == "closed" and closed["status"] == "unknown"
        assert closed["observed_at"] == before["observed_at"]
    finally:
        owner.close()


@pytest.mark.parametrize("field,value", [("generated_at", []), ("generated_at", "invalid"),
    ("generated_at", None), ("unified", []), ("active_worker", [])])
def test_additive_producer_refuses_malformed_base_fields_as_control_refused(tmp_path, monkeypatch, field, value):
    runtime, controller, _, _ = _runtime(tmp_path, monkeypatch)
    try:
        runtime.recover()
        body = controller.snapshot()
        body[field] = value
        with pytest.raises(cc.ControlRefused):
            cc.validate_snapshot_v3(body)
    finally:
        runtime.close()
        controller.close()


def test_installed_codec_bounds_and_source_identity_exclude_cache_state(tmp_path, monkeypatch):
    runtime, controller, _, _ = _runtime(tmp_path, monkeypatch)
    try:
        identity = cc._loaded_producer_build_identity()
        runtime.recover()
        rows = _rows(controller)
        assert cc._loaded_producer_build_identity() == identity
        assert "installed_dependency:runtime_aggregates" in identity["included_symbols"]
        altered = copy.deepcopy(rows)
        altered["actor"]["data"]["items"] = [{}] * 17
        with pytest.raises(ValueError, match="row bound"):
            A.validate_bundle(altered)
        with monkeypatch.context() as fault:
            fault.setitem(A.ROW_LIMITS, "actor", 15)
            assert cc._loaded_producer_build_identity() != identity
        assert A.plain(A.freeze(A.source_identity())) == A.source_identity()
    finally:
        runtime.close()
        controller.close()


def test_actual_actor_finish_is_not_scheduler_settlement_and_replays_without_relaunch(tmp_path):
    import os
    from . import actor_lifecycle, actor_preparation, scheduling, unified_driver as ud, worker_lifecycle as wl
    from .test_source_build_execution import _profiled_build_runtime, _plan
    from .test_actor_preparation import _budgets
    work = tmp_path / "declared-work"
    work.mkdir()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    # Declared non-kernel build advice only: never invoke configure/build here.
    materialized, registry, _, actor_profiles = _profiled_build_runtime(runtime_root,
        source_revision="d" * 40, build_plan=_plan(work))
    controller, runtime = _start(materialized, registry)
    try:
        assert runtime.tick().status == "settled"
        profiles = runtime.profile_executor.planner_profiles(time.monotonic())
        original = runtime.driver
        driver = ud.UnifiedCampaignDriver(resolved_campaign=original.resolved, controller=controller,
            scheduler_engine=original.scheduler, profiles=profiles, evidence_index=original.evidence,
            runtime_anchors=original.runtime_anchors, runtime_dimensions=original.runtime_dimensions,
            experiment_plans=original.experiment_plans, profile_requests=original.profile_requests,
            actor_identities=original.actor_identities, native_artifact_sink_ref=original.sink_ref,
            execution_inputs=original.execution_inputs, executable_work_kinds={"actor_preparation"})
        selected = driver.materialize_actor(driver.tick(now=time.monotonic()))
        adapter = actor_lifecycle.ActorLifecycleAdapter(controller=controller, persistence=controller,
            config=actor_lifecycle.ActorLifecycleConfig(campaign_digest=controller.config_digest,
                cwd=runtime_root, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                max_stage_seconds=5., teardown_seconds=1., max_retained_output_bytes=4096),
            target_profile_owner=controller)
        consumer = actor_preparation.ActorPreparationConsumer(resolved_campaign=original.resolved,
            profiles=actor_profiles, budgets=_budgets(), capability=adapter, clock=time.monotonic,
            clock_domain=wl.monotonic_clock_domain(), max_output_bytes=4096)
        seen = []
        code = cc.CampaignController.reserve_actor_preparation.__code__
        previous = sys.getprofile()
        def observe(frame, event, value):
            if frame.f_code is code and event == "return" and value is not None:
                seen.append(A.plain(controller.actor_preparation_observation()))
        sys.setprofile(observe)
        try:
            result = consumer.prepare(selected.actor_request, stage_plan_digest=selected.stage_plan_digest)
        finally:
            sys.setprofile(previous)
        assert result.status == "proposed"
        assert seen and all(row["data"]["pending_count"] >= 1 for row in seen)
        runtime._observe_aggregates()
        final = _rows(controller)["actor"]
        assert final["data"]["pending_count"] == 0 and final["data"]["finished_count"] == 2
        assert final["data"]["executor_installed"] is False  # standalone actor activation is still absent
        assert all(item["phase"] == "finished_unsettled" and item["settlement_outcome"] is None
                   for item in final["data"]["items"])
        config, resolved, store = original.scheduler.config, original.resolved, controller.store
    finally:
        runtime.close()
        controller.close()
    engine = scheduling.SchedulerEngine(config, scheduling.initial_state(config, resolved.campaign_id))
    with cc.CampaignController(resolved, store, snapshot_version=3, scheduler_engine=engine,
                              readiness_check=lambda: (True, None)) as reopened:
        replay = reopened.actor_preparation_observation()
        assert [{k: v for k, v in item.items() if k != "clock_known"} for item in replay["data"]["items"]] == [
            {k: v for k, v in item.items() if k != "clock_known"} for item in final["data"]["items"]]
        assert all(not item["clock_known"] for item in replay["data"]["items"])  # no registered current clock
        assert replay["data"]["finished_count"] == final["data"]["finished_count"]
