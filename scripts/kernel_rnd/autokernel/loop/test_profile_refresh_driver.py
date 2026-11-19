"""AKU-06i driver-level refresh wiring over real tiny subprocesses (no hardware)."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from .. import journal
from . import actor_preparation_state as state
from . import profile_preparation as pp, scheduling, standalone_inputs as si
from . import unified_driver as ud, worker_lifecycle as wl
from .test_profile_preparation_runtime import (
    SyntheticSelectedProfileProvider, _fixture, _start)
from .test_standalone_inputs import _verifier
from .test_campaign_control import _command

DEBT = "profile_refresh_unavailable:original_request_consumed:fresh_predeclared_request_required"


class DynamicProvider(SyntheticSelectedProfileProvider):
    """Test-only author of selected metadata for whichever (derived) request ran."""

    def close_held_receipt(self, *, request, **kw):
        original = ud.ProfilePreparationRequest.from_dict(json.loads(request.argv[1]))
        self.requests[request.request_id] = original
        return super().close_held_receipt(request=request, **kw)


def _registry(tmp_path, materialized, target, binding, cap):
    provider = DynamicProvider(tmp_path / "containers", materialized.inputs.profile_requests)
    installed = pp.InstalledProfilePreparationBinding({target: binding},
                                                      refresh_attempt_cap=cap)
    return si.ProviderRegistry({
        "fixture-lifecycle": si.ProviderBinding(lifecycle_provider=provider),
        "fixture-readiness": si.ProviderBinding(readiness_check=lambda: (True, None))},
        evidence_verifiers={"fixture-evidence": _verifier()},
        profile_bindings={"tiny-profile": installed})


def _setup(tmp_path, cap, **fixture_kw):
    materialized, _, target, binding, counter, _ = _fixture(
        tmp_path, selected_provider=True, **fixture_kw)
    return materialized, _registry(tmp_path, materialized, target, binding, cap), target, counter


def _events(controller, event, schema=None):
    rows = journal.Journal(str(controller.store / "journal"),
                           campaign_id=controller.resolved.campaign_id).read_all()
    return [r.payload for r in rows if r.kind == journal.KIND_ACTOR_PREPARATION
            and r.payload["event"] == event
            and (schema is None or r.payload["schema"] == schema)]


def _expire(runtime, controller, target):
    row = controller.current_verified_profile_result(target)["profile_event"]
    runtime._clock = lambda: row["valid_until"]
    return row


def test_refresh_derived_with_real_settlement_digest_and_attribution(tmp_path):
    materialized, registry, target, counter = _setup(tmp_path, cap=3)
    controller, runtime = _start(materialized, registry)
    try:
        assert runtime.tick().status == "settled"
        old = ud._thaw(controller.current_verified_profile_result(target))
        row1 = _expire(runtime, controller, target)
        second = runtime.tick()
        assert second.status == "settled", second.reason
        assert DEBT not in second.reason
        new = ud._thaw(controller.current_verified_profile_result(target))
        row2 = new["profile_event"]
        assert row2["schema"] == state.PROFILE_V2_SCHEMA and row2["generation"] == 2
        assert row2["predecessor_profile_digest"] == row1["target_profile_digest"]
        assert row2["predecessor_request_digest"] == row1["profile_request_digest"]
        # the REAL settlement of the predecessor, not a placeholder
        assert row2["predecessor_settlement_digest"] == state.digest(old["settlement"])
        assert row2["predecessor_settlement_digest"] != "9" * 64
        # attribution: the successor request/transition are not the predecessor's
        assert row2["profile_request_digest"] != row1["profile_request_digest"]
        assert row2["transition_id"] != row1["transition_id"]
        assert new["settlement"]["transition_id"] == row2["transition_id"]
        assert pp.ProfilePreparationExecutionReceipt.from_dict(
            second.execution_receipt).body["profile_reference"]["profile_request_digest"] \
            == row2["profile_request_digest"]
        # the older profile can never be the current attempt's publication
        work = runtime.driver.materialize_profile  # noqa: F841 (typed path exists)
        assert pp._current_publication(
            old, SimpleNamespace(catalog_id=row2["catalog_id"], transition_id=row2["transition_id"],
                                 profile_request=ud.ProfilePreparationRequest.from_dict(
                                     row2["profile_request"]),
                                 stage_plan_digest=row2["stage_plan_digest"]),
            None, 0.0, "x", 0.0) is None
        assert counter.read_text() == "xx"
        assert len(_events(controller, "PROFILE_REQUEST")) == 1
        # only one receipt is current for the target after the successor
        assert controller.current_actor_profile(target).target_profile_digest \
            == row2["target_profile_digest"]
    finally:
        runtime.close()
        controller.close()


def test_failed_successor_attempt_keeps_older_profile_and_open_request(tmp_path):
    materialized, registry, target, counter = _setup(tmp_path, cap=3, fail_from=1)
    controller, runtime = _start(materialized, registry)
    try:
        assert runtime.tick().status == "settled"
        row1 = _expire(runtime, controller, target)
        failed = runtime.tick()
        assert failed.execution_receipt["disposition"] == "failed"
        assert failed.execution_receipt["profile_reference"] is None
        projection = controller.actor_preparation_projection()
        assert projection.profiles[target]["target_profile_digest"] == row1["target_profile_digest"]
        assert projection.profile_generations[target] == 1
        opened = projection.profile_requests[target]
        assert opened["generation"] == 2
        # the failed attempt is never attributed to the older profile's publication
        assert controller.current_verified_profile_result(target)["profile_event"] == row1
        assert len(_events(controller, "PROFILE_REQUEST")) == 1
        assert counter.read_text() == "xx"
    finally:
        runtime.close()
        controller.close()


def test_expiry_then_restart_reuses_the_open_request(tmp_path):
    materialized, registry, target, counter = _setup(tmp_path, cap=3)
    config = materialized.manifest.driver_config
    args = SimpleNamespace(store=config.store_path,
                           config_generation=config.config_generation, snapshot_version=3)
    factory = si.runtime_factory(materialized, registry)
    controller, runtime = factory(materialized.resolved, args)
    controller.apply_command(_command(materialized.resolved, "resume", "resume", 0))
    assert runtime.recover().status == "recovered"
    try:
        assert runtime.tick().status == "settled"
        row1 = _expire(runtime, controller, target)
        runtime.driver.refresh_installed_profiles(runtime.profile_executor, now=row1["valid_until"])
        derived = runtime.driver.profile_requests[target].to_dict()
        assert derived["stage_proposal"]["proposal_id"].endswith(":g2")
        assert len(_events(controller, "PROFILE_REQUEST")) == 1
        # repeated refresh at the same boundary mints nothing new
        runtime.driver.refresh_installed_profiles(runtime.profile_executor, now=row1["valid_until"])
        assert len(_events(controller, "PROFILE_REQUEST")) == 1
    finally:
        runtime.close()
        controller.close()
    reopened, restarted = factory(materialized.resolved, args)
    try:
        assert restarted.recover().status == "recovered"
        assert reopened.actor_preparation_projection().profile_requests[target][
            "profile_request"] == derived
        restarted._clock = lambda: row1["valid_until"]
        restarted.driver.refresh_installed_profiles(
            restarted.profile_executor, now=row1["valid_until"])
        assert restarted.driver.profile_requests[target].to_dict() == derived
        assert len(_events(reopened, "PROFILE_REQUEST")) == 1  # reused, not re-derived
        result = restarted.tick()
        assert result.status == "settled" and DEBT not in result.reason
        assert reopened.actor_preparation_projection().profile_generations[target] == 2
        assert not reopened.actor_preparation_projection().profile_requests
    finally:
        restarted.close()
        reopened.close()


def test_exhausted_budget_keeps_original_debt_reason(tmp_path):
    materialized, registry, target, counter = _setup(tmp_path, cap=2)
    controller, runtime = _start(materialized, registry)
    try:
        assert runtime.tick().status == "settled"
        _expire(runtime, controller, target)
        assert runtime.tick().status == "settled"  # generation 2 == cap
        _expire(runtime, controller, target)
        waiting = runtime.tick()
        assert waiting.status == "waiting"
        assert DEBT in waiting.reason
        assert len(_events(controller, "PROFILE_REQUEST")) == 1  # none beyond the cap
        assert counter.read_text() == "xx"
    finally:
        runtime.close()
        controller.close()


def test_v1_behaviour_unchanged_without_opt_in(tmp_path):
    materialized, registry, target, counter = _setup(tmp_path, cap=None)
    controller, runtime = _start(materialized, registry)
    try:
        assert runtime.tick().status == "settled"
        _expire(runtime, controller, target)
        waiting = runtime.tick()
        assert waiting.status == "waiting" and DEBT in waiting.reason
        assert not _events(controller, "PROFILE_REQUEST")
        assert counter.read_text() == "x"
    finally:
        runtime.close()
        controller.close()


def test_refresh_cap_validation(tmp_path):
    materialized, _, target, binding, _, _ = _fixture(tmp_path, selected_provider=True)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(pp.ProfilePreparationRefused):
            pp.InstalledProfilePreparationBinding({target: binding}, refresh_attempt_cap=bad)


def test_refresh_is_fair_across_two_targets(tmp_path):
    cpu_root, gpu_root = tmp_path / "cpu", tmp_path / "gpu"
    cpu_root.mkdir()
    gpu_root.mkdir()
    cpu = _fixture(cpu_root)
    gpu = _fixture(gpu_root, backend="gpu")
    document = cpu[0].manifest.to_dict()
    resolved = cpu[0].resolved.to_dict()
    resolved["targets"] += gpu[0].resolved.to_dict()["targets"]
    resolved["resources"]["gpu_ids"] = ["ROCm0"]
    (cpu_root / "resolved.json").write_text(json.dumps(resolved))
    for field in ("runtime_anchors", "profile_requests"):
        document["driver_config"][field].update(gpu[0].manifest.to_dict()["driver_config"][field])
    for field in ("runtime_dimensions", "experiment_plans", "execution_inputs"):
        document["driver_config"][field] = {}
    document["manifest_digest"] = si._digest({key: value for key, value in document.items()
                                             if key != "manifest_digest"})
    materialized = si.materialize(si.StartupManifest.from_dict(document))
    installed = pp.InstalledProfilePreparationBinding({cpu[2]: cpu[3], gpu[2]: gpu[3]},
                                                      refresh_attempt_cap=3)
    provider = DynamicProvider(cpu_root / "containers", materialized.inputs.profile_requests)
    registry = si.ProviderRegistry({
        "fixture-lifecycle": si.ProviderBinding(lifecycle_provider=provider),
        "fixture-readiness": si.ProviderBinding(readiness_check=lambda: (True, None))},
        evidence_verifiers={"fixture-evidence": _verifier()},
        profile_bindings={"tiny-profile": installed})
    controller, runtime = _start(materialized, registry)
    try:
        first = runtime.tick()
        first_target = first.execution_receipt["profile_reference"]["target_revision_digest"]
        first_row = controller.current_verified_profile_result(first_target)["profile_event"]
        runtime._clock = lambda: first_row["valid_until"] - 1.0  # first stays valid
        second = runtime.tick()
        second_target = second.execution_receipt["profile_reference"]["target_revision_digest"]
        assert {first_target, second_target} == {cpu[2], gpu[2]}
        rows = [controller.current_verified_profile_result(t)["profile_event"]
                for t in (cpu[2], gpu[2])]
        runtime._clock = lambda: max(r["valid_until"] for r in rows)  # both expired
        served = []
        for _ in range(2):
            result = runtime.tick()
            assert result.status == "settled", result.reason
            served.append(result.execution_receipt["profile_reference"]["target_revision_digest"])
        assert set(served) == {cpu[2], gpu[2]}  # neither starved nor served twice
        projection = controller.actor_preparation_projection()
        assert projection.profile_generations == {cpu[2]: 2, gpu[2]: 2}
        assert not projection.profile_requests
        assert len(_events(controller, "PROFILE_REQUEST")) == 2
    finally:
        runtime.close()
        controller.close()
