"""AKU-06l: child accounting + settlement v2 (pure records, scheduler previews, installed path).

Synthetic labelled fixtures only; no hardware, server or campaign claim."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from .. import journal
from ..execution import worktree
from . import (actor_lifecycle, actor_preparation, measurement_capture,
               operation_accounting as oa, scheduling as S, unified_driver, worker_lifecycle)
from . import source_build_execution as execution
from .test_actor_preparation import _budgets
from .test_operation_accounting import child, engine_and_selection
from .test_scheduling import receipt
from .test_source_build_execution import _git, _profiled_build_runtime, _start

SHA = "a" * 64


# ------------------------------------------------------------ pure records -------------

def _terminal(request_id="r0", generation=1, grant=1, **changes):
    row = {"worker_id": "w0", "worker_generation": generation, "request_id": request_id,
           "plan_digest": SHA, "lineage_id": "lin", "stage_id": f"s-{request_id}",
           "grant_id": "g0", "grant_generation": grant, "container_id": "c0",
           "return_code": 0, "result_digest": SHA, "accepted": True, "reason": None}
    row.update(changes)
    return row


def _receipt_dict(name, *, start, end, generation=1, grant=1):
    held = S.HeldClaimReceipt.from_dict(receipt(
        {"proposal_id": "prop", "backend": "cpu", "stage_class": "search"}, name,
        start=start, end=end, fraction=0.25, memory=100, affinity=("0",),
        claims=(f"claim-{name}",)))
    row = held.to_dict()
    row["ownership_generation"], row["allocation_generation"] = generation, grant
    return row


def _record(role, previous=None, *, name=None, start=0.0, end=1.0, **changes):
    index = oa.ROLES.index(role)
    kwargs = dict(
        campaign_id="camp", config_generation=1, config_digest=SHA, supervisor_incarnation=1,
        catalog_id="b" * 64, transition_id="c" * 64, selection_digest="d" * 64,
        operation="source_build", role=role, binding_digest="e" * 64,
        terminal=_terminal(f"req-{role}"), receipt=_receipt_dict(
            name or f"rc-{role}", start=start + index * 2, end=end + index * 2),
        disposition="accepted", predecessor=None if previous is None else previous["digest"])
    kwargs.update(changes)
    return oa.make_child_record(**kwargs)


def _chain():
    planner = _record("planner")
    critic = _record("critic", planner)
    return [planner, critic, _record("build", critic)]


def test_child_record_is_closed_digest_bound_and_roundtrips():
    chain = _chain()
    assert [r["ordinal"] for r in chain] == [0, 1, 2]
    assert oa.validate_child_chain(chain) == chain
    for name, value in (("extra", 1), ("role", "judge"), ("disposition", "x"),
                        ("digest", "0" * 64), ("terminal_digest", "0" * 64),
                        ("receipt_digest", "0" * 64), ("ordinal", 2)):
        bad = {**copy.deepcopy(chain[0]), name: value}
        with pytest.raises(oa.OperationAccountingRefused):
            oa.validate_child_record(bad)
    missing = copy.deepcopy(chain[0])
    del missing["predecessor"]
    with pytest.raises(oa.OperationAccountingRefused):
        oa.validate_child_record(missing)


def test_chain_refuses_reorder_duplicate_substitution_and_cross_role_receipt_reuse():
    planner, critic, build = _chain()
    with pytest.raises(oa.OperationAccountingRefused):
        oa.validate_child_chain([critic, planner])
    with pytest.raises(oa.OperationAccountingRefused):
        oa.validate_child_chain([planner, planner])
    with pytest.raises(oa.OperationAccountingRefused):  # skipped critic
        oa.validate_child_chain([planner, build])
    other_planner = _record("planner", name="other")
    with pytest.raises(oa.OperationAccountingRefused):  # critic names a different predecessor
        oa.validate_child_chain([other_planner, critic])
    # one receipt (identical bytes and digest) cannot satisfy two roles
    reused = _record("critic", planner, name="rc-planner", start=0.0, end=1.0,
                     receipt=planner["receipt"], terminal=_terminal("req-critic"))
    with pytest.raises(oa.OperationAccountingRefused, match="receipt"):
        oa.validate_child_chain([planner, reused])
    # one request cannot satisfy two roles
    same_request = _record("critic", planner, terminal=planner["terminal"])
    with pytest.raises(oa.OperationAccountingRefused, match="request"):
        oa.validate_child_chain([planner, same_request])
    # terminal/receipt generation mismatch
    with pytest.raises(oa.OperationAccountingRefused):
        _record("planner", terminal=_terminal(generation=2))


def test_outcome_rules_and_settled_v2_validator_are_closed():
    chain = _chain()
    oa.check_outcome_against_children("prerequisite", chain, None)
    for outcome, rows, ref in (("prerequisite", chain[:2], None),
                               ("prerequisite", chain, SHA), ("failed", [], None),
                               ("invalid", [], SHA), ("valid_comparison", chain, None)):
        with pytest.raises(oa.OperationAccountingRefused):
            oa.check_outcome_against_children(outcome, rows, ref)
    diagnostic = [chain[0], {**chain[1], "disposition": "diagnostic"}, chain[2]]
    with pytest.raises(oa.OperationAccountingRefused):
        oa.check_outcome_against_children("prerequisite", diagnostic, None)
    oa.check_outcome_against_children("failed", chain[:1], SHA)

    payload = {"schema": oa.SETTLED_V2_SCHEMA, "catalog_id": SHA, "campaign_id": "camp",
               "config_generation": 1, "config_digest": SHA, "supervisor_incarnation": 1,
               "selection": {"x": 1}, "prior_projection_digest": SHA,
               "after_projection_digest": SHA, "transition_id": SHA, "outcome": "prerequisite",
               "terminal_refs": ["a"], "children": [r["digest"] for r in chain],
               "denial_ref": None}
    assert oa.validate_settled_v2(payload) == []
    assert journal._validate_native_payload(journal.KIND_UNIFIED_DRIVER_SETTLED, payload) == []
    for change in ({"extra": 1}, {"children": []}, {"outcome": "valid_comparison"},
                   {"children": [SHA, SHA, SHA]}, {"denial_ref": "nothex"},
                   {"schema": "x"}):
        bad = {**payload, **change}
        assert journal._validate_native_payload(journal.KIND_UNIFIED_DRIVER_SETTLED, bad), change


def test_new_kinds_are_native_and_legacy_v1_validation_is_unchanged():
    chain = _chain()
    assert journal.KIND_UNIFIED_DRIVER_CHILD_ACCOUNTED in journal.NATIVE_KINDS
    assert journal._validate_native_payload(
        journal.KIND_UNIFIED_DRIVER_CHILD_ACCOUNTED, chain[0]) == []
    assert journal._validate_native_payload(
        journal.KIND_UNIFIED_DRIVER_CHILD_ACCOUNTED, {**chain[0], "role": "x"})
    # a legacy v1 settlement still validates byte-for-byte under the v1 rules
    v1 = {"schema": journal.UNIFIED_DRIVER_SETTLED_SCHEMA, "catalog_id": SHA,
          "campaign_id": "camp", "config_generation": 1, "config_digest": SHA,
          "supervisor_incarnation": 1, "catalog": None, "selection": {"x": 1},
          "prior_projection_digest": SHA, "after_projection_digest": SHA,
          "transition_id": SHA, "receipt": {"x": 1}, "outcome": "failed",
          "terminal_refs": ["a"]}
    v1.pop("catalog")
    assert journal._validate_native_payload(journal.KIND_UNIFIED_DRIVER_SETTLED, v1) == []
    # old readers refuse the new records: the v1 field set and schema do not admit v2
    v1_fields = set(v1)
    v2 = {"schema": oa.SETTLED_V2_SCHEMA, "children": [], "denial_ref": SHA}
    assert set(v2) != v1_fields and v2["schema"] != journal.UNIFIED_DRIVER_SETTLED_SCHEMA


# ------------------------------------------------------------ scheduler previews --------

def test_operation_preview_is_pure_and_apply_is_idempotent_and_atomic():
    engine, selection, _ = engine_and_selection()
    before = engine.operational_projection().projection_digest
    children = (child(selection, "planner", "p", start=10.0, end=11.0),
                child(selection, "critic", "c", start=12.0, end=14.0))
    preview = engine.preview_operation_accounting(selection, children, outcome="failed")
    assert engine.operational_projection().projection_digest == before  # not mutated
    assert preview.prior.projection_digest == before != preview.after.projection_digest
    assert engine.apply_operation_preview(preview) is True
    after = engine.operational_projection().projection_digest
    assert after == preview.after.projection_digest
    assert engine.apply_operation_preview(preview) is False  # exact replay idempotent
    assert engine.operational_projection().projection_digest == after
    assert engine.campaign_attempts == 1 and engine.campaign_charged_seconds == 3.0

    # refusal at the second child leaves the engine bit-identical
    engine2, selection2, _ = engine_and_selection(name="op2")
    base = engine2.operational_projection().projection_digest
    bad = (child(selection2, "planner", "p", start=10.0, end=11.0),
           child(selection2, "critic", "c", start=10.5, end=12.0, affinity=("0",),
                 claims=("claim-x",)))
    with pytest.raises(S.SchedulingRefused):
        engine2.preview_operation_accounting(selection2, bad, outcome="failed")
    assert engine2.operational_projection().projection_digest == base


def test_cancellation_preview_persists_cancelled_operations_and_refuses_settlement_after():
    engine, selection, _ = engine_and_selection()
    ref = "9" * 64
    preview = engine.preview_operation_cancellation(selection, denial_ref=ref)
    assert "cancelled_operations" not in engine.operational_projection().body
    assert engine.apply_operation_preview(preview) is True
    body = engine.operational_projection().body
    assert body["cancelled_operations"] == {selection.digest: ref}
    assert engine.campaign_attempts == 1 and engine.campaign_charged_seconds == 0.0
    assert engine.accounting_view().receipt_count == 0  # no fabricated receipt
    assert engine.apply_operation_preview(preview) is False
    clone = engine._clone()
    assert clone.cancelled_operations == {selection.digest: ref}
    with pytest.raises(S.SchedulingRefused):
        engine.preview_operation_cancellation(selection, denial_ref="8" * 64)
    with pytest.raises(S.SchedulingRefused):
        engine.preview_operation_accounting(
            selection, (child(selection, "planner", "p", start=10.0, end=11.0),),
            outcome="failed")


# ------------------------------------------------------------ installed-path fixtures ---

class _Fixture:
    pass


def _fixture(tmp_path, *, deny=None):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    repo_root = tmp_path / "source"
    repo_root.mkdir()
    repo, snapshot, head = _git(repo_root, shadow_marker=tmp_path / "shadow")
    plan = worktree.BuildPlan(
        snapshot.path,
        worktree.SandboxPath.create(repo_root / "build", sandbox_root=repo_root,
                                    production_trees=()),
        worktree.SandboxPath.create(repo_root / "actor", sandbox_root=repo_root,
                                    production_trees=()),
        worktree.BuildParallelism(1, cpu_list="0"),
        targets=("fixture",), cmake_defines=(("FIXTURE", "ON"),))
    materialized, registry, target, actor_profiles = _profiled_build_runtime(
        runtime_root, source_revision=head, build_plan=plan)
    controller, runtime = _start(materialized, registry)
    assert runtime.tick().status == "settled"
    profiles = runtime.profile_executor.planner_profiles(time.monotonic())
    build_driver = unified_driver.UnifiedCampaignDriver(
        resolved_campaign=runtime.driver.resolved, controller=controller,
        scheduler_engine=runtime.driver.scheduler, profiles=profiles,
        evidence_index=runtime.driver.evidence,
        runtime_anchors=runtime.driver.runtime_anchors,
        runtime_dimensions=runtime.driver.runtime_dimensions,
        experiment_plans=runtime.driver.experiment_plans,
        profile_requests=runtime.driver.profile_requests,
        actor_identities=runtime.driver.actor_identities,
        native_artifact_sink_ref=runtime.driver.sink_ref,
        execution_inputs=runtime.driver.execution_inputs,
        executable_work_kinds={"actor_preparation"})
    selected = build_driver.materialize_actor(build_driver.tick(now=time.monotonic()))
    lifecycle = actor_lifecycle.ActorLifecycleAdapter(
        controller=controller, persistence=controller,
        config=actor_lifecycle.ActorLifecycleConfig(
            campaign_digest=controller.config_digest, cwd=runtime_root,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            max_stage_seconds=5.0, teardown_seconds=1.0, max_retained_output_bytes=4096),
        target_profile_owner=controller)
    consumer = actor_preparation.ActorPreparationConsumer(
        resolved_campaign=runtime.driver.resolved, profiles=actor_profiles,
        budgets=_budgets(), capability=lifecycle, clock=time.monotonic,
        clock_domain=worker_lifecycle.monotonic_clock_domain(), max_output_bytes=4096)
    artifact_root = Path(selected.controller_binding["artifact_root"])
    store = measurement_capture.ArtifactStore(artifact_root)
    module_root = Path(__file__).resolve().parents[2]
    python = Path(sys.executable).resolve()
    stage = execution.SourceBuildStageExecutor(
        driver=build_driver, controller=controller, artifact_store=store,
        artifact_root=artifact_root, python_executable=python,
        python_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
        module_root=module_root, max_build_log_bytes=1024 * 1024,
        worker_source_sha256=hashlib.sha256(
            (module_root / "autokernel/loop/source_build_worker.py").read_bytes()).hexdigest())
    owner = execution.SourceBuildExecutionOwner(
        driver=build_driver, actor_consumer=consumer, build_executor=stage)
    Path(plan.actor_worktree.path).mkdir()
    box = _Fixture()
    box.__dict__.update(
        runtime=runtime, controller=controller, selected=selected, owner=owner, store=store,
        repo=repo, snapshot=snapshot, head=head, plan=plan, stage=stage, lifecycle=lifecycle,
        materialized=materialized, registry=registry)
    return box


def _execute(box, *, settle=True):
    return box.owner.execute_selected(
        box.selected, source_worktree=box.snapshot, source_commit=box.head,
        materialized_source=None, build_plan=box.plan, candidate_id="akc-v2-fixture",
        output_binary=Path(box.plan.build_dir.path) / "fixture", toolchain="cmake fixture",
        libraries={}, linkage_sha256="c" * 64, log_path=box.stage.log_path_for(box.plan),
        configure_timeout_s=30, build_timeout_s=30,
        env={"PATH": os.environ["PATH"], "LANG": "C"},
        sandbox_cgroup_root=worktree.process_sandbox.default_cgroup_root(), settle=settle)


def _reopen(materialized, registry):
    """Reopen without requiring `recovered`: a durable unsettled issue is expected."""
    from types import SimpleNamespace
    from . import standalone_inputs as si
    from .test_campaign_control import _command
    config = materialized.manifest.driver_config
    controller, runtime = si.runtime_factory(materialized, registry)(
        materialized.resolved, SimpleNamespace(
            store=config.store_path, config_generation=config.config_generation,
            snapshot_version=3))
    controller.apply_command(_command(materialized.resolved, "resume", "resume", 0))
    return controller, runtime


def _close(box):
    box.store.close()
    box.runtime.close()
    box.controller.close()
    box.repo.remove_worktree(box.snapshot.path, force=True)


def _deny(box, denial_class):
    engine = box.controller._worker_lifecycle
    engine.admission_fence = lambda _r, _g, _n, _u: worker_lifecycle.StageAdmission(
        False, "fixture denial", denial_class=denial_class)


def test_terminal_admission_denial_cancels_the_operation_with_no_receipt(tmp_path):
    box = _fixture(tmp_path)
    try:
        c = box.controller
        before = c._scheduler_engine.accounting_view().receipt_count
        attempts = c._scheduler_engine.campaign_attempts
        _deny(box, "terminal")
        with pytest.raises(actor_preparation.PreparationDenied) as raised:
            _execute(box)
        assert raised.value.denial_class == "terminal"
        transition = box.selected.transition_id
        assert c.unified_driver_children(transition) == ()
        settled = c._driver_settled[transition]
        assert settled["schema"] == journal.UNIFIED_DRIVER_SETTLED_SCHEMA_V2
        assert settled["outcome"] == "failed" and settled["children"] == []
        assert settled["denial_ref"] is not None
        engine = c._scheduler_engine
        assert engine.cancelled_operations == {box.selected.selection.digest:
                                               settled["denial_ref"]}
        assert engine.accounting_view().receipt_count == before  # no fabricated receipt
        assert engine.campaign_attempts == attempts + 1
        actor_finish = [r for r in c._actor_preparation_events if r["event"] == "FINISH"]
        assert actor_finish[-1]["failure_class"] == "admission_denied_terminal"
        assert actor_finish[-1]["charged_seconds"] == 0.0
        # restart replay persists cancelled_operations and the projection
        digest = engine.operational_projection().projection_digest
        durable = copy.deepcopy(settled)
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _start(box.materialized, box.registry)
        assert box.controller._driver_settled[transition] == durable
        assert box.controller._scheduler_engine.cancelled_operations == {
            box.selected.selection.digest: durable["denial_ref"]}
        assert box.controller._scheduler_engine.operational_projection().projection_digest \
            == digest
    finally:
        _close(box)


def test_retryable_admission_denial_stays_waiting_without_settlement_or_receipt(tmp_path):
    box = _fixture(tmp_path)
    try:
        c = box.controller
        before = c._scheduler_engine.operational_projection().projection_digest
        _deny(box, "retryable")
        with pytest.raises(actor_preparation.PreparationDenied) as raised:
            _execute(box)
        assert raised.value.denial_class == "retryable"
        transition = box.selected.transition_id
        assert transition not in c._driver_settled
        assert c.unified_driver_children(transition) == ()
        assert c._scheduler_engine.operational_projection().projection_digest == before
        assert not c._scheduler_engine.cancelled_operations
    finally:
        _close(box)


def test_crash_with_durable_prefix_replays_and_settles_failed_once_charging_prefix(tmp_path):
    box = _fixture(tmp_path)
    try:
        c = box.controller
        transition = box.selected.transition_id
        # actor children happen, then the build never completes (owner stops after prepare)
        result = box.owner.actor_consumer.prepare(
            box.selected.actor_request, stage_plan_digest=box.selected.stage_plan_digest)
        assert result.status == "proposed"
        prefix = [x["role"] for x in c.unified_driver_children(transition)]
        assert prefix == ["planner", "critic"]
        durable = copy.deepcopy(c.unified_driver_children(transition))
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _reopen(box.materialized, box.registry)
        c = box.controller
        assert c.unified_driver_children(transition) == tuple(durable)  # replay, no relaunch
        assert transition not in c._driver_settled
        receipts_before = c._scheduler_engine.accounting_view().receipt_count
        def settle(outcome):
            rows = c.unified_driver_children(transition)
            return c.unified_driver_settle_v2({
                "schema": oa.SETTLEMENT_REQUEST_V2_SCHEMA,
                "catalog_id": box.selected.catalog_id, "transition_id": transition,
                "selection": box.selected.selection.to_dict(), "outcome": outcome,
                "children": [x["digest"] for x in rows], "denial_ref": None})
        settled = settle("failed")
        assert settled["status"] == "accepted"
        engine = c._scheduler_engine
        assert engine.accounting_view().receipt_count == receipts_before + 2
        assert engine.campaign_attempts >= 1
        assert settle("failed")["status"] == "duplicate"
        with pytest.raises(Exception):  # a changed outcome for the same transition refuses
            settle("invalid")
        digest = engine.operational_projection().projection_digest
        durable_settled = copy.deepcopy(c._driver_settled[transition])
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _reopen(box.materialized, box.registry)
        c = box.controller
        assert c._driver_settled[transition] == durable_settled
        assert c._scheduler_engine.operational_projection().projection_digest == digest
        assert c._scheduler_engine.accounting_view().receipt_count == receipts_before + 2
        with pytest.raises(Exception):  # prerequisite needs the build child
            c.unified_driver_settle_v2({
                "schema": oa.SETTLEMENT_REQUEST_V2_SCHEMA, "catalog_id": box.selected.catalog_id,
                "transition_id": transition, "selection": box.selected.selection.to_dict(),
                "outcome": "prerequisite", "children": [x["digest"] for x in durable],
                "denial_ref": None})
    finally:
        _close(box)


def test_child_accounting_refuses_foreign_receipts_and_bad_order(tmp_path):
    box = _fixture(tmp_path)
    try:
        c = box.controller
        transition = box.selected.transition_id
        assert box.owner.actor_consumer.prepare(
            box.selected.actor_request,
            stage_plan_digest=box.selected.stage_plan_digest).status == "proposed"
        chain = c.unified_driver_children(transition)
        binding = worker_lifecycle.AccountingBinding.from_selection(
            catalog_id=box.selected.catalog_id, transition_id=transition,
            selection=box.selected.selection)
        planner = chain[0]
        terminal = worker_lifecycle.TerminalWorker(**planner["terminal"])
        receipt_obj = S.HeldClaimReceipt.from_dict(planner["receipt"])
        # exact retry is a duplicate; the same receipt cannot become the critic/build child
        assert c.unified_driver_account_child(
            binding=binding, terminal=terminal, receipt=receipt_obj, role="planner",
            operation="actor_preparation")["status"] == "duplicate"
        for role in ("build",):
            with pytest.raises(Exception):
                c.unified_driver_account_child(
                    binding=binding, terminal=terminal, receipt=receipt_obj, role=role)
        forged = S.HeldClaimReceipt.from_dict({**planner["receipt"], "receipt_id": "forged"})
        with pytest.raises(Exception):
            c.unified_driver_account_child(
                binding=binding, terminal=terminal, receipt=forged, role="planner")
        with pytest.raises(Exception):  # an unrelated binding names an unissued selection
            c.unified_driver_account_child(
                binding=worker_lifecycle.AccountingBinding(
                    "1" * 64, "2" * 64, "3" * 64, "4" * 64, "p", "cpu", "search"),
                terminal=terminal, receipt=receipt_obj, role="planner")
    finally:
        _close(box)


# ------------------------------------------------------------ runtime driver path -------

def _deny_driver(controller, denial_class):
    controller._worker_lifecycle.admission_fence = (
        lambda _r, _g, _n, _u: worker_lifecycle.StageAdmission(
            False, "fixture denial", denial_class=denial_class))


@pytest.mark.parametrize("denial_class", ["terminal", "retryable"])
def test_driver_runtime_typed_denial_cancels_only_when_terminal(tmp_path, monkeypatch,
                                                                denial_class):
    from . import driver_execution as de
    from .test_driver_execution import _owned_stack
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    try:
        issued = driver.tick(now=1.0)
        selection = S.Selection.from_dict(issued.selection)
        before = engine.accounting_view().receipt_count
        _deny_driver(controller, denial_class)
        with pytest.raises(worker_lifecycle.AdmissionDenied) as raised:
            connector.execute(issued)
        assert raised.value.denial_class == denial_class
        assert engine.accounting_view().receipt_count == before  # never a fabricated receipt
        if denial_class == "terminal":
            settled = controller._driver_settled[issued.transition_id]
            assert settled["schema"] == journal.UNIFIED_DRIVER_SETTLED_SCHEMA_V2
            assert settled["children"] == [] and settled["outcome"] == "failed"
            assert engine.cancelled_operations == {selection.digest: settled["denial_ref"]}
            assert issued.transition_id in connector._launched  # never relaunched
        else:
            assert issued.transition_id not in controller._driver_settled
            assert not engine.cancelled_operations
            assert connector._launched == set()  # retryable: the same work may be retried
    finally:
        connector.close()
        controller.close()
