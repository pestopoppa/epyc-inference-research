"""Actual CampaignController/Journal persistence for A2 runtime screens."""
from __future__ import annotations

import copy

import pytest

from .. import journal as journal_module
from . import a2_execution_state as state
from . import campaign_control
from . import discovery_execution as X
from . import discovery_screen as D
from .test_campaign_control import _resolved
from .test_discovery_screen import Invoker, _context, _pair, _plan


def _open(controller, logical_id="a2-logical-1", *, pair=None, plan=None, invoker=None):
    pair = pair or _pair()
    plan = plan or _plan(pair)
    invoker = invoker or Invoker(plan)
    execution = X.DiscoveryExecution(
        controller=controller, logical_id=logical_id, plan=plan, pair=pair,
        context=_context(plan), invoker=invoker)
    invoker.frame_digest = execution.screen.frame_digest
    return execution, invoker


def test_actual_controller_journal_restart_reattests_old_bank_and_appends_new_owner(tmp_path):
    store = tmp_path / "service"
    with campaign_control.CampaignController(_resolved("camp-1"), store) as first:
        execution, invoker = _open(first)
        bank = execution.create_bank()
        assert len(invoker.calls) == 3
        original_execution_id = execution.execution_id
        original_supervisor = first.supervisor_incarnation
        assert all(row["supervisor_incarnation"] == original_supervisor
                   for row in (entry.payload for entry in first._journal.read_all()
                               if entry.kind == journal_module.KIND_A2_RUNTIME_EXECUTION))

    with campaign_control.CampaignController(_resolved("camp-1"), store) as restarted:
        execution, invoker = _open(restarted)
        assert execution.execution_id == original_execution_id
        before_retry = restarted._journal.read_all()
        original_terminal = copy.deepcopy(execution.screen._events[1])
        assert restarted.append_a2_runtime_transition(
            logical_id="a2-logical-1", plan=execution.plan,
            event=original_terminal) == original_terminal
        assert restarted._journal.read_all() == before_retry
        restored = execution.create_bank()
        assert restored.to_dict() == bank.to_dict()
        assert invoker.calls == []
        receipt = execution.run_screen(restored)
        assert len(receipt.candidate_results) == 3
        assert [arm for arm, *_ in invoker.calls] == ["candidate"] * 3
        owners = [entry.payload["supervisor_incarnation"]
                  for entry in restarted._journal.read_all()
                  if entry.kind == journal_module.KIND_A2_RUNTIME_EXECUTION]
        assert owners[:7] == [original_supervisor] * 7
        assert owners[7:] == [restarted.supervisor_incarnation] * 7


def test_same_controller_persists_exact_bank_reference_and_runs_candidates_only(tmp_path):
    with campaign_control.CampaignController(
            _resolved("camp-1"), tmp_path / "service") as controller:
        source, source_invoker = _open(controller, logical_id="bank-source")
        bank = source.create_bank()
        target, target_invoker = _open(controller, logical_id="bank-consumer")
        receipt = target.run_screen(bank, bank_verifier=source.screen.bank_verifier())
        assert len(source_invoker.calls) == 3
        assert [row[0] for row in target_invoker.calls] == ["candidate"] * 3
        rows = [entry.payload for entry in controller._journal.read_all()
                if entry.kind == journal_module.KIND_A2_RUNTIME_EXECUTION
                and entry.record_id == target.execution_id]
        assert len(rows) == 8
        reference = rows[0]["event"]
        assert reference["schema"] == state.BANK_REFERENCE_SCHEMA
        assert reference["source_execution_id"] == source.execution_id
        assert reference["bank_digest"] == bank.bank_digest == receipt.bank_digest
        assert len(reference["source_anchor_event_digests"]) == 7
        assert all(row["event"].get("phase") != "anchor_bank" for row in rows[1:])


def test_clean_restart_reopens_original_bank_source_without_anchor_launches(tmp_path):
    store = tmp_path / "service"
    with campaign_control.CampaignController(_resolved("camp-1"), store) as controller:
        source, _ = _open(controller, logical_id="bank-source")
        bank = source.create_bank()
        target, target_invoker = _open(controller, logical_id="bank-consumer")
        expected = target.run_screen(bank, bank_verifier=source.screen.bank_verifier())
        assert len(target_invoker.calls) == 3

    with campaign_control.CampaignController(_resolved("camp-1"), store) as restarted:
        target, replay_invoker = _open(restarted, logical_id="bank-consumer")
        assert target.reused_bank is not None
        assert target.reused_bank.to_dict() == bank.to_dict()
        actual = target.run_screen(target.reused_bank)
        assert actual.to_dict() == expected.to_dict()
        assert replay_invoker.calls == []


def test_restart_refuses_bank_reference_when_original_source_is_missing(tmp_path):
    source_store = tmp_path / "source-service"
    with campaign_control.CampaignController(_resolved("camp-1"), source_store) as controller:
        source, _ = _open(controller, logical_id="bank-source")
        bank = source.create_bank()
        target, _ = _open(controller, logical_id="bank-consumer")
        target.run_screen(bank, bank_verifier=source.screen.bank_verifier())
        reference_entry = next(
            copy.deepcopy(entry) for entry in controller._journal.read_all()
            if entry.kind == journal_module.KIND_A2_RUNTIME_EXECUTION
            and entry.record_id == target.execution_id
            and entry.payload["event"].get("schema") == state.BANK_REFERENCE_SCHEMA)

    orphan_store = tmp_path / "orphan-service"
    orphan = campaign_control.CampaignController(_resolved("camp-1"), orphan_store)
    orphan.__enter__()
    orphan._journal.append(
        journal_module.KIND_A2_RUNTIME_EXECUTION, reference_entry.payload,
        campaign_id="camp-1", record_id=reference_entry.record_id)
    orphan.close()
    with pytest.raises(journal_module.JournalCorruption, match="source is absent"):
        campaign_control.CampaignController(_resolved("camp-1"), orphan_store).__enter__()


def test_restart_refuses_forged_bank_reference_despite_valid_reference_digest(tmp_path):
    store = tmp_path / "service"
    controller = campaign_control.CampaignController(_resolved("camp-1"), store)
    controller.__enter__()
    source, _ = _open(controller, logical_id="bank-source")
    source.create_bank()
    source_entries = controller._a2_execution_entries[source.execution_id]
    reference = state.make_bank_reference(
        source_values=[entry.payload for entry in source_entries],
        source_journal_entry_ids=[entry.event_id for entry in source_entries],
        target_plan_digest=source.plan.digest,
        target_frame_digest=source.screen.frame_digest)
    reference["bank_digest"] = "f" * 64
    reference["reference_digest"] = state._digest({
        name: value for name, value in reference.items() if name != "reference_digest"})
    transition = state.make_transition(
        campaign_id="camp-1", config_generation=controller.config_generation,
        config_digest=controller.config_digest,
        supervisor_incarnation=controller.supervisor_incarnation,
        logical_id="forged-consumer", event=reference)
    controller._journal.append(
        journal_module.KIND_A2_RUNTIME_EXECUTION, transition,
        campaign_id="camp-1", record_id=transition["execution_id"])
    controller.close()
    with pytest.raises(journal_module.JournalCorruption, match="original source history"):
        campaign_control.CampaignController(_resolved("camp-1"), store).__enter__()


def test_restart_refuses_bank_reference_rebound_to_a_stale_frame(tmp_path):
    store = tmp_path / "service"
    controller = campaign_control.CampaignController(_resolved("camp-1"), store)
    controller.__enter__()
    source, _ = _open(controller, logical_id="bank-source")
    source.create_bank()
    source_entries = controller._a2_execution_entries[source.execution_id]
    reference = state.make_bank_reference(
        source_values=[entry.payload for entry in source_entries],
        source_journal_entry_ids=[entry.event_id for entry in source_entries],
        target_plan_digest=source.plan.digest,
        target_frame_digest=source.screen.frame_digest)
    reference["target_frame_digest"] = "e" * 64
    reference["reference_digest"] = state._digest({
        name: value for name, value in reference.items() if name != "reference_digest"})
    transition = state.make_transition(
        campaign_id="camp-1", config_generation=controller.config_generation,
        config_digest=controller.config_digest,
        supervisor_incarnation=controller.supervisor_incarnation,
        logical_id="stale-frame-consumer", event=reference)
    controller._journal.append(
        journal_module.KIND_A2_RUNTIME_EXECUTION, transition,
        campaign_id="camp-1", record_id=transition["execution_id"])
    controller.close()
    with pytest.raises(journal_module.JournalCorruption, match="source frame differs"):
        campaign_control.CampaignController(_resolved("camp-1"), store).__enter__()


def test_import_refuses_stale_frame_and_external_verifier_without_local_source(tmp_path):
    with campaign_control.CampaignController(
            _resolved("camp-1"), tmp_path / "service") as controller:
        source, _ = _open(controller, logical_id="bank-source")
        bank = source.create_bank()
        changed_context = _context(source.plan).to_dict()
        changed_context["resource_claim"] = {
            "claim_id": "different-claim", "region": "cpu-quarter"}
        invoker = Invoker(source.plan)
        target = X.DiscoveryExecution(
            controller=controller, logical_id="stale-frame-consumer",
            plan=source.plan, pair=source.pair, context=changed_context, invoker=invoker)
        invoker.frame_digest = target.screen.frame_digest
        with pytest.raises(campaign_control.ControlRefused, match="original sealed source"):
            target.run_screen(bank, bank_verifier=source.screen.bank_verifier())
        assert invoker.calls == []

    with campaign_control.CampaignController(
            _resolved("camp-1"), tmp_path / "foreign-service") as foreign:
        foreign_source, _ = _open(foreign, logical_id="foreign-source")
        foreign_bank = foreign_source.create_bank()
        foreign_verifier = foreign_source.screen.bank_verifier()
    with campaign_control.CampaignController(
            _resolved("camp-1"), tmp_path / "empty-service") as empty:
        target, invoker = _open(empty, logical_id="unproven-consumer")
        with pytest.raises(campaign_control.ControlRefused, match="original sealed source"):
            target.run_screen(foreign_bank, bank_verifier=foreign_verifier)
        assert invoker.calls == []


def test_uncertain_append_poison_restart_replays_pending_and_exact_retry_keeps_bytes(tmp_path):
    store = tmp_path / "service"
    controller = campaign_control.CampaignController(_resolved("camp-1"), store)
    controller.__enter__()
    execution, invoker = _open(controller)
    original_append = controller._journal.append

    def append_then_lose_reply(*args, **kwargs):
        original_append(*args, **kwargs)
        raise OSError("reply lost after fsync")

    controller._journal.append = append_then_lose_reply
    with pytest.raises(OSError, match="reply lost"):
        execution.create_bank()
    assert len(invoker.calls) == 0
    assert controller._poisoned is True
    controller.close()

    with campaign_control.CampaignController(_resolved("camp-1"), store) as restarted:
        replayed, invoker = _open(restarted)
        assert len(replayed.pending_intents) == 1
        with pytest.raises(D.DiscoveryScreenRefused, match="reconciliation"):
            replayed.create_bank()
        assert invoker.calls == []
        before = restarted._journal.read_all()
        event = copy.deepcopy(replayed.screen._events[0])
        assert restarted.append_a2_runtime_transition(
            logical_id="a2-logical-1", plan=replayed.plan, event=event) == event
        after = restarted._journal.read_all()
        assert after == before


def test_uncertain_terminal_append_recovers_exact_native_result_without_reissue(tmp_path):
    store = tmp_path / "service"
    controller = campaign_control.CampaignController(_resolved("camp-1"), store)
    controller.__enter__()
    execution, first_invoker = _open(controller)
    original_append = controller._journal.append

    def lose_terminal_reply(kind, payload, **kwargs):
        entry = original_append(kind, payload, **kwargs)
        if (kind == journal_module.KIND_A2_RUNTIME_EXECUTION
                and payload["event"]["state"] == "TERMINAL"):
            raise OSError("terminal reply lost after fsync")
        return entry

    controller._journal.append = lose_terminal_reply
    with pytest.raises(OSError, match="terminal reply lost"):
        execution.create_bank()
    assert len(first_invoker.calls) == 1
    first_launch = "launch:process-0"
    controller.close()

    with campaign_control.CampaignController(_resolved("camp-1"), store) as restarted:
        replayed, second_invoker = _open(restarted)
        assert replayed.pending_intents == ()
        bank = replayed.create_bank()
        assert len(second_invoker.calls) == 2
        assert bank.anchor_results[0].proof.invocation_identity["launch_id"] == first_launch


def test_same_logical_execution_identity_drift_refuses(tmp_path):
    with campaign_control.CampaignController(
            _resolved("camp-1"), tmp_path / "service") as controller:
        execution, _ = _open(controller)
        execution.create_bank()
        changed = execution.plan.to_dict()
        changed["target_revision"] = "changed-target"
        changed_plan = type(execution.plan).from_dict(changed)
        with pytest.raises(campaign_control.ControlRefused, match="identity drift"):
            _open(controller, plan=changed_plan, pair=execution.pair,
                  invoker=Invoker(changed_plan))


def test_closed_old_owner_callback_refuses_after_restart(tmp_path):
    store = tmp_path / "service"
    first = campaign_control.CampaignController(_resolved("camp-1"), store)
    first.__enter__()
    execution, _ = _open(first)
    execution.create_bank()
    event = copy.deepcopy(execution.screen._events[0])
    first.close()
    with campaign_control.CampaignController(_resolved("camp-1"), store):
        with pytest.raises(campaign_control.ControlRefused, match="closed"):
            execution._append_phase(event)


def test_journal_native_validator_rejects_tampered_transition(tmp_path):
    journal = journal_module.Journal(str(tmp_path / "journal"), campaign_id="campaign-1")
    journal.initialize()
    pair, plan = _pair(), _plan(_pair())
    invoker = Invoker(plan)
    probe = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=invoker,
                              phase_sink=lambda event: event)
    unit = sorted(plan.expected_units, key=lambda item: item.order_index)[0]
    event = D._event("anchor_bank", "INTENT", 0, plan.digest, probe.frame_digest, {
        "unit": unit.to_dict(), "producer_identity": {
            "frame_digest": probe.frame_digest,
            "recipe_snapshot_digest": pair.anchor.snapshot_digest,
            "recipe_execution_digest": pair.anchor.execution_digest}})
    row = state.make_transition(
        campaign_id="campaign-1", config_generation=1, config_digest="3" * 64,
        supervisor_incarnation=1, logical_id="logical", event=event)
    row["execution_id"] = "4" * 64
    with pytest.raises(ValueError, match="identity differs"):
        journal.append(journal_module.KIND_A2_RUNTIME_EXECUTION, row)


def test_phase_append_and_replay_use_index_without_rescanning_journal(tmp_path):
    with campaign_control.CampaignController(
            _resolved("camp-1"), tmp_path / "service") as controller:
        execution, _ = _open(controller)
        controller._journal.read_all = lambda: (_ for _ in ()).throw(
            AssertionError("hot path rescanned Journal"))
        bank = execution.create_bank()
        assert len(bank.anchor_results) == 3
        replay = controller.replay_a2_runtime_execution(
            logical_id="a2-logical-1", plan_digest=execution.plan.digest,
            frame_digest=execution.screen.frame_digest)
        assert len(replay["events"]) == 7


def test_controller_mutex_is_not_held_during_invocation(tmp_path):
    with campaign_control.CampaignController(
            _resolved("camp-1"), tmp_path / "service") as controller:
        pair, plan = _pair(), _plan(_pair())

        class CheckingInvoker(Invoker):
            def invoke(self, unit, recipe):
                assert not controller._mutex._is_owned()
                return super().invoke(unit, recipe)

        execution, invoker = _open(
            controller, pair=pair, plan=plan, invoker=CheckingInvoker(plan))
        assert len(execution.create_bank().anchor_results) == 3
        assert len(invoker.calls) == 3


def test_controller_refuses_phase_intent_outside_frozen_plan_membership(tmp_path):
    with campaign_control.CampaignController(
            _resolved("camp-1"), tmp_path / "service") as controller:
        execution, _ = _open(controller)
        unit = sorted(execution.plan.expected_units,
                      key=lambda item: item.order_index)[0].to_dict()
        unit["unit_id"] = "forged-unit"
        event = D._event("anchor_bank", "INTENT", 0, execution.plan.digest,
                         execution.screen.frame_digest, {"unit": unit,
                         "producer_identity": {
                             "frame_digest": execution.screen.frame_digest,
                             "recipe_snapshot_digest": execution.pair.anchor.snapshot_digest,
                             "recipe_execution_digest": execution.pair.anchor.execution_digest}})
        with pytest.raises(campaign_control.ControlRefused, match="frozen plan"):
            controller.append_a2_runtime_transition(
                logical_id="a2-logical-1", plan=execution.plan, event=event)
        assert controller.replay_a2_runtime_execution(
            logical_id="a2-logical-1", plan_digest=execution.plan.digest,
            frame_digest=execution.screen.frame_digest)["events"] == ()
