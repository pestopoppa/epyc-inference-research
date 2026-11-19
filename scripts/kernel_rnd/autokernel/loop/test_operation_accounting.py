"""AKU-06l hermetic slice: original multi-child accounting, typed no-launch cancellation.

Synthetic labelled data only; no hardware or campaign claim."""
from __future__ import annotations

import pytest

from . import scheduling as S
from .test_scheduling import config, proposal, receipt, vector

DENIAL = "d" * 64


def engine_and_selection(*, name="op", seed=None, **cfg):
    cfg_obj = S.SchedulerConfig.from_dict(config(**cfg))
    engine = S.SchedulerEngine(cfg_obj, S.initial_state(cfg_obj, "scheduler"))
    prop = proposal(name, seed=seed)
    selection = engine.select_stage([prop], now=1.0)
    assert selection.status == "selected"
    return engine, selection, prop


def child(selection, role, name, *, start, end, affinity=("0",), claims=None,
          fraction=0.25, memory=100):
    # Original child receipts keep their own (setup/build) provider identity,
    # deliberately NOT the selected proposal's identity.
    row = receipt({"proposal_id": f"{role}-request", "backend": "cpu",
                   "stage_class": "search"}, name, start=start, end=end,
                  fraction=fraction, memory=memory, affinity=affinity,
                  claims=claims or (f"claim-{name}",))
    return S.OperationChild(role, selection.digest, S.HeldClaimReceipt.from_dict(row))


def test_operation_accounts_original_children_once_without_composite_receipt():
    engine, selection, _prop = engine_and_selection()
    children = [child(selection, "planner", "r1", start=0.0, end=2.0),
                child(selection, "critic", "r2", start=2.0, end=3.0),
                child(selection, "build", "r3", start=3.0, end=7.0)]
    attempts = engine.campaign_attempts
    assert engine.account_operation(selection, children, outcome="prerequisite") is True
    assert engine.campaign_attempts == attempts + 1            # one attempt, not three
    assert engine.campaign_charged_seconds == pytest.approx(7.0)
    state = engine.export_state()
    assert [r.receipt_id for r in state.receipts] == ["r1", "r2", "r3"]
    assert [r.selection_digest for r in state.accounted_receipts] == [selection.digest] * 3
    assert engine.issued_selection_digests == ()
    # exact retry is idempotent, no second attempt
    assert engine.account_operation(selection, children, outcome="prerequisite") is False
    assert engine.campaign_attempts == attempts + 1


def test_operation_failure_prefixes_are_valid_and_gaps_are_not_required():
    for roles in (1, 2):
        engine, selection, _ = engine_and_selection()
        kids = [child(selection, "planner", "r1", start=0.0, end=2.0),
                child(selection, "critic", "r2", start=5.0, end=6.0)][:roles]
        assert engine.account_operation(selection, kids, outcome="failed")
        assert engine.campaign_charged_seconds == pytest.approx(
            sum(k.receipt.ended_at - k.receipt.started_at for k in kids))


def test_operation_refuses_bad_roles_order_and_foreign_selection_atomically():
    engine, selection, _ = engine_and_selection()
    before = engine.operational_projection().projection_digest
    planner = child(selection, "planner", "r1", start=0.0, end=2.0)
    build = child(selection, "build", "r3", start=3.0, end=4.0)
    critic = child(selection, "critic", "r2", start=2.0, end=3.0)
    for bad in ([build], [critic, planner], [planner, planner], [planner, build], []):
        with pytest.raises(S.SchedulingRefused):
            engine.account_operation(selection, bad, outcome="failed")
    other = S.OperationChild("planner", "e" * 64, planner.receipt)
    with pytest.raises(S.SchedulingRefused, match="different selection"):
        engine.account_operation(selection, [other], outcome="failed")
    assert engine.operational_projection().projection_digest == before
    assert engine.export_state().receipts == ()


def test_cross_role_receipt_reuse_is_refused_even_with_identical_bytes():
    engine, selection, _ = engine_and_selection()
    planner = child(selection, "planner", "r1", start=0.0, end=2.0)
    critic = S.OperationChild("critic", selection.digest, planner.receipt)
    with pytest.raises(S.SchedulingRefused, match="unique"):
        engine.account_operation(selection, [planner, critic], outcome="failed")
    assert engine.campaign_attempts == 0


def test_changed_retry_and_changed_same_id_content_refuse():
    engine, selection, _ = engine_and_selection()
    first = [child(selection, "planner", "r1", start=0.0, end=2.0)]
    engine.account_operation(selection, first, outcome="failed")
    changed = [child(selection, "planner", "r1", start=0.0, end=3.0)]
    with pytest.raises(S.SchedulingRefused):
        engine.account_operation(selection, changed, outcome="failed")
    with pytest.raises(S.SchedulingRefused):
        engine.account_operation(selection, first, outcome="invalid")


def test_partitioned_overlap_is_occupancy_seconds_exceeding_wall_time():
    engine, selection, _ = engine_and_selection()
    kids = [child(selection, "planner", "r1", start=0.0, end=4.0, affinity=("0",)),
            child(selection, "critic", "r2", start=1.0, end=5.0, affinity=("1",))]
    engine.account_operation(selection, kids, outcome="failed")
    assert engine.campaign_charged_seconds == pytest.approx(8.0)   # wall time is 5.0


@pytest.mark.parametrize("kw2,match", [
    (dict(affinity=("0",)), "affinity"),
    (dict(affinity=("1",), claims=("claim-r1",)), "physical claim"),
    (dict(affinity=("1",), fraction=0.9), "capacity"),
    (dict(affinity=("1",), memory=10**9), "capacity"),
])
def test_overlap_without_disjoint_partition_or_within_capacity_refuses(kw2, match):
    engine, selection, _ = engine_and_selection()
    kids = [child(selection, "planner", "r1", start=0.0, end=4.0, fraction=0.5),
            child(selection, "critic", "r2", start=1.0, end=5.0, **{"fraction": 0.5, **kw2})]
    with pytest.raises(S.SchedulingRefused, match=match):
        engine.account_operation(selection, kids, outcome="failed")
    assert engine.campaign_attempts == 0 and engine.export_state().receipts == ()


def test_overlapping_gpu_children_refuse():
    engine, selection, _ = engine_and_selection()
    gpu = lambda name, role, start, end: S.OperationChild(
        role, selection.digest, S.HeldClaimReceipt.from_dict(receipt(
            {"proposal_id": role, "backend": "cpu", "stage_class": "search"}, name,
            start=start, end=end, fraction=0.1, memory=10, gpus=("gpu0",),
            claims=(f"claim-{name}",), affinity=(str(start),))))
    with pytest.raises(S.SchedulingRefused, match="GPU"):
        engine.account_operation(selection, [gpu("r1", "planner", 0.0, 3.0),
                                             gpu("r2", "critic", 1.0, 4.0)], outcome="failed")


def test_terminal_cancellation_is_explicit_zero_cost_one_attempt_no_receipt():
    engine, selection, _ = engine_and_selection(name="cancel")
    before_view = engine.accounting_view()
    assert engine.cancel_operation_before_launch(selection, denial_ref=DENIAL) is True
    assert engine.campaign_attempts == 1
    assert engine.campaign_charged_seconds == 0.0
    assert engine.export_state().receipts == () and engine.export_state().accounted_receipts == ()
    assert engine.accounting_view().view_digest == before_view.view_digest
    assert engine.issued_selection_digests == ()
    assert engine.operational_projection().body["cancelled_operations"] == {
        selection.digest: DENIAL}
    # idempotent for the same denial; refused for a different denial or later settlement
    assert engine.cancel_operation_before_launch(selection, denial_ref=DENIAL) is False
    assert engine.campaign_attempts == 1
    with pytest.raises(S.SchedulingRefused, match="different denial"):
        engine.cancel_operation_before_launch(selection, denial_ref="a" * 64)
    with pytest.raises(S.SchedulingRefused, match="cancelled"):
        engine.account_operation(
            selection, [child(selection, "planner", "r1", start=0.0, end=1.0)],
            outcome="failed")


def test_cancellation_refuses_retryable_unissued_and_bad_ref():
    engine, selection, _ = engine_and_selection(name="c2")
    with pytest.raises(S.SchedulingRefused, match="terminal"):
        engine.cancel_operation_before_launch(selection, denial_ref=DENIAL,
                                              denial_class="retryable")
    with pytest.raises(S.SchedulingRefused, match="SHA-256"):
        engine.cancel_operation_before_launch(selection, denial_ref="nope")
    assert engine.campaign_attempts == 0 and engine.issued_selection_digests
    engine.fail_stage_before_claim(selection)           # legacy path unchanged
    assert engine.campaign_attempts == 0
    with pytest.raises(S.SchedulingRefused, match="issued"):
        engine.cancel_operation_before_launch(selection, denial_ref=DENIAL)


def test_cancellation_counts_a_seed_attempt_with_zero_charged_seconds():
    engine, selection, _ = engine_and_selection(name="sd", seed="seed-1")
    engine.cancel_operation_before_launch(selection, denial_ref=DENIAL)
    seed = engine.export_state().seed_accounts[0]
    assert seed.attempts == 1 and seed.charged_seconds == 0.0


def test_legacy_operational_projection_has_no_cancellation_key():
    engine, selection, _ = engine_and_selection()
    assert "cancelled_operations" not in engine.operational_projection().body
    engine.fail_stage_before_claim(selection)
    assert "cancelled_operations" not in engine.operational_projection().body
