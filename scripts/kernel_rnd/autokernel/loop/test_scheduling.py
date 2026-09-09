from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from . import scheduling as S


def vector(*, fraction=1.0, gpus=(), memory=1000):
    return {"schema": S.VECTOR_SCHEMA, "physical_region_fraction": fraction,
            "gpu_devices": list(gpus), "memory_reservation_bytes": memory}


def config(**changes):
    row = {"schema": S.CONFIG_SCHEMA, "config_id": "provisional-v1",
           "max_stage_seconds": 10.0, "noncoverage_slots": 2,
           "reservation_slots": {}, "reservation_shares": {}, "campaign_attempt_cap": 20,
           "campaign_charged_seconds_cap": 100.0, "seed_attempt_cap": 2,
           "seed_charged_seconds_cap": 15.0,
           "capacity": vector(gpus=("gpu0",), memory=10000),
           "weights_source": "provisional manifest constants",
           "apportionment_rule": "explicit normalized beneficiary shares",
           "adaptive_rule_id": None, "normal_weight": 1.0, "seed_weight": 2.0,
           "seed_valid_comparison_cap": 3}
    row.update(changes)
    return row


def proposal(name="p", *, frontier=None, seed=None, backend="cpu", submitted=1.0,
             stage="search", duration=5.0, claims=None, eligible=True,
             reservation=None, full=False, alias=None, chunked=False, target=None):
    return {"schema": S.PROPOSAL_SCHEMA, "proposal_id": name,
            "submitted_at": submitted, "backend": backend,
            "target_revision": target or f"revision-{name}",
            "alias_identity": alias or f"alias-{name}",
            "frontier_id": frontier, "production_frontier": frontier is not None,
            "seed_id": seed, "stage_class": stage,
            "estimated_duration_seconds": duration,
            "estimated_claims": claims or vector(fraction=0.5, memory=500),
            "eligible": eligible, "eligibility_ref": "trusted:eligible-receipt",
            "reservation_kind": reservation, "full_region": full,
            "compatibility_authority_refs": [], "safe_chunking_declared": chunked}


def receipt(prop, name="r", *, start=0.0, end=5.0, fraction=0.5, gpus=(),
            memory=500, claims=("cpu-region",), affinity=("0",), shares=None):
    return {"schema": S.RECEIPT_SCHEMA, "receipt_id": name,
            "proposal_id": prop["proposal_id"], "backend": prop["backend"],
            "stage_class": prop["stage_class"], "started_at": start, "ended_at": end,
            "ownership_generation": 1, "allocation_generation": 1,
            "physical_claim_ids": list(claims), "physical_region_fraction": fraction,
            "gpu_device_ids": list(gpus), "memory_reservation_bytes": memory,
            "affinity_cores": list(affinity),
            "beneficiary_shares": shares or {prop["proposal_id"]: 1.0}}


def choose(cfg, state, rows, now=0.0, outages=()):
    return S.select_stage(cfg, state, rows, now=now, outages=outages)


def account(cfg, state, selected, prop, *, rid="r", outcome="invalid", **kwargs):
    return S.account_stage(cfg, state, selected, receipt(prop, rid, **kwargs), outcome=outcome)


def test_strict_direct_normalization_freezes_nested_inputs_and_rejects_bool_nan_unknown():
    source = config()
    cfg = S.SchedulerConfig.from_dict(source)
    source["reservation_slots"]["seed"] = [0]
    assert dict(cfg.reservation_slots) == {}
    with pytest.raises(TypeError):
        cfg.reservation_slots["seed"] = (0,)
    with pytest.raises(FrozenInstanceError):
        cfg.noncoverage_slots = 9
    for bad in (True, float("nan"), float("inf")):
        with pytest.raises(S.SchedulingRefused):
            S.SchedulerConfig.from_dict(config(max_stage_seconds=bad))
    malformed = config()
    malformed["unknown"] = 1
    with pytest.raises(S.SchedulingRefused):
        S.SchedulerConfig.from_dict(malformed)
    with pytest.raises(S.SchedulingRefused, match="backend"):
        S.StageProposal.from_dict(proposal(backend="remote"))
    outage = {"schema": S.OUTAGE_SCHEMA, "outage_id": "o", "kind": "resource",
              "started_at": 0.0, "ended_at": None, "reason": "fixture",
              "backend": "remote", "frontier_id": None}
    with pytest.raises(S.SchedulingRefused, match="backend"):
        S.Outage.from_dict(outage)
    with pytest.raises(S.SchedulingRefused):
        S.StageProposal(**(S.StageProposal.from_dict(proposal()).to_dict()
                           | {"estimated_claims": {"mutable": True}}))


def test_frozen_frontier_round_ignores_arrival_flood_until_next_round():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    a = proposal("a", frontier="prod-a")
    state, selected = choose(cfg, state, [a])
    assert state.frozen_frontier == ("prod-a",) and selected.proposal.proposal_id == "a"
    state = account(cfg, state, selected, a)
    b = proposal("b", frontier="prod-b", submitted=0)
    noncoverage = proposal("n")
    state, selected = choose(cfg, state, [a, b, noncoverage])
    assert state.frozen_frontier == ("prod-a",)
    assert selected.proposal.proposal_id == "n"
    state = account(cfg, state, selected, noncoverage, rid="r2", claims=("other",))
    state, selected = choose(cfg, state, [a, b])
    assert state.round_number == 2 and state.frozen_frontier == ("prod-a", "prod-b")


def test_every_expensive_stage_class_consumes_exactly_one_slot():
    for stage in sorted(S.STAGE_CLASSES):
        cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
        state = S.initial_state(cfg, "scheduler")
        prop = proposal(stage, stage=stage)
        state, selected = choose(cfg, state, [prop])
        state = account(cfg, state, selected, prop)
        assert state.used_noncoverage == (stage,)
        assert state.campaign_attempts == 1


def test_seed_fifo_invalid_cap_unblocks_later_seed_and_alias_gets_no_new_boost():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=2, seed_attempt_cap=1))
    state = S.initial_state(cfg, "scheduler")
    old = proposal("old", seed="seed-old", submitted=1, alias="same")
    later = proposal("later", seed="seed-later", submitted=2)
    state, selected = choose(cfg, state, [later, old])
    assert selected.proposal.proposal_id == "old"
    state = account(cfg, state, selected, old, outcome="invalid")
    state, selected = choose(cfg, state, [old, later])
    assert selected.proposal.proposal_id == "later"
    state = account(cfg, state, selected, later, rid="r2", claims=("other",))
    duplicate = proposal("alias", seed="seed-alias", alias="same", submitted=0,
                         target=old["target_revision"])
    state, _ = choose(cfg, state, [duplicate])
    shared = next(seed for seed in state.seed_accounts if seed.seed_id == "seed-old")
    assert shared.boosted is False and shared.seed_ids == ("seed-old", "seed-alias")
    assert shared.first_submitted_at == 1


def test_only_oldest_seed_per_backend_holds_boost_until_exhausted():
    cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=1, seed_attempt_cap=2, seed_charged_seconds_cap=100))
    state = S.initial_state(cfg, "scheduler")
    old = proposal("old", seed="seed-old", submitted=1, duration=9)
    later = proposal("later", seed="seed-later", submitted=2, duration=1)
    state, selected = choose(cfg, state, [later, old])
    assert selected.proposal.proposal_id == "old"
    state = account(cfg, state, selected, old, end=1, outcome="invalid")
    assert next(seed for seed in state.seed_accounts if seed.seed_id == "seed-later").boosted is False
    state, selected = choose(cfg, state, [later, old], now=2)
    assert selected.proposal.proposal_id == "old"
    state = account(cfg, state, selected, old, rid="r2", start=2, end=3,
                    claims=("other",), outcome="invalid")
    assert next(seed for seed in state.seed_accounts if seed.seed_id == "seed-later").boosted is True
    reused = proposal("reused", seed="seed-old", backend="gpu")
    with pytest.raises(S.SchedulingRefused, match="different immutable identity"):
        choose(cfg, state, [reused], now=4)


def test_seed_requires_k_and_conflicting_reservations_refuse():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=0))
    state = S.initial_state(cfg, "scheduler")
    _, result = choose(cfg, state, [proposal("seed", seed="s")])
    assert result.status == "refused" and "K>=1" in result.reasons[0]
    with pytest.raises(S.SchedulingRefused, match="conflicts"):
        S.SchedulerConfig.from_dict(config(
            reservation_slots={"validation": [0], "calibration": [0]},
            reservation_shares={"validation": 0.5, "calibration": 0.5}))
    reserved = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=1, reservation_slots={"seed": [0]},
        reservation_shares={"seed": 1.0}))
    _, missing = choose(reserved, S.initial_state(reserved, "scheduler"), [proposal("normal")])
    assert missing.status == "waiting" and "seed" in missing.reasons[0]


def test_temporarily_ineligible_frozen_frontier_waits_without_resetting_round():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    ready = proposal("a", frontier="prod")
    state, _ = choose(cfg, state, [ready])
    blocked = proposal("a", frontier="prod", eligible=False)
    retained, result = choose(cfg, state, [blocked])
    assert result.status == "waiting"
    assert retained.round_number == 1 and retained.frozen_frontier == ("prod",)


def test_seed_boost_ends_at_three_valid_comparisons_across_rounds():
    cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=1, seed_attempt_cap=10, seed_charged_seconds_cap=100))
    state = S.initial_state(cfg, "scheduler")
    prop = proposal("seed", seed="s")
    for index in range(3):
        state, selected = choose(cfg, state, [prop], now=index * 6)
        state = account(cfg, state, selected, prop, rid=f"r{index}",
                        start=index * 6, end=index * 6 + 5,
                        claims=(f"claim-{index}",), outcome="valid_comparison")
    seed = state.seed_accounts[0]
    assert seed.valid_comparisons == 3 and seed.boosted is False
    state, selected = choose(cfg, state, [prop], now=20)
    assert selected.status == "selected", "completed boost must retain ordinary exploration budget"


def test_held_integral_uses_claim_fraction_not_affinity_and_gpu_pays_host():
    cpu = proposal("cpu")
    gpu = proposal("gpu", backend="gpu", claims=vector(fraction=0.25, gpus=("gpu0",)))
    rows = [receipt(cpu, "cpu-r", end=4, fraction=0.25, affinity=("0", "1")),
            receipt(gpu, "gpu-r", start=4, end=10, fraction=0.25, gpus=("gpu0",),
                    claims=("cpu-region",), affinity=("0",))]
    view = S.charge_receipts(rows)
    assert view.physical_region_seconds == pytest.approx(2.5)
    assert view.gpu_device_seconds == {"gpu0": 6.0}
    assert view.held_seconds == 10.0
    with pytest.raises(TypeError):
        view.gpu_device_seconds["gpu0"] = 0
    assert S.AccountingView.from_dict(view.to_dict()) == view
    with pytest.raises(S.SchedulingRefused, match="digest"):
        S.AccountingView.from_dict(view.to_dict() | {"held_seconds": 11})
    with pytest.raises(S.SchedulingRefused, match="physical claim ID"):
        S.HeldClaimReceipt.from_dict(receipt(cpu, claims=()))


def test_duplicate_receipt_idempotent_but_conflict_and_overlap_refuse():
    prop = proposal("p")
    row = receipt(prop)
    assert S.charge_receipts([row, row]).receipt_count == 1
    changed = dict(row, ended_at=4.0)
    with pytest.raises(S.SchedulingRefused, match="different content"):
        S.charge_receipts([row, changed])
    overlap = receipt(prop, "r2", start=4, end=6)
    with pytest.raises(S.SchedulingRefused, match="overlapping"):
        S.charge_receipts([row, overlap])


def test_shared_build_is_charged_once_and_explicitly_apportioned():
    prop = proposal("build", stage="build")
    row = receipt(prop, end=10, shares={"a": 0.25, "b": 0.75})
    view = S.charge_receipts([row])
    assert view.held_seconds == 10
    assert view.beneficiary_seconds == {"a": 2.5, "b": 7.5}
    bad = dict(row, beneficiary_shares={"a": 0.5, "b": 0.6})
    with pytest.raises(S.SchedulingRefused, match="sum"):
        S.HeldClaimReceipt.from_dict(bad)


def test_round_debt_seed_boost_and_receipts_survive_roundtrip_restart():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    prop = proposal("seed", seed="s")
    state, selected = choose(cfg, state, [prop])
    state = account(cfg, state, selected, prop, outcome="valid_comparison")
    restored = S.SchedulerState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()
    assert restored.used_noncoverage == ("seed",)
    assert restored.seed_accounts[0].attempts == 1
    assert restored.receipts[0].receipt_id == "r"


def test_deficit_uses_actual_held_claim_service_not_estimate():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    prop = proposal("actual", claims=vector(fraction=1.0, memory=9000), duration=9)
    state, selected = choose(cfg, state, [prop])
    state = account(cfg, state, selected, prop, end=2, fraction=0.1, memory=100)
    assert state.deficits["cpu"] == pytest.approx(0.2)


def test_dominant_service_normalizes_cpu_claim_against_current_capacity():
    cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=1, capacity=vector(fraction=0.5, memory=10000)))
    state = S.initial_state(cfg, "scheduler")
    prop = proposal("half-capacity", claims=vector(fraction=0.5, memory=0))
    state, selected = choose(cfg, state, [prop])
    state = account(cfg, state, selected, prop, end=2, fraction=0.5, memory=0)
    assert state.deficits["cpu"] == pytest.approx(2.0)


def test_normalized_dominant_service_and_seed_weight_order_noncoverage_only():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.SchedulerState.from_dict(
        S.initial_state(cfg, "scheduler").to_dict() | {"deficits": {"cpu": 5.0, "gpu": 0.0}})
    cpu = proposal("cpu", backend="cpu")
    gpu = proposal("gpu", backend="gpu", claims=vector(fraction=0.5, gpus=("gpu0",)))
    _, selected = choose(cfg, state, [cpu, gpu])
    assert selected.proposal.proposal_id == "gpu"
    production = proposal("production", frontier="prod", backend="cpu")
    _, selected = choose(cfg, state, [cpu, gpu, production])
    assert selected.proposal.proposal_id == "production"


def test_capacity_epoch_preserves_prior_service_attempts_and_boost_history():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    prop = proposal("seed", seed="s")
    state, selected = choose(cfg, state, [prop])
    state = account(cfg, state, selected, prop)
    next_cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=1,
        capacity=vector(fraction=1, gpus=("gpu0", "gpu1"), memory=20000)))
    changed = S.change_capacity(state, next_cfg)
    assert changed.accounting_epoch == 2
    assert changed.capacity.gpu_devices == ("gpu0", "gpu1")
    assert changed.campaign_attempts == state.campaign_attempts
    assert changed.seed_accounts == state.seed_accounts
    assert changed.deficits == state.deficits
    changed, selected = choose(next_cfg, changed, [prop], now=6)
    assert selected.status == "selected" and changed.accounting_epoch == 2
    changed_policy = config(noncoverage_slots=2,
                            capacity=vector(fraction=1, gpus=("gpu0", "gpu1"), memory=20000))
    with pytest.raises(S.SchedulingRefused, match="policy"):
        S.change_capacity(state, S.SchedulerConfig.from_dict(changed_policy))


def test_due_full_region_reservation_waits_and_never_backfills_incompatible_work():
    cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=1, reservation_slots={"full_region": [0]},
        reservation_shares={"full_region": 1.0}))
    state = S.initial_state(cfg, "scheduler")
    ordinary = proposal("ordinary")
    state, selected = choose(cfg, state, [ordinary])
    assert selected.status == "waiting" and "full_region" in selected.reasons[0]
    full = proposal("full", claims=vector(fraction=1), full=True,
                    reservation="full_region")
    _, selected = choose(cfg, state, [ordinary, full])
    assert selected.proposal.proposal_id == "full"
    running = S.SchedulerState.from_dict(state.to_dict() | {"existing_stage_until": 8.0})
    _, wait = choose(cfg, running, [full], now=3)
    assert wait.status == "waiting"
    assert wait.service_bound_seconds == 10 and wait.existing_stage_seconds == 5


def test_oversized_infeasible_budgets_and_outages_are_distinct():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    _, oversized = choose(cfg, state, [proposal("large", duration=11)])
    assert oversized.status == "refused" and "oversized" in oversized.reasons[0]
    _, still_oversized = choose(cfg, state, [proposal("large", duration=11, chunked=True)])
    assert still_oversized.status == "refused"
    _, inactive_large_does_not_veto = choose(
        cfg, state, [proposal("large", duration=11, eligible=False), proposal("ready")])
    assert inactive_large_does_not_veto.status == "selected"
    _, infeasible = choose(cfg, state, [proposal(
        "gpu1", claims=vector(fraction=0.5, gpus=("gpu1",)))])
    assert infeasible.status == "refused" and "infeasible" in infeasible.reasons[0]
    outage = {"schema": S.OUTAGE_SCHEMA, "outage_id": "o", "kind": "authority",
              "started_at": 1.0, "ended_at": None, "reason": "grant absent",
              "backend": None, "frontier_id": None}
    _, waiting = choose(cfg, state, [proposal()], now=4, outages=[outage])
    assert waiting.status == "waiting" and waiting.outage_seconds == 3
    capped = S.SchedulerState.from_dict(
        state.to_dict() | {"campaign_attempts": cfg.campaign_attempt_cap})
    _, exhausted = choose(cfg, capped, [proposal()])
    assert "attempt budget" in exhausted.reasons[0]


def test_outage_freezes_round_and_bound_future_events_refuse_and_duplicates_are_idempotent():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    first = proposal("a", frontier="prod-a")
    outage = {"schema": S.OUTAGE_SCHEMA, "outage_id": "o", "kind": "authority",
              "started_at": 1.0, "ended_at": None, "reason": "authority absent",
              "backend": None, "frontier_id": None}
    state, waiting = choose(cfg, state, [first], now=2, outages=[outage, outage])
    assert state.round_number == 1 and state.frozen_frontier == ("prod-a",)
    assert waiting.service_bound_seconds == 20 and waiting.outage_seconds == 1
    later = proposal("b", frontier="prod-b")
    retained, waiting = choose(cfg, state, [first, later], now=3, outages=[outage])
    assert retained.frozen_frontier == ("prod-a",)
    assert waiting.service_bound_seconds == 20
    future = dict(outage, outage_id="future", started_at=4.0)
    with pytest.raises(S.SchedulingRefused, match="later than"):
        choose(cfg, state, [first], now=3, outages=[future])
    with pytest.raises(S.SchedulingRefused, match="different content"):
        choose(cfg, state, [first], now=3,
               outages=[outage, dict(outage, reason="different")])


def test_scoped_outage_does_not_block_other_backend_and_overlap_is_union_time():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    cpu = proposal("cpu", frontier="cpu-frontier")
    outages = [
        {"schema": S.OUTAGE_SCHEMA, "outage_id": "gpu", "kind": "resource",
         "started_at": 0.0, "ended_at": 10.0, "reason": "gpu unavailable",
         "backend": "gpu", "frontier_id": None},
        {"schema": S.OUTAGE_SCHEMA, "outage_id": "authority", "kind": "authority",
         "started_at": 5.0, "ended_at": 15.0, "reason": "other target authority",
         "backend": "gpu", "frontier_id": None},
    ]
    _, selected = choose(cfg, state, [cpu], now=15, outages=outages)
    assert selected.status == "selected" and selected.proposal.backend == "cpu"
    assert selected.outage_seconds == 15.0


def test_actual_overrun_is_charged_and_fences_successors():
    cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=1, capacity=vector(fraction=0.5, memory=1000)))
    state = S.initial_state(cfg, "scheduler")
    prop = proposal("bounded", claims=vector(fraction=0.5, memory=500), duration=5)
    state, selected = choose(cfg, state, [prop])
    state = account(cfg, state, selected, prop, end=15, fraction=1.0, memory=1500)
    assert state.campaign_attempts == 1 and state.campaign_charged_seconds == 15
    assert len(state.receipts) == 1 and len(state.accounted_receipts) == 1
    assert any("exceeded" in reason for reason in state.successor_fences)
    _, blocked = choose(cfg, state, [prop], now=16)
    assert blocked.status == "refused" and "exceeded" in " ".join(blocked.reasons)


def test_indexed_engine_hot_path_does_not_export_or_rescan_receipt_history(monkeypatch):
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    engine = S.SchedulerEngine(cfg, S.initial_state(cfg, "scheduler"))
    first = proposal("first")
    selected = engine.select_stage([first], now=0)
    engine.account_stage(selected, receipt(first, "r1", claims=("claim-1",)), outcome="invalid")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot path touched serialized receipt history")

    monkeypatch.setattr(S.SchedulerState, "to_dict", forbidden)
    monkeypatch.setattr(S, "_check_receipt_overlaps", forbidden)
    second = proposal("second")
    selected = engine.select_stage([second], now=6)
    engine.account_stage(selected, receipt(
        second, "r2", start=6, end=8, claims=("claim-2",)), outcome="failed")
    assert engine.accounting_view().receipt_count == 2


def test_indexed_seed_hot_path_does_not_scan_historical_accounts():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=2, seed_attempt_cap=1))
    engine = S.SchedulerEngine(cfg, S.initial_state(cfg, "scheduler"))
    old = proposal("old", seed="seed-old", submitted=0)
    later = proposal("later", seed="seed-later", submitted=1)
    selected = engine.select_stage([old, later], now=0)
    assert selected.proposal.seed_id == "seed-old"

    class NoHistoryScan(list):
        def __iter__(self):
            raise AssertionError("hot path scanned historical seed accounts")

    engine.seed_accounts = NoHistoryScan(engine.seed_accounts)
    engine.account_stage(selected, receipt(old, "old-receipt"), outcome="invalid")
    selected = engine.select_stage([later], now=6)
    assert selected.proposal.seed_id == "seed-later"
    engine.account_stage(selected, receipt(
        later, "later-receipt", start=6, end=8, claims=("later-claim",)),
        outcome="invalid")


def test_sustained_scoped_unavailable_frontier_becomes_debt_not_global_stall():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    cpu = proposal("cpu", frontier="cpu", backend="cpu")
    gpu = proposal("gpu", frontier="gpu", backend="gpu",
                   claims=vector(fraction=0.5, gpus=("gpu0",)))
    optional = proposal("optional", backend="cpu")
    outage = {"schema": S.OUTAGE_SCHEMA, "outage_id": "gpu-down", "kind": "resource",
              "started_at": 0.0, "ended_at": None, "reason": "GPU unavailable",
              "backend": "gpu", "frontier_id": "gpu"}
    state, selected = choose(cfg, state, [cpu, gpu, optional], now=0, outages=[outage])
    assert selected.proposal.frontier_id == "cpu"
    state = account(cfg, state, selected, cpu)
    state, selected = choose(cfg, state, [cpu, gpu, optional], now=6, outages=[outage])
    assert selected.proposal.proposal_id == "optional"
    state = account(cfg, state, selected, optional, rid="optional-receipt",
                    start=6, end=8, claims=("optional-claim",))
    next_cpu = proposal("cpu-next", frontier="cpu", backend="cpu", submitted=9)
    state, selected = choose(cfg, state, [next_cpu, gpu], now=9, outages=[outage])
    assert selected.status == "selected" and selected.proposal.frontier_id == "cpu"
    assert "gpu" in state.coverage_debt
    assert "gpu" not in state.used_coverage
    assert "suspending unavailable coverage debt" in selected.reasons[0]


def test_future_full_region_reservation_skips_unverified_backfill():
    cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=2, reservation_slots={"full_region": [1]},
        reservation_shares={"full_region": 0.5}))
    engine = S.SchedulerEngine(cfg, S.initial_state(cfg, "scheduler"))
    labelled = proposal("labelled", reservation=None,
                        alias="ordinary", full=False)
    labelled["compatibility_authority_refs"] = ["actor:claims-compatible"]
    full = proposal("exclusive", reservation="full_region", full=True,
                    claims=vector(fraction=1.0, memory=500))
    selected = engine.select_stage([labelled, full], now=0)
    assert selected.proposal.proposal_id == "exclusive"
    assert selected.slot_kind == "full_region" and selected.slot_index == 1
    assert engine.skipped_noncoverage == [0]


def test_future_full_region_preserves_earlier_reserved_seed():
    cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=2, reservation_slots={"seed": [0], "full_region": [1]},
        reservation_shares={"seed": 0.5, "full_region": 0.5}))
    state = S.initial_state(cfg, "scheduler")
    seed = proposal("seed", seed="seed-one")
    full = proposal("exclusive", reservation="full_region", full=True,
                    claims=vector(fraction=1.0, memory=500))
    state, selected = choose(cfg, state, [seed, full])
    assert selected.proposal.proposal_id == "seed" and selected.slot_index == 0
    assert state.skipped_noncoverage == ()
    state = account(cfg, state, selected, seed)
    state, selected = choose(cfg, state, [seed, full], now=6)
    assert selected.proposal.proposal_id == "exclusive"
    assert selected.slot_kind == "full_region" and selected.slot_index == 1
    assert state.skipped_noncoverage == ()


def test_select_rejects_forged_round_slot_accounting():
    cfg = S.SchedulerConfig.from_dict(config(
        noncoverage_slots=1, reservation_slots={"validation": [0]},
        reservation_shares={"validation": 1.0}))
    base = S.initial_state(cfg, "scheduler").to_dict()
    overused = S.SchedulerState.from_dict(base | {"round_number": 1,
        "used_noncoverage": ["a", "b"], "round_reservations": {"0": "validation"}})
    with pytest.raises(S.SchedulingRefused, match="more than K"):
        choose(cfg, overused, [])
    skipped = S.SchedulerState.from_dict(base | {"round_number": 1,
        "skipped_noncoverage": [9], "round_reservations": {"0": "validation"}})
    with pytest.raises(S.SchedulingRefused, match="outside K"):
        choose(cfg, skipped, [])
    mismatched = S.SchedulerState.from_dict(base | {"round_number": 1,
        "round_reservations": {}})
    with pytest.raises(S.SchedulingRefused, match="reservations"):
        choose(cfg, mismatched, [])


def test_bound_is_n_plus_k_times_d_and_no_provider_or_quality_claim_exists(monkeypatch):
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=2, max_stage_seconds=7))
    state = S.initial_state(cfg, "scheduler")
    rows = [proposal("a", frontier="a"), proposal("b", frontier="b")]
    _, selected = choose(cfg, state, rows)
    assert selected.service_bound_seconds == 28
    assert selected.execution_authorized is False
    fixed = S.adaptive_weights(cfg, state, [{"utilization": 1.0}])
    assert fixed == {"status": "adaptation_not_configured", "weights": {}}
    assert not hasattr(S, "provider") and not hasattr(S, "acquire")


def test_v1_refuses_adaptive_configuration_and_reports_fixed_path():
    with pytest.raises(S.SchedulingRefused, match="fixed weights"):
        S.SchedulerConfig.from_dict(config(adaptive_rule_id="registered:reward-v1"))
    cfg = S.SchedulerConfig.from_dict(config())
    assert S.adaptive_weights(cfg, S.initial_state(cfg, "scheduler"), [{"forged": True}]) == {
        "status": "adaptation_not_configured", "weights": {}}


def test_forged_selection_cannot_bypass_round_or_execution_boundary():
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    state = S.initial_state(cfg, "scheduler")
    prop = proposal("p")
    state, selected = choose(cfg, state, [prop])
    forged = selected.to_dict() | {"execution_authorized": True}
    with pytest.raises(S.SchedulingRefused):
        S.Selection.from_dict(forged)
    stale = S.Selection.from_dict(selected.to_dict() | {"round_number": 99})
    with pytest.raises(S.SchedulingRefused, match="round"):
        S.account_stage(cfg, state, stale, receipt(prop), outcome="invalid")
    relabelled = S.Selection.from_dict(selected.to_dict() | {"slot_kind": "validation"})
    with pytest.raises(S.SchedulingRefused, match="not issued"):
        S.account_stage(cfg, state, relabelled, receipt(prop), outcome="invalid")
