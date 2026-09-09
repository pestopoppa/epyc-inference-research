from __future__ import annotations

import copy

import pytest

from . import discovery_screen as D
from . import experiment_plan as E
from . import unified_planner as U
from .test_experiment_plan import plan_dict, raw_dict
from .test_unified_planner import canonical_recipe, dimension


def _pair(*, kind="threads", anchor=4, candidate=8, backend="cpu"):
    recipe = canonical_recipe(backend=backend,
                              threads=anchor if isinstance(anchor, int) else 4,
                              policy_keys=("OMP_NUM_THREADS",))
    return U.enumerate_runtime_dimensions(
        recipe, [dimension(kind=kind, anchor=anchor, candidate=candidate)])[0]


def _plan(pair, *, epoch="e" * 64):
    raw = plan_dict(phase="discovery", record_class="discovery_screen",
                    intended_use="nominate", n=3, paired=False, unit="process",
                    instrument="serving", category="CANDIDATE", epoch=epoch)
    raw.update(protocol_ref=D.A2_PROTOCOL, protocol_status="ratified",
               metric="aggregate_tok_s", metric_direction="higher",
               changed_factors=[pair.dimension.kind],
               required_witnesses=sorted(D.MANDATORY_WITNESSES),
               anchor_identity=dict(U.serving_arm_identity(pair.anchor)),
               candidate_identity=dict(U.serving_arm_identity(pair.candidate)))
    units = [item for arm in ("anchor", "candidate")
             for item in raw["expected_units"] if item["arm"] == arm]
    for index, unit in enumerate(units):
        unit["order_index"] = index
        unit["process_id"] = f"process-{index}"
    raw["expected_units"] = units
    return E.ExperimentPlan.from_dict(raw)


def _context(plan=None):
    if plan is None:
        plan = _plan(_pair())
    return D.RuntimeFrameContext.from_dict({
        "schema": D.CONTEXT_SCHEMA,
        "evaluator_identity": {"id": "planned-serving", "digest": "1" * 64},
        "runtime_source_identity": {"revision": "a" * 40, "digest": "2" * 64},
        "linkage_identity": {"receipt": "linkage:1", "dso_set": "4" * 64},
        "frequency_power_envelope": {"policy": "fixed", "receipt": "power:1"},
        "resource_claim": {"claim_id": "claim-1", "region": "cpu-quarter"},
        "policy_digest": D._digest(dict(plan.policy_snapshot)),
        "host_epoch": plan.epoch,
    })


class PhaseStore:
    def __init__(self, fail_after=None):
        self.events = []
        self.fail_after = fail_after

    def __call__(self, event):
        self.events.append(copy.deepcopy(event))
        if self.fail_after is not None and len(self.events) == self.fail_after:
            raise OSError("reply lost after durable append")
        return copy.deepcopy(event)


class Invoker:
    def __init__(self, plan, *, overlap=False, values=None):
        self.plan = plan
        self.overlap = overlap
        self.values = iter(values or (10, 11, 12, 13, 14, 15))
        self.calls = []
        self.frame_digest = None

    def invoke(self, unit, recipe):
        self.calls.append((unit.arm, unit.unit_id, recipe.execution_digest))
        witnesses = {name: {"status": "pass", "ref": f"trusted:{name}:{unit.unit_id}"}
                     for name in D.MANDATORY_WITNESSES}
        value = next(self.values)
        raw = E.RawUnit.from_dict(raw_dict(
            self.plan, unit.unit_id, value=value, witnesses=witnesses))
        observations = ({"metric": self.plan.metric, "value": value,
                         "unit_id": unit.unit_id},)
        proof = D.InvocationProof.from_dict({
            "schema": D.PROOF_SCHEMA,
            "producer_identity": {
                "frame_digest": self.frame_digest,
                "recipe_snapshot_digest": recipe.snapshot_digest,
                "recipe_execution_digest": recipe.execution_digest},
            "started_at": float(len(self.calls)), "ended_at": float(len(self.calls)) + .5,
            "witness_refs": {name: f"trusted:{name}:{unit.unit_id}"
                             for name in D.MANDATORY_WITNESSES},
            "overlapping_competing_inference": self.overlap,
            "ordinary_load": {"builds": 7, "load": 99.0},
            "invocation_identity": {"unit_id": unit.unit_id,
                                    "process_id": unit.process_id,
                                    "launch_id": f"launch:{unit.process_id}"},
            "observations_digest": D._digest(list(observations)),
        })
        return D.InvocationResult(raw, proof, observations)


def _screen(pair=None, *, invoker=None, store=None, epoch="e" * 64):
    pair = pair or _pair()
    plan = _plan(pair, epoch=epoch)
    invoker = invoker or Invoker(plan)
    store = store or PhaseStore()
    screen = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=invoker, phase_sink=store)
    invoker.frame_digest = screen.frame_digest
    return screen, invoker, store


def test_exact_three_bank_three_candidate_zero_fresh_anchor_and_nomination_use():
    screen, invoker, store = _screen()
    bank = screen.create_bank()
    assert [arm for arm, *_ in invoker.calls] == ["anchor"] * 3
    receipt = screen.screen(bank)
    assert [arm for arm, *_ in invoker.calls] == ["anchor"] * 3 + ["candidate"] * 3
    assert receipt.candidate_invocations == 3 and receipt.anchor_invocations == 0
    assert receipt.non_promotable is True
    assert len(store.events) == 14  # intent+terminal per call and one seal per phase
    candidate_view = E.admissible_units(
        screen.plan, tuple(item.raw_unit for item in receipt.candidate_results))
    assert not candidate_view.complete
    permitted = E.eligibility(
        screen.plan, candidate_view, "nominate", current_epoch=screen.plan.epoch,
        nomination_verifier=screen.nomination_verifier(),
        nomination_receipt=receipt.to_dict())
    assert permitted.status == "permitted"
    untrusted = E.eligibility(
        screen.plan, candidate_view, "nominate", current_epoch=screen.plan.epoch,
        nomination_receipt=receipt.to_dict())
    assert untrusted.status != "permitted"


def test_gpu_runtime_pair_uses_the_same_generic_a2_cardinality():
    screen, invoker, _ = _screen(pair=_pair(backend="gpu"))
    receipt = screen.screen(screen.create_bank())
    assert screen.pair.anchor.backend == "gpu"
    assert receipt.anchor_invocations == 0
    assert [arm for arm, *_ in invoker.calls] == ["anchor"] * 3 + ["candidate"] * 3


def test_invalid_plan_and_changed_frame_refuse_before_any_invocation():
    pair = _pair()
    plan = _plan(pair)
    invoker = Invoker(plan)
    store = PhaseStore()
    malformed = plan.to_dict()
    malformed["changed_factors"] = ["threads", "batch"]
    with pytest.raises(D.DiscoveryScreenRefused):
        bad_plan = E.ExperimentPlan.from_dict(malformed)
        D.A2RuntimeScreen(bad_plan, pair, _context(bad_plan),
                          invoker=invoker, phase_sink=store)
    assert invoker.calls == [] and store.events == []

    screen, invoker, _ = _screen(pair=pair)
    bank = screen.create_bank()
    changed = _context(screen.plan).to_dict()
    changed["frequency_power_envelope"] = {"policy": "different", "receipt": "power:2"}
    other = D.A2RuntimeScreen(screen.plan, pair, changed,
                              invoker=invoker, phase_sink=PhaseStore())
    with pytest.raises(D.DiscoveryScreenRefused, match="common frame"):
        other.screen(bank)
    assert len(invoker.calls) == 3


@pytest.mark.parametrize("change", ["prompts", "estimator", "witnesses", "policy",
                                     "target"])
def test_bank_common_semantics_change_refuses_before_candidate_invocation(change):
    pair = _pair()
    original, _, _ = _screen(pair=pair)
    bank = original.create_bank()
    raw = original.plan.to_dict()
    if change == "prompts":
        raw["expected_units"][3]["expected_prompt_ids"] = ["different"]
    elif change == "estimator":
        raw["estimator_id"] = "different-estimator"
    elif change == "witnesses":
        raw["required_witnesses"] = sorted((*D.MANDATORY_WITNESSES, "extra"))
    elif change == "policy":
        raw["policy_snapshot"] = {"reference": "different-policy",
                                  "digest": "f" * 64}
    else:
        raw["target_revision"] = "different-target"
    changed = E.ExperimentPlan.from_dict(raw)
    invoker = Invoker(changed, values=(20, 21, 22))
    other = D.A2RuntimeScreen(changed, pair, _context(changed), invoker=invoker,
                              phase_sink=PhaseStore())
    invoker.frame_digest = other.frame_digest
    with pytest.raises(D.DiscoveryScreenRefused, match="common frame"):
        other.screen(bank, bank_verifier=original.bank_verifier())
    assert invoker.calls == []


def test_sealed_bank_reuses_across_candidate_values_with_zero_fresh_anchors():
    first, first_invoker, _ = _screen(pair=_pair(candidate=8))
    bank = first.create_bank()
    second_pair = _pair(candidate=12)
    second_plan = _plan(second_pair)
    second_invoker = Invoker(second_plan, values=(20, 21, 22))
    second = D.A2RuntimeScreen(second_plan, second_pair, _context(second_plan),
                               invoker=second_invoker, phase_sink=PhaseStore())
    second_invoker.frame_digest = second.frame_digest
    receipt = second.screen(bank, bank_verifier=first.bank_verifier())
    assert [arm for arm, *_ in second_invoker.calls] == ["candidate"] * 3
    assert receipt.anchor_invocations == 0
    assert len(first_invoker.calls) == 3


def test_registered_environment_unset_is_one_same_artifact_runtime_factor():
    pair = _pair(kind="env", anchor={"key": "OMP_NUM_THREADS", "value": "4"},
                 candidate={"key": "OMP_NUM_THREADS", "value": None})
    assert pair.anchor.executable == pair.candidate.executable
    assert pair.anchor.dsos == pair.candidate.dsos
    assert dict(pair.anchor.relevant_environment)["OMP_NUM_THREADS"] == "4"
    assert dict(pair.candidate.relevant_environment)["OMP_NUM_THREADS"] is None
    plan = _plan(pair)
    invoker = Invoker(plan)
    screen = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=invoker,
                               phase_sink=PhaseStore())
    assert screen.frame["factor"]["field"] == "env"


def test_ordinary_load_is_retained_but_overlapping_inference_blocks():
    screen, invoker, _ = _screen()
    bank = screen.create_bank()
    receipt = screen.screen(bank)
    assert all(item.proof.ordinary_load["builds"] == 7
               for item in receipt.candidate_results)
    blocked_invoker = Invoker(screen.plan, overlap=True)
    blocked = D.A2RuntimeScreen(screen.plan, screen.pair, _context(screen.plan),
                                invoker=blocked_invoker, phase_sink=PhaseStore())
    blocked_invoker.frame_digest = blocked.frame_digest
    with pytest.raises(D.DiscoveryScreenRefused, match="overlaps"):
        blocked.screen(bank, bank_verifier=screen.bank_verifier())
    assert len(blocked_invoker.calls) == 1


def test_flagged_ordinary_load_remains_admissible_but_reused_launch_does_not():
    pair = _pair()
    plan = _plan(pair)

    class Flagged(Invoker):
        def invoke(self, unit, recipe):
            result = super().invoke(unit, recipe)
            raw = result.raw_unit.to_dict()
            raw.update(recorded_screen="flagged_but_retained", reason="ordinary build load")
            return D.InvocationResult(E.RawUnit.from_dict(raw), result.proof,
                                      result.observations)

    flagged = Flagged(plan)
    screen = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=flagged,
                               phase_sink=PhaseStore())
    flagged.frame_digest = screen.frame_digest
    assert len(screen.create_bank().anchor_results) == 3

    class Reused(Invoker):
        def invoke(self, unit, recipe):
            result = super().invoke(unit, recipe)
            proof = result.proof.to_dict()
            proof["invocation_identity"]["launch_id"] = "one-shared-launch"
            return D.InvocationResult(result.raw_unit, D.InvocationProof.from_dict(proof),
                                      result.observations)

    reused = Reused(plan)
    refused = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=reused,
                                phase_sink=PhaseStore())
    reused.frame_digest = refused.frame_digest
    with pytest.raises(D.DiscoveryScreenRefused, match="independent invocation"):
        refused.create_bank()


def test_crash_replay_reuses_terminal_and_pending_intent_requires_reconciliation():
    pair = _pair()
    plan = _plan(pair)
    invoker = Invoker(plan)
    store = PhaseStore(fail_after=2)  # terminal append durable, reply lost
    screen = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=invoker, phase_sink=store)
    invoker.frame_digest = screen.frame_digest
    with pytest.raises(OSError):
        screen.create_bank()
    assert len(invoker.calls) == 1
    with pytest.raises(D.DiscoveryScreenRefused, match="outcome is uncertain"):
        screen.create_bank()
    resumed_store = PhaseStore()
    resumed = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=invoker,
                                phase_sink=resumed_store, events=store.events)
    invoker.frame_digest = resumed.frame_digest
    bank = resumed.create_bank()
    assert len(bank.anchor_results) == 3 and len(invoker.calls) == 3

    pending_store = PhaseStore(fail_after=1)
    pending_invoker = Invoker(plan)
    pending = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=pending_invoker,
                                phase_sink=pending_store)
    pending_invoker.frame_digest = pending.frame_digest
    with pytest.raises(OSError):
        pending.create_bank()
    replay = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=pending_invoker,
                               phase_sink=PhaseStore(), events=pending_store.events)
    with pytest.raises(D.DiscoveryScreenRefused, match="reconciliation"):
        replay.create_bank()
    assert pending_invoker.calls == []


def test_crash_after_bank_or_screen_seal_replays_without_extra_invocations():
    pair = _pair()
    plan = _plan(pair)
    bank_invoker = Invoker(plan)
    bank_store = PhaseStore(fail_after=7)
    bank_run = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=bank_invoker,
                                 phase_sink=bank_store)
    bank_invoker.frame_digest = bank_run.frame_digest
    with pytest.raises(OSError):
        bank_run.create_bank()
    replay = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=bank_invoker,
                               phase_sink=PhaseStore(), events=bank_store.events)
    bank_invoker.frame_digest = replay.frame_digest
    assert len(replay.create_bank().anchor_results) == 3
    assert len(bank_invoker.calls) == 3

    candidate_invoker = Invoker(plan, values=(20, 21, 22))
    screen_store = PhaseStore(fail_after=7)
    clean_invoker, clean_store = Invoker(plan), PhaseStore()
    clean = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=clean_invoker,
                              phase_sink=clean_store)
    clean_invoker.frame_digest = clean.frame_digest
    trusted_bank = clean.create_bank()
    candidate = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=candidate_invoker,
                                  phase_sink=screen_store)
    candidate_invoker.frame_digest = candidate.frame_digest
    with pytest.raises(OSError):
        candidate.screen(trusted_bank, bank_verifier=clean.bank_verifier())
    resumed = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=candidate_invoker,
                                phase_sink=PhaseStore(), events=screen_store.events)
    candidate_invoker.frame_digest = resumed.frame_digest
    with pytest.raises(D.DiscoveryScreenRefused, match="trusted sealed"):
        resumed.screen(trusted_bank)
    # Serialized replay alone cannot mint live bank or nomination authority.
    assert resumed.nomination_verifier()._receipts == {}
    assert len(candidate_invoker.calls) == 3


def test_serialized_replay_is_inspectable_but_authority_requires_live_attestation():
    original, invoker, store = _screen()
    bank = original.create_bank()
    receipt = original.screen(bank)
    verifier = original.phase_verifier()

    untrusted_invoker = Invoker(original.plan)
    untrusted = D.A2RuntimeScreen(
        original.plan, original.pair, _context(original.plan), invoker=untrusted_invoker,
        phase_sink=PhaseStore(), events=store.events)
    assert untrusted.create_bank().bank_digest == bank.bank_digest
    assert untrusted.screen(bank, bank_verifier=original.bank_verifier()).receipt_digest \
        == receipt.receipt_digest
    assert untrusted.nomination_verifier()._receipts == {}

    trusted_invoker = Invoker(original.plan)
    trusted = D.A2RuntimeScreen(
        original.plan, original.pair, _context(original.plan), invoker=trusted_invoker,
        phase_sink=PhaseStore(), events=store.events, replay_verifier=verifier)
    assert trusted.create_bank().bank_digest == bank.bank_digest
    assert trusted.screen(bank).receipt_digest == receipt.receipt_digest
    assert trusted.nomination_verifier()._receipts
    assert trusted_invoker.calls == [] and untrusted_invoker.calls == []


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), True])
def test_zero_nonfinite_and_bool_samples_never_complete(value):
    pair = _pair()
    plan = _plan(pair)
    invoker = Invoker(plan, values=(value,))
    screen = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=invoker,
                               phase_sink=PhaseStore())
    invoker.frame_digest = screen.frame_digest
    with pytest.raises((D.DiscoveryScreenRefused, E.PlanValidationError)):
        screen.create_bank()


def test_a3_same_epoch_orders_by_direction_and_cross_epoch_hides_magnitude():
    screen, _, _ = _screen()
    bank = screen.create_bank()
    first = screen.screen(bank)
    stale_screen, _, _ = _screen(epoch="f" * 64)
    stale = stale_screen.screen(stale_screen.create_bank())
    assert stale.frame_digest != first.frame_digest
    rows = D.advisory_history([stale, first], current_epoch=screen.plan.epoch,
                              metric_direction="higher")
    assert rows[0]["receipt_digest"] == first.receipt_digest
    assert rows[1]["stale_epoch"] is True
    assert rows[1]["advisory_median"] is None

    changed = first.to_dict()
    changed["advisory_median"] = first.advisory_median + 0.01
    changed["receipt_digest"] = D._digest({key: value for key, value in changed.items()
                                            if key != "receipt_digest"})
    with pytest.raises(D.DiscoveryScreenRefused, match="differs from candidate"):
        D.ScreenReceipt.from_dict(changed)
    other_screen, _, _ = _screen(pair=_pair(backend="gpu"))
    other_current = other_screen.screen(other_screen.create_bank())
    assert other_current.frame_digest != first.frame_digest
    with pytest.raises(D.DiscoveryScreenRefused, match="incomparable"):
        D.advisory_history([first, other_current], current_epoch=screen.plan.epoch,
                           metric_direction="higher")


@pytest.mark.parametrize("candidate_invocations,anchor_invocations", [(3.0, 0), (3, False)])
def test_receipt_counts_are_strict_integers(candidate_invocations, anchor_invocations):
    screen, _, _ = _screen()
    receipt = screen.screen(screen.create_bank()).to_dict()
    receipt.update(candidate_invocations=candidate_invocations,
                   anchor_invocations=anchor_invocations)
    receipt["receipt_digest"] = D._digest({key: value for key, value in receipt.items()
                                            if key != "receipt_digest"})
    with pytest.raises(D.DiscoveryScreenRefused, match="cardinality"):
        D.ScreenReceipt.from_dict(receipt)


def test_completed_overlap_is_terminally_recorded_and_never_retried():
    pair = _pair()
    plan = _plan(pair)
    invoker, store = Invoker(plan, overlap=True), PhaseStore()
    screen = D.A2RuntimeScreen(plan, pair, _context(plan), invoker=invoker,
                               phase_sink=store)
    invoker.frame_digest = screen.frame_digest
    with pytest.raises(D.DiscoveryScreenRefused, match="non-admissible"):
        screen.create_bank()
    assert [row["state"] for row in store.events] == ["INTENT", "TERMINAL"]
    with pytest.raises(D.DiscoveryScreenRefused, match="cannot be rerun"):
        screen.create_bank()
    assert len(invoker.calls) == 1


def test_mutated_recipe_dso_and_unregistered_verifier_are_refused():
    screen, _, _ = _screen()
    bank = screen.create_bank()
    tampered = bank.to_dict()
    tampered["frame"]["anchor_recipe"]["dsos"][0]["sha256"] = "f" * 64
    tampered["bank_digest"] = D._digest({key: tampered[key] for key in
                                         ("schema", "frame", "anchor_results")})
    with pytest.raises(D.DiscoveryScreenRefused):
        screen.screen(tampered)
    with pytest.raises(D.DiscoveryScreenRefused):
        D.RegisteredNominationVerifier({}, object())
