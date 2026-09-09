from dataclasses import FrozenInstanceError, replace
import copy
import math
import subprocess
import sys
import types

import pytest

from autokernel.loop import experiment_plan as E


H = "a" * 64
P = "b" * 64


def plan_dict(*, phase="confirmation", record_class="strict_search",
              intended_use="rank", n=2, paired=True, unit="session",
              instrument="bench", category="CANDIDATE", epoch="epoch-1",
              metric_direction="higher", calibration_ref=None):
    units = []
    for pair in range(n):
        for arm in ("anchor", "candidate"):
            index = len(units)
            units.append({"unit_id": f"{arm}-{pair}", "arm": arm,
                          "process_id": f"proc-{index}",
                          "expected_prompt_ids": ["p1", "p2"],
                          "order_index": index,
                          "pair_id": pair if paired else None})
    return {
        "schema": E.PLAN_SCHEMA, "plan_id": "plan-1", "campaign_id": "camp-1",
        "target_revision": "revision-1", "epoch": epoch,
        "instrument_class": instrument, "category": category, "phase": phase,
        "protocol_ref": "P-AK-SEARCH-1-A2" if phase == "discovery" else "P-test",
        "protocol_status": "ratified", "record_class": record_class,
        "intended_use": intended_use, "comparison_kind": "mechanism",
        "estimand": "level", "metric": "tokens_per_second",
        "metric_direction": metric_direction, "estimator_id": "median.v1",
        "unit": unit, "changed_factors": ["factor-a"],
        "anchor_identity": {"source": "anchor", "recipe": [1, 2]},
        "candidate_identity": {"source": "candidate", "recipe": [1, 3]},
        "expected_units": units,
        "stopping": {"kind": "fixed_n", "n_per_arm": n, "paired": paired},
        "required_witnesses": ["identity"],
        "calibration_ref": calibration_ref,
        "policy_snapshot": {"reference": "MEASUREMENT.md@FLOOR-UNIT-1",
                            "digest": P},
        "continuation_allowed": False,
    }


def raw_dict(plan, unit_id, *, terminal=True, prompt_ids=None, screen="clean",
             reason=None, witnesses=None, value=10.0):
    spec = next(unit for unit in plan.expected_units if unit.unit_id == unit_id)
    return {"schema": E.UNIT_SCHEMA, "plan_digest": plan.digest,
            "unit_id": unit_id, "arm": spec.arm, "process_id": spec.process_id,
            "prompt_ids": list(prompt_ids or spec.expected_prompt_ids),
            "terminal": terminal, "value": value,
            "witnesses": witnesses or {"identity": {"status": "pass", "ref": "artifact:id"}},
            "recorded_screen": screen, "reason": reason,
            "artifact_digest": H, "observed_order_index": spec.order_index}


def complete_rows(plan):
    return tuple(E.RawUnit.from_dict(raw_dict(plan, unit.unit_id))
                 for unit in plan.expected_units)


def discovery_plan(*, intended_use="explore"):
    obj = plan_dict(phase="discovery", record_class="discovery_screen",
                    intended_use=intended_use, n=3, paired=False)
    obj["required_witnesses"] = ["identity"]
    return E.ExperimentPlan.from_dict(obj)


def receipt_dict(*, unit="session", n=24, lower=1.0, upper=2.0,
                 confidence=.95):
    return {"schema": E.CALIBRATION_SCHEMA,
            "unit": unit, "harness": "harness.v1", "n": n,
            "interval": {"lower": lower, "upper": upper,
                         "confidence": confidence, "method_ref": "bootstrap.v1"},
            "estimator_id": "median.v1", "metric": "tokens_per_second",
            "value": 1.5, "anchor_identity": {"source": "anchor"},
            "candidate_identity": {"source": "candidate"},
            "raw_sample_digest": H,
            "unit_ids": [f"cal-{i}" for i in range(n)],
            "contention_model": {"neighbors": "none"},
            "host_state": {"host": "test"},
            "policy_ref": {"reference": "MEASUREMENT.md@FLOOR-UNIT-1",
                           "digest": P}}


def test_plan_round_trip_digest_and_immutable_inputs():
    source = plan_dict()
    plan = E.ExperimentPlan.from_dict(source)
    digest = plan.digest
    source["anchor_identity"]["recipe"].append(999)
    assert plan.digest == digest
    assert E.ExperimentPlan.from_dict(plan.to_dict()) == plan
    assert list(plan.anchor_identity["recipe"]) == [1, 2]
    with pytest.raises(TypeError):
        plan.anchor_identity["source"] = "changed"
    with pytest.raises(FrozenInstanceError):
        plan.epoch = "later"


def _v2_plan_dict():
    source = plan_dict(instrument="serving")
    instrument = {"schema": "epyc.autokernel.loaded_serving_instrument_reference.v1",
                  "identity_sha256": "c" * 64, "configuration_complete": False,
                  "artifact": {"locator": "instrument.json", "sha256": "d" * 64,
                               "verified": True}}
    from autokernel import schemas
    instrument["reference_digest"] = schemas.content_hash(instrument)
    source["schema"] = E.PLAN_SCHEMA_V2
    source["loaded_instrument"] = instrument
    for arm in ("anchor_identity", "candidate_identity"):
        source[arm] |= {"schema": "epyc.autokernel.serving_arm_identity.v2",
                        "instrument_identity_sha256": "c" * 64,
                        "instrument_configuration_complete": False}
    return source


def test_v2_plan_round_trip_and_v1_reader_refuses_before_input_mutation():
    source = _v2_plan_dict()
    plan = E.ExperimentPlan.from_dict(source)
    assert plan.schema == E.PLAN_SCHEMA_V2
    assert plan.to_dict() == source

    old_source = subprocess.run([
        "git", "show",
        "d75bc9ec129be0fff8dfb0e7b476d4ce86cb4455:scripts/kernel_rnd/autokernel/loop/experiment_plan.py"
    ], check=True, capture_output=True, text=True).stdout
    old = types.ModuleType("autokernel.loop._accepted_v1_experiment_plan")
    old.__package__ = "autokernel.loop"
    sys.modules[old.__name__] = old
    try:
        exec(compile(old_source, "accepted-v1-experiment-plan.py", "exec"), old.__dict__)
    finally:
        sys.modules.pop(old.__name__, None)
    untouched = copy.deepcopy(source)
    with pytest.raises(old.PlanValidationError):
        old.ExperimentPlan.from_dict(source)
    assert source == untouched
    legacy = plan_dict()
    assert old.ExperimentPlan.from_dict(legacy).to_dict() == \
        E.ExperimentPlan.from_dict(legacy).to_dict()
    assert old.ExperimentPlan.from_dict(legacy).digest == \
        E.ExperimentPlan.from_dict(legacy).digest


def test_v2_cannot_drop_or_relabel_loaded_identity():
    source = _v2_plan_dict()
    del source["loaded_instrument"]
    with pytest.raises(E.PlanValidationError):
        E.ExperimentPlan.from_dict(source)
    continued_v1 = plan_dict()
    continued_v1["loaded_instrument"] = _v2_plan_dict()["loaded_instrument"]
    with pytest.raises(E.PlanValidationError):
        E.ExperimentPlan.from_dict(continued_v1)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(extra=True),
    lambda p: p.update(schema="unknown.v9"),
    lambda p: p.update(continuation_allowed=1),
    lambda p: p["stopping"].update(n_per_arm=True),
    lambda p: p["expected_units"][0].update(order_index=True),
    lambda p: p.update(changed_factors=["x", "x"]),
    lambda p: p.update(anchor_identity={"bad": math.inf}),
])
def test_strict_schema_duplicate_and_bool_int_rejections(mutation):
    obj = plan_dict()
    mutation(obj)
    with pytest.raises(E.PlanValidationError):
        E.ExperimentPlan.from_dict(obj)


def test_unsupported_stopping_rule_is_explicit():
    obj = plan_dict()
    obj["stopping"]["kind"] = "actor_selected"
    with pytest.raises(E.UnsupportedStoppingRule, match="unsupported"):
        E.ExperimentPlan.from_dict(obj)


def test_confirmation_pair_slots_and_distinct_arms_are_frozen():
    gap = plan_dict()
    gap["expected_units"][2]["pair_id"] = 3
    gap["expected_units"][3]["pair_id"] = 3
    with pytest.raises(E.PlanValidationError, match="contiguous"):
        E.ExperimentPlan.from_dict(gap)
    same_arm = plan_dict()
    same_arm["expected_units"][1]["arm"] = "anchor"
    with pytest.raises(E.PlanValidationError):
        E.ExperimentPlan.from_dict(same_arm)

    grouped = plan_dict()
    for index, unit in enumerate(grouped["expected_units"]):
        unit["order_index"] = (index // 2) + (0 if unit["arm"] == "anchor" else 2)
    with pytest.raises(E.PlanValidationError, match="adjacent declared order"):
        E.ExperimentPlan.from_dict(grouped)


def test_pair_shape_applies_outside_confirmation_and_unpaired_forbids_pair_ids():
    release = plan_dict(phase="release", record_class="registered_claim", n=1)
    release["expected_units"][1]["pair_id"] = None
    with pytest.raises(E.PlanValidationError, match="every unit requires pair_id"):
        E.ExperimentPlan.from_dict(release)
    observation = plan_dict(phase="observation", record_class="observation",
                            n=1, paired=False)
    observation["expected_units"][0]["pair_id"] = 0
    with pytest.raises(E.PlanValidationError, match="every pair_id must be null"):
        E.ExperimentPlan.from_dict(observation)


@pytest.mark.parametrize("phase,record_class", [
    ("release", "observation"),
    ("observation", "registered_claim"),
    ("release", "strict_search"),
    ("confirmation", "discovery_screen"),
])
def test_record_class_phase_laundering_is_rejected(phase, record_class):
    obj = plan_dict(phase=phase, record_class=record_class, n=1)
    with pytest.raises(E.PlanValidationError, match="record_class/phase mismatch"):
        E.ExperimentPlan.from_dict(obj)


def test_process_unit_rejects_pseudo_replication():
    obj = plan_dict(unit="process")
    obj["expected_units"][1]["process_id"] = obj["expected_units"][0]["process_id"]
    with pytest.raises(E.PlanValidationError, match="reuses process_id"):
        E.ExperimentPlan.from_dict(obj)


def test_zero_rows_and_dropped_unit_are_incomplete_not_vacuous():
    plan = E.ExperimentPlan.from_dict(plan_dict())
    zero = E.admissible_units(plan, ())
    assert not zero.complete and not zero.selected_rows
    assert set(zero.missing_expected_units) == {u.unit_id for u in plan.expected_units}
    dropped = E.admissible_units(plan, complete_rows(plan)[:-1])
    assert not dropped.complete
    assert dropped.independent_n == {"anchor": 1, "candidate": 1}


def test_partial_or_invalid_pair_rejects_both_arms():
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    only_anchor = E.RawUnit.from_dict(raw_dict(plan, "anchor-0"))
    view = E.admissible_units(plan, (only_anchor,))
    assert not view.selected_rows
    assert "anchor-0" in view.rejection_reasons
    assert "candidate-0" in view.rejection_reasons


def test_duplicate_raw_ids_are_rejected_not_counted_twice():
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    anchor = E.RawUnit.from_dict(raw_dict(plan, "anchor-0"))
    candidate = E.RawUnit.from_dict(raw_dict(plan, "candidate-0"))
    view = E.admissible_units(plan, (anchor, anchor, candidate))
    assert not view.complete and not view.selected_rows
    assert "duplicate raw unit_id" in view.rejection_reasons["anchor-0"]


@pytest.mark.parametrize("field,value,reason", [
    ("plan_digest", "c" * 64, "wrong plan digest"),
    ("arm", "candidate", "wrong arm"),
    ("process_id", "other", "wrong process_id"),
    ("observed_order_index", 1, "wrong observed order"),
    ("terminal", False, "not terminal"),
])
def test_unit_binding_mismatches_rejected_with_exact_reason(field, value, reason):
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    obj = raw_dict(plan, "anchor-0")
    obj[field] = value
    bad = E.RawUnit.from_dict(obj)
    other = E.RawUnit.from_dict(raw_dict(plan, "candidate-0"))
    view = E.admissible_units(plan, (bad, other))
    assert reason in " ".join(view.rejection_reasons["anchor-0"])


@pytest.mark.parametrize("witnesses,reason", [
    ({}, "missing required witness identity"),
    ({"identity": {"status": "unknown", "ref": None}}, "not passed"),
    ({"identity": {"status": "fail", "ref": "artifact:failure"}}, "not passed"),
])
def test_required_witness_missing_unknown_or_failed(witnesses, reason):
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    obj = raw_dict(plan, "anchor-0")
    obj["witnesses"] = witnesses
    bad = E.RawUnit.from_dict(obj)
    view = E.admissible_units(
        plan, (bad, E.RawUnit.from_dict(raw_dict(plan, "candidate-0"))))
    assert reason in " ".join(view.rejection_reasons["anchor-0"])


@pytest.mark.parametrize("prompt_ids,reason", [
    (["p1"], "missing expected prompts"),
    (["p1", "p2", "p3"], "extra prompts"),
    (["p1", "p1"], "duplicate prompt IDs"),
])
def test_prompt_membership_and_count_rejected(prompt_ids, reason):
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    bad = E.RawUnit.from_dict(raw_dict(plan, "anchor-0", prompt_ids=prompt_ids))
    good = E.RawUnit.from_dict(raw_dict(plan, "candidate-0"))
    view = E.admissible_units(plan, (bad, good))
    assert any(reason in item for item in view.rejection_reasons["anchor-0"])
    assert not view.selected_rows


def test_flagged_retained_is_preserved_and_counts_as_unit():
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    rows = list(complete_rows(plan))
    rows[0] = E.RawUnit.from_dict(raw_dict(
        plan, "anchor-0", screen="flagged_but_retained", reason="predeclared flag"))
    view = E.admissible_units(plan, rows)
    assert view.complete
    assert view.selected_rows[0].recorded_screen == "flagged_but_retained"
    assert view.independent_n == {"anchor": 1, "candidate": 1}


def test_lower_better_is_retained_without_sign_reinterpretation():
    plan = E.ExperimentPlan.from_dict(plan_dict(metric_direction="lower", n=1))
    view = E.admissible_units(plan, complete_rows(plan))
    assert plan.metric_direction == "lower"
    assert [row.value for row in view.selected_rows] == [10.0, 10.0]


def test_direct_constructor_objects_are_revalidated_and_frozen_at_boundary():
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    with pytest.raises(E.PlanValidationError, match="boolean"):
        E.admissible_units(replace(plan, continuation_allowed=1), ())
    raw = E.RawUnit.from_dict(raw_dict(plan, "anchor-0"))
    with pytest.raises(E.PlanValidationError, match="finite"):
        E.admissible_units(plan, (replace(raw, value=True),))

    mutable_identity = {"source": "anchor", "nested": []}
    direct = replace(plan, anchor_identity=mutable_identity)
    view = E.admissible_units(direct, ())
    frozen_digest = view.plan_digest
    mutable_identity["nested"].append("later")
    assert view.plan_digest == frozen_digest
    assert view.plan_digest != E.ExperimentPlan.from_dict(direct.to_dict()).digest


def test_forged_or_foreign_view_cannot_bypass_eligibility():
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    incomplete = E.admissible_units(plan, ())
    forged = replace(incomplete, complete=True)
    result = E.eligibility(plan, forged, "rank", current_epoch=plan.epoch)
    assert result.status == "refused" and "invalid structural evidence" in result.reasons[0]

    complete = E.admissible_units(plan, complete_rows(plan))
    foreign_plan = E.ExperimentPlan.from_dict(
        plan_dict(n=1, epoch="other-epoch"))
    result = E.eligibility(foreign_plan, complete, "rank",
                           current_epoch=foreign_plan.epoch)
    assert result.status == "refused" and "different plan" in result.reasons[0]
    tampered = replace(complete, selected_rows=complete.selected_rows[:-1])
    assert E.eligibility(plan, tampered, "rank",
                         current_epoch=plan.epoch).status == "refused"


def test_nonfinite_and_malformed_raw_are_rejected():
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    for value in (math.nan, math.inf, True):
        with pytest.raises(E.PlanValidationError, match="finite"):
            E.RawUnit.from_dict(raw_dict(plan, "anchor-0", value=value))


def test_discovery_explore_dry_plan_but_nomination_requires_complete_runtime_record():
    explore = discovery_plan()
    empty = E.admissible_units(explore, ())
    assert E.eligibility(explore, empty, "explore", current_epoch=explore.epoch).status == "permitted"
    nominate_obj = explore.to_dict()
    nominate_obj["intended_use"] = "nominate"
    nominate = E.ExperimentPlan.from_dict(nominate_obj)
    empty = E.admissible_units(nominate, ())
    result = E.eligibility(nominate, empty, "nominate", current_epoch=nominate.epoch)
    assert result.status == "refused"
    assert "complete" in " ".join(result.reasons)

    full = E.admissible_units(nominate, tuple(E.RawUnit.from_dict(raw_dict(
        nominate, unit.unit_id,
        witnesses={name: {"status": "pass", "ref": f"artifact:{name}"}
                   for name in nominate.required_witnesses}))
        for unit in nominate.expected_units))
    result = E.eligibility(nominate, full, "nominate", current_epoch=nominate.epoch)
    assert result.status == "policy_undefined"
    assert "zero new anchor launches" in " ".join(result.reasons)


@pytest.mark.parametrize("use", ["rank", "bank", "validate", "certify", "headline", "release"])
def test_discovery_authority_denials(use):
    obj = discovery_plan().to_dict()
    obj["intended_use"] = use
    plan = E.ExperimentPlan.from_dict(obj)
    view = E.admissible_units(plan, complete_rows(plan))
    result = E.eligibility(plan, view, use, current_epoch=plan.epoch)
    assert result.status == "refused"


def test_discovery_protocol_label_cannot_self_authorize():
    obj = discovery_plan().to_dict()
    obj["protocol_status"] = "unratified"
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, ()),
                           "explore", current_epoch=plan.epoch)
    assert result.status == "refused"
    joined = " ".join(result.reasons)
    assert "ratified exact" in joined


def test_cross_epoch_search_magnitude_cannot_rank():
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1))
    view = E.admissible_units(plan, complete_rows(plan))
    result = E.eligibility(plan, view, "rank", current_epoch="epoch-2")
    assert result.status == "refused"
    assert "cross-epoch" in " ".join(result.reasons)


@pytest.mark.parametrize("instrument,category", [("bench", "CANDIDATE"),
                                                  ("serving", "BASELINE")])
def test_bench_or_baseline_cannot_headline(instrument, category):
    obj = plan_dict(intended_use="headline", instrument=instrument, category=category,
                    n=1, calibration_ref="cal-1")
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, complete_rows(plan)),
                           "headline", current_epoch=plan.epoch)
    assert result.status == "refused"


def test_observation_cannot_become_claim():
    obj = plan_dict(phase="observation", record_class="observation",
                    intended_use="validate", n=1, paired=False)
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, complete_rows(plan)),
                           "validate", current_epoch=plan.epoch)
    assert result.status == "refused"


def test_serving_optimum_label_does_not_grant_observation_headline():
    obj = plan_dict(phase="observation", record_class="observation",
                    intended_use="headline", n=1, paired=False,
                    instrument="serving", category="OPTIMUM")
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, complete_rows(plan)),
                           "headline", current_epoch=plan.epoch,
                           registered_claim_grade="PASS")
    assert result.status == "refused"
    assert "observation" in " ".join(result.reasons)


@pytest.mark.parametrize("use", ["headline", "release"])
def test_strict_search_never_headlines_or_releases(use):
    obj = plan_dict(intended_use=use, n=1, calibration_ref="cal-ref")
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, complete_rows(plan)),
                           use, current_epoch=plan.epoch,
                           registered_claim_grade={"grade": "PASS"})
    assert result.status == "refused"
    assert "strict_search" in " ".join(result.reasons)


@pytest.mark.parametrize("use", [
    "validate_production", "certify_transfer", "certify_overlap",
])
def test_baseline_cannot_validate_production_or_certify_transfer(use):
    obj = plan_dict(intended_use=use, n=1, category="BASELINE",
                    instrument="serving", calibration_ref="cal-ref")
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, complete_rows(plan)),
                           use, current_epoch=plan.epoch)
    assert result.status == "refused"
    assert "BASELINE" in " ".join(result.reasons)


def test_actor_grade_is_not_trusted_claimtuple_authority():
    obj = plan_dict(intended_use="validate", n=1, calibration_ref="cal-1")
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, complete_rows(plan)),
                           "validate", current_epoch=plan.epoch,
                           registered_claim_grade={"grade": "PASS"})
    assert result.status == "policy_undefined"
    assert "grader adapter" in " ".join(result.reasons)


def test_strict_claim_missing_calibration_is_refused():
    obj = plan_dict(intended_use="validate", n=1, calibration_ref=None)
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, complete_rows(plan)),
                           "validate", current_epoch=plan.epoch)
    assert result.status == "refused"
    assert "calibration" in " ".join(result.reasons)


def test_registered_claim_still_needs_shared_grader_adapter():
    obj = plan_dict(phase="release", record_class="registered_claim",
                    intended_use="headline", n=1, paired=True,
                    instrument="serving", category="OPTIMUM", calibration_ref="cal-1")
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.eligibility(plan, E.admissible_units(plan, complete_rows(plan)),
                           "headline", current_epoch=plan.epoch,
                           registered_claim_grade="PASS")
    assert result.status == "policy_undefined"


def test_calibration_receipt_round_trip_and_minimum_n():
    receipt = E.CalibrationReceipt.from_dict(receipt_dict())
    assert E.CalibrationReceipt.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(E.PlanValidationError):
        E.CalibrationReceipt.from_dict(receipt_dict(n=23))
    mismatch = receipt_dict()
    mismatch["unit_ids"] = mismatch["unit_ids"][:-1]
    with pytest.raises(E.PlanValidationError, match="n mismatch"):
        E.CalibrationReceipt.from_dict(mismatch)


@pytest.mark.parametrize("change", [
    lambda r: r.update(unit="prompt"),
    lambda r: r["interval"].update(lower=3.0, upper=2.0),
    lambda r: r["interval"].update(confidence=math.nan),
    lambda r: r.update(unit_ids=["same"] * 24),
])
def test_calibration_wrong_unit_interval_and_duplicate_ids(change):
    obj = receipt_dict()
    change(obj)
    with pytest.raises(E.PlanValidationError):
        E.CalibrationReceipt.from_dict(obj)


def test_calibration_requires_registered_estimator_and_transfer_rule():
    receipt = E.CalibrationReceipt.from_dict(receipt_dict())
    plan = E.ExperimentPlan.from_dict(plan_dict(n=1, calibration_ref="cal-1"))
    missing_rule = E.calibration_applicability(
        receipt, plan, registered_estimators={"median.v1": lambda _: 1.5},
        registered_rule_id=None, applicability_rule=lambda _r, _p: True)
    assert missing_rule.status == "policy_undefined"
    missing_estimator = E.calibration_applicability(
        receipt, plan, registered_estimators={}, registered_rule_id="rule.v1",
        applicability_rule=lambda _r, _p: True)
    assert missing_estimator.status == "policy_undefined"


def test_calibration_wrong_plan_unit_and_replay_mismatch_require_recalibration():
    receipt = E.CalibrationReceipt.from_dict(receipt_dict())
    obj = plan_dict(n=1, calibration_ref=receipt.digest, unit="arm")
    plan = E.ExperimentPlan.from_dict(obj)
    result = E.calibration_applicability(
        receipt, plan, registered_estimators={"median.v1": lambda _: 9.0},
        registered_rule_id="rule.v1", applicability_rule=lambda _r, _p: True)
    assert result.status == "recalibration_required"
    joined = " ".join(result.reasons)
    assert "unit differs" in joined and "replay" in joined


def test_calibration_cache_and_local_dependency_invalidation():
    receipt = E.CalibrationReceipt.from_dict(receipt_dict())
    obj = plan_dict(n=1, calibration_ref=receipt.digest)
    plan = E.ExperimentPlan.from_dict(obj)
    calls = []
    rule_calls = []

    def estimator(_receipt):
        calls.append("replay")
        return 1.5

    cache = E.CalibrationCache()
    kwargs = {"registered_estimators": {"median.v1": estimator},
              "registered_rule_id": "rule.v1",
              "applicability_rule": lambda _r, _p: rule_calls.append("rule") or True,
              "cache": cache}
    first = E.calibration_applicability(receipt, plan, **kwargs)
    second = E.calibration_applicability(receipt, plan, **kwargs)
    assert first.status == "applicable" and not first.cache_hit
    assert second.status == "applicable" and second.cache_hit
    assert calls == ["replay"]
    changed = plan.to_dict()
    changed["target_revision"] = "revision-2"
    third = E.calibration_applicability(
        receipt, E.ExperimentPlan.from_dict(changed), **kwargs)
    assert not third.cache_hit and third.replay_cache_hit
    assert calls == ["replay"]
    assert rule_calls == ["rule", "rule"]
    assert len(cache) == 2
    assert cache.replay_entries == 1


def test_calibration_replay_cache_is_plan_independent_but_policy_sensitive():
    receipt = E.CalibrationReceipt.from_dict(receipt_dict())
    calls = []
    rule_calls = []

    def replay(_receipt):
        calls.append("replay")
        return 1.5

    def rule(_receipt, _plan):
        rule_calls.append("rule")
        return True

    cache = E.CalibrationCache()
    kwargs = {"registered_estimators": {"median.v1": replay},
              "registered_rule_id": "rule.v1", "applicability_rule": rule,
              "cache": cache}
    first_obj = plan_dict(n=1, calibration_ref=receipt.digest)
    first = E.ExperimentPlan.from_dict(first_obj)
    second_obj = plan_dict(n=1, calibration_ref=receipt.digest, unit="arm")
    second_obj["plan_id"] = "plan-2"
    second = E.ExperimentPlan.from_dict(second_obj)
    E.calibration_applicability(receipt, first, **kwargs)
    result = E.calibration_applicability(receipt, second, **kwargs)
    assert result.replay_cache_hit
    assert calls == ["replay"] and rule_calls == ["rule", "rule"]

    changed_policy = second.to_dict()
    changed_policy["plan_id"] = "plan-3"
    changed_policy["policy_snapshot"]["digest"] = "c" * 64
    result = E.calibration_applicability(
        receipt, E.ExperimentPlan.from_dict(changed_policy), **kwargs)
    assert not result.replay_cache_hit
    assert calls == ["replay", "replay"]

    changed_receipt_obj = receipt.to_dict()
    changed_receipt_obj["raw_sample_digest"] = "d" * 64
    changed_receipt = E.CalibrationReceipt.from_dict(changed_receipt_obj)
    changed_receipt_plan = first.to_dict()
    changed_receipt_plan["plan_id"] = "plan-4"
    changed_receipt_plan["calibration_ref"] = changed_receipt.digest
    E.calibration_applicability(
        changed_receipt, E.ExperimentPlan.from_dict(changed_receipt_plan), **kwargs)
    assert calls == ["replay", "replay", "replay"]

    changed_estimator_obj = receipt.to_dict()
    changed_estimator_obj["estimator_id"] = "median.v2"
    changed_estimator = E.CalibrationReceipt.from_dict(changed_estimator_obj)
    changed_estimator_plan = first.to_dict()
    changed_estimator_plan["plan_id"] = "plan-5"
    changed_estimator_plan["estimator_id"] = "median.v2"
    changed_estimator_plan["calibration_ref"] = changed_estimator.digest
    estimator_kwargs = dict(kwargs)
    estimator_kwargs["registered_estimators"] = {
        "median.v1": replay, "median.v2": replay}
    E.calibration_applicability(
        changed_estimator, E.ExperimentPlan.from_dict(changed_estimator_plan),
        **estimator_kwargs)
    assert calls == ["replay", "replay", "replay", "replay"]


def test_direct_calibration_constructor_is_revalidated():
    receipt = E.CalibrationReceipt.from_dict(receipt_dict())
    plan = E.ExperimentPlan.from_dict(
        plan_dict(n=1, calibration_ref=receipt.digest))
    with pytest.raises(E.PlanValidationError, match="integer"):
        E.calibration_applicability(
            replace(receipt, n=True), plan,
            registered_estimators={"median.v1": lambda _receipt: 1.5},
            registered_rule_id="rule.v1",
            applicability_rule=lambda _receipt, _plan: True)


def test_transient_calibration_callback_failures_are_not_cached():
    receipt = E.CalibrationReceipt.from_dict(receipt_dict())
    plan = E.ExperimentPlan.from_dict(
        plan_dict(n=1, calibration_ref=receipt.digest))
    replay_calls = []

    def transient_replay(_receipt):
        replay_calls.append("call")
        raise RuntimeError("temporary")

    cache = E.CalibrationCache()
    kwargs = {"registered_estimators": {"median.v1": transient_replay},
              "registered_rule_id": "rule.v1",
              "applicability_rule": lambda _r, _p: True, "cache": cache}
    for _ in range(2):
        assert E.calibration_applicability(receipt, plan, **kwargs).status == "recalibration_required"
    assert replay_calls == ["call", "call"]
    assert cache.replay_entries == cache.applicability_entries == 0

    rule_calls = []

    def transient_rule(_receipt, _plan):
        rule_calls.append("call")
        raise RuntimeError("temporary")

    kwargs["registered_estimators"] = {"median.v1": lambda _receipt: 1.5}
    kwargs["applicability_rule"] = transient_rule
    for _ in range(2):
        assert E.calibration_applicability(receipt, plan, **kwargs).status == "recalibration_required"
    assert rule_calls == ["call", "call"]
    assert cache.replay_entries == 1 and cache.applicability_entries == 0
