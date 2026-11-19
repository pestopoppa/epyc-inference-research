"""AKU-07k: Journal-level binding validator and recovery consumer (pure, synthetic)."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from . import scheduling as S
from . import worker_lifecycle as lifecycle
from .test_scheduling import config, proposal
from .test_worker_lifecycle import _binding, _bound_request, _receipt_harness


def _violations(rows, issued=None):
    try:
        lifecycle.validate_accounting_binding_history(rows, issued)
    except lifecycle.LifecycleRefused as exc:
        return [str(exc)]
    return []


def _history(bound: bool):
    harness = _receipt_harness()
    try:
        request = _bound_request(harness) if bound else harness.request("pass")
        harness.engine.run_stage(request)
        return [copy.deepcopy(r) for r in harness.events]
    finally:
        harness.close()


def test_bound_history_passes_and_legacy_unbound_history_is_unchanged():
    bound, legacy = _history(True), _history(False)
    assert _violations(bound) == []
    assert _violations(legacy) == []
    assert not any("accounting_binding" in r["data"].get("contract", {}) for r in legacy)
    assert not any("accounting_binding_digest" in r["data"] for r in legacy)


@pytest.mark.parametrize("mutate", [
    lambda held: held["data"].update(accounting_binding_digest="0" * 64),
    lambda held: held["data"].pop("accounting_binding_digest"),
    lambda held: held["data"]["receipt"].update(proposal_id="other"),
    lambda held: held["data"]["receipt"].update(backend="gpu"),
    lambda held: held["data"]["receipt"].update(stage_class="search"),
])
def test_validator_refuses_held_receipt_that_differs_from_the_launch_binding(mutate):
    rows = _history(True)
    held = next(r for r in rows if r["event"] == "OWNED_HELD_RECEIPT_RETAINED")
    mutate(held)
    assert _violations(rows)


def test_validator_refuses_binding_digest_on_an_unbound_launch():
    rows = _history(False)
    held = next(r for r in rows if r["event"] == "OWNED_HELD_RECEIPT_RETAINED")
    held["data"]["accounting_binding_digest"] = _binding().digest
    assert _violations(rows)


def _issued_for(binding_changes=None):
    cfg = S.SchedulerConfig.from_dict(config())
    engine = S.SchedulerEngine(cfg, S.initial_state(cfg, "scheduler"))
    selection = engine.select_stage([proposal("op")], now=1.0)
    row = {"catalog_id": "1" * 64, "transition_id": "2" * 64,
           "selection": selection.to_dict()}
    return row, selection


def test_verify_issued_requires_exact_catalog_transition_selection_proposal():
    row, selection = _issued_for()
    good = lifecycle.AccountingBinding.from_selection(
        catalog_id=row["catalog_id"], transition_id=row["transition_id"],
        selection=selection)
    good.verify_issued(row)
    for name, value in (("catalog_digest", "9" * 64), ("transition_digest", "9" * 64),
                        ("selection_digest", "9" * 64), ("proposal_digest", "9" * 64),
                        ("proposal_id", "other"), ("backend", "gpu"),
                        ("scheduler_stage_class", "calibration")):
        changed = replace(good, **{name: value})
        if changed == good:
            continue
        with pytest.raises(lifecycle.LifecycleRefused):
            changed.verify_issued(row)
    with pytest.raises(lifecycle.LifecycleRefused):
        good.verify_issued(None)


def test_recovery_history_refuses_binding_naming_an_unissued_or_other_selection():
    rows = _history(True)
    assert _violations(rows, {}) , "never issued"
    row, _selection = _issued_for()
    # _binding() digests are fixture constants, so they cannot equal a real issued selection
    assert _violations(
        rows, {_binding().catalog_digest: row})
