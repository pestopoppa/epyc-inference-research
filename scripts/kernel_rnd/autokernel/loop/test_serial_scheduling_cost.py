"""Factual stage forecasts; no new scheduler policy or scientific verdict."""
from copy import deepcopy
from dataclasses import replace

import pytest

from . import scheduling as s, serial_run as sr, serial_scheduling as ss
from .test_scheduling import config, proposal, receipt, vector


def _original():
    cfg = s.SchedulerConfig.from_dict(config())
    state = s.initial_state(cfg, "cost-test")
    state, selection = s.select_stage(cfg, state, [proposal()], now=1.)
    original = s.HeldClaimReceipt.from_dict(receipt(selection.proposal.to_dict(), end=4.))
    return cfg, state, selection, (original,)


def test_window_is_bounded_original_duration_p75_and_idempotent():
    cfg, state, selection, receipts = _original()
    history = None
    for index in range(12):
        stage = replace(selection.proposal, proposal_id=f"p{index}")
        issued = replace(selection, proposal=stage, proposal_digest=stage.digest)
        held = (replace(receipts[0], ended_at=float(index + 1)),)
        history = ss.retain_cost_sample(history, "target", "scope", issued, held)
    saved = deepcopy(history)
    history = ss.retain_cost_sample(history, "target", "scope", issued, held)
    assert history == saved
    assert len(history["targets"]["target"]["samples"]) == 8
    forecast = ss.duration_forecast(history, "target", "scope", proposal=selection.proposal,
                                    max_stage_seconds=20.)
    assert forecast["estimated_duration_seconds"] == 10.  # retained 5..12, rank ceil(.75*8)
    forecast["samples"][0]["held_seconds"] = 999
    assert history == saved
    assert ss.duration_forecast(history, "target", "other-scope", proposal=selection.proposal,
                                max_stage_seconds=20.) is None
    assert ss.duration_forecast(None, "target", "scope", proposal=selection.proposal,
                                max_stage_seconds=20.) is None


@pytest.mark.parametrize("change", ["recipe", "requests", "source", "cor", "runtime",
                                    "geometry", "epoch", "capacity", "stage", "claims", "setup"])
def test_compatibility_never_pools_other_original_inputs(change):
    cfg, state, selection, _ = _original()
    values = dict(binding={"documents": {"recipe": "a", "requests": "b"}},
                  anchor={"commit": "c", "path": "/original/build"},
                  cor_anchor={"commit": "d"}, runtime_recipe={"sha256": "e"},
                  geometry="quarter", preparation={"cpu_calibration": None},
                  proposal=selection.proposal, state=state)
    original = ss.cost_scope(**values)
    changed = deepcopy({k: v for k, v in values.items() if k not in {"proposal", "state"}})
    changed.update(proposal=values["proposal"], state=state)
    if change in {"recipe", "requests"}:
        changed["binding"]["documents"][change] = "changed"
    elif change == "source":
        changed["anchor"]["commit"] = "changed"
    elif change == "cor":
        changed["cor_anchor"]["commit"] = "changed"
    elif change == "runtime":
        changed["runtime_recipe"]["sha256"] = "changed"
    elif change == "geometry":
        changed["geometry"] = "half"
    elif change == "epoch":
        changed["state"] = replace(state, accounting_epoch=state.accounting_epoch + 1)
    elif change == "capacity":
        capacity = s.ResourceVector.from_dict(vector(gpus=("other-gpu",)))
        changed["state"] = replace(state, capacity=capacity, capacity_digest=capacity.digest)
    elif change == "stage":
        changed["proposal"] = replace(selection.proposal, stage_class="validation")
    elif change == "claims":
        changed["proposal"] = replace(selection.proposal,
                                      estimated_claims=s.ResourceVector.from_dict(vector(fraction=.25)))
    else:
        changed["preparation"]["cpu_calibration"] = "5"
    assert ss.cost_scope(**changed) != original


def test_actual_selector_uses_cost_only_after_original_coverage_and_seed_rules():
    cfg = s.SchedulerConfig.from_dict(config(capacity=vector(gpus=("gpu0",))))
    rows = {"cpu": s.StageProposal.from_dict(proposal("cpu", duration=5.)),
            "gpu": s.StageProposal.from_dict(proposal("gpu", frontier="production-gpu",
                backend="gpu", claims=vector(gpus=("gpu0",)), duration=5.)),
            "seed": s.StageProposal.from_dict(proposal("seed", seed="new-seed", duration=5.))}
    manifest = ss.SerialSchedulerManifest("selector-cost", cfg, rows)
    state = s.initial_state(cfg, manifest.scheduler_id)
    forecasts = {"cpu": {"stage_class": "search", "estimated_duration_seconds": .1}}
    first_state, selected, index = ss.select_target(manifest, state, tuple(rows), now=1.,
        stage_number=0, duration_forecasts=forecasts)
    assert tuple(rows)[index] == "gpu" and selected.slot_kind == "coverage"
    held = s.HeldClaimReceipt.from_dict(receipt(selected.proposal.to_dict(), gpus=("gpu0",)))
    state = s.account_stage(cfg, first_state, selected, held, outcome="valid_comparison")
    _, selected, index = ss.select_target(manifest, state, tuple(rows), now=6.,
        stage_number=1, duration_forecasts=forecasts)
    assert tuple(rows)[index] == "seed" and selected.slot_kind == "seed"
    # No coverage/seed due: the original selector receives the learned duration.
    rows = {key: replace(value, production_frontier=False, frontier_id=None, seed_id=None)
            for key, value in rows.items() if key != "seed"}
    manifest = ss.SerialSchedulerManifest("ordinary-cost", cfg, rows)
    state = s.initial_state(cfg, manifest.scheduler_id)
    _, selected, _ = ss.select_target(manifest, state, tuple(rows), now=1.,
        stage_number=0, duration_forecasts=forecasts)
    assert selected.proposal.estimated_duration_seconds == .1
    assert selected.service_bound_seconds == s.select_stage(cfg, state, tuple(rows.values()), now=1.)[1].service_bound_seconds
    assert cfg.adaptive_rule_id is None and cfg.normal_weight == 1 and cfg.seed_weight == 2


def test_malformed_history_is_not_an_accepted_duration():
    cfg, state, selection, receipts = _original()
    history = ss.retain_cost_sample(None, "target", "scope", selection, receipts)
    history["targets"]["target"]["samples"][0]["held_seconds"] = -1
    with pytest.raises(ss.SerialSchedulingRefused):
        ss.duration_forecast(history, "target", "scope", proposal=selection.proposal,
                             max_stage_seconds=cfg.max_stage_seconds)


@pytest.mark.parametrize("terminal,outcome,learn", [
    ("complete", "measured_null", True), ("complete", "keep_candidate", True),
    ("complete", "kept", False), ("complete", "measurement_invalid", False),
    ("complete", "runtime_observed", False), ("complete", "bench_failed", False),
    ("stopped", "measured_null", False),
])
def test_original_accounting_always_charges_but_truncated_or_changed_work_does_not_train(
        tmp_path, monkeypatch, terminal, outcome, learn):
    cfg, state, selection, receipts = _original()
    manifest = ss.SerialSchedulerManifest(state.scheduler_id, cfg, {"target": selection.proposal})
    owner_state = {"scheduler_state": state.to_dict()}
    active = {"selected_id": "target", "scheduler_selection": selection.to_dict(),
              "scheduler_selection_sha256": selection.digest}
    body = {"schema": sr.CONTINUATION_SCHEMA_V2, "held_claim_evidence": {},
            "selected_target": {}, "terminal": terminal, "outcome_counts": {outcome: 1},
            "input_argv": [], "current_anchor": {"commit": "original"}, "cor_anchor": None}
    monkeypatch.setattr(ss, "reopen_held_receipts", lambda *_args, **_kwargs: receipts)
    settled = sr._scheduled_account(owner_state, manifest, active, body, tmp_path)
    assert settled.campaign_attempts == 1 and settled.campaign_charged_seconds == 4.
    assert ("cost_forecast" in owner_state) is learn
    previous = deepcopy(owner_state.get("cost_forecast"))
    owner_state["scheduler_state"] = settled.to_dict()
    assert sr._scheduled_account(owner_state, manifest, active, body, tmp_path) == settled
    assert owner_state.get("cost_forecast") == previous  # original accounting replay adds nothing


def test_corrupt_optional_cost_summary_keeps_original_estimate_and_reports_diagnostic(monkeypatch):
    cfg, state, selection, _ = _original()
    manifest = ss.SerialSchedulerManifest(state.scheduler_id, cfg, {"target": selection.proposal})
    original = ["--target-id", "target", "--worktree", "/fixture/source"]
    body = {"input_argv": original, "current_anchor": {"commit": "original"}, "cor_anchor": None}
    owner_state = {"cost_forecast": {"policy": "malformed"}, "runtime_recovery": {},
                   "last_results": {"0": {"path": "/fixture/continuation", "sha256": "original"}},
                   "source_results": {}}
    monkeypatch.setattr(sr, "load_resume", lambda *_args: (body, "original"))
    assert sr._cost_forecasts(owner_state, manifest, state, [(0, original)],
                             {"target": {"scope": "full", "candidate": None}}) == {}
    assert "target" in owner_state["cost_forecast_errors"]
