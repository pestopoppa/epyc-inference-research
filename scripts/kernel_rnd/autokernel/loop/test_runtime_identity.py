"""DS41-C97: runtime treatments are deduplicated by what they change, not by name.

Pure fixtures: canonical launches are resolved in memory, nothing is launched or measured.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from . import actors, archive, gates, loop, runtime_identity, serial_scheduling
from ..controller import experiments
from .test_unified_planner import canonical_recipe

KEYS = ("OMP_WAIT_POLICY", "OMP_PROC_BIND")
REQUESTS = "e" * 64


def _anchor(threads=4):
    return canonical_recipe(threads=threads, policy_keys=KEYS,
                            launch_extra={"OMP_WAIT_POLICY": "active", "OMP_PROC_BIND": "spread"})


def _context(anchor, **extra):
    return {"runtime_anchor": anchor.to_dict(), "runtime_env_keys": list(KEYS),
            "runtime_request_digest": REQUESTS, **extra}


def _pair(anchor, mechanism, key="OMP_WAIT_POLICY", value="passive"):
    return actors._runtime_pair({"kind": "env", "candidate": {"key": key, "value": value}},
                                _context(anchor), mechanism)


def _attempt(pair, mechanism, effect=-0.59807):
    return {"status": "runtime_observed", "mechanism_id": mechanism,
            "statement": "Change only OMP_WAIT_POLICY", "falsifier": "matched A/B",
            "target_surface": "launch_env.OMP_WAIT_POLICY", "target_symbol": "OMP_WAIT_POLICY",
            "effect_fraction": effect, "runtime_pair": pair.to_dict(),
            "comparison": {"effect": effect, "runtime_pair": pair.to_dict(),
                           "recipe_hash": pair.anchor.template_hash,
                           "request_digest": REQUESTS}}


def _record_first(store, anchor):
    """The 13:05Z row: `akm-ds41-omp-passive-wait-diagnostic`, -59.807%."""
    first = _pair(anchor, "akm-ds41-omp-passive-wait-diagnostic")
    receipts = []
    assert archive.record(store, _attempt(first, first.dimension.dimension_id),
                          epoch="epoch-1305", recorded_at="2026-09-30T13:05:42Z",
                          campaign_id="ak-loop", journal_receipt_out=receipts)
    return receipts[0]["attempt_id"]


def _iterate(hypothesis, context, gate_reason="sentinel: runtime gate reached"):
    reached = []

    def forbidden(*args, **kwargs):
        pytest.fail("a runtime treatment reached author/measure/commit")

    def gate(hyp, paths):
        reached.append(hyp.mechanism_id)
        return False, [gates.Verdict("correctness", False, gate_reason)]

    outcome = loop.iterate(
        planner=SimpleNamespace(propose=lambda _context: hypothesis, author=forbidden),
        critic=SimpleNamespace(review_hypothesis=forbidden, review_patch=forbidden),
        context=context, gate=gate, measure=forbidden, commit=forbidden)
    return outcome, reached


def _hypothesis(pair, **extra):
    return loop.Hypothesis(pair.dimension.dimension_id, "passive OpenMP waiting",
                           "matched serving A/B", "launch_env.OMP_WAIT_POLICY",
                           "OMP_WAIT_POLICY", runtime_pair=pair, **extra)


def test_same_env_delta_under_a_new_mechanism_id_is_refused_naming_the_prior_row(tmp_path):
    anchor = _anchor()
    prior_id = _record_first(tmp_path, anchor)
    observed = runtime_identity.observed(tmp_path, anchor.to_dict(), REQUESTS)
    assert [row["attempt_id"] for row in observed] == [prior_id]
    assert observed[0]["describe"] == "OMP_WAIT_POLICY: active -> passive"

    renamed = _pair(anchor, "akm-ds41-passive-omp-wait-diagnostic")
    outcome, reached = _iterate(_hypothesis(renamed),
                                _context(anchor, runtime_treatments_observed=observed))
    assert reached == []
    assert outcome.status == "refused_duplicate"
    assert outcome.refusal_gate == runtime_identity.REFUSAL_GATE == "runtime_treatment_identity"
    assert outcome.duplicate_of == prior_id
    assert outcome.prior_effect == pytest.approx(-0.59807)
    assert outcome.prior_epoch == "epoch-1305"
    reason = outcome.reasons[0]
    assert prior_id in reason and "akm-ds41-omp-passive-wait-diagnostic" in reason
    assert "-59.807%" in reason and "OMP_WAIT_POLICY: active -> passive" in reason
    row = outcome.to_attempt()
    assert row["status"] == "refused_duplicate" and row["duplicate_of"] == prior_id
    # The refusal itself measured nothing, so it never enters the ledger.
    assert archive.record(tmp_path, row, epoch="epoch-1610", recorded_at="2026-09-30T16:10Z",
                          campaign_id="ak-loop")
    assert len(runtime_identity.observed(tmp_path, anchor.to_dict(), REQUESTS)) == 1


def test_identity_ignores_the_mechanism_id_and_the_dimension_wording():
    anchor = _anchor()
    one = runtime_identity.pair_identity(_pair(anchor, "akm-a"), REQUESTS)
    two = runtime_identity.pair_identity(_pair(anchor, "akm-completely-different"), REQUESTS)
    assert one["identity"] == two["identity"]
    assert one["delta"] == {"env": {"OMP_WAIT_POLICY": ["active", "passive"]}}


def test_a_different_delta_is_admitted(tmp_path):
    anchor = _anchor()
    _record_first(tmp_path, anchor)
    observed = runtime_identity.observed(tmp_path, anchor.to_dict(), REQUESTS)
    for pair in (_pair(anchor, "akm-proc-bind-close", key="OMP_PROC_BIND", value="close"),
                 _pair(anchor, "akm-wait-unset", value=None)):
        outcome, reached = _iterate(_hypothesis(pair),
                                    _context(anchor, runtime_treatments_observed=observed))
        assert reached == [pair.dimension.dimension_id]
        assert outcome.status == "runtime_refused"
        assert outcome.refusal_gate != runtime_identity.REFUSAL_GATE


def test_the_same_delta_under_a_changed_recipe_hash_is_admitted(tmp_path):
    anchor = _anchor()
    _record_first(tmp_path, anchor)
    moved = _anchor(threads=8)
    assert moved.template_hash != anchor.template_hash
    observed = runtime_identity.observed(tmp_path, moved.to_dict(), REQUESTS)
    assert observed == []
    pair = _pair(moved, "akm-ds41-passive-omp-wait-diagnostic")
    outcome, reached = _iterate(_hypothesis(pair),
                                _context(moved, runtime_treatments_observed=observed))
    assert reached == [pair.dimension.dimension_id] and outcome.status == "runtime_refused"
    # Even handed the old frame's row, the identity differs: the frame is bound in.
    stale = runtime_identity.observed(tmp_path, anchor.to_dict(), REQUESTS)
    assert runtime_identity.duplicate(pair, _context(moved, runtime_treatments_observed=stale)) is None


def test_the_same_delta_under_different_frozen_requests_is_admitted(tmp_path):
    anchor = _anchor()
    _record_first(tmp_path, anchor)
    assert runtime_identity.observed(tmp_path, anchor.to_dict(), "f" * 64) == []
    rows = runtime_identity.observed(tmp_path, anchor.to_dict(), REQUESTS)
    pair = _pair(anchor, "akm-renamed")
    assert runtime_identity.duplicate(pair, {"runtime_treatments_observed": rows,
                                             "runtime_request_digest": "f" * 64}) is None


def test_a_declared_arm_continuation_is_exempt(tmp_path):
    anchor = _anchor()
    _record_first(tmp_path, anchor)
    observed = runtime_identity.observed(tmp_path, anchor.to_dict(), REQUESTS)
    pair = _pair(anchor, "akm-ds41-omp-passive-wait-diagnostic")
    outcome, reached = _iterate(_hypothesis(pair, declared_runtime_arm="omp-passive"),
                                _context(anchor, runtime_treatments_observed=observed))
    assert reached and outcome.status == "runtime_refused"
    assert "declared_runtime_arm" not in _hypothesis(pair, declared_runtime_arm="x").to_dict()


def test_prompt_lists_the_settled_treatment(tmp_path):
    anchor = _anchor()
    prior_id = _record_first(tmp_path, anchor)
    observed = runtime_identity.observed(tmp_path, anchor.to_dict(), REQUESTS)
    text = actors.render_context({"runtime_treatments_observed": observed})
    assert "## Runtime treatments already measured on this recipe — do NOT re-propose" in text
    assert ("runtime treatment `OMP_WAIT_POLICY: active -> passive` measured -59.807% "
            "(runtime_observed) on this recipe as `akm-ds41-omp-passive-wait-diagnostic`") in text
    assert prior_id[:12] in text and "do not re-propose" in text
    assert "Runtime treatments already measured" not in actors.render_context({})


def test_reconcile_backfills_rows_that_predate_the_ledger_once(tmp_path):
    anchor = _anchor()
    pair = _pair(anchor, "akm-ds41-omp-passive-wait-diagnostic")
    with experiments.ExperimentStore(tmp_path) as store:     # no write-side hook
        receipts = []
        assert store.record(_attempt(pair, "akm-ds41-omp-passive-wait-diagnostic"),
                            epoch="epoch-1305", recorded_at="2026-09-30T13:05:42Z",
                            campaign_id="ak-loop", receipt_out=receipts)
        store.record({"status": "runtime_observed", "mechanism_id": "legacy-no-pair",
                      "reason": "pre-schema"}, epoch="epoch-0",
                     recorded_at="2026-09-01T00:00:00Z", campaign_id="ak-loop")
    rows = runtime_identity.observed(tmp_path, anchor.to_dict(), REQUESTS)
    assert [row["attempt_id"] for row in rows] == [receipts[0]["attempt_id"]]
    with runtime_identity.Ledger(tmp_path) as ledger:
        assert len(ledger.known()) == 2          # the legacy row is filed, never re-read
        assert runtime_identity.reconcile(tmp_path, ledger) == 0


def test_observed_fails_open_when_the_ledger_is_unreadable(tmp_path, capsys):
    (tmp_path / runtime_identity.LEDGER_NAME).write_bytes(b"not a sqlite database" * 64)
    assert runtime_identity.observed(tmp_path, _anchor().to_dict(), REQUESTS) == []
    assert "runtime treatment ledger unavailable" in capsys.readouterr().err


def test_refused_duplicate_is_a_scheduler_invalid_outcome():
    assert serial_scheduling.one_iteration_outcome("complete", {"refused_duplicate": 1}) == "invalid"


def test_threads_delta_is_the_argv_change_not_the_dimension_name():
    anchor = _anchor()
    context = _context(anchor)
    first = actors._runtime_pair({"kind": "threads", "candidate": 8}, context, "akm-t8")
    again = actors._runtime_pair({"kind": "threads", "candidate": 8}, context, "akm-eight")
    one = runtime_identity.pair_identity(first, REQUESTS)
    assert one["identity"] == runtime_identity.pair_identity(again, REQUESTS)["identity"]
    assert one["delta"]["argv"]["-t"] == ["4", "8"]
    # replace() keeps the pair; the identity is a function of the pair, not the hypothesis.
    hypothesis = replace(_hypothesis(first), mechanism_id="akm-other")
    assert runtime_identity.pair_identity(hypothesis.runtime_pair, REQUESTS) == one
