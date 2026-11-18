"""R23-48 leave-one-out planning and judgement (pure)."""
from __future__ import annotations

import pytest

from autokernel.controller import loo_arms as loo


def test_one_arm_per_keep_and_already_measured_skipped():
    arms = loo.plan_arms("champ1", ["k1", "k2", "k3"])
    assert [a.keep for a in arms] == ["k1", "k2", "k3"]
    assert len({a.arm_id() for a in arms}) == 3
    done = [arms[0].arm_id()]
    assert [a.keep for a in loo.plan_arms("champ1", ["k1", "k2", "k3"], already_measured=done)] == ["k2", "k3"]
    # a NEW champion re-plans every keep: the cost is n_keeps per promotion
    assert len(loo.plan_arms("champ2", ["k1", "k2", "k3"], already_measured=done)) == 3


def test_plan_refuses_empty_or_duplicate():
    with pytest.raises(ValueError):
        loo.plan_arms("", ["k1"])
    with pytest.raises(ValueError):
        loo.plan_arms("c", ["k1", "k1"])


@pytest.mark.parametrize("effect,floor,verdict", [
    (2.0, 0.9, loo.KEEP_CONFIRMED),
    (0.3, 0.9, loo.CANDIDATE_DROP),         # removal neutral: the lever is not demonstrably paying
    (-1.39, 0.9, loo.CANDIDATE_DROP),       # the INF-70 shape: removal helps
    (None, 0.9, loo.INCONCLUSIVE),
    (2.0, None, loo.INCONCLUSIVE),          # no floor: no verdict
    (2.0, 0.0, loo.INCONCLUSIVE),
])
def test_judge(effect, floor, verdict):
    assert loo.judge(effect, floor) == verdict


def test_retraction_record_carries_both_vectors_and_needs_a_floor():
    arm = loo.plan_arms("c", ["k1"])[0]
    rec = loo.retraction_record(arm, champion_samples=[1.0, 1.1], reverted_samples=[1.3, 1.4],
                                effect_pct=-1.39, floor_pct=0.9)
    assert rec["status"] == "retracted_by_loo" and rec["verdict"] == loo.CANDIDATE_DROP
    assert rec["champion_samples"] == [1.0, 1.1] and rec["reverted_samples"] == [1.3, 1.4]
    with pytest.raises(ValueError):                    # a lever that still helps is not retracted
        loo.retraction_record(arm, champion_samples=[1], reverted_samples=[1],
                              effect_pct=3.0, floor_pct=0.9)
    with pytest.raises(ValueError):
        loo.retraction_record(arm, champion_samples=[], reverted_samples=[1],
                              effect_pct=-1.0, floor_pct=0.9)


def test_summarize():
    out = loo.summarize({"a": loo.CANDIDATE_DROP, "b": loo.KEEP_CONFIRMED, "c": loo.INCONCLUSIVE})
    assert out["candidate_drops"] == ["a"] and out["inconclusive"] == ["c"]
