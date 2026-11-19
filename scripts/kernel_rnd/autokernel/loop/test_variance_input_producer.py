"""Variance-input producer. ALL records below are SYNTHETIC fixtures (no launch, no host):
they prove record-shape consumption, typed refusal and the producer -> variance_keep hand-off."""
import copy
import random

import pytest

from . import variance_input_producer as vip
from . import variance_keep as vk

FLOOR = {"pct": 2.0, "unit": "session", "recipe_hash": "a" * 64}


def _record(n=20, *, seed=7, tighter=True):
    """SYNTHETIC serving_runtime_ab.v1 shape; anchor has a downside tail, candidate is stable."""
    rng = random.Random(seed)
    anchor = [100.0 - (rng.uniform(6, 10) if i % 4 == 0 else rng.uniform(0, 0.4)) for i in range(n)]
    cand = [100.0 - (rng.uniform(0, 0.3) if tighter else (rng.uniform(6, 10) if i % 4 == 0 else 0.1))
            for i in range(n)]
    membership = []
    for i in range(n):
        order = ("anchor", "candidate") if i % 2 == 0 else ("candidate", "anchor")
        for position, arm in enumerate(order):
            membership.append({"ordinal": 2 * i + position, "pair_index": i, "arm": arm,
                               "arm_sample_index": i, "residency_sha256": "0" * 64})
    return {"schema": "epyc.autokernel.serving_runtime_ab.v1", "recipe_hash": "a" * 64,
            "candidate_recipe_hash": "b" * 64, "effect_unit": "process", "pairs": n,
            "runtime_pair": {"synthetic": True}, "anchor_samples": anchor, "candidate_samples": cand,
            "launch_membership": membership}


def _refused(code, **kwargs):
    args = {"record": _record(), "candidate_id": "cand-1", "floor": FLOOR} | kwargs
    with pytest.raises(vip.VarianceInputRefused) as caught:
        vip.produce(**args)
    assert caught.value.code == code


def test_document_is_exactly_the_closed_variance_keep_input(tmp_path):
    out = vip.produce(_record(), candidate_id="cand-1", floor=FLOOR)
    assert set(out.document) == {"schema", "candidate_id", "floor_pct", "floor_unit", "units"}
    assert out.document["schema"] == vk.INPUT_SCHEMA and len(out.document["units"]) == 20
    path = tmp_path / "variance_input.json"
    path.write_text(vip.dumps(out.document))
    assert vk.load_input(path) == out.document


def test_end_to_end_synthetic_paired_launches_to_variance_keep_verdict(tmp_path):
    out = vip.produce(_record(tighter=True), candidate_id="cand-1", floor=FLOOR)
    assert out.verdict["verdict"] == vk.KEEP_VARIANCE and out.verdict["n_units"] == 20
    path = tmp_path / "vi.json"
    path.write_text(vip.dumps(out.document))
    kept = vk.keep_record_fields(path, tmp_path / "keep-frontier.json")
    assert kept["variance_keep"] == out.verdict
    same = vip.produce(_record(tighter=False), candidate_id="cand-2", floor=FLOOR)
    assert same.verdict["verdict"] == vk.REFUSE and vk.R_NOT_TIGHTER in same.verdict["reasons"]


def test_short_record_yields_insufficient_n_through_variance_keeps_own_path():
    out = vip.produce(_record(n=13), candidate_id="cand-1", floor=FLOOR)
    assert out.verdict["verdict"] == vk.REFUSE and out.verdict["reasons"] == [vk.R_INSUFFICIENT_N]
    assert len(out.document["units"]) == 13


def test_pairing_follows_launch_membership_not_list_order():
    record = _record()
    for row in record["launch_membership"]:
        if row["arm"] == "candidate":
            row["arm_sample_index"] = (row["pair_index"] + 1) % 20
    out = vip.produce(record, candidate_id="c", floor=FLOOR)
    assert out.document["units"][0]["candidate"] == record["candidate_samples"][1]


def test_unpaired_or_non_separate_launches_are_refused():
    short = _record()
    short["candidate_samples"].pop()
    _refused(vip.C_UNPAIRED, record=short)
    for mutate in (lambda m: m.pop(),
                   lambda m: m[0].__setitem__("arm", "candidate"),
                   lambda m: m[1].__setitem__("ordinal", m[0]["ordinal"]),
                   lambda m: m[0].__setitem__("pair_index", 99)):
        bad = _record()
        mutate(bad["launch_membership"])
        _refused(vip.C_UNPAIRED, record=bad)
    legacy = _record()
    del legacy["launch_membership"]
    _refused(vip.C_NOT_MATCHED, record=legacy)


def test_recipe_unit_floor_and_schema_refusals():
    _refused(vip.C_RECIPE, floor={**FLOOR, "recipe_hash": "c" * 64})
    missing = _record()
    del missing["candidate_recipe_hash"]
    _refused(vip.C_RECIPE, record=missing)
    _refused(vip.C_UNIT, unit="process")
    _refused(vip.C_UNIT, floor={**FLOOR, "unit": "process"})
    _refused(vip.C_FLOOR, floor=None)
    _refused(vip.C_FLOOR, floor={**FLOOR, "pct": 0})
    _refused(vip.C_FLOOR, floor={**FLOOR, "pct": True})
    legacy = _record()
    legacy["schema"] = "epyc.autokernel.serving_ab.v2"
    _refused(vip.C_SCHEMA, record=legacy)
    plain = _record()
    plain["runtime_pair"] = None
    _refused(vip.C_SCHEMA, record=plain)
    _refused(vip.C_CANDIDATE, candidate_id="")
    nonfinite = _record()
    nonfinite["anchor_samples"][3] = float("nan")
    _refused(vip.C_VALUE, record=nonfinite)
    neg = _record()
    neg["candidate_samples"][3] = -1.0
    _refused(vip.C_VALUE, record=neg)


def test_producer_does_not_mutate_its_input():
    record = _record()
    before = copy.deepcopy(record)
    vip.produce(record, candidate_id="c", floor=FLOOR)
    assert record == before
