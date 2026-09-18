"""Focused, hardware-free tests for integrity held-out serving confirmation."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace

import pytest

from . import heldout_serving
from . import run
from . import serving


def _recipe():
    return SimpleNamespace(np=1)


def _request(prompt, **controls):
    return json.dumps({"prompt": prompt, "n_predict": 512, "seed": 42,
                       **controls}, sort_keys=True).encode()


def _row(effect, decisive):
    return SimpleNamespace(to_dict=lambda: {"effect": effect, "decisive": decisive,
        "noise_floor_pct": 2.0, "request_digest": "heldout",
        "effect_unit": "process", "floor_unit": "process"})


def test_prompt_id_only_is_not_an_unseen_request():
    with pytest.raises(ValueError, match="request bytes"):
        heldout_serving.validate_requests(
            _recipe(), (("public", _request([1, 2])),),
            (("new-id", _request([1, 2])),))


def test_distinct_request_bytes_are_admitted():
    public, heldout = heldout_serving.validate_requests(
        _recipe(), (("public", _request([1, 2])),),
        (("heldout", _request([2, 1])),))
    assert public != heldout


def test_changed_prompt_length_or_controls_refuses_shape_claim():
    for changed in (_request([1, 2, 3]), _request([2, 1], seed=43)):
        with pytest.raises(ValueError, match="preserve token count"):
            heldout_serving.validate_requests(
                _recipe(), (("public", _request([1, 2])),),
                (("heldout", changed),))


def test_heldout_floor_path_is_request_bound_and_not_public_floor(tmp_path):
    recipe = SimpleNamespace(np=1, name="glm-serving")
    public = (("public", _request([1, 2])),)
    heldout = (("heldout", _request([2, 1])),)
    original = serving.floor_path(tmp_path, recipe, frozen_requests=public,
                                  instrument=serving.MATCHED_INSTRUMENT, pairs=5)
    rotated = serving.floor_path(tmp_path, recipe, frozen_requests=heldout,
                                 instrument=serving.MATCHED_INSTRUMENT, pairs=5)
    assert original != rotated
    assert heldout_serving.validate_requests(recipe, public, heldout)[1] in rotated.name


def test_calibration_only_branch_precedes_proposal_pool():
    source = (Path(__file__).parent / "run.py").read_text()
    end = source.index("        publish_held_claims()\n        elapsed = time.time() - started")
    branch = source.rfind("            if args.heldout_calibration_only:", 0, end)
    assert branch > 0
    assert "pooled = pool.PoolResult(outcomes=[]" in source[branch:end]
    assert "elif args.validate_source_continuation:" in source[branch:end]
    assert "else:\n                pooled = run_pooled()" in source[branch:end]


def test_heldout_floor_survives_accumulator_tip_advance(monkeypatch, tmp_path):
    """A keep changes the tip digest, but not the protected COR calibration."""
    reference = tmp_path / "protected-cor"
    first_tip = tmp_path / "tip-before-keep"
    second_tip = tmp_path / "tip-after-keep"
    calls = []

    def fake_arm(_launch, build):
        return SimpleNamespace(execution_digest=Path(build).name, build=Path(build))

    def fake_load(_store, _recipe, arm, *, frozen_requests, instrument, pairs):
        calls.append(arm.build)
        value = 6.351 if arm.build == reference else None
        return tmp_path / arm.build.name, SimpleNamespace(floor_pct=value)

    monkeypatch.setattr(run, "_cpu_arm", fake_arm)
    monkeypatch.setattr(run, "_load_source_floor", fake_load)
    for tip in (first_tip, second_tip):
        floor_store, reading = run._load_heldout_floor(
            tmp_path, object(), object(), tip_build=tip,
            reference_build=reference, frozen_requests=(("heldout", b"bytes"),),
            instrument=serving.MATCHED_INSTRUMENT, pairs=5)
        assert reading.floor_pct == 6.351
        assert floor_store == tmp_path / "protected-cor"
    assert calls == [first_tip, reference, second_tip, reference]


def test_calibration_targets_protected_cor_build():
    source = (Path(__file__).parent / "run.py").read_text()
    assert "heldout_anchor = (args.cor_build or args.anchor_build)" in source


@pytest.mark.parametrize("effect,decisive,expected", [
    (-0.08, True, False), (-0.01, False, True), (0.01, False, True),
    (0.08, True, True), (0.01, None, False),
])
def test_decision_retains_nondecisive_candidates_but_vetoes_regressions(
        effect, decisive, expected):
    with tempfile.TemporaryDirectory() as tmp:
        record = heldout_serving.decide(
            store=Path(tmp), mechanism_id="candidate", screen=_row(0.02, True),
            heldout=_row(effect, decisive), public_digest="public",
            heldout_digest="heldout", floor_path=Path(tmp) / "floor.json")
        assert record["promoted"] is expected
        assert len(list((Path(tmp) / "confirm").glob("candidate.*.json"))) == 1


def test_invalid_effect_or_request_identity_never_promotes():
    with tempfile.TemporaryDirectory() as tmp:
        for effect, digest in ((float("nan"), "heldout"), (0.01, "public")):
            row = _row(effect, False).to_dict()
            row["request_digest"] = digest
            heldout = SimpleNamespace(to_dict=lambda row=row: row)
            record = heldout_serving.decide(
                store=Path(tmp), mechanism_id="invalid", screen=_row(0.02, True),
                heldout=heldout, public_digest="public", heldout_digest="heldout",
                floor_path=Path(tmp) / "floor.json")
            assert not record["promoted"]
