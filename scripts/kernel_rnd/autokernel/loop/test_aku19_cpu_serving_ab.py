"""AKU-19 code half: the CPU serving gate emits an `epyc.autokernel.serving_ab.v1` record.

Fixture only: `_measure_once` is stubbed, nothing launches. The first LIVE record is a
separate (blocked) step and is not produced here.
"""
import json
from pathlib import Path
from unittest import mock

from autokernel.loop import serving

CPU_RECIPE = serving.Recipe(name="cpu-gate", model="/m/t.gguf", device="none", ngl=0,
                            np=2, n_predict=8, temperature=0.6, cpu_list="96-103")
REQUIRED = ("recipe", "recipe_hash", "metric", "np", "pairs", "anchor_tok_s",
            "candidate_tok_s", "effect", "effect_pct", "effect_unit", "floor_unit",
            "noise_floor_pct", "decisive", "anchor_samples", "candidate_samples")


def _stub(values):
    it = iter(values)
    return lambda *_a, **_k: next(it)


def test_cpu_gate_emits_serving_ab_v1_record(tmp_path):
    with mock.patch.object(serving, "_measure_once", side_effect=_stub([100.0, 110.0] * 3)):
        out = serving.compare(CPU_RECIPE, Path("/a"), Path("/c"), pairs=3, floor_pct=2.0,
                              floor_unit=serving.COMPARE_EFFECT_UNIT)
    assert out["schema"] == "epyc.autokernel.serving_ab.v1"
    assert all(k in out for k in REQUIRED)
    assert out["recipe"] == "cpu-gate" and out["pairs"] == 3
    assert abs(out["effect_pct"] - 10.0) < 1e-6 and out["decisive"] is True
    # The record is a JSON artifact and round-trips with its schema intact.
    path = tmp_path / "serving_ab.json"
    path.write_text(json.dumps(out, default=str))
    assert serving.comparison_instrument_matches(json.loads(path.read_text()),
                                                 instrument=serving.LEGACY_INSTRUMENT, pairs=3)


def test_cpu_gate_null_is_still_a_v1_record():
    with mock.patch.object(serving, "_measure_once", side_effect=_stub([100.0, 100.5] * 3)):
        out = serving.compare(CPU_RECIPE, Path("/a"), Path("/c"), pairs=3, floor_pct=2.0,
                              floor_unit=serving.COMPARE_EFFECT_UNIT)
    assert out["schema"] == "epyc.autokernel.serving_ab.v1" and out["decisive"] is not True
