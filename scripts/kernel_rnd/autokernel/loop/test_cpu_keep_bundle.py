import json

import pytest

from . import cpu_keep_bundle as cb

K = dict(keep_id="k1", effect_pct=4.5, magnitude="provisional", contention_label="pre-hook",
         statistic="arm_level_permutation", floor_pct=5.0, floor_unit="process")


def mk(**over):
    return cb.build(recipe="cpu-q38fn-np4", recipe_hash="h" * 8, champion_of_record="a" * 40,
                    tip="b" * 40, keeps=[{**K, **over}])


def test_roundtrip_and_filename(tmp_path):
    b = mk()
    path = cb.write(tmp_path, b)
    assert path.name == "accumulator-bundle.cpu.cpu-q38fn-np4.json"
    assert cb.read(path) == b and abs(b["compounded_bench_pct"] - 4.5) < 1e-9
    assert not list(tmp_path.glob(".accumulator-bundle-cpu-*"))


def test_unsafe_recipe_names_do_not_collide():
    assert cb.recipe_key("a/b") != cb.recipe_key("a_b") and "/" not in cb.recipe_key("a/b")


@pytest.mark.parametrize("over", [
    dict(magnitude="settled"),                                   # pre-hook cannot settle
    dict(magnitude="settled", contention_label="quiet", statistic="sign_test"),
    dict(magnitude="settled", contention_label="quiet", floor_pct=5.0, effect_pct=4.5),
    dict(magnitude="settled", contention_label="quiet", floor_pct=None),
    dict(statistic=""), dict(contention_label="disjoint"), dict(effect_pct=float("nan")),
    dict(floor_unit="bogus"),
])
def test_refusals(over):
    with pytest.raises(cb.CpuBundleError):
        mk(**over)


def test_settled_needs_permutation_quiet_and_clear_of_floor():
    b = mk(magnitude="settled", contention_label="quiet", effect_pct=8.0)
    assert b["keeps"][0]["magnitude"] == "settled"


def test_reader_refuses_tampering(tmp_path):
    b = mk()
    for bad in ({**b, "compounded_bench_pct": 9.0}, {**b, "extra": 1}, {**b, "schema": "x"}):
        with pytest.raises(cb.CpuBundleError):
            cb.from_dict(bad)


def test_fold_filter_requires_surface():
    got = cb.cpu_keeps_in_fold([{**K, "surface": "cpu"}, {**K, "keep_id": "g", "surface": "gpu"}])
    assert [k["keep_id"] for k in got] == ["k1"]
    with pytest.raises(cb.CpuBundleError):
        cb.cpu_keeps_in_fold([dict(K)])
