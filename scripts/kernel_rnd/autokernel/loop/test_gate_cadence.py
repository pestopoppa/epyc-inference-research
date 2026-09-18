"""R23-54 (operator ruling 2026-09-08): the serving gate fires on a fixed KEEP CADENCE —
every 4 keeps — REGARDLESS of the compounded bench estimate.

WHY THESE TESTS EXIST. The gate's first firing (2026-09-08) measured the bench proxy
saying +5.958% while serving said "cannot tell, probably slightly negative" (-2.18%,
n=10). The threshold trigger is built entirely on that proxy, so the proxy was deciding
when to spend the only instrument that could contradict it. These tests pin the second,
proxy-independent trigger and — just as important — pin that it did NOT weaken the
fail-closed guards it was added alongside.
"""
import json
import shutil
import tempfile
from pathlib import Path

from autokernel.loop import accumulate as A


P = A.AccumulatorPolicy(fire_multiple=2.5)
FLOOR = 3.536          # the calibrated serving floor; threshold = 8.84%
UNDER = 1.0            # a compounded gain nowhere near the threshold


def _bundle(pct=0.0):
    return A.Bundle(champion_of_record="cor0", tip="cor0", compounded_bench_pct=pct)


def test_the_ruling_is_four_keeps():
    assert A.SERVING_GATE_EVERY_KEEPS == 4
    assert A.AccumulatorPolicy().every_keeps == 4


def test_counter_increments_on_every_keep():
    b = _bundle()
    assert b.keeps_since_serving_gate == 0
    for i in range(3):
        b.add_keep(f"m{i}", f"c{i}", UNDER)
        assert b.keeps_since_serving_gate == i + 1
    assert len(b.keeps) == 3


def test_cadence_fires_on_the_fourth_keep_with_the_threshold_not_met():
    """The ruling's whole point: four keeps whose compounded bench is FAR below the
    fire threshold must still buy a real serving reading."""
    b = _bundle()
    for i in range(3):
        b.add_keep(f"m{i}", f"c{i}", UNDER)
        assert A.gate_trigger(b, FLOOR, P) is None, f"fired early at keep {i + 1}"
        assert A.decide_after_keep(b, FLOOR, P) is A.Decision.ACCUMULATE
    b.add_keep("m3", "c3", UNDER)
    assert b.compounded_bench_pct < P.fire_threshold_pct(FLOOR)   # threshold NOT met
    assert A.gate_trigger(b, FLOOR, P) == "cadence"
    assert A.decide_after_keep(b, FLOOR, P) is A.Decision.FIRE_SERVING


def test_threshold_still_fires_early_before_the_cadence():
    """The cheap trigger is not removed — a bundle that clears the threshold on keep 1
    fires immediately, and says so."""
    b = _bundle()
    b.add_keep("m0", "c0", 9.10)               # > 8.84
    assert b.keeps_since_serving_gate == 1     # cadence nowhere near
    assert A.gate_trigger(b, FLOOR, P) == "threshold"
    assert A.decide_after_keep(b, FLOOR, P) is A.Decision.FIRE_SERVING


def test_both_when_the_threshold_is_met_on_the_cadence_keep():
    b = _bundle()
    for i in range(3):
        b.add_keep(f"m{i}", f"c{i}", 2.0 * (i + 1))
    b.add_keep("m3", "c3", 9.10)
    assert b.keeps_since_serving_gate == 4 and b.compounded_bench_pct >= 8.84
    assert A.gate_trigger(b, FLOOR, P) == "both"


def test_counter_resets_when_the_gate_runs_whatever_the_outcome():
    """It counts READINGS TAKEN, not verdicts won. Resetting only on a promote would
    make a diverged bundle re-fire the expensive gate on every subsequent keep."""
    for row in ({"decisive": True, "effect": 0.07, "effect_pct": 7.0},      # PROMOTE
                {"decisive": False, "effect": 0.005, "effect_pct": 0.5},    # DIVERGED
                {"decisive": True, "effect": -0.02, "effect_pct": -2.0},    # DIVERGED
                {"decisive": None, "effect": 0.5, "effect_pct": 50.0}):     # DIVERGED
        b = _bundle()
        for i in range(4):
            b.add_keep(f"m{i}", f"c{i}", UNDER)
        A.resolve(b, dict(row, noise_floor_pct=FLOOR), P)   # the gate ran
        b.mark_serving_gate_fired()
        assert b.keeps_since_serving_gate == 0
        assert A.gate_trigger(b, FLOOR, P) is None, "gate re-fires immediately after firing"
        # and the cadence starts over: three more keeps accumulate, the fourth fires.
        for i in range(3):
            b.add_keep(f"n{i}", f"d{i}", UNDER)
            assert A.gate_trigger(b, FLOOR, P) is None
        b.add_keep("n3", "d3", UNDER)
        assert A.gate_trigger(b, FLOOR, P) == "cadence"


def test_an_uncalibrated_floor_still_blocks_both_triggers():
    """The fail-closed guard R23-54 did NOT touch: without a floor the gate's `decisive`
    is None, so it can only ever return DIVERGED — spending it on cadence would burn
    llama-server hours on a reading that cannot promote anything."""
    b = _bundle()
    for i in range(8):
        b.add_keep(f"m{i}", f"c{i}", 99.0)
    assert b.keeps_since_serving_gate == 8
    assert A.gate_trigger(b, None, P) is None
    assert A.decide_after_keep(b, None, P) is A.Decision.ACCUMULATE


def test_a_cadence_firing_cannot_pass_vacuously():
    """A cadence firing gets no easier verdict than a threshold one: the same
    fail-closed classification decides, so an indecisive serving row still HOLDS the
    champion of record. (2026-09-08's own reading was exactly this shape.)"""
    b = _bundle()
    for i in range(4):
        b.add_keep(f"m{i}", f"c{i}", UNDER)
    assert A.gate_trigger(b, FLOOR, P) == "cadence"
    r = A.resolve(b, {"decisive": False, "effect": -0.0218, "effect_pct": -2.18,
                      "noise_floor_pct": FLOOR}, P)
    assert r["outcome"] is A.Outcome.DIVERGED
    assert r["new_champion_of_record"] == "cor0"
    assert r["action"] is A.DivergenceAction.HOLD


def test_a_bundle_written_before_the_field_existed_loads_with_zero():
    """A pre-validity v1 bundle stays readable as unknown legacy history."""
    store = Path(tempfile.mkdtemp())
    try:
        old = {"schema": A.Bundle.LEGACY_SCHEMA,
               "champion_of_record": "cor0", "tip": "k1",
               "keeps": ["m1", "m2"], "compounded_bench_pct": 5.19}   # no counter field
        (store / A.Bundle.FILENAME).write_text(json.dumps(old), encoding="utf-8")
        b = A.Bundle.from_dict(old)
        assert b.keeps_since_serving_gate == 0
        got, note = A.load_bundle(store, anchor_commit="k1",
                                  is_ancestor=lambda a, c: True)
        assert got.keeps == ["m1", "m2"] and got.keeps_since_serving_gate == 0
        assert got.measurement_validity == A.MEASUREMENT_UNKNOWN_LEGACY
        assert "restored 2 keep(s)" in note
        assert A.gate_trigger(got, FLOOR, P) is None   # no unscheduled gate on restore
    finally:
        shutil.rmtree(store, ignore_errors=True)


def test_the_counter_round_trips_through_save_and_load():
    store = Path(tempfile.mkdtemp())
    try:
        b = A.Bundle(champion_of_record="cor0", tip="cor0")
        b.add_keep("m1", "k1", 2.0)
        b.add_keep("m2", "k2", 3.0)
        assert b.keeps_since_serving_gate == 2
        b.save(store)
        on_disk = json.loads((store / A.Bundle.FILENAME).read_text())
        assert on_disk["keeps_since_serving_gate"] == 2
        assert on_disk["schema"] == "epyc.autokernel.accumulator_bundle.v2"
        got, _ = A.load_bundle(store, anchor_commit="k2", is_ancestor=lambda a, c: True)
        assert got.keeps_since_serving_gate == 2
        # and it survives a restart mid-cadence: two more keeps, then it fires.
        got.add_keep("m3", "k3", 4.0)
        assert A.gate_trigger(got, FLOOR, P) is None
        got.add_keep("m4", "k4", 5.0)
        assert A.gate_trigger(got, FLOOR, P) == "cadence"
    finally:
        shutil.rmtree(store, ignore_errors=True)


def test_every_keeps_zero_disables_only_the_cadence():
    """Defensive: a policy with the cadence switched off must fall back to exactly the
    pre-R23-54 behaviour, never to 'fires on every keep'."""
    p0 = A.AccumulatorPolicy(fire_multiple=2.5, every_keeps=0)
    b = _bundle()
    for i in range(6):
        b.add_keep(f"m{i}", f"c{i}", UNDER)
    assert A.gate_trigger(b, FLOOR, p0) is None
    b.add_keep("big", "cbig", 9.10)
    assert A.gate_trigger(b, FLOOR, p0) == "threshold"
