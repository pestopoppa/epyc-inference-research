"""Post-keep measurement overhead (2026-10-03, DS41 duty cycle).

Two schedules, both flags with defaults ON that never touch the resume binding:
  * the champion-of-record-vs-tip accumulate bench runs every N keeps (default 3) and
    always on the keep that fires the R23-54 cadence gate;
  * an anchor guard whose object digests are IDENTICAL may skip its A/A session-health
    sample inside a window after a clean A/A. The digest check, heal and abort never skip.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import accumulate, anchor, loop, run, serial_run as sr
from .test_anchor import CHAMPION, FLOOR, _comparison


POLICY = accumulate.AccumulatorPolicy()  # every_keeps = 8 (operator 2026-09-28)


def _bundle(since_gate=0, validity=accumulate.MEASUREMENT_CURRENT):
    b = accumulate.Bundle(champion_of_record="a" * 40, tip="b" * 40)
    b.keeps_since_serving_gate = since_gate
    b.measurement_validity = validity
    return b


# ------------------------------------------------------------- accumulate cadence

def test_default_cadence_benches_every_third_keep_and_before_the_gate():
    due = []
    b = _bundle()
    for n in range(1, 9):
        reason = accumulate.bench_due(b, POLICY)
        due.append((n, reason))
        if reason is None:
            b.add_unmeasured_keep(f"k{n}", f"{n:040x}")
        else:
            b.add_keep(f"k{n}", f"{n:040x}", 1.0 * n)
    assert [n for n, r in due if r] == [3, 6, 8]
    assert dict(due)[8] == "before_serving_gate"
    assert dict(due)[3] == dict(due)[6] == "cadence"
    # The gate fires on a MEASURED bundle (the 8th keep re-measured).
    assert b.measurement_validity == accumulate.MEASUREMENT_CURRENT
    assert accumulate.gate_trigger(b, 10.0, POLICY) == "cadence"


def test_every_one_is_the_pre_change_behaviour():
    b = _bundle()
    assert all(accumulate.bench_due(b, POLICY, 1) == "every_keep" for _ in range(3))


def test_deferred_keep_cannot_fire_threshold_and_keeps_history():
    b = _bundle()
    b.add_keep("k1", "c" * 40, 50.0, comparison_evidence={"path": "/x", "sha256": "d" * 64})
    b.add_unmeasured_keep("k2", "e" * 40)
    assert b.keeps == ["k1", "k2"] and b.tip == "e" * 40
    assert b.keeps_since_serving_gate == 2
    assert b.compounded_bench_pct == 50.0  # retained as history, never re-invented
    assert b.measurement_validity == accumulate.MEASUREMENT_DEFERRED_BENCH
    assert accumulate.gate_trigger(b, 1.0, POLICY) is None  # 50% >> 2.5x1% but stale
    # Durable: the journal schema accepts the new validity and round-trips it.
    again = accumulate.Bundle.from_dict(json.loads(json.dumps(b.to_dict())))
    assert again.to_dict() == b.to_dict()


def test_deferred_bundle_survives_the_journal(tmp_path):
    b = accumulate.Bundle(champion_of_record="a" * 40, tip="a" * 40)
    b.add_unmeasured_keep("k1", "b" * 40)
    b.save(tmp_path)
    restored, _note = accumulate.load_bundle(tmp_path, anchor_commit="b" * 40,
                                             is_ancestor=lambda _a, _b: True)
    assert restored.measurement_validity == accumulate.MEASUREMENT_DEFERRED_BENCH
    assert restored.keeps_since_serving_gate == 1


@pytest.mark.parametrize("validity", [accumulate.MEASUREMENT_STALE_TIP_ADVANCE,
                                      accumulate.MEASUREMENT_STALE_RUNTIME_RECIPE,
                                      accumulate.MEASUREMENT_UNKNOWN_LEGACY])
def test_other_stale_states_remeasure_immediately(validity):
    assert accumulate.bench_due(_bundle(validity=validity), POLICY) == "stale_measurement"


def test_forced_bench_and_unarmed_gate():
    assert accumulate.bench_due(_bundle(), POLICY, force=True) == "forced"
    # No calibrated serving floor: the cadence gate cannot fire, so keep 9+ is not
    # "before the gate" -- only the N-cadence measures.
    assert accumulate.bench_due(_bundle(since_gate=9), POLICY, gate_armed=False) is None
    assert accumulate.bench_due(_bundle(since_gate=8), POLICY, gate_armed=False) == "cadence"
    assert accumulate.bench_due(_bundle(since_gate=9), POLICY) == "before_serving_gate"


# ------------------------------------------------------------- anchor A/A window

def _built():
    return SimpleNamespace(passed=True, reason="")


def _verify(tmp_path, *, digests, aa_skip, compared):
    seen = []

    def compare(a, f):
        compared.append((a, f))
        return _comparison(0.1)

    verdict = anchor.verify(
        champion_commit=CHAMPION, anchor_build=tmp_path / "anchor",
        build=lambda _p: _built(), compare=compare, noise_floor_pct=FLOOR,
        scratch_build=tmp_path / "scratch", digest=lambda p: digests[Path(p).name],
        clean=lambda _p: None, on_verdict=seen.append, aa_skip=aa_skip)
    return verdict, seen


def test_identical_digests_inside_window_skip_the_aa_without_a_number(tmp_path):
    compared = []
    verdict, seen = _verify(tmp_path, digests={"anchor": "x" * 64, "scratch": "x" * 64},
                            aa_skip=lambda: "recent clean A/A", compared=compared)
    assert compared == []
    assert verdict.passed and not verdict.excursion and verdict.pairs == 0
    assert verdict.effect_pct is None
    attempt = verdict.to_attempt()
    assert attempt["status"] == "anchor_verified" and attempt["effect_fraction"] is None
    # No comparison: the DS41-C69 floor-carry reader filters the row out (it is not A/A
    # evidence), so a skip can neither admit nor refuse a carried floor by itself.
    assert "comparison" not in verdict.to_dict()
    assert verdict.to_dict()["aa_skipped"] == "recent clean A/A"
    assert seen == [verdict]


def test_missing_digest_always_measures_even_if_skip_is_offered(tmp_path):
    compared = []
    verdict, _ = _verify(tmp_path, digests={"anchor": None, "scratch": "x" * 64},
                         aa_skip=lambda: "recent clean A/A", compared=compared)
    assert len(compared) == 1 and verdict.effect_pct is not None


def test_differing_digests_still_abort_and_never_consult_skip(tmp_path):
    asked = []
    with pytest.raises(loop.RunAborted):
        _verify(tmp_path, digests={"anchor": "x" * 64, "scratch": "y" * 64},
                aa_skip=lambda: asked.append(1) or "skip", compared=[])
    assert asked == []


def test_skip_returning_none_measures(tmp_path):
    compared = []
    verdict, _ = _verify(tmp_path, digests={"anchor": "x" * 64, "scratch": "x" * 64},
                         aa_skip=lambda: None, compared=compared)
    assert len(compared) == 1 and verdict.pairs == 5


# ------------------------------------------------------------- flags / binding

def test_cli_declares_both_schedules_with_defaults(capsys):
    with pytest.raises(SystemExit):
        run.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--accumulate-bench-every-keeps" in text and "(default: 3)" in text
    assert "--anchor-guard-aa-window-s" in text and "(default: 21600.0)" in text


@pytest.mark.parametrize("flag,value", [("--accumulate-bench-every-keeps", "0"),
                                        ("--anchor-guard-aa-window-s", "-1")])
def test_cli_refuses_invalid_schedule(flag, value):
    with pytest.raises(SystemExit):
        run.main([flag, value, "--dry-run"])


def test_schedule_flags_do_not_change_the_resume_binding(tmp_path):
    doc = tmp_path / "doc.json"
    doc.write_text("{}")
    base = ["--model", "/m.gguf", "--pairs", "5"]
    tuned = [*base, "--accumulate-bench-every-keeps", "1",
             "--anchor-guard-aa-window-s=0"]
    assert sr.resume_binding(base) == sr.resume_binding(tuned)
    assert sr.resume_binding(base) != sr.resume_binding([*base, "--fire-multiple", "3"])


def test_serial_roster_accepts_schedule_flags_as_common_args(tmp_path):
    from .test_serial_roster import _inputs
    from . import serial_roster
    _resolved, _owners, argv = _inputs(tmp_path, backends=("cpu",))
    common = tmp_path / "common.json"
    common.write_text(json.dumps(["--accumulate-bench-every-keeps", "3",
                                  "--anchor-guard-aa-window-s=21600"]))
    targets, _skipped, _cpus = serial_roster.build_targets(
        Path(sr.option(argv, "--resolved-campaign")), Path(sr.option(argv, "--owned-targets")),
        target_root=Path(sr.option(argv, "--state-dir")) / "targets", common_path=common)
    assert sr.option(targets[0], "--accumulate-bench-every-keeps") == "3"
    assert sr.option(targets[0], "--anchor-guard-aa-window-s") == "21600"
