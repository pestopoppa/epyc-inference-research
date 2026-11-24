"""AK-EVAL-FUT: futility-only interim looks. Pure replay + synthetic arms only."""
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from . import bench, cpu_screen, futility, gates, loop, run, serving
from .test_matched_serving import _floor, _isolated_gpu_exclusion  # noqa: F401
from .test_resolved_recipe import BUILD, _resolve

FIXTURE = Path(__file__).parent / "fixtures" / "futility_replay_20261124.json"


def _group(name):
    rows = json.loads(FIXTURE.read_text())["groups"][name]
    sigma = statistics.median(statistics.stdev(r["log_ratios_pct"]) for r in rows)
    return rows, sigma


@pytest.mark.parametrize("group,z", [("ds41_serving", futility.Z_SERVING),
                                     ("q38fn_serving", futility.Z_SERVING),
                                     ("gpu_bench_loop_memory", futility.Z_BENCH)])
def test_replay_saves_28pct_and_stops_no_winner(group, z):
    rows, sigma = _group(group)
    result = futility.replay(rows, sigma_pair_pct=sigma, z=z)
    assert result["saved_frac"] >= 0.28, result
    assert result["winners_stopped"] == 0, result


def test_gpu_era_rejects_the_lower_z():
    rows, sigma = _group("gpu_bench_loop_memory")
    assert futility.replay(rows, sigma_pair_pct=sigma, z=2.5)["winners_stopped"] > 0


def test_look_never_answers_and_waits_for_min_pairs():
    assert futility.look([-9.0], sigma_pair_pct=1.0, bar_pct=1.0, z=2.5)["decision"] == "continue"
    assert futility.look([-9.0, -9.0], sigma_pair_pct=1.0, bar_pct=1.0, z=2.5)["decision"] == "futile"
    # A huge win only ever continues: there is no stop-on-answer.
    assert futility.look([50.0, 50.0], sigma_pair_pct=1.0, bar_pct=1.0, z=2.5)["decision"] == "continue"
    with pytest.raises(ValueError):
        futility.Rule(1.0, 1.0, 2.5, min_pairs=1)
    assert futility.Rule(1.0, 1.0, 2.5).plan()["stop_on_answer"] is False


def _fake_run_once(values):
    def run_once(binary, model, **_):
        return values[str(binary)], {"resident": True, "peak_vram_bytes": 1 << 31,
                                     "peak_kfd_processes": 1}
    return run_once


def _bench(candidate_value, rule):
    with mock.patch.object(bench, "run_once",
                           side_effect=_fake_run_once({"/a": 100.0, "/c": candidate_value})):
        return bench.compare(bench.Arm("a", Path("/a")), bench.Arm("c", Path("/c")),
                             Path("/m.gguf"), pp=512, tg=0, pairs=5, noise_floor_pct=1.0,
                             warmup_pairs=0, futility=rule)


def test_bench_futile_stop_is_a_null_that_never_reaches_keep():
    rule = futility.Rule(sigma_pair_pct=0.5, bar_pct=1.0, z=3.5)
    lost = _bench(90.0, rule)
    assert lost.stop_reason == "futility" and len(lost.anchor_samples) == 2
    assert lost.decisive is False and lost.effect < 0      # not even a regression
    row = lost.to_dict()
    assert row["stopping"] == futility.STOPPING and row["pairs_run"] == 2 and row["pairs"] == 5
    assert "FUTILE" in loop._null_reason(lost)
    # A small positive effect stopped as futile must not ride the accumulate path.
    small = _bench(100.2, futility.Rule(sigma_pair_pct=0.1, bar_pct=1.0, z=2.5))
    assert small.stop_reason == "futility" and small.effect > 0 and small.decisive is False
    winner = _bench(105.0, rule)
    assert winner.stop_reason is None and len(winner.anchor_samples) == 5 and winner.decisive


def test_iterate_never_commits_a_positive_futile_stop():
    small = _bench(100.2, futility.Rule(sigma_pair_pct=0.1, bar_pct=1.0, z=2.5))
    hypothesis = loop.Hypothesis("fut", "statement", "falsifier", "pp512", "ggml/src/x.c")
    paths = ("ggml/src/x.c",)
    planner = mock.Mock()
    planner.propose.return_value = hypothesis
    planner.author.return_value = paths
    critic = cpu_screen.RetainedCritic({"hypothesis": hypothesis.to_dict(), "paths": list(paths)})
    commit = mock.Mock()
    outcome = loop.iterate(planner=planner, critic=critic, context={},
        measure=lambda _h, _p: small,
        gate=lambda _h, _p: (True, [gates.Verdict("oracle", True)]),
        commit=commit, accumulate_valid_positive=True)
    assert outcome.status == "measured_null" and "FUTILE" in " ".join(outcome.reasons)
    commit.assert_not_called()


def test_serving_futile_stop_records_plan_and_matches_instrument(monkeypatch):
    recipe, launch, requests, row, calls = _floor(monkeypatch)
    sigma = futility.sigma_from_aa_samples(row["anchor_samples"], row["candidate_samples"])
    rule = futility.Rule(sigma_pair_pct=sigma, bar_pct=row["floor_pct"], z=futility.Z_SERVING)

    def measure(recipe_, build, port, *, evidence, **kwargs):
        evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                         "status": "not_applicable", "window_start": time.time(),
                         "window_end": time.time(), "samples": 0})
        return 5.0 if str(build) == "/candidate" else 10.0
    monkeypatch.setattr(serving, "_measure_once", measure)
    out = serving.compare(recipe, BUILD, Path("/candidate"), pairs=5, floor_pct=row["floor_pct"],
        floor_unit=serving.CALIBRATION_UNIT, floor_record=row,
        instrument=serving.MATCHED_INSTRUMENT, anchor_resolved_recipe=launch,
        candidate_resolved_recipe=_resolve(recipe, backend="cpu", build=Path("/candidate")),
        frozen_requests=requests,
        floor_request_digest=serving.request_digest(recipe, requests), futility=rule)
    assert out["stop_reason"] == "futility" and out["pairs_run"] == 2 and out["decisive"] is False
    plan = out["measurement_plan"]
    assert plan["stopping"] == futility.STOPPING and plan["futility"]["z"] == futility.Z_SERVING
    assert serving.comparison_instrument_matches(out, instrument=serving.MATCHED_INSTRUMENT, pairs=5)
    tampered = {**out, "measurement_plan": {**plan, "futility": {**plan["futility"], "stop_on_answer": True}}}
    assert not serving.comparison_instrument_matches(tampered, instrument=serving.MATCHED_INSTRUMENT, pairs=5)
    assert run.ServingComparison(out).stop_reason == "futility"
    with pytest.raises(serving.RecipeError):
        serving.compare(recipe, BUILD, Path("/candidate"), pairs=5, floor_pct=None,
            instrument=serving.MATCHED_INSTRUMENT, anchor_resolved_recipe=launch,
            candidate_resolved_recipe=_resolve(recipe, backend="cpu", build=Path("/candidate")),
            frozen_requests=requests, futility=rule)


def test_run_rule_is_on_by_default_and_never_on_an_uncalibrated_bar():
    from . import run_args
    parser = run_args.build_parser([], description="t", run_module=run)
    assert parser.get_default("futility_looks") == "on"
    args = SimpleNamespace(futility_looks="off", futility_z=None, futility_min_pairs=2)
    assert run._futility_rule(args, 1.0, record={}) is None
    del args.futility_looks   # absent attribute resolves to the default (on): still no rule without a bar
    assert run._futility_rule(args, None, on_bench=True) is None
    args.futility_looks = "on"
    assert run._futility_rule(args, None, on_bench=True) is None
    args.serving_instrument = serving.MATCHED_INSTRUMENT
    record = {"anchor_samples": [10.0, 10.1, 9.9, 10.0], "candidate_samples": [10.0, 9.9, 10.1, 10.2]}
    rule = run._futility_rule(args, 1.5, record=record)
    assert rule.z == futility.Z_SERVING and rule.bar_pct == 1.5
    assert rule.sigma_source == "matched_serving_floor_aa_samples"


def test_retained_comparison_accepts_a_futile_stop_but_not_a_short_fixed_run():
    from .comparison_validation import ComparisonRefused, validate_comparison
    base = {"pairs": 5, "anchor_samples": [10.0, 10.1], "candidate_samples": [9.0, 9.1]}
    validate_comparison({**base, "stop_reason": "futility", "pairs_run": 2}, require_provenance=False)
    with pytest.raises(ComparisonRefused):
        validate_comparison(base, require_provenance=False)
    with pytest.raises(ComparisonRefused):
        validate_comparison({**base, "stop_reason": "futility", "pairs_run": 3}, require_provenance=False)
