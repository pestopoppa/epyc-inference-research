"""Established runtime combinations retain an exact effective dispatch witness."""
from types import SimpleNamespace
import json
import subprocess
import sys

import pytest

from . import actors, gates, loop, runtime_combinations
from .test_runtime_arms import _launch
from .unified_planner import RuntimeArmPair


def test_combination_skips_critic_and_dispatches_witnessed_launch(tmp_path):
    anchor = _launch(tmp_path / "build")
    context = {"runtime_anchor": anchor.to_dict(), "runtime_env_keys": ["OMP_PLACES"]}
    pair = actors._runtime_pair({"kind": "combination", "candidate": [
        {"kind": "threads", "candidate": 32},
        {"kind": "env", "candidate": {"key": "OMP_PLACES", "value": "threads"}},
    ]}, context, "combo")
    assert RuntimeArmPair.from_dict(pair.to_dict()) == pair
    assert pair.anchor.executable == pair.candidate.executable
    assert pair.anchor.dsos == pair.candidate.dsos
    assert runtime_combinations.established(pair.dimension, ["OMP_PLACES"])
    assert not runtime_combinations.established(pair.dimension, [])
    hypothesis = loop.Hypothesis("combo", "combined options", "no benefit", "runtime", "combo",
                                 runtime_pair=pair)
    seen = []

    def forbidden(*args, **kwargs):
        pytest.fail("established combination called a critic, author, or build")

    def gate(hypothesis, paths):
        # A short-lived fixture chooses and reports a route from the actual argv
        # and environment it receives; this is not hardware evidence.
        for launch in (pair.anchor, pair.candidate):
            argv = launch.command_argv
            env = dict(launch.launch_env)
            child = subprocess.run(
                [sys.executable, "-c", "import json,os,sys; "
                 "t=int(sys.argv[sys.argv.index('-t')+1]); "
                 "p=os.environ['OMP_PLACES']; print(json.dumps({"
                 "'threads':t,'omp_places':p,'dispatch':'fixture-team'+str(t)+'-'+p}))",
                 *argv[1:]],
                env=env, check=True, text=True, capture_output=True)
            seen.append(json.loads(child.stdout))
        return True, [gates.Verdict("correctness", True, "fixture")]

    outcome = loop.iterate(
        planner=SimpleNamespace(propose=lambda _: hypothesis, author=forbidden),
        critic=SimpleNamespace(review_hypothesis=forbidden, review_patch=forbidden),
        context=context, gate=gate,
        measure=lambda *_: SimpleNamespace(decisive=None, to_dict=lambda: {}), commit=forbidden)
    assert outcome.status == "runtime_observed"
    assert seen == [{"threads": 48, "omp_places": "cores", "dispatch": "fixture-team48-cores"},
                    {"threads": 32, "omp_places": "threads", "dispatch": "fixture-team32-threads"}]


def test_combination_rejects_duplicate_or_uninstalled_dimensions(tmp_path):
    anchor = _launch(tmp_path / "build")
    context = {"runtime_anchor": anchor.to_dict(), "runtime_env_keys": ["OMP_PLACES"]}
    env = {"kind": "env", "candidate": {"key": "OMP_PLACES", "value": "threads"}}
    with pytest.raises(actors.ProviderTransient, match="repeats"):
        actors._runtime_pair({"kind": "combination", "candidate": [env, env]}, context, "dup")
    with pytest.raises(actors.ProviderTransient, match="installed runtime keys"):
        actors._runtime_pair({"kind": "combination", "candidate": [env,
            {"kind": "env", "candidate": {"key": "UNDECLARED", "value": "1"}}]},
            context, "unknown")


def test_multi_option_pair_cannot_be_mislabeled_single_factor_a2(tmp_path):
    from . import discovery_screen
    from .test_discovery_screen import _context, _plan
    anchor = _launch(tmp_path / "build")
    context = {"runtime_anchor": anchor.to_dict(), "runtime_env_keys": ["OMP_PLACES"]}
    pair = actors._runtime_pair({"kind": "combination", "candidate": [
        {"kind": "threads", "candidate": 32},
        {"kind": "env", "candidate": {"key": "OMP_PLACES", "value": "threads"}},
    ]}, context, "combo")
    plan = _plan(pair)
    with pytest.raises(discovery_screen.DiscoveryScreenRefused, match="single-factor"):
        discovery_screen._frame(plan, pair, _context(plan))


def test_combination_preserves_explicit_environment_unset(tmp_path):
    anchor = _launch(tmp_path / "build")
    context = {"runtime_anchor": anchor.to_dict(), "runtime_env_keys": ["OMP_PLACES"]}
    pair = actors._runtime_pair({"kind": "combination", "candidate": [
        {"kind": "threads", "candidate": 32},
        {"kind": "env", "candidate": {"key": "OMP_PLACES", "value": None}},
    ]}, context, "unset")
    assert "OMP_PLACES" not in dict(pair.candidate.launch_env)
    assert dict(pair.anchor.launch_env)["OMP_PLACES"] == "cores"
    assert RuntimeArmPair.from_dict(pair.to_dict()) == pair


def test_uninstalled_combo_never_gets_critic_bypass(tmp_path):
    anchor = _launch(tmp_path / "build")
    context = {"runtime_anchor": anchor.to_dict(), "runtime_env_keys": ["OMP_PLACES"]}
    pair = actors._runtime_pair({"kind": "combination", "candidate": [
        {"kind": "threads", "candidate": 32},
        {"kind": "env", "candidate": {"key": "OMP_PLACES", "value": "threads"}},
    ]}, context, "combo")
    hypothesis = loop.Hypothesis("combo", "combined options", "no benefit", "runtime", "combo",
                                 runtime_pair=pair)
    calls = []

    def review(*_):
        calls.append(True)
        return loop.Review(False, "not installed")

    def forbidden(*_args, **_kwargs):
        pytest.fail("uninstalled combination reached execution")

    recorded = []
    loop.iterate(planner=SimpleNamespace(propose=lambda _: hypothesis, author=forbidden),
                 critic=SimpleNamespace(review_hypothesis=review, review_patch=forbidden),
                 context={**context, "runtime_env_keys": []}, hypothesis_rounds=1,
                 gate=forbidden, measure=forbidden, commit=forbidden,
                 record_abandoned=recorded.append)
    assert calls == [True]
    assert recorded[0].status == "hypothesis_rejected"


def test_combo_resolution_failure_refuses_before_any_runtime_execution(tmp_path, monkeypatch):
    from . import unified_planner
    anchor = _launch(tmp_path / "build")
    context = {"runtime_anchor": anchor.to_dict(), "runtime_env_keys": ["OMP_PLACES"]}
    original = unified_planner._resolve_variant

    def fail_combination(base, dimension, value):
        if dimension.kind == "combination":
            raise unified_planner.PlanningRefused("injected combination resolution failure")
        return original(base, dimension, value)

    monkeypatch.setattr(unified_planner, "_resolve_variant", fail_combination)
    with pytest.raises(actors.ProviderTransient, match="injected combination"):
        actors._runtime_pair({"kind": "combination", "candidate": [
            {"kind": "threads", "candidate": 32},
            {"kind": "env", "candidate": {"key": "OMP_PLACES", "value": "threads"}},
        ]}, context, "combo")
