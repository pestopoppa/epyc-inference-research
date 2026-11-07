"""Provider circuit breaker (provider_breaker.py) and its terminal plumbing."""
import json
from types import SimpleNamespace

import pytest

from . import campaign, pipeline, provider_breaker as pb, serial_run as sr, serial_scheduling


def transient(reason="codex: failed 4 times"):
    return SimpleNamespace(status="planner_transient", reasons=[reason])


def author_failure(*outcomes):
    return SimpleNamespace(status="authoring_harness_failure", reasons=["x"],
                           hypothesis_pending={"authoring_failures": [
                               {"outcome": outcome} for outcome in outcomes]})


def ok(status="measured_null"):
    return SimpleNamespace(status=status, reasons=[])


def test_streak_of_n_provider_transients_parks():
    breaker = pb.ProviderBreaker(3)
    assert [breaker.observe(transient()) for _ in range(3)] == [False, False, True]
    assert breaker.tripped and breaker.streak == 3
    assert breaker.last_reason == "codex: failed 4 times"
    assert not breaker.observe(transient())     # trips exactly once
    assert pb.terminal(stop_asked=False, breaker=breaker) == pb.PROVIDER_PARKED


def test_n_minus_one_then_success_resets():
    breaker = pb.ProviderBreaker(3)
    for _ in range(2):
        breaker.observe(transient())
    breaker.observe(ok())
    assert breaker.streak == 0
    for _ in range(2):
        breaker.observe(transient())
    assert not breaker.tripped
    assert pb.terminal(stop_asked=False, breaker=breaker) == "complete"


@pytest.mark.parametrize("status", ["abstained", "hypothesis_rejected", "gate_refused",
                                    "patch_rejected", "regression", "bench_failed"])
def test_science_outcomes_never_count_and_reset_the_streak(status):
    breaker = pb.ProviderBreaker(2)
    breaker.observe(transient())
    breaker.observe(ok(status))
    breaker.observe(transient())
    assert not breaker.tripped and breaker.streak == 1


@pytest.mark.parametrize("status", ["stopped_mid_formation", "lane_error"])
def test_neutral_outcomes_neither_count_nor_reset(status):
    breaker = pb.ProviderBreaker(2)
    breaker.observe(transient())
    breaker.observe(ok(status))
    assert breaker.streak == 1
    breaker.observe(transient())
    assert breaker.tripped


def test_author_provider_transient_counts_but_genuine_authoring_failure_resets():
    assert pb.classify(author_failure("transient", "transient")) == "provider"
    assert pb.classify(author_failure("transient", "abstained")) == "success"
    assert pb.classify(author_failure()) == "success"


def test_operator_stop_wins_over_park():
    breaker = pb.ProviderBreaker(1)
    breaker.observe(transient())
    assert pb.terminal(stop_asked=True, breaker=breaker) == "stopped"


def test_tripped_breaker_stops_dispatch_at_iteration_boundary():
    breaker = pb.ProviderBreaker(2)
    budget = pipeline.Budget(10, should_stop=lambda: breaker.tripped)
    drawn = 0
    while budget.take():
        drawn += 1
        breaker.observe(transient())
    assert drawn == 2 and budget.remaining == 8


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8", None])
def test_threshold_validation(value):
    with pytest.raises(ValueError):
        pb.validate_threshold(value)


def _resources(**extra):
    return {"schema": campaign.RESOURCE_SCHEMA, "cpu_logical": [0], "gpu_ids": [],
            "stage_timeout_s": 60, "build_timeout_s": 60, "build_jobs": 1,
            "max_builds": 1, **extra}


def test_campaign_resources_config_surface():
    default = campaign.ResourceRequest.from_dict(_resources())
    assert default.provider_park_after is None
    assert default.effective_provider_park_after == pb.DEFAULT_PARK_AFTER
    assert "provider_park_after" not in default.to_dict()   # digests unchanged
    set_ = campaign.ResourceRequest.from_dict(_resources(provider_park_after=3))
    assert set_.effective_provider_park_after == 3
    assert campaign.ResourceRequest.from_dict(set_.to_dict()) == set_
    for bad in (0, -2, True, "3", 2.0):
        with pytest.raises(campaign.ManifestError):
            campaign.ResourceRequest.from_dict(_resources(provider_park_after=bad))


def test_parked_child_is_a_scheduler_stop():
    assert serial_scheduling.one_iteration_outcome(pb.PROVIDER_PARKED, {}) == "failed"
    assert serial_scheduling.stage_outcome(pb.PROVIDER_PARKED, {}, 3) == "failed"


def test_parked_continuation_loads_for_resume(tmp_path):
    out, worktree = tmp_path / "out", tmp_path / "wt"
    out.mkdir(), worktree.mkdir()
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    argv = ["--worktree", str(worktree), "--model", str(model), "--out", str(out),
            "--iterations", "10"]
    row = sr.continuation(
        argv=argv, binding=sr.input_binding(argv), terminal=pb.PROVIDER_PARKED,
        worktree=worktree, branch=sr.champion.CANONICAL_BRANCH, model=model,
        selected_target=None, anchor_build=tmp_path, anchor_commit="a" * 40,
        cor_build=tmp_path, cor_commit="a" * 40, iterations_requested=10,
        outcomes=[transient() for _ in range(8)])
    (out / "loop-run.json").write_text(json.dumps({"iterations": []}))
    path = out / "loop-continuation.json"
    path.write_text(json.dumps(row))
    prior, _sha = sr.load_resume(path, argv)
    assert prior["terminal"] == pb.PROVIDER_PARKED
    assert prior["iterations_completed"] == 8
