"""Same-binary runtime execution through the existing loop; no model hardware."""
from contextlib import contextmanager
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from . import actors, archive, gates, loop, run, serving
from .test_legacy_cpu_serving import _requests, _server
from .unified_planner import RuntimeDimension, enumerate_runtime_dimensions


def _fixture(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    model = tmp_path / "not-a-model"
    model.write_bytes(b"fixture: no model is loaded")
    recipe = serving.Recipe(name="runtime-http", model=str(model), device="none", ngl=0,
                            np=2, n_predict=8, cpu_list=None)
    anchor, log, pids = _server(tmp_path / "same-build", recipe, port, 10.0)
    pair = enumerate_runtime_dimensions(anchor, (RuntimeDimension(
        "threads-test", "threads", recipe.threads, recipe.threads + 1,
        "original-hypothesis:threads-test"),))[0]
    return pair, log, pids


def test_actual_loop_runtime_http_without_author_build_or_source_commit(tmp_path):
    pair, log, pids = _fixture(tmp_path)
    hypothesis = loop.Hypothesis("threads-test", "thread overhead", "same rate", "runtime", "threads",
                                 runtime_pair=pair)
    held = []

    @contextmanager
    def tail():
        held.append(True)
        try:
            yield
        finally:
            held.append(False)

    def forbidden(*args, **kwargs):
        pytest.fail("runtime-only treatment reached source author/build/promotion")

    def measure(hypothesis, paths):
        assert held[-1] and paths == ()
        return run.ServingComparison(serving.compare(
            pair.anchor.template, Path(pair.anchor.build_dir), Path(pair.anchor.build_dir),
            pairs=1, floor_pct=None, port=pair.anchor.port,
            anchor_resolved_recipe=pair.anchor, candidate_resolved_recipe=pair.candidate,
            runtime_pair=pair, frozen_requests=_requests()))

    outcome = loop.iterate(
        planner=SimpleNamespace(propose=lambda context: hypothesis, author=forbidden),
        critic=SimpleNamespace(review_hypothesis=lambda *args: loop.Review(True), review_patch=forbidden),
        context={}, gate=lambda *args: (True, [gates.Verdict("correctness", True, "fixture only")]),
        measure=measure, commit=forbidden, tail_session=tail)
    assert outcome.status == "runtime_observed" and outcome.champion_head is None
    row = outcome.comparison.to_dict()
    assert row["decisive"] is None and row["noise_floor_pct"] is None
    assert row["runtime_pair"] == pair.to_dict()
    assert row["recipe_hash"] == pair.anchor.template.recipe_hash
    assert row["candidate_recipe_hash"] == pair.candidate.template.recipe_hash
    assert held == [True, False]
    assert len(pids.read_text().splitlines()) == 2
    bodies = [json.loads(line) for line in log.read_text().splitlines()]
    assert sorted(bodies) == sorted(body.hex() for _, body in _requests() for _ in range(4))
    assert archive.record(tmp_path / "memory", outcome.to_attempt(), epoch="fixture-epoch",
                          recorded_at=loop._now(), campaign_id="fixture-runtime")
    receipts = list((tmp_path / "memory/serving-beliefs").glob("*.json"))
    assert len(receipts) == 1
    assert row["belief_capture"]["inputs"]["protocol_id"] == ""


def test_runtime_cannot_borrow_source_floor_before_launch(tmp_path, monkeypatch):
    pair, _, pids = _fixture(tmp_path)
    monkeypatch.setattr(serving, "_measure_once", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(serving.ServingFloorMismatch, match="strict frame"):
        serving.compare(pair.anchor.template, Path(pair.anchor.build_dir), Path(pair.anchor.build_dir),
                        pairs=1, floor_pct=7.8, port=pair.anchor.port, runtime_pair=pair,
                        anchor_resolved_recipe=pair.anchor, candidate_resolved_recipe=pair.candidate)
    assert not pids.exists()


def test_planner_binds_candidate_to_original_context(tmp_path):
    pair, _, _ = _fixture(tmp_path)
    context = {"target": {"recipe": pair.anchor.to_dict()},
               "runtime_anchor": pair.anchor.to_dict(), "runtime_env_keys": []}
    # Use the existing CPU-target spelling, not a serialized launch permission.
    context["target"]["scope"] = "experimental candidate, NOT canonical champion"
    generated = actors._runtime_pair({"kind": "threads", "candidate": pair.candidate.template.threads},
                                      context, "threads-test")
    assert generated.to_dict() == pair.to_dict()
    with pytest.raises((actors.ProviderTransient, ValueError)):
        actors._runtime_pair({"kind": "threads", "candidate": pair.anchor.template.threads},
                             context, "no-op")
    with pytest.raises(actors.ProviderTransient, match="installed runtime keys"):
        actors._runtime_pair({"kind": "env", "candidate": {"key": "LLAMA_ARG_CTX_SIZE", "value": "1"}},
                             context, "workload-change")


def test_actual_main_cpu_runtime_five_iterations_no_source_change():
    from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion
    test_existing_main_cpu_five_iterations_preserves_canonical_champion(False, runtime_only=True)


@pytest.mark.parametrize("installed", [True, False])
def test_only_installed_runtime_option_skips_routine_critics(tmp_path, installed):
    pair, _, _ = _fixture(tmp_path)
    hypothesis = loop.Hypothesis("threads-test", "overhead", "same rate", "runtime", "threads",
                                 runtime_pair=pair)
    calls = []

    def review(*args):
        calls.append("hypothesis")
        return loop.Review(True)

    def forbidden(*args):
        pytest.fail("runtime treatment reached source author/diff/commit")

    comparison = run.ServingComparison({"effect": 0.0, "decisive": None,
        "noise_floor_pct": None, "metric": "aggregate_tok_s", "recipe": pair.anchor.template.name})
    outcome = loop.iterate(
        planner=SimpleNamespace(propose=lambda context: hypothesis, author=forbidden),
        critic=SimpleNamespace(review_hypothesis=review, review_patch=forbidden),
        context={"runtime_anchor": pair.anchor.to_dict(), "runtime_env_keys": []}
                if installed else {},
        gate=lambda *args: (True, []), measure=lambda *args: comparison, commit=forbidden)
    assert outcome.status == "runtime_observed"
    assert calls == ([] if installed else ["hypothesis"])
