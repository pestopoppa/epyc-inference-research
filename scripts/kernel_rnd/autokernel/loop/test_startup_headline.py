"""Restart recovery through real main/refresh; all device edges are doubles."""
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from . import accumulate, bench, gates, headline_admissibility, hotspots, pool, production, run
from . import test_promotion_targets as fixture_module


@pytest.mark.parametrize("carryover", [False, True])
@pytest.mark.parametrize("measurement_fails", [False, True])
def test_restart_refresh_is_claimed_before_lanes_and_failure_still_dispatches(
        carryover, measurement_fails):
    fixture = fixture_module.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    events = []
    held = {"claim": False, "quiet": False}
    model = fixture.root / f"{bench.MEASURED_FLOOR_MODEL_STEM}.gguf"
    clock = lambda: datetime(2026, 10, 30, tzinfo=timezone.utc)
    stamp = lambda: clock().isoformat().replace("+00:00", "Z")
    label = "production-consolidated-v10"
    samples = [100.0] * headline_admissibility.MIN_LAUNCHES_PER_ARM
    accumulate.Bundle(champion_of_record=fixture.tip, tip=fixture.tip).save(fixture.store)
    # A prior refresh against a different freeze is precisely the restart defect.
    published = fixture.store / production.FILENAME
    published.write_text(json.dumps({"schema": production.SCHEMA,
        "baseline": {"commit": "0" * 40}, "champion": {"commit": fixture.tip}}))
    stale = published.read_bytes()
    if carryover:
        production.write_baseline(fixture.store, production_commit=fixture.tip,
            production_label=label, measured_commit=fixture.tip, samples=samples,
            protocol=production.protocol(model=model, surface="pp512"),
            measured_at=stamp(), source={"fixture": True}, host_facts={},
            host_facts_note="fixture", now=stamp)

    @contextmanager
    def claim(*args, **kwargs):
        held["claim"] = True
        events.append("claim")
        try:
            yield {"device_id": "fixture"}
        finally:
            held["claim"] = False
            events.append("release")

    @contextmanager
    def quiet(*args, **kwargs):
        assert held["claim"]
        held["quiet"] = True
        try:
            yield
        finally:
            held["quiet"] = False

    def build(source_root, build_dir, **kwargs):
        assert held["claim"] and not held["quiet"]
        assert Path(source_root) == fixture.repo
        events.append("build")
        binary = Path(build_dir) / "bin/llama-bench"
        binary.parent.mkdir(parents=True)
        binary.write_text("fixture")
        return gates.Verdict("compile", True)

    def compare(base, champion, measured_model, **kwargs):
        assert held["claim"] and held["quiet"]
        assert champion.binary == fixture.startup_anchor / "bin/llama-bench"
        events.append("compare")
        if measurement_fails:
            raise RuntimeError("fixture measurement unavailable")
        return bench.Comparison(surface=kwargs["surface"], anchor_samples=samples,
            candidate_samples=[105.0] * len(samples), effect=0.05,
            estimator="median_over_median", pairs=len(samples),
            noise_floor_pct=kwargs["noise_floor_pct"], residency={}, model=str(measured_model))

    def measure(champion, measured_model, **kwargs):
        assert held["claim"] and held["quiet"]
        assert champion.binary == fixture.startup_anchor / "bin/llama-bench"
        events.append("measure")
        if measurement_fails:
            raise RuntimeError("fixture measurement unavailable")
        return bench.Measurement(surface=kwargs["surface"], samples=[105.0] * len(samples),
                                 model=str(measured_model), launches=len(samples), residency={})

    actual_refresh = production.refresh

    def refresh(**kwargs):
        assert held["claim"]
        assert kwargs["champion_commit"] == fixture.tip
        assert kwargs["champion_build"] == fixture.startup_anchor
        events.append("refresh")
        outcome = actual_refresh(**kwargs, resolve=lambda: (fixture.tip, label),
            baseline_root=fixture.root / "baseline", host_facts=lambda: {},
            host_changes=(), clock=clock, now=stamp)
        assert outcome.published is not measurement_fails, outcome.reason
        events.append("refreshed")
        return outcome

    def provision(*args, **kwargs):
        assert held["claim"] and events[-1] == "refreshed"
        if measurement_fails:
            assert published.read_bytes() == stale
        else:
            body = json.loads(published.read_text())
            assert body["baseline"]["commit"] == fixture.tip
            assert body["champion"]["commit"] == fixture.tip
            assert body["baseline_carryover"]["used"] is carryover
        events.append("lanes")
        return []

    argv = ["--worktree", str(fixture.repo), "--anchor-build", str(fixture.startup_anchor),
        "--model", str(model), "--store", str(fixture.store), "--iterations", "1",
        "--workers", "1", "--worker-root", str(fixture.root / "lanes"),
        "--worker-build-root", str(fixture.root / "builds"),
        "--allow-untested-actor-seats", "fixture: no actor is invoked"]
    try:
        with mock.patch.object(run.claim, "hold", claim), \
             mock.patch.object(run, "_gpu_quiet_measurement_window", quiet), \
             mock.patch.object(gates, "compiles", build), \
             mock.patch.object(production, "BASELINE_TREE", fixture.repo), \
             mock.patch.object(production, "refresh", refresh), \
             mock.patch.object(bench, "compare", compare), \
             mock.patch.object(bench, "measure", measure), \
             mock.patch.object(run.hotspots, "profile", side_effect=hotspots.ProfileFailed("fixture")), \
             mock.patch.object(run.workload_contract, "verify_workload",
                               return_value=SimpleNamespace(n_embd=1536, dominant_quant="Q4_K")), \
             mock.patch.object(pool, "provision", provision), \
             mock.patch.object(pool, "drive", return_value=pool.PoolResult(outcomes=[])), \
             redirect_stdout(io.StringIO()):
            assert run.main(argv) == 0
        assert events == (["claim", "refresh", "measure", "refreshed", "lanes", "release"]
                          if carryover else
                          ["claim", "refresh", "build", "compare", "refreshed", "lanes", "release"])
    finally:
        fixture.doCleanups()
