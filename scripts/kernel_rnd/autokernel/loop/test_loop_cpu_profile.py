"""Direct original-request profiling; synthetic perf and tiny HTTP server only."""
from contextlib import closing
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from . import actors, cpu_profile as cp, measurement_capture as mc, planned_serving as ps, resolved_recipe as rr
from .test_cpu_profile_runtime import fixture
from .test_glm_frozen_requests import _request
from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion as _run


def produce(tmp_path, *, failure=""):
    _, _, _, config, events, _ = fixture(tmp_path, original_request=_request(), perf_failure=failure)
    recipe = rr.resolved_recipe_from_dict(config["resolved_recipe"])
    prompts = ps.FrozenPromptManifest.from_dict(config["prompt_manifest"])
    (tmp_path / "direct").mkdir()
    result = cp.profile_loop(recipe, prompts, store_root=tmp_path / "direct",
        perf_path=config["profiler"]["path"], timeout_s=20,
        server_interpreter=config["profiler"]["server_interpreter"])
    return result, config, events


def test_direct_original_token_cache_request_and_raw_reopen(tmp_path):
    result, config, events = produce(tmp_path)
    assert result["status"] == "observed"
    assert result["hotspots"][0]["symbol"] == "synthetic_kernel"
    assert result["hotspots"][0]["sampled_period_fraction"] == 1.0
    path = Path(result["record"])
    reference = {"locator": path.name, "sha256": result["record_sha256"], "verified": True}
    with closing(mc.ArtifactStore(path.parent)) as store:
        body = cp.reopen_loop_profile(reference, store=store)
        assert "loaded_identity" not in body and "model_verification" not in body
        assert body["settings"]["resolved_recipe"] == cp.ob._freeze(config["resolved_recipe"])
        for phase in body["phases"]:
            assert json.loads(bytes.fromhex(phase["response"]["request_hex"])) == _request()
            assert phase["observed_predicted_n"] == 512
            assert phase["tools"][0]["identity"]["pid"] != body["processes"]["server"]["pid"]
        record = json.loads(path.read_text())
        assert record["profile_claim_tuple"]["protocol_id"] == ""
        assert "validation_claim_tuple" not in record
        script = Path(body["phases"][1]["script"]["path"])
        script.write_bytes(script.read_bytes() + b"changed")
        with pytest.raises(cp.CpuProfileRefused):
            cp.reopen_loop_profile(reference, store=store)
    for line in events.read_text().splitlines():
        with pytest.raises(ProcessLookupError):
            os.kill(int(line.split()[1]), 0)


def test_permission_denial_has_no_live_child(tmp_path):
    with pytest.raises(cp.CpuProfileRefused):
        produce(tmp_path, failure="denied")
    for line in (tmp_path / "child-events").read_text().splitlines():
        with pytest.raises(ProcessLookupError):
            os.kill(int(line.split()[1]), 0)


def test_existing_loop_profiles_start_and_keep_not_null_or_calibration():
    seen, contexts = [], []
    def observed(recipe, prompts, **kwargs):
        assert json.loads(prompts.prompts[0].body) == _request()
        seen.append(recipe)
        return {"status": "observed", "execution_digest": recipe.execution_digest,
            "hotspots": [{"symbol": "synthetic", "period": 100, "sampled_period_fraction": 1.0}],
            "record": "synthetic-test-only"}
    _run(False, profile_observer=observed, profile_contexts=contexts)
    assert len(seen) == 2  # startup + one actual source keep; four nulls do not reprofile
    assert seen[0].execution_digest != seen[1].execution_digest
    assert len(contexts) == 5
    assert [row["execution_digest"] for row in contexts] == [seen[0].execution_digest] * 4 + [seen[1].execution_digest]


def test_unavailable_is_visible_and_uncertain_cleanup_stops_before_actor():
    contexts = []
    _run(False, profile_contexts=contexts)
    assert all(row["status"] == "unavailable" and "test-only" in row["reason"] for row in contexts)
    capture = object.__new__(cp.CpuProfileCapture)
    capture.cleanup_uncertain = "synthetic owned child terminal unknown"
    with pytest.raises(cp.CpuProfileCleanupUncertain):
        capture.finish()
    contexts = []
    def uncertain(*args, **kwargs):
        raise cp.CpuProfileCleanupUncertain("synthetic owned child terminal unknown")
    with pytest.raises(cp.CpuProfileCleanupUncertain):
        _run(False, profile_observer=uncertain, profile_contexts=contexts)
    assert contexts == []  # no actor request or subsequent measurement after uncertainty


def test_actual_planner_prompt_contains_sampled_symbols_or_unavailable_reason(tmp_path):
    emitted = []
    response = {"mechanism_id": "cpu-render-fixture", "statement": "synthetic test mechanism",
        "falsifier": "synthetic result does not improve", "target_surface": "cpu",
        "target_symbol": "synthetic_kernel"}
    def provider(prompt, **kwargs):
        emitted.append(prompt)
        return json.dumps(response)
    context = {"target": {"recipe": {"backend": "cpu"}}, "cpu_profile": {
        "status": "observed", "record": "/original/cpu-profile.json", "record_sha256": "b" * 64,
        "execution_digest": "a" * 64, "prompt_manifest_digest": "c" * 64,
        "hotspots": [{"dso": "libggml-cpu.so", "symbol": "synthetic_kernel",
            "period": 400, "sampled_period_fraction": 0.25}]}}
    with patch.object(actors, "_run_agent", side_effect=provider):
        actors.AgentPlanner(tmp_path).propose(context)
        context["cpu_profile"] = {"status": "unavailable", "reason": "perf permission denied"}
        actors.AgentPlanner(tmp_path).propose(context)
    assert "synthetic_kernel" in emitted[0] and "25.00% | 400" in emitted[0]
    assert "/original/cpu-profile.json" in emitted[0]
    assert "not exact CPU cost, wall-time share" in emitted[0]
    assert "| share | ns | calls |" not in emitted[0] and "no profile yet" not in emitted[0]
    assert "CPU profile unavailable: perf permission denied" in emitted[1]
