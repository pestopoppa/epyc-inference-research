"""Direct original-request profiling; synthetic perf and tiny HTTP server only."""
from contextlib import closing
import copy
from io import BytesIO
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from . import actors, cpu_profile as cp, measurement_capture as mc, planned_serving as ps, resolved_recipe as rr, serial_run as sr, status
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


def test_direct_loop_budget_covers_observed_glm53_perf_expansion():
    # The 2026-09-17 v20 warmup record was 93,760,128 bytes, but its complete
    # perf-script rendering was 114,397,198 bytes. The old 96 MiB parser cap
    # rejected the otherwise completed observation after exactly 96 MiB.
    budgets = cp.LOOP_BUDGETS
    assert budgets["max_raw_file_bytes"] > 93_760_128
    assert budgets["max_parser_bytes"] > 114_397_198
    reserved = (2 * budgets["max_raw_file_bytes"] + 2 * budgets["max_parser_bytes"]
        + 2 * (cp.ns.MAX_REQUEST_BYTES + cp.ns.MAX_RESPONSE_BYTES)
        + 9 * cp.MAX_DIAGNOSTIC_BYTES + budgets["max_metadata_bytes"])
    assert budgets["max_total_raw_bytes"] >= reserved


def test_direct_original_token_cache_request_and_raw_reopen(tmp_path):
    result, config, events = produce(tmp_path)
    assert result["status"] == "observed"
    assert result["hotspots"][0]["symbol"] == "synthetic_kernel"
    assert result["hotspots"][0]["sampled_period_fraction"] == 1.0
    assert result["location_attribution"]["sampled_tid_count"] == 1
    assert result["location_attribution"]["execution_nodes"]
    assert result["location_attribution"]["low_high_sync_threads"][0]["sampled_cpus"]
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
        forged = copy.deepcopy(cp._plain(body))
        forged["phases"][1]["topology"]["cpu_to_numa_node"][str(
            result["location_attribution"]["low_high_sync_threads"][0]["sampled_cpus"][0])] += 1
        with pytest.raises(cp.CpuProfileRefused, match="topology digest"):
            cp._reopen_phases(forged, forged["settings"])
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


def test_original_profile_reference_reuses_only_exact_identity_and_refuses_tamper(tmp_path):
    result, _config, _events = produce(tmp_path)
    commit = "a" * 40
    result["anchor_commit"] = commit
    reference = sr.cpu_profile_reference(result, store=tmp_path / "direct",
        anchor_commit=commit, scope="half")
    assert reference["record"]["sha256"] == result["record_sha256"]
    assert sr.cpu_profile_reference(result, store=tmp_path / "direct",
        anchor_commit="b" * 40, scope="half") is None
    assert sr.cpu_profile_reference(result, store=tmp_path / "direct",
        anchor_commit=commit, scope="full_confirmation")["scope"] == "full_confirmation"
    kwargs = {"store_root": tmp_path / "direct", "anchor_commit": commit,
        "execution_digest": result["execution_digest"],
        "prompt_manifest_digest": result["prompt_manifest_digest"], "scope": "half"}
    assert cp.cached_loop_observation(reference, **kwargs)["ranked_levers"] == result["ranked_levers"]
    for changed in ({"anchor_commit": "b" * 40},
                    {"execution_digest": "b" * 64},
                    {"prompt_manifest_digest": "b" * 64},
                    {"scope": "full"}):
        with patch.object(cp, "loop_observation", side_effect=AssertionError("stale profile reopened")):
            assert cp.cached_loop_observation(reference, **(kwargs | changed)) is None
    with pytest.raises(sr.SerialRefused, match="identity"):
        sr._cpu_profile_reference(reference, store=tmp_path / "direct",
            anchor_commit="b" * 40, scope="half")
    path = Path(result["record"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(sr.SerialRefused, match="original record changed"):
        sr._cpu_profile_reference(reference, store=tmp_path / "direct",
            anchor_commit=commit, scope="half")


def test_profile_continuation_is_optional_and_old_batch_still_reopens(tmp_path):
    result, config, _events = produce(tmp_path)
    result["anchor_commit"] = "a" * 40
    recipe = rr.resolved_recipe_from_dict(config["resolved_recipe"])
    launch, prompts, out = (tmp_path / name for name in
                            ("launch.json", "prompts.json", "batch"))
    launch.write_text(json.dumps(recipe.to_dict()))
    prompts.write_text(json.dumps(config["prompt_manifest"]))
    argv = ["--worktree", str(tmp_path), "--model", recipe.model.path,
        "--anchor-build", str(recipe.build_dir), "--store", str(tmp_path / "direct"),
        "--experimental-branch", "ak/experimental/profile-fixture",
        "--cpu-serving-launch", str(launch), "--frozen-prompts", str(prompts),
        "--iterations", "1", "--out", str(out)]
    reference = sr.cpu_profile_reference(result, store=tmp_path / "direct",
        anchor_commit="a" * 40, scope="full")
    row = sr.continuation(argv=argv, binding=sr.input_binding(argv), terminal="complete",
        worktree=tmp_path, branch="ak/experimental/profile-fixture",
        model=recipe.model.path, selected_target=None,
        anchor_build=recipe.build_dir, anchor_commit="a" * 40,
        iterations_requested=1, outcomes=[type("Outcome", (), {"status": "abstained"})()],
        cpu_profile_reference=reference)
    status.write_json(out, "loop-run.json", {"continuation": row})
    status.write_json(out, "loop-continuation.json", row)
    assert sr.load_completed(out / "loop-continuation.json")[0]["cpu_profile_reference"] == reference
    old = {key: value for key, value in row.items() if key != "cpu_profile_reference"}
    status.write_json(out, "loop-continuation.json", old)
    assert "cpu_profile_reference" not in sr.load_completed(out / "loop-continuation.json")[0]


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
            "period": 400, "sampled_period_fraction": 0.25}],
        "ranked_levers": [{"family": "quantized-matmul-q4", "period": 400,
            "sampled_period_fraction": 0.25, "symbols": [],
            "evidence_kind": "current-request-sampled-user-cycles"}],
        "location_attribution": {"sampled_tid_count": 2, "active_tid_count": 2,
            "active_period_cutoff": 50.0,
            "execution_nodes": [{"numa_node": 1, "sampled_period_fraction": 0.6,
                "sync_fraction_within_node": 0.3}],
            "low_high_sync_threads": [{"tid": 12, "sampled_cpus": [7],
                "execution_nodes": [1], "sync_fraction_within_tid": 0.3}]}}}
    with patch.object(actors, "_run_agent", side_effect=provider):
        actors.AgentPlanner(tmp_path).propose(context)
        context["cpu_profile"] = {"status": "unavailable", "reason": "perf permission denied"}
        actors.AgentPlanner(tmp_path).propose(context)
    assert "synthetic_kernel" in emitted[0] and "25.00% | 400" in emitted[0]
    assert "quantized-matmul-q4" in emitted[0]
    assert "highest-share unresolved causal mechanism" in emitted[0]
    assert "Where sampled threads executed" in emitted[0]
    assert "not measure remote-memory traffic" in emitted[0]
    assert "/original/cpu-profile.json" in emitted[0]
    assert "not exact CPU cost, wall-time share" in emitted[0]
    assert "| share | ns | calls |" not in emitted[0] and "no profile yet" not in emitted[0]
    assert "CPU profile unavailable: perf permission denied" in emitted[1]


def test_sampled_cpu_reduction_is_request_scoped_and_rejects_missing_location():
    rows = (b"11/12 [007] 1.000000000: 100 cycles:u: abc ggml_barrier (/tmp/lib.so)\n"
            b"11/12 [007] 2.000000000: 200 cycles:u: abc ggml_vec_dot_q8_0_q8_0 (/tmp/lib.so)\n"
            b"11/13 [009] 2.100000000: 300 cycles:u: abc ggml_barrier (/tmp/lib.so)\n"
            b"11/13 [009] 4.000000000: 400 cycles:u: abc ggml_barrier (/tmp/lib.so)\n")
    kwargs = {"pid": 11, "tids": {12, 13}, "interval": [1.0, 3.0],
        "max_bytes": 4096, "max_rows": 16, "max_symbols": 16}
    result = cp.reduce_perf_script(BytesIO(rows), cpu_to_node={"7": 0, "9": 1}, **kwargs)
    assert result["sampled_period_total"] == 600
    assert result["outside_request_samples"] == 1
    assert result["location_periods"] == [
        {"tid": 12, "cpu": 7, "numa_node": 0,
         "family": "dense-q8-dot-matmul", "period": 200},
        {"tid": 12, "cpu": 7, "numa_node": 0,
         "family": "thread-synchronization-and-work-balance", "period": 100},
        {"tid": 13, "cpu": 9, "numa_node": 1,
         "family": "thread-synchronization-and-work-balance", "period": 300}]
    projection = cp.location_attribution(result)
    assert [x["numa_node"] for x in projection["execution_nodes"]] == [0, 1]
    assert projection["execution_nodes"][1]["sync_fraction_within_node"] == 1.0
    with pytest.raises(cp.CpuProfileRefused, match="no captured NUMA mapping"):
        cp.reduce_perf_script(BytesIO(rows), cpu_to_node={"7": 0}, **kwargs)
    with pytest.raises(cp.CpuProfileRefused, match="unsupported/lost/malformed"):
        cp.reduce_perf_script(BytesIO(rows.replace(b" [007]", b"")),
                              cpu_to_node={"7": 0, "9": 1}, **kwargs)


def test_ranked_levers_aggregate_symbols_without_model_hardcoding():
    rows = [
        {"dso": "/build/libggml-cpu.so", "symbol": "ggml_vec_dot_q8_0_q8_0", "period": 270},
        {"dso": "/build/libggml-cpu.so", "symbol":
         "mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>", "period": 200},
        {"dso": "/build/libggml-cpu.so", "symbol": "ggml_barrier", "period": 120},
        {"dso": "/usr/lib/libgomp.so", "symbol": "[unknown]", "period": 80},
    ]
    ranked = cp.ranked_levers(rows, 1000)
    assert [row["family"] for row in ranked[:3]] == [
        "dense-q8-dot-matmul", "quantized-matmul-q4",
        "thread-synchronization-and-work-balance"]
    assert ranked[2]["sampled_period_fraction"] == pytest.approx(0.2)
