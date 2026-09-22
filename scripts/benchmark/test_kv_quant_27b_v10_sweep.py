from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import kv_quant_27b_v10_sweep as runner


def _args(**overrides) -> SimpleNamespace:
    base = dict(execute=False, probe_binary=False, target_model=runner.DEFAULT_TARGET_MODEL,
                reps=runner.REPS, context=runner.CONTEXT, max_tokens=runner.MAX_TOKENS,
                seed=runner.SEED, startup_timeout=1, request_timeout=1)
    base.update(overrides)
    return SimpleNamespace(**base)


def _binary() -> Path:
    return Path("/mnt/raid0/llm/kernels/production/gpu/llama-server")


def _plan() -> dict:
    model = {"path": str(runner.DEFAULT_TARGET_MODEL), "bytes": runner.TARGET_MODEL_BYTES,
             "sha256": None, "stable": None}
    return runner.build_plan(_args(), _binary(), model, probe=False)


# --------------------------------------------------------------------------- guards
def test_flash_attention_is_fixed_and_a_varied_matrix_is_refused() -> None:
    plan = _plan()
    assert {run["argv"][run["argv"].index("-fa") + 1] for run in plan["runs"]} == {"on"}
    assert runner.flash_attention_guard([run["argv"] for run in plan["runs"]]) == (True, "ok")

    varied = [list(plan["runs"][0]["argv"])]
    varied[0][varied[0].index("-fa") + 1] = "off"
    ok, reason = runner.flash_attention_guard(varied)
    assert ok is False
    assert "1.52x" in reason and "flash attention, not KV" in reason

    doubled = list(plan["runs"][0]["argv"]) + ["-fa", "on"]
    assert runner.flash_attention_guard([doubled])[0] is False


def test_cells_cannot_carry_a_flash_attention_setting() -> None:
    assert not hasattr(runner.CELLS[0], "flash_attention")
    assert runner.FLASH_ATTENTION == "on"


def test_mixed_kv_is_refused_with_the_fa_all_quants_reason() -> None:
    assert runner.homogeneous_kv_guard() == (True, "ok")
    mixed = (runner.Cell("X_mixed", "q4_0", "f16"),)
    ok, reason = runner.homogeneous_kv_guard(mixed)
    assert ok is False
    assert "GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF" in reason
    assert "watchdog-killed" in reason
    assert "X_mixed" in reason


def test_every_arm_is_homogeneous_and_there_are_exactly_three() -> None:
    assert [(cell.name, cell.cache_k, cell.cache_v) for cell in runner.CELLS] == [
        ("A_f16_kv", "f16", "f16"),
        ("B_q8_0_kv", "q8_0", "q8_0"),
        ("C_q4_0_kv", "q4_0", "q4_0"),
    ]
    assert all(cell.cache_k == cell.cache_v for cell in runner.CELLS)


# --------------------------------------------------------------------------- plan
def test_plan_cardinality_counterbalance_and_launch_count() -> None:
    plan = _plan()
    assert runner.TOTAL_LAUNCHES == 30
    assert plan["total_launches"] == 30 == len(plan["runs"])
    assert plan["reps_per_cell_per_depth"] == 5
    assert plan["fresh_server_per_replicate"] is True
    assert len({run["port"] for run in plan["runs"]}) == 30
    for cell in runner.CELLS:
        for depth in runner.DEPTHS:
            matching = [run for run in plan["runs"] if run["cell"] == cell.name and run["depth"] == depth.name]
            assert sorted(run["rep"] for run in matching) == [1, 2, 3, 4, 5]
    # No arm and no depth is always first.
    firsts = [plan["runs"][index]["cell"] for index in range(0, 30, 6)]
    assert len(set(firsts)) > 1
    depth_firsts = [plan["runs"][index]["depth"] for index in range(0, 30, 6)]
    assert set(depth_firsts) == {depth.name for depth in runner.DEPTHS}


def test_two_depths_short_and_deep() -> None:
    assert [depth.target_prefill_tokens for depth in runner.DEPTHS] == [2048, 32768]
    for depth in runner.DEPTHS:
        assert depth.min_prefill_tokens < depth.target_prefill_tokens < depth.max_prefill_tokens


def test_depth_prompt_is_deterministic_and_lands_in_band() -> None:
    prompt_id, text = runner.common.PROMPT_SPECS[0]
    for depth in runner.DEPTHS:
        first = runner.depth_prompt(depth, text)
        assert first == runner.depth_prompt(depth, text)
        estimated = len(first) / runner.CHARS_PER_TOKEN
        assert depth.min_prefill_tokens <= estimated <= depth.max_prefill_tokens
        assert text in first
        assert "REFERENCE LOG" in first
    assert prompt_id == "primes"


def test_argv_shape_is_the_declared_launch_and_carries_no_drafter() -> None:
    argv = runner.server_argv(_binary(), runner.CELLS[2], 19999, runner.CONTEXT, runner.SEED)
    assert argv[argv.index("-c") + 1] == "65536"
    assert argv[argv.index("-ngl") + 1] == "all"
    assert argv[argv.index("-dev") + 1] == "ROCm0"
    assert argv[argv.index("-fa") + 1] == "on"
    assert argv[argv.index("--cache-type-k") + 1] == "q4_0"
    assert argv[argv.index("--cache-type-v") + 1] == "q4_0"
    assert argv[argv.index("--temp") + 1] == "0"
    assert argv[argv.index("--top-k") + 1] == "1"
    assert argv[argv.index("--seed") + 1] == str(runner.SEED)
    assert "-md" not in argv and "--spec-type" not in argv
    assert str(runner.DEFAULT_TARGET_MODEL) in argv


def test_request_body_disables_prompt_cache() -> None:
    body = runner.request_body("hello", 512, 7)
    assert body["cache_prompt"] is False
    assert body["temperature"] == 0 and body["top_k"] == 1 and body["seed"] == 7


# --------------------------------------------------------------------------- kernel store
def test_store_resolution_points_at_the_production_kernel_not_the_frozen_tree() -> None:
    resolution = runner.store_resolution()
    assert resolution["store_path"] == "/mnt/raid0/llm/kernels/production/gpu"
    assert resolution["resolved"] is True and resolution["binary_exists"] is True
    assert "/mnt/raid0/llm/kernels/builds/" in resolution["resolved_bin_dir"]
    for forbidden in runner.FORBIDDEN_BINARY_PARENTS:
        assert resolution["resolved_bin_dir"] != str(forbidden)
    assert str(runner.FROZEN_SOURCE_TREE / "build-hip" / "bin") in resolution["forbidden_parents"]


def test_resolved_binary_matches_the_pinned_v10_digest() -> None:
    identity = runner.binary_identity(runner.resolved_binary(), probe=False)
    assert identity["binary_sha256_matches"] is True
    assert identity["expected_version_line"] == "version: 10303 (ffc1bac82)"
    assert identity["probed"] is False
    assert "deferred" in identity["server_version"]


def test_dry_run_identity_defers_every_execution_derived_field() -> None:
    identity = runner.binary_identity(runner.resolved_binary(), probe=False)
    assert identity["version_line_matches"] is None
    assert "deferred" in identity["linkage_receipt"]


def test_source_cleanliness_allows_only_non_build_untracked_paths() -> None:
    clean = {key: {"returncode": 0, "stdout": ""} for key in ("tracked_diff", "index_diff", "untracked")}
    assert runner.git_state_is_clean(clean) is True
    assert runner.git_state_is_clean({**clean, "tracked_diff": {"returncode": 0, "stdout": "ggml/src/x.cu\n"}}) is False
    assert runner.git_state_is_clean({**clean, "index_diff": {"returncode": 1, "stdout": ""}}) is False
    allowed = {**clean, "untracked": {"returncode": 0,
                                      "stdout": ".gitnexusignore\ntools/math-tools/external/eigen/\n"}}
    assert runner.git_state_is_clean(allowed) is True
    rogue = {**clean, "untracked": {"returncode": 0, "stdout": "ggml/src/rogue.cu\n"}}
    assert runner.git_state_is_clean(rogue) is False


def test_the_live_production_tree_passes_the_cleanliness_contract() -> None:
    assert runner.source_identity()["clean"] is True


# --------------------------------------------------------------------------- linkage receipt
def test_linkage_receipt_must_be_a_verifier_verdict_not_an_env_string() -> None:
    libraries = [{"soname": soname, "sha256": digest, "stable": True}
                 for soname, digest in runner.EXPECTED_LIBRARY_SHA256.items()]
    good = {"verdict": "pass", "pass_line_present": True, "inspected_count": len(libraries),
            "required_sonames_satisfied": True, "inspected_libraries": libraries}
    assert runner.linkage_receipt_valid(good) == (True, "ok")
    assert runner.linkage_receipt_valid({**good, "verdict": "vacuous"})[0] is False
    vacuous = {**good, "inspected_count": 0, "inspected_libraries": []}
    assert "vacuous" in runner.linkage_receipt_valid(vacuous)[1]
    assert runner.linkage_receipt_valid({**good, "required_sonames_satisfied": False})[0] is False
    drifted = [dict(row) for row in libraries]
    drifted[0]["sha256"] = "0" * 64
    assert "sha256 differs" in runner.linkage_receipt_valid({**good, "inspected_libraries": drifted})[1]
    stranger = libraries + [{"soname": "libggml-rogue.so.0", "sha256": "1" * 64, "stable": True}]
    assert "not in the pinned v10 set" in runner.linkage_receipt_valid(
        {**good, "inspected_libraries": stranger, "inspected_count": len(stranger)})[1]


# --------------------------------------------------------------------------- residency / arm identity
def test_residency_requires_full_offload_and_the_requested_kv_types() -> None:
    cell = runner.CELLS[1]
    good = (f"loading model '{runner.DEFAULT_TARGET_MODEL}'\n"
            "offloaded 65/65 layers to GPU\n"
            "ROCm0 model buffer size = 27000.00 MiB\n"
            "K (q8_0): 1088.00 MiB, V (q8_0): 1088.00 MiB")
    parsed = runner.parse_log_residency(good, cell)
    assert parsed["passed"] is True
    assert parsed["kv_k_mib"] == 1088.0 and parsed["kv_v_mib"] == 1088.0
    assert parsed["kv_buffer_total_mib"] == 2176.0

    wrong_arm = good.replace("K (q8_0)", "K (f16)")
    mis = runner.parse_log_residency(wrong_arm, cell)
    assert mis["passed"] is False and mis["unexpected_kv_type_lines"]

    partial = good.replace("offloaded 65/65", "offloaded 60/65")
    assert runner.parse_log_residency(partial, cell)["passed"] is False
    assert runner.parse_log_residency(good.replace("offloaded 65/65 layers to GPU\n", ""), cell)["passed"] is False


# --------------------------------------------------------------------------- record gates
def _response(prompt_tokens: int) -> dict:
    content = ("Enumerate the primes below thirty in ascending order and add them together. "
               "RESULT_JSON: {\"primes\":[2,3,5,7,11,13,17,19,23,29],\"sum\":129}")
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 128},
            "timings": {"prompt_ms": 500.0, "predicted_ms": 1000.0,
                        "prompt_per_second": 100.0, "predicted_per_second": 128.0}}


def test_prefill_depth_out_of_band_is_refused() -> None:
    lifecycle = {"fully_contained_valid": True, "fully_contained_sample_count": 2}
    depth = runner.DEPTHS[1]
    record = runner.record_from_response(_response(32000), "primes", 1, lifecycle, depth)
    assert record["prompt_tokens"] == 32000 and record["depth"] == "d32k"
    with pytest.raises(RuntimeError, match="out of band"):
        runner.record_from_response(_response(2100), "primes", 1, lifecycle, depth)


def test_completion_token_floor_runs_before_timing_counts() -> None:
    lifecycle = {"fully_contained_valid": True, "fully_contained_sample_count": 1}
    response = _response(2000)
    response["usage"]["completion_tokens"] = 4
    with pytest.raises(RuntimeError, match="completion token floor"):
        runner.record_from_response(response, "primes", 1, lifecycle, runner.DEPTHS[0])


def test_semantic_gate_rejects_a_wrong_answer() -> None:
    lifecycle = {"fully_contained_valid": True, "fully_contained_sample_count": 1}
    response = _response(2000)
    response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"].replace("129", "130")
    with pytest.raises(RuntimeError, match="sanity/semantic gate failed"):
        runner.record_from_response(response, "primes", 1, lifecycle, runner.DEPTHS[0])


# --------------------------------------------------------------------------- matrix / summaries
def _record(prompt_id: str, index: int) -> dict:
    return {"prompt_id": prompt_id, "prompt_index": index, "finish_reason": "stop",
            "semantic_validation": {"passed": True}, "response_sanity": {"passed": True},
            "request_lifecycle": {"fully_contained_valid": True, "fully_contained_sample_count": 1}}


def _row(cell: str, depth: str, rep: int, decode: float = 20.0) -> dict:
    return {"cell": cell, "depth": depth, "rep": rep, "status": "ok",
            "records": [_record(name, index) for index, (name, _) in enumerate(runner.common.PROMPT_SPECS, 1)],
            "residency": {"passed": True}, "cleanup": {"dead": True}, "post_cleanup_clean": True,
            "post_cleanup_vram_settled": True, "prompt_ms": 100.0, "decode_ms": 200.0,
            "prompt_tokens": 2000, "completion_tokens": 300, "prompt_tps": 10.0, "decode_tps": decode,
            "kv_buffer_total_mib": 2176.0, "kv_k_mib": 1088.0, "kv_v_mib": 1088.0}


def _rows(decode_by_cell: dict[str, float] | None = None) -> list[dict]:
    decode_by_cell = decode_by_cell or {}
    return [_row(cell.name, depth.name, rep, decode_by_cell.get(cell.name, 20.0))
            for cell in runner.CELLS for depth in runner.DEPTHS for rep in range(1, 6)]


def test_matrix_is_fail_closed_on_every_missing_proof() -> None:
    rows = _rows()
    assert runner.matrix_valid(rows, 5) == (True, "ok")
    assert runner.matrix_valid(rows[:-1], 5)[0] is False
    for mutation in ({"cleanup": {"dead": False}}, {"post_cleanup_clean": False},
                     {"post_cleanup_vram_settled": False}, {"residency": {"passed": False}},
                     {"status": "cleanup_failed"}):
        broken = _rows()
        broken[-1].update(mutation)
        assert runner.matrix_valid(broken, 5)[0] is False
    truncated = _rows()
    truncated[-1]["records"] = truncated[-1]["records"][:-1]
    assert runner.matrix_valid(truncated, 5)[0] is False


def test_kv_cost_is_reported_against_the_f16_reference_arm() -> None:
    rows = _rows({"A_f16_kv": 20.0, "B_q8_0_kv": 22.0, "C_q4_0_kv": 18.0})
    summaries = {
        f"{cell.name}|{depth.name}": runner.summarize_cell(
            [row for row in rows if row["cell"] == cell.name and row["depth"] == depth.name],
            cell.name, depth.name, 5)
        for cell in runner.CELLS for depth in runner.DEPTHS
    }
    comparison = runner.kv_cost_comparison(summaries, "d32k", True)
    assert comparison["status"] == "observed"
    assert comparison["baseline_cell"] == "A_f16_kv"
    assert comparison["arms"]["B_q8_0_kv"]["decode_tps"]["arm_over_reference_ratio"] == 22.0 / 20.0
    assert comparison["arms"]["C_q4_0_kv"]["decode_tps"]["arm_vs_reference_percent"] == pytest.approx(-10.0)
    assert runner.kv_cost_comparison(summaries, "d32k", False)["status"] == "unavailable"
    summaries["B_q8_0_kv|d32k"]["all_ok"] = False
    assert runner.kv_cost_comparison(summaries, "d32k", True)["status"] == "unavailable"


def test_identity_drift_after_a_complete_matrix_invalidates_it() -> None:
    witness = {"source": {"head": "a"}, "binary": {"sha256": "b"}, "target_model": {"sha256": "c"},
               "harness": {"sha256": "d"}, "harness_snapshot": {"sha256": "d"},
               "kernel_store": {"resolved_bin_dir": "e"}}
    assert runner.identity_witness_matches(witness, dict(witness)) == (True, "ok")
    ok, reason = runner.identity_witness_matches(witness, {**witness, "kernel_store": {"resolved_bin_dir": "f"}})
    assert ok is False and "kernel_store differs" in reason


# --------------------------------------------------------------------------- belief kernel write side
def _summary(status: str = "ok", kernel_proven: bool = True) -> dict:
    rows = _rows({"A_f16_kv": 20.0, "B_q8_0_kv": 22.0, "C_q4_0_kv": 18.0})
    summaries = {
        f"{cell.name}|{depth.name}": runner.summarize_cell(
            [row for row in rows if row["cell"] == cell.name and row["depth"] == depth.name],
            cell.name, depth.name, 5)
        for cell in runner.CELLS for depth in runner.DEPTHS
    }
    return {
        "status": status, "n": 5, "production_named_kernel": True,
        "candidate": {"binary": {"version_line_matches": kernel_proven, "binary_sha256": "a" * 64,
                                 "linkage_receipt": {"verdict": "pass"}}},
        "kernel_store": {"resolved_bin_dir": "/mnt/raid0/llm/kernels/builds/gpu-20260921-ffc1bac82/bin"},
        "cell_summaries": summaries,
        "kv_cost_by_depth": {depth.name: runner.kv_cost_comparison(summaries, depth.name, True)
                             for depth in runner.DEPTHS},
        "target_model": {"path": str(runner.DEFAULT_TARGET_MODEL)},
        "hardware_state": {"gpu_product": {}},
        "device_claim": {"survived_window": True},
        "warmup_discard_policy": "x", "cpu_interference_policy": "y",
        "exact_plan": {"fixed_recipe": {"context": runner.CONTEXT}},
    }


def _capture_rows(**kwargs) -> list[dict]:
    return runner.belief_capture_rows(_summary(**kwargs), run_id="run-TEST",
                                      scored_path="data/gpu-mi210/x/summary.json",
                                      scored_sha256="b" * 64,
                                      emitted_at="2026-09-22T12:00:00+00:00")


def test_capture_emits_one_valid_row_per_arm_depth_metric() -> None:
    rows = _capture_rows()
    assert len(rows) == len(runner.CELLS) * len(runner.DEPTHS) * 2 == 12
    assert all(runner.validate_row(row) == [] for row in rows)
    assert {row["metric"] for row in rows} == {"gpu_decode_tps", "gpu_prefill_tps"}
    assert {row["protocol_id"] for row in rows} == {"P-GPU-1"}
    assert {row["reps"] for row in rows} == {5}
    assert len({row["measurement_id"] for row in rows}) == 12
    assert all(row["measurement_id"].startswith("kvq_") for row in rows)


def test_reference_arm_is_baseline_and_quantized_arms_are_candidates() -> None:
    rows = _capture_rows()
    by_cell = {row["extra"]["arm"]["cache_k"]: row["category"] for row in rows}
    assert by_cell["f16"] == "BASELINE"
    assert by_cell["q8_0"] == "CANDIDATE" and by_cell["q4_0"] == "CANDIDATE"


def test_claim_text_carries_the_figure_and_the_attestation() -> None:
    row = next(row for row in _capture_rows() if row["metric"] == "gpu_decode_tps")
    assert "t/s" in row["claim"]
    assert "[P-GPU-1, n=5, 2026-09-22, attest " in row["claim"]
    assert "duty_cycle=bursty" in row["claim"]


def test_k_and_v_are_reported_separately() -> None:
    row = _capture_rows()[0]
    assert row["extra"]["kv_buffer_k_mib"]["median"] == 1088.0
    assert row["extra"]["kv_buffer_v_mib"]["median"] == 1088.0
    assert "kv_buffer_total_mib" in row["extra"]


def test_a_failed_matrix_emits_zero_rows_and_never_back_fills() -> None:
    assert _capture_rows(status="failed") == []


def test_an_unproven_kernel_drops_the_protocol_rather_than_manufacturing_one() -> None:
    rows = _capture_rows(kernel_proven=False)
    assert {row["protocol_id"] for row in rows} == {""}
    assert all(row["extra"]["protocol_omitted_reason"] for row in rows)
    assert all(runner.validate_row(row) == [] for row in rows)


def test_row_self_hash_refuses_a_mutated_row() -> None:
    row = dict(_capture_rows()[0])
    row["value"] = 999.0
    problems = runner.validate_row(row)
    assert "row_sha256 does not match the row body" in problems


def test_validate_row_is_the_shared_contract_and_catches_structural_defects() -> None:
    assert runner.validate_row("not a row") == ["row is not an object"]
    assert "missing keys" in runner.validate_row({})[0]
    row = dict(_capture_rows()[0])
    for key, value, fragment in (("category", "GREAT", "category must be one of"),
                                 ("reps", 0, "reps must be a positive int"),
                                 ("metric", "vibes", "is not one this producer emits"),
                                 ("scored_sha256", "short", "64 hex"),
                                 ("claim", "  ", "claim text must carry the figure"),
                                 ("date", "yesterday", "date must be YYYY-MM-DD")):
        broken = dict(row)
        broken[key] = value
        broken["row_sha256"] = runner.row_digest(broken)
        assert any(fragment in problem for problem in runner.validate_row(broken)), key


def test_the_producer_never_grades() -> None:
    """An adapter projects; claim_tuple.grade() decides. No ladder may live here."""
    # Scan CODE only: the module's prose is allowed to name the rules it defers to,
    # otherwise the guard would forbid its own explanation.
    code = "\n".join(line.split("#", 1)[0] for line in
                     Path(runner.__file__).read_text(encoding="utf-8").splitlines())
    for level in ("\"Witnessed\"", "\"Verified\"", "\"Judged\"", "\"Attested\"", "\"Anchored\""):
        assert level not in code
    assert "register_ladder(" not in code
    assert "def grade(" not in code


def test_sidecar_is_written_atomically_beside_the_scored_artifact(tmp_path: Path) -> None:
    summary = _summary()
    summary_path = tmp_path / "summary.json"
    runner.write_json(summary_path, summary)
    receipt = runner.write_belief_measurements(tmp_path, summary, summary_path)
    assert receipt["written"] is True and receipt["rows"] == 12
    assert receipt["schema"] == "epyc.vidya.kv_quant_27b_v10_capture.v1"
    sidecar = tmp_path / "belief_measurements.jsonl"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    for line in lines:
        row = json.loads(line)
        assert runner.validate_row(row) == []
        assert row["scored_sha256"] == receipt["scored_sha256"]
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------- CLI
def test_cli_refuses_to_vary_the_fixed_recipe() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--reps", "3"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--context", "4096"])
    args = runner.parse_args([])
    assert args.execute is False and args.probe_binary is False


def test_help_states_the_launch_count(capsys) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--help"])
    text = capsys.readouterr().out
    assert "30 fresh llama-server launches" in text
    assert "30 fresh llama-server launches" in runner.__doc__.replace("\n", " ") or "= 30" in text


def test_render_matrix_prints_every_arm_and_argv() -> None:
    rendered = runner.render_matrix(_plan())
    assert rendered.count("/mnt/raid0/llm/kernels/production/gpu/llama-server") >= 30
    for cell in runner.CELLS:
        assert cell.name in rendered
    assert "-fa on FIXED in every arm" in rendered
