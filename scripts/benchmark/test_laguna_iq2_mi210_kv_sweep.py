from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import laguna_iq2_mi210_kv_sweep as runner


def _args() -> SimpleNamespace:
    return SimpleNamespace(execute=False, binary=runner.DEFAULT_BINARY, source_root=runner.DEFAULT_SOURCE_ROOT,
                           target_model=runner.DEFAULT_TARGET_MODEL, reps=runner.REPS, context=runner.CONTEXT,
                           max_tokens=runner.MAX_TOKENS, seed=runner.SEED, startup_timeout=1, request_timeout=1)


def _record(prompt_id: str, index: int) -> dict:
    return {"prompt_id": prompt_id, "prompt_index": index, "finish_reason": "stop", "semantic_validation": {"passed": True},
            "response_sanity": {"passed": True}, "request_lifecycle": {"fully_contained_valid": True, "fully_contained_sample_count": 1}}


def _row(cell: str, rep: int) -> dict:
    return {"cell": cell, "rep": rep, "status": "ok", "records": [_record(name, index) for index, (name, _) in enumerate(runner.common.PROMPT_SPECS, 1)],
            "residency": {"passed": True}, "cleanup": {"dead": True}, "post_cleanup_clean": True, "post_cleanup_vram_settled": True,
            "prompt_ms": 100.0, "decode_ms": 200.0, "prompt_tps": 10.0, "decode_tps": 20.0}


def test_fixed_identities_and_plan_cardinality_counterbalance() -> None:
    plan = runner.build_plan(_args(), {"bytes": runner.TARGET_MODEL_BYTES, "sha256": runner.TARGET_MODEL_SHA256})
    assert runner.EXPECTED_HEAD == "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
    assert plan["observation_only"] is True and plan["promotion_gate"] is False
    assert len(plan["runs"]) == 15
    assert [(cell.name, cell.cache_k, cell.cache_v, cell.flash_attention) for cell in runner.CELLS] == [
        ("A_q8_kv_fa_on", "q8_0", "q8_0", True),
        ("B_f16_kv_fa_on", "f16", "f16", True),
        ("C_f16_kv_fa_off", "f16", "f16", False),
    ]
    assert [row["cell"] for row in plan["runs"][:6]] == ["A_q8_kv_fa_on", "B_f16_kv_fa_on", "C_f16_kv_fa_off", "B_f16_kv_fa_on", "C_f16_kv_fa_off", "A_q8_kv_fa_on"]
    assert plan["fixed_recipe"]["dflash"] == "forbidden"
    assert plan["candidate"]["binary"]["binary_sha256"] == runner.EXPECTED_SERVER_SHA256
    assert {
        item["soname"]: item["sha256"]
        for item in plan["candidate"]["binary"]["local_llama_ggml_libraries"]
    } == runner.EXPECTED_LOCAL_LIBRARY_SHA256
    assert "version: 10107" in (
        plan["candidate"]["binary"]["server_version"]["stdout"]
        + plan["candidate"]["binary"]["server_version"]["stderr"]
    )
    assert runner.fixed_identities_valid(
        {"head_matches": True, "clean": True},
        {"binary": str(runner.DEFAULT_BINARY), "binary_sha256": runner.EXPECTED_SERVER_SHA256, "artifact": {"stable": True},
         "server_version": {"returncode": 0, "stdout": "version: 10107"},
         "local_llama_ggml_libraries": [{"soname": soname, "sha256": digest, "stable": True} for soname, digest in runner.EXPECTED_LOCAL_LIBRARY_SHA256.items()]},
        {"path": str(runner.DEFAULT_TARGET_MODEL), "stable": True, "bytes": runner.TARGET_MODEL_BYTES, "sha256": runner.TARGET_MODEL_SHA256},
    ) == (True, "ok")


def test_git_state_clean_requires_success_and_empty_output() -> None:
    clean = {key: {"returncode": 0, "stdout": ""} for key in ("tracked_diff", "index_diff", "untracked")}
    assert runner.git_state_is_clean(clean) is True
    assert runner.git_state_is_clean({**clean, "tracked_diff": {"returncode": 0, "stdout": "changed.py\n"}}) is False
    assert runner.git_state_is_clean({**clean, "index_diff": {"returncode": 1, "stdout": ""}}) is False


def test_argv_differences_are_only_kv_and_flash_attention() -> None:
    args = _args()
    a, b, c = [runner.server_argv(args, cell, 20000) for cell in runner.CELLS]
    assert [a[a.index("--cache-type-k") + 1], a[a.index("--cache-type-v") + 1], a[a.index("-fa") + 1]] == ["q8_0", "q8_0", "on"]
    assert [b[b.index("--cache-type-k") + 1], b[b.index("--cache-type-v") + 1], b[b.index("-fa") + 1]] == ["f16", "f16", "on"]
    assert [c[c.index("--cache-type-k") + 1], c[c.index("--cache-type-v") + 1], c[c.index("-fa") + 1]] == ["f16", "f16", "off"]
    assert "-md" not in a + b + c


def test_summary_fails_closed_for_missing_cell_or_cleanup_or_residency() -> None:
    rows = [_row(cell.name, rep) for cell in runner.CELLS for rep in range(1, 6)]
    assert runner.matrix_valid(rows) == (True, "ok")
    summaries = {
        cell.name: runner.summarize_cell(
            [row for row in rows if row["cell"] == cell.name],
            cell.name,
        )
        for cell in runner.CELLS
    }
    summaries[runner.CELLS[0].name]["decode_tps"]["median"] = 21.0
    assert runner.bounded_best_observed(summaries)["cell"] == runner.CELLS[0].name
    comparison = runner.a_b_comparison(summaries, post_execution_identity_valid=True)
    assert comparison["status"] == "observed"
    assert comparison["decode_tps"]["b_over_a_ratio"] == 20.0 / 21.0
    assert comparison["prompt_tps"]["b_vs_a_percent"] == 0.0
    summaries[runner.CELLS[1].name]["all_ok"] = False
    unavailable = runner.a_b_comparison(summaries, post_execution_identity_valid=True)
    assert unavailable["status"] == "unavailable"
    assert unavailable["decode_tps"]["b_over_a_ratio"] is None
    assert unavailable["prompt_tps"]["b_vs_a_percent"] is None
    summaries[runner.CELLS[1].name]["all_ok"] = True
    assert runner.a_b_comparison(summaries, post_execution_identity_valid=False)["status"] == "unavailable"
    assert runner.bounded_best_observed(summaries, post_execution_identity_valid=False)["status"] == "unavailable"
    rows[-1]["cleanup"] = {"dead": False}
    assert runner.matrix_valid(rows)[0] is False
    rows[-1]["cleanup"] = {"dead": True}
    rows[-1]["post_cleanup_clean"] = False
    assert runner.matrix_valid(rows)[0] is False
    rows[-1]["post_cleanup_clean"] = True
    rows[-1]["records"] = rows[-1]["records"][:-1]
    assert runner.matrix_valid(rows)[0] is False
    rows[-1]["records"] = [_record(name, index) for index, (name, _) in enumerate(runner.common.PROMPT_SPECS, 1)]
    rows[-1]["records"][0]["prompt_id"] = "wrong"
    assert runner.matrix_valid(rows)[0] is False
    rows[-1]["records"][0]["prompt_id"] = runner.common.PROMPT_SPECS[0][0]
    rows[-1]["records"][0]["request_lifecycle"]["fully_contained_valid"] = False
    assert runner.matrix_valid(rows)[0] is False
    assert runner.summarize_cell([], runner.CELLS[0].name)["all_ok"] is False


def test_post_execution_identity_match_and_drift_gate() -> None:
    witness = {
        "source": {"head": "a"}, "binary": {"sha256": "b"}, "target_model": {"sha256": "c"},
        "harness": {"sha256": "d"}, "harness_snapshot": {"sha256": "d"}, "binding": {"server": "b"},
    }
    assert runner.identity_witness_matches(witness, dict(witness)) == (True, "ok")
    drifted = {**witness, "binary": {"sha256": "changed"}}
    valid, reason = runner.identity_witness_matches(witness, drifted)
    assert valid is False and "binary differs" in reason


def test_residency_requires_exact_target_full_offload_and_requested_kv() -> None:
    cell = runner.CELLS[0]
    good = f"loading model '{runner.DEFAULT_TARGET_MODEL}'\noffloaded 49/49 layers to GPU\nROCm0 model buffer size = 0.00 MiB\nROCm0 model buffer size = 35538.61 MiB\nK (q8_0): 1.00 MiB, V (q8_0): 1.00 MiB"
    assert runner.parse_log_residency(good, cell)["passed"] is True
    assert runner.parse_log_residency(good.replace("q8_0", "f16", 1), cell)["passed"] is False


def test_vram_settlement_polls_until_delayed_valid_sample(monkeypatch) -> None:
    snapshots = [{"sample": 1}, {"sample": 2}]
    settled = iter((False, True))
    monkeypatch.setattr(runner.common, "process_snapshot", lambda: {"clean": True})
    monkeypatch.setattr(runner.common, "process_guard_clean", lambda *_args: (True, "ok"))
    monkeypatch.setattr(runner.common, "collect_rocm_snapshot", lambda: snapshots.pop(0))
    monkeypatch.setattr(runner.common, "snapshot_is_valid", lambda _snapshot: True)
    monkeypatch.setattr(runner.common, "vram_settled", lambda *_args: next(settled))
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    success, reason, samples = runner.poll_vram_settlement({"before": True}, 20000, timeout_s=1, interval_s=0)
    assert (success, reason) == (True, "ok")
    assert [sample["vram_settled"] for sample in samples] == [False, True]
    assert all(sample["rocm_valid"] and sample["process_guard_clean"] for sample in samples)


def test_vram_settlement_fails_closed_at_deadline(monkeypatch) -> None:
    monkeypatch.setattr(runner.common, "process_snapshot", lambda: {"clean": True})
    monkeypatch.setattr(runner.common, "process_guard_clean", lambda *_args: (True, "ok"))
    monkeypatch.setattr(runner.common, "collect_rocm_snapshot", lambda: {"still": "high-vram"})
    monkeypatch.setattr(runner.common, "snapshot_is_valid", lambda _snapshot: True)
    monkeypatch.setattr(runner.common, "vram_settled", lambda *_args: False)
    success, reason, samples = runner.poll_vram_settlement({"before": True}, 20000, timeout_s=0, interval_s=0)
    assert success is False
    assert reason == "ROCm VRAM did not settle before deadline"
    assert len(samples) == 1 and samples[0]["vram_settled"] is False


def test_vram_settlement_does_not_wait_out_process_contamination(monkeypatch) -> None:
    monkeypatch.setattr(runner.common, "process_snapshot", lambda: {"contaminated": True})
    monkeypatch.setattr(runner.common, "process_guard_clean", lambda *_args: (False, "KFD owner present"))
    monkeypatch.setattr(runner.common, "collect_rocm_snapshot", lambda: {"settled": True})
    monkeypatch.setattr(runner.common, "snapshot_is_valid", lambda _snapshot: True)
    monkeypatch.setattr(runner.common, "vram_settled", lambda *_args: True)
    success, reason, samples = runner.poll_vram_settlement(
        {"before": True}, 20000, timeout_s=30, interval_s=1
    )
    assert (success, reason) == (False, "KFD owner present")
    assert len(samples) == 1 and samples[0]["process_guard_clean"] is False


def test_interrupt_during_startup_cleans_up_writes_partial_result_and_reraises(monkeypatch, tmp_path: Path) -> None:
    class FakeProcess:
        pid = 1234

    monkeypatch.setattr(runner.common, "collect_rocm_snapshot", lambda: {"capture": "before"})
    monkeypatch.setattr(runner.common, "snapshot_is_valid", lambda _snapshot: True)
    monkeypatch.setattr(runner.common, "process_snapshot", lambda: {"processes": "clean"})
    monkeypatch.setattr(runner.common, "process_guard_clean", lambda *_args: (True, "ok"))
    monkeypatch.setattr(runner.common, "vram_settled", lambda *_args: True)
    monkeypatch.setattr(runner.common, "terminate", lambda proc: {"pid": proc.pid, "dead": True, "process_group": {"returncode": 1}})
    monkeypatch.setattr(runner.common, "wait_for_health", lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    with pytest.raises(KeyboardInterrupt):
        runner.run_replicate(_args(), runner.CELLS[0], 1, 20000, tmp_path, {"server": {}, "models": {}})

    result = __import__("json").loads((tmp_path / "runs" / "A_q8_kv_fa_on_rep1" / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "interrupted"
    assert result["cleanup"]["dead"] is True
    assert result["post_cleanup_clean"] is True
    assert result["records"] == []


def test_interrupt_during_cleanup_evidence_writes_result_before_reraising(monkeypatch, tmp_path: Path) -> None:
    class FakeProcess:
        pid = 1234

    snapshots = iter(({"processes": "initial"}, KeyboardInterrupt()))

    def process_snapshot() -> dict:
        value = next(snapshots)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(runner.common, "collect_rocm_snapshot", lambda: {"capture": "before"})
    monkeypatch.setattr(runner.common, "snapshot_is_valid", lambda _snapshot: True)
    monkeypatch.setattr(runner.common, "process_snapshot", process_snapshot)
    monkeypatch.setattr(runner.common, "process_guard_clean", lambda *_args: (True, "ok"))
    monkeypatch.setattr(runner.common, "terminate", lambda proc: {"pid": proc.pid, "dead": True})
    monkeypatch.setattr(runner.common, "wait_for_health", lambda *_args: (_ for _ in ()).throw(RuntimeError("startup failed")))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    with pytest.raises(KeyboardInterrupt):
        runner.run_replicate(_args(), runner.CELLS[0], 1, 20000, tmp_path, {"server": {}, "models": {}})

    result = __import__("json").loads((tmp_path / "runs" / "A_q8_kv_fa_on_rep1" / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "cleanup_failed"
    assert result["cleanup"]["dead"] is True
    assert result["cleanup_evidence_error"] == "KeyboardInterrupt while collecting cleanup evidence"
