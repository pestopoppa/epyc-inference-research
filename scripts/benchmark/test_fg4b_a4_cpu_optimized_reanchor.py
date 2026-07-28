from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fg4b_a4_cpu_optimized_reanchor as runner


def test_git_identity_uses_absolute_trusted_binary() -> None:
    assert runner.GIT == Path("/usr/bin/git")
    assert runner.GIT.is_file()


def test_command_is_production_shaped_and_rejects_legacy_bench_shape() -> None:
    command = runner.build_server_command()
    assert command[:3] == ["taskset", "-c", runner.CPU_LIST]
    assert "numactl" not in command
    assert "llama-bench" not in " ".join(command)
    server_index = command.index(str(runner.LLAMA_SERVER))
    for option, value in (("-c", "32768"), ("-ub", "8192"), ("-ctk", "q8_0"), ("-ctv", "q8_0"), ("--spec-type", "draft-mtp"), ("--spec-draft-n-max", "4")):
        index = command.index(option, server_index)
        assert command[index + 1] == value
    assert "--reasoning" in command and command[command.index("--reasoning") + 1] == "off"


def test_dry_run_never_spawns_and_has_no_registry_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "start_server", lambda *_: pytest.fail("dry run spawned"))
    payload = runner.dry_run_payload(runner.parse_args([]))
    assert payload["registry_mutation"] is False
    assert payload["required_regions"] == ["q0", "q1"]
    assert payload["metric"] == "timings.predicted_per_second"
    assert payload["decision_grade"] is False
    assert payload["protocol_status"] == "candidate_protocol_pending_ratification"
    assert payload["proposal_apply_eligible"] is False


def holder_rows(*, tag: str = "fg4b", pid: int = 123, role: str = "bench") -> list[dict]:
    holder = {
        "role": role,
        "request_tag": tag,
        "pid": pid,
        "regions": ["q0", "q1"],
    }
    return [
        {
            "region": region,
            "global_held": region in {"q0", "q1"},
            "holders": [holder] if region in {"q0", "q1"} else [],
        }
        for region in ("q0", "q1", "q2", "q3")
    ]


def test_region_check_rejects_q2_only_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = holder_rows()
    for row in rows:
        row["global_held"] = row["region"] == "q2"
        row["holders"] = []
    monkeypatch.setattr(runner, "_region_status", lambda: rows)
    monkeypatch.setattr(runner, "ancestor_pids", lambda: {123})
    with pytest.raises(runner.ReanchorRefusal, match="q2-only"):
        runner.verify_held_footprint(claim_tag="fg4b")


def test_region_check_accepts_actual_footprint(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = holder_rows()
    monkeypatch.setattr(runner, "_region_status", lambda: rows)
    monkeypatch.setattr(runner, "ancestor_pids", lambda: {123})
    assert runner.verify_held_footprint(claim_tag="fg4b") == rows


@pytest.mark.parametrize(
    ("rows", "ancestors", "match"),
    [
        (holder_rows(tag="wrong"), {123}, "not held by exactly one"),
        (holder_rows(role="wrong"), {123}, "not held by exactly one"),
        (holder_rows(pid=999), {123}, "not in this runner's ancestor"),
    ],
)
def test_region_check_rejects_wrong_tag_role_or_nonancestor(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict],
    ancestors: set[int],
    match: str,
) -> None:
    monkeypatch.setattr(runner, "_region_status", lambda: rows)
    monkeypatch.setattr(runner, "ancestor_pids", lambda: ancestors)
    with pytest.raises(runner.ReanchorRefusal, match=match):
        runner.verify_held_footprint(claim_tag="fg4b")


def test_long_decode_rejects_short_or_missing_timing() -> None:
    response = {"timings": {"predicted_n": 32, "predicted_per_second": 12.0}}
    with pytest.raises(runner.ReanchorRefusal, match="returned 32"):
        runner.parse_sample(response, 1)
    response = {"timings": {"predicted_n": 513, "predicted_per_second": 12.0}}
    with pytest.raises(runner.ReanchorRefusal, match="returned 513"):
        runner.parse_sample(response, 1)
    response = {
        "timings": {"predicted_n": 512, "predicted_per_second": 0.0},
        "choices": [{"finish_reason": "length"}],
    }
    with pytest.raises(runner.ReanchorRefusal, match="no positive"):
        runner.parse_sample(response, 1)


def test_execute_refuses_without_explicit_window_grant() -> None:
    args = runner.parse_args(["--execute"])
    with pytest.raises(runner.ReanchorRefusal, match="i-have-operator-grant"):
        runner.execute(args)


def test_long_decode_accepts_server_timing() -> None:
    response = {
        "timings": {"predicted_n": 512, "predicted_per_second": 42.5, "prompt_n": 18},
        "choices": [{"finish_reason": "length", "message": {"content": "x" * 10}}],
    }
    sample = runner.parse_sample(response, 1)
    assert sample.predicted_per_second == 42.5
    assert sample.predicted_n == 512
    assert runner.completion_payload(512)["ignore_eos"] is True


def test_warmup_requires_three_stable_exact_samples() -> None:
    speeds = iter((6.0, 8.0, 10.0, 10.1, 9.9))

    def request(_payload: dict) -> dict:
        return {
            "timings": {
                "predicted_n": runner.WARMUP_TOKENS,
                "predicted_per_second": next(speeds),
                "prompt_n": 10,
            },
            "choices": [{"finish_reason": "length"}],
        }

    samples = runner.collect_warmup_samples(request)
    assert len(samples) == 5
    assert runner.warmup_is_stable(samples)


def test_warmup_refuses_unstable_or_short_response() -> None:
    ordinal = 0

    def unstable(_payload: dict) -> dict:
        nonlocal ordinal
        ordinal += 1
        return {
            "timings": {
                "predicted_n": runner.WARMUP_TOKENS,
                "predicted_per_second": 10.0 if ordinal % 2 else 20.0,
            },
            "choices": [{"finish_reason": "length"}],
        }

    with pytest.raises(runner.ReanchorRefusal, match="warmup failed"):
        runner.collect_warmup_samples(unstable)
    with pytest.raises(runner.ReanchorRefusal, match="expected exactly 64"):
        runner.collect_warmup_samples(
            lambda _payload: {
                "timings": {"predicted_n": 63, "predicted_per_second": 10.0},
                "choices": [{"finish_reason": "length"}],
            }
        )


def test_finish_reason_must_be_length() -> None:
    response = {
        "timings": {"predicted_n": 512, "predicted_per_second": 42.5},
        "choices": [{"finish_reason": "stop"}],
    }
    with pytest.raises(runner.ReanchorRefusal, match="finish_reason='stop'"):
        runner.parse_sample(response, 1)


def test_live_affinity_must_match_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: runner.expected_affinity())
    assert runner.verify_live_affinity(123) == sorted(runner.expected_affinity())
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    with pytest.raises(runner.ReanchorRefusal, match="affinity mismatch"):
        runner.verify_live_affinity(123)


def test_protocol_receipt_binds_contract_instrument_and_amendment(tmp_path: Path) -> None:
    amendment = tmp_path / "MEASUREMENT-amendment.md"
    amendment.write_text("human reviewed protocol\n")
    receipt = tmp_path / "receipt.json"
    identity = runner.instrument_identity()
    receipt.write_text(json.dumps({
        "schema": runner.PROTOCOL_ATTESTATION_SCHEMA,
        "status": "ratified",
        "protocol_id": runner.PROTOCOL_ID,
        "contract": runner.protocol_contract(),
        "instrument_sha256": identity["sha256"],
        "instrument": {
            "repository": identity["repository"],
            "repository_commit": identity["repository_commit"],
            "repository_tree": identity["repository_tree"],
            "path": identity["path"],
        },
        "human_amendment": {
            "path": str(amendment),
            "sha256": runner.sha256(amendment),
        },
        "reviewed_at": "2026-07-28T17:00:00+00:00",
        "reviewer": "operator",
    }))
    assert runner.validate_protocol_attestation(receipt)["receipt_sha256"]
    payload = json.loads(receipt.read_text())
    payload["instrument_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload))
    with pytest.raises(runner.ReanchorRefusal, match="exact instrument hash"):
        runner.validate_protocol_attestation(receipt)


@pytest.mark.parametrize(
    "field",
    ["repository", "repository_commit", "repository_tree", "path"],
)
def test_protocol_receipt_requires_every_instrument_identity_field(
    tmp_path: Path,
    field: str,
) -> None:
    amendment = tmp_path / "MEASUREMENT-amendment.md"
    amendment.write_text("human reviewed protocol\n")
    identity = runner.instrument_identity()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema": runner.PROTOCOL_ATTESTATION_SCHEMA,
        "status": "ratified",
        "protocol_id": runner.PROTOCOL_ID,
        "contract": runner.protocol_contract(),
        "instrument_sha256": identity["sha256"],
        "instrument": {
            key: value for key, value in identity.items() if key != "sha256" and key != field
        },
        "human_amendment": {
            "path": str(amendment),
            "sha256": runner.sha256(amendment),
        },
        "reviewed_at": "2026-07-28T17:00:00+00:00",
        "reviewer": "operator",
    }))
    with pytest.raises(runner.ReanchorRefusal, match="incomplete instrument identity"):
        runner.validate_protocol_attestation(receipt)


def test_protocol_receipt_rejects_dirty_authoritative_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment = tmp_path / "MEASUREMENT-amendment.md"
    amendment.write_text("human reviewed protocol\n")
    identity = runner.instrument_identity()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema": runner.PROTOCOL_ATTESTATION_SCHEMA,
        "status": "ratified",
        "protocol_id": runner.PROTOCOL_ID,
        "contract": runner.protocol_contract(),
        "instrument_sha256": identity["sha256"],
        "instrument": {key: value for key, value in identity.items() if key != "sha256"},
        "human_amendment": {
            "path": str(amendment),
            "sha256": runner.sha256(amendment),
        },
        "reviewed_at": "2026-07-28T17:00:00+00:00",
        "reviewer": "operator",
    }))
    original_git_output = runner._git_output

    def dirty_git_output(*args: str) -> str:
        if args[:1] == ("status",):
            return " M scripts/benchmark/fg4b_a4_cpu_optimized_reanchor.py"
        return original_git_output(*args)

    monkeypatch.setattr(runner, "_git_output", dirty_git_output)
    with pytest.raises(runner.ReanchorRefusal, match="worktree is dirty"):
        runner.validate_protocol_attestation(receipt)


def test_atomic_publish_failure_removes_partial_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "evidence.json").write_text("{}")
    output = tmp_path / "terminal"
    monkeypatch.setattr(os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        runner.atomic_publish(staging, output)
    assert not staging.exists()
    assert not output.exists()


def test_atomic_publish_fsyncs_files_and_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "evidence.json").write_text("{}")
    (staging / "COMPLETE").write_text("")
    output = tmp_path / "terminal"
    real_fsync = os.fsync
    calls: list[int] = []

    def recording_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    runner.atomic_publish(staging, output)
    assert output.is_dir()
    assert (output / "COMPLETE").is_file()
    assert len(calls) >= 5  # two files, staging dir, parent before + after rename


def test_atomic_publish_removes_output_when_post_rename_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "evidence.json").write_text("{}")
    (staging / "COMPLETE").write_text("")
    output = tmp_path / "terminal"
    real_fsync = os.fsync
    calls = 0

    def fail_second_parent_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise OSError("post-rename fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_second_parent_fsync)
    with pytest.raises(OSError, match="post-rename fsync failed"):
        runner.atomic_publish(staging, output)
    assert not staging.exists()
    assert not output.exists()


def test_cold_cache_preparation_success_and_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "host_memory_numa_snapshot",
        lambda: {"captured_at": "2026-07-28T17:00:00+00:00", "meminfo": "ok", "numa_balancing": "0"},
    )
    results = iter((SimpleNamespace(returncode=0, stdout=""), SimpleNamespace(returncode=0, stdout="3\n")))
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: next(results))
    record = runner.prepare_cold_cache()
    assert record["sync"]["returncode"] == 0
    assert record["drop_caches"]["returncode"] == 0

    results = iter((SimpleNamespace(returncode=0, stdout=""), SimpleNamespace(returncode=1, stdout="denied")))
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: next(results))
    with pytest.raises(runner.ReanchorRefusal, match="drop_caches=3 unavailable"):
        runner.prepare_cold_cache()

    calls = 0

    def failed_sync(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stdout="sync failed")

    monkeypatch.setattr(runner.subprocess, "run", failed_sync)
    with pytest.raises(runner.ReanchorRefusal, match="sync failed"):
        runner.prepare_cold_cache()
    assert calls == 1


def test_exclusive_server_rejects_competitor_and_checks_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "process_tree_pids", lambda _pid: {10, 11})
    monkeypatch.setattr(
        runner,
        "find_llama_processes",
        lambda: [{"pid": "10", "args": "llama-server"}, {"pid": "99", "args": "llama-server"}],
    )
    monkeypatch.setattr(runner, "verify_live_affinity", lambda _pid: [0])
    with pytest.raises(runner.ReanchorRefusal, match="competing inference"):
        runner.verify_exclusive_server(10)

    calls: list[int] = []
    monkeypatch.setattr(
        runner,
        "find_llama_processes",
        lambda: [{"pid": "10", "args": "llama-server"}],
    )
    monkeypatch.setattr(
        runner,
        "verify_live_affinity",
        lambda pid: calls.append(pid) or [0],
    )
    runner.verify_exclusive_server(10)
    runner.verify_exclusive_server(10)
    assert calls == [10, 10]


def test_content_hash_manifest_excludes_itself_and_complete(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text("{}")
    (tmp_path / "COMPLETE").write_text("")
    manifest = runner.write_content_hash_manifest(tmp_path)
    rows = json.loads(manifest.read_text())["files"]
    assert [row["path"] for row in rows] == ["evidence.json"]


def test_proposal_is_evidence_bound_and_non_applying() -> None:
    evidence = {"mean_tokens_per_second": 42.0, "runtime_identity": {"llama_commit": runner.EXPECTED_LLAMA_COMMIT}}
    result = runner.proposal(evidence, evidence_file_sha256="a" * 64)
    assert result["mode"] == "proposal_only"
    assert result["must_not_apply_automatically"] is True
    assert result["apply_eligibility"] == "candidate_protocol_pending_ratification"
    assert "llama-bench tg512" in result["not_comparable_to"]
    assert len(result["evidence_sha256"]) == 64


def test_proposal_hash_matches_exact_written_evidence_bytes(tmp_path: Path) -> None:
    evidence = {
        "mean_tokens_per_second": 42.0,
        "runtime_identity": {"llama_commit": runner.EXPECTED_LLAMA_COMMIT},
    }
    evidence_path = tmp_path / "evidence.json"
    runner.write_json(evidence_path, evidence)
    exact_hash = runner.sha256(evidence_path)
    result = runner.proposal(evidence, evidence_file_sha256=exact_hash)
    assert result["evidence_sha256"] == runner.sha256(evidence_path)
    assert result["evidence_hash_semantics"] == "exact_written_evidence_json_bytes"
