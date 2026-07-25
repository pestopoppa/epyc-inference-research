from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import laguna_pgpu1_dflash_runner as runner
from pgpu1_artifact_completeness_audit import audit_artifact


HEAD = "a" * 40
SERVER_SHA = "b" * 64


def _command(stdout: str = "", returncode: int = 0) -> dict:
    return {"returncode": returncode, "stdout": stdout, "stderr": ""}


def _clean_processes() -> dict:
    return {
        "model_binaries": {
            "returncode": 0,
            "commands": {
                "llama-server": _command(returncode=1),
                "llama-cli": _command(returncode=1),
                "llama-bench": _command(returncode=1),
            },
            "proc_owners": [],
        },
        "autopilot": _command(returncode=1),
        "listeners_lsof": _command(returncode=1),
        "listeners_proc": {"returncode": 0, "owners": []},
        "kfd_lsof": _command(returncode=1),
        "kfd_proc": {"returncode": 0, "owners": []},
        "rocm_pids": _command("No KFD PIDs"),
    }


def _rocm(
    pid: int | None = None,
    *,
    dynamic: bool = False,
    vram_bytes: int = 13_094_912,
    gpus: str = "1",
    mapped_pid: int | None = None,
    mapped_devices: tuple[int, ...] = (0,),
    include_mapping: bool = True,
) -> dict:
    pid_row = f"{pid}\tllama-server\t{gpus}\t{vram_bytes}\t0\t0\n" if pid is not None else ""
    mapping = ""
    if pid is not None and include_mapping:
        owner = pid if mapped_pid is None else mapped_pid
        mapping = (
            "GPUs Indexed by PID:\n"
            f"PID {owner} is using {len(mapped_devices)} DRM device(s):\n"
            + "".join(f"{device}\n" for device in mapped_devices)
        )
    captures = [
        _command(
            "KFD Processes:\n"
            "PID PROCESS NAME GPU(s) VRAM USED\n"
            f"{pid_row}"
            f"{mapping}"
            "GPU use (%): 37\n"
            f"VRAM Total Used Memory (B): {vram_bytes}\n"
        )
    ]
    if not dynamic:
        captures.extend([_command("sclk: 1700Mhz mclk: 1600Mhz"), _command("Average Graphics Package Power: 180W"), _command("Temperature (Sensor edge): 52C")])
    return {"available": True, "captures": captures}


def _live_processes(pid: int = 1234, port: int = 19880) -> dict:
    return {
        "model_binaries": {
            "returncode": 0,
            "commands": {
                "llama-server": _command(f"{pid} llama-server\n"),
                "llama-cli": _command(returncode=1),
                "llama-bench": _command(returncode=1),
            },
            "proc_owners": [{"pid": pid, "comm": "llama-server", "exe": "llama-server", "exe_path": str(runner.DEFAULT_BINARY), "exe_resolved": str(runner.DEFAULT_BINARY.resolve())}],
        },
        "autopilot": _command(returncode=1),
        "listeners_lsof": _command(f"llama-ser {pid} user TCP 127.0.0.1:{port} (LISTEN)\n"),
        "listeners_proc": {"returncode": 0, "owners": [{"pid": pid, "fd": "10", "port": port}]},
        "kfd_lsof": _command(f"llama-ser {pid} user /dev/kfd\n"),
        "kfd_proc": {"returncode": 0, "owners": [{"pid": pid, "fd": "5", "target": "/dev/kfd"}]},
        "rocm_pids": _command(f"{pid} llama-server 0 13094912\n"),
    }


def _hardware() -> dict:
    return {
        "gpu_product": _command("Card series: AMD Instinct MI210"),
        "gfx_target": _command("Name: gfx90a"),
        "rocm_runtime": _command("HIP version: 6.4.43483"),
        "rocm_driver": _command("Driver version: 6.12.12"),
        "kernel": _command("Linux 6.8.0 x86_64 GNU/Linux"),
    }


def _identity(*, untracked: str = "", head: str = HEAD) -> dict:
    return {
        "binary": str(runner.DEFAULT_BINARY),
        "binary_sha256": SERVER_SHA,
        "artifact": {"path": str(runner.DEFAULT_BINARY), "resolved_path": str(runner.DEFAULT_BINARY.resolve()), "dev": 1, "inode": 2, "bytes": 300, "mtime_ns": 4, "sha256": SERVER_SHA, "stable": True},
        "environment": runner.runtime_env(runner.DEFAULT_BINARY),
        "local_llama_ggml_libraries": [
            {"soname": "libllama.so", "path": str(runner.DEFAULT_BINARY.parent / "libllama.so"), "resolved_path": str(runner.DEFAULT_BINARY.parent / "libllama.so"), "dev": 1, "inode": 3, "bytes": 100, "mtime_ns": 4, "sha256": "c" * 64, "stable": True},
            {"soname": "libggml.so", "path": str(runner.DEFAULT_BINARY.parent / "libggml.so"), "resolved_path": str(runner.DEFAULT_BINARY.parent / "libggml.so"), "dev": 1, "inode": 4, "bytes": 200, "mtime_ns": 4, "sha256": "d" * 64, "stable": True},
        ],
        "git": {
            "source_root": str(runner.DEFAULT_SOURCE_ROOT),
            "branch": _command(runner.EXPECTED_BRANCH + "\n"),
            "tracked_diff": _command(),
            "index_diff": _command(),
            "untracked": _command(untracked),
            "commit": _command(head + "\n"),
        },
        "server_version": _command(f"llama-server HIP commit {head[:9]}"),
        "ldd": _command(
            f"libllama.so => {runner.DEFAULT_BINARY.parent}/libllama.so\n"
            f"libggml.so => {runner.DEFAULT_BINARY.parent}/libggml.so\n"
        ),
    }


def _harness() -> dict:
    return {"path": "runner", "sha256": "f" * 64, "stable": True, "git": {}, "tracked": _command(), "worktree_unchanged": _command(), "index_unchanged": _command()}


def _model_identities() -> tuple[dict, dict]:
    return (
        {"path": str(runner.DEFAULT_TARGET_MODEL), "bytes": runner.TARGET_MODEL_BYTES, "sha256": runner.TARGET_MODEL_SHA256},
        {"path": str(runner.DEFAULT_DRAFTER_MODEL), "bytes": runner.DRAFTER_MODEL_BYTES, "sha256": runner.DRAFTER_MODEL_SHA256},
    )


def _records(arm_name: str) -> list[dict]:
    return [
        {
            "prompt_index": index,
            "prompt_id": prompt_id,
            "finish_reason": "stop",
            "assistant_content_sha256": f"{arm_name}-{prompt_id}",
            "prompt_tokens": 10,
            "completion_tokens": 100,
            "prompt_ms": 100.0,
            "decode_ms": 200.0,
            "draft_n": 10 if arm_name == "dflash" else 0,
            "draft_n_accepted": 6 if arm_name == "dflash" else 0,
            "semantic_validation": {"passed": True},
            "request_lifecycle": {"fully_contained_valid": True, "fully_contained_sample_count": 1},
        }
        for index, (prompt_id, _) in enumerate(runner.PROMPT_SPECS, 1)
    ]


def _result(arm: runner.Arm, rep: int) -> dict:
    records = _records(arm.name)
    return {
        "arm": arm.name,
        "rep": rep,
        "status": "ok",
        "prompt_count": len(records),
        "prompt_tps": 100.0,
        "decode_tps": 500.0,
        "draft_n": 30 if arm.speculative else 0,
        "draft_n_accepted": 18 if arm.speculative else 0,
        "draft_acceptance_rate": 0.6 if arm.speculative else 0.0,
        "records": records,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()


def _configure_provenance(monkeypatch, root: Path) -> tuple[Path, str, str]:
    governance = root / "governance"
    production = root / "production"
    for repo in (governance, production):
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.invalid")
        _git(repo, "config", "user.name", "Test")
    (production / "kernel.txt").write_text("v7\n", encoding="utf-8")
    _git(production, "add", "kernel.txt")
    _git(production, "commit", "-m", "v7")
    rollback_head = _git(production, "rev-parse", "HEAD")
    _git(production, "branch", runner.ROLLBACK_BRANCH)
    (production / "kernel.txt").write_text("v8\n", encoding="utf-8")
    _git(production, "commit", "-am", "v8")
    production_head = _git(production, "rev-parse", "HEAD")
    _git(production, "branch", runner.EXPECTED_BRANCH)
    monkeypatch.setattr(runner, "GOVERNANCE_REPO", governance)
    monkeypatch.setattr(runner, "PROMOTION_ATTESTATION_RELATIVE_PATH", Path("attestation.json"))
    monkeypatch.setattr(runner, "PROMOTION_ATTESTATION_PATH", governance / "attestation.json")
    monkeypatch.setattr(runner, "DEFAULT_SOURCE_ROOT", production)
    return governance, production_head, rollback_head


def _commit_attestation(governance: Path, path: Path) -> None:
    _git(governance, "add", "--", str(path.relative_to(governance)))
    _git(governance, "commit", "-m", "attestation")


def _attestation(governance: Path, production_head: str = HEAD, rollback_head: str = "c" * 40) -> Path:
    path = runner.PROMOTION_ATTESTATION_PATH
    path.write_text(
        json.dumps(
            {
                "schema": runner.PROMOTION_ATTESTATION_SCHEMA,
                "status": "production_promoted_pending_gpu_certification",
                "production_branch": runner.EXPECTED_BRANCH,
                "production_head": production_head,
                "frozen": False,
                "promoted_at": "2026-07-24T00:00:00Z",
                "server_binary": {"path": str(runner.DEFAULT_BINARY), "sha256": SERVER_SHA},
                "rollback": {
                    "branch": runner.ROLLBACK_BRANCH,
                    "head": rollback_head,
                    "backup_ref": f"refs/heads/{runner.ROLLBACK_BRANCH}",
                    "source_ref": f"refs/heads/{runner.EXPECTED_BRANCH}",
                },
            }
        ),
        encoding="utf-8",
    )
    _commit_attestation(governance, path)
    return path


def _execute_args(attestation: Path, head: str = HEAD) -> object:
    args = runner.parse_args(["--attestation-ref", "attest-test"])
    args.expected_production_head = head
    args.expected_server_sha256 = SERVER_SHA
    args.attestation_ref = attestation
    args.target_identity, args.drafter_identity = _model_identities()
    return args


def test_fixed_protocol_and_execute_requires_provisional_promotion_identity(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--reps", "4"])
    assert runner.parse_args(["--reps", "7"]).reps == 7
    with pytest.raises(SystemExit):
        runner.parse_args(["--target-model", "/tmp/not-laguna.gguf"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--execute", "--attestation-ref", "attest"])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance, production_head, rollback_head = _configure_provenance(monkeypatch, root)
        attestation = _attestation(governance, production_head, rollback_head)
        args = runner.parse_args(
            [
                "--execute",
                "--attestation-ref",
                str(attestation),
                "--expected-production-head",
                production_head,
                "--expected-server-sha256",
                SERVER_SHA,
            ]
        )
        assert args.expected_production_head == production_head
        assert args.attestation_identity["sha256"] == runner.sha256_file(attestation)


def test_exact_arm_contract_kv_quant_and_operator_script() -> None:
    args = runner.parse_args([])
    base = runner.server_argv(args, runner.BASE_ARM, 19000)
    dflash = runner.server_argv(args, runner.DFLASH_ARM, 19001)
    expected_pairs = {
        "-dev": "ROCm0",
        "-ot": "token_embd.weight=ROCm0",
        "-fa": "on",
        "--cache-type-k": "f16",
        "--cache-type-v": "f16",
        "--reasoning": "off",
        "--reasoning-budget": "0",
    }
    for flag, value in expected_pairs.items():
        assert base[base.index(flag) + 1] == value
        assert dflash[dflash.index(flag) + 1] == value
    assert base[base.index("-ngl") + 1] == "all"
    assert base.count("-v") == dflash.count("-v") == 1
    assert "--spec-type" not in base
    assert dflash[dflash.index("--spec-type") + 1] == "draft-dflash"
    assert dflash[dflash.index("--spec-draft-device") + 1] == "ROCm0"
    assert dflash[dflash.index("--spec-draft-type-k") + 1] == "q8_0"
    assert dflash[dflash.index("--spec-draft-type-v") + 1] == "q8_0"
    script = runner.render_operator_run_script(args)
    assert "LAGUNA_PGPU1_PROVISIONAL_ATTESTATION_REF" in script
    assert "LAGUNA_PGPU1_PROMOTED_HEAD" in script
    assert "LAGUNA_PGPU1_PROMOTED_SERVER_SHA256" in script
    assert "--expected-production-head" in script
    assert "--expected-server-sha256" in script
    assert "exec /usr/bin/env -i" in script
    assert f"'PATH={runner.SAFE_PATH}'" in script


def test_plan_is_exact_and_records_provisional_promotion_requirements() -> None:
    args = runner.parse_args([])
    plan = runner.build_plan(args, {"path": "target"})
    assert len(plan["cells"]) == 10
    assert [cell["rep"] for cell in plan["cells"] if cell["arm"] == "base"] == [1, 2, 3, 4, 5]
    assert [(cell["rep"], cell["arm"]) for cell in plan["cells"][:4]] == [(1, "base"), (1, "dflash"), (2, "dflash"), (2, "base")]
    assert all(cell["prompt_count"] == len(runner.PROMPT_SPECS) for cell in plan["cells"])
    assert plan["fixed_prompt_pack"] == [{"id": prompt_id, "text": text} for prompt_id, text in runner.PROMPT_SPECS]
    assert plan["target_kv_quant"] == {"k": "f16", "v": "f16"}
    assert plan["drafter_kv_quant"] == {"k": "q8_0", "v": "q8_0"}
    assert "n >= 10" in plan["rep_policy"]
    assert plan["provisional_promotion_identity"]["expected_head"] == "execute_required"
    assert plan["source_untracked_allowlist"] == runner.SOURCE_UNTRACKED_ALLOWLIST


def test_normalize_prompt_requires_normalized_output_sum_not_input_total() -> None:
    prompt = dict(runner.PROMPT_SPECS)["normalize"]
    assert "input total is 10" in prompt
    assert "JSON `sum` must be the sum of the normalized values" in prompt
    assert "must be 1.0" not in prompt
    assert "Do not report the input total in JSON" in prompt


def test_evidence_and_server_environments_are_closed_and_scrub_parent_knobs(monkeypatch) -> None:
    monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "1.2.3")
    monkeypatch.setenv("GGML_IQK", "1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "7")
    capture = runner.run_capture(["/usr/bin/env"])
    child_env = dict(line.split("=", 1) for line in capture["stdout"].splitlines())
    assert child_env == runner.evidence_env()
    assert capture["environment"] == runner.evidence_env()
    server_env = runner.runtime_env(runner.DEFAULT_BINARY)
    assert server_env["HIP_VISIBLE_DEVICES"] == server_env["ROCR_VISIBLE_DEVICES"] == "0"
    assert "LD_PRELOAD" not in server_env
    assert "HSA_OVERRIDE_GFX_VERSION" not in server_env
    assert "GGML_IQK" not in server_env
    assert set(server_env) == {*runner.BASE_ENVIRONMENT, "LD_LIBRARY_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}
    rejected = runner.run_capture(["/usr/bin/env"], env={**runner.evidence_env(), "LD_PRELOAD": "/tmp/evil.so"})
    assert rejected["returncode"] is None
    assert rejected["exec_error"] == "non-allowlisted subprocess environment"
    assert not runner.subprocess_env_is_allowed({**runner.evidence_env(), "PATH": "relative:/usr/bin"})


def test_production_guard_is_frozen_and_allowlist_is_exact() -> None:
    assert runner.production_identity_valid(_identity(), HEAD, SERVER_SHA) == (True, "ok")
    assert runner.production_identity_valid(_identity(untracked=".gitnexusignore\ntools/math-tools/generated.txt\n"), HEAD, SERVER_SHA) == (True, "ok")
    assert not runner.production_identity_valid(_identity(untracked=".gitnexusignore.bak\n"), HEAD, SERVER_SHA)[0]
    assert not runner.production_identity_valid(_identity(), "a" * 39, SERVER_SHA)[0]
    assert not runner.production_identity_valid(_identity(), HEAD, "c" * 64)[0]
    dirty = _identity()
    dirty["git"]["tracked_diff"] = _command("ggml.cpp\n")
    assert not runner.production_identity_valid(dirty, HEAD, SERVER_SHA)[0]
    failed_untracked_scan = _identity()
    failed_untracked_scan["git"]["untracked"] = _command(returncode=128)
    assert not runner.production_identity_valid(failed_untracked_scan, HEAD, SERVER_SHA)[0]
    missing_library_hash = _identity()
    missing_library_hash["local_llama_ggml_libraries"][0]["sha256"] = None
    assert not runner.production_identity_valid(missing_library_hash, HEAD, SERVER_SHA)[0]


def test_local_llama_and_ggml_libraries_are_all_hashed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        llama = root / "libllama.so"
        ggml = root / "libggml-hip.so"
        llama.write_bytes(b"llama")
        ggml.write_bytes(b"ggml")
        ldd = _command(f"libllama.so => {llama} (0x1)\nlibggml-hip.so => {ggml} (0x2)\nlibc.so => /lib/libc.so (0x3)\n")
        identities = runner.local_library_identities(ldd)
        assert [row["soname"] for row in identities] == ["libggml-hip.so", "libllama.so"]
        assert {row["sha256"] for row in identities} == {runner.sha256_file(llama), runner.sha256_file(ggml)}


def test_model_and_hardware_provenance_are_semantically_validated() -> None:
    target, drafter = _model_identities()
    assert runner.model_identities_valid(target, drafter) == (True, "ok")
    assert not runner.model_identities_valid({**target, "bytes": target["bytes"] + 1}, drafter)[0]
    assert runner.hardware_state_is_valid(_hardware())
    wrong_gpu = _hardware()
    wrong_gpu["gpu_product"] = _command("AMD Radeon")
    assert not runner.hardware_state_is_valid(wrong_gpu)
    missing_gfx = _hardware()
    missing_gfx["gfx_target"] = _command("")
    assert not runner.hardware_state_is_valid(missing_gfx)


def test_provisional_promotion_attestation_is_durable_hashed_and_exact(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance, production_head, rollback_head = _configure_provenance(monkeypatch, root)
        attestation = _attestation(governance, production_head, rollback_head)
        outside = root / "promotion.json"
        outside.write_bytes(attestation.read_bytes())
        identity, reason = runner.load_promotion_attestation(outside, production_head, SERVER_SHA)
        assert identity is None
        assert "canonical governance path" in reason
        identity, reason = runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)
        assert reason == "ok"
        assert identity["sha256"] == runner.sha256_file(attestation)
        original_bytes = attestation.read_bytes()
        attestation.write_bytes(original_bytes + b"\n")
        assert runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)[0] is None
        _git(governance, "add", "--", "attestation.json")
        assert runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)[0] is None
        _git(governance, "restore", "--staged", "--worktree", "--", "attestation.json")
        assert runner.load_promotion_attestation(attestation, "e" * 40, SERVER_SHA)[0] is None
        document = json.loads(attestation.read_text(encoding="utf-8"))
        document["frozen"] = True
        attestation.write_text(json.dumps(document), encoding="utf-8")
        assert runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)[0] is None
        _commit_attestation(governance, attestation)
        document["frozen"] = False
        document["status"] = "production_frozen"
        attestation.write_text(json.dumps(document), encoding="utf-8")
        _commit_attestation(governance, attestation)
        assert runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)[0] is None
        document["status"] = "production_promoted_pending_gpu_certification"
        document["promoted_at"] = "not-a-time"
        attestation.write_text(json.dumps(document), encoding="utf-8")
        _commit_attestation(governance, attestation)
        assert runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)[0] is None
        document["promoted_at"] = "2026-07-24T00:00:00Z"
        document["rollback"]["backup_ref"] = "relative-ref"
        attestation.write_text(json.dumps(document), encoding="utf-8")
        _commit_attestation(governance, attestation)
        assert runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)[0] is None
        document["rollback"]["backup_ref"] = f"refs/heads/{runner.ROLLBACK_BRANCH}"
        document["rollback"]["head"] = "d" * 40
        attestation.write_text(json.dumps(document), encoding="utf-8")
        _commit_attestation(governance, attestation)
        assert runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)[0] is None
        document["rollback"]["head"] = rollback_head
        document["rollback"]["source_ref"] = f"refs/heads/{runner.ROLLBACK_BRANCH}"
        attestation.write_text(json.dumps(document), encoding="utf-8")
        _commit_attestation(governance, attestation)
        assert runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)[0] is None


def test_attestation_rejects_committed_alternate_via_canonical_symlink(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance, production_head, rollback_head = _configure_provenance(monkeypatch, root)
        attestation = _attestation(governance, production_head, rollback_head)
        alternate = governance / "alternate.json"
        alternate.write_bytes(attestation.read_bytes())
        _git(governance, "add", "alternate.json")
        _git(governance, "commit", "-m", "alternate")
        attestation.unlink()
        attestation.symlink_to(alternate)
        identity, reason = runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)
        assert identity is None
        assert "regular non-symlink" in reason


def test_attestation_parse_uses_only_provenance_verified_bytes(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance, production_head, rollback_head = _configure_provenance(monkeypatch, root)
        attestation = _attestation(governance, production_head, rollback_head)
        verify = runner.verified_governance_attestation

        def verify_then_replace(path: Path):
            raw, identity, reason = verify(path)
            path.write_text('{"status":"replacement"}', encoding="utf-8")
            return raw, identity, reason

        monkeypatch.setattr(runner, "verified_governance_attestation", verify_then_replace)
        identity, reason = runner.load_promotion_attestation(attestation, production_head, SERVER_SHA)
        assert reason == "ok"
        assert identity["document"]["status"] == "production_promoted_pending_gpu_certification"
        assert identity["sha256"] != runner.sha256_file(attestation)


def test_rocm_parser_and_live_binding_reject_wrong_or_extra_owners() -> None:
    pid, port = 1234, 19880
    assert runner.rocm_dynamic_command(pid) == [
        "rocm-smi",
        "--showpids",
        "--showpidgpus",
        str(pid),
        "--showmeminfo",
        "vram",
        "--showuse",
    ]
    dynamic = _rocm(pid, dynamic=True)
    assert runner.snapshot_is_valid(dynamic)
    assert runner.parse_rocm_pid_rows(dynamic) == [{"pid": pid, "process_name": "llama-server", "gpus": "1", "vram_bytes": 13_094_912}]
    assert runner.parse_rocm_pid_gpu_rows(dynamic) == [
        {"pid": pid, "declared_device_count": 1, "devices": [0], "malformed": False}
    ]
    assert runner.live_binding_is_valid(_live_processes(pid, port), dynamic, pid, port, runner.DEFAULT_BINARY) == (True, "ok")
    for mutate in ("model", "kfd", "listener", "rocm"):
        processes = _live_processes(pid, port)
        snapshot = dynamic
        if mutate == "model":
            processes["model_binaries"]["proc_owners"].append({"pid": 999, "comm": "llama-server", "exe": "llama-server"})
        elif mutate == "kfd":
            processes["kfd_proc"]["owners"].append({"pid": 999, "target": "/dev/kfd"})
        elif mutate == "listener":
            processes["listeners_proc"]["owners"] = [{"pid": 999, "port": port}]
        else:
            snapshot = _rocm(999, dynamic=True)
        assert not runner.live_binding_is_valid(processes, snapshot, pid, port, runner.DEFAULT_BINARY)[0]
    wrong_exe = _live_processes(pid, port)
    wrong_exe["model_binaries"]["proc_owners"][0]["exe_resolved"] = "/tmp/llama-server"
    assert not runner.live_binding_is_valid(wrong_exe, dynamic, pid, port, runner.DEFAULT_BINARY)[0]
    assert not runner.snapshot_is_valid({"available": True, "captures": [_command("VRAM Total Used Memory (B): 1")]})
    assert not runner.snapshot_is_valid({"available": True, "captures": []})


@pytest.mark.parametrize(
    ("mapped_devices", "include_mapping"),
    [((1,), True), ((0, 1), True), ((), True), ((0,), False)],
)
def test_live_binding_rejects_wrong_multi_or_missing_physical_gpu_identity(
    mapped_devices: tuple[int, ...],
    include_mapping: bool,
) -> None:
    pid, port = 1234, 19880
    snapshot = _rocm(
        pid,
        dynamic=True,
        mapped_devices=mapped_devices,
        include_mapping=include_mapping,
    )
    valid, reason = runner.live_binding_is_valid(_live_processes(pid, port), snapshot, pid, port, runner.DEFAULT_BINARY)
    assert valid is False
    assert "ROCm physical-device mapping" in reason


@pytest.mark.parametrize("gpus", ["0,1", "0, 1", "GPU0", "-"])
def test_live_binding_rejects_malformed_showpids_device_field(gpus: str) -> None:
    pid, port = 1234, 19880
    snapshot = _rocm(pid, dynamic=True, gpus=gpus)
    valid, reason = runner.live_binding_is_valid(
        _live_processes(pid, port),
        snapshot,
        pid,
        port,
        runner.DEFAULT_BINARY,
    )
    assert valid is False
    assert "ROCm PID mapping" in reason


def test_live_binding_rejects_wrong_or_extra_physical_gpu_pid_mapping() -> None:
    pid, port = 1234, 19880
    wrong_pid = _rocm(pid, dynamic=True, mapped_pid=999)
    extra_pid = _rocm(pid, dynamic=True)
    extra_pid["captures"][0]["stdout"] += "PID 999 is using 1 DRM device(s):\n0\n"
    for snapshot in (wrong_pid, extra_pid):
        valid, reason = runner.live_binding_is_valid(
            _live_processes(pid, port),
            snapshot,
            pid,
            port,
            runner.DEFAULT_BINARY,
        )
        assert valid is False
        assert "ROCm physical-device mapping" in reason


def _log_line(seconds: str, function: str, message: str, module: str = "") -> str:
    prefix = f"0.03.{seconds}.001 I "
    if module:
        prefix += f"{module}    "
    return f"{prefix}{function}: {message}\n"


def _residency_log(*, speculative: bool = True) -> str:
    text = _log_line("100", "load_model", f"loading model '{runner.DEFAULT_TARGET_MODEL}'", "srv")
    text += _log_line("200", "load_tensors", "offloaded 49/49 layers to GPU")
    text += _log_line("201", "load_tensors", "       ROCm0 model buffer size = 35538.61 MiB")
    for offset, size in enumerate((102.0, 192.0), 300):
        text += _log_line(str(offset), "llama_kv_cache", f"     ROCm0 KV buffer size = {size:.2f} MiB")
        text += _log_line(str(offset + 1), "llama_kv_cache", f"size = {size:.2f} MiB ( 4096 cells), K (f16): {size / 2:.2f} MiB, V (f16): {size / 2:.2f} MiB")
    if speculative:
        text += _log_line("500", "common_speculative_init_result", f"loading draft model '{runner.DEFAULT_DRAFTER_MODEL}'")
        text += _log_line("600", "load_tensors", "offloaded 7/7 layers to GPU")
        text += _log_line("601", "load_tensors", "       ROCm0 model buffer size = 2126.77 MiB")
        text += _log_line("700", "llama_kv_cache", "     ROCm0 KV buffer size = 51.00 MiB")
        text += _log_line("701", "llama_kv_cache", "size = 51.00 MiB ( 4096 cells), K (q8_0): 25.50 MiB, V (q8_0): 25.50 MiB")
    return text


def test_log_residency_requires_anchored_target_f16_and_drafter_q8_kv_lines() -> None:
    base = runner.parse_log_residency(_residency_log(speculative=False), runner.BASE_ARM)
    dflash = runner.parse_log_residency(_residency_log(), runner.DFLASH_ARM)
    assert base["passed"] and base["target_positive_rocm0_model_buffers_mib"] == [35538.61]
    assert dflash["passed"] and dflash["drafter_positive_rocm0_model_buffers_mib"] == [2126.77]
    assert len(dflash["target_positive_f16_kv_buffers"]) == 2
    assert len(dflash["drafter_positive_q8_kv_buffers"]) == 1
    assert not runner.parse_log_residency(_residency_log().replace("2126.77 MiB", "0.00 MiB"), runner.DFLASH_ARM)["passed"]
    assert not runner.parse_log_residency(_residency_log().replace("K (f16)", "K (q8_0)", 1), runner.DFLASH_ARM)["passed"]
    assert not runner.parse_log_residency(_residency_log().replace("V (f16)", "V (q8_0)", 1), runner.DFLASH_ARM)["passed"]
    assert not runner.parse_log_residency(_residency_log().replace("K (q8_0)", "K (f16)", 1), runner.DFLASH_ARM)["passed"]
    assert not runner.parse_log_residency(_residency_log().replace("V (q8_0)", "V (f16)", 1), runner.DFLASH_ARM)["passed"]
    assert not runner.parse_log_residency("noise offloaded 49/49 layers to GPU\nnoise ROCm0 model buffer size = 1 MiB\n", runner.BASE_ARM)["passed"]


def _response() -> dict:
    return {
        "timings": {
            "prompt_ms": 1.5,
            "predicted_ms": 2.5,
            "prompt_per_second": 20.0,
            "predicted_per_second": 30.0,
            "draft_n": 10,
            "draft_n_accepted": 6,
        },
        "usage": {"prompt_tokens": 11, "completion_tokens": 13},
    }


@pytest.mark.parametrize(
    ("section", "field", "bad"),
    [
        ("usage", "prompt_tokens", True),
        ("usage", "prompt_tokens", 1.5),
        ("usage", "completion_tokens", "13"),
        ("timings", "prompt_ms", False),
        ("timings", "predicted_ms", float("nan")),
        ("timings", "prompt_per_second", float("inf")),
        ("timings", "draft_n", 10.0),
        ("timings", "draft_n_accepted", True),
    ],
)
def test_timing_and_counter_types_are_strict(section: str, field: str, bad: object) -> None:
    response = _response()
    response[section][field] = bad
    with pytest.raises(RuntimeError):
        runner.timings_from_response(response, speculative=True)


def test_timing_counters_require_valid_relationships() -> None:
    parsed = runner.timings_from_response(_response(), speculative=True)
    assert parsed["draft_n"] == 10
    response = _response()
    response["timings"]["draft_n_accepted"] = 11
    with pytest.raises(RuntimeError):
        runner.timings_from_response(response, speculative=True)
    response = _response()
    del response["timings"]["draft_n"]
    with pytest.raises(RuntimeError, match="draft counters"):
        runner.timings_from_response(response, speculative=True)


def test_summary_math_and_json_reject_nonfinite_values() -> None:
    with pytest.raises(RuntimeError):
        runner.median_mad([1.0, float("nan")])
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            runner.write_json(Path(tmp) / "bad.json", {"value": float("inf")})


def test_semantic_validators_accept_concise_explanations_and_reject_bad_structure_or_json() -> None:
    concise = "Trial division through the square root identifies each prime. "
    varied = "Visit values in encounter order, recurse into lists and objects, and emit only scalar leaves. "
    assert runner.semantic_validation("primes", concise + 'RESULT_JSON: {"primes":[2,3,5,7,11,13,17,19,23,29],"sum":129}')["passed"]
    assert runner.semantic_validation("nested_flatten", varied + 'RESULT_JSON: {"values":[1,2,3,4,5]}')["passed"]
    assert runner.semantic_validation("normalize", "Divide by the total and preserve the zero-sum rule. RESULT_JSON: {\"normalized\":[0,0.20,0.30,0.50],\"sum\":1.0}")["passed"]
    assert not runner.semantic_validation("primes", 'RESULT_JSON: {"primes":[2,3,5,7,11,13,17,19,23,29],"sum":129}')["passed"]
    assert not runner.semantic_validation("primes", concise + 'RESULT_JSON: {"primes":[2,3,5,7,11,13,17,19,23,29],"sum":128}')["passed"]
    assert not runner.semantic_validation("nested_flatten", varied + 'RESULT_JSON: {"values":[true,2,3,4,5]}')["passed"]
    assert not runner.semantic_validation("normalize", concise + 'RESULT_JSON: {"normalized":[0,0.2,0.3,0.4],"sum":1.0}')["passed"]
    assert not runner.semantic_validation("normalize", concise + 'RESULT_JSON: {"normalized":[0,0.2,0.3,0.5],"sum":10}')["passed"]
    assert not runner.semantic_validation("primes", concise + 'RESULT_JSON: {"primes":[2],"sum":2}\nextra')["passed"]
    punctuation_padding = ". " * 100
    assert not runner.semantic_validation(
        "primes",
        punctuation_padding
        + 'RESULT_JSON: {"primes":[2,3,5,7,11,13,17,19,23,29],"sum":129}',
    )["passed"]
    seven_words = "one two three four five six seven "
    assert not runner.semantic_validation(
        "primes",
        seven_words
        + 'RESULT_JSON: {"primes":[2,3,5,7,11,13,17,19,23,29],"sum":129}',
    )["passed"]
    eight_words = seven_words + "eight "
    assert runner.semantic_validation(
        "primes",
        eight_words
        + 'RESULT_JSON: {"primes":[2,3,5,7,11,13,17,19,23,29],"sum":129}',
    )["passed"]


def test_finish_reason_requires_stop() -> None:
    assert runner.finish_reason_from_response({"choices": [{"finish_reason": "stop"}]}) == "stop"
    with pytest.raises(RuntimeError, match="did not finish normally"):
        runner.finish_reason_from_response({"choices": [{"finish_reason": "length"}]})
    for malformed in ({}, {"choices": []}, {"choices": "not-a-list"}, {"choices": [None]}):
        with pytest.raises(RuntimeError, match="exactly one completion choice"):
            runner.finish_reason_from_response(malformed)


def test_matrix_cardinality_rejects_vacuous_missing_duplicate_and_bad_prompt() -> None:
    results = [_result(arm, rep) for arm in runner.ARMS for rep in range(1, 6)]
    assert runner.matrix_cardinality_valid(results, 5) == (True, "ok")
    assert runner.summarize_arm([], runner.BASE_ARM, 5)["all_ok"] is False
    assert not runner.matrix_cardinality_valid(results[:-1], 5)[0]
    duplicate = copy.deepcopy(results)
    duplicate[-1]["rep"] = 4
    assert not runner.matrix_cardinality_valid(duplicate, 5)[0]
    wrong_prompt = copy.deepcopy(results)
    wrong_prompt[0]["records"][0]["prompt_id"] = "other"
    assert not runner.matrix_cardinality_valid(wrong_prompt, 5)[0]
    failed_semantics = copy.deepcopy(results)
    failed_semantics[0]["records"][0]["semantic_validation"]["passed"] = False
    assert not runner.matrix_cardinality_valid(failed_semantics, 5)[0]
    failed_finish = copy.deepcopy(results)
    failed_finish[0]["records"][0]["finish_reason"] = "length"
    assert not runner.matrix_cardinality_valid(failed_finish, 5)[0]


def test_request_sampler_captures_binding_while_query_is_active(monkeypatch) -> None:
    calls: list[str] = []

    def evidence(pid: int, port: int, binary: Path, phase: str, *_args) -> dict:
        started = time.monotonic()
        time.sleep(0.002)
        ended = time.monotonic()
        calls.append(phase)
        return {"phase": phase, "pid": pid, "port": port, "binary": str(binary), "sample_started_monotonic": started, "sample_ended_monotonic": ended, "valid": True, "reason": "ok"}

    def query(*_args) -> dict:
        time.sleep(0.04)
        return {"ok": True}

    monkeypatch.setattr(runner, "collect_live_binding_evidence", evidence)
    monkeypatch.setattr(runner, "query_chat", query)
    response, elapsed, lifecycle = runner.query_with_live_samples(19880, {}, 10, 1234, runner.DEFAULT_BINARY, 2)
    assert response == {"ok": True}
    assert elapsed > 0
    assert lifecycle["fully_contained_valid"] is True
    assert lifecycle["fully_contained_sample_count"] >= 1
    assert lifecycle["samples"] and calls[0].startswith("during_request_2_")
    for sample in lifecycle["samples"][: lifecycle["fully_contained_sample_count"]]:
        assert sample["sample_started_monotonic"] >= lifecycle["request_started_monotonic"]


def test_request_sampler_rejects_samples_not_fully_contained(monkeypatch) -> None:
    def slow_evidence(_pid: int, _port: int, _binary: Path, phase: str) -> dict:
        started = time.monotonic()
        time.sleep(0.03)
        return {"phase": phase, "sample_started_monotonic": started, "sample_ended_monotonic": time.monotonic(), "valid": True}

    monkeypatch.setattr(runner, "collect_live_binding_evidence", slow_evidence)
    monkeypatch.setattr(runner, "query_chat", lambda *_args: {"ok": True})
    with pytest.raises(RuntimeError, match="request lifecycle binding failed"):
        runner.query_with_live_samples(19880, {}, 10, 1234, runner.DEFAULT_BINARY, 1)


def test_live_sample_brackets_rocm_with_process_binding(monkeypatch) -> None:
    pid, port = 1234, 19880
    snapshots = [_live_processes(pid, port), _live_processes(pid, port)]
    monkeypatch.setattr(runner, "process_snapshot", lambda: snapshots.pop(0))
    monkeypatch.setattr(runner, "collect_dynamic_rocm_snapshot", lambda target_pid: _rocm(target_pid, dynamic=True))
    sample = runner.collect_live_binding_evidence(pid, port, runner.DEFAULT_BINARY, "during")
    assert sample["valid"] is True
    assert sample["sample_started_monotonic"] <= sample["sample_ended_monotonic"]
    assert "processes_before" in sample and "processes_after" in sample

    contaminated = _live_processes(pid, port)
    contaminated["model_binaries"]["proc_owners"][0]["exe_resolved"] = "/tmp/wrong"
    snapshots = [_live_processes(pid, port), contaminated]
    monkeypatch.setattr(runner, "process_snapshot", lambda: snapshots.pop(0))
    assert runner.collect_live_binding_evidence(pid, port, runner.DEFAULT_BINARY, "during")["valid"] is False


def test_stable_file_identity_rejects_hash_race(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "artifact"
        artifact.write_bytes(b"stable")
        original_fstat = runner.os.fstat
        calls = 0

        def changed_after_hash(fd: int):
            nonlocal calls
            calls += 1
            value = original_fstat(fd)
            if calls >= 2:
                return SimpleNamespace(st_dev=value.st_dev, st_ino=value.st_ino, st_size=value.st_size, st_mtime_ns=value.st_mtime_ns + 1)
            return value

        monkeypatch.setattr(runner.os, "fstat", changed_after_hash)
        assert runner.stable_file_identity(artifact)["stable"] is False


def test_live_artifacts_reject_wrong_binary_libs_and_model_mappings(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        binary, llama, ggml, target, drafter = (root / name for name in ("llama-server", "libllama.so", "libggml.so", "target model.gguf", "drafter.gguf"))
        for path in (binary, llama, ggml, target, drafter):
            path.write_bytes(path.name.encode())
        binding = {"server": {"artifact": runner.stable_file_identity(binary), "local_llama_ggml_libraries": [runner.stable_file_identity(llama), runner.stable_file_identity(ggml)]}, "models": {"target": runner.stable_file_identity(target), "drafter": runner.stable_file_identity(drafter)}}
        expected_exe = binary.stat()
        original_stat = runner.os.stat
        monkeypatch.setattr(runner.os, "stat", lambda path, *args, **kwargs: expected_exe if str(path) == "/proc/1234/exe" else original_stat(path, *args, **kwargs))

        def maps(*paths: Path) -> dict:
            lines = []
            for path in paths:
                value = path.stat()
                escaped_path = str(path).replace(" ", "\\040")
                lines.append(f"00400000-00401000 r--p 00000000 {os.major(value.st_dev):02x}:{os.minor(value.st_dev):02x} {value.st_ino} {escaped_path}\n")
                if path.name.startswith(("libllama", "libggml")):
                    lines.append(f"00401000-00402000 r-xp 00001000 {os.major(value.st_dev):02x}:{os.minor(value.st_dev):02x} {value.st_ino} {escaped_path}\n")
            return {"returncode": 0, "stdout": "".join(lines)}

        monkeypatch.setattr(runner, "proc_maps", lambda _pid: maps(llama, ggml, target, drafter))
        assert runner.live_artifacts_valid(1234, binding, True) == (True, "ok")
        wrong = binary.stat()
        monkeypatch.setattr(runner.os, "stat", lambda path, *args, **kwargs: SimpleNamespace(st_dev=wrong.st_dev, st_ino=wrong.st_ino + 1, st_size=wrong.st_size, st_mtime_ns=wrong.st_mtime_ns) if str(path) == "/proc/1234/exe" else original_stat(path, *args, **kwargs))
        assert "exe identity" in runner.live_artifacts_valid(1234, binding, True)[1]
        monkeypatch.setattr(runner.os, "stat", lambda path, *args, **kwargs: expected_exe if str(path) == "/proc/1234/exe" else original_stat(path, *args, **kwargs))
        monkeypatch.setattr(runner, "proc_maps", lambda _pid: maps(llama, target, drafter))
        assert "libllama/libggml" in runner.live_artifacts_valid(1234, binding, True)[1]
        monkeypatch.setattr(runner, "proc_maps", lambda _pid: maps(llama, ggml, target))
        assert "drafter GGUF" in runner.live_artifacts_valid(1234, binding, True)[1]
        monkeypatch.setattr(runner, "proc_maps", lambda _pid: maps(llama, ggml, drafter))
        assert "target GGUF" in runner.live_artifacts_valid(1234, binding, True)[1]
        monkeypatch.setattr(runner, "proc_maps", lambda _pid: maps(llama, ggml, target, drafter))
        assert "forbidden" in runner.live_artifacts_valid(1234, binding, False)[1]


def test_terminate_sigkills_lingering_process_group_children(monkeypatch) -> None:
    class ExitedParent:
        pid = 424242
        returncode = 0

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout: int) -> int:
            return 0

    pgrep_results = iter([_command("424243 child\n"), _command(returncode=1)])
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(runner, "run_capture", lambda *_args, **_kwargs: next(pgrep_results))
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    evidence = runner.terminate(ExitedParent())
    assert signals == [(424242, signal.SIGKILL)]
    assert evidence["kill_sent"] is True
    assert evidence["dead"] is True


def test_execute_requires_complete_matrix_and_final_clean_state(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance, production_head, rollback_head = _configure_provenance(monkeypatch, root)
        args = _execute_args(_attestation(governance, production_head, rollback_head), production_head)
        monkeypatch.setattr(runner, "binary_identity", lambda *_: _identity(head=production_head))
        monkeypatch.setattr(runner, "production_identity_valid", lambda *_: (True, "ok"))
        monkeypatch.setattr(runner, "collect_hardware_state", _hardware)
        monkeypatch.setattr(runner, "collect_rocm_snapshot", _rocm)
        monkeypatch.setattr(runner, "harness_identity", lambda: _harness())
        monkeypatch.setattr(runner, "immutable_model_identity", lambda path: _model_identities()[0] if path == runner.DEFAULT_TARGET_MODEL else _model_identities()[1])
        monkeypatch.setattr(runner, "process_snapshot", _clean_processes)
        monkeypatch.setattr(runner, "run_replicate", lambda _args, arm, rep, _port, _output, _refs: _result(arm, rep))
        output = root / "first"
        summary = runner.execute(args, output, runner.build_plan(args, args.target_identity))
        assert summary["status"] == "ok"
        assert summary["matrix_cardinality_valid"] is True
        assert summary["final_guard_valid"] is True
        assert summary["execution_binding_valid"] is True

        snapshots = [_clean_processes(), _clean_processes()]
        snapshots[-1]["kfd_proc"]["owners"] = [{"pid": 999, "target": "/dev/kfd"}]
        monkeypatch.setattr(runner, "process_snapshot", lambda: snapshots.pop(0))
        output = root / "second"
        summary = runner.execute(args, output, runner.build_plan(args, args.target_identity))
        assert summary["status"] == "failed"
        assert summary["final_guard_valid"] is False


def test_execute_rejects_transient_library_swap_before_it_is_restored(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance, production_head, rollback_head = _configure_provenance(monkeypatch, root)
        args = _execute_args(_attestation(governance, production_head, rollback_head), production_head)
        changed = _identity(head=production_head)
        changed["local_llama_ggml_libraries"][0]["sha256"] = "e" * 64
        identities = iter([
            _identity(head=production_head),
            _identity(head=production_head),
            changed,
            _identity(head=production_head),
        ])
        monkeypatch.setattr(runner, "binary_identity", lambda *_: next(identities))
        monkeypatch.setattr(runner, "production_identity_valid", lambda *_: (True, "ok"))
        monkeypatch.setattr(runner, "collect_hardware_state", _hardware)
        monkeypatch.setattr(runner, "collect_rocm_snapshot", _rocm)
        monkeypatch.setattr(runner, "harness_identity", lambda: _harness())
        monkeypatch.setattr(runner, "immutable_model_identity", lambda path: _model_identities()[0] if path == runner.DEFAULT_TARGET_MODEL else _model_identities()[1])
        monkeypatch.setattr(runner, "process_snapshot", _clean_processes)
        monkeypatch.setattr(runner, "run_replicate", lambda _args, arm, rep, _port, _output, _refs: _result(arm, rep))
        summary = runner.execute(args, root / "run", runner.build_plan(args, args.target_identity))
        assert summary["status"] == "failed"
        assert summary["execution_binding_valid"] is False
        assert summary["post_execution_identity"]["binding_unchanged"] is True
        assert summary["per_replicate_bindings_valid"] is False
        assert summary["results"][0]["status"] == "error"
        assert "binding_error" in summary["results"][0]


def test_execute_fails_when_provisional_attestation_mutates(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        governance, production_head, rollback_head = _configure_provenance(monkeypatch, root)
        attestation_path = _attestation(governance, production_head, rollback_head)
        args = _execute_args(attestation_path, production_head)
        monkeypatch.setattr(runner, "binary_identity", lambda *_: _identity(head=production_head))
        monkeypatch.setattr(runner, "production_identity_valid", lambda *_: (True, "ok"))
        monkeypatch.setattr(runner, "collect_hardware_state", _hardware)
        monkeypatch.setattr(runner, "collect_rocm_snapshot", _rocm)
        monkeypatch.setattr(runner, "harness_identity", lambda: _harness())
        monkeypatch.setattr(runner, "immutable_model_identity", lambda path: _model_identities()[0] if path == runner.DEFAULT_TARGET_MODEL else _model_identities()[1])
        monkeypatch.setattr(runner, "process_snapshot", _clean_processes)

        def mutate_attestation(_args, arm, rep, _port, _output, _refs):
            document = json.loads(attestation_path.read_text(encoding="utf-8"))
            document["rollback"]["head"] = "d" * 40
            attestation_path.write_text(json.dumps(document), encoding="utf-8")
            return _result(arm, rep)

        monkeypatch.setattr(runner, "run_replicate", mutate_attestation)
        summary = runner.execute(args, root / "run", runner.build_plan(args, args.target_identity))
        assert summary["status"] == "failed"
        assert summary["execution_binding_valid"] is False
        assert summary["post_execution_identity"]["binding_unchanged"] is False


def test_response_sanity_and_artifact_auditor_negative_path() -> None:
    coherent = "This deterministic answer explains the algorithm carefully, includes edge cases, and gives enough varied technical detail to remain readable and useful. " * 4
    assert runner.response_sanity(coherent)["passed"] is True
    assert runner.response_sanity("word " * 80)["passed"] is False
    assert runner.response_sanity("x" * 100)["passed"] is False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "summary.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        assert audit_artifact(root)["status"] == "incomplete"
