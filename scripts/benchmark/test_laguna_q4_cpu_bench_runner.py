from __future__ import annotations

import json
import signal
import sys
import tempfile
import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace

try:
    import pytest
except ModuleNotFoundError:
    class _Raises:
        def __init__(self, exc: type[Exception], match: str = "") -> None:
            self.exc, self.match = exc, match
        def __enter__(self): return self
        def __exit__(self, typ, value, _trace):
            if typ is None or not issubclass(typ, self.exc):
                raise AssertionError(f"expected {self.exc.__name__}")
            if self.match and self.match not in str(value):
                raise AssertionError(f"{self.match!r} not in {value!r}")
            return True
    class _MonkeyPatch:
        def __init__(self):
            self.undo = []
        def setattr(self, obj, name, value):
            old = getattr(obj, name)
            self.undo.append((obj, name, old))
            setattr(obj, name, value)
        def restore(self):
            for obj, name, old in reversed(self.undo):
                setattr(obj, name, old)
    class pytest:  # type: ignore[no-redef]
        MonkeyPatch = _MonkeyPatch
        raises = _Raises

sys.path.insert(0, str(Path(__file__).parent))
import laguna_q4_cpu_bench_runner as runner


def production_runtime_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    facts: dict[str, object] = {
        "schema": "epyc.orchestrator.runtime_facts",
        "generated_at": "x",
        "runtime_stack": {
            "stack_numa_mode": "both",
            "selected_ports": list(runner.EXPECTED_PRODUCTION_PORTS),
        },
        "state": {},
    }
    live: list[dict[str, object]] = []
    state = facts["state"]
    assert isinstance(state, dict)
    for index, port in enumerate(runner.EXPECTED_PRODUCTION_PORTS):
        model = f"/m/{port}.gguf"
        state[f"server_{port}"] = {"pid": index + 1, "port": port, "model_path": model}
        live.append({
            "pid": index + 1,
            "port": port,
            "model": model,
            "exe": str(runner.BINARY.resolve()),
            "listener_owned": True,
        })
    return facts, live


def predictions() -> list[dict[str, str]]:
    return [
        {
            "instance_id": instance_id,
            "model_name_or_path": "laguna_q4_cpu_v8",
            "model_patch": "" if index >= 27 else f"patch-{index}",
        }
        for index, instance_id in enumerate(runner.SWE_IDS)
    ]


def official_report() -> dict[str, object]:
    completed = list(runner.SWE_IDS[:27])
    resolved = completed[:18]
    unresolved = completed[18:]
    empty = list(runner.SWE_IDS[27:])
    return {
        "schema_version": 2,
        "total_instances": 40,
        "submitted_instances": 40,
        "completed_instances": 27,
        "resolved_instances": 18,
        "unresolved_instances": 9,
        "empty_patch_instances": 13,
        "error_instances": 0,
        "submitted_ids": list(runner.SWE_IDS),
        "completed_ids": completed,
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
        "empty_patch_ids": empty,
        "incomplete_ids": [],
        "error_ids": [],
    }


def test_fixed_protocol_context_kv_environment_and_official_harness() -> None:
    swe, lcb = runner.SUITES
    assert (swe["context"], swe["port"], swe["max_tokens"]) == (49152, 18094, 3072)
    assert (lcb["context"], lcb["port"], lcb["max_tokens"]) == (8192, 18095, 4096)
    for suite in runner.SUITES:
        argv = runner.server_argv(suite)
        assert argv[:6] == [
            str(runner.TASKSET),
            "-c",
            "0-95",
            str(runner.NUMACTL),
            "--interleave=all",
            str(runner.BINARY),
        ]
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        context_index = argv.index("-c", argv.index(str(runner.BINARY)) + 1)
        assert argv[context_index + 1] == str(suite["context"])
        assert argv[argv.index("-ctk") + 1] == "f16"
        assert argv[argv.index("-ctv") + 1] == "f16"
        assert argv[argv.index("-ngl") + 1] == "0"
        assert "--no-op-offload" in argv and "--no-mmap" in argv
    env = runner.clean_env()
    assert env["GGML_IQK"] == "1"
    assert env["LD_LIBRARY_PATH"] == str(runner.BINARY.parent)
    assert "KMP_BLOCKTIME" not in env
    official = runner.official_swe_argv(Path("/tmp/run"), Path("/tmp/predictions.json"))
    assert official[:3] == [str(runner.TASKSET), "-c", "184-191"]
    assert official[official.index("--max_workers") + 1] == "8"
    assert official[official.index("--cache_level") + 1] == "env"
    assert official[official.index("--instance_ids") + 1 : official.index("--max_workers")] == list(
        runner.SWE_IDS
    )
    assert runner.numa_prewarm_argv()[:6] == [
        str(runner.TASKSET), "-c", "0-95", str(runner.NUMACTL), "--interleave=all", "dd"
    ]
    assert runner.BENCH_CPUSET == frozenset(range(96))


def test_numa_prewarm_records_interleaved_full_model_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    evidence = runner.numa_prewarm(tmp_path, runner.SUITES[0])
    assert calls == [(runner.numa_prewarm_argv(), {
        "env": runner.clean_env(), "text": True, "capture_output": True, "check": False,
    })]
    assert evidence["placement"] == {"taskset": "0-95", "numa_policy": "interleave=all"}
    assert json.loads((tmp_path / "swe_oracle.numa_prewarm.json").read_text())["returncode"] == 0


def test_numa_prewarm_fails_closed_and_persists_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=9, stdout="", stderr="dd failed"),
    )
    with pytest.raises(RuntimeError, match="NUMA prewarm failed"):
        runner.numa_prewarm(tmp_path, runner.SUITES[1])
    evidence = json.loads((tmp_path / "lcb_hard.numa_prewarm.json").read_text())
    assert evidence["returncode"] == 9 and evidence["stderr"] == "dd failed"


def test_question_contracts_pin_hash_and_exact_ordered_ids() -> None:
    contracts = [runner.question_contract(suite) for suite in runner.SUITES]
    assert [contract["count"] for contract in contracts] == [40, 53]
    assert contracts[0]["ids"] == list(runner.SWE_IDS)
    assert contracts[1]["ids"] == list(runner.LCB_IDS)


def test_file_identity_is_content_and_inode_bound(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"pinned")
    expected = runner.sha256(artifact)
    identity = runner.file_identity(artifact, expected)
    assert identity["sha256"] == expected
    assert identity["inode"] == artifact.stat().st_ino
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        runner.file_identity(artifact, "0" * 64)


def test_runtime_facts_authorize_exact_server_pid_port_and_resolved_model() -> None:
    facts, live = production_runtime_fixture()
    expected = {
        (row["pid"], row["port"], row["model"])
        for row in live
    }
    assert runner.runtime_authorizations(facts) == expected
    assert len(runner.runtime_guard(live_rows=live, facts=facts)["authorized_live_rows"]) == 24
    live[0]["listener_owned"] = False
    with pytest.raises(RuntimeError, match="unknown or misbound"):
        runner.runtime_guard(live_rows=live, facts=facts)


def test_runtime_guard_allows_only_accelerated_external_sidecars() -> None:
    facts, live = production_runtime_fixture()
    hip = str((runner.LLAMA_ROOT / "build-hip/bin/llama-server").resolve())
    live.append({
        "pid": 999, "port": 18092, "model": str(runner.MODEL), "exe": hip,
        "listener_owned": True, "argv_sha256": "a" * 64, "cpus_allowed_list": "184-191",
        "thread_cpu_allowed_lists": [{"tid": 999, "cpus_allowed_list": "184-191"}],
    })
    evidence = runner.runtime_guard(live_rows=live, facts=facts)
    assert evidence["external_accelerated_rows"][0]["cpus_allowed_list"] == "184-191"
    assert evidence["external_accelerated_rows"][0]["argv_sha256"] == "a" * 64
    live[-1]["listener_owned"] = False
    with pytest.raises(RuntimeError, match="unknown or misbound"):
        runner.runtime_guard(live_rows=live, facts=facts)


def test_runtime_guard_rejects_accelerated_sidecar_on_bench_cores() -> None:
    facts, live = production_runtime_fixture()
    live.append({
        "pid": 999, "port": 18092, "model": str(runner.MODEL),
        "exe": str((runner.LLAMA_ROOT / "build-hip/bin/llama-server").resolve()),
        "listener_owned": True, "argv_sha256": "b" * 64, "cpus_allowed_list": "64-127",
        "thread_cpu_allowed_lists": [{"tid": 999, "cpus_allowed_list": "64-127"}],
    })
    with pytest.raises(RuntimeError, match="overlaps CPU bench cores"):
        runner.runtime_guard(live_rows=live, facts=facts)


def test_runtime_guard_rejects_hidden_worker_overlap_under_disjoint_leader() -> None:
    facts, live = production_runtime_fixture()
    live.append({
        "pid": 999, "port": 18092, "model": str(runner.MODEL),
        "exe": str((runner.LLAMA_ROOT / "build-hip/bin/llama-server").resolve()),
        "listener_owned": True, "argv_sha256": "c" * 64, "cpus_allowed_list": "184-191",
        "thread_cpu_allowed_lists": [
            {"tid": 999, "cpus_allowed_list": "184-191"},
            {"tid": 1000, "cpus_allowed_list": "64-95"},
        ],
    })
    with pytest.raises(RuntimeError, match="overlaps CPU bench cores"):
        runner.runtime_guard(live_rows=live, facts=facts)


def test_runtime_guard_requires_the_both_mode_24_port_contract() -> None:
    facts, live = production_runtime_fixture()
    runtime_stack = facts["runtime_stack"]
    assert isinstance(runtime_stack, dict)
    runtime_stack["stack_numa_mode"] = "quarter"
    with pytest.raises(RuntimeError, match="24-port both-mode"):
        runner.runtime_guard(live_rows=live, facts=facts)


def test_live_process_capture_requires_actual_listener_owner(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    (proc / "net").mkdir()
    (process / "comm").write_text("llama-server\n")
    (process / "cmdline").write_bytes(
        b"llama-server\0--port\0" + b"18092\0-m\0/models/laguna.gguf\0"
    )
    (process / "status").write_text("Name:\tllama-server\nCpus_allowed_list:\t184-191\n")
    (process / "exe").symlink_to(runner.BINARY)
    (process / "fd/9").symlink_to("socket:[77]")
    header = "sl local_address rem_address st tx_queue rx_queue tr tm retr uid timeout inode\n"
    listener = "0: 0100007F:46AC 00000000:0000 0A 0:0 0:0 0 0 0 77\n"
    (proc / "net/tcp").write_text(header + listener)
    (proc / "net/tcp6").write_text(header)
    rows = runner.live_llama_rows(proc)
    assert rows == [
        {
            "pid": 123,
            "port": 18092,
            "model": "/models/laguna.gguf",
            "exe": str(runner.BINARY.resolve()),
            "listener_owned": True,
            "argv_sha256": runner.hashlib.sha256(
                b"\0".join(
                    [b"llama-server", b"--port", b"18092", b"-m", b"/models/laguna.gguf"]
                )
            ).hexdigest(),
            "cpus_allowed_list": "184-191",
        }
    ]
    (process / "fd/9").unlink()
    assert runner.live_llama_rows(proc)[0]["listener_owned"] is False


def test_runtime_maps_require_every_pinned_library_and_content(
    tmp_path: Path,
) -> None:
    library = tmp_path / "libllama.so"
    library.write_bytes(b"runtime")
    stat = library.stat()
    static = {
        "runtime_artifacts": [
            {
                "path": str(library),
                "sha256": runner.sha256(library),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "bytes": stat.st_size,
            }
        ]
    }
    maps = tmp_path / "proc/123/maps"
    maps.parent.mkdir(parents=True)
    maps.write_text(f"7f00-7f01 r-xp 00000000 00:00 0 {library}\n")
    evidence = runner.target_runtime_maps(123, static, tmp_path / "proc")
    assert evidence["mapped"][0]["sha256"] == runner.sha256(library)
    library.write_bytes(b"runtimE")
    with pytest.raises(RuntimeError, match="content changed"):
        runner.target_runtime_maps(123, static, tmp_path / "proc")


def test_owned_executable_is_bound_to_preflight_content(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"binary")
    stat = binary.stat()
    static = {
        "binary": {
            "path": str(binary),
            "sha256": runner.sha256(binary),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "bytes": stat.st_size,
        }
    }
    process = tmp_path / "proc/123"
    process.mkdir(parents=True)
    (process / "exe").symlink_to(binary)
    assert runner.target_executable_identity(123, static, tmp_path / "proc")["bytes"] == 6
    binary.write_bytes(b"BINARY")
    with pytest.raises(RuntimeError, match="content changed"):
        runner.target_executable_identity(123, static, tmp_path / "proc")


def test_execution_gates_include_runtime_autopilot_memory_and_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "mem_available_kib", lambda: runner.MEM_AVAILABLE_MIN_KIB)
    monkeypatch.setattr(runner, "port_free", lambda _: True)
    monkeypatch.setattr(runner, "runtime_guard", lambda: {"exact": True})
    monkeypatch.setattr(runner, "live_autopilot", lambda: [])
    gates = runner.execution_gates()
    assert gates["passed"] is True
    assert gates["campaign_boundary"] == {
        "architect_bench_decoupled_from_e8_quality": True,
        "operator_directive_date": "2026-07-27",
    }
    monkeypatch.setattr(runner, "live_autopilot", lambda: [{"pid": 9}])
    assert "AutoPilot supervisor or child is live" in runner.execution_gates()["failures"]
    monkeypatch.setattr(runner, "live_autopilot", lambda: [])
    monkeypatch.setattr(runner, "port_free", lambda _: False)
    assert "required bench port is occupied" in runner.execution_gates()["failures"]


def test_prepare_is_inference_free_refuses_overwrite_and_requires_fresh_execute_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "validate_static", lambda _: {"identity": "ok"})
    monkeypatch.setattr(
        runner, "execution_gates", lambda: {"passed": False, "failures": ["runtime busy"]}
    )
    monkeypatch.setattr(
        runner,
        "question_contract",
        lambda suite: {"count": len(suite["ids"]), "ids": list(suite["ids"])},
    )
    output = tmp_path / "dry-run"
    plan = runner.prepare(output, False, [])
    assert plan["execute"] is False
    assert plan["gates"]["passed"] is False
    with pytest.raises(RuntimeError, match="overwrite"):
        runner.prepare(output, False, [])
    with pytest.raises(RuntimeError, match="timestamped"):
        runner.prepare(tmp_path / "execute", True, ["--execute"])
    execute = tmp_path / "laguna-q4-cpu-v8-20260726T235959Z"
    assert runner.prepare(execute, True, ["--execute"])["execute"] is True


def test_owned_process_group_cleanup_kills_surviving_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, signal.Signals]] = []
    snapshots = iter(([100, 101], [101], [], []))

    class Process:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: int) -> int:
            self.returncode = 0
            return 0

    monkeypatch.setattr(runner, "process_group_members", lambda _: list(next(snapshots)))
    monkeypatch.setattr(runner.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    monkeypatch.setattr(runner, "port_free", lambda _: True)
    result = runner.cleanup_owned(Process(), 100, runner.SUITES[0])
    assert calls == [(100, signal.SIGTERM), (100, signal.SIGKILL)]
    assert result["members_before"] == [100, 101]
    assert result["members_after_kill"] == []


def test_readiness_fails_closed_on_early_server_exit() -> None:
    process = SimpleNamespace(pid=123, poll=lambda: 1)
    with pytest.raises(RuntimeError, match="exited before readiness"):
        runner.wait_ready(process, runner.SUITES[0], {}, timeout_s=1)


def test_structured_raw_artifacts_require_exact_ordered_denominator(
    tmp_path: Path,
) -> None:
    suite = runner.SUITES[0]
    (tmp_path / "swe_oracle.results.json").write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "suite": "swebench_oracle",
                        "n": 40,
                        "n_questions": 40,
                        "errors": 0,
                        "accuracy": 0.0,
                        "correct": 0,
                    }
                ]
            }
        )
    )
    rows = [
        {"suite": "swebench_oracle", "id": instance_id, "request_error": "", "correct": False}
        for instance_id in runner.SWE_IDS
    ]
    row_path = tmp_path / "swe_oracle.per_question.jsonl"
    row_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert runner.validate_raw_artifacts(tmp_path, suite)["count"] == 40
    rows.reverse()
    row_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="denominator"):
        runner.validate_raw_artifacts(tmp_path, suite)


def test_raw_evaluator_timeout_still_cleans_owned_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cleaned: list[int] = []

    class Process:
        pid = 222

    monkeypatch.setattr(runner, "execution_gates", lambda: {"passed": True, "failures": []})
    monkeypatch.setattr(runner, "verify_model_identity", lambda: {"model": "pinned"})
    monkeypatch.setattr(runner, "numa_prewarm", lambda *_args: {"prewarm": "ok"})
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(runner, "wait_ready", lambda *args, **kwargs: {"mapped": True})
    monkeypatch.setattr(
        runner, "run_owned_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("swe_oracle evaluator timed out")),
    )
    monkeypatch.setattr(
        runner,
        "cleanup_owned",
        lambda process, pgid, suite: cleaned.append(pgid) or {"pgid": pgid},
    )
    with pytest.raises(RuntimeError, match="evaluator timed out"):
        runner.run_raw_suite(tmp_path, runner.SUITES[0], {})
    assert cleaned == [222]
    assert json.loads((tmp_path / "swe_oracle.cleanup.json").read_text())["pgid"] == 222


def test_predictions_and_official_report_enforce_terminal_40_task_partition(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(predictions()))
    evidence = runner.validate_predictions(prediction_path)
    assert evidence["empty_patch_count"] == 13
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(official_report()))
    score = runner.validate_official_swe_report(report_path, evidence)
    assert score["resolved"] == 18
    assert score["denominator"] == 40
    assert score["score"] == 0.45

    broken = official_report()
    broken["error_instances"] = 1
    broken["error_ids"] = [runner.SWE_IDS[0]]
    report_path.write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="denominator drift or harness errors"):
        runner.validate_official_swe_report(report_path, evidence)

    duplicate = official_report()
    duplicate["submitted_ids"] = [*runner.SWE_IDS[:-1], runner.SWE_IDS[0]]
    report_path.write_text(json.dumps(duplicate))
    with pytest.raises(RuntimeError, match="duplicate"):
        runner.validate_official_swe_report(report_path, evidence)


def test_official_swe_pipeline_uses_converter_and_official_report_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "swe_oracle.per_question.jsonl").write_text("{}\n")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> SimpleNamespace:
        calls.append(argv)
        if str(runner.CONVERTER) in argv:
            (tmp_path / "swe_predictions.json").write_text(json.dumps(predictions()))
        else:
            report = tmp_path / "laguna_q4_cpu_v8.laguna-q4-cpu-v8-20260726.json"
            report.write_text(json.dumps(official_report()))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    def fake_owned(argv: list[str], **_: object) -> SimpleNamespace:
        calls.append(argv)
        if str(runner.CONVERTER) in argv:
            (tmp_path / "swe_predictions.json").write_text(json.dumps(predictions()))
        else:
            report = tmp_path / "laguna_q4_cpu_v8.laguna-q4-cpu-v8-20260726.json"
            report.write_text(json.dumps(official_report()))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(runner, "run_owned_command", fake_owned)
    result = runner.run_official_swe(tmp_path)
    assert len(calls) >= 2
    assert result["decision_score"]["score"] == 0.45
    assert any("swebench.harness.run_evaluation" in call for call in calls)


def test_terminal_summary_forbids_raw_swe_as_decision_metric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "laguna-q4-cpu-v8-20260726T235959Z"

    def fake_prepare(path: Path, *_: object, **__: object) -> dict[str, object]:
        path.mkdir()
        return {
            "gates": {"passed": True, "failures": []},
            "static_identity": {"identity": "ok"},
        }

    raw_results = iter(
        (
            {"raw_artifacts": {"accuracy": 0.99, "count": 40}},
            {"raw_artifacts": {"accuracy": 0.25, "count": 53}},
        )
    )
    monkeypatch.setattr(runner, "prepare", fake_prepare)
    class Monitor:
        def __init__(self, *_: object) -> None: pass
        def start(self) -> None: pass
        def sample(self, _: str) -> None: pass
        def stop(self) -> None: pass
    monkeypatch.setattr(runner, "CampaignMonitor", Monitor)
    monkeypatch.setattr(runner, "run_raw_suite", lambda *args: next(raw_results))
    monkeypatch.setattr(
        runner,
        "run_official_swe",
        lambda _: {"decision_score": {"score": 0.45, "denominator": 40}},
    )
    assert (
        runner.main(
            [
                "--execute",
                "--verify-model-sha",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["official_swe"]["score"] == 0.45
    assert summary["lcb_code_execution"]["accuracy"] == 0.25
    assert summary["raw_swe_accuracy_decision_use"] == "forbidden"


def test_runtime_guard_rejects_missing_required_servers() -> None:
    facts, live = production_runtime_fixture()
    with pytest.raises(RuntimeError, match="missing="):
        runner.runtime_guard(live_rows=live[:-1], facts=facts)
    state = facts["state"]
    assert isinstance(state, dict)
    state.pop(f"server_{runner.EXPECTED_PRODUCTION_PORTS[-1]}")
    with pytest.raises(RuntimeError, match="missing production ports"):
        runner.runtime_authorizations(facts)


def test_runtime_guard_ignores_only_inactive_noncontract_fact_rows() -> None:
    facts, live = production_runtime_fixture()
    state = facts["state"]
    assert isinstance(state, dict)
    state["server_stale_embedding"] = {
        "pid": 999,
        "port": 8096,
        "model_path": "/m/stale.gguf",
    }
    assert len(runner.runtime_authorizations(facts)) == 24
    assert len(runner.runtime_guard(live_rows=live, facts=facts)["authorized_live_rows"]) == 24
    stale_live = {
        "pid": 999,
        "port": 8096,
        "model": "/m/stale.gguf",
        "exe": str(runner.BINARY.resolve()),
        "listener_owned": True,
    }
    with pytest.raises(RuntimeError, match="invalid="):
        runner.runtime_guard(live_rows=[*live, stale_live], facts=facts)


def test_runtime_guard_allows_only_registered_candidate(monkeypatch) -> None:
    facts, live = production_runtime_fixture()
    candidate = {"pid": 99, "port": runner.SUITES[0]["port"], "model": str(runner.MODEL),
                 "exe": str(runner.BINARY.resolve()), "listener_owned": True}
    monkeypatch.setattr(runner, "OWNED_SIDECARS", {99: {"pgid": 99, "port": candidate["port"]}})
    assert runner.runtime_guard(live_rows=[*live, candidate], facts=facts)["owned_sidecars"] == [candidate]
    # Startup and cleanup are owned by run_raw_suite; monitor must not fail
    # merely because the registered sidecar is not yet listening or has exited.
    assert runner.runtime_guard(live_rows=live, facts=facts)["owned_sidecars"] == []
    loading = {**candidate, "listener_owned": False}
    assert runner.runtime_guard(live_rows=[*live, loading], facts=facts)["owned_sidecars"] == [loading]
    wrong_port = {**candidate, "port": runner.SUITES[1]["port"]}
    with pytest.raises(RuntimeError, match="identity changed"):
        runner.runtime_guard(live_rows=[*live, wrong_port], facts=facts)
    monkeypatch.setattr(runner, "OWNED_SIDECARS", {99: {"pgid": 99, "port": candidate["port"]}, 100: {"pgid": 100, "port": 1}})
    with pytest.raises(RuntimeError, match="registration"):
        runner.runtime_guard(live_rows=[*live, candidate], facts=facts)
    monkeypatch.setattr(runner, "OWNED_SIDECARS", {})
    with pytest.raises(RuntimeError, match="unknown or misbound"):
        runner.runtime_guard(live_rows=[*live, candidate], facts=facts)


def test_docker_transition_allows_preexisting_running_container() -> None:
    snapshot = "abc123 running orchestrator-api\n"
    runner.validate_docker_container_transition(snapshot, snapshot)
    assert runner.parse_docker_container_rows(snapshot) == {
        "abc123": "running orchestrator-api"
    }


def test_docker_transition_rejects_new_stopped_or_running_container() -> None:
    before = "abc123 running orchestrator-api\n"
    for new_row in ("def456 exited swe-task\n", "def456 running swe-task\n"):
        with pytest.raises(RuntimeError, match="new Docker containers"):
            runner.validate_docker_container_transition(before, before + new_row)


def test_docker_transition_rejects_existing_state_or_name_drift() -> None:
    before = "abc123 running orchestrator-api\n"
    for after in ("abc123 exited orchestrator-api\n", "abc123 running renamed-api\n"):
        with pytest.raises(RuntimeError, match="changed pre-existing"):
            runner.validate_docker_container_transition(before, after)


def test_campaign_container_cleanup_requires_exact_name_and_label() -> None:
    run_id = runner.OFFICIAL_SWE_RUN_ID
    owned = {
        "Name": f"/{run_id}",
        "Config": {"Labels": {"swebench.run_id": run_id}},
    }
    wrong_label = {
        "Name": f"/{run_id}",
        "Config": {"Labels": {"swebench.run_id": "another-run"}},
    }
    before = "old running api\n"
    after = before + "new running task\nambiguous exited task2\n"
    removable, residual = runner.classify_new_campaign_containers(
        before,
        after,
        {"new": owned, "ambiguous": wrong_label},
        run_id,
    )
    assert removable == ["new"]
    assert residual == ["ambiguous"]


def test_official_timeout_persists_unproven_new_container_without_removing(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "swe_oracle.per_question.jsonl").write_text("{}\n")
    owned_calls = 0

    def fake_owned(argv: list[str], **_: object) -> SimpleNamespace:
        nonlocal owned_calls
        owned_calls += 1
        if owned_calls == 1:
            (tmp_path / "swe_predictions.json").write_text(json.dumps(predictions()))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise RuntimeError("official SWE harness timed out")

    ps_calls = 0
    removed: list[str] = []

    def fake_run(argv: list[str], **_: object) -> SimpleNamespace:
        nonlocal ps_calls
        if "images" in argv:
            return SimpleNamespace(returncode=0, stdout="repo@sha256:1\n", stderr="")
        if "inspect" in argv:
            payload = [{"Name": "/unrelated", "Config": {"Labels": {}}}]
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        if "rm" in argv:
            removed.append(argv[-1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "ps" in argv:
            ps_calls += 1
            text = "" if ps_calls == 1 else "new123 running unrelated\n"
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr(runner, "run_owned_command", fake_owned)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="terminal evidence"):
        runner.run_official_swe(tmp_path)

    terminal = json.loads((tmp_path / "swe_docker_terminal.json").read_text())
    assert "timed out" in terminal["execution_error"]
    assert terminal["residual_unproven_ids"] == ["new123"]
    assert terminal["removed_owned_ids"] == []
    assert removed == []


def exercise_official_docker_failure(
    monkeypatch, tmp_path: Path, failure: str
) -> tuple[dict[str, object], list[str]]:
    (tmp_path / "swe_oracle.per_question.jsonl").write_text("{}\n")
    owned_calls = 0

    def fake_owned(argv: list[str], **_: object) -> SimpleNamespace:
        nonlocal owned_calls
        owned_calls += 1
        if owned_calls == 1:
            (tmp_path / "swe_predictions.json").write_text(json.dumps(predictions()))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise RuntimeError("official timeout")

    ps_calls = 0
    remove_targets: list[str] = []
    before = "old running external-api\n"
    after = before + "new running campaign\n"
    ownership = [{
        "Name": f"/{runner.OFFICIAL_SWE_RUN_ID}",
        "Config": {"Labels": {"swebench.run_id": runner.OFFICIAL_SWE_RUN_ID}},
    }]

    def fake_run(argv: list[str], **_: object) -> SimpleNamespace:
        nonlocal ps_calls
        if "images" in argv:
            return SimpleNamespace(returncode=0, stdout="repo@sha256:1\n", stderr="")
        if "inspect" in argv:
            if failure == "inspect":
                return SimpleNamespace(returncode=1, stdout="", stderr="inspect failed")
            return SimpleNamespace(returncode=0, stdout=json.dumps(ownership), stderr="")
        if "rm" in argv:
            remove_targets.append(argv[-1])
            if failure == "rm":
                return SimpleNamespace(returncode=1, stdout="", stderr="rm failed")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "ps" in argv:
            ps_calls += 1
            if ps_calls == 1:
                return SimpleNamespace(returncode=0, stdout=before, stderr="")
            if ps_calls == 2:
                return SimpleNamespace(returncode=0, stdout=after, stderr="")
            if failure == "final_ps":
                return SimpleNamespace(returncode=1, stdout="", stderr="final ps failed")
            final = after if failure in {"inspect", "rm"} else before
            return SimpleNamespace(returncode=0, stdout=final, stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr(runner, "run_owned_command", fake_owned)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="terminal evidence"):
        runner.run_official_swe(tmp_path)
    terminal = json.loads((tmp_path / "swe_docker_terminal.json").read_text())
    assert terminal["containers_before"] == ["old running external-api"]
    assert "old" not in remove_targets
    return terminal, remove_targets


def test_official_inspect_failure_persists_evidence_and_does_not_remove(
    monkeypatch, tmp_path: Path
) -> None:
    terminal, remove_targets = exercise_official_docker_failure(
        monkeypatch, tmp_path, "inspect"
    )
    assert terminal["residual_unproven_ids"] == ["new"]
    assert any("inspect new" in error for error in terminal["postflight_errors"])
    assert remove_targets == []


def test_official_rm_failure_persists_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    terminal, remove_targets = exercise_official_docker_failure(
        monkeypatch, tmp_path, "rm"
    )
    assert terminal["cleanup_failed_ids"] == ["new"]
    assert any("rm new" in error for error in terminal["postflight_errors"])
    assert remove_targets == ["new"]


def test_official_final_ps_failure_persists_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    terminal, remove_targets = exercise_official_docker_failure(
        monkeypatch, tmp_path, "final_ps"
    )
    assert any("final ps" in error for error in terminal["postflight_errors"])
    assert terminal["removed_owned_ids"] == ["new"]
    assert remove_targets == ["new"]


def test_raw_artifacts_recompute_lcb_metric_from_rows(tmp_path: Path) -> None:
    suite = runner.SUITES[1]
    rows = [{"suite": suite["external_name"], "id": item, "request_error": "", "correct": False}
            for item in suite["ids"]]
    (tmp_path / "lcb_hard.per_question.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "lcb_hard.results.json").write_text(json.dumps({"suites": [{
        "suite": suite["external_name"], "n": 53, "n_questions": 53,
        "errors": 0, "correct": 53, "accuracy": 1.0,
    }]}))
    with pytest.raises(RuntimeError, match="raw denominator"):
        runner.validate_raw_artifacts(tmp_path, suite)


def load_tests(loader: unittest.TestLoader, _tests: unittest.TestSuite, _pattern: str) -> unittest.TestSuite:
    """Execute the legacy function tests through the documented stdlib runner."""
    suite = unittest.TestSuite()
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        def case(fn=fn):
            patch = pytest.MonkeyPatch()
            with tempfile.TemporaryDirectory() as directory:
                kwargs = {}
                if "tmp_path" in inspect.signature(fn).parameters:
                    kwargs["tmp_path"] = Path(directory)
                if "monkeypatch" in inspect.signature(fn).parameters:
                    kwargs["monkeypatch"] = patch
                try:
                    fn(**kwargs)
                finally:
                    if hasattr(patch, "restore"):
                        patch.restore()
        suite.addTest(unittest.FunctionTestCase(case, description=name))
    return suite
