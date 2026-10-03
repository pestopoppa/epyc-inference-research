"""Tests for safe stack reload behavior."""

from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from scripts.server import orchestrator_stack as stack
from scripts.server import stack_commands


@pytest.fixture
def isolated_runtime_facts_tmp(monkeypatch, tmp_path: Path) -> Path:
    """Keep direct stack lifecycle tests away from the host runtime manifest."""
    monkeypatch.setitem(stack_commands._PATHS, "tmp_dir", tmp_path)
    return tmp_path


def _stack_gate_args(**overrides) -> Namespace:
    defaults = {
        "dev": False,
        "validate_only": False,
        "migrate_to": None,
        "dry_run": False,
        "skip_stack_change_gate": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_stack_change_launch_gate_runs_canonical_command(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(cmd, 0, stdout="summary: ok\n", stderr="")

    monkeypatch.setattr(stack_commands.subprocess, "run", fake_run)

    assert stack_commands._run_stack_change_launch_gate(_stack_gate_args())

    assert captured["cmd"] == list(stack_commands.STACK_CHANGE_LAUNCH_GATE_COMMAND)
    # The gate runs from the resolved project root; a worktree path is valid,
    # so equality with the canonical path resolver is what is load-bearing.
    assert Path(str(captured["cwd"])) == Path(stack_commands._PATHS["project_root"])
    out = capsys.readouterr().out
    assert "stack-change-gate" in out
    assert "summary: ok" in out


def test_repo_short_sha_uses_repo_root(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(cmd, 0, stdout="abcdef1\n", stderr="")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)

    assert stack._repo_short_sha(tmp_path) == "abcdef1"
    assert captured["cmd"] == [
        "git",
        "-C",
        str(tmp_path),
        "rev-parse",
        "--short",
        "HEAD",
    ]
    assert captured["timeout"] == 5


def test_repo_short_sha_returns_none_on_git_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="not a repo")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)

    assert stack._repo_short_sha(tmp_path) is None


def test_start_orchestrator_sets_tool_sentinels_by_default(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 4321

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return FakeProc()

    monkeypatch.setattr(stack, "LOG_DIR", tmp_path)
    monkeypatch.setattr(stack, "_pids_on_port", lambda _port: [])
    monkeypatch.setattr(stack, "_write_orchestrator_marker", lambda **_kwargs: tmp_path / "marker")
    monkeypatch.setattr(stack, "wait_for_health", lambda *args, **kwargs: True)
    monkeypatch.setattr(stack, "_set_oom_protection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stack.subprocess, "Popen", fake_popen)

    info = stack.start_orchestrator()

    assert info is not None
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["AUTOPILOT_TOOL_SENTINELS"] == "1"
    assert env["ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT"] == "1"


def test_start_orchestrator_refuses_unmocked_pytest_spawn(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "unit-test")

    with pytest.raises(RuntimeError, match="refusing to start the orchestrator API from pytest"):
        stack.start_orchestrator()


@pytest.fixture
def _no_running_cpu_bench(monkeypatch):
    """main() refuses lifecycle commands (exit 2) while ANY CPU bench runs on the
    host; the parser tests below are about argument defaults, not that guard."""
    monkeypatch.setattr(stack, "guard_against_running_bench", lambda _command, _force: True)


def test_start_parser_compiles_registry_by_default(monkeypatch, _no_running_cpu_bench) -> None:
    captured: dict[str, object] = {}

    def fake_cmd_start(args: Namespace) -> int:
        captured["compile_registry"] = args.compile_registry
        captured["numa_mode"] = args.numa_mode
        return 0

    monkeypatch.setattr(stack_commands, "cmd_start", fake_cmd_start)
    monkeypatch.setattr(sys, "argv", ["orchestrator_stack.py", "start"])

    assert stack.main() == 0
    assert captured["compile_registry"] is True
    # ESC-8 Fix 4: --numa-mode default is now None (inferred from the running
    # fleet by cmd_start) instead of a hardcoded "full".
    assert captured["numa_mode"] is None


def test_start_parser_accepts_explicit_numa_mode_both(monkeypatch, _no_running_cpu_bench) -> None:
    captured: dict[str, object] = {}

    def fake_cmd_start(args: Namespace) -> int:
        captured["numa_mode"] = args.numa_mode
        return 0

    monkeypatch.setattr(stack_commands, "cmd_start", fake_cmd_start)
    monkeypatch.setattr(sys, "argv", ["orchestrator_stack.py", "start", "--numa-mode", "both"])

    assert stack.main() == 0
    assert captured["numa_mode"] == "both"


def test_start_parser_accepts_no_compile_registry(monkeypatch, _no_running_cpu_bench) -> None:
    captured: dict[str, object] = {}

    def fake_cmd_start(args: Namespace) -> int:
        captured["compile_registry"] = args.compile_registry
        return 0

    monkeypatch.setattr(stack_commands, "cmd_start", fake_cmd_start)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator_stack.py", "start", "--no-compile-registry"],
    )

    assert stack.main() == 0
    assert captured["compile_registry"] is False


def test_cmd_start_compiles_registry_by_default_arg_absence(
    monkeypatch, isolated_runtime_facts_tmp: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_or_compile(**kwargs):
        calls.append(kwargs)
        return {"roles": {}}

    import src.registry.registry_compiler as compiler

    monkeypatch.setattr(compiler, "load_or_compile", fake_load_or_compile)

    args = _stack_gate_args(
        stack_profile="default",
        validate_only=True,
        compile_descriptors=False,
        allow_incomplete_descriptors=False,
    )

    assert stack_commands.cmd_start(args) == 0
    assert len(calls) == 1
    assert calls[0]["output_path"].name == "model_registry.yaml"
    assert calls[0]["cache_key_path"].name == ".lean_cache_key"
    assert not (isolated_runtime_facts_tmp / "orchestrator_runtime_facts.json").exists()


def test_cmd_start_infers_missing_numa_mode_from_realized_fleet(
    monkeypatch, isolated_runtime_facts_tmp: Path
) -> None:
    # ESC-8 Fix 4: with no --numa-mode, cmd_start infers the mode from the
    # running fleet instead of defaulting to "full".
    from scripts.server import realized_fleet

    modes: list[str] = []

    class FakeRegistryLoader:
        pass

    def fake_filter(servers, mode):
        modes.append(mode)
        return servers

    monkeypatch.setattr(realized_fleet, "derive_realized_numa_mode", lambda **_k: "quarter")
    monkeypatch.setattr(stack_commands, "RegistryLoader", FakeRegistryLoader)
    monkeypatch.setattr(stack_commands, "apply_host_prerequisites", lambda **_kwargs: True)
    monkeypatch.setattr(stack_commands, "check_free_memory", lambda: 999)
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(stack_commands, "_filter_by_numa_mode", fake_filter)
    monkeypatch.setattr(stack_commands, "_prewarm_all", lambda *a, **k: None)
    monkeypatch.setattr(stack_commands, "is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        stack_commands,
        "start_server",
        lambda port, roles, *a, **k: stack_commands.ProcessInfo(
            role=roles[0],
            pid=123,
            port=port,
            started_at="now",
            model_path="dev",
            log_file="dev.log",
        ),
    )
    monkeypatch.setattr(
        stack_commands,
        "start_orchestrator",
        lambda *_args, **_kwargs: stack_commands.ProcessInfo(
            role="orchestrator",
            pid=456,
            port=8000,
            started_at="now",
            model_path="dev",
            log_file="dev.log",
        ),
    )
    saved: list[dict[str, stack_commands.ProcessInfo]] = []
    monkeypatch.setattr(stack_commands, "save_state", lambda state: saved.append(dict(state)))

    args = _stack_gate_args(
        dev=True,
        only=None,
        include_warm=None,
        skip_host_prereqs=True,
        compile_registry=False,
        compile_descriptors=False,
        skip_stack_change_gate=True,
        repair_embeddings=False,
    )

    assert not hasattr(args, "numa_mode")
    assert stack_commands.cmd_start(args) == 0
    assert modes == ["quarter"]
    assert saved
    assert (isolated_runtime_facts_tmp / "orchestrator_runtime_facts.json").is_file()


def test_cmd_start_cold_start_falls_back_to_env_then_both(
    monkeypatch, isolated_runtime_facts_tmp: Path
) -> None:
    # Q38-T6: with no --numa-mode AND no live fleet, cmd_start must read
    # ORCHESTRATOR_STACK_NUMA_MODE before falling back to the ratified
    # production mode 'both' — never the retired 'quarter'. This is what makes
    # an unflagged cold start produce the production full+half lineup.
    from scripts.server import realized_fleet

    modes: list[str] = []

    class FakeRegistryLoader:
        pass

    def fake_filter(servers, mode):
        modes.append(mode)
        return servers

    monkeypatch.setattr(realized_fleet, "derive_realized_numa_mode", lambda **_k: None)
    monkeypatch.setattr(stack_commands, "RegistryLoader", FakeRegistryLoader)
    monkeypatch.setattr(stack_commands, "apply_host_prerequisites", lambda **_kwargs: True)
    monkeypatch.setattr(stack_commands, "check_free_memory", lambda: 999)
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(stack_commands, "_filter_by_numa_mode", fake_filter)
    monkeypatch.setattr(stack_commands, "_prewarm_all", lambda *a, **k: None)
    monkeypatch.setattr(stack_commands, "is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        stack_commands,
        "start_server",
        lambda port, roles, *a, **k: stack_commands.ProcessInfo(
            role=roles[0],
            pid=123,
            port=port,
            started_at="now",
            model_path="dev",
            log_file="dev.log",
        ),
    )
    monkeypatch.setattr(
        stack_commands,
        "start_orchestrator",
        lambda *_args, **_kwargs: stack_commands.ProcessInfo(
            role="orchestrator",
            pid=456,
            port=8000,
            started_at="now",
            model_path="dev",
            log_file="dev.log",
        ),
    )
    saved: list[dict[str, stack_commands.ProcessInfo]] = []
    monkeypatch.setattr(stack_commands, "save_state", lambda state: saved.append(dict(state)))

    args = _stack_gate_args(
        dev=True,
        only=None,
        include_warm=None,
        skip_host_prereqs=True,
        compile_registry=False,
        compile_descriptors=False,
        skip_stack_change_gate=True,
        repair_embeddings=False,
    )
    assert not hasattr(args, "numa_mode")

    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    assert stack_commands.cmd_start(args) == 0
    assert modes == ["both"], "cold start must default to the ratified production mode 'both'"

    modes.clear()
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    assert stack_commands.cmd_start(args) == 0
    assert modes == ["full"], (
        "cold start must honor ORCHESTRATOR_STACK_NUMA_MODE before the default"
    )


def test_cmd_start_repair_embeddings_uses_configured_embedder_pool(
    monkeypatch, isolated_runtime_facts_tmp: Path
) -> None:
    from scripts.maintenance import repair_episodic_embeddings as repair

    class FakeRegistryLoader:
        pass

    diagnose_reports = iter([Namespace(healthy=False), Namespace(healthy=True)])
    repair_calls: list[dict[str, object]] = []

    monkeypatch.setattr(stack_commands, "_run_stack_change_launch_gate", lambda _args: True)
    monkeypatch.setattr(stack_commands, "RegistryLoader", FakeRegistryLoader)
    monkeypatch.setattr(stack_commands, "apply_host_prerequisites", lambda **_kwargs: True)
    monkeypatch.setattr(stack_commands, "check_free_memory", lambda: 999)
    monkeypatch.setattr(stack_commands, "validate_model_paths", lambda: [])
    monkeypatch.setattr(stack_commands, "validate_against_registry", lambda: [])
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(repair, "diagnose", lambda *_args, **_kwargs: next(diagnose_reports))
    monkeypatch.setattr(repair, "print_report", lambda _report: None)
    monkeypatch.setattr(repair, "run_repair", lambda **kwargs: repair_calls.append(kwargs))

    args = _stack_gate_args(
        dev=False,
        only=["missing_role"],
        include_warm=None,
        skip_host_prereqs=True,
        compile_registry=False,
        compile_descriptors=False,
        skip_stack_change_gate=True,
        repair_embeddings=True,
    )

    assert stack_commands.cmd_start(args) == 1
    assert repair_calls
    assert repair_calls[0]["servers"] == repair.DEFAULT_EMBEDDER_SERVERS
    assert repair_calls[0]["base_port"] == repair.DEFAULT_EMBEDDER_BASE_PORT
    assert repair_calls[0]["max_db_growth"] == repair.DEFAULT_MAX_DB_GROWTH
    assert not (isolated_runtime_facts_tmp / "orchestrator_runtime_facts.json").exists()


def test_stack_change_launch_gate_failure_blocks_launch(monkeypatch, capsys) -> None:
    def fake_run(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="summary: failed\n", stderr="boom\n")

    monkeypatch.setattr(stack_commands.subprocess, "run", fake_run)

    assert not stack_commands._run_stack_change_launch_gate(_stack_gate_args())
    out = capsys.readouterr().out
    assert "summary: failed" in out
    assert "boom" in out
    assert "refusing launch" in out


def test_stack_change_launch_gate_has_explicit_skip(monkeypatch, capsys) -> None:
    def fake_run(*_args, **_kwargs):
        raise AssertionError("gate subprocess should not run")

    monkeypatch.setattr(stack_commands.subprocess, "run", fake_run)

    assert stack_commands._run_stack_change_launch_gate(
        _stack_gate_args(skip_stack_change_gate=True)
    )
    assert "SKIPPED (--skip-stack-change-gate)" in capsys.readouterr().out


def test_reload_embedders_uses_listener_pid_helper(monkeypatch) -> None:
    original_info = stack.ProcessInfo(
        role="embedder",
        pid=111,
        port=8090,
        started_at="before",
        model_path="old",
        log_file="old.log",
    )
    replacement_info = stack.ProcessInfo(
        role="embedder",
        pid=222,
        port=8090,
        started_at="after",
        model_path="new",
        log_file="new.log",
    )
    state = {"server_8090": original_info, "embedder": original_info}
    killed: list[int] = []
    pid_helper_calls: list[int] = []
    saved: list[dict[str, stack.ProcessInfo]] = []
    refreshed: list[tuple[str, dict[str, stack.ProcessInfo]]] = []

    monkeypatch.setattr(stack, "EMBEDDER_PORTS", [8090])
    monkeypatch.setattr(stack, "load_state", lambda: state)
    monkeypatch.setattr(stack, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(
        stack_commands,
        "_refresh_runtime_facts_manifest",
        lambda source, state, **_kw: refreshed.append((source, dict(state))),
    )
    monkeypatch.setattr(stack, "RegistryLoader", lambda: object())
    monkeypatch.setattr(stack, "kill_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(stack, "is_port_in_use", lambda port: port == 8090)
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)

    def fake_pids_on_port(port: int) -> list[int]:
        pid_helper_calls.append(port)
        return [333]

    monkeypatch.setattr(stack, "_pids_on_port", fake_pids_on_port)
    monkeypatch.setattr(
        stack,
        "start_server",
        lambda *args, **kwargs: replacement_info,
    )

    rc = stack.cmd_reload(Namespace(components=["embedders"]))

    assert rc == 0
    assert killed == [111, 333]
    assert pid_helper_calls == [8090]
    assert saved[-1]["server_8090"] == replacement_info
    assert saved[-1]["embedder"] == replacement_info
    assert refreshed == [("stack_reload", saved[-1])]


def test_reload_document_formalizer_uses_auxiliary_starter(monkeypatch) -> None:
    old_info = stack.ProcessInfo(
        role="document_formalizer",
        pid=111,
        port=9001,
        started_at="before",
        model_path="old",
        log_file="old.log",
    )
    new_info = stack.ProcessInfo(
        role="document_formalizer",
        pid=222,
        port=9001,
        started_at="after",
        model_path="LightOnOCR-2-1B-bbox",
        log_file="document_formalizer.log",
    )
    state = {"document_formalizer": old_info}
    killed: list[int] = []
    pid_helper_calls: list[int] = []
    saved: list[dict[str, stack.ProcessInfo]] = []
    refreshed: list[tuple[str, dict[str, stack.ProcessInfo]]] = []

    monkeypatch.setattr(stack, "load_state", lambda: state)
    monkeypatch.setattr(stack, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(
        stack_commands,
        "_refresh_runtime_facts_manifest",
        lambda source, state, **_kw: refreshed.append((source, dict(state))),
    )
    monkeypatch.setattr(
        stack,
        "RegistryLoader",
        lambda: (_ for _ in ()).throw(AssertionError("registry should not load")),
    )
    monkeypatch.setattr(stack, "kill_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)

    def fake_pids_on_port(port: int) -> list[int]:
        pid_helper_calls.append(port)
        return [333]

    monkeypatch.setattr(stack, "_pids_on_port", fake_pids_on_port)
    monkeypatch.setattr(stack, "start_document_formalizer", lambda **_kw: new_info)

    rc = stack.cmd_reload(Namespace(components=["document_formalizer"]))

    assert rc == 0
    assert killed == [333]
    assert pid_helper_calls == [9001]
    assert saved[-1] == {"document_formalizer": new_info}
    assert refreshed == [("stack_reload", {"document_formalizer": new_info})]


# =============================================================================
# W5 — reload must RESTART an aux service, not merely kill it
# =============================================================================
#
# THE DEFECT THESE TESTS PIN (fixed 2026-08-02):
#
# `whisper` (9000) and `sd_server` (8190) are in PORT_MAP but in neither
# HOT_SERVERS nor WARM_SERVERS. `cmd_reload` dispatched auxiliaries through a
# hand-written `elif` chain that named only document_formalizer and
# handoff_dashboard, so these two fell through to `elif component in PORT_MAP:`.
# That branch killed every listener on the port and then called `start_server()`
# — the llama-server path — with `roles=["whisper"]`, a role no registry
# declares. The start could only fail, so the command killed the service, never
# restarted it, and returned 1.
#
# WHY THESE FAIL ON THE PRE-FIX CODE: each asserts rc == 0, that the service's
# own starter ran, and that a live state row was written. Against the old
# dispatch the run reaches `start_server()` instead — which these tests make
# raise — so it errors out before any of that. The `RegistryLoader` and
# `start_server` traps below are the load-bearing part: without them the old
# code would merely return 1 and the tests would still fail, but for a reason
# that does not distinguish "took the llama-server path" from "starter said no".
@pytest.mark.parametrize(
    ("component", "port", "starter", "model_label"),
    [
        ("whisper", 9000, "start_whisper", "whisper.cpp large-v3-turbo"),
        ("sd_server", 8190, "start_sd_server", "ernie-image-turbo-Q8_0.gguf"),
        ("tts", 9002, "start_tts", "Qwen3-TTS-12Hz-0.6B"),
    ],
)
def test_reload_aux_service_restarts_it(
    monkeypatch, component: str, port: int, starter: str, model_label: str
) -> None:
    old_info = stack.ProcessInfo(
        role=component, pid=111, port=port, started_at="before",
        model_path="old", log_file="old.log",
    )
    new_info = stack.ProcessInfo(
        role=component, pid=222, port=port, started_at="after",
        model_path=model_label, log_file=f"{component}.log",
    )
    state = {component: old_info}
    killed: list[int] = []
    pid_helper_calls: list[int] = []
    saved: list[dict[str, stack.ProcessInfo]] = []
    started: list[str] = []

    monkeypatch.setattr(stack, "load_state", lambda: state)
    monkeypatch.setattr(stack, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(
        stack_commands, "_refresh_runtime_facts_manifest", lambda *a, **kw: None
    )
    # An aux service is not a registry role. Touching either of these means the
    # llama-server path was taken — the exact pre-fix behaviour.
    monkeypatch.setattr(
        stack, "RegistryLoader",
        lambda: (_ for _ in ()).throw(AssertionError("registry must not load for an aux service")),
    )
    monkeypatch.setattr(
        stack, "start_server",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError(f"{component} must not route through start_server()")
        ),
    )
    monkeypatch.setattr(stack, "kill_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)

    def fake_pids_on_port(p: int) -> list[int]:
        pid_helper_calls.append(p)
        return [333]

    monkeypatch.setattr(stack, "_pids_on_port", fake_pids_on_port)

    def fake_start(**_kw):
        started.append(component)
        return new_info

    monkeypatch.setattr(stack, starter, fake_start)

    rc = stack.cmd_reload(Namespace(components=[component]))

    assert rc == 0, f"reload {component} must succeed"
    assert killed == [333], "the old listener must be stopped"
    assert pid_helper_calls == [port], "must act on the declared port"
    # The restart half — this is what was missing.
    assert started == [component], f"reload {component} must call its own starter"
    assert saved[-1] == {component: new_info}, "state must record the RESTARTED process"


def test_reload_aux_service_reports_failure_without_leaving_a_stale_state_row(
    monkeypatch,
) -> None:
    """A starter that fails must return 1 AND drop the dead row from state.

    Leaving the pre-kill ProcessInfo behind would make `status` report a PID that
    was killed seconds earlier — the state file is an observation of a launch, so
    a failed relaunch must not leave the previous observation standing.
    """
    old_info = stack.ProcessInfo(
        role="whisper", pid=111, port=9000, started_at="before",
        model_path="old", log_file="old.log",
    )
    state = {"whisper": old_info}
    saved: list[dict[str, stack.ProcessInfo]] = []

    monkeypatch.setattr(stack, "load_state", lambda: state)
    monkeypatch.setattr(stack, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(stack_commands, "_refresh_runtime_facts_manifest", lambda *a, **kw: None)
    monkeypatch.setattr(stack, "kill_process", lambda _pid: None)
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(stack, "_pids_on_port", lambda _port: [333])
    monkeypatch.setattr(stack, "start_whisper", lambda **_kw: None)

    assert stack.cmd_reload(Namespace(components=["whisper"])) == 1
    assert "whisper" not in state


def test_every_declared_aux_service_is_reloadable(monkeypatch) -> None:
    """No declared service may be startable-but-not-reloadable.

    This is the structural guard, and it is the one that would have caught the
    original defect without anyone thinking to test whisper specifically: it
    derives its cases from the registry, so a service added to
    launch_manifest.yaml is covered the moment it is declared. A test that named
    the services explicitly would have been written against the same incomplete
    list that caused the bug.
    """
    from scripts.server.stack_manifest import AUX_SERVICES

    monkeypatch.setattr(stack, "save_state", lambda _value: None)
    monkeypatch.setattr(stack_commands, "_refresh_runtime_facts_manifest", lambda *a, **kw: None)
    monkeypatch.setattr(stack, "kill_process", lambda _pid: None)
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(stack, "_pids_on_port", lambda _port: [])
    monkeypatch.setattr(
        stack, "RegistryLoader",
        lambda: (_ for _ in ()).throw(AssertionError("registry must not load for an aux service")),
    )
    monkeypatch.setattr(
        stack, "start_server",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("aux service hit start_server()")),
    )

    for name, service in AUX_SERVICES.items():
        info = stack.ProcessInfo(
            role=name, pid=222, port=service.port, started_at="after",
            model_path=service.model_label, log_file=service.log,
        )
        calls: list[str] = []

        def fake_start(_name: str = name, _info=info, **_kw) -> stack.ProcessInfo:
            calls.append(_name)
            return _info

        monkeypatch.setattr(stack, "load_state", lambda: {})
        monkeypatch.setattr(stack, f"start_{name}", fake_start, raising=False)
        monkeypatch.setattr(stack, "start_aux_service", lambda n, **_kw: fake_start(n))

        assert stack.cmd_reload(Namespace(components=[name])) == 0, f"reload {name} failed"
        assert calls == [name], f"reload {name} did not start {name}"


def test_reload_refreshes_runtime_facts_manifest_after_successful_state_save(
    monkeypatch,
    tmp_path: Path,
) -> None:
    new_info = stack.ProcessInfo(
        role="orchestrator",
        pid=222,
        port=8000,
        started_at="after",
        model_path="uvicorn",
        log_file="orchestrator.log",
    )
    killed: list[int] = []
    saved: list[dict[str, stack.ProcessInfo]] = []
    refreshed: list[tuple[str, dict[str, stack.ProcessInfo]]] = []

    monkeypatch.setattr(stack, "load_state", lambda: {})
    monkeypatch.setattr(stack, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(stack, "kill_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(stack, "_pids_on_port", lambda port: [111] if port == 8000 else [])
    # cmd_reload passes stack_numa_mode= to start_orchestrator; accept it here.
    monkeypatch.setattr(stack, "start_orchestrator", lambda _profile=None, **_kw: new_info)
    monkeypatch.setattr(
        stack_commands,
        "_refresh_runtime_facts_manifest",
        lambda source, state, **_kw: refreshed.append((source, dict(state)))
        or tmp_path / "facts.json",
    )

    rc = stack.cmd_reload(Namespace(components=["orchestrator"], profile="production"))

    assert rc == 0
    assert killed == [111]
    assert saved[-1] == {"orchestrator": new_info}
    assert refreshed == [("stack_reload", {"orchestrator": new_info})]


def test_preserved_process_info_records_listener_pid(monkeypatch) -> None:
    monkeypatch.setattr(stack_commands, "_pids_on_port", lambda port: [777])

    info = stack_commands._preserved_process_info(
        "frontdoor",
        8070,
        "preserved:frontdoor",
        "llama-server-8070.log",
    )

    assert info is not None
    assert info == stack_commands.ProcessInfo(
        role="frontdoor",
        pid=777,
        port=8070,
        started_at=info.started_at,
        model_path="preserved:frontdoor",
        log_file="llama-server-8070.log",
    )


def test_scan_known_ports_derives_manifest_aux_ports(monkeypatch) -> None:
    captured: list[list[int]] = []

    monkeypatch.setattr(stack_commands, "HOT_SERVERS", [{"port": 8070}])
    monkeypatch.setattr(stack_commands, "WARM_SERVERS", [{"port": 8085}])
    monkeypatch.setattr(stack_commands, "NUMA_REPLICA_PORTS", {8180})
    monkeypatch.setattr(stack_commands, "DOCKER_SERVICES", [{"port": 8088}])
    monkeypatch.setattr(stack_commands, "_stack_prior_serving_ports", lambda: set())
    monkeypatch.setattr(
        stack_commands,
        "PORT_MAP",
        {
            "orchestrator": 8000,
            "sd_server": 8190,
            "whisper": 9000,
            "document_formalizer": 9001,
            "manifest_only_aux": 9010,
        },
    )

    def fake_scan(ports):
        captured.append(list(ports))
        return {}

    monkeypatch.setattr(stack_commands._stack_processes, "scan_known_ports", fake_scan)

    assert stack_commands._scan_known_ports() == {}
    assert captured == [[8000, 8070, 8085, 8088, 8180, 8190, 9000, 9001, 9010]]


def test_stack_prior_serving_ports_only_reads_live_roles(tmp_path: Path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        yaml.safe_dump(
            {
                "roles": {
                    "frontdoor": {
                        "deployment_status": "live_stack",
                        "serving": {"ports": [8070, 8080, 8180]},
                    },
                    "candidate": {
                        "deployment_status": "benchmark_or_candidate",
                        "serving": {"ports": [9999]},
                    },
                    "malformed": {
                        "deployment_status": "live_stack",
                        "serving": {"ports": ["9000", None]},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert stack_commands._stack_prior_serving_ports(priors) == {8070, 8080, 8180}


def test_scan_known_ports_includes_stack_prior_live_ports(monkeypatch) -> None:
    captured: list[list[int]] = []

    monkeypatch.setattr(stack_commands, "HOT_SERVERS", [{"port": 8070}])
    monkeypatch.setattr(stack_commands, "WARM_SERVERS", [])
    monkeypatch.setattr(stack_commands, "NUMA_REPLICA_PORTS", set())
    monkeypatch.setattr(stack_commands, "DOCKER_SERVICES", [])
    monkeypatch.setattr(stack_commands, "PORT_MAP", {"orchestrator": 8000})
    monkeypatch.setattr(stack_commands, "_stack_prior_serving_ports", lambda: {9123})

    def fake_scan(ports):
        captured.append(list(ports))
        return {}

    monkeypatch.setattr(stack_commands._stack_processes, "scan_known_ports", fake_scan)

    assert stack_commands._scan_known_ports() == {}
    assert captured == [[8000, 8070, 9123]]


def test_status_attestation_detects_expected_model_basename() -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )

    assert (
        stack_commands._status_attestation(
            info,
            alive=True,
            cmdline=["llama-server", "-m", "/canonical/current.gguf"],
        )
        == "ok"
    )


def test_status_attestation_detects_model_drift() -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )

    assert (
        stack_commands._status_attestation(
            info,
            alive=True,
            cmdline=["llama-server", "-m", "/models/stale.gguf"],
        )
        == "model-drift"
    )


def test_status_attestation_detects_expected_mmproj_basename() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_vision",
        pid=123,
        port=8086,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="vision.log",
    )

    assert (
        stack_commands._status_attestation(
            info,
            alive=True,
            cmdline=[
                "llama-server",
                "-m",
                "/models/current.gguf",
                "--mmproj",
                "/canonical/mmproj-model-f16.gguf",
            ],
            launch_requirements={"mmproj_path": "/models/mmproj-model-f16.gguf"},
        )
        == "ok"
    )


def test_status_attestation_detects_mmproj_drift() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_vision",
        pid=123,
        port=8086,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="vision.log",
    )

    assert (
        stack_commands._status_attestation(
            info,
            alive=True,
            cmdline=[
                "llama-server",
                "-m",
                "/models/current.gguf",
                "--mmproj",
                "/models/stale-mmproj.gguf",
            ],
            launch_requirements={"mmproj_path": "/models/current-mmproj.gguf"},
        )
        == "mmproj-drift"
    )


def test_runtime_attestation_warnings_report_model_drift(monkeypatch) -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"frontdoor": info})
    monkeypatch.setattr(stack_commands, "_scan_known_ports", lambda: {})
    monkeypatch.setattr(stack_commands, "_stack_prior_launch_contracts", lambda: {})
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: ["llama-server", "-m", "/models/stale.gguf"],
    )

    assert stack_commands.runtime_attestation_warnings() == [
        "frontdoor pid 123 expected current.gguf; live cmdline has stale.gguf"
    ]


def test_runtime_attestation_warnings_report_unmanaged_listener(monkeypatch) -> None:
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(stack_commands, "_scan_known_ports", lambda: {8070: [123, 456]})

    assert stack_commands.runtime_attestation_warnings() == [
        "known stack port 8070 has unmanaged listener pid(s) 123,456"
    ]


def test_runtime_attestation_warnings_report_runtime_flag_drift(monkeypatch) -> None:
    info = stack_commands.ProcessInfo(
        role="worker_general",
        pid=123,
        port=8072,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="worker.log",
    )

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"worker_general": info})
    monkeypatch.setattr(stack_commands, "_scan_known_ports", lambda: {8072: [123]})
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        stack_commands,
        "_stack_prior_launch_contracts",
        lambda: {
            "worker_general": {
                "requirements": {
                    "model_path": "/models/current.gguf",
                    "draft_model_path": "/models/draft.gguf",
                },
                "runtime": {
                    "binary_path": "/opt/llama/bin/llama-server",
                    "cache": {
                        "context_tokens": 16384,
                        "slots": 1,
                        "ubatch": 512,
                        "kv_type_k": "q8_0",
                        "kv_type_v": "q8_0",
                        "no_mmap": False,
                        "mlock": True,
                    },
                    "flags": {
                        "flash_attn": True,
                        "jinja": True,
                        "spec": {
                            "enabled": True,
                            "type": "draft-mtp",
                            "draft_max": 2,
                            "draft_p_min": 0.0,
                            "threads_draft": 1,
                        },
                    },
                },
                "ports": [8072],
            }
        },
    )
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: [
            "/opt/llama/bin/llama-server",
            "-m",
            "/models/current.gguf",
            "-md",
            "/models/draft.gguf",
            "-c",
            "8192",
            "-np",
            "1",
            "-ub",
            "512",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--mlock",
            "--flash-attn",
            "off",
            "--jinja",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
            "--draft-p-min",
            "0.0",
            "--threads-draft",
            "1",
        ],
    )

    assert stack_commands.runtime_attestation_warnings() == [
        "worker_general pid 123 runtime context_tokens expected 16384; live cmdline has 8192",
        "worker_general pid 123 runtime flash_attn expected True; live cmdline has False",
    ]


def _worker_general_launch_contract(model_path: str, draft_model_path: str) -> tuple[dict, dict]:
    return (
        {
            "model_path": model_path,
            "draft_model_path": draft_model_path,
        },
        {
            "binary_path": "/opt/llama/bin/llama-server",
            "cache": {
                "context_tokens": 16384,
                "slots": 1,
                "ubatch": 512,
                "kv_type_k": "q8_0",
                "kv_type_v": "q8_0",
                "no_mmap": False,
            },
            "flags": {
                "flash_attn": False,
                "jinja": False,
                "reasoning": "off",
                "spec": {
                    "enabled": True,
                    "type": "draft-mtp",
                    "draft_model_path": draft_model_path,
                    "draft_max": 2,
                    "draft_p_min": 0.0,
                    "threads_draft": 16,
                },
            },
        },
    )


def test_worker_general_builder_omits_same_file_embedded_draft(monkeypatch) -> None:
    monkeypatch.setattr(
        stack,
        "_stack_prior_launch",
        lambda _role: _worker_general_launch_contract(
            "/models/gemma-mtp.gguf",
            "/models/gemma-mtp.gguf",
        ),
    )
    monkeypatch.setattr(stack, "_resolve_thread_count", lambda _role, _idx: "96")

    cmd = stack._build_worker_general_command(8072, "/models/fallback.gguf", None)

    assert "-md" not in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "2"
    assert cmd[cmd.index("--threads-draft") + 1] == "16"


def test_worker_general_builder_keeps_separate_draft_model(monkeypatch) -> None:
    monkeypatch.setattr(
        stack,
        "_stack_prior_launch",
        lambda _role: _worker_general_launch_contract(
            "/models/gemma-target.gguf",
            "/models/gemma-assistant.gguf",
        ),
    )
    monkeypatch.setattr(stack, "_resolve_thread_count", lambda _role, _idx: "96")

    cmd = stack._build_worker_general_command(8072, "/models/fallback.gguf", None)

    assert cmd[cmd.index("-md") + 1] == "/models/gemma-assistant.gguf"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"


def test_worker_general_launch_pins_target_and_draft_devices_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        stack,
        "_stack_prior_launch",
        lambda _role: _worker_general_launch_contract(
            "/models/gemma-target.gguf",
            "/models/gemma-assistant.gguf",
        ),
    )
    monkeypatch.setattr(stack, "_resolve_thread_count", lambda _role, _idx: "96")

    cmd = stack.build_server_command(
        None,
        8072,
        worker_pool_mode=True,
        worker_type="explore",
    )

    assert cmd[cmd.index("--device") + 1] == "none"
    assert cmd[cmd.index("--device-draft") + 1] == "none"


def test_generic_spec_launch_pins_target_and_draft_devices_to_cpu(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stack, "SLOT_SAVE_DIR", tmp_path)
    monkeypatch.setattr(stack, "_resolve_thread_count", lambda _role, _idx: "96")
    monkeypatch.setattr(
        stack,
        "_stack_prior_launch",
        lambda _role: (
            {"model_path": "/models/frontdoor.gguf"},
            {
                "binary_path": "/opt/llama/bin/llama-server",
                "cache": {
                    "context_tokens": 32768,
                    "slots": 1,
                    "ubatch": 8192,
                    "kv_type_k": "q8_0",
                    "kv_type_v": "q8_0",
                    "slot_save_path": str(tmp_path / "frontdoor"),
                },
                "flags": {
                    "flash_attn": False,
                    "jinja": False,
                    "spec": {
                        "enabled": True,
                        "type": "draft-mtp",
                        "draft_model_path": "/models/frontdoor.gguf",
                        "draft_max": 4,
                    },
                },
            },
        ),
    )
    role_config = Namespace(
        name="frontdoor",
        model=Namespace(full_path="/models/frontdoor.gguf"),
        acceleration=Namespace(type="none"),
    )

    cmd = stack.build_server_command(role_config, 8070)

    assert cmd[cmd.index("--device") + 1] == "none"
    assert cmd[cmd.index("--device-draft") + 1] == "none"


def test_generic_gpu_role_emits_its_declared_device_instead_of_none(
    monkeypatch, tmp_path
) -> None:
    """The default builder must EMIT a declared device, not be overwritten to CPU.

    Regression for the 2026-08-01 finding: architect_general declares
    `device: ROCm0` + `ngl: all`, resolved correctly to the HIP binary, and then
    launched as `--device none --device-draft none` because `_build_role_command`
    never read `flags["device"]` and `_append_device_args` assumed CPU. It served a
    GPU-declared 27B on 24 CPU threads while attesting healthy.
    """
    monkeypatch.setattr(stack, "SLOT_SAVE_DIR", tmp_path)
    monkeypatch.setattr(stack, "_resolve_thread_count", lambda _role, _idx: "24")
    monkeypatch.setattr(
        stack,
        "_stack_prior_launch",
        lambda _role: (
            {"model_path": "/models/architect.gguf"},
            {
                "binary_path": "/opt/llama/build-hip/bin/llama-server",
                "cache": {
                    "context_tokens": 16384,
                    "slots": 2,
                    "ubatch": 8192,
                    "slot_save_path": str(tmp_path / "architect_general"),
                },
                "flags": {
                    "flash_attn": True,
                    "jinja": True,
                    "device": "ROCm0",
                    "n_gpu_layers": "all",
                    "spec": {
                        "enabled": True,
                        "type": "draft-mtp",
                        "draft_model_path": "/models/architect.gguf",
                        "draft_max": 4,
                    },
                },
            },
        ),
    )
    role_config = Namespace(
        name="architect_general",
        model=Namespace(full_path="/models/architect.gguf"),
        acceleration=Namespace(type="none"),
    )

    cmd = stack.build_server_command(role_config, 8083)

    assert cmd[cmd.index("--device") + 1] == "ROCm0"
    assert "none" not in _all_values(cmd, "--device")
    # -ngl must ride along: selecting a device while offloading nothing to it is a
    # GPU launch in name only.
    assert cmd[cmd.index("-ngl") + 1] == "all"
    # The NEXTN self-draft lives in the target's own GGUF, so it must not be
    # stranded on the CPU under a GPU-resident target.
    assert cmd[cmd.index("--device-draft") + 1] == "ROCm0"


def _all_values(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag and i + 1 < len(cmd)]


def test_runtime_attestation_warns_when_live_device_contradicts_declaration() -> None:
    """The attestation gap that made the architect_general defect invisible.

    Every other declared field was checked; `device` was not. A process could run on
    the wrong processor and still pass.
    """
    info = stack_commands.ProcessInfo(
        role="architect_general",
        pid=1935263,
        port=8083,
        started_at="now",
        model_path="/models/architect.gguf",
        log_file="architect.log",
    )

    warnings = stack_commands._runtime_attestation_warnings(
        "architect_general",
        info,
        [
            "/opt/llama/build-hip/bin/llama-server",
            "-m",
            "/models/architect.gguf",
            "--device",
            "none",
            "--device-draft",
            "none",
            "--spec-type",
            "draft-mtp",
        ],
        {
            "requirements": {"model_path": "/models/architect.gguf"},
            "runtime": {
                "binary_path": "/opt/llama/build-hip/bin/llama-server",
                "flags": {
                    "device": "ROCm0",
                    "spec": {"enabled": True, "type": "draft-mtp"},
                },
            },
        },
    )

    assert any("device expected ROCm0" in warning for warning in warnings)
    assert any("live cmdline has none" in warning for warning in warnings)
    assert any("device_draft expected ROCm0" in warning for warning in warnings)


def test_runtime_attestation_warns_when_cpu_role_lands_on_a_gpu() -> None:
    """Mirror image: an UNDECLARED device means CPU, so a live ROCm0 is drift too."""
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/frontdoor.gguf",
        log_file="frontdoor.log",
    )

    warnings = stack_commands._runtime_attestation_warnings(
        "frontdoor",
        info,
        [
            "llama-server",
            "-m",
            "/models/frontdoor.gguf",
            "--device",
            "ROCm0",
        ],
        {
            "requirements": {"model_path": "/models/frontdoor.gguf"},
            "runtime": {"flags": {"spec": {"enabled": False}}},
        },
    )

    assert any("device expected none" in warning for warning in warnings)


def test_runtime_attestation_passes_when_device_matches_declaration() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_vision",
        pid=123,
        port=8086,
        started_at="now",
        model_path="/models/vl.gguf",
        log_file="vision.log",
    )

    warnings = stack_commands._runtime_attestation_warnings(
        "worker_vision",
        info,
        [
            "llama-server",
            "-m",
            "/models/vl.gguf",
            "--device",
            "ROCm0",
        ],
        {
            "requirements": {"model_path": "/models/vl.gguf"},
            "runtime": {"flags": {"device": "ROCm0", "spec": {"enabled": False}}},
        },
    )

    assert warnings == []


# ---------------------------------------------------------------------------
# Derived attestation coverage (2026-08-02)
#
# The attestation checklist used to be a hand-written sequence of field names, so
# anything the compiler emitted but nobody had listed was never compared and the
# omission was invisible. These tests pin the derived behaviour: the fields that
# were previously declared-but-unverified now fire, and a field with no mapping is
# itself reported instead of being silently skipped.
# ---------------------------------------------------------------------------


def _attestation_info(role: str = "architect_general", port: int = 8083):
    return stack_commands.ProcessInfo(
        role=role,
        pid=321,
        port=port,
        started_at="now",
        model_path="/models/m.gguf",
        log_file="a.log",
    )


_ATTEST_BASE_CMDLINE = [
    "/opt/llama/bin/llama-server",
    "-m",
    "/models/m.gguf",
    "--device",
    "ROCm0",
]


def _attest_contract(flags: dict | None = None, cache: dict | None = None) -> dict:
    return {
        "requirements": {"model_path": "/models/m.gguf"},
        "runtime": {
            "binary_path": "/opt/llama/bin/llama-server",
            "cache": cache or {},
            "flags": {"device": "ROCm0", **(flags or {})},
        },
        "ports": [8083],
    }


def test_runtime_attestation_warns_when_declared_ngl_is_absent_from_cmdline() -> None:
    """A GPU role declaring -ngl but launched without it offloads NOTHING.

    Same failure path as the `device` omission: `--device ROCm0` with no -ngl is a
    GPU launch in name only, and the hand-written checklist never looked.
    """
    warnings = stack_commands._runtime_attestation_warnings(
        "architect_general",
        _attestation_info(),
        _ATTEST_BASE_CMDLINE,
        _attest_contract(flags={"n_gpu_layers": "all"}),
    )

    assert warnings == [
        "architect_general pid 321 runtime n_gpu_layers expected all; "
        "live cmdline has no -ngl"
    ]


def test_runtime_attestation_warns_on_declared_serving_flag_drift() -> None:
    """-ngl / --image-min-tokens / --cache-ram are emitted from the record.

    `cache_ram: 0` is a DECLARED value (it disables the prompt cache), so the
    comparison must be against presence, never truthiness.
    """
    warnings = stack_commands._runtime_attestation_warnings(
        "worker_vision",
        _attestation_info(role="worker_vision", port=8086),
        _ATTEST_BASE_CMDLINE
        + ["-ngl", "0", "--image-min-tokens", "256"],
        _attest_contract(
            flags={"n_gpu_layers": 999, "image_min_tokens": 1024, "cache_ram": 0}
        ),
    )

    assert warnings == [
        "worker_vision pid 321 runtime cache_ram expected 0; "
        "live cmdline has no --cache-ram",
        "worker_vision pid 321 runtime image_min_tokens expected 1024; "
        "live cmdline has 256",
        "worker_vision pid 321 runtime n_gpu_layers expected 999; live cmdline has 0",
    ]


def test_runtime_attestation_passes_when_declared_serving_flags_match() -> None:
    warnings = stack_commands._runtime_attestation_warnings(
        "worker_vision",
        _attestation_info(role="worker_vision", port=8086),
        _ATTEST_BASE_CMDLINE
        + ["-ngl", "999", "--image-min-tokens", "1024", "--cache-ram", "0"],
        _attest_contract(
            flags={"n_gpu_layers": 999, "image_min_tokens": 1024, "cache_ram": 0}
        ),
    )

    assert warnings == []


def test_runtime_attestation_warns_on_unmapped_declared_field() -> None:
    """THE inversion: a declared field nothing verifies is a finding, not a skip."""
    warnings = stack_commands._runtime_attestation_warnings(
        "architect_general",
        _attestation_info(),
        _ATTEST_BASE_CMDLINE,
        _attest_contract(flags={"tensor_split": "0.5,0.5"}),
    )

    assert warnings == [
        "architect_general pid 321 declares runtime field runtime.flags.tensor_split "
        "but nothing attests it (no entry in _RUNTIME_FIELD_CHECKS)"
    ]


def test_runtime_attestation_reports_lost_coverage_for_floor_field() -> None:
    """The hand list survives as a FLOOR: deleting a mapping is caught.

    Coverage is gained by deriving from the producer, and none is lost, because a
    required field dropping out of the mapping table is itself reported.
    """
    # No `device` in this record: the path is required by the FLOOR alone, which is
    # what makes a deleted mapping detectable even when the producer stops emitting.
    contract = {
        "requirements": {"model_path": "/models/m.gguf"},
        "runtime": {
            "binary_path": "/opt/llama/bin/llama-server",
            "cache": {},
            "flags": {},
        },
        "ports": [8083],
    }
    removed = stack_commands._RUNTIME_FIELD_CHECKS.pop("runtime.flags.device")
    try:
        warnings = stack_commands._runtime_attestation_warnings(
            "architect_general",
            _attestation_info(),
            ["/opt/llama/bin/llama-server", "-m", "/models/m.gguf"],
            contract,
        )
    finally:
        stack_commands._RUNTIME_FIELD_CHECKS["runtime.flags.device"] = removed

    assert warnings == [
        "architect_general pid 321 attestation coverage lost: runtime.flags.device "
        "is a required attested field with no entry in _RUNTIME_FIELD_CHECKS"
    ]


def test_every_floor_field_is_mapped() -> None:
    unmapped = [
        path
        for path in stack_commands._ATTESTED_RUNTIME_FIELDS
        if path not in stack_commands._RUNTIME_FIELD_CHECKS
    ]
    assert unmapped == []


def test_every_compiled_runtime_field_is_mapped() -> None:
    """The producer's own keys must all resolve, for every live role.

    This is the test that would have failed on 2026-08-01, when `device` was
    emitted by the compiler and absent from the checklist.
    """
    contracts = stack_commands._stack_prior_launch_contracts()
    if not contracts:
        pytest.skip("no compiled stack priors available")

    unmapped: set[str] = set()
    for contract in contracts.values():
        declared = stack_commands._declared_runtime_field_paths(
            contract.get("requirements") or {},
            contract.get("runtime") or {},
        )
        unmapped |= {
            path for path in declared if path not in stack_commands._RUNTIME_FIELD_CHECKS
        }

    assert sorted(unmapped) == []


def test_runtime_attestation_warns_when_spec_draft_declarations_disagree() -> None:
    """`-md` comes from spec.draft_model_path; the draft check reads requirements.

    One fact declared twice: compare the two declarations so a divergence is a
    finding rather than a check pointed at the value the launcher did not use.
    """
    warnings = stack_commands._runtime_attestation_warnings(
        "frontdoor",
        _attestation_info(role="frontdoor", port=8070),
        [
            "/opt/llama/bin/llama-server",
            "-m",
            "/models/m.gguf",
            "-md",
            "/models/d.gguf",
            "--spec-type",
            "draft-mtp",
        ],
        {
            "requirements": {
                "model_path": "/models/m.gguf",
                "draft_model_path": "/models/d.gguf",
            },
            "runtime": {
                "binary_path": "/opt/llama/bin/llama-server",
                "cache": {},
                "flags": {
                    "spec": {
                        "enabled": True,
                        "draft_model_path": "/models/other.gguf",
                    }
                },
            },
            "ports": [8070],
        },
    )

    assert warnings == [
        "frontdoor pid 321 runtime spec.draft_model_path is /models/other.gguf, but "
        "requirements.draft_model_path is /models/d.gguf; one fact is declared twice "
        "and the two declarations disagree"
    ]


def test_runtime_attestation_warns_on_declaration_the_launcher_cannot_emit() -> None:
    """--kv-hadamard was removed in v6; declaring it true is an unkept promise."""
    warnings = stack_commands._runtime_attestation_warnings(
        "frontdoor",
        _attestation_info(role="frontdoor", port=8070),
        _ATTEST_BASE_CMDLINE,
        _attest_contract(cache={"kv_hadamard": True}),
    )

    assert len(warnings) == 1
    assert "declares kv_hadamard=True but the launcher cannot emit it" in warnings[0]
    assert "no runtime effect" in warnings[0]


def test_runtime_attestation_ignores_kv_hadamard_declared_false() -> None:
    assert (
        stack_commands._runtime_attestation_warnings(
            "frontdoor",
            _attestation_info(role="frontdoor", port=8070),
            _ATTEST_BASE_CMDLINE,
            _attest_contract(cache={"kv_hadamard": False}),
        )
        == []
    )


@pytest.mark.parametrize(
    ("declared", "extra", "expect_warning"),
    [
        # Declared unified, launched with -kvu: clean.
        (True, ["-np", "2", "--kv-unified"], False),
        (True, ["-np", "2", "-kvu"], False),
        # THE 2026-09-24 DEFECT: declared unified, launched with an explicit -np and
        # no -kvu. llama-server resolves split KV (server.cpp:145-150). Must warn.
        (True, ["-np", "2"], True),
        # Last flag wins: a trailing negation defeats an earlier -kvu.
        (True, ["-np", "2", "-kvu", "--no-kv-unified"], True),
        # No -np at all = auto slots = unified by the kernel's own default.
        (True, [], False),
        # Declared split: explicit -np with no flag IS split; -kvu is drift.
        (False, ["-np", "2"], False),
        (False, ["-np", "2", "--no-kv-unified"], False),
        (False, ["-np", "2", "--kv-unified"], True),
    ],
)
def test_runtime_attestation_checks_kv_unified_as_resolved_by_the_server(
    declared: bool, extra: list[str], expect_warning: bool
) -> None:
    warnings = stack_commands._runtime_attestation_warnings(
        "architect_general",
        _attestation_info(),
        _ATTEST_BASE_CMDLINE + extra,
        _attest_contract(cache={"kv_unified": declared}),
    )
    kvu = [w for w in warnings if "kv_unified" in w]
    assert bool(kvu) is expect_warning, warnings


def test_runtime_attestation_checks_declared_draft_kv_types() -> None:
    contract = _attest_contract(cache={"draft_kv_type_k": "q8_0", "draft_kv_type_v": "q8_0"})
    clean = stack_commands._runtime_attestation_warnings(
        "architect_general",
        _attestation_info(),
        _ATTEST_BASE_CMDLINE + ["-ctkd", "q8_0", "-ctvd", "q8_0"],
        contract,
    )
    assert clean == []
    drift = stack_commands._runtime_attestation_warnings(
        "architect_general", _attestation_info(), _ATTEST_BASE_CMDLINE, contract
    )
    assert len([w for w in drift if "draft_kv_type" in w]) == 2, drift


def test_runtime_attestation_ignores_undeclared_kv_unified() -> None:
    assert (
        stack_commands._runtime_attestation_warnings(
            "architect_general",
            _attestation_info(),
            _ATTEST_BASE_CMDLINE + ["-np", "2"],
            _attest_contract(cache={"kv_unified": None}),
        )
        == []
    )


def test_runtime_attestation_accepts_embedded_nextn_without_md() -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/qwen-mtp.gguf",
        log_file="frontdoor.log",
    )

    warnings = stack_commands._runtime_attestation_warnings(
        "frontdoor",
        info,
        [
            "llama-server",
            "-m",
            "/models/qwen-mtp.gguf",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "4",
        ],
        {
            "requirements": {
                "model_path": "/models/qwen-mtp.gguf",
                "draft_model_path": "/models/qwen-mtp.gguf",
            },
            "runtime": {
                "flags": {
                    "spec": {
                        "enabled": True,
                        "type": "draft-mtp",
                        "draft_max": 4,
                    }
                }
            },
        },
    )

    assert warnings == []


def test_runtime_attestation_still_warns_when_separate_draft_missing() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_general",
        pid=123,
        port=8072,
        started_at="now",
        model_path="/models/gemma.gguf",
        log_file="worker.log",
    )

    warnings = stack_commands._runtime_attestation_warnings(
        "worker_general",
        info,
        [
            "llama-server",
            "-m",
            "/models/gemma.gguf",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
        ],
        {
            "requirements": {
                "model_path": "/models/gemma.gguf",
                "draft_model_path": "/models/gemma-assistant.gguf",
            },
            "runtime": {
                "flags": {
                    "spec": {
                        "enabled": True,
                        "type": "draft-mtp",
                        "draft_max": 2,
                    }
                }
            },
        },
    )

    assert warnings == [
        (
            "worker_general pid 123 runtime draft_model_path expected "
            "gemma-assistant.gguf; live cmdline has no -md"
        )
    ]


def test_launch_contract_for_process_canonicalizes_alias_role() -> None:
    info = stack_commands.ProcessInfo(
        role="worker_explore",
        pid=123,
        port=9999,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="worker.log",
    )
    contracts = {
        "worker_general": {
            "requirements": {"model_path": "/models/current.gguf"},
            "runtime": {"cache": {"context_tokens": 16384}},
            "ports": [8072],
        }
    }

    assert (
        stack_commands._launch_contract_for_process(
            "worker_explore",
            info,
            contracts,
        )
        == contracts["worker_general"]
    )


def test_runtime_attestation_warnings_match_replica_by_port(monkeypatch) -> None:
    info = stack_commands.ProcessInfo(
        role="server",
        pid=123,
        port=8282,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="worker-8282.log",
    )

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"server_8282": info})
    monkeypatch.setattr(stack_commands, "_scan_known_ports", lambda: {8282: [123]})
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        stack_commands,
        "_stack_prior_launch_contracts",
        lambda: {
            "worker_general": {
                "requirements": {
                    "model_path": "/models/current.gguf",
                    "draft_model_path": "/models/draft.gguf",
                },
                "runtime": {
                    "flags": {
                        "spec": {
                            "enabled": True,
                            "type": "draft-mtp",
                            "draft_max": 2,
                        },
                    },
                },
                "ports": [8282],
            }
        },
    )
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: [
            "llama-server",
            "-m",
            "/models/current.gguf",
            "-md",
            "/models/draft.gguf",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
        ],
    )

    assert stack_commands.runtime_attestation_warnings() == []


def test_cmd_status_prints_model_attestation_warning(monkeypatch, capsys) -> None:
    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )
    saved: list[dict[str, stack_commands.ProcessInfo]] = []

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"frontdoor": info})
    monkeypatch.setattr(stack_commands, "save_state", lambda state: saved.append(dict(state)))
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(stack_commands, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: ["llama-server", "-m", "/models/stale.gguf"],
    )

    rc = stack_commands.cmd_status(Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "ATTEST" in out
    assert "model-drift" in out
    assert "expected current.gguf" in out
    assert "live cmdline has stale.gguf" in out
    # ESC-8 Fix 2: `status` is now read-only — it no longer persists state.
    assert saved == []


def test_cmd_status_renders_absent_optional_auxiliary_once(monkeypatch, capsys) -> None:
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(stack_commands, "_episodic_embedding_status_line", lambda: "memory ok")

    assert stack_commands.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert sum(line.startswith("whisper") for line in out.splitlines()) == 1
    assert "9000" in out
    assert "unavailable_optional" in out
    assert "No components running" not in out


def test_cmd_status_surfaces_declared_service_missing_state_row(monkeypatch, capsys) -> None:
    """Active stack-prior services stay visible when state persistence misses them."""
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(
        stack_commands,
        "live_stack_role_records",
        lambda *_args, **_kwargs: {"frontdoor": {"serving": {"ports": [8070]}}},
    )
    monkeypatch.setattr(stack_commands, "DOCKER_SERVICES", [])
    monkeypatch.setattr(stack_commands, "OPTIONAL_AUXILIARY_ROLES", frozenset())
    monkeypatch.setattr(stack_commands, "is_port_in_use", lambda port: port == 8070)
    monkeypatch.setattr(
        stack_commands,
        "wait_for_health",
        lambda port, **_kwargs: port == 8070,
    )
    monkeypatch.setattr(stack_commands, "_episodic_embedding_status_line", lambda: "memory ok")

    assert stack_commands.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "frontdoor" in out
    assert "healthy" in out
    assert "state-missing" in out
    assert "manifest-declared; no state row" in out


def test_cmd_status_marks_declared_down_but_omits_inactive_manifest_roles(monkeypatch, capsys) -> None:
    """A declared-but-down service differs from a warm role absent from active priors."""
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(
        stack_commands,
        "live_stack_role_records",
        lambda *_args, **_kwargs: {"frontdoor": {"serving": {"ports": [8070]}}},
    )
    monkeypatch.setattr(stack_commands, "DOCKER_SERVICES", [])
    monkeypatch.setattr(stack_commands, "OPTIONAL_AUXILIARY_ROLES", frozenset())
    monkeypatch.setattr(stack_commands, "is_port_in_use", lambda _port: False)
    monkeypatch.setattr(stack_commands, "_episodic_embedding_status_line", lambda: "memory ok")

    assert stack_commands.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "frontdoor" in out
    assert "unavailable" in out
    assert "state-missing" in out
    assert "worker_fast" not in out
    assert "8102" not in out


def test_cmd_status_surfaces_manifest_declared_docker_service_without_state(monkeypatch, capsys) -> None:
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(stack_commands, "live_stack_role_records", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        stack_commands,
        "DOCKER_SERVICES",
        [{"name": "nextplaid-code", "port": 8088, "health_path": "/health"}],
    )
    monkeypatch.setattr(stack_commands, "OPTIONAL_AUXILIARY_ROLES", frozenset())
    monkeypatch.setattr(stack_commands, "is_port_in_use", lambda _port: False)
    monkeypatch.setattr(stack_commands, "_episodic_embedding_status_line", lambda: "memory ok")

    assert stack_commands.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "nextplaid-code" in out
    assert "8088" in out
    assert "unavailable" in out
    assert "state-missing" in out


def test_cmd_status_does_not_duplicate_present_optional_auxiliary(monkeypatch, capsys) -> None:
    info = stack_commands.ProcessInfo(
        role="whisper",
        pid=123,
        port=9000,
        started_at="now",
        model_path="faster-whisper-large-v3-turbo-int8",
        log_file="whisper.log",
    )
    monkeypatch.setattr(stack_commands, "load_state", lambda: {"whisper": info})
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(stack_commands, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(stack_commands, "_episodic_embedding_status_line", lambda: "memory ok")
    monkeypatch.setattr(stack_commands._stack_processes, "process_cmdline", lambda _pid: [])

    assert stack_commands.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    whisper_lines = [line for line in out.splitlines() if line.startswith("whisper")]
    assert len(whisper_lines) == 1
    # A PRESENT optional auxiliary must not ALSO be listed as unavailable.
    #
    # This used to assert `"unavailable_optional" not in out`, which tested the
    # right thing only by accident: whisper was the sole optional auxiliary, so
    # the string could not appear for any other reason. Registering `tts` as a
    # second optional auxiliary (2026-08-02, W4) made that proxy false — tts is
    # legitimately unavailable in this fixture and SHOULD be reported. Scope the
    # assertion to the service under test instead of the whole status surface.
    assert "unavailable_optional" not in whisper_lines[0]


def test_cmd_status_prints_episodic_embedding_health(monkeypatch, capsys) -> None:
    from scripts.maintenance import repair_episodic_embeddings as repair

    info = stack_commands.ProcessInfo(
        role="frontdoor",
        pid=123,
        port=8070,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="frontdoor.log",
    )

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"frontdoor": info})
    monkeypatch.setattr(stack_commands, "save_state", lambda _state: None)
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(stack_commands, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: ["llama-server", "-m", "/models/current.gguf"],
    )
    monkeypatch.setattr(
        repair,
        "diagnose",
        lambda *_args, **_kwargs: repair.HealthReport(
            n_db_routing=100,
            n_faiss_vectors=10,
            n_reembedded=50,
            overlap_live=0.5,
            faiss_coverage=0.1,
            healthy=False,
            orphan_count=90,
        ),
    )

    assert stack_commands.cmd_status(Namespace()) == 0
    out = capsys.readouterr().out
    assert "Episodic FAISS: ORPHANED" in out
    assert "10/100 indexed vectors" in out
    assert "90 live repairable lag/stale" in out


def test_episodic_status_keeps_legacy_training_artifact_non_blocking(monkeypatch) -> None:
    from scripts.maintenance import repair_episodic_embeddings as repair

    monkeypatch.setattr(
        repair,
        "diagnose",
        lambda *_args, **_kwargs: repair.HealthReport(
            n_db_routing=100,
            n_faiss_vectors=100,
            n_reembedded=900,
            overlap_live=0.42,
            faiss_coverage=1.0,
            healthy=True,
            orphan_count=0,
            n_id_map=100,
            id_map_overlap_live=1.0,
            id_map_matches_faiss=True,
            n_db_indexed=100,
            reembedded_stale_count=800,
        ),
    )

    line = stack_commands._episodic_embedding_status_line()

    assert line.startswith("Episodic FAISS: healthy")
    assert "ORPHANED" not in line
    assert "0 live repairable lag/stale" in line
    assert "training artifact reembedded overlap 42.0% (non-blocking, 800 stale)" in line


def test_cmd_status_prints_mmproj_attestation_warning(monkeypatch, capsys) -> None:
    info = stack_commands.ProcessInfo(
        role="worker_vision",
        pid=123,
        port=8086,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="vision.log",
    )
    saved: list[dict[str, stack_commands.ProcessInfo]] = []

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"server_8086": info})
    monkeypatch.setattr(stack_commands, "save_state", lambda state: saved.append(dict(state)))
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(stack_commands, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: [
            "llama-server",
            "-m",
            "/models/current.gguf",
            "--mmproj",
            "/models/stale-mmproj.gguf",
        ],
    )
    monkeypatch.setattr(
        stack_commands,
        "_stack_prior_launch_contracts",
        lambda: {
            "worker_vision": {
                "requirements": {"mmproj_path": "/models/current-mmproj.gguf"},
                "runtime": {},
                "ports": [8086],
            }
        },
    )

    rc = stack_commands.cmd_status(Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "mmproj-drift" in out
    assert "expected mmproj current-mmproj.gguf" in out
    assert "live cmdline has stale-mmproj.gguf" in out
    # ESC-8 Fix 2: `status` is now read-only — it no longer persists state.
    assert saved == []


def test_cmd_status_prints_runtime_contract_warning(monkeypatch, capsys) -> None:
    info = stack_commands.ProcessInfo(
        role="worker_general",
        pid=123,
        port=8072,
        started_at="now",
        model_path="/models/current.gguf",
        log_file="worker.log",
    )
    saved: list[dict[str, stack_commands.ProcessInfo]] = []

    monkeypatch.setattr(stack_commands, "load_state", lambda: {"worker_general": info})
    monkeypatch.setattr(stack_commands, "save_state", lambda state: saved.append(dict(state)))
    monkeypatch.setattr(stack_commands.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(stack_commands, "wait_for_health", lambda *a, **kw: True)
    monkeypatch.setattr(
        stack_commands,
        "_stack_prior_launch_contracts",
        lambda: {
            "worker_general": {
                "requirements": {"model_path": "/models/current.gguf"},
                "runtime": {
                    "cache": {"context_tokens": 16384},
                    "flags": {"flash_attn": True},
                },
                "ports": [8072],
            }
        },
    )
    monkeypatch.setattr(
        stack_commands._stack_processes,
        "process_cmdline",
        lambda _pid: [
            "llama-server",
            "-m",
            "/models/current.gguf",
            "-c",
            "8192",
            "--flash-attn",
            "off",
        ],
    )

    rc = stack_commands.cmd_status(Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "runtime context_tokens expected 16384; live cmdline has 8192" in out
    assert "runtime flash_attn expected True; live cmdline has False" in out
    # ESC-8 Fix 2: `status` is now read-only — it no longer persists state.
    assert saved == []


def test_reload_server_port_resolves_sub_full_instances() -> None:
    """server_<port> must resolve for ANY managed instance, not just primaries.

    PORT_MAP maps a role to its primary port only, so before _reload_server_port
    the sub-full instances (frontdoor's 8080/8180 halves) were unreachable by
    reload: the request printed "Unknown component", restarted nothing, and the
    stale half kept passing /health on its old cmdline — which is how a declared
    serving flag (chat_template_file) deployed to the full but never to a half.
    """
    from scripts.server.stack_manifest import HOT_SERVERS, PORT_MAP, WARM_SERVERS

    # Role-name spelling: unchanged PORT_MAP behavior.
    assert stack_commands._reload_server_port("frontdoor") == PORT_MAP["frontdoor"]

    # Every managed instance port is addressable via its state-file key.
    for server in HOT_SERVERS + WARM_SERVERS:
        port = server["port"]
        assert stack_commands._reload_server_port(f"server_{port}") == port

    # sd_server IS a PORT_MAP name, but cmd_reload's AUX_SERVICES branch
    # precedes this resolution (W5 ordering fix) — the helper only reports the
    # mapping, it never re-routes an auxiliary through start_server().
    assert stack_commands._reload_server_port("sd_server") == PORT_MAP["sd_server"]

    # Non-managed spellings stay unresolved (fall through to docker/unknown).
    assert stack_commands._reload_server_port("server_9999") is None
    assert stack_commands._reload_server_port("server_80a0") is None
    assert stack_commands._reload_server_port("server_") is None
    assert stack_commands._reload_server_port("bogus") is None


def test_reload_sub_full_half_restarts_with_its_own_numa_instance(
    monkeypatch,
) -> None:
    """reload server_8080 must kill the half's listener and relaunch it with the
    manifest entry's roles + numa_instance — the path that redeploys declared
    serving flags to a sub-full instance."""
    entry = next(
        server
        for server in stack_commands.HOT_SERVERS + stack_commands.WARM_SERVERS
        if server["port"] == 8080
    )
    killed: list[int] = []
    saved: list[dict] = []
    calls: list[tuple] = []
    new_info = stack.ProcessInfo(
        role=entry["roles"][0], pid=222, port=8080, started_at="after",
        model_path="new", log_file="new.log",
    )

    monkeypatch.setattr(stack, "load_state", lambda: {})
    monkeypatch.setattr(stack, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(
        stack_commands, "_refresh_runtime_facts_manifest", lambda *a, **kw: None
    )
    monkeypatch.setattr(stack, "kill_process", lambda pid: killed.append(pid))
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(stack, "_pids_on_port", lambda _port: [333])
    monkeypatch.setattr(stack, "RegistryLoader", lambda: object())
    monkeypatch.setattr(stack_commands, "RegistryLoader", lambda: object())

    def fake_start_server(port, roles, registry, *args, **kwargs):
        calls.append((port, list(roles), kwargs.get("numa_instance")))
        return new_info

    monkeypatch.setattr(stack, "start_server", fake_start_server)

    rc = stack.cmd_reload(Namespace(components=["server_8080"]))

    assert rc == 0
    assert killed == [333], "the stale half's listener must be stopped"
    assert calls == [(8080, entry["roles"], entry.get("numa_instance", 0))], (
        "relaunch must carry the manifest entry's roles and numa_instance"
    )
    assert saved[-1]["server_8080"] == new_info


def test_reload_unknown_component_fails_loudly(monkeypatch) -> None:
    """An unresolvable component must exit non-zero, never report success.

    Exiting 0 on "Unknown component" is exactly how the sub-full defect hid:
    the reload restarted nothing, the old process kept answering /health, and
    the caller read success.
    """
    monkeypatch.setattr(stack, "load_state", lambda: {})
    monkeypatch.setattr(stack, "save_state", lambda _value: None)
    monkeypatch.setattr(
        stack_commands, "_refresh_runtime_facts_manifest", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        stack, "start_server",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("unknown component must not reach start_server()")
        ),
    )

    assert stack.cmd_reload(Namespace(components=["server_9999"])) == 1


def test_stack_change_launch_gate_threads_numa_mode_as_flag_not_env(monkeypatch, capsys) -> None:
    """NIB2-69: the pipeline ignores ambient ORCHESTRATOR_STACK_NUMA_MODE, so an
    explicit start mode must reach the gate as --numa-mode or it is lost."""
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="summary: ok\n", stderr="")

    monkeypatch.setattr(stack_commands.subprocess, "run", fake_run)

    assert stack_commands._run_stack_change_launch_gate(_stack_gate_args(numa_mode="quarter"))

    assert captured["cmd"] == [
        *stack_commands.STACK_CHANGE_LAUNCH_GATE_COMMAND,
        "--numa-mode",
        "quarter",
    ]
    assert captured["env"] is None


# --- STACKCHG-DFLASH2-20261003: the VL backend follows the vision role's tier ---------

def _start_api_capturing_env(monkeypatch, tmp_path: Path) -> dict:
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 4321

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return FakeProc()

    monkeypatch.setattr(stack, "LOG_DIR", tmp_path)
    monkeypatch.setattr(stack, "_pids_on_port", lambda _port: [])
    monkeypatch.setattr(stack, "_write_orchestrator_marker", lambda **_kwargs: tmp_path / "marker")
    monkeypatch.setattr(stack, "wait_for_health", lambda *args, **kwargs: True)
    monkeypatch.setattr(stack, "_set_oom_protection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stack.subprocess, "Popen", fake_popen)
    assert stack.start_orchestrator() is not None
    return captured["env"]  # type: ignore[return-value]


@pytest.mark.parametrize("tier,expected", [("warm", "server"), ("hot", "auto")])
def test_vl_backend_follows_vision_tier(monkeypatch, tmp_path: Path, tier, expected) -> None:
    from scripts.server import stack_manifest

    monkeypatch.delenv("ORCHESTRATOR_VISION_VL_BACKEND", raising=False)
    meta = {**stack_manifest.ROLE_LAUNCH_META,
            "worker_vision": {**stack_manifest.ROLE_LAUNCH_META["worker_vision"], "tier": tier}}
    monkeypatch.setattr(stack_manifest, "ROLE_LAUNCH_META", meta)
    assert _start_api_capturing_env(monkeypatch, tmp_path)["ORCHESTRATOR_VISION_VL_BACKEND"] == expected


def test_explicit_vl_backend_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORCHESTRATOR_VISION_VL_BACKEND", "cli")
    assert _start_api_capturing_env(monkeypatch, tmp_path)["ORCHESTRATOR_VISION_VL_BACKEND"] == "cli"
