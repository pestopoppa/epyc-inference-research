from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent))
import laguna_cpu_dflash_observation_runner as runner


def valid_semantic_content(prompt_index: int, variant: int = 0) -> str:
    explanations = (
        "A reliable trial-division check tests possible divisors only through each candidate's square root. Composite values are eliminated while the surviving integers remain in ascending order, after which ordinary integer addition gives the total.",
        "The traversal sorts each object's keys before descending recursively and retains array index order. Numbers, strings, null, and booleans are emitted when encountered, so container structure disappears without changing scalar types.",
        "The positive total is ten, so division by that total preserves nonnegativity and produces values summing to one. A separate zero-total branch avoids division by zero and returns a zero vector with the original length.",
    )
    markers = (
        (
            "PRIMES: 11,13,17,19,23,29,31,37,41,43,47\nSUM: 311",
            "PRIMES: 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47\nSUM: 311",
        ),
        ("FLAT: [2,1,\"hi\",3,null,false]", "FLAT: [ 2, 1, \"hi\", 3, null, false ]"),
        ("NORMALIZED: [0,0.2,0.3,0.5]\nZERO_CASE: [0,0,0]", "NORMALIZED: [0.0,2e-1,3e-1,5e-1]\nZERO_CASE: [0.0,0e0,0]"),
    )
    return explanations[prompt_index - 1] + "\n" + markers[prompt_index - 1][variant]


def valid_summary_rows() -> list[dict[str, object]]:
    rows = []
    for cell in runner.balanced_schedule():
        prompts = [
            {
                "prompt_index": index,
                "semantic_validation": {"valid": True, "task": runner.SEMANTIC_TASKS[index - 1]},
                "content_sha256": f"{cell['lane']}-{cell['rep']}-{index}",
                "prompt_tokens": 1,
                "completion_tokens": 100,
                "prompt_ms": 1,
                "decode_ms": 2,
            }
            for index in range(1, 4)
        ]
        speculative = cell["arm"] == runner.DFLASH.name
        rows.append({
            **cell,
            "status": "ok",
            "prompt_rows": prompts,
            "prompt_tps": 1000.0,
            "decode_tps": 50000.0,
            "completion_tokens": 300,
            "warmup": {"status": "pass"},
            "draft_n": 30 if speculative else None,
            "draft_n_accepted": 15 if speculative else None,
            "cleanup": {"status": "pass"},
            "iqk_engagement": {
                "status": "pass",
                "lane": cell["lane"],
                "active_type_codes": [12] if cell["lane"] == "q4_k_m" else [],
            },
        })
    return rows


def test_fixed_cpu_recipe_and_dflash_placement() -> None:
    args = runner.parse_args([])
    assert args.context == 4096
    assert args.max_tokens == 320
    assert args.min_completion_tokens == 64
    base = runner.server_argv(runner.lanes()[0], runner.BASE, 19000)
    dflash = runner.server_argv(runner.lanes()[0], runner.DFLASH, 19001)
    assert base[:6] == ["taskset", "-c", "0-95", "numactl", "--interleave=all", str(runner.CANONICAL_BINARY)]
    for value in ("-t", "96", "-tb", "-fa", "on", "-dev", "none", "-ngl", "0", "--no-op-offload", "--no-mmap"):
        assert value in base
    for flag, value in (("--spec-draft-device", "none"), ("--spec-draft-ngl", "0"), ("--spec-type", "draft-dflash"), ("--spec-draft-n-max", "15"), ("--spec-draft-type-k", "q8_0")):
        index = dflash.index(flag)
        assert dflash[index + 1] == value
    assert runner.child_env()["LD_LIBRARY_PATH"] == f"{runner.CANONICAL_BINARY.parent}:{runner.LLVM20_LIBDIR}"
    assert runner.child_env()["GGML_IQK"] == "1"
    assert runner.child_env()["KMP_BLOCKTIME"] == "10"
    for argv in (base, dflash):
        assert argv[argv.index("--reasoning") + 1] == "off"
        assert argv[argv.index("--reasoning-budget") + 1] == "0"
        assert argv.count("--no-mmap") == 1
        assert "--mmap" not in argv
    with pytest.raises(RuntimeError, match="no conflicting"):
        runner.validate_server_argv([*base, "--mmap"])


def test_child_environment_is_exact_and_scrubs_hostile_inheritance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hostile = {
        "LD_PRELOAD": "/tmp/hostile.so",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GGML_IQK": "0",
        "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    assert runner.child_env() == runner.EXECUTION_ENV
    assert runner.child_env()["GGML_IQK"] == "1"
    assert {key for key in runner.child_env() if key.startswith("GGML_")} == {"GGML_IQK"}
    assert not (set(hostile) - {"GGML_IQK"}).intersection(runner.child_env())
    assert not set(hostile).intersection(runner.CONTROL_ENV)
    subprocess_kwargs = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*_: object, **kwargs: object) -> Completed:
        subprocess_kwargs.update(kwargs)
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.run_capture(["true"])["ok"] is True
    assert subprocess_kwargs["env"] == runner.CONTROL_ENV

    process_dir = tmp_path / "77"
    process_dir.mkdir()
    exact_raw = b"\0".join(f"{key}={value}".encode() for key, value in runner.child_env().items()) + b"\0"
    (process_dir / "environ").write_bytes(exact_raw)
    assert runner.target_process_environment(77, tmp_path) == runner.child_env()
    (process_dir / "environ").write_bytes(exact_raw + b"LD_PRELOAD=/tmp/hostile.so\0")
    with pytest.raises(RuntimeError, match="exact allowlist"):
        runner.target_process_environment(77, tmp_path)


def test_exact_process_guard_ignores_runner_argv_and_rejects_real_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    runner_argv_process = [{"pid": 22, "comm": "python3"}]
    assert runner.exact_llama_processes(runner_argv_process) == []
    monkeypatch.setattr(runner.os, "getpid", lambda: 22)
    monkeypatch.setattr(runner, "proc_exe_path", lambda _: Path("/tmp/build/bin/llama-server"))
    exact = runner.exact_llama_processes([{"pid": 23, "comm": "llama-server"}])
    assert exact == [{"pid": 23, "comm": "llama-server", "exe": "/tmp/build/bin/llama-server", "reason": "exact llama executable"}]
    clean = {"processes": {"exact_llama_processes": [], "autopilot_processes": [], "kfd_owner": False, "rocm_owner": False}}
    runner.ensure_quiet_cpu_only(clean)
    with pytest.raises(RuntimeError, match="contaminated"):
        runner.ensure_quiet_cpu_only({"processes": {**clean["processes"], "exact_llama_processes": exact}})


def test_live_runtime_identity_and_listener_are_bound_to_target_pid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid = 77
    process_dir = tmp_path / str(pid)
    (process_dir / "fd").mkdir(parents=True)
    binary = tmp_path / "candidate/llama-server"
    local_lib = tmp_path / "candidate/libllama.so"
    openmp = tmp_path / "llvm/libomp.so.5"
    binary.parent.mkdir()
    openmp.parent.mkdir()
    binary.write_bytes(b"binary")
    local_lib.write_bytes(b"llama")
    openmp.write_bytes(b"openmp")
    (process_dir / "exe").symlink_to(binary)
    (process_dir / "maps").write_text(
        f"1000-2000 r-xp 00000000 00:00 1 {local_lib}\n"
        f"2000-3000 r-xp 00000000 00:00 2 {openmp}\n"
    )
    (process_dir / "fd/9").symlink_to("socket:[12345]")
    runtime = {
        "server": runner.stable_file_identity(binary),
        "local_llama_ggml_libraries": [runner.stable_file_identity(local_lib)],
        "openmp_runtime": runner.stable_file_identity(openmp),
    }
    assert runner.target_executable_evidence(pid, runtime["server"], tmp_path)["resolved_path"] == str(binary)
    mapped = runner.target_mapped_runtime_evidence(pid, runtime, tmp_path)
    assert len(mapped["mapped_runtime_artifacts"]) == 2
    monkeypatch.setattr(
        runner,
        "tcp_listeners",
        lambda: [{"family": "tcp", "port": 19000, "inode": 12345, "raw": "row"}],
    )
    assert runner.target_listener_evidence(pid, 19000, tmp_path)["target_fd"]["fd"] == "9"

    wrong_binary = tmp_path / "wrong/llama-server"
    wrong_binary.parent.mkdir()
    wrong_binary.write_bytes(b"wrong")
    (process_dir / "exe").unlink()
    (process_dir / "exe").symlink_to(wrong_binary)
    with pytest.raises(RuntimeError, match="not the pinned candidate"):
        runner.target_executable_evidence(pid, runtime["server"], tmp_path)

    monkeypatch.setattr(
        runner,
        "tcp_listeners",
        lambda: [{"family": "tcp", "port": 19000, "inode": 99999, "raw": "row"}],
    )
    with pytest.raises(RuntimeError, match="not owned"):
        runner.target_listener_evidence(pid, 19000, tmp_path)
    monkeypatch.setattr(
        runner,
        "tcp_listeners",
        lambda: [
            {"family": "tcp", "port": 19000, "inode": 12345, "raw": "row"},
            {"family": "tcp6", "port": 19000, "inode": 12346, "raw": "row"},
        ],
    )
    with pytest.raises(RuntimeError, match="exactly one listener"):
        runner.target_listener_evidence(pid, 19000, tmp_path)

    (process_dir / "maps").write_text(f"1000-2000 r-xp 00000000 00:00 1 {local_lib}\n")
    with pytest.raises(RuntimeError, match="does not map every pinned"):
        runner.target_mapped_runtime_evidence(pid, runtime, tmp_path)

    (process_dir / "maps").write_text(
        f"1000-2000 r-xp 00000000 00:00 1 {local_lib}\n"
        f"2000-3000 r-xp 00000000 00:00 2 {openmp}\n"
    )
    for name in ("libgomp.so.1", "libomp.so.9", "libllama-alt.so", "libggml-alt.so", "libllama.so"):
        alternate = tmp_path / "alternate" / name
        alternate.parent.mkdir(exist_ok=True)
        alternate.write_bytes(b"alternate")
        (process_dir / "maps").write_text(
            f"1000-2000 r-xp 00000000 00:00 1 {local_lib}\n"
            f"2000-3000 r-xp 00000000 00:00 2 {openmp}\n"
            f"3000-4000 r-xp 00000000 00:00 3 {alternate}\n"
        )
        with pytest.raises(RuntimeError, match="unpinned OpenMP or llama/ggml runtime"):
            runner.target_mapped_runtime_evidence(pid, runtime, tmp_path)

    local_alias = tmp_path / "aliases/libllama.so"
    openmp_alias = tmp_path / "aliases/libomp.so.5"
    local_alias.parent.mkdir()
    local_alias.symlink_to(local_lib)
    openmp_alias.symlink_to(openmp)
    (process_dir / "maps").write_text(
        f"1000-2000 r-xp 00000000 00:00 1 {local_alias}\n"
        f"2000-3000 r-xp 00000000 00:00 2 {openmp_alias}\n"
    )
    symlinked = runner.target_mapped_runtime_evidence(pid, runtime, tmp_path)
    assert symlinked["relevant_runtime_maps"] == sorted((str(local_lib), str(openmp)))


def test_autopilot_detection_uses_case_insensitive_comm_and_full_cmdline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = [
        {"pid": 10, "comm": "AutoPilot"},
        {"pid": 11, "comm": "python3"},
        {"pid": 12, "comm": "python3"},
        {"pid": 13, "comm": "AUTOPILOT"},
    ]
    cmdlines = {
        10: "python3 worker.py",
        11: "python3 -m EPYC.AutoPilot --run",
        12: "python3 worker.py",
        13: None,
    }
    monkeypatch.setattr(runner, "proc_cmdline", lambda pid, _: cmdlines[pid])
    assert runner.observed_autopilot_processes(processes) == [
        {"pid": 10, "comm": "AutoPilot", "cmdline": "python3 worker.py"},
        {"pid": 11, "comm": "python3", "cmdline": "python3 -m EPYC.AutoPilot --run"},
        {"pid": 13, "comm": "AUTOPILOT", "cmdline": None},
    ]


def test_live_cpu_evidence_uses_target_proc_not_invalid_numactl_pid_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    pid = 77
    processes = {
        "exact_llama_processes": [{"pid": pid}],
        "autopilot_processes": [],
        "kfd_owner": False,
        "rocm_owner": False,
    }
    hardware = {"ok": True, "returncode": 0, "stdout": "available: 4 nodes (0-3)\n", "stderr": ""}
    monkeypatch.setattr(runner, "system_snapshot", lambda: {"processes": processes, "numactl_hardware": hardware})
    monkeypatch.setattr(
        runner,
        "target_process_policy",
        lambda target_pid, nodes: {"cpus_allowed_list": "0-95", "interleave_policy": "interleave:0-3", "pid": target_pid, "nodes": nodes},
    )
    monkeypatch.setattr(runner, "target_process_environment", lambda target_pid: {"pid": str(target_pid)})
    monkeypatch.setattr(runner, "target_executable_evidence", lambda target_pid, _: {"pid": target_pid})
    monkeypatch.setattr(runner, "target_mapped_runtime_evidence", lambda target_pid, _: {"pid": target_pid})
    monkeypatch.setattr(runner, "target_listener_evidence", lambda target_pid, port: {"pid": target_pid, "port": port})
    monkeypatch.setattr(
        runner,
        "thread_affinity_evidence",
        lambda target_pid: {"status": "pass", "pid": target_pid},
    )
    monkeypatch.setattr(runner, "parse_numastat_residency", lambda _, nodes: {"nodes": nodes, "total_mib": 10.0})
    calls = []

    def capture(argv: list[str], **_: object) -> dict[str, object]:
        calls.append(argv)
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "numastat",
            "stderr": "",
        }

    monkeypatch.setattr(runner, "run_capture", capture)
    evidence = runner.live_cpu_evidence(pid, 19000, {"server": {}})
    assert evidence["target_process_policy"]["pid"] == pid
    assert evidence["thread_affinity"] == {"status": "pass", "pid": pid}
    assert calls == [["numastat", "-p", str(pid)]]
    assert all(command[0] != "numactl" for command in calls)


def test_target_process_policy_requires_affinity_subset_and_consistent_interleave(tmp_path: Path) -> None:
    process_dir = tmp_path / "77"
    process_dir.mkdir()
    status = process_dir / "status"
    numa_maps = process_dir / "numa_maps"
    status.write_text("Name:\tllama-server\nCpus_allowed_list:\t0-95\n")
    numa_maps.write_text(
        "00400000 interleave:0-3 file=/tmp/llama-server N0=1\n"
        "7fff0000 interleave:0-3 stack anon=1 N1=1\n"
    )
    policy = runner.target_process_policy(77, [0, 1, 2, 3], tmp_path)
    assert policy["cpus_allowed_list"] == "0-95"
    assert policy["interleave_policy"] == "interleave:0-3"
    assert policy["numa_map_rows"] == 2

    status.write_text("Name:\tllama-server\nCpus_allowed_list:\t0\n")
    policy = runner.target_process_policy(77, [0, 1, 2, 3], tmp_path)
    assert policy["cpus_allowed_list"] == "0"

    status.write_text("Name:\tllama-server\nCpus_allowed_list:\t0-95,100\n")
    with pytest.raises(RuntimeError, match="escapes required"):
        runner.target_process_policy(77, [0, 1, 2, 3], tmp_path)

    status.write_text("Name:\tllama-server\nCpus_allowed_list:\t0-95\n")
    numa_maps.write_text("00400000 interleave:0-3 file=x\n7fff0000 default stack\n")
    with pytest.raises(RuntimeError, match="consistent interleave"):
        runner.target_process_policy(77, [0, 1, 2, 3], tmp_path)
    numa_maps.write_text("00400000 interleave:0-1 file=x\n7fff0000 interleave:0-1 stack\n")
    with pytest.raises(RuntimeError, match="all available nodes"):
        runner.target_process_policy(77, [0, 1, 2, 3], tmp_path)


def write_thread_status(
    proc_root: Path,
    pid: int,
    tid: int,
    cpus_allowed_list: str,
) -> None:
    status = proc_root / str(pid) / "task" / str(tid) / "status"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        f"Name:\tllama-server\nCpus_allowed_list:\t{cpus_allowed_list}\n"
    )


def test_thread_affinity_accepts_openmp_team_with_leader_on_cpu_zero(
    tmp_path: Path,
) -> None:
    write_thread_status(tmp_path, 77, 77, "0")
    write_thread_status(tmp_path, 77, 78, "1-95")
    evidence = runner.thread_affinity_evidence(77, proc_root=tmp_path)
    assert evidence["union_cpus"] == list(range(96))
    assert evidence["threads"] == [
        {"tid": 77, "cpus_allowed_list": "0", "cpus": [0]},
        {"tid": 78, "cpus_allowed_list": "1-95", "cpus": list(range(1, 96))},
    ]


@pytest.mark.parametrize(
    ("cpus_allowed_list", "error"),
    [
        ("0,96", "escapes required CPU set"),
        ("0-94", "union does not exactly cover"),
        ("0-foo", "malformed Cpus_allowed_list"),
    ],
)
def test_thread_affinity_rejects_outside_incomplete_or_malformed_masks(
    tmp_path: Path,
    cpus_allowed_list: str,
    error: str,
) -> None:
    write_thread_status(tmp_path, 77, 77, cpus_allowed_list)
    with pytest.raises(RuntimeError, match=error):
        runner.thread_affinity_evidence(77, proc_root=tmp_path)


def test_thread_affinity_fails_closed_on_unreadable_thread_status(
    tmp_path: Path,
) -> None:
    status = tmp_path / "77/task/77/status"
    status.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="cannot read server thread status"):
        runner.thread_affinity_evidence(77, proc_root=tmp_path)


def test_thread_affinity_retries_bounded_thread_list_churn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_thread_status(tmp_path, 77, 77, "0")
    write_thread_status(tmp_path, 77, 78, "1-95")
    task_dir = tmp_path / "77/task"
    original_iterdir = Path.iterdir
    calls = 0

    def churn_once(path: Path):
        nonlocal calls
        if path == task_dir:
            calls += 1
            if calls == 2:
                return iter([task_dir / "77"])
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", churn_once)
    evidence = runner.thread_affinity_evidence(
        77,
        proc_root=tmp_path,
        max_attempts=2,
    )
    assert evidence["attempt"] == 2


def test_thread_affinity_fails_after_bounded_thread_list_churn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_thread_status(tmp_path, 77, 77, "0")
    write_thread_status(tmp_path, 77, 78, "1-95")
    task_dir = tmp_path / "77/task"
    original_iterdir = Path.iterdir
    calls = 0

    def churn_forever(path: Path):
        nonlocal calls
        if path == task_dir:
            calls += 1
            if calls % 2 == 0:
                return iter([task_dir / "77"])
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", churn_forever)
    with pytest.raises(RuntimeError, match="did not stabilize"):
        runner.thread_affinity_evidence(
            77,
            proc_root=tmp_path,
            max_attempts=2,
        )


def test_numactl_and_numastat_require_exact_all_node_residency() -> None:
    hardware = {"ok": True, "returncode": 0, "stdout": "available: 4 nodes (0-3)\n", "stderr": ""}
    assert runner.numactl_available_nodes(hardware) == [0, 1, 2, 3]
    residency_text = """
                           Node 0          Node 1          Node 2          Node 3           Total
                  --------------- --------------- --------------- --------------- ---------------
Total                       10.00           20.00           30.00           40.00          100.00
"""
    capture = {"ok": True, "returncode": 0, "stdout": residency_text, "stderr": ""}
    residency = runner.parse_numastat_residency(capture, [0, 1, 2, 3])
    assert residency["node_mib"] == {"0": 10.0, "1": 20.0, "2": 30.0, "3": 40.0}
    with pytest.raises(RuntimeError, match="all available nodes"):
        runner.parse_numastat_residency(capture, [0, 1])
    zero_node = dict(capture)
    zero_node["stdout"] = residency_text.replace("10.00", "0.00", 1).replace("100.00", "90.00")
    with pytest.raises(RuntimeError, match="every interleave node"):
        runner.parse_numastat_residency(zero_node, [0, 1, 2, 3])


def test_host_tuning_requires_all_governors_thp_numa_and_meminfo(tmp_path: Path) -> None:
    cpu_root = tmp_path / "cpu"
    thp_root = tmp_path / "thp"
    numa_path = tmp_path / "numa_balancing"
    meminfo_path = tmp_path / "meminfo"
    (cpu_root / "cpu0/cpufreq").mkdir(parents=True)
    (cpu_root / "cpu1/cpufreq").mkdir(parents=True)
    thp_root.mkdir()
    (cpu_root / "online").write_text("0-1\n")
    governor_paths = [
        cpu_root / "cpu0/cpufreq/scaling_governor",
        cpu_root / "cpu1/cpufreq/scaling_governor",
    ]
    for path in governor_paths:
        path.write_text("performance\n")
    (thp_root / "enabled").write_text("[always] madvise never\n")
    (thp_root / "defrag").write_text("[always] defer madvise never\n")
    numa_path.write_text("0\n")
    meminfo_rows = []
    for key in runner.THP_MEMINFO_KEYS:
        unit = "" if key.startswith("HugePages_") else " kB"
        meminfo_rows.append(f"{key}: 0{unit}")
    meminfo_path.write_text("\n".join(meminfo_rows) + "\n")

    snapshot = runner.host_tuning_snapshot(cpu_root, thp_root, numa_path, meminfo_path)
    assert snapshot["online_cpus"] == [0, 1]
    assert set(snapshot["scaling_governors"].values()) == {"performance"}
    assert snapshot["transparent_hugepage"]["enabled"]["active"] == "always"
    assert set(snapshot["thp_meminfo"]) == set(runner.THP_MEMINFO_KEYS)

    governor_paths[1].write_text("powersave\n")
    with pytest.raises(RuntimeError, match="every online CPU"):
        runner.host_tuning_snapshot(cpu_root, thp_root, numa_path, meminfo_path)
    governor_paths[1].write_text("performance\n")
    governor_paths[1].unlink()
    with pytest.raises(RuntimeError, match="unreadable"):
        runner.host_tuning_snapshot(cpu_root, thp_root, numa_path, meminfo_path)
    governor_paths[1].write_text("performance\n")

    (thp_root / "enabled").write_text("always madvise never\n")
    with pytest.raises(RuntimeError, match="malformed"):
        runner.host_tuning_snapshot(cpu_root, thp_root, numa_path, meminfo_path)
    (thp_root / "enabled").write_text("[madvise] always never\n")
    with pytest.raises(RuntimeError, match="both be always"):
        runner.host_tuning_snapshot(cpu_root, thp_root, numa_path, meminfo_path)
    (thp_root / "enabled").write_text("[always] madvise never\n")

    numa_path.write_text("1\n")
    with pytest.raises(RuntimeError, match="must be 0"):
        runner.host_tuning_snapshot(cpu_root, thp_root, numa_path, meminfo_path)
    numa_path.write_text("0\n")
    meminfo_path.write_text("\n".join(meminfo_rows[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="missing required"):
        runner.host_tuning_snapshot(cpu_root, thp_root, numa_path, meminfo_path)


def test_proc_tcp_listener_parser_and_exact_port_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "tcp_listeners", lambda: [{"family": "tcp", "port": 19000, "raw": "x"}])
    assert runner.port_closed(19000) is False
    monkeypatch.setattr(runner, "tcp_listeners", lambda: [])
    assert runner.port_closed(19000) is True


def test_request_monitor_records_cpu_and_rejects_swap_process_kfd_and_pid_reuse() -> None:
    def sample(
        *,
        at: float,
        busy: int,
        target_ticks: int,
        starttime: int = 500,
        hz: int = 100,
        swap_in: int = 0,
        swap_out: int = 0,
        forbidden: list[dict[str, object]] | None = None,
        kfd_users: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "captured_at": "test",
            "monotonic_s": at,
            "clock_ticks_per_second": hz,
            "aggregate_cpu_busy_ticks": busy,
            "target": {"pid": 77, "starttime": starttime, "cpu_ticks": target_ticks},
            "swap_io_pages": {"pswpin": swap_in, "pswpout": swap_out},
            "forbidden_processes": forbidden or [],
            "kfd_fd_snapshot": {"users": kfd_users or []},
        }

    before = sample(at=1.0, busy=1000, target_ticks=100)
    clean_after = sample(at=2.0, busy=1100, target_ticks=195)
    evidence = runner.validate_request_monitor_samples([before, clean_after])
    assert evidence["status"] == "pass"
    assert evidence["external_cpu_use"] == "non_gating_telemetry_only"
    assert "external_cpu_ceiling_cores" not in evidence
    assert "external_cpu_cores" not in evidence["intervals"][0]
    assert evidence["intervals"][0]["signed_external_cpu_cores_observation"] == pytest.approx(0.05)

    contaminated = sample(at=2.0, busy=1600, target_ticks=100)
    positive = runner.validate_request_monitor_samples([before, contaminated])
    assert positive["intervals"][0]["signed_external_cpu_cores_observation"] == pytest.approx(6.0)
    counter_skew = runner.validate_request_monitor_samples([
        before,
        sample(at=2.0, busy=1100, target_ticks=250),
    ])
    assert counter_skew["intervals"][0]["signed_external_cpu_cores_observation"] == pytest.approx(-0.5)
    with pytest.raises(RuntimeError, match="swap IO"):
        runner.validate_request_monitor_samples([before, sample(at=2.0, busy=1100, target_ticks=195, swap_in=1)])
    with pytest.raises(RuntimeError, match="competing inference"):
        runner.validate_request_monitor_samples([
            before,
            sample(at=2.0, busy=1100, target_ticks=195, forbidden=[{"pid": 88, "comm": "llama-server"}]),
        ])
    with pytest.raises(RuntimeError, match="KFD"):
        runner.validate_request_monitor_samples([
            before,
            sample(at=2.0, busy=1100, target_ticks=195, kfd_users=[{"pid": 99}]),
        ])
    with pytest.raises(RuntimeError, match="identity changed"):
        runner.validate_request_monitor_samples([
            before,
            sample(at=2.0, busy=1100, target_ticks=195, starttime=501),
        ])
    with pytest.raises(RuntimeError, match="counters regressed"):
        runner.validate_request_monitor_samples([
            before,
            sample(at=2.0, busy=999, target_ticks=195),
        ])
    with pytest.raises(RuntimeError, match="counters regressed"):
        runner.validate_request_monitor_samples([
            before,
            sample(at=2.0, busy=1100, target_ticks=99),
        ])
    with pytest.raises(RuntimeError, match="nonpositive or nonfinite"):
        runner.validate_request_monitor_samples([
            before,
            sample(at=1.0, busy=1100, target_ticks=195),
        ])
    with pytest.raises(RuntimeError, match="clock tick rate changed"):
        runner.validate_request_monitor_samples([
            before,
            sample(at=2.0, busy=1100, target_ticks=195, hz=250),
        ])
    with pytest.raises(RuntimeError, match="no interval"):
        runner.validate_request_monitor_samples([
            before,
            sample(at=1.1, busy=1010, target_ticks=109),
        ])


def test_monitored_query_records_mixed_counter_delta_without_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter((
        {
            "captured_at": "start",
            "monotonic_s": 1.0,
            "clock_ticks_per_second": 100,
            "aggregate_cpu_busy_ticks": 1000,
            "target": {"pid": 77, "starttime": 500, "cpu_ticks": 100},
            "swap_io_pages": {"pswpin": 0, "pswpout": 0},
            "forbidden_processes": [],
            "kfd_fd_snapshot": {"users": []},
        },
        {
            "captured_at": "end",
            "monotonic_s": 2.0,
            "clock_ticks_per_second": 100,
            "aggregate_cpu_busy_ticks": 1600,
            "target": {"pid": 77, "starttime": 500, "cpu_ticks": 100},
            "swap_io_pages": {"pswpin": 0, "pswpout": 0},
            "forbidden_processes": [],
            "kfd_fd_snapshot": {"users": []},
        },
    ))
    monkeypatch.setattr(runner, "request_interference_snapshot", lambda _: next(snapshots))
    monkeypatch.setattr(runner, "query_chat", lambda *_: {"response": "ok"})
    response, evidence, error = runner.monitored_query(19000, {"request": "body"}, 77)
    assert response == {"response": "ok"}
    assert evidence["status"] == "pass"
    assert error is None
    assert evidence["intervals"][0]["signed_external_cpu_cores_observation"] == pytest.approx(6.0)


def test_kfd_fd_users_is_exact_and_excludes_runner(tmp_path: Path) -> None:
    clean = tmp_path / "100"
    (clean / "fd").mkdir(parents=True)
    (clean / "fd/1").symlink_to("/dev/null")
    assert runner.kfd_fd_users(tmp_path, current_pid=999)["users"] == []
    owner = tmp_path / "101"
    (owner / "fd").mkdir(parents=True)
    (owner / "comm").write_text("real-gpu-owner\n")
    (owner / "exe").symlink_to("/usr/bin/env")
    (owner / "fd/7").symlink_to("/dev/kfd")
    assert runner.kfd_fd_users(tmp_path, current_pid=999)["users"] == [{"pid": 101, "comm": "real-gpu-owner", "exe": str(Path("/usr/bin/env").resolve()), "fd": "7"}]
    assert runner.kfd_fd_users(tmp_path, current_pid=101)["users"] == []


def test_recipe_drift_and_noncanonical_q8_are_rejected() -> None:
    for argv in (("--reps", "4"), ("--context", "2048"), ("--max-tokens", "128"), ("--q8-model", "/tmp/x.gguf"), ("--q8-model", "/tmp/x.part")):
        with pytest.raises(SystemExit):
            runner.parse_args(list(argv))


def test_plan_is_complete_observation_only_matrix() -> None:
    plan = runner.build_plan()
    assert plan["schema"] == "epyc.laguna_cpu_dflash_observation.plan.v5"
    assert len(plan["cells"]) == 20
    assert all(cell["prompt_count"] == 3 and cell["seed"] == 424242 for cell in plan["cells"])
    assert [(cell["rep"], cell["lane"], cell["arm"]) for cell in plan["cells"][:4]] == [
        (1, "q4_k_m", "base"),
        (1, "q4_k_m", "dflash"),
        (1, "q8_0", "dflash"),
        (1, "q8_0", "base"),
    ]
    assert [cell["schedule_position"] for cell in plan["cells"]] == list(range(1, 21))
    assert plan["recipe"]["semantic_tasks"] == list(runner.SEMANTIC_TASKS)
    assert plan["recipe"]["ggml_iqk"] == "1"
    assert plan["recipe"]["mmap"] is False
    assert plan["recipe"]["warmup_policy"] == runner.WARMUP_POLICY
    assert plan["recipe"]["warmup_policy"]["measured_prompt_order"] == [1, 2, 3]
    assert plan["recipe"]["host_requirements"]["thp_defrag_active"] == "always"
    assert all("exactly one" in prompt.lower() for prompt in runner.PROMPTS)
    assert not any("words" in prompt.lower() for prompt in runner.PROMPTS)
    for prompt, prefix in zip(runner.PROMPTS, ("PRIMES:", "FLAT:", "NORMALIZED:"), strict=True):
        assert prompt.index(prefix) > prompt.lower().index("explain")
    assert plan["recipe"]["prompt_protocol"] == runner.PROMPT_PROTOCOL
    assert plan["recipe"]["prompt_protocol"]["result_lines_terminal"] is True
    assert plan["observation_policy"]["decision_grade"] is False
    assert plan["observation_policy"]["promotion_gate"] is False
    assert plan["observation_policy"]["march_no_go_reopened"] is False
    assert plan["observation_policy"]["external_cpu_accounting"] == (
        "record_only_signed_delta_from_mixed_proc_counter_sources"
    )
    assert plan["observation_policy"]["external_cpu_use"] == "non_gating_telemetry_only"
    assert "request_external_cpu_ceiling_cores" not in plan["recipe"]["host_requirements"]


def test_schedule_counterbalances_lanes_and_arm_first_positions() -> None:
    cells = runner.balanced_schedule()
    lane_first = [next(cell for cell in cells if cell["rep"] == rep)["lane"] for rep in range(1, runner.REPS + 1)]
    assert lane_first == ["q4_k_m", "q8_0", "q4_k_m", "q8_0", "q4_k_m"]
    assert sum(cell["pair_position"] == 1 and cell["arm"] == "base" for cell in cells) == 5
    assert sum(cell["pair_position"] == 1 and cell["arm"] == "dflash" for cell in cells) == 5
    for lane in ("q4_k_m", "q8_0"):
        first_arms = [
            next(
                cell
                for cell in cells
                if cell["lane"] == lane and cell["rep"] == rep and cell["pair_position"] == 1
            )["arm"]
            for rep in range(1, runner.REPS + 1)
        ]
        assert abs(first_arms.count("base") - first_arms.count("dflash")) == 1

    broken = json.loads(json.dumps(cells))
    broken[4]["lane"], broken[6]["lane"] = broken[6]["lane"], broken[4]["lane"]
    with pytest.raises(RuntimeError, match="lane order|each lane"):
        runner.validate_schedule_contract(broken)


def test_validate_q8_rejects_partial_and_wrong_size(tmp_path: Path) -> None:
    part = tmp_path / "model.part"
    part.write_bytes(b"x")
    with pytest.raises(ValueError, match=".part"):
        runner.validate_q8(part)
    wrong = tmp_path / "model.gguf"
    wrong.write_bytes(b"x")
    with pytest.raises(ValueError, match="bytes"):
        runner.validate_q8(wrong)


def test_validate_models_hashes_each_artifact_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models = {
        "q4": tmp_path / "laguna-Q4_K_M.gguf",
        "q8": tmp_path / "laguna-Q8_0.gguf",
        "drafter": tmp_path / "laguna-DFlash-BF16.gguf",
    }
    contents = {"q4": b"q4", "q8": b"q8-eight", "drafter": b"draft"}
    expected_hashes = {"q4": "q4-sha", "q8": "q8-sha", "drafter": "drafter-sha"}
    for name, path in models.items():
        path.write_bytes(contents[name])
    monkeypatch.setattr(runner, "Q4_MODEL", models["q4"])
    monkeypatch.setattr(runner, "Q8_MODEL", models["q8"])
    monkeypatch.setattr(runner, "DRAFTER_MODEL", models["drafter"])
    monkeypatch.setattr(runner, "Q4_BYTES", len(contents["q4"]))
    monkeypatch.setattr(runner, "Q8_BYTES", len(contents["q8"]))
    monkeypatch.setattr(runner, "DRAFTER_BYTES", len(contents["drafter"]))
    monkeypatch.setattr(runner, "Q4_SHA256", expected_hashes["q4"])
    monkeypatch.setattr(runner, "Q8_SHA256", expected_hashes["q8"])
    monkeypatch.setattr(runner, "DRAFTER_SHA256", expected_hashes["drafter"])
    calls: list[Path] = []

    def fake_sha256(path: Path) -> str:
        calls.append(path)
        return expected_hashes[next(name for name, model_path in models.items() if model_path == path)]

    monkeypatch.setattr(runner, "sha256_file", fake_sha256)
    identities = runner.validate_models()
    assert {name: identity["sha256"] for name, identity in identities.items()} == expected_hashes
    assert calls == [models["q4"], models["q8"], models["drafter"]]


def test_postflight_identity_and_strict_json_fail_closed(tmp_path: Path) -> None:
    preflight = {
        "artifacts": {
            "server": {"path": "/candidate/llama-server", "sha256": "server"},
            "local_llama_ggml_libraries": [{"path": "/candidate/libllama.so", "sha256": "lib"}],
            "models": {"q8": {"path": "/dev/shm/q8.gguf", "sha256": "q8"}},
            "runner": {"path": "/runner.py", "sha256": "runner"},
        }
    }
    runner.require_matching_postflight(preflight, json.loads(json.dumps(preflight)))
    postflight = json.loads(json.dumps(preflight))
    postflight["artifacts"]["models"]["q8"]["sha256"] = "changed"
    with pytest.raises(RuntimeError, match="identity changed"):
        runner.require_matching_postflight(preflight, postflight)
    with pytest.raises(ValueError, match="Out of range"):
        runner.write_json(tmp_path / "nan.json", {"metric": float("nan")})


def test_provenance_rejects_dirty_version_and_ldd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / "build-v8-cpu/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    (binary.parent / "libllama.so").write_bytes(b"llama")
    (binary.parent / "libggml.so").write_bytes(b"ggml")
    openmp = tmp_path / "llvm/libomp.so.5"
    openmp.parent.mkdir()
    openmp.write_bytes(b"openmp")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(runner, "CANONICAL_BINARY", binary)
    monkeypatch.setattr(runner, "CANONICAL_SOURCE", source)
    monkeypatch.setattr(runner, "EXPECTED_BRANCH", "test-branch")
    monkeypatch.setattr(runner, "EXPECTED_HEAD", "abcdef1234567890")
    monkeypatch.setattr(runner, "EXPECTED_SERVER_SHA256", "hash")
    monkeypatch.setattr(runner, "EXPECTED_LOCAL_LIBRARY_SHA256", {"libllama.so": "hash", "libggml.so": "hash"})
    monkeypatch.setattr(runner, "OPENMP_RUNTIME", openmp)
    monkeypatch.setattr(runner, "EXPECTED_OPENMP_RUNTIME_SHA256", "hash")
    monkeypatch.setattr(runner, "sha256_file", lambda _: "hash")

    def clean_capture(argv: list[str], **_: object) -> dict[str, object]:
        text = " ".join(argv)
        if argv[0] == "git" and "rev-parse" in text:
            return {"ok": True, "stdout": "abcdef1234567890\n", "stderr": "", "returncode": 0}
        if argv[0] == "git" and "branch" in text:
            return {"ok": True, "stdout": "test-branch\n", "stderr": "", "returncode": 0}
        if argv[0] == "git":
            return {"ok": True, "stdout": "", "stderr": "", "returncode": 0}
        if argv[0] == "ldd":
            return {"ok": True, "stdout": f"libllama.so => {binary.parent}/libllama.so (0x0)\nlibggml.so => {binary.parent}/libggml.so (0x0)\nlibgomp.so.1 => {openmp} (0x0)\n", "stderr": "", "returncode": 0}
        return {"ok": True, "stdout": "build abcdef1\n", "stderr": "", "returncode": 0}

    monkeypatch.setattr(runner, "run_capture", clean_capture)
    provenance = runner.validate_candidate_provenance()
    assert provenance["source_clean"] is True
    assert {Path(identity["path"]).name for identity in provenance["local_library_identities"]} == {
        "libllama.so",
        "libggml.so",
    }
    assert provenance["openmp_runtime_identity"]["path"] == str(openmp)

    def dirty_capture(argv: list[str], **kwargs: object) -> dict[str, object]:
        result = clean_capture(argv, **kwargs)
        if argv[0] == "git" and "status" in argv:
            result["stdout"] = " M src/file.cpp\n"
        return result

    monkeypatch.setattr(runner, "run_capture", dirty_capture)
    with pytest.raises(RuntimeError, match="tracked/index"):
        runner.validate_candidate_provenance()

    def bad_version(argv: list[str], **kwargs: object) -> dict[str, object]:
        result = clean_capture(argv, **kwargs)
        if argv[0] == str(binary):
            result["stdout"] = "other build\n"
        return result

    monkeypatch.setattr(runner, "run_capture", bad_version)
    with pytest.raises(RuntimeError, match="version"):
        runner.validate_candidate_provenance()

    def bad_ldd(argv: list[str], **kwargs: object) -> dict[str, object]:
        result = clean_capture(argv, **kwargs)
        if argv[0] == "ldd":
            result["stdout"] = "libllama.so => /usr/lib/libllama.so (0x0)\n"
        return result

    monkeypatch.setattr(runner, "run_capture", bad_ldd)
    with pytest.raises(RuntimeError, match="ldd"):
        runner.validate_candidate_provenance()

    wrong_openmp = tmp_path / "wrong/libgomp.so.1"
    wrong_openmp.parent.mkdir()
    wrong_openmp.write_bytes(b"wrong")

    def wrong_openmp_capture(argv: list[str], **kwargs: object) -> dict[str, object]:
        result = clean_capture(argv, **kwargs)
        if argv[0] == "ldd":
            result["stdout"] = (
                f"libllama.so => {binary.parent}/libllama.so (0x0)\n"
                f"libggml.so => {binary.parent}/libggml.so (0x0)\n"
                f"libgomp.so.1 => {wrong_openmp} (0x0)\n"
            )
        return result

    monkeypatch.setattr(runner, "run_capture", wrong_openmp_capture)
    with pytest.raises(RuntimeError, match="canonical LLVM20"):
        runner.validate_candidate_provenance()


def test_dflash_requires_explicit_nonzero_counters() -> None:
    content = valid_semantic_content(1)
    response = {"usage": {"prompt_tokens": 4, "completion_tokens": 100}, "timings": {"prompt_n": 4, "predicted_n": 100, "prompt_ms": 4, "predicted_ms": 10}, "choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    with pytest.raises(RuntimeError, match="draft_n"):
        runner.response_row(response, runner.DFLASH, 1)
    response["timings"].update({"draft_n": 0, "draft_n_accepted": 0})
    with pytest.raises(RuntimeError, match="invalid"):
        runner.response_row(response, runner.DFLASH, 1)
    response["timings"].update({"draft_n": 10, "draft_n_accepted": 6})
    row = runner.response_row(response, runner.DFLASH, 1)
    assert row["draft_n_accepted"] == 6
    assert "draft_n" not in runner.response_row(response, runner.BASE, 1)
    response["timings"]["draft_n_accepted"] = 11
    with pytest.raises(RuntimeError, match="exceeds"):
        runner.response_row(response, runner.DFLASH, 1)
    response["timings"]["draft_n_accepted"] = True
    with pytest.raises(RuntimeError, match="non-integral"):
        runner.response_row(response, runner.DFLASH, 1)


@pytest.mark.parametrize("value", [True, "1.25", "not-a-number", float("nan"), float("inf"), float("-inf")])
def test_metric_ms_rejects_bool_nonnumeric_and_nonfinite(value: object) -> None:
    with pytest.raises(RuntimeError, match="nonnumeric|nonfinite"):
        runner.metric_ms({"prompt_ms": value}, "prompt_ms")


def test_response_rejects_reasoning_content_under_reasoning_off() -> None:
    content = valid_semantic_content(1)
    response = {
        "timings": {"prompt_n": 4, "predicted_n": 100, "prompt_ms": 4, "predicted_ms": 10},
        "choices": [{"finish_reason": "stop", "message": {"content": content, "reasoning_content": "hidden reasoning"}}],
    }
    with pytest.raises(RuntimeError, match="reasoning_content"):
        runner.response_row(response, runner.BASE, 1)
    response["choices"][0]["message"]["reasoning_content"] = "   "
    assert runner.response_row(response, runner.BASE, 1)["content"] == content


def test_response_validity_rejects_garbage_without_byte_equality_requirement() -> None:
    assert runner.anti_garbage_validity("a" * 100)["valid"] is False
    assert runner.anti_garbage_validity("This is a coherent response with enough varied alphabetic characters to pass the local anti-garbage check.")["valid"] is True


@pytest.mark.parametrize(
    ("prompt_index", "wrong_result"),
    [
        (1, "PRIMES: 11,13,17,19,23,29,31,37,41,43,47\nSUM: 312"),
        (2, 'FLAT: [2,1,"hi",3,false,null]'),
        (3, "NORMALIZED: [0,0.2,0.2,0.6]\nZERO_CASE: [0,0,0]"),
    ],
)
def test_semantic_validators_reject_coherent_but_wrong_outputs(prompt_index: int, wrong_result: str) -> None:
    prose = (
        "This answer gives a coherent, fluent explanation with varied language and enough detail to pass "
        "the anti-garbage heuristic, but its machine-checkable result is deliberately incorrect."
    )
    content = prose + "\n" + wrong_result
    assert runner.anti_garbage_validity(content)["valid"] is True
    with pytest.raises(RuntimeError, match="semantic validation"):
        runner.validate_prompt_semantics(content, prompt_index)


@pytest.mark.parametrize("prompt_index", [1, 2, 3])
def test_semantic_validators_accept_result_last_wording_and_formatting(prompt_index: int) -> None:
    first = valid_semantic_content(prompt_index, 0)
    second = valid_semantic_content(prompt_index, 1)
    assert first != second
    first_result = runner.validate_prompt_semantics(first, prompt_index)
    second_result = runner.validate_prompt_semantics(second, prompt_index)
    assert first_result["valid"] is True
    assert second_result["valid"] is True
    assert first_result["task"] == second_result["task"] == runner.SEMANTIC_TASKS[prompt_index - 1]
    assert first_result["terminal_footer"]["valid"] is True


@pytest.mark.parametrize(
    ("prompt_index", "label"),
    [(1, "PRIMES"), (2, "FLAT"), (3, "NORMALIZED")],
)
def test_semantic_validators_reject_duplicate_or_missing_result_lines(
    prompt_index: int,
    label: str,
) -> None:
    content = valid_semantic_content(prompt_index)
    result_line = next(line for line in content.splitlines() if line.startswith(f"{label}:"))
    with pytest.raises(RuntimeError, match=f"exactly one nonempty {label}:"):
        runner.validate_prompt_semantics(f"{content}\n{result_line}", prompt_index)
    missing = "\n".join(line for line in content.splitlines() if line != result_line)
    with pytest.raises(RuntimeError, match=f"exactly one nonempty {label}:"):
        runner.validate_prompt_semantics(missing, prompt_index)


@pytest.mark.parametrize("prompt_index", [1, 2, 3])
def test_semantic_validators_reject_result_first_or_trailing_prose(prompt_index: int) -> None:
    content = valid_semantic_content(prompt_index)
    lines = content.splitlines()
    footer_count = 2 if prompt_index in {1, 3} else 1
    footer = lines[-footer_count:]
    prose = lines[:-footer_count]
    with pytest.raises(RuntimeError, match="terminal result footer"):
        runner.validate_prompt_semantics("\n".join(footer + prose), prompt_index)
    with pytest.raises(RuntimeError, match="terminal result footer"):
        runner.validate_prompt_semantics(content + "\nThis trailing prose invalidates the footer.", prompt_index)


def test_prime_semantics_enforces_bounded_non_enumerated_rationale() -> None:
    footer = "PRIMES: 11,13,17,19,23,29,31,37,41,43,47\nSUM: 311"
    three_sentences = (
        "Trial division checks possible factors. Confirmed primes are retained. Their values are summed.\n"
        f"{footer}"
    )
    with pytest.raises(RuntimeError, match="one or two prime rationale sentences"):
        runner.validate_prompt_semantics(three_sentences, 1)
    enumerated = f"A concise method was used.\n- 10 is composite.\n{footer}"
    with pytest.raises(RuntimeError, match="forbids bullet or numbered"):
        runner.validate_prompt_semantics(enumerated, 1)
    too_long = " ".join(["method"] * 81) + ".\n" + footer
    with pytest.raises(RuntimeError, match="80-word ceiling"):
        runner.validate_prompt_semantics(too_long, 1)


def test_response_rejects_length_finish_even_when_semantics_pass() -> None:
    response = {
        "timings": {"prompt_n": 4, "predicted_n": 100, "prompt_ms": 4, "predicted_ms": 10},
        "choices": [{"finish_reason": "length", "message": {"content": valid_semantic_content(1)}}],
    }
    with pytest.raises(RuntimeError, match="finish_reason='length'"):
        runner.response_row(response, runner.BASE, 1)


def test_iqk_engagement_requires_q4_k_type_and_no_q8_false_claim(tmp_path: Path) -> None:
    log = tmp_path / "server.stderr"
    q4, q8 = runner.lanes()
    log.write_text(
        "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=12 activation=8 ne00=2048)\n"
        "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=14 activation=8 ne00=2048)\n"
    )
    evidence = runner.iqk_engagement_evidence(log, q4)
    assert evidence["active_type_codes"] == [12, 14]
    assert evidence["raw_log_identity"]["sha256"]

    log.write_text("[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=16 activation=8 ne00=2048)\n")
    with pytest.raises(RuntimeError, match="type=12"):
        runner.iqk_engagement_evidence(log, q4)
    log.write_text(
        "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=12 activation=8 ne00=2048)\n"
        "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=16 activation=8 ne00=2048)\n"
    )
    with pytest.raises(RuntimeError, match="native IQ"):
        runner.iqk_engagement_evidence(log, q4)

    log.write_text("")
    assert runner.iqk_engagement_evidence(log, q8)["active_type_codes"] == []
    log.write_text("[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=12 activation=8 ne00=2048)\n")
    with pytest.raises(RuntimeError, match="falsely reported"):
        runner.iqk_engagement_evidence(log, q8)
    log.write_text("[iqk] ACTIVE: malformed evidence\n")
    with pytest.raises(RuntimeError, match="malformed"):
        runner.iqk_engagement_evidence(log, q4)


def test_pinned_artifact_constants_are_exact() -> None:
    assert runner.Q4_BYTES == 75173103200
    assert runner.Q4_SHA256 == "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
    assert runner.DRAFTER_BYTES == 2233764000
    assert runner.DRAFTER_SHA256 == "24614292a4477f3ae5203c3875edcde0bc219f02616a9c9f65791e29b18a67ee"
    assert runner.EXPECTED_HEAD == "1977a5d78a5a9c0b1e0050105f8741b7d0a00284"
    assert runner.OPENMP_RUNTIME == Path("/usr/lib/llvm-20/lib/libomp.so.5")
    assert runner.EXPECTED_OPENMP_RUNTIME_SHA256 == "98b1f8225260f138243e8e3e7578b83802e998a240f841dc1944a908bf1aee70"


def test_weighted_aggregation_and_per_prompt_equality() -> None:
    assert runner.weighted_tps([{"n": 100, "ms": 1000}, {"n": 100, "ms": 3000}], "n", "ms") == 50.0
    rows = valid_summary_rows()
    summary = runner.summarize(rows)
    assert len(summary["output_stability_observation"]["rows"]) == 30
    assert summary["status"] == "ok"
    rows[-1]["prompt_rows"][-1]["content_sha256"] = "not-equal"
    assert runner.summarize(rows)["status"] == "ok"
    assert runner.summarize(rows)["output_stability_observation"]["exact_equality_rate"] < 1.0
    complete_rows = [dict(row) for row in rows]
    rows.pop()
    with pytest.raises(RuntimeError, match="key set"):
        runner.summarize(rows)
    rows = rows + [rows[0]]
    with pytest.raises(RuntimeError, match="key set"):
        runner.summarize(rows)
    rows = [dict(row) for row in complete_rows]
    rows[0]["cleanup"] = {"status": "fail"}
    with pytest.raises(RuntimeError, match="cleanup"):
        runner.summarize(rows)
    rows = json.loads(json.dumps(complete_rows))
    rows[0]["warmup"] = {"status": "fail"}
    with pytest.raises(RuntimeError, match="warmup"):
        runner.summarize(rows)
    rows = json.loads(json.dumps(complete_rows))
    rows[0]["prompt_rows"][0]["semantic_validation"]["valid"] = False
    with pytest.raises(RuntimeError, match="complete replicates"):
        runner.summarize(rows)
    rows = json.loads(json.dumps(complete_rows))
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(RuntimeError, match="schedule"):
        runner.summarize(rows)
    rows = json.loads(json.dumps(complete_rows))
    rows[0]["iqk_engagement"]["active_type_codes"] = []
    with pytest.raises(RuntimeError, match="IQK engagement"):
        runner.summarize(rows)
    rows = json.loads(json.dumps(complete_rows))
    rows[10]["iqk_engagement"]["active_type_codes"] = [12]
    with pytest.raises(RuntimeError, match="IQK engagement"):
        runner.summarize(rows)


def test_summarize_surfaces_primary_error_and_status_before_warmup_or_numeric_validation() -> None:
    rows = valid_summary_rows()
    rows[0]["status"] = "error"
    rows[0]["primary_error"] = "RuntimeError('semantic validation failed')"
    rows[0]["warmup"] = None
    rows[0]["prompt_tps"] = 0
    with pytest.raises(RuntimeError, match="primary error: RuntimeError\\('semantic validation failed'\\)"):
        runner.summarize(rows)

    rows = valid_summary_rows()
    rows[0]["status"] = "error"
    rows[0]["warmup"] = None
    rows[0]["prompt_tps"] = 0
    with pytest.raises(RuntimeError, match="cell status is not ok: 'error'"):
        runner.summarize(rows)


def test_summarize_rejects_reordered_rows_before_primary_error_preflight() -> None:
    rows = valid_summary_rows()
    rows[0], rows[1] = rows[1], rows[0]
    rows[0]["status"] = "error"
    rows[0]["primary_error"] = "RuntimeError('semantic validation failed')"
    rows[0]["warmup"] = None
    with pytest.raises(RuntimeError, match="balanced paired schedule mismatch"):
        runner.summarize(rows)


@pytest.mark.parametrize("bad_value", [True, 0, -1, float("nan"), float("inf"), float("-inf")])
def test_summarize_rejects_invalid_counts_timings_tps_and_draft_metrics(bad_value: object) -> None:
    rows = valid_summary_rows()
    rows[0]["prompt_tps"] = bad_value
    with pytest.raises(RuntimeError, match="prompt_tps"):
        runner.summarize(rows)
    rows = valid_summary_rows()
    rows[0]["prompt_rows"][0]["prompt_ms"] = bad_value
    with pytest.raises(RuntimeError, match="prompt_ms"):
        runner.summarize(rows)
    rows = valid_summary_rows()
    rows[1]["draft_n"] = bad_value
    with pytest.raises(RuntimeError, match="draft_n"):
        runner.summarize(rows)


def test_cleanup_rejects_descendant_listener_and_kfd_residue(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 99
        returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: int) -> int:
            self.returncode = 0
            return 0

    proc = FakeProcess()
    monkeypatch.setattr(runner.os, "getpgid", lambda _: 99)
    monkeypatch.setattr(runner.os, "killpg", lambda *_: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "port_closed", lambda _: True)
    monkeypatch.setattr(runner, "process_group_members", lambda _: [101])
    monkeypatch.setattr(runner, "system_snapshot", lambda: {"processes": {"exact_llama_processes": [], "autopilot_processes": [], "kfd_owner": False, "rocm_owner": False}})
    with pytest.raises(RuntimeError, match="cleanup"):
        runner.cleanup(proc, 19000)  # type: ignore[arg-type]
    monkeypatch.setattr(runner, "process_group_members", lambda _: [])
    monkeypatch.setattr(runner, "system_snapshot", lambda: {"processes": {"exact_llama_processes": [], "autopilot_processes": [{"pid": 3}], "kfd_owner": False, "rocm_owner": False}})
    with pytest.raises(RuntimeError, match="cleanup"):
        runner.cleanup(FakeProcess(), 19000)  # type: ignore[arg-type]
    monkeypatch.setattr(runner, "system_snapshot", lambda: {"processes": {"exact_llama_processes": [], "autopilot_processes": [], "kfd_owner": True, "rocm_owner": False}})
    with pytest.raises(RuntimeError, match="cleanup"):
        runner.cleanup(FakeProcess(), 19000)  # type: ignore[arg-type]


def test_run_replicate_writes_prompt_artifacts_with_mocked_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeProcess:
        pid = 77
        returncode = None

        def poll(self) -> int | None:
            return self.returncode

    popen_kwargs = {}

    def fake_popen(*_: object, **kwargs: object) -> FakeProcess:
        popen_kwargs.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "wait_for_health", lambda _: None)
    monkeypatch.setattr(runner, "find_free_port", lambda: 19000)
    monkeypatch.setattr(runner, "cleanup", lambda *_: {"status": "pass"})
    monkeypatch.setattr(
        runner,
        "iqk_engagement_evidence",
        lambda _, lane: {"status": "pass", "lane": lane.name, "active_type_codes": [12]},
    )
    monkeypatch.setattr(
        runner,
        "write_boundary_evidence",
        lambda path, *_: runner.write_json(path, {"status": "pass"}) or {"status": "pass"},
    )
    calls = iter(range(3))

    def monitored(_: int, body: dict[str, object], __: int) -> tuple[dict[str, object], dict[str, object], None]:
        messages = body["messages"]
        assert isinstance(messages, list)
        prompt = messages[0]["content"]
        if prompt == runner.WARMUP_PROMPT:
            return {
                "timings": {"prompt_n": 10, "predicted_n": 20, "prompt_ms": 10, "predicted_ms": 20},
                "choices": [{"message": {"content": "1,2,3,4,5"}}],
            }, {"status": "pass"}, None
        index = next(calls)
        content = valid_semantic_content(index + 1)
        return {
            "usage": {"prompt_tokens": 10, "completion_tokens": 100},
            "timings": {"prompt_n": 10, "predicted_n": 100, "prompt_ms": 10, "predicted_ms": 20, "draft_n": 10, "draft_n_accepted": 5},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        }, {"status": "pass"}, None

    monkeypatch.setattr(runner, "monitored_query", monitored)
    result = runner.run_replicate(
        runner.lanes()[0],
        runner.DFLASH,
        1,
        tmp_path,
        {"q4": {"sha256": "x"}},
        {"server": {}},
        1,
        1,
        1,
    )
    rep_dir = tmp_path / "runs/q4_k_m_dflash_rep1"
    assert result["status"] == "ok"
    assert popen_kwargs["env"] == runner.child_env()
    assert result["warmup"]["status"] == "pass"
    assert len(result["prompt_rows"]) == 3
    assert (rep_dir / "warmup_policy.json").is_file()
    assert (rep_dir / "warmup_monitor.json").is_file()
    assert (rep_dir / "request_1.json").is_file()
    assert (rep_dir / "request_1_monitor.json").is_file()
    assert (rep_dir / "request_1_pre_evidence.json").is_file()
    assert (rep_dir / "request_3_post_evidence.json").is_file()
    assert (rep_dir / "response_3.json").is_file()
    assert (rep_dir / "iqk_engagement.json").is_file()
    assert (rep_dir / "result.json").is_file()


def test_run_replicate_fails_on_transient_boundary_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeProcess:
        pid = 77

        def poll(self) -> int | None:
            return None

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_, **__: FakeProcess())
    monkeypatch.setattr(runner, "wait_for_health", lambda _: None)
    monkeypatch.setattr(runner, "find_free_port", lambda: 19000)
    monkeypatch.setattr(runner, "cleanup", lambda *_: {"status": "pass"})
    samples = iter((
        {"status": "pass"},
        {"status": "pass"},
        {"status": "pass"},
        RuntimeError("transient KFD owner"),
    ))

    def boundary(path: Path, *_: object) -> dict[str, object]:
        sample = next(samples)
        if isinstance(sample, Exception):
            runner.write_json(path, {"status": "fail", "error": repr(sample)})
            raise sample
        runner.write_json(path, sample)
        return sample

    monkeypatch.setattr(runner, "write_boundary_evidence", boundary)
    monkeypatch.setattr(runner, "iqk_engagement_evidence", lambda _, lane: {"status": "pass", "lane": lane.name, "active_type_codes": [12]})
    calls = 0

    def monitored(*_: object) -> tuple[dict[str, object], dict[str, object], None]:
        nonlocal calls
        calls += 1
        content = "1,2,3,4,5" if calls == 1 else valid_semantic_content(1)
        return {
            "timings": {"prompt_n": 10, "predicted_n": 100, "prompt_ms": 10, "predicted_ms": 20},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        }, {"status": "pass"}, None

    monkeypatch.setattr(runner, "monitored_query", monitored)
    result = runner.run_replicate(
        runner.lanes()[0],
        runner.BASE,
        1,
        tmp_path,
        {"q4": {}},
        {"server": {}},
        1,
        1,
        1,
    )
    evidence = json.loads((tmp_path / "runs/q4_k_m_base_rep1/request_1_post_evidence.json").read_text())
    assert result["status"] == "error"
    assert "transient KFD owner" in result["primary_error"]
    assert result["warmup"]["status"] == "pass"
    assert evidence["status"] == "fail"
    assert "transient KFD owner" in evidence["error"]


def test_execute_writes_timestamped_run_and_uses_mocked_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models = {"q4": {"sha256": "q4"}, "q8": {"sha256": "q8"}, "drafter": {"sha256": "d"}}
    execution_identity = {
        "candidate": {"head": "abc"},
        "models": models,
        "artifacts": {
            "server": {"sha256": "server"},
            "local_llama_ggml_libraries": [{"sha256": "lib"}],
            "openmp_runtime": {"sha256": "openmp"},
            "models": models,
            "runner": {"sha256": "runner"},
        },
    }
    monkeypatch.setattr(runner, "collect_execution_identity", lambda: json.loads(json.dumps(execution_identity)))
    monkeypatch.setattr(runner, "system_snapshot", lambda: {"processes": {"exact_llama_processes": [], "autopilot_processes": [], "kfd_owner": False, "rocm_owner": False}})
    monkeypatch.setattr(runner, "run_stamp", lambda: "run-fixed")

    def fake_replicate(
        lane: runner.Lane,
        arm: runner.Arm,
        rep: int,
        run_dir: Path,
        _: dict[str, object],
        runtime_artifacts: dict[str, object],
        schedule_position: int,
        lane_position: int,
        pair_position: int,
    ) -> dict[str, object]:
        assert runtime_artifacts["openmp_runtime"] == {"sha256": "openmp"}
        prompts = [{"prompt_index": index, "semantic_validation": {"valid": True, "task": runner.SEMANTIC_TASKS[index - 1]}, "content_sha256": f"{lane.name}:{rep}:{index}", "prompt_tokens": 1, "completion_tokens": 100, "prompt_ms": 1, "decode_ms": 1} for index in range(1, 4)]
        return {"lane": lane.name, "arm": arm.name, "rep": rep, "schedule_position": schedule_position, "lane_position": lane_position, "pair_position": pair_position, "status": "ok", "warmup": {"status": "pass"}, "prompt_rows": prompts, "prompt_tps": 1.0, "decode_tps": 1.0, "completion_tokens": 300, "draft_n": 3 if arm.speculative else None, "draft_n_accepted": 2 if arm.speculative else None, "cleanup": {"status": "pass"}, "iqk_engagement": {"status": "pass", "lane": lane.name, "active_type_codes": [12] if lane.name == "q4_k_m" else []}}

    monkeypatch.setattr(runner, "run_replicate", fake_replicate)
    report = runner.execute(tmp_path)
    run_dir = Path(report["run_dir"])
    assert (run_dir / "identity.json").is_file()
    assert (run_dir / "postflight_identity.json").is_file()
    assert (run_dir / "preflight.json").is_file()
    assert (run_dir / "schedule.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert report["status"] == "ok"


def test_execute_writes_partial_observation_summary_after_strict_semantic_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = {"q4": {"sha256": "q4"}, "q8": {"sha256": "q8"}, "drafter": {"sha256": "d"}}
    execution_identity = {
        "candidate": {"head": "abc"},
        "models": models,
        "artifacts": {"server": {"sha256": "server"}, "local_llama_ggml_libraries": [{"sha256": "lib"}], "openmp_runtime": {"sha256": "openmp"}, "models": models, "runner": {"sha256": "runner"}},
    }
    rows = valid_summary_rows()
    for row in rows:
        if row["lane"] == "q8_0" and row["arm"] == runner.BASE.name:
            row["status"] = "error"
            row["primary_error"] = "RuntimeError('semantic validation failed')"
            row["prompt_rows"][0]["semantic_validation"]["valid"] = False
    by_key = {(row["lane"], row["arm"], row["rep"]): row for row in rows}
    monkeypatch.setattr(runner, "collect_execution_identity", lambda: json.loads(json.dumps(execution_identity)))
    monkeypatch.setattr(runner, "system_snapshot", lambda: {"processes": {"exact_llama_processes": [], "autopilot_processes": [], "kfd_owner": False, "rocm_owner": False}})
    monkeypatch.setattr(runner, "run_stamp", lambda: "run-partial")
    monkeypatch.setattr(runner, "run_replicate", lambda lane, arm, rep, *_args: by_key[(lane.name, arm.name, rep)])

    with pytest.raises(runner.RunFailure, match="cell primary error: RuntimeError"):
        runner.execute(tmp_path)

    partial = json.loads((tmp_path / "run-partial/partial_summary.json").read_text())
    assert partial["schema"] == "epyc.laguna_cpu_dflash_observation.partial_summary.v1"
    assert partial["status"] == "failed/partial"
    assert partial["observation_only"] is True
    assert partial["non_gating"] is True
    assert "cell primary error: RuntimeError" in partial["strict_summary_failure"]
    assert partial["cell_counts"] == {
        "expected": 20,
        "observed": 20,
        "successful_complete": 15,
        "failed_or_incomplete": 5,
        "unexpected_or_unattributed": 0,
    }
    assert partial["arm_summaries"]["q4_k_m_base"]["prompt_tps"]["n"] == 5
    assert partial["arm_summaries"]["q4_k_m_dflash"]["draft_counters"]["acceptance"] == 0.5
    assert partial["arm_summaries"]["q8_0_dflash"]["decode_tps"]["n"] == 5
    assert partial["arm_summaries"]["q8_0_dflash"]["draft_counters"]["acceptance"] == 0.5
    assert "prompt_tps" not in partial["arm_summaries"]["q8_0_base"]
    assert partial["arm_summaries"]["q8_0_base"]["failures"][0]["primary_error"] == "RuntimeError('semantic validation failed')"
    assert partial["arm_summaries"]["q4_k_m_dflash"]["decode_ratio_vs_base_higher_better"] == 1.0
    q8_ratio = partial["arm_summaries"]["q8_0_dflash"]["decode_ratio_vs_base_higher_better"]
    assert q8_ratio["status"] == "unavailable"
    assert "base" in q8_ratio["reason"]
    with pytest.raises(RuntimeError, match="cell primary error: RuntimeError"):
        runner.summarize(rows)
    next(row for row in rows if row["lane"] == "q8_0" and row["arm"] == runner.BASE.name)["warmup"] = None
    historical_partial = runner.partial_summary(rows, "legacy warmup omission")
    assert historical_partial["arm_summaries"]["q8_0_base"]["failures"][0]["warmup_status"] is None


def test_partial_summary_rejects_duplicate_and_missing_expected_cells() -> None:
    rows = valid_summary_rows()
    duplicate = next(
        row
        for row in rows
        if row["lane"] == "q4_k_m"
        and row["arm"] == runner.BASE.name
        and row["rep"] == 5
    )
    duplicate["rep"] = 4

    partial = runner.partial_summary(rows, "duplicate cell")

    q4_base = partial["arm_summaries"]["q4_k_m_base"]
    assert q4_base["status"] == "incomplete"
    assert q4_base["successful_complete_replicates"] == 0
    assert q4_base["failed_or_incomplete_replicates"] == 5
    assert q4_base["duplicate_reps"] == [4]
    assert q4_base["missing_reps"] == [5]
    assert q4_base["unexpected_or_duplicate_rows"] == 1
    assert "prompt_tps" not in q4_base
    q4_ratio = partial["arm_summaries"]["q4_k_m_dflash"][
        "decode_ratio_vs_base_higher_better"
    ]
    assert q4_ratio["status"] == "unavailable"
    assert partial["schedule_sequence"]["status"] == "fail"
    assert partial["cell_counts"]["successful_complete"] == 0
    assert partial["cell_counts"]["failed_or_incomplete"] == 20
    assert partial["cell_counts"]["unexpected_or_unattributed"] == 1


def test_partial_summary_rejects_prompt_contract_drift() -> None:
    rows = valid_summary_rows()
    drifted = next(
        row
        for row in rows
        if row["lane"] == "q4_k_m"
        and row["arm"] == runner.DFLASH.name
        and row["rep"] == 1
    )
    drifted["prompt_rows"][0]["semantic_validation"]["task"] = "wrong_task"

    partial = runner.partial_summary(rows, "prompt drift")

    q4_dflash = partial["arm_summaries"]["q4_k_m_dflash"]
    assert q4_dflash["status"] == "incomplete"
    assert q4_dflash["successful_complete_replicates"] == 4
    assert "task-specific semantic evidence" in q4_dflash["failures"][0]["reason"]
    assert q4_dflash["decode_ratio_vs_base_higher_better"]["status"] == "unavailable"
    assert partial["cell_counts"] == {
        "expected": 20,
        "observed": 20,
        "successful_complete": 19,
        "failed_or_incomplete": 1,
        "unexpected_or_unattributed": 0,
    }
    assert partial["schedule_sequence"]["status"] == "pass"


def test_partial_summary_records_malformed_rows_without_masking_failure() -> None:
    partial = runner.partial_summary(
        [None, {"lane": [], "arm": "base"}],
        "malformed rows",
    )

    assert partial["cell_counts"] == {
        "expected": 20,
        "observed": 2,
        "successful_complete": 0,
        "failed_or_incomplete": 20,
        "unexpected_or_unattributed": 2,
    }
    assert partial["schedule_sequence"]["status"] == "fail"
    assert partial["unattributed_rows"][0]["row_type"] == "NoneType"
    assert "unexpected lane/arm identity" in partial["unattributed_rows"][1]["reason"]


def test_partial_summary_rejects_reordered_complete_cells() -> None:
    rows = valid_summary_rows()
    rows[0], rows[1] = rows[1], rows[0]

    partial = runner.partial_summary(rows, "schedule drift")

    assert partial["schedule_sequence"]["status"] == "fail"
    assert partial["cell_counts"]["successful_complete"] == 0
    assert all(
        summary["status"] == "incomplete"
        for summary in partial["arm_summaries"].values()
    )
    assert (
        partial["arm_summaries"]["q4_k_m_dflash"][
            "decode_ratio_vs_base_higher_better"
        ]["status"]
        == "unavailable"
    )


def test_dry_run_writes_no_inference_artifacts(tmp_path: Path) -> None:
    assert runner.main(["--output-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "prepared_no_inference"
