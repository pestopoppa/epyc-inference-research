from __future__ import annotations

import sys
import signal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import laguna_q4_cpu_config_discovery as runner


def row(cell: runner.Cell, value: float, rep: int = 1) -> dict:
    return {"cell": cell.name, "rep": rep, "status": "ok", "measurement": {"decode_tok_s": value}}


def test_fixed_recipe_and_cells_are_bounded() -> None:
    assert len(runner.CELLS) == 4 and runner.REPS == 3
    argv = runner.server_argv(runner.CELLS[0])
    assert argv[:6] == [str(runner.base.TASKSET), "-c", "0-95", str(runner.base.NUMACTL), "--interleave=all", str(runner.base.BINARY)]
    assert argv[argv.index("-c", 6) + 1] == "49152"
    assert argv[argv.index("-b") + 1] == "2048" and argv[argv.index("-ub") + 1] == "2048"
    assert argv[argv.index("-ngl") + 1] == "0" and "--no-mmap" in argv
    assert all(argv.count(flag) == 1 for flag in ("-t", "-tb", "-b", "-ub"))
    body = runner.request_body("prompt", 512)
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.6, 0.95, 20)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_each_cell_binds_omp_threads_to_server_threads() -> None:
    for cell in runner.CELLS:
        environment = runner.server_env(cell)
        assert environment["OMP_NUM_THREADS"] == str(cell.threads)
        assert environment["GGML_IQK"] == "1"


def test_selection_requires_both_median_margin_and_minimum() -> None:
    rows = []
    for rep in range(1, 4):
        rows += [row(runner.CELLS[0], 10, rep), row(runner.CELLS[1], 10.3, rep), row(runner.CELLS[2], 12, rep), row(runner.CELLS[3], 9, rep)]
    summaries, selection = runner.summarize(rows)
    assert selection["status"] == "candidate_selected"
    assert selection["selected_cell"] == runner.CELLS[2].name
    assert summaries[runner.CELLS[1].name]["all_ok"]


def test_selection_retains_baseline_on_incomplete_or_weak_candidates() -> None:
    rows = []
    for rep in range(1, 4):
        rows += [
            row(runner.CELLS[0], 10, rep),
            row(runner.CELLS[1], (10.5, 9.9, 10.5)[rep - 1], rep),
            row(runner.CELLS[2], 10, rep),
            row(runner.CELLS[3], 10, rep),
        ]
    _, selection = runner.summarize(rows)
    assert selection["status"] == "baseline_retained"
    assert selection["selected_cell"] == runner.CELLS[0].name


def test_incomplete_baseline_invalidates_selection_and_campaign_state() -> None:
    rows = [row(runner.CELLS[0], 10, rep) for rep in range(1, 3)]
    _, selection = runner.summarize(rows)
    assert selection["status"] == "invalid"
    assert selection["selected_cell"] is None
    schedule = [{"cell": cell.name, "rep": rep} for cell in runner.CELLS for rep in range(1, 4)]
    state = runner.campaign_state("invalid_fatal", schedule, rows, fatal_error="boom")
    assert state["decision_valid"] is False
    assert state["fatal_error"] == "boom"
    assert len(state["remaining_not_attempted"]) == 10


def test_post_campaign_identity_failure_clears_statistical_selection() -> None:
    selection = {
        "status": "candidate_selected",
        "selected_cell": runner.CELLS[2].name,
    }
    terminal = {
        "status": "invalid_fatal",
        "decision_valid": False,
        "fatal_error": "post-campaign model identity failed",
    }
    bound = runner.bind_selection_to_campaign(selection, terminal)
    assert bound["status"] == "invalid"
    assert bound["selected_cell"] is None
    assert "post-campaign model identity failed" in bound["reason"]


def test_continuity_explicitly_records_absent_gpu_only_after_kfd_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "hip_sidecars", lambda: [])
    monkeypatch.setattr(runner, "kfd_client_pids", lambda: [])
    monkeypatch.setattr(runner, "read_swap", lambda: {"SwapTotal": 0, "SwapFree": 0})
    sample = runner.continuity_sample(None)
    assert sample["gpu_workload_status"] == "absent_no_hip_llama_server_detected"
    assert sample["kfd_client_pids"] == []


def test_continuity_fails_on_unattested_kfd_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "hip_sidecars", lambda: [])
    monkeypatch.setattr(runner, "kfd_client_pids", lambda: [1234])
    with pytest.raises(RuntimeError, match="unattested /dev/kfd"):
        runner.continuity_sample(None)


def hip_row() -> dict:
    return {
        "pid": 123,
        "port": 18072,
        "model": "/model.gguf",
        "exe": str((runner.base.LLAMA_ROOT / "build-hip/bin/llama-server").resolve()),
        "listener_owned": True,
    }


def write_cgroup(proc_root: Path, value: str) -> None:
    path = proc_root / "123" / "cgroup"
    path.parent.mkdir(parents=True)
    path.write_text(f"0::{value}\n")


def test_hip_sidecar_accepts_exact_all_tid_mask_and_cgroup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cgroup(tmp_path, runner.SIDE_CGROUP)
    monkeypatch.setattr(runner.base, "live_llama_rows", lambda: [hip_row()])
    monkeypatch.setattr(
        runner.base,
        "proc_thread_cpu_allowed_lists",
        lambda pid, proc_root: [
            {"tid": pid, "cpus_allowed_list": "184-191"},
            {"tid": pid + 1, "cpus_allowed_list": "184-191"},
        ],
    )
    rows = runner.hip_sidecars(tmp_path)
    assert len(rows) == 1
    assert rows[0]["cgroup"] == runner.SIDE_CGROUP
    assert len(rows[0]["thread_cpu_allowed_lists"]) == 2


def test_hip_sidecar_rejects_wrong_worker_thread_affinity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cgroup(tmp_path, runner.SIDE_CGROUP)
    monkeypatch.setattr(runner.base, "live_llama_rows", lambda: [hip_row()])
    monkeypatch.setattr(
        runner.base,
        "proc_thread_cpu_allowed_lists",
        lambda pid, proc_root: [
            {"tid": pid, "cpus_allowed_list": "184-191"},
            {"tid": pid + 1, "cpus_allowed_list": "0-191"},
        ],
    )
    with pytest.raises(RuntimeError, match="GPU sidecar.*affinity"):
        runner.hip_sidecars(tmp_path)


def test_hip_sidecar_rejects_wrong_cgroup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_cgroup(tmp_path, "/")
    monkeypatch.setattr(runner.base, "live_llama_rows", lambda: [hip_row()])
    monkeypatch.setattr(
        runner.base,
        "proc_thread_cpu_allowed_lists",
        lambda pid, proc_root: [{"tid": pid, "cpus_allowed_list": "184-191"}],
    )
    with pytest.raises(RuntimeError, match="cgroup"):
        runner.hip_sidecars(tmp_path)


def test_monitor_lifecycle_encloses_startup_and_never_samples_stale_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidates = []

    def fake_sample(pid):
        candidates.append(pid)
        return {"at": "x", "swap": {"SwapTotal": 0, "SwapFree": 0}}

    monkeypatch.setattr(runner, "continuity_sample", fake_sample)
    monkeypatch.setattr(runner, "read_swap", lambda: {"SwapTotal": 0, "SwapFree": 0})
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    monitor = runner.ContinuityMonitor(
        tmp_path / "continuity.jsonl", interval_s=60, window_samples=1
    )
    monitor.start_prelaunch()
    assert monitor.started and monitor.thread.is_alive()
    assert candidates == [None]
    monitor.attach_candidate(123)
    assert candidates[-1] == 123
    assert monitor.detach_candidate_before_teardown() == 123
    monitor.post_cleanup_window()
    monitor.close()
    assert candidates == [None, 123, 123, None]
    assert not monitor.thread.is_alive()


def test_monitor_transient_sidecar_failure_is_terminal_and_close_before_start_is_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monitor = runner.ContinuityMonitor(tmp_path / "continuity.jsonl")
    monitor.initial_swap = {"SwapTotal": 0, "SwapFree": 0}
    monkeypatch.setattr(
        runner,
        "continuity_sample",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("transient sidecar escape")),
    )
    with pytest.raises(RuntimeError, match="transient sidecar escape"):
        monitor.sample("prelaunch")
    with pytest.raises(RuntimeError, match="transient sidecar escape"):
        monitor.ensure_healthy()
    monitor.close()


def test_iqk_engagement_requires_exact_active_evidence() -> None:
    positive = runner.iqk_engagement(
        "other\n[iqk] ACTIVE: ik_llama GEMM kernels engaged\n"
    )
    assert positive["engaged"] is True and len(positive["matching_lines"]) == 1
    with pytest.raises(RuntimeError, match="IQK ACTIVE"):
        runner.iqk_engagement("[iqk] prepared but inactive")


def response_payload(tokens: int = 256) -> dict:
    return {
        "usage": {"completion_tokens": tokens, "prompt_tokens": 10},
        "timings": {"predicted_per_second": 8.1},
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "patch"},
            }
        ],
    }


def test_measurement_token_floor_and_error_paths() -> None:
    assert (
        runner.validate_measurement(
            response_payload(256), 1.0, min_completion_tokens=256
        )["decode_tok_s"]
        == 8.1
    )
    with pytest.raises(RuntimeError, match="token floor"):
        runner.validate_measurement(
            response_payload(255), 1.0, min_completion_tokens=256
        )
    bad = response_payload()
    bad["timings"] = {}
    with pytest.raises(RuntimeError, match="positive server decode"):
        runner.validate_measurement(bad, 1.0, min_completion_tokens=256)


def test_cleanup_accepts_only_expected_exit_pid_absence_and_free_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, _timeout):
            self.returncode = -signal.SIGTERM
            return self.returncode

    monkeypatch.setattr(runner.os, "killpg", lambda *_args: None)
    monkeypatch.setattr(runner, "pid_exists", lambda _pid: False)
    monkeypatch.setattr(runner.base, "port_free", lambda _port: True)
    evidence = runner.terminate(FakeProcess())
    assert evidence["expected_exit"] and evidence["pid_absent"] and evidence["port_free"]

    monkeypatch.setattr(runner, "pid_exists", lambda _pid: True)
    process = FakeProcess()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.terminate(process)
