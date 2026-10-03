"""Wall-clock launch / stop banners in stack-managed server logs.

llama-server log lines carry process-relative time only, and the stack appends
every launch to the same file, so a week of traffic could only be placed in time by
chaining exit lines back from a state file (workspace-89, 2026-10-03). Every launch
branch of ``start_server`` now writes a banner before ``Popen`` and the pid after it;
every kill path writes a stop banner into the log the process's stdout points at.
No real process is started or signalled here.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from scripts.server import orchestrator_stack as osk
from scripts.server import stack_log_banner as banner
from scripts.server import stack_processes
from scripts.server.stack_manifest import WORKER_POOL_MODELS

ARGV = ["numactl", "--interleave=all", "/k/production/cpu/llama-server", "-m", "/m/q.gguf",
        "--port", "8070"]


def test_launch_banner_then_pid_then_sidecar(tmp_path):
    log_path = tmp_path / "llama-server-8070.log"
    with open(log_path, "a") as log:
        rec = banner.write_launch_banner(
            log, port=8070, roles=["frontdoor", "worker_explore"], argv=ARGV,
            env={"LD_LIBRARY_PATH": "/k/production/cpu"}, binary=ARGV[2],
        )
        log.write("server output\n")
        banner.write_launch_pid(log, rec, 26641)

    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("=== launch 20") and "port=8070" in lines[0]
    assert "roles=frontdoor,worker_explore" in lines[0] and rec["launch_id"] in lines[0]
    detail = json.loads(lines[1][len("=== launch-detail "):-len(" ===")])
    assert detail["argv"] == ARGV and detail["model_path"] == "/m/q.gguf"
    assert detail["ld_library_path"] == "/k/production/cpu"
    assert lines[3].startswith("=== launched ") and "pid=26641" in lines[3]

    sidecar = json.loads((tmp_path / "server_launches" / "8070.json").read_text())
    assert sidecar["pid"] == 26641 and sidecar["launch_id"] == rec["launch_id"]
    assert sidecar["argv_sha256"] == banner.argv_sha256(ARGV) and len(sidecar["argv_sha256"]) == 64
    assert sidecar["binary"] == ARGV[2] and sidecar["roles"] == ["frontdoor", "worker_explore"]
    history = (tmp_path / "server_launches" / "launches.jsonl").read_text().splitlines()
    assert len(history) == 1 and json.loads(history[0])["launch_id"] == rec["launch_id"]


def test_argv_digest_is_boundary_sensitive():
    assert banner.argv_sha256(["a b", "c"]) != banner.argv_sha256(["a", "b c"])


def test_banner_failure_never_raises(tmp_path):
    class _Broken:
        name = str(tmp_path / "x.log")

        def write(self, *_):
            raise OSError("disk full")

        def flush(self):
            raise OSError("disk full")

    rec = banner.write_launch_banner(_Broken(), port=1, roles=["r"], argv=["llama-server"])
    banner.write_launch_pid(_Broken(), rec, 1)  # must not raise


# ── every start_server branch writes the banner (fake Popen) ─────────────────


@pytest.fixture
def harness(monkeypatch, tmp_path):
    def fake_popen(argv, **kwargs):
        log = kwargs["stdout"]
        log.write("fake server output\n")
        log.flush()
        return types.SimpleNamespace(pid=4242)

    monkeypatch.setattr(osk.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(osk, "build_server_command",
                        lambda *a, **k: ["/k/bin/llama-server", "-m", "/m/fake.gguf", "--port", "1"])
    monkeypatch.setattr(osk, "_bench_guarded_numa_prefix", lambda *a, **k: ["numactl", "--interleave=all"])
    monkeypatch.setattr(osk, "_embedding_spawn_prefix", lambda *a, **k: ["taskset", "-c", "0-3"])
    monkeypatch.setattr(osk, "_write_llama_marker", lambda *a, **k: None)
    monkeypatch.setattr(osk, "wait_for_health", lambda port, timeout=0: True)
    monkeypatch.setattr(osk, "pre_evict_nodes", lambda gib, **kw: (True, "faked"))
    monkeypatch.setattr(osk, "placement_summary", lambda pid: "faked")
    monkeypatch.setattr(osk, "_renice_all_threads", lambda *a, **k: None)
    monkeypatch.setattr(osk, "LOG_DIR", tmp_path)
    monkeypatch.setattr(osk, "_stack_prior_runtime_overrides", lambda role: ("/k/bin/llama-server", []))
    monkeypatch.setattr(osk, "_runtime_requirements_for_role", lambda registry, role: ("/k/bin", []))
    monkeypatch.setattr(osk, "_stack_prior_launch", lambda role: ({"model_path": "/m/fake.gguf"}, {}))
    model = types.SimpleNamespace(name="fake-model", full_path="/m/fake.gguf")
    registry = types.SimpleNamespace(get_role=lambda r: types.SimpleNamespace(model=model, name=r))
    return types.SimpleNamespace(registry=registry, log_dir=tmp_path)


BRANCHES = {
    "registry": dict(roles=["frontdoor"], kw={}),
    "vision": dict(roles=["worker_vision"], kw={"vision_mode": True}),
    "gpu_shadow_lane": dict(roles=["frontdoor"], kw={"gpu_shadow_lane_mode": True}),
    "worker_pool": dict(roles=["worker_general"],
                        kw={"worker_pool_mode": True, "worker_type": sorted(WORKER_POOL_MODELS)[0]}),
    "embedding": dict(roles=["embedder"], kw={"embedding_mode": True}),
    "eval_batch": dict(roles=["frontdoor"], kw={"eval_batch_frontdoor_mode": True}),
}


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_every_launch_branch_writes_banner_before_output(harness, branch):
    spec = BRANCHES[branch]
    assert osk.start_server(8070, spec["roles"], harness.registry, **spec["kw"]) is not None
    [log_path] = list(Path(harness.log_dir).glob("*.log"))
    lines = log_path.read_text().splitlines()
    assert lines[0].startswith("=== launch ") and "port=8070" in lines[0]
    assert lines[1].startswith("=== launch-detail ")
    assert lines.index("fake server output") > 1
    assert any(l.startswith("=== launched ") and "pid=4242" in l for l in lines)
    sidecar = json.loads((Path(harness.log_dir) / "server_launches" / "8070.json").read_text())
    assert sidecar["pid"] == 4242 and sidecar["binary"] == "/k/bin/llama-server"
    assert sidecar["argv"][-4:] == ["-m", "/m/fake.gguf", "--port", "1"]
    assert sidecar["model_path"] == "/m/fake.gguf"


# ── stop banners from kill_process_tree (no real signals) ────────────────────


def _fake_kill_env(monkeypatch, alive_after: set[int]):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(stack_processes, "collect_descendants", lambda pid: [])
    monkeypatch.setattr(stack_processes.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(stack_processes, "pid_alive", lambda pid: pid in alive_after)
    monkeypatch.setattr(stack_processes.time, "sleep", lambda s: None)
    return sent


def test_stop_banner_brackets_a_graceful_stop(monkeypatch, tmp_path):
    log_path = tmp_path / "llama-server-8070.log"
    log_path.write_text("server output\n")
    monkeypatch.setattr(banner, "stdout_log_of", lambda pid, log_dir=None: log_path)
    sent = _fake_kill_env(monkeypatch, alive_after=set())
    assert stack_processes.kill_process_tree(777001, timeout=2)
    lines = log_path.read_text().splitlines()
    assert lines[1].startswith("=== stop ") and "pid=777001" in lines[1]
    assert lines[2].startswith("=== stopped ") and "result=exited_on_sigterm" in lines[2]
    assert len(sent) == 1


def test_stop_banner_records_sigkill_escalation(monkeypatch, tmp_path):
    log_path = tmp_path / "llama-server-8083.log"
    log_path.write_text("")
    monkeypatch.setattr(banner, "stdout_log_of", lambda pid, log_dir=None: log_path)
    state = {"alive": {777002}}
    sent: list = []
    monkeypatch.setattr(stack_processes, "collect_descendants", lambda pid: [])
    monkeypatch.setattr(stack_processes.time, "sleep", lambda s: None)

    def _kill(pid, sig):
        sent.append(sig)
        if sig == stack_processes.signal.SIGKILL:
            state["alive"].discard(pid)

    monkeypatch.setattr(stack_processes.os, "kill", _kill)
    monkeypatch.setattr(stack_processes, "pid_alive", lambda pid: pid in state["alive"])
    assert stack_processes.kill_process_tree(777002, timeout=1)
    assert "result=killed_sigkill" in log_path.read_text().splitlines()[-1]


def test_no_banner_for_a_process_without_a_stack_log(monkeypatch, tmp_path):
    monkeypatch.setattr(banner, "stdout_log_of", lambda pid, log_dir=None: None)
    _fake_kill_env(monkeypatch, alive_after=set())
    assert stack_processes.kill_process_tree(777003, timeout=1)
    assert list(tmp_path.iterdir()) == []


def test_stdout_log_of_only_accepts_regular_log_files_in_the_log_dir(monkeypatch, tmp_path):
    inside = tmp_path / "logs" / "llama-server-8070.log"
    inside.parent.mkdir()
    inside.write_text("")
    outside = tmp_path / "elsewhere.log"
    outside.write_text("")
    targets = {1: str(inside), 2: str(outside), 3: "/dev/pts/4", 4: str(tmp_path / "logs" / "x.txt")}
    monkeypatch.setattr(banner.os, "readlink", lambda p: targets[int(p.split("/")[2])])
    assert banner.stdout_log_of(1, inside.parent) == inside
    assert banner.stdout_log_of(2, inside.parent) is None
    assert banner.stdout_log_of(3, inside.parent) is None
    assert banner.stdout_log_of(4, inside.parent) is None
