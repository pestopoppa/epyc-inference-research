"""Isolation is opt-in and setup failures must never become scorer verdicts."""
import subprocess
import os

import pytest

from scripts.benchmark import code_exec_scorer as scorer


def test_default_scaffold_preserves_scoring():
    assert scorer.score_code("print(3)", [{"input": "", "output": "3"}])["correct"]
    assert not scorer.score_code("print(4)", [{"input": "", "output": "3"}])["correct"]


def test_missing_sandbox_fails_closed():
    with pytest.raises(scorer.SandboxUnavailable):
        scorer.score_code("print(3)", [{"input": "", "output": "3"}],
                          isolation="bubblewrap", bwrap_exe="/nonexistent/bwrap")


def test_outside_runtime_refused(tmp_path):
    executable = tmp_path / "bwrap"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    with pytest.raises(scorer.SandboxUnavailable, match="interpreters under /usr"):
        scorer._run_once("pass", "", 1, 1, 128, python_exe="/tmp/python",
                         isolation="bubblewrap", bwrap_exe=str(executable))


def test_mount_allowlist_and_namespaces(tmp_path):
    executable = tmp_path / "bwrap"
    executable.write_text("stub")
    executable.chmod(0o755)
    source = tmp_path / "sol.py"
    source.write_text("pass")
    cmd = scorer._bubblewrap_command("/usr/bin/python3", source, str(executable), "marker")
    assert all(flag in cmd for flag in ("--unshare-user", "--unshare-net", "--unshare-pid"))
    binds = [(cmd[i + 1], cmd[i + 2]) for i, value in enumerate(cmd) if value == "--ro-bind"]
    assert binds == [("/usr", "/usr"), (str(source), "/work/sol.py")]
    assert ["--tmpfs", "/tmp"] == cmd[cmd.index("--tmpfs"):cmd.index("--tmpfs") + 2]


def test_launcher_failure_is_not_a_model_verdict(tmp_path):
    executable = tmp_path / "bwrap"
    executable.write_text("#!/bin/sh\necho setup-refused >&2\nexit 1\n")
    executable.chmod(0o755)
    with pytest.raises(scorer.SandboxUnavailable, match="setup-refused"):
        scorer._run_once("print(3)", "", 5, 4, 512, python_exe="/usr/bin/python3",
                         isolation="bubblewrap", bwrap_exe=str(executable))


def test_readiness_marker_is_removed_from_candidate_output(monkeypatch, tmp_path):
    executable = tmp_path / "bwrap"
    executable.write_text("stub")
    executable.chmod(0o755)
    class FakeProcess:
        returncode = 0
        pid = 123
        def __init__(self, args, **kwargs):
            self.args = args
        def communicate(self, **kwargs):
            return self.args[-2] + "3\n", ""
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    ok, output = scorer._run_once("print(3)", "", 5, 4, 512,
                                   python_exe="/usr/bin/python3", isolation="bubblewrap",
                                   bwrap_exe=str(executable))
    assert ok and output == "3\n"


def test_docker_requires_image_id():
    with pytest.raises(scorer.SandboxUnavailable, match="immutable local image"):
        scorer.score_code("print(3)", [{"output": "3"}], isolation="docker",
                          docker_image="python:latest")


def test_docker_requires_valid_affinity_subset(monkeypatch):
    image = "sha256:" + "a" * 64
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 3})
    for cpuset in (None, "", "2-1", "3,4", "2-999999", "2;echo x"):
        with pytest.raises(scorer.SandboxUnavailable):
            scorer._run_once("pass", "", 2, 1, 128, isolation="docker",
                             docker_image=image, docker_cpuset=cpuset)
    assert scorer._parse_cpuset("2-3,5") == {2, 3, 5}


@pytest.mark.skipif(not os.getenv("CODEEXEC_TEST_DOCKER_IMAGE") or not os.getenv("CODEEXEC_TEST_CPUSET"), reason="no explicit local image or claimed CPU set")
def test_docker_actual_boundary_and_timeout(tmp_path):
    image = os.environ["CODEEXEC_TEST_DOCKER_IMAGE"]
    cpuset = os.environ["CODEEXEC_TEST_CPUSET"]
    secret = tmp_path / "host-secret"
    secret.write_text("host-only")
    code = (f"import socket,os\nprint(','.join(map(str,sorted(os.sched_getaffinity(0)))))\ntry:\n open({str(secret)!r}).read()\n print('LEAK')\n"
            "except OSError:\n print('denied')\n"
            "s=socket.socket(); s.settimeout(1)\nprint(s.connect_ex(('1.1.1.1',80)))\n")
    ok, out = scorer._run_once(code, "", 15, 8, 256, isolation="docker", docker_image=image, docker_cpuset=cpuset)
    assert ok and out.splitlines() == [','.join(map(str,sorted(scorer._parse_cpuset(cpuset)))), "denied", "101"]
    assert scorer.score_code("print(3)", [{"output": "3"}], isolation="docker",
                             docker_image=image, docker_cpuset=cpuset)["correct"]
    assert scorer._run_once("import time; time.sleep(30)", "", 2, 8, 256,
                            isolation="docker", docker_image=image, docker_cpuset=cpuset) == (False, "__timeout__")
