"""Contract tests for the shared bounded llama-mtmd CLI probe."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import pytest

from src.services.mtmd_probe import run_mtmd_probe
from src.services.lightonocr_llama_server import _probe_mtmd_cli
from src.vision.analyzers import vl_describe
from src.vision.analyzers.vl_describe import VLDescribeAnalyzer


ROOT = Path(__file__).resolve().parents[2]
_HELPER = ROOT / "scripts/lib/mtmd_probe.sh"


def _candidate(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_shared_probe_merges_output_and_prefixes_candidate_library_dir(tmp_path, monkeypatch):
    cli = _candidate(tmp_path / "build" / "llama-mtmd-cli",
                     'printf "ld=%s\\n" "$LD_LIBRARY_PATH"\n'
                     'printf "version: 10303 (abc123)\\n" >&2\n')
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old/lib:/older/lib")

    proc = run_mtmd_probe(cli)

    assert proc is not None and proc.returncode == 0
    assert proc.stdout.splitlines() == [
        f"ld={cli.parent.resolve()}:/old/lib:/older/lib",
        "version: 10303 (abc123)",
    ]


def test_shared_probe_moves_an_existing_candidate_dir_to_front_once(tmp_path, monkeypatch):
    cli = _candidate(tmp_path / "build" / "llama-mtmd-cli",
                     'printf "%s\\n" "$LD_LIBRARY_PATH"\n')
    resolved = str(cli.parent.resolve())
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/old/lib:{resolved}:/older/lib:{resolved}")

    proc = run_mtmd_probe(cli)

    assert proc is not None and proc.returncode == 0
    assert proc.stdout.strip() == f"{resolved}:/old/lib:/older/lib"


def test_shared_probe_uses_symlink_target_directory_for_library_prefix(tmp_path, monkeypatch):
    target = _candidate(tmp_path / "real-build" / "llama-mtmd-cli",
                        'printf "ld=%s\\n" "$LD_LIBRARY_PATH"\n')
    alias_dir = tmp_path / "configured"
    alias_dir.mkdir()
    alias = alias_dir / "llama-mtmd-cli"
    alias.symlink_to(target)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old/lib")

    proc = run_mtmd_probe(alias)

    assert proc is not None and proc.returncode == 0
    assert proc.stdout.strip() == f"ld={target.parent.resolve()}:/old/lib"


def test_shared_probe_preserves_nonzero_status_and_complete_merged_text(tmp_path):
    cli = _candidate(tmp_path / "bin" / "llama-mtmd-cli",
                     'printf "version: 10303 (abc123)\\n"\n'
                     'printf "probe failed\\n" >&2\nexit 7\n')

    proc = run_mtmd_probe(cli)

    assert proc is not None and proc.returncode == 7
    assert proc.stdout == "version: 10303 (abc123)\nprobe failed\n"
    assert _probe_mtmd_cli(cli) is None
    assert VLDescribeAnalyzer._mtmd_runs(cli) is None


def test_python_callers_keep_distinct_strict_and_tolerant_parsers(tmp_path):
    cli = _candidate(tmp_path / "bin" / "llama-mtmd-cli",
                     'printf "prefix version: build-10303\\n"\n')

    assert _probe_mtmd_cli(cli) is None
    assert VLDescribeAnalyzer._mtmd_runs(cli) == "prefix version: build-10303"


def test_shared_probe_reports_nonexecutable_candidate(tmp_path):
    proc = run_mtmd_probe(tmp_path / "missing" / "llama-mtmd-cli")

    assert proc is not None and proc.returncode == 126
    assert proc.stdout == ""


def test_vl_resolver_keeps_configured_then_production_candidate_order(monkeypatch, tmp_path):
    # Hosted CI supplies an intentionally absent configured executable before eager
    # config imports; remove that override locally so this test exercises fallback order.
    monkeypatch.delenv("ORCHESTRATOR_PATHS_LLAMA_MTMD", raising=False)
    monkeypatch.delenv("LLAMA_MTMD_CLI", raising=False)
    root = tmp_path / "llama.cpp"
    configured = root / "configured/bin/llama-mtmd-cli"
    successful = root / "build-hip/bin/llama-mtmd-cli"
    monkeypatch.setattr(vl_describe, "LLAMA_MTMD_CLI", configured)
    seen = []

    def probe(candidate):
        seen.append(candidate)
        return "version: 10303 (abc123)" if candidate == successful else None

    monkeypatch.setattr(VLDescribeAnalyzer, "_mtmd_runs", staticmethod(probe))
    analyzer = object.__new__(VLDescribeAnalyzer)

    assert analyzer._resolve_mtmd_cli() == successful
    assert seen == [configured, root / "build/bin/llama-mtmd-cli", successful]


def test_shared_probe_captures_timeout_status_and_output(tmp_path, monkeypatch):
    cli = _candidate(tmp_path / "bin" / "llama-mtmd-cli", 'exit 0\n')
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    timeout = _candidate(fake_bin / "timeout",
                         'printf "argv=%s\\n" "$*"\n'
                         'exit 124\n')
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")

    proc = run_mtmd_probe(cli)

    assert proc is not None and proc.returncode == 124
    assert proc.stdout.strip() == f"argv=--kill-after=2 20 {cli} --version"
    assert timeout.exists()


@pytest.mark.skipif(
    os.environ.get("EPYC_RUN_HOSTED_TIMEOUT_CONTROL") != "1",
    reason="real 20-second timeout and child-death control is hosted-only",
)
def test_timeout_kills_term_ignoring_child_and_leaves_no_process(tmp_path):
    pid_file = tmp_path / "child.pid"
    cli = tmp_path / "bin" / "llama-mtmd-cli"
    cli.parent.mkdir(parents=True)
    cli.write_text(
        "#!/usr/bin/env python3\n"
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)

    child_pid = int(pid_file.read_text(encoding="utf-8").strip())

    def identity(pid):
        """Return Linux PID plus start-time identity; zombies count as present."""
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        # comm is parenthesized and may itself contain spaces or ')'.
        fields_after_comm = stat_text[stat_text.rfind(")") + 2:].split()
        return (pid, fields_after_comm[19])  # field 22: starttime

    child_identity = identity(child_pid)
    assert child_identity is not None

    def wait_gone(expected, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if identity(expected[0]) != expected:
                return True
            time.sleep(0.05)
        return identity(expected[0]) != expected

    try:
        proc = run_mtmd_probe(cli)
        assert proc is not None and proc.returncode in (124, 137)
        assert wait_gone(child_identity), (
            f"timeout left captured TERM-ignoring child alive: {child_identity}"
        )
    finally:
        # Cleanup is limited to the exact captured process instance, never a reused PID.
        if identity(child_pid) == child_identity:
            os.kill(child_pid, signal.SIGKILL)
            assert wait_gone(child_identity), (
                f"captured child survived SIGKILL or was not reaped: {child_identity}"
            )


def _copy_env_library(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    lib = project / "scripts/lib"
    lib.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/lib/env.sh", lib / "env.sh")
    shutil.copy2(_HELPER, lib / "mtmd_probe.sh")
    return lib / "env.sh"


def _source_env(env_sh: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", 'set -euo pipefail; source "$1"; printf "%s\\n" "$ORCHESTRATOR_PATHS_LLAMA_MTMD"',
         "bash", str(env_sh)],
        env=env, text=True, capture_output=True, check=False,
    )


def test_env_sh_rejects_nonzero_version_output_then_selects_next_candidate(tmp_path):
    env_sh = _copy_env_library(tmp_path)
    app_cli = _candidate(tmp_path / "project/llama.cpp/build/llama-mtmd-cli",
                         'printf "version: bad but nonzero\\n"\nexit 7\n')
    fallback = _candidate(tmp_path / "llm/llama.cpp/build/bin/llama-mtmd-cli",
                          'printf "version: 10303 (abc123)\\n"\n')
    env = dict(os.environ)
    env.update({
        "ORCHESTRATOR_PATHS_LLM_ROOT": str(tmp_path / "llm"),
        "ORCHESTRATOR_PATHS_PROJECT_ROOT": str(tmp_path / "project"),
        "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN": str(app_cli.parent),
        "LD_LIBRARY_PATH": "/prior/lib",
    })
    env.pop("ORCHESTRATOR_PATHS_LLAMA_MTMD", None)

    result = _source_env(env_sh, env)

    assert result.returncode == 0
    assert result.stdout.strip() == str(fallback)


def test_env_sh_preserves_explicit_mtmd_override_without_probing(tmp_path):
    env_sh = _copy_env_library(tmp_path)
    marker = tmp_path / "was-probed"
    _candidate(tmp_path / "project/llama.cpp/build/llama-mtmd-cli",
               f'touch "{marker}"\nprintf "version: 10303 (abc123)\\n"\n')
    override = str(tmp_path / "operator-selected/llama-mtmd-cli")
    env = dict(os.environ)
    env.update({
        "ORCHESTRATOR_PATHS_LLM_ROOT": str(tmp_path / "llm"),
        "ORCHESTRATOR_PATHS_PROJECT_ROOT": str(tmp_path / "project"),
        "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN": str(tmp_path / "project/llama.cpp/build"),
        "ORCHESTRATOR_PATHS_LLAMA_MTMD": override,
    })

    result = _source_env(env_sh, env)

    assert result.returncode == 0
    assert result.stdout.strip() == override
    assert not marker.exists()
