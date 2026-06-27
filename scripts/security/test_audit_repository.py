from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "audit_repository.py"


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


def _init_repo(root: Path) -> None:
    _run_git(root, "init")
    _run_git(root, "config", "user.email", "test@example.invalid")
    _run_git(root, "config", "user.name", "Security Audit Test")


def _audit(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *extra],
        check=False,
        capture_output=True,
        text=True,
    )


def test_audit_repository_passes_expected_research_artifacts(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write(tmp_path / "scripts" / "benchmark" / "run.py", "print('ok')\n")
    _write(tmp_path / "benchmarks" / "results" / "large.jsonl", b"x" * 128)
    _run_git(tmp_path, "add", ".")

    result = _audit(tmp_path, "--large-file-limit-bytes", "64")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "security audit ok" in result.stdout


def test_audit_repository_flags_secret_files_and_literals(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write(tmp_path / ".env", "OPENAI_API_KEY=sk-" + "a" * 40 + "\n")
    _write(tmp_path / "scripts" / "tool.py", "TOKEN='ghp_" + "b" * 36 + "'\n")
    _run_git(tmp_path, "add", ".")

    result = _audit(tmp_path, "--json")

    assert result.returncode == 1
    assert "secret_filename" in result.stdout
    assert "secret_literal" in result.stdout
    assert ".env" in result.stdout
    assert "scripts/tool.py" in result.stdout


def test_audit_repository_flags_unexpected_large_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write(tmp_path / "research" / "drop.bin", b"x" * 128)
    _run_git(tmp_path, "add", ".")

    result = _audit(tmp_path, "--large-file-limit-bytes", "64")

    assert result.returncode == 1
    assert "unexpected_large_file" in result.stdout
    assert "research/drop.bin" in result.stdout
