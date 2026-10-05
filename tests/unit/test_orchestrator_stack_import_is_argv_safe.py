"""Importing the stack launcher must have NO argv side effect.

Regression for the import-time ``os.execv`` re-exec in
``scripts/server/orchestrator_stack.py``: it replaced the *importer's* process
with ``orchestrator_stack.py <importer's argv[1:]>``. These probes import an
external, source-pinned fixture clone so the real host capacity guard is active
without importing the accepted production lineup into the pytest process.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.ni63_capacity_child.source_clone import (
    assert_real_available_headroom,
    materialize_source_clone,
    probe_environment,
    read_real_meminfo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = Path(sys.executable)
HOSTILE_ARGV = [
    ["--master-registry", "foo.yaml", "--revision", "deadbeef"],
    ["start", "--only", "worker_general"],
    ["--help"],
    ["stop", "--all"],
]


def _foreign_interpreter() -> str:
    """Return a real interpreter distinct from the project venv interpreter."""
    resolved_venv = VENV_PY.resolve(strict=True)
    candidates = [
        "/usr/bin/python3",
        shutil.which("python3"),
        getattr(sys, "_base_executable", None),
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        resolved = Path(candidate).resolve(strict=True)
        if resolved != resolved_venv:
            return str(resolved)
    pytest.fail("no real interpreter distinct from the project venv python is available")


@pytest.fixture(scope="module")
def fit_clone(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # The fixture's 0.01 GiB role plus tiny KV declaration is covered with the
    # fixture-only 1 GiB reserve against actual, unchanged runner meminfo.
    meminfo = read_real_meminfo()
    assert_real_available_headroom(0.0101, 1.0, meminfo)
    private_parent = tmp_path_factory.mktemp("ni63-capacity-child")
    return materialize_source_clone(
        REPO_ROOT,
        private_parent,
        variant="fit",
        project_venv_python=VENV_PY,
    )


def _run_probe(clone: Path, code: str, extra_argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_foreign_interpreter(), "-c", code, *extra_argv],
        cwd=str(clone),
        capture_output=True,
        text=True,
        env=probe_environment(clone),
        timeout=300,
        check=False,
    )


def _import_code(clone: Path, trailing: str = "") -> str:
    return (
        "import os, sys; sys.path.insert(0, %r); pid = os.getpid();"
        "import scripts.server.orchestrator_stack as m;"
        "assert m.__file__ and os.getpid() == pid;"
        "%s"
    ) % (str(clone), trailing)


def test_import_with_hostile_argv_does_not_exit_or_print(fit_clone: Path) -> None:
    for extra in HOSTILE_ARGV:
        proc = _run_probe(fit_clone, _import_code(fit_clone), extra)
        assert proc.returncode == 0, (
            f"importing orchestrator_stack with argv {extra!r} exited "
            f"{proc.returncode}: {proc.stderr.strip()!r}"
        )
        assert proc.stdout == "", f"import wrote to stdout with argv {extra!r}: {proc.stdout!r}"
        assert proc.stderr == "", f"import wrote to stderr with argv {extra!r}: {proc.stderr!r}"


def test_import_does_not_replace_the_importing_process(fit_clone: Path) -> None:
    code = _import_code(
        fit_clone,
        "sys.stderr.write('SURVIVED' if os.getpid() == pid else 'REPLACED')",
    )
    proc = _run_probe(fit_clone, code, ["--master-registry", "foo.yaml"])
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.strip() == "SURVIVED", (proc.stdout, proc.stderr)


def test_reexec_helper_is_still_available_for_the_entry_point(fit_clone: Path) -> None:
    """Keep callable/main-guard/single-execv checks inside the real foreign child."""
    code = _import_code(
        fit_clone,
        "source = open(m.__file__, encoding='utf-8').read();"
        "assert callable(m._reexec_under_project_venv);"
        "assert 'if __name__ == \\\"__main__\\\":\\n    # Runs BEFORE' in source;"
        "assert source.count('os.execv(') == 1",
    )
    proc = _run_probe(fit_clone, code, [])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_host_capacity_guard_refuses_oversized_fixture(tmp_path: Path) -> None:
    private_parent = tmp_path
    clone = materialize_source_clone(
        REPO_ROOT,
        private_parent,
        variant="oversized",
        project_venv_python=VENV_PY,
    )
    proc = _run_probe(clone, _import_code(clone), [])
    assert proc.returncode != 0, "oversized synthetic lineup unexpectedly passed host capacity guard"
    assert proc.stdout == ""
    assert "Declared serving_shape lineup does not fit the hardware" in proc.stderr
    assert "device host (CPU RAM) OVERSUBSCRIBED" in proc.stderr
