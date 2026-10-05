"""Kernel path overrides must bypass the store lookup only when explicitly set."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_isolated(code: str, *, env: dict[str, str], args: tuple[str, ...] = ()) -> None:
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _clean_kernel_override_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ORCHESTRATOR_PATHS_LLAMA_CPP_BIN", None)
    env.pop("ORCHESTRATOR_PATHS_LLAMA_MTMD", None)
    env.pop("ORCHESTRATOR_PATHS_LLAMA_SERVER", None)
    return env


def test_explicit_kernel_path_overrides_bypass_store_lookup_on_import() -> None:
    env = _clean_kernel_override_env()
    env["ORCHESTRATOR_PATHS_LLAMA_CPP_BIN"] = "/fixture/kernel-bin"
    env["ORCHESTRATOR_PATHS_LLAMA_MTMD"] = "/fixture/llama-mtmd-cli"
    env["ORCHESTRATOR_PATHS_LLAMA_SERVER"] = "/fixture/llama-server"

    _run_isolated(
        """
        import os
        from pathlib import Path

        import src.registry.kernel_paths as kernel_paths

        def fail_if_looked_up(backend):
            raise AssertionError(f"unexpected production store lookup: {backend}")

        kernel_paths.backend_dir = fail_if_looked_up
        kernel_paths.server_binary = fail_if_looked_up

        from src.config.models import PathsConfig, VisionConfig, WorkerPoolPathsConfig
        import scripts.server.stack_paths as stack_paths

        assert PathsConfig().llama_cpp_bin == Path(os.environ["ORCHESTRATOR_PATHS_LLAMA_CPP_BIN"])
        assert VisionConfig().llama_mtmd_cli == Path(os.environ["ORCHESTRATOR_PATHS_LLAMA_MTMD"])
        assert WorkerPoolPathsConfig().llama_server_path == Path(
            os.environ["ORCHESTRATOR_PATHS_LLAMA_SERVER"]
        )
        assert stack_paths._PATHS["llama_cpp_bin"] == Path(
            os.environ["ORCHESTRATOR_PATHS_LLAMA_CPP_BIN"]
        )

        os.environ["ORCHESTRATOR_PATHS_LLAMA_CPP_BIN"] = ""
        os.environ["ORCHESTRATOR_PATHS_LLAMA_MTMD"] = ""
        os.environ["ORCHESTRATOR_PATHS_LLAMA_SERVER"] = ""
        assert PathsConfig().llama_cpp_bin == Path("")
        assert VisionConfig().llama_mtmd_cli == Path("")
        assert WorkerPoolPathsConfig().llama_server_path == Path("")
        assert stack_paths._get_paths()["llama_cpp_bin"] == Path("")
        """,
        env=env,
    )


def test_unset_overrides_still_fail_closed_for_missing_production_store(
    tmp_path: Path,
) -> None:
    env = _clean_kernel_override_env()
    missing_store = str(tmp_path / "missing-kernel-store")

    _run_isolated(
        """
import sys
from pathlib import Path

import src.registry.kernel_paths as kernel_paths

kernel_paths.PRODUCTION_ROOT = Path(sys.argv[1])
assert not kernel_paths.PRODUCTION_ROOT.exists()

from src.config.models import PathsConfig, VisionConfig, WorkerPoolPathsConfig

for construct in (PathsConfig, VisionConfig, WorkerPoolPathsConfig):
    try:
        construct()
    except kernel_paths.KernelPathError:
        pass
    else:
        raise AssertionError(f"{construct.__name__} accepted a missing production store")
        """,
        env=env,
        args=(missing_store,),
    )

    _run_isolated(
        """
        import sys
        from pathlib import Path

        import src.registry.kernel_paths as kernel_paths

        kernel_paths.PRODUCTION_ROOT = Path(sys.argv[1])
        assert not kernel_paths.PRODUCTION_ROOT.exists()

        try:
            import scripts.server.stack_paths
        except kernel_paths.KernelPathError:
            pass
        else:
            raise AssertionError("stack_paths accepted a missing production store")
        """,
        env=env,
        args=(missing_store,),
    )
