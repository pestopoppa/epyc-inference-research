"""The declared graph dependency must construct the real project graph."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_supported_dependency_constructs_project_graph_in_fresh_process() -> None:
    env = dict(os.environ)
    env.update(
        ORCHESTRATOR_MOCK_MODE="true",
        ORCHESTRATOR_PATHS_LLAMA_CPP_BIN="/fixture/unused-cpu-bin",
        ORCHESTRATOR_PATHS_LLAMA_MTMD="/fixture/unused-mtmd",
        ORCHESTRATOR_PATHS_LLAMA_SERVER="/fixture/unused-llama-server",
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            from importlib.metadata import version
            from packaging.specifiers import SpecifierSet
            from pydantic_graph import Graph
            from src.graph.graph import orchestration_graph

            assert version("pydantic-graph") in SpecifierSet(">=1.80.0,<2")
            assert isinstance(orchestration_graph, Graph)
            assert orchestration_graph.name == "orchestration"
        """)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
