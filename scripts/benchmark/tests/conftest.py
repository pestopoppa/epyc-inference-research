"""Pytest path setup for the benchmark test suite.

The research repo ships no test infra, so these tests are written to run both
under pytest (if it is ever installed) AND via the self-contained stdlib runner
in each test module's ``__main__``. This conftest only ensures the benchmark
directory (and repo scripts dir) are importable so ``import run_batch_entry``
and ``import clean_window_manifest`` resolve during pytest collection.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _BENCHMARK_DIR.parent
_RESEARCH_ROOT = _SCRIPTS_DIR.parent

for _p in (str(_RESEARCH_ROOT), str(_SCRIPTS_DIR), str(_BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
