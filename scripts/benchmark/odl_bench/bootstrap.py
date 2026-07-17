"""sys.path + interpreter bootstrap for reaching the orchestrator PDF backends.

Wave-2 B3 (ODL structural bench wiring). This module is READ-ONLY glue: it locates
the ``epyc-orchestrator`` checkout and puts it on ``sys.path`` so that
``src.services.pdf_router`` (owned + preserved by the sibling B2 agent) can be
imported without editing a single orchestrator file.

Resolution order for the orchestrator root (first existing wins):
  1. ``$EPYC_ORCHESTRATOR_ROOT`` (explicit override)
  2. ``/workspace/repos/epyc-orchestrator`` (single-source symlink)
  3. ``/mnt/raid0/llm/epyc-orchestrator`` (canonical tree)

Likewise for the opendataloader-bench harness root (used by the adapter to run
scoring in the bench's OWN venv):
  1. ``$OPENDATALOADER_BENCH_ROOT``
  2. ``/workspace/repos/opendataloader-bench``
  3. ``/mnt/raid0/llm/opendataloader-bench``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ORCH_CANDIDATES = (
    os.environ.get("EPYC_ORCHESTRATOR_ROOT"),
    "/workspace/repos/epyc-orchestrator",
    "/mnt/raid0/llm/epyc-orchestrator",
)

_BENCH_CANDIDATES = (
    os.environ.get("OPENDATALOADER_BENCH_ROOT"),
    "/workspace/repos/opendataloader-bench",
    "/mnt/raid0/llm/opendataloader-bench",
)


def _first_existing(candidates) -> Path | None:
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.exists():
            return p
    return None


def orchestrator_root() -> Path | None:
    """Absolute path to the epyc-orchestrator checkout, or None if not found."""
    return _first_existing(_ORCH_CANDIDATES)


def bench_root() -> Path | None:
    """Absolute path to the opendataloader-bench checkout, or None if not found."""
    return _first_existing(_BENCH_CANDIDATES)


def bench_python() -> Path | None:
    """Path to the bench's own interpreter (has Levenshtein/apted; py3.11).

    Scoring MUST run under this interpreter, not the research venv (py3.14 without
    the bench's scoring deps). Returns None if the bench venv is absent.
    """
    root = bench_root()
    if root is None:
        return None
    py = root / ".venv" / "bin" / "python"
    return py if py.exists() else None


def ensure_orchestrator_on_path() -> Path | None:
    """Insert the orchestrator root at the front of sys.path (idempotent).

    Returns the resolved root, or None if the orchestrator could not be located
    (in which case callers should fall back to stubbed / fake backends).
    """
    root = orchestrator_root()
    if root is None:
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
