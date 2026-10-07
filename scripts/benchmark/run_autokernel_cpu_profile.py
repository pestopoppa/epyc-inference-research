#!/usr/bin/env python3
"""Installed one-request entry point; only the campaign worker should execute it."""
from pathlib import Path
import sys

# Pin both package namespaces to this installed checkout before any import.
# Ambient PYTHONPATH may name the canonical clone while this entry is a worktree.
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts/kernel_rnd"))
from autokernel.loop.cpu_profile import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
