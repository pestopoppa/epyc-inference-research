#!/usr/bin/env python3
"""Installed one-request entry point; only the campaign worker should execute it."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))
from autokernel.loop.cpu_profile import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
