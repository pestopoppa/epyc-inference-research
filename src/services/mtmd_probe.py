"""Thin Python bridge to the shared Bash llama-mtmd-cli probe."""
from __future__ import annotations

from pathlib import Path
import subprocess


_HELPER = Path(__file__).resolve().parents[2] / "scripts" / "lib" / "mtmd_probe.sh"


def run_mtmd_probe(path: Path) -> subprocess.CompletedProcess[str] | None:
    """Run the shared 20-second, candidate-library-prefixed probe.

    The Bash helper returns merged stdout/stderr and the actual `timeout`/binary status.
    Callers retain their own version-line parsing policy.
    """
    try:
        return subprocess.run(
            [str(_HELPER), str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=22,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


__all__ = ["run_mtmd_probe"]
