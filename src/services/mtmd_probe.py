"""Thin Python bridge to the shared Bash llama-mtmd-cli probe."""
from __future__ import annotations

from pathlib import Path
import subprocess


_HELPER = Path(__file__).resolve().parents[2] / "scripts" / "lib" / "mtmd_probe.sh"


def run_mtmd_probe(path: Path) -> subprocess.CompletedProcess[str] | None:
    """Run the shared bounded, candidate-library-prefixed probe.

    The Bash helper uses a 20-second timeout plus a 2-second KILL grace and
    returns merged stdout/stderr with the actual `timeout`/binary status.
    Callers retain their own version-line parsing policy.
    """
    try:
        return subprocess.run(
            [str(_HELPER), str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=25,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


__all__ = ["run_mtmd_probe"]
