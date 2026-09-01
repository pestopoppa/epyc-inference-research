#!/usr/bin/env python3
"""Read the operator inbox without letting it end the run (R22-6).

WHY THIS IS HARDENED. `run.py`'s inline reader did a bare per-file
`read_text(encoding="utf-8")`. One invalid-UTF-8 or unreadable file in the LIVE
inbox therefore raised inside `build_context()` on every iteration, so every lane
errored, the pool's consecutive-error breaker tripped, and the run died. The
operator's own injection channel doubled as a kill switch. Here each file is read
under its own try/except: an unreadable file is SKIPPED with an
`inbox_file_unreadable` note and every readable seed still reaches the planner.
The note goes to the run log (stdout) via `note` — the cheapest surface the
operator already reads — and is repeated on every iteration the bad file remains,
which is deliberate: a one-shot note scrolls away, a repeated one names the file
until someone fixes or removes it.

WHY IT IS RE-READ EVERY ITERATION, NOT ONCE AT STARTUP (rationale moved verbatim
from `run.py`, whose `build_context` calls this fresh each time). The inbox is how
a hypothesis reaches the planner from outside the loop -- from a handoff, a
backlog row, or the operator mid-run. Reading it once means anything dropped in
after launch is invisible until the next restart, which is how the channel stayed
empty while the backlog held measured levers for the exact kernels the planner was
re-deriving.

This module lives in `controller/` because it is a LIBRARY the loop calls (like
`build_recipe` and `anchor_integrity`), not loop control flow: what to DO with the
seeds stays in `run.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def read_inbox(inbox_dir: Path,
               note: Callable[[str], object] = print) -> list[str]:
    """Every readable `*.md` seed, sorted by name; unreadable files noted, never raised.

    `note` receives one `inbox_file_unreadable` line per skipped file and defaults
    to `print` so the note lands in the run log. Injected so a test can capture it.

    Catches `OSError` (permissions, dangling symlinks, I/O faults) and
    `UnicodeDecodeError` (a ValueError, NOT an OSError -- the bare reader died on
    exactly this) and nothing broader: a MemoryError or a KeyboardInterrupt must
    still propagate.
    """
    if not inbox_dir.is_dir():
        return []
    texts: list[str] = []
    for path in sorted(inbox_dir.glob("*.md")):
        try:
            texts.append(path.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeDecodeError) as exc:
            note(f"inbox     inbox_file_unreadable {path.name}: "
                 f"{type(exc).__name__}: {exc} — file skipped, run continues")
    return texts


__all__ = ["read_inbox"]
