"""AK-FAILRULE-1 pre-build rules (adopted 2026-11-06; evidence
/mnt/raid0/llm/tmp/ak-failrule1-20261106/REPORT.md, replay.json).

Rule 1 -- nonempty-suite preflight. A candidate whose correctness oracle would select
0 cases (`oracle_unavailable` at the end of the gate) is refused BEFORE the build, with
the typed gate `oracle_suite_empty`. The suite listing runs once per (anchor binary
sha256, backend, op) and is cached under that key. The key MUST be the anchor binary: a
`target_surface`-keyed variant replayed against the stores caught a real keep.

Rule 2 -- blank-only-hunk lint. A diff with any hunk whose every changed line is blank
(the stray EOF blank-line hunk) is refused before the critic and the build. 13 retained
patches flagged, 0 of 22 commits in the keep ranges.

Pure and injectable: no inference, no process ownership; the lister is one bounded
`test-backend-ops support` call (graph construction only, no compute).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Callable, Mapping
from .tool_output import strip_ansi

#: Typed refusal values. `gate` is the Verdict/refusal_gate string; both are rule gates
#: (a refusal is a scope verdict of the rule, never an authoring defect or evidence).
SUITE_EMPTY_GATE = "oracle_suite_empty"
BLANK_HUNK_CLASS = "blank_only_hunk"

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
LIST_TIMEOUT_S = 120


def blank_only_hunks(diff_text: str) -> list[tuple[str, int]]:
    """`(path, hunk_index)` of every hunk whose changed lines are all blank.

    A hunk with no `+`/`-` lines at all is not flagged (nothing changed)."""
    flagged: list[tuple[str, int]] = []
    path, index, changed, blank = "", -1, 0, True

    def close():
        if index >= 0 and changed and blank:
            flagged.append((path, index))

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            close()
            index, changed, blank = -1, 0, True
            path = line.rsplit(" b/", 1)[-1]
        elif _HUNK.match(line):
            close()
            index, changed, blank = index + 1, 0, True
        elif index >= 0 and line[:1] in ("+", "-") \
                and not line.startswith(("+++ ", "--- ")):
            changed += 1
            if line[1:].strip():
                blank = False
    close()
    return flagged


def blank_only_hunk_refusal(diff_text: str) -> str | None:
    flagged = blank_only_hunks(diff_text)
    if not flagged:
        return None
    where = ", ".join(f"{p}#hunk{i + 1}" for p, i in flagged[:5])
    return (f"{BLANK_HUNK_CLASS}: the diff has hunk(s) that change only blank lines "
            f"({where}); remove the stray whitespace/EOF-newline edit")


def _sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def list_suite_cases(binary: Path, ops: str, backend: str, env: Mapping | None = None) -> int:
    """Supported-case count of the selected ops on `backend`, from `support` mode."""
    done = subprocess.run([str(binary), "support", "-o", ops, "-b", backend],
                          capture_output=True, text=True, timeout=LIST_TIMEOUT_S,
                          env=dict(env) if env is not None else None)
    count = 0
    for line in strip_ansi(done.stdout + done.stderr).splitlines():
        line = line.rstrip()
        if re.match(r"^\s+\w+\(.*\): SUPPORTED$", line):
            count += 1
    return count


class SuitePreflight:
    """Once-per-(anchor binary digest, backend, op) nonempty-suite check."""

    def __init__(self, lister: Callable[[Path, str, str], int] = list_suite_cases,
                 selector: Callable[[str], str] | None = None):
        self._lister = lister
        self._selector = selector
        self._cache: dict[tuple[str, str, str], int] = {}
        self.listings = 0

    def cases(self, binary: Path, op: str, backend: str) -> int:
        key = (_sha256(binary), backend, op)
        if key not in self._cache:
            selector = self._selector(op) if self._selector else op
            self.listings += 1
            self._cache[key] = int(self._lister(binary, selector, backend))
        return self._cache[key]

    def refusal(self, anchor_build: Path, ops, backend: str) -> str | None:
        """A typed reason when any op's suite is empty on the ANCHOR binary, else None.

        A missing binary or a lister fault is NOT a refusal: the existing gate keeps
        its own `oracle_unavailable` verdict for harness faults (fail to the old path)."""
        binary = Path(anchor_build) / "bin" / "test-backend-ops"
        if not binary.is_file():
            return None
        for op in ops:
            try:
                n = self.cases(binary, op, backend)
            except (OSError, subprocess.SubprocessError, ValueError):
                return None
            if n <= 0:
                return (f"{SUITE_EMPTY_GATE}: test-backend-ops selects 0 {op} cases on "
                        f"{backend} for the anchor binary; the correctness oracle would "
                        "be empty -- harness/route fact, NOT evidence about the patch")
        return None
