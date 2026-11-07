#!/usr/bin/env python3
"""AK-TOOL-UNSOURCED-LINT: every figure shown to the planner must resolve to a raw path.

EVIDENCE CASE. The 11-06 progress table carried H6 "1.017x" with no packet; the real
result was INCONCLUSIVE 0.9997 (data audit 4378a207d). A planner prior built on a number
nobody can open is worse than none.

TWO SHAPES.
  * a nested RECORD (a seed/belief row): a numeric leaf is a figure; it is sourced iff it
    or an ancestor dict carries a source key (`SOURCE_KEYS`, searched recursively through
    the enclosing nesting) whose value is a path that RESOLVES (exists).
  * rendered TEXT: a ratio/percent token (`1.017x`, `+0.17%`) is sourced iff its line
    names a path that resolves (`path`, `path:LINE`, ...).

FLAG vs REFUSE (decision). Default is FLAG: the token is rendered with `UNSOURCED_MARK`
appended and the context header counts them. Refusing would starve the planner of
operator-asserted priors on a lint's say-so; a visible mark lets it discount them and lets
a reviewer grep. `strict=True` raises `UnsourcedFigureRefused` for callers that gate.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping

UNSOURCED_MARK = "[UNSOURCED-FIGURE]"
SOURCE_KEYS = ("result_path", "raw_path", "result_json", "evidence_path", "source_path",
               "source", "result")
#: Keys whose numbers are bookkeeping, not claims.
IGNORED_KEYS = frozenset({"schema_version", "version", "n", "pairs", "count", "rank", "line",
                          "seed", "port", "pid", "timestamp", "at", "ts"})
FIGURE_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:x\b|%)")
PATH_RE = re.compile(r"[\w./\-]+\.(?:json|jsonl|md|log|txt|csv|py)(?::\d+)?")


class UnsourcedFigureRefused(RuntimeError):
    def __init__(self, figures: list[str]):
        self.figures = figures
        super().__init__(f"{len(figures)} figure(s) without a resolvable raw path: "
                         + "; ".join(figures[:8]))


def _root() -> Path:
    return Path(os.environ.get("EPYC_ROOT") or os.environ.get("EPYC_ROOT_REPO") or "/workspace")


def _default_exists(root: Path) -> Callable[[str], bool]:
    def exists(token: str) -> bool:
        path = Path(re.sub(r":\d+$", "", token))
        return (path if path.is_absolute() else root / path).is_file()
    return exists


def _resolves(value: Any, exists: Callable[[str], bool]) -> bool:
    if isinstance(value, str):
        return exists(value)
    if isinstance(value, (list, tuple)):
        return any(_resolves(v, exists) for v in value)
    return False


def _subtree_sourced(node: Any, exists: Callable[[str], bool]) -> bool:
    """Recursive key search: does any dict anywhere under `node` carry a resolvable source?"""
    if isinstance(node, Mapping):
        return any(_resolves(node.get(k), exists) for k in SOURCE_KEYS) or \
            any(_subtree_sourced(v, exists) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_subtree_sourced(v, exists) for v in node)
    return False


def find_unsourced(record: Any, *, exists: Callable[[str], bool] | None = None,
                   root: Path | None = None, _inherited: bool = False,
                   _path: str = "") -> list[str]:
    """Dotted paths of numeric leaves with no resolvable source on or above them."""
    exists = exists or _default_exists(root or _root())
    out: list[str] = []
    if isinstance(record, Mapping):
        own = _inherited or any(_resolves(record.get(k), exists) for k in SOURCE_KEYS)
        # Direct numeric leaves are sourced by a source key ANYWHERE in this record's
        # subtree (sibling evidence blocks); children of a NON-root record inherit that finding; the root passes down only
        # an own-level source, so one sourced top-level claim never launders its neighbours.
        sourced = own or _subtree_sourced(record, exists)
        for key, value in record.items():
            if key in SOURCE_KEYS:
                continue
            here = f"{_path}.{key}" if _path else str(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if not sourced and key not in IGNORED_KEYS:
                    out.append(here)
            else:
                out.extend(find_unsourced(value, exists=exists, _inherited=(sourced if _path else own), _path=here))
    elif isinstance(record, (list, tuple)):
        for i, value in enumerate(record):
            out.extend(find_unsourced(value, exists=exists, _inherited=_inherited,
                                      _path=f"{_path}[{i}]"))
    return out


def mark_text(text: str, *, exists: Callable[[str], bool] | None = None,
              root: Path | None = None, strict: bool = False,
              section_level: bool = False) -> tuple[str, list[str]]:
    """Append `UNSOURCED_MARK` after each figure on a line with no resolvable path.

    `section_level=True` resolves against the whole text instead of the line (a profile
    table cites its record once above the rows)."""
    exists = exists or _default_exists(root or _root())
    flagged: list[str] = []
    if section_level and any(exists(m.group(0)) for m in PATH_RE.finditer(text)):
        return text, flagged
    lines = []
    for line in text.split("\n"):
        if FIGURE_RE.search(line) and UNSOURCED_MARK not in line \
                and not any(exists(m.group(0)) for m in PATH_RE.finditer(line)):
            flagged.extend(m.group(0) for m in FIGURE_RE.finditer(line))
            line = FIGURE_RE.sub(lambda m: f"{m.group(0)} {UNSOURCED_MARK}", line)
        lines.append(line)
    if strict and flagged:
        raise UnsourcedFigureRefused(flagged)
    return "\n".join(lines), flagged


__all__ = ["IGNORED_KEYS", "SOURCE_KEYS", "UNSOURCED_MARK", "UnsourcedFigureRefused",
           "find_unsourced", "mark_text"]
