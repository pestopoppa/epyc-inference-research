#!/usr/bin/env python3
"""AK-TOOL-CHAMPION-CONTEXT (b): rulings in force + per-port e2e status for the planner.

Data lives in `champion_rulings.json` (each entry cites a source path:line and commit);
this module only renders it. An entry whose source does not resolve is rendered with the
unsourced mark rather than silently trusted. The stale-anchor hard check is
`champion.verify_anchor_current`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import figure_lint

DATA = Path(__file__).with_name("champion_rulings.json")
HEADING = "Champion and rulings in force (data: loop/champion_rulings.json)"


def load(path: Path = DATA) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(entry: dict[str, Any], exists: Callable[[str], bool]) -> str:
    src = str(entry.get("source") or "")
    tag = f"src: {src}" if src and exists(src) else f"src: {src or 'none'} {figure_lint.UNSOURCED_MARK}"
    return f"[{tag}{', commit ' + entry['commit'] if entry.get('commit') else ''}]"


def render(data: dict[str, Any] | None = None, *,
           exists: Callable[[str], bool] | None = None) -> list[str]:
    data = data if data is not None else load()
    exists = exists or figure_lint._default_exists(figure_lint._root())
    champ = data.get("champion") or {}
    lines = [HEADING,
             f"- champion: {champ.get('name')} {champ.get('commit')} {_src(champ, exists)}",
             "- rulings in force:"]
    lines += [f"  - {r['id']}: {r['text']} {_src(r, exists)}" for r in data.get("rulings", [])]
    lines.append("- per-port e2e status:")
    lines += [f"  - {p['port']}: {p['e2e_status']} {_src(p, exists)}" for p in data.get("ports", [])]
    return lines


__all__ = ["DATA", "HEADING", "load", "render"]
