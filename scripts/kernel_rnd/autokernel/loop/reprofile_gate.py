"""AK-EVAL-KEEPTIME: skip the per-keep reprofile when the keep cannot have moved it.

A keep's reprofile re-derives the hotspots of the new champion. When the kept diff
touches none of the profiled hotspot symbols (no hunk header and no changed line
names one), the retained observation still describes where the time goes, and
the reprofile is skipped with a logged reason. Conservative by construction: any
overlap, an empty/unreadable diff, or no retained observation means reprofile.
"""
from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def changed_identifiers(diff_text: str) -> set[str]:
    """Identifiers in hunk headers and in added/removed lines of a unified diff."""
    found: set[str] = set()
    for line in (diff_text or "").splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("@@"):
            found.update(_IDENT.findall(line.split("@@", 2)[-1]))
        elif line.startswith(("+", "-")):
            found.update(_IDENT.findall(line[1:]))
    return found


def hotspot_name(row) -> str | None:
    """`ggml_vec_dot_q4_K_q8_K` from a row dict / "dso:sym(args)" / C++ qualified name."""
    symbol = row.get("symbol") if isinstance(row, dict) else row
    if not isinstance(symbol, str) or not symbol:
        return None
    if not isinstance(row, dict):
        _dso, sep, rest = symbol.partition(":")
        if sep and not rest.startswith(":"):
            symbol = rest
    symbol = symbol.split("(", 1)[0].split("<", 1)[0].split(".", 1)[0]
    name = symbol.rsplit("::", 1)[-1].strip()
    return name if _IDENT.fullmatch(name) else None


def skip_reason(diff_text: str, hotspots) -> str | None:
    """None -> reprofile. A string -> the logged reason the retained profile stands."""
    names = {name for name in map(hotspot_name, hotspots or ()) if name}
    if not names:
        return None
    changed = changed_identifiers(diff_text)
    if not changed or names & changed:
        return None
    return (f"kept diff touches none of the {len(names)} profiled hotspot symbols "
            f"({len(changed)} changed identifiers); retained observation reused")
