"""JSON-tree explode/implode, table-of-contents and pointer helpers for actor_context.py, split verbatim from actor_context.py (code-budget split)."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

#: A JSON container is split into a directory only past this size and above this depth;
#: smaller ones stay one pretty-printed file. On the run-8 bundle that is one file per
#: shared-history row (~2k chars) and per target key (<= 11.5k), 30 JSON files: a
#: first cut at 1.5k / depth 3 made 196 files of a few bytes each, one call per field.
SPLIT_MIN_CHARS = 6000
MAX_SPLIT_DEPTH = 2
TOC_MAX_CHILDREN = 14

_POINTER = re.compile(r"^<same as (\$[^>]*)>$")
_POINTER_STEP = re.compile(r"\.([^.\[\]]+)|\[(\d+)\]")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _dump(value: Any) -> str:
    """Exactly how `render_context` prints a JSON block."""
    return json.dumps(value, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# JSON tree: explode / implode
# ---------------------------------------------------------------------------
def _names(keys: Iterable[str]) -> list[str]:
    used: set[str] = set()
    out = []
    for key in keys:
        base = (_SAFE.sub("_", key).strip("._") or "_")[:60]
        name, n = base, 1
        while name.lower() in used or name.lower() == "_index":
            n += 1
            name = f"{base}~{n}"
        used.add(name.lower())
        out.append(name)
    return out


def explode(value: Any, path: Path, *, depth: int = 0) -> None:
    """Write `value` at `path` (a `.json` file, or a directory with `_index.json`
    naming its children in order). Lossless: `implode(path) == value`."""
    if not _is_dir(value, depth):
        path.with_name(path.name + ".json").write_text(_dump(value) + "\n", encoding="utf-8")
        return
    path.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        keys = sorted(value)
        names = _names(keys)
        index = {"type": "object", "entries": [[k, n] for k, n in zip(keys, names)]}
        children = [(n, value[k]) for k, n in zip(keys, names)]
    else:
        width = max(4, len(str(len(value) - 1)))
        names = [f"{i:0{width}d}" for i in range(len(value))]
        index = {"type": "array", "entries": names}
        children = list(zip(names, value))
    (path / "_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    for name, child in children:
        explode(child, path / name, depth=depth + 1)


def implode(path: Path) -> Any:
    """Inverse of `explode` (`path` without the `.json` suffix)."""
    if path.is_dir():
        index = json.loads((path / "_index.json").read_text(encoding="utf-8"))
        if index["type"] == "object":
            return {key: implode(path / name) for key, name in index["entries"]}
        return [implode(path / name) for name in index["entries"]]
    return json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))


def _k(chars: int) -> str:
    return f"{chars / 1000:.1f}k" if chars >= 1000 else str(chars)


def _is_dir(value: Any, depth: int) -> bool:
    """Whether `explode` writes `value` (at `depth`) as a directory."""
    return (isinstance(value, (dict, list)) and bool(value) and depth < MAX_SPLIT_DEPTH
            and len(_dump(value)) > SPLIT_MIN_CHARS)


def _json_toc(value: Any, label: str, *, depth: int = 0) -> list[str]:
    """Key -> size lines for every directory level `explode` writes; a key ending in
    `/` is a directory, anything else one `.json` file. Largest-first beyond the cap."""
    if not isinstance(value, (dict, list)) or depth >= MAX_SPLIT_DEPTH:
        return []
    items = (sorted(value.items()) if isinstance(value, dict)
             else [(f"[{i}]", v) for i, v in enumerate(value)])
    ranked = items if len(items) <= TOC_MAX_CHILDREN else sorted(
        items, key=lambda kv: -len(_dump(kv[1])))[:TOC_MAX_CHILDREN]
    parts = [f"{key}{'/' if _is_dir(child, depth + 1) else ''} {_k(len(_dump(child)))}"
             for key, child in ranked]
    more = len(items) - len(ranked)
    lines = [f"{'  ' * depth}- {label}: " + " · ".join(parts)
             + (f" · (+{more} more)" if more else "")]
    for key, child in ranked:
        if _is_dir(child, depth + 1):
            lines.extend(_json_toc(child, f"{label}.{key}" if not key.startswith("[")
                                   else f"{label}{key}", depth=depth + 1))
    return lines


def _resolve(root: Any, value: Any, _hops: int = 0) -> Any:
    """Follow a `<same as $.path>` pointer left by `_dedupe_subtrees`."""
    match = _POINTER.match(value) if isinstance(value, str) else None
    if match is None or _hops > 8:
        return value
    node = root
    for key, index in _POINTER_STEP.findall(match.group(1)[1:]):
        try:
            node = node[int(index)] if index else node[key]
        except (KeyError, IndexError, TypeError, ValueError):
            return value
    return _resolve(root, node, _hops + 1)


def _get(root: Any, *path: str) -> Any:
    node = root
    for key in path:
        node = _resolve(root, node)
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return _resolve(root, node)
