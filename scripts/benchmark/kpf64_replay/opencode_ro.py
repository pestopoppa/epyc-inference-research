"""Shared read-only access to the operator's opencode snapshots (operator ruling 2026-10-22).

Conversation MINING reads ``/mnt/raid0/llm/opencode-db/{epyc.db,host-daniele.db}`` (symlink to
``/mnt/raid0/llm/opencode/readonly/``): hourly consistent ``sqlite3 .backup`` copies, root:root 0444,
``journal_mode=DELETE``, full cumulative history, at most ~1 h stale. Content lives in ``session``,
``message`` and ``part`` (JSON ``data`` column); the ``event`` change-feed is pruned by a reaper and is
never read here.

The LIVE stores (``~/.local/share/opencode/opencode.db``, ``/mnt/raid0/llm/opencode/<src>/opencode.db``)
are writable and are REFUSED: ``open_ro`` raises ``LiveOpencodeStoreError``.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote

OPENCODE_RO_DIR = Path("/mnt/raid0/llm/opencode-db")
OPENCODE_RO_DBS: dict[str, str] = {
    "epyc": str(OPENCODE_RO_DIR / "epyc.db"),
    "host": str(OPENCODE_RO_DIR / "host-daniele.db"),
}
_LIVE_ROOTS = (Path(os.path.expanduser("~/.local/share/opencode")),
               Path("/home/node/.local/share/opencode"),
               Path("/mnt/raid0/llm/opencode"))
_RO_REAL = Path("/mnt/raid0/llm/opencode/readonly")


class LiveOpencodeStoreError(ValueError):
    """A mining reader was pointed at a LIVE (writable) opencode store."""


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_live_opencode_path(path: "Path | str") -> bool:
    """True for anything under ~/.local/share/opencode or /mnt/raid0/llm/opencode/<src>/ except readonly/.

    Lexical AND symlink-resolved paths are both checked (a symlink into a live store is caught;
    ``/mnt/raid0/llm/opencode-db`` -> ``.../opencode/readonly`` is allowed).
    """
    raw = Path(os.path.expanduser(str(path)))
    for cand in (Path(os.path.abspath(raw)), raw.resolve()):
        if _under(cand, _RO_REAL) or _under(cand, OPENCODE_RO_DIR):
            continue
        if any(_under(cand, root) for root in _LIVE_ROOTS):
            return True
    return False


def refuse_live_opencode_path(path: "Path | str") -> str:
    if is_live_opencode_path(path):
        raise LiveOpencodeStoreError(
            f"refusing to mine LIVE opencode store {path}: it is writable and could be corrupted. "
            f"Use the read-only snapshots {', '.join(OPENCODE_RO_DBS.values())} "
            f"(refreshed hourly; operator ruling 2026-10-22).")
    return str(path)


def open_ro(path: "Path | str", *, timeout: float = 5, retries: int = 1, retry_delay: float = 1.0,
            _connect: Callable[..., sqlite3.Connection] = sqlite3.connect) -> sqlite3.Connection:
    """Open a snapshot ``?mode=ro`` + ``PRAGMA query_only=1``; retry once (the copy is atomically replaced hourly)."""
    p = refuse_live_opencode_path(path)
    uri = f"file:{quote(p, safe='/')}?mode=ro"
    last: Exception | None = None
    for attempt in range(retries + 1):
        conn = None
        try:
            conn = _connect(uri, uri=True, timeout=timeout)
            conn.execute("PRAGMA query_only=1")
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            return conn
        except sqlite3.Error as exc:
            last = exc
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            if attempt < retries:
                time.sleep(retry_delay)
    assert last is not None
    raise last


def source_for(path: "Path | str") -> str:
    """epyc | host for the known snapshots, else ``unknown`` (never guessed)."""
    try:
        rp = Path(path).resolve()
    except OSError:
        rp = Path(path)
    for tag, known in OPENCODE_RO_DBS.items():
        if str(path) == known or rp == Path(known).resolve():
            return tag
    return "unknown"


def resolve_dbs(explicit: "Iterable[str] | str | None") -> list[tuple[str, str]]:
    """[(source, path)]: the explicit path(s) (live ones refused) or, by default, BOTH snapshots (absent files skipped)."""
    if isinstance(explicit, str):
        explicit = [explicit]
    if explicit:
        return [(source_for(p), refuse_live_opencode_path(p)) for p in explicit]
    return [(tag, p) for tag, p in OPENCODE_RO_DBS.items() if Path(p).exists()]
