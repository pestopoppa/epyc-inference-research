"""DP-A (b): the private corpus repo as the custody root for private STT strata.

The ONLY git work tree a private path may sit in is the local clone of the private repo
(``data/stt-cs4/private_custody.json``). The clone qualifies only when (a) the path lies
under it, (b) the nearest enclosing git tree IS the clone root (no nested/foreign tree) and
(c) the clone's ``.git/config`` names the private repo as a remote. Every other git tree,
public repos included, stays refused. Stdlib only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]
CONFIG = Path(__file__).resolve().parents[3] / "data" / "stt-cs4" / "private_custody.json"
ENV_ROOT = "STT_PRIVATE_CORPUS_ROOT"   # relocates the clone (tests, other hosts); remote check still applies


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def custody_root() -> Path:
    return Path(os.environ.get(ENV_ROOT) or load_config()["local_clone"]).resolve(strict=False)


def _remote_matches(root: Path) -> bool:
    cfg = root / ".git" / "config"          # a .git FILE (worktree/submodule) is not an accepted clone
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError:
        return False
    frag = load_config()["remote_name_fragment"]
    return any(ln.strip().startswith("url") and frag in ln for ln in text.splitlines())


def _nearest_tree(p: Path) -> Optional[Path]:
    for d in (p, *p.parents):
        if os.path.lexists(d / ".git"):
            return d
    return None


def is_private_custody_path(path: PathLike) -> bool:
    """True iff ``path`` is inside the verified private-repo clone (and no other git tree)."""
    root = custody_root()
    for base in (Path(os.path.abspath(Path(path).expanduser())), Path(path).expanduser().resolve(strict=False)):
        tree = _nearest_tree(base)
        if tree is None or tree.resolve(strict=False) != root:
            return False
        try:
            base.resolve(strict=False).relative_to(root)
        except ValueError:
            return False
    return _remote_matches(root)
