"""Freeze the MemRL store the offline gate reads (design §3.3). BUILT, run by the main session.

Copies from ``<orch checkout>/orchestration/repl_memory/``:

* ``sessions/episodic.db`` -- via the sqlite online-backup API from a ``mode=ro`` connection
  (a consistent copy even if the API writes meanwhile; the live file is never opened
  read-write);
* ``sessions/embeddings.faiss`` + ``sessions/id_map.npy`` -- under a SHARED flock on the store's
  own ``.episodic_faiss.lock`` (every store save/reload takes it exclusively, so the pair cannot
  change mid-copy); the source is re-hashed after the copy and must match;
* ``kuzu_db/{failure,hypothesis}_graph`` (+ ``.wal``) -- the GraphEnhancedRetriever's graphs,
  when present (the API builds that retriever while ``specialist_routing`` is on).

``id_map.npy`` is mandatory: without it the store silently starts an EMPTY index. Writes
``snapshot_manifest.json`` (sha256 of every file, sizes, source mtimes, the retrieval config
``semantic_k/min_similarity/min_q_value/q_weight/top_n`` and ``chat.review_low_q_threshold`` as the
orchestrator code at ``--code-root`` resolves them, the code commit, the retriever kind). Refuses
to overwrite an existing snapshot (``--force`` replaces it; a run pinned to the old one then
refuses to resume). Run it right before stage 1.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bridge import InstrumentFault, git_state, retrieval_config_dict

SNAPSHOT_DIR = Path("/mnt/raid0/llm/tmp/ri18/store-snapshot")
MANIFEST = "snapshot_manifest.json"
SCHEMA = "ri18-store-snapshot/v1"
REQUIRED = ("episodic.db", "embeddings.faiss", "id_map.npy")
GRAPH_FILES = ("failure_graph", "failure_graph.wal", "hypothesis_graph", "hypothesis_graph.wal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_sqlite(src: Path, dst: Path) -> None:
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    check = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
    try:
        if check.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise InstrumentFault(f"{dst}: quick_check failed")
    finally:
        check.close()


def take_snapshot(source_root: Path, dest: Path, *, code_root: str, retrieval: dict[str, Any],
                  threshold: float, retriever_kind: str, force: bool = False) -> dict[str, Any]:
    sessions = source_root / "orchestration/repl_memory/sessions"
    kuzu = source_root / "orchestration/repl_memory/kuzu_db"
    for name in REQUIRED:
        if not (sessions / name).is_file():
            raise InstrumentFault(f"{sessions / name} is missing; a snapshot without it is wrong")
    if (dest / MANIFEST).exists() and not force:
        raise InstrumentFault(f"{dest} already holds a snapshot; pass --force to replace it")
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files: dict[str, Any] = {}
    _copy_sqlite(sessions / "episodic.db", tmp / "episodic.db")
    lock_path = sessions / ".episodic_faiss.lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            for name in ("embeddings.faiss", "id_map.npy"):
                shutil.copy2(sessions / name, tmp / name)
                if sha256_file(sessions / name) != sha256_file(tmp / name):
                    raise InstrumentFault(f"{name} changed during the copy")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    graphs = []
    for name in GRAPH_FILES:
        if (kuzu / name).is_file():
            shutil.copy2(kuzu / name, tmp / name)
            graphs.append(name)
    if retriever_kind == "graph" and not {"failure_graph", "hypothesis_graph"} <= set(graphs):
        raise InstrumentFault(f"retriever 'graph' needs the kuzu graphs; found {graphs} in {kuzu}")
    for path in sorted(tmp.iterdir()):
        src = (sessions / path.name) if (sessions / path.name).exists() else (kuzu / path.name)
        files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size,
                            "source": str(src),
                            "source_mtime": datetime.fromtimestamp(
                                src.stat().st_mtime, timezone.utc).isoformat()}
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "code_root": code_root,
        "code_root_git": git_state(code_root),
        "files": files,
        "retrieval_config": retrieval,
        "review_low_q_threshold": threshold,
        "retriever": retriever_kind,
        "copy_method": {"episodic.db": "sqlite backup API from mode=ro",
                        "faiss+id_map": "copy under LOCK_SH on .episodic_faiss.lock, re-hashed",
                        "graphs": "plain copy"},
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    return manifest


def snapshot_pin(dest: Path) -> dict[str, Any]:
    """What a run manifest pins: the manifest's own sha plus the per-file shas and config."""
    path = dest / MANIFEST
    doc = json.loads(path.read_text())
    return {"dir": str(dest), "manifest_sha256": sha256_file(path),
            "files": {k: {"sha256": v["sha256"]} for k, v in doc["files"].items()},
            "retrieval_config": doc["retrieval_config"],
            "review_low_q_threshold": doc["review_low_q_threshold"],
            "retriever": doc["retriever"]}


def verify_snapshot(dest: Path, pin: dict[str, Any]) -> None:
    """Refuse a snapshot whose files moved since the run pinned it."""
    if not pin or not pin.get("files"):
        raise InstrumentFault("run manifest pins no snapshot")
    if sha256_file(dest / MANIFEST) != pin["manifest_sha256"]:
        raise InstrumentFault(f"{dest / MANIFEST} changed since the run pinned it")
    for name, meta in pin["files"].items():
        if sha256_file(dest / name) != meta["sha256"]:
            raise InstrumentFault(f"snapshot file {name} drifted from its pinned sha256")


def resolve_config(code_root: str) -> tuple[dict[str, Any], float]:
    """Retrieval config + review threshold as the orchestrator at ``code_root`` resolves them."""
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from orchestration.repl_memory.retriever import RetrievalConfig
    from src.config import get_config

    return retrieval_config_dict(RetrievalConfig()), float(get_config().chat.review_low_q_threshold)
