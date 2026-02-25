"""Checkpoint I/O, seen-question tracking, and prompt hashing.

Provides atomic file append via fcntl locking. Deduplicates the previously
copy-pasted checkpoint writers (append_checkpoint, _checkpoint_3way) into a
single ``checkpoint_result`` backed by ``_atomic_append``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from seeding_types import ComparativeResult, EVAL_DIR, RoleResult, SEEN_FILE

__all__ = [
    "_atomic_append",
    "_prompt_hash",
    "checkpoint_result",
    "load_checkpoint",
    "load_seen_questions",
    "record_seen",
]


# ── Low-level helpers ────────────────────────────────────────────────


def _ensure_eval_dir() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _atomic_append(path: "str | os.PathLike[str]", line: str) -> None:
    """Append *line* to *path* with fcntl exclusive lock + fsync."""
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ── Checkpoint read/write ────────────────────────────────────────────


def load_checkpoint(session_id: str) -> list[ComparativeResult]:
    """Load completed results from a session's JSONL checkpoint."""
    path = EVAL_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    results: list[ComparativeResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                role_results: dict[str, RoleResult] = {}
                for k, v in data.get("role_results", {}).items():
                    role_results[k] = RoleResult(**v)
                results.append(ComparativeResult(
                    suite=data["suite"],
                    question_id=data["question_id"],
                    prompt=data.get("prompt", ""),
                    expected=data.get("expected", ""),
                    dataset_source=data.get("dataset_source", "yaml"),
                    prompt_hash=data.get("prompt_hash", ""),
                    timestamp=data.get("timestamp", ""),
                    role_results=role_results,
                    rewards=data.get("rewards", {}),
                    rewards_injected=data.get("rewards_injected", 0),
                ))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return results


def checkpoint_result(session_id: str, result: Any) -> None:
    """Append one result (ComparativeResult or ThreeWayResult) to the session JSONL.

    Accepts any dataclass; serialises via ``dataclasses.asdict()``.
    This replaces the previously duplicated ``append_checkpoint`` and
    ``_checkpoint_3way`` functions.
    """
    _ensure_eval_dir()
    path = EVAL_DIR / f"{session_id}.jsonl"
    line = json.dumps(asdict(result), ensure_ascii=False)
    _atomic_append(path, line)


# Legacy aliases — keep the old function names working for callers.
append_checkpoint = checkpoint_result
_checkpoint_3way = checkpoint_result


# ── Seen-question tracking ───────────────────────────────────────────


def load_seen_questions() -> set[str]:
    """Load all prompt_ids ever evaluated across all sessions."""
    seen: set[str] = set()
    if not EVAL_DIR.exists():
        return seen

    for path in EVAL_DIR.glob("*.jsonl"):
        if path.name == "seen_questions.jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pid = data.get("prompt_id", "")
                        if pid:
                            seen.add(pid)
                    except json.JSONDecodeError:
                        continue
        else:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pid = data.get("question_id", "")
                        if pid:
                            seen.add(pid)
                    except json.JSONDecodeError:
                        continue

    return seen


def record_seen(prompt_id: str, suite: str, session_id: str) -> None:
    """Append to the global seen questions file."""
    _ensure_eval_dir()
    entry = {
        "prompt_id": prompt_id,
        "suite": suite,
        "session": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_append(SEEN_FILE, json.dumps(entry))
