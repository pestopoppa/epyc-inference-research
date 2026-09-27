"""ProgressLogger.log_durable and flush serialization (HS-19a S6 follow-up).

The HS-19a live acceptance failed S6 because the parent->child lineage row sat in a
uvicorn worker's ``buffer_size=10`` batch buffer: it reached disk only after nine more
rows in that worker, or at shutdown. ``log_durable`` writes such rows through while
leaving ordinary ``log`` batching exactly as it was.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from orchestration.repl_memory.progress_logger import (
    EventType,
    ProgressEntry,
    ProgressLogger,
)
from src.api.routes.v1_subagent_link import (
    SESSION_LOG_KIND,
    SubagentLink,
    log_subagent_link,
)


def _rows(log_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def _entry(task_id: str) -> ProgressEntry:
    return ProgressEntry(event_type=EventType.TASK_STARTED, task_id=task_id, data={})


def test_plain_log_still_batches(tmp_path):
    pl = ProgressLogger(log_dir=tmp_path, buffer_size=10)
    for i in range(9):
        pl.log(_entry(f"t{i}"))
    assert _rows(tmp_path) == []
    pl.log(_entry("t9"))
    assert [r["task_id"] for r in _rows(tmp_path)] == [f"t{i}" for i in range(10)]


def test_log_durable_is_on_disk_when_it_returns_and_keeps_order(tmp_path):
    pl = ProgressLogger(log_dir=tmp_path, buffer_size=10)
    pl.log(_entry("before"))
    pl.log_durable(_entry("durable"))
    assert [r["task_id"] for r in _rows(tmp_path)] == ["before", "durable"]
    assert pl._buffer == []
    pl.log(_entry("after"))  # batching resumes for ordinary rows
    assert [r["task_id"] for r in _rows(tmp_path)] == ["before", "durable"]


def test_log_durable_is_a_noop_when_disabled(tmp_path):
    pl = ProgressLogger(log_dir=tmp_path, buffer_size=10)
    pl._disabled = True
    pl.log_durable(_entry("x"))
    assert _rows(tmp_path) == [] and pl._buffer == []


def test_subagent_link_row_is_durable_with_the_real_logger(tmp_path):
    """The S6 regression: one link row, no later traffic, no shutdown -> still on disk."""
    pl = ProgressLogger(log_dir=tmp_path, buffer_size=10)
    link = SubagentLink(
        session_id="ses_child",
        session_id_source="body",
        session_id_mismatch=False,
        parent_session_id="ses_parent",
        parent_session_id_source="header",
        parent_session_id_mismatch=False,
        agent_name="general",
        depth=1,
        depth_basis="observed",
        newly_linked=True,
    )
    assert log_subagent_link(pl, link, chat_id="chatcmpl-1", user_id="u")
    (row,) = _rows(tmp_path)
    assert row["event_type"] == "session_created"
    assert row["data"]["kind"] == SESSION_LOG_KIND
    assert (row["data"]["session_id"], row["data"]["parent_session_id"]) == (
        "ses_child",
        "ses_parent",
    )


def test_concurrent_log_and_flush_neither_lose_nor_duplicate(tmp_path):
    """flush runs from worker threads (Q-scoring) while the event loop logs."""
    pl = ProgressLogger(log_dir=tmp_path, buffer_size=7)
    n_writers, per_writer = 4, 300
    stop = threading.Event()

    def writer(w: int) -> None:
        for i in range(per_writer):
            if i % 50 == 0:
                pl.log_durable(_entry(f"w{w}-{i}"))
            else:
                pl.log(_entry(f"w{w}-{i}"))

    def flusher() -> None:
        while not stop.is_set():
            pl.flush()

    flushers = [threading.Thread(target=flusher) for _ in range(2)]
    writers = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
    for t in flushers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in flushers:
        t.join()
    pl.flush()

    ids = [r["task_id"] for r in _rows(tmp_path)]
    expected = {f"w{w}-{i}" for w in range(n_writers) for i in range(per_writer)}
    assert len(ids) == len(expected)
    assert set(ids) == expected
