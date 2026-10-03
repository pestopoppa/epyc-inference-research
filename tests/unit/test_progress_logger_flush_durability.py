"""Progress rows must not sit in a per-worker buffer until a clean shutdown.

The API runs six uvicorn workers, each with its own ``ProgressLogger`` buffer
(``buffer_size=10``). At 20-40 tasks a day a worker's rows could stay in memory for
days and were lost outright when the process ended without the lifespan flush
(run 1767532, 2026-10-01: six workers started, none finished cleanly; tasks
chat-4b4d0684 and chat-c79dcb45 have ``task_started`` and no terminal row).
"""

from __future__ import annotations

import atexit
import json
from pathlib import Path

from orchestration.repl_memory import progress_logger as pl
from orchestration.repl_memory.progress_logger import ProgressLogger


def _rows(log_dir: Path) -> list[dict]:
    out: list[dict] = []
    for f in sorted(log_dir.glob("*.jsonl")):
        out.extend(json.loads(line) for line in f.read_text().splitlines() if line.strip())
    return out


def test_terminal_task_row_is_on_disk_when_the_call_returns(tmp_path):
    logger = ProgressLogger(log_dir=tmp_path, buffer_size=10, max_buffer_age_s=None)
    logger.log_task_started(task_id="chat-1", task_ir={"objective": "x"}, routing_decision=["frontdoor"], routing_strategy="rules")
    assert _rows(tmp_path) == []  # an ordinary row still batches
    logger.log_task_completed(task_id="chat-1", success=True, completion_meta={"prompt_eval_ms": 5.0})
    events = [r["event_type"] for r in _rows(tmp_path)]
    # The completed row AND everything buffered ahead of it, in order.
    assert events[0] == "task_started" and events[-1] == "task_completed"


def test_failed_task_row_is_durable_too(tmp_path):
    logger = ProgressLogger(log_dir=tmp_path, buffer_size=10, max_buffer_age_s=None)
    logger.log_task_completed(task_id="chat-2", success=False, details="boom")
    assert [r["event_type"] for r in _rows(tmp_path)] == ["task_failed"]


def test_aged_buffer_flushes_on_next_log(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(pl.time, "monotonic", lambda: clock[0])
    logger = ProgressLogger(log_dir=tmp_path, buffer_size=10, max_buffer_age_s=30.0)
    logger.log_memory_update(task_id="t", memory_id="m", old_q=0.1, new_q=0.2, reward=1.0)
    assert _rows(tmp_path) == []
    clock[0] += 29.0
    logger.log_memory_update(task_id="t", memory_id="m", old_q=0.2, new_q=0.3, reward=1.0)
    assert _rows(tmp_path) == []
    clock[0] += 2.0  # oldest buffered entry is now 31 s old
    logger.log_memory_update(task_id="t", memory_id="m", old_q=0.3, new_q=0.4, reward=1.0)
    assert len(_rows(tmp_path)) == 3
    # The age clock restarts with the next buffered entry, not the old one.
    clock[0] += 5.0
    logger.log_memory_update(task_id="t", memory_id="m", old_q=0.4, new_q=0.5, reward=1.0)
    assert len(_rows(tmp_path)) == 3


def test_age_flush_can_be_disabled(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(pl.time, "monotonic", lambda: clock[0])
    logger = ProgressLogger(log_dir=tmp_path, buffer_size=10, max_buffer_age_s=None)
    logger.log_memory_update(task_id="t", memory_id="m", old_q=0.1, new_q=0.2, reward=1.0)
    clock[0] += 10_000.0
    logger.log_memory_update(task_id="t", memory_id="m", old_q=0.2, new_q=0.3, reward=1.0)
    assert _rows(tmp_path) == []


def test_exit_hook_flushes_the_tail(tmp_path, monkeypatch):
    registered: list = []
    monkeypatch.setattr(atexit, "register", lambda fn, *args: registered.append((fn, args)))
    logger = ProgressLogger(log_dir=tmp_path, buffer_size=10, max_buffer_age_s=None)
    logger.log_memory_update(task_id="t", memory_id="m", old_q=0.1, new_q=0.2, reward=1.0)
    assert _rows(tmp_path) == []
    fn, args = registered[-1]
    fn(*args)
    assert len(_rows(tmp_path)) == 1


def test_exit_hook_tolerates_a_collected_logger():
    import weakref

    class _Gone:
        pass

    ref = weakref.ref(_Gone())  # referent is already collected
    pl._flush_at_exit(ref)  # must not raise
