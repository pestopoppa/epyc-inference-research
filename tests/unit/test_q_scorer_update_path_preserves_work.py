"""RTG-02 Finding B: the UPDATE write path dropped `assigned_role` and `work`.

`EpisodicStore.store()` is the ONLY site that ever wrote the TR-3.2 tri-role and
the M-11a2b `work` payload. Every update branch in `QScorer` —

  * the pre-linked `routing_decision.memory_id` branch, and
  * the `ORCHESTRATOR_Q_TD_WRITE` find-or-update branch
    (`q_scorer._update_routing_memory`, the branch production actually takes,
    since `orchestrator_stack.py` sets `ORCHESTRATOR_Q_TD_WRITE=1`)

— called `update_q_value()` and returned before reaching `store()`. The update
branch fires ~11x more often than create (SUM(update_count)=668,070 vs 59,337
rows), so both fields were effectively never populated: the values were computed
off the very entries already passed in, then dropped on the floor.

These tests assert on the SQLite row after an UPDATE, with
`ORCHESTRATOR_Q_TD_WRITE=1` set before the scorer is constructed.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from orchestration.repl_memory.episodic_store import EpisodicStore
from orchestration.repl_memory.progress_logger import ProgressLogger, ProgressReader
from orchestration.repl_memory.q_scorer import QScorer


class _StubEmbedder:
    """Deterministic non-degenerate unit vectors keyed by objective, so two runs
    of the same objective land on the same row (the find-or-update precondition)."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vec(self, seed_text: str) -> np.ndarray:
        rng = np.random.default_rng(
            int.from_bytes(seed_text.encode()[:4].ljust(4, b"\0"), "little")
        )
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_task_ir(self, task_ir: dict) -> np.ndarray:
        return self._vec(str(task_ir.get("objective", "")))

    def embed_failure_context(self, failure_context: dict) -> np.ndarray:
        return self._vec(str(failure_context.get("reason", "")))


@pytest.fixture
def wiring(tmp_path, monkeypatch):
    """Real store + progress logger/reader + QScorer, with Q_TD_WRITE ON."""
    monkeypatch.setenv("ORCHESTRATOR_Q_TD_WRITE", "1")
    monkeypatch.setenv("ORCHESTRATOR_Q_TD_MATCH_K", "10")
    store = EpisodicStore(db_path=tmp_path / "sessions", use_faiss=True)
    log_dir = tmp_path / "progress"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(log_dir=log_dir, buffer_size=1)
    scorer = QScorer(
        store=store,
        embedder=_StubEmbedder(store.embedding_dim),
        logger=logger,
        reader=ProgressReader(log_dir=log_dir),
    )
    yield scorer, store, logger
    store.close()


OBJECTIVE = "summarise the quarterly kernel report"


def _run_task(
    logger: ProgressLogger,
    scorer: QScorer,
    task_id: str,
    *,
    assigned_role: str | None = None,
    work: dict | None = None,
) -> dict:
    routing_meta = {"decision_source": "rules"}
    if assigned_role is not None:
        routing_meta["assigned_role"] = assigned_role
    logger.log_task_started(
        task_id=task_id,
        task_ir={"task_type": "chat", "objective": OBJECTIVE, "priority": "normal"},
        routing_decision=["frontdoor"],
        routing_strategy="rules",
        routing_meta=routing_meta,
    )
    logger.log_task_completed(
        task_id=task_id,
        success=True,
        completion_meta={"work": work} if work else None,
    )
    logger.flush()
    # ProgressReader memoizes read_recent for _RECENT_CACHE_TTL seconds, so a
    # second task logged within the window would be invisible to the scorer.
    scorer.reader._recent_cache = None
    return scorer._score_task(task_id)


def _rows(store: EpisodicStore) -> list[tuple[str | None, dict]]:
    store.flush()
    conn = sqlite3.connect(store.sqlite_path)
    try:
        raw = conn.execute(
            "SELECT assigned_role, context, update_count FROM memories "
            "WHERE action_type = 'routing'"
        ).fetchall()
    finally:
        conn.close()
    return [(role, json.loads(ctx or "{}"), count) for role, ctx, count in raw]


def test_update_path_backfills_assigned_role_and_work(wiring):
    """First observation carries neither field; the UPDATE that follows must
    carry both onto the SAME row instead of discarding them."""
    scorer, store, logger = wiring

    first = _run_task(logger, scorer, "task-1")
    assert first["memories_created"] == 1, first

    rows = _rows(store)
    assert len(rows) == 1
    assert rows[0][0] is None, "precondition: row starts with no assigned_role"
    assert "work" not in rows[0][1], "precondition: row starts with no work"

    second = _run_task(
        logger,
        scorer,
        "task-2",
        assigned_role="thinker",
        work={"answer": "the kernel regressed 3%", "reasoning": "compared v8 to v9"},
    )
    assert second["memories_updated"] == 1, second
    assert second["memories_created"] == 0, "must reuse the row, not append"

    rows = _rows(store)
    assert len(rows) == 1, "find-or-update must not create a second row"
    role, context, _ = rows[0]
    assert role == "thinker", "UPDATE path dropped assigned_role"
    assert context.get("work"), "UPDATE path dropped the work payload"
    assert context["work"]["answer"] == "the kernel regressed 3%"
    assert context["work"]["reasoning"] == "compared v8 to v9"


def test_update_path_merges_and_never_overwrites(wiring):
    """MERGE, not overwrite: an existing role and existing work keys survive, and
    a later observation may only FILL gaps."""
    scorer, store, logger = wiring

    _run_task(
        logger,
        scorer,
        "task-1",
        assigned_role="worker",
        work={"answer": "first answer"},
    )
    rows = _rows(store)
    assert rows[0][0] == "worker" and rows[0][1]["work"]["answer"] == "first answer"

    # A later observation disagrees on the role and brings a different answer
    # plus a NEW field. Neither established value may be clobbered.
    second = _run_task(
        logger,
        scorer,
        "task-2",
        assigned_role="verifier",
        work={"answer": "second answer", "reasoning": "newly captured"},
    )
    assert second["memories_updated"] == 1

    role, context, _ = _rows(store)[0]
    assert role == "worker", "first observation's role must not be overwritten"
    assert context["work"]["answer"] == "first answer", "captured work overwritten"
    assert context["work"]["reasoning"] == "newly captured", "gap not filled"


def test_update_path_preserves_unrelated_context_keys(wiring):
    """The merge rewrites `context` JSON, so every other key must survive."""
    scorer, store, logger = wiring
    _run_task(logger, scorer, "task-1")
    before = _rows(store)[0][1]

    _run_task(logger, scorer, "task-2", assigned_role="thinker", work={"answer": "a"})
    after = _rows(store)[0][1]

    for key, value in before.items():
        assert after[key] == value, f"context key {key!r} was mutated by the merge"


def test_update_still_bumps_q_value_and_update_count(wiring):
    """The backfill must ride ALONGSIDE the Q-value update, not replace it."""
    scorer, store, logger = wiring
    _run_task(logger, scorer, "task-1")
    _run_task(logger, scorer, "task-2", assigned_role="thinker", work={"answer": "a"})
    assert _rows(store)[0][2] == 1, "update_count must still increment"


def test_merge_row_metadata_is_a_noop_without_inputs(tmp_path):
    """An update with nothing to carry must not touch the row at all."""
    store = EpisodicStore(db_path=tmp_path / "sessions", use_faiss=True)
    try:
        embedder = _StubEmbedder(store.embedding_dim)
        memory_id = store.store(
            embedding=embedder.embed_task_ir({"objective": OBJECTIVE}),
            action="frontdoor",
            action_type="routing",
            context={"objective": OBJECTIVE},
            outcome="success",
            initial_q=0.5,
        )
        store.flush()
        assert store.merge_row_metadata(memory_id) is False
        assert store.merge_row_metadata(memory_id, assigned_role="  ", work={}) is False
        assert store.merge_row_metadata("no-such-id", assigned_role="worker") is False
    finally:
        store.close()
