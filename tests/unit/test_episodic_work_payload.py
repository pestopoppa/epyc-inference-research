"""M-11a2b — the `work` payload must reach `memories.context` in SQLite.

The four work fields (answer / tool_calls / repl_steps / reasoning) were declared
by the record contract on 2026-07-27 and then never passed by anything: measured
2026-08-03, **0 of 59,337 rows** carried `work`. Same class of gap as `model_id`
(NULL on all 59,337) and `assigned_role` (0 of 59,337).

These tests assert against the DATABASE, not against the record object — a
dataclass that holds the answer proves nothing if the value never crosses the
`store()` boundary. Every assertion below re-reads the row through SQLite.

Also pinned here: the capture policy (redaction reuse + size bounds), because a
work payload is the first content in this store that can carry a credential or
an unbounded blob.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pytest

from orchestration.repl_memory.episodic_store import EpisodicStore
from orchestration.repl_memory.memory_record import (
    WORK_ITEM_MAX_CHARS,
    WORK_MAX_ITEMS,
    WORK_TEXT_MAX_CHARS,
    build_memory_record,
    build_work_payload,
    extract_work,
)
from orchestration.repl_memory.progress_logger import EventType, ProgressEntry
from orchestration.repl_memory.q_scorer import QScorer, ScoringConfig


class _FakeEmbedder:
    """Deterministic embeddings keyed by objective text."""

    def embed_task_ir(self, context):
        obj = (context or {}).get("objective") or ""
        seed = int.from_bytes(hashlib.sha256(obj.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(1024).astype(np.float32)

    def embed_failure_context(self, context):
        obj = (context or {}).get("reason") or ""
        seed = int.from_bytes(hashlib.sha256(obj.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(1024).astype(np.float32)


class _FakeLogger:
    def __init__(self):
        self.memory_updates = []
        self.logged = []

    def log_memory_update(self, memory_id, old_q, new_q, reward, task_id):
        self.memory_updates.append((memory_id, old_q, new_q, reward, task_id))

    def log(self, entry):
        self.logged.append(entry)


@pytest.fixture(autouse=True)
def _isolate_kernel_path_config(tmp_path, monkeypatch):
    """Keep offline memory tests out of the runner's production kernel store.

    Configuration normally derives these three defaults through
    ``kernel_paths``. The selected tests only exercise metadata, fake embeddings,
    and temporary SQLite/FAISS storage, so supply per-test placeholder paths at
    the existing configuration boundary before any constructor reads them.
    """
    paths = tmp_path / "unused-kernel-paths"
    expected = {
        "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN": str(paths / "cpu"),
        "ORCHESTRATOR_PATHS_LLAMA_MTMD": str(paths / "mtmd"),
        "ORCHESTRATOR_PATHS_LLAMA_SERVER": str(paths / "llama-server"),
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    from src.config import get_config, reset_config
    from src.features import reset_features

    reset_config()
    reset_features()
    config = get_config()
    assert str(config.paths.llama_cpp_bin) == expected["ORCHESTRATOR_PATHS_LLAMA_CPP_BIN"]
    assert str(config.vision.llama_mtmd_cli) == expected["ORCHESTRATOR_PATHS_LLAMA_MTMD"]
    assert str(config.worker_pool.llama_server_path) == expected["ORCHESTRATOR_PATHS_LLAMA_SERVER"]
    yield
    reset_features()
    reset_config()


@pytest.fixture
def scorer(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_Q_TD_WRITE", "0")
    monkeypatch.setenv("ORCHESTRATOR_Q_TD_MATCH_K", "10")
    store = EpisodicStore(db_path=tmp_path / "sessions", use_faiss=True)
    sc = QScorer(
        store=store,
        embedder=_FakeEmbedder(),
        logger=_FakeLogger(),
        reader=None,
        config=ScoringConfig(learning_rate=0.1, temporal_decay_rate=None),
    )
    yield sc
    store.close()


def _contexts_from_sqlite(store) -> list[dict]:
    """Re-read every stored context straight out of the DB file."""
    conn = sqlite3.connect(store.sqlite_path)
    try:
        rows = conn.execute("SELECT context FROM memories").fetchall()
    finally:
        conn.close()
    return [json.loads(r[0]) for r in rows]


def _task_started(objective="solve X", task_type="chat"):
    return ProgressEntry(
        event_type=EventType.TASK_STARTED,
        task_id="t",
        data={"task_type": task_type, "objective": objective, "priority": "normal"},
    )


def _routing_decision(action="worker_general"):
    return ProgressEntry(
        event_type=EventType.ROUTING_DECISION,
        task_id="t",
        data={"routing": [action]},
    )


def _task_completed(work: dict | None = None, **extra):
    data = dict(extra)
    if work is not None:
        data["work"] = work
    return ProgressEntry(
        event_type=EventType.TASK_COMPLETED,
        task_id="t",
        data=data,
        outcome="success",
    )


# --- the progress-log write site --------------------------------------------


class TestWorkReachesTheDatabase:
    def test_routing_write_persists_the_work_payload(self, scorer):
        outcome = _task_completed(
            work={
                "answer": "sorted(x) is the builtin",
                "tool_calls": [{"tool_name": "grep", "elapsed_ms": 3, "success": True}],
                "repl_steps": [{"step": 1, "ok": True, "code": "print(sorted([2,1]))"}],
                "reasoning": "the builtin already does this",
            }
        )
        res = scorer._update_routing_memory(
            "t1", _task_started("sort a list"), _routing_decision(), 0.6,
            task_outcome=outcome,
        )
        assert res["memories_created"] == 1

        (ctx,) = _contexts_from_sqlite(scorer.store)
        work = ctx["work"]
        assert work["answer"] == "sorted(x) is the builtin"
        assert work["tool_calls"][0]["tool_name"] == "grep"
        assert work["repl_steps"][0]["code"] == "print(sorted([2,1]))"
        assert work["reasoning"] == "the builtin already does this"

    def test_work_is_not_embedded(self, scorer):
        """Embedding the answer would make retrieval match solutions to solutions."""
        scorer._update_routing_memory(
            "t1", _task_started("sort a list"), _routing_decision(), 0.6,
            task_outcome=_task_completed(work={"answer": "MAGIC_ANSWER_TOKEN"}),
        )
        rec = build_memory_record(objective="sort a list", task_type="chat")
        assert "MAGIC_ANSWER_TOKEN" not in rec.embedding_text()

    def test_no_work_on_the_outcome_stores_no_work_key(self, scorer):
        """The pre-M-11a2b shape must still write cleanly, with no empty stub."""
        scorer._update_routing_memory(
            "t1", _task_started("sort a list"), _routing_decision(), 0.6,
            task_outcome=_task_completed(answer_chars=17),
        )
        (ctx,) = _contexts_from_sqlite(scorer.store)
        assert "work" not in ctx
        assert ctx["objective"] == "sort a list"

    def test_missing_task_outcome_is_not_an_error(self, scorer):
        """Callers that predate the parameter must keep working."""
        res = scorer._update_routing_memory(
            "t1", _task_started("sort a list"), _routing_decision(), 0.6,
        )
        assert res["memories_created"] == 1
        (ctx,) = _contexts_from_sqlite(scorer.store)
        assert "work" not in ctx


# --- the external write site -------------------------------------------------


class TestExternalScorePath:
    def test_flat_work_keys_land_in_work_not_metrics(self, scorer):
        """A work field filed as telemetry is stored where no reader looks."""
        scorer.score_external_result(
            task_description="write a parser",
            action="worker_general",
            reward=0.5,
            context={
                "task_type": "coder",
                "answer": "def parse(s): ...",
                "reasoning": "recursive descent is enough here",
                "elapsed_seconds": 12.5,
            },
        )
        (ctx,) = _contexts_from_sqlite(scorer.store)
        assert ctx["work"]["answer"] == "def parse(s): ..."
        assert ctx["work"]["reasoning"] == "recursive descent is enough here"
        # telemetry still routes to metrics, and work does NOT
        assert ctx["metrics"]["elapsed_seconds"] == 12.5
        assert "answer" not in ctx["metrics"]
        assert "reasoning" not in ctx["metrics"]

    def test_nested_work_dict_is_accepted(self, scorer):
        scorer.score_external_result(
            task_description="write a parser",
            action="worker_general",
            reward=0.5,
            context={"task_type": "coder", "work": {"answer": "def parse(s): ..."}},
        )
        (ctx,) = _contexts_from_sqlite(scorer.store)
        assert ctx["work"]["answer"] == "def parse(s): ..."
        assert "work" not in ctx.get("metrics", {})


# --- the capture policy ------------------------------------------------------


class TestCapturePolicy:
    def test_credentials_are_redacted_before_storage(self, scorer):
        secret = "AKIAIOSFODNN7EXAMPLE"
        scorer._update_routing_memory(
            "t1", _task_started("deploy it"), _routing_decision(), 0.6,
            task_outcome=_task_completed(
                work={
                    "answer": f"export AWS_KEY={secret} and then run it",
                    "repl_steps": [{"step": 1, "code": f"key = '{secret}'"}],
                }
            ),
        )
        raw = json.dumps(_contexts_from_sqlite(scorer.store))
        assert secret not in raw, "a credential reached the episodic store"
        assert "[REDACTED:aws_access_key]" in raw

    def test_reuses_the_repo_redaction_policy(self):
        """No second pattern list: the same helper the REPL/tool paths use."""
        from src.repl_environment import redaction as repo_redaction

        payload = build_work_payload(answer="token sk-ant-" + "a" * 30)
        assert "[REDACTED:anthropic_key]" in payload["answer"]
        # and the marker is the repo's, not a locally invented one
        assert any(
            repl == "[REDACTED:anthropic_key]"
            for _, _, repl in repo_redaction._CREDENTIAL_PATTERNS
        )

    def test_oversize_answer_is_bounded_in_the_database(self, scorer):
        scorer._update_routing_memory(
            "t1", _task_started("loop forever"), _routing_decision(), 0.6,
            task_outcome=_task_completed(work={"answer": "z" * 500_000}),
        )
        (ctx,) = _contexts_from_sqlite(scorer.store)
        stored = ctx["work"]["answer"]
        assert len(stored) < 500_000
        assert stored.startswith("z" * 100)
        assert f"truncated at {WORK_TEXT_MAX_CHARS} chars" in stored

    def test_oversize_step_list_is_bounded_and_says_so(self):
        payload = build_work_payload(
            repl_steps=[{"step": i, "code": "pass"} for i in range(WORK_MAX_ITEMS + 40)]
        )
        steps = payload["repl_steps"]
        # kept entries + one sentinel describing the drop
        assert len(steps) == WORK_MAX_ITEMS + 1
        assert steps[0] == {"_elided_entries": 40}
        # the TAIL is kept: it is what produced the answer
        assert steps[-1]["step"] == WORK_MAX_ITEMS + 39

    def test_oversize_single_step_is_bounded(self):
        payload = build_work_payload(repl_steps=[{"step": 1, "code": "x" * 40_000}])
        encoded = json.dumps(payload["repl_steps"])
        assert len(encoded) < 40_000
        assert f"truncated at {WORK_ITEM_MAX_CHARS} chars" in encoded

    def test_the_objective_is_still_never_truncated(self):
        """The 200-char objective cap was the 2026-07-27 defect; not reintroduced."""
        rec = build_memory_record(objective="q" * 50_000, answer="short")
        assert len(rec.to_context()["objective"]) == 50_000


# --- the producer helper -----------------------------------------------------


class TestEndToEndThroughTheRealChannel:
    """Pipeline helper -> progress JSONL -> ProgressReader -> _score_task -> SQLite.

    The unit tests above call `_update_routing_memory` directly. This one drives
    the ACTUAL production channel, including the JSON round-trip through the
    progress log — the hop where a payload that only exists in a Python object
    would silently vanish.
    """

    def test_work_survives_the_full_production_path(self, tmp_path):
        from orchestration.repl_memory.progress_logger import ProgressLogger, ProgressReader
        from src.api.routes.chat_pipeline.telemetry import work_completion_meta

        log_dir = tmp_path / "progress"
        log_dir.mkdir()
        progress_logger = ProgressLogger(log_dir=log_dir)
        store = EpisodicStore(db_path=tmp_path / "sessions", use_faiss=True)
        try:
            scorer = QScorer(
                store=store,
                embedder=_FakeEmbedder(),
                logger=progress_logger,
                reader=ProgressReader(log_dir=log_dir),
                config=ScoringConfig(learning_rate=0.1, temporal_decay_rate=None),
            )

            progress_logger.log_task_started(
                task_id="T1",
                task_ir={
                    "task_type": "coder",
                    "objective": "reverse a linked list",
                    "priority": "normal",
                },
                routing_decision=["worker_general"],
                routing_strategy="rules",
            )
            progress_logger.log_task_completed(
                task_id="T1",
                success=True,
                details="Direct answer mode (worker_general), 1.0s",
                completion_meta={
                    "producer_role": "worker_general",
                    "final_answer_role": "worker_general",
                    "answer_chars": 42,
                    # exactly what direct_stage / stages / repl_executor now emit
                    **work_completion_meta(
                        answer="iterative three-pointer swap; key = sk-ant-" + "z" * 30,
                        tool_calls=[
                            {"tool_name": "grep", "elapsed_ms": 2, "success": True}
                        ],
                        repl_steps=[{"step": 1, "ok": True, "code": "print('ok')"}],
                    ),
                },
            )
            progress_logger.flush()

            result = scorer._score_task("T1")
            assert result.get("memories_created") == 1

            (ctx,) = _contexts_from_sqlite(store)
            work = ctx["work"]
            assert ctx["objective"] == "reverse a linked list"
            assert work["answer"].startswith("iterative three-pointer swap")
            assert work["tool_calls"][0]["tool_name"] == "grep"
            assert work["repl_steps"][0]["code"] == "print('ok')"
            # redaction happened at the producer, so the secret never entered
            # EITHER store
            assert "sk-ant-" not in work["answer"]
            assert "[REDACTED:anthropic_key]" in work["answer"]
            jsonl = "".join(p.read_text() for p in log_dir.glob("*.jsonl"))
            assert "sk-ant-" not in jsonl
        finally:
            store.close()


class TestWorkCompletionMeta:
    def test_empty_work_yields_no_key(self):
        from src.api.routes.chat_pipeline.telemetry import work_completion_meta

        assert work_completion_meta() == {}
        assert work_completion_meta(answer="", tool_calls=[], repl_steps=[]) == {}

    def test_populated_work_is_nested_under_work(self):
        from src.api.routes.chat_pipeline.telemetry import work_completion_meta

        meta = work_completion_meta(answer="42", tool_calls=[{"tool_name": "calc"}])
        assert meta["work"]["answer"] == "42"
        assert meta["work"]["tool_calls"][0]["tool_name"] == "calc"

    def test_extract_work_reads_the_producer_shape(self):
        from src.api.routes.chat_pipeline.telemetry import work_completion_meta

        meta = work_completion_meta(answer="42", reasoning="because")
        completion_meta = {"producer_role": "worker_general", **meta}
        work = extract_work(completion_meta)
        assert work == {"answer": "42", "reasoning": "because"}
