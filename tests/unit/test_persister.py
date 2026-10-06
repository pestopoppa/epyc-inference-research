"""Comprehensive tests for session persister.

Tests coverage for src/session/persister.py (16% coverage).
Focus on checkpoint triggers, finding sync, and lifecycle events.
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock


from src.session.persister import (
    CHECKPOINT_IDLE_MINUTES,
    CHECKPOINT_TURN_INTERVAL,
    IdleMonitor,
    SessionPersister,
    SUMMARY_IDLE_HOURS,
)
from src.session.models import (
    Session,
    SessionStatus,
)
from src.repl_environment import safe_pickle
from src.repl_environment import REPLEnvironment
from src.session.sqlite_store import SQLiteSessionStore


class TestSessionPersisterInit:
    """Test SessionPersister initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        assert persister.session_store is session_store
        assert persister.session_id == "sess_123"
        assert persister._task_id == "task_456"
        assert persister._turn_count == 0

    def test_init_with_optional_params(self):
        """Test initialization with optional parameters."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        llm_summarizer = Mock()
        progress_logger = Mock()

        persister = SessionPersister(
            session_store,
            "sess_123",
            llm_summarizer=llm_summarizer,
            progress_logger=progress_logger,
        )

        assert persister.llm_summarizer is llm_summarizer
        assert persister.progress_logger is progress_logger


class TestOnTurn:
    """Test on_turn() tracking."""

    def test_on_turn_increments_count(self):
        """Test turn counter increment."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        result = persister.on_turn()

        assert persister._turn_count == 1
        assert result["turn"] == 1

    def test_on_turn_updates_activity(self):
        """Test activity timestamp update."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        before = persister._last_activity
        time.sleep(0.01)
        persister.on_turn()

        assert persister._last_activity > before

    def test_on_turn_triggers_checkpoint(self):
        """Test checkpoint trigger on turn interval."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")
        repl_env = Mock()
        repl_env.checkpoint.return_value = {"version": 1, "artifacts": {}}
        repl_env.context = "test"
        repl_env.get_findings.return_value = []
        repl_env.clear_findings = Mock()

        # Advance to checkpoint threshold
        persister._turn_count = CHECKPOINT_TURN_INTERVAL - 1

        result = persister.on_turn(repl_env)

        assert result["action"] == "checkpoint"


class TestShouldCheckpoint:
    """Test checkpoint trigger conditions."""

    def test_should_checkpoint_turn_based(self):
        """Test turn-based checkpoint trigger."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        # Not enough turns yet
        persister._turn_count = CHECKPOINT_TURN_INTERVAL - 1
        assert persister.should_checkpoint() is False

        # At threshold
        persister._turn_count = CHECKPOINT_TURN_INTERVAL
        assert persister.should_checkpoint() is True

    def test_should_checkpoint_time_based(self):
        """Test time-based checkpoint trigger."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        # Simulate idle time
        persister._last_checkpoint_time = time.time() - (CHECKPOINT_IDLE_MINUTES * 60 + 1)

        assert persister.should_checkpoint() is True


class TestSaveCheckpoint:
    """Test checkpoint saving."""

    def test_save_checkpoint_basic(self):
        """Test basic checkpoint save."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")
        repl_env = Mock()
        repl_env.checkpoint.return_value = {
            "version": 1,
            "artifacts": {"key": "value"},
            "execution_count": 5,
            "exploration_calls": 3,
        }
        repl_env.context = "test context"
        repl_env.get_findings.return_value = []
        repl_env.clear_findings = Mock()

        checkpoint = persister.save_checkpoint(repl_env, trigger="explicit")

        assert checkpoint.session_id == "sess_123"
        assert checkpoint.trigger == "explicit"
        assert checkpoint.execution_count == 5
        assert checkpoint.artifacts["key"] == "value"
        session_store.save_checkpoint.assert_called_once_with(checkpoint)

    def test_save_checkpoint_computes_hash(self):
        """Test checkpoint computes context hash."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")
        repl_env = Mock()
        repl_env.checkpoint.return_value = {"version": 1, "artifacts": {}}
        repl_env.context = "test context"
        repl_env.get_findings.return_value = []
        repl_env.clear_findings = Mock()

        checkpoint = persister.save_checkpoint(repl_env)

        # Should have computed hash
        assert checkpoint.context_hash.startswith("sha256:")

    def test_save_checkpoint_with_progress_logger(self):
        """Test checkpoint logging to progress logger."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        progress_logger = Mock()
        persister = SessionPersister(session_store, "sess_123", progress_logger=progress_logger)

        repl_env = Mock()
        repl_env.checkpoint.return_value = {"version": 1, "artifacts": {}}
        repl_env.context = "test"
        repl_env.get_findings.return_value = []
        repl_env.clear_findings = Mock()

        persister.save_checkpoint(repl_env)

        progress_logger.log_session_checkpointed.assert_called_once()

    def test_save_checkpoint_persists_user_globals(self):
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")
        repl_env = Mock()
        repl_env.checkpoint.return_value = {
            "version": 1,
            "artifacts": {},
            "user_globals": {"total": 9},
            "variable_lineage": {"total": {"role": "frontdoor", "saved_at_ts": 1.0}},
            "skipped_user_globals": ["tmp_lambda"],
        }
        repl_env.context = "test context"
        repl_env.get_findings.return_value = []
        repl_env.clear_findings = Mock()

        checkpoint = persister.save_checkpoint(repl_env)
        assert checkpoint.user_globals == {"total": 9}
        assert checkpoint.variable_lineage["total"]["role"] == "frontdoor"
        assert checkpoint.skipped_user_globals == ["tmp_lambda"]

    def test_save_checkpoint_forwards_pickled_globals(self):
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session
        persister = SessionPersister(session_store, "sess_123")
        envelope = safe_pickle.dumps({"alpha", "beta"})
        repl_env = Mock()
        repl_env.checkpoint.return_value = {
            "version": 1,
            "artifacts": {},
            "pickled_globals": {"persist_set": envelope},
            "variable_lineage": {"persist_set": {"saved_at_ts": 1.0}},
        }
        repl_env.context = "test context"
        repl_env.get_findings.return_value = []
        repl_env.clear_findings = Mock()

        checkpoint = persister.save_checkpoint(repl_env)
        assert checkpoint.pickled_globals == {"persist_set": envelope}

    def test_sqlite_checkpoint_restores_pickled_global(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_SESSION_HMAC_KEY", "ni07-persister-test-key")
        store = SQLiteSessionStore(
            db_path=tmp_path / "sessions.db",
            embeddings_path=tmp_path / "embeddings.npy",
        )
        session = Session.create(name="pickle", working_directory=str(tmp_path))
        store.create_session(session)
        persister = SessionPersister(store, session.id)
        source = REPLEnvironment(context="persist set")
        source._globals["persist_set"] = {"alpha", "beta"}

        try:
            saved = persister.save_checkpoint(source, trigger="explicit")
            loaded = store.get_latest_checkpoint(session.id)
            assert loaded is not None
            assert loaded.pickled_globals == saved.pickled_globals

            restored = REPLEnvironment(context="resume set")
            result = restored.restore(loaded.to_dict())
            assert "persist_set" in result["restored"]
            assert restored._globals["persist_set"] == {"alpha", "beta"}
        finally:
            store.close()

    def test_combined_globals_cap_evicts_oldest_across_json_and_pickle(self, monkeypatch):
        import src.session.persister as persister_module

        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session
        persister = SessionPersister(session_store, "sess_123")
        old_envelope = {"b64": "o" * 220, "hmac": "x" * 64}
        newer_envelope = {"b64": "n" * 220, "hmac": "x" * 64}
        repl_env = Mock()
        repl_env.checkpoint.return_value = {
            "version": 1,
            "artifacts": {},
            "user_globals": {"new_json": "j" * 220},
            "pickled_globals": {"old_pickle": old_envelope, "new_pickle": newer_envelope},
            "variable_lineage": {
                "old_pickle": {"saved_at_ts": 1.0},
                "new_pickle": {"saved_at_ts": 3.0},
                "new_json": {"saved_at_ts": 2.0},
            },
        }
        repl_env.context = "test context"
        repl_env.get_findings.return_value = []
        repl_env.clear_findings = Mock()
        monkeypatch.setattr(persister_module, "CHECKPOINT_GLOBALS_HARD_BYTES", 500)
        monkeypatch.setattr(persister_module, "CHECKPOINT_GLOBALS_WARN_BYTES", 10_000)

        checkpoint = persister.save_checkpoint(repl_env)

        assert checkpoint.skipped_user_globals == ["old_pickle", "new_json"]
        assert checkpoint.pickled_globals == {"new_pickle": newer_envelope}
        assert checkpoint.user_globals == {}
        assert checkpoint.variable_lineage == {"new_pickle": {"saved_at_ts": 3.0}}


class TestSyncFindings:
    """Test findings synchronization."""

    def test_sync_findings_empty(self):
        """Test syncing with no findings."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")
        repl_env = Mock()
        repl_env.get_findings.return_value = []
        repl_env.clear_findings = Mock()

        synced = persister.sync_findings(repl_env)

        assert synced == 0
        repl_env.clear_findings.assert_not_called()

    def test_sync_findings_with_data(self):
        """Test syncing findings to store."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")
        repl_env = Mock()
        repl_env.get_findings.return_value = [
            {
                "id": "find_1",
                "content": "Important finding",
                "timestamp": time.time(),
                "source": {},
                "tags": ["important"],
            }
        ]
        repl_env.clear_findings = Mock()

        synced = persister.sync_findings(repl_env)

        assert synced == 1
        session_store.add_finding.assert_called_once()
        repl_env.clear_findings.assert_called_once()


class TestHeuristicFindings:
    """Test heuristic finding extraction."""

    def test_extract_heuristic_findings_key_pattern(self):
        """Test extraction of KEY: pattern."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        text = "KEY: This is an important finding.\nNormal text here."
        findings = persister.extract_heuristic_findings(text)

        assert len(findings) > 0
        assert any("important finding" in f["content"] for f in findings)

    def test_extract_heuristic_findings_multiple_patterns(self):
        """Test extraction of multiple pattern types."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        text = """
        KEY FINDING: First finding here.
        IMPORTANT: Second finding here.
        NOTE: Third finding here.
        CONCLUSION: Final finding here.
        """
        findings = persister.extract_heuristic_findings(text)

        assert len(findings) >= 3

    def test_add_heuristic_findings(self):
        """Test adding heuristic findings to store."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        text = "KEY: Important finding here with enough length."
        added = persister.add_heuristic_findings(text, min_confidence=0.5)

        assert added >= 1
        session_store.add_finding.assert_called()


class TestCheckIdle:
    """Test idle monitoring."""

    def test_check_idle_no_action_needed(self):
        """Test check_idle when no action needed."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        result = persister.check_idle()

        assert result["actions"] == []

    def test_check_idle_checkpoint_needed(self):
        """Test check_idle when checkpoint needed."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        persister = SessionPersister(session_store, "sess_123")

        # Simulate idle time exceeding threshold
        persister._last_activity = time.time() - (CHECKPOINT_IDLE_MINUTES * 60 + 1)
        persister._last_checkpoint_time = time.time() - (CHECKPOINT_IDLE_MINUTES * 60 + 1)

        result = persister.check_idle()

        assert "checkpoint_needed" in result["actions"]

    def test_check_idle_summary_generated(self):
        """Test check_idle triggers summary generation."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc) - timedelta(hours=SUMMARY_IDLE_HOURS + 1),
        )
        session_store.get_session.return_value = mock_session
        session_store.get_findings.return_value = []

        persister = SessionPersister(session_store, "sess_123")

        # Simulate long idle time
        persister._last_activity = time.time() - (SUMMARY_IDLE_HOURS * 3600 + 1)

        result = persister.check_idle()

        assert "summary_generated" in result["actions"]
        assert persister._summary_generated is True


class TestGenerateSummary:
    """Test summary generation."""

    def test_generate_summary_with_llm(self):
        """Test summary generation using LLM."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            name="Test Session",
            message_count=10,
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session
        session_store.get_findings.return_value = []

        llm_summarizer = Mock(return_value="LLM-generated summary")
        persister = SessionPersister(session_store, "sess_123", llm_summarizer=llm_summarizer)

        summary = persister._generate_summary()

        assert summary == "LLM-generated summary"
        llm_summarizer.assert_called_once()

    def test_generate_summary_fallback_heuristic(self):
        """Test summary generation falls back to heuristic."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            message_count=5,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session
        session_store.get_findings.return_value = []

        # No LLM summarizer provided
        persister = SessionPersister(session_store, "sess_123")

        summary = persister._generate_summary()

        assert "5 messages" in summary
        assert "1.0 hours" in summary


class TestLifecycleEvents:
    """Test MemRL lifecycle event emitters."""

    def test_emit_session_created(self):
        """Test session created event."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            name="New Session",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        progress_logger = Mock()
        persister = SessionPersister(session_store, "sess_123", progress_logger=progress_logger)

        persister.emit_session_created()

        progress_logger.log_session_created.assert_called_once()

    def test_emit_session_resumed(self):
        """Test session resumed event."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_789",  # New task ID after fork
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        progress_logger = Mock()
        persister = SessionPersister(session_store, "sess_123", progress_logger=progress_logger)

        persister.emit_session_resumed(previous_task_id="task_456", document_changes=2)

        progress_logger.log_session_resumed.assert_called_once()

    def test_emit_session_archived(self):
        """Test session archived event."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            message_count=20,
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session
        session_store.get_findings.return_value = []

        progress_logger = Mock()
        persister = SessionPersister(session_store, "sess_123", progress_logger=progress_logger)

        persister.emit_session_archived()

        progress_logger.log_session_archived.assert_called_once()


class TestIdleMonitor:
    """Test IdleMonitor background monitoring."""

    def test_idle_monitor_init(self):
        """Test idle monitor initialization."""
        session_store = Mock()
        monitor = IdleMonitor(session_store)

        assert monitor.session_store is session_store
        assert monitor._persisters == {}

    def test_get_persister_creates_new(self):
        """Test persister creation on first access."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        monitor = IdleMonitor(session_store)

        persister = monitor.get_persister("sess_123")

        assert persister.session_id == "sess_123"
        assert "sess_123" in monitor._persisters

    def test_get_persister_reuses_existing(self):
        """Test persister reuse."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session

        monitor = IdleMonitor(session_store)

        persister1 = monitor.get_persister("sess_123")
        persister2 = monitor.get_persister("sess_123")

        assert persister1 is persister2

    def test_cleanup_stale_persisters(self):
        """Test cleanup of persisters for archived sessions."""
        session_store = Mock()
        mock_session = Session(
            id="sess_123",
            task_id="task_456",
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
        )
        session_store.get_session.return_value = mock_session
        session_store.list_sessions.return_value = [mock_session]

        monitor = IdleMonitor(session_store)

        # Create persisters for two sessions
        monitor.get_persister("sess_123")
        monitor.get_persister("sess_archived")

        # Only sess_123 is active
        removed = monitor.cleanup_stale_persisters()

        assert removed == 1
        assert "sess_123" in monitor._persisters
        assert "sess_archived" not in monitor._persisters
