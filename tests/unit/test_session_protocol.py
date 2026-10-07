#!/usr/bin/env python3
"""Unit tests for src/session/protocol.py."""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.session.models import (
    Checkpoint,
    Finding,
    FindingSource,
    Session,
    SessionMessage,
    SessionDocument,
)
from src.session.protocol import (
    BaseSessionStore,
    CHECKPOINT_PROTOCOL_CURRENT_VERSION,
    WhereFilter,
    normalize_checkpoint_for_repl_restore,
)


class TestWhereFilter:
    """Test WhereFilter type alias and filter operations."""

    def test_where_filter_simple(self):
        """Test simple WhereFilter dict."""
        filter: WhereFilter = {"status": "active"}
        assert filter["status"] == "active"

    def test_where_filter_in_operator(self):
        """Test $in operator in WhereFilter."""
        filter: WhereFilter = {"status": {"$in": ["active", "idle"]}}
        assert "$in" in filter["status"]
        assert filter["status"]["$in"] == ["active", "idle"]

    def test_where_filter_comparison_operators(self):
        """Test comparison operators in WhereFilter."""
        # Greater than or equal
        filter_gte: WhereFilter = {"message_count": {"$gte": 10}}
        assert filter_gte["message_count"]["$gte"] == 10

        # Less than or equal
        filter_lte: WhereFilter = {"message_count": {"$lte": 100}}
        assert filter_lte["message_count"]["$lte"] == 100

        # Not equal
        filter_ne: WhereFilter = {"status": {"$ne": "archived"}}
        assert filter_ne["status"]["$ne"] == "archived"

    def test_where_filter_combined(self):
        """Test combined WhereFilter conditions."""
        filter: WhereFilter = {
            "status": {"$in": ["active", "idle"]},
            "message_count": {"$gte": 5},
            "project": "test-project",
        }

        assert len(filter) == 3
        assert filter["project"] == "test-project"


class MockSessionStore(BaseSessionStore):
    """Mock implementation of BaseSessionStore for testing."""

    def __init__(self):
        self.sessions = {}
        self.documents = {}
        self.findings = {}
        self.checkpoints = {}
        self.messages = []

    def create_session(self, session: Session) -> Session:
        if session.id in self.sessions:
            raise ValueError(f"Session {session.id} already exists")
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def update_session(self, session: Session) -> Session:
        if session.id not in self.sessions:
            raise ValueError(f"Session {session.id} not found")
        self.sessions[session.id] = session
        return session

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.messages = [m for m in self.messages if m.session_id != session_id]
            return True
        return False

    def append_message(
        self, session_id, turn_id, role, text, *, spoken_text=None, display=None, created_at=None
    ) -> SessionMessage:
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        stamp = created_at or datetime.now(timezone.utc)
        message = SessionMessage(
            id=len(self.messages) + 1,
            session_id=session_id,
            turn_id=turn_id,
            role=role,
            text=text,
            spoken_text=spoken_text,
            display=display,
            created_at=stamp,
            updated_at=stamp,
        )
        self.messages.append(message)
        return message

    def get_messages(self, session_id, *, limit=200) -> list[SessionMessage]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        return [m for m in self.messages if m.session_id == session_id][-limit:]

    def list_sessions(
        self,
        where: WhereFilter | None = None,
        order_by: str = "last_active",
        descending: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Session]:
        # Simple implementation for testing
        sessions = list(self.sessions.values())
        return sessions[offset : offset + limit if limit else None]

    def add_document(self, document: SessionDocument) -> SessionDocument:
        self.documents[document.id] = document
        return document

    def get_documents(self, session_id: str) -> list[SessionDocument]:
        return [d for d in self.documents.values() if d.session_id == session_id]

    def update_document(self, document: SessionDocument) -> SessionDocument:
        self.documents[document.id] = document
        return document

    def add_finding(self, finding: Finding) -> Finding:
        self.findings[finding.id] = finding
        return finding

    def get_findings(self, session_id: str) -> list[Finding]:
        return [f for f in self.findings.values() if f.session_id == session_id]

    def update_finding(self, finding: Finding) -> Finding:
        self.findings[finding.id] = finding
        return finding

    def delete_finding(self, finding_id: str) -> bool:
        if finding_id in self.findings:
            del self.findings[finding_id]
            return True
        return False

    def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        self.checkpoints[checkpoint.id] = checkpoint
        return checkpoint

    def get_latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        session_checkpoints = [
            cp for cp in self.checkpoints.values() if cp.session_id == session_id
        ]
        return session_checkpoints[-1] if session_checkpoints else None

    def get_checkpoints(self, session_id: str, limit: int = 10) -> list[Checkpoint]:
        session_checkpoints = [
            cp for cp in self.checkpoints.values() if cp.session_id == session_id
        ]
        return session_checkpoints[-limit:]

    def search_sessions(
        self,
        query: str,
        search_in: list[str] | None = None,
        limit: int = 10,
    ) -> list[Session]:
        # Simple mock implementation
        return []

    def search_findings(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[Finding]:
        # Simple mock implementation
        return []

    def store_embedding(
        self,
        session_id: str,
        embedding: np.ndarray,
        content_type: str = "session",
    ) -> int:
        # Mock implementation
        return 0

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        content_type: str = "session",
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> list[tuple[str, float]]:
        # Mock implementation
        return []

    def add_tag(self, session_id: str, tag: str) -> bool:
        session = self.get_session(session_id)
        if session and tag not in session.tags:
            session.tags.append(tag)
            return True
        return False

    def remove_tag(self, session_id: str, tag: str) -> bool:
        session = self.get_session(session_id)
        if session and tag in session.tags:
            session.tags.remove(tag)
            return True
        return False

    def get_sessions_by_tag(self, tag: str) -> list[Session]:
        return [s for s in self.sessions.values() if tag in s.tags]

    def close(self) -> None:
        pass


class TestBaseSessionStore:
    """Test BaseSessionStore abstract methods."""

    @pytest.fixture
    def store(self):
        """Create a mock session store."""
        return MockSessionStore()

    def test_archive_session(self, store):
        """Test archive_session updates status."""
        session = Session.create()
        store.create_session(session)

        result = store.archive_session(session.id)

        assert result is True
        archived = store.get_session(session.id)
        assert archived.status.value == "archived"

    def test_archive_nonexistent_session(self, store):
        """Test archive_session returns False for missing session."""
        result = store.archive_session("nonexistent")
        assert result is False

    def test_build_resume_context(self, store, tmp_path):
        """Test build_resume_context assembles full context."""
        # Create session with document and finding
        session = Session.create()
        store.create_session(session)

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        from datetime import datetime, timezone

        doc = SessionDocument(
            id="doc1",
            session_id=session.id,
            file_path=str(test_file),
            file_hash=SessionDocument.compute_file_hash(test_file),
            processed_at=datetime.now(timezone.utc),
        )
        store.add_document(doc)

        finding = Finding(
            id="f1",
            session_id=session.id,
            content="Test finding",
            source=FindingSource.USER_MARKED,
            created_at=datetime.now(timezone.utc),
        )
        store.add_finding(finding)

        # Build resume context
        context = store.build_resume_context(session.id)

        assert context is not None
        assert context.session.id == session.id
        assert len(context.documents) == 1
        assert len(context.findings) == 1
        assert context.context_summary  # Should have formatted summary

    def test_build_resume_context_missing_session(self, store):
        """Test build_resume_context returns None for missing session."""
        context = store.build_resume_context("nonexistent")
        assert context is None


class TestCheckpointProtocolNormalization:
    def test_normalize_legacy_payload_missing_version(self):
        payload = {
            "artifacts": {"a": 1},
            "execution_count": 3,
            "exploration_calls": 1,
            "user_globals": {"x": 1},
        }
        normalized, diag = normalize_checkpoint_for_repl_restore(payload)
        assert normalized["version"] == 1
        assert normalized["user_globals"]["x"] == 1
        assert diag["source_version"] == 0
        assert diag["compat_mode"] == "legacy_upgrade"
        assert diag["version_present"] is False

    def test_normalize_forward_payload_drops_unknown_fields(self):
        payload = {
            "protocol_version": CHECKPOINT_PROTOCOL_CURRENT_VERSION + 1,
            "artifacts": {},
            "execution_count": 0,
            "exploration_calls": 0,
            "future_field": {"x": "y"},
        }
        normalized, diag = normalize_checkpoint_for_repl_restore(payload)
        assert normalized["version"] == 1
        assert "future_field" not in normalized
        assert diag["compat_mode"] == "forward_downgrade"
        assert "future_field" in diag["dropped_fields"]


class TestSessionStoreProtocol:
    """Test SessionStore protocol compliance."""

    def test_mock_store_implements_protocol(self):
        """Test MockSessionStore implements SessionStore protocol."""
        from src.session.protocol import SessionStore

        store = MockSessionStore()

        # Should be recognized as SessionStore
        assert isinstance(store, SessionStore)

    def test_session_crud_operations(self):
        """Test basic CRUD operations."""
        store = MockSessionStore()

        # Create
        session = Session.create(name="test")
        created = store.create_session(session)
        assert created.id == session.id

        # Read
        retrieved = store.get_session(session.id)
        assert retrieved is not None
        assert retrieved.name == "test"

        # Update
        retrieved.name = "updated"
        updated = store.update_session(retrieved)
        assert updated.name == "updated"

        # Delete
        deleted = store.delete_session(session.id)
        assert deleted is True
        assert store.get_session(session.id) is None

    def test_tag_operations(self):
        """Test tag add/remove/search operations."""
        store = MockSessionStore()

        session = Session.create()
        store.create_session(session)

        # Add tag
        added = store.add_tag(session.id, "important")
        assert added is True

        # Get by tag
        tagged_sessions = store.get_sessions_by_tag("important")
        assert len(tagged_sessions) == 1
        assert tagged_sessions[0].id == session.id

        # Remove tag
        removed = store.remove_tag(session.id, "important")
        assert removed is True
