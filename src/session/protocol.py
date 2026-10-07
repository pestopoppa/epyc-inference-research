"""SessionStore protocol - abstract interface for session persistence.

This protocol is designed to be ChromaDB-compatible, enabling future migration
from SQLite to ChromaDB for semantic search capabilities.

Interface patterns:
- list(where={}) uses ChromaDB-style metadata filters
- search(query) supports both text and semantic search
- Embeddings are stored alongside metadata for future vector search
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from src.session.models import (
    Checkpoint,
    Finding,
    ResumeContext,
    Session,
    SessionConversationSummary,
    SessionMessage,
    SessionDocument,
)

logger = logging.getLogger(__name__)


# Type alias for ChromaDB-style where filters
# Examples:
#   {"status": "active"}
#   {"status": {"$in": ["active", "idle"]}}
#   {"message_count": {"$gte": 10}}
WhereFilter = dict[str, Any]


# Checkpoint payload protocol consumed by REPL restore().
# Version 0: legacy payloads without explicit protocol_version.
# Version 1: explicit protocol_version field + current normalized shape.
CHECKPOINT_PROTOCOL_CURRENT_VERSION = 1
CHECKPOINT_PROTOCOL_LEGACY_VERSION = 0

# Fields expected by REPL restore() payload normalization.
CHECKPOINT_RESTORE_REQUIRED_FIELDS = (
    "artifacts",
    "execution_count",
    "exploration_calls",
)
CHECKPOINT_RESTORE_OPTIONAL_FIELDS = (
    "exploration_tokens",
    "exploration_events",
    "grep_hits_buffer",
    "findings_buffer",
    "context_length",
    "task_id",
    "research_context",
    "user_globals",
    "variable_lineage",
    "skipped_user_globals",
    "pickled_globals",
)


def normalize_checkpoint_for_repl_restore(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize persisted checkpoint payload to REPL restore schema v1.

    Compatibility behavior:
    - Missing protocol_version is treated as legacy v0 and upgraded.
    - Older versions are best-effort upgraded to v1.
    - Newer versions are best-effort downgraded to v1 by dropping unknown fields.
    """
    raw = dict(payload or {})
    protocol_raw = raw.get("protocol_version")
    parse_error = False
    if protocol_raw is None:
        source_version = CHECKPOINT_PROTOCOL_LEGACY_VERSION
    else:
        try:
            source_version = int(protocol_raw)
        except Exception:
            source_version = CHECKPOINT_PROTOCOL_LEGACY_VERSION
            parse_error = True

    known_fields = set(CHECKPOINT_RESTORE_REQUIRED_FIELDS) | set(CHECKPOINT_RESTORE_OPTIONAL_FIELDS)
    known_fields.update({"version", "protocol_version", "id", "session_id", "created_at", "context_hash", "message_count", "trigger"})
    dropped_fields: list[str] = []

    normalized = dict(raw)
    if source_version > CHECKPOINT_PROTOCOL_CURRENT_VERSION:
        for key in list(normalized.keys()):
            if key not in known_fields:
                dropped_fields.append(key)
                normalized.pop(key, None)

    # REPL restore() currently expects version=1.
    normalized["version"] = 1
    normalized.setdefault("artifacts", {})
    normalized.setdefault("execution_count", 0)
    normalized.setdefault("exploration_calls", 0)
    normalized.setdefault("user_globals", {})
    normalized.setdefault("variable_lineage", {})
    normalized.setdefault("skipped_user_globals", [])
    normalized.setdefault("pickled_globals", {})

    missing_required = [
        key for key in CHECKPOINT_RESTORE_REQUIRED_FIELDS
        if key not in raw
    ]

    if source_version == CHECKPOINT_PROTOCOL_CURRENT_VERSION and not parse_error:
        compat_mode = "exact"
    elif source_version > CHECKPOINT_PROTOCOL_CURRENT_VERSION:
        compat_mode = "forward_downgrade"
    else:
        compat_mode = "legacy_upgrade"

    diagnostics = {
        "source_version": source_version,
        "target_version": CHECKPOINT_PROTOCOL_CURRENT_VERSION,
        "version_present": protocol_raw is not None,
        "parse_error": parse_error,
        "compat_mode": compat_mode,
        "dropped_fields": dropped_fields,
        "missing_required_fields": missing_required,
    }
    return normalized, diagnostics


@runtime_checkable
class SessionStore(Protocol):
    """Abstract interface for session persistence.

    Implementations:
    - SQLiteSessionStore: Current implementation using SQLite + numpy
    - ChromaDBSessionStore: Future implementation for semantic search

    All methods are synchronous. For async contexts, wrap in executor.
    """

    # =========================================================================
    # Session CRUD
    # =========================================================================

    def create_session(self, session: Session) -> Session:
        """Create a new session.

        Args:
            session: Session to create (id should be pre-generated)

        Returns:
            The created session

        Raises:
            ValueError: If session with same id already exists
        """
        ...

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The session ID

        Returns:
            The session, or None if not found
        """
        ...

    def update_session(self, session: Session) -> Session:
        """Update an existing session.

        Args:
            session: Session with updated fields

        Returns:
            The updated session

        Raises:
            ValueError: If session does not exist
        """
        ...

    def append_message(
        self,
        session_id: str,
        turn_id: str,
        role: str,
        text: str,
        *,
        spoken_text: str | None = None,
        display: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        fencing_token: int | None = None,
    ) -> SessionMessage:
        """Append one conversation message to an existing session."""
        ...

    def get_messages(self, session_id: str, *, limit: int = 200) -> list[SessionMessage]:
        """Return the newest bounded message window in chronological order."""
        ...

    def save_conversation_summary(
        self,
        session_id: str,
        through_message_id: int,
        summary: str,
        *,
        updated_at: datetime | None = None,
        fencing_token: int | None = None,
    ) -> SessionConversationSummary:
        """Persist caller-produced summary state through one message frontier."""
        ...

    def get_conversation_summary(self, session_id: str) -> SessionConversationSummary | None:
        """Return the latest compact state for a session, if one was saved."""
        ...

    def refresh_conversation_summary(
        self,
        session_id: str,
        summarize: Callable[
            [SessionConversationSummary | None, list[SessionMessage]], str
        ],
        *,
        fencing_token: int | None = None,
    ) -> SessionConversationSummary | None:
        """Refresh prior state using the bounded unseen suffix via a caller."""
        ...

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated data.

        Args:
            session_id: The session ID

        Returns:
            True if deleted, False if not found
        """
        ...

    def list_sessions(
        self,
        where: WhereFilter | None = None,
        order_by: str = "last_active",
        descending: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Session]:
        """List sessions with optional filtering.

        Args:
            where: ChromaDB-style filter (e.g., {"status": "active"})
            order_by: Field to sort by
            descending: Sort order
            limit: Maximum results (None = unlimited)
            offset: Skip first N results

        Returns:
            List of matching sessions
        """
        ...

    def archive_session(self, session_id: str) -> bool:
        """Archive a session (move to cold storage).

        Args:
            session_id: The session ID

        Returns:
            True if archived, False if not found
        """
        ...

    # =========================================================================
    # Document tracking
    # =========================================================================

    def add_document(self, document: SessionDocument) -> SessionDocument:
        """Add a document to a session.

        Args:
            document: Document metadata to add

        Returns:
            The added document
        """
        ...

    def get_documents(self, session_id: str) -> list[SessionDocument]:
        """Get all documents for a session.

        Args:
            session_id: The session ID

        Returns:
            List of documents
        """
        ...

    def update_document(self, document: SessionDocument) -> SessionDocument:
        """Update document metadata (e.g., after re-processing).

        Args:
            document: Document with updated fields

        Returns:
            The updated document
        """
        ...

    # =========================================================================
    # Findings
    # =========================================================================

    def add_finding(self, finding: Finding) -> Finding:
        """Add a key finding to a session.

        Args:
            finding: Finding to add

        Returns:
            The added finding
        """
        ...

    def get_findings(self, session_id: str) -> list[Finding]:
        """Get all findings for a session.

        Args:
            session_id: The session ID

        Returns:
            List of findings
        """
        ...

    def update_finding(self, finding: Finding) -> Finding:
        """Update a finding (e.g., confirm LLM extraction).

        Args:
            finding: Finding with updated fields

        Returns:
            The updated finding
        """
        ...

    def delete_finding(self, finding_id: str) -> bool:
        """Delete a finding.

        Args:
            finding_id: The finding ID

        Returns:
            True if deleted, False if not found
        """
        ...

    # =========================================================================
    # Checkpoints
    # =========================================================================

    def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        """Save a checkpoint for crash recovery.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            The saved checkpoint
        """
        ...

    def get_latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a session.

        Args:
            session_id: The session ID

        Returns:
            Latest checkpoint, or None if no checkpoints exist
        """
        ...

    def get_checkpoints(self, session_id: str, limit: int = 10) -> list[Checkpoint]:
        """Get recent checkpoints for a session.

        Args:
            session_id: The session ID
            limit: Maximum number of checkpoints

        Returns:
            List of checkpoints, newest first
        """
        ...

    # =========================================================================
    # Search
    # =========================================================================

    def search_sessions(
        self,
        query: str,
        search_in: list[str] | None = None,
        limit: int = 10,
    ) -> list[Session]:
        """Search sessions by text query.

        Args:
            query: Search query
            search_in: Fields to search (default: ["name", "summary", "last_topic"])
            limit: Maximum results

        Returns:
            List of matching sessions, ranked by relevance
        """
        ...

    def search_findings(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[Finding]:
        """Search findings by text query.

        Args:
            query: Search query
            session_id: Limit to specific session (None = all sessions)
            limit: Maximum results

        Returns:
            List of matching findings
        """
        ...

    # =========================================================================
    # Resume context
    # =========================================================================

    def build_resume_context(self, session_id: str) -> ResumeContext | None:
        """Build the full resume context for a session.

        This includes:
        - Session metadata
        - All documents (with change detection)
        - Key findings
        - Formatted context summary

        Args:
            session_id: The session ID

        Returns:
            Resume context, or None if session not found
        """
        ...

    # =========================================================================
    # Embeddings (for future ChromaDB semantic search)
    # =========================================================================

    def store_embedding(
        self,
        session_id: str,
        embedding: np.ndarray,
        content_type: str = "session",
    ) -> int:
        """Store an embedding for semantic search.

        Args:
            session_id: The session ID
            embedding: The embedding vector (1024-dim for TaskEmbedder)
            content_type: Type of content ("session", "finding", "document")

        Returns:
            The embedding index
        """
        ...

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        content_type: str = "session",
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Search by embedding similarity.

        Args:
            embedding: Query embedding vector
            content_type: Filter by content type
            limit: Maximum results
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of (session_id, similarity_score) tuples
        """
        ...

    # =========================================================================
    # Tags (for project organization)
    # =========================================================================

    def add_tag(self, session_id: str, tag: str) -> bool:
        """Add a tag to a session.

        Args:
            session_id: The session ID
            tag: Tag to add

        Returns:
            True if added, False if already exists
        """
        ...

    def remove_tag(self, session_id: str, tag: str) -> bool:
        """Remove a tag from a session.

        Args:
            session_id: The session ID
            tag: Tag to remove

        Returns:
            True if removed, False if not found
        """
        ...

    def get_sessions_by_tag(self, tag: str) -> list[Session]:
        """Get all sessions with a specific tag.

        Args:
            tag: The tag to filter by

        Returns:
            List of sessions with the tag
        """
        ...

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the store and release resources."""
        ...


class BaseSessionStore(ABC):
    """Base class with common functionality for SessionStore implementations.

    Subclasses must implement the abstract methods.
    """

    @abstractmethod
    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        ...

    @abstractmethod
    def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        ...

    @abstractmethod
    def append_message(
        self,
        session_id: str,
        turn_id: str,
        role: str,
        text: str,
        *,
        spoken_text: str | None = None,
        display: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        fencing_token: int | None = None,
    ) -> SessionMessage:
        """Append a transcript message, honoring any active session lease."""
        ...

    @abstractmethod
    def get_messages(
        self, session_id: str, *, limit: int = 200
    ) -> list[SessionMessage]:
        """Read a bounded transcript window in chronological order."""
        ...

    @abstractmethod
    def save_conversation_summary(
        self,
        session_id: str,
        through_message_id: int,
        summary: str,
        *,
        updated_at: datetime | None = None,
        fencing_token: int | None = None,
    ) -> SessionConversationSummary:
        """Persist caller-produced state through one message frontier."""
        ...

    @abstractmethod
    def get_conversation_summary(
        self, session_id: str
    ) -> SessionConversationSummary | None:
        """Read the latest compact transcript state."""
        ...

    @abstractmethod
    def refresh_conversation_summary(
        self,
        session_id: str,
        summarize: Callable[
            [SessionConversationSummary | None, list[SessionMessage]], str
        ],
        *,
        fencing_token: int | None = None,
    ) -> SessionConversationSummary | None:
        """Refresh compact state from the bounded unseen message suffix."""
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        ...

    @abstractmethod
    def list_sessions(
        self,
        where: WhereFilter | None = None,
        order_by: str = "last_active",
        descending: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Session]:
        """List sessions with filtering."""
        ...

    def archive_session(self, session_id: str) -> bool:
        """Archive a session by updating its status."""
        session = self.get_session(session_id)
        if not session:
            return False
        from src.session.models import SessionStatus

        session.status = SessionStatus.ARCHIVED
        self.update_session(session)
        return True

    @abstractmethod
    def add_document(self, document: SessionDocument) -> SessionDocument:
        """Add a document."""
        ...

    @abstractmethod
    def get_documents(self, session_id: str) -> list[SessionDocument]:
        """Get documents for a session."""
        ...

    @abstractmethod
    def update_document(self, document: SessionDocument) -> SessionDocument:
        """Update a document."""
        ...

    @abstractmethod
    def add_finding(self, finding: Finding) -> Finding:
        """Add a finding."""
        ...

    @abstractmethod
    def get_findings(self, session_id: str) -> list[Finding]:
        """Get findings for a session."""
        ...

    @abstractmethod
    def update_finding(self, finding: Finding) -> Finding:
        """Update a finding."""
        ...

    @abstractmethod
    def delete_finding(self, finding_id: str) -> bool:
        """Delete a finding."""
        ...

    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        """Save a checkpoint."""
        ...

    @abstractmethod
    def get_latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        """Get latest checkpoint."""
        ...

    @abstractmethod
    def get_checkpoints(self, session_id: str, limit: int = 10) -> list[Checkpoint]:
        """Get checkpoints for a session."""
        ...

    @abstractmethod
    def search_sessions(
        self,
        query: str,
        search_in: list[str] | None = None,
        limit: int = 10,
    ) -> list[Session]:
        """Search sessions."""
        ...

    @abstractmethod
    def search_findings(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[Finding]:
        """Search findings."""
        ...

    def build_resume_context(self, session_id: str) -> ResumeContext | None:
        """Build resume context with document change detection."""
        from pathlib import Path
        from src.session.models import DocumentChangeInfo

        session = self.get_session(session_id)
        if not session:
            return None

        documents = self.get_documents(session_id)
        findings = self.get_findings(session_id)

        # Check for document changes
        document_changes: list[DocumentChangeInfo] = []
        warnings: list[str] = []

        for doc in documents:
            path = Path(doc.file_path)
            if not path.exists():
                document_changes.append(
                    DocumentChangeInfo(
                        file_path=doc.file_path,
                        old_hash=doc.file_hash,
                        new_hash=None,
                        exists=False,
                    )
                )
                warnings.append(f"Source file missing: {doc.file_path}")
            else:
                current_hash = SessionDocument.compute_file_hash(path)
                if current_hash != doc.file_hash:
                    document_changes.append(
                        DocumentChangeInfo(
                            file_path=doc.file_path,
                            old_hash=doc.file_hash,
                            new_hash=current_hash,
                            exists=True,
                        )
                    )
                    warnings.append(f"Source file changed: {doc.file_path}")

        # Build context
        checkpoint = self.get_latest_checkpoint(session_id)
        context = ResumeContext(
            session=session,
            documents=documents,
            findings=findings,
            document_changes=document_changes,
            warnings=warnings,
            context_summary="",  # Will be formatted below
            checkpoint=checkpoint,
        )
        context.context_summary = context.format_for_injection()

        return context

    @abstractmethod
    def store_embedding(
        self,
        session_id: str,
        embedding: np.ndarray,
        content_type: str = "session",
    ) -> int:
        """Store an embedding."""
        ...

    @abstractmethod
    def search_by_embedding(
        self,
        embedding: np.ndarray,
        content_type: str = "session",
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Search by embedding."""
        ...

    @abstractmethod
    def add_tag(self, session_id: str, tag: str) -> bool:
        """Add a tag."""
        ...

    @abstractmethod
    def remove_tag(self, session_id: str, tag: str) -> bool:
        """Remove a tag."""
        ...

    @abstractmethod
    def get_sessions_by_tag(self, tag: str) -> list[Session]:
        """Get sessions by tag."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the store."""
        ...
