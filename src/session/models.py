"""Session persistence data models.

Dataclasses for session state, documents, findings, and checkpoints.
All models support JSON serialization via to_dict/from_dict methods.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import get_config

logger = logging.getLogger(__name__)
_lifecycle_cfg = get_config().session_lifecycle
# Elision caps for the resume summary injected into the model's context.
# Defaults preserve historical behaviour (12/8) — raising them changes live
# prompt content, so that is a measured decision, not a default flip. See
# handoffs/active/repl-session-memory-maturity.md D-e.
MAX_RESUME_VARIABLES = 12
MAX_RESUME_SKIPPED = 8

_ACTIVE_TO_IDLE_HOURS = _lifecycle_cfg.active_to_idle_hours
_IDLE_TO_STALE_HOURS = _lifecycle_cfg.idle_to_stale_days * 24.0


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"  # Recently used
    IDLE = "idle"  # Between active and stale thresholds
    STALE = "stale"  # Exceeds stale threshold (shows "welcome back" summary)
    ARCHIVED = "archived"  # Explicitly archived (cold storage)


class FindingSource(str, Enum):
    """How a finding was created."""

    USER_MARKED = "user_marked"  # User explicitly marked via mark_finding()
    LLM_EXTRACTED = "llm_extracted"  # Auto-extracted by LLM
    HEURISTIC = "heuristic"  # Rule-based extraction


@dataclass
class SessionMessage:
    """One durable message within a session turn.

    Message history is retained with its session: archiving keeps it, while
    explicit session deletion removes it. ``display`` is structured JSON and
    remains separate from text that may be spoken.
    """

    id: int
    session_id: str
    turn_id: str
    role: str
    text: str
    spoken_text: str | None
    display: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


@dataclass
class SessionDocument:
    """A document processed within a session.

    Tracks source file hash for change detection on resume.
    """

    id: str
    session_id: str
    file_path: str
    file_hash: str  # SHA-256 of file contents
    processed_at: datetime
    total_pages: int = 0
    cache_path: str | None = None  # Relative path to OCR cache

    @staticmethod
    def compute_file_hash(path: str | Path) -> str:
        """Compute SHA-256 hash of file contents."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "processed_at": self.processed_at.isoformat(),
            "total_pages": self.total_pages,
            "cache_path": self.cache_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionDocument:
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            processed_at=datetime.fromisoformat(data["processed_at"]),
            total_pages=data.get("total_pages", 0),
            cache_path=data.get("cache_path"),
        )


@dataclass
class Finding:
    """A key finding or insight from a session.

    Findings are user-curated or LLM-extracted important facts
    that should persist across session resumes.
    """

    id: str
    session_id: str
    content: str
    source: FindingSource
    created_at: datetime
    confidence: float = 1.0  # 1.0 for user-marked, 0-1 for LLM-extracted
    confirmed: bool = False  # True if user confirmed LLM extraction
    tags: list[str] = field(default_factory=list)
    # Source reference (optional)
    source_file: str | None = None
    source_section_id: str | None = None
    source_page: int | None = None
    source_turn: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content": self.content,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
            "confidence": self.confidence,
            "confirmed": self.confirmed,
            "tags": self.tags,
            "source_file": self.source_file,
            "source_section_id": self.source_section_id,
            "source_page": self.source_page,
            "source_turn": self.source_turn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            content=data["content"],
            source=FindingSource(data["source"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            confidence=data.get("confidence", 1.0),
            confirmed=data.get("confirmed", False),
            tags=data.get("tags", []),
            source_file=data.get("source_file"),
            source_section_id=data.get("source_section_id"),
            source_page=data.get("source_page"),
            source_turn=data.get("source_turn"),
        )


@dataclass
class Session:
    """A conversation session with persistence metadata.

    Sessions track:
    - Basic metadata (name, timestamps, message count)
    - Project/tag associations
    - MemRL integration (task_id, resume lineage)
    - Document references
    """

    id: str
    name: str | None = None
    project: str | None = None
    tags: list[str] = field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_checkpoint_at: datetime | None = None
    # Conversation tracking
    message_count: int = 0
    working_directory: str | None = None
    # MemRL integration
    task_id: str | None = None  # Links to episodic memory
    resume_count: int = 0
    lineage: list[str] = field(default_factory=list)  # [task_id_v1, task_id_v2, ...]
    # Content embedding (for future ChromaDB semantic search)
    embedding_id: int | None = None  # Index into embeddings array
    # Summary (generated after 2hr idle)
    summary: str | None = None
    last_topic: str | None = None

    @classmethod
    def create(
        cls,
        name: str | None = None,
        project: str | None = None,
        working_directory: str | None = None,
    ) -> Session:
        """Create a new session with generated ID."""
        session_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        return cls(
            id=session_id,
            name=name,
            project=project,
            working_directory=working_directory,
            task_id=task_id,
            lineage=[task_id],
        )

    def fork_task_id(self) -> str:
        """Create a new task_id for resume, preserving lineage.

        Returns the new task_id for MemRL tracking.
        """
        self.resume_count += 1
        new_task_id = f"{self.lineage[0]}__r{self.resume_count}"
        self.lineage.append(new_task_id)
        self.task_id = new_task_id
        return new_task_id

    def update_activity(self) -> None:
        """Update last_active timestamp and recalculate status."""
        self.last_active = datetime.now(timezone.utc)
        self._update_status()

    def _update_status(self) -> None:
        """Update status based on time since last activity."""
        if self.status == SessionStatus.ARCHIVED:
            return  # Don't auto-change archived sessions

        now = datetime.now(timezone.utc)
        last_active = self.last_active
        if last_active.tzinfo is None:
            # Backward compatibility for legacy naive timestamps persisted as UTC.
            last_active = last_active.replace(tzinfo=timezone.utc)
        idle_hours = (now - last_active).total_seconds() / 3600

        if idle_hours < _ACTIVE_TO_IDLE_HOURS:
            self.status = SessionStatus.ACTIVE
        elif idle_hours < _IDLE_TO_STALE_HOURS:
            self.status = SessionStatus.IDLE
        else:
            self.status = SessionStatus.STALE

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "project": self.project,
            "tags": self.tags,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "last_checkpoint_at": (
                self.last_checkpoint_at.isoformat() if self.last_checkpoint_at else None
            ),
            "message_count": self.message_count,
            "working_directory": self.working_directory,
            "task_id": self.task_id,
            "resume_count": self.resume_count,
            "lineage": self.lineage,
            "embedding_id": self.embedding_id,
            "summary": self.summary,
            "last_topic": self.last_topic,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            name=data.get("name"),
            project=data.get("project"),
            tags=data.get("tags", []),
            status=SessionStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            last_checkpoint_at=(
                datetime.fromisoformat(data["last_checkpoint_at"])
                if data.get("last_checkpoint_at")
                else None
            ),
            message_count=data.get("message_count", 0),
            working_directory=data.get("working_directory"),
            task_id=data.get("task_id"),
            resume_count=data.get("resume_count", 0),
            lineage=data.get("lineage", []),
            embedding_id=data.get("embedding_id"),
            summary=data.get("summary"),
            last_topic=data.get("last_topic"),
        )


@dataclass
class Checkpoint:
    """A snapshot of REPL state for crash recovery.

    Checkpoints are created:
    - Every 5 conversation turns
    - After 30 minutes of idle time
    - On explicit /save command
    """

    id: str
    session_id: str
    created_at: datetime
    # REPL state
    context_hash: str  # SHA-256 of context string
    artifacts: dict[str, Any]  # JSON-serializable artifacts only
    execution_count: int
    exploration_calls: int
    # Metadata
    message_count: int
    trigger: str  # "turns", "idle", "explicit", "summary"
    user_globals: dict[str, Any] = field(default_factory=dict)
    variable_lineage: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped_user_globals: list[str] = field(default_factory=list)
    # Signed, allowlist-checked pickles for values JSON cannot hold (D-a).
    # {name: envelope} — see src/repl_environment/safe_pickle.py.
    pickled_globals: dict[str, Any] = field(default_factory=dict)
    # Session checkpoint payload protocol version for restore compatibility.
    # 0 indicates legacy payloads that did not persist explicit version metadata.
    protocol_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "context_hash": self.context_hash,
            "artifacts": self.artifacts,
            "execution_count": self.execution_count,
            "exploration_calls": self.exploration_calls,
            "message_count": self.message_count,
            "trigger": self.trigger,
            "user_globals": self.user_globals,
            "variable_lineage": self.variable_lineage,
            "skipped_user_globals": self.skipped_user_globals,
            "pickled_globals": self.pickled_globals,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            context_hash=data["context_hash"],
            artifacts=data["artifacts"],
            execution_count=data["execution_count"],
            exploration_calls=data["exploration_calls"],
            message_count=data["message_count"],
            trigger=data["trigger"],
            user_globals=data.get("user_globals", {}),
            variable_lineage=data.get("variable_lineage", {}),
            skipped_user_globals=data.get("skipped_user_globals", []),
            pickled_globals=data.get("pickled_globals", {}) or {},
            protocol_version=int(data.get("protocol_version", 0) or 0),
        )


@dataclass
class DocumentChangeInfo:
    """Information about a changed source document."""

    file_path: str
    old_hash: str
    new_hash: str | None  # None if file no longer exists
    exists: bool


@dataclass
class ResumeContext:
    """Context injection payload for session resume.

    Returned by POST /sessions/{id}/resume to provide
    the LLM with relevant context from the previous session.
    """

    session: Session
    documents: list[SessionDocument]
    findings: list[Finding]
    # Changes since last session
    document_changes: list[DocumentChangeInfo]
    warnings: list[str]
    # Context for LLM
    context_summary: str  # Formatted summary for injection
    checkpoint: Checkpoint | None = None
    last_exchanges: list[dict[str, Any]] | None = None  # Optional conversation history

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "session": self.session.to_dict(),
            "documents": [d.to_dict() for d in self.documents],
            "findings": [f.to_dict() for f in self.findings],
            "document_changes": [
                {
                    "file_path": c.file_path,
                    "old_hash": c.old_hash,
                    "new_hash": c.new_hash,
                    "exists": c.exists,
                }
                for c in self.document_changes
            ],
            "warnings": self.warnings,
            "context_summary": self.context_summary,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "last_exchanges": self.last_exchanges,
        }

    def format_for_injection(
        self,
        max_variables: int = MAX_RESUME_VARIABLES,
        max_skipped: int = MAX_RESUME_SKIPPED,
    ) -> str:
        """Format context for LLM context injection.

        Returns a markdown-formatted string suitable for prepending
        to the conversation context.

        Args:
            max_variables: Restored variables to list before eliding. Pass 0 for
                no limit — a resumed agent that cannot see a variable it saved
                will re-derive it, which costs more than the listing does.
            max_skipped: Non-serializable variable names to list before eliding.
        """
        lines = [
            f"# Session Resumed: {self.session.name or self.session.id[:8]}",
            f"Last active: {self.session.last_active.strftime('%Y-%m-%d %H:%M')} "
            f"({self.session.message_count} messages)",
            "",
        ]

        # Documents
        if self.documents:
            lines.append("## Documents")
            for doc in self.documents:
                status = "processed"
                for change in self.document_changes:
                    if change.file_path == doc.file_path:
                        status = "CHANGED" if change.exists else "MISSING"
                        break
                lines.append(f"- {doc.file_path} ({doc.total_pages} pages, {status})")
            lines.append("")

        # Key findings
        if self.findings:
            lines.append("## Key Findings from Previous Session")
            for i, finding in enumerate(self.findings[:10], 1):  # Limit to 10
                lines.append(f"{i}. {finding.content}")
            if len(self.findings) > 10:
                lines.append(f"... and {len(self.findings) - 10} more findings")
            lines.append("")

        # Last topic
        if self.session.last_topic:
            lines.append("## Last Conversation Topic")
            lines.append(self.session.last_topic)
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        # Restored variables summary
        if self.checkpoint and self.checkpoint.user_globals:
            lines.append("## Variables (from previous request)")
            items = list(self.checkpoint.user_globals.items())
            # Curated variables (the agent explicitly called remember()) sort
            # first, so the elision cap never drops something it was told matters.
            lineage_all = self.checkpoint.variable_lineage
            items.sort(key=lambda kv: not lineage_all.get(kv[0], {}).get("curated", False))
            shown = items if max_variables <= 0 else items[:max_variables]
            for key, value in shown:
                lineage = self.checkpoint.variable_lineage.get(key, {})
                role = lineage.get("role", "unknown")
                value_type = type(value).__name__
                mark = "★ " if lineage.get("curated") else ""
                note = lineage.get("note")
                suffix = f" — {note}" if note else ""
                lines.append(f"- {mark}`{key}` ({value_type}, role={role}){suffix}")
            if len(items) > len(shown):
                lines.append(f"... and {len(items) - len(shown)} more variables")
            skipped = self.checkpoint.skipped_user_globals
            if skipped:
                shown_skipped = skipped if max_skipped <= 0 else skipped[:max_skipped]
                suffix = (
                    f" (and {len(skipped) - len(shown_skipped)} more)"
                    if len(skipped) > len(shown_skipped)
                    else ""
                )
                lines.append(
                    "Skipped non-serializable variables: "
                    f"{', '.join(shown_skipped)}{suffix}"
                )
            lines.append("")

        return "\n".join(lines)
