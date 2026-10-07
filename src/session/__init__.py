"""Session persistence module for conversation continuity.

This module provides:
- SessionStore protocol (ChromaDB-ready abstract interface)
- SQLiteSessionStore implementation
- Session, Finding, Checkpoint models
- DocumentCache for OCR result caching
- SessionPersister for checkpoint/resume logic
- IdleMonitor for background session monitoring

Storage location: /workspace/orchestration/repl_memory/sessions/
"""

from __future__ import annotations

import logging

from src.session.document_cache import DocumentCache, get_document_cache
from src.session.models import (
    Checkpoint,
    DocumentChangeInfo,
    Finding,
    FindingSource,
    ResumeContext,
    Session,
    SessionMessage,
    SessionDocument,
    SessionStatus,
)
from src.session.persister import IdleMonitor, SessionPersister
from src.session.protocol import SessionStore
from src.session.sqlite_store import SQLiteSessionStore

logger = logging.getLogger(__name__)

__all__ = [
    # Protocol
    "SessionStore",
    # Implementation
    "SQLiteSessionStore",
    # Document caching
    "DocumentCache",
    "get_document_cache",
    # Persistence management
    "SessionPersister",
    "IdleMonitor",
    # Models
    "Session",
    "SessionMessage",
    "SessionDocument",
    "SessionStatus",
    "Finding",
    "FindingSource",
    "Checkpoint",
    "ResumeContext",
    "DocumentChangeInfo",
]
