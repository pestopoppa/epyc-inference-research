"""Session persistence manager with checkpoint triggers.

Handles:
- Checkpoint triggers (every 5 turns, 30 min idle, explicit)
- Auto-summary generation after 2hr idle
- Session lifecycle events for MemRL integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from src.session.models import Checkpoint, Finding, FindingSource, Session
from src.config import get_config
from src.repl_environment.types import REPLTimeout

if TYPE_CHECKING:
    from src.session.sqlite_store import SQLiteSessionStore
    from src.repl_environment import REPLEnvironment
    from orchestration.repl_memory.progress_logger import ProgressLogger

logger = logging.getLogger(__name__)


# Checkpoint trigger thresholds (centralized tunables from config)
_persist_cfg = get_config().session_persistence
CHECKPOINT_TURN_INTERVAL = _persist_cfg.checkpoint_turn_interval
CHECKPOINT_IDLE_MINUTES = _persist_cfg.checkpoint_idle_minutes
SUMMARY_IDLE_HOURS = _persist_cfg.summary_idle_hours
CHECKPOINT_GLOBALS_WARN_BYTES = int(_persist_cfg.checkpoint_globals_warn_mb) * 1024 * 1024
CHECKPOINT_GLOBALS_HARD_BYTES = int(_persist_cfg.checkpoint_globals_hard_mb) * 1024 * 1024


class SessionPersister:
    """Manages session checkpoint and summary lifecycle.

    Responsibilities:
    - Track turn count and trigger checkpoints
    - Monitor idle time for auto-checkpoint and auto-summary
    - Store checkpoints to SessionStore
    - Generate context injection for session resume

    Usage:
        persister = SessionPersister(session_store, session_id)

        # After each REPL turn
        persister.on_turn(repl_env)

        # Check if checkpoint needed
        if persister.should_checkpoint():
            persister.save_checkpoint(repl_env)

        # For idle monitoring (call periodically)
        persister.check_idle()
    """

    def __init__(
        self,
        session_store: SQLiteSessionStore,
        session_id: str,
        llm_summarizer: Callable[[str], str] | None = None,
        progress_logger: ProgressLogger | None = None,
    ):
        """Initialize persister for a session.

        Args:
            session_store: The session store for persistence.
            session_id: ID of the session to manage.
            llm_summarizer: Optional function to generate LLM summaries.
                            Takes context string, returns summary string.
            progress_logger: Optional ProgressLogger for MemRL integration.
        """
        self.session_store = session_store
        self.session_id = session_id
        self.llm_summarizer = llm_summarizer
        self.progress_logger = progress_logger

        # State tracking
        self._turn_count = 0
        self._last_checkpoint_turn = 0
        self._last_activity = time.time()
        self._last_checkpoint_time = time.time()
        self._summary_generated = False

        # Get session for task_id tracking
        session = session_store.get_session(session_id)
        self._task_id = session.task_id if session else session_id

    def on_turn(self, repl_env: REPLEnvironment | None = None) -> dict[str, Any]:
        """Called after each REPL turn to update state.

        Args:
            repl_env: Optional REPL environment for checkpoint data.

        Returns:
            Dict with action taken (if any).
        """
        self._turn_count += 1
        self._last_activity = time.time()

        # Update session activity
        session = self.session_store.get_session(self.session_id)
        if session:
            session.update_activity()
            session.message_count = self._turn_count
            self.session_store.update_session(session)

        result = {"turn": self._turn_count, "action": None}

        # Check if checkpoint needed
        if self.should_checkpoint() and repl_env is not None:
            self.save_checkpoint(repl_env, trigger="turns")
            result["action"] = "checkpoint"

        return result

    def should_checkpoint(self) -> bool:
        """Check if a checkpoint should be created.

        Returns:
            True if checkpoint criteria met.
        """
        # Turn-based trigger
        turns_since_checkpoint = self._turn_count - self._last_checkpoint_turn
        if turns_since_checkpoint >= CHECKPOINT_TURN_INTERVAL:
            return True

        # Time-based trigger (idle)
        idle_seconds = time.time() - self._last_checkpoint_time
        if idle_seconds >= CHECKPOINT_IDLE_MINUTES * 60:
            return True

        return False

    def save_checkpoint(
        self,
        repl_env: REPLEnvironment,
        trigger: str = "explicit",
    ) -> Checkpoint:
        """Save a checkpoint of the current REPL state.

        Args:
            repl_env: The REPL environment to checkpoint.
            trigger: What triggered the checkpoint ("turns", "idle", "explicit", "summary").

        Returns:
            The saved Checkpoint.
        """
        if getattr(repl_env, "timed_out", False) is True:
            timeout_error = getattr(repl_env, "timeout_error", None)
            if not isinstance(timeout_error, str):
                timeout_error = "REPLTimeout: Execution timed out"
            raise REPLTimeout(timeout_error.removeprefix("REPLTimeout: "))

        # Get REPL checkpoint data
        repl_checkpoint = repl_env.checkpoint()

        # Compute context hash (for integrity verification)
        context_hash = hashlib.sha256(repl_env.context.encode()).hexdigest()[:16]
        user_globals = dict(repl_checkpoint.get("user_globals", {}))
        pickled_globals = dict(repl_checkpoint.get("pickled_globals", {}))
        variable_lineage = dict(repl_checkpoint.get("variable_lineage", {}))
        skipped_user_globals = list(repl_checkpoint.get("skipped_user_globals", []))
        evicted_user_globals: list[str] = []

        def checkpoint_payload_size() -> int:
            return len(
                json.dumps(
                    {"user_globals": user_globals, "pickled_globals": pickled_globals},
                    default=str,
                ).encode("utf-8")
            )

        payload_size = checkpoint_payload_size()
        if payload_size >= CHECKPOINT_GLOBALS_WARN_BYTES:
            logger.warning(
                "Checkpoint globals payload is large: %.1fMB for session=%s",
                payload_size / (1024 * 1024),
                self.session_id[:8],
            )
        if payload_size > CHECKPOINT_GLOBALS_HARD_BYTES and (user_globals or pickled_globals):
            ordered = sorted(
                user_globals.keys() | pickled_globals.keys(),
                key=lambda k: (
                    float(variable_lineage.get(k, {}).get("saved_at_ts", 0.0)),
                    k,
                ),
            )
            while ordered and payload_size > CHECKPOINT_GLOBALS_HARD_BYTES:
                victim = ordered.pop(0)
                user_globals.pop(victim, None)
                pickled_globals.pop(victim, None)
                variable_lineage.pop(victim, None)
                evicted_user_globals.append(victim)
                payload_size = checkpoint_payload_size()
            if evicted_user_globals:
                logger.warning(
                    "Evicted %d globals to enforce checkpoint size cap for session=%s",
                    len(evicted_user_globals),
                    self.session_id[:8],
                )
                skipped_user_globals.extend(evicted_user_globals)

        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            created_at=datetime.now(timezone.utc),
            context_hash=f"sha256:{context_hash}",
            artifacts=repl_checkpoint.get("artifacts", {}),
            execution_count=repl_checkpoint.get("execution_count", 0),
            exploration_calls=repl_checkpoint.get("exploration_calls", 0),
            message_count=self._turn_count,
            trigger=trigger,
            user_globals=user_globals,
            pickled_globals=pickled_globals,
            variable_lineage=variable_lineage,
            skipped_user_globals=skipped_user_globals,
        )

        # Save to store
        self.session_store.save_checkpoint(checkpoint)

        # Sync findings from REPL to store
        synced_findings = self.sync_findings(repl_env)

        # Update tracking state
        self._last_checkpoint_turn = self._turn_count
        self._last_checkpoint_time = time.time()

        # Emit event to ProgressLogger
        if self.progress_logger is not None:
            self.progress_logger.log_session_checkpointed(
                session_id=self.session_id,
                task_id=self._task_id,
                checkpoint_id=checkpoint.id,
                trigger=trigger,
                message_count=self._turn_count,
                findings_synced=synced_findings,
            )

        logger.info(
            f"Checkpoint saved: session={self.session_id[:8]}, "
            f"trigger={trigger}, turn={self._turn_count}, findings_synced={synced_findings}"
        )

        return checkpoint

    def sync_findings(self, repl_env: REPLEnvironment) -> int:
        """Sync findings from REPL buffer to SessionStore.

        Converts REPL findings to Finding models and stores them.
        Clears the REPL findings buffer after successful sync.

        Args:
            repl_env: The REPL environment with findings.

        Returns:
            Number of findings synced.
        """
        repl_findings = repl_env.get_findings()
        if not repl_findings:
            return 0

        synced = 0
        for f in repl_findings:
            source_ctx = f.get("source", {})
            finding = Finding(
                id=f["id"],
                session_id=self.session_id,
                content=f["content"],
                source=FindingSource.USER_MARKED,
                created_at=datetime.fromtimestamp(f["timestamp"]),
                confidence=1.0,  # User-marked = full confidence
                confirmed=True,
                tags=f.get("tags", []),
                source_file=source_ctx.get("file"),
                source_page=source_ctx.get("page"),
                source_section_id=source_ctx.get("section"),
                source_turn=f.get("turn"),
            )
            self.session_store.add_finding(finding)
            synced += 1

            # Emit finding event to ProgressLogger
            if self.progress_logger is not None:
                self.progress_logger.log_session_finding(
                    session_id=self.session_id,
                    task_id=self._task_id,
                    finding_id=finding.id,
                    source=finding.source.value,
                    confidence=finding.confidence,
                    tags=finding.tags,
                )

        # Clear REPL buffer after successful sync
        repl_env.clear_findings()

        logger.debug(f"Synced {synced} findings to session {self.session_id[:8]}")
        return synced

    def extract_heuristic_findings(
        self,
        text: str,
        source_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract potential findings from text using heuristics.

        Looks for patterns like:
        - "KEY:" or "KEY FINDING:" prefixes
        - "IMPORTANT:" prefix
        - "FINDING:" prefix
        - "NOTE:" prefix (lower confidence)
        - "CONCLUSION:" prefix

        Args:
            text: Text to scan for findings (e.g., REPL output).
            source_context: Optional context (file, page, section).

        Returns:
            List of extracted finding dicts with content and confidence.
        """
        import re

        findings = []

        # Patterns with associated confidence levels
        patterns = [
            (r"(?:KEY(?:\s+FINDING)?|FINDING|IMPORTANT|CONCLUSION)\s*:\s*(.+?)(?:\n|$)", 0.8),
            (r"NOTE\s*:\s*(.+?)(?:\n|$)", 0.5),
            (r"\*\*([^*]+)\*\*", 0.4),  # **emphasized text** (lower confidence)
        ]

        for pattern, confidence in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                content = match.group(1).strip()
                if len(content) >= 10:  # Skip very short matches
                    findings.append(
                        {
                            "content": content,
                            "confidence": confidence,
                            "source": source_context or {},
                            "pattern": pattern[:30],  # For debugging
                        }
                    )

        # Deduplicate by content
        seen = set()
        unique_findings = []
        for f in findings:
            content_key = f["content"].lower()[:100]
            if content_key not in seen:
                seen.add(content_key)
                unique_findings.append(f)

        return unique_findings

    def add_heuristic_findings(
        self,
        text: str,
        source_context: dict[str, Any] | None = None,
        min_confidence: float = 0.5,
    ) -> int:
        """Extract and store heuristic findings from text.

        Findings are stored with source=HEURISTIC and confirmed=False,
        allowing users to review and confirm them later.

        Args:
            text: Text to scan for findings.
            source_context: Optional context (file, page, section).
            min_confidence: Minimum confidence to include (default 0.5).

        Returns:
            Number of findings added.
        """
        extracted = self.extract_heuristic_findings(text, source_context)

        added = 0
        for f in extracted:
            if f["confidence"] < min_confidence:
                continue

            source_ctx = f.get("source", {})
            finding = Finding(
                id=str(uuid.uuid4()),
                session_id=self.session_id,
                content=f["content"],
                source=FindingSource.HEURISTIC,
                created_at=datetime.now(timezone.utc),
                confidence=f["confidence"],
                confirmed=False,  # Needs user review
                tags=[],
                source_file=source_ctx.get("file"),
                source_page=source_ctx.get("page"),
                source_section_id=source_ctx.get("section"),
            )
            self.session_store.add_finding(finding)
            added += 1

        if added > 0:
            logger.info(f"Added {added} heuristic findings to session {self.session_id[:8]}")

        return added

    def check_idle(self) -> dict[str, Any]:
        """Check idle time and trigger actions if needed.

        Call this periodically (e.g., every minute) to handle:
        - Auto-checkpoint after 30 min idle
        - Auto-summary after 2hr idle

        Returns:
            Dict with actions taken.
        """
        result = {"actions": []}
        now = time.time()
        idle_seconds = now - self._last_activity

        # 30 min idle -> checkpoint
        if idle_seconds >= CHECKPOINT_IDLE_MINUTES * 60:
            checkpoint_age = now - self._last_checkpoint_time
            if checkpoint_age >= CHECKPOINT_IDLE_MINUTES * 60:
                # Can't create checkpoint without REPL, but mark that it's needed
                result["actions"].append("checkpoint_needed")
                result["idle_minutes"] = idle_seconds / 60

        # 2 hr idle -> summary
        if idle_seconds >= SUMMARY_IDLE_HOURS * 3600:
            if not self._summary_generated:
                self._generate_summary()
                result["actions"].append("summary_generated")
                self._summary_generated = True

        return result

    def _generate_summary(self) -> str | None:
        """Generate an LLM summary for the session.

        Uses the llm_summarizer callback if provided, otherwise
        generates a heuristic summary from session data.

        Returns:
            The generated summary, or None if generation failed.
        """
        session = self.session_store.get_session(self.session_id)
        if not session:
            return None

        # Get findings for context
        findings = self.session_store.get_findings(self.session_id)
        findings_text = "\n".join(f"- {f.content}" for f in findings[:10])

        if self.llm_summarizer is not None:
            # Build context for LLM
            context = f"""Session: {session.name or session.id[:8]}
Messages: {session.message_count}
Duration: {(session.last_active - session.created_at).total_seconds() / 3600:.1f} hours

Key Findings:
{findings_text if findings_text else "(none recorded)"}

Last Topic: {session.last_topic or "(not recorded)"}
"""
            try:
                summary = self.llm_summarizer(context)
            except Exception as e:
                logger.warning(f"LLM summary generation failed: {e}")
                summary = self._heuristic_summary(session, findings)
        else:
            summary = self._heuristic_summary(session, findings)

        # Save summary to session
        session.summary = summary
        self.session_store.update_session(session)

        logger.info(f"Summary generated for session {self.session_id[:8]}")
        return summary

    def _heuristic_summary(self, session: Session, findings: list) -> str:
        """Generate a simple heuristic summary.

        Args:
            session: The session to summarize.
            findings: List of findings from the session.

        Returns:
            A brief summary string.
        """
        duration = (session.last_active - session.created_at).total_seconds() / 3600
        parts = [
            f"Session with {session.message_count} messages over {duration:.1f} hours.",
        ]

        if session.last_topic:
            parts.append(f"Last topic: {session.last_topic}")

        if findings:
            parts.append(f"{len(findings)} key findings recorded.")

        return " ".join(parts)

    def get_status(self) -> dict[str, Any]:
        """Get current persister status.

        Returns:
            Dict with status information.
        """
        now = time.time()
        return {
            "session_id": self.session_id,
            "turn_count": self._turn_count,
            "turns_since_checkpoint": self._turn_count - self._last_checkpoint_turn,
            "idle_seconds": now - self._last_activity,
            "seconds_since_checkpoint": now - self._last_checkpoint_time,
            "summary_generated": self._summary_generated,
            "checkpoint_threshold_turns": CHECKPOINT_TURN_INTERVAL,
            "checkpoint_threshold_idle_min": CHECKPOINT_IDLE_MINUTES,
            "summary_threshold_idle_hours": SUMMARY_IDLE_HOURS,
        }

    # =========================================================================
    # MemRL Integration - Lifecycle Event Emitters
    # =========================================================================

    def emit_session_created(self) -> None:
        """Emit session created event to ProgressLogger.

        Call this when a new session is created.
        """
        if self.progress_logger is None:
            return

        session = self.session_store.get_session(self.session_id)
        if session:
            self.progress_logger.log_session_created(
                session_id=self.session_id,
                task_id=self._task_id,
                name=session.name,
                project=session.project,
            )

    def emit_session_resumed(
        self,
        previous_task_id: str,
        document_changes: int = 0,
    ) -> None:
        """Emit session resumed event to ProgressLogger.

        Call this when a session is resumed.

        Args:
            previous_task_id: Task ID before resume (before fork).
            document_changes: Number of source documents that changed.
        """
        if self.progress_logger is None:
            return

        session = self.session_store.get_session(self.session_id)
        if session:
            # Update task_id to the new forked value
            self._task_id = session.task_id

            self.progress_logger.log_session_resumed(
                session_id=self.session_id,
                task_id=self._task_id,
                previous_task_id=previous_task_id,
                resume_count=session.resume_count,
                message_count=session.message_count,
                document_changes=document_changes,
            )

    def emit_session_archived(self) -> None:
        """Emit session archived event to ProgressLogger.

        Call this when a session is archived.
        """
        if self.progress_logger is None:
            return

        session = self.session_store.get_session(self.session_id)
        findings = self.session_store.get_findings(self.session_id)

        if session:
            self.progress_logger.log_session_archived(
                session_id=self.session_id,
                task_id=self._task_id,
                message_count=session.message_count,
                findings_count=len(findings),
                summary_generated=session.summary is not None,
            )


class IdleMonitor:
    """Background monitor for session idle time.

    Tracks multiple sessions and triggers actions when they become idle.
    Designed to be called periodically (e.g., every minute) from a
    background task or cron job.
    """

    def __init__(self, session_store: SQLiteSessionStore):
        """Initialize idle monitor.

        Args:
            session_store: The session store to monitor.
        """
        self.session_store = session_store
        self._persisters: dict[str, SessionPersister] = {}

    def get_persister(self, session_id: str) -> SessionPersister:
        """Get or create a persister for a session.

        Args:
            session_id: The session ID.

        Returns:
            SessionPersister for the session.
        """
        if session_id not in self._persisters:
            self._persisters[session_id] = SessionPersister(self.session_store, session_id)
        return self._persisters[session_id]

    def check_all_sessions(self) -> list[dict[str, Any]]:
        """Check all active sessions for idle actions.

        Returns:
            List of action results for each session.
        """
        results = []

        # Get active/idle sessions
        sessions = self.session_store.list_sessions(
            where={"status": {"$in": ["active", "idle"]}},
            limit=100,
        )

        for session in sessions:
            persister = self.get_persister(session.id)

            # Sync persister state from session
            persister._turn_count = session.message_count
            persister._last_activity = session.last_active.timestamp()
            if session.last_checkpoint_at:
                persister._last_checkpoint_time = session.last_checkpoint_at.timestamp()

            # Check for idle actions
            result = persister.check_idle()
            if result.get("actions"):
                results.append(
                    {
                        "session_id": session.id,
                        "session_name": session.name,
                        **result,
                    }
                )

        return results

    def cleanup_stale_persisters(self) -> int:
        """Remove persisters for archived/deleted sessions.

        Returns:
            Number of persisters removed.
        """
        active_sessions = {
            s.id
            for s in self.session_store.list_sessions(
                where={"status": {"$ne": "archived"}},
                limit=1000,
            )
        }

        to_remove = [sid for sid in self._persisters if sid not in active_sessions]

        for sid in to_remove:
            del self._persisters[sid]

        return len(to_remove)
