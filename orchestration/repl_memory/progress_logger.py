"""
ProgressLogger: Lightweight structured logging for orchestration events.

All tiers publish progress logs that the Q-scorer agent periodically processes.
This implements the "lab book" pattern from the async Q-scoring architecture.

Log format is JSONL (one JSON object per line) for efficient streaming reads.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.workload_model import capture_workload_class
from src.orchestration.interaction import (
    INTERACTION_POLICY_VERSION as _INTERACTION_POLICY_VERSION,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

# Serializes flushes within a process. ``flush`` runs on the event loop (buffer full,
# durable rows) AND from worker threads (Q-scoring via asyncio.to_thread / the scoring
# pool), so two unserialized flushes could write the same entries twice or drop one
# appended between the write loop and ``clear()``. ``log`` stays lock-free: flush
# removes exactly the prefix it wrote, so a concurrent append is never lost.
_FLUSH_LOCK = threading.Lock()

# A buffered entry older than this is flushed by the next ``log`` call. The API runs
# several uvicorn workers, each with its own buffer; at 20-40 tasks a day a 10-entry
# buffer could hold a worker's rows for days, and lost them outright when the process
# ended without the lifespan flush (run 1767532, 2026-10-01: six workers started, none
# shut down cleanly; two c95 tasks have task_started and no terminal row).
DEFAULT_MAX_BUFFER_AGE_S = 30.0


def _flush_at_exit(ref: "weakref.ReferenceType[ProgressLogger]") -> None:
    logger_obj = ref()
    if logger_obj is None:
        return
    try:
        logger_obj.flush()
    except Exception:
        pass

# Default log path (on RAID array, or fallback to workspace for devcontainer)
_RAID_LOG_PATH = _REPO_ROOT / "logs/progress"
_WORKSPACE_LOG_PATH = Path("/workspace/logs/progress")

# Use RAID path if available, otherwise fallback to workspace
DEFAULT_LOG_PATH = _RAID_LOG_PATH if _RAID_LOG_PATH.parent.exists() else _WORKSPACE_LOG_PATH

TASK_RECORD_SCHEMA_VERSION = "task_record.v1"
_TASK_RECORD_CACHE_LIMIT = 10_000
# Objective length retained in the progress JSONL (the offline analysis substrate).
# Deliberately larger than the live-routing TASK_IR_OBJECTIVE_LEN (200): the old
# 200-char cap merged distinct tasks sharing a prefix, which corrupts within-objective
# counterfactual/competence analysis at promotion grade (DAR handoff L493). This field
# is log-only — it is never read by routing, embedding, or reward, so raising it leaves
# live routing unaffected. Capped (not unbounded) to bound log bloat on long-context tasks.
PROGRESS_OBJECTIVE_LOG_LEN = 2000
_TOKEN_COUNT_KEYS = (
    "total_tokens",
    "tokens",
    "tokens_generated",
    "output_tokens",
    "completion_tokens",
)


def _get_fallback_log_dir() -> Path:
    """Get a fallback log directory when default paths aren't writable.

    Returns a temp directory that will be cleaned up on process exit.
    Used in CI environments where neither RAID nor workspace paths exist.
    """
    import tempfile

    return Path(tempfile.mkdtemp(prefix="progress_logs_"))


class EventType(str, Enum):
    """Types of orchestration events."""

    # Task lifecycle
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    # Routing events
    ROUTING_DECISION = "routing_decision"
    ROUTING_FALLBACK = "routing_fallback"

    # Delegation events
    DELEGATION_DECISION = "delegation_decision"

    # Escalation events
    ESCALATION_TRIGGERED = "escalation_triggered"
    ESCALATION_RESOLVED = "escalation_resolved"
    ESCALATION_FAILED = "escalation_failed"

    # Gate events
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    #: CJ-8. The gate never DECIDED — it timed out, it raised, or the gate name
    #: does not exist. Emphatically NOT a `GATE_FAILED`: `q_reward` charges -0.1
    #: per GATE_FAILED, so folding an infra failure in there converts a harness
    #: defect into negative learning signal about a model that was never checked.
    #: An infra failure is the ABSENCE of a measurement, never a bad one.
    GATE_INCONCLUSIVE = "gate_inconclusive"

    # REPL exploration events
    EXPLORATION_STARTED = "exploration_started"
    EXPLORATION_STRATEGY = "exploration_strategy"
    EXPLORATION_COMPLETED = "exploration_completed"

    # Formalizer events
    FORMALIZER_INVOKED = "formalizer_invoked"

    # Plan review events (architect-in-the-loop)
    PLAN_REVIEWED = "plan_reviewed"

    # Q-scoring events (from Q-scorer agent)
    Q_VALUE_UPDATED = "q_value_updated"
    MEMORY_STORED = "memory_stored"

    # Session lifecycle events (for session persistence)
    SESSION_CREATED = "session_created"
    SESSION_RESUMED = "session_resumed"
    SESSION_CHECKPOINTED = "session_checkpointed"
    SESSION_ARCHIVED = "session_archived"
    SESSION_FINDING_ADDED = "session_finding_added"


@dataclass
class ProgressEntry:
    """A single progress log entry."""

    event_type: EventType
    task_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_tier: Optional[str] = None  # "A", "B1", "B2", "B3", "C", "D"
    agent_role: Optional[str] = None  # "frontdoor", "coder", etc.

    # Event-specific data
    data: Dict[str, Any] = field(default_factory=dict)

    # Linking to memory
    memory_id: Optional[str] = None  # If this event is linked to a memory entry

    # Outcome (for completed events)
    outcome: Optional[str] = None  # "success", "failure", "partial"
    outcome_details: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = {
            "event_type": self.event_type.value,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_tier": self.agent_tier,
            "agent_role": self.agent_role,
            "data": self.data,
            "memory_id": self.memory_id,
            "outcome": self.outcome,
            "outcome_details": self.outcome_details,
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> ProgressEntry:
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        return cls(
            event_type=EventType(d["event_type"]),
            task_id=d["task_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            agent_tier=d.get("agent_tier"),
            agent_role=d.get("agent_role"),
            data=d.get("data", {}),
            memory_id=d.get("memory_id"),
            outcome=d.get("outcome"),
            outcome_details=d.get("outcome_details"),
        )


class ProgressLogger:
    """
    Append-only progress logger for orchestration events.

    Log files are organized by date:
    - progress/2026-01-13.jsonl
    - progress/2026-01-14.jsonl
    - ...

    Each line is a JSON object (JSONL format).
    """

    def __init__(
        self,
        log_dir: Path = DEFAULT_LOG_PATH,
        buffer_size: int = 10,  # Flush after N entries
        max_buffer_age_s: float = DEFAULT_MAX_BUFFER_AGE_S,
    ):
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self.max_buffer_age_s = max_buffer_age_s
        self._buffer: List[ProgressEntry] = []
        self._buffer_since: float | None = None  # monotonic time of oldest unflushed entry
        self._disabled = False
        # Interpreter exit without the API lifespan flush (scripts, a crashed
        # worker that still unwinds) must not drop the tail of the buffer.
        atexit.register(_flush_at_exit, weakref.ref(self))

        # Ensure log directory exists (fall back to temp dir if not writable)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fall back to temp directory (e.g., in CI environment)
            self.log_dir = _get_fallback_log_dir()
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # Can't create any directory - disable logging
                self._disabled = True

    def _get_log_path(self, dt: datetime) -> Path:
        """Get log file path for a given datetime."""
        date_str = dt.strftime("%Y-%m-%d")
        return self.log_dir / f"{date_str}.jsonl"

    def _task_record_cache(self) -> Dict[str, Dict[str, Any]]:
        cache = getattr(self, "_task_record_pending", None)
        if not isinstance(cache, dict):
            cache = {}
            self._task_record_pending = cache
        return cache

    @staticmethod
    def _text_ref(text: object) -> str | None:
        if not isinstance(text, str) or not text:
            return None
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"progress-text-sha256:{digest}"

    @staticmethod
    def _token_count(data: Dict[str, Any]) -> int | None:
        prompt_tokens = data.get("prompt_tokens")
        completion_tokens = data.get("completion_tokens")
        if (
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and prompt_tokens >= 0
            and isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
            and completion_tokens >= 0
        ):
            return prompt_tokens + completion_tokens
        for key in _TOKEN_COUNT_KEYS:
            value = data.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    @staticmethod
    def _normalize_operator_verdict(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace(" ", "_")
        return normalized or None

    @staticmethod
    def _task_workload_class(task_ir: Dict[str, Any]) -> str:
        return capture_workload_class(task_ir)

    def _remember_task_record_start(
        self,
        task_id: str,
        task_ir: Dict[str, Any],
        routing_decision: List[str],
        routing_strategy: str,
        started_at: datetime,
    ) -> None:
        cache = self._task_record_cache()
        if len(cache) >= _TASK_RECORD_CACHE_LIMIT:
            cache.pop(next(iter(cache)), None)
        objective = task_ir.get("objective", "")
        workload_class = self._task_workload_class(task_ir)
        cache[task_id] = {
            "schema_version": TASK_RECORD_SCHEMA_VERSION,
            "task_id": task_id,
            "class": task_ir.get("task_type") or "unknown",
            "workload_class": workload_class,
            "prompt_ref": self._text_ref(objective),
            "prompt_chars": len(objective) if isinstance(objective, str) else None,
            "route_taken": [str(role) for role in routing_decision],
            "routing_strategy": routing_strategy,
            "started_ts_utc": started_at.isoformat(),
            "_started_at": started_at,
        }

    def _complete_task_record(
        self,
        task_id: str,
        success: bool,
        details: Optional[str],
        completion_data: Dict[str, Any],
        completed_at: datetime,
        operator_verdict: Optional[str] = None,
        operator_verdict_details: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        record = self._task_record_cache().pop(task_id, None)
        if not record:
            return None
        started_at = record.pop("_started_at", None)
        if isinstance(started_at, datetime):
            record["wall_s"] = round(
                max(0.0, (completed_at - started_at).total_seconds()),
                3,
            )
        else:
            record["wall_s"] = None
        record["completed_ts_utc"] = completed_at.isoformat()
        record["tokens"] = self._token_count(completion_data)
        record["outcome"] = "success" if success else "failure"
        if details:
            record["outcome_details_ref"] = self._text_ref(details)
        normalized_verdict = self._normalize_operator_verdict(operator_verdict)
        if normalized_verdict:
            record["operator_verdict"] = normalized_verdict
            record["operator_verdict_source"] = "explicit_operator"
            if operator_verdict_details:
                record["operator_verdict_details_ref"] = self._text_ref(
                    operator_verdict_details
                )
        return record

    def log(self, entry: ProgressEntry) -> None:
        """
        Log a progress entry.

        Args:
            entry: Progress entry to log
        """
        if self._disabled:
            return

        now = time.monotonic()
        if not self._buffer or self._buffer_since is None:
            self._buffer_since = now
        self._buffer.append(entry)

        if len(self._buffer) >= self.buffer_size or (
            self.max_buffer_age_s is not None
            and now - self._buffer_since >= self.max_buffer_age_s
        ):
            self.flush()

    def log_durable(self, entry: ProgressEntry) -> None:
        """Log an entry that must be on disk when this call returns.

        For records other processes read back promptly — e.g. the HS-19a
        parent->child lineage row. ``log`` batches per process (``buffer_size``)
        and the API runs several uvicorn workers, each with its own buffer, so a
        low-traffic row can otherwise sit in memory until later traffic or
        shutdown. The entry is appended behind anything already buffered and the
        whole buffer is flushed, so on-disk order within this process is kept.
        Batching of ordinary ``log`` calls is unchanged.
        """
        if self._disabled:
            return
        self._buffer.append(entry)
        self.flush()

    def flush(self) -> None:
        """Flush buffered entries to disk."""
        if self._disabled or not self._buffer:
            return

        with _FLUSH_LOCK:
            # Snapshot, write, then drop exactly the written prefix: an entry a
            # concurrent ``log`` appends meanwhile stays for the next flush.
            pending = list(self._buffer)
            if not pending:
                return

            # Group entries by date
            by_date: Dict[str, List[ProgressEntry]] = {}
            for entry in pending:
                date_key = entry.timestamp.strftime("%Y-%m-%d")
                if date_key not in by_date:
                    by_date[date_key] = []
                by_date[date_key].append(entry)

            # Write to files
            for date_key, entries in by_date.items():
                log_path = self.log_dir / f"{date_key}.jsonl"
                with open(log_path, "a") as f:
                    for entry in entries:
                        f.write(entry.to_json() + "\n")

            del self._buffer[: len(pending)]
            self._buffer_since = time.monotonic() if self._buffer else None

    def log_task_started(
        self,
        task_id: str,
        task_ir: Dict[str, Any],
        routing_decision: List[str],
        routing_strategy: str,  # "learned" or "rules"
        routing_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log task start with routing decision."""
        task_started_entry = ProgressEntry(
            event_type=EventType.TASK_STARTED,
            task_id=task_id,
            data={
                "task_type": task_ir.get("task_type"),
                "workload_class": self._task_workload_class(task_ir),
                "objective": task_ir.get("objective", "")[:PROGRESS_OBJECTIVE_LOG_LEN],
                "priority": task_ir.get("priority"),
            },
        )
        self.log(task_started_entry)

        self.log(
            ProgressEntry(
                event_type=EventType.ROUTING_DECISION,
                task_id=task_id,
                data={
                    "routing": routing_decision,
                    "strategy": routing_strategy,
                    "delegation_policy_version": self.DELEGATION_POLICY_VERSION,
                    **(routing_meta or {}),
                },
            )
        )
        self._remember_task_record_start(
            task_id,
            task_ir,
            routing_decision,
            routing_strategy,
            task_started_entry.timestamp,
        )
        try:
            from src.runtime.live_telemetry import emit_lifecycle_transition

            identity = {
                "task_id": task_id,
                "request_id": (routing_meta or {}).get("request_id"),
                "batch_id": (routing_meta or {}).get("batch_id"),
                "role": str(routing_decision[0]) if routing_decision else None,
            }
            workload_class = (routing_meta or {}).get("workload_class") or self._task_workload_class(
                task_ir
            )
            emit_lifecycle_transition(
                "queued",
                source_ts=task_started_entry.timestamp.timestamp(),
                details={
                    "task_type": task_ir.get("task_type"),
                    "workload_class": workload_class,
                    "request_priority": (routing_meta or {}).get("request_priority"),
                },
                **identity,
            )
            emit_lifecycle_transition(
                "route_selected",
                details={
                    "routing": [str(role) for role in routing_decision],
                    "strategy": routing_strategy,
                },
                **identity,
            )
        except Exception:
            # Live telemetry is strictly observational and must never affect
            # durable progress logging or request execution.
            pass

    # Current policy version — bump when delegation logic changes materially
    DELEGATION_POLICY_VERSION = "1.0"
    INTERACTION_POLICY_VERSION = _INTERACTION_POLICY_VERSION

    def log_interaction(
        self,
        task_id: str,
        complexity: str,
        action: str,
        confidence: float,
        difficulty_score: float = 0.0,
        difficulty_band: str = "",
        interaction_type: str = "delegate",
    ) -> None:
        """Log proactive interaction decision for MemRL Q-learning.

        Called by ProactiveDelegator.log_delegation_decision() after routing
        a task by complexity.

        Args:
            task_id: Task identifier.
            complexity: TaskComplexity value (trivial/simple/moderate/complex).
            action: Delegation action (direct/repl/specialist/architect).
            confidence: Routing confidence from MemRL (1.0 if heuristic-only).
            difficulty_score: Prompt difficulty [0, 1] from difficulty_signal classifier.
            difficulty_band: "easy" | "medium" | "hard" from difficulty_signal classifier.
            interaction_type: delegate | consult | verify | route.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.DELEGATION_DECISION,
                task_id=task_id,
                data={
                    "complexity": complexity,
                    "action": action,
                    "confidence": confidence,
                    "interaction_type": interaction_type,
                    "interaction_policy_version": self.INTERACTION_POLICY_VERSION,
                    "delegation_policy_version": self.DELEGATION_POLICY_VERSION,
                    "difficulty_score": round(difficulty_score, 4),
                    "difficulty_band": difficulty_band,
                },
            )
        )

    def log_delegation(
        self,
        task_id: str,
        complexity: str,
        action: str,
        confidence: float,
        difficulty_score: float = 0.0,
        difficulty_band: str = "",
    ) -> None:
        """Log proactive delegation decision for MemRL Q-learning."""
        self.log_interaction(
            task_id=task_id,
            complexity=complexity,
            action=action,
            confidence=confidence,
            difficulty_score=difficulty_score,
            difficulty_band=difficulty_band,
            interaction_type="delegate",
        )

    def log_consult(
        self,
        task_id: str,
        skill: str,
        consultant_role: str,
        requester_role: str,
        confidence: float,
        *,
        outcome: str = "advisory",
        reason: str = "",
    ) -> None:
        """Log an internal consult decision/advisory event."""
        self.log(
            ProgressEntry(
                event_type=EventType.DELEGATION_DECISION,
                task_id=task_id,
                data={
                    "interaction_type": "consult",
                    "interaction_policy_version": self.INTERACTION_POLICY_VERSION,
                    "skill": skill,
                    "consultant_role": consultant_role,
                    "requester_role": requester_role,
                    "confidence": confidence,
                    "outcome": outcome,
                    "reason": reason,
                },
            )
        )

    def log_task_completed(
        self,
        task_id: str,
        success: bool,
        details: Optional[str] = None,
        completion_meta: Optional[Dict[str, Any]] = None,
        operator_verdict: Optional[str] = None,
        operator_verdict_details: Optional[str] = None,
    ) -> None:
        """Log task completion."""
        completion_data = dict(completion_meta or {})
        meta_verdict_details = completion_data.pop("operator_verdict_details", None)
        if operator_verdict is None:
            operator_verdict = completion_data.get("operator_verdict")  # type: ignore[assignment]
        normalized_operator_verdict = self._normalize_operator_verdict(operator_verdict)
        if normalized_operator_verdict:
            completion_data["operator_verdict"] = normalized_operator_verdict
            if operator_verdict_details is None and isinstance(meta_verdict_details, str):
                operator_verdict_details = meta_verdict_details
            if operator_verdict_details:
                completion_data["operator_verdict_details_ref"] = self._text_ref(
                    operator_verdict_details
                )
        try:
            # RI-16: final routing-stage latency for this task (mode, review gate and
            # total added since the routing_decision event). Context-local and
            # task-id matched, so it is only ever attached to the timed request.
            from src.runtime.routing_stage_timing import telemetry_for

            stage_telemetry = telemetry_for(task_id)
            if stage_telemetry is not None:
                completion_data.update(stage_telemetry)
        except Exception:
            pass
        completed_at = datetime.now(timezone.utc)
        task_record = self._complete_task_record(
            task_id,
            success,
            details,
            completion_data,
            completed_at,
            normalized_operator_verdict,
            operator_verdict_details,
        )
        if task_record:
            completion_data["task_record_v1"] = task_record
        # Durable: the terminal row is what the Q-scorer and every offline analysis
        # key on, and it is the row a buffered-then-killed worker used to lose.
        self.log_durable(
            ProgressEntry(
                event_type=EventType.TASK_COMPLETED if success else EventType.TASK_FAILED,
                task_id=task_id,
                timestamp=completed_at,
                data=completion_data,
                outcome="success" if success else "failure",
                outcome_details=details,
            )
        )
        try:
            from src.runtime.live_telemetry import emit_lifecycle_transition

            cached_route = (task_record or {}).get("route_taken") or []
            emit_lifecycle_transition(
                "completed" if success else "failed",
                task_id=task_id,
                role=(completion_data.get("producer_role") or (cached_route[-1] if cached_route else None)),
                source_ts=completed_at.timestamp(),
                details={"outcome": "success" if success else "failure"},
            )
        except Exception:
            pass

    def log_gate_result(
        self,
        task_id: str,
        gate_name: str,
        passed: bool,
        agent_tier: str,
        agent_role: str,
        error_message: Optional[str] = None,
        verdict: Optional[str] = None,
        cause: Optional[str] = None,
    ) -> None:
        """Log a gate result — pass, fail, or NEVER DECIDED.

        ``verdict``/``cause`` are the CJ-8 pair from ``GateResult``. They are
        optional so a caller that predates the three-valued gate keeps its exact
        previous behaviour: absent ``verdict`` falls back to ``passed``.

        ``passed`` is NOT reinterpreted. An undecided gate still arrives with
        ``passed=False`` and still blocks everywhere ``passed`` is consulted; the
        only thing that changes is which EVENT is written, and therefore whether
        the reward writer charges a penalty for it.
        """
        undecided = verdict == "out-of-coverage"
        if undecided:
            event_type = EventType.GATE_INCONCLUSIVE
            outcome = "inconclusive"
        else:
            event_type = EventType.GATE_PASSED if passed else EventType.GATE_FAILED
            outcome = "success" if passed else "failure"
        data = {
            "gate_name": gate_name,
            "error_message": error_message[:500] if error_message else None,
        }
        if undecided:
            data["cause"] = cause
        self.log(
            ProgressEntry(
                event_type=event_type,
                task_id=task_id,
                agent_tier=agent_tier,
                agent_role=agent_role,
                data=data,
                outcome=outcome,
            )
        )

    def log_escalation(
        self,
        task_id: str,
        from_tier: str,
        to_tier: str,
        reason: str,
        memory_id: Optional[str] = None,
    ) -> None:
        """Log escalation event."""
        self.log(
            ProgressEntry(
                event_type=EventType.ESCALATION_TRIGGERED,
                task_id=task_id,
                agent_tier=from_tier,
                data={
                    "from_tier": from_tier,
                    "to_tier": to_tier,
                    "reason": reason,
                },
                memory_id=memory_id,
            )
        )
        try:
            from src.runtime.live_telemetry import emit_lifecycle_transition

            transition_details = {
                "from_role": from_tier,
                "to_role": to_tier,
                "reason": reason,
            }
            emit_lifecycle_transition(
                "rerouted",
                task_id=task_id,
                role=to_tier,
                details=transition_details,
            )
            emit_lifecycle_transition(
                "escalated",
                task_id=task_id,
                role=to_tier,
                details=transition_details,
            )
        except Exception:
            pass

    def log_escalation_outcome(
        self,
        task_id: str,
        resolved: bool,
        details: Optional[str] = None,
    ) -> None:
        """Log escalation resolution."""
        self.log(
            ProgressEntry(
                event_type=(
                    EventType.ESCALATION_RESOLVED if resolved else EventType.ESCALATION_FAILED
                ),
                task_id=task_id,
                outcome="success" if resolved else "failure",
                outcome_details=details,
            )
        )

    def log_exploration(
        self,
        task_id: str,
        query: str,
        strategy_used: str,
        tokens_spent: int,
        success: bool,
        function_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Log REPL exploration event with tool usage breakdown."""
        data: Dict[str, Any] = {
            "query": query[:200],
            "strategy": strategy_used,
            "tokens_spent": tokens_spent,
        }
        if function_counts:
            data["function_counts"] = function_counts
        self.log(
            ProgressEntry(
                event_type=EventType.EXPLORATION_COMPLETED,
                task_id=task_id,
                data=data,
                outcome="success" if success else "failure",
            )
        )

    def log_formalizer_invocation(
        self,
        task_id: str,
        service: str,
        endpoint: str,
        pages: int = 0,
        elapsed_sec: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log document formalizer / OCR service invocation."""
        self.log(
            ProgressEntry(
                event_type=EventType.FORMALIZER_INVOKED,
                task_id=task_id,
                data={
                    "service": service,
                    "endpoint": endpoint,
                    "pages": pages,
                    "elapsed_sec": round(elapsed_sec, 2),
                },
                outcome="success" if success else "failure",
                outcome_details=error,
            )
        )

    def log_memory_update(
        self,
        memory_id: str,
        old_q: float,
        new_q: float,
        reward: float,
        task_id: str,
    ) -> None:
        """Log Q-value update (from Q-scorer agent)."""
        self.log(
            ProgressEntry(
                event_type=EventType.Q_VALUE_UPDATED,
                task_id=task_id,
                memory_id=memory_id,
                data={
                    "old_q": old_q,
                    "new_q": new_q,
                    "reward": reward,
                },
            )
        )

    # =========================================================================
    # Session lifecycle events
    # =========================================================================

    def log_session_created(
        self,
        session_id: str,
        task_id: str,
        name: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        """Log session creation event.

        Args:
            session_id: The session UUID.
            task_id: Initial task ID for MemRL lineage.
            name: Optional session name.
            project: Optional project identifier.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_CREATED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "name": name,
                    "project": project,
                },
            )
        )

    def log_session_resumed(
        self,
        session_id: str,
        task_id: str,
        previous_task_id: str,
        resume_count: int,
        message_count: int,
        document_changes: int = 0,
    ) -> None:
        """Log session resume event.

        Args:
            session_id: The session UUID.
            task_id: New forked task ID (e.g., "original__r1").
            previous_task_id: Task ID before resume.
            resume_count: Total resume count for this session.
            message_count: Messages in session at resume time.
            document_changes: Number of source documents that changed.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_RESUMED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "previous_task_id": previous_task_id,
                    "resume_count": resume_count,
                    "message_count": message_count,
                    "document_changes": document_changes,
                },
            )
        )

    def log_session_checkpointed(
        self,
        session_id: str,
        task_id: str,
        checkpoint_id: str,
        trigger: str,
        message_count: int,
        findings_synced: int = 0,
    ) -> None:
        """Log session checkpoint event.

        Args:
            session_id: The session UUID.
            task_id: Current task ID.
            checkpoint_id: The checkpoint UUID.
            trigger: What triggered checkpoint ("turns", "idle", "explicit").
            message_count: Messages at checkpoint time.
            findings_synced: Number of findings synced to store.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_CHECKPOINTED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "checkpoint_id": checkpoint_id,
                    "trigger": trigger,
                    "message_count": message_count,
                    "findings_synced": findings_synced,
                },
            )
        )

    def log_session_archived(
        self,
        session_id: str,
        task_id: str,
        message_count: int,
        findings_count: int,
        summary_generated: bool = False,
    ) -> None:
        """Log session archive event.

        Args:
            session_id: The session UUID.
            task_id: Final task ID.
            message_count: Total messages in session.
            findings_count: Total findings in session.
            summary_generated: Whether LLM summary was generated.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_ARCHIVED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "message_count": message_count,
                    "findings_count": findings_count,
                    "summary_generated": summary_generated,
                },
                outcome="success",
            )
        )

    def log_session_finding(
        self,
        session_id: str,
        task_id: str,
        finding_id: str,
        source: str,
        confidence: float,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log finding added to session.

        Args:
            session_id: The session UUID.
            task_id: Current task ID.
            finding_id: The finding UUID.
            source: Finding source ("user_marked", "heuristic", "llm_extracted").
            confidence: Finding confidence score (0-1).
            tags: Optional finding tags.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_FINDING_ADDED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "finding_id": finding_id,
                    "source": source,
                    "confidence": confidence,
                    "tags": tags or [],
                },
            )
        )

    def __del__(self):
        """Flush on destruction."""
        self.flush()


class ProgressReader:
    """
    Reader for progress logs.

    Used by Q-scorer agent to process completed tasks.
    """

    def __init__(self, log_dir: Path = DEFAULT_LOG_PATH):
        self.log_dir = log_dir

    def read_date(self, date_str: str) -> List[ProgressEntry]:
        """Read all entries for a specific date."""
        log_path = self.log_dir / f"{date_str}.jsonl"
        if not log_path.exists():
            return []

        entries = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(ProgressEntry.from_json(line))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Skip malformed entries
                        continue
        return entries

    def read_recent(self, days: int = 7) -> List[ProgressEntry]:
        """Read entries from the last N days."""
        entries = []
        today = datetime.now(timezone.utc)

        for i in range(days):
            date = today - __import__("datetime").timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            entries.extend(self.read_date(date_str))

        return entries

    def read_all(self, days: int = 31) -> List[ProgressEntry]:
        """Read all entries from the last N days (bulk loading).

        Convenience wrapper over read_recent with a configurable range,
        intended for trajectory extraction and replay evaluation.
        """
        return self.read_recent(days=days)

    # Cache for read_recent to avoid re-parsing 100s of MBs of logs.
    # Shared across get_unscored_tasks and get_task_trajectory so that
    # scoring a batch of 10 tasks doesn't re-parse logs 10× (~4.7 GB).
    _recent_cache: "List[ProgressEntry] | None" = None
    _recent_cache_days: int = 0
    _recent_cache_ts: float = 0.0
    _RECENT_CACHE_TTL: float = 120.0  # seconds

    def _read_recent_cached(self, days: int = 1) -> "List[ProgressEntry]":
        """Cached wrapper around read_recent."""
        import time as _time
        now = _time.monotonic()
        if (
            self._recent_cache is not None
            and days <= self._recent_cache_days
            and (now - self._recent_cache_ts) < self._RECENT_CACHE_TTL
        ):
            return self._recent_cache
        result = self.read_recent(days)
        self._recent_cache = result
        self._recent_cache_days = days
        self._recent_cache_ts = now
        return result

    # Cache for get_unscored_tasks to avoid re-parsing 100s of MBs of logs
    _unscored_cache: "List[str] | None" = None
    _unscored_cache_ts: float = 0.0
    _UNSCORED_CACHE_TTL: float = 120.0  # seconds

    def get_unscored_tasks(self, days: int = 1) -> List[str]:
        """
        Find task IDs that have completed but not been Q-scored.

        Looks for TASK_COMPLETED without corresponding Q_VALUE_UPDATED.
        Results are cached for 120s to avoid repeated parsing of large logs.

        Note: Default reduced from 7 to 1 day — progress logs can be
        100s of MBs; scanning 7 days in background caused 100% CPU on
        all uvicorn workers.
        """
        import time as _time
        now = _time.monotonic()
        if (
            self._unscored_cache is not None
            and (now - self._unscored_cache_ts) < self._UNSCORED_CACHE_TTL
        ):
            return self._unscored_cache

        entries = self._read_recent_cached(days)

        # Single-pass scan for completed, scored, and routing entries
        completed_tasks: set[str] = set()
        scored_memory_ids: set[str] = set()
        routing_entries: list[tuple[str, str | None]] = []

        for entry in entries:
            if entry.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
                completed_tasks.add(entry.task_id)
            elif entry.event_type == EventType.Q_VALUE_UPDATED:
                scored_memory_ids.add(entry.memory_id)
            elif entry.event_type == EventType.ROUTING_DECISION:
                routing_entries.append((entry.task_id, entry.memory_id))

        # Find tasks whose routing memories haven't been scored
        unscored: set[str] = set()
        for task_id, memory_id in routing_entries:
            if task_id in completed_tasks:
                if memory_id and memory_id not in scored_memory_ids:
                    unscored.add(task_id)
                elif not memory_id:
                    unscored.add(task_id)

        result = list(unscored)
        self._unscored_cache = result
        self._unscored_cache_ts = now
        return result

    def get_task_trajectory(self, task_id: str, days: int = 1) -> List[ProgressEntry]:
        """Get all entries for a specific task.

        Uses the shared read_recent cache so that scoring a batch of
        tasks doesn't re-parse the full log for each one.
        """
        entries = self._read_recent_cached(days)
        return [e for e in entries if e.task_id == task_id]
