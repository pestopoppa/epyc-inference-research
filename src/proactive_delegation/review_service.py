"""Architect review and aggregation services.

Decision-plane core (H3: RD-1/3/5/6/8/9 + H1 TM-3). The heavyweight additions to
``ArchitectReviewService`` implement an explicit, typed, evidence-linked review
decision surface in SHADOW mode: it emits/records but never enforces. Enforcement
is gated by ``review_decision_enforce`` (blocked on H-LB LB-6) and is NOT wired
here. Model calls stay on the existing ``primitives.llm_call`` seam; every unit is
exercised with stub/fake completion callables (zero real inference).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, TYPE_CHECKING

from src.config import get_config
from src.config import _registry_timeout
from src.proactive_delegation.review_grammar import (
    parse_review_decision,
    review_decision_response_schema,
)
from src.proactive_delegation.types import (
    ArchitectReview,
    PlanReviewResult,
    ReviewDecision,
    SubtaskResult,
)
from src.structured_output.repair import parse_with_repair, primitives_completer

# CP1: the RD-3 mechanical verifier-precedence + reject-admissibility constants now
# live in the deterministic reducer (policy_reducer), which SUBSUMES this precedence.
# review_service emits recommendations + findings and DELEGATES the precedence
# decision to the reducer; the constants are re-exported here for back-compat
# (existing importers and tests read them off this module).
from src.proactive_delegation.policy_reducer import (
    FA_CANDIDATE,
    FR_CANDIDATE,  # noqa: F401 — re-exported for back-compat (tests/consumers import it off this module)
    OBJECTIVE_EVIDENCE_KINDS,
    conclusive_verdict as _reducer_conclusive_verdict,
    fail_certificates as _reducer_fail_certificates,
    verifier_precedence_recommendation,
)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives

logger = logging.getLogger(__name__)

# Default review timeout from registry
_REVIEW_TIMEOUT = float(_registry_timeout("external", "review_service", 15))


# ── Decision-plane trace categories + constants ───────────────────────────────
# These strings match ``src.trace.store.EventCategory`` values (kept as literals so
# the service does not hard-depend on the trace package at import time; the emit
# helper imports the store lazily and best-effort).
CAT_REVIEW_DECISION = "review_decision"
CAT_REVIEW_ESCALATION = "review_escalation"
CAT_PLAN_REMINDER = "plan_reminder"

# RD-3 verifier-precedence disagreement categories (FA_CANDIDATE/FR_CANDIDATE) and
# the RD-8 objective-evidence kinds (OBJECTIVE_EVIDENCE_KINDS) are defined in
# policy_reducer and imported above. A conclusive objective verdict overrides the
# reviewer; the disagreement is logged as a FA/FR *candidate* for the calibration
# ledger (H4). An evidence-free reject is inadmissible as enforcement (overcorrection
# runs 10:1–440:1). The DECISION about which precedence branch applies is the
# reducer's (verifier_precedence_recommendation); this service only composes the
# resulting ArchitectReview + shadow trace emission.

# RD-5 env gate mirroring safety_gate.SAFETY_GATE_WARN_ONLY (default ON): a would-be
# BLOCKING decision is downgraded + logged rather than enforced.
REVIEW_WARN_ONLY_ENV = "REVIEW_DECISION_WARN_ONLY"

# RD-7 Trinity tri-role axis: a review-decision turn IS a Verifier turn. Tag every
# emitted review-plane trace event's detail with the Trinity ``assigned_role`` so the
# tri-role shadow telemetry (scripts/analysis/trinity_shadow_telemetry.py, which scans
# the telemetry detail/``data`` for ``assigned_role``) can correlate review dispatches
# with Verifier-role semantics. This axis is ORTHOGONAL to the model role carried in
# ``Event.role`` (e.g. ``architect_general``); it mirrors the convention in
# routing_decision.routing_meta()'s ``assigned_role`` field. Value == TrinityRole.VERIFIER.value.
REVIEW_ASSIGNED_ROLE = "verifier"

# UTM-P1a: the ``Event.harness`` stamp on every review-plane trace row — the harness
# that PRODUCED the review turn. The review service is an in-process component of the
# orchestrator at every production construction site (delegator, parallel step
# executor, the chat_review route), so the default names the orchestrator; a driver
# that is a different harness (e.g. an offline replay) passes its own ``harness=``,
# and ``harness=None`` leaves the column NULL ("never captured") rather than guessing.
REVIEW_HARNESS = "orchestrator"

# RA-10 artifact schema_version stamp, kept as a literal so this module does not
# hard-depend on the trace package at import time (same pattern as CAT_* above).
#: RD-9 plan-rubric axes, in ``PLAN_REVIEW_RUBRIC_PROMPT`` order (RC-9 snapshot).
PLAN_RUBRIC_AXES: tuple[str, ...] = ("phase_coverage", "order", "executor_alignment")

# Must track ``src.trace.review_ledger.REVIEW_DECISION_SCHEMA_VERSION``.
REVIEW_DECISION_SCHEMA_VERSION = "1.0.0"

# TD-21.6: json_schema payloads for the abbreviated review-plane verdicts. Each is
# passed BOTH as the wire constraint on the first ``llm_call`` (TD-21.0 landed:
# json_schema now reaches the model on both the /completion and /v1 lanes) AND as
# the ``parse_with_repair`` validation/extraction target on a miss -- one schema,
# two uses, so the repair turn can never drift from what the model was asked for.
# ``additionalProperties`` is stated explicitly here (parse_with_repair's
# ``_closed_schema`` would default it anyway, TD-21.30(d)) so the wire payload and
# the validator are visibly the same closed shape.
REVIEW_ABBREV_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["d"],
    "properties": {
        # "request_changes" (the canonical ReviewDecision spelling) is included
        # alongside the prompt's abbreviated "changes": the pre-TD-21.6 parser
        # accepted ANY valid ReviewDecision literal here (decision =
        # ReviewDecision(decision_str) after only mapping "changes"), so a model
        # (or a caller's canned/replayed response) that emits the unabbreviated
        # word must not be schema-rejected into a spurious repair turn.
        "d": {
            "type": "string",
            "enum": ["approve", "changes", "request_changes", "escalate", "reject"],
        },
        "s": {"type": "number", "minimum": 0, "maximum": 1},
        "f": {"type": "string"},
        "c": {"type": "array", "items": {"type": "string"}},
    },
}

_PLAN_REVIEW_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "step": {"type": "string"},
        "op": {"type": "string", "enum": ["reroute", "drop", "add", "reorder"]},
        "v": {"type": "string"},
    },
}

REVIEW_PLAN_ABBREV_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["d"],
    "properties": {
        "d": {"type": "string", "enum": ["ok", "reorder", "drop", "add", "reroute"]},
        "s": {"type": "number", "minimum": 0, "maximum": 1},
        "f": {"type": "string"},
        "p": {"type": "array", "items": _PLAN_REVIEW_PATCH_SCHEMA},
    },
}

PLAN_RUBRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approve", "request_changes", "reject_to_empty", "escalate"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "phase_coverage": {"type": "boolean"},
        "order": {"type": "boolean"},
        "executor_alignment": {"type": "boolean"},
        "advisory": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "feedback": {"type": "string"},
            },
        },
    },
}

_TASKIR_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "actor": {"type": "string"},
        "action": {"type": "string"},
        "out": {"type": "array", "items": {"type": "string"}},
        "outputs": {"type": "array", "items": {"type": "string"}},
        "in": {"type": "array", "items": {"type": "string"}},
        "inputs": {"type": "array", "items": {"type": "string"}},
    },
}

TASKIR_STEPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        # minItems=1: an empty steps[] is schema-VALID JSON but a useless plan --
        # TD-21.6 closes this the same way as the other sites: a repair turn is
        # tried, and if that ALSO comes back empty it is the terminal-failure path
        # (single-step fallback + parse_failed=True), never a silently-accepted
        # empty plan.
        "steps": {"type": "array", "minItems": 1, "items": _TASKIR_STEP_SCHEMA},
    },
}

# TD-21.6: the sentinel decision value review_plan()/review_plan_rubric() return on
# a terminal (fished-invalid, repair-exhausted) parse failure. Deliberately IDENTICAL
# to an entry in ``src.trace.review_ledger.PARSE_FAILURE_DECISIONS`` so the offline
# ledger materializer (``scripts/analysis/reviewer_events_to_ledger.py``, which reads
# ``detail["decision"]`` verbatim into the ledger row) correctly buckets this as a
# parse failure instead of silently counting a fabricated "ok"/"approve" as a real
# verdict.
PARSE_FAILURE_SENTINEL = "parse_failure"


def _pairing_task_key(value: Any | None) -> str | None:
    """UTM-P1a: normalize a caller-supplied task identity for ``Event.task_key``.

    Pass-through only -- never derived. ``None`` / empty / whitespace -> ``None``
    (the column stays NULL = "never captured").
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pairing_ordinal(value: Any | None) -> int | None:
    """UTM-P1a: normalize a caller-supplied turn index for ``Event.turn_ordinal``.

    Accepts a non-negative ``int`` only (``bool`` is refused: ``True`` is not turn
    1). Anything else -> ``None`` so a malformed index never drops the row.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _estimate_tokens(text: str | None) -> int:
    """Cheap local token estimate (~4 chars/token) for decisions with no usage data."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _plan_executors(plan_steps: list[dict[str, Any]] | None) -> str | None:
    """Deduped, sorted actor/executor set of a plan, ``"|"``-joined.

    TM-8 executor-model-id: plan-review trace rows carry the set of actors the plan
    assigns so plan-compliance metrics (intake-835) can attribute per executor.
    """
    actors = sorted(
        {
            str(s.get("actor", s.get("role", ""))).strip()
            for s in (plan_steps or [])
            if str(s.get("actor", s.get("role", ""))).strip()
        }
    )
    return "|".join(actors) if actors else None


def build_review_decision_artifact(
    review: "ArchitectReview",
    *,
    latency_ms: float | None = None,
    tokens: dict[str, int] | None = None,
    decision_id: str | None = None,
    reviewed_at: str | None = None,
    role: str | None = None,
    executor_model_id: str | None = None,
) -> dict[str, Any]:
    """Compose a schema-valid ReviewDecision artifact (review_decision.schema.json).

    RD-12: per-decision latency + token accounting lands in the artifact's
    ``telemetry`` block (``wall_ms`` / ``tokens_in`` / ``tokens_out``) — exactly the
    channels ``review_decision_to_ledger_row`` reads, so the artifact feeds the H4
    calibration ledger and H-LB. Pure function (no emission, no I/O); the reviewer's
    ``blocking``/``advisory``/``evidence``/``verifier_requests`` channels map
    structurally (hard-stop stays separate from advisory, per schema).
    """
    tokens = tokens or {}
    return {
        "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
        "decision_id": decision_id or f"revdec-{uuid.uuid4().hex[:12]}",
        "subtask_id": review.subtask_id,
        "reviewed_at": reviewed_at or datetime.now(timezone.utc).isoformat(),
        "decision": review.decision.value,
        "confidence": review.confidence,
        "blocking": {"tripwire": review.tripwire},
        "advisory": {"score": review.score, "feedback": review.feedback},
        "evidence": list(review.evidence or []),
        "verifier_requests": list(review.verifier_requests or []),
        "telemetry": {
            "wall_ms": latency_ms,
            "tokens_in": tokens.get("tokens_in", 0),
            "tokens_out": tokens.get("tokens_out", 0),
        },
        "provenance": {"role": role or "reviewer", "executor_model_id": executor_model_id},
    }


class AggregationService:
    """Service for combining outputs from multiple specialists.

    Strategies:
    - concatenate: Simple concatenation with headers
    - merge_code: Merge code outputs intelligently
    - structured: Combine into structured JSON
    """

    def aggregate(
        self,
        results: list[SubtaskResult],
        strategy: str = "concatenate",
    ) -> str:
        """Aggregate multiple specialist outputs into final result."""
        if not results:
            return ""

        if strategy == "concatenate":
            return self._aggregate_concatenate(results)
        elif strategy == "merge_code":
            return self._aggregate_merge_code(results)
        elif strategy == "structured":
            return self._aggregate_structured(results)
        else:
            logger.warning(f"Unknown aggregation strategy '{strategy}', using concatenate")
            return self._aggregate_concatenate(results)

    def _aggregate_concatenate(self, results: list[SubtaskResult]) -> str:
        """Simple concatenation with section headers."""
        sections = []
        for result in results:
            if (result.success or getattr(result, "partial", False)) and result.output:
                header = f"## {result.subtask_id} ({result.role})"
                sections.append(f"{header}\n\n{result.output}")
        return "\n\n---\n\n".join(sections)

    def _aggregate_merge_code(self, results: list[SubtaskResult]) -> str:
        """Merge code outputs, handling imports and dependencies."""
        imports = set()
        code_blocks = []

        for result in results:
            if (not result.success and not getattr(result, "partial", False)) or not result.output:
                continue

            lines = result.output.split("\n")
            current_block = []

            for line in lines:
                # Extract imports
                if line.startswith("import ") or line.startswith("from "):
                    imports.add(line)
                else:
                    current_block.append(line)

            if current_block:
                code_blocks.append(
                    f"# From {result.subtask_id} ({result.role})\n" + "\n".join(current_block)
                )

        # Combine imports at top, then code blocks
        output_parts = []
        if imports:
            output_parts.append("\n".join(sorted(imports)))
        if code_blocks:
            output_parts.append("\n\n".join(code_blocks))

        return "\n\n".join(output_parts)

    def _aggregate_structured(self, results: list[SubtaskResult]) -> str:
        """Combine into structured JSON output."""
        structured = {
            "results": [
                {
                    "subtask_id": r.subtask_id,
                    "role": r.role,
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                }
                for r in results
            ],
            "summary": {
                "total": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
            },
        }
        return json.dumps(structured, indent=2)


class ArchitectReviewService:
    """Service for architect to review specialist outputs.

    The architect evaluates outputs against the original spec and provides
    feedback for iteration or approval.

    IMPORTANT: Prompts are designed for minimal token output from expensive
    architect models (Qwen3-235B, Qwen3-Coder-480B).

    RD-6 framing-neutrality audit (legacy prompts below): ``REVIEW_PROMPT_TEMPLATE``
    and ``QUICK_REVIEW_PROMPT`` were audited for the two measured attack surfaces —
    (a) "assume competent / expert authored this" priming and (b) explain-then-fix as
    the PRIMARY path (which doubles false-rejects, intake-836). Both legacy prompts
    are already free of (a) and put the verdict field FIRST while demanding
    "JSON only (no explanation)", so they do not fall into (b). They are therefore
    left byte-identical (the legacy ``review()``/``review_plan()`` paths must not
    change). The framing-neutral, sanitized-package, pointwise reviewer for the new
    decision plane is ``FRAMING_NEUTRAL_REVIEW_PROMPT`` + ``review_candidate()``, and
    the plan rubric (phase-coverage/order/executor-alignment, NOT prose) is
    ``PLAN_REVIEW_RUBRIC_PROMPT`` + ``review_plan_rubric()``.
    """

    # Concise review prompt - minimize architect output tokens
    REVIEW_PROMPT_TEMPLATE = """Review specialist output. Be BRIEF.

Objective: {objective}
Subtask: {action}
Output (truncated):
{output}

Reply JSON only (no explanation):
{{"d":"approve|changes|escalate|reject","s":0.0-1.0,"f":"<10 words","c":["fix1"]}}

d=decision, s=score, f=feedback, c=changes (optional, max 3 items)"""

    # Even more compact for simple approve/reject decisions
    QUICK_REVIEW_PROMPT = """Review: {action}
Output: {output_preview}
Reply: {{"d":"approve|changes","s":0.0-1.0,"f":"<5 words"}}"""

    # EVL-42 1c-fix (b): quick_mode keeps its small preview budget, but goes through
    # the same head+tail condensation so the conclusion is never silently dropped.
    QUICK_REVIEW_MAX_CHARS = 200

    @staticmethod
    def condense_output(output: str, max_chars: int) -> str:
        """Condense ``output`` to roughly ``max_chars`` for the reviewer prompt.

        EVL-42 1c-fix (b). The legacy behaviour was a silent ``output[:500]``
        prefix cut: a long-but-correct specialist answer whose conclusion sits
        past char 500 was judged on an unfinished fragment, biasing
        ``all_approved`` (→ memrl reward) against verbose-but-right outputs.

        Within budget → returned verbatim. Over budget → head (60%) + tail (40%)
        joined by an explicit marker stating how many chars were elided, so the
        reviewer sees the conclusion and knows the gap is ours, not the model's.
        The marker is not counted against the budget. ``max_chars <= 0`` disables
        condensation.
        """
        if max_chars <= 0 or len(output) <= max_chars:
            return output
        head = max(1, (max_chars * 6) // 10)
        tail = max(1, max_chars - head)
        elided = len(output) - head - tail
        marker = f"\n[... {elided} chars elided by reviewer budget; {len(output)} total ...]\n"
        return output[:head] + marker + output[-tail:]

    # RD-6: framing-neutral, pointwise, single-candidate reviewer over the SANITIZED
    # CandidatePackage view. Verdict FIRST; fixes only AFTER the verdict, phrased as
    # checkable artifacts (not prose advice); no competence priming; no comparison to
    # alternatives. Output shape matches review_decision_response_schema so
    # parse_review_decision() (RA-9) validates it.
    FRAMING_NEUTRAL_REVIEW_PROMPT = """Adjudicate ONE candidate against its objective.

Objective: {objective}
Acceptance checks:
{acceptance_checks}
Candidate output:
{outputs}

Rules:
- Judge only THIS candidate against the acceptance checks. Do not compare to alternatives.
- Decide FIRST. Do not explain or reason before the decision.
- Any fix belongs AFTER the verdict, phrased as a checkable test — not prose advice.

Reply JSON only:
{{"decision":"approve|request_changes|request_evidence|reject|reject_to_empty|escalate","confidence":0.0-1.0,"blocking":{{"tripwire":false}},"advisory":{{"score":0.0-1.0,"feedback":"<=12 words"}}}}"""

    # RD-9: plan rubric — checks phase-coverage / order / executor-alignment ONLY.
    # Explicitly NOT prose quality; penalizes over-specification as much as gaps.
    PLAN_REVIEW_RUBRIC_PROMPT = """Adjudicate a PLAN (not prose). Decide FIRST.

Objective: {objective}
Type: {task_type}
Plan steps:
{steps}

Judge ONLY these three axes — ignore wording/prose quality, do not reward verbosity:
- phase_coverage: does the plan cover every phase the objective needs (no missing step)?
- order: is every step ordered after the steps it depends on?
- executor_alignment: is each step assigned to an actor able to execute it?
Penalize over-specification (needless steps) as much as gaps.

Reply JSON only:
{{"decision":"approve|request_changes|reject_to_empty|escalate","confidence":0.0-1.0,"phase_coverage":true,"order":true,"executor_alignment":true,"advisory":{{"score":0.0-1.0,"feedback":"<=12 words"}}}}"""

    # TaskIR generation prompt (for future use when frontdoor queries architect)
    TASKIR_GENERATION_PROMPT = """Break down task into subtasks. Be MINIMAL.

Task: {objective}

Reply JSON only:
{{"steps":[{{"id":"S1","actor":"coder|worker|math","action":"<10 words","out":["file.py"]}}]}}

Rules:
- Max 5 steps
- actor: coder (code), worker (docs/tests), math (proofs)
- action: imperative, <10 words
- out: expected output files"""

    def __init__(
        self,
        primitives: "LLMPrimitives",
        architect_role: str | None = None,
        *,
        reviewer_role: "str | None" = None,
        warn_only: bool | None = None,
        trace_sink: "Callable[[Any], None] | None" = None,
        trace_db_path: "str | None" = None,
        harness: "str | None" = REVIEW_HARNESS,
    ):
        """Initialize the review service.

        Args:
            primitives: LLM primitives seam (all model calls go through it).
            architect_role: Explicit role override. When ``None`` (default), RD-1's
                config-level reviewer binding resolves it (``resolve_reviewer_role``);
                the default binding is ``architect_general`` so behavior is unchanged.
            reviewer_role: Optional binding override forwarded to the resolver.
            warn_only: RD-5 shadow-downgrade of would-be blocking decisions. ``None``
                reads ``REVIEW_DECISION_WARN_ONLY`` (default ON), mirroring safety_gate.
            trace_sink: TM-3 test seam — a callable receiving each ``Event`` instead of
                writing through to the store. ``None`` → best-effort write via emit.py.
            trace_db_path: Optional trace DB path override (tests point at a temp DB).
            harness: UTM-P1a ``Event.harness`` stamp for every emitted trace row — the
                harness producing the review turn (default ``REVIEW_HARNESS``,
                ``"orchestrator"``). ``None`` leaves the column NULL.
        """
        self.primitives = primitives
        cfg = get_config()
        deleg_cfg = cfg.delegation
        # RD-1: config-level reviewer role binding (default → architect_general).
        if architect_role is None:
            from src.roles import resolve_reviewer_role

            architect_role = str(resolve_reviewer_role(override=reviewer_role, config=cfg))
        self.architect_role = architect_role
        self.max_review_tokens = deleg_cfg.max_review_tokens
        self.max_taskir_tokens = deleg_cfg.max_taskir_tokens
        self.max_plan_review_tokens = deleg_cfg.max_plan_review_tokens
        # EVL-42 1c-fix (b): reviewer sees up to this many chars of the candidate
        # (head+tail condensed with an elision marker beyond it), not a 500-char prefix.
        self.review_output_max_chars = int(
            getattr(deleg_cfg, "review_output_max_chars", 4000)
        )
        # RD-5: warn-only shadow downgrade (env-gated; default ON, mirrors safety_gate).
        if warn_only is None:
            warn_only = os.environ.get(REVIEW_WARN_ONLY_ENV, "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self.warn_only = warn_only
        # TM-3: trace emission sink (injectable for tests; default → src/trace/emit.py).
        self._trace_sink = trace_sink
        self._trace_db_path = trace_db_path
        # UTM-P1a: pairing-key stamp (see REVIEW_HARNESS). Empty string == None.
        self.harness = (str(harness).strip() or None) if harness is not None else None
        # RD-12: distinct fallback counters. ``parse_failure_count`` counts reviewer
        # emissions that could not be parsed into a decision (incremented EXACTLY once
        # per failed parse); ``model_call_failures`` counts llm_call raising before any
        # response. One event lands in exactly one counter — never dropped, never
        # double-counted. These are per-service-instance accumulators the RD-12 replay
        # harness reads off the live service.
        self._parse_failure_count = 0
        self._model_call_failures = 0

    def _llm_call_no_think(self, *args: Any, **kwargs: Any) -> Any:
        """``self.primitives.llm_call`` with thinking OFF (RI-23).

        Every reviewer call here is a short structured emission (128-256 tokens, often
        schema-constrained) that a thinking block would consume. A no-op wrapper while
        ``thinking_roles_chat_lane`` is off.
        """
        from src.chat_completions_roles import thinking_off

        with thinking_off():
            return self.primitives.llm_call(*args, **kwargs)

    @property
    def parse_failure_count(self) -> int:
        """Distinct count of reviewer emissions that failed to parse (RD-12)."""
        return self._parse_failure_count

    @property
    def model_call_failures(self) -> int:
        """Distinct count of model calls that raised before producing a response."""
        return self._model_call_failures

    # ── TM-3 / RD-* trace emission (best-effort, never observable) ────────────

    def _emit_review_event(
        self,
        *,
        category: str,
        summary: str,
        detail: Any,
        status: str | None = None,
        role: str | None = None,
        session_id: Any | None = None,
        trial_id: int | None = None,
        task_key: Any | None = None,
        turn_ordinal: Any | None = None,
    ) -> None:
        """Write-through a review-plane trace event. NEVER raises, NEVER alters the
        caller's return value (TM-3 always-on emission). When a ``trace_sink`` is
        injected it receives the ``Event``; otherwise the event is committed via
        ``src/trace/emit.py`` (write-through, content-addressed, idempotent).

        UTM-P1a pairing keys (event schema v2): ``harness`` is the service-level
        stamp (``self.harness``); ``task_key`` / ``turn_ordinal`` are passed ONLY by
        call sites that genuinely hold them (a TaskIR ``task_id``, a plan step index)
        and stay NULL otherwise. ``seed`` is never stamped here: ``llm_call`` carries
        no seed, so the service cannot know one. A malformed key degrades to NULL
        instead of raising -- a bad ordinal must not cost the row it rides on.
        """
        try:
            from src.trace.store import Event, EventSource, detail_to_json

            # RD-7: tag the emitted detail with the Trinity Verifier-role axis (mirror of
            # routing_meta()'s ``assigned_role``). Non-mutating shallow merge — the
            # caller's dict is untouched and the service's returns stay byte-identical.
            if isinstance(detail, dict) and "assigned_role" not in detail:
                detail = {**detail, "assigned_role": REVIEW_ASSIGNED_ROLE}

            ev = Event(
                ts_utc="",  # emit() stamps 'now' when empty
                source=EventSource.REVIEW_PLANE,
                source_path="",  # emit() assigns a content-addressed synthetic path
                source_line=None,
                session_id=str(session_id) if session_id is not None else None,
                trial_id=trial_id if isinstance(trial_id, int) else None,
                role=role or self.architect_role,
                category=category,
                status=status,
                summary=summary,
                detail_json=detail_to_json(detail),
                harness=self.harness,
                task_key=_pairing_task_key(task_key),
                turn_ordinal=_pairing_ordinal(turn_ordinal),
            )
            if self._trace_sink is not None:
                self._trace_sink(ev)
                return
            from src.trace.emit import emit

            if self._trace_db_path is not None:
                emit(ev, db_path=self._trace_db_path)
            else:
                emit(ev)
        except Exception as exc:  # pragma: no cover - emission is best-effort
            logger.debug("Review trace emission skipped: %s", exc)

    @staticmethod
    def _response_tokens(response: Any, prompt: str | None = None) -> dict[str, int]:
        """Per-decision token accounting (TM-3 / RD-12).

        Prefers real counts off a response object (``tokens_generated`` on
        ``LLMResult``, a ``usage`` mapping with prompt/completion counts) and falls
        back to a length-based estimate for plain-string completions (the
        ``llm_call`` seam returns ``str``).

        RD-12 adds the PROMPT side: ``tokens_in`` is read from ``usage.prompt_tokens``
        when present, else estimated from the prompt text (``prompt`` argument). The
        returned dict feeds BOTH the trace-row detail and the decision artifact's
        ``telemetry`` block (``tokens_in`` / ``tokens_out`` — the channels
        ``review_decision_to_ledger_row`` reads), which is what H-LB consumes.
        """
        text = (
            response
            if isinstance(response, str)
            else str(getattr(response, "text", response) or "")
        )
        tokens_out = _estimate_tokens(text)
        tg = getattr(response, "tokens_generated", None)
        if isinstance(tg, int) and tg > 0:
            tokens_out = tg
        tokens_in = _estimate_tokens(prompt) if prompt else 0
        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            for key in ("completion_tokens", "output_tokens", "tokens_out"):
                val = usage.get(key)
                if isinstance(val, int) and val > 0:
                    tokens_out = val
                    break
            for key in ("prompt_tokens", "input_tokens", "tokens_in"):
                val = usage.get(key)
                if isinstance(val, int) and val > 0:
                    tokens_in = val
                    break
        return {"tokens_in": int(tokens_in), "tokens_out": int(tokens_out), "chars_out": len(text)}

    def review(
        self,
        spec: dict[str, Any],
        subtask: dict[str, Any],
        output: str,
        quick_mode: bool = False,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
        executor_model_id: str | None = None,
        turn_ordinal: Any | None = None,
    ) -> ArchitectReview:
        """Have architect review a specialist's output.

        TM-3: every invocation emits a REVIEW_DECISION trace event (with per-decision
        latency_ms + token counts) regardless of downstream acting. The RETURNED
        ``ArchitectReview`` is byte-identical to the pre-decision-plane behavior.

        UTM-P1a.2: ``turn_ordinal`` is the caller's own 0-based review-iteration
        index for this step (the delegator's review-fix loop holds one; a single
        one-shot review does not). Pass-through only -- the service never counts
        turns itself, so an unsupplied ordinal stays NULL = "never captured".

        RD-12: the emitted detail carries prompt+completion tokens (``tokens``),
        ``phase="review"``, and (when known) ``executor_model_id``; a reviewer
        emission that fails to parse is counted distinctly (``parse_failure_count``)
        and flagged ``parse_ok=False`` — never dropped, never double-counted.
        """
        subtask_id = subtask.get("id", "unknown")
        action = subtask.get("action", "")

        # EVL-42 1c-fix (b): head+tail condensation with an explicit elision marker
        # (was a silent output[:500] prefix cut — see condense_output()).
        output_truncated = self.condense_output(output, self.review_output_max_chars)

        # Build concise review prompt - only include objective, not full spec
        objective = spec.get("objective", "")[:200]

        if quick_mode:
            prompt = self.QUICK_REVIEW_PROMPT.format(
                action=action[:50],
                output_preview=self.condense_output(output, self.QUICK_REVIEW_MAX_CHARS),
            )
        else:
            prompt = self.REVIEW_PROMPT_TEMPLATE.format(
                objective=objective,
                action=action,
                output=output_truncated,
            )

        start = time.perf_counter()
        parse_ok = True
        model_call_failed = False
        parse_failure: str | None = None
        repaired = False
        tokens: dict[str, int] = {"tokens_in": 0, "tokens_out": 0, "chars_out": 0}
        try:
            # Call architect with strict token limit. TD-21.6/TD-21.0: schema on the
            # wire constrains the reviewer at generation time, not only at parse time.
            response = self._llm_call_no_think(
                prompt,
                role=self.architect_role,
                n_tokens=self.max_review_tokens,
                json_schema=REVIEW_ABBREV_SCHEMA,
            )
        except Exception as e:
            logger.warning(f"Architect review call failed: {e}", exc_info=True)
            model_call_failed = True
            parse_ok = False
            self._model_call_failures += 1
            # Default to request_changes on a MODEL-CALL exception -- unchanged.
            # This is a distinct, already-labeled failure mode (model_call_failures)
            # from the silent-default-VERDICT bug TD-21.6 targets, which lives
            # specifically in the PARSE path below (there is genuinely no reviewer
            # emission to interpret here, only an exception).
            review = ArchitectReview(
                subtask_id=subtask_id,
                decision=ReviewDecision.REQUEST_CHANGES,
                feedback=f"Review failed: {e}",
                score=0.3,
            )
        else:
            tokens = self._response_tokens(response, prompt)
            # TD-21.6: fish -> validate -> ONE repair turn on a miss. Never a
            # default verdict: a terminal failure withholds as REQUEST_EVIDENCE
            # (RD-6's existing admissibility-safe pattern from review_candidate) --
            # a value this method's own prompt NEVER offers the model, so its
            # presence is unambiguous evidence of a parse failure to any
            # downstream branch, unlike the old "request_changes" default, which
            # was indistinguishable from a genuine reviewer verdict.
            outcome = parse_with_repair(
                response,
                schema=REVIEW_ABBREV_SCHEMA,
                complete=primitives_completer(
                    self.primitives, self.architect_role, n_tokens=self.max_review_tokens
                ),
                site="review_service.review",
                # TD-21.34: `s`(core)/`f`(eedback)/`c`(hanges) must come from
                # what the reviewer already said in its (malformed) reply --
                # a fabricated score is the same class of bug as an invented
                # `tier`. `d`(ecision) is exempt: the abbreviated enum tokens
                # ("request_changes", etc.) are a classification the reviewer
                # rarely spells out verbatim even when it clearly reached
                # that verdict in prose.
                require_evidence=True,
                evidence_exempt={"d"},
            )
            if outcome.status in ("parsed", "repaired"):
                try:
                    review_data = outcome.value
                    # Map abbreviated keys to full names
                    decision_str = review_data.get("d", review_data.get("decision", "changes"))
                    # Normalize decision string
                    if decision_str == "changes":
                        decision_str = "request_changes"
                    decision = ReviewDecision(decision_str)

                    review = ArchitectReview(
                        subtask_id=subtask_id,
                        decision=decision,
                        feedback=review_data.get("f", review_data.get("feedback", "")),
                        score=float(review_data.get("s", review_data.get("score", 0.5))),
                        suggested_changes=review_data.get(
                            "c", review_data.get("suggested_changes", [])
                        ),
                        approved_output=output if decision == ReviewDecision.APPROVE else None,
                    )
                    repaired = outcome.status == "repaired"
                except Exception as e:
                    # Defense in depth only (the schema already constrains `d` to
                    # valid ReviewDecision values) -- never let an unexpected
                    # mapping error fall through to a default verdict either.
                    logger.warning(f"Architect review mapping failed: {e}", exc_info=True)
                    parse_ok = False
                    parse_failure = f"post_repair_mapping_error: {e}"[:200]
                    self._parse_failure_count += 1
                    review = ArchitectReview(
                        subtask_id=subtask_id,
                        decision=ReviewDecision.REQUEST_EVIDENCE,
                        feedback=f"parse_failure:{parse_failure}",
                        score=0.0,
                        confidence=0.0,
                    )
            else:
                parse_ok = False
                parse_failure = outcome.reason or "unparseable_response"
                self._parse_failure_count += 1
                review = ArchitectReview(
                    subtask_id=subtask_id,
                    decision=ReviewDecision.REQUEST_EVIDENCE,
                    feedback=f"parse_failure:{parse_failure}",
                    score=0.0,
                    confidence=0.0,
                )

        # TM-3: always-on shadow emission (side effect only; return is unchanged).
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"review {subtask_id}: {review.decision.value}",
            status=review.decision.value,
            detail={
                "mode": "review",
                "phase": "review",
                "subtask_id": subtask_id,
                "decision": review.decision.value,
                "score": review.score,
                "confidence": review.confidence,
                "tripwire": review.tripwire,
                "quick_mode": quick_mode,
                "parse_ok": parse_ok,
                "parse_failure": parse_failure,
                "repaired": repaired,
                "model_call_failed": model_call_failed,
                "executor_model_id": executor_model_id or subtask.get("role"),
                "latency_ms": latency_ms,
                "tokens": tokens,
            },
            session_id=session_id,
            trial_id=trial_id,
            # UTM-P1a: the TaskIR's own task identity, passed through when the caller
            # supplied one (delegator: ``task_ir["task_id"]``). Never derived from
            # ``subtask_id`` -- step ids like "S1" recur across unrelated tasks.
            task_key=spec.get("task_id") if isinstance(spec, dict) else None,
            # UTM-P1a.2: the caller's review-iteration index, when it holds one.
            turn_ordinal=turn_ordinal,
        )
        return review

    def review_plan(
        self,
        objective: str,
        task_type: str,
        plan_steps: list[dict[str, Any]],
        timeout_seconds: float = _REVIEW_TIMEOUT,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
        executor_model_id: str | None = None,
    ) -> PlanReviewResult | None:
        """Have architect review a plan before specialist execution.

        Returns PlanReviewResult on success, None on timeout/error.
        Non-blocking: never prevents request completion.

        TM-3: every invocation emits a REVIEW_DECISION trace event (latency_ms +
        token counts) regardless of whether ``plan_review`` acts — this is the
        DECOUPLED shadow-emission path (RD-5): the trace flows even when the
        ``plan_review`` feature (which requires ``memrl``) is off. The RETURNED value
        is byte-identical to the pre-decision-plane behavior.

        RD-12: an unparseable reviewer emission is counted distinctly
        (``parse_failure_count``) and flagged ``parse_ok=False`` in the trace row;
        the returned ``PlanReviewResult`` is unchanged.
        """
        from src.prompt_builders import build_plan_review_prompt

        prompt = build_plan_review_prompt(objective, task_type, plan_steps)

        start = time.perf_counter()
        parse_ok = True
        parse_failure: str | None = None
        repaired = False
        try:
            response = self._llm_call_no_think(
                prompt,
                role=self.architect_role,
                n_tokens=self.max_plan_review_tokens,
                json_schema=REVIEW_PLAN_ABBREV_SCHEMA,
            )
        except Exception as e:
            logger.warning(f"Plan review failed: {e}", exc_info=True)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._model_call_failures += 1
            self._emit_review_event(
                category=CAT_REVIEW_DECISION,
                summary="plan_review: error",
                status="error",
                detail={
                    "mode": "plan_review",
                    "phase": "plan",
                    "error": str(e),
                    "parse_ok": False,
                    "repaired": False,
                    "model_call_failed": True,
                    "executor_model_id": executor_model_id or _plan_executors(plan_steps),
                    "latency_ms": latency_ms,
                    "tokens": {"tokens_in": 0, "tokens_out": 0, "chars_out": 0},
                },
                session_id=session_id,
                trial_id=trial_id,
            )
            return None  # Non-blocking -- proceed without review

        # TD-21.6: fish -> validate -> ONE repair turn on a miss.
        outcome = parse_with_repair(
            response,
            schema=REVIEW_PLAN_ABBREV_SCHEMA,
            complete=primitives_completer(
                self.primitives, self.architect_role, n_tokens=self.max_plan_review_tokens
            ),
            site="review_service.review_plan",
            # TD-21.34: same reasoning as `review_service.review` -- `s`/`f`
            # must be copied from the reviewer's own (malformed) reply, `d`
            # is exempt (classification), and `p[].op` is exempt for the
            # same reason (`reroute`/`drop`/`add`/`reorder` is a
            # classification of the patch, not a copied fact).
            require_evidence=True,
            evidence_exempt={"d", "op"},
        )
        if outcome.status in ("parsed", "repaired"):
            review_data = outcome.value
            repaired = outcome.status == "repaired"
            decision = review_data.get("d", review_data.get("decision", "ok"))
            # Normalize valid decisions (defense in depth: the schema already
            # constrains `d` to this set)
            valid_decisions = {"ok", "reorder", "drop", "add", "reroute"}
            if decision not in valid_decisions:
                decision = "ok"
            score = float(review_data.get("s", review_data.get("score", 0.5)))
            feedback = review_data.get("f", review_data.get("feedback", ""))
            patches = review_data.get("p", review_data.get("patches", []))
        else:
            # TD-21.6: terminal failure -- never a default "ok". PARSE_FAILURE_SENTINEL
            # ("parse_failure") is the literal string src.trace.review_ledger's
            # PARSE_FAILURE_DECISIONS matches, so the offline ledger materializer
            # correctly buckets this row instead of miscounting a fabricated approval.
            # Non-blocking contract preserved: patches=[] means _apply_plan_review
            # no-ops and execution proceeds through the normal no-plan/default route.
            parse_ok = False
            parse_failure = outcome.reason or "unparseable_response"
            self._parse_failure_count += 1
            decision = PARSE_FAILURE_SENTINEL
            score = 0.0
            feedback = f"parse_failure:{parse_failure}"
            patches = []

        result = PlanReviewResult(
            decision=decision,
            score=score,
            feedback=feedback,
            patches=patches,
            raw_response=response[:200],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"plan_review: {result.decision}",
            status=result.decision,
            detail={
                "mode": "plan_review",
                "phase": "plan",
                "decision": result.decision,
                "score": result.score,
                "patches": len(result.patches),
                "parse_ok": parse_ok,
                "parse_failure": parse_failure,
                "repaired": repaired,
                "model_call_failed": False,
                "executor_model_id": executor_model_id or _plan_executors(plan_steps),
                "latency_ms": latency_ms,
                "tokens": self._response_tokens(response, prompt),
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return result

    def generate_taskir(self, objective: str) -> dict[str, Any]:
        """Have architect generate minimal TaskIR for an objective.

        TD-21.6: schema on the wire (``TASKIR_STEPS_SCHEMA``, ``minItems: 1`` so an
        empty ``steps[]`` is a schema miss, not a silently-accepted empty plan) plus
        ONE ``parse_with_repair`` turn on a miss. A model-call exception AND a
        terminal (repair-exhausted) parse failure now return the SAME explicit
        single-step fallback plan, marked ``parse_failed=True`` + ``error=<reason>``
        so it is distinguishable from a genuine TaskIR -- never a silent default.
        Before this fix, a genuine JSON-parse failure (as opposed to an `llm_call`
        exception) fell through ``_parse_review_response``'s sentinel into an
        EMPTY ``steps[]`` returned as if it were a valid plan; that path is closed.

        No production caller consumes this method today (dead since RD-2/H2); the
        review-plane counters (``parse_failure_count`` / ``model_call_failures``)
        are therefore NOT touched here -- unchanged from the pre-existing design
        ("TaskIR generation is not a review decision", see
        ``test_generate_taskir_does_not_touch_counters``).
        """
        prompt = self.TASKIR_GENERATION_PROMPT.format(
            objective=objective[:300],  # Truncate long objectives
        )

        def _fallback(reason: str) -> dict[str, Any]:
            return {
                "task_id": f"arch-{uuid.uuid4().hex[:8]}",
                "objective": objective,
                "plan": {
                    "steps": [
                        {
                            "id": "S1",
                            "actor": "coder",
                            "action": objective[:50],
                            "outputs": ["output.txt"],
                        }
                    ]
                },
                "parse_failed": True,
                "error": reason,
            }

        try:
            response = self._llm_call_no_think(
                prompt,
                role=self.architect_role,
                n_tokens=self.max_taskir_tokens,
                json_schema=TASKIR_STEPS_SCHEMA,
            )
        except Exception as e:
            logger.warning(f"TaskIR generation failed: {e}", exc_info=True)
            return _fallback(f"model_call_failed: {e}"[:200])

        outcome = parse_with_repair(
            response,
            schema=TASKIR_STEPS_SCHEMA,
            complete=primitives_completer(
                self.primitives, self.architect_role, n_tokens=self.max_taskir_tokens
            ),
            site="review_service.generate_taskir",
            # TD-21.34: unlike `proactive_stage.py`'s plan-step schema, `actor`
            # here has NO enum constraint and no closest-match instruction --
            # it must be a literal actor the objective/reply already names, so
            # no exemption.
            require_evidence=True,
        )
        if outcome.status not in ("parsed", "repaired"):
            return _fallback(outcome.reason or "unparseable_response")

        taskir_data = outcome.value
        # Normalize abbreviated format to full TaskIR
        steps = taskir_data.get("steps", [])
        normalized_steps = []
        for i, step in enumerate(steps):
            normalized_steps.append(
                {
                    "id": step.get("id", f"S{i + 1}"),
                    "actor": step.get("actor", "worker"),
                    "action": step.get("action", ""),
                    "outputs": step.get("out", step.get("outputs", [])),
                    "inputs": step.get("in", step.get("inputs", [])),
                }
            )

        result = {
            "task_id": f"arch-{uuid.uuid4().hex[:8]}",
            "objective": objective,
            "plan": {"steps": normalized_steps},
        }
        if outcome.status == "repaired":
            result["repaired"] = True
        return result

    def _parse_review_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from architect response.

        TD-21.6: no longer the primary path for review()/review_plan()/
        review_plan_rubric()/generate_taskir() -- those now route through
        ``parse_with_repair`` (schema validation + ONE repair turn, never a
        silent default). Kept as a standalone JSON-fishing utility, still
        exercised directly by ``TestArchitectReviewServiceHelpers``.
        """
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Could not parse review response: {response[:200]}")
            # Marked fallback so a direct caller of this legacy helper can still
            # distinguish a parse failure from a real verdict.
            return {
                "_parse_fallback": True,
                "decision": "request_changes",
                "feedback": "Parse error",
            }

    # ══════════════════════════════════════════════════════════════════════════
    # Decision plane (SHADOW-ONLY): RD-3 / RD-5 / RD-6 / RD-8 / RD-9.
    #
    # These methods EMIT and RECORD but never enforce. Enforcement is gated by the
    # ``review_decision_enforce`` flag (blocked on H-LB LB-6) and is intentionally
    # NOT wired here. They stay on the ``primitives.llm_call`` seam and are fully
    # exercisable with stub completion callables.
    # ══════════════════════════════════════════════════════════════════════════

    # ── rendering helpers ────────────────────────────────────────────────────

    @staticmethod
    def _render_outputs(outputs: list[dict[str, Any]] | None) -> str:
        lines = []
        for o in (outputs or [])[:8]:
            ref = str(o.get("ref", ""))[:400]
            lines.append(f"- [{o.get('type', 'artifact')}] {ref}")
        return "\n".join(lines) or "(none)"

    @staticmethod
    def _render_checks(checks: list[dict[str, Any]] | None) -> str:
        lines = []
        for c in (checks or [])[:8]:
            lines.append(f"- {c.get('id', '?')}: {str(c.get('statement', ''))[:120]}")
        return "\n".join(lines) or "(none)"

    @staticmethod
    def _render_plan_steps(plan_steps: list[dict[str, Any]] | None) -> str:
        lines = []
        for s in (plan_steps or [])[:8]:
            sid = s.get("id", "S?")
            actor = s.get("actor", s.get("role", "worker"))
            action = str(s.get("action", ""))[:50]
            deps = s.get("deps", s.get("inputs", []))
            dep_str = f" ({','.join(str(d) for d in deps[:3])})" if deps else ""
            lines.append(f"{sid}:{actor}:{action}{dep_str}")
        return "\n".join(lines) or "(no steps)"

    # ── RD-6: framing-neutral, pointwise, single-candidate review ─────────────

    def review_candidate(
        self,
        sanitized_view: dict[str, Any],
        *,
        subtask_id: str = "candidate",
        session_id: Any | None = None,
        trial_id: int | None = None,
        executor_model_id: str | None = None,
        task_key: Any | None = None,
    ) -> ArchitectReview:
        """Pointwise review of a PRE-SANITIZED CandidatePackage view (RD-6).

        UTM-P1a.2: ``task_key`` is the caller's cross-harness task identity (the
        replay harness passes its corpus ``task_id``), threaded through to
        ``Event.task_key`` exactly like ``review()`` threads ``spec["task_id"]``.
        Never derived from ``subtask_id`` or ``task_ref``; unsupplied -> NULL.

        ``sanitized_view`` MUST be the ``candidate_package.schema.json``
        ``sanitized_view`` projection (author self-assessment / confidence assertions
        / quality labels already stripped at assembly time). This method consumes
        ONLY that projection; if it detects framing-leaking keys it logs and ignores
        them (defense-in-depth — sanitization is an assembly-time contract).

        Emits a REVIEW_DECISION trace event with latency + tokens. A parse failure is
        recorded and withheld as REQUEST_EVIDENCE — never a reject (admissibility).
        RD-12: a failed parse is counted distinctly (``parse_failure_count``) and the
        trace detail carries ``phase="review"`` + ``executor_model_id``.

        TD-21.7: the wire call now carries ``review_decision_response_schema()`` —
        ``review_grammar.py`` has shipped this json_schema payload since RA-7; only
        the ``llm_call`` kwarg was ever missing (X4). The existing balanced-brace
        fish + strict Draft202012 validation + typed ``ParseFailure`` stays the
        FIRST parse attempt (unchanged, still against the FULL
        ``review_decision.schema.json``); on a miss, ONE ``parse_with_repair`` turn
        (against the same compact schema) is tried before withholding — the
        pre-existing REQUEST_EVIDENCE backstop is now the residual, not the
        default outcome of every miss.
        """
        for banned in ("author_self_assessment", "author_confidence_assertion", "quality_labels"):
            if isinstance(sanitized_view, dict) and banned in sanitized_view:
                logger.warning(
                    "review_candidate received an UNSANITIZED package (found %r); "
                    "ignoring framing-leaking field",
                    banned,
                )

        objective = str((sanitized_view or {}).get("objective", ""))[:400]
        outputs = self._render_outputs((sanitized_view or {}).get("outputs"))
        checks = self._render_checks((sanitized_view or {}).get("acceptance_checks"))
        prompt = self.FRAMING_NEUTRAL_REVIEW_PROMPT.format(
            objective=objective, outputs=outputs, acceptance_checks=checks
        )
        compact_schema = review_decision_response_schema()

        start = time.perf_counter()
        response: Any = ""
        model_call_failed = False
        repaired = False
        try:
            response = self._llm_call_no_think(
                prompt,
                role=self.architect_role,
                n_tokens=self.max_review_tokens,
                json_schema=compact_schema,
            )
        except Exception as exc:
            logger.warning("review_candidate model call failed: %s", exc)
            response = ""
            model_call_failed = True
            self._model_call_failures += 1

        text = response if isinstance(response, str) else str(getattr(response, "text", "") or "")
        obj, failure = parse_review_decision(text)
        if obj is None and not model_call_failed and text:
            # TD-21.7: ONE repair turn against the compact schema before
            # withholding. Skipped when there is genuinely no text to repair
            # (model-call failure) — nothing for the repair turn to copy.
            outcome = parse_with_repair(
                text,
                schema=compact_schema,
                complete=primitives_completer(
                    self.primitives, self.architect_role, n_tokens=self.max_review_tokens
                ),
                site="review_service.review_candidate",
                # TD-21.34: `confidence` is REQUIRED by `compact_schema`, so a
                # malformed reply with no stated confidence is exactly the
                # grammar-forces-a-required-field fabrication risk; `advisory.
                # score`/`feedback`/`blocking.blocking_issues[].summary` must
                # likewise come from the reviewer's own reply. `decision` is
                # exempt (classification, same as `review()`'s `d`).
                require_evidence=True,
                evidence_exempt={"decision"},
            )
            if outcome.status in ("parsed", "repaired"):
                obj = outcome.value
                repaired = outcome.status == "repaired"
                failure = None
        if obj is None:
            # Admissibility: a parse failure must NOT become a reject — withhold.
            review = ArchitectReview(
                subtask_id=subtask_id,
                decision=ReviewDecision.REQUEST_EVIDENCE,
                feedback=f"parse_failure:{failure.reason.value if failure else 'unknown'}",
                score=0.0,
                confidence=0.0,
            )
            parse_ok = False
            if not model_call_failed:
                # A model-call failure is counted on its own counter; a genuine
                # unparseable emission lands on the parse-failure counter. Exactly
                # one increment, never both.
                self._parse_failure_count += 1
        else:
            review = self._architect_review_from_decision(obj, subtask_id)
            parse_ok = True

        latency_ms = (time.perf_counter() - start) * 1000.0
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"review_candidate {review.subtask_id}: {review.decision.value}",
            status=review.decision.value,
            detail={
                "mode": "review_candidate",
                "phase": "review",
                "framing_neutral": True,
                "pointwise": True,
                "candidate_ref": (sanitized_view or {}).get("task_ref"),
                "decision": review.decision.value,
                "confidence": review.confidence,
                "score": review.score,
                "tripwire": review.tripwire,
                "parse_ok": parse_ok,
                "parse_failure": failure.to_dict() if failure else None,
                "repaired": repaired,
                "model_call_failed": model_call_failed,
                "executor_model_id": executor_model_id,
                "latency_ms": latency_ms,
                "tokens": self._response_tokens(response, prompt),
            },
            session_id=session_id,
            trial_id=trial_id,
            # UTM-P1a.2: pass-through of the caller's task identity (replay corpus id).
            task_key=task_key,
        )
        return review

    @staticmethod
    def _architect_review_from_decision(obj: dict[str, Any], subtask_id: str) -> ArchitectReview:
        """Map a validated review_decision object → ArchitectReview (RD-6)."""
        blocking = obj.get("blocking") or {}
        advisory = obj.get("advisory") or {}
        return ArchitectReview(
            subtask_id=obj.get("subtask_id", subtask_id),
            decision=ReviewDecision(obj["decision"]),
            feedback=str(advisory.get("feedback", "")),
            score=float(advisory.get("score", 0.0) or 0.0),
            suggested_changes=[
                i["summary"] for i in blocking.get("blocking_issues", []) if "summary" in i
            ],
            approved_output=None,
            confidence=float(obj.get("confidence", 0.0) or 0.0),
            tripwire=bool(blocking.get("tripwire", False)),
            evidence=obj.get("evidence", []) or [],
            verifier_requests=obj.get("verifier_requests", []) or [],
        )

    # ── RD-3: verifier precedence (mechanical, three-valued) ──────────────────

    @staticmethod
    def _conclusive_verdict(report: dict[str, Any]) -> str:
        """Aggregate a VerificationReport-shaped dict → pass|fail|inconclusive.

        Precedence over the reviewer applies ONLY to conclusive (pass/fail) verdicts;
        ``inconclusive`` hands control back to the reviewer. Uses ``summary.
        conclusive_verdict`` when present, else derives from required checks
        (any required inconclusive → inconclusive; else any required fail → fail;
        else ≥1 required pass and none fail/inconclusive → pass; else inconclusive).

        CP1: delegates to ``policy_reducer.conclusive_verdict`` — the reducer subsumes
        this precedence mechanic. Behavior is byte-identical (same algorithm).
        """
        return _reducer_conclusive_verdict(report)

    @staticmethod
    def _fail_certificates(report: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect failing checks' certificates — the request_evidence payload.

        CP1: delegates to ``policy_reducer.fail_certificates`` (subsumed).
        """
        return _reducer_fail_certificates(report)

    def apply_verifier_precedence(
        self,
        review: ArchitectReview,
        report: dict[str, Any] | None,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
        executor_model_id: str | None = None,
    ) -> ArchitectReview:
        """Mechanical verifier-precedence over a reviewer decision (RD-3).

        Rules (conclusive objective verdicts override reviewer claims):
          1. reviewer-approve + conclusive-gate-FAIL → emit ``fa_candidate``; override
             to REQUEST_EVIDENCE with the failing certificate(s) attached.
          2. reviewer-reject + conclusive-gate-PASS → emit ``fr_candidate``; downgrade
             to REQUEST_EVIDENCE (never keep a reject when objective PASSED — rule 3).
          3. objective-PASS + advisory-low reject-class → REQUEST_EVIDENCE (folded into
             rule 2: objective PASS never yields a reject).
          4. ``inconclusive`` → defer to the reviewer (return unchanged).

        Emits a disagreement trace event on override. Returns the (possibly adjusted)
        review; whether it is ACTED on is the shadow/enforce split (RD-5), not here.

        CP1: the precedence DECISION (which disagreement class, what adjusted verdict)
        is now the reducer's — ``policy_reducer.verifier_precedence_recommendation``.
        This method only composes the resulting ``ArchitectReview`` + shadow trace so
        the return value and emitted event stay byte-identical to the RD-3 behavior.
        """
        verdict = self._conclusive_verdict(report or {})
        adjusted_decision, trace_cat = verifier_precedence_recommendation(
            review.decision.value, verdict
        )
        if trace_cat is None:  # inconclusive verdict, or reviewer/verifier agreement
            return review

        fail_certs = self._fail_certificates(report or {})
        adjusted: ArchitectReview

        if trace_cat == FA_CANDIDATE:  # rule 1: reviewer-approve + conclusive-fail
            evidence = list(review.evidence or []) + [
                {
                    "kind": "gate_result",
                    "ref": fc.get("check_id"),
                    "summary": "conclusive FAIL certificate",
                }
                for fc in fail_certs
            ]
            adjusted = replace(
                review,
                decision=ReviewDecision(adjusted_decision),
                approved_output=None,
                evidence=evidence,
            )
        else:  # FR_CANDIDATE — rules 2 + 3: reviewer-reject + conclusive-pass
            adjusted = replace(review, decision=ReviewDecision(adjusted_decision), tripwire=False)

        self._emit_review_event(
            category=trace_cat,
            summary=f"verifier precedence {trace_cat}: "
            f"{review.decision.value}->{adjusted.decision.value}",
            status=verdict,
            detail={
                "kind": trace_cat,
                "phase": "verification",
                "conclusive_verdict": verdict,
                "original_decision": review.decision.value,
                "adjusted_decision": adjusted.decision.value,
                "report_id": (report or {}).get("report_id"),
                "certificates": fail_certs,
                "executor_model_id": executor_model_id,
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return adjusted

    # ── RD-5: warn-only shadow downgrade ──────────────────────────────────────

    def apply_warn_only(
        self,
        review: ArchitectReview,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
        executor_model_id: str | None = None,
    ) -> ArchitectReview:
        """Env-gated shadow downgrade of a would-be BLOCKING decision (RD-5).

        Mirrors ``safety_gate.warn_only``: when active, a blocking verdict (REJECT /
        REJECT_TO_EMPTY, or ``tripwire=True``) is downgraded to advisory
        REQUEST_CHANGES and LOGGED — never enforced. When ``warn_only`` is off the
        review is returned unchanged (enforcement remains separately gated by
        ``review_decision_enforce``, which is blocked on H-LB LB-6).
        """
        if not self.warn_only:
            return review
        is_blocking = review.tripwire or review.decision in (
            ReviewDecision.REJECT,
            ReviewDecision.REJECT_TO_EMPTY,
        )
        if not is_blocking:
            return review
        downgraded = replace(review, decision=ReviewDecision.REQUEST_CHANGES, tripwire=False)
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"warn_only downgrade {review.subtask_id}",
            status="warn_only",
            detail={
                "mode": "warn_only_downgrade",
                "phase": "decision",
                "warn_only_active": True,
                "original_decision": review.decision.value,
                "downgraded_decision": downgraded.decision.value,
                "original_tripwire": review.tripwire,
                "executor_model_id": executor_model_id,
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return downgraded

    # ── RD-8: reject-admissibility + escalate stub ────────────────────────────

    @staticmethod
    def check_reject_admissibility(review: ArchitectReview) -> bool:
        """True if a REJECT/REJECT_TO_EMPTY carries ≥1 objective-evidence item.

        Non-reject decisions are trivially admissible. Objective-evidence kinds are
        gate_result / test_result / scorer_result (RD-8).
        """
        if review.decision not in (ReviewDecision.REJECT, ReviewDecision.REJECT_TO_EMPTY):
            return True
        return any(
            (e or {}).get("kind") in OBJECTIVE_EVIDENCE_KINDS for e in (review.evidence or [])
        )

    def mark_reject_admissibility(
        self,
        review: ArchitectReview,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
        executor_model_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a decision artifact flagging ``unverified_rejection`` (RD-8).

        A reject with no objective evidence is marked ``unverified_rejection=True`` in
        the emitted artifact + a trace event and (in shadow mode) recorded but never
        acted on. The returned dict is a superset of ``ArchitectReview.to_dict()`` so
        the ArchitectReview object itself stays schema-compatible.
        """
        admissible = self.check_reject_admissibility(review)
        unverified = (
            review.decision in (ReviewDecision.REJECT, ReviewDecision.REJECT_TO_EMPTY)
            and not admissible
        )
        artifact = review.to_dict()
        artifact["unverified_rejection"] = unverified
        if unverified:
            self._emit_review_event(
                category=CAT_REVIEW_DECISION,
                summary=f"unverified_rejection {review.subtask_id}",
                status="unverified_rejection",
                detail={
                    "mode": "reject_admissibility",
                    "phase": "decision",
                    "decision": review.decision.value,
                    "unverified_rejection": True,
                    "reason": "reject without objective evidence (gate/test/scorer)",
                    "evidence_kinds": [(e or {}).get("kind") for e in (review.evidence or [])],
                    "executor_model_id": executor_model_id,
                },
                session_id=session_id,
                trial_id=trial_id,
            )
        return artifact

    def escalate(
        self,
        review: ArchitectReview,
        *,
        reason: str = "",
        session_id: Any | None = None,
        trial_id: int | None = None,
        executor_model_id: str | None = None,
    ) -> ArchitectReview:
        """RD-8 escalate stub: route to the escalation pipeline surface.

        Emits a REVIEW_ESCALATION trace event and returns the decision unchanged.
        No UI (HS-4-gated). This is the server-side stub surface the escalation
        pipeline consumes.
        """
        self._emit_review_event(
            category=CAT_REVIEW_ESCALATION,
            summary=f"escalate {review.subtask_id}",
            status="escalate",
            detail={
                "phase": "escalation",
                "subtask_id": review.subtask_id,
                "decision": review.decision.value,
                "confidence": review.confidence,
                "reason": reason or review.feedback,
                "executor_model_id": executor_model_id,
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return review

    # ── RD-3/5/8 shadow orchestration entry point ─────────────────────────────

    def shadow_decide(
        self,
        review: ArchitectReview,
        *,
        verification_report: dict[str, Any] | None = None,
        session_id: Any | None = None,
        trial_id: int | None = None,
        latency_ms: float | None = None,
        tokens: dict[str, int] | None = None,
        executor_model_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the full SHADOW decision pipeline over a review (RD-3/5/8).

        Order: verifier precedence → warn-only downgrade → escalate-stub (if escalate)
        → reject-admissibility marking. Emits trace only; never enforces or mutates
        external state. Returns a decision artifact dict (superset of
        ``ArchitectReview.to_dict()``). This is the single entry point a caller in
        shadow mode (gated by ``review_decision_shadow``) invokes.

        RD-12: when ``latency_ms`` / ``tokens`` are supplied, the returned artifact
        carries a schema-aligned ``telemetry`` block (``wall_ms`` / ``tokens_in`` /
        ``tokens_out`` — the channels ``review_decision_to_ledger_row`` reads, so the
        artifact feeds the H4 ledger + H-LB).

        TM-8: EVERY invocation emits one REVIEW_DECISION trace row (``phase=
        "decision"``), so a review invocation that exercises no sub-step still
        produces a trace row — this is what makes the coverage gate measurable.
        """
        r = review
        if verification_report is not None:
            r = self.apply_verifier_precedence(
                r, verification_report, session_id=session_id, trial_id=trial_id,
                executor_model_id=executor_model_id,
            )
        r = self.apply_warn_only(
            r, session_id=session_id, trial_id=trial_id, executor_model_id=executor_model_id,
        )
        if r.decision == ReviewDecision.ESCALATE:
            self.escalate(
                r, session_id=session_id, trial_id=trial_id, executor_model_id=executor_model_id,
            )
        artifact = self.mark_reject_admissibility(
            r, session_id=session_id, trial_id=trial_id, executor_model_id=executor_model_id,
        )
        artifact["shadow"] = True
        # RD-12: schema-aligned telemetry block (telemetry.wall_ms / tokens_in / tokens_out).
        if latency_ms is not None or tokens:
            artifact["telemetry"] = {
                "wall_ms": latency_ms,
                "tokens_in": (tokens or {}).get("tokens_in", 0),
                "tokens_out": (tokens or {}).get("tokens_out", 0),
            }
        # TM-8: always-on emission so every shadow invocation yields a trace row
        # (the sub-steps above only emit on disagreement/downgrade/admissibility).
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"shadow_decide {r.subtask_id}: {r.decision.value}",
            status=r.decision.value,
            detail={
                "mode": "shadow_decide",
                "phase": "decision",
                "shadow": True,
                "decision": r.decision.value,
                "confidence": r.confidence,
                "tripwire": r.tripwire,
                "unverified_rejection": artifact.get("unverified_rejection"),
                "executor_model_id": executor_model_id,
                "latency_ms": latency_ms,
                "tokens": tokens or {"tokens_in": 0, "tokens_out": 0, "chars_out": 0},
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return artifact

    # ── RD-9: plan-review specifics ───────────────────────────────────────────

    def review_plan_rubric(
        self,
        objective: str,
        task_type: str,
        plan_steps: list[dict[str, Any]],
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
        executor_model_id: str | None = None,
    ) -> dict[str, Any]:
        """Structured plan review keyed to phase-coverage/order/executor-alignment (RD-9).

        Returns a structured dict (NOT ``PlanReviewResult``) so it is additive to the
        legacy ``review_plan()`` path. Prose quality is deliberately ignored;
        over-specification is penalized like gaps. Shadow-only; emits a trace event.

        TD-21.6: schema on the wire + ONE ``parse_with_repair`` turn on a miss. A
        terminal failure -- either a model-call exception or a repair-exhausted
        parse -- now returns ``decision=PARSE_FAILURE_SENTINEL`` ("parse_failure",
        the literal string ``src.trace.review_ledger.PARSE_FAILURE_DECISIONS``
        matches) with the phase_coverage/order/executor_alignment axes left
        ``None`` (unknown), instead of the old silent "approve" + axes-default-True
        that was indistinguishable from a genuine reviewer verdict.
        """
        steps = self._render_plan_steps(plan_steps)
        prompt = self.PLAN_REVIEW_RUBRIC_PROMPT.format(
            objective=str(objective)[:200], task_type=task_type, steps=steps
        )

        start = time.perf_counter()
        response: Any = ""
        model_call_failed = False
        repaired = False
        try:
            response = self._llm_call_no_think(
                prompt,
                role=self.architect_role,
                n_tokens=self.max_plan_review_tokens,
                json_schema=PLAN_RUBRIC_SCHEMA,
            )
        except Exception as exc:
            logger.warning("review_plan_rubric model call failed: %s", exc)
            response = ""
            model_call_failed = True
            self._model_call_failures += 1

        parse_failed = False
        parse_failure_reason: str | None = None
        data: dict[str, Any] = {}
        if model_call_failed:
            parse_failed = True  # no response was ever produced -- no verdict to trust
        elif response:
            outcome = parse_with_repair(
                response,
                schema=PLAN_RUBRIC_SCHEMA,
                complete=primitives_completer(
                    self.primitives, self.architect_role, n_tokens=self.max_plan_review_tokens
                ),
                site="review_service.review_plan_rubric",
                # TD-21.34: `confidence` is REQUIRED (same fabrication risk as
                # `review_candidate`'s); `advisory.score`/`advisory.feedback`
                # must come from the reviewer's own reply.
                # `phase_coverage`/`order`/`executor_alignment` are booleans
                # (auto-exempt already). `decision` is exempt (classification).
                require_evidence=True,
                evidence_exempt={"decision"},
            )
            if outcome.status in ("parsed", "repaired"):
                data = outcome.value
                repaired = outcome.status == "repaired"
            else:
                parse_failed = True
                parse_failure_reason = outcome.reason or "unparseable_response"
        else:
            parse_failed = True

        if parse_failed and not model_call_failed:
            self._parse_failure_count += 1

        if parse_failed:
            result = {
                "decision": PARSE_FAILURE_SENTINEL,
                "confidence": 0.0,
                "phase_coverage": None,
                "order": None,
                "executor_alignment": None,
                "score": 0.0,
                "feedback": f"parse_failure:{parse_failure_reason or 'model_call_failed'}",
            }
        else:
            decision = data.get("decision", data.get("d", "approve"))
            valid = {"approve", "request_changes", "reject_to_empty", "escalate"}
            if decision not in valid:
                decision = "approve"
            advisory = data.get("advisory") or {}
            result = {
                "decision": decision,
                "confidence": float(data.get("confidence", 0.5) or 0.5),
                "phase_coverage": bool(data.get("phase_coverage", True)),
                "order": bool(data.get("order", True)),
                "executor_alignment": bool(data.get("executor_alignment", True)),
                "score": float(advisory.get("score", data.get("s", 0.5)) or 0.5),
                "feedback": str(advisory.get("feedback", data.get("f", ""))),
            }
        latency_ms = (time.perf_counter() - start) * 1000.0
        parse_ok = not parse_failed
        # RC-9: persistable rubric + per-item snapshots. Grades only when the
        # emission parsed — the axis values above are None (never a silent
        # default) otherwise.
        rubric_snapshot = {
            "rubric_id": "plan_rubric",
            "template_sha256": hashlib.sha256(
                self.PLAN_REVIEW_RUBRIC_PROMPT.encode("utf-8")
            ).hexdigest(),
            "items": list(PLAN_RUBRIC_AXES),
        }
        per_item_grades = (
            [{"item": axis, "binary": int(result[axis])} for axis in PLAN_RUBRIC_AXES]
            if parse_ok
            else None
        )
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"plan_rubric: {result['decision']}",
            status=result["decision"],
            detail={
                "mode": "plan_rubric",
                "phase": "plan",
                **result,
                "rubric": rubric_snapshot,
                "per_item_grades": per_item_grades,
                "parse_ok": parse_ok,
                "parse_failure": (parse_failure_reason or "unparseable_response")
                if (parse_failed and not model_call_failed)
                else None,
                "repaired": repaired,
                "model_call_failed": model_call_failed,
                "executor_model_id": executor_model_id or _plan_executors(plan_steps),
                "latency_ms": latency_ms,
                "tokens": self._response_tokens(response, prompt),
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return result

    @staticmethod
    def plan_review_reject_to_empty_fallback(decision: "ReviewDecision | str") -> bool:
        """RD-9: True when a plan-review verdict means 'discard the plan → no-plan default'.

        A ``reject_to_empty`` plan is worse than none: the caller, on True, drops the
        plan and proceeds with its default no-plan workflow. WIRE-POINT (documented for
        the delegator owner; delegator.py is NOT edited here): the plan-execution loop
        calls this on the plan verdict and, on True, replaces the plan with the no-plan
        default rather than iterating on a bad plan.
        """
        d = decision.value if isinstance(decision, ReviewDecision) else str(decision)
        return d == "reject_to_empty"

    def build_plan_reminder(
        self,
        approved_plan: list[dict[str, Any]] | None,
        *,
        cadence_n: int = 5,
        step_index: int = 0,
        emit: bool = False,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> str | None:
        """RD-9 cheap plan-reminder re-injection knob (PREFERRED over re-review).

        Given an APPROVED plan and a cadence N, return a compact reminder message to
        re-inject every N steps/tool-calls (fires when ``step_index > 0`` and
        ``step_index % cadence_n == 0``), else ``None``.

        WIRE-POINT (documented for the delegator owner; delegator.py NOT edited here):
        the execution loop calls this each step and, on a non-``None`` return, prepends
        the reminder to the next specialist prompt. Set ``emit=True`` to also record a
        PLAN_REMINDER trace event (needed for plan-compliance metrics, intake-835).
        """
        if not approved_plan or cadence_n <= 0 or step_index <= 0 or step_index % cadence_n != 0:
            return None
        steps = self._render_plan_steps(approved_plan)
        message = "Reminder — you are executing this approved plan; stay on it:\n" + steps
        if emit:
            self._emit_review_event(
                category=CAT_PLAN_REMINDER,
                summary=f"plan reminder @step {step_index}",
                status="reminder",
                detail={
                    "phase": "reminder",
                    "step_index": step_index,
                    "cadence_n": cadence_n,
                    "n_steps": len(approved_plan),
                    "executor_model_id": _plan_executors(approved_plan),
                },
                session_id=session_id,
                trial_id=trial_id,
                # UTM-P1a: the plan step index is the one genuine turn index the
                # review plane holds (0-based per the schema; step_index > 0 here).
                turn_ordinal=step_index,
            )
        return message

    @staticmethod
    def iteration_bound_by_compliance_trend(
        compliance_trend: list[float] | None,
        *,
        drift_threshold: float = 0.5,
        collapse_threshold: float = 0.2,
    ) -> str:
        """RD-9 policy HOOK (intended policy documented; minimal implementation).

        INTENDED POLICY: iteration bounds are keyed to the plan-compliance trend —
        ``drift`` → cheap reminder re-injection; ``collapse`` → re-plan. This hook maps
        a per-step compliance-score sequence (each in [0,1]) to one of
        ``'ok' | 'reminder' | 'replan'``. It is the surface the delegator/autopilot
        policy will call; the numeric thresholds are RD-11 tuning-surface candidates
        (declared to ``config_applicator.apply_params``, not owned here).
        """
        if not compliance_trend:
            return "ok"
        latest = float(compliance_trend[-1])
        if latest <= collapse_threshold:
            return "replan"
        if latest <= drift_threshold:
            return "reminder"
        return "ok"
