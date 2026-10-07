"""memory_record.py — the single shape every episodic memory is written in.

WHY THIS EXISTS
---------------
A 2026-07-27 audit of the episodic store found that its six write sites had each
grown their own record shape and their own embedding text, with four measurable
consequences:

1. **Two incompatible key names for the same field.** 30,571 rows put the task
   text in ``objective``; 27,562 put it in ``task_description`` alongside
   telemetry (``question_id``, ``elapsed_seconds``, ``tokens_generated``, ...).
   Both are real task memories — an earlier reading of this split as "half the
   store is telemetry with no task text" was WRONG and is corrected here — but a
   consumer that reads only one key silently sees half the store, which is
   exactly how that misreading happened. The contract has one field name.
2. **Objectives were truncated to 200 chars at write time**, so the text was
   *gone*, not merely unindexed. 12.6% of rows sat exactly at the cap, and only
   2,639 distinct objectives existed across 54,960 rows (96.9% collision).
3. **At least four embedding conventions** coexisted, so the vector encoded
   *which writer produced the row* — a probe read the writer path off the
   embedding at ROW-AUC 0.906-0.940, and the two paths differed up to 8x in
   failure rate.
4. **Nothing stored the work.** No answer, no tool calls, no REPL steps, no
   reasoning. So retrieval could only ever return "here is a similar task stub,
   and it succeeded" — never "here is how it was solved". That ceiling is
   visible in SkillBank's output: all 57 distilled skills are thin routing
   heuristics, because a routing heuristic is the only thing extractable from
   ``(stub, role, success-bit)``.

Operator decision, 2026-07-27: store the works — answers, tool calls, REPL
steps, reasoning traces. Storage is not the constraint (answers measure ~1.1 KB
mean; p50 is 58 tokens), and distillation is measured at 10:1.

THE ONE INVARIANT
-----------------
**What gets EMBEDDED is not what gets STORED.**

    embedding_text()  -> the task only, one convention, bounded length.
                         This is what similarity search matches on, so it must
                         describe the QUESTION. Embedding the answer would make
                         retrieval match solutions against solutions.

    to_context()      -> the full record: untruncated objective plus the work.
                         Never fed to the embedder.

Telemetry rides in ``metrics`` and is excluded from the embedding text by
construction, so performance fields can never influence similarity search.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

#: Cap on the text handed to the embedder. BGE-large truncates around 512
#: tokens anyway, so a generous character bound loses nothing while keeping the
#: convention stable. THIS DOES NOT TRUNCATE STORED TEXT — see to_context().
EMBED_TEXT_MAX_CHARS = 2000

#: Schema marker on every stored context, so a future reader can tell records
#: written under this contract from the pre-2026-07-27 free-form ones.
RECORD_VERSION = 1

# ---------------------------------------------------------------------------
# Work-payload capture policy (M-11a2b, 2026-08-03)
# ---------------------------------------------------------------------------
# The work fields below existed from 2026-07-27 but nothing ever passed them, so
# 0 of 59,337 rows carried `work`. Wiring them means episodic rows now hold model
# OUTPUT and TOOL OUTPUT, which is the first content in this store that can carry
# a credential or an unbounded blob. Two properties are enforced here, at the
# single construction chokepoint (`build_memory_record`), rather than at each
# write site — the same reasoning as the degenerate-embedding refusal in
# EpisodicStore.store(): a guarantee placed at N call sites is lost by caller
# N+1.
#
# 1. REDACTION reuses the repo's ONE credential policy,
#    src.repl_environment.redaction.redact_if_enabled (feature-gated on
#    credential_redaction, default True). No second pattern list is invented
#    here. Note redact_credentials() SKIPS inputs above 1 MB, so the caps below
#    also keep every scanned string inside the scanner's working range.
# 2. CAPS are the repo's existing values, not new ones:
#      WORK_TEXT_MAX_CHARS  = src/tools/file/read.py's stored-text cap (32_000)
#      WORK_MAX_ITEMS       = REPLState.CODE_LOG_MAX_STEPS (200)
#      WORK_ITEM_MAX_CHARS  = REPLState.CODE_LOG_MAX_CHARS (4_000)
#    Sizing sanity: answers measure ~1.1 KB mean / 58 tokens p50, so the text cap
#    is ~29x the mean and does not bite in normal operation — it exists to bound
#    a looping generation or a dumped file, not to compress ordinary work.
#
# The objective is deliberately NOT capped: destroying it at write time was the
# measured 2026-07-27 defect and is not reintroduced here.

#: Per-field cap on stored work TEXT (answer, reasoning).
WORK_TEXT_MAX_CHARS = 32_000
#: Cap on retained tool_calls / repl_steps entries.
WORK_MAX_ITEMS = 200
#: Per-entry cap on a serialized tool_call / repl_step.
WORK_ITEM_MAX_CHARS = 4_000


def _redact(text: str) -> str:
    """Apply the repo's single credential-redaction policy, failing open.

    Imported lazily: memory_record is imported by offline maintenance scripts
    that must not drag in the API feature system.
    """
    try:
        from src.repl_environment.redaction import redact_if_enabled

        return redact_if_enabled(text)
    except Exception:  # pragma: no cover - redaction must never fail a write
        return text


#: Recognises this module's own truncation marker, so a second pass over an
#: already-bounded string can tell "I did this" from "this needs doing".
_TRUNCATION_MARKER_RE = re.compile(
    r"\n\n\[\.\.\. truncated at ([0-9]{1,20}) chars, total was ([0-9]{1,20})\]\Z"
)


def sanitize_work_text(text: Any, max_chars: int = WORK_TEXT_MAX_CHARS) -> str | None:
    """Redact credentials from, then bound, one work text field.

    Redaction runs BEFORE truncation so a secret is matched against the whole
    string, and again AFTER when truncation actually fired — the second pass is
    what covers inputs above redact_credentials()' 1 MB scan ceiling, where the
    first pass is a no-op by design.

    IDEMPOTENT (fixed 2026-09-14, RTG-02). The policy is applied twice on the
    live create path — once by `chat_pipeline.telemetry.work_completion_meta` on
    its way into the progress JSONL, then again by `build_memory_record` on the
    way into `memories.context` — and the docstring claimed idempotence the code
    did not have. A string already truncated to `max_chars` is `max_chars` PLUS
    the marker long, so the second pass re-truncated it and appended a SECOND
    marker reporting `total was <max_chars + len(marker)>`: the row's own
    provenance actively lied about how much was elided (measured: 32,500 real
    chars reported as 32,049). An already-marked value at this same cap is
    returned unchanged only when the marker starts within the cap. A marker
    appended to an oversized untrusted value is not provenance: accepting it
    would bypass both the storage bound and (for inputs above 1 MB) the second
    redaction pass.
    """
    if text is None:
        return None
    value = text if isinstance(text, str) else str(text)
    value = _redact(value)
    if len(value) > max_chars:
        marker = _TRUNCATION_MARKER_RE.search(value)
        if (marker and int(marker.group(1)) == max_chars
                and marker.start() <= max_chars):
            # Only trust our marker when its prefix is itself within the cap.
            # Otherwise arbitrary input could forge a suffix marker to bypass
            # both the size bound and the post-truncation redaction pass.
            return value
        original_len = len(value)
        value = _redact(value[:max_chars])
        value += f"\n\n[... truncated at {max_chars} chars, total was {original_len}]"
    return value


def sanitize_work_items(items: Any, max_items: int = WORK_MAX_ITEMS) -> list[Any]:
    """Bound a tool_calls / repl_steps list and redact the text inside it.

    Keeps the LAST `max_items` entries (a truncated trajectory's tail is the part
    that produced the answer) and records the drop in a sentinel entry so a
    reader can never mistake a bounded list for a complete one.

    IDEMPOTENT (fixed 2026-09-14, RTG-02) — see sanitize_work_text for why the
    policy runs twice on the live create path. A previously bounded list is
    `max_items` entries PLUS the `_elided_entries` sentinel, so the second pass
    saw max_items+1, dropped one entry — which, being at the FRONT, was the
    sentinel itself — and inserted a fresh `_elided_entries: 1`. The row then
    claimed 1 entry was elided when 50 were, AND silently lost the real first
    retained entry. The leading sentinel is now recognised, lifted out before
    counting, and its count carried into the total.
    """
    if not items:
        return []
    if not isinstance(items, (list, tuple)):
        items = [items]
    entries = list(items)

    # Lift a sentinel this policy wrote on an earlier pass, so it is neither
    # counted as payload nor mistaken for the entry that should be dropped.
    already_dropped = 0
    if (
        entries
        and isinstance(entries[0], dict)
        and set(entries[0]) == {"_elided_entries"}
        and isinstance(entries[0]["_elided_entries"], int)
    ):
        already_dropped = entries[0]["_elided_entries"]
        entries = entries[1:]

    dropped = already_dropped
    if len(entries) > max_items:
        dropped += len(entries) - max_items
        entries = entries[-max_items:]

    out: list[Any] = []
    for entry in entries:
        try:
            encoded = json.dumps(entry, default=str)
        except Exception:
            encoded = json.dumps(str(entry))
        cleaned = sanitize_work_text(encoded, max_chars=WORK_ITEM_MAX_CHARS)
        if cleaned == encoded:
            # Unchanged by redaction/truncation — keep the original structure.
            out.append(entry)
            continue
        try:
            out.append(json.loads(cleaned))
        except Exception:
            # Truncation broke the JSON; keep the redacted text form instead of
            # dropping the entry, so the record still says what was tried.
            out.append({"_truncated_entry": cleaned})
    if dropped:
        out.insert(0, {"_elided_entries": dropped})
    return out


def build_work_payload(
    *,
    answer: Any = None,
    tool_calls: Any = None,
    repl_steps: Any = None,
    reasoning: Any = None,
) -> dict[str, Any]:
    """The ONE producer of a `work` dict, for both stores.

    Returns {} when there is nothing to record, so a caller can splat it into a
    telemetry dict unconditionally without inventing an empty-work convention.
    """
    payload: dict[str, Any] = {}
    answer_text = sanitize_work_text(answer)
    if answer_text:
        payload["answer"] = answer_text
    calls = sanitize_work_items(tool_calls)
    if calls:
        payload["tool_calls"] = calls
    steps = sanitize_work_items(repl_steps)
    if steps:
        payload["repl_steps"] = steps
    reasoning_text = sanitize_work_text(reasoning)
    if reasoning_text:
        payload["reasoning"] = reasoning_text
    return payload


#: Keys a caller may use to hand work in through a flat context dict. Kept in one
#: place so `score_external_result` and the legacy adapter agree on the set.
WORK_KEYS = ("answer", "tool_calls", "repl_steps", "reasoning")


def extract_work(source: Any) -> dict[str, Any]:
    """Pull a work payload out of a dict that may nest it or carry it flat.

    Accepts both ``{"work": {"answer": ...}}`` (the progress-log/telemetry shape)
    and ``{"answer": ...}`` (a caller passing work fields at the top level of a
    context dict). Returns {} when neither is present.
    """
    if not isinstance(source, dict):
        return {}
    nested = source.get("work")
    if isinstance(nested, dict) and nested:
        return {k: nested.get(k) for k in WORK_KEYS if nested.get(k)}
    return {k: source.get(k) for k in WORK_KEYS if source.get(k)}


#: Separator between ``key:value`` segments in every embedding convention.
EMBED_SEGMENT_SEPARATOR = " | "


def join_embedding_segments(segments: Any, max_chars: int = EMBED_TEXT_MAX_CHARS) -> str:
    """THE shared shape of every embedding text: ``key:value | key:value``.

    Every embedder input in the episodic memory family -- the task convention
    (``MemoryRecord.embedding_text``) and the failure / exploration /
    classification conventions in ``embedder.py`` -- is assembled here, so the
    separator, the ``None`` rule and the length bound cannot drift between
    writers (EPD-3-R8; the same defect EPD-3-R2/R3 fixed for the task text).

    Rules:
      * ``segments`` is an ordered iterable of ``(key, value)``; order is kept.
      * ``value is None`` drops the segment. A caller that wants an optional
        field dropped when EMPTY passes ``value or None``; a caller that wants
        a mandatory segment passes ``""`` and gets ``key:``. The literal text
        ``None`` is never emitted.
      * The joined text is capped at ``max_chars`` (default
        ``EMBED_TEXT_MAX_CHARS``). Per-field truncation stays with the caller,
        because it is part of that convention's meaning.
    """
    parts = [f"{key}:{value}" for key, value in segments if value is not None]
    return EMBED_SEGMENT_SEPARATOR.join(parts)[:max_chars]


@dataclass
class MemoryRecord:
    """One episodic memory, in the shape every write site must produce."""

    # ---- the task (drives the embedding) ----
    objective: str
    task_type: str | None = None
    priority: str | None = None

    # ---- the work (stored, never embedded) ----
    answer: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    repl_steps: list[Any] = field(default_factory=list)
    reasoning: str | None = None

    # ---- provenance and telemetry (stored, never embedded) ----
    source: str = "unknown"
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def embedding_text(self) -> str:
        """The ONE embedding convention, for every write site.

        Matches the historical ``type: | objective: | priority:`` shape so
        vectors written before and after this change remain comparable, but is
        now produced in exactly one place instead of four.
        """
        return join_embedding_segments(
            (
                ("type", self.task_type or None),
                ("objective", (self.objective or "").strip()),
                ("priority", self.priority or None),
            )
        )

    def to_context(self) -> dict[str, Any]:
        """The stored payload. This method truncates nothing.

        ``objective`` is the complete text: the old 200-char cap destroyed it at
        write time and is not reproduced.

        The work fields arrive already redacted and already bounded — the
        capture policy is applied once, in ``build_memory_record``, so it cannot
        be lost by a write site that assembles a record by hand and calls this
        directly.
        """
        ctx: dict[str, Any] = {
            "record_version": RECORD_VERSION,
            "task_type": self.task_type,
            "objective": self.objective,
            "priority": self.priority,
            "source": self.source,
        }
        work: dict[str, Any] = {}
        if self.answer is not None:
            work["answer"] = self.answer
        if self.tool_calls:
            work["tool_calls"] = self.tool_calls
        if self.repl_steps:
            work["repl_steps"] = self.repl_steps
        if self.reasoning is not None:
            work["reasoning"] = self.reasoning
        if work:
            ctx["work"] = work
        if self.metrics:
            ctx["metrics"] = self.metrics
        if self.extra:
            ctx["extra"] = self.extra
        return ctx

    def is_task_memory(self) -> bool:
        """True when this row belongs in the semantic index at all.

        A record with no task text under EITHER historical key is pure telemetry;
        embedding it would produce a vector of a number-blob. Measured on the
        2026-07-27 store this is 0 rows — both writer paths do carry task text,
        just under different key names — so this is a guard against future drift
        rather than a cleanup of an existing population.
        """
        return bool((self.objective or "").strip())


def embedding_text_for(
    objective: str | None,
    task_type: str | None = None,
    priority: str | None = None,
) -> str:
    """The canonical embedding text for a loose (objective, task_type, priority).

    THE ONE ENTRY POINT for write sites that hold the task fields but not a
    ``MemoryRecord`` — a TaskIR dict, a seed row, a reward-injection payload.
    It exists so those sites cannot re-spell the convention: every one of them
    was measured drifting from ``MemoryRecord.embedding_text()`` in different
    ways (EPD-3-R2/R3, 2026-08-12) — key PRESENCE instead of truthiness (so
    ``priority:None`` could be emitted), no ``.strip()``, no length cap, extra
    ``constraints:``/``input_types:`` segments, a hard-coded ``type:chat``, and
    one site that embedded raw text with no prefix at all.

    This is deliberately a thin delegation: the convention itself lives in
    ``MemoryRecord.embedding_text()`` and nowhere else.
    """
    return MemoryRecord(
        objective=objective or "",
        task_type=task_type,
        priority=priority,
    ).embedding_text()


def build_memory_record(
    *,
    objective: str | None,
    task_type: str | None = None,
    priority: str | None = None,
    answer: str | None = None,
    tool_calls: list[Any] | None = None,
    repl_steps: list[Any] | None = None,
    reasoning: str | None = None,
    source: str = "unknown",
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> MemoryRecord:
    """Construct a MemoryRecord. The only supported way to build one.

    The work fields are redacted and bounded HERE, so every write site inherits
    the policy and no call site can opt out of it (see the capture-policy note at
    the top of this module). Sanitizing is idempotent: an already-bounded,
    already-redacted payload passes through unchanged, which keeps
    record_from_legacy_context(rec.to_context()) a fixed point.
    """
    work = build_work_payload(
        answer=answer,
        tool_calls=tool_calls,
        repl_steps=repl_steps,
        reasoning=reasoning,
    )
    return MemoryRecord(
        objective=objective or "",
        task_type=task_type,
        priority=priority,
        answer=work.get("answer"),
        tool_calls=list(work.get("tool_calls") or []),
        repl_steps=list(work.get("repl_steps") or []),
        reasoning=work.get("reasoning"),
        source=source,
        metrics=dict(metrics or {}),
        extra=dict(extra or {}),
    )


def record_from_legacy_context(context: dict[str, Any]) -> MemoryRecord:
    """Adapt a pre-2026-07-27 free-form context dict to a MemoryRecord.

    Both historical writer shapes are recognised:

      path A (progress log): ``{task_type, objective, priority}``
      path B (external):     ``{task_description, source, question_id,
                                elapsed_seconds, tokens_generated, ...}``

    Both carry real task text; only the key name differs. Path B's remaining
    non-task keys are routed into ``metrics``, so they are stored but can never
    reach the embedding.
    """
    ctx = dict(context or {})
    # Current-contract fields must be unwrapped before the remaining legacy
    # bookkeeping keys are collected as metrics.  In particular, the schema
    # marker is not telemetry and must not be nested into ``metrics``.
    ctx.pop("record_version", None)
    objective = ctx.pop("objective", None) or ctx.pop("task_description", None)
    task_type = ctx.pop("task_type", None)
    priority = ctx.pop("priority", None)
    source = ctx.pop("source", "legacy")
    work = ctx.pop("work", {}) or {}
    metrics = ctx.pop("metrics", {}) or {}
    extra = ctx.pop("extra", {}) or {}
    if not isinstance(metrics, dict):
        metrics = {"legacy_metrics": metrics}
    if not isinstance(extra, dict):
        extra = {"legacy_extra": extra}
    return MemoryRecord(
        objective=objective or "",
        task_type=task_type,
        priority=priority,
        answer=work.get("answer"),
        tool_calls=list(work.get("tool_calls") or []),
        repl_steps=list(work.get("repl_steps") or []),
        reasoning=work.get("reasoning"),
        source=source,
        # Unknown legacy keys are telemetry/bookkeeping — stored, not embedded.
        metrics={**metrics, **ctx},
        extra=extra,
    )


def context_size_bytes(context: dict[str, Any]) -> int:
    """Serialized size of a stored context, for budget accounting."""
    return len(json.dumps(context, default=str).encode("utf-8"))
