"""Provisional CS-4 binary routing catalogue for typed-decision shadowing.

The LOCAL/ORCH strings reflect the current CS-4 handoff description only;
they are provisional until the corpus annotation schema is available and
verified. This module prepares a mockable shadow seam. It does not route a
request, enable the shadow feature, or alter an incumbent decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from src.typed_decisions import Question, QuestionKind
from src.typed_decisions.shadow import submit_shadow

PROVISIONAL_CS4_LABELS = ("LOCAL", "ORCH")
_SURFACE = "conversation.cs24_option_ii"


def build_conversation_question(
    *,
    question_id: str = "conversation_route",
    labels: Sequence[str] = PROVISIONAL_CS4_LABELS,
) -> Question:
    """Build one CHOICE question using only the provisional CS-4 binary labels.

    ``labels`` is exposed so callers/tests can pass a catalog explicitly, but
    this preparation seam refuses every catalog other than the provisional
    two-label set. It must be revisited against the actual CS-4 schema before
    any controller integration.
    """
    catalog = tuple(labels)
    if catalog != PROVISIONAL_CS4_LABELS:
        raise ValueError(
            "CS-4 routing catalog is provisional and must be exactly ('LOCAL', 'ORCH')"
        )
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("question_id must be a non-empty string")
    return Question(
        id=question_id,
        kind=QuestionKind.CHOICE,
        text="Classify this conversation turn using the provisional CS-4 routing labels.",
        options=catalog,
        criteria=(
            "Provisional labels only: use LOCAL or ORCH; the CS-4 annotation schema is not yet verified.",
        ),
    )


def submit_conversation_shadow(
    primitives,
    *,
    state: str,
    incumbent: Mapping[str, object],
    role: str,
    log_path: str | Path | None = None,
    question_id: str = "conversation_route",
    labels: Sequence[str] = PROVISIONAL_CS4_LABELS,
) -> bool:
    """Submit a JSON-mode binary shadow; never changes the incumbent decision.

    All operational gates, including the default-off feature flag, configured
    sink and bounded nonblocking queue, remain owned by generic ``submit_shadow``.
    Confidence stays in the existing telemetry record and is never thresholded.
    Invalid provisional catalogs and unexpected submit failures fail open.
    """
    try:
        question = build_conversation_question(question_id=question_id, labels=labels)
        return submit_shadow(
            primitives,
            surface=_SURFACE,
            state=state,
            questions=(question,),
            incumbent=incumbent,
            role=role,
            log_path=log_path,
            mode="json",
        )
    except Exception:
        # The host's incumbent remains authoritative even if catalog creation
        # or shadow submission fails.
        return False
