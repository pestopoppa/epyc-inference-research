"""Operation preview and no-launch cancellation for the scheduler (AK-SPLIT-SCHEDULING).

Moved out of ``scheduling.py`` with no behaviour change: the functions take the
``SchedulerEngine`` and use its shared ``_validate_issued`` / ``_check_slot`` /
``_consume_slot`` / ``_replace_seed_after_attempt`` helpers; the engine methods of the
same names are bound to these functions at import time (scheduling imports this last)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .scheduling import (OperationChild, OperationalProjection, SchedulerEngine,
                         SchedulingRefused, Selection, _normalize, _text)


@dataclass(frozen=True, slots=True)
class OperationPreview:
    """Locally derived result of one atomic operation settlement or cancellation."""

    kind: str
    selection: Selection
    children: tuple[OperationChild, ...]
    outcome: str
    denial_ref: str | None
    prior: OperationalProjection
    after: OperationalProjection


def cancel_before_launch(engine: SchedulerEngine, selection: Selection | Mapping[str, Any], *,
                         denial_ref: str, denial_class: str = "terminal") -> bool:
    """Explicit terminal no-launch cancellation: one failed attempt, zero seconds.

    No held receipt exists and none is fabricated. Idempotent for the exact same
    denial reference; a different reference or a later settlement refuses."""
    selection = _normalize(selection, Selection)
    ref = _text(denial_ref, "denial_ref")
    if len(ref) != 64 or any(char not in "0123456789abcdef" for char in ref):
        raise SchedulingRefused("denial_ref must be a lowercase SHA-256")
    if denial_class != "terminal":
        raise SchedulingRefused("only a terminal denial may cancel an operation")
    prior = engine.cancelled_operations.get(selection.digest)
    if prior is not None:
        if prior != ref:
            raise SchedulingRefused("selection was cancelled with a different denial")
        return False
    if engine._records_by_selection.get(selection.digest):
        raise SchedulingRefused("selection already has accounted receipts")
    engine._validate_issued(selection, "cancellation is not an issued selection",
                            "cancellation selection differs from scheduler state")
    proposal = selection.proposal
    engine._check_slot(selection)
    engine._consume_slot(selection)
    engine._replace_seed_after_attempt(proposal, duration=0, outcome="failed")
    engine.round_seed_used = engine.round_seed_used or proposal.seed_id is not None
    engine.campaign_attempts += 1
    engine.cancelled_operations[selection.digest] = ref
    engine.issued_selection_digests = ()
    return True


def preview_accounting(engine: SchedulerEngine, selection: Selection | Mapping[str, Any],
                       children: Sequence[OperationChild], *, outcome: str) -> OperationPreview:
    """Validate one multi-child settlement atomically; the engine is not mutated.

    The preview runs on an isolated copy, so a refusal at any child leaves this
    engine bit-identical (AKU-06l)."""
    selection = _normalize(selection, Selection)
    prior = engine.operational_projection()
    trial = engine._clone()
    if not trial.account_operation(selection, children, outcome=outcome):
        raise SchedulingRefused("operation accounting preview changed nothing")
    return OperationPreview("operation", selection, tuple(children), outcome, None,
                            prior, trial.operational_projection())


def preview_cancellation(engine: SchedulerEngine, selection: Selection | Mapping[str, Any], *,
                         denial_ref: str) -> OperationPreview:
    selection = _normalize(selection, Selection)
    prior = engine.operational_projection()
    trial = engine._clone()
    if not trial.cancel_operation_before_launch(selection, denial_ref=denial_ref):
        raise SchedulingRefused("operation cancellation preview changed nothing")
    return OperationPreview("cancellation", selection, (), "failed", denial_ref,
                            prior, trial.operational_projection())


def apply_preview(engine: SchedulerEngine, preview: OperationPreview) -> bool:
    """Apply a locally derived operation preview; exact replay is idempotent."""
    if not isinstance(preview, OperationPreview):
        raise SchedulingRefused("operation preview is untyped")
    current = engine.operational_projection().projection_digest
    if current == preview.after.projection_digest:
        return False
    if current != preview.prior.projection_digest:
        raise SchedulingRefused("operation preview is stale")
    if preview.kind == "operation":
        changed = engine.account_operation(preview.selection, preview.children,
                                           outcome=preview.outcome)
    else:
        changed = engine.cancel_operation_before_launch(
            preview.selection, denial_ref=preview.denial_ref)
    if engine.operational_projection().projection_digest != preview.after.projection_digest:
        raise SchedulingRefused("operation application differs from preview")
    return changed


SchedulerEngine.cancel_operation_before_launch = cancel_before_launch
SchedulerEngine.preview_operation_accounting = preview_accounting
SchedulerEngine.preview_operation_cancellation = preview_cancellation
SchedulerEngine.apply_operation_preview = apply_preview
