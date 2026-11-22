"""AKU-06l3: one lifecycle proof for a v2 operation child, across a controller restart.

A child is proven, in order, by (1) the current lifecycle's exact current-binding terminal
and provider-held receipt; (2) an earlier generation's terminal this lifecycle owns exactly
(reconciled after restart) with its retained held receipt, the proof v1 accepts for a
recovered attempt; or (3) when the lifecycle holds neither, the controller's
accounting-only restored held index (``restored_stale_held_accounting``: the AKU-07j/07h
authority already used for restored v1 held settlements).
The restored entry must match the exact terminal and must be unbound or bound to the same
accounting binding. Neither path grants result or stdout authority; both are read-only."""
from __future__ import annotations

from typing import Any, Mapping

from . import worker_lifecycle

_MISSING = object()


class ChildProofRefused(Exception):
    """The supplied child terminal is not the exact current or restored lifecycle proof."""


def _key(terminal: Any) -> tuple[str, str, str, str]:
    return (terminal.request_id, terminal.plan_digest, terminal.lineage_id, terminal.stage_id)


def child_proof(lifecycle: Any, restored_index: Mapping[tuple, Any], terminal: Any,
                binding_digest: str | None) -> tuple[Any, Any]:
    """Return ``(terminal, held receipt)`` proving one child, or ``(None, None)``.

    The current lifecycle wins whenever it has a current-binding terminal for the key.
    Its refusal on an earlier generation's terminal ("ambiguous across worker
    generations") falls back to the restored index only when that index holds the key."""
    request_id, plan_digest, lineage_id, stage_id = _key(terminal)
    entry = restored_index.get(_key(terminal), _MISSING)
    try:
        exact = lifecycle.terminal_for_request(
            request_id=request_id, plan_digest=plan_digest,
            lineage_id=lineage_id, stage_id=stage_id)
    except worker_lifecycle.LifecycleRefused as lookup_refused:
        # An earlier generation's terminal this lifecycle reconciled after restart: the
        # exact owned terminal with its retained held receipt (the v1 recovered proof).
        try:
            return terminal, lifecycle.trusted_held_claim_receipt(terminal)
        except (worker_lifecycle.LifecycleRefused, worker_lifecycle.WaitingAuthority):
            if entry is _MISSING:
                raise lookup_refused from None
        exact = None
    if exact is not None:
        return exact, lifecycle.trusted_held_claim_receipt(exact)
    if entry is _MISSING:
        return None, None
    if entry is None:
        raise ChildProofRefused("restored held accounting key is reused (ambiguous)")
    restored, receipt, bound = entry[0], entry[1], entry[2]
    if worker_lifecycle.TerminalWorker(**restored.__dict__) != terminal:
        raise ChildProofRefused("restored held accounting terminal differs")
    if bound not in {None, binding_digest}:
        raise ChildProofRefused("restored held accounting is bound to another selection")
    return worker_lifecycle.TerminalWorker(**restored.__dict__), receipt


_TERMINAL_FIELDS = ("worker_id", "worker_generation", "request_id", "plan_digest",
                    "lineage_id", "stage_id", "grant_id", "grant_generation",
                    "container_id", "return_code", "result_digest", "accepted", "reason")


def terminal_body(terminal: Any) -> dict[str, Any]:
    """The child record's closed terminal body."""
    return {name: getattr(terminal, name) for name in _TERMINAL_FIELDS}


def duplicate_child(chain, terminal: Any, receipt: Any, role: str,
                    binding_digest: str) -> dict[str, Any] | None:
    """An exact retry of an already durable child is a duplicate (also after settlement,
    so a lost settle reply retries idempotently); a conflicting retry refuses."""
    prior = next((c for c in chain if c["request_id"] == terminal.request_id
                  or c["receipt"]["receipt_id"] == receipt.receipt_id), None)
    if prior is None:
        return None
    if (prior["role"] == role and prior["terminal"] == terminal_body(terminal)
            and prior["receipt"] == receipt.to_dict()
            and prior["binding_digest"] == binding_digest):
        return {"status": "duplicate", "digest": prior["digest"], "role": role}
    raise ChildProofRefused("retry differs from durable record")


__all__ = ["ChildProofRefused", "child_proof", "duplicate_child", "terminal_body"]
