"""AK-PM-12: the loop-side PRODUCER of execution-gated lesson-admission receipts.

A KEPT attempt row (status "kept", with its `champion_head` commit) becomes one
`LESSON_ADMISSION_EVALUATED` receipt through `lesson_native.produce`.  This module is a pure
adapter: it starts no process and no hardware, grades nothing, and never raises into the
caller (a refusal is returned as a reason; the loop's keep is unaffected).

The cap is the hard part.  `lesson_native._cap_source_valid` refuses every
`workload_roofline` provenance until an evidence verifier AND a selected workload exist, so
an operational keep today records `UNKNOWN / cap_missing` -- an honest write-side receipt, no
native re-run, nothing admitted.  A cap table passed here is forwarded unchanged; whether it
authorises a native run is decided by `produce`, never by this adapter.

Idempotent: the receipt key `(task_id, candidate_id, commit)` is checked against the
journal's existing receipts before appending, so a replayed/resumed keep adds nothing.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
REOPEN_WHEN = ("the kernel/route this keep changed is rebuilt, or the champion moves past "
               "the commit this lesson was recorded at")


def lesson_text(row: Mapping[str, Any]) -> str:
    """The lesson is the keep's own declared mechanism + falsifier; never synthesised."""
    parts = [str(row.get(k) or "").strip() for k in ("statement", "falsifier")]
    statement, falsifier = parts
    return f"{statement} (falsifier: {falsifier})" if statement and falsifier else statement


def request_for(row: Mapping[str, Any], *, campaign_id: str = "ak-loop",
                caps: Mapping[str, Any] | None = None) -> tuple[dict | None, str]:
    """(produce kwargs or None, reason).  None when the row is not an eligible keep."""
    if row.get("status") != "kept":
        return None, "not_a_keep"
    commit = row.get("champion_head")
    if not isinstance(commit, str) or not _SHA40.fullmatch(commit):
        return None, "no_full_commit"
    mechanism = row.get("mechanism_id")
    lesson = lesson_text(row)
    if not isinstance(mechanism, str) or not mechanism.strip() or not lesson:
        return None, "no_declared_lesson"
    task_id = f"{campaign_id}:{row.get('target_surface') or '?'}:{mechanism}"
    cap = (caps or {}).get(task_id)
    kwargs = dict(task_id=task_id, candidate_id=mechanism, commit=commit, lesson=lesson,
                  reopen_when=REOPEN_WHEN, task_prompt=None, argv=None, timeout=None,
                  evaluator_sha256=None, protocol_id=None, artifact_root=None,
                  upper_cap=None if not isinstance(cap, Mapping) else cap.get("upper_cap"),
                  cap_provenance=None if not isinstance(cap, Mapping) else cap.get("cap_provenance"))
    return kwargs, "eligible"


def _already_recorded(journal: Any, kwargs: Mapping[str, Any]) -> bool:
    try:
        events = journal.read_all() if hasattr(journal, "read_all") else journal.events()
    except Exception:  # noqa: BLE001 - an unreadable view must not duplicate silently
        return False
    key = (kwargs["task_id"], kwargs["candidate_id"], kwargs["commit"])
    for event in events:
        payload = getattr(event, "payload", None) or {}
        if (payload.get("schema") == "autokernel.lesson_admission.v1"
                and (payload.get("task_id"), payload.get("candidate_id"),
                     payload.get("commit")) == key):
            return True
    return False


def submit_keep(journal: Any, row: Mapping[str, Any], *, caps: Mapping[str, Any] | None = None,
                campaign_id: str = "ak-loop",
                producer: Any = None) -> dict:
    """Record one keep's receipt.  Returns {"recorded", "reason", "decision"}; never raises."""
    kwargs, reason = request_for(row, campaign_id=campaign_id, caps=caps)
    if kwargs is None:
        return {"recorded": False, "reason": reason, "decision": None}
    if _already_recorded(journal, kwargs):
        return {"recorded": False, "reason": "already_recorded", "decision": None}
    if producer is None:
        from ..lesson_native import produce as producer
    try:
        entry = producer(journal, **kwargs)
    except Exception as exc:  # noqa: BLE001 - observe-only: a refusal never costs the keep
        return {"recorded": False, "reason": f"producer_refused: {type(exc).__name__}: {exc}"[:300],
                "decision": None}
    return {"recorded": True, "reason": entry.payload["reason"],
            "decision": entry.payload["decision"]}


def submit_keeps(journal: Any, rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> list[dict]:
    return [submit_keep(journal, row, **kwargs) for row in rows if row.get("status") == "kept"]
