"""Identical-state abstention stop: do not re-ask a planner that already abstained on
exactly the state it would see again.

Measured (AK-EFF P2 replay, 2026-11-07, `ak-eff-p2-replay-20261107/RESULT.md`): 13
consecutive DS41 planner calls (3.14 h) abstained on the unchanged worktree 00d118d44
with near-verbatim restatements of the same reason. Nothing the planner could see had
changed between the calls -- only the abstained rows themselves were appended.

Mechanism. When a lane's planner abstains, the lane records a STATE KEY: sha256 over
the lane's worktree HEAD commit and tree, the lane name and planner backend, and the
rendered planner context projected onto what can change the answer (belief, inbox,
pending/accepted work, portfolio rows, do-not-repeat history, target, program, lane
config). Before the lane's next planner call the key is recomputed; when it equals the
key of the lane's last abstention and that state has already abstained `skip_after`
times, the call is NOT made: `PlannerSameStateSkip` ends the iteration as the typed
status `planner_same_state_skip`, naming the prior abstention's attempt id. The run
then parks exactly like `provider_breaker` (folded into `should_stop`; terminal
`planner_same_state_parked`; an operator stop still wins; resumable with --resume-run).

The projection drops only what an abstention ITSELF changes: rows whose status is an
abstention, a skip or a no-answer (transient, mid-formation stop, lane error), the
pool health alarm derived from those rows, and per-row record timestamps. Any other
difference -- a new measured row, an inbox file, a belief change, a different HEAD --
changes the key and the planner is asked. An unreadable worktree yields no key and the
planner is always asked (fail-open toward calling, never toward skipping).

Also here: `retry_budget_shadow`, a LOG-ONLY annotation. A row whose mechanism already
has >= `RETRY_BUDGET_SHADOW_AFTER` prior `patch_rejected` rows in the store carries the
count and what a retry budget would have done. It never refuses anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping

PLANNER_SAME_STATE_SKIP = "planner_same_state_skip"
from .provider_breaker import SAME_STATE_PARKED  # noqa: E402 -- one spelling
DEFAULT_SKIP_AFTER = 1
#: Overrides `resources.planner_same_state_skip_after`: a non-negative integer, 0 = off.
ENV = "AK_PLANNER_SAME_STATE_SKIP_AFTER"

#: Row statuses an abstention (or a non-answer) appends without changing what the
#: planner knows about the kernel. Everything else is state.
_SELF_STATUSES = frozenset({"abstained", PLANNER_SAME_STATE_SKIP, "planner_transient",
                            "stopped_mid_formation", "lane_error"})
#: Context keys derived only from recent outcomes (incl. the abstentions themselves)
#: or rebuilt per round inside one iteration.
_VOLATILE_CONTEXT = frozenset({"loop_health_alarm", "prior_hypothesis_rejections"})
#: Per-row record bookkeeping that is not evidence.
_VOLATILE_ROW = frozenset({"turn_recorded_at", "recorded_at", "journal_receipt"})
_HISTORY_KEYS = ("prior_experiments", "shared_prior_experiments")

RETRY_BUDGET_SHADOW_AFTER = 4


def validate_skip_after(value: Any, label: str = "planner_same_state_skip_after") -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer (0 disables)")
    return value


def resolve_skip_after(configured: int | None, env: Mapping[str, str] | None = None) -> int:
    """The effective threshold: env override, else the campaign knob, else default."""
    raw = (os.environ if env is None else env).get(ENV)
    if raw is not None and raw.strip() != "":
        try:
            return validate_skip_after(int(raw.strip()), ENV)
        except ValueError as exc:
            raise ValueError(f"{ENV} must be a non-negative integer (0 disables)") from exc
    return DEFAULT_SKIP_AFTER if configured is None else validate_skip_after(configured)


def _history(rows: Any) -> Any:
    if not isinstance(rows, list):
        return rows
    kept = []
    for row in rows:
        if isinstance(row, Mapping):
            if row.get("status") in _SELF_STATUSES:
                continue
            row = {k: v for k, v in row.items() if k not in _VOLATILE_ROW}
        kept.append(row)
    return kept


def context_projection(context: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the planner context an abstention cannot change by itself."""
    out = {}
    for key, value in context.items():
        if key in _VOLATILE_CONTEXT:
            continue
        out[key] = _history(value) if key in _HISTORY_KEYS else value
    return out


def worktree_identity(workspace: Path | str) -> str | None:
    """`<HEAD> <tree> <porcelain digest>` of the lane worktree, None if unreadable."""
    try:
        head = subprocess.run(["git", "-C", str(workspace), "rev-parse", "HEAD",
                               "HEAD^{tree}"], capture_output=True, text=True, timeout=60)
        dirty = subprocess.run(["git", "-C", str(workspace), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode != 0 or dirty.returncode != 0:
        return None
    return " ".join(head.stdout.split() +
                    [hashlib.sha256(dirty.stdout.encode()).hexdigest()])


def state_key(context: Mapping[str, Any], *, worktree: str | None, lane: str,
              planner: str = "") -> str | None:
    if worktree is None:
        return None
    body = json.dumps({"worktree": worktree, "lane": lane, "planner": planner,
                       "context": context_projection(context)},
                      sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(body.encode()).hexdigest()


class PlannerSameStateSkip(Exception):
    """The planner call was skipped: its last abstention was on this exact state."""

    def __init__(self, *, lane: str, key: str, prior_abstention_id: str | None,
                 repeats: int) -> None:
        self.lane, self.key = lane, key
        self.prior_abstention_id, self.repeats = prior_abstention_id, repeats
        super().__init__(
            f"{PLANNER_SAME_STATE_SKIP}: lane {lane} planner state {key[:16]} is identical "
            f"to its last abstention ({prior_abstention_id or 'unrecorded'}, "
            f"{repeats} abstention(s) on this state); planner not called")

    def to_dict(self) -> dict[str, Any]:
        return {"lane": self.lane, "state_key": self.key,
                "prior_abstention_id": self.prior_abstention_id,
                "abstentions_on_state": self.repeats}


class SameStateGuard:
    """Per-lane last-abstention keys. A lane whose next planner state equals its last
    abstention PARKS (`park_wait`) until another lane records real evidence (the
    change epoch moves); the guard trips -- the whole run parks -- only when every
    lane is parked at once. A single-lane run trips on its first skip."""

    #: Poll granularity for `should_stop` while parked (as `pipeline._belief_wait`).
    POLL_S = 5.0

    def __init__(self, skip_after: int = DEFAULT_SKIP_AFTER, lanes: int = 1) -> None:
        self.skip_after = validate_skip_after(skip_after)
        if type(lanes) is not int or lanes < 1:
            raise ValueError("lanes must be a positive integer")
        self.lanes = lanes
        self.tripped = False
        self._announced = False
        self.last_skip: dict[str, Any] | None = None
        self.epoch = 0
        self._last: dict[str, dict[str, Any]] = {}
        self._parked: dict[str, tuple[float, str]] = {}
        self._waits: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)

    @property
    def enabled(self) -> bool:
        return self.skip_after > 0

    def check(self, lane: str, key: str | None) -> None:
        """Raise `PlannerSameStateSkip` instead of a planner call on a repeated state."""
        if not self.enabled or key is None:
            return
        with self._lock:
            last = self._last.get(lane)
            if last is None or last["key"] != key or last["repeats"] < self.skip_after:
                return
            skip = PlannerSameStateSkip(lane=lane, key=key, repeats=last["repeats"],
                                        prior_abstention_id=last.get("abstention_id"))
            self.last_skip = skip.to_dict()
        raise skip

    def park_wait(self, lane: str, should_stop=None) -> bool:
        """Park `lane` until new evidence (True: re-check), or the run parks or is
        stopped (False: skip). Every lane parked at once trips the guard."""
        import time
        from datetime import datetime, timezone
        start = time.monotonic()
        with self._changed:
            epoch0 = self.epoch
            self._parked[lane] = (start, datetime.now(timezone.utc).isoformat())
            if len(self._parked) >= self.lanes:
                self.tripped = True
                self._changed.notify_all()
            try:
                while self.epoch == epoch0 and not self.tripped:
                    if should_stop is not None and should_stop():
                        break
                    self._changed.wait(self.POLL_S)
                woke = self.epoch != epoch0 and not self.tripped
            finally:
                self._parked.pop(lane, None)
                row = self._waits.setdefault(lane, {"n": 0, "seconds": 0.0})
                row["n"] += 1
                row["seconds"] = round(row["seconds"] + time.monotonic() - start, 3)
        return woke and not (should_stop is not None and should_stop())

    def note_answer(self, lane: str, key: str | None, abstained: bool) -> None:
        """Fold one planner answer: an abstention records (or repeats) its key."""
        with self._lock:
            if not abstained or key is None:
                self._last.pop(lane, None)
                return
            last = self._last.get(lane)
            repeats = last["repeats"] + 1 if last is not None and last["key"] == key else 1
            self._last[lane] = {"key": key, "repeats": repeats, "abstention_id": None,
                                "pending": True}

    def observe(self, outcome: Any) -> bool:
        """Fold one RECORDED outcome; True exactly when this observation trips.
        Any status outside `_SELF_STATUSES` is new evidence: it wakes parked lanes."""
        lane = str(getattr(outcome, "branch_id", None) or "").partition(":")[2]
        status = getattr(outcome, "status", None)
        with self._changed:
            last = self._last.get(lane)
            if status == "abstained" and last is not None and last.get("pending"):
                receipt = getattr(outcome, "journal_receipt", None)
                last["abstention_id"] = (receipt.get("attempt_id")
                                         if isinstance(receipt, Mapping) else None)
                last["pending"] = False
            if status not in _SELF_STATUSES:
                self.epoch += 1
                self._changed.notify_all()
            if status == PLANNER_SAME_STATE_SKIP and not self._announced:
                # The trip itself happened in `park_wait` (all lanes parked); this
                # recorded skip row is where the run announces it, once.
                self.tripped = self._announced = True
                self._changed.notify_all()
                return True
            return False

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"skip_after": self.skip_after, "tripped": self.tripped,
                    "lane_count": self.lanes, "change_epoch": self.epoch,
                    "last_skip": self.last_skip,
                    "parked": {lane: {"since": since}
                               for lane, (_m, since) in sorted(self._parked.items())},
                    "same_state_waits": {lane: dict(row)
                                         for lane, row in sorted(self._waits.items())},
                    "lanes": {lane: {k: v for k, v in row.items() if k != "pending"}
                              for lane, row in sorted(self._last.items())}}


class GuardedPlanner:
    """Wraps a lane's planner: key before `propose`, record after; all else delegated.

    On a repeated state the lane parks; when another lane's evidence wakes it, the
    context is rebuilt (`refresh_context`, merged into the iteration's own dict) and
    the key recomputed, so the planner is asked only if the state really changed."""

    def __init__(self, inner: Any, guard: SameStateGuard, *, workspace: Path | str,
                 lane: str, should_stop=None, refresh_context=None) -> None:
        self._inner, self._guard = inner, guard
        self._workspace, self._lane = workspace, lane
        self._should_stop, self._refresh = should_stop, refresh_context

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _key(self, context: Mapping[str, Any]) -> str | None:
        backend = getattr(self._inner, "backend", None)
        describe = getattr(backend, "describe", None)
        return state_key(context, worktree=worktree_identity(self._workspace),
                         lane=self._lane,
                         planner=str(describe()) if callable(describe) else "")

    def propose(self, context: Mapping[str, Any]):
        from .loop import Abstain
        key = None
        while self._guard.enabled:
            key = self._key(context)
            try:
                self._guard.check(self._lane, key)
                break
            except PlannerSameStateSkip:
                if not self._guard.park_wait(self._lane, self._should_stop):
                    raise
            if self._refresh is not None and isinstance(context, dict):
                context.update(self._refresh())
        result = self._inner.propose(context)
        self._guard.note_answer(self._lane, key, isinstance(result, Abstain))
        return result


def retry_budget_shadow(store_root: Path | str, attempt: Mapping[str, Any],
                        *, after: int = RETRY_BUDGET_SHADOW_AFTER) -> dict[str, Any] | None:
    """LOG-ONLY: the annotation for a row whose mechanism has >= `after` prior
    `patch_rejected` rows; None below it, without a mechanism, or on any read fault."""
    mechanism = attempt.get("mechanism_id")
    if not isinstance(mechanism, str) or not mechanism:
        return None
    try:
        from ..controller import experiments
        with experiments.ExperimentStore(Path(store_root)) as store:
            prior = sum(1 for row in store.rows_for_mechanism(mechanism)
                        if row.get("status") == "patch_rejected")
    except Exception:          # noqa: BLE001 -- an annotation never fails a record
        return None
    if prior < after:
        return None
    return {"mode": "log_only", "mechanism_id": mechanism,
            "prior_patch_rejected": prior, "threshold": after, "would_refuse": True}


__all__ = ["DEFAULT_SKIP_AFTER", "ENV", "GuardedPlanner", "PLANNER_SAME_STATE_SKIP",
           "PlannerSameStateSkip", "RETRY_BUDGET_SHADOW_AFTER", "SAME_STATE_PARKED",
           "SameStateGuard", "context_projection", "resolve_skip_after",
           "retry_budget_shadow", "state_key", "validate_skip_after",
           "worktree_identity"]
