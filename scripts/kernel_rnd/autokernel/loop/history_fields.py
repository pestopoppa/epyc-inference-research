"""AK-HR-3: nullable, additive write-side fields for closed-history replay.

AK-HR-1's gap table (`data/ak-history-replay-20261020/census.json`, `g_gap_table`) listed
what the AutoKernel stores never recorded: a key joining an actor call to its attempt, a
campaign-wide trial counter and cumulative token counter, a failure class and per-stage
seconds on the candidate record, and a typed knob descriptor. This module is the one place
that produces those values. NOTHING here selects, orders, gates or budgets any work: it
observes, and every consumer of the values is a reader.

Contract (the RB-lineage telemetry pattern, `85c7b91e`):

  * Fields are NULLABLE and ADDITIVE. A row written before this module existed stays valid;
    a row whose value is unknown carries `null`, never an invented value.
  * The closed, self-hashed `actor_call.v1` schema is NOT widened. The attempt-to-call key
    rides on the sibling `actor_call_metrics.v1` line (`actors._record_metrics`), exactly
    as `actor_metrics.py` already does for its own fields.
  * `mechanism_family_bucket` reuses `actors._mechanism_family` (the existing 4-family
    helper); there is no new taxonomy. It is NOT named `mechanism_family` because an
    abstained row already carries the planner's free-text `mechanism_family` under that key
    (`loop.Outcome.abstained_identity`), and overwriting it would change an existing field.
  * The trial context is a `contextvars.ContextVar`: it follows the lane's thread through
    `loop.iterate`. A best-of author panel runs each member in a copy of that context
    (`bestof.AuthorPanel._race`), so panel calls carry the lane's `trial_id` too.

The attempt-to-call join key is `trial_id`: minted at the start of `loop.iterate`, written
on every actor call's metrics line during that iteration and on the iteration's experiments
payload. `attempt_id` is derived only at record time, so it cannot be known to a call.
"""
from __future__ import annotations

import contextvars
import secrets
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

#: Stages whose seconds a trial records. `planner`, `critic` and `author` are the actor
#: calls' wall time (`actors._record_metrics`); `gate` and `measure` wrap the callables
#: `loop.iterate` is handed. AK-HR-3b (2026-10-24): `build` and `correctness` split the
#: injected gate. `gates.run_all` times every check it runs and files it by the verdict's
#: gate name (`BUILD_VERDICT_GATES` -> `build`, any other verdict -> `correctness`); the
#: GPU gate's build, which runs before its quiet window outside `run_all`, is wrapped in
#: `timed("build", ...)`. So `build + correctness <= gate`: the remainder is the gate's own
#: scope analysis (git diffs, route admission) and window waits outside the checks. A
#: GPU correctness check's own reentrant quiet-window wait is inside `correctness`. A
#: stage that did not run is `null`, never zero (a runtime arm has no `build`).
STAGES = ("planner", "critic", "author", "gate", "build", "correctness", "measure")
#: `gates.compiles` verdict names: a check returning one of these is build time.
BUILD_VERDICT_GATES = frozenset({"configure", "compile"})

#: The payload keys `Outcome.to_attempt` adds (all nullable) when a trial was observed.
ATTEMPT_KEYS = ("trial_id", "failure_class", "stage_seconds", "trial_tokens",
                "mechanism_family_bucket", "knob_descriptor")
#: Added by `archive.record` from the store (campaign-wide, monotone per store+epoch).
COUNTER_KEYS = ("trial_ordinal", "cumulative_tokens")
#: The metrics-line keys `actors._record_metrics` adds (all nullable).
METRICS_KEYS = ("trial_id", "mechanism_family_bucket")


class Trial:
    """One `loop.iterate` turn's observations. Mutated only by the lane's own thread."""

    def __init__(self, trial_id: str | None = None) -> None:
        self.trial_id = trial_id or secrets.token_hex(16)
        self._lock = threading.Lock()   # best-of panel members share one Trial
        self.stage_seconds: dict[str, float | None] = {stage: None for stage in STAGES}
        self.call_failure_classes: list[str] = []
        self.decoded_tokens: int | None = None
        self.prompt_tokens: int | None = None
        self.mechanism_family_bucket: str | None = None

    def add_seconds(self, stage: str, seconds: float) -> None:
        if stage in self.stage_seconds and isinstance(seconds, (int, float)):
            with self._lock:
                self.stage_seconds[stage] = round((self.stage_seconds[stage] or 0.0)
                                                  + max(0.0, float(seconds)), 3)

    def add_tokens(self, decoded: Any, prompt: Any) -> None:
        with self._lock:
            for name, value in (("decoded_tokens", decoded), ("prompt_tokens", prompt)):
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    setattr(self, name, (getattr(self, name) or 0) + value)

    def snapshot(self) -> dict[str, Any]:
        """The nullable payload fields. `trial_tokens` is decoded+prompt, null when no call
        of the trial reported tokens (hosted codex/claude lines report none)."""
        tokens = None
        if self.decoded_tokens is not None or self.prompt_tokens is not None:
            tokens = (self.decoded_tokens or 0) + (self.prompt_tokens or 0)
        return {
            "trial_id": self.trial_id,
            # The last call-level failure class seen in the trial; null when no call failed.
            "failure_class": self.call_failure_classes[-1] if self.call_failure_classes else None,
            "stage_seconds": dict(self.stage_seconds),
            "trial_tokens": tokens,
            "mechanism_family_bucket": self.mechanism_family_bucket,
        }


_CURRENT: contextvars.ContextVar[Trial | None] = contextvars.ContextVar(
    "ak_history_trial", default=None)


def current() -> Trial | None:
    return _CURRENT.get()


def begin(trial_id: str | None = None) -> contextvars.Token:
    return _CURRENT.set(Trial(trial_id))


def end(token: contextvars.Token) -> None:
    try:
        _CURRENT.reset(token)
    except ValueError:      # reset from a different context: leave it, observation only
        _CURRENT.set(None)


def family_bucket(row: Mapping[str, Any]) -> str | None:
    """The existing 4-family helper's value for a hypothesis/attempt row, else null."""
    from .actors import _mechanism_family
    try:
        return _mechanism_family(row)
    except Exception:       # noqa: BLE001 -- evidence, never a reason to fail the loop
        return None


def note_hypothesis(hypothesis: Any) -> None:
    """Record the proposed hypothesis' family bucket on the current trial (no-op without one)."""
    trial = current()
    if trial is None or hypothesis is None:
        return
    try:
        row = hypothesis.to_dict()
    except Exception:       # noqa: BLE001
        return
    trial.mechanism_family_bucket = family_bucket(row)


def note_call(role: str | None, wall_s: Any, failure_class: str | None,
              opencode_stats: Mapping[str, Any] | None) -> None:
    """Called by `actors._record_metrics` for every actor call. No-op without a trial."""
    trial = current()
    if trial is None:
        return
    if role in STAGES:
        trial.add_seconds(role, wall_s)
    if failure_class:
        trial.call_failure_classes.append(str(failure_class))
    totals = (opencode_stats or {}).get("totals") or {}
    if totals:
        # The same decoded+prompt axis AK-HR-1(c) counted (`actor_metrics.TOTAL_FIELDS`).
        trial.add_tokens(totals.get("decoded_tokens"), totals.get("prompt_tokens"))


def note_check(verdict_gate: Any, seconds: Any) -> None:
    """Called by `gates.run_all` after each check: its seconds, filed as `build` when the
    verdict is a build verdict, else `correctness`. No-op without a trial."""
    trial = current()
    if trial is None:
        return
    stage = "build" if verdict_gate in BUILD_VERDICT_GATES else "correctness"
    trial.add_seconds(stage, seconds)


def metrics_fields() -> dict[str, Any]:
    """The nullable keys for a metrics line; both null outside a trial."""
    trial = current()
    return {"trial_id": trial.trial_id if trial else None,
            "mechanism_family_bucket": trial.mechanism_family_bucket if trial else None}


def timed(stage: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an injected stage callable: same arguments, result and exceptions, plus the
    elapsed seconds on the current trial. A wrapper must never change behavior."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        trial = current()
        started = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            if trial is not None:
                trial.add_seconds(stage, time.monotonic() - started)
    return wrapper


@contextmanager
def trial_scope(trial_id: str | None = None) -> Iterator[Trial]:
    token = begin(trial_id)
    try:
        yield current()     # type: ignore[misc]
    finally:
        end(token)


def knob_descriptor(hypothesis: Any) -> dict[str, Any] | None:
    """A typed knob descriptor (type tag plus parameters), known BEFORE build.

    Runtime arms: the registered `RuntimeDimension` the pair varies (kind, id, anchor and
    candidate values). Source candidates: null. The planner's reply schema declares no
    parameters for a source patch, and adding one would change what the planner is asked,
    which this task forbids; the gap stays listed for a decision that may change it."""
    pair = getattr(hypothesis, "runtime_pair", None)
    dimension = getattr(pair, "dimension", None)
    if dimension is None:
        return None
    try:
        return {"type": "runtime_dimension", "kind": dimension.kind,
                "dimension_id": dimension.dimension_id,
                "anchor": _plain(dimension.anchor), "candidate": _plain(dimension.candidate)}
    except Exception:       # noqa: BLE001
        return None


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
