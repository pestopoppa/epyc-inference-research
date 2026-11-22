"""Standalone work-kind dispatch through each kind's owner (AKU-07h4).

The standalone runtime asks the driver for the closed kind of the exact issued
intent and hands it to the one installed owner of that kind, both on first
execution and on process-restart recovery of a restored pending intent. A kind
with no installed owner is refused, never executed by another kind's owner.
Continuous evidence is not a selection: the installed feed owner drains on the
runtime thread inside every driver tick, before planning (feed_runtime).
"""
from __future__ import annotations

from typing import Any, Callable

from . import unified_driver

ACTOR_KIND = "actor_preparation"


class ActorDispatchRefused(RuntimeError):
    pass


class InstalledActorDispatch:
    """Owner adapter for selected actor work.

    ``run`` and ``recover`` receive the exact ``SelectedActorWork`` re-materialized
    from the current issued intent through the controller binding; both return a
    receipt with ``to_dict()`` (for example a source-build execution owner's
    settlement). Recovery never re-runs the actor itself: it is the owner's restart
    settlement of the same transition.
    """

    def __init__(self, *, run: Callable[[Any], Any], recover: Callable[[Any], Any]) -> None:
        if not callable(run) or not callable(recover):
            raise ActorDispatchRefused("actor dispatch needs run and recover owners")
        self._run, self._recover = run, recover

    def _selected(self, driver, outcome):
        try:
            selected = driver.materialize_actor(outcome)
        except unified_driver.DriverRefused as exc:
            raise ActorDispatchRefused(f"selected actor work is not current: {exc}") from exc
        if not isinstance(selected, unified_driver.SelectedActorWork):
            raise ActorDispatchRefused("actor owner needs typed selected actor work")
        if selected.transition_id != outcome.transition_id:
            raise ActorDispatchRefused("selected actor work belongs to another transition")
        return selected

    def execute(self, driver, outcome):
        return self._run(self._selected(driver, outcome))

    def recover_issued(self, driver, outcome):
        return self._recover(self._selected(driver, outcome))


def installed_kinds(runtime) -> set[str]:
    kinds = {"runtime_comparison"}
    if runtime.driver.preparation_owner is not None:
        kinds.add("calibration_preparation")
    if runtime.profile_executor is not None:
        kinds.add("profile_preparation")
    if getattr(runtime, "actor_executor", None) is not None:
        kinds.add(ACTOR_KIND)
    return kinds


def dispatch(runtime, kind: str, outcome, *, recover: bool = False):
    """Run (or restart-settle) the issued intent through the owner of its kind."""
    owner = {"profile_preparation": runtime.profile_executor,
             ACTOR_KIND: getattr(runtime, "actor_executor", None)}.get(kind)
    if kind in ("profile_preparation", ACTOR_KIND):
        if owner is None:
            raise ActorDispatchRefused(f"{kind} owner is not installed")
        return (owner.recover_issued if recover else owner.execute)(runtime.driver, outcome)
    return (runtime.executor.recover_issued(outcome) if recover
            else runtime.executor.execute(outcome))
