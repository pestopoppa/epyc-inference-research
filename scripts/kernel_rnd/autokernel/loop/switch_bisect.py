"""Kill-switch bisect planner (AK-TOOL-SWITCH-BISECT), split verbatim from experiment_plan.py (code-budget split)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .experiment_plan_fields import PlanValidationError



# ---------------------------------------------------------------------------
# Kill-switch bisect (AK-TOOL-SWITCH-BISECT): a pure planner / state machine.
# It yields the next arm to run and folds results; it launches nothing.
# Lifted from the C7 Q38FN bisect job (replicate -> all-off -> singles).
# ---------------------------------------------------------------------------
BISECT_NOT_REPRODUCED = "NOT_REPRODUCED"
BISECT_NOT_SWITCHABLE = "NOT_SWITCHABLE"
BISECT_INCONCLUSIVE = "INCONCLUSIVE"
BISECT_MULTI_CAUSE = "MULTI_CAUSE"  # all-off restores, no single does (extension of the grammar)


@dataclass(frozen=True)
class SwitchSpec:
    """One kill switch. Lower `suspicion` runs first; `breadth` breaks ties
    (narrow switch before the broader one that contains it)."""
    name: str
    env: Mapping[str, str]
    witnesses: tuple[str, ...] = ()   # witness keys this port logs when ON
    suspicion: int = 100
    breadth: int = 1


class SwitchBisect:
    """Ordered plan: default (replicate) -> all_off -> single-off arms, stop on first answer.

    `ln` is the regression measure of an arm against the reference (higher = worse);
    an arm "restores" when ln <= tol. `default_witnesses` must be logged by the
    default arm (non-vacuity); an off arm must NOT log the witnesses of the
    switches it turned off. A violated rule voids the unit; `max_void` voids on
    one arm give INCONCLUSIVE.
    """

    DEFAULT, ALL_OFF = "default", "all_off"

    def __init__(self, specs: Iterable[SwitchSpec], *, tol: float, repro_min: float,
                 default_witnesses: Iterable[str] = (), max_void: int = 2) -> None:
        self.specs = sorted(specs, key=lambda s: (s.suspicion, s.breadth, s.name))
        if not self.specs or len({s.name for s in self.specs}) != len(self.specs):
            raise PlanValidationError("bisect needs unique, non-empty switch specs")
        self.tol, self.repro_min, self.max_void = tol, repro_min, max_void
        self.default_witnesses = tuple(default_witnesses)
        self.units: dict[str, float] = {}
        self.voids: dict[str, int] = {}
        self.verdict: str | None = None
        self.detail: dict[str, Any] = {}

    def _arm(self, name: str) -> dict[str, Any]:
        if name == self.DEFAULT:
            return {"arm": name, "env": {}, "must_absent": (), "must_present": self.default_witnesses}
        if name == self.ALL_OFF:
            env: dict[str, str] = {}
            for s in self.specs:
                env.update(s.env)
            return {"arm": name, "env": env, "must_present": (),
                    "must_absent": tuple(sorted({w for s in self.specs for w in s.witnesses}))}
        s = next(x for x in self.specs if x.name == name)
        return {"arm": name, "env": dict(s.env), "must_present": (), "must_absent": tuple(s.witnesses)}

    def order(self) -> list[str]:
        return [self.DEFAULT, self.ALL_OFF] + [s.name for s in self.specs]

    def next_arm(self) -> dict[str, Any] | None:
        if self.verdict is not None:
            return None
        for name in self.order():
            if name not in self.units:
                return self._arm(name)
        return None

    def fold(self, arm: str, ln: float | None, witnesses_seen: Iterable[str] = (),
             valid: bool = True) -> str | None:
        """Fold one result; returns the verdict once the bisect has answered."""
        if self.verdict is not None:
            raise PlanValidationError("bisect already answered: " + self.verdict)
        want = self.next_arm()
        if want is None or want["arm"] != arm:
            raise PlanValidationError(f"expected arm {want and want['arm']!r}, got {arm!r}")
        seen = set(witnesses_seen)
        broken = [w for w in want["must_absent"] if w in seen] + \
                 [f"missing:{w}" for w in want["must_present"] if w not in seen]
        if not valid or ln is None or broken:
            self.voids[arm] = self.voids.get(arm, 0) + 1
            if self.voids[arm] >= self.max_void:
                return self._end(BISECT_INCONCLUSIVE, reason=f"{arm}: {self.max_void} void units",
                                 witness_rule_broken=broken)
            return None
        self.units[arm] = ln
        if arm == self.DEFAULT and ln < self.repro_min:
            return self._end(BISECT_NOT_REPRODUCED, default_ln=ln)
        if arm == self.ALL_OFF and ln > self.tol:
            return self._end(BISECT_NOT_SWITCHABLE, all_off_ln=ln)
        if arm not in (self.DEFAULT, self.ALL_OFF) and ln <= self.tol:
            return self._end(f"CULPRIT({arm})", culprit=arm, ln=ln, not_run=[
                n for n in self.order() if n not in self.units])
        if self.next_arm() is None:
            singles = {s.name: self.units[s.name] for s in self.specs}
            return self._end(BISECT_MULTI_CAUSE, single_ln=singles, best_single=min(singles, key=singles.get))
        return None

    def _end(self, verdict: str, **detail: Any) -> str:
        self.verdict, self.detail = verdict, detail
        return verdict
