"""Immutable planned-unit transport. These records confer no launch authority."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import types
from typing import Any, Mapping

from . import experiment_plan as ep

SELECTION_SCHEMA = "epyc.autokernel.selected_plan_unit_range.v1"
DISPATCH_SCHEMA = "epyc.autokernel.selected_unit_dispatch.v1"
MAX_RANGE_UNITS = 1024


class SelectionRefused(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise SelectionRefused("selection identity must be bounded nonempty text")
    return value


def _sha(value: Any) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise SelectionRefused("selection digest must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class SelectedPlanUnitRange:
    plan_digest: str
    start_order: int
    stop_order: int
    unit_ids: tuple[str, ...]
    unit_specs_digest: str

    def __post_init__(self) -> None:
        _sha(self.plan_digest)
        _sha(self.unit_specs_digest)
        if (type(self.start_order) is not int or type(self.stop_order) is not int
                or self.start_order < 0 or self.stop_order <= self.start_order
                or self.stop_order - self.start_order > MAX_RANGE_UNITS
                or not isinstance(self.unit_ids, (tuple, list))
                or len(self.unit_ids) != self.stop_order - self.start_order):
            raise SelectionRefused("selection must be a bounded nonempty absolute range")
        ids = tuple(_text(item) for item in self.unit_ids)
        if len(set(ids)) != len(ids):
            raise SelectionRefused("selected units must be unique")
        object.__setattr__(self, "unit_ids", ids)

    def body(self) -> dict[str, Any]:
        return {"schema": SELECTION_SCHEMA, "plan_digest": self.plan_digest,
                "start_order": self.start_order, "stop_order": self.stop_order,
                "unit_ids": list(self.unit_ids), "unit_specs_digest": self.unit_specs_digest}

    @property
    def digest(self) -> str:
        return _digest(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "selection_digest": self.digest}

    @classmethod
    def from_dict(cls, value: Any) -> "SelectedPlanUnitRange":
        fields = {"schema", "plan_digest", "start_order", "stop_order", "unit_ids",
                  "unit_specs_digest", "selection_digest"}
        if (not isinstance(value, Mapping) or set(value) != fields
                or value["schema"] != SELECTION_SCHEMA):
            raise SelectionRefused("selection fields/schema differ")
        result = cls(value["plan_digest"], value["start_order"], value["stop_order"],
                     value["unit_ids"], value["unit_specs_digest"])
        if _sha(value["selection_digest"]) != result.digest:
            raise SelectionRefused("selection digest differs")
        return result

    @classmethod
    def from_plan(cls, plan: ep.ExperimentPlan, start_order: int,
                  stop_order: int) -> "SelectedPlanUnitRange":
        if not isinstance(plan, ep.ExperimentPlan):
            raise SelectionRefused("selection requires the original typed plan")
        plan = ep.ExperimentPlan.from_dict(plan.to_dict())
        ordered = tuple(sorted(plan.expected_units, key=lambda unit: unit.order_index))
        if (type(start_order) is not int or type(stop_order) is not int
                or not 0 <= start_order < stop_order <= len(ordered)
                or stop_order - start_order > MAX_RANGE_UNITS):
            raise SelectionRefused("selection is outside the original plan")
        units = ordered[start_order:stop_order]
        if tuple(unit.order_index for unit in units) != tuple(range(start_order, stop_order)):
            raise SelectionRefused("selection does not preserve absolute plan order")
        return cls(plan.digest, start_order, stop_order,
                   tuple(unit.unit_id for unit in units),
                   _digest([unit.to_dict() for unit in units]))

    def units(self, plan: ep.ExperimentPlan) -> tuple[ep.UnitSpec, ...]:
        if not isinstance(plan, ep.ExperimentPlan):
            raise SelectionRefused("selection requires the original typed plan")
        plan = ep.ExperimentPlan.from_dict(plan.to_dict())
        original = type(self).from_plan(plan, self.start_order, self.stop_order)
        if self != original:
            raise SelectionRefused("selection differs from complete original plan membership")
        return tuple(sorted(plan.expected_units, key=lambda unit: unit.order_index))[
            self.start_order:self.stop_order]


@dataclass(frozen=True)
class SelectedUnitDispatch:
    """One transport attempt, not a scheduler selection, grant, or A2 permit."""

    plan_digest: str
    target_revision: str
    selection: SelectedPlanUnitRange
    attempt_id: str

    def __post_init__(self) -> None:
        _sha(self.plan_digest)
        _text(self.target_revision)
        _text(self.attempt_id)
        if (type(self.selection) is not SelectedPlanUnitRange
                or self.selection.plan_digest != self.plan_digest
                or len(self.selection.unit_ids) != 1):
            raise SelectionRefused("selected transport requires exactly one original unit")

    def body(self) -> dict[str, Any]:
        return {"schema": DISPATCH_SCHEMA, "plan_digest": self.plan_digest,
                "target_revision": self.target_revision,
                "selection": self.selection.to_dict(), "attempt_id": self.attempt_id,
                "execution_authorized": False}

    @property
    def digest(self) -> str:
        return _digest(self.body())

    @property
    def request_id(self) -> str:
        return self.digest

    @property
    def lineage_id(self) -> str:
        return f"selected-unit:{self.digest}"

    @property
    def stage_id(self) -> str:
        return f"selected-unit:{self.selection.digest}"

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "dispatch_digest": self.digest}

    @classmethod
    def from_dict(cls, value: Any) -> "SelectedUnitDispatch":
        if (not isinstance(value, Mapping) or set(value) != {"schema", "plan_digest",
                "target_revision", "selection", "attempt_id", "execution_authorized",
                "dispatch_digest"} or value["schema"] != DISPATCH_SCHEMA
                or value["execution_authorized"] is not False):
            raise SelectionRefused("selected dispatch fields/schema/authority differ")
        result = cls(value["plan_digest"], value["target_revision"],
                     SelectedPlanUnitRange.from_dict(value["selection"]), value["attempt_id"])
        if _sha(value["dispatch_digest"]) != result.digest:
            raise SelectionRefused("selected dispatch digest differs")
        return result

    def validate_plan(self, plan: ep.ExperimentPlan) -> None:
        self.selection.units(plan)
        if self.target_revision != plan.target_revision:
            raise SelectionRefused("selected dispatch target differs from original plan")


def loaded_source_identity() -> dict[str, Any]:
    """Retain actual code and explicitly pin the finite callable-default scope.

    Generic callable_identity correctly leaves callable defaults unproven. This
    closure retains that honest record and supplies their separately verified
    loaded identities; it never strips defaults from a synthetic function.
    """
    import time
    from . import lifecycle_observation as lo, observation_binding as ob
    from . import planned_serving as ps, unified_worker as worker

    clock_identity, clock_provenance = ob._loaded_builtin_identity(
        time.monotonic, clock_name="monotonic")

    def default_identity(value: Any) -> Any:
        if not callable(value):
            status, stable = lo._stable_json_value(value)
            if status != "pinned":
                raise SelectionRefused("transport default configuration is unproven")
            return {"value": stable}
        # A Python default retains the actual original function object even when
        # its module global is later rebound. Pin that object, not today's name.
        if value is not time.monotonic and type(value) is not types.FunctionType:
            raise SelectionRefused("transport default must be an actual Python function or pinned clock")
        identity = clock_identity if value is time.monotonic else lo.callable_identity(value)
        if (identity["implementation_status"] != "pinned"
                or identity["configuration_status"] != "pinned"):
            raise SelectionRefused("transport default callable source is incomplete")
        return {"callable": identity}

    functions = (worker.OwnedWorkerStageProvider.__init__,
                 worker.run_prepared_stage, ps.run_planned_comparison)
    pure_functions = (_digest, _text, _sha, SelectedPlanUnitRange.__post_init__,
        SelectedPlanUnitRange.body, SelectedPlanUnitRange.digest.fget,
        SelectedPlanUnitRange.to_dict, SelectedUnitDispatch.__post_init__,
        SelectedUnitDispatch.body, SelectedUnitDispatch.digest.fget,
        SelectedUnitDispatch.to_dict, SelectedUnitDispatch.request_id.fget,
        SelectedUnitDispatch.lineage_id.fget, SelectedUnitDispatch.stage_id.fget,
        worker.PreparedPlannedServingStage.body,
        worker.PreparedPlannedServingStage.selected_dispatch.fget,
        worker.PreparedPlannedServingStage.selected_range.fget,
        worker.OwnedWorkerStageProvider.complete,
        worker.PlannedWorkerInvocation._finish_completion)
    if any(type(function) is not types.FunctionType for function in pure_functions):
        raise SelectionRefused("transport supporting callable must be an actual Python function")
    pure_identities = [lo.callable_identity(function) for function in pure_functions]
    if any(row["implementation_status"] != "pinned" or row["configuration_status"] != "pinned"
           for row in pure_identities):
        raise SelectionRefused("transport supporting source/configuration is incomplete")
    records = []
    for function in functions:
        if type(function) is not types.FunctionType:
            raise SelectionRefused("transport callable must be an actual Python function")
        identity = lo.callable_identity(function)
        if (identity["implementation_status"] != "pinned"
                or getattr(function, "__self__", None) is not None
                or function.__closure__):
            raise SelectionRefused("transport implementation/bound configuration is incomplete")
        configuration = {
            "defaults": [default_identity(value) for value in (function.__defaults__ or ())],
            "kwdefaults": {key: default_identity(value)
                           for key, value in (function.__kwdefaults__ or {}).items()}}
        records.append({"identity": identity, "explicit_configuration": configuration,
                        "explicit_configuration_digest": _digest(configuration)})
    return {"schema": "epyc.autokernel.selected_unit_transport_source.v1",
            "default_clock_provenance": clock_provenance, "callables": records,
            "supporting_callables": pure_identities,
            "schemas": {"selection": SELECTION_SCHEMA, "dispatch": DISPATCH_SCHEMA,
                "prepared": worker.PREPARED_SCHEMA_V4, "result": worker.RESULT_SCHEMA_V3,
                "result_reference": worker.RESULT_REFERENCE_SCHEMA_V3,
                "artifact": ps.ARTIFACT_SCHEMA_V3, "run": ps.RUN_SCHEMA_V3},
            "max_range_units": MAX_RANGE_UNITS}
