"""Driver execution receipt record, split verbatim from driver_execution.py (code-budget split)."""
from __future__ import annotations

from .field_validation import require_sha256 as _require_sha256
import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping
from . import restart_native_citations as citations
from . import campaign_control
from . import scheduling
from . import unified_driver
from . import unified_worker


EXECUTION_RECEIPT_SCHEMA = "epyc.autokernel.unified_driver_execution_receipt.v1"


EXECUTION_RECEIPT_SCHEMA_V2 = "epyc.autokernel.unified_driver_execution_receipt.v2"


class DriverExecutionRefused(RuntimeError):
    """The selected work cannot safely cross the execution boundary."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DriverExecutionRefused(f"{label} must be nonempty text")
    return value


def _sha(value: Any, label: str) -> str:
    return _require_sha256(_text(value, label), label, DriverExecutionRefused)


@dataclass(frozen=True)
class DriverExecutionReceipt:
    catalog_id: str
    transition_id: str
    prepared_digest: str
    request_id: str
    lineage_id: str
    stage_id: str
    terminal: Mapping[str, Any]
    result_reference: Mapping[str, Any] | None
    native_measurement_ids: tuple[str, ...]
    settlement_request: Mapping[str, Any]
    settlement_receipt: Mapping[str, Any]
    schema: str = EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema not in {EXECUTION_RECEIPT_SCHEMA, EXECUTION_RECEIPT_SCHEMA_V2}:
            raise DriverExecutionRefused("execution receipt schema is unsupported")
        calibration = self.schema == EXECUTION_RECEIPT_SCHEMA_V2
        for name in ("catalog_id", "transition_id", "prepared_digest"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("request_id", "lineage_id", "stage_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if len(set(self.native_measurement_ids)) != len(self.native_measurement_ids):
            raise DriverExecutionRefused("execution receipt native identities are invalid")
        for value in self.native_measurement_ids:
            _text(value, "native measurement id")
        for name in ("terminal", "settlement_request", "settlement_receipt"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise DriverExecutionRefused(f"{name} must be an object")
            object.__setattr__(self, name, unified_driver._freeze(copy.deepcopy(dict(value))))
        terminal_fields = {"worker_id", "worker_generation", "request_id", "plan_digest",
            "lineage_id", "stage_id", "grant_id", "grant_generation", "container_id",
            "return_code", "result_digest", "accepted", "reason"}
        if set(self.terminal) != terminal_fields \
                or type(self.terminal["accepted"]) is not bool \
                or self.terminal["request_id"] != self.request_id \
                or self.terminal["lineage_id"] != self.lineage_id \
                or self.terminal["stage_id"] != self.stage_id:
            raise DriverExecutionRefused("execution receipt terminal binding differs")
        reference = None
        if self.result_reference is not None:
            if not isinstance(self.result_reference, Mapping):
                raise DriverExecutionRefused("result_reference must be an object or null")
            reference = unified_worker.PlannedWorkerResultReference.from_dict(
                unified_driver._thaw(self.result_reference))
            object.__setattr__(self, "result_reference", unified_driver._freeze(
                reference.to_dict()))
        for name in ("worker_generation", "grant_generation"):
            if (not isinstance(self.terminal[name], int)
                    or isinstance(self.terminal[name], bool) or self.terminal[name] < 1):
                raise DriverExecutionRefused(f"terminal {name} is invalid")
        _sha(self.terminal["plan_digest"], "terminal plan digest")
        selection_request = unified_driver._thaw(self.settlement_request)
        expected_request = {"schema", "catalog_id", "transition_id", "selection",
                            "receipt", "outcome", "terminal_refs"}
        if (set(selection_request) != expected_request
                or selection_request["schema"] != campaign_control.DRIVER_SETTLEMENT_SCHEMA
                or selection_request["catalog_id"] != self.catalog_id
                or selection_request["transition_id"] != self.transition_id
                or selection_request["outcome"] not in (
                    {"invalid", "failed", "calibration"} if calibration else {"invalid", "failed"})):
            raise DriverExecutionRefused("execution receipt settlement request differs")
        selection = scheduling.Selection.from_dict(selection_request["selection"])
        held = scheduling.HeldClaimReceipt.from_dict(selection_request["receipt"])
        if (selection.proposal is None or held.proposal_id != selection.proposal.proposal_id
                or held.proposal_id != self.request_id
                or held.ownership_generation != self.terminal["worker_generation"]
                or held.allocation_generation != self.terminal["grant_generation"]):
            raise DriverExecutionRefused("execution receipt held/selection binding differs")
        if calibration and (selection.proposal.stage_class != "calibration"
                            or held.stage_class != "calibration"):
            raise DriverExecutionRefused("v2 receipt requires the selected calibration stage")
        if self.terminal["accepted"]:
            if (reference is None
                    or reference.worker_id != self.terminal["worker_id"]
                    or reference.worker_generation != self.terminal["worker_generation"]
                    or reference.prepared_digest != self.prepared_digest
                    or self.terminal["return_code"] != 0
                    or self.terminal["reason"] is not None):
                raise DriverExecutionRefused("execution result reference/terminal differs")
            result_envelope_digest = unified_worker._digest(reference.to_dict())
            if self.terminal["result_digest"] != result_envelope_digest:
                raise DriverExecutionRefused("execution receipt terminal references differ")
            if selection_request["outcome"] in {"invalid", "calibration"}:
                expected_refs = [f"native:{item}" for item in self.native_measurement_ids]
                if calibration:
                    from . import serving_preparation as preparation
                    raw_refs = selection_request["terminal_refs"]
                    collection_refs = [item for item in raw_refs
                                       if isinstance(item, str) and item.startswith("calibration-collected:")]
                    if len(collection_refs) != 1 or reference.schema != unified_worker.RESULT_REFERENCE_SCHEMA_V2:
                        raise DriverExecutionRefused("v2 receipt needs exactly one native calibration collection")
                    try:
                        collected = preparation.CollectedCalibrationReference.from_dict(json.loads(
                            collection_refs[0].removeprefix("calibration-collected:")))
                    except (ValueError, TypeError) as exc:
                        raise DriverExecutionRefused("calibration collection reference is malformed") from exc
                    if collected.declaration_digest != selection.proposal.eligibility_ref:
                        raise DriverExecutionRefused("calibration collection declaration differs from selection")
                    expected_refs.append("calibration-collected:" + unified_driver._canonical(
                        collected.to_dict()).decode())
                expected_refs.append(f"result:{result_envelope_digest}")
                if (not self.native_measurement_ids
                        or selection_request["terminal_refs"] != expected_refs):
                    raise DriverExecutionRefused(
                        "execution receipt terminal references differ")
            else:
                terminal_digest = unified_driver._digest(
                    unified_driver._thaw(self.terminal))
                refs = selection_request["terminal_refs"]  # AKU-07h3: + retained citations
                if (self.native_measurement_ids or refs[:1] != [f"lifecycle:{terminal_digest}"]
                        or not all(map(citations.is_retained_ref, refs[1:]))):
                    raise DriverExecutionRefused(
                        "operational failure receipt binding differs")
        else:
            terminal_digest = unified_driver._digest(unified_driver._thaw(self.terminal))
            if (reference is not None or self.native_measurement_ids
                    or selection_request["outcome"] != "failed"
                    or selection_request["terminal_refs"]
                       != [f"lifecycle:{terminal_digest}"]):
                raise DriverExecutionRefused("failed execution receipt binding differs")
        settlement = self.settlement_receipt
        if (set(settlement) != {"schema", "transition_id", "status",
                               "accounting_projection_digest"}
                or settlement["schema"] != campaign_control.DRIVER_SETTLEMENT_RECEIPT_SCHEMA
                or settlement["transition_id"] != self.transition_id
                or settlement["status"] not in {"accepted", "duplicate"}):
            raise DriverExecutionRefused("execution settlement receipt differs")
        _sha(settlement["accounting_projection_digest"], "accounting projection digest")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": self.schema, "catalog_id": self.catalog_id,
            "transition_id": self.transition_id, "prepared_digest": self.prepared_digest,
            "request_id": self.request_id, "lineage_id": self.lineage_id,
            "stage_id": self.stage_id, "terminal": unified_driver._thaw(self.terminal),
            "result_reference": (None if self.result_reference is None
                                 else unified_driver._thaw(self.result_reference)),
            "native_measurement_ids": list(self.native_measurement_ids),
            "settlement_request": unified_driver._thaw(self.settlement_request),
            "settlement_receipt": unified_driver._thaw(self.settlement_receipt),
        }
        return {**body, "receipt_digest": unified_driver._digest(body)}

    @classmethod
    def from_dict(cls, value: Any) -> "DriverExecutionReceipt":
        fields = {"schema", "catalog_id", "transition_id", "prepared_digest",
                  "request_id", "lineage_id", "stage_id", "terminal",
                  "result_reference", "native_measurement_ids", "settlement_request",
                  "settlement_receipt", "receipt_digest"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise DriverExecutionRefused("execution receipt fields differ")
        body = {key: copy.deepcopy(value[key]) for key in fields - {"receipt_digest"}}
        supplied = _sha(value["receipt_digest"], "execution receipt digest")
        if supplied != unified_driver._digest(body):
            raise DriverExecutionRefused("execution receipt digest differs")
        ids = body.pop("native_measurement_ids")
        if not isinstance(ids, (list, tuple)):
            raise DriverExecutionRefused("native measurement ids must be an array")
        return cls(native_measurement_ids=tuple(ids), **body)
