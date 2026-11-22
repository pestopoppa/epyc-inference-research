"""Profile preparation execution receipt, split verbatim from profile_preparation.py (code-budget split)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from . import actor_preparation_state as state
from . import campaign_control as cc
from . import scheduling
from . import unified_driver as ud


RECEIPT_SCHEMA = "epyc.autokernel.profile_preparation_execution_receipt.v1"


class ProfilePreparationRefused(ValueError):
    pass


def _plain(value):
    return ud._thaw(value)


@dataclass(frozen=True)
class ProfilePreparationExecutionReceipt:
    """Integrity-checked reporting value; parsing never restores original issuance."""
    body: Mapping[str, Any]

    def __post_init__(self):
        row = _plain(self.body)
        fields = {"schema", "catalog_id", "transition_id", "selected_work", "selected_work_digest",
                  "request_id", "stage_id", "terminal", "provider_held_receipt",
                  "profile_reference", "disposition", "settlement_request", "settlement_receipt"}
        if (not isinstance(row, Mapping) or set(row) != fields
                or row["schema"] != RECEIPT_SCHEMA
                or row["disposition"] not in {"profile_verified", "failed"}):
            raise ProfilePreparationRefused("profile execution receipt fields/schema differ")
        state._canonical(row)
        work = ud.SelectedProfileWork.from_dict(row["selected_work"])
        held = scheduling.HeldClaimReceipt.from_dict(row["provider_held_receipt"])
        terminal = row["terminal"]
        terminal_fields = {"worker_id", "worker_generation", "request_id", "plan_digest",
            "lineage_id", "stage_id", "grant_id", "grant_generation", "container_id",
            "return_code", "result_digest", "accepted", "reason"}
        if (not isinstance(terminal, Mapping) or set(terminal) != terminal_fields
                or type(terminal["accepted"]) is not bool):
            raise ProfilePreparationRefused("profile receipt terminal fields differ")
        request_digest = state.digest(work.profile_request.to_dict())
        if (row["catalog_id"] != work.catalog_id or row["transition_id"] != work.transition_id
                or row["selected_work_digest"] != state.digest(work.to_dict())
                or row["request_id"] != "profile-" + request_digest[:24]
                or row["stage_id"] != "target-profile-" + request_digest[:24]
                or terminal["request_id"] != row["request_id"]
                or terminal["stage_id"] != row["stage_id"]
                or terminal["plan_digest"] != work.stage_plan_digest
                or terminal["lineage_id"] != work.transition_id
                or held.ownership_generation != terminal["worker_generation"]
                or held.allocation_generation != terminal["grant_generation"]
                or held.proposal_id != work.selection.proposal.proposal_id
                or held.backend != work.selection.proposal.backend
                or held.stage_class != work.selection.proposal.stage_class):
            raise ProfilePreparationRefused("profile receipt selected/terminal/held identity differs")
        settlement = row["settlement_request"]
        settlement_fields = {"schema", "catalog_id", "transition_id", "selection", "receipt",
                             "outcome", "terminal_refs"}
        if (not isinstance(settlement, Mapping) or set(settlement) != settlement_fields
                or settlement["schema"] != cc.DRIVER_SETTLEMENT_SCHEMA
                or settlement["catalog_id"] != work.catalog_id
                or settlement["transition_id"] != work.transition_id
                or settlement["selection"] != work.selection.to_dict()
                or settlement["receipt"] != held.to_dict()):
            raise ProfilePreparationRefused("profile receipt settlement identity differs")
        reference = row["profile_reference"]
        if row["disposition"] == "profile_verified":
            ref_fields = {"profile_event_digest", "target_profile_digest", "profile_request_digest",
                          "catalog_id", "transition_id", "verifier_ref", "target_revision_digest"}
            if (not isinstance(reference, Mapping) or set(reference) != ref_fields
                    or terminal["accepted"] is not True or terminal["return_code"] != 0
                    or terminal["reason"] is not None
                    or reference["catalog_id"] != work.catalog_id
                    or reference["transition_id"] != work.transition_id
                    or reference["profile_request_digest"] != request_digest
                    or reference["target_revision_digest"] != work.profile_request.target_revision_digest
                    or reference["verifier_ref"] !=
                       f"controller-worker:{terminal['worker_id']}:{terminal['worker_generation']}"
                    or settlement["outcome"] != "prerequisite"
                    or settlement["terminal_refs"] != [reference["verifier_ref"]]):
                raise ProfilePreparationRefused("profile receipt original profile reference differs")
            for name in ("profile_event_digest", "target_profile_digest"):
                state._sha(reference[name], name)
            state._sha(terminal["result_digest"], "profile terminal result")
        elif (reference is not None or settlement["outcome"] != "failed"
                or settlement["terminal_refs"] != ["lifecycle:" + state.digest(terminal)]):
            raise ProfilePreparationRefused("failed profile receipt cannot carry a profile")
        accepted = row["settlement_receipt"]
        if (not isinstance(accepted, Mapping)
                or set(accepted) != {"schema", "transition_id", "status", "accounting_projection_digest"}
                or accepted["schema"] != cc.DRIVER_SETTLEMENT_RECEIPT_SCHEMA
                or accepted["transition_id"] != work.transition_id
                or accepted["status"] not in {"accepted", "duplicate"}):
            raise ProfilePreparationRefused("profile receipt durable settlement differs")
        state._sha(accepted["accounting_projection_digest"], "accounting projection")
        object.__setattr__(self, "body", ud._freeze(row))

    def to_dict(self):
        row = _plain(self.body)
        return row | {"receipt_digest": state.digest(row)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or "receipt_digest" not in value:
            raise ProfilePreparationRefused("profile reporting receipt lacks integrity digest")
        row = _plain(value)
        digest = row.pop("receipt_digest")
        if digest != state.digest(row):
            raise ProfilePreparationRefused("profile reporting receipt digest differs")
        return cls(row)
