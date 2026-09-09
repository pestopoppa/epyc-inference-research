"""Closed inputs and live-claim adapter for pre-measurement model verification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..evaluator import c3_epyc_tensor_capture as tensor_capture
from . import observation_binding as ob
from . import worker_lifecycle as wl

SPEC_SCHEMA = "epyc.autokernel.scheduled_model_preparation.v1"
BINDING_SCHEMA = "epyc.autokernel.scheduled_model_preparation_binding.v1"


class ModelPreparationRefused(ValueError):
    pass


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ModelPreparationRefused(f"{label} must be non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ModelPreparationRefused(f"{label} must be lowercase SHA-256")
    return value


def _absolute(value: Any, label: str) -> str:
    path = Path(_text(value, label))
    if not path.is_absolute():
        raise ModelPreparationRefused(f"{label} must be absolute")
    return str(path)


@dataclass(frozen=True)
class ScheduledModelPreparation:
    target_revision_digest: str
    recipe_execution_digest: str
    entry_path: str
    entry_sha256: str
    inventory_identity: Mapping[str, Any]
    preparation_digest: str
    schema: str = SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SPEC_SCHEMA:
            raise ModelPreparationRefused("unsupported model preparation schema")
        object.__setattr__(self, "target_revision_digest",
                           _sha(self.target_revision_digest, "target_revision_digest"))
        object.__setattr__(self, "recipe_execution_digest",
                           _sha(self.recipe_execution_digest, "recipe_execution_digest"))
        object.__setattr__(self, "entry_path", _absolute(self.entry_path, "entry_path"))
        object.__setattr__(self, "entry_sha256", _sha(self.entry_sha256, "entry_sha256"))
        raw = self.inventory_identity
        fields = {"model_id", "model_manifest", "model_manifest_sha256", "model_sha256"}
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ModelPreparationRefused("inventory_identity has missing or unknown fields")
        identity = {
            "model_id": _absolute(raw["model_id"], "inventory model_id"),
            "model_manifest": _absolute(raw["model_manifest"], "inventory model_manifest"),
            "model_manifest_sha256": _sha(
                raw["model_manifest_sha256"], "inventory model_manifest_sha256"),
            "model_sha256": _sha(raw["model_sha256"], "inventory model_sha256"),
        }
        object.__setattr__(self, "inventory_identity", MappingProxyType(identity))
        supplied = _sha(self.preparation_digest, "preparation_digest")
        expected = wl._digest(self.body())
        if supplied != expected:
            raise ModelPreparationRefused("model preparation digest mismatch")
        object.__setattr__(self, "preparation_digest", supplied)

    @classmethod
    def from_dict(cls, value: Any) -> "ScheduledModelPreparation":
        fields = {"schema", "target_revision_digest", "recipe_execution_digest",
                  "entry_path", "entry_sha256", "inventory_identity",
                  "preparation_digest"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ModelPreparationRefused("model preparation has missing or unknown fields")
        return cls(value["target_revision_digest"], value["recipe_execution_digest"],
                   value["entry_path"], value["entry_sha256"],
                   value["inventory_identity"], value["preparation_digest"],
                   value["schema"])

    def identity(self) -> tensor_capture.CaptureModelIdentity:
        row = self.inventory_identity
        return tensor_capture.CaptureModelIdentity(
            row["model_id"], Path(row["model_manifest"]),
            row["model_manifest_sha256"], row["model_sha256"])

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "target_revision_digest": self.target_revision_digest,
                "recipe_execution_digest": self.recipe_execution_digest,
                "entry_path": self.entry_path, "entry_sha256": self.entry_sha256,
                "inventory_identity": dict(self.inventory_identity)}

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "preparation_digest": self.preparation_digest}


class ActiveObservationPreparationClaim:
    """Live capability facade; every held check re-enters the lifecycle owner."""

    def __init__(self, *, lifecycle: Any, start: Any, unit: Any, fence: Any,
                 initial_claim: Mapping[str, Any]) -> None:
        self._lifecycle, self._start, self._unit, self._fence = lifecycle, start, unit, fence
        self._initial = ob._freeze(ob._plain(initial_claim))
        self.claim_id = _text(initial_claim.get("active_claim_ref"), "active claim reference")

    def is_held(self) -> bool:
        try:
            current = self._lifecycle.describe_active_observation_claim(
                start=self._start, unit_id=self._unit.unit_id,
                process_generation_id=self._unit.process_id,
                deadline=self._fence.valid_until)
        except Exception:
            return False
        return ob._plain(current) == ob._plain(self._initial)

    def describe(self) -> str:
        return f"active observation claim {self.claim_id}"


__all__ = ["ActiveObservationPreparationClaim", "BINDING_SCHEMA",
           "ModelPreparationRefused", "SPEC_SCHEMA", "ScheduledModelPreparation"]
