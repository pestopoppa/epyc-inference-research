"""Static experimental datatype targets for gfx90a kernel authoring.

These are task contracts, not hardware capabilities or performance claims.
In particular, gfx90a has no native FP8 MFMA instruction.  The first FP8 target
therefore stores weights in FP8, decodes/upcasts in software, and chooses a
bf16 compute path only after the real shape/profile says MFMA or vector GEMV is
appropriate.  Cross-vendor latency and throughput never enter prompt context.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


SCHEMA = "epyc.autokernel.datatype_targets.v1"
FP8_TARGET_ID = "fp8_weight_bf16_compute_gfx90a"
NVFP4_TARGET_ID = "nvfp4_microscaled_gfx90a"


class DatatypeTargetError(ValueError):
    """A datatype target selection is invalid or overclaims gfx90a support."""


@dataclass(frozen=True)
class DatatypeTarget:
    target_id: str
    state: str
    storage_formats: tuple[str, ...]
    decode_path: str
    compute_paths: tuple[str, ...]
    accumulator: str
    native_gfx90a_mfma: bool
    prerequisites: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "state": self.state,
            "storage_formats": list(self.storage_formats),
            "decode_path": self.decode_path,
            "compute_paths": list(self.compute_paths),
            "accumulator": self.accumulator,
            "native_gfx90a_mfma": self.native_gfx90a_mfma,
            "prerequisites": list(self.prerequisites),
            "claim_boundary": self.claim_boundary,
        }


TARGETS = {
    FP8_TARGET_ID: DatatypeTarget(
        target_id=FP8_TARGET_ID,
        state="experimental_authoring_target",
        storage_formats=("fp8_e4m3fn", "fp8_e5m2"),
        decode_path="software_decode_and_upcast_to_bf16",
        compute_paths=("bf16_vector_gemv", "bf16_mfma_when_shape_is_compute_bound"),
        accumulator="fp32",
        native_gfx90a_mfma=False,
        prerequisites=(
            "independent_storage_format_decoder",
            "c2_correctness_and_hostile_distributions",
            "exact_shape_strongest_baseline",
            "upcast_cost_attribution",
            "whole_model_exit_gate",
        ),
        claim_boundary=(
            "Software FP8 weight storage is eligible only with decode/upcast into a supported BF16 "
            "compute path and no native-FP8-compute or compute-headroom claim; fresh MI210 evidence "
            "is required, byte-count reduction alone proves no performance gain, and BF16 MFMA is "
            "forbidden as a batch-one assumption"
        ),
    ),
    NVFP4_TARGET_ID: DatatypeTarget(
        target_id=NVFP4_TARGET_ID,
        state="deferred_until_fp8_upcast_path_is_measured",
        storage_formats=("nvfp4_e2m1_microscaled",),
        decode_path="not_selected",
        compute_paths=(),
        accumulator="unselected",
        native_gfx90a_mfma=False,
        prerequisites=("fp8_weight_bf16_compute_gfx90a_terminal_result",),
        claim_boundary="No authoring campaign until the simpler FP8 upcast target closes",
    ),
}


def select(target_ids: Sequence[str]) -> tuple[DatatypeTarget, ...]:
    requested = tuple(target_ids)
    if not requested or len(requested) != len(set(requested)):
        raise DatatypeTargetError("datatype target selection must be non-empty and unique")
    if any(not isinstance(target_id, str) or not target_id for target_id in requested):
        raise DatatypeTargetError("datatype target ids must be non-empty strings")
    unknown = sorted(set(requested) - set(TARGETS))
    if unknown:
        raise DatatypeTargetError(f"unknown datatype target ids: {unknown}")
    return tuple(TARGETS[target_id] for target_id in requested)


def target_context_item(target_ids: Sequence[str]):
    """Return hash-bound, non-numeric authoring context for selected targets."""
    from .controller.authoring_contract import ContextItem, assert_prompt_hygiene

    targets = select(target_ids)
    payload = {
        "schema": SCHEMA,
        "authority": "design_target_only",
        "target_architecture": "gfx90a",
        "hardware_facts": {
            "native_fp8_mfma": False,
            "wavefront_size": 64,
        },
        "targets": [target.to_dict() for target in targets],
        "excluded": {
            "cross_vendor_latency_or_throughput": True,
            "hopper_or_cdna3_capability_transfer": True,
        },
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert_prompt_hygiene(content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return ContextItem(
        source_ref=f"datatype-targets://{SCHEMA}/{digest}",
        purpose="experimental gfx90a datatype authoring target",
        content=content,
    )


__all__ = [
    "FP8_TARGET_ID", "NVFP4_TARGET_ID", "SCHEMA", "TARGETS",
    "DatatypeTarget", "DatatypeTargetError", "select", "target_context_item",
]
