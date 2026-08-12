"""Hash-bound P2-5j host-placement context for AutoKernel discovery.

The full four-arm result is preserved.  The adapter never emits only a winner,
and explicitly carries the receipt's no-kernel-speedup/no-carve authority
boundary into the discovery context.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts.benchmark import autokernel_p2_5j_receipt as receipt


CONTEXT_SCHEMA = "epyc.autokernel.p2_5j_placement_context.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PlacementContextError(ValueError):
    """A P2-5j receipt cannot safely become AutoKernel context."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlacementContextError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlacementContextError(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    rendered = _text(value, label)
    if not _SHA256_RE.fullmatch(rendered):
        raise PlacementContextError(f"{label} must be a lowercase SHA-256")
    return rendered


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlacementContextError(f"{label} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered <= 0:
        raise PlacementContextError(f"{label} must be positive and finite")
    return rendered


@dataclass(frozen=True)
class PlacementArm:
    arm: str
    role: str
    cpu_list: str
    cpu_region: str
    numa_node: int
    relation: str
    n: int
    median_decode_tps: float
    mad_decode_tps: float
    median_paired_ratio_to_incumbent: float


@dataclass(frozen=True)
class PlacementContext:
    receipt_ref: str
    receipt_file_sha256: str
    receipt_self_sha256: str
    campaign_id: str
    device_id: str
    pci_bdf: str
    device_numa_node: int
    arms: tuple[PlacementArm, ...]
    selected_arm: str
    observed_leader_arm: str
    verdict_status: str
    requires_ceiling_rederivation: bool

    def discovery_context(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_SCHEMA,
            "evidence": {
                "receipt_ref": self.receipt_ref,
                "receipt_file_sha256": self.receipt_file_sha256,
                "receipt_self_sha256": self.receipt_self_sha256,
                "campaign_id": self.campaign_id,
            },
            "device": {
                "device_id": self.device_id,
                "pci_bdf": self.pci_bdf,
                "numa_node": self.device_numa_node,
            },
            "host_placement_arms": [{
                "arm": arm.arm,
                "role": arm.role,
                "cpu_list": arm.cpu_list,
                "cpu_region": arm.cpu_region,
                "numa_node": arm.numa_node,
                "relation": arm.relation,
                "n": arm.n,
                "median_decode_tps": arm.median_decode_tps,
                "mad_decode_tps": arm.mad_decode_tps,
                "median_paired_ratio_to_incumbent": (
                    arm.median_paired_ratio_to_incumbent),
            } for arm in self.arms],
            "placement_verdict": {
                "status": self.verdict_status,
                "selected_arm": self.selected_arm,
                "observed_leader_arm": self.observed_leader_arm,
                "requires_np_context_ceiling_rederivation": (
                    self.requires_ceiling_rederivation),
            },
            "authority": {
                "placement_context_only": True,
                "observation_only": True,
                "kernel_speedup_claim": False,
                "carve_authorized": False,
                "production_activation_authorized": False,
            },
        }


def load_placement_context(
    path: str | Path, *, expected_sha256: str | None = None,
) -> PlacementContext:
    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and file_sha != _sha(
            expected_sha256, "expected_sha256"):
        raise PlacementContextError(
            f"receipt hash mismatch: expected {expected_sha256}, observed {file_sha}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlacementContextError("P2-5j receipt is not valid UTF-8 JSON") from exc
    root = _mapping(payload, "receipt")
    if root.get("schema") != receipt.SCHEMA:
        raise PlacementContextError(f"receipt.schema must be {receipt.SCHEMA}")
    if root.get("status") != "passed":
        raise PlacementContextError("only passed P2-5j receipts may feed discovery")
    if root.get("authority") != receipt.AUTHORITY:
        raise PlacementContextError("P2-5j authority boundary is absent or changed")
    self_sha = _sha(root.get("receipt_sha256"), "receipt_sha256")
    if self_sha != receipt.receipt_sha256(root):
        raise PlacementContextError("P2-5j receipt self-hash mismatch")
    definitions = _mapping(root.get("arm_definitions"), "arm_definitions")
    summaries = _mapping(root.get("arm_summaries"), "arm_summaries")
    if set(definitions) != set(receipt.ARM_SPECS) or set(summaries) != set(receipt.ARM_SPECS):
        raise PlacementContextError("receipt must preserve all four P2-5j arms")
    arms = []
    for arm, expected in receipt.ARM_SPECS.items():
        definition = _mapping(definitions.get(arm), f"arm_definitions.{arm}")
        summary = _mapping(summaries.get(arm), f"arm_summaries.{arm}")
        for field in ("cpu_list", "cpu_region", "numa_node", "relation", "role"):
            if definition.get(field) != expected[field] or summary.get(field) != expected[field]:
                raise PlacementContextError(f"{arm} {field} differs from the protocol")
        n = summary.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n != receipt.REQUIRED_BLOCKS:
            raise PlacementContextError(f"{arm}.n must be {receipt.REQUIRED_BLOCKS}")
        mad = summary.get("mad_decode_tps")
        if isinstance(mad, bool) or not isinstance(mad, (int, float)) or not math.isfinite(mad) or mad < 0:
            raise PlacementContextError(f"{arm}.mad_decode_tps must be finite and non-negative")
        arms.append(PlacementArm(
            arm=arm,
            role=expected["role"],
            cpu_list=expected["cpu_list"],
            cpu_region=expected["cpu_region"],
            numa_node=expected["numa_node"],
            relation=expected["relation"],
            n=n,
            median_decode_tps=_number(
                summary.get("median_decode_tps"), f"{arm}.median_decode_tps"),
            mad_decode_tps=float(mad),
            median_paired_ratio_to_incumbent=_number(
                summary.get("median_paired_ratio_to_incumbent"),
                f"{arm}.median_paired_ratio_to_incumbent"),
        ))
    identity = _mapping(root.get("identity"), "identity")
    device = _mapping(identity.get("device"), "identity.device")
    verdict = _mapping(root.get("verdict"), "verdict")
    selected = _text(verdict.get("selected_arm"), "verdict.selected_arm")
    if selected not in receipt.ARM_SPECS:
        raise PlacementContextError("selected_arm is unknown")
    observed_leader = _text(
        verdict.get("observed_leader_arm"), "verdict.observed_leader_arm")
    if observed_leader not in receipt.ARM_SPECS:
        raise PlacementContextError("observed_leader_arm is unknown")
    if selected != "I" or verdict.get("device_local_move_authorized") is not False:
        raise PlacementContextError("observation-only receipt cannot select a device-local arm")
    for field in ("kernel_speedup_claim", "carve_authorized",
                  "production_activation_authorized"):
        if verdict.get(field) is not False:
            raise PlacementContextError(f"verdict.{field} must remain false")
    return PlacementContext(
        receipt_ref=str(source_path),
        receipt_file_sha256=file_sha,
        receipt_self_sha256=self_sha,
        campaign_id=_text(root.get("campaign_id"), "campaign_id"),
        device_id=_text(device.get("device_id"), "identity.device.device_id"),
        pci_bdf=_text(device.get("pci_bdf"), "identity.device.pci_bdf"),
        device_numa_node=int(device.get("numa_node")),
        arms=tuple(arms),
        selected_arm=selected,
        observed_leader_arm=observed_leader,
        verdict_status=_text(verdict.get("status"), "verdict.status"),
        requires_ceiling_rederivation=(
            verdict.get("requires_np_context_ceiling_rederivation") is True),
    )
