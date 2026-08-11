#!/usr/bin/env python3
"""Compile-only kernel artifact comparison for the AutoKernel T0 preflight.

The parser consumes already-captured ``llvm-objdump``/``roc-obj`` text and
launches no process. Any resource or instruction-class movement makes the
performance claim *unconfirmed*: it is a pre-measurement veto, never evidence
that the candidate is incorrect or slower.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from . import schemas


MODULE_ID = "autokernel.artifact_diff/v1"

_KERNEL_PATTERNS = (
    re.compile(r"^\s*(?:Kernel|\.name)\s*[:=]\s*'?([^'\s]+)'?\s*$"),
    re.compile(r"^\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$"),
)
_RESOURCE_PATTERNS = {
    "vgpr": re.compile(r"^\s*\.(?:vgpr_count|amdhsa_next_free_vgpr)\s*[:= ]\s*(\d+)"),
    "sgpr": re.compile(r"^\s*\.(?:sgpr_count|amdhsa_next_free_sgpr)\s*[:= ]\s*(\d+)"),
    "scratch": re.compile(
        r"^\s*\.(?:scratch_size|private_segment_fixed_size|amdhsa_private_segment_fixed_size)"
        r"\s*[:= ]\s*(\d+)"),
}
_INSTRUCTION = re.compile(
    r"^\s*(?:[0-9a-fA-F]+:\s*)?(?:(?:[0-9a-fA-F]{2,8})\s+)*"
    r"([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)?)\b")


class ArtifactDiffError(ValueError):
    pass


def _instruction_class(mnemonic: str) -> str:
    stem = mnemonic.split(".", 1)[0]
    for prefix, label in (
            ("global_", "global_memory"), ("flat_", "global_memory"),
            ("buffer_", "global_memory"), ("ds_", "lds"),
            ("s_load", "scalar_memory"), ("s_store", "scalar_memory"),
            ("v_mfma", "matrix"), ("v_wmma", "matrix"),
            ("v_", "vector_alu"), ("s_", "scalar_alu")):
        if stem.startswith(prefix):
            return label
    return "other"


@dataclass(frozen=True)
class KernelArtifactStats:
    kernel: str
    vgpr: int
    sgpr: int
    scratch_bytes: int
    instruction_mix: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.kernel.strip():
            raise ArtifactDiffError("kernel name must be non-empty")
        for name in ("vgpr", "sgpr", "scratch_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ArtifactDiffError(f"{name} must be a non-negative integer")
        if tuple(sorted(self.instruction_mix)) != self.instruction_mix:
            raise ArtifactDiffError("instruction_mix must be sorted")
        if not self.instruction_mix:
            raise ArtifactDiffError("instruction_mix must not be empty")
        for name, count in self.instruction_mix:
            if not name or isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ArtifactDiffError("instruction_mix rows need a name and positive count")


@dataclass(frozen=True)
class ArtifactSnapshot:
    artifact_ref: str
    extractor_id: str
    kernels: tuple[KernelArtifactStats, ...]

    def __post_init__(self) -> None:
        if not self.artifact_ref.strip() or not self.extractor_id.strip():
            raise ArtifactDiffError("snapshot needs artifact_ref and extractor_id")
        names = [row.kernel for row in self.kernels]
        if not names or len(names) != len(set(names)):
            raise ArtifactDiffError("snapshot needs one or more uniquely named kernels")

    def by_kernel(self) -> Mapping[str, KernelArtifactStats]:
        return {row.kernel: row for row in self.kernels}


@dataclass(frozen=True)
class ArtifactMovement:
    kernel: str
    field: str
    anchor: object
    candidate: object


@dataclass(frozen=True)
class ArtifactDiff:
    anchor_ref: str
    candidate_ref: str
    movements: tuple[ArtifactMovement, ...]

    @property
    def claim_check(self) -> schemas.Check:
        if not self.movements:
            return schemas.Check(schemas.PASS, (
                "kernel register, scratch and instruction-class summaries are unchanged",))
        reasons = tuple(
            f"{row.kernel}: {row.field} moved from {row.anchor!r} to {row.candidate!r}"
            for row in self.movements)
        return schemas.Check(schemas.COULD_NOT_CHECK, reasons + (
            "compile-only movement makes the performance A/B unconfirmed; this vetoes "
            "the claim before T1 but does not fail correctness or disprove the candidate",))


def parse_objdump_text(text: str, *, artifact_ref: str,
                       extractor_id: str = "llvm-objdump/amdgpu-metadata+disassembly/v1"
                       ) -> ArtifactSnapshot:
    """Parse resource metadata and a coarse instruction mix from captured text."""
    if not isinstance(text, str) or not text.strip():
        raise ArtifactDiffError("objdump text must be non-empty")
    current: str | None = None
    resources: dict[str, dict[str, int]] = {}
    mixes: dict[str, Counter] = {}
    for line in text.splitlines():
        matched_name = None
        for pattern in _KERNEL_PATTERNS:
            match = pattern.match(line)
            if match:
                matched_name = match.group(1)
                break
        if matched_name is not None:
            current = matched_name
            resources.setdefault(current, {})
            mixes.setdefault(current, Counter())
            continue
        if current is None:
            continue
        resource_matched = False
        for field, pattern in _RESOURCE_PATTERNS.items():
            match = pattern.match(line)
            if match:
                resources[current][field] = int(match.group(1))
                resource_matched = True
                break
        if resource_matched:
            continue
        match = _INSTRUCTION.match(line)
        if match and not line.lstrip().startswith("."):
            mixes[current][_instruction_class(match.group(1))] += 1

    rows: list[KernelArtifactStats] = []
    for kernel in sorted(resources):
        missing = {"vgpr", "sgpr", "scratch"}.difference(resources[kernel])
        if missing:
            raise ArtifactDiffError(
                f"kernel {kernel!r} is missing resource fields {sorted(missing)}")
        if not mixes[kernel]:
            raise ArtifactDiffError(f"kernel {kernel!r} has no parsed instructions")
        rows.append(KernelArtifactStats(
            kernel, resources[kernel]["vgpr"], resources[kernel]["sgpr"],
            resources[kernel]["scratch"], tuple(sorted(mixes[kernel].items()))))
    return ArtifactSnapshot(artifact_ref, extractor_id, tuple(rows))


def compare_artifacts(anchor: ArtifactSnapshot,
                      candidate: ArtifactSnapshot) -> ArtifactDiff:
    if anchor.extractor_id != candidate.extractor_id:
        raise ArtifactDiffError("anchor and candidate use different artifact extractors")
    left = anchor.by_kernel()
    right = candidate.by_kernel()
    movements: list[ArtifactMovement] = []
    for kernel in sorted(set(left).union(right)):
        if kernel not in left:
            movements.append(ArtifactMovement(kernel, "kernel_presence", "absent", "present"))
            continue
        if kernel not in right:
            movements.append(ArtifactMovement(kernel, "kernel_presence", "present", "absent"))
            continue
        for field in ("vgpr", "sgpr", "scratch_bytes", "instruction_mix"):
            before, after = getattr(left[kernel], field), getattr(right[kernel], field)
            if before != after:
                movements.append(ArtifactMovement(kernel, field, before, after))
    return ArtifactDiff(anchor.artifact_ref, candidate.artifact_ref, tuple(movements))


def require_confirmed_for_t1(diff: ArtifactDiff | None) -> schemas.Check:
    """The T0 seam used before any performance/inference process may start."""
    if diff is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no compile-only artifact diff was supplied; GPU performance remains unconfirmed",))
    if not isinstance(diff, ArtifactDiff):
        raise TypeError("require_confirmed_for_t1 takes an ArtifactDiff or None")
    return diff.claim_check
