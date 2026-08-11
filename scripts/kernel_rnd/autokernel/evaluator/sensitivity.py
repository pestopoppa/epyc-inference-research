"""Fail-closed input-sensitivity screening for AutoKernel task populations.

The trusted evaluator supplies distances and digests.  This module only reduces
them; it performs no I/O and launches no workload.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from .. import schemas


REQUIRED_TRANSFORMS = frozenset(("identity", "x3", "x0p01", "negate"))
TRUSTED_PRODUCER = "trusted_evaluator"
SEED_VARIATION = "seed_variation"
TRANSFORM_VARIATION = "transform_variation"
_RECEIPT_RE = re.compile(
    r"^AK_SENS_V1 suite_version=(?P<version>[0-9a-f]{7,64}) "
    r"suite_seed=(?P<seed>[0-9]+) "
    r"transforms=(?P<transforms>identity,x3,x0p01,negate) "
    r"inputs=(?P<inputs>[0-9a-f]{16}(?:,[0-9a-f]{16}){3}) "
    r"outputs=(?P<outputs>[0-9a-f]{16}(?:,[0-9a-f]{16}){3})$")


@dataclass(frozen=True)
class SensitivityReceipt:
    suite_version: str
    suite_seed: int
    transforms: tuple[str, ...]
    input_hashes: tuple[str, ...]
    output_hashes: tuple[str, ...]


def parse_sensitivity_receipt(value: str) -> SensitivityReceipt:
    match = _RECEIPT_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"malformed sensitivity receipt: {value!r}")
    transforms = tuple(match.group("transforms").split(","))
    return SensitivityReceipt(
        suite_version=match.group("version"), suite_seed=int(match.group("seed")),
        transforms=transforms, input_hashes=tuple(match.group("inputs").split(",")),
        output_hashes=tuple(match.group("outputs").split(",")))


@dataclass(frozen=True)
class SensitivityObservation:
    suite_version: str
    operation: str
    case_id: str
    shape: tuple[int, ...]
    seed: int
    transform: str
    input_digest: str
    output_digest: str
    input_distance_from_seed_anchor: float
    output_distance_from_seed_anchor: float
    reference_only: bool
    produced_by: str
    evidence_ref: str

    def __post_init__(self) -> None:
        text = {
            "suite_version": self.suite_version,
            "operation": self.operation,
            "case_id": self.case_id,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "produced_by": self.produced_by,
            "evidence_ref": self.evidence_ref,
        }
        if any(not isinstance(value, str) or not value.strip() for value in text.values()):
            raise ValueError("sensitivity observation text fields must be non-empty")
        if not self.shape or any(isinstance(value, bool) or not isinstance(value, int)
                                 or value <= 0 for value in self.shape):
            raise ValueError("sensitivity shape must contain positive integers")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("sensitivity seed must be a non-negative integer")
        if self.transform not in REQUIRED_TRANSFORMS:
            raise ValueError(f"unknown sensitivity transform: {self.transform!r}")
        for name, value in (
                ("input_distance_from_seed_anchor", self.input_distance_from_seed_anchor),
                ("output_distance_from_seed_anchor", self.output_distance_from_seed_anchor)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")


@dataclass(frozen=True)
class SensitivityUnit:
    axis: str
    operation: str
    case_id: str
    shape: tuple[int, ...]
    slice_id: str
    sample_count: int
    input_changed: bool
    output_changed: bool
    scoreable: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SensitivityReport:
    suite_version: str
    units: tuple[SensitivityUnit, ...]
    check: schemas.Check

    @property
    def unscoreable_units(self) -> tuple[SensitivityUnit, ...]:
        return tuple(unit for unit in self.units if not unit.scoreable)

    def to_dict(self) -> dict:
        return {
            "suite_version": self.suite_version,
            "check": {"outcome": self.check.outcome, "reasons": list(self.check.reasons)},
            "units": [
                {
                    "axis": unit.axis,
                    "operation": unit.operation,
                    "case_id": unit.case_id,
                    "shape": list(unit.shape),
                    "slice_id": unit.slice_id,
                    "sample_count": unit.sample_count,
                    "input_changed": unit.input_changed,
                    "output_changed": unit.output_changed,
                    "scoreable": unit.scoreable,
                    "reasons": list(unit.reasons),
                }
                for unit in self.units
            ],
        }


def reduce_input_sensitivity(
        observations: Sequence[SensitivityObservation], *, min_seeds: int = 3,
        min_input_distance: float = 1e-9,
        min_output_distance: float = 1e-9) -> SensitivityReport:
    """Reduce one fixed-shape, reference-only 3-seed x 4-transform population.

    A surface is scoreable only when its materialized inputs vary both across
    seeds and across the declared transforms, and its reference output moves
    on both axes.  An insensitive unit is a finding, not a passing correctness
    case. Missing coverage and untrusted captures are COULD_NOT_CHECK.
    """
    if isinstance(min_seeds, bool) or not isinstance(min_seeds, int) or min_seeds < 3:
        raise ValueError("sensitivity screening requires at least three seeds")
    for name, value in (("min_input_distance", min_input_distance),
                        ("min_output_distance", min_output_distance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    if not observations:
        return SensitivityReport(
            suite_version="unknown", units=(),
            check=schemas.Check(schemas.COULD_NOT_CHECK, ("no sensitivity observations",)))

    versions = {item.suite_version for item in observations}
    if len(versions) != 1:
        return SensitivityReport(
            suite_version="mixed", units=(),
            check=schemas.Check(schemas.COULD_NOT_CHECK,
                                ("sensitivity observations mix suite versions",)))
    suite_version = next(iter(versions))
    trust_errors = []
    if any(not item.reference_only for item in observations):
        trust_errors.append("sensitivity population is not reference-only")
    if any(item.produced_by != TRUSTED_PRODUCER for item in observations):
        trust_errors.append("sensitivity population was not produced by the trusted evaluator")
    if trust_errors:
        return SensitivityReport(
            suite_version=suite_version, units=(),
            check=schemas.Check(schemas.COULD_NOT_CHECK, tuple(trust_errors)))

    grouped: dict[tuple[str, str, tuple[int, ...], str], list[SensitivityObservation]] = {}
    for item in observations:
        grouped.setdefault(
            (item.operation, item.case_id, item.shape, item.transform), []).append(item)

    by_surface_seed: dict[tuple[str, str, tuple[int, ...], int], set[str]] = {}
    for item in observations:
        by_surface_seed.setdefault(
            (item.operation, item.case_id, item.shape, item.seed), set()).add(item.transform)
    coverage_errors = []
    for (operation, case_id, shape, seed), transforms in sorted(by_surface_seed.items()):
        if transforms != REQUIRED_TRANSFORMS:
            coverage_errors.append(
                f"{operation}({case_id}){shape}/seed={seed} transform coverage is "
                f"{sorted(transforms)}, "
                f"expected {sorted(REQUIRED_TRANSFORMS)}")

    units = []
    for (operation, case_id, shape, transform), items in sorted(grouped.items()):
        seeds = {item.seed for item in items}
        reasons = []
        if len(seeds) != len(items):
            reasons.append("duplicate seed observations")
        if len(seeds) < min_seeds:
            reasons.append(f"only {len(seeds)} distinct seeds; require {min_seeds}")
        anchor_rows = [item for item in items
                       if item.input_distance_from_seed_anchor == 0.0]
        if len(anchor_rows) != 1:
            reasons.append("unit must identify exactly one zero-distance seed anchor")
        non_anchor = [item for item in items
                      if item.input_distance_from_seed_anchor != 0.0]
        input_changed = bool(non_anchor) and all(
            item.input_distance_from_seed_anchor > min_input_distance
            and item.input_digest != anchor_rows[0].input_digest
            for item in non_anchor) if len(anchor_rows) == 1 else False
        output_changed = bool(non_anchor) and any(
            item.output_distance_from_seed_anchor > min_output_distance
            and item.output_digest != anchor_rows[0].output_digest
            for item in non_anchor) if len(anchor_rows) == 1 else False
        if not input_changed:
            reasons.append("seed population did not measurably change every non-anchor input")
        if input_changed and not output_changed:
            reasons.append("output is invariant across input seeds or input-insensitive")
        units.append(SensitivityUnit(
            axis=SEED_VARIATION,
            operation=operation, case_id=case_id, shape=shape,
            slice_id=f"transform={transform}", sample_count=len(seeds),
            input_changed=input_changed,
            output_changed=output_changed,
            scoreable=not reasons, reasons=tuple(reasons)))

    by_seed: dict[tuple[str, str, tuple[int, ...], int], list[SensitivityObservation]] = {}
    for item in observations:
        by_seed.setdefault(
            (item.operation, item.case_id, item.shape, item.seed), []).append(item)
    for (operation, case_id, shape, seed), items in sorted(by_seed.items()):
        by_transform = {item.transform: item for item in items}
        reasons = []
        if len(by_transform) != len(items):
            reasons.append("duplicate transform observations")
        if set(by_transform) != REQUIRED_TRANSFORMS:
            reasons.append(
                f"transform coverage is {sorted(by_transform)}, "
                f"expected {sorted(REQUIRED_TRANSFORMS)}")
        anchor = by_transform.get("identity")
        non_anchor = [item for transform, item in by_transform.items()
                      if transform != "identity"]
        input_changed = anchor is not None and len(non_anchor) == 3 and all(
            item.input_digest != anchor.input_digest for item in non_anchor)
        output_changed = anchor is not None and bool(non_anchor) and any(
            item.output_digest != anchor.output_digest for item in non_anchor)
        if not input_changed:
            reasons.append("value transforms did not change every materialized input")
        if input_changed and not output_changed:
            reasons.append("output is invariant across value transforms or input-insensitive")
        units.append(SensitivityUnit(
            axis=TRANSFORM_VARIATION,
            operation=operation, case_id=case_id, shape=shape,
            slice_id=f"seed={seed}", sample_count=len(by_transform),
            input_changed=input_changed, output_changed=output_changed,
            scoreable=not reasons, reasons=tuple(reasons)))

    if coverage_errors:
        check = schemas.Check(schemas.COULD_NOT_CHECK, tuple(coverage_errors))
    else:
        failures = [
            f"{unit.operation}({unit.case_id}){unit.shape}/{unit.axis}/"
            f"{unit.slice_id}: "
            f"{'; '.join(unit.reasons)}"
            for unit in units if not unit.scoreable
        ]
        check = (schemas.Check(schemas.FAIL, tuple(failures)) if failures
                 else schemas.Check(schemas.PASS))
    return SensitivityReport(suite_version=suite_version, units=tuple(units), check=check)


def _shape_from_params(params: str) -> tuple[int, ...]:
    bracketed = re.search(r"(?:ne|shape)=\[([0-9, ]+)\]", params)
    if bracketed is not None:
        shape = tuple(int(value.strip()) for value in bracketed.group(1).split(","))
    else:
        values = re.findall(r"(?:^|,)(?:m|n|k)=([0-9]+)", params)
        shape = tuple(int(value) for value in values)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"cannot derive a positive shape from op params: {params!r}")
    return shape


def observations_from_csv_rows(
        rows: Sequence[Mapping[str, str]], *, expected_seeds: Sequence[int]) \
        -> tuple[SensitivityObservation, ...]:
    """Bind producer CSV receipts into reducer observations across fixed cases."""
    seeds = tuple(expected_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must contain at least three distinct seeds")
    parsed = []
    for row in rows:
        for field in ("op_name", "op_params", "sensitivity_receipt"):
            if not row.get(field):
                raise ValueError(f"sensitivity CSV row is missing {field}")
        receipt = parse_sensitivity_receipt(row["sensitivity_receipt"])
        if receipt.suite_seed not in seeds:
            raise ValueError(f"unexpected sensitivity suite seed {receipt.suite_seed}")
        parsed.append((row, receipt))
    versions = {receipt.suite_version for _row, receipt in parsed}
    if len(versions) != 1:
        raise ValueError("sensitivity CSV rows mix producer suite versions")

    by_case: dict[tuple[str, str], dict[int, SensitivityReceipt]] = {}
    for row, receipt in parsed:
        key = (row["op_name"], row["op_params"])
        if receipt.suite_seed in by_case.setdefault(key, {}):
            raise ValueError(f"duplicate sensitivity row for {key} seed {receipt.suite_seed}")
        by_case[key][receipt.suite_seed] = receipt

    observations = []
    anchor_seed = seeds[0]
    for (operation, params), receipts in sorted(by_case.items()):
        if set(receipts) != set(seeds):
            raise ValueError(
                f"{operation}({params}) has seeds {sorted(receipts)}, expected {sorted(seeds)}")
        anchor = receipts[anchor_seed]
        shape = _shape_from_params(params)
        for seed in seeds:
            receipt = receipts[seed]
            if receipt.transforms != anchor.transforms:
                raise ValueError(f"{operation}({params}) transform order drifted across seeds")
            for index, transform in enumerate(receipt.transforms):
                input_changed = receipt.input_hashes[index] != anchor.input_hashes[index]
                output_changed = receipt.output_hashes[index] != anchor.output_hashes[index]
                observations.append(SensitivityObservation(
                    suite_version=receipt.suite_version, operation=operation,
                    case_id=params, shape=shape, seed=seed, transform=transform,
                    input_digest=receipt.input_hashes[index],
                    output_digest=receipt.output_hashes[index],
                    input_distance_from_seed_anchor=1.0 if input_changed else 0.0,
                    output_distance_from_seed_anchor=1.0 if output_changed else 0.0,
                    reference_only=True, produced_by=TRUSTED_PRODUCER,
                    evidence_ref=f"csv://{operation}/{receipt.suite_seed}/{transform}"))
    return tuple(observations)
