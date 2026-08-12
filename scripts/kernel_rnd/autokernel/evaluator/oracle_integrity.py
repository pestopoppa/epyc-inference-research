"""Fail-closed hostile-distribution and checker-isolation gates."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from .. import schemas


HOSTILE_DISTRIBUTIONS = (
    "baseline", "alternating", "sparse_outlier", "cancellation")
_HOSTILE_RE = re.compile(
    r"^AK_HOSTILE_V1 suite_version=(?P<version>[0-9a-f]{7,64}) "
    r"suite_seed=(?P<seed>[0-9]+) "
    r"distributions=baseline,alternating,sparse_outlier,cancellation "
    r"inputs=(?P<inputs>[0-9a-f]{16}(?:,[0-9a-f]{16}){3}) "
    r"completed=(?P<completed>[0-9]+)$")
_CHECKER_RE = re.compile(
    r"^AK_CHECKER_V1 suite_version=(?P<version>[0-9a-f]{7,64}) "
    r"oracle=host-double sibling_cpu=(?P<sibling>[01]) "
    r"cpu_reference=(?P<reference>[01]) tu_use_hip=(?P<hip>[01]) "
    r"tu_device_code=(?P<device>[01]) "
    r"tu_force_mmq=(?P<mmq>[01]) tu_cuda_fa=(?P<fa>[01]) "
    r"tu_rocwmma_fattn=(?P<rocwmma>[01])$")


@dataclass(frozen=True)
class HostileReceipt:
    suite_version: str
    suite_seed: int
    input_digests: tuple[str, ...]
    completed: int


@dataclass(frozen=True)
class CheckerReceipt:
    suite_version: str
    sibling_cpu: bool
    cpu_reference: bool
    tu_use_hip: bool
    tu_device_code: bool
    tu_force_mmq: bool
    tu_cuda_fa: bool
    tu_rocwmma_fattn: bool

    @property
    def isolated(self) -> bool:
        return (
            self.sibling_cpu and self.cpu_reference and
            not self.tu_device_code and not self.tu_force_mmq and
            not self.tu_cuda_fa and not self.tu_rocwmma_fattn)


def parse_hostile_receipt(value: str) -> HostileReceipt:
    match = _HOSTILE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"malformed hostile-distribution receipt: {value!r}")
    return HostileReceipt(
        suite_version=match.group("version"), suite_seed=int(match.group("seed")),
        input_digests=tuple(match.group("inputs").split(",")),
        completed=int(match.group("completed")))


def parse_checker_receipt(value: str) -> CheckerReceipt:
    match = _CHECKER_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"malformed checker-isolation receipt: {value!r}")
    return CheckerReceipt(
        suite_version=match.group("version"),
        sibling_cpu=match.group("sibling") == "1",
        cpu_reference=match.group("reference") == "1",
        tu_use_hip=match.group("hip") == "1",
        tu_device_code=match.group("device") == "1",
        tu_force_mmq=match.group("mmq") == "1",
        tu_cuda_fa=match.group("fa") == "1",
        tu_rocwmma_fattn=match.group("rocwmma") == "1")


def _row_failed(row: Mapping[str, str]) -> bool:
    return (
        row.get("supported") != "1" or row.get("hard_failure", "0") == "1" or
        bool(row.get("error_message")))


def evaluate_hostile_rows(
        rows: Sequence[Mapping[str, str]], *, expected_seed: int,
        expected_suite_version: str) -> schemas.Check:
    if not rows:
        return schemas.Check(
            schemas.COULD_NOT_CHECK, ("no hostile-distribution rows",))
    parsed = []
    try:
        for row in rows:
            if not row.get("op_name") or not row.get("op_params"):
                raise ValueError("hostile row is missing its operation or shape identity")
            parsed.append(parse_hostile_receipt(row.get("hostile_receipt", "")))
    except ValueError as error:
        return schemas.Check(schemas.COULD_NOT_CHECK, (str(error),))
    identity_errors = []
    if any(item.suite_seed != expected_seed for item in parsed):
        identity_errors.append("hostile-distribution suite seed drifted")
    if any(item.suite_version != expected_suite_version for item in parsed):
        identity_errors.append("hostile-distribution suite version drifted")
    if identity_errors:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(identity_errors))
    failures = []
    for row, receipt in zip(rows, parsed):
        surface = f"{row['op_name']}({row['op_params']})"
        if receipt.completed != len(HOSTILE_DISTRIBUTIONS):
            failures.append(
                f"{surface} completed {receipt.completed}/4 hostile distributions")
        if len(set(receipt.input_digests)) != len(HOSTILE_DISTRIBUTIONS):
            failures.append(f"{surface} did not materialize four distinct input populations")
        if _row_failed(row):
            failures.append(f"{surface} failed or was unsupported under a hostile distribution")
    return (schemas.Check(schemas.FAIL, tuple(failures)) if failures
            else schemas.Check(schemas.PASS))


def evaluate_checker_rows(
        rows: Sequence[Mapping[str, str]], *,
        expected_suite_version: str) -> schemas.Check:
    evidence_rows = [
        row for row in rows
        if row.get("property_receipt") or row.get("reference_receipt")]
    if not evidence_rows:
        return schemas.Check(
            schemas.COULD_NOT_CHECK, ("no property or reference checker rows",))
    parsed = []
    try:
        for row in evidence_rows:
            parsed.append(parse_checker_receipt(row.get("checker_receipt", "")))
    except ValueError as error:
        return schemas.Check(schemas.COULD_NOT_CHECK, (str(error),))
    if any(item.suite_version != expected_suite_version for item in parsed):
        return schemas.Check(
            schemas.COULD_NOT_CHECK, ("checker-isolation suite version drifted",))
    failures = []
    for row, receipt in zip(evidence_rows, parsed):
        surface = f"{row.get('op_name', '?')}({row.get('op_params', '?')})"
        if not receipt.isolated:
            failures.append(f"{surface} checker is not isolated from accelerated device paths")
    return (schemas.Check(schemas.FAIL, tuple(failures)) if failures
            else schemas.Check(schemas.PASS))
