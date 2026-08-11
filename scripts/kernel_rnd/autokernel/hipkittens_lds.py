"""gfx90a LDS bank/phase solver and AutoKernel diagnostic adapter.

HipKittens' published CDNA3/CDNA4 topology is not evidence about CDNA2.  This
module carries only the transferable experiment method: collect the
``SQ_INSTS_LDS`` and ``SQ_LDS_BANK_CONFLICT`` counters for controlled
``ds_read_b128`` dispatches, solve the topology from the observations, and
project a hash-bound diagnostic context for a kernel-authoring controller.

The solver has no GPU or subprocess authority.  The benchmark runner owns the
claim and capture; this module is deterministic and offline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "epyc.autokernel.hipkittens_lds.v1"
CONTEXT_SCHEMA = "epyc.autokernel.hipkittens_lds_context.v1"
ARCH = "gfx90a"
WAVE_SIZE = 64
ACCESS_BANKS = 4
BANK_CANDIDATES = (16, 32, 64, 128)
UPSTREAM_REPO = "https://github.com/HazyResearch/HipKittens"
UPSTREAM_COMMIT = "a288366e4245528f74540b3fe446637cf8345745"
UPSTREAM_METHOD = "analysis/paper_experiments/phases/ds_read_b128"
TARGET_KERNEL = "autokernel_ds_read_b128_probe"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class LdsSolverError(ValueError):
    """The capture cannot support a gfx90a LDS-topology conclusion."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LdsSolverError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LdsSolverError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise LdsSolverError(f"{label} must be an integer")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise LdsSolverError(f"{label} must be an integer") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise LdsSolverError(f"{label} must be a finite integer")
    rendered = int(number)
    if rendered < minimum:
        raise LdsSolverError(f"{label} must be >= {minimum}")
    return rendered


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    kind: str
    thread_a: int
    thread_b: int
    bank_base: int | None
    repetition: int

    def __post_init__(self) -> None:
        if self.kind not in ("bank", "phase"):
            raise LdsSolverError("probe kind must be bank or phase")
        if not (0 <= self.thread_a < WAVE_SIZE and 0 <= self.thread_b < WAVE_SIZE):
            raise LdsSolverError("probe threads must be in one 64-lane wave")
        if self.thread_a == self.thread_b:
            raise LdsSolverError("probe threads must differ")
        if self.kind == "bank" and (self.bank_base is None or self.bank_base < ACCESS_BANKS):
            raise LdsSolverError("bank probes require bank_base >= 4")
        if self.kind == "phase" and self.bank_base is not None:
            raise LdsSolverError("phase probes do not carry bank_base")
        if self.repetition < 0:
            raise LdsSolverError("repetition must be non-negative")


@dataclass(frozen=True)
class CounterSample:
    dispatch_id: int
    kernel_name: str
    lds_insts: int
    conflict_cycles: int

    @property
    def conflict(self) -> bool:
        return self.conflict_cycles > 0


@dataclass(frozen=True)
class BankSolution:
    bank_count: int
    tested_bases: tuple[int, ...]
    conflict_bases: tuple[int, ...]
    candidate_mismatches: Mapping[int, int]


@dataclass(frozen=True)
class PhaseSolution:
    groups: tuple[tuple[int, ...], ...]
    tested_pairs: int

    @property
    def phase_count(self) -> int:
        return len(self.groups)


def bank_cases(*, max_bank: int = 127, repetitions: int = 3) -> tuple[ProbeCase, ...]:
    if max_bank < max(BANK_CANDIDATES) - ACCESS_BANKS:
        raise LdsSolverError("max_bank cannot distinguish the 128-bank candidate")
    if repetitions < 1:
        raise LdsSolverError("repetitions must be positive")
    return tuple(
        ProbeCase(
            case_id=f"bank-{bank_base:03d}-r{repetition}", kind="bank",
            thread_a=0, thread_b=1, bank_base=bank_base, repetition=repetition)
        for bank_base in range(ACCESS_BANKS, max_bank + 1)
        for repetition in range(repetitions)
    )


def phase_cases(*, repetitions: int = 1) -> tuple[ProbeCase, ...]:
    if repetitions < 1:
        raise LdsSolverError("repetitions must be positive")
    return tuple(
        ProbeCase(
            case_id=f"phase-{a:02d}-{b:02d}-r{repetition}", kind="phase",
            thread_a=a, thread_b=b, bank_base=None, repetition=repetition)
        for a in range(WAVE_SIZE)
        for b in range(a + 1, WAVE_SIZE)
        for repetition in range(repetitions)
    )


def expected_bank_conflict(bank_base: int, bank_count: int) -> bool:
    """Whether ds_read_b128 starting banks alias after LDS-bank wraparound.

    The vector's component banks issue in phases.  gfx90a's conflict counter
    reports the starting-bank alias, not arbitrary overlap between the two
    four-bank address intervals; the live control pattern is N, 2N, 3N.
    """
    if bank_base < ACCESS_BANKS or bank_count < 2 * ACCESS_BANKS:
        raise LdsSolverError("invalid bank geometry")
    return bank_base % bank_count == 0


def _median_conflict(samples: Iterable[CounterSample]) -> bool:
    values = [row.conflict_cycles for row in samples]
    if not values:
        raise LdsSolverError("probe case has no counter samples")
    return statistics.median(values) > 0


def solve_bank_count(cases: Sequence[ProbeCase], samples: Sequence[CounterSample],
                     *, candidates: Sequence[int] = BANK_CANDIDATES) -> BankSolution:
    if len(cases) != len(samples):
        raise LdsSolverError(
            f"bank case/sample count differs: {len(cases)} != {len(samples)}")
    grouped: dict[int, list[CounterSample]] = {}
    for case, sample in zip(cases, samples, strict=True):
        if case.kind != "bank" or case.bank_base is None:
            raise LdsSolverError("bank solver received a non-bank case")
        grouped.setdefault(case.bank_base, []).append(sample)
    observed = {base: _median_conflict(rows) for base, rows in grouped.items()}
    if len(observed) < 32:
        raise LdsSolverError("bank solver needs at least 32 distinct offsets")
    if not any(observed.values()) or all(observed.values()):
        raise LdsSolverError("bank capture lacks both conflict and no-conflict controls")
    mismatches = {
        bank_count: sum(
            actual != expected_bank_conflict(base, bank_count)
            for base, actual in observed.items())
        for bank_count in candidates
    }
    exact = [bank_count for bank_count, count in mismatches.items() if count == 0]
    if len(exact) != 1:
        raise LdsSolverError(
            f"bank topology is not uniquely solved; exact candidates={exact}, "
            f"mismatches={mismatches}")
    bank_count = exact[0]
    return BankSolution(
        bank_count=bank_count,
        tested_bases=tuple(sorted(observed)),
        conflict_bases=tuple(base for base in sorted(observed) if observed[base]),
        candidate_mismatches=mismatches,
    )


def solve_phases(cases: Sequence[ProbeCase], samples: Sequence[CounterSample]
                 ) -> PhaseSolution:
    if len(cases) != len(samples):
        raise LdsSolverError(
            f"phase case/sample count differs: {len(cases)} != {len(samples)}")
    grouped: dict[tuple[int, int], list[CounterSample]] = {}
    for case, sample in zip(cases, samples, strict=True):
        if case.kind != "phase":
            raise LdsSolverError("phase solver received a non-phase case")
        pair = tuple(sorted((case.thread_a, case.thread_b)))
        grouped.setdefault(pair, []).append(sample)
    expected_pairs = WAVE_SIZE * (WAVE_SIZE - 1) // 2
    if len(grouped) != expected_pairs:
        raise LdsSolverError(
            f"phase capture has {len(grouped)} unique pairs; expected {expected_pairs}")
    conflicts = {pair for pair, rows in grouped.items() if _median_conflict(rows)}

    # Same-phase must be an equivalence relation.  Build connected components,
    # then require every component to be a clique; noisy or missing edges refuse.
    adjacency = {lane: set() for lane in range(WAVE_SIZE)}
    for a, b in conflicts:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(range(WAVE_SIZE))
    groups = []
    while remaining:
        root = min(remaining)
        component = {root}
        frontier = [root]
        while frontier:
            lane = frontier.pop()
            for peer in adjacency[lane] - component:
                component.add(peer)
                frontier.append(peer)
        remaining -= component
        for a in component:
            missing = (component - {a}) - adjacency[a]
            if missing:
                raise LdsSolverError(
                    f"phase conflict relation is not transitive: lane {a} misses "
                    f"{sorted(missing)}")
        groups.append(tuple(sorted(component)))
    groups.sort(key=lambda row: row[0])
    if len(groups) < 1 or any(not row for row in groups):
        raise LdsSolverError("phase solver produced an empty topology")
    return PhaseSolution(groups=tuple(groups), tested_pairs=len(grouped))


def _column(fieldnames: Sequence[str], aliases: Sequence[str], label: str) -> str:
    folded = {name.casefold(): name for name in fieldnames}
    for alias in aliases:
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    raise LdsSolverError(f"profiler CSV is missing {label}; columns={fieldnames}")


def load_counter_samples(path: str | Path, *, expected_sha256: str,
                         target_kernel: str = TARGET_KERNEL) -> tuple[CounterSample, ...]:
    """Read rocprofv2/v3 long or wide counter CSV, ordered by dispatch id."""
    source = Path(path)
    if sha256_file(source) != _sha(expected_sha256, "expected_sha256"):
        raise LdsSolverError("profiler CSV hash mismatch")
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        if reader.fieldnames is None:
            raise LdsSolverError("profiler CSV has no header")
        dispatch_col = _column(
            reader.fieldnames, ("Dispatch_ID", "Dispatch_Id", "DispatchId"),
            "dispatch id")
        kernel_col = _column(
            reader.fieldnames, ("Kernel_Name", "KernelName", "Name"),
            "kernel name")
        fields = {name.casefold(): name for name in reader.fieldnames}
        long_format = "counter_name" in fields and "counter_value" in fields
        rows: dict[int, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, 2):
            kernel = _text(row.get(kernel_col), f"row {row_number} kernel")
            if target_kernel not in kernel:
                continue
            dispatch_id = _integer(
                row.get(dispatch_col), f"row {row_number} dispatch id")
            slot = rows.setdefault(dispatch_id, {"kernel": kernel})
            if slot["kernel"] != kernel:
                raise LdsSolverError(f"dispatch {dispatch_id} has multiple kernels")
            if long_format:
                name = _text(row.get(fields["counter_name"]),
                             f"row {row_number} counter name")
                if name in ("SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT"):
                    if name in slot:
                        raise LdsSolverError(
                            f"dispatch {dispatch_id} repeats counter {name}")
                    slot[name] = _integer(
                        row.get(fields["counter_value"]),
                        f"row {row_number} counter value")
            else:
                for name in ("SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT"):
                    column = fields.get(name.casefold())
                    if column is None:
                        raise LdsSolverError(f"profiler CSV is missing {name}")
                    slot[name] = _integer(row.get(column), f"row {row_number} {name}")
    samples = []
    for dispatch_id, row in sorted(rows.items()):
        missing = [name for name in ("SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT")
                   if name not in row]
        if missing:
            raise LdsSolverError(
                f"dispatch {dispatch_id} is missing counters {missing}")
        samples.append(CounterSample(
            dispatch_id=dispatch_id, kernel_name=row["kernel"],
            lds_insts=row["SQ_INSTS_LDS"],
            conflict_cycles=row["SQ_LDS_BANK_CONFLICT"],
        ))
    if not samples:
        raise LdsSolverError(f"profiler CSV has no {target_kernel} dispatches")
    return tuple(samples)


@dataclass(frozen=True)
class LdsTopologyContext:
    receipt_ref: str
    receipt_sha256: str
    campaign_id: str
    source_commit: str
    bank_count: int
    phase_groups: tuple[tuple[int, ...], ...]
    transfer_class: str

    def discovery_context(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_SCHEMA,
            "evidence": {
                "receipt_ref": self.receipt_ref,
                "receipt_sha256": self.receipt_sha256,
                "source_commit": self.source_commit,
            },
            "target_arch": ARCH,
            "instruction": "ds_read_b128",
            "lds_bank_count": self.bank_count,
            "phase_count": len(self.phase_groups),
            "phase_groups": [list(row) for row in self.phase_groups],
            "hipkittens_swizzle_transfer": self.transfer_class,
            "authority": "diagnostic_only",
        }


def load_topology_context(path: str | Path, *, expected_sha256: str | None = None
                          ) -> LdsTopologyContext:
    receipt_path = Path(path).resolve()
    raw = receipt_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _sha(
            expected_sha256, "expected_sha256"):
        raise LdsSolverError("LDS topology receipt hash mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LdsSolverError("LDS topology receipt is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise LdsSolverError(f"receipt schema must be {SCHEMA}")
    if payload.get("status") != "pass" or payload.get("authority") != "diagnostic_only":
        raise LdsSolverError("only passing diagnostic-only receipts may feed discovery")
    if payload.get("target_arch") != ARCH:
        raise LdsSolverError(f"receipt target_arch must be {ARCH}")
    bank = payload.get("bank_solution")
    phase = payload.get("phase_solution")
    if not isinstance(bank, Mapping) or not isinstance(phase, Mapping):
        raise LdsSolverError("receipt is missing topology solutions")
    groups_raw = phase.get("groups")
    if not isinstance(groups_raw, list):
        raise LdsSolverError("phase_solution.groups must be a list")
    groups = tuple(tuple(_integer(v, "phase lane") for v in row)
                   for row in groups_raw if isinstance(row, list))
    if sorted(v for row in groups for v in row) != list(range(WAVE_SIZE)):
        raise LdsSolverError("phase groups must partition all 64 lanes")
    bank_count = _integer(bank.get("bank_count"), "bank_count", minimum=1)
    transfer = _text(payload.get("swizzle_transfer_class"),
                     "swizzle_transfer_class")
    if transfer not in ("topology_matches_cdna3", "retune_required"):
        raise LdsSolverError("unknown swizzle transfer class")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise LdsSolverError("receipt.source must be an object")
    return LdsTopologyContext(
        receipt_ref=str(receipt_path), receipt_sha256=digest,
        campaign_id=_text(payload.get("campaign_id"), "campaign_id"),
        source_commit=_text(source.get("commit"), "source.commit"),
        bank_count=bank_count, phase_groups=groups, transfer_class=transfer,
    )


def topology_context_item(path: str | Path, *, expected_sha256: str):
    """Return a priced authoring-context item without importing on campaign path."""
    from .controller.authoring_contract import ContextItem, assert_prompt_hygiene

    context = load_topology_context(path, expected_sha256=expected_sha256)
    content = json.dumps(
        context.discovery_context(), sort_keys=True, separators=(",", ":"))
    assert_prompt_hygiene(content)
    return ContextItem(
        source_ref=f"gfx90a-lds://{context.receipt_sha256}",
        purpose="gfx90a LDS topology and HipKittens swizzle-transfer diagnostic",
        content=content,
    )


def summarize_samples(samples: Sequence[CounterSample]) -> dict[str, Any]:
    if not samples:
        raise LdsSolverError("cannot summarize an empty counter capture")
    return {
        "dispatches": len(samples),
        "lds_insts_min": min(row.lds_insts for row in samples),
        "lds_insts_max": max(row.lds_insts for row in samples),
        "conflict_cycles_min": min(row.conflict_cycles for row in samples),
        "conflict_cycles_max": max(row.conflict_cycles for row in samples),
        "conflicting_dispatches": sum(row.conflict for row in samples),
        "finite": all(math.isfinite(float(row.conflict_cycles)) for row in samples),
    }
