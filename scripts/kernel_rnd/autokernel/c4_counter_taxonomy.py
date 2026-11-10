#!/usr/bin/env python3
"""C4 counter-taxonomy cross-checks (INF48-LIVE-3: RVP-C4-6, RVP-C4-7 offline leg).

Offline only. Nothing here runs a profiler or touches the GPU.

* `load_enumeration` reads OUR gfx90a taxonomy: the 465-counter `rocprofv2
  --list-counters` enumeration committed at data/gfx90a-counters/ (taken on our own
  MI210, 2026-08-03). Each entry is BASIC (carries a per-block simultaneous limit) or
  DERIVED (carries an `NAME = expr` line).
* `load_vendor_partition` reads AMD's own gfx90a/MI200 partition, transcribed
  verbatim from IntelliPerf (MIT, first-party AMD) at
  c4_reference/intelliperf_mi2_input.json: 74 counters over 6 rocprofv3 passes.
* `cross_check` diffs the two (RVP-C4-6) and flags the counters the intake row named
  at risk.
* `hawkeye_plan` resolves the nine Hawkeye cdna4 counters (RVP-C4-7) to their basic
  inputs and proves a single-pass schedule against the per-block limits. Whether they
  actually COLLECT is a GPU question (the C4-7 row); this answers only whether they
  are DECLARED for gfx90a and fit one pass.

Declared is not collected: `rocprof` v1 SQ/TA counters read zero on this box even
though v1's XML declares them, so presence in an enumeration is never reported as
a collection result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
RESEARCH_ROOT = HERE.parents[2]
ENUMERATION_PATH = RESEARCH_ROOT / "data/gfx90a-counters/gfx90a-counters-20260803.txt"
VENDOR_PARTITION_PATH = HERE / "c4_reference/intelliperf_mi2_input.json"
VENDOR_PARTITION_SOURCE = {
    "repo": "github.com/AMDResearch/intelliperf",
    "commit": "6dc899dd",
    "submodule_path": "external/guided-tuning",
    "submodule_tree": "7b078376a216a4ee4c3cd8c2f59fa0adc6c78940",
    "file": "configs/MI2/input.json",
    "git_blob": "6cf9eda36cb96476eb51c26d21559a4cfbdd08cf",
    "sha256": "9c0249f66a1747934f30317a75dc32db776cf2cb8c0ee191f8ed0c9297d54cec",
    "license": "MIT",
    "intake": "intake-1223#record",
}

#: RVP-C4-6: named at risk in the intake row; availability on ROCm 6.2/gfx90a was
#: UNVERIFIED there. SQ_ACCUM_PREV_HIRES is an accumulator whose value aliases the
#: counter scheduled immediately before it ("For counter N, increment by the value of
#: counter N-1"), so its reading depends on pass ORDER, not just pass membership.
AT_RISK = ("TCP_GATE_EN1_sum", "TCP_GATE_EN2_sum", "SQ_ACCUM_PREV_HIRES")

#: RVP-C4-7: the nine counters intake-1218#record (Hawkeye, cdna4) treats as
#: load-bearing.
HAWKEYE_NINE = (
    "SQ_WAVES", "SQ_INSTS_VMEM", "SQ_INSTS_VALU", "SQ_INSTS_VALU_MFMA_BF16",
    "TCC_HIT_sum", "SQ_INSTS_LDS", "TCC_REQ_sum", "SQ_LDS_BANK_CONFLICT", "FETCH_SIZE",
)

_ENTRY = re.compile(r"^\s*gfx90a:\d+ : (?P<name>[A-Za-z0-9_]+) : ")
_LIMIT = re.compile(r"block (?P<block>[A-Z0-9]+) can only handle (?P<n>\d+) counters")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNCS = frozenset(("sum", "max", "min", "CU_NUM", "SE_NUM", "SIMD_NUM", "GRBM_COUNT_X"))
_CONSTANTS = frozenset(("CU_NUM", "SE_NUM", "SIMD_NUM", "MAX_WAVE_SIZE", "TCC_NUM"))


class TaxonomyError(ValueError):
    pass


@dataclass(frozen=True)
class Counter:
    name: str
    block: str | None          # basic counters only
    block_limit: int | None    # basic counters only
    expr: str | None           # derived counters only

    @property
    def derived(self) -> bool:
        return self.expr is not None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_enumeration(text: str) -> dict[str, Counter]:
    """Parse `rocprofv2 --list-counters` text into name -> Counter."""
    entries: dict[str, dict] = {}
    current: str | None = None
    for line in text.splitlines():
        match = _ENTRY.match(line)
        if match:
            current = match.group("name")
            entries.setdefault(current, {"block": None, "limit": None, "expr": None})
            continue
        if current is None:
            continue
        limit = _LIMIT.search(line)
        if limit:
            entries[current]["block"] = limit.group("block")
            entries[current]["limit"] = int(limit.group("n"))
            continue
        stripped = line.strip()
        if stripped.startswith(f"{current} = "):
            entries[current]["expr"] = stripped[len(current) + 3:].strip()
    if not entries:
        raise TaxonomyError("enumeration holds no gfx90a counters")
    return {name: Counter(name, row["block"], row["limit"], row["expr"])
            for name, row in entries.items()}


def load_enumeration(path: Path = ENUMERATION_PATH) -> dict[str, Counter]:
    return parse_enumeration(Path(path).read_text(encoding="utf-8"))


def load_vendor_partition(path: Path = VENDOR_PARTITION_PATH,
                          expected_sha256: str | None = VENDOR_PARTITION_SOURCE["sha256"],
                          ) -> list[tuple[str, tuple[str, ...]]]:
    """[(pass_name, counters)] from IntelliPerf's MI2 input.json, hash-checked."""
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise TaxonomyError(f"vendor partition {path} does not match pinned sha256")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    passes = []
    for job in payload.get("jobs", ()):
        pmc = job.get("pmc")
        if not isinstance(pmc, list) or not all(isinstance(c, str) for c in pmc):
            raise TaxonomyError("vendor job pmc must be a list of names")
        passes.append((str(job.get("output_file")), tuple(pmc)))
    if not passes:
        raise TaxonomyError("vendor partition has no passes")
    return passes


def basic_inputs(name: str, taxonomy: Mapping[str, Counter],
                 _seen: frozenset[str] = frozenset()) -> frozenset[str]:
    """Transitive BASIC counters a (possibly derived) counter needs."""
    if name in _seen:
        raise TaxonomyError(f"derived-counter cycle at {name}")
    counter = taxonomy.get(name)
    if counter is None:
        raise TaxonomyError(f"{name} is not in the gfx90a taxonomy")
    if not counter.derived:
        return frozenset((name,))
    out: set[str] = set()
    for ident in _IDENT.findall(counter.expr or ""):
        if ident in _FUNCS or ident in _CONSTANTS or ident not in taxonomy:
            continue
        out |= basic_inputs(ident, taxonomy, _seen | {name})
    return frozenset(out)


def block_load(names: Iterable[str], taxonomy: Mapping[str, Counter]) -> dict[str, int]:
    basics: set[str] = set()
    for name in names:
        basics |= basic_inputs(name, taxonomy)
    load: dict[str, int] = {}
    for basic in basics:
        block = taxonomy[basic].block or "UNKNOWN"
        load[block] = load.get(block, 0) + 1
    return dict(sorted(load.items()))


def pass_violations(names: Iterable[str], taxonomy: Mapping[str, Counter]) -> dict[str, tuple[int, int]]:
    """block -> (needed, limit) for every block a single pass would overflow."""
    limits = {c.block: c.block_limit for c in taxonomy.values() if c.block}
    return {block: (need, limits[block])
            for block, need in block_load(names, taxonomy).items()
            if block in limits and need > limits[block]}


def cross_check(taxonomy: Mapping[str, Counter],
                passes: Sequence[tuple[str, Sequence[str]]]) -> dict:
    """RVP-C4-6: vendor partition vs our enumeration. Offline; nothing collected."""
    vendor = sorted({name for _, pmc in passes for name in pmc})
    missing = [name for name in vendor if name not in taxonomy]
    per_pass = []
    for pass_name, pmc in passes:
        known = [name for name in pmc if name in taxonomy]
        per_pass.append({
            "pass": pass_name,
            "counters": len(pmc),
            "absent_from_our_enumeration": [n for n in pmc if n not in taxonomy],
            "block_load": block_load(known, taxonomy),
            "over_block_limit": {k: list(v) for k, v in pass_violations(known, taxonomy).items()},
            "accum_prev_hires_predecessor": (
                pmc[pmc.index("SQ_ACCUM_PREV_HIRES") - 1]
                if "SQ_ACCUM_PREV_HIRES" in pmc and pmc.index("SQ_ACCUM_PREV_HIRES") > 0
                else None),
        })
    at_risk = []
    for name in AT_RISK:
        counter = taxonomy.get(name)
        passes_with = [p for p, pmc in passes if name in pmc]
        at_risk.append({
            "counter": name,
            "declared_in_our_enumeration": counter is not None,
            "kind": None if counter is None else ("derived" if counter.derived else "basic"),
            "expr": None if counter is None else counter.expr,
            "vendor_passes": passes_with,
            "collection_status": "UNVERIFIED (declared only; no execution performed)",
        })
    return {
        "schema": "epyc.autokernel.c4_counter_crosscheck.v1",
        "our_enumeration": {"path": str(ENUMERATION_PATH.relative_to(RESEARCH_ROOT)),
                            "counters": len(taxonomy),
                            "basic": sum(not c.derived for c in taxonomy.values()),
                            "derived": sum(c.derived for c in taxonomy.values())},
        "vendor": {**VENDOR_PARTITION_SOURCE, "passes": len(passes),
                   "distinct_counters": len(vendor)},
        "vendor_counters_absent_from_ours": missing,
        "vendor_counters_present_in_ours": len(vendor) - len(missing),
        "per_pass": per_pass,
        "at_risk": at_risk,
        "framing": "independent vendor cross-check on a CLOSED taxonomy, not a rescue",
    }


def hawkeye_plan(taxonomy: Mapping[str, Counter], names: Sequence[str] = HAWKEYE_NINE) -> dict:
    """RVP-C4-7 offline leg: declared? derived inputs? one-pass schedulable?"""
    rows = []
    for name in names:
        counter = taxonomy.get(name)
        rows.append({
            "counter": name,
            "declared": counter is not None,
            "kind": None if counter is None else ("derived" if counter.derived else "basic"),
            "basic_inputs": sorted(basic_inputs(name, taxonomy)) if counter else [],
        })
    present = [row["counter"] for row in rows if row["declared"]]
    violations = pass_violations(present, taxonomy)
    return {
        "schema": "epyc.autokernel.c4_hawkeye_plan.v1",
        "counters": rows,
        "absent": [row["counter"] for row in rows if not row["declared"]],
        "block_load": block_load(present, taxonomy),
        "single_pass": not violations,
        "over_block_limit": {k: list(v) for k, v in violations.items()},
        "collection_status": "UNVERIFIED until the C4-7 GPU row returns",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("what", choices=("crosscheck", "hawkeye"))
    args = parser.parse_args(argv)
    taxonomy = load_enumeration()
    if args.what == "crosscheck":
        out = cross_check(taxonomy, load_vendor_partition())
    else:
        out = hawkeye_plan(taxonomy)
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
