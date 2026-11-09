"""gfx90a counter substrate: rocprof-v1 stats profile (RVP-C4-5), taxonomy
cross-check (RVP-C4-6), Hawkeye nine-counter presence (RVP-C4-7 offline half) and
the counter-citation rule for planner signals (RVP-CTR-1).

The substrate is the 465 SQ/TA/TCC/... counters enumerated by this ROCm 6.2 rocprof
into ``data/gfx90a-counters/gfx90a-counters-20260803.txt``.  A name in that listing
is ENUMERATED; it is not proof of a nonzero collected value, which is why a missing
reading is ``None`` here and never zero.
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
COUNTER_LISTING = REPO / "data" / "gfx90a-counters" / "gfx90a-counters-20260803.txt"
INTELLIPERF_MI2 = Path(__file__).with_name("fixtures") / "intelliperf_6dc899dd_MI2_input.json"

#: rocprof v1 --stats kernel table (rocm-6.2.0 libexec/rocprofiler/dform.py gen_data_bins).
STATS_COLUMNS = ("Name", "Calls", "TotalDurationNs", "AverageNs", "Percentage")
HAWKEYE_NINE = ("SQ_WAVES", "SQ_INSTS_VMEM", "SQ_INSTS_VALU", "SQ_INSTS_VALU_MFMA_BF16",
                "TCC_HIT_sum", "SQ_INSTS_LDS", "TCC_REQ_sum", "SQ_LDS_BANK_CONFLICT",
                "FETCH_SIZE")
AT_RISK = ("TCP_GATE_EN1_sum", "TCP_GATE_EN2_sum", "SQ_ACCUM_PREV_HIRES")


class SignalError(ValueError):
    pass


# --------------------------------------------------------------------- C4-5
@dataclass(frozen=True)
class KernelStat:
    name: str
    calls: int
    total_ns: int
    average_ns: float
    percentage: float


def stats_argv(rocprof: str, output_csv: Path, command: Sequence[str]) -> list[str]:
    """rocprof v1 (NOT rocprofv2/v3): ``--stats`` writes ``<output stem>.stats.csv``."""
    if not str(output_csv).endswith(".csv"):
        raise SignalError("-o must end in .csv so rocprof derives <stem>.stats.csv")
    return [rocprof, "--tool-version", "1", "--stats", "-o", str(output_csv), *command]


def find_stats_csv(directory: Path) -> Path:
    """The kernel table is ``*.stats.csv``; hip/hsa/copy tables end ``_stats.csv``."""
    found = sorted(p for p in Path(directory).glob("*.stats.csv"))
    if len(found) != 1:
        raise SignalError(f"expected exactly one *.stats.csv in {directory}, found {len(found)}")
    return found[0]


def parse_stats_csv(text: str) -> list[KernelStat]:
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    if not rows or tuple(c.strip().strip('"') for c in rows[0]) != STATS_COLUMNS:
        raise SignalError(f"stats header must be exactly {STATS_COLUMNS}")
    out = []
    for r in rows[1:]:
        if len(r) != 5:
            raise SignalError(f"stats row has {len(r)} fields: {r}")
        try:
            calls, total = int(r[1]), int(r[2])
            avg, pct = float(r[3]), float(r[4])
        except ValueError as exc:
            raise SignalError(f"non-numeric stats row: {r}") from exc
        if calls < 1 or total < 0 or pct < 0:
            raise SignalError(f"impossible stats row: {r}")
        out.append(KernelStat(r[0].strip().strip('"'), calls, total, avg, pct))
    if not out:
        raise SignalError("stats table is empty")
    return out


def profile(command: Sequence[str], out_dir: Path, *, rocprof: str, env: Mapping[str, str] | None = None,
            timeout_s: int = 1800,
            runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[KernelStat]:
    """Run ``command`` under rocprof v1 --stats and return kernels by descending time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = runner(stats_argv(rocprof, out_dir / "prof.csv", command), capture_output=True, text=True,
                  timeout=timeout_s, env=None if env is None else dict(env))
    if done.returncode != 0:
        raise SignalError(f"rocprof v1 rc={done.returncode}: {(done.stderr or done.stdout)[-300:]}")
    rows = parse_stats_csv(find_stats_csv(out_dir).read_text(encoding="utf-8"))
    return sorted(rows, key=lambda r: -r.total_ns)


# --------------------------------------------------------------------- listing
@dataclass(frozen=True)
class CounterInfo:
    name: str
    block: str | None          # hardware block, None for derived / unblocked
    limit: int | None          # counters of that block per pass
    expr: str | None           # derived-metric expression


_HEAD = re.compile(r"^\s*gfx90a:\d+ : (\S+) : ")
_BLOCK = re.compile(r"block (\S+) can only handle (\d+) counters")
_EXPR = re.compile(r"^\s+(\S+) = (.+)$")


def load_listing(path: Path = COUNTER_LISTING) -> dict[str, CounterInfo]:
    info: dict[str, list] = {}
    cur = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _HEAD.match(line)
        if m:
            cur = m.group(1)
            info.setdefault(cur, [None, None, None])
            continue
        if cur is None:
            continue
        b = _BLOCK.search(line)
        if b:
            info[cur][0], info[cur][1] = b.group(1), int(b.group(2))
            continue
        e = _EXPR.match(line)
        if e and e.group(1) == cur:
            info[cur][2] = e.group(2)
    return {k: CounterInfo(k, *v) for k, v in info.items()}


def _hardware(name: str, listing: Mapping[str, CounterInfo], _seen=None) -> set[str]:
    """Expand a derived counter to the hardware counters that occupy block slots."""
    c = listing.get(name)
    if c is None:
        return set()
    if c.block is not None or c.expr is None:
        return {name}
    seen = _seen or set()
    out: set[str] = set()
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", c.expr):
        if ident in ("sum", "max", "min", "avg") or ident == name or ident in seen:
            continue
        out |= _hardware(ident, listing, seen | {name})
    return out


def pass_legality(counters: Iterable[str], listing: Mapping[str, CounterInfo]) -> dict:
    """Per-block slot use of one pass versus the listing's per-block limits."""
    hw: set[str] = set()
    for n in counters:
        hw |= _hardware(n, listing)
    per: dict[str, int] = {}
    cap: dict[str, int] = {}
    for n in hw:
        c = listing[n]
        if c.block:
            per[c.block] = per.get(c.block, 0) + 1
            cap[c.block] = c.limit
    over = {b: (k, cap[b]) for b, k in per.items() if k > cap[b]}
    return {"slots": per, "over_limit": over, "legal": not over}


# --------------------------------------------------------------------- C4-6 / C4-7
def load_reference_partition(path: Path = INTELLIPERF_MI2) -> list[list[str]]:
    text = re.sub(r",(\s*[\]}])", r"\1", Path(path).read_text(encoding="utf-8"))
    return [list(j["pmc"]) for j in json.loads(text)["jobs"]]


def cross_check(partition: Sequence[Sequence[str]], listing: Mapping[str, CounterInfo],
                ours: Iterable[str] = ()) -> dict:
    flat = [c for p in partition for c in p]
    uniq = sorted(set(flat))
    absent = [c for c in uniq if c not in listing]
    return {
        "reference_passes": len(partition), "reference_total": len(flat), "reference_unique": len(uniq),
        "absent_from_validated": absent,
        "at_risk": {c: ("absent" if c not in listing else "enumerated") for c in AT_RISK},
        "at_risk_in_reference": [c for c in AT_RISK if c in uniq],
        "pass_legality": [pass_legality(p, listing) for p in partition],
        "not_in_our_set": sorted(set(uniq) - set(ours)) if ours else None,
    }


def hawkeye_presence(listing: Mapping[str, CounterInfo]) -> dict:
    absent = [c for c in HAWKEYE_NINE if c not in listing]
    return {"enumerated": [c for c in HAWKEYE_NINE if c in listing], "absent": absent,
            "pass_legality": pass_legality(HAWKEYE_NINE, listing),
            "collection_confirmed": False}   # C4-7 needs a GPU run: see row spec


# --------------------------------------------------------------------- CTR-1
@dataclass(frozen=True)
class CounterReading:
    name: str
    value: float | None        # None = not collected; never coerced to zero


DECLARED_SUBSET = {            # signal -> counters (all must be in the listing)
    "valu_mfma_busy": ("SQ_INSTS_VALU", "SQ_INSTS_VALU_MFMA_BF16", "SQ_BUSY_CYCLES"),
    "lds_bank_conflict": ("SQ_LDS_BANK_CONFLICT", "SQ_INSTS_LDS"),
    "memory_stall": ("SQ_WAIT_INST_LDS", "SQ_WAIT_ANY"),
    "occupancy": ("SQ_WAVES", "SQ_LEVEL_WAVES", "SQ_ACCUM_PREV_HIRES"),
    "bandwidth": ("FETCH_SIZE", "WRITE_SIZE"),
}


def declared_counters() -> tuple[str, ...]:
    return tuple(sorted({c for v in DECLARED_SUBSET.values() for c in v}))


def planner_block(readings: Sequence[CounterReading]) -> dict:
    """What the planner sees: present counters with values, absent ones listed as absent."""
    have = {r.name: r.value for r in readings if r.value is not None}
    return {"present": have,
            "absent": [c for c in declared_counters() if c not in have],
            "rule": "cite a PRESENT counter; absent counters are unknown, never zero"}


def validate_idea_citations(ideas: Sequence[Mapping[str, object]], block: Mapping) -> list[dict]:
    """Every idea must cite >=1 counter present in ``block``; citing an absent one is a defect."""
    present = set(block["present"])
    absent = set(block["absent"])
    out = []
    for i, idea in enumerate(ideas):
        cited = list(idea.get("cites") or [])
        problems = []
        if not cited:
            problems.append("no counter cited")
        problems += [f"cites absent counter {c}" for c in cited if c in absent]
        problems += [f"cites undeclared counter {c}" for c in cited if c not in present | absent]
        if cited and not (set(cited) & present):
            problems.append("cites no present counter")
        out.append({"index": i, "ok": not problems, "problems": problems})
    return out
