#!/usr/bin/env python3
"""Which clock evidence compromises a GPU launch: strict, or a ratified request-phase band.

`gpu_clock_policy.json` beside this module selects the policy, and it ships as
`{"mode": "strict"}`: `min == max` over the whole launch window, the historical rule. The
band path exists for a host pinned to `power_dpm_force_performance_level = high`, where
the APCC_DFLL droops on current spikes (see `residency.SCLK_HWMON`) and the strict rule
therefore flags every launch. It judges the REQUEST PHASE only, so a clock moving during
model load is not held against the measurement.

The numbers are deliberately absent from this code. `droop_band_mhz` is a measurement
policy value the operator ratifies; no default exists and band mode refuses to parse
without it. There is no environment override, because a run must not be able to loosen
its own validity rule. Malformed config raises: it never falls back to either mode.

Fail-closed rule for band mode: anything short of positive proof (missing trace, too few
or too sparse request-phase samples, an unreadable clock, a level other than `high`) is
evidence of an unstable launch.

CLI: `python -m ...loop.clock_policy --records FILE` is the smoke. See `smoke`.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

SCHEMA = "epyc.autokernel.gpu_clock_policy.v1"
CONFIG_PATH = Path(__file__).with_name("gpu_clock_policy.json")
STRICT, BAND = "strict", "request_phase_band"
#: The only level the band path may run under; a config naming another is malformed.
PINNED_LEVEL = "high"
_BAND_KEYS = frozenset({"schema", "mode", "pinned_level", "nominal_sclk_mhz", "droop_band_mhz",
                        "min_request_samples", "max_sample_gap_s", "ratification"})


class ClockPolicyError(ValueError):
    """The policy file cannot be read as a valid policy."""


@dataclass(frozen=True)
class ClockPolicy:
    mode: str = STRICT
    pinned_level: str | None = None
    nominal_sclk_mhz: int | None = None
    droop_band_mhz: int | None = None
    min_request_samples: int | None = None
    max_sample_gap_s: float | None = None
    ratification: str | None = None
    sha256: str | None = None

    @property
    def is_band(self) -> bool:
        return self.mode == BAND

    def describe(self) -> dict:
        return {key: getattr(self, key) for key in (
            "mode", "pinned_level", "nominal_sclk_mhz", "droop_band_mhz",
            "min_request_samples", "max_sample_gap_s", "ratification", "sha256")}


def _positive(raw: Mapping, key: str, kinds: tuple) -> object:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, kinds) or value <= 0:
        raise ClockPolicyError(f"{key} must be a positive number, got {value!r}")
    return value


def parse(text: str) -> ClockPolicy:
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise ClockPolicyError(f"not JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ClockPolicyError("policy must be a JSON object")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if raw.get("mode") == STRICT:
        if set(raw) - {"schema", "mode"}:
            raise ClockPolicyError(f"strict policy carries band keys: {sorted(set(raw) - {'schema', 'mode'})}")
        if raw.get("schema") != SCHEMA:
            raise ClockPolicyError(f"schema must be {SCHEMA}")
        return ClockPolicy(sha256=sha)
    if raw.get("mode") != BAND:
        raise ClockPolicyError(f"unknown mode {raw.get('mode')!r}")
    if set(raw) != _BAND_KEYS or raw.get("schema") != SCHEMA:
        raise ClockPolicyError(f"band policy needs exactly {sorted(_BAND_KEYS)} and schema {SCHEMA}")
    if raw["pinned_level"] != PINNED_LEVEL:
        raise ClockPolicyError(f"pinned_level must be {PINNED_LEVEL!r}")
    nominal = _positive(raw, "nominal_sclk_mhz", (int,))
    band = _positive(raw, "droop_band_mhz", (int,))
    if band >= nominal:
        raise ClockPolicyError("droop_band_mhz must be below nominal_sclk_mhz")
    if not isinstance(raw["ratification"], str) or not raw["ratification"].strip():
        raise ClockPolicyError("ratification must name the operator's ratification record")
    return ClockPolicy(mode=BAND, pinned_level=PINNED_LEVEL, nominal_sclk_mhz=nominal,
                       droop_band_mhz=band,
                       min_request_samples=_positive(raw, "min_request_samples", (int,)),
                       max_sample_gap_s=float(_positive(raw, "max_sample_gap_s", (int, float))),
                       ratification=raw["ratification"], sha256=sha)


def load(path: Path | None = None) -> ClockPolicy:
    """The active policy. A missing file is the strict rule; a malformed one raises."""
    try:
        text = (CONFIG_PATH if path is None else path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ClockPolicy()
    return parse(text)


def _request_phase(record: Mapping) -> tuple[list[tuple[float, int]], list[tuple[float, object]], str | None]:
    """(samples in the request phase, level transitions as epoch pairs, reason if unusable)."""
    trace = record.get("clock_trace")
    if not isinstance(trace, Mapping):
        return [], [], "clock_trace_missing"
    try:
        t0, offsets, clocks = float(trace["t0"]), list(trace["t_ms"]), list(trace["mhz"])
        levels = [(t0 + float(ms) / 1000.0, level) for ms, level in trace["level_changes"]]
        start, end = float(record["request_start"]), float(record["request_end"])
    except (KeyError, TypeError, ValueError):
        return [], [], "clock_trace_malformed"
    if len(offsets) != len(clocks) or end < start:
        return [], [], "clock_trace_malformed"
    points = [(t0 + float(ms) / 1000.0, int(mhz)) for ms, mhz in zip(offsets, clocks)]
    return [p for p in points if start <= p[0] <= end], levels, None


def _covers_request_phase(record: Mapping) -> bool:
    try:
        return (float(record["window_start"]) <= float(record["request_start"])
                and float(record["request_end"]) <= float(record["window_end"]))
    except (KeyError, TypeError, ValueError):
        return False


def band_verdict(record: Mapping, policy: ClockPolicy) -> dict:
    """Judge one launch record against a band policy, from its trace alone."""
    reasons: list[str] = []
    if not _covers_request_phase(record):
        reasons.append("request_phase_not_covered")
    samples, levels, problem = _request_phase(record)
    if problem is not None:
        reasons.append(problem)
    clocks = [mhz for _, mhz in samples]
    readable = [mhz for mhz in clocks if mhz > 0]
    if problem is None:
        start, end = float(record["request_start"]), float(record["request_end"])
        if len(samples) < policy.min_request_samples:
            reasons.append("too_few_request_samples")
        if len(readable) != len(clocks):
            reasons.append("unreadable_clock_in_request_phase")
        edges = [start, *(t for t, _ in samples), end]
        if max(b - a for a, b in zip(edges, edges[1:])) > policy.max_sample_gap_s:
            reasons.append("sample_gap_exceeds_limit")
        at_start = [level for at, level in levels if at <= start]
        during = [level for at, level in levels if start < at <= end]
        if not at_start:
            reasons.append("pinned_level_unobserved")
        elif at_start[-1] != policy.pinned_level or any(v != policy.pinned_level for v in during):
            reasons.append("not_pinned_high")
        floor = policy.nominal_sclk_mhz - policy.droop_band_mhz
        if readable and min(readable) < floor:
            reasons.append("droop_below_band")
        if readable and max(readable) > policy.nominal_sclk_mhz:
            reasons.append("clock_above_nominal")
    return {"mode": BAND, "stable": not reasons, "reasons": reasons,
            "request_samples": len(samples),
            "request_sclk_min_mhz": min(readable) if readable else None,
            "request_sclk_max_mhz": max(readable) if readable else None,
            "policy_sha256": policy.sha256}


def launch_is_unstable(record: Mapping, policy: ClockPolicy) -> bool:
    """The evidence question `serving._independent_evidence` asks about one launch."""
    if policy.is_band and record.get("backend") != "cpu":
        return not band_verdict(record, policy)["stable"]
    return record.get("clock_stable") is False


def smoke(records: Sequence[Mapping], policy: ClockPolicy,
          level_reader: Callable[[], str | None], sclk_reader: Callable[[], int]) -> list[str]:
    """Reasons a live run under `policy` would refuse these launches; empty means pass.

    Stricter than the guard on purpose. The live guard re-measures a few flagged launches
    and refuses past a budget, while a smoke flags any. A GPU record that never stated its
    clock also fails here, since no launch evidence is not a passing smoke.
    """
    refusals: list[str] = []
    gpu = [r for r in records if isinstance(r, Mapping) and r.get("backend") != "cpu"]
    if not gpu:
        refusals.append("no_gpu_launch_records")
    for index, record in enumerate(gpu):
        if policy.is_band:
            verdict = band_verdict(record, policy)
            if not verdict["stable"]:
                refusals.append(f"launch[{index}]: {','.join(verdict['reasons'])}")
        elif record.get("clock_stable") is not True:
            refusals.append(f"launch[{index}]: strict clock_stable={record.get('clock_stable')!r}")
    if policy.is_band:
        level = level_reader()
        if level != policy.pinned_level:
            refusals.append(f"host level is {level!r}, the band path requires {policy.pinned_level!r}")
    if not sclk_reader():
        refusals.append("hwmon sclk unreadable")
    return refusals


def _read_records(path: Path) -> list[Mapping]:
    """A JSON document or JSONL of residency records or serving observations."""
    text = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
        rows = rows if isinstance(rows, list) else [rows]
    except ValueError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row["residency"] if isinstance(row, Mapping) and "residency" in row else row
            for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clock-policy smoke; exit 1 on any refusal.")
    parser.add_argument("--records", type=Path, required=True,
                        help="serving observations or residency records from a smoke launch")
    parser.add_argument("--policy", type=Path, default=None,
                        help="candidate policy file; defaults to the committed one")
    args = parser.parse_args(argv)
    from . import residency
    try:
        policy = load(args.policy)
        records = _read_records(args.records)
    except (OSError, ValueError, KeyError) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    refusals = smoke(records, policy, residency.performance_level, residency.sclk_mhz)
    print(json.dumps({"policy": policy.describe(), "launches": len(records),
                      "refusals": refusals}, indent=1, sort_keys=True))
    return 1 if refusals else 0


__all__ = ["BAND", "CONFIG_PATH", "ClockPolicy", "ClockPolicyError", "PINNED_LEVEL", "SCHEMA",
           "STRICT", "band_verdict", "launch_is_unstable", "load", "parse", "smoke"]

if __name__ == "__main__":
    raise SystemExit(main())
