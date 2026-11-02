#!/usr/bin/env python3
"""Offline, fail-closed KVU-12.6 wave comparison. Never launches inference.

Input is a hash-pinned predeclaration (see test_kvu126_compare.py for its shape).
Each arm references actual waves_capture.jsonl and a separate native unique-cell
occupancy record bound to the wave's row_sha256. Nominal L*np, saved-token counts
and used_cells_est are deliberately not occupancy evidence. A producer must first
capture the timed occupancy record; this analyzer cannot retrofit one.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import kvu_cpu_matrix as matrix


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_pinned(root, item):
    path = root / item["path"]
    data = path.read_bytes()
    if digest(data) != item["sha256"]:
        raise ValueError(f"digest mismatch: {path}")
    return data


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing or nonfinite {name}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {name}")
    return value


def wave_observation(root, item, spec, min_samples):
    rows = [json.loads(line) for line in read_pinned(root, item["wave_file"]).splitlines() if line.strip()]
    selected = [r for r in rows if r.get("wave") == item["wave"]]
    if len(selected) != 1:
        raise ValueError("wave missing or ambiguous")
    row = selected[0]
    problems = matrix.validate_row(row)
    if problems:
        raise ValueError("invalid producer row: " + ", ".join(problems))
    for key in ("np", "L", "c"):
        if row.get(key) != spec[key]:
            raise ValueError(f"{key} differs from predeclaration")
    if row.get("result") != "OK" or row.get("kv_unified_from_log") != "true" or row.get("wave", 0) < 1:
        raise ValueError("wave is incomplete, non-unified, or a prefill wave")
    streams = row["np"]
    for field in ("prompt_n", "cache_n"):
        values = row.get(field)
        if not isinstance(values, list) or len(values) != streams:
            raise ValueError(f"missing per-stream {field}")
        for value in values:
            number(value, field)
    rates = row.get("decode_tps_per_stream")
    if not isinstance(rates, list) or len(rates) != streams:
        raise ValueError("missing per-stream decode rate")
    for value in rates:
        number(value, "decode_tps_per_stream", positive=True)
    number(row.get("decode_tps_sum"), "decode_tps_sum", positive=True)

    occupancy = json.loads(read_pinned(root, item["occupancy_file"]))
    if occupancy.get("schema") != "epyc.kvu126.native_unique_cells.v1" or occupancy.get("row_sha256") != row["row_sha256"]:
        raise ValueError("occupancy evidence not bound to producer wave")
    start = number(occupancy.get("decode_start_monotonic_s"), "decode start")
    end = number(occupancy.get("decode_end_monotonic_s"), "decode end")
    if end <= start:
        raise ValueError("invalid decode window")
    samples = occupancy.get("samples")
    if not isinstance(samples, list) or len(samples) < min_samples:
        raise ValueError("too few native occupancy samples")
    if not occupancy.get("native_source") or len(occupancy.get("native_source_sha256", "")) != 64:
        raise ValueError("missing native occupancy source or digest")
    native = json.loads(read_pinned(root, {"path": occupancy["native_source"], "sha256": occupancy["native_source_sha256"]}))
    for key in ("row_sha256", "decode_start_monotonic_s", "decode_end_monotonic_s", "samples"):
        if native.get(key) != occupancy.get(key):
            raise ValueError(f"native source disagrees on {key}")
    observed = []
    times = []
    for sample in samples:
        t = number(sample.get("monotonic_s"), "sample time")
        cells = number(sample.get("unique_cells"), "unique_cells")
        if not start <= t <= end or not isinstance(cells, int) or cells > row["c"]:
            raise ValueError("occupancy outside decode window or pool")
        observed.append(cells)
        times.append(t)
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("occupancy sample times are not strictly increasing")
    return {"row_sha256": row["row_sha256"], "wave": row["wave"],
            "identity": {k: row.get(k) for k in ("binary_sha256", "model", "mtp", "draft_depth", "cpu_list", "threads", "prompt_mode", "cache_idle_slots_argv")},
            "occupancy_source": occupancy["native_source"], "occupancy_source_sha256": occupancy["native_source_sha256"],
            "occupancy_samples": observed, "occupancy_median": statistics.median(observed),
            "prompt_n": row["prompt_n"], "resume_cache_n": row["cache_n"],
            "decode_tps_sum": row["decode_tps_sum"]}


def analyze(plan, root):
    if plan.get("schema") != "epyc.kvu126.comparison_plan.v1":
        raise ValueError("invalid predeclaration schema")
    tolerance = number(plan.get("max_occupancy_fraction_gap"), "occupancy tolerance")
    if tolerance > 1:
        raise ValueError("occupancy tolerance must be a fraction")
    min_samples = plan.get("min_occupancy_samples")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
        raise ValueError("at least two occupancy samples must be predeclared")
    min_waves = plan.get("min_waves_per_arm")
    if isinstance(min_waves, bool) or not isinstance(min_waves, int) or min_waves < 1:
        raise ValueError("minimum waves per arm must be predeclared")
    arms = {}
    for name, spec in plan["arms"].items():
        if len(spec.get("observations", [])) < min_waves:
            raise ValueError(f"too few observations for {name}")
        arms[name] = [wave_observation(root, item, spec, min_samples) for item in spec["observations"]]
        if len({x["row_sha256"] for x in arms[name]}) != len(arms[name]):
            raise ValueError(f"duplicate wave for {name}")
    comparisons = []
    for pair in plan["comparisons"]:
        left, right = pair["left"], pair["right"]
        lspec, rspec = plan["arms"][left], plan["arms"][right]
        if lspec["c"] != rspec["c"]:
            raise ValueError("different pools cannot use the same-pool comparison")
        lobs, robs = arms[left], arms[right]
        identities = [x["identity"] for x in lobs + robs]
        if any(any(value is None for value in identity.values()) for identity in identities):
            raise ValueError("missing comparison identity")
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("binary, model, recipe or placement mismatch")
        lm = statistics.median(x["occupancy_median"] for x in lobs)
        rm = statistics.median(x["occupancy_median"] for x in robs)
        gap = abs(lm - rm) / lspec["c"]
        comparisons.append({"left": left, "right": right, "occupancy_fraction_gap": gap,
                            "occupancy_matched": gap <= tolerance,
                            "left_decode_tps_median": statistics.median(x["decode_tps_sum"] for x in lobs),
                            "right_decode_tps_median": statistics.median(x["decode_tps_sum"] for x in robs),
                            "interpretation": "descriptive only; no performance or causal verdict"})
    if not comparisons:
        raise ValueError("no predeclared comparisons")
    return {"schema": "epyc.kvu126.comparison_result.v1", "status": "COMPLETE",
            "occupancy_tolerance_fraction": tolerance,
            "scope": "descriptive; model weights, process custody, repeat independence and scientific admission require separate verification",
            "arms": arms, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    try:
        raw = args.plan.read_bytes()
        if digest(raw) != args.expected_plan_sha256:
            raise ValueError("predeclaration digest mismatch")
        result = analyze(json.loads(raw), args.plan.resolve().parent)
        result["plan_sha256"] = digest(raw)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "INVALID", "reason": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
