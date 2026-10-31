#!/usr/bin/env python3
"""RT-6.2 diagnostic: calibration lag of the historical DeepInfer-style surrogate.

Separate from the original held-out same/cross-domain calibration and DI-081 gate.
Only earlier complete decode blocks of the SAME independent trace window calibrate
one target. All history/lag cells use identical eligible targets; no future block
or other prompt is admitted. No new model run, original-result rewrite or grade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement_replay as replay  # noqa: E402
import trace_io as tio  # noqa: E402

REFERENCE_COMMIT = "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01"
REFERENCE_SHA256 = "52241603827a1a1d19a395d48f5d618a7336fd1ceee631127e6782c644f736dc"


def select_targets(window_ids: np.ndarray, histories: list[int], lags: list[int], limit: int = 0):
    if (not histories or not lags or min(histories) < 1 or min(lags) < 0 or limit < 0):
        raise ValueError("histories positive, lags/limit nonnegative")
    targets = []
    start = max(histories) + max(lags)
    for window in np.unique(window_ids):
        indices = np.flatnonzero(window_ids == window)
        if indices.size and np.any(np.diff(indices) != 1):
            raise ValueError("independent window blocks must be contiguous")
        targets.extend((int(window), int(indices[0]), int(index)) for index in indices[start:])
    if limit and len(targets) > limit:
        chosen = np.unique(np.linspace(0, len(targets) - 1, limit).astype(int))
        targets = [targets[int(index)] for index in chosen]
    return targets


def sweep(loads, window_ids, histories, lags, bins, redundant, place_policy=replay.place_deepinfer, *, limit=0, seed=3):
    loads = np.asarray(loads)
    if (loads.ndim != 2 or loads.shape[0] != len(window_ids) or not loads.shape[1]
            or not np.isfinite(loads).all() or (loads < 0).any()):
        raise ValueError("invalid step load matrix/window IDs")
    if bins < 1 or redundant < 0:
        raise ValueError("bins positive and redundancy nonnegative")
    targets = select_targets(np.asarray(window_ids), histories, lags, limit)
    output = []
    for history in histories:
        for lag in lags:
            cells = {policy: [] for policy in ("historical_surrogate",)}
            provenance = []
            unknown_targets = []
            for window, first, target in targets:
                end = target - lag
                begin = end - history
                assert first <= begin < end <= target
                calibration = loads[begin:end]
                if loads[target].sum() <= 0:
                    unknown_targets.append({"window": window, "target_block": target, "reason": "zero_target_load"})
                    continue
                surrogate = place_policy(calibration, bins, redundant)
                for policy, placement in (("historical_surrogate", surrogate),):
                    point = loads[target:target + 1]
                    # Existing scorer and realizable dispatch, identical target seed across cells.
                    ideal = replay.eval_static(point, placement, bins)
                    random = replay.eval_static(point, placement, bins, np.random.default_rng(seed + target))
                    cells[policy].append({"idealized_max_over_mean": float(ideal[0]),
                                          "realizable_max_over_mean": float(random[0])})
                provenance.append({"window": window, "target_block": target,
                                   "history_begin": begin, "history_end_exclusive": end,
                                   "calibration_sha256": hashlib.sha256(np.asarray(calibration, dtype="<f8").tobytes()).hexdigest(),
                                   "target_sha256": hashlib.sha256(np.asarray(loads[target], dtype="<f8").tobytes()).hexdigest(),
                                   "placement_sha256": hashlib.sha256(tio.canonical_json(surrogate).encode()).hexdigest()})
            valid = bool(provenance)
            output.append({"history_blocks": history, "lag_blocks": lag, "status": "observed" if valid else "unsupported",
                           "reason": None if valid else "no_common_nonzero_target_blocks",
                           "target_count": len(provenance), "selected_targets": len(targets),
                           "upstream_equal_capacity_compatible": (loads.shape[1] + redundant) % bins == 0,
                           "unknown_targets": unknown_targets, "targets": provenance,
                           "policies": {policy: {"n": len(rows),
                               "idealized_max_over_mean": float(np.mean([r["idealized_max_over_mean"] for r in rows])) if rows else None,
                               "realizable_max_over_mean": float(np.mean([r["realizable_max_over_mean"] for r in rows])) if rows else None}
                               for policy, rows in cells.items()}})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domains", default="")
    parser.add_argument("--histories", default="8,32,128")
    parser.add_argument("--lags", default="0,1,8")
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--redundant-frac", type=float, default=0.0625)
    parser.add_argument("--tokens-per-step", type=int, choices=(1, 3), default=1)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)
    histories, lags = [int(x) for x in args.histories.split(",")], [int(x) for x in args.lags.split(",")]
    domains = args.domains.split(",") if args.domains else sorted(tio.discover(args.results_dir))
    rows, traces = [], []
    for domain in domains:
        trace = tio.load_domain(args.results_dir, domain, parts=("decode",), verify=True)
        trace = trace.select(trace.phase == tio.DECODE)
        traces.append(trace)
        # Independent decode sequences are not a continuous service-history stream.
        for window in np.unique(trace.win):
            positions = trace.pos[trace.win == window]
            if np.any(np.diff(positions) != 1):
                raise ValueError("decode sequence positions must be contiguous before blocking")
        block, count, windows = tio.blocks(trace, args.tokens_per_step)
        for layer in range(min(trace.L, args.layers) if args.layers else trace.L):
            experts = int(trace.n_experts[layer])
            loads = tio.layer_loads(trace.layer_ids(layer), block, count, experts)
            rows.append({"domain": domain, "layer": layer, "experts": experts,
                         "sweep": sweep(loads, windows, histories, lags, args.bins,
                                        round(args.redundant_frac * experts),
                                        limit=args.max_targets, seed=args.seed)})
    source = Path(__file__).resolve()
    doc = {"schema": "epyc.moe.calibration_lag_diagnostic.v1", "scope": "retained_decode_within_sequence_diagnostic_only",
           "metric": "max_bin_load_over_mean_lower_is_better", "original_DI081_decision": "unchanged_not_evaluated",
           "historical_results_rewritten": False, "reference_commit": REFERENCE_COMMIT,
           "reference_sha256": REFERENCE_SHA256, "policy": "historical_surrogate_not_upstream_reference",
           "reference_execution": "not_performed_by_this_analysis",
           "producer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "recipe": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "rows": rows}
    doc["dependency_sha256"] = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                for name, module in (("placement_replay", replay), ("trace_io", tio))}
    doc = tio.stamp(doc, traces, "moe_routing/calibration_lag_replay.py")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "rows": len(rows), "content_sha256": doc["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
