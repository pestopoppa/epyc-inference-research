#!/usr/bin/env python3
"""Post-hoc localization of already captured F32 output bytes; no new runs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def audit(root: Path):
    result = json.loads((root / "result.json").read_text())
    report = {"schema": "kvu19a2.output-difference-audit.v1", "cases": {},
              "scope": "post-hoc output-byte localization, not source-causation or a numerical threshold"}
    for case in (78, 80):
        rows = [r for r in result["runs"] if r["case"] == case]
        arms = {}
        for arm in ("base", "candidate"):
            group = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["repeat"])
            if len(group) != 5 or {r["repeat"] for r in group} != set(range(1, 6)):
                raise ValueError("incomplete original cohort")
            vectors = []
            for r in group:
                raw = (root / f"case{case}-{arm}-r{r['repeat']}" / f"out_{case:02d}.bin").read_bytes()
                if hashlib.sha256(raw).hexdigest() != r["output_sha256"] or not raw or len(raw) % 4:
                    raise ValueError("raw output missing, changed, or malformed")
                values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite output; numerical summary undefined")
                vectors.append(values)
            if len({len(v) for v in vectors}) != 1:
                raise ValueError("output dimensions differ")
            data = np.stack(vectors)
            spread = np.max(data, axis=0) - np.min(data, axis=0)
            changed = np.flatnonzero(spread > 0)
            breaks = np.flatnonzero(np.diff(changed) != 1) + 1
            groups = np.split(changed, breaks) if len(changed) else []
            arms[arm] = {"elements": data.shape[1], "varying_elements": len(changed),
                         "first_varying_flat_indices": changed[:20].tolist(),
                         "varying_flat_ranges_inclusive": [[int(g[0]), int(g[-1])] for g in groups],
                         "largest_within_arm_abs_difference": float(np.max(spread)),
                         "mean_within_arm_abs_spread": float(np.mean(spread)),
                         "nonfinite_elements": 0}
        report["cases"][str(case)] = arms
    return report


if __name__ == "__main__":
    output = audit(Path(sys.argv[1]))
    print(json.dumps(output, sort_keys=True, indent=2))
