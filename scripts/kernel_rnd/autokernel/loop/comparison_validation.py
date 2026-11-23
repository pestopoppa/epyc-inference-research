"""Fail-closed validation for retained paired measurements.

The retained artifact is an integrity binding within the store's trust boundary;
a content digest cannot authenticate a writer who controls the store itself.
"""
import math
import json
import statistics
from pathlib import Path


class ComparisonRefused(ValueError):
    pass


def validate_comparison(row, *, require_provenance=True):
    if not isinstance(row, dict):
        raise ComparisonRefused("comparison must be an object")
    try:
        json.dumps(row, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ComparisonRefused("comparison contains invalid or nonfinite data") from exc
    pairs = row.get("pairs")
    if type(pairs) is not int or pairs < 1:
        raise ComparisonRefused("comparison requires a positive pair count")
    for arm in ("anchor_samples", "candidate_samples"):
        values = row.get(arm)
        if not isinstance(values, list) or len(values) != pairs:
            raise ComparisonRefused(f"{arm} must contain every paired measurement")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ComparisonRefused(f"{arm} must be finite and positive")
    if require_provenance:
        for key in ("surface", "model"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ComparisonRefused(f"comparison {key} must be nonempty")
    fields = ("measured_value", "baseline_value", "baseline_path", "measured_under", "measurement_ref")
    present = [key in row and row[key] is not None for key in fields]
    if any(present) and not all(present) or require_provenance and not all(present):
        raise ComparisonRefused("comparison provenance block must be all-or-none")
    if not any(present):
        return row
    if (row["measured_value"] != statistics.median(row["candidate_samples"])
            or row["baseline_value"] != statistics.median(row["anchor_samples"])):
        raise ComparisonRefused("comparison values differ from paired arms")
    for key in ("baseline_path", "measured_under"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ComparisonRefused(f"{key} must be nonempty")
    if not Path(row["baseline_path"]).is_absolute():
        raise ComparisonRefused("baseline path must be absolute")
    if (len(row["measured_under"]) != 64 or any(
            char not in "0123456789abcdef" for char in row["measured_under"])):
        raise ComparisonRefused("measured_under must be a lowercase execution digest")
    if not isinstance(row["measurement_ref"], dict):
        raise ComparisonRefused("measurement reference must be an object")
    return row
