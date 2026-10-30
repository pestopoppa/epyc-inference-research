"""R24-9: compare prospectively recorded effects for identical measured work.

The code digest is necessary but insufficient: the work key also binds the measured
baseline, measurement epoch, model, metric, surface, recipe and launch arguments.
Unknown inputs are not asserted equal. This check does not rank or skip work.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..controller import anchor_integrity, experiments
from . import loop

SCHEMA = "epyc.autokernel.same_code_tripwire.v1"


class ObservedComparison:
    """Carry tripwire evidence without changing the native comparison's decisions."""

    def __init__(self, original: Any, evidence: dict):
        self.original, self.evidence = original, evidence

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def to_dict(self) -> dict:
        return {**self.original.to_dict(), "same_code_tripwire": self.evidence}


def inspect(store_root: Path, comparison: Any, *, anchor_build: Path,
            candidate_build: Path, library: str, measurement_epoch: str,
            launch_identity: dict) -> ObservedComparison:
    """Attach prospective code/work identity; refuse an above-floor repeat excursion.

    A missing digest, identity input, floor, or previous row is reported as such.
    `effect` is the native signed improvement fraction on both sides; no metric
    inversion or new grading threshold is introduced.
    """
    native = comparison.to_dict()
    def code(build):
        bin_dir = Path(build) / "bin"
        executable = "llama-server" if str(native.get("surface") or "").startswith("serving:") else "llama-bench"
        required = (bin_dir / library, bin_dir / executable)
        if not all(path.is_file() for path in required):
            return None
        paths = sorted({*bin_dir.glob("*.so*"), *required}, key=lambda p: p.name)
        rows = []
        for path in paths:
            if not path.is_file():
                continue
            sections = {name for name, _, _ in anchor_integrity.section_spans(path)}
            need = ({".text", ".hip_fatbin"}
                    if path.name == library and library == "libggml-hip.so"
                    else {".text"})
            if not need <= sections:
                return None
            digest = anchor_integrity.code_digest(path)
            if not digest:
                return None
            rows.append((path.name, digest))
        return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()

    a = code(anchor_build)
    c = code(candidate_build)
    model = native.get("model") or launch_identity.get("model")
    required = (measurement_epoch, native.get("surface"), model,
                native.get("effect_unit"), launch_identity)
    evidence = {"schema": SCHEMA, "status": "unavailable", "anchor_code_sha256": a,
                "candidate_code_sha256": c, "work_key": None}
    if not a or not c or any(x is None or x == "" for x in required):
        evidence["reason"] = "code digest or measured work identity unavailable"
        return ObservedComparison(comparison, evidence)
    work = {"schema": SCHEMA, "measurement_epoch": measurement_epoch,
            "anchor_code_sha256": a, "candidate_code_sha256": c,
            "surface": native["surface"], "model": model,
            "effect_unit": native["effect_unit"], "launch_identity": launch_identity,
            "recipe_hash": native.get("recipe_hash"), "metric": native.get("metric"),
            "request_digest": native.get("request_digest"),
            "baseline_scope": native.get("baseline_scope")}
    # Serving work must carry its original recipe and request identity. An absent
    # field is unknown, never interchangeable with a later, populated record.
    if str(native.get("surface") or "").startswith("serving:") and not all(
            (work["recipe_hash"], work["metric"], work["request_digest"])):
        evidence["reason"] = "serving recipe, metric or request identity unavailable"
        return ObservedComparison(comparison, evidence)
    key = hashlib.sha256(json.dumps(work, sort_keys=True, separators=(",", ":"),
                                    default=str).encode()).hexdigest()
    evidence["work_key"] = key
    effect = native.get("effect")
    floor = native.get("noise_floor_pct")
    if not isinstance(effect, (int, float)) or not math.isfinite(effect):
        evidence["reason"] = "valid native effect unavailable"
        return ObservedComparison(comparison, evidence)
    with experiments.ExperimentStore(store_root) as store:
        prior = store.prior_same_code(key)
    if prior is None:
        evidence["status"] = "first_observation"
        return ObservedComparison(comparison, evidence)
    evidence.update(prior_attempt_id=prior["attempt_id"],
                    prior_effect_fraction=prior["effect_fraction"],
                    effect_fraction=effect, prior_floor_pct=prior["floor_pct"],
                    floor_pct=floor)
    old_floor = prior["floor_pct"]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
               for x in (floor, old_floor)):
        evidence["status"] = "floor_unavailable"
        return ObservedComparison(comparison, evidence)
    evidence["delta_pct"] = (effect - prior["effect_fraction"]) * 100.0
    # The CURRENT comparison's existing calibrated floor decides this reading.
    # The old floor is retained as provenance, not combined into a new policy.
    evidence["comparison_floor_pct"] = floor
    if abs(evidence["delta_pct"]) > floor:
        evidence["status"] = "same_code_excursion"
        raise loop.MeasurementInvalid(
            "same-code remeasurement exceeds the existing floor; session/evaluator excursion",
            {"same_code_tripwire": evidence, "comparison": native})
    evidence["status"] = "within_floor"
    return ObservedComparison(comparison, evidence)
