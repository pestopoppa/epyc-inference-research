"""Prospective direct-serving observations, using the established belief row vocabulary.

No campaign/plan/grant identity or scientific protocol is invented. Original native
vectors and resolved inputs are retained; CPU allowed lists remain diagnostic facts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CAPTURE_SCHEMA = "epyc.vidya.legacy_serving_capture.v1"
RECEIPT_SCHEMA = "epyc.vidya.legacy_serving_receipt.v1"
PRODUCER_ID = "autokernel.loop.serving_beliefs/v1"
MAX_BYTES = 64 << 20


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def prepare(recipe, *, anchor, candidate, anchor_build, candidate_build, frozen_requests, pairs):
    if type(pairs) is not int or not 1 <= pairs <= 64:
        raise ValueError("belief capture supports at most 64 original pairs")
    if recipe.metric != "aggregate_tok_s":
        raise ValueError("belief capture only describes the original aggregate token-rate metric")
    inputs = {"producer": PRODUCER_ID,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "recipe": recipe.to_dict(), "pairs": pairs,
            "resolved_arms": {"anchor": None if anchor is None else anchor.to_dict(),
                              "candidate": None if candidate is None else candidate.to_dict()},
            "build_paths": {"anchor": str(anchor_build), "candidate": str(candidate_build)},
            "requests": None if frozen_requests is None else [
                [prompt_id, hashlib.sha256(body).hexdigest()]
                for prompt_id, body in frozen_requests],
            "protocol_id": "", "loaded_instrument_attestation": "not_recorded"}
    return json.loads(json.dumps(inputs, allow_nan=False))


def native_body(comparison):
    # The existing loop wrapper adds these two display/subject labels AFTER compare.
    return {key: value for key, value in comparison.items()
            if key not in {"belief_capture", "surface", "baseline_scope"}}


def finish(comparison, inputs):
    native = native_body(comparison)
    native_digest = digest(native)
    capture_id = digest({"native_sha256": native_digest, "inputs": inputs})
    rows = []
    for arm, category in (("anchor", "BASELINE"), ("candidate", "CANDIDATE")):
        samples = native[f"{arm}_samples"]
        windows = native[f"{arm}_residency"]
        if len(samples) != inputs["pairs"] or len(windows) != len(samples):
            raise ValueError("belief capture requires exact completed launch membership")
        resolved = inputs["resolved_arms"][arm]
        date = datetime.fromtimestamp(max(row["window_end"] for row in windows), timezone.utc).isoformat()
        rows.append({"measurement_id": f"legacy-serving:{capture_id}:{arm}",
                     "metric": native["metric"], "value": native[f"{arm}_tok_s"],
                     "unit": "t/s", "metric_direction": "higher_better", "category": category,
                     "claim": f"{arm} observed median aggregate serving rate; protocol and scientific witnesses unqualified",
                     "date": date, "protocol_id": "", "reps": len(samples),
                     "reps_basis": "scored:original independent server launches; not threads/prompts/affinity samples",
                     "extra": {"arm": arm, "capture_id": capture_id,
                               "native_sha256": native_digest,
                               "build_path": inputs["build_paths"][arm],
                               "recipe_hash": native["recipe_hash"],
                               "request_digest": native.get("request_digest"),
                               "resolved_snapshot_digest": None if resolved is None else resolved.get("snapshot_digest"),
                               "execution_digest": None if resolved is None else resolved.get("execution_digest"),
                               "model_path": inputs["recipe"]["model"],
                               "applicability": "direct_serving_observation_only",
                               "cpu_facts": "dependency_only_not_placement_or_contention_proof"}})
    capture = {"schema": CAPTURE_SCHEMA, "inputs": inputs, "capture_id": capture_id,
               "native_sha256": native_digest, "belief_measurements": rows}
    capture["capture_sha256"] = digest(capture)
    if len(json.dumps({"native": native, "capture": capture}).encode()) > MAX_BYTES:
        raise ValueError("belief capture byte budget exceeded")
    return capture


def _write_exact(root: Path, name: str, body):
    from .status import write_json
    expected = json.dumps(body, indent=2, sort_keys=True).encode()
    if len(expected) > MAX_BYTES:
        raise ValueError("belief export byte budget exceeded")
    path = root / name
    if path.exists():
        with path.open("rb") as stream:
            if stream.read(MAX_BYTES + 1) != expected:
                raise ValueError("belief export path already holds different bytes")
    else:
        write_json(root, name, body, prefix=".serving-belief-")
    with path.open("rb") as stream:
        actual = stream.read(MAX_BYTES + 1)
    if actual != expected:
        raise ValueError("belief export readback differs")
    return {"path": name, "sha256": hashlib.sha256(actual).hexdigest(), "size": len(actual)}


def export(store_root, attempt, *, campaign_id, epoch, recorded_at):
    comparison = attempt.get("comparison")
    if not isinstance(comparison, dict) or "belief_capture" not in comparison:
        return None  # Pre-hook and non-serving records are never reconstructed.
    capture = comparison["belief_capture"]
    if capture != finish(comparison, capture["inputs"]):
        raise ValueError("prospective serving capture does not rederive")
    root = Path(store_root) / "serving-beliefs"
    native = {"comparison": comparison, "campaign_id": campaign_id,
              "epoch": epoch, "recorded_at": recorded_at,
              "mechanism_id": attempt.get("mechanism_id")}
    source = _write_exact(root / "sources", f"{capture['capture_id']}.json", native)
    source["path"] = "sources/" + source["path"]
    receipt = {"schema": RECEIPT_SCHEMA, "capture_id": capture["capture_id"],
               "capture_sha256": capture["capture_sha256"], "native_reference": source}
    _write_exact(root, f"{capture['capture_id']}.json", receipt)
    return root / f"{capture['capture_id']}.json"
