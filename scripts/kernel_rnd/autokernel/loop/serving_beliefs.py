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
RUNTIME_CAPTURE_SCHEMA = "epyc.vidya.legacy_serving_capture.v2"
MATCHED_CAPTURE_SCHEMA = "epyc.vidya.legacy_serving_capture.v3"
RECEIPT_SCHEMA = "epyc.vidya.legacy_serving_receipt.v1"
PRODUCER_ID = "autokernel.loop.serving_beliefs/v1"
MAX_BYTES = 64 << 20


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def prepare(recipe, *, anchor, candidate, anchor_build, candidate_build, frozen_requests, pairs,
            candidate_recipe=None, runtime_pair=None, measurement_plan=None):
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
    if runtime_pair is not None:
        inputs.update(runtime_pair=runtime_pair.to_dict(),
                      candidate_recipe=candidate_recipe.to_dict())
    if measurement_plan is not None:
        inputs["measurement_plan"] = measurement_plan
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
                               "recipe_hash": (native["candidate_recipe_hash"]
                                               if arm == "candidate" and "runtime_pair" in inputs
                                               else native["recipe_hash"]),
                               "request_digest": native.get("request_digest"),
                               "resolved_snapshot_digest": None if resolved is None else resolved.get("snapshot_digest"),
                               "execution_digest": None if resolved is None else resolved.get("execution_digest"),
                               "model_path": inputs["recipe"]["model"],
                               **({"measurement_plan": inputs["measurement_plan"]}
                                  if "measurement_plan" in inputs else {}),
                               "applicability": "direct_serving_observation_only",
                               "cpu_facts": "dependency_only_not_placement_or_contention_proof"}})
    capture = {"schema": (MATCHED_CAPTURE_SCHEMA if "measurement_plan" in inputs else
                          RUNTIME_CAPTURE_SCHEMA if "runtime_pair" in inputs else CAPTURE_SCHEMA),
               "inputs": inputs, "capture_id": capture_id,
               "native_sha256": native_digest, "belief_measurements": rows}
    capture["capture_sha256"] = digest(capture)
    if len(_canonical_bytes({"native": native, "capture": capture})) > MAX_BYTES:
        raise ValueError("belief capture byte budget exceeded")
    return capture


def _write_exact(root: Path, name: str, body):
    from . import archive

    expected = _canonical_bytes(body)
    if len(expected) > MAX_BYTES:
        raise ValueError("belief export byte budget exceeded")
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    try:
        archive._retain_bytes(path, expected)
    except archive.RatchetRefused as exc:
        raise ValueError("belief export path already holds different bytes") from exc
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


class PlannerFeedback:
    """Lazy synchronous bridge; diagnostic faults never change loop outcomes."""

    def __init__(self, store_root, root_repo=None):
        import os
        import threading

        self.store_root = Path(store_root)
        self.root_repo = Path(root_repo or os.environ.get("EPYC_ROOT_REPO", "/workspace"))
        self._lock = threading.Lock()
        self._reader = None
        self._failure = None

    def _load(self):
        import importlib
        import sys

        expected = (self.root_repo / "scripts/vidya/adapters/autokernel_legacy_serving.py").resolve()
        if not expected.is_file():
            raise ValueError(f"serving belief reader is unavailable: {expected}")
        sys.path.insert(0, str(self.root_repo / "scripts/vidya"))
        module = importlib.import_module("adapters.autokernel_legacy_serving")
        if Path(module.__file__).resolve() != expected:
            raise ValueError("loaded serving belief reader differs from the selected ROOT")
        self._reader = module.PlannerFeedback(self.store_root)

    def _failed(self, exc):
        import sys

        self._failure = f"{type(exc).__name__}: {exc}"[:800]
        print(f"warning: serving belief feedback unavailable: {self._failure}", file=sys.stderr)

    def exported(self, receipt):
        with self._lock:
            try:
                # A concrete new export is a bounded recovery opportunity; do not
                # permanently disable feedback after a transient startup I/O fault.
                if self._reader is None:
                    self._load()
                if self._reader is not None:
                    self._reader.ingest(receipt)
                    self._failure = None
            except Exception as exc:
                self._failed(exc)

    def context(self, scope):
        with self._lock:
            try:
                if callable(scope):
                    scope = scope()
                if scope is None:
                    return {"status": "scope_unavailable", "rows": [], "errors": [],
                            "qualified_measurement": False}
                if self._reader is None and self._failure is None:
                    self._load()
                if self._reader is not None:
                    result = self._reader.context(scope, as_of=datetime.now(timezone.utc).isoformat())
                    if self._failure:
                        result["errors"].append(self._failure)
                    return result
            except Exception as exc:
                self._failed(exc)
            return {"status": "unavailable", "rows": [], "errors": [self._failure],
                    "qualified_measurement": False}


def feedback_scope(*, epoch, recipe, resolved, frozen_requests, anchor_build):
    """Original workload/anchor identity, never a model-family or recipe-name match."""
    from .serving import request_digest

    if recipe is None or resolved is None or frozen_requests is None:
        return None
    return {"epoch": epoch, "recipe_hash": recipe.recipe_hash,
            "request_digest": request_digest(recipe, frozen_requests),
            "model": resolved.model.to_dict(),
            "anchor_execution_digest": resolved.execution_digest,
            "anchor_build": str(anchor_build)}
