"""AK-lineage diagnostics WRITE-SIDE adapter (VB-AK-LINEAGE-TAXONOMY-WIRE).

Wraps `lineage_edit_diagnostics` (proposal-level cycling detector + 9-label edit taxonomy)
into one immutable, digest-bound receipt carrying what a belief reader needs and cannot
reconstruct later: the store SHA-256, the proposal/attempt id join (id, label, recycled,
status), the detector version and threshold, label counts WITH the unlabeled remainder, and
the per-store n beside each rate.  Observation only: it grades nothing, chooses nothing and
never touches the store (read-only open, file hashed in place).  The labels are a text
proxy; the line-level authoring-bound capture stays separate (`lineage_capture`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from . import archive, lineage_edit_diagnostics as led

SCHEMA = "epyc.autokernel.lineage_taxonomy_receipt.v1"
PRODUCER_ID = "autokernel.loop.lineage_taxonomy_wire/v1"
DETECTOR_VERSION = "proposal-jaccard-shingle/v1"
METRIC_CYCLING = "autokernel_proposal_cycling_rate"


def _canonical(body: Any) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(body: Any) -> str:
    return hashlib.sha256(_canonical(body)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _store(path: Path, threshold: float) -> dict:
    path = Path(path).resolve(strict=True)
    rows = led.load(path)
    cycling = led.detect_cycling(rows, threshold)
    flags = cycling["flags"]
    labels = [led.classify(r["statement"]) for r in rows]
    counts = {lab: labels.count(lab) for lab in (*led.LABELS, "unlabeled")}
    n = len(rows)
    return {"path": str(path), "sha256": _file_sha(path), "n": n,
            "cycling": {"recycled": cycling["recycled"], "added": cycling["added"],
                        "rate": cycling["cycling_rate"]},
            "label_counts": counts, "unlabeled": counts["unlabeled"],
            "labeled_fraction": ((n - counts["unlabeled"]) / n) if n else None,
            "join": [{"attempt_id": r["id"], "label": lab, "recycled": bool(flag),
                      "status": r["status"]} for r, lab, flag in zip(rows, labels, flags)]}


def build(stores: Sequence[Path], *, producer_commit: str, threshold: float = 0.6,
          observed_at: str) -> dict:
    """Describe the stores exactly as read; no replay or grading inference."""
    if len(producer_commit) != 40 or any(c not in "0123456789abcdef" for c in producer_commit):
        raise ValueError("lineage taxonomy receipt requires a full producer commit")
    body = [_store(Path(p), threshold) for p in stores]
    material = {"schema": SCHEMA, "producer_id": PRODUCER_ID,
                "authority": "observation_only_no_selection_or_promotion",
                "detector": {"version": DETECTOR_VERSION, "threshold": threshold,
                             "taxonomy_version": led.SCHEMA, "labels": list(led.LABELS),
                             "labels_are_text_proxy": True},
                "producer_commit": producer_commit, "stores": body}
    receipt_id = _sha(material)
    measurements = [{
        "measurement_id": f"lineage-taxonomy:{receipt_id}:{s['sha256'][:12]}",
        "metric": METRIC_CYCLING, "value": s["cycling"]["rate"], "unit": "fraction",
        "metric_direction": "lower_better", "category": "CANDIDATE",
        "claim": ("Share of this store's proposals that return to an earlier same-symbol "
                  "proposal (Jaccard shingle match with an intervening different proposal); "
                  "descriptive proposal-level proxy, not the line-level code lineage"),
        "date": observed_at, "protocol_id": "", "reps": s["n"],
        "reps_basis": "scored:proposals in the store, not launches",
        "attestation_path": s["path"], "attestation_locator": s["path"],
        "attestation_sha256": s["sha256"], "attestation_verified": True,
        "extra": {"receipt_id": receipt_id, "n": s["n"], "unlabeled": s["unlabeled"],
                  "detector_version": DETECTOR_VERSION, "threshold": threshold},
    } for s in body if s["n"]]
    receipt = {**material, "receipt_id": receipt_id, "belief_measurements": measurements}
    receipt["receipt_sha256"] = _sha(receipt)
    return receipt


def publish(out_root: Path, stores: Sequence[Path], **kwargs: Any) -> Path:
    """Retain one immutable sidecar `<out_root>/lineage-taxonomy/<receipt_id>.json`."""
    receipt = build(stores, **kwargs)
    directory = Path(out_root) / "lineage-taxonomy"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{receipt['receipt_id']}.json"
    archive._retain_bytes(path, _canonical(receipt))
    return path


__all__ = ["DETECTOR_VERSION", "METRIC_CYCLING", "PRODUCER_ID", "SCHEMA", "build", "publish"]
