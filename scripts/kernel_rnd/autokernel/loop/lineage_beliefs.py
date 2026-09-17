"""Prospective, report-only lineage observation over committed loop outcomes.

The producer writes ClaimTuple-shaped data before any reader projects it.  This
module never grades a claim, chooses a parent, reads a model, or starts hardware.
Historic experiment rows have no producer receipt and are never backfilled.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from . import archive


SCHEMA = "epyc.autokernel.lineage_belief_receipt.v1"
PRODUCER_ID = "autokernel.loop.lineage_beliefs/v1"
METRIC = "autokernel_spawn_lineage_capture_fraction"
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(body: Any) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(body: Any) -> str:
    return hashlib.sha256(_canonical(body)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(outcome: Any, *, index: int, epoch: str) -> tuple[dict[str, Any], bool]:
    lineage = {key: getattr(outcome, key, None)
               for key in ("spawn_parent", "branch_id", "width", "depth")}
    journal = getattr(outcome, "journal_receipt", None)
    valid_journal = (isinstance(journal, dict)
                     and journal.get("campaign_id") == "ak-loop"
                     and journal.get("epoch_sha256") == epoch
                     and isinstance(journal.get("recorded_at"), str)
                     and bool(journal["recorded_at"])
                     and isinstance(journal.get("attempt_id"), str)
                     and bool(_SHA64.fullmatch(journal["attempt_id"]))
                     and isinstance(journal.get("payload_sha256"), str)
                     and bool(_SHA64.fullmatch(journal["payload_sha256"]))
                     and all(journal.get(key) == value
                             for key, value in lineage.items()))
    valid_lineage = (isinstance(lineage["spawn_parent"], str)
                     and bool(_SHA40.fullmatch(lineage["spawn_parent"]))
                     and isinstance(lineage["branch_id"], str)
                     and bool(lineage["branch_id"])
                     and type(lineage["width"]) is int and lineage["width"] > 0
                     and type(lineage["depth"]) is int and lineage["depth"] > 0)
    complete = valid_journal and valid_lineage
    return ({"completion_index": index, "status": outcome.status,
             "lineage": lineage, "journal": journal if valid_journal else None,
             "capture_complete": complete}, complete)


def build(outcomes: Sequence[Any], *, epoch: str, anchor_commit: str,
          producer_commit: str, producer_file_sha256: str,
          journal_root: Path, run_artifact: Path | None = None) -> dict[str, Any]:
    """Describe exact write-time rows; no replay or grading inference is made."""
    if not _SHA40.fullmatch(anchor_commit) or not _SHA40.fullmatch(producer_commit):
        raise ValueError("lineage receipt requires full source and producer commits")
    if not _SHA64.fullmatch(epoch) or not _SHA64.fullmatch(producer_file_sha256):
        raise ValueError("lineage receipt requires epoch and producer file SHA-256")
    source = None
    if run_artifact is not None:
        run_artifact = Path(run_artifact).resolve(strict=True)
        if not run_artifact.is_file():
            raise ValueError("loop-run artifact is not a regular file")
        source = {"path": str(run_artifact), "sha256": _file_sha256(run_artifact),
                  "schema": "epyc.autokernel.loop_run.v1"}
    rows, complete = [], 0
    for index, outcome in enumerate(outcomes):
        row, good = _row(outcome, index=index, epoch=epoch)
        rows.append(row)
        complete += int(good)
    dated = [row["journal"]["recorded_at"] for row in rows if row["journal"]]
    observed_at = max(dated) if dated else datetime.now(timezone.utc).isoformat()
    material = {"schema": SCHEMA, "producer_id": PRODUCER_ID,
                "authority": "observation_only_no_selection_or_promotion",
                "epoch_sha256": epoch, "anchor_commit": anchor_commit,
                "producer_source": {"research_commit": producer_commit,
                                    "run_py_sha256": producer_file_sha256},
                "journal": {"path": str(Path(journal_root).resolve() / "experiments.db"),
                            "table": "experiments", "rows": rows},
                "loop_run": source}
    capture_id = _sha(material)
    measurements = []
    if rows:
        measurements.append({
            "measurement_id": f"lineage:{capture_id}",
            "metric": METRIC, "value": complete / len(rows), "unit": "fraction",
            "metric_direction": "higher_better", "category": "CANDIDATE",
            "claim": ("Fraction of this run's outcomes with producer-captured spawn "
                      "lineage and a committed experiment-journal row; diagnostic only"),
            "date": observed_at, "protocol_id": "", "reps": len(rows),
            "reps_basis": "scored:run outcomes, not launches or prompts",
            "attestation_path": source["path"] if source else "",
            "attestation_locator": source["path"] if source else material["journal"]["path"],
            "attestation_sha256": source["sha256"] if source else "",
            "attestation_verified": True if source else None,
            "extra": {"capture_id": capture_id, "complete": complete,
                      "total": len(rows), "anchor_commit": anchor_commit,
                      "producer_research_commit": producer_commit},
        })
    receipt = {**material, "capture_id": capture_id,
               "belief_measurements": measurements}
    receipt["receipt_sha256"] = _sha(receipt)
    return receipt


def publish(store_root: Path, outcomes: Sequence[Any], **kwargs: Any) -> Path:
    """Retain one immutable sidecar after the run/journal have committed."""
    store_root = Path(store_root)
    receipt = build(outcomes, journal_root=store_root, **kwargs)
    directory = store_root / "lineage-beliefs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{receipt['capture_id']}.json"
    archive._retain_bytes(path, _canonical(receipt))
    return path


__all__ = ["METRIC", "PRODUCER_ID", "SCHEMA", "build", "publish"]
