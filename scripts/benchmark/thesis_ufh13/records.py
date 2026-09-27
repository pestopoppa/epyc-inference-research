"""Per-question records: append-only, fsynced, resumable.

One JSON line per (arm, item). A line is written only after the item finished (answered, failed
or timed out), so a crash loses at most the item in flight, and ``done_keys`` lets a restart skip
everything already on disk. Failures are FINAL (pre-registration: every item is in the
denominator, no retries); only an item with no line is run again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .arms import ARMS, CONSULTANT_ROLE

RECORD_SCHEMA = "ufh13-thesis-record/v1"
RECORDS_NAME = "records.jsonl"
MANIFEST_NAME = "run_manifest.json"


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every complete line. A torn final line (crash mid-write) is ignored, never repaired."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("schema") == RECORD_SCHEMA:
                rows.append(row)
    return rows


def done_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(row["arm"], row["item_id"]) for row in rows}


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    with open(path, "a") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def summarize_receipts(arm: str, receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold one item's escalation receipts (one per /v1 call) into its cost and routing fields.

    ``consultant_device_seconds`` is llama-server prompt+decode time on the consultant's server:
    for A0 the whole request (the role is pinned to the consultant), otherwise the receipt's own
    consultant sum. ``None`` whenever a receipt lacks the number: an unmeasured cost is recorded
    as unmeasured, never as zero.
    """
    spec = ARMS[arm]
    out: dict[str, Any] = {
        "receipt_count": len(receipts),
        "escalation_enabled": None,
        "escalation_disabled_reasons": [],
        "escalation_fired": False,
        "escalation_triggers": [],
        "escalation_to_roles": [],
        "escalation_models": [],
        "review_verdicts": [],
        "final_answer_role": None,
        "consultant_ports": [],
        "consultant_device_seconds": None,
        "request_device_seconds": None,
        "non_consultant_device_seconds": None,
        "cost_problems": [],
    }
    if not receipts:
        out["cost_problems"].append("no_receipt")
        return out
    enabled = [bool(r.get("enabled")) for r in receipts]
    out["escalation_enabled"] = all(enabled)
    out["escalation_disabled_reasons"] = sorted(
        {str(r.get("disabled_reason")) for r in receipts if r.get("disabled_reason")}
    )
    consultant_total = 0.0
    request_total = 0.0
    for receipt in receipts:
        steps = [s for s in receipt.get("steps") or [] if isinstance(s, dict)]
        if receipt.get("fired"):
            out["escalation_fired"] = True
        for step in steps:
            out["escalation_triggers"].append(step.get("trigger"))
            out["escalation_to_roles"].append(step.get("to_role"))
            if step.get("model_id"):
                out["escalation_models"].append(step.get("model_id"))
            if step.get("trigger") == "review_gate":
                out["review_verdicts"].append(step.get("outcome"))
        out["final_answer_role"] = receipt.get("final_answer_role")
        for port in receipt.get("consultant_ports") or []:
            if port not in out["consultant_ports"]:
                out["consultant_ports"].append(port)
        request_ds = _num(receipt.get("request_device_seconds"))
        if request_ds is None:
            out["cost_problems"].append("request_device_seconds_missing")
        else:
            request_total += request_ds
        if spec.consultant_is_whole_request:
            if receipt.get("from_role") != CONSULTANT_ROLE:
                out["cost_problems"].append(
                    f"a0_served_by_{receipt.get('from_role')}_not_{CONSULTANT_ROLE}"
                )
            if request_ds is not None:
                consultant_total += request_ds
        else:
            consultant_ds = _num(receipt.get("consultant_device_seconds"))
            if consultant_ds is None:
                out["cost_problems"].append("consultant_device_seconds_missing")
            else:
                consultant_total += consultant_ds
    if spec.expect_escalation_enabled and not out["escalation_enabled"]:
        out["cost_problems"].append(
            "escalation_not_enabled:" + ",".join(out["escalation_disabled_reasons"] or ["?"])
        )
    if any(r.get("disabled_reason") == "flag_off" for r in receipts):
        out["cost_problems"].append("flag_off")
    if not any(p.endswith("_missing") for p in out["cost_problems"]):
        out["consultant_device_seconds"] = round(consultant_total, 6)
        out["request_device_seconds"] = round(request_total, 6)
        out["non_consultant_device_seconds"] = round(request_total - consultant_total, 6)
    return out
