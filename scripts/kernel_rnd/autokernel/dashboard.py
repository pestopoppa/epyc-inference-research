#!/usr/bin/env python3
"""Terminal AutoKernel campaign result -> durable dashboard contract v2.

This is a deliberately small replacement for the deleted ``surface`` package.
The current campaign driver owns one candidate run and journals its terminal
``STOP_STATE``; this module projects that already-fsynced event into the existing
hub contract.  It never runs work, reads live host state, promotes a candidate,
or claims that a banked candidate is a champion.

Freshness comes from ``JournalEntry.written_at``.  Re-exporting an old entry
therefore cannot make a dead loop look fresh.  Sections the current driver does
not own (champion, headroom, release package) are explicit ``not_reported``
values rather than invented summaries.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from . import journal, schemas, storage

MODULE_ID = "autokernel.dashboard/v2"
DEFAULT_EXPORT_PATH = "/mnt/raid0/llm/autokernel/surface/kernel_dashboard.json"

OBSERVATION_NOTICE = (
    "Every figure here is an OBSERVATION (MEASUREMENT.md). This terminal campaign "
    "view never authorizes keep/revert/deploy/promote beyond the campaign result "
    "already recorded, and AutoKernel holds no freeze or cutover authority."
)


class DashboardError(Exception):
    """The terminal record cannot be represented or exported safely."""


def _unreported(reason: str) -> dict:
    return {"status": schemas.SECTION_NOT_REPORTED, "as_of": None, "reason": reason}


def _result_from(entry: journal.JournalEntry) -> Mapping[str, Any]:
    if not isinstance(entry, journal.JournalEntry):
        raise DashboardError("entry must be a journal.JournalEntry")
    if entry.kind != journal.KIND_STOP_STATE:
        raise DashboardError(
            f"terminal dashboard export needs {journal.KIND_STOP_STATE}, got {entry.kind}")
    result = entry.payload.get("result")
    if not isinstance(result, Mapping):
        raise DashboardError("STOP_STATE payload.result must be a mapping")
    if result.get("schema") != "epyc.autokernel.campaign_result.v1":
        raise DashboardError("STOP_STATE payload.result is not a campaign_result.v1")
    campaign_id = result.get("campaign_id")
    if campaign_id != entry.campaign_id or campaign_id != entry.payload.get("campaign_id"):
        raise DashboardError(
            "campaign identity disagrees across the journal envelope, STOP_STATE, and result")
    return result


def _blocking_for(result: Mapping[str, Any]) -> list[dict]:
    state = str(result.get("state") or "UNKNOWN")
    error = result.get("error")
    conditions = []
    if state == "error":
        conditions.append({"kind": "CAMPAIGN_ERROR", "origin": "controller_stop",
                           "detail": str(error or "campaign stopped with an error")})
    elif state == "preflight_refused":
        preflight = result.get("preflight")
        reasons = preflight.get("reasons") if isinstance(preflight, Mapping) else None
        conditions.append({
            "kind": "PREFLIGHT_REFUSED", "origin": "controller_stop",
            "detail": "; ".join(map(str, reasons or ())) or "campaign preflight refused",
        })
    elif state == "t0_failed":
        conditions.append({"kind": "T0_FAILED", "origin": "evaluator_coverage",
                           "detail": "candidate failed correctness/build before speed ranking"})
    return conditions


def build_terminal_contract(entry: journal.JournalEntry, *,
                            exported_at: Optional[str] = None) -> dict:
    """Build a valid v2 contract from one terminal, already-journaled result."""
    result = _result_from(entry)
    spec = result.get("spec")
    if not isinstance(spec, Mapping):
        raise DashboardError("campaign result.spec must be a mapping")
    campaign_id = str(result["campaign_id"])
    state = str(result.get("state") or "unknown")
    backend = str(spec.get("backend") or "unknown")
    decision = result.get("decision")
    decision = decision if isinstance(decision, Mapping) else {}
    standing = ("keep" if decision.get("keep") is True else
                "revert" if decision.get("keep") is False else "not_ranked")
    as_of = entry.written_at
    conditions = _blocking_for(result)

    sections = {
        schemas.DASHBOARD_SECTION_CAMPAIGN: {
            "status": schemas.SECTION_OBSERVED, "as_of": as_of,
            "campaign_id": campaign_id, "state": state, "seq": entry.seq,
            "stopped": True, "candidate_id": result.get("candidate_id"),
            "executed": bool(result.get("executed")),
        },
        schemas.DASHBOARD_SECTION_CHAMPION: _unreported(
            "the current one-candidate driver banks a result; it does not mint a champion"),
        schemas.DASHBOARD_SECTION_BACKEND_STANDING: {
            "status": schemas.SECTION_OBSERVED, "as_of": as_of,
            "backends": {backend: {
                "standing": standing, "candidate_id": result.get("candidate_id"),
                "recipe_id": spec.get("recipe_id"), "state": state,
                "decision_reason": decision.get("reason"),
            }},
        },
        schemas.DASHBOARD_SECTION_HEADROOM: _unreported(
            "the terminal campaign record carries no storage or budget headroom observation"),
        schemas.DASHBOARD_SECTION_BLOCKING: {
            "status": schemas.SECTION_OBSERVED, "as_of": as_of, "open": conditions,
        },
        schemas.DASHBOARD_SECTION_CLAIMS: {
            "status": schemas.SECTION_OBSERVED, "as_of": as_of,
            "held": [], "released": result.get("releases") or [],
        },
        schemas.DASHBOARD_SECTION_RELEASE_PACKAGE: _unreported(
            "the current campaign driver produces a banked result, not a release package"),
    }
    produced_at = schemas.dashboard_liveness_timestamp(sections)
    unreported = schemas.dashboard_unreported_sections(sections)
    document = {
        "schema": schemas.SCHEMA_KERNEL_DASHBOARD_V2,
        "contract_version": 2,
        "campaign_id": campaign_id,
        "produced_at": produced_at,
        "generated_at": produced_at,
        "exported_at": exported_at or datetime.now(timezone.utc).isoformat(),
        "producer": {
            "module_id": MODULE_ID,
            "run": {"campaign_id": campaign_id, "controller_seq": entry.seq,
                    "controller_state": state, "ledger_receipt": entry.event_id},
        },
        "sections": sections,
        "degraded": bool(unreported),
        "unreported_sections": unreported,
        "observation_notice": OBSERVATION_NOTICE,
    }
    violations = schemas.validate_kernel_dashboard_v2(document)
    if violations:
        raise DashboardError("terminal dashboard contract is invalid: " + "; ".join(violations))
    return document


def _assert_destination(path: str | os.PathLike) -> Path:
    raw = storage.assert_not_scratch(path, what="AutoKernel dashboard export")
    resolved = Path(raw).resolve(strict=False)
    for tree in storage.PRODUCTION_TREES:
        production = Path(tree).resolve(strict=False)
        if resolved == production or production in resolved.parents:
            raise DashboardError(f"dashboard export may not enter frozen tree {production}")
    repo = Path(storage.REPO_ROOT).resolve(strict=False)
    if resolved == repo or repo in resolved.parents:
        raise DashboardError("dashboard export is a derived view and may not enter the checkout")
    if resolved.suffix != ".json":
        raise DashboardError("dashboard export path must end in .json")
    return resolved


def export_terminal_entry(entry: journal.JournalEntry, *,
                          path: str | os.PathLike = DEFAULT_EXPORT_PATH,
                          exported_at: Optional[str] = None) -> Path:
    """Atomically export one terminal entry; returning means rename + dir fsync."""
    target = _assert_destination(path)
    document = build_terminal_contract(entry, exported_at=exported_at)
    data = schemas.canonical_bytes(document) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise DashboardError(f"short dashboard write: {written} of {len(data)} bytes")
        os.fsync(fd)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    os.replace(temp, target)
    dir_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return target


__all__ = [
    "MODULE_ID", "DEFAULT_EXPORT_PATH", "OBSERVATION_NOTICE", "DashboardError",
    "build_terminal_contract", "export_terminal_entry",
]
