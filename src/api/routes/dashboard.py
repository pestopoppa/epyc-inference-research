"""Interactive orchestrator dashboard — server topology, active tasks, live streams.

Routes:
    GET  /dashboard                         — single-page HTML UI
    GET  /dashboard/api/topology            — static topology (roles, ports, services)
    GET  /dashboard/api/snapshot            — current state of all slots + counters
    GET  /dashboard/events/stream           — 1Hz SSE: snapshot + recent decisions
    GET  /dashboard/api/task/{task_id}      — full task detail (prompt + REPL history)
    GET  /dashboard/events/task/{task_id}   — 5Hz SSE: live token stream for one task
    POST /dashboard/api/autopilot_control   — operator pause/resume latch

Mostly read-only observer. Polls existing llama-server /slots endpoints and tails
the progress JSONL log; the only mutating endpoint is the explicit operator
pause/resume latch for AutoPilot dispatch.

SSH access: from your laptop, `ssh -L 8000:localhost:8000 daniele@<host>` then
open http://localhost:8000/dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import sqlite3
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from src.roles import Role
from src.runtime.live_telemetry import (
    live_batch_activity_summary,
    live_telemetry_frame,
    live_telemetry_sequence,
)
from src.api.routes.dashboard_snapshot import (
    INFLIGHT_MAX_AGE_DEFAULT_S,
    count_log_events as _count_log_events_impl,
    scan_orchestrator_tasks as _scan_orchestrator_tasks_impl,
    scan_recent_decisions as _scan_recent_decisions_impl,
    todays_progress_log as _todays_progress_log_impl,
)
from src.api.routes.dashboard_freshness import (
    Source as _FreshnessSource,
    envelope as _freshness_envelope,
    value_consistency as _value_consistency,
)
from src.api.routes.dashboard_panels import (
    PANELS as _PANELS,
    PANELS_BY_KEY as _PANELS_BY_KEY,
    _latest_tap_events_mtime,
)
from src.api.routes.dashboard_tap import (
    _INFERENCE_TAP_EVENTS_PATH,
    _INFERENCE_TAP_PATH,
    _REPL_TAP_PATH,
    _TAP_SENTINEL_PATH,
    _parse_inference_sections,
    _parse_structured_tap_requests,
    _parse_trial_state,
    _read_tail,
    _read_tap_events_tail,
)
from src.api.routes.dashboard_tasks import (
    _find_section_by_objective,
    _find_structured_request_by_id,
    _find_structured_request_by_task_id,
    _image_reference_for_task,
    _objective_for_task,
    _task_events,
    _task_text_snapshot,
    _terminal_answer_for_task,
)
from src.api.routes.dashboard_topology import (
    active_stack_numa_mode,
    active_stack_numa_mode_resolution,
    role_aliases,
    _clean_model_name,
    _discover_llama_models,
    _discover_llama_ports,
    _discover_llama_processes,
    _port_hint,
    _process_info_by_match,
    _role_color,
    base_role,
    expected_stack_services,
    manifest_declared_substrate,
)
from src.api.routes.dashboard_topology import _load_state_services as _load_state_services_impl
from src.autopilot_core.action_identity import (
    EPHEMERAL_ACTION_KEYS,
    config_fingerprint_from_row as core_config_fingerprint_from_row,
)
from src.autopilot_core.journal_reconstruction import (
    fold_supersession_events as core_fold_supersession_events,
    latest_journal_run_rows as core_latest_journal_run_rows,
    objectives_from_journal_row as core_objectives_from_journal_row,
    parse_journal_ts as core_parse_journal_ts,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.instrument_era_guard import (
    _parse_epoch as core_parse_era_epoch,
    instrument_eras_path,
)
from src.autopilot_core.learning_exclusions import (
    WITHIN_NOISE_EXCLUSIONS,
)
from src.autopilot_core.pareto_math import (
    dominates as core_pareto_dominates,
    hypervolume as _pareto_hypervolume_impl,
)
from src.autopilot_core.tier_specs import (
    DEFAULT_FRONTIER_TIER,
    LEGACY_OBJECTIVE_POLICY,
    PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
    RATE_4D_OBJECTIVE_POLICY,
    RESOURCE_LANES_V2_RATE_4D_OBJECTIVE_POLICY,
    TASK_RATE_OBJECTIVE_POLICY,
    quality_from_row,
    seq_task_rate_qph_from_row,
)

# The replay tool owns the canonical counterfactual archive pass and the
# tokens-per-solved derivation. Reuse it for the historical policy comparison
# and companion telemetry rather than reimplementing either calculation.
from scripts.analysis.task_rate_goodput_replay import (
    _archive as _task_rate_replay_archive,
    _frontier_for as _task_rate_replay_frontier,
    _tokens_per_solved_task as _task_rate_tokens_per_solved,
)
from scripts.autopilot.phase_status import (
    build_phase_health_report,
)
from scripts.autopilot.autopilot_restart_advisor import (
    build_restart_advice as _build_autopilot_restart_advice,
)
from scripts.autopilot.state_lock import (
    OPERATOR_PAUSE_OWNER,
    clear_pause_lease,
    state_write_lock,
    supersede_pause_lease,
)
from scripts.autopilot.state_ownership import clear_halt_latch, halt_latch_message

_REPO_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)
router = APIRouter()

ORCHESTRATOR_LOG_DIR = _REPO_ROOT / "logs"
PROGRESS_LOG_DIR = ORCHESTRATOR_LOG_DIR / "progress"
AUTOPILOT_LOG = ORCHESTRATOR_LOG_DIR / "autopilot.log"
AUTOPILOT_PHASE_PATH = Path("/mnt/raid0/llm/tmp/autopilot_phase.json")
ORCHESTRATOR_STATE_PATH = ORCHESTRATOR_LOG_DIR / "orchestrator_state.json"
REPO_READINESS_DIR = Path("/mnt/raid0/llm/epyc-root/data/repo_readiness")
REPO_READINESS_PROGRESS_DIR = Path("/mnt/raid0/llm/epyc-root/progress/2026-06")


def _load_state_services() -> list[dict[str, Any]]:
    """Wrapper supplying ORCHESTRATOR_STATE_PATH to dashboard_topology helper."""
    return _load_state_services_impl(ORCHESTRATOR_STATE_PATH)


def _stamp(payload: dict[str, Any], key: str, *, now: float | None = None) -> dict[str, Any]:
    """Attach a uniform ``_freshness`` envelope to a panel payload, in place.

    Additive and non-breaking — existing keys are untouched, so current frontend
    code keeps working while new code reads ``payload["_freshness"]``. Sources
    and thresholds come from the central ``dashboard_panels`` registry keyed by
    ``key``; an unregistered key (guarded against by ``test_dashboard_panels``)
    degrades to a minimal live envelope rather than crashing the endpoint.
    """
    now = time.time() if now is None else now
    spec = _PANELS_BY_KEY.get(key)
    sources = spec.live_sources() if spec is not None else []
    payload["_freshness"] = _freshness_envelope(
        sources, now=now, generated_at=payload.get("generated_at", now)
    )
    return payload


def _canonical_role_name(role: Any) -> str:
    raw = str(role or "unknown").split(":", 1)[0]
    return str(Role.from_string(raw) or raw)


def _read_autopilot_phase() -> dict[str, Any]:
    try:
        data = json.loads(AUTOPILOT_PHASE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _autopilot_phase_health() -> dict[str, Any]:
    try:
        return build_phase_health_report(path=AUTOPILOT_PHASE_PATH)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "unavailable",
            "path": str(AUTOPILOT_PHASE_PATH),
            "blockers": [f"phase health unavailable: {exc}"],
        }


def _port_role_shape(role_label: Any, port: int | None = None) -> tuple[str, str]:
    """Return the physical topology role and shape serving ``port``.

    Discovery labels are logical names and still contain retired ``.qN``
    suffixes.  NUMA_CONFIG owns physical placement, including auxiliary lanes
    such as unsuffixed ``eval_batch_frontdoor`` on a real half-machine shape.
    """
    label = str(role_label or "")
    if not label:
        return "", ""
    if port is not None:
        try:
            from src.runtime.instance_topology import (
                get_instance_regions,
                topology_instance_for_port,
            )

            owner = topology_instance_for_port(int(port))
            if owner is not None:
                role, idx = owner
                regions = get_instance_regions().get((role, idx), frozenset())
                if regions:
                    return base_role(role), _shape_for_regions(regions)
        except Exception:  # noqa: BLE001 — topology enrichment must fail open
            pass
    raw = label.split(":", 1)[0]
    match = re.match(r"^(.+?)\.(q\d+|half\d+|full)$", raw)
    if match:
        return base_role(match.group(1)), match.group(2)
    match = re.match(r"^(.+?)_(\d+)$", raw)
    if match and match.group(1) == "embedder":
        return base_role(match.group(1)), f"_{match.group(2)}"
    return base_role(raw), ""


def _alias_to_topology_roles(port_roles: dict[int, str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for role_label in set(port_roles.values()):
        role, _shape = _port_role_shape(role_label)
        if not role:
            continue
        try:
            for alias in role_aliases(role):
                alias_base = base_role(str(alias))
                if alias_base:
                    aliases[alias_base] = role
        except Exception:
            continue
    return aliases


def _instance_for_tap_request(
    role_info: dict[str, Any] | None,
    *,
    instance_idx: Any = None,
    instance_shape: Any = None,
) -> dict[str, Any] | None:
    instances = role_info.get("instances") if isinstance(role_info, dict) else None
    if not isinstance(instances, list) or not instances:
        return None
    try:
        idx = int(instance_idx) if instance_idx not in (None, "") else None
    except (TypeError, ValueError):
        idx = None
    if idx is not None:
        for inst in instances:
            try:
                if int(inst.get("idx")) == idx:
                    return inst
            except (AttributeError, TypeError, ValueError):
                continue
    shape = str(instance_shape or "")
    if shape:
        for inst in instances:
            if str(inst.get("shape") or "") == shape:
                return inst
    return instances[0] if isinstance(instances[0], dict) else None


def _enrich_structured_tap_requests(
    requests: list[dict[str, Any]],
    *,
    port_roles: dict[int, str] | None = None,
    region_locks: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fill missing topology/lock metadata for dashboard display.

    Older tap events can carry only the logical role and backend port. The live
    tap panel can still show them, but the CPU-region grid needs the physical
    lock role, instance shape, and regions to reconcile with /proc locks.
    """
    if not requests:
        return requests
    # Fail open: the enrichment pulls from the locks/topology domain, and the
    # tap panel's CONTENT must keep rendering even when that domain is broken —
    # tap requests without lock metadata beat an empty tap panel.
    try:
        port_roles = port_roles if port_roles is not None else _port_roles_cached()
        region_locks = region_locks if region_locks is not None else _region_locks_cached()
        by_role = region_locks.get("by_role") if isinstance(region_locks, dict) else {}
        by_role = by_role if isinstance(by_role, dict) else {}
        alias_to_topology = _alias_to_topology_roles(port_roles)
        enriched: list[dict[str, Any]] = []
        for req in requests:
            out = dict(req)
            try:
                port = int(out.get("port")) if out.get("port") not in (None, "") else None
            except (TypeError, ValueError):
                port = None
            port_role, port_shape = (
                _port_role_shape(port_roles.get(port, ""), port) if port is not None else ("", "")
            )
            logical_role = base_role(str(out.get("role") or ""))
            topology_role = (
                str(out.get("topology_role") or "")
                or alias_to_topology.get(logical_role, "")
                or port_role
                or logical_role
            )
            lock_role = str(out.get("lock_role") or "") or topology_role
            if topology_role and not out.get("topology_role"):
                out["topology_role"] = topology_role
            if lock_role and not out.get("lock_role"):
                out["lock_role"] = lock_role

            role_info = by_role.get(lock_role) or by_role.get(topology_role)
            inst = _instance_for_tap_request(
                role_info,
                instance_idx=out.get("instance_idx"),
                instance_shape=out.get("instance_shape") or port_shape,
            )
            if inst is not None:
                if out.get("instance_idx") in (None, "") and inst.get("idx") is not None:
                    out["instance_idx"] = inst.get("idx")
                if not out.get("instance_shape") and inst.get("shape"):
                    out["instance_shape"] = inst.get("shape")
                if not out.get("instance_regions") and inst.get("regions"):
                    out["instance_regions"] = list(inst.get("regions") or [])
            enriched.append(out)
        return enriched
    except Exception:
        return requests


def _latest_matching_file(root: Path, pattern: str) -> Path | None:
    try:
        candidates = [path for path in root.glob(pattern) if path.is_file()]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _repo_readiness_summary(
    *,
    data_dir: Path | None = None,
    progress_dir: Path | None = None,
    top_n: int = 12,
) -> dict[str, Any]:
    data_dir = data_dir or REPO_READINESS_DIR
    progress_dir = progress_dir or REPO_READINESS_PROGRESS_DIR
    report_path = _latest_matching_file(data_dir, "repo_readiness_[0-9]*.json")
    queue_path = _latest_matching_file(data_dir, "repo_readiness_remediation_queue_*.json")
    report = _load_json_file(report_path)
    queue = _load_json_file(queue_path)
    queue_items = [item for item in queue.get("items", []) if isinstance(item, dict)]
    priority_counts: dict[str, int] = {}
    for item in queue_items:
        priority = str(item.get("priority") or "unknown")
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    top_items = sorted(
        queue_items,
        key=lambda item: (
            str(item.get("priority") or "P9"),
            str(item.get("repo") or ""),
            str(item.get("criterion_id") or ""),
        ),
    )[:top_n]
    markdown_path = _latest_matching_file(
        progress_dir,
        "repo-readiness-remediation-*.md",
    )
    repos = report.get("repos") if isinstance(report.get("repos"), dict) else {}
    return {
        "available": bool(report or queue),
        "authority": "advisory",
        "autopilot_gate": False,
        "report_path": str(report_path) if report_path else None,
        "queue_path": str(queue_path) if queue_path else None,
        "markdown_path": str(markdown_path) if markdown_path else None,
        "generated_at": queue.get("generated_at") or report.get("generated_at"),
        "portfolio_level": (report.get("portfolio") or {}).get("maturity")
        if isinstance(report.get("portfolio"), dict)
        else None,
        "repo_levels": {
            repo: data.get("maturity")
            for repo, data in sorted(repos.items())
            if isinstance(data, dict)
        },
        "queue_version": queue.get("version"),
        "item_count": queue.get("item_count", len(queue_items) if queue_items else 0),
        "priority_counts": priority_counts,
        "top_items": top_items,
    }


def _structured_tap_active(structured_requests: list[dict[str, Any]]) -> bool:
    """True only when structured tap evidence shows current live inference.

    The sentinel file can outlive the request that created it, and tailed
    structured records intentionally retain quiet/stalled history for diagnosis.
    Treat only non-quiet running records as active so the dashboard does not
    present old planner/eval requests as live work after AutoPilot stops.
    """
    for req in structured_requests:
        if str(req.get("status") or "").lower() != "running":
            continue
        quiet_s = req.get("quiet_s")
        try:
            if quiet_s is not None and float(quiet_s) >= 15.0:
                continue
        except (TypeError, ValueError):
            pass
        return True
    return False


# Widen the tail read only for lock-holder recovery (below). The live panel
# parses a fixed ~1 MB tail; under autopilot-eval load inference_tap_events.jsonl
# rotates at 512 MB every day or two, so 1 MB spans only seconds. A request that
# holds a CPU-region lock but is briefly between visible tap events (a long
# prompt-prefill emitting no chunks yet, while concurrent cross-role chunk
# traffic pushes its `start` event past the tail) is genuinely live yet absent
# from the parsed set, and the dashboard mis-renders it as an "off-tap holder".
_OFFWINDOW_RECOVERY_BYTES = 8 * 1024 * 1024


def _tap_request_role_keys(req: dict[str, Any]) -> set[str]:
    """Role identifiers a parsed tap request can match a lock holder on.

    Mirrors the client's `structuredTapMatchesLockHolder`: a holder role matches
    if it equals the request's lock_role, topology_role, role, or base_role(role).
    """
    keys: set[str] = set()
    for key in (req.get("lock_role"), req.get("topology_role"), req.get("role")):
        if key:
            keys.add(str(key))
    role = req.get("role")
    if role:
        keys.add(base_role(str(role)))
    return keys


def _lock_holder_identities(
    region_locks: dict[str, Any] | None,
) -> list[tuple[str, int, tuple[str, ...]]]:
    """(role, instance_idx, holder_pids) for every instance currently holding a
    CPU-region lock, read from the /proc-derived region_locks payload.

    Region locks are fcntl.flock locks surfaced via /proc/locks (see
    src.runtime.cpu_region_lock), so the kernel drops them the instant the owning
    process exits — a holder here is ALWAYS a live dispatch, never an orphan.
    """
    by_role = region_locks.get("by_role") if isinstance(region_locks, dict) else None
    if not isinstance(by_role, dict):
        return []
    out: list[tuple[str, int, tuple[str, ...]]] = []
    for role, bucket in by_role.items():
        if not isinstance(bucket, dict):
            continue
        pids_by_idx: dict[int, set[str]] = {}
        for region in bucket.get("regions", []) or []:
            if not isinstance(region, dict) or not region.get("held"):
                continue
            pids = [str(p) for p in (region.get("holder_pids") or [])]
            for idx in region.get("holder_instance_idxs") or []:
                if str(idx).lstrip("-").isdigit():
                    pids_by_idx.setdefault(int(idx), set()).update(pids)
        for idx, pids in pids_by_idx.items():
            out.append((str(role), idx, tuple(sorted(pids))))
    return out


def _recover_offwindow_lock_holder_requests(
    parsed_requests: list[dict[str, Any]],
    region_locks: dict[str, Any] | None,
    *,
    max_requests: int,
    now_epoch: float,
) -> list[dict[str, Any]]:
    """Fold genuinely-running lock holders that aged out of the live tail back in.

    Matches the client's holder↔request rule (role AND (instance_idx OR pid)):
    for any active region-lock holder with no matching parsed request, read a
    wider reverse window (`_OFFWINDOW_RECOVERY_BYTES`) once and merge its newest
    still-running request. This is what stops the false "off-tap CPU-region
    holder" cards for autopilot eval / numeric_trial traffic — which IS tapped,
    but whose `start` event scrolls out of the 1 MB window during a long prefill
    (tap events stamp os.getpid(), the same process that holds the flock, so the
    holder pid and the tap pid are the same worker).

    Bounded on purpose: the wider read runs only when a holder is otherwise
    invisible (rare — a streaming holder stays in the 1 MB window via its chunk
    events), so the common path pays nothing beyond a couple of set builds.
    """
    holders = _lock_holder_identities(region_locks)
    if not holders:
        return parsed_requests

    represented: set[tuple[str, int]] = set()
    represented_role_pids: set[tuple[str, str]] = set()
    for req in parsed_requests:
        pid = req.get("pid")
        role_keys = _tap_request_role_keys(req)
        if pid not in (None, ""):
            represented_role_pids.update((role_key, str(pid)) for role_key in role_keys)
        idx_raw = req.get("instance_idx")
        if not str(idx_raw if idx_raw is not None else "").lstrip("-").isdigit():
            continue
        idx = int(idx_raw)
        for role_key in role_keys:
            represented.add((role_key, idx))

    unmatched = {
        (role, idx)
        for role, idx, pids in holders
        if (role, idx) not in represented
        and not any((role, str(pid)) in represented_role_pids for pid in pids)
    }
    if not unmatched:
        return parsed_requests

    wide_tail = _read_tap_events_tail(
        _INFERENCE_TAP_EVENTS_PATH, max_bytes=_OFFWINDOW_RECOVERY_BYTES
    )
    if not wide_tail:
        return parsed_requests

    seen_ids = {str(r.get("request_id") or "") for r in parsed_requests}
    recovered: list[dict[str, Any]] = []
    # _parse_structured_tap_requests returns most-recent-updated first, so the
    # first running match for a holder is its newest request.
    for req in _parse_structured_tap_requests(
        wide_tail, max_requests=max_requests, now_epoch=now_epoch
    ):
        if not unmatched:
            break
        if req.get("status") == "complete":
            continue
        rid = str(req.get("request_id") or "")
        if rid in seen_ids:
            continue
        idx_raw = req.get("instance_idx")
        if not str(idx_raw if idx_raw is not None else "").lstrip("-").isdigit():
            continue
        idx = int(idx_raw)
        matched = next(
            (
                (role_key, idx)
                for role_key in _tap_request_role_keys(req)
                if (role_key, idx) in unmatched
            ),
            None,
        )
        if matched is None:
            continue
        recovered.append(req)
        seen_ids.add(rid)
        unmatched.discard(matched)
    return recovered + parsed_requests if recovered else parsed_requests


def _structured_tap_requests_for_dashboard(
    *,
    max_requests: int,
    now_epoch: float | None = None,
    region_locks: dict[str, Any] | None = None,
    port_roles: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Read, parse, and enrich structured tap rows for live dashboard panels."""
    structured_tail = _read_tap_events_tail(_INFERENCE_TAP_EVENTS_PATH, max_bytes=1024 * 1024)
    now = time.time() if now_epoch is None else now_epoch
    structured_requests = _parse_structured_tap_requests(
        structured_tail,
        max_requests=max_requests,
        now_epoch=now,
    )
    # Resolve the region-lock frame once, then reconcile it against the parsed
    # window: recover live holders whose tap events aged out of the fixed tail
    # (else they surface as false "off-tap holder" cards) before enriching with
    # lock metadata off that same frame.
    resolved_locks = region_locks if region_locks is not None else _region_locks_cached()
    structured_requests = _recover_offwindow_lock_holder_requests(
        structured_requests,
        resolved_locks,
        max_requests=max_requests,
        now_epoch=now,
    )
    enriched = _enrich_structured_tap_requests(
        structured_requests,
        port_roles=port_roles,
        region_locks=resolved_locks,
    )
    # Live prefill/decode progress from the last /slots sample (RTG-47): the
    # tap is silent through a long prefill, but the slot knows how many prompt
    # tokens it has ingested. Attach-only-from-fresh-evidence; see the helper.
    return _attach_slot_progress(enriched, now=now)


# ---------------------------------------------------------------------------
# Inference taps — live prompt + response stream written by autopilot's
# seeding harness to /mnt/raid0/llm/tmp/*. These are the same files the
# autopilot --tui tails. Reading them is cheap (just stat + seek-to-end)
# and gives us the full token stream + REPL history for free.
# ---------------------------------------------------------------------------


@router.get("/dashboard/api/inference_tap")
async def inference_tap_snapshot(max_sections: int = 20) -> JSONResponse:
    """Return parsed inference sections + current prompt + REPL tail.

    Source: /mnt/raid0/llm/tmp/{inference_tap.log, autopilot_prompt_tap.txt,
    repl_tap.log} — the same files autopilot_tui.py tails.
    """
    inference_tail = _read_tail(_INFERENCE_TAP_PATH, max_bytes=512 * 1024)
    sections = _parse_inference_sections(inference_tail, max_sections=max_sections)
    now_epoch = time.time()
    structured_requests = _structured_tap_requests_for_dashboard(
        max_requests=max_sections,
        now_epoch=now_epoch,
    )
    tap_active = _structured_tap_active(structured_requests)
    repl_tail = _read_tail(_REPL_TAP_PATH, max_bytes=64 * 1024)
    # Just take the last ~3000 chars of REPL for compactness
    repl_tail = repl_tail[-3000:] if repl_tail else ""

    # File mtimes — surface staleness
    def mtime(p: Path) -> float | None:
        try:
            return p.stat().st_mtime
        except Exception:
            return None

    return JSONResponse(
        _stamp(
            {
                "tap_active": tap_active,
                "tap_sentinel_active": _TAP_SENTINEL_PATH.exists(),
                "inference_sections": sections,
                "structured_requests": structured_requests,
                "inference_tap_mtime": mtime(_INFERENCE_TAP_PATH),
                # Shard-aware: right after a 512MB rotation the base file is missing
                # until the next append; the newest shard's mtime is the truth.
                "structured_tap_mtime": _latest_tap_events_mtime(),
                "repl_tail": repl_tail,
                "repl_tap_mtime": mtime(_REPL_TAP_PATH),
                "now": now_epoch,
            },
            "inference_tap",
            now=now_epoch,
        )
    )


@router.get("/dashboard/api/contention")
async def contention_gate_snapshot(request: Request) -> JSONResponse:
    """Cross-role admission-gate metrics + per-role scheduling state.

    2026-05-24 cross-role-bw-aware-routing. Returns:
      - matrix_status: "ok"|"missing"|"stale"|"invalid"
      - active_decodes_by_role: {role: count} from region-lock holders
      - contention_blocked_count: {"roleA+roleB": int}
      - contention_wait_seconds: cumulative
      - contention_timeout_count, contention_admitted_count, etc.
      - per_role_scheduling: {role: {quarter_preference_order, migrations_started,
            migration_failures, kv_migration: {enabled, dispatch_path, ...}}}
          for every role backed by ConcurrencyAwareBackend
      - generated_at: time.time() for client cache-busting
    """
    import time as _time

    try:
        from src.scheduling.contention_gate import get_gate

        snap = get_gate().metrics_snapshot()
    except Exception as exc:  # noqa: BLE001
        snap = {"error": str(exc), "matrix_status": "unavailable"}
    try:
        from src.metrics import migration_counters

        snap["migration_counters"] = migration_counters.snapshot()
    except Exception as exc:  # noqa: BLE001
        snap["migration_counters"] = {"error": str(exc)}

    # Per-role scheduling state (quarter preference + migration counts).
    # Fetched here because it lives on app.state.llm_primitives._backends,
    # which is per-request-injectable but not module-singleton.
    per_role: dict[str, Any] = {}
    try:
        # The ConcurrencyAwareBackend (which holds migration counters / placement state) lives on
        # the REAL-inference primitives (state._real_primitives, lazily built by _init_primitives
        # with the full:-prefixed server_urls), NOT the startup state.llm_primitives (constructed
        # without server_urls → no backends). Prefer the real one so per_role_scheduling actually
        # reflects live CAB placement/migrations (J2/J3 observability fix, 2026-05-27).
        primitives = getattr(request.app.state, "_real_primitives", None) or getattr(
            request.app.state, "llm_primitives", None
        )
        if primitives is None:
            from src.api.state import get_state

            api_state = get_state()
            primitives = getattr(api_state, "_real_primitives", None) or getattr(
                api_state, "llm_primitives", None
            )
        if primitives is not None:
            for role, backend in getattr(primitives, "_backends", {}).items():
                if not hasattr(backend, "_quarter_preference_order"):
                    continue
                per_role[role] = {
                    "quarter_preference_order": list(
                        getattr(backend, "_quarter_preference_order", [])
                    ),
                    "migrations_started": int(getattr(backend, "_migrations", 0)),
                    "migration_failures": int(getattr(backend, "_migration_failures", 0)),
                    # WP-4 reverse migration (quarter→full on load drop) — surfaced for J2/J3
                    # observation (forward = migrations_started above; this is the reverse leg).
                    "reverse_migrations": int(
                        sum((getattr(backend, "_reverse_migration_counts", {}) or {}).values())
                    ),
                    "kv_migration": (
                        backend.kv_migration_status()
                        if hasattr(backend, "kv_migration_status")
                        else {}
                    ),
                }
    except Exception as exc:  # noqa: BLE001
        per_role = {"_error": str(exc)}
    snap["per_role_scheduling"] = per_role
    snap["generated_at"] = _time.time()
    return JSONResponse(_stamp(snap, "contention", now=snap["generated_at"]))


# ── region_locks helpers (module-scope so tests can import) ─────────────


def _shape_for_regions(regs: "frozenset[str] | set[str] | list[str]") -> str:
    """Canonical column-bucket name for an instance covering `regs`.

    Returns one of: "full" (all 4 quarters), "half0" (q0+q1), "half1"
    (q2+q3), "q0".."q3" (single quarter), or "+".join(sorted(regs)) as
    a fallback for exotic shapes not yet in NUMA_CONFIG.
    """
    rs = set(regs)
    if rs == {"q0", "q1", "q2", "q3"}:
        return "full"
    if rs == {"q0", "q1"}:
        return "half0"
    if rs == {"q2", "q3"}:
        return "half1"
    if len(rs) == 1:
        return next(iter(rs))
    return "+".join(sorted(rs))


def _panel_shapes_from_matrix(sr, primary_shape: str) -> set[str]:
    """Within-role shapes the contention matrix marks CO-PLACEABLE for a role.

    NOTE: as of the 2026-07-10 region-locks-grid fix this no longer gates the
    panel's visible shapes — the grid shows every configured instance so live
    heavy-role quarters (no co-placement pairs) still appear. Retained as the
    canonical reader of `same_role` co-placement semantics for future use.

    Strict interpretation:
      - sr is None                       → empty set (role not in matrix → hide)
      - sr has no instance_pairs         → {primary_shape} (single-instance or
                                            'primary only' roles like ingest)
      - sr has instance_pairs            → union of all a/b labels, with the
                                            matrix's "full" alias translated to
                                            the role's primary shape

    `primary_shape` is the canonical shape of the role's idx=0 NUMA_CONFIG
    instance ("full", "half0", "half1", or "qN") — i.e. what the matrix means
    when it says "full" for that specific role.
    """
    if sr is None:
        return set()
    if not sr.instance_pairs:
        return {primary_shape}
    out: set[str] = set()
    for ip in sr.instance_pairs:
        for label in (ip.a, ip.b):
            out.add(primary_shape if label == "full" else label)
    return out


def _resolve_pid_to_instance_idx(
    role_pid_regions: dict[str, dict[str, set[str]]],
    role_instances_by_regions: dict[str, dict["frozenset[str]", int]],
) -> dict[tuple[str, str], int]:
    """Map each (role, pid) to its NUMA_CONFIG instance idx by matching the
    set of regions the PID currently holds against the role's instance
    region-sets.

    Each NUMA_CONFIG instance has a unique region set within a role, so
    holding {q0,q1} unambiguously identifies the half0 instance, etc.
    Locks are acquired by orchestrator uvicorn workers (not llama-server
    processes), so this region-set-based lookup is the only reliable
    way to attribute holders to instances.
    """
    out: dict[tuple[str, str], int] = {}
    for role, pid_regs in role_pid_regions.items():
        region_map = role_instances_by_regions.get(role) or {}
        for pid, regs in pid_regs.items():
            idx = region_map.get(frozenset(regs))
            if idx is not None:
                out[(role, pid)] = idx
    return out


def _region_lock_blocked_by_roles(
    row_role: str,
    active_roles: set[str],
    matrix: Any | None,
) -> list[str]:
    """Roles whose active locks should render as wait/blocking for row_role.

    Same-role overlaps are always real lock waits because the lock files are
    keyed by (role, region). Cross-role overlaps follow the contention matrix:
    allow means concurrent dispatch is expected, queue/block means the row role
    would wait behind that holder for background/autopilot traffic.
    """
    blocked: set[str] = set()
    for holder_role in active_roles:
        if not holder_role:
            continue
        if holder_role == row_role:
            blocked.add(holder_role)
            continue
        if matrix is None:
            blocked.add(holder_role)
            continue
        try:
            from src.scheduling.contention import PairDecision, TrafficClass, pair_policy

            decision = pair_policy(
                row_role,
                holder_role,
                TrafficClass.BACKGROUND,
                matrix=matrix,
            )
        except Exception:
            blocked.add(holder_role)
            continue
        if decision != PairDecision.ALLOW:
            blocked.add(holder_role)
    return sorted(blocked)


def _occupancy_tokens(occupancy: dict[str, Any] | None) -> tuple[str, ...]:
    """Stable identity for one occupancy-ledger observation."""
    entries = occupancy.get("entries") if isinstance(occupancy, dict) else []
    if not isinstance(entries, list):
        return ()
    return tuple(
        sorted(
            str(entry.get("token") or "")
            for entry in entries
            if isinstance(entry, dict) and entry.get("token")
        )
    )


def _occupancy_instance_pairs(
    occupancy: dict[str, Any] | None,
    *,
    shared_only: bool = False,
) -> set[tuple[str, int]]:
    """Return exact ``(role, instance_idx)`` leases from the occupancy ledger."""
    entries = occupancy.get("entries") if isinstance(occupancy, dict) else []
    if not isinstance(entries, list):
        return set()
    pairs: set[tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or (shared_only and not entry.get("shared")):
            continue
        role = str(entry.get("role") or "")
        idx = entry.get("instance_idx")
        if role and str(idx if idx is not None else "").lstrip("-").isdigit():
            pairs.add((role, int(idx)))
    return pairs


def _proc_instance_pairs(by_role: dict[str, Any]) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for role, bucket in by_role.items():
        if not isinstance(bucket, dict):
            continue
        for idx in bucket.get("proc_active_instance_idxs", bucket.get("active_instance_idxs", [])):
            if str(idx).lstrip("-").isdigit():
                pairs.add((str(role), int(idx)))
    return pairs


def _lock_root_provenance(tmp_dir: Path) -> dict[str, Any]:
    """Describe whether region-lock telemetry is rooted in a test sandbox."""
    path = str(tmp_dir)
    normalized = path.replace("\\", "/").lower()
    suspicious_test_root = any(
        marker in normalized
        for marker in ("/pytest-of-", "/pytest-", "/.pytest_tmp/", "/pytest_tmp/")
    )
    if os.environ.get("ORCHESTRATOR_TMP_DIR"):
        source = "ORCHESTRATOR_TMP_DIR"
    elif os.environ.get("ORCHESTRATOR_PATHS_TMP_DIR"):
        source = "ORCHESTRATOR_PATHS_TMP_DIR"
    else:
        source = "default"
    issue = ""
    if suspicious_test_root:
        issue = (
            "region-lock root points into a pytest temporary directory; "
            "serving-process placement telemetry has invalid production provenance"
        )
    return {
        "path": path,
        "source": source,
        "suspicious_test_root": suspicious_test_root,
        "valid_for_production": not suspicious_test_root,
        "issue": issue,
    }


def _placement_telemetry_status(
    *,
    activity: dict[int, dict[str, Any]],
    port_roles: dict[int, str],
    region_locks: dict[str, Any],
) -> dict[str, Any]:
    """Compare live CPU `/slots` roles with the canonical placement lease.

    The sources are sampled sequentially, so disagreement is telemetry to
    surface, not permission to rewrite either fact. A stable disagreement is
    exactly the placement-vs-dashboard failure class operators need to see.
    """
    by_role = region_locks.get("by_role") if isinstance(region_locks, dict) else {}
    lock_roles = set(by_role) if isinstance(by_role, dict) else set()
    busy_roles: set[str] = set()
    busy_ports: dict[str, list[int]] = {}
    capacity_observations: list[dict[str, Any]] = []
    matrix = region_locks.get("display_matrix") if isinstance(region_locks, dict) else {}
    matrix_rows = matrix.get("rows") if isinstance(matrix, dict) else []
    configured_capacity: dict[tuple[str, str], int] = {}
    if isinstance(matrix_rows, list):
        for row in matrix_rows:
            if not isinstance(row, dict):
                continue
            row_role = _canonical_role_name(row.get("role"))
            cells = row.get("cells")
            if not row_role or not isinstance(cells, list):
                continue
            for cell in cells:
                if not isinstance(cell, dict) or cell.get("capacity") is None:
                    continue
                shape = str(cell.get("shape") or "")
                if shape:
                    configured_capacity[(row_role, shape)] = max(1, int(cell.get("capacity") or 1))
    for raw_port, slot_state in activity.items():
        if not isinstance(slot_state, dict):
            continue
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            continue
        role, shape = _port_role_shape(port_roles.get(port, ""), port)
        role = _canonical_role_name(role)
        if role and role in lock_roles and int(slot_state.get("n_active") or 0) > 0:
            busy_roles.add(role)
            busy_ports.setdefault(role, []).append(port)
        observed_capacity = max(0, int(slot_state.get("n_total") or 0))
        expected_capacity = configured_capacity.get((role, shape or "full"))
        if role and expected_capacity is not None and observed_capacity > 0:
            capacity_observations.append(
                {
                    "role": role,
                    "shape": shape or "full",
                    "port": port,
                    "configured": expected_capacity,
                    "observed": observed_capacity,
                    "matches": observed_capacity == expected_capacity,
                }
            )

    occupancy = region_locks.get("occupancy") if isinstance(region_locks, dict) else {}
    lease_roles = {role for role, _idx in _occupancy_instance_pairs(occupancy)}
    coherence = region_locks.get("coherence") if isinstance(region_locks, dict) else {}
    region_status = str((coherence or {}).get("status") or "unknown")
    lock_root = region_locks.get("lock_root") if isinstance(region_locks, dict) else {}
    provenance_invalid = bool(isinstance(lock_root, dict) and lock_root.get("suspicious_test_root"))
    capacity_conflicts = [item for item in capacity_observations if not item["matches"]]
    conflict = bool(busy_roles and lease_roles and busy_roles != lease_roles)
    if provenance_invalid:
        status = "invalid_provenance"
    elif conflict:
        status = "conflict"
    elif capacity_conflicts:
        status = "capacity_conflict"
    elif region_status == "transition":
        status = "transition"
    elif busy_roles == lease_roles or (not busy_roles and not lease_roles):
        status = "coherent"
    else:
        status = "uncorroborated"
    return {
        "status": status,
        "conflict": conflict,
        "provenance_invalid": provenance_invalid,
        "lock_root": lock_root if isinstance(lock_root, dict) else {},
        "capacity_observations": capacity_observations,
        "capacity_conflicts": capacity_conflicts,
        "busy_roles": sorted(busy_roles),
        "busy_ports": {role: sorted(ports) for role, ports in sorted(busy_ports.items())},
        "lease_roles": sorted(lease_roles),
        "region_lock_status": region_status,
        "observed_at": time.time(),
    }


def _placement_frame_signature(region_locks: dict[str, Any]) -> tuple[Any, ...]:
    """Identity of placement ownership at one side of a `/slots` sample."""
    occupancy = region_locks.get("occupancy") if isinstance(region_locks, dict) else {}
    entries = occupancy.get("entries") if isinstance(occupancy, dict) else []
    leases = tuple(
        sorted(
            (
                str(entry.get("token") or ""),
                str(entry.get("role") or ""),
                int(entry.get("instance_idx") or 0),
                int(entry.get("capacity") or 1),
                bool(entry.get("shared")),
            )
            for entry in (entries if isinstance(entries, list) else [])
            if isinstance(entry, dict)
        )
    )
    global_lock = region_locks.get("global_lock") if isinstance(region_locks, dict) else {}
    global_pids = tuple(sorted(str(pid) for pid in ((global_lock or {}).get("holder_pids") or [])))
    return leases, global_pids


async def _sample_slots_with_placement_window(
    active_mode: str,
    *,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Bracket `/slots` with placement reads and retry one torn window.

    The raw signals are never coerced. A stable bracket certifies that any
    lease-vs-backend count difference is a real lifecycle-stage difference;
    an unstable bracket remains explicitly torn after the bounded retry.
    """
    attempts = max(1, int(max_attempts))
    result: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        frame_started_at = time.time()
        placement_before = _region_locks_payload(active_mode)
        placement_before_at = time.time()
        slots_started_at = time.time()
        slots_by_port, slots_poll_meta = await _poll_all_slots()
        slots_completed_at = time.time()
        placement_after = _region_locks_payload(active_mode)
        placement_after_at = time.time()
        stable = _placement_frame_signature(placement_before) == _placement_frame_signature(
            placement_after
        )
        result = {
            "attempt": attempt,
            "max_attempts": attempts,
            "frame_started_at": frame_started_at,
            "placement_before": placement_before,
            "placement_before_at": placement_before_at,
            "slots_by_port": slots_by_port,
            "slots_poll_meta": slots_poll_meta,
            "slots_started_at": slots_started_at,
            "slots_completed_at": slots_completed_at,
            "placement_after": placement_after,
            "placement_after_at": placement_after_at,
            "placement_stable": stable,
        }
        # Retrying unavailable slots would turn an explicit unknown into delay,
        # not knowledge. Retry only a genuinely torn ownership window.
        if stable or attempt >= attempts:
            break
    return result


def _annotate_live_observability_frame(
    *,
    region_locks: dict[str, Any],
    activity: dict[int, dict[str, Any]],
    port_roles: dict[int, str],
    structured_requests: list[dict[str, Any]],
    placement_telemetry: dict[str, Any],
    sample: dict[str, Any],
    tap_observed_at: float,
    event_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach faithful lease/backend/tap lifecycle signals to one frame."""
    event_frame = dict(event_frame or {})
    frame_sequence = int(event_frame.get("frame_sequence") or 0)
    event_sequence = int(event_frame.get("event_sequence") or 0)
    frame_id = (
        f"live-p{os.getpid()}-f{frame_sequence}-e{event_sequence}"
        f"-a{int(sample.get('attempt') or 1)}"
    )
    meta = sample.get("slots_poll_meta") if isinstance(sample, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    unanswered_ports = {
        int(port) for port in (meta.get("unanswered_ports") or []) if str(port).isdigit()
    }
    slots_complete = bool(
        int(meta.get("ports") or 0) > 0
        and int(meta.get("answered") or 0) == int(meta.get("ports") or 0)
        and int(meta.get("timed_out") or 0) == 0
    )

    role_signals: dict[str, dict[str, Any]] = {}

    def role_bucket(role: str) -> dict[str, Any]:
        return role_signals.setdefault(
            role,
            {
                "role": role,
                "lock_domain": "cpu" if role in (region_locks.get("by_role") or {}) else "none",
                "placement": {"active": 0, "capacity": None, "tokens": [], "request_tags": []},
                "backend": {
                    "active": 0,
                    "capacity": 0,
                    "ports": [],
                    "task_ids": [],
                    "by_shape": {},
                },
                "tap": {
                    "live": 0,
                    "decoding": 0,
                    "quiet": 0,
                    "request_ids": [],
                    "by_shape": {},
                },
            },
        )

    occupancy = region_locks.get("occupancy") if isinstance(region_locks, dict) else {}
    occupancy_entries = occupancy.get("entries") if isinstance(occupancy, dict) else []
    for entry in occupancy_entries if isinstance(occupancy_entries, list) else []:
        if not isinstance(entry, dict):
            continue
        role = _canonical_role_name(entry.get("role"))
        if not role:
            continue
        placement = role_bucket(role)["placement"]
        placement["active"] += 1
        placement["capacity"] = max(
            int(placement.get("capacity") or 0), int(entry.get("capacity") or 1)
        )
        placement["tokens"].append(str(entry.get("token") or ""))
        request_tag = str(entry.get("request_tag") or "")
        if request_tag and request_tag not in placement["request_tags"]:
            placement["request_tags"].append(request_tag)

    for raw_port, state in activity.items():
        if not isinstance(state, dict):
            continue
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            continue
        role, shape = _port_role_shape(port_roles.get(port, ""), port)
        role = _canonical_role_name(role)
        if not role:
            continue
        backend = role_bucket(role)["backend"]
        backend["ports"].append(port)
        shape = shape or "full"
        shape_backend = backend["by_shape"].setdefault(
            shape, {"active": 0, "capacity": 0, "ports": [], "task_ids": []}
        )
        shape_backend["ports"].append(port)
        if port in unanswered_ports or int(state.get("n_total") or 0) <= 0:
            backend["unknown"] = True
            shape_backend["unknown"] = True
            continue
        observed_active = max(0, int(state.get("n_active") or 0))
        observed_capacity = max(0, int(state.get("n_total") or 0))
        task_ids = [
            slot.get("task_id")
            for slot in (state.get("active_slots") or [])
            if isinstance(slot, dict) and slot.get("task_id") is not None
        ]
        backend["active"] += observed_active
        backend["capacity"] += observed_capacity
        backend["task_ids"].extend(task_ids)
        shape_backend["active"] += observed_active
        shape_backend["capacity"] += observed_capacity
        shape_backend["task_ids"].extend(task_ids)

    causal_observations: list[dict[str, Any]] = []
    for request in structured_requests:
        if not isinstance(request, dict):
            continue
        role = _canonical_role_name(
            request.get("topology_role") or request.get("lock_role") or request.get("role")
        )
        status = str(request.get("status") or "").lower()
        is_live = bool(request.get("is_live")) or status in {"running", "quiet"}
        if request.get("request_id"):
            causal_observations.append(
                {
                    "request_id": str(request.get("request_id")),
                    "parent_request_id": request.get("parent_request_id"),
                    "role": role,
                    "status": status or "unknown",
                    "event_count": int(request.get("event_count") or 0),
                    "updated_at_epoch": request.get("updated_at_epoch"),
                }
            )
        if not role or not is_live:
            continue
        tap = role_bucket(role)["tap"]
        request_shape = str(request.get("instance_shape") or "full")
        shape_tap = tap["by_shape"].setdefault(
            request_shape, {"live": 0, "decoding": 0, "quiet": 0, "request_ids": []}
        )
        tap["live"] += 1
        shape_tap["live"] += 1
        if status == "running" and int(request.get("chunk_count") or 0) > 0:
            tap["decoding"] += 1
            shape_tap["decoding"] += 1
        elif status == "quiet":
            tap["quiet"] += 1
            shape_tap["quiet"] += 1
        request_id = str(request.get("request_id") or "")
        if request_id:
            tap["request_ids"].append(request_id)
            shape_tap["request_ids"].append(request_id)

    count_disagreements: list[dict[str, Any]] = []
    for role, signals in sorted(role_signals.items()):
        placement = signals["placement"]
        backend = signals["backend"]
        tap = signals["tap"]
        lease_active = int(placement.get("active") or 0)
        backend_active = int(backend.get("active") or 0)
        backend_unknown = bool(backend.get("unknown"))
        if (
            signals["lock_domain"] == "cpu"
            and not backend_unknown
            and lease_active != backend_active
        ):
            count_disagreements.append(
                {
                    "role": role,
                    "placement_active": lease_active,
                    "backend_active": backend_active,
                    "meaning": "lifecycle_stage_difference_or_torn_sample",
                }
            )
        if tap["live"] and lease_active == 0 and signals["lock_domain"] == "cpu":
            lifecycle = "waiting_for_placement"
        elif lease_active and backend_unknown:
            lifecycle = "lease_admitted_backend_unknown"
        elif lease_active > backend_active:
            lifecycle = "lease_admitted_backend_queued"
        elif backend_active and tap["decoding"]:
            lifecycle = "decoding_observed"
        elif backend_active:
            lifecycle = "backend_processing_tap_quiet"
        elif lease_active:
            lifecycle = "lease_admitted"
        elif tap["quiet"]:
            lifecycle = "post_model_or_quiet"
        elif tap["live"]:
            lifecycle = "routed_waiting_signal"
        else:
            lifecycle = "idle_or_released"
        signals["lifecycle"] = lifecycle
        signals["identity_join"] = "role/cohort; per-request backend join unavailable"
        signals["observed_at"] = {
            "placement": sample.get("placement_after_at"),
            "backend_window_start": sample.get("slots_started_at"),
            "backend_window_end": sample.get("slots_completed_at"),
            "tap": tap_observed_at,
        }

    event_backend_by_role: dict[str, int] = {}
    for request_state in event_frame.get("requests") or []:
        if not isinstance(request_state, dict) or request_state.get("terminal"):
            continue
        if request_state.get("stage") not in {"backend_dispatched", "first_output"}:
            continue
        event_role = _canonical_role_name(
            request_state.get("topology_role") or request_state.get("role")
        )
        if event_role:
            event_backend_by_role[event_role] = event_backend_by_role.get(event_role, 0) + 1
    watchdog_backend_by_role = {
        role: int(signals.get("backend", {}).get("active") or 0)
        for role, signals in role_signals.items()
        if not signals.get("backend", {}).get("unknown")
    }
    reconciliation_differences = [
        {
            "role": role,
            "event_backend_active": event_backend_by_role.get(role, 0),
            "watchdog_backend_active": watchdog_backend_by_role.get(role, 0),
        }
        for role in sorted(set(event_backend_by_role) | set(watchdog_backend_by_role))
        if event_backend_by_role.get(role, 0) != watchdog_backend_by_role.get(role, 0)
    ]

    placement_stable = bool(sample.get("placement_stable"))
    newest_active_event_ts = max(
        (
            float(state.get("source_ts") or 0.0)
            for state in event_frame.get("requests") or []
            if isinstance(state, dict) and not state.get("terminal")
        ),
        default=0.0,
    )
    event_age_s = (
        max(0.0, tap_observed_at - newest_active_event_ts) if newest_active_event_ts else None
    )
    if not placement_stable:
        reconciliation_status = "torn"
        reconciliation_reason = "placement changed across the watchdog window"
    elif not slots_complete:
        reconciliation_status = "unknown"
        reconciliation_reason = "backend watchdog is incomplete"
    elif event_frame.get("degraded"):
        reconciliation_status = "degraded"
        reconciliation_reason = "lifecycle queue/history overflow is surfaced"
    elif reconciliation_differences and event_age_s is not None and event_age_s > 10.0:
        reconciliation_status = "stale"
        reconciliation_reason = "active lifecycle state has not advanced for over 10 seconds"
    elif reconciliation_differences:
        reconciliation_status = "drift"
        reconciliation_reason = "event lifecycle and /slots watchdog counts disagree"
    else:
        reconciliation_status = "agreement"
        reconciliation_reason = "event lifecycle agrees with the independent /slots watchdog"

    if not placement_stable:
        status = "torn"
        reason = "placement ownership changed across the /slots sampling window"
    elif not slots_complete:
        status = "unknown"
        reason = "one or more /slots sources were unavailable"
    elif placement_telemetry.get("conflict") or placement_telemetry.get("capacity_conflicts"):
        status = "conflict"
        reason = "placement role/capacity conflicts with backend telemetry"
    elif event_frame.get("degraded"):
        status = "degraded"
        reason = "lifecycle reducer overflow/eviction is explicitly degraded"
    elif count_disagreements:
        status = "stage_divergence"
        reason = "lease admission and backend processing counts differ"
    else:
        status = "certified"
        reason = "placement was stable across a complete /slots sample"

    live_frame = {
        "frame_id": frame_id,
        "frame_sequence": frame_sequence,
        "event_sequence": event_sequence,
        "status": status,
        "window_certified": placement_stable and slots_complete,
        "reason": reason,
        "attempt": int(sample.get("attempt") or 1),
        "max_attempts": int(sample.get("max_attempts") or 1),
        "placement_stable": placement_stable,
        "slots_complete": slots_complete,
        "unanswered_ports": sorted(unanswered_ports),
        "count_disagreements": count_disagreements,
        "observed_at": {
            "frame_started": sample.get("frame_started_at"),
            "placement_before": sample.get("placement_before_at"),
            "slots_started": sample.get("slots_started_at"),
            "slots_completed": sample.get("slots_completed_at"),
            "placement_after": sample.get("placement_after_at"),
            "tap": tap_observed_at,
        },
        "roles": role_signals,
        "lifecycle": event_frame,
        "reconciliation": {
            "status": reconciliation_status,
            "reason": reconciliation_reason,
            "event_age_s": event_age_s,
            "event_backend_by_role": event_backend_by_role,
            "watchdog_backend_by_role": watchdog_backend_by_role,
            "differences": reconciliation_differences,
            "history_overwritten": False,
        },
        "source_contract": {
            "placement": "cpu_region occupancy ledger",
            "backend": "llama-server /slots reconciliation watchdog",
            "tap": "structured inference tap request observations",
            "lifecycle": "bounded in-process sequenced lifecycle reducer",
        },
        "push_bridge": {
            "transition_source": "orchestrator lifecycle reducer",
            "coalescing": "SSE frame emission at no more than 2 Hz",
            "token_events_excluded": True,
            "delivery": "existing multiplex SSE triggers coupled snapshot reduction",
            "reconciliation": "/slots remains independent and never overwrites transitions",
            "gap_status": (
                "degraded_overflow_surfaced"
                if event_frame.get("degraded")
                else "sequenced_in_process"
            ),
        },
        "causal_observations": sorted(
            causal_observations,
            key=lambda item: (
                float(item.get("updated_at_epoch") or 0.0),
                str(item.get("request_id") or ""),
            ),
        )[-40:],
    }

    # Annotate, never replace, the raw placement matrix. Its numerator remains
    # a lease-reference count; backend and tap signals stay independently named.
    rows = (region_locks.get("display_matrix") or {}).get("rows") or []
    for row in rows if isinstance(rows, list) else []:
        role = _canonical_role_name(row.get("role")) if isinstance(row, dict) else ""
        signals = role_signals.get(role)
        if not signals:
            continue
        for cell in row.get("cells") or []:
            if not isinstance(cell, dict) or cell.get("state") != "active":
                continue
            lease_active = int(cell.get("active") or 0)
            capacity = max(1, int(cell.get("capacity") or 1))
            shape = str(cell.get("shape") or "full")
            backend = signals["backend"].get("by_shape", {}).get(shape, {})
            shape_tap = signals["tap"].get("by_shape", {}).get(shape, {})
            backend_text = "?" if backend.get("unknown") else str(backend.get("active") or 0)
            tap_decoding = int(shape_tap.get("decoding") or 0)
            cell["lease_active"] = lease_active
            cell["backend_active"] = (
                None if backend.get("unknown") else int(backend.get("active") or 0)
            )
            cell["backend_capacity"] = int(backend.get("capacity") or 0) or None
            cell["tap_decoding"] = tap_decoding
            cell["lifecycle"] = signals["lifecycle"]
            cell["label"] = (
                f"🔒 {lease_active}/{capacity} lease · "
                f"⚙ {backend_text}/{capacity} slots · ✦ {tap_decoding} tap"
            )
            cell["title"] = (
                f"{role}.{cell.get('shape') or 'instance'} — placement lease refs "
                f"{lease_active}/{capacity}; backend /slots processing "
                f"{backend_text}/{capacity}; tap decoding observed {tap_decoding}; "
                f"lifecycle={signals['lifecycle']}; frame={frame_id} ({status})"
            )

    placement_telemetry["status"] = status
    placement_telemetry["count_disagreements"] = count_disagreements
    placement_telemetry["frame_id"] = frame_id
    region_locks["live_frame"] = live_frame
    region_locks["frame_id"] = frame_id
    return live_frame


def _filter_instance_regions_for_mode(
    topology: dict[tuple[str, int], frozenset[str]],
    numa_mode: str,
) -> dict[tuple[str, int], frozenset[str]]:
    """Mirror stack_manifest._filter_by_numa_mode for region-lock rendering."""
    if numa_mode == "both":
        return topology
    try:
        from scripts.server.stack_numa import NUMA_CONFIG
    except Exception:
        return topology

    out: dict[tuple[str, int], frozenset[str]] = {}
    for (role, idx), regs in topology.items():
        cfg = NUMA_CONFIG.get(role) if isinstance(NUMA_CONFIG, dict) else None
        full_idx = cfg.get("full_instance_idx") if isinstance(cfg, dict) else None
        instances = cfg.get("instances") if isinstance(cfg, dict) else None
        if not isinstance(full_idx, int) or not isinstance(instances, list) or len(instances) <= 1:
            out[(role, idx)] = regs
            continue
        if numa_mode == "full" and idx == full_idx:
            out[(role, idx)] = regs
        elif numa_mode == "quarter" and idx != full_idx:
            out[(role, idx)] = regs
    return out


# 2026-09-27 ARCHITECT SWAP (operator-decided): the MI210 27B (:8083) now serves
# architect_critic (host of coder_escalation + ingest_long_context); architect_general
# is Flash-Next on the full CPU instance (:8074) and DOES hold CPU region locks.
# ingest_long_context added: it has ridden the :8083 GPU process since 2026-09-22.
_GPU_RESIDENT_ROLES = frozenset(
    {
        "architect_critic",
        "coder_escalation",
        "ingest_long_context",
        "worker_vision",
        "vision_escalation",
    }
)
_SPEECH_ROLES = frozenset({"whisper", "tts"})


def _region_locks_payload(
    numa_mode: str | None = None,
    *,
    _coherence_retries: int = 1,
    _coherence_attempt: int = 0,
) -> dict[str, Any]:
    """Return the current per-region lock snapshot as plain JSON data."""
    import os
    from pathlib import Path

    # Resolve realized-fleet-first (C1) so the `launch_selected` overlay (M2)
    # tracks the live fleet, not a launch-time env hint. A caller that passes an
    # explicit `numa_mode` (the coherent snapshot) still governs the grid, but the
    # provenance always reflects the same resolver so the badge cannot disagree.
    resolution = active_stack_numa_mode_resolution()
    active_mode = numa_mode or resolution["mode"]
    try:
        from src.runtime.cpu_region_lock import (
            _current_lock_owner_pids,
            _tmp_dir,
            read_region_lock_payload,
            read_region_occupancy,
        )

        tmp_dir = _tmp_dir()
        occupancy_before_observed_at = time.time()
        occupancy = read_region_occupancy()
    except Exception:
        tmp_dir = Path("/mnt/raid0/llm/tmp")
        occupancy = {"entries": [], "per_region": {}}
        occupancy_before_observed_at = time.time()
        from src.runtime.cpu_region_lock import _current_lock_owner_pids

        def read_region_lock_payload(_path: Path) -> None:  # type: ignore[no-redef]
            return None

    # Pull the configured (role, idx) → {regions} map. Region-lock observability
    # must stay structural: even when the active launch mode is "full", the
    # operator still needs to see quarter shapes and whether they are merely not
    # selected by this stack run. Runtime holders are still sourced from /proc
    # below; this topology is only the display/attribution map.
    try:
        from src.runtime.instance_topology import get_instance_regions, ATOMIC_REGIONS

        topology = get_instance_regions()
        launch_topology = _filter_instance_regions_for_mode(topology, active_mode)
        launch_selected_instances = set(launch_topology)
        all_regions = list(ATOMIC_REGIONS)
    except Exception:
        topology = {}
        launch_selected_instances = set()
        all_regions = ["q0", "q1", "q2", "q3"]

    # Role INCLUSION starts with the operator-curated contention matrix, then
    # adds every topology-backed role present in the atomic occupancy ledger.
    # Auxiliary lanes need not be benchmarked matrix roles to own real CPU
    # regions; an admitted placement can never be projected as "no lock domain".
    # Visible SHAPES, however, are the role's actual configured/running instances
    # (built below), NOT the matrix `same_role.instance_pairs`. Those pairs encode
    # which within-role shapes can CO-PLACE (a measured contention property) —
    # narrower than "which instances exist and are dispatchable". The heavy roles
    # (ingest_long_context 80B, architect_general) run quarter servers that are
    # live one-at-a-time dispatch targets yet have no co-placement pairs; gating
    # visibility on pairs wrongly hid them. The matrix still drives the ×-blocking
    # colouring below via `_region_lock_blocked_by_roles`. Lock-file existence is
    # NOT used to gate wiring — it's just a runtime hot/cold signal (cells stay
    # ✅ Ready until the backend first acquires a lock).
    matrix = None
    try:
        from src.scheduling.contention import load_contention_matrix

        matrix = load_contention_matrix()
    except Exception:
        matrix = None  # fail-open: panel still renders with NUMA-only fallback below

    # Build instance_topology from NUMA_CONFIG, filtered to matrix-allowed shapes.
    instance_topology_all: dict[str, list[dict[str, Any]]] = {}
    for (role, idx), regs in topology.items():
        if not regs:
            continue
        instance_topology_all.setdefault(role, []).append(
            {
                "idx": idx,
                "regions": sorted(regs),
                "span": len(regs),
                "shape": _shape_for_regions(regs),
                "is_full": len(regs) >= 2,  # deprecated
                "launch_selected": (role, idx) in launch_selected_instances,
            }
        )
    for role in instance_topology_all:
        instance_topology_all[role].sort(key=lambda x: (-x["span"], x["regions"]))

    raw_occupancy_entries = occupancy.get("entries") if isinstance(occupancy, dict) else []
    occupancy_entries = raw_occupancy_entries if isinstance(raw_occupancy_entries, list) else []

    # Determine the panel rows + per-role visible shapes. A role is INCLUDED iff
    # the matrix represents it; its VISIBLE SHAPES are its configured instances
    # (numa-mode-filtered above), so every dispatchable instance shows.
    panel_roles: set[str] = set()
    role_allowed_shapes: dict[str, set[str]] = {}
    if matrix is not None:
        try:
            from src.scheduling.contention import matrix_measured_roles

            measured_roles = matrix_measured_roles(matrix)
        except (AttributeError, TypeError):
            # Keep lightweight test/downgrade matrix objects compatible with the
            # pre-existing same-role-only shape.
            measured_roles = set(getattr(matrix, "same_role", {}))
        for role in measured_roles:
            insts = instance_topology_all.get(role) or []
            if not insts:
                continue  # matrix mentions a role we have no NUMA_CONFIG for
            role_allowed_shapes[role] = {i["shape"] for i in insts}
            panel_roles.add(role)
    if matrix is None:
        # Fallback: matrix missing/unreadable → every multi-instance role.
        for role, insts in instance_topology_all.items():
            if len(insts) >= 2:
                panel_roles.add(role)
                role_allowed_shapes[role] = {i["shape"] for i in insts}
    for entry in occupancy_entries:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "")
        insts = instance_topology_all.get(role) or []
        if not insts:
            continue
        panel_roles.add(role)
        role_allowed_shapes[role] = {i["shape"] for i in insts}

    # Filter the per-role instance list to the role's visible (configured) shapes.
    instance_topology: dict[str, list[dict[str, Any]]] = {}
    for role in panel_roles:
        allowed = role_allowed_shapes[role]
        instance_topology[role] = [
            i for i in (instance_topology_all.get(role) or []) if i["shape"] in allowed
        ]

    # Per-role: held-region-set → instance idx, used to attribute lock holders
    # to a specific instance after we scan /proc/locks. Resolve against the
    # full NUMA topology, not only matrix-visible panel shapes: a runtime holder
    # can be on a shape the matrix omits for idle display, and unresolved holders
    # make the panel report "no locks held" while regions are visibly occupied.
    role_instances_by_regions: dict[str, dict[frozenset[str], int]] = {
        role: {
            frozenset(inst["regions"]): inst["idx"]
            for inst in (instance_topology_all.get(role) or [])
        }
        for role in panel_roles
    }

    def _payload_instance_idx(
        payload: dict[str, Any] | None,
        *,
        role: str,
        region: str,
        holders: list[str],
    ) -> int | None:
        if not payload:
            return None
        if str(payload.get("role") or "") != role:
            return None
        if str(payload.get("region") or "") != region:
            return None
        pid = payload.get("pid")
        if holders and str(pid) not in {str(holder) for holder in holders}:
            return None
        try:
            idx = int(payload.get("instance_idx"))
        except (TypeError, ValueError):
            return None
        return idx if idx >= 0 else None

    # Pass 1a: collect raw lock-holder PIDs per (role, region) for matrix-included
    # roles only. Locks are acquired by orchestrator uvicorn workers (not by
    # llama-server processes). Newer lock holders also write a JSON payload under
    # the flock with direct instance_idx attribution; older/no-payload holders
    # fall back to the SET of regions a worker is currently holding.
    out: list[dict[str, Any]] = []
    # The host-wide GLOBAL.* lock domain is held for the duration of every CPU
    # inference admission but is NOT a contention-matrix role, so the
    # `role not in panel_roles` filter below silently dropped it. It was
    # therefore possible for the panel to report "no locks held" while four
    # GLOBAL quarters were flocked by a live request. Captured separately.
    global_lock_regions: list[dict[str, Any]] = []
    role_pid_regions: dict[str, dict[str, set[str]]] = {}  # role → pid → held regions
    try:
        for p in sorted(tmp_dir.glob("cpu_region.*.lock")):
            stem = p.stem  # "cpu_region.<role>.<region>"
            parts = stem.split(".", 2)
            if len(parts) < 3:
                continue
            _prefix, role, region = parts[0], parts[1], parts[2]
            if role == "GLOBAL":
                g_holders = _current_lock_owner_pids(p)
                g_payload = read_region_lock_payload(p) if g_holders else None
                global_lock_regions.append(
                    {
                        "region": region,
                        "lock_path": str(p),
                        "holder_pids": g_holders,
                        "held": bool(g_holders),
                        "lock_payload": g_payload,
                    }
                )
                continue
            if role not in panel_roles:
                continue  # role isn't in the matrix → don't surface lock state for it
            holders = _current_lock_owner_pids(p)
            payload = read_region_lock_payload(p) if holders else None
            payload_idx = _payload_instance_idx(
                payload,
                role=role,
                region=region,
                holders=holders,
            )
            out.append(
                {
                    "role": role,
                    "region": region,
                    "lock_path": str(p),
                    "holder_pids": holders,
                    "holder_instance_idxs": [payload_idx] if payload_idx is not None else [],
                    "lock_payload": payload if payload_idx is not None else None,
                    "held": bool(holders),
                    "wired": True,
                }
            )
            if holders and payload_idx is None:
                role_pid_regions.setdefault(role, {})
                for pid in holders:
                    role_pid_regions[role].setdefault(pid, set()).add(region)
    except Exception as exc:
        return {"error": str(exc), "entries": []}

    proc_observed_at = time.time()
    try:
        occupancy_after_observed_at = time.time()
        occupancy_after = read_region_occupancy()
    except Exception:
        occupancy_after_observed_at = time.time()
        occupancy_after = occupancy
    tokens_before = _occupancy_tokens(occupancy)
    tokens_after = _occupancy_tokens(occupancy_after)
    occupancy_changed_during_scan = tokens_before != tokens_after
    if occupancy_changed_during_scan and _coherence_retries > 0:
        return _region_locks_payload(
            active_mode,
            _coherence_retries=_coherence_retries - 1,
            _coherence_attempt=_coherence_attempt + 1,
        )
    if occupancy_changed_during_scan:
        # The after-sample is the newest placement fact. The coherence block
        # below marks this frame transitional so consumers do not interpret a
        # mixed proc/ledger frame as a confident holder identity.
        occupancy = occupancy_after

    # Pass 1b: resolve held-region SET per PID to a NUMA_CONFIG instance idx.
    pid_to_instance = _resolve_pid_to_instance_idx(role_pid_regions, role_instances_by_regions)
    for entry in out:
        if not entry["holder_pids"] or entry["holder_instance_idxs"]:
            continue
        idxs = {
            pid_to_instance[(entry["role"], pid)]
            for pid in entry["holder_pids"]
            if (entry["role"], pid) in pid_to_instance
        }
        entry["holder_instance_idxs"] = sorted(idxs)

    visible_instances: dict[str, list[dict[str, Any]]] = {
        role: list(insts) for role, insts in instance_topology.items()
    }
    for (role, _pid), idx in pid_to_instance.items():
        insts = visible_instances.setdefault(role, [])
        if any(inst["idx"] == idx for inst in insts):
            continue
        runtime_inst = next(
            (inst for inst in (instance_topology_all.get(role) or []) if inst["idx"] == idx),
            None,
        )
        if runtime_inst is not None:
            insts.append({**runtime_inst, "runtime_only": True})
    for insts in visible_instances.values():
        insts.sort(key=lambda x: (-x["span"], x["regions"]))

    # Pass 2: ensure every panel role has a row even when no lock file exists
    # yet on disk (a wired-but-never-dispatched role like vision_escalation
    # before its first request). Synthesize empty (free) region entries for
    # all 4 quarters so the frontend can render ✅ Ready cells.
    seen_role_regions = {(e["role"], e["region"]) for e in out}
    for role in sorted(panel_roles):
        for region in all_regions:
            if (role, region) in seen_role_regions:
                continue
            out.append(
                {
                    "role": role,
                    "region": region,
                    "lock_path": str(tmp_dir / f"cpu_region.{role}.{region}.lock"),
                    "holder_pids": [],
                    "holder_instance_idxs": [],
                    "lock_payload": None,
                    "held": False,
                    "wired": True,  # matrix-membership = wired; lock files appear lazily
                }
            )

    # Group by role for easier dashboard rendering.
    by_role: dict[str, dict[str, Any]] = {}
    for entry in out:
        bucket = by_role.setdefault(
            entry["role"],
            {
                "wired": True,
                "regions": [],
                "instances": visible_instances.get(entry["role"], []),
                "active_instance_idxs": set(),
                "blocked_by_roles": [],
            },
        )
        region_item = {
            "region": entry["region"],
            "held": entry["held"],
            "holder_pids": entry["holder_pids"],
            "holder_instance_idxs": entry["holder_instance_idxs"],
        }
        if entry.get("lock_payload"):
            region_item["lock_payload"] = entry["lock_payload"]
        bucket["regions"].append(region_item)
        for idx in entry["holder_instance_idxs"]:
            bucket["active_instance_idxs"].add(idx)

    # `/proc` is the live flock witness; the occupancy ledger is the atomic
    # cross-process admission boundary. Shared native-batch cohorts cannot put
    # mutable attribution in their flock files, so the ledger is canonical for
    # their role, instance and staged load. Preserve the raw proc view alongside
    # the canonical view for coherence diagnostics.
    occupancy_pairs = _occupancy_instance_pairs(occupancy)
    shared_occupancy_pairs = _occupancy_instance_pairs(occupancy, shared_only=True)
    shared_regions = {
        str(region)
        for entry in (occupancy.get("entries", []) if isinstance(occupancy, dict) else [])
        if isinstance(entry, dict) and entry.get("shared")
        for region in (entry.get("regions") or [])
    }
    for role, bucket in by_role.items():
        proc_idxs = {int(idx) for idx in bucket["active_instance_idxs"]}
        bucket["proc_active_instance_idxs"] = sorted(proc_idxs)
        canonical_idxs = set(proc_idxs)
        if shared_occupancy_pairs:
            # A proc holder on regions governed by a stable shared ledger is
            # corroboration only. Do not let a torn/misattributed proc sample
            # override the ledger's serving-process cohort.
            for idx in tuple(canonical_idxs):
                inst = next(
                    (
                        item
                        for item in bucket.get("instances", [])
                        if str(item.get("idx", "")).lstrip("-").isdigit()
                        and int(item["idx"]) == idx
                    ),
                    None,
                )
                if (
                    inst is not None
                    and shared_regions.intersection(
                        str(region) for region in inst.get("regions", [])
                    )
                    and (role, idx) not in shared_occupancy_pairs
                ):
                    canonical_idxs.discard(idx)
        canonical_idxs.update(idx for pair_role, idx in occupancy_pairs if pair_role == role)
        bucket["active_instance_idxs"] = sorted(canonical_idxs)

    active_lock_roles = {role for role, bucket in by_role.items() if bucket["active_instance_idxs"]}
    for role, bucket in by_role.items():
        bucket["blocked_by_roles"] = _region_lock_blocked_by_roles(
            role,
            active_lock_roles,
            matrix,
        )

    # Quarters are RETIRED as a deployable shape (2026-07-30): every CPU role
    # runs 1 full + 2 halves. q0-q3 remain the atomic LOCK granularity
    # internally — a full holds all four, a half holds two — but rendering a
    # column per quarter implied four deployable instances per role and made
    # the panel unreadable. Shapes are what the operator deploys, so shapes are
    # what the panel shows. `entries` below still carries per-quarter lock
    # state for anything that needs the atomic view.
    display_columns = [
        {"key": "full", "label": "Full"},
        {"key": "half0", "label": "Half0"},
        {"key": "half1", "label": "Half1"},
    ]
    held_by_region: dict[str, list[dict[str, Any]]] = {}
    held_region_identities: set[tuple[str, int, str]] = set()

    def _record_held_region(role: str, idx: int, shape: str, region: str) -> None:
        identity = (role, idx, region)
        if identity in held_region_identities:
            return
        held_region_identities.add(identity)
        held_by_region.setdefault(region, []).append({"role": role, "idx": idx, "shape": shape})

    # Fail-safe source: the occupancy ledger is the admission authority. Seed
    # physical occupancy directly from it before any display-role projection so
    # a missing matrix row or /proc attribution can never paint occupied cores
    # as ready.
    for entry in occupancy_entries:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "unknown")
        try:
            idx = int(entry.get("instance_idx"))
        except (TypeError, ValueError):
            idx = -1
        inst = next(
            (
                item
                for item in (instance_topology_all.get(role) or [])
                if int(item.get("idx", -2)) == idx
            ),
            None,
        )
        shape = str((inst or {}).get("shape") or f"idx{idx}")
        regions = entry.get("regions") if isinstance(entry.get("regions"), list) else []
        for region in regions:
            _record_held_region(role, idx, shape, str(region))
    for role, bucket in by_role.items():
        inst_by_idx = {
            int(inst["idx"]): inst
            for inst in bucket.get("instances", [])
            if str(inst.get("idx", "")).lstrip("-").isdigit()
        }
        for idx in bucket.get("active_instance_idxs", []):
            inst = inst_by_idx.get(int(idx))
            if inst is None:
                continue
            for region in inst.get("regions", []):
                _record_held_region(
                    role,
                    int(idx),
                    str(inst.get("shape", f"idx{idx}")),
                    str(region),
                )

    display_rows: list[dict[str, Any]] = []

    def _configured_serving_capacity(role: str, instance_idx: int) -> tuple[int, str]:
        """Declared llama-server slots for this physical serving instance.

        The occupancy ledger's ``capacity`` is the placement-lease admission
        width.  Ordinary requests take an exclusive lease, so that value is 1
        even when the selected server has several native slots.  The lock grid
        labels active/available SERVING slots, therefore its denominator must
        come from the same per-shape declaration the launcher uses for ``-np``.
        """
        try:
            from scripts.server.stack_manifest import resolve_slots

            decision = resolve_slots(role, numa_instance=instance_idx)
            return max(1, int(decision.slots)), str(decision.source)
        except Exception as exc:
            return 1, f"fallback:error:{type(exc).__name__}"

    for role in sorted(by_role):
        bucket = by_role[role]
        insts = list(bucket.get("instances", []))
        active_idxs = {int(idx) for idx in bucket.get("active_instance_idxs", [])}
        cells: list[dict[str, Any]] = []
        for col in display_columns:
            inst = next((i for i in insts if i.get("shape") == col["key"]), None)
            if inst is None:
                cells.append(
                    {"state": "na", "label": "—", "title": f"{role} has no {col['label']} shape"}
                )
                continue
            idx = int(inst["idx"])
            regions = [str(r) for r in inst.get("regions", [])]
            configured_capacity, configured_capacity_source = _configured_serving_capacity(
                role, idx
            )
            instance_load = [
                entry
                for entry in occupancy_entries
                if str(entry.get("role") or "") == role and entry.get("instance_idx") == idx
            ]
            active = len(instance_load)
            lease_capacity = max(
                [int(entry.get("capacity") or 1) for entry in instance_load] or [1]
            )
            capacity = max(
                lease_capacity,
                configured_capacity,
            )
            if idx in active_idxs:
                cells.append(
                    {
                        "state": "active",
                        "shape": col["key"],
                        "label": f"⚡ {active}/{capacity}" if active else "⚡",
                        "active": active,
                        "capacity": capacity,
                        "lease_capacity": lease_capacity,
                        "configured_capacity": configured_capacity,
                        "configured_capacity_source": configured_capacity_source,
                        "load": min(1.0, active / capacity) if active else 1.0,
                        "title": (
                            f"{role}.{col['label']} ACTIVE — {active or 'unknown'} of "
                            f"{capacity} configured serving slot(s); placement lease "
                            f"admission capacity={lease_capacity}; instance idx={idx} "
                            f"holding {{{','.join(regions)}}}"
                        ),
                    }
                )
                continue
            blocking: list[str] = []
            for region in regions:
                blockers = [f"{h['role']}.{h['shape']}" for h in held_by_region.get(region, [])]
                if blockers:
                    blocking.append(f"{region}←{','.join(blockers)}")
            if blocking:
                cells.append(
                    {
                        "state": "blocked",
                        "shape": col["key"],
                        "label": "×",
                        "title": (
                            f"{role}.{col['label']} LOCKED — physical cores "
                            f"{','.join(regions)} overlap active placement leases: "
                            f"{' · '.join(blocking)}"
                        ),
                    }
                )
            else:
                selected = bool(inst.get("launch_selected", True))
                mode_note = (
                    ""
                    if selected
                    else f" — configured but not selected by stack_numa_mode={active_mode}"
                )
                cells.append(
                    {
                        "state": "ready",
                        "label": "✅",
                        "title": (
                            f"{role}.{col['label']} FREE — regions "
                            f"{{{','.join(regions)}}}{mode_note}"
                        ),
                    }
                )
        display_rows.append({"role": role, "cells": cells})

    # Panel COMPLETENESS (operator report 2026-07-26): the matrix gate above
    # means only lock-domain roles (the quarter-split trio) ever got a row, so
    # the panel read as "only frontdoor / ingest_long_context / worker_general
    # exist". Append every OTHER active serving instance — live role set from
    # the same ps(1)-based discovery the topology panel uses
    # (`_port_roles_cached`), configured shapes from NUMA topology — as
    # DISPLAY-ONLY rows with lock state "n/a (no lock domain)". Lock semantics
    # are untouched: these roles have no cpu_region lock domain, and the rows
    # exist so the operator sees the whole fleet, not to gate anything.
    # Fail-open like the rest of this panel: discovery errors → no extra rows,
    # never a broken panel.
    non_lock_roles: dict[str, dict[str, Any]] = {}
    try:
        live_port_roles = _port_roles_cached()
    except Exception:
        live_port_roles = {}
    role_ports: dict[str, list[int]] = {}
    for port, port_role in (live_port_roles or {}).items():
        # Quarter lanes are labeled "role.qN" by the port discovery — fold them
        # onto the base role so a lock-domain role's quarters never spawn a
        # duplicate "no lock domain" row.
        base_role = str(port_role).split(".", 1)[0]
        try:
            role_ports.setdefault(base_role, []).append(int(port))
        except (TypeError, ValueError):
            continue
    _NO_LOCK_STATE = "n/a (no lock domain)"

    # Rows the operator does NOT want in a "what is holding compute" view: the
    # 6-instance embedder pool and an IDLE default-off eval-batch lane add noisy
    # rows. Active eval-batch lanes are promoted into `by_role` above because
    # their occupancy entry proves a real lock domain; only an idle auxiliary
    # lane reaches this display filter.
    def _hidden_from_region_panel(name: str) -> bool:
        return name.startswith("embedder") or name.startswith("eval_batch")

    # Speech services are NOT llama-server processes, so port discovery never
    # sees them and they were absent from the panel entirely — invisible since
    # they went live on the MI210 on 2026-08-02. Fold them in from the launch
    # manifest, the same table `start`/`reload` dispatch off, so a newly
    # declared service appears without another edit here.
    try:
        from scripts.server.stack_manifest import AUX_SERVICES

        for _svc in AUX_SERVICES.values():
            _name = str(getattr(_svc, "name", ""))
            if _name in _SPEECH_ROLES and isinstance(getattr(_svc, "port", None), int):
                role_ports.setdefault(_name, []).append(int(_svc.port))
    except Exception:  # noqa: BLE001 — fail open, never break the panel
        pass

    for role in sorted(set(role_ports) | set(instance_topology_all)):
        if role in by_role:
            continue
        insts = list(instance_topology_all.get(role) or [])
        ports = sorted(role_ports.get(role, []))
        # Populated for EVERY non-lock role, hidden or not: the exclusion above
        # is a DISPLAY filter, and the `continue` used to sit here, which silently
        # dropped the embedder pool and the eval-batch lane out of the API payload
        # too — the opposite of what that comment promises.
        non_lock_roles[role] = {
            "ports": ports,
            "instances": insts,
            "lock_state": _NO_LOCK_STATE,
        }
        if _hidden_from_region_panel(role):
            continue
        ports_note = f" · ports {','.join(str(p) for p in ports)}" if ports else ""
        inst_by_shape = {str(i.get("shape")): i for i in insts}
        cells = []
        for col in display_columns:
            inst = inst_by_shape.get(col["key"])
            if inst is not None:
                regions = [str(r) for r in inst.get("regions", [])]
                cells.append(
                    {
                        "state": "nolock",
                        "label": "n/a",
                        "title": (
                            f"{role}.{col['label']} — regions {{{','.join(regions)}}}"
                            f"{ports_note} · {_NO_LOCK_STATE}"
                        ),
                    }
                )
            elif inst_by_shape or col["key"] != "full":
                cells.append(
                    {"state": "na", "label": "—", "title": f"{role} has no {col['label']} shape"}
                )
            else:
                # No NUMA shape at all — GPU-resident roles and the speech
                # services. "n/a" told the operator nothing: these ARE live and
                # dispatchable, they simply hold no CPU region. Say what they
                # are and that they are up, which is the question this panel is
                # being read to answer.
                if role in _GPU_RESIDENT_ROLES:
                    where, mark = "GPU (MI210)", "▣"
                elif role in _SPEECH_ROLES:
                    where, mark = "GPU (MI210) · speech", "▣"
                else:
                    where, mark = "no CPU region", "·"
                cells.append(
                    {
                        "state": "nolock",
                        "label": f"{mark} {where}",
                        "title": (
                            f"{role} — resident on {where}{ports_note}; holds no CPU "
                            f"region lock, so no Full/Half shape applies"
                        ),
                    }
                )
        display_rows.append(
            {
                "role": role,
                "label": f"{role}{ports_note.replace(' · ports ', ' :')}",
                "cells": cells,
                "no_lock_domain": True,
                "lock_state": _NO_LOCK_STATE,
                "ports": ports,
            }
        )

    proc_pairs = _proc_instance_pairs(by_role)
    ledger_pairs = _occupancy_instance_pairs(occupancy)
    if occupancy_changed_during_scan:
        coherence_status = "transition"
    elif proc_pairs == ledger_pairs:
        coherence_status = "coherent"
    else:
        coherence_status = "mismatch"
    coherence = {
        "status": coherence_status,
        "retries": _coherence_attempt,
        "occupancy_changed_during_scan": occupancy_changed_during_scan,
        "retry_exhausted": occupancy_changed_during_scan and _coherence_retries <= 0,
        "proc_pairs": [list(pair) for pair in sorted(proc_pairs)],
        "occupancy_pairs": [list(pair) for pair in sorted(ledger_pairs)],
        "shared_occupancy_pairs": [list(pair) for pair in sorted(shared_occupancy_pairs)],
        "tokens_before": list(tokens_before),
        "tokens_after": list(tokens_after),
        "occupancy_before_observed_at": occupancy_before_observed_at,
        "proc_observed_at": proc_observed_at,
        "occupancy_after_observed_at": occupancy_after_observed_at,
    }

    feature_flag = os.environ.get("ORCHESTRATOR_PER_REGION_LOCKS", "0").strip()
    return {
        "per_region_locks_enabled": feature_flag in {"1", "true", "yes", "on"},
        "stack_numa_mode": active_mode,
        "stack_numa_mode_provenance": resolution,
        "matrix_loaded": matrix is not None,
        "tmp_dir": str(tmp_dir),
        "lock_root": _lock_root_provenance(tmp_dir),
        "entries": out,
        "by_role": by_role,
        # Host-wide admission lock (cpu_region.GLOBAL.*). Not a matrix role, so
        # it is not in `entries`/`by_role`; surfaced here so "held across N
        # regions" can never read 0 while a global admission is in force.
        "global_lock": {
            "regions": sorted(global_lock_regions, key=lambda r: str(r.get("region"))),
            "held": any(r["held"] for r in global_lock_regions),
            "holder_pids": sorted(
                {pid for r in global_lock_regions for pid in (r.get("holder_pids") or [])}
            ),
            "held_region_count": sum(1 for r in global_lock_regions if r["held"]),
        },
        "occupancy": occupancy,
        "coherence": coherence,
        # Display-only completeness rows: active/configured serving roles with
        # NO cpu_region lock domain (full-span roles, aliases, embedders).
        # NOTE: these roles (GPU-resident: architect_critic since the 2026-09-27
        # swap — architect_general before —, worker_vision,
        # speech) NEVER take a CPU region lock by design. Inference on them is
        # real but invisible to this panel — that is why the lock grid must
        # never be read as the fleet's activity signal. `/slots` occupancy in
        # `topology_activity` is the signal that covers them.
        "non_lock_roles": non_lock_roles,
        "display_matrix": {
            "columns": display_columns,
            "rows": display_rows,
            "row_kind": "role",
            "role_count": len(by_role),
            "instance_count": sum(len(bucket.get("instances", [])) for bucket in by_role.values()),
            "launch_mode": active_mode,
            "topology_mode": "all_configured",
            "active_holder_count": sum(
                len(bucket.get("active_instance_idxs", [])) for bucket in by_role.values()
            ),
            "held_regions": sorted(held_by_region),
        },
        "topology_quartered_roles": sorted(panel_roles),  # back-compat field
        "now": time.time(),
    }


@router.get("/dashboard/api/region_locks")
async def region_locks_snapshot() -> JSONResponse:
    """Per-CPU-region placement state and its `/proc` lock corroboration.

    Shared cohorts are attributed by the atomic occupancy ledger; a
    `/proc/locks` scan corroborates flock liveness. Static topology from
    `instance_topology.get_instance_regions()` lets the panel surface
    ALL roles that *have* quartered instances — not just the ones whose
    dispatch path created a lock file. Roles with quarter instances but no
    lock files are tagged
    ``wired=False`` so the dashboard can render them greyed out with
    an explanatory badge (they were quartered at launch but their
    backend never acquires `cpu_region_lock`, so cross-process
    concurrency isn't actually enforced for them yet).
    """
    payload = _region_locks_payload()
    payload["live_frame"] = {
        "frame_id": None,
        "status": "unknown",
        "window_certified": False,
        "reason": (
            "standalone region-lock sample is uncoupled from /slots and tap; "
            "use /dashboard/api/snapshot for live cross-panel claims"
        ),
        "observed_at": {"placement_after": payload.get("now")},
    }
    return JSONResponse(_stamp(payload, "region_locks"))


# Severity ranking for folding many panels into one dashboard-data health verdict.
_HEALTH_SEVERITY = {"fresh": 0, "aging": 1, "stale": 2, "dead": 3}
_HEALTH_STATUS = {"fresh": "ok", "aging": "ok", "stale": "degraded", "dead": "degraded"}


def _serve_path_health(now: float) -> dict[str, Any]:
    """Classify the snapshot serve path from this worker's build vitals.

    Stale when a build attempt is outstanding past the stall threshold (hang)
    or the newest attempt errored (crash loop). A worker with no recent demand
    (no open tab / SSE) has last_attempt == last_success and stays fresh —
    idle is not degraded. Per-worker stats: with 6 uvicorn workers one health
    call samples one worker; repeat curls sample the pool.
    """
    s = dict(_SNAPSHOT_BUILD_STATS)
    s["pid"] = os.getpid()
    attempt, success, err_ts = s["last_attempt_ts"], s["last_success_ts"], s["last_error_ts"]
    cls = "fresh"
    reason = ""
    if err_ts is not None and (success is None or err_ts > success):
        cls = "stale"
        reason = f"snapshot serve path erroring (worker {s['pid']}): {s['last_error']}"
    elif (
        attempt is not None
        and (success is None or attempt > success)
        and now - attempt > _SNAPSHOT_SERVE_PATH_STALL_S
    ):
        cls = "stale"
        reason = (
            f"snapshot serve path stalled (worker {s['pid']}): build started "
            f"{round(now - attempt)}s ago, no result"
        )
    s["staleness_class"] = cls
    s["reason"] = reason
    return s


@router.get("/dashboard/api/health")
async def dashboard_health(probe: str | None = None) -> JSONResponse:
    """Fold every registered panel's freshness into one dashboard-data health view.

    This is the anti-whack-a-mole guard. Each prior "dashboard panel stale"
    incident was a different producer dying silently; here, if ANY file-backed
    producer stops advancing, this endpoint reports ``degraded`` naming the
    offending panel + source, so breakage is caught loudly by curl/monitor
    instead of by eyeballing a panel. Live panels (recomputed per request) are
    fresh by construction and never drag the verdict down on their own — their
    failure mode is the SERVE PATH, covered by the serve_path block below and
    the on-demand ``?probe=snapshot`` real-build check (the one probe a hang
    cannot hide from).
    """
    now = time.time()
    panels_out: list[dict[str, Any]] = []
    worst = "fresh"
    for spec in _PANELS:
        env = _freshness_envelope(spec.live_sources(), now=now)
        cls = env["staleness_class"]
        if _HEALTH_SEVERITY[cls] > _HEALTH_SEVERITY[worst]:
            worst = cls
        panels_out.append(
            {
                "key": spec.key,
                "title": spec.title,
                "endpoint": spec.endpoint,
                "mechanism": spec.mechanism,
                "live": spec.live,
                "staleness_class": cls,
                "worst_age_s": env["worst_age_s"],
                "reason": env["reason"],
                "sources": env["sources"],
            }
        )

    serve_path = _serve_path_health(now)
    if _HEALTH_SEVERITY[serve_path["staleness_class"]] > _HEALTH_SEVERITY[worst]:
        worst = serve_path["staleness_class"]

    probe_result: dict[str, Any] | None = None
    if probe == "snapshot":
        t0 = time.time()
        try:
            await asyncio.wait_for(snapshot(), timeout=10.0)
            probe_result = {"ok": True, "duration_s": round(time.time() - t0, 3)}
        except asyncio.TimeoutError:
            probe_result = {"ok": False, "timeout_s": 10.0}
        except Exception as exc:
            probe_result = {
                "ok": False,
                "error": str(exc),
                "duration_s": round(time.time() - t0, 3),
            }
        if not probe_result.get("ok") and _HEALTH_SEVERITY["stale"] > _HEALTH_SEVERITY[worst]:
            worst = "stale"

    body: dict[str, Any] = {
        "status": _HEALTH_STATUS[worst],
        "worst_class": worst,
        "generated_at": now,
        "panel_count": len(panels_out),
        "degraded_panels": [
            p["key"] for p in panels_out if p["staleness_class"] in ("stale", "dead")
        ],
        "serve_path": serve_path,
        "panels": panels_out,
    }
    if probe_result is not None:
        body["probe"] = probe_result
    return JSONResponse(body)


@router.get("/dashboard/events/raw_tap")
async def raw_tap_stream(request: Request, tail_bytes: int = 8192) -> StreamingResponse:
    """True byte-level streaming SSE — mirrors what autopilot's TUI does.

    Tails inference_tap.log at 10Hz, pushing any new bytes since the last
    read directly to the client. The tap writer in eval_tower flushes
    per-token during generation, so this is genuine real-time streaming
    (same source the TUI uses).

    Anti-buffering headers ensure intermediate proxies and uvicorn itself
    don't batch small chunks — required for per-token feel in the browser.
    """

    async def event_gen():
        try:
            fh = open(_INFERENCE_TAP_PATH, "rb")
            fh.seek(0, 2)
            size = fh.tell()
            start = max(0, size - tail_bytes)
            fh.seek(start)
            initial = fh.read().decode("utf-8", errors="replace")
            yield "data: " + json.dumps({"chunk": initial, "initial": True}) + "\n\n"
            heartbeat_counter = 0
            while True:
                if await request.is_disconnected():
                    fh.close()
                    return
                pos = fh.tell()
                fh.seek(pos)
                raw = fh.read(8192)
                if raw:
                    chunk = raw.decode("utf-8", errors="replace")
                    yield "data: " + json.dumps({"chunk": chunk, "initial": False}) + "\n\n"
                    heartbeat_counter = 0
                else:
                    heartbeat_counter += 1
                    # Send an SSE comment every ~3s to keep the connection
                    # warm + flush any TCP buffer (Nagle / browser-side).
                    if heartbeat_counter >= 30:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                    await asyncio.sleep(0.1)
        except Exception as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
            try:
                fh.close()  # type: ignore[has-type]
            except Exception:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx / reverse-proxy buffering
            "Connection": "keep-alive",
        },
    )


@router.get("/dashboard/events/structured_tap")
async def structured_tap_stream(request: Request) -> StreamingResponse:
    """SSE stream of request-grouped structured tap snapshots."""

    async def event_gen():
        last_mtime = -1.0
        last_emit = 0.0
        while True:
            if await request.is_disconnected():
                return
            try:
                mtime = (
                    _INFERENCE_TAP_EVENTS_PATH.stat().st_mtime
                    if _INFERENCE_TAP_EVENTS_PATH.exists()
                    else 0.0
                )
            except Exception:
                mtime = 0.0
            now_epoch = time.time()
            # Also repaint open requests as they age into "quiet"; waiting
            # only for mtime changes leaves a silent running request looking
            # green forever until another tap event arrives.
            if mtime != last_mtime or (now_epoch - last_emit) >= 2.0:
                last_mtime = mtime
                last_emit = now_epoch
                tail = _read_tap_events_tail(_INFERENCE_TAP_EVENTS_PATH, max_bytes=1024 * 1024)
                structured_requests = _parse_structured_tap_requests(
                    tail,
                    max_requests=40,
                    now_epoch=now_epoch,
                )
                enriched_requests = _enrich_structured_tap_requests(structured_requests)
                payload = json.dumps(
                    {
                        "tap_active": _structured_tap_active(enriched_requests),
                        "tap_sentinel_active": _TAP_SENTINEL_PATH.exists(),
                        "structured_requests": enriched_requests,
                        "structured_tap_mtime": mtime or None,
                        "now": now_epoch,
                    }
                )
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/dashboard/events/autopilot_log")
async def autopilot_log_stream(request: Request, tail_bytes: int = 16384) -> StreamingResponse:
    """Byte-level SSE tail of autopilot.log — surfaces every phase autopilot
    is in (eval / seed_batch / meta-planning via `claude -p` / rollback /
    safety violation / etc.).

    The inference_tap.log only sees /chat traffic. seed_specialist_routing,
    meta-planning, eval scoring, and rollback decisions all log to
    autopilot.log without ever hitting /chat. This panel is the single
    "what is autopilot doing right now" view.

    Initial payload: last `tail_bytes` of the log. Subsequent: incremental
    appends, pushed within ~100ms of being written.
    """
    log_path = AUTOPILOT_LOG

    async def event_gen():
        try:
            if not log_path.exists():
                yield "data: " + json.dumps({"error": f"{log_path} does not exist"}) + "\n\n"
                return
            fh = open(log_path, "rb")
            fh.seek(0, 2)
            size = fh.tell()
            start = max(0, size - tail_bytes)
            fh.seek(start)
            initial = fh.read().decode("utf-8", errors="replace")
            yield "data: " + json.dumps({"chunk": initial, "initial": True}) + "\n\n"
            heartbeat_counter = 0
            while True:
                if await request.is_disconnected():
                    fh.close()
                    return
                pos = fh.tell()
                fh.seek(pos)
                raw = fh.read(8192)
                if raw:
                    chunk = raw.decode("utf-8", errors="replace")
                    yield "data: " + json.dumps({"chunk": chunk, "initial": False}) + "\n\n"
                    heartbeat_counter = 0
                else:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 30:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                    await asyncio.sleep(0.1)
        except Exception as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
            try:
                fh.close()  # type: ignore[has-type]
            except Exception:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/dashboard/events/planner_tap")
async def planner_tap_stream(request: Request, tail_bytes: int = 16384) -> StreamingResponse:
    """Byte-level SSE tail of the planner tap file written by
    controller_io.invoke_controller. Streams each `claude -p` event
    (system init / assistant thinking / tool_use / tool_result / result)
    summarized as it's emitted — no more 60-120s black-box waits during
    meta-planning.

    The tap is APPENDED across planner sessions with section separators,
    so the panel shows recent planner history at a glance.
    """
    planner_tap_path = Path("/mnt/raid0/llm/tmp/planner_tap.log")

    async def event_gen():
        try:
            if not planner_tap_path.exists():
                # Create empty file so the SSE doesn't error before the
                # first planner session writes anything.
                planner_tap_path.parent.mkdir(parents=True, exist_ok=True)
                planner_tap_path.touch()
            fh = open(planner_tap_path, "rb")
            fh.seek(0, 2)
            size = fh.tell()
            start = max(0, size - tail_bytes)
            fh.seek(start)
            initial = fh.read().decode("utf-8", errors="replace")
            yield "data: " + json.dumps({"chunk": initial, "initial": True}) + "\n\n"
            heartbeat_counter = 0
            while True:
                if await request.is_disconnected():
                    fh.close()
                    return
                pos = fh.tell()
                fh.seek(pos)
                raw = fh.read(8192)
                if raw:
                    chunk = raw.decode("utf-8", errors="replace")
                    yield "data: " + json.dumps({"chunk": chunk, "initial": False}) + "\n\n"
                    heartbeat_counter = 0
                else:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 30:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                    await asyncio.sleep(0.1)
        except Exception as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
            try:
                fh.close()  # type: ignore[has-type]
            except Exception:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/dashboard/events/inference_tap")
async def inference_tap_stream(request: Request) -> StreamingResponse:
    """SSE stream — emit a new payload whenever any tap file mtime advances.

    Polls mtimes at 2Hz; pushes a full snapshot when any tap file has changed
    since the last push.
    """

    async def event_gen():
        last_mtimes = {"inference": 0.0, "structured": 0.0, "repl": 0.0}
        while True:
            if await request.is_disconnected():
                return
            try:
                inf_m = _INFERENCE_TAP_PATH.stat().st_mtime if _INFERENCE_TAP_PATH.exists() else 0.0
                structured_m = (
                    _INFERENCE_TAP_EVENTS_PATH.stat().st_mtime
                    if _INFERENCE_TAP_EVENTS_PATH.exists()
                    else 0.0
                )
                rpl_m = _REPL_TAP_PATH.stat().st_mtime if _REPL_TAP_PATH.exists() else 0.0
            except Exception:
                inf_m = structured_m = rpl_m = 0.0
            changed = (
                inf_m > last_mtimes["inference"]
                or structured_m > last_mtimes["structured"]
                or rpl_m > last_mtimes["repl"]
            )
            if changed:
                last_mtimes = {
                    "inference": inf_m,
                    "structured": structured_m,
                    "repl": rpl_m,
                }
                # Build the payload — same shape as the snapshot endpoint
                inference_tail = _read_tail(_INFERENCE_TAP_PATH, max_bytes=256 * 1024)
                sections = _parse_inference_sections(inference_tail, max_sections=10)
                now_epoch = time.time()
                # ONE assembly path with the poll endpoint (RTG-47): this
                # stream used to run its own parse+enrich, which silently
                # lacked everything the funnel adds — offwindow lock-holder
                # recovery and live slot_progress. The page renders from THIS
                # stream, so a 33k-token prefill showed no ↑ counter while the
                # poll endpoint (which nobody watches) carried it.
                enriched_requests = _structured_tap_requests_for_dashboard(
                    max_requests=10,
                    now_epoch=now_epoch,
                )
                payload = json.dumps(
                    {
                        "tap_active": _structured_tap_active(enriched_requests),
                        "tap_sentinel_active": _TAP_SENTINEL_PATH.exists(),
                        "inference_sections": sections,
                        "structured_requests": enriched_requests,
                        "inference_tap_mtime": inf_m,
                        "structured_tap_mtime": structured_m,
                        "repl_tap_mtime": rpl_m,
                        # Server epoch, matching the poll payload: slot_progress
                        # sampled_at is a server clock, and the page differences
                        # it — give it the same clock instead of the browser's.
                        "now": now_epoch,
                    }
                )
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Process status — autopilot alive/dead + orchestrator uptime
# ---------------------------------------------------------------------------


@router.get("/dashboard/api/process_status")
async def process_status() -> JSONResponse:
    """Aliveness check for autopilot + count of GEPA worker subprocesses."""
    autopilot = _process_info_by_match("autopilot.py start")
    # Count spawn_main children (GEPA workers + reembed workers)
    try:
        import subprocess

        out = subprocess.run(
            ["pgrep", "-cf", "spawn_main"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        n_workers = int(out) if out else 0
    except Exception:
        n_workers = 0
    # Recent autopilot log tail (last 3 substantive lines)
    recent_lines: list[str] = []
    last_log_age_s: float | None = None
    if AUTOPILOT_LOG.exists():
        try:
            mtime = AUTOPILOT_LOG.stat().st_mtime
            last_log_age_s = time.time() - mtime
            size = AUTOPILOT_LOG.stat().st_size
            with open(AUTOPILOT_LOG, "rb") as f:
                if size > 16 * 1024:
                    f.seek(-16 * 1024, 2)
                tail = f.read().decode("utf-8", errors="ignore")
            recent_lines = [line for line in tail.splitlines() if line.strip()][-5:]
        except Exception:
            pass

    # Stream-source mtimes — drive the live-dot activity-state badges in the
    # left-column panels. The browser computes its own age from the timestamp
    # plus a freshness threshold; we just report mtimes (None when missing).
    def _age_s(p: Path) -> float | None:
        try:
            return time.time() - p.stat().st_mtime
        except Exception:
            return None

    phase = _read_autopilot_phase()
    phase_health = _autopilot_phase_health()
    outcome_progress = (
        phase_health.get("outcome_progress") if isinstance(phase_health, dict) else None
    )
    if not isinstance(outcome_progress, dict):
        outcome_progress = {}
    phase_age_s = phase_health.get("heartbeat_age_s")
    if phase_age_s is None:
        phase_age_s = _age_s(AUTOPILOT_PHASE_PATH)
    if phase and not (autopilot or {}).get("running"):
        phase = dict(phase)
        phase.setdefault("idle_reason", "autopilot process not running")

    planner_tap_path = Path("/mnt/raid0/llm/tmp/planner_tap.log")
    planner_tap_mtime_s: float | None = None
    planner_tap_precedes_process = False
    try:
        planner_tap_mtime_s = planner_tap_path.stat().st_mtime
        phase_health_dict = phase_health if isinstance(phase_health, dict) else {}
        process_started_at_s = phase_health_dict.get("process_started_at_s")
        if isinstance(process_started_at_s, (int, float)):
            planner_tap_precedes_process = planner_tap_mtime_s < float(process_started_at_s)
    except OSError:
        pass

    return JSONResponse(
        _stamp(
            {
                "autopilot": autopilot,
                "gepa_worker_count": n_workers,
                "last_autopilot_log_age_s": last_log_age_s,
                "autopilot_recent_lines": recent_lines,
                "autopilot_phase": phase,
                "autopilot_phase_health": phase_health,
                "autopilot_outcome_progress": outcome_progress,
                "autopilot_state": _autopilot_state_summary(),
                # H2: shared autopilot-plane epoch so a client can cross-check
                # this panel against the other three and discard a torn set.
                "state_generation": _autopilot_state_generation(),
                "autopilot_phase_age_s": phase_age_s,
                "inference_tap_age_s": _age_s(_INFERENCE_TAP_PATH),
                "planner_tap_age_s": _age_s(planner_tap_path),
                "planner_tap_mtime_s": planner_tap_mtime_s,
                "planner_tap_precedes_autopilot_start": planner_tap_precedes_process,
            },
            "process_status",
        ),
        headers=_NO_STORE_HEADERS,
    )


@router.post("/dashboard/api/autopilot_control")
async def autopilot_control(request: Request) -> JSONResponse:
    """Operator-owned AutoPilot pause/resume latch.

    This endpoint writes only the existing ``autopilot_state.json`` control
    fields consumed by ``scripts/autopilot/autopilot.py``. It does not kill
    processes, rewrite journals, or apply measurement decisions.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        result = _apply_autopilot_control_action(
            action=str(body.get("action") or ""),
            note=str(body.get("note") or ""),
        )
    except ValueError as exc:
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        logger.exception("dashboard autopilot control failed")
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=500)
    return JSONResponse(result, headers=_NO_STORE_HEADERS)


@router.get("/dashboard/api/repo_readiness")
async def repo_readiness() -> JSONResponse:
    """Passive repo-readiness queue summary for dashboard pickup.

    This intentionally exposes the root scorer's latest report as advisory
    planning input only. It is not an AutoPilot authority gate and does not
    mutate queue state.
    """
    return JSONResponse(
        _stamp(_repo_readiness_summary(), "repo_readiness"),
        headers=_NO_STORE_HEADERS,
    )


_INSIGHT_GRAPH_DEFAULT_LIMIT = 120
_INSIGHT_GRAPH_MAX_LIMIT = 240
_INSIGHT_GRAPH_DEFAULT_DEPTH = 1

_OPTIMIZATION_BRIEF_CACHE_TTL_S = 30.0
_OPTIMIZATION_BRIEF_CACHE: dict[str, Any] = {
    "created_monotonic": 0.0,
    "generated_at": None,
    "payload": None,
}
_OPTIMIZATION_BRIEF_LOCK = asyncio.Lock()


def _optimization_brief_cache_view(
    payload: dict[str, Any],
    *,
    generated_at: float,
    stale: bool = False,
    warning: str = "",
) -> dict[str, Any]:
    """Attach freshness without mutating the cached synthesis payload."""
    view = dict(payload)
    view["synthesis"] = {
        "generated_at": generated_at,
        "age_s": max(0.0, time.time() - generated_at),
        "stale": stale,
        "warning": warning,
    }
    return view


async def _optimization_brief_payload() -> dict[str, Any]:
    """Build the synthesis off the API event loop and cache reconnects."""
    now = time.monotonic()
    cached = _OPTIMIZATION_BRIEF_CACHE.get("payload")
    created = float(_OPTIMIZATION_BRIEF_CACHE.get("created_monotonic") or 0.0)
    generated_at = float(_OPTIMIZATION_BRIEF_CACHE.get("generated_at") or 0.0)
    if isinstance(cached, dict) and now - created < _OPTIMIZATION_BRIEF_CACHE_TTL_S:
        return _optimization_brief_cache_view(cached, generated_at=generated_at)

    async with _OPTIMIZATION_BRIEF_LOCK:
        now = time.monotonic()
        cached = _OPTIMIZATION_BRIEF_CACHE.get("payload")
        created = float(_OPTIMIZATION_BRIEF_CACHE.get("created_monotonic") or 0.0)
        generated_at = float(_OPTIMIZATION_BRIEF_CACHE.get("generated_at") or 0.0)
        if isinstance(cached, dict) and now - created < _OPTIMIZATION_BRIEF_CACHE_TTL_S:
            return _optimization_brief_cache_view(cached, generated_at=generated_at)

        from scripts.autopilot.optimization_brief import build_optimization_brief

        try:
            payload = await asyncio.to_thread(build_optimization_brief)
        except Exception as exc:
            if isinstance(cached, dict) and generated_at > 0:
                return _optimization_brief_cache_view(
                    cached,
                    generated_at=generated_at,
                    stale=True,
                    warning=str(exc),
                )
            raise
        generated_at = time.time()
        _OPTIMIZATION_BRIEF_CACHE.update(
            {
                "created_monotonic": time.monotonic(),
                "generated_at": generated_at,
                "payload": payload,
            }
        )
        return _optimization_brief_cache_view(payload, generated_at=generated_at)


@router.get("/dashboard/api/insight_graph")
async def insight_graph(
    focus: str | None = None,
    depth: int = _INSIGHT_GRAPH_DEFAULT_DEPTH,
    limit: int = _INSIGHT_GRAPH_DEFAULT_LIMIT,
) -> JSONResponse:
    """Return a bounded, read-only graph of StrategyStore and journal insights."""
    payload = _insight_graph_payload(focus=focus, depth=depth, limit=limit)
    # H2: shared autopilot-plane epoch (cross-panel coherence cross-check).
    payload["state_generation"] = _autopilot_state_generation()
    return JSONResponse(_stamp(payload, "insight_graph"), headers=_NO_STORE_HEADERS)


@router.get("/dashboard/api/optimization_brief")
async def optimization_brief() -> JSONResponse:
    """Read-only operator synthesis: what optimizes orchestrator performance.

    Aggregates the planner's assessments into a templated narrative, the
    incumbent best config, a lever ledger ranked by fANOVA importance +
    cluster-best value, and the explored boundary (ruled-out guardrails /
    queued hypotheses) — with a decision-grade vs observation trust banner.
    Never promotes or mutates; fails soft so the dashboard cannot 500 on it.
    """
    try:
        payload = await _optimization_brief_payload()
    except Exception as exc:  # noqa: BLE001 — synthesis must not break the page
        payload = {"read_only": True, "error": str(exc)}
    return JSONResponse(_stamp(payload, "optimization_brief"), headers=_NO_STORE_HEADERS)


# AP-50 decision cockpit. Computed per request from every journal shard + the
# Optuna study + state (cached briefly per era/status view), never from the
# generated digest. Read-only; the page lives on the hub (:8100/cockpit).
_DECISION_COCKPIT_CACHE_TTL_S = 30.0
_DECISION_COCKPIT_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_DECISION_COCKPIT_LOCK = asyncio.Lock()
_DECISION_COCKPIT_CACHE_MAX = 16


async def _decision_cockpit_payload(era: str | None, status: str | None) -> dict[str, Any]:
    key = (str(era or "current"), str(status or ""))
    now = time.monotonic()
    hit = _DECISION_COCKPIT_CACHE.get(key)
    if hit is not None and now - hit[0] < _DECISION_COCKPIT_CACHE_TTL_S:
        return hit[1]
    async with _DECISION_COCKPIT_LOCK:
        hit = _DECISION_COCKPIT_CACHE.get(key)
        if hit is not None and time.monotonic() - hit[0] < _DECISION_COCKPIT_CACHE_TTL_S:
            return hit[1]
        from scripts.autopilot.decision_cockpit import build_decision_cockpit

        orch = _AUTOPILOT_STATE_PATH.parent
        payload = await asyncio.to_thread(
            build_decision_cockpit,
            journal_dir=_AUTOPILOT_JOURNAL_PATH.parent,
            state_path=_AUTOPILOT_STATE_PATH,
            eras_path=orch / "instrument_eras.yaml",
            study_db=orch / "optuna_study.db",
            checkpoints_dir=orch / "autopilot_checkpoints",
            era=era,
            status_filter=status,
        )
        if len(_DECISION_COCKPIT_CACHE) >= _DECISION_COCKPIT_CACHE_MAX:
            _DECISION_COCKPIT_CACHE.clear()
        _DECISION_COCKPIT_CACHE[key] = (time.monotonic(), payload)
        return payload


@router.get("/dashboard/api/decision_cockpit")
async def decision_cockpit(era: str | None = None, status: str | None = None) -> JSONResponse:
    """AP-50: proposed/executed/valid/kept/promoted/live funnel, deltas, levers, provenance.

    Fails closed: a builder exception is reported as ``health.status=degraded`` with
    every section absent, never as an empty (clean-looking) cockpit.
    """
    try:
        payload = dict(await _decision_cockpit_payload(era, status))
    except Exception as exc:  # noqa: BLE001 — the page must render the failure, not 500
        payload = {
            "schema": "epyc.autopilot.decision_cockpit.v1",
            "read_only": True,
            "error": str(exc)[:300],
            "health": {"status": "degraded", "reasons": [f"builder failed: {exc}"[:300]]},
        }
    return JSONResponse(_stamp(payload, "decision_cockpit"), headers=_NO_STORE_HEADERS)


@router.get("/dashboard/api/decision_cockpit/health")
async def decision_cockpit_health() -> JSONResponse:
    """Data probe for the cockpit: HTTP 200 only for ``ok``; 503 for absent/degraded."""
    try:
        payload = await _decision_cockpit_payload(None, None)
        health = dict(payload.get("health") or {"status": "degraded", "reasons": ["no health fold"]})
        health["evidence_freshness"] = payload.get("evidence_freshness")
    except Exception as exc:  # noqa: BLE001
        health = {"status": "degraded", "reasons": [f"builder failed: {exc}"[:300]]}
    code = 200 if health.get("status") == "ok" else 503
    return JSONResponse(health, status_code=code, headers=_NO_STORE_HEADERS)


# ---------------------------------------------------------------------------
# Per-node detail (for topology click)
# ---------------------------------------------------------------------------


@router.get("/dashboard/api/node/{port}")
async def node_detail(port: int) -> JSONResponse:
    """Full detail for one topology node: health, slots, recent decisions routed to it."""
    label = _port_hint(port)
    # PID + cmd from ps
    proc_info: dict[str, Any] = {}
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-eo", "pid,etime,pcpu,pmem,rss,cmd"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        port_arg = f"--port {port}"
        for line in out.splitlines()[1:]:
            if port_arg in line and "grep" not in line:
                parts = line.split(None, 5)
                if len(parts) >= 6:
                    proc_info = {
                        "pid": int(parts[0]),
                        "etime": parts[1],
                        "pcpu_cumulative": float(parts[2]),
                        "pmem": float(parts[3]),
                        "rss_kb": int(parts[4]),
                        "cmd": parts[5][:400],
                    }
                    break
    except Exception:
        pass

    # Live /slots state
    slots_data: list[dict[str, Any]] = []
    health_status: str | None = None
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            try:
                resp = await client.get(f"http://127.0.0.1:{port}/slots")
                if resp.status_code == 200:
                    raw = resp.json()
                    if isinstance(raw, list):
                        for s in raw:
                            prompt = s.get("prompt", "") or ""
                            if isinstance(prompt, list):
                                prompt = " ".join(str(x) for x in prompt)
                            content = s.get("content", "") or ""
                            if isinstance(content, list):
                                content = " ".join(str(x) for x in content)
                            slots_data.append(
                                {
                                    "id": s.get("id"),
                                    "id_task": s.get("id_task"),
                                    "is_processing": s.get("is_processing"),
                                    "n_decoded": s.get("n_decoded"),
                                    "n_prompt_tokens": s.get("n_prompt_tokens"),
                                    "prompt_preview": prompt[:180],
                                    "content_preview": content[:200],
                                    "content_len": len(content),
                                }
                            )
                    health_status = "ok"
                else:
                    health_status = f"http {resp.status_code}"
            except httpx.ConnectError:
                health_status = "connection_refused"
            except httpx.TimeoutException:
                health_status = "timeout"
            except Exception as exc:
                health_status = f"error: {type(exc).__name__}"
    except Exception:
        pass

    # Recent decisions routed to this node's role (from progress JSONL)
    role_base = label.split(".")[0]
    log_path = _todays_progress_log()
    recent_routed: list[dict[str, Any]] = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("event_type") != "routing_decision":
                        continue
                    data = e.get("data", {})
                    if data.get("chosen_action") == role_base:
                        recent_routed.append(
                            {
                                "task_id": e.get("task_id"),
                                "timestamp": e.get("timestamp", ""),
                                "decision_source": data.get("decision_source"),
                                "classifier_confidence": data.get("classifier_confidence"),
                                "verifier_p_success": data.get("verifier_p_success"),
                            }
                        )
            recent_routed = recent_routed[-15:]
        except Exception:
            pass

    return JSONResponse(
        {
            "port": port,
            "label": label,
            "role": role_base,
            "color": _role_color(label),
            "process": proc_info,
            "health_status": health_status,
            "slots": slots_data,
            "n_slots": len(slots_data),
            "n_processing": sum(1 for s in slots_data if s.get("is_processing")),
            "recent_routed_count": len(recent_routed),
            "recent_routed": recent_routed,
        }
    )


# ---------------------------------------------------------------------------
# Topology endpoint
# ---------------------------------------------------------------------------


# Dashboard source paths used by the /version endpoint below to surface
# build state to the browser, so users can tell when a hard-reload is needed.
_DASHBOARD_HTML_FOR_VERSION = Path(__file__).parent / "dashboard.html"
_DASHBOARD_PY_FOR_VERSION = Path(__file__)
_REPO_ROOT_FOR_VERSION = Path(__file__).resolve().parents[3]
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}

# Fleet startup timestamp — read from the marker file written by
# scripts/server/orchestrator_stack.start_orchestrator() before uvicorn launches.
# Live measurement (2026-05-23) confirmed uvicorn forks workers BEFORE importing
# the app, so each worker imports dashboard.py independently. The per-worker
# `time.time()` fallback would produce different timestamps across workers
# (observed: ~15ms drift among the 6 workers), which would cause the autopilot
# watcher to detect spurious restarts when consecutive requests hit different
# workers. Reading the atomic-write marker file gives every worker the same
# value as long as the marker exists before uvicorn's Popen — which the launch
# script guarantees.
try:
    from scripts.server.fleet_markers import (
        read_orchestrator_marker_metadata as _read_orch_marker_metadata,
    )

    _marker_meta = _read_orch_marker_metadata()
    _SERVER_STARTED_AT = _marker_meta["started_at"] if _marker_meta is not None else time.time()
    _SERVER_LAUNCH_GIT_SHA = _marker_meta.get("git_sha") if _marker_meta is not None else None
except Exception:
    _SERVER_STARTED_AT = time.time()
    _SERVER_LAUNCH_GIT_SHA = None


def _read_git_short_sha() -> str | None:
    """Read current HEAD short SHA from .git/ — no subprocess.

    Handles both branch checkouts (.git/HEAD = "ref: refs/heads/main\n") and
    detached HEAD (.git/HEAD = "<sha>\n"). Returns None on any error.
    """
    try:
        head_file = _REPO_ROOT_FOR_VERSION / ".git" / "HEAD"
        head = head_file.read_text().strip()
        if head.startswith("ref: "):
            ref_path = _REPO_ROOT_FOR_VERSION / ".git" / head.split(" ", 1)[1].strip()
            if ref_path.exists():
                return ref_path.read_text().strip()[:7]
            # Fall back to packed-refs
            packed = _REPO_ROOT_FOR_VERSION / ".git" / "packed-refs"
            if packed.exists():
                target_ref = head.split(" ", 1)[1].strip()
                for line in packed.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and parts[1] == target_ref:
                        return parts[0][:7]
            return None
        # Detached HEAD: file directly contains the SHA
        return head[:7] if len(head) >= 7 else None
    except Exception:
        return None


_AUTOPILOT_STATE_PATH = (
    Path(__file__).resolve().parents[3] / "orchestration" / "autopilot_state.json"
)
_AUTOPILOT_JOURNAL_PATH = (
    Path(__file__).resolve().parents[3] / "orchestration" / "autopilot_journal.jsonl"
)
_STRATEGY_STORE_PATH = (
    Path(__file__).resolve().parents[3] / "orchestration" / "repl_memory" / "strategies"
)
_PLANNER_HINT_SEEDS_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "autopilot" / "operator_seed_strategies.yaml"
)
_AUTOPILOT_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
_AUTOPILOT_TMP_LOG_DIR = Path("/mnt/raid0/llm/tmp")
_AUTOPILOT_CONTROL_AUDIT_PATH = _AUTOPILOT_LOG_DIR / "autopilot_operator_control.jsonl"
_EPHEMERAL_ACTION_KEYS = EPHEMERAL_ACTION_KEYS
_WITHIN_NOISE_EXCL = WITHIN_NOISE_EXCLUSIONS
_BASELINE_PROMOTION_EVENT_TYPE = "baseline_promotion"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return loaded


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _autopilot_state_summary(
    *,
    state_path: Path = _AUTOPILOT_STATE_PATH,
) -> dict[str, Any]:
    try:
        state = _read_json_object(state_path)
    except Exception as exc:
        return {
            "exists": state_path.exists(),
            "error": str(exc),
            "paused": None,
        }
    return {
        "exists": state_path.exists(),
        "paused": bool(state.get("paused", False)),
        "pause_reason": state.get("pause_reason"),
        # Who holds the pause, and whether an automated pauser was superseded —
        # a collision that is detected but not shown is still a silent collision.
        "pause_owner": state.get("pause_owner"),
        "pause_collision": state.get("pause_collision"),
        "dispatch_deficiency": state.get("_dispatch_deficiency"),
        "trial_counter": state.get("trial_counter"),
        # RTG-55 MHS-3 operability: fail-closed leakage-guard status (vocabulary
        # availability, consecutive rejections, alarm state). None before the first
        # autopilot start that carries the monitor.
        "eval_leakage_guard": state.get("eval_leakage_guard"),
    }


def _apply_autopilot_control_action(
    *,
    action: str,
    note: str = "",
    state_path: Path = _AUTOPILOT_STATE_PATH,
    audit_path: Path = _AUTOPILOT_CONTROL_AUDIT_PATH,
) -> dict[str, Any]:
    action = str(action or "").strip().lower()
    if action not in {"pause", "resume"}:
        raise ValueError("action must be 'pause' or 'resume'")
    note = str(note or "").strip()[:240]
    # H4: this dashboard write is one of 5+ processes doing a whole-file
    # read-modify-write of autopilot_state.json under `uvicorn --workers 6`.
    # Atomic tmp+replace prevents torn reads but NOT lost updates: without mutual
    # exclusion this pause/resume, based on a stale read, can clobber (or be
    # clobbered by) the autopilot daemon's per-trial save or host_health's
    # cache-flush. Serialize the ENTIRE read→modify→write across processes with
    # the shared flock. Kept short (no sleeps/inference inside) per the lock
    # contract; the audit append below runs OUTSIDE the lock. Fails open.
    with state_write_lock(state_path):
        state = _read_json_object(state_path)
        paused_pre = bool(state.get("paused", False))
        reason_pre = state.get("pause_reason")

        collision: dict[str, Any] | None = None
        halt_latch: dict[str, Any] | None = None
        if action == "pause":
            # Same operator authority as `autopilot.py pause`: SUPERSEDE any
            # automated pause lease held by an in-flight config-apply or cache
            # flush, so that operation cannot resume AutoPilot behind this click
            # when it finishes. Without this the dashboard button is a bypass of
            # the CLI's interlock.
            collision = supersede_pause_lease(state, OPERATOR_PAUSE_OWNER)
            state["pause_reason"] = note or "dashboard operator pause"
        else:
            state["paused"] = False
            clear_pause_lease(state)
            state.pop("pause_reason", None)
            # Clearing the skip/meta halt latch is NOT ours to do unilaterally.
            # `consecutive_skip_actions`, `last_invalid_*` and
            # `consecutive_meta_actions` are daemon-owned: the AutoPilot daemon
            # holds them in one in-memory dict for the whole run and rewrites them
            # wholesale at its next save. This endpoint used to zero them anyway,
            # report "resume ok", and have the halt reappear seconds later on the
            # daemon's next save — silent success over a lost write. The gated
            # helper writes them only when nothing will overwrite us (daemon
            # absent) and otherwise reports that it delegated to the daemon, which
            # clears the latch from its own memory on the resume edge.
            halt_latch = clear_halt_latch(state_path, state)

        _atomic_write_json(state_path, state)
    row = {
        "ts": _utc_iso(),
        "source": "dashboard",
        "action": action,
        "note": note,
        "state_path": str(state_path),
        "paused_pre": paused_pre,
        "paused_post": bool(state.get("paused", False)),
        "pause_reason_pre": reason_pre,
        "pause_reason_post": state.get("pause_reason"),
        "pause_collision": collision,
        "halt_latch": halt_latch,
        "trial_counter": state.get("trial_counter"),
    }
    _append_jsonl(audit_path, row)
    return {
        "status": "ok",
        "action": action,
        "state_path": str(state_path),
        "audit_path": str(audit_path),
        "paused_pre": paused_pre,
        "paused": bool(state.get("paused", False)),
        "pause_reason": state.get("pause_reason"),
        "pause_collision": collision,
        # Whether the halt latch actually came off, and if not, who will clear it.
        # A resume that could not clear the latch must SAY SO here; the UI renders
        # `halt_latch_message`.
        "halt_latch": halt_latch,
        "halt_latch_message": halt_latch_message(halt_latch),
        "dispatch_deficiency": state.get("_dispatch_deficiency"),
        "trial_counter": state.get("trial_counter"),
    }


def _parse_journal_ts(value: Any) -> float | None:
    return core_parse_journal_ts(value)


def _pareto_objectives_from_journal(entry: dict[str, Any]) -> list[float] | None:
    return core_objectives_from_journal_row(entry)


def _config_fingerprint_from_row(row: dict[str, Any]) -> str:
    return core_config_fingerprint_from_row(row)


def _pareto_dominates(a: list[float], b: list[float]) -> bool:
    return core_pareto_dominates(a, b)


def _latest_journal_run_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return core_latest_journal_run_rows(rows)


def _effective_journal_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    folded_rows, _ = core_fold_supersession_events(rows)
    return [row for row in folded_rows if row.get("trial_id") is not None]


def _pareto_hypervolume(points: list[list[float]]) -> float:
    """Compatibility wrapper for tests; implementation lives in autopilot_core."""
    return _pareto_hypervolume_impl(points)


def _suite_metric_for_dashboard(
    entry: dict[str, Any],
    suite: str,
) -> dict[str, Any] | None:
    """Return compact suite-specific eval metrics for dashboard display."""
    details = entry.get("eval_details")
    if not isinstance(details, dict):
        return None
    nested_details = details.get("details")
    if not isinstance(nested_details, dict):
        nested_details = {}

    per_suite_quality = details.get("per_suite_quality")
    if not isinstance(per_suite_quality, dict):
        per_suite_quality = nested_details.get("per_suite_quality")
    if not isinstance(per_suite_quality, dict):
        per_suite_quality = {}

    per_suite_counts = details.get("per_suite_counts")
    if not isinstance(per_suite_counts, dict):
        per_suite_counts = nested_details.get("per_suite_counts")
    if not isinstance(per_suite_counts, dict):
        per_suite_counts = {}

    question_results = details.get("question_results")
    if not isinstance(question_results, list):
        question_results = []

    suite_rows = [
        row
        for row in question_results
        if isinstance(row, dict) and str(row.get("suite") or "") == suite
    ]
    quality = per_suite_quality.get(suite)
    count = per_suite_counts.get(suite)
    correct = sum(1 for row in suite_rows if row.get("correct") is True)
    if count is None and suite_rows:
        count = len(suite_rows)
    if quality is None and suite_rows:
        quality = 3.0 * correct / max(1, len(suite_rows))
    if quality is None and count is None:
        return None
    try:
        quality_value = float(quality) if quality is not None else None
    except (TypeError, ValueError):
        quality_value = None
    try:
        count_value = int(count) if count is not None else None
    except (TypeError, ValueError):
        count_value = None
    return {
        "suite": suite,
        "quality": quality_value,
        "count": count_value,
        "correct": correct if suite_rows else None,
    }


def _shape_pareto_entry(entry: dict[str, Any]) -> dict[str, Any]:
    # Strip heavy config_snapshot for transport — plotted points only need
    # objectives + identity metadata. Caller can drill via trial_id if needed.
    obj = entry.get("objectives") or [0.0, 0.0, 0.0, 0.0]
    if len(obj) < 4:
        obj = list(obj) + [0.0] * (4 - len(obj))
    real_suite = _suite_metric_for_dashboard(entry, "real_suite_v1")
    return {
        "trial_id": entry.get("trial_id"),
        "objectives": list(obj[:4]),
        "git_tag": entry.get("git_tag", ""),
        "species": entry.get("species", ""),
        "is_production_best": bool(entry.get("is_production_best", False)),
        "timestamp": entry.get("timestamp", ""),
        "reasoning": (entry.get("reasoning") or "")[:200],
        "eval_tier": entry.get("eval_tier", entry.get("tier", DEFAULT_FRONTIER_TIER)),
        "speed_deinflated": bool(entry.get("speed_deinflated", False)),
        "real_suite_v1": real_suite,
    }


# ── Pareto companion telemetry (display-only) ───────────────────────────────
# W3 was flipped on 2026-08-04: the archive policy, not this telemetry helper,
# owns live dominance. These fields let the dashboard compare rate/goodput and
# offered load without feeding the archive, safety gate, or keep/revert logic.

_CERTIFIED_RATE_OBJECTIVE_POLICIES = {
    PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
    RESOURCE_LANES_V2_RATE_4D_OBJECTIVE_POLICY,
    RATE_4D_OBJECTIVE_POLICY,
}
_TASK_RATE_NULL_FIELDS: dict[str, Any] = {
    "task_rate_qph": None,
    "goodput_qph": None,
    "tokens_per_solved": None,
    "offered_load": None,
    "task_rate_status": "unavailable",
    "task_rate_policy": None,
}


def _offered_load_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Offered-load descriptor for one journal row, or None when not recorded.

    W5 future-proofing (objective-task-rate-goodput.md): the next instrument era
    plans multi-profile evals (sparse-1 / steady-3 / burst-N offered load), and
    `task_rate` is only meaningful AT a given offered load — the same config can
    win closed-loop concurrency-3 while losing sparse single-request traffic.
    Shipping the descriptor per entry now means historical points remain
    attributable to their operating point once profiles diverge, instead of
    silently mixing task_rate numbers measured under different arrival patterns.
    """
    eval_details = row.get("eval_details")
    if not isinstance(eval_details, dict):
        return None
    details = eval_details.get("details")
    if not isinstance(details, dict):
        details = {}
    out: dict[str, Any] = {}
    concurrency = eval_details.get("eval_concurrency", details.get("eval_concurrency"))
    try:
        if concurrency is not None:
            out["eval_concurrency"] = int(concurrency)
    except (TypeError, ValueError):
        pass
    profile = details.get("test_profile")
    if isinstance(profile, str) and profile:
        out["profile"] = profile
    return out or None


def _task_rate_fields_from_row(row: dict[str, Any] | None) -> dict[str, Any]:
    """Certified task-rate telemetry fields for one folded journal row.

    Legacy/unstamped rows did not run under a task-rate objective instrument,
    so deriving q/h from whatever historical ``total`` and ``eval_wall_s``
    happen to exist is a counterfactual reconstruction, not measured evidence.
    Fail closed for those rows. Certified rate-policy rows use the same paired
    sequential rate helper as the live objective; aborted/implausibly short
    measurements therefore return ``None`` rather than giant fake rates.
    """
    if not isinstance(row, dict):
        return dict(_TASK_RATE_NULL_FIELDS)
    eval_details = row.get("eval_details")
    if not isinstance(eval_details, dict):
        eval_details = {}
    policy = str(eval_details.get("objective_policy_live") or row.get("objective_policy_live") or "")
    if policy not in _CERTIFIED_RATE_OBJECTIVE_POLICIES:
        return {
            **_TASK_RATE_NULL_FIELDS,
            "tokens_per_solved": _task_rate_tokens_per_solved(row),
            "offered_load": _offered_load_from_row(row),
            "task_rate_status": "legacy_or_unstamped_policy",
            "task_rate_policy": policy or None,
        }
    try:
        task_rate = seq_task_rate_qph_from_row(row)
    except Exception:
        task_rate = None
    if task_rate is None:
        return {
            **_TASK_RATE_NULL_FIELDS,
            "tokens_per_solved": _task_rate_tokens_per_solved(row),
            "offered_load": _offered_load_from_row(row),
            "task_rate_status": "invalid_measurement",
            "task_rate_policy": policy,
        }
    # RTG-23: `float(row.get("quality") or 0.0)` could not distinguish a MEASURED
    # zero from a row whose eval never ran, so an unmeasured trial rendered as a
    # real 0.00 goodput. Absence is now reported as absence.
    quality = quality_from_row(row)
    if quality is None:
        return {
            **_TASK_RATE_NULL_FIELDS,
            "task_rate_qph": round(task_rate, 2),
            "tokens_per_solved": _task_rate_tokens_per_solved(row),
            "offered_load": _offered_load_from_row(row),
            "task_rate_status": "quality_unmeasured",
            "task_rate_policy": policy,
        }
    return {
        "task_rate_qph": round(task_rate, 2),
        "goodput_qph": round((quality / 3.0) * task_rate, 2),
        "tokens_per_solved": _task_rate_tokens_per_solved(row),
        "offered_load": _offered_load_from_row(row),
        "task_rate_status": "measured",
        "task_rate_policy": policy,
    }


def _decision_question_mix_from_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the inner decision-question difficulty mix for one eval row.

    ``row["tier"]`` is the OUTER EvalTower lane (T1/T2/T3). The T1 decision
    core independently samples question difficulty tiers, currently under the
    equal-thirds policy. Keeping these concepts separate prevents a ``T1``
    lane glyph from falsely implying T1-only questions.
    """
    if not isinstance(row, dict):
        return None
    eval_details = row.get("eval_details")
    if not isinstance(eval_details, dict):
        return None
    details = eval_details.get("details")
    if not isinstance(details, dict):
        return None
    profile = details.get("test_profile")
    if not isinstance(profile, dict):
        profile = {}
    provenance = details.get("tier_mix_provenance")
    if not isinstance(provenance, dict):
        provenance = profile.get("tier_mix_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    raw_targets = provenance.get("tier_mix_targets")
    if not isinstance(raw_targets, dict):
        raw_targets = details.get("question_tier_mix")
    if not isinstance(raw_targets, dict):
        return None

    targets: dict[str, int] = {}
    for raw_tier, raw_count in raw_targets.items():
        try:
            tier = int(raw_tier)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if tier >= 0 and count >= 0:
            targets[str(tier)] = count
    if not targets:
        return None

    def _int_or_none(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    partition_counts = details.get("partition_counts")
    if not isinstance(partition_counts, dict):
        partition_counts = {}
    decision_filter = details.get("decision_partition_filter")
    if not isinstance(decision_filter, dict):
        decision_filter = {}
    target_n = _int_or_none(provenance.get("drawn_n"))
    if target_n is None:
        target_n = _int_or_none(profile.get("n_questions"))
    if target_n is None:
        target_n = _int_or_none(decision_filter.get("decision_n_questions"))
    if target_n is None:
        target_n = sum(targets.values())
    scored_n = _int_or_none(details.get("n_scored"))
    if scored_n is None:
        scored_n = _int_or_none(partition_counts.get("core"))
    return {
        "policy": str(provenance.get("tier_mix_policy") or "") or None,
        "targets": dict(sorted(targets.items(), key=lambda item: int(item[0]))),
        "target_n": target_n,
        "scored_n": scored_n,
        "core_id": str(details.get("core_id") or profile.get("core_id") or "") or None,
        "rotation_index": _int_or_none(provenance.get("core_rotation_index")),
    }


def _attach_task_rate_telemetry(
    entry_lists: Iterable[list[dict[str, Any]]],
    rows_by_tid: dict[int, dict[str, Any]],
) -> None:
    """Stamp shaped Pareto entries with W3b-C task-rate telemetry, in place.

    Shaped journal-reconstruction entries carry only objectives + identity, so
    the rate fields are re-joined from the SAME folded journal rows the panel
    already reconstructs from, keyed by trial_id (exactly how the replay tool
    joins its report tables).
    """
    for entries in entry_lists:
        for entry in entries:
            try:
                tid = int(entry.get("trial_id"))
            except (TypeError, ValueError):
                tid = None
            entry.update(
                _task_rate_fields_from_row(rows_by_tid.get(tid) if tid is not None else None)
            )


def _attach_decision_question_mix(
    entry_lists: Iterable[list[dict[str, Any]]],
    rows_by_tid: dict[int, dict[str, Any]],
) -> None:
    """Attach outer-lane-independent decision-question mix context."""
    for entries in entry_lists:
        for entry in entries:
            try:
                tid = int(entry.get("trial_id"))
            except (TypeError, ValueError):
                tid = None
            entry["decision_question_mix"] = _decision_question_mix_from_row(
                rows_by_tid.get(tid) if tid is not None else None
            )


_DASHBOARD_OBJECTIVE_POLICIES = {
    LEGACY_OBJECTIVE_POLICY,
    TASK_RATE_OBJECTIVE_POLICY,
    PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
    RESOURCE_LANES_V2_RATE_4D_OBJECTIVE_POLICY,
    RATE_4D_OBJECTIVE_POLICY,
}


def _active_pareto_objective_policy(state: dict[str, Any]) -> tuple[str, str | None]:
    """Return the state-declared live policy, failing visibly to legacy replay."""
    raw = str(state.get("pareto_objective_policy") or LEGACY_OBJECTIVE_POLICY)
    if raw in _DASHBOARD_OBJECTIVE_POLICIES:
        return raw, None
    return (
        LEGACY_OBJECTIVE_POLICY,
        f"unknown pareto_objective_policy={raw!r}; showing legacy comparator",
    )


def _objective_policy_comparison_summary(
    rows: list[dict[str, Any]] | None,
    state: dict[str, Any],
    active_policy: str,
) -> dict[str, Any]:
    """Compare the historical t/s frontier with the already-live objective.

    This is descriptive telemetry, not a flip gate: W3 was ratified on
    2026-08-04. The active policy comes from ``autopilot_state.json`` so the
    dashboard never compares legacy t/s against the obsolete 3D shadow policy
    while AutoPilot is actually ranking a newer 4D task-rate instrument.
    """
    out: dict[str, Any] = {
        "legacy_policy": LEGACY_OBJECTIVE_POLICY,
        "active_policy": active_policy,
        "canonical_tier": DEFAULT_FRONTIER_TIER,
        "legacy_frontier_trial_ids": [],
        "active_frontier_trial_ids": [],
        "dropped_legacy_count": 0,
        "added_active_count": 0,
        "frontiers_differ": False,
        "error": None,
    }
    if active_policy == LEGACY_OBJECTIVE_POLICY:
        return out
    if not rows:
        out["error"] = "no journal rows available"
        return out
    try:
        legacy = _task_rate_replay_archive(
            rows,
            state,
            objective_policy=LEGACY_OBJECTIVE_POLICY,
            current_run_only=True,
        )
        active = _task_rate_replay_archive(
            rows,
            state,
            objective_policy=active_policy,
            current_run_only=True,
        )
    except (Exception, SystemExit) as exc:  # SystemExit: replay raises on empty archive
        out["error"] = f"divergence replay failed: {exc}"
        return out

    def _frontier_ids(archive: dict[str, Any]) -> list[int]:
        ids: set[int] = set()
        for entry in _task_rate_replay_frontier(archive):
            try:
                ids.add(int(entry.get("trial_id")))
            except (TypeError, ValueError):
                continue
        return sorted(ids)

    legacy_ids = _frontier_ids(legacy)
    active_ids = _frontier_ids(active)
    dropped = sorted(set(legacy_ids) - set(active_ids))
    added = sorted(set(active_ids) - set(legacy_ids))
    out.update(
        {
            "legacy_frontier_trial_ids": legacy_ids,
            "active_frontier_trial_ids": active_ids,
            "dropped_legacy_trial_ids": dropped,
            "dropped_legacy_count": len(dropped),
            "added_active_trial_ids": added,
            "added_active_count": len(added),
            "frontiers_differ": legacy_ids != active_ids,
        }
    )
    return out


# ── Instrument-era display regions (all-era Pareto view) ────────────────────
# The append-only era registry (orchestration/instrument_eras.yaml, human-owned)
# is the source of truth for metric-comparability boundaries. The all-era Pareto
# view labels every point with the era region it falls in, so the operator sees
# the full performance progression without silently mixing instruments: the one
# codified read-time rescale (pre-E2 speed deinflation) is applied; every later
# boundary (e.g. the E5 v6+iqk kernel cutover) is labeled, never rescaled.
_PARETO_ERA_SCOPES = {"autopilot_speed", "autopilot_quality"}
# The only era boundary with a codified read-time rescale: speeds journaled
# before E2 (2026-06-01 double-count fix) multiply by
# `pareto_pre_epoch_speed_factor` (state knob, default 0.5).
_PARETO_SPEED_DEINFLATE_ERA_ID = "E2"


def _era_short_label(era_id: str) -> str:
    """Compact chart label for a registry era id: 'E5-autopilot-speed' → 'E5'."""
    m = re.match(r"(E\d+[a-z]?)", str(era_id))
    return m.group(1) if m else str(era_id)


def _autopilot_era_regions() -> tuple[list[dict[str, Any]], str | None]:
    """Chronological era regions for the autopilot Pareto plots.

    Reads the instrument-era registry and slices time at every autopilot-scoped
    `from` boundary. Region 0 is a synthetic 'pre-<first era>' interval so rows
    older than the earliest registered boundary still get a label. Returns
    (regions, error); on any registry problem the regions are empty and the
    error string is surfaced to the client instead of guessing boundaries.
    """
    try:
        data = yaml.safe_load(instrument_eras_path().read_text())
    except Exception as exc:
        return [], f"instrument-era registry unavailable: {exc}"
    if not isinstance(data, dict) or not isinstance(data.get("eras"), list):
        return [], "instrument-era registry must contain an eras list"

    boundaries: dict[float, list[str]] = {}
    for row in data["eras"]:
        if not isinstance(row, dict):
            continue
        if str(row.get("scope", "")).strip() not in _PARETO_ERA_SCOPES:
            continue
        from_ts = core_parse_era_epoch(row.get("from"))
        if from_ts is None:
            continue
        boundaries.setdefault(from_ts, []).append(str(row.get("id", "")))
    if not boundaries:
        return [], "no autopilot-scoped era rows with a from boundary in registry"

    ordered = sorted(boundaries.items())
    first_label = "+".join(sorted({_era_short_label(i) for i in ordered[0][1]}))
    regions: list[dict[str, Any]] = [
        {
            "index": 0,
            "id": f"pre-{first_label}",
            "era_ids": [],
            "from_ts": None,
            "until_ts": ordered[0][0],
        }
    ]
    for i, (ts, ids) in enumerate(ordered):
        regions.append(
            {
                "index": i + 1,
                "id": "+".join(sorted({_era_short_label(era_id) for era_id in ids})),
                "era_ids": ids,
                "from_ts": ts,
                "until_ts": ordered[i + 1][0] if i + 1 < len(ordered) else None,
            }
        )
    return regions, None


def _era_region_index_for_ts(regions: list[dict[str, Any]], ts: float | None) -> int | None:
    if ts is None:
        return None
    for region in regions:
        start, end = region.get("from_ts"), region.get("until_ts")
        if (start is None or ts >= start) and (end is None or ts < end):
            return region["index"]
    return None


def _label_pareto_entries_with_eras(
    entries: list[dict[str, Any]], regions: list[dict[str, Any]]
) -> None:
    """Stamp each shaped entry with the era region its timestamp falls in.

    Era honesty (quality axis, 2026-07-26): entries from any region OTHER than
    the latest one additionally carry ``historical_instrument: True`` — they
    were measured under an earlier instrument era and are historical priors,
    not comparable to current-era points. Label ONLY — values are NEVER
    rescaled here (the sole rescale is the codified pre-E2 speed deinflation
    applied upstream). Entries with no parsable timestamp get ``None`` (era
    unknown), never ``False``, so the client cannot mistake them for
    current-instrument points.
    """
    latest_index = max((r["index"] for r in regions), default=None)
    for entry in entries:
        idx = _era_region_index_for_ts(regions, _parse_journal_ts(entry.get("timestamp")))
        entry["era_index"] = idx
        entry["era"] = regions[idx]["id"] if idx is not None else None
        entry["historical_instrument"] = (
            (idx != latest_index) if idx is not None and latest_index is not None else None
        )


def _autopilot_journal_shards() -> list[Path]:
    """All autopilot journal shards, oldest→newest.

    The autopilot rotates its journal on restart — trial 999 ended
    ``autopilot_journal.jsonl`` and the live run continued into
    ``autopilot_journal_1.jsonl`` at trial 1000+ — matching the multi-path
    convention already used by ``optimization_brief.DEFAULT_JOURNAL_PATHS``. The
    dashboard must read the base file PLUS every ``autopilot_journal_<n>.jsonl``
    rotation, or its journal-derived panels (gepa, pareto frontier, trial
    progress) silently freeze at the last trial in the base file. That is exactly
    how the gepa/frontier panels sat at trial 999 for days while the live run
    advanced past 1073 in ``_1`` — a stale-panel bug the freshness health check
    surfaced. Sorted by numeric rotation suffix so the merge preserves trial
    order even past ``_9`` → ``_10``; the un-suffixed base file sorts first. Only
    pure numeric suffixes are included — snapshot/backup files with other
    suffixes are ignored.
    """
    base = _AUTOPILOT_JOURNAL_PATH
    stem = base.stem  # "autopilot_journal"
    shard_re = re.compile(rf"{re.escape(stem)}_(\d+)\.jsonl$")
    ordered: list[tuple[int, Path]] = []
    try:
        candidates = list(base.parent.glob(f"{stem}*.jsonl"))
    except OSError:
        candidates = []
    for p in candidates:
        if p.name == base.name:
            ordered.append((-1, p))
            continue
        m = shard_re.match(p.name)
        if m:
            ordered.append((int(m.group(1)), p))
    return [p for _, p in sorted(ordered, key=lambda t: t[0])]


def _read_autopilot_journal_rows(path: Path | None = None) -> list[dict[str, Any]] | None:
    # With no explicit path, read ALL journal shards (base + rotations) in trial
    # order so journal-derived panels follow the live run across a rotation
    # instead of freezing at the last trial in the base file. An explicit path
    # keeps single-file behaviour for callers that want exactly one shard.
    paths = [path] if path is not None else _autopilot_journal_shards()
    rows: list[dict[str, Any]] = []
    any_read = False
    for p in paths:
        if not p or not p.exists():
            continue
        try:
            with open(p) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
            any_read = True
        except Exception:
            continue
    return rows if any_read else None


# ── H2: shared snapshot epoch for the autopilot-state plane ──────────────────
# The pareto / autopilot_progress / process_status / insight_graph panels each
# INDEPENDENTLY re-read autopilot_state.json + the rotating journal at their OWN
# request instant, with no shared generation token — so panel A (trial N) can
# render beside panel B (trial N-1): a cross-panel tear. The prior "coherent
# snapshot" fix (_snapshot_impl) covered only the LIVE-INFERENCE plane. These
# helpers give the autopilot plane a single, cheap, stat-based generation token
# so the four panels can be cross-checked / assembled into one coherent frame.


def _newest_journal_shard_stat() -> tuple[Path, int, int] | None:
    """``(path, st_size, st_mtime_ns)`` of the newest journal shard, or ``None``.

    Picks the shard with the newest mtime across the base file + every
    ``autopilot_journal_<n>.jsonl`` rotation (the live run appends to the
    highest-suffix shard, but choosing by mtime is robust to a base file touched
    after a rotation). Pure ``stat()`` — no file-content read.
    """
    best: tuple[Path, int, int] | None = None
    for p in _autopilot_journal_shards():
        try:
            st = p.stat()
        except OSError:
            continue
        if best is None or st.st_mtime_ns > best[2]:
            best = (p, st.st_size, st.st_mtime_ns)
    return best


def _autopilot_state_generation(*, state_path: Path | None = None) -> str:
    """Shared coherence token for the autopilot-state plane (H2).

    Combines ``autopilot_state.json``'s ``(st_mtime_ns, st_size)`` with the
    NEWEST journal shard's ``(name, size, st_mtime_ns)`` into one opaque string.
    Any write to state.json OR any append/rotation of the journal changes the
    token, so the four autopilot panels — which each re-read these files
    independently — can be cross-checked: a client holding two panels whose
    ``state_generation`` tokens DIFFER knows it has a TORN set (panel A at trial
    N beside panel B at trial N-1) and must discard/refetch. Cheap: pure
    ``stat()``, never reads the file bodies. ``state_path`` resolves the module
    global at CALL time so tests (and any future path override) take effect.
    """
    state_path = state_path if state_path is not None else _AUTOPILOT_STATE_PATH
    try:
        st = state_path.stat()
        state_tok = f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        state_tok = "absent"
    shard = _newest_journal_shard_stat()
    if shard is None:
        journal_tok = "absent"
    else:
        p, size, mtime_ns = shard
        journal_tok = f"{p.name}:{size}:{mtime_ns}"
    return f"state={state_tok}|journal={journal_tok}"


def _journal_max_trial_id(rows: list[dict[str, Any]] | None) -> int | None:
    """Highest integer ``trial_id`` across ``rows`` (the append-only journal)."""
    best: int | None = None
    for row in rows or []:
        try:
            tid = int(row.get("trial_id"))
        except (TypeError, ValueError):
            continue
        if best is None or tid > best:
            best = tid
    return best


def _state_trial_counter(*, state_path: Path | None = None) -> int | None:
    """``trial_counter`` from autopilot_state.json as an int, or ``None``.

    Resolves the module global at CALL time so a monkeypatched path takes effect.
    """
    state_path = state_path if state_path is not None else _AUTOPILOT_STATE_PATH
    try:
        state = _read_json_object(state_path)
    except Exception:
        return None
    try:
        return int(state.get("trial_counter"))
    except (TypeError, ValueError):
        return None


def _autopilot_snapshot_sources() -> list["_FreshnessSource"]:
    """Freshness sources for the combined autopilot frame: state + journal.

    Thresholds mirror the ``autopilot_progress`` / ``gepa`` panel specs so the
    combined frame's staleness badge agrees with the per-panel ones.
    """
    try:
        state_m: float | None = _AUTOPILOT_STATE_PATH.stat().st_mtime
    except OSError:
        state_m = None
    shard = _newest_journal_shard_stat()
    journal_m = (shard[2] / 1e9) if shard is not None else None
    return [
        _FreshnessSource("autopilot_state", state_m, 300, 1800),
        _FreshnessSource("autopilot_journal", journal_m, 600, 3600),
    ]


def _read_strategy_store_rows(path: Path | None = None) -> list[dict[str, Any]] | None:
    """Read StrategyStore rows from the on-disk SQLite mirror in read-only mode."""
    store_dir = path or _STRATEGY_STORE_PATH
    db_path = store_dir / "strategies.db"
    if not db_path.exists():
        return None

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.5)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA query_only = 1")
        except Exception:
            pass
        rows: list[dict[str, Any]] = []
        for row in conn.execute(
            "SELECT id, description, insight, source_trial_id, species, created_at, "
            "metadata_json, entry_type, evidence_trial_ids "
            "FROM strategies ORDER BY created_at ASC"
        ):
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            try:
                evidence_trial_ids_raw = json.loads(row["evidence_trial_ids"] or "[]")
            except Exception:
                evidence_trial_ids_raw = []
            evidence_trial_ids: list[int] = []
            if isinstance(evidence_trial_ids_raw, list):
                for item in evidence_trial_ids_raw:
                    try:
                        evidence_trial_ids.append(int(item))
                    except (TypeError, ValueError):
                        continue
            rows.append(
                {
                    "id": row["id"],
                    "description": row["description"],
                    "insight": row["insight"],
                    "source_trial_id": row["source_trial_id"],
                    "species": row["species"],
                    "created_at": row["created_at"],
                    "metadata": metadata,
                    "entry_type": row["entry_type"] or "raw",
                    "evidence_trial_ids": evidence_trial_ids,
                }
            )
        return rows
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _read_planner_hint_seed_rows(path: Path | None = None) -> list[dict[str, Any]] | None:
    """Read planner-hint seed rows from the curated YAML source."""
    seed_path = path or _PLANNER_HINT_SEEDS_PATH
    if not seed_path.exists():
        return None
    try:
        import yaml
    except Exception:
        return None

    try:
        loaded = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(loaded, list):
        return None

    rows: list[dict[str, Any]] = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        insight = str(item.get("insight") or "").strip()
        source_handoff = str(item.get("source_handoff") or "").strip()
        if not slug or not title or not description or not insight or not source_handoff:
            continue
        evidence_trial_ids: list[int] = []
        evidence_raw = item.get("evidence_trial_ids")
        if isinstance(evidence_raw, list):
            for evidence_id in evidence_raw:
                try:
                    evidence_trial_ids.append(int(evidence_id))
                except (TypeError, ValueError):
                    continue
        bind_identifiers_raw = item.get("bind_identifiers")
        bind_identifiers = (
            [str(value).strip() for value in bind_identifiers_raw if str(value).strip()]
            if isinstance(bind_identifiers_raw, list)
            else []
        )
        rows.append(
            {
                "id": f"seed:{slug}",
                "description": description,
                "insight": insight,
                "source_trial_id": None,
                "species": str(item.get("species") or "").strip(),
                "created_at": "",
                "metadata": {
                    "tranche": str(item.get("tranche") or "").strip(),
                    "confidence": str(item.get("confidence") or "").strip(),
                    "bind_status": str(item.get("bind_status") or "").strip().lower(),
                    "bind_identifiers": bind_identifiers,
                    "seed_campaign": slug,
                    "source_handoff": source_handoff,
                    "seeded_reason": str(item.get("seeded_reason") or "").strip(),
                    "seeded_from": str(seed_path.name),
                },
                "entry_type": str(item.get("entry_type") or "pattern").strip() or "pattern",
                "evidence_trial_ids": evidence_trial_ids,
                "planner_hint": True,
                "slug": slug,
                "title": title,
            }
        )
    return rows


def _insight_graph_payload(
    *,
    focus: str | None = None,
    depth: int = _INSIGHT_GRAPH_DEFAULT_DEPTH,
    limit: int = _INSIGHT_GRAPH_DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Build a bounded, read-only graph over journal rows and StrategyStore data."""
    try:
        depth = max(0, min(int(depth), 2))
    except (TypeError, ValueError):
        depth = _INSIGHT_GRAPH_DEFAULT_DEPTH
    try:
        limit = max(24, min(int(limit), _INSIGHT_GRAPH_MAX_LIMIT))
    except (TypeError, ValueError):
        limit = _INSIGHT_GRAPH_DEFAULT_LIMIT

    journal_rows = _read_autopilot_journal_rows()
    journal_rows = journal_rows or []
    journal_rows, journal_meta = _latest_journal_run_rows(
        _effective_journal_trial_rows(journal_rows)
    )
    strategy_rows = _read_strategy_store_rows() or []
    planner_hint_rows = _read_planner_hint_seed_rows() or []
    if strategy_rows:
        graph_rows = [{**row, "_graph_kind": "strategy"} for row in strategy_rows]
        graph_source = "strategy_store"
    elif planner_hint_rows:
        graph_rows = [{**row, "_graph_kind": "planner_hint"} for row in planner_hint_rows]
        graph_source = "planner_hint_seed"
    else:
        graph_rows = []
        graph_source = "empty"

    def _compact(value: Any, *, max_chars: int = 140) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

    def _strategy_status(metadata: dict[str, Any], entry_type: str) -> tuple[str, str]:
        bind_status = str(metadata.get("bind_status") or "").strip().lower()
        generated_from = str(metadata.get("generated_from") or "").strip().lower()
        tranche = str(metadata.get("tranche") or "").strip().lower()
        if generated_from == "journal_frontier":
            return "applied", "applied"
        if bind_status == "live":
            return "live", "applied"
        if bind_status in {"context", "future"}:
            return bind_status, "pending"
        if tranche == "guardrail":
            return "guardrail", "guardrail"
        if tranche == "frozen":
            return "frozen", "frozen"
        if entry_type == "convention":
            return "convention", "applied"
        if entry_type == "pattern":
            return "pattern", "applied"
        return entry_type or "raw", "raw"

    def _journal_status(row: dict[str, Any]) -> tuple[str, str]:
        status = str(row.get("pareto_status") or "journal").strip().lower()
        if row.get("bug_corrupted_by"):
            return "corrupted", "corrupted"
        if status == "frontier":
            return "frontier", "frontier"
        try:
            tier = int(row.get("tier", DEFAULT_FRONTIER_TIER))
        except (TypeError, ValueError):
            tier = DEFAULT_FRONTIER_TIER
        if tier < 1:
            return "audit", "audit"
        return status or "journal", status or "journal"

    def _state_color(kind: str, state_group: str) -> str:
        if kind == "journal":
            return {
                "frontier": "var(--good, #34d399)",
                "audit": "var(--warn, #fbbf24)",
                "corrupted": "var(--bad, #f87171)",
                "dominated": "var(--dim, #64748b)",
            }.get(state_group, "var(--muted, #94a3b8)")
        if kind in {"strategy", "planner_hint"}:
            return {
                "applied": "var(--accent, #38bdf8)",
                "pending": "var(--warn, #fbbf24)",
                "live": "var(--good, #34d399)",
                "context": "var(--muted, #94a3b8)",
                "guardrail": "var(--accent, #38bdf8)",
                "frozen": "var(--dim, #64748b)",
                "future": "var(--dim, #64748b)",
                "pattern": "var(--good, #34d399)",
                "convention": "var(--accent, #38bdf8)",
                "raw": "var(--dim, #64748b)",
            }.get(state_group, "var(--accent, #38bdf8)")
        if kind == "campaign":
            return "var(--accent, #8b5cf6)"
        if kind == "handoff":
            return "var(--bad, #fb7185)"
        return "var(--muted, #94a3b8)"

    def _node_label_text(node: dict[str, Any]) -> str:
        if node["kind"] == "journal":
            return f"T{node.get('trial_id')}"
        if node["kind"] in {"strategy", "planner_hint"}:
            return node.get("title") or node.get("description") or node["id"]
        if node["kind"] == "campaign":
            return node.get("campaign") or node["id"]
        if node["kind"] == "handoff":
            return node.get("handoff") or node["id"]
        return node["id"]

    def _node_sort_key(node: dict[str, Any]) -> tuple[Any, ...]:
        kind_rank = {
            "planner_hint": 0,
            "strategy": 1,
            "campaign": 2,
            "handoff": 3,
            "journal": 4,
        }.get(node["kind"], 5)
        if node["kind"] == "journal":
            try:
                recency = -int(node.get("trial_id") or 0)
            except (TypeError, ValueError):
                recency = 0
        else:
            recency = str(node.get("created_at") or "")
        return (
            kind_rank,
            node.get("distance", 99),
            -node.get("degree", 0),
            recency,
            node.get("label", ""),
        )

    def _node_matches_focus(node: dict[str, Any], focus_text: str) -> bool:
        haystack: list[str] = [
            node.get("id", ""),
            node.get("label", ""),
            node.get("title", ""),
            node.get("summary", ""),
            node.get("state", ""),
            node.get("state_group", ""),
            node.get("kind", ""),
            node.get("slug", ""),
            node.get("species", ""),
            node.get("entry_type", ""),
            node.get("campaign", ""),
            node.get("handoff", ""),
            node.get("bind_status", ""),
            node.get("source_handoff", ""),
            node.get("seed_campaign", ""),
            node.get("seeded_reason", ""),
            node.get("tranche", ""),
        ]
        trial_id = node.get("trial_id")
        if trial_id is not None:
            haystack.append(str(trial_id))
        source_trial_id = node.get("source_trial_id")
        if source_trial_id is not None:
            haystack.append(str(source_trial_id))
        return any(
            focus_text == text.lower() or focus_text in text.lower() for text in haystack if text
        )

    def _focus_match_key(
        node_id: str, node: dict[str, Any], focus_text: str
    ) -> tuple[int, tuple[Any, ...]]:
        exact_fields = [
            node_id,
            node.get("slug", ""),
            node.get("label", ""),
            node.get("title", ""),
            node.get("seed_campaign", ""),
            node.get("source_handoff", ""),
        ]
        exact = any(focus_text == str(field).lower() for field in exact_fields if field)
        kind_rank = {
            "campaign": 0,
            "handoff": 1,
            "planner_hint": 2,
            "strategy": 3,
            "journal": 4,
        }.get(node.get("kind"), 5)
        return (0 if exact else 1, (kind_rank, *_node_sort_key(node)))

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    adjacency: dict[str, set[str]] = {}
    journal_by_trial: dict[int, dict[str, Any]] = {}
    strategy_by_id: dict[str, dict[str, Any]] = {}
    strategy_ids_by_trial: dict[int, list[str]] = {}
    strategy_ids_by_campaign: dict[str, list[str]] = {}
    strategy_ids_by_handoff: dict[str, list[str]] = {}

    def _add_node(node: dict[str, Any]) -> None:
        node_id = node["id"]
        existing = nodes.get(node_id)
        if existing is None:
            nodes[node_id] = node
        else:
            existing.update({k: v for k, v in node.items() if v not in (None, "", [], {}, ())})

    def _add_edge(
        source: str, target: str, kind: str, label: str = "", weight: float = 1.0
    ) -> None:
        if not source or not target or source == target:
            return
        key = (source, target, kind)
        if key in edges:
            return
        edge = {
            "id": f"{source}->{target}:{kind}",
            "source": source,
            "target": target,
            "kind": kind,
            "label": label,
            "weight": weight,
            "color": {
                "projection": "var(--good, #34d399)",
                "evidence": "var(--accent, #38bdf8)",
                "campaign": "var(--accent, #a78bfa)",
                "handoff": "var(--bad, #fb7185)",
                "supersedes": "var(--warn, #fbbf24)",
                "parent": "var(--muted, #94a3b8)",
            }.get(kind, "var(--muted, #94a3b8)"),
        }
        edges[key] = edge
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    # Journal nodes first so the strategy edges can hook into them.
    for row in journal_rows:
        trial_id = row.get("trial_id")
        try:
            trial_id_int = int(trial_id)
        except (TypeError, ValueError):
            continue
        journal_by_trial[trial_id_int] = row
        status, state_group = _journal_status(row)
        summary_parts = [
            _compact(
                row.get("hypothesis") or row.get("reasoning") or row.get("expected_mechanism") or ""
            ),
            _compact(row.get("failure_analysis") or row.get("self_criticism") or ""),
        ]
        summary = " · ".join(part for part in summary_parts if part)
        node = {
            "id": f"journal:{trial_id_int}",
            "kind": "journal",
            "label": f"T{trial_id_int}",
            "title": f"trial {trial_id_int}",
            "subtitle": f"{_compact(row.get('species') or 'unknown')} · {_compact(row.get('action_type') or 'trial')} · {status}",
            "summary": summary,
            "trial_id": trial_id_int,
            "species": row.get("species") or "",
            "action_type": row.get("action_type") or "",
            "state": status,
            "state_group": state_group,
            "created_at": row.get("timestamp") or "",
            "color": _state_color("journal", state_group),
            "depth": 99,
            "degree": 0,
            "score": float(row.get("trial_id") or 0),
            "detail": {
                "quality": row.get("quality"),
                "speed": row.get("speed"),
                "pareto_status": row.get("pareto_status"),
                "tier": row.get("tier"),
            },
            "raw": row,
        }
        _add_node(node)

    # Planner-hint / StrategyStore nodes and the campaign/handoff buckets they belong to.
    for row in graph_rows:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        row_kind = str(row.get("_graph_kind") or "strategy")
        status, state_group = _strategy_status(metadata, row.get("entry_type") or "raw")
        source_trial_id = row.get("source_trial_id")
        try:
            source_trial_id_int = int(source_trial_id) if source_trial_id is not None else None
        except (TypeError, ValueError):
            source_trial_id_int = None
        campaign = str(metadata.get("seed_campaign") or "").strip()
        source_handoff = str(metadata.get("source_handoff") or "").strip()
        bind_status = str(metadata.get("bind_status") or "").strip().lower()
        bind_identifiers = (
            metadata.get("bind_identifiers")
            if isinstance(metadata.get("bind_identifiers"), list)
            else []
        )
        title = _compact((metadata.get("insight_format") or {}).get("title"))
        if not title:
            title = _compact(row.get("description") or row["id"], max_chars=72)
        summary = _compact(
            (metadata.get("insight_format") or {}).get("generalized_content")
            or row.get("insight")
            or row.get("description")
        )
        evidence_ids = list(row.get("evidence_trial_ids") or [])
        node = {
            "id": f"{row_kind}:{row['id']}",
            "kind": row_kind,
            "label": title,
            "title": title,
            "subtitle": " · ".join(
                part
                for part in (
                    _compact(row.get("entry_type") or "raw"),
                    _compact(row.get("species") or "unknown"),
                    _compact(metadata.get("tranche") or ""),
                    status,
                )
                if part
            ),
            "summary": summary,
            "state": status,
            "state_group": state_group,
            "entry_type": row.get("entry_type") or "raw",
            "species": row.get("species") or "",
            "source_trial_id": source_trial_id_int,
            "created_at": row.get("created_at") or "",
            "seed_campaign": campaign,
            "source_handoff": source_handoff,
            "bind_status": bind_status,
            "bind_identifiers": list(bind_identifiers),
            "confidence": metadata.get("confidence"),
            "tranche": metadata.get("tranche"),
            "seeded_reason": metadata.get("seeded_reason"),
            "color": _state_color("strategy", state_group),
            "depth": 99,
            "degree": 0,
            "score": float(source_trial_id_int or 0),
            "evidence_trial_ids": evidence_ids,
            "detail": {
                "bind_status": bind_status,
                "seeded_by": metadata.get("seeded_by"),
                "seeded_date": metadata.get("seeded_date"),
                "seeded_reason": metadata.get("seeded_reason"),
            },
            "raw": row,
        }
        _add_node(node)
        strategy_by_id[row["id"]] = node
        if source_trial_id_int is not None:
            strategy_ids_by_trial.setdefault(source_trial_id_int, []).append(row["id"])
        if campaign:
            strategy_ids_by_campaign.setdefault(campaign, []).append(row["id"])
        if source_handoff:
            strategy_ids_by_handoff.setdefault(source_handoff, []).append(row["id"])

    # Campaign and handoff nodes are the clickable buckets that make the staged
    # operator seeds explorable without needing to visualize every seed row as a
    # separate cluster head.
    for campaign, ids in strategy_ids_by_campaign.items():
        members = [
            strategy_by_id.get(strategy_id)
            for strategy_id in ids
            if strategy_by_id.get(strategy_id)
        ]
        cluster_label = (
            "planner hint rows"
            if any((member or {}).get("kind") == "planner_hint" for member in members)
            else "strategy rows"
        )
        node_id = f"campaign:{campaign}"
        _add_node(
            {
                "id": node_id,
                "kind": "campaign",
                "label": campaign,
                "title": campaign,
                "subtitle": f"{len(members)} {cluster_label}",
                "summary": _compact(
                    next(
                        (
                            (member.get("detail") or {}).get("seeded_reason")
                            for member in members
                            if (member.get("detail") or {}).get("seeded_reason")
                        ),
                        "",
                    )
                ),
                "state": "campaign",
                "state_group": "campaign",
                "color": _state_color("campaign", "campaign"),
                "degree": 0,
                "depth": 99,
                "score": float(len(members)),
                "detail": {"member_count": len(members)},
            }
        )
        for strategy_id in ids:
            strategy_node = strategy_by_id.get(strategy_id) or {}
            _add_edge(
                node_id,
                strategy_node.get("id", f"strategy:{strategy_id}"),
                "campaign",
                label="seed campaign",
                weight=1.4,
            )
            source_trial_id = strategy_node.get("source_trial_id")
            if source_trial_id is not None and source_trial_id in journal_by_trial:
                _add_edge(
                    node_id,
                    f"journal:{source_trial_id}",
                    "projection",
                    label="source trial",
                    weight=0.9,
                )

    for handoff, ids in strategy_ids_by_handoff.items():
        members = [
            strategy_by_id.get(strategy_id)
            for strategy_id in ids
            if strategy_by_id.get(strategy_id)
        ]
        cluster_label = (
            "planner hint rows"
            if any((member or {}).get("kind") == "planner_hint" for member in members)
            else "strategy rows"
        )
        node_id = f"handoff:{handoff}"
        _add_node(
            {
                "id": node_id,
                "kind": "handoff",
                "label": handoff,
                "title": handoff,
                "subtitle": f"{len(members)} {cluster_label}",
                "summary": _compact(
                    next(
                        (
                            (member.get("detail") or {}).get("seeded_reason")
                            for member in members
                            if (member.get("detail") or {}).get("seeded_reason")
                        ),
                        "",
                    )
                ),
                "state": "handoff",
                "state_group": "handoff",
                "color": _state_color("handoff", "handoff"),
                "degree": 0,
                "depth": 99,
                "score": float(len(members)),
                "detail": {"member_count": len(members)},
            }
        )
        for strategy_id in ids:
            strategy_node = strategy_by_id.get(strategy_id) or {}
            _add_edge(
                node_id,
                strategy_node.get("id", f"strategy:{strategy_id}"),
                "handoff",
                label="source handoff",
                weight=1.2,
            )
            source_trial_id = strategy_node.get("source_trial_id")
            if source_trial_id is not None and source_trial_id in journal_by_trial:
                _add_edge(
                    node_id,
                    f"journal:{source_trial_id}",
                    "projection",
                    label="source trial",
                    weight=0.9,
                )

    # Link strategy rows to their journal evidence and source trial when present.
    for node in strategy_by_id.values():
        source_trial_id = node.get("source_trial_id")
        if source_trial_id is not None and source_trial_id in journal_by_trial:
            _add_edge(
                f"journal:{source_trial_id}",
                node["id"],
                "projection",
                label="source trial",
                weight=1.6,
            )
        for evidence_trial_id in node.get("evidence_trial_ids") or []:
            if evidence_trial_id in journal_by_trial:
                _add_edge(
                    f"journal:{evidence_trial_id}",
                    node["id"],
                    "evidence",
                    label="evidence",
                    weight=1.0,
                )
        if node.get("bind_identifiers"):
            node["summary"] = node.get("summary") or _compact(
                ", ".join(str(x) for x in node["bind_identifiers"])
            )

    # Journal lineage edges from append-only supersessions and parent links.
    supersession_targets = {}
    for row in journal_rows:
        if row.get("type") == "supersession":
            targets = (
                row.get("target_trial_ids") if isinstance(row.get("target_trial_ids"), list) else []
            )
            for target in targets:
                try:
                    supersession_targets[int(target)] = (
                        int(row.get("trial_id")) if row.get("trial_id") is not None else None
                    )
                except (TypeError, ValueError):
                    continue
    for trial_id, row in journal_by_trial.items():
        parent_trial = row.get("parent_trial")
        try:
            parent_trial_id = int(parent_trial) if parent_trial is not None else None
        except (TypeError, ValueError):
            parent_trial_id = None
        if parent_trial_id is not None and parent_trial_id in journal_by_trial:
            _add_edge(
                f"journal:{parent_trial_id}",
                f"journal:{trial_id}",
                "parent",
                label="parent",
                weight=0.8,
            )
        superseded_by = supersession_targets.get(trial_id)
        if superseded_by is not None and superseded_by in journal_by_trial:
            _add_edge(
                f"journal:{trial_id}",
                f"journal:{superseded_by}",
                "supersedes",
                label="supersession",
                weight=0.8,
            )

    # Pick root nodes. If the caller asked for a focus, prefer those matches;
    # otherwise center the graph on the active understanding: live/pending
    # strategy rows and current-run frontier journal rows.
    focus_norm = " ".join(str(focus or "").split()).lower()
    focus_matches = [
        node_id
        for node_id, node in sorted(
            nodes.items(),
            key=lambda item: _focus_match_key(item[0], item[1], focus_norm),
        )
        if focus_norm and _node_matches_focus(node, focus_norm)
    ]
    if focus_matches:
        roots = focus_matches[:8]
        focus_reason = "matched focus query"
    else:
        roots = [
            node_id
            for node_id, node in sorted(
                nodes.items(),
                key=lambda item: _node_sort_key(item[1]),
            )
            if (
                (
                    node["kind"] in {"strategy", "planner_hint"}
                    and node.get("state_group")
                    in {"applied", "pending", "context", "guardrail", "frozen", "live"}
                )
                or (node["kind"] == "journal" and node.get("state_group") in {"frontier", "audit"})
            )
        ][:16]
        if not roots:
            roots = [
                node_id
                for node_id, node in sorted(
                    nodes.items(),
                    key=lambda item: _node_sort_key(item[1]),
                )
            ][:8]
        focus_reason = "default active understanding"

    # Breadth-first expansion around the roots. This gives a subgraph that is
    # explorable without flooding the browser with the whole store.
    distances: dict[str, int] = {root_id: 0 for root_id in roots}
    queue: deque[str] = deque(roots)
    while queue:
        node_id = queue.popleft()
        current_depth = distances.get(node_id, 0)
        if current_depth >= depth:
            continue
        for neighbor_id in adjacency.get(node_id, set()):
            if neighbor_id not in nodes:
                continue
            if neighbor_id in distances:
                continue
            distances[neighbor_id] = current_depth + 1
            queue.append(neighbor_id)

    if not distances:
        distances = {node_id: 0 for node_id in list(nodes)[: min(limit, 8)]}

    for node_id, node in nodes.items():
        node["degree"] = len(adjacency.get(node_id, set()))
        node["distance"] = distances.get(node_id, 99)

    selected_ids = {node_id for node_id, node in nodes.items() if node.get("distance", 99) <= depth}
    for root_id in roots:
        selected_ids.add(root_id)
    if len(selected_ids) > limit:
        root_nodes = [nodes[root_id] for root_id in roots if root_id in selected_ids]
        remaining_nodes = sorted(
            (
                nodes[node_id]
                for node_id in selected_ids
                if node_id not in {node["id"] for node in root_nodes}
            ),
            key=_node_sort_key,
        )
        selected_nodes = root_nodes + remaining_nodes[: max(0, limit - len(root_nodes))]
        selected_ids = {node["id"] for node in selected_nodes}

    selected_nodes = [dict(nodes[node_id]) for node_id in selected_ids if node_id in nodes]
    selected_nodes.sort(key=_node_sort_key)
    selected_edges = [
        dict(edge)
        for edge in edges.values()
        if edge["source"] in selected_ids and edge["target"] in selected_ids
    ]
    selected_edges.sort(key=lambda edge: (edge["kind"], edge["source"], edge["target"]))

    state_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for node in selected_nodes:
        state_counts[node["state_group"]] = state_counts.get(node["state_group"], 0) + 1
        kind_counts[node["kind"]] = kind_counts.get(node["kind"], 0) + 1

    focus_node_id = None
    if roots:
        focus_node_id = roots[0]
    focus_node = nodes.get(focus_node_id) if focus_node_id else None
    focus_summary = {
        "query": focus,
        "reason": focus_reason,
        "matches": focus_matches,
        "focus_node_id": focus_node_id,
        "focus_label": focus_node.get("label") if focus_node else None,
        "focus_kind": focus_node.get("kind") if focus_node else None,
    }

    return {
        "available": bool(nodes),
        "read_only": True,
        "focus": focus_summary,
        "source": {
            "journal_path": str(_AUTOPILOT_JOURNAL_PATH),
            "journal_run_start_index": journal_meta.get("journal_run_start_index"),
            "journal_run_start_trial_id": journal_meta.get("journal_run_start_trial_id"),
            "strategy_store_path": str(_STRATEGY_STORE_PATH / "strategies.db"),
            "planner_hint_seed_path": str(_PLANNER_HINT_SEEDS_PATH),
            "graph_source": graph_source,
        },
        "summary": {
            "journal_rows": len(journal_rows),
            "strategy_rows": len(strategy_rows),
            "planner_hint_rows": len(planner_hint_rows),
            "node_count": len(selected_nodes),
            "edge_count": len(selected_edges),
            "state_counts": state_counts,
            "kind_counts": kind_counts,
        },
        "nodes": selected_nodes,
        "edges": selected_edges,
    }


def _shape_baseline_promotion_event(event: dict[str, Any]) -> dict[str, Any]:
    proof = event.get("proof") if isinstance(event.get("proof"), dict) else {}
    result_metrics = (
        event.get("result_metrics") if isinstance(event.get("result_metrics"), dict) else {}
    )
    previous_quality = event.get("previous_quality")
    new_quality = event.get("new_quality")
    try:
        quality_delta = (
            float(new_quality) - float(previous_quality)
            if previous_quality is not None and new_quality is not None
            else None
        )
    except (TypeError, ValueError):
        quality_delta = None
    return {
        "source_trial_id": event.get("source_trial_id"),
        "tier": event.get("tier"),
        "previous_quality": previous_quality,
        "new_quality": new_quality,
        "quality_delta": quality_delta,
        "timestamp": event.get("timestamp", ""),
        "reason": event.get("reason", ""),
        "policy_version": event.get("policy_version", ""),
        "actor": event.get("actor", ""),
        "matrix_status": proof.get("matrix_status"),
        "speed_metric_mode": proof.get("speed_metric_mode"),
        "result_quality": result_metrics.get("quality"),
        "result_speed": result_metrics.get("speed"),
        "pareto_status": result_metrics.get("pareto_status"),
    }


def _baseline_promotion_summary(
    rows: list[dict[str, Any]] | None,
    *,
    current_run_only: bool = True,
    limit: int = 20,
) -> dict[str, Any]:
    selected_rows = list(rows or [])
    if current_run_only:
        selected_rows, _meta = _latest_journal_run_rows(selected_rows)
    latest_trial_id = None
    for row in selected_rows:
        try:
            trial_id = int(row.get("trial_id"))
        except (TypeError, ValueError):
            continue
        if latest_trial_id is None or trial_id > latest_trial_id:
            latest_trial_id = trial_id
    events = [row for row in selected_rows if row.get("type") == _BASELINE_PROMOTION_EVENT_TYPE]
    events.sort(key=lambda row: _parse_journal_ts(row.get("timestamp")) or 0)
    recent = [_shape_baseline_promotion_event(row) for row in events[-limit:]]
    latest_promotion_trial_id = recent[-1].get("source_trial_id") if recent else None
    trials_since_promotion = None
    try:
        if latest_trial_id is not None and latest_promotion_trial_id is not None:
            trials_since_promotion = max(0, int(latest_trial_id) - int(latest_promotion_trial_id))
    except (TypeError, ValueError):
        trials_since_promotion = None
    return {
        "count": len(events),
        "recent": recent,
        "latest_trial_id": latest_trial_id,
        "latest_promotion_trial_id": latest_promotion_trial_id,
        "trials_since_promotion": trials_since_promotion,
    }


def _trial_outcome_summary(
    rows: list[dict[str, Any]] | None,
    *,
    current_run_only: bool = True,
) -> dict[str, Any]:
    """Summarize keep/revert and learning-exclusion outcomes from journal rows.

    The dashboard only reports rates that can be derived from existing journal
    fields. If a category is absent, its rate stays ``None`` rather than being
    inferred from another bucket.
    """
    selected_rows = list(rows or [])
    if current_run_only:
        selected_rows, _meta = _latest_journal_run_rows(selected_rows)
    promotion_trial_ids = [
        int(row["source_trial_id"])
        for row in selected_rows
        if row.get("type") == _BASELINE_PROMOTION_EVENT_TYPE
        if str(row.get("source_trial_id") or "").isdigit()
    ]
    selected_rows = _effective_journal_trial_rows(selected_rows)

    keep_revert_total = 0
    keepable_count = 0
    wasted_eval_count = 0
    learning_excluded_count = 0
    active_trial_count = 0
    regression_count = 0

    for row in selected_rows:
        bug = str(row.get("bug_corrupted_by") or "").strip()
        outcome_rate_eligible = not bug or bug == "mad_noise"
        outcome_status = str(row.get("outcome_status") or "ok").strip().lower()
        action_type = str(row.get("action_type") or "").strip()
        is_active_trial = (
            outcome_rate_eligible
            and outcome_status not in {"invalid", "skipped"}
            and action_type not in {"distill_knowledge", "reset_memories"}
        )
        if is_active_trial:
            active_trial_count += 1
            deficiency = str(row.get("deficiency_category") or "").strip().lower()
            failure_analysis = str(row.get("failure_analysis") or "").lower()
            if deficiency in {"regression", "per_suite_regression"} or (
                "regression" in failure_analysis
            ):
                regression_count += 1
        if bug and bug != "mad_noise":
            continue
        decision = str(row.get("keep_revert_decision") or "").strip()
        eval_details = row.get("eval_details")
        learning_exclusion_by = ""
        if isinstance(eval_details, dict):
            learning_exclusion = eval_details.get("learning_exclusion")
            if isinstance(learning_exclusion, dict):
                learning_exclusion_by = str(learning_exclusion.get("by") or "").strip()
        is_learning_excluded = bool(learning_exclusion_by) or decision == "excluded"
        if decision in {"keep", "revert", "excluded", "unchanged"}:
            keep_revert_total += 1
            if decision == "keep":
                keepable_count += 1
            elif decision == "revert":
                wasted_eval_count += 1
        if is_learning_excluded:
            learning_excluded_count += 1

    def _rate(count: int, total: int) -> float | None:
        if total <= 0:
            return None
        return round(count / total, 3)

    def _per_100(count: int, total: int) -> float | None:
        if total <= 0:
            return None
        return round((count / total) * 100.0, 3)

    return {
        "keepable_rate": {
            "count": keepable_count,
            "total": keep_revert_total,
            "rate": _rate(keepable_count, keep_revert_total),
        },
        "wasted_eval_rate": {
            "count": wasted_eval_count,
            "total": keep_revert_total,
            "rate": _rate(wasted_eval_count, keep_revert_total),
        },
        "learning_excluded_rate": {
            "count": learning_excluded_count,
            "total": keep_revert_total,
            "rate": _rate(learning_excluded_count, keep_revert_total),
        },
        "active_trial_count": active_trial_count,
        "regression_per_active_trial": {
            "count": regression_count,
            "total": active_trial_count,
            "rate": _rate(regression_count, active_trial_count),
        },
        "promotions_per_100_active_trials": {
            "count": len(promotion_trial_ids),
            "total": active_trial_count,
            "per_100": _per_100(len(promotion_trial_ids), active_trial_count),
        },
    }


def _autopilot_current_code_health() -> dict[str, Any] | None:
    """Return the current-code health report when the phase snapshot exists."""
    try:
        report = build_phase_health_report(
            path=AUTOPILOT_PHASE_PATH,
            require_current_code=True,
        )
        if str(report.get("status") or "") in {"missing", "unavailable"}:
            return None
        report = dict(report)
        try:
            report["restart_advice"] = _build_autopilot_restart_advice(
                report,
                max_trials=int(os.environ.get("AUTOPILOT_DASHBOARD_RESTART_MAX_TRIALS", "3000")),
            )
        except Exception as exc:  # noqa: BLE001
            report["restart_advice"] = {
                "advisor_version": "autopilot_restart_advisor.v1",
                "ok": False,
                "status": "manual_attention",
                "restart_needed": False,
                "safe_to_restart_now": False,
                "reason": f"restart advice unavailable: {exc}",
                "blockers": [f"restart advice unavailable: {exc}"],
            }
        return report
    except Exception:
        return None


def _pareto_from_journal(
    session_start_ts: float | None,
    *,
    current_run_only: bool = False,
    max_trial_id: int | None = None,
    deinflate_before_ts: float | None = None,
    deinflate_factor: float = 1.0,
    exclude_before_ts: float | None = None,
    objective_policy: str = LEGACY_OBJECTIVE_POLICY,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Reconstruct Pareto data from the append-only journal."""
    if rows is None:
        rows = _read_autopilot_journal_rows()
    if rows is None:
        return None

    return reconstruct_archive_from_journal_rows(
        rows,
        session_start_ts,
        current_run_only=current_run_only,
        max_trial_id=max_trial_id,
        deinflate_before_ts=deinflate_before_ts,
        deinflate_factor=deinflate_factor,
        exclude_before_ts=exclude_before_ts,
        objective_policy=objective_policy,
    )


def _newest_autopilot_log() -> Path | None:
    """Return the most-recently-modified autopilot stdout log, if any."""
    try:
        candidates = []
        if AUTOPILOT_LOG.exists():
            candidates.append(AUTOPILOT_LOG)
        candidates.extend(_AUTOPILOT_LOG_DIR.glob("autopilot_restart_*.log"))
        candidates.extend(_AUTOPILOT_TMP_LOG_DIR.glob("autopilot_restart_*.log"))
        candidates.extend(_AUTOPILOT_TMP_LOG_DIR.glob("autopilot_authority_*.log"))
        candidates = sorted(
            (p for p in candidates if p.exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None
    except Exception:
        return None


def _tail_deep_eval_progress(log_path: Path) -> dict[str, int | str] | None:
    """Scan the autopilot log for the most-recent `T<n> progress: X/Y` line.

    The eval tower emits these lines as each question completes, so this gives
    the *true* completion fraction for a deep_eval trial — far more accurate
    than the historical-median estimate. Returns a dict with `eval_label`,
    `completed`, and `total`, or None.

    Matches any tier digit (`T0`..`T3` and future tiers) — the label tracks the
    eval tier, so a hardcoded `T[12]` silently dropped the T3 expert/hard lane back
    to the p90 estimate. `\\d+` keeps this generic as new tiers land, and the
    matched tier label is surfaced so the dashboard can display `T3 400/500`
    instead of a generic tower fraction.
    """
    if not log_path.exists():
        return None
    pat = re.compile(r"(T\d+) progress: (\d+)/(\d+)")
    try:
        # Tail-read: tier-2 evals can run hours and the log can be large; read
        # only the last 64 KB to find the most recent progress marker.
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            chunk = fh.read().decode("utf-8", errors="replace")
        last_match = None
        for m in pat.finditer(chunk):
            last_match = m
        if last_match:
            return {
                "eval_label": last_match.group(1),
                "completed": int(last_match.group(2)),
                "total": int(last_match.group(3)),
            }
    except Exception:
        pass
    return None


@router.get("/dashboard/api/autopilot_progress")
async def autopilot_progress() -> JSONResponse:
    """Live progress estimate for the autopilot's currently-running trial.

    Sources, in priority order:
      1. For deep_eval/structural_experiment: tails the autopilot stdout log
         for `Tn progress: X/Y` lines — these are the authoritative per-question
         completion markers, so percent = X/Y exactly (no estimation) and the
         tier label is surfaced as `eval_label`.
      2. Otherwise: full quantile distribution (p25/p50/p75/p90) of historical
         durations for the same `action_type`, derived from successive
         `timestamp` deltas in the journal (autopilot runs trials serially,
         so gap ≈ runtime).
      3. Fallback: aggregate distribution across all action types.

    The journal schema has NO numeric duration field — only an ISO `timestamp`
    written when the trial completes. Earlier versions of this endpoint looked
    for `trial_duration_s`/`wall_time_s`/`duration_s` which don't exist.

    Bar denominator is **p90, not p50**: trial-runtime distributions are
    heavily right-skewed (rollback has p90 ≈ 6× p50), so a p50 bar saturates
    to 99% on the majority of trials and conveys nothing. With p90 as the
    "expected ceiling," ~10% of trials cross into the slow tail (`slow_tail=True`),
    making that signal actually meaningful. The frontend draws ticks at
    p25/p50/p75 inside the bar so the operator can see where elapsed sits
    in the distribution at a glance; `now_percentile` is the precise CDF rank.
    """
    out: dict[str, Any] = {
        "in_flight": False,
        "autopilot_alive": False,
        "trial_id": None,
        "action_type": None,
        "eval_label": None,
        "started_at": None,
        "elapsed_s": None,
        "expected_s": None,  # log_tail: extrapolated projected total; others: p50 (legacy meta label)
        "percent": None,  # elapsed/p90 * 100 (action_p50/aggregate_p50); X/Y * 100 (log_tail)
        "recent_avg_duration_s": None,
        "recent_p25": None,  # distribution quantiles used by the tick-style bar
        "recent_p50": None,
        "recent_p75": None,
        "recent_p90": None,  # bar denominator for action_p50/aggregate_p50
        "now_percentile": None,  # CDF rank of elapsed within same-action history, 0.0..1.0
        "slow_tail": False,  # True when elapsed > p90 — bar saturates, signals genuine tail
        "percent_source": None,  # "log_tail" | "action_p50" | "aggregate_p50" | "fallback"
        "n_action_type_samples": None,
        "log_tail_progress": None,  # {"completed": X, "total": Y} when percent_source=log_tail
        "baseline_promotions": {
            "count": 0,
            "recent": [],
            "latest_trial_id": None,
            "latest_promotion_trial_id": None,
            "trials_since_promotion": None,
        },
        "outcome_kpis": {
            "keepable_rate": {
                "count": 0,
                "total": 0,
                "rate": None,
            },
            "wasted_eval_rate": {
                "count": 0,
                "total": 0,
                "rate": None,
            },
            "learning_excluded_rate": {
                "count": 0,
                "total": 0,
                "rate": None,
            },
            "active_trial_count": 0,
            "regression_per_active_trial": {
                "count": 0,
                "total": 0,
                "rate": None,
            },
            "promotions_per_100_active_trials": {
                "count": 0,
                "total": 0,
                "per_100": None,
            },
        },
        "current_code_health": None,
    }
    current_code_health = _autopilot_current_code_health()
    # Is autopilot alive? Quick pgrep-equivalent
    try:
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                cmd = (p / "cmdline").read_bytes().decode(errors="replace")
            except Exception:
                continue
            if "autopilot.py" in cmd and "start" in cmd:
                out["autopilot_alive"] = True
                break
    except Exception:
        pass

    # Read state.json for in_flight_trial
    if _AUTOPILOT_STATE_PATH.exists():
        try:
            state = json.loads(_AUTOPILOT_STATE_PATH.read_text())
            in_flight = state.get("in_flight_trial") or {}
            if in_flight and in_flight.get("trial_id") is not None:
                started_at = float(in_flight.get("started_at", 0)) or None
                elapsed = (time.time() - started_at) if started_at else None
                out.update(
                    {
                        "in_flight": True,
                        "trial_id": in_flight.get("trial_id"),
                        "action_type": (in_flight.get("action") or {}).get("type"),
                        "started_at": started_at,
                        "elapsed_s": elapsed,
                    }
                )
        except Exception:
            pass

    # Per-action_type durations from journal: successive-timestamp deltas
    # (autopilot is serial, so entry[i].timestamp − entry[i-1].timestamp
    # approximates the runtime of entry[i] modulo ~seconds of dispatch overhead).
    by_action: dict[str, list[float]] = {}
    all_durations: list[float] = []
    raw_entries = _read_autopilot_journal_rows()
    if raw_entries:
        try:
            entries = list(raw_entries)
            baseline_promotions = _baseline_promotion_summary(
                raw_entries,
                current_run_only=True,
            )
            outcome_kpis = _trial_outcome_summary(
                raw_entries,
                current_run_only=True,
            )
            entries = _effective_journal_trial_rows(entries)
            # Sort by timestamp ascending so deltas make sense even if the
            # journal was rewritten out-of-order (shouldn't happen, but cheap).
            entries.sort(key=lambda e: _parse_journal_ts(e.get("timestamp")) or 0)
            for i in range(1, len(entries)):
                a = _parse_journal_ts(entries[i].get("timestamp"))
                b = _parse_journal_ts(entries[i - 1].get("timestamp"))
                if a is None or b is None:
                    continue
                dt = a - b
                # Drop: negative gaps, sub-30s (likely back-to-back fast failures
                # or quarantined trials), and >6h (likely overnight pauses or
                # exogenous-pause windows that don't reflect trial runtime).
                if not (30 < dt < 6 * 3600):
                    continue
                # Trust filter: skip if THIS entry was bug-quarantined
                if entries[i].get("bug_corrupted_by"):
                    continue
                at = entries[i].get("action_type") or "unknown"
                by_action.setdefault(at, []).append(dt)
                all_durations.append(dt)
            out["baseline_promotions"] = baseline_promotions
            out["outcome_kpis"] = outcome_kpis
        except Exception:
            pass
    if current_code_health is not None:
        out["current_code_health"] = current_code_health

    def _stats(durs: list[float]) -> dict[str, float] | None:
        if not durs:
            return None
        ss = sorted(durs)
        n = len(ss)

        def q(p: float) -> float:
            return ss[min(n - 1, int(n * p))]

        return {
            "p25": q(0.25),
            "p50": q(0.5),
            "p75": q(0.75),
            "p90": q(0.9),
            "avg": sum(ss) / n,
            "n": n,
        }

    def _cdf_rank(value: float, durs: list[float]) -> float | None:
        # Fraction of past durations <= value. Used so the bar can answer
        # "where does *now* sit in the historical distribution?" instead of
        # the meaningless elapsed/p50 ratio. With small n, this is granular
        # (n=18 → 5.5pp steps) but still strictly more informative than a
        # saturating bar capped at 99%.
        if not durs:
            return None
        return sum(1 for d in durs if d <= value) / len(durs)

    # 1. Log-tail (authoritative for deep_eval-style multi-question evals)
    if out["in_flight"] and out["action_type"] in ("deep_eval", "structural_experiment"):
        log = _newest_autopilot_log()
        if log:
            progress = _tail_deep_eval_progress(log)
            if progress:
                completed = progress["completed"]
                total = progress["total"]
                if total > 0:
                    pct = (completed / total) * 100.0
                    out["percent"] = round(min(99.0, max(0.0, pct)), 1)
                    out["percent_source"] = "log_tail"
                    out["eval_label"] = progress["eval_label"]
                    out["log_tail_progress"] = {"completed": completed, "total": total}
                    # Also extrapolate an expected_s for the meta line:
                    # if X/Y done in elapsed_s, projected total ≈ elapsed_s × Y/X
                    if completed > 0 and out["elapsed_s"]:
                        out["expected_s"] = round(out["elapsed_s"] * total / completed, 1)

    # 2. Action-type-stratified distribution (when log-tail didn't apply)
    if out["expected_s"] is None and out["in_flight"]:
        at = out["action_type"] or "unknown"
        durs = by_action.get(at, [])
        s = _stats(durs)
        if s and s["n"] >= 2:
            out["expected_s"] = round(s["p50"], 1)
            out["recent_p25"] = round(s["p25"], 1)
            out["recent_p50"] = round(s["p50"], 1)
            out["recent_p75"] = round(s["p75"], 1)
            out["recent_p90"] = round(s["p90"], 1)
            out["recent_avg_duration_s"] = round(s["avg"], 1)
            out["n_action_type_samples"] = s["n"]
            out["percent_source"] = "action_p50"
            if out["elapsed_s"] is not None:
                rank = _cdf_rank(out["elapsed_s"], durs)
                if rank is not None:
                    out["now_percentile"] = round(rank, 3)

    # 3. Aggregate distribution across all action types
    if out["expected_s"] is None and out["in_flight"]:
        s = _stats(all_durations)
        if s:
            out["expected_s"] = round(s["p50"], 1)
            out["recent_p25"] = round(s["p25"], 1)
            out["recent_p50"] = round(s["p50"], 1)
            out["recent_p75"] = round(s["p75"], 1)
            out["recent_p90"] = round(s["p90"], 1)
            out["recent_avg_duration_s"] = round(s["avg"], 1)
            out["n_action_type_samples"] = s["n"]
            out["percent_source"] = "aggregate_p50"
            if out["elapsed_s"] is not None:
                rank = _cdf_rank(out["elapsed_s"], all_durations)
                if rank is not None:
                    out["now_percentile"] = round(rank, 3)

    # 4. Hard fallback only if we have literally no journal data
    if out["expected_s"] is None and out["in_flight"]:
        out["expected_s"] = 1200.0
        out["percent_source"] = "fallback"

    # Percent: bar fill ratio.
    #   - log_tail: already set above to X/Y (true progress).
    #   - action_p50/aggregate_p50: elapsed/p90. p90 is "the slow-tail edge" —
    #     ~90% of past trials of this action_type finish by then, so the bar
    #     saturates only when we're genuinely in the tail. Allowed to exceed
    #     100 (slow_tail=True) so the JS can render "p93+" without lying.
    #   - fallback: elapsed/1200, clamped at 100.
    if out["percent"] is None and out["in_flight"] and out["elapsed_s"] is not None:
        denom = out["recent_p90"] if out["recent_p90"] else out["expected_s"]
        if denom:
            pct = (out["elapsed_s"] / denom) * 100.0
            out["slow_tail"] = pct > 100.0
            # Cap at 150 — anything beyond is visually identical "deep tail";
            # the precise number is in elapsed_s if anyone needs it.
            out["percent"] = round(min(150.0, max(0.0, pct)), 1)

    # H2: shared autopilot-plane epoch (cross-panel coherence cross-check).
    out["state_generation"] = _autopilot_state_generation()
    return JSONResponse(_stamp(out, "autopilot_progress"), headers=_NO_STORE_HEADERS)


@router.get("/dashboard/api/pareto")
async def pareto(max_dominated: int = 600, scope: str = "current") -> JSONResponse:
    """Return the autopilot's Pareto archive for visualization.

    Prefer reconstructing all trusted progress from the append-only journal. This
    keeps the dashboard useful when autopilot is down and protects the panel
    from a stale `pareto_archive` cache inside autopilot_state.json.

    Objectives in the current archive follow ``pareto_objective_policy`` from
    AutoPilot state. Their stable 4D shape is (quality, rate, -cost,
    reliability), but the rate unit is policy-owned: legacy rows use request
    t/s while the live W3 policy uses questions/hour.

    `scope` selects the reconstruction window:
      - "current" (default): the operational current-era progress view —
        latest trial-id-reset segment, rows before `pareto_exclude_before_ts`
        dropped. This is the decision-grade view.
      - "all_eras" (also "all"/"history"): every journaled trial across
        instrument eras, era-labeled from orchestration/instrument_eras.yaml.
        Pre-E2 speeds get the codified ×`pareto_pre_epoch_speed_factor`
        deinflation so the speed axis is comparable; later era boundaries
        (e.g. the E5 v6+iqk kernel cutover) are labeled, never rescaled.
        Comparative visualization only — not decision-grade (MEASUREMENT.md:
        cross-era dominance never gates decisions). Assumes trial ids are
        monotonic across the retained journal shards (true since the 2026-05
        scrubs); a future deliberate rewind would interleave this view but
        leaves scope="current" correct.
    """
    data: dict[str, Any] = {}
    state_error: str | None = None
    if _AUTOPILOT_STATE_PATH.exists():
        try:
            data = json.loads(_AUTOPILOT_STATE_PATH.read_text())
        except Exception as exc:
            state_error = f"failed to parse autopilot_state.json: {exc}"

    state_trial_counter = None
    try:
        state_trial_counter = int(data.get("trial_counter"))
    except (TypeError, ValueError):
        state_trial_counter = None
    active_objective_policy, objective_policy_warning = _active_pareto_objective_policy(data)

    # The operator-facing Pareto/GEPA plots are current-run progress charts,
    # not "this API process since reload" and not all historical campaigns.
    # Use the latest journal segment after a trial-id reset so restarts inside
    # the current run survive while pre-current-stack rows stay out.
    # A deliberate Pareto rebase (e.g. the 2026-06-01 speed double-count correction)
    # records a stable `pareto_epoch_ts`: trials before it were measured under the old
    # metric and must NOT be reconstructed onto the frontier (they would dominate the
    # honest post-fix points on the changed axis). The trial counter is NOT reset on a
    # rebase, so `current_run_only` alone can't exclude them — scope by timestamp too.
    pareto_epoch_ts = None
    try:
        pareto_epoch_ts = float(data.get("pareto_epoch_ts") or 0.0) or None
    except (TypeError, ValueError):
        pareto_epoch_ts = None
    # Option (iii): rather than EXCLUDING pre-fix trials, keep them but de-inflate their
    # double-counted speed (factor tunable via state; default 0.5 ≈ the ~2x inflation of
    # no-thinking eval trials) so the metric correction is visible on the plot.
    deinflate_factor = 0.5
    try:
        deinflate_factor = float(data.get("pareto_pre_epoch_speed_factor", 0.5))
    except (TypeError, ValueError):
        deinflate_factor = 0.5
    try:
        pareto_exclude_before_ts = float(data.get("pareto_exclude_before_ts") or 0.0) or None
    except (TypeError, ValueError):
        pareto_exclude_before_ts = None
    # Do NOT cap reconstruction at state_trial_counter. The append-only journal —
    # not the periodically-saved state counter — is the source of truth for
    # progress (that is the whole reason this endpoint reconstructs from the
    # journal). `autopilot_state.json` is rewritten every trial in the happy path,
    # but a crash / SIGKILL / rewind between saves leaves the counter STALE; the
    # old `max_trial_id=state_trial_counter` cap then silently dropped every
    # journal row newer than the last save (the recurring "dashboard frozen at
    # ~700 while the journal holds 1000 trials" report). `current_run_only` +
    # `pareto_epoch_ts` already scope to the live run, so the cap only ever
    # truncated real data. We now surface the divergence as a warning instead.
    journal_rows = _read_autopilot_journal_rows()
    state_archive_present = isinstance(data.get("pareto_archive"), dict) and bool(
        data.get("pareto_archive")
    )
    baseline_promotions = _baseline_promotion_summary(
        journal_rows,
        current_run_only=True,
    )
    all_eras = str(scope).strip().lower() in {"all", "all_eras", "history"}
    era_regions: list[dict[str, Any]] = []
    era_registry_error: str | None = None
    if all_eras:
        era_regions, era_registry_error = _autopilot_era_regions()
        # E2 is the only rescaled boundary (see _PARETO_SPEED_DEINFLATE_ERA_ID).
        # Registry unreadable → fail open WITHOUT deinflation and surface the
        # error; the state's pareto_epoch_ts is NOT a substitute (it advances on
        # every rebase — currently the E5 cutover — and would falsify honest
        # E2..E5 speeds if used as the deinflate boundary here).
        deinflate_e2_ts = next(
            (
                region["from_ts"]
                for region in era_regions
                if _PARETO_SPEED_DEINFLATE_ERA_ID in (region.get("era_ids") or [])
            ),
            None,
        )
        journal_archive = _pareto_from_journal(
            None,
            current_run_only=False,
            max_trial_id=None,
            deinflate_before_ts=deinflate_e2_ts,
            deinflate_factor=deinflate_factor if deinflate_e2_ts is not None else 1.0,
            exclude_before_ts=None,
            # One common, explicitly historical coordinate system for the
            # cross-era cloud. It is never decision-grade and is not described
            # as AutoPilot's live frontier.
            objective_policy=LEGACY_OBJECTIVE_POLICY,
            rows=journal_rows,
        )
        source = "journal_all_eras"
        source_reason = (
            "reconstructed from ALL journal shards across instrument eras "
            "(era-labeled; comparative visualization, not decision-grade)"
        )
    else:
        journal_archive = _pareto_from_journal(
            None,
            current_run_only=True,
            max_trial_id=None,
            deinflate_before_ts=pareto_epoch_ts,
            deinflate_factor=deinflate_factor,
            exclude_before_ts=pareto_exclude_before_ts,
            objective_policy=active_objective_policy,
            rows=journal_rows,
        )
        source = "journal_current_run"
        source_reason = (
            "reconstructed from latest trial-id reset segment in autopilot_journal.jsonl"
        )

    stale_state_warning = None
    if journal_archive:
        _j_max = journal_archive.get("journal_max_trial_id")
        if (
            state_trial_counter is not None
            and isinstance(_j_max, int)
            and state_trial_counter < _j_max
        ):
            stale_state_warning = {
                "state_trial_counter": state_trial_counter,
                "journal_max_trial_id": _j_max,
                "detail": (
                    f"autopilot_state.json trial_counter ({state_trial_counter}) lags "
                    f"the journal (max trial {_j_max}); showing all journaled trials. "
                    "State file is likely stale (autopilot crashed or was rewound "
                    "between saves)."
                ),
            }

    if journal_archive:
        archive = journal_archive
    elif data:
        archive = data.get("pareto_archive", {}) or {}
        source = "state_archive"
        source_reason = "journal unavailable or no trusted current-run entries"
    else:
        return JSONResponse(
            _stamp(
                {
                    "available": False,
                    "state_generation": _autopilot_state_generation(),
                    "reason": state_error
                    or "autopilot_state.json and autopilot_journal.jsonl not found",
                    "frontier": [],
                    "dominated": [],
                    "hypervolume_history": [],
                },
                "pareto",
            )
        )

    legacy_state_archive_warning = None
    if source == "state_archive":
        legacy_state_archive_warning = {
            "state_archive_present": state_archive_present,
            "journal_rows_available": len(journal_rows),
            "detail": (
                "dashboard fell back to autopilot_state.json:pareto_archive; "
                "treat this as a legacy state-cache view and run strict archive "
                "authority validation before using it for decisions"
            ),
        }

    canonical_tier = int(archive.get("canonical_tier", DEFAULT_FRONTIER_TIER))
    active_instrument_eras = (
        data.get("active_instrument_eras")
        if isinstance(data.get("active_instrument_eras"), dict)
        else {}
    )
    frontier_rerun_required = (
        data.get("frontier_rerun_required")
        if isinstance(data.get("frontier_rerun_required"), dict)
        else None
    )
    multitier_policy = (
        data.get("multitier_promotion_policy")
        if isinstance(data.get("multitier_promotion_policy"), dict)
        else {}
    )
    multitier_bundle = (
        data.get("multitier_baseline_bundle")
        if isinstance(data.get("multitier_baseline_bundle"), dict)
        else {}
    )
    multitier_pending = (
        data.get("multitier_pending_validation")
        if isinstance(data.get("multitier_pending_validation"), dict)
        else None
    )
    multitier_tiers = multitier_bundle.get("tiers") or {}
    multitier_promotion = {
        "enabled": multitier_policy.get("enabled") is True,
        "policy_version": multitier_policy.get("policy_version")
        or multitier_bundle.get("policy_version"),
        "required_tiers": multitier_policy.get("required_tiers") or [2, 3],
        "baseline_tiers": sorted(
            int(tier)
            for tier in multitier_tiers
            if str(tier).isdigit()
        )
        if isinstance(multitier_tiers, dict)
        else [],
        "status": (
            str(multitier_pending.get("status") or "pending")
            if multitier_pending
            else "ready" if multitier_policy.get("enabled") is True else "disabled"
        ),
        "next_tier": multitier_pending.get("next_tier") if multitier_pending else None,
        "candidate": multitier_pending.get("candidate") if multitier_pending else None,
        "last_event": data.get("multitier_last_event")
        if isinstance(data.get("multitier_last_event"), dict)
        else None,
    }
    frontiers_by_tier_raw = archive.get("frontiers_by_tier", {}) or {}
    hv_history_by_tier_raw = archive.get("hv_history_by_tier", {}) or {}
    frontier_raw = (
        frontiers_by_tier_raw.get(str(canonical_tier))
        or frontiers_by_tier_raw.get(canonical_tier)
        or archive.get("frontier", [])
        or []
    )
    all_raw = archive.get("all_entries", []) or []
    t0_audit_raw = archive.get("t0_audit", []) or []
    t0_audit_shaped = [_shape_pareto_entry(e) for e in t0_audit_raw]
    hv_history = (
        hv_history_by_tier_raw.get(str(canonical_tier))
        or hv_history_by_tier_raw.get(canonical_tier)
        or archive.get("hypervolume_history", [])
        or []
    )
    frontier = [_shape_pareto_entry(e) for e in frontier_raw]
    frontiers_by_tier = {
        str(tier): [_shape_pareto_entry(e) for e in entries]
        for tier, entries in frontiers_by_tier_raw.items()
    }
    frontier_ids = {f["trial_id"] for f in frontier if f["trial_id"] is not None}

    # Dominated entries: newest first, trimmed to max_dominated to bound payload.
    dominated_only = [e for e in all_raw if e.get("trial_id") not in frontier_ids]
    dominated_only.sort(key=lambda e: e.get("trial_id") or 0, reverse=True)
    dominated_shaped = [_shape_pareto_entry(e) for e in dominated_only[:max_dominated]]

    hv_shaped: list[list[float]] = []
    for h in hv_history:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            try:
                hv_shaped.append([int(h[0]), float(h[1])])
            except (TypeError, ValueError):
                continue

    # Join task_rate/goodput/tokens-per-solved (+ offered_load) back onto every
    # shipped point from the SAME folded journal rows this panel reconstructs.
    # These are descriptive companions; current-scope dominance is owned by
    # ``active_objective_policy`` above.
    rows_by_tid: dict[int, dict[str, Any]] = {}
    for row in _effective_journal_trial_rows(journal_rows or []):
        try:
            rows_by_tid[int(row.get("trial_id"))] = row
        except (TypeError, ValueError):
            continue
    _attach_task_rate_telemetry(
        (
            frontier,
            dominated_shaped,
            t0_audit_shaped,
            *frontiers_by_tier.values(),
        ),
        rows_by_tid,
    )
    _attach_decision_question_mix(
        (
            frontier,
            dominated_shaped,
            t0_audit_shaped,
            *frontiers_by_tier.values(),
        ),
        rows_by_tid,
    )
    current_decision_question_mix = next(
        (
            mix
            for _, row in sorted(rows_by_tid.items(), reverse=True)
            if (mix := _decision_question_mix_from_row(row)) is not None
        ),
        None,
    )
    objective_policy_comparison = _objective_policy_comparison_summary(
        journal_rows,
        data,
        active_objective_policy,
    )

    # All-era view: stamp every shipped point with its era region, and report
    # per-region trial ranges so the client can shade era bands on the timeline.
    eras_payload: list[dict[str, Any]] | None = None
    if all_eras and era_regions:
        for entries in (
            frontier,
            dominated_shaped,
            t0_audit_shaped,
            *frontiers_by_tier.values(),
        ):
            _label_pareto_entries_with_eras(entries, era_regions)
        region_stats: dict[int, dict[str, Any]] = {}
        seen_trial_ids: set[Any] = set()
        # `frontier` mirrors the canonical tier of `frontiers_by_tier`, so
        # iterating the by-tier lists + dominated + t0 covers every point once.
        for entries in (dominated_shaped, t0_audit_shaped, *frontiers_by_tier.values()):
            for entry in entries:
                tid, idx = entry.get("trial_id"), entry.get("era_index")
                if idx is None or tid is None or tid in seen_trial_ids:
                    continue
                seen_trial_ids.add(tid)
                stats = region_stats.setdefault(
                    idx, {"n_points": 0, "first_trial_id": None, "last_trial_id": None}
                )
                stats["n_points"] += 1
                stats["first_trial_id"] = (
                    tid if stats["first_trial_id"] is None else min(stats["first_trial_id"], tid)
                )
                stats["last_trial_id"] = (
                    tid if stats["last_trial_id"] is None else max(stats["last_trial_id"], tid)
                )
        eras_payload = [
            {
                **region,
                **region_stats.get(
                    region["index"],
                    {"n_points": 0, "first_trial_id": None, "last_trial_id": None},
                ),
            }
            for region in era_regions
        ]

    return JSONResponse(
        _stamp(
            {
                "available": True,
                # H2: shared autopilot-plane epoch (cross-panel coherence check).
                "state_generation": _autopilot_state_generation(),
                "source": source,
                "source_reason": source_reason,
                "scope": "all_eras" if all_eras else "current",
                "objective_policy": (
                    LEGACY_OBJECTIVE_POLICY if all_eras else active_objective_policy
                ),
                "active_objective_policy": active_objective_policy,
                "objective_policy_warning": objective_policy_warning,
                "decision_grade": not all_eras and objective_policy_warning is None,
                "objective_policy_context": (
                    "historical_legacy_comparator" if all_eras else "active"
                ),
                "active_instrument_eras": active_instrument_eras,
                "pareto_epoch_ts": pareto_epoch_ts,
                "pareto_exclude_before_ts": pareto_exclude_before_ts,
                "frontier_rerun_required": frontier_rerun_required,
                "multitier_promotion": multitier_promotion,
                "eras": eras_payload,
                "era_registry_error": era_registry_error if all_eras else None,
                # Visibility into why trials may be missing from the frontier, so the
                # operator never has to guess whether the plot is stale or the data is.
                "stale_state_warning": stale_state_warning,
                "legacy_state_archive_warning": legacy_state_archive_warning,
                "archive_authority": {
                    "source": source,
                    "journal_rows_available": len(journal_rows),
                    "state_archive_present": state_archive_present,
                    "state_error": state_error,
                    "using_legacy_state_archive": source == "state_archive",
                },
                # Historical comparator only. W3 already flipped; this never
                # presents a pending decision or gates AutoPilot.
                "objective_policy_comparison": objective_policy_comparison,
                # `canonical_tier` / `eval_tier` are OUTER lanes. This is the
                # latest INNER question-difficulty composition of the core.
                "current_decision_question_mix": current_decision_question_mix,
                "state_trial_counter": state_trial_counter,
                "journal_max_trial_id": archive.get("journal_max_trial_id")
                if isinstance(archive, dict)
                else None,
                "exclusions": archive.get("exclusions") if isinstance(archive, dict) else None,
                "baseline_promotions": baseline_promotions,
                "frontier": frontier,
                "frontiers_by_tier": frontiers_by_tier,
                "dominated": dominated_shaped,
                "t0_audit": t0_audit_shaped,
                "hypervolume_history": hv_shaped,
                "hv_history_by_tier": hv_history_by_tier_raw,
                "totals": {
                    "frontier_size": len(frontier),
                    "all_entries": len(all_raw),
                    "hv_points": len(hv_shaped),
                },
                "canonical_tier": canonical_tier,
                "session_start_ts": archive.get("session_start_ts"),
                "journal_run_start_index": archive.get("journal_run_start_index"),
                "journal_run_start_trial_id": archive.get("journal_run_start_trial_id"),
                "journal_run_start_ts": archive.get("journal_run_start_ts"),
                "objective_axes": [
                    {"key": "quality", "index": 0, "direction": "max", "label": "quality"},
                    {
                        "key": "rate",
                        "index": 1,
                        "direction": "max",
                        "label": (
                            "median request t/s (historical comparator)"
                            if all_eras or active_objective_policy == LEGACY_OBJECTIVE_POLICY
                            else "task rate (questions/hour)"
                        ),
                    },
                    {"key": "neg_cost", "index": 2, "direction": "max", "label": "-cost"},
                    {"key": "reliability", "index": 3, "direction": "max", "label": "reliability"},
                ],
            },
            "pareto",
        )
    )


_AUTOPILOT_SNAPSHOT_MAX_COHERENCE_ATTEMPTS = 2


@router.get("/dashboard/api/autopilot_snapshot")
async def autopilot_snapshot(scope: str = "current") -> JSONResponse:
    """One coherent frame of the four autopilot-state panels (H2).

    The pareto / autopilot_progress / process_status / insight_graph panels each
    re-read autopilot_state.json + the rotating journal at their OWN request
    instant, so a client polling them separately can splice panel A (trial N)
    beside panel B (trial N-1). This endpoint builds all four inside ONE call,
    stamps them with ONE shared ``state_generation`` token, and re-checks the
    token after the build: if the underlying state advanced mid-build the frame
    is rebuilt (bounded retries) so the returned set is coherent
    (``coherent: true``). A client should prefer this endpoint; if it still
    polls the four panels individually it cross-checks their ``state_generation``
    and discards any set whose tokens disagree. Also carries the H5
    ``value_consistency`` verdict (state ``trial_counter`` vs journal max trial).
    """
    return await _autopilot_snapshot_impl(scope=scope)


async def _autopilot_snapshot_impl(scope: str = "current") -> JSONResponse:
    now = time.time()
    frame: dict[str, Any] = {}
    attempts = 0
    while attempts < _AUTOPILOT_SNAPSHOT_MAX_COHERENCE_ATTEMPTS:
        attempts += 1
        gen_before = _autopilot_state_generation()
        # ONE shared read of the journal shards for the frame-level summary
        # (reusing the existing rotation-aware reader); the per-panel logic is
        # reused unchanged via the endpoint calls below rather than duplicated.
        journal_rows = _read_autopilot_journal_rows() or []
        journal_max_trial = _journal_max_trial_id(journal_rows)
        state_trial_counter = _state_trial_counter()
        process = json.loads((await process_status()).body)
        progress = json.loads((await autopilot_progress()).body)
        pareto_body = json.loads((await pareto(scope=scope)).body)
        insight = json.loads((await insight_graph()).body)
        gen_after = _autopilot_state_generation()
        frame = {
            "gen_before": gen_before,
            "gen_after": gen_after,
            "journal_rows_available": len(journal_rows),
            "journal_max_trial": journal_max_trial,
            "state_trial_counter": state_trial_counter,
            "panels": {
                "process_status": process,
                "autopilot_progress": progress,
                "pareto": pareto_body,
                "insight_graph": insight,
            },
        }
        # gen_before == gen_after ⇒ nothing wrote state.json / the journal while
        # the four panels were built ⇒ they reflect ONE instant. A mismatch is a
        # detected tear; rebuild once more before giving up.
        if gen_before == gen_after:
            break

    gen = frame["gen_before"]
    coherent = frame["gen_before"] == frame["gen_after"]
    panels = frame["panels"]
    # Normalize every panel to the frame's single token so the assembled set is
    # self-consistent even in the (flagged) incoherent case.
    for panel in panels.values():
        if isinstance(panel, dict):
            panel["state_generation"] = gen
    consistency = _value_consistency(frame["state_trial_counter"], frame["journal_max_trial"])
    body: dict[str, Any] = {
        "generated_at": now,
        "state_generation": gen,
        "coherent": coherent,
        "coherence_attempts": attempts,
        "state_trial_counter": frame["state_trial_counter"],
        "journal_max_trial_id": frame["journal_max_trial"],
        "journal_rows_available": frame["journal_rows_available"],
        "value_consistency": consistency,
        "panels": panels,
    }
    # Freshness envelope built inline from the autopilot-plane sources (state +
    # journal) with the H5 value-consistency verdict attached — no dashboard_panels
    # registry key required, so this stays additive.
    body["_freshness"] = _freshness_envelope(
        _autopilot_snapshot_sources(),
        now=now,
        generated_at=now,
        consistency=consistency,
    )
    return JSONResponse(body, headers=_NO_STORE_HEADERS)


@router.get("/dashboard/api/version")
async def version() -> JSONResponse:
    """Return current build state for the hard-reload-needed indicator.

    The browser polls this every ~30s. If the returned `dashboard_html_mtime`
    or `git_sha` differs from what was captured at page load, the dashboard
    shows a "new build — hard-reload" badge.

    Returns:
        git_sha: short SHA of the orchestrator repo HEAD (or None if read fails)
        dashboard_html_mtime: float epoch seconds; bumps on every save
        dashboard_py_mtime: float epoch seconds; bumps when route handlers
            change (would require an orchestrator API reload to take effect,
            but useful to surface in case the file changed without restart)
        server_started_at: float epoch seconds; bumps on orchestrator restart
        server_launch_git_sha: short SHA stamped by the launcher at restart
    """

    def mtime(p: Path) -> float | None:
        try:
            return p.stat().st_mtime
        except Exception:
            return None

    return JSONResponse(
        _stamp(
            {
                "git_sha": _read_git_short_sha(),
                "dashboard_html_mtime": mtime(_DASHBOARD_HTML_FOR_VERSION),
                "dashboard_py_mtime": mtime(_DASHBOARD_PY_FOR_VERSION),
                "server_started_at": _SERVER_STARTED_AT,
                "server_launch_git_sha": _SERVER_LAUNCH_GIT_SHA,
            },
            "build_rev",
        )
    )


@router.get("/dashboard/api/llama_fleet_ids")
async def llama_fleet_ids() -> JSONResponse:
    """Aggregate per-port llama-server fleet-startup markers.

    Used by the autopilot's OrchestratorWatcher to:
      1. Resolve role→port lookups (each marker carries the canonical
         role list served by that llama-server process).
      2. Detect operator-initiated reloads (when a stored `started_at`
         differs from a fresh poll, that port restarted).
      3. Distinguish operator-initiated reloads (`source=stack_commands`)
         from external restarts (any other source value).

    See scripts/server/fleet_markers.py for the marker file format and
    handoffs/active/autopilot-exogenous-restart-resilience.md sections
    5.1 + 5.2 for the design.

    Returns:
        per_port: {port: {started_at: float, source: str, roles: list[str]}}
        now: float epoch seconds (for the client's freshness math)
    """
    try:
        from scripts.server.fleet_markers import discover_llama_markers

        per_port = discover_llama_markers()
    except Exception as exc:
        return JSONResponse({"error": str(exc), "per_port": {}, "now": time.time()})
    # JSON keys must be strings; convert int ports to strings for transport.
    return JSONResponse(
        {
            "per_port": {str(p): m for p, m in per_port.items()},
            "now": time.time(),
        }
    )


def _topology_activity_payload(
    window_s: float = 600.0,
    *,
    now: float | None = None,
    structured_requests: list[dict[str, Any]] | None = None,
    slots_by_port: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Per-role recent activity stats for the topology strip.

    TWO INDEPENDENT SIGNALS, never folded into each other (2026-08-04):

      - REACHABILITY (`running`, `running_instance_count`, `running_ports`):
        the listener exists in the ps(1) scan. This proves the server is UP.
        It proves NOTHING about whether it is working. Rendering it as "idle"
        was the defect this contract exists to prevent — a health probe cannot
        carry an activity claim.
      - OCCUPANCY (`slots_busy`, `slots_total`, `busy_ports`, `busy`,
        `occupancy_observed`): llama-server `/slots` `is_processing` counted
        across the role's instances. This is the ONLY field that may be
        rendered as busy/idle, and only when `occupancy_observed` is true.
        When `slots_by_port` is absent or every port failed to answer,
        `occupancy_observed` is False and the state is UNKNOWN, not idle —
        a failed probe must not read as "nothing is running".

    Aggregates from these sources:
      - llama-server /slots (via `slots_by_port`) — ground-truth occupancy.
        Unlike the tap/journal sources below it sees ALL traffic, including
        requests that never went through the orchestrator (direct benchmark
        or eval hits), which are exactly the ones the old panel called "idle".
      - inference_tap_events.jsonl request groups — provides concurrency-safe
        request count, last activity timestamp, and TIMINGS (t/s).
      - recent_completed_tasks (last 10 min, from progress JSONL) —
        provides per-role per-task durations.

    Returns:
        {
          "<role>": {
            "n_recent": int,                  # tap sections matching role in window
            "n_completed": int,               # JSONL chat tasks matching role in window
            "last_activity_age_s": float,     # seconds since last tap section
            "avg_tps_recent": float | None,   # mean of TIMINGS t/s across recent sections
            "avg_duration_s": float | None,   # mean chat-task duration
          },
          ...
        }

    Cheap: structured tap parse is already what the live tap polling uses; we
    just aggregate it here. Do not use legacy inference_tap.log sections for
    this panel: plaintext tap writes are not cross-process atomic and can
    interleave prompt/response sections during concurrent eval traffic.
    """
    now = now or time.time()
    if structured_requests is None:
        structured_tail = _read_tap_events_tail(_INFERENCE_TAP_EVENTS_PATH, max_bytes=1024 * 1024)
        structured_requests = _parse_structured_tap_requests(
            structured_tail,
            max_requests=160,
            now_epoch=now,
            quiet_after_s=max(15.0, min(window_s, 60.0)),
        )

    per_role: dict[str, dict[str, Any]] = {}
    live_ports = _discover_llama_ports()
    for svc in expected_stack_services():
        role = base_role(svc.get("role") or "")
        if not role:
            continue
        port = svc.get("port")
        bucket = per_role.setdefault(
            role,
            {
                "n_recent": 0,
                "n_completed": 0,
                "last_activity_age_s": None,
                "_tps_samples": [],
                "_live_tps_samples": [],
                "_duration_samples": [],
                "expected": True,
                "expected_ports": [],
                "running_ports": [],
                "running": False,
            },
        )
        if isinstance(port, int):
            if port not in bucket["expected_ports"]:
                bucket["expected_ports"].append(port)
            if port in live_ports and port not in bucket["running_ports"]:
                bucket["running_ports"].append(port)
                bucket["running"] = True

    # Seed a bucket for every role the ps(1) scan actually sees serving. The
    # topology strip builds its rows from that same scan, so without this a
    # rendered row can have NO stats object at all and fall through to "—",
    # which reads as "nothing happening" rather than "no data for this row".
    for _port, _label in live_ports.items():
        _role = base_role(_label)
        if not _role:
            continue
        _b = per_role.setdefault(
            _role,
            {
                "n_recent": 0,
                "n_completed": 0,
                "last_activity_age_s": None,
                "_tps_samples": [],
                "_live_tps_samples": [],
                "_duration_samples": [],
                "expected": False,
                "expected_ports": [],
                "running_ports": [],
                "running": False,
            },
        )
        if _port not in _b["running_ports"]:
            _b["running_ports"].append(_port)
            _b["running"] = True

    for req in structured_requests:
        role = base_role(req.get("topology_role") or req.get("role") or "")
        if not role:
            continue
        bucket = per_role.setdefault(
            role,
            {
                "n_recent": 0,
                "n_completed": 0,
                "last_activity_age_s": None,
                "_tps_samples": [],
                "_live_tps_samples": [],
                "_duration_samples": [],
                "expected_ports": [],
                "running_ports": [],
                "running": False,
            },
        )
        try:
            updated_epoch = float(req.get("updated_at_epoch") or req.get("started_at_epoch") or 0.0)
        except (TypeError, ValueError):
            updated_epoch = 0.0
        age = max(0.0, now - updated_epoch) if updated_epoch else None
        if age is not None and age <= window_s:
            bucket["n_recent"] += 1
            if bucket["last_activity_age_s"] is None or age < bucket["last_activity_age_s"]:
                bucket["last_activity_age_s"] = age
        timings_raw = req.get("timings_raw")
        if isinstance(timings_raw, dict):
            try:
                tps = float(timings_raw.get("tps") or 0.0)
            except (TypeError, ValueError):
                tps = 0.0
            if tps > 0:
                bucket["_tps_samples"].append(tps)
        elif req.get("timings"):
            m = re.search(r"([\d.]+)\s*t/s", str(req.get("timings") or ""))
            if m:
                try:
                    bucket["_tps_samples"].append(float(m.group(1)))
                except ValueError:
                    pass
        port = req.get("port")
        if isinstance(port, int) and port in live_ports and port not in bucket["running_ports"]:
            bucket["running_ports"].append(port)
            bucket["running"] = True
        elif isinstance(port, str):
            try:
                port_int = int(port)
            except ValueError:
                pass
            else:
                if port_int in live_ports and port_int not in bucket["running_ports"]:
                    bucket["running_ports"].append(port_int)
                    bucket["running"] = True
        # Live decode rate for an in-flight request (from chunk-timestamp span).
        # An open tap request proves the role is actively serving, so mark it
        # running even when its dispatched port didn't intersect the /proc scan.
        tps_live = req.get("tps_live")
        if isinstance(tps_live, (int, float)) and tps_live > 0 and req.get("status") == "running":
            bucket["_live_tps_samples"].append(float(tps_live))
            bucket["running"] = True

    # Augment from recent_completed_tasks (gives chat-XXX-tracked durations).
    log_path = _todays_progress_log()
    completed_pairs, _rolling, _cum = _scan_recent_decisions(log_path)
    _, recent_completed = _scan_orchestrator_tasks(
        log_path,
        in_flight_max_age_s=90.0,
        completed_window_s=window_s,
        max_items=80,
    )
    for t in recent_completed:
        role = base_role(t.get("final_role") or t.get("chosen_action") or "")
        if not role:
            continue
        bucket = per_role.setdefault(
            role,
            {
                "n_recent": 0,
                "n_completed": 0,
                "last_activity_age_s": None,
                "_tps_samples": [],
                "_live_tps_samples": [],
                "_duration_samples": [],
                "expected_ports": [],
                "running_ports": [],
                "running": False,
            },
        )
        bucket["n_completed"] += 1
        dur = t.get("duration_s")
        if isinstance(dur, (int, float)) and dur > 0:
            bucket["_duration_samples"].append(float(dur))
        # Per-task decode t/s (from progress-JSONL completion telemetry) so the
        # completed-request per-role average has data even when the structured
        # tap window held no `timings` event for the role.
        tps = t.get("tps")
        if isinstance(tps, (int, float)) and tps > 0:
            bucket["_tps_samples"].append(float(tps))

    # Ground-truth occupancy from llama-server /slots, folded per BASE ROLE.
    # `slots_by_port` maps port -> list of slot dicts; a port that failed the
    # poll maps to [] and is recorded as unreachable rather than as 0 busy —
    # "we could not ask" and "we asked and it said idle" are different states
    # and must not collapse into the same rendering.
    # Keyed by PORT first. A role's occupancy is then the union over the ports
    # it is served by — which is what makes ALIAS roles correct: worker_explore
    # / toolrunner are served by worker_general's llama-servers, so they must
    # report that server's real occupancy rather than "unknown". Keying by the
    # port's primary ps(1) label instead would leave every alias role blank.
    occ_by_port: dict[int, dict[str, Any]] = {}
    if slots_by_port is not None:
        for port, slots in slots_by_port.items():
            if not slots:
                # Empty list == port did not answer (or has no slots array).
                # NOT the same as "answered: zero busy" — kept distinguishable.
                occ_by_port[port] = {"answered": False, "busy": 0, "total": 0}
                continue
            n_busy = sum(1 for s in slots if s.get("is_processing"))
            occ_by_port[port] = {"answered": True, "busy": n_busy, "total": len(slots)}

    # Ports grouped by the ps(1) SERVING label. This is the key the topology
    # strip uses for its row ids (`activity_<base>`), so occupancy MUST be
    # reachable under it. The stack manifest disagrees with the ps scan about
    # role naming — the manifest files 8072 under `toolrunner` and 8082/8182
    # under `worker_explore`, while ps reports all three as `worker_general*`.
    # Keying on the manifest alone therefore left the `worker_general` strip
    # row with expected_ports=[] and no occupancy at all, which is one of the
    # reasons that row could read "idle" while its server was working.
    ps_ports_by_role: dict[str, list[int]] = {}
    for _port, _label in live_ports.items():
        _r = base_role(_label)
        if _r:
            ps_ports_by_role.setdefault(_r, []).append(_port)

    def _role_occupancy(role: str, ports: list[int]) -> dict[str, Any] | None:
        """Fold measured per-port occupancy over one role's ports.

        Union of (a) ports whose ps serving label maps to this role and
        (b) the role's manifest/observed ports, so both naming schemes
        resolve. Aliases legitimately overlap; the fleet roll-up is summed
        per port, never per role, so overlap cannot double-count.
        """
        if slots_by_port is None:
            return None
        candidates = set(ports) | set(ps_ports_by_role.get(role, []))
        seen = [p for p in sorted(candidates) if p in occ_by_port]
        if not seen:
            return None
        acc = {
            "slots_busy": 0,
            "slots_total": 0,
            "busy_ports": [],
            "unreachable_ports": [],
            "answered_ports": [],
            "busy_by_port": {},
        }
        for p in seen:
            o = occ_by_port[p]
            if not o["answered"]:
                acc["unreachable_ports"].append(p)
                continue
            acc["answered_ports"].append(p)
            acc["slots_busy"] += o["busy"]
            acc["slots_total"] += o["total"]
            acc["busy_by_port"][str(p)] = {"busy": o["busy"], "total": o["total"]}
            if o["busy"]:
                acc["busy_ports"].append(p)
        return acc

    # Collapse internal sample lists to aggregates.
    out: dict[str, dict[str, Any]] = {}
    for role, b in per_role.items():
        tps_samples = b.pop("_tps_samples")
        live_tps_samples = b.pop("_live_tps_samples", [])
        dur_samples = b.pop("_duration_samples")
        b["expected_ports"] = sorted(b.get("expected_ports") or [])
        b["running_ports"] = sorted(b.get("running_ports") or [])
        b["expected_instance_count"] = len(b["expected_ports"])
        b["running_instance_count"] = len(b["running_ports"])
        # `running*` above is REACHABILITY ONLY (ps scan). Occupancy below is
        # the separate, stronger claim and is the only one the UI may render
        # as busy/idle.
        # Prefer the role's own expected ports (covers alias roles and roles
        # whose listener is momentarily absent from the ps scan); fall back to
        # the ports actually seen running for it.
        occ = _role_occupancy(role, list(b["expected_ports"]) + list(b["running_ports"]))
        if occ is None:
            b["slots_busy"] = None
            b["slots_total"] = None
            b["busy_ports"] = []
            b["busy_by_port"] = {}
            b["unreachable_ports"] = []
            b["occupancy_observed"] = False
            b["busy"] = None
        else:
            b["slots_busy"] = occ["slots_busy"]
            b["slots_total"] = occ["slots_total"]
            b["busy_ports"] = sorted(occ["busy_ports"])
            b["busy_by_port"] = occ["busy_by_port"]
            b["unreachable_ports"] = sorted(occ["unreachable_ports"])
            # Observed only if at least one instance actually answered /slots.
            b["occupancy_observed"] = bool(occ["answered_ports"])
            b["busy"] = (occ["slots_busy"] > 0) if occ["answered_ports"] else None
        b["avg_tps_recent"] = (sum(tps_samples) / len(tps_samples)) if tps_samples else None
        # Live in-flight decode rate (mean of running instances' estimates),
        # kept distinct from avg_tps_recent (completed-request average) so the
        # strip can show a real-time number mid-generation instead of blank.
        b["live_tps"] = (
            (sum(live_tps_samples) / len(live_tps_samples)) if live_tps_samples else None
        )
        b["live_tps_n"] = len(live_tps_samples)
        b["avg_duration_s"] = (sum(dur_samples) / len(dur_samples)) if dur_samples else None
        out[role] = b
    # Fleet-level occupancy roll-up, so a consumer never has to re-derive it
    # (and so the strip header and the load panel cannot disagree about it).
    # Summed over PORTS, not over roles: alias roles share their host's ports,
    # so folding the per-role numbers here would double-count them.
    fleet_busy = sum(o["busy"] for o in occ_by_port.values() if o["answered"])
    fleet_total = sum(o["total"] for o in occ_by_port.values() if o["answered"])
    any_observed = any(o["answered"] for o in occ_by_port.values())
    return _stamp(
        {
            "per_role": out,
            "window_s": window_s,
            "now": now,
            # Names the SOURCE of the busy/idle claim so the operator can tell
            # a real measurement from an absent one at a glance.
            "occupancy_source": "llama_server_slots"
            if slots_by_port is not None
            else "unavailable",
            "occupancy_observed": any_observed,
            "fleet_slots_busy": fleet_busy if any_observed else None,
            "fleet_slots_total": fleet_total if any_observed else None,
        },
        "topology_activity",
        now=now,
    )


@router.get("/dashboard/api/topology_activity")
async def topology_activity(window_s: float = 600.0) -> JSONResponse:
    # Poll llama-server /slots here so the STANDALONE panel carries the same
    # ground-truth occupancy as the coherent snapshot. Without this the panel
    # can only ever report reachability, which is what made a saturated role
    # render as "idle · servers up".
    try:
        slots_by_port, _meta = await _poll_all_slots()
    except Exception:
        slots_by_port = None  # unknown, NOT zero — renders as "activity unknown"
    return JSONResponse(_topology_activity_payload(window_s=window_s, slots_by_port=slots_by_port))


def _build_topology_nodes(numa_mode: str | None = None) -> list[dict[str, Any]]:
    """Build the topology node list (roles, ports, colors, backing models).

    Factored out of ``/dashboard/api/topology`` so the snapshot endpoint can
    embed the SAME structure under one ``generated_at``. The topology strip and
    the lock/activity overlays that decorate it then derive from one coherent
    object and cannot disagree about which roles exist — this is what closes the
    post-reboot 'frozen strip' class of dashboard staleness, where the strip was
    fetched on a different cadence than the overlays keyed to it.
    """
    active_mode = numa_mode or active_stack_numa_mode()
    llama_ports = _discover_llama_ports()
    llama_models = _discover_llama_models()
    # M3 attribution metadata (fleet-marker / launch-contract / unmanaged /
    # …) per discovered listener. Kept as a SEPARATE lookup so tests that
    # monkeypatch `_discover_llama_ports` keep working unchanged — ports absent
    # from this map simply render without attribution fields, exactly the
    # pre-M3 node shape.
    try:
        llama_procs = _discover_llama_processes()
    except Exception:
        llama_procs = {}
    services = _load_state_services()
    expected_services = expected_stack_services(active_mode)
    expected_by_port = {
        svc["port"]: svc for svc in expected_services if isinstance(svc.get("port"), int)
    }
    seen_ports: set[int] = set()
    nodes: list[dict[str, Any]] = []

    # Orchestrator at the center.
    nodes.append(
        {
            "id": "orchestrator",
            "label": "orchestrator",
            "role": "orchestrator",
            "port": 8000,
            "color": _role_color("orchestrator"),
            "kind": "orchestrator",
        }
    )
    seen_ports.add(8000)

    # Llama-servers from /proc scan.
    for port, role in sorted(llama_ports.items()):
        if port in seen_ports:
            continue
        expected = expected_by_port.get(port, {})
        seen_ports.add(port)
        if role.startswith("mi210_"):
            # Direct-access GPU testbed: first-class node ahead of stack
            # integration (operator-decided 2026-07-05). Its traffic bypasses
            # the orchestrator pipeline, so slot activity is visible but the
            # structured tap won't carry its tokens until it's stack-routed.
            node_kind = "gpu-llama-server"
        elif role.startswith("extern_"):
            node_kind = "external-llama-server"
        else:
            node_kind = "llama-server"
        node = {
            "id": f"port_{port}",
            "label": role,
            "role": role,
            "port": port,
            "color": _role_color(role),
            "kind": node_kind,
            # Model actually loaded by this llama-server (-m GGUF basename,
            # vendor-prefix + shard-suffix stripped). Surfaced so the topology
            # strip can label each role with its backing model + quant.
            "model": llama_models.get(port, ""),
            # Alias roles served by the same process (e.g. frontdoor port 8070
            # also serves coder_escalation + worker_summarize). Surfaced so the
            # dashboard can render them under the primary role label.
            "aliases": role_aliases(role),
            "expected": bool(expected),
            "running": True,
            "manifest_roles": expected.get("roles", []),
        }
        # M3: how the stack vouches for this listener (fleet-marker /
        # launch-contract / service-hint / unmanaged / unverified). A demoted
        # extern node additionally carries `lane_hint` — the configured lane
        # label it would have (mis)inherited — and `marker_stale` when a
        # left-behind marker for a previous process exists on its port.
        proc = llama_procs.get(port) or {}
        if proc.get("attribution"):
            node["attribution"] = proc["attribution"]
        if proc.get("lane_hint"):
            node["lane_hint"] = proc["lane_hint"]
        if proc.get("marker_stale"):
            node["marker_stale"] = True
        # Substrate (gpu/cpu) from the process's binary path — role names carry
        # no substrate (architect_general was a GPU role until the 2026-09-27
        # ARCHITECT SWAP made it the CPU :8074 process, and nothing in its name
        # says so), so the page must not be left to guess from role lists.
        # `substrate_source` says WHICH evidence produced the value: "process"
        # (binary path / /proc), "role-kind" (the mi210 testbed kind), or —
        # for expected-but-not-running nodes below — "manifest" (declared, not
        # observed). A declared substrate must never read as an observed one.
        if proc.get("substrate"):
            node["substrate"] = proc["substrate"]
            node["substrate_source"] = "process"
        elif node_kind == "gpu-llama-server":
            node["substrate"] = "gpu"
            node["substrate_source"] = "role-kind"
        nodes.append(node)

    # Auxiliary services not already covered.
    for svc in services:
        port = svc.get("port")
        if not port or port in seen_ports:
            continue
        expected = expected_by_port.get(port, {})
        seen_ports.add(port)
        svc_node = {
            "id": svc["name"],
            "label": svc["name"],
            "role": svc["role"],
            "port": port,
            "color": _role_color(svc["role"]),
            "kind": "service",
            "model": _clean_model_name(svc.get("model", "")),
            "expected": bool(expected),
            "running": bool(svc.get("running")),
            "manifest_roles": expected.get("roles", []),
        }
        if svc.get("substrate"):
            svc_node["substrate"] = svc["substrate"]
            svc_node["substrate_source"] = "process"
        nodes.append(svc_node)

    # Expected stack servers that are not currently visible via /proc or the
    # state file still get topology rows so the activity panel can show them as
    # expected/down/no recent activity instead of silently omitting them.
    for svc in expected_services:
        port = svc.get("port")
        role = str(svc.get("role") or "")
        if not port or port in seen_ports or not role:
            continue
        seen_ports.add(port)
        expected_node: dict[str, Any] = {
            "id": f"expected_{port}",
            "label": svc.get("name") or role,
            "role": role,
            "port": port,
            "color": _role_color(role),
            "kind": "expected-stack-server",
            "model": "",
            "aliases": [r for r in svc.get("roles", [])[1:] if isinstance(r, str)],
            "expected": True,
            "running": False,
            "manifest_roles": svc.get("roles", []),
            "embedding": bool(svc.get("embedding")),
            "vision": bool(svc.get("vision")),
            "worker_pool": bool(svc.get("worker_pool")),
        }
        # No process exists for this node, so no process evidence can. Take the
        # substrate from the manifest DECLARATION and say so — the page must not
        # fall to its port/name heuristic, and the value must not read as
        # observed (RTG-47 data plane).
        declared = manifest_declared_substrate(svc.get("roles") or [role])
        if declared:
            expected_node["substrate"] = declared["substrate"]
            expected_node["substrate_source"] = declared["source"]
            expected_node["substrate_declared_by"] = declared["declared_by"]
        nodes.append(expected_node)

    return nodes


_TOPOLOGY_NODES_CACHE: dict[str, Any] = {"ts": 0.0, "nodes": None}
_TOPOLOGY_NODES_TTL_S = 3.0

# Region locks + port discovery feed BOTH the 2 Hz snapshot tick and the
# structured-tap producers (per SSE connection, per worker). Recomputing them
# per call means one slow /proc pass or ps(1) scan stalls the topology,
# region-locks AND live-tap panels together — the exact trio-staleness failure
# domain. TTLs sit far below the client's 12s/30s badge escalation thresholds.
_REGION_LOCKS_CACHE: dict[str, Any] = {"ts": 0.0, "payload": None}
_REGION_LOCKS_TTL_S = 1.0
_PORT_ROLES_CACHE: dict[str, Any] = {"ts": 0.0, "ports": None}
_PORT_ROLES_TTL_S = 2.0


def _region_locks_cached(numa_mode: str | None = None) -> dict[str, Any]:
    """TTL-cached `_region_locks_payload()` that fails open to the last good
    payload (marked `stale_cache` + `error`) instead of raising into the
    serve path."""
    active_mode = numa_mode or active_stack_numa_mode()
    now = time.time()
    c = _REGION_LOCKS_CACHE
    payload = c.get("payload")
    if (
        payload is not None
        and c.get("numa_mode") == active_mode
        and (now - c["ts"]) < _REGION_LOCKS_TTL_S
    ):
        return payload
    try:
        payload = _region_locks_payload(active_mode)
    except Exception as exc:
        stale = c.get("payload")
        if stale is None:
            return {"error": str(exc), "entries": [], "by_role": {}}
        return {**stale, "error": str(exc), "stale_cache": True}
    c["ts"] = now
    c["numa_mode"] = active_mode
    c["payload"] = payload
    return payload


def _port_roles_cached() -> dict[int, str]:
    """TTL-cached `_discover_llama_ports()` (a ps(1) subprocess per call)."""
    now = time.time()
    c = _PORT_ROLES_CACHE
    ports = c.get("ports")
    if ports is not None and (now - c["ts"]) < _PORT_ROLES_TTL_S:
        return ports
    ports = _discover_llama_ports()
    c["ts"] = now
    c["ports"] = ports
    return ports


def _topology_nodes_cached(numa_mode: str | None = None) -> list[dict[str, Any]]:
    """Topology nodes with a short TTL.

    Topology STRUCTURE (which roles/ports exist) changes only on stack
    changes/reboots, so rebuilding it (~90ms of /proc scanning) on every 2 Hz
    snapshot tick — across 6 uvicorn workers — would burn real CPU on the
    inference host for no benefit. A 3s TTL keeps the snapshot cheap while still
    reflecting a stack change within 3s. The live overlays (region locks, slot
    activity) are still rebuilt every tick, so per-role liveness stays
    real-time; only the node scaffold is cached. Per-worker cache (separate
    processes) is fine — the structure is identical across workers.
    """
    active_mode = numa_mode or active_stack_numa_mode()
    now = time.time()
    c = _TOPOLOGY_NODES_CACHE
    nodes = c.get("nodes")
    if (
        nodes is not None
        and c.get("numa_mode") == active_mode
        and (now - c["ts"]) < _TOPOLOGY_NODES_TTL_S
    ):
        return nodes
    nodes = _build_topology_nodes(active_mode)
    c["ts"] = now
    c["numa_mode"] = active_mode
    c["nodes"] = nodes
    return nodes


@router.get("/dashboard/api/topology")
async def topology() -> JSONResponse:
    """Return the live topology: nodes with role + display color + port.

    Uncached: the standalone endpoint is polled only every 5s, so its ~90ms
    rebuild is cheap here, and staying uncached keeps it deterministic w.r.t.
    injected inputs (the 2 Hz snapshot is the path that needs the TTL cache).
    """
    _topo_now = time.time()
    resolution = active_stack_numa_mode_resolution()
    active_mode = resolution["mode"]
    return JSONResponse(
        _stamp(
            {
                "nodes": _build_topology_nodes(active_mode),
                "generated_at": _topo_now,
                "stack_numa_mode": active_mode,
                # Provenance so the operator sees which source won and whether a
                # lower-precedence source (env/manifest) contradicts the realized
                # fleet — the C1 defense against a lying env silently inverting
                # the fleet-health picture.
                "stack_numa_mode_provenance": resolution,
            },
            "topology",
            now=_topo_now,
        )
    )


# ---------------------------------------------------------------------------
# Snapshot endpoint — point-in-time state of all slots + counters
# ---------------------------------------------------------------------------


async def _poll_slot(client: httpx.AsyncClient, port: int) -> list[dict[str, Any]] | None:
    """Fetch /slots from a single llama-server.

    Returns None on ANY failure (timeout, non-200, malformed body) — NOT [].
    The distinction is load-bearing: [] means "this server answered and has no
    slots", None means "we could not measure it". Collapsing the two made a
    timing-out server indistinguishable from an idle one, and a llama-server
    that is busy DECODING is precisely the one most likely to miss this
    deadline — so the old behaviour dropped busy servers out of the occupancy
    count and reported the remainder as if it were the whole fleet.
    (Observed 2026-08-04: port 8182 timed out on nearly every poll while
    generating, and `slots_poll_meta` still reported answered=18, timed_out=0.)
    """
    try:
        # Split connect budget out of the 1.5s total: a SYN-blackholed port
        # must not consume the whole per-request budget before first byte.
        resp = await client.get(
            f"http://127.0.0.1:{port}/slots",
            # 2026-08-04: raised 1.5s -> 8s. llama-server answers /slots on the SAME
            # event loop that runs the batch, so a server inside a long prefill cannot
            # reply. Measured: port 8182 at n_prompt_tokens=45057 timed out at 10s and
            # answered in 0.9s once given 25s. At 1.5s the BUSIEST server was the one
            # most likely to vanish from the occupancy count — the dashboard got more
            # idle-looking the harder the fleet worked.
            #
            # 8s is a deliberate compromise, not a cure: no timeout reliably catches a
            # server mid-45k-token prefill, and this endpoint is polled on a UI cadence
            # so it cannot be unbounded. Ports still missed are now reported as
            # `unanswered_ports` and the count is labelled a lower bound, so the blind
            # spot is visible rather than silently folded into "idle".
            timeout=httpx.Timeout(8.0, connect=0.5),
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, list):
            return None
        return data
    except Exception:
        return None


# Overall wall-clock budget for the ~29-port fan-out. Per-request timeouts do
# not bound the aggregate under load; snapshot() sits on the serve path of the
# topology / region-locks / live-tap panels, so an unbounded fan-out stales all
# three at once. Ports that miss the deadline report [] and are counted in
# slots_poll_meta rather than dropped silently.
_SLOTS_FANOUT_DEADLINE_S = 10.0

# Last successful /slots sample, shared with the (synchronous) tap funnel so it
# can attach live prefill/decode progress WITHOUT triggering its own fan-out —
# the 2 Hz snapshot stream keeps this warm whenever anyone is watching. RTG-47:
# `slot_progress` is how the machine page shows "tokens ingested so far" during
# a long prefill (the tap itself is silent until the first token).
_LAST_SLOTS: dict[str, Any] = {"ts": 0.0, "by_port": {}}
_SLOT_PROGRESS_FRESH_S = 15.0


def _attach_slot_progress(
    requests: list[dict[str, Any]],
    *,
    slots_by_port: dict[int, list[dict[str, Any]]] | None = None,
    sampled_at: float | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Attach live ``slot_progress`` to tap requests from the last /slots sample.

    Attach-only-from-evidence rules:
    * the sample must be FRESH (≤ ``_SLOT_PROGRESS_FRESH_S``) — a stale sample
      attaches nothing rather than asserting yesterday's prefill;
    * only ``is_processing`` slots count — a finished slot retains its counters
      and attributing it to a request would revive a dead task's numbers;
    * a ``complete`` request never attaches — the busy slot on its port belongs
      to whichever request is still running there, not to it;
    * exactly one processing slot AND one live candidate request on the port ⇒
      unambiguous; several of either ⇒ port-level AGGREGATE with
      ``ambiguous: true`` (the tap does not know llama's internal task ids, so
      per-slot attribution would be a guess); none ⇒ no field at all.
    """
    now = time.time() if now is None else now
    if slots_by_port is None:
        sampled_at = float(_LAST_SLOTS.get("ts") or 0.0)
        slots_by_port = _LAST_SLOTS.get("by_port") or {}
    else:
        sampled_at = now if sampled_at is None else sampled_at
    if not slots_by_port or (now - sampled_at) > _SLOT_PROGRESS_FRESH_S:
        return requests

    def _req_port(req: dict[str, Any]) -> int | None:
        try:
            return int(req.get("port")) if req.get("port") not in (None, "") else None
        except (TypeError, ValueError):
            return None

    live_on_port: dict[int, int] = {}
    for req in requests:
        port = _req_port(req)
        if port is not None and str(req.get("status") or "") != "complete":
            live_on_port[port] = live_on_port.get(port, 0) + 1

    out_reqs: list[dict[str, Any]] = []
    for req in requests:
        out = req
        port = _req_port(req)
        terminal = str(req.get("status") or "") == "complete"
        busy = [
            s for s in (slots_by_port.get(port) or [])
            if isinstance(s, dict) and s.get("is_processing")
        ] if port is not None and not terminal else []
        if busy:
            def _decoded(slot: dict[str, Any]) -> int | None:
                nxt = slot.get("next_token")
                if isinstance(nxt, list) and nxt and isinstance(nxt[0], dict):
                    val = nxt[0].get("n_decoded")
                    return int(val) if isinstance(val, (int, float)) else None
                return None
            decoded = [d for d in (_decoded(s) for s in busy) if d is not None]
            out = dict(req)
            out["slot_progress"] = {
                "is_processing": True,
                "n_prompt_tokens": sum(int(s.get("n_prompt_tokens") or 0) for s in busy),
                "n_prompt_tokens_processed": sum(
                    int(s.get("n_prompt_tokens_processed") or 0) for s in busy),
                "n_decoded": sum(decoded) if decoded else None,
                "sampled_at": round(sampled_at, 3),
                "ambiguous": len(busy) > 1 or live_on_port.get(port, 0) > 1,
            }
        out_reqs.append(out)
    return out_reqs


async def _poll_all_slots() -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Concurrently poll /slots on every discovered llama-server port.

    Returns (slots_by_port, slots_poll_meta). Every discovered port is present
    in slots_by_port; ports that errored or missed the fan-out deadline map to
    [] and are tallied in the meta.

    `answered` counts ports that RETURNED SLOT DATA, not ports whose coroutine
    completed. Previously a per-request timeout was swallowed inside
    `_poll_slot` (which returned []), so its task completed normally and was
    counted as answered — the meta reported a clean fan-out while a port
    contributed nothing. `unanswered_ports` names them explicitly so a caller
    can mark the affected roles' occupancy as a LOWER BOUND instead of
    silently shrinking the denominator.
    """
    ports = list(_discover_llama_ports().keys())
    out: dict[int, list[dict[str, Any]]] = {}
    started = time.time()
    if not ports:
        return out, {
            "ports": 0,
            "answered": 0,
            "timed_out": 0,
            "duration_s": 0.0,
            "unanswered_ports": [],
        }
    unanswered: list[int] = []
    async with httpx.AsyncClient() as client:
        tasks = {port: asyncio.ensure_future(_poll_slot(client, port)) for port in ports}
        await asyncio.wait(tasks.values(), timeout=_SLOTS_FANOUT_DEADLINE_S)
        answered = 0
        for port, task in tasks.items():
            result = None
            if task.done() and not task.cancelled() and task.exception() is None:
                result = task.result()
            else:
                task.cancel()
            if result is None:
                # Could not measure: deadline miss, per-request timeout,
                # non-200, or malformed body. NOT "answered with 0 slots".
                out[port] = []
                unanswered.append(port)
            else:
                out[port] = result
                answered += 1
    meta = {
        "ports": len(ports),
        "answered": answered,
        "timed_out": len(ports) - answered,
        "duration_s": round(time.time() - started, 3),
        "unanswered_ports": sorted(unanswered),
    }
    # Keep the last sample warm for the synchronous tap funnel (slot_progress).
    # Updated even on a poor fan-out: unanswered ports carry [], which attaches
    # nothing — equivalent to stale, without pinning yesterday's numbers.
    if answered:
        _LAST_SLOTS.update({"ts": time.time(), "by_port": out})
    return out, meta


# Snapshot scanners moved to dashboard_snapshot.py — wrappers preserve in-file API
# (route handlers below call _todays_progress_log() / _scan_recent_decisions() / etc.).


def _todays_progress_log() -> Path:
    return _todays_progress_log_impl(PROGRESS_LOG_DIR)


def _scan_recent_decisions(
    path: Path,
    window_s: float = 600.0,
    max_items: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    # 2026-05-25: window widened 50 → 200 — at typical chat traffic the
    # last-50 cut was overwhelmingly frontdoor (~93% of all chat hits today),
    # making the panel look like routing was frontdoor-only when in reality
    # 6+ distinct base roles get hit per day. 200 gives a more honest mix
    # without paying the cost of scanning the whole day's JSONL.
    return _scan_recent_decisions_impl(path, window_s=window_s, max_items=max_items)


def _scan_orchestrator_tasks(
    path: Path,
    in_flight_max_age_s: float = INFLIGHT_MAX_AGE_DEFAULT_S,
    completed_window_s: float = 600.0,
    max_items: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _scan_orchestrator_tasks_impl(
        path,
        in_flight_max_age_s=in_flight_max_age_s,
        completed_window_s=completed_window_s,
        max_items=max_items,
    )


# Always-show window for in-flight tasks (Fix 3). A task younger than this is
# kept regardless of live slot state: its llama-server slot may not have flipped
# to is_processing yet (the /slots poll is 2 Hz), or the poll briefly missed the
# server. Older tasks must be corroborated by a busy slot on their role's
# server, which is what drops restart-orphans.
_FRESH_INFLIGHT_S = 20.0


def _gate_inflight_by_live_slots(
    tasks: list[dict[str, Any]],
    role_busy: dict[str, int],
    alias_to_topology_role: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Reconcile JSONL-derived in-flight tasks with live slot occupancy.

    The progress JSONL alone cannot tell a genuinely-running task from a
    restart-orphan (a `task_started` whose terminal event was lost when the API
    restarted). The live `/slots` poll can: if a role's server reports zero
    busy slots, nothing for that role is actually in flight. So per base role we
    keep at most ``max(busy_slots, fresh_tasks)`` of the newest started-but-
    unterminated tasks. This makes the in-flight list, the active badge, and the
    slot dots agree, hides stale orphans, and keeps long-running tasks visible
    for as long as their slot stays busy.
    """
    alias_to_topology_role = alias_to_topology_role or {}

    by_role: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        logical_role = _canonical_role_name(t.get("role") or "unknown")
        topology_role = alias_to_topology_role.get(logical_role, logical_role)
        by_role.setdefault(topology_role, []).append(t)
    gated: list[dict[str, Any]] = []
    for role, group in by_role.items():
        group.sort(key=lambda x: x.get("age_s", 0.0))  # newest first
        busy = role_busy.get(role, 0)
        n_fresh = sum(1 for t in group if t.get("age_s", 0.0) <= _FRESH_INFLIGHT_S)
        for idx, task in enumerate(group[: max(busy, n_fresh)]):
            annotated = dict(task)
            raw_role = annotated.get("role") or "unknown"
            logical_role = _canonical_role_name(raw_role)
            topology_role = alias_to_topology_role.get(logical_role, logical_role)
            if topology_role != str(raw_role):
                annotated["topology_role"] = topology_role
            if idx < busy:
                annotated["live_state"] = "decoding"
                annotated["live_state_reason"] = "live slot busy for topology role"
            else:
                annotated["live_state"] = "pending"
                annotated["live_state_reason"] = (
                    "fresh task_started with no busy slot yet; likely queued, "
                    "pre-inference, or local post-processing"
                )
            gated.append(annotated)
    gated.sort(key=lambda x: x.get("age_s", 0.0))
    return gated


def _coherent_display_activity(
    activity: dict[int, dict[str, Any]],
    *,
    structured_requests: list[dict[str, Any]],
    region_locks: dict[str, Any],
    port_roles: dict[int, str],
    topology_nodes: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Return activity suitable for live dashboard painting.

    Raw llama-server `/slots` is useful diagnostic telemetry, but it is not an
    authoritative CPU-holder signal: slots can remain busy/ambiguous while the
    structured tap and `/proc` lock scan have already moved on. The dashboard's
    visible active topology state must therefore be corroborated by either a
    current structured tap request or a current CPU-region lock. Device/direct
    access servers are kept as raw slot occupancy because they intentionally
    bypass the orchestrator tap/lock path.
    """
    active_ports: set[int] = set()
    for req in structured_requests:
        if str(req.get("status") or "").lower() != "running":
            continue
        try:
            quiet_s = float(req.get("quiet_s") or 0.0)
        except (TypeError, ValueError):
            quiet_s = 0.0
        if quiet_s >= 15.0:
            continue
        try:
            active_ports.add(int(req.get("port")))
        except (TypeError, ValueError):
            pass

    locks_by_role = region_locks.get("by_role") if isinstance(region_locks, dict) else {}
    locks_by_role = locks_by_role if isinstance(locks_by_role, dict) else {}
    for port, role_label in port_roles.items():
        role, shape = _port_role_shape(role_label, port)
        if not role:
            continue
        info = locks_by_role.get(role) or {}
        active_idxs = {
            int(i) for i in info.get("active_instance_idxs", []) if str(i).lstrip("-").isdigit()
        }
        if not active_idxs:
            continue
        instances = info.get("instances") if isinstance(info.get("instances"), list) else []
        for inst in instances:
            try:
                idx = int(inst.get("idx"))
            except (AttributeError, TypeError, ValueError):
                continue
            if idx not in active_idxs:
                continue
            inst_shape = str(inst.get("shape") or "")
            if shape and inst_shape != shape:
                continue
            # A shape-less base port represents the primary/full endpoint; do
            # not let a quarter holder make the base row look active.
            if not shape and inst_shape not in {"", "full", "half0"}:
                continue
            active_ports.add(int(port))

    node_kind_by_port = {
        int(n["port"]): str(n.get("kind") or "")
        for n in topology_nodes
        if isinstance(n, dict) and isinstance(n.get("port"), int)
    }
    display: dict[int, dict[str, Any]] = {}
    for port, entry in activity.items():
        item = dict(entry)
        kind = node_kind_by_port.get(int(port), "")
        is_cpu_llama = kind == "llama-server"
        if is_cpu_llama and int(port) not in active_ports:
            item["n_active"] = 0
            item["active_slots"] = []
        display[int(port)] = item
    return display


def _count_log_events(
    path: Path,
    patterns: dict[str, str],
    window_s: float = 600.0,
) -> dict[str, int]:
    return _count_log_events_impl(path, patterns, window_s=window_s)


# Per-worker serve-path vitals for /dashboard/api/health. The health endpoint
# otherwise only stats producer FILES, so a hang or exception inside the
# snapshot serve path (the topology/region-locks/live-tap failure domain) was
# invisible to it — panels froze while health stayed green.
_SNAPSHOT_BUILD_STATS: dict[str, Any] = {
    "last_attempt_ts": None,
    "last_success_ts": None,
    "last_duration_s": None,
    "last_error": None,
    "last_error_ts": None,
    "build_count": 0,
}
_SNAPSHOT_SERVE_PATH_STALL_S = 30.0


@router.get("/dashboard/api/snapshot")
async def snapshot() -> JSONResponse:
    """Point-in-time state: all slots + recent decisions + counters."""
    stats = _SNAPSHOT_BUILD_STATS
    started = time.time()
    stats["last_attempt_ts"] = started
    try:
        resp = await _snapshot_impl()
    except Exception as exc:
        stats["last_error"] = str(exc)
        stats["last_error_ts"] = time.time()
        raise
    done = time.time()
    stats["last_success_ts"] = done
    stats["last_duration_s"] = round(done - started, 3)
    stats["build_count"] += 1
    return resp


@router.get("/dashboard/api/live_activity")
async def live_activity() -> JSONResponse:
    """Read-only API request lifecycle counts for safe batch drain checks."""
    return JSONResponse(live_batch_activity_summary())


async def _snapshot_impl() -> JSONResponse:
    numa_resolution = active_stack_numa_mode_resolution()
    active_mode = numa_resolution["mode"]
    progress_log = _todays_progress_log()
    recent, rolling, cumulative = _scan_recent_decisions(progress_log)
    orch_log = ORCHESTRATOR_LOG_DIR / "orchestrator.log"
    log_counts = _count_log_events(
        orch_log,
        {
            "inference_aborted": r"Inference aborted",
            "inference_lock_timeout": r"Inference lock timeout",
            "slot_erase": r"erasing slots on holder port",
            "watchdog_force_release": r"Lock hold watchdog: force-releasing",
        },
    )

    # Live-plane coherence bracket. Placement is sampled on both sides of the
    # independently polled backend watchdog, with one bounded retry if the
    # owner changes during that window. Raw disagreements survive into the
    # frame; this only determines whether their timing is certifiable.
    live_sample = await _sample_slots_with_placement_window(active_mode)
    slots_by_port = live_sample["slots_by_port"]
    slots_poll_meta = live_sample["slots_poll_meta"]

    # Derive per-node activity from slot states + live busy slots per base role.
    port_roles = _discover_llama_ports()
    role_busy: dict[str, int] = {}
    alias_to_topology_role: dict[str, str] = {}
    for role_label in set(port_roles.values()):
        role = _canonical_role_name(role_label)
        if not role:
            continue
        try:
            for alias in role_aliases(role):
                alias_base = _canonical_role_name(alias)
                if alias_base:
                    alias_to_topology_role[alias_base] = role
        except Exception:
            continue
    activity: dict[int, dict[str, Any]] = {}
    for port, slots in slots_by_port.items():
        n_total = len(slots)
        n_active = sum(1 for s in slots if s.get("is_processing"))
        role = _canonical_role_name(port_roles.get(port, ""))
        if role:
            role_busy[role] = role_busy.get(role, 0) + n_active
        active_slots: list[dict[str, Any]] = []
        for s in slots:
            if not s.get("is_processing"):
                continue
            # v6 /slots dropped prompt/content — those fields are permanently
            # empty. Token text lives in the structured tap; slots carry only
            # occupancy + token counts.
            active_slots.append(
                {
                    "slot_id": s.get("id"),
                    "task_id": s.get("id_task") if s.get("id_task", -1) >= 0 else None,
                    "tokens_decoded": s.get("n_decoded"),
                    "prompt_tokens": s.get("n_prompt_tokens"),
                    "next_token": s.get("next_token"),
                }
            )
        activity[port] = {
            "n_total": n_total,
            "n_active": n_active,
            "active_slots": active_slots,
        }

    in_flight_tasks, recent_completed_tasks = _scan_orchestrator_tasks(progress_log)
    # Gate in-flight tasks on live slot occupancy so the task list, the active
    # badge, and the slot dots can't disagree (drops restart-orphans).
    in_flight_tasks = _gate_inflight_by_live_slots(
        in_flight_tasks,
        role_busy,
        alias_to_topology_role=alias_to_topology_role,
    )

    _snap_now = time.time()
    region_locks = live_sample["placement_after"]
    placement_telemetry = _placement_telemetry_status(
        activity=activity,
        port_roles=port_roles,
        region_locks=region_locks,
    )
    region_locks["placement_telemetry"] = placement_telemetry
    structured_requests = _structured_tap_requests_for_dashboard(
        max_requests=80,
        now_epoch=_snap_now,
        region_locks=region_locks,
        port_roles=port_roles,
    )
    tap_observed_at = time.time()
    reducer_frame = live_telemetry_frame()
    live_frame = _annotate_live_observability_frame(
        region_locks=region_locks,
        activity=activity,
        port_roles=port_roles,
        structured_requests=structured_requests,
        placement_telemetry=placement_telemetry,
        sample=live_sample,
        tap_observed_at=tap_observed_at,
        event_frame=reducer_frame,
    )
    topology_nodes = _topology_nodes_cached(active_mode)
    display_activity = _coherent_display_activity(
        activity,
        structured_requests=structured_requests,
        region_locks=region_locks,
        port_roles=port_roles,
        topology_nodes=topology_nodes,
    )
    topology_activity_payload = _topology_activity_payload(
        window_s=600.0,
        now=_snap_now,
        structured_requests=structured_requests,
        # Reuse THIS tick's fan-out: the strip's busy/idle claim and the
        # `activity` slot dots then derive from one and the same /slots read
        # and cannot disagree within a frame.
        slots_by_port=slots_by_port,
    )
    return JSONResponse(
        _stamp(
            {
                "generated_at": _snap_now,
                "frame_id": live_frame["frame_id"],
                "live_frame": live_frame,
                "api_lifecycle": reducer_frame.get("batch_activity") or {},
                # Coherent live correlation: topology + region locks + activity are all
                # built within THIS single snapshot() call and stamped with one
                # generated_at, so the strip, the region-lock grid, and the slot/activity
                # dots the frontend renders from this one object can never reflect
                # different instants. This is the keystone that kills the "tap shows
                # active inference beside a 'no locks held' grid" inconsistency.
                "stack_numa_mode": active_mode,
                "stack_numa_mode_provenance": numa_resolution,
                "topology": {
                    "nodes": topology_nodes,
                    "stack_numa_mode": active_mode,
                    "stack_numa_mode_provenance": numa_resolution,
                },
                "activity": activity,
                "display_activity": display_activity,
                "placement_telemetry": placement_telemetry,
                # Fan-out degradation is data, not a silent gap: timed_out > 0 means
                # some ports' slots are missing from `activity` this frame.
                "slots_poll_meta": slots_poll_meta,
                "in_flight_tasks": in_flight_tasks,
                "recent_completed_tasks": recent_completed_tasks,
                "structured_requests": structured_requests,
                "tap_active": _structured_tap_active(structured_requests),
                "structured_tap_mtime": _latest_tap_events_mtime(),
                "topology_activity": topology_activity_payload,
                "live_busy_by_role": role_busy,
                "region_locks": region_locks,
                "recent_decisions": recent,
                "source_counts_rolling": rolling,
                "source_counts_cumulative": cumulative,
                "log_counts": log_counts,
            },
            "snapshot",
            now=_snap_now,
        )
    )


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


@router.get("/dashboard/events/stream")
async def stream(request: Request) -> StreamingResponse:
    """Push-first Server-Sent Events stream of authoritative live frames."""

    async def event_gen():
        async for payload in _snapshot_payloads():
            if await request.is_disconnected():
                return
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Multiplexed event stream
#
# A browser caps HTTP/1.1 at ~6 concurrent connections per host. The dashboard
# opened FIVE always-on EventSources (snapshot, structured_tap, raw_tap,
# autopilot_log, planner_tap), leaving almost no room for fetch() polling and
# fully starving a second tab (2 x 5 = 10 > 6). This endpoint fans every
# always-on source into ONE SSE connection, tagging each with a named SSE event
# so the client dispatches by name. One tab now holds one connection.
#
# The standalone /dashboard/events/{stream,structured_tap,raw_tap,autopilot_log,
# planner_tap} endpoints are retained: they still back the client's one-shot
# initial-payload fetch fallback, per-stream curl debugging, and the legacy
# multi-connection path (client kill-switch sseMultiplex=0).
# ---------------------------------------------------------------------------

_PLANNER_TAP_PATH = Path("/mnt/raid0/llm/tmp/planner_tap.log")


async def _snapshot_payloads():
    """Yield transition-driven snapshots, coalesced to at most 2 Hz.

    Runs until cancelled — the multiplex main loop is the SOLE reader of the
    request's disconnect channel; concurrent is_disconnected() reads across
    producers would race the ASGI receive channel.  A two-second watchdog frame
    independently re-polls ``/slots`` even when no lifecycle transition lands.
    """
    last_event_sequence: tuple[tuple[str, int], ...] = ()
    last_emit_monotonic = 0.0
    while True:
        now_monotonic = time.monotonic()
        event_sequence = live_telemetry_sequence()
        transition_due = event_sequence != last_event_sequence
        watchdog_due = now_monotonic - last_emit_monotonic >= 2.0
        coalesce_ready = now_monotonic - last_emit_monotonic >= 0.5
        if watchdog_due or (transition_due and coalesce_ready):
            try:
                resp = await snapshot()
                payload = resp.body.decode("utf-8")  # type: ignore[union-attr]
                payload_frame = json.loads(payload)
                lifecycle = (payload_frame.get("live_frame") or {}).get("lifecycle") or {}
                emitted_event_sequence = tuple(
                    sorted(
                        (str(pid), int(sequence or 0))
                        for pid, sequence in (lifecycle.get("source_sequences") or {}).items()
                    )
                )
            except Exception as exc:
                payload = json.dumps({"error": str(exc)})
                emitted_event_sequence = last_event_sequence
            # A transition may land while the independent /slots watchdog is
            # building this payload. Acknowledge only the sequence actually
            # present in the emitted frame so that transition triggers the
            # next coalesced push instead of waiting for the watchdog tick.
            last_event_sequence = emitted_event_sequence
            last_emit_monotonic = time.monotonic()
            yield payload
        await asyncio.sleep(0.1)


async def _structured_tap_payloads():
    """Yield request-grouped structured tap JSON (mirrors /events/structured_tap)."""
    last_mtime = -1.0
    last_emit = 0.0
    while True:
        try:
            # Shard-aware: the base file is missing between rotation and the
            # next append; tracking it alone would stall change detection.
            mtime = _latest_tap_events_mtime() or 0.0
        except Exception:
            mtime = 0.0
        now_epoch = time.time()
        if mtime != last_mtime or (now_epoch - last_emit) >= 2.0:
            last_mtime = mtime
            last_emit = now_epoch
            # Shared helper: parses the live tail, recovers off-window lock
            # holders (so this stream stops flagging tapped autopilot-eval
            # traffic as "off-tap"), and enriches — same frame as the snapshot.
            enriched_requests = _structured_tap_requests_for_dashboard(
                max_requests=40,
                now_epoch=now_epoch,
            )
            yield json.dumps(
                {
                    "tap_active": _structured_tap_active(enriched_requests),
                    "tap_sentinel_active": _TAP_SENTINEL_PATH.exists(),
                    "structured_requests": enriched_requests,
                    "structured_tap_mtime": mtime or None,
                    "now": now_epoch,
                }
            )
        await asyncio.sleep(0.5)


async def _tail_file_payloads(
    path: Path,
    tail_bytes: int,
    *,
    create: bool = False,
):
    """Yield byte-tail JSON payloads ({chunk, initial}) for a growing text file.

    Shared body for the raw_tap / autopilot_log / planner_tap byte streams.
    Runs until cancelled (see _snapshot_payloads on disconnect handling).
    """
    fh = None
    try:
        if not path.exists():
            if create:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            else:
                yield json.dumps({"error": f"{path} does not exist"})
                return
        fh = open(path, "rb")
        fh.seek(0, 2)
        size = fh.tell()
        start = max(0, size - tail_bytes)
        fh.seek(start)
        initial = fh.read().decode("utf-8", errors="replace")
        yield json.dumps({"chunk": initial, "initial": True})
        while True:
            raw = fh.read(8192)
            if raw:
                yield json.dumps(
                    {
                        "chunk": raw.decode("utf-8", errors="replace"),
                        "initial": False,
                    }
                )
            else:
                await asyncio.sleep(0.1)
    except Exception as exc:
        yield json.dumps({"error": str(exc)})
    finally:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


@router.get("/dashboard/events/multiplex")
async def multiplex_stream(request: Request) -> StreamingResponse:
    """All always-on dashboard streams over a single SSE connection.

    Each source is emitted as ``event: <name>`` (snapshot, structured_tap,
    raw_tap, autopilot_log, planner_tap) so the browser dispatches by name via
    addEventListener. Collapses five persistent connections into one — see the
    section header above for the connection-limit rationale.
    """

    async def event_gen():
        producers = {
            "snapshot": _snapshot_payloads(),
            "structured_tap": _structured_tap_payloads(),
            "raw_tap": _tail_file_payloads(_INFERENCE_TAP_PATH, 8192),
            "autopilot_log": _tail_file_payloads(AUTOPILOT_LOG, 16384),
            "planner_tap": _tail_file_payloads(
                _PLANNER_TAP_PATH,
                16384,
                create=True,
            ),
        }
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        async def pump(name: str, gen):
            try:
                async for payload in gen:
                    await queue.put((name, payload))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                try:
                    await queue.put((name, json.dumps({"error": str(exc)})))
                except Exception:
                    pass

        tasks = [asyncio.create_task(pump(n, g)) for n, g in producers.items()]
        try:
            yield ": multiplex open\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    name, payload = await asyncio.wait_for(queue.get(), timeout=3.0)
                except asyncio.TimeoutError:
                    # Keep the single connection warm during idle windows.
                    yield ": heartbeat\n\n"
                    continue
                yield f"event: {name}\ndata: {payload}\n\n"
        finally:
            for t in tasks:
                t.cancel()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Task detail
# ---------------------------------------------------------------------------


_TASK_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _task_image_metadata(task_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe whether a task's input image can be rendered by the dashboard."""
    ref = _image_reference_for_task(events)
    metadata: dict[str, Any] = {
        "has_image": ref["has_image"],
        "source": ref["source"],
        "available": False,
        "url": None,
        "filename": "",
        "reason": "",
    }
    if not ref["has_image"]:
        return metadata
    if ref["source"] != "path" or not ref["path"]:
        metadata["reason"] = "Image bytes were not retained in task telemetry."
        return metadata

    from src.api.routes.path_validation import validate_api_path

    try:
        image_path = validate_api_path(ref["path"])
    except HTTPException:
        metadata["reason"] = "The recorded image path is outside the dashboard allowlist."
        return metadata
    metadata["filename"] = image_path.name
    if image_path.suffix.lower() not in _TASK_IMAGE_MEDIA_TYPES:
        metadata["reason"] = "The recorded image format is not browser-renderable."
        return metadata
    if not image_path.is_file():
        metadata["reason"] = "The recorded image file is no longer available."
        return metadata
    metadata["available"] = True
    metadata["url"] = f"/dashboard/api/task/{quote(task_id, safe='')}/image"
    return metadata


@router.get("/dashboard/api/task/{task_id}/image")
async def task_image(task_id: str) -> FileResponse:
    """Serve the path-backed input image recorded for a vision task."""
    events = _task_events(task_id, _todays_progress_log())
    ref = _image_reference_for_task(events)
    if ref["source"] != "path" or not ref["path"]:
        raise HTTPException(status_code=404, detail="Task has no path-backed image")

    from src.api.routes.path_validation import validate_api_path

    image_path = validate_api_path(ref["path"])
    media_type = _TASK_IMAGE_MEDIA_TYPES.get(image_path.suffix.lower())
    if media_type is None:
        raise HTTPException(status_code=415, detail="Unsupported dashboard image format")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Task image is no longer available")
    return FileResponse(
        image_path,
        media_type=media_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/dashboard/api/task/{task_id}.txt")
async def task_text(task_id: str) -> Any:
    """Return a plain-text snapshot of a task — for clipboard, curl, downstream LLMs.

    Mirrors the source-priority logic of /dashboard/api/task/{task_id}
    (live slot → tap_section → empty) so the .txt output stays
    consistent with what the dashboard panel displays.
    """
    log_path = _todays_progress_log()
    events = _task_events(task_id, log_path)
    structured_tap = _find_structured_request_by_id(task_id)
    if structured_tap is not None and task_id.startswith("tap_"):
        text = _task_text_snapshot(task_id, events, None, tap_section=structured_tap)
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(text)

    slots_by_port, _slots_meta = await _poll_all_slots()
    found_slot = None
    for port, slots in slots_by_port.items():
        for s in slots:
            if str(s.get("id_task")) == task_id:
                found_slot = s
                break
        if found_slot:
            break

    # Tap-section fallback when no live slot. Prefer the structured event
    # stream by task_id (deterministic mapping) over the plaintext substring
    # matcher (vulnerable to interleaved per-append writes producing
    # syntactically-valid but cross-contaminated records — see 2026-05-30
    # chat-83123001/chat-c7bf9580 incident).
    tap_section = None
    if found_slot is None:
        if not task_id.startswith("tap_"):
            tap_section = _find_structured_request_by_task_id(task_id)
        if tap_section is None:
            objective = _objective_for_task(events)
            producer_role = None
            for ev in reversed(events):
                if ev.get("event_type") == "task_completed":
                    producer_role = (ev.get("data") or {}).get("producer_role")
                    break
            if not producer_role:
                for ev in events:
                    if ev.get("event_type") == "routing_decision":
                        producer_role = (ev.get("data") or {}).get("chosen_action")
                        break
            tap_section = _find_section_by_objective(objective, expected_role=producer_role)

    text = _task_text_snapshot(task_id, events, found_slot, tap_section=tap_section)
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(text)


@router.get("/dashboard/api/task/{task_id}")
async def task_detail(task_id: str) -> JSONResponse:
    """Return all events for a task_id + tap section with the response text.

    Task text comes exclusively from the structured tap: v6 /slots carries no
    prompt/content, so the old match-a-live-slot-by-prompt-substring path
    (_find_slot_by_objective) could never match again and was removed. The
    slot fields are kept as nulls for response-shape compatibility.
    """
    log_path = _todays_progress_log()
    events = _task_events(task_id, log_path)
    structured_tap = _find_structured_request_by_id(task_id)
    if structured_tap is not None and task_id.startswith("tap_"):
        return JSONResponse(
            {
                "task_id": task_id,
                "objective": structured_tap.get("prompt") or "",
                "events": events,
                "active_slot_port": None,
                "active_slot_id": None,
                "slot": None,
                "tap_section": structured_tap,
                "terminal_answer": _terminal_answer_for_task(events),
                "image": _task_image_metadata(task_id, events),
            }
        )

    objective = _objective_for_task(events)

    # Prefer the structured event stream by task_id (deterministic mapping by
    # request metadata) over the plaintext substring matcher, which is
    # vulnerable to interleaved per-append writes (chat-83123001/chat-c7bf9580
    # cross-contamination, 2026-05-30). If the structured stream has nothing
    # for this task id, fall back to plaintext; producer_role from
    # task_completed constrains the role-filtered pass and also blocks the
    # unsafe global fallback in _find_section_by_objective.
    tap_section = None
    if not task_id.startswith("tap_"):
        tap_section = _find_structured_request_by_task_id(task_id)
    if tap_section is None:
        producer_role = None
        for ev in reversed(events):  # task_completed is usually near the end
            if ev.get("event_type") == "task_completed":
                producer_role = (ev.get("data") or {}).get("producer_role")
                break
        if not producer_role:
            for ev in events:
                if ev.get("event_type") == "routing_decision":
                    producer_role = (ev.get("data") or {}).get("chosen_action")
                    break
        tap_section = _find_section_by_objective(objective, expected_role=producer_role)

    return JSONResponse(
        {
            "task_id": task_id,
            "objective": objective,
            "events": events,
            "active_slot_port": None,
            "active_slot_id": None,
            "slot": None,
            "tap_section": tap_section,
            "terminal_answer": _terminal_answer_for_task(events),
            "image": _task_image_metadata(task_id, events),
        }
    )


@router.get("/dashboard/events/task/{task_id}")
async def task_stream(task_id: str, request: Request) -> StreamingResponse:
    """SSE stream of a task's live inference output, sourced from the structured
    tap event stream (inference_tap_events.jsonl).

    History: this used to poll llama-server ``/slots`` and correlate a slot to
    the task by substring-matching the objective against the slot's ``prompt``
    field, reading generated text from the slot's ``content`` field. As of the
    2026-06-26 v6 llama.cpp cutover, ``/slots`` no longer exposes per-slot
    ``prompt`` or ``content`` (upstream privacy change), so the matcher could
    neither locate the serving slot nor read any tokens — the live token feed
    went permanently silent. The structured tap already emits per-token
    ``chunk`` events keyed by ``task_id`` and is the concurrency-safe attribution
    source, so we tail it directly instead.

    The first content frame carries ``reset: true`` so the client discards
    whatever ``/dashboard/api/task`` rendered as the initial (response-so-far)
    body and lets this stream become the single source of truth for the
    streamed text — avoiding a double-render of the prefix. Tolerant of brief
    gaps before the task appears in the tap; gives up after `IDLE_GIVEUP` idle
    samples with no matching tap request.
    """

    async def event_gen():
        emitted = 0  # chars of response already streamed to the client
        first_content_frame = True
        log_path = _todays_progress_log()
        events = _task_events(task_id, log_path)
        # Quick exit: if the task already terminated, the initial GET already
        # rendered its full captured response — don't re-stream (would duplicate).
        terminal_seen = any(
            e.get("event_type") in ("task_completed", "task_failed", "escalation_triggered")
            for e in events
        )
        if terminal_seen:
            yield (
                "data: "
                + json.dumps({"delta": "", "done": True, "reason": "task_completed_already"})
                + "\n\n"
            )
            return

        idle_ticks = 0
        IDLE_GIVEUP = 60  # 60 ticks * 0.25s = 15s with no tap request before giving up
        # Live-tap popup rows key by `tap_<request_id>` (no orchestrator task_id).
        # Resolve those by request_id (reverse-grep recovers the full, growing
        # response from anywhere in the multi-GB tap); chat-* ids keep resolving
        # by their task_id field.
        is_tap = task_id.startswith("tap_")
        while True:
            if await request.is_disconnected():
                return
            req = (
                _find_structured_request_by_id(task_id)
                if is_tap
                else _find_structured_request_by_task_id(task_id)
            )
            if req is not None:
                idle_ticks = 0
                response = str(req.get("response") or "")
                if len(response) > emitted:
                    delta = response[emitted:]
                    emitted = len(response)
                    payload = json.dumps(
                        {
                            "delta": delta,
                            "content_len": len(response),
                            "tokens_decoded": (req.get("timings_raw") or {}).get("tokens"),
                            "matched_port": req.get("port"),
                            "reset": first_content_frame,
                            "done": False,
                        }
                    )
                    first_content_frame = False
                    yield f"data: {payload}\n\n"
                # A completed structured request (end/timings event) is authoritative.
                if req.get("status") == "complete":
                    yield (
                        "data: "
                        + json.dumps({"delta": "", "done": True, "reason": "tap_complete"})
                        + "\n\n"
                    )
                    return
            else:
                idle_ticks += 1
                # Re-check terminal state every few ticks
                if idle_ticks % 20 == 0:
                    fresh_events = _task_events(task_id, log_path)
                    if any(
                        e.get("event_type")
                        in ("task_completed", "task_failed", "escalation_triggered")
                        for e in fresh_events
                    ):
                        yield (
                            "data: "
                            + json.dumps({"delta": "", "done": True, "reason": "task_completed"})
                            + "\n\n"
                        )
                        return
                if idle_ticks >= IDLE_GIVEUP:
                    yield (
                        "data: "
                        + json.dumps({"delta": "", "done": True, "reason": "idle_timeout"})
                        + "\n\n"
                    )
                    return
                # Heartbeat so the client knows we're still searching
                if idle_ticks % 5 == 0:
                    yield (
                        "data: "
                        + json.dumps({"delta": "", "searching": True, "idle_ticks": idle_ticks})
                        + "\n\n"
                    )
            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# GEPA progress
# ---------------------------------------------------------------------------


@router.get("/dashboard/api/gepa")
async def gepa_status() -> JSONResponse:
    """Recent GEPA progress lines + parsed trial state + sentinel completion count."""
    # Early exits still carry the freshness envelope: a missing/unreadable
    # autopilot log is exactly the dead-producer case the envelope must report
    # (an unstamped body hides it from /api/health consumers).
    if not AUTOPILOT_LOG.exists():
        return JSONResponse(_stamp({"active": False, "lines": [], "state": {}}, "gepa"))
    try:
        size = AUTOPILOT_LOG.stat().st_size
        with open(AUTOPILOT_LOG, "rb") as f:
            if size > 256 * 1024:
                f.seek(-256 * 1024, 2)
            tail = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return JSONResponse(_stamp({"active": False, "lines": [], "state": {}}, "gepa"))

    lines = tail.splitlines()
    gepa_lines = [
        line
        for line in lines
        if "gepa" in line.lower()
        or "Trial" in line
        or "sentinel" in line.lower()
        or "Dispatching action" in line
        or "prompt_forge" in line
    ][-30:]
    trial_state = _parse_trial_state(tail)

    # Sentinel completion progress: count chat-XXX task_completed events since
    # the last "evaluating baseline" line in the autopilot log.
    sentinels_done = 0
    trial_start_ts: float | None = None
    for line in lines:
        if "GEPA: evaluating baseline" in line:
            # Parse the timestamp from the line (format: 2026-05-21 17:24:44,...)
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m:
                try:
                    trial_start_ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                except Exception:
                    pass
    if trial_start_ts is not None and trial_state.get("baseline_score") is None:
        # We're mid-baseline: count completed chat tasks since trial_start_ts
        log_path = _todays_progress_log()
        if log_path.exists():
            with open(log_path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("event_type") != "task_completed":
                        continue
                    tid = e.get("task_id")
                    if not tid or not tid.startswith("chat-"):
                        continue
                    try:
                        ts = datetime.fromisoformat(
                            e["timestamp"].replace("Z", "+00:00")
                        ).timestamp()
                    except Exception:
                        continue
                    # autopilot log times are local (CEST); JSONL is UTC. Add
                    # a generous timezone tolerance (7200s = 2h).
                    if ts >= trial_start_ts - 7200:
                        sentinels_done += 1
    trial_state["sentinels_completed"] = sentinels_done

    # Last 10 trials from autopilot_journal for trajectory. Read ALL journal
    # shards (base + rotations) via the shard-aware helper — a raw byte-tail on
    # the base file alone silently freezes at the last trial before a rotation
    # (e.g. trial 999 in autopilot_journal.jsonl while the live run advances in
    # autopilot_journal_1.jsonl). See _autopilot_journal_shards() for the full
    # rationale; this is the same stale-panel bug the frontier/trial panels
    # already fixed by routing through _read_autopilot_journal_rows().
    recent_trials: list[dict[str, Any]] = []
    journal_rows = _read_autopilot_journal_rows()
    if journal_rows:
        try:
            # Include killed/bug-corrupted placeholders here (tagged, not dropped).
            # They are still excluded from every FRONTIER/HV computation upstream,
            # but silently filtering them from the trajectory list made the panel
            # look frozen after a mid-trial kill/restart — the operator's newest
            # trial simply vanished with no signal it had died. The client greys +
            # tags these rows so recent activity is always visible.
            for j in _effective_journal_trial_rows(journal_rows)[-15:]:
                eval_details = (
                    j.get("eval_details") if isinstance(j.get("eval_details"), dict) else {}
                )
                learning_exclusion = (
                    eval_details.get("learning_exclusion") if isinstance(eval_details, dict) else {}
                )
                if not isinstance(learning_exclusion, dict):
                    learning_exclusion = {}
                learning_excluded_by = str(learning_exclusion.get("by") or "").strip()
                keep_revert_decision = str(j.get("keep_revert_decision") or "").strip()
                bug_corrupted_by = str(j.get("bug_corrupted_by") or "").strip()
                recent_trials.append(
                    {
                        "trial_id": j.get("trial_id"),
                        "timestamp": j.get("timestamp", ""),
                        "species": j.get("species"),
                        # Tier is REQUIRED context here: quality is scored per-tier and
                        # is NOT comparable across tiers (T3 expert/hard rows sit well
                        # below T1 by design), so a tier-less trajectory row reads a
                        # healthy T3 eval as a quality regression. Default to the
                        # canonical tier when absent (legacy rows predate the field).
                        "tier": j.get("tier", DEFAULT_FRONTIER_TIER),
                        "decision_question_mix": _decision_question_mix_from_row(j),
                        "quality": j.get("quality"),
                        "speed": j.get("speed"),
                        "cost": j.get("cost"),
                        "reliability": j.get("reliability"),
                        "pareto_status": j.get("pareto_status"),
                        "real_suite_v1": _suite_metric_for_dashboard(j, "real_suite_v1"),
                        # Non-empty when the trial was killed mid-flight or otherwise
                        # quarantined; the client renders these muted + tagged.
                        "bug_corrupted_by": bug_corrupted_by or None,
                        "quarantine_label": (
                            "killed" if "killed" in bug_corrupted_by else "corrupted"
                        )
                        if bug_corrupted_by
                        else None,
                        "learning_excluded_by": learning_excluded_by or None,
                        "keep_revert_decision": keep_revert_decision or None,
                        "exclusion_label": (
                            "seq-refuted"
                            if learning_excluded_by == "seq_refuted"
                            else (
                                "excluded"
                                if learning_excluded_by or keep_revert_decision == "excluded"
                                else None
                            )
                        ),
                        "description": (j.get("config_snapshot", {}).get("description") or "")[
                            :140
                        ],
                    }
                )
        except Exception:
            pass

    # GEPA optimizer-provenance windows (display-only honesty labels, 2026-07-26).
    # Canonical constant lives in scripts/autopilot/optimization_brief.py (cites
    # the 2026-07-25 GEPA no-op incident); fail-open to [] so a scripts-path
    # import problem can never break the panel.
    try:
        from scripts.autopilot.optimization_brief import GEPA_PROVENANCE_WINDOWS

        all_provenance_windows = list(GEPA_PROVENANCE_WINDOWS)
    except Exception:
        all_provenance_windows = []

    # Only ship caveats that apply to at least one row actually rendered in
    # ``recent_trials``. The historical window remains canonical and will
    # reappear whenever an affected row is in view; showing it over an entirely
    # post-fix trajectory falsely claimed that "trials below" were hatched.
    recent_trial_timestamps = [
        ts
        for trial in recent_trials
        if (ts := _parse_journal_ts(trial.get("timestamp"))) is not None
    ]
    provenance_windows = [
        window
        for window in all_provenance_windows
        if any(
            (window.get("from_ts") is None or ts >= float(window["from_ts"]))
            and (window.get("until_ts") is None or ts < float(window["until_ts"]))
            for ts in recent_trial_timestamps
        )
    ]

    return JSONResponse(
        _stamp(
            {
                "active": bool(gepa_lines),
                "lines": gepa_lines,
                "state": trial_state,
                "recent_trials": recent_trials,
                "current_decision_question_mix": next(
                    (
                        trial["decision_question_mix"]
                        for trial in reversed(recent_trials)
                        if trial.get("decision_question_mix") is not None
                    ),
                    None,
                ),
                # Trials timestamped inside any window carry broken optimizer
                # provenance (the reflective-mutation path was a no-op); the
                # client hatches/greys those rows + shows a caveat badge.
                "provenance_windows": provenance_windows,
            },
            "gepa",
        )
    )


# ---------------------------------------------------------------------------
# Single-page HTML (extracted to dashboard.html; re-read on each request).
#
# The HTML is read fresh from disk on every /dashboard hit instead of cached
# at module load, so edits to dashboard.html take effect after a browser
# reload — no orchestrator API restart needed. Cost: one ~43 KB file read
# per page hit (Linux page cache makes repeat reads essentially free).
# Use this path for HTML/CSS/JS hotfixes; Python route handlers still
# require `orchestrator_stack reload orchestrator`.
# ---------------------------------------------------------------------------

_DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"


def _read_dashboard_html() -> str:
    """Read the dashboard HTML file fresh from disk."""
    return _DASHBOARD_HTML_PATH.read_text()


# Backwards-compat: tests + external readers historically check
# `_DASHBOARD_HTML` as a module attribute. Module-level __getattr__ resolves
# it lazily on each access by re-reading the file.
def __getattr__(name: str) -> str:
    if name == "_DASHBOARD_HTML":
        return _read_dashboard_html()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@router.get("/dashboard")
async def dashboard_page() -> HTMLResponse:
    """Serve the single-page dashboard (HTML re-read from disk per request)."""
    return HTMLResponse(
        _read_dashboard_html(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )
