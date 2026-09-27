"""Snapshot scanners extracted from dashboard.py (2026-05-21 refactor).

Mines the orchestrator's progress JSONL log for routing decisions, in-flight /
recently-completed chat tasks, and pattern counts. All scanners are pure
file-reads; the dashboard route layer composes them into a single snapshot
JSON response.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.api.routes.dashboard_topology import base_role

logger = logging.getLogger(__name__)

# Per-role in-flight age ceilings. A started chat task is treated as in-flight
# only until it is this old without a terminal event; older ones are assumed to
# be restart-orphans (the API restarted and the task_completed/_failed line was
# never written). Slow roles legitimately hold a slot far longer than the
# default for a single generation, so they get a wider ceiling — otherwise a
# live ingest/architect task is misclassified as an orphan and vanishes from the
# topology panel while it is still generating. Live-slot gating in the snapshot
# route is the authoritative guard against surfacing *stale* orphans, so these
# ceilings can be generous.
INFLIGHT_MAX_AGE_DEFAULT_S = 300.0
INFLIGHT_MAX_AGE_BY_ROLE_S: dict[str, float] = {
    "ingest_long_context": 1800.0,   # long-context ingest, ~0.5 t/s
    "architect_general": 900.0,      # long reasoning, ~0.4 t/s (full-CPU Flash-Next since 2026-09-27)
    # 2026-09-27 ARCHITECT SWAP: the :8083 27B is architect_critic and hosts
    # coder_escalation + ingest_long_context; same ceiling as its coder alias.
    "architect_critic": 900.0,
    "coder_escalation": 900.0,
}


def todays_progress_log(progress_log_dir: Path) -> Path:
    """Today's progress JSONL path under `progress_log_dir`."""
    return progress_log_dir / f"{date.today().isoformat()}.jsonl"


def scan_recent_decisions(
    path: Path, window_s: float = 600.0, max_items: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Tail today's progress JSONL, return recent decisions + counters.

    Returns (recent_list, source_counts_rolling, source_counts_cumulative).

    source_counts_rolling additionally carries non-counter payloads under
    underscore-prefixed keys (filtered out of percentage math by the frontend):
        _verifier_verdicts   — {accept: n, reject: n} for in-window decisions
        _today_roles         — {role: count} across the FULL day's decisions
                               (not the window) so the panel can show routing
                               diversity even when the window is dominated by
                               chat traffic to one role
        _classifier_conf     — {high: n, medium: n, low: n} for in-window
                               classifier-source decisions only. The denominator
                               is `source_rolling.classifier`, NOT total. Learned
                               and rules sources structurally never carry a
                               classifier_confidence; the old "100% no-confidence"
                               line was true but misleading — it just meant the
                               window happened to be all learned.
    """
    recent: deque = deque(maxlen=max_items)
    source_rolling: dict[str, int] = {}
    source_cumulative: dict[str, int] = {}
    verifier_verdicts: dict[str, int] = {}
    today_roles: dict[str, int] = {}
    classifier_conf: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    now = time.time()
    if not path.exists():
        return list(recent), source_rolling, source_cumulative
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event_type") != "routing_decision":
                    continue
                d = e.get("data", {})
                src = d.get("decision_source") or d.get("strategy") or "?"
                source_cumulative[src] = source_cumulative.get(src, 0) + 1
                # Cumulative role mix across the whole day — the panel uses this
                # to show diversity even when the window is one-role-dominated.
                routing = d.get("routing")
                if isinstance(routing, list) and routing:
                    role0 = base_role(str(routing[0])) or str(routing[0])
                    today_roles[role0] = today_roles.get(role0, 0) + 1
                ts_str = e.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = now
                age = now - ts
                if age <= window_s:
                    source_rolling[src] = source_rolling.get(src, 0) + 1
                    if d.get("verifier_verdict"):
                        v = d["verifier_verdict"]
                        verifier_verdicts[v] = verifier_verdicts.get(v, 0) + 1
                    # Classifier-confidence histogram: ONLY meaningful when
                    # decision_source=classifier. Other sources have no
                    # confidence field by design.
                    if src == "classifier":
                        c = d.get("classifier_confidence")
                        if isinstance(c, (int, float)):
                            if c >= 0.8:
                                classifier_conf["high"] += 1
                            elif c >= 0.5:
                                classifier_conf["medium"] += 1
                            else:
                                classifier_conf["low"] += 1
                # Build a compact summary
                recent.append({
                    "task_id": e.get("task_id"),
                    "ts": ts_str,
                    "age_s": round(age, 1),
                    "source": src,
                    "chosen_action": d.get("chosen_action") or "",
                    "classifier_confidence": d.get("classifier_confidence"),
                    "verifier_p_success": d.get("verifier_p_success"),
                    "verifier_verdict": d.get("verifier_verdict"),
                    "verifier_shadow": d.get("verifier_shadow"),
                })
    except Exception as exc:
        logger.debug("scan_recent_decisions failed: %s", exc)
    source_rolling["_verifier_verdicts"] = verifier_verdicts  # type: ignore[assignment]
    source_rolling["_today_roles"] = today_roles               # type: ignore[assignment]
    source_rolling["_classifier_conf"] = classifier_conf       # type: ignore[assignment]
    return list(recent), source_rolling, source_cumulative


def scan_orchestrator_tasks(
    path: Path,
    in_flight_max_age_s: float = INFLIGHT_MAX_AGE_DEFAULT_S,
    completed_window_s: float = 600.0,
    max_items: int = 40,
    role_max_age_overrides: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan today's progress JSONL for in-flight + recently-completed chat tasks.

    Uses the orchestrator's `chat-XXX` task_ids (not llama-server's internal
    numeric id_task). Returns (in_flight, recent_completed). Every task carries a
    canonical ``role`` field (base-normalised `chosen_action` for in-flight,
    `final_role` for completed) so all dashboard surfaces group by one key.

    NB: we DO NOT time-filter routing_decision events during the scan — we
    need them merged into every started/completed task even if the routing
    happened minutes before our window. Tasks that started older than their
    role's in-flight ceiling (``role_max_age_overrides`` falling back to
    ``in_flight_max_age_s``) and never completed are treated as orphans
    (typically killed by an API restart) and excluded from in-flight.
    """
    overrides = INFLIGHT_MAX_AGE_BY_ROLE_S if role_max_age_overrides is None else role_max_age_overrides
    if not path.exists():
        return [], []
    now = time.time()
    started: dict[str, dict[str, Any]] = {}
    terminal_events: dict[str, dict[str, Any]] = {}
    routing_meta: dict[str, dict[str, Any]] = {}
    plan_reviews: dict[str, dict[str, Any]] = {}
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = e.get("task_id")
                if not tid or not isinstance(tid, str) or not tid.startswith("chat-"):
                    continue
                ev = e.get("event_type")
                ts_str = e.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
                data = e.get("data", {})
                if ev == "task_started":
                    started[tid] = {
                        "task_id": tid,
                        "started_at": ts,
                        "age_s": now - ts,
                        "objective": (data.get("objective", "") or "")[:200],
                        "task_type": data.get("task_type"),
                        "priority": data.get("priority"),
                    }
                elif ev == "routing_decision":
                    routing_meta[tid] = {
                        "chosen_action": data.get("chosen_action") or "",
                        "decision_source": data.get("decision_source") or "",
                        "classifier_confidence": data.get("classifier_confidence"),
                        "verifier_p_success": data.get("verifier_p_success"),
                        "verifier_verdict": data.get("verifier_verdict"),
                        "difficulty_band": data.get("difficulty_band"),
                        # Vision reference (see routing_decision.routing_meta):
                        # carried onto BOTH in-flight and completed cards
                        # because routing_meta is merged into every task below
                        # before the terminal-event branch. `image_path` is a
                        # host path, not a URL — a renderer must resolve it
                        # through a serving route, never emit it as an <img src>.
                        "image_path": data.get("image_path") or "",
                        "has_image": bool(data.get("has_image")),
                        "image_source": data.get("image_source") or "",
                    }
                elif ev == "plan_reviewed":
                    # A terminal failure following a plan review is materially
                    # different from an inference/server failure. Preserve the
                    # architect's decision on the outcome card.
                    plan_reviews[tid] = {
                        "plan_review_decision": data.get("decision") or "",
                        "plan_review_feedback": (data.get("feedback") or "")[:200],
                        "plan_review_role": e.get("agent_role") or "architect_general",
                    }
                elif ev in ("task_completed", "task_failed", "escalation_triggered"):
                    terminal_events[tid] = {
                        "event_type": ev,
                        "ended_at": ts,
                        "age_s": now - ts,
                        "final_role": data.get("final_answer_role") or data.get("producer_role"),
                        # Decode telemetry (chat_pipeline llm_completion_meta) —
                        # carried through so the dashboard can show tok/s per
                        # completed task instead of only wall-clock duration.
                        "tokens_generated": data.get("tokens_generated"),
                        "generation_ms": data.get("generation_ms"),
                        "prompt_eval_ms": data.get("prompt_eval_ms"),
                    }
    except Exception as exc:
        logger.debug("scan_orchestrator_tasks failed: %s", exc)

    in_flight: list[dict[str, Any]] = []
    recent_completed: list[dict[str, Any]] = []
    for tid, s in started.items():
        s.update(routing_meta.get(tid, {}))
        if tid not in terminal_events:
            # Canonical grouping key (Fix 2): base-normalised routed role.
            role = base_role(s.get("chosen_action") or "")
            s["role"] = role or (s.get("chosen_action") or "")
            cutoff = overrides.get(role, in_flight_max_age_s)
            if s["age_s"] <= cutoff:
                in_flight.append(s)
        else:
            t = terminal_events[tid]
            if t["age_s"] > completed_window_s:
                continue
            s["ended_at"] = t["ended_at"]
            s["end_age_s"] = t["age_s"]
            s["outcome"] = t["event_type"]
            s["duration_s"] = round(t["ended_at"] - s["started_at"], 2)
            s["final_role"] = t.get("final_role")
            s.update(plan_reviews.get(tid, {}))
            # Decode t/s = generated tokens / generation time. Guard missing
            # fields (e.g. failure completions with no completion_meta) so the
            # `tps` key is simply absent — downstream must null-check.
            tok = t.get("tokens_generated")
            gen_ms = t.get("generation_ms")
            if isinstance(tok, (int, float)) and isinstance(gen_ms, (int, float)) and tok > 0 and gen_ms > 0:
                s["tokens_generated"] = int(tok)
                s["generation_ms"] = float(gen_ms)
                s["prompt_eval_ms"] = t.get("prompt_eval_ms")
                s["tps"] = round(tok / (gen_ms / 1000.0), 1)
            # Completed tasks group by producer role, falling back to the route.
            s["role"] = base_role(s.get("final_role") or s.get("chosen_action") or "")
            recent_completed.append(s)
    in_flight.sort(key=lambda x: x["age_s"])
    recent_completed.sort(key=lambda x: x["end_age_s"])
    return in_flight[:max_items], recent_completed[:max_items]


def count_log_events(
    path: Path, patterns: dict[str, str], window_s: float = 600.0,
) -> dict[str, int]:
    """Tail the log at `path` and count occurrences of `patterns` regexes."""
    counts = {key: 0 for key in patterns}
    if not path.exists():
        return counts
    # Try a recent-tail-only optimization: read last 256KB if file is big
    try:
        size = path.stat().st_size
        if size > 256 * 1024:
            with open(path, "rb") as f:
                f.seek(-256 * 1024, 2)
                tail = f.read().decode("utf-8", errors="ignore")
        else:
            with open(path) as f:
                tail = f.read()
    except Exception:
        return counts
    compiled = {k: re.compile(v) for k, v in patterns.items()}
    for line in tail.splitlines():
        for key, regex in compiled.items():
            if regex.search(line):
                counts[key] += 1
    return counts
