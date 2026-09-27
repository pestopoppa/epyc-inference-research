"""Slot management, HTTP calls to the orchestrator, and tool telemetry.

Handles llama-server slot erasure, busy-port detection, progress polling,
and the core ``call_orchestrator_forced`` function.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from seeding_types import (
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_TIMEOUT,
    HEAVY_PORTS,
    PROJECT_ROOT,
    STACK_SCRIPT,
)
from seeding_infra import _wait_for_heavy_models_idle
from seeding_scoring import _inband_error_text

__all__ = [
    "_SLOT_ERASE_CAPABILITY",
    "_busy_heavy_ports",
    "_call_orchestrator_with_slot_poll",
    "_erase_slots",
    "_force_erase_and_verify",
    "_normalize_tool_telemetry",
    "_read_slot_progress",
    "_recover_heavy_ports_if_stuck",
    "_surface_inband_error",
    "call_orchestrator_forced",
]

logger = logging.getLogger("seed_specialist_routing")

if TYPE_CHECKING:
    import httpx


# Per-port slot erase strategy cache.
# None = unknown, str = preferred strategy, False = unsupported on this build.
_SLOT_ERASE_CAPABILITY: dict[int, str | None | bool] = {}


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a non-negative float from the environment."""
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# ── Eval reconnect backoff (REL-2) ───────────────────────────────────
# 2026-07 incidents: two eval runs burned ~680 and ~532 questions when the
# orchestrator API was reloaded mid-run — every in-flight /chat POST hit a
# CONNECTION-level failure (connection refused / reset) that was recorded as a
# terminal per-question error. `call_orchestrator_forced` now retries CONNECTION
# failures ONLY (never timeouts, never HTTP status errors, never in-band error
# banners) with exponential backoff before honestly error-rowing, so a transient
# API restart is survived while persistent unreachability stays an excluded
# error (REL-1). Timeout semantics (04411baf) are untouched: timeouts are not
# in the reconnect set and fall straight through to their prior terminal path.

# Reasons (from src.observability.classify_exception) that count as a
# connection-level failure worth a reconnect backoff. Deliberately EXCLUDES
# every timeout reason (connect_timeout / read_timeout / timeout) and
# http_status. `request_error` is the classifier's bucket for non-timeout
# transport faults — httpx ReadError / WriteError / RemoteProtocolError, i.e.
# a peer reset or "server disconnected" mid-reload — which are exactly the
# connection-reset flavors we want to survive.
_RECONNECT_REASONS = frozenset({"connect_error", "request_error"})
_TRANSPORT_TIMEOUT_REASONS = frozenset(
    {"connect_timeout", "read_timeout", "write_timeout", "pool_timeout", "socket_timeout", "timeout"}
)
_FAILURE_PROVENANCE_SCHEMA = "epyc.failure_provenance.v1"


def _error_text(exc: BaseException, detail: str) -> str:
    """Return a NON-EMPTY error string for ``exc``.

    ``str(httpx.ReadTimeout(""))`` is the empty string, and an empty ``error``
    is FALSY: every downstream `if not error:` guard then treats a transport
    timeout as a successful generation and scores its empty answer against the
    gold answer, manufacturing a WRONG verdict from a measurement that was
    never made. Preserve the exception's own message whenever it has one (so
    existing error-text assertions are unchanged) and fall back to the
    classifier's `Type: msg` detail — which is never empty — when it does not.
    """
    text = str(exc)
    return text if text.strip() else detail


def _classify_exc(exc: BaseException) -> tuple[str, str]:
    """Return ``(reason, detail)`` for ``exc`` via the shared observability
    classifier, with a self-contained fallback so this module never hard-depends
    on ``src`` being importable. Mirrors the reason strings resilient_post stamps
    into its meta dict so both request paths agree on what is reconnectable."""
    # E8 retains the timeout subclass: a PoolTimeout is client contention, not
    # proof that the server exhausted its generation budget.
    try:
        import httpx
        if isinstance(exc, httpx.PoolTimeout):
            return ("pool_timeout", f"{type(exc).__name__}: {exc}")
        if isinstance(exc, httpx.ConnectTimeout):
            return ("connect_timeout", f"{type(exc).__name__}: {exc}")
        if isinstance(exc, httpx.ReadTimeout):
            return ("read_timeout", f"{type(exc).__name__}: {exc}")
        if isinstance(exc, httpx.WriteTimeout):
            return ("write_timeout", f"{type(exc).__name__}: {exc}")
        if isinstance(exc, httpx.TimeoutException):
            return ("timeout", f"{type(exc).__name__}: {exc}")
    except Exception:
        pass
    if isinstance(exc, socket.timeout):
        return ("socket_timeout", f"{type(exc).__name__}: {exc}")
    try:
        from pathlib import Path

        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from src.observability import classify_exception  # type: ignore

        return classify_exception(exc)
    except Exception:
        return ("unexpected_error", f"{type(exc).__name__}: {exc}")


def _client_transport_timeout_provenance(
    *,
    reason: str,
    exc: BaseException,
    role: str,
    workload_class: str | None,
    max_queue_wait_ms: int,
) -> dict[str, Any]:
    """Describe an observed client timeout without asserting server state.

    A client-side timeout cannot prove whether the server began generation or
    how much work it completed. Those server-side fields are omitted entirely,
    which makes this class structurally ineligible for admission-time retries.
    """
    if reason not in _TRANSPORT_TIMEOUT_REASONS:
        raise ValueError(f"not a transport-timeout reason: {reason!r}")
    return {
        "schema": _FAILURE_PROVENANCE_SCHEMA,
        "class": "client_transport_timeout",
        "code": reason,
        "phase": "client_transport",
        "exception_class": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "exception_reason": reason,
        "role": str(role or ""),
        "workload_class": str(workload_class or ""),
        "max_queue_wait_ms": int(max_queue_wait_ms),
    }


def _slot_erase_timeout_provenance(*, elapsed_s: float, server_generation_observed: bool, exc: BaseException | None, role: str, workload_class: str | None, max_queue_wait_ms: int) -> dict[str, Any]:
    """Record a watchdog timeout without claiming a reviewed server timeout."""
    exception_class = (
        f"{type(exc).__module__}.{type(exc).__qualname__}"
        if exc is not None else "concurrent.futures.TimeoutError"
    )
    return {
        "schema": _FAILURE_PROVENANCE_SCHEMA,
        "class": "slot_erase_timeout",
        "code": "timeout_after_slot_erase",
        "phase": "execution_watchdog",
        "exception_class": exception_class,
        "exception_reason": "future_timeout" if exc is None else _classify_exc(exc)[0],
        "elapsed_ms": int(round(max(0.0, elapsed_s) * 1000)),
        "server_generation_observed": bool(server_generation_observed),
        "role": str(role or ""),
        "workload_class": str(workload_class or ""),
        "max_queue_wait_ms": int(max_queue_wait_ms),
    }


# ── Slot management ──────────────────────────────────────────────────


def _erase_slots(port: int, *, all_slots: bool = False) -> None:
    """Force-cancel in-progress inference on a llama-server port.

    After a timeout the server may still be grinding on the old request.
    Erasing slots prevents cascading timeouts on subsequent requests.

    Args:
        port: llama-server port.
        all_slots: If True, erase ALL slots (including idle ones) to clear
            stale KV cache and prevent cross-request context contamination.
            Use this between independent eval questions.
    """
    import httpx

    def _erase_slot_with_strategy(slot_id: int, strategy: str) -> int | None:
        if strategy == "POST_QUERY":
            resp = httpx.post(
                f"http://localhost:{port}/slots/{slot_id}?action=erase",
                timeout=8,
            )
        elif strategy == "GET_QUERY":
            resp = httpx.get(
                f"http://localhost:{port}/slots/{slot_id}?action=erase",
                timeout=8,
            )
        elif strategy == "POST_JSON":
            resp = httpx.post(
                f"http://localhost:{port}/slots/{slot_id}",
                json={"action": "erase"},
                timeout=8,
            )
        else:
            return None
        return resp.status_code

    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=5)
        if resp.status_code != 200:
            return
        erased_slots: list[int] = []
        for slot in resp.json():
            # When all_slots=True, erase every slot (idle or processing)
            # to clear stale KV cache between eval questions.
            # When all_slots=False (default), only erase processing slots.
            if not all_slots and not slot.get("is_processing"):
                continue
            slot_id = slot.get("id", 0)
            cap = _SLOT_ERASE_CAPABILITY.get(port)
            if cap is False:
                continue
            strategies: list[str]
            if isinstance(cap, str):
                strategies = [cap]
            else:
                strategies = ["POST_QUERY", "GET_QUERY", "POST_JSON"]

            unsupported_codes = {404, 405, 501}
            saw_transient = False
            erased = False
            for strategy in strategies:
                try:
                    status = _erase_slot_with_strategy(slot_id, strategy)
                except Exception:
                    saw_transient = True
                    continue

                if status == 200:
                    _SLOT_ERASE_CAPABILITY[port] = strategy
                    erased_slots.append(slot_id)
                    erased = True
                    break
                if status not in unsupported_codes:
                    saw_transient = True

            if erased:
                continue

            if isinstance(cap, str):
                # Cached strategy failed; reset to unknown so we can probe again.
                _SLOT_ERASE_CAPABILITY[port] = None
            elif not saw_transient:
                _SLOT_ERASE_CAPABILITY[port] = False
                logger.warning(
                    f"  slot erase unsupported on port {port}; disabling erase attempts"
                )
        if erased_slots:
            ids = ", ".join(str(s) for s in erased_slots)
            logger.info(f"  → erased {len(erased_slots)} slot(s) on port {port} [{ids}]")
    except Exception as e:
        logger.warning("  [erase-slots] port %d: %s", port, e)


def _force_erase_and_verify(
    port: int, max_attempts: int = 3, verify_delay: float = 1.5,
) -> bool:
    """Aggressively erase slots and verify they stopped.

    Unlike ``_erase_slots`` this resets the capability cache so we never
    skip a port due to stale ``False`` entries, and it retries with
    verification polling between attempts.

    Returns True if the port is idle after cleanup.
    """
    import httpx

    if port <= 0:
        return True
    _SLOT_ERASE_CAPABILITY.pop(port, None)

    for attempt in range(1, max_attempts + 1):
        _erase_slots(port)
        time.sleep(verify_delay)
        try:
            resp = httpx.get(f"http://localhost:{port}/slots", timeout=5)
            if resp.status_code == 200:
                slots = resp.json()
                if not any(s.get("is_processing", False) for s in slots):
                    logger.info(
                        "  [force-erase] port %d idle after attempt %d", port, attempt,
                    )
                    return True
        except Exception:
            pass
        logger.warning(
            "  [force-erase] port %d still busy after attempt %d/%d",
            port, attempt, max_attempts,
        )
    logger.warning("  [force-erase] port %d stuck after %d attempts", port, max_attempts)
    return False


def _busy_heavy_ports(timeout_s: float = 2.0) -> list[int]:
    """Return heavy-model ports that currently report is_processing=True."""
    import httpx

    busy: list[int] = []
    for port in sorted(HEAVY_PORTS):
        try:
            resp = httpx.get(f"http://localhost:{port}/slots", timeout=timeout_s)
            if resp.status_code != 200:
                continue
            slots = resp.json()
            if any(bool(s.get("is_processing", False)) for s in slots):
                busy.append(port)
        except Exception:
            continue
    return busy


def _read_slot_progress(port: int, timeout_s: float = 1.0) -> dict[str, Any] | None:
    """Read lightweight progress counters from llama-server /slots."""
    import httpx

    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=timeout_s)
        if resp.status_code != 200:
            return None
        slots = resp.json()
        if not isinstance(slots, list) or not slots:
            return None

        # Prefer actively processing slot for live progress.
        slot = None
        for s in slots:
            if bool(s.get("is_processing", False)):
                slot = s
                break
        if slot is None:
            slot = slots[0]

        nt = {}
        next_tokens = slot.get("next_token")
        if isinstance(next_tokens, list) and next_tokens:
            nt = next_tokens[0] or {}

        decoded_raw = nt.get("n_decoded", 0)
        remain_raw = nt.get("n_remain", 0)
        task_raw = slot.get("id_task", 0)
        try:
            decoded = int(decoded_raw or 0)
        except Exception:
            decoded = 0
        try:
            remain = int(remain_raw or 0)
        except Exception:
            remain = 0
        try:
            task_id = int(task_raw or 0)
        except Exception:
            task_id = 0

        return {
            "is_processing": bool(slot.get("is_processing", False)),
            "task_id": task_id,
            "n_decoded": max(0, decoded),
            "n_remain": remain,
        }
    except Exception:
        return None


# ── Orchestrator HTTP calls ──────────────────────────────────────────


def _call_orchestrator_with_slot_poll(
    *,
    prompt: str,
    force_role: str,
    force_mode: str,
    url: str,
    timeout: int,
    image_path: str,
    cache_prompt: bool | None,
    client: "httpx.Client | None",
    allow_delegation: bool | None,
    log_label: str,
    poll_port: int,
    session_id: str = "",
    scoring_method: str = "",
    stop_sequences: list[str] | None = None,
    request_priority: str | None = None,
    workload_class: str | None = None,
    batch_id: int | str | None = None,
    batch_placement_mode: str | None = None,
    watcher: Any | None = None,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Call orchestrator while polling slot progress for live visibility.

    Optional ``watcher`` (OrchestratorWatcher) — when supplied, exogenous
    reloads of the orchestrator API or the target llama-server are detected
    and the request is retried after waiting for /health. Backward-compatible:
    watcher=None preserves the legacy direct-post-with-exception-swallow
    behavior.
    """

    progress: dict[str, Any] = {
        "max_decoded": 0,
        "last_decoded": 0,
        "last_remain": 0,
        "task_id": 0,
        "source": "",
    }
    t0 = time.perf_counter()
    log_every_s = 5.0
    log_delta_tokens = 128
    last_log_at = t0
    last_logged_decoded = 0
    heartbeat_interval = 120.0
    last_heartbeat = t0
    # Hardening for orphaned/stalled llama-server streams: the coarse
    # request timeout is intentionally large for architect roles, but if the
    # slot counters stop moving for minutes the current request is not making
    # useful progress. Defaults are conservative and can be disabled with 0.
    slot_stall_watchdog_s = _env_float("SEEDING_SLOT_STALL_WATCHDOG_S", 150.0)
    slot_idle_orphan_s = _env_float("SEEDING_SLOT_IDLE_ORPHAN_WATCHDOG_S", 30.0)
    slot_idle_completion_grace_s = _env_float(
        "SEEDING_SLOT_IDLE_COMPLETION_GRACE_S",
        20.0,
    )
    last_progress_at = t0
    last_progress_decoded = 0
    last_progress_task_id = 0
    seen_processing_slot = False
    idle_since: float | None = None

    def _run() -> dict[str, Any]:
        return call_orchestrator_forced(
            prompt=prompt,
            force_role=force_role,
            force_mode=force_mode,
            url=url,
            timeout=timeout,
            image_path=image_path,
            cache_prompt=cache_prompt,
            client=client,
            allow_delegation=allow_delegation,
            session_id=session_id,
            scoring_method=scoring_method,
            stop_sequences=stop_sequences,
            request_priority=request_priority,
            workload_class=workload_class,
            batch_id=batch_id,
            batch_placement_mode=batch_placement_mode,
            watcher=watcher,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        while True:
            try:
                resp = fut.result(timeout=1.0)
                elapsed = time.perf_counter() - t0
                break
            except concurrent.futures.TimeoutError:
                elapsed_now = time.perf_counter() - t0

                # ── Proactive timeout: erase slot before httpx timeout ──
                # When approaching the timeout, preemptively kill the
                # server-side generation so the chain (llama.cpp →
                # orchestrator → httpx) unwinds cleanly and the port is
                # free for the next strategy.
                erase_margin = 15
                if elapsed_now >= timeout - erase_margin and poll_port > 0:
                    logger.warning(
                        "  [timeout-erase] %s at %.0fs/%.0fs — erasing port %d",
                        log_label, elapsed_now, timeout, poll_port,
                    )
                    _force_erase_and_verify(poll_port, max_attempts=2, verify_delay=1.0)
                    # Give httpx a moment to receive the response now
                    # that the server-side generation is stopped.
                    try:
                        resp = fut.result(timeout=12.0)
                        elapsed = time.perf_counter() - t0
                    except concurrent.futures.TimeoutError:
                        elapsed = time.perf_counter() - t0
                        resp = {
                            "answer": "",
                            "error": f"timeout after slot erase ({elapsed:.0f}s)",
                            "failure_provenance": _slot_erase_timeout_provenance(
                                elapsed_s=elapsed, server_generation_observed=seen_processing_slot,
                                exc=None, role=force_role, workload_class=workload_class,
                                max_queue_wait_ms=min(int(timeout * 1000), 90_000),
                            ),
                        }
                    except Exception as exc:
                        elapsed = time.perf_counter() - t0
                        resp = {
                            "answer": "",
                            "error": f"timeout after slot erase ({elapsed:.0f}s)",
                            "failure_provenance": _slot_erase_timeout_provenance(
                                elapsed_s=elapsed, server_generation_observed=seen_processing_slot,
                                exc=exc, role=force_role, workload_class=workload_class,
                                max_queue_wait_ms=min(int(timeout * 1000), 90_000),
                            ),
                        }
                    break

                if poll_port <= 0:
                    now_hb = time.perf_counter()
                    if (now_hb - last_heartbeat) >= heartbeat_interval:
                        logger.info(
                            "    ... still waiting for %s (%ds elapsed)",
                            log_label,
                            int(now_hb - t0),
                        )
                        last_heartbeat = now_hb
                    continue
                sp = _read_slot_progress(poll_port, timeout_s=1.0)
                if not sp:
                    continue
                decoded = int(sp.get("n_decoded", 0) or 0)
                is_processing = bool(sp.get("is_processing", False))
                task_id = int(sp.get("task_id", 0) or 0)
                now = t0 + elapsed_now
                if is_processing:
                    seen_processing_slot = True
                    idle_since = None
                elif seen_processing_slot:
                    if idle_since is None:
                        idle_since = now
                    idle_for = now - idle_since
                    if slot_idle_orphan_s > 0 and idle_for >= slot_idle_orphan_s:
                        try:
                            resp = fut.result(timeout=slot_idle_completion_grace_s)
                            elapsed = time.perf_counter() - t0
                        except concurrent.futures.TimeoutError:
                            elapsed = time.perf_counter() - t0
                            resp = {
                                "answer": "",
                                "error": (
                                    f"slot idle while request pending after {elapsed:.0f}s "
                                    f"on port {poll_port}"
                                ),
                                "failure_reason": "slot_idle_orphan",
                            }
                            logger.warning(
                                "  [slot-idle-orphan] %s port=%d pending %.0fs "
                                "(last task=%s decoded=%s)",
                                log_label,
                                poll_port,
                                elapsed,
                                progress["task_id"],
                                progress["last_decoded"],
                            )
                        except Exception as exc:
                            elapsed = time.perf_counter() - t0
                            resp = {"answer": "", "error": str(exc)}
                        break

                if decoded > progress["max_decoded"]:
                    progress["max_decoded"] = decoded
                progress["last_decoded"] = decoded
                progress["last_remain"] = int(sp.get("n_remain", 0) or 0)
                progress["task_id"] = task_id
                progress["source"] = "slots_poll"

                if decoded > last_progress_decoded or (
                    task_id and task_id != last_progress_task_id
                ):
                    last_progress_at = now
                    last_progress_decoded = decoded
                    last_progress_task_id = task_id
                elif (
                    is_processing
                    and slot_stall_watchdog_s > 0
                    and decoded > 0
                    and (now - last_progress_at) >= slot_stall_watchdog_s
                ):
                    stalled_for = now - last_progress_at
                    logger.warning(
                        "  [slot-stall] %s port=%d task=%s decoded=%d unchanged for %.0fs; erasing",
                        log_label,
                        poll_port,
                        task_id,
                        decoded,
                        stalled_for,
                    )
                    _force_erase_and_verify(poll_port, max_attempts=2, verify_delay=1.0)
                    try:
                        resp = fut.result(timeout=12.0)
                        elapsed = time.perf_counter() - t0
                    except (concurrent.futures.TimeoutError, Exception):
                        elapsed = time.perf_counter() - t0
                        resp = {
                            "answer": "",
                            "error": (
                                f"slot stalled on port {poll_port} after "
                                f"{stalled_for:.0f}s with {decoded} tokens"
                            ),
                            "failure_reason": "slot_stalled_no_progress",
                        }
                    break

                if (
                    (now - last_log_at) >= log_every_s
                    or (decoded - last_logged_decoded) >= log_delta_tokens
                ):
                    elapsed = now - t0
                    logger.debug(
                        "  [slot-progress] %s port=%s task=%s decoded=%s remain=%s elapsed=%.1fs",
                        log_label,
                        poll_port,
                        progress["task_id"],
                        decoded,
                        progress["last_remain"],
                        elapsed,
                    )
                    last_log_at = now
                    last_logged_decoded = decoded

                # Heartbeat every 120s so TUI left panel stays alive
                now_hb = now
                if (now_hb - last_heartbeat) >= heartbeat_interval:
                    elapsed_hb = now_hb - t0
                    decoded_hb = progress["max_decoded"]
                    logger.info(
                        "    ... still waiting for %s (%ds elapsed, %d tokens so far)",
                        log_label,
                        int(elapsed_hb),
                        decoded_hb,
                    )
                    last_heartbeat = now_hb
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                resp = {"answer": "", "error": str(exc)}
                break

    return resp, elapsed, progress


def _recover_heavy_ports_if_stuck(url: str, busy_ports: list[int]) -> bool:
    """Attempt targeted backend recovery when heavy ports appear stuck.

    IMPORTANT: avoid full-stack restart in seeding loop.
    """
    if not busy_ports:
        return True
    if os.environ.get("SEEDING_ENABLE_TARGETED_RELOAD", "0") != "1":
        logger.warning(
            "  [recover] heavy ports stuck but targeted reload is disabled "
            "(set SEEDING_ENABLE_TARGETED_RELOAD=1 to enable)"
        )
        return False

    logger.warning(f"  [recover] heavy ports stuck: {busy_ports} — targeted reload")

    port_to_component = {
        8070: "frontdoor",  # frontdoor / coder_escalation / worker_summarize all share this server
        8072: "worker_general",
        # 2026-09-27 ARCHITECT SWAP: :8074 = architect_general (Flash-Next CPU),
        # :8083 = architect_critic (the MI210 27B; coder_escalation/ingest ride it).
        8074: "architect_general",
        8083: "architect_critic",
        8085: "ingest_long_context",
        8087: "vision_escalation",
    }
    components: list[str] = []
    for p in busy_ports:
        c = port_to_component.get(p)
        if c and c not in components:
            components.append(c)

    if not components:
        logger.warning("  [recover] no reloadable components mapped for busy ports")
        return False

    cmd = [
        sys.executable,
        str(STACK_SCRIPT),
        "reload",
        *components,
    ]
    try:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            tail = "\n".join((res.stderr or "").strip().splitlines()[-6:])
            logger.warning(f"  [recover] targeted reload failed (rc={res.returncode}) {tail}")
            return False
    except Exception as exc:
        logger.warning(f"  [recover] targeted reload exception: {exc}")
        return False

    _wait_for_heavy_models_idle(max_wait=180)
    still_busy = _busy_heavy_ports(timeout_s=2.0)
    if still_busy:
        logger.warning(f"  [recover] heavy ports still busy after recovery: {still_busy}")
        return False
    logger.info("  [recover] heavy ports cleared after recovery")
    return True


# ── Tool telemetry normalization ─────────────────────────────────────


def _normalize_tool_telemetry(data: dict[str, Any]) -> None:
    """Normalize tool telemetry fields for downstream consistency.

    Ensures tools_used, tools_called, and tool_timings are aligned even when
    older/partial API responses omit one of the fields.
    """
    if not isinstance(data, dict):
        return

    tools_called = data.get("tools_called") or []
    if not isinstance(tools_called, list):
        tools_called = [str(tools_called)]

    tool_timings = data.get("tool_timings") or []
    if not isinstance(tool_timings, list):
        tool_timings = []

    tools_used_raw = data.get("tools_used", 0)
    try:
        tools_used = int(tools_used_raw or 0)
    except Exception:
        tools_used = 0

    if tool_timings and not tools_called:
        tools_called = [str(t.get("tool_name", "?")) for t in tool_timings]

    inferred_used = max(tools_used, len(tools_called), len(tool_timings))

    # If we have tool names but no timing rows, synthesize placeholders
    # rather than dropping telemetry dimensions.
    if inferred_used > 0 and not tool_timings and tools_called:
        tool_timings = [
            {"tool_name": str(name), "elapsed_ms": 0.0, "success": True}
            for name in tools_called
        ]

    data["tools_called"] = tools_called
    data["tool_timings"] = tool_timings
    data["tools_used"] = inferred_used


def _surface_inband_error(data: dict[str, Any]) -> None:
    """REL-1 Guard 1: surface an in-band ``[ERROR: ...]`` answer into ``error``.

    The orchestrator circuit breaker / llm primitives can return an HTTP-200
    body whose ``answer`` IS an error banner ("[ERROR: Backend unavailable
    (circuit open): ...]") with ``error=None``. Downstream seeding scoring keys
    infra-vs-task classification off ``data["error"]`` only, so without this the
    banner is scored as a WRONG answer and a 0.0 reward is injected into MemRL,
    training the learned router that the role "failed" when the backend was
    merely unavailable. Copy the banner into ``data["error"]`` (only when no
    structured error is already present — HTTP ``error_code``/``error_detail``
    wins) so ``_classify_error`` marks it INFRASTRUCTURE and the row is
    excluded. The raw ``answer`` is left untouched. Mirrors eval_tower.py
    Guard 1. Idempotent; safe to call on any response dict.
    """
    if not isinstance(data, dict) or data.get("error"):
        return
    inband = _inband_error_text(data.get("answer", ""))
    if inband is not None:
        data["error"] = inband
        data.setdefault("failure_reason", "inband_error")


# ── Core orchestrator call ───────────────────────────────────────────


def call_orchestrator_forced(
    prompt: str,
    force_role: str,
    force_mode: str = "direct",
    url: str = DEFAULT_ORCHESTRATOR_URL,
    timeout: int = DEFAULT_TIMEOUT,
    image_path: str = "",
    cache_prompt: bool | None = None,
    client: "httpx.Client | None" = None,
    allow_delegation: bool | None = None,
    session_id: str = "",
    scoring_method: str = "",
    stop_sequences: list[str] | None = None,
    request_priority: str | None = None,
    workload_class: str | None = None,
    batch_id: int | str | None = None,
    batch_placement_mode: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
    n_probs: int | None = None,
    output_schema: dict[str, Any] | None = None,
    prompt_root: str | None = None,
    watcher: Any | None = None,
    llama_port: int | None = None,
    eval_fence: bool | None = None,
) -> dict[str, Any]:
    """Call orchestrator with forced role and mode routing.

    Args:
        prompt: The prompt to process.
        force_role: Force routing to this role.
        force_mode: Force execution mode (direct/repl/delegated).
        url: Orchestrator API URL.
        timeout: Request timeout in seconds.
        image_path: Optional image path for vision tasks.
        cache_prompt: Override prompt caching (None=default).
        client: Reusable httpx.Client for connection pooling.
        allow_delegation: Override delegation (None=feature flag, True=allow, False=disable).
        session_id: Optional session ID for cross-request persistence (Phase 3 checkpoints).
        request_priority: Optional admission priority override. Defaults to the
            legacy background seeding priority.
        workload_class: Optional traffic-class stamp for attribution/shadow
            routing. Omitted when not supplied to preserve legacy payload shape.
        batch_id: Optional batch identifier for inference-tap attribution.
        batch_placement_mode: Optional burst placement intent forwarded to the
            orchestrator. Omitted by default for old-server compatibility.
        tools: Optional OpenAI-compatible function-tool schemas to forward to the
            orchestrator request. Omitted by default to preserve legacy eval traffic.
        tool_choice: Optional OpenAI-compatible tool choice policy for tools.
        max_tokens: Optional response-token cap forwarded to `/chat`. Omitted
            by default to preserve legacy payload shape.
        n_probs: Optional llama.cpp top-k probability capture forwarded to
            `/chat`. Omitted by default to preserve legacy payload shape.
        output_schema: Optional JSON schema forwarded to `/chat` for direct
            backend-constrained output. Omitted by default to preserve legacy
            payload shape.
        prompt_root: Optional scratch prompt root forwarded to `/chat` for
            AutoPilot prompt-isolation evals. Omitted by default to preserve
            legacy payload shape.
        watcher: Optional OrchestratorWatcher (autopilot.scripts.autopilot.
            orchestrator_watch) — when supplied, exogenous reloads of the
            orchestrator API or the target llama-server are detected and
            the request is retried after waiting for /health. Backward-
            compatible: watcher=None preserves the legacy direct-post-with-
            exception-swallow behavior exactly.
        eval_fence: AP-54 eval knowledge fence request flag, forwarded to
            `/chat` as ``eval_fence``. Omitted when None (legacy payload shape).
            An API build that predates the field ignores it (pydantic
            extra='ignore'); the response then carries no ``eval_fence`` echo and
            the caller must record the row as unfenced.
        llama_port: Optional explicit port hint for the target llama-server.
            When omitted, the watcher resolves it from force_role via
            /llama_fleet_ids.

    Returns:
        Response dict with answer, tokens, timing, etc. When a watcher is
        supplied, an extra "_meta" key carries the resilient_post meta dict
        (clean / exogenous_recovered / exogenous_unrecovered / external_restart
        / real_failure / retry_count / wait_s / marker_changes) for the
        seeding/eval pipeline to propagate up to EvalResult.
    """
    import httpx

    timeout = max(1, int(math.ceil(float(timeout))))

    # ── Guard 3 (REL-1 eval-honesty): deadline-starvation floor ──────────
    # 2026-07-21 EV-11c incident: client-deadline starvation on long MATH-tail
    # questions shrank the effective per-call budget to ~1s; those doomed calls
    # 504'd and tripped the production circuit breaker, which then served
    # in-band "[ERROR: ...]" text as answers. Refuse to fire an eval llama call
    # whose remaining budget is below a sane minimum — fail the question
    # pre-send as a deadline_starved ERROR (REL-1: an excluded row, not a wrong
    # answer) instead of tripping the breaker. `client_deadline_unix_s` below is
    # computed as `now + timeout`, so the remaining budget for the llama call is
    # exactly `timeout`. Scoped to eval traffic (workload_class == "eval_batch")
    # so the 14 non-eval callers of this function keep their EXACT legacy path.
    if str(workload_class or "") == "eval_batch":
        _min_llama_budget_s = _env_float("AUTOPILOT_EVAL_MIN_LLAMA_BUDGET_S", 30.0)
        if _min_llama_budget_s > 0 and float(timeout) < _min_llama_budget_s:
            logger.error(
                "  [deadline-starved] REFUSED pre-send: role=%s budget=%.1fs "
                "< floor=%.1fs — failing question as error rather than firing a "
                "doomed call that trips the circuit breaker",
                force_role, float(timeout), _min_llama_budget_s,
            )
            return {
                "answer": "",
                "error": (
                    f"deadline_starved: eval llama budget {float(timeout):.1f}s "
                    f"< floor {_min_llama_budget_s:.0f}s (role={force_role}); "
                    "refused pre-send"
                ),
                "failure_reason": "deadline_starved",
            }

    if str(workload_class or "") == "eval_batch":
        # Queueing and inference share one absolute request deadline.  Reserving
        # at most half of that deadline for placement guarantees that a request
        # admitted at the queue limit still retains an equally large inference
        # budget.  The 300s ceiling is finite but accommodates mixed-role burst
        # waves; the former fixed 90s cap failed healthy requests whenever all
        # EvalTower workers temporarily occupied another physical cohort.
        max_queue_wait_ms = min(int(timeout * 500), 300_000)
    else:
        max_queue_wait_ms = min(int(timeout * 1000), 90_000)

    payload: dict[str, Any] = {
        "prompt": prompt,
        "real_mode": True,
        "force_role": force_role,
        "force_mode": force_mode,
        "timeout_s": timeout,
        "client_deadline_unix_s": time.time() + float(timeout),
        # Phase C (cross-role-bw-aware-routing): seeder traffic is background.
        # The orchestrator's contention gate queues these behind any active
        # foreground decode on a known-bad/unknown pair (e.g. when frontdoor
        # is decoding, an ingest probe is held until frontdoor releases).
        # Without this stamp, autopilot probes contend with user chats and
        # crater both sides per the 2026-05-24 contention matrix.
        "request_priority": (
            request_priority
            if str(request_priority or "").strip()
            else "background"
        ),
        "max_queue_wait_ms": max_queue_wait_ms,
    }
    if image_path:
        payload["image_path"] = image_path
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    if allow_delegation is not None:
        payload["allow_delegation"] = allow_delegation
    if session_id:
        payload["session_id"] = session_id
    if scoring_method:
        payload["scoring_method"] = scoring_method
    if stop_sequences:
        payload["stop_sequences"] = stop_sequences
    if workload_class:
        payload["workload_class"] = workload_class
    if batch_id is not None:
        payload["batch_id"] = batch_id
    if batch_placement_mode:
        payload["batch_placement_mode"] = batch_placement_mode
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if max_tokens is not None:
        payload["max_tokens"] = max(1, int(max_tokens))
    if n_probs is not None:
        payload["n_probs"] = max(0, int(n_probs))
    if output_schema is not None:
        payload["output_schema"] = output_schema
    if prompt_root:
        payload["x_orchestrator_prompt_root"] = str(prompt_root)
    if eval_fence is not None:
        payload["eval_fence"] = bool(eval_fence)

    def _timeout_result(exc: BaseException, reason: str) -> dict[str, Any]:
        _, detail = _classify_exc(exc)
        return {
            "answer": "",
            "error": _error_text(exc, detail),
            # Structural disposition signal. Without it the only evidence that
            # this row is a transport failure is the exception's text, which a
            # substring classifier must recognise to exclude — and silently
            # scores as a WRONG answer when it does not.
            "failure_reason": reason,
            "failure_provenance": _client_transport_timeout_provenance(
                reason=reason,
                exc=exc,
                role=force_role,
                workload_class=workload_class,
                max_queue_wait_ms=int(payload["max_queue_wait_ms"]),
            ),
        }

    def _transport_error_result(exc: BaseException, reason: str) -> dict[str, Any]:
        """Terminal non-timeout transport/protocol failure (HTTP 4xx/5xx raised
        by raise_for_status, an unparseable body, a protocol error). Carries the
        structural ``failure_reason`` so the row is classified as INFRA-FAILED
        rather than scored as a wrong answer."""
        _, detail = _classify_exc(exc)
        return {
            "answer": "",
            "error": _error_text(exc, detail),
            "failure_reason": reason,
        }

    def _execute_direct() -> dict[str, Any]:
        """One direct POST + response parse. RAISES on transport/connection
        exceptions; returns the parsed response dict (including structured HTTP
        error bodies) otherwise. This is the EXACT legacy request body, lifted
        into a helper so the legacy swallow-path and the eval reconnect-backoff
        path share identical response handling."""
        if client is not None:
            response = client.post(f"{url}/chat", json=payload, timeout=timeout)
        else:
            response = httpx.post(
                f"{url}/chat",
                json=payload,
                timeout=timeout,
            )
        if response.status_code >= 400:
            try:
                data = response.json()
            except Exception:
                response.raise_for_status()
                raise
            if isinstance(data, dict) and (
                data.get("error_code") or data.get("error_detail")
            ):
                error_code = data.get("error_code")
                if error_code and not data.get("error"):
                    data["error"] = data.get("error_detail") or f"HTTP {error_code}"
                # A non-2xx status means the SERVER REFUSED the request — a
                # per-slot context overflow returns 400. That is a capacity
                # fact about the fleet, never evidence about answer quality,
                # so record it structurally instead of leaving the disposition
                # to a substring match on the body text.
                data.setdefault("http_status", int(response.status_code))
                data.setdefault("failure_reason", "http_status")
                _surface_inband_error(data)
                _normalize_tool_telemetry(data)
                return data
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            error_code = data.get("error_code")
            if error_code and not data.get("error"):
                data["error"] = data.get("error_detail") or f"HTTP {error_code}"
            _surface_inband_error(data)
            _normalize_tool_telemetry(data)
        return data

    def _execute_watcher() -> dict[str, Any]:
        """Watcher path: delegate to resilient_post for exogenous-reload
        detection. Attaches the meta dict as ``_meta``. Lazy-imports the
        autopilot module so the legacy path never touches it."""
        import sys as _sys
        from pathlib import Path as _Path
        _ap_dir = _Path(__file__).resolve().parents[1] / "autopilot"
        if str(_ap_dir) not in _sys.path:
            _sys.path.insert(0, str(_ap_dir))
        from resilient_http import resilient_post  # type: ignore[import-not-found]

        data, meta = resilient_post(
            f"{url}/chat",
            json=payload,
            timeout=timeout,
            client=client,
            watcher=watcher,
            llama_port=llama_port,
            llama_role=force_role,
        )
        if isinstance(data, dict):
            error_code = data.get("error_code")
            if error_code and not data.get("error"):
                data["error"] = data.get("error_detail") or f"HTTP {error_code}"
            _surface_inband_error(data)
            _normalize_tool_telemetry(data)
            # Attach meta as _meta so downstream consumers can inspect without
            # disturbing existing data keys.
            data["_meta"] = meta
            reason = str(meta.get("reason") or "")
            # Same structural stamp as the direct path: a watcher-path failure
            # must be classifiable as INFRA-FAILED without reading its message.
            if reason and not meta.get("clean") and "failure_reason" not in data:
                data["failure_reason"] = reason
            if (
                data.get("error")
                and "failure_provenance" not in data
                and reason in _TRANSPORT_TIMEOUT_REASONS
            ):
                data["failure_provenance"] = _client_transport_timeout_provenance(
                    reason=reason,
                    exc=RuntimeError(str(meta.get("detail") or data.get("error") or reason)),
                    role=force_role,
                    workload_class=workload_class,
                    max_queue_wait_ms=int(payload["max_queue_wait_ms"]),
                )
        return data

    # ── Non-eval traffic: preserve legacy terminal behavior. Critical for the
    # non-autopilot callers of this function (14 impacted symbols per GitNexus
    # blast-radius audit). Exceptions remain swallowed. The only additive
    # shape is typed provenance for an observed transport timeout; connection
    # failures and all successful responses remain unchanged.
    if str(workload_class or "") != "eval_batch":
        if watcher is None:
            try:
                return _execute_direct()
            except Exception as exc:
                reason, _detail = _classify_exc(exc)
                if reason in _TRANSPORT_TIMEOUT_REASONS:
                    return _timeout_result(exc, reason)
                return _transport_error_result(exc, reason)
        return _execute_watcher()

    # ── Eval traffic: reconnect backoff on CONNECTION-level failures ─────
    # A mid-run API reload refuses/resets in-flight POSTs. Retry ONLY those
    # connection-level failures with exponential backoff (1s,2s,4s,8s,16s…),
    # bounded by AUTOPILOT_EVAL_RECONNECT_MAX_S (default 60s of cumulative
    # sleep), then honestly error-row so persistent unreachability stays an
    # excluded REL-1 error. Timeouts / HTTP errors / in-band banners are NOT
    # reconnectable and keep their prior terminal semantics.
    max_reconnect_s = _env_float("AUTOPILOT_EVAL_RECONNECT_MAX_S", 60.0)
    delay = 1.0
    slept = 0.0
    attempts = 0
    last_detail = ""
    while True:
        attempts += 1
        if watcher is None:
            try:
                return _execute_direct()
            except Exception as exc:
                reason, _detail = _classify_exc(exc)
                if reason not in _RECONNECT_REASONS:
                    # Not connection-level (timeout / http status / other):
                    # Client timeouts carry typed provenance but never claim
                    # that server-side generation did not start.
                    if reason in _TRANSPORT_TIMEOUT_REASONS:
                        return _timeout_result(exc, reason)
                    return _transport_error_result(exc, reason)
                last_detail = str(exc)
        else:
            data = _execute_watcher()
            meta = data.get("_meta") if isinstance(data, dict) else None
            meta = meta if isinstance(meta, dict) else {}
            # Clean success or a watcher-recovered request → return as-is.
            if meta.get("clean") or meta.get("exogenous_recovered"):
                return data
            reason = str(meta.get("reason") or "")
            if reason not in _RECONNECT_REASONS:
                # Terminal non-connection failure (timeout / http / in-band /
                # recovered): return whatever resilient_post produced.
                return data
            last_detail = str(meta.get("detail") or data.get("error") or "")

        # Reached only on a CONNECTION-level failure. Back off while budget
        # remains, else honestly error-row.
        if slept + delay > max_reconnect_s:
            logger.error(
                "  [eval-reconnect] role=%s api unreachable after backoff "
                "(%d attempts, waited %.0fs, budget %.0fs): %s",
                force_role, attempts, slept, max_reconnect_s, last_detail,
            )
            return {
                "answer": "",
                "error": (
                    f"api_unreachable_after_backoff: {last_detail} "
                    f"(role={force_role}, attempts={attempts}, "
                    f"waited={slept:.0f}s, budget={max_reconnect_s:.0f}s)"
                ),
                "failure_reason": "api_unreachable_after_backoff",
            }
        logger.warning(
            "  [eval-reconnect] role=%s connection failure (%s); backing off "
            "%.0fs then retrying (attempt %d, waited %.0fs/%.0fs)",
            force_role, last_detail, delay, attempts, slept, max_reconnect_s,
        )
        time.sleep(delay)
        slept += delay
        delay *= 2
