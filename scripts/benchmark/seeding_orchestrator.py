"""Slot management, HTTP calls to the orchestrator, and tool telemetry.

Handles llama-server slot erasure, busy-port detection, progress polling,
and the core ``call_orchestrator_forced`` function.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
import sys
import time
from typing import Any

from seeding_types import (
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_TIMEOUT,
    HEAVY_PORTS,
    PROJECT_ROOT,
    ROLE_PORT,
    STACK_SCRIPT,
    state,
)
from seeding_infra import _wait_for_heavy_models_idle

__all__ = [
    "_SLOT_ERASE_CAPABILITY",
    "_busy_heavy_ports",
    "_call_orchestrator_with_slot_poll",
    "_erase_slots",
    "_force_erase_and_verify",
    "_normalize_tool_telemetry",
    "_read_slot_progress",
    "_recover_heavy_ports_if_stuck",
    "call_orchestrator_forced",
]

logger = logging.getLogger("seed_specialist_routing")


# Per-port slot erase strategy cache.
# None = unknown, str = preferred strategy, False = unsupported on this build.
_SLOT_ERASE_CAPABILITY: dict[int, str | None | bool] = {}


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
                    logger.info(f"  → erased slot {slot_id} on port {port}")
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
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Call orchestrator while polling slot progress for live visibility."""

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
                    except (concurrent.futures.TimeoutError, Exception):
                        elapsed = time.perf_counter() - t0
                        resp = {
                            "answer": "",
                            "error": f"timeout after slot erase ({elapsed:.0f}s)",
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
                if decoded > progress["max_decoded"]:
                    progress["max_decoded"] = decoded
                progress["last_decoded"] = decoded
                progress["last_remain"] = int(sp.get("n_remain", 0) or 0)
                progress["task_id"] = int(sp.get("task_id", 0) or 0)
                progress["source"] = "slots_poll"

                now = time.perf_counter()
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
                now_hb = time.perf_counter()
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
        8080: "coder_escalation",
        8081: "coder_escalation",
        8083: "architect_general",
        8084: "architect_coding",
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

    Returns:
        Response dict with answer, tokens, timing, etc.
    """
    import httpx

    payload: dict[str, Any] = {
        "prompt": prompt,
        "real_mode": True,
        "force_role": force_role,
        "force_mode": force_mode,
        "timeout_s": timeout,
        "client_deadline_unix_s": time.time() + float(timeout),
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

    try:
        if client is not None:
            response = client.post(f"{url}/chat", json=payload, timeout=timeout)
        else:
            response = httpx.post(
                f"{url}/chat",
                json=payload,
                timeout=timeout,
            )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            error_code = data.get("error_code")
            if error_code and not data.get("error"):
                data["error"] = data.get("error_detail") or f"HTTP {error_code}"
            _normalize_tool_telemetry(data)
        return data
    except Exception as e:
        return {"answer": "", "error": str(e)}
