#!/usr/bin/env python3
"""MemRL Episodic Seeding for 3-Way Routing.

.. deprecated::
    This monolithic file is **DEPRECATED** and moved to ``deprecated/``.
    The canonical entry point is now ``seed_specialist_routing.py`` (formerly v2)
    which delegates to the extracted ``seeding_*.py`` modules.
    This file is archived for reference only — do NOT import from it.

THE canonical evaluation script for training frontdoor probability estimation.
Runs each question through multiple configurations, scores deterministically,
and injects rewards so MemRL learns optimal routing decisions.

## Preferred Mode: 3-Way Routing (--3way)

Uses binary rewards for faithful P(success|action) estimation:
  - SELF:direct  → Frontdoor without tools
  - SELF:repl    → Frontdoor with tools, delegation disabled
  - ARCHITECT    → Architect with full delegation freedom
  - WORKER       → Scored via delegation chain attribution

Binary reward: 1.0 for pass, 0.0 for fail. Cost stored in metadata for Optuna.
TD learning with α=0.1 converges Q-values to empirical success rates.

Usage (3-way mode - RECOMMENDED):
    # THE command. Binary rewards for faithful probability estimation.
    python scripts/benchmark/seed_specialist_routing.py \\
      --3way --suites all --sample-size 10

    # Dry run (no reward injection)
    python scripts/benchmark/seed_specialist_routing.py \\
      --3way --dry-run --suites thinking --sample-size 3

## Legacy Mode: Comparative Rewards

Uses cost-weighted comparative rewards (deprecated for new seeding):
  specialist correct & frontdoor wrong → +1.0 (specialist clearly better)
  specialist wrong & frontdoor right   → -0.5 (specialist worse)
  both correct                         → 0.5 - λ*max(0, cost_ratio-1) (cost-aware)
  both wrong                           → -0.3 (neither helps)

Usage (legacy mode):
    # Continuous comparative seeding (legacy)
    python scripts/benchmark/seed_specialist_routing.py \\
      --continuous --suites all --sample-size 10 --cooldown 2.0 --preflight

    # Quick stats from all sessions
    python scripts/benchmark/seed_specialist_routing.py --stats
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bootstrap: needed before seeding_types can be imported. Same value as seeding_types.PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy HTTP client logs (httpx, httpcore, urllib3)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Per-port slot erase strategy cache.
# None = unknown, str = preferred strategy, False = unsupported on this build.
_SLOT_ERASE_CAPABILITY: dict[int, str | None | bool] = {}

# ── Re-exports from extracted modules (test backwards compatibility) ──
# Tests import these symbols from this file; keep them accessible here.

from seeding_types import (  # noqa: E402, F401
    ARCHITECT_MODES,
    ARCHITECT_ROLES,
    ComparativeResult,
    DEBUG_PROMPTS_DIR,
    DEFAULT_MODES,
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_ROLES,
    DEFAULT_SUITES,
    DEFAULT_TIMEOUT,
    ESCALATION_REWARD,
    EVAL_DIR,
    HEAVY_PORTS,
    HealthCheckError,
    MODEL_PORTS,
    ROLE_COST_TIER,
    ROLE_PORT,
    RoleResult,
    SEEN_FILE,
    VISION_MODES,
    VISION_ROLES,
    state,
)
from seeding_rewards import (  # noqa: E402, F401
    DEFAULT_BASELINE_TPS,
    _inject_escalation_chains_http,
    _inject_rewards_http,
    compute_comparative_rewards,
    detect_escalation_chains,
    # Phase 4: Binary rewards for 3-way routing
    success_reward,
    compute_3way_rewards,
    score_delegation_chain,
    compute_tool_value,
)
from seeding_types import (  # noqa: E402 (additional imports for 3-way routing)
    ACTION_SELF_DIRECT,
    ACTION_SELF_REPL,
    ACTION_ARCHITECT,
    ACTION_WORKER,
    THREE_WAY_ACTIONS,
)
from seeding_infra import (  # noqa: E402, F401
    MAX_RECOVERY_ATTEMPTS,
    _attempt_recovery,
    _check_server_health,
    _wait_for_heavy_models_idle,
    run_preflight,
)


# ── Signal handlers ──────────────────────────────────────────────────


def _handle_sigint(sig, frame):
    if state.shutdown:
        state.close_poll_client()
        sys.exit(1)
    state.shutdown = True
    logger.info("[SIGINT] Finishing current question, then stopping...")


def _handle_sigterm(sig, frame):
    state.shutdown = True
    state.close_poll_client()


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigterm)


# ── Checkpoint management ─────────────────────────────────────────────


def _ensure_eval_dir():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_checkpoint(session_id: str) -> list[ComparativeResult]:
    """Load completed results from a session's JSONL checkpoint."""
    path = EVAL_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Reconstruct ComparativeResult from serialized dict
                role_results = {}
                for k, v in data.get("role_results", {}).items():
                    role_results[k] = RoleResult(**v)
                results.append(ComparativeResult(
                    suite=data["suite"],
                    question_id=data["question_id"],
                    prompt=data.get("prompt", ""),
                    expected=data.get("expected", ""),
                    dataset_source=data.get("dataset_source", "yaml"),
                    prompt_hash=data.get("prompt_hash", ""),
                    timestamp=data.get("timestamp", ""),
                    role_results=role_results,
                    rewards=data.get("rewards", {}),
                    rewards_injected=data.get("rewards_injected", 0),
                ))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return results


def append_checkpoint(session_id: str, result: ComparativeResult):
    """Append one result to the session's JSONL file (atomic-ish)."""
    _ensure_eval_dir()
    path = EVAL_DIR / f"{session_id}.jsonl"
    line = json.dumps(asdict(result), ensure_ascii=False)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _checkpoint_3way(session_id: str, result: "ThreeWayResult"):
    """Append one 3-way result to the session's JSONL file (atomic-ish)."""
    _ensure_eval_dir()
    path = EVAL_DIR / f"{session_id}.jsonl"
    line = json.dumps(asdict(result), ensure_ascii=False)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_seen_questions() -> set[str]:
    """Load all prompt_ids ever evaluated across all sessions."""
    seen: set[str] = set()
    if not EVAL_DIR.exists():
        return seen

    for path in EVAL_DIR.glob("*.jsonl"):
        if path.name == "seen_questions.jsonl":
            # Read dedicated seen file
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pid = data.get("prompt_id", "")
                        if pid:
                            seen.add(pid)
                    except json.JSONDecodeError:
                        continue
        else:
            # Read from checkpoint files
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pid = data.get("question_id", "")
                        if pid:
                            seen.add(pid)
                    except json.JSONDecodeError:
                        continue

    return seen


def record_seen(prompt_id: str, suite: str, session_id: str):
    """Append to the global seen questions file."""
    _ensure_eval_dir()
    entry = {
        "prompt_id": prompt_id,
        "suite": suite,
        "session": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(SEEN_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ── Question sampling ─────────────────────────────────────────────────


def _load_from_dataset_adapter(
    suite_name: str, sample_count: int, seed: int,
) -> list[dict]:
    """Sample questions from HF dataset adapters."""
    try:
        from dataset_adapters import get_adapter, ADAPTER_SUITES
    except ImportError:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from dataset_adapters import get_adapter, ADAPTER_SUITES
        except ImportError:
            return []

    if suite_name not in ADAPTER_SUITES:
        return []

    adapter = get_adapter(suite_name)
    if adapter is None:
        return []

    prompts = adapter.sample(n=sample_count, seed=seed)
    if prompts:
        logger.info(f"  [{suite_name}] Sampled {len(prompts)} from "
                     f"{adapter.total_available} HF dataset questions (seed={seed})")
    return prompts


def _load_from_yaml(
    suite_name: str, sample_count: int, seed: int,
) -> list[dict]:
    """Fall back to static YAML debug prompts."""
    try:
        import yaml
    except ImportError:
        return []

    yaml_path = DEBUG_PROMPTS_DIR / f"{suite_name}.yaml"
    if not yaml_path.exists():
        return []

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    questions = data.get("questions", [])
    if not questions:
        return []

    rng = random.Random(seed)
    n = min(sample_count, len(questions))
    sampled = rng.sample(questions, n)
    logger.info(f"  [{suite_name}] Sampled {n}/{len(questions)} from YAML (seed={seed})")

    result = []
    for q in sampled:
        result.append({
            "id": q["id"],
            "suite": suite_name,
            "prompt": q["prompt"].strip(),
            "context": "",
            "expected": q.get("expected", ""),
            "image_path": q.get("image_path", ""),
            "tier": q.get("tier", 1),
            "scoring_method": q.get("scoring_method", "exact_match"),
            "scoring_config": q.get("scoring_config", {}),
            "dataset_source": "yaml",
        })
    return result


def sample_unseen_questions(
    suites: list[str],
    sample_per_suite: int,
    seen: set[str],
    seed: int,
    *,
    use_pool: bool = True,
    allow_reseen: bool = False,
) -> list[dict]:
    """Sample questions not in the seen set, interleaved across suites.

    If use_pool=True (default), tries the pre-extracted question pool first
    (~100ms). Falls back to HF dataset adapters, then YAML.

    If ``allow_reseen`` (debug mode), backfills with seen questions when a
    suite is exhausted.  Normal mode skips exhausted suites.

    Returns questions interleaved by suite (round-robin) so the orchestrator
    sees diverse question types early rather than processing one suite at a time.
    """
    suite_names = DEFAULT_SUITES if suites == ["all"] else suites

    # Try the pre-extracted pool first
    if use_pool:
        try:
            from question_pool import POOL_FILE, build_pool, load_pool, sample_from_pool

            if not POOL_FILE.exists():
                logger.info("Question pool not found — building automatically (one-time)...")
                build_pool()

            pool = load_pool()
            if pool:
                result = sample_from_pool(
                    pool, suite_names, sample_per_suite, seed, seen,
                    allow_reseen=allow_reseen,
                )
                if result:
                    logger.info(f"Sampled {len(result)} questions from pool (fast path)")
                    return result
                logger.info("Pool returned no results — falling back to adapters")
        except Exception as e:
            logger.warning(f"Pool loading failed ({e}) — falling back to adapters")

    per_suite: list[list[dict]] = []

    for suite_name in suite_names:
        # Request 20x to ensure enough unseen after dedup (adapters may cap)
        oversample = sample_per_suite * 20

        prompts = _load_from_dataset_adapter(suite_name, oversample, seed)
        if not prompts:
            prompts = _load_from_yaml(suite_name, oversample, seed)

        # Filter out seen questions
        fresh = [p for p in prompts if p["id"] not in seen]
        if len(fresh) < len(prompts):
            filtered = len(prompts) - len(fresh)
            logger.info(f"  [{suite_name}] Filtered {filtered} previously seen questions")

        per_suite.append(fresh[:sample_per_suite])

    # Interleave: round-robin across suites (thinking_q1, general_q1, math_q1, ..., thinking_q2, ...)
    all_prompts: list[dict] = []
    max_len = max((len(s) for s in per_suite), default=0)
    for i in range(max_len):
        for suite_questions in per_suite:
            if i < len(suite_questions):
                all_prompts.append(suite_questions[i])

    return all_prompts


# ── Core functions ────────────────────────────────────────────────────


def _erase_slots(port: int) -> None:
    """Force-cancel in-progress inference on a llama-server port.

    After a timeout the server may still be grinding on the old request.
    Erasing slots prevents cascading timeouts on subsequent requests.
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
            if slot.get("is_processing"):
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
    # Reset capability cache — a previous transient failure should not
    # permanently disable erase attempts.
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
            # If status is unknown, defer to existing call-path error handling.
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

    Returns:
        Response dict with answer, tokens, timing, etc.
    """
    import httpx

    payload: dict[str, Any] = {
        "prompt": prompt,
        "real_mode": True,
        "force_role": force_role,
        "force_mode": force_mode,
    }
    if image_path:
        payload["image_path"] = image_path
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    if allow_delegation is not None:
        payload["allow_delegation"] = allow_delegation

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

    try:
        if client is not None:
            # Use per-request timeout override
            response = client.post(f"{url}/chat", json=payload, timeout=timeout)
        else:
            response = httpx.post(
                f"{url}/chat",
                json=payload,
                timeout=timeout,
            )
        response.raise_for_status()
        data = response.json()
        # Surface structured orchestrator failures through a single "error" field.
        if isinstance(data, dict):
            error_code = data.get("error_code")
            if error_code and not data.get("error"):
                data["error"] = data.get("error_detail") or f"HTTP {error_code}"
            _normalize_tool_telemetry(data)
        return data
    except Exception as e:
        return {"answer": "", "error": str(e)}


def score_answer_deterministic(
    answer: str,
    expected: str,
    scoring_method: str = "exact_match",
    scoring_config: dict[str, Any] | None = None,
) -> bool:
    """Score an answer deterministically."""
    from benchmark.debug_scorer import score_answer

    return score_answer(answer, expected, scoring_method, scoring_config or {})


INFRA_PATTERNS = [
    "timed out", "timeout", "connection", "refused",
    "unreachable", "502", "503", "504", "connecterror",
    "readtimeout", "backend down", "server error",
    "server disconnected without sending a response",
    "remoteprotocolerror", "connection reset", "broken pipe",
    "temporarily unavailable", "name or service not known",
]


def _classify_error(error_str: str | None) -> str:
    """Classify error as infrastructure or task failure."""
    if error_str is None:
        return "none"
    error_lower = error_str.lower()
    if any(p in error_lower for p in INFRA_PATTERNS):
        return "infrastructure"
    return "task_failure"


# ── Phase 4: 3-Way Routing Evaluation ─────────────────────────────────


def _is_coding_task(prompt: str) -> bool:
    """Heuristic to determine if a task is coding-related.

    Used to select architect_coding vs architect_general.
    """
    coding_indicators = [
        "code", "function", "implement", "debug", "refactor",
        "class", "method", "algorithm", "bug", "error",
        "syntax", "compile", "runtime", "test", "unittest",
        "python", "javascript", "typescript", "rust", "go",
        "def ", "async ", "import ", "return ", "class ",
    ]
    prompt_lower = prompt.lower()
    return any(ind in prompt_lower for ind in coding_indicators)


def _adaptive_timeout_s(
    *,
    role: str,
    mode: str,
    prompt: str,
    is_vl: bool,
    hard_timeout_s: int,
) -> int:
    """Return a generous per-call timeout.

    Previous per-role caps (frontdoor=180, vision=240, etc.) caused premature
    INFRA classifications when the server was still generating.  The llama.cpp
    server keeps generating after client disconnect, so tight timeouts only
    waste the work.  Use a flat 600s ceiling; optimize later once we have
    solid per-role telemetry.
    """
    return max(60, int(hard_timeout_s or DEFAULT_TIMEOUT))


def _bump_timeout_from_observed(
    *,
    current_s: int,
    observed_s: float,
    factor: float,
    slack_s: int,
    hard_timeout_s: int,
    role_cap_s: int,
) -> int:
    """Increase timeout based on observed earlier stage runtime for this question.

    With the flat 600s ceiling from _adaptive_timeout_s, this function now
    only raises current_s if the observed time suggests it's too low.
    """
    if observed_s <= 0:
        return current_s
    observed_budget = int(observed_s * factor + slack_s)
    return max(current_s, min(observed_budget, max(60, int(hard_timeout_s or DEFAULT_TIMEOUT))))


def _eval_single_config(
    prompt: str,
    expected: str,
    scoring_method: str,
    scoring_config: dict,
    role: str,
    mode: str,
    url: str,
    timeout: int,
    client: "httpx.Client",
    allow_delegation: bool,
    image_path: str = "",
    log_label: str = "",
    format_fn=None,
) -> tuple[RoleResult, dict]:
    """Call the orchestrator, score, build RoleResult, and handle infra errors.

    Returns:
        (role_result, raw_response_dict) tuple.
    """
    port = ROLE_PORT.get(role, 0)
    did_recover_precheck = False
    if port in HEAVY_PORTS:
        # Do not spend a full 600s here; call timeout and idle wait should share budget.
        idle_wait_cap = max(30, min(120, int(timeout // 2) if timeout else 120))
        _wait_for_heavy_models_idle(max_wait=idle_wait_cap)

        busy_ports = _busy_heavy_ports(timeout_s=2.0)
        if busy_ports:
            # Best-effort attempt to clear stragglers before issuing a new call.
            for bp in busy_ports:
                _erase_slots(bp)
            time.sleep(1.0)
            still_busy = _busy_heavy_ports(timeout_s=2.0)
            if still_busy:
                did_recover_precheck = _recover_heavy_ports_if_stuck(url, still_busy)

    logger.info(f"  → {log_label} ({role}:{mode}, timeout={timeout}s)...")
    resp, elapsed, slot_progress = _call_orchestrator_with_slot_poll(
        prompt=prompt,
        force_role=role,
        force_mode=mode,
        url=url,
        timeout=timeout,
        image_path=image_path,
        cache_prompt=False,
        client=client,
        allow_delegation=allow_delegation,
        log_label=log_label,
        poll_port=port,
    )
    max_decoded = int(slot_progress.get("max_decoded", 0) or 0)
    if max_decoded > 0:
        resp["tokens_generated_estimate"] = max_decoded
    resp["backend_task_id"] = int(slot_progress.get("task_id", 0) or 0)
    resp["slot_progress_source"] = str(slot_progress.get("source", "") or "")

    answer = resp.get("answer", "")
    error = resp.get("error")
    error_type = _classify_error(error)

    if error_type == "infrastructure":
        passed = None
    elif error:
        passed = False
    else:
        passed = score_answer_deterministic(answer, expected, scoring_method, scoring_config)

    rr = RoleResult(
        role=role,
        mode=mode,
        answer=answer or "",
        passed=bool(passed) if passed is not None else False,
        elapsed_seconds=elapsed,
        error=error,
        error_type=error_type,
        tokens_generated=resp.get("tokens_generated", 0),
        tools_used=resp.get("tools_used", 0),
        tools_called=resp.get("tools_called", []),
        delegation_events=resp.get("delegation_events", []),
        tools_success=resp.get("tools_success"),
        delegation_success=resp.get("delegation_success"),
        routed_to=resp.get("routed_to", ""),
        role_history=resp.get("role_history", []),
        predicted_tps=resp.get("predicted_tps", 0.0),
        generation_ms=resp.get("generation_ms", 0.0),
        tokens_generated_estimate=resp.get("tokens_generated_estimate", 0),
        backend_task_id=resp.get("backend_task_id", 0),
        slot_progress_source=resp.get("slot_progress_source", ""),
    )

    if error_type == "infrastructure" and resp.get("tokens_generated", 0) == 0:
        target_port = ROLE_PORT.get(role, 0)
        if target_port:
            _force_erase_and_verify(target_port)
        # One recovery retry for zero-token infra failures on heavy paths.
        if port in HEAVY_PORTS and not did_recover_precheck:
            busy_now = _busy_heavy_ports(timeout_s=2.0)
            if _recover_heavy_ports_if_stuck(url, busy_now):
                logger.info(f"  [retry] {log_label} retry after recovery")
                resp2, elapsed2, slot_progress2 = _call_orchestrator_with_slot_poll(
                    prompt=prompt,
                    force_role=role,
                    force_mode=mode,
                    url=url,
                    timeout=timeout,
                    image_path=image_path,
                    cache_prompt=False,
                    client=client,
                    allow_delegation=allow_delegation,
                    log_label=log_label,
                    poll_port=port,
                )
                max_decoded2 = int(slot_progress2.get("max_decoded", 0) or 0)
                max_decoded = max(max_decoded, max_decoded2)
                if max_decoded > 0:
                    resp2["tokens_generated_estimate"] = max_decoded
                resp2["backend_task_id"] = int(slot_progress2.get("task_id", 0) or 0) or int(
                    slot_progress.get("task_id", 0) or 0
                )
                resp2["slot_progress_source"] = (
                    str(slot_progress2.get("source", "") or "")
                    or str(slot_progress.get("source", "") or "")
                )

                answer2 = resp2.get("answer", "")
                error2 = resp2.get("error")
                error_type2 = _classify_error(error2)
                if error_type2 == "infrastructure":
                    passed2 = None
                elif error2:
                    passed2 = False
                else:
                    passed2 = score_answer_deterministic(
                        answer2, expected, scoring_method, scoring_config
                    )

                rr = RoleResult(
                    role=role,
                    mode=mode,
                    answer=answer2 or "",
                    passed=bool(passed2) if passed2 is not None else False,
                    elapsed_seconds=elapsed + elapsed2,
                    error=error2,
                    error_type=error_type2,
                    tokens_generated=resp2.get("tokens_generated", 0),
                    tools_used=resp2.get("tools_used", 0),
                    tools_called=resp2.get("tools_called", []),
                    delegation_events=resp2.get("delegation_events", []),
                    tools_success=resp2.get("tools_success"),
                    delegation_success=resp2.get("delegation_success"),
                    routed_to=resp2.get("routed_to", ""),
                    role_history=resp2.get("role_history", []),
                    predicted_tps=resp2.get("predicted_tps", 0.0),
                    generation_ms=resp2.get("generation_ms", 0.0),
                    tokens_generated_estimate=resp2.get("tokens_generated_estimate", 0),
                    backend_task_id=resp2.get("backend_task_id", 0),
                    slot_progress_source=resp2.get("slot_progress_source", ""),
                )
                resp = resp2
                error_type = error_type2

    if format_fn is not None:
        final_error = rr.error
        final_passed = rr.passed
        for line in format_fn(log_label, final_passed, final_error, rr.elapsed_seconds, resp,
                              infra=(error_type == "infrastructure")):
            logger.info(line)

    return rr, resp


def _compute_3way_metadata(
    role_results: dict[str, RoleResult],
    arch_results: dict[str, dict[str, Any]],
    prompt: str,
    suite: str,
    passed_direct: bool,
    passed_repl: bool,
    self_role: str,
    self_direct_mode: str,
    self_repl_mode: str,
    arch_mode: str,
) -> dict[str, Any]:
    """Compute metadata dict (tool value, cost metrics, architect eval)."""
    metadata = compute_tool_value(passed_direct, passed_repl)
    metadata["suite"] = suite
    metadata["cache_disabled"] = True

    # Determine best architect (prefer generation_ms over elapsed)
    best_arch = None
    for ar, res in arch_results.items():
        if res["passed"] is True:
            if best_arch is None:
                best_arch = ar
            else:
                cur_time = res.get("generation_ms") or (res["elapsed_seconds"] * 1000)
                best_time = arch_results[best_arch].get("generation_ms") or (
                    arch_results[best_arch]["elapsed_seconds"] * 1000
                )
                if cur_time < best_time:
                    best_arch = ar
    if best_arch is None:
        for ar, res in arch_results.items():
            if res["passed"] is not None:
                best_arch = ar
                break

    metadata["architect_eval"] = {
        "general": arch_results.get("architect_general"),
        "coding": arch_results.get("architect_coding"),
        "best": best_arch,
        "heuristic_would_pick": "architect_coding" if _is_coding_task(prompt) else "architect_general",
    }
    metadata["architect_role"] = best_arch or ""

    direct_key = f"{self_role}:{self_direct_mode}"
    repl_key = f"{self_role}:{self_repl_mode}"
    metadata["cost_metrics"] = {
        ACTION_SELF_DIRECT: {
            "elapsed_seconds": role_results[direct_key].elapsed_seconds,
            "tokens_generated": role_results[direct_key].tokens_generated,
            "tokens_generated_estimate": role_results[direct_key].tokens_generated_estimate,
            "predicted_tps": role_results[direct_key].predicted_tps,
            "generation_ms": role_results[direct_key].generation_ms,
            "backend_task_id": role_results[direct_key].backend_task_id,
            "slot_progress_source": role_results[direct_key].slot_progress_source,
        },
        ACTION_SELF_REPL: {
            "elapsed_seconds": role_results[repl_key].elapsed_seconds,
            "tokens_generated": role_results[repl_key].tokens_generated,
            "tokens_generated_estimate": role_results[repl_key].tokens_generated_estimate,
            "predicted_tps": role_results[repl_key].predicted_tps,
            "generation_ms": role_results[repl_key].generation_ms,
            "tools_used": role_results[repl_key].tools_used,
            "backend_task_id": role_results[repl_key].backend_task_id,
            "slot_progress_source": role_results[repl_key].slot_progress_source,
        },
    }
    if best_arch:
        arch_key = f"{best_arch}:{arch_mode}"
        metadata["cost_metrics"][ACTION_ARCHITECT] = {
            "elapsed_seconds": arch_results[best_arch]["elapsed_seconds"],
            "tokens_generated": role_results[arch_key].tokens_generated,
            "tokens_generated_estimate": role_results[arch_key].tokens_generated_estimate,
            "predicted_tps": role_results[arch_key].predicted_tps,
            "generation_ms": role_results[arch_key].generation_ms,
            "role_history": role_results[arch_key].role_history,
            "backend_task_id": role_results[arch_key].backend_task_id,
            "slot_progress_source": role_results[arch_key].slot_progress_source,
        }

    infra_flags = [
        rr.error_type == "infrastructure"
        for rr in role_results.values()
        if rr is not None
    ]
    metadata["all_infra"] = bool(infra_flags) and all(infra_flags)

    return metadata


def evaluate_question_3way(
    prompt_info: dict,
    url: str,
    timeout: int,
    client: "httpx.Client",
    dry_run: bool = False,
) -> tuple[dict[str, RoleResult], dict[str, float], dict[str, Any]]:
    """Evaluate one question across the 3-way routing matrix.

    Test configurations:
    1. SELF:direct - Frontdoor without tools (direct mode)
    2. SELF:repl - Frontdoor with tools, delegation disabled
    3. ARCHITECT - Architect with full delegation freedom

    WORKER is scored indirectly via delegation chains.
    """
    prompt = prompt_info["prompt"]
    expected = prompt_info.get("expected", "")
    scoring_method = prompt_info.get("scoring_method", "exact_match")
    scoring_config = prompt_info.get("scoring_config", {})
    suite = prompt_info["suite"]
    image_path = prompt_info.get("image_path", "")
    is_vl = bool(image_path)

    role_results: dict[str, RoleResult] = {}

    from eval_log_format import (
        format_self_direct, format_self_repl, format_architect_result,
        format_reward_skip, format_all_infra_skip,
    )

    # ── VL-aware role mapping ──
    if is_vl:
        self_role = "worker_vision"
        self_direct_mode = "direct"
        # React has been subsumed by repl; keep a single SELF:repl action.
        self_repl_mode = "repl"
        arch_role = "vision_escalation"
    else:
        self_role = "frontdoor"
        self_direct_mode = "direct"
        self_repl_mode = "repl"
        arch_role = None

    # ── Configuration 1: SELF:direct ──
    timeout_direct = _adaptive_timeout_s(
        role=self_role,
        mode=self_direct_mode,
        prompt=prompt,
        is_vl=is_vl,
        hard_timeout_s=timeout,
    )
    rr_direct, _ = _eval_single_config(
        prompt, expected, scoring_method, scoring_config,
        role=self_role, mode=self_direct_mode,
        url=url, timeout=timeout_direct, client=client,
        allow_delegation=False, image_path=image_path,
        log_label=ACTION_SELF_DIRECT, format_fn=format_self_direct,
    )
    role_results[f"{self_role}:{self_direct_mode}"] = rr_direct

    # ── Cleanup: ensure port is idle before next strategy ──
    if rr_direct.error_type == "infrastructure" or rr_direct.error:
        direct_port = ROLE_PORT.get(self_role, 0)
        if direct_port in HEAVY_PORTS:
            _force_erase_and_verify(direct_port)

    # ── Configuration 2: SELF:repl ──
    timeout_repl = _adaptive_timeout_s(
        role=self_role,
        mode=self_repl_mode,
        prompt=prompt,
        is_vl=is_vl,
        hard_timeout_s=timeout,
    )
    # Hard prompts often require materially longer REPL orchestration than direct.
    # Use direct elapsed as a per-question lower bound to avoid premature INFRA.
    if rr_direct.error_type != "infrastructure":
        timeout_repl = _bump_timeout_from_observed(
            current_s=timeout_repl,
            observed_s=rr_direct.elapsed_seconds,
            factor=2.2 if not is_vl else 1.8,
            slack_s=30,
            hard_timeout_s=timeout,
            role_cap_s=300 if not is_vl else 260,
        )
    rr_repl, _ = _eval_single_config(
        prompt, expected, scoring_method, scoring_config,
        role=self_role, mode=self_repl_mode,
        url=url, timeout=timeout_repl, client=client,
        allow_delegation=False, image_path=image_path,
        log_label=ACTION_SELF_REPL, format_fn=format_self_repl,
    )
    role_results[f"{self_role}:{self_repl_mode}"] = rr_repl

    # ── Configuration 3: ARCHITECT (dual evaluation) ──
    if is_vl:
        arch_roles_to_eval = [arch_role]
    else:
        arch_roles_to_eval = ["architect_general", "architect_coding"]

    arch_results: dict[str, dict[str, Any]] = {}
    arch_mode = "direct" if is_vl else "delegated"

    for ar in arch_roles_to_eval:
        timeout_arch = _adaptive_timeout_s(
            role=ar,
            mode=arch_mode,
            prompt=prompt,
            is_vl=is_vl,
            hard_timeout_s=timeout,
        )
        observed_base = rr_direct.elapsed_seconds
        if rr_repl.error_type != "infrastructure" and rr_repl.elapsed_seconds > observed_base:
            observed_base = rr_repl.elapsed_seconds
        timeout_arch = _bump_timeout_from_observed(
            current_s=timeout_arch,
            observed_s=observed_base,
            factor=4.0 if not is_vl else 2.5,
            slack_s=60 if not is_vl else 40,
            hard_timeout_s=timeout,
            role_cap_s=max(540, int(timeout)) if not is_vl else 360,
        )
        rr_arch, resp_arch = _eval_single_config(
            prompt, expected, scoring_method, scoring_config,
            role=ar, mode=arch_mode,
            url=url, timeout=timeout_arch, client=client,
            allow_delegation=not is_vl, image_path=image_path,
            log_label=ACTION_ARCHITECT,
            format_fn=lambda label, passed, error, elapsed, resp, infra=False: format_architect_result(label, passed, error, elapsed, resp),
        )
        role_results[f"{ar}:{arch_mode}"] = rr_arch

        # Determine passed for architect (None for infra errors)
        passed_arch = None if rr_arch.error_type == "infrastructure" else rr_arch.passed
        arch_results[ar] = {
            "passed": passed_arch,
            "elapsed_seconds": rr_arch.elapsed_seconds,
            "tokens_generated": rr_arch.tokens_generated,
            "predicted_tps": rr_arch.predicted_tps,
            "generation_ms": rr_arch.generation_ms,
            "tools_used": rr_arch.tools_used,
            "tools_called": rr_arch.tools_called,
            "role_history": rr_arch.role_history,
            "error": rr_arch.error,
            "error_type": rr_arch.error_type,
        }

    # ── Compute 3-way rewards (binary for faithful P(success)) ──
    passed_direct = rr_direct.passed
    passed_repl = rr_repl.passed
    error_type_direct = rr_direct.error_type
    error_type_repl = rr_repl.error_type

    rewards: dict[str, float] = {}

    if error_type_direct == "infrastructure":
        for line in format_reward_skip(ACTION_SELF_DIRECT):
            logger.info(line)
    else:
        rewards[ACTION_SELF_DIRECT] = success_reward(passed_direct)

    if error_type_repl == "infrastructure":
        for line in format_reward_skip(ACTION_SELF_REPL):
            logger.info(line)
    else:
        rewards[ACTION_SELF_REPL] = success_reward(passed_repl)

    valid_results = {k: v for k, v in arch_results.items() if v["passed"] is not None}
    if valid_results:
        passed_arch_any = any(v["passed"] for v in valid_results.values())
        rewards[ACTION_ARCHITECT] = success_reward(passed_arch_any)
    else:
        for line in format_all_infra_skip(ACTION_ARCHITECT):
            logger.info(line)

    worker_rewards = score_delegation_chain(role_results)
    rewards.update(worker_rewards)

    # ── Metadata ──
    metadata = _compute_3way_metadata(
        role_results, arch_results, prompt, suite,
        passed_direct, passed_repl,
        self_role, self_direct_mode, self_repl_mode, arch_mode,
    )

    for action, reward in sorted(rewards.items()):
        logger.info(f"    reward[{action}] = {reward:.1f}")

    return role_results, rewards, metadata


# Embedder server ports for precomputing embeddings
EMBEDDER_PORTS = [8090, 8091, 8092, 8093, 8094, 8095]


def _precompute_embedding(
    task_description: str,
    client: "httpx.Client",
) -> list[float] | None:
    """Precompute embedding for task_description using embedder servers.

    Tries each embedder port until one succeeds. Returns None on failure
    (caller will fall back to letting the API compute the embedding).

    Args:
        task_description: Text to embed (will be truncated to 200 chars).
        client: HTTP client for requests.

    Returns:
        List of float embeddings, or None on failure.
    """
    # Build the text the same way q_scorer does
    text = f"type:chat | objective:{task_description[:200]}"

    for port in EMBEDDER_PORTS:
        try:
            resp = client.post(
                f"http://127.0.0.1:{port}/embedding",
                json={"content": text},
                timeout=5.0,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            # Parse llama-server response format
            if "embedding" in data:
                embedding_data = data["embedding"]
                if isinstance(embedding_data[0], list):
                    return embedding_data[0]
                return embedding_data
            elif "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
        except Exception:
            continue

    logger.debug("All embedder servers failed, will let API compute embedding")
    return None


# Background executor for async reward injection (fire-and-forget)
_reward_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_reward_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the background executor for reward injection."""
    global _reward_executor
    if _reward_executor is None:
        _reward_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="reward_inject"
        )
    return _reward_executor


def _inject_single_reward(
    url: str,
    payload: dict[str, Any],
    action_key: str,
) -> bool:
    """Inject a single reward (runs in background thread)."""
    import httpx
    try:
        with httpx.Client() as client:
            resp = client.post(f"{url}/chat/reward", json=payload, timeout=30)
            if resp.status_code == 200:
                return True
            else:
                logger.debug(f"Reward injection failed for {action_key}: HTTP {resp.status_code}")
                return False
    except Exception as e:
        logger.debug(f"Reward injection error for {action_key}: {e}")
        return False


def _inject_3way_rewards_http(
    prompt: str,
    suite: str,
    question_id: str,
    rewards: dict[str, float],
    metadata: dict[str, Any],
    url: str,
    client: "httpx.Client",
) -> int:
    """Inject 3-way rewards via HTTP API (async, non-blocking).

    Q-values receive binary rewards for faithful probability estimation.
    Cost metrics are stored in context for later Optuna threshold optimization.

    Precomputes the embedding once and reuses it for all reward injections.
    Submissions are fire-and-forget to avoid blocking the eval loop.

    Returns number of rewards submitted (not necessarily injected yet).
    """
    cost_metrics = metadata.get("cost_metrics", {})

    # Precompute embedding once for all reward injections (same task_description)
    embedding = _precompute_embedding(prompt[:200], client)

    executor = _get_reward_executor()
    submitted = 0

    for action_key, reward in rewards.items():
        # Build context with cost metrics for this specific action
        action_cost = cost_metrics.get(action_key, {})
        tokens_generated = int(action_cost.get("tokens_generated", 0) or 0)
        tokens_estimate = int(action_cost.get("tokens_generated_estimate", 0) or 0)

        context = {
            "task_type": suite,
            "source": "3way_eval",
            "question_id": question_id,
            "action_type": "routing",
            # Tool value metadata (flat scalars)
            "tools_helped": metadata.get("tools_helped", False),
            "tools_neutral": metadata.get("tools_neutral", False),
            "tools_hurt": metadata.get("tools_hurt", False),
            "tool_advantage": metadata.get("tool_advantage", 0),
            # Cost metrics for this action (for Optuna later)
            "elapsed_seconds": action_cost.get("elapsed_seconds", 0.0),
            "tokens_generated": tokens_generated,
            "tokens_generated_estimate": tokens_estimate,
            "tokens_generated_effective": (
                tokens_generated if tokens_generated > 0 else tokens_estimate
            ),
            "predicted_tps": action_cost.get("predicted_tps", 0.0),
            "generation_ms": action_cost.get("generation_ms", 0.0),
            "tools_used": action_cost.get("tools_used", 0),
            "backend_task_id": action_cost.get("backend_task_id", 0),
            "slot_progress_source": action_cost.get("slot_progress_source", ""),
        }

        payload = {
            "task_description": prompt[:200],
            "action": action_key,
            "reward": reward,
            "context": context,
        }
        # Include precomputed embedding if available
        if embedding is not None:
            payload["embedding"] = embedding

        # Fire-and-forget: submit to background executor
        executor.submit(_inject_single_reward, url, payload, action_key)
        submitted += 1

    return submitted


# ── Combo building ───────────────────────────────────────────────────


def _build_role_mode_combos(
    roles: list[str],
    modes: list[str],
) -> list[tuple[str, str]]:
    """Build (role, mode) combinations with two invariants:

    1. MODE-FIRST: Cycle through modes before roles, so consecutive calls
       hit different backend servers (natural cooldown for each server).
    2. HEAVY SEPARATION: Heavy model combos (architects, ingest) are never
       adjacent. Light combos are interleaved between them so light work
       runs while heavy servers cool down.

    The idle-wait in evaluate_question() enforces that heavy models are
    actually idle before any request, but good ordering reduces idle-wait
    time by doing useful light work in the gaps.
    """
    all_modes = list(modes)
    for m in sorted(ARCHITECT_MODES):
        if m not in all_modes:
            all_modes.append(m)

    light: list[tuple[str, str]] = []
    heavy: list[tuple[str, str]] = []

    for mode in all_modes:
        for role in roles:
            port = ROLE_PORT.get(role, 0)
            is_heavy = port in HEAVY_PORTS
            if role in ARCHITECT_ROLES:
                if mode in ARCHITECT_MODES:
                    heavy.append((role, mode))
            elif role in VISION_ROLES:
                if mode in VISION_MODES.get(role, {"direct"}):
                    (heavy if is_heavy else light).append((role, mode))
            else:
                if mode in modes:
                    (heavy if is_heavy else light).append((role, mode))

    # Interleave: spread heavy combos evenly across the light sequence.
    # With N light and M heavy: place one heavy after every N//M light combos.
    if not heavy:
        return light
    if not light:
        return heavy

    result: list[tuple[str, str]] = []
    gap = max(1, len(light) // len(heavy))
    heavy_iter = iter(heavy)
    next_heavy = next(heavy_iter, None)

    for i, combo in enumerate(light):
        result.append(combo)
        if next_heavy is not None and (i + 1) % gap == 0:
            result.append(next_heavy)
            next_heavy = next(heavy_iter, None)

    # Append remaining heavy combos at the end
    if next_heavy is not None:
        result.append(next_heavy)
    for h in heavy_iter:
        result.append(h)

    return result


def _deduplicate_roles(
    roles: list[str],
    server_urls: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Remove roles that share the same backend URL."""
    from src.config import get_config as _get_cfg
    urls = server_urls or _get_cfg().server_urls.as_dict()
    seen: dict[str, str] = {}
    unique: list[str] = []
    aliases: dict[str, str] = {}

    for role in roles:
        url = urls.get(role, "")
        if url and url in seen:
            aliases[role] = seen[url]
        else:
            if url:
                seen[url] = role
            unique.append(role)

    return unique, aliases


def _modes_for_role(role: str, modes: list[str]) -> list[str]:
    """Return the effective mode list for a role."""
    if role in ARCHITECT_ROLES:
        return sorted(ARCHITECT_MODES)
    if role in VISION_ROLES:
        return sorted(VISION_MODES.get(role, {"direct"}))
    return list(modes)


# ── Main evaluation loop ──────────────────────────────────────────────


def evaluate_question(
    prompt_info: dict,
    combos: list[tuple[str, str]],
    alias_map: dict[str, str],
    modes: list[str],
    url: str,
    timeout: int,
    client: "httpx.Client",
    skip_cache: bool = False,
    cooldown: float = 0.0,
    dry_run: bool = False,
    escalation_chains: bool = False,
) -> ComparativeResult | None:
    """Evaluate one question across all role×mode combos.

    Returns ComparativeResult or None if shutdown requested.
    """
    suite = prompt_info["suite"]
    qid = prompt_info["id"]
    prompt = prompt_info["prompt"]
    expected = prompt_info.get("expected", "")
    scoring_method = prompt_info.get("scoring_method", "exact_match")
    scoring_config = prompt_info.get("scoring_config", {})
    image_path = prompt_info.get("image_path", "")
    dataset_source = prompt_info.get("dataset_source", "yaml")

    # Smart combo filtering: VL → vision + frontdoor; text → non-vision
    is_vl = bool(image_path)
    if is_vl:
        active_combos = [
            (r, m) for r, m in combos
            if r in VISION_ROLES or r == "frontdoor"
        ]
    else:
        active_combos = [
            (r, m) for r, m in combos
            if r not in VISION_ROLES
        ]

    role_results: dict[str, RoleResult] = {}
    cache_prompt_val = False if skip_cache else None

    SLOW_ROLES = {"architect_general", "architect_coding"}
    SLOW_ROLE_TIMEOUT = max(timeout, 300)

    for combo_idx, (role, mode) in enumerate(active_combos):
        if state.shutdown:
            return None

        # Before hitting a heavy model, confirm all heavy ports are idle.
        target_port = ROLE_PORT.get(role, 0)
        if target_port in HEAVY_PORTS:
            _wait_for_heavy_models_idle()

        key = f"{role}:{mode}"
        if target_port in HEAVY_PORTS:
            logger.info(f"  → {key} (heavy model, expect 30-120s)...")
        role_timeout = SLOW_ROLE_TIMEOUT if role in SLOW_ROLES else timeout
        q_start = time.perf_counter()
        response = call_orchestrator_forced(
            prompt, role, mode, url, role_timeout,
            image_path=image_path, cache_prompt=cache_prompt_val,
            client=client,
        )
        q_elapsed = time.perf_counter() - q_start

        if cooldown > 0 and combo_idx < len(active_combos) - 1:
            time.sleep(cooldown)

        answer = response.get("answer", "")
        error = response.get("error")
        tokens_generated = response.get("tokens_generated", 0)
        tools_used = response.get("tools_used", 0)
        tools_called = response.get("tools_called", [])
        routed_to = response.get("routed_to", "")
        role_history = response.get("role_history", [])
        routing_strategy = response.get("routing_strategy", "")
        turns = response.get("turns", 0)
        tokens_used = response.get("tokens_used", 0)
        formalization_applied = response.get("formalization_applied", False)
        cache_stats = response.get("cache_stats")
        predicted_tps = response.get("predicted_tps", 0.0)
        generation_ms = response.get("generation_ms", 0.0)
        prompt_eval_ms = response.get("prompt_eval_ms", 0.0)
        http_overhead_ms = response.get("http_overhead_ms", 0.0)

        if error:
            passed = False
            # After a timeout/error on a heavy port, erase its slots so the
            # server isn't still grinding when the next combo arrives.
            if target_port in HEAVY_PORTS and tokens_generated == 0:
                _erase_slots(target_port)
        else:
            passed = score_answer_deterministic(answer, expected, scoring_method, scoring_config)

        role_results[key] = RoleResult(
            role=role,
            mode=mode,
            answer=answer or "",
            passed=passed,
            elapsed_seconds=q_elapsed,
            error=error,
            tokens_generated=tokens_generated,
            tools_used=tools_used,
            tools_called=tools_called,
            routed_to=routed_to,
            role_history=role_history,
            routing_strategy=routing_strategy,
            turns=turns,
            tokens_used=tokens_used,
            formalization_applied=formalization_applied,
            cache_stats=cache_stats,
            predicted_tps=predicted_tps,
            generation_ms=generation_ms,
            prompt_eval_ms=prompt_eval_ms,
            http_overhead_ms=http_overhead_ms,
        )

        status = "PASS" if passed else ("ERROR" if error else "FAIL")
        display_tps = predicted_tps if predicted_tps > 0 else 0

        # Main line: role:mode → status (time, speed, tokens)
        parts = [f"  {key:30s} → {status} ({q_elapsed:.1f}s"]
        if display_tps > 0:
            parts.append(f", {display_tps:.1f} t/s")
        parts.append(f", {tokens_generated} tok)")
        logger.info("".join(parts))

        # Detail lines: better formatting for readability
        indent = "  " + " " * 30 + "   "

        # Line 2: Chain (if delegated)
        if role_history and len(role_history) > 1:
            logger.info(f"{indent}chain: {' → '.join(role_history)}")

        # Line 3: Tools (show all, dedupe consecutive repeats)
        if tools_used > 0:
            if tools_called:
                # Dedupe consecutive repeated tools
                deduped = []
                for t in tools_called:
                    if not deduped or deduped[-1] != t:
                        deduped.append(t)
                tool_str = ", ".join(deduped)
            else:
                tool_str = "?"
            logger.info(f"{indent}tools({tools_used}): {tool_str}")

        # Line 4: Timing breakdown (in seconds for readability)
        timing_parts = []
        if generation_ms > 0:
            timing_parts.append(f"gen={generation_ms/1000:.1f}s")
        if prompt_eval_ms > 0:
            timing_parts.append(f"prompt={prompt_eval_ms/1000:.1f}s")
        if formalization_applied:
            timing_parts.append("formalized")
        if timing_parts:
            logger.info(f"{indent}{', '.join(timing_parts)}")

    # Compute comparative rewards (baseline is frontdoor:direct)
    rewards = compute_comparative_rewards(role_results, baseline_key="frontdoor:direct")

    # Clone rewards and results to aliased (deduplicated) roles
    for alias, canonical in alias_map.items():
        for mode in _modes_for_role(alias, modes):
            canonical_key = f"{canonical}:{mode}"
            alias_key = f"{alias}:{mode}"
            if canonical_key in rewards:
                rewards[alias_key] = rewards[canonical_key]
            if canonical_key in role_results:
                role_results[alias_key] = role_results[canonical_key]

    # Inject rewards immediately (per-question, not batched)
    rewards_injected = 0
    if not dry_run:
        # Use HTTP API (works whether running in-process or externally)
        comp_for_inject = ComparativeResult(
            suite=suite, question_id=qid, prompt=prompt[:200],
            expected=expected[:200], rewards=rewards,
        )
        rewards_injected = _inject_rewards_http(comp_for_inject, url, client)

    # Escalation chains: detect cheap-fail → expensive-pass patterns
    escalation_data: list[dict[str, Any]] = []
    if escalation_chains and not dry_run:
        escalation_data = detect_escalation_chains(role_results)
        if escalation_data:
            comp_for_esc = ComparativeResult(
                suite=suite, question_id=qid, prompt=prompt[:200],
                expected=expected[:200], rewards=rewards,
            )
            esc_injected = _inject_escalation_chains_http(
                comp_for_esc, escalation_data, url, client,
            )
            rewards_injected += esc_injected
            for chain in escalation_data:
                logger.info(
                    f"    escalation: {chain['from_role']}:{chain['from_mode']} → "
                    f"{chain['to_role']}:{chain['to_mode']} "
                    f"reward={chain['reward']:+.2f}"
                )

    # Log rewards
    for key, reward in sorted(rewards.items()):
        alias_tag = ""
        role_part = key.split(":")[0]
        if role_part in alias_map:
            alias_tag = f" (={alias_map[role_part]})"
        logger.info(f"    reward[{key}] = {reward:+.2f}{alias_tag}")

    return ComparativeResult(
        suite=suite,
        question_id=qid,
        prompt=prompt[:200],
        expected=expected[:200],
        dataset_source=dataset_source,
        prompt_hash=_prompt_hash(prompt),
        timestamp=datetime.now(timezone.utc).isoformat(),
        role_results=role_results,
        rewards=rewards,
        rewards_injected=rewards_injected,
    )


def run_batch(
    suites: list[str],
    roles: list[str],
    modes: list[str],
    sample_per_suite: int,
    seed: int,
    url: str,
    timeout: int,
    session_id: str,
    dry_run: bool = False,
    skip_cache: bool = False,
    cooldown: float = 0.0,
    no_dedup: bool = False,
    escalation_chains: bool = False,
    use_pool: bool = True,
) -> list[ComparativeResult]:
    """Run one evaluation batch: sample, evaluate per-question, checkpoint.

    DEPRECATED: Use run_batch_3way() with --3way flag for new seeding.
    This legacy mode uses cost-weighted comparative rewards which conflate
    P(success) with cost. The 3-way mode uses binary rewards for faithful
    probability estimation.
    """
    import warnings
    warnings.warn(
        "Legacy comparative seeding is deprecated. Use --3way for binary rewards "
        "and faithful P(success) estimation.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Deduplicate roles
    alias_map: dict[str, str] = {}
    if not no_dedup:
        unique_roles, alias_map = _deduplicate_roles(roles)
        if alias_map:
            for alias, canonical in sorted(alias_map.items()):
                from src.config import get_config as _get_cfg2
                canon_url = _get_cfg2().server_urls.as_dict().get(canonical, "?")
                logger.info(f"Dedup: {alias} → {canonical} (same backend {canon_url})")
        roles_to_test = unique_roles
    else:
        roles_to_test = list(roles)

    combos = _build_role_mode_combos(roles_to_test, modes)
    combo_keys = [f"{r}:{m}" for r, m in combos]

    # Health check (continuous loop pre-checks this, but one-shot mode needs it)
    if not _check_server_health(url):
        raise HealthCheckError(f"API unreachable: {url}")

    # Load existing checkpoint + seen set
    completed = load_checkpoint(session_id)
    completed_ids = {r.question_id for r in completed}
    seen = load_seen_questions()
    logger.info(f"Checkpoint: {len(completed)} completed, {len(seen)} total seen")

    # Sample unseen questions (debug mode backfills with seen when exhausted)
    questions = sample_unseen_questions(
        suites, sample_per_suite, seen, seed,
        use_pool=use_pool, allow_reseen=debugger is not None,
    )
    questions = [q for q in questions if q["id"] not in completed_ids]

    if not questions:
        logger.info("No unseen questions available. Try a different seed or suite.")
        return completed

    vl_count = sum(1 for p in questions if p.get("image_path"))
    text_count = len(questions) - vl_count

    logger.info(f"\n{'='*60}")
    logger.info(f"Session: {session_id}")
    logger.info(f"Batch: {len(questions)} questions ({text_count} text, {vl_count} VL)")
    logger.info(f"Combos: {len(combos)} ({', '.join(combo_keys)})")
    heavy_count = sum(1 for r, m in combos if ROLE_PORT.get(r, 0) in HEAVY_PORTS)
    if heavy_count:
        logger.info(f"  Heavy combos per question: {heavy_count} (30-120s each)")
    logger.info(f"Seed: {seed}  Rewards: {'off' if dry_run else 'on'}")
    logger.info(f"{'='*60}\n")

    SLOW_ROLE_TIMEOUT = max(timeout, 300)
    import httpx as _httpx
    _client = _httpx.Client(timeout=SLOW_ROLE_TIMEOUT)

    new_results: list[ComparativeResult] = []
    consecutive_zero_success = 0

    try:
        for i, prompt_info in enumerate(questions):
            if state.shutdown:
                logger.info(f"\n[Stopped after {i} questions]")
                break

            qid = prompt_info["id"]
            suite = prompt_info["suite"]
            is_vl = bool(prompt_info.get("image_path"))
            logger.info(f"[{i+1}/{len(questions)}] {suite}/{qid} ({'VL' if is_vl else 'text'})")

            result = evaluate_question(
                prompt_info, combos, alias_map, modes,
                url, timeout, _client,
                skip_cache=skip_cache, cooldown=cooldown, dry_run=dry_run,
                escalation_chains=escalation_chains,
            )

            if result is None:
                break  # Shutdown

            # Checkpoint immediately
            append_checkpoint(session_id, result)
            if result.rewards_injected > 0:
                record_seen(result.question_id, result.suite, session_id)
            new_results.append(result)

            # Track consecutive failures for abort
            any_success = any(rr.error is None for rr in result.role_results.values())
            if any_success:
                consecutive_zero_success = 0
            else:
                consecutive_zero_success += 1
                if consecutive_zero_success >= 3:
                    logger.error(
                        f"Aborting: {consecutive_zero_success} consecutive questions "
                        f"with zero successful combos — server appears dead"
                    )
                    break
    finally:
        _client.close()

    all_results = completed + new_results
    return all_results


# ── Stats / summary ───────────────────────────────────────────────────


def print_batch_summary(
    results: list[ComparativeResult],
    roles: list[str],
    modes: list[str],
    alias_map: dict[str, str] | None = None,
) -> None:
    """Print summary of results."""
    alias_map = alias_map or {}
    combos = _build_role_mode_combos(roles, modes)
    combo_keys = [f"{r}:{m}" for r, m in combos]

    key_stats: dict[str, dict[str, Any]] = {
        k: {"pass": 0, "fail": 0, "error": 0,
            "total_tokens": 0, "total_elapsed": 0.0,
            "samples": 0, "predicted_tps_sum": 0.0, "predicted_tps_count": 0,
            "total_reward": 0.0}
        for k in combo_keys
    }

    for comp in results:
        for key, rr in comp.role_results.items():
            if key not in key_stats:
                continue
            key_stats[key]["samples"] += 1
            key_stats[key]["total_tokens"] += rr.tokens_generated
            key_stats[key]["total_elapsed"] += rr.elapsed_seconds
            if rr.predicted_tps > 0:
                key_stats[key]["predicted_tps_sum"] += rr.predicted_tps
                key_stats[key]["predicted_tps_count"] += 1
            if rr.error:
                key_stats[key]["error"] += 1
            elif rr.passed:
                key_stats[key]["pass"] += 1
            else:
                key_stats[key]["fail"] += 1
        for key, reward in comp.rewards.items():
            if key in key_stats:
                key_stats[key]["total_reward"] += reward

    print(f"\n{'='*100}")
    print("COMPARATIVE EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"Questions: {len(results)}")
    if alias_map:
        dedup_strs = [f"{a} → {c}" for a, c in sorted(alias_map.items())]
        print(f"Deduplicated: {', '.join(dedup_strs)}")

    print(f"\n{'Role:Mode':30s} {'Pass':>5s} {'Fail':>5s} {'Err':>4s} {'Acc%':>6s} {'Avg t/s':>8s} {'Reward':>8s}")
    print("-" * 75)
    for key in combo_keys:
        s = key_stats[key]
        total = s["pass"] + s["fail"]
        acc = s["pass"] / total * 100 if total > 0 else 0
        if s["predicted_tps_count"] > 0:
            avg_tps = s["predicted_tps_sum"] / s["predicted_tps_count"]
        else:
            avg_tps = s["total_tokens"] / s["total_elapsed"] if s["total_elapsed"] > 0 else 0
        role_part = key.split(":")[0]
        alias_tag = f" (={alias_map[role_part]})" if role_part in alias_map else ""
        print(
            f"{key:30s} {s['pass']:5d} {s['fail']:5d} {s['error']:4d} "
            f"{acc:5.1f}% {avg_tps:7.1f} {s['total_reward']:+7.1f}"
            f"{alias_tag}"
        )

    # Total rewards injected
    total_injected = sum(r.rewards_injected for r in results)
    print(f"\nRewards injected: {total_injected}")


def print_stats():
    """Aggregate stats across all seeding sessions."""
    if not EVAL_DIR.exists():
        print("No evaluation data found.")
        return

    sessions: dict[str, list[ComparativeResult]] = {}
    for path in sorted(EVAL_DIR.glob("seeding_*.jsonl")):
        sid = path.stem
        results = load_checkpoint(sid)
        if results:
            sessions[sid] = results

    if not sessions:
        print("No seeding sessions found.")
        return

    print(f"\n{'='*60}")
    print("ALL SEEDING SESSIONS")
    print(f"{'='*60}")

    total_questions = 0
    all_combo_stats: dict[str, dict[str, int]] = {}

    for sid, results in sessions.items():
        total_questions += len(results)
        ts = results[0].timestamp[:10] if results and results[0].timestamp else "?"
        print(f"  {sid:45s} {len(results):4d} questions  {ts}")

        for comp in results:
            for key, rr in comp.role_results.items():
                if key not in all_combo_stats:
                    all_combo_stats[key] = {"pass": 0, "fail": 0, "error": 0, "total": 0}
                all_combo_stats[key]["total"] += 1
                if rr.error:
                    all_combo_stats[key]["error"] += 1
                elif rr.passed:
                    all_combo_stats[key]["pass"] += 1
                else:
                    all_combo_stats[key]["fail"] += 1

    print(f"\nTotal questions: {total_questions}")
    print(f"Sessions: {len(sessions)}")

    seen = load_seen_questions()
    print(f"Unique questions seen: {len(seen)}")

    if all_combo_stats:
        print(f"\nAggregate accuracy by role×mode:")
        print(f"  {'Role:Mode':30s} {'Pass':>5s} {'Fail':>5s} {'Err':>4s} {'Acc%':>6s} {'N':>5s} {'≥3?':>4s}")
        print("  " + "-" * 60)
        for key in sorted(all_combo_stats.keys()):
            s = all_combo_stats[key]
            total = s["pass"] + s["fail"]
            acc = s["pass"] / total * 100 if total > 0 else 0
            # ≥3 observations = MemRL confidence threshold met
            confident = "YES" if s["total"] >= 3 else "no"
            print(
                f"  {key:30s} {s['pass']:5d} {s['fail']:5d} {s['error']:4d} "
                f"{acc:5.1f}% {s['total']:5d} {confident:>4s}"
            )

    # Coverage: combos with ≥3 observations
    covered = sum(1 for s in all_combo_stats.values() if s["total"] >= 3)
    total_combos = len(all_combo_stats)
    print(f"\nMemRL coverage: {covered}/{total_combos} combos have ≥3 observations")


# ── Phase 4: 3-Way Batch Runner ───────────────────────────────────────


@dataclass
class ThreeWayResult:
    """Result from 3-way routing evaluation."""

    suite: str
    question_id: str
    prompt: str
    expected: str
    timestamp: str = ""
    role_results: dict[str, RoleResult] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    rewards_injected: int = 0


def run_batch_3way(
    suites: list[str],
    sample_per_suite: int,
    seed: int,
    url: str,
    timeout: int,
    session_id: str,
    dry_run: bool = False,
    on_progress: "Callable[[int, int, str, str], None] | None" = None,
    use_pool: bool = True,
    debugger: "ClaudeDebugger | None" = None,
) -> list[ThreeWayResult]:
    """Run one 3-way evaluation batch.

    Each question is tested through:
    1. SELF:direct (frontdoor, no tools)
    2. SELF:repl (frontdoor, tools, no delegation)
    3. ARCHITECT (architect with full delegation)

    Binary rewards injected for faithful probability estimation.

    Args:
        on_progress: Optional callback ``(idx, total, suite, qid)`` called
            at the start of each question.  Used by the TUI to update the
            status bar.
        debugger: Optional ClaudeDebugger for pipeline monitoring.
    """
    import httpx as _httpx

    # Health check
    if not _check_server_health(url):
        raise HealthCheckError(f"API unreachable: {url}")

    # Load seen questions
    seen = load_seen_questions()
    logger.info(f"Previously seen questions: {len(seen)}")

    # Sample unseen questions (debug mode backfills with seen when exhausted)
    questions = sample_unseen_questions(
        suites, sample_per_suite, seen, seed,
        use_pool=use_pool, allow_reseen=debugger is not None,
    )
    if not questions:
        logger.info("No unseen questions available.")
        return []

    logger.info(f"\n{'='*60}")
    logger.info(f"3-Way Routing Evaluation: {len(questions)} questions")
    logger.info(f"Session: {session_id}")
    logger.info(f"Actions: {', '.join(THREE_WAY_ACTIONS)}")
    logger.info(f"Seed: {seed}  Rewards: {'off' if dry_run else 'on'}")
    logger.info("Cache control: cache_prompt=False for all 3-way eval calls (fair timing)")
    logger.info(f"{'='*60}\n")

    _client = _httpx.Client(timeout=max(timeout, 300))
    results: list[ThreeWayResult] = []

    try:
        for i, prompt_info in enumerate(questions):
            if state.shutdown:
                logger.info(f"\n[Stopped after {i} questions]")
                break

            qid = prompt_info["id"]
            suite = prompt_info["suite"]
            logger.info(f"[{i+1}/{len(questions)}] {suite}/{qid}")

            if on_progress is not None:
                on_progress(i + 1, len(questions), suite, qid, prompt_info.get("prompt", ""))

            # Run 3-way evaluation
            role_results, rewards, metadata = evaluate_question_3way(
                prompt_info, url, timeout, _client, dry_run,
            )
            if metadata.get("all_infra"):
                logger.warning("  All roles failed due to infra. Attempting recovery...")
                recovered = _attempt_recovery(url)
                if recovered:
                    logger.info("  Recovery successful — continuing")
                else:
                    logger.warning("  Recovery failed — sleeping 30s before continuing")
                    time.sleep(30)

            # Inject rewards
            rewards_injected = 0
            if not dry_run:
                rewards_injected = _inject_3way_rewards_http(
                    prompt_info["prompt"][:200],
                    suite,
                    qid,
                    rewards,
                    metadata,
                    url,
                    _client,
                )
                logger.info(f"  Injected {rewards_injected} rewards")

            result = ThreeWayResult(
                suite=suite,
                question_id=qid,
                prompt=prompt_info["prompt"][:200],
                expected=prompt_info.get("expected", "")[:200],
                timestamp=datetime.now(timezone.utc).isoformat(),
                role_results=role_results,
                rewards=rewards,
                metadata=metadata,
                rewards_injected=rewards_injected,
            )
            results.append(result)

            # Checkpoint immediately (answers, scores, timing for post-hoc analysis)
            _checkpoint_3way(session_id, result)

            # Only mark as seen when rewards were actually injected
            if rewards_injected > 0:
                record_seen(qid, suite, session_id)

            # Pipeline debugger: build diagnostics for each role result
            if debugger is not None:
                try:
                    from src.pipeline_monitor.diagnostic import build_diagnostic, append_diagnostic

                    for config_key, rr in role_results.items():
                        diag = build_diagnostic(
                            question_id=f"{suite}/{qid}",
                            suite=suite,
                            config=config_key,
                            role=rr.role,
                            mode=rr.mode,
                            passed=rr.passed,
                            answer=rr.answer,
                            expected=prompt_info.get("expected", ""),
                            scoring_method=prompt_info.get("scoring_method", "exact_match"),
                            error=rr.error,
                            error_type=rr.error_type,
                            tokens_generated=rr.tokens_generated,
                            elapsed_s=rr.elapsed_seconds,
                            role_history=rr.role_history,
                            delegation_events=rr.delegation_events,
                            tools_used=rr.tools_used,
                            tools_called=rr.tools_called,
                            tap_offset_bytes=rr.tap_offset_bytes,
                            tap_length_bytes=rr.tap_length_bytes,
                            repl_tap_offset_bytes=rr.repl_tap_offset_bytes,
                            repl_tap_length_bytes=rr.repl_tap_length_bytes,
                        )
                        append_diagnostic(diag)
                        debugger.add_diagnostic(diag)
                    debugger.end_question()
                except Exception as e:
                    logger.warning(f"[DEBUG] Debugger error (non-fatal): {e}")

    finally:
        _client.close()
        if debugger is not None:
            debugger.flush()

    return results


def print_3way_summary(results: list[ThreeWayResult]) -> None:
    """Print summary of 3-way evaluation results."""
    if not results:
        print("No results to summarize.")
        return

    # Aggregate stats per action
    action_stats: dict[str, dict[str, Any]] = {
        a: {"pass": 0, "fail": 0, "error": 0, "total_reward": 0.0, "n": 0}
        for a in THREE_WAY_ACTIONS
    }

    tool_value_stats = {"helped": 0, "neutral": 0, "hurt": 0}

    for result in results:
        # Action stats from rewards (binary: 1.0=pass, 0.0=fail)
        for action, reward in result.rewards.items():
            if action in action_stats:
                action_stats[action]["n"] += 1
                action_stats[action]["total_reward"] += reward
                if reward >= 0.5:
                    action_stats[action]["pass"] += 1
                else:
                    action_stats[action]["fail"] += 1

        # Tool value
        meta = result.metadata
        if meta.get("tools_helped"):
            tool_value_stats["helped"] += 1
        elif meta.get("tools_hurt"):
            tool_value_stats["hurt"] += 1
        else:
            tool_value_stats["neutral"] += 1

    print(f"\n{'='*70}")
    print("3-WAY ROUTING EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Questions: {len(results)}")

    print(f"\n{'Action':20s} {'Pass':>5s} {'Fail':>5s} {'Acc%':>7s} {'Q̄':>7s}")
    print("-" * 50)
    for action in THREE_WAY_ACTIONS:
        s = action_stats[action]
        total = s["pass"] + s["fail"]
        acc = s["pass"] / total * 100 if total > 0 else 0
        avg_q = s["total_reward"] / s["n"] if s["n"] > 0 else 0.5
        print(f"{action:20s} {s['pass']:5d} {s['fail']:5d} {acc:6.1f}% {avg_q:6.3f}")

    print(f"\nTool Value (SELF:direct vs SELF:repl):")
    print(f"  Tools helped: {tool_value_stats['helped']}")
    print(f"  Tools neutral: {tool_value_stats['neutral']}")
    print(f"  Tools hurt: {tool_value_stats['hurt']}")

    total_injected = sum(r.rewards_injected for r in results)
    print(f"\nRewards injected: {total_injected}")


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MemRL Episodic Seeding for 3-Way Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (3-way mode - RECOMMENDED):
  # THE command. Run continuously for days:
  %(prog)s --3way --continuous --suites all --sample-size 10 --preflight

  # One-shot batch:
  %(prog)s --3way --suites thinking coder --sample-size 5

  # Dry run (no reward injection):
  %(prog)s --3way --dry-run --suites thinking --sample-size 3

Examples (legacy mode - DEPRECATED):
  # Quick stats from all sessions:
  %(prog)s --stats
""",
    )
    parser.add_argument(
        "--suites", nargs="+", default=DEFAULT_SUITES,
        help=f"Suites to evaluate (default: {' '.join(DEFAULT_SUITES)})",
    )
    parser.add_argument(
        "--roles", nargs="+", default=DEFAULT_ROLES,
        help=f"Roles to compare (default: {' '.join(DEFAULT_ROLES)})",
    )
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        help=f"Execution modes to test (default: {' '.join(DEFAULT_MODES)}). "
        "Architect roles always use direct+delegated.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Questions per suite per batch (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: timestamp)",
    )
    parser.add_argument(
        "--url", default=DEFAULT_ORCHESTRATOR_URL,
        help=f"Orchestrator URL (default: {DEFAULT_ORCHESTRATOR_URL})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Request timeout (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score only, don't inject rewards",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (default: auto-generated). Only for one-shot mode.",
    )
    parser.add_argument(
        "--skip-cache", action="store_true",
        help="Disable KV cache reuse (cache_prompt=False).",
    )
    parser.add_argument(
        "--cooldown", type=float, default=0.0,
        help="Seconds between requests (default: 0). Reduces server memory pressure.",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable URL-based role deduplication.",
    )
    # Continuous mode
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run batches continuously until Ctrl+C. Checkpoints per question.",
    )
    parser.add_argument(
        "--continuous-interval", type=int, default=30,
        help="Seconds between continuous batches (default: 30)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume a specific session ID (e.g. seeding_20260201_143022)",
    )
    # Preflight & stats
    parser.add_argument(
        "--preflight", action="store_true",
        help="Run health checks and smoke test before starting.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show aggregate stats from all sessions, then exit.",
    )
    parser.add_argument(
        "--no-escalation-chains", action="store_true",
        help="Disable escalation chain reward detection (enabled by default). "
        "Escalation chains detect when cheap models fail but expensive models "
        "pass on the same question, and inject escalation rewards into MemRL.",
    )
    # Phase 4: 3-way routing mode
    parser.add_argument(
        "--3way", action="store_true", dest="three_way",
        help="Use 3-way routing evaluation (SELF:direct, SELF:repl, ARCHITECT). "
        "Binary rewards for faithful probability estimation. "
        "Ignores --roles and --modes flags.",
    )
    parser.add_argument(
        "--tui", action="store_true",
        help="Rich split-screen TUI (requires --3way)",
    )
    # Question pool options
    parser.add_argument(
        "--rebuild-pool", action="store_true",
        help="Rebuild the pre-extracted question pool from all adapters, then exit.",
    )
    parser.add_argument(
        "--no-pool", action="store_true",
        help="Bypass the question pool and load from HF adapters directly (slow).",
    )
    # Claude-in-the-loop debugging
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Claude-in-the-loop debugging (requires --3way).",
    )
    parser.add_argument(
        "--debug-batch-size", type=int, default=5,
        help="Diagnostic records per Claude invocation (default: 5).",
    )
    parser.add_argument(
        "--debug-threshold", type=float, default=0.3,
        help="Anomaly score threshold for diagnostics (default: 0.3).",
    )
    parser.add_argument(
        "--debug-auto-commit", action="store_true",
        help="Auto-commit each debug batch (easy git revert).",
    )
    parser.add_argument(
        "--debug-dry-run", action="store_true",
        help="Compute diagnostics and log what would be sent to Claude, but don't invoke.",
    )

    args = parser.parse_args()

    # Rebuild pool — build and exit
    if args.rebuild_pool:
        from question_pool import build_pool
        import time as _time
        t0 = _time.monotonic()
        stats = build_pool()
        elapsed = _time.monotonic() - t0
        total = sum(stats.values())
        print(f"Pool rebuilt in {elapsed:.1f}s: {total} questions across {len(stats)} suites")
        for suite, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {suite:25s} {count:>6,d}")
        return

    # Stats mode — just print and exit
    if args.stats:
        print_stats()
        return

    # Preflight
    if args.preflight:
        if not run_preflight(args.url):
            sys.exit(1)

    # Session ID
    if args.resume:
        session_id = args.resume
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"seeding_{ts}" if not args.three_way else f"3way_{ts}"

    # Default seed
    base_seed = args.seed if args.seed is not None else int(time.time())

    # ── Claude-in-the-loop debugger setup ──
    # --debug-dry-run / --debug-auto-commit / --debug-batch-size / --debug-threshold
    # all imply --debug
    if args.debug_dry_run or args.debug_auto_commit:
        args.debug = True
    _debugger = None
    if args.debug:
        if not args.three_way:
            logger.warning("--debug requires --3way. Ignoring.")
        else:
            from src.pipeline_monitor.claude_debugger import ClaudeDebugger
            _debugger = ClaudeDebugger(
                project_root=PROJECT_ROOT,
                batch_size=args.debug_batch_size,
                anomaly_threshold=args.debug_threshold,
                auto_commit=args.debug_auto_commit,
                dry_run=args.debug_dry_run,
            )
            logger.info(f"[DEBUG] Claude-in-the-loop debugger enabled "
                        f"(batch_size={args.debug_batch_size}, "
                        f"threshold={args.debug_threshold})")

    # ── Phase 4: 3-Way Routing Mode ──
    if args.three_way:
        # TUI setup (--tui flag)
        from contextlib import nullcontext

        if args.tui:
            from seeding_tui import SeedingTUI
            tui = SeedingTUI(session_id=session_id)
            tui_ctx = tui
        else:
            tui = None
            tui_ctx = nullcontext()

        def _on_progress(idx: int, total: int, suite: str, qid: str, question: str = "") -> None:
            if tui is not None:
                tui.update_progress(idx, total, suite, qid, question=question)

        with tui_ctx:
            if args.continuous:
                # Continuous 3-way mode
                batch = 0
                consecutive_failures = 0
                all_results: list[ThreeWayResult] = []
                logger.info(f"Starting continuous 3-way evaluation: session={session_id}")
                logger.info(f"  Ctrl+C to stop gracefully (finishes current question)")

                while not state.shutdown:
                    # Health gate with auto-recovery
                    if not _check_server_health(args.url):
                        consecutive_failures += 1
                        if consecutive_failures > MAX_RECOVERY_ATTEMPTS:
                            logger.error(f"API unrecoverable after {MAX_RECOVERY_ATTEMPTS} attempts.")
                            break
                        backoff = min(30 * (2 ** (consecutive_failures - 1)), 600)
                        logger.warning(f"API down. Attempting recovery...")
                        recovered = _attempt_recovery(args.url)
                        if recovered:
                            logger.info("Recovery successful — resuming")
                            consecutive_failures = 0
                            continue
                        logger.warning(f"Recovery failed — sleeping {backoff}s")
                        for _ in range(backoff):
                            if state.shutdown:
                                break
                            time.sleep(1)
                        continue
                    consecutive_failures = 0

                    batch += 1
                    batch_seed = base_seed + batch
                    logger.info(f"\n[3-Way Batch {batch}, seed={batch_seed}]")

                    try:
                        results = run_batch_3way(
                            suites=args.suites,
                            sample_per_suite=args.sample_size,
                            seed=batch_seed,
                            url=args.url,
                            timeout=args.timeout,
                            session_id=session_id,
                            dry_run=args.dry_run,
                            on_progress=_on_progress,
                            use_pool=not args.no_pool,
                            debugger=_debugger,
                        )
                        all_results.extend(results)

                        if not results:
                            logger.info("No unseen questions. Waiting 60s...")
                            for _ in range(60):
                                if state.shutdown:
                                    break
                                time.sleep(1)

                    except Exception as e:
                        logger.error(f"Batch failed: {e}")
                        time.sleep(10)

            else:
                # One-shot 3-way mode
                logger.info(f"Starting 3-way routing evaluation: session={session_id}")
                results = run_batch_3way(
                    suites=args.suites,
                    sample_per_suite=args.sample_size,
                    seed=base_seed,
                    url=args.url,
                    timeout=args.timeout,
                    session_id=session_id,
                    dry_run=args.dry_run,
                    on_progress=_on_progress,
                    use_pool=not args.no_pool,
                    debugger=_debugger,
                )

        # Summary printed AFTER TUI context exits (normal terminal restored)
        if args.continuous:
            print_3way_summary(all_results)
        else:
            print_3way_summary(results)
        return

    # Compute alias_map for summary display
    alias_map: dict[str, str] = {}
    if not args.no_dedup:
        _, alias_map = _deduplicate_roles(args.roles)

    if args.continuous:
        # ── Continuous mode ──
        batch = 0
        consecutive_failures = 0
        logger.info(f"Starting continuous evaluation: session={session_id}")
        logger.info(f"  Ctrl+C to stop gracefully (finishes current question)")

        while not state.shutdown:
            # ── Health gate with auto-recovery ──
            if not _check_server_health(args.url):
                consecutive_failures += 1
                if consecutive_failures > MAX_RECOVERY_ATTEMPTS:
                    logger.error(
                        f"API unrecoverable after {MAX_RECOVERY_ATTEMPTS} attempts. Exiting."
                    )
                    break
                backoff = min(30 * (2 ** (consecutive_failures - 1)), 600)
                logger.warning(
                    f"API down (attempt {consecutive_failures}/{MAX_RECOVERY_ATTEMPTS}). "
                    f"Attempting recovery..."
                )
                recovered = _attempt_recovery(args.url)
                if recovered:
                    logger.info("  Recovery successful — resuming evaluation")
                    consecutive_failures = 0
                    continue
                logger.warning(f"  Recovery failed — sleeping {backoff}s before retry")
                for _ in range(backoff):
                    if state.shutdown:
                        break
                    time.sleep(1)
                continue
            consecutive_failures = 0

            batch += 1
            batch_seed = base_seed + batch
            logger.info(f"\n[Batch {batch}, seed={batch_seed}]")

            try:
                results = run_batch(
                    suites=args.suites,
                    roles=args.roles,
                    modes=args.modes,
                    sample_per_suite=args.sample_size,
                    seed=batch_seed,
                    url=args.url,
                    timeout=args.timeout,
                    session_id=session_id,
                    dry_run=args.dry_run,
                    skip_cache=args.skip_cache,
                    cooldown=args.cooldown,
                    no_dedup=args.no_dedup,
                    escalation_chains=not args.no_escalation_chains,
                    use_pool=not args.no_pool,
                )
            except HealthCheckError:
                # API died mid-batch — loop back to health gate
                logger.warning("API died during batch — will attempt recovery")
                continue

            if results:
                print_batch_summary(results, args.roles, args.modes, alias_map=alias_map)

            if state.shutdown:
                break

            logger.info(f"\n[Sleeping {args.continuous_interval}s before next batch...]")
            for _ in range(args.continuous_interval):
                if state.shutdown:
                    break
                time.sleep(1)

        logger.info(f"\nSession complete: {session_id}")
        logger.info(f"  Run --stats to see aggregate results")

    else:
        # ── One-shot mode (original behavior with HF datasets) ──
        results = run_batch(
            suites=args.suites,
            roles=args.roles,
            modes=args.modes,
            sample_per_suite=args.sample_size,
            seed=base_seed,
            url=args.url,
            timeout=args.timeout,
            session_id=session_id,
            dry_run=args.dry_run,
            skip_cache=args.skip_cache,
            cooldown=args.cooldown,
            no_dedup=args.no_dedup,
            escalation_chains=not args.no_escalation_chains,
            use_pool=not args.no_pool,
        )

        if results:
            print_batch_summary(results, args.roles, args.modes, alias_map=alias_map)

        # Save JSON output (legacy format for backwards compat)
        output_path = args.output
        if output_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                PROJECT_ROOT / "benchmarks" / "results" / "orchestrator"
                / f"seeding_{ts}.json"
            )

        all_combos = _build_role_mode_combos(args.roles, args.modes)
        output_data = {
            "config": {
                "suites": args.suites,
                "roles": args.roles,
                "modes": args.modes,
                "combos": [f"{r}:{m}" for r, m in all_combos],
                "sample_size": args.sample_size,
                "seed": base_seed,
                "dry_run": args.dry_run,
                "dedup": not args.no_dedup,
                "alias_map": alias_map,
            },
            "results": [asdict(r) for r in results],
            "timestamp": datetime.utcnow().isoformat(),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"JSONL checkpoint: {EVAL_DIR / f'{session_id}.jsonl'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        state.close_poll_client()
