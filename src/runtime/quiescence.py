"""INF-78 OAB-3 (R2): per-request suppression of trailing orchestrator work.

Why this exists
---------------
The AutoKernel loop calls ``/chat`` for a plan or a patch, and its NEXT step is a CPU
measurement window. Any orchestrator-owned work still running after the reply — MemRL
q-scoring, a typed-decision shadow call, a KV migration, the idle-time scoring batch — lands inside that window and poisons it. ``ChatRequest.quiescent_after=True``
asks the orchestrator to start none of it.

Two mechanisms, because the work has two shapes:

1. **Per-request launch sites** check :func:`suppress` (a ContextVar carrier, same shape as
   the AP-54 knowledge fence and ``gate_observation``). ``asyncio`` tasks, ``asyncio.to_thread``
   and ``ensure_future`` copy the context, so a carrier installed in ``chat()`` is visible at
   every launch site on the request path. The carrier records what it suppressed and the
   response echoes it (``ChatResponse.quiescence``).

2. **The ambient idle-scoring loop** (``api.services.memrl.background_cleanup``) runs outside
   any request, on ONE elected uvicorn worker, while the quiescent request may be served by
   any of the six. So the request also takes a cross-process **hold**: one small file per
   request under :func:`hold_dir` whose content is a unix deadline. At request start the
   deadline covers the whole request budget plus the post-reply window; at the reply it is
   rewritten to ``now + hold_after_s()``. The loop skips its tick, and an in-flight batch
   stops between tasks, while any hold is live. Task ids whose scoring was suppressed are
   also written to an exclusion list so the idle loop never scores them LATER (it would
   otherwise pick them up from the progress-log backlog after the hold expires — i.e. in the
   middle of the caller's next measurement).

Default (flag absent/False): :func:`begin` installs nothing and every hook is a no-op, so
production behaviour is unchanged. Acceptance is a witness, not this promise —
``src/runtime/trailing_work_witness.py`` samples the servers' CPU after the reply.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOLD_DIR_ENV = "ORCHESTRATOR_QUIESCENCE_DIR"
HOLD_AFTER_ENV = "ORCHESTRATOR_QUIESCENCE_HOLD_S"
# Covers the R2 witness window (60 s) with margin. Override per deployment via env.
DEFAULT_HOLD_AFTER_S = 120.0
# The progress-log backlog looks back one day (ProgressReader.get_unscored_tasks(days=1)),
# so an exclusion older than that can never matter again.
EXCLUSION_TTL_S = 2 * 86400.0
MAX_RECORDED = 64


@dataclass
class QuiescenceCarrier:
    """Per-request state. Shared by reference across context copies."""

    request_id: str
    suppressed: list[str] = field(default_factory=list)
    suppressed_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, label: str) -> None:
        with self._lock:
            self.suppressed_count += 1
            if label not in self.suppressed and len(self.suppressed) < MAX_RECORDED:
                self.suppressed.append(label)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "quiescent_after": True,
                "suppressed": list(self.suppressed),
                "suppressed_count": self.suppressed_count,
                "hold_after_s": hold_after_s(),
            }


_carrier: ContextVar[QuiescenceCarrier | None] = ContextVar("orchestrator_quiescence", default=None)


# ── per-request carrier ──────────────────────────────────────────────────────


def begin(quiescent_after: bool, *, request_id: str = "", budget_s: float | None = None) -> QuiescenceCarrier | None:
    """Install the carrier (and the start-of-request hold). False installs nothing."""
    if not quiescent_after:
        _carrier.set(None)
        return None
    rid = _safe_id(request_id) or f"q{os.getpid()}-{time.monotonic_ns()}"
    carrier = QuiescenceCarrier(request_id=rid)
    _carrier.set(carrier)
    # Hold through the whole request: an idle-scoring batch that STARTED mid-request would
    # otherwise run on past the reply.
    _write_hold(rid, time.time() + float(budget_s or 3600.0) + hold_after_s())
    return carrier


def finish(carrier: QuiescenceCarrier | None) -> None:
    """At the reply: shrink this request's hold to ``now + hold_after_s()``."""
    if carrier is not None:
        _write_hold(carrier.request_id, time.time() + hold_after_s())


def clear() -> None:
    """Remove the carrier. Call in a ``finally`` so it never outlives its request."""
    _carrier.set(None)


def current() -> QuiescenceCarrier | None:
    return _carrier.get()


def active() -> bool:
    return _carrier.get() is not None


def suppress(label: str) -> bool:
    """Launch-site hook. True means: do NOT start ``label``; it has been recorded."""
    carrier = _carrier.get()
    if carrier is None:
        return False
    carrier.record(label)
    return True


def suppress_scoring(task_id: str) -> bool:
    """``suppress('memrl_q_scoring')`` plus a durable exclusion of ``task_id`` from the
    idle-scoring backlog, so it is not scored later during the caller's measurement."""
    if not suppress("memrl_q_scoring"):
        return False
    exclude_task(task_id)
    return True


# ── cross-process hold + exclusions (file based) ─────────────────────────────


def hold_after_s() -> float:
    try:
        return max(0.0, float(os.environ.get(HOLD_AFTER_ENV, DEFAULT_HOLD_AFTER_S)))
    except ValueError:
        return DEFAULT_HOLD_AFTER_S


def hold_dir() -> Path:
    override = os.environ.get(HOLD_DIR_ENV, "").strip()
    if override:
        return Path(override)
    try:
        from src.config import get_config

        base = Path(get_config().paths.tmp_dir)
    except Exception:
        base = Path("/mnt/raid0/llm/tmp")
    return base / "orchestrator_quiescence"


def _safe_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value or ""))[:96]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_hold(request_id: str, deadline_unix: float) -> None:
    with contextlib.suppress(OSError):
        _atomic_write(hold_dir() / "holds" / f"{_safe_id(request_id)}.until", f"{deadline_unix:.3f}\n")


def quiet_active(now: float | None = None) -> bool:
    """True while any quiescent request's hold is live (any process). Prunes expired holds.

    Never raises; an unreadable directory reads as NOT quiet (the idle loop keeps its
    production behaviour), and the witness is what proves quiescence, not this check.
    """
    now = time.time() if now is None else now
    holds = hold_dir() / "holds"
    live = False
    try:
        entries = list(os.scandir(holds))
    except OSError:
        return False
    for entry in entries:
        if not entry.name.endswith(".until"):
            continue
        try:
            deadline = float(Path(entry.path).read_text(encoding="utf-8").strip() or 0.0)
        except (OSError, ValueError):
            continue
        if deadline > now:
            live = True
        else:
            with contextlib.suppress(OSError):
                os.unlink(entry.path)
    return live


def exclude_task(task_id: str) -> None:
    if not task_id:
        return
    with contextlib.suppress(OSError):
        _atomic_write(hold_dir() / "excluded" / _safe_id(task_id), f"{time.time():.3f}\n")


def is_excluded(task_id: str) -> bool:
    """True when ``task_id``'s scoring was suppressed by a quiescent request."""
    if not task_id:
        return False
    return (hold_dir() / "excluded" / _safe_id(task_id)).exists()


def prune_exclusions(now: float | None = None) -> None:
    now = time.time() if now is None else now
    try:
        entries = list(os.scandir(hold_dir() / "excluded"))
    except OSError:
        return
    for entry in entries:
        with contextlib.suppress(OSError):
            if now - entry.stat().st_mtime > EXCLUSION_TTL_S:
                os.unlink(entry.path)
