"""Per-call serving telemetry: one JSONL record per HTTP call to a llama-server.

Why this exists (workspace-89 prefill-share analysis, 2026-10-03): the llama-server
logs carry no wall clock and no caller, ``progress/*.jsonl`` carried no prompt timing,
and the only way to attribute a week of model-server traffic to a role or client was
to reconstruct it from slot ids and timestamps relative to process start. This module
is the write side that makes that reconstruction unnecessary.

Choke point
    ``LlamaServerBackend.infer`` and ``LlamaServerBackend.infer_stream_text`` are
    wrapped with :func:`recorded_call`. Every HTTP call the orchestrator makes to a
    llama-server through the backend layer — direct, via ``CachingBackend``, via
    ``ConcurrencyAwareBackend`` or via ``model_server`` — passes through one of the
    two, so one record is written per backend call whatever the caller.

Layers above contribute what only they know, through context variables that never
change a call's behaviour:

* :func:`stage_caller` — the primitives layer stages WHO is calling (role, request /
  task / session ids, workload class) and WHEN it started waiting, before it takes
  the inference / region lock. The wrapped call consumes the stage, so the gap
  between staging and dispatch is the queue / lock wait. A call that never reaches
  the backend (lock timeout, cancellation while queued) is recorded by
  :func:`abandon_staged`, with what is known.
* :func:`note` — anything below the wrapper adds facts it parsed (llama.cpp's
  ``timings`` object, the endpoint, the instance a ``ConcurrencyAwareBackend``
  placed the call on).

Provenance needed for a belief-kernel ``ClaimTuple`` projection rides every record:
the orchestrator commit the process started on, a per-process run id, and the
server launch identity (argv sha256, binary, served model) that the stack writes to
``logs/server_launches/<port>.json`` at launch (``scripts/server/stack_log_banner.py``).
A record whose server predates that sidecar says so (``server.identity_source =
"absent"``) rather than guessing.

Writing never affects inference: every failure in this module is swallowed.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from src.runtime.git_head import resolve_git_head

try:  # pragma: no cover - fcntl exists on every supported host
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

SCHEMA = "epyc.orchestrator.serving_call.v1"

#: Path override. ``off`` / ``0`` / ``none`` disables the log (the test suite does).
LOG_ENV = "ORCHESTRATOR_SERVING_CALLS_LOG"
MAX_MB_ENV = "ORCHESTRATOR_SERVING_CALLS_MAX_MB"
KEEP_ENV = "ORCHESTRATOR_SERVING_CALLS_KEEP"
_DEFAULT_MAX_MB = 64
_DEFAULT_KEEP = 8
_DISABLED_VALUES = {"off", "0", "none", "false", "disabled"}

#: The subset of llama.cpp's ``timings`` object every record carries verbatim.
TIMING_KEYS = (
    "cache_n",
    "prompt_n",
    "prompt_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_second",
    "draft_n",
    "draft_n_accepted",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_write_lock = threading.Lock()

# Context: the caller staged by the primitives layer, the facts noted during a call,
# and a re-entrancy flag so `infer_stream_text -> infer` writes ONE record.
_STAGED: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "serving_calls_staged", default=None
)
_NOTES: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "serving_calls_notes", default=None
)
_IN_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "serving_calls_in_call", default=False
)


# ---------------------------------------------------------------------------
# Process provenance (computed once, at import: the code that is LOADED)
# ---------------------------------------------------------------------------


_PROCESS = {
    "orch_commit": os.environ.get("ORCHESTRATOR_GIT_SHA") or resolve_git_head(_REPO_ROOT),
    "run_id": uuid.uuid4().hex,
    "pid": os.getpid(),
    "host": socket.gethostname(),
    "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
}


def process_provenance() -> dict[str, Any]:
    """The per-process provenance block (commit, run id, pid, host, start)."""
    prov = dict(_PROCESS)
    if prov["pid"] != os.getpid():  # forked worker: own run id, inherited commit
        _PROCESS.update(run_id=uuid.uuid4().hex, pid=os.getpid())
        prov = dict(_PROCESS)
    return prov


# ---------------------------------------------------------------------------
# Paths and the server-launch sidecar
# ---------------------------------------------------------------------------


def _log_dir() -> Path:
    return Path(os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", str(_REPO_ROOT / "logs")))


def log_path() -> Path | None:
    """Where records go, or None when disabled."""
    override = os.environ.get(LOG_ENV, "").strip()
    if override.lower() in _DISABLED_VALUES:
        return None
    if override:
        return Path(override)
    return _log_dir() / "serving_calls" / "serving_calls.jsonl"


def launch_sidecar_dir() -> Path:
    """Directory the stack writes per-port launch identity into (see stack_log_banner)."""
    return _log_dir() / "server_launches"


_SIDECAR_CACHE: dict[int, tuple[float, dict[str, Any] | None]] = {}


def server_identity(port: int | None) -> dict[str, Any]:
    """Launch identity for the server on ``port``, from the stack's sidecar.

    Re-read whenever the sidecar's mtime changes (a reload rewrites it). Absence is
    reported, never filled: ``identity_source`` is ``"absent"`` when no sidecar
    exists — e.g. a server launched before the sidecar was introduced.
    """
    if not port:
        return {"identity_source": "absent"}
    path = launch_sidecar_dir() / f"{int(port)}.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"identity_source": "absent"}
    cached = _SIDECAR_CACHE.get(int(port))
    if cached is None or cached[0] != mtime:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = None
        _SIDECAR_CACHE[int(port)] = (mtime, data if isinstance(data, dict) else None)
        cached = _SIDECAR_CACHE[int(port)]
    data = cached[1]
    if not data:
        return {"identity_source": "unreadable"}
    keep = (
        "launch_id",
        "launched_at",
        "pid",
        "roles",
        "argv_sha256",
        "binary",
        "binary_realpath",
        "model_path",
        "ld_library_path",
        "stack_commit",
    )
    out = {k: data.get(k) for k in keep if k in data}
    out["identity_source"] = "stack_sidecar"
    return out


def _port_of(url: str | None) -> int | None:
    if not url:
        return None
    try:
        return urlparse(url).port
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Context API
# ---------------------------------------------------------------------------


def stage_caller(**fields: Any) -> None:
    """Stage who is about to call a backend; the wait clock starts now.

    Called by the primitives layer BEFORE it takes the inference / region lock. The
    next wrapped backend call in this context consumes it.
    """
    try:
        staged = {k: v for k, v in fields.items() if v not in (None, "", {}, [])}
        staged["_ts0"] = time.time()
        _STAGED.set(staged)
    except Exception:
        pass


_QUEUE_KEYS = ("placement_wait_ms", "instance_idx", "instance_full")


def annotate_staged(**fields: Any) -> None:
    """Add dispatch-layer facts (instance placement) to the staged caller.

    Called by ``ConcurrencyAwareBackend`` between acquiring an instance and calling
    it, i.e. after staging and before the recorded call starts. Without a staged
    caller (a direct backend user) a stage is created that carries only these facts
    and no wait clock, so no wait is claimed for it.
    """
    try:
        staged = _STAGED.get()
        if staged is None:
            staged = {}
            _STAGED.set(staged)
        for key, value in fields.items():
            if value is not None:
                staged[key] = value
    except Exception:
        pass


def clear_staged() -> None:
    """Drop a staged caller that no backend call consumed (e.g. a cache hit)."""
    _STAGED.set(None)


def note(**fields: Any) -> None:
    """Attach facts to the call in flight (no-op outside a recorded call)."""
    notes = _NOTES.get()
    if notes is None:
        return
    try:
        for key, value in fields.items():
            if value is not None:
                notes[key] = value
    except Exception:
        pass


def note_timings(timings: Any, *, endpoint: str | None = None, stream: bool | None = None) -> None:
    """Record llama.cpp's ``timings`` object (verbatim subset) for the call in flight."""
    # Only a SERVER timings object counts: it always carries prompt_n / predicted_n.
    # A client-synthesized stand-in (the /completion early-stop branch builds one
    # from wall time) must never be recorded as server-measured.
    if isinstance(timings, dict) and ("prompt_n" in timings or "predicted_n" in timings):
        note(timings={k: timings[k] for k in TIMING_KEYS if k in timings})
    if endpoint is not None or stream is not None:
        note(endpoint=endpoint, stream=stream)


def note_usage(usage: Any) -> None:
    """Record an OpenAI-style ``usage`` object for the call in flight."""
    if isinstance(usage, dict) and usage:
        keep = ("prompt_tokens", "completion_tokens", "total_tokens")
        note(usage={k: usage[k] for k in keep if k in usage})


# ---------------------------------------------------------------------------
# Record construction and writing
# ---------------------------------------------------------------------------


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _rotation() -> tuple[int, int]:
    try:
        max_mb = int(os.environ.get(MAX_MB_ENV, str(_DEFAULT_MAX_MB)))
    except ValueError:
        max_mb = _DEFAULT_MAX_MB
    try:
        keep = int(os.environ.get(KEEP_ENV, str(_DEFAULT_KEEP)))
    except ValueError:
        keep = _DEFAULT_KEEP
    return max_mb * 1024 * 1024, max(1, keep)


def _maybe_rotate(path: Path) -> None:
    """Size-based rotation ``f -> f.1 -> ... -> f.keep``. Caller holds the flock."""
    max_bytes, keep = _rotation()
    if max_bytes <= 0:
        return
    try:
        if path.stat().st_size < max_bytes:
            return
    except OSError:
        return
    for i in range(keep, 0, -1):
        src = path if i == 1 else path.with_name(f"{path.name}.{i - 1}")
        dst = path.with_name(f"{path.name}.{i}")
        if i == keep:
            try:
                dst.unlink()
            except OSError:
                pass
        try:
            if src.exists():
                src.rename(dst)
        except OSError:
            pass


def record_digest(record: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of ``record`` minus its own ``record_sha256``.

    Lets a reader re-derive each line's digest (a self-hashed row): a mutated line
    no longer matches, so an attestation built on it grades down rather than up.
    """
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    canonical = json.dumps(_json_safe(body), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_record(record: dict[str, Any]) -> bool:
    """Append one self-hashed record under a cross-process flock (several uvicorn workers)."""
    path = log_path()
    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = _json_safe(dict(record))
        record["record_sha256"] = record_digest(record)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        lock_path = path.with_name(path.name + ".lock")
        with _write_lock, open(lock_path, "a") as lock_fh:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                _maybe_rotate(path)
                with open(path, "a") as fh:
                    fh.write(line)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return True
    except Exception:
        return False


def classify_outcome(result: Any, exc: BaseException | None, early_stop: bool) -> str:
    """One word for how the call ended.

    ``ok`` · ``early_stop`` (the caller's on_chunk stopped the stream: FINAL marker,
    repetition guard or client cancel) · ``timeout`` · ``cancelled`` ·
    ``context_overflow`` · ``failed`` · ``exception``.
    """
    if exc is not None:
        text = f"{type(exc).__name__} {exc}".lower()
        if "timeout" in text:
            return "timeout"
        if "cancel" in text:
            return "cancelled"
        return "exception"
    reason = " ".join(
        str(getattr(result, attr, "") or "")
        for attr in ("completion_reason", "failure_reason", "error_message")
    ).lower()
    if getattr(result, "context_overflow", None):
        return "context_overflow"
    if "timeout" in reason:
        return "timeout"
    if "cancel" in reason:
        return "cancelled"
    if not getattr(result, "success", True):
        return "failed"
    if early_stop:
        return "early_stop"
    return "ok"


def _caller_block(staged: dict[str, Any] | None) -> dict[str, Any]:
    caller = {
        k: v for k, v in (staged or {}).items() if not k.startswith("_") and k not in _QUEUE_KEYS
    }
    caller["source"] = "primitives" if staged and "_ts0" in staged else "unstaged"
    return caller


def build_record(
    *,
    method: str,
    role_config: Any,
    request: Any,
    base_url: str | None,
    ts_start: float,
    ts_end: float,
    result: Any = None,
    exc: BaseException | None = None,
    staged: dict[str, Any] | None = None,
    notes: dict[str, Any] | None = None,
    dispatched: bool = True,
) -> dict[str, Any]:
    """Assemble one ``serving_call.v1`` record (pure, for tests and the writer)."""
    notes = dict(notes or {})
    port = _port_of(base_url)
    early_stop = bool(notes.pop("early_stop", False))
    timings = notes.pop("timings", None)
    model = getattr(role_config, "model", None)
    queue: dict[str, Any] = {}
    if staged and "_ts0" in staged:
        queue["pre_dispatch_wait_ms"] = round(
            max(0.0, (ts_start - staged["_ts0"]) * 1000.0), 3
        )
    for key in _QUEUE_KEYS:
        if staged and key in staged:
            queue[key] = staged[key]
        if key in notes:
            queue[key] = notes.pop(key)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "record_id": uuid.uuid4().hex,
        "ts_start": _iso(ts_start),
        "ts_end": _iso(ts_end),
        "wall_ms": round((ts_end - ts_start) * 1000.0, 3),
        "dispatched": dispatched,
        "method": method,
        "role": getattr(role_config, "name", None) or getattr(request, "role", None),
        "request_role": getattr(request, "role", None),
        "server": {"base_url": base_url, "port": port, **server_identity(port)},
        "model_registry": getattr(model, "name", None) if model is not None else None,
        "caller": _caller_block(staged),
        "queue": queue,
        "timings": timings,
        "timings_source": "server" if timings else "absent",
        "outcome": classify_outcome(result, exc, early_stop),
        "provenance": process_provenance(),
    }
    if result is not None:
        record["result"] = {
            "success": getattr(result, "success", None),
            "completion_reason": getattr(result, "completion_reason", None) or None,
            "failure_reason": getattr(result, "failure_reason", None) or None,
            "error_message": (str(getattr(result, "error_message", "") or "")[:300] or None),
            "tokens_generated": getattr(result, "tokens_generated", None),
            "prompt_tokens": getattr(result, "prompt_tokens", None),
            "cached_prompt_tokens": getattr(result, "cached_prompt_tokens", None),
            "prompt_eval_ms": getattr(result, "prompt_eval_ms", None),
            "generation_ms": getattr(result, "generation_ms", None),
            "first_token_ms": getattr(result, "first_token_ms", None) or None,
            "stream_chunks": getattr(result, "stream_chunks", None) or None,
        }
    if exc is not None:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
    if request is not None:
        record["request"] = {
            "n_tokens": getattr(request, "n_tokens", None),
            "prompt_chars": len(getattr(request, "prompt", None) or ""),
            "timeout_s": getattr(request, "timeout", None),
            "slot_id": getattr(request, "slot_id", None),
            "chat_payload": getattr(request, "chat_payload", None) is not None,
        }
    if notes:
        record["notes"] = notes
    return record


def abandon_staged(exc: BaseException | None = None) -> None:
    """Record a staged call that never reached a backend, then clear the stage.

    A call that WAS dispatched consumed its stage and wrote its own record, so this
    is a no-op for it. What is left is a call that died queued — lock timeout,
    cancellation or deadline while waiting — which used to leave no trace at all.
    """
    staged = _STAGED.get()
    if not staged:
        return
    _STAGED.set(None)
    try:
        now = time.time()
        record = build_record(
            method="undispatched",
            role_config=None,
            request=None,
            base_url=staged.get("backend_url"),
            ts_start=staged.get("_ts0", now),
            ts_end=now,
            exc=exc,
            staged=None,
            dispatched=False,
        )
        record["role"] = staged.get("role")
        record["caller"] = _caller_block(staged)
        record["queue"] = {
            "pre_dispatch_wait_ms": round(max(0.0, (now - staged.get("_ts0", now)) * 1000.0), 3)
        }
        write_record(record)
    except Exception:
        pass


def _wrap_on_chunk(on_chunk: Callable[[str], Any] | None, notes: dict[str, Any]):
    if on_chunk is None:
        return None

    def _observed(content: str) -> Any:
        if "first_chunk_at" not in notes:
            notes["first_chunk_at"] = time.time()
        try:
            return on_chunk(content)
        except StopIteration:
            notes["early_stop"] = True
            raise

    return _observed


def recorded_call(method: str) -> Callable:
    """Decorate a backend entry point ``(self, role_config, request, ...)``.

    Writes exactly one record per outermost call, after the call returns or raises;
    the call's result and exceptions pass through untouched.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self: Any, role_config: Any, request: Any, *args: Any, **kwargs: Any):
            if _IN_CALL.get() or log_path() is None:
                return fn(self, role_config, request, *args, **kwargs)
            staged = _STAGED.get()
            _STAGED.set(None)
            notes: dict[str, Any] = {}
            in_token = _IN_CALL.set(True)
            notes_token = _NOTES.set(notes)
            if "on_chunk" in kwargs:
                kwargs["on_chunk"] = _wrap_on_chunk(kwargs["on_chunk"], notes)
            elif args and callable(args[0]):
                args = (_wrap_on_chunk(args[0], notes),) + tuple(args[1:])
            ts_start = time.time()
            result: Any = None
            error: BaseException | None = None
            try:
                result = fn(self, role_config, request, *args, **kwargs)
                return result
            except BaseException as exc:  # recorded, then re-raised unchanged
                error = exc
                raise
            finally:
                _IN_CALL.reset(in_token)
                _NOTES.reset(notes_token)
                try:
                    first = notes.pop("first_chunk_at", None)
                    if first is not None:
                        notes["first_chunk_ms"] = round((first - ts_start) * 1000.0, 3)
                    record = build_record(
                        method=method,
                        role_config=role_config,
                        request=request,
                        base_url=getattr(getattr(self, "config", None), "base_url", None),
                        ts_start=ts_start,
                        ts_end=time.time(),
                        result=result,
                        exc=error,
                        staged=staged,
                        notes=notes,
                    )
                    write_record(record)
                except Exception:
                    pass

        return wrapper

    return decorator
