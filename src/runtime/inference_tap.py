"""Inference tap for prompt/response visibility with low-overhead modes.

Activated by setting INFERENCE_TAP_FILE env var to a file path.
When active, model calls write prompt/response to the tap file so that
``tail -f`` shows live activity.

Streaming policy is configurable via ``INFERENCE_TAP_STREAM_MODE``:

- ``safe`` (default): stream only non-heavy roles; heavy roles use stable
  non-streaming inference and write response after completion.
- ``force``: stream all roles (highest live fidelity, highest contention risk).
- ``off``: disable streaming path even when tap is active.

Performance impact is zero when disabled — is_active() is an O(1)
env-var check with no I/O.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time as _time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.roles import Role
from src.registry.stack_priors import (
    DEFAULT_OUTPUT as DEFAULT_STACK_PRIORS,
    live_stack_safe_non_stream_roles,
)

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


_ENV_KEY = "INFERENCE_TAP_FILE"
_ENV_EVENTS_KEY = "INFERENCE_TAP_EVENTS_FILE"
_ENV_STREAM_MODE = "INFERENCE_TAP_STREAM_MODE"
# Size-based rotation for the structured events JSONL. Without it the file grows
# unbounded under autopilot-eval load (observed 2.3 GB on 2026-05-31), which both
# wastes disk and slows every dashboard reverse-grep lookup. Rotation happens under
# the same cross-process flock as the append, by rename — safe because writers open
# the events file fresh per event (no long-lived fd), so the next append after a
# rotate transparently creates a new file.
_ENV_EVENTS_MAX_MB = "INFERENCE_TAP_EVENTS_MAX_MB"
_ENV_EVENTS_KEEP = "INFERENCE_TAP_EVENTS_KEEP"
_ENV_SAFE_NON_STREAM_MIN_MEM_GB = "INFERENCE_TAP_SAFE_NON_STREAM_MIN_MEM_GB"
_DEFAULT_EVENTS_MAX_MB = 512
_DEFAULT_EVENTS_KEEP = 3
_DEFAULT_SAFE_NON_STREAM_MIN_MEM_GB = 64.0

# Sentinel file written by the TUI so that API workers (separate processes)
# can discover the tap path without needing the env var.
_SENTINEL: str | None = None


def _get_sentinel() -> str:
    global _SENTINEL
    if _SENTINEL is None:
        try:
            from src.config import get_config
            _SENTINEL = str(get_config().paths.tmp_dir / ".inference_tap_active")
        except Exception:
            _SENTINEL = "/mnt/raid0/llm/tmp/.inference_tap_active"
    return _SENTINEL

# Module-level lock for serialising writes across threads
_write_lock = threading.Lock()

# Current tap section for the active inference call. Concurrency-aware
# backends use this to attach the selected instance after dispatch.
_current_writer: contextvars.ContextVar["TapWriter | None"] = contextvars.ContextVar(
    "inference_tap_current_writer",
    default=None,
)

_TOPOLOGY_HASH_CACHE: str | None = None

# Cache sentinel reads — the file only changes when the TUI starts/stops,
# so 5-second staleness is fine and avoids per-request I/O.
_sentinel_cache: tuple[str, float] = ("", 0.0)


def _safe_non_stream_min_mem_gb() -> float:
    raw = os.environ.get(
        _ENV_SAFE_NON_STREAM_MIN_MEM_GB,
        str(_DEFAULT_SAFE_NON_STREAM_MIN_MEM_GB),
    )
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_SAFE_NON_STREAM_MIN_MEM_GB


def _safe_non_stream_roles_from_stack_priors(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> frozenset[str] | None:
    """Derive tap safe-mode non-stream roles from generated stack-prior memory."""
    return live_stack_safe_non_stream_roles(
        stack_priors_path,
        min_mem_gb=_safe_non_stream_min_mem_gb(),
    )


def _degraded_safe_non_stream_roles_from_stack_manifest() -> frozenset[str] | None:
    """Return no degraded safe-mode policy when generated priors are unavailable."""
    return None


def _all_known_roles() -> frozenset[str]:
    return frozenset(str(role) for role in Role)


def safe_non_stream_roles(
    stack_priors_path: Path | None = None,
) -> frozenset[str]:
    """Return the current safe-mode non-stream role set.

    Keep this dynamic so stack-prior or threshold changes are not frozen at
    module import. Tests and legacy callers may monkeypatch
    ``SAFE_NON_STREAM_ROLES``; honor that explicit override when present.
    """
    override = globals().get("SAFE_NON_STREAM_ROLES")
    if isinstance(override, frozenset):
        return override
    if stack_priors_path is None:
        stack_priors_path = DEFAULT_STACK_PRIORS
    derived = _safe_non_stream_roles_from_stack_priors(stack_priors_path)
    if derived is not None:
        return derived
    return _degraded_safe_non_stream_roles_from_stack_manifest() or _all_known_roles()


def __getattr__(name: str) -> Any:
    if name == "SAFE_NON_STREAM_ROLES":
        return safe_non_stream_roles()
    raise AttributeError(name)


def _read_sentinel() -> str:
    """Read the sentinel file, caching the result for 5 seconds."""
    global _sentinel_cache
    now = _time.monotonic()
    if now - _sentinel_cache[1] < 5.0:
        return _sentinel_cache[0]
    try:
        with open(_get_sentinel()) as f:
            val = f.read().strip()
    except (FileNotFoundError, OSError):
        val = ""
    _sentinel_cache = (val, now)
    return val


def is_active() -> bool:
    """Return True when the inference tap is enabled."""
    return bool(os.environ.get(_ENV_KEY) or _read_sentinel())


def _tap_path() -> str:
    """Return the configured tap file path."""
    return os.environ.get(_ENV_KEY, "") or _read_sentinel()


def _structured_event_path(tap_path: str | None = None) -> str:
    """Return the JSONL event stream path for structured tap metadata."""
    override = os.environ.get(_ENV_EVENTS_KEY, "").strip()
    if override:
        return override
    path = (tap_path or _tap_path() or "").strip()
    if not path or path == os.devnull:
        return ""
    try:
        p = Path(path)
        if str(p) == "/dev/null":
            return ""
        if p.name == "inference_tap.log":
            return str(p.with_name("inference_tap_events.jsonl"))
        suffix = p.suffix or ".log"
        return str(p.with_suffix(f"{suffix}.events.jsonl"))
    except Exception:
        return ""


def stream_mode() -> str:
    """Return normalized tap stream mode: safe|force|off."""
    mode = (os.environ.get(_ENV_STREAM_MODE, "safe") or "safe").strip().lower()
    if mode in {"safe", "force", "off"}:
        return mode
    return "safe"


def should_stream_role(role: str) -> bool:
    """Whether tap should force streaming transport for this role."""
    mode = stream_mode()
    if mode == "off":
        return False
    if mode == "force":
        return True
    policy = safe_non_stream_roles()
    normalized_role = Role.from_string(role)
    if normalized_role is None:
        role_name = str(role)
        return role_name not in policy and policy != _all_known_roles()
    normalized = str(normalized_role)
    return normalized not in policy


def _topology_hash() -> str:
    """Best-effort live topology hash used to join tap events to matrix data."""
    global _TOPOLOGY_HASH_CACHE
    if _TOPOLOGY_HASH_CACHE is not None:
        return _TOPOLOGY_HASH_CACHE
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
        from src.scheduling.contention import topology_fingerprint

        _TOPOLOGY_HASH_CACHE = topology_fingerprint(NUMA_CONFIG)
    except Exception:
        _TOPOLOGY_HASH_CACHE = ""
    return _TOPOLOGY_HASH_CACHE


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _events_rotation_config() -> tuple[int, int]:
    """(max_bytes, keep) for the events JSONL, env-overridable. max_bytes<=0 disables."""
    try:
        max_mb = int(os.environ.get(_ENV_EVENTS_MAX_MB, str(_DEFAULT_EVENTS_MAX_MB)))
    except ValueError:
        max_mb = _DEFAULT_EVENTS_MAX_MB
    try:
        keep = int(os.environ.get(_ENV_EVENTS_KEEP, str(_DEFAULT_EVENTS_KEEP)))
    except ValueError:
        keep = _DEFAULT_EVENTS_KEEP
    return max_mb * 1024 * 1024, max(1, keep)


def _maybe_rotate_events(p: Path) -> None:
    """Rotate the events file when it exceeds the size cap. Caller MUST hold the
    cross-process flock. Shifts `<f> -> <f>.1 -> <f>.2 ... -> <f>.keep` (oldest
    dropped). Best-effort: any failure leaves the current file in place so a write
    can still proceed."""
    max_bytes, keep = _events_rotation_config()
    if max_bytes <= 0:
        return
    try:
        if p.stat().st_size < max_bytes:
            return
    except OSError:
        return
    # Shift oldest-first so no slot is clobbered before it is moved.
    for i in range(keep, 0, -1):
        src = p if i == 1 else p.with_name(f"{p.name}.{i - 1}")
        dst = p.with_name(f"{p.name}.{i}")
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


def _write_structured_event(path: str, event: dict[str, Any]) -> None:
    """Append one JSONL event under a cross-process file lock.

    The legacy plaintext tap is intentionally unchanged. This structured
    stream is the concurrency-safe attribution source for dashboards and
    post-hoc analysis.
    """
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lock_path = p.with_suffix(p.suffix + ".lock")
        line = json.dumps(
            {str(k): _json_safe(v) for k, v in event.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        with open(lock_path, "a") as lock_fh:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                # Rotate while the lock is held (writers open the events file fresh
                # per append, so a rename here is picked up transparently below).
                _maybe_rotate_events(p)
                with open(p, "a") as event_fh:
                    event_fh.write(line)
                    event_fh.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        # The tap must never affect inference behavior.
        return


def emit_request_event(event_type: str, **fields: Any) -> bool:
    """Append one REQUEST-level event to the structured tap stream.

    Unlike a ``tap_section``, this belongs to no single model call: it records
    a fact about a whole API request (TE-1: the /v1 escalation receipt). It
    writes to the JSONL events file only (never the plaintext tap) and carries
    NO ``request_id`` key on purpose: request_id identifies one model call, and
    the dashboard groups events by it, so a request-level event carrying one
    would render as a phantom in-flight call. Callers identify the request with
    their own keys (``chat_id``, ``request_keys``). Returns False, with no I/O
    beyond the O(1) activity check, when the tap is off.
    """
    if not is_active():
        return False
    path = _structured_event_path()
    if not path:
        return False
    event = {
        "event": event_type,
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "ts_epoch": _time.time(),
        "pid": os.getpid(),
        "topology_hash": _topology_hash(),
        **{k: v for k, v in fields.items() if k != "request_id"},
    }
    _write_structured_event(path, event)
    return True


def annotate_current_tap(**metadata: Any) -> bool:
    """Attach metadata to the active tap section, if one exists."""
    writer = _current_writer.get()
    if writer is None:
        return False
    writer.set_metadata(**metadata)
    return True


class TapWriter:
    """Thread-safe writer that appends tap output to a file.

    Keeps a per-section file descriptor open and closes on context exit.
    This avoids open/close on every streamed chunk.
    """

    def __init__(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        self._path = path
        self._event_path = _structured_event_path(path)
        self._fh = None
        self._role = ""
        self._start_emitted = False
        self._end_emitted = False
        self._metadata: dict[str, Any] = dict(metadata or {})
        request_id = str(self._metadata.get("request_id") or "").strip()
        if not request_id:
            request_id = uuid.uuid4().hex
        self._metadata.update(
            {
                "request_id": request_id,
                "pid": os.getpid(),
                "topology_hash": self._metadata.get("topology_hash") or _topology_hash(),
            }
        )

    @property
    def request_id(self) -> str:
        return str(self._metadata.get("request_id") or "")

    def _emit_event(self, event_type: str, **fields: Any) -> None:
        if not self._event_path:
            return
        now = _time.time()
        event = {
            "event": event_type,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "ts_epoch": now,
            **self._metadata,
            **fields,
        }
        _write_structured_event(self._event_path, event)

    def _append(self, text: str) -> None:
        with _write_lock:
            if self._fh is None:
                self._fh = open(self._path, "a")
            self._fh.write(text)
            self._fh.flush()

    def write_header(self, role: str) -> None:
        self._role = role
        self._metadata["role"] = role
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append(
            f"{'=' * 72}\n"
            f"[{ts}] ROLE={role} REQUEST={self.request_id} PID={os.getpid()}\n"
            f"{'-' * 72}\n"
            f"PROMPT:\n"
        )

    def write_prompt(self, prompt: str, max_chars: int = 8000) -> None:
        # 8000c cap (was 2000c). The dashboard's task-detail tap-section
        # matcher does substring search on the user objective inside this
        # field; long system prompts + chat templates would push the user
        # portion past 2000c, breaking the match and producing "(empty)"
        # INFERENCE STREAM views for completed tasks. 8000c covers all
        # observed system+template+user combinations in the production stack.
        if len(prompt) > max_chars:
            text = prompt[:max_chars] + f"\n... [{len(prompt) - max_chars} chars truncated]"
        else:
            text = prompt
        self._append(text + "\n" + "-" * 72 + "\nRESPONSE:\n")
        if not self._start_emitted:
            self._start_emitted = True
            self._emit_event(
                "start",
                prompt=text,
                prompt_len=len(prompt),
                prompt_preview_len=len(text),
                prompt_truncated=len(prompt) > max_chars,
            )

    def write_chunk(self, chunk: str) -> None:
        """Write a single streaming chunk (called per SSE event)."""
        self._append(chunk)
        if chunk:
            self._emit_event("chunk", text=chunk, text_len=len(chunk))

    def write_response(self, text: str) -> None:
        """Write a complete response when non-stream path is used."""
        if text:
            self._append(text)
            self._emit_event("response", text=text, text_len=len(text))

    def write_timings(
        self,
        tokens: int,
        prompt_ms: float,
        gen_ms: float,
        tps: float,
        prompt_tokens: int | None = None,
    ) -> None:
        """Record the request's terminal timings.

        ``prompt_tokens`` is llama-server's OWN terminal prompt-token count
        (``tokens_evaluated`` on /completion, ``usage.prompt_tokens`` on
        chat/completions — the same figure ``/slots`` reports as
        ``n_prompt_tokens`` mid-run). It is recorded ONLY when the server
        reported a positive value, and always together with
        ``prompt_tokens_source="server_terminal"`` so a reader can tell a
        measured count from the character-length estimate a card would
        otherwise fall back to (RTG-47 data plane). A request that was never
        observed mid-run therefore still carries TRUE prompt tokens at
        completion; an early-stopped or usage-less request carries none and
        stays legibly an estimate rather than a fabricated zero.
        """
        total_s = (prompt_ms + gen_ms) / 1000.0
        measured_prompt = (
            int(prompt_tokens)
            if isinstance(prompt_tokens, (int, float))
            and not isinstance(prompt_tokens, bool)
            and prompt_tokens > 0
            else None
        )
        prompt_note = f", prompt_tokens={measured_prompt}" if measured_prompt else ""
        self._append(
            f"\n{'-' * 72}\n"
            f"TIMINGS: {tokens} tokens in {total_s:.2f}s "
            f"(prompt={prompt_ms:.0f}ms, gen={gen_ms:.0f}ms, {tps:.1f} t/s{prompt_note})\n"
            f"{'=' * 72}\n\n"
        )
        fields: dict[str, Any] = {
            "tokens": tokens,
            "prompt_ms": prompt_ms,
            "gen_ms": gen_ms,
            "tps": tps,
            "total_s": total_s,
        }
        if measured_prompt is not None:
            fields["prompt_tokens"] = measured_prompt
            fields["prompt_tokens_source"] = "server_terminal"
        self._emit_event("timings", **fields)

    def set_metadata(self, **metadata: Any) -> None:
        clean = {k: v for k, v in metadata.items() if v is not None}
        if not clean:
            return
        self._metadata.update(clean)
        self._emit_event("metadata", **clean)

    def close(self) -> None:
        if not self._end_emitted:
            self._end_emitted = True
            self._emit_event("end")
        with _write_lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None


class _NullWriter:
    """No-op writer when tap is disabled."""

    def write_header(self, role: str) -> None:
        pass

    def write_prompt(self, prompt: str, max_chars: int = 2000) -> None:
        pass

    def write_chunk(self, chunk: str) -> None:
        pass

    def write_response(self, text: str) -> None:
        pass

    def write_timings(
        self,
        tokens: int,
        prompt_ms: float,
        gen_ms: float,
        tps: float,
        prompt_tokens: int | None = None,
    ) -> None:
        pass

    def close(self) -> None:
        pass

    def set_metadata(self, **metadata: Any) -> None:
        pass


@contextmanager
def tap_section(role: str, prompt: str, metadata: dict[str, Any] | None = None):
    """Context manager that yields a TapWriter (or _NullWriter if inactive).

    Usage::

        with tap_section(role, prompt) as tap:
            result = backend.infer_stream_text(role_config, request,
                                                on_chunk=tap.write_chunk)
            tap.write_timings(result.tokens_generated, ...)
    """
    if not is_active():
        yield _NullWriter()
        return

    path = _tap_path()
    writer = TapWriter(path, metadata=metadata)
    token = _current_writer.set(writer)
    try:
        writer.write_header(role)
        writer.write_prompt(prompt)
        yield writer
    finally:
        try:
            writer.close()
        finally:
            _current_writer.reset(token)
