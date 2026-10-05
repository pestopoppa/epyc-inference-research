"""Per-server history of recently served prompt prefixes (KVU-15c, 2026-10-03).

Why. KVU-15a sizes the one-long-prefill rule on NEW tokens: a long prompt whose
prefix a slot already caches prefills only its suffix. It read that prefix from
the prompt TEXT ``GET /slots`` returns for an idle slot, but llama-server v10
returns slot text only under ``LLAMA_SERVER_SLOTS_DEBUG``
(``server-context.cpp`` ``to_json`` :699-730, ``slot.to_json(slots_debug == 0)``
at :2541), so in production that credit never applied. This module is the
orchestrator-side source: what the orchestrator itself served, per server.

What. Every completed, dispatched call writes a ``serving_call.v1`` record
(``src/backends/serving_calls.py``); the same hook remembers its prompt here —
the UFH14-B4 ``prefix_fp`` fingerprint at the denser history ladder
(``serving_calls.prefix_ladder``), the server-measured prompt tokens
(``timings.prompt_n + cache_n``) and the tokens generated. A new request to the
same server is credited with the LONGEST ladder depth at which its fingerprint
equals a remembered one: those leading characters were prefilled by a call that
has finished, so a slot or ``--cache-ram`` holds them unless they were evicted.

Survival (when a remembered prefix stops counting). The credit is a heuristic
and errs low:

* ``--cache-ram`` volume. llama-server's prompt cache evicts by SIZE, not age.
  An entry stops counting once the tokens served on that server after it, plus
  its own size, exceed the cache's capacity in tokens:
  ``cache_ram_mib * MiB / ORCHESTRATOR_KV_POOL_CACHE_CREDIT_BYTES_PER_TOKEN``
  (default 65536 B/token — above the 38,960 B/token measured on
  Qwen3.8-27B-Q8_0, ``src/registry/prefix_cache_policy.py``, so capacity is
  UNDER-stated). ``--cache-ram`` comes from the compiled registry per port; an
  undeclared value means the server default (8192 MiB); ``0`` means no prompt
  cache, and on a unified multi-slot pool (the only kind the admission gate
  guards) an idle conversation then survives nowhere: no credit.
* Time. ``ORCHESTRATOR_KV_POOL_CACHE_CREDIT_WINDOW_S`` (default 1800 s) bounds
  an entry's age whatever the volume.
* Idle-slot purge (STACKCHG-8083BATCH-20261004). On a unified pool launched with
  ``--no-cache-idle-slots`` an idle conversation stays in its SLOT; it reaches the
  RAM cache only when its slot is reused, and is lost without a copy when the server
  purges idle slots because the pool is full. An entry then also has to pass a
  no-purge bound: its own size plus the (slots - 1) largest other remembered entries
  of the same launch must fit in the pool -- if they do, the pool can never have
  been full while it sat idle. Entries the history no longer holds (pruned by age or
  count, or traffic that bypassed the orchestrator) are not seen, so this errs low
  like the rest of the credit, not safe. With idle-slot caching on (the server
  default) the volume rule alone applies, unchanged.
* Relaunch. An entry carries the server's launch id (the stack's per-port launch
  sidecar, ``serving_calls.server_identity``); after a relaunch the cache is
  empty, so an entry from another launch never counts.

Host-wide. The API runs several uvicorn workers; a follow-up turn is often
served by a different worker than the turn before it. With
``ORCHESTRATOR_KV_PREFIX_HISTORY_HOST_WIDE`` on (the default) each server's
history is also a small JSON file next to the long-prefill lease files
(``{tmp_dir}/kv_prefix_history.{host}_{port}.json``), rewritten atomically under
an flock and read (mtime-cached) by every worker. A failed file read or write
degrades to this process's own history. ``ORCHESTRATOR_KV_PREFIX_HISTORY=0``
turns the source off (whole-prompt sizing unless ``/slots`` text is present).

The history is fed from the serving record, so ``ORCHESTRATOR_SERVING_CALLS_LOG
=off`` also stops it feeding (the credit then falls back, never over-credits).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

HISTORY_ENV = "ORCHESTRATOR_KV_PREFIX_HISTORY"
HOST_WIDE_ENV = "ORCHESTRATOR_KV_PREFIX_HISTORY_HOST_WIDE"
MAX_ENTRIES_ENV = "ORCHESTRATOR_KV_PREFIX_HISTORY_MAX_ENTRIES"
DEFAULT_MAX_ENTRIES = 64
WINDOW_S_ENV = "ORCHESTRATOR_KV_POOL_CACHE_CREDIT_WINDOW_S"
DEFAULT_WINDOW_S = 1800.0
BYTES_PER_TOKEN_ENV = "ORCHESTRATOR_KV_POOL_CACHE_CREDIT_BYTES_PER_TOKEN"
DEFAULT_BYTES_PER_TOKEN = 65536
#: llama.cpp v10 ``--cache-ram`` default (common/common.h).
SERVER_DEFAULT_CACHE_RAM_MIB = 8192
FILE_PREFIX = "kv_prefix_history."
#: Outcomes whose prompt the server finished prefilling (so it was cached).
SERVED_OUTCOMES = ("ok", "early_stop")
_MIB = 1024 * 1024
_OFF = {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _default_cache_ram(url: str) -> int | None:
    from src.backends.context_limits import get_context_limit_resolver

    return get_context_limit_resolver().cache_ram_mib(url)


def _default_idle_residency(url: str) -> tuple[int, int] | None:
    from src.backends.context_limits import get_context_limit_resolver

    return get_context_limit_resolver().idle_slot_residency(url)


def _default_launch_id(url: str) -> str | None:
    from src.backends import serving_calls

    return serving_calls.server_identity(serving_calls._port_of(url)).get("launch_id")


def _history_dir() -> Path:
    # The long-prefill lease's directory (the orchestrator tmp dir), so the test
    # suite's hermetic lease-dir fixture isolates this file too.
    from src.runtime import long_prefill_lease

    return long_prefill_lease.lease_dir()


@dataclass(frozen=True)
class HistoryMatch:
    """The best remembered prefix for a request."""

    matched_chars: int        # ladder depth at which the fingerprints agree
    entry_chars: int          # the remembered prompt's fingerprinted length
    entry_prompt_tokens: int | None  # server-measured prompt tokens of that call
    age_s: float
    tokens_since: int         # tokens served on the server after it

    def tokens_per_char(self) -> float | None:
        if self.entry_prompt_tokens and self.entry_chars > 0:
            return self.entry_prompt_tokens / self.entry_chars
        return None


class PrefixHistory:
    """In-process (and host-wide file) LRU of served prefixes, per server."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        cache_ram_mib: Callable[[str], int | None] | None = None,
        launch_id: Callable[[str], str | None] | None = None,
        host_wide: bool | None = None,
        directory: Callable[[], Path] | None = None,
        idle_residency: Callable[[str], tuple[int, int] | None] | None = None,
    ) -> None:
        self._clock = clock
        self._cache_ram_fn = cache_ram_mib or _default_cache_ram
        self._idle_residency_fn = idle_residency or _default_idle_residency
        self._launch_id_fn = launch_id or _default_launch_id
        self._host_wide = host_wide
        self._dir_fn = directory or _history_dir
        self._lock = threading.Lock()
        # key -> {"cum": int, "entries": [entry, ...]} (oldest first)
        self._mem: dict[str, dict[str, Any]] = {}
        # path -> (mtime_ns, size, state) for the host-wide file
        self._file_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}

    # -- configuration --------------------------------------------------------
    @staticmethod
    def enabled() -> bool:
        return os.environ.get(HISTORY_ENV, "1").strip().lower() not in _OFF

    def host_wide(self) -> bool:
        if self._host_wide is not None:
            return bool(self._host_wide)
        return os.environ.get(HOST_WIDE_ENV, "1").strip().lower() not in _OFF

    @staticmethod
    def key(url: str) -> str:
        from src.runtime.long_prefill_lease import server_key

        return server_key(url)

    def path(self, url: str) -> Path:
        return self._dir_fn() / f"{FILE_PREFIX}{self.key(url)}.json"

    def capacity_tokens(self, url: str) -> int | None:
        """Tokens ``url``'s ``--cache-ram`` holds (under-stated); 0 = no cache."""
        try:
            mib = self._cache_ram_fn(url)
        except Exception:
            mib = None
        if mib is None:
            mib = SERVER_DEFAULT_CACHE_RAM_MIB
        if mib <= 0:
            return 0
        per_token = max(1, _env_int(BYTES_PER_TOKEN_ENV, DEFAULT_BYTES_PER_TOKEN))
        return int(mib) * _MIB // per_token

    def idle_residency(self, url: str) -> tuple[int, int] | None:
        """``(pool_tokens, slots)`` when idle slots stay resident (see the module
        docstring, "Idle-slot purge"), else None. Never raises."""
        try:
            value = self._idle_residency_fn(url)
        except Exception:
            return None
        if (isinstance(value, tuple) and len(value) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value)):
            return value
        return None

    @staticmethod
    def _purge_safe(entry: dict[str, Any], entries: list[dict[str, Any]],
                    launch_id: str | None, pool_tokens: int, slots: int) -> bool:
        """True when ``entry`` cannot have been purged from an idle slot: its size plus
        the ``slots - 1`` largest OTHER entries of the same launch fit in the pool."""
        others = sorted(
            (int(e.get("size_tokens", 0)) for e in entries
             if e is not entry and e.get("launch_id") == launch_id),
            reverse=True,
        )
        return int(entry.get("size_tokens", 0)) + sum(others[: max(0, slots - 1)]) <= pool_tokens

    def _launch_id(self, url: str) -> str | None:
        try:
            value = self._launch_id_fn(url)
        except Exception:
            return None
        return str(value) if value else None

    # -- survival -------------------------------------------------------------
    def _alive(self, entry: dict[str, Any], cum: int, now: float, capacity: int,
               window_s: float, launch_id: str | None) -> bool:
        if entry.get("launch_id") != launch_id:
            return False
        if window_s > 0 and now - float(entry.get("ts", 0.0)) > window_s:
            return False
        since = max(0, cum - int(entry.get("cum", 0)))
        return since + int(entry.get("size_tokens", 0)) <= capacity

    def _prune(self, state: dict[str, Any], now: float, capacity: int,
               launch_id: str | None) -> None:
        window_s = _env_float(WINDOW_S_ENV, DEFAULT_WINDOW_S)
        cum = int(state.get("cum", 0))
        keep = [e for e in state.get("entries", [])
                if self._alive(e, cum, now, capacity, window_s, launch_id)]
        limit = max(1, _env_int(MAX_ENTRIES_ENV, DEFAULT_MAX_ENTRIES))
        state["entries"] = keep[-limit:]

    # -- host-wide file -------------------------------------------------------
    def _read_file(self, path: Path) -> dict[str, Any] | None:
        try:
            st = path.stat()
        except OSError:
            return None
        cached = self._file_cache.get(str(path))
        if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(state, dict) or not isinstance(state.get("entries"), list):
            return None
        self._file_cache[str(path)] = (st.st_mtime_ns, st.st_size, state)
        return state

    def _append_file(self, url: str, entry: dict[str, Any], now: float, capacity: int,
                     launch_id: str | None) -> bool:
        path = self.path(url)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path.with_name(path.name + ".lock"), "a") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        state = json.loads(path.read_text())
                        if not isinstance(state, dict) or not isinstance(
                                state.get("entries"), list):
                            state = {}
                    except (OSError, ValueError):
                        state = {}
                    state.setdefault("entries", [])
                    state["cum"] = int(state.get("cum", 0)) + int(entry["size_tokens"])
                    stored = dict(entry, cum=state["cum"])
                    state["entries"].append(stored)
                    state["schema"] = "epyc.orchestrator.kv_prefix_history.v1"
                    self._prune(state, now, capacity, launch_id)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
                    tmp.write_text(json.dumps(state, separators=(",", ":")))
                    os.replace(tmp, path)
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            return True
        except Exception:
            logger.debug("prefix history: cannot write %s", path, exc_info=True)
            return False

    # -- API ------------------------------------------------------------------
    def remember(self, url: str, ladder: dict[str, Any] | None, *,
                 prompt_tokens: int | None = None, generated_tokens: int | None = None,
                 ts: float | None = None) -> None:
        """Remember one served call's prompt on ``url``. Never raises."""
        try:
            if not url or not ladder or not ladder.get("fp") or not self.enabled():
                return
            now = self._clock() if ts is None else float(ts)
            chars = int(ladder.get("chars") or 0)
            measured = prompt_tokens if prompt_tokens and prompt_tokens > 0 else None
            # Size in the cache: the measured prompt, else a conservative (high)
            # estimate, plus what was generated after it.
            size = (measured if measured else -(-chars // 3)) + max(0, int(generated_tokens or 0))
            launch_id = self._launch_id(url)
            entry = {
                "id": uuid.uuid4().hex[:12],
                "ts": now,
                "chars": chars,
                "fp": {str(k): v for k, v in ladder["fp"].items()},
                "prompt_tokens": measured,
                "size_tokens": int(size),
                "launch_id": launch_id,
            }
            capacity = self.capacity_tokens(url) or 0
            with self._lock:
                state = self._mem.setdefault(self.key(url), {"cum": 0, "entries": []})
                state["cum"] += int(size)
                state["entries"].append(dict(entry, cum=state["cum"]))
                self._prune(state, now, capacity, launch_id)
            if self.host_wide():
                self._append_file(url, entry, now, capacity, launch_id)
        except Exception:
            logger.debug("prefix history: remember failed", exc_info=True)

    def lookup(self, url: str, ladder: dict[str, Any] | None
               ) -> tuple[HistoryMatch | None, str | None]:
        """``(best match, None)`` or ``(None, reason)`` — reason is one of
        ``disabled``, ``no_fingerprint``, ``cache_ram_off``, ``no_history``,
        ``no_match`` or ``idle_purge_risk`` (a fingerprint matched, but every matching
        entry fails the idle-slot no-purge bound). Never raises."""
        try:
            if not self.enabled():
                return None, "disabled"
            if not url or not ladder or not ladder.get("fp"):
                return None, "no_fingerprint"
            capacity = self.capacity_tokens(url)
            if not capacity:
                return None, "cache_ram_off"
            state = self._read_file(self.path(url)) if self.host_wide() else None
            if state is None:
                with self._lock:
                    mem = self._mem.get(self.key(url))
                    state = {"cum": mem["cum"], "entries": list(mem["entries"])} if mem else None
            if not state or not state.get("entries"):
                return None, "no_history"
            now = self._clock()
            window_s = _env_float(WINDOW_S_ENV, DEFAULT_WINDOW_S)
            launch_id = self._launch_id(url)
            cum = int(state.get("cum", 0))
            want = sorted(((int(d), h) for d, h in ladder["fp"].items()), reverse=True)
            residency = self.idle_residency(url)
            best: HistoryMatch | None = None
            purge_rejected = False
            for entry in reversed(state["entries"]):  # newest first: ties keep the newest
                if not self._alive(entry, cum, now, capacity, window_s, launch_id):
                    continue
                fp = entry.get("fp") or {}
                for depth, digest in want:
                    if fp.get(str(depth)) == digest:
                        if residency is not None and not self._purge_safe(
                                entry, state["entries"], launch_id, *residency):
                            purge_rejected = True
                            break
                        if best is None or depth > best.matched_chars:
                            best = HistoryMatch(
                                matched_chars=depth,
                                entry_chars=int(entry.get("chars") or 0),
                                entry_prompt_tokens=_int(entry.get("prompt_tokens")),
                                age_s=round(max(0.0, now - float(entry.get("ts", now))), 3),
                                tokens_since=max(0, cum - int(entry.get("cum", 0))),
                            )
                        break
            if best is None:
                return None, "idle_purge_risk" if purge_rejected else "no_match"
            return best, None
        except Exception:
            logger.debug("prefix history: lookup failed", exc_info=True)
            return None, "no_history"

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            self._file_cache.clear()


def remember_record(record: dict[str, Any], ladder: dict[str, Any] | None) -> None:
    """Remember the prompt of a ``serving_call.v1`` record if the server finished
    prefilling it (dispatched, outcome ``ok``/``early_stop``). Never raises."""
    try:
        if not record.get("dispatched") or record.get("outcome") not in SERVED_OUTCOMES:
            return
        url = (record.get("server") or {}).get("base_url")
        if not url:
            return
        timings = record.get("timings") or {}
        result = record.get("result") or {}
        usage = (record.get("notes") or {}).get("usage") or {}
        prompt_n, cache_n = _int(timings.get("prompt_n")), _int(timings.get("cache_n"))
        if prompt_n is not None:
            prompt_tokens: int | None = prompt_n + (cache_n or 0)
        else:
            prompt_tokens = _int(result.get("prompt_tokens")) or _int(usage.get("prompt_tokens"))
        generated = (_int(timings.get("predicted_n")) or _int(result.get("tokens_generated"))
                     or _int(usage.get("completion_tokens")) or 0)
        get_prefix_history().remember(url, ladder, prompt_tokens=prompt_tokens,
                                      generated_tokens=generated)
    except Exception:
        logger.debug("prefix history: remember_record failed", exc_info=True)


_history: PrefixHistory | None = None
_history_lock = threading.Lock()


def get_prefix_history() -> PrefixHistory:
    global _history
    with _history_lock:
        if _history is None:
            _history = PrefixHistory()
        return _history


def set_prefix_history(history: PrefixHistory | None) -> None:
    """Replace (or with None, reset) the process-wide history (tests)."""
    global _history
    with _history_lock:
        _history = history


__all__ = [
    "HistoryMatch",
    "PrefixHistory",
    "get_prefix_history",
    "remember_record",
    "set_prefix_history",
]
