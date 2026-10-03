"""Host-wide "one long prefill at a time" lease, one flock file per server.

KVU-15a (2026-10-03). ``src.scheduling.kv_pool_admission`` admits at most one
long prefill per llama-server. Its lease used to live in each uvicorn worker's
memory, so the API's six workers each had their own and two of them could start
long prefills on the same server seconds apart (the /slots observation only sees
a prefill after LONG_PREFILL_OBSERVE_TOKENS processed tokens plus its ~1.5 s
cache). This module makes the lease a host-wide primitive.

Primitive — the same one the per-region inference locks use
(``src.runtime.cpu_region_lock``): an exclusive ``fcntl.flock`` on a file under
the orchestrator tmp dir, ``{tmp_dir}/kv_prefill_lease.{host}_{port}.lock``,
keyed per server (loopback spellings of the host collapse to ``local``, so
``localhost:8083`` and ``127.0.0.1:8083`` are one key). The flock lives on an
open file description, so

* the kernel drops it when the holding process dies — a crashed or SIGKILLed
  worker never strands the lease (a pid-in-a-file lease would);
* two threads of ONE process that open the file separately also exclude each
  other, so the in-process and cross-process rules are the same rule.

The file body carries a JSON attribution payload (pid, ticket, url, expiry) for
diagnostics only; the flock alone is liveness truth, exactly as for the region
locks.

Every acquisition is NON-BLOCKING (``LOCK_NB``). Waiting is the admission
queue's job (a polled, deadline/cancel-bounded loop), so a lease wait can never
become an unbounded kernel wait.

Lock ordering (deadlock freedom). A request takes, in this order:

  1. the per-backend request semaphore   (``src/api/admission.py``, in-process)
  2. the KV-pool ticket and, for a long prefill, THIS lease
  3. the inference / CPU-region locks    (``heavy_model.lock``,
     ``cpu_region.{role}.{region}.lock``, ``cpu_region.GLOBAL.*``, the
     occupancy mutex)

and releases in reverse. Nothing acquires the lease while holding a lock of
level 3, so a lease holder may wait for a region lock but a region-lock holder
never waits for the lease: no cycle. And because the lease is only ever tried
non-blocking, even a future out-of-order caller degrades to a bounded poll, not
a deadlock.

``fork()`` without ``exec`` would share the open file description with the
child (and with it the lease) until the child exits; ``open()`` fds are
close-on-exec, so ``subprocess`` children do not inherit it.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import time
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit

from src.runtime import cpu_region_lock as _region

logger = logging.getLogger(__name__)

LEASE_FILE_PREFIX = "kv_prefill_lease."
_LOOPBACK = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", "ip6-localhost", ""}


def lease_dir() -> Path:
    """Where lease files live: the orchestrator tmp dir, resolved exactly as the
    region locks resolve it (``ORCHESTRATOR_TMP_DIR`` /
    ``ORCHESTRATOR_PATHS_TMP_DIR`` / ``/mnt/raid0/llm/tmp``). Tests patch this."""
    return _region._tmp_dir()


def server_key(url: str) -> str:
    """``host_port`` for ``url``, loopback hosts folded to ``local``."""
    raw = (url or "").strip()
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        host, port = raw, None
    if host in _LOOPBACK:
        host = "local"
    key = f"{host}_{port if port is not None else 0}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def lease_path(url: str) -> Path:
    return lease_dir() / f"{LEASE_FILE_PREFIX}{server_key(url)}.lock"


class LeaseHandle:
    """An acquired lease: the open file whose exclusive flock IS the lease."""

    __slots__ = ("url", "path", "_fh")

    def __init__(self, url: str, path: Path, fh: IO[bytes]) -> None:
        self.url = url
        self.path = path
        self._fh = fh

    @property
    def held(self) -> bool:
        return self._fh is not None

    def release(self) -> None:
        """Clear the payload, unlock, close. Idempotent; never raises."""
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            _region._clear_lock_payload(fh)
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                fh.close()
            except OSError:
                pass


def try_acquire(url: str, *, payload: dict[str, Any] | None = None) -> LeaseHandle | None:
    """Take ``url``'s lease without blocking: a handle, or None when another
    holder (any process, or another open of this one) has it. An I/O error is
    logged and reported as None (held), which fails safe: the caller waits."""
    path = lease_path(url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+b")
    except OSError as exc:
        logger.warning("long-prefill lease: cannot open %s: %s", path, exc)
        return None
    try:
        if not _region._try_flock(fh.fileno(), fcntl.LOCK_EX):
            fh.close()
            return None
    except OSError as exc:
        fh.close()
        logger.warning("long-prefill lease: flock %s failed: %s", path, exc)
        return None
    body = {"schema_version": 1, "pid": os.getpid(), "url": url, "acquired_at": time.time()}
    body.update(payload or {})
    _region._write_lock_payload(fh, body)
    return LeaseHandle(url, path, fh)


def is_held(url: str) -> bool:
    """Is ``url``'s lease held by anyone (this process included)? A shared,
    non-blocking probe: it never conflicts with another prober and only
    momentarily with a concurrent ``try_acquire`` (which then retries on its
    next poll). A missing file means nobody ever took it."""
    path = lease_path(url)
    if not path.exists():
        return False
    try:
        fh = open(path, "rb")
    except OSError:
        return False
    try:
        got = _region._try_flock(fh.fileno(), fcntl.LOCK_SH)
        if got:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return not got
    except OSError:
        return True  # cannot tell: fail safe (treat as held)
    finally:
        fh.close()


def holder_info(url: str) -> dict[str, Any] | None:
    """The attribution payload of a HELD lease (diagnostics), else None."""
    if not is_held(url):
        return None
    payload = _region.read_region_lock_payload(lease_path(url))
    if payload is not None:
        payload = dict(payload)
        payload["owner_pids"] = _region._current_lock_owner_pids(lease_path(url))
    return payload


__all__ = [
    "LeaseHandle",
    "holder_info",
    "is_held",
    "lease_dir",
    "lease_path",
    "server_key",
    "try_acquire",
]
