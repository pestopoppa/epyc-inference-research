"""Serialized worker-status heartbeat and terminal lifecycle.

This is deliberately only a worker publisher.  It does not decide whether a worker
is live, restart it, or supervise campaign state.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import threading
import time
from typing import Any


class HeartbeatStopError(RuntimeError):
    """The heartbeat could not be joined, so a terminal state was not published."""


class WorkerStatusPublisher:
    """Serialize progress writes and stop the heartbeat before a terminal write."""

    TERMINAL_STATES = frozenset({"complete", "failed"})

    def __init__(
            self,
            writer: Callable[..., None],
            heartbeat_payload: Callable[[], Mapping[str, Any]],
            *,
            interval_s: float,
            join_timeout_s: float,
            error_sink: Callable[[str], None],
    ) -> None:
        if (not math.isfinite(interval_s) or interval_s <= 0
                or not math.isfinite(join_timeout_s) or join_timeout_s <= 0):
            raise ValueError(
                "heartbeat interval and join timeout must be positive finite numbers")
        self._writer = writer
        self._heartbeat_payload = heartbeat_payload
        self._interval_s = interval_s
        self._join_timeout_s = join_timeout_s
        self._error_sink = error_sink
        self._write_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closing = False
        self._terminal_attempted = False
        self._last_state: str | None = None
        self._last_step: str | None = None

    def start(self) -> None:
        """Start once.  The caller publishes ``starting`` before calling this."""
        with self._write_lock:
            if self._closing:
                return
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._heartbeat,
                name="status-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def publish(self, state: str, *args: Any, step: str | None = None,
                **kwargs: Any) -> bool:
        """Publish a nonterminal snapshot, atomically retaining the latest step."""
        if state in self.TERMINAL_STATES:
            raise ValueError("terminal states must be published with close()")
        with self._write_lock:
            if self._stop.is_set() or self._closing or self._terminal_attempted:
                return False
            if step is not None:
                self._last_step = step
            self._writer(state, *args, step=self._last_step, **kwargs)
            self._last_state = state
            return True

    def close(self, state: str, *args: Any, step: str | None = None,
              **kwargs: Any) -> bool:
        """Stop/join, then publish one terminal snapshot.

        One monotonic deadline bounds acquiring the lifecycle locks and joining the
        heartbeat. If it expires, no terminal state is written: a still-running
        status write could otherwise later overwrite it. The terminal writer itself
        is synchronous filesystem I/O and is not falsely claimed to be time-bounded.
        """
        if state not in self.TERMINAL_STATES:
            raise ValueError(f"not a terminal worker state: {state}")
        # Fence new writes before waiting on a lock an in-flight writer may hold.
        self._stop.set()
        self._closing = True
        deadline = time.monotonic() + self._join_timeout_s
        if not self._acquire_before(self._close_lock, deadline):
            raise self._timeout(state, "acquiring the close lock")
        try:
            if self._terminal_attempted:
                return False
            thread = self._thread
            if thread is threading.current_thread():
                raise HeartbeatStopError(
                    f"heartbeat cannot publish terminal {state!r} from its own thread")
            if thread is not None:
                thread.join(self._remaining(deadline))
                if thread.is_alive():
                    raise self._timeout(state, "joining the heartbeat")
            if not self._acquire_before(self._write_lock, deadline):
                raise self._timeout(state, "acquiring the status write lock")
            try:
                if self._terminal_attempted:
                    return False
                if step is not None:
                    self._last_step = step
                # Fence all later nonterminal writes even if the terminal writer fails.
                self._terminal_attempted = True
                self._writer(state, *args, step=self._last_step, **kwargs)
                return True
            finally:
                self._write_lock.release()
        finally:
            self._close_lock.release()

    def close_failed(self, original: BaseException, *args: Any,
                     **kwargs: Any) -> None:
        """Attempt failed status without replacing an in-flight body exception."""
        try:
            self.close("failed", *args, **kwargs)
        except BaseException as reporting_error:
            message = ("worker status failure while reporting original "
                       f"{type(original).__name__}: {reporting_error!r}")
            try:
                original.add_note(message)
            except (AttributeError, TypeError):
                pass
            try:
                self._error_sink(message)
            except BaseException:
                # Status reporting must never replace the original run failure.
                pass

    def _heartbeat(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                payload = dict(self._heartbeat_payload())
                with self._write_lock:
                    if self._stop.is_set() or self._closing:
                        continue
                    # During claim acquisition and profiling, "starting" remains
                    # the truthful last state; only the run path advances to running.
                    if self._last_state is not None:
                        self._writer(self._last_state, step=self._last_step, **payload)
            except BaseException as exc:
                try:
                    self._error_sink(f"heartbeat skipped: {exc!r}")
                except BaseException:
                    pass

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    @classmethod
    def _acquire_before(cls, lock: threading.Lock, deadline: float) -> bool:
        return lock.acquire(timeout=cls._remaining(deadline))

    def _timeout(self, state: str, phase: str) -> HeartbeatStopError:
        return HeartbeatStopError(
            f"status shutdown exceeded {self._join_timeout_s:g}s while {phase}; "
            f"terminal {state!r} not published")
