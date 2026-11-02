"""Optional KPF-64 cooperative quiescence; no server restart or slot serialization.

A quiescent process must remain alive with its server and RAM reservation intact.
This module does not grant CPU time, freeze processes, or certify replay equivalence.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import threading
import time


class PauseGate:
    def __init__(self):
        self.cond = threading.Condition()
        self.paused = False
        self.active = 0
        self.paused_at = None
        self.paused_seconds = 0.0

    def enter(self, stop: threading.Event) -> bool:
        with self.cond:
            while self.paused and not stop.is_set():
                self.cond.wait(0.1)
            if stop.is_set():
                return False
            self.active += 1
            return True

    def leave(self) -> None:
        with self.cond:
            if self.active <= 0:
                raise RuntimeError('pause gate active count underflow')
            self.active -= 1
            self.cond.notify_all()

    def pause(self, timeout: float) -> bool:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('pause timeout must be positive')
        deadline = time.monotonic() + timeout
        with self.cond:
            if self.paused:
                raise RuntimeError('already paused')
            self.paused = True
            self.paused_at = time.monotonic()
            self.cond.notify_all()
            while self.active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Failed pause is never a grant to freeze or release a region.
                    self.paused_seconds += time.monotonic() - self.paused_at
                    self.paused_at = None
                    self.paused = False
                    self.cond.notify_all()
                    return False
                self.cond.wait(remaining)
            return True

    def resume(self) -> None:
        with self.cond:
            if not self.paused:
                raise RuntimeError('not paused')
            self.paused_seconds += time.monotonic() - self.paused_at
            self.paused_at = None
            self.paused = False
            self.cond.notify_all()

    def _active_time(self) -> float:
        now = time.monotonic()
        return now - self.paused_seconds - ((now - self.paused_at) if self.paused else 0.0)

    def active_sleep(self, stop: threading.Event, seconds: float) -> float:
        """Wait only active time; a paused interval does not consume think time."""
        remaining = max(0.0, seconds)
        waited = 0.0
        with self.cond:
            while remaining > 0 and not stop.is_set():
                while self.paused and not stop.is_set():
                    self.cond.wait(0.1)
                if stop.is_set():
                    break
                start = self._active_time()
                self.cond.wait(min(remaining, 0.1))
                elapsed = min(remaining, max(0.0, self._active_time() - start))
                waited += elapsed
                remaining -= elapsed
        return waited


class Control:
    """Private local control socket; an authenticated OS owner controls one run."""
    def __init__(self, path: Path, gate: PauseGate):
        self.path, self.gate = path, gate
        self.stop = threading.Event()
        parent = path.parent.stat()
        if parent.st_uid != os.geteuid() or stat.S_IMODE(parent.st_mode) & 0o077:
            raise PermissionError('pause socket directory must be private and owned by this UID')
        if path.exists() or path.is_symlink():
            raise FileExistsError('pause socket already exists')
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.bind(str(path))
            os.chmod(path, 0o600)
            self.sock.listen(1)
            self.sock.settimeout(0.1)
        except BaseException:
            self.sock.close()
            raise
        self.identity = (path.lstat().st_dev, path.lstat().st_ino)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(1)
                    raw = conn.recv(4096)
                    message = json.loads(raw)
                    op = message.get('op')
                    if op == 'pause':
                        ok = self.gate.pause(float(message.get('timeout_s', 30)))
                        response = {'ok': ok, 'state': 'quiescent' if ok else 'running'}
                    elif op == 'resume':
                        self.gate.resume()
                        response = {'ok': True, 'state': 'running'}
                    elif op == 'status':
                        with self.gate.cond:
                            response = {'ok': True, 'state': 'paused' if self.gate.paused else 'running',
                                        'active': self.gate.active}
                    else:
                        raise ValueError('unknown control operation')
                except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
                    response = {'ok': False, 'error': str(exc)}
                with contextlib.suppress(OSError):
                    conn.sendall(json.dumps(response).encode() + b'\n')

    def close(self) -> None:
        self.stop.set()
        self.sock.close()
        self.thread.join(timeout=2)
        try:
            st = self.path.lstat()
            if (st.st_dev, st.st_ino) == self.identity:
                self.path.unlink()
        except FileNotFoundError:
            pass
