"""Original local resource owners for a GPU run without a run-wide CPU claim."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
import uuid

from . import claim, loop, scratch

CHILD_PHASE_DIR = "AK_GPU_LOCAL_PHASE_DIR"


@contextmanager
def child_build(cpu_list, *, paths=()):
    """An author sandbox's own acquired CPU owner, never a parent's observed one."""
    directory = os.environ.get(CHILD_PHASE_DIR)
    if not directory:
        yield
        return
    owner = LocalPhases(cpu_list=cpu_list, quiet=False, should_stop=lambda: False,
                        on_wait=lambda kind: None)
    path = Path(directory) / f"{os.getpid()}-{uuid.uuid4().hex}.json"
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps({"pid": os.getpid(), "cpu_list": cpu_list}) + "\n")
    try:
        with owner.build():
            with owner.compile_env(dict(os.environ), paths=paths) as env:
                from .procguard import ENV_SCOPE
                previous = os.environ.get(ENV_SCOPE)
                os.environ[ENV_SCOPE] = env[ENV_SCOPE]
                try:
                    yield
                finally:
                    if previous is None:
                        os.environ.pop(ENV_SCOPE, None)
                    else:
                        os.environ[ENV_SCOPE] = previous
    finally:
        if owner.capture_error is None and owner.phases:
            path.write_text(json.dumps(owner.phases[0], allow_nan=False) + "\n")
        # A release/capture error is not a completed child phase. Keep the pending
        # marker so the parent refuses accounting even if the child exits normally.
        if owner.capture_error is None:
            pending.unlink()


class LocalPhases:
    """Keep closed CPU-build/quiet facts; hosted planning owns neither resource.

    The device owner remains the run's original context. Quiet ownership excludes
    CPU measurements, but never masquerades as physical CPU ownership or cost.
    """

    def __init__(self, *, cpu_list, quiet, should_stop, on_wait):
        self.cpu_list = cpu_list
        self.quiet = quiet
        self.should_stop = should_stop
        self.on_wait = on_wait
        self.phases = []
        self.child_dir = None
        self.capture_error = None
        self._mutex = threading.RLock()
        self._depth = threading.local()

    def closed_phases(self):
        if self.capture_error is not None:
            raise claim.ClaimRefused(f"GPU local owner capture failed: {self.capture_error}")
        rows = list(self.phases)
        if self.child_dir is not None:
            if any(self.child_dir.glob("*.pending")):
                raise claim.ClaimRefused("GPU child build lacks its original release receipt")
            for path in sorted(self.child_dir.glob("*.json")):
                if len(rows) >= 4096 or path.stat().st_size > 1024 * 1024:
                    raise claim.ClaimRefused("GPU child phase capture exceeds bounds")
                rows.append(json.loads(path.read_text()))
        return rows

    @contextmanager
    def _hold(self, kind, provider):
        # Local tails are serialized by the loop. This also keeps resource
        # transitions disjoint if another callback arrives from a worker thread.
        with self._mutex:
            if self.capture_error is not None:
                raise claim.ClaimRefused(f"GPU local owner capture failed: {self.capture_error}")
            level = getattr(self._depth, "kind", None)
            if level:
                if level != kind:
                    raise claim.ClaimRefused("GPU local build and compute phases cannot nest")
                yield
                return
            receipt = None
            report_at = 0.
            while True:
                if self.should_stop():
                    raise loop.TailRefused(f"stopped before GPU {kind} acquired its claim")
                native_entered = False
                enter_returned = False

                def on_acquired():
                    nonlocal native_entered
                    native_entered = True

                try:
                    owner = provider(on_acquired)
                    receipt = owner.__enter__()
                    enter_returned = True
                    if not native_entered:
                        # A nonconforming provider cannot prove pre-entry absence.
                        # It did return from enter, so always close it before refusal.
                        owner.__exit__(None, None, None)
                        raise claim.ClaimRefused("GPU local provider omitted native-entry callback")
                    break
                except BaseException as exc:
                    if (native_entered or enter_returned or not isinstance(exc, Exception)
                            or not claim.region_lock_busy(exc)):
                        # Entry can fail after the provider acquired the resource
                        # but before its observer/receipt exists. It is still cost.
                        self.capture_error = f"{type(exc).__name__}: {exc}"
                        raise
                    if time.monotonic() >= report_at:
                        self.on_wait(kind)
                        report_at = time.monotonic() + 30.
            self._depth.kind = kind
            try:
                yield
            finally:
                self._depth.kind = None
                try:
                    owner.__exit__(None, None, None)
                    component = receipt.retained_interval()
                    if self.capture_error is None:
                        self.phases.append({"kind": kind, "component": component})
                except BaseException as exc:
                    self.capture_error = f"{type(exc).__name__}: {exc}"
                    raise

    def build(self):
        return self._hold("build", lambda entered: claim.hold_cpu(
            self.cpu_list, role="build", on_acquired=entered))

    @contextmanager
    def compile_env(self, env=None, *, paths=()):
        """Cleanup uncertainty poisons publication even when native release succeeds."""
        try:
            with scratch.captured_child_env(env, paths=paths) as captured:
                yield captured
        except scratch.ScratchRefused as exc:
            self.capture_error = f"{type(exc).__name__}: {exc}"
            raise

    @contextmanager
    def compute(self):
        if not self.quiet:
            # Legacy launchers already own a continuous exclusive quiet hold.
            yield
            return
        with self._hold("gpu_compute", lambda entered: claim.hold_gpu_quiet_measurement(
                retain=True, on_acquired=entered)):
            yield
