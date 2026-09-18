"""Belief-kernel write-side capture for GPU runner drivers: portable root, loud failures.

The capture writers (``<name>_capture.py``) live in epyc-root under ``scripts/vidya/adapters`` so
the writer and the strict reader cannot drift into two dialects of one schema. Drivers used to reach
them with ``sys.path.insert(0, "/workspace/scripts/vidya")`` inside a ``try/except Exception`` that
wrote the failure into a log. That broke in two ways (VB-RUNNER-PATHS, root
``handoffs/active/vidya-belief-substrate-program.md``):

* **Not portable.** ``/workspace`` is one container's view of the root checkout. The root is now
  named by ``EPYC_ROOT``, and an unset or wrong value is REFUSED, never guessed.
* **Silent.** A GPU run whose capture failed still exited 0, so a run without its write-time rows
  looked complete. Rows can never be back-filled on read, so the gap was permanent and unseen.
  Now a failure is printed to stderr, recorded, and turns the driver's exit code into
  ``EXIT_CAPTURE_FAILED``. The run's own results are still persisted.

Drivers call ``preflight()`` BEFORE starting any server, so a missing root costs no GPU time.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, TextIO

ENV_VAR = "EPYC_ROOT"
ADAPTERS_REL = Path("scripts/vidya/adapters")
#: The driver's own work succeeded but at least one write-time belief capture did not.
EXIT_CAPTURE_FAILED = 3


class CaptureUnavailable(RuntimeError):
    """The root checkout or a capture module cannot be resolved."""


def root_checkout(env: Mapping[str, str] | None = None) -> Path:
    """The epyc-root checkout named by ``EPYC_ROOT``. Refuses rather than guessing a default."""
    raw = (os.environ if env is None else env).get(ENV_VAR, "").strip()
    if not raw:
        raise CaptureUnavailable(
            f"{ENV_VAR} is not set: point it at the epyc-root checkout whose "
            f"{ADAPTERS_REL} holds the capture writers (no default is guessed)")
    root = Path(raw).expanduser().resolve()
    if not (root / ADAPTERS_REL).is_dir():
        raise CaptureUnavailable(f"{ENV_VAR}={raw} has no {ADAPTERS_REL} (resolved {root})")
    return root


def load_capture(name: str, env: Mapping[str, str] | None = None) -> ModuleType:
    """Import ``<root>/scripts/vidya/adapters/<name>.py`` by path, from that root only.

    Loading by file path (not ``sys.path``) means an ``adapters`` package from some other checkout
    already on the path can never shadow the one ``EPYC_ROOT`` names.
    """
    root = root_checkout(env)
    path = root / ADAPTERS_REL / f"{name}.py"
    if not path.is_file():
        raise CaptureUnavailable(f"capture module missing: {path}")
    spec = importlib.util.spec_from_file_location(f"epyc_root_capture_{name}", path)
    if spec is None or spec.loader is None:
        raise CaptureUnavailable(f"cannot load capture module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "write_belief_measurements", None)):
        raise CaptureUnavailable(f"{path} has no write_belief_measurements()")
    return module


def preflight(*names: str, env: Mapping[str, str] | None = None) -> dict[str, ModuleType]:
    """Resolve every capture module a driver will need, before any compute is spent."""
    return {name: load_capture(name, env) for name in names}


class CaptureLog:
    """Runs capture calls, never hides a failure, and folds failures into the exit code."""

    def __init__(self, stream: TextIO | None = None):
        self.stream = stream
        self.written: dict[str, str] = {}
        self.failed: dict[str, str] = {}

    def run(self, label: str, fn: Callable[[], Any]) -> str:
        """Call ``fn``; return the sidecar path, or a ``REFUSED ...`` string on failure."""
        try:
            side = str(fn())
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and surfaced
            msg = f"REFUSED {type(exc).__name__}: {exc}"
            self.failed[label] = msg
            print(f"!!! BELIEF CAPTURE FAILED [{label}]: {msg}", file=self.stream or sys.stderr,
                  flush=True)
            return msg
        self.written[label] = side
        return side

    def as_record(self) -> dict[str, Any]:
        return {"written": dict(self.written), "failed": dict(self.failed),
                "root": os.environ.get(ENV_VAR, "")}

    def exit_code(self, rc: int = 0) -> int:
        """``rc`` if the driver already failed, else ``EXIT_CAPTURE_FAILED`` on any capture failure."""
        if rc:
            return rc
        if self.failed:
            print(f"!!! {len(self.failed)} belief capture(s) failed: {sorted(self.failed)}; "
                  f"exit {EXIT_CAPTURE_FAILED}", file=self.stream or sys.stderr, flush=True)
            return EXIT_CAPTURE_FAILED
        return 0
