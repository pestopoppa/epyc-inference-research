"""Runtime-owner execution thread and factory validation for campaign_service.py, split verbatim from campaign_service.py (code-budget split)."""
from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time
from typing import Any, Callable

from .campaign import ResolvedCampaign
from .campaign_control import CampaignController, ControlRefused


RuntimeFactory = Callable[
    [ResolvedCampaign, argparse.Namespace],
    tuple[CampaignController, Any],
]


class _RuntimeExecution:
    """Own the sole non-HTTP runtime thread and retain its terminal state."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.result: Any = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="campaign-runtime-owner", daemon=False)

    def _run(self) -> None:
        from . import standalone_runtime as runtime_module

        try:
            result = self.runtime.run(self.stop_event)
            if not isinstance(result, runtime_module.RuntimeTickResult):
                raise ControlRefused("runtime worker returned an untyped result")
            with self._lock:
                self.result = result
        except BaseException as exc:
            with self._lock:
                self.error = exc

    def start(self) -> None:
        if self.thread.ident is not None:
            raise ControlRefused("runtime run thread was already started")
        self.thread.start()

    def request_stop(self) -> None:
        self.runtime.request_stop()
        self.stop_event.set()

    def finished(self) -> bool:
        return not self.thread.is_alive()

    def join(self, deadline: float) -> bool:
        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not self.thread.is_alive()


def _close_runtime(runtime: Any, deadline: float):
    from . import standalone_runtime as runtime_module

    result = runtime.close(deadline=deadline)
    if not isinstance(result, runtime_module.RuntimeShutdownResult):
        raise ControlRefused("runtime close returned an untyped result")
    return result


def _runtime_owner(runtime_factory: RuntimeFactory, resolved: ResolvedCampaign,
                   args: argparse.Namespace
                   ) -> tuple[CampaignController, Any]:
    from . import standalone_runtime as runtime_module

    if not callable(runtime_factory):
        raise ControlRefused("runtime_factory must be callable or absent")
    value = runtime_factory(resolved, args)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ControlRefused("runtime_factory must return (controller, runtime)")
    controller, runtime = value
    if not isinstance(controller, CampaignController) \
            or not isinstance(runtime, runtime_module.StandaloneRuntime):
        raise ControlRefused(
            "runtime_factory returned an untyped owner; factory retains cleanup authority")
    if runtime.controller is not controller:
        raise ControlRefused("runtime and service controllers differ; typed owners remain retained")
    try:
        if controller.snapshot_version != 3:
            raise ControlRefused("runtime service requires snapshot v3")
        if (controller.resolved.to_dict() != resolved.to_dict()
                or controller.config_generation != args.config_generation
                or controller.store != Path(args.store).absolute()):
            raise ControlRefused("runtime owner identity differs from service configuration")
        # AKU-09g3: the standalone producer emits projection v4 (report-only
        # resource/candidate sections); producers without this owner keep v3.
        controller.connect_aggregate_projection(runtime)
        controller.snapshot()  # public active-owner check after factory-controlled replay
        return controller, runtime
    except BaseException as exc:
        runtime.request_stop()
        closed = _close_runtime(runtime, time.monotonic() + args.shutdown_deadline)
        if closed.status == "closed":
            controller.close()
            raise
        raise ControlRefused(
            f"runtime owner validation failed and cleanup remains unresolved: "
            f"{closed.reason}") from exc
