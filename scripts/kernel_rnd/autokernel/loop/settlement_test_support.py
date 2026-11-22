"""AKU-06l3d: test helpers that follow the runtime's settlement default (v1 or v2).

Fault injection must patch the settle entry point the runtime actually calls, and
assertions read the original held receipt from a v1 row or a single-stage v2 row's child."""
from __future__ import annotations

from .profile_preparation import _settled_receipt
from .standalone_runtime import StandaloneRuntimeConfig
from .unified_driver import _thaw


def settle_name(config: StandaloneRuntimeConfig | None = None) -> str:
    """The controller method the runtime's executors call to settle one attempt."""
    on = (config or StandaloneRuntimeConfig()).operation_settlement
    return "unified_driver_settle_v2" if on else "unified_driver_settle"


SETTLE = settle_name()


def settled_receipt(controller, settled) -> dict:
    """The original held receipt of a durable v1 or single-stage v2 settlement row."""
    return dict(_settled_receipt(controller, _thaw(settled)))


__all__ = ["SETTLE", "settle_name", "settled_receipt"]
