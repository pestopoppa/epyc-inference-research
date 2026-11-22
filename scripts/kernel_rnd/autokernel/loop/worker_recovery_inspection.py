"""Worker recovery inspection record and helpers, split verbatim from worker_lifecycle.py (code-budget split)."""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Protocol


class LifecycleRefused(RuntimeError):
    """Launch, recovery, or result use failed closed."""


def _text(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.strip() or "\0" in value
            or len(value) > 16 * 1024):
        raise LifecycleRefused(f"{label} must be non-empty text")
    return value


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LifecycleRefused(f"{label} must be a positive integer")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or (positive and value <= 0)):
        raise LifecycleRefused(f"{label} must be finite" + (" and positive" if positive else ""))
    return float(value)


@dataclass(frozen=True)
class GrantReceipt:
    grant_id: str
    generation: int
    deadline: float
    clock_domain: str
    revoked: bool = False
    renewal_ok: bool = True

    def __post_init__(self) -> None:
        _text(self.grant_id, "grant_id")
        _positive(self.generation, "grant_generation")
        _finite(self.deadline, "grant deadline")
        _text(self.clock_domain, "grant clock_domain")
        if type(self.revoked) is not bool or type(self.renewal_ok) is not bool:
            raise LifecycleRefused("grant status flags must be boolean")

    def revalidated(self) -> "GrantReceipt":
        return GrantReceipt(self.grant_id, self.generation, self.deadline,
                            self.clock_domain, self.revoked, self.renewal_ok)


class OwnedContainer(Protocol):
    path: Path

    def create(self) -> None: ...
    def identity(self) -> Mapping[str, Any]: ...
    def add(self, pid: int) -> None: ...
    def pids(self) -> tuple[int, ...]: ...
    def populated(self) -> bool: ...
    def signal_all(self, signum: int, identities: Mapping[int, int]) -> bool: ...
    def kill(self) -> None: ...
    def wait_empty(self, timeout: float) -> bool: ...
    def close_and_remove(self) -> None: ...


@dataclass(frozen=True)
class AuthorizedLaunch:
    grant: GrantReceipt
    container_id: str
    container: OwnedContainer

    def __post_init__(self) -> None:
        if not isinstance(self.grant, GrantReceipt):
            raise LifecycleRefused("authorized launch grant is untyped")
        if (not isinstance(self.container_id, str)
                or not self.container_id.startswith("epyc-autokernel-")
                or "/" in self.container_id):
            raise LifecycleRefused("authorized launch container_id is invalid")
        required = ("create", "identity", "add", "pids", "populated", "signal_all",
                    "kill", "wait_empty", "close_and_remove")
        if not isinstance(getattr(self.container, "path", None), Path) \
                or any(not callable(getattr(self.container, name, None)) for name in required):
            raise LifecycleRefused("authorized launch container is malformed")
        object.__setattr__(self, "grant", self.grant.revalidated())


@dataclass(frozen=True)
class RecoveryInspection:
    status: str
    authorization: AuthorizedLaunch | None
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"absent_released", "exact", "unknown"}:
            raise LifecycleRefused("recovery inspection status is invalid")
        _text(self.reason, "recovery inspection reason")
        if (self.status == "exact") != isinstance(self.authorization, AuthorizedLaunch):
            raise LifecycleRefused("recovery inspection authorization disagrees with status")
