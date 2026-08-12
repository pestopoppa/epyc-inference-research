#!/usr/bin/env python3
"""Fail-closed identity and installation roots for external kernel providers.

Provider implementations may be useful search oracles without being eligible
AutoKernel source.  This module owns the filesystem half of that boundary: a
provider prefix is an isolated candidate location, never a shared system ROCm
installation and never one of the frozen production kernel trees.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import worktree


class ProviderIsolationError(ValueError):
    """A provider prefix could alter shared or production state."""


PROHIBITED_PROVIDER_PREFIXES = (
    "/opt/rocm",
    "/usr",
    *worktree.PRODUCTION_TREES,
    *worktree.PRODUCTION_TREE_ALIASES,
)


def _under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


@dataclass(frozen=True)
class IsolatedProviderPrefix:
    """An absolute provider root proven outside shared and frozen trees."""

    path: str

    @classmethod
    def create(cls, path: str, *, prohibited: Iterable[str] =
               PROHIBITED_PROVIDER_PREFIXES) -> "IsolatedProviderPrefix":
        if not isinstance(path, str) or not path or not os.path.isabs(path):
            raise ProviderIsolationError("provider isolation root must be absolute")
        resolved = os.path.realpath(path)
        if resolved == os.path.sep:
            raise ProviderIsolationError("filesystem root is not an isolated provider prefix")
        blocked = tuple(os.path.realpath(item) for item in prohibited)
        matches = tuple(root for root in blocked
                        if _under(resolved, root) or _under(root, resolved))
        if matches:
            raise ProviderIsolationError(
                f"provider isolation root {resolved!r} overlaps prohibited prefix "
                f"{matches[0]!r}")
        return cls(resolved)

    def child(self, *parts: str) -> Path:
        child = Path(self.path, *parts).resolve(strict=False)
        if not _under(str(child), self.path):
            raise ProviderIsolationError("provider child path escapes its isolated prefix")
        return child


__all__ = [
    "IsolatedProviderPrefix", "PROHIBITED_PROVIDER_PREFIXES",
    "ProviderIsolationError",
]
