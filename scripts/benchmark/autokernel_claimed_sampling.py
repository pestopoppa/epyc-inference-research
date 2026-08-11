#!/usr/bin/env python3
"""Fail-closed teardown for a sampled AutoKernel device claim."""
from __future__ import annotations

from typing import Any


def stop_sampler_and_release(*, sampler: Any, claim: Any) -> tuple[Any, Any, tuple[BaseException, ...]]:
    """Stop sampling and release independently so either failure cannot mask the other."""
    sampling_receipt = None
    released_receipt = None
    errors: list[BaseException] = []
    if sampler is not None:
        try:
            sampling_receipt = sampler.stop()
        except BaseException as exc:
            errors.append(exc)
    try:
        released_receipt = claim.release()
    except BaseException as exc:
        errors.append(exc)
    return sampling_receipt, released_receipt, tuple(errors)


def error_payload(errors: tuple[BaseException, ...]) -> list[dict[str, str]]:
    return [{"type": type(exc).__name__, "message": str(exc)} for exc in errors]
