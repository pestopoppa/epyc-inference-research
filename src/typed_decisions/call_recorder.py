"""Per-call accounting for typed-decision measurements (TD-29.M0).

The pilots used to read ``_last_inference_meta`` once per case, after the
runner returned, so a JSON-mode case that took a corrective retry reported
only its LAST call's tokens and no prompt-side numbers at all
(``tool_args_pilot.py`` per-case ``tokens``, TD-4 receipt re-read 2026-09-29).
``record_llm_calls`` fixes that at the seam: it wraps ONE primitives
instance's ``llm_call`` for the duration of a ``with`` block and snapshots the
inference meta immediately after every call, so a case's cost can be split
into decode tokens, extra calls and prefill.

Fields per call (``None`` when the backend did not report them — absent
telemetry is never recorded as ``0``):

* ``tokens`` — generated tokens (``meta["tokens"]``);
* ``prompt_tokens`` — the server's own ``usage.prompt_tokens`` (whole prompt);
* ``cache_n`` — prompt tokens served from the KV cache
  (``meta["cached_prompt_tokens"]``: ``usage.prompt_tokens_details.cached_tokens``
  or ``timings.cache_n``);
* ``prompt_n`` — prompt tokens actually evaluated, ``prompt_tokens - cache_n``
  (llama-server's ``timings.prompt_n``); ``None`` unless both are known;
* ``prompt_ms`` / ``gen_ms`` / ``elapsed_ms`` / ``completion_reason``.

The server only reports the prompt-side numbers on the ``/v1`` chat path
(``inference.py`` sets ``prompt_tokens``/``cached_prompt_tokens`` only when a
``chat_payload`` is present); roles on the raw ``/completion`` path record
``None`` there. The native runner's ``/tokenize`` requests are not
``llm_call``s and are not counted.

The wrapper is an INSTANCE attribute, so ``isinstance(primitives,
LLMPrimitives)`` and every other attribute stay untouched, and it is removed on
exit even when the block raises. Not thread-safe by design: the pilots call
serially, and the meta getter is per-context anyway.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CallLog", "record_llm_calls", "read_last_meta"]

_MISSING = object()


def read_last_meta(primitives: Any) -> dict[str, Any] | None:
    """Return the most recent call's inference meta, per-call-safe when possible."""
    getter = getattr(primitives, "get_last_inference_meta", None)
    meta = getter() if callable(getter) else getattr(primitives, "_last_inference_meta", None)
    if not isinstance(meta, Mapping):
        return None
    return {str(key): value for key, value in meta.items()}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _call_record(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    meta = meta or {}
    prompt_tokens = _number(meta.get("prompt_tokens"))
    cache_n = _number(meta.get("cached_prompt_tokens"))
    prompt_n = (
        max(0.0, prompt_tokens - cache_n)
        if prompt_tokens is not None and cache_n is not None
        else None
    )
    reason = meta.get("completion_reason")
    return {
        "tokens": _number(meta.get("tokens")),
        "prompt_tokens": prompt_tokens,
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        "prompt_ms": _number(meta.get("prompt_ms")),
        "gen_ms": _number(meta.get("gen_ms")),
        "elapsed_ms": _number(meta.get("elapsed_ms")),
        "completion_reason": reason if isinstance(reason, str) else None,
        "meta_present": bool(meta),
    }


def _sum(values: list[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


@dataclass
class CallLog:
    """Calls observed inside one ``record_llm_calls`` block, in call order."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    errors: int = 0

    def summary(self) -> dict[str, Any]:
        """Totals over every call. Sums skip unknowns; ``*_missing`` counts them."""

        def column(name: str) -> list[float | None]:
            return [call[name] for call in self.calls]

        return {
            "call_count": len(self.calls),
            "call_errors": self.errors,
            "tokens_total": _sum(column("tokens")),
            "tokens_last_call": self.calls[-1]["tokens"] if self.calls else None,
            "prompt_tokens_total": _sum(column("prompt_tokens")),
            "cache_n_total": _sum(column("cache_n")),
            "prompt_n_total": _sum(column("prompt_n")),
            "prompt_ms_total": _sum(column("prompt_ms")),
            "gen_ms_total": _sum(column("gen_ms")),
            "tokens_missing": sum(1 for value in column("tokens") if value is None),
            "prompt_n_missing": sum(1 for value in column("prompt_n") if value is None),
        }


@contextlib.contextmanager
def record_llm_calls(primitives: Any) -> Iterator[CallLog]:
    """Wrap ``primitives.llm_call`` and log every call's meta until the block exits.

    A call that raises is still logged (its meta is whatever the backend left,
    usually the previous call's, so it is recorded with ``error=True`` and no
    numbers) and the exception propagates unchanged.
    """
    log = CallLog()
    original = getattr(primitives, "llm_call", None)
    if not callable(original):
        yield log
        return
    instance_attrs = getattr(primitives, "__dict__", {})
    shadowed = instance_attrs.get("llm_call", _MISSING)

    def recording_llm_call(*args: Any, **kwargs: Any) -> Any:
        try:
            result = original(*args, **kwargs)
        except Exception:
            log.errors += 1
            record = _call_record(None)
            record["error"] = True
            log.calls.append(record)
            raise
        record = _call_record(read_last_meta(primitives))
        record["error"] = False
        log.calls.append(record)
        return result

    primitives.llm_call = recording_llm_call
    try:
        yield log
    finally:
        if shadowed is _MISSING:
            with contextlib.suppress(AttributeError):
                del primitives.llm_call
        else:
            primitives.llm_call = shadowed
