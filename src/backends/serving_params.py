"""Per-server serving parameters derived from what the stack already measures (UFH14-B1, F1).

Origin: DS41-C95 / UFH14-A1 (root handoff ``agentic-serving-harness-fixes.md``, defects
D1/D2; graded F12 run). llama-server sends neither headers nor tokens while it prefills,
so a client whose timeout is shorter than the SILENT prefill abandons the request — and
the server keeps prefilling it, so the retry doubles the load (D1). The F12 fix used
constants for one server (1 h idle timeout, compaction at 150k of 196,608). This module
derives the same numbers for ANY server, from two sources the orchestrator already has
and never from a per-model table:

* per-request context — ``ContextLimitResolver.limit_for_url(url).per_request_n_ctx``
  (``src/backends/context_limits.py``: live ``/props`` slot n_ctx, the v10 clamp and the
  model ``ctx_max`` cap, registry fallback);
* prefill rate — the serving-call records (``src/backends/serving_calls.py``,
  ``epyc.orchestrator.serving_call.v1``): ``timings.prompt_n`` / ``timings.prompt_ms`` /
  ``timings.cache_n`` of long (>= 8192 uncached tokens) prefills on the same port, from
  the CURRENT server launch when the launch sidecar names one. Fallback fields:
  ``result.prompt_eval_ms`` with ``result.prompt_tokens - result.cached_prompt_tokens``.

Prefill slows with context (850 -> 487 tok/s from 2k to 80k on the 27B, ~340 at 157k in
F12), so a sample measured at context ``s`` is scaled to a target context ``c > s`` by
``sqrt(s / c)`` and the low quantile is taken (contended samples make it lower still).

Derived numbers:

* ``prefill_allowance_s(prompt_tokens)`` — the per-call addition to a request's HTTP
  timeout: ``QUEUE_FACTOR x prompt / rate_at(prompt) x MARGIN`` for a prompt of at least
  ``MIN_PREFILL_SAMPLE_TOKENS``; None below it or when no rate is measured (the caller
  keeps today's timeout, and the record says ``unmeasured``). ``QUEUE_FACTOR`` = 2: one
  long prefill that slipped past the pool gate (another worker, a direct client) plus ours.
* ``idle_timeout_s`` — the worst-case silent interval for a WHOLE-window request, for
  clients that set one static idle timeout (passthrough read timeout, opencode / codex
  provider configs): same rule at ``c = per_request_n_ctx``, clamped to
  ``[MIN_IDLE_S, MAX_IDLE_S]``.
* ``compact_at`` — ``COMPACT_FRACTION x per_request_n_ctx`` (F12: 150,000 / 196,608 = 0.763),
  for a compactor that must fire with room for the compaction request itself.

Nothing here is a new table: the resolver holds a bounded cache of samples it re-reads
from the serving-call log every ``REFRESH_S`` (other uvicorn workers write there too) plus
the records this process writes (``observe_record``, fed by ``serving_calls.write_record``).
Every failure degrades to "unknown", never to an invented number.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.backends.context_limits import split_urls

log = logging.getLogger(__name__)

SCHEMA = "epyc.orchestrator.serving_params.v1"

#: A record is a prefill-rate sample only when it prefilled at least this many NEW
#: tokens: short prompts are dominated by fixed overhead and overstate the long rate.
#: The same bound gates the per-call allowance (shorter prompts: today's timeout).
MIN_PREFILL_SAMPLE_TOKENS = 8192
#: Fewer long samples than this on a port: the rate is unknown.
MIN_PREFILL_SAMPLES = 3
#: Low quantile of the context-scaled per-sample rates (the slow tail).
PREFILL_LOW_QUANTILE = 0.10
#: rate(c) = rate(s) * (s / c) ** exponent for c > s. Fit to the 27B: 487 tok/s at 80k
#: predicts ~350 at 157k (F12 measured ~340).
CONTEXT_SCALING_EXPONENT = 0.5
#: Extrapolate as little as possible: when at least MIN_PREFILL_SAMPLES samples ran at
#: >= NEAR_CONTEXT_FRACTION of the target context, only those are used (sqrt scaling from
#: 8k to 262k under-predicts ~2-3x: 850 tok/s at 2k -> ~96 at 157k vs ~340 measured).
NEAR_CONTEXT_FRACTION = 0.5
#: Silent interval = QUEUE_FACTOR prefills (one slipped past the gate + ours) x MARGIN.
QUEUE_FACTOR = 2.0
MARGIN = 1.25
#: Clamp of the whole-window idle timeout (client-facing static setting only).
MIN_IDLE_S = 600
MAX_IDLE_S = 14_400
#: Upper bound of a per-call allowance (a stuck server is still detected).
MAX_ALLOWANCE_S = MAX_IDLE_S
#: Auto-compaction threshold as a fraction of the per-request window.
COMPACT_FRACTION = 0.76

#: Samples kept per port, and how often the log tail is re-read.
MAX_SAMPLES_PER_PORT = 256
REFRESH_S = 600.0
LOG_TAIL_BYTES = 8 * 1024 * 1024
LOG_SHARDS = 2  # serving_calls.jsonl and .1: the newest records


@dataclass(frozen=True)
class PrefillSample:
    prompt_n: int     # uncached prompt tokens the server processed
    prompt_ms: float
    ctx: int          # cache_n + prompt_n: the context the prefill ran at
    launch_id: str | None = None
    record_id: str | None = None

    @property
    def rate(self) -> float:
        return self.prompt_n / (self.prompt_ms / 1000.0)


def _pos_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def sample_from_record(rec: Any) -> tuple[int, PrefillSample] | None:
    """``(port, sample)`` for a serving-call record that measured a long prefill, else None."""
    if not isinstance(rec, dict):
        return None
    server = rec.get("server") if isinstance(rec.get("server"), dict) else {}
    port = _pos_int(server.get("port"))
    if port is None:
        return None
    timings = rec.get("timings") if isinstance(rec.get("timings"), dict) else {}
    n = _pos_int(timings.get("prompt_n"))
    ms = timings.get("prompt_ms")
    cache_n = _pos_int(timings.get("cache_n")) or 0
    if n is None or not isinstance(ms, (int, float)) or isinstance(ms, bool) or ms <= 0:
        result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        total = _pos_int(result.get("prompt_tokens"))
        cached = _pos_int(result.get("cached_prompt_tokens")) or 0
        ms = result.get("prompt_eval_ms")
        n = (total - cached) if total and total > cached else None
        cache_n = cached
    if n is None or n < MIN_PREFILL_SAMPLE_TOKENS:
        return None
    if not isinstance(ms, (int, float)) or isinstance(ms, bool) or ms <= 0:
        return None
    launch_id = server.get("launch_id") if isinstance(server.get("launch_id"), str) else None
    record_id = rec.get("record_id") if isinstance(rec.get("record_id"), str) else None
    return port, PrefillSample(prompt_n=int(n), prompt_ms=float(ms), ctx=int(cache_n + n),
                               launch_id=launch_id, record_id=record_id)


def rate_at(samples: Sequence[PrefillSample], ctx: int) -> float | None:
    """Conservative prefill rate (tok/s) for a cold prefill at context ``ctx``: the
    samples near ``ctx`` when there are enough (``NEAR_CONTEXT_FRACTION``), else all;
    each scaled DOWN to ``ctx`` when it ran at a smaller context (never scaled up), then
    the ``PREFILL_LOW_QUANTILE``. None with fewer than ``MIN_PREFILL_SAMPLES``."""
    if len(samples) < MIN_PREFILL_SAMPLES or ctx <= 0:
        return None
    near = [s for s in samples if s.ctx >= NEAR_CONTEXT_FRACTION * ctx]
    if len(near) >= MIN_PREFILL_SAMPLES:
        samples = near
    rates = []
    for s in samples:
        scale = (s.ctx / ctx) ** CONTEXT_SCALING_EXPONENT if s.ctx < ctx else 1.0
        rates.append(s.rate * scale)
    rates.sort()
    return rates[min(len(rates) - 1, int(math.floor(PREFILL_LOW_QUANTILE * len(rates))))]


def silent_s(tokens: int, rate: float) -> float:
    """QUEUE_FACTOR prefills of ``tokens`` at ``rate``, times MARGIN."""
    return QUEUE_FACTOR * tokens / rate * MARGIN


@dataclass(frozen=True)
class ServingParams:
    """F1 for one server. None fields = unknown: leave that knob where it is."""

    url: str
    per_request_n_ctx: int | None
    ctx_source: str                  # ContextLimit.source, or "unknown"
    compact_at: int | None
    prefill_samples: int
    prefill_source: str              # "measured" | "unmeasured"
    window_prefill_tps: float | None
    idle_timeout_s: int | None       # whole-window silent bound, clamped; None = unmeasured
    samples: tuple[PrefillSample, ...] = field(default=(), repr=False, compare=False)

    def prefill_allowance_s(self, prompt_tokens: int) -> int | None:
        """Seconds to ADD to a call's timeout for its silent prefill; None = no change
        (short prompt, or no measured rate). The prompt is capped at the window: a larger
        one is refused by the server / the request cap, not timed."""
        tokens = int(prompt_tokens or 0)
        if tokens < MIN_PREFILL_SAMPLE_TOKENS:
            return None
        if self.per_request_n_ctx:
            tokens = min(tokens, int(self.per_request_n_ctx))
        rate = rate_at(self.samples, tokens)
        if not rate or rate <= 0:
            return None
        return int(min(MAX_ALLOWANCE_S, math.ceil(silent_s(tokens, rate))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "url": self.url,
            "per_request_n_ctx": self.per_request_n_ctx,
            "ctx_source": self.ctx_source,
            "compact_at": self.compact_at,
            "prefill_samples": self.prefill_samples,
            "prefill_source": self.prefill_source,
            "window_prefill_tps": (round(self.window_prefill_tps, 1)
                                   if self.window_prefill_tps else None),
            "idle_timeout_s": self.idle_timeout_s,
        }


def derive(url: str, *, per_request_n_ctx: int | None, ctx_source: str,
           samples: Sequence[PrefillSample]) -> ServingParams:
    """Pure: ServingParams for ``url`` from its window and its prefill samples."""
    window = _pos_int(per_request_n_ctx)
    tps = rate_at(samples, window) if window else None
    idle = None
    if window and tps:
        idle = int(min(MAX_IDLE_S, max(MIN_IDLE_S, math.ceil(silent_s(window, tps)))))
    return ServingParams(
        url=url,
        per_request_n_ctx=window,
        ctx_source=ctx_source if window else "unknown",
        compact_at=int(window * COMPACT_FRACTION) if window else None,
        prefill_samples=len(samples),
        prefill_source="measured" if tps else "unmeasured",
        window_prefill_tps=tps,
        idle_timeout_s=idle,
        samples=tuple(samples),
    )


def _tail_lines(path: Path, max_bytes: int = LOG_TAIL_BYTES) -> list[str]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = data.splitlines()
    return lines[1:] if size > max_bytes else lines  # the first line may be cut


def _default_record_lines() -> Iterable[str]:
    from src.backends import serving_calls

    path = serving_calls.log_path()
    if path is None:
        return []
    shards = [path] + [path.with_name(f"{path.name}.{i}") for i in range(1, LOG_SHARDS)]
    out: list[str] = []
    for shard in reversed(shards):  # oldest first, so the newest win the deque
        out.extend(_tail_lines(shard))
    return out


def _default_launch_id(port: int) -> str | None:
    from src.backends import serving_calls

    ident = serving_calls.server_identity(port)
    value = ident.get("launch_id")
    return value if isinstance(value, str) else None


class ServingParamsResolver:
    """Per-URL ServingParams: context from ContextLimitResolver, rate from serving records."""

    def __init__(
        self,
        *,
        context_resolver: Any = None,
        record_lines: Callable[[], Iterable[str]] | None = None,
        launch_id: Callable[[int], str | None] | None = None,
        refresh_s: float = REFRESH_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._context_resolver = context_resolver
        self._record_lines = record_lines or _default_record_lines
        self._launch_id = launch_id or _default_launch_id
        self._refresh_s = max(0.0, float(refresh_s))
        self._clock = clock
        self._lock = threading.Lock()
        self._samples: dict[int, deque[PrefillSample]] = {}
        self._seen: set[str] = set()
        self._loaded_at: float | None = None

    # -- samples ----------------------------------------------------------
    def _add(self, port: int, sample: PrefillSample) -> None:
        if sample.record_id:
            if sample.record_id in self._seen:
                return
            self._seen.add(sample.record_id)
        self._samples.setdefault(port, deque(maxlen=MAX_SAMPLES_PER_PORT)).append(sample)

    def observe_record(self, record: Any) -> None:
        """Feed one serving-call record (``serving_calls.write_record``). Never raises."""
        try:
            hit = sample_from_record(record)
            if hit is not None:
                with self._lock:
                    self._add(*hit)
        except Exception:
            log.debug("serving params: observe failed", exc_info=True)

    def _refresh_if_due(self) -> None:
        now = self._clock()
        with self._lock:
            if self._loaded_at is not None and now - self._loaded_at < self._refresh_s:
                return
            self._loaded_at = now
        try:
            lines = list(self._record_lines())
        except Exception:
            log.debug("serving params: record log unreadable", exc_info=True)
            return
        parsed = []
        for line in lines:
            try:
                hit = sample_from_record(json.loads(line))
            except ValueError:
                continue
            if hit is not None:
                parsed.append(hit)
        with self._lock:
            for port, sample in parsed:
                self._add(port, sample)
            if len(self._seen) > 16 * MAX_SAMPLES_PER_PORT:
                keep = {s.record_id for dq in self._samples.values() for s in dq if s.record_id}
                self._seen = keep

    def samples_for_port(self, port: int | None) -> list[PrefillSample]:
        """This port's samples from the CURRENT server launch when the stack sidecar
        names one (a relaunch with other flags/binary measures afresh), else all."""
        if port is None:
            return []
        self._refresh_if_due()
        with self._lock:
            samples = list(self._samples.get(port, ()))
        try:
            current = self._launch_id(port)
        except Exception:
            current = None
        if current:
            samples = [s for s in samples if s.launch_id in (None, current)]
        return samples

    # -- lookups ----------------------------------------------------------
    def _contexts(self):
        if self._context_resolver is not None:
            return self._context_resolver
        from src.backends.context_limits import get_context_limit_resolver

        return get_context_limit_resolver()

    def for_url(self, url: str | None) -> ServingParams | None:
        url = (split_urls(url) or [""])[0]
        if not url:
            return None
        from urllib.parse import urlparse

        try:
            port = urlparse(url).port
        except ValueError:
            port = None
        try:
            limit = self._contexts().limit_for_url(url)
        except Exception:
            limit = None
        return derive(
            url,
            per_request_n_ctx=getattr(limit, "per_request_n_ctx", None),
            ctx_source=str(getattr(limit, "source", "unknown")),
            samples=self.samples_for_port(port),
        )

    def prefill_allowance(self, urls: str | Sequence[str] | None,
                          prompt_tokens: int) -> dict[str, Any] | None:
        """The binding (largest) per-call allowance across ``urls`` (a fleet role may land
        on any instance), as a record-ready dict; None when no instance can say."""
        if isinstance(urls, str) or urls is None:
            urls = split_urls(urls)
        if int(prompt_tokens or 0) < MIN_PREFILL_SAMPLE_TOKENS:
            return None
        best: dict[str, Any] | None = None
        unmeasured: list[str] = []
        for url in urls:
            params = self.for_url(url)
            if params is None:
                continue
            allowance = params.prefill_allowance_s(prompt_tokens)
            if allowance is None:
                unmeasured.append(params.url)
                continue
            if best is None or allowance > best["allowance_s"]:
                best = {"allowance_s": allowance, "url": params.url,
                        "prompt_tokens_est": int(prompt_tokens),
                        "prefill_samples": params.prefill_samples,
                        "per_request_n_ctx": params.per_request_n_ctx}
        if best is None:
            return {"allowance_s": 0, "prefill_source": "unmeasured",
                    "prompt_tokens_est": int(prompt_tokens), "urls": unmeasured} if unmeasured else None
        best["prefill_source"] = "measured"
        if unmeasured:
            best["unmeasured_urls"] = unmeasured
        return best


_resolver: ServingParamsResolver | None = None
_resolver_lock = threading.Lock()


def get_serving_params_resolver() -> ServingParamsResolver:
    global _resolver
    with _resolver_lock:
        if _resolver is None:
            _resolver = ServingParamsResolver()
        return _resolver


def set_serving_params_resolver(resolver: ServingParamsResolver | None) -> None:
    """Install (or with None, reset) the process-wide resolver. For tests."""
    global _resolver
    with _resolver_lock:
        _resolver = resolver


def observe_record(record: Any) -> None:
    """Module-level feed for ``serving_calls.write_record``; never raises, never
    instantiates the resolver just to feed it (no resolver yet = nothing reads yet,
    and the first read loads the log tail anyway)."""
    resolver = _resolver
    if resolver is not None:
        resolver.observe_record(record)


# -- F2: per-turn thinking budget (llama-server request fields) ------------------------
#: F12: what the server injects into the reasoning when a turn's budget is spent.
THINK_BUDGET_MESSAGE = ("\n\n[Thinking budget for this turn is used up. Stop deliberating now "
                        "and act on your best current conclusion.]\n")


def thinking_budget_fields(tokens: int, message: str | None = THINK_BUDGET_MESSAGE) -> dict[str, Any]:
    """``thinking_budget_tokens`` (+ ``reasoning_budget_message``) for one chat-lane body;
    llama-server v10 honours both on /v1/chat/completions and /v1/responses. {} for 0."""
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError(f"thinking budget must be a non-negative int, got {tokens!r}")
    if tokens == 0:
        return {}
    out: dict[str, Any] = {"thinking_budget_tokens": tokens}
    if message:
        out["reasoning_budget_message"] = message
    return out
