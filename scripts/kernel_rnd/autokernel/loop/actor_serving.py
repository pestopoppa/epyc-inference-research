"""Client-side serving parameters for an actor talking to a LOCAL llama-server (UFH14-B1, F1).

Origin: DS41-C95 / UFH14-A1 (agentic-serving-harness-fixes.md D1, D2; graded F12 run,
`/mnt/raid0/llm/tmp/ds41-c95/grading/report.md`). The 27B harness failures were mostly
client/serving defects, not model capability:

* D1 -- the client's stream idle timeout (codex ~300 s; opencode 1.18.31's provider
  `headerTimeout` and `chunkTimeout` both default to 300000 ms) is shorter than a long
  SILENT prefill. llama-server sends neither headers nor SSE chunks while it prefills,
  so a 150k-token cold prefill (~460 s on the 27B) is aborted by the client, the server
  keeps prefilling the abandoned request, and the retry doubles the load.
* The client's context window must be the server's real per-request context, and
  auto-compaction must fire well below it (F12: 150k of 196,608).

The F12 fix used constants (1 h idle timeout, 150k compaction) for ONE server. This module
derives the same three numbers for ANY local server from what the stack already measures,
never from a per-model table:

* per-request context -- the server's own `GET /props`
  (`default_generation_settings.n_ctx`), the same live source the orchestrator's
  `src/backends/context_limits.py` reads first. v10 clamps every slot to `n_ctx_train`
  (`server-context.cpp:1316-1322`), so this IS the per-request cap context_limits
  reports (262144 on :8083 after STACKCHG-KVPOOL-20261003);
* prefill rate -- the orchestrator's serving-call records
  (`<orchestrator>/logs/serving_calls/serving_calls.jsonl`, `epyc.orchestrator.
  serving_call.v1`, UFH14-B5): `timings.prompt_n` / `timings.prompt_ms` of long
  (>= 8192-token) prefills on the same port, scaled down to the full window and taken
  at a low quantile (prefill slows with context: 850 -> 487 tok/s from 2k to 80k on the
  27B, ~340 tok/s at 157k in F12);
* idle timeout = (one queued long prefill ahead of us + our own full-window prefill)
  x margin. Since UFH14-B2 the orchestrator's pool gate lets one long prefill run per
  server and holds the rest, so a request can sit silent behind one other.

With no measured rate (too few long records yet) the idle timeout falls back to the
ceiling (4 h), and the derivation says so: F12's 1 h was proven on the GPU 27B, and an
unmeasured server (a CPU one at 262k context) may prefill slower than 1 h covers. A long
idle timeout costs little -- the actor's own wall budget and hard timeout bound the call
-- while a short one is exactly D1. Non-local providers (hosted APIs) get nothing
from here: their clients' own defaults stay.

A per-call answer is cached for `CACHE_TTL_S` per base URL. Every failure degrades to
"unknown" (no change to that knob), never to an invented number.

CLI (prepares static provider configs; reads /props unless --offline):
    python -m scripts.kernel_rnd.autokernel.loop.actor_serving --base-url http://127.0.0.1:8083/v1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

#: Hosts treated as a local llama-server. Anything else is a hosted API: no F1 change.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

#: Where the orchestrator writes serving-call records (UFH14-B5); rotated `.1` .. `.N`.
ORCHESTRATOR_ROOT_ENV = "AK_ORCHESTRATOR_ROOT"
DEFAULT_ORCHESTRATOR_ROOT = "/mnt/raid0/llm/epyc-orchestrator"
SERVING_RECORDS_REL = "logs/serving_calls/serving_calls.jsonl"
SERVING_RECORDS_ENV = "AK_SERVING_CALLS_LOG"
#: The live file plus one rotated shard, and only each one's tail: the newest records
#: matter most, and the read stays cheap (<= 16 MB, once per CACHE_TTL_S per server).
SERVING_RECORDS_MAX_SHARDS = 1
SERVING_RECORDS_TAIL_BYTES = 8 * 1024 * 1024

#: A record is a prefill-rate sample only when it prefilled at least this many tokens:
#: short prompts are dominated by fixed overhead and overstate the long-prompt rate.
MIN_PREFILL_SAMPLE_TOKENS = 8192
#: Fewer long samples than this: the rate is "unknown" (fallback idle timeout).
MIN_PREFILL_SAMPLES = 3
#: The low quantile of the window-scaled per-sample rates (conservative: the slow tail).
PREFILL_LOW_QUANTILE = 0.10
#: Prefill throughput falls with context. A sample measured at context c is scaled to
#: the full window W by sqrt(c / W) (c < W). Fit to the 27B: 487 tok/s at 80k predicts
#: ~350 at 157k (F12 measured ~340); 850 at 2k under-predicts (conservative).
CONTEXT_SCALING_EXPONENT = 0.5

#: Silent interval = QUEUE_FACTOR full-window prefills (one held behind the pool gate's
#: single long prefill, plus our own), times MARGIN.
QUEUE_FACTOR = 2.0
MARGIN = 1.25
MIN_IDLE_S = 600
MAX_IDLE_S = 14_400
#: Used only when no prefill rate is measured: the ceiling (see the module docstring;
#: F12's proven 1 h is a GPU-27B number and would be a constant in disguise).
FALLBACK_IDLE_S = MAX_IDLE_S

#: Auto-compaction threshold as a fraction of the per-request window (F12: 150,000 of
#: 196,608 = 0.763). Low enough that the compaction request itself plus one step's
#: tool output stays inside the window.
COMPACT_FRACTION = 0.76

PROPS_TIMEOUT_S = 2.0
CACHE_TTL_S = 300.0

#: Codex's default provider stream idle timeout and opencode 1.18.31's provider
#: `headerTimeout` / `chunkTimeout` defaults (read from the installed binary).
CLIENT_DEFAULT_IDLE_MS = 300_000

_CACHE: dict[str, tuple[float, "ServingParams"]] = {}
_CACHE_LOCK = threading.Lock()


# ------------------------------------------------------------------------------------
# facts
# ------------------------------------------------------------------------------------
def server_root(base_url: str) -> str:
    """`http://h:p/v1` -> `http://h:p` (llama-server's /props and /slots live at the root)."""
    url = str(base_url).rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def port_of(base_url: str) -> int | None:
    try:
        return urlparse(str(base_url)).port
    except ValueError:
        return None


def is_local(base_url: str | None) -> bool:
    """True for a llama-server on this host (loopback base URL)."""
    if not base_url:
        return False
    try:
        host = urlparse(str(base_url)).hostname
    except ValueError:
        return False
    return host in LOOPBACK_HOSTS


def fetch_props(base_url: str, timeout_s: float = PROPS_TIMEOUT_S) -> dict | None:
    """`GET <root>/props` (no inference, no slot). None on any failure."""
    try:
        with urllib.request.urlopen(server_root(base_url) + "/props", timeout=timeout_s) as resp:
            body = json.loads(resp.read())
    except Exception:  # noqa: BLE001 -- a missing fact is "unknown", never an error
        return None
    return body if isinstance(body, dict) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def per_request_ctx(props: Mapping[str, Any] | None) -> int | None:
    """The per-request context from a `/props` body: `default_generation_settings.n_ctx`
    (the slot n_ctx, already clamped to n_ctx_train by v10), as context_limits reads it."""
    if not isinstance(props, Mapping):
        return None
    settings = props.get("default_generation_settings")
    return _positive_int(settings.get("n_ctx")) if isinstance(settings, Mapping) else None


def serving_record_paths(root: str | Path | None = None) -> list[Path]:
    """The serving-call log and its rotated shards, newest first."""
    override = os.environ.get(SERVING_RECORDS_ENV, "").strip()
    if override:
        base = Path(override)
    else:
        base = Path(root or os.environ.get(ORCHESTRATOR_ROOT_ENV) or DEFAULT_ORCHESTRATOR_ROOT
                    ) / SERVING_RECORDS_REL
    paths = [base] + [base.with_name(f"{base.name}.{i}")
                      for i in range(1, SERVING_RECORDS_MAX_SHARDS + 1)]
    return [p for p in paths if p.is_file()]


def _tail_lines(path: Path, max_bytes: int = SERVING_RECORDS_TAIL_BYTES) -> list[str]:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = data.splitlines()
    return lines[1:] if size > max_bytes else lines   # the first line may be cut


@dataclass(frozen=True)
class PrefillSample:
    prompt_n: int
    prompt_ms: float
    ctx: int          # cache_n + prompt_n: the context the prefill ran at

    @property
    def rate(self) -> float:
        return self.prompt_n / (self.prompt_ms / 1000.0)


def prefill_samples(port: int | None, lines: Iterable[str]) -> list[PrefillSample]:
    """Long-prefill samples for `port` from serving-call record lines (bad lines skipped).
    Primary fields: `timings.prompt_n` / `timings.prompt_ms` / `timings.cache_n` (the
    server's own); fallback `result.prompt_eval_ms` with `result.prompt_tokens -
    result.cached_prompt_tokens`."""
    out: list[PrefillSample] = []
    if port is None:
        return out
    needle = str(port)
    for line in lines:
        if needle not in line:   # cheap pre-filter before parsing
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        server = rec.get("server") if isinstance(rec.get("server"), dict) else {}
        caller = rec.get("caller") if isinstance(rec.get("caller"), dict) else {}
        rec_port = server.get("port") or caller.get("port")
        if rec_port is None and server.get("base_url"):
            rec_port = port_of(server["base_url"])
        if _positive_int(rec_port) != port:
            continue
        timings = rec.get("timings") if isinstance(rec.get("timings"), dict) else {}
        n = _positive_int(timings.get("prompt_n"))
        ms = timings.get("prompt_ms")
        cache_n = _positive_int(timings.get("cache_n")) or 0
        if n is None or not isinstance(ms, (int, float)) or ms <= 0:
            result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            total = _positive_int(result.get("prompt_tokens"))
            cached = _positive_int(result.get("cached_prompt_tokens")) or 0
            ms = result.get("prompt_eval_ms")
            n = (total - cached) if total and total > cached else None
            cache_n = cached
        if n is None or n < MIN_PREFILL_SAMPLE_TOKENS or not isinstance(ms, (int, float)) or ms <= 0:
            continue
        out.append(PrefillSample(prompt_n=int(n), prompt_ms=float(ms), ctx=int(cache_n + n)))
    return out


def window_prefill_tps(samples: Sequence[PrefillSample], window: int | None) -> float | None:
    """A conservative prefill rate for a full-`window` cold prefill: each sample's rate
    scaled to the window (`CONTEXT_SCALING_EXPONENT`), then the low quantile. None with
    fewer than `MIN_PREFILL_SAMPLES` samples."""
    if len(samples) < MIN_PREFILL_SAMPLES:
        return None
    # Prefer samples near the window: scaling an 8k sample up to 262k under-predicts the
    # rate ~3x (850 tok/s at 2k -> ~96 at 157k; F12 measured ~340). Same rule as the
    # orchestrator proposal (serving_params.py).
    near = [s for s in samples if window and s.ctx >= window / 2]
    if len(near) >= MIN_PREFILL_SAMPLES:
        samples = near
    rates = []
    for s in samples:
        scale = 1.0
        if window and s.ctx < window:
            scale = (s.ctx / window) ** CONTEXT_SCALING_EXPONENT
        rates.append(s.rate * scale)
    rates.sort()
    idx = min(len(rates) - 1, int(math.floor(PREFILL_LOW_QUANTILE * len(rates))))
    return rates[idx]


# ------------------------------------------------------------------------------------
# derived parameters
# ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ServingParams:
    """F1 for one local server. `context_window` / `compact_at` None = unknown (leave the
    client's own setting). `idle_timeout_ms` is always set for a local server."""
    base_url: str
    idle_timeout_ms: int
    context_window: int | None
    compact_at: int | None
    prefill_tps: float | None
    prefill_samples: int
    ctx_source: str          # "props" | "given" | "unknown"
    idle_source: str         # "measured" | "fallback_unmeasured"
    derived_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                   time.gmtime()))

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        if out["prefill_tps"] is not None:
            out["prefill_tps"] = round(out["prefill_tps"], 1)
        return out


def idle_timeout_s(window: int | None, prefill_tps: float | None) -> tuple[int, str]:
    """(seconds, source). Measured: QUEUE_FACTOR full-window prefills x MARGIN, clamped
    to [MIN_IDLE_S, MAX_IDLE_S]; else the unmeasured fallback."""
    if not window or not prefill_tps or prefill_tps <= 0:
        return FALLBACK_IDLE_S, "fallback_unmeasured"
    silent = QUEUE_FACTOR * window / prefill_tps
    return int(min(MAX_IDLE_S, max(MIN_IDLE_S, math.ceil(silent * MARGIN)))), "measured"


def compact_threshold(window: int | None) -> int | None:
    return int(window * COMPACT_FRACTION) if window else None


def derive(base_url: str, *, props: Mapping[str, Any] | None = None,
           n_ctx: int | None = None, record_lines: Iterable[str] | None = None,
           fetch: bool = True) -> ServingParams | None:
    """F1 for `base_url`, or None for a non-local provider. `props` / `n_ctx` /
    `record_lines` inject facts (tests, the CLI's --offline); otherwise /props is read
    live (when `fetch`) and the serving records from disk."""
    if not is_local(base_url):
        return None
    window, ctx_source = None, "unknown"
    if n_ctx:
        window, ctx_source = int(n_ctx), "given"
    else:
        body = props if props is not None else (fetch_props(base_url) if fetch else None)
        window = per_request_ctx(body)
        ctx_source = "props" if window else "unknown"
    if record_lines is None:
        record_lines = [line for path in serving_record_paths() for line in _tail_lines(path)]
    samples = prefill_samples(port_of(base_url), record_lines)
    tps = window_prefill_tps(samples, window)
    idle_s, idle_source = idle_timeout_s(window, tps)
    return ServingParams(base_url=str(base_url).rstrip("/"), idle_timeout_ms=idle_s * 1000,
                         context_window=window, compact_at=compact_threshold(window),
                         prefill_tps=tps, prefill_samples=len(samples),
                         ctx_source=ctx_source, idle_source=idle_source)


def resolve(base_url: str | None, *, ttl_s: float = CACHE_TTL_S) -> ServingParams | None:
    """`derive` cached per base URL for `ttl_s` (an actor call reads it once at launch)."""
    if not is_local(base_url):
        return None
    key = str(base_url).rstrip("/")
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] < ttl_s:
            return hit[1]
    params = derive(key)
    with _CACHE_LOCK:
        _CACHE[key] = (now, params)
    return params


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# ------------------------------------------------------------------------------------
# per-client projections
# ------------------------------------------------------------------------------------
#: opencode 1.18.31's `ProviderTransform.maxOutputTokens` default (limit.output 0).
OPENCODE_DEFAULT_OUTPUT = 32_000


def opencode_provider_options(model: str, params: ServingParams | None) -> dict:
    """The per-call config block that raises opencode's provider `headerTimeout` and
    `chunkTimeout` (both 300000 ms by default; llama-server sends no header and no chunk
    while it prefills) to the derived idle timeout. Deep-merged over the global
    provider entry, so its npm/baseURL stay. {} when there is nothing to apply."""
    if params is None or not model or "/" not in model:
        return {}
    provider = model.split("/", 1)[0]
    ms = int(params.idle_timeout_ms)
    return {"provider": {provider: {"options": {"headerTimeout": ms, "chunkTimeout": ms}}}}


def opencode_limits(params: ServingParams | None, *, context_limit: int,
                    output_limit: int) -> tuple[int, int]:
    """(context_limit, output_limit) for opencode under F1. opencode compacts at
    `limit.context - maxOutputTokens`, so: C never exceeds the server window, and the
    compaction point C - O never exceeds `compact_at`. A configured C at or under both
    bounds is kept as is (an operator pool share wins); C = 0 (opencode's "never
    compact proactively") becomes the derived one. Unknown window: unchanged."""
    if params is None or not params.context_window:
        return context_limit, output_limit
    window, compact_at = int(params.context_window), int(params.compact_at or 0)
    out = int(output_limit)
    effective_out = out or OPENCODE_DEFAULT_OUTPUT
    ctx = int(context_limit) or window
    ctx = min(ctx, window)
    if compact_at and ctx - effective_out > compact_at:
        ctx = min(window, compact_at + effective_out)
    if out and out >= ctx:
        out = max(1, ctx // 4)
    return ctx, out


def codex_config_overrides(params: ServingParams | None, provider_id: str) -> list[str]:
    """`codex exec -c` overrides for a codex model provider pointed at a local server
    (the exact F12 knobs: `model_providers.<id>.stream_idle_timeout_ms`,
    `model_context_window`, `model_auto_compact_token_limit`). [] for none."""
    if params is None:
        return []
    out = ["-c", f"model_providers.{provider_id}.stream_idle_timeout_ms={int(params.idle_timeout_ms)}"]
    if params.context_window:
        out += ["-c", f"model_context_window={int(params.context_window)}"]
    if params.compact_at:
        out += ["-c", f"model_auto_compact_token_limit={int(params.compact_at)}"]
    return out


# ------------------------------------------------------------------------------------
# F2 per-turn thinking budget (llama-server request fields)
# ------------------------------------------------------------------------------------
#: F12: the message the server injects when a turn's reasoning budget is spent.
THINK_BUDGET_MESSAGE = ("\n\n[Thinking budget for this turn is used up. Stop deliberating now "
                        "and act on your best current conclusion: call one tool, or write "
                        "the requested JSON.]\n")


def think_budget_fields(tokens: int, message: str = THINK_BUDGET_MESSAGE) -> dict:
    """The llama-server request fields of the F12 per-turn thinking cap
    (`thinking_budget_tokens` + `reasoning_budget_message`; v10 honours both on
    /v1/chat/completions and /v1/responses). {} for tokens <= 0."""
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError(f"think budget must be a non-negative int, got {tokens!r}")
    if tokens == 0:
        return {}
    return {"thinking_budget_tokens": tokens, **({"reasoning_budget_message": message}
                                                  if message else {})}


def opencode_think_budget(model: str, tokens: int) -> dict:
    """The per-call config block putting the thinking cap in every request BODY of the
    call: the model's `options` (unknown keys pass through `@ai-sdk/openai-compatible`
    into the body, as `chat_template_kwargs` does, see `actor_opencode_config`)."""
    fields = think_budget_fields(tokens)
    if not fields:
        return {}
    if not model or "/" not in model:
        raise ValueError(f"think budget needs a provider/model id, got {model!r}")
    provider, model_id = model.split("/", 1)
    return {"provider": {provider: {"models": {model_id: {"options": fields}}}}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--n-ctx", type=int, default=0, help="skip /props: use this window")
    ap.add_argument("--offline", action="store_true", help="never read /props")
    ap.add_argument("--opencode-model", default="", help="also print the opencode block")
    ap.add_argument("--codex-provider", default="", help="also print the codex -c overrides")
    args = ap.parse_args(argv)
    params = derive(args.base_url, n_ctx=args.n_ctx or None, fetch=not args.offline)
    out: dict[str, Any] = {"params": params.to_dict() if params else None}
    if args.opencode_model:
        out["opencode"] = opencode_provider_options(args.opencode_model, params)
    if args.codex_provider:
        out["codex"] = codex_config_overrides(params, args.codex_provider)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
