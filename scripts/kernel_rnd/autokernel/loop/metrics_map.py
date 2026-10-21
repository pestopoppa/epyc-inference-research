"""AKR-SG-2 follow-up: map llama-server ``/metrics`` (Prometheus text) into the paired serving-gate report.

The names are the ones THIS fork's server actually exports (read from ``tools/server`` of the v10 tree,
2026-10-21): the ``llamacpp:`` counters/gauges below. The server must be started with ``--metrics``.
Not exported by this tree: a KV-cache usage gauge (``kv_cache_usage_ratio`` / ``kv_cache_tokens`` exist
upstream but are NOT in v10); the report therefore carries ``kv`` as ``"not_exposed"`` and NEVER zero.

  queue series   requests_processing (gauge), requests_deferred (gauge)   -> ``queue``
  throughput     prompt_tokens_total / tokens_predicted_total (counters),
                 *_seconds_total (counters)                               -> window deltas
  batching       n_decode_total (counter), n_busy_slots_per_decode (gauge), n_tokens_max (gauge)
Counters are reported as deltas between two scrapes; gauges as the scrape values (start/end/max).
Standard library only.
"""
from __future__ import annotations

import re
import urllib.request

PREFIX = "llamacpp:"
COUNTERS = ("prompt_tokens_total", "prompt_seconds_total", "tokens_predicted_total", "tokens_predicted_seconds_total", "n_decode_total")
GAUGES = ("requests_processing", "requests_deferred", "n_busy_slots_per_decode", "n_tokens_max", "prompt_tokens_seconds", "predicted_tokens_seconds")
KV_NOT_EXPOSED = "not_exposed"
_LINE = re.compile(r"^(llamacpp:[A-Za-z0-9_]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]Inf)\s*$")


def parse(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if m:
            try:
                out[m.group(1)[len(PREFIX):]] = float(m.group(2))
            except ValueError:
                pass
    return out


def scrape(base_url: str, timeout_s: float = 5.0) -> dict[str, float]:
    with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout_s) as r:
        got = parse(r.read().decode("utf-8", "replace"))
    if not got:
        raise RuntimeError(f"{base_url}/metrics returned no llamacpp: series (server started without --metrics?)")
    return got


def window_report(before: dict[str, float], after: dict[str, float], samples: list[dict[str, float]] | None = None) -> dict:
    """Counters -> deltas (a counter that went backwards means a server restart: reported, not clamped);
    gauges -> start/end/max over every scrape given. Missing series are listed, never zero-filled."""
    rep: dict = {"counters": {}, "gauges": {}, "missing": [], "kv": KV_NOT_EXPOSED, "counter_reset": []}
    for k in COUNTERS:
        if k in before and k in after:
            d = after[k] - before[k]
            if d < 0:
                rep["counter_reset"].append(k)
            rep["counters"][k] = d
        else:
            rep["missing"].append(k)
    allv = [before, *(samples or []), after]
    for k in GAUGES:
        vals = [s[k] for s in allv if k in s]
        if vals:
            rep["gauges"][k] = {"start": before.get(k), "end": after.get(k), "max": max(vals)}
        else:
            rep["missing"].append(k)
    c = rep["counters"]
    if c.get("tokens_predicted_seconds_total", 0) > 0 and "tokens_predicted_total" in c:
        rep["server_decode_tok_s"] = c["tokens_predicted_total"] / c["tokens_predicted_seconds_total"]
    return rep
