"""YARN-AGG-1(4): per-length re-tokenization. Stdlib only; the tokenizer is the SERVED model.

Nominal lengths (generator sizing used a GPT-4 tokenizer; intake-2281#record) are never used to label a
cell. Every context is re-tokenized with the served tokenizer (llama-server ``POST /tokenize``) or its
real ``prompt_n`` is read from the server timings, the result is binned by that real count, and
``label_zone`` REFUSES any input that is not a measured count.

Zones (native window N = 262144 = ``--yarn-orig-ctx``; ``reserve`` = tokens kept for question + answer):
  * ``native``        real_n + reserve <= N
  * ``beyond-native`` real_n > N
  * ``boundary``      otherwise: usable as NEITHER (reported, excluded from native/beyond verdicts)
"""

from __future__ import annotations

import json
import statistics
import urllib.request
from typing import Any, Callable, Iterable, Mapping

NATIVE_CTX = 262144
DEFAULT_RESERVE = 4096
BIN_STEP = 32768
#: "model_tokenizer_json" = the served model's own tokenizer.json read offline (same vocabulary, no server);
#: llama.cpp's /tokenize can differ by a few tokens (special tokens), so the first window re-checks against
#: server_timings prompt_n and that value supersedes this one.
MEASURED_SOURCES = ("served_tokenizer", "server_timings", "model_tokenizer_json")
#: a cell whose real median differs from its nominal target by more than this is flagged.
NOMINAL_MISMATCH_FRAC = 0.10


class NominalLabelError(ValueError):
    """Raised when a native / beyond-native label is requested from anything but a measured count."""


class ServedTokenizer:
    """llama-server ``/tokenize`` client (base like http://localhost:18083, no /v1). Never starts a server."""

    def __init__(self, base_url: str, timeout_s: float = 600.0, opener: Callable | None = None):
        self.base = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._open = opener or urllib.request.urlopen

    def count(self, text: str) -> int:
        req = urllib.request.Request(f"{self.base}/tokenize", data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with self._open(req, timeout=self.timeout_s) as r:
            toks = json.loads(r.read()).get("tokens")
        if not isinstance(toks, list):
            raise RuntimeError("/tokenize returned no token list")
        return len(toks)


def real_prompt_n(timings: Mapping[str, Any]) -> int:
    """Real context size of a served request: prompt_n + cache_n (a cached-prefix request splits them)."""
    pn, cn = timings.get("prompt_n"), timings.get("cache_n") or 0
    if not isinstance(pn, int):
        raise NominalLabelError("timings carry no integer prompt_n")
    return pn + int(cn)


def label_zone(real_n: int, *, source: str, native: int = NATIVE_CTX, reserve: int = DEFAULT_RESERVE) -> str:
    if source not in MEASURED_SOURCES:
        raise NominalLabelError(f"zone label refused: source {source!r} is not a measured count {MEASURED_SOURCES}")
    if not isinstance(real_n, int) or isinstance(real_n, bool) or real_n <= 0:
        raise NominalLabelError(f"zone label refused: real_n={real_n!r}")
    if real_n > native:
        return "beyond-native"
    if real_n + reserve <= native:
        return "native"
    return "boundary"


def bin_label(real_n: int, step: int = BIN_STEP) -> str:
    """Half-open bin on the REAL count, e.g. (229376, 262144] -> ``229376-262144``."""
    hi = -(-real_n // step) * step
    return f"{hi - step}-{hi}"


def measure_records(records: Iterable[Mapping[str, Any]], *, count: Callable[[str], int],
                    text_key: str = "context", id_key: str = "id") -> list[dict]:
    """Add real_n / bin / zone to each record by re-tokenizing its context text with ``count``."""
    out = []
    for r in records:
        n = count(r[text_key])
        out.append({id_key: r.get(id_key), "nominal_len": r.get("nominal_len"), "real_n": n,
                    "bin": bin_label(n), "zone": label_zone(n, source="served_tokenizer"),
                    "n_source": "served_tokenizer"})
    return out


def cell_report(rows: list[Mapping[str, Any]], nominal_target: int) -> dict:
    """Per-cell summary; flags a nominal/real mismatch instead of silently trusting the nominal label."""
    ns = [r["real_n"] for r in rows]
    if not ns:
        return {"nominal_target": nominal_target, "n": 0, "zone": None, "flags": ["empty"]}
    zones = {r["zone"] for r in rows}
    med = statistics.median(ns)
    flags = []
    if abs(med - nominal_target) > NOMINAL_MISMATCH_FRAC * nominal_target:
        flags.append("nominal_mismatch")
    if len(zones) > 1:
        flags.append("mixed_zones")
    return {"nominal_target": nominal_target, "n": len(ns), "real_min": min(ns), "real_median": med,
            "real_max": max(ns), "zone": zones.pop() if len(zones) == 1 else "mixed",
            "bins": sorted({r["bin"] for r in rows}), "flags": flags}
