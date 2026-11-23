"""The sampling-regime axis of loop evidence (AK-TEMP-AXIS).

Greedy-only evidence parked tree drafting while sampled trees accept 7-11% more tokens per
verify at the served temperature (DICE replay, research 03c100462). A speculative-decoding
or acceptance result is a statement about ONE sampling regime; this module is the single place
that names it, normalises it and asks whether it covers what production serves.

A sampling value is the string ``"unknown"`` (not provable from the run's own record), one
arm ``{temperature, top_p, top_k, exact_path}``, or a list of arms. Unset fields are None.
Nothing here invents a temperature: served values are read from the orchestrator registry's
role ``generation_defaults`` plus the :8083 direct-traffic constant, and an unreadable
registry means "cannot say", never a default.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

UNKNOWN = "unknown"
KEYS = ("temperature", "top_p", "top_k", "exact_path")
#: :8083 direct agent traffic (30 of 32 /slots snapshots, intake-ak-1123 rev1 research).
DIRECT_8083_TEMPERATURE = 1.0
REGISTRY_ENV = "EPYC_ORCH_REGISTRY"
REGISTRY_DEFAULT = Path("/workspace/repos/epyc-orchestrator/orchestration/model_registry.yaml")
#: Above this a temperature is the "high" regime; (0, LOW_MAX] is "low". T == 0 is greedy.
LOW_MAX = 0.5

_SPEC = re.compile(r"(?<!no-)draft|dflash|dspine|dtree|tree.?verif|speculat|spec-?dec|accept(?:ance|ed.length)"
                   r"|verify.*(?<!no-)draft|(?<!no-)draft.*verify", re.I)


def arm(temperature: Any, top_p: Any = None, top_k: Any = None, exact_path: Any = None) -> dict:
    """One validated sampling arm."""
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError(f"sampling.temperature must be a finite number >= 0, got {temperature!r}")
    return {"temperature": float(temperature), "top_p": top_p, "top_k": top_k, "exact_path": exact_path}


def normalize(value: Any) -> Any:
    """Canonical form: UNKNOWN, or a list of validated arms. None/absent is UNKNOWN."""
    if value is None or value == UNKNOWN:
        return UNKNOWN
    items = [value] if isinstance(value, Mapping) else value
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError(f"sampling must be 'unknown', an arm or a non-empty list of arms, got {value!r}")
    out = []
    for item in items:
        if not isinstance(item, Mapping) or set(item) - set(KEYS) - {"basis"} or "temperature" not in item:
            raise ValueError(f"sampling arm needs temperature and only {KEYS}: {item!r}")
        a = arm(*(item.get(k) for k in KEYS))
        if item.get("basis"):
            a["basis"] = str(item["basis"])
        out.append(a)
    return out


def arms(value: Any) -> list[dict]:
    """The arms of a sampling value ([] when unknown)."""
    n = normalize(value)
    return [] if n == UNKNOWN else n


def temperatures(value: Any) -> set[float]:
    return {a["temperature"] for a in arms(value)}


def is_spec_decode(record: Mapping[str, Any]) -> bool:
    """A record about drafting / acceptance / tree verification, judged on its structured text."""
    return bool(_SPEC.search(" ".join(str(record.get(k) or "") for k in (
        "mechanism_id", "target_surface", "target_symbol", "statement"))))


def backfill(record: Mapping[str, Any], *, proven: Mapping[str, Any] | None = None) -> dict:
    """Return the record with `sampling` set when it is a spec-decode record lacking one.

    `proven` maps mechanism_id -> arm dict for runs whose own record proves the regime;
    anything else spec-decode is UNKNOWN. An existing sampling value is never overwritten.
    """
    out = dict(record)
    if "sampling" in out or not is_spec_decode(out):
        return out
    known = (proven or {}).get(out.get("mechanism_id"))
    out["sampling"] = normalize(known) if known is not None else UNKNOWN
    return out


def regime(temperature: float) -> str:
    return "greedy" if temperature == 0 else "low" if temperature <= LOW_MAX else "high"


def _registry(registry: Any) -> Mapping[str, Any]:
    if isinstance(registry, Mapping):
        return registry
    import yaml
    path = Path(registry or os.environ.get(REGISTRY_ENV) or REGISTRY_DEFAULT)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def role_temperatures(registry: Any = None) -> dict[str, float]:
    """role -> generation_defaults.temperature, numeric entries only (no fallback value)."""
    roles = _registry(registry).get("roles") or {}
    out = {}
    for name, body in roles.items():
        t = ((body or {}).get("generation_defaults") or {}).get("temperature")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            out[name] = float(t)
    return out


def served_temperatures(registry: Any = None) -> tuple[float, ...]:
    """Every temperature production serves: registry role defaults plus :8083 direct traffic."""
    return tuple(sorted({*role_temperatures(registry).values(), DIRECT_8083_TEMPERATURE}))


def required_regimes(served: Iterable[float]) -> frozenset[str]:
    """Sampled regimes evidence must cover. T == 0 is the reproducibility arm: neither required
    nor sufficient, which is exactly what makes a greedy-only record re-openable."""
    return frozenset(regime(t) for t in served) - {"greedy"}


def covers_served(sampling: Any, served: Iterable[float]) -> bool:
    return required_regimes(served) <= {regime(t) for t in temperatures(sampling)}


def reopen_scope(row: Mapping[str, Any], served: Iterable[float]) -> str | None:
    """'greedy' when a spec-decode/acceptance negative is only as wide as its sampling set and
    that set misses a served regime (do_not_repeat_scope: greedy); None otherwise."""
    if not is_spec_decode(row):
        return None
    return None if covers_served(row.get("sampling"), served) else "greedy"
