"""census_coverage.py -- which (model, device) paths the registry SERVES, and what the census still lacks for them (RSL-R1).

PURE: no census state, no writes. `served_set(registry)` reads the compiled lean registry view; `coverage(census, served)`
is what the census integrator calls. Source choice: the LEAN compiled view (orchestrator `orchestration/model_registry.yaml`,
the same file `sampling_axis.py` reads, override `EPYC_ORCH_REGISTRY`). The research MASTER has no `draft_max` for the 27B
(DRAFT-SEL-1 removed it there; `registry_compiler` projects it from `stack_topology.drafter_selection`), so the master
cannot answer "what n does this path verify" -- the lean view can.

Live = a non-alias `server_mode` host with a `model_role`. Device: `ROCm*`/`MI210*` -> `mi210`, absent/`none` -> `cpu`.
PLANNED (operator directive 2026-11-24): A3B-MTP on the MI210 is tracked AS IF served because a second GPU is coming.
It is reported as `planned`, never as a mismatch and never stripped; its n-set is the same model_role's draft_max on the
host that serves it today. A path that is neither live nor planned stays a mismatch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REGISTRY_ENV = "EPYC_ORCH_REGISTRY"
REGISTRY_DEFAULT = Path("/workspace/repos/epyc-orchestrator/orchestration/model_registry.yaml")
#: The n values the census harness measures. Fallback (registry unreadable) is exactly this -- the pre-RSL-R1 SERVED_N.
MEASURED_N = (1, 3, 4, 8, 16)
FALLBACK_N = frozenset(MEASURED_N)
#: (registry model_role, device) -> (census model key, parity-op `path`). The registry does not carry census keys.
CENSUS_KEY = {
    ("qwen38_27b_q8_local", "mi210"): ("27b-q8_0", "27b-df2-mi210"),
    ("qwen36_35b_a3b_mtp_q8_local", "cpu"): ("a3b-q8_0-cpu", "a3b-mtp-cpu"),
    ("qwen36_35b_a3b_mtp_q8_local", "mi210"): ("a3b-q8_0", "a3b-mtp-mi210"),
    ("qwen38_flash_next_ud_iq4xs_local", "cpu"): ("q38fn-ud-iq4xs-cpu", "flash-next-cpu"),
}
#: Operator directive 2026-11-24 (see module docstring): tracked as if served; status `planned`.
PLANNED = (("qwen36_35b_a3b_mtp_q8_local", "mi210"),)


def load_registry(path=None) -> dict | None:
    """The registry dict, or None if unreadable/not a mapping (the caller decides what None means)."""
    p = Path(path or os.environ.get(REGISTRY_ENV) or REGISTRY_DEFAULT)
    try:
        import yaml
        with open(p) as f:
            d = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    except Exception:  # noqa: BLE001 -- any read/parse failure is "unreadable"
        return None
    return d if isinstance(d, dict) else None


def _device(entry: dict) -> str:
    dev = str(entry.get("device") or "none").lower()
    return "mi210" if dev.startswith(("rocm", "mi210")) else "cpu"


def _draft_max(name: str, entry: dict, registry: dict) -> int | None:
    for acc in (entry.get("acceleration"), ((registry.get("roles") or {}).get(name) or {}).get("acceleration")):
        dm = (acc or {}).get("draft_max") if isinstance(acc, dict) else None
        if isinstance(dm, int) and not isinstance(dm, bool) and dm >= 0:
            return dm
    return None


def _entry(model: str, path: str | None, status: str, dm: int | None, slots: int | None, role: str) -> dict:
    # n-set: one slot verifies 1..draft_max+1 rows; `n_batch_max` = every slot verifying at once.
    n = frozenset(range(1, dm + 2)) if dm is not None else frozenset()
    return {"model": model, "path": path, "status": status, "model_role": role, "draft_max": dm, "slots": slots,
            "n": n, "n_batch_max": (slots or 1) * (dm + 1) if dm is not None else None}


def served_set(registry: dict) -> dict:
    """{(census model, device): {status: live|planned, path, n, ...}} from the registry.

    Live registry hosts with no CENSUS_KEY mapping land under (model_role, device) with path None: coverage reports
    them as `unmapped` rather than dropping them."""
    live: dict = {}
    for name, e in sorted((registry.get("server_mode") or {}).items()):
        if not isinstance(e, dict) or e.get("alias_of") or not e.get("model_role"):
            continue
        role, dev = e["model_role"], _device(e)
        dm, slots = _draft_max(name, e, registry), e.get("slots") if isinstance(e.get("slots"), int) else None
        old = live.get((role, dev))
        if old and old[0] is not None and (dm is None or old[0] >= dm):
            continue
        live[(role, dev)] = (dm, slots)
    out = {}
    for (role, dev), (dm, slots) in live.items():
        model, path = CENSUS_KEY.get((role, dev), (role, None))
        out[(model, dev)] = _entry(model, path, "live", dm, slots, role)
    for role, dev in PLANNED:
        model, path = CENSUS_KEY[(role, dev)]
        if (model, dev) in out:  # it became live: live wins
            continue
        dm, slots = next(((d, s) for (r, _), (d, s) in live.items() if r == role), (None, None))
        out[(model, dev)] = _entry(model, path, "planned", dm, slots, role)
    return out


def served_n(registry: dict | None = None, grid=MEASURED_N) -> frozenset:
    """The census n values to admit: grid values a LIVE path can verify in one batch (slots x (draft_max+1)).

    Per-slot n alone (draft_max+1 = 8 for the 27B) would drop the n=16 cells (np=4 batched verify) that are measured today.
    FALLBACK_N (= the pre-RSL-R1 hand-typed set) when the registry is unreadable or yields nothing: that is
    FAIL-OPEN -- it admits exactly what was admitted before and never narrows, so a missing registry can only
    over-admit cells, never lose them. A stderr line says so."""
    reg = registry if registry is not None else load_registry()
    caps = [e["n_batch_max"] for e in (served_set(reg).values() if reg else ())
            if e["status"] == "live" and e["n_batch_max"] is not None]
    n = frozenset(g for g in grid if caps and g <= max(caps))
    if not n:
        print("census_coverage: registry unreadable/empty -> SERVED_N fallback (fail-open, unchanged set)", file=sys.stderr)
        return FALLBACK_N
    return n


def n_report(registry: dict | None = None, grid=MEASURED_N) -> dict:
    """Per-path admitted n, computed from the registry on every call (RSL-R1; never at import): for each served (live or
    planned) path, the grid values <= slots x (draft_max+1), keyed by CENSUS MODEL. {by_model, fallback, fallback_n}.
    `fallback` is None, or WHY the fail-open set (FALLBACK_N) is in force -- the caller journals it, stderr alone is lost.
    A model the registry gives no draft_max for is named in `fallback` and falls back to FALLBACK_N; the rest stay derived."""
    reg = registry if registry is not None else load_registry()
    out = {"by_model": {}, "fallback": None, "fallback_n": sorted(FALLBACK_N)}
    if not reg:
        return {**out, "fallback": "registry-unreadable"}
    bad = []
    for (model, _dev), e in sorted(served_set(reg).items()):
        if e["path"] is None:
            continue
        if e["n_batch_max"] is None:   # a PLANNED path with no live twin has no draft_max by construction: not a fallback
            bad += [model] if e["status"] == "live" else []
            continue
        out["by_model"][model] = out["by_model"].get(model, frozenset()) | frozenset(g for g in grid if g <= e["n_batch_max"])
    if bad or not out["by_model"]:
        out["fallback"] = ("no draft_max for " + ", ".join(bad)) if bad else "no served path"
    return out


def coverage(census: dict, served: dict) -> dict:
    """Pure. `served` = served_set(...). Returns

    paths:      {"model|device": {status, path, n, uncovered_ops}} -- parity ops (served:true) on the path with no
                census record (record.op_path + record.op). Planned paths are listed here, status `planned`.
    mismatch:   parity ops marked served:true whose path is neither live nor planned (a real defect).
    unmapped:   live registry hosts with no CENSUS_KEY mapping (coverage cannot be judged for them).
    """
    have = {(r.get("op_path"), r.get("op")) for r in (census.get("records") or {}).values() if isinstance(r, dict)}
    ops = [o for o in ((census.get("parity") or {}).get("ops") or []) if isinstance(o, dict)]
    paths, known = {}, set()
    for (model, dev), e in sorted(served.items()):
        if e["path"] is None:
            continue
        known.add(e["path"])
        paths[f"{model}|{dev}"] = {
            "status": e["status"], "path": e["path"], "n": sorted(e["n"]),
            "uncovered_ops": [o["op"] for o in ops if o.get("path") == e["path"] and o.get("served")
                              and (e["path"], o["op"]) not in have]}
    return {"paths": paths,
            "mismatch": [{"op": o.get("op"), "path": o.get("path")} for o in ops
                         if o.get("served") and o.get("path") not in known],
            "unmapped": sorted(f"{m}|{d}" for (m, d), e in served.items() if e["path"] is None)}
