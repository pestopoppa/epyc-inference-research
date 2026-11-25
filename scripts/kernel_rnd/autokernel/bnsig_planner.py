"""AK-BNSIG P2: bottleneck signatures in the planner -- seed section, planner block, WARN-first proposal gate.

Lives OUTSIDE loop/ because the loop code budget (check_regrowth_guards.py) had 23 lines of headroom; loop/ keeps
only one-line hooks (census_update.surface_seed_section, actor_render, actors._proposal_from_raw, run.py, run_args).

Input: census ``records[k].signature`` written by ``census_update.apply_signature`` (P1).

Proposal gate modes (``--signature-prune``), copied from amdahl.py's ladder:
- ``off``: nothing computed, no planner block;
- ``advisory`` (default): every proposal is assessed; a proposal whose declared limiter is in its target kernel's
  ``ruled_out`` set gets verdict ``signature_conflict``, a logged WARN and the assessment on the hypothesis;
- ``demote``: as advisory, and the assessment carries ``demoted: True`` for a tail hook to time it last.
NEVER refuses and never raises (refusal is the separate AK-BNSIG-PRUNE-REFUSE decision). A MARGINAL class fired, so it
is never "ruled out"; an unknown target kernel or an undeclared limiter is ``unknown`` / ``undeclared``, never pruned.

A proposal declares its limiter as ``sig:<class>`` in its statement or falsifier, or as an optional ``limiter``
field (string or list).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from .loop import bottleneck_signature as bs

MODES = ("off", "advisory", "demote")
DEFAULT_MODE = "advisory"
CONTEXT_KEY = "bottleneck_signatures"
LIMITERS = ("over_fetch", "over_subscribed", "issue_bound", "latency_bound", "not_in_flight")
SIG_RE = re.compile(r"\bsig:([a-z_]+)")
LOG = logging.getLogger("autokernel.bnsig")


def lever_map() -> dict[str, str]:
    return {c: bs.LEVER[c] for c in LIMITERS}


def _ruled_out(sig: Mapping[str, Any]) -> list[str]:
    fired = set(sig.get("classes") or []) | set(sig.get("marginal") or [])
    return [c for c in sig.get("ruled_out") or [] if c not in fired]


def _hotspot(rid: str, rec: Mapping[str, Any]) -> dict[str, Any]:
    s = rec["signature"]
    return {"record": rid, "kernel": rec.get("kernel_class") or rid, "rank": rec.get("rank"),
            "primary": s.get("primary") or s.get("class"), "classes": list(s.get("classes") or []),
            "marginal": list(s.get("marginal") or []), "ruled_out": _ruled_out(s),
            "band_position": {a: v for a, v in (s.get("band_position") or {}).items() if v is not None},
            "lever": bs.LEVER.get(s.get("primary") or s.get("class")), "source_row": s.get("source_row"),
            "result_sha256": s.get("result_sha256"), "stale_vs_timing": bool(s.get("stale_vs_timing"))}


def hotspots(census: Mapping[str, Any], rids: Any = None) -> list[dict[str, Any]]:
    """Signed records (optionally only `rids`), by rank (unranked last)."""
    recs = census.get("records") or {}
    keys = [k for k in (recs if rids is None else rids) if (recs.get(k) or {}).get("signature")]
    out = [_hotspot(k, recs[k]) for k in keys]
    return sorted(out, key=lambda h: (h["rank"] is None, h["rank"] or 0, h["record"]))


def _fmt_pos(pos: Mapping[str, Any]) -> str:
    return ", ".join(f"{a} {v:.2f}" for a, v in sorted(pos.items())) or "-"


def seed_lines(surface: Mapping[str, Any], census: Mapping[str, Any]) -> list[str]:
    """Extra lines for census_update.surface_seed_section: one `limiter:` line per signed census row of the surface."""
    hs = hotspots(census, surface.get("census_rows") or [])
    if not hs:
        return []
    out = ["Bottleneck signatures (AK-BNSIG; band position 0 = S16 healthy, 1 = S64 over-subscribed):"]
    for h in hs:
        out.append(f"- limiter: `{h['record']}` primary {h['primary']}; classes {h['classes']}; marginal {h['marginal']}; "
                   f"band positions {_fmt_pos(h['band_position'])}; lever: {h['lever'] or '-'}")
    out.append("Lever map: " + "; ".join(f"{c} -> {v}" for c, v in lever_map().items()))
    return out + [""]


def planner_view(census: Mapping[str, Any] | None, *, mode: str = DEFAULT_MODE, limit: int = 12) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"signature-prune mode {mode!r} not in {MODES}")
    if mode == "off":
        return {"status": "off"}
    hs = hotspots(census or {})
    if not hs:  # nothing signed: no block at all, so today's prompt stays byte-identical
        return {"status": "off"}
    return {"status": "observed", "mode": mode, "hotspots": hs[:limit], "lever_map": lever_map()}


def render(view: Mapping[str, Any]) -> list[str]:
    if view.get("status") != "observed":
        return []
    lines = ["", "## Bottleneck signatures (AK-BNSIG, counters vs stream refs S16/S64)",
             "| kernel | primary | classes | marginal | ruled out | band positions (0=S16, 1=S64) | lever |",
             "|---|---|---|---|---|---|---|"]
    for h in view["hotspots"]:
        lines.append(f"| `{h['kernel']}` | {h['primary']} | {', '.join(h['classes'])} | {', '.join(h['marginal']) or '-'} | "
                     f"{', '.join(h['ruled_out']) or '-'} | {_fmt_pos(h['band_position'])} | {h['lever'] or '-'} |")
    lines += ["", "Lever map: " + "; ".join(f"{c} -> {v}" for c, v in view["lever_map"].items()),
              "Declare the limiter you target as `sig:<class>` in the statement. A proposal targeting a limiter the "
              f"kernel's counters RULED OUT is flagged signature_conflict (mode `{view['mode']}`; never refused)."]
    return lines


def _declared(body: Mapping[str, Any]) -> list[str]:
    lim = body.get("limiter")
    out = [lim] if isinstance(lim, str) else list(lim or [])
    for k in ("statement", "falsifier"):
        out += SIG_RE.findall(str(body.get(k) or ""))
    return sorted({x.strip().replace("-", "_") for x in out if x})


def _match(view: Mapping[str, Any], symbol: str) -> list[dict[str, Any]]:
    sym = (symbol or "").lower()
    return [h for h in view.get("hotspots") or [] if sym and h["kernel"].lower() in sym or
            (len(sym) >= 4 and sym in h["kernel"].lower())]


def assess(body: Mapping[str, Any], view: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Assessment of one proposal body against the planner view; None when off/absent. Never raises."""
    if not view or view.get("status") in (None, "off"):
        return None
    try:
        mode, declared = view.get("mode", DEFAULT_MODE), _declared(body)
        hit = _match(view, str(body.get("target_symbol") or "")) if view.get("status") == "observed" else []
        # a kernel is ruled out of a limiter only if EVERY matched signed record rules it out
        ruled = set.intersection(*(set(h["ruled_out"]) for h in hit)) if hit else set()
        conflict = sorted(set(declared) & ruled)
        verdict = ("unknown" if not hit else "undeclared" if not declared
                   else "signature_conflict" if conflict else "consistent")
        row = {"mode": mode, "verdict": verdict, "declared": declared, "conflict": conflict,
               "records": [h["record"] for h in hit], "ruled_out": sorted(ruled),
               "demoted": mode == "demote" and verdict == "signature_conflict"}
    except Exception as exc:  # advisory: an assessment failure is reported, never a refusal
        return {"mode": view.get("mode"), "verdict": "error", "reason": repr(exc), "demoted": False}
    if verdict == "signature_conflict":
        LOG.warning("AK-BNSIG WARN signature_conflict: %s targets %s ruled out for %s",
                    body.get("mechanism_id"), conflict, row["records"])
    return row


def demoted(row: Mapping[str, Any] | None) -> bool:
    return bool(row and row.get("demoted"))


def context_entry(args: Any, cpu: bool) -> dict[str, Any]:
    """run.py hook: {CONTEXT_KEY: view} for a GPU target when not off, else {}."""
    mode = getattr(args, "signature_prune", DEFAULT_MODE) or DEFAULT_MODE
    if cpu or mode == "off":
        return {}
    path = getattr(args, "signature_census", None)
    if path is None:
        from .loop import census_update
        path = census_update.CENSUS
    try:
        census = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        LOG.warning("AK-BNSIG census unreadable, no signature block: %s", exc)
        return {}
    view = planner_view(census, mode=mode)
    return {CONTEXT_KEY: view} if view.get("status") == "observed" else {}


def add_arguments(parser) -> None:
    parser.add_argument("--signature-prune", choices=MODES, default=DEFAULT_MODE,
                        help="AK-BNSIG: flag proposals whose declared sig:<limiter> the target kernel's counter "
                             "signature ruled out; advisory logs a WARN, demote also marks it for last; never refuses")
    parser.add_argument("--signature-census", type=Path, default=None,
                        help="bandwidth census JSON carrying records[k].signature (default: the packaged census)")


__all__ = ["CONTEXT_KEY", "DEFAULT_MODE", "MODES", "add_arguments", "assess", "context_entry", "demoted",
           "hotspots", "lever_map", "planner_view", "render", "seed_lines"]
