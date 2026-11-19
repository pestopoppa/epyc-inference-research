"""Cross-lane quality gate: is a lane's ppl_contract keep admissible at GLOBAL promotion?

A low-bit lane's keep is accepted under the layered ppl_contract (loop/gates.py, section
"ppl_contract numerics"), not bit-exactness. The global champion advances across lanes, so
promotion must decide per lane whether that lane's own contract evidence covers the exact
candidate being promoted. Pure: it takes evidence mappings and returns verdicts; it runs
nothing and measures nothing. Fail CLOSED: absent, malformed, non-finite, mismatched or
under-sized evidence refuses. Thresholds are the contract's own (imported or mirrored
verbatim from gates.py); this module never defines a looser one.

Lane evidence (one mapping per lane):
  lane, candidate_commit, anchor_commit
  nmse:      {served_shapes: bool, widths: [int], quant_types: [str], max_nmse: float}
  ppl:       {anchor_ppl, candidate_ppl, reference_ppl}   (wikitext2)
  coherence: {prompts: [{prompt_tokens, agreement}], reference_agreement: [float]}
  canary:    {tokens, distinct4_ratio, anchor_distinct4_ratio, longest_repeat_run}
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from . import gates

#: test-backend-ops' MUL_MAT / MUL_MAT_ID per-case NMSE bound the contract's layer (a) uses.
NMSE_BOUND = 5e-4
PPL_REL_BAR = gates.PPL_REL_BAR
AGREEMENT_FLOOR = gates.PPL_CONTRACT_AGREEMENT_FLOOR
PROD_PROMPT_TOKENS_MIN = gates.PPL_CONTRACT_PROD_PROMPT_TOKENS_MIN
PROD_PROMPT_COUNT_MIN = gates.PPL_CONTRACT_PROD_PROMPT_COUNT_MIN
CANARY_TOKENS_MIN = gates.PPL_CONTRACT_CANARY_TOKENS
CANARY_MAX_REPEAT_RUN = gates.PPL_CONTRACT_CANARY_MAX_REPEAT_RUN
CANARY_DISTINCT4_DROP = gates.PPL_CONTRACT_CANARY_DISTINCT4_DROP
SERVED_WIDTHS = frozenset(gates.PPL_CONTRACT_SERVED_WIDTHS)


@dataclass(frozen=True)
class LaneVerdict:
    lane: str
    admissible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionVerdict:
    admissible: bool
    lanes: tuple[LaneVerdict, ...]
    reasons: tuple[str, ...]


def _num(v: Any) -> "float | None":
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def lane_verdict(ev: Any, *, candidate_commit: str) -> LaneVerdict:
    """Layers (a)-(d) for one lane, all required; every failure is recorded."""
    if not isinstance(ev, Mapping):
        return LaneVerdict("?", False, ("lane evidence is not a mapping",))
    lane = str(ev.get("lane") or "?")
    bad: list[str] = []

    def need(name: str) -> Mapping:
        v = ev.get(name)
        if not isinstance(v, Mapping):
            bad.append(f"{name}: evidence missing")
            return {}
        return v

    if not ev.get("lane"):
        bad.append("lane name missing")
    if not candidate_commit or ev.get("candidate_commit") != candidate_commit:
        bad.append("candidate_commit differs from the promoted commit: evidence is for a different build")
    if not ev.get("anchor_commit"):
        bad.append("anchor_commit missing")

    n = need("nmse")
    if n:
        mx = _num(n.get("max_nmse"))
        if n.get("served_shapes") is not True:
            bad.append("nmse: not measured on served shapes")
        widths = n.get("widths")
        if not isinstance(widths, (list, tuple)) or not SERVED_WIDTHS <= set(widths):
            bad.append(f"nmse: served widths {sorted(SERVED_WIDTHS)} not all covered")
        if not n.get("quant_types"):
            bad.append("nmse: no quant types listed")
        if mx is None:
            bad.append("nmse: max_nmse missing or non-finite")
        elif mx > NMSE_BOUND:
            bad.append(f"nmse: max {mx:.3g} > bound {NMSE_BOUND:.3g}")

    p = need("ppl")
    if p:
        a, c, r = (_num(p.get(k)) for k in ("anchor_ppl", "candidate_ppl", "reference_ppl"))
        if a is None or c is None or a <= 0 or c <= 0:
            bad.append("ppl: anchor/candidate ppl missing or non-positive")
        else:
            if abs(c - a) / a > PPL_REL_BAR:
                bad.append(f"ppl: |delta| {abs(c - a) / a:.4%} > {PPL_REL_BAR:.2%} vs anchor")
            if r is None or r <= 0:
                bad.append("ppl: fixed-reference ppl missing (cumulative drift unbounded)")
            elif abs(c - r) / r > PPL_REL_BAR and abs(c - r) / r > abs(a - r) / r:
                bad.append(f"ppl: cumulative drift vs fixed reference {abs(c - r) / r:.4%} exceeds {PPL_REL_BAR:.2%}")

    co = need("coherence")
    if co:
        prompts = co.get("prompts")
        ok_prompts = [q for q in prompts if isinstance(q, Mapping)] if isinstance(prompts, (list, tuple)) else []
        prod = [q for q in ok_prompts if (_num(q.get("prompt_tokens")) or 0) >= PROD_PROMPT_TOKENS_MIN]
        if len(prod) < PROD_PROMPT_COUNT_MIN:
            bad.append(f"coherence: {len(prod)} production-length prompts (>= {PROD_PROMPT_TOKENS_MIN} tokens), "
                       f"need {PROD_PROMPT_COUNT_MIN}")
        for i, q in enumerate(prod):
            ag = _num(q.get("agreement"))
            if ag is None or ag < AGREEMENT_FLOOR:
                bad.append(f"coherence: prompt {i} agreement {ag} below floor {AGREEMENT_FLOOR}")
        ref = co.get("reference_agreement")
        if not isinstance(ref, (list, tuple)) or len(ref) < PROD_PROMPT_COUNT_MIN or \
                any((_num(x) is None or _num(x) < AGREEMENT_FLOOR) for x in ref):
            bad.append("coherence: agreement vs fixed reference missing or below floor")

    cn = need("canary")
    if cn:
        toks, d4, d4a, run = (_num(cn.get(k)) for k in
                              ("tokens", "distinct4_ratio", "anchor_distinct4_ratio", "longest_repeat_run"))
        if toks is None or toks < CANARY_TOKENS_MIN:
            bad.append(f"canary: {toks} tokens < {CANARY_TOKENS_MIN}")
        if d4 is None or d4a is None or run is None:
            bad.append("canary: repetition metrics missing")
        else:
            if d4 < d4a - CANARY_DISTINCT4_DROP:
                bad.append(f"canary: distinct-4gram {d4:.3f} dropped > {CANARY_DISTINCT4_DROP} below anchor {d4a:.3f}")
            if run > CANARY_MAX_REPEAT_RUN:
                bad.append(f"canary: longest repeat run {run:g} > {CANARY_MAX_REPEAT_RUN}")
    return LaneVerdict(lane, not bad, tuple(bad))


def promotion_verdict(candidate_commit: str, ppl_contract_lanes: Any,
                      lane_evidence: Mapping[str, Any]) -> PromotionVerdict:
    """Global-champion promotion: every lane that carries a ppl_contract keep needs its own
    admissible evidence for `candidate_commit`. `ppl_contract_lanes` = lanes whose bundle
    includes such a keep (from the ledger/trailer); a lane listed but without evidence refuses.
    An empty lane list is vacuous, so it refuses too (cannot prove nothing was folded)."""
    lanes = sorted({str(x) for x in ppl_contract_lanes}) if isinstance(ppl_contract_lanes, (list, tuple, set, frozenset)) else []
    if not lanes:
        return PromotionVerdict(False, (), ("no ppl_contract lanes declared: nothing to accept, refusing vacuous pass",))
    out = []
    for lane in lanes:
        ev = lane_evidence.get(lane) if isinstance(lane_evidence, Mapping) else None
        if ev is None:
            out.append(LaneVerdict(lane, False, ("no lane evidence supplied",)))
            continue
        v = lane_verdict(ev, candidate_commit=candidate_commit)
        if v.lane != lane:
            v = LaneVerdict(lane, False, v.reasons + (f"evidence names lane {v.lane!r}",))
        out.append(v)
    refused = [v.lane for v in out if not v.admissible]
    return PromotionVerdict(not refused, tuple(out),
                            tuple(f"lane {x} refused" for x in refused))


EVIDENCE_DIR = "cross_lane"


def load_lane_evidence(store: Any) -> dict[str, Any]:
    """Read ``<store>/cross_lane/<lane>.json`` into ``{lane: evidence}``. Unreadable or
    non-object files are skipped, so a damaged record can only REFUSE (no evidence)."""
    import json
    from pathlib import Path
    out: dict[str, Any] = {}
    root = Path(store) / EVIDENCE_DIR if store is not None else None
    if root is None or not root.is_dir():
        return out
    for path in sorted(root.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            out[path.stem] = doc
    return out


def fold_verdict(commits: Any, lanes: Any, lane_evidence: Any) -> dict[str, Any]:
    """Promotion-path adapter: the promoted commit is the last of the series."""
    commits = list(commits)
    v = promotion_verdict(commits[-1] if commits else "", lanes, lane_evidence or {})
    return {"admissible": v.admissible, "reasons": list(v.reasons),
            "lanes": {x.lane: {"admissible": x.admissible, "reasons": list(x.reasons)}
                      for x in v.lanes}}


__all__ = ["EVIDENCE_DIR", "fold_verdict", "load_lane_evidence", "LaneVerdict", "PromotionVerdict", "lane_verdict", "promotion_verdict", "NMSE_BOUND"]
