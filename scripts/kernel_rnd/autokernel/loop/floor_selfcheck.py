"""AKX-AUTO-4: calibration self-check for a matched noise-floor calibration.

Two independent screens on a freshly measured A/A floor, both pure functions of the
recorded samples (no host probe, no clock, no I/O except the two small lineage helpers
at the bottom):

1. `find_degraded_block` -- a CONTIGUOUS run of pairs whose throughput LEVEL sits below
   the calibration's own reference level. This is the 2026-10-05 Q38FN signature: the
   floor was 6.528% because pairs 8..23 of 24 ran at ~45 tok/s after the first eight ran
   at ~50 (a 5.5-minute degraded-host block). Both arms degrade together inside the
   block, so the damage is not visible per pair as |effect|; it is visible as a level
   shift that persists in time. The existing evidence-based guard
   (`serving._apply_outlier_guard`) only sees pairs carrying cpu_lifecycle/clock
   evidence, and the MAD diagnostic is per-pair, so a plateau of degraded pairs passes
   both. A single low pair is deliberately NOT flagged here (that is the outlier
   guard's domain); a run is.
2. `assess_spread` -- the floor's magnitude against the lineage's prior floors for the
   SAME target (3x, the same factor `run._carry_forward_floor` already applies to
   unguarded floors, with the prior clipped below at `MIN_PRIOR_FLOOR_PCT` so a
   0.1% -> 0.4% wobble on a very quiet host is not "anomalous"). Two independent
   calibrations that AGREE on the higher floor are accepted (two-sample persistence):
   a real regime change must not be refused forever.

The result is data; callers decide. `serving.calibrate_floor` raises
`HostDegradedDuringCalibration` on a degraded block; the spread verdict needs the prior
floors, so it is applied by the call site that files the floor (see `assess_floor`).
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

SCHEMA = "epyc.autokernel.floor_selfcheck.v1"

#: A pair is degraded when its level is more than this fraction below the reference.
#: Chosen against the recorded Q38FN floors: clean calibrations show worst single-pair
#: level dips of <= ~3.5%; the 2026-10-05 block sits ~9-35% below.
DEGRADED_LEVEL_FRACTION = 0.05
#: Consecutive degraded pairs that make a block (1-2 low pairs are outlier-guard turf).
MIN_BLOCK_PAIRS = 3
#: The reference level is this quantile of per-pair levels. Upper-quantile on purpose:
#: a median reference collapses when most of a calibration is degraded (16 of 24 in the
#: motivating case).
REFERENCE_QUANTILE = 0.75
#: Spread-anomaly factor vs the prior floor, and the prior's lower clip (percent).
SPREAD_FACTOR = 3.0
MIN_PRIOR_FLOOR_PCT = 1.0
#: Two calibrations agree when their floors are within this relative tolerance.
PERSISTENCE_REL_TOL = 0.25

ACCEPT = "accept"
REJECT_DEGRADED_BLOCK = "reject_degraded_block"
RECALIBRATE_SPREAD = "recalibrate_spread"


@dataclass(frozen=True)
class Assessment:
    verdict: str
    reasons: tuple = ()
    degraded_blocks: tuple = ()  # ((first_pair, last_pair), ...) inclusive
    reference_level: float | None = None
    floor_pct: float | None = None
    prior_floor_pct: float | None = None
    spread_ratio: float | None = None
    persisted: bool = False
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict == ACCEPT

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "verdict": self.verdict, "reasons": list(self.reasons),
                "degraded_blocks": [list(b) for b in self.degraded_blocks],
                "reference_level": self.reference_level, "floor_pct": self.floor_pct,
                "prior_floor_pct": self.prior_floor_pct, "spread_ratio": self.spread_ratio,
                "persisted": self.persisted}


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def pair_levels(anchor: Sequence[float], candidate: Sequence[float]) -> list[float]:
    """Per-pair throughput level (arithmetic mean of the pair's two launches), in pair
    (= time) order."""
    if len(anchor) != len(candidate):
        raise ValueError("anchor/candidate sample counts differ")
    return [(a + c) / 2.0 for a, c in zip(anchor, candidate)]


def find_degraded_block(anchor: Sequence[float], candidate: Sequence[float], *,
                        fraction: float = DEGRADED_LEVEL_FRACTION,
                        min_pairs: int = MIN_BLOCK_PAIRS,
                        reference_level: float | None = None
                        ) -> tuple[list[tuple[int, int]], float | None]:
    """`(blocks, reference_level)`: inclusive `(first, last)` pair-index runs of at least
    `min_pairs` consecutive pairs whose level is `fraction` below the reference. Empty
    or non-finite/non-positive input yields no blocks (nothing to judge)."""
    levels = pair_levels(anchor, candidate)
    if len(levels) < min_pairs or not all(math.isfinite(x) and x > 0 for x in levels):
        return [], None
    ref = reference_level if reference_level is not None else _quantile(levels, REFERENCE_QUANTILE)
    cutoff = ref * (1.0 - fraction)
    blocks, start = [], None
    for i, level in enumerate(levels + [math.inf]):
        if level < cutoff:
            start = i if start is None else start
        elif start is not None:
            if i - start >= min_pairs:
                blocks.append((start, i - 1))
            start = None
    return blocks, ref


def assess_spread(floor_pct: float, prior_floor_pcts: Sequence[float], *,
                  refused_floor_pcts: Sequence[float] = ()) -> Assessment:
    """Floor magnitude vs the median of the lineage's prior floors for this target."""
    priors = [p for p in prior_floor_pcts if isinstance(p, (int, float)) and p > 0]
    if not priors or not isinstance(floor_pct, (int, float)) or floor_pct != floor_pct:
        return Assessment(ACCEPT, ("no prior floor to compare" if not priors else "no floor",),
                          floor_pct=floor_pct)
    prior = statistics.median(priors)
    bar = SPREAD_FACTOR * max(prior, MIN_PRIOR_FLOOR_PCT)
    ratio = floor_pct / prior
    if floor_pct <= bar:
        return Assessment(ACCEPT, floor_pct=floor_pct, prior_floor_pct=prior, spread_ratio=ratio)
    agreeing = [r for r in refused_floor_pcts if isinstance(r, (int, float)) and r > 0
                and abs(r - floor_pct) <= PERSISTENCE_REL_TOL * floor_pct]
    if agreeing:
        return Assessment(ACCEPT, (f"floor {floor_pct}% exceeds {bar:.3f}% but an independent "
                                   f"refused calibration agrees ({agreeing[0]}%): persistent, "
                                   "accepted",), floor_pct=floor_pct, prior_floor_pct=prior,
                          spread_ratio=ratio, persisted=True)
    return Assessment(
        RECALIBRATE_SPREAD,
        (f"floor {floor_pct}% is {ratio:.1f}x the prior {prior:.3f}% for this target "
         f"(bar {SPREAD_FACTOR:g}x of max(prior, {MIN_PRIOR_FLOOR_PCT:g}%) = {bar:.3f}%); "
         "recalibrate",), floor_pct=floor_pct, prior_floor_pct=prior, spread_ratio=ratio)


def assess_floor(anchor: Sequence[float], candidate: Sequence[float], floor_pct: float | None = None,
                 prior_floor_pcts: Sequence[float] = (), *,
                 refused_floor_pcts: Sequence[float] = ()) -> Assessment:
    """Both screens. A degraded block outranks the spread verdict (it is the cause)."""
    blocks, ref = find_degraded_block(anchor, candidate)
    if blocks:
        spans = ", ".join(f"pairs {a}-{b}" for a, b in blocks)
        spread = (assess_spread(floor_pct, prior_floor_pcts, refused_floor_pcts=refused_floor_pcts)
                  if floor_pct is not None else None)
        return Assessment(
            REJECT_DEGRADED_BLOCK,
            (f"contiguous degraded block ({spans} of {len(anchor)}; level more than "
             f"{DEGRADED_LEVEL_FRACTION:.0%} below reference {ref:.2f})",),
            degraded_blocks=tuple(blocks), reference_level=ref, floor_pct=floor_pct,
            prior_floor_pct=spread.prior_floor_pct if spread else None,
            spread_ratio=spread.spread_ratio if spread else None)
    if floor_pct is None:
        return Assessment(ACCEPT, reference_level=ref)
    spread = assess_spread(floor_pct, prior_floor_pcts, refused_floor_pcts=refused_floor_pcts)
    return Assessment(spread.verdict, spread.reasons, reference_level=ref, floor_pct=floor_pct,
                      prior_floor_pct=spread.prior_floor_pct, spread_ratio=spread.spread_ratio,
                      persisted=spread.persisted)


# ---- lineage helpers (small, read-only except `record_refusal`) -------------------

REFUSALS_FILENAME = "floor-selfcheck-refusals.jsonl"


def prior_floor_pcts(lineage_dir: Path | str, *, exclude: Path | str | None = None,
                     name: str | None = None) -> list[float]:
    """`floor_pct` of every sealed floor under `<lineage>/<identity>/<name or
    serving-floor.*.json>` that is not `suspect` and was not degraded-block-rejected
    (so a floor this check would refuse never becomes the next one's baseline).
    `exclude` is the floor being judged."""
    lineage = Path(lineage_dir)
    out = []
    pattern = name or "serving-floor.*.json"
    for path in sorted(lineage.glob(f"*/{pattern}")):
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        try:
            row = json.loads(path.read_text())
            pct = row["floor_pct"]
            if row.get("suspect") or not isinstance(pct, (int, float)) or pct <= 0:
                continue
            if not assess_floor(row["anchor_samples"], row["candidate_samples"]).ok:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        out.append(float(pct))
    return out


def record_refusal(lineage_dir: Path | str, assessment: Assessment) -> Path:
    """Append a refused-floor record so the NEXT calibration can show persistence."""
    path = Path(lineage_dir) / REFUSALS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(assessment.to_dict(), sort_keys=True) + "\n")
    return path


def refused_floor_pcts(lineage_dir: Path | str) -> list[float]:
    """Spread-refused floors only: a degraded-block refusal says nothing about the
    host's real spread, so it can never vouch for a later reading."""
    path = Path(lineage_dir) / REFUSALS_FILENAME
    out = []
    try:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("verdict") == RECALIBRATE_SPREAD and isinstance(rec.get("floor_pct"), (int, float)):
                out.append(float(rec["floor_pct"]))
    except (OSError, ValueError):
        pass
    return out


class FloorSelfcheckRefused(RuntimeError):
    """A freshly measured floor failed the self-check and must not be filed."""

    def __init__(self, assessment: Assessment):
        super().__init__(f"floor self-check refused the calibration [AKX-AUTO-4]: "
                         f"{'; '.join(assessment.reasons)}")
        self.assessment = assessment


def check_before_write(store: Path | str, row: Mapping) -> Assessment:
    """The call-site entry: judge a calibration row BEFORE it is written to the lineage
    `<store>/runtime-source-floors/<recipe_hash>/`. Raises `FloorSelfcheckRefused` (a
    spread refusal is recorded so an agreeing re-measure is accepted next time). Any
    exception unwinds through the caller's existing failure path, which restores a
    claimed REMEASURE_REQUEST to pending: the re-measure is therefore automatic."""
    lineage = Path(store) / "runtime-source-floors" / str(row["recipe_hash"])
    assessment = assess_floor(row["anchor_samples"], row["candidate_samples"], row["floor_pct"],
                              prior_floor_pcts(lineage), refused_floor_pcts=refused_floor_pcts(lineage))
    if not assessment.ok:
        if assessment.verdict == RECALIBRATE_SPREAD:
            record_refusal(lineage, assessment)
        raise FloorSelfcheckRefused(assessment)
    return assessment
