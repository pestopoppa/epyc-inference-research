#!/usr/bin/env python3
"""Prove the promoted anchor IS the champion, before the loop draws any more work.

WHY. Run 18 promoted `5ad3e36d` at 11:03. Measured effects went from a median of
-1.441% (n=16, best +0.060%) to **-9.539%** (n=114, best -5.642%) -- the near-exact
inverse of the +9.321% champion advance just kept. Every candidate after the promotion
measured as though the champion's own patch were missing from one side. The lane source
worktrees were all correctly at `5ad3e36d`, so the source was right and the binary in
the anchor slot was not the champion. Nothing asserted that correspondence: the
promotion's tests checked that `pool.promote_anchor` ran and made its directory -- a
question about the CODE, not the ARTIFACT. 6.5 hours, 114 void measurements, and no
assertion in the package could have said so. (`pool.promote_anchor` now BUILDS the
champion into the anchor slot rather than renaming a build directory into it, which
removes the leading cause. This guard is what proves that rebuild worked.)

THE CHECK. Build the champion commit FRESH into a scratch directory with the recipe and
cmake defines the loop builds candidates with, then A/A it against the promoted anchor
through the same `bench.compare` a real comparison uses, at the run's pair count. Two
builds of one commit must be indistinguishable, so the effect must land INSIDE the floor.
The MAGNITUDE is compared directly, never via `Comparison.decisive`: `decisive` folds in
a drift veto and returns False for a drifting arm, so a guard reading it as
"indistinguishable" would wave run 18's -9.539% through on any run whose arms happened
to be moving. This guard must never pass for a reason unrelated to its own question.

The scratch directory is REMOVED before the build: an incremental rebuild is exactly the
mechanism capable of producing a binary that does not match its source, and a guard must
not be exposed to the fault class it exists to detect. Cost is one build plus one A/A
per keep; run 18 kept 1 in 159. No flag: a guard that can be turned off is not a guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Callable

from . import loop as loop_mod

#: Where the fresh champion build goes -- never an anchor generation or a lane's build
#: directory, or the guard would end up comparing a directory against itself.
SCRATCH_BUILD = Path("/mnt/raid0/llm/tmp/build-anchor-verify")
#: The id the verdict files under, so the check is greppable in `experiments.md`.
MECHANISM_ID = "anchor-aa-guard"


@dataclass(frozen=True)
class AnchorVerdict:
    """One promotion's A/A. Structured, because a verdict only in a log is not audit.

    `passed` answers the guard's one question — does the anchor slot hold the
    champion — so a hash-proven EXCURSION (R22-3: code digests identical, A/A above
    the floor) is `passed=True, excursion=True`: the reading indicts the measurement
    session, never the anchor, and the run continues. `evidence` carries the full
    `Comparison.to_dict()` and both code digests, because run 21's abort left no
    samples/drift/clock record to reason from after the fact.
    """
    passed: bool
    champion_commit: str
    anchor: str
    effect_pct: float
    noise_floor_pct: float
    surface: str
    pairs: int
    detail: str
    excursion: bool = False
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"check": MECHANISM_ID, "passed": self.passed, "pairs": self.pairs,
                "champion_commit": self.champion_commit, "anchor": self.anchor,
                "effect_pct": self.effect_pct, "surface": self.surface,
                "floor_pct": self.noise_floor_pct, "detail": self.detail,
                "excursion": self.excursion, **(self.evidence or {})}

    def to_attempt(self) -> dict[str, Any]:
        return {"status": ("anchor_guard_excursion" if self.excursion else
                           "anchor_verified" if self.passed else "anchor_mismatch"),
                "mechanism_id": MECHANISM_ID, "target_surface": self.surface,
                "effect_fraction": self.effect_pct / 100.0, "reason": self.detail,
                "anchor_guard": self.to_dict()}


def verify(*, champion_commit: str, anchor_build: Path,
           build: Callable[[Path], Any], compare: Callable[[Path, Path], Any],
           noise_floor_pct: float, scratch_build: Path = SCRATCH_BUILD,
           digest: Callable[[Path], str | None] | None = None,
           clean: Callable[[Path], None] = lambda p: shutil.rmtree(p, True),
           on_verdict: Callable[[AnchorVerdict], Any] | None = None,
           on_step: Callable[[str], Any] = lambda _label: None) -> AnchorVerdict:
    """A/A the promoted anchor against a fresh champion build; abort the run if they differ.

    `build(scratch) -> gates.Verdict` and `compare(anchor, fresh) -> Comparison` are
    injected, so the guard is exercised with no GPU, no ROCm and no API key. The anchor
    is the FIRST arm, putting the reported effect on the same scale and sign as the run's
    own rows: run 18's -9.539% is what this would have printed. `on_verdict` fires on
    BOTH outcomes and BEFORE the abort -- it is how the verdict reaches the store and the
    status file, and a guard nobody can audit afterwards is what let run 18 happen. Raises
    `loop.RunAborted`, the run-ending path: a wrong anchor voids every measurement after
    it, so continuing is worse than stopping.

    `digest` (R22-3, root-caused by R21-10) is the HASH PRE-CHECK: builds of one
    commit are deterministic on this host, so `digest(build_dir)` -- code sections
    only, RUNPATH and friends excluded (`controller/anchor_integrity.py`) -- settles
    the guard's question before a pair is spent. Digests DIFFER: run 18's fault class
    proven; one heal (rmtree + rebuild + re-hash) is allowed for a corrupted scratch
    build, then abort naming both digests. Digests IDENTICAL: the anchor provably IS
    the champion; the A/A still runs as a session-health sample, but an above-floor
    reading is an instrument EXCURSION (run 21 aborted a healthy run on a 4.2σ one:
    +1.765% against a pooled A/A σ of 0.417%) -- recorded, never an abort. None from
    `digest`, or no digest wired, falls back to the A/A-only behaviour above.
    """
    clean(scratch_build)
    on_step("anchor guard: building the champion fresh")
    built = build(scratch_build)
    if not getattr(built, "passed", False):
        raise loop_mod.RunAborted(
            f"anchor guard: champion {champion_commit[:12]} would not build "
            f"({getattr(built, 'reason', '') or 'no reason given'}), so the promoted "
            f"anchor is UNCHECKED — which is what produced 114 void measurements")
    a_dig, f_dig = [digest(p) if digest else None for p in (anchor_build, scratch_build)]
    if a_dig and f_dig and a_dig != f_dig:
        # HEAL ONCE, scratch side only: the guard's own fresh build is the artifact it
        # controls, and a transient corruption there must not abort a healthy run. The
        # last published step ("building the champion fresh") stays accurate meanwhile.
        clean(scratch_build)
        f_dig = digest(scratch_build) if build(scratch_build).passed else None
        if a_dig != f_dig:
            verdict = AnchorVerdict(
                False, champion_commit, str(anchor_build), 0.0, noise_floor_pct,
                # Both digests named in full: the abort IS the evidence (run 21's
                # left none), and a truncated hash cannot be re-checked by hand.
                "none", 0, f"anchor guard: code-section digests DIFFER even after one "
                f"heal — promoted anchor {Path(anchor_build).name} is {a_dig}, a fresh "
                f"champion build is {f_dig or 'unavailable'}. Run 18's fault class, "
                f"proven with zero pairs spent. Aborting",
                evidence={"anchor_digest": a_dig, "fresh_digest": f_dig})
            if on_verdict is not None: on_verdict(verdict)  # noqa: E701 — loop budget
            raise loop_mod.RunAborted(verdict.detail)
    on_step("anchor guard: A/A against the promoted anchor")
    comparison = compare(anchor_build, scratch_build)
    effect_pct = comparison.effect * 100.0
    passed = abs(effect_pct) <= noise_floor_pct
    excursion = not passed and bool(a_dig) and a_dig == f_dig
    head = (f"anchor guard: a fresh build of champion {champion_commit[:12]} measured "
            f"{effect_pct:+.3f}% against promoted anchor {Path(anchor_build).name} over "
            f"{comparison.pairs} {comparison.surface} pairs, floor {noise_floor_pct:.3f}%")
    verdict = AnchorVerdict(
        passed=passed or excursion, champion_commit=champion_commit,
        anchor=str(anchor_build), effect_pct=effect_pct, excursion=excursion,
        noise_floor_pct=noise_floor_pct, surface=comparison.surface,
        pairs=comparison.pairs, evidence={"comparison": comparison.to_dict(),
                                          "anchor_digest": a_dig, "fresh_digest": f_dig},
        detail=(f"{head} — inside the floor: the anchor slot holds the champion" if passed
                else f"{head} — ABOVE the floor with IDENTICAL code digests: the "
                f"anchor IS the champion (R21-10 instrument excursion); continuing"
                if excursion
                else f"{head} — two builds of ONE commit cannot differ by this much. The "
                f"anchor slot does NOT hold the champion, so every effect measured "
                f"against it is void. Aborting rather than making more"))
    if on_verdict is not None: on_verdict(verdict)  # noqa: E701 — loop budget
    if not verdict.passed:
        raise loop_mod.RunAborted(verdict.detail)
    return verdict


__all__ = ["MECHANISM_ID", "SCRATCH_BUILD", "AnchorVerdict", "verify"]
