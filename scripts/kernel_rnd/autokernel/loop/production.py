#!/usr/bin/env python3
"""Publish the champion's gain over the FROZEN PRODUCTION kernel, at every advance.

WHY THIS EXISTS
---------------
The Kernel R&D dashboard's headline is defined as one thing and one thing only: the
champion tree's measured gain over frozen production v9. It was taken ONCE, by hand,
for champion `5ad3e36d` (+8.524% decode). The champion has advanced twice since, so
the panel renders SUPERSEDED with no current figure -- correct, and useless. A
headline that only a person can refresh is a headline that is wrong by default.

It cannot be computed from what the loop already publishes. Every per-iteration
effect on that page is a MARGINAL against an anchor that advances on every keep, so
each has a different baseline; composing them arithmetically would claim a
measurement no run ever took. This program has made that error once already. The
number can only ever come from ONE direct A/B, both arms in the same session --
absolute throughput on this host is not comparable across sessions.

WHY IT IS AFFORDABLE. Run 19 produced 2 keeps in 11 hours, and a paired A/B at 20
pairs is ~181 s of device time: about 0.09%. The operator's ruling: "the champion
advances so rarely that performing the proper A/B measurement whenever the champion
advances is totally reasonable."

TWO BUILDS THAT DO NOT HAPPEN HERE
  * The CHAMPION arm is the anchor slot `pool.promote_anchor` just built from the
    champion commit, and which `anchor.verify` just proved IS the champion. Building
    it again would pay a second full build for a binary already on disk and already
    A/A-verified -- and a second build is a second chance to bench something that is
    not the champion.
  * The BASELINE arm is frozen production, resolved LIVE from the canonical frozen
    tree at every refresh (a promotion advances that tree, and the headline must
    follow it -- never a stale pinned sha). Its build is cached per commit: built at
    most once per freeze, then reused. A missing cache means "build it once", never
    "rebuild every time"; a new promotion means one cache miss, one build.

WHY A FAILURE HERE MUST NOT STOP THE RUN. This is a REPORTING refresh, not a
correctness gate. `anchor.verify` aborts because a wrong anchor voids every
measurement after it; nothing downstream of this depends on it at all. If the
baseline is missing or the A/B fails, the previous bundle stays exactly where it is
and the panel keeps reading SUPERSEDED -- which is the correct degraded state, and
strictly better than killing a run that is otherwise producing science. So
`refresh` NEVER raises and never uses `loop.RunAborted`.

THE EFFECT IS NOT WRITTEN INTO THE ATTEMPT ROW. `to_attempt` records the refresh in
durable memory with `effect_fraction: None` and the number in the prose. Every other
row in that table is a marginal against the advancing anchor, and the planner reads
that table back as its memory; dropping a CUMULATIVE +8.5% into it, in the column
the planner compares against, is the composition error one level down.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

from . import status

#: The contract `dashboard/loop_status.py` reads. Its reader REFUSES a bundle whose
#: `baseline.commit` is not the frozen production commit rather than relabelling it,
#: so these two constants are load bearing: they are the only reason the number
#: shown under that heading is the number that heading names.
SCHEMA = "epyc.autokernel.champion_vs_production.v1"
FILENAME = "champion-vs-production.json"
MECHANISM_ID = "champion-vs-production"

#: The canonical FROZEN production tree. The baseline commit is resolved LIVE from its
#: HEAD, never pinned here: the promotion process is what advances that tree, so the
#: headline's baseline follows a promotion with no constant for anyone to forget.
#: (Operator, 2026-08-31: "once we promote a new frozen version in the future, the
#: comparison should be against the newly promoted version, NOT stale v9. This is a
#: classic mistake." The previous revision of this file had made it: a hardcoded
#: `BASELINE_COMMIT` that would have silently kept measuring v9 past a v10 freeze.)
FROZEN_TREE = Path("/mnt/raid0/llm/llama.cpp")
#: The production branch contract (`production-consolidated-v9`, future `-v10`, ...) —
#: the same family `scripts/session/verify_llama_cpp.sh` enforces in the root repo. A
#: tree on any other branch is NOT the freeze, and no headline is measured against it.
FROZEN_BRANCH_PREFIX = "production-consolidated-"
#: Baseline builds are cached PER COMMIT under this root, `production-baseline-<sha12>`.
#: A promotion is then a cache miss that builds once, never an overwrite.
BASELINE_ROOT = Path("/mnt/raid0/llm/tmp")
#: v9's verified prebuilt, ADOPTED as the cache entry for exactly this sha (verified
#: 2026-08-30: 584/584 CPU symbols, 918/918 GPU device kernels identical to the shipped
#: libraries). A legacy-path fallback rather than a rebuild or a symlink; it is never
#: used for any other commit.
LEGACY_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
LEGACY_BUILD = Path("/mnt/raid0/llm/tmp/v9v-build-base")
#: Source for the at-most-once build. NOT `FROZEN_TREE`: the production tree is frozen
#: and may not be built in. The builder checks this copy's HEAD against the LIVE
#: resolved commit, so after a promotion a stale copy refuses loudly instead of
#: publishing a number under the new freeze's name.
BASELINE_TREE = Path("/mnt/raid0/llm/tmp/v9v-base-tree")
#: Written into a baseline build this module drove, so a cache that is later pointed
#: at some other tree is caught instead of published under production's name.
PROVENANCE = "baseline-provenance.json"
#: Days, not minutes: a cumulative A/B is a deliberate, expensive act rather than a
#: per-iteration beat. Matches the reader's own default envelope.
STALE_AFTER_S = 14 * 86400


class Unavailable(RuntimeError):
    """The measurement could not be taken. Reporting only -- the run continues."""


def _read(tree: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(tree), *args],
                          capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise Unavailable(f"cannot resolve the frozen production kernel from {tree}: "
                          f"{done.stderr.strip()[:200]}")
    return done.stdout.strip()


def resolve_frozen(tree: Path = FROZEN_TREE) -> tuple[str, str]:
    """`(commit, label)` of the frozen production kernel, resolved LIVE from the tree.

    The branch is cross-checked against the production contract FIRST: a resolver that
    returned whatever HEAD it found would headline against an experimental checkout the
    day someone left the tree on one. Failing the check raises `Unavailable`, so the
    refresh is refused and recorded while the run continues.
    """
    label = _read(tree, "branch", "--show-current")
    # Refusing to publish a headline measured against an unknown tree.
    if not label.startswith(FROZEN_BRANCH_PREFIX):
        raise Unavailable(f"{tree} is on branch '{label or '(detached)'}', not the "
                          f"production contract '{FROZEN_BRANCH_PREFIX}*'")
    return _read(tree, "rev-parse", "HEAD"), label


def baseline_slot(commit: str, *, root: Path = BASELINE_ROOT,
                  legacy: Path = LEGACY_BUILD) -> Path:
    """The cache directory for one frozen commit's baseline build.

    Keyed by commit, so a promotion is a cache miss that builds once. The v9 prebuilt
    is adopted in place for exactly `LEGACY_COMMIT` (see the constant), because it is
    already verified against production's shipped libraries and rebuilding a
    known-good tree is the opposite of the cache contract.
    """
    if commit == LEGACY_COMMIT and is_built(legacy):
        return legacy
    return root / f"production-baseline-{commit[:12]}"


def is_built(build_dir: Path | str) -> bool:
    """Whether a build directory holds the binary this comparison needs."""
    return (Path(build_dir) / "bin" / "llama-bench").is_file()


def declared_commit(build_dir: Path | str) -> str | None:
    """The commit a cached baseline build declares, or None if it declares nothing.

    None is ACCEPTED: the verified prebuilt predates this stamp, and refusing it
    would force a rebuild of a tree that is known-good -- the opposite of the cache
    contract. A stamp that DISAGREES is refused, because that is the only reading
    that means someone re-pointed the cache.
    """
    try:
        body = json.loads((Path(build_dir) / PROVENANCE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return body.get("commit") if isinstance(body, dict) else None


@dataclass(frozen=True)
class Refresh:
    """One attempt at the headline. `published` False is a normal, survivable state."""
    published: bool
    reason: str
    path: Path | None = None
    effect_fraction: float | None = None

    def to_attempt(self) -> dict[str, Any]:
        return {"status": ("champion_vs_production" if self.published
                           else "champion_vs_production_unavailable"),
                "mechanism_id": MECHANISM_ID,
                # Deliberately None -- see the module docstring. The number is in the
                # bundle and in `reason`, never in the planner's marginals column.
                "effect_fraction": None,
                "reason": self.reason}


def _ensure_baseline(baseline_build: Path, commit: str, label: str,
                     build_baseline: Callable[[Path, str], Any] | None,
                     on_step: Callable[[str], Any]) -> None:
    """Frozen production, built at most once PER COMMIT. Raises `Unavailable`, never
    aborts. `commit`/`label` are the LIVE-resolved freeze, and the builder receives
    the commit so it can refuse a source tree that has not followed the promotion."""
    declared = declared_commit(baseline_build)
    if declared is not None and declared != commit:
        raise Unavailable(
            f"the cached baseline build at {baseline_build} declares commit "
            f"{declared[:12]}, not the frozen production kernel {commit[:12]}; "
            f"publishing it would put a number under a heading it does not measure")
    if is_built(baseline_build):
        return
    if build_baseline is None:
        raise Unavailable(
            f"there is no baseline build at {baseline_build} and no builder was "
            f"wired, so there is nothing to measure the champion against")
    on_step("champion-vs-production: building the frozen production baseline (once)")
    verdict = build_baseline(baseline_build, commit)
    if not getattr(verdict, "passed", False) or not is_built(baseline_build):
        raise Unavailable(
            f"the frozen production kernel {commit[:12]} would not build into "
            f"{baseline_build} ({getattr(verdict, 'reason', '') or 'no reason given'})")
    status.write_json(Path(baseline_build), PROVENANCE,
                      {"commit": commit, "label": label},
                      prefix=".prov-")


def refresh(*, store: Path, champion_commit: str, champion_build: Path,
            compare: Callable[[Path, Path], Any],
            baseline_build: Path | None = None,
            baseline_root: Path = BASELINE_ROOT,
            resolve: Callable[[], tuple[str, str]] = resolve_frozen,
            build_baseline: Callable[[Path, str], Any] | None = None,
            on_step: Callable[[str], Any] = lambda _label: None,
            now: Callable[[], str] = status._now) -> Refresh:
    """Measure the champion against frozen production and publish it. NEVER raises.

    `compare(baseline_build, champion_build) -> bench.Comparison` is injected, so this
    is exercised with no GPU, no ROCm and no build. The BASELINE is the first arm, so
    a positive effect reads as "the champion is faster than production" -- the sign
    the headline is stated in.

    Everything the bundle says about the measurement is taken from the `Comparison`
    the benchmark returned, never from a caller's parameters: a surface or pair count
    passed in alongside the comparison is a second source of truth for one fact, and
    the one that gets published is the one nobody measured.
    """
    try:
        # Resolved LIVE, inside the containment: a promotion moves the frozen tree, so
        # a resolver failure (or a tree off the production branch contract) refuses
        # THIS refresh and never ends the run.
        frozen_commit, frozen_label = resolve()
        slot = (Path(baseline_build) if baseline_build is not None
                else baseline_slot(frozen_commit, root=baseline_root))
        _ensure_baseline(slot, frozen_commit, frozen_label, build_baseline, on_step)
        on_step("champion-vs-production: A/B against the frozen production kernel")
        comparison = compare(slot, Path(champion_build))
        # Per-champion, so publishing a new bundle never overwrites the raw record the
        # PREVIOUS bundle points at. `evidence` in a superseded bundle must still
        # resolve, or the number it carries stops being auditable the moment it ages.
        evidence = status.write_json(
            Path(store), f"champion-vs-production.{champion_commit[:12]}.json",
            comparison.to_dict(), prefix=".cvp-")
        target = status.write_json(Path(store), FILENAME, {
            "schema": SCHEMA,
            "generated_at": now(),
            "stale_after_s": STALE_AFTER_S,
            "baseline": {"commit": frozen_commit, "label": frozen_label,
                         "build": str(slot)},
            "champion": {"commit": champion_commit, "build": str(champion_build)},
            "effect_fraction": float(comparison.effect),
            "metric": f"{comparison.surface}_tok_s",
            "metric_direction": "higher_better",
            "surface": comparison.surface,
            "pairs": comparison.pairs,
            "noise_floor_pct": comparison.noise_floor_pct,
            "evidence": str(evidence),
            "mechanism_id": MECHANISM_ID,
        }, prefix=".cvp-")
        # Built INSIDE the containment. The headline sentence formats fields off the
        # comparison, and a formatting error on the SUCCESS path would otherwise be
        # the one exception in this module that still reached the loop.
        published = Refresh(
            True,
            f"champion {champion_commit[:12]} measures "
            f"{comparison.effect * 100.0:+.3f}% against frozen production "
            f"{frozen_commit[:12]} ({frozen_label}) over {comparison.pairs} "
            f"{comparison.surface} pairs, floor {comparison.noise_floor_pct}%",
            target, float(comparison.effect))
    except Unavailable as exc:
        return Refresh(False, f"champion-vs-production NOT refreshed: {exc}. The "
                              f"previous bundle stands, so the headline reads "
                              f"SUPERSEDED rather than a number measured against "
                              f"something that is not frozen production")
    except Exception as exc:  # noqa: BLE001 -- reporting must never end a run
        return Refresh(False, f"champion-vs-production refresh FAILED "
                              f"({type(exc).__name__}: {exc}). The previous bundle "
                              f"stands and the run continues; this is a reporting "
                              f"refresh, not a correctness gate")
    return published


__all__ = ["BASELINE_ROOT", "BASELINE_TREE", "FILENAME", "FROZEN_BRANCH_PREFIX",
           "FROZEN_TREE", "LEGACY_BUILD", "LEGACY_COMMIT", "MECHANISM_ID", "PROVENANCE",
           "SCHEMA", "STALE_AFTER_S", "Refresh", "Unavailable", "baseline_slot",
           "declared_commit", "is_built", "refresh", "resolve_frozen"]
