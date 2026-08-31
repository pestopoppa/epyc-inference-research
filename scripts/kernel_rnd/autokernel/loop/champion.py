#!/usr/bin/env python3
"""Startup refusal: the loop optimises THE single champion branch, or does not start.

THE INVARIANT. Inference research owns ONE champion branch, aggregating ALL
improvement work — manual research AND AutoKernel keeps — between production
promotions. There is never a second base worth optimising, so a loop pointed at
anything else is spending the device on work that can never be promoted.

WHY THIS EXISTS. On 2026-08-30 the rebuilt loop was seeded from bare frozen v9
(`0db32c06`) as a NEW branch `ak/loop-champion-20260828`, while the real champion —
`ak/champion/llama-cpp-0db32c06e3e5` at `270b48ed`, +3371/−146 over v9, carrying
DFlash2 + iqk + speculative-decoding work admitted through the manual pipeline — sat
one branch over in the same repo. Runs 18–20 optimised the wrong base. Nothing
checked, because the invariant lived in sessions' memory of the lifecycle phase, not
in code. This module is the check; the operator's words on discovering it were
"I NEVER WANT TO SEE YOU MAKE THIS MISTAKE EVER AGAIN."

WHAT IS CHECKED, before any claim is taken or GPU work starts:

  * the worktree HEAD is ATTACHED to the named champion branch, at its tip. Attachment
    is the stronger half: the sequential path commits with `branch="HEAD"`, so a
    worktree merely parked AT the tip while detached (or attached to a sibling at the
    same commit — the incident's day-zero state) would grow keeps that the champion
    branch never receives. Attached-at-tip is what makes the invariant hold BY
    CONSTRUCTION after startup, not just at it.
  * the anchor build descends from this champion. `pool.promote_anchor` writes
    `provenance.json` naming the `champion_commit` it built; that commit must be
    ancestor-or-equal of the worktree HEAD. An `anchor-gen-*` directory carries the
    file by contract, so its absence there is a refusal outright; a hand-built anchor
    without one is unattestable by git and needs `--allow-unverified-anchor`, said
    loudly.
  * `--champion-branch <other>` is itself the risk (someone points it at a fork), but
    the canonical name legitimately changes at the next production promotion — so a
    branch that has DIVERGED from the canonical one warns loudly and does not refuse.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

#: THE single champion branch, as of the 2026-08-31 reconciliation. This name changes
#: at a production promotion and nowhere else.
CANONICAL_BRANCH = "ak/champion/llama-cpp-0db32c06e3e5"

#: Every refusal names its origin, so the reader six months from now knows this is a
#: scar, not ceremony.
INCIDENT = ("This guard exists because of the 2026-08-31 single-champion incident: "
            "runs 18-20 optimised bare v9 on a sibling branch while the real "
            "champion sat one branch over, unchecked.")


class StartupRefused(SystemExit):
    """The worktree or anchor is not the single champion. Non-zero exit, loud message.

    A `SystemExit` subclass so an unhandled refusal exits 1 with the message on
    stderr — no traceback burying the two SHAs the operator needs to read.
    """


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=600)


def verify_worktree(worktree: Path, branch: str) -> str:
    """HEAD must BE the champion branch — attached to it, at its tip. Returns the sha."""
    tip = _git(worktree, "rev-parse", "--verify", f"refs/heads/{branch}")
    if tip.returncode != 0:
        raise StartupRefused(f"REFUSED: champion branch '{branch}' does not exist "
                             f"in {worktree}. {INCIDENT}")
    head = _git(worktree, "rev-parse", "HEAD").stdout.strip()
    attached = _git(worktree, "symbolic-ref", "-q", "HEAD").stdout.strip()
    # Keeps land on HEAD, so a detached tree — or one attached to a sibling — would
    # grow a base the champion never receives. Check out the champion branch, or name
    # the intended one with --champion-branch.
    if head != tip.stdout.strip() or attached != f"refs/heads/{branch}":
        raise StartupRefused(
            f"REFUSED: worktree {worktree} is on {attached or 'a detached HEAD'} at "
            f"{head}, not the champion branch '{branch}' at {tip.stdout.strip()}. "
            f"{INCIDENT}")
    return head


def verify_anchor(anchor_build: Path, worktree: Path, head: str, *,
                  allow_unverified: bool = False) -> None:
    """The anchor binary must descend from THIS champion, or be waived by name.

    `provenance.json` is written by `pool.promote_anchor` beside every build it makes,
    naming the `champion_commit`. Ancestor-or-equal of HEAD is the bar: an anchor a
    few keeps old is a stale baseline the A/A guard will catch, but an anchor from a
    DIFFERENT lineage makes every effect a comparison across bases — run 18's 114
    void measurements, refused at startup instead of found six hours in. An attested
    anchor from the wrong lineage is refused even under `allow_unverified`: the flag
    waives a MISSING attestation on a hand-built dir, never a wrong one.

    (`built_at` in the real records currently holds a path, not a timestamp — known
    defect R18-B. Nothing here reads it.)
    """
    prov = Path(anchor_build) / "provenance.json"
    if not prov.is_file():
        # anchor-gen-* dirs carry provenance BY CONTRACT; absence there is a defect,
        # not a hand-built anchor, and no flag talks past it.
        if Path(anchor_build).name.startswith("anchor-gen-"):
            raise StartupRefused(f"REFUSED: {anchor_build} carries no provenance.json,"
                                 f" which anchor-gen-* dirs write by contract "
                                 f"(pool.promote_anchor). {INCIDENT}")
        # Git cannot attest what a hand-built build directory was compiled from.
        if not allow_unverified:
            raise StartupRefused(
                f"REFUSED: hand-built anchor {anchor_build} has no provenance.json. "
                f"Pass --allow-unverified-anchor to proceed anyway. {INCIDENT}")
        # If this binary was not built from the champion, every effect is void.
        print(f"WARNING   anchor {anchor_build} is UNATTESTED: no provenance.json, "
              f"proceeding only because --allow-unverified-anchor was given")
        return
    named = str(json.loads(prov.read_text(encoding="utf-8")).get(
        "champion_commit") or "")
    if not named or _git(worktree, "merge-base", "--is-ancestor",
                         named, head).returncode != 0:
        # The anchor was built from a different lineage than the champion being
        # optimised, so every measured effect would compare across bases.
        raise StartupRefused(
            f"REFUSED: {prov} names champion_commit {named or '(missing)'}, not an "
            f"ancestor-or-equal of worktree HEAD {head}. {INCIDENT}")


def warn_divergence(worktree: Path, branch: str) -> None:
    """`--champion-branch <other>` never passes silently once it has forked.

    Not a refusal: the canonical name legitimately changes at the next production
    promotion, and a warning the operator reads beats a refusal they script around.
    An ancestor/descendant relationship (one tip IS the merge-base) is the legitimate
    rename case and stays quiet.
    """
    if branch == CANONICAL_BRANCH:
        return
    canon = _git(worktree, "rev-parse", "--verify", f"refs/heads/{CANONICAL_BRANCH}")
    if canon.returncode != 0:
        print(f"WARNING   --champion-branch {branch}: canonical {CANONICAL_BRANCH} "
              f"is absent in {worktree}, so divergence cannot be checked")
        return
    tip = _git(worktree, "rev-parse", branch).stdout.strip()
    base = _git(worktree, "merge-base", canon.stdout.strip(), tip).stdout.strip()
    # If it is not the post-promotion successor, this run optimises a fork.
    if base not in {canon.stdout.strip(), tip}:
        print(f"WARNING   --champion-branch {branch} has DIVERGED from "
              f"{CANONICAL_BRANCH} (merge-base {base[:12]} is neither tip). "
              f"{INCIDENT}")


def verify_startup(*, worktree: Path, branch: str, anchor_build: Path,
                   allow_unverified_anchor: bool = False) -> str:
    """The whole gate. Runs before the claim, the census, even the dry run's wiring
    proof — a refusal that costs four git reads must never queue behind anything.
    Returns the verified champion head."""
    head = verify_worktree(worktree, branch)
    warn_divergence(worktree, branch)
    verify_anchor(anchor_build, worktree, head, allow_unverified=allow_unverified_anchor)
    return head


__all__ = ["CANONICAL_BRANCH", "INCIDENT", "StartupRefused", "verify_anchor",
           "verify_startup", "verify_worktree", "warn_divergence"]
