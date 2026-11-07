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
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys

#: THE single champion branch, as of the 2026-08-31 reconciliation. This name changes
#: at a production promotion and nowhere else.
#:
#: RESEEDED 2026-09-22 at the v10 freeze. `ak/champion/llama-cpp-0db32c06e3e5`
#: was promoted WHOLESALE -- its tip IS v10 -- so the new champion is cut from
#: `production-consolidated-v10` @ `ffc1bac82` and its standing versus production
#: is ZERO by construction. The retired branch is kept as the tag
#: `retired/ak-champion-llama-cpp-0db32c06e3e5`. That zero is the testable
#: success criterion of the reset: a headline gain published before the first
#: keep of the new cycle proves the reseed did not happen.
CANONICAL_BRANCH = "ak/champion/llama-cpp-ffc1bac82eec"

#: Every refusal names its origin, so the reader six months from now knows this is a
#: scar, not ceremony.
INCIDENT = ("This guard exists because of the 2026-08-31 single-champion incident: "
            "runs 18-20 optimised bare v9 on a sibling branch while the real "
            "champion sat one branch over, unchecked.")


class StartupRefused(SystemExit):
    #: typed reason (see REASON_*); None for the legacy untyped refusals.
    reason: str | None = None
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
    verify_contains_champion(worktree, head)
    return head


def _verify_experimental_identity(anchor_build: Path, worktree: Path, head: str) -> None:
    """Verify the existing manual CPU build receipt without relabelling it provenance."""
    identity_path = anchor_build / "IDENTITY.json"
    try:
        fd = os.open(identity_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
                raise StartupRefused("REFUSED: experimental anchor identity is not a bounded regular file")
            raw = stream.read(1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > 1024 * 1024 or (before.st_dev, before.st_ino, before.st_size,
                                     before.st_mtime_ns) != (after.st_dev, after.st_ino,
                                                            after.st_size, after.st_mtime_ns):
            raise StartupRefused("REFUSED: experimental anchor identity changed while read")
        row = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StartupRefused(f"REFUSED: cannot reopen experimental anchor identity: {exc}") from exc
    required = {"schema", "kind", "head", "source", "source_status", "build",
                "build_dir", "files"}
    if not isinstance(row, dict) or set(row) != required \
            or row.get("schema") != "epyc.champion-candidate-build.v1" \
            or row.get("kind") != "serving" or row.get("head") != head \
            or Path(str(row.get("source"))).resolve() != worktree.resolve() \
            or row.get("source_status") not in {"", "clean"}:
        raise StartupRefused("REFUSED: experimental anchor identity differs from selected source/head")
    if _git(worktree, "status", "--porcelain").stdout:
        raise StartupRefused("REFUSED: experimental source worktree is not clean")
    files = row.get("files")
    if not isinstance(files, list) or not (1 <= len(files) <= 64):
        raise StartupRefused("REFUSED: experimental anchor identity has invalid file inventory")
    root = anchor_build.resolve()
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"} \
                or not isinstance(item["path"], str) or not isinstance(item["sha256"], str) \
                or len(item["sha256"]) != 64 or not isinstance(item["size"], int) \
                or isinstance(item["size"], bool) or item["size"] < 0:
            raise StartupRefused("REFUSED: malformed experimental anchor file identity")
        path = Path(item["path"])
        try:
            if path.resolve().parent != root / "bin" or path.name in seen:
                raise StartupRefused("REFUSED: experimental anchor inventory escapes or repeats bin files")
            seen.add(path.name)
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != item["size"]:
                    raise StartupRefused("REFUSED: experimental anchor file size/type differs")
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
            if digest != item["sha256"] or (before.st_dev, before.st_ino, before.st_size,
                                            before.st_mtime_ns) != (after.st_dev, after.st_ino,
                                                                   after.st_size, after.st_mtime_ns):
                raise StartupRefused("REFUSED: experimental anchor file identity differs")
        except OSError as exc:
            raise StartupRefused(f"REFUSED: cannot reopen experimental anchor file: {exc}") from exc
    if "llama-server" not in seen:
        raise StartupRefused("REFUSED: experimental anchor identity omits llama-server")


def verify_anchor(anchor_build: Path, worktree: Path, head: str, *,
                  allow_unverified: bool = False,
                  experimental_identity: bool = False) -> None:
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
        if experimental_identity:
            _verify_experimental_identity(Path(anchor_build), Path(worktree), head)
            return
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


#: Typed refusal reasons (AK-TOOL-CHAMPION-CONTEXT). A caller or test branches on
#: `StartupRefused.reason`, never on message text.
REASON_STALE_ANCHOR = "stale_anchor"
REASON_NO_ATTESTATION = "anchor_unattested"


#: Explicit launch override (run.py is custody-pinned, so no argv flag): a NON-EMPTY
#: reason in this env var lets a non-champion anchor start (calibration, replay of an
#: older candidate). The reason is printed; an empty value is no override.
OVERRIDE_ENV = "AK_NON_CHAMPION_ANCHOR_REASON"
REASON_NOT_CHAMPION = "anchor_not_current_champion"


def _refuse(reason: str, message: str) -> None:
    refusal = StartupRefused(f"REFUSED [{reason}]: {message} Override: set "
                             f"{OVERRIDE_ENV}='<why>'. {INCIDENT}")
    refusal.reason = reason
    raise refusal


def project_champion_tip(worktree: Path) -> str:
    """The CURRENT project champion = the tip of `CANONICAL_BRANCH` (the branch the
    kernel manifests name, docs/reference/kernel-manifests/llama-v11-manifest.md),
    resolved from git at launch -- never a hardcoded sha."""
    return _git(worktree, "rev-parse", "--verify", "-q",
                f"refs/heads/{CANONICAL_BRANCH}^{{commit}}").stdout.strip()


def verify_contains_champion(worktree: Path, head: str) -> None:
    """LAUNCH-TIME refusal on EVERY startup path that verifies the worktree (including
    the store-anchor path, which never reaches `verify_anchor_current`): the verified
    head must contain the current project champion tip. Waived only by a non-empty
    `OVERRIDE_ENV`."""
    override = os.environ.get(OVERRIDE_ENV, "").strip()
    tip = project_champion_tip(worktree)
    if not tip:
        problem = (f"{CANONICAL_BRANCH} does not resolve in {worktree}; the current "
                   f"project champion is unknown.")
    elif _git(worktree, "merge-base", "--is-ancestor", tip, head).returncode != 0:
        problem = (f"worktree head {head[:12]} does not contain the current project "
                   f"champion {CANONICAL_BRANCH} @ {tip[:12]}.")
    else:
        return
    if override:
        print(f"WARNING: champion-currency refusal OVERRIDDEN by {OVERRIDE_ENV}="
              f"{override!r}: {problem}", file=sys.stderr)
        return
    _refuse(REASON_NOT_CHAMPION, problem)


def verify_anchor_current(anchor_build: Path, worktree: Path, head: str, *,
                          experimental_identity: bool = False,
                          allow_unverified: bool = False) -> None:
    """LAUNCH-TIME refusal (AK-TOOL-CHAMPION-CONTEXT, narrowed): the anchor must be the
    current project champion, or carry it.

    The anchor commit is provenance `champion_commit` (or, for an experimental-identity
    build without provenance, the IDENTITY.json-proven `head`). It must (a) equal the
    verified worktree head and (b) contain the current tip of `CANONICAL_BRANCH`
    (equal, or an experimental branch freshly cut from it). An experimental branch cut
    from an OLD champion (C5 after the C7r CAS) is refused `anchor_not_current_champion`;
    a provenance commit behind head is `stale_anchor`; no attestation is
    `anchor_unattested`. Any refusal is waived only by a non-empty `OVERRIDE_ENV`."""
    override = os.environ.get(OVERRIDE_ENV, "").strip()
    try:
        _verify_anchor_current(Path(anchor_build), Path(worktree), head,
                               experimental_identity=experimental_identity,
                               allow_unverified=allow_unverified)
    except StartupRefused as refusal:
        if not override:
            raise
        print(f"WARNING: champion-currency refusal OVERRIDDEN by {OVERRIDE_ENV}="
              f"{override!r}: {refusal}", file=sys.stderr)


def _verify_anchor_current(anchor_build: Path, worktree: Path, head: str, *,
                           experimental_identity: bool, allow_unverified: bool = False) -> None:
    prov = anchor_build / "provenance.json"
    named = ""
    if prov.is_file():
        try:
            named = str(json.loads(prov.read_text(encoding="utf-8")).get(
                "champion_commit") or "")
        except (OSError, ValueError):
            named = ""
    elif experimental_identity:
        named = head  # `_verify_experimental_identity` proved IDENTITY.json head == head
    if not named and allow_unverified:
        # --allow-unverified-anchor is the explicit, logged waiver for a hand-built anchor
        # with no provenance; champion currency is still enforced on the worktree head by
        # `verify_contains_champion` (called from `verify_worktree`).
        return
    if not named:
        _refuse(REASON_NO_ATTESTATION, f"anchor {anchor_build} carries no champion_commit, "
                f"so it cannot be proven to be the CURRENT champion.")
    resolved = _git(worktree, "rev-parse", "--verify", f"{named}^{{commit}}").stdout.strip()
    if resolved != head:
        _refuse(REASON_STALE_ANCHOR, f"anchor {anchor_build} was built from {named}, but the "
                f"verified worktree head is {head}. Rebuild the anchor (pool.promote_anchor).")
    tip = project_champion_tip(worktree)
    if not tip:
        _refuse(REASON_NOT_CHAMPION, f"{CANONICAL_BRANCH} does not resolve in {worktree}; "
                f"the current project champion is unknown.")
    if _git(worktree, "merge-base", "--is-ancestor", tip, head).returncode != 0:
        _refuse(REASON_NOT_CHAMPION, f"anchor {head[:12]} does not contain the current "
                f"project champion {CANONICAL_BRANCH} @ {tip[:12]}.")


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
                   allow_unverified_anchor: bool = False,
                   experimental_identity: bool = False,
                   require_current_anchor: bool = True) -> str:
    """The whole gate. Runs before the claim, the census, even the dry run's wiring
    proof — a refusal that costs four git reads must never queue behind anything.
    Returns the verified champion head."""
    head = verify_worktree(worktree, branch)
    warn_divergence(worktree, branch)
    verify_anchor(anchor_build, worktree, head, allow_unverified=allow_unverified_anchor,
                  experimental_identity=experimental_identity)
    if require_current_anchor:
        verify_anchor_current(anchor_build, worktree, head,
                              experimental_identity=experimental_identity,
                              allow_unverified=allow_unverified_anchor)
    return head


__all__ = ["CANONICAL_BRANCH", "INCIDENT", "StartupRefused", "verify_anchor",
           "OVERRIDE_ENV", "REASON_NO_ATTESTATION", "REASON_NOT_CHAMPION", "REASON_STALE_ANCHOR",
           "project_champion_tip", "verify_anchor_current", "verify_contains_champion",
           "verify_startup", "verify_worktree", "warn_divergence"]
