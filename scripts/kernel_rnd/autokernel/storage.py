#!/usr/bin/env python3
"""storage.py — the AutoKernel storage plane: durability classes, quota, tombstoned expiry (§5.8, §3.7).

WHY THIS MODULE EXISTS
----------------------
Two constraints meet here and neither may be relaxed to satisfy the other.

  * **Evidence must be durable, not merely hashed.** `MEASUREMENT.md:146-156`
    (operator-ratified 2026-08-02) requires the evidence behind any ratified or
    production-affecting claim to live in-repo under
    `epyc-inference-research/data/<campaign>/` with a `SHA256SUMS` and a README
    stating what was measured, when, and which claim it backs; scratch paths
    (`/mnt/raid0/llm/tmp/...`) MUST NOT be the citation of record. The origin was
    not hypothetical: the master registry was found citing 158 unique scratch
    paths, including the MMMU-250 result that had gated a live vision-model
    cutover. A hash over an artifact that no longer exists is an assertion, not a
    verification, and it degrades silently. Here a citation resolving to a
    scratch root is an ERROR — `assert_not_scratch()` raises and
    `verify_durability()` returns FAIL — never a warning that gets triaged later.

  * **The disk is nearly full and the loop cannot stop being honest about it.**
    Measured on this host 2026-08-03: `/mnt/raid0` is 3.75 TiB with **157.7 GiB
    free (96% used)**; `epyc-inference-research/data/` is **118 GiB, of which
    0.33 GiB is tracked in git** — i.e. 99.7% of the evidence tree is already
    outside version control; the 13 llama worktrees total 41 GiB, the largest
    single one 15 GiB. A campaign that creates one build worktree therefore
    consumes ~10% of all remaining headroom in a single step. Without a plane,
    the loop halts on ENOSPC within a handful of campaigns (§5.8), and the
    tempting fix — delete something — collides with invariant 7 ("all outcomes
    are durable") and with `MEASUREMENT.md:223-229`, which puts reclamation under
    OPERATOR authority.

The reconciliation, and the whole point of this module: **the primary RECORD
survives even when the artifact does not.** Reclaiming an expirable artifact
writes a tombstone through the journal *before* the bytes go, carrying the hash,
durability class, size, kind, rule id and reason. That keeps expiry consistent
with the prime directive at `MEASUREMENT.md:173-176` — *never destroy primary
records; demote, label, or re-derive interpretations* — because what is destroyed
is a derived blob whose existence is now itself a durable record.

WHAT A DURABILITY CLASS IS FOR (§3.7)
-------------------------------------
`carried_in_git` / `durable_untracked` / `hash_and_provenance_only` exist so a
later verifier can tell **a defect from an expected absence**. Without the class,
a missing 15 GiB build tree and a missing 4 KiB summary.json look identical: both
are "path does not resolve". With it, the first is expected and the second is a
loss. This is why `classify()` REFUSES to classify a path that does not exist:
inferring `hash_and_provenance_only` from absence is exactly the laundering
operation the class was invented to prevent.

DELETION AUTHORITY
------------------
`MEASUREMENT.md:223-229` makes disk hygiene "an operator call, not
contamination". `expire_artifact()` therefore:

  * refuses any retention class other than `expirable`, and any expirable kind
    outside the three §5.8 names;
  * refuses anything whose durability class is `carried_in_git` (git already
    carries those bytes; deleting the worktree copy reclaims nothing and loses
    the checkout);
  * deletes only strictly beneath a campaign-owned root the CALLER declares —
    the default is the empty tuple, so a caller that declares nothing can delete
    nothing;
  * categorically denies the frozen production trees and any `.git` directory,
    regardless of what roots were declared;
  * requires every precondition fact its rule names, and refuses when a fact is
    absent — an unreadable fact is not a satisfied fact;
  * is **dry-run by default**, matching `kernel_store.py`'s purge/rewind, and
    needs an explicit `force=True` to touch anything.

Journal writes are two-phase (`intent` then `reclaimed`) sharing one
content-addressed `tombstone_id`. A crash between them leaves an intent with no
completion, which is a detectable, recoverable state; the one-phase alternatives
either lose the record entirely (delete-then-journal) or leave a tombstone
asserting bytes are gone when they are not (journal-then-crash).

THIRD OUTCOME
-------------
`verify_durability()` and `check_evidence_root_layout()` return
`schemas.Check`, whose outcome is PASS / FAIL / COULD_NOT_CHECK. "We could not
determine whether this citation is durable" is reported as itself. It is not a
pass (which would hide a real loss) and not a fail (which would manufacture one).

This module performs filesystem I/O and, in `GitTrackedIndex` only, one
read-only `git ls-files` query. It launches no service, runs no inference, and
runs no benchmark.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# --- schemas import -------------------------------------------------------
# Package-relative FIRST, `sys.path` only as a genuine last resort. The earlier
# preamble was an unconditional `sys.path.insert(0, <autokernel dir>)` followed
# by a flat `import schemas`, and it broke the package in two ways that only
# appear when more than one AutoKernel module is loaded in one process:
#
#   * `autokernel.schemas` and a flat `schemas` are two DIFFERENT module objects
#     executed from the same file, so `storage.Check is schemas.Check` was False
#     and every `isinstance(verdict, schemas.Check)` across the storage seam
#     silently said no. One source of truth that exists twice is not one source
#     of truth (AutoPilot scar item 12, §2.5 — ambient import identity);
#   * it put `autokernel/` on `sys.path[0]`, so `import resource` anywhere later
#     in the process resolved to `autokernel/resource/__init__.py` instead of the
#     STDLIB `resource` module. `resource/__init__.py` documents that exact
#     shadowing as forbidden; a sibling module was causing it at import time.
#
# The identity assertion is the mitigation for the fallback branch: the module
# we bind MUST be the schemas.py sitting next to this file, or we refuse to
# import at all rather than run against someone else's contracts.
_SCHEMAS_PATH = Path(__file__).resolve().parent / "schemas.py"
try:
    from . import schemas  # noqa: E402
except ImportError:  # imported flat, e.g. after sys.path.insert(<this dir>)
    _HERE = str(_SCHEMAS_PATH.parent)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import schemas  # type: ignore[no-redef]  # noqa: E402

if Path(schemas.__file__).resolve() != _SCHEMAS_PATH:
    raise ImportError(
        f"autokernel.storage bound a foreign schemas module: {schemas.__file__} "
        f"is not {_SCHEMAS_PATH}"
    )

COULD_NOT_CHECK = schemas.COULD_NOT_CHECK
FAIL = schemas.FAIL
PASS = schemas.PASS
Check = schemas.Check

# =============================================================================
# Roots and thresholds — every number below was measured on this host, not
# assumed. Re-measure before changing one; the comment names what it was.
# =============================================================================

# scripts/kernel_rnd/autokernel/storage.py -> the epyc-inference-research root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# The mandated evidence home (`MEASUREMENT.md:146-156`).
EVIDENCE_DIRNAME = "data"

# Scratch roots. A citation under any of these is non-durable BY DEFINITION.
# This tuple is the one thing in the file that must never acquire an exemption:
# a blacklist grows an exemption every time it is inconvenient, and the one
# exemption that must never be grantable is the scratch root. Kept byte-equal to
# `scripts/validate/check_evidence_durability.py:EPHEMERAL_ROOTS`, which is the
# ratified enforcer named by the constitution; `test_storage.py` imports that
# file and asserts the two agree, because two independent copies of a security
# boundary is how one of them quietly loses an entry.
EPHEMERAL_ROOTS = (
    "/mnt/raid0/llm/tmp",
    "/tmp",
    "/var/tmp",
    "/dev/shm",
    "/run/user",
)

# Frozen production kernel trees (CLAUDE.md, 2026-07-25 v8 + 2026-07-31 speech
# freeze). Invariant 3: no actor modifies a production tree. Expiry denies these
# unconditionally — before consulting the caller's declared roots, so a
# mis-declared root cannot authorise reclaiming production.
PRODUCTION_TREES = (
    "/mnt/raid0/llm/llama.cpp",
    "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp",
)

# Above this, an artifact is recorded hash-and-provenance-only rather than
# carried. Calibrated, not guessed: the largest file currently tracked under
# `data/` is 35.4 MiB (a summary.json), the whole tracked evidence tree is
# 0.33 GiB, and `.git` is 633 MiB. 100 MiB sits an order of magnitude above
# real evidence and an order of magnitude below the artifacts §5.8 calls
# "permanent, large" (multi-GiB imatrix files, 13-15 GiB build trees).
DEFAULT_CARRY_THRESHOLD_BYTES = 100 * 1024 * 1024

_GIB = 1024 ** 3

# Host headroom floor. Measured 2026-08-03: 157.7 GiB free of 3.75 TiB.
DEFAULT_HEADROOM_FLOOR_GB = 50.0

# The largest single allocation a campaign makes in one step: a llama build
# worktree. Measured 2026-08-03 — `llama.cpp-experimental` 13 GiB, its preserved
# copy 15 GiB, 13 worktrees totalling 41 GiB.
DEFAULT_LARGEST_SINGLE_ALLOCATION_GB = 15.0

# A profiler trace younger than this is still plausibly informing a live
# lineage; §5.8 only makes traces "older than the lineage they informed"
# expirable, and age alone never suffices (the rule also requires the lineage to
# be closed).
DEFAULT_MIN_PROFILER_TRACE_AGE_DAYS = 30

#: Bound, not re-compiled — the digest shape has one owner. See `schemas.require`.
_SHA256_RE = schemas.SHA256_RE
# A campaign directory name. Deliberately narrow: this string becomes a path
# under the repo, so `..`, absolute paths, and separators are excluded by the
# pattern rather than by a later check that could be reordered away.
_CAMPAIGN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# =============================================================================
# Vocabularies (§5.8 retention table)
# =============================================================================

RETENTION_CLASSES = frozenset({
    "permanent_in_repo",   # events, reduced metrics, patches, hashes, manifests, README/SHA256SUMS
    "permanent_large",     # champion binaries; incumbent production binaries N-1 and N-2
    "expirable",           # the ONLY class expire_artifact() will act on
    "never_stored",        # candidate outputs used as a correctness oracle
})

# §5.8 names exactly three expirable things. Adding a fourth is a ratification
# event (the retention rule is ratified once in AK0), not a code change.
EXPIRABLE_KINDS = frozenset({
    "rejected_candidate_build_tree",
    "retired_campaign_worktree",
    "stale_profiler_trace",
})

# Candidate statuses whose build tree may be reclaimed. `banked` and any
# champion-bearing candidate are absent on purpose: those are the trees a
# rollback or a re-score needs.
_RECLAIMABLE_CANDIDATE_STATUSES = frozenset({
    "rejected", "build_failed", "superseded", "invalid",
})
_RETIRED_CAMPAIGN_STATUSES = frozenset({"closed", "superseded", "aborted"})

# Storage / quota states. DISK_PRESSURE and quota exhaustion are DISTINCT stop
# states (§8.1, §8.11): the first means the host is nearly out of space, the
# second means this campaign spent its budget while the host is fine. Conflating
# them would either halt the loop for a budget it could raise, or let it keep
# allocating on a host that cannot afford it.
STORAGE_OK = "STORAGE_OK"
DISK_PRESSURE = "DISK_PRESSURE"
QUOTA_OK = "QUOTA_OK"
QUOTA_WARN = "QUOTA_WARN"
QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"

# The tombstone record. Not registered into `schemas.SCHEMA_REGISTRY`: a module
# that mutates another module's global table at import time makes validation
# depend on import order. The journal wires this validator explicitly, and a
# later `schemas` version should absorb the record.
SCHEMA_ARTIFACT_TOMBSTONE = "epyc.autokernel.artifact_tombstone.v1"
TOMBSTONE_ID_PREFIX = "akt-"
RECLAMATION_STATES = frozenset({"intent", "reclaimed", "failed"})

# Marker left in a generated README so an unfilled stub cannot pass for a
# compliant one. `MEASUREMENT.md:146-156` requires the README to state what was
# measured, when, and which claim it backs; a directory that merely HAS a README
# satisfies none of that.
README_STUB_MARKER = "<!-- AUTOKERNEL-README-STUB: unfilled -->"
# The literal the stub writes into each unanswered row. Checked SEPARATELY from
# the marker: a marker is one line a caller can delete, and "delete the thing the
# check inspects" must not be a way to pass. The placeholder text sits in the
# answer cells themselves, so removing it means answering the question.
README_PLACEHOLDER = "TODO — fill in"
SHA256SUMS_NAME = "SHA256SUMS"
README_NAME = "README.md"

# One `sha256sum` output line. A manifest that exists but carries no hash line is
# as uncheckable as an empty one; only the *shape* is asserted here, because
# verifying the hashes is `sha256sum -c`'s job, not this checker's.
_SHA256SUMS_LINE_RE = re.compile(r"^[0-9a-f]{64}\s[\s*]?\S")


# =============================================================================
# Errors — every one of these is a refusal, never a degraded result
# =============================================================================

class StorageError(Exception):
    """Base for every refusal in the storage plane."""


class ScratchCitationError(StorageError, ValueError):
    """A citation of record resolved to a scratch root (`MEASUREMENT.md:146-156`)."""


class UnclassifiablePath(StorageError):
    """A durability class could not be determined, so none is returned.

    Raised rather than defaulting, because every plausible default is a lie: a
    missing artifact reported as `hash_and_provenance_only` turns a loss into an
    expected absence, and an unknown git state reported as `durable_untracked`
    turns a tracked file into an untracked one.
    """


class EvidenceRootError(StorageError):
    """The mandated `data/<campaign>/` layout could not be created or is unusable."""


class ExpiryRefused(StorageError):
    """A reclamation was refused. Deletion authority is narrow and rule-bound."""


# =============================================================================
# Path facts
# =============================================================================

def _norm(path: str | os.PathLike) -> str:
    """Absolute, symlink-free-at-the-parents, `..`-free string form.

    `os.path.realpath` is deliberate: a scratch path reachable through a symlink
    is still a scratch path, and the whole `EPHEMERAL_ROOTS` guard is a prefix
    test that `..` or a symlink would otherwise walk straight past.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be a string or PathLike, got {type(path).__name__}")
    text = os.fspath(path)
    if not text:
        raise ValueError("path must not be empty")
    return os.path.realpath(os.path.abspath(text))


def _under(path: str, root: str) -> bool:
    """True when `path` is `root` itself or strictly beneath it."""
    root = root.rstrip("/") or "/"
    return path == root or path.startswith(root + "/")


def _root_match_forms(roots: Sequence[str]) -> tuple:
    """Each declared root plus its realpath, when they differ.

    Every path this module tests has been through `_norm`, i.e. `realpath`. The
    guard roots had NOT: they were compared as literal strings. Today none of
    them is a symlink on this host (verified 2026-08-03), so the two forms
    coincide — but this repository's own working-tree identity rule makes
    `/workspace/repos/<name>` a symlink to `/mnt/raid0/llm/<name>` (CLAUDE.md),
    so the day a guard root acquires a link the prefix test would stop matching
    and fail OPEN, silently. Matching both forms can only ever add matches.
    """
    forms: list[str] = []
    for root in roots:
        if root not in forms:
            forms.append(root)
        try:
            real = os.path.realpath(root)
        except OSError:  # pragma: no cover — realpath does not raise on Linux
            continue
        if real not in forms:
            forms.append(real)
    return tuple(forms)


_EPHEMERAL_ROOT_FORMS = _root_match_forms(EPHEMERAL_ROOTS)


def production_tree_forms() -> tuple:
    """Guard forms for the frozen trees, derived at USE, not at import.

    Deliberately not a module-level constant like the scratch forms: those sit on
    a hot path (one call per citation), while this one is consulted once per
    reclamation, and freezing it at import would mean the invariant-3 guard
    could never be exercised against a symlinked production root — the exact
    configuration this repository's working-tree identity rule creates.
    """
    return _root_match_forms(PRODUCTION_TREES)


def is_scratch_path(path: str | os.PathLike) -> bool:
    """True when the path resolves under a scratch root, symlinks included."""
    resolved = _norm(path)
    return any(_under(resolved, root) for root in _EPHEMERAL_ROOT_FORMS)


def assert_not_scratch(path: str | os.PathLike, *, what: str = "citation") -> str:
    """Return the resolved path, or raise if it is a scratch path.

    An ERROR, not a warning: a scratch citation is one `tmp` sweep away from
    being unverifiable, and the sweep leaves no event behind.
    """
    resolved = _norm(path)
    for root in _EPHEMERAL_ROOT_FORMS:
        if _under(resolved, root):
            raise ScratchCitationError(
                f"{what} resolves to the scratch root {root!r}: {resolved!r}. "
                f"Evidence of record must live under "
                f"{EVIDENCE_DIRNAME}/<campaign>/ in the repository "
                f"(MEASUREMENT.md:146-156)."
            )
    return resolved


# =============================================================================
# git tracked-ness — "untracked looks identical to committed" on the filesystem
# =============================================================================

class TrackedIndex:
    """Answers 'is this path carried in git?'. Implementations must never guess.

    Subclasses return True/False. A subclass that cannot answer raises; it does
    NOT return False, because a tracked file misreported as untracked is exactly
    how a durability verifier would bless a file git does not carry.
    """

    def is_tracked(self, path: str | os.PathLike) -> bool:
        raise NotImplementedError

    def contains_repo(self, path: str | os.PathLike) -> bool:
        """True when `path` lies inside this index's working tree."""
        raise NotImplementedError


class StaticTrackedIndex(TrackedIndex):
    """A tracked-path set supplied by the caller. The test double, and the shape
    a journal-side snapshot of the index would take."""

    def __init__(self, repo_root: str | os.PathLike, tracked: Iterable[str]):
        self.repo_root = _norm(repo_root)
        # Stored repo-relative, exactly as `git ls-files` emits them.
        self._tracked = frozenset(str(p).lstrip("/") for p in tracked)

    def contains_repo(self, path: str | os.PathLike) -> bool:
        return _under(_norm(path), self.repo_root)

    def is_tracked(self, path: str | os.PathLike) -> bool:
        resolved = _norm(path)
        if not self.contains_repo(resolved):
            raise UnclassifiablePath(
                f"{resolved!r} is outside the working tree {self.repo_root!r}; "
                "this index cannot answer for it"
            )
        rel = os.path.relpath(resolved, self.repo_root)
        if rel == ".":
            # The working-tree root itself. `relpath` gives "." and `git ls-files`
            # never emits a leading "./", so the prefix test below could not match
            # and a repository tracking thousands of files reported its own root
            # untracked — which `classify()` then turned into durable_untracked
            # (or hash_and_provenance_only) for the whole tree.
            return bool(self._tracked)
        if rel in self._tracked:
            return True
        # A directory counts as carried when git carries anything beneath it;
        # git has no directory entries, so the prefix test IS the question.
        prefix = rel.rstrip("/") + "/"
        return any(t.startswith(prefix) for t in self._tracked)


class GitTrackedIndex(TrackedIndex):
    """Tracked-path set read once from `git ls-files -z`.

    One read-only query per repository, cached. It RAISES when git is missing,
    the directory is not a repository, or the command fails — a durability
    verdict computed from a failed git query would report every tracked file as
    untracked, which is the single worst wrong answer this module can give.
    """

    def __init__(self, repo_root: str | os.PathLike, *, timeout: float = 120.0):
        self.repo_root = _norm(repo_root)
        if not os.path.isdir(self.repo_root):
            raise UnclassifiablePath(f"not a directory: {self.repo_root!r}")
        try:
            completed = subprocess.run(
                ["git", "-C", self.repo_root, "ls-files", "-z"],
                capture_output=True, check=False, timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise UnclassifiablePath(
                f"git is not available, so tracked-ness cannot be determined "
                f"for {self.repo_root!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise UnclassifiablePath(
                f"`git ls-files` timed out for {self.repo_root!r}"
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", "replace").strip()
            raise UnclassifiablePath(
                f"`git ls-files` failed in {self.repo_root!r} "
                f"(exit {completed.returncode}): {stderr}"
            )
        names = completed.stdout.decode("utf-8", "surrogateescape").split("\0")
        self._delegate = StaticTrackedIndex(self.repo_root, (n for n in names if n))

    def contains_repo(self, path: str | os.PathLike) -> bool:
        return self._delegate.contains_repo(path)

    def is_tracked(self, path: str | os.PathLike) -> bool:
        return self._delegate.is_tracked(path)


# =============================================================================
# classify() — durability class of an artifact that actually exists
# =============================================================================

@dataclass(frozen=True)
class Classification:
    """The recorded class plus the facts it was derived from.

    The facts ride along on purpose: a class with no derivation is an assertion,
    and the next verifier has to redo the work to disagree with it.
    """

    durability_class: str
    resolved_path: str
    in_repo: bool
    tracked: Optional[bool]
    size_bytes: int
    carry_threshold_bytes: int


def classify(
    path: str | os.PathLike,
    *,
    tracked_index: Optional[TrackedIndex] = None,
    carry_threshold_bytes: int = DEFAULT_CARRY_THRESHOLD_BYTES,
) -> Classification:
    """Classify an existing artifact into one of `schemas.DURABILITY_CLASSES`.

    Rules, in order:
      1. a scratch path is not a class at all — `ScratchCitationError`;
      2. a path that does not exist raises `UnclassifiablePath`. Absence is NOT
         `hash_and_provenance_only`: that class means "too large to carry, so we
         deliberately kept only the hash", and letting absence produce it would
         relabel every loss as an intended design decision (§3.7 exists to keep
         those two apart);
      3. inside a working tree and tracked -> `carried_in_git`;
      4. inside a working tree, untracked, at or under the carry threshold ->
         `durable_untracked`;
      5. anything else — untracked and oversized, or outside every working tree —
         -> `hash_and_provenance_only`, because nothing versions those bytes.

    `tracked_index` is required for any in-repo path; without it steps 3 and 4
    cannot be told apart and the call raises rather than picking one.
    """
    resolved = assert_not_scratch(path, what="artifact")
    if tracked_index is None:
        # "No index" is indistinguishable from "you forgot the index", and the
        # difference decides between carried_in_git and durable_untracked.
        raise UnclassifiablePath(
            f"no tracked_index supplied, so git tracked-ness of {resolved!r} is "
            "unknown; pass GitTrackedIndex(repo_root) or a StaticTrackedIndex"
        )
    if not os.path.lexists(resolved):
        raise UnclassifiablePath(
            f"{resolved!r} does not exist; absence is not a durability class. "
            "Record hash_and_provenance_only deliberately, with the hash and the "
            "provenance, or treat this as the loss it may be."
        )
    size = measure_usage(resolved).bytes_on_disk

    in_repo = tracked_index.contains_repo(resolved)
    if in_repo:
        tracked = tracked_index.is_tracked(resolved)
        if tracked:
            klass = "carried_in_git"
        elif size <= carry_threshold_bytes:
            klass = "durable_untracked"
        else:
            klass = "hash_and_provenance_only"
    else:
        tracked = False
        klass = "hash_and_provenance_only"

    # The vocabulary is owned by schemas.py; a class this module invented would
    # be rejected downstream by the record validators, so catch it here.
    if klass not in schemas.DURABILITY_CLASSES:
        raise UnclassifiablePath(
            f"{klass!r} is not one of {sorted(schemas.DURABILITY_CLASSES)}"
        )
    return Classification(
        durability_class=klass,
        resolved_path=resolved,
        in_repo=in_repo,
        tracked=tracked,
        size_bytes=size,
        carry_threshold_bytes=carry_threshold_bytes,
    )


# =============================================================================
# Campaign evidence root — created, never assumed
# =============================================================================

@dataclass(frozen=True)
class EvidenceRoot:
    path: str
    campaign_id: str
    created: bool
    readme_path: str
    sha256sums_path: str
    layout: Check


def campaign_evidence_root(campaign_id: str, *, repo_root: str | os.PathLike = REPO_ROOT) -> str:
    """Resolve `data/<campaign_id>/` without creating anything.

    Raises on a campaign id that is not a safe single path segment, so a caller
    cannot walk out of the evidence tree through the id.
    """
    if not isinstance(campaign_id, str) or not _CAMPAIGN_NAME_RE.match(campaign_id):
        raise EvidenceRootError(
            f"campaign_id {campaign_id!r} is not a safe directory name "
            f"(must match {_CAMPAIGN_NAME_RE.pattern})"
        )
    root = _norm(repo_root)
    candidate = os.path.join(root, EVIDENCE_DIRNAME, campaign_id)
    resolved = os.path.realpath(candidate)
    if not _under(resolved, os.path.join(root, EVIDENCE_DIRNAME)):
        raise EvidenceRootError(
            f"campaign evidence root {resolved!r} escapes "
            f"{os.path.join(root, EVIDENCE_DIRNAME)!r}"
        )
    return resolved


def ensure_campaign_evidence_root(
    campaign_id: str,
    *,
    repo_root: str | os.PathLike = REPO_ROOT,
    claim: str = "",
    what_was_measured: str = "",
    measured_at: str = "",
) -> EvidenceRoot:
    """Create `data/<campaign>/` with a `SHA256SUMS` and a README stub.

    Idempotent, and non-destructive: an existing README or SHA256SUMS is left
    exactly as it is. The README stub carries `README_STUB_MARKER` so that
    `check_evidence_root_layout()` still FAILS until a human fills in what was
    measured, when, and which claim it backs — creating the layout satisfies the
    SHAPE the constitution requires, never its CONTENT, and a checker that
    accepted the stub would certify an empty promise.
    """
    root = campaign_evidence_root(campaign_id, repo_root=repo_root)
    existed = os.path.isdir(root)
    if os.path.lexists(root) and not existed:
        raise EvidenceRootError(f"{root!r} exists and is not a directory")
    os.makedirs(root, exist_ok=True)

    sums_path = os.path.join(root, SHA256SUMS_NAME)
    if not os.path.lexists(sums_path):
        # Created empty: a campaign with no artifacts yet has an honest empty
        # manifest. `check_evidence_root_layout` fails it only once artifacts
        # appear beside it.
        #
        # `"x"` is the right primitive — it is the racing process, not the
        # lexists() above, that decides who wins — but the FileExistsError it
        # raises was uncaught, so two sessions opening the same campaign made
        # this function, documented as idempotent, raise a bare OSError. Losing
        # the race IS the idempotent outcome: the file the caller wanted exists.
        try:
            with open(sums_path, "x", encoding="utf-8"):
                pass
        except FileExistsError:
            pass

    readme_path = os.path.join(root, README_NAME)
    if not os.path.lexists(readme_path):
        try:
            with open(readme_path, "x", encoding="utf-8") as fh:
                fh.write(_readme_stub(campaign_id, claim, what_was_measured,
                                      measured_at))
        except FileExistsError:
            pass

    return EvidenceRoot(
        path=root,
        campaign_id=campaign_id,
        created=not existed,
        readme_path=readme_path,
        sha256sums_path=sums_path,
        layout=check_evidence_root_layout(root),
    )


def _readme_stub(campaign_id: str, claim: str, what: str, when: str) -> str:
    """The three questions `MEASUREMENT.md:146-156` requires a README to answer."""
    unfilled = [] if (claim and what and when) else [README_STUB_MARKER]
    body = [
        f"# {campaign_id} — AutoKernel campaign evidence",
        "",
        "Evidence root mandated by `MEASUREMENT.md:146-156`: the artifacts behind any",
        "ratified or production-affecting claim live here, in-repo, with `SHA256SUMS`",
        "and this README. A scratch path is never the citation of record.",
        "",
        "| | |",
        "|---|---|",
        f"| what was measured | {what or 'TODO — fill in'} |",
        f"| when | {when or 'TODO — fill in (UTC)'} |",
        f"| which claim it backs | {claim or 'TODO — fill in'} |",
        "",
        "## Durability classes (§3.7)",
        "",
        "Every artifact cited from this root carries one of `carried_in_git`,",
        "`durable_untracked`, `hash_and_provenance_only`, so a verifier can tell a",
        "defect from an expected absence. Oversized artifacts are recorded",
        "hash-and-provenance-only and the citation says so.",
        "",
        "## Verifying",
        "",
        "```",
        f"cd {EVIDENCE_DIRNAME}/{campaign_id} && sha256sum -c {SHA256SUMS_NAME}",
        "```",
        "",
    ]
    return "\n".join(unfilled + body)


def check_evidence_root_layout(root: str | os.PathLike) -> Check:
    """PASS / FAIL / COULD_NOT_CHECK on the mandated `data/<campaign>/` layout."""
    try:
        resolved = _norm(root)
    except (TypeError, ValueError) as exc:
        return Check(COULD_NOT_CHECK, (f"evidence root is unusable: {exc}",))
    if not os.path.isdir(resolved):
        return Check(FAIL, (f"evidence root {resolved!r} does not exist as a directory",))

    reasons: list[str] = []
    try:
        with os.scandir(resolved) as entries:
            names = sorted(e.name for e in entries)
    except OSError as exc:
        return Check(COULD_NOT_CHECK, (f"cannot list {resolved!r}: {exc}",))

    readme = os.path.join(resolved, README_NAME)
    sums = os.path.join(resolved, SHA256SUMS_NAME)
    if README_NAME not in names:
        reasons.append(f"{README_NAME} is missing (MEASUREMENT.md:146-156)")
    else:
        try:
            with open(readme, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            return Check(COULD_NOT_CHECK, (f"cannot read {readme!r}: {exc}",))
        # Three independent conditions, because the marker alone was passable by
        # DELETING it: strip one HTML comment (or truncate the file, or point it
        # at /dev/null) and a README whose three answer cells still read
        # "TODO — fill in" scored PASS. A check you can satisfy by removing what
        # it inspects is not a check.
        if not text.strip():
            reasons.append(
                f"{README_NAME} is empty: it states neither what was measured, "
                "when, nor which claim it backs (MEASUREMENT.md:146-156)"
            )
        elif README_STUB_MARKER in text:
            reasons.append(
                f"{README_NAME} is still the unfilled stub: it does not yet state "
                "what was measured, when, and which claim it backs"
            )
        elif README_PLACEHOLDER in text:
            reasons.append(
                f"{README_NAME} still carries the {README_PLACEHOLDER!r} "
                "placeholder: at least one of what/when/which-claim is unanswered "
                "(MEASUREMENT.md:146-156)"
            )
    if SHA256SUMS_NAME not in names:
        reasons.append(f"{SHA256SUMS_NAME} is missing (MEASUREMENT.md:146-156)")
    else:
        try:
            sums_size = os.path.getsize(sums)
            with open(sums, "r", encoding="utf-8", errors="replace") as fh:
                sums_text = fh.read()
        except OSError as exc:
            return Check(COULD_NOT_CHECK, (f"cannot stat {sums!r}: {exc}",))
        artifacts = [n for n in names if n not in (README_NAME, SHA256SUMS_NAME)]
        if artifacts and sums_size == 0:
            reasons.append(
                f"{SHA256SUMS_NAME} is empty while {len(artifacts)} artifact(s) sit "
                "beside it — the hashes that make them checkable were never written"
            )
        elif artifacts and not any(
            _SHA256SUMS_LINE_RE.match(line) for line in sums_text.splitlines()
        ):
            # Size alone was the whole test, so a single space or a comment made
            # a manifest "non-empty" and the root compliant. Non-empty is not
            # checkable; a `sha256sum`-shaped line is the minimum that is.
            reasons.append(
                f"{SHA256SUMS_NAME} is non-empty but contains no "
                "'<sha256>  <name>' line, so nothing beside it is checkable with "
                f"`sha256sum -c` ({len(artifacts)} artifact(s) present)"
            )
    return Check(FAIL, tuple(reasons)) if reasons else Check(PASS)


# =============================================================================
# Usage accounting, quota, and DISK_PRESSURE
# =============================================================================

@dataclass(frozen=True)
class Usage:
    """Measured occupancy of a tree.

    `bytes_on_disk` (512-byte blocks x st_blocks) is what a quota must count —
    it is what the filesystem actually spent. `bytes_apparent` is kept beside it
    because sparse files make the two disagree by orders of magnitude and a
    single number would hide which one you are looking at.
    """

    root: str
    bytes_on_disk: int
    bytes_apparent: int
    file_count: int
    dir_count: int
    hardlink_duplicates: int


def measure_usage(root: str | os.PathLike) -> Usage:
    """Walk `root` and total its occupancy. Each inode counted once.

    Raises `OSError` on any unreadable entry instead of skipping it. A quota
    computed from a partial walk silently under-reports, and under-reporting is
    how a campaign blows a budget while every component reports healthy — the
    fail-open pattern this project has been bitten by repeatedly.
    """
    resolved = _norm(root)
    st = os.lstat(resolved)
    if not os.path.isdir(resolved) or os.path.islink(resolved):
        return Usage(
            root=resolved,
            bytes_on_disk=st.st_blocks * 512,
            bytes_apparent=st.st_size,
            file_count=1,
            dir_count=0,
            hardlink_duplicates=0,
        )

    on_disk = apparent = files = dirs = dupes = 0
    seen: set[tuple[int, int]] = set()
    stack = [resolved]
    while stack:
        current = stack.pop()
        dirs += 1
        # `os.scandir` holds a directory file descriptor; without the context
        # manager it leaks one per directory and trips ResourceWarning.
        with os.scandir(current) as entries:
            for entry in entries:
                entry_stat = entry.stat(follow_symlinks=False)
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                    on_disk += entry_stat.st_blocks * 512
                    continue
                files += 1
                key = (entry_stat.st_dev, entry_stat.st_ino)
                if entry_stat.st_nlink > 1:
                    if key in seen:
                        dupes += 1
                        continue
                    seen.add(key)
                on_disk += entry_stat.st_blocks * 512
                apparent += entry_stat.st_size
    on_disk += st.st_blocks * 512
    return Usage(
        root=resolved,
        bytes_on_disk=on_disk,
        bytes_apparent=apparent,
        file_count=files,
        dir_count=dirs,
        hardlink_duplicates=dupes,
    )


@dataclass(frozen=True)
class StoragePolicy:
    """Campaign storage budget plus host headroom floor.

    `campaign_quota_gb` comes from the campaign manifest's
    `budgets.max_storage_gb` (§7.1); nothing here invents a budget.
    """

    campaign_quota_gb: float
    headroom_floor_gb: float = DEFAULT_HEADROOM_FLOOR_GB
    largest_single_allocation_gb: float = DEFAULT_LARGEST_SINGLE_ALLOCATION_GB
    allocation_safety_factor: float = 2.0
    quota_warn_fraction: float = 0.8
    carry_threshold_bytes: int = DEFAULT_CARRY_THRESHOLD_BYTES
    min_profiler_trace_age_days: int = DEFAULT_MIN_PROFILER_TRACE_AGE_DAYS
    # Roots the campaign owns and may reclaim within. EMPTY BY DEFAULT: a caller
    # that declares nothing can delete nothing (§5.8, MEASUREMENT.md:223-229).
    owned_roots: tuple = ()

    def __post_init__(self) -> None:
        # EVERY numeric field goes through the finite screen, not just the three
        # that happened to be listed. `allocation_safety_factor` was outside it,
        # and NaN passes `< 1` (all NaN comparisons are False) while
        # `max(floor, 15.0 * nan)` returns the declared floor — so a NaN safety
        # factor silently switched OFF the allocation-step floor with no error
        # anywhere. `inf` was worse: it survived the constructor and blew up as
        # an OverflowError inside `effective_floor_bytes`, far from the caller
        # that supplied it.
        for name in ("campaign_quota_gb", "headroom_floor_gb",
                     "largest_single_allocation_gb", "allocation_safety_factor",
                     "quota_warn_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
            if value != value:
                raise ValueError(
                    f"{name} must be a finite number, got NaN — every NaN "
                    "comparison is False, so a NaN here disables the bound it is "
                    "supposed to impose instead of failing"
                )
            if value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
        if self.allocation_safety_factor < 1:
            raise ValueError(
                "allocation_safety_factor must be >= 1: a floor smaller than one "
                "allocation step lets a single build cross from healthy to ENOSPC"
            )
        if not 0 < self.quota_warn_fraction <= 1:
            raise ValueError("quota_warn_fraction must be in (0, 1]")
        if isinstance(self.min_profiler_trace_age_days, bool) or \
                not isinstance(self.min_profiler_trace_age_days, int):
            raise TypeError(
                "min_profiler_trace_age_days must be an int, got "
                f"{type(self.min_profiler_trace_age_days).__name__} "
                "(True compares as 1 and would silently become a 1-day minimum)"
            )
        if self.min_profiler_trace_age_days < 0:
            raise ValueError("min_profiler_trace_age_days must be >= 0")
        object.__setattr__(self, "owned_roots", tuple(_norm(r) for r in self.owned_roots))

    @property
    def campaign_quota_bytes(self) -> int:
        return int(self.campaign_quota_gb * _GIB)

    @property
    def effective_floor_bytes(self) -> int:
        """The floor actually enforced.

        `max(declared floor, largest single allocation x safety factor)` — a
        declared floor of 5 GiB is meaningless on a host whose next legal action
        writes a 15 GiB worktree.
        """
        stepped = self.largest_single_allocation_gb * self.allocation_safety_factor
        return int(max(self.headroom_floor_gb, stepped) * _GIB)


@dataclass(frozen=True)
class StorageState:
    state: str
    free_bytes: int
    total_bytes: int
    floor_bytes: int
    reasons: tuple = ()

    @property
    def pressured(self) -> bool:
        return self.state == DISK_PRESSURE


def disk_pressure(path: str | os.PathLike, policy: StoragePolicy) -> StorageState:
    """`DISK_PRESSURE` when free space on `path`'s filesystem is below the floor.

    Raises `OSError` when the filesystem cannot be interrogated. There is no
    "assume healthy" branch: a controller that cannot read free space has no
    business allocating, and the §8.1 stop state exists precisely so it stops.
    """
    resolved = _norm(path)
    stat = os.statvfs(resolved)
    # f_bavail, not f_bfree: the root-reserved blocks are not ours to spend.
    free = stat.f_bavail * stat.f_frsize
    total = stat.f_blocks * stat.f_frsize
    floor = policy.effective_floor_bytes
    if free < floor:
        return StorageState(
            state=DISK_PRESSURE,
            free_bytes=free,
            total_bytes=total,
            floor_bytes=floor,
            reasons=(
                f"free {free / _GIB:.1f} GiB is below the {floor / _GIB:.1f} GiB "
                f"headroom floor on {resolved!r}",
            ),
        )
    return StorageState(state=STORAGE_OK, free_bytes=free, total_bytes=total,
                        floor_bytes=floor)


@dataclass(frozen=True)
class QuotaState:
    state: str
    used_bytes: int
    limit_bytes: int
    fraction: float
    reasons: tuple = ()

    @property
    def exhausted(self) -> bool:
        return self.state == QUOTA_EXHAUSTED


def campaign_quota_state(usage: Usage, policy: StoragePolicy) -> QuotaState:
    """Account this campaign's occupancy against its declared storage budget.

    Exhaustion here is a `BUDGET_STOP` input, NOT `DISK_PRESSURE` (§8.11): the
    campaign overspent, which says nothing about the host. Keeping the two apart
    is what lets an operator raise a budget without pretending the disk grew.
    """
    if not isinstance(usage, Usage):
        raise TypeError(f"usage must be a Usage, got {type(usage).__name__}")
    limit = policy.campaign_quota_bytes
    used = usage.bytes_on_disk
    fraction = (used / limit) if limit > 0 else float("inf") if used > 0 else 0.0
    if limit == 0:
        if used > 0:
            return QuotaState(QUOTA_EXHAUSTED, used, limit, fraction,
                              (f"campaign quota is 0 GiB but {used} bytes are "
                               f"occupied under {usage.root!r}",))
        return QuotaState(QUOTA_OK, used, limit, 0.0)
    if used >= limit:
        return QuotaState(QUOTA_EXHAUSTED, used, limit, fraction,
                          (f"campaign used {used / _GIB:.2f} GiB of its "
                           f"{limit / _GIB:.2f} GiB budget",))
    if fraction >= policy.quota_warn_fraction:
        return QuotaState(QUOTA_WARN, used, limit, fraction,
                          (f"campaign used {fraction * 100:.0f}% of its "
                           f"{limit / _GIB:.2f} GiB budget",))
    return QuotaState(QUOTA_OK, used, limit, fraction)


# =============================================================================
# Hashing helpers — a tombstone without a hash proves nothing
# =============================================================================

def hash_file(path: str | os.PathLike, *, chunk_bytes: int = 1024 * 1024) -> str:
    """SHA-256 of one file's bytes."""
    digest = hashlib.sha256()
    with open(_norm(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree_manifest(root: str | os.PathLike) -> str:
    """Content hash of a whole tree, via a canonical `{relpath: sha256}` manifest.

    A directory has no bytes of its own, so the identity of a build tree is the
    identity of its manifest. Reuses `schemas.content_hash`, which refuses the
    encodings that would silently produce a different hash for the same tree.
    Symlinks are recorded by target rather than followed — following them would
    hash bytes that live somewhere else and are not being reclaimed.
    """
    resolved = _norm(root)
    manifest: dict[str, str] = {}
    stack = [resolved]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                rel = os.path.relpath(entry.path, resolved)
                if entry.is_symlink():
                    manifest[rel] = "symlink:" + os.readlink(entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    # Recorded, not merely descended into. A manifest of files
                    # alone is blind to directory structure: a tree carrying an
                    # extra EMPTY directory hashed byte-identically to one
                    # without it, so two materially different build trees shared
                    # one identity — and this hash IS the identity a tombstone
                    # records for a tree that is about to stop existing.
                    manifest[rel] = "dir:"
                    stack.append(entry.path)
                else:
                    manifest[rel] = hash_file(entry.path)
    return schemas.content_hash({"tree_manifest_v1": manifest})


# =============================================================================
# Tombstone record
# =============================================================================

def validate_artifact_tombstone(obj: Any) -> list:
    """Validate a tombstone. Returns violations; empty means valid; never raises.

    Matches `schemas.validate_*`'s contract on purpose — a rejected tombstone is
    itself something the journal must be able to record.
    """
    out: list = []
    if not isinstance(obj, Mapping):
        return [f"record: expected a mapping, got {type(obj).__name__}"]
    if obj.get("schema") != SCHEMA_ARTIFACT_TOMBSTONE:
        out.append(f"schema: expected {SCHEMA_ARTIFACT_TOMBSTONE!r}, "
                   f"got {obj.get('schema')!r}")

    # A tombstone is a machine record; §1.3 gives AutoKernel no authority to
    # declare, and a reclamation event is a tempting place to smuggle one in.
    for path in schemas.find_authority_flavoured_keys(obj):
        out.append(f"{path}: authority-flavoured key is forbidden in a tombstone (§1.3)")

    def need_str(name, *, choices=None, pattern=None, hint=""):
        value = obj.get(name)
        if name not in obj:
            out.append(f"{name}: required field is missing")
            return None
        if not isinstance(value, str) or not value.strip():
            out.append(f"{name}: expected a non-empty string, got {value!r}")
            return None
        if choices is not None and value not in choices:
            out.append(f"{name}: {value!r} is not one of {sorted(choices)}")
            return None
        if pattern is not None and not pattern.match(value):
            out.append(f"{name}: {value!r} {hint}")
            return None
        return value

    need_str("tombstone_id")
    if isinstance(obj.get("tombstone_id"), str) and \
            not obj["tombstone_id"].startswith(TOMBSTONE_ID_PREFIX):
        out.append(f"tombstone_id: must start with {TOMBSTONE_ID_PREFIX!r}")
    need_str("campaign_id")
    artifact_path = need_str("artifact_path")
    if artifact_path is not None and not artifact_path.startswith("/"):
        out.append("artifact_path: must be absolute")
    need_str("artifact_sha256", pattern=_SHA256_RE,
             hint="is not a lowercase hex sha256")
    klass = need_str("durability_class", choices=schemas.DURABILITY_CLASSES)
    if klass == "carried_in_git":
        out.append("durability_class: a carried_in_git artifact is not reclaimable — "
                   "git already holds those bytes and deleting the checkout "
                   "reclaims nothing")
    need_str("retention_class", choices={"expirable"})
    need_str("expirable_kind", choices=EXPIRABLE_KINDS)
    need_str("rule_id")
    need_str("reason")
    need_str("actor")
    need_str("reclamation_state", choices=RECLAMATION_STATES)

    for name in ("size_bytes", "file_count"):
        value = obj.get(name)
        if name not in obj:
            out.append(f"{name}: required field is missing")
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            out.append(f"{name}: expected a non-negative integer, got {value!r}")

    when = obj.get("reclaimed_at")
    if "reclaimed_at" not in obj:
        out.append("reclaimed_at: required field is missing")
    elif not isinstance(when, str):
        out.append(f"reclaimed_at: expected a string, got {type(when).__name__}")
    else:
        try:
            parsed = datetime.fromisoformat(when)
        except ValueError:
            out.append(f"reclaimed_at: {when!r} is not an ISO-8601 timestamp")
        else:
            if parsed.tzinfo is None:
                out.append(f"reclaimed_at: {when!r} has no timezone offset")

    if not isinstance(obj.get("preconditions"), Mapping):
        out.append("preconditions: expected a mapping of the facts that satisfied "
                   "the retention rule")
    if obj.get("reclamation_state") == "failed" and not obj.get("error"):
        out.append("error: required and non-empty when reclamation_state == 'failed'")
    return out


def tombstone_id(campaign_id: str, artifact_path: str, artifact_sha256: str,
                 expirable_kind: str, rule_id: str) -> str:
    """Content-addressed id over the identity of the reclamation.

    Deliberately excludes the timestamp so the `intent` and `reclaimed` records
    of one reclamation share an id, and so a retried expiry of the same artifact
    is recognisable as the same event rather than a second, unexplained one.
    """
    body = {
        "campaign_id": campaign_id,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "expirable_kind": expirable_kind,
        "rule_id": rule_id,
    }
    return TOMBSTONE_ID_PREFIX + schemas.content_hash(body)[:32]


# =============================================================================
# Expiry — rule-bound, dry-run by default, tombstone before bytes
# =============================================================================

@dataclass(frozen=True)
class ExpirableArtifact:
    """A reclamation request. Every field is stated by the caller, never inferred.

    `retention_class` defaults to nothing useful on purpose: the caller must say
    `expirable` out loud, because the §5.8 table has four classes and three of
    them are never reclaimable.
    """

    path: str
    campaign_id: str
    sha256: str
    durability_class: str
    expirable_kind: str
    reason: str
    rule_id: str
    actor: str
    retention_class: str = "permanent_in_repo"
    preconditions: Mapping[str, Any] = field(default_factory=dict)
    declared_size_bytes: Optional[int] = None


@dataclass(frozen=True)
class ExpiryRule:
    """The facts a kind's reclamation requires, and what they must say."""

    kind: str
    required_facts: tuple
    describe: str


EXPIRY_RULES = {
    "rejected_candidate_build_tree": ExpiryRule(
        kind="rejected_candidate_build_tree",
        required_facts=("candidate_id", "candidate_status", "champion_status",
                        "evaluation_events_journaled"),
        describe="a build tree may go only after its candidate is out of the running "
                 "and every outcome it produced is already durable (invariant 7)",
    ),
    "retired_campaign_worktree": ExpiryRule(
        kind="retired_campaign_worktree",
        required_facts=("campaign_status", "champion_artifacts_preserved",
                        "evaluation_events_journaled"),
        describe="a campaign worktree may go only after the campaign is retired and "
                 "its champion artifacts are preserved elsewhere",
    ),
    "stale_profiler_trace": ExpiryRule(
        kind="stale_profiler_trace",
        required_facts=("informed_lineage_id", "lineage_closed", "trace_age_days"),
        describe="a trace may go only when it is older than the lineage it informed "
                 "AND that lineage is closed (§5.8); age alone never suffices",
    ),
}


class JournalTombstoneSink:
    """Adapts the AK1 event journal to the sink contract `expire_artifact` needs.

    Two different shapes meet here and neither should be bent to the other:

      * this module wants `append(record) -> event_id`, so the expiry path can be
        tested without a journal on disk and so the sink can be anything durable;
      * `journal.Journal` exposes `append(kind, payload, *, campaign_id=...) ->
        JournalEntry`, owns the closed event-kind vocabulary, and validates a
        native TOMBSTONE payload of `{artifact_sha256, storage_class, size_bytes,
        reason}` — refusing any storage class but `expirable`.

    The translation is one renamed key (`retention_class` -> the journal's
    `storage_class`) plus a `path` alias. Duck-typed on purpose: importing
    `journal` here would make two modules that already both depend on
    `schemas.py` depend on each other as well, for one method call.

    The journal RAISES on an invalid payload, so a tombstone that fails its
    native contract stops the reclamation before any byte is deleted. The
    journal's native check is deliberately WEAKER than this module's — it knows
    nothing of `tombstone_id`, `durability_class` or `reclamation_state` — so
    the sink re-validates the record against `validate_artifact_tombstone`
    first. `plan_expiry` validates only the `intent` record it builds; the
    `reclaimed` and `failed` variants are constructed afterwards by mutating a
    copy, and they reached the journal carrying this module's schema string
    without this module's validator ever having seen them.
    """

    KIND = "TOMBSTONE"

    def __init__(self, journal: Any, *, kind: str = KIND):
        if not callable(getattr(journal, "append", None)):
            raise TypeError("journal must expose append(kind, payload, ...)")
        self._journal = journal
        self._kind = kind

    def append(self, record: Mapping[str, Any]) -> Any:
        violations = validate_artifact_tombstone(record)
        if violations:
            raise StorageError(
                "refusing to journal an invalid artifact tombstone: "
                + "; ".join(violations)
            )
        payload = dict(record)
        # The journal's own name for the §5.8 retention class.
        payload["storage_class"] = payload.get("retention_class")
        payload.setdefault("path", payload.get("artifact_path"))
        entry = self._journal.append(self._kind, payload,
                                     campaign_id=payload.get("campaign_id"))
        # Both an entry object and a bare id are accepted; `expire_artifact`
        # is the one that decides whether what came back proves durability.
        return getattr(entry, "event_id", entry)


@dataclass(frozen=True)
class ExpiryOutcome:
    """What an expiry call did. `DRY_RUN` wrote nothing at all — not even a journal
    record — because a dry run that journals is not a dry run."""

    state: str            # "DRY_RUN" | "RECLAIMED"
    tombstone: dict
    measured_size_bytes: int
    measured_file_count: int
    deleted: bool
    journal_event_ids: tuple = ()


def _refuse(message: str) -> None:
    raise ExpiryRefused(message)


def _check_preconditions(artifact: ExpirableArtifact, policy: StoragePolicy) -> None:
    rule = EXPIRY_RULES[artifact.expirable_kind]
    facts = artifact.preconditions
    if not isinstance(facts, Mapping):
        _refuse(f"preconditions must be a mapping, got {type(facts).__name__}")
    missing = [name for name in rule.required_facts if name not in facts]
    if missing:
        _refuse(
            f"{artifact.expirable_kind}: missing precondition fact(s) {missing} — "
            f"{rule.describe}. A fact we cannot read is not a fact that is true."
        )

    if artifact.expirable_kind == "rejected_candidate_build_tree":
        status = facts["candidate_status"]
        if status not in _RECLAIMABLE_CANDIDATE_STATUSES:
            _refuse(f"candidate_status {status!r} is not one of "
                    f"{sorted(_RECLAIMABLE_CANDIDATE_STATUSES)}; a candidate still in "
                    "the running keeps its build tree")
        if facts["champion_status"] != "none":
            _refuse(f"champion_status is {facts['champion_status']!r}: a candidate on "
                    "the frontier or in a champion keeps its build tree")
        if facts["evaluation_events_journaled"] is not True:
            _refuse("evaluation_events_journaled is not True: the outcomes must be "
                    "durable before the bytes go (invariant 7)")
    elif artifact.expirable_kind == "retired_campaign_worktree":
        status = facts["campaign_status"]
        if status not in _RETIRED_CAMPAIGN_STATUSES:
            _refuse(f"campaign_status {status!r} is not one of "
                    f"{sorted(_RETIRED_CAMPAIGN_STATUSES)}")
        if facts["champion_artifacts_preserved"] is not True:
            _refuse("champion_artifacts_preserved is not True")
        if facts["evaluation_events_journaled"] is not True:
            _refuse("evaluation_events_journaled is not True")
    elif artifact.expirable_kind == "stale_profiler_trace":
        if facts["lineage_closed"] is not True:
            _refuse(f"lineage {facts['informed_lineage_id']!r} is not closed: a trace "
                    "is expirable only once the lineage it informed is done with it")
        age = facts["trace_age_days"]
        if isinstance(age, bool) or not isinstance(age, (int, float)):
            _refuse(f"trace_age_days must be a number, got {type(age).__name__}")
        if age < policy.min_profiler_trace_age_days:
            _refuse(f"trace is {age} day(s) old, below the "
                    f"{policy.min_profiler_trace_age_days}-day minimum")


def plan_expiry(
    artifact: ExpirableArtifact,
    policy: StoragePolicy,
    *,
    now: Optional[datetime] = None,
) -> ExpiryOutcome:
    """Validate a reclamation and build its tombstone WITHOUT writing anything.

    Every refusal below raises `ExpiryRefused`. Refusing loudly is the point:
    deletion authority is operator-only outside these narrow classes
    (`MEASUREMENT.md:223-229`), so a silent "nothing to do" would be
    indistinguishable from a rule that quietly stopped applying.
    """
    if not isinstance(artifact, ExpirableArtifact):
        raise TypeError(
            f"artifact must be an ExpirableArtifact, got {type(artifact).__name__}"
        )
    if not isinstance(policy, StoragePolicy):
        raise TypeError(f"policy must be a StoragePolicy, got {type(policy).__name__}")

    if artifact.retention_class not in RETENTION_CLASSES:
        _refuse(f"retention_class {artifact.retention_class!r} is not one of "
                f"{sorted(RETENTION_CLASSES)}")
    if artifact.retention_class != "expirable":
        _refuse(
            f"retention_class {artifact.retention_class!r} is not reclaimable: "
            "expiry runs only on 'expirable' (§5.8). Everything else is an "
            "operator decision (MEASUREMENT.md:223-229)."
        )
    if artifact.expirable_kind not in EXPIRY_RULES:
        _refuse(f"expirable_kind {artifact.expirable_kind!r} is not one of "
                f"{sorted(EXPIRY_RULES)}; §5.8 names exactly these three")
    if artifact.durability_class not in schemas.DURABILITY_CLASSES:
        _refuse(f"durability_class {artifact.durability_class!r} is not one of "
                f"{sorted(schemas.DURABILITY_CLASSES)}")
    if artifact.durability_class == "carried_in_git":
        _refuse("carried_in_git artifacts are not reclaimable: git already holds "
                "those bytes, so deleting the working copy reclaims nothing and "
                "loses the checkout")
    if not isinstance(artifact.sha256, str) or not _SHA256_RE.match(artifact.sha256):
        _refuse("sha256 must be a lowercase hex sha256 — a tombstone without a hash "
                "records that something was removed but not WHAT "
                "(use hash_file()/hash_tree_manifest())")
    for name in ("campaign_id", "reason", "rule_id", "actor"):
        value = getattr(artifact, name)
        if not isinstance(value, str) or not value.strip():
            _refuse(f"{name} must be a non-empty string")

    # `_norm` resolves symlinks, so the symlink test must run on the LITERAL
    # final component — otherwise a link into a legitimate owned root would pass
    # every containment check and delete its target while the link survived.
    literal = os.path.abspath(os.fspath(artifact.path))
    if os.path.islink(literal):
        _refuse(f"{literal!r} is a symlink; deleting through a symlink is how you "
                "remove something you did not mean to")

    resolved = _norm(artifact.path)
    if is_scratch_path(resolved):
        _refuse(
            f"{resolved!r} is under a scratch root; scratch hygiene is not this "
            "function's authority and needs no tombstone (a scratch path is never "
            "a citation of record, MEASUREMENT.md:146-156)"
        )
    for tree in production_tree_forms():
        if _under(resolved, tree):
            _refuse(f"{resolved!r} is inside the FROZEN production tree {tree!r} "
                    "(invariant 3) — denied regardless of declared owned roots")
        # The containment test has TWO directions and only one of them was
        # guarded. `rmtree` is recursive: a target that CONTAINS a production
        # tree destroys it just as thoroughly as a target inside one, and the
        # inside-test never fires because the production tree is the descendant.
        # Concretely, `owned_roots=('/mnt/raid0',)` with `path='/mnt/raid0/llm'`
        # produced an approved plan for 3.39 TB spanning all three frozen
        # kernel trees. Invariant 3 does not care which way the prefix runs.
        if _under(tree, resolved):
            _refuse(f"{resolved!r} CONTAINS the FROZEN production tree {tree!r}: "
                    "reclaiming it deletes production recursively (invariant 3) — "
                    "denied regardless of declared owned roots")
    parts = resolved.split("/")
    if ".git" in parts:
        _refuse(f"{resolved!r} is inside a .git directory; reclaiming repository "
                "internals is never a storage-plane action")
    if not policy.owned_roots:
        _refuse("policy.owned_roots is empty: a caller that declares no campaign-owned "
                "root may reclaim nothing")
    if not any(_under(resolved, root) and resolved != root for root in policy.owned_roots):
        _refuse(f"{resolved!r} is not strictly beneath any declared owned root "
                f"{list(policy.owned_roots)}")
    if not os.path.lexists(resolved):
        _refuse(
            f"{resolved!r} does not exist. A missing expirable artifact is an "
            "UNRECORDED loss, not a completed expiry — the bytes are gone and no "
            "tombstone says why. Record it deliberately."
        )

    _check_preconditions(artifact, policy)

    usage = measure_usage(resolved)
    if artifact.declared_size_bytes is not None:
        if artifact.declared_size_bytes != usage.bytes_on_disk:
            _refuse(
                f"declared_size_bytes {artifact.declared_size_bytes} != measured "
                f"{usage.bytes_on_disk}: the artifact changed since it was recorded, "
                "so the recorded hash may no longer describe it"
            )

    when = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    record = {
        "schema": SCHEMA_ARTIFACT_TOMBSTONE,
        "tombstone_id": tombstone_id(artifact.campaign_id, resolved, artifact.sha256,
                                     artifact.expirable_kind, artifact.rule_id),
        "campaign_id": artifact.campaign_id,
        "artifact_path": resolved,
        "artifact_sha256": artifact.sha256,
        "durability_class": artifact.durability_class,
        "retention_class": artifact.retention_class,
        "expirable_kind": artifact.expirable_kind,
        "rule_id": artifact.rule_id,
        "reason": artifact.reason,
        "actor": artifact.actor,
        "size_bytes": usage.bytes_on_disk,
        "file_count": usage.file_count,
        "preconditions": dict(artifact.preconditions),
        "reclaimed_at": when.isoformat(),
        "reclamation_state": "intent",
    }
    violations = validate_artifact_tombstone(record)
    if violations:
        # Unreachable via the guards above; kept because journalling an invalid
        # primary record is worse than refusing the reclamation.
        _refuse("the tombstone this reclamation would write is invalid: "
                + "; ".join(violations))
    return ExpiryOutcome(
        state="DRY_RUN",
        tombstone=record,
        measured_size_bytes=usage.bytes_on_disk,
        measured_file_count=usage.file_count,
        deleted=False,
    )


def expire_artifact(
    artifact: ExpirableArtifact,
    policy: StoragePolicy,
    *,
    journal: Any = None,
    force: bool = False,
    now: Optional[datetime] = None,
) -> ExpiryOutcome:
    """Reclaim an expirable artifact, tombstone first. DRY-RUN unless `force=True`.

    The journal contract is one method::

        journal.append(record: Mapping[str, Any]) -> str   # returns an event id

    It is REQUIRED on the `force` path and there is no default: an expiry that
    falls back to "journal unavailable, delete anyway" destroys the record the
    whole design exists to keep (invariant 7, `MEASUREMENT.md:173-176`).

    Sequence on `force=True`:
      1. append the tombstone with `reclamation_state: "intent"` and require a
         non-empty event id back — a journal that returns nothing has not
         demonstrated that the record is durable;
      2. delete the bytes;
      3. verify the path is gone (never report success unverified);
      4. append the same tombstone with `reclamation_state: "reclaimed"`.

    A crash between 1 and 4 leaves an intent with no completion: detectable and
    recoverable. On a deletion failure a `"failed"` record carrying the error is
    appended and the exception re-raised, so the journal never claims bytes are
    gone while they are still on disk.
    """
    plan = plan_expiry(artifact, policy, now=now)
    if not force:
        return plan

    if journal is None:
        _refuse(
            "force=True requires a journal: the tombstone is written BEFORE the "
            "bytes, and an expiry with nowhere to write it is a silent deletion"
        )
    append = getattr(journal, "append", None)
    if not callable(append):
        raise TypeError("journal must expose a callable append(record) -> event_id")

    intent = dict(plan.tombstone)
    intent_id = append(intent)
    if not isinstance(intent_id, str) or not intent_id.strip():
        _refuse(
            f"journal.append returned {intent_id!r} instead of an event id; the "
            "tombstone is not demonstrably durable, so the bytes stay"
        )

    target = plan.tombstone["artifact_path"]
    try:
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.unlink(target)
        if os.path.lexists(target):
            raise OSError(f"{target!r} still exists after deletion")
    except OSError as exc:
        failed = dict(plan.tombstone)
        failed["reclamation_state"] = "failed"
        failed["error"] = f"{type(exc).__name__}: {exc}"
        try:
            append(failed)
        except Exception:
            # The DELETION error is the one the caller must see and the one that
            # says whether the bytes are still there. Letting the journal's own
            # exception replace it turned an OSError into, e.g., a RuntimeError
            # and demoted the real cause to __context__, where an `except OSError`
            # caller never looks.
            raise exc from None
        raise

    done = dict(plan.tombstone)
    done["reclamation_state"] = "reclaimed"
    done_id = append(done)
    if not isinstance(done_id, str) or not done_id.strip():
        # The intent id was validated and the completion id was not, so a journal
        # that returned None produced state="RECLAIMED" with
        # journal_event_ids=("ev-1", None): a success-shaped result asserting a
        # durable completion record that was never demonstrated. The bytes are
        # already gone, so this cannot be undone — but it must not be reported as
        # a clean reclamation. The intent record is on disk, which is the same
        # detectable, recoverable state a crash here would leave.
        raise StorageError(
            f"{target!r} was deleted and journal.append returned {done_id!r} "
            "instead of an event id for the 'reclaimed' record: the completion is "
            "not demonstrably durable. The 'intent' record "
            f"({intent_id!r}) is the recovery anchor — reconcile it against the "
            "filesystem rather than assuming the reclamation completed cleanly."
        )
    return ExpiryOutcome(
        state="RECLAIMED",
        tombstone=done,
        measured_size_bytes=plan.measured_size_bytes,
        measured_file_count=plan.measured_file_count,
        deleted=True,
        journal_event_ids=(intent_id, done_id),
    )


# =============================================================================
# verify_durability — PASS / FAIL / COULD_NOT_CHECK, per citation
# =============================================================================

@dataclass(frozen=True)
class CitationVerdict:
    index: int
    path: Optional[str]
    declared_class: Optional[str]
    check: Check

    @property
    def outcome(self) -> str:
        return self.check.outcome


def verify_durability(
    citations: Sequence[Mapping[str, Any]],
    *,
    tracked_index: Optional[TrackedIndex] = None,
    carry_threshold_bytes: int = DEFAULT_CARRY_THRESHOLD_BYTES,
) -> tuple:
    """Check each citation's recorded durability class against reality.

    A citation is a mapping with `path` and `durability_class`, plus `sha256`
    and `provenance` when the class is `hash_and_provenance_only`.

    Verdicts:
      * FAIL — a scratch path (an ERROR by `MEASUREMENT.md:146-156`, never a
        warning); a class outside `schemas.DURABILITY_CLASSES`; `carried_in_git`
        that git does not carry; `durable_untracked` whose path is gone;
        `hash_and_provenance_only` with no hash or no provenance — that class is
        a promise to have recorded both, so an empty one is a bare assertion.
      * COULD_NOT_CHECK — no `tracked_index` for an in-repo class question; a
        path we are not permitted to stat; an index that disclaims the path.
        This is a THIRD outcome: reporting it as PASS hides a real loss, and
        reporting it as FAIL manufactures one.
      * PASS — the recorded class is the class the filesystem and git agree on.
    """
    if isinstance(citations, Mapping) or isinstance(citations, (str, bytes)):
        raise TypeError("citations must be a sequence of citation mappings")
    verdicts = []
    for index, citation in enumerate(citations):
        try:
            verdicts.append(
                _verify_one(index, citation, tracked_index, carry_threshold_bytes))
        except (StorageError, OSError) as exc:
            # A per-citation verifier must return a verdict per citation. A
            # `TrackedIndex` that disclaims a path RAISES by design (that is the
            # whole reason it never guesses), and the `durable_untracked` branch
            # called it unguarded — so one unanswerable citation destroyed the
            # verdicts of every other citation in the batch, including the FAILs.
            # Narrow on purpose: only refusals and I/O become COULD_NOT_CHECK.
            # A TypeError here is a bug in this module and must still escape.
            path = citation.get("path") if isinstance(citation, Mapping) else None
            declared = (citation.get("durability_class")
                        if isinstance(citation, Mapping) else None)
            verdicts.append(CitationVerdict(
                index, path if isinstance(path, str) else None, declared,
                Check(COULD_NOT_CHECK,
                      (f"citation could not be evaluated: "
                       f"{type(exc).__name__}: {exc}",))))
    return tuple(verdicts)


def _verify_one(
    index: int,
    citation: Any,
    tracked_index: Optional[TrackedIndex],
    carry_threshold_bytes: int = DEFAULT_CARRY_THRESHOLD_BYTES,
) -> CitationVerdict:
    if not isinstance(citation, Mapping):
        return CitationVerdict(index, None, None, Check(
            FAIL, (f"citation is not a mapping, got {type(citation).__name__}",)))
    raw_path = citation.get("path")
    declared = citation.get("durability_class")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return CitationVerdict(index, None, declared, Check(
            FAIL, ("citation has no usable 'path'",)))
    if declared not in schemas.DURABILITY_CLASSES:
        return CitationVerdict(index, raw_path, declared, Check(
            FAIL, (f"durability_class {declared!r} is not one of "
                   f"{sorted(schemas.DURABILITY_CLASSES)}; an unclassified citation "
                   "cannot distinguish a defect from an expected absence (§3.7)",)))

    try:
        resolved = _norm(raw_path)
    except (TypeError, ValueError) as exc:
        return CitationVerdict(index, raw_path, declared, Check(
            COULD_NOT_CHECK, (f"path is unusable: {exc}",)))
    if is_scratch_path(resolved):
        return CitationVerdict(index, resolved, declared, Check(
            FAIL, (f"citation resolves to a scratch root: {resolved!r}. Scratch "
                   "paths MUST NOT be the citation of record "
                   "(MEASUREMENT.md:146-156).",)))

    if declared == "hash_and_provenance_only":
        reasons = []
        sha = citation.get("sha256")
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            reasons.append("hash_and_provenance_only requires a lowercase hex "
                           "'sha256'; without it there is nothing to check against")
        provenance = citation.get("provenance")
        if not isinstance(provenance, str) or not provenance.strip():
            reasons.append("hash_and_provenance_only requires a non-empty "
                           "'provenance' saying where the artifact came from and why "
                           "it is not carried (MEASUREMENT.md:146-156)")
        if reasons:
            return CitationVerdict(index, resolved, declared, Check(FAIL, tuple(reasons)))
        return _verify_hash_and_provenance_only(
            index, resolved, declared, tracked_index, carry_threshold_bytes)

    # `os.path.lexists` swallows every OSError, which would turn "we are not
    # permitted to look" into "it is not there" — a manufactured loss. lstat
    # keeps the two apart: FileNotFoundError is an absence, anything else is an
    # inability to evaluate.
    try:
        os.lstat(resolved)
        exists = True
    except FileNotFoundError:
        exists = False
    except OSError as exc:
        return CitationVerdict(index, resolved, declared, Check(
            COULD_NOT_CHECK, (f"cannot stat {resolved!r}: {exc}",)))

    if declared == "durable_untracked":
        if not exists:
            return CitationVerdict(index, resolved, declared, Check(
                FAIL, (f"{resolved!r} is recorded durable_untracked but does not "
                       "exist; nothing versions it and nothing holds it",)))
        if tracked_index is None:
            return CitationVerdict(index, resolved, declared, Check(
                COULD_NOT_CHECK,
                ("path exists, but without a tracked_index we cannot confirm it is "
                 "inside a working tree rather than loose on the filesystem",)))
        if not tracked_index.contains_repo(resolved):
            return CitationVerdict(index, resolved, declared, Check(
                FAIL, (f"{resolved!r} is recorded durable_untracked but lies outside "
                       "the working tree; it should be hash_and_provenance_only",)))
        return CitationVerdict(index, resolved, declared, Check(PASS))

    # declared == "carried_in_git"
    if tracked_index is None:
        return CitationVerdict(index, resolved, declared, Check(
            COULD_NOT_CHECK,
            ("no tracked_index supplied; git tracked-ness is unknown and "
             "'the file is on disk' is not the question — untracked looks "
             "identical to committed on the filesystem",)))
    try:
        if not tracked_index.contains_repo(resolved):
            return CitationVerdict(index, resolved, declared, Check(
                FAIL, (f"{resolved!r} is recorded carried_in_git but lies outside "
                       "the working tree",)))
        tracked = tracked_index.is_tracked(resolved)
    except UnclassifiablePath as exc:
        return CitationVerdict(index, resolved, declared, Check(
            COULD_NOT_CHECK, (f"tracked-ness could not be determined: {exc}",)))
    if not tracked:
        return CitationVerdict(index, resolved, declared, Check(
            FAIL, (f"{resolved!r} is recorded carried_in_git but git does not track "
                   "it; the citation claims a durability the repository does not "
                   "provide",)))
    if not exists:
        return CitationVerdict(index, resolved, declared, Check(
            COULD_NOT_CHECK,
            ("git tracks the path but it is absent from the working tree — a "
             "checkout state question, not a durability loss",)))
    return CitationVerdict(index, resolved, declared, Check(PASS))


def _verify_hash_and_provenance_only(
    index: int,
    resolved: str,
    declared: str,
    tracked_index: Optional[TrackedIndex],
    carry_threshold_bytes: int,
) -> CitationVerdict:
    """Is `hash_and_provenance_only` the class this artifact is ENTITLED to?

    `classify()` refuses to derive this class from absence, on the stated grounds
    that doing so "would relabel every loss as an intended design decision". The
    verifier then accepted the very claim classify refuses to produce: a deleted,
    git-TRACKED, in-repo file re-declared `hash_and_provenance_only` with any
    syntactically valid hash and any non-empty prose scored PASS, where the same
    file scored COULD_NOT_CHECK as `carried_in_git` and FAIL as
    `durable_untracked`. The one door classify locks was standing open in the
    function that actually runs over a registry, and relabelling was the key.

    The class means "too large or too far outside the tree to carry, so we
    deliberately kept only the hash". That premise is checkable whenever an index
    is supplied, and where it is contradicted the citation is a misdeclaration,
    not a pass.
    """
    if tracked_index is None:
        # Nothing to contradict the claim with. PASS on the fields alone, as before.
        return CitationVerdict(index, resolved, declared, Check(PASS))
    try:
        in_repo = tracked_index.contains_repo(resolved)
    except UnclassifiablePath as exc:
        return CitationVerdict(index, resolved, declared, Check(
            COULD_NOT_CHECK, (f"cannot tell whether {resolved!r} is inside a "
                              f"working tree: {exc}",)))
    if not in_repo:
        # Outside every working tree: nothing versions those bytes, so the class
        # is exactly right and the hash is all there ever was.
        return CitationVerdict(index, resolved, declared, Check(PASS))
    try:
        tracked = tracked_index.is_tracked(resolved)
    except UnclassifiablePath as exc:
        return CitationVerdict(index, resolved, declared, Check(
            COULD_NOT_CHECK, (f"tracked-ness could not be determined: {exc}",)))
    if tracked:
        return CitationVerdict(index, resolved, declared, Check(
            FAIL, (f"{resolved!r} is recorded hash_and_provenance_only but git "
                   "TRACKS it: the class asserts only a hash was kept, while the "
                   "repository carries the bytes. The correct class is "
                   "carried_in_git, and recording this one hides whether the "
                   "working-tree copy is a loss or a checkout state (§3.7).",)))
    try:
        st = os.lstat(resolved)
    except FileNotFoundError:
        return CitationVerdict(index, resolved, declared, Check(
            COULD_NOT_CHECK,
            (f"{resolved!r} is recorded hash_and_provenance_only, lies inside a "
             "working tree, and is absent. That class is a claim about SIZE — "
             "too large to carry — and an absent artifact cannot substantiate "
             "it. Whether this is an expected absence or a loss is exactly what "
             "§3.7 exists to distinguish, and here it is undetermined.",)))
    except OSError as exc:
        return CitationVerdict(index, resolved, declared, Check(
            COULD_NOT_CHECK, (f"cannot stat {resolved!r}: {exc}",)))
    if not os.path.isdir(resolved) and st.st_blocks * 512 <= carry_threshold_bytes:
        return CitationVerdict(index, resolved, declared, Check(
            FAIL, (f"{resolved!r} is recorded hash_and_provenance_only but is an "
                   f"in-repo file of {st.st_blocks * 512} bytes on disk, at or "
                   f"under the {carry_threshold_bytes}-byte carry threshold: it "
                   "is small enough to carry, so the class understates what is "
                   "actually recoverable. The correct class is durable_untracked.",)))
    return CitationVerdict(index, resolved, declared, Check(PASS))


__all__ = [
    "REPO_ROOT", "EVIDENCE_DIRNAME", "EPHEMERAL_ROOTS", "PRODUCTION_TREES",
    "DEFAULT_CARRY_THRESHOLD_BYTES", "DEFAULT_HEADROOM_FLOOR_GB",
    "DEFAULT_LARGEST_SINGLE_ALLOCATION_GB", "DEFAULT_MIN_PROFILER_TRACE_AGE_DAYS",
    "RETENTION_CLASSES", "EXPIRABLE_KINDS", "EXPIRY_RULES",
    "STORAGE_OK", "DISK_PRESSURE", "QUOTA_OK", "QUOTA_WARN", "QUOTA_EXHAUSTED",
    "SCHEMA_ARTIFACT_TOMBSTONE", "TOMBSTONE_ID_PREFIX", "RECLAMATION_STATES",
    "README_STUB_MARKER", "README_PLACEHOLDER", "SHA256SUMS_NAME", "README_NAME",
    "production_tree_forms",
    "StorageError", "ScratchCitationError", "UnclassifiablePath",
    "EvidenceRootError", "ExpiryRefused",
    "is_scratch_path", "assert_not_scratch",
    "TrackedIndex", "StaticTrackedIndex", "GitTrackedIndex",
    "Classification", "classify",
    "EvidenceRoot", "campaign_evidence_root", "ensure_campaign_evidence_root",
    "check_evidence_root_layout",
    "Usage", "measure_usage", "StoragePolicy", "StorageState", "disk_pressure",
    "QuotaState", "campaign_quota_state",
    "hash_file", "hash_tree_manifest",
    "validate_artifact_tombstone", "tombstone_id", "JournalTombstoneSink",
    "ExpirableArtifact", "ExpiryRule", "ExpiryOutcome",
    "plan_expiry", "expire_artifact",
    "CitationVerdict", "verify_durability",
]
