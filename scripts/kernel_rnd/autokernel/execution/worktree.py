"""autokernel.execution.worktree — experimental worktrees, builds, receipts, teardown.

WHAT THIS MODULE IS FOR
=======================
AK2's biggest gap. Before it, nothing in this project created an experimental
worktree, namespaced a branch, made a pathspec-limited commit in the shared
clone, or produced a build identity receipt. Design §5.3 and §8.5 describe all
four; `evaluator/integrity.py` §8.5.1 (2) consumes the fourth. This is the
implementation.

Four capabilities, in the order a campaign uses them:

1. **Anchor and create.** `resolve_anchor()` reads the CURRENT tip of a
   production branch; `create_campaign_worktree()` checks that tip out into
   `llama.cpp-ak-<campaign_id>` on a branch namespaced `ak/<campaign_id>/…`.
2. **Commit, always by pathspec.** `Worktree.commit_paths()` names every path it
   commits. There is no code path here that stages a directory, a glob, or `.`.
3. **Build, and receipt it.** `BuildPlan` constructs the argv,
   `run_build()` executes it under a PID this module owns, `parse_build_log()`
   reads the result, and `BuildIdentity` is the receipt —
   `to_build_provenance()` hands `evaluator.integrity` exactly the record it
   already asks for.
4. **Tear down, and prove it.** `teardown_worktree()` removes the worktree,
   deletes the branch, and returns fingerprints proving every frozen production
   tree is byte-identical to before.

THE SAFETY PROPERTY, AND WHY IT IS STRUCTURAL
=============================================
The requirement is not "it checks whether the path is production". A check is a
branch someone can add a second, unchecked call site around. The requirement is
that a production content mutation is **not expressible** in this module's
types. Two classes carry that:

* :class:`GitRepo` may address ANY repository, including a frozen production
  tree — it has to, because resolving the production tip and running
  ``git worktree add`` are how anchoring works. Its verb allowlist therefore
  **excludes every content-mutating verb** (`commit`, `checkout`, `switch`,
  `reset`, `merge`, `rebase`, `apply`, `am`, `stash`, `clean`, `restore`, `add`,
  `rm`, `mv`, `push`, `pull`, `cherry-pick`, `revert`, `tag`, `gc`, `filter-branch`).
  There is no method on it that writes a working tree or an index.
* :class:`Worktree` carries every content-mutating capability, and its
  constructor takes a :class:`SandboxPath` — a value type that **cannot be
  constructed** for a path that resolves inside a frozen production tree. Not
  "raises if you call the wrong method": the object that names the tree does not
  exist.

So the composition — a mutating verb aimed at a production tree — has no type to
travel in. `test_worktree.py` proves the two halves separately and proves the
composition is unreachable, including via a symlink, a `..` traversal, a
destination whose parent link resolves into a frozen tree, and a branch name
that merely *starts with* `production-`.

THREE ROUTES ROUND THAT PROPERTY, FOUND BY RED TEAM 2026-08-03
--------------------------------------------------------------
Each was a real, demonstrated write into a frozen clone, and each is now closed
with its own regression test in `TestRedTeam*`:

1. **`git worktree add` is a content-mutating verb.** `worktree` was on
   `GitRepo.ALLOWED_VERBS` (it must be — that is how anchoring works) and
   `_WORKTREE_MUTATING_SUBCOMMANDS` was defined and *read by nothing*.
   `repo._git("worktree", "add", "<inside the frozen tree>", commit)` wrote a
   complete checkout there. A guard that is written and never wired is the
   defect this package keeps producing; `_git` now enforces the set, and the two
   typed entry points build their argv through `_git_admin`.
2. **`git config` writes, and a linked worktree's config is the CLONE's.**
   `config` was allowed on both classes for reads; nothing distinguished a read
   from a write, and a write from a campaign worktree lands in the frozen
   clone's shared `.git/config`. `_require_config_is_a_read` now separates them.
3. **`GIT_DIR`/`GIT_WORK_TREE` ignore `-C` entirely.** With those set in the
   ambient environment, `Worktree.commit_paths()` staged a file into the frozen
   tree's index — the `SandboxPath` was true and irrelevant, because git never
   consulted the directory. Every process this module launches now runs with the
   git redirect variables stripped (`_sanitized_env`).

The fourth was the type itself: `SandboxPath.create()` defaulted to
`production_trees=()`, so the safety type's DEFAULT spelling checked nothing and
`SandboxPath.create("/mnt/raid0/llm/llama.cpp")` succeeded. The default is now
the real frozen set; `()` is still expressible and is now something a caller
has to write down.

`SandboxPath` resolves with `os.path.realpath` before it compares, and compares
**path components**, never string prefixes. `/mnt/raid0/llm/llama.cpp-ak-x` is a
sibling of `/mnt/raid0/llm/llama.cpp` and must be allowed even though its name
starts with the production tree's name; `/mnt/raid0/llm/llama.cpp/build` is
inside it and must not be. A `startswith` test gets exactly one of those right.

WHY THE ANCHOR IS RE-RESOLVED AT CREATION
=========================================
INC-20260706-iqk-missing-subsystem: an experimental branch forked from an old
production tip accumulated work while production moved, and the candidate
silently lacked every optimization that had landed since. CLAUDE.md's step 1 is
therefore *"pull fresh production → experimental — start from the current
production tip; never accumulate on a long-lived branch forked from an old
tip."* `create_campaign_worktree()` re-resolves the branch tip at the moment it
creates the worktree and raises :class:`StaleAnchor` if it moved since the
anchor was taken. The incident becomes a failed call rather than a code review.

WHY `GGML_CCACHE=OFF` IS THE DEFAULT
====================================
`ggml/CMakeLists.txt:125` defaults `GGML_CCACHE` to `ON`, and every recorded
build log on this host carries *"ccache found, compilation results will be
cached"*. §8.5.1 (2) requires a **clean build from the recorded snapshot** and
gives the reason: *"an incremental build can link stale objects and hide the
error that the snapshot would surface, which would make the actor's build state
part of the artifact."* A shared ccache reintroduces precisely that across build
directories — a "fresh build directory" whose objects come from a cache
populated by a different tree is not a clean build, and `build_dir_pre_build_digest
== EMPTY_TREE_SHA256` cannot see it. So this module turns ccache OFF unless the
caller says `allow_ccache=True`, and either way `BuildLogFacts.ccache_enabled`
records what actually happened, because the log is the only witness.

PROCESS DISCIPLINE
==================
Every process is launched by :func:`_run_owned` with `start_new_session=True`,
so the child is a session leader whose process-group id equals the PID we
captured. On timeout the escalation is TERM → grace → KILL against **that pgid
and no other**, followed by a reap, and :class:`ProcessDisposition` records
`verified_dead` from the reap rather than from optimism. There is no name
pattern anywhere: INC-20260731 is what a name pattern costs on a shared host —
`llama-server -m` matched another agent's server twice, and `earlyoom` died
because its own command line contains the names it guards.
:func:`audit_no_name_pattern_process_ops` proves the absence from this module's
own AST, the way `evaluator.integrity` proves its own.

WHAT THIS MODULE DOES NOT DO
============================
It runs no benchmark and no inference — those need a held claim
(`resource/device_claim.py` for GPU, its CPU sibling for the region) and belong
to the executor modules beside this one. It never writes a registry row, an era
row, an AutoPilot baseline or a production-named branch. It does not commit on
the caller's behalf outside a campaign branch.

Governing design: `handoffs/active/autokernel-research-loop.md` §5.3, §8.5,
§8.5.1, phase AK2. Governing protocol: `P-AK-SEARCH-1` (Annex K).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import time
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Iterable, Mapping, Optional, Sequence

from .. import schemas
from ..evaluator import integrity
from . import sandbox as process_sandbox

__all__ = [
    # identity
    "DESIGN_SECTION", "PROTOCOL_ID",
    # policy constants
    "PRODUCTION_TREES", "PRODUCTION_TREE_ALIASES", "PROTECTED_BRANCH_PREFIXES",
    "DEFAULT_WORKTREE_ROOT", "DEFAULT_BUILD_ROOT", "CONTENT_MUTATING_VERBS",
    "CLEAN_BUILD_CMAKE_DEFINES",
    # errors
    "WorktreeError", "ProductionTreeViolation", "UnsafePath", "UnsafeBranch",
    "UnsafePathspec", "GitCommandFailed", "StaleAnchor", "ProductionMutated",
    "BuildDirNotFresh", "ProcessEscalationFailed", "HostTooContended",
    "ArtifactNotFromThisBuild",
    # value types
    "SandboxPath", "SafeBranch", "Pathspec", "ProcessDisposition",
    # git plane
    "GitRepo", "Worktree", "Anchor", "resolve_anchor",
    "campaign_worktree_path", "snapshot_worktree_path",
    "create_campaign_worktree", "create_snapshot_worktree",
    # immutability proof
    "TreeFingerprint", "fingerprint_tree", "ImmutabilityProof", "prove_unchanged",
    "TeardownReceipt", "teardown_worktree",
    # build plane
    "BuildParallelism", "BuildPlan", "BuildLogFacts", "parse_build_log",
    "BuildResult", "run_build", "BuildIdentity", "build_identity",
    "default_build_dir", "with_parallelism",
    # self-audit
    "audit_no_name_pattern_process_ops",
]

DESIGN_SECTION = "autokernel-research-loop.md §5.3, §8.5, §8.5.1 (AK2)"
PROTOCOL_ID = "P-AK-SEARCH-1/v1"

PASS = schemas.PASS
FAIL = schemas.FAIL
COULD_NOT_CHECK = schemas.COULD_NOT_CHECK


# =============================================================================
# Policy constants
#
# Mirrors, named as mirrors. `storage.PRODUCTION_TREES` and
# `evaluator.correctness.PRODUCTION_TREE_ROOTS` already carry these; a third
# copy is how one of them quietly loses an entry, so `test_worktree.py` imports
# both and asserts this superset agrees with each.
# =============================================================================

#: The frozen production kernel trees (CLAUDE.md 2026-07-25 v8 freeze +
#: 2026-07-31 speech-kernel freeze). Invariant 3: no actor builds in or modifies
#: a production tree.
PRODUCTION_TREES = (
    "/mnt/raid0/llm/llama.cpp",
    "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp",
)

#: Symlink aliases for the same trees. `/workspace/repos/<name>` is a symlink to
#: `/mnt/raid0/llm/<name>` — one clone, two names (CLAUDE.md, *Working-tree
#: identity*). `SandboxPath` resolves symlinks before comparing, so these are
#: redundant *when the link exists*; they are listed anyway because a denial that
#: depends on a symlink still being present is a denial that a `rm` turns off.
PRODUCTION_TREE_ALIASES = (
    "/workspace/repos/epyc-llama",
    "/workspace/repos/epyc-whisper",
    "/workspace/repos/epyc-qwentts",
)

#: A branch name is refused if it starts with any of these, case-folded.
#:
#: `schemas._PRODUCTION_BRANCH_RE` matches `production-(consolidated|speech)-vN`
#: exactly. That is right for validating a record; it is too narrow for a guard
#: on a name we are about to hand to `git branch -D`. `production-consolidated-v9`
#: does not exist yet and `production-experimental-scratch` never will, but
#: neither should be deletable by this module, so the prefix is what is refused.
PROTECTED_BRANCH_PREFIXES = ("production-", "prod-", "release-", "stable-")

#: Where campaign worktrees live (design §5.3: *"a dedicated worktree under
#: `/mnt/raid0/llm/`"*). 13 worktrees already sit here; the `-ak-` namespace is
#: what keeps a campaign from colliding with them or with another session's
#: `llama.cpp-experimental`.
DEFAULT_WORKTREE_ROOT = "/mnt/raid0/llm"

#: Where build directories live. Deliberately NOT inside the worktree:
#: `integrity.check_clean_build_from_snapshot` sub-check (d) FAILs a `build_dir`
#: that resolves inside the actor's worktree, because then *"the actor's build
#: state would become part of the artifact"*.
DEFAULT_BUILD_ROOT = "/mnt/raid0/llm/ak-build"

#: Verbs that write a working tree, an index, a ref, or an object store.
#: `GitRepo` — the only class that may name a production tree — excludes all of
#: them from its allowlist, which is what makes "mutate production" unexpressible
#: rather than merely refused.
CONTENT_MUTATING_VERBS = frozenset({
    "add", "am", "apply", "checkout", "cherry-pick", "clean", "commit",
    "filter-branch", "gc", "merge", "mv", "notes", "pull", "push", "rebase",
    "reflog", "reset", "restore", "revert", "rm", "stash", "switch", "tag",
    "update-index", "update-ref", "write-tree",
})

#: Defines forced ON for a clean build from a recorded snapshot. See the module
#: docstring — `GGML_CCACHE` defaults to `ON` upstream
#: (`ggml/CMakeLists.txt:125`) and a shared object cache defeats §8.5.1 (2)
#: without leaving a trace in `build_dir_pre_build_digest`.
CLEAN_BUILD_CMAKE_DEFINES = (("GGML_CCACHE", "OFF"),)

#: Binaries whose whole purpose is to select processes by NAME PATTERN.
#: INC-20260731 is what using one on this host costs. `audit_no_name_pattern_
#: process_ops` refuses any of these appearing in an argv built by this module,
#: and `test_worktree.py` asserts this tuple still contains all of them — an
#: audit whose denylist can be emptied is an audit you pass by deleting what it
#: inspects.
_NAME_PATTERN_BINARIES = (
    "pkill", "pgrep", "killall", "killall5", "skill", "snice", "pidof", "fuser",
)

#: Environment variables that RE-TARGET a git command regardless of `-C`.
#: `git -C <sandbox> add` with `GIT_DIR`/`GIT_WORK_TREE` pointing at a frozen
#: tree stages into the FROZEN tree — the `SandboxPath` on the `Worktree` is
#: true and irrelevant, because git never looked at the cwd. Red-team probe
#: 2026-08-03 demonstrated exactly that (`A  smuggled.txt` in the frozen index).
#: Every process this module launches therefore runs with these removed:
#: the path in the argv is the only thing that decides which repository is
#: touched. `GIT_CONFIG*` is included because `GIT_CONFIG_COUNT`/`_KEY_n`/
#: `_VALUE_n` inject config (`core.hooksPath`, `alias.*`) into that same command.
_GIT_ENV_REDIRECTS = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES", "GIT_PREFIX", "GIT_INDEX_VERSION",
)
_GIT_ENV_REDIRECT_PREFIXES = ("GIT_CONFIG",)

#: `git config` forms that only READ. Anything else writes a config file, and
#: for a linked worktree that file is the SHARED `.git/config` of the frozen
#: clone the worktree hangs off — `-C <sandbox>` does not make it a sandbox
#: write. Both `GitRepo` and `Worktree` allow `config` for reads only.
_CONFIG_READ_ONLY_FLAGS = frozenset({
    "--get", "--get-all", "--get-regexp", "--get-urlmatch", "--get-color",
    "--get-colorbool", "--list", "-l", "--name-only", "--null", "-z",
    "--type", "--bool", "--int", "--path", "--includes", "--no-includes",
    "--show-origin", "--show-scope", "--default", "--local", "--global",
    "--system", "--worktree", "--file", "-f", "--blob",
})

#: `ak-` is not decoration. Because the worktree directory name is
#: `<source_tree>-<campaign_id>`, requiring the id to start with `ak-` is what
#: structurally produces the `llama.cpp-ak-…` namespace design §5.3 asks for.
_CAMPAIGN_ID_RE = re.compile(r"^ak-[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
_SOURCE_TREE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,62}$")
_CPU_LIST_RE = re.compile(r"^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$")
_CMAKE_DEFINE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


# =============================================================================
# Errors
# =============================================================================

class WorktreeError(RuntimeError):
    """Base for every refusal and failure raised by this module."""


class ProductionTreeViolation(WorktreeError):
    """A frozen production tree was named where only an experimental tree may be.

    Raised by `SandboxPath` at construction. It is a *constructor* error and not
    a method error on purpose: the caller never gets an object it could pass on.
    """


class UnsafePath(WorktreeError):
    """A path is not absolute, escapes its sandbox, or cannot be resolved."""


class UnsafeBranch(WorktreeError):
    """A branch name is protected, badly namespaced, or not a legal git ref."""


class UnsafePathspec(WorktreeError):
    """A commit pathspec is empty, magic, wildcarded, or leaves the worktree."""


class GitCommandFailed(WorktreeError):
    """A git invocation returned non-zero. Carries argv, code and output."""

    def __init__(self, argv: Sequence[str], returncode: int, output: str):
        self.argv = tuple(argv)
        self.returncode = returncode
        self.output = output
        super().__init__(
            f"git command failed (exit {returncode}): {' '.join(self.argv)}\n"
            f"{output.strip()}")


class StaleAnchor(WorktreeError):
    """The production tip moved between resolving the anchor and using it.

    INC-20260706-iqk-missing-subsystem in one exception: a worktree forked from
    a tip that is no longer the tip is missing everything that landed since, and
    the failure mode is silent — the candidate builds, runs, and is slower than
    production for a reason nobody attributes to the fork point.
    """


class ProductionMutated(WorktreeError):
    """A frozen production tree is not byte-identical to its `before` fingerprint.

    This is raised, never returned as a flag. A teardown that changed production
    is not a degraded result; it is the one outcome this module exists to make
    impossible, so it stops the caller.
    """


class BuildDirNotFresh(WorktreeError):
    """The build directory already had content. §8.5.1 (2) requires it fresh."""


class HostTooContended(WorktreeError):
    """The 1-minute load average exceeds the cap the caller declared.

    `BuildParallelism.load_average_cap` used to be recorded in the receipt and
    read by nothing, which is the worst of both: the artifact said a cap was in
    force and no cap was ever applied. It is now a precondition, and the
    observed load is recorded next to it.
    """


class ArtifactNotFromThisBuild(WorktreeError):
    """A receipt was asked to attest a binary that is not under its build_dir.

    `build_identity(output_binary=…)` hashes whatever path it is handed. Handed
    a production binary it would emit a receipt saying the candidate produced
    it, with a real digest of a real file — evidence that is true field by field
    and false as a whole. Containment is checked instead.
    """


class ProcessEscalationFailed(WorktreeError):
    """A process this module launched survived TERM and KILL.

    Reported rather than papered over: CLAUDE.md's process rule is *"never report
    success until confirmed"*, and an unconfirmed kill on a shared host is the
    beginning of the next incident.
    """


# =============================================================================
# Small validators — the scalar ones are `schemas.require`; `_req_tree_digest`
# below is this module's own, and the comment on it says why it has to be
# =============================================================================
#
# These used to be local "on purpose, so this module does not depend on another
# module's privates". The purpose was real and the outcome was that this copy
# silently lacked the placeholder-digest rejection the `t0_provider` copy had.
# `schemas.require` is not another module's privates — it is the bottom layer
# every module here already imports, and the field type is public.

_req_str = schemas.require.str
_req_sha256 = schemas.require.sha256
_req_commit = schemas.require.commit


def _req_tree_digest(value: Any, label: str) -> str:
    """A tree digest, where `integrity.EMPTY_TREE_SHA256` is a MEASURED observation.

    `_req_sha256` refuses `schemas.is_placeholder_digest`, and that set contains the
    sha256 of the empty input. For a *tree* digest that is not filler: a fresh build
    directory is empty, `integrity.hash_source_tree` of an empty tree hashes an empty
    manifest, and `run_build(require_fresh_build_dir=True)` REQUIRES the digest to be
    exactly that value (`:2180`). Refusing it here would refuse the one reading that
    proves the build directory was clean.

    Every other filler is still refused, because a repeated hex character is never
    what a manifest hashes to.
    """
    if isinstance(value, str) and value == integrity.EMPTY_TREE_SHA256:
        return value
    return _req_sha256(value, label)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "+00:00")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_regular_sink(path: str):
    """Open a new evaluator-owned stream without following or replacing links."""
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    facts = os.fstat(descriptor)
    if (not stat.S_ISREG(facts.st_mode) or facts.st_nlink != 1
            or facts.st_uid != os.geteuid()
            or stat.S_IMODE(facts.st_mode) != 0o600):
        os.close(descriptor)
        raise WorktreeError(f"stream sink is not a one-link regular file: {path}")
    return os.fdopen(descriptor, "w+b")


def _stream_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid,
        "mode": stat.S_IMODE(value.st_mode), "nlink": value.st_nlink,
        "size": value.st_size, "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _read_and_revalidate_open_stream(handle: Any, path: str) -> tuple[bytes, dict[str, int]]:
    """Read the writer fd and prove the pathname still names that inode."""
    handle.flush()
    os.fsync(handle.fileno())
    before = os.fstat(handle.fileno())
    linked_before = os.stat(path, follow_symlinks=False)
    if (_stream_identity(before) != _stream_identity(linked_before)
            or before.st_nlink != 1 or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600):
        raise WorktreeError(f"stream pathname no longer names its writer inode: {path}")
    os.lseek(handle.fileno(), 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(handle.fileno(), 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(handle.fileno())
    linked_after = os.stat(path, follow_symlinks=False)
    if (not (_stream_identity(before) == _stream_identity(after)
             == _stream_identity(linked_before) == _stream_identity(linked_after))):
        raise WorktreeError(f"stream changed during fd-bound read: {path}")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise WorktreeError(f"stream size changed during fd-bound read: {path}")
    return raw, _stream_identity(after)


def _revalidate_open_stream(handle: Any, path: str,
                            expected: Mapping[str, int]) -> None:
    current = _stream_identity(os.fstat(handle.fileno()))
    linked = _stream_identity(os.stat(path, follow_symlinks=False))
    if current != dict(expected) or linked != dict(expected):
        raise WorktreeError(f"stream identity changed before receipt publication: {path}")


def _sealed_process_receipt(path: str, body: Mapping[str, Any]) -> str:
    """Publish one append-only, self-hashed owned-process receipt."""
    payload = dict(body)
    payload["receipt_sha256"] = schemas.content_hash(payload)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    name = os.path.basename(path)
    temporary_name = f".{name}.{os.getpid()}.tmp"
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(parent, directory_flags)
    handle = None
    try:
        temporary = os.path.join(parent, temporary_name)
        handle = _exclusive_regular_sink(temporary)
        handle.write(raw)
        written, identity = _read_and_revalidate_open_stream(handle, temporary)
        if written != raw:
            raise WorktreeError("temporary process receipt bytes changed")
        os.link(temporary_name, name, src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd, follow_symlinks=False)
        linked = _stream_identity(os.stat(name, dir_fd=directory_fd,
                                          follow_symlinks=False))
        linked_fd = _stream_identity(os.fstat(handle.fileno()))
        if (linked != linked_fd or linked["nlink"] != 2
                or (linked["device"], linked["inode"])
                != (identity["device"], identity["inode"])):
            raise WorktreeError("published process receipt is not its writer inode")
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        final_identity = _stream_identity(os.fstat(handle.fileno()))
        _revalidate_open_stream(handle, path, final_identity)
    finally:
        if handle is not None:
            handle.close()
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return hashlib.sha256(raw).hexdigest()


def _read_single_link_stream(path: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise WorktreeError(f"stream is not a one-link regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_nlink) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_nlink)):
            raise WorktreeError(f"stream changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _process_start_ticks(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise WorktreeError(f"cannot bind owned child {pid} to /proc start ticks") from exc
    close_paren = raw.rfind(b")")
    fields = raw[close_paren + 1:].split() if close_paren >= 0 else []
    if len(fields) < 20:
        raise WorktreeError(f"cannot parse /proc/{pid}/stat for owned child")
    return int(fields[19])


# =============================================================================
# Path algebra
#
# Every comparison in this section is COMPONENT-WISE over a realpath. Two
# properties fall out, and both are load-bearing:
#   * `/mnt/raid0/llm/llama.cpp-ak-x` is NOT inside `/mnt/raid0/llm/llama.cpp`,
#     though its string starts with it — a campaign worktree must be creatable;
#   * `/mnt/raid0/llm/llama.cpp/../llama.cpp/build` and a symlink pointing into
#     a frozen tree both ARE inside it — a traversal must not be an escape hatch.
# A `str.startswith` test gets the first one wrong in the dangerous direction.
# =============================================================================

#: Sentinel for "the real frozen set", so that an explicit `production_trees=()`
#: stays expressible while OMITTING the argument can no longer mean "unprotected".
_FROZEN_SET: Any = object()


def frozen_tree_paths() -> tuple:
    """Every spelling of a frozen production tree: the trees and their aliases.

    One function so `SandboxPath`, `teardown_worktree` and `build_identity`
    cannot drift apart on what "production" means — a third copy is how one of
    them quietly loses an entry.
    """
    return tuple(PRODUCTION_TREES) + tuple(PRODUCTION_TREE_ALIASES)


def _real(path: Any, label: str) -> str:
    """Absolute, symlink-resolved, normalized. Non-existent tails are fine.

    `os.path.realpath` resolves every component that exists and appends the rest
    lexically, which is exactly right for a destination that does not exist yet:
    a symlinked PARENT is still resolved, so a worktree whose parent link points
    into a frozen tree cannot hide behind the leaf being absent.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise UnsafePath(f"{label}: expected a path, got {path!r}")
    text = os.fspath(path)
    if not text:
        raise UnsafePath(f"{label}: empty path")
    if "\0" in text:
        raise UnsafePath(f"{label}: NUL byte in path")
    if not os.path.isabs(text):
        raise UnsafePath(
            f"{label}: {text!r} must be absolute. A relative path is interpreted "
            "against whatever the process happens to have as its cwd, which is not a "
            "property this module is willing to depend on")
    try:
        resolved = os.path.realpath(text)
    except OSError as exc:  # pragma: no cover - realpath is near-total
        raise UnsafePath(f"{label}: {text!r} could not be resolved ({exc})") from exc
    if not os.path.isabs(resolved):  # pragma: no cover - defensive
        raise UnsafePath(f"{label}: {text!r} resolved to a relative path {resolved!r}")
    return resolved


def _components(path: str) -> tuple:
    """Path components of an already-resolved absolute path."""
    parts = [p for p in path.split(os.sep) if p]
    return tuple(parts)


def _is_within(child: str, parent: str) -> bool:
    """True when resolved `child` IS `parent` or is nested under it."""
    cp = _components(child)
    pp = _components(parent)
    return len(cp) >= len(pp) and cp[:len(pp)] == pp


def _touches_production(resolved: str, trees: Sequence[str]) -> tuple:
    """Frozen trees that `resolved` is inside of, or that are inside `resolved`.

    BOTH directions are refused. Inside-of is the obvious case. Contains-a-tree
    is the one that gets forgotten: `SandboxPath("/mnt/raid0/llm")` would make a
    teardown's `rm -rf`-shaped operation able to reach every frozen tree at once,
    so a sandbox that CONTAINS production is not a sandbox.
    """
    hits = []
    for tree in trees:
        try:
            tree_real = os.path.realpath(tree)
        except OSError:  # pragma: no cover - defensive
            tree_real = tree
        for candidate in {tree, tree_real}:
            if _is_within(resolved, candidate) or _is_within(candidate, resolved):
                hits.append(tree)
                break
    return tuple(hits)


@dataclass(frozen=True)
class SandboxPath:
    """A path that has been PROVEN not to be a frozen production tree.

    This type is the safety property. Every content-mutating entry point in this
    module takes a `SandboxPath`, and a `SandboxPath` naming a frozen tree cannot
    be constructed — so "mutate a production tree" is not a call you can write,
    as opposed to a call that gets refused at runtime by a check somebody might
    add a second, unchecked path around.

    `path` is the REALPATH. Keeping the caller's spelling would defeat the whole
    exercise: the check would run on one string and the git command on another.
    `declared` keeps the original for the receipt, because a receipt that hides
    the fact that the caller wrote a symlink is a receipt that answers the wrong
    question.
    """

    path: str
    declared: str
    sandbox_root: Optional[str]
    production_trees: tuple

    def __post_init__(self) -> None:
        # Re-validate on construction so `dataclasses.replace` and unpickling
        # cannot produce an unchecked instance. `frozen=True` stops mutation; it
        # does not stop construction by another route.
        if not isinstance(self.path, str) or not os.path.isabs(self.path):
            raise UnsafePath(f"SandboxPath.path must be an absolute path, got {self.path!r}")
        if any(part in (".", "..") for part in _components(self.path)):
            raise UnsafePath(
                f"SandboxPath.path {self.path!r} still contains a '.' or '..' component; "
                "it was not resolved")
        if not isinstance(self.production_trees, tuple):
            raise TypeError("SandboxPath.production_trees must be a tuple")
        hits = _touches_production(self.path, self.production_trees)
        if hits:
            raise ProductionTreeViolation(
                f"{self.declared!r} resolves to {self.path!r}, which is inside or contains "
                f"the frozen production tree(s) {list(hits)}. Production kernel trees are "
                "immutable (CLAUDE.md v8 + speech freeze, invariant 3); no worktree, build "
                "directory or commit target may name one")
        if self.sandbox_root is not None:
            if not _is_within(self.path, self.sandbox_root):
                raise UnsafePath(
                    f"{self.declared!r} resolves to {self.path!r}, which is outside the "
                    f"sandbox root {self.sandbox_root!r}")

    # -- construction ------------------------------------------------------
    @classmethod
    def create(cls, path: Any, *, sandbox_root: Any = None,
               production_trees: Any = _FROZEN_SET,
               label: str = "path") -> "SandboxPath":
        """The one constructor. `production_trees` defaults to the frozen set.

        The default is the REAL frozen set, not `()`. It used to be `()`, which
        meant the type whose entire job is "this path is provably not
        production" was unprotected in its default spelling: red-team probe
        2026-08-03 built `SandboxPath.create("/mnt/raid0/llm/llama.cpp")` and got
        an object that `Worktree` accepts and `commit`/`checkout`/`clean` act
        on. A safety type whose safe form is opt-in is a check, and a check the
        next call site forgets.

        It stays a parameter so tests can point a temporary directory at a fake
        "production" tree and exercise the refusal without touching the real one,
        and so a future fourth frozen tree is a constant edit. `production_trees=()`
        remains expressible for a test of the compliant path — but it is now a
        thing someone WROTE, not a thing that happened.
        """
        if production_trees is _FROZEN_SET:
            production_trees = frozen_tree_paths()
        declared = os.fspath(path) if isinstance(path, (str, os.PathLike)) else repr(path)
        resolved = _real(path, label)
        root = None if sandbox_root is None else _real(sandbox_root, f"{label}.sandbox_root")
        return cls(path=resolved, declared=declared, sandbox_root=root,
                   production_trees=tuple(production_trees))

    @classmethod
    def in_sandbox(cls, path: Any, **kwargs: Any) -> "SandboxPath":
        """`create()` with the real frozen-tree set, named explicitly."""
        return cls.create(path, production_trees=frozen_tree_paths(), **kwargs)

    # -- behaviour ---------------------------------------------------------
    def __str__(self) -> str:
        return self.path

    def __fspath__(self) -> str:
        return self.path

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path)

    def child(self, *parts: str) -> "SandboxPath":
        """A descendant, re-validated. Refuses `..` in the supplied parts."""
        for part in parts:
            _req_str(part, "SandboxPath.child part")
            if part in (".", "..") or os.path.isabs(part) or "\0" in part:
                raise UnsafePath(f"SandboxPath.child: unusable component {part!r}")
        joined = os.path.join(self.path, *parts)
        return SandboxPath.create(joined, sandbox_root=self.path,
                                  production_trees=self.production_trees,
                                  label="SandboxPath.child")

    def to_dict(self) -> dict:
        return {"path": self.path, "declared": self.declared,
                "sandbox_root": self.sandbox_root}


# =============================================================================
# Branch names
# =============================================================================

#: Characters git refuses in a ref name (git-check-ref-format(1)), plus the ones
#: a shell would find interesting. `git` is never invoked through a shell here,
#: so the second group is belt-and-braces; the first is correctness.
_BAD_REF_CHARS = set(" ~^:?*[\\\x7f") | {chr(c) for c in range(32)}


@dataclass(frozen=True)
class SafeBranch:
    """A branch name proven to be namespaced `ak/` and not protected.

    Like `SandboxPath`, this is a type and not a check: `GitRepo.delete_branch()`
    and `create_campaign_worktree()` take a `SafeBranch`, so a call that deletes
    or checks out `production-consolidated-v8` cannot be spelled.

    The protected test is a case-folded PREFIX test, not
    `schemas._PRODUCTION_BRANCH_RE`. That regex is right for validating a record
    that already exists; it is too narrow for a guard in front of `branch -D`,
    where `production-consolidated-v9` and `production-anything` must be equally
    undeletable.

    ORDER MATTERS, and one honest caveat. Today the `ak/` namespace requirement
    below already excludes every `production-*` name, so the prefix check is
    DEFENCE IN DEPTH rather than the load-bearing guard — a mutation run proved
    exactly that by emptying `PROTECTED_BRANCH_PREFIXES` and watching nothing
    fail. It is kept, and deliberately runs FIRST, for two reasons: it is what
    still refuses a production branch if the namespace rule is ever relaxed (to
    admit `experimental/…`, say), and running first means the *error message*
    names the real objection instead of "not namespaced under ak/".
    `test_worktree.py` asserts which guard fired, so the redundancy cannot
    silently become an absence.
    """

    name: str

    def __post_init__(self) -> None:
        name = _req_str(self.name, "SafeBranch.name")
        folded = name.casefold()
        for prefix in PROTECTED_BRANCH_PREFIXES:
            if folded.startswith(prefix):
                raise UnsafeBranch(
                    f"{name!r} starts with the protected prefix {prefix!r}. Production and "
                    "release branches are frozen (invariant 3); this module may neither "
                    "check one out, commit to one, nor delete one")
        if not name.startswith("ak/"):
            raise UnsafeBranch(
                f"{name!r} must be namespaced under 'ak/' (design §5.3). The clone is "
                "shared with live sessions and 13 pre-existing worktrees; an un-namespaced "
                "branch is a collision waiting for the next `git worktree add`")
        if name.startswith("-"):
            raise UnsafeBranch(f"{name!r} starts with '-' and would parse as a git option")
        if set(name) & _BAD_REF_CHARS:
            bad = sorted(set(name) & _BAD_REF_CHARS)
            raise UnsafeBranch(f"{name!r} contains ref-illegal character(s) {bad!r}")
        if ".." in name or "@{" in name or "//" in name:
            raise UnsafeBranch(f"{name!r} contains '..', '@{{' or '//' and is not a legal ref")
        if name.endswith((".lock", "/", ".")) or name.startswith("/"):
            raise UnsafeBranch(f"{name!r} is not a legal ref name")
        for component in name.split("/"):
            if not component or component.startswith(".") or component.endswith(".lock"):
                raise UnsafeBranch(f"{name!r} has an illegal path component {component!r}")

    @classmethod
    def for_campaign(cls, campaign_id: str, leaf: str) -> "SafeBranch":
        """`ak/<campaign_id>/<leaf>` — the namespace design §5.3 specifies."""
        validate_campaign_id(campaign_id)
        leaf = _req_str(leaf, "leaf")
        if "/" in leaf:
            raise UnsafeBranch(
                f"leaf {leaf!r} must not contain '/'; pass the components you want or build "
                "the name yourself, so the namespace depth is never accidental")
        return cls(name=f"ak/{campaign_id}/{leaf}")

    def __str__(self) -> str:
        return self.name


def validate_campaign_id(campaign_id: Any) -> str:
    """A campaign id must start with `ak-`; the worktree namespace depends on it.

    `<source_tree>-<campaign_id>` is the directory name, so `ak-` here is what
    makes `llama.cpp-ak-…` true by construction rather than by convention.
    """
    if not isinstance(campaign_id, str) or not _CAMPAIGN_ID_RE.match(campaign_id):
        raise UnsafeBranch(
            f"campaign_id {campaign_id!r} must match {_CAMPAIGN_ID_RE.pattern!r} — it must "
            "start with 'ak-', because the worktree directory name is derived from it and "
            "that is what produces the 'llama.cpp-ak-…' namespace")
    return campaign_id


# =============================================================================
# Pathspecs
# =============================================================================

_PATHSPEC_WILDCARDS = set("*?[]")


@dataclass(frozen=True)
class Pathspec:
    """A non-empty tuple of LITERAL relative paths inside one worktree.

    Design §8.5: *"pathspec-limited commits"*. The scar behind it: the clone at
    `/mnt/raid0/llm/<name>` and `/workspace/repos/<name>` is ONE clone shared
    with live sessions, and a wholesale `git add` in a shared tree sweeps in
    whatever another session had staged. So every commit here names its paths.

    Literal only — no `.`, no `*`, no `:(glob)`, no `:!exclude`. A glob is a
    promise about a set whose membership is decided later by the filesystem, and
    "later" is when the other session's file appears. Magic pathspecs are refused
    for the same reason plus a sharper one: `:/` re-roots to the repository top,
    which is precisely the containment this type exists to provide.
    """

    paths: tuple
    worktree_root: str

    def __post_init__(self) -> None:
        if not isinstance(self.paths, tuple) or not self.paths:
            raise UnsafePathspec(
                "a pathspec-limited commit needs at least one path. An empty pathspec is "
                "how `git commit --` becomes `git commit -a` by accident")
        root = _req_str(self.worktree_root, "Pathspec.worktree_root")
        seen = set()
        for raw in self.paths:
            rel = _req_str(raw, "pathspec entry")
            if rel in seen:
                raise UnsafePathspec(f"duplicate pathspec entry {rel!r}")
            seen.add(rel)
            if rel.startswith(":"):
                raise UnsafePathspec(
                    f"{rel!r} is a magic pathspec. `:/` re-roots to the repository top and "
                    "`:(exclude)` inverts the set; both defeat pathspec limiting")
            if rel.startswith("-"):
                raise UnsafePathspec(f"{rel!r} starts with '-' and would parse as an option")
            if os.path.isabs(rel):
                raise UnsafePathspec(f"{rel!r} must be relative to the worktree root")
            if "\0" in rel:
                raise UnsafePathspec(f"{rel!r} contains a NUL byte")
            if set(rel) & _PATHSPEC_WILDCARDS:
                raise UnsafePathspec(
                    f"{rel!r} contains a wildcard. A pathspec-limited commit names its "
                    "paths; a glob names a set the filesystem decides at commit time")
            parts = [p for p in rel.split("/") if p]
            if not parts or any(p in (".", "..") for p in parts):
                raise UnsafePathspec(
                    f"{rel!r} contains a '.' or '..' component, or is empty; a traversal is "
                    "how a pathspec leaves the worktree it was limited to")
            resolved = os.path.realpath(os.path.join(root, rel))
            if not _is_within(resolved, root):
                raise UnsafePathspec(
                    f"{rel!r} resolves to {resolved!r}, outside the worktree {root!r}")

    @classmethod
    def create(cls, paths: Iterable[Any], worktree: "Worktree") -> "Pathspec":
        return cls(paths=tuple(paths), worktree_root=worktree.path.path)

    def as_args(self) -> tuple:
        return tuple(self.paths)


# =============================================================================
# Owned-process execution
#
# Everything in this module that runs a program runs it here. One entry point
# means one place where the process-discipline rules are enforced, and one AST
# shape for `audit_no_name_pattern_process_ops` to check.
# =============================================================================

@dataclass(frozen=True)
class ProcessDisposition:
    """What happened to a process THIS module launched.

    `verified_dead` is derived from a successful reap, not from "we sent a
    signal". CLAUDE.md: *"After killing a process, verify it is dead; escalate
    SIGTERM → SIGKILL; never report success until confirmed."*
    """

    argv: tuple
    pid: int
    pgid: int
    exit_code: Optional[int]
    timed_out: bool
    signals_sent: tuple
    verified_dead: bool
    duration_s: float
    started_at: str
    sandbox_receipt: Optional[Mapping[str, Any]] = None
    sandbox_teardown: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict:
        return {"argv": list(self.argv), "pid": self.pid, "pgid": self.pgid,
                "exit_code": self.exit_code, "timed_out": self.timed_out,
                "signals_sent": list(self.signals_sent),
                "verified_dead": self.verified_dead,
                "duration_s": round(self.duration_s, 6), "started_at": self.started_at,
                "sandbox_receipt": (dict(self.sandbox_receipt)
                                    if self.sandbox_receipt is not None else None),
                "sandbox_teardown": (dict(self.sandbox_teardown)
                                     if self.sandbox_teardown is not None else None)}


def _validate_argv(argv: Sequence[str]) -> tuple:
    """Refuse an argv that is empty, non-string, or names a name-pattern process tool.

    EVERY element is checked, not just `argv[0]`. Checking only the program was
    a guard reading the wrong input: this module's own `BuildPlan._prefix()`
    emits `taskset -c <list> <cmake>`, so `argv[0]` is `taskset` and the program
    that actually executes is at index 3. Red-team probe 2026-08-03 built
    `('taskset', '-c', '0-1', 'pkill', …)` and `_validate_argv` accepted it while
    refusing the identical plan without the `taskset` prefix. Any wrapper —
    `taskset`, `env`, `nice`, `timeout`, `numactl` — has the same shape.

    A `-D…=fuser`-style define is one element and its basename is the whole
    element, so the widened check does not fire on cmake arguments; the cost of
    the remaining false-positive space is a loud refusal, never a silent pass.
    """
    if not argv:
        raise WorktreeError("empty argv")
    out = []
    for item in argv:
        if not isinstance(item, str):
            raise WorktreeError(f"argv entries must be strings, got {item!r}")
        if "\0" in item:
            raise WorktreeError("NUL byte in argv")
        out.append(item)
    for index, item in enumerate(out):
        program = os.path.basename(item)
        if program in _NAME_PATTERN_BINARIES:
            raise WorktreeError(
                f"argv[{index}] is {item!r}: {program!r} selects processes by NAME PATTERN "
                "and is refused. INC-20260731: a name-pattern kill took out another "
                "session's llama-server twice and killed earlyoom, whose own argv contains "
                "the names it guards. Signal only a PID this module captured")
    return tuple(out)


def _sanitized_env(env: Optional[Mapping[str, str]]) -> dict:
    """`env` (or the inherited environment) with every git RE-TARGETING variable removed.

    `git -C <sandbox>` decides nothing when `GIT_DIR` is set: git skips
    discovery and uses the variable. So a `Worktree` whose path is a proven
    `SandboxPath` will happily `add`/`commit` into a frozen production tree if
    that variable is in the ambient environment — demonstrated, not theorised.
    Stripping it here rather than in `_git` means the same protection covers the
    BUILD, whose `-- ggml commit:` line is read out of git and would otherwise
    report the redirected repository's commit into the receipt.
    """
    base = dict(os.environ if env is None else env)
    for name in list(base):
        if name in _GIT_ENV_REDIRECTS or name.startswith(_GIT_ENV_REDIRECT_PREFIXES):
            base.pop(name, None)
    return base


def _require_config_is_a_read(args: Sequence[str]) -> None:
    """Refuse every `git config` form that writes.

    A linked worktree shares the clone's `.git`, so `git config` run with `-C`
    pointing at a campaign worktree edits the config of the FROZEN clone that
    worktree was cut from. The allowlist is of read flags; a bare
    `git config <name> <value>` (two non-flag operands) is a write and is
    refused by the operand count, which is the form that has no flag to allow.
    """
    operands = [a for a in args if not a.startswith("-")]
    for flag in args:
        if flag.startswith("-") and flag.split("=", 1)[0] not in _CONFIG_READ_ONLY_FLAGS:
            raise ProductionTreeViolation(
                f"`git config {flag}` is not a read. In a linked worktree `git config` "
                "writes the SHARED .git/config of the clone the worktree was cut from — "
                "for a campaign worktree that is the frozen production clone")
    reading = any(f.split("=", 1)[0] in ("--get", "--get-all", "--get-regexp",
                                         "--get-urlmatch", "--get-color",
                                         "--get-colorbool", "--list", "-l")
                  for f in args)
    if not reading or len(operands) > 1:
        raise ProductionTreeViolation(
            "`git config` is available for reads only (--get/--get-all/--get-regexp/"
            f"--list with at most one operand); got {list(args)!r}, which sets a value "
            "in the clone's shared config")


def _terminate_owned(proc: "subprocess.Popen", pgid: int, *,
                     grace_s: float) -> tuple:
    """TERM → grace → KILL against a process group WE created. Returns signals sent.

    `start_new_session=True` made the child a session leader, so `pgid == pid` —
    the group is exactly this child and its descendants, and killing it cannot
    reach anything we did not launch. That is the whole reason for the new
    session: a build forks a compiler tree, and killing only the top PID leaves
    the compilers running on a shared box.
    """
    sent = []
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, grace_s)):
        if proc.poll() is not None:
            break
        try:
            os.killpg(pgid, sig)
            sent.append(sig.name)
        except ProcessLookupError:
            break
        except PermissionError as exc:  # pragma: no cover - would mean pgid reuse
            raise ProcessEscalationFailed(
                f"not permitted to signal process group {pgid}: {exc}") from exc
        try:
            proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue
    return tuple(sent)


def _run_owned(argv: Sequence[str], *, cwd: Optional[str] = None,
               timeout_s: float = 300.0, env: Optional[Mapping[str, str]] = None,
               stdout_path: Optional[str] = None,
               kill_grace_s: float = 10.0,
               sandbox_policy: Optional[process_sandbox.SandboxPolicy] = None,
               sandbox_receipt_path: Optional[str] = None,
               process_receipt_prefix: Optional[str] = None) -> tuple:
    """Run `argv` as an owned session leader. Returns `(disposition, output_text)`.

    Never `shell=True`: a shell would reintroduce word-splitting, globbing and
    `$(...)` over strings this module spent three value types making safe.
    Never a pipe into another process either — `feedback_pipe_hazards` in one
    line, and for a build the exit code must be the compiler's, not a filter's.

    When `stdout_path` is given the child writes there directly, so the log
    survives a timeout; the text is read back afterwards. Otherwise output is
    captured through a pipe.
    """
    argv = _validate_argv(argv)
    if (sandbox_policy is None) != (sandbox_receipt_path is None):
        raise WorktreeError(
            "sandbox_policy and sandbox_receipt_path must be supplied together")
    if sandbox_policy is not None and not isinstance(
            sandbox_policy, process_sandbox.SandboxPolicy):
        raise TypeError("sandbox_policy must be a SandboxPolicy")
    spawn_argv: Sequence[str] = argv
    executed_env = _sanitized_env(env)
    if sandbox_policy is not None:
        spawn_argv = sandbox_policy.wrap(argv, receipt_path=sandbox_receipt_path)
        executed_env["PYTHONDONTWRITEBYTECODE"] = "1"
    epoch_token = None
    intent_path = None
    if process_receipt_prefix is not None:
        epoch_token = secrets.token_hex(32)
        executed_env["AUTOKERNEL_OWNED_PROCESS_EPOCH"] = epoch_token
        intent_path = process_receipt_prefix + "-intent.json"
        _sealed_process_receipt(intent_path, {
            "schema": "epyc.autokernel.owned_process_intent.v1",
            "argv": list(argv), "epoch_token": epoch_token,
            "stdout_path": stdout_path,
            "sandbox_receipt_path": sandbox_receipt_path,
            "sandbox_policy_sha256": (sandbox_policy.policy_sha256
                                      if sandbox_policy is not None else None),
            "sandbox_token": (sandbox_policy.token
                              if sandbox_policy is not None else None),
            "cgroup_root": (sandbox_policy.cgroup_root
                            if sandbox_policy is not None else None),
        })
    started = time.monotonic()
    started_at = _utc_now_iso()
    sink = None
    if stdout_path is not None:
        sink = _exclusive_regular_sink(stdout_path)
        stdout_target: Any = sink
    else:
        stdout_target = subprocess.PIPE
    try:
        with subprocess.Popen(
                spawn_argv, cwd=cwd, env=executed_env,
                stdout=stdout_target, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                close_fds=True) as proc:
            pid = proc.pid
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:  # pragma: no cover - child already reaped
                pgid = pid
            if process_receipt_prefix is not None:
                try:
                    _sealed_process_receipt(process_receipt_prefix + "-start.json", {
                        "schema": "epyc.autokernel.owned_process_start.v1",
                        "intent_receipt_sha256": _sha256_file(intent_path),
                        "epoch_token": epoch_token,
                        "argv": list(argv), "pid": pid, "pgid": pgid,
                        "process_start_ticks": _process_start_ticks(pid),
                        "started_at": started_at,
                        "stdout_path": stdout_path,
                        "sandbox_receipt_path": sandbox_receipt_path,
                    })
                except BaseException:
                    _terminate_owned(proc, pgid, grace_s=kill_grace_s)
                    proc.wait()
                    if sandbox_policy is not None:
                        cgroup_path = sandbox_policy.cgroup_path(pid)
                        if cgroup_path.exists():
                            process_sandbox.cleanup_cgroup(sandbox_policy, pid)
                    raise
            signals_sent: tuple = ()
            timed_out = False
            captured = b""
            try:
                captured, _ = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                signals_sent = _terminate_owned(proc, pgid, grace_s=kill_grace_s)
                try:
                    captured, _ = proc.communicate(timeout=kill_grace_s)
                except subprocess.TimeoutExpired:  # pragma: no cover - post-SIGKILL
                    captured = b""
            exit_code = proc.poll()
            verified_dead = exit_code is not None

        stream_raw: bytes | None = None
        stream_identity: dict[str, int] | None = None
        if stdout_path is not None:
            assert sink is not None
            stream_raw, stream_identity = _read_and_revalidate_open_stream(
                sink, stdout_path)
            text = stream_raw.decode("utf-8", "replace")
        else:
            text = (captured or b"").decode("utf-8", "replace")

        if timed_out and not verified_dead:
            raise ProcessEscalationFailed(
                f"pid {pid} (pgid {pgid}) survived {list(signals_sent)}; running "
                f"{' '.join(argv)}. Not reporting a clean teardown for a process still alive")

        sandbox_receipt = None
        sandbox_teardown = None
        if sandbox_policy is not None:
            try:
                sandbox_receipt = process_sandbox.read_receipt(sandbox_receipt_path)
                process_sandbox.verify_receipt(
                    sandbox_receipt, policy=sandbox_policy, pid=pid, argv=argv)
                sandbox_teardown = process_sandbox.cleanup_cgroup(sandbox_policy, pid)
            except process_sandbox.SandboxError as exc:
                cleanup_note = ""
                cgroup_path = sandbox_policy.cgroup_path(pid)
                if cgroup_path.exists():
                    try:
                        process_sandbox.cleanup_cgroup(sandbox_policy, pid)
                        cleanup_note = "; the owned cgroup was drained after refusal"
                    except process_sandbox.SandboxError as cleanup_exc:
                        cleanup_note = (
                            "; additionally, owned-cgroup cleanup failed: "
                            f"{cleanup_exc}")
                raise WorktreeError(
                    "candidate build containment did not produce a verified activation "
                    f"receipt and teardown: {exc}{cleanup_note}") from exc

        disposition = ProcessDisposition(
            argv=argv, pid=pid, pgid=pgid, exit_code=exit_code, timed_out=timed_out,
            signals_sent=signals_sent, verified_dead=verified_dead,
            duration_s=time.monotonic() - started, started_at=started_at,
            sandbox_receipt=sandbox_receipt, sandbox_teardown=sandbox_teardown)
        if process_receipt_prefix is not None:
            _sealed_process_receipt(process_receipt_prefix + "-terminal.json", {
                "schema": "epyc.autokernel.owned_process_terminal.v2",
                "start_receipt_sha256": _sha256_file(
                    process_receipt_prefix + "-start.json"),
                "disposition": disposition.to_dict(),
                "stdout_path": stdout_path,
                "stdout_sha256": (hashlib.sha256(stream_raw).hexdigest()
                                   if stream_raw is not None else None),
                "stdout_identity": stream_identity,
            })
            if stdout_path is not None:
                assert sink is not None and stream_identity is not None
                _revalidate_open_stream(sink, stdout_path, stream_identity)
        return disposition, text
    finally:
        if sink is not None:
            sink.close()


def _run_guarded_patch_input(worktree: "Worktree", patch_bytes: bytes, *,
                             check_only: bool) -> tuple:
    """Run the one command allowed to receive child stdin in this module.

    Every other subprocess launched here keeps ``stdin=DEVNULL`` in
    :func:`_run_owned`.  Patch bytes are already resident and content-addressed;
    they are supplied directly to ``git apply -`` so no mutable patch path can
    change between validation and application.  The argv is closed here rather
    than caller-supplied, and the receiver must be the mutation-capable
    :class:`Worktree` type.
    """
    if not isinstance(worktree, Worktree):
        raise TypeError("guarded patch input requires a Worktree")
    if not isinstance(patch_bytes, bytes) or not patch_bytes:
        raise TypeError("patch_bytes must be non-empty immutable bytes")
    args = ["git", "-C", worktree.path.path, "apply"]
    if check_only:
        args.append("--check")
    args += ["--whitespace=error-all", "--", "-"]
    argv = _validate_argv(args)
    started = time.monotonic()
    started_at = _utc_now_iso()
    with subprocess.Popen(
            argv, env=_sanitized_env(None), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
            start_new_session=True, close_fds=True) as proc:
        pid = proc.pid
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:  # pragma: no cover - child already reaped
            pgid = pid
        signals_sent: tuple = ()
        timed_out = False
        try:
            captured, _ = proc.communicate(input=patch_bytes, timeout=worktree.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            signals_sent = _terminate_owned(proc, pgid, grace_s=10.0)
            captured, _ = proc.communicate()
        exit_code = proc.poll()
    disposition = ProcessDisposition(
        argv=argv, pid=pid, pgid=pgid, exit_code=exit_code,
        timed_out=timed_out, signals_sent=signals_sent,
        verified_dead=exit_code is not None,
        duration_s=time.monotonic() - started, started_at=started_at)
    return disposition, (captured or b"").decode("utf-8", "replace")


# =============================================================================
# The git plane — two classes, one boundary
# =============================================================================

class GitRepo:
    """A repository this module may READ and may attach a worktree to.

    **May name a frozen production tree**, because that is how anchoring works:
    resolving `production-consolidated-v8`'s tip and checking that commit out
    into a separate directory both require addressing the frozen clone. What
    keeps that safe is not a check but the verb allowlist below — every verb
    that writes a working tree, index or object is absent, so no method here can
    modify what it addresses.

    `git worktree add` does write: it creates `.git/worktrees/<name>/` in the
    shared clone. That is administrative metadata, not content — the production
    working tree, its checked-out branch and its index are untouched, and
    `create_campaign_worktree()` proves exactly that with a before/after
    `TreeFingerprint`. There is no way to add a worktree to this repository
    without writing that metadata: linked worktrees share one `.git`, so
    pointing `-C` at `llama.cpp-experimental` writes the same directory.
    """

    #: Read-only plumbing/porcelain plus worktree administration. Deliberately a
    #: closed set: an unknown verb is refused rather than passed through, so
    #: adding a capability to this class is a visible edit here.
    ALLOWED_VERBS: ClassVar[frozenset] = frozenset({
        "rev-parse", "status", "symbolic-ref", "show-ref", "for-each-ref",
        "cat-file", "ls-files", "ls-tree", "merge-base", "rev-list", "log",
        "diff", "diff-tree", "config", "version", "worktree", "branch",
        "describe", "count-objects",
    })

    #: `branch` is in the allowlist because listing branches is a read. The
    #: MUTATING branch operation is `delete_branch()`, which takes a `SafeBranch`
    #: and builds its own argv; the generic path refuses the mutating flags.
    _BRANCH_MUTATING_FLAGS: ClassVar[frozenset] = frozenset({
        "-d", "-D", "--delete", "-m", "-M", "--move", "-c", "-C", "--copy",
        "-f", "--force", "--set-upstream-to", "-u", "--unset-upstream", "--edit-description",
    })

    _WORKTREE_MUTATING_SUBCOMMANDS: ClassVar[frozenset] = frozenset({
        "add", "remove", "prune", "move", "repair", "lock", "unlock",
    })

    def __init__(self, path: Any, *, timeout_s: float = 120.0):
        self.path = _real(path, "GitRepo.path")
        self.timeout_s = float(timeout_s)
        if not os.path.isdir(self.path):
            raise UnsafePath(f"GitRepo.path {self.path!r} is not a directory")
        git_marker = os.path.join(self.path, ".git")
        if not os.path.exists(git_marker):
            raise UnsafePath(
                f"GitRepo.path {self.path!r} has no .git; refusing to run git commands "
                "against a directory that is not a work tree")

    # -- the one gate every git call passes through ------------------------
    def _git(self, *args: str, timeout_s: Optional[float] = None) -> str:
        """Run `git -C <repo> <args>`. Refuses any verb outside `ALLOWED_VERBS`.

        The refusal is not defence in depth for its own sake: it is what makes
        the claim "no `GitRepo` method mutates content" checkable by a test that
        does not have to read every method.
        """
        if not args:
            raise WorktreeError("GitRepo._git needs a verb")
        verb = args[0]
        if verb in CONTENT_MUTATING_VERBS or verb not in self.ALLOWED_VERBS:
            raise ProductionTreeViolation(
                f"git verb {verb!r} is not available on GitRepo. This class may address a "
                "FROZEN production tree, so it carries no content-mutating capability; use "
                "a Worktree, which cannot be constructed for a production path")
        if verb == "branch":
            for flag in args[1:]:
                # `--set-upstream-to=X` is the same flag as `--set-upstream-to X`;
                # an exact-match test sees two different strings and passes one.
                if flag.split("=", 1)[0] in self._BRANCH_MUTATING_FLAGS:
                    raise ProductionTreeViolation(
                        f"`git branch {flag}` mutates refs and is not available on the "
                        "generic path; use GitRepo.delete_branch(SafeBranch)")
        if verb == "worktree":
            # `worktree add` WRITES A FULL CHECKOUT wherever it is pointed —
            # red-team probe 2026-08-03 used this path to materialise a working
            # tree INSIDE a frozen clone, past every SandboxPath in the module,
            # because `_WORKTREE_MUTATING_SUBCOMMANDS` was defined and never
            # read. The guarded entry points build their own argv (as
            # `delete_branch` already did) and go through `_git_admin`.
            for sub in args[1:]:
                if sub in self._WORKTREE_MUTATING_SUBCOMMANDS:
                    raise ProductionTreeViolation(
                        f"`git worktree {sub}` writes a working tree or the clone's worktree "
                        "metadata and is not available on the generic path. Use "
                        "GitRepo.add_worktree(SandboxPath, …) or "
                        "GitRepo.remove_worktree(SandboxPath), whose destination is typed")
        if verb == "config":
            # For a LINKED worktree `git config` writes the SHARED `.git/config`
            # of the clone it hangs off — which, for a campaign worktree, is the
            # frozen production clone. Reads only, on both classes.
            _require_config_is_a_read(args[1:])
        return self._git_admin(*args, timeout_s=timeout_s)

    def _git_admin(self, *args: str, timeout_s: Optional[float] = None) -> str:
        """Run `git -C <repo> <args>` with the argv already decided by a guarded method.

        Reachable only from `_git` (after its refusals) and from the two worktree
        administration methods, which take a `SandboxPath` and construct their
        own argv. Keeping it separate is what lets `_git` refuse
        `worktree add` outright instead of trying to tell a guarded call from an
        arbitrary one by inspecting strings.

        It still refuses `CONTENT_MUTATING_VERBS` and anything outside
        `ALLOWED_VERBS`: an escape hatch added to carry two guarded calls must
        not quietly become a second, unguarded `_git`. The ONLY thing it relaxes
        is the subcommand check its callers have already satisfied by
        construction.
        """
        verb = args[0] if args else ""
        if verb in CONTENT_MUTATING_VERBS or verb not in self.ALLOWED_VERBS:
            raise ProductionTreeViolation(
                f"git verb {verb!r} is not available on GitRepo, on the admin path either. "
                "This class may address a FROZEN production tree")
        argv = ("git", "-C", self.path) + tuple(args)
        disposition, text = _run_owned(
            argv, timeout_s=self.timeout_s if timeout_s is None else timeout_s)
        if disposition.exit_code != 0:
            raise GitCommandFailed(argv, disposition.exit_code or -1, text)
        return text

    # -- reads -------------------------------------------------------------
    def head_commit(self) -> str:
        return _req_commit(self._git("rev-parse", "HEAD").strip(), "HEAD")

    def branch_tip(self, branch: str) -> str:
        """Resolve `refs/heads/<branch>` — the tip, not whatever HEAD happens to be."""
        _req_str(branch, "branch")
        if branch.startswith("-"):
            raise UnsafeBranch(f"branch {branch!r} would parse as a git option")
        out = self._git("rev-parse", "--verify", "--end-of-options",
                        f"refs/heads/{branch}").strip()
        return _req_commit(out, f"refs/heads/{branch}")

    def commit_parents(self, commit: str) -> tuple:
        """Return the ordered parent commits of one immutable commit object."""
        commit = _req_commit(commit, "commit")
        fields = self._git("rev-list", "--parents", "-n", "1", commit).split()
        if not fields or fields[0] != commit:
            raise WorktreeError(
                f"git rev-list did not return the requested commit {commit!r}: {fields!r}")
        return tuple(_req_commit(parent, "parent commit") for parent in fields[1:])

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Prove a commit-graph relationship without mutating either tree."""
        ancestor = _req_commit(ancestor, "ancestor")
        descendant = _req_commit(descendant, "descendant")
        try:
            self._git("merge-base", "--is-ancestor", ancestor, descendant)
        except GitCommandFailed as exc:
            if exc.returncode == 1:
                return False
            raise
        return True

    def current_branch(self) -> Optional[str]:
        try:
            return self._git("symbolic-ref", "--short", "--quiet", "HEAD").strip() or None
        except GitCommandFailed:
            return None  # detached HEAD

    def status_porcelain(self) -> str:
        return self._git("status", "--porcelain")

    def worktree_paths(self) -> tuple:
        out = self._git("worktree", "list", "--porcelain")
        return tuple(line.split(" ", 1)[1].strip()
                     for line in out.splitlines() if line.startswith("worktree "))

    def branch_exists(self, branch: Any) -> bool:
        name = branch.name if isinstance(branch, SafeBranch) else _req_str(branch, "branch")
        try:
            self._git("show-ref", "--verify", "--quiet", f"refs/heads/{name}")
        except GitCommandFailed:
            return False
        return True

    # -- worktree administration (metadata-only on the addressed repo) -----
    def add_worktree(self, dest: SandboxPath, commit: str, *,
                     branch: Optional[SafeBranch] = None,
                     detach: bool = False, timeout_s: float = 900.0) -> "Worktree":
        """Check `commit` out into `dest`. `dest` is a `SandboxPath` — that is the guard.

        A production path cannot reach this method because it cannot become a
        `SandboxPath`; a production branch cannot reach it because it cannot
        become a `SafeBranch`.
        """
        if not isinstance(dest, SandboxPath):
            raise TypeError(
                "add_worktree(dest=...) requires a SandboxPath. A str would let a frozen "
                "production path through, which is the one thing this signature prevents")
        if branch is not None and not isinstance(branch, SafeBranch):
            raise TypeError("add_worktree(branch=...) requires a SafeBranch")
        if branch is None and not detach:
            raise WorktreeError(
                "add_worktree needs either a SafeBranch or detach=True. A worktree on "
                "whatever branch git picks is how a campaign ends up on another session's "
                "branch in a shared clone")
        _req_commit(commit, "commit")
        if dest.exists and os.listdir(dest.path):
            raise UnsafePath(
                f"{dest.path!r} already exists and is not empty; refusing to reuse a "
                "directory whose contents this module did not create")
        args = ["worktree", "add"]
        if detach:
            args.append("--detach")
        if branch is not None:
            args += ["-b", branch.name]
        args += [dest.path, commit]
        self._git_admin(*args, timeout_s=timeout_s)
        return Worktree(dest, repo=self, branch=branch, source_commit=commit)

    def remove_worktree(self, dest: SandboxPath, *, force: bool = False,
                        timeout_s: float = 300.0) -> None:
        if not isinstance(dest, SandboxPath):
            raise TypeError("remove_worktree(dest=...) requires a SandboxPath")
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(dest.path)
        self._git_admin(*args, timeout_s=timeout_s)
        self._git_admin("worktree", "prune")

    def delete_branch(self, branch: SafeBranch, *, force: bool = True) -> None:
        """Delete a campaign branch. `SafeBranch` is why this cannot delete production."""
        if not isinstance(branch, SafeBranch):
            raise TypeError(
                "delete_branch requires a SafeBranch. A plain string is how "
                "'production-consolidated-v8' becomes deletable")
        argv = ("git", "-C", self.path, "branch", "-D" if force else "-d", "--",
                branch.name)
        disposition, text = _run_owned(argv, timeout_s=self.timeout_s)
        if disposition.exit_code != 0:
            raise GitCommandFailed(argv, disposition.exit_code or -1, text)


class Worktree:
    """An EXPERIMENTAL worktree. Carries every content-mutating capability.

    Its constructor takes a `SandboxPath`, so it cannot name a frozen production
    tree — that is the other half of the structural property. Together with
    `GitRepo` having no mutating verb, "mutate production" has no type to travel
    in.
    """

    #: Worktree-LOCAL mutations only. `stash` is deliberately absent though it
    #: would be convenient for a candidate repair: a linked worktree shares the
    #: clone's ref namespace and object store, so `git stash` in
    #: `llama.cpp-ak-…` writes `refs/stash` in the FROZEN clone, where another
    #: session's `git stash list` would find it — and `TreeFingerprint` (HEAD,
    #: symbolic-ref, `status --porcelain`) cannot see a ref that is not HEAD, so
    #: the immutability proof would hold while the shared clone had changed.
    #: `commit_paths()` covers the same need without leaving the worktree.
    ALLOWED_VERBS: ClassVar[frozenset] = frozenset({
        "add", "commit", "status", "rev-parse", "diff", "ls-files", "checkout",
        "restore", "clean", "symbolic-ref", "show-ref", "apply",
        "config", "log", "merge-base",
    })

    def __init__(self, path: SandboxPath, *, repo: GitRepo,
                 branch: Optional[SafeBranch], source_commit: str,
                 timeout_s: float = 300.0):
        if not isinstance(path, SandboxPath):
            raise TypeError(
                "Worktree(path=...) requires a SandboxPath. This class can commit, "
                "checkout and clean; letting a bare string name its tree would make a "
                "frozen production tree reachable by every one of those")
        if not isinstance(repo, GitRepo):
            raise TypeError("Worktree(repo=...) requires a GitRepo")
        if branch is not None and not isinstance(branch, SafeBranch):
            raise TypeError("Worktree(branch=...) requires a SafeBranch or None")
        self.path = path
        self.repo = repo
        self.branch = branch
        self.source_commit = _req_commit(source_commit, "Worktree.source_commit")
        self.timeout_s = float(timeout_s)

    def _git(self, *args: str, timeout_s: Optional[float] = None) -> str:
        if not args:
            raise WorktreeError("Worktree._git needs a verb")
        if args[0] not in self.ALLOWED_VERBS:
            raise WorktreeError(f"git verb {args[0]!r} is not available on Worktree")
        if args[0] == "config":
            # A campaign worktree is a LINKED worktree: its `.git` is a pointer
            # into the frozen clone, so a config write here lands in the frozen
            # clone's shared config. The SandboxPath is true and does not help.
            _require_config_is_a_read(args[1:])
        argv = ("git", "-C", self.path.path) + tuple(args)
        disposition, text = _run_owned(
            argv, timeout_s=self.timeout_s if timeout_s is None else timeout_s)
        if disposition.exit_code != 0:
            raise GitCommandFailed(argv, disposition.exit_code or -1, text)
        return text

    # -- reads -------------------------------------------------------------
    def head_commit(self) -> str:
        return _req_commit(self._git("rev-parse", "HEAD").strip(), "HEAD")

    def is_ancestor(self, ancestor: str, descendant: Optional[str] = None) -> bool:
        """Read-only ancestry proof scoped to this experimental worktree."""
        return self.repo.is_ancestor(ancestor, descendant or self.head_commit())

    def status_porcelain(self) -> str:
        return self._git("status", "--porcelain")

    def is_clean(self) -> bool:
        """Clean means no tracked modification AND no untracked file.

        `--porcelain` with the default `-unormal` reports both. A "clean" that
        ignored untracked files would let an uncommitted source file into the
        snapshot digest while the record said `clean: true` — and the snapshot
        digest is what §8.5.1 (2) compares the build against.
        """
        return self.status_porcelain().strip() == ""

    def unified_diff_from_source(self) -> str:
        """Committed candidate delta from the immutable worktree source commit."""
        return self._git(
            "diff", "--no-ext-diff", "--unified=3",
            f"{self.source_commit}..{self.head_commit()}", "--")

    def apply_patch_bytes(self, patch_bytes: bytes) -> dict:
        """Check then apply immutable patch bytes through the guarded stdin route.

        A ``GitRepo`` can never reach this method.  Both invocations consume the
        same in-memory ``bytes`` object; there is no pathname to swap between
        the check and the mutation.
        """
        checked, check_text = _run_guarded_patch_input(
            self, patch_bytes, check_only=True)
        if checked.exit_code != 0:
            raise GitCommandFailed(checked.argv, checked.exit_code or -1, check_text)
        applied, apply_text = _run_guarded_patch_input(
            self, patch_bytes, check_only=False)
        if applied.exit_code != 0:
            raise GitCommandFailed(applied.argv, applied.exit_code or -1, apply_text)
        return {
            "check": checked.to_dict(), "apply": applied.to_dict(),
            "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
        }

    def commit_argv_for_paths(self, paths: Iterable[Any], message: str) -> tuple:
        """The exact pathspec-limited commit argv used by ``commit_paths``."""
        spec = Pathspec.create(paths, self)
        _req_str(message, "message")
        return ("git", "-C", self.path.path, "commit", "-m", message,
                "--", *spec.as_args())

    # -- the pathspec-limited commit --------------------------------------
    def commit_paths(self, paths: Iterable[Any], message: str, *,
                     author_date: Optional[str] = None,
                     allow_empty: bool = False) -> Optional[str]:
        """Stage and commit EXACTLY `paths`. Returns the new commit, or None if nothing changed.

        Two pathspec-limited commands, not one:

        * ``git add -- <paths>`` — needed because `git commit -- <paths>` will
          not pick up a file git has never seen, and a new kernel file is the
          normal case for a candidate;
        * ``git commit -m <msg> -- <paths>`` — the pathspec form deliberately
          IGNORES the index for everything else, so anything another session
          staged in this shared clone cannot ride along. That is the whole point
          and it is why the second command repeats the paths rather than
          trusting the `add`.

        `--` separates paths from options in both, so a file called `-f` is a
        file.
        """
        spec = Pathspec.create(paths, self)
        _req_str(message, "message")
        self._git("add", "--", *spec.as_args())
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        if author_date is not None:
            args += ["--date", _req_str(author_date, "author_date")]
        args += ["--"] + list(spec.as_args())
        try:
            self._git(*args)
        except GitCommandFailed as exc:
            lowered = exc.output.lower()
            if "nothing to commit" in lowered or "no changes added" in lowered:
                return None
            raise
        return self.head_commit()

    def snapshot_digest(self, *, exclude_git: bool = True) -> integrity.TreeDigest:
        """Content-address this worktree with `integrity.hash_source_tree`.

        `.git` is excluded by default and the exclusion is stated, not implied:
        `hash_source_tree` deliberately excludes NOTHING by default because *"a
        default `.git` exclusion would be a silent behaviour with a
        security-shaped consequence"*. Here the exclusion is the correct one and
        is recorded in the receipt — the snapshot is the SOURCE closure, and
        `.git` contains loose objects and index state that differ between two
        trees holding identical source.
        """
        excludes = (".git",) if exclude_git else ()
        return integrity.hash_source_tree(self.path.path, exclude_dir_names=excludes)

    def to_record(self) -> dict:
        """The `worktree` block `schemas.validate_candidate` expects."""
        return {"path": self.path.path,
                "branch": self.branch.name if self.branch else None,
                "source_commit": self.source_commit,
                "clean": self.is_clean()}


# =============================================================================
# Anchoring — "the CURRENT production tip", made checkable
# =============================================================================

@dataclass(frozen=True)
class Anchor:
    """A production branch tip, resolved at a moment, with the tree's fingerprint.

    The fingerprint rides along so a later teardown can prove the tree it
    anchored to is unchanged without having to trust that somebody remembered to
    take a `before` reading.
    """

    repo: str
    branch: str
    commit: str
    resolved_at: str
    fingerprint: "TreeFingerprint"

    def to_dict(self) -> dict:
        return {"repo": self.repo, "branch": self.branch, "commit": self.commit,
                "resolved_at": self.resolved_at,
                "fingerprint": self.fingerprint.to_dict()}


def resolve_anchor(repo: Any, branch: str, *,
                   expected_commit: Optional[str] = None) -> Anchor:
    """Resolve the CURRENT tip of `branch` in `repo`, and fingerprint the tree.

    `expected_commit` is an optional assertion, not a default. Passing the v8
    freeze commit here turns "I believe production is at 67a433bf4" into a
    checked precondition; omitting it takes whatever the tip is, which is what a
    campaign that must start from the *current* tip actually wants.
    """
    git = repo if isinstance(repo, GitRepo) else GitRepo(repo)
    tip = git.branch_tip(branch)
    if expected_commit is not None and tip != _req_commit(expected_commit, "expected_commit"):
        raise StaleAnchor(
            f"{branch!r} in {git.path!r} is at {tip}, not the expected {expected_commit}. "
            "Refusing to anchor to a commit the caller believes is the tip when it is not")
    return Anchor(repo=git.path, branch=branch, commit=tip,
                  resolved_at=_utc_now_iso(), fingerprint=fingerprint_tree(git))


def campaign_worktree_path(campaign_id: str, *, source_tree: str = "llama.cpp",
                           root: Any = DEFAULT_WORKTREE_ROOT) -> SandboxPath:
    """`<root>/<source_tree>-<campaign_id>` as a validated `SandboxPath`.

    With `campaign_id` forced to start with `ak-`, this is design §5.3's
    `llama.cpp-ak-<campaign_id>` by construction. The `SandboxPath` then refuses
    the result if it lands inside a frozen tree — which it would if a caller
    passed `root=/mnt/raid0/llm/llama.cpp`.
    """
    validate_campaign_id(campaign_id)
    if not _SOURCE_TREE_RE.match(_req_str(source_tree, "source_tree")):
        raise UnsafePath(f"source_tree {source_tree!r} is not a plain directory name")
    root_path = _real(root, "worktree root")
    return SandboxPath.in_sandbox(os.path.join(root_path, f"{source_tree}-{campaign_id}"),
                                  sandbox_root=root_path, label="campaign worktree")


def snapshot_worktree_path(campaign_id: str, candidate_id: str, *,
                           source_tree: str = "llama.cpp",
                           root: Any = DEFAULT_WORKTREE_ROOT) -> SandboxPath:
    """A candidate-specific sibling used only for a detached clean build."""
    validate_campaign_id(campaign_id)
    if not isinstance(candidate_id, str) or not _SOURCE_TREE_RE.match(candidate_id) \
            or not candidate_id.startswith("akc-"):
        raise ValueError("candidate_id must be an akc- prefixed plain path component")
    if not _SOURCE_TREE_RE.match(_req_str(source_tree, "source_tree")):
        raise UnsafePath(f"source_tree {source_tree!r} is not a plain directory name")
    root_path = _real(root, "worktree root")
    return SandboxPath.in_sandbox(
        os.path.join(root_path, f"{source_tree}-{campaign_id}-{candidate_id}-snapshot"),
        sandbox_root=root_path, label="snapshot worktree")


def create_campaign_worktree(anchor: Anchor, campaign_id: str, *,
                             leaf: str = "base",
                             source_tree: str = "llama.cpp",
                             root: Any = DEFAULT_WORKTREE_ROOT,
                             require_current_tip: bool = True) -> tuple:
    """Create `llama.cpp-ak-<campaign_id>` from the anchor. Returns `(Worktree, proof)`.

    `require_current_tip` re-resolves the branch tip HERE and raises
    :class:`StaleAnchor` if it moved since `anchor` was taken. That is
    INC-20260706-iqk-missing-subsystem expressed as a precondition: a campaign
    that forks from a stale tip is missing every optimization that landed since,
    and it fails silently — the candidate builds and simply loses to production
    for a reason nobody attributes to the fork point. Turning it off is possible
    and is a deliberate act with a name.

    The returned `ImmutabilityProof` covers the ANCHOR repository, which may be a
    frozen production tree. `git worktree add` writes `.git/worktrees/<name>/`
    there; it must not touch the working tree, the checked-out branch or the
    index, and this is where that is demonstrated rather than asserted.
    """
    git = GitRepo(anchor.repo)
    before = fingerprint_tree(git)
    if require_current_tip:
        tip = git.branch_tip(anchor.branch)
        if tip != anchor.commit:
            raise StaleAnchor(
                f"{anchor.branch!r} moved from {anchor.commit[:12]} to {tip[:12]} since the "
                f"anchor was resolved at {anchor.resolved_at}. CLAUDE.md step 1: start from "
                "the CURRENT production tip; never accumulate on a branch forked from an old "
                "one (INC-20260706-iqk-missing-subsystem)")
    dest = campaign_worktree_path(campaign_id, source_tree=source_tree, root=root)
    branch = SafeBranch.for_campaign(campaign_id, leaf)
    worktree = git.add_worktree(dest, anchor.commit, branch=branch)
    after = fingerprint_tree(git)
    proof = prove_unchanged(before, after)
    if not proof.holds:
        raise ProductionMutated(
            f"creating {dest.path!r} changed {git.path!r}: {proof.differences}")
    return worktree, proof


def create_snapshot_worktree(repo: Any, commit: str, dest: Any) -> tuple:
    """A DETACHED worktree at `commit` — the pristine source root for a clean build.

    §8.5.1 (2): *"T0 compiles from the content-addressed source snapshot in a
    fresh build directory — never from the actor's incremental tree."* This is
    how the snapshot becomes a directory: a second worktree, detached, holding
    exactly the committed tree, with none of the actor's uncommitted state. It
    is what makes `BuildProvenance.source_root != actor_worktree` true rather
    than aspirational.
    """
    git = repo if isinstance(repo, GitRepo) else GitRepo(repo)
    before = fingerprint_tree(git)
    dest_path = dest if isinstance(dest, SandboxPath) else SandboxPath.in_sandbox(
        dest, label="snapshot worktree")
    worktree = git.add_worktree(dest_path, commit, detach=True)
    proof = prove_unchanged(before, fingerprint_tree(git))
    if not proof.holds:
        raise ProductionMutated(
            f"creating snapshot worktree {dest_path.path!r} changed {git.path!r}: "
            f"{proof.differences}")
    return worktree, proof


# =============================================================================
# Immutability proof
# =============================================================================

@dataclass(frozen=True)
class TreeFingerprint:
    """What must be identical before and after: HEAD, the ref HEAD points at, status.

    Not a content hash of the whole tree — a 13 GiB llama.cpp checkout is too
    expensive to hash on every teardown, and the three facts here are exactly
    what "the working tree, branch and index are inviolate" means. `status
    --porcelain` covers the index and the working tree together: a staged change
    shows as `M ` and an unstaged one as ` M`, so a mutation of either moves the
    string.
    """

    path: str
    head_commit: str
    symbolic_ref: Optional[str]
    status_porcelain: str
    captured_at: str

    @property
    def sha256(self) -> str:
        return _sha256_text("\x00".join([
            self.path, self.head_commit, self.symbolic_ref or "",
            self.status_porcelain]))

    def to_dict(self) -> dict:
        return {"path": self.path, "head_commit": self.head_commit,
                "symbolic_ref": self.symbolic_ref,
                "status_porcelain": self.status_porcelain,
                "status_line_count": len(self.status_porcelain.splitlines()),
                "captured_at": self.captured_at, "sha256": self.sha256}


def fingerprint_tree(repo: Any) -> TreeFingerprint:
    """Read-only. Safe against a frozen production tree — it is the point of it."""
    git = repo if isinstance(repo, GitRepo) else GitRepo(repo)
    return TreeFingerprint(
        path=git.path, head_commit=git.head_commit(),
        symbolic_ref=git.current_branch(), status_porcelain=git.status_porcelain(),
        captured_at=_utc_now_iso())


@dataclass(frozen=True)
class ImmutabilityProof:
    """`before`/`after` plus the named differences. `holds` is DERIVED, never set."""

    before: TreeFingerprint
    after: TreeFingerprint
    differences: tuple

    @property
    def holds(self) -> bool:
        return not self.differences

    def to_dict(self) -> dict:
        return {"proof": "autokernel.execution.worktree.immutability/v1",
                "path": self.before.path, "holds": self.holds,
                "differences": list(self.differences),
                "before": self.before.to_dict(), "after": self.after.to_dict()}


def prove_unchanged(before: TreeFingerprint, after: TreeFingerprint) -> ImmutabilityProof:
    """Compare two fingerprints of the SAME tree, naming every difference."""
    if not isinstance(before, TreeFingerprint) or not isinstance(after, TreeFingerprint):
        raise TypeError("prove_unchanged takes two TreeFingerprints")
    if before.path != after.path:
        raise ValueError(
            f"cannot compare fingerprints of different trees: {before.path!r} vs "
            f"{after.path!r}")
    diffs = []
    if before.head_commit != after.head_commit:
        diffs.append(f"HEAD {before.head_commit[:12]} -> {after.head_commit[:12]}")
    if before.symbolic_ref != after.symbolic_ref:
        diffs.append(f"branch {before.symbolic_ref!r} -> {after.symbolic_ref!r}")
    if before.status_porcelain != after.status_porcelain:
        diffs.append(
            f"`git status --porcelain` changed "
            f"({len(before.status_porcelain.splitlines())} -> "
            f"{len(after.status_porcelain.splitlines())} lines)")
    return ImmutabilityProof(before=before, after=after, differences=tuple(diffs))


@dataclass(frozen=True)
class TeardownReceipt:
    """What teardown removed, and the proof production is byte-identical."""

    worktree_path: str
    worktree_removed: bool
    branch: Optional[str]
    branch_deleted: bool
    branch_exists_after: bool
    production_proofs: tuple
    torn_down_at: str
    discarded_status_porcelain: str
    was_dirty: bool
    #: Every frozen tree that EXISTS on this host and was actually fingerprinted.
    #: Present because `witness_trees` is an override and an override that
    #: replaced the frozen set used to leave the receipt saying
    #: `all_production_trees_unchanged: true` having witnessed none of them.
    production_trees_witnessed: tuple = ()
    production_trees_unwitnessed: tuple = ()

    @property
    def all_production_trees_witnessed(self) -> bool:
        return not self.production_trees_unwitnessed

    def to_dict(self) -> dict:
        return {"receipt": "autokernel.execution.worktree.teardown/v1",
                "worktree_path": self.worktree_path,
                "worktree_removed": self.worktree_removed,
                "branch": self.branch, "branch_deleted": self.branch_deleted,
                "branch_exists_after": self.branch_exists_after,
                "was_dirty": self.was_dirty,
                "discarded_status_porcelain": self.discarded_status_porcelain,
                "production_proofs": [p.to_dict() for p in self.production_proofs],
                "production_trees_witnessed": list(self.production_trees_witnessed),
                "production_trees_unwitnessed": list(self.production_trees_unwitnessed),
                "all_production_trees_witnessed": self.all_production_trees_witnessed,
                # BOTH conjuncts. "Unchanged" over a set nobody looked at is the
                # sentence a reader would believe and the one that would be false.
                "all_production_trees_unchanged": (
                    self.all_production_trees_witnessed
                    and all(p.holds for p in self.production_proofs)),
                "torn_down_at": self.torn_down_at}

    @property
    def content_hash(self) -> str:
        return schemas.content_hash(self.to_dict())


def teardown_worktree(worktree: Worktree, *, delete_branch: bool = True,
                      force: bool = True,
                      witness_trees: Sequence[Any] = ()) -> TeardownReceipt:
    """Remove the worktree, delete its branch, and PROVE production is untouched.

    `witness_trees` defaults to the frozen production trees that actually exist
    on this host. Each is fingerprinted before and after; any difference raises
    :class:`ProductionMutated`. A teardown that silently changed a frozen tree
    and returned a receipt saying so is worse than one that crashes, because the
    receipt would be believed.

    Passing `witness_trees` REPLACES that default — a test with a temporary
    clone needs exactly that. What it may not do is quietly buy the headline:
    the receipt records `production_trees_witnessed` and
    `production_trees_unwitnessed`, and `all_production_trees_unchanged` is
    false unless every frozen tree present on this host was actually read. It
    used to be `all(p.holds …)` over whatever list the caller supplied, so a
    teardown that witnessed one temporary directory produced a receipt asserting
    all three frozen trees were unchanged.

    `force=True` is the default, and the reasoning is worth stating because a
    forcing default usually is not:

    * the tree being removed is a `SandboxPath`, so it provably is not a frozen
      production tree — the destructive force is bounded by the type, not by the
      caller remembering a keyword;
    * a candidate worktree at teardown is almost always dirty (a failed build
      leaves objects, an abandoned repair leaves edits), and §5.8 classes
      rejected-candidate trees as *expirable*. A default that failed on the
      normal case would be worked around with `force=True` at every call site,
      which is the same behaviour with the discipline removed;
    * nothing is discarded silently. `discarded_status_porcelain` captures
      `git status --porcelain` BEFORE removal, so the receipt names every file
      that went. Record the gap, do not patch it.
    """
    if not isinstance(worktree, Worktree):
        raise TypeError("teardown_worktree takes a Worktree")
    trees = list(witness_trees) if witness_trees else [
        t for t in PRODUCTION_TREES if os.path.isdir(os.path.join(t, ".git"))]
    repos = [t if isinstance(t, GitRepo) else GitRepo(t) for t in trees]
    # The owning clone is always a witness: `worktree remove` writes its
    # `.git/worktrees/`, so it is the tree most likely to move.
    if all(r.path != worktree.repo.path for r in repos):
        repos.append(worktree.repo)
    before = [fingerprint_tree(r) for r in repos]

    discarded = worktree.status_porcelain() if os.path.isdir(worktree.path.path) else ""
    worktree.repo.remove_worktree(worktree.path, force=force)
    removed = not os.path.exists(worktree.path.path)

    branch_deleted = False
    if delete_branch and worktree.branch is not None:
        worktree.repo.delete_branch(worktree.branch)
        branch_deleted = True
    branch_exists_after = (worktree.branch is not None
                           and worktree.repo.branch_exists(worktree.branch))

    proofs = tuple(prove_unchanged(b, fingerprint_tree(r))
                   for b, r in zip(before, repos))
    witnessed = {p.before.path for p in proofs}
    on_host = tuple(t for t in frozen_tree_paths()
                    if os.path.isdir(os.path.join(t, ".git")))
    seen = tuple(t for t in on_host if os.path.realpath(t) in witnessed or t in witnessed)
    unseen = tuple(t for t in on_host if t not in seen)
    broken = [p for p in proofs if not p.holds]
    if broken:
        raise ProductionMutated(
            "teardown changed a witnessed tree: "
            + "; ".join(f"{p.before.path}: {p.differences}" for p in broken))

    return TeardownReceipt(
        worktree_path=worktree.path.path, worktree_removed=removed,
        branch=worktree.branch.name if worktree.branch else None,
        branch_deleted=branch_deleted, branch_exists_after=branch_exists_after,
        production_proofs=proofs, torn_down_at=_utc_now_iso(),
        discarded_status_porcelain=discarded, was_dirty=bool(discarded.strip()),
        production_trees_witnessed=seen, production_trees_unwitnessed=unseen)


# =============================================================================
# The build
# =============================================================================

@dataclass(frozen=True)
class BuildParallelism:
    """How much of a SHARED machine this build may take. No default exists.

    `jobs` is required, deliberately. A default would be either full width —
    antisocial on a box that tonight carries load average ~67 and six resident
    `llama-server` instances from another session — or a small number that
    quietly makes every build slow. Making the caller say it means the answer is
    always a decision somebody made.
    """

    jobs: int
    cpu_list: Optional[str] = None
    load_average_cap: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.jobs, int) or isinstance(self.jobs, bool) or self.jobs < 1:
            raise ValueError(f"BuildParallelism.jobs must be a positive int, got {self.jobs!r}")
        total = os.cpu_count() or 1
        if self.jobs > total:
            raise ValueError(
                f"BuildParallelism.jobs={self.jobs} exceeds the {total} CPUs this host "
                "reports; oversubscribing a shared box does not make the build faster")
        if self.cpu_list is not None:
            if not _CPU_LIST_RE.match(_req_str(self.cpu_list, "cpu_list")):
                raise ValueError(
                    f"cpu_list {self.cpu_list!r} must be a taskset list like '0-47' or "
                    "'0-23,48-71'; it is passed to taskset as a literal and nothing else "
                    "is accepted there")
        if self.load_average_cap is not None:
            cap = float(self.load_average_cap)
            if not cap > 0:
                raise ValueError("load_average_cap must be positive when given")

    @classmethod
    def share_of_machine(cls, fraction: float, **kwargs: Any) -> "BuildParallelism":
        """`fraction` of the host's CPUs, at least one. Still an explicit decision."""
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
        total = os.cpu_count() or 1
        return cls(jobs=max(1, int(total * fraction)), **kwargs)

    def to_dict(self) -> dict:
        return {"jobs": self.jobs, "cpu_list": self.cpu_list,
                "load_average_cap": self.load_average_cap}


@dataclass(frozen=True)
class BuildPlan:
    """Everything needed to configure and build, and nothing that runs anything.

    Splitting argv CONSTRUCTION from execution is what makes tonight's work
    possible under contention: the argv is fully testable without compiling, and
    `test_worktree.py` asserts every flag this plan produces.

    `build_dir` must not be inside `actor_worktree` — that is not this module's
    opinion, it is `integrity.check_clean_build_from_snapshot` sub-check (d),
    which FAILs when it is, because *"the actor's build state would become part
    of the artifact"*. Refusing it here means a plan that would have produced a
    failing gate cannot be built in the first place.
    """

    source_root: SandboxPath
    build_dir: SandboxPath
    actor_worktree: SandboxPath
    parallelism: BuildParallelism
    targets: tuple = ()
    build_type: str = "Release"
    cmake_defines: tuple = ()
    generator: Optional[str] = None
    allow_ccache: bool = False
    cmake: str = "cmake"

    def __post_init__(self) -> None:
        for name in ("source_root", "build_dir", "actor_worktree"):
            if not isinstance(getattr(self, name), SandboxPath):
                raise TypeError(
                    f"BuildPlan.{name} must be a SandboxPath — a str would let a frozen "
                    "production tree become a build target")
        if not isinstance(self.parallelism, BuildParallelism):
            raise TypeError("BuildPlan.parallelism must be a BuildParallelism")
        if self.build_dir.path == self.source_root.path:
            raise UnsafePath(
                "build_dir must not be the source root; an in-source build makes the "
                "snapshot digest a function of the build")
        if _is_within(self.build_dir.path, self.actor_worktree.path):
            raise UnsafePath(
                f"build_dir {self.build_dir.path!r} is inside the actor worktree "
                f"{self.actor_worktree.path!r}. §8.5.1 (2) fails exactly this: the actor's "
                "build state would become part of the artifact")
        if not isinstance(self.targets, tuple):
            raise TypeError("BuildPlan.targets must be a tuple")
        for target in self.targets:
            if not _req_str(target, "target").replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"target {target!r} is not a plain cmake target name")
        if not isinstance(self.cmake_defines, tuple):
            raise TypeError("BuildPlan.cmake_defines must be a tuple of (name, value)")
        for entry in self.cmake_defines:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"cmake_defines entry {entry!r} must be a (name, value) pair")
            name, value = entry
            if not _CMAKE_DEFINE_NAME_RE.match(_req_str(name, "cmake define name")):
                raise ValueError(f"cmake define name {name!r} is not a plain identifier")
            _req_str(str(value), "cmake define value")
        if _req_str(self.build_type, "build_type") not in (
                "Release", "RelWithDebInfo", "Debug", "MinSizeRel"):
            raise ValueError(f"build_type {self.build_type!r} is not a cmake build type")
        # The program is data too. `cmake` used to be an unvalidated free string
        # while `targets` and `cmake_defines` were both pattern-checked, so the
        # one field that decides WHICH PROGRAM RUNS was the loosest in the
        # dataclass — and with `cpu_list` set it lands at argv[3] behind
        # `taskset`, where an argv[0]-only guard never looked.
        cmake = _req_str(self.cmake, "cmake")
        program = os.path.basename(cmake)
        if os.sep in cmake and not os.path.isabs(cmake):
            raise ValueError(
                f"cmake {cmake!r} must be a bare program name or an absolute path; a "
                "relative path resolves against a cwd this module does not control")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$", program):
            raise ValueError(f"cmake {cmake!r} is not a plain program name")
        if program in _NAME_PATTERN_BINARIES:
            raise ValueError(
                f"cmake={cmake!r} names a NAME-PATTERN process tool (INC-20260731)")

    # -- derived -----------------------------------------------------------
    @property
    def effective_defines(self) -> tuple:
        """Caller defines plus the clean-build forcings, caller-last-wins per name.

        `GGML_CCACHE=OFF` is prepended rather than appended so an explicit caller
        define of the same name still wins — the forcing is a default, and
        `allow_ccache=True` removes it entirely. What is not negotiable is that
        the outcome ends up in the receipt.
        """
        forced = () if self.allow_ccache else CLEAN_BUILD_CMAKE_DEFINES
        merged: dict = {}
        for name, value in tuple(forced) + tuple(self.cmake_defines):
            merged[name] = value
        return tuple(sorted(merged.items()))

    def configure_argv(self) -> tuple:
        argv = [self.cmake, "-S", self.source_root.path, "-B", self.build_dir.path,
                f"-DCMAKE_BUILD_TYPE={self.build_type}"]
        if self.generator is not None:
            argv += ["-G", _req_str(self.generator, "generator")]
        argv += [f"-D{name}={value}" for name, value in self.effective_defines]
        return self._prefix() + tuple(argv)

    def build_argv(self) -> tuple:
        argv = [self.cmake, "--build", self.build_dir.path,
                "-j", str(self.parallelism.jobs)]
        for target in self.targets:
            argv += ["--target", target]
        return self._prefix() + tuple(argv)

    def _prefix(self) -> tuple:
        """`taskset -c <list>` when the caller confined the build. Nothing else.

        No `nice`, no `ionice`, no wrapper that a reader would have to know
        about to reproduce the build: everything that changes the build's
        environment must be visible in `BuildIdentity.command`.
        """
        if self.parallelism.cpu_list is None:
            return ()
        return ("taskset", "-c", self.parallelism.cpu_list)

    def to_dict(self) -> dict:
        return {"source_root": self.source_root.path, "build_dir": self.build_dir.path,
                "actor_worktree": self.actor_worktree.path,
                "parallelism": self.parallelism.to_dict(),
                "targets": list(self.targets), "build_type": self.build_type,
                "cmake_defines": [list(d) for d in self.effective_defines],
                "generator": self.generator, "allow_ccache": self.allow_ccache,
                "configure_command": list(self.configure_argv()),
                "build_command": list(self.build_argv())}


def default_build_dir(campaign_id: str, candidate_id: str, *,
                      root: Any = DEFAULT_BUILD_ROOT) -> SandboxPath:
    """`<root>/<campaign_id>/<candidate_id>` — outside every worktree by construction."""
    validate_campaign_id(campaign_id)
    if not isinstance(candidate_id, str) or not candidate_id.startswith("akc-"):
        raise ValueError(f"candidate_id {candidate_id!r} must start with 'akc-'")
    if not _SOURCE_TREE_RE.match(candidate_id):
        raise ValueError(f"candidate_id {candidate_id!r} is not a plain directory name")
    root_path = _real(root, "build root")
    return SandboxPath.in_sandbox(os.path.join(root_path, campaign_id, candidate_id),
                                  sandbox_root=root_path, label="build dir")


# =============================================================================
# Build-log parsing
#
# Every pattern here was written against RECORDED build output from this host,
# checked in under `execution/testdata/` with its provenance. Guessing at a
# compiler's phrasing produces a parser that reports zero errors on a failed
# build, which is the single worst wrong answer this function can give.
# =============================================================================

_RE_COMPILER_ID = re.compile(r"^-- The (\w+) compiler identification is (.+)$")
_RE_BUILD_FILES = re.compile(r"^-- Build files have been written to: (.+)$")
_RE_CCACHE = re.compile(r"^-- ccache found, compilation results will be cached")
_RE_GGML_VERSION = re.compile(r"^-- ggml version:\s*(.+?)\s*$")
_RE_GGML_COMMIT = re.compile(r"^-- ggml commit:\s*(.+?)\s*$")
_RE_BUILT_TARGET = re.compile(r"^\[\s*\d+%\] Built target (.+?)\s*$")
_RE_BUILDING = re.compile(r"^\[\s*\d+%\] Building (?:C|CXX|ASM|HIP|CUDA) object ")
_RE_LINKING = re.compile(r"^\[\s*\d+%\] Linking \w+ (?:executable|shared library|static library) (.+?)\s*$")
_RE_DIAG = re.compile(r"^(?P<where>[^\s].*?):\s(?P<kind>error|warning|fatal error):\s(?P<msg>.*)$")
_RE_LD_ERROR = re.compile(r"^(?:/usr/bin/ld|collect2|ld\.lld|lld):\s*(?:.*\s)?error:?\s*(.*)$")
_RE_LD_UNDEF = re.compile(r"^/usr/bin/ld: (.*): undefined reference to (.+)$")
_RE_MAKE_FAIL = re.compile(r"^g?make(?:\[\d+\])?: \*\*\* (.+?)\s*(?:Error \d+)?$")
_RE_NINJA_FAIL = re.compile(r"^ninja: build stopped: (.+)$")
_RE_CMAKE_ERROR = re.compile(r"^CMake Error(?: at (.+))?:")


@dataclass(frozen=True)
class BuildLogFacts:
    """What a build log says. Derived only — this record asserts nothing itself.

    `succeeded_by_log` is deliberately SEPARATE from the exit code, and
    `BuildResult` compares the two. The two disagreeing is a real condition on
    this host: a build piped through anything loses the compiler's status under
    the default (non-`pipefail`) shell, so "exit 0 with `gmake: *** Error 2` in
    the log" is not hypothetical. `feedback_pipe_hazards` is the memo; this
    field is the detector.
    """

    configured: bool
    build_dir_from_log: Optional[str]
    compiler_ids: tuple
    ccache_enabled: bool
    ggml_version: Optional[str]
    ggml_commit: Optional[str]
    ggml_commit_dirty: bool
    built_targets: tuple
    linked_outputs: tuple
    compile_units: int
    warning_count: int
    errors: tuple
    make_failures: tuple
    succeeded_by_log: bool

    def to_dict(self) -> dict:
        return {"configured": self.configured,
                "build_dir_from_log": self.build_dir_from_log,
                "compiler_ids": [list(c) for c in self.compiler_ids],
                "ccache_enabled": self.ccache_enabled,
                "ggml_version": self.ggml_version, "ggml_commit": self.ggml_commit,
                "ggml_commit_dirty": self.ggml_commit_dirty,
                "built_targets": list(self.built_targets),
                "linked_outputs": list(self.linked_outputs),
                "compile_units": self.compile_units,
                "warning_count": self.warning_count,
                "errors": list(self.errors), "make_failures": list(self.make_failures),
                "succeeded_by_log": self.succeeded_by_log}

    @property
    def first_error(self) -> Optional[str]:
        return self.errors[0] if self.errors else None


def parse_build_log(text: Any, *, max_errors: int = 50) -> BuildLogFacts:
    """Parse a cmake+make/ninja build log into checkable facts.

    Diagnostic lines are matched with an anchored pattern that requires
    `<something>: error: <msg>`. A looser `"error" in line` test would count
    every one of the hundreds of `-Wmaybe-uninitialized` NOTE lines in a real
    llama.cpp log, and a source file named `error_handling.cpp` compiles into an
    error report.
    """
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", "replace")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TypeError(f"build log must be text, got {type(text).__name__}")
    lines = text.splitlines()

    compiler_ids: list = []
    ccache = False
    configured = False
    build_dir = None
    ggml_version = None
    ggml_commit = None
    targets: list = []
    linked: list = []
    compile_units = 0
    warnings = 0
    errors: list = []
    make_failures: list = []

    for line in lines:
        stripped = line.rstrip()
        match = _RE_COMPILER_ID.match(stripped)
        if match:
            compiler_ids.append((match.group(1), match.group(2).strip()))
            continue
        if _RE_CCACHE.match(stripped):
            ccache = True
            continue
        match = _RE_BUILD_FILES.match(stripped)
        if match:
            configured = True
            build_dir = match.group(1).strip()
            continue
        match = _RE_GGML_VERSION.match(stripped)
        if match:
            ggml_version = match.group(1)
            continue
        match = _RE_GGML_COMMIT.match(stripped)
        if match:
            ggml_commit = match.group(1)
            continue
        match = _RE_BUILT_TARGET.match(stripped)
        if match:
            targets.append(match.group(1))
            continue
        if _RE_BUILDING.match(stripped):
            compile_units += 1
            continue
        match = _RE_LINKING.match(stripped)
        if match:
            linked.append(match.group(1))
            continue
        match = _RE_MAKE_FAIL.match(stripped)
        if match:
            make_failures.append(stripped)
            continue
        match = _RE_NINJA_FAIL.match(stripped)
        if match:
            make_failures.append(stripped)
            continue
        if _RE_CMAKE_ERROR.match(stripped):
            errors.append(stripped)
            continue
        match = _RE_LD_UNDEF.match(stripped)
        if match:
            errors.append(stripped)
            continue
        match = _RE_LD_ERROR.match(stripped)
        if match:
            errors.append(stripped)
            continue
        match = _RE_DIAG.match(stripped)
        if match:
            if match.group("kind") == "warning":
                warnings += 1
            else:
                errors.append(stripped)

    deduped: list = []
    seen = set()
    for err in errors:
        if err not in seen:
            seen.add(err)
            deduped.append(err)
        if len(deduped) >= max_errors:
            break

    dirty = bool(ggml_commit) and ggml_commit.endswith("-dirty")
    # POSITIVE evidence is required, not merely an absence of complaints. An
    # empty log used to report `succeeded_by_log=True` — "nothing failed" — so a
    # log that was truncated, lost its sink, or never got written read as a
    # clean build, and `log_disagrees_with_exit_code` then agreed with an exit 0
    # that came from a wrapper. A build that says nothing has not said it
    # succeeded.
    saw_work = bool(configured or targets or linked or compile_units)
    return BuildLogFacts(
        configured=configured, build_dir_from_log=build_dir,
        compiler_ids=tuple(compiler_ids), ccache_enabled=ccache,
        ggml_version=ggml_version, ggml_commit=ggml_commit,
        ggml_commit_dirty=dirty, built_targets=tuple(targets),
        linked_outputs=tuple(linked), compile_units=compile_units,
        warning_count=warnings, errors=tuple(deduped),
        make_failures=tuple(make_failures),
        succeeded_by_log=saw_work and not deduped and not make_failures)


# =============================================================================
# Running the build
# =============================================================================

@dataclass(frozen=True)
class BuildResult:
    """A build that ran: dispositions, log, parsed facts, and the disagreement flag."""

    plan: BuildPlan
    configure: Optional[ProcessDisposition]
    build: Optional[ProcessDisposition]
    log_path: str
    log_sha256: str
    facts: BuildLogFacts
    build_dir_pre_build_digest: str
    build_dir_created_for_this_build: bool
    #: 1-minute load average read immediately before configure, when a cap was
    #: declared. `None` means no cap was declared — never "the cap passed".
    load_average_at_start: Optional[float] = None
    log_identity: Optional[Mapping[str, int]] = None
    result_receipt_path: Optional[str] = None
    result_receipt_sha256: Optional[str] = None

    @property
    def exit_code(self) -> Optional[int]:
        if self.build is not None:
            return self.build.exit_code
        return self.configure.exit_code if self.configure is not None else None

    @property
    def succeeded(self) -> bool:
        """The EXIT CODE is authoritative. The log is corroboration."""
        return self.exit_code == 0

    @property
    def log_disagrees_with_exit_code(self) -> bool:
        """Exit code says one thing, log says the other. Always worth surfacing.

        Exit 0 with `gmake: *** Error 2` in the log means the status came from
        something other than the compiler — a pipe, a wrapper, a `|| true`. A
        campaign that accepted that would carry a broken binary forward as a
        clean build.
        """
        return self.succeeded != self.facts.succeeded_by_log

    def to_dict(self) -> dict:
        return {"plan": self.plan.to_dict(),
                "configure": self.configure.to_dict() if self.configure else None,
                "build": self.build.to_dict() if self.build else None,
                "log_path": self.log_path, "log_sha256": self.log_sha256,
                "facts": self.facts.to_dict(), "exit_code": self.exit_code,
                "succeeded": self.succeeded,
                "log_disagrees_with_exit_code": self.log_disagrees_with_exit_code,
                "build_dir_pre_build_digest": self.build_dir_pre_build_digest,
                "build_dir_created_for_this_build": self.build_dir_created_for_this_build,
                "load_average_at_start": self.load_average_at_start,
                "log_identity": (dict(self.log_identity)
                                 if self.log_identity is not None else None),
                "result_receipt_path": self.result_receipt_path,
                "result_receipt_sha256": self.result_receipt_sha256}


def run_build(plan: BuildPlan, *, log_path: Any,
              configure_timeout_s: float = 900.0,
              build_timeout_s: float = 14400.0,
              env: Optional[Mapping[str, str]] = None,
              require_fresh_build_dir: bool = True,
              sandbox_cgroup_root: Optional[str] = None) -> BuildResult:
    """Configure, build, capture the log, and return the facts.

    The build directory is created here and its pre-build digest is taken with
    `integrity.hash_source_tree` BEFORE anything runs, because that digest is
    what `check_clean_build_from_snapshot` compares against `EMPTY_TREE_SHA256`.
    Taking it afterwards would prove nothing at all.

    Configure and build are two owned, sandboxed processes and two log sections,
    appended to one evaluator-owned file.  Candidate CMake is executable input:
    both phases therefore run with the same fail-closed Landlock/seccomp/cgroup
    boundary as candidate benchmarks.  On a configure failure the build is not
    attempted: cmake will happily "build" a stale cache and the resulting binary
    would be from a configuration nobody recorded.
    """
    if not isinstance(plan, BuildPlan):
        raise TypeError("run_build takes a BuildPlan")
    log = _real(log_path, "log_path")
    if _is_within(log, plan.build_dir.path):
        raise UnsafePath(
            "build log and sandbox activation receipts must be evaluator-owned "
            "outside the candidate-writable build directory")
    os.makedirs(os.path.dirname(log), exist_ok=True)

    created = not plan.build_dir.exists
    os.makedirs(plan.build_dir.path, exist_ok=True)
    pre_digest = integrity.hash_source_tree(plan.build_dir.path).sha256
    if require_fresh_build_dir and pre_digest != integrity.EMPTY_TREE_SHA256:
        raise BuildDirNotFresh(
            f"{plan.build_dir.path!r} is not empty (digest {pre_digest[:12]}, empty tree is "
            f"{integrity.EMPTY_TREE_SHA256[:12]}). §8.5.1 (2) requires a FRESH build "
            "directory: an incremental build can link stale objects and hide the error the "
            "snapshot would surface")

    candidate_tmp = os.path.join(plan.build_dir.path, ".autokernel-tmp")
    os.makedirs(candidate_tmp, mode=0o700, exist_ok=False)
    build_env = dict(os.environ if env is None else env)
    build_env["TMPDIR"] = candidate_tmp
    build_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if sandbox_cgroup_root is not None:
        cgroup_root = Path(sandbox_cgroup_root)
        if (not cgroup_root.is_absolute() or cgroup_root.is_symlink()
                or not cgroup_root.is_dir()):
            raise WorktreeError("sandbox cgroup root is not an exact directory")
    sandbox_policy = process_sandbox.SandboxPolicy(
        writable_root=plan.build_dir.path,
        **({"cgroup_root": sandbox_cgroup_root}
           if sandbox_cgroup_root is not None else {}))
    configure_sandbox_receipt = log + ".configure-sandbox.json"
    build_sandbox_receipt = log + ".build-sandbox.json"

    # The declared cap is now a PRECONDITION. It was a recorded field that
    # nothing read: `parallelism.load_average_cap` rode into the receipt and
    # into `BuildIdentity.parallelism`, so the artifact stated a restraint that
    # was never applied. Tonight's host carries load ~67 with six resident
    # llama-server instances — a 96-way build under that is both bad data and
    # theft from whoever is measuring.
    load_now: Optional[float] = None
    cap = plan.parallelism.load_average_cap
    if cap is not None:
        load_now = os.getloadavg()[0]
        if load_now > cap:
            raise HostTooContended(
                f"1-minute load average is {load_now:.2f}, above the declared cap "
                f"{cap:.2f}. Refusing to start a {plan.parallelism.jobs}-way build: the "
                "cap was recorded in the receipt, so it has to be the thing that happened")

    if os.path.lexists(log):
        raise WorktreeError(f"build log already exists; refusing overwrite: {log}")
    configure_stream = log + ".configure.stream"
    build_stream = log + ".build.stream"
    process_prefix = log + ".configure-process"
    configure_disp, configure_text = _run_owned(
        plan.configure_argv(), timeout_s=configure_timeout_s, env=build_env,
        stdout_path=configure_stream, process_receipt_prefix=process_prefix,
        sandbox_policy=sandbox_policy,
        sandbox_receipt_path=configure_sandbox_receipt)
    sections: list = ["=== configure: " + " ".join(plan.configure_argv()) + "\n",
                      configure_text]

    build_disp = None
    if configure_disp.exit_code == 0:
        build_disp, build_text = _run_owned(
            plan.build_argv(), timeout_s=build_timeout_s, env=build_env,
            stdout_path=build_stream,
            process_receipt_prefix=log + ".build-process",
            sandbox_policy=sandbox_policy,
            sandbox_receipt_path=build_sandbox_receipt)
        sections.append("=== build: " + " ".join(plan.build_argv()) + "\n")
        sections.append(build_text)

    combined = "".join(sections)
    handle = _exclusive_regular_sink(log)
    try:
        handle.write(combined.encode("utf-8"))
        raw, log_identity = _read_and_revalidate_open_stream(handle, log)
        if raw != combined.encode("utf-8"):
            raise WorktreeError("combined build log differs from its writer bytes")
        facts = parse_build_log(combined)
        receipt_path = log + ".result.json"
        receipt_body = {
            "schema": "epyc.autokernel.build_process_result.v1",
            "plan": plan.to_dict(),
            "configure": configure_disp.to_dict() if configure_disp else None,
            "build": build_disp.to_dict() if build_disp else None,
            "log_path": log, "log_sha256": hashlib.sha256(raw).hexdigest(),
            "log_identity": log_identity, "facts": facts.to_dict(),
            "build_dir_pre_build_digest": pre_digest,
            "build_dir_created_for_this_build": created,
            "load_average_at_start": load_now,
        }
        receipt_sha = _sealed_process_receipt(receipt_path, receipt_body)
        _revalidate_open_stream(handle, log, log_identity)
    finally:
        handle.close()

    return BuildResult(
        plan=plan, configure=configure_disp, build=build_disp, log_path=log,
        log_sha256=hashlib.sha256(raw).hexdigest(), facts=facts,
        build_dir_pre_build_digest=pre_digest,
        build_dir_created_for_this_build=created,
        load_average_at_start=load_now, log_identity=log_identity,
        result_receipt_path=receipt_path,
        result_receipt_sha256=receipt_sha)


# =============================================================================
# The build identity receipt
#
# Shaped to what `evaluator/integrity.py` ALREADY consumes, rather than to
# something it would have to be adapted to: `to_build_provenance()` returns an
# `integrity.BuildProvenance` and `to_candidate_records()` returns the `build`
# and `artifacts` blocks `schemas.validate_candidate` expects.
# =============================================================================

@dataclass(frozen=True)
class BuildIdentity:
    """WHAT was built, from WHICH source closure, with WHICH toolchain, producing WHICH binary.

    Every field is required. `integrity.BuildProvenance` makes the same choice
    for the same stated reason — *"a defaulted attestation is an attestation
    nobody made, and this record's whole job is to be an attestation somebody
    made"* — and a receipt that silently defaults its toolchain is a receipt that
    cannot answer the only question anyone will ever ask it.
    """

    candidate_id: str
    campaign_id: str
    worktree_record: dict
    snapshot_sha256: str
    snapshot_file_count: int
    snapshot_total_bytes: int
    source_root: str
    build_dir: str
    build_dir_created_for_this_build: bool
    build_dir_pre_build_digest: str
    actor_worktree: str
    production_tree_paths: tuple
    toolchain: str
    compiler: str
    configure_command: str
    command: str
    cmake_defines: tuple
    parallelism: dict
    build_log_path: str
    build_log_sha256: str
    output_binary_path: str
    output_binary_sha256: str
    library_sha256s: tuple
    incremental_output_binary_sha256: Optional[str]
    sandbox_receipts: tuple
    log_facts: BuildLogFacts
    exit_code: Optional[int]
    duration_s: float
    created_at: str
    linkage_sha256: Optional[str] = None
    notes: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("akc-"):
            raise ValueError("BuildIdentity.candidate_id must start with 'akc-'")
        validate_campaign_id(self.campaign_id)
        _req_sha256(self.snapshot_sha256, "snapshot_sha256")
        _req_tree_digest(self.build_dir_pre_build_digest, "build_dir_pre_build_digest")
        _req_sha256(self.build_log_sha256, "build_log_sha256")
        _req_sha256(self.output_binary_sha256, "output_binary_sha256")
        if self.incremental_output_binary_sha256 is not None:
            _req_sha256(self.incremental_output_binary_sha256,
                        "incremental_output_binary_sha256")
        for name in ("source_root", "build_dir", "actor_worktree"):
            value = _req_str(getattr(self, name), f"BuildIdentity.{name}")
            if not value.startswith("/"):
                raise ValueError(f"BuildIdentity.{name} must be absolute, got {value!r}")
            if any(p in (".", "..") for p in _components(value)):
                raise ValueError(
                    f"BuildIdentity.{name} {value!r} is unnormalized; "
                    "integrity.BuildProvenance refuses it at the door and it cannot be "
                    "tested for containment")
        for name in ("toolchain", "compiler", "command", "configure_command",
                     "build_log_path", "output_binary_path"):
            _req_str(getattr(self, name), f"BuildIdentity.{name}")
        phases = []
        for row in self.sandbox_receipts:
            if not isinstance(row, Mapping):
                raise TypeError("BuildIdentity.sandbox_receipts entries must be mappings")
            phase = row.get("phase")
            if phase not in ("configure", "build"):
                raise ValueError(f"unknown build sandbox phase {phase!r}")
            if not isinstance(row.get("activation"), Mapping) \
                    or not isinstance(row.get("teardown"), Mapping):
                raise ValueError(
                    f"build sandbox phase {phase} needs activation and teardown receipts")
            if row["activation"].get("sandbox_id") != process_sandbox.SANDBOX_ID:
                raise ValueError(f"build sandbox phase {phase} names another implementation")
            if row["activation"].get("writable_root") != self.build_dir:
                raise ValueError(f"build sandbox phase {phase} names another writable root")
            if not row["teardown"].get("verified_empty") \
                    or not row["teardown"].get("removed"):
                raise ValueError(f"build sandbox phase {phase} teardown is not complete")
            phases.append(phase)
        if sorted(phases) != ["build", "configure"]:
            raise ValueError("BuildIdentity requires configure and build sandbox receipts")

    # -- projections -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "receipt": "autokernel.execution.worktree.build_identity/v1",
            "candidate_id": self.candidate_id, "campaign_id": self.campaign_id,
            "worktree": dict(self.worktree_record),
            "source_snapshot": {"snapshot_sha256": self.snapshot_sha256,
                                "file_count": self.snapshot_file_count,
                                "total_bytes": self.snapshot_total_bytes},
            "source_root": self.source_root, "build_dir": self.build_dir,
            "build_dir_created_for_this_build": self.build_dir_created_for_this_build,
            "build_dir_pre_build_digest": self.build_dir_pre_build_digest,
            "actor_worktree": self.actor_worktree,
            "production_tree_paths": list(self.production_tree_paths),
            "toolchain": self.toolchain, "compiler": self.compiler,
            "configure_command": self.configure_command, "command": self.command,
            "cmake_defines": [list(d) for d in self.cmake_defines],
            "parallelism": dict(self.parallelism),
            "build_log_path": self.build_log_path,
            "build_log_sha256": self.build_log_sha256,
            "output_binary_path": self.output_binary_path,
            "output_binary_sha256": self.output_binary_sha256,
            "library_sha256s": [list(x) for x in self.library_sha256s],
            "sandbox_receipts": [
                {"phase": row["phase"],
                 "activation": dict(row["activation"]),
                 "teardown": dict(row["teardown"])}
                for row in self.sandbox_receipts],
            "linkage_sha256": self.linkage_sha256,
            "incremental_output_binary_sha256": self.incremental_output_binary_sha256,
            "log_facts": self.log_facts.to_dict(),
            "exit_code": self.exit_code, "duration_s": round(self.duration_s, 3),
            "created_at": self.created_at, "notes": list(self.notes),
        }

    @property
    def content_hash(self) -> str:
        return schemas.content_hash(self.to_dict())

    def to_build_provenance(self) -> integrity.BuildProvenance:
        """The record `integrity.check_clean_build_from_snapshot` already takes.

        This is the whole reason `integrity.py` was read before this module was
        written: producing a shape it would have to be adapted to would have made
        the §8.5.1 gate depend on an adapter nobody owns.
        """
        return integrity.BuildProvenance(
            candidate_id=self.candidate_id,
            snapshot_sha256=self.snapshot_sha256,
            source_root=self.source_root,
            build_dir=self.build_dir,
            build_dir_created_for_this_build=self.build_dir_created_for_this_build,
            build_dir_pre_build_digest=self.build_dir_pre_build_digest,
            actor_worktree=self.actor_worktree,
            production_tree_paths=tuple(self.production_tree_paths),
            toolchain=self.toolchain, compiler=self.compiler, command=self.command,
            build_log_path=self.build_log_path,
            build_log_sha256=self.build_log_sha256,
            output_binary_sha256=self.output_binary_sha256,
            incremental_output_binary_sha256=self.incremental_output_binary_sha256)

    def to_candidate_records(self) -> dict:
        """The `worktree`, `source_snapshot`, `build` and `artifacts` blocks of §7.3.

        Two fields the schema requires are deliberately ABSENT when this module
        does not know them, rather than filled with a plausible-looking digest:

        * `source_snapshot.patch_bundle_sha256` — this module never sees the
          patch bundle;
        * `artifacts.linkage_sha256` — the linkage proof comes from
          `epyc-inference-research/scripts/utils/verify_ggml_linkage.sh`, and it
          is not a build detail. CLAUDE.md: the three kernel trees run three ggml
          generations, so *"a binary that inherits another tree's ggml runs
          silently wrong"*. A digest invented here would attest to a linkage
          nobody checked.

        `schemas.validate_candidate` then names the gap by field. That is the
        intended behaviour — record the gap, do not patch it — and it is what
        makes the caller supply the real value instead of inheriting a fake one.
        """
        artifacts: dict = {"binary_sha256": self.output_binary_sha256,
                           "library_sha256s": dict(self.library_sha256s)}
        if self.linkage_sha256 is not None:
            artifacts["linkage_sha256"] = self.linkage_sha256
        return {
            "worktree": dict(self.worktree_record),
            "source_snapshot": {"snapshot_sha256": self.snapshot_sha256},
            "build": {"toolchain": self.toolchain, "compiler": self.compiler,
                      "command": self.command, "build_dir": self.build_dir,
                      "log_path": self.build_log_path,
                      "log_sha256": self.build_log_sha256,
                      "sandbox_receipts": [
                          {"phase": row["phase"],
                           "activation": dict(row["activation"]),
                           "teardown": dict(row["teardown"])}
                          for row in self.sandbox_receipts]},
            "artifacts": artifacts,
        }


def _compiler_identity(facts: BuildLogFacts) -> Optional[str]:
    """`GNU 15.2.0` from `-- The CXX compiler identification is GNU 15.2.0`.

    CXX first, then C, then whatever the log offered. The C++ compiler is the
    one that builds the kernels; reporting the C compiler when the two differ
    would name the wrong toolchain in the receipt.
    """
    by_lang = dict(facts.compiler_ids)
    for lang in ("CXX", "HIP", "CUDA", "C"):
        if lang in by_lang:
            return f"{lang} {by_lang[lang]}"
    return None


def build_identity(result: BuildResult, *, candidate_id: str, campaign_id: str,
                   worktree: Worktree, snapshot: integrity.TreeDigest,
                   output_binary: Any, toolchain: str,
                   compiler: Optional[str] = None,
                   libraries: Mapping[str, Any] = (),
                   linkage_sha256: Optional[str] = None,
                   incremental_output_binary_sha256: Optional[str] = None,
                   production_trees: Sequence[str] = (),
                   notes: Sequence[str] = ()) -> BuildIdentity:
    """Assemble the receipt from a `BuildResult` and the facts only the caller has.

    `compiler` falls back to the log's own compiler identification rather than to
    a string this function makes up. If the log did not say and the caller did
    not either, that is a `ValueError` — a receipt whose `compiler` field is
    `"unknown"` passes every schema and answers nothing.

    `production_trees` ADDS to the frozen set; it cannot shrink below it.
    `integrity.check_clean_build_from_snapshot` sub-check (e) tests
    `build_dir`/`source_root` for containment **in the list this receipt
    carries** — so the producer used to be able to hand the gate an empty
    denylist and pass it vacuously. Verified 2026-08-03: a `BuildProvenance`
    with `build_dir=/mnt/raid0/llm/llama.cpp/build` and
    `production_tree_paths=()` returns PASS on `no_production_tree_build`. A
    gate whose denylist is supplied by the party being gated is not a gate, and
    the producer is the side that can be fixed without editing the evaluator.
    """
    if not isinstance(result, BuildResult):
        raise TypeError("build_identity takes a BuildResult")
    if not isinstance(snapshot, integrity.TreeDigest):
        raise TypeError("build_identity(snapshot=...) takes an integrity.TreeDigest")
    binary = _real(output_binary, "output_binary")
    build_dir = result.plan.build_dir.path
    if not _is_within(binary, build_dir):
        raise ArtifactNotFromThisBuild(
            f"output_binary {binary!r} is not under the build directory {build_dir!r}. "
            "This function hashes the file it is handed: given a path outside the build "
            "it would emit a receipt attesting that THIS build produced THAT binary, with "
            "a real digest of a real file — every field true and the record false. The "
            "artifact of a build is what the build wrote")
    for name, path in dict(libraries).items():
        lib = _real(path, f"libraries[{name}]")
        if not _is_within(lib, build_dir):
            raise ArtifactNotFromThisBuild(
                f"library {name!r} at {lib!r} is not under the build directory "
                f"{build_dir!r}; a receipt may not attest a library this build did not "
                "produce (CLAUDE.md: a binary that inherits another tree's ggml runs "
                "silently wrong)")
    resolved_compiler = compiler or _compiler_identity(result.facts)
    if not resolved_compiler:
        raise ValueError(
            "compiler identity is unknown: the build log carried no '-- The CXX compiler "
            "identification is …' line (a re-used cmake cache does not re-print it) and no "
            "compiler was passed. Refusing to write a receipt that cannot say what built it")
    libs = tuple(sorted((name, _sha256_file(path)) for name, path in dict(libraries).items()))
    durations = [d.duration_s for d in (result.configure, result.build) if d is not None]
    sandbox_rows = []
    for phase, disposition in (("configure", result.configure), ("build", result.build)):
        if disposition is None or disposition.sandbox_receipt is None \
                or disposition.sandbox_teardown is None:
            raise ValueError(
                f"cannot identify a candidate build without verified {phase} sandbox "
                "activation and teardown receipts")
        sandbox_rows.append({
            "phase": phase,
            "activation": dict(disposition.sandbox_receipt),
            "teardown": dict(disposition.sandbox_teardown),
        })
    extra = list(notes)
    if result.facts.ccache_enabled:
        extra.append(
            "ccache was ACTIVE during this build: objects may come from a cache populated "
            "by another tree, so a fresh build directory does not by itself make this a "
            "clean build (§8.5.1 (2))")
    if result.facts.ggml_commit_dirty:
        extra.append(
            f"ggml commit reported as {result.facts.ggml_commit!r} — the '-dirty' suffix "
            "means the tree that built had uncommitted changes, so the snapshot digest is "
            "not the thing that built")
    if result.plan.parallelism.load_average_cap is not None:
        observed = ("not recorded" if result.load_average_at_start is None
                    else f"{result.load_average_at_start:.2f}")
        extra.append(
            f"declared load-average cap {result.plan.parallelism.load_average_cap:.2f}; "
            f"1-minute load at build start was {observed} (run_build checks the cap before "
            "configure — a cap that is only recorded is not a cap)")
    if result.log_disagrees_with_exit_code:
        extra.append(
            f"exit code {result.exit_code} disagrees with the log "
            f"(succeeded_by_log={result.facts.succeeded_by_log}); the status may have come "
            "from something other than the compiler")

    return BuildIdentity(
        candidate_id=candidate_id, campaign_id=campaign_id,
        worktree_record=worktree.to_record(),
        snapshot_sha256=snapshot.sha256,
        snapshot_file_count=snapshot.file_count,
        snapshot_total_bytes=snapshot.total_bytes,
        source_root=result.plan.source_root.path,
        build_dir=result.plan.build_dir.path,
        build_dir_created_for_this_build=result.build_dir_created_for_this_build,
        build_dir_pre_build_digest=result.build_dir_pre_build_digest,
        actor_worktree=result.plan.actor_worktree.path,
        production_tree_paths=tuple(sorted(
            set(frozen_tree_paths()) | {_req_str(t, "production_trees[]")
                                        for t in production_trees})),
        toolchain=_req_str(toolchain, "toolchain"), compiler=resolved_compiler,
        configure_command=" ".join(result.plan.configure_argv()),
        command=" ".join(result.plan.build_argv()),
        cmake_defines=result.plan.effective_defines,
        parallelism=result.plan.parallelism.to_dict(),
        build_log_path=result.log_path, build_log_sha256=result.log_sha256,
        output_binary_path=binary, output_binary_sha256=_sha256_file(binary),
        library_sha256s=libs, linkage_sha256=linkage_sha256,
        incremental_output_binary_sha256=incremental_output_binary_sha256,
        sandbox_receipts=tuple(sandbox_rows),
        log_facts=result.facts, exit_code=result.exit_code,
        duration_s=sum(durations), created_at=_utc_now_iso(), notes=tuple(extra))


# =============================================================================
# Self-audit — the process-discipline property, proven from this module's AST
#
# `evaluator/integrity.py` proves it CANNOT spawn. This module must spawn, so it
# proves the weaker but load-bearing property instead: it only ever signals a
# process group it created, it never invokes a name-pattern process tool, and it
# never asks a shell to re-interpret an argv it validated.
# =============================================================================

#: The only expression allowed as `os.killpg`'s first argument. It is a local
#: assigned from `os.getpgid(proc.pid)` on a child started with
#: `start_new_session=True`, so the group is exactly the child we launched.
_OWNED_PGID_NAMES = ("pgid",)


def _module_source() -> str:
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as handle:
        return handle.read()


def _dotted(node: ast.AST) -> str:
    parts: list = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def audit_no_name_pattern_process_ops(source: Optional[str] = None, *,
                                      denylist: Sequence[str] = _NAME_PATTERN_BINARIES
                                      ) -> schemas.Check:
    """Prove from the AST that this module cannot do a name-pattern process op.

    Four findings, each with a test that fails without it:

    * `NAME_PATTERN_BINARY_IN_ARGV` — a `pkill`/`pgrep`/`killall`/… literal used
      as a command. INC-20260731 in one rule.
    * `UNOWNED_SIGNAL` — `os.kill(...)` at all, or `os.killpg(x, …)` where `x`
      is not the owned-pgid local. Signalling something we did not launch is
      denial 8's second clause.
    * `SHELL_TRUE` — `shell=True` anywhere; it hands a validated argv back to a
      word-splitter.
    * `SUBPROCESS_OUTSIDE_RUN_OWNED` — a `subprocess.*` call outside
      `_run_owned` or the closed `_run_guarded_patch_input` route.

    The denylist is a PARAMETER so a test can pass a shorter one and watch a
    finding disappear; `test_worktree.py` separately asserts the module default
    still contains every entry, because an audit whose denylist can be emptied is
    an audit you pass by deleting what it inspects.
    """
    text = _module_source() if source is None else source
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:  # pragma: no cover - the module imported
        return schemas.Check(COULD_NOT_CHECK, reasons=(f"AST_PARSE_FAILED: {exc}",))

    forbidden = {name for name in denylist}
    findings: list = []

    # The module-level denylist assignment is the one place these names may
    # appear as literals. It is exempted by NODE, not by value, so a literal
    # smuggled in anywhere else is still caught.
    exempt_nodes = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_NAME_PATTERN"):
                    for sub in ast.walk(node.value):
                        exempt_nodes.add(id(sub))

    # `_validate_argv` refuses these at runtime and names them in its message;
    # that guard's own comparison must not read as a violation of itself.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_argv":
            for sub in ast.walk(node):
                exempt_nodes.add(id(sub))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in exempt_nodes:
                continue
            if node.value in forbidden:
                findings.append(
                    f"NAME_PATTERN_BINARY_IN_ARGV: string literal {node.value!r} at line "
                    f"{node.lineno}")
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name == "os.kill":
                findings.append(
                    f"UNOWNED_SIGNAL: os.kill at line {node.lineno}; signal the process "
                    "GROUP this module created, never a bare pid")
            if name == "os.killpg":
                first = node.args[0] if node.args else None
                if not (isinstance(first, ast.Name) and first.id in _OWNED_PGID_NAMES):
                    findings.append(
                        f"UNOWNED_SIGNAL: os.killpg at line {node.lineno} whose first "
                        f"argument is not one of {list(_OWNED_PGID_NAMES)}")
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) \
                        and kw.value.value is True:
                    findings.append(f"SHELL_TRUE: shell=True at line {node.lineno}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name in {
                "_run_owned", "_run_guarded_patch_input"}:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and _dotted(sub.func).startswith("subprocess."):
                if _dotted(sub.func) in ("subprocess.TimeoutExpired",):
                    continue
                findings.append(
                    f"SUBPROCESS_OUTSIDE_RUN_OWNED: {_dotted(sub.func)} in {node.name} at "
                    f"line {sub.lineno}")

    if findings:
        return schemas.Check(FAIL, reasons=tuple(sorted(set(findings))))
    return schemas.Check(PASS, reasons=(
        "no name-pattern process tool in any argv; every signal targets the process group "
        "this module created; no shell=True; subprocess is reached only through "
        "_run_owned or the immutable guarded patch-input route",))


def with_parallelism(plan: BuildPlan, parallelism: BuildParallelism) -> BuildPlan:
    """A copy of `plan` at a different width — the polite-rebuild path.

    Re-running `__post_init__` through `dataclasses.replace` is the point: a
    variant plan is validated exactly as hard as an original one.

    Spelled `dataclasses.replace`, never a bare `replace` imported from it.
    `release/test_release_integration._DENIED_ATTRS` lists `replace` because
    `Path.replace` is the move-a-stable-kernel-symlink primitive and an AST
    cannot tell the two apart; `dataclasses.replace` is that denylist's one
    exempt spelling. Using the exempt spelling keeps the audit precise instead of
    making it argue about this module.
    """
    if not isinstance(parallelism, BuildParallelism):
        raise TypeError("with_parallelism takes a BuildParallelism")
    return dataclasses.replace(plan, parallelism=parallelism)
