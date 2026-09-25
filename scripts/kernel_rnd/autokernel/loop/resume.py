#!/usr/bin/env python3
"""Resume abandoned candidates at the stage they reached, instead of re-deriving them.

    # from the repo root; census imports `autokernel.*`, so scripts/kernel_rnd too
    PYTHONPATH=scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.resume \\
        backfill --store <store> --row <attempt-id-prefix> [--repo <tree>] \\
        [--scratch <dir>] [--apply]
    PYTHONPATH=scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.resume \\
        scan --store <store> --epoch <sha> --anchor <sha> [--surface S] [--model M] \\
        [--repo <tree>]

WHY. Stops and refusals discarded the work in flight. Across DS41 runs 3-9c that
was the largest drop class: ~300 actor-minutes and ~120 measure-minutes, and 0 of
~15 such attempts ever reached a build. Run 9c is the concrete case: the critic
accepted an actscale hoist twice, `op_scope` refused it twice under a rule that was
wrong (fixed in c7215eb5), and a stop then landed. The patches were retained under
<store>/patches, but nothing could take them to the build.

THE RECORD. Every row that ends with accepted work still in flight carries
`resume_checkpoints` (schema `loop.CHECKPOINT_SCHEMA`):

    stage                   "build"  -- a critic-accepted patch that never reached a
                                        measurement (refused by a gate, or stopped)
                            "author" -- a critic-accepted hypothesis whose authoring
                                        was interrupted by a stop or a transient
    hypothesis              the exact Hypothesis.to_dict()
    critic_hypothesis       the accepted critic:hypothesis provenance row
    critic_patch            (build) the accepted critic:patch provenance row
    retained_patch          (build) {patch_file, metadata_file, patch_sha256}
    refusal_gate / refusal_reason / gate_rules_fingerprint   (build, gate refusals)
    prior_patch_rejections / patch_rounds_remaining          (author)
    hypothesis_round / patch_round
    resumed_from / resume_depth                              lineage
    anchor_commit / epoch_sha256 / target                    bound by the owner

THE ALGORITHM (`prepare`), on every launch, before any fresh hypothesis is drawn:

 1. scan the store for checkpoints in the CURRENT epoch; drop any already claimed
    at this anchor (`<store>/resume-claims.sqlite3`);
 2. eligibility: a build checkpoint needs a retained patch and a refusal by a RULE
    gate (`RULE_GATES`) whose rules have changed since (or were never recorded);
    an author checkpoint needs a patch round left. Ineligible ones stay unclaimed,
    so a later rule change can still reopen them;
 3. group by hypothesis; per group, try the most advanced first (build > author,
    then newest): re-validate it -- anchor, epoch and target match, chain depth,
    the carried critic verdicts, and for a build the patch bytes against the row's
    and the sidecar's sha256 and a clean apply at the anchor. A failure is recorded
    as `resume_rejected` (its own row, and a claim) and the next sibling is tried;
 4. the queue hands each survivor to a lane on its next iteration (`take`), which
    CLAIMS it first (at most once per anchor; siblings become `superseded`), so a
    crash mid-resume cannot duplicate it. The lane re-verifies and applies the
    bytes, and `loop.iterate` re-runs the host integrity check and every CURRENT
    gate -- the old verdict is never trusted -- before any build or measurement.

The backfill turns a row that predates checkpoints (run 9c's 22d950a4...) plus its
retained patches into one resumable `gate_refused` row. It is a tool the operator's
session runs; `--apply` is the only write, and it only appends.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Mapping, Sequence

from ..controller import experiments
from . import loop

CLAIMS_FILE = "resume-claims.sqlite3"
BACKFILL_SCHEMA = "epyc.autokernel.resume_backfill.v1"
PATCH_ARCHIVE_SCHEMA = "epyc.autokernel.source_patch_archive.v1"
STAGE_RANK = {"build": 2, "author": 1}
#: Deterministic pre-build rule gates. A refusal by one of these is a verdict of the
#: RULE, not of the patch, so it is resumable once the rule changes. A compile or
#: correctness failure is the patch's own and is never resumed at build.
RULE_GATES = frozenset({"op_scope"})
#: A stop during a resumed candidate writes a new checkpoint; bound the chain.
MAX_RESUME_DEPTH = 3
SCANNED_STATUSES = ("gate_refused", "stopped_mid_formation", "planner_transient")
MAX_PATCH_BYTES = 4 * 1024 * 1024
CLAIM_STATES = ("resumed", "rejected", "superseded")


class ResumeRejected(RuntimeError):
    """A resumable candidate failed re-validation; `check` names the rule."""

    def __init__(self, check: str, reason: str):
        super().__init__(reason)
        self.check = check


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------------ checkpoint binding


def target_identity(*, measurement_surface: str | None, model: str | Path | None) -> dict:
    """The target a checkpoint is bound to: what the candidate was measured FOR."""
    return {"measurement_surface": None if measurement_surface is None
            else str(measurement_surface),
            "model": None if model is None else str(model)}


def bind_checkpoints(attempt: dict, *, epoch: str, anchor_commit: str,
                     target: Mapping[str, Any]) -> dict:
    """Owner-side: stamp where each checkpoint on this row may be resumed.

    The anchor is the lane base the candidate was formed on (`spawn_parent`) when
    the pool recorded one, else the current anchor. A build checkpoint whose patch
    pointer the loop could not know yet (the owner retained it while recording)
    takes the row's `retained_patch`.
    """
    checkpoints = attempt.get("resume_checkpoints")
    if not isinstance(checkpoints, list):
        return attempt
    anchor = attempt.get("spawn_parent") or anchor_commit
    for entry in checkpoints:
        if not isinstance(entry, dict):
            continue
        entry["anchor_commit"] = anchor
        entry["epoch_sha256"] = epoch
        entry["target"] = dict(target)
        if entry.get("stage") == "build" and not entry.get("retained_patch"):
            pointer = attempt.get("retained_patch")
            if isinstance(pointer, dict) and pointer.get("patch_sha256"):
                entry["retained_patch"] = dict(pointer)
    return attempt


# ------------------------------------------------------------------ claims ledger


class ClaimLedger:
    """At-most-once per (checkpoint, anchor). The PRIMARY KEY is the whole guarantee.

    A claim is taken BEFORE a resumed candidate touches a lane, so a crash
    mid-resume leaves it consumed rather than duplicated; `settle` adds the result.
    """

    def __init__(self, store_root: Path, *, read_only: bool = False) -> None:
        self.path = Path(store_root) / CLAIMS_FILE
        self.read_only = read_only
        self.db = None
        if read_only:
            if self.path.is_file():
                self.db = sqlite3.connect(self.path.resolve().as_uri() + "?immutable=1",
                                          uri=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS claims ("
            "checkpoint_id TEXT NOT NULL, anchor_commit TEXT NOT NULL, "
            "state TEXT NOT NULL, stage TEXT, epoch_sha256 TEXT, mechanism_id TEXT, "
            "source_attempt_id TEXT, claimed_at TEXT NOT NULL, detail TEXT, "
            "result_status TEXT, settled_at TEXT, "
            "PRIMARY KEY (checkpoint_id, anchor_commit))")
        self.db.commit()

    def close(self) -> None:
        if self.db is not None:
            self.db.close()

    def __enter__(self) -> "ClaimLedger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def claimed(self, anchor_commit: str) -> set[str]:
        if self.db is None:
            return set()
        return {row[0] for row in self.db.execute(
            "SELECT checkpoint_id FROM claims WHERE anchor_commit=?", (anchor_commit,))}

    def claim(self, checkpoint_id: str, anchor_commit: str, *, state: str,
              stage: str | None = None, epoch: str | None = None,
              mechanism_id: str | None = None, source_attempt_id: str | None = None,
              detail: str | None = None) -> bool:
        """True only for the one caller that took it."""
        if state not in CLAIM_STATES:
            raise ValueError(f"unknown claim state {state!r}")
        if self.read_only:
            raise RuntimeError("claim ledger opened read-only")
        try:
            self.db.execute("INSERT INTO claims (checkpoint_id, anchor_commit, state, stage, "
                            "epoch_sha256, mechanism_id, source_attempt_id, claimed_at, detail) "
                            "VALUES (?,?,?,?,?,?,?,?,?)",
                            (checkpoint_id, anchor_commit, state, stage, epoch, mechanism_id,
                             source_attempt_id, _now(), detail))
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            self.db.rollback()
            return False

    def settle(self, checkpoint_id: str, anchor_commit: str, *, result_status: str,
               detail: str | None = None) -> None:
        if self.read_only:
            raise RuntimeError("claim ledger opened read-only")
        self.db.execute("UPDATE claims SET result_status=?, settled_at=?, "
                        "detail=coalesce(?, detail) WHERE checkpoint_id=? AND anchor_commit=?",
                        (result_status, _now(), detail, checkpoint_id, anchor_commit))
        self.db.commit()

    def rows(self) -> list[dict]:
        if self.db is None:
            return []
        cursor = self.db.execute("SELECT * FROM claims ORDER BY claimed_at")
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


# ------------------------------------------------------------------ patches


def _git_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(name, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    if extra:
        env.update(extra)
    return env


def _git(repo: Path, *args: str, input_bytes: bytes | None = None,
         check: bool = True, env: Mapping[str, str] | None = None) -> bytes:
    done = subprocess.run(["git", "-C", str(repo), *args], input=input_bytes,
                          capture_output=True, timeout=600, env=_git_env(env))
    if check and done.returncode != 0:
        raise ResumeRejected("git", f"git {' '.join(args[:2])} failed: "
                             f"{done.stderr.decode('utf-8', 'replace').strip()[:400]}")
    return done.stdout


_SAFE_PATH = re.compile(r"[A-Za-z0-9_./+-]+")


def patch_paths(patch: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(touched paths, paths that must pre-exist at the anchor) from a git diff."""
    touched: list[str] = []
    preexisting: list[str] = []
    current: list[str] | None = None
    for raw in patch.decode("utf-8", "surrogateescape").splitlines():
        if raw.startswith("diff --git "):
            match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", raw)
            if match is None:
                raise ResumeRejected("patch_paths", f"unparseable diff header: {raw[:200]}")
            current = [match.group(1), match.group(2)]
            for name in match.groups():
                parts = name.split("/")
                if not _SAFE_PATH.fullmatch(name) or any(
                        part in {"", ".", ".."} or part.startswith(".") for part in parts):
                    raise ResumeRejected("patch_paths", f"unsafe path in patch: {name!r}")
            touched.append(match.group(2))
            preexisting.append(match.group(1))
        elif current is not None and raw.startswith("new file mode"):
            preexisting.pop()
            current = None
    if not touched:
        raise ResumeRejected("patch_paths", "retained patch touches no file")
    return tuple(dict.fromkeys(touched)), tuple(dict.fromkeys(preexisting))


def _read_bounded(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ResumeRejected("patch_file", f"retained patch is not a regular file: {path}")
        raw = stream.read(MAX_PATCH_BYTES + 1)
    if len(raw) > MAX_PATCH_BYTES:
        raise ResumeRejected("patch_file", f"retained patch exceeds {MAX_PATCH_BYTES} bytes")
    return raw


def verify_retained_patch(pointer: Mapping[str, Any], *, anchor_commit: str,
                          mechanism_id: str | None) -> bytes:
    """The exact accepted bytes, or a refusal naming which binding failed."""
    if not isinstance(pointer, Mapping) or not pointer.get("patch_file") \
            or not pointer.get("patch_sha256"):
        raise ResumeRejected("patch_pointer", "checkpoint carries no retained patch pointer")
    patch_path = Path(pointer["patch_file"])
    sidecar_path = Path(pointer.get("metadata_file") or patch_path.with_suffix(".json"))
    try:
        raw = _read_bounded(patch_path)
    except FileNotFoundError:
        raise ResumeRejected("patch_file", f"retained patch is missing: {patch_path}") from None
    actual = _sha256(raw)
    if actual != pointer["patch_sha256"]:
        raise ResumeRejected("patch_sha256", f"retained patch bytes changed: sha256 {actual} "
                             f"!= recorded {pointer['patch_sha256']}")
    try:
        sidecar = json.loads(_read_bounded(sidecar_path))
    except FileNotFoundError:
        raise ResumeRejected("sidecar", f"patch sidecar is missing: {sidecar_path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeRejected("sidecar", f"patch sidecar is unreadable: {exc}") from None
    if not isinstance(sidecar, dict) or sidecar.get("schema") != PATCH_ARCHIVE_SCHEMA:
        raise ResumeRejected("sidecar", "patch sidecar schema is not the source patch archive")
    if sidecar.get("patch_sha256") != actual:
        raise ResumeRejected("patch_sha256", f"patch sha256 {actual} != sidecar "
                             f"{sidecar.get('patch_sha256')}")
    if sidecar.get("patch_file") not in (None, patch_path.name):
        raise ResumeRejected("sidecar", "patch sidecar names a different patch file")
    if sidecar.get("original_head") != anchor_commit:
        raise ResumeRejected("anchor", f"patch was retained on {sidecar.get('original_head')}, "
                             f"not the anchor {anchor_commit}")
    if mechanism_id is not None and sidecar.get("mechanism_id") not in (None, mechanism_id):
        raise ResumeRejected("sidecar", f"patch sidecar names mechanism "
                             f"{sidecar.get('mechanism_id')}, not {mechanism_id}")
    return raw


def _scratch_tree(repo: Path, anchor_commit: str, patch: bytes,
                  scratch: Path | None) -> tuple[tempfile.TemporaryDirectory, Path]:
    """Pre-image files at the anchor, copied out by object reads only.

    The source repository is never checked out, indexed or written: this is a
    READ of the anchor's blobs into a scratch directory outside any repository.
    """
    _touched, preexisting = patch_paths(patch)
    holder = tempfile.TemporaryDirectory(prefix=".resume-apply-",
                                         dir=None if scratch is None else str(scratch))
    root = Path(holder.name)
    for name in preexisting:
        probe = subprocess.run(["git", "-C", str(repo), "cat-file", "-e",
                                f"{anchor_commit}:{name}"], capture_output=True,
                               timeout=60, env=_git_env())
        if probe.returncode != 0:
            holder.cleanup()
            raise ResumeRejected("patch_apply", f"{name} does not exist at the anchor "
                                 f"{anchor_commit[:12]}")
        blob = _git(repo, "cat-file", "blob", f"{anchor_commit}:{name}")
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    return holder, root


def _apply_outside_repo(root: Path, patch: bytes, *extra: str) -> subprocess.CompletedProcess:
    # GIT_CEILING_DIRECTORIES stops discovery at the scratch root's parent, so
    # `git apply` runs as a plain patch tool and never finds an enclosing repo.
    return subprocess.run(["git", "apply", "--binary", *extra, "-"], cwd=root, input=patch,
                          capture_output=True, timeout=600,
                          env=_git_env({"GIT_CEILING_DIRECTORIES": str(root.parent)}))


def patch_applies(repo: Path, anchor_commit: str, patch: bytes, *,
                  scratch: Path | None = None) -> None:
    """Refuse unless the retained patch applies cleanly to the anchor's tree."""
    holder, root = _scratch_tree(Path(repo), anchor_commit, patch, scratch)
    try:
        done = _apply_outside_repo(root, patch, "--check")
        if done.returncode != 0:
            raise ResumeRejected("patch_apply", "retained patch no longer applies at the "
                                 f"anchor {anchor_commit[:12]}: "
                                 f"{done.stderr.decode('utf-8', 'replace').strip()[:400]}")
    finally:
        holder.cleanup()


def preview_op_scope(repo: Path, anchor_commit: str, patch: bytes, hypothesis: Mapping,
                     *, scratch: Path | None = None) -> dict:
    """Diagnostic replay of the CURRENT pre-build `op_scope` rule on a retained patch.

    Mirrors `run.py`'s gate inputs (post-image text, anchor text, -U0 hunks) in a
    scratch copy. It is a preview for the operator, never an admission: the resumed
    candidate re-runs the real gate in its lane before any build.
    """
    from . import gates
    touched, _pre = patch_paths(patch)
    iqk = {"ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp", "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"}
    if not (len(touched) == 1 and (touched[0] in iqk or touched[0] == "ggml/src/ggml-cpu/ops.cpp")):
        scope = gates.affected_op_scope(touched, target_surface=hypothesis["target_surface"],
                                        target_symbol=hypothesis["target_symbol"])
    else:
        holder, root = _scratch_tree(Path(repo), anchor_commit, patch, scratch)
        try:
            name = touched[0]
            before = (root / name).read_text(encoding="utf-8")
            done = _apply_outside_repo(root, patch)
            if done.returncode != 0:
                return {"passed": False, "reason": "patch does not apply at the anchor"}
            after = (root / name).read_text(encoding="utf-8")
            (root / "pre.cpp").write_text(before, encoding="utf-8")
            diff = subprocess.run(["git", "diff", "--no-index", "-U0", "pre.cpp", name],
                                  cwd=root, capture_output=True, timeout=600,
                                  env=_git_env({"GIT_CEILING_DIRECTORIES": str(root.parent)}))
            scope = gates.affected_op_scope(
                touched, target_surface=hypothesis["target_surface"],
                target_symbol=hypothesis["target_symbol"], source_text=after,
                pre_source_text=before, patch_text=diff.stdout.decode("utf-8", "replace"))
        finally:
            holder.cleanup()
    if isinstance(scope, gates.Verdict):
        return {"passed": scope.passed, "gate": scope.gate, "reason": scope.reason}
    return {"passed": True, "gate": "op_scope", "scope": list(scope)}


def materialize(worktree: Path, patch: bytes, *, expected_base: str) -> tuple[str, ...]:
    """Apply the retained bytes to a freshly reset lane; return the touched paths."""
    worktree = Path(worktree)
    head = _git(worktree, "rev-parse", "HEAD").decode().strip()
    if head != expected_base:
        raise ResumeRejected("anchor", f"lane is at {head[:12]}, not the anchor "
                             f"{expected_base[:12]} the patch was formed on")
    if _git(worktree, "status", "--porcelain", "--untracked-files=all", "--",
            "ggml/src/", "src/").strip():
        raise ResumeRejected("lane_dirty", "lane is not clean before restoring the patch")
    check = subprocess.run(["git", "-C", str(worktree), "apply", "--check", "--binary", "-"],
                           input=patch, capture_output=True, timeout=600, env=_git_env())
    if check.returncode != 0:
        raise ResumeRejected("patch_apply", "retained patch does not apply to the lane: "
                             f"{check.stderr.decode('utf-8', 'replace').strip()[:400]}")
    _git(worktree, "apply", "--binary", "-", input_bytes=patch)
    touched, _pre = patch_paths(patch)
    return touched


def discard(worktree: Path, patch: bytes) -> None:
    """Reverse exactly the restored bytes (a rejected resume leaves no residue)."""
    _git(Path(worktree), "apply", "-R", "--binary", "-", input_bytes=patch)


# ------------------------------------------------------------------ scan and plan


@dataclass(frozen=True)
class Candidate:
    checkpoint_id: str
    attempt_id: str
    recorded_at: str
    row_status: str
    mechanism_id: str | None
    checkpoint: Mapping[str, Any]

    @property
    def stage(self) -> str:
        return str(self.checkpoint.get("stage"))

    @property
    def hypothesis(self) -> Mapping[str, Any]:
        value = self.checkpoint.get("hypothesis")
        return value if isinstance(value, Mapping) else {}

    @property
    def group(self) -> tuple[str | None, str]:
        return (self.hypothesis.get("mechanism_id") or self.mechanism_id,
                _sha256(_canonical(dict(self.hypothesis)).encode()))

    @property
    def rank(self) -> tuple:
        return (STAGE_RANK.get(self.stage, 0), self.recorded_at,
                int(self.checkpoint.get("patch_round") or 0))

    def summary(self) -> dict:
        pointer = self.checkpoint.get("retained_patch") or {}
        return {"checkpoint_id": self.checkpoint_id, "stage": self.stage,
                "mechanism_id": self.group[0], "row_status": self.row_status,
                "recorded_at": self.recorded_at,
                "patch_round": self.checkpoint.get("patch_round"),
                "patch_file": pointer.get("patch_file") if isinstance(pointer, Mapping) else None}


def _connect(store_root: Path, *, immutable: bool) -> sqlite3.Connection:
    path = Path(store_root) / "experiments.db"
    if not path.is_file():
        raise FileNotFoundError(f"no experiment store at {path}")
    mode = "immutable=1" if immutable else "mode=ro"
    connection = sqlite3.connect(path.resolve().as_uri() + "?" + mode, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def scan(store_root: Path, *, epoch: str, immutable: bool = False) -> list[Candidate]:
    """Every checkpoint recorded in this epoch, oldest first."""
    placeholders = ",".join("?" for _ in SCANNED_STATUSES)
    connection = _connect(store_root, immutable=immutable)
    try:
        rows = connection.execute(
            "SELECT attempt_id, recorded_at, status, mechanism_id, payload FROM experiments "
            f"WHERE epoch_sha256=? AND status IN ({placeholders}) "
            "AND instr(payload, '\"resume_checkpoints\"') > 0 ORDER BY recorded_at, rowid",
            (epoch, *SCANNED_STATUSES)).fetchall()
    finally:
        connection.close()
    found: list[Candidate] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            continue
        for index, entry in enumerate(payload.get("resume_checkpoints") or ()):
            if not isinstance(entry, dict) or entry.get("schema") != loop.CHECKPOINT_SCHEMA:
                continue
            entry = dict(entry)
            if entry.get("stage") == "build" and not entry.get("retained_patch") \
                    and isinstance(payload.get("retained_patch"), dict):
                entry["retained_patch"] = dict(payload["retained_patch"])
            found.append(Candidate(f"{row['attempt_id']}#{index}", row["attempt_id"],
                                   row["recorded_at"], row["status"], row["mechanism_id"],
                                   entry))
    return found


def other_epoch_checkpoints(store_root: Path, *, epoch: str, immutable: bool = False) -> int:
    """Rows carrying checkpoints in OTHER epochs: never resumed here, but a launch
    that expected to resume them (a changed screen scope or instrument moves the
    epoch) must say so rather than silently scanning zero."""
    placeholders = ",".join("?" for _ in SCANNED_STATUSES)
    try:
        connection = _connect(store_root, immutable=immutable)
    except FileNotFoundError:
        return 0
    try:
        return int(connection.execute(
            "SELECT count(*) FROM experiments WHERE epoch_sha256<>? "
            f"AND status IN ({placeholders}) "
            "AND instr(payload, '\"resume_checkpoints\"') > 0",
            (epoch, *SCANNED_STATUSES)).fetchone()[0])
    finally:
        connection.close()


def ineligible_reason(candidate: Candidate, *, rules_fingerprint: str) -> str | None:
    """Why this checkpoint is not resumable NOW (it stays unclaimed), or None."""
    ck = candidate.checkpoint
    if candidate.stage not in STAGE_RANK:
        return f"unknown stage {candidate.stage!r}"
    if not candidate.hypothesis.get("mechanism_id"):
        return "checkpoint carries no hypothesis"
    if candidate.stage == "build":
        pointer = ck.get("retained_patch")
        if not isinstance(pointer, Mapping) or not pointer.get("patch_sha256"):
            return "no retained patch"
        gate = ck.get("refusal_gate")
        if gate is not None and gate not in RULE_GATES:
            return f"refused by {gate}, a verdict on the patch rather than a rule"
        recorded = ck.get("gate_rules_fingerprint")
        if gate is not None and recorded is not None and recorded == rules_fingerprint:
            return f"{gate} rules unchanged since the refusal"
    elif int(ck.get("patch_rounds_remaining") or 0) < 1:
        return "no patch round left"
    return None


def prevalidate(candidate: Candidate, *, epoch: str, anchor_commit: str,
                target: Mapping[str, Any], repo: Path | None,
                scratch: Path | None = None) -> bytes | None:
    """Everything checkable before a lane is touched. Returns the patch for a build."""
    ck = candidate.checkpoint
    if int(ck.get("resume_depth") or 0) >= MAX_RESUME_DEPTH:
        raise ResumeRejected("depth", f"resume chain depth {ck.get('resume_depth')} reached "
                             f"the limit {MAX_RESUME_DEPTH}")
    formed = ck.get("anchor_commit")
    if not formed:
        raise ResumeRejected("anchor", "checkpoint names no anchor")
    if formed != anchor_commit:
        raise ResumeRejected("anchor", f"anchor changed: formed on {formed[:12]}, "
                             f"current {anchor_commit[:12]}")
    if ck.get("epoch_sha256") not in (None, epoch):
        raise ResumeRejected("epoch", f"epoch changed: {str(ck.get('epoch_sha256'))[:12]} "
                             f"!= {epoch[:12]}")
    recorded_target = ck.get("target") if isinstance(ck.get("target"), Mapping) else {}
    for key in ("measurement_surface", "model"):
        then, now = recorded_target.get(key), target.get(key)
        if then is not None and now is not None and str(then) != str(now):
            raise ResumeRejected("target", f"target {key} changed: {then} != {now}")
    try:
        hypothesis = loop.Hypothesis(**dict(candidate.hypothesis))
    except (TypeError, ValueError) as exc:
        raise ResumeRejected("hypothesis", f"checkpoint hypothesis is malformed: {exc}") from None
    if hypothesis.runtime_pair is not None:
        raise ResumeRejected("hypothesis", "runtime treatments are not resumable")
    if candidate.mechanism_id not in (None, hypothesis.mechanism_id):
        raise ResumeRejected("hypothesis", "checkpoint hypothesis differs from its row")
    verdict = ck.get("critic_hypothesis")
    if not isinstance(verdict, Mapping) or not verdict.get("accepted"):
        raise ResumeRejected("critic", "no accepted critic:hypothesis verdict is carried")
    if candidate.stage != "build":
        return None
    verdict = ck.get("critic_patch")
    if not isinstance(verdict, Mapping) or not verdict.get("accepted"):
        raise ResumeRejected("critic", "no accepted critic:patch verdict is carried")
    patch = verify_retained_patch(ck.get("retained_patch"), anchor_commit=anchor_commit,
                                  mechanism_id=hypothesis.mechanism_id)
    if repo is not None:
        patch_applies(repo, anchor_commit, patch, scratch=scratch)
    return patch


def rejection_attempt(candidate: Candidate, check: str, reason: str, *,
                      anchor_commit: str) -> dict:
    """The `resume_rejected` row. Its identity is deterministic, so a re-rejection
    after a crash between the row and the claim cannot duplicate it."""
    hypothesis = dict(candidate.hypothesis)
    row = {key: hypothesis.get(key) for key in (
        "mechanism_id", "statement", "falsifier", "target_surface", "target_symbol")}
    row.update({
        "status": loop.RESUME_REJECTED, "turn_recorded_at": _now(),
        "reason": f"resume re-validation refused ({check}): {reason}",
        "refusal_gate": f"resume:{check}", "resumed_from": candidate.checkpoint_id,
        "resume_stage": candidate.stage,
        # experiments._attempt_id prefers this key: one row per (checkpoint, anchor).
        "proposal_sha256": _sha256(
            f"resume_rejected\n{candidate.checkpoint_id}\n{anchor_commit}".encode()),
    })
    return row


@dataclass
class ResumePoint:
    """One claimed candidate, handed to `loop.iterate(resume=...)`."""
    checkpoint_id: str
    stage: str
    hypothesis: Any
    checkpoint: Mapping[str, Any]
    provenance: tuple[dict, ...]
    prior_patch_rejections: tuple[str, ...]
    patch_rounds: int
    _materialize: Callable[[], Sequence[str]] | None = None
    _discard: Callable[[], None] | None = None
    #: (check, reason) when the point failed re-validation at hand-out time. The
    #: loop disposes it as `resume_rejected` through the owner's ordinary recorder
    #: (under the pool's record lock) and draws fresh work.
    stale: tuple[str, str] | None = None

    @property
    def label(self) -> str:
        return (f"resuming {self.hypothesis.mechanism_id} at {self.stage} "
                f"(from {self.checkpoint_id})")

    def materialize(self) -> Sequence[str]:
        if self._materialize is None:
            raise ResumeRejected("stage", f"a {self.stage} resume has no patch to restore")
        return self._materialize()

    def discard(self) -> None:
        if self._discard is not None:
            self._discard()


def resume_point(candidate: Candidate, *, patch: bytes | None = None,
                 worktree: Path | None = None, base: str | None = None,
                 anchor_commit: str | None = None) -> ResumePoint:
    ck = candidate.checkpoint
    hypothesis = loop.Hypothesis(**dict(candidate.hypothesis))
    provenance = tuple(dict(ck[key]) for key in ("critic_hypothesis", "critic_patch")
                       if isinstance(ck.get(key), Mapping)
                       and (key == "critic_hypothesis" or candidate.stage == "build"))
    materializer = discarder = None
    if candidate.stage == "build":
        def materializer():
            # Re-read and re-verify at the moment of use: the bytes the scan
            # checked are not assumed to be the bytes on disk now.
            fresh = verify_retained_patch(ck.get("retained_patch"),
                                          anchor_commit=anchor_commit or base,
                                          mechanism_id=hypothesis.mechanism_id)
            if patch is not None and fresh != patch:
                raise ResumeRejected("patch_sha256", "retained patch changed after the scan")
            return materialize(worktree, fresh, expected_base=anchor_commit or base)

        def discarder():
            discard(worktree, patch if patch is not None else verify_retained_patch(
                ck.get("retained_patch"), anchor_commit=anchor_commit or base,
                mechanism_id=hypothesis.mechanism_id))
    return ResumePoint(
        checkpoint_id=candidate.checkpoint_id, stage=candidate.stage, hypothesis=hypothesis,
        checkpoint=dict(ck), provenance=provenance,
        prior_patch_rejections=tuple(str(item) for item in
                                     (ck.get("prior_patch_rejections") or ())),
        patch_rounds=(1 if candidate.stage == "build"
                      else max(1, int(ck.get("patch_rounds_remaining") or 1))),
        _materialize=materializer, _discard=discarder)


class ResumeQueue:
    """Validated candidates, most advanced first; `take` claims before it hands out."""

    def __init__(self, *, store_root: Path, entries: Sequence[tuple[Candidate, bytes | None]],
                 siblings: Mapping[str, Sequence[str]], anchor_commit: str,
                 epoch: str) -> None:
        self.store_root = Path(store_root)
        self.entries = list(entries)
        self.siblings = {key: tuple(value) for key, value in siblings.items()}
        self.anchor_commit = anchor_commit
        self.epoch = epoch
        self.handed_out: list[str] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.entries)

    def take(self, worker, base: str | None) -> ResumePoint | None:
        with self._lock:
            if not self.entries:
                return None
            with ClaimLedger(self.store_root) as ledger:
                while self.entries:
                    candidate, patch = self.entries.pop(0)
                    if base != self.anchor_commit:
                        # The champion advanced under the run: consumed as rejected,
                        # and handed out stale so the lane records WHY.
                        reason = (f"anchor changed during the run: lane base "
                                  f"{str(base)[:12]}, formed on {self.anchor_commit[:12]}")
                        if not ledger.claim(candidate.checkpoint_id, self.anchor_commit,
                                            state="rejected", stage=candidate.stage,
                                            epoch=self.epoch, mechanism_id=candidate.group[0],
                                            source_attempt_id=candidate.attempt_id,
                                            detail=f"anchor: {reason}"):
                            continue
                        point = resume_point(candidate, patch=patch)
                        point.stale = ("anchor", reason)
                        self.handed_out.append(candidate.checkpoint_id)
                        return point
                    if not ledger.claim(candidate.checkpoint_id, self.anchor_commit,
                                        state="resumed", stage=candidate.stage,
                                        epoch=self.epoch, mechanism_id=candidate.group[0],
                                        source_attempt_id=candidate.attempt_id,
                                        detail=f"lane {getattr(worker, 'name', worker)}"):
                        continue    # another process took it: at most once
                    for sibling in self.siblings.get(candidate.checkpoint_id, ()):
                        ledger.claim(sibling, self.anchor_commit, state="superseded",
                                     stage=None, epoch=self.epoch,
                                     mechanism_id=candidate.group[0],
                                     detail=f"superseded by {candidate.checkpoint_id}")
                    self.handed_out.append(candidate.checkpoint_id)
                    return resume_point(candidate, patch=patch,
                                        worktree=getattr(worker, "worktree", None),
                                        base=base, anchor_commit=self.anchor_commit)
            return None


def prepare(store_root: Path, *, epoch: str, anchor_commit: str, target: Mapping[str, Any],
            repo: Path | None, rules_fingerprint: str | None = None,
            on_rejected: Callable[[dict], None] | None = None,
            scratch: Path | None = None, dry_run: bool = False) -> tuple[ResumeQueue, dict]:
    """Scan, choose and re-validate. `dry_run` writes nothing (not even a claim)."""
    rules_fingerprint = rules_fingerprint or loop.gate_rules_fingerprint()
    report: dict[str, Any] = {"epoch": epoch, "anchor_commit": anchor_commit,
                              "target": dict(target), "queued": [], "rejected": [],
                              "ineligible": [], "already_claimed": 0, "scanned": 0}
    try:
        candidates = scan(store_root, epoch=epoch, immutable=dry_run)
    except FileNotFoundError:
        candidates = []
    report["scanned"] = len(candidates)
    report["other_epoch_rows"] = other_epoch_checkpoints(store_root, epoch=epoch,
                                                         immutable=dry_run)
    ledger = ClaimLedger(store_root, read_only=dry_run)
    try:
        claimed = ledger.claimed(anchor_commit)
        groups: dict[tuple, list[Candidate]] = {}
        for candidate in candidates:
            if candidate.checkpoint_id in claimed:
                report["already_claimed"] += 1
                continue
            groups.setdefault(candidate.group, []).append(candidate)
        entries: list[tuple[Candidate, bytes | None]] = []
        siblings: dict[str, list[str]] = {}
        for members in groups.values():
            chosen = None
            for candidate in sorted(members, key=lambda item: item.rank, reverse=True):
                why = ineligible_reason(candidate, rules_fingerprint=rules_fingerprint)
                if why is not None:
                    report["ineligible"].append({**candidate.summary(), "reason": why})
                    continue
                try:
                    patch = prevalidate(candidate, epoch=epoch, anchor_commit=anchor_commit,
                                        target=target, repo=repo, scratch=scratch)
                except ResumeRejected as exc:
                    report["rejected"].append({**candidate.summary(), "check": exc.check,
                                               "reason": str(exc)})
                    if not dry_run and ledger.claim(
                            candidate.checkpoint_id, anchor_commit, state="rejected",
                            stage=candidate.stage, epoch=epoch,
                            mechanism_id=candidate.group[0],
                            source_attempt_id=candidate.attempt_id,
                            detail=f"{exc.check}: {exc}"[:2000]) and on_rejected is not None:
                        on_rejected(rejection_attempt(candidate, exc.check, str(exc),
                                                      anchor_commit=anchor_commit))
                    continue
                chosen = candidate
                entries.append((candidate, patch))
                report["queued"].append(candidate.summary())
                break
            if chosen is not None:
                siblings[chosen.checkpoint_id] = [item.checkpoint_id for item in members
                                                  if item is not chosen]
        entries.sort(key=lambda entry: entry[0].rank, reverse=True)
        report["queued"].sort(key=lambda row: (STAGE_RANK.get(row["stage"], 0),
                                               row["recorded_at"]), reverse=True)
    finally:
        ledger.close()
    return (ResumeQueue(store_root=store_root, entries=entries, siblings=siblings,
                        anchor_commit=anchor_commit, epoch=epoch),
            report)


# ------------------------------------------------------------------ backfill


def _resolve_row(connection: sqlite3.Connection, row_id: str) -> sqlite3.Row:
    rows = connection.execute("SELECT * FROM experiments WHERE attempt_id LIKE ?",
                              (row_id + "%",)).fetchall()
    if len(rows) != 1:
        raise ValueError(f"row {row_id!r} matches {len(rows)} experiment rows; need exactly 1")
    return rows[0]


def _rounds(provenance: Sequence[Mapping[str, Any]]) -> list[tuple[dict, dict, int]]:
    """(critic:patch accepted row, the gate row that followed it, provenance index)."""
    rounds = []
    for index, row in enumerate(provenance):
        if row.get("decision") == "critic:patch" and row.get("accepted"):
            for later in range(index + 1, len(provenance)):
                decision = str(provenance[later].get("decision") or "")
                if decision.startswith("gate:"):
                    rounds.append((dict(row), dict(provenance[later]), later))
                    break
                if decision.startswith("critic:"):
                    break
    return rounds


def backfill_plan(store_root: Path, *, row_id: str, patch: Path | None = None,
                  repo: Path | None = None, scratch: Path | None = None) -> dict:
    """Read-only. Everything `--apply` would write, and every check it rests on."""
    store_root = Path(store_root)
    connection = _connect(store_root, immutable=True)
    try:
        row = _resolve_row(connection, row_id)
    finally:
        connection.close()
    payload = json.loads(row["payload"])
    source = row["attempt_id"]
    if payload.get("resume_checkpoints"):
        raise ValueError(f"row {source[:12]} already carries resume checkpoints")
    mechanism = payload.get("mechanism_id")
    anchor = payload.get("spawn_parent")
    if not mechanism or not anchor:
        raise ValueError("row names no mechanism or no lane base (spawn_parent)")
    hypothesis = {key: payload.get(key) for key in (
        "mechanism_id", "statement", "falsifier", "target_surface", "target_symbol")}
    loop.Hypothesis(**hypothesis)
    provenance = [dict(item) for item in payload.get("validator_provenance") or ()]
    critic_hypothesis = next((item for item in provenance if item.get("decision")
                              == "critic:hypothesis" and item.get("accepted")), None)
    if critic_hypothesis is None:
        raise ValueError("row carries no accepted critic:hypothesis verdict")
    rounds = _rounds(provenance)
    refused = [entry for entry in rounds if not entry[1].get("accepted")]
    if not refused:
        raise ValueError("row carries no critic-accepted patch that a gate refused")
    lane = str(payload.get("branch_id") or "").partition(":")[2] or None
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", mechanism)[:80]
    lane_label = re.sub(r"[^A-Za-z0-9_.-]", "_", lane or "")[:40]
    found = []
    for path in sorted((store_root / "patches").glob(f"{label}.{lane_label or '*'}.*.patch")):
        try:
            raw = verify_retained_patch({"patch_file": str(path.resolve()),
                                         "metadata_file": str(path.with_suffix(".json").resolve()),
                                         "patch_sha256": _sha256(path.read_bytes())},
                                        anchor_commit=anchor, mechanism_id=mechanism)
        except ResumeRejected:
            continue
        found.append((path.stat().st_mtime_ns, path, raw))
    found.sort()
    retained = [{"patch_file": str(path.resolve()),
                 "metadata_file": str(path.with_suffix(".json").resolve()),
                 "patch_sha256": _sha256(raw),
                 "mtime": datetime.fromtimestamp(mtime / 1e9, timezone.utc)
                 .isoformat().replace("+00:00", "Z")} for mtime, path, raw in found]
    if patch is not None:
        chosen_path = Path(patch).resolve()
        matches = [index for index, item in enumerate(retained)
                   if item["patch_file"] == str(chosen_path)]
        if not matches:
            raise ValueError(f"{patch} is not a verified retained patch of this row")
        position = matches[0]
    else:
        position = len(retained) - 1
    if not retained:
        raise ValueError("no verified retained patch for this mechanism, lane and anchor")
    if len(retained) != len(rounds):
        raise ValueError(f"{len(retained)} retained patches but {len(rounds)} accepted "
                         "authoring rounds: the pairing is ambiguous, pass --patch")
    critic_patch, gate_row, until = rounds[position]
    if gate_row.get("accepted"):
        raise ValueError("that round passed its gate; it is not a gate refusal to resume")
    chosen = retained[position]
    pointer = {key: chosen[key] for key in ("patch_file", "metadata_file", "patch_sha256")}
    gate = str(gate_row["decision"]).partition(":")[2]
    scope = payload.get("research_scope") or {}
    model = scope.get("model")
    target = target_identity(measurement_surface=scope.get("measurement_surface"),
                             model=model.get("path") if isinstance(model, Mapping) else model)
    patch_round = position + 1
    checkpoint = {
        "schema": loop.CHECKPOINT_SCHEMA, "stage": "build", "hypothesis": hypothesis,
        "critic_hypothesis": critic_hypothesis, "critic_patch": critic_patch,
        "refusal_gate": gate, "refusal_reason": gate_row.get("reason"),
        # Unknown: the refusal predates fingerprints. The resumed candidate re-runs
        # every current gate, so unknown can only cost a pre-build refusal.
        "gate_rules_fingerprint": None, "retained_patch": pointer,
        "hypothesis_round": 1, "patch_round": patch_round,
        "resumed_from": None, "resume_depth": 0,
        "anchor_commit": anchor, "epoch_sha256": row["epoch_sha256"], "target": target,
    }
    attempt = {
        **hypothesis, "status": "gate_refused", "turn_recorded_at": _now(),
        "reason": gate_row.get("reason") or f"{gate} refused",
        "refusal_gate": gate,
        "gates": [{"gate": gate, "passed": False, "reason": gate_row.get("reason"),
                   "detail": None}],
        "hypothesis_round": 1, "patch_round": patch_round,
        "prior_rejection_prompt": patch_round > 1,
        "validator_provenance": provenance[:until + 1],
        "retained_patch": pointer,
        "resume_checkpoints": [checkpoint],
        "backfilled_from": {
            "schema": BACKFILL_SCHEMA, "attempt_id": source,
            "recorded_at": row["recorded_at"], "status": row["status"],
            "chosen_round": patch_round,
            "rounds": [{"round": index + 1, "patch": item,
                        "gate": rounds[index][1].get("decision"),
                        "gate_accepted": bool(rounds[index][1].get("accepted"))}
                       for index, item in enumerate(retained)],
            "pairing": "retained patches ordered by mtime, paired in order with the "
                       "row's critic-accepted authoring rounds",
            "tool": "autokernel.loop.resume backfill"},
        # experiments._attempt_id prefers this key: re-running the backfill is a no-op.
        "proposal_sha256": _sha256(f"resume-backfill\n{source}\n{pointer['patch_sha256']}"
                                   .encode()),
    }
    for key in ("spawn_parent", "branch_id", "width", "depth", "research_scope", "cpu_screen"):
        if payload.get(key) is not None:
            attempt[key] = payload[key]
    checks: dict[str, Any] = {"patch_sha256_matches_sidecar": True,
                              "sidecar_original_head_is_anchor": True}
    if repo is not None:
        raw = verify_retained_patch(pointer, anchor_commit=anchor, mechanism_id=mechanism)
        try:
            patch_applies(repo, anchor, raw, scratch=scratch)
            checks["applies_cleanly_at_anchor"] = True
        except ResumeRejected as exc:
            checks["applies_cleanly_at_anchor"] = f"NO: {exc}"
        try:
            checks["current_op_scope_preview"] = preview_op_scope(
                repo, anchor, raw, hypothesis, scratch=scratch)
        except Exception as exc:      # noqa: BLE001 -- a preview only
            checks["current_op_scope_preview"] = f"unavailable: {type(exc).__name__}: {exc}"
    checks["gate_rules_fingerprint_now"] = loop.gate_rules_fingerprint()
    connection = _connect(store_root, immutable=True)
    try:
        attempt_id = experiments._attempt_id(attempt, campaign_id=row["campaign_id"])
        present = connection.execute("SELECT 1 FROM experiments WHERE attempt_id=?",
                                     (attempt_id,)).fetchone() is not None
    finally:
        connection.close()
    return {"source_attempt_id": source, "epoch_sha256": row["epoch_sha256"],
            "campaign_id": row["campaign_id"], "attempt_id": attempt_id,
            "already_present": present, "attempt": attempt, "checks": checks,
            "retained_patches": retained, "chosen_patch": chosen}


def backfill_apply(store_root: Path, plan: Mapping[str, Any]) -> bool:
    """Append the planned row (idempotent on its attempt id). The only write."""
    with experiments.ExperimentStore(store_root) as store:
        added = store.record(plan["attempt"], epoch=plan["epoch_sha256"],
                             recorded_at=_now(), campaign_id=plan["campaign_id"])
        store.write_markdown(epoch=plan["epoch_sha256"])
    return added


# ------------------------------------------------------------------ CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    back = commands.add_parser("backfill", help="turn a pre-checkpoint row into a "
                               "resumable record (dry-run unless --apply)")
    back.add_argument("--store", type=Path, required=True)
    back.add_argument("--row", required=True, help="attempt id or unique prefix")
    back.add_argument("--patch", type=Path, help="retained patch to resume "
                      "(default: the latest critic-accepted round)")
    back.add_argument("--repo", type=Path, help="git repo holding the anchor; enables the "
                      "read-only clean-apply check and the op_scope preview")
    back.add_argument("--scratch", type=Path, help="scratch directory for the apply check")
    back.add_argument("--apply", action="store_true", help="append the row (the only write)")
    look = commands.add_parser("scan", help="read-only: what a launch would resume")
    look.add_argument("--store", type=Path, required=True)
    look.add_argument("--epoch", required=True)
    look.add_argument("--anchor", required=True)
    look.add_argument("--surface")
    look.add_argument("--model")
    look.add_argument("--repo", type=Path)
    look.add_argument("--scratch", type=Path)
    args = parser.parse_args(argv)
    if args.command == "backfill":
        try:
            plan = backfill_plan(args.store, row_id=args.row, patch=args.patch,
                                 repo=args.repo, scratch=args.scratch)
        except (ValueError, FileNotFoundError, ResumeRejected) as exc:
            print(f"backfill refused: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({key: plan[key] for key in (
            "source_attempt_id", "attempt_id", "already_present", "epoch_sha256",
            "campaign_id", "chosen_patch", "retained_patches", "checks")}, indent=2))
        print(json.dumps({"resume_checkpoints": plan["attempt"]["resume_checkpoints"],
                          "backfilled_from": plan["attempt"]["backfilled_from"]}, indent=2))
        if not args.apply:
            print("dry-run: nothing written (pass --apply to append this row)")
            return 0
        added = backfill_apply(args.store, plan)
        print(f"{'appended' if added else 'already present'} {plan['attempt_id']}")
        return 0
    _queue, report = prepare(args.store, epoch=args.epoch, anchor_commit=args.anchor,
                             target=target_identity(measurement_surface=args.surface,
                                                    model=args.model),
                             repo=args.repo, scratch=args.scratch, dry_run=True)
    print(json.dumps(report, indent=2))
    return 0


__all__ = ["BACKFILL_SCHEMA", "CLAIMS_FILE", "Candidate", "ClaimLedger", "MAX_RESUME_DEPTH",
           "RULE_GATES", "ResumePoint", "ResumeQueue", "ResumeRejected", "backfill_apply",
           "backfill_plan", "bind_checkpoints", "discard", "ineligible_reason", "materialize",
           "patch_applies", "patch_paths", "prepare", "prevalidate", "preview_op_scope",
           "rejection_attempt", "resume_point", "scan", "target_identity",
           "verify_retained_patch"]


if __name__ == "__main__":
    raise SystemExit(main())
