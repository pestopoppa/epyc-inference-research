#!/usr/bin/env python3
"""Resume abandoned candidates at the stage they reached, instead of re-deriving them.

    # from the repo root; census imports `autokernel.*`, so scripts/kernel_rnd too
    PYTHONPATH=scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.resume \\
        backfill --store <store> --row <attempt-id-prefix> [--repo <tree>] \\
        [--scratch <dir>] [--apply]
    PYTHONPATH=scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.resume \\
        backfill-critic2 --store <store> --row <attempt-id-prefix> --patch <file> \\
        --epoch <sha> [--epoch-reason "..."] [--base <sha>] [--lost-in-row <prefix>] \\
        [--repo <tree>] [--scratch <dir>] [--apply]
    PYTHONPATH=scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.resume \\
        scan --store <store> --epoch <sha> --anchor <sha> [--surface S] [--model M] \\
        [--repo <tree>]
    PYTHONPATH=scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.resume \\
        reopen --store <store> --checkpoint <attempt_id>#<i> --reason "..." \\
        [--anchor <sha>] [--apply]
    PYTHONPATH=scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.resume \\
        reinstate --store <store> --row <latest-checkpoint-row> --rejection <row> \\
        [--rejection <row> ...] --epoch <sha> [--attempts-used N] [--apply]

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
                            "critic2" -- an AUTHORED patch whose critic pass 2 never
                                        returned a verdict (a critic transient or auth
                                        failure, a stop, a lane_error, an author
                                        report-path failure over a real diff)
                            "author" -- a critic-accepted hypothesis whose authoring
                                        was interrupted by a stop or a transient,
                                        or whose patch rounds ran out on author
                                        failures (`patch_rounds_exhausted`) or on a
                                        scope rejection (`scope_blocked`)
    hypothesis              the exact Hypothesis.to_dict()
    critic_hypothesis       the accepted critic:hypothesis provenance row
    critic_patch            (build) the accepted critic:patch provenance row
    retained_patch          (build, critic2) {patch_file, metadata_file, patch_sha256}
    refusal_gate / refusal_reason / gate_rules_fingerprint   (build, gate refusals)
    prior_patch_rejections / patch_rounds_remaining          (author, critic2; the
                            remaining count INCLUDES the round in flight)
    hypothesis_round / patch_round
    resumed_from / resume_depth                              lineage
    anchor_commit / epoch_sha256 / target                    bound by the owner

THE ALGORITHM (`prepare`), on every launch, before any fresh hypothesis is drawn:

 1. scan the store for checkpoints in the CURRENT epoch; drop any already claimed
    at this anchor (`<store>/resume-claims.sqlite3`);
 2. eligibility: a build checkpoint needs a retained patch and a refusal by a RULE
    gate (`RULE_GATES`) whose rules have changed since (or were never recorded);
    a critic2 checkpoint needs a retained patch; an author checkpoint needs a patch
    round left. Ineligible ones stay unclaimed, so a later rule change can still
    reopen them;
 3. group by hypothesis; per group, try the most advanced first (build > critic2 >
    author, then newest): re-validate it -- anchor, epoch and target match, chain
    depth, the carried critic verdicts, and for a build or critic2 the patch bytes
    against the row's and the sidecar's sha256 and a clean apply at the anchor. A
    critic2 resume restores the bytes and runs critic pass 2 for real, then the gates
    and the measurement (no planner or author call); a rejection there continues the
    ordinary patch rounds with the author. A failure is recorded
    as `resume_rejected` (its own row, and a claim) and the next sibling is tried;
 4. the queue hands each survivor to a lane on its next iteration (`take`), which
    CLAIMS it first (at most once per anchor; siblings become `superseded`), so a
    crash mid-resume cannot duplicate it. The lane re-verifies and applies the
    bytes, and `loop.iterate` re-runs the host integrity check and every CURRENT
    gate -- the old verdict is never trusted -- before any build or measurement.

PENDING HYPOTHESES. An accepted hypothesis whose patch rounds all end in rejection
stays pending at the author (`loop.PATCH_ROUNDS_EXHAUSTED`): its row carries an
author checkpoint with every patch rejection as feedback and
`author_attempts_used/budget`, so the next draw re-authors it before the planner is
asked -- at launch through `prepare`, and in-run through `ResumeQueue` refreshing
when it runs dry. The budget spent retires it (`loop.HYPOTHESIS_RETIRED`, no
checkpoint). A `loop.SCOPE_BLOCKED` checkpoint records the route and rule that
blocked it (`scope_block`) and stays ineligible until `loop.scope_rules_fingerprint`
changes, which re-admits it with no operator action. `pending_hypotheses` is the
read-only view the planner prompt and `loop-status.json` show; `reinstate` re-pends
one a pre-policy iteration dropped.

SETTLING. A validity outcome of the resumed round (a current-gate, compile or
correctness refusal, a measured null or regression, a keep, a re-validation refusal)
consumes the claim; a ROUND disposition inside it (`ROUND_DISPOSITIONS`: one
patch rejected, one gate refusal) leaves it to the iteration's outcome. An
INFRASTRUCTURE fault (`INFRASTRUCTURE_STATUSES`: an exception
contained as `lane_error`) releases it back to resumable, at most `INFRA_RETRIES`
times per (checkpoint, anchor), counted in the ledger; `claim_events` logs every
release, exhaustion, reopen and re-claim. `reopen` is the operator's logged override
for a claim consumed anyway (DS41 run 9d: eab36f3e...#0, lost to a lane_error before
releases existed).

The backfill turns a row that predates checkpoints (run 9c's 22d950a4...) plus its
retained patches into one resumable `gate_refused` row. It is a tool the operator's
session runs; `--apply` is the only write, and it only appends.

`backfill-critic2` does the same for an author patch saved OUTSIDE the store before
checkpoints could carry one (DS41 runs 10d/10e): it takes an author-stage checkpoint
row (the hypothesis and its accepted critic:hypothesis verdict) and the saved patch,
and plans one `planner_transient` row carrying one critic2 checkpoint, bound to an
explicit `--epoch` (a rebind from the source row's epoch needs `--epoch-reason`,
recorded in the row). `--apply` retains the patch in `<store>/patches` (the same
immutable content-addressed pair `archive.retain_patch` writes) and appends that row.
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
STAGE_RANK = {"build": 3, "critic2": 2, "author": 1}
#: Stages that restore retained patch bytes (`loop.PATCH_STAGES`).
PATCH_STAGES = loop.PATCH_STAGES
CRITIC2_BACKFILL_SCHEMA = "epyc.autokernel.resume_backfill_critic2.v1"
#: Deterministic pre-build rule gates. A refusal by one of these is a verdict of the
#: RULE, not of the patch, so it is resumable once the rule changes. A compile or
#: correctness failure is the patch's own and is never resumed at build.
RULE_GATES = frozenset({"op_scope"})
#: A stop during a resumed candidate writes a new checkpoint; bound the chain.
MAX_RESUME_DEPTH = 3
#: `lane_error` rows carry a checkpoint only when an authored patch was waiting on
#: critic pass 2 when the lane faulted (`pipeline.run_pool`); the scan's
#: `resume_checkpoints` filter excludes every other lane_error row.
SCANNED_STATUSES = ("gate_refused", "stopped_mid_formation", "planner_transient",
                    "lane_error", loop.PATCH_ROUNDS_EXHAUSTED, loop.SCOPE_BLOCKED)
MAX_PATCH_BYTES = 4 * 1024 * 1024
CLAIM_STATES = ("resumed", "rejected", "superseded")
#: A claim handed back: by an infrastructure fault during the resumed round (bounded
#: by `INFRA_RETRIES`) or by an operator's logged `reopen`. A released claim is not
#: "claimed" -- the next launch re-validates and re-claims it like a fresh one.
RELEASED = "released"
#: Outcomes of a resumed round that are faults of the HARNESS, not verdicts on the
#: candidate: an exception contained as the lane's `lane_error` (DS41 run 9d: a
#: RatchetRefused re-retaining the restored patch). They release the claim instead
#: of consuming it. Every other outcome -- a gate or compile or correctness refusal,
#: a measured null, a regression, a keep, a re-validation refusal -- consumes it.
INFRASTRUCTURE_STATUSES = frozenset({"lane_error"})
#: Releases per (checkpoint, anchor) before an infrastructure fault consumes it too:
#: a fault that recurs every time is the setup, and the operator's `reopen` remains.
INFRA_RETRIES = 2
#: Dispositions of ONE ROUND of a resumed candidate, recorded while its iteration is
#: still running: a verdict on that patch, never on the resumed hypothesis. They do
#: not settle the claim; the iteration's own outcome does. DS41 run 10g: the first
#: `patch_rejected` of a resumed critic2 round consumed eed08b5f...#0 at once, so
#: nothing that iteration did afterwards (another round, a pending author checkpoint,
#: a lane_error release) could still speak for the claim.
ROUND_DISPOSITIONS = frozenset({"patch_rejected", "gate_refused"})


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


def retain_checkpoint_patches(checkpoints: list, retain: Callable[[str], Any]) -> list:
    """Owner-side, before binding: give each pending critic2 checkpoint its patch.

    `retain(mechanism_id)` retains the lane's diff (`archive.retain_patch`) and
    returns the patch path, or None when the lane holds no diff. A critic2
    checkpoint the loop could not point yet takes that pointer; one whose lane holds
    no diff, or whose retention failed, is DROPPED (an author checkpoint beside it
    still resumes the round). Returns the list, modified in place.
    """
    if not isinstance(checkpoints, list):
        return checkpoints
    kept: list = []
    for entry in checkpoints:
        if isinstance(entry, dict) and entry.get("stage") == "critic2" \
                and not entry.get("retained_patch"):
            mechanism = (entry.get("hypothesis") or {}).get("mechanism_id") or "unnamed"
            try:
                path = retain(mechanism)
            except Exception as exc:      # noqa: BLE001 -- the row still lands
                print(f"warning: critic2 patch retention failed: {type(exc).__name__}: "
                      f"{exc}", file=sys.stderr)
                path = None
            if path is None:
                continue
            path = Path(path)
            entry["retained_patch"] = {
                "patch_file": str(path.resolve()),
                "metadata_file": str(path.with_suffix(".json").resolve()),
                "patch_sha256": _sha256(path.read_bytes())}
        kept.append(entry)
    checkpoints[:] = kept
    return checkpoints


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
        # Additive migration: a ledger written before releases existed gains the
        # count in place. Older code keeps working (it names its INSERT columns, and
        # it counts a released row as claimed, which is the conservative reading).
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(claims)")}
        if "retries" not in columns:
            self.db.execute("ALTER TABLE claims ADD COLUMN retries INTEGER NOT NULL DEFAULT 0")
        # Append-only: every release, exhaustion, reopen and re-claim, with its reason.
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS claim_events ("
            "event_id INTEGER PRIMARY KEY AUTOINCREMENT, checkpoint_id TEXT NOT NULL, "
            "anchor_commit TEXT NOT NULL, event TEXT NOT NULL, at TEXT NOT NULL, "
            "prior_state TEXT, prior_result_status TEXT, retries INTEGER, "
            "actor TEXT, reason TEXT)")
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
            "SELECT checkpoint_id FROM claims WHERE anchor_commit=? AND state<>?",
            (anchor_commit, RELEASED))}

    def _event(self, checkpoint_id: str, anchor_commit: str, event: str, *,
               prior: Mapping[str, Any] | None, retries: int | None,
               actor: str, reason: str | None) -> None:
        self.db.execute(
            "INSERT INTO claim_events (checkpoint_id, anchor_commit, event, at, prior_state, "
            "prior_result_status, retries, actor, reason) VALUES (?,?,?,?,?,?,?,?,?)",
            (checkpoint_id, anchor_commit, event, _now(),
             None if prior is None else prior.get("state"),
             None if prior is None else prior.get("result_status"), retries, actor,
             None if reason is None else str(reason)[:2000]))

    def _row(self, checkpoint_id: str, anchor_commit: str) -> dict | None:
        cursor = self.db.execute("SELECT * FROM claims WHERE checkpoint_id=? AND anchor_commit=?",
                                 (checkpoint_id, anchor_commit))
        row = cursor.fetchone()
        if row is None:
            return None
        found = dict(zip([column[0] for column in cursor.description], row))
        found.setdefault("retries", 0)
        return found

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
        # A RELEASED claim is taken again by exactly one caller: the conditional
        # UPDATE is atomic, so two racing launches cannot both re-take it.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self._row(checkpoint_id, anchor_commit)
            taken = self.db.execute(
                "UPDATE claims SET state=?, stage=coalesce(?, stage), "
                "epoch_sha256=coalesce(?, epoch_sha256), mechanism_id=coalesce(?, mechanism_id), "
                "source_attempt_id=coalesce(?, source_attempt_id), claimed_at=?, detail=?, "
                "result_status=NULL, settled_at=NULL "
                "WHERE checkpoint_id=? AND anchor_commit=? AND state=?",
                (state, stage, epoch, mechanism_id, source_attempt_id, _now(), detail,
                 checkpoint_id, anchor_commit, RELEASED)).rowcount == 1
            if taken:
                self._event(checkpoint_id, anchor_commit, f"reclaimed:{state}", prior=prior,
                            retries=int(prior.get("retries") or 0), actor="loop",
                            reason=detail)
            self.db.commit()
            return taken
        except BaseException:
            self.db.rollback()
            raise

    def settle(self, checkpoint_id: str, anchor_commit: str, *, result_status: str,
               detail: str | None = None) -> None:
        if self.read_only:
            raise RuntimeError("claim ledger opened read-only")
        self.db.execute("UPDATE claims SET result_status=?, settled_at=?, "
                        "detail=coalesce(?, detail) WHERE checkpoint_id=? AND anchor_commit=?",
                        (result_status, _now(), detail, checkpoint_id, anchor_commit))
        self.db.commit()

    def settle_outcome(self, checkpoint_id: str, anchor_commit: str, *, result_status: str,
                       detail: str | None = None,
                       max_retries: int = INFRA_RETRIES) -> str:
        """Settle a resumed round's outcome; returns what happened to the claim.

        "settled"   -- a validity outcome: the claim is consumed (the result recorded).
                       A pending-hypothesis outcome (`patch_rounds_exhausted`,
                       `scope_blocked`) consumes it too: its row carries the NEXT
                       checkpoint, so the hypothesis stays resumable under a new id.
        "pending"   -- a round disposition (`ROUND_DISPOSITIONS`) of a resumed
                       candidate whose iteration is still running: nothing changes;
                       the iteration's outcome settles it.
        "released"  -- an infrastructure fault (`INFRASTRUCTURE_STATUSES`) with retries
                       left: the claim goes back to resumable and the count increments.
        "exhausted" -- an infrastructure fault with no retry left: consumed.
        "kept"      -- the claim was not an open resumed one (already settled by a
                       validity disposal in this iteration, rejected, superseded or
                       released): left exactly as it is.
        "absent"    -- no such claim.
        """
        if self.read_only:
            raise RuntimeError("claim ledger opened read-only")
        if result_status in ROUND_DISPOSITIONS:
            return "pending"
        if result_status not in INFRASTRUCTURE_STATUSES:
            self.settle(checkpoint_id, anchor_commit, result_status=result_status)
            return "settled"
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self._row(checkpoint_id, anchor_commit)
            if prior is None:
                self.db.rollback()
                return "absent"
            if prior["state"] != "resumed" or prior.get("result_status") is not None:
                self.db.rollback()
                return "kept"
            retries = int(prior.get("retries") or 0)
            if retries >= max_retries:
                self.db.execute(
                    "UPDATE claims SET result_status=?, settled_at=?, detail=? "
                    "WHERE checkpoint_id=? AND anchor_commit=?",
                    (result_status, _now(),
                     f"infrastructure retry budget ({max_retries}) spent: "
                     f"{result_status}: {detail or ''}"[:2000], checkpoint_id, anchor_commit))
                self._event(checkpoint_id, anchor_commit, "exhausted", prior=prior,
                            retries=retries, actor="loop", reason=detail)
                self.db.commit()
                return "exhausted"
            self.db.execute(
                "UPDATE claims SET state=?, retries=?, result_status=?, settled_at=?, detail=? "
                "WHERE checkpoint_id=? AND anchor_commit=?",
                (RELEASED, retries + 1, result_status, _now(),
                 f"released after infrastructure fault {retries + 1}/{max_retries}: "
                 f"{result_status}: {detail or ''}"[:2000], checkpoint_id, anchor_commit))
            self._event(checkpoint_id, anchor_commit, "released", prior=prior,
                        retries=retries + 1, actor="loop", reason=detail)
            self.db.commit()
            return "released"
        except BaseException:
            self.db.rollback()
            raise

    def reopen(self, checkpoint_id: str, anchor_commit: str, *, reason: str,
               actor: str = "operator") -> dict:
        """Operator override: hand a consumed claim back, with a logged reason.

        The retry count is kept, so a reopened claim that then faults with its budget
        spent is consumed after this one attempt; the event log keeps every prior state.
        """
        if self.read_only:
            raise RuntimeError("claim ledger opened read-only")
        if not reason or not reason.strip():
            raise ValueError("a reopen needs a reason")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self._row(checkpoint_id, anchor_commit)
            if prior is None:
                raise ValueError(f"no claim for {checkpoint_id} at {anchor_commit}")
            if prior["state"] == RELEASED:
                raise ValueError(f"{checkpoint_id} is already released (resumable)")
            self.db.execute(
                "UPDATE claims SET state=?, detail=? WHERE checkpoint_id=? AND anchor_commit=?",
                (RELEASED, f"reopened by {actor}: {reason}"[:2000], checkpoint_id,
                 anchor_commit))
            self._event(checkpoint_id, anchor_commit, "reopened", prior=prior,
                        retries=int(prior.get("retries") or 0), actor=actor, reason=reason)
            self.db.commit()
            return prior
        except BaseException:
            self.db.rollback()
            raise

    def rows(self) -> list[dict]:
        if self.db is None:
            return []
        cursor = self.db.execute("SELECT * FROM claims ORDER BY claimed_at")
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def events(self, checkpoint_id: str | None = None) -> list[dict]:
        if self.db is None:
            return []
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                               "AND name='claim_events'").fetchone():
            return []
        cursor = self.db.execute(
            "SELECT * FROM claim_events" + (" WHERE checkpoint_id=?" if checkpoint_id else "")
            + " ORDER BY event_id", (checkpoint_id,) if checkpoint_id else ())
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


def ineligible_reason(candidate: Candidate, *, rules_fingerprint: str,
                      scope_fingerprint: str | None = None) -> str | None:
    """Why this checkpoint is not resumable NOW (it stays unclaimed), or None.

    A pending accepted hypothesis (an author checkpoint from `patch_rounds_exhausted`
    or `scope_blocked`) is refused only when its authoring budget is spent, or while
    the scope rules that blocked it are unchanged (`loop.scope_rules_fingerprint`).
    """
    ck = candidate.checkpoint
    if candidate.stage not in STAGE_RANK:
        return f"unknown stage {candidate.stage!r}"
    if not candidate.hypothesis.get("mechanism_id"):
        return "checkpoint carries no hypothesis"
    if candidate.stage in PATCH_STAGES:
        pointer = ck.get("retained_patch")
        if not isinstance(pointer, Mapping) or not pointer.get("patch_sha256"):
            return "no retained patch"
    if candidate.stage == "critic2":
        # Absent means the round in flight only (it counts itself).
        remaining = ck.get("patch_rounds_remaining")
        if remaining is not None and int(remaining) < 1:
            return "no patch round left"
    elif candidate.stage == "build":
        gate = ck.get("refusal_gate")
        if gate is not None and gate not in RULE_GATES:
            return f"refused by {gate}, a verdict on the patch rather than a rule"
        recorded = ck.get("gate_rules_fingerprint")
        if gate is not None and recorded is not None and recorded == rules_fingerprint:
            return f"{gate} rules unchanged since the refusal"
    elif int(ck.get("patch_rounds_remaining") or 0) < 1:
        return "no patch round left"
    budget = ck.get("author_attempts_budget")
    if budget is not None and int(ck.get("author_attempts_used") or 0) >= int(budget):
        return (f"authoring budget spent ({ck.get('author_attempts_used')}/{budget} "
                "attempts)")
    block = ck.get("scope_block")
    if isinstance(block, Mapping):
        recorded = block.get("scope_rules_fingerprint")
        current = scope_fingerprint or loop.scope_rules_fingerprint()
        if recorded is not None and recorded == current:
            return (f"scope-blocked on {block.get('route')} by {block.get('source')}: "
                    f"{str(block.get('rule') or '')[:200]}; the scope rules are unchanged")
    return None


def depth_limit(checkpoint: Mapping[str, Any]) -> int:
    """`MAX_RESUME_DEPTH` interruption hops, plus one hop per authoring attempt a
    pending hypothesis is budgeted (each re-authoring is a resume by design and is
    bounded by that budget), plus one for a scope-blocked re-admission."""
    extra = int(checkpoint.get("author_attempts_budget") or 0)
    if isinstance(checkpoint.get("scope_block"), Mapping):
        extra += 1
    return MAX_RESUME_DEPTH + extra


def prevalidate(candidate: Candidate, *, epoch: str, anchor_commit: str,
                target: Mapping[str, Any], repo: Path | None,
                scratch: Path | None = None) -> bytes | None:
    """Everything checkable before a lane is touched. Returns the patch for a build."""
    ck = candidate.checkpoint
    if int(ck.get("resume_depth") or 0) >= depth_limit(ck):
        raise ResumeRejected("depth", f"resume chain depth {ck.get('resume_depth')} reached "
                             f"the limit {depth_limit(ck)}")
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
    if candidate.stage not in PATCH_STAGES:
        return None
    if candidate.stage == "build":
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
    if candidate.stage in PATCH_STAGES:
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
        # critic2: its own round (critic pass 2 on the restored bytes) plus the
        # author rounds that were left, exactly as the interrupted round had them.
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

    #: IN-RUN pending hypotheses (`enable_pending_refresh`): an empty queue re-scans
    #: the store for accepted hypotheses a lane left pending THIS run
    #: (`loop.PENDING_HYPOTHESIS_STATUSES`), so the next draw re-authors them before
    #: the planner is asked. Launch-time `prepare` covers earlier runs.
    refresh_target: Mapping[str, Any] | None = None
    #: Extra binding keywords for `scan` / `prevalidate` (the launch's own binding).
    refresh_bind: Mapping[str, Any] = {}
    #: checkpoint_id -> (check, reason) for a refreshed entry that failed
    #: re-validation: handed out stale, so the lane records WHY.
    stale: Mapping[str, tuple[str, str]] = {}

    def enable_pending_refresh(self, target: Mapping[str, Any], **bind: Any) -> "ResumeQueue":
        """Turn on the in-run refresh. `target` is the launch's `target_identity`;
        `bind` is forwarded to `scan` and `prevalidate` (whatever binding keywords the
        launch's own `prepare` used), so a refreshed checkpoint binds exactly like a
        launch-time one."""
        self.refresh_target = dict(target)
        self.refresh_bind = dict(bind)
        self.stale = {}
        return self

    def _refresh(self) -> None:
        """Queue accepted hypotheses left pending since the last scan (author stage)."""
        try:
            candidates = [candidate for candidate in
                          scan(self.store_root, epoch=self.epoch, **self.refresh_bind)
                          if candidate.row_status in loop.PENDING_HYPOTHESIS_STATUSES]
        except FileNotFoundError:
            return
        # The writable ledger (as `take` uses): a read-only one is opened immutable and
        # would not see claims still in this process's WAL.
        with ClaimLedger(self.store_root) as ledger:
            claimed = ledger.claimed(self.anchor_commit)
        known = set(self.handed_out) | {entry[0].checkpoint_id for entry in self.entries}
        groups: dict[tuple, list[Candidate]] = {}
        for candidate in candidates:
            if candidate.checkpoint_id in claimed or candidate.checkpoint_id in known:
                continue
            groups.setdefault(candidate.group, []).append(candidate)
        rules = loop.gate_rules_fingerprint()
        scope = loop.scope_rules_fingerprint()
        for members in groups.values():
            for candidate in sorted(members, key=lambda item: item.rank, reverse=True):
                if ineligible_reason(candidate, rules_fingerprint=rules,
                                     scope_fingerprint=scope) is not None:
                    continue
                try:
                    prevalidate(candidate, epoch=self.epoch, anchor_commit=self.anchor_commit,
                                target=self.refresh_target or {}, repo=None,
                                **self.refresh_bind)
                except ResumeRejected as exc:
                    self.stale[candidate.checkpoint_id] = (exc.check, str(exc))
                self.entries.append((candidate, None))
                self.siblings[candidate.checkpoint_id] = tuple(
                    item.checkpoint_id for item in members if item is not candidate)
                break

    def __len__(self) -> int:
        return len(self.entries)

    def take(self, worker, base: str | None) -> ResumePoint | None:
        with self._lock:
            if not self.entries and self.refresh_target is not None:
                try:
                    self._refresh()
                except Exception as exc:      # noqa: BLE001 -- fresh research still runs
                    print(f"resume    pending-hypothesis refresh failed: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
            if not self.entries:
                return None
            with ClaimLedger(self.store_root) as ledger:
                while self.entries:
                    candidate, patch = self.entries.pop(0)
                    stale = self.stale.pop(candidate.checkpoint_id, None) \
                        if self.stale else None
                    if stale is not None:
                        if not ledger.claim(candidate.checkpoint_id, self.anchor_commit,
                                            state="rejected", stage=candidate.stage,
                                            epoch=self.epoch, mechanism_id=candidate.group[0],
                                            source_attempt_id=candidate.attempt_id,
                                            detail=f"{stale[0]}: {stale[1]}"[:2000]):
                            continue
                        point = resume_point(candidate, patch=patch)
                        point.stale = stale
                        self.handed_out.append(candidate.checkpoint_id)
                        return point
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


# ------------------------------------------------------------------ pending hypotheses


PENDING_SUMMARY_LIMIT = 12


def _claim_rows_live(store_root: Path) -> list[dict]:
    """Every claim row, read `mode=ro` (never `immutable`: a live run's claims may still
    sit in the WAL) and without creating the ledger when it does not exist yet."""
    path = Path(store_root) / CLAIMS_FILE
    if not path.is_file():
        return []
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        cursor = connection.execute("SELECT * FROM claims")
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def pending_hypotheses(store_root: Path, *, epoch: str, anchor_commit: str | None = None,
                       limit: int = PENDING_SUMMARY_LIMIT, **bind: Any) -> list[dict]:
    """Accepted hypotheses pending authoring in this epoch, newest first. Read-only.

    One row per hypothesis: its NEWEST `patch_rounds_exhausted` / `scope_blocked`
    checkpoint, unless that checkpoint's claim at the anchor is already settled (the
    re-authoring resolved it: kept, measured, retired or re-pended under a newer row)
    or it was formed on another anchor. `state` is "pending" (the next draw resumes
    it), "in_flight" (a lane holds its claim), "scope_blocked" or "budget_spent"
    (not resumable now; `blocked_reason` says why). Feeds the planner prompt (so it
    does not re-propose them) and `loop-status.json`. `bind` is forwarded to `scan`
    (the launch's own binding keywords).
    """
    try:
        candidates = [candidate for candidate in scan(store_root, epoch=epoch, **bind)
                      if candidate.row_status in loop.PENDING_HYPOTHESIS_STATUSES]
    except (FileNotFoundError, sqlite3.DatabaseError):
        return []
    claims = {(row["checkpoint_id"], row["anchor_commit"]): row
              for row in _claim_rows_live(store_root)}
    newest: dict[tuple, Candidate] = {}
    for candidate in candidates:      # oldest first: the last one per group wins
        newest[candidate.group] = candidate
    rules = loop.gate_rules_fingerprint()
    scope = loop.scope_rules_fingerprint()
    out: list[dict] = []
    for candidate in sorted(newest.values(), key=lambda item: item.recorded_at,
                            reverse=True):
        ck = candidate.checkpoint
        formed = ck.get("anchor_commit")
        if anchor_commit is not None and formed not in (None, anchor_commit):
            continue
        claim = claims.get((candidate.checkpoint_id, anchor_commit or formed))
        state = "pending"
        if claim is not None and claim.get("state") != RELEASED:
            if claim.get("state") != "resumed" or claim.get("result_status") is not None:
                continue
            state = "in_flight"
        blocked = ineligible_reason(candidate, rules_fingerprint=rules,
                                    scope_fingerprint=scope)
        if state == "pending" and blocked is not None:
            state = "scope_blocked" if isinstance(ck.get("scope_block"), Mapping) \
                else "budget_spent" if "budget" in blocked else "blocked"
        used = int(ck.get("author_attempts_used") or 0)
        budget = ck.get("author_attempts_budget")
        prior = [str(item) for item in ck.get("prior_patch_rejections") or ()]
        hypothesis = candidate.hypothesis
        row = {"mechanism_id": candidate.group[0],
               "statement": str(hypothesis.get("statement") or "")[:300],
               "target": f"{hypothesis.get('target_surface')}::{hypothesis.get('target_symbol')}",
               "status": candidate.row_status, "state": state,
               "checkpoint_id": candidate.checkpoint_id,
               "recorded_at": candidate.recorded_at,
               "author_attempts_used": used,
               "author_attempts_budget": None if budget is None else int(budget),
               "author_attempts_remaining": (None if budget is None
                                             else max(0, int(budget) - used)),
               "patch_rejections": len(prior),
               "last_patch_rejection": prior[-1][:300] if prior else None}
        if blocked is not None and state != "in_flight":
            row["blocked_reason"] = blocked
        if isinstance(ck.get("scope_block"), Mapping):
            row["scope_block"] = {key: ck["scope_block"].get(key)
                                  for key in ("route", "rule", "source")}
        out.append(row)
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ reinstate


REINSTATE_SCHEMA = "epyc.autokernel.resume_reinstate_hypothesis.v1"


def reinstate_plan(store_root: Path, *, row_id: str, rejection_rows: Sequence[str],
                   epoch: str, epoch_reason: str | None = None,
                   checkpoint_index: int | None = None,
                   status: str = loop.PATCH_ROUNDS_EXHAUSTED,
                   scope_rule: str | None = None, attempts_used: int | None = None,
                   budget: int = loop.HYPOTHESIS_AUTHOR_ATTEMPTS,
                   patch_rounds: int = loop.PATCH_ROUNDS,
                   reason: str | None = None,
                   measurement_epoch: str | None = None) -> dict:
    """Read-only. One pending-hypothesis row for an ACCEPTED hypothesis a pre-policy
    iteration dropped after its patch rounds ran out (DS41 run 10g).

    `row_id` names the row carrying the hypothesis's LATEST checkpoint (any stage:
    its hypothesis, accepted critic:hypothesis verdict, anchor, target and lineage are
    carried). `rejection_rows` are the `patch_rejected` / `gate_refused` rows whose
    verbatim reasons become the author's feedback (ordered by recorded_at, after any
    the checkpoint already carried). `attempts_used` defaults to the number of
    distinct authoring attempts those rows came from (their `resumed_from`, else their
    own row lineage). The planned row carries one AUTHOR checkpoint bound to `epoch`
    (a rebind from the source row's epoch needs `epoch_reason`); `measurement_epoch`,
    when given, is stamped as the checkpoint's `measurement_epoch_sha256` -- the
    identity a launch that binds resume on the measurement epoch matches, whatever
    actor configuration moved the full epoch. Nothing is written.
    """
    store_root = Path(store_root)
    if status not in loop.PENDING_HYPOTHESIS_STATUSES:
        raise ValueError(f"--status must be one of {sorted(loop.PENDING_HYPOTHESIS_STATUSES)}")
    if status == loop.SCOPE_BLOCKED and not (scope_rule and scope_rule.strip()):
        raise ValueError("--status scope_blocked needs --scope-rule naming the route/rule")
    if not re.fullmatch(r"[0-9a-f]{64}", str(epoch or "")):
        raise ValueError("--epoch must be a 64-hex epoch sha256")
    if measurement_epoch is not None and not re.fullmatch(r"[0-9a-f]{64}", measurement_epoch):
        raise ValueError("--measurement-epoch must be a 64-hex epoch sha256")
    if not rejection_rows:
        raise ValueError("name at least one --rejection row (the patch rejections)")
    connection = _connect(store_root, immutable=True)
    try:
        row = _resolve_row(connection, row_id)
        rejected = [_resolve_row(connection, item) for item in rejection_rows]
    finally:
        connection.close()
    payload = json.loads(row["payload"])
    source = row["attempt_id"]
    entries = [(position, entry) for position, entry in
               enumerate(payload.get("resume_checkpoints") or ())
               if isinstance(entry, dict) and entry.get("schema") == loop.CHECKPOINT_SCHEMA]
    if checkpoint_index is not None:
        entries = [item for item in entries if item[0] == checkpoint_index]
    if len(entries) != 1:
        raise ValueError(f"row {source[:12]} carries {len(entries)} checkpoint(s)"
                         + ("" if checkpoint_index is None else f" at index {checkpoint_index}")
                         + "; need exactly 1 (pass --checkpoint-index)")
    index, ck = entries[0][0], dict(entries[0][1])
    source_checkpoint = f"{source}#{index}"
    hypothesis = ck.get("hypothesis")
    if not isinstance(hypothesis, Mapping) or not hypothesis.get("mechanism_id"):
        raise ValueError("the source checkpoint carries no hypothesis")
    parsed = loop.Hypothesis(**dict(hypothesis))
    if parsed.runtime_pair is not None:
        raise ValueError("runtime treatments are not authored")
    critic_hypothesis = ck.get("critic_hypothesis")
    if not isinstance(critic_hypothesis, Mapping) or not critic_hypothesis.get("accepted"):
        raise ValueError("the source checkpoint carries no accepted critic:hypothesis verdict")
    anchor = ck.get("anchor_commit") or payload.get("spawn_parent")
    if not anchor:
        raise ValueError("the source checkpoint names no anchor")
    source_epoch = ck.get("epoch_sha256") or row["epoch_sha256"]
    rebind = epoch != source_epoch
    if rebind and not (epoch_reason and epoch_reason.strip()):
        raise ValueError(f"--epoch {epoch[:12]} differs from the source checkpoint's epoch "
                         f"{str(source_epoch)[:12]}: pass --epoch-reason")
    feedback = [str(item) for item in ck.get("prior_patch_rejections") or ()]
    attempts: list[str] = []
    rejection_record = []
    for item in sorted(rejected, key=lambda found: (found["recorded_at"], found["attempt_id"])):
        body = json.loads(item["payload"])
        if item["status"] not in ROUND_DISPOSITIONS:
            raise ValueError(f"rejection row {item['attempt_id'][:12]} is {item['status']}, "
                             f"not one of {sorted(ROUND_DISPOSITIONS)}")
        same = {key: body.get(key) for key in ("mechanism_id", "statement", "falsifier",
                                               "target_surface", "target_symbol")}
        if same != {key: hypothesis.get(key) for key in same}:
            raise ValueError(f"rejection row {item['attempt_id'][:12]} carries a different "
                             "hypothesis than the source checkpoint")
        text = str(body.get("reason") or "").strip()
        if text and text not in feedback:
            feedback.append(text)
        lineage = str(body.get("resumed_from") or item["attempt_id"])
        if lineage not in attempts:
            attempts.append(lineage)
        rejection_record.append({"attempt_id": item["attempt_id"], "status": item["status"],
                                 "recorded_at": item["recorded_at"],
                                 "epoch_sha256": item["epoch_sha256"],
                                 "refusal_gate": body.get("refusal_gate"),
                                 "resumed_from": body.get("resumed_from"),
                                 "patch_round": body.get("patch_round")})
    used = len(attempts) if attempts_used is None else int(attempts_used)
    budget = max(1, int(budget))
    if status == loop.PATCH_ROUNDS_EXHAUSTED and used >= budget:
        raise ValueError(f"{used} authoring attempt(s) already spent of a budget of {budget}: "
                         "this hypothesis would be retired, not pending (raise --budget or "
                         "pass --attempts-used)")
    carried = feedback[-loop.MAX_CARRIED_PATCH_REJECTIONS:]
    checkpoint = {
        "schema": loop.CHECKPOINT_SCHEMA, "stage": "author",
        "hypothesis": dict(hypothesis), "critic_hypothesis": dict(critic_hypothesis),
        "hypothesis_round": int(ck.get("hypothesis_round") or 1), "patch_round": 0,
        "prior_patch_rejections": carried,
        "patch_rounds_remaining": max(1, int(patch_rounds)),
        "author_attempts_used": used, "author_attempts_budget": budget,
        "resumed_from": source_checkpoint,
        "resume_depth": int(ck.get("resume_depth") or 0) + 1,
        "anchor_commit": anchor, "epoch_sha256": epoch,
        "target": dict(ck.get("target") or {}),
    }
    if measurement_epoch is not None:
        checkpoint["measurement_epoch_sha256"] = measurement_epoch
    state = {"class": status, "author_attempts_used": used, "author_attempts_budget": budget,
             "author_attempts_remaining": max(0, budget - used),
             "patch_rounds_per_attempt": max(1, int(patch_rounds)),
             "last_rejection": {"class": "scope" if status == loop.SCOPE_BLOCKED
                                else "authoring", "source": "critic:patch",
                                "rule": scope_rule if status == loop.SCOPE_BLOCKED else None}}
    if status == loop.SCOPE_BLOCKED:
        checkpoint["scope_block"] = state["scope_block"] = {
            "route": f"{parsed.target_surface}::{parsed.target_symbol}",
            "rule": scope_rule.strip(), "source": "critic:patch",
            "scope_rules_fingerprint": loop.scope_rules_fingerprint()}
    summary = (reason.strip() if reason and reason.strip() else
               f"reinstated {status}: an ACCEPTED hypothesis dropped after its patch rounds "
               f"ran out (pre-policy); {len(rejection_record)} patch rejection(s) carried as "
               f"author feedback, {used}/{budget} authoring attempts spent")
    attempt = {
        **dict(hypothesis), "status": status, "turn_recorded_at": _now(),
        "reason": " | ".join([summary, *carried]), "refusal_gate": "critic:patch",
        "hypothesis_round": checkpoint["hypothesis_round"], "patch_round": 0,
        "prior_rejection_prompt": bool(carried),
        "validator_provenance": [{**dict(critic_hypothesis), "resumed_from": source_checkpoint,
                                  "changed_subsequent_search": False}],
        "hypothesis_pending": state,
        "resume_checkpoints": [checkpoint],
        "spawn_parent": anchor,
        "reinstated_from": {
            "schema": REINSTATE_SCHEMA, "attempt_id": source,
            "checkpoint_id": source_checkpoint, "checkpoint_stage": ck.get("stage"),
            "recorded_at": row["recorded_at"], "status": row["status"],
            "source_epoch_sha256": source_epoch, "epoch_sha256": epoch,
            "epoch_rebound": rebind,
            "epoch_rebind_reason": epoch_reason.strip() if rebind else None,
            "measurement_epoch_sha256": measurement_epoch,
            "rejections": rejection_record, "attempt_lineages": attempts,
            "tool": "autokernel.loop.resume reinstate"},
        # experiments._attempt_id prefers this key: re-running the reinstate is a no-op.
        "proposal_sha256": _sha256("resume-reinstate\n{}\n{}\n{}\n{}".format(
            source_checkpoint, epoch, status,
            ",".join(item["attempt_id"] for item in rejection_record)).encode()),
    }
    for key in ("branch_id", "width", "depth", "research_scope", "cpu_screen"):
        if payload.get(key) is not None:
            attempt[key] = payload[key]
    with ClaimLedger(store_root, read_only=True) as ledger:
        source_claims = [dict(item) for item in ledger.rows()
                         if item["checkpoint_id"] == source_checkpoint]
    candidate = Candidate("<planned>#0", "<planned>", _now(), status,
                          parsed.mechanism_id, checkpoint)
    checks: dict[str, Any] = {
        "source_claims": source_claims,
        # An open source claim would still resume on its own: reinstating it too
        # would queue the same hypothesis twice (the group dedupes, but say so).
        "source_checkpoint_consumed": any(
            item.get("state") != RELEASED and (item.get("result_status") is not None
                                              or item.get("state") in ("rejected", "superseded"))
            for item in source_claims),
        "epoch_rebound": rebind,
        "ineligible_now": ineligible_reason(candidate,
                                            rules_fingerprint=loop.gate_rules_fingerprint()),
        **_live_status(store_root),
    }
    checks["epoch_matches_live_loop_status"] = checks["live_loop_status_epoch"] == epoch
    checks["anchor_matches_live_loop_status"] = checks["live_loop_status_anchor"] == anchor
    checks["surface_matches_live_loop_status"] = (
        checks["live_loop_status_surface"] == checkpoint["target"].get("measurement_surface"))
    try:
        prevalidate(candidate, epoch=epoch, anchor_commit=anchor,
                    target=checkpoint["target"], repo=None)
        checks["prevalidates_at_anchor"] = True
    except ResumeRejected as exc:
        checks["prevalidates_at_anchor"] = f"NO ({exc.check}): {exc}"
    connection = _connect(store_root, immutable=True)
    try:
        attempt_id = experiments._attempt_id(attempt, campaign_id=row["campaign_id"])
        present = connection.execute("SELECT 1 FROM experiments WHERE attempt_id=?",
                                     (attempt_id,)).fetchone() is not None
    finally:
        connection.close()
    return {"source_attempt_id": source, "source_checkpoint_id": source_checkpoint,
            "epoch_sha256": epoch, "campaign_id": row["campaign_id"],
            "attempt_id": attempt_id, "checkpoint_id": f"{attempt_id}#0",
            "already_present": present, "attempt": attempt, "checks": checks}


def reinstate_apply(store_root: Path, plan: Mapping[str, Any]) -> bool:
    """The only write: append the planned row (idempotent on its attempt id)."""
    with experiments.ExperimentStore(store_root) as store:
        added = store.record(plan["attempt"], epoch=plan["epoch_sha256"],
                             recorded_at=_now(), campaign_id=plan["campaign_id"])
        store.write_markdown(epoch=plan["epoch_sha256"])
    return added


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


# ------------------------------------------------------------------ backfill: critic2


def _author_checkpoint(payload: Mapping[str, Any], source: str,
                       index: int | None) -> tuple[int, dict]:
    entries = [(position, entry) for position, entry in
               enumerate(payload.get("resume_checkpoints") or ())
               if isinstance(entry, dict) and entry.get("schema") == loop.CHECKPOINT_SCHEMA
               and entry.get("stage") == "author"]
    if index is not None:
        entries = [item for item in entries if item[0] == index]
    if len(entries) != 1:
        raise ValueError(f"row {source[:12]} carries {len(entries)} author checkpoint(s)"
                         + ("" if index is None else f" at index {index}")
                         + "; need exactly 1 (pass --checkpoint-index)")
    return entries[0][0], dict(entries[0][1])


def _live_status(store_root: Path) -> dict:
    """What the store's current run published (loop-status.json), for the report only:
    a mismatch is shown, never acted on (the next launch may differ from this one)."""
    try:
        status = json.loads(_read_bounded(Path(store_root) / "loop-status.json"))
    except (OSError, ResumeRejected, ValueError):
        status = {}
    status = status if isinstance(status, dict) else {}
    pick = lambda key: status.get(key) if isinstance(status.get(key), str) else None
    return {"live_loop_status_epoch": pick("epoch_sha256"),
            "live_loop_status_surface": pick("surface"),
            "live_loop_status_anchor": pick("anchor_commit")}


def backfill_critic2_plan(store_root: Path, *, row_id: str, patch: Path, epoch: str,
                          epoch_reason: str | None = None, base: str | None = None,
                          lost_in_row: str | None = None, lane: str = "lane0",
                          checkpoint_index: int | None = None, repo: Path | None = None,
                          scratch: Path | None = None, surface: str | None = None,
                          surface_reason: str | None = None) -> dict:
    """Read-only. A critic2 checkpoint for an author patch saved outside the store.

    `row_id` names the row whose AUTHOR checkpoint carries the hypothesis and its
    accepted critic:hypothesis verdict (DS41: 629ca6ab...). `patch` is the saved lane
    diff; its base is `base`, else `base.txt` beside it, and must be that checkpoint's
    anchor. `epoch` is explicit: the row is bound to the epoch the NEXT launch runs in,
    and a rebind away from the source row's epoch must say why (`epoch_reason`,
    recorded in the row). The target is the source checkpoint's; `surface` rebinds its
    measurement surface (needs `surface_reason`, recorded in the row) when the next
    launch measures on another one -- resume refuses a target mismatch otherwise.
    Nothing is written; `backfill_critic2_apply` writes the plan.
    """
    from . import archive
    store_root = Path(store_root)
    if not re.fullmatch(r"[0-9a-f]{64}", str(epoch or "")):
        raise ValueError("--epoch must be a 64-hex epoch sha256")
    connection = _connect(store_root, immutable=True)
    try:
        row = _resolve_row(connection, row_id)
        lost = _resolve_row(connection, lost_in_row) if lost_in_row else None
    finally:
        connection.close()
    payload = json.loads(row["payload"])
    source = row["attempt_id"]
    index, author = _author_checkpoint(payload, source, checkpoint_index)
    source_checkpoint = f"{source}#{index}"
    hypothesis = author.get("hypothesis")
    if not isinstance(hypothesis, Mapping) or not hypothesis.get("mechanism_id"):
        raise ValueError("the author checkpoint carries no hypothesis")
    parsed = loop.Hypothesis(**dict(hypothesis))
    if parsed.runtime_pair is not None:
        raise ValueError("runtime treatments carry no patch")
    critic_hypothesis = author.get("critic_hypothesis")
    if not isinstance(critic_hypothesis, Mapping) or not critic_hypothesis.get("accepted"):
        raise ValueError("the author checkpoint carries no accepted critic:hypothesis verdict")
    anchor = author.get("anchor_commit") or payload.get("spawn_parent")
    if not anchor:
        raise ValueError("the author checkpoint names no anchor")
    patch = Path(patch)
    raw = _read_bounded(patch)
    touched, _preexisting = patch_paths(raw)
    base_file = None
    if base is None:
        base_file = patch.parent / "base.txt"
        try:
            base = _read_bounded(base_file).decode("utf-8").strip()
        except FileNotFoundError:
            raise ValueError(f"no --base and no {base_file}") from None
    if base != anchor:
        raise ValueError(f"the patch was formed on {base}, not the checkpoint's anchor {anchor}")
    source_epoch = row["epoch_sha256"]
    rebind = epoch != source_epoch
    if rebind and not (epoch_reason and epoch_reason.strip()):
        raise ValueError(f"--epoch {epoch[:12]} differs from the source row's epoch "
                         f"{source_epoch[:12]}: pass --epoch-reason saying why the patch "
                         "is still valid there")
    target = dict(author.get("target") or {})
    source_surface = target.get("measurement_surface")
    surface_rebound = surface is not None and surface != source_surface
    if surface_rebound:
        if not (surface_reason and surface_reason.strip()):
            raise ValueError(f"--surface {surface!r} differs from the checkpoint's "
                             f"{source_surface!r}: pass --surface-reason")
        target["measurement_surface"] = surface
    lost_record = None
    if lost is not None:
        lost_payload = json.loads(lost["payload"])
        lost_hypotheses = [entry.get("hypothesis") for entry in
                           lost_payload.get("resume_checkpoints") or ()
                           if isinstance(entry, dict)]
        same = any(_canonical(item) == _canonical(dict(hypothesis))
                   for item in lost_hypotheses if isinstance(item, Mapping))
        if not same:
            raise ValueError(f"--lost-in-row {lost['attempt_id'][:12]} carries a different "
                             "hypothesis than the source checkpoint")
        lost_record = {"attempt_id": lost["attempt_id"], "status": lost["status"],
                       "recorded_at": lost["recorded_at"],
                       "epoch_sha256": lost["epoch_sha256"],
                       # The provider's stderr is left in the source row: it can
                       # quote credentials (run 10e's 401 quoted a masked key).
                       "reason": str(lost_payload.get("reason") or "")
                       .split(" stderr=", 1)[0][:500]}
    mechanism = parsed.mechanism_id
    target_path = archive.retained_patch_path(store_root, raw, head=anchor, lane=lane,
                                              mechanism_id=mechanism).resolve()
    sidecar = archive.patch_sidecar(target_path, raw, head=anchor, lane=lane,
                                    mechanism_id=mechanism,
                                    worktree=str(patch.resolve().parent))
    pointer = {"patch_file": str(target_path),
               "metadata_file": str(target_path.with_suffix(".json")),
               "patch_sha256": _sha256(raw)}
    already_retained = False
    if target_path.exists():
        if _read_bounded(target_path) != raw:
            raise ValueError(f"{target_path} exists with different bytes")
        already_retained = True
    insertions = sum(1 for line in raw.decode("utf-8", "replace").splitlines()
                     if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in raw.decode("utf-8", "replace").splitlines()
                    if line.startswith("-") and not line.startswith("---"))
    remaining = author.get("patch_rounds_remaining")
    checkpoint = {
        "schema": loop.CHECKPOINT_SCHEMA, "stage": "critic2",
        "hypothesis": dict(hypothesis), "critic_hypothesis": dict(critic_hypothesis),
        "hypothesis_round": int(author.get("hypothesis_round") or 1),
        # The author checkpoint is written before its round's number is taken, so
        # the round the saved patch was authored in is one past it.
        "patch_round": int(author.get("patch_round") or 0) + 1,
        "prior_patch_rejections": list(author.get("prior_patch_rejections") or ()),
        "patch_rounds_remaining": max(1, int(remaining if remaining is not None else 1)),
        "retained_patch": pointer,
        "resumed_from": source_checkpoint,
        "resume_depth": int(author.get("resume_depth") or 0) + 1,
        "anchor_commit": anchor, "epoch_sha256": epoch,
        "target": target,
    }
    reason = ("backfilled critic2 checkpoint: an authored patch lost before critic pass 2 "
              "returned a verdict; resume restores it and runs critic pass 2")
    if lost_record is not None:
        reason += f" (lost in {lost_record['attempt_id'][:12]}: {lost_record['reason'][:200]})"
    attempt = {
        **dict(hypothesis), "status": "planner_transient", "turn_recorded_at": _now(),
        "reason": reason,
        "hypothesis_round": checkpoint["hypothesis_round"],
        "patch_round": checkpoint["patch_round"],
        "prior_rejection_prompt": bool(checkpoint["prior_patch_rejections"]),
        "validator_provenance": [{**dict(critic_hypothesis),
                                  "resumed_from": source_checkpoint,
                                  "changed_subsequent_search": False}],
        "resume_checkpoints": [checkpoint],
        "backfilled_from": {
            "schema": CRITIC2_BACKFILL_SCHEMA, "attempt_id": source,
            "checkpoint_id": source_checkpoint, "recorded_at": row["recorded_at"],
            "status": row["status"], "source_epoch_sha256": source_epoch,
            "epoch_sha256": epoch, "epoch_rebound": rebind,
            "epoch_rebind_reason": (epoch_reason.strip() if rebind else None),
            "source_measurement_surface": source_surface,
            "surface_rebound": surface_rebound,
            "surface_rebind_reason": (surface_reason.strip() if surface_rebound else None),
            "lost_in": lost_record,
            "patch_source": {"file": str(patch.resolve()), "sha256": _sha256(raw),
                             "bytes": len(raw), "insertions": insertions,
                             "deletions": deletions, "touched_paths": list(touched),
                             "base": base,
                             "base_file": None if base_file is None else str(base_file.resolve())},
            "tool": "autokernel.loop.resume backfill-critic2"},
        # experiments._attempt_id prefers this key: re-running the backfill is a no-op.
        "proposal_sha256": _sha256(f"resume-backfill-critic2\n{source_checkpoint}\n"
                                   f"{pointer['patch_sha256']}\n{epoch}".encode()),
    }
    attempt["spawn_parent"] = anchor
    for key in ("branch_id", "width", "depth", "research_scope", "cpu_screen"):
        if payload.get(key) is not None:
            attempt[key] = payload[key]
    checks: dict[str, Any] = {
        "patch_sha256_matches_sidecar": sidecar["patch_sha256"] == pointer["patch_sha256"],
        "sidecar_original_head_is_anchor": sidecar["original_head"] == anchor,
        "base_is_checkpoint_anchor": True,
        "patch_already_retained": already_retained,
        "epoch_rebound": rebind,
        "surface_rebound": surface_rebound,
        **_live_status(store_root),
    }
    checks["epoch_matches_live_loop_status"] = checks["live_loop_status_epoch"] == epoch
    checks["surface_matches_live_loop_status"] = (
        checks["live_loop_status_surface"] == target.get("measurement_surface"))
    if repo is not None:
        try:
            patch_applies(repo, anchor, raw, scratch=scratch)
            checks["applies_cleanly_at_anchor"] = True
        except ResumeRejected as exc:
            checks["applies_cleanly_at_anchor"] = f"NO: {exc}"
    connection = _connect(store_root, immutable=True)
    try:
        attempt_id = experiments._attempt_id(attempt, campaign_id=row["campaign_id"])
        present = connection.execute("SELECT 1 FROM experiments WHERE attempt_id=?",
                                     (attempt_id,)).fetchone() is not None
    finally:
        connection.close()
    return {"source_attempt_id": source, "source_checkpoint_id": source_checkpoint,
            "epoch_sha256": epoch, "campaign_id": row["campaign_id"],
            "attempt_id": attempt_id, "checkpoint_id": f"{attempt_id}#0",
            "already_present": present, "attempt": attempt, "checks": checks,
            "patch_bytes": raw, "sidecar": sidecar, "lane": lane}


def backfill_critic2_apply(store_root: Path, plan: Mapping[str, Any]) -> dict:
    """The only writes: retain the patch (immutable, content-addressed; a no-op when
    already there) and append the planned row (idempotent on its attempt id)."""
    from . import archive
    checkpoint = plan["attempt"]["resume_checkpoints"][0]
    pointer = checkpoint["retained_patch"]
    kept = archive.retain_patch_bytes(
        store_root, plan["patch_bytes"], head=checkpoint["anchor_commit"], lane=plan["lane"],
        mechanism_id=checkpoint["hypothesis"]["mechanism_id"],
        worktree=plan["sidecar"]["worktree"])
    if str(kept.resolve()) != pointer["patch_file"]:
        raise ResumeRejected("patch_pointer", f"retained at {kept}, planned "
                             f"{pointer['patch_file']}")
    verify_retained_patch(pointer, anchor_commit=checkpoint["anchor_commit"],
                          mechanism_id=checkpoint["hypothesis"]["mechanism_id"])
    with experiments.ExperimentStore(store_root) as store:
        added = store.record(plan["attempt"], epoch=plan["epoch_sha256"],
                             recorded_at=_now(), campaign_id=plan["campaign_id"])
        store.write_markdown(epoch=plan["epoch_sha256"])
    return {"retained_patch": str(kept.resolve()), "appended": added,
            "attempt_id": plan["attempt_id"]}


# ------------------------------------------------------------------ reopen


def _immutable_db(path: Path) -> sqlite3.Connection | None:
    if not path.is_file():
        return None
    connection = sqlite3.connect(path.resolve().as_uri() + "?immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def reopen_plan(store_root: Path, *, checkpoint_id: str, reason: str,
                anchor_commit: str | None = None,
                rules_fingerprint: str | None = None) -> dict:
    """Read-only (every database opened immutable=1). What `--apply` would change.

    Also reports whether the next launch would actually queue the checkpoint once
    it is released (its row, eligibility and retained-patch verification), and the
    dispatch-registry rows of the exact candidate, so a reopen is never a no-op in
    disguise.
    """
    store_root = Path(store_root)
    if not reason or not reason.strip():
        raise ValueError("a reopen needs a reason")
    attempt_id, sep, index = checkpoint_id.rpartition("#")
    if not sep or not attempt_id or not index.isdigit():
        raise ValueError(f"checkpoint must be <attempt_id>#<index>, not {checkpoint_id!r}")
    with ClaimLedger(store_root, read_only=True) as ledger:
        rows = [row for row in ledger.rows() if row["checkpoint_id"] == checkpoint_id
                and (anchor_commit is None or row["anchor_commit"] == anchor_commit)]
        events = ledger.events(checkpoint_id)
    if not rows:
        raise ValueError(f"no claim for {checkpoint_id}"
                         + (f" at anchor {anchor_commit}" if anchor_commit else ""))
    if len(rows) > 1:
        raise ValueError(f"{checkpoint_id} is claimed at {len(rows)} anchors; pass --anchor "
                         f"({', '.join(row['anchor_commit'] for row in rows)})")
    current = dict(rows[0])
    current.setdefault("retries", 0)
    plan: dict[str, Any] = {
        "checkpoint_id": checkpoint_id, "anchor_commit": current["anchor_commit"],
        "current": current, "events_so_far": events,
        "refused": ("already released (resumable): nothing to reopen"
                    if current["state"] == RELEASED else None),
        "would_set": {"state": RELEASED,
                      "detail": f"reopened by operator: {reason}"[:2000],
                      "retries": int(current.get("retries") or 0)},
        "would_append_event": {"event": "reopened", "prior_state": current["state"],
                               "prior_result_status": current.get("result_status"),
                               "retries": int(current.get("retries") or 0),
                               "actor": "operator", "reason": reason},
    }
    checkpoint: dict[str, Any] = {"row": None}
    identities: set[str] = set()
    connection = _immutable_db(store_root / "experiments.db")
    if connection is not None:
        try:
            row = connection.execute(
                "SELECT attempt_id, recorded_at, status, mechanism_id, epoch_sha256, payload "
                "FROM experiments WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is not None:
                payload = json.loads(row["payload"])
                entries = payload.get("resume_checkpoints") or []
                entry = entries[int(index)] if int(index) < len(entries) else None
                checkpoint["row"] = {"attempt_id": row["attempt_id"], "status": row["status"],
                                     "recorded_at": row["recorded_at"],
                                     "mechanism_id": row["mechanism_id"]}
                if isinstance(entry, dict):
                    entry = dict(entry)
                    if entry.get("stage") == "build" and not entry.get("retained_patch") \
                            and isinstance(payload.get("retained_patch"), dict):
                        entry["retained_patch"] = dict(payload["retained_patch"])
                    candidate = Candidate(checkpoint_id, row["attempt_id"], row["recorded_at"],
                                          row["status"], row["mechanism_id"], entry)
                    checkpoint.update(candidate.summary())
                    checkpoint["epoch_matches_claim"] = (
                        entry.get("epoch_sha256") in (None, current.get("epoch_sha256")))
                    checkpoint["ineligible_now"] = ineligible_reason(
                        candidate, rules_fingerprint=(rules_fingerprint
                                                      or loop.gate_rules_fingerprint()))
                    if entry.get("stage") in PATCH_STAGES:
                        try:
                            verify_retained_patch(entry.get("retained_patch"),
                                                  anchor_commit=current["anchor_commit"],
                                                  mechanism_id=candidate.group[0])
                            checkpoint["retained_patch_verifies"] = True
                        except ResumeRejected as exc:
                            checkpoint["retained_patch_verifies"] = f"NO ({exc.check}): {exc}"
                source = (payload.get("backfilled_from") or {}).get("attempt_id")
                related = [payload]
                if source:
                    found = connection.execute("SELECT payload FROM experiments WHERE "
                                               "attempt_id=?", (source,)).fetchone()
                    if found is not None:
                        related.append(json.loads(found["payload"]))
                related.extend(json.loads(item["payload"]) for item in connection.execute(
                    "SELECT payload FROM experiments "
                    "WHERE json_extract(payload, '$.resumed_from') = ?", (checkpoint_id,)))
                identities = {item["attempt_identity"] for item in related
                              if isinstance(item.get("attempt_identity"), str)}
        finally:
            connection.close()
    plan["checkpoint"] = checkpoint
    registry: list[dict] = []
    connection = _immutable_db(store_root / "dispatch-identity.sqlite3")
    if connection is not None and identities:
        try:
            from .dispatch_guard import ANSWER_STATUSES
            for identity in sorted(identities):
                found = connection.execute("SELECT * FROM attempts WHERE identity=?",
                                           (identity,)).fetchone()
                if found is None:
                    continue
                entry = dict(found)
                # The one-retry bound an ordinary dispatch would hit; a resumed build
                # is admitted past it unless the identity was answered.
                entry["one_retry_bound_reached"] = int(entry["dispatch_count"]) >= 2
                entry["resumed_build_admitted"] = entry["status"] not in ANSWER_STATUSES
                registry.append(entry)
        finally:
            connection.close()
    plan["dispatch_registry"] = registry
    return plan


def reopen_apply(store_root: Path, plan: Mapping[str, Any], *, reason: str) -> dict:
    """The only write: release the claim and append the reopen event."""
    with ClaimLedger(store_root) as ledger:
        ledger.reopen(plan["checkpoint_id"], plan["anchor_commit"], reason=reason)
        return next(row for row in ledger.rows()
                    if row["checkpoint_id"] == plan["checkpoint_id"]
                    and row["anchor_commit"] == plan["anchor_commit"])


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
    crit = commands.add_parser("backfill-critic2", help="turn an author patch saved "
                               "outside the store into a resumable critic2 record "
                               "(dry-run unless --apply)")
    crit.add_argument("--store", type=Path, required=True)
    crit.add_argument("--row", required=True, help="row carrying the AUTHOR checkpoint "
                      "(hypothesis + accepted critic:hypothesis): attempt id or prefix")
    crit.add_argument("--checkpoint-index", type=int, help="author checkpoint index on "
                      "--row when it carries several")
    crit.add_argument("--patch", type=Path, required=True, help="the saved author patch")
    crit.add_argument("--base", help="commit the patch was formed on "
                      "(default: base.txt beside --patch)")
    crit.add_argument("--epoch", required=True, help="epoch the NEXT launch runs in "
                      "(store/loop-status.json epoch_sha256)")
    crit.add_argument("--epoch-reason", help="required when --epoch differs from the "
                      "source row's epoch; recorded in the row")
    crit.add_argument("--surface", help="rebind the checkpoint's measurement surface "
                      "(default: the source checkpoint's)")
    crit.add_argument("--surface-reason", help="required with a differing --surface; "
                      "recorded in the row")
    crit.add_argument("--lost-in-row", help="the row whose round lost this patch "
                      "(recorded as provenance; must carry the same hypothesis)")
    crit.add_argument("--lane", default="lane0", help="lane label for the retained file")
    crit.add_argument("--repo", type=Path, help="git repo holding the anchor; enables the "
                      "read-only clean-apply check")
    crit.add_argument("--scratch", type=Path, help="scratch directory for the apply check")
    crit.add_argument("--apply", action="store_true", help="retain the patch and append "
                      "the row (the only writes)")
    rein = commands.add_parser("reinstate", help="re-pend an ACCEPTED hypothesis a "
                               "pre-policy iteration dropped after its patch rounds ran out "
                               "(dry-run unless --apply)")
    rein.add_argument("--store", type=Path, required=True)
    rein.add_argument("--row", required=True, help="row carrying the hypothesis's latest "
                      "checkpoint: attempt id or prefix")
    rein.add_argument("--checkpoint-index", type=int)
    rein.add_argument("--rejection", action="append", default=[], required=True,
                      help="a patch_rejected/gate_refused row of this hypothesis (repeat)")
    rein.add_argument("--epoch", required=True, help="epoch the NEXT launch runs in "
                      "(store/loop-status.json epoch_sha256)")
    rein.add_argument("--epoch-reason")
    rein.add_argument("--measurement-epoch", help="stamp the checkpoint's "
                      "measurement_epoch_sha256 (what a measurement-epoch launch binds on)")
    rein.add_argument("--status", default=loop.PATCH_ROUNDS_EXHAUSTED,
                      choices=sorted(loop.PENDING_HYPOTHESIS_STATUSES))
    rein.add_argument("--scope-rule", help="with --status scope_blocked: the route/rule")
    rein.add_argument("--attempts-used", type=int, help="authoring attempts already spent "
                      "(default: distinct attempts among the --rejection rows)")
    rein.add_argument("--budget", type=int, default=loop.HYPOTHESIS_AUTHOR_ATTEMPTS)
    rein.add_argument("--reason", help="row reason (default: a generated summary)")
    rein.add_argument("--apply", action="store_true", help="append the row (the only write)")
    look = commands.add_parser("scan", help="read-only: what a launch would resume")
    look.add_argument("--store", type=Path, required=True)
    look.add_argument("--epoch", required=True)
    look.add_argument("--anchor", required=True)
    look.add_argument("--surface")
    look.add_argument("--model")
    look.add_argument("--repo", type=Path)
    look.add_argument("--scratch", type=Path)
    again = commands.add_parser("reopen", help="operator: hand a consumed resume claim "
                                "back, with a logged reason (dry-run unless --apply)")
    again.add_argument("--store", type=Path, required=True)
    again.add_argument("--checkpoint", required=True, help="<attempt_id>#<index>")
    again.add_argument("--reason", required=True)
    again.add_argument("--anchor", help="anchor commit, when claimed at several")
    again.add_argument("--apply", action="store_true",
                       help="release the claim and log the event (the only write)")
    args = parser.parse_args(argv)
    if args.command == "reopen":
        try:
            plan = reopen_plan(args.store, checkpoint_id=args.checkpoint, reason=args.reason,
                               anchor_commit=args.anchor)
        except (ValueError, FileNotFoundError, sqlite3.DatabaseError) as exc:
            print(f"reopen refused: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(plan, indent=2, default=str))
        if plan["refused"]:
            print(f"reopen refused: {plan['refused']}", file=sys.stderr)
            return 2
        if not args.apply:
            print("dry-run: nothing written (pass --apply to release the claim)")
            return 0
        try:
            row = reopen_apply(args.store, plan, reason=args.reason)
        except ValueError as exc:
            print(f"reopen refused: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"reopened": row}, indent=2, default=str))
        return 0
    if args.command == "reinstate":
        try:
            plan = reinstate_plan(args.store, row_id=args.row, rejection_rows=args.rejection,
                                  epoch=args.epoch, epoch_reason=args.epoch_reason,
                                  checkpoint_index=args.checkpoint_index, status=args.status,
                                  scope_rule=args.scope_rule, attempts_used=args.attempts_used,
                                  budget=args.budget, reason=args.reason,
                                  measurement_epoch=args.measurement_epoch)
        except (ValueError, FileNotFoundError, ResumeRejected, sqlite3.DatabaseError) as exc:
            print(f"reinstate refused: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({key: plan[key] for key in (
            "source_attempt_id", "source_checkpoint_id", "attempt_id", "checkpoint_id",
            "already_present", "epoch_sha256", "campaign_id", "checks")}, indent=2,
            default=str))
        print(json.dumps({"hypothesis_pending": plan["attempt"]["hypothesis_pending"],
                          "resume_checkpoints": plan["attempt"]["resume_checkpoints"],
                          "reinstated_from": plan["attempt"]["reinstated_from"]}, indent=2))
        if plan["checks"]["prevalidates_at_anchor"] is not True:
            print("reinstate refused: the planned checkpoint does not prevalidate",
                  file=sys.stderr)
            return 2
        if not args.apply:
            print("dry-run: nothing written (pass --apply to append this row)")
            return 0
        added = reinstate_apply(args.store, plan)
        print(f"{'appended' if added else 'already present'} {plan['attempt_id']}")
        return 0
    if args.command == "backfill-critic2":
        try:
            plan = backfill_critic2_plan(
                args.store, row_id=args.row, patch=args.patch, epoch=args.epoch,
                epoch_reason=args.epoch_reason, base=args.base, lost_in_row=args.lost_in_row,
                lane=args.lane, checkpoint_index=args.checkpoint_index, repo=args.repo,
                scratch=args.scratch, surface=args.surface,
                surface_reason=args.surface_reason)
        except (ValueError, FileNotFoundError, ResumeRejected, sqlite3.DatabaseError) as exc:
            print(f"backfill-critic2 refused: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({key: plan[key] for key in (
            "source_attempt_id", "source_checkpoint_id", "attempt_id", "checkpoint_id",
            "already_present", "epoch_sha256", "campaign_id", "checks")}, indent=2))
        print(json.dumps({"resume_checkpoints": plan["attempt"]["resume_checkpoints"],
                          "backfilled_from": plan["attempt"]["backfilled_from"],
                          "would_retain": {"patch_file": plan["attempt"]["resume_checkpoints"]
                                           [0]["retained_patch"]["patch_file"],
                                           "sidecar": plan["sidecar"]}}, indent=2))
        if plan["checks"].get("applies_cleanly_at_anchor", True) is not True:
            print("backfill-critic2 refused: the patch does not apply at the anchor",
                  file=sys.stderr)
            return 2
        if not args.apply:
            print("dry-run: nothing written (pass --apply to retain the patch and append "
                  "this row)")
            return 0
        done = backfill_critic2_apply(args.store, plan)
        print(f"{'appended' if done['appended'] else 'already present'} "
              f"{done['attempt_id']}; patch retained at {done['retained_patch']}")
        return 0
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


__all__ = ["BACKFILL_SCHEMA", "CLAIMS_FILE", "CRITIC2_BACKFILL_SCHEMA", "Candidate",
           "PENDING_SUMMARY_LIMIT", "REINSTATE_SCHEMA", "ROUND_DISPOSITIONS", "depth_limit",
           "pending_hypotheses", "reinstate_apply", "reinstate_plan",
           "ClaimLedger", "INFRA_RETRIES", "PATCH_STAGES", "STAGE_RANK",
           "backfill_critic2_apply", "backfill_critic2_plan", "retain_checkpoint_patches",
           "INFRASTRUCTURE_STATUSES", "MAX_RESUME_DEPTH", "RELEASED", "RULE_GATES",
           "ResumePoint", "ResumeQueue", "ResumeRejected", "backfill_apply",
           "backfill_plan", "bind_checkpoints", "discard", "ineligible_reason", "materialize",
           "patch_applies", "patch_paths", "prepare", "prevalidate", "preview_op_scope",
           "rejection_attempt", "reopen_apply", "reopen_plan", "resume_point", "scan",
           "target_identity",
           "verify_retained_patch"]


if __name__ == "__main__":
    raise SystemExit(main())
