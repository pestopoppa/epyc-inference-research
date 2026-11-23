"""Journal-backed phase boundaries for the current pooled loop.

Phase records are operational history, never scientific evidence or launch grants.
An interrupted measurement is invalid; an unresolved integration requires explicit
reconciliation before a new pool can start.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
import secrets
import stat
import threading
from pathlib import Path
from typing import Any, Mapping

from .. import journal, schemas

SCHEMA = "epyc.autokernel.loop_phase.v1"
PHASES = frozenset({"iteration", "gate", "measurement", "integration"})


class PhaseJournalError(RuntimeError):
    pass


def validate(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "event", "phase", "attempt_id", "draw_id", "lane", "parent", "mechanism", "hypothesis", "hypothesis_digest", "mutation", "result"}
    if not isinstance(row, Mapping) or set(row) != fields or row.get("schema") != SCHEMA:
        raise PhaseJournalError("phase event fields/schema differ")
    if (not isinstance(row["event"], str) or not isinstance(row["phase"], str)
            or row["event"] not in {"BEGIN", "MUTATION_INTENT", "COMPLETE", "INVALID", "UNKNOWN", "RECONCILED", "ROLLED_BACK"}
            or row["phase"] not in PHASES):
        raise PhaseJournalError("phase event kind differs")
    for field in ("attempt_id", "draw_id", "lane", "parent", "mechanism"):
        if not isinstance(row[field], str) or not row[field] or len(row[field]) > 512:
            raise PhaseJournalError(f"invalid phase {field}")
    hypothesis = row["hypothesis"]
    if not isinstance(hypothesis, Mapping) or hypothesis.get("mechanism_id") != row["mechanism"]:
        raise PhaseJournalError("invalid original hypothesis")
    digest = row["hypothesis_digest"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise PhaseJournalError("invalid hypothesis digest")
    try:
        if schemas.content_hash(dict(hypothesis)) != digest:
            raise PhaseJournalError("hypothesis provenance digest differs")
    except (TypeError, ValueError, RecursionError) as exc:
        raise PhaseJournalError("hypothesis provenance is not canonical JSON") from exc
    mutation = row["mutation"]
    if mutation is not None:
        if (not isinstance(mutation, Mapping) or not isinstance(mutation.get("kind"), str)
                or mutation.get("kind") not in {"source", "runtime"}):
            raise PhaseJournalError("invalid mutation binding")
        fields = ({"kind", "repo", "branch", "candidate_tree"} if mutation["kind"] == "source"
                  else {"kind", "anchor_snapshot", "candidate_snapshot"})
        if set(mutation) != fields or any(not isinstance(value, str) or not value
                                          for value in mutation.values()):
            raise PhaseJournalError("mutation binding fields differ")
        if mutation["kind"] == "source" and not os.path.isabs(mutation["repo"]):
            raise PhaseJournalError("source mutation requires an absolute repository")
        pins = ([mutation["candidate_tree"]] if mutation["kind"] == "source" else
                [mutation["anchor_snapshot"], mutation["candidate_snapshot"]])
        if any(len(pin) not in ({40, 64} if mutation["kind"] == "source" else {64})
               or any(c not in "0123456789abcdef" for c in pin) for pin in pins):
            raise PhaseJournalError("mutation identity is malformed")
    if row["phase"] != "integration" and mutation is not None:
        raise PhaseJournalError("non-integration phase has a mutation binding")
    if row["event"] in {"BEGIN", "INVALID"} and mutation is not None:
        raise PhaseJournalError("pre-mutation phase has a mutation binding")
    if row["event"] in {"MUTATION_INTENT", "UNKNOWN", "RECONCILED", "ROLLED_BACK"} and mutation is None:
        raise PhaseJournalError("integration mutation binding is missing")
    if row["event"] == "COMPLETE" and row["phase"] == "integration" and mutation is None:
        raise PhaseJournalError("completed integration lacks its mutation binding")
    if row["result"] is not None and (not isinstance(row["result"], str) or len(row["result"]) > 512):
        raise PhaseJournalError("invalid phase result")
    if row["event"] in {"BEGIN", "MUTATION_INTENT"} and row["result"] is not None:
        raise PhaseJournalError("phase intent has a result")
    if row["event"] == "COMPLETE" and row["phase"] == "integration":
        if mutation["kind"] == "source":
            head = row["result"]
            if (not isinstance(head, str) or len(head) not in {40, 64}
                    or any(c not in "0123456789abcdef" for c in head)):
                raise PhaseJournalError("source completion lacks its committed identity")
        else:
            path, sep, digest = (row["result"] or "").rpartition("#")
            if (not sep or not os.path.isabs(path) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise PhaseJournalError("runtime completion lacks its selection receipt")
    if row["event"] == "RECONCILED":
        decision, sep, reference = (row["result"] or "").partition(":")
        path, hash_sep, digest = reference.rpartition("#")
        if (decision not in {"applied", "not_applied"} or not sep or not hash_sep
                or not os.path.isabs(path) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise PhaseJournalError("invalid reconciliation proof")
    if row["event"] == "ROLLED_BACK":
        path, sep, proof = (row["result"] or "").rpartition("#")
        if (row["phase"] != "integration" or mutation is None or mutation["kind"] != "source"
                or not sep or not os.path.isabs(path) or len(proof) != 64
                or any(c not in "0123456789abcdef" for c in proof)):
            raise PhaseJournalError("invalid rollback receipt")
    if row["event"] == "MUTATION_INTENT" and row["phase"] != "integration":
        raise PhaseJournalError("only integration may mark mutation")
    if row["event"] in {"UNKNOWN", "RECONCILED"} and row["phase"] != "integration":
        raise PhaseJournalError("only integration may be unresolved")
    return dict(row)


def _advance(phases, row):
    key = row["attempt_id"]
    prior = phases.get(key)
    if row["event"] == "BEGIN":
        if prior is not None:
            raise PhaseJournalError("duplicate phase begin")
    else:
        if (prior is None or any(prior[name] != row[name]
                for name in ("phase", "draw_id", "lane", "parent", "mechanism", "hypothesis", "hypothesis_digest"))):
            raise PhaseJournalError("phase completion lacks matching begin")
        expected = ({"BEGIN"} if row["event"] in {"MUTATION_INTENT", "INVALID"} else
                    {"MUTATION_INTENT"} if row["event"] in {"UNKNOWN", "ROLLED_BACK"} else
                    {"MUTATION_INTENT", "UNKNOWN"} if row["event"] == "RECONCILED" else
                    {"MUTATION_INTENT"} if row["phase"] == "integration" else {"BEGIN"})
        if prior["event"] not in expected:
            raise PhaseJournalError("invalid phase transition")
        if prior["event"] in {"MUTATION_INTENT", "UNKNOWN"} and prior["mutation"] != row["mutation"]:
            raise PhaseJournalError("integration mutation binding changed")
    phases[key] = row


def _read_proof(store: Path, path: str, digest: str) -> dict[str, Any]:
    try:
        if not Path(path).resolve().is_relative_to(Path(store).resolve()):
            raise PhaseJournalError("integration proof is outside the store")
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, PhaseJournalError):
            raise
        raise PhaseJournalError(f"integration proof path is invalid: {exc}") from exc
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o022 or info.st_nlink != 1
                    or info.st_size > 64 * 1024 * 1024):
                raise PhaseJournalError("integration proof identity is unsafe")
            chunks = []
            remaining = info.st_size
            while remaining:
                chunk = os.read(fd, min(remaining, 1024 * 1024))
                if not chunk:
                    raise PhaseJournalError("integration proof ended early")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise PhaseJournalError("integration proof grew during read")
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise PhaseJournalError(f"integration proof unavailable: {exc}") from exc
    raw = b"".join(chunks)
    if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or hashlib.sha256(raw).hexdigest() != digest):
        raise PhaseJournalError("integration proof digest differs")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate proof key")
            out[key] = value
        return out
    try:
        body = json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              ValueError(f"non-finite token {token}")))
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise PhaseJournalError("integration proof is not strict JSON") from exc
    if not isinstance(body, dict):
        raise PhaseJournalError("integration proof is not an object")
    return body


def _verify_terminal(row: Mapping[str, Any], store: Path) -> None:
    if (row["event"] == "COMPLETE" and row["phase"] == "integration"
            and row["mutation"]["kind"] == "runtime"):
        path, _, digest = row["result"].rpartition("#")
        body = _read_proof(store, path, digest)
        recipe = body.get("current_recipe")
        if (body.get("schema") not in {"epyc.autokernel.direct_runtime_selection.v1",
                                      "epyc.autokernel.keep_grade_runtime_selection.v1"}
                or not isinstance(recipe, dict)
                or recipe.get("snapshot_digest") != row["mutation"]["candidate_snapshot"]):
            raise PhaseJournalError("runtime selection receipt differs from the integration")
    elif row["event"] == "ROLLED_BACK":
        path, _, digest = row["result"].rpartition("#")
        body = _read_proof(store, path, digest)
        if (body.get("schema") != "epyc.autokernel.accumulator_comparison_resolution.v1"
                or body.get("outcome") != "interaction_regression"
                or body.get("mechanism_id") != row["mechanism"]
                or body.get("prior_tip") != row["parent"]
                or body.get("candidate_tree") != row["mutation"]["candidate_tree"]
                or not isinstance(body.get("candidate_tip"), str)
                or len(body["candidate_tip"]) not in {40, 64}
                or any(c not in "0123456789abcdef" for c in body["candidate_tip"])):
            raise PhaseJournalError("rollback receipt does not bind the integration")
        for field, phase, tip in (("first", "initial", body["candidate_tip"]),
                                  ("repeat", "repeat", body["candidate_tip"]),
                                  ("restored_tip_recheck", "restored-prior-tip", row["parent"])):
            ref = body.get(field)
            if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                    or not isinstance(ref["path"], str) or not os.path.isabs(ref["path"])
                    or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in ref["sha256"])):
                raise PhaseJournalError("rollback comparison reference is malformed")
            comparison = _read_proof(store, ref["path"], ref["sha256"])
            if (comparison.get("schema") != "epyc.autokernel.accumulator_comparison.v1"
                    or comparison.get("phase") != phase
                    or comparison.get("mechanism_id") != row["mechanism"]
                    or comparison.get("prior_tip") != row["parent"]
                    or comparison.get("measured_tip") != tip
                    or not isinstance(comparison.get("comparison"), dict)):
                raise PhaseJournalError("rollback comparison does not bind the integration")
    elif row["event"] == "RECONCILED":
        decision, _, reference = row["result"].partition(":")
        path, _, digest = reference.rpartition("#")
        body = _read_proof(store, path, digest)
        if body != {"schema": "epyc.autokernel.integration_reconciliation.v1",
                    "attempt_id": row["attempt_id"], "parent": row["parent"],
                    "mutation": row["mutation"], "disposition": decision}:
            raise PhaseJournalError("reconciliation receipt does not bind the integration")


def project(entries, *, store: Path | None = None) -> dict[str, dict[str, Any]]:
    phases: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.kind == journal.KIND_LOOP_PHASE:
            row = validate(entry.payload)
            if (row["event"] in {"ROLLED_BACK", "RECONCILED"}
                    or (row["event"] == "COMPLETE" and row["phase"] == "integration"
                        and row["mutation"]["kind"] == "runtime")):
                if store is None:
                    raise PhaseJournalError("integration proof store is required")
                _verify_terminal(row, store)
            _advance(phases, row)
    return {key: {**row, "recovery_state": (
                "unresolved_integration" if row["phase"] == "integration"
                and row["event"] in {"MUTATION_INTENT", "UNKNOWN"} else
                "invalid_interrupted" if row["event"] == "BEGIN" else row["event"].lower())}
            for key, row in phases.items()}


def _verify_shards(book):
    for shard in book.shards():
        info = os.lstat(shard.path)
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise PhaseJournalError("journal shard identity is unsafe")


def _verified_entries(book):
    _verify_shards(book)
    report = book.scan(validate_payloads=True)
    if report.defects or report.torn_tail is not None:
        raise PhaseJournalError("journal has invalid records or a torn tail")
    ordered = sorted(report.entries, key=lambda item: (item.shard_index, item.line_number))
    if any(right.seq <= left.seq for left, right in zip(ordered, ordered[1:])):
        raise PhaseJournalError("journal sequence is not strictly increasing")
    return ordered


class PhaseJournal:
    def __init__(self, book: journal.Journal, *, allow_reconcile: bool = False):
        self.book = book
        self.lock = threading.RLock()
        self.poisoned = False
        self.local = threading.local()
        root = os.lstat(book.root)
        if not stat.S_ISDIR(root.st_mode) or stat.S_ISLNK(root.st_mode) or root.st_uid != os.getuid():
            raise PhaseJournalError("phase journal root identity is unsafe")
        self.root_identity = (root.st_dev, root.st_ino, root.st_uid, root.st_mode)
        self.lease_path = os.path.join(book.root, ".loop-phase-owner.lock")
        self.lease = os.open(self.lease_path,
                             os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            info = os.fstat(self.lease)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise PhaseJournalError("phase owner lock identity is unsafe")
            try:
                fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PhaseJournalError("another loop owns the phase journal") from exc
            self.lease_identity = (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink)
            with book.write_lock():
                self.phases = project(_verified_entries(book), store=Path(book.root).parent)
            if self._unresolved() and not allow_reconcile:
                raise PhaseJournalError("unresolved integration requires reconciliation")
        except BaseException:
            os.close(self.lease)
            self.lease = -1
            raise

    def close(self):
        with self.lock:
            if self.lease >= 0:
                os.close(self.lease)
                self.lease = -1
    def _unresolved(self):
        return [row for row in self.phases.values()
                if row["phase"] == "integration"
                and row["event"] in {"MUTATION_INTENT", "UNKNOWN"}]

    def reconcile(self, attempt_id: str, verify):
        """Resolve an interrupted integration only through an owner-supplied verifier.

        The verifier reopens the retained refs and returns `disposition`, `path`,
        and `sha256`. The store-local JSON proof binds the attempt, original parent,
        mutation, and disposition under `integration_reconciliation.v1`. This
        operation does not advance a pointer or confer scientific evidence.
        """
        if not callable(verify):
            raise PhaseJournalError("reconciliation requires a verifier")
        with self.lock:
            row = self.phases.get(attempt_id)
            if row is None or (row not in self._unresolved() and row["event"] != "RECONCILED"):
                raise PhaseJournalError("integration is not unresolved")
            proof = verify(copy.deepcopy(row))
            if (not isinstance(proof, Mapping) or set(proof) != {"disposition", "path", "sha256"}
                    or not isinstance(proof["disposition"], str)
                    or proof["disposition"] not in {"applied", "not_applied"}
                    or not isinstance(proof["path"], str) or not os.path.isabs(proof["path"])
                    or not isinstance(proof["sha256"], str) or len(proof["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in proof["sha256"])):
                raise PhaseJournalError("reconciliation verifier returned invalid proof")
            event = {key: row[key] for key in
                     ("schema", "event", "phase", "attempt_id", "draw_id", "lane", "parent", "mechanism", "hypothesis", "hypothesis_digest", "mutation", "result")}
            event.update(event="RECONCILED", result=proof["disposition"] + ":" + proof["path"] + "#" + proof["sha256"])
            validate(event)
            _verify_terminal(event, Path(self.book.root).parent)
            if row["event"] == "RECONCILED":
                if row["result"] != event["result"]:
                    raise PhaseJournalError("reconciliation retry conflicts")
                return
            self._append(event)
            try:
                _verify_terminal(event, Path(self.book.root).parent)
            except PhaseJournalError:
                self.poisoned = True
                raise

    def _append(self, row):
        from .loop import RunAborted
        with self.lock:
            if self.poisoned or self.lease < 0:
                raise RunAborted("phase journal is closed or requires recovery")
            try:
                root = os.lstat(self.book.root)
                if (root.st_dev, root.st_ino, root.st_uid, root.st_mode) != self.root_identity:
                    raise PhaseJournalError("phase journal root identity changed")
                held = os.fstat(self.lease)
                named = os.lstat(self.lease_path)
                identity = lambda st: (st.st_dev, st.st_ino, st.st_uid, st.st_mode, st.st_nlink)
                if identity(held) != self.lease_identity or identity(named) != self.lease_identity:
                    raise PhaseJournalError("phase owner lock identity changed")
                if row.get("event") == "BEGIN" and self._unresolved():
                    raise RunAborted("unresolved integration requires reconciliation")
                _verify_shards(self.book)
                candidate = validate(row)
                prospective = dict(self.phases)
                _advance(prospective, candidate)
                self.book.append(journal.KIND_LOOP_PHASE, candidate)
                self.phases = prospective
            except RunAborted:
                raise
            except Exception as exc:
                self.poisoned = True
                raise RunAborted(f"phase journal failed: {exc}") from exc

    def mark_integration(self, mutation: Mapping[str, str]):
        row = getattr(self.local, "integration", None)
        if row is None:
            raise PhaseJournalError("integration mutation lacks its phase")
        binding = dict(mutation)
        self._append({**row, "event": "MUTATION_INTENT", "mutation": binding})
        row["mutation"] = binding
        self.local.mutating = True

    def bind_runtime_completion(self, receipt: Mapping[str, str]):
        row = getattr(self.local, "integration", None)
        if (row is None or not getattr(self.local, "mutating", False)
                or row["mutation"]["kind"] != "runtime"
                or not isinstance(receipt, Mapping) or set(receipt) != {"path", "sha256"}):
            raise PhaseJournalError("runtime selection lacks its integration intent")
        event = {**row, "event": "COMPLETE",
                 "result": str(receipt["path"]) + "#" + str(receipt["sha256"])}
        validate(event)
        _verify_terminal(event, Path(self.book.root).parent)
        self.local.runtime_completion = event["result"]

    def mutation_binding(self) -> dict[str, str]:
        row = getattr(self.local, "integration", None)
        if row is None or not getattr(self.local, "mutating", False):
            raise PhaseJournalError("no active integration mutation")
        return copy.deepcopy(row["mutation"])

    def mark_rollback(self, receipt: Mapping[str, str]):
        row = getattr(self.local, "integration", None)
        if row is None or not getattr(self.local, "mutating", False):
            raise PhaseJournalError("rollback lacks a pending integration")
        if not isinstance(receipt, Mapping) or set(receipt) != {"path", "sha256"}:
            raise PhaseJournalError("rollback receipt fields differ")
        event = {**row, "event": "ROLLED_BACK",
                 "result": str(receipt["path"]) + "#" + str(receipt["sha256"])}
        validate(event)
        _verify_terminal(event, Path(self.book.root).parent)
        self._append(event)
        try:
            _verify_terminal(event, Path(self.book.root).parent)
        except PhaseJournalError:
            self.poisoned = True
            raise
        self.local.rolled_back = True

    @contextmanager
    def phase(self, phase: str, lane: str, parent: str, mechanism: str,
              draw_id: str, hypothesis: Mapping[str, Any]):
        row = {"schema": SCHEMA, "event": "BEGIN", "phase": phase,
               "attempt_id": secrets.token_hex(16), "draw_id": draw_id, "lane": lane,
               "parent": parent, "mechanism": mechanism,
               "hypothesis": dict(hypothesis), "hypothesis_digest": schemas.content_hash(dict(hypothesis)),
               "mutation": None, "result": None}
        self._append(row)
        result = {"value": None}
        if phase == "integration":
            self.local.integration = row
            self.local.mutating = False
            self.local.rolled_back = False
            self.local.runtime_completion = None
        try:
            yield result
        except BaseException as exc:
            if phase == "integration" and self.local.rolled_back:
                raise
            uncertain = phase == "integration" and self.local.mutating
            self._append({**row, "event": "UNKNOWN" if uncertain else "INVALID",
                          "result": type(exc).__name__})
            if uncertain:
                from .loop import RunAborted
                raise RunAborted("integration outcome is unresolved; reconcile journal and refs") from exc
            raise
        else:
            if phase == "integration" and not self.local.mutating:
                from .loop import RunAborted
                self._append({**row, "event": "INVALID", "result": "mutation_not_marked"})
                raise RunAborted("integration finished without a mutation intent")
            value = result["value"]
            if phase == "integration" and row["mutation"]["kind"] == "runtime":
                value = self.local.runtime_completion
            if value is not None and not isinstance(value, str):
                value = str(value)
            event = {**row, "event": "COMPLETE", "result": value}
            try:
                validate(event)
                if phase == "integration" and row["mutation"]["kind"] == "runtime":
                    _verify_terminal(event, Path(self.book.root).parent)
            except PhaseJournalError as exc:
                self._append({**row, "event": "UNKNOWN", "result": type(exc).__name__})
                from .loop import RunAborted
                raise RunAborted("integration completion is not provable") from exc
            self._append(event)
        finally:
            if phase == "integration":
                self.local.integration = None
                self.local.mutating = False
                self.local.rolled_back = False
                self.local.runtime_completion = None
