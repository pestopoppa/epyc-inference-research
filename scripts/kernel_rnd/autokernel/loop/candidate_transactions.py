"""Durable candidate transitions under the campaign controller's sole writer.

This module adds no WAL or lock.  Every operation runs through
``CampaignController.candidate_transaction`` and uses its already-held supervisor
lease plus the existing Journal.  Git preparation only retains exact, pre-existing
commits under an owned experimental ref namespace; it never commits source, moves a
branch, touches production refs, or creates measurement authority.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .. import journal as journal_module
from . import candidate_manifest as cm
from . import campaign_control as campaign_control_module
from . import measurement_capture as mc
from . import status


OBJECT_DIR = "candidate-objects"
POINTER_FILE = "candidate-state.json"
POINTER_SCHEMA = "epyc.autokernel.candidate_state_pointer.v1"
OWNED_REF_PREFIX = "refs/autokernel/candidates"
TRANSITION_RECEIPT_SCHEMA = "epyc.autokernel.candidate_transition_receipt.v1"
FROZEN_PRODUCTION_ROOTS = frozenset(Path(value) for value in (
    "/mnt/raid0/llm/llama.cpp", "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp",
))
_FROZEN_BRANCH_PREFIXES = ("production-consolidated-", "production-speech-")
_GIT_REDIRECT_ENV = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_INDEX_FILE",
})


class CandidateTransactionError(RuntimeError):
    """A candidate mutation was invalid or could not be prepared safely."""


class CandidateRecoveryRequired(CandidateTransactionError):
    """Durable intent cannot be completed without restoring named objects/refs."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise CandidateTransactionError(f"value is not canonical finite JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateTransactionError(f"{label} must be a non-empty string")
    return value


def _intent_digest(operation: str, expected_state_digest: str | None,
                   operation_payload: Mapping[str, Any],
                   prepared_objects: Sequence[Mapping[str, str]],
                   prepared_refs: Sequence[Mapping[str, str]]) -> str:
    return _digest({"operation": operation,
                    "expected_state_digest": expected_state_digest,
                    "operation_payload": dict(operation_payload),
                    "prepared_objects": [dict(item) for item in prepared_objects],
                    "prepared_refs": [dict(item) for item in prepared_refs]})


def _state_digest(state: cm.CandidateState) -> str:
    return _digest(state.to_dict())


def _run_git(repo: Path, *args: str, check: bool = True) -> str:
    env = os.environ.copy()
    for name in _GIT_REDIRECT_ENV:
        env.pop(name, None)
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, timeout=60, env=env)
    if check and done.returncode != 0:
        raise CandidateTransactionError(
            f"git {' '.join(args)}: {(done.stderr or done.stdout).strip()[:300]}")
    return done.stdout.strip()


@dataclass(frozen=True)
class PreparedRef:
    repo_id: str
    ref: str
    commit: str
    tree: str
    object_format: str

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


class GitCandidateBackend:
    """Retain exact existing source commits under an owned immutable namespace."""

    def __init__(self, repos: Mapping[str, Path], *, campaign_id: str) -> None:
        if not isinstance(repos, Mapping):
            raise CandidateTransactionError("validated repositories must be a mapping")
        if any(name in os.environ for name in _GIT_REDIRECT_ENV):
            raise CandidateTransactionError("ambient Git repository redirection is refused")
        self._repos: dict[str, Path] = {}
        self._repo_identities: dict[str, tuple[int, int, int, int]] = {}
        self._git_dirs: dict[str, Path] = {}
        self._git_dir_identities: dict[str, tuple[int, int, int, int]] = {}
        self._common_dirs: dict[str, Path] = {}
        self._common_dir_identities: dict[str, tuple[int, int, int, int]] = {}
        self._planned: dict[tuple[str, str], PreparedRef] = {}
        for repo_id, supplied in repos.items():
            repo_id = _text(repo_id, "repository id")
            if not isinstance(supplied, (str, os.PathLike)):
                raise CandidateTransactionError(
                    f"repository {repo_id!r} path must be path-like")
            path = Path(supplied).resolve()
            actual = Path(_run_git(path, "rev-parse", "--show-toplevel")).resolve()
            if path != actual:
                raise CandidateTransactionError(
                    f"repository {repo_id!r} must name its exact root")
            if path in FROZEN_PRODUCTION_ROOTS:
                raise CandidateTransactionError("canonical frozen production checkout is refused")
            branch = _run_git(path, "symbolic-ref", "--short", "-q", "HEAD",
                              check=False)
            if branch.startswith(_FROZEN_BRANCH_PREFIXES):
                raise CandidateTransactionError(
                    f"repository {repo_id!r} is checked out on a frozen production branch")
            self._repos[repo_id] = path
            info = os.lstat(path)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise CandidateTransactionError(
                    f"repository {repo_id!r} has unsafe root identity")
            self._repo_identities[repo_id] = (
                info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode))
            git_dir = Path(_run_git(path, "rev-parse", "--absolute-git-dir")).resolve()
            git_info = os.lstat(git_dir)
            if not stat.S_ISDIR(git_info.st_mode) or git_info.st_uid != os.getuid():
                raise CandidateTransactionError(
                    f"repository {repo_id!r} has unsafe Git directory identity")
            self._git_dirs[repo_id] = git_dir
            self._git_dir_identities[repo_id] = (
                git_info.st_dev, git_info.st_ino, git_info.st_uid,
                stat.S_IMODE(git_info.st_mode))
            common_dir = Path(_run_git(
                path, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
            common_info = os.lstat(common_dir)
            if not stat.S_ISDIR(common_info.st_mode) or common_info.st_uid != os.getuid():
                raise CandidateTransactionError(
                    f"repository {repo_id!r} has unsafe Git common-directory identity")
            self._common_dirs[repo_id] = common_dir
            self._common_dir_identities[repo_id] = (
                common_info.st_dev, common_info.st_ino, common_info.st_uid,
                stat.S_IMODE(common_info.st_mode))
        self._campaign_token = hashlib.sha256(
            _text(campaign_id, "campaign id").encode()).hexdigest()[:20]

    def plan(self, transaction_id: str,
             sources: Sequence[cm.SourceIdentity]) -> tuple[PreparedRef, ...]:
        transaction_token = hashlib.sha256(
            _text(transaction_id, "transaction id").encode()).hexdigest()
        planned: list[PreparedRef] = []
        for source in sorted(sources, key=lambda item: item.repo_id):
            repo = self._verified_repo(source.repo_id)
            if Path(source.path).resolve() != repo:
                raise CandidateTransactionError(
                    f"source path for {source.repo_id!r} does not match validated repository")
            object_format = _run_git(repo, "rev-parse", "--show-object-format")
            if object_format != source.object_format:
                raise CandidateTransactionError(
                    f"object format changed for repository {source.repo_id!r}")
            commit = _run_git(repo, "rev-parse", f"{source.commit}^{{commit}}")
            tree = _run_git(repo, "rev-parse", f"{source.commit}^{{tree}}")
            if commit != source.commit or tree != source.tree:
                raise CandidateTransactionError(
                    f"source object identity mismatch for repository {source.repo_id!r}")
            repo_token = hashlib.sha256(source.repo_id.encode()).hexdigest()[:20]
            ref = (f"{OWNED_REF_PREFIX}/{self._campaign_token}/"
                   f"{transaction_token}/{repo_token}")
            _run_git(repo, "check-ref-format", ref)
            planned.append(PreparedRef(source.repo_id, ref, commit, tree, object_format))
        if len(planned) != len(self._repos):
            extra = sorted(set(self._repos) - {item.repo_id for item in planned})
            if extra:
                raise CandidateTransactionError(
                    f"validated repository set contains unrelated entries: {extra}")
        for item in planned:
            self._planned[(item.repo_id, item.ref)] = item
        return tuple(planned)

    def prepare(self, item: PreparedRef) -> None:
        self._validate_planned(item)
        repo = self._repo(item)
        current = _run_git(repo, "rev-parse", "-q", "--verify", item.ref, check=False)
        if current:
            if current != item.commit:
                raise CandidateRecoveryRequired(
                    f"owned prepared ref changed: {item.repo_id}:{item.ref}")
            return
        try:
            _run_git(repo, "update-ref", item.ref, item.commit, "")
        except CandidateTransactionError:
            current = _run_git(repo, "rev-parse", "-q", "--verify", item.ref,
                               check=False)
            if current != item.commit:
                raise

    def verify(self, item: PreparedRef) -> None:
        self._validate_planned(item)
        repo = self._repo(item)
        current = _run_git(repo, "rev-parse", "-q", "--verify", item.ref, check=False)
        if current != item.commit:
            raise CandidateRecoveryRequired(
                f"restore exact prepared ref {item.repo_id}:{item.ref} -> {item.commit}")
        tree = _run_git(repo, "rev-parse", f"{current}^{{tree}}")
        if tree != item.tree:
            raise CandidateRecoveryRequired(
                f"prepared source tree changed for {item.repo_id}:{item.ref}")

    def _repo(self, item: PreparedRef) -> Path:
        try:
            return self._verified_repo(item.repo_id)
        except CandidateTransactionError as exc:
            raise CandidateRecoveryRequired(str(exc)) from exc

    def _verified_repo(self, repo_id: str) -> Path:
        if any(name in os.environ for name in _GIT_REDIRECT_ENV):
            raise CandidateRecoveryRequired("ambient Git repository redirection appeared")
        repo = self._repos.get(repo_id)
        if repo is None:
            raise CandidateTransactionError(
                f"supply validated repository {repo_id!r} to verify prepared ref")
        try:
            info = os.lstat(repo)
            git_info = os.lstat(self._git_dirs[repo_id])
            common_info = os.lstat(self._common_dirs[repo_id])
        except OSError as exc:
            raise CandidateRecoveryRequired(
                f"validated repository identity unavailable for {repo_id!r}: {exc}") from exc
        current = (info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode))
        git_current = (git_info.st_dev, git_info.st_ino, git_info.st_uid,
                       stat.S_IMODE(git_info.st_mode))
        common_current = (common_info.st_dev, common_info.st_ino, common_info.st_uid,
                          stat.S_IMODE(common_info.st_mode))
        branch = _run_git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False)
        if (current != self._repo_identities[repo_id]
                or git_current != self._git_dir_identities[repo_id]
                or common_current != self._common_dir_identities[repo_id]
                or repo in FROZEN_PRODUCTION_ROOTS
                or branch.startswith(_FROZEN_BRANCH_PREFIXES)
                or Path(_run_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo
                or Path(_run_git(repo, "rev-parse", "--absolute-git-dir")).resolve()
                != self._git_dirs[repo_id]
                or Path(_run_git(repo, "rev-parse", "--path-format=absolute",
                                 "--git-common-dir")).resolve()
                != self._common_dirs[repo_id]):
            raise CandidateRecoveryRequired(
                f"validated repository root/Git identity changed for {repo_id!r}")
        return repo

    def _validate_planned(self, item: PreparedRef) -> None:
        if not isinstance(item, PreparedRef):
            raise CandidateTransactionError("prepared ref must be a PreparedRef")
        expected = self._planned.get((item.repo_id, item.ref))
        if expected != item:
            raise CandidateTransactionError(
                "prepared ref was not derived by this backend for this campaign/request")


def _prepared_ref(value: Mapping[str, Any]) -> PreparedRef:
    required = {"repo_id", "ref", "commit", "tree", "object_format"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CandidateRecoveryRequired("malformed prepared-ref record")
    result = PreparedRef(*(_text(value[key], f"prepared ref {key}")
                           for key in ("repo_id", "ref", "commit", "tree",
                                       "object_format")))
    lengths = {"sha1": 40, "sha256": 64}
    if not result.ref.startswith(OWNED_REF_PREFIX + "/"):
        raise CandidateRecoveryRequired("prepared ref is outside owned namespace")
    if result.object_format not in lengths:
        raise CandidateRecoveryRequired("prepared ref has unsupported object format")
    size = lengths[result.object_format]
    for label, oid in (("commit", result.commit), ("tree", result.tree)):
        if len(oid) != size or any(char not in "0123456789abcdef" for char in oid):
            raise CandidateRecoveryRequired(f"prepared ref {label} has malformed object id")
    return result


class CandidateTransactions:
    """Replay and execute candidate transitions through one campaign controller."""

    def __init__(self, controller, *, git_backend: GitCandidateBackend | None = None,
                 verifiers: Mapping[str, tuple[Callable[..., bool],
                                               Callable[..., bool] | None]] | None = None,
                 fault_hook: Callable[[str], None] | None = None) -> None:
        self.controller = controller
        self.git_backend = git_backend
        self.verifiers = dict(verifiers or {})
        self.fault_hook = fault_hook or (lambda _point: None)

    def inspect(self) -> dict[str, Any]:
        return self.controller.candidate_transaction(self._inspect_locked)

    def _inspect_locked(self, context) -> dict[str, Any]:
        replay = self._replay(context)
        if replay.pending is not None:
            transaction_id = replay.pending["transaction_id"]
            raise CandidateRecoveryRequired(
                f"candidate transaction {transaction_id!r} has durable intent without completion")
        return self._snapshot(
            replay.state, context.completed_candidate_ids(),
            replay.historical_validation_receipt)

    def initialize(self, *, request_id: str, state: cm.CandidateState,
                   manifest: cm.CandidateManifest) -> dict[str, Any]:
        state, manifest = state.validated(), manifest.validated()
        payload = {"state": state.to_dict(), "manifest": manifest.to_dict()}
        return self._execute("init", request_id, payload, manifest.sources)

    def integrate(self, *, request_id: str, previous: cm.CandidateManifest,
                  candidate: cm.CandidateManifest,
                  threshold_signal: bool = False) -> dict[str, Any]:
        previous, candidate = previous.validated(), candidate.validated()
        payload = {"previous": previous.to_dict(), "candidate": candidate.to_dict(),
                   "threshold_signal": threshold_signal,
                   "integration_request_id": _text(
                       request_id, "candidate transaction request id")}
        return self._execute("integrate", request_id, payload, candidate.sources)

    def start_batch(self, *, request_id: str, batch: cm.ValidationBatch,
                    row_set: cm.RequiredRowSet, candidate: cm.CandidateManifest,
                    comparator: cm.CandidateManifest) -> dict[str, Any]:
        payload = {"batch": batch.validated().to_dict(),
                   "row_set": cm.RequiredRowSet.from_dict(row_set.to_dict()).to_dict(),
                   "candidate": candidate.validated().to_dict(),
                   "comparator": comparator.validated().to_dict()}
        return self._execute("start_batch", request_id, payload, ())

    def record_row(self, *, request_id: str, batch_id: str,
                   row_set: cm.RequiredRowSet,
                   row_state: cm.ValidationRowState) -> dict[str, Any]:
        payload = {"batch_id": _text(batch_id, "batch id"),
                   "row_set": cm.RequiredRowSet.from_dict(row_set.to_dict()).to_dict(),
                   "row_state": cm.ValidationRowState.from_dict(
                       row_state.to_dict()).to_dict()}
        return self._execute("record_row", request_id, payload, ())

    def complete_batch(self, *, request_id: str,
                       batch: cm.ValidationBatch) -> dict[str, Any]:
        payload = {"batch": batch.validated().to_dict()}
        return self._execute("complete_batch", request_id, payload, ())

    def advance_validated(self, *, request_id: str, verifier_id: str,
                          batch: cm.ValidationBatch, row_set: cm.RequiredRowSet,
                          candidate: cm.CandidateManifest,
                          comparator: cm.CandidateManifest,
                          loo_plans: Mapping[str, cm.LOOPlan] | None = None,
                          loo_results: Mapping[str, cm.LOOResult] | None = None) -> dict[str, Any]:
        payload = {
            "verifier_id": _text(verifier_id, "verifier id"),
            "batch": batch.validated().to_dict(),
            "row_set": cm.RequiredRowSet.from_dict(row_set.to_dict()).to_dict(),
            "candidate": candidate.validated().to_dict(),
            "comparator": comparator.validated().to_dict(),
            "loo_plans": {key: value.to_dict()
                          for key, value in sorted((loo_plans or {}).items())},
            "loo_results": {key: value.to_dict()
                            for key, value in sorted((loo_results or {}).items())},
        }
        return self._execute("advance_validated", request_id, payload, ())

    def _execute(self, operation: str, request_id: str,
                 operation_payload: Mapping[str, Any],
                 sources: Sequence[cm.SourceIdentity]) -> dict[str, Any]:
        request_id = _text(request_id, "candidate transaction request id")
        payload = copy.deepcopy(dict(operation_payload))

        def locked(context):
            replay = self._replay(context)
            expected = _state_digest(replay.state) if replay.state is not None else None
            refs: tuple[PreparedRef, ...] = ()
            if sources:
                if self.git_backend is None:
                    raise CandidateTransactionError(
                        "a validated Git backend is required to retain source identities")
                refs = self.git_backend.plan(request_id, sources)
            ref_rows = [item.to_dict() for item in refs]
            object_rows = self._planned_objects(operation, payload)
            digest = _intent_digest(operation, expected, payload, object_rows, ref_rows)
            completed_pair = context.completed_candidate(request_id)
            if completed_pair is not None:
                completed_intent = completed_pair["intent"]
                completed = completed_pair["completion"]
                original_digest = _intent_digest(
                    operation, completed_intent["data"]["expected_state_digest"],
                    payload, object_rows, ref_rows)
                if (completed["operation"] != operation
                        or completed["payload_digest"] != original_digest
                        or completed_intent["data"]["operation_payload"] != payload
                        or completed_intent["data"]["prepared_objects"] != object_rows
                        or completed_intent["data"]["prepared_refs"] != ref_rows):
                    raise CandidateTransactionError(
                        "candidate transaction id was reused with different semantics")
                self._verify_refs(refs)
                self._verify_documents(operation, payload)
                assert replay.state is not None
                self._write_projection(replay.state)
                return copy.deepcopy(dict(completed["data"]["result"]))
            if replay.pending is not None:
                pending = replay.pending
                if (pending["transaction_id"] != request_id
                        or pending["operation"] != operation
                        or pending["payload_digest"] != digest):
                    raise CandidateRecoveryRequired(
                        f"resolve pending candidate transaction "
                        f"{pending['transaction_id']!r} before new work")
                pending_refs = tuple(_prepared_ref(item)
                                     for item in pending["data"]["prepared_refs"])
                if replay.pending_prepared:
                    self._verify_refs(pending_refs)
                    self._verify_documents(operation, payload)
            else:
                next_state, result = self._apply(operation, replay.state, payload)
                intent_data = {"expected_state_digest": expected,
                               "operation_payload": payload,
                               "prepared_objects": object_rows,
                               "prepared_refs": ref_rows}
                self.fault_hook("before_intent")
                context.append(phase="INTENT", transaction_id=request_id,
                               operation=operation, payload_digest=digest,
                               data=intent_data)
                self.fault_hook("after_intent")
                self.fault_hook("before_objects")
                self._write_documents(operation, payload)
                self.fault_hook("after_objects")
                for item in refs:
                    assert self.git_backend is not None
                    self.fault_hook(f"before_ref:{item.repo_id}")
                    self.git_backend.prepare(item)
                    self.fault_hook(f"after_ref:{item.repo_id}")
                self._verify_refs(refs)
                context.append(
                    phase="PREPARED", transaction_id=request_id,
                    operation=operation, payload_digest=digest,
                    data={"prepared_objects": object_rows,
                          "prepared_refs": ref_rows})
                self.fault_hook("after_prepared")
                self.fault_hook("before_commit")
                result = dict(result)
                state_row = next_state.to_dict()
                context.append(
                    phase="COMMITTED", transaction_id=request_id,
                    operation=operation, payload_digest=digest,
                    data={"state": state_row, "state_digest": _digest(state_row),
                          "result": result,
                          "transition_receipt": self._transition_receipt(
                              operation, digest, expected, state_row, payload)})
                self._verify_refs(refs)
                self.fault_hook("after_commit")
                self._write_immutable("state", state_row)
                self.fault_hook("before_projection")
                self._write_projection(next_state)
                self.fault_hook("after_projection")
                return copy.deepcopy(result)

            # Before PREPARED, retry may safely finish its own exact named refs.
            pending_refs = tuple(_prepared_ref(item)
                                 for item in pending["data"]["prepared_refs"])
            next_state, result = self._apply(operation, replay.state, payload)
            if not replay.pending_prepared:
                self._write_documents(operation, payload)
                for item in pending_refs:
                    assert self.git_backend is not None
                    self.git_backend.prepare(item)
                self._verify_refs(pending_refs)
                context.append(
                    phase="PREPARED", transaction_id=request_id,
                    operation=operation, payload_digest=digest,
                    data={"prepared_objects": object_rows,
                          "prepared_refs": [item.to_dict() for item in pending_refs]})
                self.fault_hook("after_prepared")
            else:
                self._verify_refs(pending_refs)
                self._verify_documents(operation, payload)
            state_row = next_state.to_dict()
            context.append(
                phase="COMMITTED", transaction_id=request_id,
                operation=operation, payload_digest=digest,
                data={"state": state_row, "state_digest": _digest(state_row),
                      "result": dict(result),
                      "transition_receipt": self._transition_receipt(
                          operation, digest, expected, state_row, payload)})
            self._verify_refs(pending_refs)
            self._write_immutable("state", state_row)
            self.fault_hook("before_projection")
            self._write_projection(next_state)
            self.fault_hook("after_projection")
            return copy.deepcopy(dict(result))

        return self.controller.candidate_transaction(locked)

    def _verify_refs(self, refs: Sequence[PreparedRef]) -> None:
        if not refs:
            return
        if self.git_backend is None:
            raise CandidateRecoveryRequired("validated Git backend is required for recovery")
        for item in refs:
            self.git_backend.verify(item)

    def _apply(self, operation: str, state: cm.CandidateState | None,
               payload: Mapping[str, Any]
               ) -> tuple[cm.CandidateState, Mapping[str, Any]]:
        if operation == "init":
            if state is not None:
                raise cm.TransitionError("candidate state is already initialized")
            manifest = cm.CandidateManifest.from_dict(payload["manifest"])
            next_state = cm.CandidateState.from_dict(payload["state"])
            if (next_state.integration_tip != manifest.manifest_digest
                    or next_state.production_ref_digest != manifest.production_ref_digest):
                raise cm.TransitionError("initial state does not bind the initial manifest")
            if (manifest.keeps or next_state.validated_candidate is not None
                    or next_state.keeps_since_gate != 0
                    or next_state.threshold_generation != 0
                    or next_state.covered_threshold_generation != 0
                    or next_state.validation_debt or next_state.active_batches
                    or next_state.completed_batches or next_state.integrated_requests
                    or next_state.outstanding_summary_refs or next_state.stale_summary_refs):
                raise cm.TransitionError(
                    "initialization requires a new empty unvalidated candidate state")
            return next_state, self._result(next_state)
        if state is None:
            raise CandidateRecoveryRequired("candidate state is not initialized")
        if operation == "integrate":
            next_state, due = cm.integrate_candidate(
                state, cm.CandidateManifest.from_dict(payload["previous"]),
                cm.CandidateManifest.from_dict(payload["candidate"]),
                request_id=_text(payload["integration_request_id"],
                                 "integration request id"),
                threshold_signal=payload["threshold_signal"])
            return next_state, {**self._result(next_state), "gate_due": due}
        if operation == "start_batch":
            next_state = cm.start_batch(
                state, cm.ValidationBatch.from_dict(payload["batch"]),
                cm.RequiredRowSet.from_dict(payload["row_set"]),
                cm.CandidateManifest.from_dict(payload["candidate"]),
                cm.CandidateManifest.from_dict(payload["comparator"]))
            return next_state, self._result(next_state)
        if operation == "record_row":
            batch_id = _text(payload["batch_id"], "batch id")
            active = {item.batch_id: item for item in state.active_batches}
            if batch_id not in active:
                raise cm.TransitionError("row transaction names no active batch")
            updated = cm.record_row(
                active[batch_id], cm.RequiredRowSet.from_dict(payload["row_set"]),
                cm.ValidationRowState.from_dict(payload["row_state"]))
            active[batch_id] = updated
            next_state = replace(
                state, active_batches=tuple(sorted(active.values(),
                                                   key=lambda item: item.batch_id))).validated()
            return next_state, self._result(next_state)
        if operation == "complete_batch":
            next_state = cm.complete_batch(
                state, cm.ValidationBatch.from_dict(payload["batch"]))
            return next_state, self._result(next_state)
        if operation == "advance_validated":
            verifier_id = _text(payload["verifier_id"], "verifier id")
            registered = self.verifiers.get(verifier_id)
            if registered is None:
                raise cm.TrustedVerificationRequired(
                    f"trusted verifier {verifier_id!r} is not registered")
            verifier, loo_verifier = registered
            plans = {key: cm.LOOPlan.from_dict(value)
                     for key, value in payload["loo_plans"].items()}
            results = {key: cm.LOOResult.from_dict(value)
                       for key, value in payload["loo_results"].items()}
            next_state = cm.advance_validated(
                state, cm.ValidationBatch.from_dict(payload["batch"]),
                cm.RequiredRowSet.from_dict(payload["row_set"]),
                cm.CandidateManifest.from_dict(payload["candidate"]),
                cm.CandidateManifest.from_dict(payload["comparator"]),
                verifier=verifier, loo_plans=plans, loo_results=results,
                loo_verifier=loo_verifier)
            return next_state, self._result(next_state)
        raise CandidateTransactionError(f"unsupported candidate operation {operation!r}")

    def _transition_receipt(
            self, operation: str, payload_digest: str,
            expected_state_digest: str | None, state: Mapping[str, Any],
            payload: Mapping[str, Any]) -> dict[str, Any]:
        verifier_id = payload.get("verifier_id") if operation == "advance_validated" else None
        return {
            "schema": TRANSITION_RECEIPT_SCHEMA,
            "operation": operation,
            "payload_digest": payload_digest,
            "expected_state_digest": expected_state_digest,
            "resulting_state_digest": _digest(dict(state)),
            "decision": ("trusted_verifiers_accepted"
                         if operation == "advance_validated" else "pure_transition"),
            "verifier_id": verifier_id,
        }

    def _replay_apply(
            self, operation: str, state: cm.CandidateState | None,
            payload: Mapping[str, Any], receipt: Mapping[str, Any],
            payload_digest: str) -> tuple[cm.CandidateState, Mapping[str, Any]]:
        if state is None and operation != "init":
            raise CandidateRecoveryRequired("candidate state is not initialized")
        if operation != "advance_validated":
            next_state, result = self._apply(operation, state, payload)
        else:
            assert state is not None
            next_state = self._replay_validated_transition(state, payload)
            result = self._result(next_state)
        expected_receipt = self._transition_receipt(
            operation, payload_digest,
            _state_digest(state) if state is not None else None,
            next_state.to_dict(), payload)
        if dict(receipt) != expected_receipt:
            raise CandidateRecoveryRequired(
                "candidate durable transition receipt disagrees with replay")
        return next_state, result

    @staticmethod
    def _replay_validated_transition(
            state: cm.CandidateState, payload: Mapping[str, Any]) -> cm.CandidateState:
        """Reconstruct a recorded advance structurally, without granting fresh trust."""
        state = state.validated()
        batch = cm.ValidationBatch.from_dict(payload["batch"])
        row_set = cm.RequiredRowSet.from_dict(payload["row_set"])
        candidate = cm.CandidateManifest.from_dict(payload["candidate"])
        comparator = cm.CandidateManifest.from_dict(payload["comparator"])
        cm._validate_batch_obligations(batch, row_set, candidate, comparator)
        if candidate.production_ref_digest != state.production_ref_digest:
            raise cm.TransitionError(
                "batch candidate production reference mismatches state")
        if state.validated_candidate != batch.expected_validated_predecessor:
            raise cm.TransitionError("validated-candidate predecessor CAS mismatch")
        if (state.validated_candidate is not None
                and comparator.manifest_digest != state.validated_candidate):
            raise cm.TransitionError("comparator is not the expected validated predecessor")
        if not any(item == batch for item in state.completed_batches):
            raise cm.TransitionError("validated batch is not an immutable completed batch")
        states = {item.row_id: item for item in batch.rows}
        for row in row_set.rows:
            if not row.required:
                continue
            state_row = states.get(row.row_id)
            if (state_row is None or state_row.status != "passed"
                    or state_row.receipt is None
                    or not batch.matches_receipt(state_row.receipt)
                    or state_row.receipt.row_id != row.row_id
                    or state_row.receipt.use_disposition != "permitted"):
                raise cm.TransitionError(
                    f"required row {row.row_id!r} recorded receipt is inapplicable")
        plans = {key: cm.LOOPlan.from_dict(value)
                 for key, value in payload["loo_plans"].items()}
        results = {key: cm.LOOResult.from_dict(value)
                   for key, value in payload["loo_results"].items()}
        required_loo = set(batch.required_loo_keep_ids)
        if (required_loo != {item.keep_id for item in candidate.keeps}
                or required_loo - set(results) or required_loo - set(plans)):
            raise cm.TransitionError("recorded LOO obligations are incomplete")
        for keep_id in batch.required_loo_keep_ids:
            result = results[keep_id]
            plan = plans[keep_id]
            if (plan.status != "planned" or plan.derived_manifest_digest is None
                    or plan.candidate_manifest_digest != candidate.manifest_digest
                    or plan.keep_id != keep_id
                    or plan.binary_execution_digest != candidate.build_set_digest
                    or result.plan_digest != plan.plan_digest
                    or result.candidate_manifest_digest != candidate.manifest_digest
                    or result.keep_id != keep_id
                    or result.derived_manifest_digest != plan.derived_manifest_digest
                    or result.row_set_digest != row_set.row_set_digest
                    or result.disposition not in {
                        "neutral", "supports_keep", "supports_removal"}):
                raise cm.TransitionError(
                    f"recorded LOO result for {keep_id!r} is structurally ineligible")
        debt = set(state.validation_debt)
        debt.discard(batch.batch_id)
        return replace(
            state, validated_candidate=candidate.manifest_digest,
            validation_debt=tuple(sorted(debt)),
            stale_summary_refs=tuple(sorted(set(state.stale_summary_refs)
                                             | set(state.outstanding_summary_refs))),
            outstanding_summary_refs=()).validated()

    @staticmethod
    def _result(state: cm.CandidateState) -> dict[str, Any]:
        return {"state_digest": _state_digest(state),
                "integration_tip": state.integration_tip,
                "validated_candidate": state.validated_candidate,
                "gate_due": state.gate_due,
                "validation_debt": list(state.validation_debt)}

    @dataclass(frozen=True)
    class _Replay:
        state: cm.CandidateState | None
        pending: Mapping[str, Any] | None
        pending_prepared: bool
        historical_validation_receipt: Mapping[str, Any] | None
        position: int
        last_seq: int

    def _replay(self, context) -> "CandidateTransactions._Replay":
        entries = context.entries
        offset = context.entry_offset
        cached = context.projection_cache()
        if cached is not None and not isinstance(cached, self._Replay):
            raise CandidateRecoveryRequired("candidate in-memory projection cache is malformed")
        if cached is not None and cached.position == offset:
            state = cached.state
            pending = cached.pending
            pending_prepared = cached.pending_prepared
            validation_receipt = cached.historical_validation_receipt
        else:
            if offset != 0:
                raise CandidateRecoveryRequired(
                    "candidate projection cache/event position disagree")
            state = None
            pending = None
            pending_prepared = False
            validation_receipt = None
        for entry in entries:
            if entry.kind != journal_module.KIND_CANDIDATE_TRANSACTION:
                continue
            row = entry.payload
            violations = journal_module._validate_native_payload(entry.kind, row)
            if violations:
                raise CandidateRecoveryRequired(
                    "invalid candidate transaction event: " + "; ".join(violations))
            if (row["campaign_id"] != context.campaign_id
                    or row["config_generation"] != context.config_generation
                    or row["config_digest"] != context.config_digest
                    or row["supervisor_incarnation"] > context.supervisor_incarnation):
                raise CandidateRecoveryRequired(
                    "candidate transaction event has foreign campaign/controller binding")
            transaction_id = row["transaction_id"]
            if row["phase"] == "INTENT":
                if pending is not None:
                    raise CandidateRecoveryRequired(
                        "candidate transaction history has overlapping/duplicate intent")
                expected = _state_digest(state) if state is not None else None
                if row["data"]["expected_state_digest"] != expected:
                    raise CandidateRecoveryRequired(
                        "candidate intent expected-state CAS does not match replay")
                pending = row
                pending_prepared = False
                continue
            if row["phase"] == "PREPARED":
                if (pending is None or pending_prepared
                        or transaction_id != pending["transaction_id"]
                        or row["operation"] != pending["operation"]
                        or row["payload_digest"] != pending["payload_digest"]
                        or row["data"]["prepared_objects"]
                        != pending["data"]["prepared_objects"]
                        or row["data"]["prepared_refs"]
                        != pending["data"]["prepared_refs"]):
                    raise CandidateRecoveryRequired(
                        "candidate preparation has no exact preceding intent")
                pending_prepared = True
                continue
            if pending is None or not pending_prepared \
                    or transaction_id != pending["transaction_id"] \
                    or row["operation"] != pending["operation"] \
                    or row["payload_digest"] != pending["payload_digest"]:
                raise CandidateRecoveryRequired(
                    "candidate completion has no exact preceding intent")
            planned_objects = self._planned_objects(
                row["operation"], pending["data"]["operation_payload"])
            if pending["data"]["prepared_objects"] != planned_objects:
                raise CandidateRecoveryRequired(
                    "candidate intent prepared-object hashes disagree with payload")
            try:
                expected_state, expected_result = self._replay_apply(
                    row["operation"], state, pending["data"]["operation_payload"],
                    row["data"]["transition_receipt"], row["payload_digest"])
                committed_state = cm.CandidateState.from_dict(row["data"]["state"])
            except (KeyError, TypeError, ValueError, cm.CandidateError,
                    CandidateTransactionError) as exc:
                raise CandidateRecoveryRequired(
                    f"candidate transaction replay payload is invalid: {exc}") from exc
            if (committed_state != expected_state
                    or dict(row["data"]["result"]) != dict(expected_result)):
                raise CandidateRecoveryRequired(
                    "candidate completion disagrees with deterministic replay")
            state = committed_state
            if row["operation"] == "advance_validated":
                validation_receipt = row["data"]["transition_receipt"]
            pending = None
            pending_prepared = False
        result = self._Replay(
            state, pending, pending_prepared, validation_receipt,
            offset + len(entries),
            entries[-1].seq if entries else (cached.last_seq if cached else 0))
        context._update_projection_cache_trusted(
            result, authority=campaign_control_module._CANDIDATE_REPLAYER_TOKEN)
        return result

    def _object_root(self) -> Path:
        return Path(self.controller.store) / OBJECT_DIR

    def _write_immutable(self, kind: str, value: Mapping[str, Any]) -> str:
        if kind not in {"manifest", "row-set", "batch", "row-state", "loo-plan",
                        "loo-result", "state"}:
            raise CandidateTransactionError(f"unknown candidate object kind {kind!r}")
        try:
            store = mc.ArtifactStore(self._object_root())
            try:
                store.write(f"candidate:{kind}", dict(value))
            finally:
                store.close()
        except mc.CaptureError as exc:
            raise CandidateRecoveryRequired(
                f"candidate artifact publication failed for {kind}: {exc}") from exc
        return _digest(dict(value))

    def _verify_immutable(self, kind: str, value: Mapping[str, Any]) -> str:
        root = self._object_root()
        if not os.path.lexists(root):
            raise CandidateRecoveryRequired(
                f"restore immutable candidate object {kind}:{_digest(dict(value))}")
        try:
            store = mc.ArtifactStore(root)
            try:
                store.verify(f"candidate:{kind}", dict(value))
            finally:
                store.close()
        except mc.CaptureError as exc:
            raise CandidateRecoveryRequired(
                f"restore immutable candidate object {kind}:{_digest(dict(value))}: "
                f"{exc}") from exc
        return _digest(dict(value))

    def _documents(self, operation: str,
                   payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
        keys = {"init": (("state", "state"), ("manifest", "manifest")),
                "integrate": (("manifest", "previous"), ("manifest", "candidate")),
                "start_batch": (("batch", "batch"), ("row-set", "row_set"),
                                ("manifest", "candidate"), ("manifest", "comparator")),
                "record_row": (("row-set", "row_set"), ("row-state", "row_state")),
                "complete_batch": (("batch", "batch"),),
                "advance_validated": (("batch", "batch"), ("row-set", "row_set"),
                                      ("manifest", "candidate"),
                                      ("manifest", "comparator"))}[operation]
        documents = [(kind, payload[key]) for kind, key in keys]
        if operation == "advance_validated":
            documents.extend(("loo-plan", value) for value in payload["loo_plans"].values())
            documents.extend(("loo-result", value)
                             for value in payload["loo_results"].values())
        return documents

    def _write_documents(self, operation: str, payload: Mapping[str, Any]) -> None:
        for kind, value in self._documents(operation, payload):
            self._write_immutable(kind, value)

    def _planned_objects(self, operation: str,
                         payload: Mapping[str, Any]) -> list[dict[str, str]]:
        return [{"kind": kind, "digest": _digest(dict(value))}
                for kind, value in self._documents(operation, payload)]

    def _verify_documents(self, operation: str, payload: Mapping[str, Any]) -> None:
        for kind, value in self._documents(operation, payload):
            self._verify_immutable(kind, value)

    def _write_projection(self, state: cm.CandidateState) -> None:
        state_row = state.to_dict()
        state_digest = self._write_immutable("state", state_row)
        status.write_json(
            self.controller.store, POINTER_FILE,
            {"schema": POINTER_SCHEMA, "state_digest": state_digest,
             "integration_tip": state.integration_tip,
             "validated_candidate": state.validated_candidate,
             "gate_due": state.gate_due,
             "validation_debt": list(state.validation_debt)},
            prefix=".candidate-state-")

    @staticmethod
    def _snapshot(state: cm.CandidateState | None,
                  completed: Sequence[str],
                  validation_receipt: Mapping[str, Any] | None) -> dict[str, Any]:
        return {"schema": POINTER_SCHEMA,
                "initialized": state is not None,
                "state": state.to_dict() if state is not None else None,
                "state_digest": _state_digest(state) if state is not None else None,
                "completed_transactions": list(completed),
                "historical_validation_receipt": copy.deepcopy(validation_receipt),
                "current_evidence_eligibility": (
                    "requires_live_registered_verification"
                    if state is not None and state.validated_candidate is not None
                    else "not_applicable")}


__all__ = ["CandidateRecoveryRequired", "CandidateTransactionError",
           "CandidateTransactions", "GitCandidateBackend", "OBJECT_DIR",
           "OWNED_REF_PREFIX", "POINTER_FILE", "POINTER_SCHEMA", "PreparedRef",
           "TRANSITION_RECEIPT_SCHEMA"]
