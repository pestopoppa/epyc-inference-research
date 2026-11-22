"""Contained patch authoring from accepted source advice (AKU-06d3).

Advice is prose; it never becomes patch bytes here. The owner issues a FRESH
attempt reservation per accepted advice, freezes the immutable source context and
assigns the scope (files/symbols, change class, diff bound) itself. The contained
worker returns only bounded stdout authenticated with that attempt's private key;
the owner constructs the native ``SourcePatchManifest`` from its own scope plus
those bytes and binds it to the advice through ``bind_source_preparation``.
A finished advice (accepted or failed) cannot be reserved again. Nothing here
grants execution, mutates source or builds.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import secrets
from types import MappingProxyType
from typing import Any, Mapping

from .. import source_candidate as source
from ..controller.discovery_controller import AuthoringAssignment
from . import source_build_preparation as bridge, unified_driver as driver

OUTPUT_SCHEMA = "epyc.autokernel.contained_authoring_output.v1"
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024


class AuthoringRefused(RuntimeError):
    pass


def advice_digest(selected_actor_work, preparation_result) -> str:
    """Identity of one accepted advice: the selected transition plus its exact result."""
    if not isinstance(selected_actor_work, driver.SelectedActorWork):
        raise AuthoringRefused("advice identity needs typed selected actor work")
    row = {"selected_work": selected_actor_work.to_dict(),
           "request_digest": preparation_result.request_digest,
           "actor_profile_digest": preparation_result.actor_profile_digest,
           "proposed_output": driver._thaw(preparation_result.proposed_output)}
    return hashlib.sha256(bridge._bytes(row)).hexdigest()


def _scope(proposal: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], int]:
    change = proposal.get("change") if isinstance(proposal, Mapping) else None
    entries = change.get("files_and_symbols") if isinstance(change, Mapping) else None
    bound = change.get("estimated_diff_size") if isinstance(change, Mapping) else None
    if not isinstance(entries, list) or not entries or type(bound) is not int or bound < 1:
        raise AuthoringRefused("owner scope needs files_and_symbols and a positive diff bound")
    symbols: dict[str, set[str]] = {}
    for entry in entries:
        if not isinstance(entry, str) or ":" not in entry:
            raise AuthoringRefused(f"scope entry {entry!r} is not path:symbol")
        path, symbol = entry.rsplit(":", 1)
        symbols.setdefault(path, set()).add(symbol)
    files = tuple(sorted(symbols))
    return files, {p: tuple(sorted(symbols[p])) for p in files}, bound


@dataclass(frozen=True)
class AuthoringReservation:
    """Immutable attempt: source context and scope are owner-assigned, not worker input."""

    attempt_id: str
    advice_digest: str
    source_context: Mapping[str, Any]
    scope: Mapping[str, Any]
    max_output_bytes: int
    worker_key: bytes = field(repr=False)

    def worker_brief(self) -> dict[str, Any]:
        """What the contained worker sees: context, scope, advice; never authority."""
        return {"attempt_id": self.attempt_id, "advice_digest": self.advice_digest,
                "source_context": dict(self.source_context), "scope": dict(self.scope),
                "max_output_bytes": self.max_output_bytes, "execution_authorized": False}


def sign_output(worker_key: bytes, attempt_id: str, stdout: bytes) -> str:
    """The contained worker's authentication over its exact stdout bytes."""
    return hmac.new(worker_key, attempt_id.encode() + b"\0" + stdout, hashlib.sha256).hexdigest()


def encode_output(attempt_id: str, patch_bytes: bytes) -> bytes:
    return bridge._bytes({"schema": OUTPUT_SCHEMA, "attempt_id": attempt_id,
                          "patch_base64": base64.b64encode(patch_bytes).decode("ascii")})


@dataclass(frozen=True)
class AuthoredPatch:
    attempt_id: str
    advice_digest: str
    manifest: source.SourcePatchManifest
    bound: bridge.BoundSourcePreparation
    stdout_sha256: str


class ContainedAuthoringOwner:
    """Single owner of authoring attempts for one campaign driver."""

    def __init__(self, *, max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
                 finished_advice: frozenset[str] = frozenset(), journal=None):
        if type(max_output_bytes) is not int or max_output_bytes < 1:
            raise AuthoringRefused("output bound must be a positive integer")
        self.max_output_bytes = max_output_bytes
        # A durable AuthoringChainJournal (AKU-06d4) replays finished advice on restart.
        self.journal = journal
        self._finished: set[str] = set(finished_advice) | (
            set(journal.state.finished_advice) if journal is not None else set())
        self._open: dict[str, tuple[AuthoringReservation, dict[str, Any]]] = {}

    @property
    def finished_advice(self) -> frozenset[str]:
        return frozenset(self._finished)

    def reserve(self, *, selected_actor_work, preparation_result, resolved_campaign,
                assignment: AuthoringAssignment,
                legacy_proposal: Mapping[str, Any]) -> AuthoringReservation:
        if not isinstance(assignment, AuthoringAssignment):
            raise AuthoringRefused("assignment must be controller-owned typed input")
        try:
            row = bridge._common(selected_actor_work, preparation_result, resolved_campaign,
                                 "source")
        except bridge.PreparationBindingRefused as exc:
            raise AuthoringRefused(f"advice is not accepted source advice: {exc}") from exc
        digest = advice_digest(selected_actor_work, preparation_result)
        if digest in self._finished:
            raise AuthoringRefused("finished advice cannot be reserved again")
        if any(r.advice_digest == digest for r, _ in self._open.values()):
            raise AuthoringRefused("advice already has an open authoring attempt")
        proposal = json.loads(bridge._bytes(legacy_proposal))
        if proposal.get("proposal_id") != assignment.proposal_id:
            raise AuthoringRefused("legacy proposal differs from the assignment")
        files, symbols, bound = _scope(proposal)
        context = {"campaign_id": assignment.campaign_id,
                   "proposal_id": assignment.proposal_id,
                   "candidate_id": assignment.candidate_id,
                   "source_tree": source.SOURCE_TREE,
                   "production_base_commit": assignment.production_base_commit,
                   "instrument_commit": assignment.instrument_commit,
                   "advice": row["advice"]}
        scope = {"change_class": proposal.get("change_class"),
                 "mechanism_id": row["advice"]["mechanism"],
                 "declared_files": list(files),
                 "declared_symbols": {p: list(v) for p, v in symbols.items()},
                 "max_changed_lines": bound}
        reservation = AuthoringReservation(
            "aka-" + secrets.token_hex(16), digest, MappingProxyType(context),
            MappingProxyType(scope), self.max_output_bytes,
            secrets.token_bytes(32))
        if self.journal is not None:
            self.journal.reserved(reservation.attempt_id, digest,
                                  selected_actor_work.actor_request.proposal.target_revision_digest)
        self._open[reservation.attempt_id] = (reservation, {
            "selected_actor_work": selected_actor_work,
            "preparation_result": preparation_result, "resolved_campaign": resolved_campaign,
            "assignment": assignment, "legacy_proposal": proposal})
        return reservation

    def fail(self, attempt_id: str) -> None:
        """Close an attempt without output; its advice is finished, never re-granted."""
        reservation, _ = self._take(attempt_id)
        self._finished.add(reservation.advice_digest)
        if self.journal is not None:
            self.journal.failed(attempt_id, reservation.advice_digest, "worker_failed")

    def _take(self, attempt_id):
        entry = self._open.pop(attempt_id, None)
        if entry is None:
            raise AuthoringRefused("unknown or already finished authoring attempt")
        return entry

    def accept(self, attempt_id: str, stdout: bytes, mac: str) -> AuthoredPatch:
        """Authenticate bounded stdout; construct and bind the native manifest.

        Any refusal finishes the attempt and its advice (no retry on the same grant).
        """
        reservation, inputs = self._take(attempt_id)
        self._finished.add(reservation.advice_digest)
        try:
            authored = self._authored(reservation, inputs, attempt_id, stdout, mac)
        except AuthoringRefused as exc:
            if self.journal is not None:
                self.journal.failed(attempt_id, reservation.advice_digest, exc)
            raise
        if self.journal is not None:
            self.journal.authored(authored)
        return authored

    def _authored(self, reservation, inputs, attempt_id, stdout, mac) -> AuthoredPatch:
        if not isinstance(stdout, bytes) or len(stdout) > reservation.max_output_bytes:
            raise AuthoringRefused("worker output is not bounded bytes")
        if not isinstance(mac, str) or not hmac.compare_digest(
                sign_output(reservation.worker_key, attempt_id, stdout), mac):
            raise AuthoringRefused("worker output is not authenticated for this attempt")
        try:
            row = json.loads(stdout)
            if (not isinstance(row, dict) or set(row) != {"schema", "attempt_id", "patch_base64"}
                    or row["schema"] != OUTPUT_SCHEMA or row["attempt_id"] != attempt_id):
                raise ValueError("envelope fields/attempt differ")
            patch = base64.b64decode(row["patch_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise AuthoringRefused(f"worker output envelope is malformed: {exc}") from exc
        ctx, scope = reservation.source_context, reservation.scope
        try:
            manifest = source.SourcePatchManifest(
                campaign_id=ctx["campaign_id"], proposal_id=ctx["proposal_id"],
                candidate_id=ctx["candidate_id"], source_tree=ctx["source_tree"],
                production_base_commit=ctx["production_base_commit"],
                instrument_commit=ctx["instrument_commit"],
                change_class=scope["change_class"],
                declared_files=tuple(scope["declared_files"]),
                declared_symbols={p: tuple(v) for p, v in scope["declared_symbols"].items()},
                mechanism_id=scope["mechanism_id"],
                patch_sha256=hashlib.sha256(patch).hexdigest(), patch_bytes=patch)
            bound = bridge.bind_source_preparation(authored_manifest=manifest, **inputs)
        except (source.SourceCandidateError, bridge.PreparationBindingRefused) as exc:
            raise AuthoringRefused(f"authored patch refused by native policy: {exc}") from exc
        return AuthoredPatch(attempt_id, reservation.advice_digest, manifest, bound,
                             hashlib.sha256(stdout).hexdigest())
