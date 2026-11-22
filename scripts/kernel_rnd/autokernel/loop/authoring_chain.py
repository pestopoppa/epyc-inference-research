"""Durable advice -> authored patch -> source commit -> build identity chain (AKU-06d4).

One append-only, hash-chained JSONL journal owned by the contained authoring owner.
Restart replays it: every reserved advice stays finished (an attempt open at the
crash is abandoned, never re-granted), and each stage must cite the exact digest of
the stage before it. A build stage accepts only a typed ``worktree.BuildIdentity``
that succeeded and whose source snapshot is the committed tree; a forwarded runner
return value is not build identity. Planner prerequisites are refreshed only from
replayed BUILD_VERIFIED records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from types import MappingProxyType
from typing import Any, Mapping

from ..execution import worktree
from . import source_build_preparation as bridge

SCHEMA = "epyc.autokernel.authoring_chain.v1"
BUILD_PREREQUISITE = "candidate_build_verified_required"
_STAGES = ("RESERVED", "AUTHORED", "COMMITTED", "BUILD_VERIFIED")
_FIELDS = {
    "RESERVED": {"attempt_id", "advice_digest", "target_revision_digest"},
    "AUTHORED": {"attempt_id", "advice_digest", "patch_bundle_sha256",
                 "preparation_digest", "stdout_sha256"},
    "FAILED": {"attempt_id", "advice_digest", "reason"},
    "COMMITTED": {"advice_digest", "preparation_digest", "source_commit",
                  "source_tree_digest", "patch_bundle_sha256"},
    "BUILD_VERIFIED": {"advice_digest", "source_commit", "candidate_id",
                       "build_identity_sha256", "output_binary_sha256"},
}


class ChainRefused(RuntimeError):
    pass


def _self_digest(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canon({k: v for k, v in row.items() if k != "sha256"})).hexdigest()


def _canon(row: Mapping[str, Any]) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


@dataclass
class AdviceChain:
    advice_digest: str
    target_revision_digest: str
    attempt_id: str
    stage: str = "RESERVED"
    records: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChainState:
    chains: Mapping[str, AdviceChain]
    head: str
    count: int

    @property
    def finished_advice(self) -> frozenset[str]:
        return frozenset(self.chains)

    def verified_builds(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(dict(c.records["BUILD_VERIFIED"]))
                     for c in self.chains.values() if c.stage == "BUILD_VERIFIED")


def _apply(chains: dict, row: Mapping[str, Any]) -> None:
    kind = row.get("kind")
    if kind not in _FIELDS or set(row) != _FIELDS[kind] | {"schema", "seq", "kind", "prev",
                                                         "sha256"}:
        raise ChainRefused(f"chain record fields/kind differ: {kind!r}")
    digest = row["advice_digest"]
    chain = chains.get(digest)
    if kind == "RESERVED":
        if chain is not None:
            raise ChainRefused("advice reserved twice; a finished advice is never re-granted")
        chains[digest] = AdviceChain(digest, row["target_revision_digest"], row["attempt_id"],
                                     records={"RESERVED": dict(row)})
        return
    if chain is None:
        raise ChainRefused(f"{kind} cites advice that was never reserved")
    if kind == "FAILED":
        if chain.stage != "RESERVED" or row["attempt_id"] != chain.attempt_id:
            raise ChainRefused("FAILED does not close the open attempt")
        chain.stage = "FAILED"
    else:
        previous = _STAGES[_STAGES.index(kind) - 1]
        if chain.stage != previous:
            raise ChainRefused(f"{kind} out of order after {chain.stage}")
        prior = chain.records[previous]
        if kind == "AUTHORED" and row["attempt_id"] != chain.attempt_id:
            raise ChainRefused("AUTHORED attempt differs from the reservation")
        if kind == "COMMITTED" and (
                row["preparation_digest"] != prior["preparation_digest"]
                or row["patch_bundle_sha256"] != prior["patch_bundle_sha256"]):
            raise ChainRefused("COMMITTED source differs from the authored patch")
        if kind == "BUILD_VERIFIED" and row["source_commit"] != prior["source_commit"]:
            raise ChainRefused("BUILD_VERIFIED commit differs from the committed source")
        chain.stage = kind
    chain.records[kind] = dict(row)


def replay(path: str) -> ChainState:
    """Rebuild state from disk, verifying the hash chain and every stage binding."""
    chains: dict[str, AdviceChain] = {}
    head, count = "0" * 64, 0
    if os.path.exists(path):
        with open(path, "rb") as handle:
            for line in handle.read().splitlines():
                row = json.loads(line)
                if (not isinstance(row, dict) or row.get("schema") != SCHEMA
                        or row.get("seq") != count or row.get("prev") != head
                        or row.get("sha256") != _self_digest(row)):
                    raise ChainRefused(f"chain journal record {count} is not hash-chained")
                _apply(chains, row)
                head, count = hashlib.sha256(_canon(row)).hexdigest(), count + 1
    return ChainState(MappingProxyType(chains), head, count)


class AuthoringChainJournal:
    """Single writer; each append re-validates against the replayed state, then fsyncs."""

    def __init__(self, path: str):
        self.path = path
        self.state = replay(path)

    def _append(self, kind: str, **values: Any) -> Mapping[str, Any]:
        row = {"schema": SCHEMA, "seq": self.state.count, "kind": kind,
               "prev": self.state.head, **values}
        row["sha256"] = _self_digest(row)
        chains = dict(self.state.chains)
        if row.get("advice_digest") in chains:
            old = chains[row["advice_digest"]]
            chains[row["advice_digest"]] = AdviceChain(
                old.advice_digest, old.target_revision_digest, old.attempt_id,
                old.stage, dict(old.records))
        _apply(chains, row)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, _canon(row) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        self.state = ChainState(MappingProxyType(chains),
                                hashlib.sha256(_canon(row)).hexdigest(), self.state.count + 1)
        return row

    def reserved(self, attempt_id, advice_digest, target_revision_digest):
        return self._append("RESERVED", attempt_id=attempt_id, advice_digest=advice_digest,
                            target_revision_digest=target_revision_digest)

    def failed(self, attempt_id, advice_digest, reason):
        return self._append("FAILED", attempt_id=attempt_id, advice_digest=advice_digest,
                            reason=str(reason)[:500])

    def authored(self, authored) -> Mapping[str, Any]:
        return self._append("AUTHORED", attempt_id=authored.attempt_id,
                            advice_digest=authored.advice_digest,
                            patch_bundle_sha256=authored.manifest.patch_bundle_sha256,
                            preparation_digest=authored.bound.digest,
                            stdout_sha256=authored.stdout_sha256)

    def committed(self, advice_digest: str,
                  capability: bridge.MaterializedSourceCapability) -> Mapping[str, Any]:
        if not isinstance(capability, bridge.MaterializedSourceCapability):
            raise ChainRefused("source commit needs the materialized source capability")
        bridge._validate_materialized_source(capability)
        return self._append("COMMITTED", advice_digest=advice_digest,
                            preparation_digest=capability["preparation_digest"],
                            source_commit=capability["source_commit"],
                            source_tree_digest=capability["source_tree_digest"],
                            patch_bundle_sha256=capability["patch_bundle_sha256"])

    def build_verified(self, advice_digest: str, identity: Any) -> Mapping[str, Any]:
        if not isinstance(identity, worktree.BuildIdentity):
            raise ChainRefused("a runner return is not verified build identity")
        chain = self.state.chains.get(advice_digest)
        committed = chain.records.get("COMMITTED") if chain else None
        if committed is None:
            raise ChainRefused("build cites advice with no committed source")
        if (identity.exit_code != 0 or not identity.log_facts.succeeded_by_log
                or identity.snapshot_sha256 != committed["source_tree_digest"]):
            raise ChainRefused("build identity did not succeed on the committed source tree")
        return self._append(
            "BUILD_VERIFIED", advice_digest=advice_digest,
            source_commit=committed["source_commit"], candidate_id=identity.candidate_id,
            build_identity_sha256=hashlib.sha256(_canon(identity.to_dict())).hexdigest(),
            output_binary_sha256=identity.output_binary_sha256)


def refresh_prerequisites(runtime_prerequisites: Mapping[str, tuple[str, ...]],
                          state: ChainState) -> Mapping[str, tuple[str, ...]]:
    """Clear the build prerequisite only for targets with a replayed verified build."""
    verified = {c.target_revision_digest for c in state.chains.values()
                if c.stage == "BUILD_VERIFIED"}
    out = {}
    for target, reasons in runtime_prerequisites.items():
        kept = tuple(r for r in reasons if not (r == BUILD_PREREQUISITE and target in verified))
        if kept:
            out[target] = kept
    return MappingProxyType(out)
