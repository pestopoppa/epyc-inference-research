"""Schema-valid durable candidate records for the lean campaign driver."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

from . import schemas
from .execution import worktree
from .source_candidate import AppliedSourceCandidate, parameter_patch_bundle_sha256

__all__ = ["CandidateRecordError", "build_candidate_record", "append_candidate_idempotent"]


class CandidateRecordError(RuntimeError):
    pass


def _sorted_strings(values: Sequence[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _directory_gb(path: str) -> float:
    total = 0
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    continue
    return total / 1_000_000_000.0


def build_candidate_record(*, proposal: Mapping[str, Any], candidate_id: str,
                           campaign_id: str, production_base_commit: str,
                           instrument_commit: str, source_commit: str,
                           actor: worktree.Worktree,
                           identity: Optional[worktree.BuildIdentity],
                           build_result: Optional[worktree.BuildResult],
                           source_application: Optional[AppliedSourceCandidate],
                           status: str, evaluator_id: str,
                           evaluator_bundle_sha256: str,
                           evaluator_runtime_source_label_ref: str,
                           resource_claim_receipt: str,
                           host_receipt: str,
                           evaluation_event_ids: Sequence[str] = (),
                           derived_surface_tokens: Sequence[str] = (),
                           traced_surface_tokens: Sequence[str] = (),
                           dispatch_predicates: Sequence[str] = (),
                           protocol_ids: Sequence[str] = ("P-AK-SEARCH-1/v1",),
                           same_seed_repeat_runs: int = 0,
                           banking_verdict: Optional[Mapping[str, Any]] = None,
                           derived_verdicts: Optional[Mapping[str, Any]] = None,
                           created_at: str) -> dict:
    """Build a candidate record solely from measured/validated inputs.

    A failed build carries no ``artifacts`` block; candidate.v1 permits that
    only for ``status=build_failed``.  No placeholder binary or linkage digest
    is invented to make the shape pass.
    """
    if status not in schemas.CANDIDATE_STATUSES:
        raise CandidateRecordError(f"unknown candidate status {status!r}")
    if not isinstance(actor, worktree.Worktree):
        raise TypeError("actor must be a Worktree")
    if actor.head_commit() != source_commit or not actor.is_clean():
        raise CandidateRecordError(
            "candidate actor must be a clean detached snapshot at source_commit")
    change_class = proposal["change_class"]
    parameter = change_class == "parameter"
    if source_application is None and not parameter:
        raise CandidateRecordError("source candidate record requires its applied source artifact")
    if source_application is not None and parameter:
        raise CandidateRecordError("parameter candidate cannot carry a source application")

    if source_application is not None:
        patch_sha = source_application.manifest.patch_bundle_sha256
        actual_files = list(source_application.actual_files)
        hunk_ids = list(source_application.actual_hunk_ids)
        symbols = list(source_application.actual_symbols)
        mechanism_id = source_application.manifest.mechanism_id
        feature_assignments: dict[str, Any] = {}
    else:
        patch_sha = parameter_patch_bundle_sha256(
            proposal=proposal, candidate_id=candidate_id)
        actual_files, hunk_ids = [], []
        symbols = ["<parameter>:GGML_IQK"]
        mechanism_id = "autokernel.parameter.ggml_iqk/v1"
        surface = proposal["change"]["parameter_surface"]["candidate"]
        feature_assignments = {"GGML_IQK": surface["ggml_iqk"]}

    if identity is not None:
        blocks = identity.to_candidate_records()
        identity_worktree = blocks["worktree"]
        if identity.source_root != actor.path.path \
                or identity_worktree.get("path") != actor.path.path \
                or identity_worktree.get("source_commit") != source_commit:
            raise CandidateRecordError(
                "candidate actor/worktree must be the detached snapshot that actually built")
        if identity_worktree.get("branch") is None:
            identity_worktree["branch"] = "ak/detached-candidate"
        blocks["source_snapshot"]["patch_bundle_sha256"] = patch_sha
        if "linkage_sha256" not in blocks["artifacts"]:
            raise CandidateRecordError(
                "BuildIdentity has no measured linkage_sha256; candidate artifacts "
                "cannot be recorded without the tree-local linkage proof")
        build_block = blocks["build"]
        artifacts = blocks["artifacts"]
        snapshot_sha = blocks["source_snapshot"]["snapshot_sha256"]
    elif status == "build_failed" and build_result is not None:
        plan = build_result.plan
        if plan.source_root.path != actor.path.path:
            raise CandidateRecordError(
                "failed-build actor must be the detached source snapshot that was built")
        compiler = next((f"{lang} {version}" for lang, version
                         in build_result.facts.compiler_ids if lang in ("CXX", "HIP", "C")),
                        "not_observed: build failed before compiler identification")
        build_block = {
            "toolchain": "cmake clean snapshot build",
            "compiler": compiler,
            "command": " ".join(plan.build_argv()),
            "build_dir": plan.build_dir.path,
            "log_path": build_result.log_path,
            "log_sha256": build_result.log_sha256,
        }
        artifacts = None
        snapshot_sha = actor.snapshot_digest().sha256
    else:
        raise CandidateRecordError(
            "a non-build-failed candidate requires a real BuildIdentity")

    schemas.require.commit(instrument_commit, "instrument_commit", error=CandidateRecordError)
    if actor.repo.commit_parents(instrument_commit) != (production_base_commit,):
        raise CandidateRecordError(
            "instrument commit is not the ratified single-child of the production base")
    if not actor.is_ancestor(production_base_commit, source_commit) \
            or not actor.is_ancestor(instrument_commit, source_commit):
        raise CandidateRecordError(
            "candidate source commit does not descend from production and instrument bases")

    dispatch_flags = sorted(feature_assignments)
    predicates = _sorted_strings(dispatch_predicates)
    record = {
        "schema": schemas.SCHEMA_CANDIDATE,
        "candidate_id": candidate_id, "campaign_id": campaign_id,
        "proposal_id": proposal["proposal_id"],
        "parent_candidate_id": proposal.get("parent_candidate_id"),
        "worktree": (dict(blocks["worktree"]) if identity is not None else {
            "path": actor.path.path,
            "branch": actor.branch.name if actor.branch else "ak/detached-candidate",
            "source_commit": source_commit,
            "clean": actor.is_clean(),
        }),
        "source_snapshot": {
            "snapshot_sha256": snapshot_sha,
            "patch_bundle_sha256": patch_sha,
        },
        "ancestry": {
            "production_base_commit": production_base_commit,
            "is_descendant_of_production_base": True,
            "proof": (
                f"git merge-base --is-ancestor {production_base_commit} {source_commit}; "
                f"instrument {instrument_commit} has exact parent {production_base_commit}"),
        },
        "build": build_block,
        "dispatch": {
            "feature_flags": dispatch_flags,
            "dispatch_predicate": "; ".join(predicates),
        },
        "affected_surface": {
            "derived_sha256": schemas.content_hash(
                {"tokens": _sorted_strings(derived_surface_tokens)}),
            "traced_sha256": (None if not traced_surface_tokens else schemas.content_hash(
                {"tokens": _sorted_strings(traced_surface_tokens)})),
            "reconciled": bool(traced_surface_tokens)
                and set(traced_surface_tokens) <= set(derived_surface_tokens),
        },
        "composition_evidence": {
            "source_tree": "llama.cpp",
            "production_base_commit": production_base_commit,
            "candidate_source_commit": source_commit,
            "patch_bundle_sha256": patch_sha,
            "actual_files": _sorted_strings(actual_files),
            "actual_hunk_ids": _sorted_strings(hunk_ids),
            "actual_symbols": _sorted_strings(symbols),
            "derived_surface_tokens": _sorted_strings(derived_surface_tokens),
            "traced_surface_tokens": _sorted_strings(traced_surface_tokens),
            "feature_flag_assignments": dict(sorted(feature_assignments.items())),
            "dispatch_predicates": predicates,
            "mechanism_id": mechanism_id,
            "change_class": change_class,
            "evaluator_id": evaluator_id,
            "evaluator_bundle_sha256": evaluator_bundle_sha256,
            "evaluator_runtime_source_label_ref": evaluator_runtime_source_label_ref,
            "protocol_ids": _sorted_strings(protocol_ids),
        },
        "determinism": {
            "class": ("not_measured" if same_seed_repeat_runs == 0 else "bitwise_stable"),
            "same_seed_repeat_runs": same_seed_repeat_runs,
        },
        "evaluator": {"id": evaluator_id, "bundle_sha256": evaluator_bundle_sha256,
                      "runtime_source_label_ref": evaluator_runtime_source_label_ref},
        "receipts": {"host_receipt": host_receipt,
                     "resource_claim_receipt": resource_claim_receipt},
        "storage": {
            "footprint_gb": _directory_gb(build_block["build_dir"]),
            "durability_class": "hash_and_provenance_only",
        },
        "evaluation_event_ids": _sorted_strings(evaluation_event_ids),
        "derived_verdicts": ({"campaign_status": status}
                              if derived_verdicts is None else dict(derived_verdicts)),
        "controller": {
            "provider": proposal["controller"]["provider"],
            "model_id": proposal["controller"]["model_id"],
            "effort": proposal["controller"]["effort"],
            "prompt_bundle_sha256": proposal["controller"]["prompt_bundle_sha256"],
        },
        "champion_status": "frontier" if status == "banked" else "none",
        "status": status, "supersession_reason": None,
        "created_at": created_at,
    }
    if banking_verdict is not None:
        record["banking_verdict"] = dict(banking_verdict)
    if artifacts is not None:
        record["artifacts"] = artifacts
    violations = schemas.validate_candidate(record)
    if violations:
        raise CandidateRecordError("candidate record is invalid: " + "; ".join(violations))
    return record


def append_candidate_idempotent(book: Any, record: Mapping[str, Any], *, kind: str) -> str:
    """Append once by candidate id + full content identity."""
    candidate_id = record.get("candidate_id")
    wanted = schemas.content_hash(record)
    with book.write_lock():
        for entry in book.read_all():
            if entry.kind == kind and entry.record_id == candidate_id:
                if schemas.content_hash(entry.payload) != wanted:
                    raise CandidateRecordError(
                        f"candidate id {candidate_id!r} already names different durable bytes")
                return entry.event_id
        return book.append(kind, dict(record), record_id=candidate_id).event_id
