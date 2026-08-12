#!/usr/bin/env python3
"""Build an observe-only AK-WM-2 archive from real completed campaign records.

The builder never invents diagnostics, outcomes, controls, or completion events.
Every projected row must resolve to a proposal-v3 event and one clean DECIDED
terminal event in the append-only campaign journal.  Diagnostic, outcome, and
matched-intervention receipts are external immutable inputs whose bytes are
hash-pinned by the build manifest.  Synthetic fixtures exercise this code, but
the CLI refuses a row that cannot join to the real journal named by the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import journal, offline_least_commitment as protocol, schemas
    from .evaluator import recipes
except ImportError:  # direct script execution
    import journal
    import offline_least_commitment as protocol
    import schemas
    from evaluator import recipes


BUILD_SCHEMA = "epyc.autokernel.least_commitment_archive_build.v1"
DIAGNOSTIC_SCHEMA = "epyc.autokernel.least_commitment_diagnostics.v1"
OUTCOME_SCHEMA = "epyc.autokernel.least_commitment_outcome.v1"
MATCH_SCHEMA = "epyc.autokernel.matched_one_factor_intervention.v1"


@dataclass(frozen=True)
class CompletedProposal:
    proposal: Mapping[str, Any]
    proposal_event_id: str
    result: Mapping[str, Any]
    completion_event_id: str

    @property
    def proposal_sha256(self) -> str:
        return schemas.content_hash(self.proposal)

    @property
    def result_sha256(self) -> str:
        return schemas.content_hash(self.result)


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _need_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: required non-empty string")
    return value


def _bound_receipt(binding: Any, *, schema: str, label: str) -> tuple[Mapping[str, Any], dict]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label}: expected {{path, sha256}} mapping")
    path_text = _need_string(binding.get("path"), f"{label}.path")
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    expected = _need_string(binding.get("sha256"), f"{label}.sha256")
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label}: SHA-256 {observed} != pinned {expected}")
    receipt = _load_json(path)
    if receipt.get("schema") != schema:
        raise ValueError(f"{label}: schema {receipt.get('schema')!r} != {schema!r}")
    return receipt, {"path": str(path), "sha256": observed}


def _completed_proposal(row: Mapping[str, Any]) -> CompletedProposal:
    root_text = _need_string(row.get("journal_root"), "row.journal_root")
    root = Path(root_text)
    if not root.is_absolute():
        raise ValueError("row.journal_root must be absolute")
    campaign_id = _need_string(row.get("campaign_id"), "row.campaign_id")
    proposal_id = _need_string(row.get("proposal_id"), "row.proposal_id")
    completion_id = _need_string(
        row.get("completion_event_id"), "row.completion_event_id")
    entries = journal.Journal(str(root), campaign_id=campaign_id).read_all()
    proposals = [
        entry for entry in entries
        if entry.kind == journal.KIND_PROPOSAL_RECORDED
        and entry.record_id == proposal_id
    ]
    if len(proposals) != 1:
        raise ValueError(
            f"{campaign_id}/{proposal_id}: expected one proposal event, got {len(proposals)}")
    proposal = proposals[0].payload
    if proposal.get("schema") not in {
            schemas.SCHEMA_PROPOSAL_V3, schemas.SCHEMA_PROPOSAL_V4}:
        raise ValueError(f"{proposal_id}: only proposal.v3/v4 is archive-eligible")
    violations = schemas.validate_record(proposal)
    if violations:
        raise ValueError(f"{proposal_id}: invalid proposal: {'; '.join(violations)}")
    terminals = [
        entry for entry in entries
        if entry.kind == journal.KIND_STOP_STATE and entry.event_id == completion_id
    ]
    if len(terminals) != 1:
        raise ValueError(
            f"{campaign_id}/{proposal_id}: completion event {completion_id!r} does not resolve")
    payload = terminals[0].payload
    result = payload.get("result") if isinstance(payload, Mapping) else None
    if not isinstance(result, Mapping):
        raise ValueError(f"{completion_id}: terminal event carries no campaign result")
    spec = result.get("spec")
    result_proposal = spec.get("proposal") if isinstance(spec, Mapping) else None
    clean = {
        "state_decided": payload.get("state") == "decided" and result.get("state") == "decided",
        "campaign": result.get("campaign_id") == campaign_id,
        "proposal": isinstance(result_proposal, Mapping)
                    and result_proposal.get("proposal_id") == proposal_id,
        "executed": result.get("executed") is True,
        "ok": result.get("ok") is True,
        "decision": isinstance(result.get("decision"), Mapping),
        "production": isinstance(result.get("production_unchanged"), Mapping)
                      and result["production_unchanged"].get("outcome") == schemas.PASS,
        "releases": isinstance(result.get("releases"), list)
                    and bool(result["releases"])
                    and all(isinstance(item, Mapping) and item.get("released") is True
                            for item in result["releases"]),
        "pairs": isinstance(result.get("pairs"), list) and bool(result["pairs"]),
    }
    failed = sorted(name for name, passed in clean.items() if not passed)
    if failed:
        raise ValueError(
            f"{completion_id}: terminal campaign is not a clean completed proposal: {failed}")
    return CompletedProposal(
        proposal=proposal, proposal_event_id=proposals[0].event_id,
        result=result, completion_event_id=completion_id)


def _metric_direction(completed: CompletedProposal) -> tuple[str, str]:
    spec = completed.result["spec"]
    recipe = recipes.get_recipe(spec["recipe_id"])
    return recipe.metric, recipe.metric_direction


def _validate_diagnostic_receipt(
        receipt: Mapping[str, Any], *, completed: CompletedProposal,
        frame_sha256: str, demand_sha256: str) -> None:
    proposal_id = completed.proposal["proposal_id"]
    checks = {
        "proposal_id": receipt.get("proposal_id") == proposal_id,
        "proposal_sha256": receipt.get("proposal_sha256") == completed.proposal_sha256,
        "representation_frame": receipt.get("representation_frame_sha256") == frame_sha256,
        "demand_frame": receipt.get("empirical_demand_weights_sha256") == demand_sha256,
        "diagnostics": isinstance(receipt.get("diagnostics"), Mapping)
                       and set(receipt["diagnostics"]) == set(protocol.DIAGNOSTICS),
        "recodings": isinstance(receipt.get("recodings"), Mapping),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"diagnostic receipt for {proposal_id} failed: {failed}")


def _validate_outcome_receipt(
        receipt: Mapping[str, Any], *, completed: CompletedProposal,
        candidate_frame_id: str, frame_sha256: str, demand_sha256: str,
        metric_direction: str) -> None:
    proposal_id = completed.proposal["proposal_id"]
    metric, observed_direction = _metric_direction(completed)
    checks = {
        "proposal_id": receipt.get("proposal_id") == proposal_id,
        "completion_event_id": receipt.get("completion_event_id")
                               == completed.completion_event_id,
        "campaign_result_sha256": receipt.get("campaign_result_sha256")
                                  == completed.result_sha256,
        "candidate_frame": receipt.get("candidate_frame_id") == candidate_frame_id,
        "representation_frame": receipt.get("representation_frame_sha256") == frame_sha256,
        "demand_frame": receipt.get("empirical_demand_weights_sha256") == demand_sha256,
        "metric": receipt.get("metric") == metric,
        "metric_direction": receipt.get("metric_direction") == metric_direction
                            == observed_direction,
        "regime": isinstance(receipt.get("regime"), str) and bool(receipt["regime"]),
        "surface": isinstance(receipt.get("surface"), str) and bool(receipt["surface"]),
        "intervention_id": isinstance(receipt.get("intervention_id"), str)
                           and bool(receipt["intervention_id"]),
        "changed_factor": isinstance(receipt.get("changed_factor"), str)
                          and bool(receipt["changed_factor"]),
        "outcome": isinstance(receipt.get("outcome"), Mapping),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"outcome receipt for {proposal_id} failed: {failed}")


def _validate_match_receipt(
        receipt: Mapping[str, Any], *, intervention: CompletedProposal,
        control: CompletedProposal, intervention_outcome: Mapping[str, Any],
        control_outcome: Mapping[str, Any], candidate_frame_id: str) -> None:
    checks = {
        "intervention_proposal": receipt.get("intervention_proposal_id")
                                 == intervention.proposal["proposal_id"],
        "control_proposal": receipt.get("control_proposal_id")
                            == control.proposal["proposal_id"],
        "intervention_completion": receipt.get("intervention_completion_event_id")
                                   == intervention.completion_event_id,
        "control_completion": receipt.get("control_completion_event_id")
                              == control.completion_event_id,
        "candidate_frame": receipt.get("candidate_frame_id") == candidate_frame_id,
        "regime": receipt.get("regime") == intervention_outcome.get("regime")
                  == control_outcome.get("regime"),
        "surface": receipt.get("surface") == intervention_outcome.get("surface")
                   == control_outcome.get("surface"),
        "changed_factor": receipt.get("changed_factor")
                          == intervention_outcome.get("changed_factor"),
        "one_factor": receipt.get("one_factor") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"matched intervention {intervention.proposal['proposal_id']} failed: {failed}")


def build_archive(manifest: Mapping[str, Any]) -> dict:
    if manifest.get("schema") != BUILD_SCHEMA:
        raise ValueError(f"schema: expected {BUILD_SCHEMA!r}")
    archive_id = _need_string(manifest.get("archive_id"), "archive_id")
    created_at = _need_string(manifest.get("created_at"), "created_at")
    candidate_frame = _need_string(
        manifest.get("candidate_frame_id"), "candidate_frame_id")
    frame_sha = _need_string(
        manifest.get("representation_frame_sha256"), "representation_frame_sha256")
    demand_sha = _need_string(
        manifest.get("empirical_demand_weights_sha256"),
        "empirical_demand_weights_sha256")
    metric_direction = _need_string(
        manifest.get("metric_direction"), "metric_direction")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows: at least one real completed proposal is required")

    loaded: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rows[{index}]: expected mapping")
        completed = _completed_proposal(row)
        proposal_id = completed.proposal["proposal_id"]
        if proposal_id in loaded:
            raise ValueError(f"rows[{index}]: duplicate proposal {proposal_id!r}")
        contract = completed.proposal["representation_contract"]
        if contract["frame_sha256"] != frame_sha \
                or contract["empirical_demand"]["weights_sha256"] != demand_sha:
            raise ValueError(
                f"{proposal_id}: proposal representation or demand frame differs")
        diagnostic, diagnostic_ref = _bound_receipt(
            row.get("diagnostic_receipt"), schema=DIAGNOSTIC_SCHEMA,
            label=f"rows[{index}].diagnostic_receipt")
        outcome, outcome_ref = _bound_receipt(
            row.get("outcome_receipt"), schema=OUTCOME_SCHEMA,
            label=f"rows[{index}].outcome_receipt")
        _validate_diagnostic_receipt(
            diagnostic, completed=completed, frame_sha256=frame_sha,
            demand_sha256=demand_sha)
        _validate_outcome_receipt(
            outcome, completed=completed, candidate_frame_id=candidate_frame,
            frame_sha256=frame_sha, demand_sha256=demand_sha,
            metric_direction=metric_direction)
        loaded[proposal_id] = {
            "completed": completed, "diagnostic": diagnostic,
            "diagnostic_ref": diagnostic_ref, "outcome": outcome,
            "outcome_ref": outcome_ref, "row": row,
        }

    projected = []
    matched = 0
    for proposal_id, item in loaded.items():
        row = item["row"]
        control_id = row.get("matched_control_id")
        match_ref = None
        if control_id is not None:
            _need_string(control_id, f"{proposal_id}.matched_control_id")
            if control_id not in loaded:
                raise ValueError(f"{proposal_id}: matched control {control_id!r} is absent")
            match, match_ref = _bound_receipt(
                row.get("matched_intervention_receipt"), schema=MATCH_SCHEMA,
                label=f"{proposal_id}.matched_intervention_receipt")
            _validate_match_receipt(
                match, intervention=item["completed"],
                control=loaded[control_id]["completed"],
                intervention_outcome=item["outcome"],
                control_outcome=loaded[control_id]["outcome"],
                candidate_frame_id=candidate_frame)
            matched += 1
        elif row.get("matched_intervention_receipt") is not None:
            raise ValueError(
                f"{proposal_id}: control row cannot carry an intervention receipt")
        outcome = item["outcome"]
        projected.append({
            "proposal_id": proposal_id,
            "completion_event_id": item["completed"].completion_event_id,
            "candidate_frame_id": candidate_frame,
            "regime": outcome["regime"], "surface": outcome["surface"],
            "intervention_id": outcome["intervention_id"],
            "changed_factor": outcome["changed_factor"],
            "matched_control_id": control_id,
            "representation_contract": item["completed"].proposal[
                "representation_contract"],
            "diagnostics": item["diagnostic"]["diagnostics"],
            "outcome": item["outcome"]["outcome"],
            "recodings": item["diagnostic"]["recodings"],
            "source_receipts": {
                "proposal_event_id": item["completed"].proposal_event_id,
                "proposal_sha256": item["completed"].proposal_sha256,
                "campaign_result_sha256": item["completed"].result_sha256,
                "diagnostic": item["diagnostic_ref"],
                "outcome": item["outcome_ref"],
                "matched_intervention": match_ref,
            },
        })
    if matched == 0:
        raise ValueError("rows: no real matched one-factor intervention receipt")
    archive = {
        "schema": protocol.ARCHIVE_SCHEMA,
        "archive_id": archive_id, "created_at": created_at,
        "protocol_id": protocol.PROTOCOL_ID, "authority": protocol.AUTHORITY,
        "candidate_frame_id": candidate_frame,
        "diagnostic_directions": manifest.get("diagnostic_directions"),
        "outcome_weights": manifest.get("outcome_weights"),
        "rows": projected,
        "build_provenance": {
            "schema": BUILD_SCHEMA,
            "representation_frame_sha256": frame_sha,
            "empirical_demand_weights_sha256": demand_sha,
            "metric_direction": metric_direction,
        },
    }
    errors = protocol.validate_archive(archive)
    if errors:
        raise ValueError("projected archive is invalid:\n- " + "\n- ".join(errors))
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = _load_json(args.manifest)
    archive = build_archive(manifest)
    args.output.write_text(
        json.dumps(archive, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
