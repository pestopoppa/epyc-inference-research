#!/usr/bin/env python3
"""Lean deterministic AutoKernel sequencer.

The sequencer orders already-materialized proposals, delegates each candidate run,
then asks :mod:`champion` to compose and re-evaluate compatible banked candidates.
It has no authoring, tree, build, benchmark, process, release, or production-write
capability.  Those actions belong exclusively to injected runners.

The module is deliberately not imported by ``controller.__init__``.  Consequently
``campaign.py`` -- which imports the controller package for the hypothesis gate --
cannot reach this loop accidentally.
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from .. import journal, schemas
from . import champion
from .shared import ControllerError

__all__ = [
    "SequencerError", "StopReason", "ProposalEnvelope", "CampaignRun",
    "CampaignRunner", "ProposalSupplier", "AnchorProvider", "StaticAnchorProvider",
    "ListProposalSupplier", "LoopBudget", "LoopResult", "Sequencer",
    "build_parser", "main",
]


class SequencerError(ControllerError):
    """The loop could not preserve its journal/identity invariants."""


class StopReason(str, Enum):
    TURN_LIMIT = "TURN_LIMIT"
    BUDGET = "BUDGET_STOP"
    NO_PROPOSAL = "NO_PROPOSAL"
    NO_PROGRESS = "NO_PROGRESS"
    ANCHOR_MOVED = "ANCHOR_MOVED"
    EVALUATOR_COVERAGE = "EVALUATOR_COVERAGE_GAP"


@dataclass(frozen=True)
class ProposalEnvelope:
    campaign: Mapping[str, Any]
    proposal: Mapping[str, Any]

    def __post_init__(self) -> None:
        campaign_violations = schemas.validate_campaign(self.campaign)
        proposal_violations = schemas.validate_proposal(self.proposal)
        if campaign_violations:
            raise SequencerError("invalid campaign: " + "; ".join(campaign_violations))
        if proposal_violations:
            raise SequencerError("invalid proposal: " + "; ".join(proposal_violations))
        if self.proposal.get("campaign_id") != self.campaign.get("campaign_id"):
            raise SequencerError("proposal and campaign ids do not match")


@dataclass(frozen=True)
class CampaignRun:
    """Final journal records returned by an execution-owning campaign adapter."""

    candidate_records: tuple[Mapping[str, Any], ...]
    evaluation_events: tuple[Mapping[str, Any], ...]


class CampaignRunner(Protocol):
    def run_campaign(self, proposal: ProposalEnvelope) -> CampaignRun: ...


class ProposalSupplier(Protocol):
    def next_proposal(self) -> Optional[ProposalEnvelope]: ...


class AnchorProvider(Protocol):
    def current_anchor(self, source_tree: str) -> champion.AnchorIdentity: ...


@dataclass(frozen=True)
class StaticAnchorProvider:
    anchors: Mapping[str, champion.AnchorIdentity]

    def current_anchor(self, source_tree: str) -> champion.AnchorIdentity:
        try:
            return self.anchors[source_tree]
        except KeyError as exc:
            raise SequencerError(f"no anchor configured for {source_tree!r}") from exc


class ListProposalSupplier:
    def __init__(self, proposals: Iterable[ProposalEnvelope]):
        self._proposals = list(proposals)
        self._offset = 0

    def next_proposal(self) -> Optional[ProposalEnvelope]:
        if self._offset >= len(self._proposals):
            return None
        result = self._proposals[self._offset]
        self._offset += 1
        return result


@dataclass(frozen=True)
class LoopBudget:
    max_turns: int = 100
    max_candidates: int = 100
    no_progress_turns: int = 3

    def __post_init__(self) -> None:
        for name in ("max_turns", "max_candidates", "no_progress_turns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class LoopResult:
    stop_reason: StopReason
    turns: int
    candidates_run: int
    champions_updated: int
    detail: str

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.sequencer_result.v1",
            "stop_reason": self.stop_reason.value, "turns": self.turns,
            "candidates_run": self.candidates_run,
            "champions_updated": self.champions_updated, "detail": self.detail,
        }


def _record_stop(book: journal.Journal, result: LoopResult) -> journal.JournalEntry:
    payload = {
        "state": result.stop_reason.value,
        "turns": result.turns,
        "candidates_run": result.candidates_run,
        "champions_updated": result.champions_updated,
        "detail": result.detail,
    }
    return champion.append_idempotent(book, journal.KIND_STOP_STATE, payload)


def _latest_champion_entry(snapshot: champion.JournalSnapshot,
                           source_tree: str) -> Optional[journal.JournalEntry]:
    matches = [entry for entry in snapshot.entries
               if entry.kind == journal.KIND_CHAMPION_UPDATED
               and entry.payload.get("source_tree") == source_tree]
    return max(matches, key=lambda entry: entry.seq) if matches else None


def _anchor_from_champion(record: Mapping[str, Any]) -> champion.AnchorIdentity:
    raw = record.get("anchor")
    if not isinstance(raw, Mapping):
        raise SequencerError("champion predates exact anchor identity; reanchor is required")
    return champion.AnchorIdentity.from_dict(raw)


def _validate_campaign_run(run: CampaignRun, envelope: ProposalEnvelope) -> None:
    if not isinstance(run, CampaignRun):
        raise SequencerError("campaign runner must return CampaignRun")
    proposal_id = envelope.proposal["proposal_id"]
    campaign_id = envelope.campaign["campaign_id"]
    candidate_ids: set[str] = set()
    for candidate in run.candidate_records:
        violations = schemas.validate_candidate(candidate)
        if violations:
            raise SequencerError("runner returned invalid candidate: " + "; ".join(violations))
        if candidate.get("proposal_id") != proposal_id or candidate.get("campaign_id") != campaign_id:
            raise SequencerError("runner candidate is not bound to the supplied proposal")
        candidate_id = candidate["candidate_id"]
        if candidate_id in candidate_ids:
            raise SequencerError("runner returned duplicate candidate identities")
        candidate_ids.add(candidate_id)
    evaluation_ids: set[str] = set()
    for event in run.evaluation_events:
        violations = schemas.validate_evaluation_event(event)
        if violations:
            raise SequencerError("runner returned invalid evaluation: " + "; ".join(violations))
        if event.get("campaign_id") != campaign_id or event.get("candidate_id") not in candidate_ids:
            raise SequencerError("runner evaluation is not bound to its returned candidate")
        if event["event_id"] in evaluation_ids:
            raise SequencerError("runner returned duplicate evaluation identities")
        evaluation_ids.add(event["event_id"])
    for candidate in run.candidate_records:
        if set(candidate.get("evaluation_event_ids") or ()) != {
                event["event_id"] for event in run.evaluation_events
                if event.get("candidate_id") == candidate["candidate_id"]}:
            raise SequencerError("candidate does not cite exactly its returned evaluations")


class Sequencer:
    def __init__(self, *, book: journal.Journal, proposal_supplier: ProposalSupplier,
                 campaign_runner: CampaignRunner,
                 composition_runner: champion.CompositionRunner,
                 anchor_provider: AnchorProvider,
                 evaluators: Mapping[str, champion.EvaluatorIdentity],
                 budget: LoopBudget = LoopBudget(),
                 reanchor_runner: Optional[champion.ReanchorRunner] = None):
        if not isinstance(book, journal.Journal):
            raise TypeError("book must be a journal.Journal")
        if set(evaluators) - schemas.SOURCE_TREES:
            raise ValueError("evaluators contain an unknown source tree")
        if not evaluators:
            raise ValueError("at least one source-tree evaluator is required")
        self.book = book
        self.proposals = proposal_supplier
        self.campaign_runner = campaign_runner
        self.composition_runner = composition_runner
        self.reanchor_runner = reanchor_runner
        self.anchor_provider = anchor_provider
        self.evaluators = dict(evaluators)
        self.budget = budget

    def _append_campaign_output(self, envelope: ProposalEnvelope,
                                run: CampaignRun) -> bool:
        _validate_campaign_run(run, envelope)
        # Candidate first is intentional: a crash leaves a visible, incomplete
        # candidate that cannot enter the frontier.  Replay resolves the exact
        # candidate and appends the missing evaluations.
        for candidate in run.candidate_records:
            champion.append_idempotent(self.book, journal.KIND_CANDIDATE_RECORDED, candidate)
        for event in run.evaluation_events:
            champion.append_idempotent(self.book, journal.KIND_EVALUATION_EVENT, event)
        return any(candidate.get("status") == "banked" for candidate in run.candidate_records)

    def _check_anchor(self, source_tree: str) -> tuple[champion.AnchorIdentity, bool]:
        current = self.anchor_provider.current_anchor(source_tree)
        if current.source_tree != source_tree:
            raise SequencerError("anchor provider returned another source tree")
        snapshot = champion.read_validated_snapshot(self.book)
        latest = _latest_champion_entry(snapshot, source_tree)
        if latest is None or (latest.payload.get("status") == "no_champion"
                              and not latest.payload.get("member_candidates")):
            return current, False
        recorded = _anchor_from_champion(latest.payload)
        if recorded.same_denominator(current):
            return current, False
        champion.record_anchor_moved(self.book, latest.payload,
                                     old_anchor=recorded, new_anchor=current)
        if self.reanchor_runner is None or not latest.payload.get("member_candidates"):
            return current, True
        try:
            champion.reanchor_champion(
                self.book, prior_champion=latest.payload, old_anchor=recorded,
                new_anchor=current, evaluator=self.evaluators[source_tree],
                runner=self.reanchor_runner)
        except Exception as exc:  # runner failures remain a hard stop
            # Do not turn the exception into a fabricated result.  The already-
            # journaled anchor-moved record is the durable failure evidence.
            raise champion.AnchorMoved(f"reanchor failed: {type(exc).__name__}: {exc}") from exc
        return current, False

    def _attempt_composition(self, source_tree: str,
                             anchor: champion.AnchorIdentity) -> tuple[bool, Optional[StopReason], str]:
        snapshot = champion.read_validated_snapshot(self.book)
        state = champion.project_source_tree(snapshot, anchor)
        # A composed candidate is evidence ABOUT members, never itself a member of
        # its next composition.
        frontier = [state.candidates[cid] for cid in state.frontier
                    if cid in state.candidates
                    and not isinstance(state.candidates[cid].record.get("composition_lineage"), Mapping)]
        if not frontier:
            if state.active_champion is None:
                champion.record_no_champion(self.book, anchor,
                                             reason="no green banked candidate")
            return False, None, "no green banked candidate"
        evaluator = self.evaluators[source_tree]
        report = champion.compatibility(frontier, anchor=anchor, evaluator=evaluator)
        if not report.compatible:
            # Never silently choose one incompatible group by id or by a local
            # percentage.  Preserve the active champion and journal the conflict.
            champion.record_rejected_composition(self.book, anchor, report)
            if report.conflicts and all(item.startswith("evaluator_or_protocol_mismatch:")
                                        for item in report.conflicts):
                return False, StopReason.EVALUATOR_COVERAGE, "; ".join(report.conflicts)
            return False, None, "; ".join(report.conflicts)
        prior = _latest_champion_entry(snapshot, source_tree)
        request = champion.composition_request(
            frontier, anchor=anchor, evaluator=evaluator,
            parent_champion_event_id=None if prior is None else prior.event_id)
        member_ids = request.member_candidates
        if state.active_champion is not None \
                and tuple(state.active_champion.get("member_candidates") or ()) == member_ids \
                and state.active_champion.get("status") in {"active", "reanchored"}:
            return False, None, "champion already covers this exact lineage"
        champion.promote_composition(self.book, request, self.composition_runner,
                                     snapshot=snapshot)
        return True, None, f"composed {list(request.member_candidates)}"

    def _stop(self, reason: StopReason, *, turns: int, candidates: int,
              champions: int, detail: str) -> LoopResult:
        result = LoopResult(reason, turns, candidates, champions, detail)
        _record_stop(self.book, result)
        return result

    def run(self) -> LoopResult:
        turns = 0
        candidate_runs = 0
        champion_updates = 0
        no_progress = 0
        while turns < self.budget.max_turns:
            anchors: dict[str, champion.AnchorIdentity] = {}
            for source_tree in sorted(self.evaluators):
                try:
                    anchor, moved = self._check_anchor(source_tree)
                except champion.AnchorMoved as exc:
                    return self._stop(StopReason.ANCHOR_MOVED, turns=turns,
                                      candidates=candidate_runs, champions=champion_updates,
                                      detail=str(exc))
                anchors[source_tree] = anchor
                if moved:
                    return self._stop(StopReason.ANCHOR_MOVED, turns=turns,
                                      candidates=candidate_runs, champions=champion_updates,
                                      detail=f"{source_tree} denominator moved; reanchor evidence unavailable")
            if candidate_runs >= self.budget.max_candidates:
                return self._stop(StopReason.BUDGET, turns=turns,
                                  candidates=candidate_runs, champions=champion_updates,
                                  detail="candidate budget exhausted")
            envelope = self.proposals.next_proposal()
            if envelope is None:
                details = []
                for source_tree, anchor in anchors.items():
                    try:
                        updated, stop, detail = self._attempt_composition(source_tree, anchor)
                    except Exception as exc:
                        return self._stop(
                            StopReason.NO_PROGRESS, turns=turns,
                            candidates=candidate_runs, champions=champion_updates,
                            detail=f"composition runner failed: {type(exc).__name__}: {exc}")
                    champion_updates += int(updated)
                    details.append(f"{source_tree}: {detail}")
                    if stop is not None:
                        return self._stop(stop, turns=turns, candidates=candidate_runs,
                                          champions=champion_updates, detail=detail)
                return self._stop(StopReason.NO_PROPOSAL, turns=turns,
                                  candidates=candidate_runs, champions=champion_updates,
                                  detail="; ".join(details))
            source_tree = envelope.campaign["source_tree"]
            if source_tree not in anchors:
                return self._stop(StopReason.EVALUATOR_COVERAGE, turns=turns,
                                  candidates=candidate_runs, champions=champion_updates,
                                  detail=f"no evaluator configured for {source_tree}")
            champion.append_idempotent(self.book, journal.KIND_CAMPAIGN_OPENED,
                                       envelope.campaign)
            champion.append_idempotent(self.book, journal.KIND_PROPOSAL_RECORDED,
                                       envelope.proposal)
            try:
                run = self.campaign_runner.run_campaign(envelope)
                banked = self._append_campaign_output(envelope, run)
            except Exception as exc:
                failure = {
                    "proposal_ref": envelope.proposal["proposal_id"],
                    "reason": f"campaign runner failed: {type(exc).__name__}: {exc}",
                }
                champion.append_idempotent(
                    self.book, journal.KIND_PROPOSAL_SKIPPED, failure)
                banked = False
            candidate_runs += 1
            turns += 1
            try:
                updated, stop, detail = self._attempt_composition(
                    source_tree, anchors[source_tree])
            except Exception as exc:
                return self._stop(
                    StopReason.NO_PROGRESS, turns=turns,
                    candidates=candidate_runs, champions=champion_updates,
                    detail=f"composition runner failed: {type(exc).__name__}: {exc}")
            champion_updates += int(updated)
            if stop is not None:
                return self._stop(stop, turns=turns, candidates=candidate_runs,
                                  champions=champion_updates, detail=detail)
            progressed = banked or updated
            no_progress = 0 if progressed else no_progress + 1
            if no_progress >= self.budget.no_progress_turns:
                return self._stop(StopReason.NO_PROGRESS, turns=turns,
                                  candidates=candidate_runs, champions=champion_updates,
                                  detail=f"{no_progress} consecutive turns produced no banked candidate or champion update")
        return self._stop(StopReason.TURN_LIMIT, turns=turns,
                          candidates=candidate_runs, champions=champion_updates,
                          detail="turn limit reached")


def _load_json(path: str) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise SequencerError(f"{path}: expected a JSON object")
    return value


def _load_factory(spec: str) -> Any:
    if ":" not in spec:
        raise SequencerError("runner factory must be module:attribute")
    module_name, attribute = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    return value() if callable(value) else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m scripts.kernel_rnd.autokernel.controller.sequencer",
        description="Lean AutoKernel proposal/campaign/composition sequencer. Inspect-only unless --run is explicit.")
    parser.add_argument("--journal-root", required=True)
    parser.add_argument("--campaign", action="append", default=[])
    parser.add_argument("--proposal", action="append", default=[])
    parser.add_argument("--identity-manifest", required=True,
                        help="JSON with anchors{} and evaluators{} by source tree")
    parser.add_argument("--runner-factory",
                        help="module:attribute returning campaign/composition[/reanchor] runners")
    parser.add_argument("--run", action="store_true",
                        help="invoke injected runners; absent means validated projection only")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--no-progress-turns", type=int, default=3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.campaign) != len(args.proposal):
        raise SequencerError("--campaign and --proposal counts must match")
    identity = _load_json(args.identity_manifest)
    anchors = {tree: champion.AnchorIdentity.from_dict(raw)
               for tree, raw in (identity.get("anchors") or {}).items()}
    evaluators = {
        tree: champion.EvaluatorIdentity.from_dict(raw)
        for tree, raw in (identity.get("evaluators") or {}).items()
    }
    book = journal.Journal(args.journal_root)
    proposals = [ProposalEnvelope(_load_json(campaign_path), _load_json(proposal_path))
                 for campaign_path, proposal_path in zip(args.campaign, args.proposal)]
    if not args.run:
        snapshot = champion.read_validated_snapshot(book)
        output = {
            "mode": "inspect_only", "journal_entries": len(snapshot.entries),
            "source_trees": {
                tree: {
                    "frontier": list(champion.project_source_tree(snapshot, anchor).frontier),
                    "champion": champion.project_source_tree(snapshot, anchor).composed_champion,
                    "incumbent_role": champion.LifecycleRole.PRODUCTION_INCUMBENT.value,
                } for tree, anchor in anchors.items()
            },
            "proposal_count": len(proposals),
        }
        print(json.dumps(output, sort_keys=True, indent=2))
        return 0
    if not args.runner_factory:
        raise SequencerError("--run requires --runner-factory")
    runners = _load_factory(args.runner_factory)
    if not isinstance(runners, (tuple, list)) or len(runners) not in {2, 3}:
        raise SequencerError("runner factory must return (campaign, composition[, reanchor])")
    loop = Sequencer(
        book=book, proposal_supplier=ListProposalSupplier(proposals),
        campaign_runner=runners[0], composition_runner=runners[1],
        reanchor_runner=None if len(runners) == 2 else runners[2],
        anchor_provider=StaticAnchorProvider(anchors), evaluators=evaluators,
        budget=LoopBudget(args.max_turns, args.max_candidates, args.no_progress_turns))
    print(json.dumps(loop.run().to_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
