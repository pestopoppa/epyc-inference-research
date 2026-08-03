"""composition.py — champion maintenance (design §8.9).

WHY THIS MODULE EXISTS
----------------------
§8.9 keeps three things apart, and every way this loop can be quietly wrong is a
way of letting two of them merge:

  * **frontier candidates** — correct, non-dominated results. Many, per campaign.
  * **THE CHAMPION** — one compatible composed lineage per **SOURCE TREE**. Note
    §1.5: CPU and GPU share `llama.cpp` and one frozen branch, so `llama_cpu` and
    `llama_gpu` campaigns converge on ONE champion. There is no per-backend
    champion to converge later.
  * **experiments**, including §8.4.1 spikes — diagnostic branches that *may
    never accumulate*. A spike "is never banked, never enters the champion, and
    never carries a correctness claim".

Four failures this module is written against, each enforced rather than
documented:

1. **Composition inferred by multiplying local speedups.** §8.9: *"After
   combining, rerun T0/T1 on the combined full candidate; never infer composition
   by multiplying local speedups."* §12 names it too (*"Summed local gains inflate
   readiness"*), and P-AK-SEARCH-1 denial 9 forbids synthesising a decision-grade
   quantity by combining search records. Making that *discouraged* is worthless.
   So it is made STRUCTURALLY IMPOSSIBLE, three ways at once:
     - `LineageMember` — the only shape a member can take inside a lineage — has
       no field that can hold a rate, an effect, an estimate or a sample, and
       `_assert_no_forbidden_fields()` fails at IMPORT if one is ever added;
     - `compose_champion()` accepts no member evidence at all. Its only evidence
       parameters resolve records **of the combined candidate**, from the journal;
     - this module contains **no multiplication, division or exponentiation
       anywhere, and reads no performance field of any record**.
       `audit_no_composed_estimate_arithmetic()` proves both from this file's own
       AST rather than asserting it in prose.
   The readiness *magnitude* is therefore not computed here at all. That is the
   point: invariant 14 gives it to a deterministic reducer over the combined
   candidate's own events, and a composer that could reach a member's number is a
   composer that could add them up.

2. **An unreconciled surface entering a composition.** §8.9: *"Only changes with
   reconciled affected-surface maps may be combined."* Reconciliation is
   `evaluator/surface.reconcile_surface()`'s job (§6.4 stage 3, invariant 18) and
   is NOT reimplemented here: `propose_lineage()` takes the
   `SurfaceReconciliation` object itself, asks `surface`'s own
   `candidate_affected_surface_block()` whether it reconciled, and additionally
   binds it to the candidate record so a healthy reconciliation cannot be
   presented for a record that says otherwise.

3. **A search that collapses to one family.** §8.9: *"Retain diversity across
   mechanism classes so one noisy early win does not collapse the search to a
   single family."* `retain_frontier()` is the only retention path, and it fills a
   per-class quota BEFORE it fills by the caller's preference order, so the
   highest-ranked family cannot evict the last representative of another. A
   capacity too small to hold the floor is `DiversityFloorUnmet`, never a silent
   truncation. Composition never RANKS — the order it is handed is the caller's,
   produced by `evaluator.api.rank_candidates`, which refuses a rank to anything
   that did not earn one.

4. **A champion forked from a dead anchor.** §8.9/AK-D22: an emergency hot-fix or
   a rollback leaves every ratio in the journal with a denominator that no longer
   exists. `compose_champion()` re-verifies anchor identity on the way in and
   REFUSES; `respond_to_anchor_move()` performs §8.9's five steps — halt, mark
   `superseded_by_anchor_move` carrying both identities, PRESERVE source/patches/
   correctness, re-anchor, notify the operator — and `plan_reanchor()` cannot
   produce a champion record for a non-empty lineage at all, which is what forces
   *"T1/T2 evidence is invalidated and re-measured"* to actually happen.

WHAT THIS MODULE IS NOT
-----------------------
It runs no inference, no benchmark and no build; it starts, stops and signals no
process; it calls no model; it computes no readiness value and it does not rank.
Its only writes are journal appends the caller asks for, through `journal.py`.

Governing instrument: `measurement/protocols/kernel-research.md` (P-AK-SEARCH-1,
RATIFIED 2026-08-03), which authorizes *"compose compatible candidates into a
champion lineage and re-measure the composition as a whole"* and denies
everything past it. Owning design:
`epyc-root/handoffs/active/autokernel-research-loop.md` §8.9, with §1.5, §4
invariants 1/6/8/14/18, §7.3, §7.5, §8.4.1, §9.6 and §12.
"""
from __future__ import annotations

import ast
import dataclasses
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from .. import journal, schemas
from ..evaluator import surface as surface_mod
from . import state_machine

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")      # mirrors schemas._COMMIT_RE

__all__ = [
    # errors
    "CompositionError", "UnreconciledSurface", "IncompatibleMember", "NotBanked",
    "ExperimentMayNotAccumulate", "CompositionEvidenceMissing",
    "DiversityFloorUnmet", "AnchorMovedRefusal", "ReanchorRefused",
    "SupersessionScopeViolation",
    # vocabulary
    "MECHANISM_CLASSES", "REQUIRED_COMBINED_TIERS", "OPTIONAL_COMBINED_TIERS",
    "COMPARISON_TIERS", "PRESERVED_TIERS", "EXPERIMENT_KINDS",
    "SUPERSEDED_BY_ANCHOR_MOVE", "REANCHOR_TRIGGER_FREEZE",
    "REANCHOR_TRIGGER_ANCHOR_MOVED", "REANCHOR_TRIGGERS",
    "BLOCKING_REANCHOR_REMEASURE",
    # the three concepts, as three types
    "FrontierCandidate", "Experiment", "LineageMember", "ComposedLineage",
    # admission
    "source_tree_for_backends", "admit_to_frontier", "record_experiment",
    "propose_lineage", "champion_branch_for",
    # the champion
    "compose_champion", "record_champion",
    # diversity
    "retain_frontier", "check_mechanism_diversity",
    # re-anchoring
    "ReanchorPlan", "plan_reanchor",
    # anchor moved
    "AnchorMoveSweep", "AnchorMoveResponse", "affected_backends_for_move",
    "plan_anchor_move_supersession", "apply_anchor_move_supersession",
    "respond_to_anchor_move",
    # self-audit
    "audit_no_composed_estimate_arithmetic",
]


# =============================================================================
# Errors — every one is a refusal, never a degraded result
# =============================================================================

class CompositionError(state_machine.ControllerError):
    """Base for every refusal here. Subclasses `ControllerError` so a caller can
    catch the controller plane's refusals as one family."""


class UnreconciledSurface(CompositionError):
    """§8.9: only changes with RECONCILED affected-surface maps may be combined."""


class IncompatibleMember(CompositionError):
    """A member that does not belong to this tree, this anchor, or this artifact."""


class NotBanked(CompositionError):
    """§9.6: banking is the evaluator's disposition. Only a banked candidate is a
    frontier candidate, and this module never sets the field it reads."""


class ExperimentMayNotAccumulate(CompositionError):
    """§8.4.1: a spike is never banked and never enters the champion."""


class CompositionEvidenceMissing(CompositionError):
    """The combined full candidate has no passing T0/T1 of its own (§8.9)."""


class DiversityFloorUnmet(CompositionError):
    """Retention capacity cannot hold one representative per mechanism class."""


class AnchorMovedRefusal(CompositionError):
    """Anchor identity moved; no further candidate work for this tree (§8.9)."""


class ReanchorRefused(CompositionError):
    """A re-anchor whose inputs contradict the champion it claims to re-anchor."""


class SupersessionScopeViolation(CompositionError):
    """An anchor-move sweep that would take out evidence §8.9 says survives."""


# =============================================================================
# Vocabulary
# =============================================================================

#: §8.9's "mechanism classes" are realized as the ratified CHANGE CLASS. It is
#: the only CLOSED vocabulary in the contracts that names a change's family
#: (`schemas.CHANGE_CLASSES`, §7.2/§9.5), and diversity has to be measured over a
#: closed set or "a new class" becomes the way to dodge the floor. Aliased rather
#: than copied so a class added to the contract is a class this floor counts.
MECHANISM_CLASSES = schemas.CHANGE_CLASSES

#: §8.9: *"After combining, rerun T0/T1 on the combined full candidate."* Both,
#: on the COMBINED id, or there is no champion.
REQUIRED_COMBINED_TIERS = ("T0", "T1")

#: §9.7 runs on the composed champion when interaction is the dominant
#: uncertainty. It is recorded when it exists and never required to exist.
OPTIONAL_COMBINED_TIERS = ("T2",)

#: Tiers whose records ARE comparisons against the anchor. These are what an
#: anchor move kills — *"only the comparisons died, not the work"* (§8.9 item 3).
COMPARISON_TIERS = ("T1", "T1a", "T1b", "T1c", "T2")

#: T0 compares ARTIFACTS, not rates (`schemas._check_anchor_measurement_ids`
#: exempts it from naming anchor measurement events), and it is where correctness
#: and source-integrity live. §8.9 item 3 preserves it.
PRESERVED_TIERS = ("T0",)

#: §8.4.1. A spike's output is a mechanism verdict; a diagnostic branch is
#: retained for what it explains. Neither accumulates.
EXPERIMENT_KINDS = frozenset({"spike", "diagnostic"})

#: The marker §8.9 item 2 names. It travels in the supersession payload together
#: with BOTH anchor identities, because a supersession that cannot say which
#: denominator replaced which is not a re-anchorable record.
SUPERSEDED_BY_ANCHOR_MOVE = "superseded_by_anchor_move"

REANCHOR_TRIGGER_FREEZE = "freeze"
REANCHOR_TRIGGER_ANCHOR_MOVED = "anchor_moved"
REANCHOR_TRIGGERS = frozenset({REANCHOR_TRIGGER_FREEZE, REANCHOR_TRIGGER_ANCHOR_MOVED})

#: §7.5 `blocking_conditions` entry for a lineage that has been re-anchored and
#: not yet re-measured. The champion is the always-green lineage; a re-anchored
#: one is not green until the combined candidate is rebuilt and re-run.
BLOCKING_REANCHOR_REMEASURE = "REANCHOR_PENDING_REMEASURE"

#: Field and key names that carry a MEASURED QUANTITY. This module may not hold
#: one, read one, or name one — see `audit_no_composed_estimate_arithmetic()`.
#: The list is deliberately wider than the evaluation-event schema: the defect is
#: "a composer that can reach a number", not "a composer that reads one specific
#: key".
_FORBIDDEN_EVIDENCE_KEYS = (
    "performance", "estimate", "raw_samples", "uncertainty", "effect",
    "effect_value", "speedup", "gain", "e_value", "paired_blocks", "rank_key",
    "tokens_per_s", "throughput", "ratio", "multiplier", "percent",
)

#: Arithmetic that could combine two measurements into a third. Absent from this
#: module's syntax, which is what makes "never infer composition by multiplying
#: local speedups" a property of the file rather than of its author's intent.
#: `Add`/`Sub` are deliberately NOT here: `"…" + "; ".join(reasons)` is this
#: module's ordinary idiom, and a guard that forbids its own idiom gets deleted.
#: The consequence is stated plainly in `audit_no_composed_estimate_arithmetic`'s
#: docstring instead of being left for a reader to discover.
_FORBIDDEN_BINOPS = (ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.MatMult)

#: Aggregation this module may not perform. §12's row is literally *"SUMMED local
#: gains inflate readiness"*, and `sum()` needs no `*` — a binop scan alone
#: leaves the named failure reachable.
_FORBIDDEN_AGGREGATORS = frozenset({
    "sum", "fsum", "prod", "mean", "fmean", "median", "average", "geometric_mean",
})


def _iso_now() -> str:
    """Timezone-aware UTC timestamp; schemas.py rejects naive ones on purpose."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _require_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what}: required and non-empty")
    return value


def _require_commit(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.match(value):
        raise ValueError(f"{what}: must be a 40-character lowercase hex sha")
    return value


def _require_candidate_id(value: Any, what: str) -> str:
    _require_text(value, what)
    if not value.startswith("akc-"):
        raise ValueError(f"{what}: {value!r} is not a candidate id (expected 'akc-…')")
    return value


def source_tree_for_backends(backends: Sequence[str]) -> str:
    """The ONE source tree a set of backends belongs to (§1.5, AK-D11).

    `llama_cpu` and `llama_gpu` both answer `llama.cpp`, which is why they share a
    champion. `serving_runtime` answers nothing: §13.5/AK-D23 route it through the
    stack-change gate, so a scheduler candidate has no kernel champion to join and
    is refused here rather than at freeze time.
    """
    if not isinstance(backends, (tuple, list)) or not backends:
        raise ValueError("backends: required, a non-empty sequence of backend names")
    trees = []
    for backend in backends:
        if backend not in schemas.BACKENDS:
            raise ValueError(
                f"backend {backend!r} is not declared {sorted(schemas.BACKENDS)}"
            )
        tree = schemas.SOURCE_TREE_BY_BACKEND.get(backend)
        if tree is None:
            raise IncompatibleMember(
                f"backend {backend!r} has no source tree: it releases through the "
                "three-gate stack-change path (§11.6, AK-D9/AK-D23), so it never "
                "joins a kernel champion"
            )
        if tree not in trees:
            trees.append(tree)
    if len(trees) > 1:
        raise IncompatibleMember(
            f"backends {list(backends)} span source trees {sorted(trees)}; a champion "
            "is one composed lineage per SOURCE TREE (§1.5, AK-D11)"
        )
    return trees[0]


# =============================================================================
# Concept 1 and 3 — frontier candidates and experiments are DIFFERENT TYPES
# =============================================================================

@dataclass(frozen=True)
class FrontierCandidate:
    """A correct, banked candidate that MAY be composed.

    Holds identity, scope and family — never a result. Built only by
    `admit_to_frontier()` from a validated §7.3 record plus the live
    `SurfaceReconciliation`, so `surface_reconciled` is `evaluator/surface`'s
    answer and not a field a caller filled in.
    """

    candidate_id: str
    source_tree: str
    backends: tuple
    mechanism_class: str
    anchor_commit: str
    surface_reconciled: bool
    derived_surface_sha256: str
    traced_surface_sha256: Optional[str]

    def __post_init__(self) -> None:
        _require_candidate_id(self.candidate_id, "FrontierCandidate.candidate_id")
        if self.mechanism_class not in MECHANISM_CLASSES:
            raise ValueError(
                f"mechanism_class {self.mechanism_class!r} is not one of "
                f"{sorted(MECHANISM_CLASSES)}"
            )
        if not isinstance(self.surface_reconciled, bool):
            raise TypeError("surface_reconciled must be a bool")
        _require_commit(self.anchor_commit, "FrontierCandidate.anchor_commit")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "source_tree": self.source_tree,
            "backends": list(self.backends),
            "mechanism_class": self.mechanism_class,
            "anchor_commit": self.anchor_commit,
            "surface_reconciled": self.surface_reconciled,
            "derived_surface_sha256": self.derived_surface_sha256,
            "traced_surface_sha256": self.traced_surface_sha256,
        }


@dataclass(frozen=True)
class Experiment:
    """A spike or diagnostic branch. §8.4.1: it *may never accumulate*.

    Deliberately a DIFFERENT TYPE from `FrontierCandidate`, carrying neither a
    source tree nor a surface: there is nothing on this object a lineage could
    consume, so "never enters the champion" is a property of the type rather than
    a rule someone has to remember. `receipt` is mandatory because *"a refuted
    spike is a first-class result that closes a direction with a receipt"* — a
    closure without one is the §19.3 suppression this project already knows about.
    """

    candidate_id: str
    kind: str
    mechanism_class: str
    receipt: str

    def __post_init__(self) -> None:
        _require_candidate_id(self.candidate_id, "Experiment.candidate_id")
        if self.kind not in EXPERIMENT_KINDS:
            raise ValueError(
                f"kind {self.kind!r} is not one of {sorted(EXPERIMENT_KINDS)}"
            )
        if self.mechanism_class not in MECHANISM_CLASSES:
            raise ValueError(
                f"mechanism_class {self.mechanism_class!r} is not one of "
                f"{sorted(MECHANISM_CLASSES)}"
            )
        _require_text(self.receipt, "Experiment.receipt")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "mechanism_class": self.mechanism_class,
            "receipt": self.receipt,
        }


def _validated_candidate(record: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{what} must be a mapping, got {type(record).__name__}")
    violations = schemas.validate_candidate(record)
    if violations:
        raise CompositionError(
            f"{what} is not a valid {schemas.SCHEMA_CANDIDATE}: " + "; ".join(violations)
        )
    return record


def _bind_reconciliation(record: Mapping[str, Any],
                         reconciliation: Any) -> bool:
    """Bind a candidate record to the LIVE reconciliation and return `reconciled`.

    The record's `affected_surface` block is a projection of a reconciliation
    (`surface.candidate_affected_surface_block`). Accepting the block alone would
    let a caller present a healthy reconciliation for a record that describes a
    different one — the same "declared equals traced" hole invariant 18 exists to
    close, one level up. So both are required and both must agree.
    """
    if not isinstance(reconciliation, surface_mod.SurfaceReconciliation):
        raise TypeError(
            "reconciliation must be an evaluator.surface.SurfaceReconciliation; "
            "reconciliation is §6.4 stage 3's job and is not re-derived here"
        )
    candidate_id = record["candidate_id"]
    if reconciliation.derived.candidate_id != candidate_id:
        raise IncompatibleMember(
            f"reconciliation is for {reconciliation.derived.candidate_id!r} but the "
            f"record is for {candidate_id!r}; reconciling across candidates would "
            "confirm the wrong surface"
        )
    block = surface_mod.candidate_affected_surface_block(reconciliation)
    recorded = record.get("affected_surface")
    if not isinstance(recorded, Mapping):
        raise CompositionError(
            f"{candidate_id}: record carries no affected_surface block"
        )
    for key in ("derived_sha256", "traced_sha256", "reconciled"):
        if recorded.get(key) != block[key]:
            raise CompositionError(
                f"{candidate_id}: record's affected_surface.{key} is "
                f"{recorded.get(key)!r} but the supplied reconciliation says "
                f"{block[key]!r}; the record and the instrument disagree about the "
                "surface, and a disagreement is never resolved in favour of the record"
            )
    return bool(block["reconciled"])


def admit_to_frontier(record: Any, reconciliation: Any, *,
                      mechanism_class: str) -> FrontierCandidate:
    """Admit a BANKED candidate to the frontier (§9.6, §8.9).

    Refuses anything not banked. Banking is decided at T1 by the evaluator's
    disposition and lands in the record; nothing here sets it, so a candidate
    cannot talk its way onto the frontier. An unbanked candidate is not a failure
    of this call — it is an `Experiment`, and `record_experiment()` is its door.

    An UNRECONCILED surface is admitted to the frontier and refused at
    composition: the frontier is "correct and non-dominated", which is a
    different question from "may be combined with something else" (§8.9). The
    refusal is `propose_lineage()`'s, where the rule actually bites.
    """
    record = _validated_candidate(record, "candidate record")
    candidate_id = record["candidate_id"]
    status = record.get("status")
    if status != "banked":
        raise NotBanked(
            f"{candidate_id}: status is {status!r}, not 'banked'. §9.6 makes banking "
            "the evaluator's disposition at T1; only a banked candidate is a frontier "
            "candidate, and a spike is never banked (§8.4.1)"
        )
    reconciled = _bind_reconciliation(record, reconciliation)
    backends = tuple(reconciliation.derived.backends)
    if not backends:
        raise IncompatibleMember(
            f"{candidate_id}: the derived affected surface names no backend, so the "
            "candidate cannot be attributed to a source tree (§1.5)"
        )
    traced = reconciliation.traced
    return FrontierCandidate(
        candidate_id=candidate_id,
        source_tree=source_tree_for_backends(backends),
        backends=backends,
        mechanism_class=mechanism_class,
        anchor_commit=record["ancestry"]["production_base_commit"],
        surface_reconciled=reconciled,
        derived_surface_sha256=reconciliation.derived.sha256(),
        traced_surface_sha256=None if traced is None else traced.sha256(),
    )


def record_experiment(record: Any, *, kind: str, mechanism_class: str,
                      receipt: str) -> Experiment:
    """Record a spike or diagnostic branch (§8.4.1). It never accumulates.

    Refuses a BANKED record: banked and experimental are mutually exclusive by
    §8.4.1's own words, and allowing both would give one candidate two identities
    — one of which composes.
    """
    record = _validated_candidate(record, "candidate record")
    candidate_id = record["candidate_id"]
    if record.get("status") == "banked":
        raise ExperimentMayNotAccumulate(
            f"{candidate_id}: a banked candidate is a frontier candidate, not an "
            "experiment. §8.4.1: a spike 'is never banked, never enters the champion, "
            "and never carries a correctness claim'"
        )
    if record.get("champion_status") not in (None, "none"):
        raise ExperimentMayNotAccumulate(
            f"{candidate_id}: champion_status is "
            f"{record.get('champion_status')!r}; an experiment never enters the champion"
        )
    return Experiment(candidate_id=candidate_id, kind=kind,
                      mechanism_class=mechanism_class, receipt=receipt)


# =============================================================================
# Concept 2 — the champion: ONE composed lineage per source tree
# =============================================================================

@dataclass(frozen=True)
class LineageMember:
    """One member of a composed lineage. Identity, family and scope — no result.

    There is no field here that can hold a rate, an effect, a sample or a
    percentage, and `_assert_no_forbidden_fields()` below fails at import if one
    is added. That is the first of the three locks against §12's *"Summed local
    gains inflate readiness"*: a composer cannot add up numbers it was never
    handed.
    """

    candidate_id: str
    mechanism_class: str
    backends: tuple
    anchor_commit: str
    derived_surface_sha256: str
    traced_surface_sha256: Optional[str]

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "mechanism_class": self.mechanism_class,
            "backends": list(self.backends),
            "anchor_commit": self.anchor_commit,
            "derived_surface_sha256": self.derived_surface_sha256,
            "traced_surface_sha256": self.traced_surface_sha256,
        }


def _assert_no_forbidden_fields(*types: Any) -> None:
    """Import-time guard: no composition type may hold a measured quantity.

    Deliberately raises at import rather than reporting later. The failure it
    prevents — a member carrying its own speedup — is not something to discover on
    the first composition of a campaign that has already been paid for.
    """
    for klass in types:
        for field in dataclasses.fields(klass):
            if field.name in _FORBIDDEN_EVIDENCE_KEYS:
                raise CompositionError(
                    f"{klass.__name__}.{field.name} names a measured quantity. §8.9: "
                    "composition is never inferred by multiplying local results, so no "
                    "composition type may carry one"
                )


_assert_no_forbidden_fields(FrontierCandidate, Experiment, LineageMember)


@dataclass(frozen=True)
class ComposedLineage:
    """A PROPOSED composition. Not a champion: it holds no evidence at all.

    A champion exists only once the combined full candidate has been built and
    re-measured (`compose_champion()`), which is the structural form of §8.9's
    *"rerun T0/T1 on the combined full candidate"*.
    """

    source_tree: str
    anchor_commit: str
    branch: str
    members: tuple

    @property
    def member_ids(self) -> tuple:
        return tuple(m.candidate_id for m in self.members)

    @property
    def backends(self) -> tuple:
        seen: set = set()
        for member in self.members:
            seen.update(member.backends)
        return tuple(sorted(seen))

    @property
    def mechanism_classes(self) -> tuple:
        return tuple(sorted({m.mechanism_class for m in self.members}))

    def to_dict(self) -> dict:
        return {
            "source_tree": self.source_tree,
            "anchor_commit": self.anchor_commit,
            "branch": self.branch,
            "members": [m.to_dict() for m in self.members],
        }


def champion_branch_for(source_tree: str, anchor_commit: str) -> str:
    """`ak/champion/<tree>-<commit[:12]>` — namespaced, and never a production branch.

    Derived rather than supplied so a champion branch cannot be named after the
    frozen branch it is anchored on (invariant 3: no actor modifies a production
    tree, and `schemas.validate_champion` refuses such a name outright).
    """
    _require_text(source_tree, "source_tree")
    _require_commit(anchor_commit, "anchor_commit")
    slug = source_tree.replace(".", "-").replace("/", "-")
    return f"ak/champion/{slug}-{anchor_commit[:12]}"


def propose_lineage(candidates: Iterable[Any], *, anchor_commit: str,
                    source_tree: Optional[str] = None,
                    branch: Optional[str] = None) -> ComposedLineage:
    """Assemble a composable lineage from frontier candidates (§8.9).

    Enforces, in this order:

      * an `Experiment` is refused by name — §8.4.1's *"never enters the
        champion"*;
      * every member's affected-surface map is RECONCILED, per `surface`'s own
        verdict — §8.9's *"only changes with reconciled affected-surface maps may
        be combined"*;
      * every member belongs to the SAME source tree — §1.5: `llama_cpu` and
        `llama_gpu` converge on one llama champion, and `whisper.cpp` never joins
        it;
      * every member is anchored on the SAME production base — invariant 1, and
        the reason a re-anchor cannot quietly keep a stale member.

    An EMPTY lineage is legal: after a freeze in which every member landed in
    production, the champion for that tree is genuinely empty, and refusing to
    express that would force a fictional member.
    """
    members: list = []
    seen: set = set()
    _require_commit(anchor_commit, "anchor_commit")
    for candidate in candidates:
        if isinstance(candidate, Experiment):
            raise ExperimentMayNotAccumulate(
                f"{candidate.candidate_id}: an experiment ({candidate.kind}) may never "
                "accumulate into the champion (§8.4.1); its output is a mechanism "
                "verdict, not a rank"
            )
        if not isinstance(candidate, FrontierCandidate):
            raise TypeError(
                f"lineage members must be FrontierCandidate, got "
                f"{type(candidate).__name__}"
            )
        if candidate.candidate_id in seen:
            raise IncompatibleMember(
                f"{candidate.candidate_id} appears twice in the lineage; a member "
                "counted twice is a composition that does not exist"
            )
        seen.add(candidate.candidate_id)
        if not candidate.surface_reconciled:
            raise UnreconciledSurface(
                f"{candidate.candidate_id}: affected-surface map is not reconciled "
                "(§6.4 stage 3 did not return PASS). §8.9: only changes with "
                "reconciled affected-surface maps may be combined"
            )
        if candidate.anchor_commit != anchor_commit:
            raise IncompatibleMember(
                f"{candidate.candidate_id} is anchored on "
                f"{candidate.anchor_commit[:12]} but the lineage is anchored on "
                f"{anchor_commit[:12]}; invariant 1 anchors every campaign on the "
                "current production tip, so a member at another base has to be "
                "rebased before it can compose"
            )
        members.append(LineageMember(
            candidate_id=candidate.candidate_id,
            mechanism_class=candidate.mechanism_class,
            backends=candidate.backends,
            anchor_commit=candidate.anchor_commit,
            derived_surface_sha256=candidate.derived_surface_sha256,
            traced_surface_sha256=candidate.traced_surface_sha256,
        ))

    # Re-derived from the members' BACKENDS, not copied from the FrontierCandidate's
    # `source_tree` field: the mapping is §1.5's, and asking it again is what makes a
    # whisper member in a llama lineage impossible rather than merely unlikely.
    trees = {source_tree_for_backends(m.backends) for m in members}
    if len(trees) > 1:
        raise IncompatibleMember(
            f"lineage members span source trees {sorted(trees)}; the champion is one "
            "composed lineage per SOURCE TREE (§1.5, AK-D11)"
        )
    if trees:
        derived_tree = trees.pop()
        if source_tree is not None and source_tree != derived_tree:
            raise IncompatibleMember(
                f"source_tree {source_tree!r} contradicts the members' own tree "
                f"{derived_tree!r}"
            )
        source_tree = derived_tree
    if source_tree is None:
        raise ValueError(
            "source_tree: required for an empty lineage — a champion is per source "
            "tree, and an empty lineage still has to say which tree it is the "
            "champion of (§1.5)"
        )
    if source_tree not in schemas.SOURCE_TREES:
        raise ValueError(
            f"source_tree {source_tree!r} is not one of {sorted(schemas.SOURCE_TREES)}"
        )
    if branch is not None:
        # A supplied branch was accepted verbatim, so a `ComposedLineage` could
        # carry the name of the FROZEN production branch and only be caught much
        # later, by `schemas.validate_champion` inside `compose_champion()`.
        # Invariant 3 is about what gets built, and the lineage is what a builder
        # reads; refusing at the point the name enters is the whole difference.
        _require_text(branch, "branch")
        if not branch.startswith("ak/"):
            raise ValueError(
                f"branch {branch!r} must be namespaced under 'ak/'; a champion branch is "
                "never a production branch (invariant 3, schemas.validate_champion)"
            )
    return ComposedLineage(
        source_tree=source_tree,
        anchor_commit=anchor_commit,
        branch=branch if branch is not None
        else champion_branch_for(source_tree, anchor_commit),
        members=tuple(members),
    )


# =============================================================================
# The combined full candidate's OWN evidence — the only evidence there is
# =============================================================================

def _event_anchor_matches(event: Mapping[str, Any],
                          anchor: state_machine.AnchorIdentity) -> tuple:
    """Reasons an event's anchor block is not this champion's anchor. Empty = match."""
    block = event.get("anchor")
    if not isinstance(block, Mapping):
        return ("event carries no anchor block; a run without an explicit anchor is "
                "INVALID, never correct (P-AK-SEARCH-1 precondition 4)",)
    reasons: list = []
    if block.get("source_commit") != anchor.commit:
        reasons.append(
            f"anchor.source_commit {str(block.get('source_commit'))[:12]} is not the "
            f"champion anchor {anchor.commit[:12]}"
        )
    binary = block.get("binary_sha256")
    linkage = block.get("linkage_sha256")
    if binary not in set(anchor.binary_sha256.values()):
        reasons.append(
            "anchor.binary_sha256 is not one of the recorded production binaries"
        )
    elif linkage not in set(anchor.linkage_sha256.values()):
        reasons.append(
            "anchor.linkage_sha256 is not one of the recorded production linkages"
        )
    else:
        # BOTH digests must belong to the SAME backend. Checking the two tables
        # independently accepts a CHIMERA — one backend's binary divided by
        # another backend's linkage — which is a denominator that never existed
        # on any host, assembled out of two that did. `AnchorIdentity` keeps the
        # tables per backend precisely so this pairing is checkable.
        paired = [b for b in sorted(anchor.binary_sha256)
                  if anchor.binary_sha256[b] == binary
                  and anchor.linkage_sha256.get(b) == linkage]
        if not paired:
            reasons.append(
                "anchor.binary_sha256 and anchor.linkage_sha256 are recorded "
                "production digests of DIFFERENT backends; no single production "
                "binary ever had that linkage, so the denominator never existed"
            )
    return tuple(reasons)


def _event_instant(event: Mapping[str, Any]) -> Optional[datetime]:
    """The event's `created_at` as an INSTANT, or None when it cannot be ordered.

    Ordering evidence by the raw string is wrong the moment two records use two
    legal encodings of the same instant: `schemas._need_timestamp` accepts any
    tz-aware ISO-8601, `_iso_now()` here emits `…Z`, and every §7 record in the
    tree emits `…+00:00`. Lexicographically `'+' < '.' < 'Z'`, so a string sort
    puts a `Z` record after a later `+00:00` one and a `+09:00` record after a
    later UTC one — i.e. it silently selects STALE evidence as "the most recent".
    """
    raw = event.get("created_at")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _select_combined_evidence(views: journal.Views, combined_candidate_id: str,
                              tier: str,
                              anchor: state_machine.AnchorIdentity) -> tuple:
    """`(chosen, newest, rejections)` for one tier of the COMBINED candidate's events.

    Every filter here is about the combined artifact: the event's `candidate_id`
    must BE the combined candidate. There is no parameter, and no branch, that
    would accept a member's event instead — which is what makes §8.9's *"never
    infer composition by multiplying local speedups"* structural rather than
    advisory.

    Two results, not one, and that is the point. `chosen` is the most recent
    PASSING event; `newest` is the most recent event of this tier measured
    against this same anchor WHATEVER its status. Returning only `chosen` let a
    re-run that FAILED be answered with the older pass it was re-running: the
    champion is the always-green lineage (§8.9), and "green as of an earlier
    measurement that has since been contradicted" is not green. The caller
    decides what a non-passing `newest` means per tier; this function never
    resolves the disagreement in favour of the pass.
    """
    rejections: list = []
    qualifying: list = []
    considered: list = []
    for event_id, event in views.evaluations.items():
        if event.get("candidate_id") != combined_candidate_id:
            continue
        if event.get("tier") != tier:
            continue
        anchor_reasons = _event_anchor_matches(event, anchor)
        if anchor_reasons:
            rejections.append(f"{event_id}: " + "; ".join(anchor_reasons))
            continue
        instant = _event_instant(event)
        if instant is None:
            rejections.append(
                f"{event_id}: created_at {event.get('created_at')!r} is not an "
                "orderable tz-aware timestamp, so this event cannot be placed "
                "against the others"
            )
            continue
        status = event.get("status")
        # A BASELINE cell measures the ANCHOR — it is the denominator, never a
        # candidate's own evidence. Selecting one as the champion's T0/T1 would
        # make the composition its own baseline (invariant 15: baseline and
        # off-recipe cells are diagnostic and never justify a release).
        claim = event.get("claim_grammar")
        category = claim.get("category") if isinstance(claim, Mapping) else None
        if category == "BASELINE":
            rejections.append(
                f"{event_id}: claim_grammar.category is 'BASELINE' — a baseline "
                "cell is the denominator, not the composition's own evidence"
            )
            continue
        considered.append((instant, event_id, status))
        if status != "pass":
            rejections.append(f"{event_id}: status is {status!r}, not 'pass'")
            continue
        if tier in COMPARISON_TIERS:
            # A rate tier's ratio must resolve to a real anchor measurement in
            # THIS journal. T0 is exempt by the contract's own carve-out: it
            # compares artifacts, not rates.
            binding = schemas.check_anchor_binding(
                event, lambda eid: views.evaluations.get(eid)
            )
            if binding.outcome != schemas.PASS:
                rejections.append(
                    f"{event_id}: anchor binding is {binding.outcome} — "
                    + "; ".join(binding.reasons)
                )
                continue
        qualifying.append((instant, event_id))
    considered.sort(key=lambda item: (item[0], item[1]))
    newest = None if not considered else (considered[-1][1], considered[-1][2])
    if not qualifying:
        return None, newest, tuple(rejections)
    qualifying.sort(key=lambda item: (item[0], item[1]))
    return qualifying[-1][1], newest, tuple(rejections)


def _readiness_block(*, combined_candidate_id: str, backends: Sequence[str],
                     views: journal.Views,
                     readiness_by_backend: Optional[Mapping[str, Any]],
                     evidence: Mapping[str, Optional[str]],
                     anchor_commit: str) -> dict:
    """§7.5 `readiness`. Cells are CITATIONS, never numbers computed here.

    Invariant 14 gives the readiness magnitude to a deterministic reducer over
    journaled records; this module's job is to make sure whatever that reducer
    supplies is bound to the COMBINED candidate's own events. So every cell it
    offers must carry `event_ids`, and every one of those must resolve, in this
    journal, to an evaluation event whose `candidate_id` is the combined
    candidate. A narrated cell has no such citation and is refused.

    `reference_signal` is always RENDERED here and never accepted, so the one
    free-text field on the record cannot carry an estimated percentage.
    """
    known = set(backends)
    by_backend: dict = {}
    # A list rather than a counter: `cited = cited + 1` was the module's only
    # numeric accumulation, and counting by collecting keeps the module free of
    # accumulation syntax without pretending that freedom is a semantic property.
    cited: list = []
    if readiness_by_backend is None:
        for backend in sorted(known):
            phases = schemas.PHASES_BY_BACKEND.get(backend)
            if phases is None:
                by_backend[backend] = {"phase_vocabulary": "undeclared", "phases": {}}
                continue
            by_backend[backend] = {
                "phase_vocabulary": "declared",
                "phases": {phase: {"event_ids": [], "measured": False}
                           for phase in sorted(phases)},
            }
    else:
        if not isinstance(readiness_by_backend, Mapping):
            raise TypeError("readiness_by_backend must be a mapping or None")
        for backend, cells in readiness_by_backend.items():
            if backend not in known:
                raise IncompatibleMember(
                    f"readiness names backend {backend!r}, which the composed "
                    f"candidate's affected surface does not reach {sorted(known)}"
                )
            if not isinstance(cells, Mapping):
                raise TypeError(f"readiness_by_backend[{backend!r}] must be a mapping")
            copied: dict = {}
            for cell_name, cell in cells.items():
                if not isinstance(cell, Mapping):
                    raise TypeError(
                        f"readiness_by_backend[{backend!r}][{cell_name!r}] must be a "
                        "mapping carrying its event_ids"
                    )
                event_ids = cell.get("event_ids")
                if not isinstance(event_ids, (list, tuple)) or not event_ids:
                    raise CompositionEvidenceMissing(
                        f"readiness_by_backend[{backend!r}][{cell_name!r}]: a readiness "
                        "cell must cite at least one evaluation event of the combined "
                        "candidate. Invariant 14: readiness is computed from records, "
                        "never narrated"
                    )
                for event_id in event_ids:
                    resolved = views.evaluations.get(event_id)
                    if resolved is None:
                        raise CompositionEvidenceMissing(
                            f"readiness cell {backend}/{cell_name} cites {event_id!r}, "
                            "which does not resolve in this journal"
                        )
                    if resolved.get("candidate_id") != combined_candidate_id:
                        raise CompositionEvidenceMissing(
                            f"readiness cell {backend}/{cell_name} cites {event_id!r}, "
                            f"an event of {resolved.get('candidate_id')!r}, not of the "
                            f"combined candidate {combined_candidate_id!r}. §8.9: the "
                            "composition is re-measured as a whole and readiness never "
                            "aggregates member results"
                        )
                    cited.append(event_id)
                copied[cell_name] = dict(cell)
            by_backend[backend] = copied

    held = [f"{tier} {event_id}" for tier, event_id in sorted(evidence.items())
            if event_id is not None]
    signal = (
        f"combined candidate {combined_candidate_id} re-measured as a whole versus "
        f"anchor {anchor_commit[:12]}; evidence {', '.join(held)}; "
        f"{len(cited)} readiness citation(s), all bound to that candidate's own events. "
        "No member result contributes and no cross-backend roll-up is formed "
        "(§8.9; P-AK-SEARCH-1 denial 9)."
    )
    return {"by_backend": by_backend, "reference_signal": signal}


def compose_champion(lineage: ComposedLineage, *,
                     combined_candidate_id: str,
                     combined_reconciliation: Any,
                     views: journal.Views,
                     recorded_anchor: state_machine.AnchorIdentity,
                     observed_anchor: Optional[state_machine.AnchorIdentity],
                     storage_gb: float,
                     readiness_by_backend: Optional[Mapping[str, Any]] = None,
                     blocking_conditions: Sequence[str] = (),
                     created_at: Optional[str] = None) -> dict:
    """Turn a proposed lineage into a §7.5 champion record. Deterministic.

    The gate order is deliberate:

      1. **anchor identity first** (§8.9, AK-D22). A composition measured against
         a denominator that no longer exists is worse than no composition, so no
         other check runs until the anchor is proved to be where BOOTSTRAP left
         it. An unverifiable anchor raises `AnchorUncheckable` — a fail-open
         anchor check has the same shape as no check;
      2. the combined artifact really is the composition (its affected surface
         reaches every backend the members reach, and it is not itself a member);
      3. the combined candidate's OWN T0 and T1 both passed against that anchor
         (§8.9). No member evidence is consulted, because none is accepted.

    Returns the record; it does not write it. `record_champion()` writes.
    """
    if not isinstance(lineage, ComposedLineage):
        raise TypeError(f"lineage must be a ComposedLineage, got {type(lineage).__name__}")
    if not isinstance(views, journal.Views):
        raise TypeError(f"views must be a journal.Views, got {type(views).__name__}")
    if not isinstance(recorded_anchor, state_machine.AnchorIdentity):
        raise TypeError("recorded_anchor must be a state_machine.AnchorIdentity")

    # ---- 1. anchor identity -------------------------------------------------
    anchor_check = state_machine.check_anchor_identity(recorded_anchor, observed_anchor)
    if anchor_check.outcome == schemas.FAIL:
        raise AnchorMovedRefusal(
            "anchor identity moved; no new candidate work for "
            f"{recorded_anchor.source_tree} (§8.9 item 1): "
            + "; ".join(anchor_check.reasons)
        )
    if anchor_check.outcome != schemas.PASS:
        raise state_machine.AnchorUncheckable(
            "anchor identity could not be verified at this composition boundary; not "
            "observing is not evidence that production stayed put (§8.9): "
            + "; ".join(anchor_check.reasons)
        )
    if recorded_anchor.source_tree != lineage.source_tree:
        raise IncompatibleMember(
            f"anchor describes {recorded_anchor.source_tree!r} but the lineage is for "
            f"{lineage.source_tree!r}"
        )
    if recorded_anchor.commit != lineage.anchor_commit:
        raise IncompatibleMember(
            f"anchor commit {recorded_anchor.commit[:12]} is not the lineage's base "
            f"{lineage.anchor_commit[:12]} (invariant 1)"
        )

    # ---- 2. the combined artifact ------------------------------------------
    _require_candidate_id(combined_candidate_id, "combined_candidate_id")
    if combined_candidate_id in lineage.member_ids:
        raise IncompatibleMember(
            f"{combined_candidate_id} is both a member and the composition; the "
            "combined full candidate is a NEW artifact built from the members (§8.9, "
            "invariant 2: no promotion-time cherry-pick reconciliation)"
        )
    combined_record = views.candidates.get(combined_candidate_id)
    if combined_record is None:
        raise CompositionEvidenceMissing(
            f"{combined_candidate_id} has no candidate record in this journal; there "
            "is nothing to prove the composition was ever built"
        )
    combined_record = _validated_candidate(combined_record, "combined candidate record")
    if combined_record.get("status") != "banked":
        raise NotBanked(
            f"{combined_candidate_id}: the combined full candidate has status "
            f"{combined_record.get('status')!r}, not 'banked'"
        )
    reconciled = _bind_reconciliation(combined_record, combined_reconciliation)
    if not reconciled:
        raise UnreconciledSurface(
            f"{combined_candidate_id}: the combined full candidate's affected-surface "
            "map is not reconciled; the composition owes the same §6.4 proof its "
            "members owe"
        )
    if combined_reconciliation.hard_failure:
        raise UnreconciledSurface(
            f"{combined_candidate_id}: `traced ⊄ derived` on the combined candidate is "
            "a hard candidate failure (invariant 18)"
        )
    combined_backends = tuple(combined_reconciliation.derived.backends)
    combined_tree = source_tree_for_backends(combined_backends)
    if combined_tree != lineage.source_tree:
        raise IncompatibleMember(
            f"{combined_candidate_id} affects tree {combined_tree!r}, not the "
            f"lineage's {lineage.source_tree!r}"
        )
    missing_backends = sorted(set(lineage.backends) - set(combined_backends))
    if missing_backends:
        raise IncompatibleMember(
            f"{combined_candidate_id} does not reach {missing_backends}, which its "
            "members do; a composition that leaves a member's backend untouched is "
            "not the composition of those members (invariant 2)"
        )
    ancestry = combined_record.get("ancestry")
    if not isinstance(ancestry, Mapping) or \
            ancestry.get("production_base_commit") != lineage.anchor_commit:
        raise IncompatibleMember(
            f"{combined_candidate_id} is based on "
            f"{str(ancestry.get('production_base_commit'))[:12] if isinstance(ancestry, Mapping) else None} "
            f"but the lineage is anchored on {lineage.anchor_commit[:12]} (invariant 1)"
        )

    # Re-checked FROM THE JOURNAL, not from the in-memory lineage. `ComposedLineage`
    # is a public type, so `propose_lineage()`'s gates can be walked around by
    # constructing one directly; the member's own durable record cannot be. §8.9's
    # "only changes with reconciled affected-surface maps may be combined" therefore
    # holds on the only path that produces a champion.
    for member_id in lineage.member_ids:
        member_record = views.candidates.get(member_id)
        if member_record is None:
            raise CompositionEvidenceMissing(
                f"member {member_id} has no candidate record in this journal; a member "
                "that is not a record is not evidence (invariant 7)"
            )
        member_surface = member_record.get("affected_surface")
        if not isinstance(member_surface, Mapping) or \
                member_surface.get("reconciled") is not True:
            raise UnreconciledSurface(
                f"member {member_id}'s own record does not declare a reconciled "
                "affected-surface map; §8.9 admits only reconciled changes to a "
                "composition"
            )
        if member_record.get("status") != "banked":
            raise NotBanked(
                f"member {member_id} has status {member_record.get('status')!r} in the "
                "journal, not 'banked'; only a banked candidate composes (§9.6)"
            )

    # ---- 3. the combined candidate's own T0 and T1 -------------------------
    evidence: dict = {}
    status_of: dict = {}
    for tier in REQUIRED_COMBINED_TIERS:
        chosen, newest, rejections = _select_combined_evidence(
            views, combined_candidate_id, tier, recorded_anchor)
        if newest is not None and newest[1] != "pass":
            # The most recent measurement of this tier CONTRADICTS the pass that
            # would otherwise have been selected. Answering a failed re-run with
            # the older pass it was re-running is how a champion stays green on
            # paper (§8.9: the champion is the always-green lineage).
            raise CompositionEvidenceMissing(
                f"{combined_candidate_id}'s most recent {tier} against anchor "
                f"{recorded_anchor.commit[:12]} is {newest[0]} with status "
                f"{newest[1]!r}, not 'pass'. An earlier passing {tier} does not "
                "survive a later contradicting one; re-measure the combined full "
                "candidate (§8.9)"
            )
        if chosen is None:
            detail = "; ".join(rejections) if rejections else "no event of this tier"
            raise CompositionEvidenceMissing(
                f"{combined_candidate_id} has no passing {tier} of its own against "
                f"anchor {recorded_anchor.commit[:12]}: {detail}. §8.9: after combining, "
                f"rerun T0/T1 on the combined full candidate — composition is never "
                "inferred from the members' results"
            )
        evidence[tier] = chosen
        status_of[tier] = "pass"
    for tier in OPTIONAL_COMBINED_TIERS:
        chosen, newest, _ = _select_combined_evidence(
            views, combined_candidate_id, tier, recorded_anchor)
        if newest is not None and newest[1] != "pass":
            # §9.7's T2 is never REQUIRED, but a T2 that FAILED is the interaction
            # effect a composition exists to expose. Recording `null` here would
            # make "the composition failed its interaction check" indistinguishable
            # from "no T2 was ever run". It is carried with its real status, which
            # makes `schemas.validate_champion` demand a blocking condition.
            evidence[tier] = newest[0]
            status_of[tier] = newest[1]
            continue
        evidence[tier] = chosen
        status_of[tier] = "pass"

    # ---- 4. the record ------------------------------------------------------
    union_input = {
        "combined": combined_reconciliation.derived.sha256(),
        "combined_traced": (None if combined_reconciliation.traced is None
                            else combined_reconciliation.traced.sha256()),
        "members": sorted(
            [m.derived_surface_sha256 for m in lineage.members]
            + [m.traced_surface_sha256 for m in lineage.members
               if m.traced_surface_sha256 is not None]
        ),
    }
    record = {
        "schema": schemas.SCHEMA_CHAMPION,
        "source_tree": lineage.source_tree,
        "anchor_commit": lineage.anchor_commit,
        "branch": lineage.branch,
        "member_candidates": list(lineage.member_ids),
        "combined_candidate_id": combined_candidate_id,
        "last_t0": {"event_id": evidence["T0"], "status": status_of["T0"]},
        "last_t1": {"event_id": evidence["T1"], "status": status_of["T1"]},
        "last_t2": (None if evidence["T2"] is None
                    else {"event_id": evidence["T2"], "status": status_of["T2"]}),
        "readiness": _readiness_block(
            combined_candidate_id=combined_candidate_id,
            backends=combined_backends,
            views=views,
            readiness_by_backend=readiness_by_backend,
            evidence=evidence,
            anchor_commit=lineage.anchor_commit,
        ),
        "affected_surface_union_sha256": schemas.content_hash(union_input),
        "storage_gb": storage_gb,
        "blocking_conditions": list(blocking_conditions),
        "created_at": created_at if created_at is not None else _iso_now(),
    }
    violations = schemas.validate_champion(record)
    if violations:
        raise CompositionError(
            "composed champion is not a valid record: " + "; ".join(violations)
        )
    return record


def record_champion(journal_obj: journal.Journal,
                    champion_record: Mapping[str, Any]) -> journal.JournalEntry:
    """Append the champion to the journal. `Journal.append()` re-validates.

    A separate call from `compose_champion()` on purpose: composing is a pure
    derivation the caller may inspect and refuse, and writing is the durable act.
    Folding them would make a rejected composition unrepresentable.
    """
    if not isinstance(journal_obj, journal.Journal):
        raise TypeError("journal_obj must be a journal.Journal")
    return journal_obj.append(journal.KIND_CHAMPION_UPDATED, champion_record)


# =============================================================================
# Diversity across mechanism classes (§8.9)
# =============================================================================

def retain_frontier(candidates: Sequence[Any], *, capacity: int,
                    min_per_class: int = 1) -> tuple:
    """Retain at most `capacity`, keeping `min_per_class` of EVERY class first.

    §8.9: *"Retain diversity across mechanism classes so one noisy early win does
    not collapse the search to a single family."* The order handed in is the
    caller's preference order (produced by `evaluator.api.rank_candidates`, which
    refuses a rank to any candidate that did not earn one) — composition does not
    rank and does not re-order. What it does is fill the per-class quota BEFORE it
    fills by preference, so a family that swept the top of the list cannot take
    the last slot of another.

    A capacity that cannot hold the floor is `DiversityFloorUnmet`, never a quiet
    truncation: silently dropping a class is exactly the collapse this exists to
    prevent, and a caller that has to choose deserves to be told.
    """
    ordered = tuple(candidates)
    seen: set = set()
    for candidate in ordered:
        if not isinstance(candidate, FrontierCandidate):
            raise TypeError(
                f"retain_frontier takes FrontierCandidate, got "
                f"{type(candidate).__name__}; experiments never accumulate (§8.4.1)"
            )
        if candidate.candidate_id in seen:
            # Retention counts ARTIFACTS. `mechanism_class` is supplied to
            # `admit_to_frontier()`, so one banked candidate can be admitted twice
            # under two class labels; without this, that one artifact fills two
            # class quotas — the collapse the floor exists to prevent, wearing two
            # hats — and the returned tuple (filtered by id) exceeds `capacity`.
            raise IncompatibleMember(
                f"{candidate.candidate_id} appears twice on the frontier, under "
                f"mechanism classes that include {candidate.mechanism_class!r}; one "
                "artifact is one representative, and diversity counted over relabelled "
                "copies of the same candidate is not diversity (§8.9)"
            )
        seen.add(candidate.candidate_id)
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an int")
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    if isinstance(min_per_class, bool) or not isinstance(min_per_class, int):
        raise TypeError("min_per_class must be an int")
    if min_per_class < 1:
        raise ValueError("min_per_class must be at least 1")

    classes: list = []
    for candidate in ordered:
        if candidate.mechanism_class not in classes:
            classes.append(candidate.mechanism_class)
    # The quota, enumerated slot-major (one per class, then a second per class,
    # …). Enumerating it is also how the floor's SIZE is obtained: written as
    # `sum(min_per_class for _ in classes)` it was a multiplication in disguise,
    # contorted to satisfy this module's own AST audit — which is how a syntactic
    # guard starts shaping code instead of describing it.
    quota = [(slot, name) for slot in range(min_per_class) for name in classes]
    required = len(quota)
    if capacity < required:
        raise DiversityFloorUnmet(
            f"capacity {capacity} cannot hold {min_per_class} representative(s) of each "
            f"of {len(classes)} mechanism class(es) {classes}; retaining fewer would "
            "collapse the search to a single family (§8.9)"
        )
    if len(ordered) <= capacity:
        return ordered

    by_class = {
        name: [c for c in ordered if c.mechanism_class == name] for name in classes
    }
    keep: set = set()
    for slot, name in quota:
        bucket = by_class[name]
        if slot < len(bucket):
            keep.add(bucket[slot].candidate_id)
    for candidate in ordered:
        if len(keep) >= capacity:
            break
        keep.add(candidate.candidate_id)
    return tuple(c for c in ordered if c.candidate_id in keep)


def check_mechanism_diversity(candidates: Sequence[Any], *, min_classes: int,
                              available_classes: Optional[Sequence[str]] = None
                              ) -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK on "has the search collapsed to one family?".

    `available_classes` — the classes that actually produced a banked candidate —
    is what turns a shortfall into a FINDING. Without it, a frontier holding one
    class may mean retention collapsed, or it may mean one class is all the
    campaign has produced so far, and those are different facts. Absence of the
    comparison is COULD_NOT_CHECK, never a confident FAIL and never a soft pass.
    """
    if isinstance(min_classes, bool) or not isinstance(min_classes, int):
        raise TypeError("min_classes must be an int")
    if min_classes < 1:
        raise ValueError("min_classes must be at least 1")
    held: list = []
    for candidate in candidates:
        if not isinstance(candidate, FrontierCandidate):
            raise TypeError(
                f"check_mechanism_diversity takes FrontierCandidate, got "
                f"{type(candidate).__name__}"
            )
        if candidate.mechanism_class not in held:
            held.append(candidate.mechanism_class)
    if not held:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the frontier is empty; there is no retention to evaluate",
        ))
    if len(held) >= min_classes:
        return schemas.Check(schemas.PASS)
    if available_classes is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the frontier holds {len(held)} mechanism class(es) {sorted(held)} against "
            f"a floor of {min_classes}, but the classes that produced a banked "
            "candidate were not supplied; a shortfall could be retention collapsing or "
            "the campaign having produced nothing else, and those are different facts",
        ))
    available = sorted({c for c in available_classes})
    if not available:
        # `reachable` would be 0 and EVERY frontier would clear a floor of zero.
        # That is the check passing because the thing it inspects was deleted: an
        # empty producing set also CONTRADICTS a non-empty frontier, since a
        # frontier candidate is by definition a banked one (§9.6).
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the frontier holds {sorted(held)} but no mechanism class was named as "
            "having produced a banked candidate; an empty producing set contradicts a "
            "non-empty frontier, and comparing against it would clear any floor",
        ))
    unaccounted = sorted({c for c in held if c not in set(available)})
    if unaccounted:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the frontier holds {unaccounted}, which the supplied producing classes "
            f"{available} do not include; a frontier candidate is banked by definition, "
            "so the two inputs describe different campaigns and neither is trusted over "
            "the other",
        ))
    reachable = min(min_classes, len(available))
    if len(held) >= reachable:
        return schemas.Check(schemas.PASS)
    return schemas.Check(schemas.FAIL, (
        f"the frontier holds {sorted(held)} while {available} produced banked "
        f"candidates; retaining {len(held)} of a reachable {reachable} class(es) "
        "collapses the search to a single family (§8.9)",
    ))


# =============================================================================
# Re-anchoring (§8.9) — at a freeze, and at an unexpected anchor move
# =============================================================================

@dataclass(frozen=True)
class ReanchorPlan:
    """§8.9's re-anchor, as a plan the caller must execute — not as a fait accompli.

    *"Members already in production are dropped, the remainder is rebased on the
    new tip, and its T1/T2 evidence is invalidated and re-measured."* The
    invalidation is easy to record and easy to forget to act on, so this type
    cannot produce a champion record for a non-empty lineage AT ALL: the only
    route back to a champion is `propose_lineage()` over rebased candidates plus
    `compose_champion()` over a rebuilt combined candidate, which demands fresh
    T0/T1 by construction.

    `rebase_sources` names the members to rebase, not rebased members: a rebased
    change is a NEW candidate with a new source commit, a new build and new
    hashes, and pretending otherwise is how a stale member survives a freeze.
    """

    source_tree: str
    trigger: str
    old_anchor_commit: str
    new_anchor_commit: str
    new_branch: str
    dropped_members: tuple
    rebase_sources: tuple
    invalidated_comparison_event_ids: tuple
    invalidated_artifact_event_ids: tuple
    requires_remeasure_tiers: tuple
    preserved_candidate_ids: tuple

    @property
    def is_empty(self) -> bool:
        """True when every member landed in production and nothing needs rebasing."""
        return not self.rebase_sources

    def to_champion_record(self, *, storage_gb: float = 0.0,
                           blocking_conditions: Sequence[str] = (),
                           created_at: Optional[str] = None) -> dict:
        """The re-anchored champion — ONLY when the lineage came out empty.

        An empty champion at the new tip is a real, recordable state: every member
        shipped, so the tree's champion is production itself. A NON-empty one is
        refused here, which is the enforcement of *"T1/T2 evidence is invalidated
        and re-measured"*: there is no way to carry the old verdicts across.
        """
        if not self.is_empty:
            raise CompositionEvidenceMissing(
                f"{len(self.rebase_sources)} member(s) still need rebasing on "
                f"{self.new_anchor_commit[:12]}; their T1/T2 evidence was invalidated by "
                "the re-anchor and must be RE-MEASURED on a rebuilt combined candidate "
                "(§8.9). Rebase, then propose_lineage() + compose_champion()"
            )
        record = {
            "schema": schemas.SCHEMA_CHAMPION,
            "source_tree": self.source_tree,
            "anchor_commit": self.new_anchor_commit,
            "branch": self.new_branch,
            "member_candidates": [],
            "combined_candidate_id": None,
            "last_t0": None,
            "last_t1": None,
            "last_t2": None,
            "readiness": {
                "by_backend": {},
                "reference_signal": (
                    f"re-anchored on {self.new_anchor_commit[:12]} ({self.trigger}); "
                    f"{len(self.dropped_members)} member(s) already in production were "
                    "dropped and the lineage is empty, so the champion is the "
                    "production tip itself (§8.9)"
                ),
            },
            "affected_surface_union_sha256": schemas.content_hash([]),
            "storage_gb": storage_gb,
            "blocking_conditions": list(blocking_conditions),
            "created_at": created_at if created_at is not None else _iso_now(),
        }
        violations = schemas.validate_champion(record)
        if violations:
            raise CompositionError(
                "re-anchored champion is not a valid record: " + "; ".join(violations)
            )
        return record

    def to_dict(self) -> dict:
        return {
            "source_tree": self.source_tree,
            "trigger": self.trigger,
            "old_anchor_commit": self.old_anchor_commit,
            "new_anchor_commit": self.new_anchor_commit,
            "new_branch": self.new_branch,
            "dropped_members": list(self.dropped_members),
            "rebase_sources": list(self.rebase_sources),
            "invalidated_comparison_event_ids": list(self.invalidated_comparison_event_ids),
            "invalidated_artifact_event_ids": list(self.invalidated_artifact_event_ids),
            "requires_remeasure_tiers": list(self.requires_remeasure_tiers),
            "preserved_candidate_ids": list(self.preserved_candidate_ids),
        }


def plan_reanchor(champion_record: Any, *,
                  new_anchor: state_machine.AnchorIdentity,
                  members_in_production: Sequence[str] = (),
                  trigger: str = REANCHOR_TRIGGER_FREEZE) -> ReanchorPlan:
    """§8.9's re-anchor, from the champion record and the new production identity."""
    if trigger not in REANCHOR_TRIGGERS:
        raise ValueError(f"trigger {trigger!r} is not one of {sorted(REANCHOR_TRIGGERS)}")
    if not isinstance(new_anchor, state_machine.AnchorIdentity):
        raise TypeError("new_anchor must be a state_machine.AnchorIdentity")
    if not isinstance(champion_record, Mapping):
        raise TypeError("champion_record must be a mapping")
    violations = schemas.validate_champion(champion_record)
    if violations:
        raise ReanchorRefused(
            "champion_record is not a valid champion: " + "; ".join(violations)
        )
    source_tree = champion_record["source_tree"]
    if new_anchor.source_tree != source_tree:
        raise ReanchorRefused(
            f"new anchor describes {new_anchor.source_tree!r} but the champion is for "
            f"{source_tree!r}; freezes are per source tree (§1.5)"
        )
    old_commit = champion_record["anchor_commit"]
    if new_anchor.commit == old_commit:
        raise ReanchorRefused(
            f"new anchor commit {new_anchor.commit[:12]} equals the champion's current "
            "anchor; a re-anchor that does not move the base invalidates evidence for "
            "nothing"
        )

    members = list(champion_record["member_candidates"])
    in_production = list(members_in_production)
    unknown = [c for c in in_production if c not in members]
    if unknown:
        raise ReanchorRefused(
            f"{unknown} are named as already in production but are not members of this "
            "champion; a member list that does not match the lineage would drop the "
            "wrong work"
        )
    dropped = tuple(c for c in members if c in set(in_production))
    remainder = tuple(c for c in members if c not in set(in_production))

    comparisons: list = []
    artifacts: list = []
    for key, sink in (("last_t1", comparisons), ("last_t2", comparisons),
                      ("last_t0", artifacts)):
        block = champion_record.get(key)
        if isinstance(block, Mapping) and isinstance(block.get("event_id"), str):
            sink.append(block["event_id"])

    tiers = list(REQUIRED_COMBINED_TIERS)
    if isinstance(champion_record.get("last_t2"), Mapping):
        tiers.append("T2")

    return ReanchorPlan(
        source_tree=source_tree,
        trigger=trigger,
        old_anchor_commit=old_commit,
        new_anchor_commit=new_anchor.commit,
        new_branch=champion_branch_for(source_tree, new_anchor.commit),
        dropped_members=dropped,
        rebase_sources=remainder,
        invalidated_comparison_event_ids=tuple(comparisons),
        invalidated_artifact_event_ids=tuple(artifacts),
        requires_remeasure_tiers=tuple(tiers),
        # §8.9 item 3: source, patches and correctness results survive a re-anchor.
        # Every member record is preserved, dropped ones included — a member that
        # shipped is still the record of how it was built.
        preserved_candidate_ids=tuple(members),
    )


# =============================================================================
# ANCHOR_MOVED (§8.9 items 1-5, AK-D22, §12)
# =============================================================================

def affected_backends_for_move(recorded: state_machine.AnchorIdentity,
                               observed: state_machine.AnchorIdentity) -> tuple:
    """Which backends an anchor move reaches. Fail-closed and tree-wide by default.

    A branch or commit move is a property of the TREE, so it reaches every backend
    the tree serves — CPU and GPU together, per §1.5. Only a pure digest
    difference can be narrower. A backend that was not observed at all counts as
    affected: an unobserved backend is not an unchanged one.
    """
    if not isinstance(recorded, state_machine.AnchorIdentity):
        raise TypeError("recorded must be a state_machine.AnchorIdentity")
    if not isinstance(observed, state_machine.AnchorIdentity):
        raise TypeError("observed must be a state_machine.AnchorIdentity")
    if (recorded.source_tree != observed.source_tree
            or recorded.branch != observed.branch
            or recorded.commit != observed.commit):
        return recorded.backends
    affected: set = set()
    for name in ("binary_sha256", "linkage_sha256"):
        was = getattr(recorded, name)
        now = getattr(observed, name)
        for backend, digest in was.items():
            if backend not in now or now[backend] != digest:
                affected.add(backend)
    for backend in set(observed.binary_sha256) - set(recorded.binary_sha256):
        affected.add(backend)
    return tuple(sorted(affected))


@dataclass(frozen=True)
class AnchorMoveSweep:
    """§8.9 item 2 and item 3, as two explicit sets rather than one sweep.

    Item 2 marks comparisons `superseded_by_anchor_move` carrying both anchor
    identities. Item 3 PRESERVES source, patches and correctness results, *"which
    remain valid — only the comparisons died, not the work"*. Naming only the
    first set would leave "what survives" to whoever reads the code next, which is
    how a preserved record gets swept the second time someone touches this.

    The sets hold JOURNAL ENTRY ids (`akj-…`), not the payloads' own `event_id`s
    (`ake-…`): `journal.append_superseded` resolves `target_event_id` against
    `JournalEntry.event_id`, and `rebuild_views` excludes by the same key. The
    record ids are carried alongside because a human reading the stop package
    needs to see which measurements died, and the two vocabularies must not be
    confused at a boundary where a wrong id is an unfixable dangling reference.
    """

    source_tree: str
    old_anchor: Mapping[str, Any]
    new_anchor: Mapping[str, Any]
    affected_backends: tuple
    reason: str
    superseded_entry_ids: tuple
    superseded_record_ids: tuple
    preserved_entry_ids: tuple
    preserved_record_ids: tuple
    preserved_candidate_ids: tuple
    entry_kind: Mapping[str, str]
    entry_tier: Mapping[str, Optional[str]]
    entry_record_id: Mapping[str, Optional[str]]
    fail_closed_candidates: tuple

    def payload_for(self, entry_id: str) -> dict:
        """The `SUPERSEDED` payload. Carries BOTH identities, per §8.9 item 2."""
        if entry_id not in self.superseded_entry_ids:
            raise SupersessionScopeViolation(
                f"{entry_id} is not in this sweep's superseded set"
            )
        return {
            "target_event_id": entry_id,
            "reason": self.reason,
            "superseded_by": None,
            SUPERSEDED_BY_ANCHOR_MOVE: True,
            "source_tree": self.source_tree,
            "old_anchor": dict(self.old_anchor),
            "new_anchor": dict(self.new_anchor),
            "affected_backends": list(self.affected_backends),
            "tier": self.entry_tier[entry_id],
            "record_id": self.entry_record_id[entry_id],
            # A COUNT, not the list. The preserved set is a property of the SWEEP
            # and is journaled once, in the ANCHOR_MOVED stop detail; copying it
            # into each of N supersession payloads makes the log grow with
            # (comparisons superseded x candidates ever recorded), so a campaign
            # long enough to need re-anchoring twice pays for it quadratically.
            # An append-only log is the one place that cost is never reclaimed.
            "preserved_candidate_count": len(self.preserved_candidate_ids),
        }

    def to_dict(self) -> dict:
        return {
            "source_tree": self.source_tree,
            "old_anchor": dict(self.old_anchor),
            "new_anchor": dict(self.new_anchor),
            "affected_backends": list(self.affected_backends),
            "reason": self.reason,
            "superseded_entry_ids": list(self.superseded_entry_ids),
            "superseded_record_ids": list(self.superseded_record_ids),
            "preserved_entry_ids": list(self.preserved_entry_ids),
            "preserved_record_ids": list(self.preserved_record_ids),
            "preserved_candidate_ids": list(self.preserved_candidate_ids),
            "fail_closed_candidates": list(self.fail_closed_candidates),
        }


def _declared_backends(raw: Any) -> Optional[list]:
    """A candidate's declared backends, or None meaning "no usable declaration".

    None is the FAIL-CLOSED answer and every unusable shape returns it. Without
    this, `{"akc-…": []}` and `{"akc-…": "llama_gpu"}` (a bare string, whose set
    is a set of CHARACTERS) both intersect nothing, so they NARROW the sweep to
    zero silently — the exact opposite of the widening the caller was promised,
    and the failure mode is a live ratio left pointing at a dead denominator.
    """
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return None
    values = list(raw)
    named = [b for b in values if isinstance(b, str) and b in schemas.BACKENDS]
    if not named or len(named) != len(values):
        return None
    return named


def plan_anchor_move_supersession(entries: Sequence[Any], *,
                                  old_anchor: state_machine.AnchorIdentity,
                                  new_anchor: state_machine.AnchorIdentity,
                                  affected_backends: Sequence[str],
                                  backends_by_candidate: Optional[Mapping[str, Sequence[str]]] = None
                                  ) -> AnchorMoveSweep:
    """Partition the journal's entries into "comparison died" and "work survives".

    Takes ENTRIES rather than `Views` because supersession targets a journal entry
    id, and `Views` is keyed by record id — planning a sweep from a view would
    produce targets that `Journal.append()` refuses as dangling, which is the
    friendlier half of the failure. The unfriendly half is a sweep that names an
    id belonging to something else.

    An evaluation event does not carry its backend, so attribution comes from
    `backends_by_candidate` — the affected-surface manifests the caller already
    holds. A candidate MISSING from that map is treated as AFFECTED and named in
    `fail_closed_candidates`: under-sweeping leaves a live ratio with a dead
    denominator, which is the failure §8.9 exists to close, so the widening goes
    the safe way and says so.
    """
    for name, value in (("old_anchor", old_anchor), ("new_anchor", new_anchor)):
        if not isinstance(value, state_machine.AnchorIdentity):
            raise TypeError(f"{name} must be a state_machine.AnchorIdentity")
    affected = tuple(affected_backends)
    if not affected:
        raise ValueError(
            "affected_backends: required and non-empty — a sweep that names no "
            "affected backend supersedes nothing and hides the move"
        )
    ordered = tuple(entries)
    for entry in ordered:
        if not isinstance(entry, journal.JournalEntry):
            raise TypeError(
                f"entries must be journal.JournalEntry, got {type(entry).__name__}"
            )

    # A record already withdrawn is not withdrawn again: a second supersession
    # would be a second, contradictory reason for the same disappearance.
    already: set = set()
    for entry in ordered:
        if entry.kind == journal.KIND_SUPERSEDED:
            target = entry.payload.get("target_event_id")
            if isinstance(target, str):
                already.add(target)

    superseded: list = []
    superseded_records: list = []
    preserved: list = []
    preserved_records: list = []
    candidates: list = []
    entry_kind: dict = {}
    entry_tier: dict = {}
    entry_record_id: dict = {}
    fail_closed: list = []
    for entry in ordered:
        if entry.event_id in already:
            continue
        entry_kind[entry.event_id] = entry.kind
        entry_record_id[entry.event_id] = entry.record_id
        if entry.kind == journal.KIND_CANDIDATE_RECORDED and entry.record_id:
            candidates.append(entry.record_id)
        if entry.kind != journal.KIND_EVALUATION_EVENT:
            entry_tier[entry.event_id] = None
            continue
        tier = entry.payload.get("tier")
        candidate_id = entry.payload.get("candidate_id")
        entry_tier[entry.event_id] = tier
        if backends_by_candidate is None:
            hit = True
        else:
            declared = _declared_backends(backends_by_candidate.get(candidate_id))
            if declared is None:
                hit = True
                if candidate_id not in fail_closed:
                    fail_closed.append(candidate_id)
            else:
                hit = bool(set(declared) & set(affected))
        if tier in COMPARISON_TIERS and hit:
            superseded.append(entry.event_id)
            superseded_records.append(entry.record_id)
        else:
            preserved.append(entry.event_id)
            preserved_records.append(entry.record_id)

    reason = (
        f"{SUPERSEDED_BY_ANCHOR_MOVE}: production identity for {old_anchor.source_tree} "
        f"moved from {old_anchor.commit[:12]} to {new_anchor.commit[:12]}; every ratio "
        "in this record has a denominator that no longer exists (§8.9, AK-D22). "
        "Candidate source, patches and correctness results are preserved — only the "
        "comparisons died."
    )
    return AnchorMoveSweep(
        source_tree=old_anchor.source_tree,
        old_anchor=old_anchor.to_dict(),
        new_anchor=new_anchor.to_dict(),
        affected_backends=affected,
        reason=reason,
        superseded_entry_ids=tuple(superseded),
        superseded_record_ids=tuple(r for r in superseded_records if r is not None),
        preserved_entry_ids=tuple(preserved),
        preserved_record_ids=tuple(r for r in preserved_records if r is not None),
        preserved_candidate_ids=tuple(sorted(set(candidates))),
        entry_kind=entry_kind,
        entry_tier=entry_tier,
        entry_record_id=entry_record_id,
        fail_closed_candidates=tuple(sorted(c for c in fail_closed if c is not None)),
    )


def apply_anchor_move_supersession(journal_obj: journal.Journal,
                                   sweep: AnchorMoveSweep) -> tuple:
    """Journal the sweep. Re-checks the scope before every single append.

    The re-check is not paranoia about this module: `AnchorMoveSweep` is a public
    type a caller can construct, and the one mistake that cannot be undone in an
    append-only log is superseding a record that should have survived. So an entry
    that is not an evaluation event, a T0 record, or an entry also listed as
    preserved is refused here, at the write, and not only where the sweep was
    planned. The tier is checked FIRST, because it is the substantive §8.9 rule
    and a contradiction between the two sets is the lesser finding.

    Crucially the kind and the tier are re-read FROM THE JOURNAL, never from the
    sweep's own `entry_kind` / `entry_tier` maps. Checking a caller-constructed
    object against fields of that same object is not a check: a sweep that simply
    ASSERTS `{entry: EVALUATION_EVENT}` and `{entry: "T1"}` passed every clause
    below while the entry it named was a `CANDIDATE_RECORDED` record or a T0
    correctness event, and `Journal.append` only verifies that a supersession
    target EXISTS, not what it is. The sweep's claims are still compared against
    the journal, because a sweep that describes the log incorrectly is not a
    sweep anyone should be writing from.
    """
    if not isinstance(journal_obj, journal.Journal):
        raise TypeError("journal_obj must be a journal.Journal")
    if not isinstance(sweep, AnchorMoveSweep):
        raise TypeError("sweep must be an AnchorMoveSweep")
    preserved = set(sweep.preserved_entry_ids)
    actual = {entry.event_id: entry for entry in journal_obj.read_all()}
    for entry_id in sweep.superseded_entry_ids:
        entry = actual.get(entry_id)
        if entry is None:
            raise SupersessionScopeViolation(
                f"{entry_id} is not an entry of this journal; a supersession target is "
                "resolved against the log, never against the sweep that names it"
            )
        kind = entry.kind
        if kind != journal.KIND_EVALUATION_EVENT:
            raise SupersessionScopeViolation(
                f"{entry_id} is a {kind!r} entry, not an evaluation event; §8.9 item 3 "
                "preserves candidate source, patches and correctness results — only "
                "the comparisons died"
            )
        tier = entry.payload.get("tier")
        if tier in PRESERVED_TIERS:
            raise SupersessionScopeViolation(
                f"{entry_id} is a {tier} record; T0 compares artifacts and carries the "
                "correctness result, which §8.9 item 3 preserves"
            )
        if tier not in COMPARISON_TIERS:
            raise SupersessionScopeViolation(
                f"{entry_id} has tier {tier!r}, which is not a comparison tier "
                f"{list(COMPARISON_TIERS)}; an anchor move kills ratios, not records"
            )
        if sweep.entry_kind.get(entry_id) != kind \
                or sweep.entry_tier.get(entry_id) != tier \
                or sweep.entry_record_id.get(entry_id) != entry.record_id:
            raise SupersessionScopeViolation(
                f"{entry_id}: the sweep describes it as "
                f"{sweep.entry_kind.get(entry_id)!r}/"
                f"{sweep.entry_tier.get(entry_id)!r} carrying record "
                f"{sweep.entry_record_id.get(entry_id)!r}, but the journal holds "
                f"{kind!r}/{tier!r} carrying {entry.record_id!r}; a sweep that "
                "misdescribes the log is never executed against it"
            )
        if entry_id in preserved:
            raise SupersessionScopeViolation(
                f"{entry_id} is in both the superseded and the preserved set"
            )
    entries: list = []
    for entry_id in sweep.superseded_entry_ids:
        entries.append(journal_obj.append(
            journal.KIND_SUPERSEDED, sweep.payload_for(entry_id)))
    return tuple(entries)


@dataclass(frozen=True)
class AnchorMoveResponse:
    """All five of §8.9's ANCHOR_MOVED steps, as one deterministic result."""

    check: schemas.Check
    source_tree: str
    affected_backends: tuple
    sweep: AnchorMoveSweep
    reanchor_plan: Optional[ReanchorPlan]
    stop_detail: Mapping[str, Any]
    operator_notice: Mapping[str, Any]
    reason: str

    def to_stop_request(self) -> state_machine.StopRequest:
        """The §8.10 `ANCHOR_MOVED` stop, ready for the machine to dispose.

        Origin is `controller`: this response is derived from records by
        deterministic code. §8.10 gives the machine disposition either way — the
        machine validates the evidence and never the author (AK-D38).
        """
        return state_machine.StopRequest(
            state=state_machine.ANCHOR_MOVED,
            reason=self.reason,
            detail=dict(self.stop_detail),
            origin="controller",
        )

    def to_operator_input_request(self) -> state_machine.StopRequest:
        """§8.9 item 5's operator notice, as the §18 four-part decision package."""
        return state_machine.StopRequest(
            state=state_machine.OPERATOR_INPUT_REQUIRED,
            reason=self.reason,
            detail=dict(self.operator_notice),
            origin="controller",
        )

    def to_dict(self) -> dict:
        return {
            "check": {"outcome": self.check.outcome, "reasons": list(self.check.reasons)},
            "source_tree": self.source_tree,
            "affected_backends": list(self.affected_backends),
            "sweep": self.sweep.to_dict(),
            "reanchor_plan": (None if self.reanchor_plan is None
                              else self.reanchor_plan.to_dict()),
            "stop_detail": dict(self.stop_detail),
            "operator_notice": dict(self.operator_notice),
            "reason": self.reason,
        }


def respond_to_anchor_move(*, recorded_anchor: state_machine.AnchorIdentity,
                           observed_anchor: Optional[state_machine.AnchorIdentity],
                           entries: Sequence[Any],
                           champion_record: Optional[Mapping[str, Any]] = None,
                           backends_by_candidate: Optional[Mapping[str, Sequence[str]]] = None,
                           members_in_production: Sequence[str] = ()
                           ) -> AnchorMoveResponse:
    """§8.9's five steps for an anchor that moved outside a loop-initiated freeze.

      1. halt new candidate work for the tree — `compose_champion()` refuses while
         the observed anchor disagrees, and `to_stop_request()` is the machine's
         `ANCHOR_MOVED` transition;
      2. mark comparison evidence `superseded_by_anchor_move` with BOTH identities;
      3. preserve candidate source, patches and correctness results;
      4. re-anchor (a `ReanchorPlan`, which cannot shortcut re-measurement); and
      5. notify the operator with a four-part decision package.

    Refuses when the anchor did NOT move (there is nothing to supersede) and when
    it could not be verified — *"not observing is not evidence that production
    stayed put"*, and a sweep built on an unverified move would supersede real
    evidence on a guess.
    """
    check = state_machine.check_anchor_identity(recorded_anchor, observed_anchor)
    if check.outcome == schemas.PASS:
        raise CompositionError(
            "anchor identity matches the value recorded at BOOTSTRAP; there is no move "
            "to respond to, and superseding evidence for a move that did not happen "
            "would destroy a live comparison"
        )
    if check.outcome != schemas.FAIL or observed_anchor is None:
        raise state_machine.AnchorUncheckable(
            "anchor identity could not be verified: " + "; ".join(check.reasons)
        )

    affected = affected_backends_for_move(recorded_anchor, observed_anchor)
    sweep = plan_anchor_move_supersession(
        entries,
        old_anchor=recorded_anchor,
        new_anchor=observed_anchor,
        affected_backends=affected,
        backends_by_candidate=backends_by_candidate,
    )
    plan = (None if champion_record is None else plan_reanchor(
        champion_record, new_anchor=observed_anchor,
        members_in_production=members_in_production,
        trigger=REANCHOR_TRIGGER_ANCHOR_MOVED,
    ))

    reason = (
        f"production identity for {recorded_anchor.source_tree} moved from "
        f"{recorded_anchor.commit[:12]} to {observed_anchor.commit[:12]} outside a "
        f"loop-initiated freeze; {len(sweep.superseded_entry_ids)} comparison record(s) "
        f"lost their denominator across backends {list(affected)} (§8.9, AK-D22)"
    )
    stop_detail = {
        "recorded_anchor": recorded_anchor.to_dict(),
        "observed_anchor": observed_anchor.to_dict(),
        "affected_backends": list(affected),
        "halted_source_tree": recorded_anchor.source_tree,
        "superseded_entry_ids": list(sweep.superseded_entry_ids),
        "superseded_record_ids": list(sweep.superseded_record_ids),
        "preserved_entry_ids": list(sweep.preserved_entry_ids),
        "preserved_record_ids": list(sweep.preserved_record_ids),
        "preserved_candidate_ids": list(sweep.preserved_candidate_ids),
        "fail_closed_candidates": list(sweep.fail_closed_candidates),
        "reanchor_plan": None if plan is None else plan.to_dict(),
    }

    rebase_count = 0 if plan is None else len(plan.rebase_sources)
    options = [
        {
            "id": "reanchor",
            "action": (
                f"Confirm {observed_anchor.commit[:12]} as the production tip, rebase "
                f"{rebase_count} retained champion member(s) on it, and re-measure "
                f"T0/T1 on a rebuilt combined candidate"
            ),
            "cost": "one rebuild plus a full T0/T1 window per rebased lineage",
            "risk": "none to the record: superseded comparisons stay in the journal",
        },
        {
            "id": "rollback",
            "action": (
                f"Roll production back to {recorded_anchor.commit[:12]} and resume the "
                "campaign against the recorded anchor"
            ),
            "cost": "an operator cutover; no re-measurement",
            "risk": "whatever the move was fixing stays unfixed",
        },
        {
            "id": "halt",
            "action": (
                f"Halt {recorded_anchor.source_tree} research until the move is "
                "explained"
            ),
            "cost": "campaign wall time",
            "risk": "none; the safe holding state",
        },
    ]
    recommendation = "reanchor" if rebase_count else "halt"
    operator_notice = {
        "context": reason,
        "options": options,
        "recommendation": recommendation,
        "default": (
            f"hold — no new candidate work for {recorded_anchor.source_tree} proceeds "
            "until the operator answers; an unexpected anchor move usually means "
            "something happened that the loop should not silently absorb (§8.9 item 5)"
        ),
    }
    return AnchorMoveResponse(
        check=check,
        source_tree=recorded_anchor.source_tree,
        affected_backends=affected,
        sweep=sweep,
        reanchor_plan=plan,
        stop_detail=stop_detail,
        operator_notice=operator_notice,
        reason=reason,
    )


# =============================================================================
# Self-audit — the guarantee, proved from this file rather than asserted
# =============================================================================

def audit_no_composed_estimate_arithmetic(source: Optional[str] = None
                                          ) -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK on "can this module compose an estimate?".

    §8.9 says *"never infer composition by multiplying local speedups"*, §12 names
    it *"Summed local gains inflate readiness"*, and P-AK-SEARCH-1 denial 9 forbids
    synthesising a decision-grade quantity by combining search records. A comment
    saying so is worth nothing, so the property is checked:

      * no `*`, `/`, `//`, `**` or `@` anywhere in this module, so there is no
        expression that could combine two measurements into a third;
      * no call to a summation or averaging builtin (`sum`, `fsum`, `prod`,
        `mean`, …), because §12's row is *"SUMMED local gains"* and `sum()`
        needs no `*`;
      * no reference to any name, attribute, keyword, definition or literal key
        that carries a measured quantity, so no record's performance block is
        reachable from here;
      * no composition dataclass declares such a field.

    COULD_NOT_CHECK when the source cannot be read, cannot be parsed, or is
    empty — a module that could not be inspected has not been shown to be clean,
    and a PASS earned on nothing is the check clearing itself.

    WHAT A PASS DOES NOT PROVE. This is a SYNTACTIC audit of ONE file, and saying
    so is part of it being honest:

      * `a + b` and `a - b` are permitted (this module concatenates strings), so
        addition of two measurements is not excluded by the binop scan; what
        excludes it is that no measured quantity is reachable from here in the
        first place, which is the property the name/attribute/key scan carries;
      * it inspects this file only. A helper in another module could do anything;
        the module-boundary claim rests on `compose_champion()` accepting no
        member evidence, not on this audit;
      * the key list is matched EXACTLY, so a computed key (`"perf" "ormance"`)
        is not detected. It is a guard against drift, not against an adversary
        with commit access.
    """
    if source is None:
        path = os.path.abspath(__file__)
        if path.endswith(".pyc"):  # pragma: no cover - defensive
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{path}: only compiled bytecode is available, so the source cannot be "
                "inspected",
            ))
        try:
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{path}: source could not be read ({exc})",
            ))
    if not isinstance(source, str):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"source is {type(source).__name__}, not text; nothing was inspected",
        ))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"source did not parse ({exc}); nothing was inspected",
        ))
    if not tree.body:
        # An empty, blank or comment-only source parses cleanly, matches no
        # forbidden node, and would therefore report PASS — the audit clearing
        # itself by being handed nothing to audit. A check that a truncated read
        # or an empty string can satisfy is not evidence about this module.
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "source contains no statements; an empty source matches no forbidden "
            "construct and a PASS on it would say nothing about this module",
        ))

    forbidden = set(_FORBIDDEN_EVIDENCE_KEYS)
    findings: list = []
    skip: set = set()
    for node in ast.walk(tree):
        # The vocabulary's own definition names every forbidden key by necessity.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_FORBIDDEN_EVIDENCE_KEYS":
                    for inner in ast.walk(node):
                        skip.add(id(inner))
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, _FORBIDDEN_BINOPS):
            findings.append(
                f"line {node.lineno}: {type(node.op).__name__} — this module contains "
                "no arithmetic that could combine two measurements"
            )
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, _FORBIDDEN_BINOPS):
            findings.append(
                f"line {node.lineno}: augmented {type(node.op).__name__}"
            )
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            findings.append(
                f"line {node.lineno}: attribute {node.attr!r} names a measured quantity"
            )
        elif isinstance(node, ast.Name) and node.id in forbidden:
            findings.append(
                f"line {node.lineno}: name {node.id!r} names a measured quantity"
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value in forbidden:
            findings.append(
                f"line {node.lineno}: literal key {node.value!r} names a measured "
                "quantity"
            )
        elif isinstance(node, (ast.arg,)) and node.arg in forbidden:
            findings.append(
                f"line {node.lineno}: parameter {node.arg!r} names a measured quantity"
            )
        elif isinstance(node, ast.keyword) and node.arg in forbidden:
            # `f(estimate=…)` is an `ast.keyword`, whose `arg` is a bare `str`,
            # not an `ast.arg` node — invisible to the scan above.
            findings.append(
                f"keyword argument {node.arg!r} names a measured quantity"
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name in forbidden:
            findings.append(
                f"line {node.lineno}: definition {node.name!r} names a measured quantity"
            )
        elif isinstance(node, ast.alias) and forbidden & {node.name, node.asname}:
            findings.append(
                f"import of {node.name!r} names a measured quantity"
            )
        elif isinstance(node, ast.Call):
            called = node.func
            name = None
            if isinstance(called, ast.Name):
                name = called.id
            elif isinstance(called, ast.Attribute):
                name = called.attr
            if name in _FORBIDDEN_AGGREGATORS:
                findings.append(
                    f"line {node.lineno}: call to {name!r} — §12's failure is SUMMED "
                    "local gains, and an aggregation needs no multiplication sign"
                )

    for klass in (FrontierCandidate, Experiment, LineageMember, ComposedLineage):
        for field in dataclasses.fields(klass):
            if field.name in forbidden:
                findings.append(
                    f"{klass.__name__}.{field.name} carries a measured quantity"
                )

    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    return schemas.Check(schemas.PASS)
