#!/usr/bin/env python3
"""do_not_repeat.py — the §19.2 memory plane `hypotheses.py` says it does not own.

WHY THIS MODULE EXISTS
----------------------
`hypotheses.py:261` states it plainly:

    This module does not BUILD that ledger (the memory-update plane owns it); it
    consumes matches and disposes them.

THE MEMORY-UPDATE PLANE DID NOT EXIST. `hypotheses.check_do_not_repeat()` is a correct
guard wired to nothing, and `selection.match_ledger()` is a correct matcher over a
`LedgerEntry` corpus nothing compiles — the same defect shape this package has now hit
four times (a guard defined and never wired, a table read instead of enforced, a
dashboard rendering clean over a dead producer). Without the ledger the loop cannot
tell **"tried and failed"** from **"never tried"**, so it re-tries dead ideas forever
and pays for each one with a resource claim.

This module folds the record — the journal plus the hypothesis ledger — into *what has
already been tried, in what regime, and how did it end*, and hands it back in EXACTLY
the shape `check_do_not_repeat()` already consumes (`hypotheses.LedgerMatch`). It
conforms to that consumer rather than inventing a shape the consumer would have to be
adapted to; `CompiledLedger` satisfies the `hypotheses.DoNotRepeatLedger` protocol as
written, `matches_for(regime, statement)` included.

WHAT IS MATCHED ON, AND WHAT IS DELIBERATELY NOT
------------------------------------------------
"Has this been tried?" is a similarity question and both errors are expensive, in
opposite directions:

* **too loose** — a genuinely new idea is rejected as already-tried. The loop goes
  sterile *while looking productive*, and nothing ever tests the suppressed family
  again (§19.3's stated reason a wrong suppression is invisible). This is the worse
  failure and it is SILENT.
* **too strict** — the loop re-runs dead ideas and burns claims. Expensive, but LOUD:
  the repeat appears in the journal.

So every ambiguity in this module resolves toward *too strict*. The concrete form of
that rule is `CLASS_PRECEDENCE`: `MATCHED_NEGATIVE` — one of the only two classes that
reject — is LAST, so every demotion condition must be absent before a match can close a
question, and `HARD_CONSTRAINT` is first only because it is a policy/hardware
prohibition that no measurement can confound.

MATCHED ON (both axes must agree):

1. **The structural target of the change.** `mechanism` — the canonical token naming
   *what change is being made* — must be EQUAL on both sides, and where both sides
   declare `ops`/`symbols`/`files`/`change_class` those must intersect. This is the
   axis `selection.match_ledger()` already gates on first (`entry.mechanism !=
   facets["mechanism"] -> skip`), reused rather than re-invented; on a proposal it is
   read from §7.1's `selection` block through `fingerprint.selection_block()`, the
   module that exists precisely because two planes once computed a proposal's identity
   differently and neither could see it.
2. **The regime.** Every regime dimension the ENTRY was measured in must be declared by
   the question AND share a value with it — `selection.match_ledger`'s rule verbatim
   (an unobserved dimension breaks the match rather than being assumed equal). §19.2:
   *"'do not repeat' without regime identity is dangerous, because this project
   repeatedly observes sign changes across architecture, substrate, batch, context and
   quant."*

NOT MATCHED ON — **the statement**. Not its words, not its length, not its similarity
to anything. `matches_for()` accepts a `statement` because the protocol passes one and
DOES NOT READ IT: `MatchQuery` has no field to hold it, so an agent rewording a tried
idea still matches (the prose changed; the mechanism and regime did not), and two
different ideas about one function do not match (same `ops`, different `mechanism`).
`audit_matching_ignores_prose()` proves this from the objects and behaviourally,
because "we don't look at the statement" is exactly the kind of property that survives
in a docstring long after a convenience `if statement in entry.statement` has been
added.

ANCHOR SENSITIVITY
------------------
A negative taken against an anchor that has since MOVED is a `SUPERSEDED_FACT`, never a
`MATCHED_NEGATIVE`. A kernel idea that failed on v7 may well win on v8 — iqk's +33-43%
prefill is the standing example — and rejecting it forever on a stale anchor is how the
loop stops being able to learn. The comparison is `evaluator.api.AnchorIdentity.
identity_matches()`, the package's existing anchor-binding work, and its three outcomes
are all honoured: PASS keeps the negative, FAIL supersedes it, and COULD_NOT_CHECK
(no anchor recorded, or a tool named on one side only) ALSO supersedes it — an
unobserved component is never a PASS, so it can never be the thing that closes a
question.

CONFOUNDED_RESULT
-----------------
2026-08-04 supplied the live example: an A/A run destroyed mid-flight by a parallel
session bringing up seven llama-servers. **A result taken under contention is not a
negative result, it is no result.** Any voided contributor — `status: invalid`, or any
`VOID:` integrity flag, `CONCURRENT_INFERENCE_CONTAMINATION` first among them — folds
the whole attempt to `CONFOUNDED_RESULT`, which is advisory, so the question stays
open and a repaired experiment is what closes it.

Note what is NOT a confounder: `co_residency: co_resident:<lineup>`. Co-residency is
scheduling DATA, not a trust gate (the amendment of 2026-07-27; some lineups are
concurrent BY DESIGN), so the trust signal is the void finding the evaluator wrote, not
the fact that something else was on the host.

EXPLAINABILITY
--------------
Every match carries which prior attempt it came from, which event ids are behind it,
which class it folded to and why (`MatchExplanation`), and every entry that was
COMPARED AND DID NOT MATCH carries the reason it did not (`LedgerLookup.near_misses`).
A rejection an agent cannot inspect is one it will route around, and a *non*-rejection
nobody can inspect is how a ledger silently becomes toothless.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not rank, bank, compose, gate, spend a claim, or write anything: `fold_journal`
is a pure function of records already on disk and `CompiledLedger` is frozen. It does
not synthesise `selection.LedgerEntry` values either, and that is deliberate rather
than an omission — `selection.REJECTING_LEDGER_CLASSES` includes `SUPERSEDED_FACT`
because a stale PROPOSAL must not execute, while `hypotheses.REJECTING_MATCH_CLASSES`
excludes it because the QUESTION stays open. `test_loop_integration` asserts that
divergence on purpose; one compiler emitting both shapes would be one place for the two
to be quietly equalised.

Governing instrument: `measurement/protocols/kernel-research.md` (P-AK-SEARCH-1,
RATIFIED 2026-08-03), §19.2 and §19.3.
"""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .. import journal, schemas
from ..evaluator import api as evaluator_api
from . import fingerprint, hypotheses
from .state_machine import ControllerError

__all__ = [
    # errors
    "DoNotRepeatError", "LedgerFoldError",
    # vocabulary
    "REGIME_DIMENSIONS", "TARGET_SET_DIMENSIONS", "DIMENSION_ALIASES",
    "CLASS_PRECEDENCE", "CONFOUNDING_STATUSES", "OUTCOME_SKIPPED",
    "CONSTRAINT_EVENT_KIND",
    # types
    "StructuralTarget", "MatchQuery", "PriorAttempt", "MatchExplanation",
    "LedgerLookup", "CompiledLedger",
    # pure functions and checks
    "canonical_token", "canonical_dimension", "normalize_dimensions",
    "structural_target", "read_facets", "fold_journal", "disposition",
    "audit_matching_ignores_prose",
]


# =============================================================================
# Errors
# =============================================================================

class DoNotRepeatError(ControllerError):
    """Base for every refusal here.

    Extends the CONTROLLER plane's base for the reason `hypotheses.HypothesisError`
    does: a driver catches one plane, not one module of it.
    """


class LedgerFoldError(DoNotRepeatError):
    """The fold was handed something it cannot read as a record.

    NEVER a degraded empty ledger: an empty ledger is the statement *"nothing has been
    tried"*, the planner acts on it, and a malformed input must not be able to say it.
    That is the same rule `OperatorHypothesisStore.load()` applies to an unreadable
    store, for the same reason.
    """


# =============================================================================
# Vocabulary
# =============================================================================

#: The regime axis: dimensions across which this project has OBSERVED sign changes
#: (§19.2 names architecture, substrate, batch, context and quant explicitly). Every
#: name here is one of `selection.LEDGER_DIMENSIONS` — the vocabulary the §19.2 ledger
#: already declares — so an entry compiled here keys on dimensions the proposal-side
#: matcher also understands. `test_do_not_repeat` asserts the containment rather than
#: trusting this comment.
REGIME_DIMENSIONS = frozenset({
    "backend", "phase", "batch", "context", "quant", "models", "shapes",
    "architecture", "substrate",
})

#: The structural-target axis, minus `mechanism` (which is single-valued and handled
#: separately). `ops`, `change_class` and `hierarchy_layer` are
#: `selection.LEDGER_DIMENSIONS` members; `symbols` and `files` are finer-grained
#: target keys named here because a hypothesis about one function is not a hypothesis
#: about the file it lives in.
#:
#: `hierarchy_layer` sits HERE and not in `REGIME_DIMENSIONS` deliberately: it says
#: which layer of the stack a change is made at, which is a property of the CHANGE, not
#: of the conditions it was measured under. Filed as a regime dimension it would demand
#: that every question state its layer before any prior negative could be seen at all,
#: which is a barrier on the wrong axis and would make the ledger silently toothless.
TARGET_SET_DIMENSIONS = ("ops", "symbols", "files", "change_class", "hierarchy_layer")

#: Spellings that mean one dimension. CLOSED and hand-written: an alias makes two
#: spellings COMPARABLE, which tightens matching (an entry dimension the question spells
#: differently would otherwise read as undeclared and break the match), so the only way
#: an alias can do harm is by unifying two dimensions that are genuinely different —
#: which is a judgement, and therefore is made here rather than by a heuristic.
DIMENSION_ALIASES: Mapping[str, str] = {
    "batch_band": "batch", "batch_size": "batch", "batches": "batch",
    "context_band": "context", "context_length": "context", "ctx": "context",
    "model": "models", "model_id": "models",
    "shape": "shapes",
    "quantization": "quant", "quant_type": "quant",
    "arch": "architecture",
    "device": "substrate", "device_class": "substrate",
    "layer": "hierarchy_layer",
    # §19.2/§7.2 spell the *phase* axis `regimes` on a proposal's `target` block
    # (`{"regimes": ["decode"]}`) and `phase` on a hypothesis regime. One axis.
    "regimes": "phase", "regime": "phase",
    "op": "ops", "operator": "ops", "operators": "ops",
    "symbol": "symbols", "file": "files",
}

#: The order a folded attempt's candidate classes are resolved in — most severe first.
#:
#: TWO PROPERTIES CARRY THE SAFETY OF THIS MODULE, and both are asserted in the suite:
#:
#: * `MATCHED_NEGATIVE` is LAST. It is the class that closes a question on evidence, so
#:   every demotion condition (a confounded contributor, a moved anchor, a partial
#:   scope, a missing regime identity) outranks it and the rejection has to survive all
#:   of them.
#: * `HARD_CONSTRAINT` is FIRST, and it is the one rejecting class that outranks
#:   `CONFOUNDED_RESULT` — a hardware, policy, correctness or ownership prohibition is
#:   not undone by a benchmark that ran under contention.
CLASS_PRECEDENCE = (
    hypotheses.MATCH_CLASS_HARD_CONSTRAINT,
    hypotheses.MATCH_CLASS_CONFOUNDED_RESULT,
    hypotheses.MATCH_CLASS_SUPERSEDED_FACT,
    hypotheses.MATCH_CLASS_CONDITIONAL_NEGATIVE,
    hypotheses.MATCH_CLASS_LOW_VALUE,
    hypotheses.MATCH_CLASS_MATCHED_NEGATIVE,
)

#: Evaluation-event statuses that say the run was not a measurement. `invalid` is the
#: protocol's word for a voided run; `crash` and `timeout` describe a run that produced
#: no comparable number. `fail` and `inconclusive` are deliberately absent — those ran.
CONFOUNDING_STATUSES = frozenset({"invalid", "crash", "timeout"})

#: The synthetic outcome for a proposal that was never executed (`PROPOSAL_SKIPPED`).
#: Distinct from every `hypotheses.RESOLUTIONS` value because "we chose not to run it"
#: is not a result, and folding it into `inconclusive` would make a scheduling decision
#: look like an experiment that failed to resolve.
OUTCOME_SKIPPED = "skipped"

#: The §19.4 bootstrap-knowledge kind that carries a compiled constraint. Named from
#: `journal.BOOTSTRAP_KNOWLEDGE_KINDS` rather than typed as a literal, so a rename
#: there is an ImportError here and not a silently empty constraint corpus.
CONSTRAINT_EVENT_KIND = "CONSTRAINT_COMPILED"

_ENTRY_ID_RE = re.compile(r"^dnr-[a-z0-9][a-z0-9_.-]*$")
_NON_TOKEN_RE = re.compile(r"[^a-z0-9]+")


# =============================================================================
# Canonicalisation — the ONLY normalisation this module performs
# =============================================================================

def canonical_token(value: Any) -> Optional[str]:
    """One comparable token, or `None` when the value carries no identity.

    TYPE-TAGGED, so `128` and `"128"` are two tokens: `selection._canonical_items`
    makes the same choice for the same reason — a batch band spelled `b128` and a batch
    size of 128 are different facts and a matcher that conflates them is matching on
    coincidence.

    The string normalisation is case-folding and separator-collapsing and NOTHING ELSE.
    There is no stemming, no edit distance, no substring containment and no token-
    overlap score anywhere in this module: `"Elementwise/norm fusion"` and
    `"elementwise_norm_fusion"` are one token because they are one identifier written
    two ways, while `"fuse norm into matmul"` and `"vectorize the norm loop"` stay two
    tokens no matter how similar an embedding would call them.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return f"bool:{'true' if value else 'false'}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"canonical_token: {value!r} is not a finite value")
        return f"float:{value!r}"
    if isinstance(value, str):
        collapsed = _NON_TOKEN_RE.sub("_", value.strip().lower()).strip("_")
        return f"str:{collapsed}" if collapsed else None
    raise TypeError(
        f"canonical_token: {type(value).__name__} has no canonical token; a match "
        "dimension is a scalar or a sequence of scalars"
    )


def _values_of(value: Any) -> tuple:
    """`(tokens, raw_values, refusal)` for one dimension's value.

    A refusal is REPORTED, never silently dropped: a dimension this function cannot
    read is a dimension the entry was measured in and the comparison cannot see, and a
    silently ignored dimension makes the match LOOSER — the failure direction that is
    invisible.
    """
    if value is None:
        return (), (), None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        items = [value]
    else:
        items = list(value)
    tokens: list = []
    raw: list = []
    for item in items:
        try:
            token = canonical_token(item)
        except (TypeError, ValueError) as exc:
            return (), (), str(exc)
        if token is None:
            continue
        if token not in tokens:
            tokens.append(token)
            raw.append(item)
    return tuple(tokens), tuple(raw), None


#: The reason an unrecognised key is reported, written ONCE and identical on both
#: axes. Both `normalize_dimensions` passes see every key, so a key that is a target
#: key looks unrecognised to the regime pass and vice versa; an identical string is
#: what lets `_merge_ignored` dedupe the two views into one honest complaint instead of
#: reporting `backend` as ignored because the target pass did not want it.
_NOT_A_DIMENSION = (
    "is not a declared match dimension (regime dimensions: "
    + ", ".join(sorted(REGIME_DIMENSIONS))
    + "; structural-target keys: mechanism, "
    + ", ".join(sorted(TARGET_SET_DIMENSIONS))
    + "). It was carried into the record, but nothing compares on it"
)


def canonical_dimension(key: str) -> str:
    """The canonical name a key contributes under, after `DIMENSION_ALIASES`."""
    return DIMENSION_ALIASES.get(key.strip().lower(), key.strip().lower())


def normalize_dimensions(mapping: Any, *, keep: frozenset) -> tuple:
    """`(dimensions, raw, ignored)` — one mapping read as match dimensions.

    `dimensions` maps a canonical dimension name to a tuple of canonical tokens; `raw`
    keeps the values as written, for the explanation; `ignored` is a tuple of
    `(key, reason)` PAIRS for every key that did not become a dimension.

    `ignored` is the part that matters, and it is a pair rather than a sentence so that
    `read_facets` can tell "no axis read this key" from "the other axis read it".
    `OperatorHypothesisStore` REFUSES an unknown key because *"a key this loader
    ignores is a key the operator believes had an effect"*; refusing is wrong at this
    layer — the fold reads records that are already durable and cannot refuse them into
    non-existence — so the key is REPORTED instead and travels with the lookup.
    """
    if not isinstance(mapping, Mapping):
        raise LedgerFoldError(
            f"match dimensions must be a mapping, got {type(mapping).__name__}"
        )
    dimensions: dict = {}
    raw: dict = {}
    ignored: list = []
    for key in sorted(mapping, key=str):
        if not isinstance(key, str):
            ignored.append((repr(key), "dimension names must be strings"))
            continue
        name = canonical_dimension(key)
        if name not in keep:
            ignored.append((key, _NOT_A_DIMENSION))
            continue
        tokens, values, refusal = _values_of(mapping[key])
        if refusal is not None:
            ignored.append((key, refusal))
            continue
        if not tokens:
            ignored.append((key, "declares no value, so it constrains nothing"))
            continue
        merged = dimensions.get(name, ())
        merged_raw = raw.get(name, ())
        dimensions[name] = merged + tuple(t for t in tokens if t not in merged)
        raw[name] = merged_raw + tuple(
            v for t, v in zip(tokens, values) if t not in merged
        )
    return dimensions, raw, tuple(ignored)


def _merge_ignored(mapping: Any, regime: Mapping, target: "StructuralTarget",
                   *pair_groups) -> tuple:
    """The keys NO axis read, rendered once each.

    A key is only ignored when neither the regime pass nor the target pass consumed it.
    Reporting the union of the two passes' complaints — which is what a naive
    concatenation gives — would tell an author that `backend` had no effect because the
    STRUCTURAL-TARGET pass did not want it, which is false and is exactly the kind of
    misinformation that gets a warning channel ignored.
    """
    consumed = set()
    if isinstance(mapping, Mapping):
        for key in mapping:
            if not isinstance(key, str):
                continue
            name = canonical_dimension(key)
            if name in regime or name in target.sets or (
                name == "mechanism" and target.is_identified
            ):
                consumed.add(key)
    by_key: dict = {}
    for group in pair_groups:
        for key, reason in group:
            if key in consumed:
                continue
            reasons = by_key.setdefault(key, [])
            if reason not in reasons:
                reasons.append(reason)
    out: list = []
    for key, reasons in sorted(by_key.items()):
        # A specific reason beats the generic one. `models: []` on a proposal is a
        # DECLARED dimension carrying no value; saying it is also "not a declared
        # dimension" — which is what the target pass thinks of it — is two answers to
        # one question and the wrong one is the memorable one.
        specific = [r for r in reasons if r != _NOT_A_DIMENSION]
        out.append(f"{key!r}: {' / '.join(specific or reasons)}")
    return tuple(out)


def read_facets(mapping: Any) -> tuple:
    """`(regime, regime_raw, target, ignored)` — one mapping read on BOTH axes.

    The single entry point every caller uses, so the regime axis and the structural-
    target axis are always read from the same mapping with the same vocabulary, and so
    "which keys did nothing" is answered once rather than once per axis.
    """
    regime, regime_raw, ignored_regime = normalize_dimensions(
        mapping, keep=REGIME_DIMENSIONS
    )
    target, ignored_target = structural_target(mapping)
    return regime, regime_raw, target, _merge_ignored(
        mapping, regime, target, ignored_regime, ignored_target
    )


# =============================================================================
# The structural target
# =============================================================================

@dataclass(frozen=True)
class StructuralTarget:
    """*What* is being changed — the axis that survives a rewording.

    `mechanism` is single-valued and REQUIRED for any match: it is the identity of the
    change itself, and it is what makes "two different ideas about the same function"
    two entries rather than one. The sets are additional constraints, applied only when
    BOTH sides declare them, because an entry that names an op and a question that does
    not is an incomplete comparison, not a mismatch.
    """

    mechanism: Optional[str] = None
    mechanism_raw: Optional[str] = None
    sets: Mapping[str, tuple] = field(default_factory=dict)
    sets_raw: Mapping[str, tuple] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mechanism is not None and not isinstance(self.mechanism, str):
            raise TypeError("mechanism must be a canonical token string or None")
        for name in ("sets", "sets_raw"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")

    @property
    def is_identified(self) -> bool:
        """Whether this target can take part in a match AT ALL.

        A target with no mechanism cannot: matching on regime alone would suppress
        every idea about a backend/phase, which is the sterile-loop failure in its
        purest form.
        """
        return self.mechanism is not None

    def agreement(self, other: "StructuralTarget") -> tuple:
        """`(agrees, reasons)` against a query's target.

        Reasons are returned for BOTH outcomes: a match must be explainable, and so
        must a near-miss, because an entry that never matches anything is
        indistinguishable from an entry that is not there.
        """
        if not self.is_identified:
            return False, (
                "the ledger entry names no mechanism, so there is nothing to compare "
                "a question against; a regime-only match would suppress every idea in "
                "that regime",
            )
        if not other.is_identified:
            return False, (
                "the question names no mechanism (regime key 'mechanism'), so it "
                "cannot be shown to repeat anything; a do-not-repeat match keys on the "
                "structural target of the change, never on the words of the statement",
            )
        if self.mechanism != other.mechanism:
            return False, (
                f"different mechanism: entry {self.mechanism_raw!r} vs question "
                f"{other.mechanism_raw!r} — two ideas about one target are two ideas",
            )
        reasons = [f"same mechanism {self.mechanism_raw!r}"]
        for name in TARGET_SET_DIMENSIONS:
            mine = self.sets.get(name)
            theirs = other.sets.get(name)
            if not mine or not theirs:
                if mine or theirs:
                    reasons.append(
                        f"{name}: declared on one side only "
                        f"({'entry' if mine else 'question'}), so it constrains nothing"
                    )
                continue
            shared = [t for t in mine if t in theirs]
            if not shared:
                return False, (
                    f"different {name}: entry {list(self.sets_raw.get(name, ()))} vs "
                    f"question {list(other.sets_raw.get(name, ()))}",
                )
            reasons.append(f"{name} overlaps on {len(shared)} value(s)")
        return True, tuple(reasons)

    def to_dict(self) -> dict:
        return {
            "mechanism": self.mechanism_raw,
            "mechanism_token": self.mechanism,
            "sets": {k: list(v) for k, v in sorted(self.sets_raw.items())},
        }


def structural_target(mapping: Any) -> tuple:
    """`(target, ignored)` — the structural target declared by a mapping.

    Reads `mechanism` plus `TARGET_SET_DIMENSIONS` (through `DIMENSION_ALIASES`) and
    nothing else. In particular it does not read, hash, or otherwise consult any prose
    field — a `statement`, a `hypothesis`, a `narrative` or a `conceptual_change` key
    in the mapping is simply not looked at.
    """
    keep = frozenset(TARGET_SET_DIMENSIONS)
    sets, sets_raw, ignored = normalize_dimensions(mapping, keep=keep)
    mechanism = None
    mechanism_raw = None
    if isinstance(mapping, Mapping):
        for key in sorted(mapping, key=str):
            if isinstance(key, str) and canonical_dimension(key) == "mechanism":
                tokens, values, refusal = _values_of(mapping[key])
                if refusal is not None:
                    ignored = ignored + ((key, refusal),)
                elif len(tokens) > 1:
                    ignored = ignored + ((key, (
                        "names more than one mechanism; a single change has one "
                        "structural identity, so none was taken"
                    )),)
                elif tokens:
                    mechanism, mechanism_raw = tokens[0], values[0]
                break
    return (
        StructuralTarget(
            mechanism=mechanism, mechanism_raw=mechanism_raw,
            sets=sets, sets_raw=sets_raw,
        ),
        ignored,
    )


# =============================================================================
# The query — note what it CANNOT hold
# =============================================================================

@dataclass(frozen=True)
class MatchQuery:
    """One question, reduced to the two axes this module matches on.

    **THERE IS NO `statement` FIELD, AND THAT IS THE POINT.** `matches_for()` is handed
    a statement by the `hypotheses.DoNotRepeatLedger` protocol and converts the regime
    alone into this object, so there is no slot along which prose could reach the
    matcher — the same structural form `Hypothesis.evidence_grade` uses to make a
    promoted grade unrepresentable rather than merely forbidden.
    `audit_matching_ignores_prose()` enumerates the fields and proves it.
    """

    regime: Mapping[str, tuple] = field(default_factory=dict)
    regime_raw: Mapping[str, tuple] = field(default_factory=dict)
    target: StructuralTarget = field(default_factory=StructuralTarget)
    ignored: tuple = ()
    #: Dimensions the question NAMED and the matcher could not use — a value it could
    #: not read, or a key that declared no value. Distinct from `ignored`, which is
    #: mostly keys that are not match dimensions at all (benign). A named-but-unusable
    #: dimension makes the query LOOSER, which is the failure direction that is
    #: invisible, so it is carried here and reaches the verdict through
    #: `LedgerLookup.why_not_comparable`.
    unusable_dimensions: tuple = ()

    @staticmethod
    def from_regime(regime: Any) -> "MatchQuery":
        if regime is None:
            regime = {}
        dimensions, raw, target, ignored = read_facets(regime)
        known = REGIME_DIMENSIONS | frozenset(TARGET_SET_DIMENSIONS) | {"mechanism"}
        named = {
            canonical_dimension(key) for key in regime
            if isinstance(key, str)
        } & known if isinstance(regime, Mapping) else set()
        used = set(dimensions) | set(target.sets)
        if target.is_identified:
            used.add("mechanism")
        return MatchQuery(
            regime=dimensions, regime_raw=raw, target=target, ignored=ignored,
            unusable_dimensions=tuple(sorted(named - used)),
        )

    def to_dict(self) -> dict:
        return {
            "regime": {k: list(v) for k, v in sorted(self.regime_raw.items())},
            "target": self.target.to_dict(),
            "ignored": list(self.ignored),
            "unusable_dimensions": list(self.unusable_dimensions),
        }


# =============================================================================
# One prior attempt
# =============================================================================

@dataclass(frozen=True)
class PriorAttempt:
    """*What was tried, in what regime, and how did it end* — one folded row.

    This is the unit `hypotheses.Attempt` is a receipt for and the unit §19.2 calls a
    ledger entry. It carries its whole provenance — every journal and hypothesis-ledger
    event id behind it — because a suppression whose receipt does not resolve reverts
    to advisory (§19.3), and a receipt that cannot be resolved by a reader is not one.
    """

    entry_id: str
    entry_class: str
    regime: Mapping[str, tuple]
    regime_raw: Mapping[str, tuple]
    target: StructuralTarget
    outcome: Optional[str] = None
    event_ids: tuple = ()
    hypothesis_ids: tuple = ()
    proposal_ids: tuple = ()
    candidate_ids: tuple = ()
    receipt: Optional[str] = None
    anchor: Optional[Any] = None
    anchor_outcome: Optional[str] = None
    conflicted: bool = False
    reopen_when: Optional[str] = None
    why: tuple = ()
    ignored: tuple = ()

    def __post_init__(self) -> None:
        if not _ENTRY_ID_RE.match(self.entry_id or ""):
            raise LedgerFoldError(
                f"entry_id: {self.entry_id!r} must start with 'dnr-' (the id prefix is "
                "the record's family, per §7)"
            )
        if self.entry_class not in hypotheses.MATCH_CLASSES:
            raise LedgerFoldError(
                f"entry_class: {self.entry_class!r} not in "
                f"{sorted(hypotheses.MATCH_CLASSES)}; this module produces §19.2 "
                "classes and nothing else, because the consumer's vocabulary is closed"
            )
        if self.receipt is not None and (
            not isinstance(self.receipt, str) or not self.receipt.strip()
        ):
            raise LedgerFoldError("receipt must be None or a non-empty string")

    def rejects(self, *, reopen_predicate_satisfied: bool = False) -> bool:
        """Whether a match on this entry would REJECT, by the CONSUMER's rule.

        Every clause is derived from `hypotheses.check_do_not_repeat()` rather than
        restated, so the two can never disagree: the class must be one of
        `REJECTING_MATCH_CLASSES`, a conflicted entry is never authoritative, an
        unreceipted match is COULD_NOT_CHECK rather than a rejection, and a satisfied
        reopen predicate excuses a `MATCHED_NEGATIVE` **and only that class** — a
        hardware or policy prohibition is not reopened by a fact about the world.
        """
        if self.entry_class not in hypotheses.REJECTING_MATCH_CLASSES:
            return False
        if self.conflicted or self.receipt is None:
            return False
        if (reopen_predicate_satisfied
                and self.entry_class == hypotheses.MATCH_CLASS_MATCHED_NEGATIVE):
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "entry_class": self.entry_class,
            "regime": {k: list(v) for k, v in sorted(self.regime_raw.items())},
            "target": self.target.to_dict(),
            "outcome": self.outcome,
            "event_ids": list(self.event_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "proposal_ids": list(self.proposal_ids),
            "candidate_ids": list(self.candidate_ids),
            "receipt": self.receipt,
            "anchor": self.anchor.short() if self.anchor is not None else None,
            "anchor_outcome": self.anchor_outcome,
            "conflicted": self.conflicted,
            "reopen_when": self.reopen_when,
            "why": list(self.why),
            "ignored": list(self.ignored),
        }


# =============================================================================
# The lookup and its explanation
# =============================================================================

@dataclass(frozen=True)
class MatchExplanation:
    """Why one entry matched, or why it did not. Never a bare boolean.

    A rejection an agent cannot inspect is one it will route around; a non-rejection
    nobody can inspect is how a ledger goes silently toothless. Both directions get a
    row.
    """

    entry_id: str
    entry_class: str
    matched: bool
    rejects: bool
    reasons: tuple
    matched_dimensions: tuple = ()
    #: Dimensions this entry was measured in that the QUESTION did not state. Non-empty
    #: only on a near-miss, and it is the difference between "this idea is new" and
    #: "nobody said enough to tell" — see `LedgerLookup.undeclared_dimensions`.
    undeclared_dimensions: tuple = ()
    event_ids: tuple = ()
    hypothesis_ids: tuple = ()
    proposal_ids: tuple = ()
    receipt: Optional[str] = None
    anchor: Optional[str] = None
    anchor_outcome: Optional[str] = None
    conflicted: bool = False
    reopen_predicate_satisfied: bool = False
    outcome: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "entry_class": self.entry_class,
            "matched": self.matched,
            "rejects": self.rejects,
            "reasons": list(self.reasons),
            "matched_dimensions": list(self.matched_dimensions),
            "undeclared_dimensions": list(self.undeclared_dimensions),
            "event_ids": list(self.event_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "proposal_ids": list(self.proposal_ids),
            "receipt": self.receipt,
            "anchor": self.anchor,
            "anchor_outcome": self.anchor_outcome,
            "conflicted": self.conflicted,
            "reopen_predicate_satisfied": self.reopen_predicate_satisfied,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class LedgerLookup:
    """One question's answer: the matches, and everything that went into deciding."""

    matches: tuple = ()
    explanations: tuple = ()
    near_misses: tuple = ()
    ignored_dimensions: tuple = ()
    query: Optional[MatchQuery] = None

    @property
    def rejecting(self) -> tuple:
        return tuple(e for e in self.explanations if e.rejects)

    @property
    def incomplete_comparisons(self) -> tuple:
        """Entries about THIS mechanism that the question did not say enough to compare.

        The hypothesis-side twin of `selection.REJECT_REGIME_IDENTITY_INCOMPLETE`
        (*"a proposal cannot escape a receipted negative by declining to say which
        regime it is in"*). A proposal is REJECTED for it, because a proposal is about
        to spend a claim; a question cannot be — the operator is not going to enumerate
        `quant` before dropping in a one-line idea — so the incompleteness is REPORTED
        instead, and it is what an agent should resolve before treating "no match" as
        "nobody has tried this".

        Without this, "genuinely new" and "under-specified" are the same empty tuple,
        which is precisely the confusion this whole module exists to end.
        """
        return tuple(e for e in self.near_misses if e.undeclared_dimensions)

    @property
    def undeclared_dimensions(self) -> tuple:
        """Every dimension some near-miss was measured in and the question did not
        state, deduplicated — the shortest list of things worth saying."""
        found: list = []
        for row in self.incomplete_comparisons:
            for dimension in row.undeclared_dimensions:
                if dimension not in found:
                    found.append(dimension)
        return tuple(sorted(found))

    @property
    def unanswerable(self) -> tuple:
        """Why NO entry in this ledger could have matched, whatever it holds.

        The HARD half of `why_not_comparable`: the question was not compared and could
        not have been, so its empty result carries no information at all. Separated
        from the incomplete-comparison case because the two deserve different force —
        `matches_for()` REFUSES this one (an empty sequence would be a false statement)
        while an incomplete comparison is a real comparison that is worth reporting and
        not worth refusing.
        """
        query = self.query
        if query is None:
            return (
                "the lookup carries no query, so nothing can be said about what it "
                "compared",
            )
        reasons: list = []
        if not query.target.is_identified:
            reasons.append(
                "the question names no 'mechanism', which is the key every match is "
                "made on, so NO entry in this ledger could have matched it. An empty "
                "result here is 'not comparable', never 'not previously tried'"
            )
        if query.unusable_dimensions:
            reasons.append(
                f"the question named {list(query.unusable_dimensions)} and the matcher "
                "could not use "
                + ("them" if len(query.unusable_dimensions) > 1 else "it")
                + f"; see {list(self.ignored_dimensions)}. A dimension that drops out "
                "makes the comparison LOOSER, and looser is the direction that fails "
                "silently"
            )
        return tuple(reasons)

    @property
    def why_not_comparable(self) -> tuple:
        """Every reason "no match" here does NOT mean "nobody has tried this".

        THIS IS THE CHANNEL THE WHOLE MODULE TURNED ON AND THEN DID NOT CONNECT.
        `matches_for()` returns matches and nothing else, so a question the matcher
        could not compare AT ALL came back as an empty sequence — and an empty sequence
        is, by `check_do_not_repeat()`'s own definition, *"it WAS consulted and matched
        nothing"*, i.e. PASS. A question that could not be compared was therefore
        reported as compared and clear.

        That is not a corner case, it is the operator's case. An operator drops in a
        one-line idea with a regime like `{"backend": "llama_gpu", "phase": "decode"}`
        and does not write a `mechanism` — the key every match keys on — so EVERY
        operator hypothesis got a clean PASS from a ledger holding a receipted negative
        about the very same idea in the very same regime. Unreadable memory read as
        empty memory, silently, on exactly the path this package's hypothesis work
        exists to serve.

        Three reasons, all of them "you cannot conclude anything from the empty result
        above", none of them a rejection:

        1. the question names no mechanism, so no entry could match it;
        2. it named a match dimension the matcher could not use;
        3. entries about THIS mechanism were compared and broke only on dimensions the
           question did not state (`incomplete_comparisons`).

        `disposition()` turns a non-empty answer here into COULD_NOT_CHECK, and
        `matches_for()` refuses outright on the `unanswerable` half. Neither ever turns
        a FAIL into anything: a receipted negative is a fact, and an incomplete question
        does not excuse it.
        """
        reasons: list = list(self.unanswerable)
        if self.incomplete_comparisons:
            reasons.append(
                f"{len(self.incomplete_comparisons)} entr"
                + ("ies" if len(self.incomplete_comparisons) > 1 else "y")
                + " about this mechanism were compared and broke only on "
                f"{list(self.undeclared_dimensions)}, which the question does not "
                "state; state them and ask again before treating this as new"
            )
        return tuple(reasons)

    def to_dict(self) -> dict:
        return {
            "matches": [m.to_dict() for m in self.matches],
            "explanations": [e.to_dict() for e in self.explanations],
            "near_misses": [e.to_dict() for e in self.near_misses],
            "incomplete_comparisons": [
                e.entry_id for e in self.incomplete_comparisons
            ],
            "undeclared_dimensions": list(self.undeclared_dimensions),
            "why_not_comparable": list(self.why_not_comparable),
            "ignored_dimensions": list(self.ignored_dimensions),
            "query": self.query.to_dict() if self.query is not None else None,
        }


# =============================================================================
# The compiled ledger
# =============================================================================

class CompiledLedger:
    """The §19.2 do-not-repeat ledger, compiled from the record.

    Implements `hypotheses.DoNotRepeatLedger`. Frozen and derived: it holds no state
    that is not a function of the events it was folded from, so it cannot be written
    back over the record and a stale one is a re-fold away rather than a corruption.
    """

    __slots__ = ("_attempts", "_current_anchor", "_satisfied", "_unusable")

    def __init__(
        self,
        attempts: Sequence = (),
        *,
        current_anchor: Optional[Any] = None,
        satisfied_reopen_predicates: frozenset = frozenset(),
        unusable: Sequence = (),
    ) -> None:
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, PriorAttempt):
                raise LedgerFoldError(
                    f"attempts[{index}]: expected a PriorAttempt, got "
                    f"{type(attempt).__name__}"
                )
        if current_anchor is not None and not isinstance(
            current_anchor, evaluator_api.AnchorIdentity
        ):
            raise LedgerFoldError(
                "current_anchor must be an evaluator.api.AnchorIdentity or None"
            )
        if not isinstance(satisfied_reopen_predicates, frozenset):
            raise LedgerFoldError("satisfied_reopen_predicates must be a frozenset")
        self._attempts = tuple(attempts)
        self._current_anchor = current_anchor
        self._satisfied = satisfied_reopen_predicates
        self._unusable = tuple(unusable)

    # ---- position ---------------------------------------------------------

    @property
    def attempts(self) -> tuple:
        return self._attempts

    @property
    def current_anchor(self):
        return self._current_anchor

    @property
    def unusable(self) -> tuple:
        """Entries that were compiled but can never match, each with its reason.

        Reported rather than dropped: a constraint that names no mechanism is a
        constraint somebody wrote and believes is in force, and silently discarding it
        is how a ledger looks populated while suppressing nothing.
        """
        return self._unusable

    def __len__(self) -> int:
        return len(self._attempts)

    # ---- the protocol -----------------------------------------------------

    def matches_for(self, regime: Mapping[str, Any], statement: str) -> tuple:
        """`hypotheses.DoNotRepeatLedger` — the matches for one question.

        `statement` is accepted because the protocol passes one and is NOT READ. It is
        type-checked (a caller passing something else has a bug worth surfacing) and
        then discarded: `MatchQuery` is built from `regime` alone and has no field that
        could hold it.

        **REFUSES a question it cannot compare** (`LedgerLookup.unanswerable`) instead
        of returning an empty tuple. The protocol's return type is a `Sequence`, so
        there is no value in it that means "I could not tell" — and the empty sequence
        is already taken: `check_do_not_repeat()` defines it as *"it WAS consulted and
        matched nothing"* and returns PASS. Handing back `()` for a question that named
        no mechanism therefore stated, in the consumer's own vocabulary, that a
        receipted negative about the very same idea did not exist. A caller that
        legitimately cannot compare says so with `None`, which is the value
        `check_do_not_repeat()` already maps to COULD_NOT_CHECK; `disposition()` does
        that join for them.
        """
        if not isinstance(statement, str):
            raise TypeError(
                f"statement must be a string, got {type(statement).__name__} — it is "
                "not matched on, but a caller passing a non-string has a bug"
            )
        lookup = self.lookup(regime)
        if lookup.unanswerable:
            raise DoNotRepeatError(
                "this ledger cannot answer that question, and an empty match set would "
                "say it did: " + "; ".join(lookup.unanswerable)
                + ". Use disposition(regime, ledger), or pass matches=None to "
                "check_do_not_repeat(), which is COULD_NOT_CHECK"
            )
        return lookup.matches

    def lookup(self, regime: Mapping[str, Any]) -> LedgerLookup:
        """Every match for one question, WITH the reasoning behind each one."""
        query = MatchQuery.from_regime(regime)
        matches: list = []
        explanations: list = []
        near_misses: list = []
        for attempt in self._attempts:
            agrees, reasons, dimensions, undeclared = self._compare(attempt, query)
            reopened = (
                attempt.reopen_when is not None
                and attempt.reopen_when in self._satisfied
            )
            row = MatchExplanation(
                entry_id=attempt.entry_id,
                entry_class=attempt.entry_class,
                matched=agrees,
                rejects=bool(agrees and attempt.rejects(
                    reopen_predicate_satisfied=reopened
                )),
                reasons=tuple(reasons),
                matched_dimensions=tuple(dimensions),
                undeclared_dimensions=tuple(undeclared),
                event_ids=attempt.event_ids,
                hypothesis_ids=attempt.hypothesis_ids,
                proposal_ids=attempt.proposal_ids,
                receipt=attempt.receipt,
                anchor=attempt.anchor.short() if attempt.anchor is not None else None,
                anchor_outcome=attempt.anchor_outcome,
                conflicted=attempt.conflicted,
                reopen_predicate_satisfied=reopened,
                outcome=attempt.outcome,
            )
            if not agrees:
                near_misses.append(row)
                continue
            explanations.append(row)
            matches.append(hypotheses.LedgerMatch(
                entry_id=attempt.entry_id,
                entry_class=attempt.entry_class,
                match_dimensions={
                    "regime": {k: list(v) for k, v in sorted(attempt.regime_raw.items())},
                    "target": attempt.target.to_dict(),
                    "matched_on": list(dimensions),
                    "outcome": attempt.outcome,
                    "event_ids": list(attempt.event_ids),
                    "why": list(attempt.why),
                },
                receipt=attempt.receipt,
                conflicted=attempt.conflicted,
                reopen_predicate_satisfied=reopened,
            ))
        return LedgerLookup(
            matches=tuple(matches),
            explanations=tuple(explanations),
            near_misses=tuple(near_misses),
            ignored_dimensions=query.ignored,
            query=query,
        )

    def explain(self, regime: Mapping[str, Any]) -> dict:
        """A canonical-JSON-safe account of one lookup, for the record."""
        block = {
            "schema": "epyc.autokernel.do_not_repeat_lookup.v1",
            "entry_count": len(self._attempts),
            "current_anchor": (
                self._current_anchor.short() if self._current_anchor is not None
                else None
            ),
            "satisfied_reopen_predicates": sorted(self._satisfied),
            "unusable_entries": [dict(row) for row in self._unusable],
        }
        block.update(self.lookup(regime).to_dict())
        schemas.canonical_json(block)
        return block

    # ---- matching ---------------------------------------------------------

    def _compare(self, attempt: PriorAttempt, query: MatchQuery) -> tuple:
        """`(agrees, reasons, matched_dimensions)`. The whole matching rule.

        Two gates, in this order:

        1. **structural target** — mechanism equality, then per-set intersection where
           both sides declare a set;
        2. **regime** — EVERY dimension the entry was measured in must be declared by
           the question and share a value with it. `selection.match_ledger` breaks the
           match on an unobserved dimension for the same reason: a dimension the
           question does not state is not a dimension that agrees, and assuming it
           would suppress a regime nobody measured.
        """
        agrees, target_reasons = attempt.target.agreement(query.target)
        if not agrees:
            return False, tuple(target_reasons), (), ()
        reasons = list(target_reasons)
        matched: list = []
        undeclared: list = []
        differing: list = []
        for dimension in sorted(attempt.regime):
            observed = query.regime.get(dimension)
            entry_values = list(attempt.regime_raw.get(dimension, ()))
            if not observed:
                undeclared.append(dimension)
                reasons.append(
                    f"the question does not declare {dimension!r}, which this entry "
                    f"was measured in ({entry_values}); an unobserved dimension is not "
                    "an agreeing one (§19.2)"
                )
                continue
            if not [t for t in attempt.regime[dimension] if t in observed]:
                differing.append(dimension)
                reasons.append(
                    f"different {dimension}: entry {entry_values} vs question "
                    f"{list(query.regime_raw.get(dimension, ()))}"
                )
                continue
            matched.append(dimension)
            reasons.append(f"{dimension} agrees on {entry_values}")
        # A DIFFERENT value outranks a missing one in the explanation: "this question
        # is about another regime" is a stronger statement than "it did not say", and
        # only the second is worth going back and fixing.
        if differing:
            return False, tuple(reasons), (), ()
        if undeclared:
            return False, tuple(reasons), (), tuple(undeclared)
        return True, tuple(reasons), tuple(matched), ()


# =============================================================================
# The fold
# =============================================================================

def _entry_id(kind: str, payload: Mapping[str, Any]) -> str:
    digest = schemas.content_hash(payload)[:12]
    slug = _NON_TOKEN_RE.sub("-", kind.strip().lower()).strip("-") or "entry"
    return f"dnr-{slug}-{digest}"


def _require_sequence(value: Any, what: str, kind) -> tuple:
    if value is None:
        # NOT `()`. `None` is what a failed read defaults to, and `()` is the statement
        # "nothing has been tried" — the same statement this function refuses `""` for
        # making three lines below, and the same one `OperatorHypothesisStore.load()`
        # refuses an unreadable store for making. A caller that genuinely has no
        # records omits the argument; a caller holding `None` does not know whether it
        # has any.
        raise LedgerFoldError(
            f"{what}: None is not an empty sequence. An empty ledger is the statement "
            "'nothing has been tried' and the planner acts on it, so it is written by "
            "omitting this argument — never by passing the value a failed read leaves "
            "behind"
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LedgerFoldError(
            f"{what} must be a sequence of {kind.__name__}, got "
            f"{type(value).__name__}; a bare string would be exploded into one "
            "'record' per character and would fold into an empty ledger, which is the "
            "statement 'nothing has been tried'"
        )
    items = tuple(value)
    for index, item in enumerate(items):
        if not isinstance(item, kind):
            raise LedgerFoldError(
                f"{what}[{index}]: expected {kind.__name__}, got "
                f"{type(item).__name__}"
            )
    return items


def _anchor_of(payload: Mapping[str, Any]):
    block = payload.get("anchor") if isinstance(payload, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    anchor, _ = evaluator_api.AnchorIdentity.parse(block)
    return anchor


def _confounders(event_id: str, payload: Mapping[str, Any]) -> tuple:
    """Every reason this evaluation event is not usable as a result.

    `co_residency` is deliberately NOT consulted. Co-residency is scheduling data, not
    a trust gate (the 2026-07-27 amendment; some lineups are concurrent by design), so
    the trust signal is the void finding the evaluator wrote — an A/A run destroyed by
    seven llama-servers coming up mid-flight surfaces as
    `VOID:CONCURRENT_INFERENCE_CONTAMINATION:…` in `integrity_flags`, which is what
    this reads.
    """
    found: list = []
    status = payload.get("status")
    if status in CONFOUNDING_STATUSES:
        found.append(f"{event_id}: status {status!r} — the run was not a measurement")
    flags = payload.get("integrity_flags")
    if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
        for flag in flags:
            if isinstance(flag, str) and flag.startswith(schemas.VOID_FLAG_PREFIX):
                found.append(
                    f"{event_id}: {flag} — a voided run is no result, so it cannot be "
                    "a negative one"
                )
    return tuple(found)


def _scope_is_partial(payload: Mapping[str, Any]) -> bool:
    scope = payload.get("scope_denominator")
    return isinstance(scope, Mapping) and scope.get("machine_subset") == "partial"


def _merge_dimensions(measured: Any, claimed: Any) -> tuple:
    """`(dimensions, displaced)` — the MEASURED regime, filled in from the CLAIMED one.

    THE MEASURED SIDE WINS OUTRIGHT on any dimension both sides declare, and this is
    the whole point of the function. Unioning the two — which is what it used to do —
    made an entry declare every value either side named, so a hypothesis the operator
    filed under `backend: llama_cpu`, attempted by a proposal that ran in a GPU
    campaign, compiled to `backend: [llama_gpu, llama_cpu]` and a CPU question was then
    rejected by a negative that was never measured on a CPU. That is the silent
    false-suppression this module names as its worse failure, rebuilt inside the fold.

    A disagreement is not a wider measurement. The proposal, its campaign and its
    evaluation events are what RAN; the hypothesis's regime is what somebody BELIEVED
    the question was about, and an entry records what ran. Dimensions the measured side
    does not declare still come from the hypothesis — that only adds constraints, and
    an entry constrained on more dimensions matches fewer questions.

    `displaced` names every dimension where the two disagreed, so the divergence lands
    in the entry's `why` rather than being silently resolved.
    """
    merged: dict = dict(measured or {})
    displaced: list = []
    for name, values in (claimed or {}).items():
        if name in merged:
            if tuple(merged[name]) != tuple(values):
                displaced.append(name)
            continue
        merged[name] = tuple(values)
    return merged, tuple(sorted(displaced))


class _Fold:
    """The fold's working state. Not public: `fold_journal` is the surface."""

    def __init__(self, current_anchor, satisfied):
        self.current_anchor = current_anchor
        self.satisfied = satisfied
        self.campaigns: dict = {}
        self.proposals: dict = {}
        self.candidate_to_proposal: dict = {}
        self.events_by_candidate: dict = {}
        self.events_by_id: dict = {}
        self.skipped: dict = {}
        self.constraints: list = []
        self.attempts: list = []
        self.unusable: list = []

    # ---- indexing ---------------------------------------------------------

    def index(self, entries: Sequence) -> None:
        for entry in entries:
            payload = entry.payload if isinstance(entry.payload, Mapping) else {}
            if entry.kind == journal.KIND_CAMPAIGN_OPENED:
                campaign_id = payload.get("campaign_id") or entry.campaign_id
                if isinstance(campaign_id, str):
                    self.campaigns[campaign_id] = payload
            elif entry.kind == journal.KIND_PROPOSAL_RECORDED:
                pid = payload.get("proposal_id") or entry.record_id
                if isinstance(pid, str):
                    self.proposals[pid] = (entry, payload)
            elif entry.kind == journal.KIND_CANDIDATE_RECORDED:
                cid = payload.get("candidate_id") or entry.record_id
                pid = payload.get("proposal_id")
                if isinstance(cid, str) and isinstance(pid, str):
                    self.candidate_to_proposal[cid] = pid
            elif entry.kind == journal.KIND_EVALUATION_EVENT:
                eid = payload.get("event_id") or entry.record_id or entry.event_id
                cid = payload.get("candidate_id")
                if isinstance(eid, str):
                    self.events_by_id[eid] = (entry, payload)
                if isinstance(cid, str):
                    self.events_by_candidate.setdefault(cid, []).append(
                        (entry, payload)
                    )
            elif entry.kind == journal.KIND_PROPOSAL_SKIPPED:
                ref = payload.get("proposal_ref")
                if isinstance(ref, str):
                    self.skipped.setdefault(ref, []).append((entry, payload))
            elif entry.kind == CONSTRAINT_EVENT_KIND:
                self.constraints.append((entry, payload))

    # ---- facets -----------------------------------------------------------

    def proposal_facets(self, proposal_id: Optional[str]) -> tuple:
        """`(regime, regime_raw, target, ignored, campaign_id)` for one proposal.

        READ FROM WHERE THE PACKAGE ALREADY PUTS THEM. `fingerprint.py` exists because
        two modules computed a proposal's identity differently and neither could see
        the disagreement, so this reads exactly the fields `fingerprint.
        mechanism_facets()` reads and from the same places:

        * `selection.mechanism` (§7.1's planner-authored block — the key
          `selection.screen_proposal` requires and `selection.match_ledger` gates on
          first) is the structural target;
        * `selection.regime_identity` and `selection.hierarchy_layer` are the declared
          regime;
        * `target.{regimes,shapes,models,ops}`, `change_class` and
          `change.files_and_symbols` fill in the rest.

        Values go through THIS module's `canonical_token`, not `fingerprint.
        canonical_items`: the latter emits `canonical_json` text (`'"decode"'`), which
        is right for a digest and wrong for a comparison against a hypothesis regime
        written by hand (`decode`) — two encodings of one value never match, and a
        suppression that silently never matches is the toothless-ledger failure.

        Backend comes off the CAMPAIGN record (§7.1), which is the only place it is
        written. Without it a GPU negative would match a CPU question, and backend is
        the single most sign-changing dimension this project has.
        """
        if proposal_id is None or proposal_id not in self.proposals:
            return {}, {}, StructuralTarget(), (), None
        _, payload = self.proposals[proposal_id]
        block = fingerprint.selection_block(payload)
        target_block = payload.get("target")
        target_block = dict(target_block) if isinstance(target_block, Mapping) else {}
        campaign_id = payload.get("campaign_id")
        campaign = self.campaigns.get(campaign_id, {})

        source: dict = dict(target_block)
        identity = block.get("regime_identity")
        if isinstance(identity, Mapping):
            source.update(identity)
        declared = payload.get("regime")
        if isinstance(declared, Mapping):
            source.update(declared)
        for key, value in (
            ("mechanism", block.get("mechanism") or payload.get("mechanism")),
            ("hierarchy_layer", block.get("hierarchy_layer")),
            ("change_class", payload.get("change_class")),
            ("backend", campaign.get("backend")),
        ):
            if value is not None and key not in source:
                source[key] = value
        change = payload.get("change")
        if isinstance(change, Mapping) and "symbols" not in source:
            source["symbols"] = change.get("files_and_symbols")
        regime, regime_raw, target, ignored = read_facets(source)
        return regime, regime_raw, target, ignored, campaign_id

    def evidence_for(self, proposal_id: Optional[str], extra_event_ids: Sequence) -> tuple:
        """`(event_ids, candidate_ids, anchors, confounders, partial, statuses)`.

        `statuses` is a tuple of `(tier, status)` PAIRS, not bare statuses. The tier is
        what tells a GATE failure from a MEASURED one — `schemas` says it outright,
        *"only T0 compares artifacts rather than rates"* — and a fold that cannot see it
        reads a build break as a negative about the mechanism.

        The join that turns "a proposal was tried" into "and here is how it ended":
        proposal -> candidate (`CANDIDATE_RECORDED.proposal_id`) -> evaluation events,
        plus any event a resolution names directly in its `evidence_refs`.

        `anchors` is one slot PER CONTRIBUTING EVENT, `None` where an event names none
        — not one anchor picked off whichever event sorted first. A negative built from
        several measurements is only current if EVERY one of them is, and a single
        chosen anchor would let a v8 T0 vouch for a v7 T1 sitting beside it.
        """
        seen: dict = {}
        candidates: list = []
        for candidate_id, mapped in sorted(self.candidate_to_proposal.items()):
            if proposal_id is not None and mapped == proposal_id:
                candidates.append(candidate_id)
                for entry, payload in self.events_by_candidate.get(candidate_id, []):
                    eid = payload.get("event_id") or entry.event_id
                    seen[eid] = payload
        for ref in extra_event_ids:
            if isinstance(ref, str) and ref in self.events_by_id:
                seen[ref] = self.events_by_id[ref][1]
        anchors: list = []
        confounders: list = []
        partial = False
        statuses: list = []
        for eid in sorted(seen):
            payload = seen[eid]
            confounders.extend(_confounders(eid, payload))
            partial = partial or _scope_is_partial(payload)
            status = payload.get("status")
            if isinstance(status, str):
                tier = payload.get("tier")
                statuses.append((tier if isinstance(tier, str) else None, status))
            anchors.append(_anchor_of(payload))
        return (
            tuple(sorted(seen)), tuple(candidates), tuple(anchors),
            tuple(confounders), partial, tuple(statuses),
        )

    # ---- classification ---------------------------------------------------

    def anchor_verdict(self, anchors: Sequence) -> tuple:
        """`(outcome, reason)` for "was ALL of this taken against the CURRENT anchor?".

        The three outcomes of `AnchorIdentity.identity_matches` all survive, and the
        two that are not PASS both demote: an anchor that moved is a stale premise, and
        an anchor that cannot be COMPARED is an unobserved component, which is never a
        PASS (`state_machine.check_anchor_identity`'s rule).
        """
        if not anchors:
            return schemas.COULD_NOT_CHECK, (
                "the negative names no anchor, so it cannot be shown to have been "
                "taken against the current one; an idea that failed on v7 may win on "
                "v8 and an unobserved anchor is never a PASS"
            )
        if any(anchor is None for anchor in anchors):
            return schemas.COULD_NOT_CHECK, (
                f"{sum(1 for a in anchors if a is None)} of the {len(anchors)} "
                "measurements behind this negative name no anchor, so the negative "
                "cannot be shown to bind to the current one"
            )
        if self.current_anchor is None:
            return schemas.COULD_NOT_CHECK, (
                "the ledger was compiled without a current anchor, so no negative can "
                "be shown to still bind; nothing here rejects on measurement grounds"
            )
        reasons: list = []
        outcome = schemas.PASS
        for anchor in anchors:
            check = anchor.identity_matches(self.current_anchor)
            if check.outcome == schemas.PASS:
                continue
            reasons.extend(check.reasons)
            # A detected difference is a FACT and outranks an incomplete observation.
            if check.outcome == schemas.FAIL or outcome == schemas.PASS:
                outcome = check.outcome
        if outcome == schemas.PASS:
            return schemas.PASS, (
                f"all {len(anchors)} measurement(s) were taken against the current "
                "anchor"
            )
        return outcome, "; ".join(reasons) or "anchor comparison failed"

    def classify(self, *, outcome, confounders, partial, anchors, has_regime) -> tuple:
        """`(entry_class, why, anchor_outcome)` for one folded attempt.

        Every condition that can DEMOTE is evaluated, and `CLASS_PRECEDENCE` picks the
        most severe of what was found — with `MATCHED_NEGATIVE` last, so it survives
        only when nothing else fired.
        """
        found: set = set()
        why: list = []
        anchor_outcome = None
        if confounders:
            found.add(hypotheses.MATCH_CLASS_CONFOUNDED_RESULT)
            why.extend(confounders)
            why.append(
                "a result taken under a voided window is not a negative result, it is "
                "no result; §19.2 requires a repaired experiment and the question "
                "stays open"
            )
        if outcome == hypotheses.RESOLUTION_REFUTED:
            anchor_outcome, anchor_why = self.anchor_verdict(anchors)
            if anchor_outcome != schemas.PASS:
                found.add(hypotheses.MATCH_CLASS_SUPERSEDED_FACT)
                why.append(f"anchor: {anchor_why}")
            else:
                why.append(f"anchor: {anchor_why}")
            if partial:
                found.add(hypotheses.MATCH_CLASS_CONDITIONAL_NEGATIVE)
                why.append(
                    "the negative was measured on a PARTIAL machine subset, so it "
                    "excludes the cells it names and cannot close the question "
                    "(a full-machine claim on a partial cell is a category error)"
                )
            if not has_regime:
                found.add(hypotheses.MATCH_CLASS_CONDITIONAL_NEGATIVE)
                why.append(
                    "no regime identity was recorded for this negative, so it can "
                    "only exclude the cells it names (§19.2: 'do not repeat' without "
                    "regime identity is dangerous)"
                )
            found.add(hypotheses.MATCH_CLASS_MATCHED_NEGATIVE)
            why.append("the hypothesis was REFUTED against its falsifier")
        elif outcome == hypotheses.RESOLUTION_CONFIRMED:
            found.add(hypotheses.MATCH_CLASS_LOW_VALUE)
            why.append(
                "already CONFIRMED under matching conditions; re-establishing it buys "
                "no information, which deprioritizes it and closes nothing"
            )
        elif outcome == OUTCOME_SKIPPED:
            found.add(hypotheses.MATCH_CLASS_LOW_VALUE)
            why.append(
                "the proposal was SKIPPED rather than executed, so nothing was learned "
                "about the question; §19.2 deprioritizes, it does not close"
            )
        # NOTHING matched, and that is a real answer: an open question, an
        # `inconclusive` resolution ("the experiment ran and did not resolve"), or an
        # attempt with no result yet. The ledger says nothing about any of them, and
        # saying nothing is not the same as saying "not tried" — the caller sees the
        # attempt in the journal either way.
        if not found:
            return None, tuple(why), anchor_outcome
        for entry_class in CLASS_PRECEDENCE:
            if entry_class in found:
                return entry_class, tuple(why), anchor_outcome
        raise LedgerFoldError(  # pragma: no cover - CLASS_PRECEDENCE covers MATCH_CLASSES
            f"no precedence for {sorted(found)}"
        )

    # ---- emission ---------------------------------------------------------

    def emit(self, **fields) -> None:
        payload = {
            "regime": {k: list(v) for k, v in sorted(fields["regime"].items())},
            "mechanism": fields["target"].mechanism,
            "outcome": fields.get("outcome"),
            "event_ids": sorted(fields.get("event_ids") or ()),
            "hypothesis_ids": sorted(fields.get("hypothesis_ids") or ()),
            "proposal_ids": sorted(fields.get("proposal_ids") or ()),
            "class": fields["entry_class"],
        }
        self.attempts.append(PriorAttempt(
            entry_id=_entry_id(fields["entry_class"], payload), **fields
        ))


def _receipt(event_ids: Sequence, evidence_refs: Sequence, anchor) -> Optional[str]:
    """§19.3's receipt: resolvable ids, bound to the anchor they were taken against.

    `None` when there is nothing a reader could resolve — and `None` is not a
    formality: `check_do_not_repeat()` turns a receipt-less rejecting match into
    COULD_NOT_CHECK rather than a rejection, which is exactly the behaviour a
    suppression nobody can verify should have. A receipt is never SYNTHESISED to make a
    match bite.
    """
    ids = sorted({r for r in list(event_ids) + list(evidence_refs) if isinstance(r, str) and r.strip()})
    if not ids:
        return None
    bound = anchor.short() if anchor is not None else "no-anchor"
    return f"{'+'.join(ids)} @ {bound}"


def fold_journal(
    *,
    journal_entries: Sequence = (),
    hypothesis_events: Sequence = (),
    current_anchor: Optional[Any] = None,
    satisfied_reopen_predicates: frozenset = frozenset(),
) -> CompiledLedger:
    """Fold the record into a do-not-repeat ledger. PURE.

    Reads, in one pass:

    * `HYPOTHESIS_OPENED / ATTEMPTED / RESOLVED / REOPENED` — through
      `hypotheses.fold_ledger()`, which owns every legality rule about that history and
      raises `HypothesisLedgerCorruption` on a contradiction. Re-implementing the fold
      here would be a second opinion about one record.
    * `PROPOSAL_RECORDED` — the structural target and the declared regime;
    * `CANDIDATE_RECORDED` — the proposal→candidate join;
    * `EVALUATION_EVENT` — how it ended: status, integrity flags, scope, ANCHOR;
    * `PROPOSAL_SKIPPED` — tried-and-not-run, which is LOW_VALUE and never a negative;
    * `CONSTRAINT_COMPILED` — §19.4 compiled constraints, the HARD_CONSTRAINT source.

    THREE THINGS IT DELIBERATELY DOES NOT EMIT AN ENTRY FOR:

    1. **an attempt that did not bear on the falsifier.** §8.4.0 exists because a
       hypothesis used to evaporate when its proposal was dispositioned *including when
       that proposal failed for an unrelated reason* — a build break, a skipped
       proposal, a voided window. Folding those into a negative would rebuild that
       defect one layer up.
    2. **an `inconclusive` resolution.** The experiment ran and did not resolve; the
       question is open and the ledger says nothing about it.
    3. **a hypothesis with no mechanism declared.** It cannot be matched on, so it is
       reported in `CompiledLedger.unusable` rather than compiled into an entry that
       would match on regime alone.

    `current_anchor=None` is a real position and not a default to be papered over: with
    no current anchor NOTHING can be shown to still bind, so every measurement-derived
    negative folds to `SUPERSEDED_FACT` and the ledger rejects only on hard
    constraints.
    """
    entries = _require_sequence(journal_entries, "journal_entries", journal.JournalEntry)
    events = _require_sequence(hypothesis_events, "hypothesis_events", hypotheses.LedgerEvent)
    if current_anchor is not None and not isinstance(
        current_anchor, evaluator_api.AnchorIdentity
    ):
        raise LedgerFoldError(
            "current_anchor must be an evaluator.api.AnchorIdentity or None; a string "
            "commit would compare unequal to every anchor and silently supersede the "
            "whole ledger"
        )
    if not isinstance(satisfied_reopen_predicates, frozenset):
        raise LedgerFoldError("satisfied_reopen_predicates must be a frozenset")

    fold = _Fold(current_anchor, satisfied_reopen_predicates)
    fold.index(entries)
    tracked = hypotheses.fold_ledger(events)

    _fold_constraints(fold)
    _fold_hypotheses(fold, tracked)
    _fold_orphan_proposals(fold, tracked)
    _mark_conflicted(fold)

    return CompiledLedger(
        tuple(fold.attempts),
        current_anchor=current_anchor,
        satisfied_reopen_predicates=satisfied_reopen_predicates,
        unusable=tuple(fold.unusable),
    )


def _fold_constraints(fold: _Fold) -> None:
    """`CONSTRAINT_COMPILED` events -> compiled §19.2 entries.

    HARD_CONSTRAINT by default: a hardware, policy, correctness or ownership
    prohibition, which is not measurement-derived, so no anchor move and no contended
    window weakens it. It still needs a §19.3 receipt to REJECT — a prohibition nobody
    can look up leaves `check_do_not_repeat()` at COULD_NOT_CHECK.

    A constraint MAY declare another class, because §19.2's constraint ledger holds all
    six and `selection.LedgerEntry` reads exactly that corpus: a negative imported from
    the §19.4 bootstrap or compiled by a memory update is a `MATCHED_NEGATIVE` with a
    `reopen_when` predicate, and refusing to compile one here would leave the consumer's
    reopen branch — `check_do_not_repeat`'s "a MATCHED_NEGATIVE whose reopen predicate
    is newly satisfied does not reject" — unreachable from its only producer. Two §19.3
    rules ride with it: an entry declaring `anchor_commit` that is not the current one
    is a SUPERSEDED_FACT (`LedgerEntry.authoritative_against`, verbatim), and a
    `conflicted` entry is carried as conflicted rather than dropped.
    """
    for entry, payload in fold.constraints:
        source = dict(payload.get("match_dimensions") or {})
        for key in ("mechanism", "regime", "backend", "phase", "ops", "change_class"):
            if key in payload and key not in source:
                source[key] = payload[key]
        if isinstance(payload.get("regime"), Mapping):
            source.pop("regime", None)
            source.update(payload["regime"])
        regime, regime_raw, target, ignored = read_facets(source)
        receipt = payload.get("receipt")
        receipt = receipt if isinstance(receipt, str) and receipt.strip() else None
        if not target.is_identified:
            fold.unusable.append({
                "event_id": entry.event_id,
                "kind": CONSTRAINT_EVENT_KIND,
                "reason": (
                    "the constraint names no mechanism, so nothing can be matched "
                    "against it; it is in force in prose and enforces nothing here"
                ),
            })
            continue
        declared_class = payload.get("entry_class", hypotheses.MATCH_CLASS_HARD_CONSTRAINT)
        if declared_class not in hypotheses.MATCH_CLASSES:
            fold.unusable.append({
                "event_id": entry.event_id,
                "kind": CONSTRAINT_EVENT_KIND,
                "reason": (
                    f"entry_class {declared_class!r} is not one of "
                    f"{sorted(hypotheses.MATCH_CLASSES)}; the §19.2 vocabulary is "
                    "closed, and an entry outside it would be a suppression with no "
                    "declared planner behaviour"
                ),
            })
            continue
        why = [
            "a compiled §19.2 constraint-ledger entry"
            if declared_class != hypotheses.MATCH_CLASS_HARD_CONSTRAINT else
            "a compiled §19.2 hard constraint: a hardware, policy, correctness or "
            "ownership prohibition, which no measurement confounds and no anchor "
            "move supersedes"
        ]
        entry_class = declared_class
        anchor_commit = payload.get("anchor_commit")
        if (declared_class in hypotheses.REJECTING_MATCH_CLASSES
                and declared_class != hypotheses.MATCH_CLASS_HARD_CONSTRAINT):
            # The anchor is checked for ABSENCE as well as for movement. A constraint
            # record declaring MATCHED_NEGATIVE is a MEASUREMENT-derived suppression
            # however it arrived, and on the measurement path `anchor_verdict()` treats
            # "names no anchor" as COULD_NOT_CHECK -> SUPERSEDED_FACT, because an
            # unobserved component is never a PASS and so can never be the thing that
            # closes a question. Requiring only that a DECLARED commit still match left
            # the opposite rule on this path: an imported negative that named no commit
            # at all rejected forever — at v8, at v7, and even with no current anchor,
            # which is the position the fold's own docstring says rejects nothing on
            # measurement grounds. A stale anchor permanently closing a question is
            # exactly what ANCHOR SENSITIVITY above exists to prevent.
            current = (
                fold.current_anchor.source_commit
                if fold.current_anchor is not None else None
            )
            if not isinstance(anchor_commit, str) or not anchor_commit.strip():
                entry_class = hypotheses.MATCH_CLASS_SUPERSEDED_FACT
                why.append(
                    "anchor: the entry names no 'anchor_commit', so it cannot be shown "
                    "to have been taken against the current one; an idea that failed on "
                    "v7 may win on v8 and an unobserved anchor is never a PASS (§19.3)"
                )
            elif anchor_commit != current:
                # `current is None` needs no branch of its own: a commit string is
                # never equal to `None`, so "no current anchor" arrives here and
                # supersedes, which is the position the fold's docstring declares.
                entry_class = hypotheses.MATCH_CLASS_SUPERSEDED_FACT
                where = (
                    current[:12] if current
                    else "unknown (the ledger was compiled without one)"
                )
                why.append(
                    f"anchor: the entry binds to {anchor_commit[:12]} and the current "
                    f"anchor is {where}; §19.3 — a suppression whose receipt no longer "
                    "resolves reverts rather than continuing to block"
                )
        reopen = payload.get("reopen_when")
        fold.emit(
            entry_class=entry_class,
            regime=regime, regime_raw=regime_raw, target=target,
            outcome=None,
            event_ids=(entry.event_id,),
            receipt=receipt,
            anchor=None,
            anchor_outcome=None,
            conflicted=payload.get("conflicted") is True,
            reopen_when=reopen if isinstance(reopen, str) and reopen.strip() else None,
            why=tuple(why),
            ignored=ignored,
        )


def _fold_hypotheses(fold: _Fold, tracked: Mapping[str, Any]) -> None:
    """Tracked questions -> entries, one per bearing attempt (or one per resolution)."""
    for hypothesis_id in sorted(tracked):
        state = tracked[hypothesis_id]
        resolution = state.resolution
        outcome = resolution.outcome if resolution is not None else None
        evidence_refs = list(resolution.evidence_refs) if resolution is not None else []
        bearing = [a for a in state.attempts if a.bears_on_falsifier]
        # Deduplicated, order preserved: two ATTEMPT events against one proposal are
        # two receipts for one thing tried, and emitting an entry per receipt would
        # put two rows with one content-addressed `entry_id` in front of a reader.
        proposals = list(dict.fromkeys(a.proposal_id for a in bearing)) or [None]
        for proposal_id in proposals:
            regime, regime_raw, target, ignored, _ = fold.proposal_facets(proposal_id)
            h_regime, h_regime_raw, h_target, h_ignored = read_facets(
                state.hypothesis.regime
            )
            merged, displaced = _merge_dimensions(regime, h_regime)
            merged_raw, _ = _merge_dimensions(regime_raw, h_regime_raw)
            if not target.is_identified:
                target = h_target
            attempt_outcome = outcome
            if proposal_id is not None and proposal_id in fold.skipped and outcome is None:
                attempt_outcome = OUTCOME_SKIPPED
            (event_ids, candidate_ids, anchors, confounders, partial,
             _statuses) = fold.evidence_for(proposal_id, evidence_refs)
            anchor = next((a for a in anchors if a is not None), None)
            # NOTE what is NOT filtered here: an unresolved or `inconclusive` question
            # is not skipped at this point, because `classify` is the ONE place that
            # decides whether a history is a ledger entry. A second filter here would
            # be a second opinion — and it would have to be kept in step with the
            # first, which is how the A/A run that seven llama-servers destroyed
            # mid-flight would go missing: it resolved NOTHING (correctly — a voided
            # window cannot resolve a falsifier), so it arrives on exactly the branch a
            # "skip the unresolved" shortcut would drop.
            entry_class, why, anchor_outcome = fold.classify(
                outcome=attempt_outcome, confounders=confounders, partial=partial,
                anchors=anchors, has_regime=bool(merged),
            )
            if entry_class is None:
                continue
            if displaced:
                why = why + tuple(
                    f"regime {name!r}: the question was FILED under "
                    f"{list(h_regime_raw.get(name, ()))} and MEASURED at "
                    f"{list(regime_raw.get(name, ()))}; the entry records what ran, so "
                    "the measured value stands and this negative says nothing about "
                    "the other one"
                    for name in displaced
                )
            if not target.is_identified:
                fold.unusable.append({
                    "event_id": hypothesis_id,
                    "kind": hypotheses.EVENT_RESOLVED,
                    "reason": (
                        "the question declares no 'mechanism' in its regime, so its "
                        "outcome cannot be matched against a later question; a "
                        "regime-only entry would suppress every idea in that regime"
                    ),
                })
                continue
            fold.emit(
                entry_class=entry_class,
                regime=merged, regime_raw=merged_raw, target=target,
                outcome=attempt_outcome,
                event_ids=event_ids,
                hypothesis_ids=(hypothesis_id,),
                proposal_ids=(proposal_id,) if proposal_id else (),
                candidate_ids=candidate_ids,
                receipt=_receipt(event_ids, evidence_refs, anchor)
                if entry_class in hypotheses.REJECTING_MATCH_CLASSES else None,
                anchor=anchor,
                anchor_outcome=anchor_outcome,
                reopen_when=None,
                why=why,
                ignored=tuple(ignored) + tuple(h_ignored),
            )


def _orphan_outcome(statuses: Sequence, confounders: Sequence) -> tuple:
    """`(outcome, conflicted, why)` from one proposal's evaluation history alone.

    An orphan has no falsifier, so `bears_on_falsifier` — the §8.4.0 field that keeps a
    hypothesis from evaporating when its proposal failed for an unrelated reason — is
    not available here. The tier is, and it carries the same distinction:

    * **`T0` is a GATE, not a measurement.** `schemas._check_anchor_measurement_ids`
      says so in the validator itself — *"only T0 compares artifacts rather than
      rates"*. A T0 failure is symbol preservation or a clean snapshot build failing
      on ONE implementation of an idea; it is not evidence about the MECHANISM, and
      folding it into a receipted `MATCHED_NEGATIVE` closes a research family on a
      build break. That is §8.4.0's original defect rebuilt one layer up: a proposal
      that failed T0, was repaired, and then PASSED T1 used to compile to
      `MATCHED_NEGATIVE` and reject the very idea it had just been shown to support.
    * **a history that both passed and failed at measurement tier is CONFLICTED**, not
      a negative. §19.3's rule already exists for this and this reuses it rather than
      inventing a seventh disposition: the entry is carried, and a suppression whose
      evidence disagrees with itself is never authoritative.
    """
    measured = [(tier, status) for tier, status in statuses if tier != "T0"]
    gate_failures = [
        tier for tier, status in statuses if tier == "T0" and status == "fail"
    ]
    outcomes = {status for _tier, status in measured}
    if "fail" in outcomes and "pass" in outcomes:
        return hypotheses.RESOLUTION_REFUTED, True, (
            "this proposal's measurement events both PASSED and FAILED; the record "
            "disagrees with itself about one candidate, and §19.3 carries such an "
            "entry as conflicted rather than letting it suppress anything",
        )
    if "fail" in outcomes:
        return hypotheses.RESOLUTION_REFUTED, False, ()
    if outcomes and outcomes <= {"pass"}:
        return hypotheses.RESOLUTION_CONFIRMED, False, ()
    if gate_failures and not measured:
        # Nothing was measured. This proposal never became evidence about its
        # mechanism, so the ledger says nothing about it — LOW_VALUE deprioritizes,
        # which is what §19.2 does with a thing that taught us nothing.
        return OUTCOME_SKIPPED, False, (
            f"the only failing evaluation event(s) were at tier {sorted(set(gate_failures))} "
            "— a T0 GATE, which compares artifacts rather than rates. The candidate "
            "never produced a measurement, so nothing was learned about the mechanism "
            "and this is not a negative about it",
        )
    if confounders:
        # Ran, produced no usable number. `classify` folds this to CONFOUNDED_RESULT
        # on the confounders alone; there is no outcome to state, and stating one
        # would invent a result the window did not produce.
        return None, False, ()
    return False, False, ()  # `False` — no entry at all, distinct from `None`


def _fold_orphan_proposals(fold: _Fold, tracked: Mapping[str, Any]) -> None:
    """Proposals nobody opened a hypothesis for — most of what a loop actually tries.

    Their outcome comes from the evaluation events alone, through `_orphan_outcome`:
    a MEASURED `fail` is the negative, a `PROPOSAL_SKIPPED` or a T0-gate-only failure
    is LOW_VALUE, and everything else leaves no entry.
    """
    claimed = {
        a.proposal_id for state in tracked.values() for a in state.attempts
    }
    for proposal_id in sorted(fold.proposals):
        if proposal_id in claimed:
            continue
        regime, regime_raw, target, ignored, _ = fold.proposal_facets(proposal_id)
        (event_ids, candidate_ids, anchors, confounders, partial,
         statuses) = fold.evidence_for(proposal_id, ())
        anchor = next((a for a in anchors if a is not None), None)
        conflicted = False
        extra_why: tuple = ()
        if proposal_id in fold.skipped and not statuses:
            outcome = OUTCOME_SKIPPED
            event_ids = event_ids + tuple(
                e.event_id for e, _ in fold.skipped[proposal_id]
            )
        else:
            outcome, conflicted, extra_why = _orphan_outcome(statuses, confounders)
            if outcome is False:
                continue
        entry_class, why, anchor_outcome = fold.classify(
            outcome=outcome, confounders=confounders, partial=partial, anchors=anchors,
            has_regime=bool(regime),
        )
        why = why + extra_why
        if entry_class is None or not target.is_identified:
            if entry_class is not None:
                fold.unusable.append({
                    "event_id": proposal_id,
                    "kind": journal.KIND_PROPOSAL_RECORDED,
                    "reason": (
                        "the proposal declares no 'mechanism', so what it changed "
                        "cannot be compared against a later question"
                    ),
                })
            continue
        fold.emit(
            entry_class=entry_class,
            regime=regime, regime_raw=regime_raw, target=target,
            outcome=outcome,
            event_ids=tuple(sorted(event_ids)),
            proposal_ids=(proposal_id,),
            candidate_ids=candidate_ids,
            receipt=_receipt(event_ids, (), anchor)
            if entry_class in hypotheses.REJECTING_MATCH_CLASSES else None,
            anchor=anchor,
            anchor_outcome=anchor_outcome,
            conflicted=conflicted,
            why=why,
            ignored=ignored,
        )


def _mark_conflicted(fold: _Fold) -> None:
    """§19.3: an entry that contradicts another about one question is never authoritative.

    Two entries with the same mechanism and the same regime that ended in DIFFERENT
    outcomes are a contradiction, and a contradiction does not get to suppress anything
    — a suppression whose evidence disagrees with itself is the shape §19.3 makes
    revert to `conflicted` rather than continue to block.
    """
    groups: dict = {}
    for index, attempt in enumerate(fold.attempts):
        if attempt.outcome is None:
            continue
        key = (
            attempt.target.mechanism,
            tuple(sorted((k, tuple(v)) for k, v in attempt.regime.items())),
        )
        groups.setdefault(key, []).append(index)
    for indexes in groups.values():
        outcomes = {fold.attempts[i].outcome for i in indexes}
        if len(outcomes) < 2:
            continue
        for i in indexes:
            attempt = fold.attempts[i]
            fold.attempts[i] = dataclasses.replace(
                attempt,
                conflicted=True,
                why=attempt.why + (
                    "CONFLICTED: this mechanism has ended in "
                    f"{sorted(outcomes)} under one regime, and a suppression whose "
                    "evidence disagrees with itself is never authoritative (§19.3)",
                ),
            )


# =============================================================================
# The disposition — the one surface a caller should ask
# =============================================================================

def disposition(regime: Mapping[str, Any], ledger: "CompiledLedger") -> schemas.Check:
    """`check_do_not_repeat()`'s verdict, WITH the one thing it cannot see.

    `check_do_not_repeat()` takes a regime and a sequence of matches, and it is right
    about every match it is given. What it cannot know — because a `Sequence` has no
    room to say it — is whether the question was ANSWERABLE. An empty sequence reaches
    it as "consulted, matched nothing" and comes back PASS, whether the ledger was
    empty or the query could not be compared against a single entry in it.

    This function is the join. It asks the ledger the question, reads
    `LedgerLookup.why_not_comparable`, and demotes **PASS and only PASS** to
    COULD_NOT_CHECK when the empty answer cannot bear the weight of "nobody has tried
    this". Everything else is `check_do_not_repeat()`'s, verbatim and unaltered:

    * FAIL is returned untouched. A receipted negative is a fact, and an
      under-specified question is not a defence against one — the same precedence
      `check_do_not_repeat` itself applies when it puts a concrete match ahead of an
      incompleteness.
    * COULD_NOT_CHECK stays COULD_NOT_CHECK, with the extra reasons appended so a
      reader learns everything that was wrong at once rather than one round at a time.
    * PASS survives only when the question could actually be compared — which is what
      makes a PASS mean something.

    The class dispositions are NOT restated here. This module produces the six §19.2
    classes and the consumer disposes of them; a second table would be a second opinion
    about what closes a question, and that is the defect shape this package keeps
    paying for.
    """
    if not isinstance(ledger, CompiledLedger):
        raise TypeError(
            f"ledger must be a CompiledLedger, got {type(ledger).__name__}"
        )
    if regime is None:
        regime = {}
    lookup = ledger.lookup(regime)
    check = hypotheses.check_do_not_repeat(regime=regime, matches=lookup.matches)
    blockers = lookup.why_not_comparable
    if not blockers or check.outcome == schemas.FAIL:
        return check
    return schemas.Check(schemas.COULD_NOT_CHECK, blockers + tuple(check.reasons))


# =============================================================================
# The structural audit — "does the matcher read prose?"
# =============================================================================

#: A regime for the audit probe. Two questions differ ONLY in their statement, so a
#: difference in the result can only be prose sensitivity.
_AUDIT_REGIME: Mapping[str, Any] = {
    "backend": "llama_gpu",
    "phase": "decode",
    "batch_band": "b128",
    "mechanism": "elementwise_norm_fusion",
}

_AUDIT_STATEMENTS = (
    "fusing G15's elementwise/norm cluster lands >= 15% on decode",
    "an entirely differently worded claim, with no vocabulary in common whatsoever",
)


def _audit_probe_ledger() -> CompiledLedger:
    """A one-entry ledger that MATCHES the probe regime.

    The anti-vacuous control: without it, a matcher that returned nothing for every
    input would satisfy "the two statements give the same answer" perfectly.
    """
    regime, regime_raw, target, _ = read_facets(_AUDIT_REGIME)
    attempt = PriorAttempt(
        entry_id="dnr-audit-probe",
        entry_class=hypotheses.MATCH_CLASS_MATCHED_NEGATIVE,
        regime=regime, regime_raw=regime_raw, target=target,
        outcome=hypotheses.RESOLUTION_REFUTED,
        receipt="ake-audit-probe @ audit",
        why=("audit probe",),
    )
    return CompiledLedger((attempt,))


def audit_matching_ignores_prose() -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK on *"can the statement change a match?"*.

    Proved from the objects and from behaviour, in the shape of
    `hypotheses.audit_no_origin_grade_promotion()`:

    1. `MatchQuery` declares NO field whose name mentions a statement, prose, text or
       narrative — there is no slot for prose to be carried in;
    2. `MatchQuery.from_regime` takes exactly one positional input, the regime;
    3. a probe ledger that DOES match returns the identical match set for two
       statements with no words in common (the control that stops point 3 from passing
       vacuously is that the probe matches at all, which is asserted first).

    COULD_NOT_CHECK when introspection itself fails, never a soft PASS.
    """
    try:
        fields = {f.name for f in dataclasses.fields(MatchQuery)}
        ledger = _audit_probe_ledger()
        baseline = ledger.matches_for(_AUDIT_REGIME, _AUDIT_STATEMENTS[0])
    except Exception as exc:  # pragma: no cover - introspection failure is not a pass
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"could not exercise the matcher ({type(exc).__name__}: {exc}); inability "
            "to evaluate is a third outcome",
        ))

    reasons: list = []
    prose_fields = sorted(
        f for f in fields
        if any(word in f.lower() for word in ("statement", "prose", "text", "narrative"))
    )
    if prose_fields:
        reasons.append(
            f"MatchQuery declares prose field(s) {prose_fields}; matching must key on "
            "the regime and the structural target, never on the words of the claim"
        )
    if not baseline:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons) + (
            "the audit probe ledger matched NOTHING, so 'the same answer for two "
            "statements' cannot be told apart from a matcher that never matches",
        ))
    for statement in _AUDIT_STATEMENTS[1:]:
        other = ledger.matches_for(_AUDIT_REGIME, statement)
        if [m.to_dict() for m in other] != [m.to_dict() for m in baseline]:
            reasons.append(
                "two questions differing ONLY in their statement produced different "
                "matches; a reworded restatement of a tried idea would escape the "
                "ledger and two different ideas could be collapsed into one"
            )
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)
