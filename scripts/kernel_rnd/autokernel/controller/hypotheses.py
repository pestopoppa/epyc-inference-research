"""hypotheses.py — the operator hypothesis channel and still-open tracking (§8.4.0, AK-D38).

WHY THIS MODULE EXISTS
----------------------
Two halves of one mechanism, and neither works without the other.

**1. The operator channel.** The operator sees things the profile does not. Without a
channel, that steering arrives out-of-band — as a standing instruction with no
falsifier, no grade and no resolution record. §8.4.0 gives it a channel and makes it
*safe by grading*: a hypothesis is stated WITH a falsifier ("if a current wall-share
map shows the cluster under 20% I am wrong"), enters at `design_prior` evidence grade,
and **can never be promoted by its origin**. §19.0 rule 4 already forbids upgrading
evidence on import, and an operator hunch is exactly the input most likely to be
treated as settled because of who said it. It is a proposal SOURCE, never an
authority: it faces the pre-run critic unchanged, obeys every §8.4 rejection
condition, and is subject to the do-not-repeat ledger like any other proposal.
Authorship is not new evidence (AK-D38).

**2. Still-open tracking, for every hypothesis regardless of origin.** AutoKernel had
the proposal `hypothesis` field (`schemas.validate_proposal`) and nothing that tracked
it, so a hypothesis evaporated the moment its proposal was dispositioned — *including*
when that proposal failed for an unrelated reason. That is the failure mode that
leaves a question feeling "already tried" with no receipt. The sibling loop does not
have this either, verified against source rather than assumed (§8.4.0 correction,
2026-08-03): `ExperimentJournal.unfalsified_hypotheses()` is a five-trial *recency
window*, its own docstring calls resolution-checking *"intentionally minimal … presence
of the falsifier string only"*, nothing marks a hypothesis resolved, the still-open
block is stagnation-gated, and the falsifier defaults to an empty string.

So, here: every hypothesis carries a falsifier, stays open until resolved
confirmed / refuted / inconclusive **with the evidence that resolved it**, and the open
set is re-surfaced into every planning round.

WHERE THE FALSIFIER IS MANDATORY (2026-08-04 amendment, operator-approved)
--------------------------------------------------------------------------
A falsifier used to be mandatory at the moment the record was MADE, which put the
whole ceremony on the one person whose barrier to entry should be zero: the operator
drops in a one-line idea, and the loop refused it for want of a predicate the loop is
better placed to write. The discipline has not been relaxed — it has been MOVED to the
point where it bites. **A falsifier is optional on operator entry and MANDATORY before
compute is spent.**

Three states are structurally distinct, and collapsing any two is the defect:

* **absent** (`falsifier=None`) — legal for an `operator`-origin hypothesis, illegal
  for every other origin, and illegal to spend a resource claim on. This is an HONEST
  statement that no predicate has been written yet.
* **placeholder** (`""`, `"tbd"`, `"n/a"`, `"?"`, …) — ALWAYS illegal, at entry and
  after. It is not the same state as absent: absent is honest, a placeholder is an
  empty string wearing a hat, and AutoPilot's falsifier defaulted to `""` while nothing
  ever objected. `_PLACEHOLDER_FALSIFIERS` therefore still fires whenever a falsifier
  IS supplied — the optional route is a route to supplying NOTHING, never to supplying
  `"tbd"`.
* **stated** — a real one-line predicted outcome. The only state a claim may be spent
  on.

`classify_falsifier()` is total over those three and is the single place the question
is answered. An agent closes state (i) with `propose_falsifier()`, which records the
predicate it wrote (and who wrote it, and why) as its own ledger event; the operator's
own statement is never rewritten, so `fingerprint` — the identity of the QUESTION — is
untouched and `QuestionRewritten` keeps meaning what it meant.

The gate itself is `ClaimAuthorization`: a frozen token whose `__post_init__` re-derives
the falsifier state from the falsifier it carries and REFUSES to exist for (i) or (ii).
`claim_for_hypothesis()` is the only route from a hypothesis to a resource claim and it
accepts nothing else, so there is no path along which state (i) or (ii) reaches a claim
— not by constructing a token, not by calling the acquirer directly from this module
(`audit_falsifier_required_before_claim()` proves the second from this module's AST).

THE MEMORY PLANE IS CONSULTED BEFORE COMPUTE, NOT AFTER (2026-08-04)
---------------------------------------------------------------------
This module used to say, at the top of `check_do_not_repeat`, that it *"does not BUILD
that ledger (the memory-update plane owns it); it consumes matches and disposes them"*
— and the memory-update plane did not exist. A correct guard wired to nothing is the
defect shape this package has hit repeatedly, and the cost here is specific: without a
ledger the loop cannot tell **"tried and failed"** from **"never tried"**, so it
re-tries dead ideas forever and pays a resource claim for each one.

`do_not_repeat.py` is now that plane, and `authorize_claim(..., ledger=...)` is the
seam. **`ledger` is a REQUIRED argument** — a default would have rebuilt the original
defect exactly, since every caller that forgot it would silently get the unconsulted
behaviour. `None` is refused too: an empty `do_not_repeat.CompiledLedger()` is one pure
call away and states the true thing, so "no memory configured" is not a position.

What the verdict does:

* **FAIL** (a receipted `MATCHED_NEGATIVE` or `HARD_CONSTRAINT` in a matching regime)
  -> `RepeatsAReceiptedNegative`, raised from `ClaimAuthorization.__post_init__`, so
  the token CANNOT EXIST and no `CLAIM_AUTHORIZED` record is written.
* **COULD_NOT_CHECK** -> the claim proceeds, and the verdict rides on the token. This
  is the operator's own case (a one-line idea rarely names a `mechanism`, so the ledger
  refuses to compare it) and blocking on it would make the operator channel unusable.
  The direction is deliberate: §19.3 — a wrong suppression is SILENT and permanent, a
  wasted re-run is LOUD and costs one claim.
* **PASS** -> proceeds, with the verdict recorded.

`claim_for_hypothesis()` re-derives BOTH gates at the door, so a token that reached an
acquirer without a verdict (`LedgerNotConsulted`) or with a FAIL edited into it is
refused there as well.

ADOPTION TRANSFERS OWNERSHIP (2026-08-04, operator-approved)
------------------------------------------------------------
*"if the agents choose to pick up one of my hypotheses, it should be removed from
OperatorHypothesisStore since it becomes owned by the agents."* `adopt()` is that move,
and it is a move between TWO DURABLE STORES with exactly three failure modes:

* **LOST** — removed from the store, never recorded in the ledger. The operator's idea
  vanishes with no trace. UNACCEPTABLE, and structurally impossible here: `adopt()`
  opens an untracked hypothesis into the ledger before it touches the file at all.
* **DUPLICATE** — recorded, not removed. Recoverable, and DETECTABLE by id
  (`adoption_duplicates()`), repairable by id (`reconcile_adoptions()`).
* **ORPHANED** — recorded and removed with nothing linking them. Prevented by writing
  the operator's own entry BYTES and the full hypothesis content INLINE in the adoption
  record, so the record stands alone if the file is never read again.

**JOURNAL FIRST, THEN REMOVE**, and the ledger is the source of truth. A crash between
the two leaves a DUPLICATE, which is the direction this is deliberately failed toward.

Two further properties make removal acceptable at all:

* **Operator traceability.** `trace()` resolves a `hypothesis_id` to its adoption, its
  attempts, its claim authorizations and its resolution. Removal without a findable
  trace is just deletion.
* **The store is OPERATOR-OWNED.** The rewrite is a BYTE SPLICE — the entry's own text
  span is cut out and every other entry is carried across unchanged, verified span by
  span before the bytes are written — then `os.replace`d atomically. An operator who
  opens their file after an adoption sees their own text, untouched, minus one entry.

Two agents must not both adopt one hypothesis, and the store is a plain file with no
lock. `OperatorHypothesisStore.adoption_lock()` uses the discipline
`resource/device_claim.py` already established rather than inventing a second one:
`flock(LOCK_EX|LOCK_NB)` on a never-unlinked sidecar, a holder payload from
`device_claim.current_holder_identity()` (pid + `/proc` start ticks + boot id), and
`device_claim.assess_holder_liveness()` to classify what it finds — where `unknown` is
a third outcome and never a soft `dead`.

THE GRADING IS THE SAFETY PROPERTY, AND IT IS STRUCTURAL
--------------------------------------------------------
There is no code path in which origin raises grade, and the absence is checked rather
than intended:

* `Hypothesis` has **no `evidence_grade` field** — it is a read-only property returning
  `entry_grade(origin)`, and `entry_grade` is total over `ORIGINS` and constant. A new
  origin that forgot to be `design_prior` cannot exist, because the function does not
  branch on origin at all; `audit_no_origin_grade_promotion()` proves both facts from
  the objects.
* The operator store **refuses an `evidence_grade` key outright** (`_REFUSED_ENTRY_KEYS`).
  A store that could state its own grade is a store that can launder a hunch into a
  measured fact by typing one word.
* A resolution's `evidence_grade` belongs to the *evidence*, lives on
  `ResolutionEvidence`, and never touches the hypothesis's entry grade. The rendered
  planner block names the two separately (`entry_evidence_grade` vs
  `resolution_evidence_grade`) so no reader can conflate them.

WHAT THIS MODULE REFUSES TO BECOME (§8.4.0 "What it must not become")
---------------------------------------------------------------------
* **Not a queue-jumping mechanism.** There is no priority, rank, weight or boost field
  anywhere in this module, and `still_open()` orders by the sequence in which questions
  were opened — a stable order, not a ranking.
* **Not a bypass.** Nothing here waives a §8.4 rejection condition, a wall-share
  ceiling, or a correctness gate. `find_authority_flavoured_keys()` is run over every
  operator entry and over every rendered block, so an authority flag added later fails
  at load instead of quietly becoming load-bearing.
* **Not a route to mark something resolved without evidence.** `resolve()` demands
  evidence references, an observation stated against the falsifier, and an explicit
  declaration that the evidence bears on that falsifier. A proposal disposition is an
  ATTEMPT (`note_attempt()`), and an attempt never closes a question — which is the
  whole point of the second half of this mechanism.

A MALFORMED STORE RAISES
------------------------
`OperatorHypothesisStore.load()` never degrades to an empty list. An empty list is a
statement — *"the operator has no open hypotheses"* — and the planner acts on it. A
truncated file, a bad mount, a JSON typo or a missing falsifier must therefore be
distinguishable from that statement, and the only distinguishable outcome is a refusal.
"No operator channel is configured" is expressed by passing no store at all.

SUBSTRATE NOTE — where a hypothesis event lands
-----------------------------------------------
`journal.KINDS` is a CLOSED vocabulary with no hypothesis kind, and `journal.py` is
outside this task's write scope. Events therefore land in `HypothesisLedger`: the same
discipline as `state_machine.TransitionLedger` (O_APPEND, fsync per record, torn tail
discarded and reported, strictly increasing seq), written under `Journal.write_lock()`
so its order is total with respect to journal appends. Co-opting a §19.4
bootstrap-knowledge kind (`PRIOR_ATOMIZED`, `PRIOR_SUPERSEDED`) was rejected for the
reason `state_machine` rejected co-opting `STOP_STATE`: those kinds name the bootstrap
corpus, other planes fold them, and a second meaning makes the first one unreadable.
**Follow-up: add `KIND_HYPOTHESIS_OPENED` / `_ATTEMPTED` / `_RESOLVED` / `_REOPENED` to
`journal.py`; `HypothesisRecorder` is the seam that makes the swap a wiring change with
no contract change here.**

NO CACHED STATE
---------------
`HypothesisTracker` holds no folded state. Every query re-reads the ledger from disk,
for the same reason `state_machine` re-reads the control latch: a cached copy is a copy
that can be written back over the record. This is O(ledger) per call and is the right
price at this scale; it is noted here rather than silently optimized.

This module runs NO inference, NO benchmark and NO build; it starts, stops and signals
NO process; it calls NO model. It writes the ledger under the controller root it is
given, and — only along the adoption path, only under the adoption lock, and only ever
by REMOVING one entry — the operator store file it is handed plus that store's sidecar
lock. It acquires no resource claim itself: `claim_for_hypothesis()` takes the acquirer
as an argument and is exercised against a fake one.

Governing instrument: `measurement/protocols/kernel-research.md` (P-AK-SEARCH-1,
RATIFIED 2026-08-03). Nothing here ranks, banks, composes or contributes to readiness.
"""
from __future__ import annotations

import ast
import dataclasses
import fcntl
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol, Sequence

from .. import journal, schemas
from ..resource import device_claim as _device_claim
from .shared import ControllerError
# `state_machine.py` was removed with the rest of the AK4 strategy plane. It
# owned exactly one thing this module needs: the base refusal class. Restoring
# ~2,000 lines for a two-line exception would be the tail wagging the dog, so
# the class moved UP into the package where both survivors can reach it — which
# is the one place a shared concern in this package has ever had.

__all__ = [
    # errors
    "HypothesisError", "HypothesisStoreError", "FalsifierMissing",
    "HypothesisLedgerCorruption", "UnknownHypothesis", "HypothesisNotOpen",
    "HypothesisAlreadyTracked", "QuestionRewritten", "ResolutionEvidenceMissing",
    "FalsifierRequiredBeforeCompute", "FalsifierAlreadyStated",
    "LedgerNotConsulted", "RepeatsAReceiptedNegative",
    "HypothesisAlreadyAdopted", "HypothesisNotInStore", "AdoptionLockUnavailable",
    "AdoptionLockInconsistent", "StoreRewriteRefused",
    # vocabulary
    "ORIGIN_OPERATOR", "ORIGIN_PLANNER", "ORIGIN_CRITIC", "ORIGIN_CONTROLLER",
    "ORIGIN_IMPORT", "ORIGINS",
    "GRADE_DESIGN_PRIOR", "GRADE_OBSERVATION", "GRADE_SOURCE_VERIFIED",
    "GRADE_IMPORTED_CLAIM", "GRADE_PROTOCOL_BOUND", "EVIDENCE_GRADES", "ENTRY_GRADE",
    "STATUS_OPEN", "RESOLUTION_CONFIRMED", "RESOLUTION_REFUTED",
    "RESOLUTION_INCONCLUSIVE", "RESOLUTIONS", "STATUSES",
    "FALSIFIER_ABSENT", "FALSIFIER_PLACEHOLDER", "FALSIFIER_STATED",
    "FALSIFIER_STATES", "FALSIFIER_STATES_REFUSING_COMPUTE",
    "FALSIFIER_SOURCE_STATED", "FALSIFIER_SOURCE_PROPOSED",
    "OWNER_OPERATOR", "OWNER_AGENTS",
    "EVENT_OPENED", "EVENT_ATTEMPTED", "EVENT_RESOLVED", "EVENT_REOPENED",
    "EVENT_FALSIFIER_PROPOSED", "EVENT_CLAIM_AUTHORIZED", "EVENT_ADOPTED",
    "LEDGER_EVENT_KINDS",
    "MATCH_CLASS_HARD_CONSTRAINT", "MATCH_CLASS_MATCHED_NEGATIVE",
    "MATCH_CLASS_CONDITIONAL_NEGATIVE", "MATCH_CLASS_CONFOUNDED_RESULT",
    "MATCH_CLASS_SUPERSEDED_FACT", "MATCH_CLASS_LOW_VALUE", "MATCH_CLASSES",
    "REJECTING_MATCH_CLASSES",
    "STORE_SCHEMA", "ROUND_BLOCK_SCHEMA", "LEDGER_FILENAME", "ADOPTION_LOCK_SUFFIX",
    # types
    "Hypothesis", "Attempt", "ResolutionEvidence", "TrackedHypothesis",
    "LedgerEvent", "LedgerRead", "IntakeReport", "LedgerMatch",
    "FalsifierProposal", "ClaimAuthorization", "Adoption", "HypothesisTrace",
    "EntrySpan", "StoreRemoval",
    # seams
    "HypothesisRecorder", "DoNotRepeatLedger",
    # implementations
    "OperatorHypothesisStore", "HypothesisLedger", "JournalOrderedRecorder",
    "HypothesisTracker",
    # pure functions and checks
    "entry_grade", "falsifier_optional_on_entry", "classify_falsifier",
    "fold_ledger", "check_do_not_repeat",
    "claim_for_hypothesis",
    "audit_no_origin_grade_promotion", "audit_falsifier_required_before_claim",
]


# =============================================================================
# Errors — every one is a refusal. None of them has a degraded-result sibling.
# =============================================================================

class HypothesisError(ControllerError):
    """Base for every refusal here.

    It extends `state_machine.ControllerError` so a driver can catch the CONTROLLER
    PLANE rather than one module of it; a loop that has to enumerate module bases
    eventually forgets one, and the one it forgets is the one that matters.
    """


class HypothesisStoreError(HypothesisError):
    """The operator store could not be read as a store.

    Never an empty list: empty means *"the operator has no hypotheses"*, the planner
    acts on that, and a mount failure must not be able to say it.
    """


class FalsifierMissing(HypothesisStoreError):
    """A hypothesis was stated without a falsifier.

    The falsifier is what makes it a resolvable hypothesis rather than a standing
    instruction. This is the refusal §8.4.0 turns on, so it has its own type.
    """


class HypothesisLedgerCorruption(HypothesisError):
    """The ledger does not describe one coherent history."""


class UnknownHypothesis(HypothesisError):
    """An operation named a hypothesis that was never opened."""


class HypothesisNotOpen(HypothesisError):
    """Resolve on a resolved question, or reopen on an open one."""


class HypothesisAlreadyTracked(HypothesisError):
    """An id that is already in the ledger was opened again."""


class QuestionRewritten(HypothesisError):
    """A tracked hypothesis's statement or falsifier changed under its own id.

    Rewriting a falsifier after evidence exists is how any hypothesis becomes
    "confirmed". A new question gets a new id; the old one stays open with its
    original falsifier until evidence closes it.
    """


class ResolutionEvidenceMissing(HypothesisError):
    """A resolution that does not carry the evidence that resolved it."""


class FalsifierRequiredBeforeCompute(HypothesisError):
    """A resource claim was going to be spent on a hypothesis with no falsifier.

    Deliberately NOT a `FalsifierMissing`/`HypothesisStoreError`: an absent falsifier
    is legal in the store (the operator's barrier to entry is zero) and illegal only
    HERE, at the point compute is committed. Typing it as a store error would send a
    reader to the operator's file looking for a defect that is not in it.
    """


class LedgerNotConsulted(HypothesisError):
    """A resource claim was going to be spent without asking the memory plane.

    Distinct from `RepeatsAReceiptedNegative`, and the distinction is the whole point:
    *"the ledger says this failed"* and *"nobody asked the ledger"* are different
    states, and collapsing them is how `check_do_not_repeat()` came to be a correct
    guard wired to nothing. Not asking is not a clear result — the module's own words —
    so the door refuses a token that carries no verdict at all.
    """


class RepeatsAReceiptedNegative(HypothesisError):
    """The §19.2 ledger already records this idea, in this regime, with a receipt.

    §8.4 rejects a repeat of a receipted negative. This is the ONE do-not-repeat class
    that refuses a claim: `SUPERSEDED_FACT`, `CONDITIONAL_NEGATIVE`, `CONFOUNDED_RESULT`
    and `LOW_VALUE` are advisory and leave the question open, and COULD_NOT_CHECK does
    NOT refuse — a wrong suppression is silent and permanent (§19.3), a re-run is loud
    and costs one claim, so the ambiguous case is failed toward spending the claim.
    """


class FalsifierAlreadyStated(HypothesisError):
    """A falsifier was proposed for a question that already has one.

    A second falsifier is a REWRITTEN falsifier, which is how any hypothesis becomes
    "confirmed" — the same defect `QuestionRewritten` names, reached from the other
    side.
    """


class HypothesisAlreadyAdopted(HypothesisError):
    """Two agents tried to adopt one operator hypothesis.

    Adoption transfers OWNERSHIP, so it happens exactly once. The loser of the race
    gets this rather than a second adoption record.
    """


class HypothesisNotInStore(HypothesisStoreError):
    """Adoption named an id the operator store does not carry."""


class AdoptionLockUnavailable(HypothesisError):
    """The store's adoption lock was held for the whole budget."""


class AdoptionLockInconsistent(HypothesisError):
    """The adoption lock was free but its recorded holder is alive or unverifiable.

    Mirrors `device_claim.DeviceClaimInconsistent`: `unknown` is a third outcome and
    never a soft `dead`, so nothing is taken.
    """


class StoreRewriteRefused(HypothesisStoreError):
    """The operator store could not be rewritten with exactly one entry removed.

    Raised BEFORE any byte is replaced. When it is raised after the adoption record is
    already durable, the resulting state is a DUPLICATE (recorded, not removed) — the
    direction this move is deliberately failed toward, detectable by
    `HypothesisTracker.adoption_duplicates()`.
    """


# =============================================================================
# Vocabulary
# =============================================================================

# Mirrors `state_machine.STOP_REQUEST_ORIGINS` and adds `import`: §19.5 imports
# historical research as typed legacy events, and an "unresolved contradiction"
# prior (§19.1) is a hypothesis with an author who is not in the room.
ORIGIN_OPERATOR = "operator"
ORIGIN_PLANNER = "planner"
ORIGIN_CRITIC = "critic"
ORIGIN_CONTROLLER = "controller"
ORIGIN_IMPORT = "import"

ORIGINS = frozenset({
    ORIGIN_OPERATOR, ORIGIN_PLANNER, ORIGIN_CRITIC, ORIGIN_CONTROLLER, ORIGIN_IMPORT,
})

# §19.1's `evidence_grade` vocabulary, verbatim.
GRADE_DESIGN_PRIOR = "design_prior"
GRADE_OBSERVATION = "observation"
GRADE_SOURCE_VERIFIED = "source_verified"
GRADE_IMPORTED_CLAIM = "imported_claim"
GRADE_PROTOCOL_BOUND = "protocol_bound"

EVIDENCE_GRADES = frozenset({
    GRADE_DESIGN_PRIOR, GRADE_OBSERVATION, GRADE_SOURCE_VERIFIED,
    GRADE_IMPORTED_CLAIM, GRADE_PROTOCOL_BOUND,
})

#: The grade EVERY hypothesis enters at, whoever stated it (§8.4.0, AK-D38, §19.0
#: rule 4). `design_prior` means "worth considering", not "probably true" (§19.1).
ENTRY_GRADE = GRADE_DESIGN_PRIOR

STATUS_OPEN = "open"
RESOLUTION_CONFIRMED = "confirmed"
RESOLUTION_REFUTED = "refuted"
RESOLUTION_INCONCLUSIVE = "inconclusive"

#: `inconclusive` is a real resolution and NOT a synonym for open: the experiment ran
#: and did not resolve the falsifier. It is reopenable on new evidence, which is what
#: stops it from becoming a permanent silent close. (Same distinction
#: `schemas.EVENT_STATUSES` draws between `inconclusive` and `invalid`.)
RESOLUTIONS = frozenset({
    RESOLUTION_CONFIRMED, RESOLUTION_REFUTED, RESOLUTION_INCONCLUSIVE,
})
STATUSES = frozenset({STATUS_OPEN}) | RESOLUTIONS

#: The three falsifier states, and they are THREE. `absent` is an honest "no predicate
#: has been written yet" and is legal for an operator entry; `placeholder` is an empty
#: string wearing a hat and is illegal everywhere; `stated` is the only one a resource
#: claim may be spent on. Collapsing (i) and (ii) into one "no usable falsifier" state
#: is the defect this vocabulary exists to prevent: it would make the honest case and
#: the dishonest case indistinguishable, and the honest case is the operator's.
FALSIFIER_ABSENT = "absent"
FALSIFIER_PLACEHOLDER = "placeholder"
FALSIFIER_STATED = "stated"

FALSIFIER_STATES = frozenset({
    FALSIFIER_ABSENT, FALSIFIER_PLACEHOLDER, FALSIFIER_STATED,
})

#: The states no resource claim may be spent on. Both of them, for different reasons,
#: and `ClaimAuthorization` refuses to exist for either.
FALSIFIER_STATES_REFUSING_COMPUTE = frozenset({
    FALSIFIER_ABSENT, FALSIFIER_PLACEHOLDER,
})

#: Where the falsifier a claim is authorized against came from. Recorded because "the
#: operator wrote this predicate" and "an agent wrote this predicate for the operator's
#: idea" are different facts, and the second one names an author who can be asked.
FALSIFIER_SOURCE_STATED = "stated_with_the_hypothesis"
FALSIFIER_SOURCE_PROPOSED = "proposed_by_an_agent"

#: Ownership, which is NOT origin. `origin` is who STATED the question and is frozen
#: forever (`_REFUSED_ENTRY_KEYS` refuses an entry that relabels its own provenance).
#: Ownership is who is working it now, and adoption is the one thing that moves it.
OWNER_OPERATOR = "operator_store"
OWNER_AGENTS = "agents"

EVENT_OPENED = "HYPOTHESIS_OPENED"
EVENT_ATTEMPTED = "HYPOTHESIS_ATTEMPTED"
EVENT_RESOLVED = "HYPOTHESIS_RESOLVED"
EVENT_REOPENED = "HYPOTHESIS_REOPENED"
#: An agent wrote the predicate the operator did not have to. Its own event, not an
#: edit of the OPENED record: an append-only ledger has no edits, and the operator's
#: own words must stay readable exactly as they were written.
EVENT_FALSIFIER_PROPOSED = "HYPOTHESIS_FALSIFIER_PROPOSED"
#: Compute was committed to this question. Recorded WITH the falsifier it was
#: authorized against, so "what did we spend the card on, and what would have refuted
#: it" is one lookup rather than a reconstruction.
EVENT_CLAIM_AUTHORIZED = "HYPOTHESIS_CLAIM_AUTHORIZED"
#: The agents took ownership of an operator hypothesis. Written BEFORE the store entry
#: is removed, with the operator's own entry bytes inline.
EVENT_ADOPTED = "HYPOTHESIS_ADOPTED"

LEDGER_EVENT_KINDS = (
    EVENT_OPENED, EVENT_ATTEMPTED, EVENT_RESOLVED, EVENT_REOPENED,
    EVENT_FALSIFIER_PROPOSED, EVENT_CLAIM_AUTHORIZED, EVENT_ADOPTED,
)

# §19.2's do-not-repeat / constraint ledger classes. This module does not BUILD that
# ledger (the memory-update plane owns it); it consumes matches and disposes them.
MATCH_CLASS_HARD_CONSTRAINT = "HARD_CONSTRAINT"
MATCH_CLASS_MATCHED_NEGATIVE = "MATCHED_NEGATIVE"
MATCH_CLASS_CONDITIONAL_NEGATIVE = "CONDITIONAL_NEGATIVE"
MATCH_CLASS_CONFOUNDED_RESULT = "CONFOUNDED_RESULT"
MATCH_CLASS_SUPERSEDED_FACT = "SUPERSEDED_FACT"
MATCH_CLASS_LOW_VALUE = "LOW_VALUE"

MATCH_CLASSES = frozenset({
    MATCH_CLASS_HARD_CONSTRAINT, MATCH_CLASS_MATCHED_NEGATIVE,
    MATCH_CLASS_CONDITIONAL_NEGATIVE, MATCH_CLASS_CONFOUNDED_RESULT,
    MATCH_CLASS_SUPERSEDED_FACT, MATCH_CLASS_LOW_VALUE,
})

#: The two classes whose §19.2 planner behaviour is "reject". The other four exclude
#: cells, demand a repaired experiment, regenerate from current source, or
#: deprioritize — all of which are advisory and none of which closes the question.
REJECTING_MATCH_CLASSES = frozenset({
    MATCH_CLASS_HARD_CONSTRAINT, MATCH_CLASS_MATCHED_NEGATIVE,
})

STORE_SCHEMA = "epyc.autokernel.operator_hypotheses.v1"
ROUND_BLOCK_SCHEMA = "epyc.autokernel.hypothesis_round_block.v1"

LEDGER_FILENAME = "hypotheses.jsonl"

#: The adoption lock sits BESIDE the store and is never unlinked, for the reason
#: `device_claim` spells out: create-and-unlink lockfiles let two processes hold
#: `LOCK_EX` on two different inodes for one path, which is two holders and no error.
ADOPTION_LOCK_SUFFIX = ".adopt.lock"

_HYPOTHESIS_ID_RE = re.compile(r"^akh-[A-Za-z0-9][A-Za-z0-9._-]*$")


def _no_duplicate_keys(pairs):
    """`json` object hook that REFUSES a repeated key instead of keeping the last.

    `json.loads` silently keeps the last value for a duplicated key. Two consequences,
    both of them this module's problem:

    * `{"falsifier": "<a real predicate>", "falsifier": "tbd"}` loads as `"tbd"` — the
      operator wrote a predicate and the loader read a placeholder. A key this loader
      resolves by position is a key the operator believes had an effect, which is the
      rule `_ALLOWED_ENTRY_KEYS` already enforces for keys it does not know.
    * a second top-level `"hypotheses"` array makes the file readable two ways: the
      PARSER takes the last, `_locate_hypotheses_array` scans to the first. Adoption
      then recorded the transfer and spliced an entry out of the array nobody reads —
      leaving the operator's entry in the file forever, with `reconcile_adoptions()`
      unable to remove it. A file this module cannot read the same way twice is refused,
      exactly as `entry_spans()` refuses a scanner/parser disagreement.
    """
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(
                f"duplicate key {key!r}: this object states it twice, and JSON "
                "resolves that by position — the value you can see is not necessarily "
                "the value that took effect"
            )
        seen[key] = value
    return seen

#: A falsifier that is one of these is an empty string wearing a hat. AutoPilot's
#: falsifier defaulted to `""` and nothing objected; these are the strings that
#: reproduce that state while passing a non-empty check.
_PLACEHOLDER_FALSIFIERS = frozenset({
    "", "-", "--", "n/a", "n.a.", "na", "none", "null", "nil", "tbd", "todo", "tbc",
    "?", "??", "???", "x", "unknown", "unclear", "unstated", "not applicable",
    "see above", "as above", "pending", "n/a — see statement",
})


def entry_grade(origin: str) -> str:
    """The evidence grade a hypothesis ENTERS at. Constant, by construction.

    The parameter exists so the constancy is ENUMERABLE — `audit_no_origin_grade_
    promotion()` and the suite both range over `ORIGINS` and assert one value comes
    back — and for no other reason: there is no branch on `origin` in this body, so a
    future origin cannot be given a higher grade without deleting this line, which the
    audit and the tests would both catch.

    §8.4.0: *"An operator hypothesis enters at `design_prior` evidence grade and can
    never be promoted by its origin"*. §19.0 rule 4: never upgrade evidence on import.
    """
    if origin not in ORIGINS:
        raise ValueError(f"origin: {origin!r} is not a declared origin {sorted(ORIGINS)}")
    return ENTRY_GRADE


def falsifier_optional_on_entry(origin: str) -> bool:
    """May a hypothesis of this ORIGIN be recorded with no falsifier at all?

    True for the operator and nobody else, and it is deliberately expressed as a
    function of `origin` — the SAME notion `entry_grade()` grades by — rather than as a
    second parallel idea of who is special. There is exactly one axis of authorship in
    this module, and the amendment moves a barrier along it; it does not add one.

    Note the two facts point in OPPOSITE directions, which is the whole design: the
    operator's origin buys a LOWER barrier to entry (`design_prior` is still the ceiling
    — `entry_grade` does not branch on origin at all) and never a higher standing.
    """
    if origin not in ORIGINS:
        raise ValueError(f"origin: {origin!r} is not a declared origin {sorted(ORIGINS)}")
    return origin == ORIGIN_OPERATOR


def classify_falsifier(value: Any) -> str:
    """Which of the THREE falsifier states `value` is in. Total, pure, one place.

    * `None` -> `absent`. Nothing was supplied. Honest.
    * a string that is empty, whitespace, or one of `_PLACEHOLDER_FALSIFIERS`
      -> `placeholder`. Something was supplied and it says nothing. A lie wearing a hat,
      and the AutoPilot defect verbatim (`falsifier` defaulted to `""`).
    * anything else -> `stated`.

    `absent` and `placeholder` are NOT merged. Merging them would mean the amendment
    that removed the operator's barrier had also opened a route to `"tbd"`, because a
    caller could satisfy "no falsifier is fine here" by typing one. They are refused in
    different places for different reasons: `placeholder` at construction, always;
    `absent` at the claim, and only there.

    A non-string, non-None falsifier raises rather than classifying: a falsifier that is
    a bool or a number is a field that changed type, which is precisely the YAML hazard
    the store is JSON to avoid.
    """
    if value is None:
        return FALSIFIER_ABSENT
    if not isinstance(value, str):
        raise TypeError(
            f"falsifier must be a string or None, got {type(value).__name__}; a "
            "falsifier that silently changes type is a falsifier that silently stops "
            "being one"
        )
    if value.strip().lower() in _PLACEHOLDER_FALSIFIERS:
        return FALSIFIER_PLACEHOLDER
    return FALSIFIER_STATED


# =============================================================================
# Small helpers — the durable-write pair is deliberately local
#
# `journal.py` and `state_machine.py` each carry their own copy of these four lines.
# Importing another module's private helper would couple two planes through a name
# neither exports; a four-line fsync wrapper is the cheaper duplication.
# =============================================================================

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_text(value: Any, what: str, *, error=ValueError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{what} is required and must be a non-empty string")
    return value


def _require_refs(value: Any, what: str, *, error=ValueError) -> tuple:
    """A non-empty tuple of non-empty strings. An empty ref list is no receipt."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise error(f"{what} must be a sequence of reference strings")
    refs = tuple(value)
    if not refs:
        raise error(f"{what} must name at least one reference; an empty list is not evidence")
    for index, ref in enumerate(refs):
        if not isinstance(ref, str) or not ref.strip():
            raise error(f"{what}[{index}] must be a non-empty reference string")
    return refs


def _reasons_from_record(value: Any) -> tuple:
    """A do-not-repeat reason list read back OUT of a record. Absent means empty.

    Separate from `_refs_from_record` because the two fields differ on the empty case:
    an empty EVIDENCE list is "no evidence" and must be refused, while an empty REASON
    list is what a clean PASS legitimately carries.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("do_not_repeat_reasons must be a sequence of strings")
    return tuple(value)


def _refs_from_record(value: Any, what: str, *, error) -> tuple:
    """Refs read back OUT of a record, before `_require_refs` sees them.

    `tuple("akj-1")` is `('a','k','j','-','1')` — five perfectly valid-looking
    references — so a `from_dict` that normalises with `tuple(...)` hands
    `_require_refs` something that passes every check it makes, and the guard that
    exists precisely to refuse a bare string never fires. A record whose ref list is a
    string (or a mapping, whose `tuple()` is its KEYS) is refused here instead.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise error(
            f"{what} must be a LIST of reference strings, not "
            f"{type(value).__name__}; a bare string would be exploded into one "
            "'reference' per character and would then satisfy every later check"
        )
    return tuple(value)


# =============================================================================
# The hypothesis itself
# =============================================================================

@dataclass(frozen=True)
class Hypothesis:
    """One question, with the falsifier that makes it resolvable.

    NOTE WHAT IS ABSENT: there is no `evidence_grade` field, no `priority`, no `rank`,
    and no `status`. Grade is derived (`entry_grade`), ranking does not exist here at
    all (§8.4.0: not a queue-jumping mechanism), and status lives in the ledger because
    a status that travels with the statement is a status somebody can edit.

    `falsifier` is `Optional[str]` and STILL HAS NO DEFAULT, which is load-bearing. An
    operator-origin hypothesis may carry `None` — that is the 2026-08-04 amendment — but
    `None` must be written out by the caller. A defaulted falsifier is exactly the
    AutoPilot shape (`falsifier` defaulted to `""` and nothing ever objected): it makes
    "we decided not to state one" indistinguishable from "we forgot the field existed".
    """

    hypothesis_id: str
    statement: str
    falsifier: Optional[str]
    origin: str
    author: str
    regime: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.hypothesis_id, "hypothesis_id")
        if not _HYPOTHESIS_ID_RE.match(self.hypothesis_id):
            raise ValueError(
                f"hypothesis_id: {self.hypothesis_id!r} must start with 'akh-' "
                "(the id prefixes are the record's family, per §7)"
            )
        _require_text(self.statement, "statement")
        # Origin is validated BEFORE the falsifier, because since 2026-08-04 the
        # falsifier rule READS origin (only the operator channel may omit one). An
        # unvalidated origin reaching that branch would decide the barrier by typo.
        if self.origin not in ORIGINS:
            raise ValueError(f"origin: {self.origin!r} not in {sorted(ORIGINS)}")
        self._check_falsifier()
        _require_text(self.author, "author")
        for name in ("regime", "source"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            # Canonicalizability is checked at construction, not at write time: a
            # hypothesis that cannot be serialized must fail before it is tracked,
            # never halfway through being recorded.
            schemas.canonical_json(dict(value))
        flagged = schemas.find_authority_flavoured_keys(
            {"regime": dict(self.regime), "source": dict(self.source)}
        )
        if flagged:
            raise ValueError(
                f"authority-flavoured keys {flagged} in a hypothesis; §8.4.0 — a "
                "hypothesis is a proposal SOURCE, never an authority, and it carries "
                "no freeze/cutover/promotion flag (§1.3, invariant 5)"
            )
        if self.created_at is not None:
            _require_text(self.created_at, "created_at")

    def _check_falsifier(self) -> None:
        """§8.4.0: a ONE-LINE predicted outcome whose absence invalidates it.

        Three states, and the two refusals here are NOT the same refusal:

        * `absent` is refused for every origin except `operator` (2026-08-04). It is
          not refused for the operator, and it is not silently upgraded either — it
          stays absent until an agent writes one, and `ClaimAuthorization` refuses to
          exist for it, which is where the discipline now bites.
        * `placeholder` is refused for EVERY origin including the operator's. The
          amendment moved a barrier; it did not open a route to `"tbd"`.
        """
        state = classify_falsifier(self.falsifier)
        if state == FALSIFIER_ABSENT:
            if not falsifier_optional_on_entry(self.origin):
                raise FalsifierMissing(
                    f"{self.hypothesis_id}: a {self.origin!r}-origin hypothesis must be "
                    "stated WITH its falsifier. Only the operator channel may enter one "
                    "without a predicate (2026-08-04): an agent that cannot say what "
                    "would refute its own idea has not had the idea yet"
                )
            return
        if state == FALSIFIER_PLACEHOLDER:
            raise FalsifierMissing(
                f"{self.hypothesis_id}: falsifier {self.falsifier!r} is a placeholder; "
                "the predecessor loop's falsifier defaulted to the empty string and "
                "nothing ever objected. A falsifier may be ABSENT on an operator entry "
                "(state 'absent'); it may never be an empty string wearing a hat"
            )
        collapsed = self.falsifier.strip().lower()
        if "\n" in self.falsifier or "\r" in self.falsifier:
            raise FalsifierMissing(
                f"{self.hypothesis_id}: falsifier must be ONE LINE — a predicted "
                "outcome whose absence invalidates the hypothesis, not a paragraph "
                "of reasoning (§8.4.0)"
            )
        if collapsed == self.statement.strip().lower():
            raise FalsifierMissing(
                f"{self.hypothesis_id}: falsifier restates the hypothesis; it must "
                "predict an OUTCOME that could fail to appear"
            )

    @property
    def evidence_grade(self) -> str:
        """Always `design_prior`. A PROPERTY, deliberately: there is no field to set.

        This is the §8.4.0 safety property in its structural form — the object cannot
        express a promoted grade, so no code path can produce one.
        """
        return entry_grade(self.origin)

    @property
    def falsifier_state(self) -> str:
        """One of `absent` / `placeholder` / `stated`. Derived, never stored.

        `placeholder` is unreachable on a constructed `Hypothesis` — `_check_falsifier`
        refuses it — and the property still returns it rather than pretending the state
        does not exist, because `classify_falsifier` is the one answer to this question
        and a second, narrower one would be a second answer.
        """
        return classify_falsifier(self.falsifier)

    @property
    def fingerprint(self) -> str:
        """Identity of the QUESTION: statement + falsifier + regime.

        Two entries with one id and different fingerprints are two questions, and the
        second is refused rather than applied (`QuestionRewritten`).
        """
        return schemas.content_hash({
            "statement": self.statement,
            "falsifier": self.falsifier,
            "regime": dict(self.regime),
        })

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "falsifier": self.falsifier,
            "origin": self.origin,
            "author": self.author,
            "regime": dict(self.regime),
            "source": dict(self.source),
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "Hypothesis":
        if not isinstance(obj, Mapping):
            raise TypeError("hypothesis record must be a mapping")
        missing = sorted(
            {"hypothesis_id", "statement", "falsifier", "origin", "author"} - set(obj)
        )
        if missing:
            raise ValueError(f"hypothesis record is missing {missing}")
        return Hypothesis(
            hypothesis_id=obj["hypothesis_id"],
            statement=obj["statement"],
            falsifier=obj["falsifier"],
            origin=obj["origin"],
            author=obj["author"],
            regime=dict(obj.get("regime") or {}),
            source=dict(obj.get("source") or {}),
            created_at=obj.get("created_at"),
        )


# =============================================================================
# Attempts and resolutions
# =============================================================================

@dataclass(frozen=True)
class Attempt:
    """A proposal that was dispositioned while this question was open.

    An attempt NEVER resolves anything, and that is the entire point: AutoKernel's
    hypothesis used to evaporate when its proposal was dispositioned, *including when
    that proposal failed for an unrelated reason* — a build break, a skipped proposal,
    a voided window. Recording the attempt is what turns "this feels already tried"
    into a receipt that says what was tried and what it did or did not bear on.
    """

    hypothesis_id: str
    proposal_id: str
    disposition: str
    bears_on_falsifier: bool
    note: str
    refs: tuple = ()
    at: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.hypothesis_id, "hypothesis_id")
        _require_text(self.proposal_id, "proposal_id")
        _require_text(self.disposition, "disposition")
        _require_text(self.note, "note")
        if not isinstance(self.bears_on_falsifier, bool):
            raise TypeError(
                "bears_on_falsifier must be an explicit bool — 'we did not consider "
                "whether this bore on the falsifier' is not a value this field takes"
            )
        if isinstance(self.refs, (str, bytes)) or not isinstance(self.refs, Sequence):
            raise TypeError("refs must be a sequence of reference strings")
        for index, ref in enumerate(self.refs):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"refs[{index}] must be a non-empty reference string")

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "proposal_id": self.proposal_id,
            "disposition": self.disposition,
            "bears_on_falsifier": self.bears_on_falsifier,
            "note": self.note,
            "refs": list(self.refs),
            "at": self.at,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "Attempt":
        if not isinstance(obj, Mapping):
            raise TypeError("attempt record must be a mapping")
        missing = sorted(
            {"hypothesis_id", "proposal_id", "disposition", "bears_on_falsifier", "note"}
            - set(obj)
        )
        if missing:
            raise ValueError(f"attempt record is missing {missing}")
        return Attempt(
            hypothesis_id=obj["hypothesis_id"],
            proposal_id=obj["proposal_id"],
            disposition=obj["disposition"],
            bears_on_falsifier=obj["bears_on_falsifier"],
            note=obj["note"],
            refs=_refs_from_record(obj.get("refs"), "refs", error=TypeError),
            at=obj.get("at"),
        )


@dataclass(frozen=True)
class ResolutionEvidence:
    """What closed a question, and what was observed against its falsifier.

    `evidence_grade` here is the grade of the EVIDENCE. It is never written back onto
    the hypothesis, whose entry grade is `design_prior` forever: an operator hypothesis
    that the loop confirms is confirmed BY THE EVIDENCE, and the record says which
    evidence, so a later reader can check it.
    """

    outcome: str
    evidence_grade: str
    evidence_refs: tuple
    falsifier_observed: str
    bears_on_falsifier: bool
    resolved_by: str
    at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.outcome not in RESOLUTIONS:
            raise ValueError(f"outcome: {self.outcome!r} not in {sorted(RESOLUTIONS)}")
        if self.evidence_grade not in EVIDENCE_GRADES:
            raise ValueError(
                f"evidence_grade: {self.evidence_grade!r} not in {sorted(EVIDENCE_GRADES)}"
            )
        if self.evidence_grade == GRADE_DESIGN_PRIOR:
            # `design_prior` is the grade every hypothesis ENTERS at, and §19.1 is
            # explicit that it means "worth considering", not "probably true". Accepting
            # it as the grade of the evidence that CLOSES a question let one prior
            # resolve another — `confirmed ... (design_prior): looks right to me` — which
            # is the promotion §8.4.0/AK-D38/§19.0 rule 4 forbid, arrived at through the
            # resolution rather than through the hypothesis. It is refused HERE, on the
            # evidence record, so no caller of `resolve()` can route around it.
            raise ResolutionEvidenceMissing(
                f"evidence_grade {GRADE_DESIGN_PRIOR!r} is the grade a hypothesis ENTERS "
                "at, not a grade of evidence: it means 'worth considering' (§19.1), and "
                "a question closed by a prior is a question closed by the hunch that "
                "opened it. Resolution costs observation or better; a question nothing "
                "has been observed about stays open"
            )
        object.__setattr__(self, "evidence_refs", _require_refs(
            self.evidence_refs, "evidence_refs", error=ResolutionEvidenceMissing
        ))
        _require_text(
            self.falsifier_observed, "falsifier_observed",
            error=ResolutionEvidenceMissing,
        )
        if self.bears_on_falsifier is not True:
            # The one rule that keeps §8.4.0's second half honest. Evidence that does
            # not bear on the falsifier is an ATTEMPT; routing it here would close a
            # question on a failure that had nothing to do with it, which is the exact
            # defect this module exists to remove.
            raise ResolutionEvidenceMissing(
                "bears_on_falsifier must be exactly True to RESOLVE a hypothesis; "
                "evidence that does not bear on the falsifier is an attempt "
                "(note_attempt), and an attempt leaves the question open"
            )
        _require_text(self.resolved_by, "resolved_by")

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "evidence_grade": self.evidence_grade,
            "evidence_refs": list(self.evidence_refs),
            "falsifier_observed": self.falsifier_observed,
            "bears_on_falsifier": self.bears_on_falsifier,
            "resolved_by": self.resolved_by,
            "at": self.at,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "ResolutionEvidence":
        if not isinstance(obj, Mapping):
            raise TypeError("resolution record must be a mapping")
        missing = sorted({
            "outcome", "evidence_grade", "evidence_refs", "falsifier_observed",
            "bears_on_falsifier", "resolved_by",
        } - set(obj))
        if missing:
            raise ValueError(f"resolution record is missing {missing}")
        return ResolutionEvidence(
            outcome=obj["outcome"],
            evidence_grade=obj["evidence_grade"],
            evidence_refs=_refs_from_record(
                obj["evidence_refs"], "evidence_refs", error=ResolutionEvidenceMissing
            ),
            falsifier_observed=obj["falsifier_observed"],
            bears_on_falsifier=obj["bears_on_falsifier"],
            resolved_by=obj["resolved_by"],
            at=obj.get("at"),
        )


# =============================================================================
# The falsifier an agent writes for an operator entry (2026-08-04 amendment)
# =============================================================================

@dataclass(frozen=True)
class FalsifierProposal:
    """The predicate an agent wrote for a hypothesis the operator entered without one.

    It is a SEPARATE record, not an edit of the hypothesis, and that is the whole
    design. `Hypothesis.fingerprint` is the identity of the QUESTION; folding a
    proposed falsifier back into the statement would change that fingerprint under a
    tracked id, which is precisely what `QuestionRewritten` exists to refuse — the
    operator's next `intake()` would then be told their own unedited file states a
    different question. So the operator's words stay exactly as written, and the
    predicate sits beside them with the name of whoever wrote it.

    `rationale` is mandatory. It is the ceremony the amendment moved OFF the operator
    and ONTO the agent: an agent that cannot say why this predicate would refute that
    statement has not proposed a falsifier, it has produced a sentence.
    """

    hypothesis_id: str
    falsifier: str
    proposed_by: str
    rationale: str
    at: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.hypothesis_id, "hypothesis_id")
        state = classify_falsifier(self.falsifier)
        if state == FALSIFIER_ABSENT:
            raise FalsifierMissing(
                f"{self.hypothesis_id}: a falsifier PROPOSAL with no falsifier in it. "
                "Absence is a legal state for an operator entry and is expressed by not "
                "proposing anything; it is not something to record"
            )
        if state == FALSIFIER_PLACEHOLDER:
            raise FalsifierMissing(
                f"{self.hypothesis_id}: proposed falsifier {self.falsifier!r} is a "
                "placeholder. The optional-on-entry route exists so the OPERATOR need "
                "not write a predicate — never so an agent can satisfy the requirement "
                "by typing 'tbd'"
            )
        if "\n" in self.falsifier or "\r" in self.falsifier:
            raise FalsifierMissing(
                f"{self.hypothesis_id}: a proposed falsifier must be ONE LINE — a "
                "predicted outcome whose absence invalidates the hypothesis, not a "
                "paragraph of reasoning (§8.4.0)"
            )
        _require_text(self.proposed_by, "proposed_by")
        _require_text(
            self.rationale, "rationale",
            error=FalsifierMissing,
        )
        if self.at is not None:
            _require_text(self.at, "at")

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "falsifier": self.falsifier,
            "proposed_by": self.proposed_by,
            "rationale": self.rationale,
            "at": self.at,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "FalsifierProposal":
        if not isinstance(obj, Mapping):
            raise TypeError("falsifier proposal record must be a mapping")
        missing = sorted(
            {"hypothesis_id", "falsifier", "proposed_by", "rationale"} - set(obj)
        )
        if missing:
            raise ValueError(f"falsifier proposal record is missing {missing}")
        return FalsifierProposal(
            hypothesis_id=obj["hypothesis_id"],
            falsifier=obj["falsifier"],
            proposed_by=obj["proposed_by"],
            rationale=obj["rationale"],
            at=obj.get("at"),
        )


# =============================================================================
# The claim gate — the point at which the falsifier stops being optional
# =============================================================================

@dataclass(frozen=True)
class ClaimAuthorization:
    """Permission to spend a resource claim on ONE hypothesis. A capability, not a flag.

    THE INVARIANT IS THE TYPE. `__post_init__` re-derives the falsifier state from the
    falsifier this token carries and refuses to construct for `absent` or `placeholder`.
    So there is no such thing as a `ClaimAuthorization` naming a hypothesis with no
    usable falsifier — not one the tracker minted, not one a caller hand-built, not one
    that came back from a seam. `claim_for_hypothesis()` accepts nothing else, which is
    what makes "a caller cannot spend a claim on state (i) or (ii) by any route" a
    property of the code rather than a rule in a docstring.

    A `Check` returning FAIL would not have done this job: a caller can ignore a FAIL.
    A token that cannot exist cannot be ignored.
    """

    hypothesis_id: str
    falsifier: str
    falsifier_source: str
    origin: str
    purpose: str
    authorized_by: str
    authorized_at: str
    ledger_seq: int
    #: What the §19.2 do-not-repeat plane said about this question when the claim was
    #: authorized: `schemas.PASS`, `schemas.COULD_NOT_CHECK`, or `None` for **nobody
    #: asked**. It has NO DEFAULT for the same reason `falsifier` has none — a defaulted
    #: verdict makes "we consulted the ledger and it was clear" indistinguishable from
    #: "we never wired the ledger up", which is the exact confusion this whole seam
    #: exists to end. `schemas.FAIL` cannot be constructed: see `__post_init__`.
    do_not_repeat_outcome: Optional[str]
    #: The reasons behind that verdict, carried so an operator reading the ledger can
    #: see WHY memory allowed the spend without re-deriving it.
    do_not_repeat_reasons: tuple = ()
    campaign_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.hypothesis_id, "hypothesis_id")
        state = classify_falsifier(self.falsifier)
        if state in FALSIFIER_STATES_REFUSING_COMPUTE:
            raise FalsifierRequiredBeforeCompute(
                f"{self.hypothesis_id}: cannot authorize a resource claim — the "
                f"falsifier is {state!r}. "
                + (
                    "A falsifier is optional on operator ENTRY and mandatory before "
                    "compute is spent: call propose_falsifier() and record the "
                    "predicate, then authorize"
                    if state == FALSIFIER_ABSENT else
                    "A placeholder falsifier is illegal everywhere; it is an empty "
                    "string wearing a hat, and it is what stops nothing"
                )
            )
        if self.falsifier_source not in (
            FALSIFIER_SOURCE_STATED, FALSIFIER_SOURCE_PROPOSED
        ):
            raise ValueError(
                f"falsifier_source: {self.falsifier_source!r} not in "
                f"{[FALSIFIER_SOURCE_STATED, FALSIFIER_SOURCE_PROPOSED]}"
            )
        if self.origin not in ORIGINS:
            raise ValueError(f"origin: {self.origin!r} not in {sorted(ORIGINS)}")
        _require_text(self.purpose, "purpose")
        _require_text(self.authorized_by, "authorized_by")
        _require_text(self.authorized_at, "authorized_at")
        if (not isinstance(self.ledger_seq, int) or isinstance(self.ledger_seq, bool)
                or self.ledger_seq < 1):
            raise ValueError(
                "ledger_seq must be the positive seq of the durable "
                f"{EVENT_CLAIM_AUTHORIZED} record; an authorization with no record "
                "behind it is not an authorization"
            )
        if self.do_not_repeat_outcome == schemas.FAIL:
            # Same discipline as the falsifier, one axis over: the token cannot EXIST
            # for a receipted repeat, so there is no object a caller could choose to
            # ignore. A `Check` returning FAIL is advice; a type that refuses to be
            # constructed is a gate.
            raise RepeatsAReceiptedNegative(
                f"{self.hypothesis_id}: the do-not-repeat ledger already records this "
                "idea in this regime with a receipt (§8.4, §19.2), so no claim may be "
                "spent on it. Reopen it on new evidence, move the anchor, or state a "
                "different question — reasons: "
                + "; ".join(str(r) for r in self.do_not_repeat_reasons)
            )
        if self.do_not_repeat_outcome is not None and self.do_not_repeat_outcome not in (
            schemas.PASS, schemas.COULD_NOT_CHECK
        ):
            raise ValueError(
                f"do_not_repeat_outcome: {self.do_not_repeat_outcome!r} is not a "
                f"schemas.Check outcome; expected {schemas.PASS!r}, "
                f"{schemas.COULD_NOT_CHECK!r} or None (nobody asked)"
            )
        # An EMPTY reason list is legal here (a clear PASS has nothing to say) and a
        # bare string is not: `tuple("no match")` is nine one-character reasons, which
        # is the same shape `_refs_from_record` exists to refuse one field over.
        if (isinstance(self.do_not_repeat_reasons, (str, bytes))
                or not isinstance(self.do_not_repeat_reasons, Sequence)):
            raise TypeError("do_not_repeat_reasons must be a sequence of strings")
        for index, reason in enumerate(self.do_not_repeat_reasons):
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"do_not_repeat_reasons[{index}] must be a non-empty string"
                )
        object.__setattr__(
            self, "do_not_repeat_reasons", tuple(self.do_not_repeat_reasons)
        )
        if self.campaign_id is not None:
            _require_text(self.campaign_id, "campaign_id")

    @property
    def evidence_grade(self) -> str:
        """Still `design_prior`. Spending compute on a question does not grade it."""
        return entry_grade(self.origin)

    @property
    def claim_purpose(self) -> str:
        """What the RESOURCE CLAIM's own receipt will say this hold is for.

        The falsifier travels into the claim journal deliberately: `device_claim`
        already refuses an unattributable claim, and this makes the attribution answer
        "what would have refuted the thing we spent the card on" without a join.
        """
        return (
            f"{self.purpose} [hypothesis {self.hypothesis_id} "
            f"({self.falsifier_source}) falsifier: {self.falsifier}]"
        )

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "falsifier": self.falsifier,
            "falsifier_source": self.falsifier_source,
            "origin": self.origin,
            "purpose": self.purpose,
            "authorized_by": self.authorized_by,
            "authorized_at": self.authorized_at,
            "ledger_seq": self.ledger_seq,
            "do_not_repeat_outcome": self.do_not_repeat_outcome,
            "do_not_repeat_reasons": list(self.do_not_repeat_reasons),
            "campaign_id": self.campaign_id,
            "entry_evidence_grade": self.evidence_grade,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "ClaimAuthorization":
        if not isinstance(obj, Mapping):
            raise TypeError("claim authorization record must be a mapping")
        missing = sorted({
            "hypothesis_id", "falsifier", "falsifier_source", "origin", "purpose",
            "authorized_by", "authorized_at", "ledger_seq",
        } - set(obj))
        if missing:
            raise ValueError(f"claim authorization record is missing {missing}")
        # `do_not_repeat_outcome` is NOT in the required set, and that is the honest
        # reading of a record written before this seam existed: it did not consult the
        # ledger, so the absent key means exactly what `None` means. Requiring it would
        # turn every historical record into ledger CORRUPTION, which is a different
        # claim and a false one.
        return ClaimAuthorization(
            hypothesis_id=obj["hypothesis_id"],
            falsifier=obj["falsifier"],
            falsifier_source=obj["falsifier_source"],
            origin=obj["origin"],
            purpose=obj["purpose"],
            authorized_by=obj["authorized_by"],
            authorized_at=obj["authorized_at"],
            ledger_seq=obj["ledger_seq"],
            do_not_repeat_outcome=obj.get("do_not_repeat_outcome"),
            do_not_repeat_reasons=_reasons_from_record(
                obj.get("do_not_repeat_reasons")
            ),
            campaign_id=obj.get("campaign_id"),
        )


#: Names that ACQUIRE a resource claim. `audit_falsifier_required_before_claim()` walks
#: this module's AST for them and requires every one to sit inside
#: `claim_for_hypothesis` — one door, provable rather than promised.
#: Frozen from the two acquiring modules on 2026-08-04 —
#: `resource/device_claim.py` (`acquire_device_claim`, `gpu_device_claim`,
#: `gpu_device_claims`) and `execution/cpu_region_claim.py`
#: (`acquire_cpu_region_claim`, `cpu_region_claim`), plus the orchestrator-side
#: `cpu_region_lock` name the CPU module is the research half of. Names, not types:
#: the check is blunt on purpose, so this module simply does not use these words on
#: anything outside the door.
_CLAIM_ACQUISITION_NAMES = frozenset({
    "acquire_device_claim", "gpu_device_claim", "gpu_device_claims",
    "acquire_cpu_region_claim", "cpu_region_claim", "cpu_region_lock",
})

#: The one function allowed to call them.
_CLAIM_DOOR = "claim_for_hypothesis"


def claim_for_hypothesis(authorization: ClaimAuthorization, acquire, /, **kwargs):
    """The ONLY route from a hypothesis to a resource claim.

    `acquire` is the caller's claim acquirer — `device_claim.acquire_device_claim`, a
    CPU region claim, or a fake in a test. This module does not import an acquirer to
    call, does not choose one, and holds no claim of its own; it stands between a
    hypothesis and whichever one the caller brought.

    `purpose` is NOT accepted from the caller. It comes off the token, carries the
    falsifier the claim is being spent against, and therefore lands in the claim
    journal — so the resource record and the question record say the same thing without
    anyone having to keep them in step.

    The type check is the gate, and the state is RE-DERIVED from the token at the door
    rather than taken on trust that `__post_init__` ran. A frozen dataclass is validated
    at construction only: `copy.copy(token)` (or a pickle round-trip, or any `__reduce__`
    seam) produces an instance without calling `__init__`, and `object.__setattr__` then
    puts `"tbd"` in the falsifier of an object whose type says that cannot happen. The
    docstring above claims this refuses a token "that came back from a seam"; two lines
    make that true rather than aspirational, and it is the same re-derivation
    `fold_ledger` already performs on every CLAIM_AUTHORIZED record.
    """
    if not isinstance(authorization, ClaimAuthorization):
        raise FalsifierRequiredBeforeCompute(
            "a resource claim may be spent on a hypothesis ONLY through a "
            f"ClaimAuthorization, got {type(authorization).__name__}. Mint one with "
            "HypothesisTracker.authorize_claim(); it cannot be minted for a hypothesis "
            "whose falsifier is absent or a placeholder, which is the point"
        )
    state = classify_falsifier(authorization.falsifier)
    if state in FALSIFIER_STATES_REFUSING_COMPUTE:
        raise FalsifierRequiredBeforeCompute(
            f"{authorization.hypothesis_id}: this ClaimAuthorization carries a "
            f"{state!r} falsifier, so it was not built by the constructor that refuses "
            "one. A token whose invariant was checked once and can be edited afterwards "
            "is a flag, not a capability"
        )
    # The SECOND gate, re-derived at the door for the same reason as the first. The
    # memory plane is consulted in `authorize_claim`; a token that reaches here with no
    # verdict came from somewhere that skipped it, and "the ledger was never consulted"
    # is precisely the state that let the loop re-try dead ideas forever.
    if authorization.do_not_repeat_outcome is None:
        raise LedgerNotConsulted(
            f"{authorization.hypothesis_id}: this ClaimAuthorization carries NO "
            "do-not-repeat verdict, so the §19.2 memory plane was never asked whether "
            "this has already been tried. Not checking is not a clear result: mint the "
            "token with HypothesisTracker.authorize_claim(..., ledger=...), which is "
            "the only thing that records one"
        )
    if authorization.do_not_repeat_outcome == schemas.FAIL:
        # Unreachable through the constructor, which refuses FAIL outright — and
        # REACHABLE through `object.__setattr__` on the frozen token, which is the very
        # seam the falsifier re-derivation above exists for. Both gates fail the same
        # way, and both are tested through that seam.
        raise RepeatsAReceiptedNegative(
            f"{authorization.hypothesis_id}: this ClaimAuthorization records a "
            "receipted repeat (§8.4), so it was not built by the constructor that "
            "refuses one"
        )
    if not callable(acquire):
        raise TypeError(f"acquire must be callable, got {type(acquire).__name__}")
    if "purpose" in kwargs:
        raise ValueError(
            "purpose is taken from the authorization, never from the caller: it carries "
            "the falsifier this claim is being spent against into the claim's own "
            "receipt, and a caller-supplied purpose could say something else"
        )
    return acquire(purpose=authorization.claim_purpose, **kwargs)


# =============================================================================
# Adoption — the move that transfers ownership from the operator to the agents
# =============================================================================

@dataclass(frozen=True)
class Adoption:
    """The record that an operator hypothesis became the agents'.

    It is written to the ledger BEFORE the store entry is removed, and it carries the
    hypothesis CONTENT and the operator's own ENTRY BYTES inline. Both, deliberately:

    * `hypothesis` is the structured content, so the ledger alone can reconstruct the
      question — a record that pointed at "entry 3 of the file" would be pointing at a
      file this very operation is about to renumber.
    * `entry_text` is the operator's literal bytes, so what the record preserves is what
      they wrote, not this module's re-rendering of it.

    `store_sha256_before` binds the record to the exact bytes the removal was computed
    against; a later reader can tell whether the file they are holding is that one.
    """

    hypothesis_id: str
    adopted_by: str
    reason: str
    store_path: str
    store_sha256_before: str
    entry_index: int
    entry_text: str
    hypothesis: Mapping[str, Any] = field(default_factory=dict)
    at: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.hypothesis_id, "hypothesis_id")
        _require_text(self.adopted_by, "adopted_by")
        _require_text(self.reason, "reason")
        _require_text(self.store_path, "store_path")
        _require_text(self.store_sha256_before, "store_sha256_before")
        if (not isinstance(self.entry_index, int)
                or isinstance(self.entry_index, bool) or self.entry_index < 0):
            raise ValueError("entry_index must be a non-negative int")
        _require_text(self.entry_text, "entry_text")
        if not isinstance(self.hypothesis, Mapping) or not self.hypothesis:
            raise ValueError(
                "hypothesis: the adopted content must be recorded INLINE. A record that "
                "refers to the file it is about to mutate is an ORPHANED record waiting "
                "to happen"
            )
        schemas.canonical_json(dict(self.hypothesis))
        if self.at is not None:
            _require_text(self.at, "at")

    @property
    def owner(self) -> str:
        return OWNER_AGENTS

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "adopted_by": self.adopted_by,
            "reason": self.reason,
            "store_path": self.store_path,
            "store_sha256_before": self.store_sha256_before,
            "entry_index": self.entry_index,
            "entry_text": self.entry_text,
            "hypothesis": dict(self.hypothesis),
            "at": self.at,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "Adoption":
        if not isinstance(obj, Mapping):
            raise TypeError("adoption record must be a mapping")
        missing = sorted({
            "hypothesis_id", "adopted_by", "reason", "store_path",
            "store_sha256_before", "entry_index", "entry_text", "hypothesis",
        } - set(obj))
        if missing:
            raise ValueError(f"adoption record is missing {missing}")
        return Adoption(
            hypothesis_id=obj["hypothesis_id"],
            adopted_by=obj["adopted_by"],
            reason=obj["reason"],
            store_path=obj["store_path"],
            store_sha256_before=obj["store_sha256_before"],
            entry_index=obj["entry_index"],
            entry_text=obj["entry_text"],
            hypothesis=dict(obj.get("hypothesis") or {}),
            at=obj.get("at"),
        )


# =============================================================================
# The operator store — an intake, never a source of truth about resolution
# =============================================================================

#: Keys the operator may write. This set is exactly what is CARRIED — anything else is
#: refused, because a key this loader silently ignores is a key the operator believes
#: had an effect.
_ALLOWED_ENTRY_KEYS = frozenset({
    "hypothesis_id", "statement", "falsifier", "author", "regime", "created_at",
})

#: Keys refused with a POINTED message. Each one is a specific laundering route:
#: stating your own grade, stating your own resolution, or stating an origin other
#: than the one the file's existence already establishes.
_REFUSED_ENTRY_KEYS: Mapping[str, str] = {
    "evidence_grade": (
        "evidence grade is DERIVED and is always 'design_prior' for a hypothesis "
        "(§8.4.0, AK-D38, §19.0 rule 4); a store that could state its own grade is a "
        "store that can launder a hunch into a measured fact"
    ),
    "grade": "see 'evidence_grade': grade is derived, never stated",
    "status": (
        "status lives in the ledger, not in the store; a hypothesis is resolved by "
        "evidence (resolve()), never by editing a file"
    ),
    "resolution": "see 'status': resolution requires the evidence that resolved it",
    "resolved": "see 'status': resolution requires the evidence that resolved it",
    "outcome": "see 'status': resolution requires the evidence that resolved it",
    "origin": (
        "every entry in the operator store has origin 'operator' by construction; "
        "stating another origin would let the channel relabel its own provenance"
    ),
    "priority": (
        "there is no priority anywhere in this mechanism — §8.4.0: an operator "
        "hypothesis is not a queue-jumping mechanism"
    ),
    "rank": "see 'priority': this mechanism has no ranking",
    "weight": "see 'priority': this mechanism has no ranking",
    "notes": (
        "a hypothesis is its statement and its falsifier; there is no third prose "
        "field, because free text beside a tracked question is exactly the planner "
        "narrative invariant 20 keeps out of a later planning context"
    ),
}


@dataclass(frozen=True)
class EntrySpan:
    """One store entry's own BYTE RANGE in the operator's file.

    `start`/`end` are character offsets into the decoded text, `end` exclusive, with
    trailing whitespace trimmed. The point of carrying offsets rather than parsed
    objects is that removal is a SPLICE: every entry that is not the one being removed
    is carried across as the operator typed it, comments-free JSON or not, whatever the
    indentation, key order, spacing or unicode escaping they chose.
    """

    hypothesis_id: str
    index: int
    start: int
    end: int

    def text_of(self, store_text: str) -> str:
        """The operator's own characters for this entry, out of the text scanned."""
        return store_text[self.start:self.end]


@dataclass(frozen=True)
class StoreRemoval:
    """What one atomic store rewrite did."""

    store_path: str
    hypothesis_id: str
    entry_index: int
    entry_text: str
    sha256_before: str
    sha256_after: str
    remaining_ids: tuple = ()


def _scan_string(text: str, index: int) -> int:
    """Index just past the JSON string starting at `text[index] == '"'`."""
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    raise StoreRewriteRefused("unterminated string while scanning the store text")


def _scan_value(text: str, index: int) -> int:
    """Index just past the JSON value starting at `index`. Balanced, string-aware."""
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        raise StoreRewriteRefused("store text ends where a value was expected")
    if text[index] == '"':
        return _scan_string(text, index)
    if text[index] in "{[":
        depth = 0
        while index < len(text):
            char = text[index]
            if char == '"':
                index = _scan_string(text, index)
                continue
            if char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
        raise StoreRewriteRefused("unbalanced container while scanning the store text")
    # A bare literal: number, true, false, null.
    start = index
    while index < len(text) and text[index] not in ",}]" and not text[index].isspace():
        index += 1
    if index == start:
        raise StoreRewriteRefused(f"unreadable token at offset {start} in the store")
    return index


def _locate_hypotheses_array(text: str) -> int:
    """Offset of the `[` that opens the top-level `hypotheses` array.

    A structural walk of the top-level object rather than a regex: a regex for
    `"hypotheses"` would happily match the same characters inside a statement, and the
    statement is operator prose — the one field guaranteed to contain whatever it likes.
    """
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "{":
        raise StoreRewriteRefused("the store text is not a JSON object")
    index += 1
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == "}":
            break
        if index >= len(text) or text[index] != '"':
            raise StoreRewriteRefused(
                f"expected a top-level key at offset {index} in the store"
            )
        key_end = _scan_string(text, index)
        key = json.loads(text[index:key_end])
        index = key_end
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != ":":
            raise StoreRewriteRefused(f"expected ':' after key {key!r} in the store")
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
        if key == "hypotheses":
            if index >= len(text) or text[index] != "[":
                raise StoreRewriteRefused("'hypotheses' is not a JSON array")
            return index
        index = _scan_value(text, index)
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == ",":
            index += 1
    raise StoreRewriteRefused("the store text has no 'hypotheses' key")


def _element_spans(text: str, array_start: int) -> tuple:
    """`(start, end)` for every element of the array opening at `array_start`."""
    spans: list = []
    index = array_start + 1
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        if text[index] == "]":
            return tuple(spans)
        start = index
        index = _scan_value(text, index)
        spans.append((start, index))
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return tuple(spans)
        raise StoreRewriteRefused(
            f"expected ',' or ']' at offset {index} in the store's hypotheses array"
        )
    raise StoreRewriteRefused("the store's hypotheses array is unterminated")


class OperatorHypothesisStore:
    """The operator-editable file of stated hypotheses.

    JSON rather than YAML on purpose. Every other durable file in this package is
    JSON, and YAML's implicit typing is a hazard for exactly the field that carries the
    safety property: a falsifier reading `no` or `NO` becomes the boolean `False`, and
    an unquoted `1.5-3` becomes a string that no longer says what it said. A falsifier
    that silently changes type is a falsifier that silently stops being one.

    Shape::

        {
          "schema": "epyc.autokernel.operator_hypotheses.v1",
          "hypotheses": [
            {
              "hypothesis_id": "akh-g15-elementwise-fusion",
              "statement": "G15's elementwise/norm cluster is where the B=128 decode time is, and fusing it lands >= 15%",
              "falsifier": "a current wall-share map shows the cluster under 20%",
              "author": "operator",
              "regime": {"backend": "llama_gpu", "phase": "decode", "batch_band": "b128"}
            }
          ]
        }

    An empty `hypotheses` list is VALID and means the operator has none. That is
    precisely why every malformed input raises instead.
    """

    __slots__ = ("path", "lock_path")

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(_require_text(path, "store path"))
        self.lock_path = self.path + ADOPTION_LOCK_SUFFIX

    def exists(self) -> bool:
        """Whether the store file is there.

        `load()` RAISES on absence rather than returning nothing, so a caller that
        legitimately has no operator channel expresses that by passing no store at all
        — not by pointing at a path that may or may not have been mounted.
        """
        return os.path.exists(self.path)

    def read_bytes(self) -> bytes:
        try:
            with open(self.path, "rb") as handle:
                return handle.read()
        except FileNotFoundError as exc:
            raise HypothesisStoreError(
                f"{self.path}: operator hypothesis store does not exist. An absent "
                "store is not an empty one — pass no store at all to declare that "
                "this campaign has no operator channel"
            ) from exc
        except OSError as exc:
            raise HypothesisStoreError(
                f"{self.path}: operator hypothesis store could not be read ({exc})"
            ) from exc

    def _digest(self, raw: bytes) -> str:
        """SHA-256 of the store's bytes, or a REFUSAL if they are not UTF-8.

        `errors="surrogateescape"` used to be here, which made this the one public
        store method whose failure escaped the module's error contract: the lone
        surrogates it produces are legal in a `str` but cannot be re-encoded, so
        `content_hash` raised a bare `UnicodeEncodeError` — not a `HypothesisStoreError`,
        not even a `ControllerError`, so a driver catching the controller plane died on
        it. A store whose bytes are not UTF-8 is an unreadable store, and this module
        has exactly one way to say that.
        """
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HypothesisStoreError(
                f"{self.path}: operator hypothesis store is not valid UTF-8 ({exc}); "
                "an undecodable store is unreadable, not empty"
            ) from exc
        return schemas.content_hash({"bytes": text})

    def content_sha256(self) -> str:
        return self._digest(self.read_bytes())

    def load(self) -> tuple:
        """Every stated hypothesis, or a refusal. NEVER a degraded empty list."""
        return self.load_with_digest()[0]

    def load_with_digest(self) -> tuple:
        """`(hypotheses, store_sha256)` from ONE read of the file.

        One read, not two: a digest taken from a second read describes a file that may
        no longer be the one that was parsed, and the digest exists precisely so a
        later reader can tell which bytes a tracked hypothesis came from.
        """
        raw = self.read_bytes()
        if not raw.strip():
            raise HypothesisStoreError(
                f"{self.path}: store file is empty. An unreadable store is not an "
                "absent one, and an absent one is not 'the operator has no "
                "hypotheses' — that statement is written as an empty 'hypotheses' list"
            )
        try:
            obj = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise HypothesisStoreError(
                f"{self.path}: unparseable operator hypothesis store: {exc}"
            ) from exc
        if not isinstance(obj, Mapping):
            raise HypothesisStoreError(
                f"{self.path}: store must be a JSON object with 'schema' and "
                f"'hypotheses', got {type(obj).__name__}"
            )
        if obj.get("schema") != STORE_SCHEMA:
            raise HypothesisStoreError(
                f"{self.path}: schema is {obj.get('schema')!r}, expected "
                f"{STORE_SCHEMA!r}; the schema string is the file's identity, not "
                "metadata beside it"
            )
        entries = obj.get("hypotheses")
        if entries is None:
            raise HypothesisStoreError(
                f"{self.path}: 'hypotheses' key is absent. Write [] to say the "
                "operator has none; absence is a truncated file"
            )
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise HypothesisStoreError(
                f"{self.path}: 'hypotheses' must be a list, got "
                f"{type(entries).__name__}"
            )
        unknown_top = sorted(set(obj) - {"schema", "hypotheses"})
        if unknown_top:
            raise HypothesisStoreError(
                f"{self.path}: unknown top-level keys {unknown_top}; a key this "
                "loader ignores is a key the operator believes had an effect"
            )

        store_sha = self._digest(raw)
        loaded: list = []
        seen: dict = {}
        for index, entry in enumerate(entries):
            hypothesis = self._load_entry(entry, index, store_sha)
            if hypothesis.hypothesis_id in seen:
                raise HypothesisStoreError(
                    f"{self.path}: hypothesis_id {hypothesis.hypothesis_id!r} appears "
                    f"at entries {seen[hypothesis.hypothesis_id]} and {index}; one id "
                    "is one question"
                )
            seen[hypothesis.hypothesis_id] = index
            loaded.append(hypothesis)
        return tuple(loaded), store_sha

    def _load_entry(self, entry: Any, index: int, store_sha: str) -> Hypothesis:
        where = f"{self.path}[{index}]"
        if not isinstance(entry, Mapping):
            raise HypothesisStoreError(
                f"{where}: each hypothesis must be an object, got "
                f"{type(entry).__name__}"
            )
        for key, why in _REFUSED_ENTRY_KEYS.items():
            if key in entry:
                raise HypothesisStoreError(f"{where}: key {key!r} is refused — {why}")
        unknown = sorted(set(entry) - _ALLOWED_ENTRY_KEYS)
        if unknown:
            raise HypothesisStoreError(
                f"{where}: unknown keys {unknown}; allowed keys are "
                f"{sorted(_ALLOWED_ENTRY_KEYS)}"
            )
        flagged = schemas.find_authority_flavoured_keys(dict(entry))
        if flagged:
            raise HypothesisStoreError(
                f"{where}: authority-flavoured keys {flagged}. §8.4.0 — the channel is "
                "steering WITHOUT authority; a hypothesis is a proposal source and "
                "carries no freeze, cutover, promotion or ratification flag"
            )

        for key in ("hypothesis_id", "statement"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise HypothesisStoreError(f"{where}: {key} is required and non-empty")
        # 2026-08-04: an operator entry MAY omit the falsifier entirely, and every entry
        # in this file is `operator` origin by construction. What it may not do is
        # supply a placeholder — `Hypothesis._check_falsifier` refuses that below, and
        # an omitted key and `"tbd"` therefore land in two different states rather than
        # one. `.get(...)` returning None is the honest state, not a default: the entry
        # said nothing, so the record says nothing.
        regime = entry.get("regime", {})
        if not isinstance(regime, Mapping):
            raise HypothesisStoreError(
                f"{where}: regime must be an object of match dimensions "
                f"(§19.2 — 'do not repeat' without regime identity is dangerous)"
            )
        author = entry.get("author", ORIGIN_OPERATOR)
        if not isinstance(author, str) or not author.strip():
            raise HypothesisStoreError(f"{where}: author must be a non-empty string")

        source = {
            "kind": "operator_store",
            "path": self.path,
            "entry_index": index,
            "store_sha256": store_sha,
        }
        try:
            return Hypothesis(
                hypothesis_id=entry["hypothesis_id"],
                statement=entry["statement"],
                falsifier=entry.get("falsifier"),
                origin=ORIGIN_OPERATOR,
                author=author,
                regime=dict(regime),
                source=source,
                created_at=entry.get("created_at"),
            )
        except FalsifierMissing as exc:
            raise FalsifierMissing(f"{where}: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise HypothesisStoreError(f"{where}: {exc}") from exc

    # ---- the operator's own bytes ------------------------------------------

    def entry_spans(self) -> tuple:
        """`(text, spans)`: the store's decoded text and one `EntrySpan` per entry.

        The spans are VERIFIED against the parse, not trusted: each sliced span is
        re-parsed and must equal the object `json.loads` found at that index. A scanner
        that drifted by one character would otherwise produce a splice that removes the
        wrong bytes from an operator-owned file, and would produce it silently.
        """
        raw = self.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HypothesisStoreError(
                f"{self.path}: operator hypothesis store is not valid UTF-8 ({exc})"
            ) from exc
        try:
            parsed = json.loads(text, object_pairs_hook=_no_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            # The splice path refuses a duplicated key for a sharper reason than the
            # loader does: a second `"hypotheses"` array is a file the parser and the
            # scanner read differently, and a splice computed against the one they
            # disagree about cuts bytes nobody is reading.
            raise HypothesisStoreError(
                f"{self.path}: unparseable operator hypothesis store: {exc}"
            ) from exc
        entries = (parsed or {}).get("hypotheses") if isinstance(parsed, Mapping) else None
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise StoreRewriteRefused(
                f"{self.path}: 'hypotheses' is not a list; refusing to splice a file "
                "whose shape this module does not recognise"
            )
        offsets = _element_spans(text, _locate_hypotheses_array(text))
        if len(offsets) != len(entries):
            raise StoreRewriteRefused(
                f"{self.path}: the scanner found {len(offsets)} entry span(s) but the "
                f"parser found {len(entries)}; refusing to rewrite an operator-owned "
                "file this module cannot read the same way twice"
            )
        spans: list = []
        for index, ((start, end), entry) in enumerate(zip(offsets, entries)):
            sliced = text[start:end]
            try:
                round_tripped = json.loads(sliced)
            except json.JSONDecodeError as exc:
                raise StoreRewriteRefused(
                    f"{self.path}[{index}]: the scanned span does not parse ({exc})"
                ) from exc
            if round_tripped != entry:
                raise StoreRewriteRefused(
                    f"{self.path}[{index}]: the scanned span is not the parsed entry; "
                    "the byte offsets have drifted and a splice would cut the wrong text"
                )
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("hypothesis_id"), str
            ):
                raise StoreRewriteRefused(
                    f"{self.path}[{index}]: entry has no string hypothesis_id"
                )
            spans.append(EntrySpan(
                hypothesis_id=entry["hypothesis_id"], index=index, start=start, end=end,
            ))
        return text, tuple(spans)

    def remove_entry(
        self, hypothesis_id: str, *, expected_sha256: Optional[str] = None
    ) -> StoreRemoval:
        """Rewrite the store with EXACTLY ONE entry spliced out. Atomic.

        Three properties, all checked rather than intended:

        1. **Byte-for-byte for everything else.** The new text is the old text with one
           span (and its separating comma) removed. Nothing is re-serialized, so key
           order, indentation, spacing, unicode escaping and the operator's own line
           breaks survive. The check that this HELD is span-by-span: after splicing, the
           file is re-scanned and every remaining entry's text must equal its text
           before.
        2. **Atomic.** Written to a temp file in the same directory, fsynced, `os.replace`d,
           and the directory fsynced. A crash never leaves a half-written store, which
           for an operator-owned file would be the worst outcome available.
        3. **Refused rather than approximated.** Every failure raises before any byte is
           replaced. `expected_sha256`, when given, is the digest the caller computed
           the removal against; a mismatch means the operator edited the file underneath
           this operation, and their edit wins.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        text, spans = self.entry_spans()
        digest_before = self._digest(text.encode("utf-8"))
        if expected_sha256 is not None and digest_before != expected_sha256:
            raise StoreRewriteRefused(
                f"{self.path}: the store changed under this operation "
                f"({digest_before} != {expected_sha256}); the operator's edit wins and "
                "nothing was removed"
            )
        target = [s for s in spans if s.hypothesis_id == hypothesis_id]
        if not target:
            raise HypothesisNotInStore(
                f"{self.path}: no entry with hypothesis_id {hypothesis_id!r}; present "
                f"ids are {[s.hypothesis_id for s in spans]}"
            )
        if len(target) > 1:  # pragma: no cover - load() refuses a duplicated id first
            raise StoreRewriteRefused(
                f"{self.path}: {hypothesis_id!r} appears {len(target)} times; one id is "
                "one question and this module will not guess which one to remove"
            )
        span = target[0]
        entry_text = span.text_of(text)

        position = span.index
        if len(spans) == 1:
            cut_from, cut_to = span.start, span.end
        elif position < len(spans) - 1:
            # Through the START of the next entry: this takes the separating comma and
            # the whitespace before the next entry, so the next entry lands on exactly
            # the indentation this one had.
            cut_from, cut_to = span.start, spans[position + 1].start
        else:
            # The last entry: take the comma that PRECEDES it, back from the end of the
            # previous entry, so no trailing comma is left behind.
            cut_from, cut_to = spans[position - 1].end, span.end
        new_text = text[:cut_from] + text[cut_to:]

        remaining = self._verify_splice(new_text, spans, text, span)
        raw_after = new_text.encode("utf-8")
        self._atomic_write(raw_after)
        return StoreRemoval(
            store_path=self.path,
            hypothesis_id=hypothesis_id,
            entry_index=span.index,
            entry_text=entry_text,
            sha256_before=digest_before,
            sha256_after=self._digest(raw_after),
            remaining_ids=remaining,
        )

    def _verify_splice(
        self, new_text: str, spans: tuple, old_text: str, removed: EntrySpan
    ) -> tuple:
        """Prove the spliced text is the old one minus exactly one entry. Pure."""
        try:
            parsed = json.loads(new_text)
        except json.JSONDecodeError as exc:
            raise StoreRewriteRefused(
                f"{self.path}: the spliced store does not parse ({exc}); nothing was "
                "written"
            ) from exc
        if not isinstance(parsed, Mapping) or parsed.get("schema") != STORE_SCHEMA:
            raise StoreRewriteRefused(
                f"{self.path}: the spliced store is not a {STORE_SCHEMA} document"
            )
        try:
            new_spans = _element_spans(new_text, _locate_hypotheses_array(new_text))
        except StoreRewriteRefused as exc:
            raise StoreRewriteRefused(
                f"{self.path}: the spliced store cannot be re-scanned ({exc})"
            ) from exc
        expected = [s for s in spans if s.index != removed.index]
        if len(new_spans) != len(expected):
            raise StoreRewriteRefused(
                f"{self.path}: splicing removed {len(spans) - len(new_spans)} entries, "
                "not 1; nothing was written"
            )
        for kept, (start, end) in zip(expected, new_spans):
            before = old_text[kept.start:kept.end]
            after = new_text[start:end]
            if before != after:
                raise StoreRewriteRefused(
                    f"{self.path}: entry {kept.hypothesis_id!r} would be REWRITTEN by "
                    "this removal. The store is operator-owned: an entry that is not "
                    "being removed must come through byte for byte"
                )
        return tuple(s.hypothesis_id for s in expected)

    def _atomic_write(self, raw: bytes) -> None:
        """Temp file, fsync, `os.replace`, fsync the directory. Never in place.

        THE RETURN VALUE OF `os.write` IS CHECKED, and it is not a formality.
        `os.write` is `write(2)`: on a filesystem that fills mid-call it writes what it
        can and RETURNS THE SHORT COUNT rather than raising, and the bytes it did not
        write are simply gone. Unchecked, this method then fsynced a truncated file,
        `os.replace`d it over the operator's store, and `remove_entry` RETURNED
        SUCCESS — leaving a store that no longer parses and every hypothesis after the
        cut permanently lost. Atomicity guarantees that the replacement is all-or-
        nothing; it says nothing about whether the thing being put there is the whole
        file.

        `HypothesisLedger.append` already checks exactly this, for exactly this reason.
        The store is the operator's OWN file and the only copy of anything they have
        not yet had adopted, so the asymmetry ran the wrong way.

        The write is retried from the short offset first, because a short write is not
        by itself an error — a signal can cause one on a file that has plenty of room.
        Only a write that cannot make progress is a refusal, and it refuses BEFORE
        `os.replace`, so the store on disk is untouched.
        """
        directory = os.path.dirname(self.path) or "."
        try:
            mode = os.stat(self.path).st_mode & 0o7777
        except OSError:  # pragma: no cover - the store was read moments ago
            mode = 0o644
        fd, tmp = tempfile.mkstemp(
            dir=directory, prefix="." + os.path.basename(self.path) + ".", suffix=".tmp"
        )
        try:
            written = 0
            while written < len(raw):
                just = os.write(fd, raw[written:])
                if just <= 0:
                    break
                written += just
            if written != len(raw):
                raise StoreRewriteRefused(
                    f"{self.path}: short write to the replacement store ({written} of "
                    f"{len(raw)} bytes) — the filesystem could not take the whole file. "
                    "Replacing the store with it would atomically install a TRUNCATED "
                    "operator file and lose every hypothesis after the cut; nothing was "
                    "written"
                )
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:  # pragma: no cover
                pass
            raise
        else:
            os.close(fd)
        try:
            os.chmod(tmp, mode)
            os.replace(tmp, self.path)
        except OSError as exc:
            try:
                os.unlink(tmp)
            except OSError:  # pragma: no cover
                pass
            raise StoreRewriteRefused(
                f"{self.path}: could not replace the store atomically ({exc})"
            ) from exc
        _fsync_dir(directory)

    # ---- the adoption lock -------------------------------------------------

    @contextmanager
    def adoption_lock(self, *, timeout_s: float = 30.0, poll_s: float = 0.05):
        """Exclude a second adopter. `flock` plus PID+start-time liveness, no heartbeat.

        The same discipline as `resource/device_claim.py`, and deliberately not a second
        idiom: `flock(LOCK_EX|LOCK_NB)` on a never-unlinked sidecar is the exclusion
        fact, the kernel releases it on process death, and the holder payload
        (`device_claim.current_holder_identity()` — pid, `/proc` start ticks, boot id,
        host) exists only so a blocked adopter can SAY who is in front of it and so a
        payload left by a crash can be classified. `device_claim.assess_holder_liveness`
        does that classification, and `unknown` is a third outcome that never licenses
        anything.

        A payload naming a LIVE holder while the lock is FREE is the inconsistency
        `device_claim` calls `DeviceClaimInconsistent`: two records disagree about who
        holds the file, so nothing is taken.
        """
        if poll_s <= 0:
            raise ValueError(f"poll_s must be positive, got {poll_s!r}")
        directory = os.path.dirname(self.lock_path) or "."
        os.makedirs(directory, exist_ok=True)
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
        single_attempt = timeout_s is not None and timeout_s <= 0
        holder = _device_claim.current_holder_identity("autokernel.hypotheses.adopt")
        while True:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            acquired = False
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError:
                    acquired = False
                if acquired:
                    self._check_stale_holder(fd, holder)
                    os.ftruncate(fd, 0)
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.write(fd, (schemas.canonical_json({
                        "store_path": self.path,
                        "holder": holder,
                        "acquired_at": _iso_now(),
                    }) + "\n").encode("utf-8"))
                    os.fsync(fd)
                    try:
                        yield holder
                    finally:
                        try:
                            os.ftruncate(fd, 0)
                            os.fsync(fd)
                        finally:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                    return
            finally:
                # Closing the descriptor releases the flock as well; the explicit
                # LOCK_UN above is so the release is visible in the code rather than
                # implied by a close.
                os.close(fd)
            if single_attempt or (deadline is not None and time.monotonic() >= deadline):
                raise AdoptionLockUnavailable(
                    f"{self.lock_path}: the operator store's adoption lock was held for "
                    f"the whole budget ({timeout_s}s). {self._holder_note()}"
                )
            time.sleep(poll_s)

    def _check_stale_holder(self, fd: int, holder: Mapping[str, Any]) -> None:
        """Refuse a lock whose recorded holder is alive or unverifiable."""
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 65536)
        if not raw.strip():
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if not isinstance(payload, Mapping):
            raise AdoptionLockInconsistent(
                f"{self.lock_path}: the lock was free but its payload is unreadable; a "
                "record this module cannot classify is not a record it will overwrite"
            )
        recorded = payload.get("holder")
        # "That payload is MINE" — needed so a process interrupted between taking the
        # lock and releasing it cannot lock ITSELF out of its own lock (it would assess
        # its own live pid and refuse). It must be the WHOLE identity
        # `current_holder_identity()` mints, HOST INCLUDED. Comparing only
        # pid/start_ticks/boot_id made `host` the one field that says "another machine"
        # and the one field not consulted — and containers sharing a kernel share
        # `/proc/sys/kernel/random/boot_id` while each has its own PID namespace, so a
        # (pid, start_ticks, boot_id) collision between two sessions on one store is
        # ordinary rather than exotic. The effect was the worst available: the
        # self-recognition path returns BEFORE `assess_holder_liveness`, so a LIVE
        # holder's lock was taken silently and two adopters ran at once.
        if isinstance(recorded, Mapping) and all(
            recorded.get(field) == holder.get(field)
            for field in ("pid", "start_ticks", "boot_id", "host")
        ):
            return
        liveness = _device_claim.assess_holder_liveness(recorded)
        if liveness.state == _device_claim.DEAD:
            return
        raise AdoptionLockInconsistent(
            f"{self.lock_path}: the lock was free but its recorded holder is "
            f"{liveness.state} ({liveness.reason}); 'unknown' is never a soft 'dead' "
            "and nothing was taken"
        )

    def _holder_note(self) -> str:
        """Who the lock file says is holding it, for a timeout message. Never raises."""
        try:
            with open(self.lock_path, "rb") as handle:
                payload = json.loads(handle.read().decode("utf-8"))
            recorded = payload.get("holder")
            liveness = _device_claim.assess_holder_liveness(recorded)
            return (
                f"recorded holder pid={recorded.get('pid')} "
                f"host={recorded.get('host')} liveness={liveness.state} "
                f"({liveness.reason})"
            )
        except Exception:  # noqa: BLE001 - a diagnostic that raises is not a diagnostic
            return "the lock file names no readable holder"


# =============================================================================
# The ledger
# =============================================================================

@dataclass(frozen=True)
class LedgerEvent:
    """One append-only line. `seq` is the ordering truth; `at` is diagnostic."""

    seq: int
    kind: str
    hypothesis_id: str
    at: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 1:
            raise ValueError("seq must be a positive int")
        if self.kind not in LEDGER_EVENT_KINDS:
            raise ValueError(
                f"kind: {self.kind!r} not in {list(LEDGER_EVENT_KINDS)}; the ledger "
                "vocabulary is closed"
            )
        _require_text(self.hypothesis_id, "hypothesis_id")
        _require_text(self.at, "at")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        schemas.canonical_json(dict(self.payload))

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "hypothesis_id": self.hypothesis_id,
            "at": self.at,
            "payload": dict(self.payload),
        }

    @property
    def receipt(self) -> str:
        return schemas.content_hash(self.to_dict())

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "LedgerEvent":
        if not isinstance(obj, Mapping):
            raise TypeError("ledger record must be a mapping")
        missing = sorted({"seq", "kind", "hypothesis_id", "at"} - set(obj))
        if missing:
            raise ValueError(f"ledger record is missing {missing}")
        return LedgerEvent(
            seq=obj["seq"],
            kind=obj["kind"],
            hypothesis_id=obj["hypothesis_id"],
            at=obj["at"],
            payload=dict(obj.get("payload") or {}),
        )


@dataclass(frozen=True)
class LedgerRead:
    events: tuple
    discarded_tail_bytes: int


class HypothesisLedger:
    """Append-only, fsynced, one JSON line per event.

    Same discipline as `state_machine.TransitionLedger` — a trailing fragment with no
    newline is a TORN APPEND (the process died mid-write), so the event never took
    effect and discarding it restores a position the record supports. The discarded
    byte count is REPORTED, never swallowed: invariant 7 says outcomes are durable, and
    a silently truncated ledger is a lost question.
    """

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(_require_text(path, "ledger path"))

    def initialize(self) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(directory)

    def append(self, event: LedgerEvent) -> LedgerEvent:
        """Write and fsync one event. Returning means DURABLE.

        REFUSES to write onto a ledger that cannot be read back, and in particular onto
        a TORN TAIL. `read()` discards a trailing fragment because the record it
        describes never took effect — but an O_APPEND write lands immediately after
        those bytes, fusing the fragment and the new record into one unparseable line.
        That line is not a torn tail (it ends in a newline), so from the next read
        onward `read()` raises and EVERY question in the ledger becomes unreadable,
        permanently, from one write that looked like it succeeded. `journal.py` faces
        the same hazard and repairs the tail under the write lock before appending
        (`_repair_torn_tail_locked`); this module has no kind in which to receipt a
        discard, so it refuses and leaves the bytes for repair rather than destroying
        the record or silently dropping them.
        """
        if not isinstance(event, LedgerEvent):
            raise TypeError(f"event must be a LedgerEvent, got {type(event).__name__}")
        # Validates through the reader on purpose: "readable back" is defined in exactly
        # one place, the same discipline `HypothesisTracker._record` applies by
        # re-folding the candidate ledger before writing. O(ledger) per append, at the
        # same scale and for the same reason as the rest of this module.
        before = self.read()
        if before.discarded_tail_bytes:
            raise HypothesisLedgerCorruption(
                f"{self.path}: {before.discarded_tail_bytes} byte(s) of torn tail have "
                "not been repaired; appending would fuse them to this record and make "
                "the whole ledger unparseable. Truncate the unterminated trailing "
                "fragment (it describes an event that never took effect) and retry"
            )
        line = (schemas.canonical_json(event.to_dict()) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = os.write(fd, line)
            if written != len(line):  # pragma: no cover - short write on a regular file
                raise HypothesisLedgerCorruption(
                    f"{self.path}: short write ({written} of {len(line)} bytes)"
                )
            os.fsync(fd)
        finally:
            os.close(fd)
        return event

    def read(self) -> LedgerRead:
        """Every durable event, or a REFUSAL. Never a degraded empty history.

        An ABSENT ledger is not an empty one, and an UNREADABLE one is not either —
        the same distinction `OperatorHypothesisStore.read_bytes` draws, which this
        method used to draw the other way round. `os.path.exists(...) -> LedgerRead((), 0)`
        meant a ledger that had been deleted, unmounted, or renamed under a live tracker
        read back as *"nothing has ever been tried"*: `state()` returned `{}`,
        `still_open()` returned `()`, and the very next `intake()` re-opened every
        operator hypothesis as brand new — destroying its attempts, its adoption and its
        resolution while reporting a clean run. That is the memory-update plane's own
        failure mode reached from the other side: the loop can no longer tell "tried and
        failed" from "never tried", and it is invisible.

        An empty FILE still reads as an empty history: `initialize()` creates exactly
        that, and a ledger somebody established with nothing in it is a real statement.
        A ledger nobody established is not.
        """
        try:
            with open(self.path, "rb") as handle:
                data = handle.read()
        except FileNotFoundError as exc:
            raise HypothesisLedgerCorruption(
                f"{self.path}: the hypothesis ledger is not there. An absent ledger is "
                "not an empty one — call initialize() to establish it. Reading absence "
                "as 'nothing has been tried' would re-open every tracked question as "
                "new and discard the receipts that say what was already attempted"
            ) from exc
        except OSError as exc:
            raise HypothesisLedgerCorruption(
                f"{self.path}: the hypothesis ledger could not be read ({exc}); an "
                "unreadable ledger is not an empty one, and this module has exactly one "
                "way to say so — a bare OSError escapes the controller error contract"
            ) from exc
        if not data:
            return LedgerRead((), 0)
        cut = data.rfind(b"\n")
        if cut == -1:
            return LedgerRead((), len(data))
        tail = len(data) - (cut + 1)
        # `data[:cut + 1]` always ends in the newline `cut` names, so `split` always
        # yields one trailing empty element that is a SEPARATOR ARTEFACT, not a line.
        # Dropping exactly that element is what lets an EMPTY line be refused: skipping
        # every empty element (the shape `state_machine.TransitionLedger.read()` still
        # has) silently discards a real blank line in the middle of the file while
        # `discarded_tail_bytes` stays 0, so a ledger that lost a record reads as intact.
        lines = data[: cut + 1].split(b"\n")[:-1]
        events: list = []
        for line_number, raw in enumerate(lines, start=1):
            if not raw.strip():
                raise HypothesisLedgerCorruption(
                    f"{self.path}:{line_number}: blank line — an append-only ledger has "
                    "no empty records, so a blank line is a lost or overwritten one"
                )
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HypothesisLedgerCorruption(
                    f"{self.path}:{line_number}: unparseable ledger line: {exc}"
                ) from exc
            try:
                events.append(LedgerEvent.from_dict(obj))
            except (TypeError, ValueError) as exc:
                raise HypothesisLedgerCorruption(
                    f"{self.path}:{line_number}: invalid ledger event: {exc}"
                ) from exc
        seqs = [e.seq for e in events]
        if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
            raise HypothesisLedgerCorruption(
                f"{self.path}: event seq numbers are not strictly increasing: {seqs}"
            )
        return LedgerRead(tuple(events), tail)


class HypothesisRecorder(Protocol):
    """The seam every ledger event is recorded through.

    `record()` returns the event AS RECORDED and RAISES if it could not be made
    durable. Nothing observable changes until it returns, so a recorder that raises is
    an event that did not happen. This is the swap point for the day `journal.py`
    grows hypothesis kinds.
    """

    def record(self, event: LedgerEvent) -> LedgerEvent:
        ...


class JournalOrderedRecorder:
    """The shipped recorder: ledger append under the JOURNAL write lock.

    The lock is not for this file's own consistency — a single O_APPEND write does not
    need one. It is so the hypothesis ledger's order is TOTAL with respect to journal
    appends, which is what lets a later reader put "this evaluation event" and "this
    hypothesis was resolved by it" in one order rather than two.
    """

    __slots__ = ("_journal", "_ledger")

    def __init__(self, journal_: journal.Journal, ledger: HypothesisLedger) -> None:
        if not isinstance(journal_, journal.Journal):
            raise TypeError("journal_ must be a journal.Journal")
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("ledger must be a HypothesisLedger")
        self._journal = journal_
        self._ledger = ledger

    def record(self, event: LedgerEvent) -> LedgerEvent:
        with self._journal.write_lock():
            return self._ledger.append(event)


# =============================================================================
# The fold — ledger events to current state
# =============================================================================

@dataclass(frozen=True)
class TrackedHypothesis:
    """One question and everything the ledger says has happened to it."""

    hypothesis: Hypothesis
    status: str
    opened_seq: int
    opened_at: str
    attempts: tuple = ()
    resolution: Optional[ResolutionEvidence] = None
    resolved_at: Optional[str] = None
    reopen_count: int = 0
    #: Resolutions that a later reopen replaced. Kept, never dropped: derived views
    #: may rewind, evidence does not disappear (invariant 8).
    superseded_resolutions: tuple = ()
    #: The predicate an agent wrote for an operator entry that arrived without one.
    #: Beside the hypothesis, never folded into it — `Hypothesis.fingerprint` is the
    #: identity of the question and must not move when a falsifier is supplied.
    falsifier_proposal: Optional[FalsifierProposal] = None
    #: Set once, by `adopt()`. Its presence IS agent ownership.
    adoption: Optional[Adoption] = None
    #: Every time compute was committed to this question, with the falsifier it was
    #: committed against.
    claim_authorizations: tuple = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status: {self.status!r} not in {sorted(STATUSES)}")

    @property
    def hypothesis_id(self) -> str:
        return self.hypothesis.hypothesis_id

    @property
    def is_open(self) -> bool:
        return self.status == STATUS_OPEN

    @property
    def evidence_grade(self) -> str:
        """The hypothesis's ENTRY grade — `design_prior`, open or resolved.

        It delegates rather than storing, so there is no slot for a resolution to
        write a higher grade into.
        """
        return self.hypothesis.evidence_grade

    @property
    def falsifier(self) -> Optional[str]:
        """The predicate in force: the hypothesis's own, else an agent's proposal.

        The hypothesis's own WINS when both exist, which cannot happen —
        `propose_falsifier()` refuses a question that already has one and the fold
        refuses the event — and the precedence is written down anyway so that if the
        impossible state ever reaches a reader, the OPERATOR'S words are the ones it
        reports.
        """
        if self.hypothesis.falsifier_state == FALSIFIER_STATED:
            return self.hypothesis.falsifier
        if self.falsifier_proposal is not None:
            return self.falsifier_proposal.falsifier
        return self.hypothesis.falsifier

    @property
    def falsifier_state(self) -> str:
        """`absent` / `placeholder` / `stated` for the predicate in force."""
        return classify_falsifier(self.falsifier)

    @property
    def falsifier_source(self) -> Optional[str]:
        if self.hypothesis.falsifier_state == FALSIFIER_STATED:
            return FALSIFIER_SOURCE_STATED
        if self.falsifier_proposal is not None:
            return FALSIFIER_SOURCE_PROPOSED
        return None

    @property
    def may_spend_a_claim(self) -> bool:
        """Whether compute may be committed to this question.

        Read-only and derived. It is a convenience for a planner surface, NOT the gate:
        the gate is `ClaimAuthorization`, which refuses to be constructed, because a
        boolean somebody has to remember to consult is a boolean somebody forgets.
        """
        return self.is_open and self.falsifier_state == FALSIFIER_STATED

    @property
    def owner(self) -> str:
        """Who is working this question now. NOT `origin`, which never moves.

        A hypothesis the agents stated is theirs from birth; an operator hypothesis is
        the operator's until it is ADOPTED, and then it is the agents'. Origin records
        who had the idea and is frozen forever — relabelling it would be exactly the
        provenance laundering `_REFUSED_ENTRY_KEYS['origin']` refuses.
        """
        if self.adoption is not None:
            return OWNER_AGENTS
        return (
            OWNER_OPERATOR if self.hypothesis.origin == ORIGIN_OPERATOR else OWNER_AGENTS
        )

    @property
    def is_agent_owned(self) -> bool:
        return self.owner == OWNER_AGENTS


def _decode_payload(factory, record: Any, what: str, event: LedgerEvent):
    """Decode one ledger payload, classifying EVERY failure as ledger corruption.

    Without this, a ledger line whose payload is malformed surfaced as whatever its
    dataclass happened to raise: a bare `ValueError` (which is not a `HypothesisError`
    at all, so the driver that catches the controller plane — the stated reason
    `HypothesisError` extends `ControllerError` — does not catch it), or a
    `FalsifierMissing`, which is a *store* error and would send a reader looking for a
    bad operator file when the defect is in the ledger.
    """
    try:
        return factory(record)
    except (TypeError, ValueError, HypothesisError) as exc:
        raise HypothesisLedgerCorruption(
            f"seq {event.seq}: {event.kind} payload {what!r} is not a valid record "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def fold_ledger(events: Sequence[LedgerEvent]) -> dict:
    """Fold ledger events into `{hypothesis_id: TrackedHypothesis}`. Pure.

    Contradictions are REFUSED, not reconciled: an attempt or resolution naming an
    unopened question, a second OPENED for one id, a resolution of a resolved
    question, or a reopen of an open one all raise. A fold that repairs its input
    cannot tell a defect from a history.
    """
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("events must be a sequence of LedgerEvent")
    # Typed BEFORE sorting: `sorted(..., key=lambda e: e.seq)` touches every element
    # first, so a per-element type check inside the loop below could only ever fire for
    # something that already has a `.seq` — the documented TypeError was unreachable for
    # a dict, which is the exact thing a caller reaching for this function has.
    for index, event in enumerate(events):
        if not isinstance(event, LedgerEvent):
            raise TypeError(
                f"events[{index}]: expected LedgerEvent, got {type(event).__name__}"
            )
    state: dict = {}
    for event in sorted(events, key=lambda e: e.seq):
        key = event.hypothesis_id
        if event.kind == EVENT_OPENED:
            if key in state:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} was opened twice; one id is one question"
                )
            hypothesis = _decode_payload(
                Hypothesis.from_dict, event.payload.get("hypothesis") or {},
                "hypothesis", event,
            )
            if hypothesis.hypothesis_id != key:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: envelope names {key!r} but the payload holds "
                    f"{hypothesis.hypothesis_id!r}"
                )
            state[key] = TrackedHypothesis(
                hypothesis=hypothesis,
                status=STATUS_OPEN,
                opened_seq=event.seq,
                opened_at=event.at,
            )
            continue

        current = state.get(key)
        if current is None:
            raise HypothesisLedgerCorruption(
                f"seq {event.seq}: {event.kind} names {key!r}, which was never "
                "opened; an append-only ledger cannot repair a dangling reference"
            )
        if event.kind == EVENT_ATTEMPTED:
            attempt = _decode_payload(
                Attempt.from_dict, event.payload.get("attempt") or {}, "attempt", event,
            )
            if attempt.hypothesis_id != key:
                # The same binding the OPENED branch enforces. Without it the receipt
                # for what was tried can be filed against a question it was never about
                # — and "what was tried, and what it bore on" is the entire product of
                # this half of the mechanism.
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: envelope names {key!r} but the attempt holds "
                    f"{attempt.hypothesis_id!r}; a receipt filed under the wrong "
                    "question is worse than no receipt"
                )
            # An attempt is recorded whatever the status: a proposal dispositioned
            # after a question was resolved is still part of that question's history.
            state[key] = dataclasses.replace(
                current, attempts=current.attempts + (attempt,)
            )
        elif event.kind == EVENT_RESOLVED:
            if not current.is_open:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} is already {current.status}; resolving a "
                    "resolved question would overwrite the evidence that closed it"
                )
            resolution = _decode_payload(
                ResolutionEvidence.from_dict, event.payload.get("resolution") or {},
                "resolution", event,
            )
            # Re-derived, not trusted — the same discipline the CLAIM_AUTHORIZED branch
            # applies. A line claiming a question was resolved while nothing could have
            # refuted it is a line saying the closure gate did not hold.
            if current.falsifier_state != FALSIFIER_STATED:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} has falsifier state "
                    f"{current.falsifier_state!r}, so no evidence could have been "
                    "observed against its falsifier; a question is disposed of — by "
                    "compute or by closure — only once a predicate exists"
                )
            state[key] = dataclasses.replace(
                current,
                status=resolution.outcome,
                resolution=resolution,
                resolved_at=event.at,
            )
        elif event.kind == EVENT_FALSIFIER_PROPOSED:
            proposal = _decode_payload(
                FalsifierProposal.from_dict, event.payload.get("proposal") or {},
                "proposal", event,
            )
            if proposal.hypothesis_id != key:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: envelope names {key!r} but the proposal holds "
                    f"{proposal.hypothesis_id!r}; a predicate filed against the wrong "
                    "question is a question that can be closed by evidence about "
                    "another one"
                )
            if current.hypothesis.falsifier_state != FALSIFIER_ABSENT:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} was stated WITH a falsifier; a proposed "
                    "one would replace it, and a rewritten falsifier is how any "
                    "hypothesis becomes 'confirmed'"
                )
            if current.falsifier_proposal is not None:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} already has a proposed falsifier "
                    f"({current.falsifier_proposal.falsifier!r}); a second proposal is "
                    "a rewrite by another name"
                )
            state[key] = dataclasses.replace(current, falsifier_proposal=proposal)
        elif event.kind == EVENT_CLAIM_AUTHORIZED:
            authorization = _decode_payload(
                ClaimAuthorization.from_dict, event.payload.get("authorization") or {},
                "authorization", event,
            )
            if authorization.hypothesis_id != key:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: envelope names {key!r} but the authorization "
                    f"holds {authorization.hypothesis_id!r}; compute charged to the "
                    "wrong question is compute nobody can account for"
                )
            # The fold re-derives the gate rather than trusting the record: a ledger
            # line claiming a claim was authorized for a question with no falsifier is
            # a line that says the gate did not hold, and the fold's job is to refuse a
            # history that contradicts itself.
            if current.falsifier_state != FALSIFIER_STATED:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} has falsifier state "
                    f"{current.falsifier_state!r}, so no claim could have been "
                    "authorized on it; a falsifier is mandatory before compute is spent"
                )
            state[key] = dataclasses.replace(
                current,
                claim_authorizations=current.claim_authorizations + (authorization,),
            )
        elif event.kind == EVENT_ADOPTED:
            adoption = _decode_payload(
                Adoption.from_dict, event.payload.get("adoption") or {},
                "adoption", event,
            )
            if adoption.hypothesis_id != key:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: envelope names {key!r} but the adoption holds "
                    f"{adoption.hypothesis_id!r}; an adoption filed under another id "
                    "removes one operator entry and records the transfer of a different "
                    "one — the ORPHANED failure mode with a receipt on top"
                )
            if current.hypothesis.origin != ORIGIN_OPERATOR:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} has origin "
                    f"{current.hypothesis.origin!r}; adoption moves ownership OUT of "
                    "the operator store and there is nothing to move for a hypothesis "
                    "that was never in it"
                )
            if current.adoption is not None:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} was already adopted by "
                    f"{current.adoption.adopted_by!r}; ownership transfers once"
                )
            state[key] = dataclasses.replace(current, adoption=adoption)
        elif event.kind == EVENT_REOPENED:
            if current.is_open:
                raise HypothesisLedgerCorruption(
                    f"seq {event.seq}: {key} is already open; reopening it would "
                    "record a state change that did not happen"
                )
            state[key] = dataclasses.replace(
                current,
                status=STATUS_OPEN,
                resolution=None,
                resolved_at=None,
                reopen_count=current.reopen_count + 1,
                superseded_resolutions=(
                    current.superseded_resolutions
                    + ((current.resolution,) if current.resolution is not None else ())
                ),
            )
        else:  # pragma: no cover - LedgerEvent.__post_init__ closes the vocabulary
            raise HypothesisLedgerCorruption(f"seq {event.seq}: unknown kind {event.kind!r}")
    return state


# =============================================================================
# Do-not-repeat disposition (§8.4, §19.2, §19.3)
# =============================================================================

@dataclass(frozen=True)
class LedgerMatch:
    """One do-not-repeat / constraint-ledger entry that matches a hypothesis.

    This module CONSUMES matches; the §19.2 ledger that produces them belongs to the
    memory-update plane. `receipt` is `None` when the entry carries none, and that is
    decisive: §8.4 rejects a repeat only when the negative was recorded *"by an entry
    carrying a receipt"*, and §19.3 exists because a wrong suppression is invisible —
    nothing ever tests it again.
    """

    entry_id: str
    entry_class: str
    match_dimensions: Mapping[str, Any] = field(default_factory=dict)
    receipt: Optional[str] = None
    conflicted: bool = False
    reopen_predicate_satisfied: bool = False

    def __post_init__(self) -> None:
        _require_text(self.entry_id, "entry_id")
        if self.entry_class not in MATCH_CLASSES:
            raise ValueError(
                f"entry_class: {self.entry_class!r} not in {sorted(MATCH_CLASSES)}"
            )
        if not isinstance(self.match_dimensions, Mapping):
            raise TypeError("match_dimensions must be a mapping")
        if self.receipt is not None:
            _require_text(self.receipt, "receipt")
        for name in ("conflicted", "reopen_predicate_satisfied"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "entry_class": self.entry_class,
            "match_dimensions": dict(self.match_dimensions),
            "receipt": self.receipt,
            "conflicted": self.conflicted,
            "reopen_predicate_satisfied": self.reopen_predicate_satisfied,
        }


class DoNotRepeatLedger(Protocol):
    """The §19.2 ledger, as this module consumes it. Nothing here implements it."""

    def matches_for(self, regime: Mapping[str, Any], statement: str) -> Sequence:
        ...


def _dimension_values(value: Any) -> frozenset:
    """A dimension's value(s) as a comparable set. Scalars are one-element sets."""
    if value is None:
        return frozenset()
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
        return frozenset({text.strip().lower()}) if text.strip() else frozenset()
    if isinstance(value, Sequence):
        out: set = set()
        for item in value:
            out |= _dimension_values(item)
        return frozenset(out)
    return frozenset({str(value).strip().lower()})


def _contradicting_dimensions(regime: Mapping[str, Any], match: "LedgerMatch") -> tuple:
    """Dimensions where the QUESTION and the MATCH declare disjoint values.

    §19.2: *"'do not repeat' without regime identity is dangerous, because this project
    repeatedly observes sign changes across architecture, substrate, batch, context and
    quant"*. `check_do_not_repeat` holds both the question's regime and the entry's own
    `match_dimensions` and used to compare them not at all — `regime` was read only to
    ask whether it was empty. A receipted `MATCHED_NEGATIVE` recorded at
    llama_cpu/prefill/b1 therefore rejected a llama_gpu/decode/b128 question, and every
    one of the 56 contradicting cells of that 4x4x4 grid rejected. That is the §19.3
    failure in its worst form: a wrong suppression is invisible because nothing ever
    tests the family again, so the loop looks productive while being sterile.

    Only a SHARED dimension with no value in common is a contradiction. A dimension one
    side does not declare is an incomplete comparison, not a disagreement — that is the
    producer's matching rule to make (`do_not_repeat.CompiledLedger._compare` breaks the
    match on an undeclared dimension), and this consumer does not second-guess it. What
    it will not do is act on a match whose own record says it is about another regime.

    Both `match_dimensions` shapes are read: the entry's dimension map may sit under a
    `regime` key (what the compiled §19.2 ledger emits, values as lists) or be the
    mapping itself (a caller that passes the regime straight through). Neither is
    invented here; both already occur.
    """
    declared = match.match_dimensions
    if not isinstance(declared, Mapping):
        return ()
    nested = declared.get("regime")
    if isinstance(nested, Mapping):
        declared = nested
    differing: list = []
    for dimension, stated in sorted(regime.items()):
        if dimension not in declared:
            continue
        theirs = _dimension_values(declared[dimension])
        ours = _dimension_values(stated)
        if theirs and ours and not (theirs & ours):
            differing.append(
                f"{dimension}: the question states {sorted(ours)} and the entry was "
                f"recorded at {sorted(theirs)}"
            )
    return tuple(differing)


def check_do_not_repeat(
    *, regime: Mapping[str, Any], matches: Optional[Sequence]
) -> schemas.Check:
    """FAIL when a hypothesis repeats a RECEIPTED negative (§8.4, §19.2, §19.3).

    **This function cannot see who stated the hypothesis.** It takes the regime and
    the matches, and nothing else — the structural form of AK-D38's *"being the
    operator's idea is not new evidence"*. There is no origin parameter to consult and
    therefore no path along which one could be.

    Three outcomes, all real:

    * `matches is None` means the ledger was NOT consulted — COULD_NOT_CHECK. A
      hypothesis nobody checked against the ledger is not a hypothesis that clears it.
    * an empty sequence means it WAS consulted and matched nothing — PASS.
    * a rejecting match with a receipt — FAIL.

    A match whose OWN declared dimensions contradict the question's regime neither
    rejects nor clears (`_contradicting_dimensions`): this consumer will not act on an
    entry that says it is about another regime, whichever producer handed it over.

    Class dispositions follow §19.2's table exactly: `HARD_CONSTRAINT` and
    `MATCHED_NEGATIVE` reject; `CONDITIONAL_NEGATIVE`, `CONFOUNDED_RESULT`,
    `SUPERSEDED_FACT` and `LOW_VALUE` are advisory — they exclude cells, demand a
    repaired experiment, regenerate from current source, or deprioritize, and none of
    them closes the question. Two further §19.2/§19.3 rules bite here: a `conflicted`
    entry is never authoritative, and a `MATCHED_NEGATIVE` whose reopen predicate is
    newly satisfied does not reject.
    """
    if matches is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the do-not-repeat ledger was not consulted; not checking is not a clear "
            "result (§8.4 rejects a repeat of a receipted negative, and this call "
            "cannot tell whether one exists)",
        ))
    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        raise TypeError("matches must be None or a sequence of LedgerMatch")
    for index, match in enumerate(matches):
        if not isinstance(match, LedgerMatch):
            raise TypeError(f"matches[{index}] must be a LedgerMatch")
    if not isinstance(regime, Mapping):
        raise TypeError("regime must be a mapping")

    # A hypothesis with no regime has no match dimensions (§19.2 — "'do not repeat'
    # without regime identity is dangerous, because this project repeatedly observes
    # sign changes across architecture, substrate, batch, context and quant"). It is
    # collected as an incompleteness rather than returned immediately: a concrete
    # receipted match is a FACT and outranks an incomplete comparison, the same
    # precedence `state_machine.check_anchor_identity` applies.
    incomplete: list = []
    if not regime:
        incomplete.append(
            "the hypothesis declares no regime, so no match dimensions exist to "
            "compare against (§19.2)"
        )

    rejecting: list = []
    advisory: list = []
    unreceipted: list = []
    mismatched: list = []
    for match in matches:
        contradicted = _contradicting_dimensions(regime, match)
        if contradicted:
            # Handed a match that its own record says is about a DIFFERENT regime. It
            # cannot reject — that is the invisible false suppression §19.3 exists for —
            # and it cannot clear either: the producer and this consumer disagree about
            # what was matched, and two records that disagree never license an action
            # here. The most likely cause is a mis-keyed `matches_by_hypothesis`, which
            # is exactly the wiring mistake whose damage nothing ever surfaces.
            mismatched.append(
                f"{match.entry_id} ({match.entry_class}) was handed as a match but its "
                f"own dimensions CONTRADICT this question — "
                + "; ".join(contradicted)
                + " — so it is not about this regime (§19.2) and neither rejects nor "
                "clears it"
            )
            continue
        if match.conflicted:
            advisory.append(
                f"{match.entry_id} ({match.entry_class}) is CONFLICTED and is never "
                "authoritative (§19.2)"
            )
            continue
        if match.entry_class not in REJECTING_MATCH_CLASSES:
            advisory.append(
                f"{match.entry_id} ({match.entry_class}) is advisory under §19.2: it "
                "excludes cells, demands a repair, or deprioritizes — it does not "
                "close the question"
            )
            continue
        if (match.entry_class == MATCH_CLASS_MATCHED_NEGATIVE
                and match.reopen_predicate_satisfied):
            advisory.append(
                f"{match.entry_id} (MATCHED_NEGATIVE) has a newly satisfied reopen "
                "predicate, so §19.2 admits it"
            )
            continue
        if match.receipt is None:
            unreceipted.append(
                f"{match.entry_id} ({match.entry_class}) matches but carries NO "
                "receipt; §8.4 rejects a repeat only when the negative carries one, "
                "and §19.3 requires a source receipt bound to the production commit"
            )
            continue
        rejecting.append(
            f"{match.entry_id} ({match.entry_class}) already records this under "
            f"matching conditions, receipt {match.receipt}"
        )

    if rejecting:
        return schemas.Check(
            schemas.FAIL,
            tuple(rejecting + unreceipted + mismatched + incomplete + advisory),
        )
    if unreceipted or mismatched or incomplete:
        # A suppression without a receipt neither rejects nor clears: §19.3 makes the
        # receipt the price of closing a family, and a match we cannot verify is
        # exactly the "wrong suppression silently closes a research family" row of §12.
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            tuple(unreceipted + mismatched + incomplete + advisory),
        )
    return schemas.Check(schemas.PASS, tuple(advisory))


# =============================================================================
# The tracker
# =============================================================================

@dataclass(frozen=True)
class IntakeReport:
    """What one reconciliation of the operator store against the ledger did."""

    store_path: Optional[str]
    store_sha256: Optional[str]
    opened: tuple = ()
    already_tracked: tuple = ()
    #: In the store AND already resolved. Leaving the line in the file does NOT
    #: reopen it — reopening requires new evidence (§19.2's reopen predicate).
    resolved_but_still_in_store: tuple = ()
    #: Tracked, open, and NO LONGER in the store. Deleting the line does not close the
    #: question; that is the evaporation this module exists to prevent. ADOPTED
    #: hypotheses are NOT reported here: they are absent from the store BY DESIGN, and
    #: reporting a designed outcome as an anomaly is how a report stops being read.
    open_but_absent_from_store: tuple = ()
    #: Adopted in the ledger and STILL in the store — the DUPLICATE left by a crash
    #: between the adoption record and the removal. Detectable by id, repairable by
    #: `HypothesisTracker.reconcile_adoptions()`, and never a loss.
    adopted_but_still_in_store: tuple = ()
    #: Stated in the store with NO falsifier. Legal (2026-08-04), tracked, and unable
    #: to consume a resource claim until an agent proposes one. Surfaced so the loop
    #: can see the work it owes the operator rather than discovering it at the gate.
    awaiting_falsifier: tuple = ()


@dataclass(frozen=True)
class HypothesisTrace:
    """Everything the record says happened to one question. The operator's read-back.

    It wraps a `TrackedHypothesis` rather than copying its fields, so there is exactly
    one place the history lives and no chance of a trace that says something the fold
    does not. `answer` renders it as one sentence, because "what happened to the
    hypothesis I wrote?" is a question with a sentence-shaped answer.
    """

    tracked: TrackedHypothesis
    #: `True`/`False` when a store was supplied, `None` when the question was asked
    #: without one. `None` is not `False`: "we did not look" is not "it is gone".
    in_store: Optional[bool] = None

    @property
    def hypothesis_id(self) -> str:
        return self.tracked.hypothesis_id

    @property
    def adoption(self) -> Optional[Adoption]:
        return self.tracked.adoption

    @property
    def answer(self) -> str:
        """One sentence: where it went, what was tried, and how it ended."""
        t = self.tracked
        parts = [
            f"{t.hypothesis_id} was opened at {t.opened_at} "
            f"(origin {t.hypothesis.origin}, author {t.hypothesis.author}, "
            f"entry grade {t.evidence_grade})"
        ]
        if t.adoption is not None:
            parts.append(
                f"adopted by {t.adoption.adopted_by} at {t.adoption.at} "
                f"({t.adoption.reason}), so it left the operator store at "
                f"{t.adoption.store_path} and is now owned by the agents"
            )
        else:
            parts.append(f"still owned by {t.owner}")
        if t.falsifier_state == FALSIFIER_STATED:
            source = (
                "as stated" if t.falsifier_source == FALSIFIER_SOURCE_STATED
                else f"proposed by {t.falsifier_proposal.proposed_by}"
            )
            parts.append(f"falsifier ({source}): {t.falsifier}")
        else:
            parts.append(
                "no falsifier yet, so no resource claim may be spent on it "
                "(propose_falsifier)"
            )
        parts.append(
            f"{len(t.attempts)} attempt(s), "
            f"{len(t.claim_authorizations)} claim authorization(s)"
        )
        if t.resolution is not None:
            parts.append(
                f"{t.resolution.outcome} at {t.resolved_at} by "
                f"{t.resolution.resolved_by} on evidence "
                f"{list(t.resolution.evidence_refs)} "
                f"({t.resolution.evidence_grade}): {t.resolution.falsifier_observed}"
            )
        else:
            parts.append(f"status {t.status}")
        if self.in_store is not None:
            parts.append(
                "the entry is still in the operator store" if self.in_store
                else "the entry is no longer in the operator store"
            )
        return "; ".join(parts) + "."

    def to_dict(self) -> dict:
        t = self.tracked
        return {
            "hypothesis_id": t.hypothesis_id,
            "statement": t.hypothesis.statement,
            "origin": t.hypothesis.origin,
            "author": t.hypothesis.author,
            "owner": t.owner,
            "entry_evidence_grade": t.evidence_grade,
            "falsifier": t.falsifier,
            "falsifier_state": t.falsifier_state,
            "falsifier_source": t.falsifier_source,
            "falsifier_proposal": (
                t.falsifier_proposal.to_dict()
                if t.falsifier_proposal is not None else None
            ),
            "status": t.status,
            "opened_at": t.opened_at,
            "reopen_count": t.reopen_count,
            "adoption": t.adoption.to_dict() if t.adoption is not None else None,
            "attempts": [a.to_dict() for a in t.attempts],
            "claim_authorizations": [
                a.to_dict() for a in t.claim_authorizations
            ],
            "resolution": (
                t.resolution.to_dict() if t.resolution is not None else None
            ),
            "resolved_at": t.resolved_at,
            "in_store": self.in_store,
            "answer": self.answer,
        }


class HypothesisTracker:
    """Still-open tracking for every hypothesis, whatever its origin.

    Holds NO folded state: every query re-reads the ledger from disk under the journal
    write lock, for the same reason `state_machine` re-reads the control latch from
    disk. A cached fold is a fold that can be written back over the record.
    `__slots__` is load-bearing — there is no slot for a state cache, so an edit that
    adds one fails at runtime rather than quietly reintroducing the shape.
    """

    __slots__ = ("_journal", "_root", "_ledger", "_recorder", "_clock", "_campaign_id")

    def __init__(
        self,
        *,
        journal_: journal.Journal,
        root: str,
        campaign_id: Optional[str] = None,
        recorder: Optional[HypothesisRecorder] = None,
        clock=None,
    ) -> None:
        if not isinstance(journal_, journal.Journal):
            raise TypeError("journal_ must be a journal.Journal")
        try:
            journal_.shards()
        except journal.JournalError as exc:
            raise HypothesisError(
                f"{journal_.root}: the journal is not readable "
                f"({type(exc).__name__}: {exc}); call Journal.initialize() before "
                "constructing a tracker over it"
            ) from exc
        self._journal = journal_
        self._root = os.path.abspath(_require_text(root, "root"))
        self._campaign_id = campaign_id
        self._clock = clock if clock is not None else _iso_now
        os.makedirs(self._root, exist_ok=True)
        self._ledger = HypothesisLedger(os.path.join(self._root, LEDGER_FILENAME))
        self._ledger.initialize()
        self._recorder = recorder if recorder is not None else JournalOrderedRecorder(
            journal_, self._ledger
        )

    # ---- position ---------------------------------------------------------

    @property
    def root(self) -> str:
        return self._root

    @property
    def ledger(self) -> HypothesisLedger:
        return self._ledger

    @property
    def journal(self) -> journal.Journal:
        """The journal this tracker's events are ordered against. READ access.

        Exposed so the memory-update plane can fold BOTH halves of the record — the
        journal's proposals and evaluation events, and this tracker's hypothesis events
        — without being handed two objects that could be from two different campaigns.
        A `Journal` is itself append-only, so handing it over grants nothing this
        module was withholding.
        """
        return self._journal

    @property
    def campaign_id(self) -> Optional[str]:
        return self._campaign_id

    def read(self) -> LedgerRead:
        """The ledger as it is on disk right now, plus any torn tail."""
        return self._ledger.read()

    def state(self) -> dict:
        """`{hypothesis_id: TrackedHypothesis}`, folded fresh from disk."""
        return fold_ledger(self._ledger.read().events)

    def get(self, hypothesis_id: str) -> TrackedHypothesis:
        tracked = self.state().get(_require_text(hypothesis_id, "hypothesis_id"))
        if tracked is None:
            raise UnknownHypothesis(
                f"{hypothesis_id!r} was never opened in {self._ledger.path}"
            )
        return tracked

    def still_open(self) -> tuple:
        """Every unresolved question, in the order the questions were opened.

        The order is stable and is NOT a ranking: §8.4.0 — an operator hypothesis is
        not a queue-jumping mechanism, so nothing in this module ranks anything.
        """
        return tuple(sorted(
            (t for t in self.state().values() if t.is_open),
            key=lambda t: t.opened_seq,
        ))

    def resolved(self) -> tuple:
        return tuple(sorted(
            (t for t in self.state().values() if not t.is_open),
            key=lambda t: t.opened_seq,
        ))

    # ---- writes -----------------------------------------------------------

    def _record(
        self, kind: str, hypothesis_id: str, payload: Mapping[str, Any]
    ) -> LedgerEvent:
        """Build, record, and return one event. Recording is what makes it real.

        Called under the journal write lock by every writer here, so the read that
        assigns `seq` and the append that consumes it cannot be interleaved.
        """
        read = self._ledger.read()
        seq = (read.events[-1].seq + 1) if read.events else 1
        event = LedgerEvent(
            seq=seq,
            kind=kind,
            hypothesis_id=hypothesis_id,
            at=self._clock(),
            payload=dict(payload),
        )
        # Re-fold with the candidate appended BEFORE writing: the fold owns every
        # legality rule, so validating through it means a caller cannot reach a state
        # the fold would refuse to read back.
        fold_ledger(tuple(read.events) + (event,))
        recorded = self._recorder.record(event)
        if not isinstance(recorded, LedgerEvent):
            raise HypothesisError(
                f"recorder returned {type(recorded).__name__}, not a LedgerEvent; an "
                "event whose record cannot be identified did not happen"
            )
        # The ENVELOPE and the PAYLOAD, not just seq and kind. Comparing only the
        # envelope left the whole decision inside the record substitutable: a recorder
        # asked to record `refuted` with one evidence ref could record `confirmed` with
        # another, return an event whose seq and kind matched, and the tracker would
        # then report the question confirmed on evidence nobody supplied. This seam is
        # the documented swap point for a future journal-kind adapter, which is exactly
        # the position from which a substituted payload would be invisible.
        for name in ("seq", "kind", "hypothesis_id", "at"):
            if getattr(recorded, name) != getattr(event, name):
                raise HypothesisError(
                    "recorder returned an event that is not the one it was asked to "
                    f"record ({name}: {getattr(recorded, name)!r} vs "
                    f"{getattr(event, name)!r})"
                )
        # A recorder MAY add to the payload — binding in a journal event id is the
        # documented reason this seam exists (`state_machine.JournalTransitionRecorder`
        # fills its binding in the same way). It may not change or drop what it was
        # given.
        for name, value in event.payload.items():
            if name not in recorded.payload:
                raise HypothesisError(
                    "recorder returned an event that is not the one it was asked to "
                    f"record (payload key {name!r} was dropped)"
                )
            if recorded.payload[name] != value:
                raise HypothesisError(
                    "recorder returned an event that is not the one it was asked to "
                    f"record (payload key {name!r} was rewritten: "
                    f"{recorded.payload[name]!r} vs {value!r})"
                )
        # And what the recorder SAYS it recorded is checked against what the ledger now
        # holds, because every query on this class reads that file and nothing else. A
        # recorder is a seam, not an authority: "recording is what makes it real" has to
        # mean the RECORD, not the return value.
        after = self._ledger.read()
        landed = after.events[-1] if after.events else None
        if (landed is None or landed.seq != event.seq or landed.kind != event.kind
                or landed.hypothesis_id != event.hypothesis_id):
            raise HypothesisError(
                f"the ledger does not end with the event just recorded ({event.kind} "
                f"#{event.seq} for {event.hypothesis_id}); it ends with "
                + (f"{landed.kind} #{landed.seq} for {landed.hypothesis_id}"
                   if landed is not None else "nothing")
            )
        for name, value in event.payload.items():
            if landed.payload.get(name) != value:
                raise HypothesisError(
                    f"the ledger holds a different payload for {event.kind} "
                    f"#{event.seq}: key {name!r} reads {landed.payload.get(name)!r}, "
                    f"not {value!r}"
                )
        return recorded

    def open_hypothesis(self, hypothesis: Hypothesis) -> LedgerEvent:
        """Track a new question. Refuses a rewrite of a tracked one."""
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis")
        with self._journal.write_lock():
            state = self.state()
            existing = state.get(hypothesis.hypothesis_id)
            if existing is not None:
                if existing.hypothesis.fingerprint != hypothesis.fingerprint:
                    raise QuestionRewritten(
                        f"{hypothesis.hypothesis_id}: a DIFFERENT question is already "
                        "tracked under this id (statement, falsifier or regime "
                        "changed). A rewritten falsifier is how any hypothesis "
                        "becomes 'confirmed' — state the new question under a new id"
                    )
                raise HypothesisAlreadyTracked(
                    f"{hypothesis.hypothesis_id} is already tracked "
                    f"({existing.status}); opening it again would restart its history"
                )
            payload = {
                "hypothesis": hypothesis.to_dict(),
                "entry_evidence_grade": hypothesis.evidence_grade,
                "fingerprint": hypothesis.fingerprint,
            }
            if self._campaign_id is not None:
                payload["campaign_id"] = self._campaign_id
            return self._record(EVENT_OPENED, hypothesis.hypothesis_id, payload)

    def note_attempt(
        self,
        hypothesis_id: str,
        *,
        proposal_id: str,
        disposition: str,
        bears_on_falsifier: bool,
        note: str,
        refs: Sequence = (),
    ) -> LedgerEvent:
        """Record that a proposal was dispositioned while this question was open.

        THIS DOES NOT RESOLVE ANYTHING, by construction — there is no branch here that
        can set a status, whatever `bears_on_falsifier` says. §8.4.0: a hypothesis used
        to evaporate when its proposal was dispositioned, including when that proposal
        failed for an unrelated reason. Resolution is `resolve()`, and it costs
        evidence.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        attempt = Attempt(
            hypothesis_id=hypothesis_id,
            proposal_id=proposal_id,
            disposition=disposition,
            bears_on_falsifier=bears_on_falsifier,
            note=note,
            refs=tuple(refs),
            at=self._clock(),
        )
        with self._journal.write_lock():
            state = self.state()
            if hypothesis_id not in state:
                raise UnknownHypothesis(
                    f"{hypothesis_id!r} was never opened; an attempt on an untracked "
                    "question is the receipt-less 'already tried' this module removes"
                )
            return self._record(
                EVENT_ATTEMPTED, hypothesis_id, {"attempt": attempt.to_dict()}
            )

    def propose_falsifier(
        self,
        hypothesis_id: str,
        *,
        falsifier: str,
        proposed_by: str,
        rationale: str,
    ) -> LedgerEvent:
        """Write the predicate the operator did not have to (2026-08-04 amendment).

        This is the agent-side half of "optional on entry, mandatory before compute".
        The operator drops in a one-line idea; an agent must propose and record a
        falsifier before that hypothesis may consume a resource claim.

        It ADDS, it never replaces. A question stated WITH a falsifier is refused here
        (`FalsifierAlreadyStated`), and so is a second proposal — both are rewrites, and
        rewriting a falsifier after evidence exists is how any hypothesis becomes
        confirmed. The hypothesis record itself is untouched, so `fingerprint` does not
        move and the operator's next `intake()` still recognises their own file.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        proposal = FalsifierProposal(
            hypothesis_id=hypothesis_id,
            falsifier=falsifier,
            proposed_by=proposed_by,
            rationale=rationale,
            at=self._clock(),
        )
        with self._journal.write_lock():
            tracked = self.state().get(hypothesis_id)
            if tracked is None:
                raise UnknownHypothesis(
                    f"{hypothesis_id!r} was never opened; a falsifier for an untracked "
                    "question is a predicate with nothing to predicate"
                )
            if tracked.hypothesis.falsifier_state != FALSIFIER_ABSENT:
                raise FalsifierAlreadyStated(
                    f"{hypothesis_id} was stated WITH the falsifier "
                    f"{tracked.hypothesis.falsifier!r}; proposing another would replace "
                    "it. A new predicate is a new question and gets a new id"
                )
            if tracked.falsifier_proposal is not None:
                raise FalsifierAlreadyStated(
                    f"{hypothesis_id} already has the proposed falsifier "
                    f"{tracked.falsifier_proposal.falsifier!r} (by "
                    f"{tracked.falsifier_proposal.proposed_by}); a second proposal is a "
                    "rewrite by another name"
                )
            if proposal.falsifier.strip().lower() == \
                    tracked.hypothesis.statement.strip().lower():
                raise FalsifierMissing(
                    f"{hypothesis_id}: the proposed falsifier restates the hypothesis; "
                    "it must predict an OUTCOME that could fail to appear"
                )
            return self._record(
                EVENT_FALSIFIER_PROPOSED, hypothesis_id, {"proposal": proposal.to_dict()}
            )

    def _consult(self, tracked: TrackedHypothesis, ledger) -> schemas.Check:
        """Ask the §19.2 memory plane about one question. NEVER a silent pass.

        The seam is a duck-typed `DoNotRepeatLedger` rather than an import, because
        `do_not_repeat` imports THIS module: the memory plane conforms to the consumer,
        and reversing that would be a cycle. `None` is not accepted — "no ledger
        configured" is not a position, since an empty `CompiledLedger` is one pure call
        away and says the true thing ("nothing has been tried") instead of nothing.

        A ledger that REFUSES the question (`CompiledLedger.matches_for` raises when it
        cannot compare) is `matches=None`, which `check_do_not_repeat` already maps to
        COULD_NOT_CHECK. That is the operator's own case: a one-line idea usually names
        no `mechanism`, so it cannot be compared, and the refusal is the ledger being
        honest rather than the ledger being empty. It does not block the claim — see
        `RepeatsAReceiptedNegative` for why the ambiguous case is failed toward
        spending — and it is recorded on the token so it is visible afterwards.

        NOT OVERCLAIMED: the ledger is a SNAPSHOT the caller compiled, before the
        journal write lock this runs under was taken. The consultation is therefore
        atomic with respect to the record it writes, and NOT with respect to another
        process appending a negative in between. `do_not_repeat.compile_for_tracker()`
        is cheap and pure, so the discipline is to recompile per round rather than to
        hold one ledger across a campaign.
        """
        if ledger is None:
            raise LedgerNotConsulted(
                f"{tracked.hypothesis_id}: authorize_claim() requires a do-not-repeat "
                "ledger. There is no 'no memory configured' position — an empty "
                "do_not_repeat.CompiledLedger() is a pure, free answer meaning 'nothing "
                "has been tried', and passing None would mean 'do not ask', which is "
                "how this guard came to be wired to nothing in the first place"
            )
        matcher = getattr(ledger, "matches_for", None)
        if not callable(matcher):
            raise TypeError(
                f"ledger must implement DoNotRepeatLedger.matches_for(regime, "
                f"statement); {type(ledger).__name__} does not"
            )
        try:
            matches = matcher(tracked.hypothesis.regime, tracked.hypothesis.statement)
        except ControllerError as exc:
            # The ledger declined to answer. `check_do_not_repeat` is total over the
            # three outcomes and already has a value for this; inventing a fourth here
            # would be a second opinion about what an unanswered question means.
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"the do-not-repeat ledger could not compare this question: {exc}",
            ))
        return check_do_not_repeat(regime=tracked.hypothesis.regime, matches=matches)

    def authorize_claim(
        self, hypothesis_id: str, *, purpose: str, authorized_by: str, ledger
    ) -> ClaimAuthorization:
        """Mint the token that lets a resource claim be spent on this question.

        THIS is also where the §19.2 memory plane is consulted, and `ledger` is a
        REQUIRED argument for exactly that reason. `hypotheses.py:261` said the ledger
        belonged to a plane that did not exist, so `check_do_not_repeat()` sat correct
        and unreachable and the loop could not tell *"tried and failed"* from *"never
        tried"*. A default would have rebuilt that: every caller that forgot the
        argument would get the old behaviour and the guard would be wired to nothing
        again, silently. A receipted repeat raises `RepeatsAReceiptedNegative`;
        everything else mints a token that CARRIES the verdict, so the spend and what
        memory said about it are one record.

        THIS is where a falsifier stops being optional. The three states are checked
        HERE and not at the point the record was made, because the point of the
        amendment was to move the barrier to where the cost is:

        * `absent`  -> `FalsifierRequiredBeforeCompute`. Propose one, then come back.
        * `placeholder` -> `FalsifierRequiredBeforeCompute`, with a different message.
          It is structurally unreachable through a constructed `Hypothesis`, and it is
          checked anyway: the two states are distinct all the way down, so a reader of
          the refusal knows whether somebody wrote nothing or wrote 'tbd'.
        * `stated`  -> a `ClaimAuthorization`, and a durable ledger record of the spend
          carrying the falsifier it was authorized against.

        A RESOLVED question is refused too. Compute spent on a question the evidence
        already closed is compute spent on nothing, and `reopen()` — which costs new
        evidence — is the way back.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        _require_text(purpose, "purpose")
        _require_text(authorized_by, "authorized_by")
        with self._journal.write_lock():
            tracked = self.state().get(hypothesis_id)
            if tracked is None:
                raise UnknownHypothesis(
                    f"{hypothesis_id!r} was never opened; compute cannot be charged to "
                    "a question no record holds"
                )
            if not tracked.is_open:
                raise HypothesisNotOpen(
                    f"{hypothesis_id} is already {tracked.status}; reopen() it on new "
                    "evidence before spending another claim on it"
                )
            state = tracked.falsifier_state
            if state in FALSIFIER_STATES_REFUSING_COMPUTE:
                # Constructed here rather than raised inline so that the ONE place this
                # refusal is worded is `ClaimAuthorization.__post_init__` — the type
                # that cannot exist for these states. A second wording would be a second
                # gate, and the second gate is the one that gets edited.
                ClaimAuthorization(
                    hypothesis_id=hypothesis_id,
                    falsifier=tracked.falsifier,
                    falsifier_source=FALSIFIER_SOURCE_STATED,
                    origin=tracked.hypothesis.origin,
                    purpose=purpose,
                    authorized_by=authorized_by,
                    authorized_at=self._clock(),
                    ledger_seq=1,
                    do_not_repeat_outcome=None,
                    campaign_id=self._campaign_id,
                )
                raise HypothesisError(  # pragma: no cover - the line above always raises
                    f"{hypothesis_id}: falsifier state {state!r} did not refuse a claim"
                )
            # Ordered AFTER the falsifier gate on purpose: a question with no predicate
            # cannot be compared against anything, so asking memory about it first would
            # report an incomparability whose real cause is the missing falsifier.
            verdict = self._consult(tracked, ledger)
            authorization = ClaimAuthorization(
                hypothesis_id=hypothesis_id,
                falsifier=tracked.falsifier,
                falsifier_source=tracked.falsifier_source,
                origin=tracked.hypothesis.origin,
                purpose=purpose,
                authorized_by=authorized_by,
                authorized_at=self._clock(),
                # Provisional: `_record` assigns the real seq, and the token is rebuilt
                # against it below. An authorization whose `ledger_seq` named a record
                # that does not exist would be an authorization with nothing behind it.
                ledger_seq=1,
                # FAIL never reaches the token: the constructor raises
                # `RepeatsAReceiptedNegative` from here, before any record is written,
                # so a refused spend leaves no CLAIM_AUTHORIZED event behind it.
                do_not_repeat_outcome=verdict.outcome,
                do_not_repeat_reasons=tuple(verdict.reasons),
                campaign_id=self._campaign_id,
            )
            event = self._record(EVENT_CLAIM_AUTHORIZED, hypothesis_id, {
                "authorization": dataclasses.replace(
                    authorization, ledger_seq=self._next_seq()
                ).to_dict(),
            })
            return ClaimAuthorization.from_dict(event.payload["authorization"])

    def repair_torn_tail(self) -> int:
        """Truncate an unterminated trailing fragment. Returns the bytes discarded.

        `HypothesisLedger.append` refuses to write onto a torn tail — correctly, since
        an O_APPEND write would fuse the fragment to the new record and make the whole
        ledger permanently unparseable. But the module offered no way to clear one, so a
        process killed mid-append left EVERY question in the campaign frozen (`adopt`,
        `resolve`, `note_attempt` and `intake` all raise `HypothesisLedgerCorruption`)
        until a human edited an append-only durable record by hand. That is not a
        recovery procedure; it is the absence of one.

        Discarding is safe and is not a lost outcome: a fragment with no terminating
        newline was never fsynced as a complete line, so the event it describes never
        took effect — the same reading `read()` already applies and `journal.py` already
        repairs (`_repair_torn_tail_locked`). Taken under the journal write lock so no
        append can interleave, and a NO-OP when there is nothing torn, so calling it is
        never a way to drop a record that did land.
        """
        with self._journal.write_lock():
            read = self._ledger.read()
            torn = read.discarded_tail_bytes
            if not torn:
                return 0
            path = self._ledger.path
            keep = os.path.getsize(path) - torn
            fd = os.open(path, os.O_WRONLY)
            try:
                os.ftruncate(fd, keep)
                os.fsync(fd)
            finally:
                os.close(fd)
            after = self._ledger.read()
            if after.discarded_tail_bytes:  # pragma: no cover - one truncate suffices
                raise HypothesisLedgerCorruption(
                    f"{path}: {after.discarded_tail_bytes} byte(s) still unterminated "
                    "after the repair; nothing further was attempted"
                )
            if after.events != read.events:  # pragma: no cover - truncation is suffix-only
                raise HypothesisLedgerCorruption(
                    f"{path}: the repair changed a DURABLE record; a torn tail is the "
                    "only thing this may remove"
                )
            return torn

    def _next_seq(self) -> int:
        """The seq `_record` will assign next. Called under the journal write lock."""
        read = self._ledger.read()
        return (read.events[-1].seq + 1) if read.events else 1

    def resolve(self, hypothesis_id: str, evidence: ResolutionEvidence) -> LedgerEvent:
        """Close a question WITH the evidence that closed it.

        `ResolutionEvidence` has already refused an empty ref list, an absent
        observation against the falsifier, and `bears_on_falsifier != True`. What is
        added here is that the question must be OPEN: overwriting a resolution would
        destroy the receipt that closed it.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        if not isinstance(evidence, ResolutionEvidence):
            raise TypeError("evidence must be a ResolutionEvidence")
        with self._journal.write_lock():
            state = self.state()
            tracked = state.get(hypothesis_id)
            if tracked is None:
                raise UnknownHypothesis(f"{hypothesis_id!r} was never opened")
            if not tracked.is_open:
                raise HypothesisNotOpen(
                    f"{hypothesis_id} is already {tracked.status}; reopen() it on new "
                    "evidence rather than overwriting the evidence that closed it"
                )
            if tracked.falsifier_state != FALSIFIER_STATED:
                # The amendment gated COMPUTE on a falsifier and left CLOSURE ungated,
                # which is the cheaper and more damaging move: an agent could mark the
                # operator's one-line idea `confirmed` having written no predicate,
                # spent no claim and run nothing, and the operator's own entry is gone
                # from the store by then. Every field of the evidence is a claim ABOUT
                # the falsifier — `falsifier_observed` says what was seen against it and
                # `bears_on_falsifier` must be exactly True — so a resolution of a
                # question in state `absent` describes an observation against a
                # predicate that does not exist.
                raise ResolutionEvidenceMissing(
                    f"{hypothesis_id}: falsifier state is "
                    f"{tracked.falsifier_state!r}, so there is nothing for this "
                    "evidence to have been observed AGAINST. A falsifier is optional on "
                    "operator entry and mandatory before the question is disposed of, "
                    "by compute or by closure — propose_falsifier() first. Recording "
                    "what was tried without closing it is note_attempt()"
                )
            stamped = evidence if evidence.at is not None else dataclasses.replace(
                evidence, at=self._clock()
            )
            return self._record(
                EVENT_RESOLVED, hypothesis_id, {"resolution": stamped.to_dict()}
            )

    def reopen(
        self,
        hypothesis_id: str,
        *,
        reason: str,
        new_evidence_refs: Sequence,
        reopened_by: str,
    ) -> LedgerEvent:
        """Reopen a resolved question, on NEW EVIDENCE.

        `new_evidence_refs` is mandatory and non-empty for the same reason
        `resolve()`'s is: §19.2's reopen predicate is a fact about the world, and
        "someone asked again" is not one. An `inconclusive` resolution is reopenable
        exactly like the other two — otherwise "inconclusive" would silently close a
        question forever, which is the failure mode in a new costume.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        _require_text(reason, "reason")
        _require_text(reopened_by, "reopened_by")
        refs = _require_refs(
            new_evidence_refs, "new_evidence_refs", error=ResolutionEvidenceMissing
        )
        with self._journal.write_lock():
            state = self.state()
            tracked = state.get(hypothesis_id)
            if tracked is None:
                raise UnknownHypothesis(f"{hypothesis_id!r} was never opened")
            if tracked.is_open:
                raise HypothesisNotOpen(
                    f"{hypothesis_id} is already open; reopening it would record a "
                    "state change that did not happen"
                )
            return self._record(EVENT_REOPENED, hypothesis_id, {
                "reason": reason,
                "new_evidence_refs": list(refs),
                "reopened_by": reopened_by,
                "superseded_outcome": tracked.status,
            })

    # ---- intake -----------------------------------------------------------

    def intake(self, store: Optional[OperatorHypothesisStore]) -> IntakeReport:
        """Reconcile the operator store into the ledger. Idempotent.

        `store=None` is how a campaign says it has NO operator channel. A configured
        store that cannot be read RAISES (`load()`), because a mount failure must not
        be able to say "the operator has no hypotheses".

        Four dispositions, and the last two are the interesting ones:

        * new id                       -> opened;
        * known id, same fingerprint   -> already tracked, no event;
        * known id, resolved           -> stays resolved. Leaving the line in the file
          does not reopen it; `reopen()` and new evidence do.
        * tracked and open, no longer in the file -> STAYS OPEN and is reported. The
          operator deleting a line means "stop offering this at intake", never "this
          question is answered".

        A known id whose statement, falsifier or regime changed raises
        `QuestionRewritten` — see `open_hypothesis()`.
        """
        if store is None:
            state = self.state()
            return IntakeReport(
                store_path=None,
                store_sha256=None,
                open_but_absent_from_store=tuple(sorted(
                    t.hypothesis_id for t in state.values()
                    if t.is_open and t.hypothesis.origin == ORIGIN_OPERATOR
                    and t.adoption is None
                )),
                awaiting_falsifier=tuple(sorted(
                    t.hypothesis_id for t in state.values()
                    if t.is_open and t.falsifier_state != FALSIFIER_STATED
                )),
            )
        if not isinstance(store, OperatorHypothesisStore):
            raise TypeError("store must be an OperatorHypothesisStore or None")

        stated, store_sha = store.load_with_digest()
        opened: list = []
        already: list = []
        resolved_in_store: list = []
        duplicated: list = []
        with self._journal.write_lock():
            state = self.state()
            # TWO PASSES, and the order is the point: classify every stated entry
            # BEFORE opening any of them. Opening as we went meant a rewritten entry
            # anywhere in the file raised after earlier entries had already been
            # committed, and the IntakeReport — the only account of what this intake
            # did — was destroyed by the exception that travelled past it. An intake
            # that half-applied and reported nothing is the shape of defect this
            # module exists to remove.
            to_open: list = []
            for hypothesis in stated:
                tracked = state.get(hypothesis.hypothesis_id)
                if tracked is None:
                    to_open.append(hypothesis)
                    continue
                if tracked.hypothesis.fingerprint != hypothesis.fingerprint:
                    raise QuestionRewritten(
                        f"{hypothesis.hypothesis_id}: the store now states a DIFFERENT "
                        "question under a tracked id (statement, falsifier or regime "
                        "changed). Rewriting a falsifier after evidence exists is how "
                        "any hypothesis becomes 'confirmed' — give the new question a "
                        "new id"
                    )
                if tracked.adoption is not None:
                    # ADOPTED and still in the file: the duplicate a crash between the
                    # adoption record and the removal leaves behind. Reported, never
                    # silently re-adopted and never re-opened.
                    duplicated.append(hypothesis.hypothesis_id)
                if tracked.is_open:
                    already.append(hypothesis.hypothesis_id)
                else:
                    resolved_in_store.append(hypothesis.hypothesis_id)
            for hypothesis in to_open:
                self.open_hypothesis(hypothesis)
                opened.append(hypothesis.hypothesis_id)
            stated_ids = {h.hypothesis_id for h in stated}
            after = self.state()
            absent = sorted(
                t.hypothesis_id for t in after.values()
                if t.is_open and t.hypothesis.origin == ORIGIN_OPERATOR
                and t.hypothesis_id not in stated_ids
                # An ADOPTED hypothesis is absent from the store BY DESIGN — that is
                # what adoption did. Reporting it as an anomaly would train a reader to
                # ignore the field that names the real one.
                and t.adoption is None
            )
            awaiting = sorted(
                t.hypothesis_id for t in after.values()
                if t.is_open and t.falsifier_state != FALSIFIER_STATED
            )
        return IntakeReport(
            store_path=store.path,
            store_sha256=store_sha,
            opened=tuple(opened),
            already_tracked=tuple(already),
            resolved_but_still_in_store=tuple(resolved_in_store),
            open_but_absent_from_store=tuple(absent),
            adopted_but_still_in_store=tuple(sorted(duplicated)),
            awaiting_falsifier=tuple(awaiting),
        )

    # ---- adoption: ownership moves from the operator to the agents ---------

    def adopt(
        self,
        hypothesis_id: str,
        store: OperatorHypothesisStore,
        *,
        adopted_by: str,
        reason: str,
        lock_timeout_s: float = 30.0,
    ) -> Adoption:
        """Pick up an operator hypothesis: record the transfer, then remove the entry.

        *"if the agents choose to pick up one of my hypotheses, it should be removed
        from OperatorHypothesisStore since it becomes owned by the agents."*

        THE ORDER IS THE DESIGN. This is a move between two durable stores, it has
        exactly three failure modes, and this one is failed toward the recoverable one:

        1. the adoption lock is taken (`flock` + PID/start-time liveness), so a second
           adopter cannot interleave;
        2. an UNTRACKED hypothesis is opened into the ledger first, so its content is
           durable before the file is touched at all — this is what makes **LOST**
           structurally impossible rather than unlikely;
        3. the `HYPOTHESIS_ADOPTED` record is appended and fsynced, carrying the
           hypothesis content AND the operator's own entry bytes INLINE — not a
           reference into the file this call is about to rewrite, which is what makes
           **ORPHANED** impossible;
        4. and only then is the entry spliced out and the store atomically replaced.

        A crash between 3 and 4 leaves the entry recorded AND present: a **DUPLICATE**.
        It is detectable by id (`adoption_duplicates()`), repairable by id
        (`reconcile_adoptions()`), and it is the direction to fail in — the operator's
        idea is in two places rather than none.

        Re-calling `adopt()` with the same `adopted_by` after such a crash COMPLETES the
        removal without writing a second record; a DIFFERENT adopter is refused, because
        adoption is a transfer of ownership and there is one owner.
        """
        _require_text(hypothesis_id, "hypothesis_id")
        _require_text(adopted_by, "adopted_by")
        _require_text(reason, "reason")
        if not isinstance(store, OperatorHypothesisStore):
            raise TypeError("store must be an OperatorHypothesisStore")

        with store.adoption_lock(timeout_s=lock_timeout_s):
            stated, store_sha = store.load_with_digest()
            by_id = {h.hypothesis_id: h for h in stated}
            text, spans = store.entry_spans()
            span = next((s for s in spans if s.hypothesis_id == hypothesis_id), None)

            with self._journal.write_lock():
                tracked = self.state().get(hypothesis_id)
                existing = tracked.adoption if tracked is not None else None
                if existing is not None:
                    if existing.adopted_by != adopted_by:
                        raise HypothesisAlreadyAdopted(
                            f"{hypothesis_id} was adopted by {existing.adopted_by!r} at "
                            f"{existing.at}; ownership transfers ONCE. It is "
                            + ("still in the operator store — a crash left a DUPLICATE; "
                               "reconcile_adoptions() completes the removal"
                               if span is not None else
                               "already out of the operator store")
                        )
                    if span is None:
                        return existing  # already complete; nothing to remove
                    # Same adopter, entry still present: finish the interrupted removal
                    # WITHOUT a second record.
                    store.remove_entry(hypothesis_id, expected_sha256=store_sha)
                    return existing

                if span is None:
                    raise HypothesisNotInStore(
                        f"{store.path}: {hypothesis_id!r} is not in the operator store, "
                        "so there is no ownership to transfer. Present ids: "
                        f"{[s.hypothesis_id for s in spans]}"
                    )
                hypothesis = by_id[hypothesis_id]
                if hypothesis.origin != ORIGIN_OPERATOR:  # pragma: no cover
                    raise HypothesisError(
                        f"{hypothesis_id}: the store produced a non-operator origin "
                        f"{hypothesis.origin!r}"
                    )
                if tracked is None:
                    # Journal-first, before a single byte of the operator's file moves.
                    self.open_hypothesis(hypothesis)
                elif tracked.hypothesis.fingerprint != hypothesis.fingerprint:
                    raise QuestionRewritten(
                        f"{hypothesis_id}: the store now states a DIFFERENT question "
                        "under a tracked id; adopting it would transfer ownership of "
                        "one question and remove the text of another"
                    )
                adoption = Adoption(
                    hypothesis_id=hypothesis_id,
                    adopted_by=adopted_by,
                    reason=reason,
                    store_path=store.path,
                    store_sha256_before=store_sha,
                    entry_index=span.index,
                    entry_text=span.text_of(text),
                    hypothesis=hypothesis.to_dict(),
                    at=self._clock(),
                )
                self._record(EVENT_ADOPTED, hypothesis_id, {
                    "adoption": adoption.to_dict(),
                    "owner": OWNER_AGENTS,
                    "entry_evidence_grade": hypothesis.evidence_grade,
                })
                # DURABLE from here. Everything below can fail, and the worst it
                # produces is a duplicate.
                store.remove_entry(hypothesis_id, expected_sha256=store_sha)
                return adoption

    def adoption_duplicates(self, store: OperatorHypothesisStore) -> tuple:
        """Ids the ledger records as ADOPTED that are STILL in the operator store.

        The detector for the one failure mode this move can produce. An empty tuple is
        a real statement — the two stores agree — which is why a store that cannot be
        read raises here rather than returning `()`.
        """
        if not isinstance(store, OperatorHypothesisStore):
            raise TypeError("store must be an OperatorHypothesisStore")
        stated = {h.hypothesis_id for h in store.load()}
        return tuple(sorted(
            t.hypothesis_id for t in self.state().values()
            if t.adoption is not None and t.hypothesis_id in stated
        ))

    def reconcile_adoptions(self, store: OperatorHypothesisStore) -> tuple:
        """Finish every interrupted adoption. Idempotent, and writes NO new record.

        The adoption record is already durable for each of these — that is what makes
        them detectable — so the repair is the removal alone. Re-recording would mint a
        second transfer of one ownership, which `fold_ledger` refuses anyway.
        """
        if not isinstance(store, OperatorHypothesisStore):
            raise TypeError("store must be an OperatorHypothesisStore")
        repaired: list = []
        with store.adoption_lock():
            while True:
                duplicates = self.adoption_duplicates(store)
                if not duplicates:
                    break
                # One at a time and re-read between: every removal changes the byte
                # offsets of everything after it, so a batch computed once would splice
                # against spans that no longer describe the file.
                store.remove_entry(duplicates[0])
                repaired.append(duplicates[0])
        return tuple(repaired)

    # ---- operator traceability --------------------------------------------

    def trace(
        self, hypothesis_id: str, store: Optional[OperatorHypothesisStore] = None
    ) -> "HypothesisTrace":
        """*"What happened to the hypothesis I wrote?"* — answered from the ledger.

        This is the requirement that makes REMOVAL acceptable at all. Taking an entry
        out of the operator's own file is deletion unless the operator can still find
        out where it went, so the trace resolves an id to its adoption, its attempts,
        every claim it authorized, and its resolution — none of which live in the file.

        `store` is optional and only fills in `in_store`: the answer must not depend on
        the file, because the file is the thing the entry was removed from.
        """
        tracked = self.get(hypothesis_id)
        in_store: Optional[bool] = None
        if store is not None:
            if not isinstance(store, OperatorHypothesisStore):
                raise TypeError("store must be an OperatorHypothesisStore or None")
            in_store = any(h.hypothesis_id == hypothesis_id for h in store.load())
        return HypothesisTrace(tracked=tracked, in_store=in_store)

    # ---- the planner surface ----------------------------------------------

    def planner_round_block(
        self,
        *,
        round_id: str,
        matches_by_hypothesis: Optional[Mapping[str, Optional[Sequence]]] = None,
        include_resolved: bool = True,
    ) -> dict:
        """The open set, re-surfaced for one planning round (§8.4.0, §6.1).

        Structured facts, not a prompt: the planner adapter owns prompt assembly, and a
        renderer that emitted instruction-shaped prose would be putting operator
        content in an instruction position, which §6.1 forbids for imported content and
        §8.4.0 forbids for this content specifically — *a proposal source, never an
        authority*.

        Every entry carries its falsifier (structurally: `Hypothesis` cannot exist
        without one) and its `entry_evidence_grade`, which is `design_prior` for every
        origin. The resolution's own grade is rendered under a DIFFERENT name,
        `resolution_evidence_grade`, so no reader can read one as the other.

        `matches_by_hypothesis` supplies §19.2 ledger matches per id. A missing id
        means the ledger was not consulted for it, and the entry carries
        COULD_NOT_CHECK — never a silent pass.
        """
        _require_text(round_id, "round_id")
        if matches_by_hypothesis is not None and not isinstance(
            matches_by_hypothesis, Mapping
        ):
            raise TypeError("matches_by_hypothesis must be a mapping or None")

        def _render(tracked: TrackedHypothesis) -> dict:
            hypothesis = tracked.hypothesis
            matches = None
            if matches_by_hypothesis is not None:
                matches = matches_by_hypothesis.get(hypothesis.hypothesis_id)
            verdict = check_do_not_repeat(regime=hypothesis.regime, matches=matches)
            entry = {
                "hypothesis_id": hypothesis.hypothesis_id,
                "statement": hypothesis.statement,
                # The predicate IN FORCE — the operator's own, or the one an agent
                # proposed for it. `falsifier_state` says which of the three states it
                # is in, and `falsifier_source` names who wrote it, so a reader can
                # never mistake "nobody has written one" for "the operator wrote this".
                "falsifier": tracked.falsifier,
                "falsifier_state": tracked.falsifier_state,
                "falsifier_source": tracked.falsifier_source,
                "stated_falsifier": hypothesis.falsifier,
                "proposed_falsifier": (
                    tracked.falsifier_proposal.to_dict()
                    if tracked.falsifier_proposal is not None else None
                ),
                "may_spend_a_claim": tracked.may_spend_a_claim,
                "origin": hypothesis.origin,
                "author": hypothesis.author,
                "owner": tracked.owner,
                "adopted": tracked.adoption is not None,
                "entry_evidence_grade": hypothesis.evidence_grade,
                "regime": dict(hypothesis.regime),
                "provenance": dict(hypothesis.source),
                "status": tracked.status,
                "opened_at": tracked.opened_at,
                "reopen_count": tracked.reopen_count,
                "attempt_count": len(tracked.attempts),
                "attempts": [a.to_dict() for a in tracked.attempts],
                "claim_authorization_count": len(tracked.claim_authorizations),
                "do_not_repeat": {
                    "outcome": verdict.outcome,
                    "reasons": list(verdict.reasons),
                },
            }
            if tracked.resolution is not None:
                entry["resolution"] = {
                    "outcome": tracked.resolution.outcome,
                    "resolution_evidence_grade": tracked.resolution.evidence_grade,
                    "evidence_refs": list(tracked.resolution.evidence_refs),
                    "falsifier_observed": tracked.resolution.falsifier_observed,
                    "resolved_by": tracked.resolution.resolved_by,
                    "resolved_at": tracked.resolved_at,
                }
            return entry

        open_entries = [_render(t) for t in self.still_open()]
        block = {
            "schema": ROUND_BLOCK_SCHEMA,
            "round_id": round_id,
            "campaign_id": self._campaign_id,
            "compiled_at": self._clock(),
            # §8.4.0, AK-D38, and the reason this whole module is safe. Rendered as a
            # field so a consumer that drops it is missing a key rather than silently
            # promoting a prior.
            "authority": "proposal_source_not_authority",
            "entry_evidence_grade": ENTRY_GRADE,
            "open_count": len(open_entries),
            "still_open": open_entries,
            # The work the loop owes the operator, named rather than discovered at the
            # gate: these are open questions no claim may be spent on until an agent
            # writes the predicate the operator did not have to.
            "awaiting_falsifier": [
                e["hypothesis_id"] for e in open_entries
                if e["falsifier_state"] != FALSIFIER_STATED
            ],
        }
        if include_resolved:
            block["resolved"] = [_render(t) for t in self.resolved()]
        flagged = schemas.find_authority_flavoured_keys(block)
        if flagged:  # pragma: no cover - no construction here produces such a key
            raise HypothesisError(
                f"round block carries authority-flavoured keys {flagged}; a planner "
                "block is a set of priors, never a grant (§1.3, invariant 5)"
            )
        # Serializability is proven, not hoped for: the block is handed to a context
        # compiler that will hash it.
        schemas.canonical_json(block)
        return block


# =============================================================================
# The structural audit (§8.4.0's safety property, checked from the objects)
# =============================================================================

#: The entry the audit hands the loader. Its falsifier is a real one-line predicted
#: outcome so that a refusal can only be about the key under test.
_AUDIT_PROBE_ENTRY: Mapping[str, Any] = {
    "hypothesis_id": "akh-audit-probe",
    "statement": "the operator store refuses to state its own evidence grade",
    "falsifier": "a probe entry carrying a grade key loads without a refusal",
}


def _store_refuses_key(key: str) -> Optional[str]:
    """`None` if the loader REFUSES an entry carrying `key`, else why that matters.

    Behavioural, not declarative. Reading `_REFUSED_ENTRY_KEYS` proves a dict has a
    string in it; it does not prove `_load_entry` consults the dict, so the whole
    refusal could be deleted from the loader with the table left in place and the audit
    would still have said PASS. This calls the loader.
    """
    probe = dict(_AUDIT_PROBE_ENTRY)
    probe[key] = GRADE_PROTOCOL_BOUND
    store = OperatorHypothesisStore("<audit-probe>")  # no I/O: _load_entry reads none
    try:
        store._load_entry(probe, 0, "<audit-probe>")
    except HypothesisStoreError:
        return None
    except Exception as exc:  # noqa: BLE001 - any refusal is a refusal, but say so
        return (
            f"the operator store rejects a stated {key!r} with {type(exc).__name__} "
            "rather than a HypothesisStoreError; the refusal is not the typed one a "
            "caller can catch"
        )
    return (
        f"the operator store LOADS an entry stating {key!r}; the one operator-editable "
        "input in this system can state its own evidence grade, which is exactly the "
        "laundering §8.4.0/AK-D38/§19.0 rule 4 forbid (the refusal table may still list "
        "the key — a table is not an enforcement)"
    )


def _store_admits_a_clean_entry() -> Optional[str]:
    """`None` if the probe entry WITHOUT a grade key loads.

    The control that stops `_store_refuses_key` from passing vacuously: a loader that
    refused everything — including every real operator hypothesis — would otherwise
    look like the strongest possible enforcement.
    """
    store = OperatorHypothesisStore("<audit-probe>")
    try:
        store._load_entry(dict(_AUDIT_PROBE_ENTRY), 0, "<audit-probe>")
    except Exception as exc:  # noqa: BLE001
        return (
            f"the operator store refuses a well-formed entry ({type(exc).__name__}: "
            f"{exc}); the grade-refusal probes below cannot be told apart from a loader "
            "that refuses everything"
        )
    return None


def audit_no_origin_grade_promotion() -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK on "can origin raise a hypothesis's grade?".

    Proved from the objects rather than asserted in prose, in the shape of
    `state_machine.audit_no_cached_control_state()`:

    1. `entry_grade` returns `design_prior` for EVERY declared origin — enumerated,
       so a new origin cannot arrive with a higher grade;
    2. `Hypothesis` declares NO dataclass field whose name mentions a grade — the only
       grade on it is a read-only property, so there is nothing to assign;
    3. `Hypothesis` is frozen, so even the property's backing objects cannot be
       reassigned after construction; and
    4. the operator store REFUSES an `evidence_grade` key, so the one editable input in
       the system cannot state its own.

    COULD_NOT_CHECK is returned when introspection itself fails, never a soft PASS.
    """
    try:
        grades = {origin: entry_grade(origin) for origin in sorted(ORIGINS)}
        fields = {f.name for f in dataclasses.fields(Hypothesis)}
        frozen = Hypothesis.__dataclass_params__.frozen
    except Exception as exc:  # pragma: no cover - introspection failure is not a pass
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"could not introspect the hypothesis contract ({type(exc).__name__}: "
            f"{exc}); inability to evaluate is a third outcome",
        ))

    reasons: list = []
    promoted = {o: g for o, g in grades.items() if g != ENTRY_GRADE}
    if promoted:
        reasons.append(
            f"entry_grade() returns a non-{ENTRY_GRADE} grade for {promoted}; §8.4.0 — "
            "a hypothesis can never be promoted by its origin"
        )
    grade_fields = sorted(f for f in fields if "grade" in f.lower())
    if grade_fields:
        reasons.append(
            f"Hypothesis declares settable grade field(s) {grade_fields}; grade must "
            "stay a derived property with nothing to assign to"
        )
    if not frozen:
        reasons.append(
            "Hypothesis is not a frozen dataclass; a mutable hypothesis can be "
            "re-graded after it has been read"
        )
    if "evidence_grade" not in _REFUSED_ENTRY_KEYS:
        reasons.append(
            "the operator store no longer refuses an 'evidence_grade' key; the one "
            "operator-editable input in this system could state its own grade"
        )
    # …and the refusal is EXERCISED, not read off a table. Point 4 used to be satisfied
    # by the presence of a string in `_REFUSED_ENTRY_KEYS`, so deleting the loop in
    # `_load_entry` that consults it left the audit reporting PASS while the store
    # happily loaded `evidence_grade: protocol_bound`.
    vacuous = _store_admits_a_clean_entry()
    if vacuous is not None:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons) + (vacuous,))
    for key in ("evidence_grade", "grade"):
        why = _store_refuses_key(key)
        if why is not None:
            reasons.append(why)
    grade_attr = getattr(Hypothesis, "evidence_grade", None)
    if not isinstance(grade_attr, property):
        reasons.append(
            "Hypothesis.evidence_grade is not a read-only property; a plain attribute "
            "is a slot something can assign a promoted grade into"
        )
    elif grade_attr.fset is not None or grade_attr.fdel is not None:
        reasons.append(
            "Hypothesis.evidence_grade has a setter or deleter; the grade must have "
            "no write path at all"
        )

    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


# =============================================================================
# The claim-gate audit — "no route from state (i) or (ii) to a claim"
# =============================================================================

def _authorization_probe(falsifier: Optional[str]) -> Optional[Exception]:
    """The exception constructing a `ClaimAuthorization` raises, or `None`."""
    try:
        ClaimAuthorization(
            hypothesis_id="akh-audit-probe",
            falsifier=falsifier,
            falsifier_source=FALSIFIER_SOURCE_STATED,
            origin=ORIGIN_OPERATOR,
            purpose="audit probe",
            authorized_by="audit",
            authorized_at="2026-08-04T00:00:00.000000Z",
            ledger_seq=1,
            # The probe is about the FALSIFIER axis, so the memory axis is set to the
            # value that lets a token exist: a PASS the audit did not have to earn.
            # Leaving it unset would make every probe fail on the wrong field and the
            # audit would report a claim gate that is shut for a reason nobody meant.
            do_not_repeat_outcome=schemas.PASS,
        )
    except Exception as exc:  # noqa: BLE001 - the audit reports the type it got
        return exc
    return None


def _claim_call_sites(source: str) -> tuple:
    """Every call to a claim-acquiring name in `source`, with its enclosing function.

    Behavioural in the only sense available to a static property: it parses THIS
    module and reports where a claim could be acquired from, rather than asserting in
    prose that only one place does.
    """
    tree = ast.parse(source)
    sites: list = []

    class Walker(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list = []

        def _enter(self, node) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_FunctionDef = _enter
        visit_AsyncFunctionDef = _enter
        visit_ClassDef = _enter

        def visit_Call(self, node) -> None:
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name in _CLAIM_ACQUISITION_NAMES:
                sites.append((name, tuple(self.scope), node.lineno))
            self.generic_visit(node)

    Walker().visit(tree)
    return tuple(sites)


def audit_falsifier_required_before_claim(source: Optional[str] = None) -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK on *"can compute be spent on a hypothesis with no
    usable falsifier?"*

    Four properties, and the third is the control that stops the other two from passing
    vacuously:

    1. a `ClaimAuthorization` for falsifier state `absent` REFUSES to be constructed,
       with the typed `FalsifierRequiredBeforeCompute`;
    2. one for state `placeholder` refuses too — separately, because the two states are
       distinct all the way down and a single merged refusal would prove only that
       something was rejected;
    3. **one for a REAL falsifier CONSTRUCTS.** Without this, a token type that refused
       everything — including every legitimate experiment — would look like the
       strongest enforcement in the file;
    4. `claim_for_hypothesis` refuses a non-token, and no claim-acquiring call appears
       anywhere in this module outside it, proved from the AST rather than promised.

    COULD_NOT_CHECK when the source cannot be read: inability to evaluate is a third
    outcome, never a soft PASS.
    """
    reasons: list = []

    absent = _authorization_probe(None)
    if not isinstance(absent, FalsifierRequiredBeforeCompute):
        reasons.append(
            "a ClaimAuthorization with NO falsifier "
            + (f"raises {type(absent).__name__} rather than "
               "FalsifierRequiredBeforeCompute" if absent is not None else
               "CONSTRUCTS; compute can be committed to a question nothing could refute")
        )
    placeholder = _authorization_probe("tbd")
    if not isinstance(placeholder, FalsifierRequiredBeforeCompute):
        reasons.append(
            "a ClaimAuthorization with a PLACEHOLDER falsifier "
            + (f"raises {type(placeholder).__name__} rather than "
               "FalsifierRequiredBeforeCompute" if placeholder is not None else
               "CONSTRUCTS; 'tbd' is an empty string wearing a hat and it stops nothing")
        )
    # The compliant-path control.
    real = _authorization_probe("a current wall-share map shows the cluster under 20%")
    if real is not None:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons) + (
            f"a ClaimAuthorization with a REAL falsifier also refuses to construct "
            f"({type(real).__name__}: {real}); the two refusals above cannot be told "
            "apart from a token type that refuses everything",
        ))

    try:
        bypassed = claim_for_hypothesis("akh-not-a-token", lambda **kw: kw)
    except FalsifierRequiredBeforeCompute:
        bypassed = None
    except Exception as exc:  # noqa: BLE001
        reasons.append(
            f"claim_for_hypothesis rejects a non-authorization with "
            f"{type(exc).__name__} rather than the typed FalsifierRequiredBeforeCompute"
        )
        bypassed = None
    if bypassed is not None:
        reasons.append(
            "claim_for_hypothesis ACCEPTS something that is not a ClaimAuthorization; "
            "the token type is the gate, and a door that takes anything is not one"
        )

    if source is None:
        try:
            with open(__file__, "r", encoding="utf-8") as handle:
                source = handle.read()
        except OSError as exc:  # pragma: no cover - unreadable own source
            return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons) + (
                f"could not read this module's source ({exc}); the one-door property "
                "cannot be evaluated, which is not the same as holding",
            ))
    try:
        sites = _claim_call_sites(source)
    except SyntaxError as exc:  # pragma: no cover - unparseable own source
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons) + (
            f"could not parse the source for claim call sites ({exc})",
        ))
    for name, scope, lineno in sites:
        if _CLAIM_DOOR not in scope:
            reasons.append(
                f"line {lineno}: {name}() is called from {'.'.join(scope) or '<module>'}"
                f", outside {_CLAIM_DOOR}(); a second route to a resource claim is a "
                "route that does not pass the falsifier gate"
            )

    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)
