#!/usr/bin/env python3
"""journal.py — AutoKernel's append-only, sharded, fsync-per-event primary record.

WHY THIS MODULE EXISTS
----------------------
`autokernel-research-loop.md` §5.5 makes one file the primary record of the whole
loop: an append-only `events.jsonl`, fsync-per-event, sharded with rotation only
past all cursors, from which every derived view (Pareto, frontier, champion,
failed-mechanism, do-not-repeat, readiness) is rebuilt. Invariants 7 and 8 say
the same thing from two directions: all outcomes are durable, and derived views
may rewind while evidence does not disappear. This module is the piece that has
to actually be true for either sentence to mean anything.

Four specific, previously-paid-for failures shape it:

  * **A restart came up with an empty frontier and nothing objected** — 232
    AutoPilot trials and roughly 16 days of compute, lost (§2.5 row 9). The
    architecture was right; there was no cardinality check. `rebuild_views()` is
    therefore paired with `assert_views_consistent()`, which BOOTSTRAP (§8.2 step
    10) calls before it proceeds. A journal holding candidates whose rebuilt view
    is empty RAISES. A deliberate rebase passes `deliberate_rebase=True` with a
    reason, so an intentional wipe is distinguishable from the failure rather
    than looking identical to it.

  * **Three separate shard-reading bugs cost this project real data.** A
    base-only read silently analysed frozen pre-rotation state for five days
    (`feedback_autopilot_journal_rotation_read_all_shards`: base frozen at trial
    999 while the live run wrote 1073 into `_1`). A lexicographic sort orders
    `_10` before `_2`. A `while os.path.exists(...)` discovery stops at the first
    missing index and silently drops everything after the hole. `shards()`
    answers all three structurally: it ENUMERATES both the live and archive
    directories (never probes an index sequence), orders by INTEGER index (never
    by name), and REFUSES a hole instead of stopping at it — a missing middle
    shard is data loss, and the only safe response to data loss is to say so.

  * **The planner re-consumed its own prose as fact** (§5.5 item 6, invariant
    20). AutoPilot's worst contamination lived in planner free text inside the
    primary journal; one instance ran 81 further trials on the same false story
    *after* the code fix landed. Prose stays in the record — invariant 7 — but
    `retrieve()` strips every `narrative` field recursively and admits one back
    only when the caller names its event id explicitly.

  * **An immutable log has no way to stop believing something.** §5.5 item 7 adds
    `RETRIEVAL_SUPERSEDED`: a belief leaves RETRIEVAL while staying in the
    RECORD. `retrieve()` honours it; `read_all()` does not, and must not — that
    asymmetry is the entire point, so the two APIs are named and tested apart.

DESIGN NOTES THAT ARE NOT OBVIOUS
---------------------------------
* **Acknowledged means fsynced.** `append()` returns only after `os.fsync()` on
  the shard fd (and on the containing directory when the shard is new). A crash
  can therefore leave a torn TRAILING line, which by definition was never
  acknowledged. `read_all()` drops it — and the next `append()` truncates it and
  writes a `TORN_APPEND_DISCARDED` event carrying the discarded bytes' length and
  hash, so the loss is itself durable rather than silent. Without that repair the
  next event would be concatenated onto the partial line and BOTH would be lost.

* **Lines are split on b"\\n" over BYTES, never `str.splitlines()`.** Canonical
  JSON is written with `ensure_ascii=False`, so a payload may legitimately
  contain U+2028/U+2029 — which `str.splitlines()` treats as line breaks and
  `b"\\n".split` does not. One record would otherwise read back as two.

* **The journal owns envelope identity.** `event_id` is assigned here and is
  unique by construction (it embeds the write-lock-serialised `seq`). Callers do
  not supply it, because a caller-supplied id would need a full-journal scan to
  prove unique and a duplicated event id silently corrupts every supersession
  reference that points at it.

* **Cursors record `last_seq`, not byte offsets.** A byte offset is a second
  source of truth about position that can disagree with the file; `seq` is
  carried in the record itself. Archiving is refused outright when no reader is
  registered — "no cursors" is not "all cursors have passed".

* **Reads do not re-validate payloads by default.** The WRITER is the schema
  gate: `append()` refuses an invalid record. Re-validating at read time would
  make a whole journal unreadable the day a schema gains a field, so payload
  validation is an explicit, separately-callable audit (`scan(validate_payloads=
  True)`) rather than an ambient precondition for reading history.

This module performs NO process management, runs NO inference, and runs NO
benchmark. It writes only inside the journal root it is given.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

# Both import shapes are supported explicitly rather than through a
# `try: from . import schemas / except ImportError: import schemas` fallback:
# that idiom swallows a genuinely missing dependency and reports it as a flat
# layout. Here a missing `schemas` module raises ImportError either way.
if __package__:
    from . import schemas
else:  # imported flat, e.g. after sys.path.insert(<this dir>)
    import schemas


# =============================================================================
# Errors — every one of these is a refusal to proceed, never a degraded mode
# =============================================================================

class JournalError(Exception):
    """Base class for every refusal this module makes."""


class JournalCorruption(JournalError):
    """The on-disk journal cannot be read as the record it claims to be."""


class ShardGapError(JournalCorruption):
    """A shard index is missing from the middle of the sequence.

    Raised rather than treated as the end of the journal. The historical bug was
    a `while os.path.exists(f"..._{n}.jsonl")` loop that stopped at the first
    absent index and silently dropped every later shard; stopping quietly is the
    behaviour that loses data, so a hole is a hard error.
    """


class ViewConsistencyError(JournalError):
    """A rebuilt view disagrees with the events it was supposedly built from."""


class SupersessionError(JournalError):
    """A supersession does not resolve to a real event, or is self-referential."""


class RetrievalCitationError(JournalError):
    """A retrieval citation names an event that may not be retrieved."""


class CursorError(JournalError):
    """A cursor operation would rewind, forge, or invent a reader."""


# =============================================================================
# Layout and vocabulary
# =============================================================================

JOURNAL_ENTRY_SCHEMA = "epyc.autokernel.journal_entry.v1"

BASE_SHARD_NAME = "events.jsonl"
ARCHIVE_DIRNAME = "archive"
CURSOR_DIRNAME = "cursors"
LOCK_NAME = ".write.lock"

# Canonical shard names only: `events.jsonl` is index 0, `events_<n>.jsonl` for
# n >= 1 with NO leading zeros. `events_007.jsonl` is deliberately unmatched —
# it would parse to the same integer index as `events_7.jsonl`, and two files
# claiming one position in the order is exactly the ambiguity that makes a
# journal unorderable.
_SHARD_RE = re.compile(r"^events(?:_([1-9][0-9]*))?\.jsonl$")
# Anything that looks like a shard but is not canonically named must be reported,
# not ignored: a shard we cannot place in the order is worse than no shard.
_SHARD_LOOKALIKE_RE = re.compile(r"^events.*\.jsonl$")

_READER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DEFAULT_MAX_SHARD_BYTES = 64 * 1024 * 1024

# Kinds whose payload IS one of the §7 records. `append()` validates these with
# schemas.py, so an invalid record cannot enter the primary journal at all.
KIND_CAMPAIGN_OPENED = "CAMPAIGN_OPENED"
KIND_PROPOSAL_RECORDED = "PROPOSAL_RECORDED"
KIND_CANDIDATE_RECORDED = "CANDIDATE_RECORDED"
KIND_EVALUATION_EVENT = "EVALUATION_EVENT"
KIND_CHAMPION_UPDATED = "CHAMPION_UPDATED"
KIND_RELEASE_PACKAGE_PREPARED = "RELEASE_PACKAGE_PREPARED"
KIND_OPERATOR_WAIVER_RECORDED = "OPERATOR_WAIVER_RECORDED"

#: The CURRENT schema string each schema-bound kind emits.
SCHEMA_BOUND_KINDS = {
    KIND_CAMPAIGN_OPENED: schemas.SCHEMA_CAMPAIGN,
    KIND_PROPOSAL_RECORDED: schemas.SCHEMA_PROPOSAL,
    KIND_CANDIDATE_RECORDED: schemas.SCHEMA_CANDIDATE,
    KIND_EVALUATION_EVENT: schemas.SCHEMA_EVALUATION_EVENT,
    KIND_CHAMPION_UPDATED: schemas.SCHEMA_CHAMPION,
    KIND_RELEASE_PACKAGE_PREPARED: schemas.SCHEMA_RELEASE_PACKAGE,
    KIND_OPERATOR_WAIVER_RECORDED: schemas.SCHEMA_OPERATOR_WAIVER,
}

# Every schema version a kind ADMITS. A kind pinned to exactly one string could
# not accept a record written before its contract was versioned, so a schema
# revision would have made the previous generation of records unappendable —
# including on a replay or an import of an existing shard. The set is closed and
# explicit: `schemas.SCHEMA_REGISTRY` decides validity, this decides whether the
# payload belongs under this kind at all.
ACCEPTED_SCHEMAS_BY_KIND = {
    **{kind: frozenset({schema}) for kind, schema in SCHEMA_BOUND_KINDS.items()},
    KIND_EVALUATION_EVENT: frozenset({
        schemas.SCHEMA_EVALUATION_EVENT_V2,
        schemas.SCHEMA_EVALUATION_EVENT_V3,
        schemas.SCHEMA_EVALUATION_EVENT_V4,
        schemas.SCHEMA_EVALUATION_EVENT_V5,
    }),
    KIND_PROPOSAL_RECORDED: frozenset({
        schemas.SCHEMA_PROPOSAL_V2,
        schemas.SCHEMA_PROPOSAL_V3,
    }),
}

# Journal-native kinds: the record shape belongs to the journal itself, so it is
# validated here.
KIND_SUPERSEDED = "SUPERSEDED"
KIND_RETRIEVAL_SUPERSEDED = "RETRIEVAL_SUPERSEDED"
KIND_TOMBSTONE = "TOMBSTONE"
KIND_TORN_APPEND_DISCARDED = "TORN_APPEND_DISCARDED"
KIND_OPERATOR_CONTROL_ACK = "OPERATOR_CONTROL_ACK"
KIND_VIEW_REBASED = "VIEW_REBASED"
KIND_PROPOSAL_SKIPPED = "PROPOSAL_SKIPPED"
KIND_STOP_STATE = "STOP_STATE"
KIND_MICROBENCH_RUN_COMPLETED = "MICROBENCH_RUN_COMPLETED"
KIND_T0_REFUSAL = "T0_REFUSAL"
KIND_COMPOSITION_REQUESTED = "COMPOSITION_REQUESTED"
KIND_COMPOSITION_FAILED = "COMPOSITION_FAILED"
KIND_COMPOSITION_REJECTED = "COMPOSITION_REJECTED"
# §3.5 preflight attestation. `resource/preflight.py` builds the verdict and its
# own docstring instructs the caller to journal `exc.result.to_dict()` verbatim
# on FAIL and COULD_NOT_CHECK — "a precondition that was checked but not recorded
# is indistinguishable from one that was skipped" — but the kind vocabulary here
# is CLOSED, so until this entry existed that instruction could not be followed:
# `append("PREFLIGHT_ATTESTATION", ...)` raised, and the only two outcomes
# invariant 7 exists for had nowhere durable to go. Two modules wrote two halves
# of one contract and neither half was wrong on its own.
KIND_PREFLIGHT_ATTESTATION = "PREFLIGHT_ATTESTATION"

# §19.4 bootstrap-knowledge event types. Their payloads are campaign-specific
# structures owned by the bootstrap corpus task, so they are checked only for
# being non-empty mappings; naming them keeps the kind vocabulary CLOSED, which
# is what stops a typo'd kind from entering the record as a new category.
BOOTSTRAP_KNOWLEDGE_KINDS = frozenset({
    "LEGACY_SOURCE_DISCOVERED", "LEGACY_EVIDENCE_IMPORTED", "PRIOR_ATOMIZED",
    "PRIOR_SOURCE_VERIFIED", "PRIOR_SUPERSEDED", "PRIOR_CONTRADICTION_LINKED",
    "CONSTRAINT_COMPILED", "SEED_COMPILED", "SEED_BLOCKED", "SEED_REOPENED",
})

NATIVE_KINDS = frozenset({
    KIND_SUPERSEDED, KIND_RETRIEVAL_SUPERSEDED, KIND_TOMBSTONE,
    KIND_TORN_APPEND_DISCARDED, KIND_OPERATOR_CONTROL_ACK, KIND_VIEW_REBASED,
    KIND_PROPOSAL_SKIPPED, KIND_STOP_STATE, KIND_PREFLIGHT_ATTESTATION,
    KIND_MICROBENCH_RUN_COMPLETED,
    KIND_T0_REFUSAL,
    KIND_COMPOSITION_REQUESTED,
    KIND_COMPOSITION_FAILED, KIND_COMPOSITION_REJECTED,
}) | BOOTSTRAP_KNOWLEDGE_KINDS

KINDS = frozenset(SCHEMA_BOUND_KINDS) | NATIVE_KINDS

# Which payload key carries the record's own identity, per kind. Explicit rather
# than "first id-ish key found": a guessed identity key silently merges two
# different records into one view slot.
RECORD_ID_KEY_BY_KIND = {
    KIND_CAMPAIGN_OPENED: "campaign_id",
    KIND_PROPOSAL_RECORDED: "proposal_id",
    KIND_CANDIDATE_RECORDED: "candidate_id",
    KIND_EVALUATION_EVENT: "event_id",
    KIND_CHAMPION_UPDATED: "branch",
    KIND_RELEASE_PACKAGE_PREPARED: "package_id",
    KIND_OPERATOR_WAIVER_RECORDED: "waiver_id",
    KIND_COMPOSITION_REQUESTED: "request_sha256",
    KIND_COMPOSITION_FAILED: "request_sha256",
    KIND_COMPOSITION_REJECTED: "attempt_sha256",
}

# §5.8 storage classes. Only the expirable class may ever be tombstoned:
# invariant 7 says evidence is not evicted, so a tombstone naming a permanent
# class is a policy violation caught at write time, not a storage detail.
STORAGE_CLASSES = frozenset({
    "permanent_in_repo", "permanent_large", "expirable", "never_stored",
})
TOMBSTONABLE_STORAGE_CLASSES = frozenset({"expirable"})


def tombstone_view_key(payload: Mapping[str, Any]) -> str:
    """The identity of one reclamation: content hash AND the path it sat at.

    The hash alone was the key, and it is not an identity. `storage.py` derives
    its own `tombstone_id` from (campaign, path, sha256, kind, rule), so two
    reclamations of BYTE-IDENTICAL trees at two different paths are two records
    there and were ONE slot here — measured: two `expire_artifact` calls, two
    distinct `tombstone_id`s, four TOMBSTONE events, and a `tombstones` view
    holding a single entry that reported only the second path, with
    `check_view_consistency` returning PASS because it recounted by the same
    key it was checking. A receipt view that silently drops a reclamation is
    the §5.8/invariant-8 loss the view exists to make impossible.

    `path` is therefore REQUIRED by `_validate_native_payload`, which is what
    makes this key total.
    """
    return f"{payload.get('artifact_sha256')}@{payload.get('path')}"

# Prose keys stripped from every retrieval result, at every depth (§5.5 item 6).
NARRATIVE_KEYS = frozenset({"narrative"})

#: Bound, not re-compiled. The digest SHAPE has one owner (`schemas`), because a
#: local `re.compile(r"^[0-9a-f]{64}$")` is the first line every re-derived digest
#: validator in this package started with, and two of those forgot to also refuse
#: a placeholder. See the `require` header in `schemas.py`.
_SHA256_RE = schemas.SHA256_RE


def _iso_now() -> str:
    """Timezone-aware UTC timestamp; schemas.py rejects naive ones on purpose."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


# =============================================================================
# Entries, defects, shards, cursors
# =============================================================================

@dataclass(frozen=True)
class JournalEntry:
    """One append-only line. `seq` is assigned under the write lock and is the
    journal's ordering truth; `written_at` is diagnostic and never sorted on."""

    event_id: str
    seq: int
    kind: str
    campaign_id: Optional[str]
    record_id: Optional[str]
    written_at: str
    payload: Mapping[str, Any]
    shard_index: int = -1        # where it was read from; -1 when not from disk
    line_number: int = -1

    def envelope(self) -> dict:
        """The exact dict that is serialised to the shard line."""
        return {
            "journal_schema": JOURNAL_ENTRY_SCHEMA,
            "event_id": self.event_id,
            "seq": self.seq,
            "kind": self.kind,
            "campaign_id": self.campaign_id,
            "record_id": self.record_id,
            "written_at": self.written_at,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class JournalDefect:
    """A line that could not be read as a record. Never silently skipped: the
    reader collects these and `read_all()` refuses to return a partial history
    as if it were the whole one."""

    shard_index: int
    line_number: int
    reason: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - formatting only
        where = f"shard {self.shard_index} line {self.line_number}"
        return f"{where}: {self.reason}{(' — ' + self.detail) if self.detail else ''}"


@dataclass(frozen=True)
class ShardRef:
    index: int
    path: str
    archived: bool


@dataclass(frozen=True)
class TornTail:
    """Bytes at the end of the last shard with no terminating newline — an
    append that was never acknowledged, i.e. a crash mid-write."""

    shard_index: int
    byte_count: int
    sha256: str
    prefix_hex: str


@dataclass(frozen=True)
class ReadReport:
    entries: tuple
    defects: tuple
    torn_tail: Optional[TornTail]


@dataclass(frozen=True)
class Cursor:
    reader_id: str
    last_seq: int
    updated_at: str


# =============================================================================
# Derived views (§5.5 item 4) — a pure function of the event list
# =============================================================================

@dataclass(frozen=True)
class Views:
    """Record-level derived state.

    `source_digest` binds the view to the exact event list it was folded from.
    That binding is what lets `check_view_consistency()` return COULD_NOT_CHECK
    instead of a confident PASS when a caller rebuilds from one read (say, a
    base-only one) and asserts against another.

    SUPERSEDED entries are excluded from the fold — the record itself says it was
    replaced. RETRIEVAL_SUPERSEDED entries are NOT excluded: they remain part of
    the record and only the retrieval layer withholds them (§5.5 item 7). The two
    sets are exposed separately so no consumer has to guess which is which.

    THESE VIEWS ARE RECORD-SCOPE, NOT RETRIEVAL-SCOPE. Slot payloads are the raw
    payloads, so they still carry every `narrative` field, including the
    narrative of a belief that has been RETRIEVAL_SUPERSEDED — which is exactly
    the prose invariant 20 exists to keep out of a planning context. Nothing
    here enforces that boundary; `Journal.retrieve()` does. A consumer that
    renders these views into a planner brief must apply
    `strip_narrative()` and `retrieval_superseded_event_ids` itself, or it
    rebuilds the contamination §5.5 item 6 was written against.
    """

    source_digest: str
    entry_count: int
    campaigns: Mapping[str, Mapping[str, Any]]
    proposals: Mapping[str, Mapping[str, Any]]
    candidates: Mapping[str, Mapping[str, Any]]
    evaluations: Mapping[str, Mapping[str, Any]]
    champions: Mapping[str, Mapping[str, Any]]
    release_packages: Mapping[str, Mapping[str, Any]]
    waivers: Mapping[str, Mapping[str, Any]]
    tombstones: Mapping[str, Mapping[str, Any]]
    frontier: tuple
    superseded_event_ids: frozenset
    retrieval_superseded_event_ids: frozenset
    stop_states: tuple

    def cardinalities(self) -> dict:
        return {
            "entries": self.entry_count,
            "campaigns": len(self.campaigns),
            "proposals": len(self.proposals),
            "candidates": len(self.candidates),
            "evaluations": len(self.evaluations),
            "champions": len(self.champions),
            "release_packages": len(self.release_packages),
            "waivers": len(self.waivers),
            "tombstones": len(self.tombstones),
            "frontier": len(self.frontier),
            "stop_states": len(self.stop_states),
        }


def events_digest(events: Sequence[JournalEntry]) -> str:
    """Content hash of the ordered (seq, event_id) spine of an event list."""
    spine = [[int(e.seq), str(e.event_id)] for e in events]
    return schemas.content_hash(spine)


def rebuild_views(events: Sequence[JournalEntry]) -> Views:
    """Fold events into derived views. Pure: no I/O, no clock, no globals.

    Two passes on purpose. Supersession events may arrive AFTER the record they
    supersede, so the superseded set has to be complete before the fold decides
    what to keep; a single streaming pass would leave a superseded record in the
    view whenever the SUPERSEDED event happened to come later, which is always.
    """
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise TypeError(f"events must be a sequence, got {type(events).__name__}")
    for i, entry in enumerate(events):
        if not isinstance(entry, JournalEntry):
            raise TypeError(
                f"events[{i}]: expected JournalEntry, got {type(entry).__name__}"
            )
        if not isinstance(entry.payload, Mapping):
            raise TypeError(
                f"events[{i}].payload: expected a mapping, got "
                f"{type(entry.payload).__name__}"
            )

    ordered = sorted(events, key=lambda e: e.seq)

    superseded: set = set()
    retrieval_superseded: set = set()
    for entry in ordered:
        if entry.kind == KIND_SUPERSEDED:
            target = entry.payload.get("target_event_id")
            if isinstance(target, str):
                superseded.add(target)
        elif entry.kind == KIND_RETRIEVAL_SUPERSEDED:
            target = entry.payload.get("target_event_id")
            if isinstance(target, str):
                retrieval_superseded.add(target)

    campaigns: dict = {}
    proposals: dict = {}
    candidates: dict = {}
    evaluations: dict = {}
    champions: dict = {}
    release_packages: dict = {}
    waivers: dict = {}
    tombstones: dict = {}
    stop_states: list = []

    slot_by_kind = {
        KIND_CAMPAIGN_OPENED: campaigns,
        KIND_PROPOSAL_RECORDED: proposals,
        KIND_CANDIDATE_RECORDED: candidates,
        KIND_EVALUATION_EVENT: evaluations,
        KIND_RELEASE_PACKAGE_PREPARED: release_packages,
        # Waivers are a §7 record with a declared identity key in
        # RECORD_ID_KEY_BY_KIND and a write-time validator, and they were still
        # folded into nothing: an OPERATOR_WAIVER_RECORDED event landed in the
        # journal, got a record_id, and then appeared in no view — while
        # check_view_consistency reported PASS, because it only inspected the
        # five families that HAD slots. §5.6 makes waivers a first-class T3
        # input; a release view that cannot see the active waivers is the wrong
        # view.
        KIND_OPERATOR_WAIVER_RECORDED: waivers,
    }

    for entry in ordered:
        if entry.event_id in superseded:
            continue
        slot = slot_by_kind.get(entry.kind)
        if slot is not None:
            if entry.record_id is None:
                # An identity-less record cannot be folded into a keyed view; it
                # stays in the record and is reported by the consistency check.
                continue
            slot[entry.record_id] = entry.payload
        elif entry.kind == KIND_CHAMPION_UPDATED:
            source_tree = entry.payload.get("source_tree")
            if isinstance(source_tree, str):
                champions[source_tree] = entry.payload
        elif entry.kind == KIND_TOMBSTONE:
            # Keyed by (artifact_sha256, path), not by the hash alone — see
            # `tombstone_view_key`. The write-time validator makes both fields
            # present, so an entry reaching here always has a total key.
            tombstones[tombstone_view_key(entry.payload)] = entry.payload
        elif entry.kind == KIND_STOP_STATE:
            stop_states.append(entry.payload)

    frontier = tuple(sorted(
        candidate_id for candidate_id, record in candidates.items()
        if record.get("status") == "banked"
    ))

    return Views(
        source_digest=events_digest(events),
        entry_count=len(events),
        campaigns=campaigns,
        proposals=proposals,
        candidates=candidates,
        evaluations=evaluations,
        champions=champions,
        release_packages=release_packages,
        waivers=waivers,
        tombstones=tombstones,
        frontier=frontier,
        superseded_event_ids=frozenset(superseded),
        retrieval_superseded_event_ids=frozenset(retrieval_superseded),
        stop_states=tuple(stop_states),
    )


def check_view_consistency(
    events: Sequence[JournalEntry], views: Views
) -> schemas.Check:
    """PASS / FAIL / COULD_NOT_CHECK on "do these views describe these events?".

    The recount here is deliberately written a SECOND time — as set comprehensions
    over the raw events rather than by calling `rebuild_views` again — so the
    check has a chance of disagreeing with the fold. A check that reuses the code
    it is checking can only ever confirm it.

    COULD_NOT_CHECK is a real outcome, not a soft pass: if the views were not
    built from these events (digest mismatch), we have not learned that they are
    consistent, and we have not learned that they are not.
    """
    if not isinstance(views, Views):
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"views is {type(views).__name__}, not a Views built by rebuild_views()",),
        )
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"events is {type(events).__name__}, not a sequence of JournalEntry",),
        )
    if any(not isinstance(e, JournalEntry) for e in events):
        return schemas.Check(
            schemas.COULD_NOT_CHECK, ("events contains a non-JournalEntry item",)
        )
    if any(not isinstance(e.payload, Mapping) for e in events):
        return schemas.Check(
            schemas.COULD_NOT_CHECK, ("events contains an entry with a non-mapping "
                                      "payload",)
        )
    if views.source_digest != events_digest(events):
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("views were rebuilt from a different event list than the one being "
             "asserted against (source_digest mismatch) — a base-only read "
             "checked against an all-shard read looks exactly like this",),
        )

    reasons: list = []

    seqs = [e.seq for e in events]
    if sorted(seqs) != seqs:
        reasons.append("events are not in ascending seq order")
    if len(set(seqs)) != len(seqs):
        reasons.append("duplicate seq numbers in the event list")
    ids = [e.event_id for e in events]
    if len(set(ids)) != len(ids):
        reasons.append("duplicate event_id in the event list")

    superseded = {
        e.payload.get("target_event_id") for e in events
        if e.kind == KIND_SUPERSEDED
    }
    superseded.discard(None)
    retrieval_superseded = {
        e.payload.get("target_event_id") for e in events
        if e.kind == KIND_RETRIEVAL_SUPERSEDED
    }
    retrieval_superseded.discard(None)

    known_ids = set(ids)
    dangling = sorted((superseded | retrieval_superseded) - known_ids)
    if dangling:
        reasons.append(
            f"supersession targets that resolve to no event: {dangling}"
        )
    if superseded != set(views.superseded_event_ids):
        reasons.append("superseded_event_ids disagrees with a recount from events")
    if retrieval_superseded != set(views.retrieval_superseded_event_ids):
        reasons.append(
            "retrieval_superseded_event_ids disagrees with a recount from events"
        )

    if views.entry_count != len(events):
        reasons.append(
            f"views.entry_count={views.entry_count} but {len(events)} events were "
            "supplied"
        )

    families = (
        ("campaigns", KIND_CAMPAIGN_OPENED, views.campaigns),
        ("proposals", KIND_PROPOSAL_RECORDED, views.proposals),
        ("candidates", KIND_CANDIDATE_RECORDED, views.candidates),
        ("evaluations", KIND_EVALUATION_EVENT, views.evaluations),
        ("release_packages", KIND_RELEASE_PACKAGE_PREPARED, views.release_packages),
        ("waivers", KIND_OPERATOR_WAIVER_RECORDED, views.waivers),
    )
    for name, kind, view in families:
        expected = {
            e.record_id for e in events
            if e.kind == kind and e.record_id is not None
            and e.event_id not in superseded
        }
        identity_less = [
            e.event_id for e in events if e.kind == kind and e.record_id is None
        ]
        if identity_less:
            reasons.append(
                f"{name}: {len(identity_less)} record(s) carry no record_id and "
                f"cannot be folded into a view: {sorted(identity_less)}"
            )
        # The AutoPilot loss, stated exactly: the journal holds records of this
        # family and the rebuilt view came up empty.
        if expected and not view:
            reasons.append(
                f"{name}: the journal holds {len(expected)} record(s) but the "
                "rebuilt view is EMPTY (§2.5 row 9 — a restart that came up with "
                "an empty frontier lost 232 trials)"
            )
        elif set(view) != expected:
            reasons.append(
                f"{name}: view holds {len(view)} distinct id(s), events name "
                f"{len(expected)}; missing={sorted(expected - set(view))}, "
                f"unexpected={sorted(set(view) - expected)}"
            )

    banked = {
        e.record_id for e in events
        if e.kind == KIND_CANDIDATE_RECORDED and e.record_id is not None
        and e.event_id not in superseded
        and isinstance(e.payload, Mapping) and e.payload.get("status") == "banked"
    }
    # Recomputed against the view's own latest-record-wins semantics: a candidate
    # banked and then re-recorded as rejected is legitimately off the frontier.
    still_banked = {
        candidate_id for candidate_id in banked
        if views.candidates.get(candidate_id, {}).get("status") == "banked"
    }
    if still_banked and not views.frontier:
        reasons.append(
            f"frontier is EMPTY while {len(still_banked)} candidate(s) are banked "
            "in the journal"
        )
    elif set(views.frontier) != still_banked:
        reasons.append(
            f"frontier {sorted(views.frontier)} disagrees with the banked "
            f"candidates {sorted(still_banked)}"
        )

    # Tombstones and stop states were folded by rebuild_views() and inspected by
    # nothing: deleting either fold outright left this checker returning PASS.
    # Tombstones are the §5.8 receipts for every deleted expirable artifact —
    # the one view whose disappearance means evidence went missing unrecorded —
    # so a view that silently drops them is the failure invariants 7 and 8 are
    # written against.
    live_tombstones = [
        e for e in events
        if e.kind == KIND_TOMBSTONE and e.event_id not in superseded
        and isinstance(e.payload, Mapping)
        and isinstance(e.payload.get("artifact_sha256"), str)
    ]
    expected_tombstones = {tombstone_view_key(e.payload) for e in live_tombstones}
    if expected_tombstones and not views.tombstones:
        reasons.append(
            f"tombstones view is EMPTY while the journal holds "
            f"{len(expected_tombstones)} tombstone(s)"
        )
    elif set(views.tombstones) != expected_tombstones:
        reasons.append(
            f"tombstones view covers {len(views.tombstones)} reclamation(s), events "
            f"name {len(expected_tombstones)}; "
            f"missing={sorted(expected_tombstones - set(views.tombstones))}, "
            f"unexpected={sorted(set(views.tombstones) - expected_tombstones)}"
        )
    # A SECOND, independently keyed recount of the same fold. `storage.py` gives
    # every tombstone it writes a content-addressed `tombstone_id` over
    # (campaign, path, sha256, kind, rule) — a strictly finer identity than this
    # module's — and the `intent`/`reclaimed`/`failed` records of ONE reclamation
    # deliberately share it. So when every live tombstone carries one, the count
    # of distinct ids is the count of distinct reclamations, and it must equal
    # the number of view slots. This is the check that catches a key formula
    # which is itself too coarse, rather than confirming it.
    declared_ids = [
        e.payload.get("tombstone_id") for e in live_tombstones
    ]
    if declared_ids and all(isinstance(i, str) and i for i in declared_ids):
        distinct_ids = set(declared_ids)
        if len(distinct_ids) != len(views.tombstones):
            reasons.append(
                f"tombstones view holds {len(views.tombstones)} slot(s) but the "
                f"journal records {len(distinct_ids)} distinct reclamation id(s) "
                f"{sorted(distinct_ids)}: two reclamations collapsed into one "
                "receipt, so one deleted artifact has no visible tombstone"
            )

    expected_stop_states = sum(
        1 for e in events
        if e.kind == KIND_STOP_STATE and e.event_id not in superseded
    )
    if len(views.stop_states) != expected_stop_states:
        reasons.append(
            f"stop_states view holds {len(views.stop_states)} entr(ies) but events "
            f"name {expected_stop_states}"
        )

    champion_trees = {
        e.payload.get("source_tree") for e in events
        if e.kind == KIND_CHAMPION_UPDATED and e.event_id not in superseded
        and isinstance(e.payload, Mapping)
    }
    champion_trees.discard(None)
    if champion_trees and not views.champions:
        reasons.append(
            f"champions view is EMPTY while the journal holds champion records "
            f"for {sorted(champion_trees)}"
        )
    elif set(views.champions) != champion_trees:
        reasons.append(
            f"champions view covers {sorted(views.champions)}, events name "
            f"{sorted(champion_trees)}"
        )

    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def assert_views_consistent(
    events: Sequence[JournalEntry],
    views: Views,
    *,
    deliberate_rebase: bool = False,
    rebase_reason: Optional[str] = None,
) -> schemas.Check:
    """BOOTSTRAP §8.2 step 10. Raises unless the views describe the events.

    `deliberate_rebase=True` is the explicit escape for an intentional wipe, and
    it demands a non-empty `rebase_reason` so the intent lands in the record
    rather than in someone's memory of why the frontier was empty that morning.
    It downgrades FAIL to a returned report — it does NOT cover COULD_NOT_CHECK,
    because "I meant to empty the views" is not an answer to "I cannot tell
    whether these views belong to these events".
    """
    check = check_view_consistency(events, views)
    if check.outcome == schemas.PASS:
        return check
    if check.outcome == schemas.COULD_NOT_CHECK:
        raise ViewConsistencyError(
            "view consistency COULD NOT BE CHECKED, which is not permission to "
            "proceed: " + "; ".join(check.reasons)
        )
    if not deliberate_rebase:
        raise ViewConsistencyError(
            "rebuilt views disagree with the journal: " + "; ".join(check.reasons)
        )
    if not isinstance(rebase_reason, str) or not rebase_reason.strip():
        raise ValueError(
            "deliberate_rebase=True requires a non-empty rebase_reason; an "
            "unexplained rebase is indistinguishable from the failure it "
            "suppresses"
        )
    return check


# =============================================================================
# Retrieval (§5.5 items 6 and 7, invariant 20)
# =============================================================================

def strip_narrative(value: Any) -> Any:
    """Recursively drop every `narrative` key.

    Recursive on purpose. `schemas.retrievable_view()` strips the top level,
    which is right for a record whose shape it knows; a journal payload can nest
    prose one level down inside a mechanism or critic block, and a top-level-only
    strip would pass it straight through to the planner.
    """
    if isinstance(value, Mapping):
        return {
            k: strip_narrative(v) for k, v in value.items()
            if k not in NARRATIVE_KEYS
        }
    if isinstance(value, list):
        return [strip_narrative(v) for v in value]
    return value


def retrieval_filter(
    entries: Sequence[JournalEntry],
    *,
    supersession_basis: Sequence[JournalEntry],
    cite_event_ids: Iterable[str] = (),
) -> list:
    """The RETRIEVAL view of a record list: superseded beliefs out, prose out.

    Two withholdings, both from §5.5:
      * an event targeted by SUPERSEDED or RETRIEVAL_SUPERSEDED is withheld — the
        loop has stopped believing it, even though the record keeps it forever;
      * every `narrative` field is stripped, at every depth.

    `cite_event_ids` is the ONLY way prose comes back, and it admits prose for
    exactly the cited events. Citing an event that was retrieval-superseded
    RAISES: that path is precisely how a withdrawn belief would walk back into a
    planning context, and letting it through quietly would rebuild the
    contamination this API exists to prevent.

    `supersession_basis` is REQUIRED, and it is required because the earlier
    signature made this function silently untrue for the most natural way to
    call it. Withholding was derived from `entries` itself, so a caller who
    narrowed the list first — `retrieval_filter([e for e in read_all() if
    e.kind == CANDIDATE_RECORDED])`, an obvious thing to write — dropped the
    RETRIEVAL_SUPERSEDED events out of the very list the withholding was
    computed from, and every withdrawn belief came straight back with the same
    confident shape as a live one. Pass the COMPLETE journal here (normally
    `Journal.read_all()`); `entries` may then be any subset.
    """
    if not isinstance(supersession_basis, Sequence) or isinstance(
        supersession_basis, (str, bytes)
    ):
        raise TypeError(
            "supersession_basis must be the complete journal (a sequence of "
            f"JournalEntry), got {type(supersession_basis).__name__}"
        )
    cited = list(cite_event_ids)
    for event_id in cited:
        if not isinstance(event_id, str):
            raise TypeError(
                f"cite_event_ids must contain strings, got {type(event_id).__name__}"
            )
    cited_set = set(cited)

    withheld: set = set()
    by_id: dict = {}
    for entry in supersession_basis:
        by_id[entry.event_id] = entry
        if entry.kind in (KIND_SUPERSEDED, KIND_RETRIEVAL_SUPERSEDED):
            target = entry.payload.get("target_event_id")
            if isinstance(target, str):
                withheld.add(target)

    unknown = sorted(cited_set - set(by_id))
    if unknown:
        raise RetrievalCitationError(
            f"cited event id(s) do not exist in this journal: {unknown}"
        )
    withdrawn = sorted(cited_set & withheld)
    if withdrawn:
        raise RetrievalCitationError(
            f"cited event id(s) were superseded out of retrieval and may not be "
            f"cited back in: {withdrawn}"
        )

    out: list = []
    for entry in entries:
        if entry.event_id in withheld:
            continue
        payload = dict(entry.payload)
        if entry.event_id not in cited_set:
            payload = strip_narrative(payload)
        out.append({
            "event_id": entry.event_id,
            "seq": entry.seq,
            "kind": entry.kind,
            "campaign_id": entry.campaign_id,
            "record_id": entry.record_id,
            "written_at": entry.written_at,
            "payload": payload,
        })
    return out


# =============================================================================
# Native payload validators
# =============================================================================

def _validate_native_payload(kind: str, payload: Mapping[str, Any]) -> list:
    out: list = []
    if kind in (KIND_SUPERSEDED, KIND_RETRIEVAL_SUPERSEDED):
        target = payload.get("target_event_id")
        if not isinstance(target, str) or not target:
            out.append("target_event_id: required, must be a non-empty event id")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            out.append(
                "reason: required and non-empty — supersession without a stated "
                "reason is indistinguishable from a deletion (invariant 8)"
            )
        if kind == KIND_SUPERSEDED:
            replacement = payload.get("superseded_by")
            if replacement is not None and not isinstance(replacement, str):
                out.append("superseded_by: must be null or an event id")
        else:
            receipt = payload.get("receipt")
            if not isinstance(receipt, str) or not receipt.strip():
                out.append(
                    "receipt: required and non-empty — §19.3 requires a receipt "
                    "on every suppressing entry, not a confident sentence"
                )
    elif kind == KIND_TOMBSTONE:
        artifact = payload.get("artifact_sha256")
        if not isinstance(artifact, str) or not _SHA256_RE.match(artifact):
            out.append("artifact_sha256: required, lowercase hex sha256")
        storage_class = payload.get("storage_class")
        if storage_class not in STORAGE_CLASSES:
            out.append(
                f"storage_class: required, one of {sorted(STORAGE_CLASSES)}"
            )
        elif storage_class not in TOMBSTONABLE_STORAGE_CLASSES:
            out.append(
                f"storage_class: {storage_class!r} may not be tombstoned; only "
                f"{sorted(TOMBSTONABLE_STORAGE_CLASSES)} expires (invariant 7 — "
                "evidence is never evicted)"
            )
        size = payload.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            out.append("size_bytes: required, a non-negative integer")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            out.append("reason: required and non-empty")
        # REQUIRED, and it was optional. The receipt view is keyed by
        # (artifact_sha256, path) because the hash alone merges two reclamations
        # of byte-identical trees into one slot; a tombstone that cannot say
        # WHICH path was removed cannot be told apart from another one, which is
        # the same "records that something was removed but not WHAT" failure
        # `storage.plan_expiry` already refuses. `storage` always supplies it.
        path = payload.get("path")
        if not isinstance(path, str) or not path.strip():
            out.append(
                "path: required and non-empty — a tombstone that does not name "
                "the path it reclaimed is indistinguishable from another "
                "reclamation of byte-identical bytes (see tombstone_view_key)"
            )
        elif not path.startswith("/"):
            out.append(f"path: {path!r} must be absolute")
    elif kind == KIND_PREFLIGHT_ATTESTATION:
        # Mirrors `preflight.PreflightResult.__post_init__` deliberately: the
        # durable record must not be able to say less than the in-memory object
        # it came from. A FAIL that cannot name what and whose is unactionable,
        # and a COULD_NOT_CHECK that cannot say why is a soft pass in disguise.
        verdict = payload.get("verdict")
        if verdict not in (schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK):
            out.append(
                f"verdict: required, one of "
                f"{sorted((schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK))}"
            )
        for key in ("basis", "observed_at"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"{key}: required and non-empty")
        scope = payload.get("scope")
        if not isinstance(scope, Mapping):
            out.append("scope: required, the mapping the preflight attested over")
        elif not isinstance(scope.get("label"), str) or not scope["label"].strip():
            out.append("scope.label: required and non-empty")
        findings = payload.get("findings")
        reasons = payload.get("reasons")
        if findings is not None and not isinstance(findings, list):
            out.append("findings: must be a list when present")
        if reasons is not None and not isinstance(reasons, list):
            out.append("reasons: must be a list when present")
        if verdict == schemas.FAIL and not findings:
            out.append(
                "findings: a FAIL attestation must name at least one finding "
                "(what is running, and whose)"
            )
        if verdict == schemas.COULD_NOT_CHECK and not reasons:
            out.append(
                "reasons: a COULD_NOT_CHECK attestation must say why it could "
                "not check — inability to evaluate is a third outcome, not a pass"
            )
    elif kind == KIND_TORN_APPEND_DISCARDED:
        for key in ("discarded_byte_count", "shard_index"):
            value = payload.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                out.append(f"{key}: required, a non-negative integer")
        if not isinstance(payload.get("discarded_sha256"), str):
            out.append("discarded_sha256: required")
    elif kind == KIND_OPERATOR_CONTROL_ACK:
        # Invariant 19: an unacked control is a hard failure, so the ack has to
        # name the control it acknowledges and when it was received.
        for key in ("control", "control_id", "received_at", "disposition"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"{key}: required and non-empty")
    elif kind == KIND_VIEW_REBASED:
        reason = payload.get("rebase_reason")
        if not isinstance(reason, str) or not reason.strip():
            out.append("rebase_reason: required and non-empty")
        if not isinstance(payload.get("suppressed_reasons"), list):
            out.append("suppressed_reasons: required, a list of the checks waived")
    elif kind == KIND_STOP_STATE:
        state = payload.get("state")
        if not isinstance(state, str) or not state.strip():
            out.append("state: required and non-empty")
    elif kind == KIND_MICROBENCH_RUN_COMPLETED:
        for key in ("campaign_id", "candidate_id", "run_id", "completed_at"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"{key}: required and non-empty")
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and not _SHA256_RE.match(run_id):
            out.append("run_id: must be a lowercase hex sha256")
        attempt = payload.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            out.append("attempt: required, a non-negative integer")
        segment = payload.get("segment")
        if segment not in ("base", "extension"):
            out.append("segment: required, one of ['base', 'extension']")
        extension_round = payload.get("extension_round")
        if segment == "base" and extension_round is not None:
            out.append("extension_round: must be null on a base run")
        if segment == "extension" and (
                not isinstance(extension_round, int)
                or isinstance(extension_round, bool)
                or extension_round < 1):
            out.append("extension_round: must be a positive integer on an extension run")
        if not isinstance(payload.get("complete"), bool):
            out.append("complete: required, a boolean")
        raw_vector = payload.get("raw_vector")
        if not isinstance(raw_vector, Mapping):
            out.append("raw_vector: required, the completed MicrobenchRun record")
        else:
            for key in ("candidate_id", "attempt", "segment", "extension_round",
                        "complete"):
                if raw_vector.get(key) != payload.get(key):
                    out.append(f"raw_vector.{key}: must equal the ledger envelope")
            if raw_vector.get("ended_at") != payload.get("completed_at"):
                out.append("raw_vector.ended_at: must equal completed_at")
            try:
                raw_id = schemas.content_hash(raw_vector)
            except (TypeError, ValueError) as exc:
                out.append(f"raw_vector: cannot be content-hashed: {exc}")
            else:
                if run_id != raw_id:
                    out.append("run_id: must be the content hash of raw_vector")
    elif kind == KIND_T0_REFUSAL:
        for key, prefix in (("campaign_id", "ak-"), ("candidate_id", "akc-")):
            value = payload.get(key)
            if not isinstance(value, str) or not value.startswith(prefix):
                out.append(f"{key}: required and must start with {prefix!r}")
        for key in ("stage", "error"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"{key}: required and non-empty")
        if payload.get("rate_measured") is not False:
            out.append("rate_measured: must be false on a pre-event T0 refusal")
    elif kind == KIND_PROPOSAL_SKIPPED:
        for key in ("proposal_ref", "reason"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"{key}: required and non-empty")
    elif kind == KIND_COMPOSITION_REQUESTED:
        for key in ("request_sha256", "combined_candidate_id", "source_tree",
                    "mode"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"{key}: required and non-empty")
        if not isinstance(payload.get("member_candidates"), list) \
                or not payload.get("member_candidates"):
            out.append("member_candidates: required non-empty list")
        if not isinstance(payload.get("anchor"), Mapping):
            out.append("anchor: required mapping")
        if not isinstance(payload.get("evaluator"), Mapping):
            out.append("evaluator: required mapping")
        if not isinstance(payload.get("required_t2_cells"), list) \
                or not payload.get("required_t2_cells"):
            out.append("required_t2_cells: required non-empty list")
    elif kind == KIND_COMPOSITION_FAILED:
        for key in ("request_sha256", "request_event_id", "source_tree",
                    "failure_class", "failure_detail"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"{key}: required and non-empty")
    elif kind == KIND_COMPOSITION_REJECTED:
        for key in ("attempt_sha256", "source_tree", "anchor_sha256",
                    "compatibility_sha256"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"{key}: required and non-empty")
        candidates = payload.get("candidate_ids")
        conflicts = payload.get("conflicts")
        if not isinstance(candidates, list) or not candidates:
            out.append("candidate_ids: required non-empty list")
        if not isinstance(conflicts, list) or not conflicts:
            out.append("conflicts: required non-empty list")
    elif kind in BOOTSTRAP_KNOWLEDGE_KINDS:
        if not payload:
            out.append("payload: must not be empty for a bootstrap-knowledge event")
    else:  # pragma: no cover - guarded by the KINDS membership test in append()
        out.append(f"kind: {kind!r} has no payload validator")
    return out


# =============================================================================
# The journal
# =============================================================================

class Journal:
    """The append-only sharded event journal rooted at `root`.

    Every mutating operation runs under one exclusive `flock`, which is also the
    lock invariant 19 wants the control latch re-read under — `write_lock()`
    exposes it for that purpose without this module owning the latch itself.
    """

    def __init__(
        self,
        root: str,
        *,
        campaign_id: Optional[str] = None,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    ) -> None:
        if not isinstance(root, str) or not root:
            raise ValueError("root must be a non-empty path")
        if not isinstance(max_shard_bytes, int) or isinstance(max_shard_bytes, bool):
            raise TypeError("max_shard_bytes must be an int")
        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be positive")
        self.root = os.path.abspath(root)
        self.campaign_id = campaign_id
        self.max_shard_bytes = max_shard_bytes
        self._archive_dir = os.path.join(self.root, ARCHIVE_DIRNAME)
        self._cursor_dir = os.path.join(self.root, CURSOR_DIRNAME)
        self._lock_path = os.path.join(self.root, LOCK_NAME)
        # flock is per open-file-description, so a second flock from this same
        # process on a NEW fd would block forever. Re-entrancy is therefore
        # tracked explicitly instead of being an accident waiting for the first
        # nested call.
        self._lock_fd: Optional[int] = None
        self._lock_depth = 0

    # ---- layout -----------------------------------------------------------

    def _any_shard_exists(self) -> bool:
        """True when the journal already has a shard, live OR archived.

        Deliberately a raw filename scan rather than `shards()`: this is asked
        BEFORE the journal is known to be well formed, and `shards()` refuses a
        malformed one. The question here is only "has this journal ever been
        written?", and a shard sitting in `archive/` answers yes.
        """
        for directory in (self.root, self._archive_dir):
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                if _SHARD_RE.match(name):
                    return True
        return False

    def initialize(self) -> None:
        """Create the journal root, base shard, and cursor directory.

        Idempotent. Directory entries are fsynced, because a shard whose
        existence is not durable is a shard the next boot does not read.

        The base shard is created only when the journal holds NO shard anywhere.
        Testing `os.path.exists(events.jsonl)` instead — which is what this did
        — is wrong in two directions, and both were reproduced:

          * once `archive_retired_shards()` has retired index 0 into `archive/`,
            re-creating a live `events.jsonl` puts index 0 in two places, and
            `shards()` then refuses the journal FOREVER — every read and every
            append raises. `initialize()` is the routine every process runs at
            startup, so the second process to start bricks the record;
          * if the base shard is LOST, fabricating an empty replacement converts
            the `ShardGapError` that exists precisely to report that loss into a
            journal that reads clean and silently returns only the post-hole
            shards. Deleting the thing the check inspects must not pass the
            check.
        """
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self._cursor_dir, exist_ok=True)
        os.makedirs(self._archive_dir, exist_ok=True)
        base = os.path.join(self.root, BASE_SHARD_NAME)
        if not self._any_shard_exists():
            fd = os.open(base, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(self.root)

    def _shard_name(self, index: int) -> str:
        return BASE_SHARD_NAME if index == 0 else f"events_{index}.jsonl"

    def _shard_path(self, index: int, *, archived: bool = False) -> str:
        directory = self._archive_dir if archived else self.root
        return os.path.join(directory, self._shard_name(index))

    def shards(self) -> list:
        """Every shard, live and archived, in INTEGER index order.

        This one method is where the three historical shard bugs die:

          * it enumerates directories instead of reading only the base file, so a
            rotated journal cannot present its frozen pre-rotation prefix as the
            whole history;
          * it sorts by `int(index)`, so `_10` follows `_9` instead of preceding
            `_2` as it does lexicographically;
          * it never probes `_1, _2, _3 …` until one is missing, and a hole in the
            enumerated sequence raises `ShardGapError` instead of being taken for
            the end of the journal.
        """
        found: dict = {}
        for archived, directory in ((False, self.root), (True, self._archive_dir)):
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                match = _SHARD_RE.match(name)
                if match is None:
                    if _SHARD_LOOKALIKE_RE.match(name):
                        raise JournalCorruption(
                            f"{os.path.join(directory, name)}: shard-like file with a "
                            "non-canonical name cannot be placed in the shard order "
                            "(expected 'events.jsonl' or 'events_<n>.jsonl', n>=1, "
                            "no leading zeros)"
                        )
                    continue
                index = 0 if match.group(1) is None else int(match.group(1))
                if index in found:
                    raise JournalCorruption(
                        f"shard index {index} exists both live and archived "
                        f"({found[index].path} and {os.path.join(directory, name)}); "
                        "one position in the order cannot hold two files"
                    )
                found[index] = ShardRef(
                    index=index,
                    path=os.path.join(directory, name),
                    archived=archived,
                )
        if not found:
            raise JournalCorruption(
                f"{self.root}: no shards found; call initialize() first"
            )
        indices = sorted(found)
        if indices[0] != 0:
            raise ShardGapError(
                f"{self.root}: base shard (index 0) is missing; the journal starts "
                f"at index {indices[0]}"
            )
        missing = [i for i in range(indices[0], indices[-1] + 1) if i not in found]
        if missing:
            raise ShardGapError(
                f"{self.root}: shard index/indices {missing} missing between 0 and "
                f"{indices[-1]}; a hole is data loss, not the end of the journal"
            )
        return [found[i] for i in indices]

    def active_shard_index(self) -> int:
        """Highest LIVE shard index — the one appends land in."""
        live = [s.index for s in self.shards() if not s.archived]
        if not live:
            raise JournalCorruption(
                f"{self.root}: every shard is archived; there is nowhere to append"
            )
        return max(live)

    # ---- locking ----------------------------------------------------------

    @contextmanager
    def write_lock(self):
        """Exclusive journal lock.

        Public because invariant 19 requires the operator-control latch to be
        re-read from disk at the top of each iteration UNDER the write lock. The
        latch itself belongs to the control plane; the lock belongs here.
        """
        if self._lock_depth == 0:
            os.makedirs(self.root, exist_ok=True)
            fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except BaseException:
                os.close(fd)
                raise
            self._lock_fd = fd
        self._lock_depth += 1
        try:
            yield
        finally:
            self._lock_depth -= 1
            if self._lock_depth == 0:
                fd, self._lock_fd = self._lock_fd, None
                if fd is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)

    # ---- append -----------------------------------------------------------

    def append(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        campaign_id: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> JournalEntry:
        """Append one event, fsync it, and return the entry that was written.

        Returning means FSYNCED. Everything below the return is refused rather
        than coerced: an unknown kind, a payload that fails its schema, a
        campaign id that contradicts the payload's own. There is no
        "best-effort" append, because a best-effort primary record is not a
        record.
        """
        if kind not in KINDS:
            raise ValueError(
                f"unknown event kind {kind!r}; the vocabulary is closed: "
                f"{sorted(KINDS)}"
            )
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"payload must be a mapping, got {type(payload).__name__}"
            )
        payload = dict(payload)

        schema_name = SCHEMA_BOUND_KINDS.get(kind)
        if schema_name is not None:
            accepted = ACCEPTED_SCHEMAS_BY_KIND[kind]
            if payload.get("schema") not in accepted:
                raise ValueError(
                    f"{kind} payload declares schema {payload.get('schema')!r}; "
                    f"expected one of {sorted(accepted)}"
                )
            violations = schemas.validate_record(payload)
            if violations:
                raise ValueError(
                    f"{kind} payload is not a valid {schema_name}: "
                    + "; ".join(violations)
                )
        else:
            violations = _validate_native_payload(kind, payload)
            if violations:
                raise ValueError(
                    f"{kind} payload is invalid: " + "; ".join(violations)
                )

        effective_campaign = campaign_id if campaign_id is not None else self.campaign_id
        payload_campaign = payload.get("campaign_id")
        if isinstance(payload_campaign, str):
            if effective_campaign is None:
                effective_campaign = payload_campaign
            elif effective_campaign != payload_campaign:
                raise ValueError(
                    f"envelope campaign_id {effective_campaign!r} contradicts the "
                    f"payload's own {payload_campaign!r}"
                )

        if record_id is None:
            key = RECORD_ID_KEY_BY_KIND.get(kind)
            if key is not None:
                value = payload.get(key)
                if isinstance(value, str) and value:
                    record_id = value
        if record_id is not None and not isinstance(record_id, str):
            raise TypeError("record_id must be a string or None")

        if "narrative" in payload and schema_name is None:
            # Journal-native payloads carry structured facts. Prose belongs in a
            # §7 record's own marked `narrative` field, where schemas.py forces
            # `narrative_retrievable: false`; smuggling it into a native payload
            # would bypass that marking entirely.
            raise ValueError(
                f"{kind} payload carries a 'narrative' field; planner prose "
                "belongs in a schema-bound record that marks it non-retrievable "
                "(§5.5 item 6)"
            )

        # Supersession is the one append that pays for a full read: a dangling
        # target is a permanent, unfixable dangling reference in an append-only
        # log, and these events are rare enough that O(n) is the right price.
        with self.write_lock():
            if kind in (KIND_SUPERSEDED, KIND_RETRIEVAL_SUPERSEDED):
                self._assert_supersession_target_exists(payload["target_event_id"])
            return self._append_locked(kind, payload, effective_campaign, record_id)

    def _assert_supersession_target_exists(self, target_event_id: str) -> None:
        existing = {e.event_id for e in self.read_all()}
        if target_event_id not in existing:
            raise SupersessionError(
                f"supersession target {target_event_id!r} does not exist in this "
                "journal; an append-only log cannot repair a dangling reference"
            )

    def _append_locked(
        self,
        kind: str,
        payload: Mapping[str, Any],
        campaign_id: Optional[str],
        record_id: Optional[str],
    ) -> JournalEntry:
        self._repair_torn_tail_locked()
        return self._write_entry_locked(kind, payload, campaign_id, record_id)

    def _write_entry_locked(
        self,
        kind: str,
        payload: Mapping[str, Any],
        campaign_id: Optional[str],
        record_id: Optional[str],
    ) -> JournalEntry:
        seq = self._next_seq_locked()
        written_at = _iso_now()
        # The id embeds the write-lock-serialised seq, so uniqueness is a
        # property of the lock rather than of a scan that could race.
        digest = schemas.content_hash(payload)[:12]
        event_id = f"akj-{seq:012d}-{digest}"
        entry = JournalEntry(
            event_id=event_id,
            seq=seq,
            kind=kind,
            campaign_id=campaign_id,
            record_id=record_id,
            written_at=written_at,
            payload=payload,
        )
        line = schemas.canonical_bytes(entry.envelope()) + b"\n"
        if b"\n" in line[:-1]:  # pragma: no cover - json escapes control chars
            raise JournalCorruption(
                "serialised entry contains an embedded newline; it would read "
                "back as two records"
            )

        index = self.active_shard_index()
        path = self._shard_path(index)
        size = os.path.getsize(path)
        if size and size + len(line) > self.max_shard_bytes:
            index = self._rotate_locked(index)
            path = self._shard_path(index)

        is_new = not os.path.exists(path)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = os.write(fd, line)
            if written != len(line):  # pragma: no cover - short append on O_APPEND
                raise JournalCorruption(
                    f"short append: wrote {written} of {len(line)} bytes"
                )
            os.fsync(fd)
        finally:
            os.close(fd)
        if is_new:
            _fsync_dir(self.root)
        return JournalEntry(
            event_id=entry.event_id,
            seq=entry.seq,
            kind=entry.kind,
            campaign_id=entry.campaign_id,
            record_id=entry.record_id,
            written_at=entry.written_at,
            payload=entry.payload,
            shard_index=index,
            line_number=-1,
        )

    def _rotate_locked(self, current_index: int) -> int:
        """Start a new shard. Always safe — appending to a fresh file loses
        nothing. RETIRING a shard is the operation gated on cursors; see
        `archive_retired_shards()`."""
        new_index = current_index + 1
        path = self._shard_path(new_index)
        if os.path.exists(path):
            raise JournalCorruption(
                f"{path}: rotation target already exists"
            )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.root)
        return new_index

    def rotate(self) -> int:
        """Force a shard rollover; returns the new active index.

        Repairs an unacknowledged torn tail FIRST, in the same order
        `_append_locked()` uses. Skipping that step is not a missing nicety, it
        is fatal: rolling over on top of a torn tail leaves the fragment in a
        shard that is no longer final, and `torn_tail()` then — correctly —
        calls that corruption. Every subsequent `torn_tail()`, `scan()`,
        `read_all()` and `append()` raises, with no recovery path. A crash
        followed by one `rotate()` would otherwise destroy the whole journal.
        """
        with self.write_lock():
            self._repair_torn_tail_locked()
            return self._rotate_locked(self.active_shard_index())

    def _next_seq_locked(self) -> int:
        """Next sequence number, read from the journal itself.

        Derived from the last complete line of the highest non-empty shard rather
        than from a counter file: a counter is a second source of truth that can
        disagree with the record, and the record is the thing that survives.
        """
        for shard in reversed(self.shards()):
            last = _last_complete_line(shard.path)
            if last is None:
                continue
            try:
                envelope = json.loads(last.decode("utf-8"))
                seq = envelope["seq"]
            except Exception as exc:
                raise JournalCorruption(
                    f"{shard.path}: last complete line is unreadable, so the next "
                    f"sequence number cannot be derived: {exc}"
                ) from exc
            if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
                raise JournalCorruption(
                    f"{shard.path}: last entry carries a non-sequence seq {seq!r}"
                )
            return seq + 1
        return 1

    def _repair_torn_tail_locked(self) -> Optional[JournalEntry]:
        """Truncate an unacknowledged partial line and record that it happened.

        A crash between `os.write` and its completion leaves bytes with no
        terminating newline. Appending on top of them would concatenate the next
        event onto the fragment and destroy BOTH. The fragment was never
        acknowledged, so removing it loses nothing — but it is removed loudly: a
        `TORN_APPEND_DISCARDED` event carries the byte count, the fragment's
        sha256, and a bounded hex prefix, which keeps invariant 7 honest.
        """
        torn = self.torn_tail()
        if torn is None:
            return None
        path = self._shard_path(torn.shard_index)
        size = os.path.getsize(path)
        fd = os.open(path, os.O_WRONLY)
        try:
            os.ftruncate(fd, size - torn.byte_count)
            os.fsync(fd)
        finally:
            os.close(fd)
        return self._write_entry_locked(
            KIND_TORN_APPEND_DISCARDED,
            {
                "shard_index": torn.shard_index,
                "discarded_byte_count": torn.byte_count,
                "discarded_sha256": torn.sha256,
                "discarded_prefix_hex": torn.prefix_hex,
                "detail": "unacknowledged partial append discarded before writing "
                          "the next event",
            },
            self.campaign_id,
            None,
        )

    def torn_tail(self) -> Optional[TornTail]:
        """The unacknowledged trailing fragment, if a crash left one."""
        shards = self.shards()
        last = shards[-1]
        if last.archived:
            # An archived shard must be complete; a torn tail there is corruption
            # of the record, not an in-flight append.
            fragment = _trailing_fragment(last.path)
            if fragment:
                raise JournalCorruption(
                    f"{last.path}: archived shard ends with an incomplete line"
                )
            return None
        for shard in shards[:-1]:
            fragment = _trailing_fragment(shard.path)
            if fragment:
                raise JournalCorruption(
                    f"{shard.path}: a non-final shard ends with an incomplete line; "
                    "only the last shard can hold an unacknowledged append"
                )
        fragment = _trailing_fragment(last.path)
        if not fragment:
            return None
        return TornTail(
            shard_index=last.index,
            byte_count=len(fragment),
            sha256=hashlib.sha256(fragment).hexdigest(),
            prefix_hex=fragment[:256].hex(),
        )

    # ---- convenience appends ---------------------------------------------

    def append_superseded(
        self, target_event_id: str, reason: str, *, superseded_by: Optional[str] = None
    ) -> JournalEntry:
        """Withdraw a record: it was REPLACED. Removed from derived views and
        from retrieval; never removed from the record (invariant 8)."""
        return self.append(KIND_SUPERSEDED, {
            "target_event_id": target_event_id,
            "reason": reason,
            "superseded_by": superseded_by,
        })

    def append_retrieval_superseded(
        self, target_event_id: str, reason: str, receipt: str
    ) -> JournalEntry:
        """Stop BELIEVING a record without replacing it (§5.5 item 7).

        The difference from `append_superseded` is the whole point of the event:
        the record stays in every derived view and in `read_all()`, and only
        `retrieve()` withholds it. That is how an append-only log gains the
        ability to stop believing something without gaining the ability to
        forget it.
        """
        return self.append(KIND_RETRIEVAL_SUPERSEDED, {
            "target_event_id": target_event_id,
            "reason": reason,
            "receipt": receipt,
        })

    def append_tombstone(
        self,
        *,
        artifact_sha256: str,
        storage_class: str,
        size_bytes: int,
        reason: str,
        path: str,
        tombstone_id: Optional[str] = None,
    ) -> JournalEntry:
        """Record a storage expiry (§5.8). The storage plane deletes the bytes;
        this event is the receipt, and only the `expirable` class may have one.

        `path` is REQUIRED (it was optional): the receipt view is keyed by
        (hash, path), so a tombstone without a path merges with any other
        reclamation of byte-identical bytes. `tombstone_id` is optional here and
        always supplied by `storage.expire_artifact`; when present it gives the
        consistency checker a second, finer identity to recount against.
        """
        payload = {
            "artifact_sha256": artifact_sha256,
            "storage_class": storage_class,
            "size_bytes": size_bytes,
            "reason": reason,
            "path": path,
        }
        if tombstone_id is not None:
            payload["tombstone_id"] = tombstone_id
        return self.append(KIND_TOMBSTONE, payload)

    def append_preflight_attestation(
        self, result: Any, *, campaign_id: Optional[str] = None
    ) -> JournalEntry:
        """Journal a §3.5 preflight verdict — PASS, FAIL, or COULD_NOT_CHECK.

        Accepts a `resource.preflight.PreflightResult` (anything exposing
        `to_dict()`) or the mapping it produces. Duck-typed on purpose: the
        journal must not import the resource plane for one attribute read, and
        the resource plane must not import the journal to be recordable.

        This exists because `require_no_concurrent_inference`'s contract could
        not be honoured without it — it hands the attestation back on the
        exception "so the caller can journal it", and the closed kind vocabulary
        had no kind to journal it AS. A precondition that was checked but not
        recorded is indistinguishable from one that was skipped, and FAIL /
        COULD_NOT_CHECK are exactly the outcomes invariant 7 is about.
        """
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
        elif isinstance(result, Mapping):
            payload = dict(result)
        else:
            raise TypeError(
                "result must be a PreflightResult (or anything with to_dict()) "
                f"or a mapping, got {type(result).__name__}"
            )
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"result.to_dict() returned {type(payload).__name__}, not a mapping"
            )
        return self.append(KIND_PREFLIGHT_ATTESTATION, payload,
                           campaign_id=campaign_id)

    def append_control_ack(
        self, *, control: str, control_id: str, received_at: str, disposition: str
    ) -> JournalEntry:
        """Acknowledge an operator control in the journal (invariant 19).

        The disk latch and the drain are the control plane's job; this records
        that the command was SEEN, which is what makes an ignored control a hard
        failure instead of a silent one.
        """
        return self.append(KIND_OPERATOR_CONTROL_ACK, {
            "control": control,
            "control_id": control_id,
            "received_at": received_at,
            "disposition": disposition,
        })

    # ---- read -------------------------------------------------------------

    def scan(self, *, validate_payloads: bool = False) -> ReadReport:
        """Read every shard and report entries, defects, and any torn tail.

        Never skips a bad line quietly: a line that will not parse becomes a
        `JournalDefect` the caller can see. `read_all()` turns any defect into an
        exception; `scan()` exists so a repair tool can look at the damage.
        """
        entries: list = []
        defects: list = []
        shards = self.shards()
        torn = self.torn_tail()
        for shard in shards:
            with open(shard.path, "rb") as fh:
                data = fh.read()
            if torn is not None and shard.index == torn.shard_index and torn.byte_count:
                # Trim to the last newline in the bytes ACTUALLY READ, not by the
                # byte count `torn_tail()` measured in an earlier stat pass.
                # Reads are lock-free, so a writer can repair that fragment and
                # append a real event between the two passes; subtracting the
                # stale count then chops the tail off an acknowledged, fsynced
                # record — and when the count happens to equal a whole line, it
                # drops that event with no defect and no exception, which is a
                # silent loss dressed as a successful read.
                cut = data.rfind(b"\n")
                data = data[: cut + 1] if cut != -1 else b""
            if not data:
                continue
            # Split on bytes. `str.splitlines()` also breaks on U+2028/U+2029,
            # which canonical JSON (ensure_ascii=False) may legitimately contain
            # inside a string value — one record would read back as two.
            raw_lines = data.split(b"\n")
            if raw_lines and raw_lines[-1] == b"":
                raw_lines.pop()
            for line_number, raw in enumerate(raw_lines, start=1):
                if not raw.strip():
                    defects.append(JournalDefect(
                        shard.index, line_number, "blank line inside the journal"
                    ))
                    continue
                entry, defect = _parse_line(raw, shard.index, line_number)
                if defect is not None:
                    defects.append(defect)
                    continue
                if validate_payloads:
                    schema_name = SCHEMA_BOUND_KINDS.get(entry.kind)
                    if schema_name is not None:
                        violations = schemas.validate_record(entry.payload)
                    else:
                        violations = _validate_native_payload(entry.kind, entry.payload)
                    if violations:
                        defects.append(JournalDefect(
                            shard.index, line_number, "payload fails validation",
                            "; ".join(violations),
                        ))
                        continue
                entries.append(entry)

        seen: dict = {}
        for entry in entries:
            if entry.event_id in seen:
                defects.append(JournalDefect(
                    entry.shard_index, entry.line_number, "duplicate event_id",
                    f"{entry.event_id} also at shard {seen[entry.event_id][0]} line "
                    f"{seen[entry.event_id][1]}",
                ))
            else:
                seen[entry.event_id] = (entry.shard_index, entry.line_number)

        entries.sort(key=lambda e: e.seq)
        return ReadReport(tuple(entries), tuple(defects), torn)

    def read_all(self) -> list:
        """The RECORD API: every event from every shard, in seq order.

        Includes retrieval-superseded events and every `narrative` field — that
        is what makes it the record. Raises on any defect rather than returning a
        partial history that would look exactly like a complete one.
        """
        report = self.scan()
        if report.defects:
            raise JournalCorruption(
                f"{self.root}: {len(report.defects)} unreadable line(s): "
                + "; ".join(str(d) for d in report.defects[:10])
            )
        return list(report.entries)

    def retrieve(
        self,
        *,
        kinds: Optional[Iterable[str]] = None,
        cite_event_ids: Iterable[str] = (),
    ) -> list:
        """The RETRIEVAL API: what the planner is allowed to read back.

        Differs from `read_all()` in exactly two ways, both required by §5.5:
        superseded and retrieval-superseded events are withheld, and `narrative`
        is stripped at every depth unless the caller cites the event id.
        """
        entries = self.read_all()
        rows = retrieval_filter(
            entries, supersession_basis=entries, cite_event_ids=cite_event_ids
        )
        if kinds is not None:
            wanted = set(kinds)
            unknown = sorted(wanted - KINDS)
            if unknown:
                raise ValueError(f"unknown kind(s) in filter: {unknown}")
            rows = [r for r in rows if r["kind"] in wanted]
        return rows

    # ---- cursors and archiving -------------------------------------------

    def _cursor_path(self, reader_id: str) -> str:
        if not isinstance(reader_id, str) or not _READER_ID_RE.match(reader_id):
            raise CursorError(
                f"invalid reader id {reader_id!r}; must match {_READER_ID_RE.pattern} "
                "(no path separators — a cursor never addresses a file outside the "
                "cursor directory)"
            )
        return os.path.join(self._cursor_dir, f"{reader_id}.json")

    def register_reader(self, reader_id: str) -> Cursor:
        """Register a reader so archiving can be gated on its progress.

        Registration is mandatory rather than implicit: an unregistered reader is
        invisible to `archive_retired_shards()`, and "invisible" is precisely the
        reader whose shard would be retired out from under it.
        """
        path = self._cursor_path(reader_id)
        with self.write_lock():
            existing = self.cursor(reader_id)
            if existing is not None:
                return existing
            cursor = Cursor(reader_id=reader_id, last_seq=0, updated_at=_iso_now())
            _atomic_write_json(path, {
                "reader_id": cursor.reader_id,
                "last_seq": cursor.last_seq,
                "updated_at": cursor.updated_at,
            })
        return cursor

    def cursor(self, reader_id: str) -> Optional[Cursor]:
        path = self._cursor_path(reader_id)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            data = json.loads(raw.decode("utf-8"))
            return Cursor(
                reader_id=data["reader_id"],
                last_seq=int(data["last_seq"]),
                updated_at=str(data["updated_at"]),
            )
        except Exception as exc:
            # An unreadable cursor must not read as "at position 0" and must not
            # read as "fully caught up": either default silently corrupts the
            # archive decision.
            raise CursorError(f"{path}: cursor is unreadable: {exc}") from exc

    def cursors(self) -> dict:
        if not os.path.isdir(self._cursor_dir):
            return {}
        out: dict = {}
        for name in sorted(os.listdir(self._cursor_dir)):
            if not name.endswith(".json"):
                continue
            reader_id = name[: -len(".json")]
            cursor = self.cursor(reader_id)
            if cursor is not None:
                out[reader_id] = cursor
        return out

    def commit_cursor(
        self, reader_id: str, last_seq: int, *, allow_rewind: bool = False
    ) -> Cursor:
        """Advance a registered reader's cursor. Refuses to move backwards.

        A rewound cursor re-opens shards that were legitimately archivable and
        makes "past all cursors" a moving target, so a rewind is an explicit,
        named act (`allow_rewind=True`) rather than whatever the caller last
        computed.
        """
        if not isinstance(last_seq, int) or isinstance(last_seq, bool) or last_seq < 0:
            raise ValueError("last_seq must be a non-negative integer")
        path = self._cursor_path(reader_id)
        with self.write_lock():
            existing = self.cursor(reader_id)
            if existing is None:
                raise CursorError(
                    f"reader {reader_id!r} is not registered; call "
                    "register_reader() first"
                )
            if last_seq < existing.last_seq and not allow_rewind:
                raise CursorError(
                    f"reader {reader_id!r} cursor would rewind from "
                    f"{existing.last_seq} to {last_seq}; pass allow_rewind=True to "
                    "mean it"
                )
            cursor = Cursor(reader_id, last_seq, _iso_now())
            _atomic_write_json(path, {
                "reader_id": cursor.reader_id,
                "last_seq": cursor.last_seq,
                "updated_at": cursor.updated_at,
            })
        return cursor

    def read_since(self, reader_id: str) -> list:
        """Events a reader has not consumed, from ALL shards.

        Deliberately not "resume from the cursor's shard": the cursor stores a
        seq, the reader re-enumerates every shard, and the filter is on seq. That
        is why a rotation between two calls cannot hide anything — there is no
        position on disk for the reader to be stuck at.
        """
        cursor = self.cursor(reader_id)
        if cursor is None:
            raise CursorError(
                f"reader {reader_id!r} is not registered; call register_reader() "
                "first"
            )
        return [e for e in self.read_all() if e.seq > cursor.last_seq]

    def archive_retired_shards(self) -> list:
        """Move fully-consumed shards into `archive/`. Only past ALL cursors.

        `BUS_PROTOCOL.md` rule 4 and §5.5 item 1: rotation happens only past all
        cursors. With no registered reader there is no evidence that anyone has
        passed anything, so this refuses — "no cursors" is not "all cursors have
        passed", and treating it as such is how a reader loses its unread tail.

        Archived shards remain part of the record: `shards()` enumerates the
        archive directory too, so `read_all()` still returns their events.
        """
        with self.write_lock():
            cursors = self.cursors()
            if not cursors:
                raise CursorError(
                    f"{self.root}: no readers are registered, so no shard can be "
                    "shown to be past all cursors; refusing to archive"
                )
            watermark = min(c.last_seq for c in cursors.values())
            active = self.active_shard_index()
            report = self.scan()
            if report.defects:
                raise JournalCorruption(
                    "refusing to archive a journal with unreadable lines: "
                    + "; ".join(str(d) for d in report.defects[:10])
                )
            max_seq_by_shard: dict = {}
            for entry in report.entries:
                prior = max_seq_by_shard.get(entry.shard_index, 0)
                if entry.seq > prior:
                    max_seq_by_shard[entry.shard_index] = entry.seq

            archived: list = []
            for shard in self.shards():
                if shard.archived or shard.index == active:
                    continue
                shard_max = max_seq_by_shard.get(shard.index)
                if shard_max is None:
                    # An empty non-active shard holds nothing to lose.
                    pass
                elif shard_max > watermark:
                    continue
                target = self._shard_path(shard.index, archived=True)
                os.makedirs(self._archive_dir, exist_ok=True)
                os.replace(shard.path, target)
                archived.append(shard.index)
            if archived:
                _fsync_dir(self.root)
                _fsync_dir(self._archive_dir)
            return archived

    # ---- bootstrap --------------------------------------------------------

    def bootstrap_views(
        self, *, deliberate_rebase: bool = False, rebase_reason: Optional[str] = None
    ) -> Views:
        """BOOTSTRAP §8.2 step 10: rebuild the views and assert against them.

        Raises unless the views describe the journal. With `deliberate_rebase`
        the disagreement is journaled as a `VIEW_REBASED` event and then allowed,
        so the record shows an operator meant it — which is exactly what was
        missing when a restart came up with an empty frontier and nothing
        objected.
        """
        events = self.read_all()
        views = rebuild_views(events)
        check = assert_views_consistent(
            events, views,
            deliberate_rebase=deliberate_rebase, rebase_reason=rebase_reason,
        )
        if check.outcome != schemas.PASS:
            self.append(KIND_VIEW_REBASED, {
                "rebase_reason": rebase_reason,
                "suppressed_reasons": list(check.reasons),
                "entry_count": len(events),
            })
        return views


# =============================================================================
# Byte-level helpers
# =============================================================================

def _parse_line(raw: bytes, shard_index: int, line_number: int):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, JournalDefect(shard_index, line_number, "not valid UTF-8", str(exc))
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, JournalDefect(shard_index, line_number, "not valid JSON", str(exc))
    if not isinstance(envelope, dict):
        return None, JournalDefect(
            shard_index, line_number, "line is not a JSON object",
            type(envelope).__name__,
        )
    if envelope.get("journal_schema") != JOURNAL_ENTRY_SCHEMA:
        return None, JournalDefect(
            shard_index, line_number, "unknown journal_schema",
            repr(envelope.get("journal_schema")),
        )
    event_id = envelope.get("event_id")
    seq = envelope.get("seq")
    kind = envelope.get("kind")
    payload = envelope.get("payload")
    written_at = envelope.get("written_at")
    problems = []
    if not isinstance(event_id, str) or not event_id:
        problems.append("event_id")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        problems.append("seq")
    if kind not in KINDS:
        problems.append("kind")
    if not isinstance(payload, dict):
        problems.append("payload")
    if not isinstance(written_at, str) or not written_at:
        problems.append("written_at")
    campaign_id = envelope.get("campaign_id")
    if campaign_id is not None and not isinstance(campaign_id, str):
        problems.append("campaign_id")
    record_id = envelope.get("record_id")
    if record_id is not None and not isinstance(record_id, str):
        problems.append("record_id")
    if problems:
        return None, JournalDefect(
            shard_index, line_number, "malformed envelope field(s)",
            ",".join(problems),
        )
    return JournalEntry(
        event_id=event_id,
        seq=seq,
        kind=kind,
        campaign_id=campaign_id,
        record_id=record_id,
        written_at=written_at,
        payload=payload,
        shard_index=shard_index,
        line_number=line_number,
    ), None


def _trailing_fragment(path: str) -> bytes:
    """Bytes after the last newline — an append that never completed.

    Widens its read window rather than giving up, for the same reason
    `_last_complete_line()` does. The earlier version read one 1 MiB window and
    raised "the file is not line-delimited JSON" when it found no newline in it,
    which is the wrong diagnosis for the case that actually produces it: a crash
    partway through a single event larger than 1 MiB. There is no payload size
    cap, so that event is legal — and raising there made `torn_tail()`,
    `read_all()` and `append()` all raise permanently, bricking the journal at
    exactly the moment the torn-tail repair exists to rescue it.
    """
    size = os.path.getsize(path)
    if size == 0:
        return b""
    window = 1 << 20
    while True:
        window = min(window, size)
        with open(path, "rb") as fh:
            fh.seek(size - window)
            tail = fh.read(window)
        index = tail.rfind(b"\n")
        if index != -1:
            return tail[index + 1:]
        if window == size:
            # No newline anywhere: the whole file is one unterminated fragment.
            return tail
        window *= 4


def _last_complete_line(path: str) -> Optional[bytes]:
    """The last newline-terminated line of a shard, or None if it has none.

    Reads a growing tail window rather than the whole file: this runs on every
    append, and slurping a 64 MiB shard per event would make fsync-per-event look
    cheap by comparison.
    """
    size = os.path.getsize(path)
    if size == 0:
        return None
    window = 1 << 16
    while True:
        window = min(window, size)
        with open(path, "rb") as fh:
            fh.seek(size - window)
            tail = fh.read(window)
        end = tail.rfind(b"\n")
        if end == -1:
            if window == size:
                return None
            window *= 4
            continue
        start = tail.rfind(b"\n", 0, end)
        if start == -1 and window < size:
            # The line may extend before the window; widen rather than return a
            # truncated line whose `seq` would then be unparseable.
            window *= 4
            continue
        line = tail[start + 1: end]
        if not line.strip():
            # A blank last line would make this function report "no entries in
            # this shard", and the next append would then REUSE a sequence
            # number — a duplicate seq silently reorders the whole journal.
            raise JournalCorruption(
                f"{path}: the last complete line is blank; the next sequence "
                "number cannot be derived from it"
            )
        return line


def _fsync_dir(path: str) -> None:
    """Make a directory entry durable — a shard whose name is not on disk after
    a crash is a shard the next boot silently does not read."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: str, obj: Mapping[str, Any]) -> None:
    """Write-temp + fsync + rename. A half-written cursor would otherwise read
    as an unparseable file and stop every archive decision."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    data = schemas.canonical_bytes(obj) + b"\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_dir(directory)


__all__ = [
    "JOURNAL_ENTRY_SCHEMA", "BASE_SHARD_NAME", "ARCHIVE_DIRNAME", "CURSOR_DIRNAME",
    "DEFAULT_MAX_SHARD_BYTES", "KINDS", "NATIVE_KINDS", "SCHEMA_BOUND_KINDS",
    "ACCEPTED_SCHEMAS_BY_KIND",
    "BOOTSTRAP_KNOWLEDGE_KINDS", "RECORD_ID_KEY_BY_KIND", "STORAGE_CLASSES",
    "TOMBSTONABLE_STORAGE_CLASSES", "NARRATIVE_KEYS",
    "KIND_CAMPAIGN_OPENED", "KIND_PROPOSAL_RECORDED", "KIND_CANDIDATE_RECORDED",
    "KIND_EVALUATION_EVENT", "KIND_CHAMPION_UPDATED",
    "KIND_RELEASE_PACKAGE_PREPARED", "KIND_OPERATOR_WAIVER_RECORDED",
    "KIND_SUPERSEDED", "KIND_RETRIEVAL_SUPERSEDED", "KIND_TOMBSTONE",
    "KIND_TORN_APPEND_DISCARDED", "KIND_OPERATOR_CONTROL_ACK", "KIND_VIEW_REBASED",
    "KIND_PROPOSAL_SKIPPED", "KIND_STOP_STATE", "KIND_PREFLIGHT_ATTESTATION",
    "KIND_MICROBENCH_RUN_COMPLETED", "KIND_T0_REFUSAL",
    "KIND_COMPOSITION_REQUESTED",
    "KIND_COMPOSITION_FAILED", "KIND_COMPOSITION_REJECTED",
    "tombstone_view_key",
    "Journal", "JournalEntry", "JournalDefect", "ShardRef", "TornTail",
    "ReadReport", "Cursor", "Views",
    "JournalError", "JournalCorruption", "ShardGapError", "ViewConsistencyError",
    "SupersessionError", "RetrievalCitationError", "CursorError",
    "rebuild_views", "check_view_consistency", "assert_views_consistent",
    "events_digest", "retrieval_filter", "strip_narrative",
]
