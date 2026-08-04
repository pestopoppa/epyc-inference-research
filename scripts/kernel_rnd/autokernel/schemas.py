#!/usr/bin/env python3
"""schemas.py — versioned data contracts for the AutoKernel research loop (AK1, §7).

WHY THIS MODULE EXISTS
----------------------
Every other AutoKernel component — journal writer, trusted evaluator, planner,
critic, champion view, release packager — reads and writes the same records. The
predecessor loop (orchestration AutoPilot) had no such contract, and the project
paid for it three separate ways, each of which this module is shaped to prevent:

  * **An unlabelled number became a release claim.** `MEASUREMENT.md:13` defines a
    claim as `(metric, protocol-id, n/reps, date, host-attestation ref)` and
    `:85-95` requires `category=OPTIMUM|BASELINE|CANDIDATE` — *"an unlabelled
    measurement is not decision-grade"*. Here `claim_grammar` is a REQUIRED block
    of `evaluation_event`, so a record physically cannot exist without it (§3.4).

  * **A ratio outlived its denominator.** The existing `kernel_eval.sh` scaffold
    made the baseline optional, so a "coherent" run with no anchor passed (§12,
    §2.2). Here `anchor` (source_commit, binary_sha256, linkage_sha256,
    measurement_event_ids) is REQUIRED; a no-anchor comparison is `invalid`,
    never correct. §8.9's `ANCHOR_MOVED` rests entirely on that binding being
    recorded per event. The single exemption — a run VOIDED for a missing anchor,
    which the protocol requires to be journaled as `INVALID` rather than
    discarded — omits the block STRUCTURALLY and may never fabricate a digest to
    fill it (`_check_anchor_block_v3`).

  * **The planner re-consumed its own prose as fact.** AutoPilot re-read planner
    free text out of its primary journal, regenerated a false story, and ran 81
    further trials on it after the code fix landed (§5.5 item 6, invariant 20).
    Here `narrative` is a separate field that every record must mark
    `narrative_retrievable: false`, and `retrievable_view()` strips it, so the
    retrieval layer cannot leak prose by forgetting to.

Two further design rules are enforced structurally rather than documented:

  * **A campaign carries no freeze or cutover authority flag** (§1.3): a kernel
    freeze crosses four human-only trust boundaries, so there is no such
    authority to delegate. `find_authority_flavoured_keys()` REJECTS any
    auth-flavoured key anywhere in a machine-authored record, so the absence is
    checked, not merely intended.

  * **A full-machine threshold can never be applied to a partial-machine cell**
    (§7.4, §12): every `evaluation_event` declares its `scope_denominator`, and
    `check_scope_denominator_admits_gate()` refuses a scope mismatch instead of
    demoting it.

CONVENTIONS
-----------
* The `schema` string carries the version and is part of the record's IDENTITY.
  A v2 record is not a v1 record with extra keys; readers dispatch on the exact
  string via `SCHEMA_REGISTRY`.
* `validate_<name>(obj) -> list[str]` NEVER raises for a validation failure. It
  returns human-readable violations so the caller can journal them (a rejected
  proposal is itself durable evidence — invariant 7). An empty list means valid.
* Serialisation helpers DO raise: `canonical_json()` refuses NaN/Infinity and
  non-string dict keys rather than emitting a silently ambiguous encoding, and
  `retrievable_view()` refuses an unknown schema rather than passing an unknown
  record through with its prose intact. Explicit failure over silent fallback.
* Checkers that need context the record does not carry return a third outcome,
  `COULD_NOT_CHECK`. Inability to evaluate is never reported as PASS and never as
  FAIL.

This module performs NO I/O, launches NO process, and runs NO inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatchcase
from typing import Any, Callable, Iterable, Mapping, Optional

# =============================================================================
# Schema identity — the version is part of the name, never metadata beside it
# =============================================================================

SCHEMA_CAMPAIGN = "epyc.autokernel.campaign.v2"
SCHEMA_PROPOSAL = "epyc.autokernel.proposal.v2"
SCHEMA_CANDIDATE = "epyc.autokernel.candidate.v1"

# The evaluation event exists in two versions AT THE SAME TIME, and both names
# are explicit. v2 records already exist in journals and are read under
# `validate_evaluation_event_v2` forever; nothing rewrites them, and no v2 record
# is upgraded in place — the schema string is the record's identity (§7 and the
# CONVENTIONS note above), so re-labelling one would be a forgery, not a migration.
#
# v3 exists because v2 could not express two things the ratified protocol
# requires (`measurement/protocols/kernel-research.md`):
#
#   * **Precondition 4 names the anchor by THREE components** — *"names its anchor
#     by source commit, binary SHA-256, and linkage SHA-256"*. `v2.anchor` carried
#     only the two digests, so the source commit travelled as an unvalidated extra
#     key inside a free-form block. In v3 `anchor.source_commit` is a REQUIRED,
#     commit-shaped field, validated by the same `_need_commit` helper that
#     `candidate.worktree.source_commit` uses.
#   * **A voided run must be journalable** — *"A voided run is journaled as
#     `INVALID` with its reason, and is never silently discarded."* v2 required
#     `anchor.binary_sha256`/`anchor.linkage_sha256` unconditionally, so the
#     ANCHOR-MISSING void — the one case where there is no digest to record — could
#     not produce a valid record at all, and `journal.Journal.append` therefore
#     refused the very record the protocol requires to exist. v3 permits `anchor`
#     to be **structurally absent**, and only for that case (see
#     `_check_anchor_block_v3`). It does NOT accept a placeholder digest in its
#     place: a fabricated hash is indistinguishable from a measured one to every
#     downstream reader, which is strictly worse than an absent block.
SCHEMA_EVALUATION_EVENT_V2 = "epyc.autokernel.evaluation_event.v2"
SCHEMA_EVALUATION_EVENT_V3 = "epyc.autokernel.evaluation_event.v3"
#: The CURRENT evaluation-event contract. New records are emitted under this one.
SCHEMA_EVALUATION_EVENT = SCHEMA_EVALUATION_EVENT_V3

SCHEMA_CHAMPION = "epyc.autokernel.champion.v1"
SCHEMA_RELEASE_PACKAGE = "epyc.autokernel.release_package.v1"
SCHEMA_OPERATOR_WAIVER = "epyc.autokernel.operator_waiver.v1"

# The `/kernel` operator-surface contract, in two versions AT THE SAME TIME for
# the same reason the evaluation event is: a consumer will meet both. v1 is the
# LEGACY, unlabelled shape the hub reads today; v2 is what AK6 emits. The rules
# and the reasoning live beside the validators further down.
SCHEMA_KERNEL_DASHBOARD_V1 = "epyc.autokernel.kernel_dashboard.v1"
SCHEMA_KERNEL_DASHBOARD_V2 = "epyc.autokernel.kernel_dashboard.v2"
#: The CURRENT operator-surface contract. New exports are emitted under this one.
SCHEMA_KERNEL_DASHBOARD = SCHEMA_KERNEL_DASHBOARD_V2


# =============================================================================
# Controlled vocabularies (§1.5, §7, §9.5, §10.4)
# =============================================================================

BACKENDS = frozenset({
    "llama_cpu", "llama_gpu", "whisper_stt", "qwentts_tts", "serving_runtime",
})

# Four production binaries, three source trees (§1.5). `serving_runtime` is
# deliberately absent: its worktree ownership is a mapping, not a fixed tree
# (§13.5), and its release path is the three-gate stack-change path (§11.6).
SOURCE_TREE_BY_BACKEND = {
    "llama_cpu": "llama.cpp",
    "llama_gpu": "llama.cpp",
    "whisper_stt": "whisper.cpp",
    "qwentts_tts": "qwentts.cpp",
}
SOURCE_TREES = frozenset(SOURCE_TREE_BY_BACKEND.values())

# §1.6 names exactly one objective rule. Adding another is a schema-version
# event, not a validator relaxation.
OBJECTIVE_RULES = frozenset({"per_phase_non_inferiority_plus_improvement"})

# §1.6 enumerates the llama phases. The speech backends' phase vocabulary and
# protocols are explicitly "to be defined" (§13.3, §13.4), so their phase names
# are only checked for being non-empty strings.
LLAMA_PHASES = frozenset({"prefill", "decode"})
PHASES_BY_BACKEND = {"llama_cpu": LLAMA_PHASES, "llama_gpu": LLAMA_PHASES}

# Invariant 15: baseline/off-recipe cells are diagnostic and never justify a
# release, so a campaign's recipe class is pinned to the production-optimal one.
RECIPE_CLASSES = frozenset({"production_optimal"})

# §7.2. `core_header` is its own risk tier, not a size band: its reach is every
# op in both the CPU and GPU builds, so it forces full-tree affected surface,
# per-backend binary comparison, and human review regardless of diff size
# (§8.5.1).
CHANGE_CLASSES = frozenset({
    "parameter", "dispatcher", "arithmetic", "layout", "fusion",
    "moe_scheduling", "recurrent", "scheduler_policy", "oracle_port",
    "core_header",
})

# §9.5 selects the cheap suite deterministically from `change_class`; a class
# with no cheap suite is rejected before it consumes a benchmark window (§7.2).
CHANGE_CLASS_CHEAP_SUITE = {
    "parameter": "successive_halving_microbench",
    "dispatcher": "per_path_microbench_and_trace",
    "arithmetic": "target_op_paired_ab",
    "layout": "kernel_ab_plus_load_and_capacity",
    "fusion": "node_launch_barrier_delta_plus_tiny_graph",
    "moe_scheduling": "expert_histogram_and_batched_graph",
    "recurrent": "bounded_decode_prefill_state_traffic",
    "scheduler_policy": "variable_arrival_replay",
    "oracle_port": "underlying_change_class_suite",
    "core_header": "full_tree_per_backend_binary_comparison",
}

CAMPAIGN_KINDS = frozenset({
    "config", "dispatch", "layout", "fusion", "scheduler", "capability",
    "oracle_port", "source_change",
})

# §5.7: the exclusion source is a CPU region claim or a GPU device claim.
# `stack` is the serving-runtime lane (§11.6), which never travels the
# kernel-freeze path.
RESOURCE_LANES = frozenset({"cpu", "gpu", "stack"})

CRITIC_STATUSES = frozenset({"pending", "pass", "fail"})

# §9: T1 is split into discriminator/tiny-graph/mechanism-receipt sub-tiers;
# T4 is the post-cutover activation watch (§11.5).
TIERS = frozenset({"T0", "T1", "T1a", "T1b", "T1c", "T2", "T3", "T4"})

CLAIM_CATEGORIES = frozenset({"OPTIMUM", "BASELINE", "CANDIDATE"})
METRIC_DIRECTIONS = frozenset({"higher_better", "lower_better"})

# `inconclusive` is DISTINCT from `invalid` (§7.4): "the experiment ran and did
# not resolve" is not "the experiment was not a measurement". Collapsing them
# either fabricates a negative result or discards a real one.
EVENT_STATUSES = frozenset({
    "pass", "fail", "inconclusive", "invalid", "timeout", "crash", "rejected",
})

# Invariant 12: a determinism class is an interface, so "we did not measure it"
# must be sayable without implying stability.
DETERMINISM_CLASSES = frozenset({
    "bitwise_stable", "bitwise_unstable", "not_measured",
})

# P-AK-SEARCH-1 "What voids a run", the two reasons whose SUBJECT is the anchor.
# ONLY a record declaring one of these may omit its `anchor` block (v3), because
# only these two say "there was no anchor identity to record" rather than "the
# anchor was fine and something else went wrong".
#
# These strings are the vocabulary of `evaluator/api.VOID_REASONS`, restated here
# rather than imported: `api` imports `schemas`, so importing back would be a
# cycle, and a validator that needs the evaluator loaded to validate a record is
# not a data contract. `evaluator/test_conformance.py` asserts the two lists agree,
# so the duplication is checked rather than trusted.
ANCHOR_VOID_REASONS = frozenset({
    "ANCHOR_MISSING_OR_MUTATED",
    "ANCHOR_GATE_FAILED",
})

#: Prefix `evaluator/api._derive()` uses when it projects a `VoidFinding` into the
#: record's top-level `integrity_flags` vector: `VOID:<reason>:<outcome>`.
VOID_FLAG_PREFIX = "VOID:"

MACHINE_SUBSETS = frozenset({"full", "partial"})

# §3.7: a verifier must be able to tell a defect from an expected absence.
DURABILITY_CLASSES = frozenset({
    "carried_in_git", "durable_untracked", "hash_and_provenance_only",
})

CANDIDATE_STATUSES = frozenset({
    "built", "build_failed", "evaluating", "banked", "rejected", "superseded",
    "invalid",
})
CHAMPION_STATUSES = frozenset({
    "none", "frontier", "champion_member", "composed_champion",
})

T3_VERDICTS = frozenset({"PASS", "FAIL", "PASS_WITH_WAIVER"})

# Fields holding planner prose. Excluded from retrieval by default (§5.5 item 6,
# invariant 20); a later proposal may cite one only by event id, which is the
# retrieval layer's job, not this module's.
NON_RETRIEVABLE_FIELDS = {
    SCHEMA_CAMPAIGN: frozenset(),
    SCHEMA_PROPOSAL: frozenset({"narrative"}),
    SCHEMA_CANDIDATE: frozenset({"narrative"}),
    SCHEMA_EVALUATION_EVENT_V2: frozenset({"narrative"}),
    SCHEMA_EVALUATION_EVENT_V3: frozenset({"narrative"}),
    SCHEMA_CHAMPION: frozenset(),
    SCHEMA_RELEASE_PACKAGE: frozenset(),
    SCHEMA_OPERATOR_WAIVER: frozenset(),
    # The operator surface renders no planner prose at all, so both versions have
    # an empty set rather than no entry: an ABSENT entry makes `retrievable_view`
    # raise on an otherwise valid record.
    SCHEMA_KERNEL_DASHBOARD_V1: frozenset(),
    SCHEMA_KERNEL_DASHBOARD_V2: frozenset(),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
# An evaluator id without a version suffix is a MUTABLE id: the bundle behind it
# can change while the id stays the same, so a resume cannot fail closed on
# drift (AK1, §12 "resume drift fails closed").
_VERSIONED_ID_RE = re.compile(r"^\S+/v\d+$")
_CO_RESIDENCY_RE = re.compile(r"^(single|co_resident:[A-Za-z0-9._:-]+)$")
# Frozen production branches (CLAUDE.md). A champion or candidate branch that
# names one is a category error, not a naming preference: invariant 3 says no
# actor builds in or modifies a production tree.
_PRODUCTION_BRANCH_RE = re.compile(r"^production-(consolidated|speech)-v\d+$")

#: SHA-256 of no bytes at all. It is well-formed hex and it is what a caller that
#: hashed nothing gets, so it names an artifact that was never read.
_EMPTY_INPUT_SHA256 = hashlib.sha256(b"").hexdigest()


def is_placeholder_digest(value: Any) -> bool:
    """True when `value` is a well-formed hex digest that no measurement produced.

    A regex over `[0-9a-f]{40,64}` cannot tell a hash from a hand-typed filler,
    and the filler is the dangerous one: `0` * 64 in an anchor is a *claim* that
    an anchor was resolved, and every downstream reader — the champion view, the
    readiness reducer, a human reading the journal — takes it for one. An ABSENT
    anchor is loud; a fabricated anchor is silent and wrong.

    Deliberately narrow, so it never falsely accuses a real digest:
      * a string of ONE repeated hex character (`0`*64, `f`*64, `a`*40 …), which
        no hash function has ever emitted for a real input; and
      * the SHA-256 of the empty input.

    It is NOT a general "does this look random" heuristic — a digest is either a
    known filler or it is treated as measured.
    """
    if not isinstance(value, str):
        return False
    if not (_SHA256_RE.match(value) or _COMMIT_RE.match(value)):
        return False
    return len(set(value)) <= 1 or value == _EMPTY_INPUT_SHA256


# =============================================================================
# `require` — validation is a FIELD TYPE, not a per-module helper
# =============================================================================
#
# WHY THIS IS HERE AND NOT COPIED INTO EACH MODULE
# ------------------------------------------------
# `_req_sha256` existed in three modules. Two of them matched `^[0-9a-f]{64}$`
# and stopped there, so `BuildProvenance.output_binary_sha256` — the identity of
# the built candidate — accepted sixty-four zeros; the third rejected it. The
# three bodies were otherwise byte-identical. That is not a coding-style problem:
# a fabricated identity reads as a resolved one to every downstream reader, and
# the copy that knew this could not tell the copies that did not.
#
# The fix is not "fix the two copies" — that was done, and the next module would
# have made a fourth. The fix is that a digest field has ONE type, it lives in
# the module every other module already imports, and the two ingredients a
# re-derivation needs are BOTH here:
#
#   * the predicate (`is_placeholder_digest`), and
#   * the shape (`SHA256_RE` / `COMMIT_RE`, public for exactly this reason).
#
# A module that wants its own digest validator must therefore either import them
# — an import a reviewer sees on the diff — or compile `^[0-9a-f]{64}$` locally,
# which `test_schemas_require.TestNoKeepSetModuleReDerivesAScalarValidator`
# refuses by name. There is no third route, and neither of the two is quiet.
#
# CONVENTIONS
# -----------
# * `error=` is the exception TYPE to raise. Modules that speak their own seam
#   language (`RecipeParameterError`, `ChainSeamError`) keep raising it; only the
#   predicate is shared, never the module's vocabulary.
# * `label` is the caller's field path and is always the first thing in the
#   message, because these fire inside `__post_init__` where the traceback names
#   the dataclass and not the field.
# * Every validator RETURNS the value, so a `__post_init__` can validate and
#   normalise in one expression.

#: Who produced a piece of evidence. `evaluator` is the only trusted producer;
#: `candidate` is a self-report and §8.5.1 refuses it as a gate result. Lives
#: here rather than in `evaluator/correctness.py` because `require.producer` is
#: the type of that field and this module may not import the evaluator.
EVIDENCE_PRODUCERS = ("evaluator", "candidate", "actor", "unknown")

#: PUBLIC on purpose. Every keep-set module used to compile its own copy of
#: these; that local `re.compile` is the first line of a re-derived validator and
#: it costs nothing to write, which is why nine modules wrote it. Naming them
#: here makes the copy a lint failure and the reuse an import.
SHA256_RE = _SHA256_RE
COMMIT_RE = _COMMIT_RE


def _require_str(value: Any, label: str, *, error=ValueError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _require_sha256(value: Any, label: str, *, error=ValueError) -> str:
    """A MEASURED sha256 digest. Well-formed hex is necessary and not sufficient."""
    if not isinstance(value, str) or not SHA256_RE.match(value):
        raise error(f"{label}: expected a lowercase sha256 hex digest, got {value!r}")
    if is_placeholder_digest(value):
        raise error(
            f"{label}: {value!r} is a placeholder digest, not a measured identity. A "
            "fabricated identity reads as a resolved one to every downstream reader, which "
            "is strictly worse than an absent one (correctness._validate_anchor_triple).")
    return value


def _require_commit(value: Any, label: str, *, error=ValueError) -> str:
    if not isinstance(value, str) or not COMMIT_RE.match(value):
        raise error(
            f"{label}: expected a full 40-hex git commit, got {value!r}. A short commit is "
            "ambiguous across a growing object store; the anchor must resolve to one tree")
    return value


def _require_abs_path(value: Any, label: str, *, error=ValueError) -> str:
    _require_str(value, label, error=error)
    if not value.startswith("/"):
        raise error(f"{label}: expected an absolute path, got {value!r}")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0, error=ValueError) -> int:
    """`bool` is not an int here: `True` as a token count is a type confusion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(f"{label}: expected an int, got {value!r}")
    if value < minimum:
        raise error(f"{label}: must be >= {minimum}, got {value!r}")
    return value


def _require_bool(value: Any, label: str, *, error=TypeError) -> bool:
    if not isinstance(value, bool):
        raise error(f"{label}: expected a bool, got {type(value).__name__}")
    return value


def _require_tuple(value: Any, label: str, *, error=TypeError) -> tuple:
    """A list is refused, not converted: a mutable field on a frozen record is a lie."""
    if not isinstance(value, tuple):
        raise error(f"{label}: expected a tuple, got {type(value).__name__}")
    return value


def _require_producer(value: Any, label: str, *, error=ValueError) -> str:
    if value not in EVIDENCE_PRODUCERS:
        raise error(f"{label}: {value!r} is not one of {list(EVIDENCE_PRODUCERS)}")
    return value


class require:
    """The field types. `require.sha256(v, "artifact.binary_sha256")`.

    A namespace, not a class — there is nothing to instantiate and nothing to
    subclass. It is spelled as a class so that `require.sha256` is one dotted
    name an AST audit can look for, and so the eight names cannot be imported
    individually into a module's own namespace where the origin stops showing.
    """

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "`require` is a namespace of field types, not a class to instantiate")

    sha256 = staticmethod(_require_sha256)
    commit = staticmethod(_require_commit)
    str = staticmethod(_require_str)
    int = staticmethod(_require_int)
    abs_path = staticmethod(_require_abs_path)
    producer = staticmethod(_require_producer)
    bool = staticmethod(_require_bool)
    tuple = staticmethod(_require_tuple)


# =============================================================================
# Canonical serialisation — other modules content-hash these records
# =============================================================================

def canonical_json(obj: Any) -> str:
    """Serialise `obj` with deterministic key ordering, for content hashing.

    Raises rather than degrading, because every failure mode here silently
    produces a WRONG hash rather than no hash:
      * non-string dict keys — `{1: "a"}` and `{"1": "a"}` would otherwise
        serialise identically, so two distinct records would collide;
      * NaN/Infinity — not JSON, and `float("nan") != float("nan")` makes any
        record containing one unequal to its own round-trip;
      * non-JSON types — `json.dumps` default coercion is where a datetime or a
        set turns into an unstable repr.
    """
    _assert_canonicalizable(obj, "$")
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    """UTF-8 encoding of `canonical_json` — the exact bytes that get hashed."""
    return canonical_json(obj).encode("utf-8")


def content_hash(obj: Any) -> str:
    """SHA-256 (hex) over the canonical encoding of `obj`."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _assert_canonicalizable(obj: Any, path: str) -> None:
    if obj is None or isinstance(obj, (bool, int, str)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"{path}: non-finite float is not canonicalizable: {obj!r}")
        return
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: dict keys must be strings for a stable content hash, "
                    f"got {type(key).__name__} {key!r}"
                )
            _assert_canonicalizable(value, f"{path}.{key}")
        return
    if isinstance(obj, (list, tuple)):
        # tuple is rejected below on purpose: it round-trips to a list, so a
        # record built with tuples would not equal its own reload.
        if isinstance(obj, tuple):
            raise TypeError(f"{path}: tuple is not canonicalizable (use a list)")
        for i, value in enumerate(obj):
            _assert_canonicalizable(value, f"{path}[{i}]")
        return
    raise TypeError(f"{path}: {type(obj).__name__} is not canonicalizable")


def retrievable_view(obj: Mapping[str, Any]) -> dict:
    """Copy of `obj` with its non-retrievable prose fields removed (§5.5 item 6).

    Raises on an unknown schema: if we cannot tell which fields are planner
    prose, returning the record unchanged would leak exactly the narrative the
    retrieval boundary exists to withhold.
    """
    if not isinstance(obj, Mapping):
        raise TypeError(f"record must be a mapping, got {type(obj).__name__}")
    schema = obj.get("schema")
    if schema not in NON_RETRIEVABLE_FIELDS:
        raise ValueError(
            f"unknown schema {schema!r}: cannot determine its non-retrievable "
            f"fields; known schemas are {sorted(NON_RETRIEVABLE_FIELDS)}"
        )
    drop = NON_RETRIEVABLE_FIELDS[schema]
    return {k: v for k, v in obj.items() if k not in drop}


# =============================================================================
# Three-outcome checkers (PASS / FAIL / COULD_NOT_CHECK)
# =============================================================================

PASS = "PASS"
FAIL = "FAIL"
COULD_NOT_CHECK = "COULD_NOT_CHECK"

#: The lattice, as data. Higher dominates. This is an ORDER, not an enum: the
#: reducer picks a maximum over it, which is why `COULD_NOT_CHECK` sitting
#: strictly between `PASS` and `FAIL` is the whole design and not an accident.
OUTCOME_SEVERITY = {PASS: 0, COULD_NOT_CHECK: 1, FAIL: 2}

#: Emitted by `Check.worst_of` when it is handed nothing. Stated as a reason, not
#: as a bare outcome, because a record that says COULD_NOT_CHECK has to say why.
EMPTY_CHECK_VECTOR_REASON = (
    "no checks were supplied: an empty check vector is COULD_NOT_CHECK, never PASS "
    "— a verdict derived from zero evidence is a fail-open, not a clean result"
)


@dataclass(frozen=True)
class Check:
    """A checker verdict. `COULD_NOT_CHECK` is a third outcome, not a soft pass."""

    outcome: str
    reasons: tuple = ()

    def __post_init__(self) -> None:
        if self.outcome not in (PASS, FAIL, COULD_NOT_CHECK):
            raise ValueError(f"invalid check outcome: {self.outcome!r}")

    @property
    def passed(self) -> bool:
        """True only for PASS. COULD_NOT_CHECK is deliberately falsy here."""
        return self.outcome == PASS

    @classmethod
    def worst_of(cls, checks: Iterable["Check"]) -> "Check":
        """FAIL > COULD_NOT_CHECK > PASS. An EMPTY vector is COULD_NOT_CHECK, never PASS.

        The empty case is the reason this exists. Nine of the eleven hand-written
        reducers in this package returned PASS from zero sub-checks, and
        `evaluator/api.py` names that failure in its own prose — *"an empty gate
        list derives to PASS and that is a fail-open verdict"* — in a module whose
        own reducer did exactly that. Zero evidence is not agreement; it is the
        third outcome.

        FAIL dominating COULD_NOT_CHECK is deliberate and is NOT a conflation:
        every non-PASS sub-reason is carried through PREFIXED WITH ITS OWN
        OUTCOME, so the combined record still says which sub-check failed and
        which merely could not be evaluated. Reasons attached to a PASS sub-check
        are dropped — a PASS carries no finding, and letting its prose ride along
        would make the reason list unattributable.

        A non-`Check` element RAISES rather than being coerced or skipped: a
        reducer that silently ignored a foreign object would report PASS for a
        vector it never actually reduced.

        `checks` is consumed exactly once, so a generator is fine — emptiness is
        detected by iteration, never by `len()` or truthiness.
        """
        outcome = PASS
        reasons: list = []
        saw_one = False
        for chk in checks:
            if not isinstance(chk, Check):
                raise TypeError(
                    f"worst_of takes schemas.Check values, got {type(chk).__name__}"
                )
            saw_one = True
            if chk.outcome == PASS:
                continue
            if OUTCOME_SEVERITY[chk.outcome] > OUTCOME_SEVERITY[outcome]:
                outcome = chk.outcome
            reasons.extend(f"[{chk.outcome}] {r}" for r in chk.reasons)
        if not saw_one:
            return cls(COULD_NOT_CHECK, (EMPTY_CHECK_VECTOR_REASON,))
        return cls(outcome, tuple(reasons))


def check_scope_denominator_admits_gate(
    event: Mapping[str, Any], gate_scope: Mapping[str, Any]
) -> Check:
    """FAIL if `gate_scope` is broader than the cell the event actually measured.

    §7.4/§12: a full-machine threshold applied to a partial-machine cell is a
    category error, and the required defence is that the gate REFUSES rather than
    demoting the cell. Returns COULD_NOT_CHECK when either side's scope is absent
    or malformed — an unreadable scope is not a matching scope.
    """
    cell = event.get("scope_denominator") if isinstance(event, Mapping) else None
    if not isinstance(cell, Mapping) or not isinstance(gate_scope, Mapping):
        return Check(COULD_NOT_CHECK, ("scope_denominator missing or not a mapping",))

    cell_subset = cell.get("machine_subset")
    gate_subset = gate_scope.get("machine_subset")
    if cell_subset not in MACHINE_SUBSETS or gate_subset not in MACHINE_SUBSETS:
        return Check(COULD_NOT_CHECK, ("machine_subset is absent or not in "
                                       f"{sorted(MACHINE_SUBSETS)}",))

    reasons = []
    if gate_subset == "full" and cell_subset == "partial":
        reasons.append("gate is calibrated full-machine but the cell measured a "
                       "partial machine")

    if cell_subset == "partial":
        cell_nodes = cell.get("numa_nodes")
        cell_devices = cell.get("devices")
        if not isinstance(cell_nodes, list) or not isinstance(cell_devices, list):
            return Check(COULD_NOT_CHECK,
                         ("partial cell does not declare numa_nodes/devices lists",))
        if not cell_nodes and not cell_devices:
            return Check(COULD_NOT_CHECK,
                         ("cell declares machine_subset=partial without naming any "
                          "numa node or device",))
        gate_nodes = gate_scope.get("numa_nodes") or []
        gate_devices = gate_scope.get("devices") or []
        missing_nodes = [n for n in gate_nodes if n not in cell_nodes]
        missing_devices = [d for d in gate_devices if d not in cell_devices]
        if missing_nodes:
            reasons.append(f"gate requires numa nodes not in the cell: {missing_nodes}")
        if missing_devices:
            reasons.append(f"gate requires devices not in the cell: {missing_devices}")

    cell_cores = cell.get("cores")
    gate_cores = gate_scope.get("cores")
    if isinstance(gate_cores, int) and not isinstance(gate_cores, bool):
        if not isinstance(cell_cores, int) or isinstance(cell_cores, bool):
            return Check(COULD_NOT_CHECK, ("cell does not declare an integer core count",))
        if gate_cores > cell_cores:
            reasons.append(f"gate requires {gate_cores} cores, cell measured {cell_cores}")

    return Check(FAIL, tuple(reasons)) if reasons else Check(PASS)


def check_anchor_binding(
    event: Mapping[str, Any],
    resolve_event: Optional[Callable[[str], Optional[Mapping[str, Any]]]] = None,
) -> Check:
    """Verify the event's ratio is bound to a real, matching anchor measurement.

    Every verdict is a ratio and a ratio needs its denominator bound (§7.4). A
    missing anchor is FAIL — `no-baseline is INVALID, never correct` (§12) — but
    resolving the referenced measurement events needs the journal, so without a
    `resolve_event` callable this returns COULD_NOT_CHECK rather than assuming
    the references are good.
    """
    anchor = event.get("anchor") if isinstance(event, Mapping) else None
    if not isinstance(anchor, Mapping):
        return Check(FAIL, ("event carries no anchor block",))
    binary = anchor.get("binary_sha256")
    if not isinstance(binary, str) or not _SHA256_RE.match(binary):
        return Check(FAIL, ("anchor.binary_sha256 is absent or not a sha256",))
    ids = anchor.get("measurement_event_ids")
    if not isinstance(ids, list):
        return Check(FAIL, ("anchor.measurement_event_ids is absent or not a list",))
    if not ids:
        return Check(FAIL, ("anchor names no measurement events to divide by",))
    if resolve_event is None:
        return Check(COULD_NOT_CHECK,
                     ("no journal resolver supplied; anchor references are unverified",))

    reasons = []
    for event_id in ids:
        resolved = resolve_event(event_id)
        if resolved is None:
            reasons.append(f"anchor measurement event {event_id!r} does not resolve")
            continue
        artifact = resolved.get("artifact") if isinstance(resolved, Mapping) else None
        got = artifact.get("binary_sha256") if isinstance(artifact, Mapping) else None
        if got != binary:
            reasons.append(
                f"anchor measurement event {event_id!r} measured binary {got!r}, "
                f"not the anchor binary {binary!r}"
            )
    return Check(FAIL, tuple(reasons)) if reasons else Check(PASS)


def check_metric_commensurability(
    backend: Optional[str], claim_grammar: Optional[Mapping[str, Any]]
) -> Check:
    """FAIL when a backend reports a metric that is not authoritative for it.

    `MEASUREMENT.md:23-30` makes task_rate and tokens/s authoritative in their own
    scopes and forbids substituting one for the other; §11.6 makes this concrete:
    a scheduler change is exactly the case where tokens are not commensurable
    across arms. The evaluation event does not carry its backend, so a caller
    that cannot supply it gets COULD_NOT_CHECK.
    """
    if backend is None or not isinstance(claim_grammar, Mapping):
        return Check(COULD_NOT_CHECK, ("backend or claim_grammar not supplied",))
    metric = claim_grammar.get("metric")
    if not isinstance(metric, str) or not metric:
        return Check(COULD_NOT_CHECK, ("claim_grammar.metric is absent",))
    if backend not in BACKENDS:
        return Check(COULD_NOT_CHECK, (f"unknown backend {backend!r}",))
    is_task_rate = "task_rate" in metric
    is_token_rate = "token" in metric and ("_s" in metric or "per_s" in metric)
    if backend == "serving_runtime" and is_token_rate:
        return Check(FAIL, ("serving_runtime must report task_rate, not tokens/s "
                            "(MEASUREMENT.md:23-30, §11.6)",))
    if backend != "serving_runtime" and is_task_rate:
        return Check(FAIL, (f"{backend} reports a token-rate metric; task_rate belongs "
                            "to the serving_runtime scope",))
    return Check(PASS)


# =============================================================================
# Authority-flag rejection (§1.3)
# =============================================================================

# Keys are scanned, values are not: `release_protocol: P-KERNEL-FREEZE-1/v1` is a
# legitimate VALUE naming the freeze protocol, whereas a KEY is where a record
# would declare that it may act on it.
_AUTHORITY_ACTION_TOKENS = frozenset({
    "freeze", "freezes", "frozen", "cutover", "cutovers", "promote", "promotes",
    "promotion", "promotions", "ratify", "ratifies", "ratification", "sign",
    "signoff", "deploy", "deployment", "release", "releases",
})
_AUTHORITY_QUALIFIER_TOKENS = frozenset({
    "auto", "automatic", "autonomous", "unattended", "unsupervised", "authority",
    "authorize", "authorized", "authorised", "approve", "approved", "allow",
    "allowed", "enable", "enabled", "may", "can", "permit", "permitted", "self",
    "grant", "granted", "override",
})
# Substring stems, for keys written without separators (`autofreeze`).
_ACTION_STEMS = ("freeze", "cutover", "promot", "ratif", "deploy", "signoff", "releas")
_STRONG_QUALIFIER_STEMS = (
    "authoriz", "authoris", "authority", "approv", "unattended", "autonomous",
    "unsupervised", "permitted", "granted", "override",
)
_BARE_AUTHORITY_KEYS = frozenset({
    "freeze", "cutover", "promote", "promotion", "ratify", "signoff", "signed",
})


def _key_tokens(key: str) -> list:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def _is_authority_flavoured(key: str) -> bool:
    tokens = set(_key_tokens(key))
    if tokens & _AUTHORITY_ACTION_TOKENS and tokens & _AUTHORITY_QUALIFIER_TOKENS:
        return True
    flat = "".join(tokens)
    if flat in _BARE_AUTHORITY_KEYS:
        return True
    for stem in _ACTION_STEMS:
        if stem not in flat:
            continue
        # "auto" is only decisive when it is glued to the action itself:
        # `autofreeze` is authority, `draft_autopilot_rebaseline_note` is not.
        if "auto" + stem in flat:
            return True
        if any(q in flat for q in _STRONG_QUALIFIER_STEMS):
            return True
    return False


def find_authority_flavoured_keys(obj: Any, path: str = "$") -> list:
    """Return dotted paths of every auth-flavoured key anywhere inside `obj`.

    §1.3: an automatic freeze crosses four human-only trust boundaries
    (`MEASUREMENT.md:140-142`), so no machine-authored record carries a freeze,
    cutover, promotion, or ratification authority flag. This scan exists so that
    absence is ENFORCED rather than merely documented — a flag added later fails
    validation instead of quietly becoming load-bearing.
    """
    found = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{path}.{key}"
            if isinstance(key, str) and _is_authority_flavoured(key):
                found.append(child)
            found.extend(find_authority_flavoured_keys(value, child))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(find_authority_flavoured_keys(value, f"{path}[{i}]"))
    return found


# =============================================================================
# Machine actors and the trust boundary (§1.3, §10.4)
#
# Both halves of this section are SHARED VOCABULARY, and they live here for one
# reason: they were owned by `release/packager.py`, one layer above the gate that
# needs them, and a rule enforced only at the outer layer is not enforced. A
# waiver attributed to `autokernel` verified as human-attested inside
# `t3.verify_waiver` — turning FAIL into PASS_WITH_WAIVER — and was refused only
# when the packager later assembled a package, so any caller reaching T3 directly
# bypassed the refusal entirely. `schemas.py` is the module every plane already
# imports, so it is the only place the two planes can share one answer.
#
# Everything here is PURE. This module performs no I/O (see the header), so the
# trust-boundary manifest is PARSED here and READ by the caller that owns the
# filesystem — `t3.human_only_boundary()`.
# =============================================================================

#: Tokens that betray a MACHINE actor in an identity field. Matched on word
#: boundaries against the lowercased identity, so a human called "Daniele" is
#: unaffected and `autokernel-daemon` is not. §10.4 makes a waiver human-authored
#: by definition and `MEASUREMENT.md:140-142` makes freeze/cutover a human-only
#: write; an automated identity in an attestation field is the loop authorising
#: itself.
MACHINE_ACTOR_TOKENS = frozenset({
    "autokernel", "autopilot", "controller", "planner", "critic", "packager",
    "evaluator", "daemon", "agent", "subagent", "bot", "cron", "timer", "scheduler",
    "loop", "runner", "worker", "automation", "robot", "script",
})

#: Every key an attestation may use to name WHO authorised it. Enumerated once:
#: a guard that scans `authorized_by` and not `approved_by` is a guard with a
#: rename-shaped hole.
ACTOR_ATTRIBUTION_FIELDS = (
    "authorized_by", "ratified_by", "approved_by", "attested_by", "granted_by",
)

_IDENTITY_TOKEN_RE = re.compile(r"[a-z0-9]+")
#: The same runs with digits treated as separators rather than as run content — see
#: `identity_candidates`, where a digit between the two words was a walk-around.
_IDENTITY_ALPHA_RE = re.compile(r"[a-z]+")
_DIGITS = "0123456789"

#: How many adjacent alphanumeric runs may be re-joined when looking for a token.
#: Bounded so a pathological identity cannot cost O(n^2) — four is past every
#: separator spelling of every token in the set.
_MAX_JOINED_RUNS = 4
#: And a ceiling on how many runs are considered at all, for the same reason.
_MAX_IDENTITY_RUNS = 64


def identity_candidates(identity: Any) -> frozenset:
    """Every string an identity could be SPELLING, for machine-token matching.

    Splitting on non-alphanumerics alone was a rename-shaped hole: `autokernel` was
    refused and `auto-kernel`, `auto_kernel`, `auto.kernel`, `auto pilot` and
    `autokernel2` all sailed through, so the guard was walk-aroundable by typing a
    separator. So the candidates are the runs, the concatenations of ADJACENT runs
    (bounded), and each of those with leading/trailing digits removed.

    It is deliberately NOT substring matching, because substring matching is the
    thing that would break the compliant path: `"scriptor"` contains `script` and
    is a perfectly good human handle. A candidate must be a whole re-joining, so
    `"scriptor"` yields only `{"scriptor"}` and matches nothing.
    """
    if not isinstance(identity, str):
        return frozenset()
    lowered = identity.lower()
    out: set = set()
    # TWO run vocabularies, and the second is not redundant. `[a-z0-9]+` treats a
    # digit as part of a run, so a digit BETWEEN the two words defeated the rejoining
    # entirely: `autokernel`, `auto-kernel`, `auto_kernel` and `auto kernel` were all
    # refused while `auto2kernel` and `auto1kernel` sailed through — the same
    # separator-shaped hole this function was written to close, with a digit as the
    # separator. `strip(_DIGITS)` did not reach it because it only strips the ends.
    # So the alphabetic-only runs are considered as well.
    for pattern in (_IDENTITY_TOKEN_RE, _IDENTITY_ALPHA_RE):
        runs = pattern.findall(lowered)[:_MAX_IDENTITY_RUNS]
        for start in range(len(runs)):
            joined = ""
            for offset in range(min(_MAX_JOINED_RUNS, len(runs) - start)):
                joined += runs[start + offset]
                out.add(joined)
                out.add(joined.strip(_DIGITS))
    out.discard("")
    return frozenset(out)


def machine_actor_tokens(identity: Any) -> tuple:
    """The machine-actor tokens `identity` contains, sorted; `()` for a human name.

    A non-string identity yields `()` rather than raising: callers that require a
    string check that separately, and a scanner that raised would be one a caller
    could disable by passing the wrong type.
    """
    return tuple(sorted(identity_candidates(identity) & MACHINE_ACTOR_TOKENS))


#: Any key of the shape `…_by` is an attribution whether or not it is enumerated.
#: `ACTOR_ATTRIBUTION_FIELDS`' own docstring says *"a guard that scans
#: `authorized_by` and not `approved_by` is a guard with a rename-shaped hole"*, and
#: the enumeration was still a closed list of five: a §10.4 waiver carrying
#: `waived_by: "autokernel"` (or `signed_by`, `issued_by`, `created_by`,
#: `requested_by`) named the loop as its own author, was seen by NOTHING, and then
#: took the no-attribution branch of `t3.verify_waiver` — which reads a document that
#: names no author as human-attested on the strength of where it lives, because the
#: preserved v8 record has no author field. So an explicit machine attribution in an
#: unenumerated key was strictly SAFER for a forger than no attribution at all.
#:
#: Enumerated-plus-shape rather than shape-only: `ACTOR_ATTRIBUTION_FIELDS` still
#: decides what counts as a NAMED HUMAN actor (a `*_by` key is not automatically an
#: authority), while this widens only the REFUSAL. Widening a refusal cannot admit
#: anything that was refused before, and the compliant control is the genuine v8
#: waiver, whose eleven keys contain no `*_by` at all.
_ATTRIBUTION_KEY_SUFFIX_RE = re.compile(r"(?:^|_)by$")

#: Attribution spellings that are not `*_by` shaped. Refusal-only, same as above.
_EXTRA_ATTRIBUTION_KEYS = ("author", "actor")


def attribution_keys(document: Any) -> tuple:
    """Every key in `document` that attributes it to somebody, sorted.

    The five enumerated fields, plus every key spelled `*_by`. A document is
    attributed by the shape of its keys, not by whether this module happened to
    enumerate the spelling its author chose.
    """
    if not isinstance(document, Mapping):
        return ()
    keys = set(ACTOR_ATTRIBUTION_FIELDS) | set(_EXTRA_ATTRIBUTION_KEYS)
    keys.update(k for k in document
                if isinstance(k, str) and _ATTRIBUTION_KEY_SUFFIX_RE.search(k))
    return tuple(sorted(keys))


def machine_attributions(document: Any) -> tuple:
    """`(field, identity, tokens)` for every attribution field naming a machine."""
    if not isinstance(document, Mapping):
        return ()
    found: list = []
    for field_name in attribution_keys(document):
        identity = document.get(field_name)
        tokens = machine_actor_tokens(identity)
        if tokens:
            found.append((field_name, identity, tokens))
    return tuple(found)


#: Where the trust boundary is DECLARED, repo-relative inside epyc-root. This
#: module never restates its contents: the manifest is the single source of truth,
#: it is human-amendment-only (its own preamble, plus a PreToolUse hook and a
#: `.sha256` pin), and a copy here would be a second boundary that drifts.
HUMAN_ONLY_PATHS_MANIFEST = "coordination/session-bus/human_only_paths.yaml"

#: The one operator-owned root this module can name without the manifest: the
#: root every preserved operator attestation the release plane reads already
#: lives under — `artifacts/operator/ratify_v8_final_freeze_20260725.json`,
#: `artifacts/operator/ratify_speech_kernel_freeze_20260731.json`, the
#: measurement-v2 ratification ledger (`MEASUREMENT.md:2`), the §11.6 freeze
#: receipt paths, and both draft speech protocol families. It is a ROOT, not a
#: path list, and it is deliberately additive to the manifest rather than a
#: substitute for it: the manifest's `paths:` block does not yet name a waiver
#: home (§3.6's open "add … to human_only_paths.yaml" item, which is a human-only
#: amendment), and when it does, this check picks it up with no code change.
OPERATOR_ATTESTATION_ROOT = "artifacts/operator"

#: Host roots the repos are checked out under (CLAUDE.md "Repository Map"). Used
#: ONLY to reduce an absolute citation to the repo-relative form the manifest is
#: written in. It never widens the boundary: a path under none of these roots
#: yields no repo-relative form at all and therefore matches nothing.
REPO_CHECKOUT_ROOTS = ("/workspace/repos", "/workspace", "/mnt/raid0/llm")

#: The directory names that ARE a repository checkout when they appear directly
#: under a checkout root — CLAUDE.md's "Repository Map", with the kernel trees
#: DERIVED from `SOURCE_TREE_BY_BACKEND` so a backend cannot gain a tree this set
#: does not know about.
#:
#: This set is what makes the repo-name strip in `repo_relative_forms` safe. The
#: strip exists so an absolute citation can be matched against a manifest glob
#: written repo-relative (`orchestration/instrument_eras.yaml` under
#: `/workspace/repos/epyc-orchestrator/`). Applied to ANY leading segment it
#: instead MANUFACTURES a repo-relative form out of a directory that is not a
#: repository: `/mnt/raid0/llm/tmp/artifacts/operator/w.json` reduced to
#: `artifacts/operator/w.json` and read as the operator attestation root, and
#: `/mnt/raid0/llm/tmp/orchestration/instrument_eras.yaml` matched the era-registry
#: glob. `/mnt/raid0/llm/tmp/` is the loop's OWN scratch root (it is where
#: `resource/device_claim.py` puts its lock files), so that was an operator-owned
#: verdict obtainable with `mkdir -p`.
REPO_CHECKOUT_NAMES = frozenset(SOURCE_TREES) | frozenset({
    "epyc-root", "epyc-orchestrator", "epyc-inference-research", "epyc-llama",
    "epyc-whisper", "epyc-qwentts",
})


@dataclass(frozen=True)
class TrustBoundary:
    """The human-only path set, as parsed from the manifest.

    `readable` is False for an absent, empty, or foreign document. That state is
    load-bearing: `operator_owned_path_check` reports COULD_NOT_CHECK rather than
    PASS when the boundary is unreadable, so deleting or emptying the manifest can
    never widen what counts as operator-owned.
    """

    globs: tuple = ()
    branches: tuple = ()
    source: str = ""

    @property
    def readable(self) -> bool:
        return bool(self.globs)

    def to_dict(self) -> dict:
        return {"globs": list(self.globs), "branches": list(self.branches),
                "source": self.source, "readable": self.readable}


def parse_trust_boundary(text: Any, *, source: str = "") -> TrustBoundary:
    """Parse `human_only_paths.yaml` text into the glob set it declares.

    Pure: text in, data out. Returns an UNREADABLE boundary (`globs=()`) for text
    that is absent, unparsable, or not this schema — never a boundary that happens
    to be empty, because an empty boundary that read as usable would admit
    everything the manifest exists to refuse.
    """
    if not isinstance(text, str) or not text.strip():
        return TrustBoundary(source=source)
    try:
        import yaml  # declared in pyproject as `pyyaml>=6.0`
    except ImportError:  # pragma: no cover - dependency is declared
        return TrustBoundary(source=source)
    try:
        loaded = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - any parse failure is "not readable"
        return TrustBoundary(source=source)
    if not isinstance(loaded, Mapping):
        return TrustBoundary(source=source)
    if not str(loaded.get("schema_version", "")).startswith(
            "session_bus.human_only_paths."):
        # A foreign document is not this boundary. Reading its `paths:` block
        # anyway is how an audit gets satisfied by a file somebody swapped.
        return TrustBoundary(source=source)

    def _globs(key: str) -> tuple:
        entries = loaded.get(key)
        if not isinstance(entries, list):
            return ()
        out = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            glob = entry.get("glob")
            if isinstance(glob, str) and glob.strip():
                out.append(glob.strip())
        return tuple(dict.fromkeys(out))

    return TrustBoundary(globs=_globs("paths"), branches=_globs("branches"),
                         source=source)


def _normalised_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        return ""
    # POSIX leaves a LEADING DOUBLE slash implementation-defined, and `normpath`
    # preserves it: `posixpath.normpath('//x')` is `'//x'`, while `'///x'` collapses
    # to `'/x'`. Measured, not assumed. Left alone that made `//workspace/artifacts/
    # operator/w.json` reduce to no repo-relative form at all (a fail-CLOSED false
    # negative on a legitimately-spelled citation) while `///workspace/...` PASSed —
    # two answers for one location, decided by a slash. Nothing in this package
    # attaches meaning to `//`, so it is collapsed here, ONCE, before any matching:
    # a consumer that checks one spelling and opens another has no guarantee at all.
    if cleaned.startswith("//"):
        cleaned = "/" + cleaned.lstrip("/")
    normalised = posixpath.normpath(cleaned)
    return normalised.rstrip("/") or "/"


def canonical_citation(document_path: Any) -> str:
    """The ONE canonical spelling of a citation: normalised, single leading slash.

    Public because a reader must CHECK and OPEN the same string. `repo_relative_forms`
    normalises internally, so a caller that checked `document_path` and then opened
    the raw text was checking one path and reading another whenever the two differed
    (`//x`, `a/./b`, a trailing slash). Returns `""` for a non-string or empty
    citation — a location that cannot be spelled is not one this module can canonicalise.
    """
    if not isinstance(document_path, str):
        return ""
    return _normalised_path(document_path)


def under_any_root(path: Any, roots: Any) -> bool:
    """Is `path` equal to, or contained by, any root in `roots`?

    Containment on RESOLVED path segments, never on a substring: `/a/bc` is not under
    `/a/b`. Both sides are canonicalised first so the answer does not depend on how
    either was spelled. An empty root set is False — a check against nothing must
    never read as "allowed".
    """
    target = canonical_citation(path)
    if not target:
        return False
    for root in (roots or ()):
        canonical_root = canonical_citation(root)
        if canonical_root and _under(target, canonical_root):
            return True
    return False


#: Ceiling on an operator waiver document, in bytes. Calibrated, not guessed: the
#: preserved v8 attestation `artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json`
#: is 1,267 bytes, three orders of magnitude below this. A reader without a ceiling
#: is a reader that will happily hash a multi-gigabyte file somebody dropped at an
#: operator-owned path.
MAX_OPERATOR_WAIVER_BYTES = 1024 * 1024


def raw_bytes_digest(raw: Any) -> str:
    """SHA-256 (hex) over RAW FILE BYTES — deliberately NOT `content_hash`.

    The two differ and the difference is load-bearing. `content_hash` digests the
    CANONICAL re-encoding of a parsed object, so it is stable across whitespace and
    key order; that is what a record's own identity wants. But an operator
    attestation is pinned by the digest of the FILE: the v8 ratification's
    `evidence_sha256.waive_q8` is
    `fcd52b61610fcc2782e11f41ffac359343233924805f83d872eeceffbb7522d7`, which is
    `sha256(waive_q8_cpu_prefill_v8_20260725.json)`; `content_hash` of the same
    parsed document is `0fc095d3…`, and matches nothing anybody ratified.

    A reader that used `content_hash` would therefore be unable to verify a single
    real operator record, and — worse — would be verifying a re-encoding it produced
    rather than the bytes that were signed.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("raw_bytes_digest: expects bytes read from a file")
    return hashlib.sha256(bytes(raw)).hexdigest()


def repo_relative_forms(document_path: Any) -> tuple:
    """The repo-relative form(s) of a citation, for matching against the manifest.

    A relative path is already in the manifest's vocabulary. An absolute path is
    reduced at a known checkout root, and the repo-name segment is offered as a
    second form so `/workspace/repos/epyc-orchestrator/orchestration/x.yaml`
    matches the manifest's `orchestration/x.yaml`. A path under no checkout root
    (`/tmp/artifacts/operator/w.json`) reduces to NOTHING and therefore matches
    nothing — the containment test is on the resolved root, never on a substring.

    The repo-name strip is taken ONLY when the segment it removes is in
    `REPO_CHECKOUT_NAMES`. Stripping any leading segment does not reduce a
    citation, it INVENTS one: `/mnt/raid0/llm/tmp/artifacts/operator/w.json` became
    `artifacts/operator/w.json` and read as operator-owned, and
    `/mnt/raid0/llm/tmp/orchestration/instrument_eras.yaml` matched the human-only
    era-registry glob. Both are inside the loop's own scratch root, so both were a
    trust-boundary PASS obtainable with `mkdir -p`.
    """
    if not isinstance(document_path, str):
        return ()
    normalised = _normalised_path(document_path)
    if not normalised or normalised == "/":
        return ()
    if not normalised.startswith("/"):
        # `..` escapes the root it was resolved against, so it names no repo-
        # relative location at all.
        return () if normalised.split("/")[0] == ".." else (normalised,)
    for root in REPO_CHECKOUT_ROOTS:
        if not normalised.startswith(root + "/"):
            continue
        rest = normalised[len(root) + 1:]
        if not rest:
            return ()
        forms = [rest]
        head, _, tail = rest.partition("/")
        if tail and head in REPO_CHECKOUT_NAMES:
            forms.append(tail)
        return tuple(dict.fromkeys(forms))
    return ()


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def operator_owned_path_check(document_path: Any, *,
                              boundary: Optional[TrustBoundary] = None) -> Check:
    """Is `document_path` somewhere an OPERATOR owns, and a machine does not?

    §10.4 requires an operator waiver to be *"stored under the trust-boundary path
    set"*. Without this, `WaiverBinding.document_path` is free text: a document the
    loop wrote to its own scratch directory hash-verifies exactly as well as a
    ratified attestation, because a hash proves only that the bytes did not change
    after somebody quoted them.

    Three outcomes, and the third is why this is safe to consult:

      * PASS — under `OPERATOR_ATTESTATION_ROOT`, or matching a glob the manifest
        declares.
      * COULD_NOT_CHECK — not under the attestation root AND the boundary is
        unreadable. The manifest might have named this path; nobody can say.
      * FAIL — the boundary is readable and names nothing that covers this path,
        or the path resolves to no repo-relative location at all.

    Deleting, emptying, or swapping the manifest therefore turns FAIL into
    COULD_NOT_CHECK, never into PASS — and callers that treat COULD_NOT_CHECK as
    "not verified" (`t3.verify_waiver` does) stay fail-closed either way.
    """
    if not isinstance(document_path, str) or not document_path.strip():
        return Check(FAIL, ("document_path: a waiver that names no location cannot be "
                            "shown to live anywhere an operator owns",))
    forms = repo_relative_forms(document_path)
    if not forms:
        return Check(FAIL, (
            f"document_path: {document_path!r} resolves to no repo-relative location "
            f"(checkout roots {list(REPO_CHECKOUT_ROOTS)}), so it is outside every "
            "path the trust boundary can speak about",))
    for form in forms:
        if _under(form, OPERATOR_ATTESTATION_ROOT):
            return Check(PASS)
    if boundary is not None and boundary.readable:
        for form in forms:
            for glob in boundary.globs:
                if fnmatchcase(form, glob):
                    return Check(PASS)
        return Check(FAIL, (
            f"document_path: {document_path!r} is neither under "
            f"{OPERATOR_ATTESTATION_ROOT!r} nor matched by any human-only path in "
            f"{boundary.source or HUMAN_ONLY_PATHS_MANIFEST}. §10.4 stores a waiver "
            "under the trust-boundary path set; a document at a path the loop can "
            "write is a document the loop can author.",))
    return Check(COULD_NOT_CHECK, (
        f"document_path: {document_path!r} is not under "
        f"{OPERATOR_ATTESTATION_ROOT!r}, and the trust-boundary manifest "
        f"({boundary.source if boundary is not None else HUMAN_ONLY_PATHS_MANIFEST}) "
        "could not be read, so whether it is operator-owned is unknown. An unreadable "
        "boundary is not an empty one.",))


# =============================================================================
# Field-level validation helpers
#
# Every helper appends violations and returns _MISSING on failure, so callers can
# keep checking the rest of the record instead of stopping at the first problem:
# a planner or journal reader wants the whole violation list at once.
# =============================================================================

_MISSING = object()


def _fetch(obj, name, out, prefix):
    if not isinstance(obj, Mapping):
        where = prefix.rstrip(".") or "record"
        out.append(f"{where}: expected a mapping, got {type(obj).__name__}")
        return _MISSING
    if name not in obj:
        out.append(f"{prefix}{name}: required field is missing")
        return _MISSING
    return obj[name]


def _need_str(obj, name, out, prefix, *, allow_empty=False, choices=None,
              pattern=None, pattern_hint=""):
    value = _fetch(obj, name, out, prefix)
    if value is _MISSING:
        return _MISSING
    if not isinstance(value, str):
        out.append(f"{prefix}{name}: expected a string, got {type(value).__name__}")
        return _MISSING
    if not allow_empty and not value.strip():
        out.append(f"{prefix}{name}: must not be empty")
        return _MISSING
    if choices is not None and value not in choices:
        out.append(f"{prefix}{name}: {value!r} is not one of {sorted(choices)}")
        return _MISSING
    if pattern is not None and not pattern.match(value):
        out.append(f"{prefix}{name}: {value!r} {pattern_hint}")
        return _MISSING
    return value


def _need_bool(obj, name, out, prefix, *, must_be=None):
    value = _fetch(obj, name, out, prefix)
    if value is _MISSING:
        return _MISSING
    if not isinstance(value, bool):
        out.append(f"{prefix}{name}: expected a boolean, got {type(value).__name__}")
        return _MISSING
    if must_be is not None and value is not must_be:
        out.append(f"{prefix}{name}: must be {str(must_be).lower()}")
        return _MISSING
    return value


def _need_int(obj, name, out, prefix, *, minimum=None, maximum=None):
    value = _fetch(obj, name, out, prefix)
    if value is _MISSING:
        return _MISSING
    if isinstance(value, bool) or not isinstance(value, int):
        out.append(f"{prefix}{name}: expected an integer, got {type(value).__name__}")
        return _MISSING
    if minimum is not None and value < minimum:
        out.append(f"{prefix}{name}: must be >= {minimum}, got {value}")
        return _MISSING
    if maximum is not None and value > maximum:
        out.append(f"{prefix}{name}: must be <= {maximum}, got {value}")
        return _MISSING
    return value


def _need_number(obj, name, out, prefix, *, minimum=None, maximum=None):
    """A budget/cost field. Missing, negative, or non-finite is an UNBOUNDED
    request, which AK1 requires validators to refuse; zero is a bounded budget
    that permits nothing, which is fail-closed and therefore legal."""
    value = _fetch(obj, name, out, prefix)
    if value is _MISSING:
        return _MISSING
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        out.append(f"{prefix}{name}: expected a number, got {type(value).__name__}")
        return _MISSING
    if isinstance(value, float) and not math.isfinite(value):
        out.append(f"{prefix}{name}: must be finite, got {value!r} (an unbounded request)")
        return _MISSING
    if minimum is not None and value < minimum:
        out.append(f"{prefix}{name}: must be >= {minimum}, got {value}")
        return _MISSING
    if maximum is not None and value > maximum:
        out.append(f"{prefix}{name}: must be <= {maximum}, got {value}")
        return _MISSING
    return value


def _need_list(obj, name, out, prefix, *, non_empty=False, item_type=None,
               item_desc="item"):
    value = _fetch(obj, name, out, prefix)
    if value is _MISSING:
        return _MISSING
    if not isinstance(value, list):
        out.append(f"{prefix}{name}: expected a list, got {type(value).__name__}")
        return _MISSING
    if non_empty and not value:
        out.append(f"{prefix}{name}: must not be empty")
        return _MISSING
    if item_type is not None:
        for i, item in enumerate(value):
            bad_bool = item_type is not bool and isinstance(item, bool)
            if bad_bool or not isinstance(item, item_type):
                out.append(f"{prefix}{name}[{i}]: expected {item_desc}, "
                           f"got {type(item).__name__}")
            elif item_type is str and not item.strip():
                out.append(f"{prefix}{name}[{i}]: must not be empty")
    return value


def _need_dict(obj, name, out, prefix, *, non_empty=False, desc=None):
    value = _fetch(obj, name, out, prefix)
    if value is _MISSING:
        return _MISSING
    if not isinstance(value, Mapping):
        detail = f" ({desc})" if desc else ""
        out.append(f"{prefix}{name}: expected a mapping{detail}, "
                   f"got {type(value).__name__}")
        return _MISSING
    if non_empty and not value:
        out.append(f"{prefix}{name}: must not be empty")
        return _MISSING
    return value


def _need_sha256(obj, name, out, prefix):
    return _need_str(obj, name, out, prefix, pattern=_SHA256_RE,
                     pattern_hint="is not a lowercase hex sha256")


def _need_commit(obj, name, out, prefix):
    return _need_str(obj, name, out, prefix, pattern=_COMMIT_RE,
                     pattern_hint="is not a full 40-hex git commit")


def _need_timestamp(obj, name, out, prefix):
    value = _need_str(obj, name, out, prefix)
    if value is _MISSING:
        return _MISSING
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        out.append(f"{prefix}{name}: {value!r} is not an ISO-8601 timestamp")
        return _MISSING
    if parsed.tzinfo is None:
        # A naive timestamp on a shared host is ambiguous across sessions, and
        # ordering the journal is how a rewind decides what to keep.
        out.append(f"{prefix}{name}: {value!r} has no timezone offset")
        return _MISSING
    return value


def _need_id(obj, name, out, prefix, expected_prefix):
    value = _need_str(obj, name, out, prefix)
    if value is _MISSING:
        return _MISSING
    if not value.startswith(expected_prefix):
        out.append(f"{prefix}{name}: {value!r} must start with {expected_prefix!r}")
        return _MISSING
    return value


def _check_schema_header(obj, expected, out) -> bool:
    if not isinstance(obj, Mapping):
        out.append(f"record: expected a mapping, got {type(obj).__name__}")
        return False
    value = obj.get("schema")
    if value is None:
        out.append("schema: required field is missing "
                   "(the version string is the record's identity)")
        return False
    if value != expected:
        out.append(f"schema: expected {expected!r}, got {value!r}")
        return False
    return True


def _reject_authority_keys(obj, out) -> None:
    for path in find_authority_flavoured_keys(obj):
        out.append(
            f"{path}: authority-flavoured key is forbidden — AutoKernel holds no "
            f"freeze/cutover/promotion authority to declare (§1.3)"
        )


def _check_narrative(obj, out, *, required: bool) -> None:
    """Prose is allowed; prose that claims to be retrievable is not (§5.5 item 6).

    `required=True` for the proposal, whose contract (§7.2) declares a narrative
    field. Elsewhere prose is optional, but the moment a record carries any, it
    must carry the marking too — an unmarked narrative is exactly the record the
    retrieval layer would let through by default.
    """
    has_narrative = "narrative" in obj
    if has_narrative and not isinstance(obj.get("narrative"), str):
        out.append("narrative: expected a string, got "
                   f"{type(obj.get('narrative')).__name__}")
    if required and not has_narrative:
        out.append("narrative: required field is missing (planner prose is a separate "
                   "field from the machine record, §5.5 item 6)")
    if not (required or has_narrative):
        return
    value = obj.get("narrative_retrievable", _MISSING)
    if value is _MISSING:
        out.append("narrative_retrievable: required field is missing "
                   "(planner prose must be marked non-retrievable, §5.5 item 6)")
    elif value is not False:
        out.append("narrative_retrievable: must be false — planner narrative is not "
                   "retrievable into a later planning context (invariant 20)")


# =============================================================================
# epyc.autokernel.campaign.v2 (§7.1)
# =============================================================================

def validate_campaign(obj: Any) -> list:
    """Validate a campaign manifest. Returns violations; empty list means valid."""
    out: list = []
    if not _check_schema_header(obj, SCHEMA_CAMPAIGN, out):
        return out

    # The scan runs first and unconditionally: a campaign that smuggles in an
    # authority flag is rejected even if every other field is malformed too.
    _reject_authority_keys(obj, out)

    _need_id(obj, "campaign_id", out, "", "ak-")
    backend = _need_str(obj, "backend", out, "", choices=BACKENDS)
    source_tree = _need_str(obj, "source_tree", out, "")
    if backend is not _MISSING and source_tree is not _MISSING:
        expected = SOURCE_TREE_BY_BACKEND.get(backend)
        if expected is not None and source_tree != expected:
            out.append(f"source_tree: backend {backend!r} lives in {expected!r}, "
                       f"got {source_tree!r}")

    anchor = _need_dict(obj, "production_anchor", out, "")
    if anchor is not _MISSING:
        repo = _need_str(anchor, "repo", out, "production_anchor.")
        if repo is not _MISSING and not repo.startswith("/"):
            out.append(f"production_anchor.repo: {repo!r} must be an absolute path")
        _need_str(anchor, "branch", out, "production_anchor.")
        _need_commit(anchor, "commit", out, "production_anchor.")

    objective = _need_dict(obj, "objective", out, "")
    if objective is not _MISSING:
        _need_str(objective, "rule", out, "objective.", choices=OBJECTIVE_RULES)
        phases = _need_list(objective, "phases", out, "objective.", non_empty=True,
                            item_type=str, item_desc="a phase name")
        allowed = PHASES_BY_BACKEND.get(backend) if backend is not _MISSING else None
        if phases is not _MISSING and allowed is not None:
            for phase in phases:
                if isinstance(phase, str) and phase not in allowed:
                    out.append(f"objective.phases: {phase!r} is not one of "
                               f"{sorted(allowed)} for backend {backend!r}")
        by_phase = _need_dict(objective, "protocol_by_phase", out, "objective.")
        if by_phase is not _MISSING and phases is not _MISSING:
            for phase in phases:
                if not isinstance(phase, str):
                    continue
                protocol = by_phase.get(phase)
                if not isinstance(protocol, str) or not protocol.strip():
                    out.append(f"objective.protocol_by_phase[{phase!r}]: every declared "
                               "phase needs its own protocol id (MEASUREMENT.md:13)")
            for phase in by_phase:
                if phase not in phases:
                    out.append(f"objective.protocol_by_phase[{phase!r}]: names a phase "
                               "the objective does not declare")
        _need_str(objective, "recipe_class", out, "objective.", choices=RECIPE_CLASSES)
        _need_list(objective, "target_regimes", out, "objective.")
        exception = _fetch(objective, "phase_trade_exception", out, "objective.")
        if exception is not _MISSING and exception is not None:
            if not isinstance(exception, Mapping):
                out.append("objective.phase_trade_exception: expected null or a mapping, "
                           f"got {type(exception).__name__}")
            else:
                # Pre-declared or absent: a trade discovered after measuring is
                # not an exception, it is a regression.
                for key in ("regressing_phase", "band", "expected_gain", "roles"):
                    if key not in exception:
                        out.append(f"objective.phase_trade_exception.{key}: required "
                                   "field is missing in a pre-declared exception")

    scope = _need_dict(obj, "scope", out, "")
    if scope is not _MISSING:
        _need_list(scope, "affected_ops", out, "scope.", item_type=str,
                   item_desc="an op name")
        _need_list(scope, "affected_arch_classes", out, "scope.", item_type=str,
                   item_desc="an architecture class")
        # A campaign whose scope hash is absent means the scope compiler never
        # ran, and §6.4 forbids the planner from filling scope itself.
        _need_sha256(scope, "derived_role_manifest_sha256", out, "scope.")

    policy = _need_dict(obj, "policy_ref", out, "")
    if policy is not _MISSING:
        _need_str(policy, "search_protocol", out, "policy_ref.")
        _need_str(policy, "release_protocol", out, "policy_ref.")
        _need_sha256(policy, "policy_bundle_sha256", out, "policy_ref.")

    budgets = _need_dict(obj, "budgets", out, "")
    if budgets is not _MISSING:
        for key in ("max_wall_hours", "max_gpu_hours", "max_cpu_region_hours",
                    "max_storage_gb"):
            _need_number(budgets, key, out, "budgets.", minimum=0)
        for key in ("max_candidates", "max_controller_tokens"):
            _need_int(budgets, key, out, "budgets.", minimum=0)

    readiness = _need_dict(obj, "readiness_reporting", out, "")
    if readiness is not _MISSING:
        # ADVISORY signal to the operator, never a trigger (§1.2, §7.1).
        _need_number(readiness, "reference_point_gain", out, "readiness_reporting.")
        _need_number(readiness, "reference_lcb_gain", out, "readiness_reporting.")

    stop = _need_dict(obj, "stop_policy", out, "")
    if stop is not _MISSING:
        for key in ("plateau_rounds", "max_consecutive_integrity_failures",
                    "max_consecutive_build_failures"):
            _need_int(stop, key, out, "stop_policy.", minimum=0)
        # OPERATING_CONSTRAINTS.md:44-46 caps command retries at 3.
        _need_int(stop, "max_command_retries", out, "stop_policy.", minimum=0, maximum=3)

    if "created_at" in obj:
        _need_timestamp(obj, "created_at", out, "")
    return out


# =============================================================================
# epyc.autokernel.proposal.v2 (§7.2)
# =============================================================================

def validate_proposal(obj: Any) -> list:
    """Validate a proposal manifest. Returns violations; empty list means valid."""
    out: list = []
    if not _check_schema_header(obj, SCHEMA_PROPOSAL, out):
        return out
    _reject_authority_keys(obj, out)

    _need_id(obj, "proposal_id", out, "", "akp-")
    _need_id(obj, "campaign_id", out, "", "ak-")
    parent = _fetch(obj, "parent_candidate_id", out, "")
    if parent is not _MISSING and parent is not None:
        if not isinstance(parent, str) or not parent.startswith("akc-"):
            out.append("parent_candidate_id: must be null (root proposal) or a "
                       "candidate id starting with 'akc-'")

    # Controller provenance, so controller A/B is computable after the fact and a
    # zero-yield proposal class can be attributed to the model that produced it.
    controller = _need_dict(obj, "controller", out, "")
    if controller is not _MISSING:
        for key in ("provider", "model_id", "effort"):
            _need_str(controller, key, out, "controller.")
        _need_sha256(controller, "prompt_bundle_sha256", out, "controller.")
        _need_sha256(controller, "context_manifest_sha256", out, "controller.")
        _need_dict(controller, "sampling_params", out, "controller.")

    cost = _need_dict(obj, "realized_cost", out, "")
    if cost is not _MISSING:
        _need_int(cost, "controller_tokens", out, "realized_cost.", minimum=0)
        for key in ("build_seconds", "evaluator_wall_seconds", "gpu_seconds",
                    "cpu_region_seconds", "storage_gb"):
            _need_number(cost, key, out, "realized_cost.", minimum=0)

    _need_str(obj, "hypothesis", out, "")
    _check_narrative(obj, out, required=True)

    change_class = _need_str(obj, "change_class", out, "", choices=CHANGE_CLASSES)
    if change_class is not _MISSING and change_class not in CHANGE_CLASS_CHEAP_SUITE:
        out.append(f"change_class: {change_class!r} maps to no cheap suite (§9.5); a "
                   "proposal with no cheap suite is rejected before it consumes a "
                   "benchmark window")

    deltas = _need_dict(obj, "declared_symbol_deltas", out, "")
    if deltas is not _MISSING:
        # §8.5.1: anything outside this declared set that the binary diff finds is
        # a hard T0 failure. An absent key is an undeclared removal waiting to
        # happen, so all three are required even when empty.
        for key in ("added", "removed", "arity_changed"):
            _need_list(deltas, key, out, "declared_symbol_deltas.", item_type=str,
                       item_desc="a symbol name")

    kind = _need_str(obj, "campaign_kind", out, "", choices=CAMPAIGN_KINDS)
    oracle = _need_dict(obj, "oracle_reference", out, "")
    if oracle is not _MISSING and kind == "oracle_port":
        for key in ("oracle", "commit", "license_check"):
            value = oracle.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"oracle_reference.{key}: required and non-empty when "
                           "campaign_kind == 'oracle_port'")

    novelty = _need_dict(obj, "novelty_basis", out, "")
    if novelty is not _MISSING:
        for key in ("prior_event_ids", "source_receipts", "do_not_repeat_matches"):
            _need_list(novelty, key, out, "novelty_basis.")

    # §8.4 ranks expected information gain FIRST, so it is not optional.
    _need_number(obj, "expected_information_gain", out, "", minimum=0)

    for key in ("target", "non_target"):
        block = _need_dict(obj, key, out, "")
        if block is not _MISSING:
            for sub in ("regimes", "shapes"):
                _need_list(block, sub, out, f"{key}.")
    target = obj.get("target")
    if isinstance(target, Mapping):
        for sub in ("ops", "models"):
            _need_list(target, sub, out, "target.")

    mechanism = _need_dict(obj, "mechanism_prediction", out, "")
    if mechanism is not _MISSING:
        _need_str(mechanism, "bottleneck_before", out, "mechanism_prediction.")
        # A proposal without a falsifiable counter is rejected (§7.2), so the
        # counter map must be non-empty, not merely present.
        _need_dict(mechanism, "expected_counter_changes", out, "mechanism_prediction.",
                   non_empty=True)
        _need_number(mechanism, "expected_wall_share_ceiling", out,
                     "mechanism_prediction.", minimum=0, maximum=1)
        _need_str(mechanism, "wall_share_receipt_id", out, "mechanism_prediction.")

    change = _need_dict(obj, "change", out, "")
    if change is not _MISSING:
        # Scored prediction only; never a scope input (§6.4, invariant 18).
        _need_list(change, "predicted_affected_surface", out, "change.")
        _need_list(change, "files_and_symbols", out, "change.")
        _need_str(change, "conceptual_change", out, "change.")
        _need_dict(change, "parameter_surface", out, "change.")
        _need_int(change, "estimated_diff_size", out, "change.", minimum=0)

    risks = _need_dict(obj, "risks", out, "")
    if risks is not _MISSING:
        for key in ("correctness", "numerical", "state_or_rollback", "resource",
                    "integrity"):
            _need_list(risks, key, out, "risks.")

    fallback = _need_dict(obj, "fallback", out, "")
    if fallback is not _MISSING:
        # Invariant 16: default-off until release, UNLESS the change is
        # structurally inseparable and the campaign explicitly carries that risk
        # class. The escape hatch has to be declared, not assumed.
        inseparable = fallback.get("structurally_inseparable") is True
        if inseparable:
            ack = fallback.get("risk_class_ack")
            if not isinstance(ack, str) or not ack.strip():
                out.append("fallback.risk_class_ack: a structurally inseparable change "
                           "must name the risk class the campaign carries (invariant 16)")
        else:
            for key in ("dispatch_guard", "kill_switch"):
                _need_str(fallback, key, out, "fallback.")

    plan = _need_dict(obj, "evaluation_plan", out, "")
    if plan is not _MISSING:
        _need_list(plan, "required_t0", out, "evaluation_plan.", non_empty=True,
                   item_type=str, item_desc="a T0 gate name")
        _need_list(plan, "required_t1", out, "evaluation_plan.", item_type=str,
                   item_desc="a T1 cell name")
        _need_list(plan, "conditional_t2", out, "evaluation_plan.")
        _need_list(plan, "profiler_questions", out, "evaluation_plan.")

    request = _need_dict(obj, "resource_request", out, "")
    if request is not _MISSING:
        _need_str(request, "lane", out, "resource_request.", choices=RESOURCE_LANES)
        _need_number(request, "expected_minutes", out, "resource_request.", minimum=0)
        _need_number(request, "expected_storage_gb", out, "resource_request.", minimum=0)

    _need_str(obj, "stop_condition", out, "")
    verdict = _need_dict(obj, "critic_verdict", out, "")
    if verdict is not _MISSING:
        _need_str(verdict, "status", out, "critic_verdict.", choices=CRITIC_STATUSES)
        _need_list(verdict, "reasons", out, "critic_verdict.")

    if "created_at" in obj:
        _need_timestamp(obj, "created_at", out, "")
    return out


# =============================================================================
# epyc.autokernel.candidate.v1 (§7.3, prose formalised)
# =============================================================================

def validate_candidate(obj: Any) -> list:
    """Validate a candidate record — the artifact's reproducibility contract.

    §7.3 states the requirement in prose and adds that the existing natural key
    `(label, ts, git_sha)` is insufficient. `candidate_natural_key()` below is
    the replacement this schema makes checkable.
    """
    out: list = []
    if not _check_schema_header(obj, SCHEMA_CANDIDATE, out):
        return out
    _reject_authority_keys(obj, out)

    _need_id(obj, "candidate_id", out, "", "akc-")
    _need_id(obj, "campaign_id", out, "", "ak-")
    _need_id(obj, "proposal_id", out, "", "akp-")
    parent = _fetch(obj, "parent_candidate_id", out, "")
    if parent is not _MISSING and parent is not None:
        if not isinstance(parent, str) or not parent.startswith("akc-"):
            out.append("parent_candidate_id: must be null or start with 'akc-'")

    worktree = _need_dict(obj, "worktree", out, "")
    if worktree is not _MISSING:
        path = _need_str(worktree, "path", out, "worktree.")
        branch = _need_str(worktree, "branch", out, "worktree.")
        if branch is not _MISSING and _PRODUCTION_BRANCH_RE.match(branch):
            out.append(f"worktree.branch: {branch!r} is a frozen production branch; "
                       "no actor builds in or modifies a production tree (invariant 3)")
        if path is not _MISSING and not path.startswith("/"):
            out.append(f"worktree.path: {path!r} must be an absolute path")
        _need_commit(worktree, "source_commit", out, "worktree.")
        # A dirty worktree means the snapshot hash is not the thing that built.
        _need_bool(worktree, "clean", out, "worktree.")

    snapshot = _need_dict(obj, "source_snapshot", out, "")
    if snapshot is not _MISSING:
        _need_sha256(snapshot, "snapshot_sha256", out, "source_snapshot.")
        _need_sha256(snapshot, "patch_bundle_sha256", out, "source_snapshot.")

    ancestry = _need_dict(obj, "ancestry", out, "")
    if ancestry is not _MISSING:
        _need_commit(ancestry, "production_base_commit", out, "ancestry.")
        # Invariant 1: every campaign is anchored on the current production tip.
        # The proof is recorded so a stale base is a validation failure, not a
        # discovery made at seal time.
        _need_bool(ancestry, "is_descendant_of_production_base", out, "ancestry.",
                   must_be=True)
        _need_str(ancestry, "proof", out, "ancestry.")

    build = _need_dict(obj, "build", out, "")
    if build is not _MISSING:
        for key in ("toolchain", "compiler", "command", "build_dir", "log_path"):
            _need_str(build, key, out, "build.")
        _need_sha256(build, "log_sha256", out, "build.")

    artifacts = _need_dict(obj, "artifacts", out, "")
    if artifacts is not _MISSING:
        _need_sha256(artifacts, "binary_sha256", out, "artifacts.")
        # CLAUDE.md: three trees run three ggml generations, so a binary that
        # inherits another tree's ggml runs silently wrong. The linkage proof is
        # part of the artifact's identity, not a build detail.
        _need_sha256(artifacts, "linkage_sha256", out, "artifacts.")
        libs = _need_dict(artifacts, "library_sha256s", out, "artifacts.")
        if libs is not _MISSING:
            for name, digest in libs.items():
                if not isinstance(digest, str) or not _SHA256_RE.match(digest):
                    out.append(f"artifacts.library_sha256s[{name!r}]: not a sha256")

    dispatch = _need_dict(obj, "dispatch", out, "")
    if dispatch is not _MISSING:
        _need_list(dispatch, "feature_flags", out, "dispatch.", item_type=str,
                   item_desc="a flag name")
        _need_str(dispatch, "dispatch_predicate", out, "dispatch.", allow_empty=True)

    surface = _need_dict(obj, "affected_surface", out, "")
    if surface is not _MISSING:
        # Invariant 18: declared equals traced. Both manifests are recorded so
        # `traced ⊄ derived` is checkable rather than asserted.
        _need_sha256(surface, "derived_sha256", out, "affected_surface.")
        traced = _fetch(surface, "traced_sha256", out, "affected_surface.")
        if traced is not _MISSING and traced is not None:
            if not isinstance(traced, str) or not _SHA256_RE.match(traced):
                out.append("affected_surface.traced_sha256: must be null (not yet "
                           "traced) or a sha256")
        reconciled = _fetch(surface, "reconciled", out, "affected_surface.")
        if reconciled is not _MISSING and not isinstance(reconciled, bool):
            out.append("affected_surface.reconciled: expected a boolean")
        if reconciled is True and (traced is _MISSING or traced is None):
            out.append("affected_surface.reconciled: cannot be true while "
                       "traced_sha256 is null (nothing was traced to reconcile)")

    determinism = _need_dict(obj, "determinism", out, "")
    if determinism is not _MISSING:
        klass = _need_str(determinism, "class", out, "determinism.",
                          choices=DETERMINISM_CLASSES)
        repeats = _need_int(determinism, "same_seed_repeat_runs", out, "determinism.",
                            minimum=0)
        if klass in ("bitwise_stable", "bitwise_unstable") and repeats == 0:
            out.append("determinism.same_seed_repeat_runs: a determinism class cannot "
                       "be claimed from zero same-seed repeats (use 'not_measured')")

    evaluator = _need_dict(obj, "evaluator", out, "")
    if evaluator is not _MISSING:
        _need_str(evaluator, "id", out, "evaluator.", pattern=_VERSIONED_ID_RE,
                  pattern_hint="is a mutable evaluator id (needs a '/vN' suffix)")
        _need_sha256(evaluator, "bundle_sha256", out, "evaluator.")

    receipts = _need_dict(obj, "receipts", out, "")
    if receipts is not _MISSING:
        _need_str(receipts, "host_receipt", out, "receipts.")
        # Invariant 9: resources are acquired, not observed. Idle sensing is
        # never a claim, so the claim receipt is mandatory.
        _need_str(receipts, "resource_claim_receipt", out, "receipts.")

    storage = _need_dict(obj, "storage", out, "")
    if storage is not _MISSING:
        _need_number(storage, "footprint_gb", out, "storage.", minimum=0)
        _need_str(storage, "durability_class", out, "storage.",
                  choices=DURABILITY_CLASSES)

    _need_list(obj, "evaluation_event_ids", out, "", item_type=str,
               item_desc="an event id")
    _need_dict(obj, "derived_verdicts", out, "")

    # Controller provenance is inherited from the proposal so a candidate can be
    # attributed without a join against a proposal that may be superseded.
    controller = _need_dict(obj, "controller", out, "")
    if controller is not _MISSING:
        for key in ("provider", "model_id", "effort"):
            _need_str(controller, key, out, "controller.")
        _need_sha256(controller, "prompt_bundle_sha256", out, "controller.")

    _need_str(obj, "champion_status", out, "", choices=CHAMPION_STATUSES)
    status = _need_str(obj, "status", out, "", choices=CANDIDATE_STATUSES)
    reason = _fetch(obj, "supersession_reason", out, "")
    if status == "superseded":
        if not isinstance(reason, str) or not reason.strip():
            out.append("supersession_reason: required and non-empty when "
                       "status == 'superseded' (invariant 8: supersession is an "
                       "event carrying a reason, never a deletion)")
    elif reason is not _MISSING and reason is not None and not isinstance(reason, str):
        out.append("supersession_reason: expected null or a string")

    _check_narrative(obj, out, required=False)
    _need_timestamp(obj, "created_at", out, "")
    return out


def candidate_natural_key(obj: Mapping[str, Any]) -> tuple:
    """The identity tuple that replaces `(label, ts, git_sha)` (§7.3).

    A candidate is the same candidate only when the campaign, the exact source
    snapshot, the exact built binary, its linkage, and the evaluator bundle that
    scored it all match. The old key collided across rebuilds and across
    evaluator changes, which is how a rescored run could overwrite a different
    measurement. Raises on a record that cannot supply the key rather than
    returning a partial tuple that would dedup wrongly.
    """
    if not isinstance(obj, Mapping):
        raise TypeError(f"candidate record must be a mapping, got {type(obj).__name__}")
    try:
        return (
            obj["campaign_id"],
            obj["source_snapshot"]["snapshot_sha256"],
            obj["artifacts"]["binary_sha256"],
            obj["artifacts"]["linkage_sha256"],
            obj["evaluator"]["bundle_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"candidate record cannot supply its natural key: {exc}"
        ) from exc


# =============================================================================
# epyc.autokernel.evaluation_event — v2 and v3 (§7.4)
# =============================================================================

def _check_anchor_measurement_ids(anchor, out, tier) -> None:
    ids = _need_list(anchor, "measurement_event_ids", out, "anchor.",
                     item_type=str, item_desc="an event id")
    if ids is not _MISSING and not ids and tier != "T0":
        out.append("anchor.measurement_event_ids: must name at least one anchor "
                   f"measurement for tier {tier!r} (only T0 compares artifacts "
                   "rather than rates)")


def _check_anchor_block_v2(obj, out, tier) -> None:
    """v2's anchor rule: the block is unconditionally required, two digests only.

    Kept EXACTLY as it was. v2 records already exist; a validator that quietly
    grew a third required field would turn every one of them into a defect
    report on the next read.
    """
    anchor = _need_dict(obj, "anchor", out, "")
    if anchor is _MISSING:
        return
    _need_sha256(anchor, "binary_sha256", out, "anchor.")
    _need_sha256(anchor, "linkage_sha256", out, "anchor.")
    _check_anchor_measurement_ids(anchor, out, tier)


def declared_anchor_void_reasons(obj: Any) -> list:
    """The ANCHOR-related void reasons a record declares, in `integrity_flags`.

    `evaluator/api._derive()` is the only producer of `integrity_flags` and it
    writes one `VOID:<reason>:<outcome>` entry per triggered void condition — the
    same vector `evaluator/test_conformance` already asserts a voided record
    carries. Reading the reason from that REQUIRED top-level list is what lets
    the v3 anchor exemption be checked structurally; the free-form
    `performance.search_discipline.void_findings` block carries the same findings
    in full, but a conditional that hangs off a free-form block is exactly the
    smuggling this version exists to end.
    """
    if not isinstance(obj, Mapping):
        return []
    flags = obj.get("integrity_flags")
    if not isinstance(flags, list):
        return []
    found = []
    for flag in flags:
        if not isinstance(flag, str) or not flag.startswith(VOID_FLAG_PREFIX):
            continue
        parts = flag.split(":")
        if len(parts) >= 2 and parts[1] in ANCHOR_VOID_REASONS and parts[1] not in found:
            found.append(parts[1])
    return found


def _check_anchor_block_v3(obj, out, tier) -> None:
    """v3's anchor rule — the exact conditional, stated once.

    `anchor` is REQUIRED, and may be omitted **only** when BOTH hold:

      1. `status == "invalid"`, and
      2. `integrity_flags` names at least one anchor-related void reason
         (`ANCHOR_VOID_REASONS`).

    Anything else — a `pass`, `fail`, `inconclusive`, `timeout`, `crash` or
    `rejected` record with no anchor, or an `invalid` record voided for some
    OTHER reason (host health, storage, hand-typed argv) — is a violation. Those
    runs had an anchor or should have had one; dropping the block would hide a
    ratio with no denominator behind a status that says nothing about anchors.

    There is no third option. `anchor: null` is refused so that absence has ONE
    representation, and a placeholder digest is refused outright: an anchor block
    that says `0`*64 is a claim that an anchor was resolved.
    """
    present = isinstance(obj, Mapping) and "anchor" in obj
    if not present:
        status = obj.get("status") if isinstance(obj, Mapping) else None
        declared = declared_anchor_void_reasons(obj)
        if status != "invalid" or not declared:
            out.append(
                "anchor: required field is missing — a record may omit its anchor ONLY "
                "when status is 'invalid' AND integrity_flags names one of "
                f"{sorted(ANCHOR_VOID_REASONS)}; this record declares status={status!r} "
                f"and anchor void reasons {declared}. There is no placeholder digest to "
                "supply instead: a fabricated anchor reads as a resolved one to every "
                "downstream reader, which is worse than an absent block"
            )
        return

    if obj.get("anchor") is None:
        out.append(
            "anchor: is null — an absent anchor is expressed by OMITTING the key, so "
            "that absence has exactly one representation and no reader has to decide "
            "whether null meant 'no anchor' or 'not filled in yet'"
        )
        return

    anchor = _need_dict(obj, "anchor", out, "")
    if anchor is _MISSING:
        return
    # Precondition 4: "names its anchor by source commit, binary SHA-256, and
    # linkage SHA-256". All three, validated the same way the candidate record
    # validates `worktree.source_commit`.
    _need_commit(anchor, "source_commit", out, "anchor.")
    _need_sha256(anchor, "binary_sha256", out, "anchor.")
    _need_sha256(anchor, "linkage_sha256", out, "anchor.")
    for name in ("source_commit", "binary_sha256", "linkage_sha256"):
        value = anchor.get(name)
        if is_placeholder_digest(value):
            out.append(
                f"anchor.{name}: {value!r} is a placeholder digest, not a measured "
                "identity. A run with no anchor omits the block; it never fills it in"
            )
    _check_anchor_measurement_ids(anchor, out, tier)


def _validate_evaluation_event(obj: Any, *, schema: str, check_anchor) -> list:
    """The body shared by every evaluation-event version.

    Only the schema string and the anchor rule differ between v2 and v3, so only
    those are parameters. Everything else is one implementation: two hand-copied
    validators drift, and the half that drifts is whichever one has fewer tests.
    """
    out: list = []
    if not _check_schema_header(obj, schema, out):
        return out
    _reject_authority_keys(obj, out)

    _need_id(obj, "event_id", out, "", "ake-")
    _need_id(obj, "campaign_id", out, "", "ak-")
    _need_id(obj, "candidate_id", out, "", "akc-")
    tier = _need_str(obj, "tier", out, "", choices=TIERS)

    # MEASUREMENT.md:13 and :85-95 — an unlabelled measurement is not
    # decision-grade, so the claim grammar is structural, not advisory (§3.4).
    claim = _need_dict(obj, "claim_grammar", out, "")
    if claim is not _MISSING:
        _need_str(claim, "category", out, "claim_grammar.", choices=CLAIM_CATEGORIES)
        _need_str(claim, "protocol_id", out, "claim_grammar.")
        _need_str(claim, "metric", out, "claim_grammar.")
        _need_str(claim, "metric_direction", out, "claim_grammar.",
                  choices=METRIC_DIRECTIONS)
        # n/reps is part of the claim; zero reps is not a measurement.
        _need_int(claim, "reps", out, "claim_grammar.", minimum=1)
        _need_str(claim, "attestation_ref", out, "claim_grammar.")

    evaluator = _need_dict(obj, "evaluator", out, "")
    if evaluator is not _MISSING:
        _need_str(evaluator, "id", out, "evaluator.", pattern=_VERSIONED_ID_RE,
                  pattern_hint="is a mutable evaluator id (needs a '/vN' suffix)")
        _need_sha256(evaluator, "bundle_sha256", out, "evaluator.")

    artifact = _need_dict(obj, "artifact", out, "")
    if artifact is not _MISSING:
        for key in ("source_sha256", "binary_sha256", "linkage_sha256"):
            _need_sha256(artifact, key, out, "artifact.")

    # Every verdict is a ratio and a ratio needs its denominator bound (§7.4).
    # An optional baseline is what let "coherent" pass in the current scaffold
    # (§12); no-baseline is INVALID, never correct.
    check_anchor(obj, out, tier)

    _need_sha256(obj, "scope_manifest_sha256", out, "")
    _need_str(obj, "host_receipt", out, "")
    _need_str(obj, "resource_claim_receipt", out, "")
    _need_str(obj, "co_residency", out, "", pattern=_CO_RESIDENCY_RE,
              pattern_hint="must be 'single' or 'co_resident:<lineup_id>'")

    # Per-case / per-question / per-iteration VECTORS. A rolled-up boolean here
    # is how a failing case disappears into an aggregate (§7.4).
    for key in ("correctness", "quality", "stability", "mechanism"):
        _need_dict(obj, key, out, "",
                   desc="a per-case vector, not a rolled-up verdict")

    # §7.4/§12: so a full-machine threshold can never be applied to a
    # partial-machine cell.
    scope = _need_dict(obj, "scope_denominator", out, "")
    if scope is not _MISSING:
        subset = _need_str(scope, "machine_subset", out, "scope_denominator.",
                           choices=MACHINE_SUBSETS)
        nodes = _need_list(scope, "numa_nodes", out, "scope_denominator.",
                           item_type=int, item_desc="a numa node index")
        devices = _need_list(scope, "devices", out, "scope_denominator.",
                             item_type=str, item_desc="a device id")
        _need_int(scope, "cores", out, "scope_denominator.", minimum=0)
        if subset == "partial" and nodes is not _MISSING and devices is not _MISSING:
            if not nodes and not devices:
                out.append("scope_denominator: machine_subset='partial' must name the "
                           "numa nodes or devices it measured, otherwise the cell's "
                           "denominator is unknown")

    determinism = _need_dict(obj, "determinism", out, "")
    if determinism is not _MISSING:
        klass = _need_str(determinism, "class", out, "determinism.",
                          choices=DETERMINISM_CLASSES)
        repeats = _need_int(determinism, "same_seed_repeat_runs", out, "determinism.",
                            minimum=0)
        if klass in ("bitwise_stable", "bitwise_unstable") and repeats == 0:
            out.append("determinism.same_seed_repeat_runs: a determinism class cannot "
                       "be claimed from zero same-seed repeats (use 'not_measured')")

    performance = _need_dict(obj, "performance", out, "")
    if performance is not _MISSING:
        samples = _need_list(performance, "raw_samples", out, "performance.")
        _need_int(performance, "paired_blocks", out, "performance.", minimum=0)
        for key in ("estimate", "uncertainty"):
            if key not in performance:
                out.append(f"performance.{key}: required field is missing "
                           "(null means 'not derived', which must be explicit)")
        # "Derived scores are reproducible from raw samples. The candidate cannot
        # supply its own trusted score." (§7.4)
        if performance.get("estimate") is not None:
            if samples is _MISSING or not samples:
                out.append("performance.estimate: an estimate without raw_samples is a "
                           "self-reported score and is not reproducible (§7.4)")

    flags = _need_list(obj, "integrity_flags", out, "", item_type=str,
                       item_desc="an integrity flag")
    status = _need_str(obj, "status", out, "", choices=EVENT_STATUSES)
    if status == "pass" and flags not in (_MISSING, None) and flags:
        # Invariant 6: correctness is lexicographically first. An event carrying
        # integrity flags cannot also be a pass.
        out.append(f"status: cannot be 'pass' while integrity_flags is non-empty "
                   f"({flags})")

    _need_list(obj, "supersedes", out, "", item_type=str, item_desc="an event id")
    _check_narrative(obj, out, required=False)
    _need_timestamp(obj, "created_at", out, "")
    return out


def validate_evaluation_event_v2(obj: Any) -> list:
    """Validate a v2 evaluation event. Reader for records already in a journal."""
    return _validate_evaluation_event(obj, schema=SCHEMA_EVALUATION_EVENT_V2,
                                      check_anchor=_check_anchor_block_v2)


def validate_evaluation_event_v3(obj: Any) -> list:
    """Validate a v3 evaluation event — the decision-grade unit of evidence."""
    return _validate_evaluation_event(obj, schema=SCHEMA_EVALUATION_EVENT_V3,
                                      check_anchor=_check_anchor_block_v3)


#: Every evaluation-event version that can still be read, by its schema string.
EVALUATION_EVENT_VALIDATORS = {
    SCHEMA_EVALUATION_EVENT_V2: validate_evaluation_event_v2,
    SCHEMA_EVALUATION_EVENT_V3: validate_evaluation_event_v3,
}


def validate_evaluation_event(obj: Any) -> list:
    """Validate an evaluation event under the version the record DECLARES.

    Dispatch, not a fallback: the schema string is the record's identity, so a v2
    record is checked by v2's rules and a v3 record by v3's, and a record naming
    neither is a violation rather than a best-effort read. Callers that mean one
    specific version (a migration, a test asserting v2 still reads) name it —
    `validate_evaluation_event_v2` / `_v3`.
    """
    if not isinstance(obj, Mapping):
        return [f"record: expected a mapping, got {type(obj).__name__}"]
    schema = obj.get("schema")
    validator = EVALUATION_EVENT_VALIDATORS.get(schema)
    if validator is None:
        return [f"schema: {schema!r} is not an AutoKernel evaluation_event schema; "
                f"known versions are {sorted(EVALUATION_EVENT_VALIDATORS)}"]
    return validator(obj)


# =============================================================================
# epyc.autokernel.champion.v1 (§7.5)
# =============================================================================

#: The condition name an operator surface gives a champion `blocking_conditions`
#: entry that is not written in the machine vocabulary the rest of the package
#: uses (`EVALUATOR_COVERAGE_GAP`, `REANCHOR_PENDING_REMEASURE`,
#: `T2_INTERACTION_FAILED`, …). The entry's own text is carried as the detail, so
#: nothing is lost and nothing is invented.
#:
#: It lives HERE, beside the record it describes, and not in the surface: the
#: champion record owns its blocking conditions, so the fallback name for one of
#: them is the record schema's to define. A surface that minted its own name for
#: it would be the private taxonomy `DASHBOARD_BLOCKING_ORIGINS` refuses to grow.
CHAMPION_BLOCKED_UNNAMED = "CHAMPION_BLOCKED"


def validate_champion(obj: Any) -> list:
    """Validate a champion record — one composed lineage per source tree."""
    out: list = []
    if not _check_schema_header(obj, SCHEMA_CHAMPION, out):
        return out
    _reject_authority_keys(obj, out)

    _need_str(obj, "source_tree", out, "", choices=SOURCE_TREES)
    _need_commit(obj, "anchor_commit", out, "")
    branch = _need_str(obj, "branch", out, "")
    if branch is not _MISSING:
        if _PRODUCTION_BRANCH_RE.match(branch):
            out.append(f"branch: {branch!r} is a frozen production branch (invariant 3)")
        elif not branch.startswith("ak/"):
            out.append(f"branch: {branch!r} must be namespaced under 'ak/'")

    members = _need_list(obj, "member_candidates", out, "", item_type=str,
                         item_desc="a candidate id")
    combined = _fetch(obj, "combined_candidate_id", out, "")
    if members is not _MISSING and members:
        # Never infer composition by multiplying local speedups: the composed
        # artifact is re-measured as a whole (§8.9), so it must exist.
        if not isinstance(combined, str) or not combined.startswith("akc-"):
            out.append("combined_candidate_id: a non-empty lineage must name the "
                       "composed candidate that was re-measured as a whole (§8.9)")
    elif combined is not _MISSING and combined is not None:
        if not isinstance(combined, str) or not combined.startswith("akc-"):
            out.append("combined_candidate_id: must be null or start with 'akc-'")

    blocking = _need_list(obj, "blocking_conditions", out, "", item_type=str,
                          item_desc="a blocking condition")
    for key in ("last_t0", "last_t1", "last_t2"):
        value = _fetch(obj, key, out, "")
        if value is _MISSING or value is None:
            continue
        if not isinstance(value, Mapping):
            out.append(f"{key}: expected null (not yet run) or a mapping, "
                       f"got {type(value).__name__}")
            continue
        _need_str(value, "event_id", out, f"{key}.")
        status = _need_str(value, "status", out, f"{key}.", choices=EVENT_STATUSES)
        if status not in (_MISSING, "pass") and not blocking:
            # The champion is the always-green lineage; a non-green champion must
            # say what is holding it (e.g. EVALUATOR_COVERAGE_GAP).
            out.append(f"blocking_conditions: must not be empty while {key}.status is "
                       f"{status!r} (the champion is the always-green lineage, §8.9)")

    readiness = _need_dict(obj, "readiness", out, "")
    if readiness is not _MISSING:
        by_backend = _need_dict(readiness, "by_backend", out, "readiness.")
        if by_backend is not _MISSING:
            for backend in by_backend:
                if backend not in BACKENDS:
                    out.append(f"readiness.by_backend[{backend!r}]: not a known backend")
        # Invariant 14: readiness is computed from records, never narrated. This
        # field is the rendered form of a computed signal, so it is required to
        # be present, and it is NOT a trigger (§1.2).
        _need_str(readiness, "reference_signal", out, "readiness.", allow_empty=True)

    _need_sha256(obj, "affected_surface_union_sha256", out, "")
    _need_number(obj, "storage_gb", out, "", minimum=0)
    if "created_at" in obj:
        _need_timestamp(obj, "created_at", out, "")
    return out


# =============================================================================
# epyc.autokernel.release_package.v1 (§7.6, prose formalised)
# =============================================================================

def validate_release_package(obj: Any) -> list:
    """Validate a release package — what AutoKernel hands the operator.

    §7.6/§11.2: the packager may seal, evaluate, plan, draft, and pre-validate.
    It may not execute any command it drafted, touch any production branch,
    symlink, era registry, or baseline file, or waive failed evidence. The
    package therefore contains DRAFTS and a pre-validated command sequence, and
    it carries no authority claim — which the authority-key scan enforces.
    """
    out: list = []
    if not _check_schema_header(obj, SCHEMA_RELEASE_PACKAGE, out):
        return out
    _reject_authority_keys(obj, out)

    _need_id(obj, "package_id", out, "", "akr-")
    _need_id(obj, "campaign_id", out, "", "ak-")
    _need_str(obj, "source_tree", out, "", choices=SOURCE_TREES)

    sealed = _need_dict(obj, "sealed_candidate", out, "")
    if sealed is not _MISSING:
        candidate_id = _need_str(sealed, "candidate_id", out, "sealed_candidate.")
        if candidate_id is not _MISSING and not candidate_id.startswith("akc-"):
            out.append("sealed_candidate.candidate_id: must start with 'akc-'")
        for key in ("seal_sha256", "binary_sha256", "linkage_sha256",
                    "build_receipt_sha256"):
            _need_sha256(sealed, key, out, "sealed_candidate.")

    verdict_block = _need_dict(obj, "t3_verdict", out, "")
    verdict = None
    if verdict_block is not _MISSING:
        verdict = _need_str(verdict_block, "verdict", out, "t3_verdict.",
                            choices=T3_VERDICTS)
        _need_sha256(verdict_block, "bundle_sha256", out, "t3_verdict.")
        _need_dict(verdict_block, "phase_results", out, "t3_verdict.", non_empty=True)

    # §10.4: waivers are hash-pinned into the T3 bundle; the evaluator verifies
    # the hash and the predicate and never judges the waiver's merits.
    waivers = _need_list(obj, "active_waivers", out, "")
    if waivers is not _MISSING:
        for i, waiver in enumerate(waivers):
            if not isinstance(waiver, Mapping):
                out.append(f"active_waivers[{i}]: expected a mapping")
                continue
            _need_str(waiver, "waiver_id", out, f"active_waivers[{i}].")
            _need_sha256(waiver, "sha256", out, f"active_waivers[{i}].")
        if verdict == "PASS_WITH_WAIVER" and not waivers:
            out.append("active_waivers: PASS_WITH_WAIVER must pin at least one waiver")
        if verdict == "PASS" and waivers:
            out.append("t3_verdict.verdict: a package pinning active waivers is "
                       "PASS_WITH_WAIVER, not PASS (§10.4)")

    _need_dict(obj, "release_plan", out, "", non_empty=True)
    _need_dict(obj, "transaction_plan", out, "", non_empty=True)

    rollback = _need_dict(obj, "rollback_plan", out, "")
    if rollback is not _MISSING:
        # §10.5: rebuilding an old commit under a drifted toolchain does not
        # reproduce the incumbent binary, so the archive is the rollback target.
        _need_str(rollback, "incumbent_archive_path", out, "rollback_plan.")
        _need_sha256(rollback, "incumbent_binary_sha256", out, "rollback_plan.")

    # Drafts for operator execution (§1.3 items 2 and 3, §11.4). They are records
    # of what the operator would write, not writes.
    _need_dict(obj, "draft_era_registry_row", out, "", non_empty=True)
    _need_str(obj, "draft_autopilot_rebaseline_note", out, "")

    linkage = _need_dict(obj, "linkage_verification", out, "")
    if linkage is not _MISSING:
        # Three outcomes: an unverifiable linkage is not a verified one.
        status = _need_str(linkage, "status", out, "linkage_verification.",
                           choices={PASS, FAIL, COULD_NOT_CHECK})
        _need_str(linkage, "receipt", out, "linkage_verification.")
        if verdict in ("PASS", "PASS_WITH_WAIVER") and status != PASS:
            out.append(f"linkage_verification.status: {status!r} cannot accompany a "
                       f"{verdict} package — a binary that inherits another tree's "
                       "ggml runs silently wrong (CLAUDE.md speech-kernel freeze)")

    commands = _need_list(obj, "operator_command_sequence", out, "", non_empty=True)
    if commands is not _MISSING:
        for i, entry in enumerate(commands):
            prefix = f"operator_command_sequence[{i}]."
            if not isinstance(entry, Mapping):
                out.append(f"operator_command_sequence[{i}]: expected a mapping")
                continue
            _need_str(entry, "command", out, prefix)
            _need_str(entry, "validation_receipt", out, prefix)
            # MEASUREMENT.md:138-145 requires every operator command to be
            # pre-validated end-to-end before it is handed over.
            _need_bool(entry, "validated", out, prefix, must_be=True)

    classes = _need_list(obj, "change_classes", out, "", item_type=str,
                         item_desc="a change class")
    if classes is not _MISSING:
        for change_class in classes:
            if isinstance(change_class, str) and change_class not in CHANGE_CLASSES:
                out.append(f"change_classes: {change_class!r} is not a known change class")

    review = _need_bool(obj, "requires_human_code_review", out, "")
    complexity = _need_dict(obj, "diff_complexity", out, "")
    touches_core = None
    if complexity is not _MISSING:
        _need_int(complexity, "diff_size", out, "diff_complexity.", minimum=0)
        _need_int(complexity, "files_touched", out, "diff_complexity.", minimum=0)
        touches_core = _need_bool(complexity, "touches_shared_core", out,
                                  "diff_complexity.")
    # §8.5.1: a core-header change is a different KIND of edit, not a large one;
    # §10.6: LLM-authored kernel C++/HIP does not reach a release package
    # unreviewed above the backend's blast-radius ceiling.
    if review is False:
        if classes is not _MISSING and "core_header" in classes:
            out.append("requires_human_code_review: must be true when a member change "
                       "class is 'core_header' (§8.5.1)")
        if touches_core is True:
            out.append("requires_human_code_review: must be true when the diff touches "
                       "shared ggml core (§10.6)")

    _need_timestamp(obj, "created_at", out, "")
    return out


# =============================================================================
# epyc.autokernel.operator_waiver.v1 (§10.4)
# =============================================================================

def validate_operator_waiver(obj: Any, *, document_path: Any = _MISSING,
                             boundary: Optional[TrustBoundary] = None) -> list:
    """Validate an operator waiver — human-authored, first-class T3 input.

    A binary PASS/FAIL gate would have blocked v8: its ratification recorded
    `promotion_decision: false` as "a non-automatic matrix verdict" released as
    "an operator-attested release decision". The generalisation carries scope,
    reason, forfeited claims, protocol binding, campaign binding, and an
    expiry/reopen predicate.

    Deliberately NOT scanned for authority-flavoured keys: this is the one
    human-authored record in the set, written under the trust-boundary path set,
    and an operator attestation is exactly the thing the machine records may not
    contain. The evaluator verifies its hash and predicate; it never judges its
    merits (§10.4).

    TWO checks here are about WHO wrote it, not what it says, and neither is a
    merits judgement:

      * `authorized_by` may not name a machine actor. §10.4 makes the waiver
        human-authored by definition, so a document the loop attributed to itself
        is not a waiver with a bad author, it is not a waiver.
      * `document_path`, WHEN THE CALLER KNOWS IT, must resolve under an
        operator-owned path (`operator_owned_path_check`). The path is not carried
        by the record — it is a fact about where the record was read from — so it
        arrives as a keyword. A caller that omits it (journal replay, registry
        dispatch, a record quoted inside another document) validates the document
        and makes no claim about its provenance; the gate that acts on a waiver
        (`t3.verify_waiver`) always knows the path and always checks it.

    A COULD_NOT_CHECK from the path check is reported as a violation here on
    purpose: this function has two states, and a provenance it cannot establish is
    not one it may assume. Callers that need the third state call
    `operator_owned_path_check` directly.
    """
    out: list = []
    if not _check_schema_header(obj, SCHEMA_OPERATOR_WAIVER, out):
        return out

    if document_path is not _MISSING:
        located = operator_owned_path_check(document_path, boundary=boundary)
        if located.outcome != PASS:
            out.extend(located.reasons)

    _need_str(obj, "waiver_id", out, "")
    _need_id(obj, "campaign_id", out, "", "ak-")
    _need_str(obj, "decision", out, "")
    _need_str(obj, "protocol", out, "")
    # The v8 waiver recorded whether the protocol itself changed; a waiver bound
    # to a protocol that moved underneath it is not the waiver that was granted.
    _need_bool(obj, "protocol_changed", out, "")
    _need_commit(obj, "candidate_head", out, "")
    _need_commit(obj, "production_head", out, "")

    scope = _need_dict(obj, "scope", out, "")
    if scope is not _MISSING:
        _need_list(scope, "excluded_models", out, "scope.", item_type=str,
                   item_desc="a model name")
        _need_list(scope, "excluded_pairs", out, "scope.")
        _need_int(scope, "remaining_matched_pairs", out, "scope.", minimum=0)

    _need_str(obj, "reason", out, "")
    # A waiver that forfeits nothing is an approval, not a waiver: the v8
    # precedent names the forfeited claim explicitly ("v8 makes no Q8
    # non-regression claim").
    _need_list(obj, "consequences", out, "", non_empty=True, item_type=str,
               item_desc="a forfeited claim")
    _need_str(obj, "authorized_by", out, "")
    for field_name, identity, tokens in machine_attributions(obj):
        out.append(
            f"{field_name}: {identity!r} names a machine actor "
            f"({', '.join(tokens)}). §10.4 waivers are human-authored "
            "(MEASUREMENT.md:140-142); a waiver the loop attributed to itself is "
            "the loop excusing its own failing cell.")

    expiry = _need_dict(obj, "expiry", out, "")
    if expiry is not _MISSING:
        expires_at = expiry.get("expires_at")
        predicate = expiry.get("reopen_predicate")
        if expires_at is not None:
            _need_timestamp(expiry, "expires_at", out, "expiry.")
        has_predicate = isinstance(predicate, str) and bool(predicate.strip())
        if expires_at is None and not has_predicate:
            out.append("expiry: a waiver needs an expiry timestamp or a reopen "
                       "predicate; an unbounded waiver never reopens the question")

    _need_timestamp(obj, "created_at", out, "")
    return out


# =============================================================================
# epyc.autokernel.kernel_dashboard.v1 / .v2 — the /kernel operator surface (AK6)
# =============================================================================
#
# WHY THIS CONTRACT IS SHAPED THE WAY IT IS
# -----------------------------------------
# The predecessor surface is the scar this schema exists to close. The hub's
# `/kernel` page reads one JSON file and is ABSENCE-TOLERANT OVER A MISSING
# DIRECTORY: when the producer is dead the page renders clean, which is exactly
# the shape of AutoPilot dying at trial 1302 and staying dead ~23 hours with
# every dashboard green. A panel that cannot distinguish "nothing is wrong" from
# "nobody is reporting" is worse than no panel, because it is trusted.
#
# Absence tolerance is still REQUIRED — the page must not crash on a missing
# producer — so the fix is not to make absence fatal, it is to make absence
# UNREPRESENTABLE AS SILENCE. Three structural rules do that, and each of them is
# a validation rule here rather than a convention in the producer:
#
#   1. **Every section is mandatory, and carries its own status.** A producer
#      cannot drop a section whose owner is dead; it must emit the section with
#      `status: "not_reported"`. A document missing a section key is INVALID, so
#      "the field is absent" is never a renderable state.
#   2. **`degraded` and `unreported_sections` are DERIVED from the sections, and
#      the validator recomputes them.** A document that claims to be healthy while
#      carrying an unreported section is refused — the same trick
#      `ReleasePackage.state` uses against `_derive_package_state`, for the same
#      reason: a summary that can be stamped independently of its evidence can be
#      wrong in the direction nobody notices.
#   3. **`produced_at` is derived from the LOOP's own record timestamps, never
#      from the export.** `dashboard_liveness_timestamp()` is the single source of
#      truth for it and the validator refuses any other value. Two consequences,
#      both deliberate:
#        * a no-op re-export cannot read as fresh — nothing the exporter does
#          moves a journaled record's timestamp (the property `server.py`'s
#          `_kernel_contract_freshness` already reaches for with "from semantic
#          run timestamps, not file mtime"); and
#        * live HOST observations — free disk, held device claims — are excluded
#          from the computation (`DASHBOARD_LIVENESS_SECTIONS`). They are measured
#          by the exporter itself, so counting them would let a surface process
#          that is merely alive manufacture freshness for a controller that is
#          dead. That is the 23-hour failure re-implemented one layer up.
#
# `exported_at` carries the wall clock and is NOT a freshness source. It is named
# differently from `produced_at` on purpose: two fields that mean different things
# must not be spellable the same way.

DASHBOARD_SECTION_CAMPAIGN = "campaign"
DASHBOARD_SECTION_CHAMPION = "champion"
DASHBOARD_SECTION_BACKEND_STANDING = "backend_standing"
DASHBOARD_SECTION_HEADROOM = "headroom"
DASHBOARD_SECTION_BLOCKING = "blocking_conditions"
DASHBOARD_SECTION_CLAIMS = "resource_claims"
DASHBOARD_SECTION_RELEASE_PACKAGE = "release_package"

#: Every section the v2 contract carries. EXACT, not minimum: an unknown section
#: key is a violation, because a consumer that renders only what it recognises
#: silently drops the one panel a future producer added.
DASHBOARD_SECTIONS = (
    DASHBOARD_SECTION_CAMPAIGN,
    DASHBOARD_SECTION_CHAMPION,
    DASHBOARD_SECTION_BACKEND_STANDING,
    DASHBOARD_SECTION_HEADROOM,
    DASHBOARD_SECTION_BLOCKING,
    DASHBOARD_SECTION_CLAIMS,
    DASHBOARD_SECTION_RELEASE_PACKAGE,
)

#: The sections whose `as_of` is a JOURNALED RECORD's timestamp — written by the
#: loop, not by the exporter — and therefore the only ones that may establish
#: liveness. Three exclusions, all for the same reason and each one load-bearing:
#:
#:   * `headroom` and `resource_claims` are live host readings taken BY the
#:     exporter. They are fresh whenever the exporter runs, so admitting them
#:     would mean a surface process could report a dead controller as fresh.
#:   * `blocking_conditions` is a DERIVED section: every condition in it is
#:     restated from the campaign, backend-standing, champion, headroom and
#:     release-package sections, so it can only ever INHERIT freshness — it owns
#:     no record of its own to establish it with. It was in this tuple until an
#:     adversarial pass found the consequence: its `as_of` was the one timestamp
#:     in the liveness set that no journaled record produced (the caller handed it
#:     to `derive_blocking_conditions`), so an exporter that stamped its own clock
#:     there made a month-dead controller render `produced_at: now`, `degraded:
#:     false` — the 23-hour AutoPilot outage with a green dashboard over it,
#:     rebuilt inside the fix for it. The producer now derives that `as_of` from
#:     the conditions' own record timestamps as well; both guards stand, because
#:     the class of defect is "one section's timestamp comes from the exporter"
#:     and a single guard against it is a single point of failure.
DASHBOARD_LIVENESS_SECTIONS = (
    DASHBOARD_SECTION_CAMPAIGN,
    DASHBOARD_SECTION_CHAMPION,
    DASHBOARD_SECTION_BACKEND_STANDING,
    DASHBOARD_SECTION_RELEASE_PACKAGE,
)

#: Three section statuses, never two. `not_reported` is the whole point: it is
#: what a dead producer looks like, and it is a value, not an omission.
SECTION_OBSERVED = "observed"
SECTION_NOT_REPORTED = "not_reported"
SECTION_REFUSED = "refused"
DASHBOARD_SECTION_STATUSES = frozenset({
    SECTION_OBSERVED, SECTION_NOT_REPORTED, SECTION_REFUSED,
})

#: Which owning module reported an open blocking condition. The CONDITION NAMES
#: themselves (`EVALUATOR_COVERAGE_GAP`, `ANCHOR_MOVED`, the phase-trade statuses)
#: are deliberately NOT enumerated here: they are owned by
#: `controller.state_machine.STOP_STATES` and `release.readiness.BLOCKERS`, which
#: import this module, so restating them would create the second source of truth
#: this contract exists to avoid. The producer binds each `kind` to the owner's
#: own constant and `test_dashboard_contract.py` proves the emitted kinds are a
#: subset of the owners' vocabularies.
DASHBOARD_BLOCKING_ORIGINS = frozenset({
    "controller_stop", "readiness", "phase_trade", "evaluator_coverage",
    "storage", "release_package", "champion",
})

_BLOCKING_KIND_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _parse_ts(value: Any):
    """An aware `datetime` for an ISO-8601 string, or None. Never raises."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def dashboard_liveness_timestamp(sections: Any) -> Optional[str]:
    """The contract's semantic `produced_at`, derived from the loop's records.

    THE single source of truth for the field: the producer calls it to fill
    `produced_at` and `validate_kernel_dashboard_v2` calls it to refuse any other
    value, so the two cannot drift and no caller can hand-stamp a fresh time onto
    stale evidence.

    Returns the newest `as_of` among the OBSERVED liveness sections, or `None`
    when none of them reported. `None` is the correct, loud answer: every consumer
    freshness classifier in this project reads a missing timestamp as `missing`
    rather than as healthy, so a fully dead loop renders as absence.
    """
    newest_raw = None
    newest_dt = None
    if not isinstance(sections, Mapping):
        return None
    for name in DASHBOARD_LIVENESS_SECTIONS:
        section = sections.get(name)
        if not isinstance(section, Mapping):
            continue
        if section.get("status") != SECTION_OBSERVED:
            continue
        parsed = _parse_ts(section.get("as_of"))
        if parsed is None:
            continue
        if newest_dt is None or parsed > newest_dt:
            newest_dt, newest_raw = parsed, section.get("as_of")
    return newest_raw


def dashboard_unreported_sections(sections: Any) -> list:
    """Sorted names of the sections whose owner did not report.

    A section that is structurally ABSENT counts as unreported, so this function
    gives the same answer for "the producer omitted the key" and "the producer
    said `not_reported`" — a consumer must never have to know the difference, and
    the validator refuses the omission separately.
    """
    out = []
    if not isinstance(sections, Mapping):
        return sorted(DASHBOARD_SECTIONS)
    for name in DASHBOARD_SECTIONS:
        section = sections.get(name)
        if not isinstance(section, Mapping):
            out.append(name)
        elif section.get("status") != SECTION_OBSERVED:
            out.append(name)
    return sorted(out)


def _check_dashboard_section(section: Any, name: str, out: list) -> None:
    prefix = f"sections.{name}."
    if not isinstance(section, Mapping):
        out.append(f"sections.{name}: required section is missing or not a mapping "
                   f"(a dead producer emits {SECTION_NOT_REPORTED!r}, never nothing — "
                   f"an omitted section renders as silence)")
        return
    status = _need_str(section, "status", out, prefix,
                       choices=DASHBOARD_SECTION_STATUSES)
    as_of = _fetch(section, "as_of", out, prefix)
    if as_of is not _MISSING and as_of is not None:
        _need_timestamp(section, "as_of", out, prefix)
    if status == SECTION_OBSERVED and name in DASHBOARD_LIVENESS_SECTIONS:
        if as_of is _MISSING or as_of is None:
            out.append(f"sections.{name}.as_of: an observed liveness section must "
                       f"carry the record timestamp it observed; without it the "
                       f"section contributes nothing to `produced_at` while still "
                       f"reading as healthy")
    if status in (SECTION_NOT_REPORTED, SECTION_REFUSED):
        # Absence must be EXPLAINED, not merely flagged. "not_reported" with no
        # reason is the same dead panel with a new label on it.
        _need_str(section, "reason", out, prefix)
        if as_of is not _MISSING and as_of is not None:
            out.append(f"sections.{name}.as_of: must be null when status is "
                       f"{status!r} — a section nobody reported has no observation "
                       f"time, and a timestamp here would lift `produced_at`")


def validate_kernel_dashboard_v2(obj: Any) -> list:
    """Validate the v2 `/kernel` dashboard contract (AK6, §7 conventions)."""
    out: list = []
    if not _check_schema_header(obj, SCHEMA_KERNEL_DASHBOARD_V2, out):
        return out
    _reject_authority_keys(obj, out)

    version = _need_int(obj, "contract_version", out, "", minimum=2, maximum=2)
    del version
    _need_id(obj, "campaign_id", out, "", "ak-")
    _need_str(obj, "observation_notice", out, "")
    # The wall clock. Required, and required to be USELESS for freshness: it is
    # here so an operator can see when the file was last touched, next to a
    # `produced_at` that a touch cannot move.
    _need_timestamp(obj, "exported_at", out, "")

    producer = _need_dict(obj, "producer", out, "")
    if producer is not _MISSING:
        _need_str(producer, "module_id", out, "producer.", pattern=_VERSIONED_ID_RE,
                  pattern_hint="is not a versioned module id ('<name>/v<n>')")
        run = _fetch(producer, "run", out, "producer.")
        if run is not _MISSING and run is not None:
            if not isinstance(run, Mapping):
                out.append("producer.run: expected null (no producing run could be "
                           f"identified) or a mapping, got {type(run).__name__}")
            else:
                # Run identity, so a consumer can tell two exports apart WITHOUT
                # the filesystem: same campaign + same controller sequence means
                # the loop did not advance between them.
                _need_id(run, "campaign_id", out, "producer.run.", "ak-")
                _need_int(run, "controller_seq", out, "producer.run.", minimum=0)
                _need_str(run, "controller_state", out, "producer.run.")
                _need_str(run, "ledger_receipt", out, "producer.run.")

    sections = _need_dict(obj, "sections", out, "")
    if sections is not _MISSING:
        for name in DASHBOARD_SECTIONS:
            _check_dashboard_section(sections.get(name), name, out)
        for name in sections:
            if name not in DASHBOARD_SECTIONS:
                out.append(f"sections.{name!r}: not a known dashboard section; "
                           f"known sections are {sorted(DASHBOARD_SECTIONS)}")

    blocking = sections.get(DASHBOARD_SECTION_BLOCKING) \
        if isinstance(sections, Mapping) else None
    if isinstance(blocking, Mapping) and blocking.get("status") == SECTION_OBSERVED:
        conditions = _need_list(blocking, "open", out,
                                f"sections.{DASHBOARD_SECTION_BLOCKING}.")
        if conditions is not _MISSING:
            for i, entry in enumerate(conditions):
                prefix = f"sections.{DASHBOARD_SECTION_BLOCKING}.open[{i}]."
                if not isinstance(entry, Mapping):
                    out.append(f"{prefix.rstrip('.')}: expected a mapping")
                    continue
                _need_str(entry, "kind", out, prefix, pattern=_BLOCKING_KIND_RE,
                          pattern_hint="is not an UPPER_SNAKE condition name")
                _need_str(entry, "origin", out, prefix,
                          choices=DASHBOARD_BLOCKING_ORIGINS)
                _need_str(entry, "detail", out, prefix)

    # -- the two derived summaries, recomputed rather than trusted -------------
    produced_at = _fetch(obj, "produced_at", out, "")
    if produced_at is not _MISSING:
        expected = dashboard_liveness_timestamp(
            sections if sections is not _MISSING else None)
        if produced_at is not None:
            _need_timestamp(obj, "produced_at", out, "")
        if produced_at != expected:
            out.append(
                f"produced_at: {produced_at!r} is not the value its own sections "
                f"yield ({expected!r}). `produced_at` is DERIVED from the loop's "
                "journaled record timestamps (dashboard_liveness_timestamp); a "
                "hand-stamped one lets a live exporter report a dead controller "
                "as fresh, which is the 23-hour AutoPilot outage with a green "
                "dashboard over it.")

    # v1 compatibility: the deployed hub reads `generated_at`. It must carry the
    # SAME semantic value, so an old reader pointed at a v2 file gets the derived
    # timestamp and not the export time.
    generated_at = _fetch(obj, "generated_at", out, "")
    if generated_at is not _MISSING and produced_at is not _MISSING:
        if generated_at != produced_at:
            out.append(
                f"generated_at: {generated_at!r} must equal produced_at "
                f"({produced_at!r}) — it is the v1 spelling of the same semantic "
                "timestamp, and a v1 reader given the export time would classify a "
                "dead loop as fresh")

    degraded = _need_bool(obj, "degraded", out, "")
    unreported = _fetch(obj, "unreported_sections", out, "")
    expected_unreported = dashboard_unreported_sections(
        sections if sections is not _MISSING else None)
    if unreported is not _MISSING:
        if not isinstance(unreported, list):
            out.append("unreported_sections: expected a list, got "
                       f"{type(unreported).__name__}")
        elif sorted(unreported) != expected_unreported:
            out.append(
                f"unreported_sections: {sorted(unreported)!r} disagrees with the "
                f"sections themselves ({expected_unreported!r}); the summary a "
                "consumer reads first may not be stampable independently of the "
                "evidence under it")
    if degraded is not _MISSING and degraded is not bool(expected_unreported):
        out.append(
            f"degraded: {degraded!r} disagrees with the sections themselves "
            f"(unreported: {expected_unreported!r}). A panel that renders clean "
            "while its producer is dead is worse than no panel, because it is "
            "trusted.")
    return out


def validate_kernel_dashboard_v1(obj: Any) -> list:
    """Validate the LEGACY `/kernel` contract — read-only, and kept readable.

    v1 is not ours to re-shape: files in this shape were written by an earlier
    exporter and the hub still reads them, so the rules here are exactly the ones
    a consumer depends on and nothing more. In particular v1 records carry NO
    `schema` key (see `detect_kernel_dashboard_version`), so this validator must
    not require one — demanding the label would make every real v1 file invalid
    and push a reader toward "unknown, render empty", which is the absence-tolerant
    failure again.

    v1 is NOT emitted by anything in this package. It exists here so a consumer
    that meets both versions has one validator per version rather than a guess.
    """
    out: list = []
    if not isinstance(obj, Mapping):
        return [f"record: expected a mapping, got {type(obj).__name__}"]
    schema = obj.get("schema")
    if schema is not None and schema != SCHEMA_KERNEL_DASHBOARD_V1:
        out.append(f"schema: expected {SCHEMA_KERNEL_DASHBOARD_V1!r} or no schema "
                   f"key at all (legacy exports are unlabelled), got {schema!r}")
    generated_at = obj.get("generated_at", _MISSING)
    if generated_at not in (_MISSING, None) and _parse_ts(generated_at) is None:
        out.append(f"generated_at: {generated_at!r} is not a timezone-aware "
                   "ISO-8601 timestamp")
    runs = obj.get("runs", _MISSING)
    if runs is not _MISSING:
        if not isinstance(runs, list):
            out.append(f"runs: expected a list, got {type(runs).__name__}")
        else:
            for i, run in enumerate(runs):
                if not isinstance(run, Mapping):
                    out.append(f"runs[{i}]: expected a mapping")
                    continue
                ts = run.get("ts", _MISSING)
                if ts not in (_MISSING, None) and _parse_ts(ts) is None:
                    out.append(f"runs[{i}].ts: {ts!r} is not a timezone-aware "
                               "ISO-8601 timestamp")
    for key in ("pareto", "best_per_model"):
        value = obj.get(key, _MISSING)
        if value is not _MISSING and not isinstance(value, list):
            out.append(f"{key}: expected a list, got {type(value).__name__}")
    totals = obj.get("totals", _MISSING)
    if totals is not _MISSING and not isinstance(totals, Mapping):
        out.append(f"totals: expected a mapping, got {type(totals).__name__}")
    return out


def detect_kernel_dashboard_version(obj: Any) -> Optional[str]:
    """The dashboard schema string for `obj`, or None when it is not one.

    A labelled document is taken at its label. An UNLABELLED one is classified by
    the v1 shape markers, because that is the only thing a legacy file carries —
    the whole reason the version is being made explicit now is that v1 never was.
    Returning None is a real answer and must be reported as "unrecognised", never
    coerced to v1: a malformed v2 file misread as a valid v1 file would render as
    an empty-but-clean panel.
    """
    if not isinstance(obj, Mapping):
        return None
    schema = obj.get("schema")
    if schema in (SCHEMA_KERNEL_DASHBOARD_V1, SCHEMA_KERNEL_DASHBOARD_V2):
        return schema
    if schema is not None:
        return None
    markers = ("db_present", "runs", "pareto", "best_per_model", "totals")
    if sum(1 for key in markers if key in obj) >= 2:
        return SCHEMA_KERNEL_DASHBOARD_V1
    return None


def validate_kernel_dashboard(obj: Any) -> list:
    """Validate a dashboard document under whichever version it is."""
    version = detect_kernel_dashboard_version(obj)
    if version is None:
        return ["schema: not a recognisable AutoKernel kernel-dashboard document "
                f"(expected {SCHEMA_KERNEL_DASHBOARD_V2!r}, or the unlabelled v1 "
                "shape)"]
    return KERNEL_DASHBOARD_VALIDATORS[version](obj)


KERNEL_DASHBOARD_VALIDATORS = {
    SCHEMA_KERNEL_DASHBOARD_V1: validate_kernel_dashboard_v1,
    SCHEMA_KERNEL_DASHBOARD_V2: validate_kernel_dashboard_v2,
}


# =============================================================================
# Registry and dispatch
# =============================================================================

SCHEMA_REGISTRY = {
    SCHEMA_CAMPAIGN: validate_campaign,
    SCHEMA_PROPOSAL: validate_proposal,
    SCHEMA_CANDIDATE: validate_candidate,
    # Both evaluation-event versions are registered. v2 is not retired and is not
    # rewritten: a journal shard written last week still validates, under its own
    # rules, forever.
    SCHEMA_EVALUATION_EVENT_V2: validate_evaluation_event_v2,
    SCHEMA_EVALUATION_EVENT_V3: validate_evaluation_event_v3,
    SCHEMA_CHAMPION: validate_champion,
    SCHEMA_RELEASE_PACKAGE: validate_release_package,
    SCHEMA_OPERATOR_WAIVER: validate_operator_waiver,
}
# The kernel-dashboard schemas are DELIBERATELY absent from this registry, and the
# omission is a boundary rather than an oversight. `SCHEMA_REGISTRY` is the
# JOURNAL-RECORD registry: `validate_record` dispatches journal lines through it,
# so anything registered here is something a journal may contain. The dashboard
# contract is a DERIVED EXPORT of those records, not one of them — appending one to
# a journal would put a rendering where evidence belongs. It also could not be
# dispatched here honestly: legacy v1 documents carry no `schema` key at all, which
# `validate_record` requires, so version detection needs
# `detect_kernel_dashboard_version` and dispatch needs
# `KERNEL_DASHBOARD_VALIDATORS`. Both live beside the validators above.

KNOWN_SCHEMAS = frozenset(SCHEMA_REGISTRY)


def validate_record(obj: Any) -> list:
    """Dispatch on the record's `schema` string and validate it.

    An unrecognised schema is a VIOLATION of the record, not an exception: a
    journal reader must be able to report an unreadable line and keep reading
    the rest of the shard. It is never treated as valid.
    """
    if not isinstance(obj, Mapping):
        return [f"record: expected a mapping, got {type(obj).__name__}"]
    schema = obj.get("schema")
    if schema is None:
        return ["schema: required field is missing "
                "(the version string is the record's identity)"]
    validator = SCHEMA_REGISTRY.get(schema)
    if validator is None:
        return [f"schema: {schema!r} is not a known AutoKernel schema; "
                f"known schemas are {sorted(KNOWN_SCHEMAS)}"]
    return validator(obj)


def is_valid(obj: Any) -> bool:
    """True only when dispatch finds a validator AND it reports no violations."""
    return not validate_record(obj)


__all__ = [
    "SCHEMA_CAMPAIGN", "SCHEMA_PROPOSAL", "SCHEMA_CANDIDATE",
    "SCHEMA_EVALUATION_EVENT", "SCHEMA_EVALUATION_EVENT_V2",
    "SCHEMA_EVALUATION_EVENT_V3", "EVALUATION_EVENT_VALIDATORS",
    "SCHEMA_CHAMPION", "SCHEMA_RELEASE_PACKAGE",
    "SCHEMA_OPERATOR_WAIVER", "SCHEMA_REGISTRY", "KNOWN_SCHEMAS",
    "BACKENDS", "SOURCE_TREES", "SOURCE_TREE_BY_BACKEND", "OBJECTIVE_RULES",
    "LLAMA_PHASES", "PHASES_BY_BACKEND", "RECIPE_CLASSES", "CHANGE_CLASSES",
    "CHANGE_CLASS_CHEAP_SUITE", "CAMPAIGN_KINDS", "RESOURCE_LANES",
    "CRITIC_STATUSES", "TIERS", "CLAIM_CATEGORIES", "METRIC_DIRECTIONS",
    "EVENT_STATUSES", "DETERMINISM_CLASSES", "MACHINE_SUBSETS",
    "DURABILITY_CLASSES", "CANDIDATE_STATUSES", "CHAMPION_STATUSES",
    "T3_VERDICTS", "NON_RETRIEVABLE_FIELDS", "ANCHOR_VOID_REASONS",
    "VOID_FLAG_PREFIX",
    "MACHINE_ACTOR_TOKENS", "ACTOR_ATTRIBUTION_FIELDS",
    "HUMAN_ONLY_PATHS_MANIFEST", "OPERATOR_ATTESTATION_ROOT", "REPO_CHECKOUT_ROOTS",
    "REPO_CHECKOUT_NAMES", "MAX_OPERATOR_WAIVER_BYTES",
    "TrustBoundary", "parse_trust_boundary", "repo_relative_forms",
    "canonical_citation", "under_any_root", "raw_bytes_digest",
    "machine_actor_tokens", "machine_attributions", "attribution_keys",
    "operator_owned_path_check",
    "canonical_json", "canonical_bytes", "content_hash", "retrievable_view",
    "candidate_natural_key", "find_authority_flavoured_keys",
    "is_placeholder_digest", "declared_anchor_void_reasons",
    "require", "EVIDENCE_PRODUCERS", "SHA256_RE", "COMMIT_RE",
    "validate_campaign", "validate_proposal", "validate_candidate",
    "validate_evaluation_event", "validate_evaluation_event_v2",
    "validate_evaluation_event_v3",
    "validate_champion", "CHAMPION_BLOCKED_UNNAMED", "validate_release_package",
    "validate_operator_waiver", "validate_record", "is_valid",
    "SCHEMA_KERNEL_DASHBOARD", "SCHEMA_KERNEL_DASHBOARD_V1",
    "SCHEMA_KERNEL_DASHBOARD_V2", "KERNEL_DASHBOARD_VALIDATORS",
    "DASHBOARD_SECTIONS", "DASHBOARD_LIVENESS_SECTIONS",
    "DASHBOARD_SECTION_CAMPAIGN", "DASHBOARD_SECTION_CHAMPION",
    "DASHBOARD_SECTION_BACKEND_STANDING", "DASHBOARD_SECTION_HEADROOM",
    "DASHBOARD_SECTION_BLOCKING", "DASHBOARD_SECTION_CLAIMS",
    "DASHBOARD_SECTION_RELEASE_PACKAGE", "DASHBOARD_SECTION_STATUSES",
    "SECTION_OBSERVED", "SECTION_NOT_REPORTED", "SECTION_REFUSED",
    "DASHBOARD_BLOCKING_ORIGINS", "dashboard_liveness_timestamp",
    "dashboard_unreported_sections", "detect_kernel_dashboard_version",
    "validate_kernel_dashboard", "validate_kernel_dashboard_v1",
    "validate_kernel_dashboard_v2",
    "Check", "PASS", "FAIL", "COULD_NOT_CHECK",
    "OUTCOME_SEVERITY", "EMPTY_CHECK_VECTOR_REASON",
    "check_scope_denominator_admits_gate", "check_anchor_binding",
    "check_metric_commensurability",
]
