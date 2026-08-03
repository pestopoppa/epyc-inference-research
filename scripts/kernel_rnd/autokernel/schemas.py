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
    §2.2). Here `anchor` (binary_sha256, linkage_sha256, measurement_event_ids)
    is REQUIRED; a no-anchor comparison is `invalid`, never correct. §8.9's
    `ANCHOR_MOVED` rests entirely on that binding being recorded per event.

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
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

# =============================================================================
# Schema identity — the version is part of the name, never metadata beside it
# =============================================================================

SCHEMA_CAMPAIGN = "epyc.autokernel.campaign.v2"
SCHEMA_PROPOSAL = "epyc.autokernel.proposal.v2"
SCHEMA_CANDIDATE = "epyc.autokernel.candidate.v1"
SCHEMA_EVALUATION_EVENT = "epyc.autokernel.evaluation_event.v2"
SCHEMA_CHAMPION = "epyc.autokernel.champion.v1"
SCHEMA_RELEASE_PACKAGE = "epyc.autokernel.release_package.v1"
SCHEMA_OPERATOR_WAIVER = "epyc.autokernel.operator_waiver.v1"


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
    SCHEMA_EVALUATION_EVENT: frozenset({"narrative"}),
    SCHEMA_CHAMPION: frozenset(),
    SCHEMA_RELEASE_PACKAGE: frozenset(),
    SCHEMA_OPERATOR_WAIVER: frozenset(),
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
# epyc.autokernel.evaluation_event.v2 (§7.4)
# =============================================================================

def validate_evaluation_event(obj: Any) -> list:
    """Validate an evaluation event — the decision-grade unit of evidence."""
    out: list = []
    if not _check_schema_header(obj, SCHEMA_EVALUATION_EVENT, out):
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
    anchor = _need_dict(obj, "anchor", out, "")
    if anchor is not _MISSING:
        _need_sha256(anchor, "binary_sha256", out, "anchor.")
        _need_sha256(anchor, "linkage_sha256", out, "anchor.")
        ids = _need_list(anchor, "measurement_event_ids", out, "anchor.",
                         item_type=str, item_desc="an event id")
        if ids is not _MISSING and not ids and tier != "T0":
            out.append("anchor.measurement_event_ids: must name at least one anchor "
                       f"measurement for tier {tier!r} (only T0 compares artifacts "
                       "rather than rates)")

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


# =============================================================================
# epyc.autokernel.champion.v1 (§7.5)
# =============================================================================

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

def validate_operator_waiver(obj: Any) -> list:
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
    """
    out: list = []
    if not _check_schema_header(obj, SCHEMA_OPERATOR_WAIVER, out):
        return out

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
# Registry and dispatch
# =============================================================================

SCHEMA_REGISTRY = {
    SCHEMA_CAMPAIGN: validate_campaign,
    SCHEMA_PROPOSAL: validate_proposal,
    SCHEMA_CANDIDATE: validate_candidate,
    SCHEMA_EVALUATION_EVENT: validate_evaluation_event,
    SCHEMA_CHAMPION: validate_champion,
    SCHEMA_RELEASE_PACKAGE: validate_release_package,
    SCHEMA_OPERATOR_WAIVER: validate_operator_waiver,
}

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
    "SCHEMA_EVALUATION_EVENT", "SCHEMA_CHAMPION", "SCHEMA_RELEASE_PACKAGE",
    "SCHEMA_OPERATOR_WAIVER", "SCHEMA_REGISTRY", "KNOWN_SCHEMAS",
    "BACKENDS", "SOURCE_TREES", "SOURCE_TREE_BY_BACKEND", "OBJECTIVE_RULES",
    "LLAMA_PHASES", "PHASES_BY_BACKEND", "RECIPE_CLASSES", "CHANGE_CLASSES",
    "CHANGE_CLASS_CHEAP_SUITE", "CAMPAIGN_KINDS", "RESOURCE_LANES",
    "CRITIC_STATUSES", "TIERS", "CLAIM_CATEGORIES", "METRIC_DIRECTIONS",
    "EVENT_STATUSES", "DETERMINISM_CLASSES", "MACHINE_SUBSETS",
    "DURABILITY_CLASSES", "CANDIDATE_STATUSES", "CHAMPION_STATUSES",
    "T3_VERDICTS", "NON_RETRIEVABLE_FIELDS",
    "canonical_json", "canonical_bytes", "content_hash", "retrievable_view",
    "candidate_natural_key", "find_authority_flavoured_keys",
    "validate_campaign", "validate_proposal", "validate_candidate",
    "validate_evaluation_event", "validate_champion", "validate_release_package",
    "validate_operator_waiver", "validate_record", "is_valid",
    "Check", "PASS", "FAIL", "COULD_NOT_CHECK",
    "check_scope_denominator_admits_gate", "check_anchor_binding",
    "check_metric_commensurability",
]
