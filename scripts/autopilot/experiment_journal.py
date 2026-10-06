"""Experiment journal: dual TSV + JSONL logging for AutoPilot trials.

Append-only with rotation (new file per 1000 trials).
"""

from __future__ import annotations

import csv
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.autopilot_core.tier_specs import DEFAULT_FRONTIER_TIER

# experiment_journal is imported both as the package path
# ``scripts.autopilot.experiment_journal`` (tests, cross-module readers) and as a
# bare ``experiment_journal`` module (autopilot.py inserts scripts/autopilot on
# sys.path). Resolve the shard iterator under both contexts.
try:
    from scripts.autopilot.journal_shards import journal_shards, shard_batch_index
except ModuleNotFoundError:  # pragma: no cover - bare-module import context
    from journal_shards import journal_shards, shard_batch_index

log = logging.getLogger(__name__)


class ExperimentJournalCorruptError(RuntimeError):
    """Raised when a journal shard has irreparable mid-file corruption.

    A torn *trailing* line (a partial final append left by a crash mid-write)
    is tolerated on read and quarantined to a `.corrupt-*` sidecar on the next
    append — see ``ExperimentJournal._repair_torn_tail``. This error signals the
    other case: a malformed line that has *valid* lines after it, which cannot
    be auto-repaired without risking data loss.

    Remediation: inspect the offending shard, then use
    ``scripts/autopilot/scrub_journal.py`` (append-only supersession events) to
    correct the record set. Never hand-edit historical trial rows in place.
    """


class DeficiencyCategory(str, Enum):
    """Structured failure classification for safety gate violations (AP-14).

    Each category maps to a specific SafetyGate check or dispatch_action guard.
    Using str mixin for natural JSON serialization in JSONL journal.
    """
    QUALITY_FLOOR = "quality_floor"
    REGRESSION = "regression"
    PER_SUITE = "per_suite_regression"
    ROUTING_DIVERSITY = "routing_diversity"
    THROUGHPUT = "throughput"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    CODE_VALIDATION = "code_validation"
    SHRINKAGE = "shrinkage"
    REVERT = "revert"
    # 2026-06-04 non-executing-action residue (graph_router deadlock fix). An
    # action that never ran an eval — so it has no quality/speed evidence — but
    # whose failure reason is actionable and MUST be fed back to the planner
    # instead of being silently dropped (the "return None, increment, continue"
    # blind spot that let 119 identical invalid structural_experiments dispatch).
    #   INVALID_ACTION: failed pre-execution validation (e.g. a feature flag
    #     whose dependency is not enabled). Carries the validator reason.
    #   DISPATCH_SKIPPED: skipped at the dispatcher (AP-9 scope violation,
    #     dirty-tree fence, unknown action type, or a handler no-op).
    INVALID_ACTION = "invalid_action"
    DISPATCH_SKIPPED = "dispatch_skipped"
    # 2026-05-23 exogenous-restart resilience (handoff Phase 5).
    # EXOGENOUS_RELOAD: trial corrupted by an operator/external service reload
    #   detected via fleet markers; at least one question stayed unrecovered.
    #   bug_corrupted_by is set to "exogenous_operator_reload" so the planner's
    #   trustworthiness gate excludes it from hypothesis chains. SafetyGate +
    #   Pareto archive are explicitly SKIPPED for this trial.
    # AUTOPILOT_KILLED: placeholder JournalEntry written by the cmd_start
    #   recovery path when in_flight_trial indicates a crash between
    #   dispatch_action and journal.record. No eval evidence available.
    EXOGENOUS_RELOAD = "exogenous_reload"
    AUTOPILOT_KILLED = "autopilot_killed_mid_trial"
    # 2026-05-24: trial completed during a host page-cache flush + NUMA re-warm
    # window (`host_health.flush_cache_with_pause()` runs serial GGUF rewarms
    # taking ~30-60s during which decode throughput is suppressed). Like
    # EXOGENOUS_RELOAD, bug_corrupted_by gets set so the planner's
    # trustworthiness gate excludes the affected trial from hypothesis chains.
    EXOGENOUS_CACHE_FLUSH = "exogenous_cache_flush"

DEFAULT_JOURNAL_DIR = Path(__file__).resolve().parents[2] / "orchestration"
MAX_TRIALS_PER_FILE = 1000
SEGMENT_SNAPSHOT_POLICY_VERSION = "journal-segment-snapshot-v1"

_BASELINE_QUALITY_RE = re.compile(r"\bbaseline\s+([0-9]+(?:\.[0-9]+)?)")
_SUITE_REGRESSION_RE = re.compile(
    r"\bSuite\s+'[^']+'\s+regression:\s+-([0-9]+(?:\.[0-9]+)?)"
)
_MAX_QUALITY_SCALE = 3.0
_LEGACY_SCALE_FAILURE_SUMMARY = (
    "legacy-scale failure_analysis omitted: references impossible 0-3 quality "
    "baseline/per-suite regression; use recorded q/s/r fields instead"
)
_HIGHER_TIER_BUDGET_CREDIT_WEIGHT = 0.15

# ── EV-14e: the baseline pin lives IN the trial record ────────────────────────
# Before this, the incumbent value a trial was judged against reached the journal
# only as PROSE inside `failure_analysis` ("Quality regression: 1.744 vs baseline
# 1.884 …"), and `_BASELINE_QUALITY_RE` above parsed it back out. A regex over a
# human sentence is not a record of a dependency: it cannot say WHICH tier
# reference, which revision (EV-14c), or which eval_quality era was compared, it
# silently reads 0.0 out of an unrelated sentence, and it disappears entirely when
# the gate PASSES (no regression line is written on a pass, so a clean trial
# recorded no baseline at all). `baseline_pin` records the reference identity
# beside the delta so the comparison is self-contained.
#
# Written by the trial writer via `build_baseline_pin()`; read by
# `baseline_pin_for()`, which prefers the structured field and falls back to the
# regex ONLY for rows written before this field existed — marking the fallback
# (`source="legacy_failure_analysis_regex"`) and logging it once per trial, so a
# legacy read is never mistaken for a recorded pin. Legacy shards stay loadable:
# the field defaults to `{}` and nothing on disk is rewritten.
BASELINE_PIN_SCHEMA_VERSION = 1
BASELINE_PIN_SOURCE_STRUCTURED = "structured"
BASELINE_PIN_SOURCE_LEGACY_REGEX = "legacy_failure_analysis_regex"
BASELINE_PIN_SOURCE_ABSENT = "absent"
_LEGACY_BASELINE_PIN_WARNED: set[int] = set()

TSV_COLUMNS = [
    "trial_id",
    "timestamp",
    "species",
    "action_type",
    "tier",
    "quality",
    "speed",
    "cost",
    "reliability",
    "pareto_status",
    "git_tag",
    "reasoning_hash",
]

SUPERSESSION_EVENT_TYPE = "supersession"
BASELINE_PROMOTION_EVENT_TYPE = "baseline_promotion"
# AP-57 (operator ruled B, 2026-09-17): the append-only receipt for a speed-axis reseed on
# the quality-refusal path (safety_gate RTG-02). It carries a post-write ``baseline_state``
# snapshot, so the baseline ledger fold replays it exactly like a promotion.
SPEED_AXIS_RESEED_EVENT_TYPE = "speed_axis_reseed"
BASELINE_LEDGER_EVENT_TYPES = (BASELINE_PROMOTION_EVENT_TYPE, SPEED_AXIS_RESEED_EVENT_TYPE)
JOURNAL_SNAPSHOT_EVENT_TYPE = "journal_snapshot"
ROLE_RESTART_BOUNDARY_EVENT_TYPE = "role_restart_boundary"


def json_sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN / ±Inf) with None for strict JSON.

    D2: ``json.dumps`` with the default ``allow_nan=True`` emits bare ``NaN`` /
    ``Infinity`` tokens, which are invalid JSON — they break ``jq`` and every
    strict parser that reads the journal, RLVR export, and autopilot state files.
    Sanitizing at each serialization boundary lets those writers pass
    ``allow_nan=False`` safely: the only reachable floats are finite, so encoding
    never raises.

    Containers are rebuilt (``dict`` values sanitized, ``list``/``tuple`` elements
    sanitized with tuples collapsed to lists, matching JSON array semantics);
    every other value (str, int, bool, None, and non-JSON objects handled by the
    encoder's ``default=str``) is returned unchanged.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(value) for value in obj]
    return obj


def has_legacy_scale_failure_analysis(text: str) -> bool:
    """True when old failure text carries impossible 0-3 quality-scale values."""
    if not text:
        return False
    baseline_values = (float(m.group(1)) for m in _BASELINE_QUALITY_RE.finditer(text))
    if any(v > _MAX_QUALITY_SCALE for v in baseline_values):
        return True
    suite_deltas = (float(m.group(1)) for m in _SUITE_REGRESSION_RE.finditer(text))
    return any(v > _MAX_QUALITY_SCALE for v in suite_deltas)


def failure_analysis_for_prompt(entry: "JournalEntry", limit: int | None = None) -> str:
    """Render failure_analysis for controller-facing prompts without stale scale leaks."""
    if has_legacy_scale_failure_analysis(entry.failure_analysis):
        text = _LEGACY_SCALE_FAILURE_SUMMARY
    else:
        text = entry.failure_analysis.replace("\n", " | ")
    if limit is not None:
        return text[:limit]
    return text



FAILURE_SIGNATURE_VERSION = 1
# Only categories emitted on actual measured violation branches of SafetyGate.check.
_FAILURE_CODES = frozenset({
    "quality_floor", "regression", "per_suite_regression", "routing_diversity", "throughput",
})


def is_learning_eligible(entry: "JournalEntry") -> bool:
    """Shared existing STM trust filters; does not change any admission decision."""
    details = entry.eval_details if isinstance(entry.eval_details, dict) else {}
    return (not entry.bug_corrupted_by and entry.outcome_status == "ok"
            and entry.keep_revert_decision != "excluded"
            and not details.get("learning_exclusion"))


def build_error_signature(*, passed: bool, retry_not_revert: bool,
                          categories: list[str], species: str, action_type: str,
                          tier: int) -> dict[str, Any] | None:
    """Capture machine branch identity at the producer, never infer it from prose."""
    if passed or retry_not_revert:
        return None
    codes = sorted({code for code in categories
                    if isinstance(code, str) and code in _FAILURE_CODES})
    if (not codes or not _machine_label(species) or not _machine_label(action_type)
            or type(tier) is not int or not 0 <= tier <= 3):
        return None
    return {"schema_version": FAILURE_SIGNATURE_VERSION, "stage": "safety_gate",
            "codes": codes, "species": species, "action_type": action_type, "tier": tier}


def build_error_scope(*, core_id: str, baseline_pin: dict[str, Any],
                      regime_digest: str, comparability: dict[str, Any]) -> dict[str, Any] | None:
    """Explicit recorded context, separate from the failure identity."""
    if not isinstance(baseline_pin, dict) or not isinstance(comparability, dict):
        return None
    fields = {"core_id": core_id, "infra_regime_digest": regime_digest,
              "eval_quality_era": baseline_pin.get("eval_quality_era"),
              "autopilot_speed_era": baseline_pin.get("autopilot_speed_era")}
    if comparability.get("status") != "COMPARABLE" or not all(
            type(value) is str and value.strip() for value in fields.values()):
        return None
    return {"schema_version": FAILURE_SIGNATURE_VERSION, "comparability": "COMPARABLE", **fields}



_SIGNATURE_KEYS = {"schema_version", "stage", "codes", "species", "action_type", "tier"}
_SCOPE_KEYS = {"schema_version", "comparability", "core_id", "infra_regime_digest",
               "eval_quality_era", "autopilot_speed_era"}


def _machine_label(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value) is not None


def _valid_error_signature(entry: "JournalEntry") -> bool:
    signature = entry.error_signature
    if type(signature) is not dict or set(signature) != _SIGNATURE_KEYS:
        return False
    codes = signature.get("codes")
    return (type(signature["schema_version"]) is int
            and signature["schema_version"] == FAILURE_SIGNATURE_VERSION
            and type(signature["stage"]) is str and signature["stage"] == "safety_gate"
            and type(codes) is list and bool(codes)
            and all(type(code) is str and code in _FAILURE_CODES for code in codes)
            and codes == sorted(set(codes))
            and _machine_label(signature["species"]) and signature["species"] == entry.species
            and _machine_label(signature["action_type"])
            and signature["action_type"] == entry.action_type
            and type(signature["tier"]) is int and 0 <= signature["tier"] <= 3
            and type(entry.tier) is int
            and signature["tier"] == entry.tier)


def _valid_error_scope(entry: "JournalEntry") -> bool:
    scope = entry.error_scope
    if type(scope) is not dict or set(scope) != _SCOPE_KEYS:
        return False
    if (type(scope["schema_version"]) is not int
            or scope["schema_version"] != FAILURE_SIGNATURE_VERSION
            or type(scope["comparability"]) is not str or scope["comparability"] != "COMPARABLE"):
        return False
    return all(type(scope[key]) is str and bool(scope[key].strip()) for key in (
        "core_id", "infra_regime_digest", "eval_quality_era", "autopilot_speed_era"))


def _countable_failure(entry: "JournalEntry") -> bool:
    return (is_learning_eligible(entry) and entry.pareto_status != "frontier"
            and _valid_error_signature(entry) and _valid_error_scope(entry))


def has_negative_evidence(entry: "JournalEntry") -> bool:
    return bool(entry.failure_analysis) or _valid_error_signature(entry)


def prior_repeated_failures(entry: "JournalEntry", entries: list["JournalEntry"]) -> int | None:
    """Total PRIOR eligible matches in this folded ledger, never a consecutive streak."""
    if not _countable_failure(entry):
        return None
    return sum(1 for prior in entries
               if prior.trial_id < entry.trial_id and _countable_failure(prior)
               and prior.error_signature == entry.error_signature
               and prior.error_scope == entry.error_scope)

def grouped_negative_evidence(entries: list["JournalEntry"],
                              history: list["JournalEntry"]) -> list[str]:
    """One most-recent compact card per typed identity/scope, preserving unknowns."""
    groups: dict[str, JournalEntry] = {}
    for entry in entries:
        if not has_negative_evidence(entry) or entry.pareto_status == "frontier":
            continue
        key = (json.dumps([entry.error_signature, entry.error_scope], sort_keys=True)
               if _valid_error_signature(entry) else f"unclassified-trial-{entry.trial_id}")
        groups[key] = entry
    return [negative_evidence_for_prompt(entry, entries=history, limit=160)
            for entry in groups.values()]

def negative_evidence_for_prompt(entry: "JournalEntry", *,
                                 entries: list["JournalEntry"] | None = None,
                                 limit: int | None = None) -> str:
    """One compact typed line, or explicitly unclassified bounded legacy prose."""
    if _valid_error_signature(entry):
        count = (prior_repeated_failures(entry, entries) if entries is not None
                 else entry.repeated_failures_prior if _countable_failure(entry) else None)
        repeat = str(count) if type(count) is int and count >= 0 else "unknown"
        signature = entry.error_signature
        scope_label = (hashlib.sha256(json.dumps(entry.error_scope, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()[:12]
                       if _valid_error_scope(entry) else "unknown")
        text = (f"failure/v1 {signature['species']}/{signature['action_type']} "
                f"tier={signature['tier']} safety_gate:" + ",".join(signature["codes"])
                + f" scope={scope_label} prior_total={repeat}")
        # Identity/count must remain intact even when a legacy prose limit is tiny.
        return text
    text = "[unclassified] " + failure_analysis_for_prompt(entry, limit=160)
    return text if limit is None else text[:limit]

def scrub_legacy_scale_text(text: str) -> str:
    """Redact any free-text field carrying impossible 0-3 quality-scale values.

    Used for controller-facing fields beyond failure_analysis (self_criticism,
    optimization_directions) that historically embedded the regression-gate
    string "… vs baseline 9.900". Returns the legacy-scale summary when the text
    references an impossible baseline/per-suite value, else the text unchanged —
    so the planner never re-surfaces a corrupt baseline that has since been fixed.
    """
    if has_legacy_scale_failure_analysis(text):
        return _LEGACY_SCALE_FAILURE_SUMMARY
    return text


def build_baseline_pin(
    *,
    tier: int,
    baseline_quality: float | None,
    candidate_quality: float | None,
    baseline_revision: int = 0,
    eval_quality_era: str = "",
    autopilot_speed_era: str = "",
    per_suite_quality: dict[str, Any] | None = None,
    per_suite_counts: dict[str, Any] | None = None,
    baseline_path: str = "",
    suppressed_by: str = "",
    captured_at: str = "",
) -> dict[str, Any]:
    """EV-14e: render the structured baseline pin recorded beside a trial's delta.

    The pin answers "what exactly was this trial compared against" without
    re-deriving anything at read time: the tier whose reference was used, that
    reference's value and its EV-14c monotonic ``revision`` (so a reference that
    MOVED after the compare is detectable from the row alone), the eval/speed
    instrument eras the reference was captured under, and the per-suite reference
    map with the counts it was measured at. ``delta`` / ``relative_delta`` are
    stored rather than recomputed so the row carries the comparison it actually
    made.

    ``suppressed_by`` names the reason the gate had NO usable reference (e.g.
    ``quality_rebaseline_hold``, ``no_same_tier_baseline``) — an absent baseline
    is recorded as absent, never as 0.0.
    """

    def _f(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    base = _f(baseline_quality)
    cand = _f(candidate_quality)
    delta = (cand - base) if (base is not None and cand is not None) else None
    relative = (delta / base) if (delta is not None and base not in (None, 0.0)) else None
    return {
        "schema_version": BASELINE_PIN_SCHEMA_VERSION,
        "source": BASELINE_PIN_SOURCE_STRUCTURED,
        "tier": int(tier),
        "baseline_quality": base,
        "baseline_revision": int(baseline_revision or 0),
        "eval_quality_era": str(eval_quality_era or ""),
        "autopilot_speed_era": str(autopilot_speed_era or ""),
        "per_suite_baseline_quality": dict(per_suite_quality or {}),
        "per_suite_baseline_counts": dict(per_suite_counts or {}),
        "baseline_path": str(baseline_path or ""),
        "candidate_quality": cand,
        "delta": delta,
        "relative_delta": relative,
        "suppressed_by": str(suppressed_by or ""),
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
    }


def _legacy_baseline_pin_from_text(text: str) -> dict[str, Any] | None:
    """Parse the pre-EV-14e prose baseline out of ``failure_analysis``.

    Legacy rows ONLY. Returns ``None`` when the prose carries no baseline at all.
    A value above the 0-3 quality scale is an already-known corrupt-era artifact
    (see ``has_legacy_scale_failure_analysis``): it is reported as suspect with
    ``baseline_quality=None`` rather than handed back as a number, so a scrubbed
    corrupt baseline cannot re-enter a comparison through the fallback.
    """
    matches = [float(m.group(1)) for m in _BASELINE_QUALITY_RE.finditer(text or "")]
    if not matches:
        return None
    first = matches[0]
    suspect = any(v > _MAX_QUALITY_SCALE for v in matches)
    return {
        "schema_version": BASELINE_PIN_SCHEMA_VERSION,
        "source": BASELINE_PIN_SOURCE_LEGACY_REGEX,
        "tier": None,
        "baseline_quality": None if suspect else first,
        "baseline_revision": None,
        "eval_quality_era": "",
        "autopilot_speed_era": "",
        "per_suite_baseline_quality": {},
        "per_suite_baseline_counts": {},
        "baseline_path": "",
        "candidate_quality": None,
        "delta": None,
        "relative_delta": None,
        "suppressed_by": "",
        "captured_at": "",
        "legacy_parsed_values": matches,
        "legacy_scale_suspect": suspect,
    }


def baseline_pin_for(entry: "JournalEntry", *, warn: bool = True) -> dict[str, Any]:
    """EV-14e reader: the structured pin when present, the legacy regex otherwise.

    Resolution order is deliberate and one-way — a row that carries
    ``baseline_pin`` is NEVER re-parsed from prose, and a row that does not is
    marked as a fallback read (``source="legacy_failure_analysis_regex"``) and
    logged once per trial. A row with neither returns
    ``source="absent"``: no baseline was recorded, which is a different fact from
    "the baseline was 0.0".
    """
    pin = getattr(entry, "baseline_pin", None)
    if isinstance(pin, dict) and pin:
        out = dict(pin)
        out.setdefault("source", BASELINE_PIN_SOURCE_STRUCTURED)
        out.setdefault("schema_version", BASELINE_PIN_SCHEMA_VERSION)
        return out
    legacy = _legacy_baseline_pin_from_text(getattr(entry, "failure_analysis", "") or "")
    if legacy is None:
        return {
            "schema_version": BASELINE_PIN_SCHEMA_VERSION,
            "source": BASELINE_PIN_SOURCE_ABSENT,
            "baseline_quality": None,
        }
    trial_id = int(getattr(entry, "trial_id", -1) or -1)
    if warn and trial_id not in _LEGACY_BASELINE_PIN_WARNED:
        _LEGACY_BASELINE_PIN_WARNED.add(trial_id)
        log.warning(
            "EV-14e legacy baseline read: trial %s carries no structured baseline_pin; "
            "recovered baseline=%s from failure_analysis prose via regex "
            "(scale_suspect=%s). This row's comparison identity (tier reference, "
            "revision, era) was never recorded and cannot be recovered.",
            trial_id,
            legacy.get("baseline_quality"),
            legacy.get("legacy_scale_suspect"),
        )
    return legacy


@dataclass
class JournalEntry:
    trial_id: int
    timestamp: str
    species: str
    action_type: str
    tier: int  # 0, 1, 2, or 3
    quality: float
    speed: float
    cost: float
    reliability: float
    pareto_status: str  # "dominated", "candidate", "frontier"
    git_tag: str = ""
    reasoning_hash: str = ""
    # Full detail goes into JSONL only
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    config_diff: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    parent_trial: int | None = None
    memory_count: int = 0
    active_flags: list[str] = field(default_factory=list)
    eval_details: dict[str, Any] = field(default_factory=dict)
    metric_schema_version: int = 1  # HLE-4 observe-only schema version.
    harness_metrics: dict[str, Any] = field(default_factory=dict)
    oracle_adequacy: dict[str, Any] = field(default_factory=dict)
    seq: dict[str, Any] = field(default_factory=dict)
    failure_analysis: str = ""
    error_signature: dict[str, Any] | None = None
    error_scope: dict[str, Any] | None = None
    repeated_failures_prior: int | None = None
    hypothesis: str = ""
    expected_mechanism: str = ""
    deficiency_category: str = ""  # AP-14: DeficiencyCategory value or empty
    instruction_token_count: int = 0  # AP-16: per-request instruction overhead
    instruction_token_ratio: float = 0.0  # AP-16: instruction_tokens / total_input
    self_criticism: str = ""  # AP-23: structured self-criticism from last trial
    keep_revert_decision: str = ""  # AP-24: "keep" | "revert" | "excluded" | ""
    optimization_directions: str = ""  # AP-24: forward-looking next-round guidance
    predicted_objectives: dict[str, float] = field(default_factory=dict)  # PEAF: controller's pre-trial forecast (empty when disabled / unforecast)
    surprise_score: float | None = None  # PEAF: L1 distance in normalised objective space; None when no forecast
    # 2026-05-23: journal-pollution tracking. When a trial's outcome was
    # caused by an orchestrator bug that has since been fixed, the operator
    # runs scripts/autopilot/scrub_journal.py to set this to the short SHA of
    # the bug-fix commit. The planner's hypothesis-chain reasoning filters
    # bug_corrupted entries out of "trustworthy" trial counts so it doesn't
    # learn wrong lessons from buggy outcomes. Empty string == trustworthy.
    bug_corrupted_by: str = ""
    bug_corrupted_reason: str = ""  # free-text operator note (~80c) for context
    # 2026-05-23: constrained-creativity planner upgrade. `falsifier` holds the
    # one-line predicted outcome whose absence would invalidate the trial's
    # hypothesis (emitted by the controller in the rationale block).
    # `rubric_scores` holds the controller's self-scoring on info_gain /
    # coherence / usefulness plus an optional synthesis note. Both default
    # empty so legacy entries stay loadable.
    falsifier: str = ""
    rubric_scores: dict[str, Any] = field(default_factory=dict)
    # 2026-06-04: outcome status for non-executing trials. "ok" for a normal
    # metric-collecting trial; "invalid" when the action failed pre-execution
    # validation (e.g. a flag dependency); "skipped" when the dispatcher dropped
    # it (AP-9 scope / dirty-tree / unknown / handler no-op). Lets the planner and
    # audits distinguish "ran and scored 0" from "never ran" — the residue that
    # was previously discarded by the bare-None skip path.
    outcome_status: str = "ok"
    # 2026-05-24: stagnation gate signal that fired for this trial (empty for
    # lean-prompt trials, or e.g. "hv_slope_10=+0.00000 < eps=0.00100;
    # last 3 trials all action_type=seed_batch" when the rich prompt was used).
    # Enables retrospective behavioral analysis: action-type diversity under
    # rich vs lean prompts without running a separate experiment.
    stagnation_signal: str = ""
    # 2026-08-10: the measurement constitution's claim tuple, captured at WRITE time.
    # MEASUREMENT_POLICY.md: a decision-gating number is (metric, protocol-id, n/reps, date,
    # attestation ref); a number without a protocol citation is an OBSERVATION and may never gate
    # a keep/revert/deploy decision. A trial already knows every element -- the schema versions,
    # the scored denominator, its own timestamp, and the content it is about to append -- so the
    # tuple is recorded rather than reconstructed later by parsing prose. Populated by
    # `measurement_tuple()` in `record()`; empty on rows written before this date.
    measurement: dict[str, Any] = field(default_factory=dict)
    # 2026-09-14 (EV-14e): the structured identity of the baseline this trial's delta
    # was measured against — see build_baseline_pin()/baseline_pin_for() above and the
    # BASELINE_PIN_* block near the top of this module. Empty on rows written before
    # this date; `baseline_pin_for()` falls back to the prose regex for those and SAYS
    # SO. Never back-filled on load: a pin invented at read time would claim a
    # comparison identity the original trial never captured.
    baseline_pin: dict[str, Any] = field(default_factory=dict)
    # 2026-09-16 (AP-55): the infra regime this trial ran in (orchestrator commit,
    # evaluator digest, kernel binary/libraries, recipe, model files, host config) —
    # see src/autopilot_core/infra_fingerprint.py. `comparability` compares it with
    # the fingerprint of the baseline reference the delta was measured against and
    # says COMPARABLE / NON_COMPARABLE / UNVERIFIED. Both default empty; rows written
    # before this date load unchanged and are never back-filled (a fingerprint
    # invented at read time would claim a regime the trial never recorded).
    infra_fingerprint: dict[str, Any] = field(default_factory=dict)
    comparability: dict[str, Any] = field(default_factory=dict)
    # 2026-09-17 (AP-63(a)): the AP-1510 run manifest bound before dispatch (sources,
    # task, evaluator, manifest_sha256), copied from the in-flight WAL marker so the
    # receipt survives the marker being cleared. Empty on rows written before this date
    # and on rows that never dispatched; never back-filled on load (a manifest built at
    # read time would attest to sources the trial never ran).
    run_manifest: dict[str, Any] = field(default_factory=dict)
    # 2026-09-17 (AP-63(a)): explicit, stored parent selection — see
    # select_lineage_parent(). `parent_trial` above keeps its legacy same-species
    # heuristic meaning (config_diff, PEAF, BSV, Pareto consume it); `lineage` records the
    # rule-named ACCEPTED parent. Empty on legacy rows; never back-filled.
    lineage: dict[str, Any] = field(default_factory=dict)


LINEAGE_SCHEMA_VERSION = 1
LINEAGE_PARENT_RULE = "latest_committed_baseline_promotion_same_species"
LINEAGE_HEURISTIC_RULE = "latest_same_species_row"


def select_lineage_parent(
    journal: "ExperimentJournal",
    species: str,
    *,
    before_trial_id: int,
) -> dict[str, Any]:
    """Select a trial's parent explicitly and say which rule chose it (AP-63(a)).

    The accepted parent is the newest trial of the same species whose baseline
    promotion was COMMITTED — i.e. that has an append-only ``baseline_promotion``
    ledger event, the existing evidence/quality gate's commit record. A row whose
    promotion is only ``pending_commit`` is not accepted, and a row later marked
    ``bug_corrupted_by`` (directly or by supersession) is never a parent. No accepted
    parent yields ``parent_trial_id=None`` rather than a fallback: an unaccepted
    parent is exactly what the orx refill disposition forbids.

    The legacy heuristic (newest same-species row, whatever its outcome) is recorded
    alongside so a reader can see where the two differ; it decides nothing here.
    """
    before = int(before_trial_id)
    entries = {
        e.trial_id: e
        for e in journal.entries_with_supersessions()
        if e.trial_id < before
    }
    heuristic = None
    for e in reversed(journal.by_species(species)):
        if e.trial_id < before:
            heuristic = e.trial_id
            break
    accepted: int | None = None
    promotion_ts = ""
    for event in reversed(journal.baseline_promotion_events()):
        try:
            source = int(event.get("source_trial_id"))
        except (TypeError, ValueError):
            continue
        row = entries.get(source)
        if row is None or row.species != species or row.bug_corrupted_by:
            continue
        accepted = source
        promotion_ts = str(event.get("timestamp") or "")
        break
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "rule": LINEAGE_PARENT_RULE,
        "species": species,
        "parent_trial_id": accepted,
        "parent_promotion_timestamp": promotion_ts,
        "heuristic_rule": LINEAGE_HEURISTIC_RULE,
        "heuristic_parent_trial_id": heuristic,
    }


def measurement_tuple(entry: "JournalEntry", *, locator: str = "") -> dict[str, Any]:
    """Extract the constitution's claim tuple from what a trial already knows.

    Deliberately reports what is ABSENT instead of inventing it. A missing protocol id makes the
    row an observation, and that is a true and useful thing to record -- filling it with a
    plausible default would manufacture warrant, which is the failure this whole program exists to
    catch. Nothing here computes a metric or re-derives a number; it only names provenance.

    `reps` is the SCORED denominator, not the question count. A 50-question suite where 47 were
    scored has n=47: quality was computed over 47, and claiming 50 overstates the sample by 6%.
    """
    ed = entry.eval_details if isinstance(entry.eval_details, dict) else {}
    hm = entry.harness_metrics if isinstance(entry.harness_metrics, dict) else {}
    details = ed.get("details") if isinstance(ed.get("details"), dict) else {}
    cfg = entry.config_snapshot if isinstance(entry.config_snapshot, dict) else {}

    parts = []
    if entry.metric_schema_version:
        parts.append(f"metric-v{entry.metric_schema_version}")
    hv = hm.get("schema_version") or hm.get("metric_version")
    if hv:
        parts.append(f"harness-v{hv}")
    policy = ed.get("objective_policy_live")
    if isinstance(policy, dict) and policy.get("policy"):
        parts.append(str(policy["policy"]))
    elif isinstance(policy, str) and policy:
        parts.append(policy)
    protocol_id = "autopilot/" + "+".join(parts) if parts else ""

    # Ordered best-to-worst denominator. `quality_denominator`/`n_scored` are what the metric was
    # actually computed over; `n_questions`/`total` are what was ATTEMPTED, which overstates n
    # whenever a question failed to score. Older rows carry only `total`, so it is kept as a last
    # resort -- but which key supplied the number is recorded, because "n=55 attempted" and
    # "n=55 scored" are different claims and a consumer must be able to tell them apart.
    reps = None
    reps_basis = ""
    for source, key, basis in ((details, "quality_denominator", "scored"),
                               (details, "n_scored", "scored"),
                               (details, "n_questions", "attempted"),
                               (details, "total", "attempted"),
                               (cfg, "n_questions", "attempted")):
        val = source.get(key) if isinstance(source, dict) else None
        if isinstance(val, int) and val > 0:
            reps, reps_basis = val, f"{basis}:{key}"
            break

    # Hash the entry's own content, excluding this block -- a digest that covered the block it
    # lives in could never be recomputed, so it would attest to nothing.
    payload = {k: v for k, v in asdict(entry).items() if k != "measurement"}
    digest = hashlib.sha256(
        json.dumps(json_sanitize(payload), sort_keys=True, default=str,
                   allow_nan=False).encode("utf-8")
    ).hexdigest()

    out: dict[str, Any] = {
        "protocol_id": protocol_id,
        "reps": reps,
        "reps_basis": reps_basis,
        # Vocabulary borrowed verbatim from AutoKernel's `claim_grammar`
        # (epyc-inference-research scripts/kernel_rnd/autokernel/schemas.py: CLAIM_CATEGORIES,
        # METRIC_DIRECTIONS), which enforces the same MEASUREMENT.md:13 claim rule as a REQUIRED
        # schema block. Aligning the two loops means one grammar, not two dialects of it.
        #
        # A trial is a CANDIDATE by construction: a proposed change being measured, never the
        # standing baseline and never a ratified optimum. MEASUREMENT.md:85-95 calls conflating
        # these the costliest recurring measurement defect in this project.
        "category": "CANDIDATE",
        # Direction is recorded, not inferred. The optimizer declares
        # `directions=["maximize"]*4` over (quality, speed, -cost, reliability) -- the third
        # objective is NEGATED cost, so the raw `cost` field stored on this row is lower_better.
        # Reading the maximize-list alone would invert it.
        "metric_directions": {"quality": "higher_better", "speed": "higher_better",
                              "cost": "lower_better", "reliability": "higher_better"},
        "date": (entry.timestamp or "")[:10],
        "attestation": {"locator": locator, "sha256": digest,
                        "git_tag": entry.git_tag or ""},
        "captured_by": "experiment_journal.measurement_tuple/v1",
    }
    # AP-55: bind the claim to its infra regime. Only present when the trial carried
    # a fingerprint; a legacy/unfingerprinted row says nothing rather than guessing.
    fp = entry.infra_fingerprint if isinstance(entry.infra_fingerprint, dict) else {}
    if fp.get("digest"):
        out["infra_fingerprint"] = str(fp["digest"])
    comp = entry.comparability if isinstance(entry.comparability, dict) else {}
    if comp.get("status"):
        out["comparability"] = str(comp["status"])
    # AP-63(a): bind the claim to the run manifest that dispatched it. Absent on
    # legacy/undispatched rows; nothing is inferred for those.
    manifest = entry.run_manifest if isinstance(entry.run_manifest, dict) else {}
    if manifest.get("manifest_sha256"):
        out["run_manifest"] = str(manifest["manifest_sha256"])
    # AP-54: whether the rollouts ran behind the eval knowledge fence, as the
    # EvalTower recorded it from the API's echo. Absent on rows written before
    # the fence existed (never back-filled).
    fence = details.get("eval_fence") if isinstance(details.get("eval_fence"), dict) else {}
    if fence.get("state"):
        out["eval_fence"] = str(fence["state"])
    if fence.get("fence_enforcement"):
        out["eval_fence_enforcement"] = str(fence["fence_enforcement"])
    # AP-55 (b)+(c): the promotion-gate legs as the trial recorded them. Absent on
    # rows written before the gate existed; nothing is inferred for those.
    gate = comp.get("promotion_gate")
    if isinstance(gate, dict) and gate:
        out["ap55_gate"] = {
            "mode": str(gate.get("mode") or ""),
            "seed_rerun": str((gate.get("seed_rerun") or {}).get("status") or gate.get("status") or ""),
            "batch_homogeneity": str(
                (gate.get("batch_homogeneity") or {}).get("status") or gate.get("status") or ""
            ),
            "hold": bool(gate.get("hold")),
        }
    missing = [name for name, present in (("protocol_id", protocol_id), ("reps", reps),
                                          ("date", out["date"]))
               if not present]
    if missing:
        # Named here so a low grade is self-explaining downstream and nobody has to re-derive
        # why a trial failed to reach Witnessed.
        out["missing"] = missing
    return out


@dataclass
class SupersessionEvent:
    """Append-only ledger event superseding fields on prior trial rows.

    This is the Phase-3 event-sourcing bridge: new tooling can append a durable
    intent record without rewriting historical trial rows. Runtime consumers can
    fold these events later; until then the legacy scrub path remains available.
    """

    target_trial_ids: list[int]
    fields: dict[str, Any]
    reason: str
    policy_version: str
    actor: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = SUPERSESSION_EVENT_TYPE


@dataclass
class BaselinePromotionEvent:
    """Append-only ledger event recording an accepted production baseline move."""

    source_trial_id: int
    tier: int
    previous_quality: float | None
    new_quality: float
    reason: str
    proof: dict[str, Any]
    result_metrics: dict[str, Any]
    baseline_state: dict[str, Any]
    policy_version: str
    actor: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = BASELINE_PROMOTION_EVENT_TYPE
    # AP-55: infra regime of the promoted reference (the source trial's fingerprint).
    infra_fingerprint: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeedAxisReseedEvent:
    """Append-only ledger event recording a speed-axis reseed on a refused promotion (AP-57).

    The quality promotion was refused (eval-instrument re-baseline hold), but the independent
    SPEED axis was re-anchored from the same in-era measurement: ``frontdoor_speed`` and
    ``autopilot_speed_era`` changed. ``baseline_state`` is the post-write snapshot.
    """

    source_trial_id: int
    tier: int
    previous_speed: float | None
    new_speed: float
    previous_speed_era: str
    new_speed_era: str
    reason: str
    refusal_reason: str
    eval_quality_era: str
    run_manifest_sha256: str
    result_metrics: dict[str, Any]
    baseline_state: dict[str, Any]
    policy_version: str
    actor: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = SPEED_AXIS_RESEED_EVENT_TYPE
    infra_fingerprint: dict[str, Any] = field(default_factory=dict)


@dataclass
class JournalSnapshotEvent:
    """Append-only segment snapshot for bounded journal replay."""

    through_trial_id: int
    snapshot: dict[str, Any]
    policy_version: str
    actor: str
    parent_snapshot_hash: str = ""
    snapshot_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = JOURNAL_SNAPSHOT_EVENT_TYPE


@dataclass
class RoleRestartBoundaryEvent:
    """Append-only event recording an intentional role-restart boundary."""

    role: str
    affected_roles: list[str]
    env_keys: list[str]
    registry_override_keys: list[str]
    status: str
    rollback_status: str
    reason: str
    policy_version: str
    actor: str
    boundary_trial_id: int | None = None
    command: str = ""
    evidence: dict[str, Any] | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = ROLE_RESTART_BOUNDARY_EVENT_TYPE


def _snapshot_hash(
    *,
    through_trial_id: int,
    snapshot: dict[str, Any],
    policy_version: str,
    parent_snapshot_hash: str = "",
) -> str:
    payload = {
        "through_trial_id": int(through_trial_id),
        "snapshot": snapshot,
        "policy_version": policy_version,
        "parent_snapshot_hash": parent_snapshot_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExperimentJournal:
    """Append-only experiment log with TSV (human-readable) + JSONL (machine-readable)."""

    def __init__(self, journal_dir: Path | None = None, *, segment_snapshots: bool = True):
        self.journal_dir = journal_dir or DEFAULT_JOURNAL_DIR
        # W3: append a chained snapshot row when a shard closes (see _maybe_close_segment).
        self.segment_snapshots = segment_snapshots
        # Replay scope for segment snapshots ({"objective_policy", "exclude_before_ts"});
        # AutoPilot keeps it aligned with the live archive-authority scope.
        self.segment_snapshot_scope: dict[str, Any] = {}
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[JournalEntry] = []
        self._ledger_events_by_batch: dict[int, list[dict[str, Any]]] = {}
        # A5 durability: count of torn trailing lines skipped during the tolerant
        # load. >0 means a prior append was interrupted; the next append will
        # quarantine + repair the shard. Exposed for observability / tests.
        self.torn_lines_skipped: int = 0
        self._load_existing()

    # ── persistence ──────────────────────────────────────────────

    def _repair_torn_tail(self, jsonl: Path) -> None:
        """Quarantine a torn final append before writing a fresh line (A5).

        Journal lines average ~36KB and can exceed 300KB, so a crash mid-write
        leaves a partial record with no trailing newline. Under an exclusive
        flock we copy the partial-tail bytes to a timestamped ``.corrupt-*``
        sidecar beside the shard, then truncate the file back to the last good
        newline. Data is never dropped unless the sidecar write succeeds first,
        so the quarantined bytes are always recoverable.

        No-op when the file is absent, empty, or already newline-terminated.
        """
        if not jsonl.exists():
            return
        with open(jsonl, "r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size == 0:
                    return
                f.seek(size - 1)
                if f.read(1) == b"\n":
                    return  # clean tail — previous append completed
                # Walk backwards in chunks to find the last newline (or SOF)
                # without loading a potentially huge shard into memory.
                newline_pos = -1
                chunk = 1 << 16
                pos = size
                while pos > 0:
                    read_start = max(0, pos - chunk)
                    f.seek(read_start)
                    buf = f.read(pos - read_start)
                    idx = buf.rfind(b"\n")
                    if idx != -1:
                        newline_pos = read_start + idx
                        break
                    pos = read_start
                truncate_to = newline_pos + 1  # keep the newline; 0 if none
                f.seek(truncate_to)
                partial = f.read(size - truncate_to)
                if not partial:
                    return
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                sidecar = jsonl.parent / f"{jsonl.name}.corrupt-{ts}"
                with open(sidecar, "wb") as scf:
                    scf.write(partial)
                    scf.flush()
                    os.fsync(scf.fileno())
                f.truncate(truncate_to)
                f.flush()
                os.fsync(f.fileno())
                log.error(
                    "quarantined torn trailing journal write: %d bytes from %s "
                    "moved to sidecar %s; shard truncated to last good newline",
                    len(partial),
                    jsonl,
                    sidecar,
                )
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _tsv_path(self, batch: int = 0) -> Path:
        suffix = f"_{batch}" if batch > 0 else ""
        return self.journal_dir / f"autopilot_journal{suffix}.tsv"

    def _jsonl_path(self, batch: int = 0) -> Path:
        suffix = f"_{batch}" if batch > 0 else ""
        return self.journal_dir / f"autopilot_journal{suffix}.jsonl"

    def _current_batch(self) -> int:
        if not self._entries:
            return 0
        return self._entries[-1].trial_id // MAX_TRIALS_PER_FILE

    def _load_existing(self) -> None:
        """Load entries from all existing JSONL shards.

        Shard discovery uses the canonical ``journal_shards`` iterator (audit
        JRN-5/6/7): every rotated shard is read in numeric batch order and a gap
        (e.g. a missing ``_2``) no longer stops discovery at that index — the old
        ``while path(batch).exists()`` loop silently dropped ``_3`` and every later
        shard once a shard was absent. The per-shard batch index (used to key
        ``_ledger_events_by_batch``) is derived from each shard's filename.

        Tolerant of a single torn *trailing* line (a partial final append left
        by a crash): it is logged and skipped here (``torn_lines_skipped`` is
        incremented) and the writer-side ``_repair_torn_tail`` quarantines it on
        the next append. Mid-file corruption — a malformed line with valid lines
        after it — is NOT auto-recoverable and raises
        ``ExperimentJournalCorruptError``.
        """
        for jsonl in journal_shards(self.journal_dir):
            batch = shard_batch_index(jsonl)
            if batch is None:  # defensive: journal_shards only yields matching files
                continue
            with open(jsonl) as f:
                lines = f.readlines()
            for line_number, raw in enumerate(lines, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    # Tolerate ONLY a torn trailing record: every later line in
                    # this shard must be blank. Otherwise it is mid-file
                    # corruption we must not silently drop.
                    if all(not later.strip() for later in lines[line_number:]):
                        log.error(
                            "torn trailing journal line in %s (line %d, %d bytes)"
                            " — skipping; it will be quarantined on next append",
                            jsonl,
                            line_number,
                            len(raw),
                        )
                        self.torn_lines_skipped += 1
                        continue
                    raise ExperimentJournalCorruptError(
                        f"{jsonl}:{line_number}: malformed journal line with "
                        f"valid lines after it (mid-file corruption): {exc}. "
                        f"Remediation: inspect the shard and use "
                        f"scripts/autopilot/scrub_journal.py (append-only "
                        f"supersession events) — never hand-edit historical rows."
                    ) from exc
                if data.get("type") and "trial_id" not in data:
                    self._ledger_events_by_batch.setdefault(batch, []).append(data)
                    continue
                entry = JournalEntry(
                    trial_id=data["trial_id"],
                    timestamp=data["timestamp"],
                    species=data["species"],
                    action_type=data["action_type"],
                    tier=data.get("tier", 0),
                    quality=data.get("quality", 0.0),
                    speed=data.get("speed", 0.0),
                    cost=data.get("cost", 0.0),
                    reliability=data.get("reliability", 0.0),
                    pareto_status=data.get("pareto_status", "dominated"),
                    git_tag=data.get("git_tag", ""),
                    reasoning_hash=data.get("reasoning_hash", ""),
                    config_snapshot=data.get("config_snapshot", {}),
                    config_diff=data.get("config_diff", {}),
                    reasoning=data.get("reasoning", ""),
                    parent_trial=data.get("parent_trial"),
                    memory_count=data.get("memory_count", 0),
                    active_flags=data.get("active_flags", []),
                    eval_details=data.get("eval_details", {}),
                    metric_schema_version=data.get(
                        "metric_schema_version",
                        data.get("eval_details", {}).get("metric_schema_version", 1),
                    ),
                    harness_metrics=data.get(
                        "harness_metrics",
                        data.get("eval_details", {}).get("harness_metrics", {}),
                    ),
                    oracle_adequacy=data.get(
                        "oracle_adequacy",
                        data.get("eval_details", {}).get("oracle_adequacy", {}),
                    ),
                    seq=data.get("seq", {}),
                    failure_analysis=data.get("failure_analysis", ""),
                    error_signature=data.get("error_signature"),
                    error_scope=data.get("error_scope"),
                    repeated_failures_prior=data.get("repeated_failures_prior"),
                    hypothesis=data.get("hypothesis", ""),
                    expected_mechanism=data.get("expected_mechanism", ""),
                    deficiency_category=data.get("deficiency_category", ""),
                    instruction_token_count=data.get("instruction_token_count", 0),
                    instruction_token_ratio=data.get("instruction_token_ratio", 0.0),
                    self_criticism=data.get("self_criticism", ""),
                    keep_revert_decision=data.get("keep_revert_decision", ""),
                    optimization_directions=data.get("optimization_directions", ""),
                    predicted_objectives=data.get("predicted_objectives", {}),
                    surprise_score=data.get("surprise_score", None),
                    bug_corrupted_by=data.get("bug_corrupted_by", ""),
                    bug_corrupted_reason=data.get("bug_corrupted_reason", ""),
                    falsifier=data.get("falsifier", ""),
                    rubric_scores=data.get("rubric_scores", {}),
                    stagnation_signal=data.get("stagnation_signal", ""),
                    outcome_status=data.get("outcome_status", "ok"),
                    # Rows written before 2026-08-10 carry no tuple. They default to empty rather
                    # than being back-filled: a tuple invented on load would claim provenance the
                    # original run never recorded.
                    measurement=data.get("measurement", {}) or {},
                    # EV-14e: absent on every row written before 2026-09-14. Defaults to
                    # empty (never reconstructed from prose here) so a legacy shard loads
                    # unchanged and `baseline_pin_for()` is the one place that decides to
                    # fall back — and marks it.
                    baseline_pin=data.get("baseline_pin", {}) or {},
                    # AP-55: absent on rows written before 2026-09-16; never back-filled.
                    infra_fingerprint=data.get("infra_fingerprint", {}) or {},
                    comparability=data.get("comparability", {}) or {},
                    # AP-63(a): absent on rows written before 2026-09-17; never back-filled.
                    run_manifest=data.get("run_manifest", {}) or {},
                    lineage=data.get("lineage", {}) or {},
                )
                self._entries.append(entry)

    def _emit_trace_event(self, entry: JournalEntry) -> None:
        """Push this trial into the live trace store as well as the journal.

        The trace store (data/trace/events.sqlite) carried role/trial/category/
        detail for 10,488 events — every one of them INGESTED from a log file
        after the fact. Nothing emitted live from the trial loop, so a trial's
        trace existed only if someone later remembered to run ingest. Trial 1460
        wrote 9 episodic rows and 0 trace events.

        Emitting here rather than at the call sites because `record()` is the one
        chokepoint every path goes through — completed trials, skips, and the
        AUTOPILOT_KILLED crash placeholder alike.

        FAILS OPEN. The trace store is an observability plane; it must never be
        able to break the journal write that just succeeded above.
        """
        try:
            from src.trace.emit import emit
            from src.trace.store import Event, EventCategory, EventSource

            status = getattr(entry, "outcome_status", None) or getattr(
                entry, "pareto_status", None
            )
            detail = {
                k: getattr(entry, k, None)
                for k in (
                    "trial_id", "species", "tier", "quality", "speed", "cost",
                    "reliability", "pareto_status", "outcome_status",
                    "failure_analysis", "deficiency_category", "memory_count",
                    "config_snapshot", "reasoning",
                )
                if getattr(entry, k, None) is not None
            }
            emit(
                Event(
                    ts_utc="",  # emit() stamps it
                    source=EventSource.AUTOPILOT_LIVE,
                    source_path="",  # emit() assigns a content-addressed key
                    trial_id=int(getattr(entry, "trial_id", 0) or 0) or None,
                    role=str(getattr(entry, "species", "") or "") or None,
                    category=EventCategory.TASK_END,
                    status=str(status) if status else None,
                    summary=(
                        f"trial {getattr(entry, 'trial_id', '?')} "
                        f"species={getattr(entry, 'species', '?')} "
                        f"tier={getattr(entry, 'tier', '?')} "
                        f"quality={getattr(entry, 'quality', '?')}"
                    ),
                    detail_json=json.dumps(json_sanitize(detail), default=str, allow_nan=False),
                )
            )
        except Exception:  # noqa: BLE001 — observability must not break the journal
            pass

    # ── writing ──────────────────────────────────────────────────

    def _maybe_close_segment(self, next_trial_id: int) -> dict[str, Any] | None:
        """W3: chain a closing snapshot row onto a shard before the journal rotates.

        When the next trial belongs to a later ``MAX_TRIALS_PER_FILE`` segment than
        the last recorded trial, the full reconstructed archive view is appended as
        a ``journal_snapshot`` event to the CLOSING shard (``append_ledger_event``
        targets the current batch), parented on the previous snapshot hash. Each
        segment therefore ends with the snapshot that lets a rebuild start there and
        fold only the newer tail.

        Skipped when disabled, when the latest snapshot already covers the last
        trial, or when there is nothing to snapshot. FAILS OPEN: a snapshot is a
        replay accelerator, never a precondition for recording a trial.
        """
        if not self.segment_snapshots or not self._entries:
            return None
        closing_batch = self._current_batch()
        if next_trial_id // MAX_TRIALS_PER_FILE <= closing_batch:
            return None
        last_trial_id = self._entries[-1].trial_id
        latest = self.latest_journal_snapshot_event() or {}
        try:
            if int(latest.get("through_trial_id", -1)) >= last_trial_id:
                return None
        except (TypeError, ValueError):
            pass
        try:
            try:
                from scripts.autopilot.journal_snapshot_create import build_archive_snapshot
            except ModuleNotFoundError:  # pragma: no cover - bare-module import context
                from journal_snapshot_create import build_archive_snapshot

            result = build_archive_snapshot(
                self,
                policy_version=SEGMENT_SNAPSHOT_POLICY_VERSION,
                **dict(self.segment_snapshot_scope or {}),
            )
            if result.status != "ready" or result.snapshot is None:
                log.warning(
                    "W3 segment snapshot for shard %d not appended: %s %s",
                    closing_batch, result.status, result.warning,
                )
                return None
            snapshot = dict(result.snapshot)
            segment_ids = [
                e.trial_id for e in self._entries
                if e.trial_id // MAX_TRIALS_PER_FILE == closing_batch
            ]
            snapshot["segment"] = {
                "batch": closing_batch,
                "shard": self._jsonl_path(closing_batch).name,
                "first_trial_id": min(segment_ids) if segment_ids else None,
                "last_trial_id": max(segment_ids) if segment_ids else None,
                "trial_count": len(segment_ids),
                "next_trial_id": int(next_trial_id),
            }
            return self.append_journal_snapshot_event(
                through_trial_id=int(result.through_trial_id),
                snapshot=snapshot,
                policy_version=result.policy_version,
                actor="experiment_journal.segment_rollover",
                parent_snapshot_hash=result.parent_snapshot_hash,
            )
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning("W3 segment snapshot for shard %d failed: %s", closing_batch, exc)
            return None

    def record(self, entry: JournalEntry) -> None:
        """Append a trial entry to both TSV and JSONL."""
        # Native snapshot at append; reads recount folded supersessions independently.
        entry.repeated_failures_prior = (prior_repeated_failures(
            entry, self.entries_with_supersessions()) if _countable_failure(entry) else None)
        self._maybe_close_segment(entry.trial_id)
        batch = entry.trial_id // MAX_TRIALS_PER_FILE
        tsv = self._tsv_path(batch)
        jsonl = self._jsonl_path(batch)

        # TSV (human-readable subset)
        write_header = not tsv.exists()
        with open(tsv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow({col: getattr(entry, col) for col in TSV_COLUMNS})

        # JSONL (full detail) — durable append (A5): repair any torn tail from a
        # prior crash, serialize the full line up front, then write under an
        # exclusive flock with fsync so a crash can only ever tear THIS trailing
        # line (never a mid-file record), and concurrent writers can't interleave.
        self._repair_torn_tail(jsonl)
        # 2026-08-10: capture the measurement constitution's claim tuple before serializing, so a
        # trial is born attested instead of being retrofitted later. Never overwrite a tuple a
        # caller already supplied, and never let this fail a trial — a provenance annotation that
        # can lose a result is worse than one that is occasionally absent, so a failure records
        # itself in the row rather than propagating.
        if not entry.measurement:
            try:
                entry.measurement = measurement_tuple(
                    entry, locator=f"{jsonl.name}#trial-{entry.trial_id}")
            except Exception as exc:  # noqa: BLE001 - see comment above
                entry.measurement = {"capture_error": f"{type(exc).__name__}: {exc}"[:200]}
        # D2: sanitize non-finite floats → null and forbid bare NaN/Infinity tokens
        # so every written line is strict, jq-parseable JSON.
        line = json.dumps(json_sanitize(asdict(entry)), default=str, allow_nan=False) + "\n"
        with open(jsonl, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self._entries.append(entry)
        self._emit_trace_event(entry)

    def append_ledger_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append a non-trial ledger event row to JSONL."""
        event = copy.deepcopy(event)
        if not event.get("type"):
            raise ValueError("ledger events require a type")
        if "trial_id" in event:
            raise ValueError("ledger events must not use the trial_id field")
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        batch = self._current_batch()
        jsonl = self._jsonl_path(batch)
        # Durable append (A5): mirror record()'s repair → serialize → locked
        # fsync'd write so ledger events survive a crash the same way trials do.
        self._repair_torn_tail(jsonl)
        # D2: strict JSON (non-finite floats → null, no bare NaN/Infinity tokens).
        line = json.dumps(json_sanitize(event), default=str, allow_nan=False) + "\n"
        with open(jsonl, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        self._ledger_events_by_batch.setdefault(batch, []).append(event)
        return event

    def append_supersession_event(
        self,
        *,
        target_trial_ids: list[int],
        fields: dict[str, Any],
        reason: str,
        policy_version: str,
        actor: str,
    ) -> dict[str, Any]:
        """Append a supersession event row to JSONL without mutating trials."""
        return self.append_ledger_event(asdict(
            SupersessionEvent(
                target_trial_ids=target_trial_ids,
                fields=fields,
                reason=reason,
                policy_version=policy_version,
                actor=actor,
            )
        ))

    def append_baseline_promotion_event(
        self,
        *,
        source_trial_id: int,
        tier: int,
        previous_quality: float | None,
        new_quality: float,
        reason: str,
        proof: dict[str, Any],
        result_metrics: dict[str, Any],
        baseline_state: dict[str, Any],
        policy_version: str = "baseline-promotion-v1",
        actor: str = "autopilot.py",
        infra_fingerprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a baseline-promotion event row without changing baseline state."""
        return self.append_ledger_event(asdict(
            BaselinePromotionEvent(
                source_trial_id=source_trial_id,
                tier=tier,
                previous_quality=previous_quality,
                new_quality=new_quality,
                reason=reason,
                proof=proof,
                result_metrics=result_metrics,
                baseline_state=baseline_state,
                policy_version=policy_version,
                actor=actor,
                infra_fingerprint=copy.deepcopy(infra_fingerprint or {}),
            )
        ))

    def append_speed_axis_reseed_event(
        self,
        *,
        source_trial_id: int,
        tier: int,
        previous_speed: float | None,
        new_speed: float,
        previous_speed_era: str,
        new_speed_era: str,
        refusal_reason: str,
        result_metrics: dict[str, Any],
        baseline_state: dict[str, Any],
        eval_quality_era: str = "",
        run_manifest_sha256: str = "",
        policy_version: str = "speed-axis-reseed-v1",
        actor: str = "autopilot.py",
        infra_fingerprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a speed-axis reseed receipt (AP-57), at most once per source trial.

        Idempotent: a second call for a ``source_trial_id`` that already has a receipt in the
        loaded ledger returns the existing row and writes nothing.
        """
        source_trial_id = int(source_trial_id)
        for existing in self.speed_axis_reseed_events():
            try:
                if int(existing.get("source_trial_id")) == source_trial_id:
                    return existing
            except (TypeError, ValueError):
                continue
        return self.append_ledger_event(asdict(
            SpeedAxisReseedEvent(
                source_trial_id=source_trial_id,
                tier=int(tier),
                previous_speed=previous_speed,
                new_speed=float(new_speed),
                previous_speed_era=str(previous_speed_era or ""),
                new_speed_era=str(new_speed_era or ""),
                reason="quality_refusal",
                refusal_reason=str(refusal_reason or ""),
                eval_quality_era=str(eval_quality_era or ""),
                run_manifest_sha256=str(run_manifest_sha256 or ""),
                result_metrics=copy.deepcopy(result_metrics),
                baseline_state=copy.deepcopy(baseline_state),
                policy_version=policy_version,
                actor=actor,
                infra_fingerprint=copy.deepcopy(infra_fingerprint or {}),
            )
        ))

    def append_journal_snapshot_event(
        self,
        *,
        through_trial_id: int,
        snapshot: dict[str, Any],
        policy_version: str,
        actor: str,
        parent_snapshot_hash: str = "",
    ) -> dict[str, Any]:
        """Append a segment snapshot row without changing replay authority."""
        snapshot_hash = _snapshot_hash(
            through_trial_id=through_trial_id,
            snapshot=snapshot,
            policy_version=policy_version,
            parent_snapshot_hash=parent_snapshot_hash,
        )
        return self.append_ledger_event(asdict(
            JournalSnapshotEvent(
                through_trial_id=int(through_trial_id),
                snapshot=copy.deepcopy(snapshot),
                policy_version=policy_version,
                actor=actor,
                parent_snapshot_hash=parent_snapshot_hash,
                snapshot_hash=snapshot_hash,
            )
        ))

    def append_role_restart_boundary_event(
        self,
        *,
        role: str,
        affected_roles: list[str] | None = None,
        env_keys: list[str] | None = None,
        registry_override_keys: list[str] | None = None,
        status: str,
        rollback_status: str = "",
        reason: str,
        policy_version: str = "role-restart-boundary-v1",
        actor: str = "config_applicator.restart_role",
        boundary_trial_id: int | None = None,
        command: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an intentional role-restart boundary event."""
        return self.append_ledger_event(asdict(
            RoleRestartBoundaryEvent(
                role=role,
                affected_roles=list(affected_roles or [role]),
                env_keys=sorted(env_keys or []),
                registry_override_keys=sorted(registry_override_keys or []),
                status=status,
                rollback_status=rollback_status,
                reason=reason,
                policy_version=policy_version,
                actor=actor,
                boundary_trial_id=boundary_trial_id,
                command=command,
                evidence=copy.deepcopy(evidence) if evidence is not None else None,
            )
        ))

    # ── queries ──────────────────────────────────────────────────

    def recent(self, n: int = 20) -> list[JournalEntry]:
        """Return last n entries."""
        return self._entries[-n:]

    def all_entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def ledger_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Return loaded append-only ledger event rows, optionally filtered by type."""
        events: list[dict[str, Any]] = []
        for batch in sorted(self._ledger_events_by_batch):
            events.extend(
                event
                for event in self._ledger_events_by_batch[batch]
                if event_type is None or event.get("type") == event_type
            )
        return events

    def supersession_events(self) -> list[dict[str, Any]]:
        """Return loaded append-only supersession event rows."""
        return self.ledger_events(SUPERSESSION_EVENT_TYPE)

    def baseline_promotion_events(self) -> list[dict[str, Any]]:
        """Return loaded append-only baseline promotion event rows."""
        return self.ledger_events(BASELINE_PROMOTION_EVENT_TYPE)

    def speed_axis_reseed_events(self) -> list[dict[str, Any]]:
        """Return loaded append-only speed-axis reseed receipts (AP-57)."""
        return self.ledger_events(SPEED_AXIS_RESEED_EVENT_TYPE)

    def baseline_ledger_events(self) -> list[dict[str, Any]]:
        """Every baseline-state-writing ledger event, in append order.

        Promotions and speed-axis reseeds both carry a post-write ``baseline_state``
        snapshot; the baseline ledger fold must see both, in order, to replay the state.
        Use :meth:`baseline_promotion_events` where only real promotions count.
        """
        return [
            event
            for event in self.ledger_events()
            if event.get("type") in BASELINE_LEDGER_EVENT_TYPES
        ]

    def journal_snapshot_events(self) -> list[dict[str, Any]]:
        """Return loaded append-only journal snapshot event rows."""
        return self.ledger_events(JOURNAL_SNAPSHOT_EVENT_TYPE)

    def role_restart_boundary_events(self) -> list[dict[str, Any]]:
        """Return loaded append-only role restart boundary event rows."""
        return self.ledger_events(ROLE_RESTART_BOUNDARY_EVENT_TYPE)

    def latest_journal_snapshot_event(self) -> dict[str, Any] | None:
        """Return the newest snapshot event by ledger order, if any."""
        events = self.journal_snapshot_events()
        return events[-1] if events else None

    def _supersession_overrides_by_trial(self) -> dict[int, dict[str, Any]]:
        entry_fields = set(JournalEntry.__dataclass_fields__)
        overrides_by_trial: dict[int, dict[str, Any]] = {}
        for event in self.supersession_events():
            fields = event.get("fields")
            targets = event.get("target_trial_ids")
            if not isinstance(fields, dict) or not isinstance(targets, list):
                continue
            filtered_fields = {
                str(name): copy.deepcopy(value)
                for name, value in fields.items()
                if str(name) in entry_fields
            }
            if not filtered_fields:
                continue
            for target in targets:
                try:
                    trial_id = int(target)
                except (TypeError, ValueError):
                    continue
                overrides_by_trial.setdefault(trial_id, {}).update(
                    copy.deepcopy(filtered_fields)
                )
        return overrides_by_trial

    def entries_with_supersessions(self) -> list[JournalEntry]:
        """Return trial entries with append-only supersession events folded in.

        This is the runtime read view: persisted trial rows stay immutable, while
        planner-facing trust and prompt helpers see operator supersession events
        such as post-hoc resource-contention exclusions.
        """
        overrides_by_trial = self._supersession_overrides_by_trial()
        if not overrides_by_trial:
            return list(self._entries)
        entries: list[JournalEntry] = []
        for entry in self._entries:
            overrides = overrides_by_trial.get(entry.trial_id)
            if overrides:
                entries.append(replace(entry, **copy.deepcopy(overrides)))
            else:
                entries.append(entry)
        return entries

    def next_trial_id(self) -> int:
        if not self._entries:
            return 0
        return self._entries[-1].trial_id + 1

    def by_species(self, species: str) -> list[JournalEntry]:
        return [e for e in self._entries if e.species == species]

    def pareto_entries(self) -> list[JournalEntry]:
        return [e for e in self._entries if e.pareto_status == "frontier"]

    def summary(self) -> dict[str, Any]:
        """Compact summary for controller consumption."""
        if not self._entries:
            return {"total_trials": 0, "species_counts": {}, "pareto_size": 0}

        species_counts: dict[str, int] = {}
        pareto_count = 0
        for e in self._entries:
            species_counts[e.species] = species_counts.get(e.species, 0) + 1
            if e.pareto_status == "frontier":
                pareto_count += 1

        last = self._entries[-1]
        return {
            "total_trials": len(self._entries),
            "species_counts": species_counts,
            "pareto_size": pareto_count,
            "last_trial_id": last.trial_id,
            "last_species": last.species,
            "last_quality": last.quality,
            "last_speed": last.speed,
        }

    def summary_text(self, last_n: int = 20) -> str:
        """Human-readable summary for LLM controller prompt."""
        s = self.summary()
        entries = self.entries_with_supersessions()
        eligible_pareto_count = sum(
            1
            for e in entries
            if e.pareto_status == "frontier" and e.tier > 0 and not e.bug_corrupted_by
        )
        lines = [
            f"Total trials: {s['total_trials']}",
            f"Eligible Pareto frontier size (T1/T2, trustworthy): {eligible_pareto_count}",
            f"Species counts: {s.get('species_counts', {})}",
        ]
        recent = entries[-last_n:]
        if recent:
            lines.append(f"\nLast {len(recent)} trials:")
            for i, e in enumerate(recent):
                prefix = f"  #{e.trial_id} [{e.species}/{e.action_type}] "
                if e.bug_corrupted_by:
                    label = (
                        "EXCLUDED_BY"
                        if str(e.bug_corrupted_by) == "seq_refuted"
                        else "CORRUPTED_BY"
                    )
                    line = (
                        prefix
                        + f"{label}={e.bug_corrupted_by} "
                        + "(metrics/reason hidden; excluded from planner trust)"
                    )
                    lines.append(line)
                    continue
                if e.tier == 0:
                    line = (
                        prefix
                        + "T0 audit-only sentinel "
                        + "(quality hidden; excluded from production frontier/baseline guards) "
                        + f"s={e.speed:.1f} c={e.cost:.3f} r={e.reliability:.2f} "
                        + f"→ {e.pareto_status}"
                    )
                else:
                    line = (
                        prefix
                        + f"T{e.tier} q={e.quality:.3f} s={e.speed:.1f} "
                        + f"c={e.cost:.3f} r={e.reliability:.2f} "
                        + f"→ {e.pareto_status}"
                    )
                if has_negative_evidence(e):
                    # Compact single-line failure summary for controller visibility.
                    # Prompt-budget trim (2026-06-10): full detail only for the most
                    # recent 6 failures (the ones the planner actually reasons about
                    # next); older failures show a short tag so the journal section
                    # stays small. Cap shortened 200→140.
                    if i >= len(recent) - 6:
                        fa_oneline = negative_evidence_for_prompt(e, entries=entries, limit=140)
                        line += f"  FAILED: {fa_oneline}"
                    else:
                        fa_oneline = negative_evidence_for_prompt(e, entries=entries, limit=60)
                        line += f"  FAILED({fa_oneline})"
                lines.append(line)
        return "\n".join(lines)

    # ── bug-corruption tracking (2026-05-23) ─────────────────────

    def trustworthy_entries(self) -> list[JournalEntry]:
        """All entries whose outcome is NOT marked bug_corrupted_by."""
        return [e for e in self.entries_with_supersessions() if not e.bug_corrupted_by]

    def trustworthiness_score(self) -> dict[str, Any]:
        """Counts of trustworthy vs bug-corrupted entries.

        Returned shape:
            {
                "total": N,
                "trustworthy": M,
                "corrupted": N - M,
                "ratio": M / N (or 1.0 when N == 0),
                "corrupted_by": {"de34dd4": K1, "b3895aa": K2, ...},
                "low_signal": bool,  # True when trustworthy < 5
            }
        """
        entries = self.entries_with_supersessions()
        total = len(entries)
        if total == 0:
            return {
                "total": 0, "trustworthy": 0, "corrupted": 0,
                "ratio": 1.0, "corrupted_by": {}, "low_signal": True,
            }
        corrupted_by: dict[str, int] = {}
        trustworthy = 0
        for e in entries:
            sha = e.bug_corrupted_by or ""
            if sha:
                corrupted_by[sha] = corrupted_by.get(sha, 0) + 1
            else:
                trustworthy += 1
        return {
            "total": total,
            "trustworthy": trustworthy,
            "corrupted": total - trustworthy,
            "ratio": trustworthy / total,
            "corrupted_by": corrupted_by,
            "low_signal": trustworthy < 5,
        }

    def recent_hypotheses(self, n: int = 3, exclude_bug_corrupted: bool = True) -> list[JournalEntry]:
        """Return the last n entries with a non-empty hypothesis field.

        When `exclude_bug_corrupted` is True, skip entries marked by
        scrub_journal so the planner's hypothesis-chain reasoning learns
        from real signal only.
        """
        pool = (
            self.trustworthy_entries() if exclude_bug_corrupted else self._entries
        )
        with_hyp = [e for e in pool if e.hypothesis]
        return with_hyp[-n:]

    def action_diversity_by_gate(
        self, window: int = 50
    ) -> dict[str, Any]:
        """Compare action-type diversity for rich-fragment trials (stagnation
        signal fired) vs lean-fragment trials (empty signal) over the last
        `window` controller-mode entries.

        Returns counts, distinct action_types, and Shannon entropy per bucket
        so operators can verify the gate is doing what it should: rich-prompt
        trials ought to explore more action_types than lean-prompt trials.
        Empty buckets return entropy=0.0 with count=0.
        """
        import math
        recent = self._entries[-window:]
        rich = [e for e in recent if e.stagnation_signal]
        lean = [e for e in recent if not e.stagnation_signal]

        def _stats(bucket: list[JournalEntry]) -> dict[str, Any]:
            if not bucket:
                return {"count": 0, "distinct_action_types": 0, "entropy_bits": 0.0}
            counts: dict[str, int] = {}
            for e in bucket:
                counts[e.action_type] = counts.get(e.action_type, 0) + 1
            total = sum(counts.values())
            h = 0.0
            for c in counts.values():
                p = c / total
                if p > 0:
                    h -= p * math.log2(p)
            return {
                "count": total,
                "distinct_action_types": len(counts),
                "entropy_bits": h,
                "histogram": counts,
            }

        return {"window": window, "rich": _stats(rich), "lean": _stats(lean)}

    def unfalsified_hypotheses(
        self, n: int = 5, exclude_bug_corrupted: bool = True
    ) -> list[tuple[int, str, str]]:
        """Return [(trial_id, hypothesis, falsifier)] for the last `n` trustworthy
        trials that carry both a hypothesis and an explicit falsifier.

        The planner consumes this list to grade new candidates against still-open
        claims — i.e. predictions that have not yet been resolved either way.
        Resolution-checking is intentionally minimal here (presence of the
        falsifier string only); semantic matching is deferred to the controller,
        which sees the trial's actual outcome via the journal summary.
        """
        pool = (
            self.trustworthy_entries() if exclude_bug_corrupted else self._entries
        )
        with_falsifier = [
            (e.trial_id, e.hypothesis, e.falsifier)
            for e in pool
            if e.hypothesis and e.falsifier
        ]
        return with_falsifier[-n:]

    def matching_trial_ids(
        self,
        *,
        trial_id_min: int | None = None,
        trial_id_max: int | None = None,
        timestamp_min: str | None = None,
        timestamp_max: str | None = None,
    ) -> list[int]:
        """Trial IDs matching the scrub filter without mutating entries."""
        matched: list[int] = []
        for e in self._entries:
            if trial_id_min is not None and e.trial_id < trial_id_min:
                continue
            if trial_id_max is not None and e.trial_id > trial_id_max:
                continue
            if timestamp_min is not None and (e.timestamp or "") < timestamp_min:
                continue
            if timestamp_max is not None and (e.timestamp or "") > timestamp_max:
                continue
            matched.append(e.trial_id)
        return matched

    def recent_failures(
        self,
        species: str | None = None,
        n: int = 10,
        exclude_bug_corrupted: bool = True,
    ) -> list[JournalEntry]:
        """Return the last n entries with non-empty failure_analysis.

        Optionally filter by species name.
        """
        entries = (
            self.entries_with_supersessions()
            if exclude_bug_corrupted
            else self._entries
        )
        failed = [
            e for e in entries
            if has_negative_evidence(e)
            and (not exclude_bug_corrupted or is_learning_eligible(e))
            and (species is None or e.species == species)
        ]
        return failed[-n:]

    def failure_analysis_for_prompt(
        self, entry: JournalEntry, limit: int | None = None
    ) -> str:
        return negative_evidence_for_prompt(entry, entries=self.entries_with_supersessions(), limit=limit)

    def suite_quality_trend(
        self, last_n: int = 10
    ) -> dict[str, list[tuple[int, float]]]:
        """Per-suite quality over the last n trials that have suite data.

        Returns {suite_name: [(trial_id, quality), ...]} sorted by trial_id.
        """
        entries_with_suites = [
            e for e in self._entries
            if e.eval_details.get("per_suite_quality")
        ][-last_n:]

        trends: dict[str, list[tuple[int, float]]] = {}
        for e in entries_with_suites:
            for suite, q in e.eval_details["per_suite_quality"].items():
                trends.setdefault(suite, []).append((e.trial_id, q))
        return trends

    # ── insights ──────────────────────────────────────────────────

    def insights_text(self, n: int = 10) -> str:
        """Synthesize actionable insights from recent trials.

        Extracts hypothesis + outcome from trials that either reached the
        Pareto frontier or failed safety gates — the two outcomes worth
        learning from.  Returns a compact text block suitable for injection
        into species prompts (cross-species fertilization).
        """
        history = self.entries_with_supersessions()
        interesting = [
            e for e in history
            if e.pareto_status == "frontier" or has_negative_evidence(e)
            if is_learning_eligible(e)
        ][-n:]
        if not interesting:
            return "(no insights yet)"

        lines: list[str] = []
        for e in interesting:
            tag = "SUCCESS" if e.pareto_status == "frontier" else "FAILED"
            hyp = e.hypothesis or e.action_type
            mechanism = e.expected_mechanism or ""
            detail = ""
            if e.pareto_status == "frontier":
                detail = f"q={e.quality:.3f} s={e.speed:.1f}"
            elif has_negative_evidence(e):
                # Compact single-line failure summary
                detail = negative_evidence_for_prompt(e, entries=history, limit=120)
            species_label = e.species
            lines.append(
                f"  [{tag}] #{e.trial_id} ({species_label}/{hyp})"
                + (f" [{mechanism}]" if mechanism else "")
                + f": {detail}"
            )
        return "\n".join(lines)

    def insights_structured(
        self,
        n: int = 30,
        exclude_bug_corrupted: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Group recent insightful trials by action_type to expose pattern + confidence.

        Each action_type bucket carries:
          - observation: one-line synthesis from the most-recent trial in
            that bucket (operator can scan; planner cites trial IDs as
            evidence).
          - trials_supporting: list[trial_id] in chronological order.
          - successes / failures: counts within the bucket.
          - confidence: 'high' (≥3 supporting, success_rate ≥ 0.66),
            'medium' (≥2 supporting), or 'low' (else).
          - latest_q / latest_s: quality + speed of the most-recent trial,
            for at-a-glance trend visibility.

        Bug-corrupted entries are excluded by default so the planner doesn't
        learn from poisoned signal.
        """
        history = self.entries_with_supersessions()
        pool = (
            [e for e in history if is_learning_eligible(e)]
            if exclude_bug_corrupted else self._entries
        )
        recent = pool[-n:]
        if not recent:
            return {}

        buckets: dict[str, list[JournalEntry]] = {}
        for e in recent:
            if not (e.pareto_status == "frontier" or has_negative_evidence(e)):
                continue
            buckets.setdefault(e.action_type or "unknown", []).append(e)

        out: dict[str, dict[str, Any]] = {}
        for action_type, entries in buckets.items():
            entries.sort(key=lambda x: x.trial_id)
            successes = sum(1 for e in entries if e.pareto_status == "frontier")
            failures = sum(1 for e in entries if has_negative_evidence(e) and e.pareto_status != "frontier")
            n_sup = len(entries)
            success_rate = successes / n_sup if n_sup else 0.0
            if n_sup >= 3 and success_rate >= 0.66:
                confidence = "high"
            elif n_sup >= 2:
                confidence = "medium"
            else:
                confidence = "low"
            latest = entries[-1]
            observation = latest.hypothesis or latest.expected_mechanism or (
                negative_evidence_for_prompt(latest, entries=history, limit=160)
            ) or "(no description)"
            out[action_type] = {
                "observation": observation[:240],
                "negative_evidence": grouped_negative_evidence(
                    entries, history),
                "trials_supporting": [e.trial_id for e in entries],
                "successes": successes,
                "failures": failures,
                "confidence": confidence,
                "latest_trial_id": latest.trial_id,
                "latest_q": latest.quality,
                "latest_s": latest.speed,
                "latest_outcome": (
                    "frontier" if latest.pareto_status == "frontier" else "failed"
                ),
            }
        return out

    def action_distribution(self, last_n: int = 30) -> dict[str, int]:
        """Count of each action_type seen in the last n entries (newest-first window).

        Used by the Adjacent-Possible / tail-sampling section to identify
        action types that are over- vs under-represented in recent history.
        """
        counts: dict[str, int] = {}
        for e in self._entries[-last_n:]:
            t = e.action_type or "unknown"
            counts[t] = counts.get(t, 0) + 1
        return counts

    def tail_action_candidates(
        self,
        known_action_types: list[str],
        last_n: int = 30,
        n_sample: int = 3,
    ) -> list[str]:
        """Sample n_sample action types that are under-represented in recent history.

        "Tail" means actions that show up 0–1 times in the last `last_n`
        trials. Used to push the planner toward considering creative
        alternatives outside its recent comfort zone (breaks local-optimum
        traps).
        """
        import random
        dist = self.action_distribution(last_n=last_n)
        tail = [t for t in known_action_types if dist.get(t, 0) <= 1]
        random.shuffle(tail)
        return tail[:n_sample]

    def insights_structured_text(
        self, n: int = 30, exclude_bug_corrupted: bool = True
    ) -> str:
        """Render insights_structured() for inclusion in the controller prompt."""
        d = self.insights_structured(n=n, exclude_bug_corrupted=exclude_bug_corrupted)
        if not d:
            return "(no insights yet — seed more trials to build evidence)"
        # Sort by confidence desc, then n supporting desc
        rank = {"high": 0, "medium": 1, "low": 2}
        items = sorted(
            d.items(),
            key=lambda kv: (rank[kv[1]["confidence"]], -len(kv[1]["trials_supporting"])),
        )
        lines: list[str] = []
        for action_type, info in items:
            lines.append(
                f"{action_type} ({info['confidence']} confidence, "
                f"{info['successes']}+/{info['failures']}- across "
                f"trials {info['trials_supporting']}):"
            )
            lines.append(f"  Observation: {info['observation']}")
            for negative in info["negative_evidence"]:
                lines.append(f"  Negative: {negative}")
            lines.append(
                f"  Latest trial #{info['latest_trial_id']} → "
                f"{info['latest_outcome']} (q={info['latest_q']:.3f} "
                f"sp={info['latest_s']:.1f})"
            )
        return "\n".join(lines)

    # ── species effectiveness ────────────────────────────────────

    def species_effectiveness(
        self, window: int | None = None
    ) -> dict[str, dict[str, float]]:
        """Effectiveness signals per species.

        ``rate`` remains the legacy Pareto-frontier rate for reports.  Budget
        rebalancing should prefer ``budget_rate``: PEAF surprise-derived
        realized information gain when present, otherwise the legacy frontier
        rate, plus a small clipped credit for successful higher-tier workflow
        eval rows.  This lets informative non-frontier trials and T2/T3
        validation successes earn exploration budget without changing old
        consumers that display Pareto counts or any production-promotion gate.
        """
        entries = self._entries[-window:] if window else self._entries
        stats: dict[str, dict[str, float]] = {}
        for e in entries:
            if e.species not in stats:
                stats[e.species] = {
                    "total": 0,
                    "pareto": 0,
                    "rate": 0.0,
                    "information_gain": 0.0,
                    "information_count": 0,
                    "information_rate": 0.0,
                    "higher_tier_quality_gain": 0.0,
                    "higher_tier_quality_count": 0,
                    "higher_tier_quality_rate": 0.0,
                    "budget_rate": 0.0,
                }
            stats[e.species]["total"] += 1
            if e.pareto_status == "frontier":
                stats[e.species]["pareto"] += 1
            trusted_ok = (
                not e.bug_corrupted_by
                and e.outcome_status == "ok"
                and not e.failure_analysis
            )
            if (
                e.surprise_score is not None
                and trusted_ok
            ):
                # PEAF surprise is already normalized objective-space distance.
                # Cap per-trial credit so one wild forecast miss cannot dominate
                # species allocation.
                info_gain = min(max(float(e.surprise_score), 0.0), 1.0)
                stats[e.species]["information_gain"] += info_gain
                stats[e.species]["information_count"] += 1
            if int(e.tier) > DEFAULT_FRONTIER_TIER and trusted_ok:
                # Tier quality is same-tier only and never compared against T1.
                # Credit is normalized within the 0-3 scale, then clipped by the
                # final blend below so validation lanes can steer exploration
                # without becoming deployment authority.
                quality_gain = min(max(float(e.quality) / 3.0, 0.0), 1.0)
                stats[e.species]["higher_tier_quality_gain"] += quality_gain
                stats[e.species]["higher_tier_quality_count"] += 1
        for sp in stats:
            total = stats[sp]["total"]
            stats[sp]["rate"] = stats[sp]["pareto"] / total if total > 0 else 0.0
            info_count = stats[sp]["information_count"]
            if info_count > 0:
                stats[sp]["information_rate"] = (
                    stats[sp]["information_gain"] / info_count
                )
                stats[sp]["budget_rate"] = stats[sp]["information_rate"]
            else:
                stats[sp]["budget_rate"] = stats[sp]["rate"]
            higher_count = stats[sp]["higher_tier_quality_count"]
            if higher_count > 0:
                stats[sp]["higher_tier_quality_rate"] = (
                    stats[sp]["higher_tier_quality_gain"] / higher_count
                )
                stats[sp]["budget_rate"] = min(
                    1.0,
                    stats[sp]["budget_rate"]
                    + _HIGHER_TIER_BUDGET_CREDIT_WEIGHT
                    * stats[sp]["higher_tier_quality_rate"],
                )
        return stats
