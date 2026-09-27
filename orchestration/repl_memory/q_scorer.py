"""
QScorer: Async Q-value update agent for episodic memory.

Runs periodically (or on-demand) to:
1. Read progress logs for completed tasks
2. Compute rewards from outcomes
3. Update Q-values in the episodic store
4. Optionally run Claude-as-Judge for graded rewards

This implements the async scoring path from the MemRL architecture,
keeping Q-value computation off the critical inference path.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# DAR-2: Contrastive Q-value updates (decision-aware-routing.md).
# When enabled, routing Q-updates include a contrastive adjustment that
# sharpens decision boundaries between alternative models. The adjustment
# is additive to the reward signal, capped at ±0.1, and zero when the
# ranking is already correct with sufficient margin.
CONTRASTIVE_Q_UPDATES = os.environ.get("CONTRASTIVE_Q_UPDATES", "1") == "1"

# DAR-3: SPO+ with exploration. Replaces the contrastive adjustment (DAR-2)
# with a convex surrogate loss that drives the selected model's Q-value toward
# the ranking that matches true costs. Epsilon-greedy exploration (default 10%)
# is configured in retriever.py. The SPO+ adjustment is computed here when
# counterfactual data (alternative Q-values) is available.
SPO_PLUS_ENABLED = os.environ.get("SPO_PLUS_ENABLED", "0") == "1"
SPO_PLUS_MARGIN = float(os.environ.get("SPO_PLUS_MARGIN", "0.05"))

# DAR-L491 write-path fix. The production routing scorer path
# (_update_routing_memory) only TD-updates when routing_decision.memory_id is
# pre-linked; the sole ROUTING_DECISION emitter (progress_logger.log_task_started)
# never sets it, so every scored routing decision fell through to a blind
# store() append. Result: 99.7% of routing rows never TD-update and the learned
# signal is an append-only buffer (176.8x duplicate (objective, action) pairs;
# see scripts/analysis/dar_write_path_audit.py). When ORCHESTRATOR_Q_TD_WRITE is
# set, the append branch first find-or-updates the existing (objective, action)
# row in place. Default OFF keeps byte-identical legacy append behavior so
# deployment is an explicit operator-boundary action.
Q_TD_WRITE = os.environ.get("ORCHESTRATOR_Q_TD_WRITE", "0") == "1"
# Candidate depth for the find-or-update similarity lookup on the append path.
Q_TD_MATCH_K = int(os.environ.get("ORCHESTRATOR_Q_TD_MATCH_K", "10"))

from src.autopilot_core.measurement_guards import (
    FAILURE_DISPOSITION_KEY,
    NON_QUALITY_DISPOSITIONS,
)
from src.classifiers.role_taxonomy import VALID_TRINITY_ROLES
from src.roles import Role

from .embedder import TaskEmbedder
from .episodic_store import EpisodicStore, outcome_from_reward
from .memory_record import (
    WORK_KEYS,
    build_memory_record,
    build_work_payload,
    extract_work,
)
from .progress_logger import EventType, ProgressEntry, ProgressLogger, ProgressReader
from .staged_scorer import StagedQScorer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
)
DEFAULT_MODEL_DESCRIPTOR_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "model_descriptors.yaml"
)
DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "derived" / "stack_priors.yaml"
)
# RTG-09: per-role wall-clock task-duration baselines (p50/p90), derived from
# retained progress logs by scripts/analysis/derive_duration_baselines.py
# under protocol id RTG09-DURATION-BASELINE-v1. Measured, not declared -- a
# different provenance from stack_priors.yaml above (which reflects the
# CURRENT stack's declared/benchmarked specs), so it is loaded independently
# rather than folded into descriptor_q_scorer_priors_by_role().
DEFAULT_DURATION_BASELINES_PATH = (
    Path(__file__).resolve().parents[2]
    / "orchestration"
    / "derived"
    / "duration_baselines_by_role.json"
)


def load_duration_baselines_by_role(
    path: Path = DEFAULT_DURATION_BASELINES_PATH,
) -> Tuple[Dict[str, Dict[str, float]], Optional[str]]:
    """Load per-role wall-clock duration baselines for the reward's speed axis.

    Returns (baselines, degraded_reason). On any failure to load, returns
    ({}, reason) -- an EMPTY dict, never a fabricated number. `q_reward.py`'s
    "role not in config.baseline_duration_by_role" guard already treats an
    absent role honestly (SKIPS the duration dimension and warns once), so a
    wholesale-empty dict degrades to "duration dimension never fires,
    surfaced via ScoringConfig.duration_baseline_degraded_reason" rather than
    silently pricing every role at some default duration.
    """
    import json

    if not path.exists():
        return {}, f"duration baselines artifact missing: {path}"
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {}, f"duration baselines artifact unreadable: {exc}"
    roles = doc.get("roles") if isinstance(doc, dict) else None
    if not isinstance(roles, dict):
        return {}, "duration baselines artifact malformed: no 'roles' object"
    out: Dict[str, Dict[str, float]] = {}
    for role, stats in roles.items():
        if not isinstance(stats, dict):
            continue
        p50 = stats.get("p50_s")
        p90 = stats.get("p90_s")
        if isinstance(p50, (int, float)) and isinstance(p90, (int, float)):
            out[str(role)] = {"p50_s": float(p50), "p90_s": float(p90)}
    return out, None

FALLBACK_BASELINE_TPS_BY_ROLE: Dict[str, float] = {
    "frontdoor": 12.7,
    "coder_escalation": 10.8,
    "architect_general": 4.3,
    "ingest_long_context": 12.0,
    "worker_general": 38.46,
    "worker_math": 38.46,
    "toolrunner": 38.46,
    "worker_vision": 15.28,
    "vision_escalation": 27.6,
}

# 2026-08-02: KEYED BY MODEL ARTIFACT, NOT BY ROLE.
#
# This was `BASELINE_QUALITY_BY_ROLE`, and role-keying made it INTERNALLY
# CONTRADICTORY BY CONSTRUCTION — roles share models, so the same GGUF carried two
# different quality numbers:
#
#   architect_general 0.94  and coder_escalation 0.915  -> both Qwen3.6-27B-MTP-Q8_0
#   worker_general    0.745 and worker_math       0.85   -> both gemma-4-26B-A4B
#
# Each stale value is the quality of the model that role USED to serve: 0.94 is the
# Qwen3.5-122B's (which moved to architect_critic in the 2026-08-01 W1 cutover) and
# 0.915 is frontdoor's 35B's. So a role repoint silently transferred another model's
# quality onto the new one, and the router scored a 27B with a 122B's number.
#
# Quality is a property of a MODEL under an instrument, never of a role name. Keying
# by role means every future repoint re-introduces this defect. Keying by artifact
# means a repointed role either resolves to its own model's measured quality or gets
# NO entry — and `q_reward.py:147` already guards with `if role in ...`, so a missing
# entry cleanly SKIPS the quality-gap penalty instead of fabricating one.
#
# Values below are retained only for artifacts whose figure was actually measured on
# THAT artifact. Qwen3.6-27B-MTP-Q8_0 is deliberately ABSENT: it has ratified
# SWE-bench Verified evidence (23/40, 57.5%) but no canonical 79-question judge-suite
# run, and SWE-bench is a different instrument on a different scale. An absent entry
# is the honest representation of that.
# DEGRADED-PATH FALLBACK ONLY. The authoritative source is
# `priors.quality_overall` in the compiled stack priors, which is the mean over
# the fixed universal benchmark set in orchestration/public_benchmarks.yaml
# (mmlu_pro, gpqa_diamond, livecodebench_v6). These entries MIRROR that
# derivation so a priors-less run does not silently disagree with a normal one.
#
# 2026-08-02: they previously did disagree. Qwen3.6-35B-A3B carried 0.895 here,
# introduced 2026-02-13 — three months BEFORE the 2026-05-04 benchmark that put
# quality_pct: 93 in the registry — so the router's fallback quality had been
# stale for two measurement cycles while the measured value sat unread.
# If you change public_benchmarks.yaml, recompile and re-mirror these.
BASELINE_QUALITY_BY_MODEL: Dict[str, float] = {
    "Qwen3.6-35B-A3B-MTP-Q8_0.gguf": 0.8387,
    "Qwen3.6-27B-MTP-Q8_0.gguf": 0.8597,
    "gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf": 0.8067,
}


def _model_id_for_action(action: str | None) -> str | None:
    """Resolve a routing/escalation action to the model_id that served it.

    `MemoryEntry.model_id` is documented as "model that produced this memory
    (for warm-start)" and was NULL on all 59,337 rows: `store()` has always
    accepted the argument and no caller ever passed it (M-11a2). Warm-start and
    any per-model analysis over the episodic corpus were therefore impossible.

    DERIVED from the compiled priors so it follows the live lineup instead of
    restating a role->model table that would drift the next time a role changes
    weights — which happened twice in the past week (architect_general
    122B->27B, coder_escalation 35B->27B).

    Returns None for an unknown action, so NULL keeps meaning "unknown" rather
    than inventing an attribution.

    NOTE what this function must NOT be used for. `assigned_role` is the Trinity
    tri-role axis (thinker/worker/verifier) and `sub_decision` is the
    orchestration sub-decision label; a SERVING role like "frontdoor" is a wrong
    value in either. `assigned_role` is instead read off the progress entry that
    already carries the classifier's output — see `_assigned_role_from_entry`.
    `sub_decision` stays NULL, which correctly means "not a sub-decision".
    """
    if not action:
        return None
    role = str(action).strip().split(".", 1)[0]
    if not role:
        return None
    try:
        import yaml

        priors = yaml.safe_load(DEFAULT_STACK_PRIORS_PATH.read_text()) or {}
    except Exception:  # noqa: BLE001
        return None
    record = (priors.get("roles") or {}).get(role)
    if not isinstance(record, dict):
        return None
    model_id = record.get("model_id")
    return str(model_id) if model_id else None


def _assigned_role_from_entry(entry: Optional[ProgressEntry]) -> Optional[str]:
    """Read the Trinity tri-role off a progress entry, or None.

    TR-3.2 classifies every request (`classify_trinity_role` in
    `src/api/routes/chat_pipeline/routing_decision.py`), and
    `progress_logger.log_task_started` merges that `routing_meta` into the
    ROUTING_DECISION entry's `data` — so `data["assigned_role"]` is the
    classifier's own output, already durable in the progress JSONL. It stopped
    there: no episodic write site ever read it back, so `memories.assigned_role`
    was NULL on all 59,337 rows and the TR-3.3 promotion decision had no
    shadow telemetry in the durable corpus to decide on.

    VALIDATE, DO NOT COERCE. `role_taxonomy.normalise_role` maps anything
    unrecognised (including None) to "worker", which is the correct *read*-side
    default per TR-1.5 but the wrong *write*-side behavior: it would stamp a
    real "worker" onto rows where the role is genuinely unknown, and silently
    launder a stale/foreign string into a Trinity role. Unknown must stay NULL
    so "worker" in this column always means the classifier said worker.
    """
    data = getattr(entry, "data", None) if entry is not None else None
    if not isinstance(data, dict):
        return None
    raw = data.get("assigned_role")
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    return candidate if candidate in VALID_TRINITY_ROLES else None


def _baseline_quality_by_role(priors: dict | None = None) -> Dict[str, float]:
    """Project the model-keyed table onto live roles via the compiled priors.

    A role whose model has no measured entry is OMITTED rather than defaulted.
    Falling back to a number would reinstate exactly the defect this replaced.
    """
    if priors is None:
        try:
            import yaml

            from pathlib import Path

            path = (
                Path(__file__).resolve().parents[2]
                / "orchestration"
                / "derived"
                / "stack_priors.yaml"
            )
            priors = yaml.safe_load(path.read_text()) or {}
        except Exception:
            # No compiled priors -> no role projection is possible. Return empty
            # rather than guessing; the consumer skips the penalty.
            return {}
    out: Dict[str, float] = {}
    for role, record in (priors.get("roles") or {}).items():
        if not isinstance(record, dict):
            continue
        if role in _NON_SCORING_HOST_ALIASES:
            # Host-only alias (e.g. "worker"): never a q_scorer scoring role,
            # see _NON_SCORING_HOST_ALIASES. Its measurement is carried by its
            # named members already.
            continue

        # PREFERRED: the role-facing figure the priors compiler already chose —
        # the role-relevant capability axis when it can rank the fleet, else the
        # universal aggregate, with `basis` recording which. Reading this rather
        # than a model-keyed constant is what makes quality multidimensional at
        # the point of use: architect_general and coder_escalation ran the SAME
        # 27B weights (its host is architect_critic since the 2026-09-27 ARCHITECT
        # SWAP) and scored 0.862 (reasoning/mmlu_pro) and 0.839
        # (coding/livecodebench_v6) respectively, because they do different jobs.
        for_role = (record.get("priors") or {}).get("quality_for_role")
        if isinstance(for_role, dict):
            value = for_role.get("value")
            if isinstance(value, (int, float)) and 0 <= float(value) <= 1:
                out[role] = float(value)
                continue

        # FALLBACK: model-keyed table, for a priors file predating the axis
        # fields. Still model-keyed, never role-keyed.
        req = (
            ((record.get("serving") or {}).get("launch") or {}).get("requirements")
            or {}
        )
        model_path = req.get("model_path")
        if not isinstance(model_path, str) or not model_path:
            continue
        quality = BASELINE_QUALITY_BY_MODEL.get(model_path.rsplit("/", 1)[-1])
        if quality is not None:
            out[role] = quality
    return out

FALLBACK_MEMORY_COST_BY_ROLE: Dict[str, float] = {
    "frontdoor": 1.0,
    "coder_escalation": 1.0,
    "architect_general": 1.0,
    "ingest_long_context": 1.0,
    "worker_general": 1.0,
    "worker_math": 1.0,
    "toolrunner": 1.0,
    "worker_vision": 1.0,
    "vision_escalation": 1.0,
}

SERVER_MODE_TPS_ROLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "frontdoor": ("frontdoor",),
    "coder_escalation": ("coder_escalation",),
    "architect_general": ("architect_general",),
    "ingest_long_context": ("ingest_long_context",),
    # The production worker server is shared by these scorer roles.
    "worker": ("worker_explore", "worker_general", "worker_math", "toolrunner"),
}

ROLE_PERFORMANCE_TPS_FALLBACKS: Dict[str, str] = {
    "worker_vision": "worker_vision",
    "vision_escalation": "vision_escalation",
}

REGISTRY_MEMORY_ROLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "frontdoor": ("frontdoor",),
    "coder_escalation": ("coder_escalation",),
    "architect_general": ("architect_general",),
    "ingest_long_context": ("ingest_long_context",),
    "worker": ("worker_explore", "worker_general", "worker_math", "toolrunner"),
}

STACK_PRIOR_SCORER_ROLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    # worker_explore remains a q_scorer action label but shares the live worker
    # serving/cost priors compiled under worker_general.
    "worker_general": ("worker_explore",),
}

# 2026-09-24: host-only aliases that must NEVER surface as their own scoring
# role. "worker" is declared in the registry/launch manifest purely so a
# legacy URL/topology lookup still resolves (server_mode.worker: alias_of
# frontdoor, shared_with: [worker_general, worker_math, worker_explore,
# toolrunner, worker_summarize, ...]) -- it compiles into
# orchestration/derived/stack_priors.yaml with deployment_status: live_stack
# because it genuinely is part of the live serving topology, but it duplicates
# information already carried by its named members and is not a `Role`. The
# registry-path loader (SERVER_MODE_TPS_ROLE_ALIASES / REGISTRY_MEMORY_ROLE_
# ALIASES above) already gets this right: it reads server_mode.worker only to
# PROJECT onto worker_explore/worker_general/worker_math/toolrunner, and never
# keeps "worker" itself as an output key. Derived generically (not just
# hardcoded to "worker") so any future non-Role host key in that same alias
# table is caught the same way: a key of SERVER_MODE_TPS_ROLE_ALIASES that is
# not itself a canonical `Role` is a host name, never a scoring role.
_NON_SCORING_HOST_ALIASES: frozenset[str] = frozenset(
    server_key
    for server_key in SERVER_MODE_TPS_ROLE_ALIASES
    if Role.from_string(server_key) is None
)

PRIOR_SOURCE_STACK_PRIORS = "stack_priors"
PRIOR_SOURCE_MODEL_DESCRIPTORS = "model_descriptors"
PRIOR_SOURCE_REGISTRY = "registry"
PRIOR_SOURCE_DEGRADED_FALLBACK = "degraded_fallback"

# 2026-08-12 (NIB2-57a): `roles.*.performance.optimized_tps` and
# `roles.*.performance.baseline_tps` are DIFFERENT MEASUREMENTS — the same model
# under the production-optimal recipe and under the unoptimized reference recipe.
# This loader used to collapse them with `optimized or baseline`, so a caller
# received a bare float and could not tell which measurement it held. Three
# distinct registry states now get three distinct source labels, so the
# unmeasured-prior case is loud instead of silent.
#   ..._OPTIMIZED    -> an optimized_tps that is genuinely separate from baseline.
#   ..._SUBSTITUTED  -> NO optimized_tps exists; the unoptimized baseline_tps is
#                       standing in for it. The number understates achievable t/s,
#                       so every cost penalty computed from it is under-applied.
#   ..._UNSEPARATED  -> both fields are present and numerically EQUAL, i.e. one
#                       measurement was written into both. The pair carries no
#                       optimization evidence; it is indistinguishable from a
#                       genuinely-optimized role unless the loader says so.
PRIOR_SOURCE_REGISTRY_OPTIMIZED_TPS = "registry_performance_optimized_tps"
PRIOR_SOURCE_REGISTRY_BASELINE_TPS_SUBSTITUTED = (
    "registry_performance_baseline_tps_substituted"
)
PRIOR_SOURCE_REGISTRY_TPS_UNSEPARATED = "registry_performance_tps_unseparated"


class QScorerPriorSourceError(RuntimeError):
    """Live q_scorer priors are missing generated stack-prior provenance."""


def _materialize_q_scorer_role_aliases(values: Dict[str, Any]) -> Dict[str, Any]:
    materialized = dict(values)
    for canonical_role, aliases in STACK_PRIOR_SCORER_ROLE_ALIASES.items():
        if canonical_role not in materialized:
            continue
        canonical_value = materialized[canonical_role]
        for alias in aliases:
            materialized.setdefault(alias, canonical_value)
    return materialized


def _materialize_q_scorer_priors(priors: QScorerPriors) -> QScorerPriors:
    return QScorerPriors(
        baseline_tps_by_role=_materialize_q_scorer_role_aliases(
            priors.baseline_tps_by_role
        ),
        baseline_quality_by_role=_materialize_q_scorer_role_aliases(
            priors.baseline_quality_by_role
        ),
        memory_cost_by_role=_materialize_q_scorer_role_aliases(priors.memory_cost_by_role),
        baseline_tps_source_by_role=_materialize_q_scorer_role_aliases(
            priors.baseline_tps_source_by_role
        ),
        baseline_quality_source_by_role=_materialize_q_scorer_role_aliases(
            priors.baseline_quality_source_by_role
        ),
        memory_cost_source_by_role=_materialize_q_scorer_role_aliases(
            priors.memory_cost_source_by_role
        ),
        degraded_reason=priors.degraded_reason,
    )


@dataclass(frozen=True)
class QScorerPriors:
    """Descriptor-aware scorer priors with registry-backed fallbacks."""

    baseline_tps_by_role: Dict[str, float]
    baseline_quality_by_role: Dict[str, float]
    memory_cost_by_role: Dict[str, float]
    baseline_tps_source_by_role: Dict[str, str] = field(default_factory=dict)
    baseline_quality_source_by_role: Dict[str, str] = field(default_factory=dict)
    memory_cost_source_by_role: Dict[str, str] = field(default_factory=dict)
    degraded_reason: str | None = None

    @property
    def uses_degraded_fallback(self) -> bool:
        """Whether any scorer prior value came from a degraded fallback table."""
        sources = (
            *self.baseline_tps_source_by_role.values(),
            *self.baseline_quality_source_by_role.values(),
            *self.memory_cost_source_by_role.values(),
        )
        return any(source == PRIOR_SOURCE_DEGRADED_FALLBACK for source in sources)


def _fallback_q_scorer_priors(*, degraded_reason: str | None = None) -> QScorerPriors:
    return _materialize_q_scorer_priors(
        QScorerPriors(
            baseline_tps_by_role=dict(FALLBACK_BASELINE_TPS_BY_ROLE),
            baseline_quality_by_role=_baseline_quality_by_role(),
            memory_cost_by_role=dict(FALLBACK_MEMORY_COST_BY_ROLE),
            baseline_tps_source_by_role={
                role: PRIOR_SOURCE_DEGRADED_FALLBACK
                for role in FALLBACK_BASELINE_TPS_BY_ROLE
            },
            baseline_quality_source_by_role={
                role: PRIOR_SOURCE_DEGRADED_FALLBACK
                for role in _baseline_quality_by_role()
            },
            memory_cost_source_by_role={
                role: PRIOR_SOURCE_DEGRADED_FALLBACK
                for role in FALLBACK_MEMORY_COST_BY_ROLE
            },
            degraded_reason=degraded_reason,
        )
    )


def _valid_quality(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if 0.0 <= parsed <= 1.0 else None


def _coerce_tps(value: Any) -> float | None:
    """Return a numeric t/s value from registry scalars or lower-bound ranges."""
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            parsed = float(match.group(0))
            return parsed if parsed > 0 else None
    return None


def _load_model_descriptors(descriptor_path: Path) -> list[dict[str, Any]]:
    try:
        from src.registry.model_descriptors import load_yaml_mapping

        data = load_yaml_mapping(descriptor_path)
    except Exception as exc:
        logger.warning("Using registry q_scorer priors; descriptor load failed: %s", exc)
        return []

    models = data.get("models", [])
    if not isinstance(models, list):
        logger.warning(
            "Using registry q_scorer priors; descriptor models field is %s",
            type(models).__name__,
        )
        return []
    return [model for model in models if isinstance(model, dict)]


def _descriptor_has_role_server_conflict(descriptor: dict[str, Any]) -> bool:
    serving = descriptor.get("serving", {})
    if isinstance(serving, dict) and serving.get("numa_policy") == "unresolved_role_server_conflict":
        return True

    gaps = descriptor.get("known_gaps", [])
    if not isinstance(gaps, list):
        return False
    return any("role-server conflict" in str(gap).lower() for gap in gaps)


def _descriptor_speed_tps(descriptor: dict[str, Any]) -> float | None:
    speed = descriptor.get("speed", {})
    if not isinstance(speed, dict):
        return None

    candidates = [
        _coerce_tps(speed.get("solo_96t_tps")),
        _coerce_tps(speed.get("quarter_48t_tps")),
        _coerce_tps(speed.get("prefill_tps")),
        _coerce_tps(speed.get("generation_tps_range")),
    ]
    positive = [candidate for candidate in candidates if candidate is not None]
    return max(positive) if positive else None


def _descriptor_overall_quality(descriptor: dict[str, Any]) -> float | None:
    quality = descriptor.get("quality", {})
    if not isinstance(quality, dict):
        return None
    suite_vector = quality.get("suite_vector", {})
    if not isinstance(suite_vector, dict):
        return None
    return _valid_quality(suite_vector.get("overall"))


def stack_prior_q_scorer_priors_by_role(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> QScorerPriors:
    """Return live q_scorer priors from the generated stack-priors contract.

    Local fallback tables remain available for explicit degraded/offline mode.
    Live scoring should prefer this generated contract so model swaps, role
    retirements, HOT/WARM tier changes, and shared-server bindings are data-only.
    """
    fallback_priors = _fallback_q_scorer_priors()
    tps_by_role = dict(fallback_priors.baseline_tps_by_role)
    quality_by_role = dict(fallback_priors.baseline_quality_by_role)
    memory_by_role = dict(fallback_priors.memory_cost_by_role)
    tps_sources = dict(fallback_priors.baseline_tps_source_by_role)
    quality_sources = dict(fallback_priors.baseline_quality_source_by_role)
    memory_sources = dict(fallback_priors.memory_cost_source_by_role)

    try:
        live_role_records = _load_valid_stack_prior_role_records(stack_priors_path)
    except Exception as exc:
        logger.warning("Using fallback q_scorer priors; stack-priors load failed: %s", exc)
        return _fallback_q_scorer_priors(
            degraded_reason=f"stack-priors load failed: {exc}"
        )

    if not live_role_records:
        return _fallback_q_scorer_priors(
            degraded_reason="stack-priors artifact has no live q_scorer roles"
        )

    for role, record in live_role_records.items():
        if role in _NON_SCORING_HOST_ALIASES:
            # A host-only alias (e.g. "worker"): live in the serving topology,
            # but not a `Role` and not a scoring target in its own right. Its
            # measurements are already carried by its named members.
            continue
        priors = record.get("priors", {})
        if not isinstance(priors, dict):
            continue
        target_roles = (role, *STACK_PRIOR_SCORER_ROLE_ALIASES.get(role, ()))
        tps = _coerce_tps(priors.get("throughput_tps"))
        if tps is not None:
            for target_role in target_roles:
                tps_by_role[target_role] = tps
                tps_sources[target_role] = PRIOR_SOURCE_STACK_PRIORS
        # quality_for_role FIRST: the role-relevant axis, which is the whole
        # point of the per-axis work. quality_overall is the fleet-wide
        # aggregate and is the right answer only when no rankable role axis
        # exists — which quality_for_role already decides, recording its
        # reasoning in `basis`. Reading quality_overall here as well would give
        # two different quality numbers for one role depending on which loader
        # ran, which is the two-sources-of-truth defect in miniature.
        for_role = priors.get("quality_for_role")
        quality = None
        if isinstance(for_role, dict):
            quality = _valid_quality(for_role.get("value"))
        if quality is None:
            quality = _valid_quality(priors.get("quality_overall"))
        if quality is not None:
            for target_role in target_roles:
                quality_by_role[target_role] = quality
                quality_sources[target_role] = PRIOR_SOURCE_STACK_PRIORS
        memory_cost = priors.get("memory_cost")
        if isinstance(memory_cost, (int, float)) and float(memory_cost) > 0:
            for target_role in target_roles:
                memory_by_role[target_role] = float(memory_cost)
                memory_sources[target_role] = PRIOR_SOURCE_STACK_PRIORS

    return _materialize_q_scorer_priors(
        QScorerPriors(
            baseline_tps_by_role=tps_by_role,
            baseline_quality_by_role=quality_by_role,
            memory_cost_by_role=memory_by_role,
            baseline_tps_source_by_role=tps_sources,
            baseline_quality_source_by_role=quality_sources,
            memory_cost_source_by_role=memory_sources,
        )
    )


def _load_valid_stack_priors(stack_priors_path: Path) -> dict[str, Any]:
    from src.registry.stack_priors import (
        load_stack_priors_artifact,
        validate_stack_priors_contract,
    )

    data = load_stack_priors_artifact(stack_priors_path)
    if data is None:
        raise ValueError(f"stack-priors artifact unavailable or malformed: {stack_priors_path}")
    contract_errors = validate_stack_priors_contract(data)
    if contract_errors:
        raise ValueError("; ".join(contract_errors[:3]))
    return data


def _load_valid_stack_prior_role_records(
    stack_priors_path: Path,
) -> dict[str, dict[str, Any]]:
    """Return generated live-role records after validating the stack-prior contract."""
    _load_valid_stack_priors(stack_priors_path)

    from src.registry.stack_priors import live_stack_role_records

    return live_stack_role_records(stack_priors_path)


def _live_stack_q_scorer_roles(stack_priors_path: Path) -> set[str]:
    live_role_records = _load_valid_stack_prior_role_records(stack_priors_path)
    live_roles: set[str] = set()
    for role in live_role_records:
        if role in _NON_SCORING_HOST_ALIASES:
            continue
        live_roles.add(role)
        live_roles.update(STACK_PRIOR_SCORER_ROLE_ALIASES.get(role, ()))
    return live_roles


def validate_live_q_scorer_prior_sources(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> list[str]:
    """Return promotion-blocking errors for live q_scorer degraded priors.

    Degraded fallback tables remain valid for offline/replay maintenance. They
    must not silently satisfy live-stack promotion when a generated stack-prior
    artifact is present and structurally valid.
    """
    try:
        _load_valid_stack_priors(stack_priors_path)
    except Exception as exc:
        return [f"q_scorer stack-priors validation failed: {exc}"]

    priors = stack_prior_q_scorer_priors_by_role(stack_priors_path)
    errors: list[str] = []
    if priors.degraded_reason:
        errors.append(f"q_scorer priors are degraded: {priors.degraded_reason}")

    source_maps = (
        ("throughput", priors.baseline_tps_source_by_role),
        ("quality", priors.baseline_quality_source_by_role),
        ("memory_cost", priors.memory_cost_source_by_role),
    )
    # 2026-08-02: an ABSENT prior and a FABRICATED one are not the same failure.
    #
    # This loop treated both as "not stack_priors" and blocked identically, which
    # made the honest state unreachable: a role whose model has no measured quality
    # CANNOT draw one from stack_priors, so demanding it is demanding the impossible
    # — and the only way to satisfy the gate was to let a degraded table supply a
    # number, i.e. to prefer a fabricated value over an admitted unknown.
    #
    #   degraded_fallback -> a NUMBER FROM SOMEWHERE ELSE is being scored as if it
    #                        described this model. Always an error. This is how the
    #                        router came to score a Qwen3.6-27B with the 122B's 0.94.
    #   <missing>         -> the model genuinely has no measured value on this
    #                        instrument, and the consumer SKIPS the term rather than
    #                        inventing one (q_reward.py guards with `if role in ...`).
    #                        Permitted only when the compiled prior agrees it is null,
    #                        so "missing" can never mask a compile that dropped a
    #                        value it should have carried.
    priors_doc = _load_valid_stack_priors(stack_priors_path)
    role_records = (priors_doc.get("roles") or {}) if isinstance(priors_doc, dict) else {}

    def _declared_null(role: str, prior_name: str) -> bool:
        if prior_name != "quality":
            return False
        record = role_records.get(role)
        if not isinstance(record, dict):
            return False
        block = record.get("priors")
        if not isinstance(block, dict):
            return False
        return "quality_overall" in block and block.get("quality_overall") is None

    for role in sorted(_live_stack_q_scorer_roles(stack_priors_path)):
        for prior_name, source_by_role in source_maps:
            source = source_by_role.get(role, "<missing>")
            if source == PRIOR_SOURCE_STACK_PRIORS:
                continue
            if source == "<missing>" and _declared_null(role, prior_name):
                # Honest absence, corroborated by the compiled artifact.
                continue
            errors.append(
                f"live q_scorer role {role!r} uses {prior_name} source "
                f"{source}; expected {PRIOR_SOURCE_STACK_PRIORS}"
            )
    return errors


def require_live_q_scorer_stack_priors(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> None:
    """Raise when live q_scorer priors are not fully stack-prior sourced."""
    errors = validate_live_q_scorer_prior_sources(stack_priors_path)
    if errors:
        raise QScorerPriorSourceError("; ".join(errors))


def descriptor_q_scorer_priors_by_role(
    descriptor_path: Path = DEFAULT_MODEL_DESCRIPTOR_PATH,
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
    stack_priors_path: Path | None = None,
) -> QScorerPriors:
    """Return scorer priors over existing roles, skipping ambiguous descriptors.

    Generated stack priors are the live default contract. Descriptors are used
    only as a gap-fill overlay for roles already present in the live/default
    priors. Role-server conflict records are intentionally ignored, so q_scorer
    does not learn from descriptor rows that disagree with production bindings.
    """
    if stack_priors_path is None and (
        descriptor_path == DEFAULT_MODEL_DESCRIPTOR_PATH
        and registry_path == DEFAULT_MODEL_REGISTRY_PATH
    ):
        stack_priors_path = DEFAULT_STACK_PRIORS_PATH

    if stack_priors_path is not None:
        return stack_prior_q_scorer_priors_by_role(stack_priors_path)

    tps_priors = registry_baseline_tps_priors(registry_path)
    tps_by_role = {
        role: prior.value for role, prior in tps_priors.items() if prior.value is not None
    }
    quality_by_role = _baseline_quality_by_role()
    memory_by_role = registry_memory_cost_by_role(registry_path)
    # Carry the per-role registry provenance through instead of stamping every
    # role with a flat "registry". A role whose number is an UNOPTIMIZED
    # baseline_tps, or whose two performance fields hold one duplicated figure,
    # now says so in baseline_tps_source_by_role — the caller-visible channel
    # this dataclass already has. Roles the registry did not speak to keep the
    # historical PRIOR_SOURCE_REGISTRY label (see registry_baseline_tps_priors).
    tps_sources = {role: prior.source for role, prior in tps_priors.items()}
    quality_sources = {
        role: PRIOR_SOURCE_DEGRADED_FALLBACK for role in quality_by_role
    }
    memory_sources = {role: PRIOR_SOURCE_REGISTRY for role in memory_by_role}

    known_tps_roles = set(tps_by_role)
    known_quality_roles = set(quality_by_role)
    for descriptor in _load_model_descriptors(descriptor_path):
        if _descriptor_has_role_server_conflict(descriptor):
            continue

        role_bindings = descriptor.get("role_bindings", {})
        if not isinstance(role_bindings, dict):
            continue
        roles = role_bindings.get("roles", [])
        if not isinstance(roles, list):
            continue

        tps = _descriptor_speed_tps(descriptor)
        quality = _descriptor_overall_quality(descriptor)
        for role in roles:
            if not isinstance(role, str):
                continue
            if tps is not None and role in known_tps_roles:
                tps_by_role[role] = tps
                tps_sources[role] = PRIOR_SOURCE_MODEL_DESCRIPTORS
            if quality is not None and role in known_quality_roles:
                quality_by_role[role] = quality
                quality_sources[role] = PRIOR_SOURCE_MODEL_DESCRIPTORS

    return _materialize_q_scorer_priors(
        QScorerPriors(
            baseline_tps_by_role=tps_by_role,
            baseline_quality_by_role=quality_by_role,
            memory_cost_by_role=memory_by_role,
            baseline_tps_source_by_role=tps_sources,
            baseline_quality_source_by_role=quality_sources,
            memory_cost_source_by_role=memory_sources,
        )
    )


def _residency_memory_cost(residency: Any) -> float | None:
    if not isinstance(residency, str):
        return None
    normalized = residency.strip().lower()
    if normalized == "hot":
        return 1.0
    if normalized == "warm":
        return 2.0
    if normalized == "cold":
        return 3.0
    return None


@dataclass(frozen=True)
class RegistryTpsPrior:
    """One role's t/s prior WITH the identity of the measurement it came from.

    Modelled on ``src/registry/model_descriptors._speed()``: optimized and
    baseline are kept as separate None-or-float fields and are never folded
    into each other, so a caller can always recover which measurement (if
    either) the scalar ``value`` actually is.

    ``field_path`` is the exact registry location the scalar was read from.
    ``None`` means the value survived from ``FALLBACK_BASELINE_TPS_BY_ROLE``
    and the registry said nothing about this role.
    """

    role: str
    value: float | None
    source: str
    field_path: str | None = None
    optimized_tps: float | None = None
    baseline_tps: float | None = None

    @property
    def substituted(self) -> bool:
        """True when an unoptimized baseline_tps is standing in for optimized_tps."""
        return self.source == PRIOR_SOURCE_REGISTRY_BASELINE_TPS_SUBSTITUTED

    @property
    def unseparated(self) -> bool:
        """True when optimized_tps and baseline_tps are one number in two fields."""
        return self.source == PRIOR_SOURCE_REGISTRY_TPS_UNSEPARATED

    @property
    def carries_optimization_evidence(self) -> bool:
        """True only when the scalar is an optimized measurement distinct from baseline."""
        return self.source == PRIOR_SOURCE_REGISTRY_OPTIMIZED_TPS


_warned_tps_provenance_roles: set[str] = set()


def _warn_tps_provenance(prior: RegistryTpsPrior) -> None:
    """Warn once per role whose t/s prior is not a separate optimized measurement.

    Every current consumer of ``registry_baseline_tps_by_role`` takes the bare
    float and divides by it; none of them can act on a new struct field without
    being changed. A log is therefore the only channel that reaches TODAY's
    callers, and NIB2-57a asks for exactly that: loud, not silent.
    """
    if prior.role in _warned_tps_provenance_roles:
        return
    if prior.substituted:
        _warned_tps_provenance_roles.add(prior.role)
        logger.warning(
            "q_scorer t/s prior for role %r is an UNOPTIMIZED baseline_tps (%.4g) "
            "standing in for a missing optimized_tps. Cost penalties computed from "
            "it are under-applied. Source: %s",
            prior.role,
            prior.value if prior.value is not None else float("nan"),
            prior.field_path,
        )
    elif prior.unseparated:
        _warned_tps_provenance_roles.add(prior.role)
        logger.warning(
            "q_scorer t/s prior for role %r has optimized_tps == baseline_tps "
            "(%.4g): one measurement written into both registry fields, so this "
            "prior carries NO evidence that the role was measured under "
            "optimization. Source: %s",
            prior.role,
            prior.value if prior.value is not None else float("nan"),
            prior.field_path,
        )


def _performance_tps(
    role: str,
    role_record: dict[str, Any],
    *,
    registry_role: str | None = None,
) -> RegistryTpsPrior | None:
    """Return the role's ``performance`` t/s prior, or None when it has none.

    ``role`` is the q_scorer role the prior will be filed under; ``registry_role``
    is the registry key the ``performance`` block was read from (they differ when
    ``ROLE_PERFORMANCE_TPS_FALLBACKS`` maps one onto the other).

    NEVER returns a bare float. ``optimized_tps or baseline_tps`` is the defect
    this replaces: the two are different measurements and the ``or`` handed the
    caller one of them with no way to tell which. The numeric preference is
    unchanged (optimized wins when present) — only the provenance is new.
    """
    source_role = registry_role or role
    perf = role_record.get("performance", {})
    if not isinstance(perf, dict):
        return None

    optimized = _coerce_tps(perf.get("optimized_tps"))
    baseline = _coerce_tps(perf.get("baseline_tps"))
    if optimized is None and baseline is None:
        return None

    if optimized is not None and baseline is not None and math.isclose(
        optimized, baseline, rel_tol=1e-9, abs_tol=0.0
    ):
        source = PRIOR_SOURCE_REGISTRY_TPS_UNSEPARATED
        value = optimized
        field_path = f"roles.{source_role}.performance.optimized_tps==baseline_tps"
    elif optimized is not None:
        source = PRIOR_SOURCE_REGISTRY_OPTIMIZED_TPS
        value = optimized
        field_path = f"roles.{source_role}.performance.optimized_tps"
    else:
        source = PRIOR_SOURCE_REGISTRY_BASELINE_TPS_SUBSTITUTED
        value = baseline
        field_path = f"roles.{source_role}.performance.baseline_tps"

    prior = RegistryTpsPrior(
        role=role,
        value=value,
        source=source,
        field_path=field_path,
        optimized_tps=optimized,
        baseline_tps=baseline,
    )
    _warn_tps_provenance(prior)
    return prior


def _registry_role_records(registry: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    roles = registry.get("roles", {})
    if not isinstance(roles, dict):
        return None
    return {
        role: record
        for role, record in roles.items()
        if isinstance(role, str) and isinstance(record, dict)
    }


def registry_baseline_tps_priors(
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> Dict[str, RegistryTpsPrior]:
    """Load QScorer t/s baselines from the lean registry WITH their provenance.

    This is the single source of truth for the registry t/s branch;
    ``registry_baseline_tps_by_role`` is a bare-float projection of it, so the
    number and its provenance can never drift apart.

    The scorer keeps a fallback table so tests, replay tools, and degraded
    maintenance scripts can run without the registry. Live roles prefer the
    deployment `server_mode.*.throughput` values; vision-only roles read
    `roles.*.performance` because they are not normal text server-mode entries —
    and that block holds TWO different measurements (`optimized_tps` and
    `baseline_tps`) which this loader keeps separate and labels per role. See
    the PRIOR_SOURCE_REGISTRY_* block above.
    """
    priors: Dict[str, RegistryTpsPrior] = {
        role: RegistryTpsPrior(role=role, value=value, source=PRIOR_SOURCE_REGISTRY)
        for role, value in FALLBACK_BASELINE_TPS_BY_ROLE.items()
    }

    def _finish() -> Dict[str, RegistryTpsPrior]:
        materialized = _materialize_q_scorer_role_aliases(priors)
        # An alias shares the canonical role's measurement but must not claim to
        # BE that role, or a provenance report would name the wrong role.
        return {
            role: (prior if prior.role == role else replace(prior, role=role))
            for role, prior in materialized.items()
        }

    try:
        from src.registry.model_descriptors import load_yaml_mapping

        data = load_yaml_mapping(registry_path)
    except Exception as exc:
        logger.warning("Using fallback q_scorer TPS baselines; registry load failed: %s", exc)
        return _finish()

    server_mode = data.get("server_mode", {})
    if isinstance(server_mode, dict):
        for server_key, target_roles in SERVER_MODE_TPS_ROLE_ALIASES.items():
            record = server_mode.get(server_key, {})
            if not isinstance(record, dict):
                continue
            tps = _coerce_tps(record.get("throughput"))
            if tps is None:
                continue
            for role in target_roles:
                priors[role] = RegistryTpsPrior(
                    role=role,
                    value=tps,
                    source=PRIOR_SOURCE_REGISTRY,
                    field_path=f"server_mode.{server_key}.throughput",
                )

    roles = _registry_role_records(data)
    if roles is not None:
        for target_role, registry_role in ROLE_PERFORMANCE_TPS_FALLBACKS.items():
            record = roles.get(registry_role)
            if record is None:
                continue
            prior = _performance_tps(target_role, record, registry_role=registry_role)
            if prior is not None:
                priors[target_role] = prior

    return _finish()


def registry_baseline_tps_by_role(
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> Dict[str, float]:
    """Bare-float projection of :func:`registry_baseline_tps_priors`.

    Numerically identical to the pre-2026-08-12 loader. Callers that need to
    know WHICH measurement a number is — optimized, an unoptimized baseline
    substituted for a missing optimized, or one figure duplicated into both
    fields — must call ``registry_baseline_tps_priors`` instead; this projection
    cannot express the difference, which is precisely why the substitution was
    invisible here for so long.
    """
    return {
        role: prior.value
        for role, prior in registry_baseline_tps_priors(registry_path).items()
        if prior.value is not None
    }


def registry_memory_cost_by_role(
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> Dict[str, float]:
    """Load per-role memory-residency costs from the live registry.

    Values are tier costs, not raw model-size costs. HOT roles normalize to
    1.0, so they do not trigger the warm-tier reward penalty. Retired roles
    absent from live server/role records are intentionally not synthesized.
    """
    costs = dict(FALLBACK_MEMORY_COST_BY_ROLE)
    try:
        from src.registry.model_descriptors import load_yaml_mapping

        data = load_yaml_mapping(registry_path)
    except Exception as exc:
        logger.warning("Using fallback q_scorer memory costs; registry load failed: %s", exc)
        return _materialize_q_scorer_role_aliases(costs)

    roles = _registry_role_records(data)
    if roles is not None:
        known_roles = set(costs)
        for role, record in roles.items():
            if role not in known_roles:
                continue
            memory = record.get("memory", {})
            if not isinstance(memory, dict):
                continue
            cost = _residency_memory_cost(memory.get("residency"))
            if cost is not None:
                costs[role] = cost

    server_mode = data.get("server_mode", {})
    if isinstance(server_mode, dict):
        for server_key, target_roles in REGISTRY_MEMORY_ROLE_ALIASES.items():
            record = server_mode.get(server_key, {})
            if not isinstance(record, dict):
                continue
            cost = _residency_memory_cost(record.get("tier"))
            if cost is None:
                continue
            for role in target_roles:
                costs[role] = cost

    return _materialize_q_scorer_role_aliases(costs)


@dataclass
class ScoringConfig:
    """Configuration for Q-scoring."""

    # Learning rate for Q-value updates
    learning_rate: float = 0.1

    # Reward values
    success_reward: float = 1.0
    failure_reward: float = -0.5
    partial_reward: float = 0.3

    # Temporal decay: Q-values decay toward neutral (0.5) over time.
    # decay_rate ^ days_elapsed is applied before each TD update.
    temporal_decay_rate: float = 0.99

    # Claude-as-Judge settings (optional)
    use_claude_judge: bool = False
    judge_model_path: Optional[Path] = None
    judge_binary: Optional[Path] = None

    # RTG-09 (2026-09-24): wall-clock task duration as a new, GRADED speed
    # dimension -- see Dimension 0 in q_reward.py and `baseline_duration_by_role`
    # below. DEFAULT-OFF (0.0): flipping this on changes compute_reward's
    # output distribution, which is a routing_reward instrument-era boundary
    # (orchestration/instrument_eras.yaml, human-amendment-only) because
    # rewards feed Q-updates continuously. The behaviour flip and the era
    # boundary must land together, by operator ratification --
    # scripts/operator/ratify_rtg09_duration_reward_20260924.sh (in the
    # epyc-root worktree) flips this to 0.20 and appends the era row in the
    # same apply. Until that ratification runs, this field being 0.0 makes
    # compute_reward byte-identical to pre-RTG-09 for every input (see
    # test_duration_default_off_reward_is_byte_identical_to_pre_change in
    # tests/unit/test_q_reward_duration.py). Tokens/sec is gameable through
    # tool calls and blind to orchestration/tool overhead -- the DAR handoff
    # measured median wall/model-compute overhead 1.60x, p90 9.09x across
    # 19,433 tasks -- so once ratified this DOES become the primary speed
    # axis and cost_penalty_lambda below is demoted at the same moment. The
    # ratification script is the only intended writer of a non-zero default
    # here; do not hand-edit this default outside that flow.
    cost_lambda_duration: float = 0.20

    # Cost-aware reward (xRouter-style correctness-gated cost penalty).
    # reward_final = quality_reward - lambda * max(0, cost_ratio - 1.0)
    # where cost_ratio = actual_elapsed / expected_elapsed.
    # Only applied when answer is correct (incorrect = 0.0, no cost term).
    # RTG-09: paired with cost_lambda_duration above -- demoted to 0.05 only
    # at the same ratified instrument-era boundary that flips duration on.
    # Unchanged (0.15) until then.
    cost_penalty_lambda: float = 0.05

    # Per-role optimized tokens/second from generated stack priors at config
    # construction time, with fallback tables for degraded/offline scripts. Used
    # to normalize cost:
    # expected_elapsed = tokens_generated / baseline_tps.
    baseline_tps_by_role: Dict[str, float] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().baseline_tps_by_role
    )
    baseline_tps_source_by_role: Dict[str, str] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().baseline_tps_source_by_role
    )

    # RTG-09: per-role wall-clock task-duration baselines {"p50_s", "p90_s"},
    # measured from retained progress logs (see `load_duration_baselines_by_role`
    # docstring). A role absent here means the duration dimension is SKIPPED for
    # it in q_reward.py, never defaulted -- the same "absence, not fabrication"
    # contract baseline_quality_by_role already follows.
    baseline_duration_by_role: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: load_duration_baselines_by_role()[0]
    )
    duration_baseline_degraded_reason: Optional[str] = field(
        default_factory=lambda: load_duration_baselines_by_role()[1]
    )

    # Per-role quality baselines from generated stack priors, with legacy
    # benchmark fallbacks for roles whose structured quality prior is still a gap.
    # Used for quality-gap penalty: penalize using expensive model when cheap suffices.
    baseline_quality_by_role: Dict[str, float] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().baseline_quality_by_role
    )
    baseline_quality_source_by_role: Dict[str, str] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().baseline_quality_source_by_role
    )

    # Per-role memory residency cost. HOT roles normalize to 1.0 and incur no
    # warm-tier penalty; non-HOT roles load from the registry when present.
    memory_cost_by_role: Dict[str, float] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().memory_cost_by_role
    )
    memory_cost_source_by_role: Dict[str, str] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().memory_cost_source_by_role
    )
    prior_degraded_reason: Optional[str] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().degraded_reason
    )

    # Multi-dimensional cost weights (tunable).
    # cost_lambda_latency is the existing cost_penalty_lambda.
    cost_lambda_quality_gap: float = 0.10  # Penalize using higher-quality model than needed
    cost_lambda_memory: float = 0.05       # Penalize non-HOT residency when HOT sufficient

    # Delegation/teacher attribution shaping.
    delegation_misattribution_penalty: float = 0.10
    specialist_credit_bonus: float = 0.05
    teacher_regret_penalty: float = 0.20
    teacher_speedup_bonus: float = 0.05

    # Scoring frequency
    min_score_interval_seconds: int = 300  # 5 minutes

    # Batch size for processing
    batch_size: int = 50


class QScorer:
    """
    Async Q-value scoring agent.

    Workflow:
    1. Read progress logs for completed tasks
    2. For each task:
       a. Find associated memory entries
       b. Compute reward from outcome
       c. Update Q-values
    3. Log scoring events
    """

    def __init__(
        self,
        store: EpisodicStore,
        embedder: TaskEmbedder,
        logger: ProgressLogger,
        reader: ProgressReader,
        config: Optional[ScoringConfig] = None,
        staged_scorer: Optional[StagedQScorer] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.logger = logger
        self.reader = reader
        self.config = config or ScoringConfig()
        self.staged_scorer = staged_scorer
        self._last_score_time: Optional[datetime] = None

    def score_pending_tasks(
        self,
        should_stop: Optional[Callable[[], bool]] = None,
        skip_task: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Score all pending tasks from progress logs.

        Args:
            should_stop: Optional predicate checked before each task; True ends the batch
                early (INF-78 OAB-3: a quiescent_after request took the post-reply hold).
            skip_task: Optional predicate; True drops a task id from the backlog before the
                batch is sliced (tasks whose scoring a quiescent request suppressed).

        Returns:
            Summary of scoring results
        """
        # Check minimum interval
        now = datetime.now(timezone.utc)
        if self._last_score_time:
            elapsed = (now - self._last_score_time).total_seconds()
            if elapsed < self.config.min_score_interval_seconds:
                return {
                    "skipped": True,
                    "reason": f"Too soon ({elapsed:.0f}s < {self.config.min_score_interval_seconds}s)",
                }

        # Find unscored tasks
        unscored_task_ids = self.reader.get_unscored_tasks()
        if skip_task is not None:
            unscored_task_ids = [t for t in unscored_task_ids if not skip_task(t)]

        if not unscored_task_ids:
            return {"tasks_processed": 0, "message": "No pending tasks to score"}

        # Process in batches
        results = {
            "tasks_processed": 0,
            "memories_updated": 0,
            "memories_created": 0,
            "errors": [],
        }

        for task_id in unscored_task_ids[: self.config.batch_size]:
            if should_stop is not None and should_stop():
                results["stopped_early"] = True
                break
            try:
                task_result = self._score_task(task_id)
                results["tasks_processed"] += 1
                results["memories_updated"] += task_result.get("memories_updated", 0)
                results["memories_created"] += task_result.get("memories_created", 0)
            except Exception as e:
                results["errors"].append({"task_id": task_id, "error": str(e)})

        self._last_score_time = now
        self.logger.flush()

        return results

    @staticmethod
    def _non_quality_disposition(task_outcome: ProgressEntry) -> Optional[str]:
        """Return the stamped disposition when this outcome carries no quality.

        Reads the structural `failure_disposition` the serving pipeline puts on
        a TASK_FAILED entry (`src.api.services.memrl.failure_disposition_meta`,
        classified by `src.autopilot_core.measurement_guards`). Returns None for
        a success, an unstamped legacy entry, or a GENUINE task failure — all of
        which keep their existing reward behaviour.
        """
        data = task_outcome.data if task_outcome and task_outcome.data else None
        if not isinstance(data, dict):
            return None
        disposition = str(data.get(FAILURE_DISPOSITION_KEY) or "").strip().lower()
        return disposition if disposition in NON_QUALITY_DISPOSITIONS else None

    def _score_task(self, task_id: str) -> Dict[str, Any]:
        """
        Score a single task.

        Args:
            task_id: Task ID to score

        Returns:
            Scoring results for this task
        """
        # Get task trajectory
        trajectory = self.reader.get_task_trajectory(task_id)

        if not trajectory:
            return {"error": "No trajectory found"}

        # Extract key events
        task_started = None
        routing_decision = None
        task_outcome = None
        gate_results = []
        gates_inconclusive = 0
        escalations = []
        plan_reviews = []

        for entry in trajectory:
            if entry.event_type == EventType.TASK_STARTED:
                task_started = entry
            elif entry.event_type == EventType.ROUTING_DECISION:
                routing_decision = entry
            elif entry.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
                task_outcome = entry
            elif entry.event_type in (EventType.GATE_PASSED, EventType.GATE_FAILED):
                gate_results.append(entry)
            elif entry.event_type == EventType.GATE_INCONCLUSIVE:
                # CJ-8, DELIBERATE OMISSION — not an oversight. A gate that never
                # decided is counted, never scored: it enters neither the reward
                # penalty (`q_reward` charges -0.1 per GATE_FAILED) nor the
                # boolean `passed` re-derivation, both of which would record a
                # harness defect as evidence about the model.
                gates_inconclusive += 1
            elif entry.event_type == EventType.ESCALATION_TRIGGERED:
                escalations.append(entry)
            elif entry.event_type == EventType.PLAN_REVIEWED:
                plan_reviews.append(entry)

        if not task_outcome:
            return {"error": "Task not completed yet"}

        # An INFRA failure is the ABSENCE of a measurement, never a bad one, so
        # it must not reach the reward writer at all. The serving pipeline
        # stamps the structural disposition onto the TASK_FAILED entry
        # (`src.api.services.memrl.failure_disposition_meta`, classified by
        # `src.autopilot_core.measurement_guards` — the one classifier the
        # seeding path already uses). Without this gate a backend blip scored
        # `failure_reward = -0.5` → `initial_q = 0.25`, below the
        # `min_q_value = 0.3` retrieval floor, evicting the memory from the
        # learned router permanently.
        #
        # The gate is HERE, at the writer, and not only at the API call site,
        # because `score_pending_tasks()` re-derives unscored tasks straight
        # from the progress journal — a skip that lived only in the request
        # handler would be undone by the next background sweep.
        #
        # A genuine task failure carries no such stamp and still writes its
        # negative reward: learning that a role gets things wrong is the signal
        # this subsystem exists to collect.
        skipped = self._non_quality_disposition(task_outcome)
        if skipped:
            return {
                "memories_updated": 0,
                "memories_created": 0,
                "skipped": skipped,
                "reward": None,
            }

        # RTG-09: wall-clock task duration for the reward's primary speed axis,
        # derived exactly as the DAR handoff specifies -- task_completed.timestamp
        # minus task_started.timestamp, matched by task_id (the two events this
        # trajectory already carries). None when TASK_STARTED was never recorded
        # for this task_id; compute_reward treats that as an explicit missing
        # case (warns once, skips only the duration dimension) rather than
        # silently defaulting.
        task_duration_s: Optional[float] = None
        if task_started and task_started.timestamp and task_outcome.timestamp:
            task_duration_s = (
                task_outcome.timestamp - task_started.timestamp
            ).total_seconds()
            if task_duration_s < 0:
                # Clock skew / out-of-order replay -- not a real duration.
                task_duration_s = None

        # Compute reward (pass completion data as optional cost/telemetry metrics)
        reward = self._compute_reward(
            task_outcome,
            gate_results,
            escalations,
            plan_reviews,
            cost_metrics=(task_outcome.data if task_outcome and task_outcome.data else None),
            task_duration_s=task_duration_s,
        )
        # Delegation credit assignment to avoid over-crediting envelope roles.
        reward = self._apply_delegation_credit(reward, routing_decision, task_outcome)

        # Apply staged reward shaping if enabled (explore early, exploit later)
        if self.staged_scorer is not None:
            task_type = ""
            if task_started and task_started.data:
                task_type = task_started.data.get("task_type", "")
            action_str = ""
            if routing_decision and routing_decision.data:
                routing = routing_decision.data.get("routing", [])
                action_str = ",".join(routing) if isinstance(routing, list) else str(routing)
            if action_str and task_type:
                reward = self.staged_scorer.compute_staged_reward(
                    reward, action_str, task_type, self.store,
                )

        # DAR-2/DAR-3: Reward adjustment (feature-flagged).
        # SPO_PLUS_ENABLED (DAR-3) supersedes CONTRASTIVE_Q_UPDATES (DAR-2) when both on.
        ranking_adj = 0.0
        ranking_source = "none"
        if routing_decision and task_started:
            if SPO_PLUS_ENABLED:
                ranking_adj = self._compute_spo_plus_adjustment(
                    task_started, routing_decision, reward,
                    margin=SPO_PLUS_MARGIN,
                )
                ranking_source = "spo_plus"
            elif CONTRASTIVE_Q_UPDATES:
                ranking_adj = self._compute_contrastive_adjustment(
                    task_started, routing_decision, reward,
                )
                ranking_source = "contrastive"

            if abs(ranking_adj) > 0.001:
                logger.info(
                    "DAR %s: adj=%.4f reward=%.3f→%.3f task=%s",
                    ranking_source, ranking_adj, reward, reward + ranking_adj, task_id,
                )

        reward_for_update = max(-1.0, min(1.0, reward + ranking_adj))

        result = {
            "memories_updated": 0,
            "memories_created": 0,
            "reward": reward,
            "contrastive_adj": ranking_adj,
            "ranking_source": ranking_source,
            # CJ-9. Undecided gates are EXCLUDED from the reward but never
            # dropped from the record: a count you cannot name is a count you
            # cannot know is zero.
            "gates_inconclusive": gates_inconclusive,
        }

        # Update or create routing memory (uses contrastive-adjusted reward).
        # `task_outcome` rides along to carry the M-11a2b `work` payload; it is
        # already guaranteed non-None here (the early return above).
        if routing_decision:
            memory_result = self._update_routing_memory(
                task_id,
                task_started,
                routing_decision,
                reward_for_update,
                task_outcome=task_outcome,
            )
            result.update(memory_result)

        # Update escalation memories (use base reward, not contrastive-adjusted).
        # `routing_decision` rides along solely to carry this task's TR-3.2
        # `assigned_role` — see _update_escalation_memory.
        for escalation in escalations:
            esc_result = self._update_escalation_memory(
                task_id, escalation, reward, routing_decision=routing_decision,
            )
            result["memories_updated"] += esc_result.get("memories_updated", 0)
            result["memories_created"] += esc_result.get("memories_created", 0)

        return result

    def _apply_delegation_credit(
        self,
        reward: float,
        routing_decision: Optional[ProgressEntry],
        task_outcome: ProgressEntry,
    ) -> float:
        """Adjust reward for delegation lineage attribution.

        Penalize envelope over-credit (architect routed but specialist produced final answer).
        Slightly bonus direct specialist attribution when selected specialist finished task.
        """
        if routing_decision is None:
            return reward
        routing = routing_decision.data.get("routing", [])
        if isinstance(routing, str):
            routed_roles = [r.strip() for r in routing.split(",") if r.strip()]
        elif isinstance(routing, list):
            routed_roles = [str(r) for r in routing]
        else:
            routed_roles = []
        final_role = str(task_outcome.data.get("final_answer_role", "") or "")
        if not routed_roles or not final_role:
            return reward

        routed_architect = any(r.startswith("architect_") for r in routed_roles)
        final_is_architect = final_role.startswith("architect_")

        if routed_architect and not final_is_architect:
            reward -= self.config.delegation_misattribution_penalty
        elif final_role in routed_roles and not final_is_architect:
            reward += self.config.specialist_credit_bonus

        return max(-1.0, min(1.0, reward))

    def _compute_reward(
        self,
        task_outcome: ProgressEntry,
        gate_results: List[ProgressEntry],
        escalations: List[ProgressEntry],
        plan_reviews: List[ProgressEntry] | None = None,
        cost_metrics: Optional[Dict[str, Any]] = None,
        task_duration_s: Optional[float] = None,
    ) -> float:
        """Compute reward — delegates to q_reward.compute_reward (Task-G refactor)."""
        return _compute_reward_impl(
            task_outcome, gate_results, escalations, plan_reviews, cost_metrics,
            config=self.config, task_duration_s=task_duration_s,
        )

    def _compute_contrastive_adjustment(
        self,
        task_started: ProgressEntry,
        routing_decision: ProgressEntry,
        reward: float,
        margin: float = 0.05,
        max_adj: float = 0.1,
    ) -> float:
        """DAR-2: Compute contrastive ranking adjustment for Q-value update.

        Sharpens decision boundaries between alternative models:
        - Success: if selected model's Q is below alternatives, boost reward
          so the TD update pushes its Q above competitors.
        - Failure: if selected model's Q is above alternatives, penalize more
          so the TD update pushes its Q below competitors.

        The adjustment is zero when:
        - The ranking is already correct with sufficient margin
        - No alternative memories with learned Q-values exist
        - The feature flag CONTRASTIVE_Q_UPDATES is off (checked by caller)

        Bounded to [-max_adj, +max_adj] to prevent runaway drift.
        With α=0.1 and max_adj=0.1, the maximum extra Q-shift per update
        is 0.01 — negligible individually, significant cumulatively.
        """
        task_context = task_started.data if task_started and task_started.data else {}
        if not task_context:
            return 0.0

        # Generate embedding for this task
        try:
            embedding = self.embedder.embed_task_ir(task_context)
        except Exception:
            return 0.0

        # Retrieve similar routing memories
        candidates = self.store.retrieve_by_similarity(
            embedding, k=10, action_type="routing",
        )
        if not candidates:
            return 0.0

        # Identify the selected action
        routing = routing_decision.data.get("routing", [])
        selected_action = routing[0] if isinstance(routing, list) and routing else str(routing)

        # Get selected memory's current Q-value
        selected_q = 0.5
        memory_id = routing_decision.memory_id
        if memory_id:
            mem = self.store.get_by_id(memory_id)
            if mem:
                selected_q = mem.q_value

        # Collect alternative Q-values (different actions, skip unlearned defaults)
        alt_q_values = []
        for c in candidates:
            if c.action != selected_action and abs(c.q_value - 0.5) > 0.001:
                alt_q_values.append(c.q_value)

        if not alt_q_values:
            return 0.0

        if reward > 0:
            # Success: push selected Q above the best alternative
            max_alt_q = max(alt_q_values)
            gap = max_alt_q + margin - selected_q
            if gap > 0:
                return min(max_adj, margin * gap)
        else:
            # Failure: push selected Q below the worst alternative
            min_alt_q = min(alt_q_values)
            gap = selected_q + margin - min_alt_q
            if gap > 0:
                return max(-max_adj, -margin * gap)

        return 0.0

    def _compute_spo_plus_adjustment(
        self,
        task_started: ProgressEntry,
        routing_decision: ProgressEntry,
        reward: float,
        margin: float = 0.05,
        max_adj: float = 0.15,
    ) -> float:
        """DAR-3: SPO+ convex surrogate adjustment for Q-value update.

        The SPO+ loss drives the predicted cost ranking toward the true cost
        ranking. Unlike DAR-2's contrastive adjustment which only compares
        selected vs best/worst alternative, SPO+ considers all alternatives:

            L_SPO+ = sum_j max(0, 2*q_hat[j] - q_true[j]) - q_hat[i*] + q_true[i*]

        where q_hat = predicted Q-values, q_true = true Q-values (estimated
        from reward), and i* = true optimal model.

        The adjustment is bounded to [-max_adj, +max_adj] to prevent
        runaway drift. When no counterfactual data is available, returns 0.0.

        Args:
            task_started: Task context entry.
            routing_decision: Routing decision entry.
            reward: Observed reward for the selected model.
            margin: Minimum gap for adjustment activation.
            max_adj: Maximum absolute adjustment.

        Returns:
            Reward adjustment in [-max_adj, +max_adj].
        """
        task_context = task_started.data if task_started and task_started.data else {}
        if not task_context:
            return 0.0

        try:
            embedding = self.embedder.embed_task_ir(task_context)
        except Exception:
            return 0.0

        candidates = self.store.retrieve_by_similarity(
            embedding, k=10, action_type="routing",
        )
        if not candidates:
            return 0.0

        # Identify selected action and its Q-value
        routing = routing_decision.data.get("routing", [])
        selected_action = routing[0] if isinstance(routing, list) and routing else str(routing)

        selected_q = 0.5
        memory_id = routing_decision.memory_id
        if memory_id:
            mem = self.store.get_by_id(memory_id)
            if mem:
                selected_q = mem.q_value

        # Collect all alternative actions with learned Q-values
        alt_actions: dict[str, float] = {}
        for c in candidates:
            if c.action != selected_action and abs(c.q_value - 0.5) > 0.001:
                # Keep the best Q-value per distinct action
                if c.action not in alt_actions or c.q_value > alt_actions[c.action]:
                    alt_actions[c.action] = c.q_value

        if not alt_actions:
            return 0.0

        # True Q-value estimate for the selected model (from observed reward)
        # Map reward to [0,1] range like initial_q does
        true_q_selected = 0.5 + (reward * 0.5)

        # SPO+ surrogate: penalize if predicted ranking disagrees with true ranking
        # For each alternative, compute the surrogate loss term
        spo_sum = 0.0
        for alt_action, alt_q_hat in alt_actions.items():
            # We don't have the true Q for alternatives (counterfactual),
            # so we use their current Q-values as the "predicted" costs
            # and the selected model's observed reward as the ground truth.
            # SPO+ term: max(0, 2*q_hat_alt - true_q_alt_estimate)
            # Since we lack true_q_alt, we use the conservative estimate:
            # If selected succeeded, alternatives would have done no better than their Q
            # If selected failed, alternatives might have done better
            spo_term = max(0.0, 2.0 * alt_q_hat - alt_q_hat)  # = alt_q_hat when positive
            spo_sum += spo_term

        # Selected model's contribution: -q_hat_selected + true_q_selected
        spo_loss = spo_sum - selected_q + true_q_selected

        # Convert loss to adjustment: positive loss → boost selected, negative → penalize
        if abs(spo_loss) < margin:
            return 0.0

        adjustment = margin * spo_loss
        return max(-max_adj, min(max_adj, adjustment))

    def _update_routing_memory(
        self,
        task_id: str,
        task_started: Optional[ProgressEntry],
        routing_decision: ProgressEntry,
        reward: float,
        task_outcome: Optional[ProgressEntry] = None,
    ) -> Dict[str, Any]:
        """Update or create routing memory.

        `task_outcome` is the TASK_COMPLETED/TASK_FAILED entry. It rides along
        solely to carry this task's `work` payload (M-11a2b) — answer, tool
        calls, REPL steps, reasoning — which the pipeline puts on that entry via
        `chat_pipeline.telemetry.work_completion_meta`. Optional and defaulted so
        a trajectory without it (or an older caller) writes exactly the
        objective-and-outcome row it wrote before.
        """
        result = {"memories_updated": 0, "memories_created": 0}

        # Check if memory already exists
        memory_id = routing_decision.memory_id

        if memory_id:
            # Update existing memory
            memory = self.store.get_by_id(memory_id)
            if memory:
                old_q = memory.q_value
                new_q = self.store.update_q_value(
                    memory_id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
                )
                self._merge_update_metadata(memory_id, routing_decision, task_outcome)
                self.logger.log_memory_update(memory_id, old_q, new_q, reward, task_id)
                result["memories_updated"] = 1
        else:
            # No pre-linked memory_id. Legacy behavior appended a fresh row per
            # observation (the append-only defect). When ORCHESTRATOR_Q_TD_WRITE
            # is set, first find-or-update the existing (objective, action) row.
            if task_started and task_started.data:
                task_context = {
                    "task_type": task_started.data.get("task_type"),
                    "objective": task_started.data.get("objective"),
                    "priority": task_started.data.get("priority"),
                }

                # Generate embedding for task context
                embedding = self.embedder.embed_task_ir(task_context)

                # Store new memory
                routing = routing_decision.data.get("routing", [])
                action = ",".join(routing) if isinstance(routing, list) else str(routing)

                if Q_TD_WRITE:
                    existing_id = self._find_existing_memory(
                        embedding, action, task_context.get("objective"),
                        action_type="routing",
                    )
                    if existing_id is not None:
                        memory = self.store.get_by_id(existing_id)
                        if memory is not None:
                            old_q = memory.q_value
                            new_q = self.store.update_q_value(
                                existing_id, reward, self.config.learning_rate,
                                temporal_decay_rate=self.config.temporal_decay_rate,
                            )
                            self._merge_update_metadata(
                                existing_id, routing_decision, task_outcome,
                            )
                            self.logger.log_memory_update(
                                existing_id, old_q, new_q, reward, task_id,
                            )
                            result["memories_updated"] = 1
                            return result

                # First observation of this (objective, action) — or flag off:
                # append a new row exactly as the legacy path always did.
                # Initial Q-value based on first observation
                initial_q = 0.5 + (reward * 0.5)  # Map reward to [0, 1]

                # One record contract for every write site (memory_record.py):
                # full untruncated objective + the work, telemetry excluded from
                # the embedding text.
                #
                # M-11a2b: the work fields were declared by the contract on
                # 2026-07-27 but nothing ever passed them, so `work` was absent
                # from all 59,337 rows and distillation could only ever learn
                # WHICH route succeeded, never HOW the task was solved. The
                # payload is redacted and size-bounded inside
                # build_memory_record, so this site cannot widen the policy.
                work = extract_work(task_outcome.data if task_outcome else None)
                record = build_memory_record(
                    objective=task_context.get("objective"),
                    task_type=task_context.get("task_type"),
                    priority=task_context.get("priority"),
                    source="progress_log",
                    answer=work.get("answer"),
                    tool_calls=work.get("tool_calls"),
                    repl_steps=work.get("repl_steps"),
                    reasoning=work.get("reasoning"),
                )
                memory_id = self.store.store(
                    embedding=embedding,
                    action=action,
                    action_type="routing",
                    model_id=_model_id_for_action(action),
                    # TR-3.2 shadow role, carried on this very entry by
                    # log_task_started's routing_meta merge. Reading it here is
                    # the last hop of classifier -> progress JSONL -> episodic
                    # store; without it the column was NULL on every row.
                    assigned_role=_assigned_role_from_entry(routing_decision),
                    context=record.to_context(),
                    outcome=outcome_from_reward(reward),
                    initial_q=initial_q,
                )

                self.logger.log(
                    ProgressEntry(
                        event_type=EventType.MEMORY_STORED,
                        task_id=task_id,
                        memory_id=memory_id,
                        data={"action_type": "routing", "initial_q": initial_q},
                    )
                )
                result["memories_created"] = 1

        return result

    def _merge_update_metadata(
        self,
        memory_id: str,
        routing_decision: Optional[ProgressEntry],
        task_outcome: Optional[ProgressEntry],
    ) -> bool:
        """Carry `assigned_role` and `work` onto a row taken by the UPDATE path.

        `store()` is the only site that ever wrote these two fields, and every
        update branch (pre-linked `memory_id`, and the ORCHESTRATOR_Q_TD_WRITE
        find-or-update branch) returns before reaching it. Since the update
        branch fires ~11x more often than create, both fields were effectively
        never populated: 0 / 59,337 rows carried `assigned_role`, 0 carried a
        top-level `work` key. The values were available on the very entries
        already passed in — they were computed and then dropped.

        The payload is built ONCE here via `build_work_payload` (the single
        producer) and merged by the store without re-sanitizing, so the policy
        runs exactly once and the row's size/elision provenance stays truthful.

        Observational: a telemetry backfill must never be the reason a completed
        task fails to be scored, so every failure is swallowed and reported as
        False.
        """
        try:
            role = _assigned_role_from_entry(routing_decision)
            work_in = extract_work(task_outcome.data if task_outcome else None)
            work = build_work_payload(
                answer=work_in.get("answer"),
                tool_calls=work_in.get("tool_calls"),
                repl_steps=work_in.get("repl_steps"),
                reasoning=work_in.get("reasoning"),
            )
            merge = getattr(self.store, "merge_row_metadata", None)
            if merge is None:
                return False
            return bool(merge(memory_id, assigned_role=role, work=work))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "assigned_role/work merge skipped for memory %s: %s", memory_id, exc,
            )
            return False

    def _find_existing_memory(
        self,
        embedding: np.ndarray,
        action: str,
        objective: Optional[str],
        action_type: str = "routing",
    ) -> Optional[str]:
        """Return the id of an existing memory of `action_type` for this exact
        (objective, action), or None.

        The find half of the ORCHESTRATOR_Q_TD_WRITE find-or-update write path.
        Uses the FAISS similarity index (fast, O(log n)) to fetch same-action
        candidates, then requires an EXACT objective+action string match so
        distinct objectives are never merged. This converges with the offline
        consolidation migration, which groups by the same (objective, action)
        key — so live TD writes and the migrated store agree on row identity.

        Shared by the routing and escalation paths. Both store their identity
        key the same way (`context.objective` + `action`) via
        build_memory_record, so one exact-match rule is correct for both; only
        the `action_type` partition differs, and it is never crossed because it
        is pushed down into the similarity query.
        """
        if objective is None:
            return None
        try:
            candidates = self.store.retrieve_by_similarity(
                embedding, k=Q_TD_MATCH_K, action_type=action_type,
            )
        except Exception:
            return None
        for c in candidates:
            if c.action == action and (c.context or {}).get("objective") == objective:
                return c.id
        return None

    def _update_escalation_memory(
        self,
        task_id: str,
        escalation: ProgressEntry,
        reward: float,
        routing_decision: Optional[ProgressEntry] = None,
    ) -> Dict[str, Any]:
        """Update or create escalation memory.

        `routing_decision` is the SAME task's ROUTING_DECISION entry, passed in
        only to carry its TR-3.2 `assigned_role`. `log_escalation` does not
        record the tri-role, but the Trinity axis is a property of the REQUEST
        (what kind of work was asked for), not of which model ended up serving
        it — an escalation is a re-route of the same request, so the task's own
        classified role is the correct value rather than an invented one.
        Default None keeps the column NULL when the caller has no routing entry.
        """
        result = {"memories_updated": 0, "memories_created": 0}

        memory_id = escalation.memory_id

        if memory_id:
            # Update existing memory
            memory = self.store.get_by_id(memory_id)
            if memory:
                old_q = memory.q_value
                new_q = self.store.update_q_value(
                    memory_id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
                )
                # Escalation rows carry no `work` payload; the tri-role is the
                # only field store() wrote that the update path dropped.
                self._merge_update_metadata(memory_id, routing_decision, None)
                self.logger.log_memory_update(memory_id, old_q, new_q, reward, task_id)
                result["memories_updated"] = 1
        else:
            # Create new escalation memory
            failure_context = {
                "from_tier": escalation.data.get("from_tier"),
                "to_tier": escalation.data.get("to_tier"),
                "reason": escalation.data.get("reason"),
            }

            embedding = self.embedder.embed_failure_context(failure_context)
            action = f"escalate:{escalation.data.get('from_tier')}->{escalation.data.get('to_tier')}"

            # Same append-only defect the routing path had: the sole production
            # emitter never populates memory_id, so this branch blind-appended a
            # fresh row per escalation (4,382 such rows at the 2026-07-22 audit).
            # Under ORCHESTRATOR_Q_TD_WRITE, find-or-update in place first.
            # The identity key is (reason, escalate:from->to) — `reason` is what
            # build_memory_record stores as `objective` below, so the finder's
            # exact-match rule keys on the same field it will later be stored under.
            if Q_TD_WRITE:
                existing_id = self._find_existing_memory(
                    embedding, action, escalation.data.get("reason"),
                    action_type="escalation",
                )
                if existing_id is not None:
                    memory = self.store.get_by_id(existing_id)
                    if memory is not None:
                        old_q = memory.q_value
                        new_q = self.store.update_q_value(
                            existing_id, reward, self.config.learning_rate,
                            temporal_decay_rate=self.config.temporal_decay_rate,
                        )
                        self._merge_update_metadata(existing_id, routing_decision, None)
                        self.logger.log_memory_update(
                            existing_id, old_q, new_q, reward, task_id,
                        )
                        result["memories_updated"] = 1
                        return result

            # First observation of this (reason, action) — or flag off: append a
            # new row exactly as the legacy path always did.
            initial_q = 0.5 + (reward * 0.5)

            # Escalation memories: the "objective" is the failure reason, and the
            # tier transition is structural metadata rather than task text, so it
            # rides in `extra` and stays out of the embedding.
            record = build_memory_record(
                objective=escalation.data.get("reason"),
                task_type="escalation",
                source="progress_log",
                extra={
                    "from_tier": escalation.data.get("from_tier"),
                    "to_tier": escalation.data.get("to_tier"),
                },
            )
            memory_id = self.store.store(
                embedding=embedding,
                action=action,
                action_type="escalation",
                model_id=_model_id_for_action(action),
                assigned_role=_assigned_role_from_entry(routing_decision),
                context=record.to_context(),
                outcome=outcome_from_reward(reward),
                initial_q=initial_q,
            )

            self.logger.log(
                ProgressEntry(
                    event_type=EventType.MEMORY_STORED,
                    task_id=task_id,
                    memory_id=memory_id,
                    data={"action_type": "escalation", "initial_q": initial_q},
                )
            )
            result["memories_created"] = 1

        return result

    def score_external_result(
        self,
        task_description: str,
        action: str,
        reward: float,
        context: Dict[str, Any] | None = None,
        embedding: List[float] | None = None,
        action_type: str = "routing",
    ) -> Dict[str, Any]:
        """Score an externally-evaluated result.

        Accepts pre-computed rewards from external scoring (e.g., the MemRL
        learning loop or debug scorer). Bypasses progress log reader and
        directly creates/updates episodic memory.

        Args:
            task_description: Description of the task.
            action: The action taken (e.g., "frontdoor:direct").
            reward: Pre-computed reward (-1.0 to 1.0).
            context: Additional context to store with the memory.
            embedding: Precomputed embedding for task_description (avoids re-embedding).
            action_type: Episodic namespace for the action. Control-plane
                actions such as plan review must not enter the routing index.

        Returns:
            Dict with memories_created and memories_updated counts.
        """
        result = {"memories_updated": 0, "memories_created": 0}
        context = context or {}
        task_type = context.get("task_type") or "chat"

        # Clamp reward to valid range
        reward = max(-1.0, min(1.0, reward))

        # Use precomputed embedding or compute it
        if embedding is not None:
            emb_array = np.array(embedding, dtype=np.float32)
        else:
            task_ir = {
                "task_type": task_type,
                # Untruncated: the 200-char cap destroyed text at write time.
                # memory_record.embedding_text() bounds the EMBEDDER input only.
                "objective": task_description,
            }
            emb_array = self.embedder.embed_task_ir(task_ir)

        # A hash pseudo-vector can only establish text identity, never semantic
        # similarity. It must not update an existing memory selected by FAISS.
        from .embedder import is_degenerate_embedding, is_hash_fallback_embedding

        embedding_text = build_memory_record(
            objective=task_description,
            task_type=task_type,
        ).embedding_text()
        if (
            is_degenerate_embedding(emb_array) is not None
            or is_hash_fallback_embedding(embedding_text, emb_array)
        ):
            logger.warning("Refusing external Q-score with a fallback or degenerate embedding")
            return result

        action_type = str(action_type or "").strip()
        if not action_type:
            raise ValueError("action_type must be a non-empty string")

        # Search for existing similar memory with same action
        # Note: retrieve_by_similarity returns memories sorted by similarity
        similar = self.store.retrieve_by_similarity(
            query_embedding=emb_array,
            k=5,
            action_type=action_type,
        )
        # Filter to high-similarity matches (similarity_score >= 0.85)
        similar = [m for m in similar if m.similarity_score >= 0.85]

        # A high cosine score is only a candidate lookup. Updating requires the
        # same normalized identity so unrelated memories cannot absorb a reward.
        def normalized(value: Any) -> str:
            return " ".join(str(value or "").split())

        expected_identity = (
            normalized(task_description),
            normalized(task_type),
            normalized(action),
        )
        updated = False
        for mem in similar:
            memory_context = mem.context if isinstance(mem.context, dict) else {}
            memory_identity = (
                normalized(memory_context.get("objective")),
                normalized(memory_context.get("task_type")),
                normalized(mem.action),
            )
            if memory_identity == expected_identity:
                old_q = mem.q_value
                new_q = self.store.update_q_value(
                    mem.id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
                )
                self.logger.log_memory_update(
                    mem.id, old_q, new_q, reward, "external"
                )
                result["memories_updated"] += 1
                updated = True
                break

        # Create new memory if no similar one found
        if not updated:
            initial_q = 0.5 + (reward * 0.5)
            # THIS is the site that put telemetry into the semantic index.
            # `context` arrives carrying elapsed_seconds / tokens_generated /
            # predicted_tps / question_id and was stored verbatim AND used as
            # embedding input, which is how 27,123 of 54,960 rows came to be
            # number-blobs with no task text. build_memory_record routes those
            # keys into `metrics`, where they are stored but never embedded.
            #
            # M-11a2b: work fields are lifted OUT of that metrics sweep first.
            # A caller may pass them nested (`context["work"]`) or flat
            # (`context["answer"]`); either way they belong in the record's work
            # slots, not in `metrics` — a work payload filed as telemetry is
            # stored under a key no reader looks at, which is the same
            # wrong-key-silent-miss shape as the objective/task_description
            # split this contract exists to close.
            work = extract_work(context)
            metrics_excluded = (
                "task_type", "objective", "priority", "task_description", "source",
                "work", *WORK_KEYS,
            )
            record = build_memory_record(
                objective=task_description,
                task_type=context.get("task_type", "chat"),
                source="external",
                answer=work.get("answer"),
                tool_calls=work.get("tool_calls"),
                repl_steps=work.get("repl_steps"),
                reasoning=work.get("reasoning"),
                metrics={
                    k: v
                    for k, v in context.items()
                    if k not in metrics_excluded
                },
            )

            memory_id = self.store.store(
                embedding=emb_array,
                action=action,
                action_type=action_type,
                model_id=_model_id_for_action(action),
                context=record.to_context(),
                outcome=outcome_from_reward(reward),
                initial_q=initial_q,
            )

            self.logger.log(
                ProgressEntry(
                    event_type=EventType.MEMORY_STORED,
                    task_id="external",
                    memory_id=memory_id,
                    data={
                        "action_type": action_type,
                        "initial_q": initial_q,
                        "source": "external_score",
                    },
                )
            )
            result["memories_created"] = 1

        return result




# ClaudeAsJudge moved to q_judge.py (2026-05-22 Task-G refactor).
# Re-exported so existing imports keep working.
from .q_judge import ClaudeAsJudge  # noqa: E402,F401

# Pure reward computation moved to q_reward.py (2026-05-22 Task-G refactor).
from .q_reward import compute_reward as _compute_reward_impl  # noqa: E402
