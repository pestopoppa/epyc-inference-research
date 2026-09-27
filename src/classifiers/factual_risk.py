"""Factual-risk scorer for routing intelligence.

Scores prompts for hallucination risk based on input-side features using
fast, deterministic regex patterns. No model inference required.

This is COMPLEMENTARY to conformal prediction (which gates on output
uncertainty in HybridRouter). Factual-risk scores INPUT characteristics;
conformal prediction scores OUTPUT uncertainty. They should not conflict:
if conformal prediction already rejects a routing, factual-risk is moot.

Deployed behind the ``factual_risk.mode`` config key:
- ``off``:     No scoring (default until validated)
- ``shadow``:  Compute and log risk, no routing changes
- ``enforce``: Risk feeds into routing/escalation/review decisions

Phase 3 of the routing-intelligence handoff.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.roles import Role


@dataclass
class FactualRiskResult:
    """Result from factual-risk assessment.

    Attributes:
        risk_score: Raw prompt-based risk in [0, 1].
        adjusted_risk_score: Risk adjusted for assigned role capability.
        risk_band: Discretized band — "low", "medium", or "high".
        risk_features: Individual feature values for telemetry/debugging.
        role_adjustment: Multiplier applied for the role.
    """

    risk_score: float = 0.0
    adjusted_risk_score: float = 0.0
    risk_band: str = "low"
    risk_features: dict[str, Any] = field(default_factory=dict)
    role_adjustment: float = 1.0


# ---------------------------------------------------------------------------
# Feature extraction — regex-only, no spaCy
# ---------------------------------------------------------------------------

# Date-seeking patterns: "when did", "what year", "what date", "in 2024", etc.
_DATE_PATTERNS = re.compile(
    r"\b(?:when\s+(?:did|was|were|is|will)|what\s+(?:year|date|day|month|century))"
    r"|\b(?:in\s+\d{4})\b"
    r"|\b(?:born|died|founded|established|invented)\b",
    re.IGNORECASE,
)

# Entity-seeking patterns: "who is", "who was", "name the", capitals indicating proper nouns
_ENTITY_PATTERNS = re.compile(
    r"\b(?:who\s+(?:is|was|were|are)|name\s+the|which\s+(?:person|company|country|city|author))"
    r"|\b(?:founder|CEO|president|inventor|discoverer)\s+of\b",
    re.IGNORECASE,
)

# Citation/source patterns: "cite", "source", "reference", "according to"
_CITATION_PATTERNS = re.compile(
    r"\b(?:cite|citation|source|reference|bibliography|according\s+to|peer.?reviewed)\b",
    re.IGNORECASE,
)

# Factual keyword patterns — words that signal factual (vs creative/code) queries
_FACTUAL_KEYWORDS = re.compile(
    r"\b(?:fact|true|false|correct|incorrect|accurate|inaccurate"
    r"|verify|confirm|evidence|proof|statistic|percentage|number of"
    r"|how\s+many|how\s+much|what\s+is\s+the|capital\s+of"
    r"|population|distance|temperature|speed|weight|height"
    r"|located|country|language|currency)\b",
    re.IGNORECASE,
)

# Uncertainty/hedging markers (presence REDUCES risk — model is being cautious)
_UNCERTAINTY_MARKERS = re.compile(
    r"\b(?:maybe|perhaps|possibly|approximately|roughly|estimated"
    r"|I(?:'m| am)\s+not\s+(?:sure|certain)|uncertain|unclear"
    r"|it\s+(?:depends|varies)|could\s+be|might\s+be)\b",
    re.IGNORECASE,
)

# Assertion-like sentence endings for claim density
_ASSERTION_PATTERN = re.compile(
    r"[A-Z][^.!?]*(?:is|are|was|were|has|have|had|will|can|does|do)\s+[^.!?]+[.!]",
)


def _extract_features(prompt: str) -> dict[str, float]:
    """Extract factual-risk features from a prompt.

    All features are normalized to [0, 1].

    Args:
        prompt: The user's input prompt.

    Returns:
        Dictionary of feature name → value.
    """
    if not prompt:
        return {
            "has_date_question": 0.0,
            "has_entity_question": 0.0,
            "has_citation_request": 0.0,
            "claim_density": 0.0,
            "factual_keyword_ratio": 0.0,
            "uncertainty_markers": 0.0,
        }

    words = prompt.split()
    n_words = max(len(words), 1)

    # Binary flags (0.0 or 1.0)
    has_date = 1.0 if _DATE_PATTERNS.search(prompt) else 0.0
    has_entity = 1.0 if _ENTITY_PATTERNS.search(prompt) else 0.0
    has_citation = 1.0 if _CITATION_PATTERNS.search(prompt) else 0.0

    # Claim density: ratio of assertion-like sentences to total sentences
    sentences = re.split(r"[.!?]+", prompt)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)
    n_assertions = len(_ASSERTION_PATTERN.findall(prompt))
    claim_density = min(n_assertions / n_sentences, 1.0)

    # Factual keyword ratio: proportion of words that are factual keywords
    factual_matches = _FACTUAL_KEYWORDS.findall(prompt)
    factual_ratio = min(len(factual_matches) / n_words, 1.0)

    # Uncertainty markers (inverse signal — presence reduces risk)
    uncertainty_matches = _UNCERTAINTY_MARKERS.findall(prompt)
    uncertainty = min(len(uncertainty_matches) / max(n_sentences, 1), 1.0)

    return {
        "has_date_question": has_date,
        "has_entity_question": has_entity,
        "has_citation_request": has_citation,
        "claim_density": claim_density,
        "factual_keyword_ratio": factual_ratio,
        "uncertainty_markers": uncertainty,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Default weights (overridden by config if available)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "has_date_question": 0.15,
    "has_entity_question": 0.15,
    "has_citation_request": 0.10,
    "claim_density": 0.25,
    "factual_keyword_ratio": 0.20,
    "uncertainty_markers": 0.15,  # applied as negative contribution
}

# Default role adjustment tiers. Role -> tier selection is derived from generated
# stack priors when available; these values remain the tier multipliers. G12
# AA-Omniscience calibration accepted deterministic 4-class scoring for this
# role-tier update on 2026-06-20.
_DEFAULT_ROLE_TIERS: dict[str, float] = {
    "tier_1": 0.727978,
    "tier_2": 0.824178,
    "tier_3": 1.0,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STACK_PRIORS_PATH = PROJECT_ROOT / "orchestration" / "derived" / "stack_priors.yaml"

_TIER_1_MIN_MODEL_MEM_GB = 60.0
_TIER_2_MIN_MODEL_MEM_GB = 18.0

_STACK_PRIOR_ROLE_TIERS_CACHE: dict[str, str] | None = None


def _tier_from_model_mem(mem_gb: float) -> str:
    if mem_gb >= _TIER_1_MIN_MODEL_MEM_GB:
        return "tier_1"
    if mem_gb >= _TIER_2_MIN_MODEL_MEM_GB:
        return "tier_2"
    return "tier_3"


def _role_tiers_from_stack_priors(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> dict[str, str]:
    """Return live role -> factual-risk tier from generated stack priors."""
    try:
        from src.registry.stack_priors import (
            live_stack_role_records,
            stack_prior_model_mem_gb,
        )
    except Exception:
        return {}

    tiers: dict[str, str] = {}
    for role, record in live_stack_role_records(stack_priors_path).items():
        mem_gb = stack_prior_model_mem_gb(record)
        if mem_gb is None:
            continue
        tiers[role] = _tier_from_model_mem(mem_gb)
    return tiers


def _stack_prior_role_tiers(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> dict[str, str]:
    global _STACK_PRIOR_ROLE_TIERS_CACHE
    if stack_priors_path == DEFAULT_STACK_PRIORS_PATH:
        if _STACK_PRIOR_ROLE_TIERS_CACHE is None:
            _STACK_PRIOR_ROLE_TIERS_CACHE = _role_tiers_from_stack_priors(stack_priors_path)
        return _STACK_PRIOR_ROLE_TIERS_CACHE
    return _role_tiers_from_stack_priors(stack_priors_path)


# Degraded tier per SERVING PROCESS (a registry server_mode host row), plus the
# legacy role names the registry does not declare. Used only when stack priors
# are unavailable. Roles that ride another role's process (shared_with /
# alias_of) are NOT listed: they resolve to their host through the registry in
# _degraded_serving_host, which is what the live derivation does too (a stack-
# prior record carries its host process's mem_gb).
# 2026-09-27 ARCHITECT SWAP (operator-decided): tier follows the PROCESS (model
# mem_gb), so architect_general (now the ~90 GB Flash-Next CPU :8074 process) is
# tier_1 and architect_critic (now the 27.05 GB MI210 27B :8083) is tier_2.
_DEGRADED_PROCESS_TIERS: dict[str, str] = {
    "architect_general": "tier_1",
    "thinking_exploration": "tier_1",
    "architect_critic": "tier_2",
    "coder_general": "tier_2",
    "frontdoor": "tier_2",
    "worker_vision": "tier_3",
}

_SERVING_HOST_BY_ROLE_CACHE: dict[str, str] | None = None


def _degraded_serving_host(role_name: str) -> str:
    """The registry ``server_mode`` host whose process serves ``role_name``.

    Returns ``role_name`` itself when it is a host, unknown to the registry, or
    the registry is unreadable (an alias then gets no degraded tier, which
    scores it as tier_3, with no discount).
    """
    global _SERVING_HOST_BY_ROLE_CACHE
    if _SERVING_HOST_BY_ROLE_CACHE is None:
        try:
            from src.registry.registry_loader import load_server_mode, serving_host_by_role

            _SERVING_HOST_BY_ROLE_CACHE = serving_host_by_role(load_server_mode())
        except Exception:
            _SERVING_HOST_BY_ROLE_CACHE = {}
    return _SERVING_HOST_BY_ROLE_CACHE.get(role_name, role_name)


def _degraded_role_tier(role_name: str) -> str | None:
    """Return the degraded role tier for a canonical role name.

    Used ONLY when generated stack priors are unavailable, so it cannot be derived
    from them. The per-process tier is a hand table (``_DEGRADED_PROCESS_TIERS``)
    and has to agree with the live derivation (``_tier_from_model_mem`` over
    stack-prior ``mem_gb``); which process serves a role is read from the
    registry. ``tests/unit/test_factual_risk.py`` guards the agreement.

    2026-08-01 W1 cutover corrections (all three previously disagreed with the
    live derivation, so the degraded path scored these roles differently from the
    healthy path):
      * ``architect_general`` tier_1 -> tier_2 — the 122B moved off it onto
        ``architect_critic``; it now serves the 27.05 GB MI210 model.
      * ``architect_critic`` added as tier_1 — it inherited the 69 GB 122B. It was
        absent entirely, so degraded mode fell through to tier_3 and gave the
        STRONGEST model in the fleet no risk discount at all.
      * ``vision_escalation`` tier_2 -> tier_3 — it is an alias on the 17.3 GB
        worker_vision process, which is below ``_TIER_2_MIN_MODEL_MEM_GB``.

    2026-09-27 ARCHITECT SWAP (operator-decided): the two architect labels swapped
    over unchanged processes, so the two tiers swap with them —
    ``architect_general`` tier_2 -> tier_1 (now Flash-Next, ~90 GB CPU :8074) and
    ``architect_critic`` tier_1 -> tier_2 (now the 27.05 GB MI210 27B :8083). The
    coder_escalation / ingest_long_context aliases resolve through the registry to
    their host (architect_critic) and stay tier_2.

    2026-09-26: the hand table still held worker_general, worker_math and
    toolrunner at tier_3 after the operator-signed 2026-09-22 cutover
    (orchestrator 860b0b2d) put them on frontdoor's 35B-A3B :8070 process, which
    live priors score tier_2. Alias rows are no longer restated here; they are
    resolved to their host through the registry.
    """
    host = _degraded_serving_host(role_name)
    return _DEGRADED_PROCESS_TIERS.get(host) or _DEGRADED_PROCESS_TIERS.get(role_name)


def _role_tier_for_role(
    role: str,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> str:
    normalized = Role.from_string(role)
    role_name = normalized.value if normalized is not None else str(role)
    return (
        _stack_prior_role_tiers(stack_priors_path).get(role_name)
        or _degraded_role_tier(role_name)
        or "tier_3"
    )


def _get_config() -> dict[str, Any]:
    """Load factual_risk config from classifier_config.yaml.

    Returns empty dict if config unavailable (scorer uses defaults).
    """
    try:
        from src.classifiers.config_loader import get_classifier_config

        cfg = get_classifier_config()
        return cfg.get("factual_risk", {})
    except Exception:
        return {}


def _compute_score(features: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Compute raw risk score from features.

    Uncertainty markers are an INVERSE signal (they reduce risk).

    Args:
        features: Feature dict from _extract_features.
        weights: Optional weight overrides.

    Returns:
        Risk score in [0, 1].
    """
    w = weights or _DEFAULT_WEIGHTS
    score = 0.0
    for name, value in features.items():
        weight = w.get(name, 0.0)
        if name == "uncertainty_markers":
            # Inverse: uncertainty reduces risk
            score += weight * (1.0 - value)
        else:
            score += weight * value
    return max(0.0, min(1.0, score))


def _role_adjustment(
    role: str,
    config: dict[str, Any] | None = None,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> float:
    """Get risk adjustment multiplier for a role.

    Stronger models reduce effective risk because they hallucinate less.

    Args:
        role: Role name (e.g. "architect_general").
        config: Optional config dict with role_adjustments.
        stack_priors_path: Generated stack-prior artifact to derive live role tiers.

    Returns:
        Multiplier in (0, 1].
    """
    tier = _role_tier_for_role(role, stack_priors_path=stack_priors_path)
    if config and "role_adjustments" in config:
        return float(config["role_adjustments"].get(tier, 1.0))
    return _DEFAULT_ROLE_TIERS.get(tier, 1.0)


def _band(score: float, config: dict[str, Any] | None = None) -> str:
    """Discretize a risk score into a band.

    Args:
        score: Risk score in [0, 1].
        config: Optional config with threshold_low/threshold_high.

    Returns:
        "low", "medium", or "high".
    """
    low = float((config or {}).get("threshold_low", 0.3))
    high = float((config or {}).get("threshold_high", 0.7))
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def assess_risk(
    prompt: str,
    role: str = "",
    config: dict[str, Any] | None = None,
) -> FactualRiskResult:
    """Assess factual-risk of a prompt.

    Fast, deterministic, regex-only. No model inference.

    Args:
        prompt: User's input prompt.
        role: Assigned role name (for per-role calibration). Empty = no adjustment.
        config: Optional config override. If None, loads from classifier_config.yaml.

    Returns:
        FactualRiskResult with scores, band, and feature breakdown.
    """
    if config is None:
        config = _get_config()

    features = _extract_features(prompt)

    # Load weights from config if available
    weights = config.get("feature_weights", None)
    if weights:
        weights = {k: float(v) for k, v in weights.items()}

    raw_score = _compute_score(features, weights)

    # Per-role adjustment
    adjustment = _role_adjustment(role, config) if role else 1.0
    adjusted = raw_score * adjustment

    risk_band = _band(adjusted, config)

    return FactualRiskResult(
        risk_score=round(raw_score, 4),
        adjusted_risk_score=round(adjusted, 4),
        risk_band=risk_band,
        risk_features=features,
        role_adjustment=adjustment,
    )


def get_configured_mode(config: dict[str, Any] | None = None) -> str:
    """Return configured factual-risk mode without sampling the canary arm."""
    import os

    env_mode = os.environ.get("ORCHESTRATOR_FACTUAL_RISK_MODE")
    if env_mode and env_mode in ("off", "shadow", "enforce"):
        return env_mode
    if config is None:
        config = _get_config()
    return str(config.get("mode", "off"))


def _canary_roll(sample_key: str, *, role: str, salt: str = "") -> float:
    payload = f"{salt}\0{role}\0{sample_key}".encode("utf-8", errors="replace")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def get_mode(
    config: dict[str, Any] | None = None,
    role: str = "",
    sample_key: str = "",
) -> str:
    """Get the current factual-risk mode.

    Environment override: ``ORCHESTRATOR_FACTUAL_RISK_MODE`` takes
    precedence over classifier_config.yaml when set.

    Supports "canary" mode (RI-10): probabilistically returns "enforce"
    for a fraction of requests, filtered by role.

    Args:
        config: Optional config override.
        role: Current routing role (for canary role filtering).
        sample_key: Stable request key for deterministic canary arm assignment.
            Callers should pass task_id/request_id. If omitted, legacy RNG sampling
            is retained for compatibility with direct/unit callers.

    Returns:
        "off", "shadow", or "enforce".
    """
    import random

    if config is None:
        config = _get_config()

    mode = get_configured_mode(config)

    if mode == "canary":
        # RI-10: Probabilistic canary — enforce for a fraction of requests
        canary_ratio = max(0.0, min(1.0, float(config.get("canary_ratio", 0.25))))
        canary_roles = config.get("canary_roles", [])
        # If role filter is set and current role doesn't match, stay in shadow
        if canary_roles and role and role not in canary_roles:
            return "shadow"
        if sample_key:
            roll = _canary_roll(
                str(sample_key),
                role=role,
                salt=str(config.get("canary_salt", "ri10-v1")),
            )
        else:
            roll = random.random()
        return "enforce" if roll < canary_ratio else "shadow"

    return mode
