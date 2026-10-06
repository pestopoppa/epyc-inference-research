#!/usr/bin/env python3
"""Validate generated stack priors against current source artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.registry.stack_priors import (  # noqa: E402
    _launch_runtime_record,
    _server_mode_launch_requirement_overrides,
    validate_stack_priors_contract,
)

DEFAULT_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
DEFAULT_DESCRIPTORS = REPO_ROOT / "orchestration" / "model_descriptors.yaml"
DEFAULT_PRIORS = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_SURFACE_EXCEPTIONS = REPO_ROOT / "orchestration" / "stack_change_guard_exceptions.yaml"
DEFAULT_SURFACE_MANIFEST = REPO_ROOT / "orchestration" / "stack_change_surface_manifest.yaml"
DEFAULT_ADD_MODEL_PROCEDURE = REPO_ROOT / "orchestration" / "procedures" / "add_model_to_registry.yaml"
DEFAULT_PROCEDURE_SCHEMA = REPO_ROOT / "orchestration" / "procedure.schema.json"
DEFAULT_ACCEPTED_GAPS = REPO_ROOT / "orchestration" / "accepted_gaps.yaml"
DEFAULT_MASTER_REGISTRY = Path(
    "/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml"
)
STANDALONE_NUMA_MODES = frozenset({"full", "quarter", "both"})

# Documented FLOOR, not the answer. `RETIRED_LIVE_ROLES` used to BE this literal:
# a hand-written restatement of "historic roles minus live roles" that could only
# ever catch staleness somebody remembered to type in. It is now derived (see
# `derive_retired_live_roles`), and the literal survives only so that a role the
# guard already catches can never be lost to a derivation change. Derivation
# failure RAISES; it never degrades silently to this floor alone.
RETIRED_LIVE_ROLE_FLOOR = frozenset({"architect_coding"})
# Master-registry keys that mark an entry as no longer deployable. `deprecated`
# and `retired` are the two vocabularies actually in use; the *_reason/*_date
# companions are accepted so an entry that carries only the prose marker is not
# read as live.
RETIRED_ROLE_MARKER_FIELDS = (
    "deprecated",
    "deprecated_reason",
    "deprecated_date",
    "deprecation_reason",
    "retired",
    "retired_reason",
    "retired_date",
)
MANIFEST_OWNED_AUXILIARY_LAUNCH_ROLES = frozenset({"worker_fast"})
MANIFEST_OWNED_DEFAULT_AUXILIARY_LAUNCH_ROLES = frozenset({"eval_batch_frontdoor"})
MANIFEST_OWNED_AUXILIARY_LAUNCH_MODES = frozenset({"embedding"})
REQUIRED_SOURCE_ARTIFACTS = (
    "registry",
    "descriptors",
    "stack_manifest",
    "stack_numa",
    "orchestrator_stack",
    "stack_paths",
    "stack_runtime",
)
# Commit-time staleness gate (P1-10). The pins in `source_artifacts` are only
# meaningful against the bytes a commit will CONTAIN, so `staged` reads the git
# index; `worktree` reads the files on disk and blames every mismatch.
STALENESS_SCOPES = ("staged", "worktree")
RECOMPILE_PRIORS_COMMAND = "uv run python scripts/registry/compile_stack_priors.py"
SOURCE_ARTIFACT_REMEDIATION = f"regenerate the priors: {RECOMPILE_PRIORS_COMMAND}"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# `git diff --cached` against an empty repository has no HEAD to diff; git's
# well-known empty-tree object stands in so the first commit is still scoped.
_EMPTY_TREE_OID = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
SURFACE_SCAN_ALLOW_MARKER = "stack-change-guard: allow"
SURFACE_SCAN_MAX_FILE_BYTES = 512 * 1024
SURFACE_EXCEPTION_CLASSIFICATIONS = frozenset(
    {
        "degraded_fallback",
        "legacy_test",
        "historical_doc",
        "intentional_live_exception",
    }
)
SURFACE_WARNING_ORDER = (
    "production_blocker",
    "waived_production_blocker",
    "legacy_test",
    "historical_doc",
)


@dataclass(frozen=True)
class GuardResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class HardcodedSurfaceRule:
    rule_id: str
    category: str
    pattern: str
    path_globs: tuple[str, ...]
    remediation: str
    exclude_globs: tuple[str, ...] = ()
    ignore_comment_lines: bool = False
    # 2026-08-02: when true, `pattern` is REPLACED at scan time by an alternation
    # over `retired_live_roles()` — the derived set — instead of being a literal
    # naming one retired role. See `_derive_retired_role_patterns`.
    derive_retired_roles: bool = False


@dataclass(frozen=True)
class SurfaceFinding:
    rule_id: str
    category: str
    path: Path
    line: int
    snippet: str
    remediation: str

    def to_warning(self) -> str:
        return (
            f"hardcoded_surface.{self.category}.{self.rule_id}: "
            f"{self.path}:{self.line}: {self.snippet} "
            f"[remediation: {self.remediation}]"
        )


@dataclass(frozen=True)
class SurfaceException:
    rule_id: str
    category: str
    path_glob: str
    classification: str
    owner: str
    rationale: str
    expires: str
    line: int | None = None

    def matches(self, finding: SurfaceFinding) -> bool:
        if self.rule_id != finding.rule_id or self.category != finding.category:
            return False
        if self.line is not None and self.line != finding.line:
            return False
        return fnmatch.fnmatch(finding.path.as_posix(), self.path_glob)

    def warning_suffix(self) -> str:
        return (
            f"classification={self.classification}; owner={self.owner}; "
            f"expires={self.expires}; rationale={self.rationale}"
        )


@dataclass(frozen=True)
class AcceptedGapDeclaration:
    """An operator-declared, owned, EXPIRING acceptance of one stack-prior gap.

    The gate used to have exactly one severity: a known gap the operator had
    consciously accepted was indistinguishable from an unsafe launch, so the only
    available answer was ORCHESTRATOR_SKIP_STACK_CHANGE_GATE=1 — which disables
    every check, including the load-bearing ones. A declaration downgrades ONE
    named gap on ONE named role to a visible warning. It never matches by
    wildcard: role and gap string must both be exact, or a declaration filed for
    a missing quality prior would also swallow a missing live server binding.
    """

    role: str
    gap: str
    reason: str
    owner: str
    declared: str
    expires: str

    def matches(self, role: str, gap: str) -> bool:
        return self.role == role and self.gap == gap

    def warning_suffix(self) -> str:
        return (
            f"owner={self.owner}; declared={self.declared}; "
            f"expires={self.expires}; reason={self.reason}"
        )


class RetiredRoleDerivationError(RuntimeError):
    """Raised when the retired-role set cannot be derived from its sources."""


# Prefix for the THIRD outcome. PASS and FAIL are not the whole vocabulary: a
# guard that cannot evaluate its condition must say so, loudly and non-zero,
# because "I found no violations" and "I could not look" are different facts and
# only one of them is an assurance. Emitted as an ERROR (never a warning), so the
# gate goes red on the instrument failing exactly as it does on the invariant
# failing.
COULD_NOT_CHECK = "COULD-NOT-CHECK"


class LaunchViewUnavailableError(RuntimeError):
    """Raised when a launch-view input cannot be evaluated at all.

    Deliberately an exception rather than a neutral return. ``{}`` / ``None`` /
    ``[]`` are indistinguishable from "nothing to report", and every caller in
    this file reads them as "no violations" — so a helper that returns one on an
    import or parse failure converts an unknown into a false assurance. Raising
    forces the caller to decide, and the caller records COULD-NOT-CHECK.
    """


def _has_retired_marker(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    for field in RETIRED_ROLE_MARKER_FIELDS:
        value = record.get(field)
        if value is None or value is False:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def _registry_role_names(path: Path, label: str) -> set[str]:
    try:
        loaded = _load_yaml(path)
    except FileNotFoundError as exc:
        raise RetiredRoleDerivationError(f"{label} registry is missing: {path}") from exc
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RetiredRoleDerivationError(f"{label} registry is unreadable ({path}): {exc}") from exc
    roles = loaded.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise RetiredRoleDerivationError(
            f"{label} registry has no mapping-valued 'roles' section: {path}"
        )
    return {name for name in roles if isinstance(name, str)}


def derive_retired_live_roles(
    *,
    master_registry_path: Path = DEFAULT_MASTER_REGISTRY,
    lean_registry_path: Path = DEFAULT_REGISTRY,
    floor: frozenset[str] = RETIRED_LIVE_ROLE_FLOOR,
) -> frozenset[str]:
    """Derive "historic roles minus live roles" instead of restating it by hand.

    Historic names come from the MASTER registry, where a decommissioned entry
    keeps its row and gains a retirement marker. Live names come from the
    COMPILED lean registry — deliberately not from the stack priors under
    validation, because the priors are the artifact being checked: sourcing
    "live" from them would make "a retired role reappeared in the priors"
    unprovable by construction.

    Raises RetiredRoleDerivationError when either source is missing or shaped
    wrong. Falling back to the floor would hand back a guard that looks like it
    is checking 40+ names while actually checking one.
    """
    try:
        master = _load_yaml(master_registry_path)
    except FileNotFoundError as exc:
        raise RetiredRoleDerivationError(
            f"master registry is missing: {master_registry_path}"
        ) from exc
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RetiredRoleDerivationError(
            f"master registry is unreadable ({master_registry_path}): {exc}"
        ) from exc
    master_roles = master.get("roles")
    if not isinstance(master_roles, dict) or not master_roles:
        raise RetiredRoleDerivationError(
            f"master registry has no mapping-valued 'roles' section: {master_registry_path}"
        )

    live_roles = _registry_role_names(lean_registry_path, "compiled lean")
    retired = {
        name
        for name, record in master_roles.items()
        if isinstance(name, str) and _has_retired_marker(record)
    }
    # A name that is retired upstream AND live downstream is a registry
    # contradiction, not a retired role. Subtract it: mis-flagging a live role
    # would make every stack change fail on a name the fleet is actually serving.
    return frozenset((retired - live_roles) | floor)


_RETIRED_LIVE_ROLES_CACHE: frozenset[str] | None = None


def retired_live_roles(*, refresh: bool = False) -> frozenset[str]:
    """Cached accessor for the derived retired-role set."""
    global _RETIRED_LIVE_ROLES_CACHE
    if refresh or _RETIRED_LIVE_ROLES_CACHE is None:
        _RETIRED_LIVE_ROLES_CACHE = derive_retired_live_roles()
    return _RETIRED_LIVE_ROLES_CACHE


def _retired_live_roles_or_error() -> tuple[frozenset[str], list[str]]:
    """Resolve the retired-role set, reporting derivation failure as an ERROR.

    The degraded floor is still used for the remaining checks so one broken
    source does not blind every other invariant — but the failure is recorded,
    never swallowed.
    """
    try:
        return retired_live_roles(), []
    except RetiredRoleDerivationError as exc:
        return RETIRED_LIVE_ROLE_FLOOR, [
            f"retired-role derivation failed: {exc}; "
            f"falling back to the documented floor {sorted(RETIRED_LIVE_ROLE_FLOOR)} "
            "for this run only"
        ]


HARDCODED_SURFACE_RULES: tuple[HardcodedSurfaceRule, ...] = (
    HardcodedSurfaceRule(
        rule_id="retired_role_in_active_code",
        category="production_blocker",
        pattern=r"\barchitect_coding\b",
        path_globs=(
            "src/**/*.py",
            "scripts/benchmark/*.py",
            "scripts/server/*.py",
        ),
        exclude_globs=(
            "scripts/validate/stack_change_guard.py",
            "scripts/benchmark/deprecated/**",
        ),
        remediation="remove from live behavior or mark explicit legacy/test-only",
        derive_retired_roles=True,
        ignore_comment_lines=True,
    ),
    HardcodedSurfaceRule(
        rule_id="retired_role_env_flag",
        category="production_blocker",
        pattern=r"\b(?:ORCHESTRATOR(?:_FEATURE)?_)?LANGGRAPH_ARCHITECT_CODING\b",
        path_globs=("scripts/server/*.py",),
        remediation="do not enable retired LangGraph architect_coding launch flags",
        ignore_comment_lines=True,
    ),
    HardcodedSurfaceRule(
        rule_id="stale_procedure_role_enum",
        category="production_blocker",
        pattern=r"\barchitect_coding\b",
        path_globs=("orchestration/procedures/*.yaml",),
        remediation="compile procedure role choices from stack priors",
        derive_retired_roles=True,
    ),
    # 2026-08-01: `retired_role_in_lean_registry` REMOVED. Its only target,
    # orchestration/model_registry_lean.yaml, was deleted — it was a stale second
    # role table from 2026-06-13 still selectable via ORCHESTRATOR_REGISTRY_MODE=lean,
    # disagreeing with the compiled registry on tier, acceleration recipe, timeouts
    # and escalation chains. The real lean artifact is orchestration/model_registry.yaml,
    # COMPILED from master, so a retired role cannot persist in it: it is regenerated,
    # not maintained. A rule matching no file is not a safety net, it is a green light
    # that means nothing. Removed with its manifest entry (the bijection is enforced).
    HardcodedSurfaceRule(
        rule_id="retired_role_in_source_access",
        category="production_blocker",
        pattern=r"\barchitect_coding\b",
        path_globs=("orchestration/source_registry.yaml",),
        remediation="remove retired roles from web-source role_access metadata",
        derive_retired_roles=True,
    ),
    # 2026-08-01: `retired_role_in_quality_signature` REMOVED with its target file.
    #
    # Worth recording WHY that file was deleted rather than fixed, because this rule
    # is the cautionary tale: it scanned model_quality_signatures.yaml for the string
    # `architect_coding` and passed clean, while all FOUR rows of that file described
    # the fleet retired 2026-05-08 at throughputs 1.4x-11x too low. None of them
    # happened to contain the one retired name the rule knew about.
    #
    # The lesson generalises to the rules that remain: the retired-role set used to be
    # a hand-written restatement of "historic roles minus live roles". A name-matching
    # guard can only catch staleness it was told to look for, so it cannot detect a
    # table that is stale in every VALUE while naming only current roles. The durable
    # form is a comparison against the compiled artifact, not a grep for known-bad
    # strings.
    #
    # 2026-08-01: both halves now fixed. The retired set is DERIVED
    # (`derive_retired_live_roles`), and value staleness is caught by
    # ROLE_FACT_SURFACE_RULES, which compares a config surface's per-role facts
    # against the compiled stack priors instead of grepping for known-bad names.
    HardcodedSurfaceRule(
        rule_id="retired_role_in_tests",
        category="legacy_test",
        pattern=r"\barchitect_coding\b",
        path_globs=("tests/**/*.py", "scripts/memory/**/*.py"),
        exclude_globs=("tests/unit/test_stack_change_guard.py",),
        remediation="label as retired-role coverage or migrate fixture to stack priors",
        derive_retired_roles=True,
    ),
    HardcodedSurfaceRule(
        rule_id="retired_role_in_operator_docs",
        category="historical_doc",
        pattern=r"\barchitect_coding\b",
        path_globs=("docs/**/*.md",),
        remediation="generate current stack tables or label snapshot as historical",
        derive_retired_roles=True,
    ),
    HardcodedSurfaceRule(
        rule_id="bilinear_model_specs_table",
        category="production_blocker",
        pattern=r"\bmodel_specs\b|\barchitect_coding\b",
        path_globs=("orchestration/repl_memory/bilinear_scorer.py",),
        remediation="derive model features from stack priors/descriptors",
    ),
    HardcodedSurfaceRule(
        rule_id="seeding_baseline_tps_table",
        category="production_blocker",
        pattern=r"\bDEFAULT_BASELINE_TPS\b|\bbaseline_tps\b|\barchitect_coding\b",
        path_globs=("scripts/benchmark/seeding_rewards.py",),
        remediation="derive seeding reward costs from stack priors",
    ),
    HardcodedSurfaceRule(
        rule_id="legacy_cli_port_probe_map",
        category="production_blocker",
        pattern=r"\b8084\b|\barchitect_coding\b|ports\s*=\s*\[8080",
        path_globs=("src/cli_orch.py",),
        remediation="derive status probes from stack priors or stack manifest API",
    ),
    HardcodedSurfaceRule(
        rule_id="static_cli_degraded_status_targets",
        category="production_blocker",
        pattern=r"\bFALLBACK_STATUS_TARGETS\s*=\s*\[",
        path_globs=("src/cli_orch.py",),
        remediation="derive degraded status targets from stack_manifest PORT_MAP/HOT_ROLES",
    ),
    HardcodedSurfaceRule(
        rule_id="static_cli_status_excluded_roles",
        category="production_blocker",
        pattern=r"^FALLBACK_STATUS_EXCLUDED_ROLES\b\s*(?::[^=]+)?=",
        path_globs=("src/cli_orch.py",),
        remediation="derive degraded status exclusions from stack_manifest ROLE_LAUNCH_META launch modes",
    ),
    HardcodedSurfaceRule(
        rule_id="static_autopilot_preflight_targets",
        category="production_blocker",
        pattern=r"\bFALLBACK_MODEL_SERVER_TARGETS\s*=\s*\[",
        path_globs=("scripts/autopilot/preflight_audit.py",),
        remediation="derive degraded model-server preflight targets from stack_manifest PORT_MAP/HOT_ROLES",
    ),
    HardcodedSurfaceRule(
        rule_id="static_autopilot_preflight_excluded_roles",
        category="production_blocker",
        pattern=r"^FALLBACK_MODEL_SERVER_EXCLUDED_ROLES\b\s*(?::[^=]+)?=",
        path_globs=("scripts/autopilot/preflight_audit.py",),
        remediation="derive degraded preflight exclusions from stack_manifest ROLE_LAUNCH_META launch modes",
    ),
    HardcodedSurfaceRule(
        rule_id="stale_corpus_quality_gate_models",
        category="production_blocker",
        pattern=r"\bFALLBACK_MODELS\s*=\s*\{|\bdefault=\[[\"']7b[\"'],\s*[\"']32b[\"']\]",
        path_globs=("scripts/benchmark/corpus_quality_gate.py",),
        remediation="derive corpus quality gate model choices from stack priors or stack_manifest roles",
    ),
    HardcodedSurfaceRule(
        rule_id="local_config_stack_prior_yaml_reader",
        category="production_blocker",
        pattern=r"\byaml\.safe_load\s*\(\s*priors_path\.read_text",
        path_globs=("src/config/models.py",),
        remediation="reuse src.registry.stack_priors helpers for config server URL defaults",
    ),
    HardcodedSurfaceRule(
        rule_id="local_q_scorer_stack_prior_yaml_reader",
        category="production_blocker",
        pattern=r"\byaml\.safe_load\s*\(\s*stack_priors_path\.read_text",
        path_globs=("orchestration/repl_memory/q_scorer.py",),
        remediation="reuse src.registry.stack_priors helpers for q_scorer stack-prior loading and validation",
    ),
    HardcodedSurfaceRule(
        rule_id="local_generated_docs_stack_prior_yaml_reader",
        category="production_blocker",
        pattern=(
            r"\bstack_priors\s*=\s*(?:_load_yaml|load_yaml)\s*\("
            r"(?:.*stack_priors\.yaml|stack_priors_path)"
        ),
        path_globs=(
            "scripts/autopilot/gen_system_card.py",
            "scripts/registry/render_stack_summary.py",
        ),
        remediation="reuse src.registry.stack_priors helpers for generated stack docs/system cards",
    ),
    HardcodedSurfaceRule(
        rule_id="static_factual_risk_role_tiers",
        category="production_blocker",
        pattern=r"^_ROLE_TO_TIER\b\s*(?::[^=]+)?=",
        path_globs=("src/classifiers/factual_risk.py",),
        remediation="derive factual-risk role capability tiers from generated stack priors",
    ),
    HardcodedSurfaceRule(
        rule_id="static_openai_model_role_order",
        category="production_blocker",
        pattern=r"^PREFERRED_ROLE_ORDER\b\s*(?::[^=]+)?=",
        path_globs=("src/api/routes/openai_compat.py",),
        remediation="derive OpenAI /models role ordering from generated stack-prior topology",
    ),
    HardcodedSurfaceRule(
        rule_id="static_chat_routing_heuristic_prior_roles",
        category="production_blocker",
        pattern=r"^_HEURISTIC_PRIOR_ROLE_CANDIDATES\b\s*(?::[^=]+)?=",
        path_globs=("src/api/routes/chat_routing.py",),
        remediation="derive chat-routing heuristic prior roles from generated stack priors",
    ),
    HardcodedSurfaceRule(
        rule_id="static_inference_lock_role_policy",
        category="production_blocker",
        pattern=(
            r"\b(?:HEAVY_ROLES|LIGHT_ROLES)\b\s*(?::[^=]+)?"
            r"=\s*frozenset\s*\(\s*\{"
        ),
        path_globs=("src/runtime/inference_lock.py",),
        remediation="derive lock role policy from stack priors; keep only explicit _LEGACY_* degraded fallbacks",
    ),
    HardcodedSurfaceRule(
        rule_id="static_inference_tap_stream_policy",
        category="production_blocker",
        pattern=(
            r"\bSAFE_NON_STREAM_ROLES\b\s*(?::[^=]+)?"
            r"=\s*frozenset\s*\(\s*\{"
        ),
        path_globs=("src/runtime/inference_tap.py",),
        remediation="derive tap stream policy from stack-prior model facts; keep only explicit _LEGACY_* degraded fallbacks",
    ),
    HardcodedSurfaceRule(
        rule_id="stale_autopilot_program_stack_guidance",
        category="production_blocker",
        pattern=(
            r"\b(?:8071|8084|architect_coding|512GB|19\.6\s*t/s|12\.7\s*t/s)\b|"
            r"\bTarget ports\b|\bWARM tier demotion\b|"
            r"\bQ-scorer frontdoor throughput\b|"
            r"\bQwen3-Coder-30B\b|\bQwen3\.5-35B\b|"
            r"\byaml\.safe_load\b"
        ),
        path_globs=("scripts/autopilot/program.md",),
        remediation="derive AutoPilot operator endpoints and tier guidance from stack priors/system card",
    ),
    HardcodedSurfaceRule(
        rule_id="static_autopilot_kv_production_ports",
        category="production_blocker",
        pattern=r"\bPRODUCTION_PORTS\s*=\s*\{",
        path_globs=("scripts/autopilot/kv_compress.py",),
        remediation="derive KV-compaction role ports from generated stack priors",
    ),
    HardcodedSurfaceRule(
        rule_id="stale_launch_wrapper_static_inventory",
        category="production_blocker",
        pattern=(
            r"\b(?:8084|architect_coding|Qwen3-Coder-480B|535GB|512GB)\b|"
            r"\bRAM breakdown\b|\bFull HOT tier \+ architects\b|"
            r"\bCore tier only, no architects\b"
        ),
        path_globs=("scripts/server/*.sh",),
        remediation="derive launcher summaries from stack_manifest/stack priors, not static model/RAM tables",
        ignore_comment_lines=True,
    ),
)


@dataclass(frozen=True)
class RoleFactSurfaceRule:
    """A config surface that RESTATES per-role facts the compiled artifact owns.

    Deliberately not a `HardcodedSurfaceRule`: there is no known-bad string to
    grep for. The rule names a role-keyed table and the fields it duplicates,
    and the check compares those values against
    ``orchestration/derived/stack_priors.yaml``. That is the only shape that
    catches drift nobody enumerated in advance — the failure that let
    model_quality_signatures.yaml describe a fleet retired 2026-05-08, at
    throughputs 1.4x-11x too low, while passing a name-matching rule clean
    because every row named a role that was still current.

    Scope is intentionally narrow: only surfaces where "the compiled artifact
    declares the same fact" is unambiguous. Roles absent from the compiled
    artifact are skipped, not flagged — absence is a different check.
    """

    rule_id: str
    category: str
    path_globs: tuple[str, ...]
    roles_key: str
    fields: tuple[str, ...]
    remediation: str
    exclude_globs: tuple[str, ...] = ()
    # Rows the surface marks as pure aliases restate NOTHING about their own
    # serving: `tier: ALIAS` is a structural marker meaning "launches no server",
    # not a serving tier, and comparing it against the compiled `hot` would be a
    # category error that buries the real mismatches under noise. Alias TARGETS
    # are already checked by src/config/stack_templates._validate_stack_prior_parity.
    alias_field: str = "alias_to"
    alias_tier_values: frozenset[str] = frozenset({"alias"})


ROLE_FACT_SURFACE_RULES: tuple[RoleFactSurfaceRule, ...] = (
    RoleFactSurfaceRule(
        rule_id="stale_role_fact_table",
        category="production_blocker",
        path_globs=("stack_templates/*.yaml",),
        exclude_globs=(),
        roles_key="roles",
        # Ports and alias targets are deliberately NOT compared here:
        # src/config/stack_templates._validate_stack_prior_parity already compares
        # them against the same generated priors. A second implementation of the
        # same comparison is a second thing to drift.
        fields=("model", "quant", "tier"),
        remediation=(
            "restate the role's model/quant/tier from the compiled stack priors "
            "(orchestration/derived/stack_priors.yaml) or delete the stale row"
        ),
    ),
)


CONSUMER_SURFACE_CLASSIFICATIONS = frozenset(
    {
        "generated",
        "typed_consumer",
        "explicit_degraded_fallback",
        "legacy_test",
        "historical_doc",
        "open_production_blocker",
    }
)

REQUIRED_CONSUMER_SURFACE_IDS = frozenset(
    {
        "admission_policy",
        "config_model_catalog",
        "dashboard_status_system_cards",
        "generated_stack_docs",
        "health_preflight_probes",
        "launch_maps",
        "lock_tap_policy",
        "planner_prompt_guidance",
        "procedure_role_enums",
        "q_scorer_priors",
        "routing_prior_consumers",
        "runtime_attestation",
        "seeding_reward_priors",
    }
)

CONSUMER_SURFACE_TEXT_FIELDS = (
    "surface_id",
    "classification",
    "owner",
    "consumer_scope",
    "source_of_truth",
    "review_cadence",
    "validation_command",
    "drift_response",
)


def hardcoded_surface_rule_inventory(
    rules: tuple[HardcodedSurfaceRule, ...] = HARDCODED_SURFACE_RULES,
    ownership_manifest: dict[str, Any] | None = None,
    role_fact_rules: tuple[RoleFactSurfaceRule, ...] = ROLE_FACT_SURFACE_RULES,
) -> dict[str, Any]:
    """Return the curated model-specific surface rules as machine-readable data."""
    ownership_by_rule = _surface_manifest_by_rule(ownership_manifest)
    consumer_surfaces = _surface_manifest_consumer_surfaces(ownership_manifest)
    return {
        "version": 1,
        "rule_count": len(rules),
        "role_fact_rule_count": len(role_fact_rules),
        "consumer_surface_count": len(consumer_surfaces),
        "categories": sorted({rule.category for rule in rules}),
        "role_fact_rules": [
            {
                "rule_id": rule.rule_id,
                "category": rule.category,
                "path_globs": list(rule.path_globs),
                "exclude_globs": list(rule.exclude_globs),
                "roles_key": rule.roles_key,
                "fields": list(rule.fields),
                "compared_against": "orchestration/derived/stack_priors.yaml",
                "remediation": rule.remediation,
                "ownership": ownership_by_rule.get(rule.rule_id, {}),
            }
            for rule in role_fact_rules
        ],
        "rules": [
            {
                "rule_id": rule.rule_id,
                "category": rule.category,
                "pattern": rule.pattern,
                "path_globs": list(rule.path_globs),
                "exclude_globs": list(rule.exclude_globs),
                "ignore_comment_lines": rule.ignore_comment_lines,
                "remediation": rule.remediation,
                "ownership": ownership_by_rule.get(rule.rule_id, {}),
            }
            for rule in rules
        ],
        "consumer_surfaces": consumer_surfaces,
    }


def _surface_manifest_by_rule(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict):
        return {}
    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, list):
        return {}
    by_rule: dict[str, dict[str, Any]] = {}
    for raw_surface in surfaces:
        if not isinstance(raw_surface, dict):
            continue
        rule_id = raw_surface.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            continue
        by_rule[rule_id] = {
            key: raw_surface[key]
            for key in (
                "owner",
                "consumer_scope",
                "promotion_blocker",
                "review_cadence",
                "evidence_command",
                "drift_response",
            )
            if key in raw_surface
        }
    return by_rule


def _surface_manifest_consumer_surfaces(
    manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        return []
    raw_surfaces = manifest.get("consumer_surfaces")
    if not isinstance(raw_surfaces, list):
        return []
    surfaces: list[dict[str, Any]] = []
    for raw_surface in raw_surfaces:
        if not isinstance(raw_surface, dict):
            continue
        surface_id = raw_surface.get("surface_id")
        if not isinstance(surface_id, str) or not surface_id:
            continue
        surfaces.append(
            {
                key: raw_surface[key]
                for key in (
                    "surface_id",
                    "classification",
                    "owner",
                    "consumer_scope",
                    "source_of_truth",
                    "promotion_blocker",
                    "review_cadence",
                    "validation_command",
                    "implementation_refs",
                    "drift_response",
                )
                if key in raw_surface
            }
        )
    return sorted(surfaces, key=lambda surface: str(surface.get("surface_id", "")))


def load_surface_manifest(path: Path = DEFAULT_SURFACE_MANIFEST) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the hardcoded-surface ownership manifest."""
    if not path.exists():
        return None, [f"missing hardcoded-surface ownership manifest: {path}"]
    try:
        return _load_yaml(path), []
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return None, [f"failed to load hardcoded-surface ownership manifest {path}: {exc}"]


def validate_surface_manifest(
    path: Path = DEFAULT_SURFACE_MANIFEST,
    *,
    rules: tuple[HardcodedSurfaceRule, ...] = HARDCODED_SURFACE_RULES,
    required_consumer_surface_ids: frozenset[str] | None = None,
    role_fact_rules: tuple[RoleFactSurfaceRule, ...] | None = None,
) -> list[str]:
    """Validate scanner-rule ownership metadata for stack-change reviews.

    The bijection covers BOTH rule families: a value-staleness rule without an
    owner is as unaccountable as a name-matching one.
    """
    if role_fact_rules is None:
        role_fact_rules = (
            ROLE_FACT_SURFACE_RULES if rules == HARDCODED_SURFACE_RULES else ()
        )
    manifest, errors = load_surface_manifest(path)
    if errors:
        return errors
    if manifest is None:
        return [f"missing hardcoded-surface ownership manifest: {path}"]
    if manifest.get("version") != 1:
        errors.append("hardcoded-surface ownership manifest version must be 1")
    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, list):
        return errors + ["hardcoded-surface ownership manifest has no list-valued 'surfaces'"]

    rules_by_id: dict[str, HardcodedSurfaceRule | RoleFactSurfaceRule] = {
        rule.rule_id: rule for rule in rules
    }
    for role_fact_rule in role_fact_rules:
        rules_by_id[role_fact_rule.rule_id] = role_fact_rule
    seen: dict[str, int] = {}
    required_text_fields = (
        "rule_id",
        "category",
        "owner",
        "consumer_scope",
        "review_cadence",
        "evidence_command",
        "drift_response",
    )
    for index, raw_surface in enumerate(surfaces, start=1):
        prefix = f"surface manifest entry #{index}"
        if not isinstance(raw_surface, dict):
            errors.append(f"{prefix} is not a mapping")
            continue
        for field in required_text_fields:
            value = raw_surface.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix} missing non-empty {field!r}")
        rule_id = raw_surface.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            continue
        rule_id = rule_id.strip()
        if rule_id in seen:
            errors.append(
                f"surface manifest rule_id {rule_id!r} is duplicated "
                f"(entries {seen[rule_id]} and {index})"
            )
        seen[rule_id] = index
        rule = rules_by_id.get(rule_id)
        if rule is None:
            errors.append(f"surface manifest entry {rule_id!r} has no scanner rule")
            continue
        category = raw_surface.get("category")
        if isinstance(category, str) and category.strip() != rule.category:
            errors.append(
                f"surface manifest entry {rule_id!r} category {category!r} "
                f"does not match scanner category {rule.category!r}"
            )
        promotion_blocker = raw_surface.get("promotion_blocker")
        expected_blocker = rule.category == "production_blocker"
        if not isinstance(promotion_blocker, bool):
            errors.append(
                f"surface manifest entry {rule_id!r} missing boolean 'promotion_blocker'"
            )
        elif promotion_blocker != expected_blocker:
            errors.append(
                f"surface manifest entry {rule_id!r} promotion_blocker={promotion_blocker!r} "
                f"does not match category policy {expected_blocker!r}"
            )

    missing = sorted(set(rules_by_id) - set(seen))
    if missing:
        errors.append(
            "hardcoded-surface ownership manifest missing rule_id(s): "
            + ", ".join(missing)
        )
    if required_consumer_surface_ids is None:
        required_consumer_surface_ids = (
            REQUIRED_CONSUMER_SURFACE_IDS
            if rules == HARDCODED_SURFACE_RULES
            else frozenset()
        )
    errors.extend(
        _validate_consumer_surface_manifest(
            manifest,
            required_consumer_surface_ids=required_consumer_surface_ids,
        )
    )
    return errors


def _validate_consumer_surface_manifest(
    manifest: dict[str, Any],
    *,
    required_consumer_surface_ids: frozenset[str],
) -> list[str]:
    if not required_consumer_surface_ids:
        return []
    errors: list[str] = []
    raw_surfaces = manifest.get("consumer_surfaces")
    if not isinstance(raw_surfaces, list):
        return [
            "model-specific consumer surface manifest has no list-valued "
            "'consumer_surfaces'"
        ]

    seen: dict[str, int] = {}
    for index, raw_surface in enumerate(raw_surfaces, start=1):
        prefix = f"consumer surface manifest entry #{index}"
        if not isinstance(raw_surface, dict):
            errors.append(f"{prefix} is not a mapping")
            continue
        for field in CONSUMER_SURFACE_TEXT_FIELDS:
            value = raw_surface.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix} missing non-empty {field!r}")
        surface_id = raw_surface.get("surface_id")
        if not isinstance(surface_id, str) or not surface_id.strip():
            continue
        surface_id = surface_id.strip()
        if surface_id in seen:
            errors.append(
                f"consumer surface manifest surface_id {surface_id!r} is duplicated "
                f"(entries {seen[surface_id]} and {index})"
            )
        seen[surface_id] = index
        classification = raw_surface.get("classification")
        if (
            isinstance(classification, str)
            and classification.strip() not in CONSUMER_SURFACE_CLASSIFICATIONS
        ):
            errors.append(
                f"consumer surface manifest entry {surface_id!r} classification "
                f"{classification!r} is not one of "
                f"{sorted(CONSUMER_SURFACE_CLASSIFICATIONS)}"
            )
        promotion_blocker = raw_surface.get("promotion_blocker")
        if not isinstance(promotion_blocker, bool):
            errors.append(
                f"consumer surface manifest entry {surface_id!r} missing boolean "
                "'promotion_blocker'"
            )
        implementation_refs = raw_surface.get("implementation_refs")
        if (
            not isinstance(implementation_refs, list)
            or not implementation_refs
            or not all(isinstance(ref, str) and ref.strip() for ref in implementation_refs)
        ):
            errors.append(
                f"consumer surface manifest entry {surface_id!r} missing non-empty "
                "string list 'implementation_refs'"
            )

    missing = sorted(required_consumer_surface_ids - set(seen))
    if missing:
        errors.append(
            "model-specific consumer surface manifest missing surface_id(s): "
            + ", ".join(missing)
        )
    return errors


def hardcoded_surface_warning_counts(warnings: Iterable[str]) -> dict[str, int]:
    """Return unique hardcoded-surface warning counts by category bucket."""
    counts: Counter[str] = Counter()
    for warning in sorted(set(warnings)):
        bucket = _hardcoded_surface_warning_bucket(warning)
        if bucket is not None:
            counts[bucket] += 1
    return dict(counts)


def _hardcoded_surface_warning_bucket(warning: str) -> str | None:
    prefix = "hardcoded_surface."
    if not warning.startswith(prefix):
        return None
    suffix = warning[len(prefix):]
    parts = suffix.split(".", 3)
    if not parts:
        return None
    if parts[0] == "waived" and len(parts) >= 2:
        return f"waived_{parts[1]}"
    return parts[0]


def _warning_summary_lines(warnings: list[str]) -> list[str]:
    unique_warnings = sorted(set(warnings))
    lines = [
        f"WARN: {len(unique_warnings)} unique stack-prior warning(s) "
        f"({len(warnings)} total)"
    ]
    surface_counts = hardcoded_surface_warning_counts(warnings)
    if surface_counts:
        ordered_keys = [key for key in SURFACE_WARNING_ORDER if key in surface_counts]
        ordered_keys.extend(sorted(set(surface_counts) - set(ordered_keys)))
        lines.append(
            "surface_warnings: "
            + ", ".join(f"{key}={surface_counts[key]}" for key in ordered_keys)
        )
    non_surface_count = sum(
        1 for warning in unique_warnings if _hardcoded_surface_warning_bucket(warning) is None
    )
    if non_surface_count:
        lines.append(f"other_warnings: {non_surface_count}")
    return lines


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return loaded


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(priors_path: Path, source: dict[str, Any]) -> Path | None:
    raw = source.get("path")
    if not isinstance(raw, str) or not raw:
        return None
    # A relative pin is relative to the checkout running this guard — the same
    # root the compiler relativised against (model_descriptors.
    # portable_source_path). Resolving it anywhere else would re-create the
    # defect that fix removed: a worktree's gate hashing another tree's files.
    # (`priors_path` stays in the signature for callers; it is no longer the anchor.)
    from src.registry.model_descriptors import resolve_portable_source_path

    return resolve_portable_source_path(raw, REPO_ROOT)


def _port_from_endpoint(endpoint: Any) -> int | None:
    if not isinstance(endpoint, str):
        return None
    match = re.search(r":(\d+)(?:/|$)", endpoint)
    if not match:
        return None
    return int(match.group(1))


def _load_yaml_mapping_or_error(path: Path, label: str) -> tuple[dict[str, Any], list[str]]:
    """Load a launch-view input, reporting unreadability as an ERROR.

    Replaces a ``try: ... except Exception: return {}`` helper. That version
    handed the launch view an empty mapping whenever the registry or descriptor
    file was missing or malformed, and the view built from it looked ordinary:
    measured 2026-07-31 on a deliberately corrupted registry, alias_host
    coverage went 6 -> 0 and launch_requirements 13 -> 7, while not one error
    named the file that had failed to parse.

    The degraded mapping is still returned so the remaining invariants keep
    running — one broken input must not blind every other check — but the
    failure is recorded, the same contract as ``_retired_live_roles_or_error``.
    """
    try:
        return _load_yaml(path), []
    except FileNotFoundError:
        return {}, [f"{COULD_NOT_CHECK}: launch-view {label} is missing: {path}"]
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {}, [f"{COULD_NOT_CHECK}: launch-view {label} is unreadable ({path}): {exc}"]


def _descriptor_by_role(descriptors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_role: dict[str, dict[str, Any]] = {}
    models = descriptors.get("models")
    if not isinstance(models, list):
        return by_role
    for descriptor in models:
        if not isinstance(descriptor, dict):
            continue
        role_bindings = descriptor.get("role_bindings")
        if not isinstance(role_bindings, dict):
            continue
        roles = role_bindings.get("roles")
        if not isinstance(roles, list):
            continue
        for role in roles:
            if isinstance(role, str):
                by_role[role] = descriptor
    return by_role


def _server_cfg_for_role(role: str, server_mode: dict[str, Any]) -> dict[str, Any] | None:
    direct = server_mode.get(role)
    if isinstance(direct, dict):
        return direct
    for cfg in server_mode.values():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("model_role") == role:
            return cfg
        shared_with = cfg.get("shared_with")
        if isinstance(shared_with, list) and role in shared_with:
            return cfg
    return None


def _launch_cfg_from_target(target: dict[str, Any]) -> dict[str, Any]:
    entries = target.get("launch_entries")
    launch_entries = entries if isinstance(entries, list) else []
    return {
        "effective_context_tokens": target.get("effective_context_tokens"),
        # Port -> shape class over the role's whole serving fleet, so this
        # independent recomputation resolves per-instance `-np` the same way the
        # compiler does. Without it an ALIAS would resolve only the instances it
        # is tagged onto and report a false mismatch on the rest.
        "port_shape_classes": target.get("port_shape_classes") or {},
        "launch": {
            "entries": launch_entries,
            "primary_roles": sorted(
                {
                    str(entry["primary_role"])
                    for entry in launch_entries
                    if isinstance(entry, dict) and isinstance(entry.get("primary_role"), str)
                }
            ),
            "modes": sorted(
                {
                    str(entry["mode"])
                    for entry in launch_entries
                    if isinstance(entry, dict) and isinstance(entry.get("mode"), str)
                }
            ),
            "requirements": target.get("launch_requirements")
            if isinstance(target.get("launch_requirements"), dict)
            else {},
        },
    }


def _realized_launch_numa_mode() -> str | None:
    """Realized-fleet NUMA mode for the launch view (module-level test seam)."""
    try:
        from scripts.server.realized_fleet import derive_realized_numa_mode

        return derive_realized_numa_mode()
    except Exception:
        return None


def _standalone_numa_mode(explicit_mode: str | None) -> tuple[str | None, str | None]:
    """Resolve the standalone guard's declared lineup without consulting the fleet or env."""
    if explicit_mode is not None:
        mode = explicit_mode.strip().lower()
        if mode not in STANDALONE_NUMA_MODES:
            return None, (
                f"{COULD_NOT_CHECK}: invalid --numa-mode {explicit_mode!r}; expected one of "
                f"{sorted(STANDALONE_NUMA_MODES)}"
            )
        return mode, None

    topology_path = REPO_ROOT / "orchestration" / "stack_topology.yaml"
    try:
        topology = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"{COULD_NOT_CHECK}: declared topology is missing: {topology_path}"
    except OSError as exc:
        return None, (
            f"{COULD_NOT_CHECK}: declared topology is unreadable ({topology_path}): {exc}"
        )
    except UnicodeError as exc:
        return None, (
            f"{COULD_NOT_CHECK}: declared topology is malformed ({topology_path}): {exc}"
        )
    except yaml.YAMLError as exc:
        return None, f"{COULD_NOT_CHECK}: declared topology is malformed ({topology_path}): {exc}"

    if not isinstance(topology, dict):
        return None, f"{COULD_NOT_CHECK}: declared topology is not a mapping ({topology_path})"
    mode = topology.get("numa_mode")
    if not isinstance(mode, str) or mode.strip() not in STANDALONE_NUMA_MODES:
        return None, (
            f"{COULD_NOT_CHECK}: declared topology has no valid numa_mode ({topology_path}); "
            f"expected one of {sorted(STANDALONE_NUMA_MODES)}"
        )
    return mode.strip(), None


def _launch_manifest_targets(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    launch_numa_mode: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return live launch ports/tier per role from the computed manifest.

    Convenience wrapper for readers that only want the view. Anything that
    GATES on the view must call `_launch_manifest_targets_or_error`: an empty
    return here means "could not evaluate" exactly as often as it means
    "nothing to report", and the two must not be collapsed at a gate.
    """
    targets, _errors = _launch_manifest_targets_or_error(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        launch_numa_mode=launch_numa_mode,
    )
    return targets


def _launch_manifest_targets_or_error(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    launch_numa_mode: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Build the launch view, reporting every input it could not evaluate.

    The predecessor returned a bare `{}` when `scripts.server.stack_manifest`
    failed to import, which is the whole launch view. Measured 2026-07-31 with
    the import blocked: launch targets 22 -> 0 and launch-alignment errors
    12 -> 0, i.e. the promotion gate went CLEAN because its instrument had
    broken. A byte-hash check over the sources does not cover this — during an
    import failure the file is byte-identical and unimportable at the same time.
    """
    errors: list[str] = []
    try:
        from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS, _filter_by_numa_mode
    except Exception as exc:  # noqa: BLE001 — report the failure, never degrade to "clean"
        return {}, [
            f"{COULD_NOT_CHECK}: launch manifest view unavailable — "
            f"scripts.server.stack_manifest did not import "
            f"({type(exc).__name__}: {exc}); EVERY launch/serving alignment "
            "assertion was skipped"
        ]

    from scripts.server.stack_numa_mode import (
        VALID_STACK_NUMA_MODES,
        env_stack_numa_mode,
    )

    if launch_numa_mode is not None:
        # NIB2-69 (2026-09-15): an EXPLICIT mode wins and nothing ambient is
        # consulted. The stack-change pipeline passes the DECLARED topology mode
        # (orchestration/stack_topology.yaml) — the same mode the priors were
        # compiled with — so the launch view and the artifact under test describe
        # the same lineup. Without it a clean shell fell through to env default
        # "full", filtered the half instances out of the view, and reported every
        # half port the (correctly `both`-compiled) priors carry as a non-launch
        # port: the 13-errors-x-3-guard-steps = 39 class on origin/main.
        numa_mode = str(launch_numa_mode).strip().lower()
        if numa_mode not in VALID_STACK_NUMA_MODES:
            return {}, [
                f"{COULD_NOT_CHECK}: invalid launch NUMA mode {launch_numa_mode!r} "
                f"(expected one of {sorted(VALID_STACK_NUMA_MODES)}); EVERY "
                "launch/serving alignment assertion was skipped"
            ]
    else:
        # Legacy resolution for callers that pass no mode (standalone guard CLI,
        # fixtures). ESC-8/WP-13: build the launch view against the REALIZED
        # fleet mode, not the ambient env default ("full" in a clean shell). A
        # quarter-realized fleet guarded against a full-mode launch view
        # mismatches wholesale (the 105-error class, 2026-07-22). Env stays the
        # fallback for fleet-less environments (tests, cold hosts).
        numa_mode = _realized_launch_numa_mode()
        if numa_mode is None:
            numa_mode = env_stack_numa_mode()
    registry, registry_errors = _load_yaml_mapping_or_error(registry_path, "registry")
    errors.extend(registry_errors)
    registry_roles = registry.get("roles") if isinstance(registry.get("roles"), dict) else {}
    server_mode = (
        registry.get("server_mode") if isinstance(registry.get("server_mode"), dict) else {}
    )
    descriptors, descriptor_errors = _load_yaml_mapping_or_error(descriptor_path, "descriptors")
    errors.extend(descriptor_errors)
    descriptor_roles = _descriptor_by_role(descriptors)

    # One report per cause, not one per server: the failure is an instrument
    # fact, and 23 copies of it would bury the invariant errors underneath.
    requirements_unavailable: str | None = None
    context_unavailable: str | None = None

    targets: dict[str, dict[str, Any]] = {}
    for tier, servers in (
        ("hot", _filter_by_numa_mode(HOT_SERVERS, numa_mode)),
        ("warm", _filter_by_numa_mode(WARM_SERVERS, numa_mode)),
    ):
        for server in servers:
            if not isinstance(server, dict):
                continue
            port = server.get("port")
            if not isinstance(port, int):
                continue
            try:
                server_context = _effective_context_for_server(server)
            except LaunchViewUnavailableError as exc:
                server_context = None
                context_unavailable = context_unavailable or str(exc)
            try:
                server_requirements = _launch_requirements_for_server(server)
            except LaunchViewUnavailableError as exc:
                server_requirements = {}
                requirements_unavailable = requirements_unavailable or str(exc)
            for role in server.get("roles") or []:
                if isinstance(role, str):
                    target = targets.setdefault(
                        role,
                        {
                            "port": port,
                            "ports": [],
                            "tier": tier,
                            "effective_context_tokens": server_context,
                            "launch_entries": [],
                            "launch_requirements": {},
                            "port_shape_classes": {},
                        },
                    )
                    target["ports"].append(port)
                    entry = _launch_entry_for_role(server, role)
                    target["launch_entries"].append(entry)
                    entry_shape_class = entry.get("cpu_shape_class")
                    if isinstance(entry_shape_class, str) and entry_shape_class:
                        target["port_shape_classes"][port] = entry_shape_class
                    target["launch_requirements"].update(server_requirements)
    # WP-13: attach the declarative alias→host relation from server_mode
    # shared_with. evidence.alias_overrides only exists for MODEL-conflicted
    # aliases (worker_math's ghost binding); same-model aliases
    # (coder_escalation/worker_summarize on frontdoor's fleet) are declared
    # ONLY here. Host key = the row's model_role when it is a launch target,
    # else the server_mode key itself.
    for server_key, cfg in server_mode.items():
        if not isinstance(cfg, dict):
            continue
        shared = cfg.get("shared_with")
        if not isinstance(shared, list):
            continue
        model_role = cfg.get("model_role")
        host_key = (
            str(model_role)
            if isinstance(model_role, str) and model_role in targets
            else str(server_key)
            if str(server_key) in targets
            else None
        )
        if host_key is None:
            continue
        for alias in shared:
            if isinstance(alias, str) and alias in targets and alias != host_key:
                targets[alias]["alias_host"] = host_key
                # WP-13 fleet convergence, extended to the shape map: an alias
                # rides its host's WHOLE fleet even though it is tagged onto only
                # the first N instances. The compiled record's alias `ports` are
                # already the host's; its `slots_by_port` must span the same set,
                # so the alias inherits the host's port -> shape-class map here.
                host_classes = targets[host_key].get("port_shape_classes")
                if isinstance(host_classes, dict):
                    alias_classes = targets[alias].setdefault("port_shape_classes", {})
                    for alias_port, alias_shape in host_classes.items():
                        alias_classes.setdefault(alias_port, alias_shape)

    for role, target in targets.items():
        descriptor = descriptor_roles.get(role) or {}
        role_cfg = registry_roles.get(role) if isinstance(registry_roles.get(role), dict) else None
        server_cfg = _server_cfg_for_role(role, server_mode)
        target["launch_requirements"].update(
            _server_mode_launch_requirement_overrides(role, server_cfg, role_cfg)
        )
        target["launch_runtime"] = _launch_runtime_record(
            role,
            descriptor,
            server_cfg,
            role_cfg,
            _launch_cfg_from_target(target),
        )

    # The target COUNT is unchanged when these fail, which is what made the
    # 2026-07-31 reproduction invisible: 22 targets reported, 0 of 22 context
    # assertions actually evaluated. Report the skip explicitly.
    if context_unavailable:
        errors.append(
            f"{COULD_NOT_CHECK}: launch context unavailable for all {len(targets)} "
            f"launch target(s) — {context_unavailable}; every "
            "serving.effective_context_tokens assertion was skipped"
        )
    if requirements_unavailable:
        errors.append(
            f"{COULD_NOT_CHECK}: launch requirements unavailable for all {len(targets)} "
            f"launch target(s) — {requirements_unavailable}; every "
            "serving.launch.requirements assertion was skipped"
        )
    return targets, errors


def _launch_mode_for_server(server: dict[str, Any]) -> str:
    if server.get("worker_pool"):
        return "worker_pool"
    if server.get("vision"):
        return "vision"
    if server.get("embedding"):
        return "embedding"
    return "default"


def _launch_entry_for_role(server: dict[str, Any], role: str) -> dict[str, Any]:
    roles = server.get("roles")
    primary_role = roles[0] if isinstance(roles, list) and roles and isinstance(roles[0], str) else role
    entry: dict[str, Any] = {
        "port": server["port"],
        "primary_role": primary_role,
        "mode": _launch_mode_for_server(server),
        "alias": role != primary_role,
    }
    numa_instance = server.get("numa_instance")
    if isinstance(numa_instance, int):
        entry["numa_instance"] = numa_instance
    # 2026-08-02: the instance's SHAPE CLASS, without which the independently
    # recomputed runtime record below cannot resolve per-instance `-np` and would
    # fall back to the role-level count for every port — reporting a mismatch
    # against the compiled artifact that is really a gap in this recomputation.
    # This is the guard's own second copy of `_launch_entry_for_role` (the
    # compiler's lives in src/registry/stack_priors.py); the duplication is
    # pre-existing and both copies must carry the field.
    try:
        from scripts.server.stack_numa import instance_shape_class

        shape_class = instance_shape_class(str(primary_role), numa_instance or 0)
    except Exception:  # noqa: BLE001 — the guard must report, not abort on import
        shape_class = None
    if isinstance(shape_class, str) and shape_class:
        entry["cpu_shape_class"] = shape_class
    worker_type = server.get("worker_type")
    if isinstance(worker_type, str):
        entry["worker_type"] = worker_type
    vision_type = server.get("vision_type")
    if isinstance(vision_type, str):
        entry["vision_type"] = vision_type
    return entry


def _launch_requirements_for_server(server: dict[str, Any]) -> dict[str, str]:
    try:
        from scripts.server.stack_manifest import (
            EXPLORE_DRAFT_MODEL,
            VISION_ESCALATION_MMPROJ,
            VISION_ESCALATION_MODEL,
            VISION_WORKER_MMPROJ,
            VISION_WORKER_MODEL,
            WORKER_POOL_MODELS,
        )
    except Exception as exc:  # noqa: BLE001 — raise, never return "no requirements"
        # Returning `{}` here made every model-path comparison vanish while the
        # guard still reported the same number of launch targets: measured
        # 2026-07-31, poisoned model paths detected 10/10 -> 8/10 with no
        # indication that two roles had stopped being checked.
        raise LaunchViewUnavailableError(
            f"scripts.server.stack_manifest launch-path constants did not import "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    requirements: dict[str, str] = {}
    mode = _launch_mode_for_server(server)
    if mode == "worker_pool":
        worker_type = str(server.get("worker_type") or "")
        model_path = WORKER_POOL_MODELS.get(worker_type)
        if model_path:
            requirements["model_path"] = str(model_path)
        if worker_type == "explore" and EXPLORE_DRAFT_MODEL:
            requirements["draft_model_path"] = str(EXPLORE_DRAFT_MODEL)
    elif mode == "vision":
        vision_type = server.get("vision_type")
        if vision_type == "worker":
            requirements["model_path"] = str(VISION_WORKER_MODEL)
            requirements["mmproj_path"] = str(VISION_WORKER_MMPROJ)
        elif vision_type == "escalation":
            requirements["model_path"] = str(VISION_ESCALATION_MODEL)
            requirements["mmproj_path"] = str(VISION_ESCALATION_MMPROJ)
    return {key: value for key, value in sorted(requirements.items()) if value}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _effective_context_for_server(server: dict[str, Any]) -> int | None:
    try:
        from scripts.server.stack_manifest import (
            DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
            LAUNCH_CONTEXT_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 — raise, never return "no context"
        # `None` is also the legitimate "this server declares no primary role"
        # answer, so returning it on an import failure hid the failure inside a
        # normal-looking result: 0 of 22 roles context-checked, target count
        # unchanged, guard green (measured 2026-07-31).
        raise LaunchViewUnavailableError(
            f"scripts.server.stack_manifest context constants did not import "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    roles = server.get("roles")
    role = roles[0] if isinstance(roles, list) and roles and isinstance(roles[0], str) else None
    if role:
        return _positive_int(LAUNCH_CONTEXT_TOKENS.get(role, DEFAULT_EFFECTIVE_CONTEXT_TOKENS))
    return None


def _normalized_launch_entries(raw_entries: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_entries, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        entry: dict[str, Any] = {}
        for field in (
            "port",
            "primary_role",
            "mode",
            "alias",
            "numa_instance",
            "worker_type",
            "vision_type",
        ):
            if field in raw_entry:
                entry[field] = raw_entry[field]
        entries.append(entry)
    return sorted(
        entries,
        key=lambda entry: (
            entry.get("port", -1),
            str(entry.get("primary_role", "")),
            str(entry.get("mode", "")),
        ),
    )


def _normalized_launch_requirements(raw_requirements: Any) -> dict[str, str]:
    if not isinstance(raw_requirements, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw_requirements.items():
        if isinstance(key, str) and value not in (None, ""):
            normalized[key] = str(value)
    return dict(sorted(normalized.items()))


def _normalized_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalized_jsonish(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalized_jsonish(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalized_launch_runtime(raw_runtime: Any) -> dict[str, Any]:
    if not isinstance(raw_runtime, dict):
        return {}
    normalized = _normalized_jsonish(raw_runtime)
    return normalized if isinstance(normalized, dict) else {}


def _launch_target_modes(target: dict[str, Any]) -> set[str]:
    entries = target.get("launch_entries")
    if not isinstance(entries, list):
        return set()
    return {
        str(entry["mode"])
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("mode"), str)
    }


def _live_prior_roles(roles: dict[str, Any]) -> set[str]:
    return {
        role
        for role, record in roles.items()
        if isinstance(role, str)
        and isinstance(record, dict)
        and record.get("deployment_status") == "live_stack"
    }


def _launch_target_is_covered_alias(
    role: str,
    target: dict[str, Any],
    live_roles: set[str],
) -> bool:
    entries = target.get("launch_entries")
    if not isinstance(entries, list) or not entries:
        return False
    primary_roles: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("alias") is not True:
            return False
        primary_role = entry.get("primary_role")
        if not isinstance(primary_role, str):
            return False
        primary_roles.add(primary_role)
    return bool(primary_roles) and role not in live_roles and primary_roles <= live_roles


def _launch_target_is_manifest_owned_auxiliary(role: str, target: dict[str, Any]) -> bool:
    modes = _launch_target_modes(target)
    if modes and modes <= MANIFEST_OWNED_AUXILIARY_LAUNCH_MODES:
        return True
    if role in MANIFEST_OWNED_DEFAULT_AUXILIARY_LAUNCH_ROLES:
        return bool(modes) and target.get("tier") == "warm" and modes <= {"default"}
    if role in MANIFEST_OWNED_AUXILIARY_LAUNCH_ROLES:
        return bool(modes) and target.get("tier") == "warm" and modes <= {"worker_pool"}
    return False


def _alias_host_role(record: dict[str, Any]) -> str | None:
    """Host role an alias record rides (evidence.alias_overrides[].served_by).

    WP-13: an alias role has no llama-server of its own — its serving view is
    the HOST's full fleet, so serving-vs-launch alignment must be judged
    against the host's launch manifest row, not the alias's tagged subset.
    Returns None unless every alias_override names the same host.
    """
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        return None
    overrides = evidence.get("alias_overrides")
    if not isinstance(overrides, list):
        return None
    hosts = {
        str(override.get("served_by"))
        for override in overrides
        if isinstance(override, dict) and override.get("served_by")
    }
    if len(hosts) != 1:
        return None
    host = hosts.pop()
    # Descriptors are model-keyed, so the HOST's own record carries the same
    # alias_overrides copy — a record whose resolved host is itself IS the
    # host and must keep full (runtime-inclusive) validation.
    if host == str(record.get("role") or ""):
        return None
    return host


def _manifest_alias_host(target: dict[str, Any]) -> str | None:
    """Return the manifest primary for a pure alias launch target.

    Registry ``server_mode.shared_with`` is useful corroboration but is not the
    source of the launch relationship.  The computed manifest's alias entries
    remain sufficient to align an alias serving fleet with its primary when a
    registry compilation has omitted that convenience metadata.
    """
    entries = target.get("launch_entries")
    if not isinstance(entries, list) or not entries:
        return None
    hosts = {
        entry.get("primary_role")
        for entry in entries
        if isinstance(entry, dict) and entry.get("alias") is True
    }
    if len(hosts) != 1 or not all(isinstance(host, str) and host for host in hosts):
        return None
    if not all(isinstance(entry, dict) and entry.get("alias") is True for entry in entries):
        return None
    return next(iter(hosts))


def validate_launch_manifest_serving_alignment(
    priors: dict[str, Any],
    *,
    launch_manifest_targets: dict[str, dict[str, Any]] | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    launch_numa_mode: str | None = None,
) -> list[str]:
    """Validate generated live serving records against current launch roles.

    ``launch_numa_mode`` selects the lineup the launch view is filtered to. Pass
    the mode the priors were compiled for; ``None`` keeps the legacy
    realized-fleet -> ambient-env resolution.
    """
    errors: list[str] = []
    if launch_manifest_targets is None:
        targets, view_errors = _launch_manifest_targets_or_error(
            registry_path=registry_path,
            descriptor_path=descriptor_path,
            launch_numa_mode=launch_numa_mode,
        )
        errors.extend(view_errors)
        derived_view = True
    else:
        targets = launch_manifest_targets
        # An injected view is a caller-authored stub, not the producer's own
        # launch view, so it cannot be held to the producer's checklist below —
        # a fixture that omits a field omitted it on purpose. Production never
        # injects: `main()` and `stack_change_pipeline` both leave this None, so
        # the derived path is THE consumer and keeps the full checklist.
        derived_view = False

    if not targets:
        # NOT `return []`. An empty launch view is this guard's own instrument
        # failing, and reporting "no alignment errors" on it takes the promotion
        # gate clean on precisely the condition the gate exists to catch
        # (measured 2026-07-31: targets 22 -> 0 took alignment errors 12 -> 0).
        errors.append(
            f"{COULD_NOT_CHECK}: launch manifest produced 0 launch targets, so no "
            "launch/serving alignment could be evaluated at all"
        )
        return errors

    roles = priors.get("roles")
    if not isinstance(roles, dict):
        # This early return LOOKS like the one above and is not fail-open: a
        # second reader, `validate_stack_priors`, independently re-reads
        # `priors['roles']` and appends "stack priors artifact has no
        # mapping-valued roles section". Returning here is only safe because
        # that reader exists — do not delete it, and do not duplicate it here.
        return errors

    live_roles = _live_prior_roles(roles)
    full_launch_coverage_required = priors.get("coverage_scope") != "explicit_active_roles"
    for role, target in sorted(targets.items()):
        if not isinstance(role, str) or not isinstance(target, dict):
            continue
        record = roles.get(role)
        if record is None and not full_launch_coverage_required:
            continue
        if isinstance(record, dict) and record.get("deployment_status") == "live_stack":
            continue
        if _launch_target_is_covered_alias(role, target, live_roles):
            continue
        if _launch_target_is_manifest_owned_auxiliary(role, target):
            continue
        if record is None:
            errors.append(
                f"launch manifest target {role!r} has no generated stack-prior role record"
            )
        elif isinstance(record, dict):
            errors.append(
                f"launch manifest target {role!r} has deployment_status "
                f"{record.get('deployment_status')!r}, expected 'live_stack'"
            )
        else:
            errors.append(
                f"launch manifest target {role!r} has non-mapping generated stack-prior role record"
            )

    for role, record in sorted(roles.items()):
        if not isinstance(role, str) or not isinstance(record, dict):
            continue
        if record.get("deployment_status") != "live_stack":
            continue
        target = targets.get(role)
        if target is None:
            errors.append(f"live role {role!r} is absent from current launch manifest")
            continue
        # WP-13: alias roles serve the HOST's fleet — align against the host's
        # launch row (superset of the alias's tagged subset), not its own.
        # Host resolution: model-conflict evidence first (alias_overrides),
        # else the declarative server_mode shared_with relation.
        host_role = (
            _alias_host_role(record)
            or target.get("alias_host")
            or _manifest_alias_host(target)
        )
        if host_role and isinstance(targets.get(host_role), dict):
            target = targets[host_role]
        serving = record.get("serving")
        if not isinstance(serving, dict):
            continue
        target_port = target.get("port")
        target_tier = target.get("tier")
        target_context = target.get("effective_context_tokens")
        raw_target_ports = target.get("ports")
        target_ports = (
            {port for port in raw_target_ports if isinstance(port, int)}
            if isinstance(raw_target_ports, list)
            else set()
        )
        target_launch_entries = _normalized_launch_entries(target.get("launch_entries"))
        target_launch_requirements = _normalized_launch_requirements(
            target.get("launch_requirements")
        )
        target_launch_runtime = _normalized_launch_runtime(target.get("launch_runtime"))

        # Derive the checklist from the PRODUCER, not from whatever the launch
        # view managed to compute. Every comparison below is guarded by "did the
        # launch view produce a value?", so any cause of that value going
        # missing — a failed import, a refactor that stops populating the field,
        # a test seam left in place — silently deletes the assertion instead of
        # failing it. The compiled priors are the producer here: if a role
        # DECLARES the fact, the fact is checkable, and a launch view that
        # cannot supply the counterpart is a COULD-NOT-CHECK, not a pass.
        declared_launch = serving.get("launch") if isinstance(serving.get("launch"), dict) else {}
        declared_requirements = _normalized_launch_requirements(
            declared_launch.get("requirements")
        )
        if derived_view and declared_requirements and not target_launch_requirements:
            errors.append(
                f"{COULD_NOT_CHECK}: role {role!r} declares serving.launch.requirements "
                f"{sorted(declared_requirements)} but the launch view produced none; "
                "the requirement comparison was skipped, not passed"
            )
        declared_context = serving.get("effective_context_tokens")
        if derived_view and isinstance(declared_context, int) and not isinstance(target_context, int):
            errors.append(
                f"{COULD_NOT_CHECK}: role {role!r} declares serving.effective_context_tokens "
                f"{declared_context} but the launch view produced none; the context "
                "comparison was skipped, not passed"
            )

        endpoint_port = _port_from_endpoint(serving.get("endpoint"))
        ports = serving.get("ports")
        port_set = {port for port in ports if isinstance(port, int)} if isinstance(ports, list) else set()
        if isinstance(target_port, int):
            if endpoint_port != target_port:
                errors.append(
                    f"role {role!r} serving.endpoint port {endpoint_port!r} "
                    f"does not match launch manifest port {target_port}"
                )
            if target_port not in port_set:
                errors.append(
                    f"role {role!r} serving.ports {sorted(port_set)} "
                    f"does not include launch manifest port {target_port}"
                )
        if target_ports:
            missing_ports = sorted(target_ports - port_set)
            extra_ports = sorted(port_set - target_ports)
            if missing_ports:
                errors.append(
                    f"role {role!r} serving.ports {sorted(port_set)} "
                    f"missing launch manifest port(s) {missing_ports}"
                )
            if extra_ports:
                errors.append(
                    f"role {role!r} serving.ports {sorted(port_set)} "
                    f"include non-launch port(s) {extra_ports}"
                )
        if isinstance(target_tier, str) and serving.get("tier") != target_tier:
            errors.append(
                f"role {role!r} serving.tier {serving.get('tier')!r} "
                f"does not match launch manifest tier {target_tier!r}"
            )
        if (
            isinstance(target_context, int)
            and serving.get("effective_context_tokens") != target_context
        ):
            errors.append(
                f"role {role!r} serving.effective_context_tokens "
                f"{serving.get('effective_context_tokens')!r} does not match "
                f"launch context {target_context}"
            )
        if target_launch_entries:
            launch = serving.get("launch")
            actual_entries = (
                _normalized_launch_entries(launch.get("entries"))
                if isinstance(launch, dict)
                else []
            )
            if host_role:
                # WP-13: an alias's recorded launch entries are the tagged
                # subset of the host fleet it rides, carrying alias-specific
                # tagging fields — require PORT containment in the host's
                # entries (dict equality can never hold across the tagging).
                host_entry_ports = {
                    entry.get("port")
                    for entry in target_launch_entries
                    if isinstance(entry, dict) and isinstance(entry.get("port"), int)
                }
                alias_extra_ports = sorted(
                    entry.get("port")
                    for entry in actual_entries
                    if isinstance(entry, dict)
                    and isinstance(entry.get("port"), int)
                    and entry.get("port") not in host_entry_ports
                )
                if alias_extra_ports:
                    errors.append(
                        f"role {role!r} serving.launch.entries port(s) "
                        f"{alias_extra_ports} absent from host {host_role!r} "
                        f"launch manifest"
                    )
            elif actual_entries != target_launch_entries:
                errors.append(
                    f"role {role!r} serving.launch.entries do not match "
                    f"launch manifest entries"
                )
        if target_launch_requirements:
            launch = serving.get("launch")
            actual_requirements = (
                _normalized_launch_requirements(launch.get("requirements"))
                if isinstance(launch, dict)
                else {}
            )
            mismatches = {
                key: {
                    "expected": expected,
                    "actual": actual_requirements.get(key),
                }
                for key, expected in target_launch_requirements.items()
                if actual_requirements.get(key) != expected
            }
            if mismatches:
                errors.append(
                    f"role {role!r} serving.launch.requirements do not match "
                    f"launch manifest requirements: {json.dumps(mismatches, sort_keys=True)}"
                )
        if target_launch_runtime and not host_role:
            # WP-13: an alias record's stored runtime is a stale artifact of its
            # standalone-row past (e.g. coder_escalation acceleration:none) — it
            # RUNS under the host's runtime by construction (one process). The
            # runtime is validated once, on the host's own row.
            launch = serving.get("launch")
            actual_runtime = (
                _normalized_launch_runtime(launch.get("runtime"))
                if isinstance(launch, dict)
                else {}
            )
            if actual_runtime != target_launch_runtime:
                errors.append(
                    f"role {role!r} serving.launch.runtime does not match "
                    "launch manifest runtime: "
                    f"{json.dumps({'expected': target_launch_runtime, 'actual': actual_runtime}, sort_keys=True)}"
                )
    return errors


def _matches_any(path: Path, patterns: tuple[str, ...]) -> bool:
    rel = path.as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def _display_path(path: Path, repo_root: Path) -> Path:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


def _retired_role_alternation(roles: Iterable[str]) -> str:
    """Build a word-bounded alternation over the retired-role names."""
    names = sorted({name for name in roles if name})
    return r"\b(?:" + "|".join(re.escape(name) for name in names) + r")\b"


def _derive_retired_role_patterns(
    rules: tuple[HardcodedSurfaceRule, ...],
) -> tuple[tuple[HardcodedSurfaceRule, ...], list[str]]:
    """Replace hand-written retired-role patterns with the DERIVED set.

    2026-08-02: these rules each carried the literal ``\\barchitect_coding\\b``
    while this very module already derives the authoritative set in
    `retired_live_roles()` — master-registry retirement markers minus whatever the
    compiled lean registry still serves. The two had diverged badly: the derivation
    returns 42 retired names and the scanner looked for exactly 1, so 41 retired
    roles could sit in live code, procedure enums, source-access metadata, tests
    and operator docs and every one of these rules would report green.

    That is the incomplete-checklist shape, and it is worse here than elsewhere
    because the producer is IN THE SAME FILE. Nothing was duplicated; a name was
    simply missing from a hand-maintained pattern, and no review of what the rule
    DID match could have surfaced what it did not.

    `RETIRED_LIVE_ROLE_FLOOR` stays a FLOOR (same shape as
    `REQUIRED_SOURCE_ARTIFACTS`): the union is scanned, so a derivation that
    silently stops reporting a known-retired role still leaves the rule with teeth.
    A derivation failure is reported, never swallowed — the floor pattern is used
    for that run so one broken source does not blind the rest of the scan.
    """
    if not any(rule.derive_retired_roles for rule in rules):
        return rules, []

    retired, errors = _retired_live_roles_or_error()
    pattern = _retired_role_alternation(set(retired) | set(RETIRED_LIVE_ROLE_FLOOR))
    return (
        tuple(
            replace(rule, pattern=pattern) if rule.derive_retired_roles else rule
            for rule in rules
        ),
        errors,
    )


def _candidate_paths(repo_root: Path, rule: HardcodedSurfaceRule) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in rule.path_globs:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            rel_path = path.relative_to(repo_root)
            if _matches_any(rel_path, rule.exclude_globs):
                continue
            paths[rel_path.as_posix()] = path
    return [paths[key] for key in sorted(paths)]


def scan_hardcoded_surfaces(
    repo_root: Path = REPO_ROOT,
    *,
    rules: tuple[HardcodedSurfaceRule, ...] = HARDCODED_SURFACE_RULES,
    categories: frozenset[str] | None = None,
) -> list[SurfaceFinding]:
    """Find curated hardcoded model/stack surfaces that can drift.

    This is intentionally narrower than a repository-wide grep. The goal is to
    turn known risky model-specific surfaces into a validator signal without
    treating historical artifacts, benchmark outputs, or generated backups as
    live stack truth.
    """
    # Rules that name retired roles get their pattern DERIVED from
    # `retired_live_roles()` here, so a role retired upstream is scanned for
    # without anyone editing a regex literal.
    #
    # The derivation errors are dropped HERE and only here: this function returns
    # findings, not errors, and `validate_stack_priors` already resolves the same
    # set through `_retired_live_roles_or_error()` and records the failure as an
    # ERROR before it ever calls us. Dropping them a second time hides nothing —
    # if the derivation is broken, the guard says so on its own line.
    rules, _reported_by_validate_stack_priors = _derive_retired_role_patterns(rules)
    findings: list[SurfaceFinding] = []
    for rule in rules:
        if categories is not None and rule.category not in categories:
            continue
        compiled = re.compile(rule.pattern)
        for path in _candidate_paths(repo_root, rule):
            try:
                if path.stat().st_size > SURFACE_SCAN_MAX_FILE_BYTES:
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            rel_path = path.relative_to(repo_root)
            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                if SURFACE_SCAN_ALLOW_MARKER in line:
                    continue
                if rule.ignore_comment_lines and stripped.startswith("#"):
                    continue
                if not compiled.search(line):
                    continue
                findings.append(
                    SurfaceFinding(
                        rule_id=rule.rule_id,
                        category=rule.category,
                        path=rel_path,
                        line=line_no,
                        snippet=line.strip()[:160],
                        remediation=rule.remediation,
                    )
                )
    return findings


def _normalized_identity(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _role_key_line(lines: list[str], role: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(role)}\s*:")
    for line_no, line in enumerate(lines, start=1):
        if pattern.match(line):
            return line_no
    return 0


def _compiled_model_identities(record: dict[str, Any]) -> list[str]:
    """Every name the compiled artifact uses for this role's model."""
    identities: list[str] = []
    for key in ("display_name", "model_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            identities.append(value.strip())
    serving = record.get("serving")
    serving = serving if isinstance(serving, dict) else {}
    launch = serving.get("launch")
    launch = launch if isinstance(launch, dict) else {}
    requirements = launch.get("requirements")
    requirements = requirements if isinstance(requirements, dict) else {}
    model_path = requirements.get("model_path")
    if isinstance(model_path, str) and model_path.strip():
        identities.append(Path(model_path.strip()).stem)
    return identities


def _identity_agrees(declared: Any, identities: list[str]) -> bool:
    """Substring-tolerant model-identity comparison.

    Config surfaces routinely write an informal name ("Qwen3-Next-80B-A3B") for a
    model the compiled artifact spells out ("Qwen3-Next-80B-A3B-Instruct"), and
    treating that as drift would bury the real signal. Containment either way is
    accepted; a distinguishing token the other side does not carry at all — an
    "-MTP" build, a different parameter count, a different family — is not.
    """
    normalized = _normalized_identity(declared)
    if not normalized:
        return True
    for identity in identities:
        candidate = _normalized_identity(identity)
        if not candidate:
            continue
        if normalized in candidate or candidate in normalized:
            return True
    return False


def _compiled_role_fact(record: dict[str, Any], field: str) -> Any:
    if field == "model":
        return _compiled_model_identities(record)
    if field == "quant":
        model = record.get("model")
        model = model if isinstance(model, dict) else {}
        return model.get("quant")
    if field == "tier":
        serving = record.get("serving")
        serving = serving if isinstance(serving, dict) else {}
        return serving.get("tier")
    return None


def _role_fact_mismatch(field: str, declared: Any, compiled: Any) -> str | None:
    if declared is None or (isinstance(declared, str) and not declared.strip()):
        return None
    if field == "model":
        identities = compiled if isinstance(compiled, list) else []
        if not identities:
            return None
        if _identity_agrees(declared, identities):
            return None
        return f"model {declared!r} != compiled {identities[0]!r}"
    if compiled is None or (isinstance(compiled, str) and not compiled.strip()):
        return None
    if _normalized_identity(declared) == _normalized_identity(compiled):
        return None
    return f"{field} {declared!r} != compiled {compiled!r}"


def _declared_row_is_alias(record: dict[str, Any], rule: RoleFactSurfaceRule) -> bool:
    alias_to = record.get(rule.alias_field)
    if isinstance(alias_to, str) and alias_to.strip():
        return True
    tier = record.get("tier")
    return isinstance(tier, str) and tier.strip().lower() in rule.alias_tier_values


def scan_role_fact_surfaces(
    priors: dict[str, Any],
    repo_root: Path = REPO_ROOT,
    *,
    rules: tuple[RoleFactSurfaceRule, ...] = ROLE_FACT_SURFACE_RULES,
    categories: frozenset[str] | None = None,
) -> list[SurfaceFinding]:
    """Compare role-keyed config tables against the compiled stack priors.

    Emits ordinary ``SurfaceFinding``s so declared exceptions, waiver-staleness
    checks and the warning-bucket accounting apply unchanged.
    """
    compiled_roles = priors.get("roles")
    if not isinstance(compiled_roles, dict):
        return []
    findings: list[SurfaceFinding] = []
    for rule in rules:
        if categories is not None and rule.category not in categories:
            continue
        for path in _candidate_paths(
            repo_root,
            HardcodedSurfaceRule(
                rule_id=rule.rule_id,
                category=rule.category,
                pattern="",
                path_globs=rule.path_globs,
                exclude_globs=rule.exclude_globs,
                remediation=rule.remediation,
            ),
        ):
            try:
                if path.stat().st_size > SURFACE_SCAN_MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8")
                declared_doc = yaml.safe_load(text)
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                continue
            if not isinstance(declared_doc, dict):
                continue
            declared_roles = declared_doc.get(rule.roles_key)
            if not isinstance(declared_roles, dict):
                continue
            lines = text.splitlines()
            rel_path = path.relative_to(repo_root)
            for role in sorted(declared_roles):
                declared_record = declared_roles.get(role)
                compiled_record = compiled_roles.get(role)
                if not isinstance(declared_record, dict) or not isinstance(compiled_record, dict):
                    continue
                if _declared_row_is_alias(declared_record, rule):
                    continue
                line_no = _role_key_line(lines, role)
                if line_no and SURFACE_SCAN_ALLOW_MARKER in lines[line_no - 1]:
                    continue
                mismatches = []
                for field in rule.fields:
                    mismatch = _role_fact_mismatch(
                        field,
                        declared_record.get(field),
                        _compiled_role_fact(compiled_record, field),
                    )
                    if mismatch is not None:
                        mismatches.append(mismatch)
                if not mismatches:
                    continue
                findings.append(
                    SurfaceFinding(
                        rule_id=rule.rule_id,
                        category=rule.category,
                        path=rel_path,
                        line=line_no,
                        snippet=f"role {role!r}: " + "; ".join(mismatches),
                        remediation=rule.remediation,
                    )
                )
    return findings


def _surface_exception_from_raw(index: int, raw: Any) -> tuple[SurfaceException | None, list[str]]:
    errors: list[str] = []
    prefix = f"surface exception #{index}"
    if not isinstance(raw, dict):
        return None, [f"{prefix} is not a mapping"]

    def required_str(field: str) -> str:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix} missing non-empty {field!r}")
            return ""
        return value.strip()

    path_glob = raw.get("path_glob", raw.get("path"))
    if not isinstance(path_glob, str) or not path_glob.strip():
        errors.append(f"{prefix} missing non-empty 'path_glob' or 'path'")
        path_glob = ""
    else:
        path_glob = path_glob.strip()

    line_raw = raw.get("line")
    line: int | None = None
    if line_raw is not None:
        if not isinstance(line_raw, int) or line_raw <= 0:
            errors.append(f"{prefix} line must be a positive integer when present")
        else:
            line = line_raw

    classification = required_str("classification")
    if classification and classification not in SURFACE_EXCEPTION_CLASSIFICATIONS:
        allowed = ", ".join(sorted(SURFACE_EXCEPTION_CLASSIFICATIONS))
        errors.append(f"{prefix} classification {classification!r} is not one of: {allowed}")

    expires = required_str("expires")
    if expires:
        try:
            expires_date = date.fromisoformat(expires)
        except ValueError:
            errors.append(f"{prefix} expires must be an ISO date YYYY-MM-DD")
        else:
            if expires_date < date.today():
                errors.append(f"{prefix} expired on {expires}")

    exception = SurfaceException(
        rule_id=required_str("rule_id"),
        category=required_str("category"),
        path_glob=path_glob,
        classification=classification,
        owner=required_str("owner"),
        rationale=required_str("rationale"),
        expires=expires,
        line=line,
    )
    return (None, errors) if errors else (exception, [])


def load_surface_exceptions(path: Path = DEFAULT_SURFACE_EXCEPTIONS) -> tuple[list[SurfaceException], list[str]]:
    """Load documented hardcoded-surface exceptions.

    Exceptions are not silent suppressions. Matching findings remain warnings,
    but production-blocker exceptions require an explicit emergency opt-in before
    strict mode can keep them as visible warnings.
    """
    if not path.exists():
        return [], []
    try:
        loaded = _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [], [f"failed to load surface exception file {path}: {exc}"]
    raw_exceptions = loaded.get("exceptions", [])
    if raw_exceptions is None:
        return [], []
    if not isinstance(raw_exceptions, list):
        return [], [f"surface exception file {path} has non-list 'exceptions'"]

    exceptions: list[SurfaceException] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_exceptions, start=1):
        exception, entry_errors = _surface_exception_from_raw(index, raw)
        errors.extend(entry_errors)
        if exception is not None:
            exceptions.append(exception)
    return exceptions, errors


def _production_blocker_waiver_errors(exceptions: list[SurfaceException]) -> list[str]:
    errors: list[str] = []
    for index, exception in enumerate(exceptions, start=1):
        if exception.category != "production_blocker":
            continue
        errors.append(
            f"surface exception #{index} waives production_blocker.{exception.rule_id}; "
            "rerun with --allow-production-blocker-waivers only for an intentional, "
            "owned emergency waiver"
        )
    return errors


def _accepted_gap_from_raw(
    index: int, raw: Any
) -> tuple[AcceptedGapDeclaration | None, list[str]]:
    errors: list[str] = []
    prefix = f"accepted gap #{index}"
    if not isinstance(raw, dict):
        return None, [f"{prefix} is not a mapping"]

    def required_str(field: str) -> str:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix} missing non-empty {field!r}")
            return ""
        return value.strip()

    role = required_str("role")
    gap = required_str("gap")
    reason = required_str("reason")
    owner = required_str("owner")

    declared = required_str("declared")
    if declared:
        try:
            date.fromisoformat(declared)
        except ValueError:
            errors.append(f"{prefix} declared must be an ISO date YYYY-MM-DD")

    # Expiry is REQUIRED and enforced. An accepted gap that never expires is a
    # silenced check with extra paperwork.
    expires = required_str("expires")
    if expires:
        try:
            expires_date = date.fromisoformat(expires)
        except ValueError:
            errors.append(f"{prefix} expires must be an ISO date YYYY-MM-DD")
        else:
            if expires_date < date.today():
                errors.append(
                    f"{prefix} for role {role or '?'} gap {gap or '?'!r} "
                    f"expired on {expires}; renew it with fresh evidence or close the gap"
                )

    declaration = AcceptedGapDeclaration(
        role=role,
        gap=gap,
        reason=reason,
        owner=owner,
        declared=declared,
        expires=expires,
    )
    return (None, errors) if errors else (declaration, [])


def load_accepted_gaps(
    path: Path = DEFAULT_ACCEPTED_GAPS,
) -> tuple[list[AcceptedGapDeclaration], list[str]]:
    """Load operator-declared, expiring acceptances of known stack-prior gaps.

    An expired or malformed declaration is dropped AND reported: the gap it
    covered goes back to being an error, and the stale declaration is an error
    of its own. Silently honouring an expired waiver is how a temporary
    acceptance becomes permanent.
    """
    if not path.exists():
        return [], []
    try:
        loaded = _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [], [f"failed to load accepted-gap file {path}: {exc}"]
    raw_gaps = loaded.get("accepted_gaps", [])
    if raw_gaps is None:
        return [], []
    if not isinstance(raw_gaps, list):
        return [], [f"accepted-gap file {path} has non-list 'accepted_gaps'"]

    declarations: list[AcceptedGapDeclaration] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_gaps, start=1):
        declaration, entry_errors = _accepted_gap_from_raw(index, raw)
        errors.extend(entry_errors)
        if declaration is not None:
            declarations.append(declaration)
    return declarations, errors


def _matching_accepted_gap(
    role: str,
    gap: str,
    declarations: list[AcceptedGapDeclaration],
) -> AcceptedGapDeclaration | None:
    for declaration in declarations:
        if declaration.matches(role, gap):
            return declaration
    return None


def _unmatched_accepted_gap_errors(
    present_gaps: set[tuple[str, str]],
    declarations: list[AcceptedGapDeclaration],
) -> list[str]:
    """Return errors for declarations whose gap is no longer present.

    Same shape as `_unmatched_surface_exception_errors`: a waiver that no longer
    corresponds to anything is drift, and drift that reports nothing is how a
    declaration file turns into a list of things nobody remembers deciding.
    """
    errors: list[str] = []
    for index, declaration in enumerate(declarations, start=1):
        if (declaration.role, declaration.gap) in present_gaps:
            continue
        errors.append(
            f"accepted gap #{index} no longer matches a stack-prior gap: "
            f"role {declaration.role!r} gap {declaration.gap!r}; remove the stale declaration"
        )
    return errors


def _accepted_gap_warning(
    scope: str,
    role: str,
    gap: str,
    declaration: AcceptedGapDeclaration,
) -> str:
    return (
        f"accepted_gap.{scope}.{role}: {gap} [declaration: {declaration.warning_suffix()}]"
    )


def _matching_surface_exception(
    finding: SurfaceFinding,
    exceptions: list[SurfaceException],
) -> SurfaceException | None:
    for exception in exceptions:
        if exception.matches(finding):
            return exception
    return None


def _surface_warning_for_finding(
    finding: SurfaceFinding,
    exceptions: list[SurfaceException],
) -> str:
    exception = _matching_surface_exception(finding, exceptions)
    if exception is None:
        return finding.to_warning()
    return (
        f"hardcoded_surface.waived.{finding.category}.{finding.rule_id}: "
        f"{finding.path}:{finding.line}: {finding.snippet} "
        f"[exception: {exception.warning_suffix()}]"
    )


def _unmatched_surface_exception_errors(
    findings: list[SurfaceFinding],
    exceptions: list[SurfaceException],
) -> list[str]:
    """Return errors for waivers that no longer match current scan findings."""
    errors: list[str] = []
    for index, exception in enumerate(exceptions, start=1):
        if any(exception.matches(finding) for finding in findings):
            continue
        line_suffix = f":{exception.line}" if exception.line is not None else ""
        errors.append(
            "surface exception "
            f"#{index} no longer matches a hardcoded-surface finding: "
            f"{exception.category}.{exception.rule_id} "
            f"{exception.path_glob}{line_suffix}; remove the stale waiver"
        )
    return errors


def stack_prior_role_choices(
    priors: dict[str, Any],
    *,
    retired_roles: frozenset[str] | None = None,
) -> list[str]:
    """Return model role choices that procedure inputs should accept."""
    roles = priors.get("roles")
    if not isinstance(roles, dict):
        return []
    if retired_roles is None:
        retired_roles = retired_live_roles()

    choices: list[str] = []
    for role, record in roles.items():
        if not isinstance(role, str) or not isinstance(record, dict):
            continue
        if role in retired_roles:
            continue
        if record.get("deployment_status") == "retired":
            continue
        choices.append(role)
    return sorted(choices)


def stack_prior_permission_role_choices(
    priors: dict[str, Any],
    *,
    retired_roles: frozenset[str] | None = None,
) -> list[str]:
    """Return live executor roles accepted by the procedure schema."""
    roles = priors.get("roles")
    if not isinstance(roles, dict):
        return []
    if retired_roles is None:
        retired_roles = retired_live_roles()

    choices: list[str] = []
    for role, record in roles.items():
        if not isinstance(role, str) or not isinstance(record, dict):
            continue
        if role in retired_roles:
            continue
        if record.get("deployment_status") == "live_stack":
            choices.append(role)
    return sorted(choices) + ["admin"]


def _procedure_input_enum(procedure_path: Path, input_name: str) -> list[str] | None:
    procedure = _load_yaml(procedure_path)
    inputs = procedure.get("inputs")
    if not isinstance(inputs, list):
        return None
    for raw_input in inputs:
        if not isinstance(raw_input, dict) or raw_input.get("name") != input_name:
            continue
        validation = raw_input.get("validation")
        if not isinstance(validation, dict):
            return None
        enum = validation.get("enum")
        if not isinstance(enum, list):
            return None
        return [str(item) for item in enum if isinstance(item, str)]
    return None


def _procedure_schema_permission_enum(schema_path: Path) -> list[str] | None:
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    try:
        enum = schema["properties"]["permissions"]["properties"]["roles"]["items"]["enum"]
    except (KeyError, TypeError):
        return None
    if not isinstance(enum, list):
        return None
    return [str(item) for item in enum if isinstance(item, str)]


def validate_procedure_role_enums(
    priors: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    procedure_path: Path | None = None,
    schema_path: Path | None = None,
    retired_roles: frozenset[str] | None = None,
) -> list[str]:
    """Validate generated procedure role enums against stack priors."""
    errors: list[str] = []
    if retired_roles is None:
        retired_roles, retired_errors = _retired_live_roles_or_error()
        errors.extend(retired_errors)
    raw_procedure_path = procedure_path
    if raw_procedure_path is None:
        resolved_procedure_path = repo_root / DEFAULT_ADD_MODEL_PROCEDURE.relative_to(REPO_ROOT)
    else:
        resolved_procedure_path = (
            raw_procedure_path
            if raw_procedure_path.is_absolute()
            else repo_root / raw_procedure_path
        )
    # A default artifact path resolved against a FOREIGN repo root is genuinely
    # absent-by-construction (fixtures, sibling checkouts) — same carve-out the
    # surface-exception loader already applies. Everywhere else, "the artifact I
    # exist to check is not there" is an error, exactly as a missing
    # `source_artifacts.<label>` is an error in `validate_stack_priors`: a check
    # you can pass by DELETING the thing it inspects is not a check.
    default_paths_for_other_repo = repo_root.resolve() != REPO_ROOT.resolve()
    if resolved_procedure_path.exists():
        expected = stack_prior_role_choices(priors, retired_roles=retired_roles)
        actual = _procedure_input_enum(resolved_procedure_path, "role")
        if actual is None:
            rel_path = _display_path(resolved_procedure_path, repo_root)
            errors.append(f"procedure role enum missing: {rel_path} input 'role'")
        elif actual != expected:
            rel_path = _display_path(resolved_procedure_path, repo_root)
            errors.append(
                f"procedure role enum drift: {rel_path} input 'role' expected {expected} "
                f"from stack priors, got {actual} "
                "[run: scripts/registry/sync_procedure_role_enums.py]"
            )
    elif raw_procedure_path is not None or not default_paths_for_other_repo:
        errors.append(
            f"{COULD_NOT_CHECK}: procedure artifact is missing: "
            f"{_display_path(resolved_procedure_path, repo_root)}; the role-enum "
            "comparison was skipped, not passed"
        )

    raw_schema_path = schema_path
    if raw_schema_path is None:
        resolved_schema_path = repo_root / DEFAULT_PROCEDURE_SCHEMA.relative_to(REPO_ROOT)
    else:
        resolved_schema_path = (
            raw_schema_path if raw_schema_path.is_absolute() else repo_root / raw_schema_path
        )
    if resolved_schema_path.exists():
        expected_permissions = stack_prior_permission_role_choices(
            priors, retired_roles=retired_roles
        )
        actual_permissions = _procedure_schema_permission_enum(resolved_schema_path)
        if actual_permissions is None:
            rel_path = _display_path(resolved_schema_path, repo_root)
            errors.append(f"procedure schema permission enum missing: {rel_path}")
        elif actual_permissions != expected_permissions:
            rel_path = _display_path(resolved_schema_path, repo_root)
            errors.append(
                f"procedure schema permission enum drift: {rel_path} expected "
                f"{expected_permissions} from live stack priors plus admin, "
                f"got {actual_permissions} "
                "[run: scripts/registry/sync_procedure_role_enums.py]"
            )
    elif raw_schema_path is not None or not default_paths_for_other_repo:
        errors.append(
            f"{COULD_NOT_CHECK}: procedure schema artifact is missing: "
            f"{_display_path(resolved_schema_path, repo_root)}; the permission-enum "
            "comparison was skipped, not passed"
        )
    return errors


# ---------------------------------------------------------------------------
# REQUIRED_SOURCE_ARTIFACTS staleness — commit-time gate
#
# P1-10 (handoffs/active/numa-topology-cutover-resume-20260730.md): a commit that
# touches a pinned source artifact without shipping regenerated priors leaves
# `source_artifacts.<label>.sha256` pointing at bytes that no longer exist. The
# EXISTING loop in `validate_stack_priors` already catches that — but only when
# somebody runs the guard, which in practice is when `orchestrator_stack.py
# start` refuses to launch. The failure therefore surfaces hours later, on a
# different machine state, to whoever tried to start the stack; twice in one day
# from two different sessions. The bytes were already wrong at `git commit`.
#
# Two things make this a different check rather than a re-run of that loop:
#
#   1. SCOPE OF CONTENT. `validate_stack_priors` hashes the WORKING TREE. At
#      commit time the question is what the COMMIT will contain, which is the
#      git index. A file edited-but-not-staged, or staged-then-further-edited,
#      gives the two a different answer, and the index is the one that matters
#      for "will this commit ship a stale pin".
#   2. SCOPE OF BLAME. This repository is stale RIGHT NOW on pins nobody in this
#      commit touched. An unconditional gate would block every unrelated commit
#      on inherited debt, and the only escape is `--no-verify`, which also skips
#      the PII and hermes-drift hooks (see .git/hooks/pre-commit.extras, where a
#      previously unscoped gate had to be narrowed for exactly this reason). So:
#      a mismatch on a path THIS commit touches is an ERROR; a mismatch the
#      commit inherits is a visible WARNING. `strict=True` promotes the latter,
#      for a CI job whose remit IS the inherited debt.
#
# Everything that is not a clean comparison is an ERROR, never a warning and
# never a silent pass: a missing artifact, a path that does not resolve, an
# absent or malformed pin, an empty `source_artifacts` section, an empty
# `REQUIRED_SOURCE_ARTIFACTS` floor, an unreadable/unparseable priors file, or a
# git index this process cannot read. A staleness check that passes because the
# thing it inspects was deleted is worse than no check.
# ---------------------------------------------------------------------------


class GitIndexUnavailableError(RuntimeError):
    """Raised when git plumbing cannot answer; never swallowed into a pass."""


def _git_bytes(repo_root: Path, args: list[str]) -> bytes:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitIndexUnavailableError(f"git {' '.join(args)} could not run: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise GitIndexUnavailableError(
            f"git {' '.join(args)} exited {proc.returncode}: {detail}"
        )
    return proc.stdout


def _git_repo_root(repo_root: Path) -> Path:
    return Path(_git_bytes(repo_root, ["rev-parse", "--show-toplevel"]).decode().strip())


def _staged_paths(repo_root: Path) -> set[str]:
    """Repo-relative paths the pending commit adds, edits, renames or deletes."""
    args = ["diff", "--cached", "--name-only", "--diff-filter=ACMRDT", "-z"]
    try:
        _git_bytes(repo_root, ["rev-parse", "--verify", "HEAD"])
    except GitIndexUnavailableError:
        args.append(_EMPTY_TREE_OID)
    raw = _git_bytes(repo_root, args).decode("utf-8", "replace")
    return {entry for entry in raw.split("\0") if entry}


def _read_file_bytes(path: Path) -> tuple[bytes | None, str | None]:
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return None, f"unreadable on disk ({exc.__class__.__name__}: {exc})"


def _repo_relative(path: Path, repo_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _committed_content(
    path: Path, *, scope: str, repo_root: Path
) -> tuple[bytes | None, str | None]:
    """Return the bytes this commit will carry for ``path`` (or why it cannot)."""
    if scope == "worktree":
        return _read_file_bytes(path)
    rel = _repo_relative(path, repo_root)
    if rel is None:
        # Pinned artifact outside the repository can never be staged, so the
        # bytes the commit is judged against are the bytes on disk.
        return _read_file_bytes(path)
    try:
        return _git_bytes(repo_root, ["show", f":{rel}"]), None
    except GitIndexUnavailableError:
        pass
    # Absent from the index: either the commit deletes it, or it was never
    # tracked (in which case the worktree copy is what a consumer will read).
    try:
        _git_bytes(repo_root, ["cat-file", "-e", f"HEAD:{rel}"])
    except GitIndexUnavailableError:
        return _read_file_bytes(path)
    return None, f"staged for DELETION by this commit (HEAD:{rel} exists, index entry gone)"


def check_source_artifact_staleness(
    priors_path: Path = DEFAULT_PRIORS,
    *,
    scope: str = "staged",
    repo_root: Path = REPO_ROOT,
    strict: bool = False,
) -> GuardResult:
    """Are the ``source_artifacts`` pins true of the bytes being committed?

    ``scope='staged'`` hashes git index content (pre-commit gate).
    ``scope='worktree'`` hashes files on disk and blames every mismatch.
    ``strict=True`` promotes inherited mismatches from warnings to errors.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if scope not in STALENESS_SCOPES:
        return GuardResult(
            errors=[f"unknown staleness scope {scope!r}; expected one of {list(STALENESS_SCOPES)}"],
            warnings=[],
        )

    # A floor of zero names would make the union below equal to whatever the
    # priors happen to declare, so emptying `source_artifacts` AND the floor
    # would check nothing and report success. Refuse.
    if not REQUIRED_SOURCE_ARTIFACTS:
        errors.append(
            "REQUIRED_SOURCE_ARTIFACTS is EMPTY: the staleness floor names no artifact, "
            "so this check would be vacuous. Refusing to pass."
        )

    resolved_root = repo_root
    staged: set[str] = set()
    if scope == "staged":
        try:
            resolved_root = _git_repo_root(repo_root)
            staged = _staged_paths(resolved_root)
        except GitIndexUnavailableError as exc:
            errors.append(
                f"cannot read the git index under {repo_root}: {exc}. "
                "Refusing to pass; use --staleness-scope worktree outside a git checkout."
            )
            return GuardResult(errors=errors, warnings=warnings)

    priors_bytes, priors_reason = _committed_content(
        priors_path, scope=scope, repo_root=resolved_root
    )
    if priors_bytes is None:
        errors.append(f"stack priors artifact {priors_path} is {priors_reason}")
        return GuardResult(errors=errors, warnings=warnings)
    try:
        priors = yaml.safe_load(priors_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        errors.append(f"stack priors artifact {priors_path} did not parse: {exc}")
        return GuardResult(errors=errors, warnings=warnings)
    if not isinstance(priors, dict):
        errors.append(f"stack priors artifact {priors_path} did not parse to a mapping")
        return GuardResult(errors=errors, warnings=warnings)

    sources = priors.get("source_artifacts")
    if not isinstance(sources, dict) or not sources:
        errors.append(
            f"stack priors artifact {priors_path} has no non-empty source_artifacts "
            "section: nothing to compare is a FAILURE, not a pass"
        )
        sources = {}

    priors_rel = _repo_relative(priors_path, resolved_root)
    # Rewriting the priors file makes every pin in it this commit's claim, so
    # blame is no longer restricted to the artifacts the commit also touched.
    blame_everything = (
        strict or scope == "worktree" or (priors_rel is not None and priors_rel in staged)
    )

    for label in sorted(set(REQUIRED_SOURCE_ARTIFACTS) | set(sources)):
        source = sources.get(label)
        if not isinstance(source, dict):
            errors.append(
                f"source_artifacts.{label} is absent or not a mapping in {priors_path}: "
                f"{source!r}"
            )
            continue
        path = _source_path(priors_path, source)
        if path is None:
            errors.append(
                f"source_artifacts.{label}.path does not resolve to a usable path: "
                f"{source.get('path')!r}"
            )
            continue
        expected = source.get("sha256")
        if not isinstance(expected, str) or not _SHA256_RE.match(expected):
            errors.append(
                f"source_artifacts.{label}.sha256 is not a sha256 digest: {expected!r}"
            )
            continue
        content, reason = _committed_content(path, scope=scope, repo_root=resolved_root)
        if content is None:
            errors.append(f"source_artifacts.{label} ({path}) is {reason}")
            continue
        actual = hashlib.sha256(content).hexdigest()
        if actual == expected:
            continue
        artifact_rel = _repo_relative(path, resolved_root)
        blamed = blame_everything or (artifact_rel is not None and artifact_rel in staged)
        message = (
            f"source_artifacts.{label} is STALE: {path} ({scope}) hashes to {actual}, "
            f"priors pin {expected}"
        )
        if blamed:
            errors.append(f"{message} — {SOURCE_ARTIFACT_REMEDIATION}")
        else:
            warnings.append(
                f"pre-existing staleness not introduced by this commit: {message} "
                f"— {SOURCE_ARTIFACT_REMEDIATION}"
            )

    return GuardResult(errors=errors, warnings=warnings)


def validate_stack_priors(
    priors_path: Path = DEFAULT_PRIORS,
    *,
    strict: bool = False,
    scan_surfaces: bool = False,
    repo_root: Path = REPO_ROOT,
    surface_categories: frozenset[str] | None = frozenset({"production_blocker"}),
    surface_exceptions_path: Path | None = DEFAULT_SURFACE_EXCEPTIONS,
    surface_manifest_path: Path | None = DEFAULT_SURFACE_MANIFEST,
    procedure_path: Path | None = None,
    procedure_schema_path: Path | None = None,
    launch_manifest_targets: dict[str, dict[str, Any]] | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    allow_production_blocker_waivers: bool = False,
    accepted_gaps_path: Path | None = DEFAULT_ACCEPTED_GAPS,
    launch_numa_mode: str | None = None,
) -> GuardResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not priors_path.exists():
        return GuardResult(errors=[f"missing stack priors artifact: {priors_path}"], warnings=[])

    priors = _load_yaml(priors_path)
    retired_roles, retired_errors = _retired_live_roles_or_error()
    errors.extend(retired_errors)

    # Declared gaps are scoped to the artifact they were declared against. A
    # tmp_path fixture must not inherit production declarations, or every
    # fixture would trip the stale-declaration check.
    accepted_gaps: list[AcceptedGapDeclaration] = []
    if accepted_gaps_path is not None:
        default_gaps_for_other_priors = (
            accepted_gaps_path == DEFAULT_ACCEPTED_GAPS
            and priors_path.resolve() != DEFAULT_PRIORS.resolve()
        )
        if not default_gaps_for_other_priors:
            accepted_gaps, accepted_gap_errors = load_accepted_gaps(accepted_gaps_path)
            errors.extend(accepted_gap_errors)
    errors.extend(validate_stack_priors_contract(priors))
    errors.extend(
        validate_launch_manifest_serving_alignment(
            priors,
            launch_manifest_targets=launch_manifest_targets,
            registry_path=registry_path,
            descriptor_path=descriptor_path,
            launch_numa_mode=launch_numa_mode,
        )
    )
    roles = priors.get("roles")
    if not isinstance(roles, dict):
        errors.append("stack priors artifact has no mapping-valued roles section")
        roles = {}

    sources = priors.get("source_artifacts") or {}
    if not isinstance(sources, dict):
        errors.append("stack priors artifact has no source_artifacts section")
        sources = {}

    # 2026-08-01: verify EVERY pin the compiler emitted, not a hand-listed subset.
    #
    # REQUIRED_SOURCE_ARTIFACTS is a second restatement of the producer's pin set
    # (src/registry/stack_priors.py `source_artifacts`), and the two had already
    # diverged: the compiler emitted 9 pins and this loop checked 7, so
    # `launch_manifest.yaml` and `stack_topology.yaml` — which now hold the
    # launcher configuration that used to live in the .py files — were pinned and
    # never verified. Mutating them changed no verdict.
    #
    # This is the same shape as the `device` field being absent from runtime
    # attestation: nothing was duplicated, something was simply MISSING from a
    # hand-maintained checklist, so no amount of reviewing what WAS checked would
    # have found it. Iterating the producer's own keys means a pin added upstream
    # is verified automatically, with no one needing to remember.
    #
    # REQUIRED_SOURCE_ARTIFACTS survives as a FLOOR: these must be present, so a
    # compiler that silently stopped emitting one is still caught.
    for label in sorted(set(REQUIRED_SOURCE_ARTIFACTS) | set(sources)):
        source = sources.get(label)
        if not isinstance(source, dict):
            errors.append(f"missing source_artifacts.{label}")
            continue
        path = _source_path(priors_path, source)
        expected = source.get("sha256")
        actual = _sha256(path) if path else None
        if not path or actual is None:
            errors.append(f"source_artifacts.{label} path is missing or unreadable: {source.get('path')!r}")
        elif expected != actual:
            errors.append(
                f"source_artifacts.{label} hash mismatch: {path} expected {expected}, got {actual}"
            )

    for role in sorted(retired_roles & set(roles)):
        record = roles.get(role) or {}
        if record.get("deployment_status") == "live_stack":
            errors.append(f"retired role {role!r} appears as live_stack")
        else:
            warnings.append(f"retired role {role!r} appears in non-live priors")

    present_gaps: set[tuple[str, str]] = set()
    for role, record in sorted(roles.items()):
        if not isinstance(record, dict):
            errors.append(f"role {role!r} record is not a mapping")
            continue
        deployment_status = record.get("deployment_status")
        serving = record.get("serving") if isinstance(record.get("serving"), dict) else {}
        priors_block = record.get("priors") if isinstance(record.get("priors"), dict) else {}
        known_gaps = record.get("known_gaps") if isinstance(record.get("known_gaps"), list) else []
        if deployment_status == "live_stack":
            if not record.get("model_id"):
                errors.append(f"live role {role!r} is missing model_id")
            if not serving.get("endpoint"):
                errors.append(f"live role {role!r} is missing serving.endpoint")
            if serving.get("tier") == "hot" and priors_block.get("memory_cost") != 1.0:
                errors.append(
                    f"live HOT role {role!r} has memory_cost={priors_block.get('memory_cost')!r}"
                )
        if known_gaps:
            undeclared = 0
            for gap in known_gaps:
                gap_text = str(gap)
                present_gaps.add((role, gap_text))
                declaration = _matching_accepted_gap(role, gap_text, accepted_gaps)
                if declaration is None:
                    undeclared += 1
                else:
                    warnings.append(
                        _accepted_gap_warning("role", role, gap_text, declaration)
                    )
            if undeclared:
                warnings.append(f"role {role!r} has {undeclared} known gap(s)")

    global_gaps = priors.get("known_global_gaps")
    if isinstance(global_gaps, dict):
        for role, gaps in sorted(global_gaps.items()):
            if not gaps:
                continue
            undeclared = 0
            for gap in gaps:
                gap_text = str(gap)
                present_gaps.add((role, gap_text))
                declaration = _matching_accepted_gap(role, gap_text, accepted_gaps)
                if declaration is None:
                    undeclared += 1
                else:
                    warnings.append(
                        _accepted_gap_warning("known_global_gaps", role, gap_text, declaration)
                    )
            if undeclared:
                warnings.append(f"known_global_gaps.{role}: {undeclared} gap(s)")
    elif global_gaps:
        errors.append("known_global_gaps must be a mapping when present")

    errors.extend(_unmatched_accepted_gap_errors(present_gaps, accepted_gaps))

    if scan_surfaces:
        if surface_manifest_path is not None:
            errors.extend(validate_surface_manifest(surface_manifest_path))
        errors.extend(
            validate_procedure_role_enums(
                priors,
                repo_root=repo_root,
                procedure_path=procedure_path,
                schema_path=procedure_schema_path,
                retired_roles=retired_roles,
            )
        )
        surface_exceptions: list[SurfaceException] = []
        if surface_exceptions_path is not None:
            default_exceptions_for_other_repo = (
                surface_exceptions_path == DEFAULT_SURFACE_EXCEPTIONS
                and repo_root.resolve() != REPO_ROOT.resolve()
            )
            if not default_exceptions_for_other_repo:
                surface_exceptions, exception_errors = load_surface_exceptions(
                    surface_exceptions_path
                )
                errors.extend(exception_errors)
                if not allow_production_blocker_waivers:
                    errors.extend(_production_blocker_waiver_errors(surface_exceptions))
        surface_findings = scan_hardcoded_surfaces(repo_root, categories=surface_categories)
        surface_findings.extend(
            scan_role_fact_surfaces(priors, repo_root, categories=surface_categories)
        )
        # Validate waiver staleness against a full-category scan. Otherwise a
        # documented legacy_test / historical_doc exception would be reported as
        # a stale waiver whenever the surface report is scoped to
        # production_blocker only (the default), because those findings are not
        # in the scoped report set. Reporting still uses ``surface_findings``.
        if surface_categories is None:
            staleness_findings = surface_findings
        else:
            staleness_findings = scan_hardcoded_surfaces(repo_root, categories=None)
            staleness_findings.extend(
                scan_role_fact_surfaces(priors, repo_root, categories=None)
            )
        errors.extend(_unmatched_surface_exception_errors(staleness_findings, surface_exceptions))
        for finding in surface_findings:
            warnings.append(_surface_warning_for_finding(finding, surface_exceptions))

    if strict and warnings:
        retained_warnings: list[str] = []
        for warning in warnings:
            # Two prefixes survive strict mode, and both mean the same thing: a
            # human with a name and a deadline has looked at this and accepted
            # it. Everything else is an error, because the gate cannot tell the
            # difference between "unmeasured" and "unsafe" on its own.
            if warning.startswith(("hardcoded_surface.waived.", "accepted_gap.")):
                retained_warnings.append(warning)
            else:
                errors.append(f"strict: {warning}")
        warnings = retained_warnings

    return GuardResult(errors=errors, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated stack priors")
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any known gaps, not only stale hashes or live-role invariants",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used by the hardcoded-surface scanner",
    )
    parser.add_argument(
        "--numa-mode",
        help=(
            "Explicit NUMA lineup to validate; defaults to the lineup declared in this "
            "checkout's orchestration/stack_topology.yaml"
        ),
    )
    parser.add_argument(
        "--skip-hardcoded-surface-scan",
        action="store_true",
        help="Skip curated hardcoded model/stack surface warnings",
    )
    parser.add_argument(
        "--all-hardcoded-surfaces",
        action="store_true",
        help="Report legacy-test and historical-doc surfaces in addition to production blockers",
    )
    parser.add_argument(
        "--hardcoded-surface-category",
        action="append",
        choices=sorted({rule.category for rule in HARDCODED_SURFACE_RULES}),
        help="Surface category to report; defaults to production_blocker",
    )
    parser.add_argument(
        "--surface-exceptions",
        type=Path,
        default=DEFAULT_SURFACE_EXCEPTIONS,
        help="YAML file documenting hardcoded-surface exceptions",
    )
    parser.add_argument(
        "--allow-production-blocker-waivers",
        action="store_true",
        help=(
            "Permit documented production-blocker hardcoded-surface waivers as "
            "visible warnings; default rejects them fail-closed"
        ),
    )
    parser.add_argument(
        "--surface-manifest",
        type=Path,
        default=DEFAULT_SURFACE_MANIFEST,
        help="YAML file documenting hardcoded-surface ownership",
    )
    parser.add_argument(
        "--accepted-gaps",
        type=Path,
        default=DEFAULT_ACCEPTED_GAPS,
        help=(
            "YAML file declaring owned, expiring acceptances of known stack-prior "
            "gaps; declared gaps stay visible warnings in strict mode"
        ),
    )
    parser.add_argument(
        "--list-hardcoded-surface-rules",
        action="store_true",
        help="Print the curated hardcoded-surface rule inventory and exit",
    )
    parser.add_argument(
        "--surface-inventory-format",
        choices=("yaml", "json"),
        default="yaml",
        help="Format for --list-hardcoded-surface-rules",
    )
    parser.add_argument(
        "--surface-summary-only",
        action="store_true",
        help="Print warning counts instead of individual warning lines",
    )
    parser.add_argument(
        "--check-source-artifact-staleness",
        action="store_true",
        help=(
            "Run ONLY the REQUIRED_SOURCE_ARTIFACTS staleness check and exit; "
            "intended as a pre-commit gate"
        ),
    )
    parser.add_argument(
        "--staleness-scope",
        choices=STALENESS_SCOPES,
        default="staged",
        help=(
            "Content the pins are compared against: 'staged' hashes the git "
            "index (what the commit will contain), 'worktree' hashes files on "
            "disk and blames every mismatch"
        ),
    )
    parser.add_argument(
        "--staleness-strict",
        action="store_true",
        help=(
            "Promote pre-existing staleness (pins this commit does not touch) "
            "from warnings to errors"
        ),
    )
    args = parser.parse_args(argv)

    if args.check_source_artifact_staleness:
        staleness = check_source_artifact_staleness(
            args.priors,
            scope=args.staleness_scope,
            repo_root=args.repo_root,
            strict=args.staleness_strict,
        )
        for warning in staleness.warnings:
            print(f"WARN: {warning}")
        if staleness.errors:
            print(f"FAIL: {len(staleness.errors)} source-artifact staleness error(s)")
            for error in staleness.errors:
                print(f"  - {error}")
            return 1
        print(
            f"OK: source artifacts match the pins in {args.priors} "
            f"({args.staleness_scope} scope)"
        )
        return 0

    if args.list_hardcoded_surface_rules:
        manifest, manifest_errors = load_surface_manifest(args.surface_manifest)
        if manifest_errors:
            print(f"FAIL: {len(manifest_errors)} surface manifest error(s)")
            for error in manifest_errors:
                print(f"  - {error}")
            return 1
        manifest_validation_errors = validate_surface_manifest(args.surface_manifest)
        if manifest_validation_errors:
            print(f"FAIL: {len(manifest_validation_errors)} surface manifest error(s)")
            for error in manifest_validation_errors:
                print(f"  - {error}")
            return 1
        inventory = hardcoded_surface_rule_inventory(ownership_manifest=manifest)
        if args.surface_inventory_format == "json":
            print(json.dumps(inventory, indent=2, sort_keys=True))
        else:
            print(yaml.safe_dump(inventory, sort_keys=False))
        return 0

    launch_numa_mode, numa_mode_error = _standalone_numa_mode(args.numa_mode)
    if numa_mode_error is not None:
        print(numa_mode_error)
        return 1

    if args.all_hardcoded_surfaces:
        surface_categories = None
    elif args.hardcoded_surface_category:
        surface_categories = frozenset(args.hardcoded_surface_category)
    else:
        surface_categories = frozenset({"production_blocker"})

    result = validate_stack_priors(
        args.priors,
        strict=args.strict,
        scan_surfaces=not args.skip_hardcoded_surface_scan,
        repo_root=args.repo_root,
        surface_categories=surface_categories,
        surface_exceptions_path=args.surface_exceptions,
        surface_manifest_path=args.surface_manifest,
        allow_production_blocker_waivers=args.allow_production_blocker_waivers,
        accepted_gaps_path=args.accepted_gaps,
        launch_numa_mode=launch_numa_mode,
    )
    if result.errors:
        print(f"FAIL: {len(result.errors)} stack-prior error(s)")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    if result.warnings:
        if args.surface_summary_only:
            print("\n".join(_warning_summary_lines(result.warnings)))
        else:
            print(f"WARN: {len(result.warnings)} stack-prior warning(s)")
            for warning in result.warnings:
                print(f"  - {warning}")
        return 0
    print(f"OK: {args.priors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
