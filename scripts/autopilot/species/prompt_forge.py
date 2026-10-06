"""Species 2 — PromptForge: LLM-guided prompt optimization.

Uses Claude CLI (Popen + session persistence) to analyze failure cases
and propose targeted prompt mutations on hot-swappable .md files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

log = logging.getLogger("autopilot.prompt_forge")

if TYPE_CHECKING:
    from scripts.autopilot.worktree_manager import ExperimentContext

import ast
import math
from enum import Enum

ORCH_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ORCH_ROOT / "orchestration" / "prompts"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Meta-Harness Tier 2: Python files that code mutations may touch.
# This is the eval trust boundary — files NOT on this list are immutable.
CODE_MUTATION_ALLOWLIST = [
    "src/prompt_builders/resolver.py",  # Prompt resolution logic
    "src/escalation.py",  # Escalation policy & retry logic
    "src/graph/escalation_helpers.py",  # Role cycle detection
    "src/tool_policy.py",  # Tool access control rules
    "src/api/routes/chat.py",  # Chat pipeline (cheap-first, routing, response)
]

# New-file code mutations are more permissive than the existing-file allowlist,
# but they stay directory-scoped. ``src/`` is for ordinary code scaffolds;
# ``schema_evolution/`` is the AutoMem/MH-9 lane for default-inert memory
# schema/scaffold proposals.
NEW_FILE_MUTATION_ROOT = PROJECT_ROOT / "src"
MEMORY_SCHEMA_MUTATION_ROOT = (
    PROJECT_ROOT / "orchestration" / "repl_memory" / "schema_evolution"
)

# RTG-55: the shape the MH-9 schema-evolution prompt asks for. It is shipped in
# the prompt verbatim and MUST itself pass ``screen_static_safety(strict=True)``
# — a test asserts that, so prompt/denylist drift cannot recur silently.
MEMORY_SCHEMA_SHAPE_EXAMPLE = '''"""Proposed plan-channel memory schema (inert)."""

SCHEMA_VERSION = 1
ACTIONS = ("APPEND", "CREATE", "UPSERT")
CHANNELS = ("status", "inventory", "strategy", "plan", "log")
SCHEMA = {
    "fields": (
        {"name": "channel", "required": True, "kind": "str"},
        {"name": "content", "required": True, "kind": "str"},
    ),
    "blockers": ("no calibration evidence for plan-channel upserts",),
}


def required_fields():
    return tuple(spec["name"] for spec in SCHEMA["fields"] if spec["required"])


def validate_action(action):
    if not isinstance(action, dict):
        return (False, "action must be a dict")
    for name in required_fields():
        if not action.get(name):
            return (False, f"missing field: {name}")
    if action.get("channel") not in CHANNELS:
        return (False, "channel must be one of CHANNELS")
    return (True, "ok")
'''

MUTATION_TYPES = [
    "targeted_fix",  # Fix specific failure patterns
    "compress",  # Reduce token count while maintaining behavior
    "few_shot_evolution",  # Add/remove/modify examples
    "crossover",  # Merge sections from two prompts
    "style_transfer",  # Apply patterns from one prompt to another
    "gepa",  # AP-19: GEPA evolutionary optimization (runs internal eval loop)
]

_MIN_VALIDATION_TRIALS = 5
_SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "aime": ("aime",),
    "coder": ("coder", "humaneval", "mbpp"),
    "cruxeval": ("cruxeval", "crux eval"),
    "debugbench": ("debugbench", "debug bench"),
    "gpqa": ("gpqa",),
    "gsm8k": ("gsm8k",),
    "hotpotqa": ("hotpotqa", "hotpot qa"),
    "livecodebench": ("livecodebench", "live code bench", "lcb"),
    "math": ("math",),
    "skill_transfer": ("skill_transfer", "skill transfer"),
    "thinking": ("thinking",),
    "usaco": ("usaco",),
}
_SUITE_TERM_TO_CANONICAL = {
    term: canonical for canonical, aliases in _SUITE_ALIASES.items() for term in aliases
}
_SUITE_TERM_RE = re.compile(
    r"(?<![\w-])("
    + "|".join(re.escape(term) for term in sorted(_SUITE_TERM_TO_CANONICAL, key=len, reverse=True))
    + r")(?![\w-])",
    re.IGNORECASE,
)
_TRIAL_REF_RE = re.compile(r"(?:\btrial\s*#?\s*|\[?t)(\d+)\]?", re.IGNORECASE)
_UNIVERSAL_TRANSFER_RE = re.compile(
    r"\b(always|never|universally|global(?:ly)?|all\s+(?:tasks|prompts|suites|benchmarks)|"
    r"every\s+(?:task|prompt|suite|benchmark))\b",
    re.IGNORECASE,
)
_FRONTDOOR_REQUIRED_MARKERS = (
    "# Front Door Orchestrator",
    "TaskIR mode",
    "Direct-answer mode",
    "Answer tags (scoped)",
)
_FRONTDOOR_CORRUPTION_MARKERS = (
    "fenced block from my response",
    "i should **not** edit the file directly",
    "one note worth flagging",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _coverage_retrieve(
    strategy_store: Any,
    query_text: str,
    *,
    journal: Any | None,
    k: int,
    species: str | None,
) -> list[Any]:
    if journal is not None and hasattr(strategy_store, "retrieve_for_journal"):
        try:
            return list(
                strategy_store.retrieve_for_journal(
                    query_text,
                    journal=journal,
                    k=k,
                    species=species,
                )
            )
        except TypeError:
            return list(
                strategy_store.retrieve_for_journal(
                    query_text,
                    journal=journal,
                    k=k,
                )
            )

    if hasattr(strategy_store, "retrieve"):
        try:
            return list(strategy_store.retrieve(query_text, k=k, species=species))
        except TypeError:
            return list(strategy_store.retrieve(query_text, k=k))

    raise AttributeError("strategy_store has neither retrieve_for_journal() nor retrieve()")


def diversity_coverage_penalty(
    query_text: str,
    strategy_store: Any | None,
    *,
    journal: Any | None = None,
    k: int = 8,
    species: str | None = "prompt_forge",
    min_density: float = 1e-6,
) -> dict[str, Any]:
    """Estimate mutation-neighborhood density from StrategyStore retrieval.

    AP-35 uses the existing strategy-memory index as an observe-only density
    proxy. ``negative_log_density`` is high in sparse neighborhoods and low
    near already-covered strategy clusters; callers decide how to present it.
    """
    query = str(query_text or "").strip()
    if strategy_store is None:
        return {
            "status": "unavailable",
            "reason": "missing_strategy_store",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }
    if not query:
        return {
            "status": "unavailable",
            "reason": "empty_query",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }

    k = max(1, int(k))
    floor = max(_safe_float(min_density, 1e-6), 1e-12)
    try:
        entries = _coverage_retrieve(
            strategy_store,
            query,
            journal=journal,
            k=k,
            species=species,
        )
    except Exception as exc:  # noqa: BLE001 - density hints must not block mutation dispatch
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "query_text": query,
            "density": 0.0,
            "negative_log_density": 0.0,
            "penalty": 0.0,
            "similar_count": 0,
            "top_matches": [],
        }

    top_matches: list[dict[str, Any]] = []
    scores: list[float] = []
    for entry in entries:
        score = max(0.0, _safe_float(getattr(entry, "similarity_score", 0.0)))
        scores.append(score)
        top_matches.append(
            {
                "id": str(getattr(entry, "id", "") or ""),
                "source_trial_id": getattr(entry, "source_trial_id", None),
                "species": str(getattr(entry, "species", "") or ""),
                "description": str(getattr(entry, "description", "") or ""),
                "insight": str(
                    getattr(entry, "generalized_content", "") or getattr(entry, "insight", "") or ""
                ),
                "similarity_score": score,
            }
        )

    if scores:
        density = sum(scores) / len(scores)
        status = "ok"
    else:
        density = 0.0
        status = "sparse"

    negative_log_density = -math.log(max(density, floor))
    return {
        "status": status,
        "reason": "ok" if scores else "no_nearby_strategy_entries",
        "query_text": query,
        "density": density,
        "negative_log_density": negative_log_density,
        "penalty": negative_log_density,
        "similar_count": len(scores),
        "top_matches": top_matches[:k],
        "interpretation": (
            "Higher negative_log_density means the mutation target is less covered "
            "by strategy memory; use as exploration pressure, not as an acceptance gate."
        ),
    }


@dataclass(frozen=True)
class TransferSafetyVerdict:
    valid: bool
    reason: str = "ok"
    warnings: tuple[str, ...] = ()
    source_suites: tuple[str, ...] = ()
    introduced_suites: tuple[str, ...] = ()
    evidence_trial_count: int = 0


@dataclass
class PromptMutation:
    file: str  # e.g., "frontdoor.md"
    mutation_type: str
    description: str
    original_content: str = ""
    mutated_content: str = ""
    git_diff: str = ""
    accepted: bool = False
    safety_valid: bool = True
    safety_reason: str = "ok"
    safety_warnings: list[str] = field(default_factory=list)
    # RTG-55 MHS-4: prompt-side effect (``MutationEffect``) and its risk weight.
    effect: MutationEffect | None = None
    effect_risk: float | None = None


class MutationEffect(str, Enum):
    """RTG-55 MHS-1 — the closed, host-normalized effect vocabulary.

    A code mutation's EFFECT is what it may DO, independent of which file it
    touches. The vocabulary is closed on purpose: an unrecognised effect
    normalizes to ``UNKNOWN`` and is never silently treated as benign.
    """

    INERT = "inert"  # New module; import executes nothing beyond defs/constants
    CONSTRAIN = "constrain"  # Add-only: no original line removed (guards, checks)
    EXPAND = "expand"  # Adds new top-level defs/classes, keeps every old one
    REPLACE = "replace"  # Rewrites or removes existing behaviour
    UNSAFE = "unsafe"  # Tripped the static safety screen; never applicable
    UNKNOWN = "unknown"  # Could not be classified (fail-closed sentinel)

    @classmethod
    def normalize(cls, raw: Any) -> MutationEffect:
        """Host-normalize an arbitrary value into the closed enum.

        Normalization is total: anything unrecognised becomes ``UNKNOWN``.
        """
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, str):
            return cls.UNKNOWN
        token = raw.strip().lower().replace("-", "_").replace(" ", "_")
        token = token.split(".")[-1][:_EFFECT_TOKEN_LIMIT]
        alias = _EFFECT_ALIASES.get(token)
        if alias is not None:
            return alias
        for member in cls:
            if member.value == token:
                return member
        return cls.UNKNOWN


# Host normalization limits (MHS-1: "host-normalized and truncated").
_EFFECT_TOKEN_LIMIT = 32
_EFFECT_REASON_LIMIT = 240

_EFFECT_ALIASES: dict[str, MutationEffect] = {
    "noop": MutationEffect.INERT,
    "no_op": MutationEffect.INERT,
    "inert": MutationEffect.INERT,
    "default_inert": MutationEffect.INERT,
    "new_file": MutationEffect.INERT,
    "guard": MutationEffect.CONSTRAIN,
    "check": MutationEffect.CONSTRAIN,
    "constrain": MutationEffect.CONSTRAIN,
    "add_only": MutationEffect.CONSTRAIN,
    "reprompt": MutationEffect.CONSTRAIN,
    "add": MutationEffect.EXPAND,
    "extend": MutationEffect.EXPAND,
    "expand": MutationEffect.EXPAND,
    "override": MutationEffect.REPLACE,
    "rewrite": MutationEffect.REPLACE,
    "replace": MutationEffect.REPLACE,
    "force": MutationEffect.REPLACE,
    "unsafe": MutationEffect.UNSAFE,
    "rejected": MutationEffect.UNSAFE,
}


def _truncate_reason(reason: str) -> str:
    """Truncate a screen reason to the host limit (never unbounded model text)."""
    text = " ".join(str(reason).split())
    if len(text) <= _EFFECT_REASON_LIMIT:
        return text
    return text[: _EFFECT_REASON_LIMIT - 1] + "…"


# ---------------------------------------------------------------------------
# RTG-55 MHS-2 — static safety screen (AST denylist).
#
# Validation is STATIC ONLY. It never writes into the live source tree and
# never imports/execs the candidate in this process.
# ---------------------------------------------------------------------------

# Callables a mutation may never invoke, at any nesting depth. Never grandfathered:
# an allowlisted file that somehow already contained these would not excuse a new one.
SAFETY_BANNED_CALLS: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "memoryview",
    }
)

# Callables a mutation may not ADD. Grandfathered when the original file already
# called them, so a targeted fix is judged on what it introduces.
SAFETY_RESTRICTED_CALLS: frozenset[str] = frozenset(
    {
        "open",
        "getattr",
        "setattr",
        "delattr",
    }
)

# Modules a mutation may never newly import, and whose attribute calls are
# rejected unless the ORIGINAL file already made the identical dotted call.
SAFETY_BANNED_MODULES: frozenset[str] = frozenset(
    {
        "builtins",
        "ctypes",
        "http",
        "importlib",
        "marshal",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "pty",
        "requests",
        "shutil",
        "signal",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
        "webbrowser",
    }
)

# Stdlib modules any mutation may import.
SAFETY_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "ast",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "logging",
        "math",
        "numbers",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "time",
        "types",
        "typing",
        "uuid",
        "warnings",
    }
)

# First-party package roots a mutation may import (the orchestrator's own code).
SAFETY_FIRST_PARTY_ROOTS: frozenset[str] = frozenset({"orchestration", "scripts", "src"})

# Dunder attributes tolerated; every other dunder attribute access is rejected
# (it is the standard sandbox-escape surface: __globals__, __class__, __subclasses__).
SAFETY_DUNDER_ATTR_ALLOWLIST: frozenset[str] = frozenset({"__name__", "__doc__", "__all__"})

# Module-level statement types any mutation may contain. Anything else at module
# level executes work at import time and is rejected.
SAFETY_TOPLEVEL_ALLOWED: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,  # `if TYPE_CHECKING:` / `if __name__ == "__main__":`
    ast.Try,  # import fallbacks
    ast.Pass,
)

# MHS-2 strict profile: the handoff's node denylist for `new_file` proposals, so
# "default-inert" is a compile-time property rather than prompt text.
SAFETY_STRICT_NODE_DENYLIST: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.While,
    ast.Lambda,
    ast.ClassDef,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)


@dataclass(frozen=True)
class StaticSafetyReport:
    """Outcome of the static screen: a verdict, its violations, and an effect."""

    safe: bool
    effect: MutationEffect
    violations: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        if self.safe:
            return "ok"
        return _truncate_reason("; ".join(self.violations) or "static safety screen failed")


def _dotted_name(node: ast.AST) -> str | None:
    """Render ``a.b.c`` attribute/name chains as a dotted string."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


def _collect_dotted_calls(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted:
                out.add(dotted)
    return out


def _collect_imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _top_level_definition_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def screen_static_safety(
    mutated: str,
    *,
    original: str = "",
    strict: bool = False,
) -> StaticSafetyReport:
    """Screen a candidate mutation statically (MHS-2). Nothing is written or executed.

    ``strict=True`` additionally applies the ``new_file`` inertness denylist.
    Capabilities the ORIGINAL file already used are grandfathered: an
    existing-file mutation is judged on what it ADDS, not on what the file has
    always done.
    """
    violations: list[str] = []
    try:
        tree = ast.parse(mutated)
    except SyntaxError as exc:
        return StaticSafetyReport(False, MutationEffect.UNSAFE, (f"syntax error: {exc}",))

    grandfathered_roots: set[str] = set()
    grandfathered_calls: set[str] = set()
    if original.strip():
        try:
            original_tree = ast.parse(original)
        except SyntaxError:
            original_tree = None
        if original_tree is not None:
            grandfathered_roots = _collect_imported_roots(original_tree)
            grandfathered_calls = _collect_dotted_calls(original_tree)

    # 1. Module-level statements must not do work at import time.
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant):
                continue  # docstring
            violations.append(
                f"top-level side effect: bare expression at line {getattr(node, 'lineno', 0)}"
            )
            continue
        if not isinstance(node, SAFETY_TOPLEVEL_ALLOWED):
            violations.append(
                f"top-level {type(node).__name__} at line {getattr(node, 'lineno', 0)} "
                "is not a def/class/import/assignment"
            )

    # 2. Imports: allowlist + first-party + grandfathered roots only.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            else:
                roots = [node.module.split(".")[0]] if node.module else []
            for root in roots:
                if root in grandfathered_roots:
                    continue
                if root in SAFETY_BANNED_MODULES:
                    violations.append(f"banned import: {root}")
                elif root not in SAFETY_IMPORT_ALLOWLIST and root not in SAFETY_FIRST_PARTY_ROOTS:
                    violations.append(f"import outside allowlist: {root}")

    # 3. Calls: banned builtins, and attribute calls on banned modules.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in SAFETY_BANNED_CALLS:
                violations.append(f"banned call: {func.id}()")
                continue
            if func.id in SAFETY_RESTRICTED_CALLS and func.id not in grandfathered_calls:
                violations.append(f"banned call: {func.id}()")
                continue
        dotted = _dotted_name(func)
        if dotted and dotted not in grandfathered_calls:
            root, _, attr = dotted.partition(".")
            if root in SAFETY_BANNED_MODULES and attr:
                violations.append(f"banned call: {dotted}()")
            elif attr and attr.split(".")[-1] in {"system", "popen", "spawn"}:
                violations.append(f"banned call: {dotted}()")

    # 4. Dunder attribute access (sandbox-escape surface).
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if (
                attr.startswith("__")
                and attr.endswith("__")
                and attr not in SAFETY_DUNDER_ATTR_ALLOWLIST
            ):
                violations.append(f"dunder attribute access: .{attr}")

    # 5. Strict (new_file) inertness denylist.
    if strict:
        for node in ast.walk(tree):
            if isinstance(node, SAFETY_STRICT_NODE_DENYLIST):
                violations.append(f"new_file denylist node: {type(node).__name__}")
            elif isinstance(node, ast.Name) and node.id.startswith("_"):
                violations.append(f"new_file underscore name: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                violations.append(f"new_file underscore attribute: .{node.attr}")

    # Deduplicate while preserving order, and cap the list.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in violations:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    ordered = ordered[:10]

    if ordered:
        return StaticSafetyReport(False, MutationEffect.UNSAFE, tuple(ordered))
    return StaticSafetyReport(
        True,
        classify_mutation_effect(original, mutated, is_new_file=strict or not original.strip()),
    )


def classify_mutation_effect(
    original: str,
    mutated: str,
    *,
    is_new_file: bool = False,
) -> MutationEffect:
    """Mechanically classify a screened mutation into the MHS-1 effect enum.

    Answers the handoff's Open Question 3: the CONSTRAIN/REPLACE split is
    derivable from the candidate itself, so it needs no separate label.
    """
    if is_new_file or not original.strip():
        return MutationEffect.INERT
    try:
        original_tree = ast.parse(original)
        mutated_tree = ast.parse(mutated)
    except SyntaxError:
        return MutationEffect.UNKNOWN

    original_names = _top_level_definition_names(original_tree)
    mutated_names = _top_level_definition_names(mutated_tree)
    if original_names - mutated_names:
        return MutationEffect.REPLACE

    original_lines = [line.strip() for line in original.splitlines() if line.strip()]
    mutated_lines = [line.strip() for line in mutated.splitlines() if line.strip()]
    remaining = list(mutated_lines)
    add_only = True
    for line in original_lines:
        if line in remaining:
            remaining.remove(line)
        else:
            add_only = False
            break
    if not add_only:
        return MutationEffect.REPLACE
    if mutated_names - original_names:
        return MutationEffect.EXPAND
    return MutationEffect.CONSTRAIN


# ---------------------------------------------------------------------------
# RTG-55 MHS-4 — ANTI-OVERRIDE risk prior.
#
# In Harness-R1's released held-out corpus every catastrophic regression came
# from an override (REPLACE-class) patch, and the trained editor converged to
# constrain-only (intake-1323#04). The prior is encoded as a closed per-effect
# risk weight: lower is safer. Ranking sorts candidates by it; the gate refuses
# any mutation whose weight reaches ``mutation_risk_gate()``. The default gate
# (1.0) refuses only UNKNOWN/UNSAFE, i.e. fail-closed on unclassified effects;
# an operator can lower it via ``AUTOPILOT_MUTATION_RISK_GATE`` (0.9 refuses
# REPLACE, i.e. constrain/expand-only). The weights are an ordinal PRIOR, not a
# calibrated probability.
#
# MHS-5 evidence (``orchestration/datasets/harness_r1_heldout_effect_corpus.json``,
# 23 valid patches, 1,270 held-out tasks): the ORDERING holds. All 4 REPLACE patches
# regressed (mean -8.4 pp, rescue:regression 0.21), while the 19 CONSTRAIN patches
# averaged +3.9 pp (rescue:regression 1.65). But CONSTRAIN is NOT regression-free:
# 4 of 19 regressed, and the single worst patch (-16.9 pp) is a hint-only CONSTRAIN
# patch. "Every catastrophic regression came from an override" does not hold on
# this corpus, so this gate ranks risk; it does not certify safety.
# ---------------------------------------------------------------------------

MUTATION_EFFECT_RISK: dict[MutationEffect, float] = {
    MutationEffect.INERT: 0.05,
    MutationEffect.CONSTRAIN: 0.25,
    MutationEffect.EXPAND: 0.50,
    MutationEffect.REPLACE: 0.90,
    MutationEffect.UNKNOWN: 1.00,
    MutationEffect.UNSAFE: math.inf,
}
MUTATION_RISK_GATE_ENV = "AUTOPILOT_MUTATION_RISK_GATE"
DEFAULT_MUTATION_RISK_GATE = 1.0


def mutation_effect_risk(effect: Any) -> float:
    """Risk weight of an effect; anything unrecognised is weighted as UNKNOWN."""
    return MUTATION_EFFECT_RISK[MutationEffect.normalize(effect)]


def mutation_risk_gate() -> float:
    """Active gate. A malformed or out-of-range override falls back to the default."""
    raw = os.environ.get(MUTATION_RISK_GATE_ENV, "").strip()
    if not raw:
        return DEFAULT_MUTATION_RISK_GATE
    try:
        value = float(raw)
    except ValueError:
        log.warning("Ignoring malformed %s=%r", MUTATION_RISK_GATE_ENV, raw)
        return DEFAULT_MUTATION_RISK_GATE
    if not math.isfinite(value) or value <= 0.0 or value > DEFAULT_MUTATION_RISK_GATE:
        # A gate above 1.0 would admit UNKNOWN effects; never allow loosening.
        log.warning("Ignoring out-of-range %s=%r", MUTATION_RISK_GATE_ENV, raw)
        return DEFAULT_MUTATION_RISK_GATE
    return value


def mutation_risk_gate_reason(effect: Any, gate: float | None = None) -> str | None:
    """Rejection reason when ``effect`` reaches the gate, else None."""
    threshold = mutation_risk_gate() if gate is None else gate
    normalized = MutationEffect.normalize(effect)
    risk = MUTATION_EFFECT_RISK[normalized]
    if risk >= threshold:
        return f"effect_risk_gate:{normalized.value} risk={risk:g} gate={threshold:g}"
    return None


def rank_mutations_by_risk(mutations: Any) -> list[Any]:
    """Order candidate mutations safest-first (stable, so ties keep proposal order)."""
    return sorted(mutations, key=lambda m: mutation_effect_risk(getattr(m, "effect", None)))


def classify_prompt_effect(original: str, mutated: str) -> MutationEffect:
    """CONSTRAIN/REPLACE split for prompt text (the prompt-side analogue of MHS-1).

    Add-only edits (every original non-blank line survives) CONSTRAIN; any
    removed or rewritten line REPLACEs. An unchanged prompt is INERT.
    """
    if mutated == original:
        return MutationEffect.INERT
    if not isinstance(original, str) or not isinstance(mutated, str):
        return MutationEffect.UNKNOWN
    remaining = [line.strip() for line in mutated.splitlines() if line.strip()]
    for line in (line.strip() for line in original.splitlines()):
        if not line:
            continue
        if line in remaining:
            remaining.remove(line)
        else:
            return MutationEffect.REPLACE
    return MutationEffect.CONSTRAIN


@dataclass
class CodeMutation:
    file: str  # Relative path, e.g. "src/escalation.py"
    mutation_type: str
    description: str
    original_content: str = ""
    mutated_content: str = ""
    git_diff: str = ""
    accepted: bool = False
    syntax_valid: bool = False
    safety_valid: bool = True
    safety_reason: str = "ok"
    safety_warnings: list[str] = field(default_factory=list)
    # RTG-55 MHS-1: the typed return-effect the static screen assigned.
    effect: MutationEffect = MutationEffect.UNKNOWN
    effect_reason: str = ""
    # RTG-55 MHS-4: the ANTI-OVERRIDE prior weight of ``effect``.
    effect_risk: float = 1.0


def _resolve_code_mutation_target(target_file: str) -> Path:
    """Resolve a code-mutation target while rejecting traversal and escapes."""
    requested = Path(target_file)
    if requested.is_absolute() or ".." in requested.parts:
        raise FileNotFoundError(f"Target file not found: {PROJECT_ROOT / target_file}")
    resolved = (PROJECT_ROOT / requested).resolve(strict=False)
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise FileNotFoundError(f"Target file not found: {PROJECT_ROOT / target_file}")
    return resolved


def new_file_mutation_roots() -> tuple[Path, ...]:
    """Directory roots where MH-9 may create brand-new Python modules."""
    return (NEW_FILE_MUTATION_ROOT, MEMORY_SCHEMA_MUTATION_ROOT)


def new_file_mutation_root_labels() -> tuple[str, ...]:
    """Planner-facing labels for sanctioned new-file mutation roots."""
    labels: list[str] = []
    for root in new_file_mutation_roots():
        try:
            labels.append(str(root.relative_to(PROJECT_ROOT)))
        except ValueError:
            labels.append(str(root))
    return tuple(labels)


def _is_under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _is_memory_schema_evolution_target(path: Path) -> bool:
    return path.is_relative_to(MEMORY_SCHEMA_MUTATION_ROOT)


def _suite_mentions(text: str) -> set[str]:
    return {
        _SUITE_TERM_TO_CANONICAL[match.group(1).lower()]
        for match in _SUITE_TERM_RE.finditer(text or "")
    }


def _added_text(original: str, mutated: str) -> str:
    original_lines = {line.strip() for line in original.splitlines() if line.strip()}
    return "\n".join(
        line.strip()
        for line in mutated.splitlines()
        if line.strip() and line.strip() not in original_lines
    )


def _trial_reference_count(text: str) -> int:
    return len({int(match) for match in _TRIAL_REF_RE.findall(text or "")})


def _prompt_integrity_reason(filename: str, content: str) -> str | None:
    """Return a rejection reason for prompt text known to be structurally corrupt."""
    if filename != "frontdoor.md":
        return None
    lowered = content.lower()
    for marker in _FRONTDOOR_CORRUPTION_MARKERS:
        if marker in lowered:
            return f"frontdoor_corruption_marker:{marker}"
    missing = [marker for marker in _FRONTDOOR_REQUIRED_MARKERS if marker not in content]
    if missing:
        return "frontdoor_missing_required_markers:" + ",".join(missing)
    return None


# ---------------------------------------------------------------------------
# RTG-55 MHS-3 — anti-leakage (UNDER-generalization) guard.
#
# ``_UNIVERSAL_TRANSFER_RE`` rejects OVER-generalization. This is the other half:
# a mutation that names a specific eval instance (question/sample/item id) is
# memorising the eval set, not fixing a behaviour. The id vocabulary is sourced
# from the eval DATA the tower actually samples from (the research question
# pool, the designed core files, the sentinel sets), never from a hand list.
# The guard is FAIL-CLOSED: if the vocabulary cannot be built, every mutation
# is rejected with ``eval_leakage_vocabulary_unavailable``.
# ---------------------------------------------------------------------------

# os.pathsep-separated override of the vocabulary sources; every listed source is
# then REQUIRED (a missing one fails closed).
EVAL_ID_VOCAB_SOURCES_ENV = "AUTOPILOT_EVAL_ID_VOCAB_SOURCES"
_EVAL_ID_KEYS = ("id", "qid", "stable_qid", "question_id")
_LEAKAGE_MIN_ID_LEN = 6
_LEAKAGE_MIN_NUMERIC_ID_LEN = 12
_LEAKAGE_MIN_FAMILY_PREFIX = 4
_LEAKAGE_MIN_FAMILY_MEMBERS = 3
_LEAKAGE_MIN_STEM = 4
_LEAKAGE_MAX_REPORTED = 5

# Generic instance references that need no vocabulary: "sample #12",
# "question id 42", "task_id == 17", "problem number 3". A qualifier (#/id/index/
# number) is required so ordinary prose such as "step 3" is not rejected.
#
# Two shapes (2026-09-16 review narrowing). PROSE — noun and qualifier separated by
# space/hyphen — accepts every separator ("question id: 42"). CODE — a snake_case
# identifier such as ``task_id`` — counts only as a COMPARISON (``==`` / ``is``):
# ``task_index = 0``, ``sample_id = 1`` and ``task_id: 7`` are ordinary
# assignment / mapping lines in code-shaped prompt text, not a pinned instance.
_LEAKAGE_GENERIC_RE = re.compile(
    r"\b(?:task|sample|item|question|problem|instance)"
    r"(?:"
    r"[\s-]*(?:#\s*|(?:id|idx|index|number|no\.)\s*(?:==|#|:|=|is)?\s*[\"']?)"
    r"|_(?:id|idx|index|number)\s*(?:==|\bis\b)\s*[\"']?"
    r")\d{1,6}\b",
    re.IGNORECASE,
)
_ID_FAMILY_TAIL_RE = re.compile(r"^(.*?[_/\-])(\d+|[0-9a-f]{8,})$", re.IGNORECASE)
_ID_NATIVE_FAMILY_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9+\-]{2,}/)\d+")
_ID_STEM_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)")


def _default_eval_id_sources() -> tuple[tuple[Path, bool], ...]:
    """(path, required) pairs for the eval-id vocabulary.

    The research question pool is the population every EvalTower draw comes
    from, so it is REQUIRED. Core and sentinel files are added when present.
    """
    override = os.environ.get(EVAL_ID_VOCAB_SOURCES_ENV, "").strip()
    if override:
        return tuple((Path(p), True) for p in override.split(os.pathsep) if p.strip())
    research_root = Path(
        os.environ.get("EPYC_RESEARCH_ROOT", "/mnt/raid0/llm/epyc-inference-research")
    )
    sources: list[tuple[Path, bool]] = [
        (research_root / "benchmarks" / "prompts" / "question_pool.jsonl", True)
    ]
    core_dir = ORCH_ROOT / "benchmarks" / "prompts"
    sources.extend((p, False) for p in sorted(core_dir.glob("core_*.jsonl")))
    autopilot_dir = Path(__file__).resolve().parents[1]
    for name in ("sentinel_questions.yaml", "tool_sentinels.yaml"):
        sources.append((autopilot_dir / name, False))
    return tuple(sources)


def _iter_eval_rows(path: Path):
    """Yield dict rows from a .jsonl / .json / .yaml eval source (metadata rows skipped)."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and not any(str(k).startswith("__") for k in row):
                    yield row
        return
    if suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported eval-id source type: {path.name}")
    if isinstance(data, dict):
        data = data.get("questions", data.get("items", []))
    for row in data or []:
        if isinstance(row, dict):
            yield row


def stable_question_qid(suite: str, prompt_text: str) -> str:
    """Text-only question identity; MUST equal ``eval_tower._stable_question_qid``.

    Duplicated rather than imported so the proposer does not import the eval
    tower; a unit test pins the parity.
    """
    payload = f"{suite}\x00{prompt_text}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _is_identifier_shaped(raw: str) -> bool:
    """An id specific enough to name ONE instance (bare small integers are not)."""
    if len(raw) < _LEAKAGE_MIN_ID_LEN:
        return False
    return not raw.isdigit() or len(raw) >= _LEAKAGE_MIN_NUMERIC_ID_LEN


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


# ---------------------------------------------------------------------------
# RTG-55 MHS-3, 2026-09-15 structural form — eval CONTENT leakage.
#
# Beyond naming an instance, a mutation can memorise the eval set by carrying
# its CONTENT. Three refusals, all over the text a mutation ADDS:
#
#   eval_content_ngram_overlap     a verbatim >=8-token n-gram shared with an eval
#                                  row (pool = every possible draw, core files,
#                                  sentinels) or with the contrastive trace bank
#                                  rendered into the proposer's context.
#   eval_expected_answer_leakage   an exact expected answer of an eval row.
#   eval_source_identity_leakage   content derived from a trace of an eval
#                                  question: the trace resolves to a question id
#                                  and the mutation shares >=3 rare tokens with
#                                  it (paraphrase survives this, not n-grams).
#
# Each reason records the matched source id. Tokens are casefolded runs of
# [A-Za-z0-9_] and non-ASCII bytes, so the match ignores punctuation/whitespace.
#
# INDEX (built in the same pass as the id vocabulary, same cache/fail-closed):
#   shingles  64-bit hashed 8-token shingles packed as (hash >> b) << b | row,
#             b = bit width of the row count (17 for the 79,479-row pool), so a
#             47-bit hash survives. False-positive rate per looked-up shingle is
#             about N / 2**47 (N = 18.5M -> 1.3e-7; ~7e-5 for a 500-shingle
#             mutation). A shingle present in >= 32 distinct rows is TEMPLATE text
#             (answer-format boilerplate) and is dropped. A hit counts when its 8
#             tokens include a fresh ANCHOR token (below) or it lies in a run of >= 8
#             consecutive hits (a >= 15-token verbatim copy). Rows over 32 KiB (1,417
#             long-context rows) keep their first/last 512 shingles exactly and a
#             width-16 winnowed sample in between, which guarantees detection of a
#             shared run of >= 8 + 16 - 1 = 23 tokens there.
#   answers   the same packing over whole expected answers of 1-7 tokens (longer
#             answers are covered by the shingles) that carry an ANCHOR token which
#             is also name-shaped in the answer (upper case, digit, underscore,
#             non-ASCII) or, in a multi-word answer, near-unique (pool tf <= 2), and
#             that are shared by < 32 rows.
#   sketch    token-frequency count-min sketch (2 x 2**22, saturating uint16).
#             ANCHOR token: >= 4 bytes, not all digits, pool frequency <= 200.
#             Bare numbers and common words never anchor anything.
# HARNESS EXCLUSION: n-grams and anchor tokens already present in the current
# prompts, the orchestrator's own code (src/**/*.py, which contains the allowlist)
# or the original content are never blamed on a mutation: text the harness
# already carries is not newly introduced (this is what keeps import/boilerplate
# lines of the long-context code documents from matching).
# Measured cost and false-positive rates: meta-harness-operator-guide.md § 7.
# ---------------------------------------------------------------------------

_CONTENT_NGRAM = 8
_CONTENT_TEMPLATE_DF = 32
_CONTENT_LONG_ROW_BYTES = 32_768
_CONTENT_LONG_EDGE_SHINGLES = 512
_CONTENT_WINNOW = 16
_CONTENT_CHUNK_BYTES = 1 << 21
_CONTENT_SKETCH_BITS = 22
_CONTENT_SKETCH_FLUSH = 1 << 24
_CONTENT_RARE_TF = 200
_CONTENT_MAX_ROWS = 1 << 24
_ANSWER_MAX_TOKENS = _CONTENT_NGRAM - 1
_ANSWER_MIN_BYTES = 4
_ANCHOR_MIN_BYTES = 4
_ANSWER_UNSHAPED_MAX_TF = 2
_SOURCE_MIN_SHINGLE_HITS = 2
# An n-gram hit whose 8 tokens are all common (no fresh anchor) counts only inside a run
# of >= 8 consecutive hit shingles, i.e. a >= 15-token verbatim copy: shared code
# boilerplate ("import os import re from pathlib import path") is not an eval instance.
_CONTENT_MIN_UNANCHORED_RUN = 8
_SOURCE_MIN_SHARED_ANCHORS = 3
_SOURCE_MAX_QIDS = 3
_ANSWER_PLACEHOLDER_RE = re.compile(r"^__\w+__$")
_HASH_PRIME = 0x100000001B3
_HASH_MASK = (1 << 64) - 1
_HASH_M1 = 0xBF58476D1CE4E5B9
_HASH_M2 = 0x94D049BB133111EB
_HASH_LEN_SALT = 0x9E3779B97F4A7C15
_HASH_SEQ_SALT = 0xD6E8FEB86659FD93
_HASH_CONSTS: dict[str, Any] = {}


def _np():
    import numpy

    return numpy


def _hash_consts() -> dict[str, Any]:
    if not _HASH_CONSTS:
        np = _np()
        word = np.zeros(256, dtype=bool)
        for lo, hi in ((48, 57), (65, 90), (97, 122), (95, 95), (128, 255)):
            word[lo : hi + 1] = True
        digit = np.zeros(256, dtype=bool)
        digit[48:58] = True
        _HASH_CONSTS.update(
            word=word,
            digit=digit,
            pinv=pow(_HASH_PRIME, -1, 1 << 64),
            pows={
                n: np.array(
                    [pow(_HASH_PRIME, n - 1 - i, 1 << 64) for i in range(n)], dtype=np.uint64
                )
                for n in range(1, _CONTENT_NGRAM + 1)
            },
        )
    return _HASH_CONSTS


def _mix64(x):
    np = _np()
    u = np.uint64
    x = x ^ (x >> u(30))
    x *= u(_HASH_M1)
    x ^= x >> u(27)
    x *= u(_HASH_M2)
    x ^= x >> u(31)
    return x


def _hash_token_buffer(buf: bytes):
    """(starts, ends, hashes) of the tokens of an already casefolded byte buffer.

    Vectorised polynomial hash: prefix sums of byte * P**i, divided by P**start
    via the modular inverse (P is odd), so each token hashes independently of its
    position. Integer overflow wraps modulo 2**64 by design.
    """
    np = _np()
    consts = _hash_consts()
    arr = np.frombuffer(buf, dtype=np.uint8)
    n = arr.size
    empty = np.empty(0, dtype=np.int64)
    if n == 0:
        return empty, empty, np.empty(0, dtype=np.uint64)
    mask = consts["word"][arr]
    edge = np.diff(mask.view(np.int8), prepend=np.int8(0), append=np.int8(0))
    starts = np.flatnonzero(edge == 1)
    ends = np.flatnonzero(edge == -1)
    if starts.size == 0:
        return starts, ends, np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        powers = np.full(n, _HASH_PRIME, dtype=np.uint64)
        powers[0] = 1
        np.cumprod(powers, out=powers)
        prefix = arr.astype(np.uint64)
        prefix += np.uint64(1)
        prefix *= powers
        del powers
        np.cumsum(prefix, out=prefix)
        raw = prefix[ends - 1] - np.where(
            starts > 0, prefix[np.maximum(starts - 1, 0)], np.uint64(0)
        )
        del prefix
        inverse = np.full(n, consts["pinv"], dtype=np.uint64)
        inverse[0] = 1
        np.cumprod(inverse, out=inverse)
        raw *= inverse[starts]
        del inverse
        raw ^= (ends - starts).astype(np.uint64) * np.uint64(_HASH_LEN_SALT)
        return starts, ends, _mix64(raw)


def _content_tokens(text: str):
    """(buffer, starts, ends, hashes) for ``text``; giant texts are hashed in pieces."""
    np = _np()
    buf = (text or "").encode("utf-8", errors="replace").lower()
    if len(buf) <= _CONTENT_CHUNK_BYTES:
        return (buf, *_hash_token_buffer(buf))
    word = _hash_consts()["word"]
    parts = []
    pos = 0
    while pos < len(buf):
        cut = min(len(buf), pos + _CONTENT_CHUNK_BYTES)
        while pos < cut < len(buf) and word[buf[cut - 1]] and word[buf[cut]]:
            cut -= 1
        if cut == pos:  # one token longer than a piece: split it
            cut = min(len(buf), pos + _CONTENT_CHUNK_BYTES)
        s, e, h = _hash_token_buffer(buf[pos:cut])
        parts.append((s + pos, e + pos, h))
        pos = cut
    return (
        buf,
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
    )


def _window_hashes(windows, length: int):
    """Hash rows of a (k, length) token-hash matrix; shingles when length == N-gram."""
    np = _np()
    with np.errstate(over="ignore"):
        combined = windows @ _hash_consts()["pows"][length]
        if length != _CONTENT_NGRAM:
            combined ^= np.uint64((length * _HASH_SEQ_SALT) & _HASH_MASK)
        return _mix64(combined)


def _sequence_hashes(token_hashes, length: int):
    """Hash of every ``length``-token window of ``token_hashes``."""
    np = _np()
    if token_hashes.size < length:
        return np.empty(0, dtype=np.uint64)
    return _window_hashes(np.lib.stride_tricks.sliding_window_view(token_hashes, length), length)


def _packed_lookup(packed, hashes, row_bits: int):
    """Row index per hash (-1 when absent) in a (hash >> b) << b | row sorted array."""
    np = _np()
    if packed is None or packed.size == 0 or hashes.size == 0:
        return np.full(hashes.size, -1, dtype=np.int64)
    shift = np.uint64(row_bits)
    top = hashes >> shift
    pos = np.searchsorted(packed, top << shift)
    clipped = np.minimum(pos, packed.size - 1)
    found = (pos < packed.size) & ((packed[clipped] >> shift) == top)
    rows = (packed[clipped] & np.uint64((1 << row_bits) - 1)).astype(np.int64)
    return np.where(found, rows, -1)


def _row_label_id(row: dict) -> str:
    for key in ("id", "qid", "stable_qid", "question_id"):
        raw = str(row.get(key) or "").strip()
        if raw:
            return raw
    return stable_question_qid(str(row.get("suite", "unknown")), str(row.get("prompt") or ""))


def _anchor_mask(buf: bytes, starts, ends, hashes, sketch):
    """Tokens specific enough to tie text to one eval item (see the notes above)."""
    np = _np()
    if hashes.size == 0:
        return np.zeros(0, dtype=bool)
    lengths = ends - starts
    digits = np.concatenate(
        ([0], np.cumsum(_hash_consts()["digit"][np.frombuffer(buf, dtype=np.uint8)]))
    )
    numeric = (digits[ends] - digits[starts]) == lengths
    mask = np.uint64((1 << _CONTENT_SKETCH_BITS) - 1)
    frequency = np.minimum(
        sketch[0][(hashes & mask).astype(np.intp)],
        sketch[1][((hashes >> np.uint64(40)) & mask).astype(np.intp)],
    )
    return (lengths >= _ANCHOR_MIN_BYTES) & ~numeric & (frequency <= _CONTENT_RARE_TF)


@dataclass(frozen=True, eq=False)
class EvalContentIndex:
    """Hashed eval content: shingles, expected answers, token frequencies (see above)."""

    shingles: Any
    answers: Any
    answer_lengths: tuple[int, ...]
    row_bits: int
    sketch: tuple[Any, Any]
    labels: tuple[str, ...]
    stats: tuple[tuple[str, Any], ...] = ()

    @property
    def nbytes(self) -> int:
        """Array bytes plus an estimate for the row-label strings."""
        return int(
            self.shingles.nbytes
            + self.answers.nbytes
            + sum(s.nbytes for s in self.sketch)
            + sum(len(label) + 49 for label in self.labels)
        )

    def label(self, row: int) -> str:
        return self.labels[row] if 0 <= row < len(self.labels) else f"row:{row}"

    def shingle_rows(self, hashes):
        return _packed_lookup(self.shingles, hashes, self.row_bits)

    def answer_rows(self, hashes):
        return _packed_lookup(self.answers, hashes, self.row_bits)

    def anchor_mask(self, buf: bytes, starts, ends, hashes):
        return _anchor_mask(buf, starts, ends, hashes, self.sketch)


class _EvalContentIndexBuilder:
    """Streams eval rows into an :class:`EvalContentIndex` with bounded memory."""

    def __init__(self) -> None:
        np = _np()
        self._np = np
        self.labels: list[str] = []
        self._source = ""
        self._segments: list[bytes] = []
        self._segment_rows: list[int] = []
        self._pending_bytes = 0
        self._keys: list[Any] = []
        self._rows: list[Any] = []
        self._sketch = (
            np.zeros(1 << _CONTENT_SKETCH_BITS, dtype=np.uint32),
            np.zeros(1 << _CONTENT_SKETCH_BITS, dtype=np.uint32),
        )
        self._sketch_pending: list[Any] = []
        self._sketch_pending_n = 0
        self._answers: list[tuple[int, str]] = []
        self._tokens = 0
        self._long_rows = 0
        self._started = time.monotonic()

    def begin_source(self, label: str) -> None:
        self._source = label

    def add_row(self, row: dict) -> None:
        index = len(self.labels)
        if index >= _CONTENT_MAX_ROWS:
            raise ValueError(f"eval content index row limit {_CONTENT_MAX_ROWS} exceeded")
        rid = _row_label_id(row)
        self.labels.append(f"{self._source}:{rid}" if self._source else rid)
        expected = row.get("expected")
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            expected = str(expected)
        if isinstance(expected, str) and expected.strip():
            self._answers.append((index, expected))
        for text in (row.get("prompt"), expected):
            if not isinstance(text, str) or not text:
                continue
            data = text.encode("utf-8", errors="replace").lower()
            if len(data) > _CONTENT_LONG_ROW_BYTES:
                self._flush()
                self._add_long(text, index)
                continue
            self._segments.append(data)
            self._segment_rows.append(index)
            self._pending_bytes += len(data) + 1
            if self._pending_bytes >= _CONTENT_CHUNK_BYTES:
                self._flush()

    def _count_tokens(self, hashes) -> None:
        self._tokens += int(hashes.size)
        self._sketch_pending.append(hashes)
        self._sketch_pending_n += int(hashes.size)
        if self._sketch_pending_n >= _CONTENT_SKETCH_FLUSH:
            self._flush_sketch()

    def _flush_sketch(self) -> None:
        np = self._np
        if not self._sketch_pending:
            return
        values = np.concatenate(self._sketch_pending)
        self._sketch_pending = []
        self._sketch_pending_n = 0
        mask = np.uint64((1 << _CONTENT_SKETCH_BITS) - 1)
        for table, part in (
            (self._sketch[0], values & mask),
            (self._sketch[1], (values >> np.uint64(40)) & mask),
        ):
            table += np.bincount(part.astype(np.intp), minlength=table.size).astype(np.uint32)

    def _emit(self, keys, rows) -> None:
        np = self._np
        if keys.size == 0:
            return
        order = np.lexsort((rows, keys))
        keys = keys[order]
        rows = rows[order]
        keep = np.ones(keys.size, dtype=bool)
        keep[1:] = (keys[1:] != keys[:-1]) | (rows[1:] != rows[:-1])
        self._keys.append(keys[keep])
        self._rows.append(rows[keep])

    def _flush(self) -> None:
        np = self._np
        if not self._segments:
            return
        buf = b"\x00".join(self._segments)
        lengths = np.fromiter((len(s) + 1 for s in self._segments), np.int64, len(self._segments))
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        segment_rows = np.asarray(self._segment_rows, dtype=np.uint32)
        self._segments, self._segment_rows, self._pending_bytes = [], [], 0
        starts, _ends, hashes = _hash_token_buffer(buf)
        if hashes.size == 0:
            return
        self._count_tokens(hashes)
        if hashes.size < _CONTENT_NGRAM:
            return
        segment = np.searchsorted(offsets, starts, side="right") - 1
        span = _CONTENT_NGRAM - 1
        valid = segment[:-span] == segment[span:]
        shingles = _sequence_hashes(hashes, _CONTENT_NGRAM)
        self._emit(shingles[valid], segment_rows[segment[:-span][valid]])

    def _add_long(self, text: str, index: int) -> None:
        np = self._np
        self._long_rows += 1
        _buf, _starts, _ends, hashes = _content_tokens(text)
        self._count_tokens(hashes)
        shingles = _sequence_hashes(hashes, _CONTENT_NGRAM)
        n = shingles.size
        if n == 0:
            return
        keep = np.zeros(n, dtype=bool)
        keep[:_CONTENT_LONG_EDGE_SHINGLES] = True
        keep[max(0, n - _CONTENT_LONG_EDGE_SHINGLES) :] = True
        if n > _CONTENT_WINNOW:
            windows = np.lib.stride_tricks.sliding_window_view(shingles, _CONTENT_WINNOW)
            keep[np.arange(windows.shape[0]) + windows.argmin(axis=1)] = True
        else:
            keep[:] = True
        self._emit(shingles[keep], np.full(int(keep.sum()), index, dtype=np.uint32))

    def _pack(self, keys_list: list, rows_list: list, row_bits: int):
        """Pack (hash, row) pairs, consuming the input lists to bound peak memory."""
        np = self._np
        shift = np.uint64(row_bits)
        packed = np.empty(sum(k.size for k in keys_list), dtype=np.uint64)
        pos = 0
        keys_list.reverse()
        rows_list.reverse()
        while keys_list:
            keys = keys_list.pop()
            rows = rows_list.pop()
            packed[pos : pos + keys.size] = ((keys >> shift) << shift) | rows.astype(np.uint64)
            pos += keys.size
        packed.sort()
        return packed

    def _drop_template(self, packed, row_bits: int):
        """Keep one entry (lowest row) per hash seen in < _CONTENT_TEMPLATE_DF rows."""
        np = self._np
        if packed.size == 0:
            return packed
        shift = np.uint64(row_bits)
        head = np.ones(packed.size, dtype=bool)
        step = 1 << 22
        for lo in range(1, packed.size, step):
            hi = min(packed.size, lo + step)
            head[lo:hi] = (packed[lo:hi] >> shift) != (packed[lo - 1 : hi - 1] >> shift)
        starts = np.flatnonzero(head)
        runs = np.diff(np.append(starts, packed.size))
        return packed[starts[runs < _CONTENT_TEMPLATE_DF]]

    def _answer_entries(self, sketch, row_bits: int):
        np = self._np
        empty = np.empty(0, dtype=np.uint64)
        if not self._answers:
            return empty, ()
        texts = [text.strip() for _row, text in self._answers]
        encoded = [t.encode("utf-8", errors="replace").lower() for t in texts]
        rows = np.fromiter((row for row, _text in self._answers), np.int64, len(texts))
        buf = b"\x00".join(encoded)
        starts, ends, hashes = _hash_token_buffer(buf)
        if hashes.size == 0:
            return empty, ()
        sizes = np.fromiter((len(e) for e in encoded), np.int64, len(encoded))
        offsets = np.concatenate(([0], np.cumsum(sizes + 1)[:-1]))
        segment = np.searchsorted(offsets, starts, side="right") - 1
        counts = np.bincount(segment, minlength=len(texts))
        first = np.searchsorted(segment, np.arange(len(texts)))
        # An answer anchor must also be NAME-shaped in the original text (an upper-case
        # letter, digit, underscore or non-ASCII byte) or near-unique in the pool (tf <= 2):
        # plain lower-case words such as "contention" are ordinary vocabulary.
        cased = np.frombuffer(b"\x00".join(t.encode("utf-8", errors="replace") for t in texts), np.uint8)
        shape = (cased >= 65) & (cased <= 90) | (cased >= 128) | (cased == 95)
        shape |= _hash_consts()["digit"][cased]
        shape_prefix = np.concatenate(([0], np.cumsum(shape)))
        shaped = (shape_prefix[ends] - shape_prefix[starts]) > 0
        mask = np.uint64((1 << _CONTENT_SKETCH_BITS) - 1)
        unique_tf = np.minimum(
            sketch[0][(hashes & mask).astype(np.intp)],
            sketch[1][((hashes >> np.uint64(40)) & mask).astype(np.intp)],
        ) <= _ANSWER_UNSHAPED_MAX_TF
        single = counts[segment] == 1  # a one-word answer must itself be name-shaped
        anchors = _anchor_mask(buf, starts, ends, hashes, sketch)
        anchors &= shaped | (unique_tf & ~single)
        anchors = anchors.astype(np.int64)
        anchor_prefix = np.concatenate(([0], np.cumsum(anchors)))
        tail = np.minimum(first + counts, anchors.size)
        eligible = (sizes >= _ANSWER_MIN_BYTES) & (counts >= 1) & (counts <= _ANSWER_MAX_TOKENS)
        eligible &= (anchor_prefix[tail] - anchor_prefix[np.minimum(first, anchors.size)]) > 0
        eligible &= np.fromiter(
            (not _ANSWER_PLACEHOLDER_RE.match(t) for t in texts), bool, len(texts)
        )
        keys, key_rows, used = [], [], []
        for length in range(1, _ANSWER_MAX_TOKENS + 1):
            chosen = np.flatnonzero(eligible & (counts == length))
            if chosen.size == 0:
                continue
            windows = hashes[first[chosen][:, None] + np.arange(length)[None, :]]
            keys.append(_window_hashes(windows, length))
            key_rows.append(rows[chosen].astype(np.uint32))
            used.append(length)
        if not keys:
            return empty, ()
        return self._drop_template(self._pack(keys, key_rows, row_bits), row_bits), tuple(used)

    def build(self) -> EvalContentIndex:
        np = self._np
        self._flush()
        self._flush_sketch()
        row_bits = max(1, (max(1, len(self.labels)) - 1).bit_length())
        row_shingles = int(sum(k.size for k in self._keys))
        shingles = self._drop_template(self._pack(self._keys, self._rows, row_bits), row_bits)
        sketch = tuple(np.minimum(t, 65535).astype(np.uint16) for t in self._sketch)
        self._sketch = ()
        answers, answer_lengths = self._answer_entries(sketch, row_bits)
        self._answers = []
        return EvalContentIndex(
            shingles=shingles,
            answers=answers,
            answer_lengths=answer_lengths,
            row_bits=row_bits,
            sketch=sketch,
            labels=tuple(self.labels),
            stats=(
                ("rows", len(self.labels)),
                ("tokens", self._tokens),
                ("long_rows", self._long_rows),
                ("row_shingles", row_shingles),
                ("shingles", int(shingles.size)),
                ("answers", int(answers.size)),
                ("build_s", round(time.monotonic() - self._started, 2)),
            ),
        )


@dataclass(frozen=True)
class EvalIdVocabulary:
    """Casefolded eval-instance identifiers plus the id families derived from them."""

    ids: frozenset[str] = frozenset()
    family_re: re.Pattern[str] | None = None
    anchored_re: re.Pattern[str] | None = None
    sources: tuple[str, ...] = ()
    error: str = ""
    # 2026-09-15 structural form: eval content index and the suite names it saw.
    # A vocabulary without a content index is NOT available (fail-closed).
    content: EvalContentIndex | None = None
    suites: frozenset[str] = frozenset()

    @property
    def available(self) -> bool:
        return not self.error and bool(self.ids) and self.content is not None

    @classmethod
    def from_rows(
        cls,
        rows: Any,
        *,
        sources: tuple[str, ...] = (),
        content_builder: _EvalContentIndexBuilder | None = None,
    ) -> EvalIdVocabulary:
        builder = content_builder if content_builder is not None else _EvalContentIndexBuilder()
        ids: set[str] = set()
        stems: set[str] = set()
        suites: set[str] = set()
        family_counts: dict[str, int] = {}
        native_families: set[str] = set()
        for row in rows:
            builder.add_row(row)
            suite = str(row.get("suite") or "").strip()
            if suite:
                suites.add(suite.casefold())
            if len(suite) >= _LEAKAGE_MIN_STEM:
                stems.add(suite.casefold())
            prompt = row.get("prompt")
            if isinstance(prompt, str) and prompt:
                # The prompt-hash qid journals and failure context carry.
                ids.add(stable_question_qid(str(row.get("suite", "unknown")), prompt))
            for nested in row.values():
                if isinstance(nested, dict):  # e.g. core files' ``core_selection``
                    for key in _EVAL_ID_KEYS[1:]:
                        raw = str(nested.get(key) or "").strip()
                        if _is_identifier_shaped(raw):
                            ids.add(raw.casefold())
            for key in _EVAL_ID_KEYS:
                raw = str(row.get(key) or "").strip()
                if not _is_identifier_shaped(raw):
                    continue
                ids.add(raw.casefold())
                if key != "id":
                    continue
                tail = _ID_FAMILY_TAIL_RE.match(raw)
                if tail and len(tail.group(1)) >= _LEAKAGE_MIN_FAMILY_PREFIX:
                    prefix = tail.group(1).casefold()
                    family_counts[prefix] = family_counts.get(prefix, 0) + 1
                for native in _ID_NATIVE_FAMILY_RE.finditer(raw):
                    native_families.add(native.group(1).casefold())
                stem = _ID_STEM_RE.match(raw)
                if stem and len(stem.group(1)) >= _LEAKAGE_MIN_STEM:
                    stems.add(stem.group(1).casefold())
        families = {p for p, n in family_counts.items() if n >= _LEAKAGE_MIN_FAMILY_MEMBERS}
        families |= native_families
        family_re = None
        if families:
            family_re = re.compile(
                r"(?<![\w])(?:"
                + "|".join(re.escape(p) for p in sorted(families, key=len, reverse=True))
                + r")(?:\d+|[0-9a-f]{8,})(?![\w])",
                re.IGNORECASE,
            )
        anchored_re = None
        if stems:
            anchored_re = re.compile(
                r"(?<![\w])(?:"
                + "|".join(re.escape(s) for s in sorted(stems, key=len, reverse=True))
                + r")(?:[\s_-]*(?:problem|question|task|sample|item|instance)s?\s*#?|\s*#)"
                r"\s*\d{1,6}\b",
                re.IGNORECASE,
            )
        return cls(
            ids=frozenset(ids),
            family_re=family_re,
            anchored_re=anchored_re,
            sources=sources,
            error="" if ids else "no_eval_ids_found",
            content=builder.build(),
            suites=frozenset(suites),
        )

    def find_leaks(self, text: str) -> list[str]:
        """Instance references in ``text``: exact ids, id-family members, anchored refs."""
        folded = (text or "").casefold()
        if not folded:
            return []
        hits: list[str] = []
        lengths = sorted({len(i) for i in self.ids})
        n = len(folded)
        for start in range(n):
            if start and _is_word_char(folded[start - 1]):
                continue
            for length in lengths:
                end = start + length
                if end > n:
                    break
                if end < n and _is_word_char(folded[end]):
                    continue
                candidate = folded[start:end]
                if candidate in self.ids:
                    hits.append(candidate)
        for pattern in (self.family_re, self.anchored_re, _LEAKAGE_GENERIC_RE):
            if pattern is not None:
                hits.extend(m.group(0).casefold() for m in pattern.finditer(folded))
        return list(dict.fromkeys(hits))


_EVAL_ID_VOCAB_CACHE: dict[tuple, EvalIdVocabulary] = {}
# Failed builds, keyed by the same file identity (path, mtime_ns, size) and held for a
# short TTL. Without it a MALFORMED 1.35 GB pool is re-parsed on every mutation. Keyed by
# file identity, so restoring or editing the file misses the entry at once; the TTL bounds
# the wait for a fix that does not change identity (e.g. a chmod).
EVAL_ID_VOCAB_NEG_TTL_ENV = "AUTOPILOT_EVAL_ID_VOCAB_NEG_TTL_S"
DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S = 60.0
_EVAL_ID_VOCAB_NEG_CACHE: dict[tuple, tuple[float, EvalIdVocabulary]] = {}


def _eval_id_vocab_neg_ttl_s() -> float:
    raw = os.environ.get(EVAL_ID_VOCAB_NEG_TTL_ENV, "").strip()
    try:
        value = float(raw) if raw else DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S
    except ValueError:
        return DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S
    return value if value >= 0 and math.isfinite(value) else DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S


def clear_eval_id_vocabulary_cache() -> None:
    """Drop both the built-vocabulary cache and the failed-build cache."""
    _EVAL_ID_VOCAB_CACHE.clear()
    _EVAL_ID_VOCAB_NEG_CACHE.clear()


def describe_eval_id_sources(
    sources: tuple[tuple[Path, bool], ...] | None = None,
) -> list[dict[str, Any]]:
    """Resolved vocabulary sources with their on-disk status (for operator messages)."""
    resolved = sources if sources is not None else _default_eval_id_sources()
    out: list[dict[str, Any]] = []
    for path, required in resolved:
        entry: dict[str, Any] = {"path": str(path), "required": bool(required)}
        try:
            stat = path.stat()
            entry.update(exists=True, size=stat.st_size, readable=os.access(path, os.R_OK))
        except OSError:
            entry.update(exists=False, size=None, readable=False)
        out.append(entry)
    return out


# Observer of every leakage-guard verdict's vocabulary state: ``fn(available, error)``.
# The autopilot installs its operability monitor here (startup preflight + circuit alarm).
_EVAL_LEAKAGE_OBSERVER: Callable[[bool, str], None] | None = None


def set_eval_leakage_observer(observer: Callable[[bool, str], None] | None) -> None:
    global _EVAL_LEAKAGE_OBSERVER
    _EVAL_LEAKAGE_OBSERVER = observer


def _notify_eval_leakage_observer(vocabulary: EvalIdVocabulary) -> None:
    observer = _EVAL_LEAKAGE_OBSERVER
    if observer is None:
        return
    try:
        observer(vocabulary.available, _vocabulary_error(vocabulary))
    except Exception as exc:  # noqa: BLE001 - an observer must never change a verdict
        log.warning("eval-leakage observer failed: %s", exc)


def load_eval_id_vocabulary(
    sources: tuple[tuple[Path, bool], ...] | None = None,
) -> EvalIdVocabulary:
    """Build (and cache by file identity) the eval-id vocabulary. Never raises.

    A missing REQUIRED source, an unreadable source, or an empty vocabulary
    returns a vocabulary whose ``error`` is set; the guard then fails closed.
    """
    resolved = sources if sources is not None else _default_eval_id_sources()
    if not resolved:
        return EvalIdVocabulary(error="no_eval_id_sources")
    key_parts: list[tuple] = []
    present: list[Path] = []
    for path, required in resolved:
        try:
            stat = path.stat()
        except OSError:
            if required:
                return EvalIdVocabulary(error=f"missing_eval_id_source:{path}")
            continue
        key_parts.append((str(path), stat.st_mtime_ns, stat.st_size))
        present.append(path)
    cache_key = tuple(key_parts)
    cached = _EVAL_ID_VOCAB_CACHE.get(cache_key)
    if cached is not None:
        return cached
    negative = _EVAL_ID_VOCAB_NEG_CACHE.get(cache_key)
    if negative is not None and time.monotonic() - negative[0] < _eval_id_vocab_neg_ttl_s():
        return negative[1]

    builder_holder: list[_EvalContentIndexBuilder] = []

    def _rows():
        for path in present:
            builder_holder[0].begin_source(path.name)
            yield from _iter_eval_rows(path)

    try:
        builder_holder.append(_EvalContentIndexBuilder())
        vocab = EvalIdVocabulary.from_rows(
            _rows(),
            sources=tuple(str(p) for p in present),
            content_builder=builder_holder[0],
        )
    except Exception as exc:  # noqa: BLE001 - any parse failure must fail CLOSED
        vocab = EvalIdVocabulary(error=_truncate_reason(f"eval_id_source_unreadable:{exc}"))
    if vocab.available:
        _EVAL_ID_VOCAB_CACHE.clear()
        _EVAL_ID_VOCAB_CACHE[cache_key] = vocab
        _EVAL_ID_VOCAB_NEG_CACHE.clear()
    else:
        _EVAL_ID_VOCAB_NEG_CACHE.clear()
        _EVAL_ID_VOCAB_NEG_CACHE[cache_key] = (time.monotonic(), vocab)
    return vocab


def _vocabulary_error(vocabulary: EvalIdVocabulary) -> str:
    if vocabulary.error:
        return vocabulary.error
    if vocabulary.ids and vocabulary.content is None:
        return "eval_content_index_missing"
    return "" if vocabulary.available else "empty"


def eval_leakage_reason(
    text: str,
    vocabulary: EvalIdVocabulary,
    *,
    original: str = "",
    trace_context: str = "",
    harness: _HarnessContent | None = None,
) -> str | None:
    """Fail-closed MHS-3 verdict: a rejection reason, or None when ``text`` is clean.

    ``text`` is the ADDED text. Order: vocabulary availability, instance ids, then
    the content refusals (n-gram, expected answer, source identity). ``original``
    and the harness corpus are excluded from the content checks; ``trace_context``
    is the proposer's failure context carrying the rendered trace bank.
    """
    _notify_eval_leakage_observer(vocabulary)
    if not vocabulary.available:
        return _truncate_reason(f"eval_leakage_vocabulary_unavailable:{_vocabulary_error(vocabulary)}")
    leaks = vocabulary.find_leaks(text)
    if leaks:
        shown = [leak[:48] for leak in leaks[:_LEAKAGE_MAX_REPORTED]]
        return _truncate_reason(f"eval_instance_leakage: refs={shown} total={len(leaks)}")
    try:
        return eval_content_leakage_reason(
            text,
            vocabulary,
            original=original,
            trace_context=trace_context,
            harness=harness,
        )
    except Exception as exc:  # noqa: BLE001 - a broken content check must fail CLOSED
        log.error("eval content leakage check failed: %s", exc)
        return _truncate_reason(f"eval_leakage_vocabulary_unavailable:content_check_failed:{exc}")


# ---------------------------------------------------------------------------
# RTG-55 MHS-3 per-call content checks (see the EvalContentIndex notes above).
# ---------------------------------------------------------------------------

_TRACE_SECTION_RE = re.compile(
    r"^## (Contrastive Execution Traces|Harness Trace IR|Recent Execution Traces)[^\n]*$",
    re.MULTILINE,
)
_CONTEXT_SECTION_END_RE = re.compile(
    r"^(?:## (?:Contrastive Execution Traces|Harness Trace IR|Recent Execution Traces|"
    r"PromptForge Convention Guardrails|Diversity Coverage Pressure|Cross-Species Insights|"
    r"Past Strategy Insights|Previously Rejected)|Trial #\d+ \()",
    re.MULTILINE,
)
_CONTRASTIVE_ENTRY_RE = re.compile(r"^\[\d+\] [^\n]*$", re.MULTILINE)
_TRACE_IR_JSON_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
_TRACE_QID_FIELD_RE = re.compile(
    r"\b(?:qid|question_id|sentinel_id|stable_qid)\b[\"']?\s*[:=]\s*[\"']?"
    r"([A-Za-z0-9][A-Za-z0-9_./:+-]{3,})",
    re.IGNORECASE,
)
_TRACE_ID_FIELDS = ("qid", "question_id", "sentinel_id", "stable_qid")


def _trace_ir_blocks(body: str) -> list[tuple[str, str]]:
    match = _TRACE_IR_JSON_RE.search(body)
    try:
        data = json.loads(match.group(1)) if match else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return [("trace_ir", body)] if body.strip() else []
    blocks: list[tuple[str, str]] = []
    for example in data.get("trace_examples") or []:
        if not isinstance(example, dict):
            continue
        label = " ".join(
            str(part)
            for part in (
                f"trial #{example.get('trial_id')}",
                example.get("outcome") or "",
                example.get("trace_hash") or "",
            )
            if part
        )
        steps = [
            str(step.get("content_preview") or "")
            for step in example.get("steps") or []
            if isinstance(step, dict)
        ]
        ids = {key: example[key] for key in _TRACE_ID_FIELDS if example.get(key)}
        blocks.append(
            (label, "\n".join([*steps, str(example.get("reason") or ""), json.dumps(ids)]))
        )
    return blocks


def _trace_blocks(context: str) -> list[tuple[str, str]]:
    """Per-example (label, text) blocks of the trace bank rendered into ``context``.

    Understands the three renderings the proposer receives (actions.py): MH-7
    contrastive traces, MH-11 trace IR JSON, and the raw recent-trace fallback.
    """
    blocks: list[tuple[str, str]] = []
    for heading in _TRACE_SECTION_RE.finditer(context or ""):
        end = _CONTEXT_SECTION_END_RE.search(context, heading.end())
        body = context[heading.end() : end.start() if end else len(context)]
        kind = heading.group(1)
        if kind.startswith("Harness Trace IR"):
            blocks.extend(_trace_ir_blocks(body))
        elif kind.startswith("Contrastive"):
            entries = list(_CONTRASTIVE_ENTRY_RE.finditer(body))
            if not entries and body.strip():
                blocks.append(("contrastive", body))
            for i, entry in enumerate(entries):
                stop = entries[i + 1].start() if i + 1 < len(entries) else len(body)
                blocks.append((entry.group(0), body[entry.end() : stop]))
        elif body.strip():
            blocks.append(("recent_traces", body))
    return [(" ".join(label.split())[:48], text) for label, text in blocks]


@dataclass(frozen=True, eq=False)
class _HarnessContent:
    """Sorted unique shingles and tokens of text the harness already carries."""

    shingles: Any
    tokens: Any
    token_files: Any = None  # per ``tokens`` entry: how many harness files contain it

    def file_count(self, token_hash) -> int:
        np = _np()
        if self.token_files is None or self.tokens.size == 0:
            return 0
        value = np.asarray([token_hash], dtype=np.uint64)
        pos = int(min(np.searchsorted(self.tokens, value)[0], self.tokens.size - 1))
        return int(self.token_files[pos]) if self.tokens[pos] == value[0] else 0


_HARNESS_CONTENT_CACHE: dict[tuple, _HarnessContent] = {}


def _text_shingles_tokens(texts) -> _HarnessContent:
    np = _np()
    shingles, tokens = [], []
    for text in texts:
        _buf, _s, _e, hashes = _content_tokens(text)
        tokens.append(np.unique(hashes))
        shingles.append(_sequence_hashes(hashes, _CONTENT_NGRAM))
    if not tokens:
        empty = np.empty(0, dtype=np.uint64)
        return _HarnessContent(shingles=empty, tokens=empty, token_files=np.empty(0, np.int64))
    unique_tokens, token_files = np.unique(np.concatenate(tokens), return_counts=True)
    return _HarnessContent(
        shingles=np.unique(np.concatenate(shingles)),
        tokens=unique_tokens,
        token_files=token_files,
    )


def harness_content(prompts_dir: Path | None = None) -> _HarnessContent:
    """Shingles/tokens of the current prompts and the orchestrator code (cached by file identity)."""
    root = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    files = sorted(root.rglob("*.md")) if root.is_dir() else []
    code_root = NEW_FILE_MUTATION_ROOT
    files += sorted(code_root.rglob("*.py")) if code_root.is_dir() else []
    files += [PROJECT_ROOT / rel for rel in CODE_MUTATION_ALLOWLIST]
    files = list(dict.fromkeys(files))
    key_parts = []
    present = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        key_parts.append((str(path), stat.st_mtime_ns, stat.st_size))
        present.append(path)
    key = tuple(key_parts)
    cached = _HARNESS_CONTENT_CACHE.get(key)
    if cached is None:
        texts = []
        for path in present:
            try:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        cached = _text_shingles_tokens(texts)
        _HARNESS_CONTENT_CACHE.clear()
        _HARNESS_CONTENT_CACHE[key] = cached
    return cached


def _in_sorted(sorted_values, values):
    np = _np()
    if sorted_values.size == 0 or values.size == 0:
        return np.zeros(values.size, dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_values, values), sorted_values.size - 1)
    return sorted_values[pos] == values


def _span_text(buf: bytes, starts, ends, first: int, last: int) -> str:
    text = buf[int(starts[first]) : int(ends[last])].decode("utf-8", errors="replace")
    return " ".join(text.split())[:60].replace('"', "'")


def _resolve_trace_qids(
    block: str, block_hashes, vocabulary: EvalIdVocabulary
) -> list[str]:
    """Question/sentinel ids a trace block belongs to: explicit fields, exact ids, content."""
    np = _np()
    qids: list[str] = [m.group(1) for m in _TRACE_QID_FIELD_RE.finditer(block)]
    qids += [leak for leak in vocabulary.find_leaks(block) if leak in vocabulary.ids]
    content = vocabulary.content
    if content is not None and block_hashes.size >= _CONTENT_NGRAM:
        rows = content.shingle_rows(_sequence_hashes(block_hashes, _CONTENT_NGRAM))
        rows = rows[rows >= 0]
        if rows.size:
            values, counts = np.unique(rows, return_counts=True)
            for index in np.argsort(-counts, kind="stable"):
                if counts[index] >= _SOURCE_MIN_SHINGLE_HITS:
                    qids.append(content.label(int(values[index])))
    return list(dict.fromkeys(q[:64] for q in qids))[:_SOURCE_MAX_QIDS]


def eval_content_leakage_reason(
    text: str,
    vocabulary: EvalIdVocabulary,
    *,
    original: str = "",
    trace_context: str = "",
    harness: _HarnessContent | None = None,
) -> str | None:
    """The three 2026-09-15 content refusals over added ``text`` (None when clean)."""
    np = _np()
    content = vocabulary.content
    if content is None:
        return "eval_leakage_vocabulary_unavailable:eval_content_index_missing"
    buf, starts, ends, hashes = _content_tokens(text)
    if hashes.size == 0:
        return None
    known = harness if harness is not None else harness_content()
    own = _text_shingles_tokens([original]) if original else None

    def known_shingle(values):
        hit = _in_sorted(known.shingles, values)
        return hit | _in_sorted(own.shingles, values) if own is not None else hit

    def known_token(values):
        hit = _in_sorted(known.tokens, values)
        return hit | _in_sorted(own.tokens, values) if own is not None else hit

    anchors = content.anchor_mask(buf, starts, ends, hashes) & ~known_token(hashes)
    anchor_prefix = np.concatenate(([0], np.cumsum(anchors)))
    shingles = _sequence_hashes(hashes, _CONTENT_NGRAM)
    fresh = ~known_shingle(shingles)
    span = np.arange(shingles.size)
    anchored = (anchor_prefix[span + _CONTENT_NGRAM] - anchor_prefix[span]) > 0

    def identifying(hit):
        """Hits whose window carries a fresh anchor, or that sit in a long verbatim run."""
        if not hit.any():
            return hit
        edges = np.diff(np.concatenate(([0], hit.view(np.int8), [0])))
        run_starts = np.flatnonzero(edges == 1)
        run_ends = np.flatnonzero(edges == -1)
        long_run = np.zeros(hit.size + 1, dtype=np.int64)
        for lo, hi in zip(run_starts, run_ends):
            if hi - lo >= _CONTENT_MIN_UNANCHORED_RUN:
                long_run[lo] += 1
                long_run[hi] -= 1
        return hit & (anchored | (np.cumsum(long_run)[:-1] > 0))

    # 1. Verbatim >= 8-token overlap: eval rows, then the rendered trace bank.
    rows = np.where(fresh, content.shingle_rows(shingles), -1)
    hits = np.flatnonzero(identifying(rows >= 0))
    if hits.size:
        first = int(hits[0])
        return _truncate_reason(
            f"eval_content_ngram_overlap: source={content.label(int(rows[first]))} "
            f'ngram="{_span_text(buf, starts, ends, first, first + _CONTENT_NGRAM - 1)}" '
            f"shared={hits.size}"
        )
    blocks = []
    for label, block in _trace_blocks(trace_context):
        _b, b_starts, b_ends, b_hashes = _content_tokens(block)
        blocks.append((label, block, _b, b_starts, b_ends, b_hashes))
    for label, block, _b, _s, _e, b_hashes in blocks:
        shared = identifying(
            fresh & _in_sorted(np.unique(_sequence_hashes(b_hashes, _CONTENT_NGRAM)), shingles)
        )
        if shared.any():
            first = int(np.flatnonzero(shared)[0])
            qids = _resolve_trace_qids(block, b_hashes, vocabulary)
            return _truncate_reason(
                f"eval_content_ngram_overlap: source=trace_bank[{label}] qids={qids} "
                f'ngram="{_span_text(buf, starts, ends, first, first + _CONTENT_NGRAM - 1)}"'
            )

    # 2. Exact expected answer. The window must carry an anchor token the harness lacks.
    for length in content.answer_lengths if anchors.any() else ():
        rows = content.answer_rows(_sequence_hashes(hashes, length))
        found = np.flatnonzero(rows >= 0)
        found = found[(anchor_prefix[found + length] - anchor_prefix[found]) > 0]
        if found.size:
            first = int(found[0])
            return _truncate_reason(
                f"eval_expected_answer_leakage: source={content.label(int(rows[first]))} "
                f'answer="{_span_text(buf, starts, ends, first, first + length - 1)}"'
            )

    # 3. Source identity: rare tokens shared with a trace that resolves to a question.
    added_anchors = np.unique(hashes[anchors])
    if blocks and added_anchors.size >= _SOURCE_MIN_SHARED_ANCHORS:
        for label, block, b_buf, b_starts, b_ends, b_hashes in blocks:
            b_anchor = content.anchor_mask(b_buf, b_starts, b_ends, b_hashes)
            b_anchor &= ~known_token(b_hashes)
            shared = np.intersect1d(added_anchors, b_hashes[b_anchor])
            if shared.size < _SOURCE_MIN_SHARED_ANCHORS:
                continue
            qids = _resolve_trace_qids(block, b_hashes, vocabulary)
            if not qids:
                continue
            positions = np.flatnonzero(anchors & _in_sorted(shared, hashes))
            words = list(
                dict.fromkeys(_span_text(buf, starts, ends, int(i), int(i)) for i in positions)
            )[:4]
            return _truncate_reason(
                f"eval_source_identity_leakage: qids={qids} trace=[{label}] "
                f"shared_tokens={words} n={shared.size}"
            )
    return None


# In-suite special-casing (2026-09-15 form): the AP-33 check refuses only suites that
# are ABSENT from the source context; a tactic keyed to a PRESENT suite is refused here.
_SUITE_MENTION_MAX_TF = 300
_SUITE_PHRASE_MAX_TF = 3000
_SUITE_MIN_POOL_TOKENS = 1_000_000
# A suite name the harness itself uses in >= 4 files (a tool, role or routing name such
# as ``web_research``, ``long_context``, ``agentic``) is harness vocabulary: it counts
# only in the KEYED form ("X suite", ``suite == "X"``), never as a bare mention/phrase.
_SUITE_HARNESS_VOCAB_FILES = 4
_SUITE_SHAPED_RE = re.compile(r"\d|_|(?:bench|qa|eval)$")
_SUITE_RULES_CACHE: dict[int, tuple[Any, tuple]] = {}


def _suite_term_pattern(term: str) -> str:
    return r"[\s_-]+".join(re.escape(part) for part in term.split())


def _suite_rules(vocabulary: EvalIdVocabulary, harness: _HarnessContent) -> tuple:
    content = vocabulary.content
    cache_key = (id(content), id(harness))
    cached = _SUITE_RULES_CACHE.get(cache_key)
    if cached is not None and cached[0] is content and cached[1] is harness:
        return cached[2]
    np = _np()
    canonical_of = {term.casefold(): canon for term, canon in _SUITE_TERM_TO_CANONICAL.items()}
    terms = {s for s in vocabulary.suites if len(s) >= 3} | set(canonical_of)
    frequency: dict[str, int] = {}
    harness_files: dict[str, int] = {}
    for term in terms:
        _b, _s, _e, hashes = _content_tokens(term)
        harness_files[term] = min((harness.file_count(h) for h in hashes), default=0)
        if hashes.size == 0 or content is None:
            frequency[term] = 0
            continue
        mask = np.uint64((1 << _CONTENT_SKETCH_BITS) - 1)
        tf = np.minimum(
            content.sketch[0][(hashes & mask).astype(np.intp)],
            content.sketch[1][((hashes >> np.uint64(40)) & mask).astype(np.intp)],
        )
        frequency[term] = int(tf.min() if hashes.size > 1 else tf[0])
    for term, canon in canonical_of.items():  # multi-word aliases inherit their suite
        if " " in term:
            frequency[term] = frequency.get(canon, frequency[term])
            harness_files[term] = harness_files.get(canon, harness_files[term])
    terms_bare = {t for t in terms if harness_files[t] < _SUITE_HARNESS_VOCAB_FILES}
    # Frequencies only mean something over a real pool; a tiny fixture pool sees every
    # word as rare, so there only benchmark-SHAPED names count as bare mentions.
    measured = dict(content.stats).get("tokens", 0) >= _SUITE_MIN_POOL_TOKENS if content else False

    def shaped(term: str) -> bool:
        joined = canonical_of.get(term, term) if " " in term else term
        return bool(_SUITE_SHAPED_RE.search(joined))

    mention = sorted(
        (
            t
            for t in terms_bare
            if frequency[t] <= _SUITE_MENTION_MAX_TF
            and len(t) >= 4
            and (shaped(t) or (measured and t in canonical_of))
        ),
        key=len,
        reverse=True,
    )
    phrase = sorted(
        set(mention) | {t for t in terms_bare if measured and frequency[t] <= _SUITE_PHRASE_MAX_TF},
        key=len,
        reverse=True,
    )
    every = sorted(terms, key=len, reverse=True)

    def alternation(items):
        return "|".join(_suite_term_pattern(t) for t in items) or r"(?!x)x"

    key_ident = r"[\w.\[\]\"'()]*(?:suite|benchmark|dataset)[\w\"'\])]*"
    mention_re = re.compile(rf"(?<!\w)({alternation(mention)})(?!\w)", re.IGNORECASE)
    phrase_re = re.compile(
        rf"(?<!\w)({alternation(phrase)})[\s-]+(?:style\s+|specific\s+)?"
        r"(?:problems?|questions?|tasks?|items?|contests?)\b",
        re.IGNORECASE,
    )
    keyed_re = re.compile(
        rf"(?<!\w)({alternation(every)})[\s-]+(?:suite|benchmark|dataset)s?\b"
        rf"|\b(?:suite|benchmark|dataset)\s*(?:is|==|=|:)\s*[\"'`]?({alternation(every)})(?!\w)"
        rf"|{key_ident}\s*(?:==|!=|\bnot\s+in\b|\bin\b)\s*[(\[{{]?\s*"
        rf"(?:[\"'][\w -]*[\"']\s*,\s*)*[\"']({alternation(every)})[\"']"
        rf"|[\"']({alternation(every)})[\"']\s*(?:==|!=|\bnot\s+in\b|\bin\b)\s*{key_ident}"
        rf"|{key_ident}\s*\.\s*(?:startswith|endswith)\(\s*[(\[]?\s*[\"']({alternation(every)})",
        re.IGNORECASE,
    )
    rules = (mention_re, phrase_re, keyed_re)
    _SUITE_RULES_CACHE.clear()
    _SUITE_RULES_CACHE[cache_key] = (content, harness, rules)
    return rules


def suite_special_casing_reason(
    original: str,
    mutated: str,
    vocabulary: EvalIdVocabulary,
    *,
    harness: _HarnessContent | None = None,
) -> str | None:
    """Refuse a mutation that keys behaviour on an eval suite (None when clean).

    Forms: KEYED ("gpqa suite", ``suite == "math"``), PHRASE ("USACO problems"),
    MENTION (a benchmark-shaped suite name whose count the mutation increases).
    """
    if not vocabulary.available:
        return None  # the fail-closed leakage verdict already refused it
    known = harness if harness is not None else harness_content()
    mention_re, phrase_re, keyed_re = _suite_rules(vocabulary, known)
    added = _added_text(original, mutated)
    for form, pattern in (("keyed", keyed_re), ("phrase", phrase_re)):
        match = pattern.search(added)
        if match:
            term = next(g for g in match.groups() if g)
            return _truncate_reason(
                f"eval_suite_special_casing: suite={term.casefold()} form={form}"
            )

    def counts(text: str) -> dict[str, int]:
        found: dict[str, int] = {}
        for match in mention_re.finditer(text or ""):
            key = " ".join(re.split(r"[\s_-]+", match.group(1).casefold()))
            found[key] = found.get(key, 0) + 1
        return found

    before = counts(original)
    for term, count in counts(mutated).items():
        if count > before.get(term, 0):
            return _truncate_reason(f"eval_suite_special_casing: suite={term} form=mention")
    return None


# --------------------------------------------------------------------------- TD-21.27: mutation extraction
#
# The mutation backend is the Claude CLI (`claude -p`, `_invoke_claude`
# below) -- an external process with no HTTP endpoint and no
# `response_format` support, so the shared `src.structured_output.repair`
# fish-then-repair idiom (TD-21.H) cannot reach it (per the TD-21 dispatch's
# HARD RULES: "apply deterministic fish improvements only and say so"). What
# follows is that deterministic-fish-only improvement, shared by both
# `_extract_mutation` (prompt bodies) and `_extract_code_mutation` (Python
# source) so a fenced-code mention inside the reply's own PROSE can never be
# mis-captured as the payload on either path -- the exact bug class
# `_extract_mutation` already guarded against alone before this change.

#: (site, status) -> count. "parsed" = a fenced/heuristic block was found;
#: "failed" = extraction found nothing and the mutation became a no-op.
#: TD-21.28's counter is `STRUCTURED_OUTPUT_REPAIR_COUNTS` (JSON-shaped);
#: this one is deliberately separate since extraction here is fenced TEXT,
#: not JSON, and there is no repair-turn dimension to key on.
MUTATION_EXTRACTION_COUNTS: dict[tuple[str, str], int] = {}


def _record_mutation_extraction(site: str, status: str) -> None:
    key = (site, status)
    MUTATION_EXTRACTION_COUNTS[key] = MUTATION_EXTRACTION_COUNTS.get(key, 0) + 1


_FENCE_BLOCK_RE = re.compile(
    r"^[ \t]*`{3,}[ \t]*([\w:.\-]*)[ \t]*\r?\n(.*?)(?:\r?\n[ \t]*`{3,}|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Line-anchored fenced blocks as ``(tag.lower(), body)``.

    Only fences whose opening backticks begin a line count as delimiters --
    an inline fenced-code mention inside prose (e.g. a sentence quoting
    ``result.index(...)``) can never be mis-captured as the payload. Shared
    by the prompt- and code-mutation extractors so both apply the identical
    rule instead of two near-copies with different edge-case coverage.
    """
    return [(tag.strip().lower(), body) for tag, body in _FENCE_BLOCK_RE.findall(text)]


class PromptForge:
    """Species 2: LLM-guided prompt mutation and optimization."""

    def __init__(
        self,
        prompts_dir: Path | None = None,
        timeout: int = 300,
        auto_commit: bool = True,
        eval_id_vocabulary: EvalIdVocabulary | None = None,
    ):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.timeout = timeout
        self.auto_commit = auto_commit
        self._session_id: str | None = None
        # RTG-55 MHS-3: injected vocabulary (tests) or the data-sourced default.
        self._injected_eval_id_vocabulary = eval_id_vocabulary

    def _eval_id_vocabulary(self) -> EvalIdVocabulary:
        if self._injected_eval_id_vocabulary is not None:
            return self._injected_eval_id_vocabulary
        return load_eval_id_vocabulary()

    def _leakage_reason(
        self,
        vocabulary: EvalIdVocabulary,
        original: str,
        mutated: str,
        failure_context: str = "",
    ) -> str | None:
        """RTG-55 MHS-3: the one leakage checker for prompt, GEPA and code mutations."""
        return eval_leakage_reason(
            _added_text(original, mutated),
            vocabulary,
            original=original,
            trace_context=failure_context,
            harness=harness_content(self.prompts_dir),
        )

    def list_prompts(self) -> list[str]:
        """List all hot-swappable prompt files (flat + roles/ subdirectory)."""
        if not self.prompts_dir.exists():
            return []
        return sorted(f.name for f in self.prompts_dir.rglob("*.md"))

    def _resolve_prompt_path(self, filename: str) -> Path:
        """Resolve prompt file, searching multiple locations."""
        root = self.prompts_dir.resolve()
        requested = Path(filename)
        path = self.prompts_dir / filename
        if requested.is_absolute() or ".." in requested.parts:
            raise FileNotFoundError(f"Prompt not found: {path}")

        def safe_existing(candidate: Path) -> Path | None:
            if not candidate.exists():
                return None
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                return None
            return resolved

        # Try exact path first (handles roles/worker_explore.md from controller)
        resolved_path = safe_existing(path)
        if resolved_path is not None:
            return resolved_path
        # Try roles/ subdirectory (flat filename like worker_explore.md)
        roles_path = self.prompts_dir / "roles" / filename
        resolved_roles_path = safe_existing(roles_path)
        if resolved_roles_path is not None:
            return resolved_roles_path
        # Try stripping roles/ prefix if controller included it redundantly
        basename = requested.name
        if basename != filename:
            for candidate in [self.prompts_dir / basename, self.prompts_dir / "roles" / basename]:
                resolved_candidate = safe_existing(candidate)
                if resolved_candidate is not None:
                    return resolved_candidate
        raise FileNotFoundError(f"Prompt not found: {path}")

    def read_prompt(self, filename: str) -> str:
        """Read a prompt file."""
        return self._resolve_prompt_path(filename).read_text()

    def write_prompt(self, filename: str, content: str) -> None:
        """Write a prompt file (picked up on next request)."""
        path = self._resolve_prompt_path(filename)
        path.write_text(content)
        log.info("Wrote prompt: %s (%d chars)", filename, len(content))

    def propose_mutation(
        self,
        target_file: str,
        mutation_type: str = "targeted_fix",
        failure_context: str = "",
        per_suite_quality: dict[str, float] | None = None,
        description: str = "",
        eval_tower=None,
        gepa_max_evals: int = 50,
        author_capture_context: dict[str, Any] | None = None,
    ) -> PromptMutation:
        """Propose a prompt mutation via Claude CLI or GEPA.

        When mutation_type="gepa", delegates to GEPA evolutionary optimization
        (AP-19). Requires eval_tower to be passed for orchestrator-based eval.

        Returns PromptMutation with the proposed changes.
        """
        if mutation_type not in MUTATION_TYPES:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        # AP-19: GEPA evolutionary optimization
        if mutation_type == "gepa":
            return self._propose_via_gepa(
                target_file=target_file,
                eval_tower=eval_tower,
                max_evals=gepa_max_evals,
                description=description,
            )

        original = self.read_prompt(target_file)

        prompt = self._build_mutation_prompt(
            target_file=target_file,
            mutation_type=mutation_type,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        # Best-effort metadata is captured at the complete-input boundary, before any author call.
        try:
            from mutation_author_capture import capture_author_context
            capture_author_context(
                prompt, operator=mutation_type, target=target_file,
                inputs={"original_content": original, "failure_context": failure_context,
                        "description": description}, context=author_capture_context,
                parameters={"timeout_s": self.timeout})
        except Exception as exc:
            log.warning("Mutation author metadata unavailable (%s)", type(exc).__name__)
        result = self._invoke_claude(prompt)
        mutated_content, extracted_ok = self._extract_mutation(result, original)

        mutation = PromptMutation(
            file=target_file,
            mutation_type=mutation_type,
            description=description or f"{mutation_type} on {target_file}",
            original_content=original,
            mutated_content=mutated_content,
        )
        if not extracted_ok:
            # TD-21.27: extraction found nothing usable in the Claude CLI
            # reply -- an explicit, counted failure (`safety_valid=False`
            # reuses the EXISTING gate `_action_prompt_mutation` already
            # checks before `apply_mutation`/eval) instead of a mutation
            # that silently costs an eval round and a wasted commit
            # checkpoint for a no-op nobody can see happened.
            mutation.safety_valid = False
            mutation.safety_reason = "mutation_extraction_failed: no usable fenced block in reply"
            return mutation
        integrity_reason = _prompt_integrity_reason(target_file, mutated_content)
        if integrity_reason:
            mutation.safety_valid = False
            mutation.safety_reason = "prompt_integrity:" + integrity_reason
            mutation.mutated_content = original
            log.warning(
                "Prompt mutation rejected by integrity guard (%s): %s",
                target_file,
                integrity_reason,
            )
            return mutation
        self._attach_transfer_safety(
            mutation,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )
        self._attach_prompt_effect(mutation)
        if not mutation.safety_valid:
            log.warning(
                "Prompt mutation rejected by transfer safety (%s): %s",
                target_file,
                mutation.safety_reason,
            )
            mutation.mutated_content = original
        return mutation

    def _attach_prompt_effect(self, mutation: PromptMutation) -> None:
        """RTG-55 MHS-4: classify a prompt mutation and apply the risk gate."""
        mutation.effect = classify_prompt_effect(
            mutation.original_content, mutation.mutated_content
        )
        mutation.effect_risk = mutation_effect_risk(mutation.effect)
        if not mutation.safety_valid:
            return
        gate_reason = mutation_risk_gate_reason(mutation.effect)
        if gate_reason is not None:
            mutation.safety_valid = False
            mutation.safety_reason = gate_reason

    def _propose_via_gepa(
        self,
        target_file: str,
        eval_tower=None,
        max_evals: int = 50,
        description: str = "",
    ) -> PromptMutation:
        """AP-19: Use GEPA evolutionary optimization to propose a mutation.

        Runs GEPA's reflective-mutation + Pareto-selection loop through the
        full orchestrator pipeline (eval_tower), returning the best candidate
        as a PromptMutation.
        """
        from .gepa_optimizer import GEPAPromptOptimizer

        if eval_tower is None:
            raise ValueError("gepa mutation requires eval_tower to be passed")

        optimizer = GEPAPromptOptimizer(
            eval_tower=eval_tower,
            prompt_forge=self,
        )
        result = optimizer.run(
            target_file=target_file,
            max_evals=max_evals,
        )

        if result is None:
            # GEPA failed — return a no-op mutation
            original = self.read_prompt(target_file)
            return PromptMutation(
                file=target_file,
                mutation_type="gepa",
                description="GEPA optimization failed — no mutation proposed",
                original_content=original,
                mutated_content=original,
            )

        mutation = result.to_prompt_mutation()
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            mutation.safety_valid = False
            mutation.safety_reason = "prompt_integrity:" + integrity_reason
            mutation.mutated_content = mutation.original_content
            log.warning(
                "GEPA prompt mutation rejected by integrity guard (%s): %s",
                mutation.file,
                integrity_reason,
            )
            return mutation
        # RTG-55 MHS-3: GEPA candidates can memorise the eval set too.
        vocabulary = self._eval_id_vocabulary()
        leakage = self._leakage_reason(
            vocabulary, mutation.original_content, mutation.mutated_content
        ) or suite_special_casing_reason(
            mutation.original_content,
            mutation.mutated_content,
            vocabulary,
            harness=harness_content(self.prompts_dir),
        )
        if leakage is not None:
            mutation.safety_valid = False
            mutation.safety_reason = leakage
            mutation.mutated_content = mutation.original_content
            log.warning(
                "GEPA prompt mutation rejected by leakage guard (%s): %s",
                mutation.file,
                leakage,
            )
            return mutation
        self._attach_prompt_effect(mutation)
        if not mutation.safety_valid:
            mutation.mutated_content = mutation.original_content
        return mutation

    def apply_mutation(self, mutation: PromptMutation) -> dict[str, Any]:
        """Apply a mutation (write file + optional git commit)."""
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected mutation: {integrity_reason}")

        # Git snapshot before
        git_before = self._capture_git_state()

        # Write the mutated prompt
        self.write_prompt(mutation.file, mutation.mutated_content)

        # Git snapshot after
        git_after = self._capture_git_state()
        mutation.git_diff = self._diff_states(git_before, git_after)
        mutation.accepted = True

        if self.auto_commit and mutation.git_diff:
            self._git_commit(
                f"autopilot: {mutation.mutation_type} on {mutation.file}\n\n{mutation.description}",
                paths=[self._resolve_prompt_path(mutation.file)],
            )

        return {
            "status": "applied",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "diff_lines": len(mutation.git_diff.splitlines()),
        }

    def revert_mutation(self, mutation: PromptMutation) -> None:
        """Revert a mutation to original content and commit the revert."""
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.original_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected revert: {integrity_reason}")

        self.write_prompt(mutation.file, mutation.original_content)
        mutation.accepted = False
        # Commit the revert so corrupted state is never the HEAD
        if self.auto_commit:
            self._git_commit(
                f"autopilot: revert prompt mutation on {mutation.file}\n\n"
                f"Reverted: {mutation.description}",
                paths=[self._resolve_prompt_path(mutation.file)],
            )
        log.info("Reverted prompt mutation on %s (committed)", mutation.file)

    # ── Claude CLI invocation ────────────────────────────────────

    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude CLI following the claude_debugger pattern."""
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--allowedTools",
            "Read,Grep,Glob",
        ]
        if self._session_id:
            cmd.extend(["--resume", self._session_id])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)

            if proc.returncode != 0:
                log.error("Claude CLI failed (rc=%d): %s", proc.returncode, stderr[:500])
                return ""

            # Parse JSON response
            try:
                response = json.loads(stdout)
                # Capture session ID for resume
                if "session_id" in response:
                    self._session_id = response["session_id"]
                return response.get("result", stdout)
            except json.JSONDecodeError:
                return stdout

        except subprocess.TimeoutExpired:
            proc.kill()
            log.error("Claude CLI timed out after %ds", self.timeout)
            return ""
        except FileNotFoundError:
            log.error("Claude CLI not found. Ensure 'claude' is on PATH.")
            return ""

    # ── prompt building ──────────────────────────────────────────

    def _build_mutation_prompt(
        self,
        target_file: str,
        mutation_type: str,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> str:
        """Build the prompt for Claude CLI to propose a mutation."""
        lines = [
            "You are an expert prompt engineer optimizing an LLM orchestration system.",
            "",
            f"## Task: {mutation_type} mutation on `{target_file}`",
            "",
        ]

        if description:
            lines.append(f"Goal: {description}\n")

        # Mutation type instructions
        type_instructions = {
            "targeted_fix": (
                "Analyze the failure cases below and make targeted edits to fix "
                "the specific failure patterns. Keep changes minimal and focused."
            ),
            "compress": (
                "Reduce the token count of this prompt while preserving its behavior. "
                "Remove redundant instructions, merge similar sections, use concise language."
            ),
            "few_shot_evolution": (
                "Improve the examples/few-shot demonstrations in this prompt. "
                "Add examples for underperforming suites, remove unhelpful ones."
            ),
            "crossover": (
                "Identify the strongest sections of this prompt and strengthen "
                "weaker sections by applying similar patterns."
            ),
            "style_transfer": (
                "Apply successful structural patterns (section organization, "
                "instruction phrasing, constraint framing) from high-performing "
                "prompts to this one."
            ),
        }
        lines.append(type_instructions.get(mutation_type, "Improve this prompt."))
        lines.append("")

        # Current prompt
        lines.append(f"## Current prompt ({target_file}):\n```markdown")
        lines.append(original_content)
        lines.append("```\n")

        # Failure context
        if failure_context:
            lines.append(f"## Recent failure cases:\n{failure_context}\n")

        # Per-suite quality
        if per_suite_quality:
            lines.append("## Per-suite quality (0-3 scale):")
            for suite, quality in sorted(per_suite_quality.items()):
                bar = "█" * int(quality) + "░" * (3 - int(quality))
                lines.append(f"  {suite}: {quality:.2f} {bar}")
            lines.append("")

        lines.append(self._negative_transfer_safety_block())
        lines.append("")

        # Output format
        lines.append(
            "## Output format:\n"
            "Return the complete mutated prompt inside a ```markdown fenced block. "
            "Also include a brief explanation of your changes in a "
            "```json:autopilot_actions block:\n"
            "```json:autopilot_actions\n"
            '{"changes": ["change1", "change2"], "rationale": "..."}\n'
            "```"
        )

        return "\n".join(lines)

    def _extract_mutation(self, result: str, original: str) -> tuple[str, bool]:
        """Extract mutated prompt from Claude's response.

        Only fences whose opening backticks begin a line are treated as block
        delimiters (`_fenced_blocks`). This prevents an inline fenced-code
        mention inside the reply's *prose* (e.g. a sentence quoting
        ``result.index(...)``) from being mis-captured as the payload — a bug
        that overwrote a prompt file with the model's prose and committed it.

        TD-21.27: returns ``(content, ok)``. ``ok=False`` means extraction
        found nothing usable — ``content`` is still ``original`` (the Claude
        CLI backend is unreachable for a repair turn, so a safe no-op is the
        only sound fallback), but the caller now gets an explicit signal
        instead of an indistinguishable "successful no-op mutation", so it
        can skip the mutation entirely (see ``propose_mutation``) rather than
        spend an eval cycle and a git commit around it.
        """
        blocks = _fenced_blocks(result)

        # Prefer an explicitly prose/markdown-tagged block.
        for tag, body in blocks:
            if tag in ("markdown", "md", "text"):
                _record_mutation_extraction("prompt_forge.mutation_extraction", "parsed")
                return body.strip(), True

        # Fallback: the largest block that is not a json/actions or object block.
        candidates = [
            body
            for tag, body in blocks
            if not tag.startswith("json")
            and not body.lstrip().startswith("{")
            and len(body.strip()) > 100
        ]
        if candidates:
            _record_mutation_extraction("prompt_forge.mutation_extraction", "parsed")
            return max(candidates, key=lambda b: len(b.strip())).strip(), True

        log.warning("Could not extract mutation from response, returning original")
        _record_mutation_extraction("prompt_forge.mutation_extraction", "failed")
        return original, False

    # ── git operations ───────────────────────────────────────────

    def _capture_git_state(self) -> dict[str, str]:
        """Capture git diff state of prompts directory."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", str(self.prompts_dir)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            return {"diff_stat": result.stdout}
        except Exception:
            return {}

    def _diff_states(self, before: dict[str, str], after: dict[str, str]) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", str(self.prompts_dir)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            return result.stdout
        except Exception:
            return ""

    def _git_commit(self, message: str, *, paths: list[Path] | None = None) -> None:
        """Commit ONLY the prompt file(s) this mutation touched (never a directory sweep)."""
        if not paths:
            log.warning("Git commit skipped: no prompt path given (refusing a directory sweep)")
            return
        self._git_commit_paths(paths, message)

    @staticmethod
    def _git_commit_paths(paths: list[Path], message: str) -> bool:
        """Stage and commit exactly ``paths``; unrelated staged or dirty files stay out.

        ``git commit --only -- <paths>`` records just those paths and leaves every other
        index entry as it was, so an operator's staged work never rides along with an
        autopilot commit (gate-frontier review, 2026-09-16).
        """
        specs = [str(Path(p)) for p in paths]
        try:
            subprocess.run(
                ["git", "add", "-A", "--", *specs],
                timeout=10,
                check=True,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
            )
            proc = subprocess.run(
                ["git", "commit", "--only", "-m", message, "--", *specs],
                timeout=10,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                log.warning(
                    "Git commit of %s failed: %s", specs, (proc.stderr or proc.stdout).strip()[:300]
                )
                return False
            log.info("Committed %s", ", ".join(Path(p).name for p in specs))
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("Git commit failed: %s", e)
            return False

    # ── Worktree-isolated mutations (AP-11) ────────────────────────

    def apply_mutation_isolated(
        self,
        mutation: PromptMutation,
        trial_name: str,
    ) -> "ExperimentContext":
        """Apply a prompt mutation in an isolated worktree.

        Returns an ExperimentContext. The caller must call ctx.accept() or
        ctx.reject() after evaluation. If neither is called, the context
        manager auto-rejects on cleanup.

        Usage:
            from scripts.autopilot.worktree_manager import WorktreeManager
            wt = WorktreeManager()
            with wt.experiment(trial_name) as ctx:
                forge.apply_mutation_in_context(ctx, mutation)
                result = tower.hybrid_eval()
                if result.quality > baseline:
                    ctx.accept(f"autopilot: {mutation.mutation_type} on {mutation.file}")
                else:
                    ctx.reject()
        """
        from scripts.autopilot.worktree_manager import WorktreeManager

        wt = WorktreeManager(PROJECT_ROOT)
        return wt.experiment(trial_name)

    def apply_mutation_in_context(
        self,
        ctx: Any,
        mutation: "PromptMutation",
    ) -> dict[str, Any]:
        """Apply a prompt mutation within an experiment context.

        The context handles file backup, worktree versioning, and
        copying the mutated file to the main repo for live eval.
        """
        integrity_reason = _prompt_integrity_reason(mutation.file, mutation.mutated_content)
        if integrity_reason:
            raise ValueError(f"prompt integrity rejected isolated mutation: {integrity_reason}")

        rel_path = f"orchestration/prompts/{mutation.file}"
        ctx.apply_file(rel_path, mutation.mutated_content)
        mutation.accepted = True
        return {
            "status": "applied_isolated",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "worktree": str(ctx.worktree_path),
        }

    def apply_code_mutation_in_context(
        self,
        ctx: Any,
        mutation: "CodeMutation",
    ) -> dict[str, Any]:
        """Apply a code mutation within an experiment context."""
        if not mutation.syntax_valid:
            return {"status": "rejected", "reason": "syntax_invalid"}
        if MutationEffect.normalize(mutation.effect) is MutationEffect.UNSAFE:
            return {"status": "rejected", "reason": "effect_unsafe"}
        gate_reason = mutation_risk_gate_reason(mutation.effect)
        if gate_reason is not None:
            return {"status": "rejected", "reason": gate_reason}
        ctx.apply_file(mutation.file, mutation.mutated_content)
        mutation.accepted = True
        return {
            "status": "applied_isolated",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "effect": MutationEffect.normalize(mutation.effect).value,
            "effect_risk": mutation_effect_risk(mutation.effect),
            "worktree": str(ctx.worktree_path),
        }

    def summary(self) -> dict[str, Any]:
        """Summary for controller."""
        prompts = self.list_prompts()
        return {
            "available_prompts": prompts,
            "n_prompts": len(prompts),
            "session_active": self._session_id is not None,
            "mutation_types": MUTATION_TYPES,
            "code_mutation_targets": CODE_MUTATION_ALLOWLIST,
        }

    # ── Meta-Harness Tier 2: Code mutations ──────────────────────

    def propose_code_mutation(
        self,
        target_file: str,
        mutation_type: str = "targeted_fix",
        failure_context: str = "",
        per_suite_quality: dict[str, float] | None = None,
        description: str = "",
        author_capture_context: dict[str, Any] | None = None,
    ) -> CodeMutation:
        """Propose a mutation to a Python code file (Tier 2 search space).

        Only files in CODE_MUTATION_ALLOWLIST may be mutated.
        """
        if mutation_type not in {"targeted_fix", "compress", "new_file"}:
            raise ValueError(f"Unknown code mutation type: {mutation_type}")

        abs_path = _resolve_code_mutation_target(target_file)
        if mutation_type == "new_file":
            roots = new_file_mutation_roots()
            if not _is_under_any(abs_path.parent, roots):
                raise ValueError(
                    f"New-file mutation blocked: {target_file} must stay under "
                    f"one of {', '.join(new_file_mutation_root_labels())}"
                )
            if not abs_path.parent.exists():
                raise FileNotFoundError(f"New-file parent directory not found: {abs_path.parent}")
            if abs_path.exists():
                raise FileExistsError(f"New-file mutation blocked: {abs_path} already exists")
            original = ""
        else:
            if target_file not in CODE_MUTATION_ALLOWLIST:
                raise ValueError(
                    f"Code mutation blocked: {target_file} not in allowlist. "
                    f"Allowed: {CODE_MUTATION_ALLOWLIST}"
                )
            if not abs_path.exists():
                raise FileNotFoundError(f"Target file not found: {abs_path}")
            original = abs_path.read_text()

        prompt = self._build_code_mutation_prompt(
            target_file=target_file,
            mutation_type=mutation_type,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        # Best-effort metadata is captured at the complete-input boundary, before any author call.
        try:
            from mutation_author_capture import capture_author_context
            capture_author_context(
                prompt, operator=mutation_type, target=target_file,
                inputs={"original_content": original, "failure_context": failure_context,
                        "description": description}, context=author_capture_context,
                parameters={"timeout_s": self.timeout})
        except Exception as exc:
            log.warning("Mutation author metadata unavailable (%s)", type(exc).__name__)
        result = self._invoke_claude(prompt)
        mutated_content, extracted_ok = self._extract_code_mutation(result, original)

        mutation = CodeMutation(
            file=target_file,
            mutation_type=mutation_type,
            description=description or f"{mutation_type} on {target_file}",
            original_content=original,
            mutated_content=mutated_content,
        )
        if not extracted_ok:
            # TD-21.27: same explicit, counted failure as the prompt-mutation
            # path. `_action_code_mutation` checks `syntax_valid` BEFORE
            # `safety_valid` -- syntax_valid=True here is not a claim that
            # anything was screened, it is what routes this rejection
            # through the transfer_safety gate (which DOES surface
            # `safety_reason` as `gate_detail`) instead of the generic,
            # reason-less syntax_validation gate, since mutated_content is
            # still the untouched, already-valid original.
            mutation.syntax_valid = True
            mutation.safety_valid = False
            mutation.safety_reason = "mutation_extraction_failed: no usable fenced block in reply"
            return mutation
        self._attach_transfer_safety(
            mutation,
            original_content=original,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description,
        )

        # Deep validation: syntax + shrinkage + public names + static safety screen.
        # STATIC ONLY — nothing is written to the repo and nothing is imported.
        report = self._screen_code_mutation(
            original,
            mutated_content,
            target_file,
            is_new_file=(mutation_type == "new_file"),
        )
        mutation.syntax_valid = report.safe
        mutation.effect = report.effect
        mutation.effect_reason = report.reason
        if not report.safe:
            log.warning("Code mutation rejected (%s): %s", target_file, report.reason)
            mutation.mutated_content = original
        if not mutation.safety_valid:
            log.warning(
                "Code mutation rejected by transfer safety (%s): %s",
                target_file,
                mutation.safety_reason,
            )
            mutation.mutated_content = original
            mutation.effect = MutationEffect.UNSAFE
            mutation.effect_reason = _truncate_reason(mutation.safety_reason)

        # RTG-55 MHS-4: ANTI-OVERRIDE risk prior + gate.
        mutation.effect_risk = mutation_effect_risk(mutation.effect)
        if report.safe and mutation.safety_valid:
            gate_reason = mutation_risk_gate_reason(mutation.effect)
            if gate_reason is not None:
                log.warning("Code mutation rejected by risk gate (%s): %s", target_file, gate_reason)
                mutation.safety_valid = False
                mutation.safety_reason = gate_reason
                mutation.mutated_content = original

        return mutation

    def apply_code_mutation(self, mutation: CodeMutation) -> dict[str, Any]:
        """Apply a code mutation with syntax validation + git safety."""
        if not mutation.syntax_valid:
            return {"status": "rejected", "reason": "syntax_invalid"}
        if MutationEffect.normalize(mutation.effect) is MutationEffect.UNSAFE:
            return {"status": "rejected", "reason": "effect_unsafe"}
        gate_reason = mutation_risk_gate_reason(mutation.effect)
        if gate_reason is not None:
            return {"status": "rejected", "reason": gate_reason}

        abs_path = PROJECT_ROOT / mutation.file

        # Git commit current state before mutation (safety net). Only this file: a
        # checkpoint must never sweep unrelated staged work into an autopilot commit.
        if abs_path.exists():
            self._git_commit_paths(
                [abs_path], f"autopilot: pre-code-mutation checkpoint ({mutation.file})"
            )  # may fail when there is nothing to commit — that's OK

        # Write the mutated code
        abs_path.write_text(mutation.mutated_content)
        mutation.accepted = True

        # Capture diff
        try:
            if mutation.mutation_type == "new_file" and not mutation.original_content:
                result = subprocess.run(
                    ["git", "diff", "--no-index", "--", "/dev/null", str(abs_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(PROJECT_ROOT),
                )
            else:
                result = subprocess.run(
                    ["git", "diff", str(abs_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(PROJECT_ROOT),
                )
            mutation.git_diff = result.stdout
        except Exception:
            mutation.git_diff = ""

        if self.auto_commit and mutation.git_diff:
            self._git_commit_file(
                abs_path,
                f"autopilot: code {mutation.mutation_type} on {mutation.file}\n\n"
                f"{mutation.description}",
            )

        return {
            "status": "applied",
            "file": mutation.file,
            "mutation_type": mutation.mutation_type,
            "effect": MutationEffect.normalize(mutation.effect).value,
            "effect_risk": mutation_effect_risk(mutation.effect),
            "diff_lines": len(mutation.git_diff.splitlines()),
        }

    def revert_code_mutation(self, mutation: CodeMutation) -> None:
        """Revert a code mutation to original content and commit the revert."""
        abs_path = PROJECT_ROOT / mutation.file
        if mutation.mutation_type == "new_file" and not mutation.original_content:
            abs_path.unlink(missing_ok=True)
        else:
            abs_path.write_text(mutation.original_content)
        mutation.accepted = False
        # Commit the revert so corrupted state is never the HEAD
        if self.auto_commit:
            if mutation.mutation_type == "new_file" and not mutation.original_content:
                self._git_commit_paths(
                    [abs_path],
                    f"autopilot: revert code mutation on {mutation.file}\n\n"
                    f"Reverted: {mutation.description}",
                )
            else:
                self._git_commit_file(
                    abs_path,
                    f"autopilot: revert code mutation on {mutation.file}\n\n"
                    f"Reverted: {mutation.description}",
                )
        log.info("Reverted code mutation on %s (committed)", mutation.file)

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax via ast.parse."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            log.warning("Syntax error in mutated code: %s", e)
            return False

    def _validate_code_mutation(
        self,
        original: str,
        mutated: str,
        target_file: str,
        *,
        is_new_file: bool = False,
    ) -> tuple[bool, str]:
        """Backwards-compatible ``(valid, reason)`` wrapper over the static screen."""
        report = self._screen_code_mutation(
            original, mutated, target_file, is_new_file=is_new_file
        )
        return report.safe, report.reason

    def _screen_code_mutation(
        self,
        original: str,
        mutated: str,
        target_file: str,
        *,
        is_new_file: bool = False,
    ) -> StaticSafetyReport:
        """Deep validation of a code mutation beyond syntax. STATIC ONLY.

        Returns (valid, reason). Checks:
        1. Syntax (ast.parse)
        2. No catastrophic size reduction (>60% shrinkage)
        3. Public names preserved (classes, functions defined at module level)
        4. RTG-55 MHS-2 static safety screen (AST denylist + effect classification)

        This function NEVER writes into the source tree and NEVER imports or
        execs the candidate in this process. The pre-RTG-55 step 4 wrote the
        model's code over the live repo file and then ``importlib`` -imported
        it, so the mutation's module top level ran unsandboxed and any
        concurrent reader saw the candidate on disk.
        """
        # 1. Syntax
        try:
            mutated_tree = ast.parse(mutated)
        except SyntaxError as e:
            return StaticSafetyReport(
                False, MutationEffect.UNSAFE, (_truncate_reason(f"syntax error: {e}"),)
            )

        # 2. Catastrophic shrinkage — reject if >60% of lines removed
        orig_lines = len(original.splitlines())
        new_lines = len(mutated.splitlines())
        if not is_new_file and orig_lines > 10 and new_lines < orig_lines * 0.4:
            return StaticSafetyReport(
                False,
                MutationEffect.REPLACE,
                (
                    f"catastrophic shrinkage: {orig_lines}→{new_lines} lines "
                    f"({100 * (1 - new_lines / orig_lines):.0f}% removed)",
                ),
            )

        # 3. Public names preserved — every class/function at module level
        #    in the original must still exist in the mutated version
        if not is_new_file:
            orig_names = _top_level_definition_names(ast.parse(original))
            missing = orig_names - _top_level_definition_names(mutated_tree)
            if missing:
                return StaticSafetyReport(
                    False,
                    MutationEffect.REPLACE,
                    (f"missing public names: {sorted(missing)}",),
                )

        # 4. Static safety screen (RTG-55 MHS-2). Replaces the pre-RTG-55
        #    "write the candidate over the live file and importlib-import it"
        #    step: no repo write, no in-process execution, no sys.path games.
        #    `new_file` proposals get the strict inertness denylist.
        report = screen_static_safety(mutated, original=original, strict=is_new_file)
        if not report.safe:
            log.warning(
                "Static safety screen rejected mutation on %s: %s", target_file, report.reason
            )
        return report

    def _build_code_mutation_prompt(
        self,
        target_file: str,
        mutation_type: str,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> str:
        """Build prompt for code mutation."""
        lines = [
            "You are an expert Python engineer optimizing an LLM orchestration system.",
            "",
            f"## Task: {mutation_type} mutation on `{target_file}`",
            "",
        ]

        if description:
            lines.append(f"Goal: {description}\n")

        type_instructions = {
            "targeted_fix": (
                "Analyze the failure cases below and make targeted edits to fix "
                "specific failure patterns. Keep changes minimal and focused. "
                "Do NOT refactor or add features beyond what's needed to fix the issue."
            ),
            "compress": (
                "Reduce complexity while preserving behavior. Remove dead code, "
                "simplify conditionals, merge redundant branches."
            ),
            "new_file": (
                "Create a new Python module at the requested path. Keep it "
                "directory-scoped, minimal, and self-contained. Do not alter "
                "existing files. Allowed roots: "
                f"{', '.join(new_file_mutation_root_labels())}."
            ),
        }
        lines.append(
            type_instructions.get(mutation_type, "Improve this code with minimal changes.")
        )
        lines.append("")

        if mutation_type == "new_file" and not original_content.strip():
            lines.append(
                f"## Current code (`{target_file}`):\n"
                "(This file does not exist yet. Create it from scratch.)\n```python"
            )
        else:
            lines.append(f"## Current code (`{target_file}`):\n```python")
        lines.append(original_content)
        lines.append("```\n")

        if failure_context:
            lines.append(f"## Context (failures, traces, insights):\n{failure_context}\n")

        if per_suite_quality:
            lines.append("## Per-suite quality (0-3 scale):")
            for suite, quality in sorted(per_suite_quality.items()):
                bar = "█" * int(quality) + "░" * (3 - int(quality))
                lines.append(f"  {suite}: {quality:.2f} {bar}")
            lines.append("")

        lines.append(
            "## Proposer-prior contract (MH-6):\n"
            "Read inputs in this order and do not skip ahead:\n"
            "1. Failed traces and recent regressions in the context above.\n"
            "2. Current frontier or accepted behavior implied by the existing code.\n"
            "3. Strategy-store or prior-mutation notes present in the context.\n"
            "4. The operator request / mutation goal.\n"
            "\n"
            "Before proposing code, estimate:\n"
            "- expected_quality_delta: signed expected quality change on the cited "
            "failure surface; use a small numeric value and say when evidence is weak.\n"
            "- expected_cost_delta: signed expected runtime/token/complexity change; "
            "use 0.0 when the change should be behavior-only.\n"
            "\n"
            "no-task-specific-hints: do not hard-code benchmark IDs, exact prompts, "
            "known answers, or dataset-specific shortcuts. Generalize only from "
            "observable failure mechanisms."
        )
        lines.append("")

        try:
            target_abs = _resolve_code_mutation_target(target_file)
        except FileNotFoundError:
            target_abs = PROJECT_ROOT / target_file
        if mutation_type == "new_file" and _is_memory_schema_evolution_target(target_abs):
            lines.append(
                "## AutoMem memory schema-evolution contract (MH-9/P2):\n"
                "- Create a default-inert schema/scaffold module describing a "
                "`MemoryAction` / `MemoryActionStore` schema proposal; importing "
                "it must not write files, start subprocesses, call inference, or "
                "touch the trace store. Importing it must DO NOTHING at all.\n"
                "- Express schema-evolution moves as prompt-free helpers, "
                "contracts, constants, or pure validators over "
                "APPEND/CREATE/UPSERT and the status/inventory/strategy/plan/log "
                "channels.\n"
                "- Do not change SafetyGate, Pareto admission, eval scoring, "
                "blacklists, thresholds, planner spend-breaker flags, or live "
                "runtime behavior.\n"
                "- Keep exports narrow and include explicit blockers when "
                "calibration, process, or validation evidence is missing.\n"
                "\n"
                "### Required SHAPE — a static validator rejects anything else\n"
                "Inertness is enforced by an AST denylist, not by trust. A "
                "proposal that breaks any rule below is DISCARDED WITHOUT "
                "REVIEW, so write to this shape exactly:\n"
                "- NO import statements of any kind — not even `dataclasses`, "
                "`enum`, `typing`, or `__future__`. Use only builtins.\n"
                "- NO `class` statements. Express the schema as plain module "
                "data: a `SCHEMA` dict, or a tuple of field-spec tuples/dicts, "
                "plus constants such as `SCHEMA_VERSION`, `ACTIONS`, "
                "`CHANNELS`.\n"
                "- NO underscore-prefixed names anywhere (no `_helper`, no "
                "`_CACHE`, no `obj._attr`). Every name must be public.\n"
                "- NO `raise`, `try`, `with`, `while`, `lambda`, `global`, "
                "`del`, `yield`, or `await`. A validator returns a "
                "`(ok, reason)` tuple instead of raising: "
                "`return (False, 'channel must be one of CHANNELS')`.\n"
                "- Module level may contain ONLY the docstring, constant "
                "assignments, and `def` statements. No calls, prints, or "
                "registration at module level.\n"
                "- Inside functions: `if`/`for`/`return`, comparisons, f-strings, "
                "and builtin calls (`len`, `isinstance`, `sorted`, `tuple`, "
                "`dict`, `str`) are all fine.\n"
                "\n"
                "Shape example (structure to copy, not content to reuse):\n"
                "```python\n"
                f"{MEMORY_SCHEMA_SHAPE_EXAMPLE}"
                "```"
            )
            lines.append("")

        lines.append(self._negative_transfer_safety_block())
        lines.append("")

        lines.append(
            "## IMPORTANT CONSTRAINTS:\n"
            "1. Return the COMPLETE modified file in a ```python fenced block\n"
            "2. Do NOT change function signatures or class names\n"
            "3. Do NOT add new dependencies\n"
            "4. Keep changes minimal — one logical change only\n"
            "5. The code must pass ast.parse() (valid Python syntax)\n"
        )

        lines.append(
            "## Output format:\n"
            "Return the complete modified file inside a ```python fenced block first. "
            "Then include a ```json:autopilot_actions block with keys "
            "`expected_quality_delta`, `expected_cost_delta`, `read_order_used`, "
            "`no_task_specific_hints`, and `rationale`."
        )

        return "\n".join(lines)

    def _negative_transfer_safety_block(self) -> str:
        return (
            "## Negative-transfer safety (AP-33):\n"
            "- Do not import tactics anchored to a benchmark suite or domain that is "
            "not present in the failure context or per-suite quality list.\n"
            f"- If fewer than {_MIN_VALIDATION_TRIALS} trial IDs are cited, phrase "
            "changes as exploratory and do not claim validation.\n"
            "- Do not turn suite-specific fixes into universal always/never/all-tasks "
            "best practices.\n"
            "- Never name a specific eval question, sample, item or task id (e.g. "
            "`<suite>_00042`, `question #12`); a change keyed to one instance memorises "
            "the eval set and is rejected automatically (RTG-55 MHS-3).\n"
            "- Prefer CONSTRAIN edits (add a check, block a bad path, re-prompt) over "
            "REPLACE edits (rewrite or force an action, hard-code an answer). Override "
            "edits carry the highest regression risk and are ranked last (RTG-55 MHS-4)."
        )

    def _attach_transfer_safety(
        self,
        mutation: PromptMutation | CodeMutation,
        *,
        original_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> TransferSafetyVerdict:
        verdict = self._transfer_safety_verdict(
            original_content=original_content,
            mutated_content=mutation.mutated_content,
            failure_context=failure_context,
            per_suite_quality=per_suite_quality,
            description=description or mutation.description,
        )
        mutation.safety_valid = verdict.valid
        mutation.safety_reason = verdict.reason
        mutation.safety_warnings = list(verdict.warnings)
        return verdict

    def _transfer_safety_verdict(
        self,
        *,
        original_content: str,
        mutated_content: str,
        failure_context: str,
        per_suite_quality: dict[str, float] | None,
        description: str,
    ) -> TransferSafetyVerdict:
        source_text = " ".join(str(suite) for suite in (per_suite_quality or {}))
        source_suites = _suite_mentions(source_text)
        if not source_suites:
            source_suites = _suite_mentions(failure_context)

        introduced_text = f"{description}\n{_added_text(original_content, mutated_content)}"
        introduced_suites = _suite_mentions(introduced_text)
        evidence_count = _trial_reference_count(failure_context)

        warnings: list[str] = []
        if failure_context.strip() and evidence_count < _MIN_VALIDATION_TRIALS:
            warnings.append(f"low_evidence_trial_count:{evidence_count}")

        # RTG-55 MHS-3: under-generalization (eval-instance leakage), fail-closed.
        # Scoped to the text the mutation ADDS; the description is controller metadata.
        vocabulary = self._eval_id_vocabulary()
        leakage = self._leakage_reason(
            vocabulary, original_content, mutated_content, failure_context
        )
        if leakage is not None:
            return TransferSafetyVerdict(
                valid=False,
                reason=leakage,
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        mismatched = introduced_suites - source_suites
        if source_suites and mismatched:
            return TransferSafetyVerdict(
                valid=False,
                reason=(
                    "domain_mismatched_anchoring:"
                    f" introduced_suites={sorted(mismatched)}"
                    f" source_suites={sorted(source_suites)}"
                ),
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        if introduced_suites and _UNIVERSAL_TRANSFER_RE.search(introduced_text):
            return TransferSafetyVerdict(
                valid=False,
                reason=(f"misapplied_best_practice: introduced_suites={sorted(introduced_suites)}"),
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        # RTG-55 MHS-3 (2026-09-15): in-suite special-casing, which the check above allows.
        special = suite_special_casing_reason(
            original_content,
            mutated_content,
            vocabulary,
            harness=harness_content(self.prompts_dir),
        )
        if special is not None:
            return TransferSafetyVerdict(
                valid=False,
                reason=special,
                warnings=tuple(warnings),
                source_suites=tuple(sorted(source_suites)),
                introduced_suites=tuple(sorted(introduced_suites)),
                evidence_trial_count=evidence_count,
            )

        return TransferSafetyVerdict(
            valid=True,
            warnings=tuple(warnings),
            source_suites=tuple(sorted(source_suites)),
            introduced_suites=tuple(sorted(introduced_suites)),
            evidence_trial_count=evidence_count,
        )

    def _extract_code_mutation(self, result: str, original: str) -> tuple[str, bool]:
        """Extract mutated Python code from Claude's response.

        TD-21.27: rebuilt on the SAME line-anchored ``_fenced_blocks`` helper
        `_extract_mutation` already uses, closing the same bug class here —
        the old ``result.split("```")`` treated ANY triple-backtick anywhere
        in the reply's prose as a delimiter, with no line-anchoring at all.
        Returns ``(content, ok)``; see `_extract_mutation` for the ``ok``
        contract (extraction failure never repairs via the Claude CLI
        backend, only fishes better and reports the miss explicitly).
        """
        blocks = _fenced_blocks(result)

        for tag, body in blocks:
            if tag in ("python", "py"):
                _record_mutation_extraction("prompt_forge.code_mutation_extraction", "parsed")
                return body.strip(), True

        candidates = [
            body
            for tag, body in blocks
            if not tag.startswith("json")
            and not body.lstrip().startswith(("json", "{"))
            and len(body.strip()) > 100
        ]
        if candidates:
            chosen = max(candidates, key=lambda b: len(b.strip())).strip()
            # Byte-identical to the pre-TD-21.27 behaviour for a bare fence
            # whose language tag landed as the body's first LINE rather than
            # on the opening delimiter (some replies fence that way).
            lines = chosen.split("\n")
            if lines and lines[0].strip() in ("python", "py"):
                chosen = "\n".join(lines[1:]).strip()
            _record_mutation_extraction("prompt_forge.code_mutation_extraction", "parsed")
            return chosen, True

        log.warning("Could not extract code mutation from response, returning original")
        _record_mutation_extraction("prompt_forge.code_mutation_extraction", "failed")
        return original, False

    def _git_commit_file(self, path: Path, message: str) -> None:
        """Git add + commit exactly one file (no sweep of other staged changes)."""
        self._git_commit_paths([path], message)
