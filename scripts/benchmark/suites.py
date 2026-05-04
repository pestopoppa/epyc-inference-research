#!/usr/bin/env python3
from __future__ import annotations

"""
Benchmark Suite Loader

Loads benchmark prompt definitions from YAML files in:
  /mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/v1/

Provides functions to:
- Load all suites
- Get questions from a suite
- Get inference parameters
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

try:
    from context_generator import build_full_prompt
except ImportError:
    from .context_generator import build_full_prompt


# Default paths
PROMPTS_DIR = "/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/v1"


@dataclass
class Question:
    """A benchmark question."""

    id: str
    tier: int
    name: str
    prompt: str
    expected: str
    scoring: list[dict[str, Any]]
    # Long context fields (optional)
    context_tokens: Optional[int] = None
    context_type: Optional[str] = None
    needle: Optional[str] = None
    needle_position: Optional[str] = None
    # Vision fields (optional)
    image_path: Optional[str] = None


@dataclass
class Suite:
    """A benchmark suite."""

    name: str
    version: int
    domain: str
    description: str
    questions: list[Question]
    inference_params: dict[str, Any]


def load_suite(name: str, prompts_dir: str = PROMPTS_DIR) -> Optional[Suite]:
    """Load a benchmark suite from YAML.

    Args:
        name: Suite name (e.g., 'thinking', 'coder').
        prompts_dir: Directory containing prompt YAML files.

    Returns:
        Suite object or None if not found.
    """
    yaml_path = Path(prompts_dir) / f"{name}.yaml"
    if not yaml_path.exists():
        return None

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Parse questions
    questions = []
    prompts = data.get("prompts", {})
    is_long_context = name == "long_context"

    for qid, qdata in prompts.items():
        raw_prompt = qdata.get("prompt", "").strip()

        # For long_context suite, generate context and build full prompt
        if is_long_context:
            context_tokens = qdata.get("context_tokens", 4000)
            context_type = qdata.get("context_type", "technical_docs")
            needle = qdata.get("needle")
            needle_position = qdata.get("needle_position", "middle")

            # Build full prompt with generated context
            full_prompt = build_full_prompt(
                context_type=context_type,
                target_tokens=context_tokens,
                question_prompt=raw_prompt,
                needle=needle,
                needle_position=needle_position,
            )
        else:
            full_prompt = raw_prompt
            context_tokens = None
            context_type = None
            needle = None
            needle_position = None

        questions.append(
            Question(
                id=qid,
                tier=qdata.get("tier", 1),
                name=qdata.get("name", qid),
                prompt=full_prompt,
                expected=qdata.get("expected", ""),
                scoring=qdata.get("scoring", []),
                context_tokens=context_tokens,
                context_type=context_type,
                needle=needle,
                needle_position=needle_position,
                image_path=qdata.get("image_path"),
            )
        )

    # Sort by tier (hardest first: T3→T2→T1) then by ID
    questions.sort(key=lambda q: (-q.tier, q.id))

    return Suite(
        name=name,
        version=data.get("version", 1),
        domain=data.get("domain", name),
        description=data.get("description", ""),
        questions=questions,
        inference_params=data.get("inference_params", {}),
    )


def get_all_suite_names(prompts_dir: str = PROMPTS_DIR) -> list[str]:
    """Get names of all available suites.

    Args:
        prompts_dir: Directory containing prompt YAML files.

    Returns:
        List of suite names (without .yaml extension).
    """
    path = Path(prompts_dir)
    if not path.exists():
        return []

    return sorted([f.stem for f in path.glob("*.yaml")])


def load_all_suites(prompts_dir: str = PROMPTS_DIR) -> dict[str, Suite]:
    """Load all available benchmark suites.

    Args:
        prompts_dir: Directory containing prompt YAML files.

    Returns:
        Dict mapping suite name to Suite object.
    """
    suites = {}
    for name in get_all_suite_names(prompts_dir):
        suite = load_suite(name, prompts_dir)
        if suite:
            suites[name] = suite
    return suites


def get_questions(suite: Suite) -> Iterator[Question]:
    """Iterate over questions in a suite."""
    yield from suite.questions


def get_questions_by_tier(suite: Suite, tier: int) -> list[Question]:
    """Get questions for a specific tier."""
    return [q for q in suite.questions if q.tier == tier]


def get_inference_params(suite: Suite, timeout_multiplier: float = 1.0) -> dict[str, Any]:
    """Get inference parameters for a suite, with optional timeout scaling.

    Args:
        suite: The benchmark suite.
        timeout_multiplier: Multiplier for timeout (based on model speed).
            1.0 = no change, 2.0 = double timeout, etc.

    Returns dict with:
        - temperature: float (default 0.6)
        - max_tokens: int (default 512)
        - timeout: int (default 180, scaled by multiplier)
    """
    params = suite.inference_params.copy()
    params.setdefault("temperature", 0.6)
    params.setdefault("max_tokens", 512)
    params.setdefault("timeout", 180)

    # Apply timeout multiplier for slow models
    if timeout_multiplier > 1.0:
        base_timeout = params["timeout"]
        params["timeout"] = int(base_timeout * timeout_multiplier)

    return params


# Role to suite mapping - comprehensive testing
# Models get all applicable suites based on their role
ROLE_SUITE_MAP = {
    # Specialists - focused suites
    "coder": ["coder", "thinking", "general", "agentic", "instruction_precision"],
    "coding": ["coder", "thinking", "general", "agentic", "instruction_precision"],
    "thinking": ["thinking", "general", "math", "agentic"],
    "reasoning": ["thinking", "general", "math", "agentic"],
    "vision": ["vl"],  # VL models only run vision suite (text-only tasks are inefficient)
    "vl": ["vl"],
    "math": ["math", "thinking", "general"],

    # Versatile roles - broad testing
    "general": ["general", "agentic", "instruction_precision", "thinking", "coder", "math", "omniscience"],
    "frontdoor": ["general", "agentic", "instruction_precision", "coder", "math", "long_context", "longbench", "needle_parameterized", "omniscience"],
    "architect": ["thinking", "coder", "agentic", "general", "instruction_precision", "math", "long_context", "longbench", "needle_parameterized", "omniscience", "aa_lcr"],

    # Context specialists — full long-context battery
    "ingest": ["long_context", "general", "agentic", "longbench", "zeroscrolls", "leval", "ruler", "needle_parameterized", "aa_lcr"],
    "long_context": ["long_context", "general", "agentic", "longbench", "zeroscrolls", "leval", "ruler", "needle_parameterized", "aa_lcr"],

    # Workers - general purpose
    "worker": ["general", "thinking", "agentic"],
    "draft": ["general", "thinking"],
}


def _resolve_role_to_suites(role_name: str) -> Optional[list[str]]:
    """Try to map a role name (or candidate role name) to a suite list.

    Match order:
      1. Exact match (e.g. "frontdoor", "coder", "architect")
      2. Prefix-on-underscore (e.g. "architect_general" → "architect")
      3. Substring contains (e.g. "long_context_qwen35" → "long_context")

    Returns None if no match — caller decides fallback.
    """
    rl = role_name.lower()

    # 1. Exact
    if rl in ROLE_SUITE_MAP:
        return ROLE_SUITE_MAP[rl]

    # 2. Prefix on underscore (architect_general → architect; coder_escalation → coder)
    if "_" in rl:
        prefix = rl.split("_")[0]
        if prefix in ROLE_SUITE_MAP:
            return ROLE_SUITE_MAP[prefix]

    # 3. Substring (covers "qwen35_coder_baseline" → coder, etc.)
    for pattern, suites in ROLE_SUITE_MAP.items():
        if pattern in rl:
            return suites

    return None


def get_suites_for_role(role: str, registry=None) -> list[str]:
    """Get applicable suite names for a model based on candidate roles.

    Args:
        role: The model registry entry name.
        registry: Optional ModelRegistry instance to check candidate_roles.

    Returns:
        List of suite names to test (union of all candidate role suites).
    """
    # If registry provided, check for candidate_roles field. Apply the same
    # exact / prefix-on-underscore / substring resolution to each candidate so
    # that names like "architect_general" map to architect's suite list (instead
    # of falling through to the default "general" only).
    if registry is not None:
        role_config = registry.get_role_config(role)
        if role_config and "candidate_roles" in role_config:
            all_suites = set()
            for candidate in role_config["candidate_roles"]:
                resolved = _resolve_role_to_suites(candidate)
                if resolved:
                    all_suites.update(resolved)
            if all_suites:
                return list(all_suites)

    # Fallback: infer from the entry name itself.
    resolved = _resolve_role_to_suites(role)
    if resolved:
        return resolved

    # Default: general suite only
    return ["general"]


if __name__ == "__main__":
    print("=== Suite Loader Test ===\n")

    # List all suites
    suite_names = get_all_suite_names()
    print(f"Available suites: {suite_names}\n")

    # Load and show each suite
    for name in suite_names:
        suite = load_suite(name)
        if suite:
            params = get_inference_params(suite)
            print(f"Suite: {name}")
            print(f"  Domain: {suite.domain}")
            print(f"  Description: {suite.description}")
            print(f"  Questions: {len(suite.questions)}")
            print(f"  Inference params: {params}")

            # Show questions by tier
            for tier in [1, 2, 3]:
                tier_qs = get_questions_by_tier(suite, tier)
                if tier_qs:
                    print(f"  Tier {tier}: {[q.id for q in tier_qs]}")
            print()

    # Test role mapping
    print("\nRole -> Suite mapping:")
    test_roles = ["coder_escalation", "worker_math", "ingest_long_context", "frontdoor", "draft_qwen25"]
    for role in test_roles:
        suites = get_suites_for_role(role)
        print(f"  {role} -> {suites}")
