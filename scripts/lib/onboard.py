#!/usr/bin/env python3
from __future__ import annotations

"""
Model Onboarding Module

Implements the 7-step /new-model flow:
1. Collect and validate model path
2. Generate optimization options
3. Assign available drafts
4. Resolve and validate binary execution
5. Assign candidate roles
6. Push to model registry
7. Offer benchmark (handled by command, not this module)

This module provides functions for steps 1-6. The command file orchestrates
the flow and handles user interaction.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

try:
    from .registry import ModelRegistry, load_registry
    from .executor import Executor, Config, get_binary, validate_binaries
    from .output_parser import parse_output
except ImportError:
    from registry import ModelRegistry, load_registry
    from executor import Executor, Config, get_binary, validate_binaries
    from output_parser import parse_output


def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    """Read timeout from model_registry.yaml."""
    try:
        reg = load_registry()
        if reg and reg._raw:
            timeouts = reg._raw.get("runtime_defaults", {}).get("timeouts", {})
            cat_data = timeouts.get(category, {})
            return cat_data.get(key, timeouts.get("default", fallback))
    except Exception as e:
        pass
    return fallback


# Default paths
DEFAULT_MODEL_BASE = "/mnt/raid0/llm/lmstudio/models"
HEALTH_CHECK_TIMEOUT = _read_registry_timeout("scripts", "onboard_health", 60)
HEALTH_CHECK_PROMPT = "What is 2+2? Answer with just the number."


@dataclass
class ModelInfo:
    """Information detected from a model file."""

    path: str  # Absolute path
    relative_path: str  # Relative to model base
    filename: str
    name: str  # Short name (no quant, no extension)
    architecture: str  # dense, moe, qwen3moe, ssm_moe_hybrid, etc.
    family: str  # Qwen2.5, Qwen3, Llama, DeepSeek-R1-Distill, etc.
    quantization: str  # Q4_K_M, Q8_0, etc.
    size_gb: float
    tier: str  # A, B, C, D
    is_draft: bool


@dataclass
class HealthCheckResult:
    """Result of a health check run."""

    success: bool
    tokens_per_second: Optional[float] = None
    flags_used: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class OnboardingResult:
    """Complete result of the onboarding process."""

    model_info: ModelInfo
    configs: list[Config]
    compatible_drafts: list[str]  # For target models
    compatible_targets: list[str]  # For draft models (patterns)
    health_check: HealthCheckResult
    suggested_roles: list[str]
    suggested_role_name: str
    registry_entry: dict[str, Any]


# =============================================================================
# Step 1: Collect and Validate Model Path
# =============================================================================


def detect_architecture(filename: str) -> str:
    """Detect model architecture from filename.

    Args:
        filename: The model filename.

    Returns:
        Architecture string.
    """
    name = filename.lower()

    # Qwen3 variants
    if "qwen3" in name:
        if "next" in name:
            return "ssm_moe_hybrid"  # SSM - NO speculative decoding!
        if re.search(r"a\d+b", name):  # Has AxB pattern (A3B, A22B, etc.)
            if "vl" in name:
                return "qwen3vlmoe"
            return "qwen3moe"

    # Other MoE architectures
    if "mixtral" in name:
        return "mixtral"
    if "deepseek" in name and ("moe" in name or re.search(r"\d+x", name)):
        return "deepseek2"
    if "glm" in name and "moe" in name:
        return "glm4moe"

    return "dense"


def detect_family(filename: str) -> str:
    """Detect model family from filename.

    Args:
        filename: The model filename.

    Returns:
        Family string.
    """
    name = filename.lower()

    if "qwen2.5" in name or "qwen2_5" in name or "qwen25" in name:
        if "coder" in name:
            return "Qwen2.5-Coder"
        if "math" in name:
            return "Qwen2.5-Math"
        if "vl" in name:
            return "Qwen2.5-VL"
        return "Qwen2.5"

    if "qwen3" in name:
        if "coder" in name:
            return "Qwen3-Coder"
        if "vl" in name:
            return "Qwen3-VL"
        if "next" in name:
            return "Qwen3-Next"
        return "Qwen3"

    if "qwen2" in name:
        return "Qwen2"

    if "deepseek-r1-distill" in name:
        if "qwen" in name:
            return "DeepSeek-R1-Distill-Qwen"
        if "llama" in name:
            return "DeepSeek-R1-Distill-Llama"
        return "DeepSeek-R1-Distill"

    if "deepseek" in name:
        return "DeepSeek"

    if "llama-3.2" in name or "llama_3_2" in name or "llama32" in name:
        return "Llama-3.2"
    if "llama-3.1" in name or "llama_3_1" in name or "llama31" in name:
        return "Llama-3.1"
    if "llama-3" in name or "llama_3" in name or "llama3" in name:
        return "Llama-3"
    if "llama" in name:
        return "Llama"

    if "gemma" in name:
        return "Gemma"
    if "glm" in name:
        return "GLM"
    if "hermes" in name:
        return "Hermes"

    return "Unknown"


def detect_quantization(filename: str) -> str:
    """Detect quantization from filename.

    Args:
        filename: The model filename.

    Returns:
        Quantization string (e.g., Q4_K_M, Q8_0).
    """
    # Common patterns
    patterns = [
        r"[._-](Q[0-9]+_K_[A-Z]+)",  # Q4_K_M, Q6_K_L
        r"[._-](Q[0-9]+_K)",  # Q4_K
        r"[._-](Q[0-9]+_[0-9]+)",  # Q8_0
        r"[._-](Q[0-9]+)",  # Q4
        r"[._-](fp16|fp32|bf16)",  # Float formats
        r"[._-](i1|i2)",  # Integer formats
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return "Unknown"


def extract_short_name(filename: str) -> str:
    """Extract short model name from filename.

    Strips: quantization suffix, GGUF extension, path components.

    Args:
        filename: The model filename.

    Returns:
        Short model name.
    """
    name = Path(filename).stem

    # Remove quantization suffixes
    quant_patterns = [
        r"[._-]Q[0-9]+_K_[A-Z]+$",
        r"[._-]Q[0-9]+_K$",
        r"[._-]Q[0-9]+_[0-9]+$",
        r"[._-]Q[0-9]+$",
        r"[._-](?:fp16|fp32|bf16|i1|i2)$",
        r"[._-]GGUF$",
        r"[._-]gguf$",
    ]

    for pattern in quant_patterns:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)

    return name


def estimate_tier(size_gb: float, is_draft: bool) -> str:
    """Estimate model tier based on size.

    Args:
        size_gb: Model size in GB.
        is_draft: Whether this is a draft model.

    Returns:
        Tier string (A, B, C, D).
    """
    if is_draft or size_gb < 2:
        return "D"
    if size_gb < 10:
        return "C"
    if size_gb < 50:
        return "B"
    return "A"


def collect_model_info(
    path: str,
    model_base: str = DEFAULT_MODEL_BASE,
) -> ModelInfo:
    """Collect and validate model path, detect properties.

    Args:
        path: Path to model file (absolute or relative).
        model_base: Base path for relative paths.

    Returns:
        ModelInfo with detected properties.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    # Resolve path
    if path.startswith("/"):
        abs_path = path
    else:
        abs_path = str(Path(model_base) / path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model file not found: {abs_path}")

    # Get relative path for registry
    if abs_path.startswith(model_base):
        relative_path = abs_path[len(model_base) :].lstrip("/")
    else:
        relative_path = abs_path

    filename = Path(abs_path).name
    size_gb = os.path.getsize(abs_path) / (1024**3)

    # Detect properties
    architecture = detect_architecture(filename)
    family = detect_family(filename)
    quantization = detect_quantization(filename)
    name = extract_short_name(filename)

    # Determine if draft based on size and name patterns
    is_draft = (
        size_gb < 2
        or "0.5b" in filename.lower()
        or "0.6b" in filename.lower()
        or "1b" in filename.lower()
        or "1.5b" in filename.lower()
        or "1.7b" in filename.lower()
        or "pard" in filename.lower()
    )

    tier = estimate_tier(size_gb, is_draft)

    return ModelInfo(
        path=abs_path,
        relative_path=relative_path,
        filename=filename,
        name=name,
        architecture=architecture,
        family=family,
        quantization=quantization,
        size_gb=size_gb,
        tier=tier,
        is_draft=is_draft,
    )


# =============================================================================
# Step 2: Generate Optimization Options
# =============================================================================


def generate_optimization_options(
    model_info: ModelInfo,
    registry: ModelRegistry,
) -> list[Config]:
    """Generate applicable optimization configs for a model.

    Args:
        model_info: Detected model information.
        registry: Model registry.

    Returns:
        List of Config objects that can be tested.
    """
    # Create a temporary executor (skip validation since model not in registry yet)
    executor = Executor(registry, validate=True)

    # Generate configs based on architecture
    # Use a placeholder role since model isn't in registry yet
    # We'll manually check constraints
    configs = [Config.baseline()]

    arch = model_info.architecture

    if arch in ("moe", "qwen3moe", "qwen3vlmoe", "mixtral", "deepseek2"):
        # MoE models - expert reduction only
        override_key = _get_moe_override_key(arch)
        for experts in [2, 4, 6, 8]:
            configs.append(Config.moe(experts, override_key))
        # MoE + lookup compound
        configs.append(Config.compound_moe_lookup(4, override_key, 4))

    elif arch in ("ssm_moe_hybrid", "qwen3next"):
        # SSM models - expert reduction ONLY, no speculation
        override_key = _get_moe_override_key(arch)
        for experts in [2, 4, 6, 8]:
            configs.append(Config.moe(experts, override_key))

    else:
        # Dense models - spec decode (if drafts available) + lookup
        if not model_info.is_draft:
            # Lookup configs
            for ngram in [3, 4, 5]:
                configs.append(Config.lookup(ngram))
            # Spec decode configs will be added after draft discovery

    return configs


def _get_moe_override_key(architecture: str) -> str:
    """Get the MoE override key for an architecture."""
    keys = {
        "moe": "qwen3moe.expert_used_count",
        "qwen3moe": "qwen3moe.expert_used_count",
        "qwen3vlmoe": "qwen3vlmoe.expert_used_count",
        "qwen3next": "qwen3next.expert_used_count",
        "ssm_moe_hybrid": "qwen3next.expert_used_count",
        "mixtral": "mixtral.expert_used_count",
        "deepseek2": "deepseek2.expert_used_count",
        "glm4moe": "glm4moe.expert_used_count",
    }
    return keys.get(architecture, "qwen3moe.expert_used_count")


# =============================================================================
# Step 3: Assign Available Drafts
# =============================================================================


def find_compatible_drafts(
    model_info: ModelInfo,
    registry: ModelRegistry,
) -> list[str]:
    """Find compatible draft models for a target model.

    Args:
        model_info: Detected model information.
        registry: Model registry.

    Returns:
        List of draft role names that are compatible.
    """
    # MoE and SSM models don't use speculative decoding
    if model_info.architecture in (
        "moe",
        "qwen3moe",
        "qwen3vlmoe",
        "mixtral",
        "deepseek2",
        "ssm_moe_hybrid",
        "qwen3next",
    ):
        return []

    # Draft models don't need drafts
    if model_info.is_draft:
        return []

    # Find drafts by matching compatible_targets patterns
    compatible = []
    model_name = model_info.name

    for draft_role in registry.get_all_roles():
        draft_config = registry.get_role_config(draft_role)
        if not draft_config:
            continue

        # Only consider Tier D (draft) models
        if draft_config.get("tier") != "D":
            continue

        # Check if model name matches any compatible_targets pattern
        patterns = draft_config.get("compatible_targets", [])
        for pattern in patterns:
            if pattern.lower() in model_name.lower():
                compatible.append(draft_role)
                break

    return compatible


def generate_compatible_targets_patterns(model_info: ModelInfo) -> list[str]:
    """Generate compatible_targets patterns for a draft model.

    Args:
        model_info: Detected model information.

    Returns:
        List of target patterns for the draft's compatible_targets field.
    """
    if not model_info.is_draft:
        return []

    family = model_info.family
    patterns = []

    # Generate patterns based on family
    if family.startswith("Qwen2.5"):
        patterns.extend(["Qwen2.5", "Qwen2"])
    elif family.startswith("Qwen3"):
        patterns.extend(["Qwen3"])
    elif family.startswith("Qwen2"):
        patterns.extend(["Qwen2"])
    elif family.startswith("DeepSeek-R1-Distill-Qwen"):
        patterns.extend(["DeepSeek-R1-Distill-Qwen", "Qwen"])
    elif family.startswith("DeepSeek-R1-Distill-Llama"):
        patterns.extend(["DeepSeek-R1-Distill-Llama", "Llama"])
    elif family.startswith("Llama"):
        patterns.extend(["Llama"])
    elif family.startswith("Gemma"):
        patterns.extend(["Gemma"])

    return patterns


# =============================================================================
# Step 4: Resolve and Validate Binary Execution
# =============================================================================


def run_health_check(
    model_info: ModelInfo,
    registry: ModelRegistry,
) -> HealthCheckResult:
    """Run health check to verify model launches correctly.

    Tries different flag combinations and records what works.

    Args:
        model_info: Detected model information.
        registry: Model registry.

    Returns:
        HealthCheckResult with success status and flags used.
    """
    # Validate binaries exist
    try:
        validate_binaries(registry)
    except FileNotFoundError as e:
        return HealthCheckResult(success=False, error_message=str(e))

    binary = get_binary("completion", registry)

    # Flag combinations to try (in order)
    flag_combos = [
        ["--no-conversation"],  # Most common for Instruct models
        [],  # No special flags
        ["--jinja"],  # Some models need this
        ["--jinja", "--no-conversation"],  # Both
    ]

    # Get MoE override if applicable
    moe_override = []
    if model_info.architecture in (
        "moe",
        "qwen3moe",
        "qwen3vlmoe",
        "mixtral",
        "deepseek2",
        "ssm_moe_hybrid",
        "qwen3next",
    ):
        key = _get_moe_override_key(model_info.architecture)
        moe_override = ["--override-kv", f"{key}=int:4"]

    # Write test prompt to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/mnt/raid0/llm/tmp"
    ) as f:
        f.write(HEALTH_CHECK_PROMPT)
        prompt_file = f.name

    try:
        for flags in flag_combos:
            cmd = [
                "timeout",
                str(HEALTH_CHECK_TIMEOUT),
                "env",
                "OMP_NUM_THREADS=1",
                "numactl",
                "--interleave=all",
                binary,
                "-m",
                model_info.path,
                "-t",
                "96",
                "-n",
                "16",
                "--temp",
                "0",
                "-f",
                prompt_file,
            ]
            cmd.extend(flags)
            cmd.extend(moe_override)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=HEALTH_CHECK_TIMEOUT + 10,
                )

                output = result.stdout + result.stderr

                # Check for success indicators
                if "tokens per second" in output:
                    # Extract speed
                    parsed = parse_output(output)
                    return HealthCheckResult(
                        success=True,
                        tokens_per_second=parsed.tokens_per_second,
                        flags_used=flags,
                    )

            except subprocess.TimeoutExpired:
                continue
            except Exception as e:
                continue

        # All attempts failed
        return HealthCheckResult(
            success=False,
            error_message="All flag combinations failed. Check model file integrity.",
        )

    finally:
        if os.path.exists(prompt_file):
            os.unlink(prompt_file)


# =============================================================================
# Step 5: Assign Candidate Roles
# =============================================================================


def suggest_candidate_roles(model_info: ModelInfo) -> list[str]:
    """Suggest candidate roles based on model properties.

    Args:
        model_info: Detected model information.

    Returns:
        List of suggested role names.
    """
    name = model_info.name.lower()
    roles = []

    # Draft models only get draft role
    if model_info.is_draft:
        return ["draft"]

    # Pattern-based role assignment
    if "coder" in name:
        roles.extend(["coder", "general"])
    if "thinking" in name or "r1" in name:
        roles.extend(["thinking", "general"])
    if "math" in name:
        roles.extend(["math", "thinking"])
    if "vl" in name:
        roles.append("vision")

    # Size-based additions
    if model_info.size_gb > 50:
        if "architect" not in roles:
            roles.append("architect")
        if "ingest" not in roles:
            roles.append("ingest")

    # Default to general if nothing else
    if not roles:
        roles = ["general"]

    # Add worker for smaller models
    if model_info.tier in ("C", "D") and "worker" not in roles:
        roles.append("worker")

    return list(set(roles))  # Deduplicate


def generate_role_name(model_info: ModelInfo, candidate_roles: list[str]) -> str:
    """Generate a registry role name for the model.

    Args:
        model_info: Detected model information.
        candidate_roles: List of candidate roles.

    Returns:
        Suggested role name (e.g., 'thinking_qwen3_30b').
    """
    # Primary role prefix
    if "draft" in candidate_roles:
        prefix = "draft"
    elif "thinking" in candidate_roles:
        prefix = "thinking"
    elif "coder" in candidate_roles:
        prefix = "coder"
    elif "math" in candidate_roles:
        prefix = "math"
    elif "vision" in candidate_roles:
        prefix = "vision"
    elif "architect" in candidate_roles:
        prefix = "architect"
    elif "ingest" in candidate_roles:
        prefix = "ingest"
    else:
        prefix = "general"

    # Clean name for role suffix
    name = model_info.name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    # Truncate if too long
    if len(name) > 30:
        name = name[:30].rstrip("_")

    return f"{prefix}_{name}"


# =============================================================================
# Step 6: Build Registry Entry
# =============================================================================


def build_registry_entry(
    model_info: ModelInfo,
    candidate_roles: list[str],
    health_check: HealthCheckResult,
    compatible_targets: list[str],
) -> dict[str, Any]:
    """Build a complete registry entry for the model.

    Args:
        model_info: Detected model information.
        candidate_roles: List of candidate roles.
        health_check: Health check result.
        compatible_targets: For draft models, target patterns.

    Returns:
        Registry entry dict matching the schema.
    """
    entry: dict[str, Any] = {
        "tier": model_info.tier,
        "description": f"Auto-added via /new-model on {date.today().isoformat()}",
        "model": {
            "name": model_info.name,
            "path": model_info.relative_path,
            "quant": model_info.quantization,
            "size_gb": round(model_info.size_gb, 1),
            "architecture": model_info.architecture,
        },
        "candidate_roles": candidate_roles,
    }

    # Add acceleration config for MoE models
    if model_info.architecture in (
        "moe",
        "qwen3moe",
        "qwen3vlmoe",
        "mixtral",
        "deepseek2",
        "ssm_moe_hybrid",
        "qwen3next",
    ):
        entry["acceleration"] = {
            "type": "moe_expert_reduction",
            "override_key": _get_moe_override_key(model_info.architecture),
        }

    # Add constraints for SSM models
    if model_info.architecture in ("ssm_moe_hybrid", "qwen3next"):
        entry["constraints"] = {
            "forbid": ["speculative_decoding", "prompt_lookup"],
        }

    # Add performance data from health check
    if health_check.success:
        entry["performance"] = {
            "baseline_tps": health_check.tokens_per_second,
            "health_check_date": date.today().isoformat(),
        }

    # Add launch quirks if special flags needed
    if health_check.flags_used:
        entry["launch_flags"] = health_check.flags_used

    # Add compatible_targets for draft models
    if model_info.is_draft and compatible_targets:
        entry["compatible_targets"] = compatible_targets

    return entry


# =============================================================================
# Main Onboarding Function
# =============================================================================


def onboard_model(
    path: str,
    registry: Optional[ModelRegistry] = None,
    model_base: str = DEFAULT_MODEL_BASE,
) -> OnboardingResult:
    """Run the complete onboarding process for a model.

    Args:
        path: Path to model file.
        registry: Optional registry instance.
        model_base: Base path for relative paths.

    Returns:
        OnboardingResult with all collected information.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If model path already in registry.
    """
    if registry is None:
        registry = load_registry()

    # Step 1: Collect and validate
    model_info = collect_model_info(path, model_base)

    # Check if already in registry
    existing_role = registry.path_exists_in_registry(model_info.path)
    if existing_role:
        raise ValueError(
            f"Model path already in registry as role '{existing_role}'"
        )

    # Step 2: Generate optimization options
    configs = generate_optimization_options(model_info, registry)

    # Step 3: Assign available drafts (or generate compatible_targets)
    if model_info.is_draft:
        compatible_drafts = []
        compatible_targets = generate_compatible_targets_patterns(model_info)
    else:
        compatible_drafts = find_compatible_drafts(model_info, registry)
        compatible_targets = []

        # Add spec decode configs for each draft
        for draft_role in compatible_drafts:
            draft_path = registry.get_model_path(draft_role)
            if draft_path and os.path.exists(draft_path):
                for k in [4, 8, 16, 24, 32, 48]:
                    configs.append(Config.spec(k, draft_path, draft_role))

    # Step 4: Run health check
    health_check = run_health_check(model_info, registry)

    # Step 5: Suggest candidate roles
    suggested_roles = suggest_candidate_roles(model_info)
    role_name = generate_role_name(model_info, suggested_roles)

    # Step 6: Build registry entry
    registry_entry = build_registry_entry(
        model_info,
        suggested_roles,
        health_check,
        compatible_targets,
    )

    return OnboardingResult(
        model_info=model_info,
        configs=configs,
        compatible_drafts=compatible_drafts,
        compatible_targets=compatible_targets,
        health_check=health_check,
        suggested_roles=suggested_roles,
        suggested_role_name=role_name,
        registry_entry=registry_entry,
    )


# =============================================================================
# CLI for Testing
# =============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python onboard.py <model_path>")
        sys.exit(1)

    path = sys.argv[1]

    print(f"=== Onboarding Model ===\n")
    print(f"Path: {path}\n")

    try:
        result = onboard_model(path)

        print("--- Model Info ---")
        print(f"  Name: {result.model_info.name}")
        print(f"  Architecture: {result.model_info.architecture}")
        print(f"  Family: {result.model_info.family}")
        print(f"  Quantization: {result.model_info.quantization}")
        print(f"  Size: {result.model_info.size_gb:.1f} GB")
        print(f"  Tier: {result.model_info.tier}")
        print(f"  Is Draft: {result.model_info.is_draft}")

        print("\n--- Optimization Configs ---")
        print(f"  Total: {len(result.configs)}")
        config_types = {}
        for c in result.configs:
            config_types[c.config_type] = config_types.get(c.config_type, 0) + 1
        for ctype, count in config_types.items():
            print(f"    {ctype}: {count}")

        if result.compatible_drafts:
            print("\n--- Compatible Drafts ---")
            for d in result.compatible_drafts:
                print(f"  - {d}")

        if result.compatible_targets:
            print("\n--- Compatible Targets (patterns) ---")
            for t in result.compatible_targets:
                print(f"  - {t}")

        print("\n--- Health Check ---")
        print(f"  Success: {result.health_check.success}")
        if result.health_check.success:
            print(f"  Speed: {result.health_check.tokens_per_second:.1f} t/s")
            if result.health_check.flags_used:
                print(f"  Flags: {result.health_check.flags_used}")
        else:
            print(f"  Error: {result.health_check.error_message}")

        print("\n--- Suggested Roles ---")
        print(f"  Roles: {result.suggested_roles}")
        print(f"  Role name: {result.suggested_role_name}")

        print("\n--- Registry Entry ---")
        import yaml

        print(yaml.dump({result.suggested_role_name: result.registry_entry}, default_flow_style=False))

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
