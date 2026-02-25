#!/usr/bin/env python3
from __future__ import annotations

"""
Model Registry Parser for Benchmark System

Parses model_registry.yaml and provides functions for:
- Loading registry data
- Getting model configurations by role
- Resolving model paths
- Getting compatible drafts for speculative decoding
- Querying runtime quirks

This module is shared with the orchestrator project.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml


# Default paths
DEFAULT_REGISTRY_PATH = "/mnt/raid0/llm/claude/orchestration/model_registry.yaml"
DEFAULT_MODEL_BASE_PATH = "/mnt/raid0/llm/lmstudio/models"


class ModelRegistry:
    """Interface to the model registry."""

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load the registry YAML file."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")

        with open(self.registry_path) as f:
            self._data = yaml.safe_load(f)

    def reload(self) -> None:
        """Reload the registry from disk."""
        self._load()

    @property
    def data(self) -> dict[str, Any]:
        """Get raw registry data."""
        return self._data

    @property
    def runtime_defaults(self) -> dict[str, Any]:
        """Get runtime defaults."""
        return self._data.get("runtime_defaults", {})

    @property
    def model_base_path(self) -> str:
        """Get the base path for models."""
        return self.runtime_defaults.get("model_base_path", DEFAULT_MODEL_BASE_PATH)

    def get_all_roles(self, include_deprecated: bool = False) -> list[str]:
        """Get all role names from the registry.

        Args:
            include_deprecated: If True, include deprecated roles.

        Returns:
            List of role names.
        """
        roles = self._data.get("roles", {})
        if include_deprecated:
            return list(roles.keys())

        return [
            role for role, config in roles.items()
            if not config.get("deprecated", False)
        ]

    def get_role_config(self, role: str) -> Optional[dict[str, Any]]:
        """Get the full configuration for a role.

        Args:
            role: The role name (e.g., 'coder_escalation').

        Returns:
            Role configuration dict or None if not found.
        """
        return self._data.get("roles", {}).get(role)

    def get_model_path(self, role: str) -> Optional[str]:
        """Get the absolute model path for a role.

        Args:
            role: The role name.

        Returns:
            Absolute path to the model file, or None.
        """
        config = self.get_role_config(role)
        if not config:
            return None

        model = config.get("model", {})
        rel_path = model.get("path")
        if not rel_path:
            return None

        # Handle absolute vs relative paths
        if rel_path.startswith("/"):
            return rel_path

        return str(Path(self.model_base_path) / rel_path)

    def get_mmproj_path(self, role: str) -> Optional[str]:
        """Get the absolute mmproj path for a VL model role.

        Args:
            role: The role name.

        Returns:
            Absolute path to the mmproj file, or None if not a VL model.
        """
        config = self.get_role_config(role)
        if not config:
            return None

        model = config.get("model", {})
        rel_path = model.get("mmproj_path")
        if not rel_path:
            return None

        # Handle absolute vs relative paths
        if rel_path.startswith("/"):
            return rel_path

        return str(Path(self.model_base_path) / rel_path)

    def get_architecture(self, role: str) -> str:
        """Get the model architecture for a role.

        Args:
            role: The role name.

        Returns:
            Architecture string (e.g., 'dense', 'moe', 'qwen3moe', 'ssm_moe_hybrid').
            Returns 'dense' as default if not specified.
        """
        config = self.get_role_config(role)
        if not config:
            return "dense"

        model = config.get("model", {})
        return model.get("architecture", "dense")

    def get_tier(self, role: str) -> Optional[str]:
        """Get the tier for a role (A, B, C, D)."""
        config = self.get_role_config(role)
        return config.get("tier") if config else None

    def get_acceleration(self, role: str) -> dict[str, Any]:
        """Get acceleration configuration for a role.

        Returns dict with:
            - type: 'moe_expert_reduction', 'speculative_decoding', 'prompt_lookup', 'none'
            - experts: int (for MoE)
            - override_key: str (for MoE)
            - draft_role: str (for spec decode)
            - k: int (for spec decode)
            - ngram_min: int (for prompt lookup)
        """
        config = self.get_role_config(role)
        if not config:
            return {"type": "none"}

        return config.get("acceleration", {"type": "none"})

    def get_constraints(self, role: str) -> dict[str, Any]:
        """Get constraints for a role.

        Returns dict with:
            - forbid: list of forbidden optimization types
            - reason: explanation
        """
        config = self.get_role_config(role)
        if not config:
            return {}

        return config.get("constraints", {})

    def get_forbidden_configs(self, role: str) -> list[str]:
        """Get list of forbidden optimization types for a role.

        Checks:
        - forbidden_configs (preferred format)
        - constraints.forbid (legacy format)
        - acceleration.disallowed (new format for VL models, SSM, etc.)
        """
        forbidden = []

        config = self.get_role_config(role)
        if config:
            # Check preferred format: forbidden_configs at role level
            forbidden.extend(config.get("forbidden_configs", []))

            # Check acceleration.disallowed (newer format)
            accel = config.get("acceleration", {})
            disallowed = accel.get("disallowed", [])
            forbidden.extend(disallowed)

        # Check legacy constraints.forbid
        constraints = self.get_constraints(role)
        forbidden.extend(constraints.get("forbid", []))

        return list(set(forbidden))  # Dedupe

    def get_quirks(self, role: str) -> list[dict[str, str]]:
        """Get runtime quirks for a role's model.

        Returns list of dicts with:
            - issue: description of the problem
            - workaround: how to fix it
            - discovered: date string
        """
        config = self.get_role_config(role)
        if not config:
            return []

        # Get model name to look up in runtime_quirks
        model_name = config.get("model", {}).get("name", "")

        # Map model names to quirk keys
        quirk_keys = self._get_quirk_keys_for_model(model_name)

        result = []
        runtime_quirks = self._data.get("runtime_quirks", {})
        for key in quirk_keys:
            if key in runtime_quirks:
                result.extend(runtime_quirks[key].get("quirks", []))

        return result

    def _get_quirk_keys_for_model(self, model_name: str) -> list[str]:
        """Map model name to runtime_quirks keys."""
        keys = []

        # Normalize model name for matching
        name_lower = model_name.lower()

        if "qwen2.5-coder-32b" in name_lower:
            keys.append("qwen25_coder_32b_instruct")
        if "qwen3-coder-30b" in name_lower:
            keys.append("qwen3_coder_30b_a3b")
        if "qwen3-coder-53b" in name_lower:
            keys.append("qwen3_coder_53b_a3b")
        if "qwen3-coder-480b" in name_lower:
            keys.append("qwen3_coder_480b")
        if "qwen3-next-80b" in name_lower:
            keys.append("qwen3_next_80b")
        if "qwen3-235b" in name_lower:
            keys.append("qwen3_235b_a22b")
        if "meta-llama-3-8b" in name_lower:
            keys.append("meta_llama_3_8b")
        if "qwen2.5-math-7b" in name_lower:
            keys.append("qwen25_math_7b")
        if "qwen2.5-vl-7b" in name_lower:
            keys.append("qwen25_vl_7b")
        if "qwen3-vl-30b" in name_lower:
            keys.append("qwen3_vl_30b")
        if "qwen2.5-coder-0.5b" in name_lower:
            keys.append("qwen25_coder_0_5b")
        if "qwen2.5-0.5b" in name_lower:
            keys.append("qwen25_0_5b")

        return keys

    def get_drafts_for_model(self, role: str) -> list[str]:
        """Get list of compatible draft model roles for speculative decoding.

        Args:
            role: The target model role.

        Returns:
            List of draft role names that are compatible.
        """
        # Check if this model is forbidden from spec decode
        forbidden = self.get_forbidden_configs(role)
        if "speculative_decoding" in forbidden:
            return []

        # Get the target model's architecture family
        config = self.get_role_config(role)
        if not config:
            return []

        # Draft models (Tier D) should never use speculative decoding
        # They're already small and fast - no need for a draft of a draft
        if config.get("tier") == "D":
            return []

        model_name = config.get("model", {}).get("name", "")

        # Find draft models by checking compatible_targets
        compatible_drafts = []
        for draft_role in self.get_all_roles():
            draft_config = self.get_role_config(draft_role)
            if not draft_config:
                continue

            # Only consider Tier D (draft) models
            if draft_config.get("tier") != "D":
                continue

            # Check if target model is in compatible_targets
            # Normalize underscores/hyphens for matching (registry uses both)
            compatible_targets = draft_config.get("compatible_targets", [])
            model_name_normalized = model_name.lower().replace("_", "-")
            for target in compatible_targets:
                target_normalized = target.lower().replace("_", "-")
                if target_normalized in model_name_normalized:
                    compatible_drafts.append(draft_role)
                    break

        return compatible_drafts

    def get_targets_for_draft(self, draft_role: str) -> list[str]:
        """Get list of target model roles that a draft can be used with.

        This is the inverse of get_drafts_for_model().

        Args:
            draft_role: The draft model role.

        Returns:
            List of target role names that this draft is compatible with.
        """
        draft_config = self.get_role_config(draft_role)
        if not draft_config:
            return []

        # Verify this is a draft model (Tier D)
        if draft_config.get("tier") != "D":
            return []

        # Get compatible_targets patterns from the draft
        compatible_targets = draft_config.get("compatible_targets", [])
        if not compatible_targets:
            return []

        # Find all target models that match the patterns
        matching_targets = []
        for role in self.get_all_roles():
            config = self.get_role_config(role)
            if not config:
                continue

            # Skip other draft models (Tier D)
            if config.get("tier") == "D":
                continue

            # Skip models that forbid speculative decoding
            forbidden = self.get_forbidden_configs(role)
            if "speculative_decoding" in forbidden:
                continue

            # Check if model name matches any compatible_targets pattern
            model_name = config.get("model", {}).get("name", "")
            for pattern in compatible_targets:
                if pattern.lower() in model_name.lower():
                    matching_targets.append(role)
                    break

        return matching_targets

    def get_command_template(self, accel_type: str) -> Optional[str]:
        """Get command template for an acceleration type."""
        templates = self._data.get("command_templates", {})
        return templates.get(accel_type)

    def get_moe_override_key(self, role: str) -> Optional[str]:
        """Get the MoE override key for a role."""
        accel = self.get_acceleration(role)
        return accel.get("override_key")

    def get_baseline_experts(self, role: str) -> int:
        """Get the baseline expert count for an MoE model.

        Returns the model's default expert count (from GGUF metadata).
        Used to avoid testing MOE configs that match baseline.
        """
        accel = self.get_acceleration(role)
        return accel.get("baseline_experts", 8)  # Default to 8 if not specified

    def get_max_context(self, role: str) -> int:
        """Get maximum context length for a model.

        Priority:
        1. model.max_context in role definition
        2. context_limits based on model family
        3. server_defaults.context_length
        4. Hardcoded 8192 fallback

        Args:
            role: The role name.

        Returns:
            Maximum context length in tokens.
        """
        config = self.get_role_config(role)
        defaults = self.runtime_defaults

        # 1. Check for explicit max_context in model definition
        if config:
            model = config.get("model", {})
            if "max_context" in model:
                return model["max_context"]

            # 2. Check context_limits based on model family
            model_name = model.get("name", "").lower()
            context_limits = defaults.get("context_limits", {})

            # Match model family patterns
            if "llama-3.1" in model_name or "llama-3-1" in model_name:
                return context_limits.get("llama3_instruct", 131072)
            elif "llama-3" in model_name and "instruct" in model_name:
                # Llama 3 Instruct: 8K (not extended like 3.1)
                return context_limits.get("llama3", 8192)
            elif "llama-3" in model_name:
                return context_limits.get("llama3", 8192)
            elif "llama-2" in model_name or "llama2" in model_name:
                return context_limits.get("llama2", 4096)
            elif "qwen3" in model_name:
                return context_limits.get("qwen3", 131072)
            elif "qwen2" in model_name:
                return context_limits.get("qwen2", 131072)
            elif "deepseek-r1" in model_name:
                return context_limits.get("deepseek_r1", 65536)
            elif "gemma-3" in model_name or "gemma3" in model_name:
                return context_limits.get("gemma3", 131072)

        # 3. Fall back to server_defaults.context_length
        server_defaults = defaults.get("server_defaults", {})
        if "context_length" in server_defaults:
            return server_defaults["context_length"]

        # 4. Hardcoded fallback
        return context_limits.get("default", 8192)

    def get_flash_attention(self, role: str) -> bool:
        """Check if flash attention should be enabled for a model.

        Priority:
        1. constraints.flash_attention in role definition
        2. server_defaults.flash_attention
        3. Default to True (most models support it)

        Args:
            role: The role name.

        Returns:
            True if flash attention should be enabled.
        """
        config = self.get_role_config(role)
        defaults = self.runtime_defaults

        # 1. Check for explicit flash_attention in constraints
        if config:
            constraints = config.get("constraints", {})
            if "flash_attention" in constraints:
                return constraints["flash_attention"]

        # 2. Fall back to server_defaults
        server_defaults = defaults.get("server_defaults", {})
        return server_defaults.get("flash_attention", True)

    def get_ubatch_size(self, role: str) -> int:
        """Get ubatch size for prompt processing.

        Priority:
        1. constraints.ubatch_size in role definition
        2. server_defaults.ubatch_size
        3. Default to 512 (llama.cpp default)

        Args:
            role: The role name.

        Returns:
            Ubatch size for prompt processing.
        """
        config = self.get_role_config(role)
        defaults = self.runtime_defaults

        # 1. Check for explicit ubatch_size in constraints
        if config:
            constraints = config.get("constraints", {})
            if "ubatch_size" in constraints:
                return constraints["ubatch_size"]

        # 2. Fall back to server_defaults
        server_defaults = defaults.get("server_defaults", {})
        return server_defaults.get("ubatch_size", 512)

    def get_baseline_tps(self, role: str) -> Optional[float]:
        """Get baseline tokens-per-second for a role.

        Args:
            role: The role name.

        Returns:
            Baseline TPS float, or None if not available.
        """
        config = self.get_role_config(role)
        if not config:
            return None

        performance = config.get("performance", {})
        return performance.get("baseline_tps")

    def get_timeout_multiplier(self, role: str, reference_tps: float = 20.0) -> float:
        """Calculate timeout multiplier based on model speed.

        Slower models need proportionally longer timeouts.
        A model at 2 t/s needs 10x the timeout of a 20 t/s model.

        Args:
            role: The role name.
            reference_tps: Reference speed for multiplier=1.0 (default: 20 t/s)

        Returns:
            Timeout multiplier (minimum 1.0, no maximum).
        """
        baseline_tps = self.get_baseline_tps(role)

        # If no TPS data, use a conservative multiplier of 2.0
        if baseline_tps is None or baseline_tps <= 0:
            return 2.0

        # Calculate multiplier: slower = higher multiplier
        multiplier = reference_tps / baseline_tps

        # Minimum 1.0 (fast models don't get shorter timeouts)
        return max(1.0, multiplier)

    def add_model_entry(self, role: str, entry: dict[str, Any]) -> None:
        """Add a new model entry to the registry and save to disk.

        Args:
            role: The role name (e.g., 'thinking_new_model').
            entry: The complete entry dict matching the registry schema.

        Raises:
            ValueError: If role already exists or entry is invalid.
        """
        # Validate role doesn't exist
        if role in self._data.get("roles", {}):
            raise ValueError(f"Role '{role}' already exists in registry")

        # Validate required fields
        required_fields = ["tier", "model"]
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Missing required field: {field}")

        model = entry.get("model", {})
        if "path" not in model:
            raise ValueError("Missing required field: model.path")
        if "architecture" not in model:
            raise ValueError("Missing required field: model.architecture")

        # Add the entry
        if "roles" not in self._data:
            self._data["roles"] = {}
        self._data["roles"][role] = entry

        # Save to disk
        self._save()

    def _save(self) -> None:
        """Save the registry back to disk."""
        with open(self.registry_path, "w") as f:
            yaml.dump(
                self._data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )

    def role_exists(self, role: str) -> bool:
        """Check if a role exists in the registry."""
        return role in self._data.get("roles", {})

    def path_exists_in_registry(self, model_path: str) -> Optional[str]:
        """Check if a model path already exists in the registry.

        Args:
            model_path: Absolute or relative path to check.

        Returns:
            Role name if found, None otherwise.
        """
        # Normalize the path for comparison
        check_path = model_path
        if not check_path.startswith("/"):
            check_path = str(Path(self.model_base_path) / check_path)

        for role in self.get_all_roles(include_deprecated=True):
            existing_path = self.get_model_path(role)
            if existing_path == check_path:
                return role

        return None


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> ModelRegistry:
    """Convenience function to load the registry."""
    return ModelRegistry(path)


def get_all_roles(include_deprecated: bool = False) -> list[str]:
    """Get all role names from the default registry."""
    return load_registry().get_all_roles(include_deprecated)


def resolve_model_path(role: str) -> Optional[str]:
    """Resolve absolute model path for a role."""
    return load_registry().get_model_path(role)


if __name__ == "__main__":
    # Test the module
    registry = load_registry()

    print("=== Model Registry Test ===\n")

    print("Roles (excluding deprecated):")
    for role in registry.get_all_roles():
        tier = registry.get_tier(role)
        arch = registry.get_architecture(role)
        path = registry.get_model_path(role)
        path_exists = os.path.exists(path) if path else False
        drafts = registry.get_drafts_for_model(role)

        print(f"  [{tier}] {role}")
        print(f"      Architecture: {arch}")
        print(f"      Path exists: {path_exists}")
        if drafts:
            print(f"      Compatible drafts: {drafts}")
        print()
