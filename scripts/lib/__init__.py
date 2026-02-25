"""
Shared library modules for benchmark and orchestrator systems.

Provides model registry parsing, inference command execution, output parsing,
quality scoring, and temperature optimization. These modules are shared between
the benchmark scripts and the orchestration system.

Modules:
    registry: Model registry YAML parser and path resolver
    executor: llama.cpp command builder and runner
    output_parser: Inference output parser (tokens/s, acceptance rate)
    scorer: Pattern-based quality scoring (0-3 scale)
    temperature_optimizer: Binary search for optimal temperature
    onboard: 7-step model onboarding flow
"""

from __future__ import annotations

from .registry import ModelRegistry, load_registry, get_all_roles, resolve_model_path
from .executor import Executor, Config, InferenceResult, run_inference, build_command

__all__ = [
    "ModelRegistry",
    "load_registry",
    "get_all_roles",
    "resolve_model_path",
    "Executor",
    "Config",
    "InferenceResult",
    "run_inference",
    "build_command",
]
