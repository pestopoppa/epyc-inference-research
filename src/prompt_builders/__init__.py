"""Unified prompt building for the orchestration system.

Package providing all prompt construction functionality:
- Root LM prompts for the orchestrator
- Escalation prompts for tier upgrades
- Step execution prompts for workers
- Role-specific system prompts

Usage:
    from src.prompt_builders import PromptBuilder, RootLMPrompt, EscalationPrompt

    builder = PromptBuilder()
    prompt = builder.build_root_lm_prompt(state="...", task="...", turn=1)
"""

# Types and configuration
from src.prompt_builders.types import (
    EscalationPrompt,
    PromptConfig,
    PromptStyle,
    RootLMPrompt,
    StepPrompt,
)

# Constants and ReAct
from src.prompt_builders.constants import (
    COMPACT_ROOT_LM_TOOLS,
    DEFAULT_ROOT_LM_RULES,
    DEFAULT_ROOT_LM_TOOLS,
    VERBOSE_ROOT_LM_TOOLS,
    REACT_FORMAT,
    REACT_TOOL_WHITELIST,
    VISION_REACT_EXECUTABLE_TOOLS,
    VISION_REACT_TOOL_WHITELIST,
    VISION_TOOL_DESCRIPTIONS,
    build_react_prompt,
)

# Builder class and module-level functions
from src.prompt_builders.builder import (
    PromptBuilder,
    build_escalation_prompt,
    build_long_context_exploration_prompt,
    build_root_lm_prompt,
    build_routing_context,
    build_stage2_review_prompt,
    build_step_prompt,
    build_task_decomposition_prompt,
)

# Review and delegation prompts
from src.prompt_builders.review import (
    build_architect_investigate_prompt,
    build_architect_synthesis_prompt,
    build_plan_review_prompt,
)

# Prompt resolver (hot-swap)
from src.prompt_builders.resolver import resolve_prompt

# Code extraction and error classification
from src.prompt_builders.code_utils import (
    auto_wrap_final,
    classify_error,
    extract_code_from_response,
    rescue_bare_name_final,
    translate_openai_tool_calls,
)

# Format detection and formalizer
from src.prompt_builders.formatting import (
    build_formalizer_prompt,
    detect_format_constraints,
)

__all__ = [
    # Types
    "PromptStyle",
    "PromptConfig",
    "RootLMPrompt",
    "EscalationPrompt",
    "StepPrompt",
    # Constants
    "COMPACT_ROOT_LM_TOOLS",
    "DEFAULT_ROOT_LM_TOOLS",
    "VERBOSE_ROOT_LM_TOOLS",
    "DEFAULT_ROOT_LM_RULES",
    "REACT_TOOL_WHITELIST",
    "VISION_REACT_TOOL_WHITELIST",
    "VISION_REACT_EXECUTABLE_TOOLS",
    "VISION_TOOL_DESCRIPTIONS",
    "REACT_FORMAT",
    "build_react_prompt",
    # Builder
    "PromptBuilder",
    "build_root_lm_prompt",
    "build_escalation_prompt",
    "build_step_prompt",
    "build_stage2_review_prompt",
    "build_routing_context",
    "build_long_context_exploration_prompt",
    "build_task_decomposition_prompt",
    # Review
    "build_plan_review_prompt",
    "build_architect_investigate_prompt",
    "build_architect_synthesis_prompt",
    # Code utils
    "extract_code_from_response",
    "auto_wrap_final",
    "rescue_bare_name_final",
    "classify_error",
    "translate_openai_tool_calls",
    # Formatting
    "detect_format_constraints",
    "build_formalizer_prompt",
    # Resolver
    "resolve_prompt",
]
