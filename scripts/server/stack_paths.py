"""Shared path + binary + health-timeout constants for the orchestrator stack.

Extracted from orchestrator_stack.py during the 2026-05-22 Tranche-7 refactor.
Lives below `stack_manifest.py` + `stack_commands.py` in the dependency graph
so those modules can import paths without creating a cycle with
`orchestrator_stack.py` (which itself imports back from the manifest).
"""

from __future__ import annotations

import os
from pathlib import Path

from src.registry.kernel_paths import backend_dir as _kernel_backend_dir


def _get_paths() -> dict[str, Path]:
    """Resolve launcher paths without importing the full config object graph.

    ``stack_paths`` is below ``stack_manifest`` in the launcher dependency graph.
    Calling ``src.config.get_config()`` here initializes ``ServerURLsConfig``,
    whose runtime-facts fallback imports ``stack_manifest`` and re-enters this
    partially initialized module.  Besides emitting a circular-import warning,
    that recursion makes cold-start NUMA inference miss valid runtime facts.

    Keep these defaults and environment keys byte-for-byte aligned with
    ``src.config.models.PathsConfig`` while leaving the low-level path module
    genuinely self-contained.
    """
    llm_root = Path(os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm"))
    project_root = Path(
        os.environ.get(
            "ORCHESTRATOR_PATHS_PROJECT_ROOT",
            str(Path(__file__).resolve().parents[2]),
        )
    )
    return {
        "llm_root": llm_root,
        "project_root": project_root,
        "models_dir": Path(
            os.environ.get("ORCHESTRATOR_PATHS_MODELS_DIR", str(llm_root / "models"))
        ),
        "model_base": Path(
            os.environ.get("ORCHESTRATOR_PATHS_MODEL_BASE", str(llm_root / "models"))
        ),
        # Kernel STORE, not a build path. `production/cpu` is repointed by a
        # kernel promotion; a `llama.cpp/build/bin` literal is not, so a launcher
        # holding one keeps running the retired kernel. `backend_dir` resolves the
        # symlink (same string as the old literal today) and RAISES rather than
        # substituting when the store is dangling.
        "llama_cpp_bin": Path(
            os.environ["ORCHESTRATOR_PATHS_LLAMA_CPP_BIN"]
            if "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN" in os.environ
            else str(_kernel_backend_dir("cpu"))
        ),
        "log_dir": Path(
            os.environ.get("ORCHESTRATOR_PATHS_LOG_DIR", str(project_root / "logs"))
        ),
        "cache_dir": Path(
            os.environ.get("ORCHESTRATOR_PATHS_CACHE_DIR", str(llm_root / "cache"))
        ),
        "tmp_dir": Path(
            os.environ.get("ORCHESTRATOR_PATHS_TMP_DIR", str(llm_root / "tmp"))
        ),
    }


_PATHS = _get_paths()


def _resolve_llama_cpp_binary(name: str, extra_candidates: tuple[Path, ...] = ()) -> Path:
    """Return the first known on-disk path for a llama.cpp helper binary."""
    primary = _PATHS["llama_cpp_bin"] / name
    for candidate in (primary, *extra_candidates):
        if candidate.exists():
            return candidate
    return primary


STATE_FILE = _PATHS["log_dir"] / "orchestrator_state.json"
LLAMA_SERVER = _resolve_llama_cpp_binary("llama-server")
LLAMA_MATH_TOOLS = _resolve_llama_cpp_binary(
    "llama-math-tools",
    (_PATHS["llm_root"] / "llama.cpp/tools/math-tools/build/llama-math-tools",),
)
# v2 binary retained for emergency fallback only. As of the 2026-06-26 v6+iqk cutover,
# all hot-tier roles use the v6 binary (production-consolidated-v6 — one kernel, incorporating
# ik_llama's iqk AVX-512 GEMM, GGML_IQK-gated; ik_llama deprecated). Previously
# coder_escalation needed v2 due to a Qwen2.5 spec-decode bug, but
# coder_escalation now runs Qwen3.6-35B-A3B Q8 (same model as frontdoor) which
# is v6-compatible.
LLAMA_SERVER_V2 = _PATHS["llama_cpp_bin"].parent / "build-v2" / "bin" / "llama-server"
_V2_ROLES: frozenset[str] = frozenset()  # was {"coder_escalation"}; empty since 2026-05-06
LOG_DIR = _PATHS["log_dir"]
# DS-3: KV state save/restore directory for dynamic stack management
SLOT_SAVE_DIR = _PATHS["cache_dir"] / "kv_slots"


# Importing ``src.config`` initializes ServerURLsConfig. Keep this below every
# path/binary attribute consumed by runtime_facts_manifest so a config bootstrap
# that imports runtime facts sees a complete stack_paths module, not a partially
# initialized one.
from src.config import _registry_timeout  # noqa: E402


# Health check timeouts from registry (single source of truth)
_HEALTH_SERVER_STARTUP = int(_registry_timeout("health", "server_startup", 120))
_HEALTH_VISION_SERVER = int(_registry_timeout("health", "vision_server", 120))
_HEALTH_WORKER_SERVER = int(_registry_timeout("health", "worker_server", 90))
