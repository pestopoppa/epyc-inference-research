#!/usr/bin/env python3
"""Model Server for hierarchical local-agent orchestration.

This module manages model processes, handles inference requests, and tracks
model health and memory residency.

Supports two backend modes:
1. LlamaCppBackend: Per-inference subprocess (traditional batch mode)
2. LlamaServerBackend: Persistent HTTP server with prefix caching (RadixAttention)

Usage:
    from src.model_server import ModelServer

    # Traditional batch mode
    server = ModelServer()
    server.load("coder_escalation")
    result = server.infer("coder_escalation", prompt="Write a function")
    server.unload("coder_escalation")

    # Server mode with prefix caching
    from src.model_server import ModelServer, create_caching_server
    server = create_caching_server(base_url="http://localhost:8080")
    result = server.infer(InferenceRequest(role="coder", prompt="..."))
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import _registry_timeout
from src.registry_loader import RegistryLoader, RoleConfig

logger = logging.getLogger(__name__)


def _same_real_model_path(left: str, right: str) -> bool:
    return os.path.realpath(left) == os.path.realpath(right)


class ModelState(Enum):
    """State of a model in the server."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ModelStatus:
    """Status of a loaded model."""

    role: str
    state: ModelState
    pid: int | None = None
    loaded_at: float | None = None
    last_inference: float | None = None
    inference_count: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "state": self.state.value,
            "pid": self.pid,
            "loaded_at": self.loaded_at,
            "last_inference": self.last_inference,
            "inference_count": self.inference_count,
            "error_message": self.error_message,
        }


@dataclass
class InferenceRequest:
    """Request for model inference.

    Timeout default comes from model_registry.yaml (runtime_defaults.timeouts.default).
    """

    role: str
    prompt: str | None = None
    prompt_file: Path | None = None
    n_tokens: int = -1
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    context_length: int = 8192
    timeout: int = field(
        default_factory=lambda: int(_registry_timeout("server", "request", 600))
    )
    stop_sequences: list[str] | None = None
    cache_prompt: bool | None = (
        None  # Override cache_prompt for this request (None = use backend default)
    )
    slot_id: int | None = None  # Target slot for prefix cache routing (-1 = auto)
    seed: int | None = None  # RNG seed; None -> backend's fixed determinism seed
    json_schema: dict[str, Any] | None = None  # Constrain output to JSON schema
    grammar: str | None = None  # GBNF grammar for constrained generation
    n_probs: int | None = None  # Optional llama.cpp completion_probabilities top-k capture
    # TD-1d.2: when True, ``n_probs``'s top-k is taken AFTER the full sampler
    # chain (grammar mask included) instead of the default pre-sampling raw
    # logits — see src/typed_decisions/native.py module docstring
    # ("Probability semantics") for why grammar-constrained callers need
    # this. False everywhere except native.py's explicit opt-in.
    post_sampling_probs: bool = False
    max_tokens: int | None = field(default=None, repr=False)
    # HS-4 P0.1: structured OpenAI chat payload (messages, tools, tool_choice) for
    # the /v1 client-executed tool mode. When set, llama-server backends send it
    # to /v1/chat/completions verbatim instead of wrapping prompt as a single
    # user turn. A real field (not a dynamic attr) so dataclasses.replace() in
    # the prefix-cache router preserves it.
    chat_payload: dict[str, Any] | None = field(default=None, repr=False)
    # RI-23: per-call chat_template_kwargs override (``chat_completions_roles.
    # thinking_off()`` -> ``{"enable_thinking": False}``), merged OVER the role's
    # registry kwargs on the /v1/chat/completions lane; inert on /completion. A real
    # field for the same dataclasses.replace() reason as chat_payload.
    chat_template_kwargs: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Sync max_tokens ↔ n_tokens bidirectionally.

        If max_tokens is explicitly provided, it overrides n_tokens.
        Otherwise max_tokens mirrors n_tokens for external callers.
        """
        if self.max_tokens is not None:
            self.n_tokens = self.max_tokens
        else:
            self.max_tokens = self.n_tokens


@dataclass
class InferenceResult:
    """Result from model inference."""

    role: str
    output: str
    tokens_generated: int
    generation_speed: float  # tokens/second
    elapsed_time: float
    success: bool
    error_message: str | None = None
    partial: bool = False
    degraded: bool = False
    failure_stage: str = ""
    failure_reason: str = ""
    # Clean timing data from llama.cpp timings object
    prompt_eval_ms: float = 0.0  # Prompt evaluation time
    generation_ms: float = 0.0  # Token generation time (excludes prompt eval)
    predicted_per_second: float = 0.0  # Clean generation-only t/s from llama.cpp
    http_overhead_ms: float = 0.0  # HTTP round-trip minus inference time (server-side overhead)
    # Speculative decoding acceptance telemetry
    n_tokens_drafted: int = 0
    n_tokens_accepted: int = 0
    acceptance_rate: float = 0.0
    first_token_ms: float = 0.0
    stream_chunks: int = 0
    completion_reason: str = ""
    completion_probabilities: list[dict[str, Any]] = field(default_factory=list)
    # HS-4 P0.1: OpenAI-shape tool calls parsed by llama-server (--jinja) on the
    # chat-completions path. Empty for every other path.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # RTG-47 data plane: the server's OWN terminal prompt-token count
    # (`tokens_evaluated` / `usage.prompt_tokens`). None means the server did
    # not report one (early-stop, transport failure, usage-less stream) — it is
    # never estimated here, so a consumer can tell measured from unknown.
    prompt_tokens: int | None = None
    # HS-4 P0.4: of ``prompt_tokens``, how many the server reused from its KV
    # cache (``usage.prompt_tokens_details.cached_tokens`` / ``timings.cache_n``)
    # on the chat-completions path. None = not reported; never estimated.
    cached_prompt_tokens: int | None = None
    # Set (to ContextOverflowInfo.to_dict()) when llama-server refused or
    # aborted the request for lack of KV context — HTTP 400
    # exceed_context_size_error or "Context size has been exceeded.". The
    # inference layer raises src.exceptions.ContextOverflowError from it
    # instead of a generic failure (src/backends/context_overflow.py).
    context_overflow: dict[str, Any] | None = None
    # RI-23: the server-split reasoning (``message.reasoning_content`` / streamed
    # ``delta.reasoning_content``) on the /v1/chat/completions lane. None = the server
    # returned none (every /completion call, thinking-off chat calls). Deliberately NOT
    # in to_dict(): surfaced through the primitives' inference meta instead.
    reasoning_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "prompt_tokens": self.prompt_tokens,
            "output": self.output,
            "tokens_generated": self.tokens_generated,
            "generation_speed": self.generation_speed,
            "elapsed_time": self.elapsed_time,
            "success": self.success,
            "error_message": self.error_message,
            "partial": self.partial,
            "degraded": self.degraded,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "prompt_eval_ms": self.prompt_eval_ms,
            "generation_ms": self.generation_ms,
            "predicted_per_second": self.predicted_per_second,
            "http_overhead_ms": self.http_overhead_ms,
            "n_tokens_drafted": self.n_tokens_drafted,
            "n_tokens_accepted": self.n_tokens_accepted,
            "acceptance_rate": self.acceptance_rate,
            "first_token_ms": self.first_token_ms,
            "stream_chunks": self.stream_chunks,
            "completion_reason": self.completion_reason,
            "completion_probabilities": self.completion_probabilities,
            "context_overflow": self.context_overflow,
        }


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    @abstractmethod
    def load(self, role_config: RoleConfig) -> int:
        """Load a model and return its process ID."""
        pass

    @abstractmethod
    def unload(self, pid: int) -> bool:
        """Unload a model by its process ID."""
        pass

    @abstractmethod
    def infer(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run inference on a model."""
        pass

    @abstractmethod
    def health_check(self, pid: int) -> bool:
        """Check if a model process is healthy."""
        pass


class LlamaCppBackend(ModelBackend):
    """Backend using llama.cpp for inference.

    Uses the correct binary for each acceleration type:
    - llama-completion: baseline, moe_expert_reduction
    - llama-speculative: speculative_decoding
    - llama-lookup: prompt_lookup
    """

    # Binary selection based on acceleration type
    BINARY_MAP = {
        "none": "llama-completion",
        "baseline": "llama-completion",
        "moe_expert_reduction": "llama-completion",
        "speculative_decoding": "llama-speculative",
        "prompt_lookup": "llama-lookup",
    }

    def __init__(self, registry: RegistryLoader):
        """Initialize the llama.cpp backend.

        Args:
            registry: Loaded model registry for command generation.
        """
        self.registry = registry
        self._processes: dict[int, subprocess.Popen] = {}

    def load(self, role_config: RoleConfig) -> int:
        """Load a model (models are loaded per-inference in llama.cpp).

        In llama.cpp batch mode, models are loaded fresh for each inference.
        This method verifies the model file exists.

        Returns:
            Placeholder PID (0 for per-inference mode).
        """
        # Verify model exists
        model_path = Path(role_config.model.full_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return 0

    def unload(self, pid: int) -> bool:
        """Unload a model.

        Returns:
            True (always succeeds for per-inference mode).
        """
        if pid in self._processes:
            try:
                self._processes[pid].terminate()
                self._processes[pid].wait(timeout=5)
            except Exception as e:
                logger.debug("Failed to terminate pid %d, killing: %s", pid, e)
                self._processes[pid].kill()
            del self._processes[pid]
        return True

    def _build_command(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> str:
        """Build the llama.cpp command with correct binary and flags.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters.

        Returns:
            Shell command string ready for execution.
        """
        from src.config import get_config

        accel_type = role_config.acceleration.type
        binary_name = self.BINARY_MAP.get(accel_type, "llama-completion")
        binary_path = get_config().paths.llama_cpp_bin / binary_name

        # SS-BENCH-GATE-c: a live CPU bench must not be tripped by this spawn.
        # Default affinity (`numactl --interleave=all`) is the incident shape, so
        # when a bench claims cores the spawn is pinned off the claim instead
        # (or refused — the bench's continuity gate invalidates runs on any
        # overlap). Quiet path keeps the legacy prefix byte-identical.
        bench_pin: str | None = None
        try:
            from scripts.server.bench_core_claim import api_enforce_placement

            bench_pin = api_enforce_placement(
                None, label=f"llama-completion ({self.__class__.__name__})"
            )
        except Exception as exc:
            raise RuntimeError(
                "refusing to spawn llama-completion: a CPU bench claims cores and "
                f"the placement guard could not satisfy it ({exc})"
            ) from exc

        # Start with timeout and NUMA wrapper
        parts = [
            f"timeout {request.timeout}",
            "env OMP_NUM_THREADS=1",
            *(
                [f"taskset -c {bench_pin}"]
                if bench_pin is not None
                else (["numactl --interleave=all"] if shutil.which("numactl") else [])
            ),
            str(binary_path),
            f"-m {role_config.model.full_path}",
        ]

        # Add acceleration-specific flags
        if accel_type == "speculative_decoding":
            draft = self.registry.get_draft_for_role(role_config.name)
            if draft and not _same_real_model_path(
                role_config.model.full_path,
                draft.model.full_path,
            ):
                parts.append(f"-md {draft.model.full_path}")
            k = role_config.acceleration.k or 16
            parts.append(f"--draft-max {k}")

        elif accel_type == "moe_expert_reduction":
            override_key = role_config.acceleration.override_key
            experts = role_config.acceleration.experts or 4
            if override_key:
                parts.append(f"--override-kv {override_key}=int:{experts}")

        elif accel_type == "prompt_lookup":
            k = role_config.acceleration.k or 16
            parts.append(f"--draft-max {k}")

        # Add common parameters
        parts.append(f"-t {self.registry._runtime_defaults.get('threads', 96)}")
        parts.append(f"-n {request.n_tokens}")

        # Sampling: explicit request overrides registry intent; omitted request
        # values fall back to role acceleration/generation defaults, then greedy.
        temp = request.temperature
        if temp is None:
            temp = role_config.acceleration.temperature
        if temp is None and role_config.generation_defaults is not None:
            temp = role_config.generation_defaults.temperature
        if temp is None:
            temp = 0.0
        parts.append(f"--temp {temp}")
        if request.top_p is not None:
            parts.append(f"--top-p {request.top_p}")
        if request.top_k is not None:
            parts.append(f"--top-k {request.top_k}")
        if request.seed is not None:
            parts.append(f"--seed {request.seed}")

        # Add prompt
        if request.prompt_file:
            parts.append(f"-f {request.prompt_file}")
        elif request.prompt:
            # Escape for shell - use double quotes and escape internal quotes
            safe_prompt = request.prompt.replace('"', '\\"')
            parts.append(f'-p "{safe_prompt}"')

        return " ".join(parts)

    def infer(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run inference using the appropriate llama.cpp binary.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters.

        Returns:
            InferenceResult with output and metrics.
        """
        start_time = time.time()

        cmd = self._build_command(role_config, request)

        try:
            # Run inference
            import shlex

            result = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                timeout=request.timeout + 10,  # Extra buffer beyond internal timeout
            )

            elapsed = time.time() - start_time
            output = result.stdout
            stderr = result.stderr

            # Parse metrics from output
            speed = self._parse_speed(output + stderr)
            tokens = self._parse_tokens(output + stderr, request.n_tokens)

            # Check for common errors
            error_msg = None
            success = result.returncode == 0

            if not success:
                error_msg = self._extract_error(stderr)

            return InferenceResult(
                role=role_config.name,
                output=output,
                tokens_generated=tokens,
                generation_speed=speed,
                elapsed_time=elapsed,
                success=success,
                error_message=error_msg,
            )

        except subprocess.TimeoutExpired:
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=request.timeout,
                success=False,
                error_message=f"Inference timed out after {request.timeout}s",
            )

        except Exception as e:
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    def health_check(self, pid: int) -> bool:
        """Check if a process is running.

        Args:
            pid: Process ID to check.

        Returns:
            True if process is running or PID is 0 (per-inference mode).
        """
        if pid == 0:
            return True  # Per-inference mode always healthy

        if pid in self._processes:
            return self._processes[pid].poll() is None

        return False

    def _parse_speed(self, output: str) -> float:
        """Parse generation speed from llama.cpp output.

        Handles multiple output formats:
        - llama-completion: "eval time = ... (X.XX tokens per second)"
        - llama-speculative: "decoded X tokens in Y.YYs, Z.ZZ t/s"
        - llama-lookup: "decoded X tokens in Y.YYs, Z.ZZ t/s"
        """
        import re

        # Pattern 1: llama-completion format
        match = re.search(r"(\d+\.\d+)\s*tokens per second", output)
        if match:
            return float(match.group(1))

        # Pattern 2: llama-speculative/lookup format
        match = re.search(r"decoded.*?(\d+\.\d+)\s*t/s", output)
        if match:
            return float(match.group(1))

        # Pattern 3: eval time format
        match = re.search(r"eval time.*?(\d+\.\d+)\s*ms.*?(\d+)\s*tokens", output)
        if match:
            ms = float(match.group(1))
            tokens = int(match.group(2))
            if ms > 0:
                return tokens / (ms / 1000.0)

        return 0.0

    def _parse_tokens(self, output: str, requested: int) -> int:
        """Parse actual token count from output."""
        import re

        # Pattern: "decoded X tokens"
        match = re.search(r"decoded\s+(\d+)\s+tokens", output)
        if match:
            return int(match.group(1))

        # Pattern: "eval: X tokens"
        match = re.search(r"eval.*?(\d+)\s+tokens", output)
        if match:
            return int(match.group(1))

        # Fallback to requested
        return requested

    def _extract_error(self, stderr: str) -> str:
        """Extract meaningful error message from stderr."""
        if not stderr:
            return "Unknown error"

        # Look for common error patterns
        lines = stderr.strip().split("\n")

        # Check for specific errors
        for line in lines:
            if "error:" in line.lower():
                return line.strip()
            if "failed" in line.lower():
                return line.strip()
            if "invalid" in line.lower():
                return line.strip()

        # Return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return "Unknown error"


class ModelServerError(Exception):
    """Error from the model server."""

    pass


@dataclass
class ModelServer:
    """Server managing model lifecycle and inference.

    Attributes:
        registry: Model registry loader.
        backend: Backend for model operations.
        models: Currently tracked model statuses.
    """

    registry: RegistryLoader = field(
        default_factory=lambda: RegistryLoader(allow_missing=True)
    )
    backend: ModelBackend | None = None
    models: dict[str, ModelStatus] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize backend if not provided."""
        if self.backend is None:
            self.backend = LlamaCppBackend(self.registry)

    def load(self, role: str) -> ModelStatus:
        """Load a model for a given role.

        Args:
            role: Registry role name (e.g., "coder_escalation").

        Returns:
            Status of the loaded model.

        Raises:
            ModelServerError: If role not found or loading fails.
        """
        try:
            role_config = self.registry.get_role(role)
        except KeyError as e:
            raise ModelServerError(f"Role not found: {role}") from e

        status = ModelStatus(
            role=role,
            state=ModelState.LOADING,
        )
        self.models[role] = status

        try:
            pid = self.backend.load(role_config)
            status.pid = pid
            status.state = ModelState.READY
            status.loaded_at = time.time()
        except Exception as e:
            status.state = ModelState.ERROR
            status.error_message = str(e)
            raise ModelServerError(f"Failed to load {role}: {e}") from e

        return status

    def unload(self, role: str) -> bool:
        """Unload a model.

        Args:
            role: Role name to unload.

        Returns:
            True if successfully unloaded.
        """
        if role not in self.models:
            return False

        status = self.models[role]
        if status.pid is not None:
            self.backend.unload(status.pid)

        status.state = ModelState.UNLOADED
        del self.models[role]
        return True

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on a model.

        Args:
            request: Inference request.

        Returns:
            Inference result.

        Raises:
            ModelServerError: If role not found.
        """
        try:
            role_config = self.registry.get_role(request.role)
        except KeyError as e:
            raise ModelServerError(f"Role not found: {request.role}") from e

        # Update status
        if request.role in self.models:
            self.models[request.role].state = ModelState.BUSY

        result = self.backend.infer(role_config, request)

        # Update status
        if request.role in self.models:
            status = self.models[request.role]
            status.state = ModelState.READY if (result.success or result.partial) else ModelState.ERROR
            status.last_inference = time.time()
            status.inference_count += 1
            if not result.success and not result.partial:
                status.error_message = result.error_message

        return result

    def get_status(self, role: str) -> ModelStatus | None:
        """Get status of a model.

        Args:
            role: Role name.

        Returns:
            Model status or None if not loaded.
        """
        return self.models.get(role)

    def list_models(self) -> list[ModelStatus]:
        """List all loaded models.

        Returns:
            List of model statuses.
        """
        return list(self.models.values())

    def health_check(self) -> dict[str, Any]:
        """Run health check on all models.

        Returns:
            Health status dictionary.
        """
        results = {}
        for role, status in self.models.items():
            healthy = True
            if status.pid is not None:
                healthy = self.backend.health_check(status.pid)
            results[role] = {
                "healthy": healthy,
                "state": status.state.value,
                "inference_count": status.inference_count,
            }

        return {
            "status": "healthy" if all(r["healthy"] for r in results.values()) else "degraded",
            "models": results,
            "timestamp": time.time(),
        }


def create_caching_server(
    base_url: str = "http://localhost:8080",
    num_slots: int = 4,
    registry_path: str | None = None,
) -> "CachingModelServer":
    """Create a ModelServer with prefix caching enabled.

    This factory function creates a server that uses the LlamaServerBackend
    with RadixAttention-style prefix caching for improved performance.

    Args:
        base_url: URL of the running llama-server instance.
        num_slots: Number of parallel slots on the server.
        registry_path: Path to model registry YAML file.

    Returns:
        CachingModelServer instance ready for inference.

    Example:
        server = create_caching_server()
        result = server.infer(InferenceRequest(role="coder", prompt="..."))
        print(f"Cache hit rate: {server.get_cache_stats()['hit_rate']:.1%}")
    """
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.prefix_cache import CachingBackend, PrefixRouter

    config = ServerConfig(base_url=base_url, num_slots=num_slots)
    backend = LlamaServerBackend(config)
    router = PrefixRouter(num_slots=num_slots)
    caching = CachingBackend(backend, router)

    registry = RegistryLoader(registry_path) if registry_path else RegistryLoader()

    return CachingModelServer(
        registry=registry,
        backend=caching,
        base_backend=backend,
    )


@dataclass
class CachingModelServer:
    """ModelServer variant with prefix caching support.

    Wraps a CachingBackend to provide prefix-aware inference with:
    - Automatic prompt routing to optimal slots
    - Cache hit/miss tracking
    - Hot prefix persistence

    Attributes:
        registry: Model registry loader.
        backend: CachingBackend wrapper.
        base_backend: Underlying LlamaServerBackend.
    """

    registry: RegistryLoader
    backend: "CachingBackend"  # noqa: F821
    base_backend: "LlamaServerBackend"  # noqa: F821

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference with automatic prefix caching.

        Args:
            request: Inference request.

        Returns:
            Inference result.

        Raises:
            ModelServerError: If role not found.
        """
        try:
            role_config = self.registry.get_role(request.role)
        except KeyError as e:
            raise ModelServerError(f"Role not found: {request.role}") from e

        return self.backend.infer(role_config, request)

    def health_check(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Health status dictionary.
        """
        healthy = self.base_backend.health_check(0)
        cache_stats = self.get_cache_stats()

        return {
            "status": "healthy" if healthy else "unhealthy",
            "cache_hit_rate": cache_stats.get("router_hit_rate", 0),
            "timestamp": time.time(),
        }

    def get_cache_stats(self) -> dict[str, float | int]:
        """Get cache performance statistics.

        Returns:
            Dictionary with hit rate, token savings, etc.
        """
        return self.backend.get_stats()



def main() -> int:
    """CLI entry point for testing.

    Usage:
        python -m src.model_server                    # List roles and health check
        python -m src.model_server <role>             # Test inference with role
        python -m src.model_server <role> "<prompt>"  # Inference with custom prompt
        python -m src.model_server --server           # Test with llama-server backend
    """
    import json
    import sys

    # Check for server mode
    if "--server" in sys.argv:
        logger.info("Testing with llama-server backend...")
        try:
            server = create_caching_server()
            health = server.health_check()
            logger.info("Server health: %s", json.dumps(health, indent=2))
            logger.info("Cache stats: %s", json.dumps(server.get_cache_stats(), indent=2))
            return 0
        except Exception as e:
            logger.error("Server test failed: %s", e)
            logger.info("Make sure llama-server is running: scripts/server/start_servers.sh")
            return 1

    server = ModelServer()

    logger.info("Model Server initialized")
    logger.info("Available roles: %s", list(server.registry.roles.keys()))

    # If role specified, run inference test
    if len(sys.argv) > 1:
        role = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a hello world function in Python"

        logger.info("Testing inference with role: %s", role)
        logger.info("Prompt: %s...", prompt[:50])

        try:
            # Load the model (validates path)
            status = server.load(role)
            logger.info("Model loaded: %s", status.state.value)

            # Run inference
            request = InferenceRequest(
                role=role,
                prompt=prompt,
                n_tokens=64,
                temperature=0.0,
                timeout=120,
            )

            logger.info("Running inference...")
            result = server.infer(request)

            logger.info("Success: %s", result.success)
            logger.info("Tokens: %d", result.tokens_generated)
            logger.info("Speed: %.2f t/s", result.generation_speed)
            logger.info("Time: %.2fs", result.elapsed_time)

            if result.error_message:
                logger.error("Error: %s", result.error_message)

            logger.info("--- Output ---\n%s", result.output[:500])

            return 0 if result.success else 1

        except ModelServerError as e:
            logger.error("Model server error: %s", e)
            return 1

    # Default: health check
    health = server.health_check()
    logger.info("Health check: %s", json.dumps(health, indent=2))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
