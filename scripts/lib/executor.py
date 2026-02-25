#!/usr/bin/env python3
from __future__ import annotations

"""
Command Executor for LLM Inference

Builds and executes llama.cpp commands for:
- Baseline inference (llama-completion)
- Speculative decoding (llama-speculative)
- MoE expert reduction (llama-completion with override)
- Prompt lookup (llama-lookup)

This module is shared with the orchestrator project.
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

try:
    from .registry import ModelRegistry, load_registry
except ImportError:
    from registry import ModelRegistry, load_registry


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


# Default inference parameters (fallbacks if registry unavailable)
DEFAULT_THREADS = 96
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TIMEOUT = _read_registry_timeout("scripts", "executor_default", 180)

# Skip prompt lookup for models larger than this (GB)
# Lookup doesn't benefit large models - they timeout waiting for verification
# and spec decode with draft models is always faster for 32B+ models
# Threshold: 20GB covers ~32B Q4_K_M and above
LOOKUP_MAX_MODEL_SIZE_GB = 20


def get_binary_paths(registry: Optional["ModelRegistry"] = None) -> dict[str, str]:
    """Get binary paths from registry (single source of truth).

    Falls back to hardcoded paths only if registry is unavailable.
    """
    fallback = {
        "base_dir": "/mnt/raid0/llm/llama.cpp/build/bin",
        "completion": "llama-completion",
        "speculative": "llama-speculative",
        "lookup": "llama-lookup",
        "cli": "llama-cli",
        "server": "llama-server",
    }

    if registry is None:
        try:
            registry = load_registry()
        except Exception as e:
            pass

    if registry and hasattr(registry, "data"):
        binaries = registry.data.get("runtime_defaults", {}).get("binaries", {})
        if binaries:
            return binaries

    return fallback


def get_binary(name: str, registry: Optional["ModelRegistry"] = None) -> str:
    """Get full path to a specific binary.

    Args:
        name: Binary name ('completion', 'speculative', 'lookup', 'cli')
        registry: Optional registry instance

    Returns:
        Full absolute path to the binary
    """
    paths = get_binary_paths(registry)
    base_dir = paths.get("base_dir", "/mnt/raid0/llm/llama.cpp/build/bin")
    binary_name = paths.get(name, name)
    return os.path.join(base_dir, binary_name)


def validate_binaries(registry: Optional["ModelRegistry"] = None) -> dict[str, str]:
    """Validate all required binaries exist.

    Raises:
        FileNotFoundError: If any binary is missing, with clear error message.

    Returns:
        Dict mapping binary name to full path (for logging/debugging).
    """
    required = ["completion", "speculative", "lookup"]
    paths = {}
    missing = []

    for name in required:
        path = get_binary(name, registry)
        paths[name] = path
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")

    if missing:
        raise FileNotFoundError(
            f"Missing llama.cpp binaries (check registry runtime_defaults.binaries):\n"
            + "\n".join(missing)
            + f"\n\nRegistry location: /mnt/raid0/llm/claude/orchestration/model_registry.yaml"
        )

    return paths


def get_server_defaults(registry: Optional["ModelRegistry"] = None) -> dict:
    """Get server defaults from registry.

    Returns dict with: port, context_length, startup_timeout, request_timeout, parallel_slots
    """
    defaults = {
        "port": 8080,
        "context_length": 131072,  # 131K - Qwen3 native limit
        "startup_timeout": 600,
        "request_timeout": 300,
        "parallel_slots": 4,
    }

    if registry is None:
        try:
            registry = load_registry()
        except Exception as e:
            pass

    if registry and hasattr(registry, "data"):
        server_cfg = registry.data.get("runtime_defaults", {}).get("server_defaults", {})
        if server_cfg:
            defaults.update(server_cfg)

    return defaults


class ServerManager:
    """Manages llama-server lifecycle for persistent model loading.

    Instead of spawning a new process per inference (which reloads the model each time),
    this keeps a server running with the model in RAM and sends HTTP requests.
    """

    def __init__(self, port: int = None, threads: int = DEFAULT_THREADS, registry: Optional["ModelRegistry"] = None):
        server_defaults = get_server_defaults(registry)
        self.port = port if port is not None else server_defaults["port"]
        self.threads = threads
        self.context_length = server_defaults["context_length"]
        self.startup_timeout = server_defaults["startup_timeout"]
        self.request_timeout = server_defaults["request_timeout"]
        self.process: Optional[subprocess.Popen] = None
        self.model_path: Optional[str] = None
        self.draft_model_path: Optional[str] = None
        self.mmproj_path: Optional[str] = None
        # Reusable HTTP session for connection pooling (health checks, inference)
        self._http_session: Optional[requests.Session] = None

    def _get_http_session(self) -> requests.Session:
        """Get or create a reusable HTTP session for connection pooling."""
        if self._http_session is None:
            self._http_session = requests.Session()
        return self._http_session

    def start(
        self,
        model_path: str,
        moe_override: Optional[str] = None,
        registry: Optional["ModelRegistry"] = None,
        no_mmap: bool = False,
        context_length: Optional[int] = None,
        role: Optional[str] = None,
        draft_model_path: Optional[str] = None,
        draft_max: Optional[int] = None,
        mmproj_path: Optional[str] = None,
    ) -> None:
        """Start llama-server with model loaded.

        Args:
            model_path: Path to the GGUF model file.
            moe_override: Optional MoE expert override (e.g., "qwen3moe.expert_used_count=int:4").
            registry: Optional registry for binary path lookup.
            no_mmap: If True, use bulk read instead of mmap (may be faster for cold loads).
            context_length: Override context length. If None, uses model's max_context from registry.
            role: Optional role name to look up model-specific max_context.
            draft_model_path: Optional path to draft model for speculative decoding.
            draft_max: Optional default K value for speculation (can be overridden per-request).
            mmproj_path: Optional path to multimodal projector for VL models.
        """
        if self.process is not None:
            self.stop()

        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.mmproj_path = mmproj_path
        binary = get_binary("server", registry)

        # Determine context length: explicit > role-based > default
        if context_length is not None:
            ctx_len = context_length
        elif role is not None and registry is not None:
            ctx_len = registry.get_max_context(role)
        else:
            ctx_len = self.context_length

        # Get optimization settings from registry (with fallbacks)
        use_flash_attn = registry.get_flash_attention(role) if role and registry else True
        ubatch_size = registry.get_ubatch_size(role) if role and registry else 8192

        cmd = [
            "numactl", "--interleave=all",
            binary,
            "-m", model_path,
            "-t", str(self.threads),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(ctx_len),
            "--parallel", "1",  # Single slot for full context (benchmarking)
            "-ub", str(ubatch_size),  # Larger batch size for faster prompt processing
        ]
        if use_flash_attn:
            cmd.extend(["-fa", "on"])  # Flash attention for faster long-context processing
        if moe_override:
            cmd.extend(["--override-kv", moe_override])
        if no_mmap:
            cmd.append("--no-mmap")
        if draft_model_path:
            cmd.extend(["-md", draft_model_path])
            if draft_max:
                cmd.extend(["--draft-max", str(draft_max)])
        if mmproj_path:
            cmd.extend(["--mmproj", mmproj_path])

        # Start server in background
        # Capture stderr to temp file for debugging if server fails
        import tempfile
        self._stderr_file = tempfile.NamedTemporaryFile(mode='w', prefix='llama_server_', suffix='.log', delete=False)

        # Debug: print server command (check for MoE override and draft)
        if moe_override:
            print(f"      [DEBUG] Server cmd includes: --override-kv {moe_override}", flush=True)
        if draft_model_path:
            print(f"      [DEBUG] Server cmd includes: -md {draft_model_path}", flush=True)
        if mmproj_path:
            print(f"      [DEBUG] Server cmd includes: --mmproj {mmproj_path}", flush=True)
        print(f"      [DEBUG] Server log: {self._stderr_file.name}", flush=True)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )

    def wait_ready(self, timeout: int = None) -> bool:
        """Wait for server to be ready by polling /health endpoint.

        Args:
            timeout: Maximum seconds to wait. Defaults to startup_timeout from registry.

        Returns:
            True if server is ready, False if timeout or error.
        """
        if timeout is None:
            timeout = self.startup_timeout
        url = f"http://127.0.0.1:{self.port}/health"
        start_time = time.time()

        session = self._get_http_session()
        while time.time() - start_time < timeout:
            try:
                response = session.get(url, timeout=2)
                if response.status_code == 200:
                    # Verify expert count from server log
                    if hasattr(self, '_stderr_file') and self._stderr_file:
                        self._stderr_file.flush()
                        try:
                            with open(self._stderr_file.name, 'r') as f:
                                for line in f:
                                    if 'n_expert_used' in line:
                                        print(f"      [DEBUG] {line.strip()}", flush=True)
                                        break
                        except Exception as e:
                            pass
                    return True
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self.process and self.process.poll() is not None:
                # Print last 20 lines of stderr for debugging
                if hasattr(self, '_stderr_file') and self._stderr_file:
                    self._stderr_file.flush()
                    try:
                        with open(self._stderr_file.name, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                print(f"    [SERVER] Process died. Last 20 lines of stderr:", flush=True)
                                for line in lines[-20:]:
                                    print(f"      {line.rstrip()}", flush=True)
                    except Exception as e:
                        pass
                return False

            time.sleep(1)

        return False

    def is_running(self) -> bool:
        """Check if server process is still running."""
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        """Stop the server process and close HTTP session."""
        if self._http_session is not None:
            self._http_session.close()
            self._http_session = None
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            self.model_path = None
            self.mmproj_path = None

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        speculative_n_max: Optional[int] = None,
        image_path: Optional[str] = None,
    ) -> "InferenceResult":
        """Run inference via HTTP API with streaming to capture partial output.

        Args:
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
            speculative_n_max: Optional K value for speculative decoding (overrides startup default).
            image_path: Optional path to image file for VL models.

        Returns:
            InferenceResult with response content and timing.
            On timeout, returns partial output collected so far.
        """
        # VL mode: use /v1/chat/completions with OpenAI multimodal format
        if self.mmproj_path is not None:
            return self._run_vl_inference(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                image_path=image_path,
            )

        url = f"http://127.0.0.1:{self.port}/completion"
        collected_chunks: list[str] = []
        timed_out = False
        timings = {}

        # Build request payload
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "cache_prompt": False,  # Fresh context for each question
            "stream": True,  # Enable streaming for incremental collection
        }
        if speculative_n_max is not None:
            payload["speculative.n_max"] = speculative_n_max

        try:
            # Use streaming to collect tokens incrementally
            session = self._get_http_session()
            response = session.post(
                url,
                json=payload,
                timeout=(30, timeout),  # (connect timeout, read timeout)
                stream=True,
            )

            if response.status_code != 200:
                return InferenceResult(
                    raw_output=f"HTTP {response.status_code}: {response.text}",
                    exit_code=1,
                    command=f"POST {url}",
                )

            # Parse SSE stream
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Skip "data: " prefix
                        if "content" in data:
                            collected_chunks.append(data["content"])
                        # Final message contains timings
                        if data.get("stop", False):
                            timings = data.get("timings", {})
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.Timeout:
            timed_out = True
        except requests.exceptions.RequestException as e:
            if not collected_chunks:
                return InferenceResult(
                    raw_output=str(e),
                    exit_code=1,
                    command=f"POST {url}",
                )
            # If we have partial content, return it despite the error
            timed_out = True

        collected_content = "".join(collected_chunks)

        # Build output in format compatible with output_parser
        tokens_per_second = timings.get("predicted_per_second", 0)
        prompt_tokens = timings.get("prompt_n", 0)
        completion_tokens = timings.get("predicted_n", len(collected_content.split()))

        raw_output = f"{collected_content}\n\n"
        if timings:
            raw_output += f"llama_perf_context: prompt eval: {prompt_tokens} tokens\n"
            raw_output += f"llama_perf_context: eval time = {timings.get('predicted_ms', 0):.2f}ms / {completion_tokens} tokens ({tokens_per_second:.2f} tokens per second)\n"
        elif timed_out:
            raw_output += f"[PARTIAL OUTPUT - timed out after collecting {len(collected_content)} chars]\n"

        return InferenceResult(
            raw_output=raw_output,
            exit_code=0 if not timed_out else -1,
            command=f"POST {url}",
            tokens_per_second=tokens_per_second if tokens_per_second else None,
            timed_out=timed_out,
        )

    def _run_vl_inference(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        image_path: Optional[str] = None,
    ) -> "InferenceResult":
        """Run VL inference via /v1/chat/completions with multimodal payload.

        When the server was started with --mmproj, use the OpenAI-compatible
        endpoint which accepts base64-encoded images.
        """
        import base64 as b64mod

        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"

        # Build message content
        content_parts = [{"type": "text", "text": prompt}]

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                img_data = b64mod.b64encode(img_file.read()).decode("utf-8")

            ext = os.path.splitext(image_path)[1].lower()
            mime_map = {
                ".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(ext, "image/png")

            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{img_data}"}
            })

        payload = {
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.time()
        try:
            session = self._get_http_session()
            response = session.post(
                url,
                json=payload,
                timeout=(30, timeout),
            )

            elapsed_s = time.time() - start_time

            if response.status_code != 200:
                return InferenceResult(
                    raw_output=f"HTTP {response.status_code}: {response.text}",
                    exit_code=1,
                    command=f"POST {url}",
                )

            data = response.json()

            # Extract response content
            content = ""
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0].get("message", {})
                content = msg.get("content", "")

            # Extract timing — llama-server may include timings in response
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            timings = data.get("timings", {})
            tokens_per_second = timings.get("predicted_per_second", 0)
            predicted_ms = timings.get("predicted_ms", 0)

            # Client-side TPS fallback if server doesn't include timings
            if not tokens_per_second and completion_tokens > 0 and elapsed_s > 0:
                tokens_per_second = completion_tokens / elapsed_s

            # Build output compatible with output_parser
            raw_output = f"{content}\n\n"
            if tokens_per_second:
                raw_output += f"llama_perf_context: prompt eval: {prompt_tokens} tokens\n"
                raw_output += f"llama_perf_context: eval time = {predicted_ms:.2f}ms / {completion_tokens} tokens ({tokens_per_second:.2f} tokens per second)\n"

            return InferenceResult(
                raw_output=raw_output,
                exit_code=0,
                command=f"POST {url}",
                tokens_per_second=tokens_per_second if tokens_per_second else None,
            )

        except requests.exceptions.Timeout:
            return InferenceResult(
                raw_output="[VL inference timed out]",
                exit_code=-1,
                command=f"POST {url}",
                timed_out=True,
            )
        except requests.exceptions.RequestException as e:
            return InferenceResult(
                raw_output=str(e),
                exit_code=1,
                command=f"POST {url}",
            )


@dataclass
class InferenceResult:
    """Result of an inference run."""

    raw_output: str  # stdout only - actual model output
    exit_code: int
    command: str
    timed_out: bool = False
    tokens_per_second: Optional[float] = None  # Direct from server response
    stderr: str = ""  # stderr only - for debugging errors

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class Config:
    """Benchmark configuration."""

    name: str  # e.g., 'baseline', 'spec_k8', 'moe4', 'lookup_n3'
    config_type: str  # 'baseline', 'spec', 'moe', 'lookup'

    # Speculative decoding
    spec_k: Optional[int] = None
    draft_model_path: Optional[str] = None

    # MoE expert reduction
    moe_experts: Optional[int] = None
    moe_override_key: Optional[str] = None

    # Prompt lookup
    lookup_ngram: Optional[int] = None

    # Speed-test optimization: if True, only measure speed (quality inherited from baseline)
    speed_test_only: bool = False
    inherits_quality_from: Optional[str] = None  # Config name to copy scores from

    @classmethod
    def baseline(cls) -> "Config":
        return cls(name="baseline", config_type="baseline")

    @classmethod
    def spec(cls, k: int, draft_path: str, draft_name: str = "") -> "Config":
        # Include draft model name in config name to distinguish between drafts
        if draft_name:
            name = f"spec_{draft_name}_k{k}"
        else:
            # Fallback: extract name from path
            draft_stem = Path(draft_path).stem if draft_path else "unknown"
            name = f"spec_{draft_stem}_k{k}"
        return cls(
            name=name,
            config_type="spec",
            spec_k=k,
            draft_model_path=draft_path,
        )

    @classmethod
    def moe(cls, experts: int, override_key: str) -> "Config":
        return cls(
            name=f"moe{experts}",
            config_type="moe",
            moe_experts=experts,
            moe_override_key=override_key,
        )

    @classmethod
    def lookup(cls, ngram: int) -> "Config":
        return cls(
            name=f"lookup_n{ngram}",
            config_type="lookup",
            lookup_ngram=ngram,
        )

    @classmethod
    def compound_moe_lookup(cls, experts: int, override_key: str, ngram: int) -> "Config":
        return cls(
            name=f"moe{experts}_lookup_n{ngram}",
            config_type="moe_lookup",
            moe_experts=experts,
            moe_override_key=override_key,
            lookup_ngram=ngram,
        )

    @classmethod
    def compound_moe_spec(
        cls, experts: int, override_key: str, k: int, draft_path: str, draft_name: str = ""
    ) -> "Config":
        """Create compound config: MoE expert reduction + speculative decoding."""
        if draft_name:
            name = f"moe{experts}_spec_{draft_name}_k{k}"
        else:
            draft_stem = Path(draft_path).stem if draft_path else "unknown"
            name = f"moe{experts}_spec_{draft_stem}_k{k}"
        return cls(
            name=name,
            config_type="moe_spec",
            moe_experts=experts,
            moe_override_key=override_key,
            spec_k=k,
            draft_model_path=draft_path,
        )

class Executor:
    """Executes llama.cpp inference commands."""

    def __init__(self, registry: Optional[ModelRegistry] = None, validate: bool = True):
        self.registry = registry or load_registry()
        # Validate binaries exist on startup - fail fast with clear error
        if validate:
            self._binary_paths = validate_binaries(self.registry)
        else:
            self._binary_paths = None

    def get_configs_for_architecture(
        self,
        architecture: str,
        role: str,
        registry: Optional[ModelRegistry] = None,
    ) -> list[Config]:
        """Get applicable configs for a model architecture.

        Args:
            architecture: Model architecture ('dense', 'moe', 'qwen3moe', 'ssm_moe_hybrid', etc.)
            role: The model role for checking constraints and drafts.
            registry: Optional registry instance.

        Returns:
            List of Config objects to test.
        """
        reg = registry or self.registry
        configs = [Config.baseline()]

        # Get forbidden optimizations
        forbidden = reg.get_forbidden_configs(role)

        # Architecture-specific configs
        # Test expert counts from 4 up to (baseline - 2), stepping by 2
        # E.g., baseline=8 → test [4, 6], baseline=6 → test [4]
        # NOTE: 2 experts often causes SIGSEGV or garbage output (see QUIRKS.md)
        MIN_SAFE_EXPERTS = 4
        if architecture in ("moe", "qwen3moe", "qwen3vlmoe", "mixtral", "deepseek2"):
            override_key = reg.get_moe_override_key(role) or "qwen3moe.expert_used_count"
            baseline_experts = reg.get_baseline_experts(role)
            max_test_experts = baseline_experts - 2  # Don't test baseline or baseline-1
            for experts in range(MIN_SAFE_EXPERTS, max_test_experts + 1, 2):  # [4, 6, ...] up to max
                configs.append(Config.moe(experts, override_key))

            # MoE + lookup compound config (speed-only: inherits quality from moe4)
            # Skip for large models - lookup times out and spec decode is always faster
            model_path = reg.get_model_path(role)
            role_config = reg.get_role_config(role)
            size_gb = role_config.get("model", {}).get("size_gb", 0) if role_config else 0
            if size_gb == 0 and model_path and os.path.exists(model_path):
                size_gb = os.path.getsize(model_path) / (1024**3)

            if "prompt_lookup" not in forbidden and size_gb < LOOKUP_MAX_MODEL_SIZE_GB:
                cfg = Config.compound_moe_lookup(4, override_key, 4)
                cfg.speed_test_only = True
                cfg.inherits_quality_from = "moe4"  # Quality from MoE, not baseline
                configs.append(cfg)

            # MoE + spec decode compound configs (if compatible drafts exist)
            # Note: Tested and found to be SLOWER than MoE alone (0.84x baseline)
            # but kept for completeness. May want to disable to save benchmark time.
            if "speculative_decoding" not in forbidden:
                drafts = reg.get_drafts_for_model(role)
                for draft_role in drafts:
                    draft_path = reg.get_model_path(draft_role)
                    if draft_path and os.path.exists(draft_path):
                        # Test MoE + spec decode at optimal expert count (usually 4)
                        for k in [8, 16, 24]:
                            cfg = Config.compound_moe_spec(
                                MIN_SAFE_EXPERTS, override_key, k, draft_path, draft_role
                            )
                            cfg.speed_test_only = True
                            cfg.inherits_quality_from = "baseline"
                            configs.append(cfg)

        elif architecture in ("ssm_moe_hybrid", "qwen3next"):
            # SSM models - MoE reduction ONLY, no speculation (SSM incompatible with all spec methods)
            override_key = reg.get_moe_override_key(role) or "qwen3moe.expert_used_count"
            baseline_experts = reg.get_baseline_experts(role)
            max_test_experts = baseline_experts - 2
            for experts in range(MIN_SAFE_EXPERTS, max_test_experts + 1, 2):
                configs.append(Config.moe(experts, override_key))

        else:
            # Dense models - try speculative decoding with ALL compatible drafts
            # NOTE: Spec decode configs are speed_test_only because quality is identical
            # to baseline (same target model). Only speed differs.
            if "speculative_decoding" not in forbidden:
                drafts = reg.get_drafts_for_model(role)
                for draft_role in drafts:
                    draft_path = reg.get_model_path(draft_role)
                    if draft_path and os.path.exists(draft_path):
                        for k in [4, 8, 16, 24, 32, 48]:
                            cfg = Config.spec(k, draft_path, draft_role)
                            cfg.speed_test_only = True
                            cfg.inherits_quality_from = "baseline"
                            configs.append(cfg)
                        # Test ALL compatible drafts, not just the first one

            # Skip lookup for draft models (Tier D) - they're already fast
            # and lookup requires substantial prompts to be effective
            role_config = reg.get_role_config(role)
            is_draft = role_config and role_config.get("tier") == "D"

            # Get model size for lookup threshold check
            model_path = reg.get_model_path(role)
            size_gb = role_config.get("model", {}).get("size_gb", 0) if role_config else 0
            if size_gb == 0 and model_path and os.path.exists(model_path):
                size_gb = os.path.getsize(model_path) / (1024**3)

            # Prompt lookup configs (speed-only: inherits quality from baseline)
            # Same target model, just different draft strategy
            # Skip for large models (>40GB) - lookup times out, spec decode is faster
            if "prompt_lookup" not in forbidden and not is_draft and size_gb < LOOKUP_MAX_MODEL_SIZE_GB:
                for ngram in [3, 4, 5]:
                    cfg = Config.lookup(ngram)
                    cfg.speed_test_only = True
                    cfg.inherits_quality_from = "baseline"
                    configs.append(cfg)

        return configs

    def build_command(
        self,
        model_path: str,
        config: Config,
        prompt_file: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        threads: int = DEFAULT_THREADS,
        mmproj_path: Optional[str] = None,
        image_path: Optional[str] = None,
        context_size: Optional[int] = None,
    ) -> list[str]:
        """Build the llama.cpp command for a configuration.

        Args:
            model_path: Path to the target model.
            config: The benchmark configuration.
            prompt_file: Path to file containing the prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            threads: Number of threads.
            mmproj_path: Path to mmproj file for VL models.
            image_path: Path to image file for VL models.
            context_size: Context size in tokens (default: llama.cpp default ~8K).

        Returns:
            Command as list of strings.
        """
        # Check if this is a vision model (determined by mmproj_path presence)
        is_vision = mmproj_path is not None

        if is_vision:
            # Vision models use llama-mtmd-cli with different invocation
            binary = get_binary("mtmd", self.registry)
            cmd = [
                "numactl", "--interleave=all",
                binary,
                "-m", model_path,
                "--mmproj", mmproj_path,
                "-t", str(threads),
                "-n", str(max_tokens),
                "--temp", str(temperature),
            ]
            # Add context size if specified (required for long prompts > 8K default)
            if context_size is not None:
                cmd.extend(["-c", str(context_size)])
            # Add image if provided
            if image_path:
                cmd.extend(["--image", image_path])
            # mtmd-cli uses -f for prompt file (same as llama-cli)
            cmd.extend(["-f", prompt_file])
            # Add MoE override if applicable (VL MoE models like Qwen3-VL-30B)
            if config.config_type == "moe":
                cmd.extend([
                    "--override-kv", f"{config.moe_override_key}=int:{config.moe_experts}",
                ])
            return cmd

        # Non-vision models: select binary based on config type
        if config.config_type == "spec" or config.config_type == "moe_spec":
            binary = get_binary("speculative", self.registry)
        elif config.config_type == "lookup" or config.config_type == "moe_lookup":
            binary = get_binary("lookup", self.registry)
        else:
            binary = get_binary("completion", self.registry)

        completion_binary = get_binary("completion", self.registry)

        # Base command with env wrapper
        cmd = [
            "numactl", "--interleave=all",
            binary,
            "-m", model_path,
            "-t", str(threads),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-f", prompt_file,
        ]

        # Add context size if specified (required for long prompts > 8K default)
        # Also set batch size to match - lookup/lookahead need batch >= prompt tokens
        # Note: context_tokens from YAML is an estimate (chars/4). Different tokenizers
        # may produce 2-3x more tokens, so we apply a 3x safety multiplier.
        if context_size is not None:
            safe_context = context_size * 3  # Safety margin for tokenizer differences
            cmd.extend(["-c", str(safe_context)])
            cmd.extend(["-b", str(safe_context)])

        # --no-conversation only works with llama-completion (prevents interactive hangs)
        if binary == completion_binary:
            cmd.append("--no-conversation")

        # Add config-specific flags
        if config.config_type == "spec":
            cmd.extend([
                "-md", config.draft_model_path,
                "--draft-max", str(config.spec_k),
            ])
        elif config.config_type == "moe":
            cmd.extend([
                "--override-kv", f"{config.moe_override_key}=int:{config.moe_experts}",
            ])
        elif config.config_type == "lookup":
            cmd.extend([
                "--draft-max", str(config.lookup_ngram or 16),
            ])
        elif config.config_type == "moe_spec":
            cmd.extend([
                "-md", config.draft_model_path,
                "--draft-max", str(config.spec_k),
                "--override-kv", f"{config.moe_override_key}=int:{config.moe_experts}",
            ])
        elif config.config_type == "moe_lookup":
            cmd.extend([
                "--draft-max", str(config.lookup_ngram or 16),
                "--override-kv", f"{config.moe_override_key}=int:{config.moe_experts}",
            ])

        return cmd

    def run_inference(
        self,
        model_path: str,
        config: Config,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        threads: int = DEFAULT_THREADS,
        timeout: int = DEFAULT_TIMEOUT,
        mmproj_path: Optional[str] = None,
        image_path: Optional[str] = None,
        context_size: Optional[int] = None,
        role: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference with the given configuration.

        Args:
            model_path: Path to the target model.
            config: The benchmark configuration.
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            threads: Number of threads.
            timeout: Timeout in seconds.
            role: Role name for looking up paged_attention config.
            mmproj_path: Path to mmproj file for VL models.
            image_path: Path to image file for VL models.
            context_size: Context size in tokens (for long prompts).

        Returns:
            InferenceResult with output and status.
        """
        # Write prompt to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="/mnt/raid0/llm/tmp"
        ) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            cmd = self.build_command(
                model_path, config, prompt_file, max_tokens, temperature, threads,
                mmproj_path=mmproj_path, image_path=image_path, context_size=context_size
            )
            cmd_str = " ".join(cmd)

            # Build environment with paged attention if recommended for this role
            env = None
            if role and self.registry:
                role_config = self.registry.get_role_config(role)
                if role_config:
                    paged_attn = role_config.get('paged_attention', {})
                    if paged_attn.get('recommended'):
                        env = os.environ.copy()
                        block_size = paged_attn.get('block_size', 64)
                        env['LLAMA_PAGED_ATTN'] = str(block_size)
                        # Log that we're using paged attention
                        print(f"      [PAGED_ATTN] Enabling block_size={block_size} for {role}", flush=True)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                # Separate stdout (model output) from stderr (llama.cpp logs/errors)
                # Timing info is in stderr, so append it to raw_output for parsing
                # Include patterns for: standard llama-cli AND llama-speculative output
                timing_lines = [
                    line for line in result.stderr.split('\n')
                    if any(x in line for x in [
                        'eval time', 'tokens per second', 'acceptance',
                        'n_predict', 'target:', 'draft:', 'total time', 'decoded'
                    ])
                ]
                raw_output = result.stdout
                if timing_lines:
                    raw_output += '\n' + '\n'.join(timing_lines)
                return InferenceResult(
                    raw_output=raw_output,
                    exit_code=result.returncode,
                    command=cmd_str,
                    stderr=result.stderr,
                )
            except subprocess.TimeoutExpired as e:
                # Capture partial output if available
                # Note: TimeoutExpired.stdout/stderr are bytes even with text=True
                partial_stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
                partial_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
                return InferenceResult(
                    raw_output=partial_stdout,
                    exit_code=-1,
                    command=cmd_str,
                    timed_out=True,
                    stderr=partial_stderr,
                )

        finally:
            # Clean up temp file
            if os.path.exists(prompt_file):
                os.unlink(prompt_file)


def build_command(
    model_path: str,
    config: Config,
    prompt_file: str,
    registry: Optional[ModelRegistry] = None,
    mmproj_path: Optional[str] = None,
    image_path: Optional[str] = None,
    context_size: Optional[int] = None,
) -> list[str]:
    """Convenience function to build a command."""
    executor = Executor(registry)
    return executor.build_command(
        model_path, config, prompt_file,
        mmproj_path=mmproj_path, image_path=image_path, context_size=context_size
    )


def run_inference(
    model_path: str,
    config: Config,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    registry: Optional[ModelRegistry] = None,
    mmproj_path: Optional[str] = None,
    image_path: Optional[str] = None,
    context_size: Optional[int] = None,
) -> InferenceResult:
    """Convenience function to run inference."""
    executor = Executor(registry)
    return executor.run_inference(
        model_path, config, prompt, timeout=timeout,
        mmproj_path=mmproj_path, image_path=image_path, context_size=context_size
    )


if __name__ == "__main__":
    # Test the module
    import sys

    print("=== Executor Test ===\n")

    registry = load_registry()
    executor = Executor(registry)

    # Test config generation for different architectures
    test_roles = ["coder_escalation", "worker_math", "ingest_long_context"]

    for role in test_roles:
        arch = registry.get_architecture(role)
        path = registry.get_model_path(role)
        print(f"Role: {role}")
        print(f"  Architecture: {arch}")
        print(f"  Path: {path}")

        configs = executor.get_configs_for_architecture(arch, role)
        print(f"  Configs ({len(configs)}):")
        for cfg in configs:
            print(f"    - {cfg.name}")
        print()

    # Test command building (dry run)
    print("Sample command for baseline:")
    cfg = Config.baseline()
    cmd = executor.build_command(
        "/mnt/raid0/llm/models/test.gguf",
        cfg,
        "/tmp/prompt.txt",
    )
    print(f"  {' '.join(cmd)}")
