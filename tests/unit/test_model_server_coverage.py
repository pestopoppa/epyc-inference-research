"""Additional coverage tests for model_server module.

Focuses on untested paths:
- LlamaCppBackend.infer() subprocess execution
- LlamaCppBackend.health_check() with processes
- LlamaCppBackend.load()/unload() process management
- create_caching_server() factory
- CachingModelServer methods
- main() CLI entry point
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.model_server import (
    CachingModelServer,
    InferenceRequest,
    InferenceResult,
    LlamaCppBackend,
    ModelServer,
    ModelServerError,
    ModelState,
    ModelStatus,
    create_caching_server,
    main,
)
from src.registry_loader import RegistryLoader


@pytest.fixture
def minimal_registry(tmp_path: Path) -> RegistryLoader:
    """Create a minimal registry for testing."""
    registry_data = {
        "runtime_defaults": {
            "model_base_path": str(tmp_path),
            "threads": 96,
            "context_length": 8192,
        },
        "roles": {
            "test_role": {
                "tier": "C",
                "description": "Test role",
                "model": {
                    "name": "test-model",
                    "path": "test-model.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 1.0,
                },
                "acceleration": {"type": "none"},
                "performance": {"baseline_tps": 10.0},
                "memory": {"residency": "hot"},
            },
        },
        "command_templates": {},
    }

    (tmp_path / "test-model.gguf").touch()

    registry_path = tmp_path / "registry.yaml"
    with registry_path.open("w") as f:
        yaml.dump(registry_data, f)

    return RegistryLoader(str(registry_path))


class TestLlamaCppBackendInfer:
    """Tests for LlamaCppBackend.infer() subprocess execution."""

    def test_infer_success(self, minimal_registry):
        """Test successful inference with subprocess."""
        backend = LlamaCppBackend(minimal_registry)
        role_config = minimal_registry.get_role("test_role")
        request = InferenceRequest(role="test_role", prompt="Hello", n_tokens=64)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello world\ndecoded 64 tokens in 2.0s, 32.0 t/s"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("src.config.get_config") as mock_config:
                mock_config.return_value.paths.llama_cpp_bin = Path("/usr/bin")
                result = backend.infer(role_config, request)

        assert result.success is True
        assert result.tokens_generated == 64
        assert result.generation_speed == 32.0
        assert "Hello world" in result.output
        # SS-BENCH-GATE-c: the placement guard (api_enforce_placement) also runs
        # subprocess (ps check) before the launch, so assert the launch itself.
        launch_calls = [
            c
            for c in mock_run.call_args_list
            if "llama-completion" in " ".join(c.args[0])
        ]
        assert len(launch_calls) == 1

    def test_infer_subprocess_error(self, minimal_registry):
        """Test inference with subprocess returning error."""
        backend = LlamaCppBackend(minimal_registry)
        role_config = minimal_registry.get_role("test_role")
        request = InferenceRequest(role="test_role", prompt="Hello", n_tokens=64)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error: model file not found"

        with patch("subprocess.run", return_value=mock_result):
            with patch("src.config.get_config") as mock_config:
                mock_config.return_value.paths.llama_cpp_bin = Path("/usr/bin")
                result = backend.infer(role_config, request)

        assert result.success is False
        assert "model file not found" in result.error_message

    def test_infer_timeout_expired(self, minimal_registry):
        """Test inference timeout handling."""
        backend = LlamaCppBackend(minimal_registry)
        role_config = minimal_registry.get_role("test_role")
        request = InferenceRequest(role="test_role", prompt="Hello", n_tokens=64, timeout=30)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            with patch("src.config.get_config") as mock_config:
                mock_config.return_value.paths.llama_cpp_bin = Path("/usr/bin")
                result = backend.infer(role_config, request)

        assert result.success is False
        assert "timed out" in result.error_message.lower()
        assert result.elapsed_time == 30

    def test_infer_general_exception(self, minimal_registry):
        """Test inference with general exception."""
        backend = LlamaCppBackend(minimal_registry)
        role_config = minimal_registry.get_role("test_role")
        request = InferenceRequest(role="test_role", prompt="Hello", n_tokens=64)

        with patch("subprocess.run", side_effect=OSError("Process failed")):
            with patch("src.config.get_config") as mock_config:
                mock_config.return_value.paths.llama_cpp_bin = Path("/usr/bin")
                result = backend.infer(role_config, request)

        assert result.success is False
        assert "Process failed" in result.error_message


class TestLlamaCppBackendHealthCheck:
    """Tests for LlamaCppBackend.health_check()."""

    def test_health_check_pid_zero(self, minimal_registry):
        """Test health check with PID 0 (per-inference mode)."""
        backend = LlamaCppBackend(minimal_registry)

        assert backend.health_check(0) is True

    def test_health_check_process_running(self, minimal_registry):
        """Test health check with running process."""
        backend = LlamaCppBackend(minimal_registry)
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running

        backend._processes[123] = mock_process

        assert backend.health_check(123) is True
        mock_process.poll.assert_called_once()

    def test_health_check_process_terminated(self, minimal_registry):
        """Test health check with terminated process."""
        backend = LlamaCppBackend(minimal_registry)
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Exited

        backend._processes[456] = mock_process

        assert backend.health_check(456) is False

    def test_health_check_unknown_pid(self, minimal_registry):
        """Test health check with unknown PID."""
        backend = LlamaCppBackend(minimal_registry)

        assert backend.health_check(999) is False


class TestLlamaCppBackendLoad:
    """Tests for LlamaCppBackend.load()."""

    def test_load_model_not_found(self, minimal_registry, tmp_path):
        """Test loading non-existent model file."""
        backend = LlamaCppBackend(minimal_registry)

        # Create registry with non-existent model
        registry_data = {
            "runtime_defaults": {"model_base_path": str(tmp_path), "threads": 96},
            "roles": {
                "missing_role": {
                    "tier": "C",
                    "description": "Test",
                    "model": {
                        "name": "missing",
                        "path": "nonexistent.gguf",
                        "quant": "Q4_K_M",
                        "size_gb": 1.0,
                    },
                    "acceleration": {"type": "none"},
                    "performance": {"baseline_tps": 10.0},
                    "memory": {"residency": "hot"},
                },
            },
            "command_templates": {},
        }

        registry_path = tmp_path / "registry.yaml"
        with registry_path.open("w") as f:
            yaml.dump(registry_data, f)

        registry = RegistryLoader(str(registry_path))
        backend = LlamaCppBackend(registry)
        role_config = registry.get_role("missing_role")

        with pytest.raises(FileNotFoundError, match="Model not found"):
            backend.load(role_config)


class TestLlamaCppBackendUnload:
    """Tests for LlamaCppBackend.unload()."""

    def test_unload_with_process_terminate(self, minimal_registry):
        """Test unloading with process termination."""
        backend = LlamaCppBackend(minimal_registry)
        mock_process = MagicMock()
        mock_process.wait.return_value = 0

        backend._processes[123] = mock_process

        result = backend.unload(123)

        assert result is True
        assert 123 not in backend._processes
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)

    def test_unload_with_process_kill(self, minimal_registry):
        """Test unloading with process that needs killing."""
        backend = LlamaCppBackend(minimal_registry)
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)

        backend._processes[123] = mock_process

        result = backend.unload(123)

        assert result is True
        assert 123 not in backend._processes
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_unload_unknown_pid(self, minimal_registry):
        """Test unloading unknown PID returns True."""
        backend = LlamaCppBackend(minimal_registry)

        result = backend.unload(999)
        assert result is True


class TestCreateCachingServer:
    """Tests for create_caching_server() factory."""

    def test_create_caching_server_default(self, tmp_path):
        """Test creating caching server with defaults."""
        # Create minimal registry with at least one role
        (tmp_path / "test-model.gguf").touch()
        registry_data = {
            "runtime_defaults": {"model_base_path": str(tmp_path), "threads": 96},
            "roles": {
                "test_role": {
                    "tier": "C",
                    "description": "Test",
                    "model": {
                        "name": "test",
                        "path": "test-model.gguf",
                        "quant": "Q4_K_M",
                        "size_gb": 1.0,
                    },
                    "acceleration": {"type": "none"},
                    "performance": {"baseline_tps": 10.0},
                    "memory": {"residency": "hot"},
                }
            },
            "command_templates": {},
        }
        registry_path = tmp_path / "registry.yaml"
        with registry_path.open("w") as f:
            yaml.dump(registry_data, f)

        # Lazy imports are done inside create_caching_server - patch at source
        with patch("src.backends.llama_server.LlamaServerBackend") as mock_backend_cls:
            with patch("src.prefix_cache.PrefixRouter") as mock_router_cls:
                with patch("src.prefix_cache.CachingBackend") as mock_caching_cls:
                    mock_backend = MagicMock()
                    mock_backend_cls.return_value = mock_backend
                    mock_router = MagicMock()
                    mock_router_cls.return_value = mock_router
                    mock_caching = MagicMock()
                    mock_caching_cls.return_value = mock_caching

                    server = create_caching_server(
                        base_url="http://localhost:8080",
                        registry_path=str(registry_path),
                    )

        assert isinstance(server, CachingModelServer)
        mock_backend_cls.assert_called_once()
        mock_router_cls.assert_called_once_with(num_slots=4)
        mock_caching_cls.assert_called_once()


class TestCachingModelServer:
    """Tests for CachingModelServer methods."""

    @pytest.fixture
    def caching_server(self, minimal_registry):
        """Create a mock CachingModelServer."""
        mock_backend = MagicMock()
        mock_base_backend = MagicMock()

        return CachingModelServer(
            registry=minimal_registry,
            backend=mock_backend,
            base_backend=mock_base_backend,
        )

    def test_infer_success(self, caching_server):
        """Test CachingModelServer.infer() success."""
        mock_result = InferenceResult(
            role="test_role",
            output="Hello",
            tokens_generated=10,
            generation_speed=50.0,
            elapsed_time=0.2,
            success=True,
        )
        caching_server.backend.infer.return_value = mock_result

        request = InferenceRequest(role="test_role", prompt="Hi")
        result = caching_server.infer(request)

        assert result.success is True
        assert result.output == "Hello"
        caching_server.backend.infer.assert_called_once()

    def test_infer_role_not_found(self, caching_server):
        """Test CachingModelServer.infer() with invalid role."""
        request = InferenceRequest(role="nonexistent", prompt="Hi")

        with pytest.raises(ModelServerError, match="Role not found"):
            caching_server.infer(request)

    def test_health_check_healthy(self, caching_server):
        """Test CachingModelServer.health_check() when healthy."""
        caching_server.base_backend.health_check.return_value = True
        caching_server.backend.get_stats.return_value = {"router_hit_rate": 0.85}

        health = caching_server.health_check()

        assert health["status"] == "healthy"
        assert health["cache_hit_rate"] == 0.85
        assert "timestamp" in health

    def test_health_check_unhealthy(self, caching_server):
        """Test CachingModelServer.health_check() when unhealthy."""
        caching_server.base_backend.health_check.return_value = False
        caching_server.backend.get_stats.return_value = {}

        health = caching_server.health_check()

        assert health["status"] == "unhealthy"

    def test_get_cache_stats(self, caching_server):
        """Test CachingModelServer.get_cache_stats()."""
        caching_server.backend.get_stats.return_value = {
            "hit_rate": 0.75,
            "token_savings": 1000,
        }

        stats = caching_server.get_cache_stats()

        assert stats["hit_rate"] == 0.75
        assert stats["token_savings"] == 1000



class TestMain:
    """Tests for main() CLI entry point."""

    def test_main_default_health_check(self, minimal_registry, tmp_path, monkeypatch):
        """Test main() with no arguments (health check)."""
        monkeypatch.setattr("sys.argv", ["model_server"])

        with patch("src.model_server.ModelServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server.registry.roles.keys.return_value = ["test_role"]
            mock_server.health_check.return_value = {
                "status": "healthy",
                "models": {},
                "timestamp": 1234567890.0,
            }
            mock_server_cls.return_value = mock_server

            result = main()

        assert result == 0
        mock_server.health_check.assert_called_once()

    def test_main_with_role_success(self, minimal_registry, tmp_path, monkeypatch):
        """Test main() with role argument (successful inference)."""
        monkeypatch.setattr("sys.argv", ["model_server", "test_role"])

        with patch("src.model_server.ModelServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server.registry.roles.keys.return_value = ["test_role"]
            mock_server.load.return_value = ModelStatus(role="test_role", state=ModelState.READY)
            mock_server.infer.return_value = InferenceResult(
                role="test_role",
                output="def hello(): print('Hello')",
                tokens_generated=64,
                generation_speed=32.0,
                elapsed_time=2.0,
                success=True,
            )
            mock_server_cls.return_value = mock_server

            result = main()

        assert result == 0
        mock_server.load.assert_called_once_with("test_role")
        mock_server.infer.assert_called_once()

    def test_main_with_role_and_prompt(self, minimal_registry, tmp_path, monkeypatch):
        """Test main() with role and custom prompt."""
        monkeypatch.setattr("sys.argv", ["model_server", "test_role", "Write a fibonacci function"])

        with patch("src.model_server.ModelServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server.registry.roles.keys.return_value = ["test_role"]
            mock_server.load.return_value = ModelStatus(role="test_role", state=ModelState.READY)
            mock_server.infer.return_value = InferenceResult(
                role="test_role",
                output="def fib(n): ...",
                tokens_generated=64,
                generation_speed=30.0,
                elapsed_time=2.0,
                success=True,
            )
            mock_server_cls.return_value = mock_server

            result = main()

        assert result == 0
        call_args = mock_server.infer.call_args
        request = call_args[0][0]
        assert request.prompt == "Write a fibonacci function"

    def test_main_with_role_error(self, minimal_registry, tmp_path, monkeypatch):
        """Test main() with role that fails to load."""
        monkeypatch.setattr("sys.argv", ["model_server", "invalid_role"])

        with patch("src.model_server.ModelServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server.registry.roles.keys.return_value = ["test_role"]
            mock_server.load.side_effect = ModelServerError("Role not found")
            mock_server_cls.return_value = mock_server

            result = main()

        assert result == 1

    def test_main_inference_failure(self, minimal_registry, tmp_path, monkeypatch):
        """Test main() with inference failure."""
        monkeypatch.setattr("sys.argv", ["model_server", "test_role"])

        with patch("src.model_server.ModelServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server.registry.roles.keys.return_value = ["test_role"]
            mock_server.load.return_value = ModelStatus(role="test_role", state=ModelState.READY)
            mock_server.infer.return_value = InferenceResult(
                role="test_role",
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=5.0,
                success=False,
                error_message="Model crashed",
            )
            mock_server_cls.return_value = mock_server

            result = main()

        assert result == 1

    def test_main_server_mode_success(self, minimal_registry, tmp_path, monkeypatch):
        """Test main() with --server flag (success)."""
        monkeypatch.setattr("sys.argv", ["model_server", "--server"])

        with patch("src.model_server.create_caching_server") as mock_create:
            mock_server = MagicMock()
            mock_server.health_check.return_value = {
                "status": "healthy",
                "cache_hit_rate": 0.8,
            }
            mock_server.get_cache_stats.return_value = {
                "hit_rate": 0.8,
                "token_savings": 500,
            }
            mock_create.return_value = mock_server

            result = main()

        assert result == 0
        mock_create.assert_called_once()
        mock_server.health_check.assert_called_once()
        mock_server.get_cache_stats.assert_called_once()

    def test_main_server_mode_failure(self, minimal_registry, tmp_path, monkeypatch):
        """Test main() with --server flag (server not running)."""
        monkeypatch.setattr("sys.argv", ["model_server", "--server"])

        with patch("src.model_server.create_caching_server") as mock_create:
            mock_create.side_effect = Exception("Connection refused")

            result = main()

        assert result == 1


class TestModelStatusEdgeCases:
    """Additional edge case tests for ModelStatus."""

    def test_status_with_all_fields(self):
        """Test ModelStatus with all fields populated."""
        status = ModelStatus(
            role="test_role",
            state=ModelState.BUSY,
            pid=12345,
            loaded_at=1234567890.0,
            last_inference=1234567900.0,
            inference_count=10,
            error_message="Previous warning",
        )

        d = status.to_dict()
        assert d["pid"] == 12345
        assert d["loaded_at"] == 1234567890.0
        assert d["last_inference"] == 1234567900.0
        assert d["inference_count"] == 10
        assert d["error_message"] == "Previous warning"


class TestInferenceRequestEdgeCases:
    """Additional edge case tests for InferenceRequest."""

    def test_request_with_prompt_file(self, tmp_path):
        """Test request with prompt_file instead of prompt."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt from file")

        request = InferenceRequest(
            role="test_role",
            prompt_file=prompt_file,
            n_tokens=128,
        )

        assert request.prompt is None
        assert request.prompt_file == prompt_file

    def test_request_with_stop_sequences(self):
        """Test request with stop sequences."""
        request = InferenceRequest(
            role="test_role",
            prompt="Hello",
            stop_sequences=["END", "STOP"],
        )

        assert request.stop_sequences == ["END", "STOP"]

    def test_request_with_cache_prompt(self):
        """Test request with cache_prompt override."""
        request = InferenceRequest(
            role="test_role",
            prompt="Hello",
            cache_prompt=True,
        )

        assert request.cache_prompt is True


class TestInferenceResultEdgeCases:
    """Additional edge case tests for InferenceResult."""

    def test_result_with_timing_data(self):
        """Test InferenceResult with clean timing data."""
        result = InferenceResult(
            role="test_role",
            output="Hello",
            tokens_generated=50,
            generation_speed=25.0,
            elapsed_time=2.0,
            success=True,
            prompt_eval_ms=500.0,
            generation_ms=1500.0,
            predicted_per_second=33.33,
            http_overhead_ms=50.0,
        )

        d = result.to_dict()
        assert d["prompt_eval_ms"] == 500.0
        assert d["generation_ms"] == 1500.0
        assert d["predicted_per_second"] == 33.33
        assert d["http_overhead_ms"] == 50.0


class TestModelServerLoadError:
    """Tests for ModelServer.load() error handling."""

    def test_load_backend_exception(self, minimal_registry, tmp_path):
        """Test load() when backend raises exception."""
        server = ModelServer(registry=minimal_registry)

        # Mock backend to raise exception
        server.backend.load = MagicMock(side_effect=Exception("Backend error"))

        with pytest.raises(ModelServerError, match="Failed to load"):
            server.load("test_role")

        # Status should show error
        status = server.get_status("test_role")
        assert status.state == ModelState.ERROR
        assert "Backend error" in status.error_message
