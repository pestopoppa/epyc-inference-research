"""Additional coverage for scripts.lib.executor non-happy paths."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch, mock_open

import pytest
import requests

import scripts.lib.executor as executor_mod
from scripts.lib.executor import (
    Config,
    Executor,
    ServerManager,
    _numa_prefix,
    build_command,
    get_binary_paths,
    get_server_defaults,
    run_inference,
    validate_binaries,
)


def test_server_manager_wait_ready_returns_false_when_process_dies():
    manager = ServerManager(port=9999)
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("not ready")
    manager._http_session = session
    manager.process = MagicMock()
    manager.process.poll.return_value = 1
    manager._stderr_file = MagicMock()
    manager._stderr_file.name = "/tmp/server.log"

    with (
        patch("builtins.open", MagicMock()),
        patch("scripts.lib.executor.time.sleep"),
    ):
        assert manager.wait_ready(timeout=1) is False


def test_server_manager_wait_ready_returns_true_on_healthy_probe():
    manager = ServerManager(port=9999)
    manager._http_session = MagicMock(get=MagicMock(return_value=MagicMock(status_code=200)))
    manager._stderr_file = MagicMock()
    manager._stderr_file.name = "/tmp/server.log"

    with patch("builtins.open", mock_open(read_data="n_expert_used = 4\n")):
        assert manager.wait_ready(timeout=1) is True


def test_numa_prefix_depends_on_numactl_availability():
    with patch("scripts.lib.executor.shutil.which", return_value="/usr/bin/numactl"):
        assert _numa_prefix() == ["numactl", "--interleave=all"]
    with patch("scripts.lib.executor.shutil.which", return_value=None):
        assert _numa_prefix() == []


def test_get_binary_paths_and_server_defaults_fallback_when_registry_load_fails():
    with patch("scripts.lib.executor.load_registry", side_effect=RuntimeError("registry down")):
        paths = get_binary_paths(None)
        defaults = get_server_defaults(None)
    assert paths["completion"] == "llama-completion"
    assert defaults["context_length"] == 131072


def test_server_manager_start_builds_expected_command_flags():
    manager = ServerManager(port=9000)
    manager.process = object()  # Force stop() call path.
    registry = MagicMock()
    registry.get_max_context.return_value = 4096
    registry.get_flash_attention.return_value = False
    registry.get_ubatch_size.return_value = 2048
    fake_log = MagicMock(name="/tmp/server.log")
    fake_log.name = "/tmp/server.log"

    with (
        patch.object(ServerManager, "stop") as stop,
        patch("scripts.lib.executor._numa_prefix", return_value=[]),
        patch("scripts.lib.executor.get_binary", return_value="/bin/llama-server"),
        patch("scripts.lib.executor.tempfile.NamedTemporaryFile", return_value=fake_log),
        patch("scripts.lib.executor.subprocess.Popen") as popen,
    ):
        manager.start(
            model_path="model.gguf",
            role="worker",
            registry=registry,
            moe_override="qwen3moe.expert_used_count=int:4",
            no_mmap=True,
            draft_model_path="draft.gguf",
            draft_max=12,
            mmproj_path="mmproj.gguf",
        )

    stop.assert_called_once()
    cmd = popen.call_args.args[0]
    assert "/bin/llama-server" in cmd
    assert "--override-kv" in cmd
    assert "--no-mmap" in cmd
    assert "-md" in cmd
    assert "--spec-draft-n-max" in cmd
    assert "--mmproj" in cmd
    assert "-fa" not in cmd  # Registry requested flash-attn off.


def test_server_manager_start_omits_same_file_draft_model_path():
    manager = ServerManager(port=9000)
    registry = MagicMock()
    registry.get_max_context.return_value = 4096
    registry.get_flash_attention.return_value = False
    registry.get_ubatch_size.return_value = 2048
    fake_log = MagicMock(name="/tmp/server.log")
    fake_log.name = "/tmp/server.log"

    with (
        patch("scripts.lib.executor._numa_prefix", return_value=[]),
        patch("scripts.lib.executor.get_binary", return_value="/bin/llama-server"),
        patch("scripts.lib.executor.tempfile.NamedTemporaryFile", return_value=fake_log),
        patch("scripts.lib.executor.subprocess.Popen") as popen,
    ):
        manager.start(
            model_path="/models/qwen-mtp.gguf",
            role="frontdoor",
            registry=registry,
            draft_model_path="/models/qwen-mtp.gguf",
            draft_max=4,
        )

    cmd = popen.call_args.args[0]
    assert "-md" not in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "4"


def test_server_manager_stop_kills_process_after_timeout():
    manager = ServerManager(port=9000)
    manager._http_session = MagicMock()
    proc = MagicMock()
    proc.wait.side_effect = [subprocess.TimeoutExpired(cmd=["x"], timeout=10), None]
    manager.process = proc
    manager.model_path = "model.gguf"
    manager.mmproj_path = "mmproj.gguf"

    manager.stop()

    manager._http_session = None
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert manager.process is None
    assert manager.model_path is None
    assert manager.mmproj_path is None


def test_executor_run_inference_nonzero_exit_sets_failure_metadata():
    executor = Executor(validate=False)
    temp_file = MagicMock()
    temp_file.name = "/tmp/fake-prompt.txt"
    completed = subprocess.CompletedProcess(
        args=["llama-cli"],
        returncode=2,
        stdout="bad output",
        stderr="fatal error\neval time = 1.0 ms",
    )

    with (
        patch("scripts.lib.executor.tempfile.NamedTemporaryFile") as named_temp,
        patch.object(Executor, "build_command", return_value=["llama-cli", "-m", "model.gguf"]),
        patch("scripts.lib.executor.subprocess.run", return_value=completed),
        patch("scripts.lib.executor.os.path.exists", return_value=False),
    ):
        named_temp.return_value.__enter__.return_value = temp_file
        result = executor.run_inference(
            model_path="model.gguf",
            config=Config.baseline(),
            prompt="prompt",
            timeout=5,
        )

    assert result.success is False
    assert result.failure_stage == "subprocess"
    assert result.failure_reason == "nonzero_exit"
    assert "eval time" in result.raw_output
    assert result.stderr == "fatal error\neval time = 1.0 ms"


def test_server_manager_vl_http_status_sets_failure_metadata():
    manager = ServerManager(port=9999)
    manager.mmproj_path = "mmproj.gguf"
    session = MagicMock()
    session.post.return_value = MagicMock(status_code=502, text="upstream down")
    manager._http_session = session

    result = manager.run_inference("prompt", timeout=1)

    assert result.success is False
    assert result.failure_stage == "http_request"
    assert result.failure_reason == "http_status"
    assert "HTTP 502" in result.raw_output


def test_server_manager_vl_request_error_sets_failure_metadata():
    manager = ServerManager(port=9999)
    manager.mmproj_path = "mmproj.gguf"
    session = MagicMock()
    session.post.side_effect = requests.exceptions.RequestException("vl request failed")
    manager._http_session = session

    result = manager.run_inference("prompt", timeout=1)

    assert result.success is False
    assert result.failure_stage == "http_request"
    assert result.failure_reason == "request_error"


def test_executor_build_command_for_vision_moe_includes_mmproj_image_and_override():
    executor = Executor(validate=False)
    cfg = Config.moe(4, "qwen3moe.expert_used_count")

    with (
        patch("scripts.lib.executor._numa_prefix", return_value=[]),
        patch("scripts.lib.executor.get_binary", return_value="/bin/llama-mtmd-cli"),
    ):
        cmd = executor.build_command(
            model_path="vision.gguf",
            config=cfg,
            prompt_file="/tmp/prompt.txt",
            mmproj_path="mmproj.gguf",
            image_path="/tmp/image.png",
            context_size=4096,
        )

    assert "--mmproj" in cmd
    assert "--image" in cmd
    assert "--override-kv" in cmd
    assert "-c" in cmd


def test_executor_build_command_lookup_uses_safe_context_and_draft_max():
    executor = Executor(validate=False)
    cfg = Config.lookup(4)

    with (
        patch("scripts.lib.executor._numa_prefix", return_value=[]),
        patch("scripts.lib.executor.get_binary", side_effect=lambda name, registry=None: f"/bin/{name}"),
    ):
        cmd = executor.build_command(
            model_path="model.gguf",
            config=cfg,
            prompt_file="/tmp/prompt.txt",
            context_size=1000,
        )

    # context_size is multiplied by 3 for safety.
    assert "-c" in cmd and "3000" in cmd
    assert "-b" in cmd and "3000" in cmd
    assert "--draft-max" in cmd and "4" in cmd


def test_convenience_functions_delegate_to_executor_methods():
    fake_exec = MagicMock()
    fake_exec.build_command.return_value = ["cmd"]
    fake_exec.run_inference.return_value = "result"

    with patch("scripts.lib.executor.Executor", return_value=fake_exec):
        cmd = build_command("model.gguf", Config.baseline(), "/tmp/prompt.txt")
        result = run_inference("model.gguf", Config.baseline(), "prompt", timeout=7)

    assert cmd == ["cmd"]
    assert result == "result"
    fake_exec.build_command.assert_called_once()
    fake_exec.run_inference.assert_called_once()


def test_executor_run_inference_enables_paged_attention_env_when_recommended():
    executor = Executor(validate=False)
    executor.registry = MagicMock()
    executor.registry.get_role_config.return_value = {
        "paged_attention": {"recommended": True, "block_size": 128}
    }
    temp_file = MagicMock()
    temp_file.name = "/tmp/fake-prompt.txt"
    completed = subprocess.CompletedProcess(
        args=["llama-cli"],
        returncode=0,
        stdout="ok",
        stderr="",
    )

    with (
        patch("scripts.lib.executor.tempfile.NamedTemporaryFile") as named_temp,
        patch.object(Executor, "build_command", return_value=["llama-cli", "-m", "model.gguf"]),
        patch("scripts.lib.executor.subprocess.run", return_value=completed) as run,
        patch("scripts.lib.executor.os.path.exists", return_value=False),
    ):
        named_temp.return_value.__enter__.return_value = temp_file
        result = executor.run_inference(
            model_path="model.gguf",
            config=Config.baseline(),
            prompt="prompt",
            timeout=5,
            role="worker",
        )

    assert result.success is True
    assert run.call_args.kwargs["env"]["LLAMA_PAGED_ATTN"] == "128"


def test_executor_run_inference_cleans_up_temp_prompt_file():
    executor = Executor(validate=False)
    temp_file = MagicMock()
    temp_file.name = "/tmp/fake-prompt-cleanup.txt"
    completed = subprocess.CompletedProcess(
        args=["llama-cli"],
        returncode=0,
        stdout="ok",
        stderr="",
    )

    with (
        patch("scripts.lib.executor.tempfile.NamedTemporaryFile") as named_temp,
        patch.object(Executor, "build_command", return_value=["llama-cli", "-m", "model.gguf"]),
        patch("scripts.lib.executor.subprocess.run", return_value=completed),
        patch("scripts.lib.executor.os.path.exists", side_effect=lambda p: p == temp_file.name),
        patch("scripts.lib.executor.os.unlink") as unlink,
    ):
        named_temp.return_value.__enter__.return_value = temp_file
        result = executor.run_inference(
            model_path="model.gguf",
            config=Config.baseline(),
            prompt="prompt",
            timeout=5,
        )

    assert result.success is True
    unlink.assert_called_once_with(temp_file.name)


def test_executor_module_standalone_import_uses_registry_fallback_path():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "lib" / "executor.py"
    module_name = "executor_fallback_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    stub_registry = ModuleType("registry")
    stub_registry.ModelRegistry = type("ModelRegistry", (), {})
    stub_registry.load_registry = lambda: None
    sys.modules[module_name] = module
    sys.modules["registry"] = stub_registry
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("registry", None)
        sys.modules.pop(module_name, None)

    assert hasattr(module, "ModelRegistry")
    assert module.DEFAULT_TIMEOUT == 180


def test_read_registry_timeout_uses_registry_runtime_defaults():
    reg = MagicMock()
    reg._raw = {"runtime_defaults": {"timeouts": {"scripts": {"executor_default": 77}, "default": 22}}}

    with patch("scripts.lib.executor.load_registry", return_value=reg):
        timeout = executor_mod._read_registry_timeout("scripts", "executor_default", 180)
    assert timeout == 77


def test_validate_binaries_raises_when_required_binary_missing():
    with (
        patch("scripts.lib.executor.get_binary", side_effect=["/bin/a", "/bin/b", "/bin/c"]),
        patch("scripts.lib.executor.os.path.exists", side_effect=[True, False, True]),
    ):
        with pytest.raises(FileNotFoundError) as exc:
            validate_binaries(registry=MagicMock())
    assert "Missing llama.cpp binaries" in str(exc.value)
    assert "speculative" in str(exc.value)


def test_server_manager_get_http_session_creates_session_once():
    manager = ServerManager(port=9999)
    session = MagicMock()

    with patch("scripts.lib.executor.requests.Session", return_value=session) as session_cls:
        assert manager._get_http_session() is session
        assert manager._get_http_session() is session

    session_cls.assert_called_once()


def test_server_manager_start_uses_explicit_context_and_flash_attention_flag():
    manager = ServerManager(port=9000)
    registry = MagicMock()
    registry.get_flash_attention.return_value = True
    registry.get_ubatch_size.return_value = 4096
    fake_log = MagicMock()
    fake_log.name = "/tmp/server.log"

    with (
        patch("scripts.lib.executor._numa_prefix", return_value=[]),
        patch("scripts.lib.executor.get_binary", return_value="/bin/llama-server"),
        patch("scripts.lib.executor.tempfile.NamedTemporaryFile", return_value=fake_log),
        patch("scripts.lib.executor.subprocess.Popen") as popen,
    ):
        manager.start(
            model_path="model.gguf",
            context_length=777,
            role="worker",
            registry=registry,
        )

    cmd = popen.call_args.args[0]
    ctx_index = cmd.index("-c")
    assert cmd[ctx_index + 1] == "777"
    assert "-fa" in cmd and "on" in cmd


def test_server_manager_start_uses_default_context_when_not_overridden():
    manager = ServerManager(port=9000)
    manager.context_length = 555
    fake_log = MagicMock()
    fake_log.name = "/tmp/server.log"

    with (
        patch("scripts.lib.executor._numa_prefix", return_value=[]),
        patch("scripts.lib.executor.get_binary", return_value="/bin/llama-server"),
        patch("scripts.lib.executor.tempfile.NamedTemporaryFile", return_value=fake_log),
        patch("scripts.lib.executor.subprocess.Popen") as popen,
    ):
        manager.start(model_path="model.gguf", registry=None, role=None)

    cmd = popen.call_args.args[0]
    ctx_index = cmd.index("-c")
    assert cmd[ctx_index + 1] == "555"


def test_server_manager_wait_ready_uses_default_timeout_and_returns_false_after_wait():
    manager = ServerManager(port=9999)
    manager.startup_timeout = 1
    manager._http_session = MagicMock()
    manager._http_session.get.side_effect = requests.exceptions.ConnectionError("not ready")

    with (
        patch("scripts.lib.executor.time.time", side_effect=[0.0, 0.0, 2.0]),
        patch("scripts.lib.executor.time.sleep") as sleep,
    ):
        assert manager.wait_ready(timeout=None) is False

    sleep.assert_called_once_with(1)


def test_server_manager_wait_ready_swallows_stderr_read_errors_on_success():
    manager = ServerManager(port=9999)
    manager._http_session = MagicMock(get=MagicMock(return_value=MagicMock(status_code=200)))
    manager._stderr_file = MagicMock()
    manager._stderr_file.name = "/tmp/server.log"

    with patch("builtins.open", side_effect=RuntimeError("read failed")):
        assert manager.wait_ready(timeout=1) is True


def test_server_manager_wait_ready_process_died_logs_stderr_tail_and_handles_read_errors():
    manager = ServerManager(port=9999)
    manager._http_session = MagicMock()
    manager._http_session.get.side_effect = requests.exceptions.ConnectionError("not ready")
    manager.process = MagicMock()
    manager.process.poll.return_value = 1
    manager._stderr_file = MagicMock()
    manager._stderr_file.name = "/tmp/server.log"

    with patch("builtins.open", mock_open(read_data="line1\nline2\n")):
        assert manager.wait_ready(timeout=1) is False

    with patch("builtins.open", side_effect=RuntimeError("log read failed")):
        assert manager.wait_ready(timeout=1) is False


def test_server_manager_is_running_reflects_process_state():
    manager = ServerManager(port=9999)
    assert manager.is_running() is False

    manager.process = MagicMock()
    manager.process.poll.return_value = None
    assert manager.is_running() is True

    manager.process.poll.return_value = 1
    assert manager.is_running() is False


def test_get_configs_for_dense_uses_model_size_fallback_when_registry_size_missing():
    executor = Executor(registry=MagicMock(), validate=False)
    reg = MagicMock()
    reg.get_forbidden_configs.return_value = []
    reg.get_drafts_for_model.return_value = []
    reg.get_role_config.return_value = {"tier": "B", "model": {"size_gb": 0}}
    reg.get_model_path.return_value = "/models/main.gguf"

    with (
        patch("scripts.lib.executor.os.path.exists", return_value=True),
        patch("scripts.lib.executor.os.path.getsize", return_value=10 * (1024**3)),
    ):
        configs = executor.get_configs_for_architecture("dense", "main", registry=reg)

    names = {cfg.name for cfg in configs}
    assert {"lookup_n3", "lookup_n4", "lookup_n5"}.issubset(names)
