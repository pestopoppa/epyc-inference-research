"""Coverage for scripts.lib.executor inference result semantics."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import requests

from scripts.lib.executor import Config, Executor, ServerManager


class _StreamingResponse:
    status_code = 200
    text = ""

    def iter_lines(self, decode_unicode: bool = True):
        yield 'data: {"content":"partial"}'
        raise requests.exceptions.Timeout("read timed out")


def test_server_manager_partial_timeout_sets_degraded_metadata():
    manager = ServerManager(port=9999)
    session = MagicMock()
    session.post.return_value = _StreamingResponse()
    manager._http_session = session

    result = manager.run_inference("prompt", timeout=1)

    assert result.success is False
    assert result.timed_out is True
    assert result.partial is True
    assert result.degraded is True
    assert result.failure_stage == "http_stream"
    assert result.failure_reason == "timeout_partial"
    assert "partial" in result.raw_output


def test_server_manager_request_error_sets_failure_metadata():
    manager = ServerManager(port=9999)
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError("connection refused")
    manager._http_session = session

    result = manager.run_inference("prompt", timeout=1)

    assert result.success is False
    assert result.timed_out is False
    assert result.partial is False
    assert result.degraded is False
    assert result.failure_stage == "http_request"
    assert result.failure_reason == "request_error"
    assert "connection refused" in result.raw_output


def test_server_manager_http_status_sets_failure_metadata():
    manager = ServerManager(port=9999)
    session = MagicMock()
    session.post.return_value = MagicMock(status_code=503, text="backend down")
    manager._http_session = session

    result = manager.run_inference("prompt", timeout=1)

    assert result.success is False
    assert result.failure_stage == "http_request"
    assert result.failure_reason == "http_status"
    assert "HTTP 503" in result.raw_output


def test_executor_subprocess_timeout_sets_partial_and_degraded_metadata():
    executor = Executor(validate=False)
    timeout_error = subprocess.TimeoutExpired(
        cmd=["llama-cli"],
        timeout=5,
        output=b"partial stdout",
        stderr=b"partial stderr",
    )
    temp_file = MagicMock()
    temp_file.name = "/tmp/fake-prompt.txt"

    with (
        patch("scripts.lib.executor.tempfile.NamedTemporaryFile") as named_temp,
        patch.object(Executor, "build_command", return_value=["llama-cli", "-m", "model.gguf"]),
        patch("scripts.lib.executor.subprocess.run", side_effect=timeout_error),
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
    assert result.timed_out is True
    assert result.partial is True
    assert result.degraded is True
    assert result.failure_stage == "subprocess"
    assert result.failure_reason == "timeout_partial"
    assert result.raw_output == "partial stdout"
    assert result.stderr == "partial stderr"


def test_server_manager_vl_timeout_sets_failure_metadata():
    manager = ServerManager(port=9999)
    manager.mmproj_path = "mmproj.gguf"
    session = MagicMock()
    session.post.side_effect = requests.exceptions.Timeout("vl timeout")
    manager._http_session = session

    result = manager.run_inference("prompt", timeout=1)

    assert result.success is False
    assert result.timed_out is True
    assert result.degraded is True
    assert result.failure_stage == "http_request"
    assert result.failure_reason == "timeout"
