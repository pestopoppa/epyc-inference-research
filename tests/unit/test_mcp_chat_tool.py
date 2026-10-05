#!/usr/bin/env python3
"""Tests for CC Local MCP chat delegation tools.

Tests tool functions directly (bypassing MCP transport).
All HTTP calls and feature flags are mocked.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from src.mcp_server import (
    _format_chat_response,
    _post_chat,
    orchestrator_chat,
    orchestrator_route_explain,
)

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _format_chat_response unit tests
# ---------------------------------------------------------------------------


class TestFormatChatResponse:
    def test_success_response(self):
        resp = {
            "answer": "The answer is 42.",
            "routed_to": "frontdoor",
            "routing_strategy": "learned",
            "mode": "direct",
            "elapsed_seconds": 2.5,
        }
        result = _format_chat_response(resp)
        assert "The answer is 42." in result
        assert "role=frontdoor" in result
        assert "strategy=learned" in result
        assert "elapsed=2.5s" in result

    def test_error_in_response(self):
        resp = {"error": "Connection refused"}
        result = _format_chat_response(resp)
        assert "Error: Connection refused" in result

    def test_orchestrator_error_code(self):
        resp = {
            "answer": "",
            "error_code": 504,
            "error_detail": "Backend timeout",
        }
        result = _format_chat_response(resp)
        assert "[Error 504]" in result
        assert "Backend timeout" in result

    def test_empty_response(self):
        result = _format_chat_response({})
        assert result == "(empty response)"


# ---------------------------------------------------------------------------
# _post_chat unit tests
# ---------------------------------------------------------------------------


class TestPostChat:
    @patch("src.mcp_server._get_api_url", return_value="http://localhost:8000")
    @patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen, mock_url):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"answer": "ok"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _post_chat({"prompt": "test", "timeout_s": 30})
        assert result == {"answer": "ok"}

    @patch("src.mcp_server._get_api_url", return_value="http://localhost:8000")
    @patch("urllib.request.urlopen", side_effect=URLError("Connection refused"))
    def test_connection_error(self, mock_urlopen, mock_url):
        result = _post_chat({"prompt": "test"})
        assert "error" in result
        assert "not reachable" in result["error"]

    @patch("src.mcp_server._get_api_url", return_value="http://localhost:8000")
    @patch("urllib.request.urlopen", side_effect=TimeoutError)
    def test_timeout(self, mock_urlopen, mock_url):
        result = _post_chat({"prompt": "test", "timeout_s": 5})
        assert "error" in result
        assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# orchestrator_chat tool tests
# ---------------------------------------------------------------------------


class TestOrchestratorChat:
    @patch.dict(os.environ, {"ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT": "0"}, clear=False)
    def test_feature_disabled(self):
        result = _run(orchestrator_chat("Hello"))
        assert "disabled" in result.lower()
        assert "ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT" in result

    def test_fake_clock_preserves_long_request_deadline_and_reports_progress(self, monkeypatch):
        import src.mcp_server as server

        payloads = []
        progress = []
        class Context:
            async def report_progress(self, current, total, message):
                progress.append((current, total, message))

        async def response(payload):
            payloads.append(payload)
            await asyncio.sleep(0.005)
            return {"answer": "ok"}

        monkeypatch.setattr(server, "_is_mcp_chat_enabled", lambda: True)
        monkeypatch.setattr(server, "_post_chat_async", response)
        monkeypatch.setattr(server.time, "time", lambda: 1_000.0)
        monkeypatch.setattr(server, "_MCP_PROGRESS_INTERVAL_S", 0.001)
        result = _run(orchestrator_chat("slow but valid", timeout_s=95, ctx=Context()))

        assert result == "ok"
        assert payloads[0]["timeout_s"] == 95
        assert payloads[0]["client_deadline_unix_s"] == 1_095.0
        assert progress[0] == (0, 100, "Submitting request to the orchestrator")
        assert progress[-1] == (100, 100, "Orchestrator request finished")
        assert {item[1] for item in progress} == {100}
        assert [item[0] for item in progress] == sorted(item[0] for item in progress)

    def test_cancellation_closes_http_client_without_orphan_request(self, monkeypatch):
        import httpx
        import src.mcp_server as server

        started = asyncio.Event()
        closed = asyncio.Event()

        class HangingClient:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                closed.set()

            async def post(self, *_args, **_kwargs):
                started.set()
                await asyncio.Future()

        async def exercise():
            monkeypatch.setattr(server, "_get_api_url", lambda: "http://orchestrator")
            monkeypatch.setattr(httpx, "AsyncClient", HangingClient)
            monkeypatch.setattr(server, "_is_mcp_chat_enabled", lambda: True)
            task = asyncio.create_task(orchestrator_chat("cancel me", timeout_s=95))
            await started.wait()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            assert closed.is_set()
            assert task.done()

        _run(exercise())

    def test_progress_callback_error_cancels_request_task(self, monkeypatch):
        import src.mcp_server as server

        started = asyncio.Event()
        cancelled = asyncio.Event()

        class Context:
            calls = 0

            async def report_progress(self, *_args):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("client progress channel closed")

        async def pending_request(_payload):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def exercise():
            monkeypatch.setattr(server, "_post_chat_async", pending_request)
            monkeypatch.setattr(server, "_MCP_PROGRESS_INTERVAL_S", 0.001)
            task = asyncio.create_task(server._await_chat_request({"timeout_s": 1}, Context()))
            await started.wait()
            try:
                await task
            except RuntimeError as exc:
                assert str(exc) == "client progress channel closed"
            else:
                raise AssertionError("progress failure should propagate")
            assert cancelled.is_set()
            assert task.done()

        _run(exercise())

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat_async")
    def test_success(self, mock_post, mock_enabled):
        async def response(_payload):
            return {
            "answer": "The answer is 42.",
            "routed_to": "coder_escalation",
            "routing_strategy": "learned",
            "mode": "repl",
            "elapsed_seconds": 3.1,
            }
        mock_post.side_effect = response
        result = _run(orchestrator_chat("What is 6 times 7?"))
        assert "42" in result
        assert "coder_escalation" in result
        # Verify payload
        call_args = mock_post.call_args[0][0]
        assert call_args["prompt"] == "What is 6 times 7?"
        assert call_args["real_mode"] is True
        assert call_args["mock_mode"] is False

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat_async")
    def test_force_role(self, mock_post, mock_enabled):
        async def response(_payload):
            return {"answer": "done", "routed_to": _RETIRED_ARCHITECT_ROLE}
        mock_post.side_effect = response
        _run(orchestrator_chat("Fix the bug", force_role=_RETIRED_ARCHITECT_ROLE))
        call_args = mock_post.call_args[0][0]
        assert call_args["force_role"] == _RETIRED_ARCHITECT_ROLE

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat_async")
    def test_connection_error(self, mock_post, mock_enabled):
        async def response(_payload):
            return {"error": "Orchestrator not reachable at http://localhost:8000: Connection refused"}
        mock_post.side_effect = response
        result = _run(orchestrator_chat("Hello"))
        assert "not reachable" in result

    @patch.dict(os.environ, {
        "ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT": "1",
        "ORCHESTRATOR_API_URL": "http://localhost:9999",
    }, clear=False)
    @patch("src.mcp_server._ORCHESTRATOR_API_URL", None)  # Reset cached value
    @patch("src.mcp_server._post_chat_async")
    def test_custom_api_url(self, mock_post):
        async def response(_payload):
            return {"answer": "ok"}
        mock_post.side_effect = response
        # _get_api_url should pick up the env var
        from src.mcp_server import _get_api_url
        # Reset the cached value to force re-read
        import src.mcp_server
        src.mcp_server._ORCHESTRATOR_API_URL = None
        url = _get_api_url()
        assert url == "http://localhost:9999"
        # Clean up
        src.mcp_server._ORCHESTRATOR_API_URL = None


# ---------------------------------------------------------------------------
# orchestrator_route_explain tool tests
# ---------------------------------------------------------------------------


class TestOrchestratorRouteExplain:
    @patch.dict(os.environ, {"ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT": "0"}, clear=False)
    def test_feature_disabled(self):
        result = orchestrator_route_explain("Hello")
        assert "disabled" in result.lower()

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat")
    def test_success(self, mock_post, mock_enabled):
        mock_post.return_value = {
            "routed_to": "architect_general",
            "routing_strategy": "rules",
            "mode": "delegated",
            "tool_required": True,
            "timeout_s": 60,
        }
        result = orchestrator_route_explain("Explain quantum computing")
        assert "architect_general" in result
        assert "rules" in result
        assert "delegated" in result
        assert "True" in result
        # Verify mock_mode was set
        call_args = mock_post.call_args[0][0]
        assert call_args["mock_mode"] is True
        assert call_args["real_mode"] is False

    @patch("src.mcp_server._is_mcp_chat_enabled", return_value=True)
    @patch("src.mcp_server._post_chat")
    def test_connection_error(self, mock_post, mock_enabled):
        mock_post.return_value = {"error": "Orchestrator not reachable"}
        result = orchestrator_route_explain("Hello")
        assert "Error:" in result
