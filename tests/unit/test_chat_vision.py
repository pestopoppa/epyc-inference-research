"""Comprehensive tests for chat vision pipeline.

Tests coverage for src/api/routes/chat_vision.py (currently under-tested).
"""

import base64
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.api.routes.chat_vision import (
    _execute_vision_tool,
    _fallback_vl_port_for_role,
    _handle_multi_file_vision,
    _handle_vision_request,
    _needs_structured_analysis,
    _safe_eval_math,
    _stack_prior_vl_urls,
    _vl_url_for_port,
    _vl_url_for_role,
    _vision_react_mode_answer,
)
from src.api.routes.vision_serving import stack_prior_vision_roles, vision_roles

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry_serving_port(role: str) -> int:
    """The port of the process that actually serves ``role``, per the registry.

    Follows `alias_of` (a row that launches no server of its own) and
    `shared_with` (a role co-hosted on another row's process) — the two forms
    orchestration/model_registry.yaml uses. Reading it here rather than pinning
    a literal is what keeps these tests honest across a role move: the
    2026-08-01 W1 vision unification retired vision_escalation's standalone
    :8087 7B and made the role an alias on worker_vision's :8086 process, and a
    pinned 8087 asserted a degraded fallback onto a port that no longer exists.
    """
    import yaml

    server_mode = yaml.safe_load(
        (_REPO_ROOT / "orchestration" / "model_registry.yaml").read_text(encoding="utf-8")
    )["server_mode"]
    seen: set[str] = set()
    while role not in seen:
        seen.add(role)
        entry = server_mode.get(role)
        if entry is not None:
            target = entry.get("alias_of")
            if target and target not in seen:
                role = target
                continue
            return int(entry["port"])
        hosts = [
            host
            for host, row in server_mode.items()
            if role in (row.get("shared_with") or [])
        ]
        assert len(hosts) == 1, f"role {role!r} is not hosted by exactly one row: {hosts}"
        role = hosts[0]
    raise AssertionError(f"role {role!r} has no resolvable server_mode port")


class TestStructuredAnalysisDetection:
    """Test structured analysis keyword detection."""

    def test_needs_structured_analysis_with_keywords(self):
        """Test detection of structured analysis keywords."""
        assert _needs_structured_analysis("analyze this diagram") is True
        assert _needs_structured_analysis("Architecture review needed") is True
        assert _needs_structured_analysis("Protocol whitepaper analysis") is True
        assert _needs_structured_analysis("security audit of smart contract") is True
        assert _needs_structured_analysis("flowchart showing data flow") is True

    def test_needs_structured_analysis_without_keywords(self):
        """Test prompts without structured keywords."""
        assert _needs_structured_analysis("what color is this?") is False
        assert _needs_structured_analysis("extract text from image") is False
        assert _needs_structured_analysis("read the sign") is False
        assert _needs_structured_analysis("") is False

    def test_needs_structured_analysis_case_insensitive(self):
        """Test case-insensitive keyword matching."""
        assert _needs_structured_analysis("ANALYZE this DIAGRAM") is True
        assert _needs_structured_analysis("FlowChart") is True


class TestVisionServingUrls:
    """Test generated stack-prior VL endpoint discovery."""

    def test_stack_prior_vl_urls_prefers_live_endpoints(self, tmp_path: Path):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  worker_vision:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:9101
      ports: [9101]
      launch:
        modes: [vision]
        entries:
          - mode: vision
  vision_escalation:
    deployment_status: live_stack
    serving:
      ports: [9107]
      launch:
        entries:
          - vision_type: escalation
  stale_vision:
    deployment_status: benchmark_or_candidate
    serving:
      endpoint: http://localhost:9999
      launch:
        modes: [vision]
""",
            encoding="utf-8",
        )

        assert _stack_prior_vl_urls(priors) == {
            "worker_vision": "http://localhost:9101",
            "vision_escalation": "http://localhost:9107",
        }

    def test_stack_prior_vision_roles_uses_generated_launch_metadata(self, tmp_path: Path):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  worker_vision:
    deployment_status: live_stack
    serving:
      launch:
        modes: [vision]
        entries: []
  document_vision:
    deployment_status: live_stack
    serving:
      launch:
        modes: [auxiliary]
        entries:
          - vision_type: document
  stale_vision:
    deployment_status: benchmark_or_candidate
    serving:
      launch:
        modes: [vision]
        entries: []
  worker_general:
    deployment_status: live_stack
    serving:
      launch:
        modes: [worker_pool]
        entries: []
""",
            encoding="utf-8",
        )

        assert stack_prior_vision_roles(priors) == {"worker_vision", "document_vision"}
        assert vision_roles(priors) == {"worker_vision", "document_vision"}

    def test_vision_roles_do_not_legacy_fallback_when_artifact_has_no_vision_roles(
        self, tmp_path: Path
    ):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  worker_general:
    deployment_status: live_stack
    serving:
      launch:
        modes: [worker_pool]
        entries: []
""",
            encoding="utf-8",
        )

        assert vision_roles(priors) == frozenset()

    def test_vl_url_for_role_falls_back_to_config_url(self, tmp_path: Path):
        """No priors artifact -> degraded resolution, which must still land on
        the port the REGISTRY declares for the role's serving process."""
        expected = _registry_serving_port("vision_escalation")
        assert _vl_url_for_role("vision_escalation", tmp_path / "missing.yaml") == (
            f"http://localhost:{expected}"
        )
        # W1: the escalation tier rides worker_vision's process, so both roles
        # resolve to the same endpoint.
        assert expected == _registry_serving_port("worker_vision")
        assert _vl_url_for_role("worker_vision", tmp_path / "missing.yaml") == (
            f"http://localhost:{expected}"
        )

    def test_vl_url_for_role_rejects_stale_degraded_fallback_when_priors_exist(
        self, tmp_path: Path
    ):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  worker_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:9102
      launch:
        modes: [worker_pool]
        entries: []
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="degraded fallback disabled"):
            _vl_url_for_role("vision_escalation", priors)

    def test_vl_url_for_port_rejects_stale_degraded_fallback_when_priors_exist(
        self, tmp_path: Path
    ):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  worker_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:9102
      launch:
        modes: [worker_pool]
        entries: []
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="degraded fallback disabled"):
            _vl_url_for_port(8086, priors)

    def test_fallback_vl_port_prefers_manifest_port(self, monkeypatch):
        monkeypatch.setattr(
            "src.api.routes.vision_serving.manifest_vl_port_for_role",
            lambda role: 9991 if role == "worker_vision" else 9997,
        )

        assert _fallback_vl_port_for_role("worker_vision") == 9991
        assert _fallback_vl_port_for_role("vision_escalation") == 9997

    def test_fallback_vl_port_uses_legacy_fallback_when_manifest_missing(self, monkeypatch):
        """With the launch manifest unreadable the hardcoded legacy table is the
        last resort — and its literals must still be the ports the registry
        declares. A legacy entry that drifts off the registry is a degraded
        fallback onto a dead server, which fails at CONNECT time instead of at
        resolution time (that is exactly what `vision_escalation: 8087` became
        after the W1 vision unification)."""
        monkeypatch.setattr(
            "src.api.routes.vision_serving.manifest_vl_port_for_role",
            lambda _role: None,
        )
        for role in ("worker_vision", "vision_escalation"):
            assert _fallback_vl_port_for_role(role) == _registry_serving_port(role), role

    def test_fallback_vl_port_rejects_unknown_degraded_role(self):
        with pytest.raises(ValueError, match="No degraded VL port fallback"):
            _fallback_vl_port_for_role("document_vision")

    def test_vl_url_for_port_resolves_generated_endpoint(self, tmp_path: Path):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  worker_vision:
    deployment_status: live_stack
    serving:
      endpoint: http://127.0.0.1:9101
      launch:
        modes: [vision]
        entries:
          - mode: vision
""",
            encoding="utf-8",
        )

        assert _vl_url_for_port(9101, priors) == "http://127.0.0.1:9101"


class TestSafeEvalMath:
    """Test safe math expression evaluation."""

    def test_safe_eval_math_basic_operations(self):
        """Test basic arithmetic operations."""
        assert _safe_eval_math("2 + 3") == 5
        assert _safe_eval_math("10 - 4") == 6
        assert _safe_eval_math("3 * 4") == 12
        assert _safe_eval_math("15 / 3") == 5
        assert _safe_eval_math("17 // 4") == 4
        assert _safe_eval_math("17 % 4") == 1
        assert _safe_eval_math("2 ** 3") == 8

    def test_safe_eval_math_complex_expressions(self):
        """Test complex nested expressions."""
        assert _safe_eval_math("(2 + 3) * 4") == 20
        assert _safe_eval_math("10 / (2 + 3)") == 2
        assert _safe_eval_math("2 ** (3 + 1)") == 16

    def test_safe_eval_math_floats(self):
        """Test floating point operations."""
        assert _safe_eval_math("2.5 + 3.5") == 6.0
        assert _safe_eval_math("10.0 / 4.0") == 2.5

    def test_safe_eval_math_unary_minus(self):
        """Test unary negation."""
        assert _safe_eval_math("-5") == -5
        assert _safe_eval_math("-(2 + 3)") == -5

    def test_safe_eval_math_blocked_functions(self):
        """Test that function calls are blocked."""
        with pytest.raises(ValueError, match="Unsupported expression"):
            _safe_eval_math("print(1)")

    def test_safe_eval_math_blocked_imports(self):
        """Test that imports are blocked."""
        with pytest.raises(SyntaxError):
            _safe_eval_math("import os")

    def test_safe_eval_math_blocked_attributes(self):
        """Test that attribute access is blocked."""
        with pytest.raises(ValueError, match="Unsupported expression"):
            _safe_eval_math("x.y")

    def test_safe_eval_math_blocked_names(self):
        """Test that variable names are blocked."""
        with pytest.raises(ValueError, match="Unsupported expression"):
            _safe_eval_math("x + 1")


class TestExecuteVisionTool:
    """Test vision tool execution dispatcher."""

    @pytest.mark.asyncio
    async def test_execute_ocr_extract_success(self):
        """Test OCR extract tool with successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Extracted text from image"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = await _execute_vision_tool('ocr_extract(image_base64="...")', "base64data")

            assert result == "Extracted text from image"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_ocr_extract_http_error(self):
        """Test OCR extract tool with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = await _execute_vision_tool('ocr_extract(image_base64="...")', "base64data")

            assert result.startswith("[OCR error: HTTP 500]")

    @pytest.mark.asyncio
    async def test_execute_ocr_extract_exception(self):
        """Test OCR extract tool with exception."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client

            result = await _execute_vision_tool('ocr_extract(image_base64="...")', "base64data")

            assert result.startswith("[OCR error:")
            assert "Connection failed" in result

    @pytest.mark.asyncio
    async def test_execute_calculate_tool(self):
        """Test calculate tool with valid expression."""
        result = await _execute_vision_tool('calculate(expression="2 + 3")', "")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_execute_calculate_tool_single_quotes(self):
        """Test calculate tool with single-quoted expression."""
        result = await _execute_vision_tool("calculate(expression='10 * 2')", "")
        assert result == "20"

    @pytest.mark.asyncio
    async def test_execute_calculate_tool_parse_error(self):
        """Test calculate tool with unparseable expression."""
        result = await _execute_vision_tool("calculate(no_expression_arg)", "")
        assert result.startswith("[ERROR: Could not parse")

    @pytest.mark.asyncio
    async def test_execute_calculate_tool_eval_error(self):
        """Test calculate tool with invalid expression."""
        result = await _execute_vision_tool('calculate(expression="1 / 0")', "")
        assert result.startswith("[Calculate error:")

    @pytest.mark.asyncio
    async def test_execute_get_current_date_tool(self):
        """Test get_current_date tool."""
        result = await _execute_vision_tool("get_current_date()", "")
        # Should return YYYY-MM-DD (Weekday) format
        assert len(result) > 10
        assert "(" in result and ")" in result

    @pytest.mark.asyncio
    async def test_execute_get_current_time_tool(self):
        """Test get_current_time tool."""
        result = await _execute_vision_tool("get_current_time()", "")
        # Should return ISO format timestamp
        assert "T" in result  # ISO format contains T separator

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test execution of unknown tool."""
        result = await _execute_vision_tool("unknown_tool(arg1='value')", "")
        assert result.startswith("[Tool 'unknown_tool' not available")
        assert "Available:" in result

    @pytest.mark.asyncio
    async def test_execute_unparseable_action(self):
        """Test execution with unparseable action string."""
        result = await _execute_vision_tool("not a valid action format", "")
        assert result.startswith("[ERROR: Could not parse action:")


class TestVisionReActMode:
    """Test vision ReAct loop execution."""

    @pytest.mark.asyncio
    async def test_vision_react_direct_answer(self):
        """Test ReAct mode with immediate final answer."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Thought: I can answer directly\nFinal Answer: The image shows a blue sky."
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, tools_used, tools_called = await _vision_react_mode_answer(
                prompt="What is in the image?",
                image_b64="fake_base64",
                mime_type="image/png",
            )

            assert answer == "The image shows a blue sky."
            assert tools_used == 0
            assert tools_called == []

    @pytest.mark.asyncio
    async def test_vision_react_with_tool_call(self):
        """Test ReAct mode with tool call followed by final answer."""
        responses = [
            # First turn: request OCR
            {
                "choices": [
                    {
                        "message": {
                            "content": 'Thought: Need to extract text\nAction: ocr_extract(image_base64="current")'
                        }
                    }
                ]
            },
            # Second turn: final answer after observation
            {
                "choices": [
                    {
                        "message": {
                            "content": "Thought: Got the text\nFinal Answer: The text says 'Hello World'"
                        }
                    }
                ]
            },
        ]

        mock_ocr_response = MagicMock()
        mock_ocr_response.status_code = 200
        mock_ocr_response.json.return_value = {"text": "Hello World"}

        call_count = 0

        async def mock_post(url, json=None, files=None):
            nonlocal call_count
            if "ocr" in url:
                return mock_ocr_response
            else:
                response = MagicMock()
                response.status_code = 200
                response.json.return_value = responses[call_count]
                call_count += 1
                return response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = mock_post
            mock_client_class.return_value = mock_client

            answer, tools_used, tools_called = await _vision_react_mode_answer(
                prompt="What text is in the image?",
                image_b64="fake_base64",
                mime_type="image/png",
            )

            assert "Hello World" in answer
            assert tools_used == 1
            assert "ocr_extract" in tools_called

    @pytest.mark.asyncio
    async def test_vision_react_max_turns_exhausted(self):
        """Test ReAct mode hitting max turns limit."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Thought: Still thinking\nAction: calculate(expression='1+1')"
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, tools_used, tools_called = await _vision_react_mode_answer(
                prompt="Test question",
                image_b64="fake_base64",
                mime_type="image/png",
                max_turns=1,
            )

            # Should synthesize answer after max turns
            assert tools_used >= 1

    @pytest.mark.asyncio
    async def test_vision_react_http_error(self):
        """Test ReAct mode with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, tools_used, tools_called = await _vision_react_mode_answer(
                prompt="Test question",
                image_b64="fake_base64",
                mime_type="image/png",
            )

            assert answer == "[Vision ReAct: no answer produced]"
            assert tools_used == 0

    @pytest.mark.asyncio
    async def test_vision_react_no_action_no_final_answer(self):
        """Test ReAct mode with response that has neither action nor final answer."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Thought: This is just a thought without action or final answer"
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, tools_used, tools_called = await _vision_react_mode_answer(
                prompt="Test question",
                image_b64="fake_base64",
                mime_type="image/png",
            )

            # Should treat response as answer and strip "Thought:" prefix
            assert "This is just a thought" in answer
            assert tools_used == 0


class TestHandleMultiFileVision:
    """Test multi-file vision handling."""

    @pytest.mark.asyncio
    async def test_handle_multi_file_vision_no_files(self):
        """Test multi-file handler with no files."""
        from src.api.models import ChatRequest

        request = ChatRequest(prompt="Test", files=[])
        primitives = Mock()
        state = Mock()

        with pytest.raises(RuntimeError, match="No files provided"):
            await _handle_multi_file_vision(request, primitives, state, "task123")

    @pytest.mark.asyncio
    async def test_handle_multi_file_vision_with_archive(self):
        """Test multi-file handler with archive extraction."""
        from pathlib import Path
        from src.api.models import ChatRequest

        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.services.archive_extractor.ArchiveExtractor") as mock_extractor_class:
                mock_extractor = Mock()
                mock_manifest = Mock()
                mock_manifest.total_files = 5
                mock_manifest.total_size_bytes = 50000
                mock_extractor.list_contents.return_value = mock_manifest

                mock_result = Mock()
                mock_result.extracted_paths = [Path("/tmp/file1.txt"), Path("/tmp/file2.txt")]
                mock_extractor.extract_all.return_value = mock_result
                mock_extractor_class.return_value = mock_extractor

                # Mock file reading
                with patch("pathlib.Path.read_text", return_value="Test content"):
                    request = ChatRequest(prompt="Summarize", files=["/path/to/archive.zip"])
                    primitives = Mock()
                    primitives.llm_call.return_value = "Summary of files"
                    state = Mock()

                    result = await _handle_multi_file_vision(request, primitives, state, "task123")

                    assert "Summary of files" in result
                    mock_extractor.extract_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_multi_file_vision_with_images(self):
        """Test multi-file handler with image files."""
        from src.api.models import ChatRequest

        mock_ocr_result = Mock()
        mock_ocr_result.text = "Extracted text from image"

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b"fake_image_data"):
                with patch("src.services.document_client.get_document_client") as mock_client_fn:
                    mock_client = AsyncMock()
                    mock_client.ocr_image.return_value = mock_ocr_result
                    mock_client_fn.return_value = mock_client

                    request = ChatRequest(prompt="Analyze", files=["/path/to/image.png"])
                    primitives = Mock()
                    primitives.llm_call.return_value = "Analysis result"
                    state = Mock()

                    result = await _handle_multi_file_vision(request, primitives, state, "task123")

                    assert "Analysis result" in result
                    mock_client.ocr_image.assert_called_once()


class TestHandleVisionRequest:
    """Test main vision request handler."""

    @pytest.mark.asyncio
    async def test_handle_vision_request_no_image(self):
        """Test vision request with no image data."""
        from src.api.models import ChatRequest

        request = ChatRequest(prompt="Test")
        primitives = Mock()
        state = Mock()

        with pytest.raises(RuntimeError, match="No image data provided"):
            await _handle_vision_request(request, primitives, state, "task123")

    @pytest.mark.asyncio
    async def test_handle_vision_request_image_too_large(self):
        """Test vision request with oversized image."""
        from src.api.models import ChatRequest

        # Create a base64 string larger than 50MB
        large_data = "A" * (51 * 1024 * 1024)
        request = ChatRequest(prompt="Test", image_base64=large_data)
        primitives = Mock()
        state = Mock()

        with pytest.raises(RuntimeError, match="Image too large"):
            await _handle_vision_request(request, primitives, state, "task123")

    @pytest.mark.asyncio
    async def test_handle_vision_request_success(self):
        """Test successful vision request."""
        from src.api.models import ChatRequest
        from PIL import Image

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        request = ChatRequest(prompt="What color is this?", image_base64=img_b64)
        primitives = Mock()
        state = Mock()

        mock_ocr_result = Mock()
        mock_ocr_result.text = ""

        mock_vl_response = MagicMock()
        mock_vl_response.status_code = 200
        mock_vl_response.json.return_value = {
            "choices": [{"message": {"content": "The image is blue"}}]
        }

        with patch("src.services.document_client.get_document_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.ocr_image.return_value = mock_ocr_result
            mock_client_fn.return_value = mock_client

            with patch("httpx.AsyncClient") as mock_http_class:
                mock_http = AsyncMock()
                mock_http.__aenter__.return_value = mock_http
                mock_http.post.return_value = mock_vl_response
                mock_http_class.return_value = mock_http

                result = await _handle_vision_request(request, primitives, state, "task123")

                assert result == "The image is blue"

    @pytest.mark.asyncio
    async def test_handle_vision_request_all_servers_fail(self):
        """Test vision request when all VL servers fail."""
        from src.api.models import ChatRequest
        from PIL import Image

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        request = ChatRequest(prompt="Test", image_base64=img_b64)
        primitives = Mock()
        state = Mock()

        mock_ocr_result = Mock()
        mock_ocr_result.text = ""

        mock_vl_response = MagicMock()
        mock_vl_response.status_code = 500
        mock_vl_response.text = "Server error"

        with patch("src.services.document_client.get_document_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.ocr_image.return_value = mock_ocr_result
            mock_client_fn.return_value = mock_client

            with patch("httpx.AsyncClient") as mock_http_class:
                mock_http = AsyncMock()
                mock_http.__aenter__.return_value = mock_http
                mock_http.post.return_value = mock_vl_response
                mock_http_class.return_value = mock_http

                with pytest.raises(RuntimeError, match="All vision paths failed"):
                    await _handle_vision_request(request, primitives, state, "task123")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("backend", "expect_legacy"),
        [("server", False), ("auto", True), ("cli", True), ("malformed", True), (None, True)],
    )
    async def test_server_only_refuses_without_legacy_vision_request(
        self, monkeypatch, backend, expect_legacy
    ):
        """Only explicit server mode skips the redundant legacy endpoint."""
        import asyncio
        from types import SimpleNamespace
        from src.api.models import ChatRequest

        if backend is None:
            monkeypatch.delenv("ORCHESTRATOR_VISION_VL_BACKEND", raising=False)
        else:
            monkeypatch.setenv("ORCHESTRATOR_VISION_VL_BACKEND", backend)

        request = ChatRequest(
            prompt="What is in this image?",
            image_base64=base64.b64encode(b"synthetic image bytes").decode("ascii"),
        )
        posts = []
        server_failure = MagicMock(status_code=503, text="unavailable")
        legacy_success = MagicMock(status_code=200)
        legacy_success.json.return_value = {"description": "legacy result"}

        async def post(url, **kwargs):
            posts.append(url)
            if url.endswith("/vision/analyze") and backend == "server":
                await asyncio.sleep(1)
            return legacy_success if url.endswith("/vision/analyze") else server_failure

        http_client = AsyncMock()
        http_client.__aenter__.return_value = http_client
        http_client.post.side_effect = post

        with (
            patch("src.services.document_client.get_document_client") as get_document_client,
            patch("httpx.AsyncClient", return_value=http_client),
            patch(
                "src.api.routes.chat_vision._get_config",
                return_value=SimpleNamespace(
                    timeouts=SimpleNamespace(vision_inference=1),
                    server_urls=SimpleNamespace(api_url="http://api"),
                ),
            ),
            patch(
                "src.api.routes.chat_vision._vl_url_for_role",
                side_effect=lambda role: f"http://{role}",
            ),
        ):
            get_document_client.return_value.ocr_image = AsyncMock(
                return_value=SimpleNamespace(text="")
            )
            if expect_legacy:
                assert await _handle_vision_request(request, Mock(), Mock(), "task") == "legacy result"
            else:
                with pytest.raises(RuntimeError, match="All vision paths failed"):
                    await asyncio.wait_for(
                        _handle_vision_request(request, Mock(), Mock(), "task"),
                        timeout=0.25,
                    )

        assert sum(url.endswith("/vision/analyze") for url in posts) == int(expect_legacy)
        assert len(posts) == 2 + int(expect_legacy)
