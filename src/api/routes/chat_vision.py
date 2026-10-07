"""Vision pipeline for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: OCR pre-processing, VL model routing, vision ReAct loop,
multi-file vision handling, and structured analysis.
"""

from __future__ import annotations

import ast as _ast
import logging
import os
import operator as _operator
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from src.config import get_config as _get_config
from src.exceptions import ArchiveExtractionError
from src.prompt_builders import (
    VISION_REACT_EXECUTABLE_TOOLS,
    VISION_TOOL_DESCRIPTIONS,
)
from src.api.routes.chat_utils import QWEN_STOP
from src.api.routes.vision_serving import (
    fallback_vl_port_for_role as _fallback_vl_port_for_role,
    stack_priors_available as _stack_priors_available,
    vision_roles as _vision_roles,
)
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_serving,
    stack_prior_serving_ports,
)

logger = logging.getLogger(__name__)

_DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[3] / "orchestration" / "derived" / "stack_priors.yaml"
)

if TYPE_CHECKING:
    from src.api.models import ChatRequest
    from src.api.state import AppState
    from src.llm_primitives import LLMPrimitives


def _needs_structured_analysis(prompt: str) -> bool:
    """Detect if a vision prompt needs full structured analysis beyond OCR.

    Structured analysis triggers VisionPipeline with VL_STRUCTURED analyzer
    for document forensics, diagram parsing, protocol architecture, etc.

    Args:
        prompt: The user's vision prompt.

    Returns:
        True if structured analysis should complement OCR.
    """
    from src.classifiers import needs_structured_analysis

    return needs_structured_analysis(prompt)


def _first_server_url(url: str) -> str:
    """Normalize a config URL value to one concrete HTTP endpoint."""
    if url.startswith("full:"):
        url = url[len("full:"):]
    return url.split(",")[0].rstrip("/")


def _stack_prior_vl_urls(
    stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH,
) -> dict[str, str]:
    """Return live VL role endpoints from generated stack priors."""
    roles = live_stack_role_records(stack_priors_path)
    if not roles:
        logger.warning("Using fallback VL URLs; no live stack-prior VL roles found")
        return {}

    urls: dict[str, str] = {}
    for role in _vision_roles(stack_priors_path):
        record = roles.get(role)
        if record is None:
            continue
        serving = stack_prior_serving(record)
        endpoint = serving.get("endpoint")
        if isinstance(endpoint, str) and endpoint:
            urls[role] = _first_server_url(endpoint)
            continue
        for port in stack_prior_serving_ports(serving):
            urls[role] = f"http://localhost:{port}"
            break
    return urls


def _fallback_vl_url_for_role(role: str) -> str:
    configured = getattr(_get_config().server_urls, role, "")
    if isinstance(configured, str) and configured:
        return _first_server_url(configured)
    return f"http://localhost:{_fallback_vl_port_for_role(role)}"


def _vl_url_for_role(
    role: str,
    stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH,
) -> str:
    urls = _stack_prior_vl_urls(stack_priors_path)
    url = urls.get(role)
    if isinstance(url, str):
        return url
    if not _stack_priors_available(stack_priors_path):
        return _fallback_vl_url_for_role(role)
    raise ValueError(
        f"No generated VL URL for role {role!r}; degraded fallback disabled "
        f"because live stack priors are available at {stack_priors_path}"
    )


def _vl_url_for_port(
    vl_port: int,
    stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH,
) -> str:
    for url in _stack_prior_vl_urls(stack_priors_path).values():
        if urlparse(url).port == vl_port:
            return url
    if _stack_priors_available(stack_priors_path):
        raise ValueError(
            f"No generated VL URL for port {vl_port}; degraded fallback disabled "
            f"because live stack priors are available at {stack_priors_path}"
        )
    for role in _vision_roles(stack_priors_path):
        if _fallback_vl_port_for_role(role) == vl_port:
            return _fallback_vl_url_for_role(role)
    return f"http://localhost:{vl_port}"


async def _handle_vision_request(
    request: "ChatRequest",
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
    force_server: str | None = None,
) -> str:
    """Route a vision request through OCR + VL pipeline.

    For OCR-heavy prompts (text extraction, sign reading):
    1. LightOnOCR-2-1B (port 9001) extracts text deterministically
    2. Extracted text + image + prompt sent to VL model for understanding

    For non-OCR prompts (chart analysis, scene understanding):
    1. Image + prompt sent directly to VL model

    Fallback chain for VL:
    1. VL worker (`worker_vision` from generated stack priors)
    2. VL escalation (`vision_escalation` from generated stack priors)
    3. Vision pipeline endpoint (legacy subprocess path)

    Args:
        request: Chat request with image_path or image_base64.
        primitives: LLMPrimitives (unused, for interface compat).
        state: Application state.
        task_id: Task ID for logging.
        force_server: Optional server constraint. "worker_vision" and
            "vision_escalation" use their generated stack-prior endpoints when
            available. None uses the default fallback chain.

    Returns:
        Answer string from the vision model.
    """
    import httpx
    import base64

    # Get image as base64
    image_b64 = request.image_base64
    if not image_b64 and request.image_path:
        from src.api.routes.path_validation import validate_api_path

        img_path = validate_api_path(request.image_path)
        if not img_path.exists():
            raise RuntimeError(f"Image not found: {request.image_path}")
        image_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    if not image_b64:
        raise RuntimeError("No image data provided (need image_path or image_base64)")

    # ── Always-on OCR pre-processing (~1s, deterministic, port 9001) ─────
    ocr_text = ""
    try:
        from src.services.document_client import get_document_client

        client = get_document_client()
        ocr_result = await client.ocr_image(
            image=image_b64,
            output_format="text",
        )
        ocr_text = ocr_result.text.strip() if ocr_result.text else ""
        if ocr_text:
            logger.info(f"OCR pre-processing extracted {len(ocr_text)} chars for VL")
    except Exception as e:
        logger.warning(f"OCR pre-processing failed (continuing without): {e}")
        ocr_text = ""

    # ── Base64 size guard (50MB max for images) ─────────────────────────
    MAX_IMG_B64_SIZE = 50 * 1024 * 1024  # 50MB
    if len(image_b64) > MAX_IMG_B64_SIZE:
        raise RuntimeError(f"Image too large ({len(image_b64) // 1024 // 1024}MB, max 50MB)")

    # ── Structured analysis for document/diagram prompts ───────────────
    structured_context = ""
    if _needs_structured_analysis(request.prompt):
        try:
            from src.vision.pipeline import VisionPipeline
            import io
            from PIL import Image

            # Decode base64 to PIL Image for VisionPipeline
            img_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(img_bytes))

            pipeline = VisionPipeline()
            result = pipeline.analyze(pil_image, analyzers=["vl_structured", "vl_ocr"])

            # Format structured findings
            parts = []
            if hasattr(result, "structured") and result.structured:
                parts.append(f"Structured elements:\n{result.structured}")
            if hasattr(result, "description") and result.description:
                parts.append(f"Visual description:\n{result.description}")
            structured_context = "\n\n".join(parts)
            if structured_context:
                logger.info(f"Structured analysis added {len(structured_context)} chars context")
        except ImportError:
            logger.debug("VisionPipeline not available, skipping structured analysis")
        except Exception as e:
            logger.warning(f"Structured analysis failed (continuing without): {e}")

    # Build VL prompt — inject OCR text and structured context
    vl_prompt = request.prompt
    context_parts = []
    if ocr_text:
        context_parts.append(f"OCR-extracted text from image:\n---\n{ocr_text}\n---")
    if structured_context:
        context_parts.append(f"Structured analysis:\n---\n{structured_context}\n---")
    if context_parts:
        context_block = "\n\n".join(context_parts)
        vl_prompt = (
            f"{context_block}\n\n"
            f"Using the above extracted information and the image, answer:\n{request.prompt}"
        )

    # Detect image MIME type from header bytes
    mime_type = "image/jpeg"  # default
    try:
        raw = base64.b64decode(image_b64[:32])
        if raw[:4] == b"\x89PNG":
            mime_type = "image/png"
        elif raw[:4] == b"RIFF":
            mime_type = "image/webp"
    except Exception as exc:
        logger.debug("MIME type detection from b64 header failed: %s", exc)

    # Build multimodal chat completion payload for llama-server
    vl_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": vl_prompt,
                    },
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
        "stop": [QWEN_STOP],
    }

    # Try VL servers — constrained if force_server is set
    if force_server == "worker_vision":
        vl_servers = [("worker_vision", _vl_url_for_role("worker_vision"))]
    elif force_server == "vision_escalation":
        vl_servers = [("vision_escalation", _vl_url_for_role("vision_escalation"))]
    else:
        vl_servers = [
            ("worker_vision", _vl_url_for_role("worker_vision")),
            ("vision_escalation", _vl_url_for_role("vision_escalation")),
        ]

    last_error = None
    for server_name, server_url in vl_servers:
        try:
            async with httpx.AsyncClient(timeout=_get_config().timeouts.vision_inference) as client:
                resp = await client.post(
                    f"{server_url}/v1/chat/completions",
                    json=vl_payload,
                )

            if resp.status_code == 200:
                data = resp.json()
                # Extract answer from chat completions format
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        logger.info(f"VL answer from {server_name} ({len(content)} chars)")
                        return content.strip()

                # Try legacy completion format
                content = data.get("content", "")
                if content:
                    return content.strip()

                last_error = f"{server_name}: empty response"
                logger.warning(f"VL {server_name}: got 200 but empty content")
            else:
                last_error = f"{server_name}: HTTP {resp.status_code}"
                logger.warning(f"VL {server_name}: HTTP {resp.status_code}: {resp.text[:200]}")

        except httpx.ConnectError:
            last_error = f"{server_name}: connection refused (server not running?)"
            logger.warning(f"VL {server_name}: connection refused at {server_url}")
        except Exception as e:
            last_error = f"{server_name}: {type(e).__name__}: {e}"
            logger.warning(f"VL {server_name} failed: {type(e).__name__}: {e}")

    # A server-only deployment has no permitted CLI/legacy fallback. The
    # legacy endpoint can otherwise spend another connect timeout probing the
    # same unavailable server before refusing the request.
    if os.environ.get("ORCHESTRATOR_VISION_VL_BACKEND", "auto").lower() == "server":
        raise RuntimeError(f"All vision paths failed. Last error: {last_error}")

    # All VL servers failed — try legacy vision pipeline as last resort
    logger.warning(f"All VL servers failed ({last_error}), trying vision pipeline")
    try:
        async with httpx.AsyncClient(timeout=_get_config().timeouts.vision_inference) as client:
            legacy_payload = {
                "image_path": request.image_path,
                "image_base64": request.image_base64,
                "analyzers": ["vl_describe"],
                "vl_prompt": request.prompt,
                "store_results": False,
            }
            resp = await client.post(
                f"{_get_config().server_urls.api_url.rstrip('/')}/vision/analyze",
                json=legacy_payload,
            )
        if resp.status_code == 200:
            data = resp.json()
            # AnalyzeResult has .description at top level
            description = data.get("description", "")
            if description:
                return description
    except Exception as e:
        logger.warning(f"Legacy vision pipeline also failed: {e}")

    raise RuntimeError(f"All vision paths failed. Last error: {last_error}")


_SAFE_OPS = {
    _ast.Add: _operator.add,
    _ast.Sub: _operator.sub,
    _ast.Mult: _operator.mul,
    _ast.Div: _operator.truediv,
    _ast.FloorDiv: _operator.floordiv,
    _ast.Mod: _operator.mod,
    _ast.Pow: _operator.pow,
    _ast.USub: _operator.neg,
}


def _safe_eval_math(expr: str) -> float | int:
    """Evaluate arithmetic expression safely — no function calls, imports, or attribute access."""
    tree = _ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, _ast.Expression):
            return _eval(node.body)
        if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, _ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, _ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    return _eval(tree)


async def _execute_vision_tool(
    action_str: str,
    image_b64: str,
) -> str:
    """Execute a tool invoked from the vision ReAct loop.

    Dispatches to:
    - ocr_extract → POST to LightOnOCR port 9001
    - calculate, get_current_date, etc. → standard Python eval

    Args:
        action_str: Parsed action string, e.g. 'ocr_extract(image_base64="...")'.
        image_b64: The base64 image data (injected for ocr_extract).

    Returns:
        Observation string from tool execution.
    """
    import re as _re

    tool_match = _re.match(r"(\w+)\((.*)\)$", action_str, _re.DOTALL)
    if not tool_match:
        return f"[ERROR: Could not parse action: {action_str}]"

    tool_name = tool_match.group(1)
    args_str = tool_match.group(2)

    if tool_name == "ocr_extract":
        # Route to LightOnOCR on port 9001
        import httpx

        try:
            _ocr_url = _get_config().server_urls.ocr_server
            async with httpx.AsyncClient(timeout=_get_config().timeouts.vision_figure) as client:
                resp = await client.post(
                    f"{_ocr_url}/v1/document/ocr",
                    json={"image_base64": image_b64},
                )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("text", data.get("full_text", ""))
                if text:
                    return text[:4000]  # Cap observation length
                return "[OCR returned empty text]"
            return f"[OCR error: HTTP {resp.status_code}]"
        except Exception as e:
            logger.warning(f"OCR tool execution failed: {e}")
            return f"[OCR error: {type(e).__name__}: {e}]"

    elif tool_name == "calculate":
        # Safe math evaluation (AST-walking, no arbitrary code execution)
        try:
            # Parse expression from args
            expr_match = _re.search(r'expression\s*=\s*"([^"]*)"', args_str)
            if not expr_match:
                expr_match = _re.search(r"expression\s*=\s*\'([^\']*)\'", args_str)
            if expr_match:
                result = _safe_eval_math(expr_match.group(1))
                return str(result)
            return "[ERROR: Could not parse calculate expression]"
        except Exception as e:
            return f"[Calculate error: {e}]"

    elif tool_name in ("get_current_date", "get_current_time"):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        if tool_name == "get_current_date":
            return now.strftime("%Y-%m-%d (%A)")
        return now.isoformat()

    else:
        available = ", ".join(sorted(VISION_REACT_EXECUTABLE_TOOLS))
        return f"[Tool '{tool_name}' not available in vision ReAct mode. Available: {available}]"


async def _vision_react_mode_answer(
    prompt: str,
    image_b64: str,
    mime_type: str,
    context: str = "",
    vl_port: int = 8086,
    max_turns: int = 2,
) -> tuple[str, int]:
    """Vision-aware ReAct loop using direct VL backend calls.

    Unlike the text-only REPL loop which uses primitives.llm_call(),
    this sends multimodal messages directly to VL backend via httpx with
    OpenAI /v1/chat/completions format.

    The image is included in the first user message only. Subsequent turns
    are text-only (tool observations). The VL model's context window carries
    the image forward across turns.

    Args:
        prompt: The user's question.
        image_b64: Base64-encoded image data.
        mime_type: Image MIME type (e.g. "image/png").
        context: Optional text context.
        vl_port: VL backend port (8086=worker_vision, 8087=vision_escalation).
        max_turns: Maximum Thought/Action/Observation cycles (capped at 2
            to stay within 300s overall timeout at 120s/turn).

    Returns:
        Tuple of (final_answer, tools_used_count).
    """
    import httpx

    tools_used = 0
    tools_called: list[str] = []

    # Build tool descriptions from single source of truth
    tool_descriptions = [f"- {desc}" for desc in VISION_TOOL_DESCRIPTIONS.values()]

    react_system = f"""You have access to the following tools:
{chr(10).join(tool_descriptions)}

Use the following format:

Question: the input question you must answer
Thought: reason about what to do next
Action: tool_name(arg1="value1")
Observation: the result of the action
... (repeat Thought/Action/Observation as needed, up to {max_turns} times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Important rules:
- Always start with a Thought before an Action
- To use OCR on the image, call: Action: ocr_extract(image_base64="current")
- After each Observation, decide if you have enough info for a Final Answer
- If no tools are needed, skip directly to Final Answer
- Be concise in your Final Answer — answer the question directly"""

    # Build initial messages — image in first user message
    context_prefix = f"Context:\n{context}\n\n" if context else ""
    question_text = f"{react_system}\n\n{context_prefix}Question: {prompt}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": question_text,
                },
            ],
        }
    ]

    # Build VL URL from generated stack-prior serving records for known ports.
    _base = _vl_url_for_port(vl_port)
    vl_url = f"{_base}/v1/chat/completions"

    for turn in range(max_turns):
        # Call VL backend
        try:
            async with httpx.AsyncClient(timeout=_get_config().timeouts.vision_inference) as client:
                resp = await client.post(
                    vl_url,
                    json={
                        "messages": messages,
                        "max_tokens": 2048,
                        "temperature": 0.0,
                        "stop": [QWEN_STOP],
                    },
                )

            if resp.status_code != 200:
                logger.warning(f"Vision ReAct turn {turn}: HTTP {resp.status_code}")
                break

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"Vision ReAct turn {turn}: no choices in response")
                break

            response_text = choices[0].get("message", {}).get("content", "").strip()
            if not response_text:
                logger.warning(f"Vision ReAct turn {turn}: empty content")
                break

        except Exception as e:
            logger.warning(f"Vision ReAct turn {turn} failed: {e}")
            break

        # Append assistant response
        messages.append({"role": "assistant", "content": response_text})

        # Check for Final Answer
        if "Final Answer:" in response_text:
            idx = response_text.index("Final Answer:")
            answer = response_text[idx + len("Final Answer:") :].strip()
            logger.info(f"Vision ReAct completed in {turn + 1} turns, {tools_used} tools")
            return answer, tools_used, tools_called

        # Parse Action line
        action_match = None
        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("Action:"):
                action_match = line[len("Action:") :].strip()
                break

        if not action_match:
            # No action and no final answer — treat response as answer
            logger.info(f"Vision ReAct turn {turn}: no Action, treating as answer")
            # Strip Thought: prefix
            lines = response_text.split("\n")
            answer_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("Thought:"):
                    stripped = stripped[len("Thought:") :].strip()
                answer_lines.append(stripped)
            return "\n".join(answer_lines).strip(), tools_used, tools_called

        # Execute the tool — extract name for tracking
        import re as _re_vt

        _vt_match = _re_vt.match(r"(\w+)\(", action_match)
        _vt_name = _vt_match.group(1) if _vt_match else action_match
        tools_used += 1
        tools_called.append(_vt_name)
        observation = await _execute_vision_tool(action_match, image_b64)
        logger.info(f"Vision ReAct turn {turn}: {action_match[:50]} → {len(observation)} chars")

        # Append observation as user message (text-only — no image re-send)
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}",
            }
        )

    # Max turns exhausted — synthesize from conversation
    logger.warning(f"Vision ReAct exhausted {max_turns} turns")
    # Ask for final answer
    messages.append(
        {
            "role": "user",
            "content": "You have used all available turns. Please provide your Final Answer now.",
        }
    )

    try:
        async with httpx.AsyncClient(timeout=_get_config().timeouts.vision_inference) as client:
            resp = await client.post(
                vl_url,
                json={
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.0,
                    "stop": [QWEN_STOP],
                },
            )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                if "Final Answer:" in content:
                    idx = content.index("Final Answer:")
                    return content[idx + len("Final Answer:") :].strip(), tools_used, tools_called
                return content, tools_used, tools_called
    except Exception as e:
        logger.warning(f"Vision ReAct final synthesis failed: {e}")

    return "[Vision ReAct: no answer produced]", tools_used, tools_called


async def _handle_multi_file_vision(
    request: "ChatRequest",
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> str:
    """Handle vision requests with multiple files or archives.

    For archives (zip, tar, etc.): Extract → process each file → aggregate.
    For multiple files: Process each → aggregate via MultiDocumentResult.

    Uses:
    - ArchiveExtractor for ZIP/TAR/7Z extraction with security constraints
    - DocumentFormalizerClient for multi-page PDF OCR
    - BatchProcessor for parallel image processing
    - Two-stage summarization for synthesis

    Args:
        request: Chat request with files field.
        primitives: LLMPrimitives for synthesis.
        state: Application state.
        task_id: Task ID for logging.

    Returns:
        Synthesized answer from all files.
    """
    from pathlib import Path
    from src.api.routes.chat_summarization import _run_two_stage_summarization

    files = request.files or []

    if not files:
        raise RuntimeError("No files provided")

    # Check for archives
    archive_extensions = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".7z"}
    extracted_files = []

    for fpath in files:
        p = Path(fpath)
        suffix = "".join(p.suffixes).lower()
        if suffix in archive_extensions or p.suffix.lower() in archive_extensions:
            try:
                from src.services.archive_extractor import ArchiveExtractor

                extractor = ArchiveExtractor()
                manifest = extractor.list_contents(fpath)
                logger.info(
                    f"Archive {p.name}: {manifest.total_files} files, "
                    f"{manifest.total_size_bytes / 1024:.1f}KB"
                )
                # Extract all (respects security limits: 500MB max, 1000 files max)
                result = extractor.extract_all(fpath)
                extracted_files.extend(str(ef) for ef in result.extracted_paths)
            except (ArchiveExtractionError, OSError, ValueError) as e:
                logger.warning(f"Archive extraction failed for {fpath}: {e}")
                extracted_files.append(fpath)  # Try to process as-is
        else:
            extracted_files.append(fpath)

    if not extracted_files:
        raise RuntimeError("No files to process after extraction")

    # Process each file — OCR for documents, VisionPipeline for images
    file_results = []
    for fpath in extracted_files:
        p = Path(fpath)
        try:
            if p.suffix.lower() == ".pdf":
                # Multi-page PDF via DocumentFormalizerClient
                from src.services.document_client import get_document_client

                client = get_document_client()
                ocr_result = await client.ocr_pdf_with_partial_success(fpath)
                file_results.append(f"[File: {p.name}]\n{ocr_result.text or '[OCR failed]'}")
            elif p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
                # Single image — OCR + VL
                import base64

                img_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                from src.services.document_client import get_document_client

                client = get_document_client()
                ocr_result = await client.ocr_image(image=img_b64, output_format="text")
                file_results.append(f"[File: {p.name}]\n{ocr_result.text or '[No text extracted]'}")
            else:
                # Text files
                content = p.read_text(errors="replace")[:10000]
                file_results.append(f"[File: {p.name}]\n{content}")
        except Exception as e:
            file_results.append(f"[File: {p.name}] Error: {e}")

    # Synthesize: two-stage if many files, direct if few
    all_content = "\n\n".join(file_results)
    if len(all_content) > 20000:
        answer, _stats = await _run_two_stage_summarization(
            request.prompt,
            all_content,
            primitives,
            state,
            task_id,
        )
    else:
        synthesis_prompt = f"{all_content}\n\nBased on the files above, answer:\n{request.prompt}"
        answer = primitives.llm_call(
            synthesis_prompt,
            role="frontdoor",
            n_tokens=2048,
            skip_suffix=True,
        )

    return answer.strip()
