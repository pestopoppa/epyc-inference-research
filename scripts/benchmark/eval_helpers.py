#!/usr/bin/env python3
"""Shared helpers for Package C eval scripts.

Provides session trace parsing, model API calls, Claude-as-Judge scoring,
token estimation, and identifier extraction.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ── Session trace parsing ────────────────────────────────────────


@dataclass
class TurnBlock:
    """Structured representation of one turn from a session trace."""
    turn_num: int
    role: str
    outcome: str
    timestamp: str
    code_hash: str
    code_lines: int
    first_line: str
    output: str
    error: str
    nudge: str
    raw_text: str


_TURN_HEADER_RE = re.compile(
    r'^### Turn (\d+) — (\S+) \[(\w+)\] \((\d{2}:\d{2}:\d{2})\)',
)
_CODE_RE = re.compile(
    r'^- Code: `([0-9a-f]+)` \((\d+) lines?\) first=`(.*?)`',
)


def parse_session_trace(path: Path) -> list[TurnBlock]:
    """Parse a session trace markdown file into structured turn blocks."""
    content = path.read_text(errors="replace")

    # Split on turn headers, keeping the header in each chunk
    raw_blocks = re.split(r'(?=^### Turn \d+)', content, flags=re.MULTILINE)

    turns = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        header = _TURN_HEADER_RE.match(block)
        if not header:
            continue

        turn_num = int(header.group(1))
        role = header.group(2)
        outcome = header.group(3)
        timestamp = header.group(4)

        # Parse sub-fields from lines
        code_hash = ""
        code_lines = 0
        first_line = ""
        output_parts = []
        error_parts = []
        nudge_parts = []

        lines = block.split("\n")[1:]  # skip header line
        current_field = None

        for line in lines:
            if line.startswith("- Code:"):
                cm = _CODE_RE.match(line)
                if cm:
                    code_hash = cm.group(1)
                    code_lines = int(cm.group(2))
                    first_line = cm.group(3)
                current_field = None
            elif line.startswith("- Output:"):
                output_parts.append(line[len("- Output:"):].strip())
                current_field = "output"
            elif line.startswith("- Error:"):
                error_parts.append(line[len("- Error:"):].strip())
                current_field = "error"
            elif line.startswith("- Nudge:"):
                nudge_parts.append(line[len("- Nudge:"):].strip())
                current_field = "nudge"
            elif line.startswith("- "):
                current_field = None
            elif current_field and line.strip():
                # Continuation line for multi-line output/error
                if current_field == "output":
                    output_parts.append(line)
                elif current_field == "error":
                    error_parts.append(line)
                elif current_field == "nudge":
                    nudge_parts.append(line)

        turns.append(TurnBlock(
            turn_num=turn_num,
            role=role,
            outcome=outcome,
            timestamp=timestamp,
            code_hash=code_hash,
            code_lines=code_lines,
            first_line=first_line,
            output="\n".join(output_parts),
            error="\n".join(error_parts),
            nudge="\n".join(nudge_parts),
            raw_text=block,
        ))

    return turns


# ── Model API calls ──────────────────────────────────────────────


def call_model(
    port: int,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    host: str = "localhost",
) -> tuple[str, dict]:
    """Call OpenAI-compatible chat completions API.

    Returns (response_text, usage_dict).
    """
    import httpx

    t0 = time.monotonic()
    resp = httpx.post(
        f"http://{host}:{port}/v1/chat/completions",
        json={
            "model": "auto",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    elapsed = time.monotonic() - t0

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    usage["elapsed_s"] = round(elapsed, 2)
    return text, usage


def judge_quality(
    original: str,
    summary: str,
    probe: str,
    port: int,
    *,
    host: str = "localhost",
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Score consolidation quality using a local LLM as judge.

    Returns {"faithfulness": 0-3, "retention": 0-3, "reason": str}.
    """
    system = (
        "Rate this consolidation on two dimensions:\n"
        "- faithfulness (0-3): Does the summary accurately represent the original? "
        "0=hallucinated, 1=major omissions, 2=minor omissions, 3=faithful\n"
        "- retention (0-3): Can the probe question be answered from the summary alone? "
        "0=impossible, 1=partially, 2=mostly, 3=fully\n\n"
        'Respond ONLY with JSON: {"faithfulness": N, "retention": N, "reason": "..."}'
    )
    user = (
        f"## Original\n{original[:3000]}\n\n"
        f"## Consolidation\n{summary[:2000]}\n\n"
        f"## Probe\n{probe[:500]}"
    )

    try:
        text, _ = call_model(
            port=port,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            host=host,
            timeout=timeout,
        )
        # Parse JSON — handle markdown code fence wrapping
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
        return json.loads(cleaned)
    except Exception as e:
        log.warning("Judge scoring failed: %s", e)
        return {"faithfulness": -1, "retention": -1, "reason": f"error: {e}"}


# ── Token estimation ─────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token heuristic)."""
    return len(text) // 4


# ── Identifier extraction ────────────────────────────────────────


_ID_PATTERNS = [
    re.compile(r'[\w./]+\.\w{1,4}'),                         # file paths
    re.compile(r'(?:def|class)\s+(\w+)'),                     # function/class names
    re.compile(r'`([0-9a-f]{8,})`'),                          # code hashes
    re.compile(r'((?:Name|Type|Value|Key|Attribute|Import)Error)'),  # error types
    re.compile(r'(?:FINAL|CALL)\s*\(\s*["\']?(\w+)'),        # tool calls
]


def extract_identifiers(text: str) -> set[str]:
    """Extract identifiers from text: file paths, function names, error codes, hashes."""
    ids: set[str] = set()
    for pattern in _ID_PATTERNS:
        for m in pattern.finditer(text):
            # Use first capturing group if present, else full match
            val = m.group(1) if m.lastindex else m.group(0)
            if val and len(val) > 2:
                ids.add(val.lower())
    return ids
