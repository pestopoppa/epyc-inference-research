"""Shared test helpers. No inference. Real token ids come from the Qwen3.8-27B tokenizer.json (the
vocabulary behind :8083 /tokenize); tests that need it skip when it or `tokenizers` is unavailable.
Override the path with COHERENCE_GATE_TEST_TOKENIZER."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
FIX = TESTS / "fixtures"
sys.path.insert(0, str(TESTS.parents[1]))  # scripts/lib -> `import coherence_gate`

TOKENIZER_PATH = os.environ.get("COHERENCE_GATE_TEST_TOKENIZER",
                                "/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw/tokenizer.json")
_VENV_SITE = "/mnt/raid0/llm/epyc-inference-research/.venv/lib/python3.13/site-packages"


def _load_tokenizer():
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        if os.path.isdir(_VENV_SITE):
            sys.path.append(_VENV_SITE)
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    if not os.path.exists(TOKENIZER_PATH):
        return None
    return Tokenizer.from_file(TOKENIZER_PATH)


TOK = _load_tokenizer()
needs_tokenizer = pytest.mark.skipif(TOK is None, reason=f"tokenizer unavailable ({TOKENIZER_PATH})")

PROSE = (FIX / "prose_excerpt.md").read_text()
TRACES = json.loads((FIX / "reasoning_traces.json").read_text())["traces"]
STORED = json.loads((FIX / "q38t7_stored_stats.json").read_text())["rows"]
V1_GOLDEN = json.loads((FIX / "v1_golden.json").read_text())["golden"]


def enc(text: str) -> list[int]:
    return TOK.encode(text, add_special_tokens=False).ids


def cut(text: str, n: int) -> tuple[str, list[int]]:
    ids = enc(text)[:n]
    return TOK.decode(ids), ids
