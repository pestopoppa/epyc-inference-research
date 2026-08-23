#!/usr/bin/env python3
"""Delegation shim — research-repo ``debug_scorer`` resolves to the orchestrator B7 scorer.

This file WAS a fully pre-B7 fork of the orchestrator's ``debug_scorer.py``
(10/10 defect classes, proven in epyc-root
``handoffs/active/scorer-fork-drift-audit-2026-07-22.md`` — residual row
"Research-repo ``debug_scorer.py`` is fully pre-B7 (10/10 defect classes, off
routing path) — port B7 or stamp research benchmarks scored with it as
pre-B7-scorer era", line ~306, ticked 2026-08-23).

It is now a thin delegation shim: every import of ``debug_scorer`` (bare, or as
``benchmark.debug_scorer`` from ``scripts/``) loads the orchestrator's
B7-hardened copy by absolute path via ``importlib`` and re-exports its public
API, so every research consumer inherits B7 semantics with ZERO scoring drift:
SCORE-03 final-answer-region, SCORE-06 boundary-anchored substring, SCORE-16
nested ``\boxed{}``, SCORE-21 vacuous-oracle rejection, SCORE-23 ``str()``-wrap,
SCORE-24 multiset F1 + capture-group guard, multiple-choice textual labels,
llm_judge fail-closed, ``math_verify``, SCORE-25 unknown-verifier rejection.

This mirrors the A2 delegation pattern proven in epyc-orchestrator's
``seeding_scoring.py`` (sys.modules-pinned same-directory importlib load) —
here the path is absolute because this file does NOT live beside the
orchestrator copy.

Fail-closed: if the orchestrator copy cannot be located, this module raises
``ImportError`` at import time — it never silently falls back to a local
pre-B7 implementation.

Usage:
    from debug_scorer import score_answer

    result = score_answer(
        answer="The answer is 42",
        expected="42",
        scoring_method="exact_match",
        scoring_config={"extract_pattern": r"#### (\\d+)"},
    )
    print(result)  # True/False
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ORCH_SCORER_KEY = "epyc_orch_debug_scorer_b7"
_ORCH_SCORER_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/debug_scorer.py"
)


def _load_orchestrator_scorer():
    """Load the orchestrator's B7 ``debug_scorer`` by absolute path.

    Mirrors ``seeding_scoring._load_orchestrator_debug_scorer``: the module is
    registered under a private ``sys.modules`` key before exec (so re-entrant
    imports resolve to the same object) and cached after first load. Raises
    ``ImportError`` — never a silent fallback — if the orchestrator copy
    cannot be located.
    """
    cached = sys.modules.get(_ORCH_SCORER_KEY)
    if cached is not None:
        return cached

    scorer_path = Path(_ORCH_SCORER_PATH)
    if not scorer_path.is_file():
        raise ImportError(
            f"orchestrator B7 debug_scorer not found at {scorer_path} — cannot "
            "delegate; refusing to fall back to pre-B7 scoring semantics"
        )
    spec = importlib.util.spec_from_file_location(_ORCH_SCORER_KEY, scorer_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load orchestrator debug_scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ORCH_SCORER_KEY] = module
    spec.loader.exec_module(module)
    return module


_orb = _load_orchestrator_scorer()
score_answer = _orb.score_answer
_extract_code_block = _orb._extract_code_block

__all__ = ["score_answer", "_extract_code_block"]


def __getattr__(name):
    """Resolve any other name against the orchestrator copy (future-proof).

    Consumers today import only ``score_answer`` and ``_extract_code_block``,
    but a future consumer may reach for ``ScoringUnavailableError``,
    ``score_batch``, ``_contains_text_unit``, etc. — resolve those against the
    loaded orchestrator module rather than adding per-name re-exports.
    """
    scorer = _load_orchestrator_scorer()
    if hasattr(scorer, name):
        return getattr(scorer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
