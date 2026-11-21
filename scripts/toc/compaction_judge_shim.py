#!/usr/bin/env python3
"""llama-server judge shim for compaction validation (TOC-CMP-JUDGE-SHIM).

Port of the Slipstream V2-Fast speculative-summary judge (alignment + non-contradiction, overall =
min of the two, one integer 0-10) onto a llama-server /v1/chat/completions endpoint:
  * single-integer output is forced with a GBNF grammar (replaces vLLM `guided_choice`),
  * thinking is disabled with chat_template_kwargs.enable_thinking=false (and asserted on the request),
  * temperature 0, max_tokens 8, cache_prompt false (a judge call must not touch served slots' caches).

Provenance / licence: the prompt text and the parse/format helpers are derived from
chenzhuofu/slipstream @ 05d48f4ddf4fbab146f95b49f802db26e303e6f4, `src/minisweagent/agents/judge.py`
(V2-Fast), which inherits the MIT licence of mini-swe-agent: Copyright (c) 2025 Kilian A. Lieret and
Carlos E. Jimenez. The MIT permission notice is reproduced in LICENSE-slipstream-MIT.txt beside this file.

Backends: `judge_one` takes any callable `post(url, payload, timeout) -> dict`; the default posts to a
llama-server over urllib. A different judge (for example an in-context Claude judge) can reuse
`build_prompt`, `parse_score` and `decide` unchanged and only replace the transport.
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from typing import Callable, Optional

SCORE_GRAMMAR = 'root ::= "10" | [0-9]\n'
_THINKING_RE = re.compile(r"<(?:seed:)?think>.*?</(?:seed:)?think>", re.DOTALL)

PROMPT_V2_FAST = """You are judging whether a set of speculative agent actions are consistent with a summary handover.

The summary describes work already done in a prior session and what remains pending. The speculative actions ran in parallel with the summary's generation — they may or may not align with the plan.

Evaluate on TWO criteria:

1. ALIGNMENT (0-10): Do the speculative actions advance PENDING items from the summary without wandering into unrelated areas?
   - 10: every action directly targets a PENDING item
   - 7-9: actions clearly advance PENDING work, minor side-exploration allowed
   - 4-6: loose relation to PENDING; mostly-neutral exploration
   - 1-3: actions wander into areas the summary did not flag as pending
   - 0: actions pursue something totally unrelated

2. NON-CONTRADICTION (0-10): Do the speculative actions avoid undoing, redoing, or contradicting COMPLETED / CHANGES items?
   - 10: zero contradictions; nothing re-runs work already listed as done
   - 7-9: possibly minor redundancy with completed items
   - 4-6: repeats some completed exploration but doesn't destroy progress
   - 1-3: contradicts or re-opens completed work
   - 0: destroys, reverts, or clearly re-does something already marked DONE

Overall score = min(alignment, non_contradiction).

Important context for scoring:
  * Read-only exploration (grep/find/cat/head/ls/python -c probe) NEVER destroys
    state. If CHANGES is empty / no file writes happened, NON-CONTRADICTION
    should be >= 8 — read-only work cannot contradict completed changes.
  * If PENDING is broadly worded (e.g. "find where X is processed"), any search
    that narrows in on X counts as alignment — not a tangent.
  * REPAIR OF BROKEN CHANGES is alignment, not contradiction. If the summary's
    CHANGES / CURRENT_STATE explicitly flags the completed edits as "broken",
    "corrupted", "syntax error", "indentation error", or "failed restoration",
    then spec actions that `git checkout` the file or re-edit it to fix the
    breakage are REPAIR, not reversion. Give NON-CONTRADICTION >= 8.
  * Multiple sed/edit commands against a file listed in CHANGES look like
    redoing, but if the commands target a bug in the agent's OWN recent edits
    (rather than destroying a completed good fix), that is alignment.

Respond with only a single integer between 0 and 10.

---
SUMMARY HANDOVER:
{summary_handover}
---
SPECULATIVE ACTIONS (in execution order):
{spec_actions}
---

Score (0-10):"""


def strip_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text).strip()


def format_spec_actions(spec_actions: list[dict], truncate_thinking: int = 1500) -> str:
    lines = []
    for i, a in enumerate(spec_actions, 1):
        lines.append(f"[{i}] $ {str(a.get('command', ''))[:500]}")
        thinking = str(a.get("thinking", "")).strip()
        if thinking:
            if len(thinking) > truncate_thinking:
                thinking = thinking[:truncate_thinking] + "…"
            lines.append(f"    Thinking: {thinking}")
    return "\n".join(lines) if lines else "(no speculative actions)"


def build_prompt(summary_handover: str, spec_actions: list[dict]) -> str:
    return PROMPT_V2_FAST.format(summary_handover=strip_thinking(summary_handover),
                                 spec_actions=format_spec_actions(spec_actions))


def build_payload(prompt: str, model: str = "judge") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8,
        "cache_prompt": False,
        "grammar": SCORE_GRAMMAR,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_score(content: str) -> Optional[int]:
    """Strict: the grammar guarantees exactly one integer 0-10; anything else is a parse failure (None)."""
    m = re.fullmatch(r"\s*(10|[0-9])\s*", content or "")
    return int(m.group(1)) if m else None


def decide(score: Optional[int], threshold: int) -> str:
    """accept / reject / unknown. A missing score is UNKNOWN, never an accept (fail-closed)."""
    if score is None:
        return "unknown"
    return "accept" if score >= threshold else "reject"


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (operator-supplied llama-server URL)
        return json.loads(r.read())


def judge_one(summary_handover: str, spec_actions: list[dict], base_url: str = "http://127.0.0.1:8080",
              threshold: int = 7, retries: int = 3, timeout: float = 120.0,
              post: Callable[[str, dict, float], dict] = _http_post) -> dict:
    payload = build_payload(build_prompt(summary_handover, spec_actions))
    assert payload["chat_template_kwargs"]["enable_thinking"] is False  # thinking-off is part of the contract
    err = ""
    for _ in range(retries):
        try:
            resp = post(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
            msg = resp["choices"][0]["message"]
            if msg.get("reasoning_content"):
                err = "reasoning_content present: thinking was not disabled"
                continue
            score = parse_score(msg.get("content") or "")
            if score is None:
                err = f"unparseable judge response: {msg.get('content')!r}"
                continue
            return {"score": score, "decision": decide(score, threshold), "error": "", "usage": resp.get("usage", {})}
        except Exception as e:  # transport errors retry, then fail closed
            err = str(e)
    return {"score": None, "decision": "unknown", "error": err, "usage": {}}


async def judge_one_async(*a, **k) -> dict:
    return await asyncio.to_thread(judge_one, *a, **k)
