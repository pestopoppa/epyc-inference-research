"""Deterministic fakes for ``--dry-run`` and the tests: no network, no inference.

* ``FakePrimitives`` stands in for ``LLMPrimitives``: it answers the verdict prompt (canaries
  exactly as a healthy reviewer would, other items by a hash rule) and the revision prompt, and
  keeps the ``total_*`` counters ``v1_escalation._counters`` reads.
* ``FakeRetriever`` answers ``retrieve_for_routing`` with hash-derived rows (some keys retrieve
  nothing, some no frontdoor rows), so every gate skip reason is exercised.
* ``FakeTransport`` answers stage 1 like ``V1Transport`` (a receipt with
  ``request_device_seconds``), correct on roughly 55% of items.
* ``StubChatReview`` is a pure-python copy of the production contract (``review_gate_score``,
  ``_should_review``, ``_architect_verdict_with_status``, ``_fast_revise``) for tests that run
  WITHOUT an orchestrator checkout. ``--dry-run`` uses the REAL orchestrator functions with the
  fake primitives/retriever whenever ``--code-root`` imports; ``--stub-orch`` forces the stub.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..thesis_ufh13.transports import TransportResult

CANARY_QUESTION = "What is 17 * 23?"
CANARY_RIGHT = "391"
CANARY_WRONG = "401"


def _h(text: str, mod: int) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) % mod


def _answer_in_verdict_prompt(prompt: str) -> str:
    """The ``A:`` field of the production verdict prompt (fallback template layout)."""
    m = re.search(r"\nA:\s*(.*?)\n\s*\nVerdict:", prompt, re.DOTALL)
    return (m.group(1) if m else prompt).strip()


class FakePrimitives:
    """Counters, last meta and ``llm_call`` with the production call signature."""

    def __init__(self, *, wrong_every: int = 3, fail_roles: tuple[str, ...] = (),
                 raise_exc: BaseException | None = None, verdict_override: str | None = None
                 ) -> None:
        self.total_calls = 0
        self.total_prompt_eval_ms = 0.0
        self.total_generation_ms = 0.0
        self.total_tokens_generated = 0
        self.total_prompt_tokens_reported = 0
        self._last_meta: dict[str, Any] | None = None
        self.wrong_every = wrong_every
        self.fail_roles = fail_roles
        self.raise_exc = raise_exc
        self.verdict_override = verdict_override
        self.calls: list[dict[str, Any]] = []

    def get_last_inference_meta(self) -> dict[str, Any] | None:
        return self._last_meta

    def llm_call(self, prompt: str, role: str = "worker_general", n_tokens: int | None = None,
                 **kwargs: Any) -> str:
        self.calls.append({"role": role, "n_tokens": n_tokens, "kwargs": sorted(kwargs)})
        if self.raise_exc is not None:
            raise self.raise_exc
        if role in self.fail_roles:
            raise RuntimeError(f"fake failure for {role}")
        if "Judge this answer" in prompt:
            answer = _answer_in_verdict_prompt(prompt)
            if self.verdict_override is not None:
                text = self.verdict_override
            elif CANARY_QUESTION in prompt:
                text = "OK" if answer == CANARY_RIGHT else "WRONG: 17*23 = 391"
            elif _h(prompt, self.wrong_every) == 0:
                text = "WRONG: the final letter does not follow from the reasoning."
            else:
                text = "OK"
            prompt_ms, gen_ms, toks = 180.0 + len(prompt) / 20, 40.0 + 8 * len(text.split()), 4
        else:
            letter = "ABCD"[_h(prompt, 4)]
            text = f"Applying the correction, the right option is {letter}.\nAnswer: {letter}"
            prompt_ms, gen_ms, toks = 900.0 + len(prompt) / 5, 2400.0, 40
        self.total_calls += 1
        self.total_prompt_eval_ms += prompt_ms
        self.total_generation_ms += gen_ms
        self.total_tokens_generated += toks
        self.total_prompt_tokens_reported += len(prompt) // 4
        self._last_meta = {"tokens": toks, "completion_reason": "stop", "transport": "fake",
                           "role": role}
        return text


@dataclass
class _Memory:
    action: str


@dataclass
class _Result:
    memory: _Memory
    q_value: float


class FakeRetriever:
    """``retrieve_for_routing(task_ir)`` -> up to 5 rows; deterministic in the objective."""

    def __init__(self) -> None:
        self.calls = 0

    def retrieve_for_routing(self, task_ir: dict[str, Any]) -> list[_Result]:
        self.calls += 1
        key = str(task_ir.get("objective", ""))
        kind = _h("kind" + key, 20)
        if kind == 0:
            return []
        rows = []
        for i in range(5):
            role = "frontdoor" if (kind != 1 and _h(f"{key}{i}r", 3) != 0) else "coder_escalation"
            q = 0.3 + 0.7 * (_h(f"{key}{i}q", 1000) / 999.0)
            rows.append(_Result(_Memory(role), q))
        return rows


def fake_state() -> Any:
    return SimpleNamespace(hybrid_router=SimpleNamespace(retriever=FakeRetriever()))


class FakeTransport:
    name = "fake-v1"

    def __init__(self, items: list[dict[str, Any]], *, fail_ids: tuple[str, ...] = ()) -> None:
        self.expected = {row["id"]: row for row in items}
        self.fail_ids = fail_ids

    def describe(self) -> dict[str, Any]:
        return {"transport": self.name, "endpoint": "fake://v1/chat/completions"}

    def ask(self, arm: str, item_id: str, prompt: str, session_id: str) -> TransportResult:
        if item_id in self.fail_ids:
            return TransportResult("timeout", error="fake timeout", session_id=session_id)
        row = self.expected[item_id]
        right = _h("ans" + item_id, 100) < 55
        if row["stratum"] == "S1":
            letter = row["expected"] if right else ("B" if row["expected"] != "B" else "C")
            short = _h("short" + item_id, 10) == 0
            text = (f"Answer: {letter}" if short else
                    f"The stem points to option {letter} because the others fail the stated "
                    f"constraint.\nAnswer: {letter}")
        else:
            value = row["expected"].strip("$") if right else "0"
            text = f"Working through the cases gives the result.\n\\boxed{{{value}}}"
        request_s = 3.0 + _h("ds" + item_id, 50) / 10
        receipt = {"enabled": False, "disabled_reason": "x_escalation_off", "fired": False,
                   "from_role": "frontdoor", "final_answer_role": "frontdoor",
                   "request_device_seconds": request_s, "consultant_device_seconds": 0.0,
                   "steps": []}
        return TransportResult("ok", text=text, finish_reason="stop", http_status=200,
                               session_id=session_id, served_role="frontdoor",
                               usage={"completion_tokens": len(text) // 4},
                               receipts=[receipt])


def fake_served_identity() -> dict[str, Any]:
    return {"server_launch_git_sha": "dryrun0", "server_started_at": 0.0, "git_sha": "dryrun0"}


# ── the production contract, in pure python (tests without an orchestrator checkout) ────


@dataclass(frozen=True)
class GateScore:
    avg_q: float | None
    n_results: int
    n_role_rows: int
    skip_reason: str
    threshold: float
    answer_chars: int
    gate_ms: float | None = None

    def fires_at(self, threshold: float) -> bool:
        return self.avg_q is not None and self.avg_q < threshold

    @property
    def triggered(self) -> bool:
        return self.fires_at(self.threshold)


class StubChatReview:
    """Mirror of ``src/api/routes/chat_review.py`` (RI-18 contract), threshold 0.6."""

    THRESHOLD = 0.6
    GateScore = GateScore

    def review_gate_score(self, state: Any, role: str, answer: str, *,
                          key_text: str | None = None) -> GateScore:
        def skip(reason: str, n: int = 0, nr: int = 0) -> GateScore:
            return GateScore(None, n, nr, reason, math.nan, len(answer))

        if not state.hybrid_router:
            return skip("no_router")
        if "architect" in str(role):
            return skip("architect_role")
        if len(answer) < 50:
            return skip("short")
        try:
            key = answer if key_text is None else key_text
            results = state.hybrid_router.retriever.retrieve_for_routing(
                {"task_type": "chat", "objective": key[:200]})
            if not results:
                return skip("no_results")
            rows = [r for r in results if r.memory.action == str(role)]
            if not rows:
                return skip("no_role_rows", len(results))
            avg = sum(r.q_value for r in rows) / len(rows)
            return GateScore(avg, len(results), len(rows), "scored", self.THRESHOLD, len(answer))
        except Exception:
            return skip("error")

    def _should_review(self, state: Any, task_id: str, role: str, answer: str) -> bool:
        return self.review_gate_score(state, role, answer).triggered

    def _architect_verdict_with_status(self, question: str, answer: str, primitives: Any,
                                       worker_digests: Any = None, context_digest: str = "",
                                       role: str | None = None, *, question_cap: int = 300
                                       ) -> tuple[str | None, str]:
        prompt = (f"Judge this answer. Respond with ONLY one line:\n\nQ: {question[:question_cap]}"
                  f"\nA: {answer[:1500]}\n\nVerdict:")
        try:
            text = primitives.llm_call(prompt, role=role or "architect_critic", n_tokens=80,
                                       skip_suffix=True).strip()
        except Exception:
            return None, "unavailable"
        up = text.upper()
        status = "ok" if up.startswith("OK") else "wrong" if up.startswith("WRONG") else (
            "unavailable")
        return (None if status == "ok" else text), status

    def _fast_revise(self, question: str, original_answer: str, corrections: str,
                     primitives: Any) -> str:
        prompt = (f"Rewrite this answer applying the corrections below.\n\nQuestion: "
                  f"{question[:300]}\n\nOriginal answer: {original_answer[:1500]}\n\n"
                  f"Corrections: {corrections}\n\nRevised answer:")
        try:
            return primitives.llm_call(prompt, role="worker_general", n_tokens=2000).strip() or (
                original_answer)
        except Exception:
            return original_answer


def stub_counters(primitives: Any) -> dict[str, float]:
    def num(name: str) -> float:
        value = getattr(primitives, name, 0)
        return 0.0 if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)

    return {"calls": num("total_calls"), "prompt_ms": num("total_prompt_eval_ms"),
            "gen_ms": num("total_generation_ms"), "tokens": num("total_tokens_generated"),
            "prompt_tokens": num("total_prompt_tokens_reported")}


def stub_quality(answer: str) -> str | None:
    """Stand-in for ``detect_output_quality_issue``: flags only an empty or repeated answer."""
    if not answer.strip():
        return "empty"
    lines = [ln for ln in answer.splitlines() if ln.strip()]
    if len(lines) >= 6 and len(set(lines)) <= 2:
        return "repetition"
    return None
