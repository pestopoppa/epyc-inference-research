"""Endpoint layer: a stdlib-only OpenAI-compatible chat client for live runs and
a deterministic dry-run stub for zero-inference testing.

Live mode talks to any OpenAI-compatible /v1/chat/completions endpoint (e.g. a
local llama-server). Tool-call arguments arrive as JSON strings and are parsed
here (a parse failure after retries is a typed PARSER failure).

Dry-run mode plays a scripted fixture (see env_state.py): turn 1 returns the
fixture's tool calls, turn 2 the fixture's final agent responses. The stub is
deterministic per seed: the seed only decorates the response envelope (ids),
never the semantics, so the same seed reproduces semantic events and verdicts.
Original per-execution UTC/elapsed observations give each trace its own identity.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .outcomes import EndpointFailure, ModelFailure, OverflowFailure, ParseFailure

CONTEXT_LENGTH_HINTS = (
    "context length",
    "maximum context",
    "max context",
    "context window",
    "token limit",
    "exceeded the model's maximum",
    "too many tokens",
    "input is too long",
)


@dataclass
class ChatResult:
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: Optional[Dict[str, Any]] = None
    transport_detail: Optional[Dict[str, Any]] = None


class ChatEndpoint:
    """OpenAI-compatible /v1/chat/completions client (stdlib urllib only)."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "none",
        model: str = "local",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retries: int = 2,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.timeout = timeout

    def complete(self, messages: List[Dict[str, Any]], seed: int = 0) -> ChatResult:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        last_err: Optional[BaseException] = None
        timeout_attempts = 0
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                parsed = self._parse(raw)
                parsed.transport_detail = {"terminal_native_timeout": False,
                    "timeout_attempts": timeout_attempts, "attempts": attempt + 1,
                    "cap_scope": "endpoint_request", "request_timeout_s": self.timeout}
                return parsed
            except OverflowFailure:
                raise
            except ParseFailure:
                raise
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504):
                    last_err = exc
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise EndpointFailure(f"endpoint HTTP {exc.code}: {exc.reason}", detail={
                    "terminal_native_timeout": False, "timeout_attempts": timeout_attempts,
                    "attempts": attempt + 1, "cap_scope": "endpoint_request",
                    "request_timeout_s": self.timeout}) from exc
            except urllib.error.URLError as exc:
                native_timeout = isinstance(exc.reason, TimeoutError)
                timeout_attempts += int(native_timeout)
                if attempt < self.retries:
                    last_err = exc
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise EndpointFailure(f"endpoint unreachable: {exc.reason}", detail={
                    "terminal_native_timeout": native_timeout,
                    "timeout_attempts": timeout_attempts, "attempts": attempt + 1,
                    "cap_scope": "endpoint_request", "request_timeout_s": self.timeout,
                }) from exc
            except TimeoutError as exc:
                timeout_attempts += 1
                if attempt < self.retries:
                    last_err = exc
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise EndpointFailure(f"endpoint timeout after {self.retries + 1} attempts", detail={
                    "terminal_native_timeout": True,
                    "timeout_attempts": timeout_attempts, "attempts": attempt + 1,
                    "cap_scope": "endpoint_request", "request_timeout_s": self.timeout,
                }) from exc
        raise EndpointFailure(f"endpoint failed after retries: {last_err}", detail={
            "terminal_native_timeout": False, "timeout_attempts": timeout_attempts,
            "attempts": self.retries + 1, "cap_scope": "endpoint_request",
            "request_timeout_s": self.timeout})

    def _parse(self, raw: Dict[str, Any]) -> ChatResult:
        try:
            choice = raw["choices"][0]
            message = choice.get("message") or {}
            finish = str(choice.get("finish_reason") or "stop")
        except (KeyError, IndexError, TypeError) as exc:
            raise ParseFailure(f"unexpected chat.completions envelope: {exc}") from exc
        if finish == "length":
            raise OverflowFailure("finish_reason=length (token budget exhausted mid-turn)")
        text = message.get("content") or ""
        tool_calls: List[Dict[str, Any]] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                arguments = json.loads(fn.get("arguments") or "{}") if fn.get("arguments") else {}
            except json.JSONDecodeError as exc:
                raise ParseFailure(f"tool call arguments are not valid JSON ({name!r}): {exc}") from exc
            if not isinstance(arguments, dict):
                raise ParseFailure(f"tool call arguments must be a JSON object ({name!r})")
            tool_calls.append({"name": name, "arguments": arguments, "id": tc.get("id") or ""})
        if not text and not tool_calls:
            raise ModelFailure("endpoint returned an empty completion with no tool calls")
        return ChatResult(text=text, tool_calls=tool_calls, finish_reason=finish, raw=raw)


class DryRunStub:
    """Scripted zero-inference responder driven by an arm fixture."""

    def __init__(self, fixture: Dict[str, Any], seed: int = 0):
        self.fixture = fixture
        self._rng = random.Random(seed)
        self._turns = 0
        self._pending = list(enumerate(fixture.get("script") or []))
        self._max_calls_per_turn = 8

    def complete(self, messages: List[Dict[str, Any]], seed: int = 0) -> ChatResult:
        self._turns += 1
        envelope = {"id": f"stub-{self._rng.randint(0, 10**9):010d}", "object": "chat.completion"}
        if self._pending:
            calls = []
            for _ in range(self._max_calls_per_turn):
                if not self._pending:
                    break
                idx, step = self._pending.pop(0)
                calls.append(
                    {
                        "name": step.get("tool", ""),
                        "arguments": step.get("arguments", {}),
                        "id": step.get("call_id") or f"call_{self._turns}_{idx}",
                        "script_index": idx,
                    }
                )
            return ChatResult(
                text="",
                tool_calls=calls,
                finish_reason="tool_calls",
                raw={"choices": [{"message": {"content": None, "tool_calls": calls}}], **envelope},
            )
        responses = self.fixture.get("agent_responses") or ["(stub: no agent responses in fixture)"]
        text = " ".join(str(r) for r in responses)
        return ChatResult(text=text, finish_reason="stop", raw={"choices": [{"message": {"content": text}}], **envelope})
