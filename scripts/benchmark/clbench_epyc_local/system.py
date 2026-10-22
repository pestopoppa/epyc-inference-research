"""CL-Bench system `epyc_local` — sends each turn to an OpenAI-compatible /v1 endpoint.

Specified by CME-5, built by CME-7. CL-Bench routes every model call through
LiteLLM with no api_base plumbed through the ICL / notepad / structured-output
path, so this registered system bypasses LiteLLM entirely: stdlib HTTP only.

Install: copy this file to ``<clbench>/src/systems/epyc_local/system.py`` and add
the one-line ``__init__.py`` (``install_into_clbench.sh`` does both). The class
registers itself with ``@register_system("epyc_local")`` and is therefore
discoverable as ``clbench run <task> --system epyc_local``.

Layout is prefix-stable for KV reuse: [system prompt] + append-only history. The
structured-output instruction is part of each STORED user message (appended once,
when the message is stored), so turn N's request begins byte-for-byte with turn
N-1's request plus its reply. (It used to be attached to the outgoing last user
message only; that made the previous user message differ at its tail, so the
longest common prefix ended inside it and a hybrid/recurrent server, which can
only resume from a checkpoint, re-prefilled that whole message every turn:
CME-7 poker smoke 2026-10-22, hit_ratio_reusable 0.795.) Truncation (rare) drops
the oldest messages.

Endpoint config (constructor args, overridable with ``--system.<arg> value``;
env fallbacks CLBENCH_EPYC_BASE_URL / CLBENCH_EPYC_API_KEY): base_url, model,
temperature, max_output_tokens, max_context_tokens, timeout_s, enable_thinking (default False).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from src.interface import ContinualLearningSystem, Observation, Query, Response
from src.registry import register_system
from src.usage import UsageEvent

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def _extract_json(text: str) -> str:
    text = text.strip()
    if "```" in text:
        for part in text.split("```")[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part[:1] in ("{", "["):
                return part
    if text[:1] in ("{", "["):
        return text
    s, e = text.find("{"), text.rfind("}")
    return text[s : e + 1] if s != -1 and e > s else text


def _schema_instruction(schema: Any) -> str:
    return (
        "\n\nRespond with ONLY a JSON object (no prose, no code fence) that conforms "
        "to this JSON Schema:\n" + json.dumps(schema.model_json_schema())
    )


class EndpointError(RuntimeError):
    pass


@register_system("epyc_local")
class EpycLocalSystem(ContinualLearningSystem):
    """Full-history in-context system over a local OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "default",
        system_prompt: str = "",
        name: str = "epyc_local",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        max_context_tokens: int = 120000,
        timeout_s: float = 600.0,
        parse_retries: int = 1,
        enable_thinking: bool | str = False,
    ):
        self._name = name
        self.base_url = (base_url or os.environ.get("CLBENCH_EPYC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.max_context_tokens = int(max_context_tokens)
        self.timeout_s = float(timeout_s)
        self.parse_retries = int(parse_retries)
        # Qwen3.6+ think by default; thinking spends the whole max_output_tokens budget and leaves content == ""
        # (CME-7 s5 smoke 2026-10-22: 4/8 first calls ended at 2048 tokens, EOF json_invalid). Thinking OFF is the rule.
        self.enable_thinking = enable_thinking if isinstance(enable_thinking, bool) else str(enable_thinking).lower() in ("1", "true", "yes", "on")
        self._api_key = os.environ.get("CLBENCH_EPYC_API_KEY", "")
        self.messages: list[dict[str, str]] = []
        self.interaction_count = 0
        self.truncation_count = 0

    # -- interface -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self.messages = []
        self.interaction_count = 0
        self.truncation_count = 0

    def observe(self, observation: Observation, next_query: Optional[Query] = None) -> None:
        content = (observation.content or "").strip()
        if content:
            self.messages.append({"role": "user", "content": f"FEEDBACK: {content}"})

    def respond(self, query: Query) -> Response:
        self.interaction_count += 1
        instruction = _schema_instruction(query.response_schema)
        self.messages.append({"role": "user", "content": (query.prompt or "(no content)") + instruction})
        self._truncate()
        action, record = None, ""
        last_err: Exception | None = None
        for attempt in range(self.parse_retries + 1):
            out = self._outgoing()
            if attempt and last_err is not None:
                out.append({"role": "assistant", "content": record})
                out.append({"role": "user", "content": f"That was not valid ({last_err}). Reply with ONLY the JSON object."})
            text, usage = self._chat(out)
            self.record_usage_event(usage)
            record = text
            try:
                action = query.response_schema.model_validate_json(_extract_json(text))
                break
            except Exception as exc:  # noqa: BLE001 - any parse/validation failure
                last_err = exc
        if action is None:
            raise EndpointError(f"response did not validate against schema after retries: {last_err}")
        self.messages.append({"role": "assistant", "content": record})
        return Response(
            action=action,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "epyc_local",
                "model": self.model,
                "truncation_count": self.truncation_count,
                "base_url": self.base_url,
            },
        )

    def get_run_artifacts(self) -> dict[str, Any]:
        return {
            "artifact_type": "epyc_local",
            "messages": list(self.messages),
            "interaction_count": self.interaction_count,
            "truncation_count": self.truncation_count,
            "model": self.model,
            "base_url": self.base_url,
        }

    # -- internals -----------------------------------------------------------
    def _prefix(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []

    def _outgoing(self) -> list[dict[str, str]]:
        return [m.copy() for m in self._prefix() + self.messages]

    def _approx_tokens(self) -> int:
        return sum(len(m["content"]) + 16 for m in self._prefix() + self.messages) // 4

    def _truncate(self) -> None:
        while len(self.messages) > 1 and self._approx_tokens() > self.max_context_tokens:
            self.messages.pop(0)
            self.truncation_count += 1

    def _chat(self, messages: list[dict[str, str]]) -> tuple[str, UsageEvent]:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # noqa: S310 - operator-set local URL
                payload = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EndpointError(f"endpoint call failed: {exc}") from exc
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(f"malformed chat response: {str(payload)[:200]}") from exc
        if not text.strip():
            fin = payload["choices"][0].get("finish_reason")
            think = bool((payload["choices"][0].get("message") or {}).get("reasoning_content"))
            raise EndpointError(f"empty content (finish_reason={fin}, reasoning_content={'present' if think else 'absent'}, "
                                f"enable_thinking={self.enable_thinking}): reasoning exhausted max_output_tokens={self.max_output_tokens}?")
        u = payload.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        usage = UsageEvent(
            call_type="completion",
            model=self.model,
            provider="epyc-local",
            input_tokens=u.get("prompt_tokens"),
            output_tokens=u.get("completion_tokens"),
            total_tokens=u.get("total_tokens"),
            cached_input_tokens=cached,
            cost_usd=None,
            pricing_source="local-endpoint-unpriced",
            response_id=payload.get("id"),
            raw_usage=u or None,
        )
        return text, usage
