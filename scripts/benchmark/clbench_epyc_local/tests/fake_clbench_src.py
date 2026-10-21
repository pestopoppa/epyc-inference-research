"""Minimal stand-ins for CL-Bench's src.interface / src.registry / src.usage, so the
adapter can be tested without LiteLLM, pydantic or any CL-Bench code being executed."""

import sys
import types
from dataclasses import dataclass
from typing import Any, Optional


def install() -> None:
    if getattr(sys.modules.get("src.interface"), "_FAKE", False):
        return
    src = types.ModuleType("src")
    src.__path__ = []  # type: ignore[attr-defined]
    iface, reg, usage = (types.ModuleType(n) for n in ("src.interface", "src.registry", "src.usage"))
    iface._FAKE = True  # type: ignore[attr-defined]

    @dataclass
    class Observation:
        content: str
        instance_complete: bool = True
        metadata: Optional[dict] = None

    @dataclass
    class Query:
        prompt: str
        response_schema: Any
        feedback: Optional[Observation] = None
        instance_id: Optional[str] = None
        instance_index: Optional[int] = None
        metadata: Optional[dict] = None

    @dataclass
    class Response:
        action: Any
        metadata: Optional[dict] = None

    class ContinualLearningSystem:
        def record_usage_event(self, e):
            self.__dict__.setdefault("_usage_events", []).append(e)

        def consume_usage_events(self):
            ev = list(self.__dict__.get("_usage_events", []))
            self.__dict__.get("_usage_events", []).clear()
            return ev

    @dataclass
    class UsageEvent:
        call_type: str
        model: str
        provider: Optional[str] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        reasoning_tokens: Optional[int] = None
        cached_input_tokens: Optional[int] = None
        cache_creation_input_tokens: Optional[int] = None
        cost_usd: Optional[float] = None
        pricing_source: Optional[str] = None
        pricing_error: Optional[str] = None
        response_id: Optional[str] = None
        raw_usage: Optional[dict] = None
        metadata: Optional[dict] = None

    registry: dict = {}
    reg.SYSTEMS_REGISTRY = registry

    def register_system(name):
        def deco(cls):
            registry[name] = cls
            return cls
        return deco

    reg.register_system = register_system
    iface.Observation, iface.Query, iface.Response = Observation, Query, Response
    iface.ContinualLearningSystem = ContinualLearningSystem
    usage.UsageEvent = UsageEvent
    for m in (src, iface, reg, usage):
        sys.modules[m.__name__] = m
