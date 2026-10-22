"""YARN-JUDGE-1: cloud judge and cloud labeller over the orchestrator cloud-judge transport. Stdlib only.

Operator ruling 2026-10-22: the YaRN judge is the pinned Claude cloud judge (``--judge-backend cloud
--cloud-judge sonnet-5-5-low``), not a served local model, so judging needs no server, no GPU, and no RAM
booking. The gold labels are produced by a DIFFERENT model family (``codex-luna-low``, gpt-5.6-luna) plus
deterministic rules. Claude-written labels would share the judge's blind spots and inflate agreement.

Transport: the orchestrator's ``typed_decisions/cloud_judges.py`` (registry ``orchestration/cloud_judges.yaml``,
or ``ORCHESTRATOR_CLOUD_JUDGES_FILE``). It runs ``claude -p --json-schema --restricted --tools ""
--strict-mcp-config`` or ``codex exec --output-schema -s read-only``, with the prompt on stdin, an empty
temporary cwd, its own process group, and a timeout kill. No API key is handled here. This is the same
pattern as research ``judge_beam_run.py`` (HS-PROF-1).

Fail-closed, like ``local_judge``: a call that fails, or a reply that does not parse, is recorded as
unavailable or unparsed. It is NEVER folded as 0.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.benchmark.yarn_gate.local_judge import (
    PROMPT_TEMPLATE, PROMPT_VERSION, TIMING_WINDOW_ENV, JudgeError, parse_verdict, prompt_sha256)

CLOUD_JUDGES_PY = Path(os.environ.get(
    "YARN_CLOUD_JUDGES_PY", "/mnt/raid0/llm/epyc-orchestrator/src/typed_decisions/cloud_judges.py"))
_KEY = "epyc_orch_cloud_judges"
DEFAULT_CLOUD_JUDGE = "sonnet-5-5-low"
DEFAULT_CLOUD_LABELER = "codex-luna-low"
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
              "total_cost_usd", "calls")

VERDICT_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["correct", "incorrect"]}},
                  "required": ["verdict"], "additionalProperties": False}


def load_cloud_judges():
    """The orchestrator cloud_judges module, loaded by absolute path (same pattern as judge_beam_run.py)."""
    if _KEY in sys.modules:
        return sys.modules[_KEY]
    if not CLOUD_JUDGES_PY.is_file():
        raise ImportError(f"orchestrator cloud_judges not found: {CLOUD_JUDGES_PY}")
    spec = importlib.util.spec_from_file_location(_KEY, CLOUD_JUDGES_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_KEY] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_spec(cj, name: str, *, transport: str | None = None):
    """Registry entry ``name``. Raises ValueError if the name is unknown or uses the wrong transport."""
    reg = cj.load_registry()
    if name not in reg:
        raise ValueError(f"unknown cloud judge {name!r}; registry {cj.registry_path()} has {sorted(reg)}")
    spec = reg[name]
    if transport and spec.transport != transport:
        raise ValueError(f"cloud judge {name!r} is {spec.transport}, need {transport}")
    return spec


def registry_sha256(cj) -> str:
    p = Path(cj.registry_path())
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""


class CloudCaller:
    """Structured-output calls over one registry entry. Thread-safe usage counters.

    The claude_cli envelope (usage, total_cost_usd, modelUsage) is captured by wrapping ``run_fn``,
    because the registry transport returns only the schema object. codex_exec emits no envelope, so its
    calls are counted and the model is the registry pin.
    """

    def __init__(self, cj, spec, *, model: str | None = None, run_fn=None):
        self.cj, self.spec = cj, spec
        self.model = spec.resolve_model(model)
        self._inner = run_fn or cj._default_run
        self._lock = threading.Lock()
        self.usage: dict[str, float] = {k: 0 for k in USAGE_KEYS}
        self.models: set[str] = set()

    def _run(self, argv, stdin_text, timeout_s, cwd):
        rc, out, err = self._inner(argv, stdin_text, timeout_s, cwd)
        env = (self.cj._first_object(out) or {}) if self.spec.transport == "claude_cli" else {}
        u = env.get("usage") if isinstance(env.get("usage"), dict) else {}
        with self._lock:
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                self.usage[k] += int(u.get(k) or 0)
            self.usage["total_cost_usd"] += float(env.get("total_cost_usd") or 0.0)
            self.usage["calls"] += 1
            mu = env.get("modelUsage")
            if isinstance(mu, dict):
                self.models.update(str(k) for k in mu)
        return rc, out, err

    def call(self, prompt: str, schema: dict) -> dict:
        try:
            res = self.cj.run_cloud_judge(self.spec, prompt, schema, model=self.model, run_fn=self._run)
        except self.cj.CloudJudgeError as exc:
            raise JudgeError(str(exc)) from exc
        if self.spec.transport != "claude_cli":
            with self._lock:
                self.models.add(str(res.resolved_model or self.model))
        return res.payload

    def preflight(self) -> None:
        """One trivial call (60 s cap). Fails fast if the CLI is missing or not logged in."""
        exe = self.cj.CLAUDE_BIN if self.spec.transport == "claude_cli" else self.cj._codex_binary()
        if self._inner is self.cj._default_run and not (Path(exe).exists() or shutil.which(Path(exe).name)):
            raise JudgeError(f"{self.spec.transport} CLI not found: {exe}")
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                  "additionalProperties": False}
        try:
            self.cj.run_cloud_judge(dataclasses.replace(self.spec, timeout_s=60.0), 'Reply with {"ok": true}.',
                                    schema, model=self.model, run_fn=self._inner)
        except self.cj.CloudJudgeError as exc:
            raise JudgeError(f"{self.spec.name}: not authenticated or unreachable: {exc}") from exc

    def identity(self) -> dict[str, Any]:
        u = dict(self.usage)
        u["total_cost_usd"] = round(float(u["total_cost_usd"]), 6)
        return {"cloud_judge": self.spec.name, "transport": self.spec.transport, "model": self.model,
                "effort": self.spec.effort, "egress": self.spec.egress, "registry": str(self.cj.registry_path()),
                "registry_sha256": registry_sha256(self.cj), "envelope_models": sorted(self.models),
                "envelope_models_source": ("claude envelope modelUsage" if self.spec.transport == "claude_cli"
                                           else "registry pin (codex_exec emits no envelope; usage not reported)"),
                "usage": u}


@dataclass(frozen=True)
class CloudJudgePin:
    """The cloud counterpart of ``local_judge.JudgePin``. Every field lands in the claim tuple.
    The claude CLI sets neither temperature nor seed, so both are recorded as None, never as a guess."""

    model: str
    cloud_judge: str
    transport: str
    effort: str
    registry_sha256: str
    prompt_version: str = PROMPT_VERSION
    prompt_sha256: str = dataclasses.field(default_factory=prompt_sha256)
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None  # the claude CLI sets no output cap; key kept for the belief-kernel adapter

    def claim_fields(self) -> dict[str, Any]:
        d = asdict(self)
        d["judge_kind"] = f"cloud-{self.transport}"
        d["comparable_to_published"] = False
        return d


class CloudJudge:
    """Same ``judge(question, reference, candidate)`` contract as ``LocalJudge``, over a cloud judge."""

    def __init__(self, caller: CloudCaller, *, allow_in_timing_window: bool = False):
        self.caller = caller
        self._allow = allow_in_timing_window
        self.pin = CloudJudgePin(model=caller.model, cloud_judge=caller.spec.name, transport=caller.spec.transport,
                                 effort=caller.spec.effort, registry_sha256=registry_sha256(caller.cj))

    def build_prompt(self, question: str, reference: Any, candidate: str) -> str:
        if isinstance(reference, (list, tuple)):
            reference = " | ".join(str(x) for x in reference)
        return PROMPT_TEMPLATE.format(question=question, reference=reference, candidate=candidate)

    def judge(self, question: str, reference: Any, candidate: str) -> dict[str, Any]:
        if os.environ.get(TIMING_WINDOW_ENV) == "1" and not self._allow:
            raise JudgeError(f"{TIMING_WINDOW_ENV}=1: judging must run outside the YaRN timing window")
        payload = self.caller.call(self.build_prompt(question, reference, candidate), VERDICT_SCHEMA)
        raw = json.dumps(payload, sort_keys=True)
        v = parse_verdict(raw)
        return {"verdict": v, "unparsed": v is None, "raw": raw}
