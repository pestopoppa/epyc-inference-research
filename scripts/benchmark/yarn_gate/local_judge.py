"""YARN-JUDGE-1: the local judge client for judged long-context cells. Stdlib only.

Replaces the declined hosted GPT-4o judge. It speaks the OpenAI chat-completions wire format to
``http://localhost:{port}/v1`` (the same shape ``debug_scorer.request_llm_judge_text`` uses for an
explicit ``judge_url``), so any llama-server or OpenAI-compatible client path serves it.

Contract:
  * The judge model id, the prompt template (version + sha256), temperature, seed and max_tokens are
    PINNED in a :class:`JudgePin`. ``pin.claim_fields()`` is what goes into the claim tuple
    (``extra`` / ``applicability``); a verdict without that record is not a claim.
  * Judging runs OUTSIDE the YaRN timing window. :class:`LocalJudge` refuses to run while
    ``YARN_TIMING_WINDOW_ACTIVE=1`` is set, and refuses a non-localhost endpoint (no hosted judge).
  * A reply that does not parse is recorded as ``unparsed`` and is NEVER folded as 0 (fail-closed,
    same rule as ``judge_beam_run.py``).
  * Results are relative to the pinned local judge and never comparable to published figures.

Nothing here starts a server. The caller serves the judge model (the worker model) in a window.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

TIMING_WINDOW_ENV = "YARN_TIMING_WINDOW_ACTIVE"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

PROMPT_VERSION = "yarn-qa-correctness/v1"
#: Binary correctness judge for long-context QA (HELMET longqa-style). The reference answer(s) are
#: shown to the judge; the judge sees the model output, never the long context.
PROMPT_TEMPLATE = """You are grading an answer to a question against reference answer(s).

Question: {question}

Reference answer(s): {reference}

Candidate answer: {candidate}

Is the candidate answer correct, i.e. does it state the same fact(s) as the reference answer(s)? Judge by meaning, not exact wording. Extra correct detail is fine; a contradiction or a missing key fact is incorrect.
Reply with exactly one line of JSON: {{"verdict": "correct"}} or {{"verdict": "incorrect"}}."""

VERDICTS = {"correct": 1, "incorrect": 0}


class JudgeError(RuntimeError):
    """Judge unavailable, or the harness is in a state where judging is forbidden."""


def prompt_sha256(template: str = PROMPT_TEMPLATE) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JudgePin:
    """Everything that makes a judged number reproducible. All fields land in the claim tuple."""

    model: str  # served model id AS REPORTED BY /v1/models, plus the GGUF sha if known
    prompt_version: str = PROMPT_VERSION
    prompt_sha256: str = field(default_factory=prompt_sha256)
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 64
    model_file_sha256: str = ""  # optional: sha256 of the served GGUF

    def claim_fields(self) -> dict[str, Any]:
        d = asdict(self)
        d["judge_kind"] = "local-openai-compat"
        d["comparable_to_published"] = False
        return d


def parse_verdict(text: str) -> int | None:
    """1 / 0, or None when the reply is not a recognisable verdict (never coerced to 0)."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0)).get("verdict")
            if isinstance(v, str) and v.strip().lower() in VERDICTS:
                return VERDICTS[v.strip().lower()]
        except (ValueError, AttributeError):
            pass
    # bare-word fallback only when exactly one of the two words occurs
    words = set(re.findall(r"\b(correct|incorrect)\b", text.lower()))
    if len(words) == 1:
        return VERDICTS[words.pop()]
    return None


def _check_endpoint(base_url: str) -> None:
    host = urlparse(base_url).hostname or ""
    if host not in LOCAL_HOSTS:
        raise JudgeError(f"judge endpoint {base_url!r} is not localhost: hosted judges are declined")


class LocalJudge:
    """``judge(question, reference, candidate) -> {"verdict": 1|0|None, ...}``."""

    def __init__(self, port: int, pin: JudgePin, *, base_url: str | None = None, timeout_s: float = 300.0,
                 allow_in_timing_window: bool = False):
        self.base_url = (base_url or f"http://localhost:{port}/v1").rstrip("/")
        _check_endpoint(self.base_url)
        self.pin = pin
        self.timeout_s = timeout_s
        self._allow = allow_in_timing_window

    def _guard(self) -> None:
        if os.environ.get(TIMING_WINDOW_ENV) == "1" and not self._allow:
            raise JudgeError(f"{TIMING_WINDOW_ENV}=1: judging must run outside the YaRN timing window")

    def served_model_id(self) -> str:
        """Ask the endpoint what it serves; callers compare it to ``pin.model``."""
        req = urllib.request.Request(f"{self.base_url}/models")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                body = json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise JudgeError(f"GET /models failed: {exc}") from exc
        data = body.get("data") or []
        return str(data[0].get("id", "")) if data else ""

    def build_prompt(self, question: str, reference: Any, candidate: str) -> str:
        if isinstance(reference, (list, tuple)):
            reference = " | ".join(str(x) for x in reference)
        return PROMPT_TEMPLATE.format(question=question, reference=reference, candidate=candidate)

    def complete(self, prompt: str) -> str:
        self._guard()
        body = {"model": self.pin.model, "temperature": self.pin.temperature, "seed": self.pin.seed,
                "max_tokens": self.pin.max_tokens, "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = json.loads(r.read())
            return str(out["choices"][0]["message"]["content"] or "")
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            raise JudgeError(f"chat/completions failed: {exc}") from exc

    def judge(self, question: str, reference: Any, candidate: str) -> dict[str, Any]:
        raw = self.complete(self.build_prompt(question, reference, candidate))
        v = parse_verdict(raw)
        return {"verdict": v, "unparsed": v is None, "raw": raw}


def judge_rows(judge: LocalJudge, rows: list[Mapping[str, Any]], *, question_key="question",
               reference_key="answer", candidate_key="output") -> dict[str, Any]:
    """Judge rows; unparsed/unavailable rows are listed, never scored 0."""
    judged, unjudged = [], []
    for i, r in enumerate(rows):
        try:
            j = judge.judge(r[question_key], r[reference_key], r[candidate_key])
        except JudgeError as exc:
            unjudged.append({"i": i, "reason": f"unavailable: {exc}"})
            continue
        if j["unparsed"]:
            unjudged.append({"i": i, "reason": "unparsed", "raw": j["raw"][:200]})
        else:
            judged.append({"i": i, "verdict": j["verdict"]})
    n = len(judged)
    return {"judge_pin": judge.pin.claim_fields(), "n_judged": n, "n_unjudged": len(unjudged),
            "accuracy": (sum(x["verdict"] for x in judged) / n) if n else None,
            "judged": judged, "unjudged": unjudged}
