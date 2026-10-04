#!/usr/bin/env python3
"""Coherence-judge client: a ``coherence_gate``-compatible ``judge_fn`` (2026-10-04).

Stdlib only (like ``scripts/autokernel_actor_cli.py``), so AutoKernel and runbooks can
load it by PATH under ``python -I`` without importing the orchestrator package::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "coherence_judge_client", "/mnt/raid0/llm/epyc-orchestrator/scripts/coherence_judge_client.py")
    cjc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cjc  # required before exec (dataclasses)
    spec.loader.exec_module(cjc)
    judge_fn = cjc.make_judge_fn(backend="local")            # or backend="cloud:codex-luna-low"
    verdict = judge_fn({"prompt": p, "base_output": b, "candidate_output": c})
    verdict.passed, verdict.verdict, verdict.confidence, verdict.calibration_id

``item_pair`` may be a mapping or an object carrying ``prompt`` plus ``base_output`` /
``base`` / ``base_text`` and ``candidate_output`` / ``candidate`` / ``candidate_text``
(optional ``rubric``), or a ``(prompt, base, candidate)`` tuple.

Contract for the gate:

* returns a :class:`Verdict` — ``passed`` is True only for ``COHERENT_EQUIVALENT``;
* raises :class:`JudgeUnavailable` when no verdict exists (window held, transport,
  unresolved) — the gate must treat the pair as UNDETERMINED, never as passed;
  ``retry_after_s`` is set when the server gave one;
* raises :class:`UncalibratedJudge` (a ``JudgeUnavailable``) when the judge has no
  passed calibration, unless ``allow_uncalibrated=True`` (``--allow-uncalibrated``).
  The server refuses first (409); the client re-checks the verdict's
  ``calibration_id`` so an older server cannot slip an uncalibrated verdict through.

CLI (one pair; exit 0 = COHERENT_EQUIVALENT, 1 = any other verdict, 2 = unavailable /
uncalibrated, 64 = usage)::

    python -I scripts/coherence_judge_client.py --pair-json pair.json \
        [--backend local|cloud:<name>] [--model M] [--allow-uncalibrated]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

DEFAULT_URL = os.environ.get("COHERENCE_JUDGE_URL", "http://127.0.0.1:8000")
ENDPOINT = "/v1/typed/coherence_judge"
PASS_VERDICT = "COHERENT_EQUIVALENT"
VERDICTS = ("COHERENT_EQUIVALENT", "DEGRADED", "INCOHERENT", "OFF_TASK")

#: (url, body, timeout_s) -> (status, json body)
PostFn = Callable[[str, dict, float], "tuple[int, dict]"]


class JudgeUnavailable(RuntimeError):
    """No verdict: the gate must mark the pair undetermined (never passed)."""

    def __init__(self, kind: str, message: str, *, retry_after_s: int | None = None, body: Any = None):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.retry_after_s = retry_after_s
        self.body = body


class UncalibratedJudge(JudgeUnavailable):
    """The judge has no passed calibration and the caller did not allow that."""


@dataclass(frozen=True)
class Verdict:
    verdict: str
    passed: bool
    confidence: float | None
    calibration_id: str | None
    backend: str
    model: str
    judge_version: str
    call_id: str | None = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> dict:
        return dict(self.raw)


def _pick(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if not isinstance(obj, Mapping) and hasattr(obj, name):
            return getattr(obj, name)
    return None


def pair_fields(item_pair: Any) -> tuple[str, str, str, str | None]:
    """(prompt, base, candidate, rubric) from any supported item_pair shape."""
    if isinstance(item_pair, (tuple, list)) and len(item_pair) == 3:
        prompt, base, candidate = item_pair
        return str(prompt), str(base), str(candidate), None
    prompt = _pick(item_pair, ("prompt", "prompt_text"))
    base = _pick(item_pair, ("base_output", "base", "base_text"))
    candidate = _pick(item_pair, ("candidate_output", "candidate", "candidate_text"))
    if prompt is None or base is None or candidate is None:
        raise ValueError("item_pair needs prompt, base_output and candidate_output")
    rubric = _pick(item_pair, ("rubric",))
    return str(prompt), str(base), str(candidate), (str(rubric) if rubric else None)


def _default_post(url: str, body: dict, timeout_s: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8") or "{}")
        except ValueError:
            return exc.code, {"error": {"type": "http_error", "message": str(exc)}}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise JudgeUnavailable("transport_error", str(exc)) from exc


def make_judge_fn(
    *,
    url: str = DEFAULT_URL,
    backend: str = "local",
    model: str | None = None,
    rubric: str | None = None,
    scoring: str = "auto",
    allow_uncalibrated: bool = False,
    caller: str | None = None,
    timeout_s: float = 600.0,
    post: PostFn | None = None,
) -> Callable[[Any], Verdict]:
    """Build ``judge_fn(item_pair) -> Verdict`` bound to one judge identity."""
    endpoint = url.rstrip("/") + ENDPOINT
    send = post or _default_post

    def judge_fn(item_pair: Any) -> Verdict:
        prompt, base, candidate, pair_rubric = pair_fields(item_pair)
        body = {
            "prompt": prompt,
            "base_output": base,
            "candidate_output": candidate,
            "rubric": pair_rubric or rubric,
            "backend": backend,
            "model": model,
            "scoring": scoring,
            "allow_uncalibrated": allow_uncalibrated,
            "caller": caller,
        }
        status, payload = send(endpoint, body, timeout_s)
        if status != 200:
            error = (payload or {}).get("error") or {}
            if not isinstance(error, dict):  # FastAPI HTTPException shape
                error = {"type": f"http_{status}", "message": str((payload or {}).get("detail"))}
            kind = str(error.get("type") or f"http_{status}")
            cls = UncalibratedJudge if kind == "judge_uncalibrated" else JudgeUnavailable
            raise cls(kind, str(error.get("message") or ""), retry_after_s=error.get("retry_after_s"), body=payload)
        label = payload.get("verdict")
        if label not in VERDICTS:
            raise JudgeUnavailable("judge_unresolved", f"server returned verdict {label!r}", body=payload)
        calibration_id = payload.get("calibration_id")
        if calibration_id is None and not allow_uncalibrated:
            raise UncalibratedJudge(
                "judge_uncalibrated", "verdict carries no calibration_id", body=payload
            )
        return Verdict(
            verdict=label,
            passed=label == PASS_VERDICT,
            confidence=payload.get("confidence"),
            calibration_id=calibration_id,
            backend=str(payload.get("backend")),
            model=str(payload.get("model")),
            judge_version=str(payload.get("judge_version")),
            call_id=payload.get("call_id"),
            raw=payload,
        )

    return judge_fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge one (prompt, base, candidate) pair.")
    parser.add_argument("--pair-json", required=True, help="JSON file: {prompt, base_output, candidate_output, rubric?}")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--backend", default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--scoring", default="auto", choices=("auto", "native", "json"))
    parser.add_argument("--allow-uncalibrated", action="store_true")
    parser.add_argument("--caller", default="cli")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 64
    with open(args.pair_json, encoding="utf-8") as handle:
        pair = json.load(handle)
    judge_fn = make_judge_fn(
        url=args.url,
        backend=args.backend,
        model=args.model,
        scoring=args.scoring,
        allow_uncalibrated=args.allow_uncalibrated,
        caller=args.caller,
        timeout_s=args.timeout_s,
    )
    try:
        verdict = judge_fn(pair)
    except JudgeUnavailable as exc:
        print(json.dumps({"error": exc.kind, "message": str(exc), "retry_after_s": exc.retry_after_s}))
        return 2
    print(json.dumps(verdict.to_dict(), sort_keys=True))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
