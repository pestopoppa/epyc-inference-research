"""Unit tests for the TD-1d cue-style benchmark harness (no live calls).

``run_mode_benchmark`` is driven with a fake primitives object that serves the
JSON arm a canned schema-valid response and builds each native arm's
``completion_probabilities`` rows from the layout the native runner attaches
right before its call. That is enough to pin the report contract: the JSON arm
runs once, each cue style gets its own native arm, ``agreements`` holds that
style's parity against JSON, and each arm reports wall_ms / tokens_generated /
failures. Accuracy against ground-truth labels is not the harness's job, and
nothing here touches a server.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from src.typed_decisions.bench import (
    BenchmarkError,
    _build_parser,
    _meta_snapshot,
    main,
    run_mode_benchmark,
)
from src.typed_decisions.native import CueStyle, _cue_text
from src.typed_decisions.types import Question, QuestionKind

ROLE = "worker"
STATE = "bench-unit state: Item A is a salmon; Item B is the number 51."

CHOICE = Question(
    id="c01",
    kind=QuestionKind.CHOICE,
    text="What is the animal type of Item A?",
    options=("fish", "bird"),
)
NOUL = Question(id="n01", kind=QuestionKind.NOUL, text="Item B is divisible by 3.")
QUESTIONS = (CHOICE, NOUL)

_VOCAB = {
    "fish": 11,
    " fish": 12,
    "bird": 21,
    " bird": 22,
    "true": 31,
    " true": 32,
    "false": 41,
    " false": 42,
    "yes": 51,
    " yes": 52,
    "no": 61,
    " no": 62,
}


class _FakeTokenizer:
    """Deterministic vocab for candidates; cue texts fall back to char ids."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[int] | None:
        self.calls.append(text)
        if text in _VOCAB:
            return [_VOCAB[text]]
        if not text:
            return []
        return [ord(char) for char in text]


def _json_response() -> str:
    return json.dumps(
        {
            "answers": {
                "c01": {
                    "choice": "fish",
                    "probabilities": {"fish": 0.8, "bird": 0.2},
                    "confidence": 0.8,
                },
                "n01": {
                    "noul": True,
                    "probabilities": {"true": 0.9, "false": 0.1},
                    "confidence": 0.9,
                },
            }
        }
    )


class _BenchPrimitives:
    """Serves the JSON arm once and synthesizes native rows from the layout.

    Native answers default to each question's first candidate; ``native_picks``
    overrides per question id and ``style_picks`` per (cue style, question id),
    so a test can make one style disagree while the others agree. Styles in
    ``broken_styles`` emit a token outside the candidate set, producing typed
    per-position failures for that arm only.
    """

    def __init__(
        self,
        *,
        json_response: str | None = None,
        json_meta: Mapping[str, Any] | None = None,
        native_picks: Mapping[str, str] | None = None,
        style_picks: Mapping[str, Mapping[str, str]] | None = None,
        broken_styles: Sequence[str] = (),
    ) -> None:
        self.json_response = json_response if json_response is not None else _json_response()
        self.json_meta = {"tokens": 50} if json_meta is None else dict(json_meta)
        self.native_picks = dict(native_picks or {})
        self.style_picks = {style: dict(picks) for style, picks in (style_picks or {}).items()}
        self.broken_styles = set(broken_styles)
        self.mock_mode = False
        self.calls: list[dict[str, Any]] = []
        self.native_layouts: list[dict[str, Any]] = []
        self._last_inference_meta: Any = None
        self._last_native_layout: Any = None

    def llm_call(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if "json_schema" in kwargs:
            self._last_inference_meta = dict(self.json_meta)
            return self.json_response
        return self._native_call()

    def _native_call(self) -> str:
        layout = self._last_native_layout or {}
        self.native_layouts.append(layout)
        style = str(layout.get("cue_style", ""))
        picks = dict(self.native_picks)
        picks.update(self.style_picks.get(style, {}))
        rows: list[dict[str, Any]] = []
        for position in layout.get("positions", []):
            for _ in range(int(position.get("cue_length", 0))):
                rows.append({"id": 999, "token": "", "logprob": 0.0, "top_logprobs": []})
            if style in self.broken_styles:
                rows.append(_unknown_row())
                continue
            candidates = position.get("candidates", [])
            default_label = candidates[0]["label"] if candidates else None
            label = picks.get(str(position["question"]["id"]), default_label)
            selected = next(candidate for candidate in candidates if candidate["label"] == label)
            rows.append(_candidate_row(selected))
        self._last_inference_meta = {
            "completion_probabilities": rows,
            "tokens": len(rows),
            "completion_reason": "limit",
        }
        return ""


def _candidate_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    token_ids = list(candidate["token_ids"])
    token_texts = list(candidate["token_texts"])
    return {
        "id": token_ids[0],
        "token": token_texts[0],
        "bytes": list(str(token_texts[0]).encode("utf-8")),
        "logprob": math.log(0.9),
        "top_logprobs": [
            {
                "id": token_id,
                "token": text,
                "bytes": list(str(text).encode("utf-8")),
                "logprob": math.log(0.9) if index == 0 else math.log(0.1),
            }
            for index, (token_id, text) in enumerate(zip(token_ids, token_texts))
        ],
    }


def _unknown_row() -> dict[str, Any]:
    return {
        "id": 12345,
        "token": "zorp",
        "bytes": [122, 111, 114, 112],
        "logprob": math.log(0.9),
        "top_logprobs": [{"id": 12345, "token": "zorp", "logprob": math.log(0.9)}],
    }


def _json_calls(primitives: _BenchPrimitives) -> list[dict[str, Any]]:
    return [call for call in primitives.calls if "json_schema" in call]


def _native_calls(primitives: _BenchPrimitives) -> list[dict[str, Any]]:
    return [call for call in primitives.calls if "grammar" in call]


def _run(primitives: _BenchPrimitives, **kwargs: Any) -> dict[str, Any]:
    return run_mode_benchmark(
        primitives,
        state=STATE,
        questions=QUESTIONS,
        role=ROLE,
        tokenize_fn=kwargs.pop("tokenize_fn", _FakeTokenizer()),
        **kwargs,
    )


# ── 1. Single style keeps the legacy two-arm report ───────────────────────


class TestSingleStyleCompat:
    def test_default_run_is_json_once_plus_native_full(self):
        primitives = _BenchPrimitives(native_picks={"c01": "bird"})

        report = _run(primitives)

        assert report["modes"] == ["json", "native"]
        assert list(report["arms"]) == ["json", "native"]
        assert report["cue_styles"] == ["full"]
        assert report["arms"]["native"]["cue_style"] == "full"
        assert report["arms"]["json"]["cue_style"] is None
        assert report["arms"]["native"]["wall_ms"] >= 0.0
        assert report["arms"]["native"]["failures"] == []
        assert report["arms"]["native"]["decisions"] == 2
        expected_tokens = (
            len(_cue_text(CHOICE, CueStyle.FULL)) + len(_cue_text(NOUL, CueStyle.FULL)) + 2
        )
        assert report["arms"]["native"]["tokens_generated"] == expected_tokens
        assert report["arms"]["json"]["tokens_generated"] == 50
        assert len(_json_calls(primitives)) == 1
        assert len(_native_calls(primitives)) == 1
        # c01 disagrees (json fish vs native bird), n01 agrees (true).
        assert report["agreement"]["comparable"] == 2
        assert report["agreement"]["agreeing"] == 1
        assert report["agreement"]["agreement_rate"] == pytest.approx(0.5)
        assert list(report["agreements"]) == ["native"]
        assert report["agreements"]["native"]["agreement_rate"] == pytest.approx(0.5)

    def test_one_explicit_style_still_names_the_arm_native(self):
        primitives = _BenchPrimitives()

        report = _run(primitives, cue_styles=(CueStyle.SHORT,))

        assert list(report["arms"]) == ["json", "native"]
        assert report["cue_styles"] == ["short"]
        assert report["arms"]["native"]["cue_style"] == "short"
        assert report["arms"]["native"]["failures"] == []
        assert report["agreement"]["agreement_rate"] == pytest.approx(1.0)


# ── 2. Multi-style sweep: JSON once, one native arm per style ─────────────


class TestCueStyleSweep:
    def test_each_style_gets_an_arm_and_its_own_agreement_vs_json(self):
        primitives = _BenchPrimitives(
            style_picks={
                "short": {"c01": "bird"},
                "id_only": {"c01": "bird", "n01": "false"},
            }
        )

        report = _run(primitives, cue_styles=("full", "short", "id_only"))

        assert report["cue_styles"] == ["full", "short", "id_only"]
        assert list(report["arms"]) == ["json", "native:full", "native:short", "native:id_only"]
        assert len(_json_calls(primitives)) == 1
        assert len(_native_calls(primitives)) == 3

        assert report["agreements"]["native:full"]["agreement_rate"] == pytest.approx(1.0)
        assert report["agreements"]["native:short"]["agreement_rate"] == pytest.approx(0.5)
        assert report["agreements"]["native:id_only"]["agreement_rate"] == pytest.approx(0.0)
        # The all-arms fold still means "every arm equals the first (json)".
        assert report["agreement"]["comparable"] == 2
        assert report["agreement"]["agreeing"] == 0
        assert report["agreement"]["disagreeing"] == 2

    def test_styles_run_in_order_and_tokens_shrink(self):
        primitives = _BenchPrimitives()

        report = _run(primitives, cue_styles=("full", "short", "id_only"))

        assert [layout["cue_style"] for layout in primitives.native_layouts] == [
            "full",
            "short",
            "id_only",
        ]
        tokens = {
            name: report["arms"][name]["tokens_generated"]
            for name in ("native:full", "native:short", "native:id_only")
        }
        assert tokens["native:full"] > tokens["native:short"] > tokens["native:id_only"]
        assert all(count is not None for count in tokens.values())

    def test_failures_are_reported_per_style_without_killing_the_run(self):
        primitives = _BenchPrimitives(broken_styles={"id_only"})

        report = _run(primitives, cue_styles=("full", "id_only"))

        assert report["arms"]["native:full"]["failures"] == []
        assert report["arms"]["native:full"]["decisions"] == 2
        id_failures = report["arms"]["native:id_only"]["failures"]
        assert len(id_failures) == 2
        assert all(failure["reason"] == "native_unknown_candidate" for failure in id_failures)
        assert report["arms"]["native:id_only"]["decisions"] == 0
        assert report["agreements"]["native:id_only"]["comparable"] == 0
        assert report["agreements"]["native:id_only"]["unresolved_pairs"] == 2

    def test_native_only_run_reports_no_invented_agreement(self):
        primitives = _BenchPrimitives()

        report = _run(primitives, modes=("native",), cue_styles=("full", "id_only"))

        assert list(report["arms"]) == ["native:full", "native:id_only"]
        assert len(_json_calls(primitives)) == 0
        assert report["agreements"] == {"native:full": None, "native:id_only": None}


# ── 3. Guards and normalization ───────────────────────────────────────────


class TestBenchGuards:
    def test_missing_llm_call_is_refused(self):
        with pytest.raises(BenchmarkError, match="live primitives"):
            run_mode_benchmark(object(), state=STATE, questions=QUESTIONS, role=ROLE)

    def test_mock_mode_is_refused(self):
        primitives = _BenchPrimitives()
        primitives.mock_mode = True

        with pytest.raises(BenchmarkError, match="mock_mode"):
            _run(primitives)

    def test_no_arm_resolving_anything_raises(self):
        primitives = _BenchPrimitives(
            json_response='{"answers": {}}',
            broken_styles={"full"},
        )

        with pytest.raises(BenchmarkError, match="nothing to compare"):
            _run(primitives)

    def test_unknown_cue_style_is_rejected(self):
        with pytest.raises(ValueError, match="unknown cue style"):
            _run(_BenchPrimitives(), cue_styles=("full", "medium"))

    def test_duplicate_cue_styles_collapse(self):
        primitives = _BenchPrimitives()

        report = _run(primitives, cue_styles=("full", "full"))

        assert list(report["arms"]) == ["json", "native"]
        assert report["cue_styles"] == ["full"]
        assert len(_native_calls(primitives)) == 1

    def test_absent_token_telemetry_is_none_not_zero(self):
        primitives = _BenchPrimitives(json_meta={"completion_reason": "eos"})

        report = _run(primitives)

        assert report["arms"]["json"]["tokens_generated"] is None
        assert report["arms"]["json"]["completion_reason"] == "eos"


# ── 4. CLI plumbing ───────────────────────────────────────────────────────


class TestBenchCli:
    def test_cue_style_is_optional_and_repeatable(self):
        parser = _build_parser()
        base = ["--live", "--state-file", "s.json", "--questions-file", "q.json"]

        assert parser.parse_args(base).cue_style is None
        args = parser.parse_args([*base, "--cue-style", "full", "--cue-style", "id_only"])
        assert args.cue_style == ["full", "id_only"]

    def test_cue_style_choices_are_restricted(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(
                ["--state-file", "s.json", "--questions-file", "q.json", "--cue-style", "medium"]
            )

    def test_live_is_required(self):
        assert main(["--state-file", "s.json", "--questions-file", "q.json"]) == 2

    def test_main_forwards_the_sweep_and_writes_the_report(self, monkeypatch, tmp_path):
        captured: dict[str, Any] = {}

        def fake_run(primitives: Any, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"ok": True}

        monkeypatch.setattr("src.typed_decisions.bench.run_mode_benchmark", fake_run)
        monkeypatch.setattr("src.typed_decisions.bench._live_primitives", lambda: object())
        monkeypatch.setattr("src.typed_decisions.bench._load_state", lambda path: STATE)
        monkeypatch.setattr(
            "src.typed_decisions.bench._load_questions", lambda path: list(QUESTIONS)
        )
        out = tmp_path / "report.json"

        code = main(
            [
                "--live",
                "--state-file",
                "s.json",
                "--questions-file",
                "q.json",
                "--cue-style",
                "full",
                "--cue-style",
                "short",
                "--out",
                str(out),
            ]
        )

        assert code == 0
        assert captured["cue_styles"] == ("full", "short")
        assert captured["role"] == "worker"
        assert json.loads(out.read_text(encoding="utf-8")) == {"ok": True}


# ── TD-21.33: _meta_snapshot's real-LLMPrimitives branch ───────────────────
class TestMetaSnapshotRealPrimitives:
    """`_meta_snapshot` prefers the per-call-safe `get_last_inference_meta()`
    getter for a REAL `LLMPrimitives`, falling back to the plain attribute for
    the hand-rolled `_BenchPrimitives` test double used elsewhere in this file.
    No live requests; a fake backend only.
    """

    def test_real_primitives_uses_the_per_call_safe_getter(self):
        from unittest.mock import Mock

        from src.llm_primitives import LLMPrimitives
        from src.model_server import InferenceResult

        prims = LLMPrimitives(mock_mode=False, server_urls={ROLE: "http://localhost:9301"})
        backend = Mock(spec=[])
        backend.infer = Mock(return_value=InferenceResult(
            role=ROLE, output="ignored", tokens_generated=3, generation_speed=1.0,
            elapsed_time=0.001, success=True, prompt_eval_ms=1.0, generation_ms=2.0,
            http_overhead_ms=0.0, completion_reason="limit",
        ))
        prims._backends[ROLE] = backend

        prims._real_call("a prompt", ROLE, n_tokens=8)

        meta = _meta_snapshot(prims)
        assert meta is not None
        assert meta["completion_reason"] == "limit"
        assert meta["tokens"] == 3

    def test_non_llmprimitives_double_falls_back_to_the_plain_attribute(self):
        class _Bare:
            def __init__(self) -> None:
                self._last_inference_meta = {"completion_reason": "eos"}

        assert _meta_snapshot(_Bare()) == {"completion_reason": "eos"}


# ── TD-29.M0: per-arm call accounting and sidecar CLI options ─────────────
class TestBenchCallAccounting:
    def test_json_retry_calls_are_counted_and_summed(self):
        class _RetryPrimitives(_BenchPrimitives):
            def llm_call(self, prompt: str, **kwargs: Any) -> str:
                if "json_schema" in kwargs and not _json_calls(self):
                    self.calls.append({"prompt": prompt, **kwargs})
                    self._last_inference_meta = {
                        "tokens": 70,
                        "prompt_tokens": 400,
                        "cached_prompt_tokens": 0,
                    }
                    return "not json"
                out = super().llm_call(prompt, **kwargs)
                if "json_schema" in kwargs:
                    self._last_inference_meta.update(
                        {"prompt_tokens": 420, "cached_prompt_tokens": 400}
                    )
                return out

        report = _run(_RetryPrimitives(), modes=("json",))

        arm = report["arms"]["json"]
        assert arm["call_count"] == 2
        assert arm["tokens_generated"] == 50
        assert arm["tokens_generated_total"] == 120.0
        assert arm["prompt_tokens_total"] == 820.0
        assert arm["cache_n_total"] == 400.0
        assert arm["prompt_n_total"] == 420.0
        assert len(arm["calls"]) == 2

    def test_native_arm_counts_its_single_call(self):
        report = _run(_BenchPrimitives(), modes=("native",))
        arm = report["arms"]["native"]
        assert arm["call_count"] == 1
        assert arm["tokens_generated_total"] == arm["tokens_generated"]
        assert arm["prompt_n_total"] is None

    def test_main_forwards_server_url_and_embeds_provenance(self, monkeypatch, tmp_path):
        captured: dict[str, Any] = {}

        def fake_live(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            "src.typed_decisions.bench.run_mode_benchmark", lambda primitives, **kw: {"ok": True}
        )
        monkeypatch.setattr("src.typed_decisions.bench._live_primitives", fake_live)
        monkeypatch.setattr("src.typed_decisions.bench._load_state", lambda path: STATE)
        monkeypatch.setattr(
            "src.typed_decisions.bench._load_questions", lambda path: list(QUESTIONS)
        )
        provenance = tmp_path / "prov.json"
        provenance.write_text(json.dumps({"binary_sha256": "abc"}), encoding="utf-8")
        out = tmp_path / "report.json"

        code = main(
            [
                "--live",
                "--state-file",
                "s.json",
                "--questions-file",
                "q.json",
                "--server-url",
                "frontdoor=http://127.0.0.1:8199",
                "--provenance-file",
                str(provenance),
                "--out",
                str(out),
            ]
        )

        assert code == 0
        assert captured == {"server_url_overrides": {"frontdoor": "http://127.0.0.1:8199"}}
        assert json.loads(out.read_text(encoding="utf-8")) == {
            "ok": True,
            "provenance": {"binary_sha256": "abc"},
        }
