"""Coherence judge (tier 2): typed-decision core, safety, calibration gate, telemetry.

Inference-free: the typed runner, the tokenizer, the primitives and the cloud CLI are
all fakes. One test drives the REAL native runner (TD-29 single-token keys) through a
fake tokenizer and a synthetic ``completion_probabilities`` capture.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from src.runtime.measurement_windows import WindowHold
from src.typed_decisions import coherence_judge as cj
from src.typed_decisions.cloud_judges import CloudJudgeSpec
from src.typed_decisions.types import Decision, DecisionResult, ParseFailure, QuestionKind

PAIR = dict(
    prompt="What is 2 + 2?",
    base_output="2 + 2 = 4.",
    candidate_output="The sum is 4.",
)


def _result(value: str | None, mode: str, *, failures=(), probs=None) -> DecisionResult:
    decisions = ()
    if value is not None:
        probs = probs or {v: (0.7 if v == value else 0.1) for v in cj.VERDICTS}
        decisions = (
            Decision(
                question_id=cj.QUESTION_ID,
                kind=QuestionKind.CHOICE,
                value=value,
                probabilities=probs,
                confidence=0.6,
                mode=mode,
                native_key="A" if mode == "native" else None,
            ),
        )
    return DecisionResult(
        decisions=decisions,
        failures=tuple(ParseFailure(r, d) for r, d in failures),
        raw_text="",
        mode=mode,
        elapsed_ms=1.0,
        prompt_sha256="0" * 64,
    )


class FakeRun:
    def __init__(self, by_mode: dict[str, DecisionResult]):
        self.by_mode = by_mode
        self.calls: list[dict[str, Any]] = []

    def __call__(self, primitives, **kwargs):
        self.calls.append(kwargs)
        return self.by_mode[kwargs["mode"]]


class FakePrimitives:
    def __init__(self):
        self.contexts: list[dict[str, Any]] = []
        self.server_urls = {"worker_general": "http://127.0.0.1:8070"}

    @contextmanager
    def request_context(self, **kwargs):
        self.contexts.append(kwargs)
        yield


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "calls.jsonl"


@pytest.fixture
def store(tmp_path):
    return cj.CalibrationStore(tmp_path / "cal")


def _judge(store, log_path, *, run=None, holds=(), primitives=None, registry=None, cloud_run=None):
    prims = primitives if primitives is not None else FakePrimitives()
    return cj.CoherenceJudge(
        primitives_fn=lambda: prims,
        holds_fn=lambda: list(holds),
        store=store,
        cloud_registry_fn=lambda: registry or {},
        cloud_run_fn=cloud_run,
        served_model_fn=lambda p, role: "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        typed_run_fn=run,
        log_path=log_path,
    ), prims


def _calibrate(store, *, backend="local", model="worker_general", mode="native", served="Qwen3.6-35B-A3B-Q4_K_M.gguf", passed=True):
    key = cj.judge_key(backend=backend, model=model, scoring_mode=mode, served_model=served)
    record = cj.seal_calibration_record(
        {"judge_key": key, "passed": passed, "created_at": "2026-10-04T00:00:00+00:00"}
    )
    store.write(record)
    return record["calibration_id"], key


def _log(log_path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text().splitlines()]


# ── local verdicts ────────────────────────────────────────────────────────


def test_native_verdict_carries_logprob_confidence_and_calibration(store, log_path):
    cal_id, key = _calibrate(store)
    run = FakeRun({"native": _result("DEGRADED", "native")})
    judge, prims = _judge(store, log_path, run=run)

    verdict = judge.judge(cj.JudgeRequest(**PAIR))

    assert verdict.verdict == "DEGRADED" and not verdict.passed
    assert verdict.confidence == pytest.approx(0.7)
    assert verdict.confidence_source == "native_token_probs"
    assert verdict.scoring_mode == "native"
    assert verdict.calibration_id == cal_id and verdict.judge_key == key
    assert verdict.backend == "local" and verdict.model == "worker_general"
    assert verdict.judge_version == cj.JUDGE_VERSION
    # One native call, the four labels as one choice question.
    assert len(run.calls) == 1
    question = run.calls[0]["questions"][0]
    assert question.options == cj.VERDICTS
    # Serving-call attribution: request_id == call_id.
    assert prims.contexts == [{"request_id": verdict.call_id, "task_id": "coherence_judge"}]


def test_call_log_hashes_inputs_never_stores_text(store, log_path):
    _calibrate(store)
    judge, _ = _judge(store, log_path, run=FakeRun({"native": _result(cj.PASS_VERDICT, "native")}))
    verdict = judge.judge(cj.JudgeRequest(**PAIR, caller="ak:ds41"))
    (record,) = _log(log_path)
    assert record["schema"] == cj.CALL_SCHEMA
    assert record["outcome"] == "verdict" and record["call_id"] == verdict.call_id
    assert record["caller"] == "ak:ds41"
    assert record["verdict"]["verdict"] == cj.PASS_VERDICT
    blob = json.dumps(record)
    assert PAIR["candidate_output"] not in blob and PAIR["prompt"] not in blob
    assert record["inputs"]["candidate_output"]["chars"] == len(PAIR["candidate_output"])


def test_auto_falls_back_to_json_without_confidence(store, log_path):
    _calibrate(store, mode="native")
    json_id, _ = _calibrate(store, mode="json")
    run = FakeRun(
        {
            "native": _result(None, "native", failures=[("native_unknown_candidate", "no probs (MTP accept path)")]),
            "json": _result("INCOHERENT", "json"),
        }
    )
    judge, _ = _judge(store, log_path, run=run)
    verdict = judge.judge(cj.JudgeRequest(**PAIR))
    assert [c["mode"] for c in run.calls] == ["native", "json"]
    assert verdict.verdict == "INCOHERENT" and verdict.scoring_mode == "json"
    assert verdict.confidence is None and verdict.probability_source == "verbalized_json"
    assert verdict.fallback == {"from": "native", "reasons": ["native_unknown_candidate"]}
    assert verdict.calibration_id == json_id


def test_auto_fallback_to_uncalibrated_mode_is_withheld(store, log_path):
    _calibrate(store, mode="native")  # only native is calibrated
    run = FakeRun(
        {
            "native": _result(None, "native", failures=[("native_unsupported_candidates", "x")]),
            "json": _result("DEGRADED", "json"),
        }
    )
    judge, _ = _judge(store, log_path, run=run)
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR))
    assert exc.value.kind == "judge_uncalibrated" and exc.value.status_code == 409
    (record,) = _log(log_path)
    assert record["outcome"] == "refused" and record["verdict"]["verdict"] == "DEGRADED"


def test_explicit_native_failure_does_not_fall_back(store, log_path):
    _calibrate(store)
    run = FakeRun({"native": _result(None, "native", failures=[("native_unknown_candidate", "x")])})
    judge, _ = _judge(store, log_path, run=run)
    with pytest.raises(cj.JudgeFailed) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, scoring="native"))
    assert exc.value.kind == "judge_unresolved"
    assert len(run.calls) == 1


def test_transport_error_is_failure_not_verdict(store, log_path):
    _calibrate(store)
    run = FakeRun({"native": _result(None, "native", failures=[("transport_error", "[ERROR: down]")])})
    judge, _ = _judge(store, log_path, run=run)
    with pytest.raises(cj.JudgeFailed) as exc:
        judge.judge(cj.JudgeRequest(**PAIR))
    assert exc.value.kind == "transport_error"
    assert _log(log_path)[0]["outcome"] == "failed"


def test_role_parked_is_a_refusal(store, log_path):
    from src.exceptions import RoleParkedError

    _calibrate(store)

    def parked(primitives, **kwargs):
        raise RoleParkedError(role="worker_general", port=8070, holder="autokernel", retry_after_s=90)

    judge, _ = _judge(store, log_path, run=parked)
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR))
    assert exc.value.kind == "role_parked" and exc.value.status_code == 503


# ── calibration gate ──────────────────────────────────────────────────────


def test_uncalibrated_judge_refused_before_inference(store, log_path):
    run = FakeRun({"native": _result("DEGRADED", "native")})
    judge, _ = _judge(store, log_path, run=run)
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR))
    assert exc.value.kind == "judge_uncalibrated"
    assert run.calls == []


def test_allow_uncalibrated_returns_null_calibration(store, log_path):
    judge, _ = _judge(store, log_path, run=FakeRun({"native": _result("OFF_TASK", "native")}))
    verdict = judge.judge(cj.JudgeRequest(**PAIR, allow_uncalibrated=True))
    assert verdict.verdict == "OFF_TASK"
    assert verdict.calibration_id is None and verdict.to_dict()["calibrated"] is False


def test_failed_calibration_record_is_not_used(store, log_path):
    _calibrate(store, passed=False)
    judge, _ = _judge(store, log_path, run=FakeRun({"native": _result("DEGRADED", "native")}))
    with pytest.raises(cj.JudgeRefused):
        judge.judge(cj.JudgeRequest(**PAIR))


def test_calibration_bound_to_served_model(store, log_path):
    _calibrate(store, served="some-other-model.gguf")
    judge, _ = _judge(store, log_path, run=FakeRun({"native": _result("DEGRADED", "native")}))
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR))
    assert exc.value.kind == "judge_uncalibrated"


def test_calibration_id_is_content_hash():
    a = cj.seal_calibration_record({"judge_key": "k", "passed": True, "created_at": "t"})
    b = cj.seal_calibration_record({"judge_key": "k", "passed": True, "created_at": "t"})
    c = cj.seal_calibration_record({"judge_key": "k", "passed": False, "created_at": "t"})
    assert a["calibration_id"] == b["calibration_id"] != c["calibration_id"]
    assert a["calibration_id"].startswith("cjcal-")


# ── measurement-window safety ─────────────────────────────────────────────


def test_local_refused_while_window_held_no_inference(store, log_path):
    _calibrate(store)
    run = FakeRun({"native": _result("DEGRADED", "native")})
    hold = WindowHold("cpu", "/x/cpu-window.json", "autokernel cpu window held", 60)
    judge, _ = _judge(store, log_path, run=run, holds=[hold])
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, allow_uncalibrated=True))
    assert exc.value.kind == "measurement_window_held"
    assert exc.value.status_code == 503 and exc.value.retry_after_s == 60
    assert run.calls == []
    assert _log(log_path)[0]["windows"][0]["window"] == "cpu"


def test_cloud_backend_exempt_from_windows(store, log_path):
    spec = CloudJudgeSpec(name="codex-luna-low", transport="codex_exec", model="gpt-5.6-luna")
    cal_id, _ = _calibrate(store, backend="cloud:codex-luna-low", model="gpt-5.6-luna", mode="structured_output", served=None)
    seen = {}

    def cloud_run(argv, stdin, timeout, cwd):
        seen["argv"], seen["stdin"] = argv, stdin
        out = Path(argv[argv.index("-o") + 1])
        out.write_text(json.dumps({"verdict": "COHERENT_EQUIVALENT", "reason": "same answer"}))
        return 0, "", ""

    hold = WindowHold("gpu", "/x/mi210.json", "gpu measurement window held", 600)
    judge, _ = _judge(
        store, log_path, holds=[hold], registry={"codex-luna-low": spec}, cloud_run=cloud_run,
        run=FakeRun({}),
    )
    verdict = judge.judge(cj.JudgeRequest(**PAIR, backend="cloud:codex-luna-low"))
    assert verdict.passed and verdict.calibration_id == cal_id
    assert verdict.confidence is None and verdict.scoring_mode == "structured_output"
    assert verdict.reason == "same answer"
    assert "--output-schema" in seen["argv"] and "read-only" in seen["argv"]
    assert PAIR["candidate_output"] in seen["stdin"]


def test_unknown_cloud_judge_and_bad_backend(store, log_path):
    judge, _ = _judge(store, log_path, run=FakeRun({}))
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend="cloud:nope"))
    assert exc.value.kind == "unknown_cloud_judge" and exc.value.status_code == 400
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend="remote"))
    assert exc.value.kind == "invalid_request"


def test_cloud_disabled_by_env(store, log_path, monkeypatch):
    monkeypatch.setenv(cj.CLOUD_ENV, "0")
    spec = CloudJudgeSpec(name="sonnet-low", transport="claude_cli", model="sonnet")
    judge, _ = _judge(store, log_path, registry={"sonnet-low": spec}, run=FakeRun({}))
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend="cloud:sonnet-low", allow_uncalibrated=True))
    assert exc.value.kind == "cloud_disabled"


def test_cloud_model_override_must_be_registered(store, log_path):
    spec = CloudJudgeSpec(name="codex-luna-low", transport="codex_exec", model="gpt-5.6-luna", models=("gpt-5.6-terra",))
    judge, _ = _judge(store, log_path, registry={"codex-luna-low": spec}, run=FakeRun({}))
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR, backend="cloud:codex-luna-low", model="gpt-9", allow_uncalibrated=True))
    assert exc.value.kind == "invalid_request"


# ── prompt material ───────────────────────────────────────────────────────


def test_clip_keeps_head_and_tail():
    text = "HEAD " + "x" * 10_000 + " TAIL"
    clipped, cut = cj.clip(text, 1000)
    assert cut and clipped.startswith("HEAD") and clipped.endswith("TAIL")
    assert "characters elided" in clipped
    assert cj.clip("short", 1000) == ("short", False)


def test_state_marks_truncation_and_rubric():
    state, truncated = cj.build_state(
        cj.JudgeRequest(prompt="p", base_output="b", candidate_output="c" * 20_000, rubric="must cite file")
    )
    assert truncated == {"prompt": False, "base_output": False, "candidate_output": True}
    assert "EXTRA CRITERIA FROM THE CALLER:\nmust cite file" in state


def test_prompt_template_sha_is_stable():
    assert cj.prompt_template_sha256() == cj.prompt_template_sha256()
    assert len(cj.prompt_template_sha256()) == 64


# ── the real native runner (TD-29 single-token keys) ──────────────────────

_CUE = 990


class _Tokenizer:
    """Verdict labels are multi-token; A..Z/0..9 keys are single tokens; else one cue id."""

    def __init__(self):
        from src.typed_decisions.routing_replay import _CODE_ALPHABET

        self.vocab: dict[str, list[int]] = {}
        for i, label in enumerate(cj.VERDICTS):
            self.vocab[label] = [100 + i, 200 + i, 300 + i]
            self.vocab[" " + label] = [110 + i, 200 + i, 300 + i]
        for i, code in enumerate(_CODE_ALPHABET):
            self.vocab[code] = [3000 + 2 * i]
            self.vocab[" " + code] = [3001 + 2 * i]

    def __call__(self, text):
        return list(self.vocab.get(text, [_CUE]))


class _NativePrimitives:
    def __init__(self, meta):
        self._last_inference_meta = meta
        self.calls = []

    def llm_call(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return ""


def test_real_native_runner_rekeys_verdicts_and_reads_probs(store, log_path):
    tok = _Tokenizer()
    ids = {text: v[0] for text, v in tok.vocab.items() if len(v) == 1}

    def entry(text, p):
        return {"id": ids[text], "token": text, "bytes": list(text.encode()), "logprob": math.log(p)}

    # Key " B" = DEGRADED is emitted with p=0.6; A=0.3, C=0.1.
    answer = {**entry(" B", 0.6), "top_logprobs": [entry(" B", 0.6), entry(" A", 0.3), entry(" C", 0.1)]}
    cue_row = {"id": _CUE, "token": "", "logprob": 0.0, "top_logprobs": []}
    prims = _NativePrimitives({"completion_probabilities": [cue_row, answer]})
    _calibrate(store)
    judge = cj.CoherenceJudge(
        primitives_fn=lambda: prims,
        holds_fn=lambda: [],
        store=store,
        cloud_registry_fn=lambda: {},
        served_model_fn=lambda p, r: "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        tokenize_fn=tok,
        log_path=log_path,
    )
    verdict = judge.judge(cj.JudgeRequest(**PAIR, scoring="native"))
    assert verdict.verdict == "DEGRADED"
    assert verdict.confidence == pytest.approx(0.6)
    assert verdict.probabilities["COHERENT_EQUIVALENT"] == pytest.approx(0.3)
    assert verdict.probabilities["OFF_TASK"] == pytest.approx(0.0)
    (call,) = prims.calls
    assert "A = COHERENT_EQUIVALENT | B = DEGRADED | C = INCOHERENT | D = OFF_TASK" in call["prompt"]
    assert call["temperature"] == 0.0


def test_in_band_role_parked_sentinel_is_a_refusal(store, log_path):
    from src.exceptions import RoleParkedError

    _calibrate(store)
    sentinel = "[ERROR: " + str(
        RoleParkedError(role="architect_critic", port=8083, holder="autokernel", retry_after_s=300)
    ) + "]"
    run = FakeRun({"native": _result(None, "native", failures=[("transport_error", sentinel)])})
    judge, _ = _judge(store, log_path, run=run)
    with pytest.raises(cj.JudgeRefused) as exc:
        judge.judge(cj.JudgeRequest(**PAIR))
    assert exc.value.kind == "role_parked" and exc.value.retry_after_s == 300
    assert _log(log_path)[0]["outcome"] == "refused"
