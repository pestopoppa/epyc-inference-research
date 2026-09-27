"""Hard-fail scorer semantics (audit A1 + A2 scorer-identity leg).

A requested scorer that cannot run — missing math_verify, a down llm_judge,
an unparseable GOLD answer, an unknown programmatic verifier — must surface
loudly (ScoringUnavailableError / ValueError), NEVER silently degrade to a
different scorer and score a wrong answer as (in)correct. The catastrophic
prior bug: math_verify's signal.alarm raises ValueError off the main thread,
the bare ``except Exception -> _score_exact_match`` ate it, and every threaded
math eval was scored with the wrong scorer.
"""

from __future__ import annotations

import concurrent.futures
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from debug_scorer import ScoringUnavailableError, score_answer  # noqa: E402


# ── math_verify: threaded correctness + no silent fallback ───────────────


def test_math_verify_scores_correctly_in_worker_thread() -> None:
    pytest.importorskip("math_verify")

    def _run(ans: str) -> bool:
        return score_answer(
            answer=ans,
            expected="\\frac{1}{2}",
            scoring_method="math_verify",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        correct = ex.submit(_run, "The result is \\boxed{\\frac{1}{2}}").result()
        wrong = ex.submit(_run, "The result is \\boxed{\\frac{1}{3}}").result()

    # If the thread/signal ValueError were still being swallowed into
    # exact_match, the correct \frac{1}{2} answer would NOT score True.
    assert correct is True
    assert wrong is False


def test_math_verify_thread_signal_error_raises_not_silent(monkeypatch) -> None:
    # Mocked variant (no real math_verify needed): parse() raises the exact
    # thread/signal ValueError math_verify raises off the main thread. The
    # scorer MUST surface ScoringUnavailableError, not a silent exact_match.
    stub = types.ModuleType("math_verify")

    def _parse(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ValueError("signal only works in main thread of the main interpreter")

    def _verify(gold, pred):  # pragma: no cover - never reached
        return True

    stub.parse = _parse
    stub.verify = _verify
    monkeypatch.setitem(sys.modules, "math_verify", stub)

    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="The result is \\boxed{\\frac{1}{2}}",
            expected="\\frac{1}{2}",
            scoring_method="math_verify",
        )


def test_math_verify_missing_dependency_raises(monkeypatch) -> None:
    # sys.modules[name] = None makes `from math_verify import ...` ImportError.
    monkeypatch.setitem(sys.modules, "math_verify", None)
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="\\boxed{2}",
            expected="2",
            scoring_method="math_verify",
        )


def test_math_verify_gold_parse_failure_raises() -> None:
    # A GOLD answer that math_verify extracts nothing from is a dataset/gold
    # defect -> ScoringUnavailableError (an item that cannot be scored), not a
    # False that would mask the defect as a model miss.
    pytest.importorskip("math_verify")
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="\\boxed{2}",
            expected="\\begin{invalid",
            scoring_method="math_verify",
        )


def test_pred_garbage_scores_false() -> None:
    # The MODEL's answer failing to parse is a task failure -> False, NOT
    # scorer-unavailability. GOLD parses fine here.
    pytest.importorskip("math_verify")
    result = score_answer(
        answer="the quick brown fox says hello",
        expected="42",
        scoring_method="math_verify",
    )
    assert result is False


# ── llm_judge: unreachable judge must raise, not substring-fallback ──────


def test_llm_judge_unreachable_raises() -> None:
    # Judge on 127.0.0.1:1 is unreachable; expected string is NOT a substring
    # of the answer, so the top-of-function substring fast-path does not fire.
    # The transport failure must raise, not silently fall back to substring.
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="the model said something entirely different",
            expected="mg/2",
            scoring_method="llm_judge",
            scoring_config={
                "judge_port": 1,
                "judge_host": "127.0.0.1",
                "timeout": 2,
            },
        )


def test_llm_judge_fast_path_is_boundary_aware() -> None:
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="the word concatenate appears here",
            expected="cat",
            scoring_method="llm_judge",
            scoring_config={
                "judge_port": 1,
                "judge_host": "127.0.0.1",
                "timeout": 2,
            },
        )


# ── llm_judge endpoint resolution: realized-first, no dead hardcoded port ────


def test_llm_judge_default_resolves_to_orchestrator_not_hardcoded_8082(monkeypatch) -> None:
    """The old default (localhost:8082, a dead worker_general *quarter* port on a
    quarters-only stack) is gone: with no explicit host/port the judge resolves
    to the ORCHESTRATOR API, which routes to a LIVE backend itself.
    """
    from debug_scorer import _resolve_llm_judge_base_url

    monkeypatch.delenv("ORCHESTRATOR_API_URL", raising=False)
    url = _resolve_llm_judge_base_url({})
    assert url == "http://localhost:8000"
    assert "8082" not in url


def test_llm_judge_default_honors_orchestrator_api_url_env(monkeypatch) -> None:
    from debug_scorer import _resolve_llm_judge_base_url

    monkeypatch.setenv("ORCHESTRATOR_API_URL", "http://localhost:18099/")
    assert _resolve_llm_judge_base_url({}) == "http://localhost:18099"


def test_llm_judge_explicit_overrides_win() -> None:
    from debug_scorer import _resolve_llm_judge_base_url

    # judge_url wins over everything.
    assert _resolve_llm_judge_base_url({"judge_url": "http://h:9/"}) == "http://h:9"
    # host+port (BOTH) take the legacy direct path.
    assert (
        _resolve_llm_judge_base_url({"judge_host": "127.0.0.1", "judge_port": 8200})
        == "http://127.0.0.1:8200"
    )


def test_llm_judge_unreachable_default_reports_resolved_url(monkeypatch) -> None:
    """A down orchestrator judge still raises honestly and names the endpoint."""
    monkeypatch.setenv("ORCHESTRATOR_API_URL", "http://127.0.0.1:1")
    with pytest.raises(ScoringUnavailableError) as excinfo:
        score_answer(
            answer="the model said something entirely different",
            expected="mg/2",
            scoring_method="llm_judge",
            scoring_config={"timeout": 2},
        )
    assert "127.0.0.1:1" in str(excinfo.value)
    assert "llm_judge_transport_error" in str(excinfo.value)


# ── llm_judge: protocol per endpoint (orchestrator /chat vs raw llama) ───
#
# Regression for the "malformed response at http://localhost:8000" failure:
# the judge spoke the raw llama OpenAI protocol (/v1/chat/completions,
# choices[].message) at the ORCHESTRATOR API, whose native eval ingress is
# POST /chat returning `answer`. Healthy orchestrator replies then parsed as
# "malformed" and every physreason/zeroscrolls judge row error-excluded.


class _FakeResp:
    """Minimal httpx.Response stand-in for mocked judge calls."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                "err",
                request=None,
                response=None,  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self._payload


def _patch_httpx_post(monkeypatch, payload: dict, status: int = 200) -> list[dict]:
    """Patch httpx.post to record calls and return a fixed fake response."""
    import httpx

    calls: list[dict] = []

    def _fake_post(url, json=None, timeout=None, **kw):  # noqa: A002
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResp(payload, status)

    monkeypatch.setattr(httpx, "post", _fake_post)
    return calls


def test_llm_judge_orchestrator_shape_success(monkeypatch) -> None:
    """Default (no override) => orchestrator /chat + parse `answer`."""
    monkeypatch.delenv("ORCHESTRATOR_API_URL", raising=False)
    monkeypatch.delenv("LLM_JUDGE_ROLE", raising=False)
    calls = _patch_httpx_post(monkeypatch, {"answer": "true"})

    result = score_answer(
        answer="the student wrote something else entirely",
        expected="mg/2",
        scoring_method="llm_judge",
        scoring_config={"timeout": 5, "_eval_batch_id": "evaltower-test-100q"},
    )
    assert result is True
    assert len(calls) == 1
    assert calls[0]["url"] == "http://localhost:8000/chat"
    body = calls[0]["json"]
    assert body["real_mode"] is True
    assert body["mock_mode"] is False
    assert body["force_mode"] == "direct"
    assert body["workload_class"] == "eval_batch"
    # 2026-09-27 ARCHITECT SWAP: the default GPU judge (the :8083 27B) is
    # architect_critic.
    assert body["force_role"] == "architect_critic"
    assert body["batch_id"] == "evaltower-test-100q"
    assert body["client_deadline_unix_s"] > 0
    assert body["max_queue_wait_ms"] == 1250
    assert body["output_schema"] == {"type": "boolean"}
    assert calls[0]["timeout"] == 10.0
    assert "prompt" in body and "messages" not in body


def test_llm_judge_preserves_full_long_form_candidate(monkeypatch) -> None:
    """Summary/table rows must not be silently reduced to their last line."""
    monkeypatch.delenv("ORCHESTRATOR_API_URL", raising=False)
    calls = _patch_httpx_post(monkeypatch, {"answer": "true"})
    candidate = "Central claim and evidence.\n| final table row |"

    assert (
        score_answer(
            answer=candidate,
            expected="Central claim supported by evidence.",
            scoring_method="llm_judge",
            scoring_config={"timeout": 5},
        )
        is True
    )

    prompt = calls[0]["json"]["prompt"]
    assert candidate in prompt
    assert "strict semantic answer judge" in prompt
    assert "physics answer equivalence judge" not in prompt


def test_llm_judge_orchestrator_shape_false_verdict(monkeypatch) -> None:
    monkeypatch.delenv("ORCHESTRATOR_API_URL", raising=False)
    _patch_httpx_post(monkeypatch, {"answer": "false"})
    result = score_answer(
        answer="the student wrote something else entirely",
        expected="mg/2",
        scoring_method="llm_judge",
        scoring_config={"timeout": 5},
    )
    assert result is False


def test_llm_judge_orchestrator_honors_judge_role(monkeypatch) -> None:
    monkeypatch.delenv("ORCHESTRATOR_API_URL", raising=False)
    calls = _patch_httpx_post(monkeypatch, {"answer": "true"})
    score_answer(
        answer="the student wrote something else entirely",
        expected="mg/2",
        scoring_method="llm_judge",
        scoring_config={"timeout": 5, "judge_role": "thinking_reasoning"},
    )
    assert calls[0]["json"]["force_role"] == "thinking_reasoning"


def test_llm_judge_orchestrator_error_body_raises(monkeypatch) -> None:
    """200-with-error (or empty answer) is scorer-unavailability, not False."""
    monkeypatch.delenv("ORCHESTRATOR_API_URL", raising=False)
    _patch_httpx_post(monkeypatch, {"answer": "", "error": "backend unavailable"})
    with pytest.raises(ScoringUnavailableError) as excinfo:
        score_answer(
            answer="the student wrote something else entirely",
            expected="mg/2",
            scoring_method="llm_judge",
            scoring_config={"timeout": 5},
        )
    assert "llm_judge_backend_error" in str(excinfo.value)
    assert "backend unavailable" in str(excinfo.value)


def test_llm_judge_llama_shape_success_via_override(monkeypatch) -> None:
    """Explicit judge_host+judge_port => raw llama /v1/chat/completions."""
    calls = _patch_httpx_post(monkeypatch, {"choices": [{"message": {"content": "true"}}]})
    result = score_answer(
        answer="the student wrote something else entirely",
        expected="mg/2",
        scoring_method="llm_judge",
        scoring_config={"judge_host": "127.0.0.1", "judge_port": 8200, "timeout": 5},
    )
    assert result is True
    assert calls[0]["url"] == "http://127.0.0.1:8200/v1/chat/completions"
    body = calls[0]["json"]
    assert "messages" in body and "prompt" not in body


def test_llm_judge_llama_shape_malformed_raises(monkeypatch) -> None:
    """Override target returning a non-llama body => malformed => raises."""
    _patch_httpx_post(monkeypatch, {"unexpected": "shape"})
    with pytest.raises(ScoringUnavailableError) as excinfo:
        score_answer(
            answer="the student wrote something else entirely",
            expected="mg/2",
            scoring_method="llm_judge",
            scoring_config={
                "judge_url": "http://judge.local:9",
                "timeout": 5,
            },
        )
    assert "llm_judge_unexpected_shape" in str(excinfo.value)


# ── programmatic: unknown verifier is a config defect ────────────────────


def test_unknown_programmatic_verifier_raises() -> None:
    with pytest.raises(ValueError):
        score_answer(
            answer="anything at all",
            expected="whatever",
            scoring_method="programmatic",
            scoring_config={"verifier": "definitely_not_a_verifier"},
        )


# ── A2 leg: seeding_scoring pins the orchestrator debug_scorer copy ──────


def test_seeding_scoring_binds_orchestrator_copy() -> None:
    research_bench = "/mnt/raid0/llm/epyc-inference-research/scripts/benchmark"
    orch_bench = str(REPO_ROOT / "scripts" / "benchmark")

    saved_path = list(sys.path)
    watched_keys = (
        "debug_scorer",
        "epyc_orch_debug_scorer",
        "seeding_scoring",
        "seeding_types",
    )
    saved_modules = {k: sys.modules.get(k) for k in watched_keys}
    try:
        # Force a fresh bare `import debug_scorer` to bind the RESEARCH copy,
        # so we prove score_answer_deterministic ignores that binding and
        # pins the orchestrator copy under its private key.
        sys.modules.pop("debug_scorer", None)
        sys.modules.pop("epyc_orch_debug_scorer", None)
        sys.path.insert(0, research_bench)
        try:
            import debug_scorer  # noqa: F401
        except Exception:
            pass

        sys.path.insert(0, orch_bench)
        import seeding_scoring  # noqa: E402

        # Trivial exact_match: last-line "42" numerically equals expected "42".
        result = seeding_scoring.score_answer_deterministic("42", "42", "exact_match")
        assert result is True

        bound = sys.modules["epyc_orch_debug_scorer"]
        assert bound.__file__ is not None
        assert bound.__file__.startswith(orch_bench)
        # The pin's contract is same-directory, not a repo-name substring
        # (which breaks in worktree checkouts): the bound scorer must be the
        # sibling debug_scorer.py of the seeding_scoring module that ran.
        assert (
            Path(bound.__file__).resolve().parent == Path(seeding_scoring.__file__).resolve().parent
        )
    finally:
        sys.path[:] = saved_path
        for key, val in saved_modules.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val
