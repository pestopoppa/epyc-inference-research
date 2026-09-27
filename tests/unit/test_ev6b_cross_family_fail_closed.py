"""EV-6b: cross-family verification must fail CLOSED, not permissively.

Defect (`handoffs/active/eval-tower-verification.md`, task EV-6b):
``check_cross_family`` was vacuous on its only production call site — the
rubric-judge dispatch in ``eval_tower.py`` passes a ROLE NAME (e.g.
``"architect_general"``) as the verifier, no ``VERIFICATION_FAMILIES``
pattern matches a role name, the family resolved to ``"unknown"``, and the
old ``gen_family == "unknown"`` clause returned True (safe to proceed) for
EVERY generator. The ``llm_judge`` scoring path (``debug_scorer.py``) ran no
cross-family check at all.

This file pins, for ``src.autopilot_core.verification_families`` (the shared
classifier) and its two callers:

  * a role resolves to its served model and gets a REAL family (fixture
    registry, never the live one);
  * same-family generator+verifier -> not independent;
  * different, known families -> independent;
  * an unknown family on EITHER side -> not independent, labelled
    ``"unverified"`` (never confused with a confirmed ``"same_family"``
    finding, and never silently treated as safe);
  * every new family pattern (gpt-oss, glm, kimi, nemotron, minimax) is
    recognised, and the pre-existing ones are untouched;
  * the rubric-judge path (opt-in, ``AUTOPILOT_RUBRIC_JUDGE_ROLES``, default
    empty) still skips a non-independent judge role -> heuristic fallback;
  * the ``llm_judge`` path (default-on scoring method, 3.8k live rows across
    physreason/zeroscrolls/leval/physics) does NOT gate on this check — a
    fail-closed GATE there would turn every one of those rows into
    scoring_failed/excluded on the current all-Qwen lineup (the default
    judge role, ``architect_general``, resolves to the same family as every
    served generator). It instead LABELS the row
    (``QuestionResult.judge_independence``: "cross_family" / "same_family"
    / "unverified"), carries the label into the compacted/journaled row, and
    rolls it up as ``judge_independence_counts`` in the aggregate details —
    so a same-family judgment is visible and can never be misread as a
    confirmed cross-family verdict, without silently discarding the
    measurement.

Every httpx call is mocked. Nothing here opens a socket, and no test depends
on the live production registry's actual contents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "autopilot", REPO_ROOT / "scripts" / "benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import src.autopilot_core.verification_families as vf  # noqa: E402
import eval_tower  # noqa: E402
import debug_scorer  # noqa: E402


# A self-contained fixture registry — deliberately NOT the live
# orchestration/model_registry.yaml, per the task instruction that these
# tests must not depend on live registry contents.
FIXTURE_REGISTRY: dict[str, Any] = {
    "roles": {
        "architect_general": {"model": {"name": "Qwen3.8-27B-Q8_0"}},
        "worker_general": {"model": {"name": "Qwen3.8-27B-Q8_0"}},
        "llama_judge_role": {"model": {"name": "Llama-3.1-70B-Instruct"}},
        "gptoss_judge_role": {"model": {"name": "gpt-oss-120b"}},
        "role_with_no_model_block": {},
    }
}


@pytest.fixture(autouse=True)
def _fixture_registry(monkeypatch):
    """Every test in this module sees the fixture registry, never the live one."""
    monkeypatch.setattr(vf, "_registry", lambda: FIXTURE_REGISTRY)
    monkeypatch.setattr(vf, "_REGISTRY_CACHE", None)


# ── role -> served-model resolution ─────────────────────────────────────


def test_role_name_resolves_to_served_model_and_real_family():
    resolved = vf.resolve_family_input("architect_general")
    assert resolved == "Qwen3.8-27B-Q8_0"
    assert vf.classify_model_family(resolved) == "qwen"


def test_role_resolution_is_explicit_registry_override_too():
    # The lower-level API also accepts an explicit registry (used by callers
    # that already hold one), bypassing the module-level fixture.
    other_registry = {"roles": {"my_role": {"model": {"name": "Llama-3-8B"}}}}
    resolved = vf.resolve_family_input("my_role", registry=other_registry)
    assert resolved == "Llama-3-8B"


def test_already_resolved_model_name_passes_through_unchanged():
    # Not a registry key -> falls back to the input unchanged (idempotent on
    # an already-resolved model name/path).
    resolved = vf.resolve_family_input("Qwen3.8-27B-Q8_0")
    assert resolved == "Qwen3.8-27B-Q8_0"


def test_role_with_no_model_block_does_not_resolve():
    resolved = vf.resolve_family_input("role_with_no_model_block")
    assert resolved == "role_with_no_model_block"
    assert vf.classify_model_family(resolved) == "unknown"


# ── cross_family_status: the core fail-closed contract ──────────────────


def test_same_family_generator_and_verifier_not_independent():
    independent, status = vf.cross_family_status("architect_general", "worker_general")
    assert independent is False
    assert status == "same_family"


def test_different_known_families_are_independent():
    independent, status = vf.cross_family_status("architect_general", "llama_judge_role")
    assert independent is True
    assert status == "cross_family"


@pytest.mark.parametrize(
    "generator,verifier",
    [
        ("totally-unrecognised-model", "architect_general"),
        ("architect_general", "totally-unrecognised-model"),
        ("totally-unrecognised-model", "also-unrecognised"),
    ],
)
def test_unknown_family_on_either_side_is_unverified_not_independent(generator, verifier):
    independent, status = vf.cross_family_status(generator, verifier)
    assert independent is False
    assert status == "unverified"
    # The defining regression check: "unverified" must never collapse into
    # "same_family" (a different, confirmed finding) or read as independent.
    assert status != "same_family"


# ── family pattern coverage ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_name,expected_family",
    [
        ("gpt-oss-120b", "gpt-oss"),
        ("GPT-OSS-20B", "gpt-oss"),
        ("GLM-4.6-355B", "glm"),
        ("glm-4.5-air", "glm"),
        ("Kimi-K2-Instruct", "kimi"),
        ("kimi-linear", "kimi"),
        ("Nemotron-Nano-9B", "nemotron"),
        ("nemotron-ultra", "nemotron"),
        ("MiniMax-M2", "minimax"),
        ("minimax-text-01", "minimax"),
        # Pre-existing families must remain intact.
        ("Qwen3.8-27B-Q8_0", "qwen"),
        ("QwQ-32B", "qwen"),
        ("Meta-Llama-3.1-70B", "llama"),
        ("DeepSeek-V3", "deepseek"),
        ("Ouro-1.4B", "ouro"),
        ("Mistral-Large", "mistral"),
        ("Gemma-3-27B", "gemma"),
    ],
)
def test_new_and_existing_families_recognised(model_name, expected_family):
    assert vf.classify_model_family(model_name) == expected_family


def test_unrecognised_model_is_unknown():
    assert vf.classify_model_family("SomeFutureArchitecture-9000") == "unknown"


# ── eval_tower.py: the rubric-judge call site (the original defect) ─────


def test_eval_tower_check_cross_family_reexports_families():
    for family in ("gpt-oss", "glm", "kimi", "nemotron", "minimax", "qwen", "llama"):
        assert family in eval_tower.VERIFICATION_FAMILIES


def test_eval_tower_role_name_verifier_now_resolves_instead_of_permissive_default():
    # THE original defect, reproduced directly: a role name as verifier used
    # to resolve to "unknown" and the permissive default let it through
    # (True) for every generator. It must now correctly detect same-family.
    assert eval_tower.check_cross_family("architect_general", "worker_general") is False
    independent, status = eval_tower.check_cross_family_status(
        "architect_general", "worker_general"
    )
    assert (independent, status) == (False, "same_family")


def test_eval_tower_cross_family_role_pairing_is_independent():
    assert eval_tower.check_cross_family("architect_general", "llama_judge_role") is True
    independent, status = eval_tower.check_cross_family_status(
        "architect_general", "llama_judge_role"
    )
    assert (independent, status) == (True, "cross_family")


def test_eval_tower_unresolvable_role_fails_closed_not_permissive():
    # An unresolvable role must NOT fall back to the old permissive default.
    assert eval_tower.check_cross_family("architect_general", "no_such_role_at_all") is False


# ── debug_scorer.py: judge-role resolution is unchanged (no gate added) ──


def test_llm_judge_force_role_precedence_is_unchanged():
    # debug_scorer.py itself carries NO EV-6b logic — this pins that its
    # existing judge-role precedence (judge_role > LLM_JUDGE_ROLE env >
    # architect_general) is untouched, since eval_tower's label computation
    # (below) reuses it rather than restating it.
    # 2026-09-27 ARCHITECT SWAP: the default judge follows the MI210 27B process
    # ("the disjoint GPU judge"), which is architect_critic now. Only the label
    # moved; the precedence order under test is unchanged.
    assert debug_scorer._llm_judge_force_role({}) == debug_scorer.DEFAULT_LLM_JUDGE_ROLE
    assert debug_scorer.DEFAULT_LLM_JUDGE_ROLE == "architect_critic"
    assert debug_scorer._llm_judge_force_role({"judge_role": "worker_general"}) == "worker_general"


def test_eval_tower_verifier_identity_reuses_debug_scorer_precedence():
    assert eval_tower._llm_judge_verifier_identity({}) == debug_scorer._llm_judge_force_role({})
    cfg = {"judge_role": "llama_judge_role"}
    assert eval_tower._llm_judge_verifier_identity(cfg) == debug_scorer._llm_judge_force_role(cfg)


# ── eval_tower.py: the llm_judge scoring path is LABELLED, never gated ──


class _Resp:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


@pytest.fixture
def judge_recorder(monkeypatch):
    calls: list[dict[str, Any]] = []
    replies: list[dict[str, Any]] = []

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - mirrors httpx
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _Resp(replies.pop(0))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.delenv("ORCHESTRATOR_API_URL", raising=False)
    return calls, replies


def _llm_judge_outcome(
    *, question_id: str, generator_model: str, judge_role: str
) -> "eval_tower._GenOutcome":
    return eval_tower._GenOutcome(
        gen_ended_at_s=0.0,
        resp={"model": generator_model},
        answer="an unrelated long answer with no shared substring",
        error=None,
        tokens=5,
        elapsed=0.01,
        question_id=question_id,
        suite="unit",
        prompt="prompt",
        expected="reference",
        stable_qid=f"qid-{question_id}",
        scoring_method="llm_judge",
        scoring_config={"judge_role": judge_role},
        eval_partition="core",
    )


def test_llm_judge_row_scored_and_labelled_same_family(judge_recorder):
    calls, replies = judge_recorder
    replies.append({"answer": "true"})
    tower = eval_tower.EvalTower()
    outcome = _llm_judge_outcome(
        question_id="q-same",
        generator_model="architect_general",  # Qwen family
        judge_role="worker_general",  # also Qwen family in the fixture registry
    )
    result = tower._score_generation({"expected": "reference"}, outcome, client=None)
    assert len(calls) == 1  # scored NORMALLY — not blocked
    assert result.correct is True
    assert result.error is None
    assert result.judge_independence == "same_family"


def test_llm_judge_row_scored_and_labelled_cross_family(judge_recorder):
    calls, replies = judge_recorder
    replies.append({"answer": "true"})
    tower = eval_tower.EvalTower()
    outcome = _llm_judge_outcome(
        question_id="q-cross",
        generator_model="architect_general",  # Qwen family
        judge_role="llama_judge_role",  # Llama family
    )
    result = tower._score_generation({"expected": "reference"}, outcome, client=None)
    assert len(calls) == 1
    assert result.correct is True
    assert result.error is None
    assert result.judge_independence == "cross_family"


def test_llm_judge_row_scored_and_labelled_unverified(judge_recorder):
    calls, replies = judge_recorder
    replies.append({"answer": "true"})
    tower = eval_tower.EvalTower()
    outcome = _llm_judge_outcome(
        question_id="q-unverified",
        generator_model="totally-unrecognised-model",
        judge_role="architect_general",
    )
    result = tower._score_generation({"expected": "reference"}, outcome, client=None)
    assert len(calls) == 1  # STILL scored — unverified is a label, not a gate
    assert result.correct is True
    assert result.error is None
    assert result.judge_independence == "unverified"
    assert result.judge_independence != "same_family"


def test_judge_independence_label_reaches_compacted_row(judge_recorder):
    _calls, replies = judge_recorder
    replies.append({"answer": "true"})
    tower = eval_tower.EvalTower()
    outcome = _llm_judge_outcome(
        question_id="q-compact",
        generator_model="architect_general",
        judge_role="worker_general",
    )
    result = tower._score_generation({"expected": "reference"}, outcome, client=None)
    row = eval_tower._compact_question_result(result)
    assert row["judge_independence"] == "same_family"


def test_judge_independence_omitted_for_non_llm_judge_rows():
    outcome = eval_tower._GenOutcome(
        gen_ended_at_s=0.0,
        resp={"model": "architect_general"},
        answer="42",
        error=None,
        tokens=5,
        elapsed=0.01,
        question_id="q-exact",
        suite="unit",
        prompt="prompt",
        expected="42",
        stable_qid="qid-exact",
        scoring_method="exact_match",
        scoring_config={},
        eval_partition="core",
    )
    tower = eval_tower.EvalTower()
    result = tower._score_generation({"expected": "42"}, outcome, client=None)
    assert result.judge_independence == ""
    row = eval_tower._compact_question_result(result)
    assert "judge_independence" not in row


def test_judge_independence_counts_in_aggregate_details(judge_recorder):
    _calls, replies = judge_recorder
    replies.extend([{"answer": "true"}, {"answer": "true"}, {"answer": "true"}])
    tower = eval_tower.EvalTower()
    cases = [
        ("q-a", "architect_general", "worker_general"),  # same_family
        ("q-b", "architect_general", "llama_judge_role"),  # cross_family
        ("q-c", "totally-unrecognised-model", "architect_general"),  # unverified
    ]
    results = [
        tower._score_generation(
            {"expected": "reference"},
            _llm_judge_outcome(question_id=qid, generator_model=gen, judge_role=judge),
            client=None,
        )
        for qid, gen, judge in cases
    ]
    aggregated = tower._aggregate(results, tier=1)
    assert aggregated.details["judge_independence_counts"] == {
        "cross_family": 1,
        "same_family": 1,
        "unverified": 1,
    }
