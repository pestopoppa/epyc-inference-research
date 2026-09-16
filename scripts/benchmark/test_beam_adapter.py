#!/usr/bin/env python3
"""CME-1 / CME-2 / SC68 — BEAMAdapter, the BEAM fold contract, and the score-time hook.

Offline only: the dataset is a synthetic fixture shaped like HF ``Mohammadta/BEAM``
(features read from the dataset card, sha 3205395e) and the GitHub repo tree; the
judge is a stub callable. No socket is opened and no model is loaded.

Run with:
    /mnt/raid0/llm/delta-Mem/.venv/bin/python -m pytest scripts/benchmark/test_beam_adapter.py -q
"""

from __future__ import annotations

import builtins
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import beam_scoring  # noqa: E402
import score_beam_run  # noqa: E402
from beam_scoring import (  # noqa: E402
    BEAM_ABILITIES,
    BEAMFoldError,
    NuggetVerdictError,
    build_nugget_judge_prompt,
    fold_beam,
    judge_question,
    parse_nugget_verdict,
)
from long_context_adapters import BEAMAdapter, BEAMLoadError  # noqa: E402

# ── fixture: the HF schema, two conversations ────────────────────────────────

#: Rubric sizes per ability (question 0, question 1) — deliberately unequal so a
#: rubric-item micro-average and the per-question macro fold disagree.
RUBRIC_SIZES = {
    "abstention": (1, 1),
    "contradiction_resolution": (4, 4),
    "event_ordering": (3, 5),
    "information_extraction": (1, 6),
    "instruction_following": (1, 1),
    "knowledge_update": (1, 1),
    "multi_session_reasoning": (2, 4),
    "preference_following": (2, 2),
    "summarization": (5, 5),
    "temporal_reasoning": (2, 2),
}


def _probing_questions(conv: str) -> dict:
    out = {}
    for a_index, (ability, sizes) in enumerate(RUBRIC_SIZES.items()):
        out[ability] = [
            {
                "question": f"[{conv}] {ability} question {i}?",
                "answer" if ability != "abstention" else "ideal_response":
                    f"reference for {ability} {i}",
                "difficulty": ("easy", "medium", "hard")[(a_index + i) % 3],
                "rubric": [f"LLM response should state: {ability} fact {i}.{k}"
                           for k in range(n)],
            }
            for i, n in enumerate(sizes)
        ]
    return out


def _chat(conv: str) -> list[list[dict]]:
    return [
        [
            {"role": "user", "id": 0, "time_anchor": "March-15-2024", "index": "1,1",
             "question_type": "main_question", "content": f"hello from {conv}"},
            {"role": "assistant", "id": 1, "time_anchor": None, "index": None,
             "question_type": None, "content": "hi, how can I help?"},
        ],
        [
            {"role": "user", "id": 2, "time_anchor": "March-16-2024", "index": "1,2",
             "question_type": "main_question", "content": "my sprint ends March 29"},
            {"role": "assistant", "id": 3, "time_anchor": None, "index": None,
             "question_type": None, "content": "noted"},
        ],
    ]


def _hf_rows() -> list[dict]:
    return [
        {
            "conversation_id": conv,
            "conversation_seed": {"category": "Coding", "id": int(conv), "subtopics": ["x"],
                                  "theme": "t", "title": "budget tracker"},
            "narratives": "labels",
            "user_profile": {"user_info": "u", "user_relationships": "r"},
            "conversation_plan": "plan",
            "user_questions": [{"messages": [["q"]], "time_anchor": "March-15-2024"}],
            "chat": _chat(conv),
            # A STRING, exactly as the card warns — Python-literal repr.
            "probing_questions": repr(_probing_questions(conv)),
        }
        for conv in ("1", "2")
    ]


def write_parquet_fixture(root: Path) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = root / "data"
    data.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(_hf_rows()), data / "100K-00000-of-00001.parquet")
    return root


def write_repo_fixture(root: Path) -> Path:
    for conv in ("1", "2"):
        conv_dir = root / "chats" / "100K" / conv
        (conv_dir / "probing_questions").mkdir(parents=True)
        # GitHub chat.json shape: batches carrying turns.
        (conv_dir / "chat.json").write_text(json.dumps(
            [{"batch_number": 1, "turns": _chat(conv)}]))
        (conv_dir / "probing_questions" / "probing_questions.json").write_text(
            json.dumps(_probing_questions(conv)))
    return root


# ── CME-1: the adapter ───────────────────────────────────────────────────────


def test_parquet_fixture_yields_one_prompt_per_probing_question(tmp_path):
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path))
    prompts = adapter.extract_all()
    assert len(prompts) == 2 * 10 * 2
    assert adapter.source_kind == "hf_parquet"
    assert adapter.accounting_summary()["dropped_rows"] == 0
    assert {p["scoring_config"]["ability"] for p in prompts} == set(BEAM_ABILITIES)


def test_prompt_dict_carries_nuggets_for_the_served_judge(tmp_path):
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path))
    by_id = {p["id"]: p for p in adapter.extract_all()}
    p = by_id["beam_100K_1_summarization_1"]
    assert p["suite"] == "beam"
    assert p["scoring_method"] == "llm_judge"
    cfg = p["scoring_config"]
    assert cfg["judge_port"] == 8082
    assert cfg["per_nugget"] is True
    assert cfg["nuggets"] == [f"LLM response should state: summarization fact 1.{k}"
                              for k in range(5)]
    assert cfg["nugget_verdict_scale"] == [0.0, 0.5, 1.0]
    assert cfg["probing_question"] == "[1] summarization question 1?"
    assert cfg["fold_version"] == beam_scoring.FOLD_VERSION
    assert p["metadata"]["n_nuggets"] == 5
    assert p["expected"] == "reference for summarization 1"


def test_prompt_is_history_then_probing_question_as_next_turn(tmp_path):
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path))
    p = next(x for x in adapter.extract_all() if x["id"] == "beam_100K_2_abstention_0")
    text = p["prompt"]
    assert "[March-15-2024] User: hello from 2" in text
    assert "Assistant: noted" in text
    assert text.rstrip().endswith("User: [2] abstention question 0?")
    assert text.index("my sprint ends March 29") < text.index("abstention question 0")
    assert p["expected"] == "reference for abstention 0"  # ideal_response family
    assert "tau_reference" not in p["scoring_config"]


def test_event_ordering_prompts_name_their_tau_reference(tmp_path):
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path))
    eo = [p for p in adapter.extract_all() if p["scoring_config"]["ability"] == "event_ordering"]
    assert len(eo) == 4 and all(p["scoring_config"]["tau_reference"] == "nuggets" for p in eo)


def test_repo_tree_fixture_matches_parquet_prompts(tmp_path):
    from_parquet = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path / "hf")).extract_all()
    repo = BEAMAdapter(data_dir=write_repo_fixture(tmp_path / "repo"))
    from_repo = repo.extract_all()
    assert repo.source_kind == "repo_tree"

    def strip(p):
        return {**p, "metadata": {k: v for k, v in p["metadata"].items() if k != "beam_source"},
                "provenance": {k: v for k, v in p["provenance"].items() if k != "beam_source"}}

    assert [strip(p) for p in from_repo] == [strip(p) for p in from_parquet]


def test_missing_pyarrow_fails_loudly_never_zero_questions(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "100K-00000-of-00001.parquet").write_bytes(b"PAR1")
    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    adapter = BEAMAdapter(data_dir=tmp_path)
    with pytest.raises(BEAMLoadError, match="pyarrow"):
        adapter.extract_all()
    with pytest.raises(BEAMLoadError):
        adapter.sample(n=3)


def test_absent_data_is_a_recorded_degradation_with_the_download_size(tmp_path):
    adapter = BEAMAdapter(data_dir=tmp_path / "nothing-here")
    assert adapter.extract_all() == []
    degraded = adapter.accounting_summary()["degraded_sources"]
    assert len(degraded) == 1 and "5,429,768 B" in degraded[0]["error"]


def test_unparseable_probing_questions_fail_loudly(tmp_path):
    root = write_repo_fixture(tmp_path)
    (root / "chats/100K/2/probing_questions/probing_questions.json").write_text(
        json.dumps("{not a literal"))
    with pytest.raises(BEAMLoadError, match="neither a Python literal nor JSON"):
        BEAMAdapter(data_dir=root).extract_all()


def test_ability_drift_and_empty_rubric_fail_loudly(tmp_path):
    root = write_repo_fixture(tmp_path)
    pq_path = root / "chats/100K/1/probing_questions/probing_questions.json"
    pq = json.loads(pq_path.read_text())
    pq["information_update"] = pq.pop("knowledge_update")
    pq_path.write_text(json.dumps(pq))
    with pytest.raises(BEAMLoadError, match="ability keys drifted"):
        BEAMAdapter(data_dir=root).extract_all()

    pq["knowledge_update"] = pq.pop("information_update")
    pq["knowledge_update"][0]["rubric"] = []
    pq_path.write_text(json.dumps(pq))
    with pytest.raises(BEAMLoadError, match="non-empty rubric"):
        BEAMAdapter(data_dir=root).extract_all()


def test_unknown_split_is_refused():
    with pytest.raises(ValueError, match="split"):
        BEAMAdapter(split="128K")


def test_stratified_sample_uses_difficulty_tiers(tmp_path):
    adapter = BEAMAdapter(data_dir=write_repo_fixture(tmp_path))
    sample = adapter.sample(n=9, seed=1, stratify=True)
    assert len(sample) == 9
    assert {p["tier"] for p in sample} == {1, 2, 3}


# ── CME-1: the five registration points ──────────────────────────────────────


def test_registered_in_adapter_suites_and_dispatch():
    from dataset_adapters import ADAPTER_SUITES, _get_long_context_adapter, get_adapter
    assert "beam" in ADAPTER_SUITES
    assert _get_long_context_adapter("BEAMAdapter") is BEAMAdapter
    assert isinstance(get_adapter("beam"), BEAMAdapter)
    # The shared lazy-import bridge still resolves its original five.
    for name in ("longbench", "zeroscrolls", "leval", "ruler", "needle_parameterized"):
        assert get_adapter(name) is not None, name


def test_registered_in_role_suite_map():
    from suites import ROLE_SUITE_MAP
    assert "beam" in ROLE_SUITE_MAP["ingest"]
    assert "beam" in ROLE_SUITE_MAP["long_context"]


# ── CME-2: the fold contract ─────────────────────────────────────────────────


def _records(verdict_for) -> list[dict]:
    """One record per fixture question; ``verdict_for(ability, q, k)`` picks each verdict."""
    out = []
    for conv in ("1", "2"):
        for ability, sizes in RUBRIC_SIZES.items():
            for q, n in enumerate(sizes):
                out.append({
                    "question_id": f"beam_100K_{conv}_{ability}_{q}",
                    "ability": ability,
                    "nugget_verdicts": [verdict_for(ability, q, k) for k in range(n)],
                })
    return out


def test_all_half_run_scores_0500_not_1000():
    """The mutation this contract exists for: a >= 0.5 binarisation in the headline."""
    folded = fold_beam(_records(lambda *_: 0.5))
    assert folded["headline"] == pytest.approx(0.500)
    assert folded["headline"] != pytest.approx(1.000)
    diag = folded["secondary_diagnostics"]
    # The binarised fold is what reads 1.000 on this run — recorded, labelled, not the headline.
    assert diag["binarised_pass_rate"] == pytest.approx(1.0)
    assert diag["binarised_pass_count"] == diag["binarised_total_checks"] == folded["n_nuggets"]
    assert "not the BEAM fold" in diag["label"]


def test_headline_is_unweighted_mean_of_ability_columns():
    # Perfect on the three single-nugget abilities, zero everywhere else.
    ones = {"abstention", "instruction_following", "knowledge_update"}
    folded = fold_beam(_records(lambda a, q, k: 1.0 if a in ones else 0.0))
    assert folded["headline"] == pytest.approx(3 / 10)
    assert folded["abilities_reported"] == sorted(BEAM_ABILITIES)
    assert folded["abilities_missing"] == []
    # Rubric-item weighting: 3 abilities x 2 questions x 2 convs x 1 nugget = 12 of all items.
    total = sum(sum(s) for s in RUBRIC_SIZES.values()) * 2
    diag = folded["secondary_diagnostics"]
    assert diag["rubric_item_micro_average"] == pytest.approx(12 / total)
    assert diag["rubric_item_micro_average"] != pytest.approx(folded["headline"])


def test_per_question_is_nugget_mean_and_per_ability_is_question_mean():
    records = [
        {"question_id": "a", "ability": "summarization", "nugget_verdicts": [1.0, 0.0, 0.5, 0.5]},
        {"question_id": "b", "ability": "summarization", "nugget_verdicts": [1.0]},
        {"question_id": "c", "ability": "abstention", "nugget_verdicts": [0.0]},
    ]
    folded = fold_beam(records)
    scores = {q["question_id"]: q["score"] for q in folded["per_question"]}
    assert scores == {"a": 0.5, "b": 1.0, "c": 0.0}
    assert folded["per_ability"]["summarization"] == {"questions": 2, "score": 0.75}
    # Two reported columns: (0.75 + 0.0) / 2 — NOT the question-weighted (0.5+1+0)/3.
    assert folded["headline"] == pytest.approx(0.375)
    assert len(folded["abilities_missing"]) == 8
    diag = folded["secondary_diagnostics"]
    assert diag["rubric_item_micro_average"] == pytest.approx(3.0 / 6)
    assert (diag["binarised_pass_count"], diag["binarised_total_checks"]) == (4, 6)


def test_event_ordering_uses_tau_norm_when_present_and_reports_the_basis():
    base = {"ability": "event_ordering", "nugget_verdicts": [0.0, 0.0]}
    only_tau = fold_beam([{**base, "question_id": "x", "tau_norm": 0.8}])
    assert only_tau["per_ability"]["event_ordering"]["score"] == pytest.approx(0.8)
    assert only_tau["event_ordering_basis"] == "tau_norm"

    fallback = fold_beam([{**base, "question_id": "x"}])
    assert fallback["per_ability"]["event_ordering"]["score"] == 0.0
    assert fallback["event_ordering_basis"] == "nugget_mean_fallback"

    mixed = fold_beam([{**base, "question_id": "x", "tau_norm": 1.0},
                       {**base, "question_id": "y"}])
    assert mixed["event_ordering_basis"] == "mixed(1/2 tau_norm)"
    # tau_norm is ignored outside event ordering.
    other = fold_beam([{"question_id": "z", "ability": "abstention",
                        "nugget_verdicts": [0.0], "tau_norm": 1.0}])
    assert other["headline"] == 0.0


@pytest.mark.parametrize("bad", [0.25, 2, -0.5, True, "1.0", None, float("nan")])
def test_fold_refuses_off_scale_verdicts(bad):
    with pytest.raises(BEAMFoldError):
        fold_beam([{"question_id": "q", "ability": "abstention", "nugget_verdicts": [bad]}])


def test_fold_refuses_empty_duplicate_and_unknown_records():
    with pytest.raises(BEAMFoldError, match="non-empty"):
        fold_beam([{"question_id": "q", "ability": "abstention", "nugget_verdicts": []}])
    rec = {"question_id": "q", "ability": "abstention", "nugget_verdicts": [1.0]}
    with pytest.raises(BEAMFoldError, match="duplicate"):
        fold_beam([rec, dict(rec)])
    with pytest.raises(BEAMFoldError, match="unknown BEAM ability"):
        fold_beam([{**rec, "ability": "information_update"}])
    with pytest.raises(BEAMFoldError, match="tau_norm"):
        fold_beam([{**rec, "ability": "event_ordering", "tau_norm": 1.5}])


# ── the per-nugget judge contract ────────────────────────────────────────────


def test_judge_prompt_actually_contains_the_question():
    prompt = build_nugget_judge_prompt("When does my sprint end?", "states March 29", "March 29")
    assert "QUESTION (what the user asked): When does my sprint end?" in prompt
    assert "RUBRIC CRITERION (what to check): states March 29" in prompt
    assert "RESPONSE TO EVALUATE: March 29" in prompt
    assert "<question>" not in prompt and "<rubric_item>" not in prompt


@pytest.mark.parametrize("text,expected", [
    ('{"score": 1.0, "reason": "ok"}', 1.0),
    ('```json\n{"score": 0.5, "reason": "partial"}\n```', 0.5),
    ('Sure. {"score": "0.0", "reason": "no"}', 0.0),
    ('{"score": 1, "reason": "int"}', 1.0),
])
def test_parse_nugget_verdict_accepts_the_three_values(text, expected):
    assert parse_nugget_verdict(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", '{"reason": "x"}',
                                  '{"score": 0.7}', '{"score": "high"}', "{broken"])
def test_parse_nugget_verdict_raises_never_scores_zero(text):
    with pytest.raises(NuggetVerdictError):
        parse_nugget_verdict(text)


def test_judge_question_calls_the_judge_once_per_nugget(tmp_path):
    adapter = BEAMAdapter(data_dir=write_repo_fixture(tmp_path))
    prompt = next(p for p in adapter.extract_all() if p["id"] == "beam_100K_1_summarization_0")
    seen = []

    def judge(text):
        seen.append(text)
        return '{"score": 0.5, "reason": "stub"}'

    record = judge_question(prompt, "a response", judge)
    assert len(seen) == 5
    assert all("[1] summarization question 0?" in s for s in seen)
    assert record == {"question_id": prompt["id"], "ability": "summarization",
                      "nugget_verdicts": [0.5] * 5}


# ── score_beam_run: the folded artifact and the SC68 hook ────────────────────


def _judged_payload() -> dict:
    return {
        "run_id": "beam-fixture-run",
        "model_role": "ingest_long_context",
        "split": "100K",
        "judge_model": "stub-judge",
        "records": _records(lambda a, q, k: (0.0, 0.5, 1.0)[(q + k) % 3]),
    }


def test_score_judged_payload_records_both_folds_and_judge_identity(tmp_path):
    scored = score_beam_run.score_judged_payload(_judged_payload())
    s = scored["summary"]
    assert s["scorer_version"] == beam_scoring.FOLD_VERSION
    assert s["fold"] == "beam_macro" and s["headline"] is not None
    assert s["secondary_diagnostics"]["rubric_item_micro_average"] is not None
    assert s["secondary_diagnostics"]["binarised_pass_count"] >= 0
    assert s["judge_model"] == "stub-judge"
    assert s["judge_prompt_version"] == beam_scoring.JUDGE_PROMPT_VERSION
    assert s["question_in_judge_prompt"] is True
    assert len(scored["per_question"]) == 40


def test_score_judged_payload_checks_records_against_the_dataset(tmp_path):
    adapter = BEAMAdapter(data_dir=write_repo_fixture(tmp_path))
    index = score_beam_run.build_prompt_index(adapter)
    payload = _judged_payload()
    assert score_beam_run.score_judged_payload(payload, index)["summary"][
        "checked_against_dataset"] is True

    short = json.loads(json.dumps(payload))
    short["records"][0]["nugget_verdicts"].append(1.0)
    with pytest.raises(BEAMFoldError, match="verdicts for"):
        score_beam_run.score_judged_payload(short, index)

    stray = json.loads(json.dumps(payload))
    stray["records"][0]["question_id"] = "beam_100K_9_abstention_0"
    with pytest.raises(BEAMFoldError, match="dataset lacks"):
        score_beam_run.score_judged_payload(stray, index)


def _root_with_capture():
    for root in (os.environ.get("EPYC_ROOT"), "/mnt/raid0/llm/epyc-root", "/workspace"):
        if root and (Path(root) / "scripts/vidya/adapters/beam_memory_capture.py").is_file():
            return root
    return None


def test_load_belief_capture_explains_a_missing_root(tmp_path, monkeypatch):
    monkeypatch.setattr(score_beam_run, "_ROOT_CANDIDATES", (str(tmp_path / "nope"),))
    with pytest.raises(SystemExit, match="beam_memory_capture"):
        score_beam_run._load_belief_capture()


@pytest.mark.skipif(_root_with_capture() is None, reason="epyc-root beam capture not on this host")
def test_belief_sidecar_round_trips_through_the_root_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(score_beam_run, "_ROOT_CANDIDATES", (_root_with_capture(),))
    capture = score_beam_run._load_belief_capture()
    scored = score_beam_run.score_judged_payload(_judged_payload())
    out = tmp_path / "beam_score.json"
    out.write_text(json.dumps(scored, indent=2))
    sidecar = capture.write_belief_measurements(
        out, summary=scored["summary"], run_id="beam-fixture-run",
        producer="score_beam_run.py", arm="full")
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert capture.validate_row(row) == []
    assert row["value"] == scored["summary"]["headline"]
    assert row["extra"]["fold"] == "beam_macro"
    assert (row["extra"]["rubric_item_micro_average"]
            == scored["summary"]["secondary_diagnostics"]["rubric_item_micro_average"])
    assert (row["extra"]["binarised_pass_count"]
            == scored["summary"]["secondary_diagnostics"]["binarised_pass_count"])
