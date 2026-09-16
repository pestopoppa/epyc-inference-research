#!/usr/bin/env python3
"""M-12 B1: a 200ch Tulving result can never be graded against 20ch gold.

The two books reuse one row-index space. Before B1 the question id was
``tulving_<variant>_ch-001_q<idx>`` for both, the harness and the scorer both
defaulted to the 20ch book, and a 200ch run would have been scored against the
wrong questions. These tests build a miniature on-disk dataset with the same
collision and check every link: adapter -> suite -> result row -> scorer.

Offline: no socket, no model. The server is a fake object.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import score_tulving_run  # noqa: E402
import tulving_episodic_adapter as tea  # noqa: E402
from score_tulving_run import (  # noqa: E402
    GoldBindingError,
    build_prompt_index,
    resolve_run_chapters,
    score_result_payload,
)
from tulving_episodic_adapter import (  # noqa: E402
    CONTEXT_FULL,
    CONTEXT_NONE,
    TulvingEpisodicAdapter,
    legacy_question_id,
    parse_question_id,
)

VARIANT = "Udefault_Sdefault_seed0"


def _book(n: int, tag: str) -> str:
    return "\n\n".join(f"Chapter {i}\n\n{tag} event {i} happened in {tag}ville." for i in range(1, n + 1))


def _rows(tag: str, count: int) -> list[dict]:
    # Same row count and shape in both books: indices collide, questions differ.
    return [
        {
            "question": f"[{tag}] Where did event {i} happen?",
            "correct_answer": [f"{tag}ville"] if i % 2 else [],
            "retrieval_type": "Spaces",
            "get": "all",
            "cue": f"(*, *, event {i}, *)",
            "chapter": -1,
            "n_chapters_correct_answer": 1 if i % 2 else 0,
        }
        for i in range(count)
    ]


def _write_book(root: Path, chapters_on_disk: int, tag: str, count: int) -> None:
    book_dir = (root / VARIANT / "books"
                / f"model_claude_itermax_10_Idefault_nbchapters_{chapters_on_disk}_nbtokens_1")
    book_dir.mkdir(parents=True)
    pd.DataFrame(_rows(tag, count)).to_parquet(book_dir / "df_qa.parquet")
    (book_dir / "book.json").write_text(json.dumps(_book(3, tag)))


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    root = tmp_path / "tulving"
    _write_book(root, 19, "short", 4)     # the "20ch" book (19 chapters on disk)
    _write_book(root, 196, "long", 6)     # the "200ch" book (196 chapters on disk)
    real_init = TulvingEpisodicAdapter.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("data_dir", root)
        real_init(self, *args, **kwargs)

    # The scorer constructs the adapter itself; point every construction at the fixture.
    monkeypatch.setattr(TulvingEpisodicAdapter, "__init__", init)
    monkeypatch.delenv(tea.CHAPTERS_ENV, raising=False)
    monkeypatch.delenv(tea.CONTEXT_MODE_ENV, raising=False)
    return root


def _items(chapters: int, mode: str = CONTEXT_FULL) -> list[dict]:
    return TulvingEpisodicAdapter(chapters=chapters, context_mode=mode).extract_all()


def _result(items: list[dict], *, recorded: bool = True, legacy_ids: bool = False,
            answer: str = "- longville") -> dict:
    """A result payload as run_benchmark writes it for these prompts."""
    rows = {}
    for item in items:
        qid = item["metadata"]["legacy_id"] if legacy_ids else item["id"]
        row = {"question_id": qid, "prompt": item["prompt"], "response": answer}
        if recorded:
            row["provenance"] = dict(item["provenance"])
        rows[qid] = row
    return {"run_id": "b1", "model_role": "ingest", "config_name": "baseline",
            "results": {"tulving_episodic": rows}}


# ── the adapter: the set is explicit, recorded, and part of the id ────────────

def test_the_two_books_collide_on_legacy_ids_but_not_on_new_ids(data_dir):
    short, long_ = _items(20), _items(200)
    legacy_short = {i["metadata"]["legacy_id"] for i in short}
    legacy_long = {i["metadata"]["legacy_id"] for i in long_}
    assert legacy_short <= legacy_long                    # the pre-B1 collision
    assert not ({i["id"] for i in short} & {i["id"] for i in long_})
    assert all("_20ch_" in i["id"] for i in short)
    assert all("_200ch_" in i["id"] for i in long_)


def test_provenance_names_the_set_and_the_book_actually_loaded(data_dir):
    item = _items(200)[0]
    prov = item["provenance"]
    assert prov["chapters"] == 200 and prov["chapter_set"] == "200ch"
    assert prov["book_chapters"] == 196
    assert prov["variant"] == VARIANT and prov["context_mode"] == CONTEXT_FULL
    assert len(prov["book_sha16"]) == 16
    assert parse_question_id(item["id"])["chapters"] == 200


def test_env_selects_the_set_for_the_harness(data_dir, monkeypatch):
    monkeypatch.setenv(tea.CHAPTERS_ENV, "200")
    assert TulvingEpisodicAdapter().extract_all()[0]["provenance"]["chapters"] == 200
    assert TulvingEpisodicAdapter(chapters=20).extract_all()[0]["provenance"]["chapters"] == 20


@pytest.mark.parametrize("bad", ["19", "196", "abc", "0"])
def test_unknown_sets_are_refused(bad, monkeypatch):
    monkeypatch.setenv(tea.CHAPTERS_ENV, bad)
    with pytest.raises(ValueError, match="chapters must be one of"):
        TulvingEpisodicAdapter()


def test_a_parquet_without_a_chapter_count_is_refused_not_used(tmp_path):
    variant_dir = tmp_path / VARIANT / "books" / "model_claude_unlabelled"
    variant_dir.mkdir(parents=True)
    pd.DataFrame(_rows("x", 1)).to_parquet(variant_dir / "df_qa.parquet")
    with pytest.raises(RuntimeError, match="refusing to guess"):
        TulvingEpisodicAdapter(data_dir=tmp_path, chapters=200)._ensure_loaded()


# ── the regression: 200ch answers are never graded against 20ch gold ──────────

def test_a_200ch_result_is_refused_by_20ch_gold(data_dir):
    payload = _result(_items(200))
    with pytest.raises(GoldBindingError, match="records 200ch, gold is 20ch"):
        score_result_payload(payload, build_prompt_index(20))


def test_a_200ch_result_scores_against_200ch_gold(data_dir):
    payload = _result(_items(200))
    assert resolve_run_chapters(payload, None) == 200
    summary = score_result_payload(payload, build_prompt_index(200))["summary"]
    assert summary["chapter_set"] == "200ch" and summary["book_chapters"] == 196
    assert summary["scored_questions"] == 6 and summary["missing_ground_truth"] == 0
    assert summary["row_binding"] == {"recorded": 6}


def test_a_pre_b1_200ch_result_is_refused_by_20ch_gold_despite_colliding_ids(data_dir):
    # The exact defect: legacy ids, which DO exist in the 20ch gold.
    payload = _result(_items(200), recorded=False, legacy_ids=True)
    gold = build_prompt_index(20)
    assert set(payload["results"]["tulving_episodic"]) & set(gold.by_legacy_id)
    with pytest.raises(GoldBindingError, match="not the 20ch gold prompt"):
        score_result_payload(payload, gold)


def test_a_pre_b1_20ch_result_is_verified_by_its_exact_prompt(data_dir):
    payload = _result(_items(20), recorded=False, legacy_ids=True, answer="- shortville")
    summary = score_result_payload(payload, build_prompt_index(20))["summary"]
    assert summary["row_binding"] == {"legacy_prompt_verified": 4}
    assert summary["scored_questions"] == 4


def test_a_pre_b1_row_without_the_book_can_never_be_bound(data_dir):
    payload = _result(_items(20, CONTEXT_NONE), recorded=False, legacy_ids=True)
    with pytest.raises(GoldBindingError, match="nothing proves which book"):
        score_result_payload(payload, build_prompt_index(20))


def test_a_mixed_run_dir_is_refused(data_dir):
    payload = _result(_items(20))
    payload["results"]["tulving_episodic"].update(_result(_items(200))["results"]["tulving_episodic"])
    with pytest.raises(GoldBindingError, match="several chapter sets"):
        resolve_run_chapters(payload, None)


def test_chapters_flag_cannot_override_the_recorded_set(data_dir):
    with pytest.raises(GoldBindingError, match="disagrees"):
        resolve_run_chapters(_result(_items(200)), 20)


def test_a_legacy_result_needs_an_explicit_set(data_dir):
    with pytest.raises(GoldBindingError, match="pass --chapters"):
        resolve_run_chapters(_result(_items(20), recorded=False, legacy_ids=True), None)


def test_a_changed_book_on_disk_is_refused(data_dir):
    payload = _result(_items(200))
    for row in payload["results"]["tulving_episodic"].values():
        row["provenance"]["book_sha16"] = "0" * 16
    with pytest.raises(GoldBindingError, match="book digest"):
        score_result_payload(payload, build_prompt_index(200))


def test_provenance_and_id_must_agree(data_dir):
    payload = _result(_items(200))
    for row in payload["results"]["tulving_episodic"].values():
        row["provenance"]["chapters"] = 20
    with pytest.raises(GoldBindingError, match="the id says 200"):
        resolve_run_chapters(payload, None)


def test_provenance_arm_must_match_the_stored_prompt(data_dir):
    payload = _result(_items(200))
    for row in payload["results"]["tulving_episodic"].values():
        row["provenance"]["context_mode"] = CONTEXT_NONE
    with pytest.raises(GoldBindingError, match="reads as 'full'"):
        score_result_payload(payload, build_prompt_index(200))


def test_main_refuses_a_200ch_result_with_chapters_20(data_dir, tmp_path, monkeypatch):
    result = tmp_path / "r.json"
    result.write_text(json.dumps(_result(_items(200))))
    monkeypatch.setattr("sys.argv", ["score_tulving_run.py", str(result), "--chapters", "20",
                                     "--out-json", str(tmp_path / "s.json")])
    with pytest.raises(SystemExit, match="disagrees"):
        score_tulving_run.main()
    assert not (tmp_path / "s.json").exists()


def test_main_scores_a_200ch_result_from_its_own_record(data_dir, tmp_path, monkeypatch):
    result = tmp_path / "r.json"
    result.write_text(json.dumps(_result(_items(200))))
    out = tmp_path / "s.json"
    monkeypatch.setattr("sys.argv", ["score_tulving_run.py", str(result), "--out-json", str(out)])
    assert score_tulving_run.main() == 0
    assert json.loads(out.read_text())["summary"]["chapters"] == 200


# ── the harness carries the record from adapter to result row ─────────────────

def test_suite_questions_carry_provenance_and_the_suite_its_params(data_dir, monkeypatch):
    import suites

    monkeypatch.setenv(tea.CHAPTERS_ENV, "200")
    suite = suites._load_adapter_suite("tulving_episodic")
    assert {q.provenance["chapters"] for q in suite.questions} == {200}
    params = suites.get_inference_params(suite)
    assert params["enable_thinking"] is False
    assert params["max_tokens"] == 1024 and params["temperature"] == 0.0


def test_result_rows_round_trip_provenance_and_tolerate_new_keys():
    from results import ModelConfigResult, QuestionResult

    row = QuestionResult(question_id="q", prompt="p", response="r",
                         provenance={"chapters": 200}, inference={"max_tokens": 1024},
                         finish_reason="stop")
    blob = ModelConfigResult("role", "m", "baseline", "run", "t",
                             results={"tulving_episodic": {"q": row}}).to_dict()
    blob["results"]["tulving_episodic"]["q"]["written_by_a_newer_harness"] = 1
    back = ModelConfigResult.from_dict(blob).results["tulving_episodic"]["q"]
    assert back.provenance == {"chapters": 200}
    assert back.inference == {"max_tokens": 1024} and back.finish_reason == "stop"


class _FakeServer:
    mmproj_path = None
    use_chat_api = False

    def __init__(self):
        self.calls = []

    def is_running(self):
        return True

    def run_inference(self, **kwargs):
        from lib.executor import InferenceResult

        self.calls.append(kwargs)
        return InferenceResult(raw_output="- longville\n\n", exit_code=0, command="fake",
                               finish_reason="length")


class _FakeResults:
    def __init__(self):
        self.rows = []

    def add_question_result(self, **kwargs):
        self.rows.append(kwargs)


def test_run_benchmark_records_provenance_and_generation_params(data_dir, monkeypatch):
    import run_benchmark
    import suites
    from lib.executor import Config

    monkeypatch.setenv(tea.CHAPTERS_ENV, "200")
    suite = suites._load_adapter_suite("tulving_episodic")
    params = suites.get_inference_params(suite)
    monkeypatch.setattr(run_benchmark, "result_exists", lambda *a, **k: False)
    ss = run_benchmark._ServerState()
    ss.server = _FakeServer()
    store = _FakeResults()
    stats = {"passed": 0, "errors": 0, "skipped": 0}
    run_benchmark._run_quality_question(
        executor=None, results_manager=store, ss=ss,
        config=Config(name="baseline", config_type="baseline"), model_path="/m.gguf",
        mmproj_path=None, role="ingest", run_id="r", suite_name="tulving_episodic",
        question=suite.questions[0], params=params, stats=stats, force=True, registry=None)

    assert stats == {"passed": 1, "errors": 0, "skipped": 0}
    call = ss.server.calls[0]
    assert call["enable_thinking"] is False and call["cache_prompt"] is True
    assert call["max_tokens"] == 1024 and call["temperature"] == 0.0
    row = store.rows[0]["question_result"]
    assert row.provenance["chapters"] == 200 and row.provenance["book_chapters"] == 196
    assert row.inference["endpoint"] == "chat_completions"
    assert row.inference["enable_thinking"] is False
    assert row.inference["enable_thinking_source"] == "suite"
    assert row.inference["max_tokens"] == 1024
    assert row.finish_reason == "length"
    # And the row it wrote binds to 200ch gold, never 20ch.
    payload = {"results": {"tulving_episodic": {row.question_id: {
        "prompt": row.prompt, "response": row.response, "provenance": row.provenance}}}}
    with pytest.raises(GoldBindingError):
        score_result_payload(payload, build_prompt_index(20))
    assert score_result_payload(payload, build_prompt_index(200))["summary"]["scored_questions"] == 1


def test_legacy_question_id_is_the_pre_b1_form():
    assert legacy_question_id(VARIANT, -1, 42) == f"tulving_{VARIANT}_ch-001_q0042"
    assert parse_question_id(legacy_question_id(VARIANT, -1, 42))["chapters"] is None
