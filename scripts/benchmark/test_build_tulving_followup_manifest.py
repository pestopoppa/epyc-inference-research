#!/usr/bin/env python3
from __future__ import annotations

from build_tulving_followup_manifest import build_followup_records, render_markdown, summarize_records


def _row(
    question_id: str,
    *,
    retrieval_type: str = "Entities",
    get_style: str = "all",
    f1: float = 0.0,
    nb_gt: int = 1,
    nb_pred: int = 1,
    kendall_tau: float | None = None,
) -> dict:
    row = {
        "question_id": question_id,
        "retrieval_type": retrieval_type,
        "get_style": get_style,
        "f1": f1,
        "precision": f1,
        "recall": f1,
        "nb_gt": nb_gt,
        "nb_pred": nb_pred,
        "ground_truth_items": ["A"] if nb_gt else [],
        "matched_gt_items": [],
    }
    if kendall_tau is not None:
        row["kendall_tau"] = kendall_tau
    return row


def test_build_followup_records_selects_three_failure_foci():
    scored = {
        "per_question": [
            _row("zero_big", nb_gt=0, nb_pred=9),
            _row("zero_small", nb_gt=0, nb_pred=2),
            _row("event_bad", retrieval_type="Event contents", f1=0.1, nb_gt=2),
            _row("event_good", retrieval_type="Event contents", f1=0.9, nb_gt=2),
            _row("detail_bad", retrieval_type="Full event details", f1=0.0, nb_gt=1),
            _row("chrono_bad", get_style="chronological", f1=0.4, nb_gt=2, kendall_tau=-1.0),
            _row("chrono_ok", get_style="chronological", f1=1.0, nb_gt=2, kendall_tau=1.0),
        ]
    }
    prompts = {
        "zero_big": {"prompt": "prompt text", "metadata": {"chapter": 1}},
    }

    records = build_followup_records(scored, prompt_index=prompts, max_per_focus=2)
    counts = summarize_records(records)["by_focus"]

    assert counts == {
        "chronology_order": 1,
        "event_content_recall": 2,
        "zero_answer_abstention": 2,
    }
    assert records[0]["question_id"] == "zero_big"
    assert records[0]["prompt"] == "prompt text"
    assert records[0]["recommended_contract"].startswith("Return exactly")
    assert all(record["question_id"] != "event_good" for record in records)
    assert all(record["question_id"] != "chrono_ok" for record in records)


def test_render_markdown_summarizes_source_and_focus_counts(tmp_path):
    scored = {
        "summary": {
            "run_id": "run",
            "model_role": "ingest_long_context",
            "avg_f1": 0.43,
            "simple_recall_score": 0.55,
            "chronological_awareness_score": 0.16,
        }
    }
    records = [
        {"focus": "zero_answer_abstention"},
        {"focus": "event_content_recall"},
        {"focus": "event_content_recall"},
    ]

    markdown = render_markdown(scored, records, score_path=tmp_path / "score.json")

    assert "Run ID: `run`" in markdown
    assert "Prompt text included: no" in markdown
    assert "Source avg F1: 0.4300" in markdown
    assert "| event_content_recall | 2 |" in markdown
    assert "not a promotion gate by itself" in markdown
