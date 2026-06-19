#!/usr/bin/env python3
from __future__ import annotations

from score_tulving_run import chronological_tau, render_markdown, score_result_payload


def _prompt(question_id: str, ground_truth: list[str], get_style: str = "all") -> dict:
    return {
        "id": question_id,
        "metadata": {
            "ground_truth_items": ground_truth,
            "retrieval_type": "Times",
            "get_style": get_style,
        },
    }


def test_chronological_tau_perfect_and_reversed():
    prompt = _prompt("q", ["Jan 1", "Feb 1", "Mar 1"], get_style="chronological")
    assert chronological_tau("- Jan 1\n- Feb 1\n- Mar 1", prompt) == 1.0
    assert chronological_tau("- Mar 1\n- Feb 1\n- Jan 1", prompt) == -1.0


def test_score_result_payload_computes_composites():
    payload = {
        "run_id": "run",
        "model_role": "ingest_long_context",
        "config_name": "baseline",
        "results": {
            "tulving_episodic": {
                "q_latest": {
                    "response": "- Feb 1",
                    "tokens_per_second": 10.0,
                    "completion_tokens": 4,
                },
                "q_chrono": {
                    "response": "- Jan 1\n- Feb 1",
                    "tokens_per_second": 12.0,
                    "completion_tokens": 8,
                },
            }
        },
    }
    prompt_index = {
        "q_latest": _prompt("q_latest", ["Feb 1"], get_style="latest"),
        "q_chrono": _prompt("q_chrono", ["Jan 1", "Feb 1"], get_style="chronological"),
    }

    scored = score_result_payload(payload, prompt_index)
    summary = scored["summary"]

    assert summary["scored_questions"] == 2
    assert summary["missing_ground_truth"] == 0
    assert summary["simple_recall_score"] == 1.0
    assert summary["chronological_awareness_score"] == 1.0
    assert summary["avg_tokens_per_second"] == 11.0


def test_score_result_payload_tracks_missing_ground_truth():
    payload = {
        "run_id": "run",
        "results": {"tulving_episodic": {"missing": {"response": "- A"}}},
    }
    scored = score_result_payload(payload, {})
    assert scored["summary"]["scored_questions"] == 0
    assert scored["summary"]["missing_ground_truth"] == 1
    assert scored["missing_ground_truth_ids"] == ["missing"]


def test_render_markdown_includes_key_metrics(tmp_path):
    scored = {
        "summary": {
            "run_id": "run",
            "model_role": "ingest_long_context",
            "config_name": "baseline",
            "scored_questions": 2,
            "result_questions": 2,
            "missing_ground_truth": 0,
            "avg_f1": 0.5,
            "simple_recall_score": 0.6,
            "chronological_awareness_score": 0.7,
            "avg_tokens_per_second": 12.345,
            "by_retrieval_type": {"Times": {"count": 2, "avg_f1": 0.5}},
        }
    }
    md = render_markdown(scored, tmp_path / "result.json")
    assert "Simple Recall Score: 0.6000" in md
    assert "Chronological Awareness Score: 0.7000" in md
    assert "| Times | 2 | 0.5000 |" in md
