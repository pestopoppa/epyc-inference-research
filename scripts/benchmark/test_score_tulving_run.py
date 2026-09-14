#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pytest

import score_tulving_run
from score_tulving_run import (
    SCORER_VERSION,
    _load_belief_capture,
    chronological_tau,
    chronological_tau_detail,
    render_markdown,
    score_result_payload,
)


def _prompt(
    question_id: str,
    ground_truth: list[str],
    get_style: str = "all",
    nb_events: int | None = None,
) -> dict:
    return {
        "id": question_id,
        "metadata": {
            "ground_truth_items": ground_truth,
            "retrieval_type": "Times",
            "get_style": get_style,
            "nb_events": len(ground_truth) if nb_events is None else nb_events,
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
    payload["results"]["tulving_episodic"]["q_all"] = {
        "response": "- Jan 1",
        "tokens_per_second": 11.0,
        "completion_tokens": 4,
    }
    prompt_index = {
        "q_latest": _prompt("q_latest", ["Feb 1"], get_style="latest"),
        "q_chrono": _prompt("q_chrono", ["Jan 1", "Feb 1"], get_style="chronological"),
        "q_all": _prompt("q_all", ["Jan 1"], get_style="all"),
    }

    scored = score_result_payload(payload, prompt_index)
    summary = scored["summary"]

    assert summary["scorer_version"] == SCORER_VERSION
    assert summary["scored_questions"] == 3
    assert summary["missing_ground_truth"] == 0
    # Simple Recall covers the "all" question ONLY — never the latest/chronological
    # legs, which have their own metric (M-12e).
    assert summary["simple_recall_questions"] == 1
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


# ── M-12e regression fixtures ────────────────────────────────────────────────
#
# One synthetic run, typed the way the real dataset types questions, built so the
# pre-M-12e scorer and the fixed scorer give PROVABLY different numbers.


def _m12e_fixture() -> tuple[dict, dict]:
    """A run where the two subsets have opposite difficulty.

    Simple Recall subset (``get == "all"``): four questions, all answered
    perfectly, one per bin 0/1/2/3-5 -> Simple Recall Score 1.0.

    Chronological Awareness subset: four questions, all answered wrongly ->
    both CA legs 0.0.

    The pre-M-12e scorer mixed the second group into Simple Recall and diluted
    bins 1 and 2 with the failing latest/chronological rows. The fixed scorer
    returns 1.0.
    """
    results = {
        # Simple Recall subset — perfect answers.
        "a_bin0": {"response": "None"},
        "a_bin1": {"response": "- Jan 1"},
        "a_bin2": {"response": "- Jan 1\n- Feb 1"},
        "a_bin35": {"response": "- Jan 1\n- Feb 1\n- Mar 1"},
        # Chronological Awareness subset — wrong answers.
        "l_1": {"response": "- Nov 9"},
        "l_2": {"response": "- Nov 9"},
        "c_1": {"response": "- Nov 9\n- Dec 9"},
        "c_2": {"response": "- Nov 9\n- Dec 9"},
    }
    prompts = {
        "a_bin0": _prompt("a_bin0", [], get_style="all"),
        "a_bin1": _prompt("a_bin1", ["Jan 1"], get_style="all"),
        "a_bin2": _prompt("a_bin2", ["Jan 1", "Feb 1"], get_style="all"),
        "a_bin35": _prompt("a_bin35", ["Jan 1", "Feb 1", "Mar 1"], get_style="all"),
        "l_1": _prompt("l_1", ["Jan 1"], get_style="latest"),
        "l_2": _prompt("l_2", ["Feb 1"], get_style="latest"),
        "c_1": _prompt("c_1", ["Jan 1", "Feb 1"], get_style="chronological"),
        "c_2": _prompt("c_2", ["Mar 1", "Apr 1"], get_style="chronological"),
    }
    payload = {
        "run_id": "m12e",
        "model_role": "ingest_long_context",
        "config_name": "memory_off",
        "results": {"tulving_episodic": results},
    }
    return payload, prompts


def test_simple_recall_uses_only_the_recall_subset():
    payload, prompts = _m12e_fixture()
    scored = score_result_payload(payload, prompts)
    summary = scored["summary"]

    assert summary["scored_questions"] == 8
    assert summary["simple_recall_questions"] == 4
    assert summary["latest_questions"] == 2
    assert summary["chronological_questions"] == 2

    # The fix: the recall subset is perfect, so the score is exactly 1.0.
    assert summary["simple_recall_score"] == 1.0
    # And the CA subset is entirely wrong, so it cannot be hiding in there.
    assert summary["chronological_awareness_score"] == 0.0


def test_pre_m12e_behaviour_would_have_differed():
    """Pin the defect itself: scoring every question gives a DIFFERENT number.

    This reproduces the pre-M-12e ``simple_inputs.append(scored)`` on every row
    and asserts the two disagree, so a regression that re-widens the subset
    cannot pass silently.
    """
    from tulving_episodic_adapter import compute_simple_recall_score

    payload, prompts = _m12e_fixture()
    scored = score_result_payload(payload, prompts)

    # Bins 1 and 2 pick up the failing latest/chronological rows and average to
    # 1/3 each, so the whole-set score is (1 + 1/3 + 1/3 + 1) / 4.
    every_question = compute_simple_recall_score(scored["per_question"])
    assert every_question == (1.0 + 1 / 3 + 1 / 3 + 1.0) / 4
    assert scored["summary"]["simple_recall_score"] == 1.0
    assert every_question != scored["summary"]["simple_recall_score"]


def test_simple_recall_bins_on_matching_events_not_item_count():
    """The bin basis is ``nb_events``, which can disagree with ``nb_gt``.

    One item can be the answer for several chapters, so a one-item answer can
    be a six-event question. Binning on item count puts it in bin 1.
    """
    payload = {
        "run_id": "bins",
        "results": {"tulving_episodic": {"q": {"response": "- Jan 1"}}},
    }
    prompts = {"q": _prompt("q", ["Jan 1"], get_style="all", nb_events=7)}

    scored = score_result_payload(payload, prompts)
    bins = scored["summary"]["simple_recall_bins"]
    assert bins["6+"]["count"] == 1
    assert bins["1"]["count"] == 0
    assert scored["summary"]["simple_recall_bin_basis"] == "nb_events"


def test_simple_recall_bin_basis_reports_the_fallback():
    payload = {
        "run_id": "bins",
        "results": {"tulving_episodic": {"q": {"response": "- Jan 1"}}},
    }
    prompt = _prompt("q", ["Jan 1"], get_style="all")
    prompt["metadata"].pop("nb_events")

    scored = score_result_payload(payload, {"q": prompt})
    assert scored["summary"]["simple_recall_bin_basis"] == "nb_gt_fallback"


def test_tau_coverage_check_fails_closed_on_partial_set():
    """A correctly ordered PARTIAL list scores 0.0, not 1.0."""
    prompt = _prompt(
        "q", ["Jan 1", "Feb 1", "Mar 1", "Apr 1"], get_style="chronological"
    )
    detail = chronological_tau_detail("- Jan 1\n- Feb 1", prompt)

    assert detail["nb_gt"] == 4
    assert detail["nb_matched"] == 2
    assert detail["coverage"] == 0.5
    assert detail["full_coverage"] is False
    assert detail["status"] == "partial"
    # The ordering it DID emit was perfect — that is exactly the trap.
    assert detail["tau_raw"] == 1.0
    assert detail["tau"] == 0.0
    assert chronological_tau("- Jan 1\n- Feb 1", prompt) == 0.0


def test_tau_full_coverage_is_scored():
    prompt = _prompt("q", ["Jan 1", "Feb 1", "Mar 1"], get_style="chronological")
    detail = chronological_tau_detail("- Jan 1\n- Feb 1\n- Mar 1", prompt)
    assert detail["full_coverage"] is True
    assert detail["status"] == "scored"
    assert detail["tau"] == 1.0


def test_tau_short_ground_truth_is_not_labelled_partial():
    prompt = _prompt("q", ["Jan 1"], get_style="chronological")
    detail = chronological_tau_detail("- Jan 1", prompt)
    assert detail["status"] == "too_short"
    assert detail["tau"] == 0.0


def test_partial_coverage_is_reported_not_silent():
    payload = {
        "run_id": "cov",
        "results": {
            "tulving_episodic": {
                "c_full": {"response": "- Jan 1\n- Feb 1"},
                "c_partial": {"response": "- Jan 1"},
            }
        },
    }
    prompts = {
        "c_full": _prompt("c_full", ["Jan 1", "Feb 1"], get_style="chronological"),
        "c_partial": _prompt(
            "c_partial", ["Jan 1", "Feb 1", "Mar 1"], get_style="chronological"
        ),
    }

    scored = score_result_payload(payload, prompts)
    assert scored["summary"]["chronological_partial_coverage"] == 1
    assert scored["chronological_partial_coverage_ids"] == ["c_partial"]


def test_unknown_get_style_enters_no_subset():
    payload = {
        "run_id": "unk",
        "results": {"tulving_episodic": {"q": {"response": "- Jan 1"}}},
    }
    prompts = {"q": _prompt("q", ["Jan 1"], get_style="surprise")}

    scored = score_result_payload(payload, prompts)
    summary = scored["summary"]
    assert summary["scored_questions"] == 1
    assert summary["simple_recall_questions"] == 0
    assert summary["unknown_get_style"] == 1
    assert scored["unknown_get_style_ids"] == ["q"]


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


def test_render_markdown_reports_partial_tau_coverage(tmp_path):
    payload, prompts = _m12e_fixture()
    scored = score_result_payload(payload, prompts)
    md = render_markdown(scored, tmp_path / "result.json")
    assert "Scorer version: 2" in md
    assert "failed closed for partial coverage" in md
    assert "Simple Recall Bins (matching events)" in md


# ── SC67: the belief-kernel write hook ───────────────────────────────────────


def test_load_belief_capture_prefers_epyc_root(tmp_path, monkeypatch):
    adapters = tmp_path / "scripts" / "vidya" / "adapters"
    adapters.mkdir(parents=True)
    (adapters / "tulving_episodic_capture.py").write_text("MARKER = 'stub'\n")
    monkeypatch.setenv("EPYC_ROOT", str(tmp_path))
    monkeypatch.setattr(
        score_tulving_run, "_ROOT_CANDIDATES", (str(tmp_path),), raising=True)
    assert _load_belief_capture().MARKER == "stub"


def test_load_belief_capture_explains_a_missing_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        score_tulving_run, "_ROOT_CANDIDATES", (str(tmp_path / "nope"),), raising=True)
    with pytest.raises(SystemExit, match="tulving_episodic_capture"):
        _load_belief_capture()


def _root_capture_module():
    """The real epyc-root capture module, if this checkout can see it.

    ``EPYC_ROOT`` wins, so a lane worktree that has the adapter before it merges can
    still exercise the round trip.
    """
    import os

    for root in (os.environ.get("EPYC_ROOT"), "/mnt/raid0/llm/epyc-root", "/workspace"):
        if root and (Path(root) / "scripts/vidya/adapters/"
                     "tulving_episodic_capture.py").is_file():
            return root
    return None


@pytest.mark.skipif(_root_capture_module() is None, reason="epyc-root not on this host")
def test_belief_sidecar_round_trips_through_the_root_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        score_tulving_run, "_ROOT_CANDIDATES", (_root_capture_module(),), raising=True)
    capture = _load_belief_capture()

    payload, prompts = _m12e_fixture()
    scored = score_result_payload(payload, prompts)
    out = tmp_path / "tulving_score.json"
    out.write_text(json.dumps(scored, indent=2))

    sidecar = capture.write_belief_measurements(
        out, summary=scored["summary"], run_id="m12e-test",
        producer="score_tulving_run.py", arm="none",
        variant="Udefault_Sdefault_seed0", chapters=196)
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert capture.validate_row(row) == []
        assert row["extra"]["scorer_version"] == SCORER_VERSION
        assert row["extra"]["arm"] == "none"


@pytest.mark.skipif(_root_capture_module() is None, reason="epyc-root not on this host")
def test_root_writer_refuses_a_pre_m12e_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        score_tulving_run, "_ROOT_CANDIDATES", (_root_capture_module(),), raising=True)
    capture = _load_belief_capture()

    payload, prompts = _m12e_fixture()
    scored = score_result_payload(payload, prompts)
    scored["summary"]["scorer_version"] = 1
    out = tmp_path / "tulving_score.json"
    out.write_text(json.dumps(scored, indent=2))

    with pytest.raises(capture.CaptureError, match="scorer_version"):
        capture.write_belief_measurements(
            out, summary=scored["summary"], run_id="m12e-test",
            producer="score_tulving_run.py", arm="none",
            variant="Udefault_Sdefault_seed0", chapters=196)
    assert not (tmp_path / "belief_measurements.jsonl").exists()
