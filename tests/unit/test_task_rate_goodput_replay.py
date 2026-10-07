from __future__ import annotations

from scripts.analysis import task_rate_goodput_replay as replay


def test_report_rows_use_folded_supersession_values(tmp_path) -> None:
    output = tmp_path / "report.md"
    rows = [
        {
            "trial_id": 2,
            "quality": 9.0,
            "speed": 99.0,
            "cost": 0.5,
            "reliability": 1.0,
            "tier": 1,
        },
        {
            "type": "supersession",
            "target_trial_ids": [2],
            "fields": {"quality": 1.23, "speed": 4.56},
        },
    ]
    legacy = {
        "frontier": [{"trial_id": 2, "objectives": [9.0, 99.0, -0.5, 1.0]}],
        "all_entries": [{"trial_id": 2}],
        "hypervolume_history": [[2, 1.0]],
    }
    task_rate = {
        "frontier": [],
        "all_entries": [],
        "hypervolume_history": [[2, 0.0]],
    }

    replay._write_report(
        output,
        journal=tmp_path / "autopilot_journal.jsonl",
        rows=rows,
        malformed=0,
        legacy=legacy,
        task_rate=task_rate,
        current_run_only=False,
    )

    report = output.read_text()

    assert "| 2 | 1.230 | 4.56 |" in report
    assert "| 2 | 9.000 | 99.00 |" not in report


def test_report_includes_baseline_promotion_evidence_for_scoped_rows(tmp_path) -> None:
    output = tmp_path / "report.md"
    rows = [
        {
            "trial_id": 2,
            "quality": 1.2,
            "speed": 10.0,
            "cost": 0.5,
            "reliability": 0.9,
            "tier": 1,
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 2,
            "tier": 1,
            "previous_quality": 1.0,
            "new_quality": 1.2,
            "timestamp": "2026-06-14T00:00:00+00:00",
            "reason": "accepted | escaped",
            "proof": {
                "matrix_status": "ok",
                "speed_metric_mode": "aggregate_batch_tps",
            },
            "result_metrics": {
                "quality": 1.2,
                "speed": 10.0,
                "pareto_status": "frontier",
            },
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 99,
            "tier": 1,
            "new_quality": 9.9,
            "reason": "out of scope",
        },
    ]
    archive = {
        "frontier": [{"trial_id": 2, "objectives": [1.2, 10.0, -0.5, 0.9]}],
        "all_entries": [{"trial_id": 2}],
        "hypervolume_history": [[2, 1.0]],
    }

    replay._write_report(
        output,
        journal=tmp_path / "autopilot_journal.jsonl",
        rows=rows,
        malformed=0,
        legacy=archive,
        task_rate=archive,
        current_run_only=True,
    )

    report = output.read_text()

    assert "## Baseline Promotion Evidence" in report
    assert "| 2 | yes | 1 | 1.000 | 1.200 | 0.200 | 1.200 | 10.00 | frontier | ok | aggregate_batch_tps | accepted \\| escaped |" in report
    assert "out of scope" not in report


def test_report_baseline_promotion_table_tolerates_incomplete_events(tmp_path) -> None:
    output = tmp_path / "report.md"
    rows = [
        {
            "trial_id": 7,
            "quality": 1.0,
            "speed": 5.0,
            "cost": 0.5,
            "reliability": 0.9,
            "tier": 1,
        },
        {"type": "baseline_promotion", "source_trial_id": 7},
    ]
    archive = {
        "frontier": [{"trial_id": 7, "objectives": [1.0, 5.0, -0.5, 0.9]}],
        "all_entries": [{"trial_id": 7}],
        "hypervolume_history": [[7, 1.0]],
    }

    replay._write_report(
        output,
        journal=tmp_path / "autopilot_journal.jsonl",
        rows=rows,
        malformed=0,
        legacy=archive,
        task_rate=archive,
        current_run_only=False,
    )

    report = output.read_text()

    assert "## Baseline Promotion Evidence" in report
    assert "| 7 | yes | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in report


def test_report_uses_dynamic_drop_threshold_and_quality_floor(tmp_path) -> None:
    output = tmp_path / "report.md"
    rows = [
        {
            "trial_id": tid,
            "quality": quality,
            "speed": 10.0,
            "cost": 0.5,
            "reliability": reliability,
            "tier": 1,
            "eval_wall_s": 100.0,
            "n_questions": 50,
        }
        for tid, quality, reliability in [
            (1, 2.0, 1.0),
            (2, 1.9, 1.0),
            (3, 1.8, 1.0),
            (4, 1.7, 1.0),
            (5, 1.6, 1.0),
            (6, 1.5, 1.0),
            (7, 0.5, 1.0),
        ]
    ]
    legacy = {
        "frontier": [
            {"trial_id": tid, "objectives": [2.0, 10.0, -0.5, 1.0]}
            for tid in range(1, 7)
        ],
        "all_entries": [{"trial_id": tid} for tid in range(1, 8)],
        "hypervolume_history": [[7, 1.0]],
    }
    task_rate = {
        "frontier": [
            {"trial_id": tid, "objectives": [2.0, 1800.0, 1.0]}
            for tid in [1, 2, 3, 7]
        ],
        "all_entries": [{"trial_id": tid} for tid in range(1, 8)],
        "hypervolume_history": [[7, 1.0]],
    }

    replay._write_report(
        output,
        journal=tmp_path / "autopilot_journal.jsonl",
        rows=rows,
        malformed=0,
        legacy=legacy,
        task_rate=task_rate,
        current_run_only=False,
    )

    report = output.read_text()

    assert "3 of 6 legacy canonical T1 frontier points fall off" in report
    assert "Raw Fable drop criterion (`>=3 of 6`) is met" in report
    assert "Task-rate promotion readiness is not ready" in report
    assert "1 quality-floor violation(s)" in report


def test_row_table_renders_unavailable_rate_as_na_and_preserves_measured_values() -> None:
    trials = list(range(1, 8))
    rows_by_tid = {
        1: {"trial_id": 1, "eval_details": {"question_results": [{"qid": "q1"}]}},
        2: {"trial_id": 2, "n_questions": "invalid", "eval_wall_s": 60.0},
        3: {"trial_id": 3, "n_questions": 0, "eval_wall_s": 60.0},
        4: {"trial_id": 4, "n_questions": 50, "eval_wall_s": 600.0},
        5: {
            "trial_id": 5,
            "eval_details": {"details": {"per_suite_counts": {}}, "eval_wall_s": 60.0},
        },
        6: {
            "trial_id": 6,
            "eval_details": {
                "details": {"per_suite_counts": {"a": 10, "b": 5}},
                "eval_wall_s": 300.0,
            },
        },
        7: {"trial_id": 7, "n_questions": 10, "eval_wall_s": float("inf")},
    }
    entries = [{"trial_id": trial} for trial in trials]

    rendered = replay._row_table(
        "rate controls",
        entries,
        rows_by_tid,
        [],
        include_dominators=False,
    )

    rows = [line for line in rendered if line.startswith("| ") and not line.startswith("|---")]
    cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in rows]
    assert [row[0] for row in cells] == [str(trial) for trial in trials]
    assert [row[5] for row in cells] == ["n/a", "n/a", "0.00", "300.00", "n/a", "180.00", "n/a"]
