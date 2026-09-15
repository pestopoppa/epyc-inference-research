import json
from pathlib import Path
import sqlite3

from . import process_metrics as metrics


def test_progression_auc_and_valid_ratio_use_attempt_steps():
    rows = [
        {"status": "planner_transient"},
        {"status": "measured_null", "effect_fraction": -0.1},
        {"status": "kept", "effect_fraction": 0.2},
        {"status": "bench_failed"},
        {"status": "unrelated_telemetry", "effect_fraction": 9},
        {"status": "measured_null", "effect_fraction": 0.1},
    ]
    result = metrics.calculate(rows)
    assert result["first_improvement_step"] == 3
    assert result["auc_best_so_far_effect_over_steps"] == 0.8
    assert result["final_best_so_far_effect_fraction"] == 0.2
    assert result["valid_step_counts"] == {"measured": 3, "denominator": 5}
    assert result["valid_step_ratio"] == 0.6


def test_critic_comparison_requires_explicit_lineage_for_both_cohorts():
    unavailable = metrics.calculate([
        {"status": "measured_null"},
        {"status": "kept", "critic_pass_1_rejected": True},
    ])["critic_pass_1_comparison"]
    assert unavailable["available"] is False
    assert unavailable["measured_null_rate_difference_after_rejection_minus_first_pass"] is None
    assert "absence of a rejection is not evidence" in unavailable["unavailable_reason"]
    assert unavailable["pass_1_rejection_rate"] is None

    available = metrics.calculate([
        {"status": "critic_reject"},
        {"status": "measured_null", "critic_pass_1_rejected": True,
         "critic_pass_1": {"accepted": True, "rejected_before_acceptance": True}},
        {"status": "kept", "critic_pass_1_rejected": True},
        {"status": "measured_null", "critic_pass_1_rejected": False,
         "critic_pass_1": {"accepted": True, "rejected_before_acceptance": False}},
        {"status": "measured_null", "critic_pass_1_rejected": False},
    ])["critic_pass_1_comparison"]
    assert available["available"] is True
    assert available["cohorts"]["accepted_after_rejection"]["eventual_measured_null_rate"] == 0.5
    assert available["cohorts"]["accepted_first_pass"]["eventual_measured_null_rate"] == 1.0
    assert available["measured_null_rate_difference_after_rejection_minus_first_pass"] == -0.5
    assert available["pass_1_decisions"] == {"rejected": 1, "accepted": 2}
    assert available["pass_1_rejection_rate"] == 1 / 3


def test_sqlite_loader_is_read_only_and_projects_only_critic_payload_fields(tmp_path):
    path = tmp_path / "experiments.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE experiments (recorded_at TEXT, status TEXT, "
                       "effect_fraction REAL, payload TEXT)")
    connection.execute("INSERT INTO experiments VALUES (?,?,?,?)", (
        "2026-01-01T00:00:00Z", "kept", 0.03,
        json.dumps({"large_irrelevant_field": "x" * 100_000,
                    "critic_pass_1_rejected": False})))
    connection.commit()
    connection.close()
    before = path.stat().st_mtime_ns
    row = list(metrics.load(path))[0]
    assert row["effect_fraction"] == 0.03
    assert row["critic_pass_1_rejected"] == 0
    assert "large_irrelevant_field" not in row
    assert path.stat().st_mtime_ns == before


def test_json_object_iterations_and_bad_shape(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"iterations": [{"status": "bench_failed"}]}))
    assert list(metrics.load(path)) == [{"status": "bench_failed"}]
    path.write_text(json.dumps({"no_attempts_here": True}))
    try:
        metrics.load(path)
    except ValueError as exc:
        assert "expected a JSON list" in str(exc)
    else:
        raise AssertionError("bad JSON shape accepted")


def test_checked_in_retrospective_artifact_is_internally_consistent():
    repo = Path(__file__).resolve().parents[4]
    artifact = json.loads((repo / "artifacts/autokernel/"
                           "s3-aku-10-process-metrics-20260915.json").read_text())
    assert artifact["schema"] == "epyc.autokernel.process_metrics_artifact.v1"
    assert len(artifact["inputs"]) == 4
    assert artifact["input_contract"]["payload_full_column_materialized"] is False
    assert artifact["input_contract"]["payloads_embedded"] is False
    assert all(source["size_bytes"] > 0 and source["snapshot_identity"]["row_count"] >= 1
               for source in artifact["inputs"])
    assert sum(source["snapshot_identity"]["row_count"]
               for source in artifact["inputs"]) == 2577
    result = artifact["aggregate_result"]
    counts = result["valid_step_counts"]
    assert result["valid_step_ratio"] == counts["measured"] / counts["denominator"]
    critic = result["critic_pass_1_comparison"]
    assert critic["available"] is False
    assert critic["pass_1_rejection_rate"] is None
    assert critic["measured_null_rate_difference_after_rejection_minus_first_pass"] is None


def test_collection_reports_each_store_and_chronological_aggregate(tmp_path):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    first.write_text(json.dumps([{"recorded_at": "2026-01-02Z", "status": "kept",
                                  "effect_fraction": 0.2}]))
    second.write_text(json.dumps([{"recorded_at": "2026-01-01Z",
                                   "status": "planner_transient"}]))
    result = metrics.analyze([first, second])
    assert len(result["stores"]) == 2
    assert result["aggregate"]["attempt_steps"] == 2
    assert result["aggregate"]["first_improvement_step"] == 2
