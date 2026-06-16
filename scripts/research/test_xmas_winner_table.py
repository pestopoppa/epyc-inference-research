from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("xmas_winner_table.py")
SPEC = importlib.util.spec_from_file_location("xmas_winner_table", MODULE_PATH)
assert SPEC is not None
xmas_winner_table = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(xmas_winner_table)


def _results_payload() -> dict:
    table = {}
    for domain in xmas_winner_table.XMAS_DOMAINS:
        table[domain] = {
            "frontdoor": {
                "correct": 2,
                "total": 3,
                "accuracy": 2 / 3,
                "wall_mean_s": 10.0,
            },
            "worker_general": {
                "correct": 3,
                "total": 3,
                "accuracy": 1.0,
                "wall_mean_s": 20.0,
            },
            "architect_general": {
                "correct": 3,
                "total": 3,
                "accuracy": 1.0,
                "wall_mean_s": 30.0,
            },
        }
    table["knowledge"]["frontdoor"] = {
        "correct": 3,
        "total": 3,
        "accuracy": 1.0,
        "wall_mean_s": 1.0,
    }
    return {
        "started_at": "2026-05-20T12:18:51Z",
        "n_tasks": 25,
        "n_models": 3,
        "summary": {"table": table},
    }


def _function_axis_results_payload() -> dict:
    table = {}
    for domain in xmas_winner_table.XMAS_DOMAINS:
        table[domain] = {}
        for function in xmas_winner_table.XMAS_FUNCTIONS:
            table[domain][function] = {
                "frontdoor": {
                    "correct": 1,
                    "total": 2,
                    "accuracy": 0.5,
                    "wall_mean_s": 1.0,
                },
                "worker_general": {
                    "correct": 2,
                    "total": 2,
                    "accuracy": 1.0,
                    "wall_mean_s": 2.0,
                },
            }
    table["knowledge"]["extract"]["frontdoor"] = {
        "correct": 2,
        "total": 2,
        "accuracy": 1.0,
        "wall_mean_s": 0.5,
    }
    return {
        "started_at": "2026-06-16T00:00:00Z",
        "n_tasks": 125,
        "n_models": 2,
        "summary": {"table": table},
    }


def test_build_winner_table_outputs_complete_5x5_with_evidence(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_results_payload()), encoding="utf-8")

    payload = xmas_winner_table.build_winner_table(
        results,
        version="xmas-test",
        fallback_role="frontdoor",
    )

    assert payload["version"] == "xmas-test"
    assert payload["cells"]["math"]["solve"] == "worker_general"
    assert payload["cells"]["knowledge"]["extract"] == "frontdoor"
    assert payload["provenance"]["winner_rule"] == "correct_desc_then_wall_mean_s_asc"
    assert (
        payload["evidence"]["math"]["solve"]["derivation"]
        == "domain_winner_reused_for_function"
    )
    assert payload["evidence"]["math"]["solve"]["sample_count"] == 3
    assert set(payload["cells"]) == set(xmas_winner_table.XMAS_DOMAINS)
    assert all(
        set(functions) == set(xmas_winner_table.XMAS_FUNCTIONS)
        for functions in payload["cells"].values()
    )


def test_build_winner_table_rejects_missing_domain(tmp_path: Path) -> None:
    raw = _results_payload()
    raw["summary"]["table"].pop("reasoning")
    results = tmp_path / "results.json"
    results.write_text(json.dumps(raw), encoding="utf-8")

    try:
        xmas_winner_table.build_winner_table(results, version="xmas-test")
    except ValueError as exc:
        assert "missing domains: reasoning" in str(exc)
    else:
        raise AssertionError("expected missing domain failure")


def test_build_winner_table_accepts_true_function_axis_summary(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_function_axis_results_payload()), encoding="utf-8")

    payload = xmas_winner_table.build_winner_table(
        results,
        version="xmas-function-axis-test",
        fallback_role="frontdoor",
    )

    assert payload["version"] == "xmas-function-axis-test"
    assert payload["provenance"]["derivation_mode"] == "function_axis_sweep"
    assert payload["cells"]["math"]["solve"] == "worker_general"
    assert payload["cells"]["knowledge"]["extract"] == "frontdoor"
    assert (
        payload["evidence"]["math"]["solve"]["source_summary_path"]
        == "summary.table.math.solve.worker_general"
    )
    assert (
        payload["evidence"]["knowledge"]["extract"]["source_summary_path"]
        == "summary.table.knowledge.extract.frontdoor"
    )


def test_build_winner_table_rejects_function_axis_missing_function(
    tmp_path: Path,
) -> None:
    raw = _function_axis_results_payload()
    raw["summary"]["table"]["math"].pop("verify")
    results = tmp_path / "results.json"
    results.write_text(json.dumps(raw), encoding="utf-8")

    try:
        xmas_winner_table.build_winner_table(results, version="xmas-test")
    except ValueError as exc:
        assert "summary.table.math missing functions: verify" in str(exc)
    else:
        raise AssertionError("expected missing function failure")
