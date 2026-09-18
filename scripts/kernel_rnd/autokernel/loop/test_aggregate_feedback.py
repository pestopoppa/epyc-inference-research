"""Actual aggregate owner/archive/ROOT reader; synthetic hardware boundary only."""
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import pytest

from . import actors, archive, serving, serving_beliefs
from .test_existing_gpu_serving_run import (
    test_existing_gpu_pool_uses_selected_requests_and_original_keep_owners as exercise,
)


@pytest.mark.parametrize("promoted", [True, False])
def test_actual_aggregate_dispositions_export_original_capture_once(monkeypatch, promoted):
    root = Path(os.environ.get("EPYC_ROOT_REPO", "/workspace"))
    if not (root / "scripts/vidya/adapters/autokernel_legacy_serving.py").is_file():
        pytest.skip("EPYC_ROOT_REPO must select the original serving feedback reader")
    original_record, original_compare = archive.record, serving.compare
    gates, comparisons, errors = [], [], []

    def compare(*args, **kwargs):
        measure = serving._measure_once

        def observe(*args, evidence, **kwargs):
            started = time.time()
            result = measure(*args, evidence=evidence, **kwargs)
            evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "gpu",
                             "status": "unproven", "samples": 0,
                             "window_start": started, "window_end": time.time(),
                             "cpu_placement": "unproven", "contention": "unproven",
                             "scope": "synthetic measurement; no hardware observed"})
            return result

        with patch.object(serving, "_measure_once", observe):
            row = original_compare(*args, **kwargs)
        comparisons.append(row)
        return row

    def record(store, attempt, **kwargs):
        added = original_record(store, attempt, **kwargs)
        if "gate_outcome" not in attempt:
            return added
        try:
            check_gate(store, attempt, added, **kwargs)
        except Exception:
            errors.append(traceback.format_exc())
        return added

    def check_gate(store, attempt, added, **kwargs):
        assert added and attempt["comparison"] is comparisons[-1]
        assert attempt["serving"] is attempt["comparison"]
        assert attempt["gate_outcome"] == ("promote" if promoted else "diverged")
        assert attempt["status"] == ("measured_serving_gate" if promoted else "measured_divergence")
        assert attempt["bundled_keeps"] == ["akm-e2e-keep"]
        assert attempt["champion_of_record"] != attempt["at_commit"]
        if not promoted:
            assert attempt["planner_evidence"]["kind"] == "serving_divergence"
            assert attempt["mechanism_id"].startswith("serving-divergence-")
        bridge = kwargs["on_serving_export"].__self__
        assert isinstance(bridge, serving_beliefs.PlannerFeedback)
        row = attempt["comparison"]
        inputs = row["belief_capture"]["inputs"]
        scope = {"epoch": kwargs["epoch"], "model": inputs["resolved_arms"]["anchor"]["model"],
                 "recipe_hash": row["recipe_hash"], "request_digest": row["request_digest"],
                 "anchor_execution_digest": inputs["resolved_arms"]["anchor"]["execution_digest"],
                 "anchor_build": inputs["build_paths"]["anchor"]}
        result = bridge.context(scope)
        assert not result["errors"] and result["status"] == "observations_only"
        capture_id = row["belief_capture"]["capture_id"]
        observed, = [item for item in result["rows"] if item["capture_id"] == capture_id]
        assert observed["mechanism_id"] == attempt["mechanism_id"]
        assert observed["candidate_tok_s"] == row["candidate_tok_s"]
        assert not result["qualified_measurement"]
        prior = archive.recall(store, epoch=kwargs["epoch"])
        text = actors.render_context({"prior_experiments": prior, "serving_observations": result})
        assert attempt["mechanism_id"] in text and attempt["status"] in text
        assert "recall, not qualified gains" in text
        receipt = Path(store) / "serving-beliefs" / f"{capture_id}.json"
        before = receipt.read_bytes(), bridge._reader.ledger.path.read_bytes(), len(comparisons)
        native = json.loads((receipt.parent / json.loads(before[0])["native_reference"]["path"]).read_text())
        assert native["comparison"] == row
        assert original_record(store, attempt, **kwargs) is False
        assert (receipt.read_bytes(), bridge._reader.ledger.path.read_bytes(), len(comparisons)) == before
        reopened = serving_beliefs.PlannerFeedback(store, root).context(scope)
        assert capture_id in {item["capture_id"] for item in reopened["rows"]}
        assert not reopened["errors"] and len(comparisons) == before[2]
        gates.append(attempt)

    monkeypatch.setenv("EPYC_ROOT_REPO", str(root))
    monkeypatch.setattr(archive, "record", record)
    monkeypatch.setattr(serving, "compare", compare)
    # Existing fixture asserts one source keep, original COR semantics/cadence,
    # original requests, no restart measurement, and complete owned teardown.
    exercise(False, True, promoted)
    assert not errors, "\n".join(errors)
    assert len(gates) == 1
