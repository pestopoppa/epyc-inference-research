#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import question_pool


class FakeMathAdapter:
    def extract_all(self) -> list[dict]:
        return [
            {
                "id": "math500_00000",
                "suite": "math",
                "prompt": "P",
                "expected": "A",
            }
        ]

    def accounting_summary(self) -> dict:
        return {
            "dropped_rows": 2,
            "dropped_by_reason": {"row_to_prompt_exception": 2},
            "degraded_sources": [{"source": "math500", "error": "partial"}],
            "source_counts": {"gsm8k": 1, "math500": 1},
        }


def test_build_pool_writes_adapter_loss_and_math500_counts(tmp_path, monkeypatch):
    fake_module = types.SimpleNamespace(
        ADAPTER_SUITES={"math"},
        YAML_ONLY_SUITES=set(),
        get_adapter=lambda suite: FakeMathAdapter(),
    )
    monkeypatch.setitem(sys.modules, "dataset_adapters", fake_module)

    out = tmp_path / "question_pool.jsonl"
    stats = question_pool.build_pool(out)
    header = json.loads(out.read_text().splitlines()[0])

    assert stats == {"math": 1}
    assert header["suites"] == {"math": 1}
    assert header["adapter_stats"]["math"]["dropped_rows"] == 2
    assert header["source_counts"]["math"] == {"gsm8k": 1, "math500": 1}
    assert header["n_math500"] == 1


def test_load_pool_warns_when_loaded_counts_disagree_with_header(tmp_path, caplog):
    pool_path = tmp_path / "question_pool.jsonl"
    header = {
        "__pool_metadata__": True,
        "generated_at": "2026-07-20T00:00:00+00:00",
        "total_questions": 2,
        "suites": {"math": 2},
    }
    row = {"id": "gsm8k_00000", "suite": "math", "prompt": "P"}
    pool_path.write_text(json.dumps(header) + "\n" + json.dumps(row) + "\n")

    with caplog.at_level(logging.WARNING):
        pool = question_pool.load_pool(pool_path, warn_stale=False)

    assert len(pool["math"]) == 1
    assert "header total_questions=2, loaded=1" in caplog.text
    assert "[math]: header=2, loaded=1" in caplog.text
