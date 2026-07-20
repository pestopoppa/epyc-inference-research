#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset_adapters import BaseAdapter, MathAdapter


class LossyAdapter(BaseAdapter):
    suite_name = "lossy"

    def _ensure_loaded(self):
        self._dataset = [
            {"kind": "ok"},
            {"kind": "empty"},
            {"kind": "raises"},
        ]

    def _row_to_prompt(self, idx: int, row: dict) -> dict | None:
        if row["kind"] == "empty":
            return None
        if row["kind"] == "raises":
            raise ValueError("bad row")
        return {"id": f"ok_{idx}", "suite": self.suite_name, "prompt": "P"}


def test_base_extract_all_counts_and_logs_dropped_rows(caplog):
    adapter = LossyAdapter()

    with caplog.at_level(logging.WARNING):
        rows = adapter.extract_all()

    assert rows == [{"id": "ok_0", "suite": "lossy", "prompt": "P"}]
    assert adapter.dropped_rows == 2
    assert adapter.dropped_by_reason == {
        "empty_prompt": 1,
        "row_to_prompt_exception": 1,
    }
    assert "[adapter:lossy] dropped 2 source row(s)" in caplog.text


def test_math_adapter_records_math500_degradation_without_dropping_gsm8k(monkeypatch):
    class FakeDatasets:
        @staticmethod
        def load_dataset(name: str, *args, **kwargs):
            if name == "gsm8k":
                return [{"question": "2+2?", "answer": "#### 4"}]
            if name == "HuggingFaceH4/MATH-500":
                raise RuntimeError("offline")
            raise AssertionError(name)

    monkeypatch.setitem(sys.modules, "datasets", FakeDatasets)
    adapter = MathAdapter()

    adapter._ensure_loaded()
    rows = adapter.extract_all()

    assert adapter.source_counts == {"gsm8k": 1, "math500": 0}
    assert adapter.degraded_sources == [{"source": "math500", "error": "offline"}]
    assert [row["id"] for row in rows] == ["gsm8k_00000"]
    assert adapter.accounting_summary()["source_counts"]["math500"] == 0
