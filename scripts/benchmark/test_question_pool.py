#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest

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


# ── EVL-12 C2: A3 build invariant (LOSS-1/2) ─────────────────────────


class _EmptyAdapter:
    def __init__(self, degraded=None, raises=None):
        self._degraded = degraded or []
        self._raises = raises

    def extract_all(self):
        if self._raises:
            raise self._raises
        return []

    def accounting_summary(self):
        return {"dropped_rows": 0, "dropped_by_reason": {}, "degraded_sources": self._degraded}


def _fake_registry(monkeypatch, adapters):
    fake_module = types.SimpleNamespace(
        ADAPTER_SUITES=set(adapters),
        YAML_ONLY_SUITES=set(),
        get_adapter=lambda suite: adapters[suite],
    )
    monkeypatch.setitem(sys.modules, "dataset_adapters", fake_module)


def test_silent_zero_suite_fails_the_build_and_keeps_the_live_pool(tmp_path, monkeypatch):
    out = tmp_path / "question_pool.jsonl"
    out.write_text("LIVE POOL\n")
    _fake_registry(monkeypatch, {"math": FakeMathAdapter(), "gaia": _EmptyAdapter()})

    with pytest.raises(question_pool.PoolBuildInvariantError, match=r"no recorded reason: \['gaia'\]"):
        question_pool.build_pool(out)
    assert out.read_text() == "LIVE POOL\n", "a refused build must not touch the existing pool"
    assert not (tmp_path / "question_pool.jsonl.tmp").exists()


def test_missing_adapter_fails_the_build(tmp_path, monkeypatch):
    _fake_registry(monkeypatch, {"math": FakeMathAdapter(), "ghost": None})
    with pytest.raises(question_pool.PoolBuildInvariantError, match=r"absent from the build: \['ghost'\]"):
        question_pool.build_pool(tmp_path / "question_pool.jsonl")


@pytest.mark.parametrize(
    "adapter, reason_fragment",
    [
        (_EmptyAdapter(raises=FileNotFoundError("gated dataset")), "extraction failed: FileNotFoundError: gated dataset"),
        (_EmptyAdapter(degraded=[{"source": "gaia", "error": "gated"}]), "degraded_sources: gaia: gated"),
    ],
)
def test_accounted_zero_suite_is_recorded_not_fatal(tmp_path, monkeypatch, adapter, reason_fragment):
    out = tmp_path / "question_pool.jsonl"
    _fake_registry(monkeypatch, {"math": FakeMathAdapter(), "gaia": adapter})

    stats = question_pool.build_pool(out)
    header = json.loads(out.read_text().splitlines()[0])
    assert stats == {"math": 1, "gaia": 0}
    assert reason_fragment in header["empty_suites"]["gaia"]
    assert "math" not in header["empty_suites"]
