from __future__ import annotations

import json
from pathlib import Path

import pytest

from autokernel.loop import historical_trajectory as trajectory


def test_source_refuses_changed_evidence(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text('{"effect_pct": 1.0}')
    with pytest.raises(ValueError, match="required evidence changed"):
        trajectory._source(path, ['"effect_pct": 2.0'])


def test_live_build_is_sorted_separate_and_marks_conflict_and_missing() -> None:
    research = Path(__file__).resolve().parents[4]
    payload = trajectory.build(
        store=Path("/mnt/raid0/llm/autokernel/loop-memory"),
        research=research, root_repo=Path("/workspace"))
    assert payload["schema"] == trajectory.SCHEMA
    assert payload["points"] == sorted(payload["points"], key=lambda p: (
        p["recorded_at"], p["model"], p["surface"], p["recipe"], p["evidence_state"]))
    keys = {(p["commit"][:8], p["model"], p["surface"], p["recipe"]): p
            for p in payload["points"]}
    assert keys[("a2728701", trajectory.DEEPSEEK, "tg128", "gpu-loop-production-shaped-v1")]["gain_pct"] == pytest.approx(12.618108)
    assert keys[("b0eb4fab", trajectory.QWEN_GPU, "dec-b4", "gpu-production-model-direct-ab")]["gain_pct"] == pytest.approx(22.442869)
    assert any(p["evidence_state"] == "overwritten_conflicting_producer_record" for p in payload["points"])
    assert any(p["model"].startswith("gemma-4") and p["gain_pct"] == pytest.approx(7.205928) for p in payload["points"])
    assert {p["surface"] for p in payload["points"] if p["commit"] == trajectory.CURRENT} == {"plain-decode", "mtp-decode"}
    assert payload["missing_checkpoints"][0]["commit"] == trajectory.CURRENT
    assert payload["active_campaign"]["expected_keep_count"] == 23
