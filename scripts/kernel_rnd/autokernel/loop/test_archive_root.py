"""ROOT selection at the existing diagnostic archive writer boundary; no inference."""
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pytest

from . import archive


@pytest.mark.parametrize("value", [None, "invalid"])
def test_missing_or_invalid_root_warns_after_experiment_commits(tmp_path, monkeypatch, capsys, value):
    for key in ("AK_HISTORY_ROOT", "EPYC_ROOT_REPO", "EPYC_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(archive, "_VIDYA_ROOT", None)
    if value:
        monkeypatch.setenv("AK_HISTORY_ROOT", str(tmp_path / value))
        # An invalid explicit override must not fall back to this valid ROOT.
        (tmp_path / "valid/scripts/vidya/adapters").mkdir(parents=True)
        monkeypatch.setenv("EPYC_ROOT", str(tmp_path / "valid"))
    store = tmp_path / "store"
    assert archive.record(store, {"status": "abstained", "mechanism_id": "fixture",
                                  "trial_id": "trial-fixture"}, epoch="fixture",
                          recorded_at="2026-10-31T00:00:00Z", campaign_id="fixture")
    stderr = capsys.readouterr().err
    assert "AK history capture failed after durable archive: CaptureUnavailable" in stderr
    assert ("has no scripts/vidya/adapters" if value else "EPYC_ROOT is not set") in stderr
    with sqlite3.connect(store / "experiments.db") as db:
        assert db.execute("SELECT count(*) FROM experiments").fetchone()[0] == 1
    assert not (store / "beliefs").exists()


def test_explicit_history_root_wins_and_foreign_cached_adapter_refuses(tmp_path, monkeypatch):
    adapters = ModuleType("adapters")
    adapters.__path__ = []
    monkeypatch.setitem(sys.modules, "adapters", adapters)
    chosen = tmp_path / "chosen"
    other = tmp_path / "other"
    for root in (chosen, other):
        (root / "scripts/vidya/adapters").mkdir(parents=True)
    expected = chosen / "scripts/vidya/adapters/ak_history.py"
    expected.write_text("# selected adapter stand-in\n")
    monkeypatch.setattr(archive, "_VIDYA_ROOT", None)
    monkeypatch.setenv("AK_HISTORY_ROOT", str(chosen))
    monkeypatch.setenv("EPYC_ROOT_REPO", str(other))
    monkeypatch.setenv("EPYC_ROOT", str(other))
    adapter = ModuleType("adapters.ak_history")
    adapter.__file__ = str(expected)
    adapter.FIELDS = ("trial_id", "trial_tokens")
    seen = []
    def append(path, **kwargs):
        seen.append((path, kwargs))
        return path / "receipt.jsonl"
    adapter.append_belief_measurement = append
    monkeypatch.setitem(sys.modules, "adapters.ak_history", adapter)
    monkeypatch.setattr(adapters, "ak_history", adapter, raising=False)
    attempt = {"trial_id": "t", "trial_tokens": 17, "unrelated": "omitted"}
    store = tmp_path / "store"
    result = archive._publish_ak_history(store, attempt, campaign_id="campaign", epoch="epoch",
                                        recorded_at="2026-10-31T00:00:00Z")
    assert result == store / "beliefs/receipt.jsonl"
    assert seen == [(store / "beliefs", {"campaign_id": "campaign",
        "store_id": str(store.resolve()), "epoch": "epoch",
        "producer": "autokernel.loop.archive/akhr3-v1",
        "trial": {"trial_id": "t", "trial_tokens": 17},
        "emitted_at": "2026-10-31T00:00:00Z"})]
    adapter.__file__ = str(other / "scripts/vidya/adapters/ak_history.py")
    with pytest.raises(RuntimeError, match="differs from selected ROOT"):
        archive._publish_ak_history(store, attempt, campaign_id="campaign", epoch="epoch",
                                    recorded_at="2026-10-31T00:00:00Z")
    assert len(seen) == 1
