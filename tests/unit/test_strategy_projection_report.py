"""Tests for the journal-frontier StrategyStore projection report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import strategy_projection_report as spr  # noqa: E402


def _frontier_row(trial_id: int) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "timestamp": "2026-06-19T00:00:00Z",
        "species": "prompt_forge",
        "action_type": "code_mutation",
        "tier": 1,
        "quality": 1.2,
        "speed": 40.0,
        "cost": 0.2,
        "reliability": 0.9,
        "pareto_status": "frontier",
        "hypothesis": "repair parser",
        "expected_mechanism": "targeted_fix",
        "outcome_status": "ok",
        "bug_corrupted_by": "",
        "eval_details": {},
    }


def test_render_markdown_summarizes_projection_report() -> None:
    rendered = spr.render_markdown(
        {
            "ok": False,
            "expected_count": 1,
            "projected_count": 0,
            "skipped_count": 2,
            "missing_count": 1,
            "unexpected_count": 0,
            "mismatch_count": 0,
            "dry_run": True,
            "would_insert_count": 1,
            "inserted_count": 0,
            "missing": [{"trial_id": 7, "strategy_id": "journal-frontier-trial-7"}],
            "unexpected": [],
            "mismatches": [],
        }
    )

    assert "# AutoPilot Strategy Projection Report" in rendered
    assert "- Status: drift" in rendered
    assert "expected=1, projected=0, skipped=2" in rendered
    assert "trial #7: journal-frontier-trial-7" in rendered


def test_cli_strict_reports_missing_projection(tmp_path: Path, capsys) -> None:
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text(
        json.dumps(_frontier_row(7)) + "\n",
        encoding="utf-8",
    )

    rc = spr.main(
        [
            "--journal-dir",
            str(journal_dir),
            "--strategy-path",
            str(strategy_path),
            "--json",
            "--strict",
        ]
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert out["ok"] is False
    assert out["missing"] == [
        {"trial_id": 7, "strategy_id": "journal-frontier-trial-7"}
    ]


def test_cli_write_missing_syncs_projection(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("EPISODIC_ALLOW_DEGRADED_EMBEDDINGS", raising=False)
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text(
        json.dumps(_frontier_row(8)) + "\n",
        encoding="utf-8",
    )
    embed_calls: list[str] = []

    def mock_owned_embedder_call(self, text: str):
        assert os.environ.get("EPISODIC_ALLOW_DEGRADED_EMBEDDINGS") == "1"
        embed_calls.append(text)
        return np.ones(1024, dtype=np.float32)

    monkeypatch.setattr(spr.TaskEmbedder, "embed_text", mock_owned_embedder_call)

    rc = spr.main(
        [
            "--journal-dir",
            str(journal_dir),
            "--strategy-path",
            str(strategy_path),
            "--json",
            "--strict",
            "--write-missing",
            "--allow-hash-fallback",
        ]
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["ok"] is True
    assert out["inserted_count"] == 1
    assert out["missing_count"] == 0
    assert out["embedding_policy"] == "hash_fallback_permitted"
    assert "which embedding was actually produced" in out["embedding_policy_warning"]
    assert embed_calls
    assert "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS" not in os.environ


def test_cli_write_missing_emits_degraded_policy_warning_in_markdown(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EPISODIC_ALLOW_DEGRADED_EMBEDDINGS", raising=False)
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text(
        json.dumps(_frontier_row(80)) + "\n",
        encoding="utf-8",
    )
    embed_calls: list[str] = []

    def mock_owned_embedder_call(self, text: str):
        assert os.environ.get("EPISODIC_ALLOW_DEGRADED_EMBEDDINGS") == "1"
        embed_calls.append(text)
        return np.ones(1024, dtype=np.float32)

    monkeypatch.setattr(spr.TaskEmbedder, "embed_text", mock_owned_embedder_call)
    rc = spr.main(
        [
            "--journal-dir", str(journal_dir),
            "--strategy-path", str(strategy_path),
            "--write-missing", "--allow-hash-fallback",
        ]
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert "## DEGRADED EMBEDDING POLICY ENABLED" in output
    assert "does not establish which embedding was actually produced" in output
    assert embed_calls
    assert "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS" not in os.environ


def test_cli_write_missing_requires_embedding_without_hash_override(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EPISODIC_ALLOW_DEGRADED_EMBEDDINGS", raising=False)
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text(
        json.dumps(_frontier_row(9)) + "\n",
        encoding="utf-8",
    )

    embed_calls: list[str] = []

    class BrokenEmbedder:
        def __init__(self, config):
            self.config = config

        def embed_text(self, text: str):
            embed_calls.append(text)
            raise RuntimeError("semantic embeddings unavailable")

    monkeypatch.setattr(spr, "TaskEmbedder", BrokenEmbedder)

    rc = spr.main(
        [
            "--journal-dir",
            str(journal_dir),
            "--strategy-path",
            str(strategy_path),
            "--json",
            "--strict",
            "--write-missing",
        ]
    )

    assert rc == 2
    assert "semantic embeddings unavailable" in capsys.readouterr().err
    assert embed_calls == ["strategy projection write preflight"]
    assert "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS" not in os.environ


def test_explicit_hash_fallback_scopes_env_and_reports_degraded_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_name = "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS"
    monkeypatch.setenv(env_name, "operator-value")
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text("", encoding="utf-8")
    observations: list[tuple[str | None, bool | None]] = []

    class ObservingStore:
        def __init__(self, *, path, embedder):
            observations.append((os.environ.get(env_name), embedder is None))

        def sync_frontier_journal_entries(self, journal, *, dry_run):
            observations.append((os.environ.get(env_name), dry_run))
            return {"ok": True, "dry_run": dry_run, "inserted_count": 1}

        def close(self):
            observations.append((os.environ.get(env_name), None))

    monkeypatch.setattr(spr, "StrategyStore", ObservingStore)
    report = spr.build_strategy_projection_report(
        journal_dir=journal_dir,
        strategy_path=strategy_path,
        write_missing=True,
        allow_hash_fallback=True,
    )

    assert observations == [("1", True), ("1", False), ("1", None)]
    assert os.environ[env_name] == "operator-value"
    assert report["allow_hash_fallback"] is True
    assert report["embedding_policy"] == "hash_fallback_permitted"
    assert "HASH FALLBACK IS PERMITTED" in report["embedding_policy_warning"]
    rendered = spr.render_markdown(report)
    assert "## DEGRADED EMBEDDING POLICY ENABLED" in rendered
    assert "does not establish which embedding was actually produced" in rendered


def test_explicit_hash_fallback_restores_unset_env_after_store_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_name = "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS"
    monkeypatch.delenv(env_name, raising=False)
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text("", encoding="utf-8")
    observed: list[str | None] = []

    class FailingStore:
        def __init__(self, *, path, embedder):
            observed.append(os.environ.get(env_name))

        def sync_frontier_journal_entries(self, journal, *, dry_run):
            observed.append(os.environ.get(env_name))
            raise RuntimeError("synthetic store failure")

        def close(self):
            observed.append(os.environ.get(env_name))

    monkeypatch.setattr(spr, "StrategyStore", FailingStore)
    with pytest.raises(RuntimeError, match="synthetic store failure"):
        spr.build_strategy_projection_report(
            journal_dir=journal_dir,
            strategy_path=strategy_path,
            write_missing=True,
            allow_hash_fallback=True,
        )

    assert observed == ["1", "1", "1"]
    assert env_name not in os.environ


def test_explicit_hash_fallback_restores_env_when_store_construction_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_name = "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS"
    monkeypatch.setenv(env_name, "prior-value")
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text("", encoding="utf-8")
    observed: list[str | None] = []

    class ConstructionFailure:
        def __init__(self, *, path, embedder):
            observed.append(os.environ.get(env_name))
            raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr(spr, "StrategyStore", ConstructionFailure)
    with pytest.raises(RuntimeError, match="synthetic constructor failure"):
        spr.build_strategy_projection_report(
            journal_dir=journal_dir,
            strategy_path=strategy_path,
            write_missing=True,
            allow_hash_fallback=True,
        )

    assert observed == ["1"]
    assert os.environ[env_name] == "prior-value"


def test_explicit_hash_fallback_restores_env_when_close_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_name = "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS"
    monkeypatch.delenv(env_name, raising=False)
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text("", encoding="utf-8")
    observed: list[str | None] = []

    class CloseFailure:
        def __init__(self, *, path, embedder):
            observed.append(os.environ.get(env_name))

        def sync_frontier_journal_entries(self, journal, *, dry_run):
            observed.append(os.environ.get(env_name))
            return {"ok": True, "dry_run": dry_run}

        def close(self):
            observed.append(os.environ.get(env_name))
            raise RuntimeError("synthetic close failure")

    monkeypatch.setattr(spr, "StrategyStore", CloseFailure)
    with pytest.raises(RuntimeError, match="synthetic close failure"):
        spr.build_strategy_projection_report(
            journal_dir=journal_dir,
            strategy_path=strategy_path,
            write_missing=True,
            allow_hash_fallback=True,
        )

    assert observed == ["1", "1", "1"]
    assert env_name not in os.environ


def test_hash_fallback_flag_does_not_change_dry_run_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_name = "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS"
    monkeypatch.delenv(env_name, raising=False)
    journal_dir = tmp_path / "journal"
    strategy_path = tmp_path / "strategies"
    journal_dir.mkdir()
    strategy_path.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text("", encoding="utf-8")
    observed: list[str | None] = []

    class DryRunStore:
        def __init__(self, *, path, embedder):
            observed.append(os.environ.get(env_name))

        def sync_frontier_journal_entries(self, journal, *, dry_run):
            observed.append(os.environ.get(env_name))
            assert dry_run is True
            return {"ok": True, "dry_run": True}

        def close(self):
            observed.append(os.environ.get(env_name))

    monkeypatch.setattr(spr, "StrategyStore", DryRunStore)
    report = spr.build_strategy_projection_report(
        journal_dir=journal_dir,
        strategy_path=strategy_path,
        write_missing=False,
        allow_hash_fallback=True,
    )

    assert observed == [None, None, None]
    assert env_name not in os.environ
    assert report["embedding_policy"] == "not_used"


def test_cli_returns_two_for_missing_strategy_path(tmp_path: Path, capsys) -> None:
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    (journal_dir / "autopilot_journal.jsonl").write_text("", encoding="utf-8")

    rc = spr.main(
        [
            "--journal-dir",
            str(journal_dir),
            "--strategy-path",
            str(tmp_path / "missing-strategies"),
        ]
    )

    assert rc == 2
    assert "strategy path does not exist" in capsys.readouterr().err
