"""GPU resume waiver and scheduler pre-claim failure regressions; no hardware."""
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from . import run, scheduling, serial_run
from . import test_promotion_targets as fixture_module


def test_resumed_main_preserves_explicit_unverified_anchor_waiver(monkeypatch):
    fixture = fixture_module.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        hand_build = fixture.root / "hand-built"
        hand_build.mkdir()
        model = fixture.root / f"{run.bench.MEASURED_FLOOR_MODEL_STEM}.gguf"
        prior = {
            "terminal": "complete", "worktree": str(fixture.repo),
            "branch": run.champion.CANONICAL_BRANCH, "model": str(model),
            "selected_target": None,
            "current_anchor": {"path": str(hand_build), "commit": fixture.tip},
            "cor_anchor": {"path": str(hand_build), "commit": fixture.tip},
        }
        base = ["--worktree", str(fixture.repo), "--anchor-build", str(hand_build),
                "--model", str(model), "--store", str(fixture.store),
                "--resume-run", str(fixture.root / "receipt.json"), "--dry-run"]
        monkeypatch.setattr(serial_run, "load_resume", lambda *_args: (prior, "a" * 64))
        monkeypatch.setattr(run.workload_contract, "verify_workload", lambda _model:
                            SimpleNamespace(n_embd=1536, dominant_quant="Q4_K"))
        monkeypatch.setattr(run, "noise_floor_pct", lambda *_args, **_kwargs: None)

        with pytest.raises(run.champion.StartupRefused, match="allow-unverified-anchor"):
            run.main(base)
        with mock.patch.object(serial_run, "verify_exact_anchor",
                               wraps=serial_run.verify_exact_anchor) as exact:
            assert run.main(base + ["--allow-unverified-anchor"]) == 0
        assert exact.call_args.kwargs["allow_unverified"] is True
    finally:
        fixture.doCleanups()


def test_preclaim_marker_is_bound_to_the_issued_selection(tmp_path):
    selection = SimpleNamespace(digest="a" * 64)
    target = {"selected_id": "gpu"}
    run._publish_preclaim_failure(tmp_path, selection, target,
                                  run.champion.StartupRefused("refused"))
    marker = json.loads((tmp_path / "loop-preclaim-failure.json").read_text())
    assert marker == {
        "schema": "epyc.autokernel.preclaim_failure.v1",
        "selection_digest": selection.digest,
        "target": target,
        "error_type": "StartupRefused",
    }


def test_missing_accounting_evidence_reports_contract_failure_not_enoent(
        tmp_path, monkeypatch):
    selection = SimpleNamespace(digest="b" * 64)
    monkeypatch.setattr(scheduling.Selection, "from_dict", lambda _body: selection)
    active = {"scheduler_selection": {},
              "scheduler_selection_sha256": selection.digest}
    with pytest.raises(serial_run.SerialRefused,
                       match="neither held-resource evidence nor a pre-claim"):
        serial_run._scheduled_failure_account(
            {}, SimpleNamespace(), active, tmp_path, ["--target-id", "gpu"])
