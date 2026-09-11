"""GPU resume waiver and scheduler pre-claim failure regressions; no hardware."""
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from . import accumulate, run, scheduling, serial_run
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


def test_resumed_cor_preserves_waiver_after_dry_run_boundary(monkeypatch):
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
        argv = [
            "--worktree", str(fixture.repo), "--anchor-build", str(hand_build),
            "--cor-build", str(hand_build), "--model", str(model),
            "--store", str(fixture.store),
            "--resume-run", str(fixture.root / "receipt.json"),
            "--allow-unverified-anchor", "--iterations", "0",
        ]
        monkeypatch.setattr(serial_run, "load_resume", lambda *_args: (prior, "a" * 64))
        monkeypatch.setattr(run.workload_contract, "verify_workload", lambda _model:
                            SimpleNamespace(n_embd=1536, dominant_quant="Q4_K"))
        monkeypatch.setattr(run, "noise_floor_pct", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(accumulate, "load_bundle", lambda *_args, **_kwargs:
                            (accumulate.Bundle(fixture.tip, fixture.tip), "restored"))

        class ReachedCorGuard(RuntimeError):
            pass

        real_exact = serial_run.verify_exact_anchor
        exact_calls = []

        def exact(*args, **kwargs):
            exact_calls.append(kwargs)
            real_exact(*args, **kwargs)
            if len(exact_calls) == 2:
                raise ReachedCorGuard

        # Stop immediately after the post-dry-run COR guard. No claim, actor,
        # build or benchmark is entered by this regression.
        with mock.patch.object(serial_run, "verify_exact_anchor", side_effect=exact), \
                pytest.raises(ReachedCorGuard):
            run.main(argv)
        assert len(exact_calls) == 2
        assert all(call["allow_unverified"] is True for call in exact_calls)
    finally:
        fixture.doCleanups()


def test_cor_startup_refusal_publishes_preclaim_marker(tmp_path):
    selection = SimpleNamespace(digest="c" * 64)
    target = {"selected_id": "gpu"}

    with pytest.raises(run.champion.StartupRefused, match="wrong COR"):
        run._verify_before_claim(
            lambda: (_ for _ in ()).throw(
                run.champion.StartupRefused("wrong COR")),
            out=tmp_path, scheduler_selection=selection, target=target)

    marker = json.loads((tmp_path / "loop-preclaim-failure.json").read_text())
    assert marker["selection_digest"] == selection.digest
    assert marker["target"] == target
    assert marker["error_type"] == "StartupRefused"


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
