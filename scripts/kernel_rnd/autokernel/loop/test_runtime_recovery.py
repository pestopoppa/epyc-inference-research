"""Real child/flock boundaries; synthetic measurement providers, no hardware."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from . import claim, runtime_admission as ra, runtime_calibration as rc
from . import runtime_recovery as recovery, serial_run as sr, scheduling, status
from .measurement_capture import ArtifactStore
from .test_serial_run import _inputs, _scheduled
from .unified_planner import RuntimeArmPair


def _child(argv):
    from . import direct_historical_control as historical
    from . import test_runtime_admission as fixture
    from .test_serving_preparation import statistics
    from .test_glm_frozen_requests import _manifest
    from ..evaluator import statistics as st
    from dataclasses import replace
    out = Path(sr.option(argv, "--out"))
    root = out.parents[1]
    target = sr._selected_identity(argv)
    source_pair = RuntimeArmPair.from_dict(json.loads((root / "fixture-pair.json").read_text()))
    context = pytest.MonkeyPatch()
    context.setattr(fixture, "_fixture", lambda path: (source_pair, None, None))
    fixture_dir = root / "fixture"
    fixture_dir.mkdir(exist_ok=True)
    pair, unused, counter, _make_owner = fixture.installed_fixture(fixture_dir, context)
    sys.modules["autokernel.loop.direct_historical_control"].NAMESPACE = "fixture-historical"
    unused.close()
    Path(sr.option(argv, "--store")).mkdir(exist_ok=True)
    store = ArtifactStore(Path(sr.option(argv, "--store")) / "runtime-preparation")
    declared = statistics()
    campaign = "ak-calibration-fixture"
    declared = replace(declared, commitment=st.StoppingRuleCommitment.commit(
        declared.stopping_rule, campaign_id=campaign, committed_at=declared.commitment.committed_at))
    prompts = _manifest({"prompt": [1, 2, 3], "n_predict": 8, "temperature": 0.0,
        "top_k": 1, "seed": 42, "cache_prompt": False, "ignore_eos": True,
        "return_tokens": True, "stream": False})
    worktree = Path(sr.__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip()
    selection = scheduling.Selection.from_dict(json.loads(Path(sr.option(argv, "--scheduler-selection")).read_text()))
    reference_path = sr.option(argv, "--runtime-recovery-reference")
    reference = json.loads(Path(reference_path).read_text()) if reference_path else None
    first = reference is None
    ordinary_failure = (root / "mode").read_text() == "source_failure"
    original_launch = rc.DirectPairWindow._launch
    def launch(window, *args, **kwargs):
        if first and window.candidate_id.startswith("akc-runtime-") and len(window.launches) == 2:
            raise rc.RuntimeLaunchBudgetExhausted("fixture between-launch deadline")
        return original_launch(window, *args, **kwargs)
    context.setattr(rc.DirectPairWindow, "_launch", launch)
    with (root / "private.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        holder = claim.HeldCpuClaim({"device_id": "cpu", "cpu_list": "0", "regions": ["fixture"]},
            [root / "private.lock"], region_fraction=selection.proposal.estimated_claims.physical_region_fraction,
            affinity=("0",))
        owner = ra.RuntimeAdmission(store=store, held_claim=holder, campaign_id=campaign,
            epoch="fixture", original=pair.anchor, prompts=prompts, statistical=declared,
            host_state={"fixture": "not host health"}, worktree=worktree, source_commit=commit,
            recovery_reference=reference)
        try:
            if not ordinary_failure:
                historical_kwargs = dict(store=store, held_claim=holder, campaign_id=campaign,
                                         window_index=51, deadline_monotonic_s=-1.0)
                historical_previous = None if first else json.loads(
                    (root / "batches/batch-000000/fixture-historical.json").read_text())
                binding = None if first else historical.prepare_replacement(store=store,
                    reference=historical_previous, held_claim=holder, recovery_reference=reference)
                _resolution, observed, retained = historical.run_or_reopen(
                    **historical_kwargs, replacement=binding)
                assert not observed.ran and observed.could_not_run_reason.startswith("HistoricalLaunchBudgetExhausted:")
                status.write_json(out, "fixture-historical.json", retained.to_dict())
                if not first:
                    assert retained.to_dict() != historical_previous
                    _resolved, old_observed, old_ref = historical.run_or_reopen(
                        **historical_kwargs, reference=historical_previous)
                    assert old_ref.to_dict() == historical_previous and not old_observed.ran
                    assert historical.run_or_reopen(**historical_kwargs, replacement=binding)[2] == retained
            if ordinary_failure:
                pass  # Deliberate unrelated setup failure, despite valid held receipts.
            elif first:
                with pytest.raises(rc.RuntimeLaunchBudgetExhausted):
                    owner.compare(pair)
                interruption = owner.interruption_reference()
                assert interruption is not None
                status.write_json(out, "loop-runtime-interruption.json", interruption)
            else:
                original_pair = owner.pending_pair()
                assert original_pair == pair
                ordinary = SimpleNamespace(propose=lambda _: pytest.fail("fresh actor drew over pending pair"))
                planner = recovery.PendingPlanner(ordinary, recovery.PendingPlanner.slot(original_pair))
                hypothesis = planner.propose({})
                result = owner.compare(hypothesis.runtime_pair)
                assert result["runtime_admission"] and result["control_panel"]["may_rank"]
                assert len(owner.state["attempts"]) == 1
                completed_count = len(counter)
                assert owner.reopen_admission(result["runtime_admission"])[0] == pair
                assert len(counter) == completed_count
            windows = [window for window, _ops, _source in owner._windows.values()
                       if window.candidate_id.startswith("akc-runtime-") and window.stratum == "selection"]
            if windows:
                window = windows[0]
                checkpoint = store.root / window.state_name
                status.write_json(out, "fixture-window.json", {
                    "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "identity": window.identity, "candidate_id": window.candidate_id,
                    "launches": window.launches, "frame": rc.ob._plain(window.frame),
                    "holder": recovery.holder_identity(holder), "launch_count": len(counter)})
        finally:
            holder._closing()
            fcntl.flock(lock, fcntl.LOCK_UN)
            holder._released_now()
    held_store = ArtifactStore(out / "held-claim-artifacts")
    try:
        held = claim.publish_intervals(held_store, selection, [holder], target=target)
    finally:
        held_store.close()
    held_ref = {"schema": sr.HELD_REFERENCE_SCHEMA, "selection_digest": selection.digest,
                "evidence": held.to_dict()}
    status.write_json(out, "loop-held-claims.json", held_ref)
    store.close()
    if first:
        return 7
    status.write_json(out, "loop-run.json", {"fixture": "synthetic runtime providers, not hardware measurements"})
    continuation = sr.continuation(argv=argv, binding=sr.input_binding(argv), terminal="complete",
        worktree=sr.option(argv, "--worktree"), branch=sr.option(argv, "--experimental-branch"),
        model=target["original_target"]["execution"]["model"]["path"], selected_target=target,
        anchor_build=sr.option(argv, "--anchor-build"), anchor_commit="a" * 40,
        cor_build=None, cor_commit=None, iterations_requested=1,
        outcomes=[SimpleNamespace(status="runtime_observed")], held_claim_evidence=held_ref)
    status.write_json(out, "loop-continuation.json", continuation)
    return 0


def test_real_serial_interruption_new_holder_same_pending_pair_no_old_sample_mix(tmp_path, monkeypatch):
    from .test_runtime_treatment import _fixture
    state_root, argv = _inputs(tmp_path, monkeypatch, rounds=2)
    # One declared CPU target. Selection attempts, original held evidence and the
    # existing serial state are real; measurements are explicit fixture providers.
    del argv[2:4]
    argv = _scheduled(tmp_path, argv)
    fixture_dir = tmp_path / "original-fixture"
    fixture_dir.mkdir()
    pair, _log, _pids = _fixture(fixture_dir)
    (state_root / "fixture-pair.json").write_text(json.dumps(pair.to_dict()))
    monkeypatch.setenv("PYTHONPATH", str(Path(sr.__file__).resolve().parents[2]))
    monkeypatch.setattr(sr, "_child_command", lambda row: [sys.executable, "-m", __name__, "--child", *row])
    with mock.patch.object(sr, "_scheduled_failure_account", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            sr.main(argv)
    before = json.loads((state_root / "serial-state.json").read_text())
    assert before["active"] and before["next_batch"] == 0
    first = state_root / "batches/batch-000000"
    assert (first / "loop-runtime-interruption.json").exists(), (first / "stderr.log").read_text()
    assert sr.main(argv) == 0
    second = state_root / "batches/batch-000001"
    saved = json.loads((state_root / "serial-state.json").read_text())
    assert not saved["failed_targets"] and saved["next_batch"] == 2
    old, new = (json.loads((directory / "fixture-window.json").read_text()) for directory in (first, second))
    assert old["candidate_id"] == new["candidate_id"] and old["identity"] != new["identity"]
    assert old["holder"] != new["holder"] and len(old["launches"]) == 2
    assert hashlib.sha256(Path(old["path"]).read_bytes()).hexdigest() == old["sha256"]
    old_refs = {row["artifact"]["sha256"] for row in old["launches"]}
    assert old_refs.isdisjoint(row["artifact"]["sha256"] for row in new["launches"])
    assert new["frame"]["pair"] == old["frame"]["pair"]
    assert new["frame"]["units"] == old["frame"]["units"]
    assert new["frame"]["calibration"] == old["frame"]["calibration"]
    assert new["launch_count"] < old["launch_count"]  # Completed controls were reopened, not relaunched.
    accounted = scheduling.SchedulerState.from_dict(saved["scheduler_state"])
    assert accounted.campaign_attempts == 2
    assert {row.outcome for row in accounted.accounted_receipts} == {"failed", "invalid"}


def test_valid_closed_claim_does_not_retry_unrelated_failure(tmp_path, monkeypatch):
    from .test_runtime_treatment import _fixture
    state_root, argv = _inputs(tmp_path, monkeypatch, rounds=2, mode="source_failure")
    del argv[2:4]
    argv = _scheduled(tmp_path, argv)
    fixture_dir = tmp_path / "original-fixture"
    fixture_dir.mkdir()
    pair, _log, _pids = _fixture(fixture_dir)
    (state_root / "fixture-pair.json").write_text(json.dumps(pair.to_dict()))
    monkeypatch.setenv("PYTHONPATH", str(Path(sr.__file__).resolve().parents[2]))
    monkeypatch.setattr(sr, "_child_command", lambda row: [sys.executable, "-m", __name__, "--child", *row])
    assert sr.main(argv) == 1
    saved = json.loads((state_root / "serial-state.json").read_text())
    assert saved["next_batch"] == 1 and "0" in saved["failed_targets"]
    assert not saved["runtime_recovery"] and not saved.get("runtime_recovery_errors")
    assert scheduling.SchedulerState.from_dict(saved["scheduler_state"]).campaign_attempts == 1


@pytest.mark.parametrize("kind", ["pair", "calibration"])
def test_unknown_pending_aborts_even_without_recovery_reference(tmp_path, monkeypatch, kind):
    from .test_runtime_admission import installed_fixture
    from .test_runtime_calibration import original_claim
    from .loop import RunAborted
    pair, store, counter, make_owner = installed_fixture(tmp_path, monkeypatch)
    try:
        with original_claim(tmp_path / "lock") as holder:
            owner = make_owner(holder)
            frame, ref = owner.calibration(pair.anchor)
            window = frame if kind == "calibration" else rc.DirectPairWindow(
                calibration=frame, calibration_reference=ref, pair=pair, candidate_id="akc-pending")
            window.pending = {"fixture": "original launch intent; no terminal teardown proof"}
            window._checkpoint()
            with pytest.raises(RunAborted, match="cleanup unresolved"):
                fresh = make_owner(holder)
                if kind == "calibration":
                    fresh.calibration(pair.anchor)
                else:
                    fresh._evaluate_pair(pair, "akc-pending", panel={})
            assert not counter
    finally:
        store.close()


_FINISH = rc.DirectPairWindow.finish
_CRASH_AFTER_CLOSE = []


def _finish_then_crash(window):
    _FINISH(window)
    if _CRASH_AFTER_CLOSE:
        _CRASH_AFTER_CLOSE.pop()
        raise RuntimeError("fixture crash after original window close, before attempt save")


def test_closed_window_reopens_original_setup_before_attempt_checkpoint(tmp_path, monkeypatch):
    from .test_runtime_admission import installed_fixture
    from .test_runtime_calibration import original_claim
    pair, store, counter, make_owner = installed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys.modules[__name__], "_CRASH_AFTER_CLOSE", [True])
    monkeypatch.setattr(rc.DirectPairWindow, "finish", _finish_then_crash)
    try:
        with original_claim(tmp_path / "lock") as holder:
            owner = make_owner(holder)
            with pytest.raises(RuntimeError, match="fixture crash"):
                owner.compare(pair)
            original_window = next(iter(owner._windows.values()))[0]
            assert original_window.boundaries["close"] is not None
            original_launches = list(original_window.launches)
            prior_count = len(counter)
        with original_claim(tmp_path / "lock") as holder:
            fresh = make_owner(holder)
            result = fresh.compare(pair)
            first = next(iter(fresh._windows.values()))[0]
            assert first.identity == original_window.identity
            assert first.launches == original_launches
            assert first._reopened and first.solution["result"] is not None
            assert result["runtime_admission"]
            assert len(counter) > prior_count  # Remaining original windows still execute.
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(_child(sys.argv[2:]))
