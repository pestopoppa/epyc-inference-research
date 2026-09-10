"""Connected direct keeper with explicit synthetic hardware/calibration providers.

Actual tiny HTTP transport/reopen is covered by test_runtime_calibration. These
tests supply calibration/window/historical fixtures, then exercise original raw
capture, correctness, ControlHarness, sequential reducer and retained selection.
They are not qualified hardware measurements.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

from ..evaluator import api, controls, statistics as st
from ..evaluator.test_api import window as fixture_window
from ..evaluator.test_controls import _available, _passing_observation
from ..evaluator.test_statistics import make_blocks, make_inputs
from . import gates, runtime_admission as admission, runtime_calibration as rc
from . import runtime_window as rw, native_server_response as response
from . import measurement_capture as mc, serving, resolved_recipe as rr
from .test_runtime_treatment import _fixture
from .test_runtime_calibration import original_claim
from .test_serving_preparation import statistics
from .test_glm_frozen_requests import _manifest
from .test_resolved_recipe import _policy
from .unified_planner import RuntimeDimension, enumerate_runtime_dimensions

_LAUNCHES = []
_FAIL_ONCE = []


def fixture_measure(template, build, port, *, response_capture, observation, frozen_requests,
                    resolved_recipe, observation_session):
    # Top-level fixture callable: no hidden closure substituted into the original
    # response producer's prospectively checked callable configuration.
    _LAUNCHES.append(resolved_recipe.execution_digest)
    rate = 50 if dict(resolved_recipe.launch_env)["GGML_IQK"] == "0" else 100
    if template.threads != serving.Recipe.__dataclass_fields__["threads"].default:
        rate *= 2
    rows, start = [], time.monotonic()
    for phase in response.PHASES:
        for slot, (name, raw) in enumerate(frozen_requests):
            request = json.loads(raw)
            payload = json.dumps({"stop": True, "content": "x" * request["n_predict"],
                "tokens": [11] * request["n_predict"], "timings": {
                    "predicted_n": request["n_predict"], "predicted_per_second": rate}}).encode()
            begin, end = time.monotonic(), time.monotonic()
            rows.append(response.RawServerResponse(phase, slot, name, raw, payload, begin, end, None))
    receipt = response_capture.seal(rows, process_pid=os.getpid(),
        request_started_monotonic_s=start, request_ended_monotonic_s=time.monotonic())
    observation.append({"server_responses": receipt, "process_pid": os.getpid(),
        "residency": {"window_start": start, "window_end": time.monotonic(),
                      "backend": "cpu", "status": "not_applicable"}})
    observation_session.finish()
    if _FAIL_ONCE and template.threads != serving.Recipe.__dataclass_fields__["threads"].default:
        _FAIL_ONCE.pop()
        observation_session.finish()
        from .loop import MeasurementInvalid
        raise MeasurementInvalid("synthetic terminal invalid arm after owned teardown", {"fixture": True})
    return rate


def installed_fixture(tmp_path, monkeypatch, *, campaign_id="ak-calibration-fixture"):
    pair, _, _ = _fixture(tmp_path)
    old = pair.anchor
    template = replace(old.template, np=1, temperature=0.0, top_k=1)
    anchor = rr.resolve_canonical_launch(template, build_dir=old.build_dir,
        command_argv=template.server_argv(Path(old.build_dir), old.port), topology_prefix=(),
        launch_environment={**dict(old.launch_env), "GGML_IQK": "1"},
        artifact_identities={"model": old.model.to_dict(), "drafter": None,
            "executable": old.executable.to_dict(), "dsos": [row.to_dict() for row in old.dsos]},
        backend="cpu", environment_policy=_policy("GGML_IQK"), port=old.port,
        runtime_binary_dir=old.runtime_binary_dir, runtime_ld_paths=old.runtime_ld_paths,
        provenance=dict(old.provenance))
    pair = enumerate_runtime_dimensions(anchor, (RuntimeDimension("threads", "threads",
        anchor.template.threads, anchor.template.threads + 1, "fixture original treatment"),))[0]
    prompts = _manifest({"prompt": [1, 2, 3], "n_predict": 8, "temperature": 0.0,
        "top_k": 1, "seed": 42, "cache_prompt": False, "ignore_eos": True,
        "return_tokens": True, "stream": False})
    declared = statistics()
    declared = replace(declared, commitment=st.StoppingRuleCommitment.commit(
        declared.stopping_rule, campaign_id=campaign_id,
        committed_at=declared.commitment.committed_at))
    def collect(frame):
        result = frame.store.write("fixture-numeric-calibration", {"frame": frame.identity})
        frame.solution = result.to_dict()
        frame._checkpoint()
        return result
    def reopen(frame, reference):
        assert frame.store.read(reference.locator, reference.sha256)["frame"] == frame.identity
        base = 50 if dict(frame.pairs["aa"].anchor.launch_env).get("GGML_IQK") == "0" else 100
        inputs = make_inputs(cell_class="serving", stopping_rule=declared.stopping_rule,
            aa_blocks=make_blocks(200, effect=0, noise=.01, seed=1, base=base),
            neutral_blocks=make_blocks(200, effect=0, noise=.01, seed=2, base=base),
            anchor_calibration_values=tuple(base + .1 * (n % 5 - 2) for n in range(200)))
        return st.solve_calibration(inputs)
    def boundaries(config, held, *, marker):
        return {"snapshot": {"marker": marker}, "claim": held.observe(), "fixture": "not host health"}
    def attest(window, rows, reduction, panel, anchor, raw_ref):
        return fixture_window(anchor_at_open=anchor, anchor_at_close=anchor,
            stopping_rule_id=window.statistical.stopping_rule.rule_id,
            order_seed=window.statistical.campaign_seed, raw_evidence_ref=raw_ref,
            **panel, **reduction.window_checks)
    def historical(**kwargs):
        body = {"fixture": "separate historical T2", "campaign": kwargs["campaign_id"], "index": kwargs["window_index"]}
        ref = kwargs["store"].write("fixture-historical", body)
        if kwargs["reference"] is not None:
            assert kwargs["reference"] == ref.to_dict()
        return _available(), _passing_observation(controls.CONTROL_HISTORICAL_WIN_REPLAY), ref
    counter = []
    monkeypatch.setattr(sys.modules[__name__], "_LAUNCHES", counter)
    monkeypatch.setattr(rc.DirectCalibration, "collect", collect)
    monkeypatch.setattr(rc.DirectCalibration, "reopen", reopen)
    monkeypatch.setattr(rc.DirectCalibration, "validity", lambda self:
        admission.schemas.Check(admission.schemas.PASS, ("synthetic host fixture, not hardware health",)))
    monkeypatch.setattr(rw, "boundary", boundaries)
    monkeypatch.setattr(rw, "launch_health", lambda *a, **k:
        admission.schemas.Check(admission.schemas.PASS, ("synthetic host fixture",)))
    monkeypatch.setattr(admission, "_window_checks", attest)
    monkeypatch.setattr(gates, "op_correctness", lambda *a, **k: gates.Verdict("correctness", True, "synthetic oracle"))
    monkeypatch.setattr(serving, "_measure_once", fixture_measure)
    historical_fixture = SimpleNamespace(run_or_reopen=historical)
    monkeypatch.setitem(sys.modules, "autokernel.loop.direct_historical_control", historical_fixture)
    monkeypatch.setattr(sys.modules["autokernel.loop"], "direct_historical_control",
                        historical_fixture, raising=False)
    store = mc.ArtifactStore(tmp_path / "captures")
    worktree = Path(__file__).resolve().parents[4]
    commit = __import__("subprocess").check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip()
    def owner(holder):
        return admission.RuntimeAdmission(store=store, held_claim=holder,
            campaign_id=campaign_id, epoch="fixture", original=anchor,
            prompts=prompts, statistical=declared, host_state={"fixture": "not host health"},
            worktree=worktree, source_commit=commit)
    return pair, store, counter, owner


def test_connected_measured_panel_keep_and_reopen_no_relaunch(tmp_path, monkeypatch):
    pair, store, counter, make_owner = installed_fixture(tmp_path, monkeypatch)
    try:
        with original_claim(tmp_path / "private.lock") as holder:
            owner = make_owner(holder)
            assert owner.selected() == pair.anchor
            row = owner.compare(pair)
            assert not row["evaluation"]["event_violations"]
            assert row["evaluation"]["event_emitted"], row["evaluation"]["event_blocked_reason"]
            assert row["decisive"] is True, row["evaluation"]
            assert row["control_panel"]["may_rank"] is True
            assert "belief_capture_error" not in row, row.get("belief_capture_error")
            assert Path(row["belief_export_receipt"]).is_file()
            assert owner.retain(row, pair.anchor) == pair.candidate
            continuation = owner.selection_reference(pair.candidate,
                current_source_commit=owner.source_commit)
            assert owner.retained_build(continuation) == Path(pair.anchor.build_dir)
            count = len(counter)
        with original_claim(tmp_path / "private.lock") as holder:
            reopened = make_owner(holder)
            assert reopened.selected() == pair.candidate
            assert len(counter) == count
            # Exact operational continuation survives a new process owner.
            assert admission.restore_selection(store=store, held_claim=holder,
                reference=continuation, worktree=owner.worktree, source_commit=owner.source_commit,
                build=pair.anchor.build_dir, prompts=owner.prompts) == pair.candidate
            assert len(counter) == count
            # Another target's verified source keep changes only the operational
            # build. The original admitted treatment and measurement stay intact.
            from .run import _cpu_arm
            next_build = tmp_path / "shared-source-keep"
            shutil.copytree(pair.candidate.build_dir, next_build)
            next_commit = "b" * 40  # Synthetic source verifier input, not a hardware claim.
            with pytest.raises(rc.RuntimeCalibrationRefused, match="source/request"):
                admission.restore_selection(store=store, held_claim=holder,
                    reference=continuation, worktree=owner.worktree, source_commit=next_commit,
                    build=next_build, prompts=owner.prompts)
            operational = admission.restore_selection(store=store, held_claim=holder,
                reference=continuation, worktree=owner.worktree, source_commit=next_commit,
                build=next_build, prompts=owner.prompts,
                source_anchor={"path": str(next_build), "commit": next_commit})
            assert operational == _cpu_arm(pair.candidate, next_build)
            moved_reference = owner.selection_reference(operational,
                current_source_commit=next_commit)
            moved = admission._read(store, "direct-runtime-selection", moved_reference)
            assert owner.retained_build(moved_reference) == Path(pair.anchor.build_dir)
            assert moved["admission"] == row["runtime_admission"]
            assert moved["owner"]["source_commit"] == owner.source_commit
            assert admission.restore_selection(store=store, held_claim=holder,
                reference=moved_reference, worktree=owner.worktree, source_commit=next_commit,
                build=next_build, prompts=owner.prompts) == operational
            assert len(counter) == count
        body = admission._read(store, "direct-runtime-admission", row["runtime_admission"])
        assert "provisional" not in json.dumps(body)
        assert body["confirmation"] is not None
        with monkeypatch.context() as changed:
            changed.setattr(admission, "_outputs", lambda *args: {})
            with pytest.raises(rc.RuntimeCalibrationRefused, match="source/original frame"):
                reopened.selected()
    finally:
        store.close()


def test_admission_uses_installed_historical_original_tiers_and_reopens(tmp_path, monkeypatch):
    from . import direct_historical_control as historical
    from .test_direct_historical_control import synthetic_execution, held as historical_held
    original_boundary, original_run = rw.boundary, admission.subprocess.run
    pair, store, counter, make_owner = installed_fixture(tmp_path, monkeypatch,
        campaign_id="ak-test-historical-original")
    monkeypatch.setitem(sys.modules, "autokernel.loop.direct_historical_control", historical)
    monkeypatch.setattr(sys.modules["autokernel.loop"], "direct_historical_control", historical)
    calls = synthetic_execution(tmp_path, monkeypatch)
    instrument_run = historical.subprocess.run
    def source_run(argv, **kwargs):
        if argv[:3] == ["git", "-C", str(historical.lc.INSTRUMENT_ROOT)]:
            return instrument_run(argv, **kwargs)
        return original_run(argv, **kwargs)
    monkeypatch.setattr(historical.subprocess, "run", source_run)
    monkeypatch.setattr(rw, "boundary", original_boundary)
    try:
        with historical_held(tmp_path) as holder:
            owner = make_owner(holder)
            row = owner.compare(pair)
            assert row["decisive"] is True, row["control_panel"]
            refs = owner.state["attempts"][0]["controls"]
            reference = refs[controls.CONTROL_HISTORICAL_WIN_REPLAY]
            body = historical._read(store, historical.NAMESPACE, reference)
            assert [item["tier"] for item in body["tier_evaluations"]] == ["T0", "T1", "T2"]
            assert calls == ["aa_calibration", "neutral_calibration", "historical_win_replay"]
            count = len(counter)
            assert owner.retain(row, pair.anchor) == pair.candidate
            assert owner.reopen_admission(row["runtime_admission"])[1]
            assert len(counter) == count and len(calls) == 3
    finally:
        store.close()


def test_measured_negative_panel_blocks_retention_despite_internal_bootstrap(tmp_path, monkeypatch):
    pair, store, counter, make_owner = installed_fixture(tmp_path, monkeypatch)
    original = admission.DirectGates.run_gates
    def broken_negative(self, request):
        result = original(self, request)
        # Deliberately defective fixture instrument: fails to catch wrong work.
        # The harness must detect its actual bad verdict and prevent retention.
        if any(row[2] == "candidate" and row[3]["recipe"]["template"]["n_predict"] <
               json.loads(self.original_prompts.prompts[0].body)["n_predict"] for row in self.rows):
            return (api.GateResult("fixture_missed_wrong_work", api.GATE_CORRECTNESS,
                                  admission.schemas.Check(admission.schemas.PASS)),)
        return result
    monkeypatch.setattr(admission.DirectGates, "run_gates", broken_negative)
    try:
        with original_claim(tmp_path / "private.lock") as holder:
            owner = make_owner(holder)
            row = owner.compare(pair)
            assert row["decisive"] is None
            assert row["control_panel"]["may_rank"] is False
            with pytest.raises(rc.RuntimeCalibrationRefused, match="lacks original admission"):
                owner.retain(row, pair.anchor)
            assert owner.selected() == pair.anchor
            assert counter
    finally:
        store.close()


def test_direct_claim_adapter_reads_actual_lock_and_does_not_restore(tmp_path):
    store = mc.ArtifactStore(tmp_path / "captures")
    try:
        with original_claim(tmp_path / "private.lock") as holder:
            adapter = rw.DirectHeldClaimAdapter(holder, cpu_list="0", store=store)
            assert adapter.attest().check.outcome == "PASS"
            with pytest.raises(ValueError, match="outside"):
                rw.DirectHeldClaimAdapter(holder, cpu_list="1", store=store)
        assert adapter.attest().check.outcome == "COULD_NOT_CHECK"
        assert len(adapter.observations) == 2
        assert adapter.claim_id == "direct-loop:" + holder._context_id
    finally:
        store.close()


def test_terminal_invalid_arm_reschedules_original_intent_without_relaunching_valid_arms(tmp_path, monkeypatch):
    pair, store, counter, make_owner = installed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys.modules[__name__], "_FAIL_ONCE", [True])
    from .loop import MeasurementInvalid
    try:
        with original_claim(tmp_path / "private.lock") as holder:
            owner = make_owner(holder)
            with pytest.raises(MeasurementInvalid) as invalid:
                owner.compare(pair)
            assert invalid.value.reschedule is not None
            windows = [window for window, _ops, _src in owner._windows.values() if window.failures]
            assert len(windows) == 1
            window = windows[0]
            assert window.pending is None and len(window.failures) == 1
            prefix = list(window.launches)
            failed = window.failures[0]
            assert failed["terminal_invalid"] is True
            assert not any(row["membership"] == failed["membership"] for row in prefix)
            row = invalid.value.reschedule()
            assert row["decisive"] is True
            assert window.launches[:len(prefix)] == prefix
            assert len([entry for entry in window.launches
                        if entry["membership"] == failed["membership"]]) == 1
            assert len(owner.state["attempts"]) == 1
            assert owner.retain(row, pair.anchor) == pair.candidate
    finally:
        store.close()


def test_preparation_budget_reuses_original_intent_and_spends_no_candidate_slot(tmp_path, monkeypatch):
    pair, store, counter, make_owner = installed_fixture(tmp_path, monkeypatch)
    try:
        with original_claim(tmp_path / "private.lock") as holder:
            owner = make_owner(holder)
            owner.deadline_monotonic_s = time.monotonic() - 1
            for _ in range(3):
                with pytest.raises(rc.RuntimeLaunchBudgetExhausted, match="deadline reached"):
                    owner.compare(pair)
            assert len(owner.state["attempts"]) == 1
            assert owner.state["attempts"][0]["result"] is None
            assert "confirmation" not in owner.state["attempts"][0]
            assert counter == []
    finally:
        store.close()


def test_missing_during_work_host_facts_invalidates_at_first_completed_launch(tmp_path, monkeypatch):
    owning_health = rw.launch_health
    pair, store, counter, make_owner = installed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(rw, "launch_health", owning_health)
    from .loop import MeasurementInvalid
    try:
        with original_claim(tmp_path / "private.lock") as holder:
            owner = make_owner(holder)
            frame, _ref = owner.calibration(pair.anchor)
            context = {"campaign_id": owner.campaign_id, "epoch": owner.epoch,
                       "comparison_id": frame.identity + ":aa", "arm": "anchor", "launch_index": 0}
            with pytest.raises(MeasurementInvalid, match="COULD_NOT_CHECK"):
                frame._launch(("aa", 0, "anchor"), pair.anchor, context)
            assert len(counter) == 1 and not frame.launches and frame.pending is None
            assert len(frame.failures) == 1 and frame.failures[0]["terminal_invalid"]
            retained = admission._read(store, "direct-calibration-launch", frame.failures[0]["artifact"])
            assert "COULD_NOT_CHECK" in retained["failure"]
            assert retained["during_work"]["samples"] == []
    finally:
        store.close()
