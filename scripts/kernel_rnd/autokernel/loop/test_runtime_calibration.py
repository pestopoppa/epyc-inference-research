"""Direct calibration retains actual tiny HTTP launches; no model or hardware gate."""
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json

import pytest

from . import claim, measurement_capture as mc, runtime_calibration as calibration
from .test_planned_serving import _prompts
from .test_runtime_treatment import _fixture
from .test_serving_preparation import statistics


@contextmanager
def original_claim(path):
    # Real kernel flock on a private fixture file, not a physical CPU reservation.
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = claim.HeldCpuClaim({"device_id": "cpu", "cpu_list": "0",
                                     "regions": ["fixture"]}, [path])
        try:
            yield receipt
        finally:
            receipt._closing()


def test_default_inputs_are_prospective_immutable_and_supplied_inputs_win(tmp_path):
    store = mc.ArtifactStore(tmp_path / "captures")
    try:
        original = calibration.declare_statistics(store=store, campaign_id="original", epoch="epoch")
        assert original.controls.calibration_block_count == 200
        assert original.controls.contribution_floor == 0.03
        assert original.controls.max_candidates == 10
        assert original.controls.confirmation_admission_count == 2
        assert original.controls.max_blocks_per_candidate == 20
        assert original.owning_rep_rule.blocks == 10
        reopened = calibration.declare_statistics(store=store, campaign_id="original", epoch="epoch")
        assert reopened.to_dict() == original.to_dict()
        supplied = statistics()
        overridden = calibration.declare_statistics(store=store, campaign_id="calibration-campaign",
                                                    epoch="supplied", supplied=supplied)
        assert overridden.to_dict() == supplied.to_dict()
        with pytest.raises(calibration.RuntimeCalibrationRefused, match="changed after"):
            calibration.declare_statistics(store=store, campaign_id="calibration-campaign",
                epoch="supplied", supplied=replace(supplied,
                    controls=replace(supplied.controls, calibration_block_count=3)))
    finally:
        store.close()


def test_positive_control_keeps_original_glm_requests_and_distinct_control_anchor():
    from .test_glm_frozen_requests import _canonical_launch, _manifest, _request
    from .test_resolved_recipe import _policy
    from . import resolved_recipe as rr
    _template, original = _canonical_launch(18311)
    original = rr.resolve_canonical_launch(original.template, build_dir=original.build_dir,
        command_argv=original.command_argv, topology_prefix=original.topology_prefix,
        launch_environment={**dict(original.launch_env), "GGML_IQK": "1"},
        artifact_identities={"model": original.model.to_dict(), "drafter": None,
            "executable": original.executable.to_dict(), "dsos": [row.to_dict() for row in original.dsos]},
        backend="cpu", environment_policy=_policy("GGML_IQK"), port=original.port,
        runtime_binary_dir=original.runtime_binary_dir, runtime_ld_paths=original.runtime_ld_paths,
        provenance=dict(original.provenance))
    before = original.to_dict()
    positive = calibration.positive_control_pair(original)
    manifest = _manifest(_request())
    ids = tuple(row.prompt_id for row in manifest.prompts)
    assert dict(original.launch_env)["GGML_IQK"] == "1"
    assert dict(positive.anchor.launch_env)["GGML_IQK"] == "0"
    assert dict(positive.candidate.launch_env)["GGML_IQK"] == "1"
    assert positive.anchor.execution_digest != positive.candidate.execution_digest
    assert positive.candidate.execution_digest == original.execution_digest
    assert positive.anchor.executable == positive.candidate.executable == original.executable
    assert positive.anchor.dsos == positive.candidate.dsos == original.dsos
    assert positive.anchor.topology_prefix == positive.candidate.topology_prefix == original.topology_prefix
    assert manifest.requests(ids, positive.anchor.template) == manifest.requests(ids, original.template)
    assert manifest.requests(ids, positive.candidate.template) == manifest.requests(ids, original.template)
    assert original.to_dict() == before


def test_actual_calibration_original_prefix_reopens_without_relaunch(tmp_path, monkeypatch):
    # Actual tiny HTTP transport, explicitly synthetic host admission. Its
    # sub-second requests cannot claim two real under-load host samples.
    monkeypatch.setattr(calibration.runtime_window, "launch_health",
        lambda *a, **k: calibration.schemas.Check(calibration.schemas.PASS, ("synthetic host fixture",)))
    pair, requests_log, pids = _fixture(tmp_path)
    prompts = _prompts(pair.anchor.template)
    store = mc.ArtifactStore(tmp_path / "captures")
    try:
        neutral = calibration.neutral_material(store=store, anchor=pair.anchor)
        assert neutral.anchor.execution_digest == neutral.candidate.execution_digest
        assert neutral.anchor.snapshot_digest != neutral.candidate.snapshot_digest
        assert calibration.neutral_material(store=store, anchor=pair.anchor) == neutral
        declared = statistics()  # Two fixture blocks cannot claim a full calibration.
        def open_frame(holder):
            return calibration.DirectCalibration(store=store, held_claim=holder,
                campaign_id="calibration-campaign", epoch="original", anchor=pair.anchor,
                neutral=neutral, prompts=prompts, statistical=declared,
                host_state={"scope": "fixture host identity, not health PASS"})
        with original_claim(tmp_path / "private.lock") as holder:
            session = open_frame(holder)
            observe, calls = holder.observe, []
            def interrupted():
                calls.append(True)
                if len(calls) == 5:
                    raise RuntimeError("fixture interruption between original launches")
                return observe()
            monkeypatch.setattr(holder, "observe", interrupted)
            with pytest.raises(RuntimeError, match="between original"):
                session.collect()
            monkeypatch.setattr(holder, "observe", observe)
            assert len(session.launches) == 2 and session.pending is None
            first = list(session.launches)
        assert len(pids.read_text().splitlines()) == 2
        with original_claim(tmp_path / "private.lock") as holder:
            recovered = open_frame(holder)
            assert recovered.launches == first
            result = recovered.collect()
            assert recovered.launches[:2] == first
            solve = recovered.reopen(result)
            assert not solve.accepted
            assert len(recovered.launches) == 8
        assert len(pids.read_text().splitlines()) == 8
        assert len(requests_log.read_text().splitlines()) == 32
        with original_claim(tmp_path / "private.lock") as holder:
            completed = open_frame(holder)
            assert completed.collect() == result
        assert len(pids.read_text().splitlines()) == 8
        # Rehashing a substituted numeric summary cannot replace original HTTP.
        original_ref = completed.launches[0]["artifact"]
        original = dict(store.read(original_ref["locator"], original_ref["sha256"]))
        original["value"] += 1
        changed = store.write("direct-calibration-launch", original)
        altered = [{**completed.launches[0], "artifact": changed.to_dict()},
                   *completed.launches[1:]]
        with pytest.raises(calibration.RuntimeCalibrationRefused, match="original HTTP rates"):
            completed._material(altered)
        with monkeypatch.context() as patch:
            patch.setattr(calibration, "_solve", lambda *args: None)
            with pytest.raises(calibration.RuntimeCalibrationRefused, match="source moved"):
                completed.reopen(result)
        # An unresolved original launch can never be turned into another launch.
        state = json.loads((store.root / completed.state_name).read_text())
        state["solution"] = None
        state["pending"] = {"original_launch": "no completed boundary"}
        from . import status
        status.write_json(store.root, completed.state_name, state)
        with original_claim(tmp_path / "private.lock") as holder:
            uncertain = open_frame(holder)
            with pytest.raises(calibration.RuntimeCalibrationRefused, match="cannot be replay-launched"):
                uncertain.collect()
        assert len(pids.read_text().splitlines()) == 8
    finally:
        store.close()


def test_retention_fault_preserves_original_measurement_exception(tmp_path, monkeypatch):
    pair, _, pids = _fixture(tmp_path)
    store = mc.ArtifactStore(tmp_path / "captures")
    try:
        neutral = calibration.neutral_material(store=store, anchor=pair.anchor)
        def failed_measurement(*args, **kwargs):
            raise RuntimeError("original measurement failure")
        # Prospective fixture source pin; this test does not claim an HTTP run.
        monkeypatch.setattr(calibration.serving, "_measure_once", failed_measurement)
        with original_claim(tmp_path / "private.lock") as holder:
            session = calibration.DirectCalibration(store=store, held_claim=holder,
                campaign_id="calibration-campaign", epoch="original", anchor=pair.anchor,
                neutral=neutral, prompts=_prompts(pair.anchor.template), statistical=statistics(),
                host_state={"scope": "fixture, no host health warrant"})
            write = store.write
            def failed_retention(namespace, body):
                if namespace == "direct-calibration-launch":
                    raise OSError("fixture retention failure")
                return write(namespace, body)
            monkeypatch.setattr(store, "write", failed_retention)
            with pytest.raises(RuntimeError, match="original measurement failure") as raised:
                session.collect()
            assert "retention also failed" in raised.value.__notes__[0]
            assert session.pending is not None and session.launches == []
        assert not pids.exists()
    finally:
        store.close()


@pytest.mark.parametrize("dry_run", [True, False])
def test_existing_run_retains_original_default_without_changing_observation_semantics(monkeypatch, dry_run):
    from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion
    from . import run
    defaults, artifacts, snapshots = [], [], []
    original = mc.ArtifactStore.write
    original_status = run.status.write_json
    def written(store, namespace, body):
        artifacts.append(namespace)
        reference = original(store, namespace, body)
        if namespace == "direct-runtime-default":
            defaults.append((body, reference.to_dict()))
            assert dict(store.read(reference.locator, reference.sha256))["selection"] == "original_default"
        return reference
    def status_written(root, name, body, **kwargs):
        if name == "loop-run.json":
            snapshots.append(body)
        return original_status(root, name, body, **kwargs)
    monkeypatch.setattr(mc.ArtifactStore, "write", written)
    monkeypatch.setattr(run.status, "write_json", status_written)
    test_existing_main_cpu_five_iterations_preserves_canonical_champion(dry_run, runtime_only=True)
    if dry_run:
        assert defaults == [] and artifacts == [] and snapshots == []
    else:
        assert len(defaults) == len(snapshots) == 1
        body, reference = defaults[0]
        assert body["selection"] == "original_default" and body["qualified"] is False
        assert body["statistics"]["controls"]["calibration_block_count"] == 200
        assert snapshots[0]["runtime_preparation"]["default_recipe"] == reference
        assert snapshots[0]["runtime_preparation"]["status"] == "observed_not_admitted"
        assert all(row["runtime_pair"]["anchor"]["execution_digest"] == body["recipe"]["execution_digest"]
                   for row in snapshots[0]["iterations"])
        assert "direct-calibration-launch" not in artifacts


def test_actual_run_calibration_option_reaches_same_claim_before_candidates(monkeypatch):
    from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion
    from . import run
    main, selected = run.main, []
    def with_calibration(argv):
        return main([*argv, "--calibrate-runtime"])
    class CollectorBoundary:
        # Installed main/claim routing check only. The separate HTTP test above
        # exercises the real collector and real private flock through restart.
        def __init__(self, **kwargs):
            selected.append(kwargs)
        def collect(self):
            raise RuntimeError("original calibration boundary reached before candidate work")
    monkeypatch.setattr(run, "main", with_calibration)
    monkeypatch.setattr(calibration, "DirectCalibration", CollectorBoundary)
    with pytest.raises(RuntimeError, match="boundary reached before candidate"):
        test_existing_main_cpu_five_iterations_preserves_canonical_champion(False, runtime_only=True)
    assert len(selected) == 1
    frame = selected[0]
    assert frame["held_claim"]["device_id"] == "cpu"
    assert frame["campaign_id"] == "ak-loop"
    assert frame["statistical"].controls.calibration_block_count == 200
    assert frame["anchor"].execution_digest == frame["neutral"].candidate.execution_digest
    assert frame["prompts"].digest == frame["host_state"]["frozen_prompt_digest"]
