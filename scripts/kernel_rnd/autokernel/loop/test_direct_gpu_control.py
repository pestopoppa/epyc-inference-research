"""Native GPU control plumbing with private locks and synthetic hardware I/O."""
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from .. import schemas
from ..evaluator import api, controls
from . import claim, direct_gpu_control as owner, runtime_admission as admission
from .measurement_capture import ArtifactStore
from .test_direct_historical_control import held
from .test_gpu_runtime import device_body
from .test_serving_preparation import statistics


@contextmanager
def claims(tmp_path):
    with held(tmp_path, "184-191") as cpu, claim.hold(tmp_path / "gpu.lock") as gpu:
        yield cpu, gpu


def available(monkeypatch):
    declaration = controls.HistoricalWinReplayDeclaration(
        "cdna2-q8-single-x4-load-fe881-bff30", "llama_gpu", "decode", "higher_better",
        controls.ReferenceBand(**owner.fixture()["approved_reference_band"]), owner.LOCATOR, "carried_in_git")
    value = controls.HistoricalWinResolution("llama_gpu", True, declaration,
        schemas.Check(schemas.PASS, ("synthetic Git index for prospective uncommitted fixture",)),
        durability_outcome=schemas.PASS)
    monkeypatch.setattr(owner, "resolution", lambda: value)
    return value


def invoke(store, pair, **kwargs):
    return owner.run_or_reopen(store=store, held_claim=pair[0], gpu_claim=pair[1],
        campaign_id="ak-gpu-control-fixture", window_index=0,
        statistical=kwargs.pop("statistical", statistics()), **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail("replay attempted new hardware I/O or original artifact write")


def test_declared_range_is_exact_original_pairs_not_confidence_or_ratio_of_medians():
    body = owner.fixture()
    assert body["approved_reference_band"] == {"low": 0.004831498653230959, "high": 0.04435475921984722}
    assert body["original_paired_effect_median"] == 0.025496918214263697
    assert body["original_paired_effect_median"] != body["original_ratio_of_medians"]


def test_original_bench_actual_tiny_child_capture_keeps_argv_and_raw_mean(tmp_path, monkeypatch):
    from .test_serving_residency import _proof, _sampler_class
    binary = tmp_path / "llama-bench"
    binary.write_text('#!/usr/bin/python3\nprint(\'[ {"n_prompt":0,"n_gen":128,"avg_ts":21.25} ]\')\n')
    binary.chmod(0o700)
    class During:
        shutdown_resolved = True
        def __init__(self, *args):
            pass
        def start(self):
            pass
        def phase(self, phase):
            pass
        def finish(self):
            pass
        def body(self):
            return {"errors": ["synthetic tiny child, no GPU/host measurement"]}
    monkeypatch.setattr(owner.rw, "GpuDuringWork", During)
    monkeypatch.setattr(owner.bench.residency, "Sampler", _sampler_class(_proof()))
    with closing(ArtifactStore(tmp_path / "artifacts")) as store, claims(tmp_path) as pair:
        actual = owner.OriginalClaims(*pair, store=store, cpu_list="184-191")
        capture = owner.NativeCapture(store, actual, None, membership=["fixture", 0, "anchor"], deadline=None)
        value, _proof_value = owner.bench.run_once(binary, Path("/unused-model"), pp=0, tg=128,
            reps=9, timeout_s=5, capture=capture)
        raw = owner._read(store, "direct-gpu-bench-launch", capture.reference.to_dict())
        assert value == 21.25 == json.loads(raw["stdout"])[0]["avg_ts"]
        assert raw["argv"] == ["taskset", "-c", "184-191", "numactl", "--interleave=all",
            str(binary), "-m", "/unused-model", "-p", "0", "-n", "128", "-r", "9",
            "-ngl", "99", "-fa", "1", "-o", "json"]
        assert raw["returncode"] == 0 and raw["error"] is None and raw["shutdown_resolved"]
        assert raw["started_monotonic_s"] < raw["ended_monotonic_s"]
        assert raw["claim_open"] != raw["claim_close"]


def test_missing_control_setup_emits_original_observation_and_pure_reopen(tmp_path, monkeypatch):
    available(monkeypatch)
    def unavailable(**kwargs):
        raise owner.GpuControlRefused("synthetic original instrument missing")
    monkeypatch.setattr(owner, "require_available", unavailable)
    with closing(ArtifactStore(tmp_path / "artifacts")) as store, claims(tmp_path) as pair:
        resolution, observations, reference = invoke(store, pair)
        assert resolution.available
        assert all(not row.ran and row.verdict is None for row in observations.values())
        result = owner._read(store, owner.NAMESPACE, reference.to_dict())
        receipt = owner._read(store, "direct-gpu-control-beliefs", result["belief_receipt"])
        assert result["completed_launches"] == 0 and result["belief_export_error"] is None
        assert all(row["protocol_id"] == "" and row["native_verdict"] is None and row["value"] == 0
                   for row in receipt["belief_measurements"])
        assert all(row["extra"]["evidence_basis"]["observation"]["ran"] is False
                   for row in receipt["belief_measurements"])
        monkeypatch.setattr(ArtifactStore, "write", forbidden)
        monkeypatch.setattr(pair[0], "observe", forbidden)
        assert invoke(store, pair, reference=reference.to_dict())[2] == reference
        monkeypatch.undo()


def synthetic_bench(tmp_path, monkeypatch, *, fail_once=None, native_failure=False):
    available(monkeypatch)
    monkeypatch.setattr(owner, "require_available", lambda **kw: (owner.resolution(),
        owner.OriginalClaims(kw["held_claim"], kw["gpu_claim"], store=kw["store"], cpu_list="184-191")))
    monkeypatch.setattr(owner.rw, "boundary", lambda config, held, **kw:
        {"snapshot": {"marker": kw["marker"]}, "claim": held.observe(), "gpu_claim": kw["gpu_claim"].observe()})
    monkeypatch.setattr(owner, "_identities", lambda *a: {"fixture": "synthetic instrument, not hardware identity"})
    monkeypatch.setattr(owner, "_collect_t0", lambda store, *a: store.write("direct-gpu-t0", {"fixture": "T0"}))
    monkeypatch.setattr(owner, "_t0_gates", lambda *a: (api.GateResult("fixture", api.GATE_CORRECTNESS,
        schemas.Check(schemas.PASS, ("synthetic T0 gate; not actual correctness evidence",))),))
    monkeypatch.setattr(owner.rw, "launch_health", lambda *a, **k: schemas.Check(schemas.PASS,
        ("synthetic host fixture, not hardware health",)))
    calls, failed = [], []
    frame = owner.fixture()["frame"]
    def run(binary, model, *, pp, tg, reps, capture):
        key = tuple(capture.membership)
        calls.append(key)
        arm = "candidate" if str(binary).startswith(frame["candidate_build"] + "/") else "anchor"
        rng = random.Random(repr(key))
        value = 100 + rng.gauss(0, .02)
        if arm == "candidate":
            value *= 1.025
        raw_row = {"build_commit": frame[arm + "_commit"][:8], "model_filename": str(model),
            "n_prompt": pp, "n_gen": tg, "n_threads": 8, "n_gpu_layers": 99, "flash_attn": True,
            "avg_ts": value, "samples_ts": [value] * reps, "samples_ns": [1_280_000_000] * reps}
        before = capture.claims.observe()[1]
        after = capture.claims.observe()[1]
        raw = {"membership": list(key), "argv": [*frame["topology"], str(binary), "-m", str(model), *frame["explicit_flags"]],
            "env": {}, "attempt": 0, "error": None, "started_monotonic_s": 10.5, "ended_monotonic_s": 13.5,
            "claim_open": before.to_dict(), "claim_close": after.to_dict(), "shutdown_resolved": True,
            "returncode": 0, "stdout": json.dumps([raw_row]), "stderr": "", "residency": {"resident": True},
            "during_work": device_body(), "ended_at": datetime.now(timezone.utc).isoformat()}
        if key == fail_once and not failed:
            failed.append(key)
            raw["error"] = "synthetic terminal invalid arm"
        capture.raw = raw
        capture.reference = capture.store.write("direct-gpu-bench-launch", raw)
        if raw["error"]:
            if native_failure:
                raise owner.bench.BenchFailed(raw["error"])
            raise owner.GpuControlRefused(raw["error"])
        return value, raw["residency"]
    monkeypatch.setattr(owner.bench, "run_once", run)
    return calls


@pytest.mark.parametrize("native_failure", [False, True])
def test_failed_setup_continues_exact_completed_prefix_and_preserves_failed_receipt(tmp_path, monkeypatch, native_failure):
    failed_key = ("aa", 0, "candidate")
    calls = synthetic_bench(tmp_path, monkeypatch, fail_once=failed_key, native_failure=native_failure)
    # Stop at the next real preparation boundary, before a numeric solve. This
    # test covers actual run_or_reopen and BenchWindow membership, not a helper
    # counter disconnected from the runtime failure->retry route.
    def end_at_solve(*args):
        raise owner.rc.RuntimeLaunchBudgetExhausted("synthetic invocation deadline at completed calibration")
    monkeypatch.setattr(owner, "_solve_material", end_at_solve)
    with closing(ArtifactStore(tmp_path / "artifacts")) as store, claims(tmp_path) as pair:
        _, failed, first = invoke(store, pair)
        assert not failed[controls.CONTROL_HISTORICAL_WIN_REPLAY].ran
        before = list(calls)
        _, _, second = invoke(store, pair)
        assert first != second
        assert calls.count(failed_key) == 2
        assert all(calls.count(key) == 1 for key in before if key != failed_key)
        original = owner._read(store, owner.NAMESPACE, first.to_dict())
        later = owner._read(store, owner.NAMESPACE, second.to_dict())
        assert original["error"].startswith("GpuControlRefused:")
        assert later["error"].startswith("RuntimeLaunchBudgetExhausted:")
        assert later["completed_launches"] == 8
        checkpoint = owner._read(store, "direct-gpu-window-checkpoint", later["window"])
        assert checkpoint["pending"] is None and len(checkpoint["invalid"]) == 1
        assert len(checkpoint["launches"]) == len({tuple(row["membership"]) for row in checkpoint["launches"]})
        before = list(calls)
        invoke(store, pair)
        assert calls == before


def test_gpu_supplier_unavailable_checked_before_calibration(tmp_path, monkeypatch):
    calls = []
    def unavailable(**kwargs):
        calls.append("preflight")
        raise owner.GpuControlRefused("missing original fixture")
    monkeypatch.setattr(owner, "require_available", unavailable)
    fake = SimpleNamespace(store=None, held_claim=None, gpu_claim=None,
        require_control_supplier=lambda anchor: admission.RuntimeAdmission.require_control_supplier(fake, anchor),
        calibration=forbidden)
    with pytest.raises(owner.GpuControlRefused, match="original fixture"):
        admission.RuntimeAdmission._controls(fake, SimpleNamespace(backend="gpu"), 0)
    assert calls == ["preflight"]


def test_native_vectors_calibration_original_tiers_and_observation_writer(tmp_path, monkeypatch):
    from ..evaluator.test_api import window as fixture_window
    calls = synthetic_bench(tmp_path, monkeypatch)
    def identities(store, frame):
        return {arm: {"source": store.write("direct-gpu-source", {
                "commit": frame[arm + "_commit"], "tree_manifest_hex": b"synthetic source".hex()}).to_dict(),
            "source_sha256": owner.hashlib.sha256(b"synthetic source").hexdigest(),
            "binary_sha256": ("a" if arm == "anchor" else "b") * 64,
            "linkage_sha256": "c" * 64, "files": []} for arm in ("anchor", "candidate")}
    monkeypatch.setattr(owner, "_identities", identities)
    def window(store, declaration, material, blocks, stats, control_id, raw_ref):
        frame = declaration["fixture"]["frame"]
        anchor = api.AnchorIdentity(frame["anchor_commit"], "a" * 64, "c" * 64, tool="llama-bench")
        return fixture_window(anchor_at_open=anchor, anchor_at_close=anchor,
            stopping_rule_id=stats.stopping_rule.rule_id, order_seed=stats.campaign_seed,
            raw_evidence_ref=raw_ref, rule_immutability=stats.stopping_rule_commitment.verify(stats.stopping_rule),
            order_randomized=stats.order_schedule("akc-control-" + control_id.replace("_", "-")).check_observed(blocks))
    monkeypatch.setattr(owner, "_window", window)
    declared = statistics()
    declared = replace(declared, controls=replace(declared.controls, calibration_block_count=200))
    with closing(ArtifactStore(tmp_path / "artifacts")) as store, claims(tmp_path) as pair:
        resolution, observations, reference = invoke(store, pair, statistical=declared)
        assert all(row.ran for row in observations.values()), {key: row.to_dict() for key, row in observations.items()}
        assert observations[controls.CONTROL_HISTORICAL_WIN_REPLAY].promoted
        result = owner._read(store, owner.NAMESPACE, reference.to_dict())
        assert [row["tier"] for row in result["tier_evaluations"][controls.CONTROL_POSITIVE]] == ["T0", "T1"]
        assert [row["tier"] for row in result["tier_evaluations"][controls.CONTROL_HISTORICAL_WIN_REPLAY]] == ["T0", "T1", "T2"]
        magnitude = observations[controls.CONTROL_HISTORICAL_WIN_REPLAY].observed_magnitude
        assert resolution.declaration.reference_band.low < magnitude < resolution.declaration.reference_band.high
        assert result["belief_export_error"] is None and result["belief_receipt"]
        before = list(calls)
        original_source = owner.source_identity()
        monkeypatch.setattr(owner.bench, "run_once", forbidden)
        monkeypatch.setattr(owner, "source_identity", lambda: original_source)
        monkeypatch.setattr(owner, "_collect_t0", forbidden)
        monkeypatch.setattr(ArtifactStore, "write", forbidden)
        assert invoke(store, pair, reference=reference.to_dict())[2] == reference
        assert calls == before
        monkeypatch.undo()
