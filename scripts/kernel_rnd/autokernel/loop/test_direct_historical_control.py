"""Hermetic original-control ownership and replay, never inference/measurements."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from . import direct_historical_control as owner
from .claim import HeldCpuClaim
from .measurement_capture import ArtifactStore


@contextmanager
def held(tmp_path, cpu_list="0-95"):
    path = tmp_path / "fixture.lock"
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        value = HeldCpuClaim({"device_id": "cpu", "cpu_list": cpu_list, "regions": ["fixture"]}, [path])
        try:
            yield value
        finally:
            value._closing()
            fcntl.flock(stream, fcntl.LOCK_UN)
            value._released_now()


def invoke(store, claim, **kwargs):
    return owner.run_or_reopen(store=store, held_claim=claim,
        campaign_id="ak-test-historical-original", window_index=0, **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail("replay attempted original launch/write/live observation")


def test_available_history_missing_execution_is_retained_not_backend_unavailable(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(owner.lc, "INSTRUMENT_BINARY", tmp_path / "absent" / "llama-bench")
    with held(tmp_path) as claim:
        resolution, observation, ref = invoke(store, claim)
        assert resolution.available
        assert not observation.ran
        assert "missing" in observation.could_not_run_reason
        assert observation.evidence_ref
        monkeypatch.setattr(owner, "_collect", forbidden)
        # Preserve the original source identity here: the reader compares it,
        # and changing the callable above intentionally changes that identity.
        original = owner._read(store, owner.NAMESPACE, ref.to_dict())
        declaration = owner._read(store, "direct-historical-declaration", original["declaration"])
        monkeypatch.setattr(owner, "source_identity", lambda: declaration["source"])
        monkeypatch.setattr(ArtifactStore, "write", forbidden)
        monkeypatch.setattr(claim, "observe", forbidden)
        fresh = ArtifactStore(store.root)
        reopened = invoke(fresh, claim, reference=ref.to_dict())
        assert reopened[0] == resolution
        assert reopened[1].to_dict() == observation.to_dict()
        assert reopened[2] == ref
        monkeypatch.undo()


def test_repeated_fresh_call_cannot_reissue_failed_attempt(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(owner.lc, "INSTRUMENT_BINARY", tmp_path / "absent" / "llama-bench")
    with held(tmp_path) as claim:
        invoke(store, claim)
        with pytest.raises(owner.HistoricalControlRefused, match="already issued"):
            invoke(store, claim)


@pytest.mark.parametrize("change", ["campaign", "window", "frame", "source"])
def test_original_reference_cannot_relabel_identity(tmp_path, monkeypatch, change):
    store = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(owner.lc, "INSTRUMENT_BINARY", tmp_path / "absent" / "llama-bench")
    with held(tmp_path) as claim:
        _, _, ref = invoke(store, claim)
        args = dict(store=store, held_claim=claim, campaign_id="ak-test-historical-original",
                    window_index=0, reference=ref.to_dict())
        if change == "campaign":
            args["campaign_id"] = "ak-foreign"
        elif change == "window":
            args["window_index"] = 1
        elif change == "frame":
            monkeypatch.setattr(owner.lc, "PROMPT_TOKENS", 2048)
        else:
            monkeypatch.setattr(owner, "source_identity", lambda: {"foreign": True})
        with pytest.raises(owner.HistoricalControlRefused):
            owner.run_or_reopen(**args)


def test_actual_rule_commitment_precedes_measurement_and_reopens_without_commit(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    rule = owner.lc._control_stopping_rule()
    declaration = {"control_campaign_id": "ak-test-historical-rule"}
    reference = owner._stopping_commitment(store, declaration, rule)
    measured_at = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(owner.st.StoppingRuleCommitment, "commit", forbidden)
    monkeypatch.setattr(ArtifactStore, "write", forbidden)
    value = owner._stopping_commitment(store, declaration, rule,
        reference=reference.to_dict(), measured_at=measured_at)
    assert value.verify(rule).outcome == "PASS"
    assert value.committed_at <= measured_at
    with pytest.raises(owner.HistoricalControlRefused, match="before measurement"):
        owner._stopping_commitment(store, declaration, rule,
            reference=reference.to_dict(), measured_at="2020-01-01T00:00:00+00:00")


def test_deadline_refuses_before_any_launch(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(owner.lc, "_measure", forbidden)
    with held(tmp_path) as claim:
        resolution, observation, _ = invoke(store, claim, deadline_monotonic_s=0.0)
        assert resolution.available and not observation.ran
        assert "budget exhausted" in observation.could_not_run_reason


def test_smaller_original_claim_is_explicit_not_run_not_smaller_control(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(owner.lc, "_measure", forbidden)
    with held(tmp_path, cpu_list="0") as claim:
        resolution, observation, _ = invoke(store, claim)
        assert resolution.available and not observation.ran
        assert "CPU footprint 0-95 is not covered" in observation.could_not_run_reason


def synthetic_execution(tmp_path, monkeypatch, *, failed_op=False, escaped=False):
    """Original native shapes with deterministic synthetic outputs, not findings.

    Only host/process I/O is substituted. Existing T0 parsers, gates, raw
    validation, calibration solve, paired reducer and control runner are real.
    """
    from .test_search_window_owner import configuration
    from .test_search_window_preflight import _stat
    from ..execution import t0_provider as tp
    source = tmp_path / "instrument"
    binaries = source / "build" / "bin"
    binaries.mkdir(parents=True)
    for name in (*owner.lc.INSTRUMENT_BUILD_TARGETS, "libggml-base.so.0", "libggml-cpu.so.0", "libggml.so.0"):
        (binaries / name).write_bytes(b"synthetic-not-executable:" + name.encode())
    monkeypatch.setattr(owner.lc, "INSTRUMENT_BINARY", binaries / "llama-bench")
    monkeypatch.setattr(owner.lc, "INSTRUMENT_ROOT", source)
    monkeypatch.setattr(tp, "_complete_anchor_toolchain", lambda *args: ("synthetic-gcc", "1"))
    original_window = owner.inference_window.InferenceCallWindow
    monkeypatch.setattr(owner.inference_window, "InferenceCallWindow", lambda **kwargs:
        original_window(tmp_path / "synthetic-inference-window.lock", **kwargs))

    class OriginalProcesses:
        def __init__(self, **kwargs):
            pass

        def run(self, argv, *, env, cwd, timeout_s):
            if any("verify_ggml_linkage.sh" in arg for arg in argv):
                fixture = Path(tp.__file__).parent / "fixtures" / "recorded_t0_linkage_pass.txt"
                if not fixture.exists():
                    fixture = next(Path(tp.__file__).parents[1].rglob("recorded_t0_linkage_pass.txt"))
                text = fixture.read_text().replace("/mnt/raid0/llm/llama.cpp-experimental/build-v8-sanitize/bin", str(binaries))
            elif "--help" in argv:
                text = ("Usage: test-backend-ops [mode] [-o <op,..>] [-b <backend>] "
                    "[--output <console|sql|csv>] [--suite-seed <n>] "
                    "[--autokernel-properties] [-j <n>]\n")
            elif str(binaries / "test-backend-ops") in argv:
                ops = owner.correctness.MANDATORY_BACKEND_OPS
                rows = [f"  {op}(type=f32): {'FAIL' if failed_op and index == 0 else 'OK'}"
                        for index, op in enumerate(ops)]
                text = ("Testing 1 devices\n\nBackend 1/1: CPU\n" + "\n".join(rows)
                        + f"\n  {len(ops) - int(failed_op)}/{len(ops)} tests passed\n"
                        + f"  Backend CPU: {'FAIL' if failed_op else 'OK'}\n"
                        + ("0/1 backends passed\nFAIL\n" if failed_op else "1/1 backends passed\nOK\n"))
            else:
                assert str(binaries / "llama-completion") in argv
                assert env["GGML_IQK"] in ("0", "1")
                text = "The capital is Paris."
            return tp.CompletedProcess(tuple(argv), tuple(sorted(env.items())), cwd, 0, text, "",
                .001, False, False, (12345,) if escaped else ())

    monkeypatch.setattr(tp, "SubprocessRunner", OriginalProcesses)
    monkeypatch.setattr(owner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=b"synthetic-original-git-tree"))
    # Retained evidence resolution is captured before replacing subprocess; it
    # must not use the synthetic source-tree command for its GitTrackedIndex.
    resolution = owner.controls.HistoricalWinReplayDeclaration("iqk-prefill-port", "llama_cpu", "prefill",
        "higher_better", owner.controls.ReferenceBand(.03, .60), owner.LOCATOR, "carried_in_git")
    raw = (Path(owner.__file__).resolve().parents[4] / owner.LOCATOR).read_bytes()
    def resolve(store):
        return store.write("direct-historical-resolution", {
            "resolution": owner.controls.HistoricalWinResolution("llama_cpu", True, resolution,
                owner.schemas.Check("PASS", ("synthetic source index, original Git-carried bytes",)),
                durability_outcome="PASS").to_dict(),
            "repo_relative_locator": owner.LOCATOR, "original_size_bytes": len(raw),
            "original_sha256": hashlib.sha256(raw).hexdigest()})
    monkeypatch.setattr(owner, "_resolve", resolve)
    monkeypatch.setattr(owner.lc, "_linkage", lambda *args: ("2" * 64, "synthetic-linkage"))
    def preflight(root, **kwargs):
        (root / "preflight.json").write_text(json.dumps({"checks": {
            "synthetic_fixture_host": {"outcome": "PASS", "reasons": ["synthetic fixture only"]}}}))
    monkeypatch.setattr(owner.lc, "_write_preflight", preflight)
    config_root = tmp_path / "host"
    config_root.mkdir()
    config = configuration(config_root, tuple(range(96)))
    for pid, parent in ((os.getpid(), 1), (1, 0)):
        proc = config_root / "proc" / str(pid)
        proc.mkdir()
        (proc / "stat").write_text(_stat(pid, parent))
        (proc / "cgroup").write_text("0::/fixture")
        (proc / "cmdline").write_bytes(b"/fixture/python\0")
    monkeypatch.setattr(owner.rw, "configuration", lambda *args, **kwargs: config)
    snapshot = owner.rw.snapshot
    def synthetic_snapshot(*args, **kwargs):
        body = snapshot(*args, **kwargs)
        body["storage"]["free_bytes"] = 300 * 1024 ** 3  # Explicit synthetic host quantity.
        return body
    monkeypatch.setattr(owner.rw, "snapshot", synthetic_snapshot)
    calls = []
    def measure(*, label, blocks, claim, held_claim, candidate_iqk, anchor_iqk,
                output_root, identity, anchor, **kwargs):
        assert held_claim.owner is claim
        held_claim.attest()
        if label == "historical_win_replay":
            assert writes.index("direct-historical-stopping-rule") < len(writes)
        calls.append(label)
        rng = random.Random(1 if label == "aa_calibration" else 2)
        count = owner.lc._fresh_pairs_per_block()
        rows, paired = [], []
        started = datetime.now(timezone.utc).isoformat()
        schedule = owner.st.OrderSchedule.derive(campaign_seed=identity.campaign_seed + "/" + label,
            candidate_id="akc-control-" + label, base_blocks=blocks)
        for index in range(blocks):
            center = 100 + rng.gauss(0, .1)
            anchors = tuple(center + rng.gauss(0, .01) for _ in range(count))
            candidates = tuple(value * (1.4 if candidate_iqk == "1" else 1 + rng.gauss(0, .001)) for value in anchors)
            block = owner.st.PairedBlock(index, owner.lc._unit_id(label=label, prompt=512),
                "selection", schedule.order_for(index), anchors, candidates,
                measured_at=started)
            paired.append(block)
            rows.append({"paired_block": block.to_list(), "complete": True, "refusals": [],
                "plan": {"pairs": count}, "invocations": [{}] * (2 * count),
                "checks": [["fixture_original_host", {"outcome": "PASS", "reasons": ["synthetic fixture"]}]]})
        receipt = lambda iqk: {"params": dict(owner.lc._params(prompt=512), ggml_iqk=iqk),
            "constructor_id": "ak.microbench.llama_cpu.prefill/v1", "constructor_sha256": "3" * 64,
            "argv_sha256": "4" * 64, "binary_path": str(binaries / "llama-bench"),
            "binary_sha256": hashlib.sha256((binaries / "llama-bench").read_bytes()).hexdigest(),
            "source_root": str(source), "library_path": str(binaries)}
        raw = {"schema": "epyc.autokernel.microbench_raw_vector.v1", "recipe_id": owner.lc.RECIPE_ID,
            "candidate_id": "akc-control-" + label,
            "campaign_seed_sha256": hashlib.sha256((identity.campaign_seed + "/" + label).encode()).hexdigest(),
            "complete": True, "refusals": [], "order_control": {"outcome": "PASS", "reasons": []},
            "order_schedule": {"orders": [row.order for row in paired]},
            "candidate_receipt": receipt(candidate_iqk), "anchor_receipt": receipt(anchor_iqk),
            "anchor_identity": anchor.to_dict(), "blocks": rows, "checks": [],
            "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat()}
        (output_root / "raw").mkdir(exist_ok=True)
        (output_root / "raw" / (label + ".json")).write_text(json.dumps(raw))
        return owner.lc._load_recorded_material(output_root, identity=identity, label=label,
            expected_blocks=blocks, prompt=512, candidate_iqk=candidate_iqk, anchor_iqk=anchor_iqk)[0]
    writes = []
    original_write = ArtifactStore.write
    def writing(self, namespace, body):
        writes.append(namespace)
        return original_write(self, namespace, body)
    monkeypatch.setattr(ArtifactStore, "write", writing)
    monkeypatch.setattr(owner.lc, "_measure", measure)
    return calls


def test_original_synthetic_processes_calibration_tiers_and_pure_reopen(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    calls = synthetic_execution(tmp_path, monkeypatch)
    with held(tmp_path) as claim:
        resolution, observation, ref = invoke(store, claim)
        assert observation.ran, observation.could_not_run_reason
        assert calls == ["aa_calibration", "neutral_calibration", "historical_win_replay"]
        assert observation.promoted, observation.verdict.to_dict()
        result = owner._read(store, owner.NAMESPACE, ref.to_dict())
        assert [row["tier"] for row in result["tier_evaluations"]] == ["T0", "T1", "T2"]
        monkeypatch.setattr(owner.lc, "_measure", forbidden)
        monkeypatch.setattr(owner.tp, "SubprocessRunner", forbidden)
        monkeypatch.setattr(ArtifactStore, "write", forbidden)
        monkeypatch.setattr(claim, "observe", forbidden)
        reopened = invoke(ArtifactStore(store.root), claim, reference=ref.to_dict())
        assert reopened[0] == resolution and reopened[1].to_dict() == observation.to_dict()
        assert reopened[2] == ref
        monkeypatch.undo()


def test_failed_original_t0_stops_before_performance(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    calls = synthetic_execution(tmp_path, monkeypatch, failed_op=True)
    with held(tmp_path) as claim:
        resolution, observation, _ = invoke(store, claim)
        assert resolution.available and not observation.ran
        assert "T0 failed" in observation.could_not_run_reason
        assert not calls


def test_undrained_original_process_is_not_retryable_control_failure(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    calls = synthetic_execution(tmp_path, monkeypatch, escaped=True)
    with held(tmp_path) as claim:
        with pytest.raises(owner.tp.ProcessEscaped):
            invoke(store, claim)
        assert not calls


def test_native_measurement_failure_cannot_be_treated_as_safe_retry(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    synthetic_execution(tmp_path, monkeypatch)
    def failed_measure(**kwargs):
        raise RuntimeError("original native raw refusal: owned child may still be live")
    monkeypatch.setattr(owner.lc, "_measure", failed_measure)
    with held(tmp_path) as claim:
        with pytest.raises(RuntimeError, match="owned child may still be live"):
            invoke(store, claim)


@pytest.mark.parametrize("change", ["raw", "anchor", "arm"])
def test_original_raw_and_build_joins_refuse_changed_material(tmp_path, monkeypatch, change):
    store = ArtifactStore(tmp_path / "artifacts")
    synthetic_execution(tmp_path, monkeypatch)
    with held(tmp_path) as claim:
        _, observation, reference = invoke(store, claim)
        assert observation.promoted
        result = owner._read(store, owner.NAMESPACE, reference.to_dict())
        declaration = owner._read(store, "direct-historical-declaration", result["declaration"])
        material = owner._read(store, "direct-historical-material", result["material"])
        path = Path(declaration["root"]) / "raw" / "historical_win_replay.json"
        raw = json.loads(path.read_text())
        if change == "raw":
            raw["blocks"][0]["paired_block"][8][0] *= 2
        elif change == "anchor":
            raw["anchor_identity"]["binary_sha256"] = "9" * 64
        else:
            raw["candidate_receipt"]["binary_sha256"] = "9" * 64
        path.write_text(json.dumps(raw))
        # Original receipt catches changed bytes. Rebinding a copied material
        # to a new CAS object still cannot change the original instrument join.
        if change != "raw":
            material["raw"]["historical_win_replay"] = store.write("direct-historical-microbench", raw).to_dict()
        with pytest.raises(owner.HistoricalControlRefused, match="raw"):
            owner._compose(store, declaration, material, "synthetic-mutated-material")
