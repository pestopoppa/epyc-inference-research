from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .. import schemas
from . import gpu_hot_residency_runner as H
from . import gpu_load_admission as A
from .split_runtime_verifier import HotResidencyIdentity


SHA = lambda char: char * 64


def identity(model: Path, *, ticks: int = 44) -> HotResidencyIdentity:
    body = {
        "schema": "epyc.autokernel.split_reward_runtime_maps.v1",
        "runtime_manifest_sha256": SHA("1"), "arm": "candidate",
        "reward_binary_sha256": SHA("2"), "hip_library_sha256": SHA("3"),
        "model_path": str(model), "model_sha256": SHA("a"), "device_id": "mi210_0",
        "kfd_pid": 123, "boot_id": "boot-a", "process_start_ticks": ticks,
        "mapped_local_sha256": {"/runtime/libggml-hip.so.0": SHA("3")},
    }
    return HotResidencyIdentity(
        runtime_manifest_sha256=SHA("1"), arm="candidate",
        reward_binary_sha256=SHA("2"), hip_library_sha256=SHA("3"),
        model_path=model, model_sha256=SHA("a"), device_id="mi210_0", kfd_pid=123,
        boot_id="boot-a", process_start_ticks=ticks,
        mapped_local_sha256=body["mapped_local_sha256"],
        identity_sha256=schemas.content_hash(body))


def decision(mode: str, ident: HotResidencyIdentity | None = None) -> A.AdmissionDecision:
    request = {"model_path": str(ident.model_path) if ident else "/models/model.gguf",
               "model_sha256": ident.model_sha256 if ident else SHA("a"),
               "device_id": ident.device_id if ident else "mi210_0",
               "expected_hot_identity_sha256": ident.identity_sha256 if ident else None}
    return A.AdmissionDecision(
        policy_version="site-v1", policy_sha256=SHA("b"), policy_file_sha256=SHA("c"),
        effective_context_sha256=SHA("d"), request=request, profile=None,
        actor_recommendation=None, mode=mode, reason="test", disqualifiers=(),
        decision_sha256=SHA("e"))


class FakeLease:
    def __init__(self, events: list[str], name: str) -> None:
        self.events, self.name, self.released = events, name, False
    def release(self) -> None:
        if not self.released:
            self.released = True
            self.events.append(f"release:{self.name}")


class FakeWindow:
    def __init__(self, events: list[str]) -> None: self.events = events
    def acquire(self) -> FakeLease:
        self.events.append("acquire:cpu-lock")
        return FakeLease(self.events, "cpu-lock")


class FakeProcess:
    def __init__(self, events: list[str]) -> None:
        self.events, self.alive, self.closed = events, True, False
    def is_alive(self) -> bool: return self.alive
    def close(self) -> None:
        self.closed = True; self.alive = False; self.events.append("close:process")


class HotResidencyRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / "model.gguf"; self.model.write_bytes(b"model")
        self.ident = identity(self.model.resolve())
        self.events: list[str] = []
        self.processes: list[FakeProcess] = []
        self.claims: list[FakeLease] = []
        self.current_proof = H.ResidencyProof(self.ident, (123,), (), 99)
        def acquire_claim() -> FakeLease:
            self.events.append("acquire:claim")
            claim = FakeLease(self.events, "claim"); self.claims.append(claim); return claim
        def load() -> FakeProcess:
            self.events.append("load")
            process = FakeProcess(self.events); self.processes.append(process); return process
        def prove(_process: FakeProcess) -> H.ResidencyProof:
            self.events.append("proof")
            return self.current_proof
        self.runner = H.HotResidencyRunner(
            window=FakeWindow(self.events), decision_validator=lambda value: self.events.append(
                f"validate:{value.mode}"), claim_acquirer=acquire_claim, loader=load,
            residency_prober=prove)

    def test_cold_serialized_lock_spans_load_and_proof_only(self) -> None:
        session = self.runner.start(decision("cold_serialized"))
        self.assertEqual(session.mode, "cold_serialized")
        self.assertEqual(self.events, ["validate:cold_serialized", "acquire:claim",
                                       "acquire:cpu-lock", "load", "proof", "release:cpu-lock"])
        session.validate_hot()
        self.assertEqual(self.events[5:], ["release:cpu-lock", "proof"])
        session.close()
        self.assertEqual(self.events[-2:], ["close:process", "release:claim"])

    def test_cold_overlap_never_touches_cpu_lock_after_sealed_validation(self) -> None:
        session = self.runner.start(decision("cold_overlap"))
        self.assertEqual(session.mode, "cold_overlap")
        self.assertEqual(self.events, ["validate:cold_overlap", "acquire:claim", "load", "proof"])
        session.close()

    def test_hot_resident_reuses_one_alive_process_without_reload_or_lock(self) -> None:
        first = self.runner.start(decision("cold_serialized"))
        self.events.clear()
        reused = self.runner.start(decision("hot_resident", self.ident))
        self.assertIs(first, reused)
        self.assertEqual(reused.mode, "hot_resident")
        self.assertEqual(self.events, ["validate:hot_resident", "proof"])
        self.assertEqual(len(self.processes), 1)
        first.close()

    def test_identity_mismatch_closes_stale_session_and_reverts_to_serialized_cold(self) -> None:
        first = self.runner.start(decision("cold_serialized"))
        self.current_proof = H.ResidencyProof(identity(self.model.resolve(), ticks=45), (123,), (), 99)
        self.events.clear()
        replacement = self.runner.start(decision("hot_resident", self.ident))
        self.assertIsNot(first, replacement)
        self.assertEqual(replacement.mode, "cold_serialized")
        self.assertEqual(self.events, ["validate:hot_resident", "proof", "close:process", "release:claim",
                                       "acquire:claim", "acquire:cpu-lock", "load",
                                       "proof", "release:cpu-lock"])
        self.assertEqual(len(self.processes), 2)
        replacement.close()

    def test_dead_process_refuses_hot_measurement_and_releases_claim(self) -> None:
        session = self.runner.start(decision("cold_serialized"))
        self.processes[0].alive = False
        with self.assertRaises(H.HotResidencyLost):
            session.validate_hot()
        self.assertTrue(session.closed)
        self.assertTrue(self.claims[0].released)
        self.assertIsNone(self.runner.session)

    def test_setup_failure_releases_window_process_and_claim_without_session(self) -> None:
        self.current_proof = H.ResidencyProof(self.ident, (999,), (), 99)
        with self.assertRaisesRegex(H.HotResidencyError, "identity KFD PID"):
            self.runner.start(decision("cold_serialized"))
        self.assertIsNone(self.runner.session)
        self.assertEqual(self.events, ["validate:cold_serialized", "acquire:claim",
                                       "acquire:cpu-lock", "load", "proof", "release:cpu-lock",
                                       "close:process", "release:claim"])

    def test_foreign_kfd_refuses_and_mints_no_hot_identity(self) -> None:
        self.current_proof = H.ResidencyProof(self.ident, (123,), (888,), 99)
        with self.assertRaisesRegex(H.HotResidencyError, "foreign KFD"):
            self.runner.start(decision("cold_overlap"))
        self.assertIsNone(self.runner.session)
        self.assertTrue(self.claims[0].released)
        self.assertTrue(self.processes[0].closed)

    def test_validator_failure_does_not_acquire_any_lock_or_claim(self) -> None:
        self.runner._decision_validator = lambda _value: (_ for _ in ()).throw(RuntimeError("unsealed"))
        with self.assertRaisesRegex(RuntimeError, "unsealed"):
            self.runner.start(decision("cold_overlap"))
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
