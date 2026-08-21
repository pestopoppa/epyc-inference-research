"""CPU-only integration tests for C6 admission at the discovery boundary."""
from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from . import discovery_controller as D
from .test_discovery_controller import FakeCritic, Lease


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Planner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def attest(self):
        from .test_discovery_controller import RUNTIME
        return {**D.SOL, "runtime": RUNTIME}

    def plan(self, *, context, workspace):
        self.calls.append(context)
        patch_bytes = (
            b"diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c\n"
            b"--- a/ggml/src/ggml.c\n+++ b/ggml/src/ggml.c\n"
            b"@@ -1 +1 @@\n-x\n+y\n")
        manifest = D.source_candidate.SourcePatchManifest(
            campaign_id="ak-c6-integration", proposal_id="akp-c6",
            candidate_id="akc-c6", source_tree="llama.cpp",
            production_base_commit="b" * 40,
            instrument_commit="d" * 40, change_class="fusion",
            declared_files=("ggml/src/ggml.c",),
            declared_symbols={"ggml/src/ggml.c": ("<file-scope>",)},
            mechanism_id="c6-integration",
            patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),
            patch_bytes=patch_bytes)
        return D.PlannedCandidate(
            "akh-c6-integration", "measure one exact candidate",
            "verification does not reproduce the gain",
            {"backend": "gpu", "phase": "decode"}, {"id": "c6"},
            manifest, manifest.patch_bundle_sha256)


class GatedScreens:
    def __init__(self, root: Path, effects=(.01, .22)) -> None:
        self.root = root
        self.effects = iter(effects)
        self.calls = 0

    def reconcile(self, _inflight):
        return D.Recovery("safe_to_start")

    def screen(self, *_args):
        self.calls += 1
        effect = next(self.effects)
        output = self.root / f"screen-{self.calls}"
        candidate_identity = {"source_commit": "c" * 40}
        anchor_identity = {"source_commit": "b" * 40}
        baseline = {
            "schema": "epyc.autokernel.gpu_screening_baseline.v2",
            "candidate_identity": candidate_identity,
            "anchor_identity": anchor_identity,
            "anchor_samples": [100.0, 100.0, 100.0],
        }
        baseline["baseline_sha256"] = D.schemas.content_hash(baseline)
        _write_json(output / "baseline-bank.json", baseline)
        oracle = {
            "schema": "epyc.autokernel.cross_arm_graphs_on_output_oracle.v1",
            "seed": 42, "repetitions": 3,
            "input_hashes": ["1" * 16, "2" * 16, "3" * 16],
            "output_hashes": ["4" * 16, "5" * 16, "6" * 16],
            "cross_arm_bitwise_equal": True,
            "graph_environment": {"GGML_CUDA_DISABLE_GRAPHS": None},
            "reward_admissible": True,
        }
        oracle["receipt_sha256"] = D.schemas.content_hash(oracle)
        samples = [100.0 * (1.0 + effect)] * 3
        effects = [(value - 100.0) / 100.0 for value in samples]
        measured_effect = statistics.median(effects)
        result = {
            "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
            "authority": D.AUTHORITY, "status": "complete",
            "state": "decided", "ok": True, "non_promotable": True,
            "promotion_claim": False, "hip_residency_proved": True,
            "runtime_graphs": "on", "baseline_center": 100.0,
            "candidate_samples": samples,
            "median_relative": measured_effect,
            "baseline_sha256": baseline["baseline_sha256"],
            "candidate_identity": candidate_identity,
            "graphs_on_output_oracle": oracle,
            "frame": {"metric": "tokens_per_second", "model": "fixture"},
            "sole_factor": {"kind": "source_patch"},
        }
        result["result_sha256"] = D.schemas.content_hash(result)
        result_path = output / "result.json"
        result_file_sha256 = _write_json(result_path, result)
        digest = lambda label: hashlib.sha256(label.encode()).hexdigest()
        return D.SealedScreen(
            receipt_path=str(result_path),
            result_sha256=result["result_sha256"],
            effect_fraction=measured_effect, classification="candidate",
            baseline_sha256=baseline["baseline_sha256"],
            source_proof_sha256=digest("source"),
            dispatch_proof_sha256=digest("dispatch"),
            exact_attribution_effect_fraction=measured_effect,
            target_runtime_effect_fraction=measured_effect,
            stages=("materialized", "built", "correctness", "attribution",
                    "measurement_graphs_off_screen",
                    "target_runtime_graphs_on_screen"),
            build_identity_sha256=digest("build"),
            correctness_receipt_sha256=digest("correctness"),
            attribution_receipt_sha256=digest("attribution"),
            graphs_off_receipt_sha256=digest("off"),
            graphs_on_receipt_sha256=result_file_sha256)


def _config(root: Path) -> D.ControllerConfig:
    evidence = root / "evidence"
    return D.ControllerConfig(
        root / "state", max_iterations=2, evidence_root=evidence,
        c6_admission_store_path=evidence / "c6-admission.jsonl",
        c6_admission_alpha=1.2, c6_admission_beta=1.2,
        c6_implausible_speedup_cap=32.0,
        c6_reopen_when=(
            "candidate_commit, anchor_commit, evaluator_commit, or exact "
            "measurement frame changes"),
        c6_evaluator_commit="d" * 40)


class TestC6AdmissionIntegration(unittest.TestCase):
    def test_real_controller_writes_only_after_two_fully_gated_screens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            state = D.run_controller(
                config, planner=Planner(), critic=FakeCritic(["accept"]),
                screener=GatedScreens(root / "screens"), lease=Lease())
            self.assertEqual(len(state["iterations"]), 2)
            self.assertIn("c6_admission_leg", state["iterations"][0])
            binding = state["iterations"][1]["c6_admission"]
            self.assertTrue(binding["admitted"])
            records = D.c6_reward_integrity.AdmissionReceiptStore(
                config.c6_admission_store_path).records()
            self.assertEqual(len(records), 1)
            receipt = records[0]["receipt"]
            self.assertEqual(receipt["candidate_commit"], "c" * 40)
            self.assertEqual(receipt["anchor_commit"], "b" * 40)
            self.assertEqual(receipt["evaluator_commit"], "d" * 40)
            self.assertEqual(receipt["alpha"], 1.2)
            self.assertEqual(receipt["beta"], 1.2)
            self.assertEqual(receipt["implausible_speedup_cap"], 32.0)
            self.assertEqual(receipt["reopen_when"], config.c6_reopen_when)
            self.assertAlmostEqual(receipt["first_turn_speedup"], 1.01)
            self.assertAlmostEqual(receipt["required_speedup"], 1.212)
            self.assertAlmostEqual(receipt["verification_speedup"], 1.22)
            self.assertEqual(
                records[0]["belief_capture"]["category"], "CANDIDATE")

    def test_confirmation_stopped_before_throughput_never_writes(self):
        class PartialConfirmation(GatedScreens):
            def screen(self, *args):
                result = super().screen(*args)
                if self.calls == 2:
                    return replace(
                        result,
                        target_runtime_effect_fraction=None,
                        stages=("materialized", "built", "correctness",
                                "attribution"))
                return result

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            state = D.run_controller(
                config, planner=Planner(), critic=FakeCritic(["accept"]),
                screener=PartialConfirmation(root / "screens"), lease=Lease())
            self.assertEqual(
                state["iterations"][1]["c6_admission"]["reason"],
                "verification_stopped_before_target_throughput_gate")
            self.assertEqual(
                D.c6_reward_integrity.AdmissionReceiptStore(
                    config.c6_admission_store_path).records(), [])

    def test_completed_runner_crash_window_replays_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            real_apply = D._apply_c6_admission

            def crash_after_append(config_, item, result, row, *, confirmation):
                real_apply(config_, item, result, row,
                           confirmation=confirmation)
                if confirmation:
                    raise RuntimeError("simulated controller crash after C6 append")

            screens = GatedScreens(root / "screens")
            with patch.object(D, "_apply_c6_admission", crash_after_append):
                with self.assertRaisesRegex(RuntimeError, "after C6 append"):
                    D.run_controller(
                        config, planner=Planner(),
                        critic=FakeCritic(["accept"]),
                        screener=screens, lease=Lease())
            store = D.c6_reward_integrity.AdmissionReceiptStore(
                config.c6_admission_store_path)
            self.assertEqual(len(store.records()), 1)
            resumed = D.run_controller(
                config, planner=Planner(), critic=FakeCritic([]),
                screener=screens, lease=Lease())
            self.assertTrue(resumed["complete"])
            self.assertEqual(len(store.records()), 1)
            self.assertIn("c6_admission", resumed["iterations"][1])

    def test_store_tamper_fails_closed_before_completed_reentry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            D.run_controller(
                config, planner=Planner(), critic=FakeCritic(["accept"]),
                screener=GatedScreens(root / "screens"), lease=Lease())
            path = config.c6_admission_store_path
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["belief_capture"]["value"] = 999.0
            path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError,
                    "C6 admission store failed validation"):
                D.run_controller(
                    config, planner=Planner(), critic=FakeCritic([]),
                    screener=GatedScreens(root / "unused"), lease=Lease())


if __name__ == "__main__":
    unittest.main()
