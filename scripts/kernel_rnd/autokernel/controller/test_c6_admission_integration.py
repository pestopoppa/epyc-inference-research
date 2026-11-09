"""CPU-only integration tests for C6 admission at the discovery boundary."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark.run_autokernel_gpu_discovery import screen_effect

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
    def __init__(self, root: Path, effects=(.01, .22),
                 pair_max: bool = False) -> None:
        self.root = root
        self.effects = iter(effects)
        self.pair_max = pair_max
        self.calls = 0

    def mutate_result(self, result: dict) -> None:
        """Test seam: alter the producer record before it is sealed."""

    def sealed_effect(self, result: dict) -> float:
        return self.measured_effect

    def reconcile(self, _inflight):
        return D.Recovery("safe_to_start")

    def screen(self, *_args):
        self.calls += 1
        effect = next(self.effects)
        output = self.root / f"screen-{self.calls}"
        center = 100.0
        anchor_samples = [center - self.calls * 0.1, center,
                          center + self.calls * 0.1]
        anchor_runs = [{"metric": center, "samples": anchor_samples}]
        # Sample spread makes the effect depend on the estimator used.
        value = center * (1.0 + effect)
        samples = [value * .999, value, value * 1.001]
        if self.pair_max:
            samples = [item * 1.02 for item in samples]
            candidate_runs = [{"metric": value, "samples": samples}]
        else:
            candidate_runs = [{"metric": value * 3.0, "samples": samples}]
        scored = screen_effect(
            anchor_samples=anchor_samples, anchor_runs=anchor_runs,
            candidate_samples=samples, candidate_runs=candidate_runs,
            pair_max=self.pair_max)
        estimator = scored["estimator"]
        self.measured_effect = scored["effect"]
        candidate_identity = {"source_commit": "c" * 40}
        anchor_identity = {"source_commit": "b" * 40}
        baseline = {
            "schema": "epyc.autokernel.gpu_screening_baseline.v2",
            "candidate_identity": candidate_identity,
            "anchor_identity": anchor_identity,
            # Every real repetition measures its own anchor bank, so its
            # content hash differs while the centre stays comparable.
            "anchor_samples": anchor_samples,
            "anchor_runs": anchor_runs,
            "estimator": estimator,
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
        result = {
            "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
            "authority": D.AUTHORITY, "status": "complete",
            "state": "decided", "ok": True, "non_promotable": True,
            "promotion_claim": False, "hip_residency_proved": True,
            "runtime_graphs": "on", "baseline_center": scored["center"],
            "model_size_bytes": 1_000_000_000,
            "candidate_samples": samples,
            "median_relative": scored["effect"],
            "estimator": estimator,
            "candidate_statistic": scored["candidate_statistic"],
            "candidate_runs": candidate_runs,
            "baseline_sha256": baseline["baseline_sha256"],
            "candidate_identity": candidate_identity,
            "graphs_on_output_oracle": oracle,
            "frame": {
                "backend": "llama_gpu", "recipe": "decode_tg128",
                "metric": "tokens_per_second", "n_prompt": 0, "n_gen": 128,
                "model": "fixture", "architecture": "gfx90a",
                "metric_contract": {"schema": (
                    "epyc.autokernel.serialized_pair_max_metric.v1"
                    if self.pair_max else
                    "epyc.autokernel.native_llama_bench_metric.v1")}},
            "sole_factor": {"kind": "source_patch"},
        }
        self.mutate_result(result)
        measured_effect = self.sealed_effect(result)
        result["result_sha256"] = D.schemas.content_hash(result)
        result_path = output / "result.json"
        result_file_sha256 = _write_json(result_path, result)
        c6_body = {
            "schema": "epyc.autokernel.c6_correctness_receipt.v1",
            "result": "PASS", "native_execution": True,
            "wrapper_used": False, "semantic_judge_gating": False,
            "determinism": {"correct": True},
            "numeric_verdicts": [{"correct": True}] * 3,
        }
        c6_body["receipt_sha256"] = D.gpu_source_proofs._hash(c6_body)
        c6_path = output / "proof" / "correctness" / "c6-receipt.json"
        _write_json(c6_path, c6_body)
        c6_ref = D.gpu_source_proofs.load_receipt(
            c6_path, schema="epyc.autokernel.c6_correctness_receipt.v1")
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
            c6_correctness_receipt_sha256=c6_ref["file_sha256"],
            c6_correctness_ref=c6_ref,
            attribution_receipt_sha256=digest("attribution"),
            graphs_off_receipt_sha256=digest("off"),
            graphs_on_receipt_sha256=result_file_sha256)


def _config(root: Path) -> D.ControllerConfig:
    evidence = root / "evidence"
    facts = (Path(D.__file__).resolve().parent.parent /
             "substrate_facts.json").resolve()
    return D.ControllerConfig(
        root / "state", max_iterations=2, evidence_root=evidence,
        c6_admission_store_path=evidence / "c6-admission.jsonl",
        c6_admission_alpha=1.2, c6_admission_beta=1.2,
        c6_roofline_authority={
            "schema": "epyc.autokernel.c6_roofline_authority.v1",
            "gpu_part": "gfx90a", "facts_path": str(facts),
            "facts_sha256": hashlib.sha256(facts.read_bytes()).hexdigest(),
            "workload": "decode_tg128", "batch_size": 1,
            "dense_model_bytes_per_token": True},
        c6_reopen_when=(
            "candidate_commit, anchor_commit, evaluator_commit, or exact "
            "measurement frame changes"),
        c6_evaluator_commit="d" * 40)


class TestC6AdmissionIntegration(unittest.TestCase):
    def setUp(self):
        def reopen_fixture_bundle(path):
            c6_path = Path(path).parent / "correctness" / "c6-receipt.json"
            c6_ref = D.gpu_source_proofs.load_receipt(
                c6_path, schema="epyc.autokernel.c6_correctness_receipt.v1")
            return SimpleNamespace(
                correctness={"body": {"c6_correctness": c6_ref}})

        self.real_bundle_loader = D.gpu_source_evidence.load_gpu_source_evidence_bundle
        self.bundle_loader = patch.object(
            D.gpu_source_evidence, "load_gpu_source_evidence_bundle",
            side_effect=reopen_fixture_bundle)
        self.bundle_loader.start()
        self.addCleanup(self.bundle_loader.stop)

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
            self.assertEqual(receipt["implausible_speedup_cap"], 14.333)
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

    def test_pair_max_estimator_recomputes_from_the_run_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            state = D.run_controller(
                config, planner=Planner(), critic=FakeCritic(["accept"]),
                screener=GatedScreens(root / "screens", pair_max=True),
                lease=Lease())
            self.assertTrue(state["iterations"][1]["c6_admission"]["admitted"])
            receipt = D.c6_reward_integrity.AdmissionReceiptStore(
                config.c6_admission_store_path).records()[0]["receipt"]
            # The sample median is deliberately 2% away from the run metric.
            self.assertAlmostEqual(receipt["verification_speedup"], 1.22)

    def test_effect_differing_by_one_ulp_from_producer_formula_is_refused(self):
        class Forged(GatedScreens):
            def mutate_result(self, result):
                result["median_relative"] = math.nextafter(
                    result["median_relative"], 1.0)

            def sealed_effect(self, result):
                return result["median_relative"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Positive control: the producer's exact value is admitted.
            D._c6_admission_leg(
                _config(root), Planner().plan(context=None, workspace=None),
                GatedScreens(root / "honest").screen(),
                evaluator_commit="d" * 40)
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError, "throughput reduction changed"):
                D._c6_admission_leg(
                    _config(root), Planner().plan(context=None, workspace=None),
                    Forged(root / "screens").screen(), evaluator_commit="d" * 40)

    def test_median_statistic_not_derived_from_samples_is_refused(self):
        class Forged(GatedScreens):
            def mutate_result(self, result):
                # The effect stays honest, so only the statistic check can fire.
                result["candidate_statistic"] *= 1.5

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError, "throughput reduction changed"):
                D._c6_admission_leg(
                    _config(root), Planner().plan(context=None, workspace=None),
                    Forged(root / "screens").screen(), evaluator_commit="d" * 40)

    def test_real_frame_change_still_breaks_verification_identity(self):
        class FrameDrift(GatedScreens):
            def mutate_result(self, result):
                if self.calls == 2:
                    result["frame"]["model"] = "a-different-model"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError,
                    "changed candidate or frame authority"):
                D.run_controller(
                    _config(root), planner=Planner(),
                    critic=FakeCritic(["accept"]),
                    screener=FrameDrift(root / "screens"), lease=Lease())

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

    def test_capture_bound_to_exact_live_producer_digest_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            D.run_controller(
                config, planner=Planner(), critic=FakeCritic(["accept"]),
                screener=GatedScreens(root / "screens"), lease=Lease())
            path = config.c6_admission_store_path
            envelope = json.loads(path.read_text(encoding="utf-8"))
            capture = envelope["belief_capture"]
            capture["producer_sha256"] = "f" * 64
            capture["capture_sha256"] = D.c6_reward_integrity.sha256_json({
                key: value for key, value in capture.items()
                if key != "capture_sha256"})
            path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError, "live module"):
                D.run_controller(
                    config, planner=Planner(), critic=FakeCritic([]),
                    screener=GatedScreens(root / "unused"), lease=Lease())

    def test_restart_recursively_reopens_both_native_c6_legs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            state = D.run_controller(
                config, planner=Planner(), critic=FakeCritic(["accept"]),
                screener=GatedScreens(root / "screens"), lease=Lease())
            first = state["iterations"][1]["c6_admission_first_leg"]
            Path(first["c6_correctness_ref"]["path"]).unlink()
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError, "cannot reopen"):
                D.run_controller(
                    config, planner=Planner(), critic=FakeCritic([]),
                    screener=GatedScreens(root / "unused"), lease=Lease())

    def test_roofline_carrier_tamper_and_unknown_gpu_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            state = D.run_controller(
                config, planner=Planner(), critic=FakeCritic(["accept"]),
                screener=GatedScreens(root / "screens"), lease=Lease())
            carrier = dict(state["iterations"][1][
                "c6_admission_first_leg"]["roofline_speedup_cap"])
            carrier["max_speedup"] = 999.0
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError, "roofline carrier"):
                D._validate_c6_roofline_carrier(carrier)
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError, "roofline authority"):
                replace(config, c6_roofline_authority={
                    **config.c6_roofline_authority, "gpu_part": "gfx999"})


class TestClassificationCarriers(unittest.TestCase):
    def test_classified_result_keeps_hotspots_and_c6_receipts(self):
        digest = lambda label: hashlib.sha256(label.encode()).hexdigest()
        hotspot = {"symbol": "q5_hot", "share": 0.5}
        screen = D.SealedScreen(
            receipt_path="/r", result_sha256=digest("r"), effect_fraction=0.1,
            classification="candidate", baseline_sha256=digest("b"),
            source_proof_sha256=digest("s"), dispatch_proof_sha256=digest("d"),
            series_key=digest("series"), hotspots=(hotspot,),
            c6_correctness_receipt_sha256=digest("c6"),
            c6_correctness_ref={"path": "/c6"})
        item = SimpleNamespace(hypothesis_id="akh-x")
        with patch.object(D, "_series_identity", return_value=(digest("series"), ()),
                          create=True), \
                patch.object(D, "_pooled_component_effects", return_value={}):
            out = D._classified_result(
                {"iterations": []}, item, screen, None)
        self.assertEqual(out.hotspots, (hotspot,))
        self.assertEqual(out.c6_correctness_ref, {"path": "/c6"})
        self.assertEqual(
            out.c6_correctness_receipt_sha256, digest("c6"))


if __name__ == "__main__":
    unittest.main()
