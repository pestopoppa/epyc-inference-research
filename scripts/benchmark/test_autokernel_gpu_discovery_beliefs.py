import copy
import math
import unittest
from pathlib import Path

from scripts.benchmark import autokernel_gpu_discovery_beliefs as beliefs
from scripts.benchmark import run_autokernel_gpu_discovery as runner
from scripts.kernel_rnd.autokernel import schemas


PRODUCER = Path(runner.__file__).resolve()


def identity(tag: str) -> dict:
    return {
        "source_commit": runner.SOURCE_COMMIT,
        "rocwmma_fattn": tag == "candidate",
        "mmq_mfma": False,
        "artifacts": {
            "binary": f"/build/{tag}/bin/llama-bench",
            "binary_sha256": ("a" if tag == "anchor" else "b") * 64,
            "libraries": {"libggml-hip.so":
                          ("c" if tag == "anchor" else "d") * 64},
        },
    }


def frame() -> dict:
    return {
        "backend": "llama_gpu", "recipe": "pp512-ngl99",
        "metric": "prefill_tokens_per_s", "metric_direction": "higher_better",
        "model": "/models/test.gguf", "model_sha256": "e" * 64,
        "source_commit": runner.SOURCE_COMMIT, "cpu_list": runner.CPU_LIST,
        "device": "AMD Instinct MI210", "architecture": "gfx90a",
    }


def runs(samples: list[float]) -> list[dict]:
    return [{"metric": sum(samples) / len(samples), "samples": samples,
             "sample_count": len(samples), "hip_residency_proved": True,
             "raw_row": {"avg_ts": sum(samples) / len(samples), "samples_ts": samples},
             "residency": [{"owned_kfd_pids": [1]}]}]


def baseline_body() -> dict:
    samples = [100.0, 102.0, 101.0]
    return {
        "schema": beliefs.BANK_SCHEMA, "status": "complete",
        "campaign_id": "ak-gpu-screen-r3", "authority": beliefs.AUTHORITY,
        "started_at": "2026-08-13T16:00:00Z", "ended_at": "2026-08-13T16:00:05Z",
        "frame": frame(),
        "sole_factor": {"name": "GGML_HIP_ROCWMMA_FATTN",
                        "anchor": "OFF", "candidate": "ON"},
        "anchor_identity": identity("anchor"),
        "candidate_identity": identity("candidate"),
        "anchor_invocations": 3,
        "anchor_samples": samples, "anchor_runs": runs(samples),
    }


def result_body(bank: dict) -> dict:
    samples = [110.0, 112.0, 111.0]
    center = sum(bank["anchor_samples"]) / 3
    effects = [(sample - center) / center for sample in samples]
    return {
        "schema": beliefs.RESULT_SCHEMA, "status": "complete",
        "campaign_id": bank["campaign_id"], "authority": beliefs.AUTHORITY,
        "started_at": bank["started_at"], "ended_at": "2026-08-13T16:00:10Z",
        "state": "decided", "ok": True, "non_promotable": True,
        "nomination": "top_k_candidate_only_not_a_keep",
        "baseline_sha256": bank["baseline_sha256"],
        "anchor_invocations": 3, "candidate_invocations": 3,
        "baseline_center": center, "candidate_samples": samples,
        "relative_effects": effects, "median_relative": sorted(effects)[1],
        "host_noise_policy": "ordinary_host_activity_recorded_not_blocking",
        "frame": bank["frame"], "sole_factor": bank["sole_factor"],
        "candidate_identity": bank["candidate_identity"],
        "candidate_runs": runs(samples), "device_sampling": {"sample_count": 2},
        "hip_residency_proved": True,
        "cpu_claim_open": {"claim_id": "cpu-one", "cpu_list": runner.CPU_LIST},
        "device_claim_open": {"claim_id": "gpu-one", "device_id": runner.DEVICE_ID},
    }


class TestGpuDiscoveryBeliefs(unittest.TestCase):
    def test_decode_frame_emits_decode_claimtuple_metrics(self) -> None:
        body = baseline_body()
        body["frame"]["recipe"] = "tg128-ngl99"
        body["frame"]["metric"] = "decode_tokens_per_s"
        bank = beliefs.attach_baseline_beliefs(body, producer_path=PRODUCER)
        result = beliefs.attach_result_beliefs(
            result_body(bank), bank=bank, producer_path=PRODUCER)
        self.assertEqual(
            bank["belief_measurements"][0]["metric"], "gpu_decode_tokens_per_s")
        self.assertEqual(
            result["belief_measurements"][1]["metric"],
            "gpu_decode_relative_effect_vs_sealed_anchor")
        self.assertIn("tg128", result["belief_measurements"][0]["measurement_id"])

    def test_baseline_and_candidate_rows_are_native_and_self_hashed(self) -> None:
        bank = beliefs.attach_baseline_beliefs(
            baseline_body(), producer_path=PRODUCER)
        result = beliefs.attach_result_beliefs(
            result_body(bank), bank=bank, producer_path=PRODUCER)

        unsigned_bank = {key: value for key, value in bank.items()
                         if key != "baseline_sha256"}
        unsigned_result = {key: value for key, value in result.items()
                           if key != "result_sha256"}
        self.assertEqual(bank["baseline_sha256"], schemas.content_hash(unsigned_bank))
        self.assertEqual(result["result_sha256"], schemas.content_hash(unsigned_result))
        self.assertEqual(len(bank["belief_measurements"]), 1)
        self.assertEqual(len(result["belief_measurements"]), 2)
        self.assertEqual(bank["belief_measurements"][0]["value"], 101.0)
        self.assertEqual(result["belief_measurements"][0]["value"], 111.0)
        self.assertEqual(result["baseline_anchor_samples"], [100.0, 102.0, 101.0])
        self.assertTrue(math.isclose(
            result["belief_measurements"][1]["value"], 10 / 101,
            rel_tol=1e-12, abs_tol=1e-12))
        for receipt in (bank, result):
            self.assertEqual(receipt["producer"]["producer_id"], beliefs.PRODUCER_ID)
            for row in receipt["belief_measurements"]:
                claimed = row.pop("measurement_sha256")
                self.assertEqual(claimed, schemas.content_hash(row))
                row["measurement_sha256"] = claimed
                self.assertFalse(row["extra"]["promotion_authority"])

    def test_serialized_pair_max_center_uses_tokens_per_mean_latency(self) -> None:
        body = baseline_body()
        body["frame"]["metric_contract"] = {
            "schema": "epyc.autokernel.serialized_pair_max_metric.v1",
            "scope": "integrity_discovery_only",
            "production_throughput_authority": False,
        }
        body["anchor_runs"][0]["metric"] = 99.0
        bank = beliefs.attach_baseline_beliefs(body, producer_path=PRODUCER)
        row = bank["belief_measurements"][0]
        self.assertEqual(row["extra"]["sealed_baseline_center"], 99.0)
        self.assertEqual(row["extra"]["baseline_center_method"],
                         "tokens_per_mean_protected_latency")
        candidate = result_body(bank)
        candidate["baseline_center"] = 99.0
        candidate["relative_effects"] = [
            (sample - 99.0) / 99.0 for sample in candidate["candidate_samples"]]
        candidate["median_relative"] = sorted(candidate["relative_effects"])[1]
        result = beliefs.attach_result_beliefs(
            candidate, bank=bank, producer_path=PRODUCER)
        self.assertEqual(result["baseline_center"], 99.0)

    def test_pre_hook_receipts_are_not_modified_or_backfilled(self) -> None:
        old = baseline_body()
        old["campaign_id"] = "ak-gpu-screen-s2-pre-hook"
        self.assertNotIn("belief_measurements", old)
        self.assertNotIn("baseline_sha256", old)

    def test_five_call_retest_binds_dynamic_reps(self) -> None:
        body = baseline_body()
        body["anchor_invocations"] = 5
        body["anchor_samples"] = [100.0, 101.0, 102.0, 103.0, 104.0]
        body["anchor_runs"] = runs(body["anchor_samples"])
        bank = beliefs.attach_baseline_beliefs(body, producer_path=PRODUCER)
        candidate = result_body(bank)
        candidate["anchor_invocations"] = 5
        candidate["candidate_invocations"] = 5
        candidate["candidate_samples"] = [110.0, 111.0, 112.0, 113.0, 114.0]
        candidate["candidate_runs"] = runs(candidate["candidate_samples"])
        candidate["baseline_center"] = sum(bank["anchor_samples"]) / 5
        candidate["relative_effects"] = [
            (sample - candidate["baseline_center"]) / candidate["baseline_center"]
            for sample in candidate["candidate_samples"]]
        candidate["median_relative"] = sorted(candidate["relative_effects"])[2]
        result = beliefs.attach_result_beliefs(
            candidate, bank=bank, producer_path=PRODUCER)
        self.assertEqual(bank["belief_measurements"][0]["reps"], 5)
        self.assertEqual(result["belief_measurements"][0]["reps"], 5)

    def test_result_refuses_a_tampered_bank_or_derived_effect(self) -> None:
        bank = beliefs.attach_baseline_beliefs(
            baseline_body(), producer_path=PRODUCER)
        tampered_bank = copy.deepcopy(bank)
        tampered_bank["anchor_samples"][0] = 1.0
        with self.assertRaisesRegex(beliefs.BeliefRefused, "sealed anchor bank"):
            beliefs.attach_result_beliefs(
                result_body(bank), bank=tampered_bank, producer_path=PRODUCER)

        bad_result = result_body(bank)
        bad_result["median_relative"] += 0.01
        with self.assertRaisesRegex(beliefs.BeliefRefused, "median_relative"):
            beliefs.attach_result_beliefs(
                bad_result, bank=bank, producer_path=PRODUCER)

    def test_refuses_missing_gpu_residency_and_authority_upgrade(self) -> None:
        bad = baseline_body()
        bad["anchor_runs"][0]["hip_residency_proved"] = False
        with self.assertRaisesRegex(beliefs.BeliefRefused, "native raw sample"):
            beliefs.attach_baseline_beliefs(bad, producer_path=PRODUCER)

        bad = baseline_body()
        bad["authority"] = "promotion"
        with self.assertRaisesRegex(beliefs.BeliefRefused, "authority"):
            beliefs.attach_baseline_beliefs(bad, producer_path=PRODUCER)


if __name__ == "__main__":
    unittest.main()
