import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark import dflash2_beliefs as B


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, base: Path):
        self.base = base
        self.root = base / "campaign"
        self.root.mkdir()
        identity = base / "identity"; identity.mkdir()
        self.files = {}
        for name, payload in (("binary", b"binary"), ("target_model", b"target"),
                              ("draft_model", b"draft"), ("runner", b"runner"),
                              ("questions", b"questions")):
            path = identity / name; path.write_bytes(payload); self.files[name] = path
        self.preflight = {
            "schema": "epyc.df2.followups_preflight.v2",
            "campaign_id": "df2-5-qwen38-concurrency-test",
            "campaign_kind": "experimental_runtime", "authority": B.AUTHORITY,
            "created_at": "2026-08-20T15:00:00+00:00",
            "source_root": "/source",
            "source_commit": "2046c64e9948671c7557428b198acebc6f416575",
            "binary": str(self.files["binary"]), "binary_sha256": sha(self.files["binary"]),
            "target_model": str(self.files["target_model"]),
            "target_model_sha256": sha(self.files["target_model"]),
            "draft_model": str(self.files["draft_model"]),
            "draft_model_sha256": sha(self.files["draft_model"]),
            "questions": str(self.files["questions"]),
            "questions_sha256": sha(self.files["questions"]),
            "runner": str(self.files["runner"]), "runner_sha256": sha(self.files["runner"]),
            "parity_client": "/parity", "parity_client_sha256": "a" * 64,
            "protocol": copy.deepcopy(B.EXPECTED_PROTOCOL),
            "route_authority": {"status": "expected_route_only"},
        }
        write_json(self.root / "preflight.json", self.preflight)
        for arm in B.ARMS:
            self.arm(arm)

    def arm(self, arm: str):
        np = int(arm.rsplit("np", 1)[1]); directory = self.root / arm; directory.mkdir()
        claim = {
            "schema": "epyc.autokernel.device_claim_receipt.v1",
            "claim_id": f"claim-{arm}", "campaign_id": self.preflight["campaign_id"],
            "device_id": "mi210_0", "purpose": f"test {arm}", "host": "host",
            "holder_pid": 11, "holder_start_ticks": 12, "holder_boot_id": "boot",
            "holder_label": arm, "lock_path": "/lock", "acquired_at": "2026-08-20T15:00:00Z",
            "expires_at": "2026-08-20T16:00:00Z", "released_at": None,
            "state": "held", "reclaimed_from": None,
        }
        write_json(directory / "claim-open.json", claim)
        write_json(directory / "claim-released.json",
                   {**claim, "released_at": "2026-08-20T15:10:00Z"})
        write_json(directory / "transport.json", {
            "schema": "epyc.df2.arm_transport.v1", "arm": arm,
            "started_at": "2026-08-20T15:00:00Z", "finished_at": "2026-08-20T15:10:00Z",
            "runner_returncode": 0, "failure": None, "claim_id": claim["claim_id"],
            "claim_released": True, "inference_window_released": True,
            "server_pid": 99, "server_returncode": 0, "runner_pid": 100,
        })
        samples = [{"ts": f"2026-08-20T15:00:0{i}Z", "server_pid": 99,
                    "kfd_pids": [99], "server_kfd_resident": True,
                    "vram_used_bytes": 1000 + i, "vram_delta_bytes": 900 + i,
                    "vram_error": None, "kfd_error": None} for i in (1, 2)]
        write_json(directory / "resource-samples.json", {
            "schema": "epyc.df2.resource_samples.v1", "baseline_vram_bytes": 100,
            "interval_s": 0.25, "samples": samples})
        models = self.preflight["target_model"] if arm.startswith("mtp_") else (
            f"{self.preflight['target_model']};{self.preflight['draft_model']}")
        decode = float(np * (50 if arm.startswith("mtp_") else 55))
        write_json(directory / "result.json", {
            "meta": {"kernel": "candidate", "binary": self.preflight["binary"],
                     "models": models, "arm": arm, "n_per_suite": 12, "seed": 42,
                     "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "enable_thinking": False, "max_tokens": 2048,
                     "runner_source_sha256": self.preflight["runner_sha256"]},
            "suites": [{"suite": "olympiadbench_hard", "n": 12, "errors": 0,
                        "throughput": {"concurrency": np, "aggregate_decode_tok_s": decode,
                                       "aggregate_total_tok_s": decode + 1,
                                       "completion_tokens": 24576, "wall_s": 100.0}}],
        })
        request = {"endpoint": "chat", "request_path": "/v1/chat/completions",
                   "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                   "enable_thinking": False}
        rows = [{"arm": arm, "suite": "olympiadbench_hard", "id": f"q-{i}",
                 "rep": 0, "seed": 42, "request_error": "", "effective_request": request,
                 "runner_source_sha256": self.preflight["runner_sha256"]} for i in range(12)]
        (directory / "pq.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        (directory / "acceptance.txt").write_text(
            "".join(f"draft acceptance = 0.50000 ( {100+i} accepted / {200+i} generated), mean len = 4.00\n"
                    for i in range(12)), encoding="utf-8")
        write_json(directory / "commands.json", {"server": ["server"], "runner": ["runner"]})
        write_json(directory / "summary.json", {
            "schema": "epyc.df2.followup_arm.v2", "arm": arm,
            "aggregate_decode_tok_s": decode, "aggregate_total_tok_s": decode + 1,
            "completion_tokens": 24576, "wall_s": 100.0, "acceptance_lines": 12,
            "resource_sample_count": 2, "kfd_resident_samples": 2,
            "positive_vram_samples": 2, "peak_vram_used_bytes": 1002,
            "claim_id": claim["claim_id"], "claim_released": True})


class DFlash2BeliefTests(unittest.TestCase):
    def test_finalizer_emits_twelve_self_hashed_bounded_claimtuple_carriers(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            receipt = B.finalize_concurrency(
                fixture.root, created_at="2026-08-20T15:15:00+00:00")
            rows = B.native_rows(fixture.root / "campaign-summary.json")
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["measurement"]["metric"] for row in rows},
                             {"decode_tokens_per_s", "draft_acceptance_fraction"})
            self.assertEqual({row["measurement"]["metric_direction"] for row in rows}, {"higher_better"})
            self.assertEqual({row["measurement"]["reps"] for row in rows}, {12})
            self.assertEqual({row["measurement"]["attestation_locator"] for row in rows},
                             {str(fixture.root / "campaign-manifest.json")})
            for row in rows:
                projected = B.project(row)
                self.assertNotIn("measurement_sha256", projected)
                self.assertTrue(projected["attestation_present"])
                self.assertTrue(projected["extra"]["experimental_runtime"])
                self.assertFalse(projected["extra"]["kernel_champion_authority"])
                self.assertFalse(projected["extra"]["promotion_authority"])
                self.assertFalse(projected["extra"]["production_authority"])
            self.assertEqual(receipt["receipt_sha256"], B._receipt_sha(receipt))

    def test_reader_refuses_artifact_or_claimtuple_mutation_even_if_outer_rehashed(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td)); B.finalize_concurrency(fixture.root)
            receipt_path = fixture.root / "campaign-summary.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["belief_measurements"][0]["value"] += 1
            receipt["belief_measurements"][0]["measurement_sha256"] = B._measurement_sha(
                receipt["belief_measurements"][0])
            receipt["receipt_sha256"] = B._receipt_sha(receipt)
            write_json(receipt_path, receipt)
            with self.assertRaisesRegex(B.DFlash2BeliefRefusal, "do not rederive"):
                B.native_rows(receipt_path)

        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td)); B.finalize_concurrency(fixture.root)
            path = fixture.root / "mtp_np2" / "pq.jsonl"
            path.write_text(path.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(B.DFlash2BeliefRefusal, "closure changed"):
                B.native_rows(fixture.root / "campaign-summary.json")

    def test_reader_refuses_missing_residency_or_unreleased_claim(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            samples_path = fixture.root / "mtp_np2" / "resource-samples.json"
            value = json.loads(samples_path.read_text())
            value["samples"][0]["server_kfd_resident"] = False
            value["samples"][1]["server_kfd_resident"] = False
            write_json(samples_path, value)
            with self.assertRaisesRegex(B.DFlash2BeliefRefusal, "KFD"):
                B.finalize_concurrency(fixture.root)

        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            path = fixture.root / "mtp_np2" / "claim-released.json"
            value = json.loads(path.read_text()); value["released_at"] = None; write_json(path, value)
            with self.assertRaisesRegex(B.DFlash2BeliefRefusal, "release chronology"):
                B.finalize_concurrency(fixture.root)

    def test_finalizer_refuses_resource_witness_outside_claimed_process_window(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            path = fixture.root / "mtp_np2" / "resource-samples.json"
            resources = json.loads(path.read_text())
            resources["samples"][0]["ts"] = "2026-08-20T16:00:01Z"
            write_json(path, resources)
            with self.assertRaisesRegex(B.DFlash2BeliefRefusal, "outside the process window"):
                B.finalize_concurrency(fixture.root)

    def test_reader_refuses_manifest_symlink_and_projection_without_source_rederivation(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td)); B.finalize_concurrency(fixture.root)
            manifest = fixture.root / "campaign-manifest.json"
            escaped = Path(td) / "escaped-manifest.json"
            escaped.write_bytes(manifest.read_bytes())
            manifest.unlink(); manifest.symlink_to(escaped)
            with self.assertRaisesRegex(B.DFlash2BeliefRefusal, "real campaign root|single-link"):
                B.native_rows(fixture.root / "campaign-summary.json")

        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td)); B.finalize_concurrency(fixture.root)
            native = B.native_rows(fixture.root / "campaign-summary.json")[0]
            forged = copy.deepcopy(native)
            forged["measurement"]["value"] += 1
            forged["measurement"]["measurement_sha256"] = B._measurement_sha(forged["measurement"])
            with self.assertRaisesRegex(B.DFlash2BeliefRefusal, "exact producer-authored"):
                B.project(forged)

    def test_completed_df2_4_and_all_pre_hook_rows_project_zero(self):
        old = {"schema": "epyc.df2.matched_np1_campaign.v1", "arms": {},
               "headline": {"dflash2_n8_decode_tok_s": 70.0}}
        self.assertEqual(B.native_rows(old), [])
        actual = Path("/workspace/repos/epyc-inference-research/artifacts/architect-bench-gpu-20260814/"
                      "dflash2_np1_20260820/campaign-summary.json")
        if actual.is_file():
            self.assertEqual(hashlib.sha256(actual.read_bytes()).hexdigest(),
                             "e4f9e21fd399c37fceca31e171be0299bcd4c35284d5a5828e3201a8bf50b053")
            self.assertEqual(B.native_rows(actual), [])


if __name__ == "__main__":
    unittest.main()
