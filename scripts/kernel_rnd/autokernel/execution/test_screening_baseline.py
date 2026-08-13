import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from . import screening_baseline as bank


class ScreeningBaselineBankTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.anchor_root = root / "anchor"
        self.candidate_root = root / "candidate"
        for arm_root in (self.anchor_root, self.candidate_root):
            (arm_root / "lib").mkdir(parents=True)
            (arm_root / "llama-bench").write_bytes(b"same executable")
            (arm_root / "lib" / "libggml.so").write_bytes(b"same DSO")

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def frame():
        return {"recipe": "decode", "model": "m", "instrument": "i",
                "anchor_ggml_iqk": "0"}

    def anchor_command(self):
        return {"arm": "anchor", "argv": ["llama-bench"],
                "env": {"GGML_IQK": "0", "OMP_NUM_THREADS": "24",
                        "LD_LIBRARY_PATH": str(self.anchor_root / "lib")},
                "params": {"ggml_iqk": "0", "threads": 24},
                "binding": {"binary": str(self.anchor_root / "llama-bench"),
                            "library_path": str(self.anchor_root / "lib")}}

    def candidate_command(self):
        return {"arm": "candidate", "argv": ["llama-bench"],
                "env": {"GGML_IQK": "1", "OMP_NUM_THREADS": "24",
                        "LD_LIBRARY_PATH": str(self.candidate_root / "lib")},
                "params": {"ggml_iqk": "1", "threads": 24},
                "binding": {"binary": str(self.candidate_root / "llama-bench"),
                            "library_path": str(self.candidate_root / "lib")}}

    def make_bank(self, samples=(100.0, 101.0, 99.0), *, closed=False):
        return bank.BaselineBank(self.frame(), samples, samples[-1],
                                 self.anchor_command(),
                                 bank.command_artifacts(self.anchor_command()),
                                 samples[-1] if closed else None)

    def test_one_bank_admits_many_candidate_screens_without_new_anchors(self):
        value = self.make_bank()
        for _ in range(8):
            value.admit(self.frame())
        self.assertEqual(len(value.anchor_samples), 3)

    def test_frame_drift_and_closed_sentinel_refuse(self):
        value = self.make_bank()
        with self.assertRaisesRegex(bank.BaselineBankError, "frame differs"):
            value.admit({"recipe": "prefill"})
        closed = self.make_bank(closed=True)
        with self.assertRaisesRegex(bank.BaselineBankError, "closed"):
            closed.admit(self.frame())

    def test_hash_tamper_refuses(self):
        value = self.make_bank().to_dict()
        value["anchor_samples"] = [9.0, 9.0, 9.0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(bank.BaselineBankError, "hash"):
                bank.load(path)

    def test_nomination_reports_uncertainty_not_a_keep(self):
        value = self.make_bank((100.0, 102.0, 101.0))
        report = value.nominate((104.0, 105.0, 106.0))
        self.assertEqual(report["nomination"], "top_k_candidate_only_not_a_keep")
        self.assertIn("nonpromotable", report["uncertainty"])

    def test_bank_is_o1_anchors_and_each_screen_is_three_candidates_zero_anchors(self):
        calls = {"anchor": 0, "candidate": 0}
        def anchor(): calls["anchor"] += 1; return 100.0
        value = bank.create(frame=self.frame(), anchor_command=self.anchor_command(),
                            invoke_anchor=anchor)
        for _ in range(7):
            report = bank.screen(frame=self.frame(), bank=value,
                                 invoke_candidate=lambda: calls.__setitem__("candidate", calls["candidate"] + 1) or 101.0,
                                 competing_inference=False,
                                 candidate_command=self.candidate_command())
            self.assertEqual((report["candidate_invocations"], report["anchor_invocations"]), (3, 0))
        self.assertEqual(calls, {"anchor": 3, "candidate": 21})

    def test_only_competing_inference_blocks_screen(self):
        value = self.make_bank()
        with self.assertRaisesRegex(bank.BaselineBankError, "competing model inference"):
            bank.screen(bank=value, frame=self.frame(), invoke_candidate=lambda: 101.,
                        competing_inference=True,
                        candidate_command=self.candidate_command())

    def test_same_value_candidate_and_extra_factor_refuse_before_sampling(self):
        value = self.make_bank()
        calls = {"candidate": 0}

        def candidate():
            calls["candidate"] += 1
            return 101.0

        same_value = self.candidate_command()
        same_value["env"]["GGML_IQK"] = "0"
        same_value["params"]["ggml_iqk"] = "0"
        with self.assertRaisesRegex(bank.BaselineBankError, "candidate GGML_IQK=1"):
            bank.screen(bank=value, frame=self.frame(), invoke_candidate=candidate,
                        competing_inference=False, candidate_command=same_value)

        extra_factor = self.candidate_command()
        extra_factor["env"]["OMP_NUM_THREADS"] = "12"
        with self.assertRaisesRegex(bank.BaselineBankError, "sole intended"):
            bank.screen(bank=value, frame=self.frame(), invoke_candidate=candidate,
                        competing_inference=False, candidate_command=extra_factor)
        self.assertEqual(calls["candidate"], 0)

    def test_bank_requires_exactly_three_anchors(self):
        with self.assertRaisesRegex(bank.BaselineBankError, "exactly three"):
            bank.create(frame=self.frame(), anchor_command=self.anchor_command(),
                        invoke_anchor=lambda: 100.0, anchor_count=2)

    def test_witness_only_counts_inference_and_unreadable_refuses(self):
        class Scan:
            unreadable_pids = {}
            @staticmethod
            def inference_like(): return ()
        with mock.patch.object(bank.preflight, "read_own_scope", return_value=object()), \
             mock.patch.object(bank.preflight, "interim_process_scan", return_value=Scan()):
            self.assertFalse(bank.competing_inference_witness()["competing"])
        broken = Scan(); broken.unreadable_pids = {42: "EACCES"}
        with mock.patch.object(bank.preflight, "read_own_scope", return_value=object()), \
             mock.patch.object(bank.preflight, "interim_process_scan", return_value=broken), \
             self.assertRaisesRegex(bank.BaselineBankError, "unreadable"):
            bank.competing_inference_witness()

    def _governed_gpu_finding(self, *, mutations=None, child_ngl="99"):
        root = Path(self._tmp.name) / "gpu-screen"
        root.mkdir(exist_ok=True)
        model = root / "small.gguf"
        model.write_bytes(b"small model")
        sealed = {
            "schema": "epyc.autokernel.gpu_discovery_preflight.v1",
            "campaign_id": "gpu-screen-1",
            "cpu_overlap_policy": "allowed_discovery_noise",
            "model": str(model),
            "model_sha256": "model-sha",
            "model_size_bytes": model.stat().st_size,
            "small_model_overlap_max_bytes": 512 * 1024 * 1024,
            "promotion_claim": False,
            "inference_executed": False,
        }
        receipt = {
            "schema": "epyc.autokernel.device_claim_receipt.v1",
            "claim_id": "akd-test", "device_id": "mi210_0", "state": "held",
            "holder_pid": 200, "holder_start_ticks": 222,
            "campaign_id": "gpu-screen-1",
            "purpose": "AutoKernel GPU candidate-only discovery test OFF->ON",
        }
        live = {
            "schema": bank.GPU_DISCOVERY_LIVE_SCHEMA, "status": "active",
            "campaign_id": "gpu-screen-1", "runner_pid": 200,
            "authority": "nonpromotable_candidate_only_discovery",
            "cpu_overlap_policy": "allowed_discovery_noise",
            "model": str(model), "model_sha256": "model-sha",
            "model_size_bytes": model.stat().st_size,
            "small_model_overlap_max_bytes": 512 * 1024 * 1024,
            "promotion_claim": False, "non_promotable": True,
            "preflight_sha256": bank.schemas.content_hash(sealed),
            "device_claim_open": receipt,
        }
        for target, key, value in mutations or ():
            {"live": live, "sealed": sealed, "receipt": receipt}[target][key] = value
        (root / "preflight.json").write_text(json.dumps(sealed), encoding="utf-8")
        (root / "live-governance.json").write_text(json.dumps(live), encoding="utf-8")
        finding = SimpleNamespace(
            pid=100, starttime_ticks=111,
            cmdline=("/build/llama-bench", "-m", str(model), "-ngl", child_ngl),
            to_dict=lambda: {"pid": 100, "starttime_ticks": 111},
        )
        identities = {
            100: {"pid": 100, "ppid": 200, "starttime_ticks": 111},
            200: {"pid": 200, "starttime_ticks": 222,
                  "cmdline": ["python3", "/repo/scripts/benchmark/"
                              "run_autokernel_gpu_discovery.py",
                              "--output-dir", str(root)]},
        }
        return finding, identities

    def test_live_receipt_bound_small_mi210_discovery_is_allowed_noise(self):
        finding, identities = self._governed_gpu_finding()
        passed = bank.schemas.Check(bank.schemas.PASS)
        with mock.patch.object(bank.preflight, "describe_pid",
                               side_effect=lambda pid: identities[pid]), \
             mock.patch.object(bank.device_claim, "check_device_claim_held",
                               return_value=passed):
            admission, reason = bank._gpu_discovery_noise_admission(finding)
        self.assertEqual(reason, "")
        self.assertEqual(admission["classification"], "allowed_discovery_noise")
        self.assertFalse(admission["promotion_claim"])

    def test_gpu_overlap_admission_fails_closed_on_adversarial_variants(self):
        variants = (
            ("released", [("live", "status", "released")], "99"),
            ("promotion", [("live", "promotion_claim", True)], "99"),
            ("large", [("live", "model_size_bytes", 512 * 1024 * 1024 + 1),
                       ("sealed", "model_size_bytes", 512 * 1024 * 1024 + 1)], "99"),
            ("wrong_device", [("receipt", "device_id", "other")], "99"),
            ("cpu_inference", [], "0"),
            ("tampered_preflight", [("sealed", "model_sha256", "tampered")], "99"),
            ("wrong_runner", [("live", "runner_pid", 201)], "99"),
        )
        passed = bank.schemas.Check(bank.schemas.PASS)
        for name, mutations, ngl in variants:
            with self.subTest(name=name):
                finding, identities = self._governed_gpu_finding(
                    mutations=mutations, child_ngl=ngl)
                with mock.patch.object(bank.preflight, "describe_pid",
                                       side_effect=lambda pid: identities[pid]), \
                     mock.patch.object(bank.device_claim, "check_device_claim_held",
                                       return_value=passed):
                    admission, reason = bank._gpu_discovery_noise_admission(finding)
                self.assertIsNone(admission)
                self.assertTrue(reason)

    def test_gpu_overlap_admission_requires_live_device_claim(self):
        finding, identities = self._governed_gpu_finding()
        failed = bank.schemas.Check(bank.schemas.FAIL, ("claim released",))
        with mock.patch.object(bank.preflight, "describe_pid",
                               side_effect=lambda pid: identities[pid]), \
             mock.patch.object(bank.device_claim, "check_device_claim_held",
                               return_value=failed):
            admission, reason = bank._gpu_discovery_noise_admission(finding)
        self.assertIsNone(admission)
        self.assertIn("not actively held", reason)


if __name__ == "__main__":
    unittest.main()
