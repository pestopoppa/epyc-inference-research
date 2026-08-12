from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from . import device_sampler as S
from . import microbench as M
from ..evaluator import devices


SNAPSHOT = """
GPU[0] : Temperature (Sensor junction) (C): 61.0
GPU[0] : mclk clock level: 3: (1600Mhz)
GPU[0] : sclk clock level: 2: (1700Mhz)
GPU[0] : Average Graphics Package Power (W): 241.0
"""


def good_runner(argv, timeout_s):
    return S.SnapshotResult(tuple(argv), 0, SNAPSHOT, "")


class DeviceSamplingReceiptTest(unittest.TestCase):
    def sample(self, offset):
        return S.TimedDeviceStateSample(
            offset, devices.DeviceStateSample(1700, 1600, 241, 61, True))

    def receipt(self, offsets=(0.0, 0.25, 0.50)):
        return S.DeviceSamplingReceipt(
            sampler_id=S.MODULE_ID, device_id="ROCm0",
            source="rocm-smi/numeric-250ms/v1",
            started_at="2026-08-11T00:00:00Z",
            ended_at="2026-08-11T00:00:01Z", interval_s=0.25,
            duration_s=1.0,
            command=(S.DEFAULT_ROCM_SMI, "-d", "0", "--showclocks",
                     "--showpower", "--showtemp"),
            samples=tuple(self.sample(value) for value in offsets))

    def test_receipt_is_content_addressed_and_builds_evaluator_state(self):
        receipt = self.receipt()
        payload = receipt.to_dict()
        self.assertEqual(payload["sample_count"], 3)
        self.assertEqual(payload["max_gap_s"], 0.25)
        self.assertEqual(len(payload["sha256"]), 64)
        state = receipt.device_state(nominal_sclk_mhz=1700, min_sclk_ratio=0.9)
        self.assertEqual(state.check().outcome, "PASS")
        self.assertEqual(state.receipt_ref, f"sha256:{payload['sha256']}")

    def test_empty_or_gapped_trace_refuses(self):
        with self.assertRaises(S.DeviceSamplingError):
            self.receipt(offsets=())
        with self.assertRaisesRegex(S.DeviceSamplingError, "cadence gap"):
            self.receipt(offsets=(0.0, 0.51))

    def test_live_gpu_sampling_is_verdict_bearing(self):
        self.assertEqual(M.check_gpu_device_sampling(
            self.receipt(), n_gpu_layers=99, live_subprocess=True).outcome, "PASS")
        self.assertEqual(M.check_gpu_device_sampling(
            None, n_gpu_layers=99, live_subprocess=True).outcome,
            "COULD_NOT_CHECK")
        throttled = S.DeviceSamplingReceipt(
            sampler_id=S.MODULE_ID, device_id="ROCm0",
            source="amdgpu-hwmon/numeric-250ms/v1",
            started_at="2026-08-11T00:00:00Z",
            ended_at="2026-08-11T00:00:01Z", interval_s=0.25,
            duration_s=1.0,
            command=(S.DEFAULT_ROCM_SMI, "-d", "0", "--showclocks",
                     "--showpower", "--showtemp"),
            samples=(S.TimedDeviceStateSample(
                0.0, devices.DeviceStateSample(800, 1600, 241, 61, True)),))
        self.assertEqual(M.check_gpu_device_sampling(
            throttled, n_gpu_layers=99, live_subprocess=True).outcome, "FAIL")

    def test_ranked_gpu_duration_floor_comes_from_local_gfx90a_receipt(self):
        floor = devices.GFX90A_RANKED_DURATION_ADMISSION
        self.assertEqual(floor.architecture, "gfx90a")
        self.assertEqual(floor.device_id, "ROCm0")
        self.assertEqual(floor.min_window_ns, 250_090_903)
        self.assertIn(
            "07788e1d488ecec062e8133dd9e11d379e5075afbcc20f80b6da37e345533431",
            floor.evidence_ref)
        self.assertEqual(
            floor.check((floor.min_window_ns,), device_id="ROCm0").outcome,
            "PASS")

    def test_sub_floor_live_gpu_window_is_unrankable(self):
        floor = devices.GFX90A_RANKED_DURATION_ADMISSION.min_window_ns
        check = M.check_gpu_ranked_duration_windows(
            (floor // 4, floor // 4), self.receipt(),
            n_gpu_layers=99, live_subprocess=True)
        self.assertEqual(check.outcome, "COULD_NOT_CHECK")
        self.assertIn("receives no speed rank", check.reasons[0])
        self.assertEqual(M.check_gpu_ranked_duration_windows(
            (floor // 2, floor - floor // 2), self.receipt(),
            n_gpu_layers=99, live_subprocess=True).outcome, "PASS")
        self.assertEqual(M.check_gpu_ranked_duration_windows(
            (floor,), self.receipt(), n_gpu_layers=0,
            live_subprocess=True).outcome, "PASS")
        self.assertEqual(M.check_gpu_ranked_duration_windows(
            (floor,), self.receipt(), n_gpu_layers=99,
            live_subprocess=False).outcome, "PASS")

    def test_gpu_duration_floor_refuses_foreign_or_missing_device_evidence(self):
        floor = devices.GFX90A_RANKED_DURATION_ADMISSION.min_window_ns
        foreign = S.DeviceSamplingReceipt(
            sampler_id=S.MODULE_ID, device_id="ROCm1",
            source="amdgpu-hwmon/numeric-250ms/v1",
            started_at="2026-08-11T00:00:00Z",
            ended_at="2026-08-11T00:00:01Z", interval_s=0.25,
            duration_s=1.0,
            command=(S.DEFAULT_ROCM_SMI, "-d", "1", "--showclocks",
                     "--showpower", "--showtemp"),
            samples=(self.sample(0.0), self.sample(0.25)))
        self.assertEqual(M.check_gpu_ranked_duration_windows(
            (floor,), foreign, n_gpu_layers=99,
            live_subprocess=True).outcome, "COULD_NOT_CHECK")
        self.assertEqual(M.check_gpu_ranked_duration_windows(
            (floor,), None, n_gpu_layers=99,
            live_subprocess=True).outcome, "COULD_NOT_CHECK")


class RocmSmiSamplingSessionTest(unittest.TestCase):
    def test_live_thread_samples_numeric_fields_on_declared_cadence(self):
        sampler = S.RocmSmiSampler(
            interval_s=0.01, command_timeout_s=0.1, runner=good_runner)
        session = sampler.start()
        time.sleep(0.035)
        receipt = session.stop()
        self.assertGreaterEqual(len(receipt.samples), 2)
        self.assertLess(receipt.duration_s, 0.1)
        self.assertTrue(all(row.sample.under_measurement_load for row in receipt.samples))
        self.assertTrue(all(row.sample.sclk_mhz == 1700 for row in receipt.samples))

    def test_parse_or_process_failure_is_fail_closed(self):
        def bad_output(argv, timeout_s):
            return S.SnapshotResult(tuple(argv), 0, "missing numeric fields", "")

        session = S.RocmSmiSampler(
            interval_s=0.01, command_timeout_s=0.1, runner=bad_output).start()
        time.sleep(0.01)
        with self.assertRaisesRegex(S.DeviceSamplingError, "failed closed"):
            session.stop()

        def bad_exit(argv, timeout_s):
            return S.SnapshotResult(tuple(argv), 2, "", "permission denied")

        session = S.RocmSmiSampler(
            interval_s=0.01, command_timeout_s=0.1, runner=bad_exit).start()
        time.sleep(0.01)
        with self.assertRaisesRegex(S.DeviceSamplingError, "exited 2"):
            session.stop()

    def test_subprocess_spawner_brackets_exact_process_lifetime(self):
        sampler = S.RocmSmiSampler(
            interval_s=0.01, command_timeout_s=0.1, runner=good_runner)
        result = M.SubprocessSpawner(device_sampler=sampler).run(
            ("/usr/bin/python3", "-c", "import time; time.sleep(0.04)"),
            {}, timeout_s=2.0)
        self.assertEqual(result.returncode, 0)
        self.assertIsNotNone(result.device_sampling_receipt)
        payload = result.to_dict()["device_sampling_receipt"]
        self.assertGreaterEqual(payload["sample_count"], 2)
        self.assertLessEqual(payload["max_gap_s"], 0.02)


class AmdgpuHwmonSnapshotRunnerTest(unittest.TestCase):
    def test_maps_logical_device_index_to_numeric_hwmon_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hwmon = root / "card2" / "device" / "hwmon" / "hwmon6"
            hwmon.mkdir(parents=True)
            fields = {
                "name": "amdgpu\n",
                "freq1_input": "1700000000\n",
                "freq2_input": "1600000000\n",
                "power1_average": "241000000\n",
                "temp2_input": "61000\n",
            }
            for name, value in fields.items():
                (hwmon / name).write_text(value, encoding="utf-8")

            runner = S.AmdgpuHwmonSnapshotRunner(sysfs_root=directory)
            result = runner((S.DEFAULT_ROCM_SMI, "-d", "0"), 0.1)
            sample = devices.parse_rocm_smi_snapshot(
                clocks_text=result.stdout, power_text=result.stdout,
                temperature_text=result.stdout, under_measurement_load=True)
            self.assertEqual(sample.sclk_mhz, 1700.0)
            self.assertEqual(sample.mclk_mhz, 1600.0)
            self.assertEqual(sample.power_w, 241.0)
            self.assertEqual(sample.temperature_c, 61.0)

    def test_missing_numeric_field_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            hwmon = Path(directory) / "card0" / "device" / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            (hwmon / "name").write_text("amdgpu\n", encoding="utf-8")
            runner = S.AmdgpuHwmonSnapshotRunner(sysfs_root=directory)
            with self.assertRaisesRegex(S.DeviceSamplingError, "freq1_input"):
                runner((S.DEFAULT_ROCM_SMI, "-d", "0"), 0.1)


if __name__ == "__main__":
    unittest.main()
