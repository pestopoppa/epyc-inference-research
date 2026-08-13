from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.benchmark import run_autokernel_gpu_discovery as gpu


def _build(root: Path, *, rocwmma: str, mfma: str) -> Path:
    build = root / f"build-{rocwmma}-{mfma}"
    bindir = build / "bin"
    bindir.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text(
        f"GGML_HIP_ROCWMMA_FATTN:BOOL={rocwmma}\n"
        f"GGML_HIP_MMQ_MFMA:BOOL={mfma}\n", encoding="utf-8")
    binary = bindir / "llama-bench"
    binary.write_bytes(b"sealed-binary")
    binary.chmod(0o755)
    (bindir / "libggml-hip.so").write_bytes(b"sealed-hip-dso")
    return build


class TestGpuDiscoveryBuildIdentity(unittest.TestCase):
    def test_seals_factor_binary_and_dsos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            identity = gpu.build_identity(
                _build(Path(directory), rocwmma="ON", mfma="OFF"))
        self.assertEqual(identity["source_commit"], gpu.SOURCE_COMMIT)
        self.assertTrue(identity["rocwmma_fattn"])
        self.assertFalse(identity["mmq_mfma"])
        self.assertEqual(len(identity["artifacts"]["binary_sha256"]), 64)
        self.assertEqual(set(identity["artifacts"]["libraries"]), {"libggml-hip.so"})

    def test_refuses_an_unsealed_factor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF")
            (build / "CMakeCache.txt").write_text(
                "GGML_HIP_ROCWMMA_FATTN:BOOL=ON\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "GGML_HIP_MMQ_MFMA"):
                gpu.build_identity(build)

    def test_flash_attention_factor_requires_one_r1m0_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF").resolve()
            identity = gpu.build_identity(build)
            factor = gpu.factor_spec(
                factor="flash_attention", anchor_build=build, candidate_build=build,
                anchor_identity=identity, candidate_identity=identity)
        self.assertEqual(factor["name"], "flash_attention")
        self.assertFalse(factor["anchor_flash_attention"])
        self.assertTrue(factor["candidate_flash_attention"])

    def test_flash_attention_factor_refuses_distinct_builds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="ON", mfma="OFF").resolve()
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF").resolve()
            with self.assertRaisesRegex(RuntimeError, "one identical build path"):
                gpu.factor_spec(
                    factor="flash_attention", anchor_build=anchor,
                    candidate_build=candidate,
                    anchor_identity=gpu.build_identity(anchor),
                    candidate_identity=gpu.build_identity(candidate))

    def test_rocwmma_factor_keeps_mmq_off_and_flash_on(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="OFF", mfma="OFF").resolve()
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF").resolve()
            factor = gpu.factor_spec(
                factor="rocwmma_fattn", anchor_build=anchor,
                candidate_build=candidate,
                anchor_identity=gpu.build_identity(anchor),
                candidate_identity=gpu.build_identity(candidate))
        self.assertEqual(factor["name"], "GGML_HIP_ROCWMMA_FATTN")
        self.assertEqual((factor["anchor"], factor["candidate"]), ("OFF", "ON"))
        self.assertTrue(factor["anchor_flash_attention"])
        self.assertTrue(factor["candidate_flash_attention"])

    def test_rocwmma_factor_refuses_mmq_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="OFF", mfma="ON").resolve()
            candidate = _build(root / "candidate", rocwmma="ON", mfma="ON").resolve()
            with self.assertRaisesRegex(RuntimeError, "MMQ_MFMA=OFF"):
                gpu.factor_spec(
                    factor="rocwmma_fattn", anchor_build=anchor,
                    candidate_build=candidate,
                    anchor_identity=gpu.build_identity(anchor),
                    candidate_identity=gpu.build_identity(candidate))


class TestGpuDiscoveryInferenceWindow(unittest.TestCase):
    def test_small_model_overlap_records_claims_without_cpu_exclusivity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "small.gguf"
            model.write_bytes(b"model")
            claims = {"regions": {"q0": [{"held": True, "role": "autokernel",
                "holder_pids": [123], "attribution": {"campaign_id": "cpu-r4"}}]}}
            with mock.patch.object(gpu.cpu_region_claim, "inspect_region_claims",
                                   return_value=claims), \
                 mock.patch.object(gpu, "_invoke_locked", return_value={"metric": 1.0}), \
                 mock.patch.object(gpu.cpu_region_claim, "acquire_cpu_region_claim") as acquire:
                result = gpu.invoke(
                    build=Path("/build"), model=model, seed=1, baseline_vram=0,
                    flash_attention=True, campaign_id="gpu-s2",
                    cpu_journal=mock.Mock(), allow_small_model_cpu_overlap=True)
        acquire.assert_not_called()
        self.assertIsNone(result["inference_call_window"])
        coverage = result["cpu_coverage"]
        self.assertEqual(coverage["cpu_overlap_policy"], "allowed_discovery_noise")
        self.assertFalse(coverage["cpu_exclusivity"])
        self.assertFalse(coverage["promotion_claim"])
        self.assertEqual(coverage["concurrent_claims"][0]["attribution"]["campaign_id"],
                         "cpu-r4")

    def test_small_model_overlap_refuses_assets_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "large.gguf"
            model.write_bytes(b"x")
            with mock.patch.object(gpu, "SMALL_MODEL_OVERLAP_MAX_BYTES", 0):
                with self.assertRaisesRegex(RuntimeError, "small-model CPU-overlap"):
                    gpu.invoke(build=Path("/build"), model=model, seed=1,
                               baseline_vram=0, flash_attention=True,
                               campaign_id="gpu-s2", cpu_journal=mock.Mock(),
                               allow_small_model_cpu_overlap=True)

    def test_owned_cpu_coverage_exists_only_inside_model_call(self) -> None:
        events = []

        class Lease:
            path = Path("/tmp/window")
            waited_s = 0.125

        class Held:
            def __enter__(self):
                events.append("window-open")
                return Lease()

            def __exit__(self, *_):
                events.append("window-close")

        class CallWindow:
            def hold(self):
                return Held()

        class Receipt:
            def to_dict(self):
                return {"claim_id": "cpu-owned"}

        class Claim:
            def receipt(self):
                return Receipt()

            def release(self):
                events.append("cpu-release")

        def measured(**_kwargs):
            events.append("model-call")
            return {"metric": 1.0}

        with mock.patch.object(gpu, "MODEL_CALL_WINDOW", CallWindow()), \
             mock.patch.object(gpu.cpu_region_claim, "acquire_cpu_region_claim",
                               return_value=Claim()), \
             mock.patch.object(gpu, "_invoke_locked", side_effect=measured):
            result = gpu.invoke(
                build=Path("/build"), model=Path("/model"), seed=1,
                baseline_vram=0, flash_attention=True,
                campaign_id="ak-gpu-test", cpu_journal=mock.Mock())
        self.assertEqual(events, ["window-open", "model-call", "cpu-release",
                                  "window-close"])
        self.assertFalse(result["cpu_coverage"]["borrowed"])

    def test_contended_cpu_claim_borrows_only_inside_model_window(self) -> None:
        events = []

        class Lease:
            path = Path("/tmp/window")
            waited_s = 0.0

        class Held:
            def __enter__(self):
                events.append("window-open")
                return Lease()

            def __exit__(self, *_):
                events.append("window-close")

        class CallWindow:
            def hold(self):
                return Held()

        class Borrowed:
            borrowed = True

            def validate(self):
                events.append("borrow-validate")

            def to_dict(self):
                return {"borrowed": True, "claim_id": "cpu-live-controls"}

        def acquire(*_args, **_kwargs):
            raise gpu.cpu_region_claim.CpuRegionClaimTimeout("held")

        def borrow(_cpu_list):
            events.append("borrow-open")
            return Borrowed()

        def measured(**_kwargs):
            events.append("model-call")
            return {"metric": 1.0}

        with mock.patch.object(gpu, "MODEL_CALL_WINDOW", CallWindow()), \
             mock.patch.object(gpu.cpu_region_claim, "acquire_cpu_region_claim",
                               side_effect=acquire), \
             mock.patch.object(gpu.inference_window, "borrow_windowed_cpu_coverage",
                               side_effect=borrow), \
             mock.patch.object(gpu, "_invoke_locked", side_effect=measured):
            result = gpu.invoke(
                build=Path("/build"), model=Path("/model"), seed=1,
                baseline_vram=0, flash_attention=True,
                campaign_id="ak-gpu-test", cpu_journal=mock.Mock())
        self.assertEqual(events, ["window-open", "borrow-open", "model-call",
                                  "borrow-validate", "window-close"])
        self.assertTrue(result["cpu_coverage"]["borrowed"])
