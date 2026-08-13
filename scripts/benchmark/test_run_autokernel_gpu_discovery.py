from pathlib import Path
import tempfile
import unittest

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
