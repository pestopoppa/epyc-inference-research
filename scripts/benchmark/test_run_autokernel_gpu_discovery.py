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
