from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .discovery_static_registry import SharedRewardRuntime, StaticRegistryError


def _bin(root: Path, *, hip: bytes) -> Path:
    bindir = root / "bin"; bindir.mkdir(parents=True)
    for name in ("llama-bench", "libllama-bench-impl.so", "libllama-common.so",
                 "libllama.so", "libggml.so", "libggml-cpu.so", "libggml-base.so"):
        (bindir / name).write_bytes(name.encode())
    (bindir / "llama-bench").chmod(0o755)
    versioned = bindir / "libggml-hip.so.0.16.0"; versioned.write_bytes(hip)
    (bindir / "libggml-hip.so.0").symlink_to(versioned.name)
    (bindir / "libggml-hip.so").symlink_to("libggml-hip.so.0")
    return root


class SharedRewardRuntimeTests(unittest.TestCase):
    def test_complete_common_reward_and_arm_only_hip_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = _bin(root / "anchor", hip=b"anchor")
            candidate = _bin(root / "candidate", hip=b"candidate")
            runtime = SharedRewardRuntime.materialize(root=root / "runtime", anchor_build=anchor,
                                                       candidate_build=candidate)
            self.assertTrue(runtime.measurement_binary.is_file())
            self.assertEqual((runtime.anchor_loader_dir / "libggml-hip.so.0").resolve().read_bytes(), b"anchor")
            self.assertEqual((runtime.candidate_loader_dir / "libggml-hip.so.0").resolve().read_bytes(), b"candidate")
            self.assertFalse((runtime.anchor_loader_dir / "libllama.so").exists())

    def test_refuses_missing_soname_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = _bin(root / "anchor", hip=b"anchor")
            candidate = _bin(root / "candidate", hip=b"candidate")
            (candidate / "bin" / "libggml-hip.so.0").unlink()
            with self.assertRaises(StaticRegistryError):
                SharedRewardRuntime.materialize(root=root / "runtime", anchor_build=anchor,
                                                candidate_build=candidate)

