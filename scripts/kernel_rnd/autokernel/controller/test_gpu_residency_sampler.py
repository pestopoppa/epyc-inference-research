from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .gpu_residency_sampler import Mi210ResidencySampler, ResidencySamplerError


def _stat(parent: int) -> str:
    return f"1 (runner) S {parent} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"


class Mi210ResidencySamplerTests(unittest.TestCase):
    def test_descendant_kfd_is_attributed_during_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kfd = root / "kfd"; kfd.mkdir(); (kfd / "42").mkdir()
            proc = root / "proc"; (proc / "42").mkdir(parents=True); (proc / "7").mkdir()
            (proc / "42" / "stat").write_text(_stat(7)); (proc / "7" / "stat").write_text(_stat(1))
            vram = root / "vram"; vram.write_text("99\n")
            result = Mi210ResidencySampler(kfd_root=kfd, vram_path=vram, proc_root=proc)(7)
            self.assertEqual(result.kfd_pids, (42,)); self.assertEqual(result.vram_bytes, 99)

    def test_foreign_kfd_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kfd = root / "kfd"; kfd.mkdir(); (kfd / "99").mkdir()
            proc = root / "proc"; (proc / "99").mkdir(parents=True); (proc / "7").mkdir()
            (proc / "99" / "stat").write_text(_stat(1)); (proc / "7" / "stat").write_text(_stat(1))
            vram = root / "vram"; vram.write_text("99\n")
            with self.assertRaises(ResidencySamplerError):
                Mi210ResidencySampler(kfd_root=kfd, vram_path=vram, proc_root=proc)(7)

    def test_empty_kfd_cannot_launder_aggregate_vram_into_residency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kfd = root / "kfd"; kfd.mkdir()
            proc = root / "proc"; (proc / "7").mkdir(parents=True)
            (proc / "7" / "stat").write_text(_stat(1))
            vram = root / "vram"; vram.write_text("999\n")
            result = Mi210ResidencySampler(kfd_root=kfd, vram_path=vram, proc_root=proc)(7)
            self.assertEqual(result.kfd_pids, (7,))
            self.assertEqual(result.vram_bytes, 0)
