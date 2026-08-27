from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .gpu_residency_sampler import (
    GpuContentionTimeout, Mi210ResidencySampler, ResidencySamplerError)


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
            self.assertEqual(result.launcher_pid, 7)

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

    def test_owner_subtree_sibling_is_not_foreign(self):
        # A KFD process that is OUR controller's descendant (a sibling leg
        # draining), not a descendant of the sampled child, must NOT crash the
        # proof — the 2026-08-27 self-flagged-KFD bug (pid 964901).
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kfd = root / "kfd"; kfd.mkdir()
            (kfd / "42").mkdir(); (kfd / "55").mkdir()
            proc = root / "proc"
            for pid in (42, 55, 7, 9, 100):
                (proc / str(pid)).mkdir(parents=True)
            # 42 -> 7 (the sampled child); 55 -> 9 -> 100 (controller root), a sibling.
            (proc / "42" / "stat").write_text(_stat(7))
            (proc / "7" / "stat").write_text(_stat(100))
            (proc / "55" / "stat").write_text(_stat(9))
            (proc / "9" / "stat").write_text(_stat(100))
            (proc / "100" / "stat").write_text(_stat(1))
            vram = root / "vram"; vram.write_text("99\n")
            sampler = Mi210ResidencySampler(kfd_root=kfd, vram_path=vram,
                                            proc_root=proc, owner_root_pid=100)
            result = sampler(7)  # no raise
            self.assertEqual(result.kfd_pids, (42,))  # only the child's leg attributed
            self.assertEqual(result.launcher_pid, 7)

    def test_genuinely_foreign_still_refuses_even_with_owner_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kfd = root / "kfd"; kfd.mkdir(); (kfd / "999").mkdir()
            proc = root / "proc"
            for pid in (999, 7, 100):
                (proc / str(pid)).mkdir(parents=True)
            (proc / "999" / "stat").write_text(_stat(1))  # unrelated tree
            (proc / "7" / "stat").write_text(_stat(100))
            (proc / "100" / "stat").write_text(_stat(1))
            vram = root / "vram"; vram.write_text("99\n")
            sampler = Mi210ResidencySampler(kfd_root=kfd, vram_path=vram,
                                            proc_root=proc, owner_root_pid=100)
            with self.assertRaises(ResidencySamplerError):
                sampler(7)

    def test_wait_until_clear_returns_once_foreign_process_drains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kfd = root / "kfd"; kfd.mkdir()
            (kfd / "999").mkdir()  # a foreign holder present initially
            proc = root / "proc"; (proc / "999").mkdir(parents=True)
            (proc / "999" / "stat").write_text(_stat(1))
            vram = root / "vram"; vram.write_text("0\n")
            calls = {"n": 0}

            def fake_sleep(_seconds):
                # After one poll the foreign holder drains.
                calls["n"] += 1
                (kfd / "999").rmdir()

            sampler = Mi210ResidencySampler(
                kfd_root=kfd, vram_path=vram, proc_root=proc,
                owner_root_pid=100, sleep=fake_sleep)
            owned = sampler.wait_until_clear(timeout_s=10.0, poll_s=0.1)
            self.assertEqual(owned, ())
            self.assertEqual(calls["n"], 1)

    def test_wait_until_clear_times_out_on_persistent_foreign(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kfd = root / "kfd"; kfd.mkdir(); (kfd / "999").mkdir()
            proc = root / "proc"; (proc / "999").mkdir(parents=True)
            (proc / "999" / "stat").write_text(_stat(1))
            vram = root / "vram"; vram.write_text("0\n")
            clock = {"t": 0.0}

            def fake_monotonic():
                return clock["t"]

            def fake_sleep(seconds):
                clock["t"] += seconds  # advance time; foreign never drains

            sampler = Mi210ResidencySampler(
                kfd_root=kfd, vram_path=vram, proc_root=proc,
                owner_root_pid=100, sleep=fake_sleep, monotonic=fake_monotonic)
            with self.assertRaises(GpuContentionTimeout):
                sampler.wait_until_clear(timeout_s=1.0, poll_s=0.5)
