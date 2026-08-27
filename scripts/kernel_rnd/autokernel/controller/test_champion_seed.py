"""Champion₀ seeding: the aggregate exists and currently equals production.

Hermetic. Builds fake ELF-less "binaries" and a fake `ldd` on PATH so the
measurement path is exercised without touching the frozen production tree.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
import unittest
import unittest.mock

from . import champion as C
from . import champion_seed as CS
from .. import journal


def _fake_tree(root: Path, *, cpu_bytes=b"cpu-binary", gpu_bytes=b"gpu-binary") -> Path:
    for sub, payload in (("build", cpu_bytes), ("build-hip", gpu_bytes)):
        binroot = root / sub / "bin"
        binroot.mkdir(parents=True)
        (binroot / "llama-server").write_bytes(payload)
        (binroot / "libggml.so.0").write_bytes(b"ggml")
    return root


def _fake_ldd(bindir: Path, *, lines: str) -> None:
    """Put a deterministic `ldd` first on PATH."""
    script = bindir / "ldd"
    script.write_text("#!/bin/bash\ncat <<'EOF'\n" + lines + "\nEOF\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


class SeedChampionTests(unittest.TestCase):

    def _with_fake_ldd(self, tmp: Path, lines: str):
        fake = tmp / "fakebin"
        fake.mkdir(exist_ok=True)
        _fake_ldd(fake, lines=lines)
        return {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}

    def test_anchor_is_measured_and_covers_exactly_the_declared_backends(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t); _fake_tree(tmp / "tree")
            env = self._with_fake_ldd(tmp, "\tlibggml.so.0 => /x/libggml.so.0 (0x00007f00)")
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                anchor = CS.production_anchor(tmp / "tree", branch="production-consolidated-v9",
                                              commit="0" * 40)
        self.assertEqual({a.backend for a in anchor.artifacts}, {"llama_cpu", "llama_gpu"})
        self.assertTrue(anchor.sealed)
        # different binaries must yield different identities
        digests = {a.binary_sha256 for a in anchor.artifacts}
        self.assertEqual(len(digests), 2, "each backend must carry its own measured digest")
        for a in anchor.artifacts:
            self.assertRegex(a.binary_sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(a.linkage_sha256, r"^[0-9a-f]{64}$")

    def test_unratified_build_is_refused(self):
        """Seeding off a tree that is not the ratified production build would
        silently re-anchor every future comparison."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t); _fake_tree(tmp / "tree")
            env = self._with_fake_ldd(tmp, "\tlibggml.so.0 => /x/libggml.so.0 (0x00007f00)")
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(CS.AnchorMeasurementError, "refusing to seed"):
                    CS.production_anchor(tmp / "tree", branch="production-consolidated-v9",
                                         commit="0" * 40,
                                         expected_binary_sha256={"llama_cpu": "0" * 64,
                                                                 "llama_gpu": "1" * 64})

    def test_unresolved_library_is_refused_not_hashed_as_absent(self):
        """'cannot resolve' and 'resolves elsewhere' must not share an identity."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t); _fake_tree(tmp / "tree")
            env = self._with_fake_ldd(tmp, "\tlibmissing.so.0 => not found")
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(CS.AnchorMeasurementError, "unresolved shared object"):
                    CS.production_anchor(tmp / "tree", branch="production-consolidated-v9",
                                         commit="0" * 40)

    def test_linkage_identity_changes_when_resolution_changes(self):
        """Identical binaries resolving different libraries are NOT the same anchor."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t); _fake_tree(tmp / "tree")
            got = []
            for lines in ("\tlibggml.so.0 => /a/libggml.so.0 (0x1)",
                          "\tlibggml.so.0 => /b/libggml.so.0 (0x1)"):
                env = self._with_fake_ldd(tmp, lines)
                with unittest.mock.patch.dict(os.environ, env, clear=True):
                    a = CS.production_anchor(tmp / "tree", branch="production-consolidated-v9",
                                             commit="0" * 40)
                got.append(a.artifacts[0].linkage_sha256)
        self.assertNotEqual(got[0], got[1],
                            "linkage identity must track WHAT THE LOADER BOUND, not the binary")

    def test_seed_is_an_existing_aggregate_not_a_blocked_one(self):
        """Champion₀ differs from record_no_champion: it exists, and nothing blocks it."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t); _fake_tree(tmp / "tree")
            env = self._with_fake_ldd(tmp, "\tlibggml.so.0 => /x/libggml.so.0 (0x1)")
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                anchor = CS.production_anchor(tmp / "tree", branch="production-consolidated-v9",
                                              commit="0" * 40)
            book = journal.Journal(str(Path(t) / "journal"))
            book.initialize()
            entry = C.seed_champion(book, anchor, reason="Champion0 = frozen production")
            blocked = C.record_no_champion(book, anchor, reason="nothing composed yet")

        seeded = entry.payload
        self.assertEqual(seeded["status"], "seeded_from_anchor")
        self.assertEqual(seeded["member_candidates"], [])
        self.assertIsNone(seeded["combined_candidate_id"])
        self.assertEqual(seeded["blocking_conditions"], [],
                         "a seed is empty, not blocked — that distinction is the requirement")
        self.assertEqual(seeded["branch"], C.champion_branch("llama.cpp", "0" * 40))

        other = blocked.payload
        self.assertEqual(other["blocking_conditions"], ["NO_GREEN_COMPOSITION"],
                         "record_no_champion must remain blocked — the seed does not replace it")


if __name__ == "__main__":
    unittest.main()
