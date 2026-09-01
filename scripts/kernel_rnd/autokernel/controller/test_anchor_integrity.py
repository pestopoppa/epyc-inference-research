#!/usr/bin/env python3
"""Code-section digests, proven against the REAL R21-10 deterministic build pair.

The staged fixture at `/mnt/raid0/llm/tmp/r2110-build-{a,b}` is two independent
builds of one commit into two directories — the probe that root-caused R21-10.
Their `libggml-hip.so` files differ by exactly ONE byte, in `.dynstr` (the
RUNPATH encodes each build directory's own path), while every code section is
bit-identical. That pair is the ground truth this module's contract is stated
against: the digest must call them IDENTICAL, and must call a flipped
`.hip_fatbin` byte DIFFERENT. Nothing here builds anything; the mutated copy
goes to scratch, never near the fixture (read-only) or the live run-22 lane.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest

from autokernel.controller import anchor_integrity

FIXTURE_A = Path("/mnt/raid0/llm/tmp/r2110-build-a")
FIXTURE_B = Path("/mnt/raid0/llm/tmp/r2110-build-b")
HAVE_PAIR = ((FIXTURE_A / anchor_integrity.LIBRARY).is_file()
             and (FIXTURE_B / anchor_integrity.LIBRARY).is_file())

needs_pair = unittest.skipUnless(
    HAVE_PAIR, "staged R21-10 deterministic build pair not present")


def _flip_byte(path: Path, offset: int) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        byte = handle.read(1)
        handle.seek(offset)
        handle.write(bytes([byte[0] ^ 0xFF]))


class _MutatedCopy(unittest.TestCase):
    """A scratch copy of fixture A's library, safe to corrupt."""

    def setUp(self):
        if not HAVE_PAIR:
            self.skipTest("staged R21-10 deterministic build pair not present")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.copy = Path(tmp.name) / "libggml-hip.so"
        # resolve(): the fixture's bin/libggml-hip.so is a symlink chain.
        shutil.copyfile((FIXTURE_A / anchor_integrity.LIBRARY).resolve(),
                        self.copy)
        self.spans = {name: (off, size) for name, off, size
                      in anchor_integrity.section_spans(self.copy)}


class TheRealPairIsHashIdentical(unittest.TestCase):
    """The primary claim: RUNPATH-only differences do not enter the digest."""

    @needs_pair
    def test_the_two_builds_digest_identically(self):
        a = anchor_integrity.build_digest(FIXTURE_A)
        b = anchor_integrity.build_digest(FIXTURE_B)
        self.assertIsNotNone(a)
        # BROKEN READS: a != b -- an exclude-list that missed `.dynstr` (or a
        # whole-file hash) flags every healthy promotion as run 18's fault class.
        self.assertEqual(a, b)

    @needs_pair
    def test_the_pair_really_does_differ_on_disk(self):
        """The identical-digest claim above is vacuous if the inputs are the same
        bytes. They are not: the RUNPATH byte differs, and this pins it."""
        a_bytes = (FIXTURE_A / anchor_integrity.LIBRARY).resolve().read_bytes()
        b_bytes = (FIXTURE_B / anchor_integrity.LIBRARY).resolve().read_bytes()
        self.assertNotEqual(a_bytes, b_bytes,
                            "fixture pair is byte-identical; it no longer proves "
                            "the RUNPATH exclusion does anything")

    @needs_pair
    def test_the_code_sections_are_all_present(self):
        names = {name for name, _off, _size
                 in anchor_integrity.section_spans(FIXTURE_A / anchor_integrity.LIBRARY)}
        # BROKEN READS: a missing `.hip_fatbin` here means the digest no longer
        # covers the device kernels -- the very bytes run 18's mismatch lived in.
        for wanted in anchor_integrity.CODE_SECTIONS:
            self.assertIn(wanted, names)


class AFlippedCodeByteChangesTheDigest(_MutatedCopy):
    """The other direction: the digest must SEE the sections it claims to cover."""

    def _differs_after_flip(self, section: str) -> bool:
        baseline = anchor_integrity.code_digest(self.copy)
        off, size = self.spans[section]
        _flip_byte(self.copy, off + size // 2)
        return anchor_integrity.code_digest(self.copy) != baseline

    def test_a_hip_fatbin_flip_is_detected(self):
        # BROKEN READS: False -- a digest that skips `.hip_fatbin` would certify
        # run 18's wrong-kernel anchor as the champion, deterministically.
        self.assertTrue(self._differs_after_flip(".hip_fatbin"))

    def test_a_text_flip_is_detected(self):
        self.assertTrue(self._differs_after_flip(".text"))

    def test_a_rodata_flip_is_detected(self):
        self.assertTrue(self._differs_after_flip(".rodata"))

    def test_a_dynstr_flip_is_NOT_detected(self):
        """`.dynstr` is where the R21-10 pair legitimately differs (RUNPATH), so a
        digest that reads it calls every healthy double-build a mismatch."""
        self.assertFalse(self._differs_after_flip(".dynstr"))


class UnhashableInputsAreNoneNeverAValue(unittest.TestCase):
    """None means "fall back to the A/A alone". A digest of garbage would instead
    hash-prove garbage -- or abort on it -- so every failure path must be None."""

    def test_a_missing_build_dir_is_none(self):
        self.assertIsNone(anchor_integrity.build_digest("/nonexistent-build"))

    def test_a_non_elf_file_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / anchor_integrity.LIBRARY
            lib.parent.mkdir(parents=True)
            lib.write_text("not an elf", encoding="utf-8")
            self.assertIsNone(anchor_integrity.build_digest(tmp))

    @needs_pair
    def test_a_truncated_library_is_none(self):
        """Truncation mid-section is a partial copy or a dying disk; hashing the
        bytes that happen to exist would give it a stable-looking identity."""
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / anchor_integrity.LIBRARY
            lib.parent.mkdir(parents=True)
            source = (FIXTURE_A / anchor_integrity.LIBRARY).resolve()
            spans = dict((name, (off, size)) for name, off, size
                         in anchor_integrity.section_spans(source))
            off, size = spans[".hip_fatbin"]
            with source.open("rb") as src, lib.open("wb") as dst:
                dst.write(src.read(off + size // 2))
            digest = anchor_integrity.code_digest(lib)
            # readelf may refuse the truncated ELF outright (None from
            # section_spans) or parse headers whose sections then run off the end
            # (None from the read loop); either way the answer is None.
            self.assertIsNone(digest)


class TheDigestIsStableAcrossReads(unittest.TestCase):
    @needs_pair
    def test_two_reads_agree(self):
        first = anchor_integrity.build_digest(FIXTURE_A)
        self.assertEqual(first, anchor_integrity.build_digest(FIXTURE_A))
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
