"""The hardened inbox reader (R22-6): the operator's injection channel must never
be a kill switch.

THE DEFECT. `run.py`'s inline reader did a bare per-file `read_text(encoding=
"utf-8")`. One invalid-UTF-8 or unreadable file in the live inbox raised inside
`build_context()` on EVERY iteration, so every lane errored, the pool breaker
tripped, and the run died. The historical-scenario class below reconstructs
exactly that directory shape and asserts the run-facing contract: context built,
good seed present, one note per unreadable file, nothing raised.

Mutation notes (both directions):
  * reverting the helper to a bare `read_text` -> `TheHistoricalScenario` raises
    (UnicodeDecodeError / OSError) instead of passing;
  * dropping the `note(...)` call while keeping the skip -> the note-count
    assertions fail (skipped-file-not-noted);
  * catching too broadly (bare `except Exception`) -> `test_it_catches_exactly`
    fails on the source inspection;
  * the seam tests in `loop/test_promotion_targets.py` catch `run.py` quietly
    reverting to an inline reader.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest

from autokernel.controller import inbox


def _unreadable_by_permissions_or_link(path: Path) -> None:
    """A file whose read raises OSError, robust to WHO runs the suite.

    chmod 000 is the historical shape, but root (some CI containers) reads
    through it; a dangling symlink raises FileNotFoundError for every uid, so it
    stands in only when permissions cannot bite.
    """
    if os.geteuid() == 0:
        path.symlink_to(path.with_name("target-that-does-not-exist"))
    else:
        path.write_text("secret", encoding="utf-8")
        path.chmod(0)


class TheHistoricalScenario(unittest.TestCase):
    """One good seed + one invalid-UTF-8 file + one chmod-000 file, together."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "inbox"
        self.dir.mkdir()
        (self.dir / "good-seed.md").write_text("  USE THE MEASURED MFMA LEVER  \n",
                                               encoding="utf-8")
        # Invalid UTF-8: raises UnicodeDecodeError -- a ValueError, which the bare
        # reader did not catch and an `except OSError` alone would not either.
        (self.dir / "bad-utf8.md").write_bytes(b"\xff\xfe\xfa seed bytes")
        _unreadable_by_permissions_or_link(self.dir / "unreadable.md")
        self.addCleanup(lambda: (self.dir / "unreadable.md").chmod(0o600)
                        if not (self.dir / "unreadable.md").is_symlink() else None)

    def test_context_builds_good_seed_present_two_notes_nothing_raised(self):
        notes: list[str] = []
        texts = inbox.read_inbox(self.dir, note=notes.append)     # must not raise
        self.assertEqual(texts, ["USE THE MEASURED MFMA LEVER"])
        self.assertEqual(len(notes), 2, notes)
        self.assertTrue(all("inbox_file_unreadable" in note for note in notes), notes)
        # Each note NAMES its file, or the operator cannot act on it.
        self.assertTrue(any("bad-utf8.md" in note for note in notes), notes)
        self.assertTrue(any("unreadable.md" in note for note in notes), notes)

    def test_the_default_note_surface_is_the_run_log(self):
        """No `note` wired -> the note lands on stdout, which the run log captures.
        BROKEN READS: silence — a skip nobody can see is a file that never existed."""
        out = io.StringIO()
        with redirect_stdout(out):
            texts = inbox.read_inbox(self.dir)
        self.assertEqual(texts, ["USE THE MEASURED MFMA LEVER"])
        self.assertEqual(out.getvalue().count("inbox_file_unreadable"), 2,
                         out.getvalue())


class TheReaderContract(unittest.TestCase):

    def _dir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_a_missing_inbox_is_an_empty_context_not_an_error(self):
        self.assertEqual(inbox.read_inbox(self._dir() / "never-created"), [])

    def test_seeds_come_back_sorted_and_stripped(self):
        root = self._dir()
        (root / "b.md").write_text("second\n", encoding="utf-8")
        (root / "a.md").write_text("  first  ", encoding="utf-8")
        (root / "notes.txt").write_text("not a seed", encoding="utf-8")
        self.assertEqual(inbox.read_inbox(root), ["first", "second"])

    def test_a_healthy_inbox_emits_no_notes(self):
        """Non-vacuity for the note assertions: the counter counts real skips, not
        every file touched."""
        root = self._dir()
        (root / "a.md").write_text("fine", encoding="utf-8")
        notes: list[str] = []
        self.assertEqual(inbox.read_inbox(root, note=notes.append), ["fine"])
        self.assertEqual(notes, [])

    def test_it_catches_exactly_oserror_and_unicodedecodeerror(self):
        """Too narrow re-arms the kill switch; too broad swallows MemoryError and
        KeyboardInterrupt. Pin the tuple in the one except clause."""
        import inspect
        body = inspect.getsource(inbox.read_inbox).split('"""', 2)[-1]
        self.assertIn("except (OSError, UnicodeDecodeError)", body)
        self.assertNotIn("except Exception", body)
        self.assertNotIn("except:", body)

    def test_every_readable_seed_survives_a_bad_neighbour_in_any_position(self):
        """Sorting places the poison first, between, and last; the good seeds must
        come through identically each time (a skip that `break`s instead of
        `continue`s passes the single-poison test)."""
        for poison in ("0-first.md", "m-middle.md", "z-last.md"):
            root = self._dir()
            (root / "b.md").write_text("beta", encoding="utf-8")
            (root / "x.md").write_text("chi", encoding="utf-8")
            (root / poison).write_bytes(b"\xff\xfe")
            notes: list[str] = []
            with self.subTest(poison=poison):
                self.assertEqual(inbox.read_inbox(root, note=notes.append),
                                 ["beta", "chi"])
                self.assertEqual(len(notes), 1, notes)


if __name__ == "__main__":
    unittest.main()
