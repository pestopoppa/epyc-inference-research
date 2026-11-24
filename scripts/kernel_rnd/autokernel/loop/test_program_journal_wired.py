"""Planning-context canary for program.md's Learnings (the loop's durable journal).

CPU-only, no inference: it stops at context assembly. It proves that what is appended to
`loop/program.md` reaches the planner in BOTH delivery modes and that no size cap cuts it:

* inline: `actors.render_context` puts the whole file in the prompt, and a prompt over
  `ARGV_PROMPT_LIMIT_BYTES` is sent on stdin, never truncated;
* variable: `actor_context` writes it as the `program_strategy` file, lists its headings in
  the jump table, and names the Learnings sections under "Read these before you propose".
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from autokernel.loop import actor_context, actors, loop

PROGRAM = loop.PROGRAM.read_text(encoding="utf-8")
LEARNINGS = [m.group(0) for m in re.finditer(r"(?m)^## Learnings .*$", PROGRAM)]


class ProgramJournalReachesPlanner(unittest.TestCase):
    def test_journal_has_learnings_sections(self):
        self.assertTrue(any("2026-11-21 (evening" in h for h in LEARNINGS), LEARNINGS)

    def test_inline_prompt_carries_program_md_whole(self):
        text = actors.render_context({"program": PROGRAM})
        self.assertIn(PROGRAM.strip(), text)
        for heading in LEARNINGS:
            self.assertIn(heading, text)

    def test_oversize_prompt_goes_to_stdin_not_truncated(self):
        backend = actors.Backend("claude", "m", "high", "/bin/x")
        prompt = actors.render_context({"program": PROGRAM}) + "x" * actors.ARGV_PROMPT_LIMIT_BYTES
        argv = backend.argv(prompt, Path("/ws"))
        self.assertNotIn(prompt, argv)  # on stdin, not in argv
        self.assertTrue(backend._prompt_on_stdin(prompt))

    def test_variable_mode_strategy_file_and_required_reading(self):
        text = actors.render_context({"program": "CPU directive.\n\n" + PROGRAM})
        sections = actor_context.split_sections(text)
        strategy = [s for s in sections if s.key == "program_strategy"]
        self.assertEqual(len(strategy), 1)
        self.assertEqual(strategy[0].text.count("## Learnings "), len(LEARNINGS))
        with tempfile.TemporaryDirectory() as tmp:
            index = actor_context._index_text(
                Path(tmp), [(s, f"sections/{s.key}.md") for s in sections], {})
        for heading in LEARNINGS:
            self.assertIn(heading, index, "Learnings heading missing from the required-reading list")
        must = index.split("Read these before you propose", 1)[1].split("| # |", 1)[0]
        for heading in LEARNINGS:
            self.assertIn(heading, must)

    def test_learnings_headings_fit_the_jump_table(self):
        heads = actor_context._headings(PROGRAM)
        self.assertLessEqual(len(heads), actor_context.TOC_MAX_HEADINGS,
                             "program.md has more headings than the variable-mode jump table lists")

    def test_other_sections_unchanged(self):
        self.assertFalse(actor_context._must_read("shared_history", "## Learnings 2026-11-21 (x)"))
        self.assertFalse(actor_context._must_read("inbox", "## settled"))
        self.assertTrue(actor_context._must_read("program_strategy", "## Learnings 2026-11-21 (x)"))


if __name__ == "__main__":
    unittest.main()


class CensusJournalReachesPlanner(unittest.TestCase):
    """The ak_runend census hook writes outside git; program_text() must carry it to the planner in both modes."""

    def test_generated_census_journal_reaches_both_delivery_modes(self):
        from autokernel.loop import census_update as cu
        with tempfile.TemporaryDirectory() as d:
            j = Path(d) / "journal.md"
            j.write_text("# header\n\n" + cu.JOURNAL_HEAD + "\n- `row-x` (abc): canary-kernel 900 -> 1100 GB/s rank 1\n")
            text = loop.program_text(j)
            self.assertIn("canary-kernel 900 -> 1100 GB/s", text)
            self.assertTrue(text.startswith(PROGRAM.rstrip("\n")[:200]))
            self.assertIn("canary-kernel", actors.render_context({"program": text}))
            heads = [m.group(0) for m in re.finditer(r"(?m)^## Learnings .*$", text)]
            self.assertIn(cu.JOURNAL_HEAD, heads)

    def test_missing_journal_leaves_program_unchanged(self):
        self.assertEqual(loop.program_text(Path("/nonexistent/journal.md")), PROGRAM)



class SurfacesBlockReachesPlanner(unittest.TestCase):
    """The regenerated top-surfaces section gets its own slot: whole, ahead of the journal tail, in both delivery modes."""

    def _journal(self, d, n_lines):
        from autokernel.loop import census_update as cu
        c = json.load(open(cu.CENSUS))
        j = Path(d) / "journal.md"
        j.write_text("# header\n\n" + cu.JOURNAL_HEAD + "\n" + "".join(f"- `r{i}` (x): line {i}\n" for i in range(n_lines)))
        self.assertTrue(cu.write_surfaces_block(str(j), c))
        self.assertFalse(cu.write_surfaces_block(str(j), c))  # regenerated in place, never appended twice
        return cu, c, j

    def test_block_survives_journal_tail_cap_in_both_modes(self):
        with tempfile.TemporaryDirectory() as d:
            cu, c, j = self._journal(d, loop.BW_CENSUS_JOURNAL_MAX_LINES + 50)
            text = loop.program_text(j)
            top = c["surfaces"][0]
            self.assertEqual(text.count(cu.SURF_HEAD), 1)
            self.assertIn(f"{top['op']} · {top['path']}", text)
            self.assertIn("| Owner |", text)
            self.assertIn("**none**", text)
            self.assertNotIn("line 0\n", text)  # tail cap still applies to the journal lines
            self.assertIn(f"line {loop.BW_CENSUS_JOURNAL_MAX_LINES + 49}", text)
            self.assertNotIn(cu.SURF_BEGIN, text)
            self.assertIn(top["op"], actors.render_context({"program": text}))
            sections = actor_context.split_sections(actors.render_context({"program": text}))
            strategy = [s for s in sections if s.key == "program_strategy"]
            self.assertEqual(len(strategy), 1)
            self.assertIn(cu.SURF_HEAD, strategy[0].text)
            with tempfile.TemporaryDirectory() as tmp:
                index = actor_context._index_text(Path(tmp), [(s, f"sections/{s.key}.md") for s in sections], {})
            must = index.split("Read these before you propose", 1)[1].split("| # |", 1)[0]
            self.assertIn(cu.SURF_HEAD, must)
            self.assertLessEqual(len(actor_context._headings(text)), actor_context.TOC_MAX_HEADINGS)

    def test_block_alone_without_journal_section(self):
        from autokernel.loop import census_update as cu
        with tempfile.TemporaryDirectory() as d:
            j = Path(d) / "journal.md"
            j.write_text("# header\n")
            cu.write_surfaces_block(str(j), json.load(open(cu.CENSUS)))
            self.assertIn(cu.SURF_HEAD, loop.program_text(j))
