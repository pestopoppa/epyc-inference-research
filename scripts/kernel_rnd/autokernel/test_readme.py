#!/usr/bin/env python3
"""The README's load-bearing claims, asserted against the tree rather than read.

A README is the first thing anyone reads and the last thing anyone updates. This
file exists because three of its claims are the kind that decide whether someone
can use this package at all, and each of them was WRONG in a previous revision of
some document in this tree:

  1. **The one command runs, and it executes nothing.** `execution/README.md`
     documented a cold start whose module path was not the parser's own `prog`
     and whose flag set exited 2. A command in a README is a promise; this file
     drives it.
  2. **The essential/deferred split is one number, in one place.** The README
     quotes `FOOTPRINT.md`, which is itself asserted against the walked import
     graph. A figure repeated in two documents drifts in one of them, so the
     rule here is that the README may not contain a campaign-path or deferred
     figure that `FOOTPRINT.md` does not also state.
  3. **A capability the docs describe actually exists.** The specific defect this
     guards is the note (since corrected) claiming the CPU-region claim was
     UNACQUIRABLE while `acquire_cpu_region_claim()` sat in the tree, callable,
     as step 1 of the execution runbook. The dangerous half of a wrong reason is
     that it tells the next reader not to bother trying.

Like `test_campaign_footprint.py`, this file never IMPORTS the package to inspect
it where a parse will do — but it does import `campaign` to drive `main()`, which
is safe precisely because dry run reads nothing from the host and spawns nothing.
`TestTheDocumentedCommandExecutesNothing` is what makes that safety an assertion
rather than a belief.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from . import campaign
from .execution import cpu_region_claim

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
FOOTPRINT = HERE / "FOOTPRINT.md"

#: The command the README tells a reader to run, as its argv tail. Kept here as
#: DATA so the test drives the same thing the document prints.
DOCUMENTED_ARGV = ["--model", "/mnt/raid0/llm/models/nonexistent-for-dry-run.gguf"]

#: Where the archaeology starts. Everything a new reader needs is above it.
ARCHAEOLOGY_HEADING = "## What is implemented"


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


class TestTheOpeningTeachesTheFourThings(unittest.TestCase):
    """Ten lines in, a reader knows what this is, how to start it, and its state.

    The previous opening led with the owning-design link and the release-authority
    scope — true, and not one of them answers "what do I type". The command was
    at line 24, below three paragraphs of governance.
    """

    def setUp(self) -> None:
        self.text = readme_text()
        self.head = self.text.split(ARCHAEOLOGY_HEADING)[0]

    def test_the_archaeology_heading_still_exists(self):
        """Control: the split above is real, so `head` is not the whole file."""
        self.assertIn(ARCHAEOLOGY_HEADING, self.text)
        self.assertLess(len(self.head), len(self.text))

    def test_the_command_appears_before_the_architecture(self):
        module_path = "scripts.kernel_rnd.autokernel.campaign"
        self.assertIn(module_path, self.head,
                      "the entrypoint command must be above the module table")

    def test_the_command_is_in_the_first_ten_non_blank_lines(self):
        """'in ten lines' is the requirement, so it is the assertion."""
        lines = [ln for ln in self.head.splitlines() if ln.strip()][:10]
        self.assertTrue(
            any("scripts.kernel_rnd.autokernel.campaign" in ln for ln in lines),
            "the one command is not within the first ten non-blank lines:\n"
            + "\n".join(lines))

    def test_the_opening_says_nothing_has_ever_been_built(self):
        """The single most important fact about this package's state.

        A reader who does not learn this will read the module table below as a
        description of a working system. It is a description of a built one.
        """
        self.assertRegex(
            self.head,
            r"no candidate has ever been built",
            "the opening must state that no candidate has ever been built")

    def test_the_opening_states_the_essential_deferred_split(self):
        self.assertIn("FOOTPRINT.md", self.head)
        self.assertRegex(self.head, r"deferred|unreachable")

    def test_the_opening_says_dry_run_is_the_default(self):
        self.assertRegex(self.head, r"[Dd]ry run is the default")


class TestTheDocumentedFiguresComeFromOnePlace(unittest.TestCase):
    """The README may quote FOOTPRINT.md's totals; it may not invent its own.

    `FOOTPRINT.md` is regenerated from the walked import graph and asserted
    against it row by row. Any campaign-path or deferred figure in the README
    that FOOTPRINT does not also state is a second source of truth, and the one
    that can disagree is the one nobody regenerates.
    """

    #: Five digits with an optional thousands comma — the shape of these totals.
    FIGURE = re.compile(r"\b\d{2},\d{3}\b")

    def setUp(self) -> None:
        self.head = readme_text().split(ARCHAEOLOGY_HEADING)[0]
        self.footprint = FOOTPRINT.read_text(encoding="utf-8")

    def test_the_footprint_document_exists_and_states_totals(self):
        """Control: the corpus this test compares against is non-empty."""
        found = set(self.FIGURE.findall(self.footprint))
        self.assertGreaterEqual(len(found), 3, "FOOTPRINT.md states no totals")

    def test_every_five_digit_figure_in_the_opening_is_in_footprint(self):
        in_readme = set(self.FIGURE.findall(self.head))
        in_footprint = set(self.FIGURE.findall(self.footprint))
        orphans = sorted(in_readme - in_footprint)
        self.assertEqual(
            orphans, [],
            f"the README states {orphans}, which FOOTPRINT.md does not. Either "
            f"regenerate FOOTPRINT.md (`--refresh`) or quote what it says; a "
            f"figure stated in two documents drifts in exactly one of them.")

    def test_the_opening_actually_quotes_some_figure(self):
        """Control: the test above passes vacuously if the README has none."""
        self.assertTrue(self.FIGURE.findall(self.head),
                        "the opening quotes no figure, so the split is unstated")


class TestTheDocumentedCommandExecutesNothing(unittest.TestCase):
    """The dry run is driven here, with every spawn primitive booby-trapped.

    "Executes nothing" is the property the whole default rests on: a driver on a
    shared host that can benchmark by accident is worse than no driver. Two of
    the six A/A runs on 2026-08-04 were destroyed by a legitimate co-tenant, and
    the co-tenant did nothing wrong. So this does not inspect the code for spawn
    calls — it replaces them with landmines and runs the documented command.
    """

    #: Every way this package could start a process. `os.fork` is included even
    #: though nothing uses it: the point is to catch a future spawn, not to
    #: enumerate today's.
    TRAPPED = (
        (subprocess, "Popen"), (subprocess, "run"), (subprocess, "call"),
        (subprocess, "check_call"), (subprocess, "check_output"),
        (os, "system"), (os, "posix_spawn"), (os, "fork"), (os, "execv"),
    )

    def setUp(self) -> None:
        self.spawns: list = []
        self._saved: list = []
        for module, name in self.TRAPPED:
            if not hasattr(module, name):
                continue
            self._saved.append((module, name, getattr(module, name)))

            def landmine(*args, _n=f"{module.__name__}.{name}", **kwargs):
                self.spawns.append((_n, args, kwargs))
                raise AssertionError(f"a dry run called {_n}")

            setattr(module, name, landmine)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for module, name, original in self._saved:
            setattr(module, name, original)

    def test_the_landmines_are_armed(self):
        """Control. Without this, a mis-wired setUp makes every test below pass.

        `subprocess.run` here is the landmine, not the real one; if the patching
        silently failed this would spawn `true` and the assertion would not fire,
        which is the shape of a fixture that removes the signal under test.
        """
        with self.assertRaises(AssertionError):
            subprocess.run(["/bin/true"])
        self.assertEqual(len(self.spawns), 1)

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = campaign.main(argv)
        return code, buffer.getvalue()

    def test_the_readmes_command_succeeds_and_spawns_nothing(self):
        code, out = self._run(DOCUMENTED_ARGV)
        self.assertEqual(self.spawns, [], "the dry run started a process")
        self.assertEqual(code, 0, out)
        self.assertIn("DRY RUN", out)

    def test_the_dry_run_emits_no_speed_number(self):
        """A dry run that produces a number is a real run with a flag on it."""
        _, out = self._run(DOCUMENTED_ARGV)
        self.assertNotIn("tokens_per_s:", out)
        self.assertIn("dry_run_composed", out)

    def test_help_spawns_nothing_either(self):
        with self.assertRaises(SystemExit) as raised:
            self._run(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(self.spawns, [])

    def test_execute_is_refused_without_the_host_attestation(self):
        """Compliant-path control's opposite: the dangerous flag is not free.

        The refusal is a non-zero RETURN, not a `SystemExit`, and it happens
        before anything is acquired — so the assertion that matters alongside
        the exit code is that no landmine tripped.
        """
        code, _ = self._run(["--execute"] + DOCUMENTED_ARGV)
        self.assertEqual(code, 2)
        self.assertEqual(self.spawns, [])


class TestDocumentedCapabilitiesExist(unittest.TestCase):
    """A doc that says a thing cannot be done, about a thing that can, is worse
    than no doc: it stops the next reader trying. The corrected note about the
    CPU-region claim is the instance; this is the guard."""

    def test_the_cpu_region_claim_is_acquirable(self):
        self.assertTrue(callable(cpu_region_claim.acquire_cpu_region_claim))

    def test_no_document_in_this_package_calls_it_unacquirable(self):
        pattern = re.compile(
            r"(cpu[- ]region claim|CPU[- ]region claim)[^.\n]{0,80}"
            r"(unacquirable|cannot be acquired|is not acquirable)",
            re.IGNORECASE)
        offenders = []
        for path in sorted(HERE.rglob("*.md")):
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(HERE)))
        self.assertEqual(
            offenders, [],
            f"{offenders} say the CPU-region claim cannot be acquired, but "
            f"cpu_region_claim.acquire_cpu_region_claim is callable and is step "
            f"1 of the execution runbook's cold start.")

    def test_the_pattern_would_catch_the_claim_it_is_looking_for(self):
        """Control: the regex above is not a permanently-green no-op."""
        pattern = re.compile(
            r"(cpu[- ]region claim|CPU[- ]region claim)[^.\n]{0,80}"
            r"(unacquirable|cannot be acquired|is not acquirable)",
            re.IGNORECASE)
        self.assertTrue(pattern.search(
            "the CPU-region claim is unacquirable today"))
        self.assertTrue(pattern.search(
            "The cpu region claim cannot be acquired from here."))


if __name__ == "__main__":
    unittest.main()
