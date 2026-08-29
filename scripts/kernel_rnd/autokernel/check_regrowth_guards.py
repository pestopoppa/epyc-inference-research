#!/usr/bin/env python3
"""Enforced guards against the growth pattern that produced this codebase.

WHY THIS EXISTS
---------------
Between 2026-07-25 and 2026-08-28 the AutoKernel package took **271,216 insertions
against 92 deletions** -- a ratio of 2,948 : 1 -- across 514 commits, 427 of them in
a single week. It reached 153,865 LOC of source and 107,707 of tests to optimise a
CUDA/HIP backend that is 42,494 LOC: 6.2x the size of its own subject. Only 29 files
/ 48,634 LOC ever executed; 68% of the source has never run.

Two mechanisms made that possible, and both are checkable:

  * **Deletion was expensive.** `test_campaign_footprint.py` (109 KB) asserts the
    contents of `FOOTPRINT.md`, and `test_readme.py` asserts `README.md`, so removing
    a dead module required regenerating documentation in the same commit. A codebase
    where deleting costs more than adding only grows.
  * **Nothing bounded the size.** There was no number anyone could point at.

These guards are deliberately blunt. The LOC budget is arbitrary, and that is the
point: it forces the conversation at 2x rather than at 50x.

Run: python3 -m scripts.kernel_rnd.autokernel.check_regrowth_guards
Exit 0 clean, 1 on a violation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

#: The rebuilt loop's budget. Arbitrary by design -- see the module docstring.
#: Raised from 3000 to 3400 on 2026-08-29, deliberately and once.
#:
#: The 3000 was set for a SEQUENTIAL loop of ~830 LOC. Concurrency -- seven lanes, a
#: serialized tail, champion arbitration, provisioning and pruning -- is `pipeline.py`
#: plus `pool.py`, ~640 lines, and it bought 5.6x throughput (run 13: 10 iterations in
#: 118.6 min; run 14: 14 in 29.8). That is a capability, not regrowth.
#:
#: Measured before raising: 2,102 lines are code and 1,071 are docstrings and comments
#: -- 34% of the package is prose. That prose is the incident record, and it is load
#: bearing: every guard here exists because something specific went wrong, and a guard
#: whose reason has been deleted gets removed by the next person who finds it
#: inconvenient. The budget counts it, which means the budget is partly a documentation
#: budget. Stated so the next reader does not mistake 3,400 for 3,400 lines of logic.
#:
#: Against the 153,865 LOC this replaces, the point of the guard stands: it forces this
#: conversation at 2x, not at 50x.
LOOP_LOC_BUDGET = 3400

#: Documentation whose CONTENTS no test may assert. Asserting a doc's text makes
#: every deletion cost a regeneration, which is how 92 deletions happen in 5 weeks.
GUARDED_DOCS = ("FOOTPRINT.md", "README.md", "program.md", "HYPOTHESES.md",
                "HYPOTHESIS_PORTFOLIO_V2.md")

#: A test that READS a guarded doc is the precondition for asserting its prose, and
#: reading is what is reliably detectable: the real assertions compare a module
#: constant against `self.text`, not an inline literal, so a regex over assert lines
#: silently finds nothing. (It did -- the first version of this guard reported zero
#: hits against test_readme.py, which is exactly the vacuous check this file exists
#: to prevent elsewhere.) Detect the COUPLING, and let the report name it.
_DOC_READ = re.compile(
    r"(?:" + "|".join(re.escape(name) for name in
                      ("FOOTPRINT.md", "README.md", "program.md", "HYPOTHESES.md",
                       "HYPOTHESIS_PORTFOLIO_V2.md")) + r")")


def loop_package_loc(root: Path) -> tuple[int, list[tuple[str, int]]]:
    """Non-test source lines in the rebuilt loop package."""
    if not root.is_dir():
        return 0, []
    rows = []
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        rows.append((str(path.relative_to(root)),
                     len(path.read_text(encoding="utf-8").splitlines())))
    return sum(count for _, count in rows), rows


def doc_coupled_tests(package: Path) -> list[tuple[str, int, str]]:
    """Test files coupled to a guarded documentation file.

    Returns (path, occurrences, first_doc_named). A test that reads FOOTPRINT.md or
    README.md makes deleting a module cost a documentation regeneration in the same
    commit -- which is how a codebase records 271,216 insertions against 92 deletions.
    """
    hits = []
    for path in sorted(package.rglob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        found = _DOC_READ.findall(text)
        if found:
            hits.append((str(path.relative_to(package)), len(found), found[0]))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--loop-package", type=Path, default=None,
                        help="the rebuilt loop package (default <package>/loop)")
    parser.add_argument("--budget", type=int, default=LOOP_LOC_BUDGET)
    parser.add_argument("--check-doc-assertions", action="store_true",
                        help="also fail on tests that pin documentation prose. Off by "
                             "default while the superseded suites still exist; the "
                             "strip (P5) is what makes this enforceable.")
    args = parser.parse_args(argv)

    loop_root = args.loop_package or (args.package / "loop")
    problems = 0

    total, rows = loop_package_loc(loop_root)
    if not rows:
        print(f"loop package not present yet at {loop_root} — LOC budget not applicable")
    else:
        print(f"loop package: {total} LOC across {len(rows)} files "
              f"(budget {args.budget})")
        for name, count in sorted(rows, key=lambda item: -item[1])[:10]:
            print(f"    {count:>6}  {name}")
        if total > args.budget:
            print(f"\nVIOLATION: the loop package is {total} LOC, over its "
                  f"{args.budget} budget by {total - args.budget}.\n"
                  f"  This budget is arbitrary and that is the point: it forces the "
                  f"conversation at 2x rather than at 50x. Raise it deliberately, in "
                  f"a commit that says why, or delete something.", file=sys.stderr)
            problems += 1

    hits = doc_coupled_tests(args.package)
    if hits:
        label = "VIOLATION" if args.check_doc_assertions else "NOTE"
        print(f"\n{label}: {len(hits)} test file(s) are coupled to guarded "
              f"documentation. Each makes a deletion cost a regeneration.")
        for name, count, doc in sorted(hits, key=lambda row: -row[1])[:10]:
            print(f"    {count:>4} reference(s) to {doc:<28} {name}")
        if args.check_doc_assertions:
            problems += 1

    print(f"\n{problems} guard violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
