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

Run: python3 -m scripts.kernel_rnd.autokernel.check_regrowth_guards (requires a reviewed baseline)
Exit 0 clean, 1 on a violation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys

#: The rebuilt loop's budget. Arbitrary by design -- see the module docstring.
#: Raised 3000 -> 3400 on 2026-08-29 and 3400 -> 3450 on 2026-08-31, each deliberately
#: and each with its reason recorded here rather than in a commit nobody re-reads.
#:
#: THE 2026-08-31 RAISE, and why it is not the regrowth this guard exists to stop.
#: The package sat at EXACTLY 3400, so no functional change of any size could land --
#: a three-line bug fix (publish the anchor commit that candidates are actually
#: measured against, not the one the run started from; run 19 advanced the champion
#: twice while the dashboard reported it stuck) did not fit even after every comment
#: was stripped from it. A budget that blocks a three-line correctness fix is not
#: bounding regrowth, it is bounding maintenance. Raised by 50 -- enough that the next
#: few honest fixes land, small enough that the conversation returns soon. Operator
#: decision, 2026-08-31, asked for explicitly after the alternatives were put to them.
#:
#: The 3000 was set for a SEQUENTIAL loop of ~830 LOC. Concurrency -- seven lanes, a
#: serialized tail, champion arbitration, provisioning and pruning -- is `pipeline.py`
#: plus `pool.py`, ~640 lines, and it bought 5.6x throughput (run 13: 10 iterations in
#: 118.6 min; run 14: 14 in 29.8). That is a capability, not regrowth.
#:
#: The prose share is NOT written here any more -- `composition()` computes it and the
#: guard prints it on every run. Three hand measurements of this one number, three
#: different answers: 34% (2026-08-29, never re-derived as the package grew), 23.8%
#: (2026-08-31, under-counted -- it missed full-line `#` comments), and the computed
#: 30.8%. A stale prose share is exactly the number a future "just bump it, most of it
#: is comments" argument leans on, so it may not live in a comment. That prose is the
#: incident record, and it is load
#: bearing: every guard here exists because something specific went wrong, and a guard
#: whose reason has been deleted gets removed by the next person who finds it
#: inconvenient. The budget counts it, which means the budget is partly a documentation
#: budget. Stated so the next reader does not mistake 3,400 for 3,400 lines of logic.
#:
#: Against the 153,865 LOC this replaces, the point of the guard stands: it forces this
#: conversation at 2x, not at 50x.
LOOP_LOC_BUDGET = 3450


#: HISTORICAL RECORD: retain the 2026-08-31–09-04 budget rationale below verbatim.
#: LOOP_CODE_BUDGET is historical only; per-file manifest enforcement is defined below.
#: There is deliberately no aggregate --code-budget override.
#: THE ENFORCED BUDGET, 2026-08-31. `LOOP_LOC_BUDGET` above is still printed, because a
#: total-line number is the review burden a reader wants to see -- but it no longer
#: FAILS anything, and this does.
#:
#: Counting every line put the guard and its own doctrine in opposition. The note above
#: says the package's prose is the incident record and is load bearing; a total-line
#: budget makes deleting that prose the cheapest way to land a fix. That is not
#: hypothetical: on 2026-08-31 the package sat at exactly its budget and a three-line
#: correctness fix did not fit even after every comment was stripped from it, and the
#: budget was raised rather than the record deleted. The next person will not be as
#: scrupulous. A guard whose cheapest satisfaction is the thing it forbids is a guard
#: that will eventually be satisfied that way.
#:
#: Code-only, so a comment buys nothing and only deleting CODE fits. 2,100 against
#: today's 1,848 is 252 lines -- one more subsystem the size of pool.py's driver, or
#: thirty small fixes, before the conversation returns. Against the 153,865 LOC this
#: replaces the ratio the module docstring defends is unchanged: it still forces the
#: argument at 2x, not at 50x.
#:
#: 2,100 -> 2,160 on 2026-09-01: the operator-approved D1-D6 production-shaped rung
#: (docs/design/autokernel-production-shaped-rung.md §5.1-§5.3). +60 lines of loop
#: code, all wiring: the (surface, workload)-keyed floor lookup in bench.py, the
#: Comparison/status/headline records carrying their rung, the three --confirm-*
#: flags, one closure and one guarded gate call at commit_pooled, and the
#: ConfirmVetoed catch at the keep gate. The gate's decision arithmetic, the parity
#: checks and the census-based family live in `controller/` (rung_confirm.py,
#: workload_contract.py) -- the uncounted library -- on purpose. Raised by the
#: minimum that lands the approved work with 3 lines to spare, because a budget the
#: package sits EXACTLY on blocks the next three-line correctness fix (the
#: 2026-08-31 lesson above).
#:
#: 2,160 -> 2,210 on 2026-09-03: the operator-directed per-role model split (the roles
#: have since been swapped -- as of 2026-09-07 the planner is gpt-5.6-sol at high via
#: `codex exec` and the critic is Claude Fable 5.1 at medium via the `claude` CLI; the
#: line count is unchanged by that, only the two default constants moved).
#: +44 lines of loop code, all wiring: the `Backend`
#: dataclass whose `argv` is the whole per-CLI contract (TOML-quoted codex effort;
#: claude headless flags plus the sandbox note the freeze overlay makes necessary),
#: `backend_for`, the two role defaults, four `--planner-*`/`--critic-*` flags and
#: the `actors` banner line that records which models drove a run -- provenance
#: the R23-19 headline incident showed is not optional. Raised by the minimum that
#: lands it with 6 lines to spare, for the same reason as the raise above.
#:
#: 2,210 -> 2,400 on 2026-09-04: R23-43, the serving-throughput measurement core (operator
#: directive: keeps are demonstrated on llama-server under the champion's canonical recipe,
#: not a llama-bench proxy -- proven necessary when +35% of dec-b4 bench moved DFlash2 serving
#: ~0%). `loop/serving.py` (+~140 code: the general recipe, the np-concurrent aggregate-tok/s
#: measurement, the paired A/B, the A/A floor) plus headroom for the keep-gate wiring in
#: loop.py/run.py. This is a NEW capability the loop did not have, not regrowth of removed prose.
#: 2,400 -> 2,500 on 2026-09-04: R23-44 compound-then-gate policy core (operator directive:
#: "collect llama-bench keeps until they compound to 2x-3x noise floor before the
#: llama-server final champion advancement gate"). `loop/accumulate.py` (+~90 code: a
#: two-tier champion -- an accumulator that advances on cheap bench keeps, a
#: champion-of-record that advances only when a bundle compounds past the serving floor
#: AND the one serving gate confirms it). A per-keep serving gate vetoes every 1-3% keep
#: against the ~3.5% floor; batching makes the serving gate resolvable. NEW capability.
#: 2,500 -> 2,560 on 2026-09-04: R23-44 observability — `accumulator_state()` in run.py
#: exposes the two-tier bundle (keeps, compounded %, fire threshold, progress fraction) in the
#: loop status so the dashboard can render keeps accumulating toward the serving gate; the
#: operator asked to watch the bundle build. NEW observability surface, not regrowth.
LOOP_CODE_BUDGET = 2560

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



#: Versioned source-count contract. Changing either identifier requires a reviewed
#: migration because the manifest is meaningful only for this exact scan and metric.
LOOP_CODE_SCOPE = "autokernel.loop.non_test_python_source.v1"
LOOP_CODE_METRIC = "python_nonblank_noncomment_nondocstring_physical_lines.v1"
BASELINE_REVISION = "cdf6faa44d81ba12adbc99b9cd1eb47ea91fd4ba"
DEFAULT_CODE_BASELINE = Path(__file__).resolve().with_name("loop_code_budget_baseline.json")


def _source_files(root: Path) -> list[Path]:
    """The exact existing source scope: recursive Python files except test_*.py."""
    return [path for path in sorted(root.rglob("*.py"))
            if not path.name.startswith("test_")]


def file_composition(path: Path) -> tuple[int, int, int]:
    """Return `(code, prose, blank)` using the existing composition semantics."""
    import ast, io, tokenize
    src = path.read_text(encoding="utf-8")
    marked: set[int] = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None and node.body:
                first = node.body[0]
                marked.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            marked.add(tok.start[0])
    code = prose = blank = 0
    for lineno, line in enumerate(src.splitlines(), 1):
        if not line.strip():
            blank += 1
        elif lineno in marked:
            prose += 1
        else:
            code += 1
    return code, prose, blank


def composition_by_file(root: Path) -> dict[str, int]:
    """Code-line ceiling inputs for exactly the files inside the guarded scope."""
    if not root.is_dir():
        return {}
    return {path.relative_to(root).as_posix(): file_composition(path)[0]
            for path in _source_files(root)}


def composition(root: Path) -> tuple[int, int, int]:
    """`(code, prose, blank)` for the loop package, computed rather than remembered.

    Every hand measurement of this number has been wrong -- see the note at
    LOOP_LOC_BUDGET. The budget counts all three, so the split is what tells a reader
    whether a raise is buying capability or documentation.
    """
    code = prose = blank = 0
    if not root.is_dir():
        return code, prose, blank
    for path in _source_files(root):
        counts = file_composition(path)
        code += counts[0]
        prose += counts[1]
        blank += counts[2]
    return code, prose, blank


def loop_package_loc(root: Path) -> tuple[int, list[tuple[str, int]]]:
    """Non-test source lines in the rebuilt loop package."""
    if not root.is_dir():
        return 0, []
    rows = []
    for path in _source_files(root):
        rows.append((path.relative_to(root).as_posix(),
                     len(path.read_text(encoding="utf-8").splitlines())))
    return sum(count for _, count in rows), rows


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def manifest_for(files: dict[str, int], *, status: str = "GENERATED") -> dict[str, object]:
    """Build the versioned manifest representation; does not write any file."""
    return {"schema_version": 1, "status": status, "source_revision": BASELINE_REVISION,
            "scope": LOOP_CODE_SCOPE, "metric": LOOP_CODE_METRIC,
            "files": dict(sorted(files.items())), "exceptions": {}}


def check_code_baseline(baseline_path: Path, current: dict[str, int]) -> list[str]:
    """Fail closed on invalid, ungenerated, drifting, or exceeded baselines."""
    try:
        raw = json.loads(baseline_path.read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"cannot read valid baseline {baseline_path}: {exc}"]
    if not isinstance(raw, dict) or set(raw) != {
            "schema_version", "status", "source_revision", "scope", "metric", "files", "exceptions"}:
        return ["baseline fields do not match the required schema"]
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        return ["baseline schema_version must be integer 1"]
    if raw["status"] != "GENERATED":
        return ["baseline is UNGENERATED; an authorized owner must review generated ceilings"]
    if raw["source_revision"] != BASELINE_REVISION:
        return ["baseline source revision differs from the frozen revision"]
    if raw["scope"] != LOOP_CODE_SCOPE:
        return [f"baseline scope must be {LOOP_CODE_SCOPE!r}"]
    if raw["metric"] != LOOP_CODE_METRIC:
        return [f"baseline metric must be {LOOP_CODE_METRIC!r}"]
    files = raw["files"]
    if not isinstance(files, dict) or not files:
        return ["baseline files must be a non-empty object"]
    exceptions = raw["exceptions"]
    if not isinstance(exceptions, dict):
        return ["exceptions must be an object"]
    for name, entry in exceptions.items():
        if (not isinstance(entry, dict) or set(entry) != {"ceiling", "approval"}
                or type(entry["ceiling"]) is not int or entry["ceiling"] < 0
                or not isinstance(entry["approval"], str) or not entry["approval"].strip()):
            return [f"invalid exception for {name!r}: require ceiling and operator approval reference"]

    errors: list[str] = []
    for name, ceiling in files.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute()
                or any(part in ("", ".", "..") for part in name.split("/"))
                or PurePosixPath(name).as_posix() != name):
            errors.append(f"invalid relative POSIX source path: {name!r}")
        if type(ceiling) is not int or ceiling < 0:
            errors.append(f"ceiling for {name!r} must be a nonnegative integer (not bool)")
    if errors:
        return errors

    expected, recorded = set(current), set(files)
    for name in sorted(set(exceptions) - expected):
        errors.append(f"stale exception path: {name}")
    for name in sorted(expected - recorded):
        if name not in exceptions:
            errors.append(f"new in-scope source has no baseline entry: {name}")
    for name in sorted(recorded - expected):
        errors.append(f"stale baseline path is outside current scope: {name}")
    for name in sorted(expected & (recorded | set(exceptions))):
        ceiling = exceptions[name]["ceiling"] if name in exceptions else files[name]
        if current[name] > ceiling:
            errors.append(f"{name}: {current[name]} code lines exceeds ceiling "
                          f"{ceiling} by {current[name] - ceiling}")
    return errors


def generated_manifest_json(files: dict[str, int]) -> str:
    """Serialize proposed counts to stdout; callers must review before adoption."""
    return json.dumps(manifest_for(files), indent=2, sort_keys=False) + "\n"

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
    parser.add_argument("--baseline", type=Path, default=DEFAULT_CODE_BASELINE,
                        help="versioned per-file source ceiling manifest")
    parser.add_argument("--emit-baseline-json", action="store_true",
                        help="print current per-file ceilings for owner review; never writes a file")
    parser.add_argument("--budget", type=int, default=LOOP_LOC_BUDGET,
                        help="informational total-LOC comparison only")
    parser.add_argument("--check-doc-assertions", action="store_true",
                        help="also fail on tests that pin documentation prose. Off by "
                             "default while the superseded suites still exist; the "
                             "strip (P5) is what makes this enforceable.")
    args = parser.parse_args(argv)

    loop_root = args.loop_package or (args.package / "loop")
    problems = 0
    total, rows = loop_package_loc(loop_root)
    if args.emit_baseline_json:
        files = composition_by_file(loop_root)
        if not files:
            print(f"cannot generate baseline: no in-scope Python source at {loop_root}",
                  file=sys.stderr)
            return 1
        print(generated_manifest_json(files), end="")
        return 0

    if not rows:
        print(f"loop package not present yet at {loop_root} — LOC budget not applicable")
    else:
        code, prose, blank = composition(loop_root)
        print(f"loop package composition: {code} code / {prose} prose / {blank} blank "
              f"({100.0 * prose / max(1, code + prose + blank):.1f}% prose)")
        print(f"loop package: {total} LOC across {len(rows)} files "
              f"(informational budget {args.budget})")
        for name, count in sorted(rows, key=lambda item: -item[1])[:10]:
            print(f"    {count:>6}  {name}")
        if total > args.budget:
            print(f"\nNOTE: the loop package is {total} LOC, over the informational "
                  f"{args.budget}; this is not a violation.")

    current = composition_by_file(loop_root)
    baseline_errors = check_code_baseline(args.baseline, current)
    if baseline_errors:
        print("\nCODE BASELINE VIOLATION:", file=sys.stderr)
        for error in baseline_errors:
            print(f"  - {error}", file=sys.stderr)
        problems += 1
    else:
        print(f"per-file code baseline: {len(current)} files within reviewed ceilings")

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
