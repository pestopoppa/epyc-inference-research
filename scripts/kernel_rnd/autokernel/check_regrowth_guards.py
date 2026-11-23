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

These guards are deliberately blunt. The loop budget is ONE number -- the measured
code-line total of loop/ against `total_ceiling` -- that cannot drift up silently and
that makes deleting cheaper than adding. History, friction measurements and the
total-budget redesign (operator-approved 2026-11-23): epyc-root audit
/mnt/raid0/llm/tmp/code-budget-audit-20261123/REPORT.md §5.

Run: python3 -m scripts.kernel_rnd.autokernel.check_regrowth_guards (requires a reviewed baseline)
Exit 0 clean, 1 on a violation (or a refused --tighten/--tighten-total).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import json
from pathlib import Path, PurePosixPath
import re
import sys

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

    Every hand measurement of this number has been wrong, so it is computed. Only
    code counts against the budget; the split tells a reader what a raise would buy.
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


_REQUIRED_KEYS = {"schema_version", "status", "source_revision", "scope", "metric", "files"}
_OPTIONAL_KEYS = {"exceptions", "total_ceiling", "total_ceiling_approval"}
#: The only commit trailer that lets total_ceiling go UP (the ratify scripts write it).
RATIFY_TRAILER = "Operator-Ratified: yes"


def warn_band(ceiling: int) -> int:
    """Per-file slack before a file is listed as a trim-or-split candidate."""
    return max(10, math.ceil(ceiling * 0.05))


def _load_baseline(baseline_path: Path) -> tuple[dict | None, list[str]]:
    try:
        raw = json.loads(baseline_path.read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, [f"cannot read valid baseline {baseline_path}: {exc}"]
    return raw, validate_baseline(raw)


def validate_baseline(raw: object) -> list[str]:
    """Schema check. `exceptions` is optional (retired by the total-budget ratify)."""
    if not isinstance(raw, dict) or not _REQUIRED_KEYS <= set(raw) <= _REQUIRED_KEYS | _OPTIONAL_KEYS:
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
    if ("total_ceiling" in raw) != ("total_ceiling_approval" in raw):
        return ["total_ceiling and total_ceiling_approval must be present together"]
    if "total_ceiling" in raw:
        if type(raw["total_ceiling"]) is not int or raw["total_ceiling"] < 0:
            return ["total_ceiling must be a nonnegative integer (not bool)"]
        if not isinstance(raw["total_ceiling_approval"], str) or not raw["total_ceiling_approval"].strip():
            return ["total_ceiling_approval must name the operator ratification"]
    files = raw["files"]
    if not isinstance(files, dict) or not files:
        return ["baseline files must be a non-empty object"]
    exceptions = raw.get("exceptions", {})
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
    return errors


def per_file_ceilings(raw: dict) -> dict[str, int]:
    """Advisory per-file ceilings; an exception overrides its files entry."""
    ceilings = dict(raw["files"])
    ceilings.update({name: entry["ceiling"] for name, entry in raw.get("exceptions", {}).items()})
    return ceilings


def effective_total_ceiling(raw: dict) -> tuple[int, str]:
    """`(ceiling, source)`. Before the operator signs total_ceiling in, the bound is the
    sum of the per-file ceilings -- no tighter and no looser than the per-file regime."""
    if "total_ceiling" in raw:
        return raw["total_ceiling"], f"total_ceiling ({raw['total_ceiling_approval']})"
    return sum(per_file_ceilings(raw).values()), "fallback: sum of per-file ceilings (total_ceiling not signed yet)"


def evaluate(baseline_path: Path, current: dict[str, int]) -> dict[str, object]:
    """FAIL only on an invalid baseline or on the measured total over the ceiling.
    New files are auto-admitted (they count against the total); per-file overruns and
    stale paths are warnings."""
    raw, errors = _load_baseline(baseline_path)
    report: dict[str, object] = {"errors": errors, "warnings": [], "candidates": [],
                                 "total": sum(current.values()), "ceiling": None,
                                 "source": None, "raw": raw}
    if errors:
        return report
    ceilings = per_file_ceilings(raw)
    ceiling, source = effective_total_ceiling(raw)
    report["ceiling"], report["source"] = ceiling, source
    total = report["total"]
    if total > ceiling:
        errors.append(f"loop/ total {total} code lines exceeds {source} {ceiling} by {total - ceiling}")
    warnings: list[str] = report["warnings"]
    for name in sorted(set(ceilings) - set(current)):
        warnings.append(f"stale baseline path (deleted or moved; its lines are freed): {name}")
    for name in sorted(set(current) - set(ceilings)):
        warnings.append(f"new module auto-admitted under the total: {name} ({current[name]} code lines)")
    for name in sorted(set(current) & set(ceilings)):
        over = current[name] - ceilings[name]
        if over > warn_band(ceilings[name]):
            report["candidates"].append(
                f"{name}: {current[name]} code lines, {over} over its advisory ceiling "
                f"{ceilings[name]} (band {warn_band(ceilings[name])})")
    return report


def check_code_baseline(baseline_path: Path, current: dict[str, int]) -> list[str]:
    """The failing subset of `evaluate`: invalid baseline or total over the ceiling."""
    return list(evaluate(baseline_path, current)["errors"])


def check_total_raise(previous: dict | None, new: dict, message: str | None) -> list[str]:
    """Raising the total bound is human-only: it needs the ratify trailer on the commit
    AND a RATIFY-* approval. `message=None` means uncommitted, which can never raise."""
    if (previous is None or "total_ceiling" not in new
            or validate_baseline(previous) or validate_baseline(new)):
        return []  # before sign-in the per-file regime's own review applies, unchanged
    before, _ = effective_total_ceiling(previous)
    after, _ = effective_total_ceiling(new)
    if after <= before:
        return []
    signed = (message is not None and RATIFY_TRAILER in message
              and str(new.get("total_ceiling_approval", "")).startswith("RATIFY-"))
    if signed:
        return []
    where = "an uncommitted edit" if message is None else "a commit without the operator ratify trailer"
    return [f"total ceiling raised {before} -> {after} by {where}; raising is human-only "
            f"(an operator ratify script under epyc-root scripts/operator/)"]


def _git(cwd: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _parse(text: str | None) -> dict | None:
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def raise_check_from_git(baseline_path: Path) -> tuple[list[str], str]:
    """Run check_total_raise on (HEAD vs working copy) and on the last commit that
    touched the baseline vs its parent. Returns `(errors, note)`."""
    cwd = baseline_path.resolve().parent
    top = _git(cwd, "rev-parse", "--show-toplevel")
    if top is None:
        return [], "raise check skipped: baseline is not in a git checkout"
    cwd = Path(top.strip()).resolve()
    rel = baseline_path.resolve().relative_to(cwd).as_posix()
    working = _parse(baseline_path.read_text(encoding="utf-8"))
    errors = check_total_raise(_parse(_git(cwd, "show", f"HEAD:{rel}")), working, None) if working else []
    last = (_git(cwd, "log", "-1", "--format=%H", "--", rel) or "").strip()
    if not last:
        return errors, "raise check: no committed history for the baseline"
    parent = _parse(_git(cwd, "show", f"{last}^:{rel}"))
    if parent is None and _git(cwd, "rev-parse", "--verify", "-q", f"{last}^") is None:
        return errors, f"raise check incomplete: parent of {last[:12]} not fetched (shallow clone?)"
    errors += check_total_raise(parent, _parse(_git(cwd, "show", f"{last}:{rel}")) or {},
                                _git(cwd, "log", "-1", "--format=%B", last) or "")
    return errors, f"raise check: last baseline commit {last[:12]} and working copy"


def _write_baseline(baseline_path: Path, raw: dict) -> None:
    baseline_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tighten(baseline_path: Path, current: dict[str, int], *, per_file: bool,
            total: bool) -> list[str]:
    """Ratchet DOWN only, and only on a green run with total_ceiling signed in.
    Returns refusals (empty on success). No number may ever go up."""
    report = evaluate(baseline_path, current)
    if report["errors"]:
        return ["refusing to tighten a red baseline"]
    raw = report["raw"]
    if "total_ceiling" not in raw:
        return ["refusing to tighten before total_ceiling is signed in: lowering per-file "
                "ceilings would lower the fallback total, and writing total_ceiling is the "
                "operator's migration signature"]
    new = json.loads(json.dumps(raw))
    if per_file:
        new["files"] = {name: min(ceiling, current[name])
                        for name, ceiling in raw["files"].items() if name in current}
        if "exceptions" in raw:
            new["exceptions"] = {name: {**entry, "ceiling": min(entry["ceiling"], current[name])}
                                 for name, entry in raw["exceptions"].items() if name in current}
        if not new["files"]:
            return ["refusing to tighten: no baseline file is still in scope"]
    if total:
        old, measured = raw["total_ceiling"], report["total"]
        new["total_ceiling"] = max(measured, old - (old - measured) // 2)
    ceilings_before, ceilings_after = per_file_ceilings(raw), per_file_ceilings(new)
    if (new["total_ceiling"] > raw["total_ceiling"]
            or any(ceilings_after[n] > ceilings_before[n] for n in ceilings_after)):
        return ["internal error: tighten would raise a number; nothing written"]
    _write_baseline(baseline_path, new)
    return []


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
    parser.add_argument("--baseline", type=Path, default=None,
                        help="versioned loop code-budget baseline")
    parser.add_argument("--emit-baseline-json", action="store_true",
                        help="print current per-file counts for owner review; never writes a file")
    parser.add_argument("--tighten", action="store_true",
                        help="on a green run, lower per-file ceilings to current size and drop stale paths")
    parser.add_argument("--tighten-total", action="store_true",
                        help="on a green run, total_ceiling = max(current, total - deleted/2)")
    parser.add_argument("--no-raise-check", action="store_true",
                        help="skip the git-history check that total_ceiling was not raised unsigned")
    parser.add_argument("--check-doc-assertions", action="store_true",
                        help="also fail on tests that pin documentation prose. Off by "
                             "default while the superseded suites still exist; the "
                             "strip (P5) is what makes this enforceable.")
    args = parser.parse_args(argv)
    baseline = args.baseline or DEFAULT_CODE_BASELINE

    loop_root = args.loop_package or (args.package / "loop")
    problems = 0
    total, rows = loop_package_loc(loop_root)
    current = composition_by_file(loop_root)
    if args.emit_baseline_json:
        if not current:
            print(f"cannot generate baseline: no in-scope Python source at {loop_root}",
                  file=sys.stderr)
            return 1
        print(generated_manifest_json(current), end="")
        return 0
    if args.tighten or args.tighten_total:
        refusals = tighten(baseline, current, per_file=args.tighten, total=args.tighten_total)
        for refusal in refusals:
            print(f"REFUSED: {refusal}", file=sys.stderr)
        if refusals:
            return 1
        print(f"tightened {baseline} (lowered only)")

    if not rows:
        print(f"loop package not present yet at {loop_root} — code budget not applicable")
    else:
        code, prose, blank = composition(loop_root)
        print(f"loop package composition: {code} code / {prose} prose / {blank} blank "
              f"({100.0 * prose / max(1, code + prose + blank):.1f}% prose)")
        print(f"loop package: {total} LOC across {len(rows)} files (informational)")
        for name, count in sorted(rows, key=lambda item: -item[1])[:10]:
            print(f"    {count:>6}  {name}")
    for label, extra in (("controller/", args.package / "controller"),
                         ("scripts/benchmark/autokernel_*", args.package.resolve().parents[1] / "benchmark")):
        if extra.is_dir():
            paths = [p for p in _source_files(extra)
                     if label.startswith("controller") or p.name.startswith("autokernel_")]
            if paths:
                print(f"informational, not budgeted: {label} {sum(file_composition(p)[0] for p in paths)} "
                      f"code lines across {len(paths)} files")

    report = evaluate(baseline, current)
    errors = list(report["errors"])
    if not args.no_raise_check and not errors:
        raise_errors, note = raise_check_from_git(baseline)
        errors += raise_errors
        print(note)
    if errors:
        print("\nCODE BUDGET VIOLATION:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        problems += 1
    else:
        print(f"loop code budget: {report['total']} / {report['ceiling']} code lines "
              f"[{report['source']}], {report['ceiling'] - report['total']} headroom")
    for warning in report["warnings"]:
        print(f"WARN: {warning}")
    if report["candidates"]:
        print(f"WARN: {len(report['candidates'])} trim-or-split candidate(s) over the per-file "
              f"band max(+10, +5%) (advisory, never fails):")
        for row in report["candidates"]:
            print(f"    {row}")

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
