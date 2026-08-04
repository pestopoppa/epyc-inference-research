#!/usr/bin/env python3
"""Fail when the model registry cites evidence that cannot be RESOLVED.

WHY THIS EXISTS
---------------
`MEASUREMENT.md:139-141` (consolidated apply-time ratification) requires evidence
HASHES in the apply-time bundle, but says nothing about whether the evidence must still
be there. That is a real hole, not a pedantic one: a bundle can hash a file sitting in
`/tmp`, be signed by the operator, and become unverifiable the moment that file is
swept. The hash proves nothing once there is no artifact left to check it against —
it degrades from a verification into an assertion, silently, with no event to notice.

On 2026-08-02 the master registry was found citing 157 distinct paths under
`/mnt/raid0/llm/tmp/` as the evidence behind ratified, production-affecting claims,
including the MMMU-250 result that had gated a live vision model cutover. Nothing had
been lost; the whole set was one cleanup away. 4.0 MiB carried nearly all of it.

WHAT THIS CHECKS, AND WHAT CHANGED (2026-08-03)
-----------------------------------------------
THE RULE NOW: a cited evidence path must RESOLVE, on this host, to something readable
that is not sitting in a scratch directory. That is all, and it is the whole invariant
the script was written for — *a hash with no artifact behind it is an assertion, not a
verification.*

THE RULE BEFORE: the same, PLUS a requirement that the artifact live inside a git
repository, ideally as a repo-relative `epyc-inference-research/data/<campaign>/` path.
Absolute paths warned; anything durable but outside a repo was a hard ERROR.

That second half is now WRONG and has been removed. The operator ruled on 2026-08-03
that research material reaches GitHub ONLY as distilled knowledge and REFERENCES in the
project wiki, never as raw material, **regardless of size** — it is a question of kind,
not of bytes. Benchmark suites, benchmark data, campaign output, run bundles and logs
belong on local disk and gitignored. A clone carries the FINDINGS, not the substrate
they were derived from. The old verdicts pushed in exactly the opposite direction: they
rewarded committing raw campaign output and failed a citation for pointing at a durable
local artifact that the ruling says must NOT be committed. `.gitignore` in this repo now
un-tracks whole campaign trees (`data/judge_suite_headtohead_20260802/`,
`data/swe_verified_preliminary_20260724/`, `data/cpu_prefill_compute/*/`, …); under the
old rule those citations were "fine" only by accident of the files still being on disk,
and a checker that graded committedness would have started failing them.

So: committedness is no longer graded. RESOLVABILITY is. `scripts/kernel_rnd/autokernel/
storage.py` already reached the same conclusion independently — its `durable_untracked`
class is a first-class VALID durability class alongside `carried_in_git`.

NOT WEAKENED. The scratch roots stay a hard, ungrantable ERROR, and they are now matched
through symlinks as well (`storage.py:is_scratch_path` already did this; a scratch path
reachable via a symlink is still a scratch path). Scratch-ness is judged on the RESOLVED
path by the same test for relative and absolute citations -- corrected 2026-08-04, when
the relative arm was found short-circuiting past the symlink guard entirely, exempting
the dominant citation form (416 of 421) from the one rule that must never be grantable.
A citation that resolves nowhere is still a hard ERROR. What was dropped is only the part
that graded WHERE a *surviving* artifact lives.

`MEASUREMENT.md` is HUMAN-AMENDMENT-ONLY (the measurement trust boundary), so this
script enforces a CONVENTION and never amends the constitution. Its §5 durability clause
still spells the old in-repo requirement and names this file as its enforcer; the 2026-
08-03 ruling supersedes that clause's *location* half, and reconciling the constitution
text is a human amendment that this script must not and does not perform.

WHAT COUNTS AS A CITATION
-------------------------
Whitelist, not blacklist. A path-shaped token in the registry is in scope only if it
resolves under a root we care about:

  * an EPHEMERAL root (`/tmp`, `/var/tmp`, `/dev/shm`, `/run`, `/mnt/raid0/llm/tmp`)
    -- always in scope, and always an error;
  * an ARTIFACT TREE (`data/`, `artifacts/`, `benchmarks/`, `measurement/`), whether
    cited relative to this repo, absolutely inside it, inside a known sibling repo, or
    on some other durable mount.

Everything else -- model weights under `/mnt/raid0/llm/models`, kernel build trees,
`/opt/rocm`, `/sys`, container mount targets -- is out of scope by construction. Those
are inputs and system paths, not measurement results, and `validate_model_registry.py`
already checks that weights exist. Defining scope as a whitelist matters, and that
reasoning is UNCHANGED by the retarget: a blacklist grows an exemption every time it is
inconvenient, and the one exemption that must never be grantable is the scratch root.

VERDICTS
--------
  OK            resolves to a readable artifact on a durable root. In-repo or not,
                tracked or gitignored, relative or absolute -- all equally fine.
  WAIVED_LOST   WARN  -- artifact is gone and the line says so, verbatim, with an
                         `ARTIFACT LOST` marker. A recorded loss, not a silent one.
  EPHEMERAL     ERROR -- cited from a scratch directory (symlinks followed). One
                         cleanup from unverifiable. Never grantable.
  MISSING       ERROR -- resolves nowhere. The hash has nothing to check.
  UNREADABLE    ERROR -- the path exists but cannot be opened, so the hash still
                         cannot be recomputed. Same failure, different cause.

Exit 0 when there are no errors. Warnings never fail unless `-W`/`--warnings-as-errors`.

USAGE
-----
    python3 scripts/validate/check_evidence_durability.py
    python3 scripts/validate/check_evidence_durability.py --fix-hint
    python3 scripts/validate/check_evidence_durability.py --json
    python3 scripts/validate/check_evidence_durability.py path/to/registry.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------- config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"

# Scratch roots. A citation under any of these is non-durable by definition; this
# tuple is the one thing in the file that must never acquire an exemption.
EPHEMERAL_ROOTS = (
    "/mnt/raid0/llm/tmp",
    "/tmp",
    "/var/tmp",
    "/dev/shm",
    "/run/user",
)

# Sibling repositories of the same coordinated workspace. SCOPE ONLY since the
# 2026-08-03 retarget: living in a sibling repo is no longer a verdict of its own
# (it used to WARN as "durable but not in the mandated home"). The tuple survives
# because it keeps scoping deterministic -- a path under one of these roots is an
# evidence citation only if its repo-relative portion starts with an artifact
# tree, the same test applied to this repo.
SIBLING_REPO_ROOTS = (
    "/mnt/raid0/llm/epyc-root",
    "/mnt/raid0/llm/epyc-orchestrator",
    "/workspace",
)

# The ARTIFACT TREES. A path is an evidence citation only if its repo-relative portion
# starts with one of these -- the same test for relative and absolute paths, so a
# citation cannot change category by being spelled differently.
#
# `scripts/`, `handoffs/`, `docs/`, `orchestration/`, `research/`, `logs/` are
# deliberately absent: those are code and prose cross-references, not measurement
# results. Whether they resolve is a real hygiene question and a different one --
# folding it in here buries the evidence signal under several dozen stale doc links,
# and `research/` in particular collides with English ("a research/catalogue model").
ARTIFACT_TREES = ("data/", "artifacts/", "benchmarks/", "measurement/")

# An artifact that is gone, recorded as gone. The marker must sit on the citing line.
LOST_MARKER = "ARTIFACT LOST"
# A line documenting where an artifact USED to live is history, not a live citation.
PROVENANCE_MARKERS = ("REPOINTED from",)

TRAILING_PUNCT = ".,;:`)]}>'\"|"

# A path token: absolute, or repo-relative with a known prefix. Brace groups are kept
# so `escalation-{baseline,candidate}-scored.json` survives to be expanded below.
# `:` is inside the class so a `...log:8` line reference is captured with the token and
# split off deliberately, rather than silently truncated by the tokeniser.
_PATH_TOKEN = re.compile(
    r'(?<![\w.\-])(?:/|(?:' + "|".join(re.escape(p) for p in ARTIFACT_TREES) + r'))'
    r'[A-Za-z0-9_./{},*+@:\-]*'
)
_LINEREF = re.compile(r':\d+$')


# ---------------------------------------------------------------------- data model

@dataclass
class Citation:
    raw: str
    path: str
    line: int
    lineref: str = ""
    verdict: str = ""
    severity: str = ""          # "error" | "warn" | "ok"
    resolved: str = ""
    hint: str = ""
    context: str = ""


@dataclass
class Result:
    registry: str
    repo: str
    citations: list = field(default_factory=list)
    campaign_issues: list = field(default_factory=list)

    @property
    def errors(self):
        return [c for c in self.citations if c.severity == "error"]

    @property
    def warnings(self):
        return [c for c in self.citations if c.severity == "warn"]


# ------------------------------------------------------------------------ extraction

def expand_braces(token: str) -> list[str]:
    """`a{x,y}b` -> [`axb`, `ayb`]. Registry prose uses shell brace groups to cite a
    pair of sibling artifacts on one line; without this they read as one missing path."""
    m = re.search(r'\{([^{}]*)\}', token)
    if not m:
        return [token]
    out = []
    for alt in m.group(1).split(","):
        out.extend(expand_braces(token[:m.start()] + alt + token[m.end():]))
    return out


def _clean(token: str) -> tuple[str, str]:
    """Strip a trailing `:<line>` reference and any trailing prose punctuation."""
    lineref = ""
    m = _LINEREF.search(token)
    if m:
        lineref, token = m.group(0), token[:m.start()]
    while token and token[-1] in TRAILING_PUNCT:
        token = token[:-1]
    return token, lineref


def _in_scope(path: str, repo: Path) -> bool:
    """In scope iff it is a scratch path, or an artifact-tree path in a known repo.

    Ephemeral is checked FIRST and has no artifact-tree requirement: anything cited out
    of a scratch directory is a durability problem whatever it is called. Everything
    else must land in an artifact tree, which is the one rule that keeps the report
    about evidence instead of about every path-shaped string in a 10k-line file.
    """
    if path.startswith(EPHEMERAL_ROOTS):
        return True
    if path.startswith(ARTIFACT_TREES):          # repo-relative
        return True
    for root in (str(repo),) + SIBLING_REPO_ROOTS:
        if path.startswith(root + "/"):
            return path[len(root) + 1:].startswith(ARTIFACT_TREES)
    # An absolute path in no known repo can still be an evidence tree somebody parked on
    # a random mount -- and since 2026-08-03 that is the EXPECTED home for raw campaign
    # output, not an anomaly, so it must stay in scope. Require the artifact-tree segment
    # to sit below the filesystem root so container mount targets like `/data/indices`,
    # which name nothing on this host, are not mistaken for evidence.
    segs = path.strip("/").split("/")
    return any(f"{s}/" in ARTIFACT_TREES for s in segs[1:-1])


def extract_citations(text: str, repo: Path = REPO_ROOT) -> list[Citation]:
    """Every in-scope path-shaped token in the registry, comments and prose included.

    Comments are deliberately NOT skipped. Most of the scratch citations this was
    written for lived in `#` commentary and inside quoted observation strings, which is
    exactly where a structured, key-driven scan would have missed all of them.
    """
    cites: list[Citation] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(m in line for m in PROVENANCE_MARKERS):
            continue
        for m in _PATH_TOKEN.finditer(line):
            for expanded in expand_braces(m.group(0)):
                path, lineref = _clean(expanded)
                if len(path) < 2 or path in ("/", "//"):
                    continue
                if not _in_scope(path, repo):
                    continue
                # A bare scratch ROOT names no artifact -- it is prose about the rule,
                # not a citation of evidence. Anything UNDER the root still counts.
                if path.rstrip("/") in [r.rstrip("/") for r in EPHEMERAL_ROOTS]:
                    continue
                cites.append(Citation(raw=m.group(0), path=path, line=lineno,
                                      lineref=lineref, context=line.strip()))
    return cites


# --------------------------------------------------------------------- classification

def _waive(c: Citation) -> Citation:
    c.verdict, c.severity = "WAIVED_LOST", "warn"
    c.resolved = c.path
    c.hint = ("recorded as lost; keep the dead path verbatim so provenance survives, "
              "and do not invent a replacement")
    return c


def _is_scratch(cited: str, resolved: Path) -> bool:
    """True when either the literal citation or what it really points at is scratch.

    The realpath arm matters: a durable-looking symlink into `/tmp` is still one sweep
    from unverifiable, and a prefix guard that only reads the spelling of the citation
    is trivially defeated by a link. `storage.py:is_scratch_path` already resolved
    symlinks for exactly this reason; the two guards now agree.
    """
    if cited.startswith(EPHEMERAL_ROOTS):
        return True
    try:
        return str(resolved.resolve()).startswith(EPHEMERAL_ROOTS)
    except (OSError, ValueError):
        return False


def classify(c: Citation, repo: Path) -> Citation:
    """Resolvability, not committedness. See the module docstring for the 2026-08-03
    retarget: whether an artifact is in git, gitignored, or on some other durable mount
    is not this checker's business. Whether it is THERE is."""
    p = c.path
    resolved = Path(p) if p.startswith("/") else repo / p
    c.resolved = str(resolved)

    # WHERE THE CITATION REALLY LANDS -- read off the RESOLVED path, never off the
    # spelling, and by the identical test for relative and absolute citations. A
    # `data/<campaign>/x.json` symlinked into `/tmp` and a literal `/tmp/x.json` are the
    # same artifact one sweep from unverifiable, so they must reach the same verdict.
    #
    # This line carried a hole until 2026-08-04. It read
    #     inside_repo = (not absolute) or resolved.resolve().is_relative_to(...)
    # which is unconditionally True for every RELATIVE citation, short-circuiting before
    # anything was resolved. The scratch guard below is gated on `not inside_repo`, so it
    # was never consulted for a relative path -- 416 of the registry's 421 citations, and
    # precisely the form `_FIX_PLAYBOOK["EPHEMERAL"]` instructs people to migrate TO. One
    # symlink from a durable-looking `data/...` path defeated the guard on the dominant
    # citation form, silently and with an `OK` verdict.
    try:
        inside_repo = resolved.resolve().is_relative_to(repo.resolve())
    except (OSError, ValueError):
        inside_repo = False

    # Repository membership is decided BEFORE the scratch heuristic, and it is now
    # membership of the RESOLVED artifact, not of the string. The ephemeral roots are a
    # proxy for "not durable"; if a path really lands inside the repo being validated, the
    # proxy has nothing left to say. This keeps the checker correct for a checkout that
    # itself lives under a scratch root (which is how the test fixtures and any `/tmp`
    # worktree run) instead of quietly failing everything -- while NOT extending that
    # exemption to a path that merely looks repo-relative and points elsewhere.
    if not inside_repo and _is_scratch(p, resolved):
        if LOST_MARKER in c.context:
            return _waive(c)
        c.verdict, c.severity = "EPHEMERAL", "error"
        # Existence and size are read off `resolved`, NEVER off the raw citation. A
        # relative citation is relative to the REPO, not to the checker's cwd, so
        # `os.path.exists(p)` answered a question about the wrong file and reported a
        # live artifact as "already GONE" — which routes the reader to the demote-or-
        # re-measure remedy for evidence that is still sitting there, recoverable. That
        # branch was unreachable while relative citations skipped this verdict entirely;
        # closing the symlink hole above made it reachable, so it is fixed here.
        src = str(resolved)
        try:
            landing = str(resolved.resolve())
        except (OSError, ValueError):
            landing = src
        via = "" if landing == src else f" (it resolves to {landing})"
        c.hint = (
            f"evidence cited from a scratch directory{via}. "
            + (f"It still exists ({_size(src)}) — move it somewhere durable now: "
               f"`mkdir -p {repo}/data/<campaign> && cp -aL {src} "
               f"{repo}/data/<campaign>/`, then repoint this citation. Raw campaign "
               f"output stays LOCAL and gitignored (2026-08-03 ruling); carry the "
               f"README/SHA256SUMS and the distilled finding, not the substrate."
               if os.path.exists(src) else
               "It is already GONE. If it is a measurement result, the claim is "
               "unverifiable and must be re-measured or demoted; if it is a build "
               f"artifact, annotate the line `# {LOST_MARKER} (...) — recorded <date>`.")
        )
        return c

    if not resolved.exists():
        return _missing(c)

    # Deliberately NOT waivable by `ARTIFACT LOST`: the marker asserts the artifact is
    # gone, and this one demonstrably is not. Letting the wrong marker silence this would
    # make the waiver a general-purpose mute button, which is the one thing it must not be.
    if not os.access(resolved, os.R_OK):
        c.verdict, c.severity = "UNREADABLE", "error"
        c.hint = ("the path exists but cannot be read, so its hash still cannot be "
                  "recomputed — the citation is an assertion. Fix the permissions or "
                  "repoint at a copy that can actually be verified.")
        return c

    # Durable and resolvable. Deliberately no further grading: in-repo vs out-of-repo
    # and tracked vs gitignored are not durability signals (2026-08-03 ruling).
    c.verdict, c.severity = "OK", "ok"
    return c


def _missing(c: Citation) -> Citation:
    if LOST_MARKER in c.context:
        return _waive(c)
    c.verdict, c.severity = "MISSING", "error"
    c.hint = (
        "citation resolves nowhere, so its hash has nothing to check against. "
        "Locate the artifact (it may live in a sibling repo, or on a durable local root "
        "outside any repo — a bare `data/...` path is ambiguous) and cite it "
        "unambiguously, or mark the line "
        f"`# {LOST_MARKER} (...) — recorded <date>` if it is genuinely gone.")
    return c


def _size(p: str) -> str:
    try:
        if os.path.isdir(p):
            n = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(p) for f in fs)
        else:
            n = os.path.getsize(p)
    except OSError:
        return "size unknown"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GiB"


# ------------------------------------------------------------------- campaign hygiene

def check_campaign_docs(cites: list[Citation], repo: Path) -> list[dict]:
    """Every `data/<campaign>/` the registry cites should carry a README.md and a
    SHA256SUMS. Reported, not fatal by default: many campaign directories predate the
    rule.

    This survives the 2026-08-03 retarget and matters MORE under it, not less. Once the
    raw campaign output is gitignored and lives only on local disk, the README and the
    SHA256SUMS are the part that IS committed -- the distilled record of what was
    measured and the hashes that let a future reader tell whether the local substrate is
    still the thing the claim was made against. They are the reference; the tree is the
    material."""
    issues = []
    seen = set()
    for c in cites:
        m = re.match(r'^data/([^/]+)', c.path)
        if not m:
            continue
        camp = m.group(1)
        if camp in seen:
            continue
        seen.add(camp)
        d = repo / "data" / camp
        if not d.is_dir():
            continue
        missing = [f for f in ("README.md", "SHA256SUMS") if not (d / f).exists()]
        if missing:
            issues.append({"campaign": camp, "missing": missing,
                           "path": f"data/{camp}"})
    return sorted(issues, key=lambda x: x["campaign"])


# ------------------------------------------------------------------------------ run

def check(registry: Path, repo: Path) -> Result:
    text = Path(registry).read_text()
    cites = [classify(c, repo) for c in extract_citations(text, repo)]
    return Result(registry=str(registry), repo=str(repo), citations=cites,
                  campaign_issues=check_campaign_docs(cites, repo))


SEV_ORDER = {"error": 0, "warn": 1, "ok": 2}


def report(res: Result, show_ok: bool, fix_hint: bool) -> None:
    counts: dict[str, int] = {}
    for c in res.citations:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1

    print(f"evidence durability :: {res.registry}")
    print(f"repository          :: {res.repo}")
    print(f"citations in scope  :: {len(res.citations)}")
    print()
    for v in ("OK", "WAIVED_LOST", "EPHEMERAL", "MISSING", "UNREADABLE"):
        if v in counts:
            mark = {"OK": "  ok  ", "WAIVED_LOST": " warn "}.get(v, " FAIL ")
            print(f"  [{mark}] {v:<18} {counts[v]:>4}")
    print()

    shown = [c for c in res.citations if show_ok or c.severity != "ok"]
    shown.sort(key=lambda c: (SEV_ORDER[c.severity], c.line))
    for c in shown:
        tag = {"error": "FAIL", "warn": "warn", "ok": "ok"}[c.severity]
        ref = f"{c.path}{c.lineref}"
        print(f"  {tag:>4}  L{c.line:<6} {c.verdict:<17} {ref}")
        if c.severity != "ok" and c.hint:
            print(f"        -> {c.hint}")

    if res.campaign_issues:
        print()
        print(f"  campaign directories missing durability docs: {len(res.campaign_issues)}")
        for i in res.campaign_issues:
            print(f"    warn  {i['path']}  missing {', '.join(i['missing'])}")

    print()
    n_err, n_warn = len(res.errors), len(res.warnings)
    print(f"errors: {n_err}   warnings: {n_warn}")

    if fix_hint:
        print()
        print("=" * 72)
        print("FIX HINTS")
        print("=" * 72)
        if not n_err and not n_warn:
            print("Nothing to fix. Every evidence citation resolves on this host to a")
            print("readable artifact on a durable root.")
        for v, group in _group(res.errors + res.warnings):
            print()
            print(f"-- {v} ({len(group)}) " + "-" * max(0, 60 - len(v)))
            print(_FIX_PLAYBOOK.get(v, "").rstrip())
            for c in group[:12]:
                print(f"     L{c.line}: {c.path}")
            if len(group) > 12:
                print(f"     ... and {len(group) - 12} more")
        if res.campaign_issues:
            print()
            print("-- CAMPAIGN_DOCS " + "-" * 55)
            print(_FIX_PLAYBOOK["CAMPAIGN_DOCS"].rstrip())


def _group(cites):
    out: dict[str, list] = {}
    for c in cites:
        out.setdefault(c.verdict, []).append(c)
    return sorted(out.items(), key=lambda kv: SEV_ORDER.get(kv[1][0].severity, 9))


_FIX_PLAYBOOK = {
    "EPHEMERAL": """
  A ratified claim is pointing at a scratch directory. Do this, in order:

    1. mkdir -p data/<campaign>_<YYYYMMDD>          # local disk; see step 5 on git
    2. cp -a <scratch path> data/<campaign>_<YYYYMMDD>/        # COPY, never move --
                                                               # leave the original until
                                                               # the operator is satisfied
    3. sha256sum the source and the copy and compare them. Do not assume cp worked.
    4. Write data/<campaign>_<YYYYMMDD>/SHA256SUMS and a README.md saying what was
       measured, when, and which registry claim it backs.
       Pattern: data/vision_mmmu_cutover_20260731/
    5. Do NOT commit the raw campaign output. Per the 2026-08-03 operator ruling,
       research material reaches GitHub only as distilled knowledge and references in
       the wiki -- never as raw material, regardless of size. Gitignore the campaign
       tree; commit the README.md, SHA256SUMS and the distilled finding. The substrate
       stays on local disk, and THIS checker is what keeps it honest by proving the
       cited path still resolves.
    6. Repoint the citation, keeping any `:<line>` suffix and the surrounding YAML and
       comments untouched.""",

    "MISSING": """
  The citation resolves nowhere, so the ratification hash has nothing to verify against.

    * A bare `data/...` path is ambiguous -- check the sibling repos (epyc-root,
      epyc-orchestrator) and any durable local evidence root before concluding the
      artifact is gone, and cite it absolutely if that is where it lives.
    * If it is genuinely gone and was a measurement RESULT, the claim is unverifiable:
      re-measure it or demote the number to a prior. Do not fabricate a replacement path.
    * If it was a build artifact (a build tree, a compiled binary), nothing reproducible
      was lost. Keep the dead path VERBATIM so provenance survives and annotate the line:
        # ARTIFACT LOST (build tree, not a measurement result) — recorded <YYYY-MM-DD>

  Note what this verdict is NOT: it is not a complaint that the artifact is untracked.
  Gitignored, out-of-repo and on another mount are all fine. Absent is not.""",

    "UNREADABLE": """
  The artifact is on disk but cannot be opened, so its hash cannot be recomputed and the
  citation is an assertion in practice even though the path resolves. Fix the mode/owner,
  or repoint at a readable copy. Do not annotate this away with ARTIFACT LOST: the
  artifact is not lost, it is unverifiable, and those want different remedies.""",

    "WAIVED_LOST": """
  Recorded losses. No action: the artifact is gone, the line says so, and the dead path
  is kept on purpose so provenance survives. Do not delete these lines and do not
  substitute a plausible-looking replacement path.""",

    "CAMPAIGN_DOCS": """
  The durability clause requires each cited data/<campaign>/ to carry:
    README.md   what was measured, when, which registry claim it backs
    SHA256SUMS  one line per carried file
  Generate SHA256SUMS from the repo root so the paths in it are repo-relative:
    ( cd <repo> && find data/<campaign> -type f ! -name SHA256SUMS -print0 \\
        | sort -z | xargs -0 sha256sum > data/<campaign>/SHA256SUMS )""",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fail when registry evidence citations are not durable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit 0 when there are no errors. See the module docstring for rationale.")
    ap.add_argument("registry", nargs="?", default=str(DEFAULT_REGISTRY),
                    help="registry YAML (default: orchestration/model_registry.yaml)")
    ap.add_argument("--repo", default=str(REPO_ROOT),
                    help="repository root citations must resolve inside")
    ap.add_argument("--fix-hint", action="store_true",
                    help="print a remediation playbook per failing verdict")
    ap.add_argument("--show-ok", action="store_true", help="list passing citations too")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable output")
    ap.add_argument("-W", "--warnings-as-errors", action="store_true",
                    help="treat warnings (recorded ARTIFACT LOST waivers) as failures")
    ap.add_argument("--require-campaign-docs", action="store_true",
                    help="fail when a cited data/<campaign>/ lacks README.md or SHA256SUMS")
    a = ap.parse_args(argv)

    registry = Path(a.registry)
    if not registry.exists():
        print(f"registry not found: {registry}", file=sys.stderr)
        return 2

    res = check(registry, Path(a.repo))

    if a.as_json:
        print(json.dumps({
            "registry": res.registry,
            "repo": res.repo,
            "errors": len(res.errors),
            "warnings": len(res.warnings),
            "citations": [asdict(c) for c in res.citations],
            "campaign_issues": res.campaign_issues,
        }, indent=1))
    else:
        report(res, a.show_ok, a.fix_hint)

    failed = bool(res.errors)
    if a.warnings_as_errors and res.warnings:
        failed = True
    if a.require_campaign_docs and res.campaign_issues:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
