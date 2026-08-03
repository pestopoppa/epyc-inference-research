#!/usr/bin/env python3
"""Fail when the model registry cites evidence that is not DURABLE.

WHY THIS EXISTS
---------------
`MEASUREMENT.md:139-141` (consolidated apply-time ratification) requires evidence
HASHES in the apply-time bundle, but says nothing about where the evidence must
LIVE. That is a real hole, not a pedantic one: a bundle can hash a file sitting in
`/tmp`, be signed by the operator, and become unverifiable the moment that file is
swept. The hash proves nothing once there is no artifact left to check it against —
it degrades from a verification into an assertion, silently, with no event to notice.

On 2026-08-02 the master registry was found citing 157 distinct paths under
`/mnt/raid0/llm/tmp/` as the evidence behind ratified, production-affecting claims,
including the MMMU-250 result that had gated a live vision model cutover. Nothing had
been lost; the whole set was one cleanup away. 4.0 MiB carried nearly all of it.

This validator closes the gap the constitution left open. Note that `MEASUREMENT.md`
is HUMAN-AMENDMENT-ONLY (the measurement trust boundary), so this script enforces a
CONVENTION and does not amend the constitution. The convention was subsequently
ratified into `MEASUREMENT.md` by the operator as *"Evidence must be DURABLE, not
merely hashed"*, which names this file as its enforcer; the script remains the
mechanism, never the authority.

WHAT COUNTS AS A CITATION
-------------------------
Whitelist, not blacklist. A path-shaped token in the registry is in scope only if it
resolves under a root we care about:

  * an EPHEMERAL root (`/tmp`, `/var/tmp`, `/dev/shm`, `/run`, `/mnt/raid0/llm/tmp`)
    -- always in scope, and always an error;
  * THIS repository, cited relatively (`data/...`) or absolutely;
  * a known SIBLING repository in the same coordinated workspace.

Everything else -- model weights under `/mnt/raid0/llm/models`, kernel build trees,
`/opt/rocm`, `/sys`, container mount targets -- is out of scope by construction. Those
are inputs and system paths, not measurement results, and `validate_model_registry.py`
already checks that weights exist. Defining scope as a whitelist matters: a blacklist
grows an exemption every time it is inconvenient, and the one exemption that must never
be grantable is the scratch root.

VERDICTS
--------
  OK                repo-relative, inside the repo, exists
  ABSOLUTE_IN_REPO  WARN  -- inside the repo but written as an absolute path
  SIBLING_REPO      WARN  -- durable and version-controlled, but not the mandated
                             `epyc-inference-research/data/<campaign>/` home
  WAIVED_LOST       WARN  -- artifact is gone and the line says so, verbatim, with an
                             `ARTIFACT LOST` marker. A recorded loss, not a silent one.
  EPHEMERAL         ERROR -- cited from a scratch directory. One cleanup from unverifiable.
  MISSING           ERROR -- resolves nowhere. The hash has nothing to check.
  OUTSIDE_REPO      ERROR -- exists, but in no repository, so nothing versions it.

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

# Sibling repositories of the same coordinated workspace. Evidence here is
# version-controlled -- durable -- but not in the mandated location, so it warns.
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
    # An absolute path in no known repo can still be an evidence tree somebody parked
    # on a random mount -- the OUTSIDE_REPO case. Require the artifact-tree segment to
    # sit below the filesystem root so container mount targets like `/data/indices`,
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


def classify(c: Citation, repo: Path) -> Citation:
    p = c.path
    absolute = p.startswith("/")
    resolved = Path(p) if absolute else repo / p
    c.resolved = str(resolved)

    try:
        inside_repo = (not absolute) or resolved.resolve().is_relative_to(repo.resolve())
    except (OSError, ValueError):
        inside_repo = False

    # Repository membership is decided BEFORE the scratch heuristic. The ephemeral roots
    # are a proxy for "not in a repository"; if a path is demonstrably inside the repo
    # being validated, the proxy has nothing left to say. This also keeps the checker
    # correct for a relocated checkout instead of quietly failing everything.
    if inside_repo:
        if not resolved.exists():
            return _missing(c)
        if absolute:
            c.verdict, c.severity = "ABSOLUTE_IN_REPO", "warn"
            rel = os.path.relpath(resolved.resolve(), repo.resolve())
            c.hint = (f"inside the repo but written absolutely; an absolute path breaks "
                      f"for any other checkout. Rewrite as `{rel}`.")
        else:
            c.verdict, c.severity = "OK", "ok"
        return c

    if p.startswith(EPHEMERAL_ROOTS):
        if LOST_MARKER in c.context:
            return _waive(c)
        c.verdict, c.severity = "EPHEMERAL", "error"
        c.hint = (
            "evidence cited from a scratch directory. "
            + (f"It still exists ({_size(p)}) — copy it now: "
               f"`cp -a {p} {repo}/data/<campaign>/` and repoint this citation at "
               f"`data/<campaign>/...`."
               if os.path.exists(p) else
               "It is already GONE. If it is a measurement result, the claim is "
               "unverifiable and must be re-measured or demoted; if it is a build "
               f"artifact, annotate the line `# {LOST_MARKER} (...) — recorded <date>`.")
        )
        return c

    if not resolved.exists():
        return _missing(c)

    for sib in SIBLING_REPO_ROOTS:
        if str(resolved).startswith(sib + "/"):
            c.verdict, c.severity = "SIBLING_REPO", "warn"
            c.hint = (f"durable (version-controlled in {sib}) but not in the mandated "
                      f"`epyc-inference-research/data/<campaign>/` home. Acceptable for "
                      f"evidence genuinely owned by that repo; a bare relative path "
                      f"pointing there is NOT — spell it absolutely so it is unambiguous.")
            return c

    c.verdict, c.severity = "OUTSIDE_REPO", "error"
    c.hint = ("exists, but under no repository — nothing versions it and nothing stops "
              f"it being deleted. Copy into `{repo}/data/<campaign>/` and repoint.")
    return c


def _missing(c: Citation) -> Citation:
    if LOST_MARKER in c.context:
        return _waive(c)
    c.verdict, c.severity = "MISSING", "error"
    c.hint = (
        "citation resolves nowhere, so its hash has nothing to check against. "
        "Locate the artifact (it may live in a sibling repo — a bare `data/...` path is "
        "ambiguous across repos) and cite it unambiguously, or mark the line "
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
    SHA256SUMS, per the constitution's durability clause. Reported, not fatal by
    default: many campaign directories predate the rule."""
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
    for v in ("OK", "ABSOLUTE_IN_REPO", "SIBLING_REPO", "WAIVED_LOST",
              "EPHEMERAL", "MISSING", "OUTSIDE_REPO"):
        if v in counts:
            mark = {"OK": "  ok  ", "ABSOLUTE_IN_REPO": " warn ", "SIBLING_REPO": " warn ",
                    "WAIVED_LOST": " warn "}.get(v, " FAIL ")
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
            print("Nothing to fix. Every evidence citation is inside the repository")
            print("and resolves to an artifact that exists.")
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

    1. mkdir -p data/<campaign>_<YYYYMMDD>
    2. cp -a <scratch path> data/<campaign>_<YYYYMMDD>/        # COPY, never move --
                                                               # leave the original until
                                                               # the operator is satisfied
    3. sha256sum the source and the copy and compare them. Do not assume cp worked.
    4. Write data/<campaign>_<YYYYMMDD>/SHA256SUMS and a README.md saying what was
       measured, when, and which registry claim it backs.
       Pattern: data/vision_mmmu_cutover_20260731/
    5. Repoint the citation to the repo-relative `data/...` path, keeping any `:<line>`
       suffix and the surrounding YAML and comments untouched.

  If the artifact is >5 MiB or matches a .gitignore rule (*.gguf, *.bin, *.safetensors),
  do NOT carry the blob. Write a `<name>.sha256` sidecar recording sha256, size and the
  scratch origin, point the citation at the sidecar, and say in the citation that the
  blob is intentionally not carried.""",

    "MISSING": """
  The citation resolves nowhere, so the ratification hash has nothing to verify against.

    * A bare `data/...` path is ambiguous across repos -- check the sibling repos
      (epyc-root, epyc-orchestrator) before concluding the artifact is gone, and cite
      it absolutely if that is where it lives.
    * If it is genuinely gone and was a measurement RESULT, the claim is unverifiable:
      re-measure it or demote the number to a prior. Do not fabricate a replacement path.
    * If it was a build artifact (a build tree, a compiled binary), nothing reproducible
      was lost. Keep the dead path VERBATIM so provenance survives and annotate the line:
        # ARTIFACT LOST (build tree, not a measurement result) — recorded <YYYY-MM-DD>""",

    "OUTSIDE_REPO": """
  The artifact exists but lives in no repository, so nothing versions it and nothing
  prevents its deletion. Copy it into data/<campaign>/ and repoint, following the
  EPHEMERAL playbook above.""",

    "ABSOLUTE_IN_REPO": """
  Inside the repo, but written as an absolute path -- it breaks in any other checkout
  and hides the fact that the evidence is in-tree. Rewrite as repo-relative:
    /mnt/raid0/llm/epyc-inference-research/data/x/y.json  ->  data/x/y.json""",

    "SIBLING_REPO": """
  Durable -- version-controlled in a sibling repo -- but not in the mandated
  epyc-inference-research/data/<campaign>/ home. Acceptable when the evidence is
  genuinely owned by that repo. Spell it as an absolute path so it cannot be misread as
  a path in THIS repo that happens not to exist.""",

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
                    help="treat warnings (absolute, sibling-repo, waived) as failures")
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
