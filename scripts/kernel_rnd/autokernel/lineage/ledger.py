"""Keep ledger, C-next fold list, presence-gate manifest, Spec-D re-check rows (AK P4).

WHY. Until 2026-11-21 the management session mapped "each keep, its commit, whether it is
in C16, and the action" by hand, through a subagent, every composite. This module makes
that mapping a pure function of git history plus the read-only ledgers:

* the champion branch (`champion.CANONICAL_BRANCH`) since the production anchor;
* the C-series fold branches (`llama.cpp-experimental-c<N>-*`);
* the DO-NOT-FOLD prose in `handoffs/active/{autokernel-akx-program,
  kernel-champion-and-promotion}.md` (read only, never edited);
* `docs/reference/kernel-manifests/*-manifest.md` (presence gates, EXCL rows).

A FEATURE is the set of commits that carry one tag (`AKX-ALL-24`, `UP-FAROWS-HIP`,
`AK-GAP-1-SAMPLE`, upstream `#30408`, ...), deduplicated by `git patch-id --stable`, so a
cherry-picked copy on a second branch is the same feature. One feature folds as ONE commit
together with its test (V6R-4b / OP-58 forward-port rule). A feature with no test is
REFUSED (`FoldRefused`) -- the DS41-C116 kernel-preservation rule
(`loop/kernel_coverage.py`: the `tests/` inventory is part of the source layer, so a test
has to exist to travel). The git repo is read only: `log`, `cherry`, `patch-id`,
`for-each-ref`, `merge-base`, `rev-parse`. Nothing here checks out, builds or publishes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence

LEDGER_SCHEMA = "epyc.autokernel.keep_ledger.v1"
FOLD_SCHEMA = "epyc.autokernel.fold_list.v1"
PRESENCE_SCHEMA = "epyc.autokernel.presence_gate.v1"
ROW_SCHEMA = "epyc.row.v3"

PRODUCTION_ANCHOR = "ffc1bac82eeca6f9099e1ccd9ba49703c460a115"  # production-consolidated-v10
CHAMPION_BRANCH = "ak/champion/llama-cpp-ffc1bac82eec"          # == champion.CANONICAL_BRANCH
FOLD_GLOB = "refs/heads/llama.cpp-experimental-c*"
_FOLD_NAME = re.compile(r"^llama\.cpp-experimental-c(\d+)-(.+)$")
# The composite of a C-series is the branch with no feature slug: `-c17-20261121` or
# `-c16-composite-20261118`; everything else is a side branch feeding it.
_COMPOSITE = re.compile(r"^llama\.cpp-experimental-c(\d+)-(?:composite-)?(\d{8})$")

TAG = re.compile(r"(?<![\w-])([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+(?:\.[A-Z0-9]+(?:-[A-Z0-9]+)*)*)(?![\w-])")
_PAREN = re.compile(r"\(([^()]*)\)")
_UPSTREAM = re.compile(r"\(#(\d+)\)")
_SHA = re.compile(r"`([0-9a-f]{7,40})`")
_ROW = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+)*-\d{8}-r\d+)\b")
_KNOB = re.compile(r"\b((?:GGML|LLAMA)_[A-Z0-9_]{3,})\b")
_DNF = re.compile(r"DO NOT FOLD|DO-NOT-FOLD")
_TEST_NAME = re.compile(r"\b(tests/[\w./-]+|test-[\w-]+)")

Git = Callable[..., str]
DROP_VERDICTS = {"DROP", "FAIL", "WITHDRAWN", "REGRESSION", "NO-KEEP"}


class FoldRefused(ValueError):
    """A fold entry that breaks the DS41-C116 / V6R-4b contract."""


def git_runner(repo: Path | str) -> Git:
    """Read-only git: refuses any subcommand outside the allow-list."""
    allowed = {"log", "cherry", "patch-id", "for-each-ref", "merge-base", "rev-parse"}

    def run(*args: str, input: str | None = None) -> str:
        if args[0] not in allowed:
            raise PermissionError(f"lineage is read-only: git {args[0]} refused")
        proc = subprocess.run(["git", "-C", str(repo), *args], input=input, text=True,
                              capture_output=True, check=False)
        if proc.returncode != 0 and args[0] not in {"merge-base", "rev-parse"}:
            raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
        return proc.stdout if proc.returncode == 0 else ""
    return run


# ------------------------------------------------------------------ git reading

@dataclass
class Commit:
    sha: str
    time: int
    subject: str
    body: str
    files: list[str]
    patch_id: str = ""

    @property
    def touches_tests(self) -> bool:
        return any(f.startswith("tests/") for f in self.files)


def read_commits(git: Git, rev_range: str) -> list[Commit]:
    """Oldest first, with files and stable patch-ids."""
    out = git("log", "--reverse", "--no-merges",
              "--format=%x1e%H%x1f%ct%x1f%s%x1f%b%x1d", "--name-only", rev_range)
    commits = []
    for chunk in out.split("\x1e")[1:]:
        head, _, files = chunk.partition("\x1d")
        sha, ct, subject, body = (head.split("\x1f") + ["", "", "", ""])[:4]
        commits.append(Commit(sha, int(ct or 0), subject.strip(), body.strip(),
                              [f for f in files.split("\n") if f.strip()]))
    if commits:
        pids = patch_ids(git, rev_range)
        for c in commits:
            c.patch_id = pids.get(c.sha, "empty-" + c.sha)
    return commits


def patch_ids(git: Git, rev_range: str) -> dict[str, str]:
    diff = git("log", "--no-merges", "-p", "--format=commit %H", rev_range)
    if not diff.strip():
        return {}
    ids = {}
    for line in git("patch-id", "--stable", input=diff).splitlines():
        pid, _, sha = line.partition(" ")
        ids[sha.strip()] = pid
    return ids


def resolve_patch_id(git: Git, sha: str) -> str | None:
    full = git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}").strip()
    if not full:
        return None
    return patch_ids(git, f"{full}^!").get(full)


def fold_branches(git: Git) -> list[str]:
    names = git("for-each-ref", "--format=%(refname:short)", FOLD_GLOB).split()
    return sorted(n for n in names if _FOLD_NAME.match(n))


def current_composite(branches: Iterable[str]) -> tuple[str | None, int]:
    """Highest-numbered composite branch (ties: latest date) and its C number."""
    best = None
    for b in branches:
        m = _COMPOSITE.match(b)
        if m and (best is None or (int(m[1]), m[2]) > best[0]):
            best = ((int(m[1]), m[2]), b)
    return (best[1], best[0][0]) if best else (None, 0)


# ------------------------------------------------------------------ prose / manifest

def feature_tag(subject: str, body: str = "") -> str | None:
    up = _UPSTREAM.search(subject)
    for paren in _PAREN.findall(subject):
        m = TAG.search(paren)
        if m:
            return m[1]
    m = TAG.search(subject.split(":", 1)[-1]) or TAG.search(subject)
    if m:
        return m[1]
    if up:
        return f"UPSTREAM-{up[1]}"
    return None


def parse_dnf(texts: Mapping[str, str]) -> list[dict[str, Any]]:
    """Every backticked sha on a line that says DO NOT FOLD / DO-NOT-FOLD (upper case,
    so the ledger header and the fixture rule are read, a lower-case "do not fold now;
    hold for C14" deferral is not)."""
    found = {}
    for name, text in texts.items():
        for no, line in enumerate(text.splitlines(), 1):
            if not _DNF.search(line):
                continue
            for sha in _SHA.findall(line):
                found.setdefault(sha, {"sha": sha, "source": f"{name}:{no}",
                                       "line": line.strip()[:240]})
    return [found[k] for k in sorted(found)]


def parse_manifest(text: str, name: str = "manifest") -> list[dict[str, Any]]:
    rows = []
    for no, line in enumerate(text.splitlines(), 1):
        if not re.match(r"^\|\s*[A-Z][A-Z0-9]*-\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        cells += [""] * (8 - len(cells))
        status = cells[4]
        word = re.sub(r"[*`\s]", "", status)[:5].upper()
        rows.append({"id": cells[0], "what": cells[1], "kind": cells[2],
                     "source": cells[3], "status": status,
                     "state": {"IN": "in", "PEND": "pending", "EXCL": "excluded",
                               "SUPER": "superseded"}.get(
                                   next((k for k in ("SUPER", "EXCL", "PEND", "IN")
                                         if word.startswith(k)), ""), "other"),
                     "presence_gate": cells[6], "owner": cells[7],
                     "shas": sorted(set(re.findall(r"\b[0-9a-f]{9,40}\b", cells[3] + " " + status))),
                     "where": f"{name}:{no}"})
    return rows


# ------------------------------------------------------------------ ledger

@dataclass
class Feature:
    tag: str
    commits: list[dict[str, Any]] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)
    hypothesis: str = ""
    keep_sha: str = ""
    gate_results: list[dict[str, Any]] = field(default_factory=list)
    knobs: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    manifest_ids: list[str] = field(default_factory=list)
    manifest_tests: list[str] = field(default_factory=list)
    in_champion: bool = False
    in_fold: bool = False
    do_not_fold: list[str] = field(default_factory=list)
    first_time: int = 0

    @property
    def has_test(self) -> bool:
        return bool(self.test_files or self.manifest_tests)

    def to_json(self) -> dict[str, Any]:
        body = asdict(self)
        body["has_test"] = self.has_test
        return body


def build_ledger(git: Git, *, champion: str = CHAMPION_BRANCH,
                 production: str = PRODUCTION_ANCHOR,
                 branches: Sequence[str] | None = None,
                 current_fold: str | None = None,
                 dnf_texts: Mapping[str, str] | None = None,
                 manifests: Mapping[str, str] | None = None,
                 verdicts: Mapping[str, str] | None = None,
                 window: int | None = 3) -> dict[str, Any]:
    """Pure function of the repo and the read-only ledgers -> keep ledger JSON."""
    branches = list(fold_branches(git) if branches is None else branches)
    if current_fold is None:
        current_fold, cnum = current_composite(branches)
    else:
        m = _FOLD_NAME.match(current_fold)
        cnum = int(m[1]) if m else 0
    if window is not None and cnum:
        # Only the live C-series feeds C-next; older side branches were either folded
        # or deliberately dropped at their own composite.
        branches = [b for b in branches
                    if (m := _FOLD_NAME.match(b)) is None or int(m[1]) >= cnum - window]
    verdicts = dict(verdicts or {})

    champ = read_commits(git, f"{production}..{champion}")
    champ_pids = {c.patch_id for c in champ}
    fold_pids: set[str] = set()
    per_branch: dict[str, list[Commit]] = {"(champion)": champ}
    for b in branches:
        base = git("merge-base", champion, b).strip() or production
        per_branch[b] = read_commits(git, f"{base}..{b}")
        if b == current_fold:
            fold_pids = {c.patch_id for c in per_branch[b]}

    dnf = parse_dnf(dnf_texts or {})
    dnf_pid = {}
    for d in dnf:
        pid = resolve_patch_id(git, d["sha"])
        dnf_pid[d["sha"]] = pid
    mrows = []
    for name, text in (manifests or {}).items():
        mrows += parse_manifest(text, name)
    excluded_shas = {s: r["id"] for r in mrows if r["state"] == "excluded" for s in r["shas"]}

    features: dict[str, Feature] = {}
    seen: dict[str, str] = {}                          # patch-id -> tag
    untagged: dict[str, str] = {}                      # subject -> tag
    for bname, commits in per_branch.items():
        for c in commits:
            # Untagged commits are identified by subject, so a rebased copy (new patch-id,
            # same change) on a second side branch is still one feature.
            tag = (seen.get(c.patch_id) or feature_tag(c.subject, c.body)
                   or untagged.setdefault(c.subject, f"COMMIT-{c.sha[:9]}"))
            if tag in features and c.patch_id not in seen and tag.startswith("COMMIT-") \
                    and any(x["subject"] == c.subject for x in features[tag].commits):
                seen[c.patch_id] = tag              # rebased duplicate: same feature, skip
                if bname not in features[tag].branches:
                    features[tag].branches.append(bname)
                continue
            f = features.setdefault(tag, Feature(tag=tag, first_time=c.time))
            if bname not in f.branches:
                f.branches.append(bname)
            if c.patch_id in seen:
                continue
            seen[c.patch_id] = tag
            f.first_time = min(f.first_time, c.time)
            f.commits.append({"sha": c.sha, "subject": c.subject, "patch_id": c.patch_id,
                              "time": c.time, "tests": c.touches_tests,
                              "in_champion": c.patch_id in champ_pids,
                              "in_fold": c.patch_id in fold_pids})
            f.test_files += [x for x in c.files if x.startswith("tests/") and x not in f.test_files]
            f.knobs += [k for k in _KNOB.findall(c.subject + " " + c.body) if k not in f.knobs]
            for row in _ROW.findall(c.subject + " " + c.body):
                if all(g["row"] != row for g in f.gate_results):
                    f.gate_results.append({"row": row, "verdict": verdicts.get(row, "cited")})
            for d in dnf:
                if (c.sha.startswith(d["sha"]) or dnf_pid.get(d["sha"]) == c.patch_id) \
                        and d["source"] not in f.do_not_fold:
                    f.do_not_fold.append(d["source"])
            for s, mid in excluded_shas.items():
                if c.sha.startswith(s) and f"manifest {mid} EXCL" not in f.do_not_fold:
                    f.do_not_fold.append(f"manifest {mid} EXCL")
            for r in mrows:
                if any(c.sha.startswith(s) for s in r["shas"]) and r["id"] not in f.manifest_ids:
                    f.manifest_ids.append(r["id"])
                    f.manifest_tests += [t for t in _TEST_NAME.findall(r["presence_gate"])
                                         if t not in f.manifest_tests]
    for f in features.values():
        f.commits.sort(key=lambda c: c["time"])
        real = [c for c in f.commits if not c["tests"]] or f.commits
        f.hypothesis = real[0]["subject"] if real else ""
        f.keep_sha = f.commits[-1]["sha"] if f.commits else ""
        f.in_champion = bool(f.commits) and all(c["in_champion"] for c in f.commits)
        f.in_fold = bool(f.commits) and all(c["in_fold"] or c["in_champion"] for c in f.commits) \
            and not f.in_champion
    return {"schema": LEDGER_SCHEMA, "champion": champion, "production": production,
            "current_fold": current_fold, "current_c": cnum, "next_c": cnum + 1,
            "branches": branches, "do_not_fold": dnf, "manifest_rows": mrows,
            "features": [features[k].to_json() for k in sorted(features)]}


def _feature(body: Mapping[str, Any]) -> Feature:
    data = {k: v for k, v in body.items() if k in Feature.__dataclass_fields__}
    return Feature(**data)


# ------------------------------------------------------------------ fold list

def fold_entry(feature: Feature | Mapping[str, Any]) -> dict[str, Any]:
    """One fold commit = one feature + its test. Refuses DS41-C116 / DNF violations."""
    f = feature if isinstance(feature, Feature) else _feature(feature)
    if f.do_not_fold:
        raise FoldRefused(f"{f.tag}: DO-NOT-FOLD ({'; '.join(f.do_not_fold)})")
    dropped = [g for g in f.gate_results if str(g.get("verdict", "")).upper() in DROP_VERDICTS]
    if dropped:
        raise FoldRefused(f"{f.tag}: gate verdict says drop ({', '.join(g['row'] + '=' + g['verdict'] for g in dropped)})")
    if not f.has_test:
        raise FoldRefused(f"{f.tag}: no test travels with this feature (DS41-C116 gate, "
                          "kernel_coverage.py; V6R-4b: a feature's test travels with it)")
    todo = [c for c in f.commits if not c.get("in_champion")]
    if not todo:
        raise FoldRefused(f"{f.tag}: nothing to fold (already in champion)")
    tests = f.test_files or f.manifest_tests
    return {"tag": f.tag, "hypothesis": f.hypothesis,
            "cherry_pick": [c["sha"] for c in todo], "squash_into_one_commit": True,
            "commit_message": f"{f.hypothesis}\n\nFold of {f.tag}: "
                              f"{len(todo)} source commit(s) squashed (V6R-4b).\n"
                              f"Tests: {', '.join(tests)}",
            "tests": tests, "knobs": f.knobs, "gate_results": f.gate_results,
            "carried_from_current_fold": f.in_fold}


def fold_list(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """C-next = the current composite's features (carried, in its order) then every
    unfolded keep oldest-first. Untested / DNF features land in `refused`, never in the
    list."""
    feats = [_feature(b) for b in ledger["features"]]
    todo = [f for f in feats if not f.in_champion]
    todo.sort(key=lambda f: (not f.in_fold, f.first_time, f.tag))
    entries, refused = [], []
    for f in todo:
        try:
            entries.append(fold_entry(f))
        except FoldRefused as exc:
            refused.append({"tag": f.tag, "reason": str(exc),
                            "branches": f.branches, "carried": f.in_fold})
    return {"schema": FOLD_SCHEMA, "candidate": f"C{ledger['next_c']}",
            "base": ledger["champion"], "current_fold": ledger["current_fold"],
            "rule": "one feature per commit with its test (V6R-4b); untested refused (DS41-C116)",
            "entries": entries, "refused": refused}


# ------------------------------------------------------------------ presence gate

def presence_manifest(ledger: Mapping[str, Any], folds: Mapping[str, Any]) -> dict[str, Any]:
    """Every feature the candidate must carry: champion keeps + fold entries, plus every
    manifest row IN the champion. Each names its gates; a gateless entry is a defect."""
    feats = {b["tag"]: b for b in ledger["features"]}
    want = [t for t, b in feats.items() if b["in_champion"]] + [e["tag"] for e in folds["entries"]]
    entries, defects = [], []
    for tag in want:
        b = feats[tag]
        gates = [{"kind": "patch-equivalent", "patch_ids": [c["patch_id"] for c in b["commits"]],
                  "shas": [c["sha"] for c in b["commits"]]}]
        gates += [{"kind": "test", "file": t} for t in (b["test_files"] or b["manifest_tests"])]
        gates += [{"kind": "strings", "needle": k} for k in b["knobs"]]
        entries.append({"id": tag, "source": "ledger", "manifest_ids": b["manifest_ids"],
                        "gates": gates})
    for r in ledger.get("manifest_rows", []):
        if r["state"] != "in":
            continue
        if not r["presence_gate"] or r["presence_gate"] in {"-", "none"}:
            defects.append(f"{r['id']}: manifest row IN champion with no presence gate ({r['where']})")
        entries.append({"id": r["id"], "source": r["where"],
                        "gates": [{"kind": "manifest", "text": r["presence_gate"]}]})
    for e in entries:
        if e["source"] == "ledger" and not any(g["kind"] != "patch-equivalent" for g in e["gates"]) \
                and feats[e["id"]]["in_champion"] is False:
            defects.append(f"{e['id']}: fold entry has only an ancestry gate")
    return {"schema": PRESENCE_SCHEMA, "candidate": folds["candidate"],
            "base": ledger["champion"], "entries": entries, "defects": defects}


def check_presence(git: Git, manifest: Mapping[str, Any], candidate_ref: str, *,
                   production: str = PRODUCTION_ANCHOR) -> dict[str, Any]:
    """Read-only: does `candidate_ref` carry every patch-equivalent gate?"""
    have = set(patch_ids(git, f"{production}..{candidate_ref}").values())
    missing = []
    for e in manifest["entries"]:
        for g in e["gates"]:
            if g["kind"] == "patch-equivalent":
                lost = [s for s, p in zip(g["shas"], g["patch_ids"]) if p not in have]
                if lost:
                    missing.append({"id": e["id"], "missing": lost})
    return {"candidate_ref": candidate_ref, "passed": not missing, "missing": missing}


# ------------------------------------------------------------------ Spec-D rows

def specd_rows(folds: Mapping[str, Any], *, date: str, owner: str = "autokernel",
               dnf_shas: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Spec D (akx-program §9) re-check rows for the composed candidate. STAGED ONLY:
    launcher and estimate are left for the owner; never published from here."""
    c = folds["candidate"].lower()
    lineage = f"specd-{c}"
    common = {"schema": ROW_SCHEMA, "owner": owner, "lineage": lineage,
              "launcher": None, "estimate_s": None, "staged_only": True,
              "handoff_task": "autokernel-akx-program.md:Spec D (composition / FOLD re-check)"}

    def row(suffix, lane, regions, rule, **extra):
        return {**common, "id": f"{lineage}-{suffix}-{date}-r1", "lane": lane,
                "regions": regions, "decision_rule": rule, **extra}
    rows = [
        row("census-footprint", "cpu", "any-cpu-quarter",
            f"Spec D steps 1-2 on {folds['candidate']}: regenerate the base census on the composed "
            "candidate and screen the WHOLE range vs production_ref (no per-keep INERT reuse). "
            "Step 6 fixture: any DO-NOT-FOLD sha present must screen TOUCHED and flag default_after=ON.",
            dnf_fixture=list(dnf_shas)),
        row("nonregression-cpu", "cpu", "any-cpu-quarter",
            "Spec D step 3: full-workload non-regression on every TOUCHED workload, confirmation-stratum "
            "fresh launches vs production_ref at production-optimal recipe. CPU pass + GPU missing = pending."),
        row("nonregression-gpu", "gpu", "gpu",
            "Spec D step 3 (GPU leg), same rule; host threads 88-95 held."),
    ]
    for e in folds["entries"]:
        slug = re.sub(r"[^a-z0-9]+", "-", e["tag"].lower()).strip("-")
        rows.append(row(f"loo-{slug}", "cpu", "any-cpu-quarter",
                        f"Spec D steps 4-5: LOO arm for {e['tag']} per TOUCHED workload "
                        "(source_loo.execute_surface) + AK-GH-1 count beside it.",
                        fold_tag=e["tag"]))
    return rows


def write_staged(rows: Sequence[Mapping[str, Any]], staging: Path) -> list[Path]:
    staging.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in rows:
        p = staging / f"{r['id']}.json"
        p.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def fold_markdown(folds: Mapping[str, Any]) -> str:
    out = [f"# {folds['candidate']} fold list (generated, do not hand-edit)", "",
           f"base `{folds['base']}`; current fold `{folds['current_fold']}`; {folds['rule']}", "",
           "| # | tag | carried | commits | tests | hypothesis |", "|---|---|---|---|---|---|"]
    for i, e in enumerate(folds["entries"], 1):
        out.append(f"| {i} | {e['tag']} | {'yes' if e['carried_from_current_fold'] else 'new'} | "
                   f"{' '.join(s[:9] for s in e['cherry_pick'])} | {', '.join(e['tests'])} | "
                   f"{e['hypothesis'].replace('|', '/')} |")
    out += ["", "## Refused", ""]
    out += [f"- **{r['tag']}** ({'carried' if r['carried'] else 'new'}): {r['reason']}"
            for r in folds["refused"]] or ["- none"]
    return "\n".join(out) + "\n"
