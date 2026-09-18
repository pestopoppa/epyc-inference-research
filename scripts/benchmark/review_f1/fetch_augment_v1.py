#!/usr/bin/env python3
"""Fetch + pin the Augment-v1 code-review golden set and its 50 PR diffs (EV-13b).

WHY A FETCH SCRIPT AND NOT COMMITTED DATA
    Upstream ``github.com/ai-code-review-evaluations/golden_comments`` carries NO
    licence (no LICENSE file; GitHub ``license: null``) => default copyright, no
    redistribution grant. This repository is PUBLIC, so the golden comments are
    NOT committed. Only the pin manifest (URLs, commit SHAs, sha256 digests,
    PR numbers, counts — facts, not content) is committed; the data is fetched
    into the git-ignored ``data/external/review_f1/augment_v1/`` tree and
    verified byte-for-byte against the manifest.

LAYOUT (under --out-root, default data/external/review_f1/augment_v1)
    upstream/<bench>.json                raw upstream files (pinned commit)
    raw/<bench>/pr-<n>.json              one Augment-v1 PR record per PR (+diff_path)
    diffs/<bench>/pr-<n>.diff            GitHub PR diff of the augment-<bench> fork
    golden_set.json                      EV-13a assembled set (assemble_golden_set.py)
    anchor_report.json                   comment -> diff identifier anchoring check

    Harness: ``--golden <root>/golden_set.json --context-dir <root>``
    (``pr_ref.diff_path`` is relative to the root).

MODES
    default           fetch, verify every sha256 against the committed manifest
                      (hard fail on mismatch), assemble, anchor-check.
    --write-manifest  (re)generate the manifest from what was fetched. Only for
                      the initial pin or a deliberate re-pin; review the diff.

Network: raw.githubusercontent.com + the ``gh`` CLI (PR list / PR diff). No
inference, no model servers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = PKG_DIR.parents[2]
sys.path.insert(0, str(PKG_DIR))
from assemble_golden_set import assemble  # noqa: E402

UPSTREAM_REPO = "ai-code-review-evaluations/golden_comments"
UPSTREAM_COMMIT = "3f2c8ab794b4ed71c099cbf0a7670555e90906dc"
FORK_OWNER = "ai-code-review-evaluations"
FORK_PREFIX = "augment-"  # the PRs are identical across the per-tool forks
BENCHES = ["cal_dot_com", "discourse", "grafana", "keycloak", "sentry"]
DEFAULT_MANIFEST = RESEARCH_ROOT / "data" / "review_f1" / "augment_v1_manifest.json"
DEFAULT_OUT = RESEARCH_ROOT / "data" / "external" / "review_f1" / "augment_v1"

# identifier-ish tokens: backticked spans, dotted/underscored names, CamelCase, call()
_TOKEN_RES = [
    re.compile(r"`([^`]{3,80})`"),
    re.compile(r"\b([A-Za-z_][A-Za-z0-9]*(?:[._][A-Za-z0-9_]+)+)\b"),
    re.compile(r"\b([a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"),
    re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\("),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "epyc-review-f1-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (pinned https URL)
        return r.read()


def gh_json(path: str) -> object:
    return json.loads(subprocess.check_output(["gh", "api", "--paginate", path]))


def gh_diff(repo: str, number: int) -> bytes:
    return subprocess.check_output(
        ["gh", "api", "-H", "Accept: application/vnd.github.diff", f"repos/{repo}/pulls/{number}"]
    )


def identifier_tokens(comment: str) -> list[str]:
    toks: list[str] = []
    for rx in _TOKEN_RES:
        for m in rx.finditer(comment):
            t = m.group(1).strip().rstrip("().,")
            if len(t) >= 3 and t not in toks:
                toks.append(t)
    return toks


def anchor_check(golden: dict, root: Path) -> dict:
    """Upstream comments carry NO file/line. Best available offline anchor: does
    >=1 identifier token named in the comment occur in the PR's diff?"""
    rows, counts = [], {"anchored": 0, "unanchored": 0, "no_identifier": 0}
    for case in golden["cases"]:
        diff = (root / case["pr_ref"]["diff_path"]).read_text(errors="replace")
        for g in case["golden_findings"]:
            toks = identifier_tokens(g["comment"])
            hits = [t for t in toks if t in diff or t.split(".")[-1] in diff]
            status = "no_identifier" if not toks else ("anchored" if hits else "unanchored")
            counts[status] += 1
            if status != "anchored":
                rows.append({"golden_id": g["golden_id"], "status": status, "tokens": toks,
                             "comment": g["comment"][:160]})
    return {"counts": counts, "non_anchored": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.out_root)
    manifest = None if args.write_manifest else json.loads(Path(args.manifest).read_text())
    new = {
        "schema_version": "review_f1.augment_v1_manifest.v1",
        "upstream": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT, "license": None,
                     "files": {}},
        "pr_source": {"owner": FORK_OWNER, "fork_prefix": FORK_PREFIX,
                      "diff_media_type": "application/vnd.github.diff"},
        "prs": [],
    }
    errors: list[str] = []

    def check(key: str, got: str, want: str | None) -> None:
        if want is not None and got != want:
            errors.append(f"sha256 mismatch {key}: got {got} want {want}")

    pinned_prs = {} if manifest is None else {(p["bench"], p["number"]): p for p in manifest["prs"]}
    for bench in BENCHES:
        url = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/code_review_benchmarks/{bench}.json"
        data = http_get(url)
        digest = sha256_bytes(data)
        check(f"upstream/{bench}", digest, None if manifest is None else manifest["upstream"]["files"][bench]["sha256"])
        (root / "upstream").mkdir(parents=True, exist_ok=True)
        (root / "upstream" / f"{bench}.json").write_bytes(data)
        records = json.loads(data)
        new["upstream"]["files"][bench] = {"url": url, "sha256": digest, "n_prs": len(records),
                                           "n_comments": sum(len(r["comments"]) for r in records)}
        fork = f"{FORK_OWNER}/{FORK_PREFIX}{bench}"
        pulls = {p["title"]: p for p in gh_json(f"repos/{fork}/pulls?state=all&per_page=100")}
        for rec in records:
            pr = pulls.get(rec["pr_title"])
            if pr is None:
                errors.append(f"no fork PR for title {rec['pr_title']!r} in {fork}")
                continue
            n = pr["number"]
            pin = pinned_prs.get((bench, n))
            if manifest is not None and pin is None:
                errors.append(f"PR {bench}#{n} not in manifest")
            diff = gh_diff(fork, n)
            ddig = sha256_bytes(diff)
            check(f"diff {bench}#{n}", ddig, pin and pin["diff_sha256"])
            if pin and (pin["head_sha"], pin["base_sha"]) != (pr["head"]["sha"], pr["base"]["sha"]):
                errors.append(f"PR {bench}#{n} head/base moved")
            rel = f"diffs/{bench}/pr-{n}.diff"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(diff)
            out_rec = {"case_id": f"{bench}__pr-{n}", "repo": fork, "number": n,
                       "pr_title": rec["pr_title"], "comments": rec["comments"], "diff_path": rel}
            (root / "raw" / bench).mkdir(parents=True, exist_ok=True)
            (root / "raw" / bench / f"pr-{n}.json").write_text(json.dumps(out_rec, indent=2, sort_keys=True))
            new["prs"].append({"bench": bench, "fork": fork, "number": n, "title": rec["pr_title"],
                               "base_ref": pr["base"]["ref"], "base_sha": pr["base"]["sha"],
                               "head_ref": pr["head"]["ref"], "head_sha": pr["head"]["sha"],
                               "diff_path": rel, "diff_sha256": ddig,
                               "n_comments": len(rec["comments"]),
                               "n_low": sum(1 for c in rec["comments"] if str(c.get("severity")).lower() == "low")})

    golden = assemble(str(root / "raw"), "augment-v1")
    (root / "golden_set.json").write_text(json.dumps(golden, indent=2, sort_keys=True))
    new["golden_set"] = {k: golden[k] for k in ("schema_version", "n_cases", "n_golden_total",
                                                "n_golden_scored", "checksum")}
    if manifest is not None:
        for k, v in manifest["golden_set"].items():
            if golden[k] != v:
                errors.append(f"golden_set.{k}: got {golden[k]} want {v}")
    report = anchor_check(golden, root)
    (root / "anchor_report.json").write_text(json.dumps(report, indent=2))
    new["anchor_check"] = report["counts"]
    new["prs"].sort(key=lambda p: (p["bench"], p["number"]))
    if args.write_manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
    print(f"{golden['n_cases']} PRs / {golden['n_golden_total']} golden "
          f"({golden['n_golden_scored']} scored) checksum={golden['checksum']}")
    print(f"anchor check: {report['counts']}")
    for e in errors:
        print("ERROR:", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
