#!/usr/bin/env python3
"""Assemble the EV-13 review-finding-F1 golden set into an internal schema.

Normalizes the OPEN Augment-v1 golden set (145 bugs, 5 repos x 10 PRs) into a
canonical, checksummed internal case schema the scorer/harness consume. Only
open-source inputs are used; the unlicensed Factory harness/scorer is NOT
vendored.

Raw Augment-v1 shape (github.com/ai-code-review-evaluations/golden_comments):
    {"pr_title": <str>, "comments": [{"comment": <str>, "severity": <str>}, ...]}
One JSON file per PR. That open format carries free-text comments + severity
but NOT a structured criterion/location, so:
  * ``criterion`` falls back to "unspecified" unless a sidecar taxonomy field
    (``category``/``criterion``) is present;
  * ``location`` is parsed from a ``file``/``line`` field or a ``path:line``
    token in the comment when available, else left None (location-agnostic).
The deterministic build-leg matcher tolerates this; the semantic LLM-judge
matcher (a later inference entry) resolves the free-text cases.

Internal case schema (one entry per PR):
    {
      "case_id":   <slug>,          # e.g. "sentry__pr-1234"
      "pr_ref":    {"repo","number","title","diff_path"?,"diff"?},
      "provenance": <str>,          # e.g. "augment-v1"
      "golden_findings": [
        {"golden_id","criterion","location":{file,line_start,line_end}|None,
         "severity","comment","provenance"}
      ]
    }

The assembled file is ``{"schema_version","source","provenance","n_cases",
"n_golden_scored","checksum","cases":[...]}`` where ``checksum`` is the
sha256 of the canonicalized ``cases`` list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "review_f1.golden.v1"
LOW_SEVERITY = "low"
_LOC_RE = re.compile(r"([\w./-]+\.\w+):(\d+)(?:-(\d+))?")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _parse_location(item: dict) -> dict | None:
    if item.get("file"):
        return {
            "file": item.get("file"),
            "line_start": item.get("line_start", item.get("line")),
            "line_end": item.get("line_end"),
        }
    m = _LOC_RE.search(str(item.get("comment", "")))
    if m:
        start = int(m.group(2))
        return {"file": m.group(1), "line_start": start, "line_end": int(m.group(3)) if m.group(3) else start}
    return None


def normalize_pr(raw: dict, repo: str, number: Any, provenance: str) -> dict:
    """Normalize one raw Augment-v1 PR record into an internal case."""
    title = raw.get("pr_title", raw.get("title", ""))
    case_id = raw.get("case_id") or f"{_slug(repo)}__pr-{number}"
    findings = []
    for i, c in enumerate(raw.get("comments", raw.get("golden_findings", []))):
        findings.append(
            {
                "golden_id": c.get("golden_id") or f"{case_id}-g{i}",
                "criterion": c.get("criterion", c.get("category", "unspecified")),
                "location": c.get("location") if "location" in c else _parse_location(c),
                "severity": str(c.get("severity", "medium")).lower(),
                "comment": c.get("comment", ""),
                "provenance": c.get("provenance", provenance),
            }
        )
    return {
        "case_id": case_id,
        "pr_ref": {
            "repo": raw.get("repo", repo),
            "number": raw.get("number", number),
            "title": title,
            **({"diff_path": raw["diff_path"]} if raw.get("diff_path") else {}),
            **({"diff": raw["diff"]} if raw.get("diff") else {}),
        },
        "provenance": provenance,
        "golden_findings": findings,
    }


def _checksum(cases: list[dict]) -> str:
    canon = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def assemble(raw_dir: str, provenance: str = "augment-v1") -> dict:
    """Assemble every ``*.json`` PR record under ``raw_dir`` into the set.

    Directory layout accepted: ``<raw_dir>/<repo>/<something>.json`` (repo from
    the parent dir) or ``<raw_dir>/<file>.json`` (repo/number from record).
    Deterministic: files are sorted before assembly so the checksum is stable.
    """
    root = Path(raw_dir)
    files = sorted(root.rglob("*.json"))
    cases = []
    for idx, path in enumerate(files):
        raw = json.loads(path.read_text())
        repo = raw.get("repo") or path.parent.name
        number = raw.get("number", path.stem)
        cases.append(normalize_pr(raw, repo, number, provenance))
    cases.sort(key=lambda c: c["case_id"])
    n_scored = sum(
        1 for c in cases for g in c["golden_findings"] if g["severity"] != LOW_SEVERITY
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(root),
        "provenance": provenance,
        "n_cases": len(cases),
        "n_golden_total": sum(len(c["golden_findings"]) for c in cases),
        "n_golden_scored": n_scored,
        "checksum": _checksum(cases),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Assemble the review_f1 golden set")
    p.add_argument("--raw-dir", required=True, help="dir of raw Augment-v1 PR JSON files")
    p.add_argument("--out", required=True, help="output assembled golden set JSON")
    p.add_argument("--provenance", default="augment-v1")
    args = p.parse_args(argv)
    assembled = assemble(args.raw_dir, args.provenance)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(assembled, indent=2, sort_keys=True))
    print(
        f"assembled {assembled['n_cases']} PRs / {assembled['n_golden_total']} golden "
        f"({assembled['n_golden_scored']} scored) -> {args.out}\nchecksum={assembled['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
