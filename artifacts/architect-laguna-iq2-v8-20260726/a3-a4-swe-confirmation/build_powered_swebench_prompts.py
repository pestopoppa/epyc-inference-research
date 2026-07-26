#!/usr/bin/env python3
"""Materialize the fixed powered SWE-oracle prompt set after gold acceptance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


WINDOW = 120
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path, expected_type: type) -> Any:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, expected_type):
        raise ValueError(f"{path} must contain a JSON {expected_type.__name__}")
    return value


def read_unique_ids(path: Path) -> list[str]:
    values = path.read_text().splitlines()
    if not values or any(not value or value != value.strip() for value in values):
        raise ValueError(f"{path} must contain one nonempty, whitespace-free ID per line")
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate IDs")
    return values


def patch_files_hunks(patch: str) -> dict[str, list[int]]:
    """Return old-file paths and old-file hunk starts, preserving legacy behavior."""
    out: dict[str, list[int]] = {}
    current: str | None = None
    for line in patch.splitlines():
        file_match = re.match(r"--- a/(.+)", line)
        if file_match:
            current = file_match.group(1)
            out.setdefault(current, [])
        hunk_match = re.match(r"@@ -(\d+)", line)
        if hunk_match and current:
            out[current].append(int(hunk_match.group(1)))
    return {path: hunks for path, hunks in out.items() if path != "/dev/null"}


def show_file(repo_dir: Path, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "--git-dir", str(repo_dir), "show", f"{commit}:{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "git show failed"
        raise ValueError(f"cannot read {path} at {commit} from {repo_dir}: {detail}")
    return result.stdout


def render_prompt(row: dict[str, Any], repos_dir: Path) -> str:
    required = ("repo", "instance_id", "base_commit", "patch", "problem_statement")
    missing = [key for key in required if not isinstance(row.get(key), str) or not row[key]]
    if missing:
        raise ValueError(f"fixture row is missing string fields: {missing}")

    repo_dir = repos_dir / row["repo"].replace("/", "__")
    if not repo_dir.is_dir():
        raise ValueError(f"missing bare source repository: {repo_dir}")

    file_hunks = patch_files_hunks(row["patch"])
    sections: list[str] = []
    for path, hunks in file_hunks.items():
        if not hunks:
            raise ValueError(f"{row['instance_id']} has no text hunks for {path}")
        content = show_file(repo_dir, row["base_commit"], path)
        lines = content.split("\n")
        if len(lines) <= 2 * WINDOW + 40:
            sections.append(f"### File: {path} (complete)\n```python\n{content}\n```")
            continue

        spans: list[tuple[int, int]] = []
        for hunk in sorted(hunks):
            low, high = max(1, hunk - WINDOW), min(len(lines), hunk + WINDOW)
            if spans and low <= spans[-1][1] + 10:
                spans[-1] = (spans[-1][0], high)
            else:
                spans.append((low, high))
        chunks = [
            f"(lines {low}-{high} of {len(lines)})\n" + "\n".join(lines[low - 1 : high])
            for low, high in spans
        ]
        sections.append(
            f"### File: {path} (excerpts)\n```python\n" + "\n...\n".join(chunks) + "\n```"
        )

    created = [
        path
        for path in re.findall(r"\+\+\+ b/(.+)", row["patch"])
        if path not in file_hunks and path != "/dev/null"
    ]
    if not sections and not created:
        raise ValueError(f"{row['instance_id']} produced no promptable patch targets")
    create_note = (
        f"\nThe fix may also require CREATING new file(s): {created} — for a new file, "
        "use a SEARCH/REPLACE block with an empty SEARCH section.\n"
        if created
        else ""
    )
    search_marker = "<" * 7 + " SEARCH"
    divider_marker = "=" * 7
    replace_marker = ">" * 7 + " REPLACE path/to/file.py"
    return f"""You are fixing a real bug in {row["repo"]}. Repository is at commit {row["base_commit"][:12]}.

## Issue
{row["problem_statement"]}

## Relevant files (pre-fix)
{chr(10).join(sections)}
{create_note}
## Your task
Produce the MINIMAL fix as one or more SEARCH/REPLACE blocks, exactly in this format:

{search_marker}
[exact lines copied verbatim from the file shown above]
{divider_marker}
[the replacement lines]
{replace_marker}

Rules: the SEARCH text must match the file content EXACTLY (copy it verbatim, preserving
indentation). Keep each SEARCH minimal (only the lines that change plus 2-3 anchor lines).
Do not modify tests. Output ONLY the SEARCH/REPLACE blocks, nothing else."""


def materialize_prompts(
    fixture_rows: list[dict[str, Any]], accepted_ids: list[str], repos_dir: Path
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in fixture_rows:
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("fixture contains a row without an instance_id")
        if instance_id in rows:
            raise ValueError(f"fixture contains duplicate instance_id: {instance_id}")
        rows[instance_id] = row

    missing = [instance_id for instance_id in accepted_ids if instance_id not in rows]
    if missing:
        raise ValueError(f"accepted IDs missing from fixture: {missing[:3]}")

    questions: list[dict[str, Any]] = []
    for instance_id in accepted_ids:
        prompt = render_prompt(rows[instance_id], repos_dir)
        questions.append(
            {
                "id": instance_id,
                "suite": "swebench_oracle",
                "prompt": prompt,
                "expected": "__patch__",
                "tier": 3,
                "scoring_method": "exact_match",
                "scoring_config": {"extract_pattern": r"(.*)"},
                "prompt_chars": len(prompt),
            }
        )
    return questions


def reject_inconsistent_existing(path: Path, content: str) -> None:
    if path.exists() and path.read_text() != content:
        raise ValueError(f"refusing to overwrite inconsistent existing output: {path}")


def write_atomic_or_verify(path: Path, content: str) -> None:
    reject_inconsistent_existing(path, content)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def derive_gold_acceptance(
    manifest_path: Path,
    manifest: dict[str, Any],
    gold_report_path: Path,
    gold_report: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    validator_path = Path(__file__).with_name("validate_powered_gold_report.py")
    spec = importlib.util.spec_from_file_location("powered_gold_report_validator", validator_path)
    if not spec or not spec.loader:
        raise ValueError(f"cannot load gold validator: {validator_path}")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    accepted, summary = validator.validate(manifest, gold_report)
    summary["manifest_sha256"] = sha256_path(manifest_path)
    summary["gold_report_sha256"] = sha256_path(gold_report_path)
    return accepted, summary


def validate_inputs(
    manifest_path: Path,
    manifest: dict[str, Any],
    gold_report_path: Path,
    gold_report: dict[str, Any],
    acceptance: dict[str, Any],
    accepted_ids_path: Path,
    accepted_ids: list[str],
    fixture_path: Path,
) -> None:
    derived_ids, expected_acceptance = derive_gold_acceptance(
        manifest_path, manifest, gold_report_path, gold_report
    )
    if accepted_ids != derived_ids:
        raise ValueError("accepted-ID file does not match the official gold-derived manifest order")
    if acceptance != expected_acceptance:
        raise ValueError("gold acceptance receipt does not match the official gold report")

    source_hash = manifest.get("source_fixture_sha256")
    if (
        not isinstance(source_hash, str)
        or not SHA256_PATTERN.fullmatch(source_hash)
        or source_hash != sha256_path(fixture_path)
    ):
        raise ValueError("source fixture does not match the manifest SHA-256")
    if sha256_path(accepted_ids_path) != expected_acceptance["accepted_ids_sha256"]:
        raise ValueError("accepted-ID file SHA-256 changed during validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold-report", type=Path, required=True)
    parser.add_argument("--gold-acceptance", type=Path, required=True)
    parser.add_argument("--accepted-ids", type=Path, required=True)
    parser.add_argument("--source-fixture", type=Path)
    parser.add_argument("--repos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest, dict)
    gold_report = load_json(args.gold_report, dict)
    acceptance = load_json(args.gold_acceptance, dict)
    accepted_ids = read_unique_ids(args.accepted_ids)
    source_fixture = manifest.get("source_fixture")
    if args.source_fixture:
        fixture_path = args.source_fixture
    elif isinstance(source_fixture, str) and source_fixture:
        fixture_path = Path(source_fixture)
    else:
        raise ValueError("candidate manifest does not name a source fixture")
    if not fixture_path.is_file():
        raise ValueError(f"source fixture does not exist: {fixture_path}")
    if not args.repos_dir.is_dir():
        raise ValueError(f"bare repository directory does not exist: {args.repos_dir}")

    validate_inputs(
        args.manifest,
        manifest,
        args.gold_report,
        gold_report,
        acceptance,
        args.accepted_ids,
        accepted_ids,
        fixture_path,
    )
    fixture_rows = load_json(fixture_path, list)
    questions = materialize_prompts(fixture_rows, accepted_ids, args.repos_dir)
    output_content = json.dumps(questions)
    prompt_sizes = sorted(question["prompt_chars"] for question in questions)
    summary = {
        "schema": "powered_swebench_oracle_prompts.v1",
        "question_count": len(questions),
        "window_lines": WINDOW,
        "manifest_sha256": sha256_path(args.manifest),
        "gold_report_sha256": sha256_path(args.gold_report),
        "gold_acceptance_sha256": sha256_path(args.gold_acceptance),
        "accepted_ids_sha256": sha256_path(args.accepted_ids),
        "source_fixture_sha256": sha256_path(fixture_path),
        "questions_sha256": sha256_bytes(output_content.encode()),
        "prompt_chars": {
            "min": prompt_sizes[0],
            "median_upper": prompt_sizes[len(prompt_sizes) // 2],
            "max": prompt_sizes[-1],
        },
    }
    summary_content = json.dumps(summary, indent=2) + "\n"
    reject_inconsistent_existing(args.output, output_content)
    reject_inconsistent_existing(args.summary_out, summary_content)
    write_atomic_or_verify(args.output, output_content)
    write_atomic_or_verify(args.summary_out, summary_content)


if __name__ == "__main__":
    main()
