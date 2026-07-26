#!/usr/bin/env python3
"""Tests for the powered SWE-oracle prompt materializer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "build_powered_swebench_prompts", HERE / "build_powered_swebench_prompts.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PromptBuilderTests(unittest.TestCase):
    def make_bare_repo(self, root: Path) -> tuple[Path, str]:
        work = root / "work"
        bare = root / "repos" / "org__repo"
        work.mkdir()
        (root / "repos").mkdir()
        subprocess.run(["git", "init", "-q", str(work)], check=True)
        subprocess.run(
            ["git", "-C", str(work), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(work), "config", "user.name", "Prompt Test"],
            check=True,
        )
        (work / "small.py").write_text("alpha\nbeta\ngamma\n")
        (work / "long.py").write_text("".join(f"line_{index:03d}\n" for index in range(400)))
        subprocess.run(["git", "-C", str(work), "add", "."], check=True)
        subprocess.run(["git", "-C", str(work), "commit", "-qm", "fixture"], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
        ).strip()
        subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
        return root / "repos", commit

    def rows(self, commit: str) -> list[dict]:
        return [
            {
                "repo": "org/repo",
                "instance_id": "org__repo-1",
                "base_commit": commit,
                "problem_statement": "Change beta.",
                "patch": (
                    "--- a/small.py\n+++ b/small.py\n"
                    "@@ -1,3 +1,3 @@\n alpha\n-beta\n+delta\n gamma\n"
                ),
            },
            {
                "repo": "org/repo",
                "instance_id": "org__repo-2",
                "base_commit": commit,
                "problem_statement": "Change a distant line.",
                "patch": (
                    "--- a/long.py\n+++ b/long.py\n@@ -300,1 +300,1 @@\n-line_299\n+changed\n"
                ),
            },
        ]

    def gold_report(self, candidates: list[str], resolved: list[str]) -> dict:
        unresolved = [value for value in candidates if value not in set(resolved)]
        return {
            "schema_version": 2,
            "submitted_ids": candidates,
            "completed_ids": resolved + unresolved,
            "resolved_ids": resolved,
            "unresolved_ids": unresolved,
            "empty_patch_ids": [],
            "incomplete_ids": [],
            "error_ids": [],
            "total_instances": len(candidates),
            "submitted_instances": len(candidates),
            "completed_instances": len(candidates),
            "resolved_instances": len(resolved),
            "unresolved_instances": len(unresolved),
            "empty_patch_instances": 0,
            "error_instances": 0,
        }

    def test_materializes_complete_and_windowed_files_in_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repos, commit = self.make_bare_repo(root)
            questions = MODULE.materialize_prompts(
                self.rows(commit), ["org__repo-2", "org__repo-1"], repos
            )
            self.assertEqual([row["id"] for row in questions], ["org__repo-2", "org__repo-1"])
            self.assertIn("### File: long.py (excerpts)", questions[0]["prompt"])
            self.assertIn("(lines 180-401 of 401)", questions[0]["prompt"])
            self.assertIn("### File: small.py (complete)", questions[1]["prompt"])
            self.assertEqual(questions[1]["prompt_chars"], len(questions[1]["prompt"]))

    def test_missing_repo_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repos, commit = self.make_bare_repo(root)
            (repos / "org__repo").rename(repos / "gone")
            with self.assertRaisesRegex(ValueError, "missing bare source repository"):
                MODULE.materialize_prompts(self.rows(commit)[:1], ["org__repo-1"], repos)

    def test_missing_fixture_id_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repos, commit = self.make_bare_repo(root)
            with self.assertRaisesRegex(ValueError, "accepted IDs missing"):
                MODULE.materialize_prompts(self.rows(commit), ["org__repo-3"], repos)

    def test_duplicate_fixture_id_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repos, commit = self.make_bare_repo(root)
            rows = self.rows(commit)
            with self.assertRaisesRegex(ValueError, "duplicate instance_id"):
                MODULE.materialize_prompts(rows + [rows[0]], ["org__repo-1"], repos)

    def test_validate_inputs_binds_manifest_acceptance_ids_and_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            report_path = root / "gold-report.json"
            accepted_path = root / "accepted.ids"
            fixture_path = root / "fixture.json"
            accepted_path.write_text("org__repo-1\n")
            fixture_path.write_text("[]")
            candidates = ["org__repo-1", "org__repo-2"]
            manifest = {
                "candidate_ids": candidates,
                "candidate_count": 2,
                "gold_validated_target_count": 1,
                "source_fixture_sha256": sha256(fixture_path),
            }
            manifest_path.write_text(json.dumps(manifest))
            report = self.gold_report(candidates, ["org__repo-1"])
            report_path.write_text(json.dumps(report))
            _, acceptance = MODULE.derive_gold_acceptance(
                manifest_path, manifest, report_path, report
            )
            MODULE.validate_inputs(
                manifest_path,
                manifest,
                report_path,
                report,
                acceptance,
                accepted_path,
                ["org__repo-1"],
                fixture_path,
            )
            manifest["source_fixture_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source fixture"):
                MODULE.validate_inputs(
                    manifest_path,
                    manifest,
                    report_path,
                    report,
                    acceptance,
                    accepted_path,
                    ["org__repo-1"],
                    fixture_path,
                )

    def test_validate_inputs_rejects_reordered_accepted_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            report_path = root / "gold-report.json"
            accepted_path = root / "accepted.ids"
            fixture_path = root / "fixture.json"
            candidates = ["org__repo-1", "org__repo-2", "org__repo-3"]
            accepted_path.write_text("org__repo-2\norg__repo-1\n")
            fixture_path.write_text("[]")
            manifest = {
                "candidate_ids": candidates,
                "candidate_count": 3,
                "gold_validated_target_count": 2,
                "source_fixture_sha256": sha256(fixture_path),
            }
            manifest_path.write_text(json.dumps(manifest))
            report = self.gold_report(candidates, candidates)
            report_path.write_text(json.dumps(report))
            _, acceptance = MODULE.derive_gold_acceptance(
                manifest_path, manifest, report_path, report
            )
            with self.assertRaisesRegex(ValueError, "manifest order"):
                MODULE.validate_inputs(
                    manifest_path,
                    manifest,
                    report_path,
                    report,
                    acceptance,
                    accepted_path,
                    ["org__repo-2", "org__repo-1"],
                    fixture_path,
                )

    def test_validate_inputs_binds_exact_gold_report_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            report_path = root / "gold-report.json"
            accepted_path = root / "accepted.ids"
            fixture_path = root / "fixture.json"
            candidates = ["org__repo-1", "org__repo-2"]
            accepted_path.write_text("org__repo-1\n")
            fixture_path.write_text("[]")
            manifest = {
                "candidate_ids": candidates,
                "candidate_count": 2,
                "gold_validated_target_count": 1,
                "source_fixture_sha256": sha256(fixture_path),
            }
            manifest_path.write_text(json.dumps(manifest))
            report = self.gold_report(candidates, ["org__repo-1"])
            report_path.write_text(json.dumps(report))
            _, acceptance = MODULE.derive_gold_acceptance(
                manifest_path, manifest, report_path, report
            )
            report["unscored_note"] = "changes the official report bytes"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "does not match"):
                MODULE.validate_inputs(
                    manifest_path,
                    manifest,
                    report_path,
                    report,
                    acceptance,
                    accepted_path,
                    ["org__repo-1"],
                    fixture_path,
                )

    def test_atomic_writer_is_idempotent_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            MODULE.write_atomic_or_verify(path, "one")
            MODULE.write_atomic_or_verify(path, "one")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                MODULE.write_atomic_or_verify(path, "two")


if __name__ == "__main__":
    unittest.main()
