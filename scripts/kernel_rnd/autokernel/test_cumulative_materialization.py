"""Real-Git, hardware-free tests for ordered cumulative source materialization."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from . import source_candidate
from .execution import worktree


PATH = "ggml/src/ggml-cuda/cumulative.cu"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


class CumulativeMaterializationTests(unittest.TestCase):
    def test_two_reviewed_nonoverlapping_patches_become_one_exact_stack(self):
        with tempfile.TemporaryDirectory(prefix="ak-cumulative-source-") as raw:
            root = Path(raw).resolve()
            repo_root = root / "repo"
            (repo_root / Path(PATH).parent).mkdir(parents=True)
            _git(repo_root, "init", "-b", "instrument")
            _git(repo_root, "config", "user.name", "AutoKernel Test")
            _git(repo_root, "config", "user.email", "ak@test.invalid")
            source = repo_root / PATH
            source.write_text(
                "int lever_one() {\n    return 1;\n}\n\n"
                "int lever_two() {\n    return 2;\n}\n")
            _git(repo_root, "add", "--", PATH)
            _git(repo_root, "commit", "-m", "production")
            production = _git(repo_root, "rev-parse", "HEAD")
            _git(repo_root, "commit", "--allow-empty", "-m", "instrument")
            instrument = _git(repo_root, "rev-parse", "HEAD")

            manifests = []
            proposals = []
            for index, (old, new, symbol) in enumerate((
                    ("return 1", "return 11", "lever_one"),
                    ("return 2", "return 22", "lever_two")), 1):
                original = source.read_text()
                source.write_text(original.replace(old, new))
                patch_text = _git(
                    repo_root, "diff", "--unified=1", "--", PATH) + "\n"
                patch_text = re.sub(
                    r"^(@@ [^@]+ @@).*$", rf"\1 int {symbol}() {{",
                    patch_text, count=1, flags=re.MULTILINE)
                patch = patch_text.encode()
                _git(repo_root, "restore", "--", PATH)
                manifest = source_candidate.SourcePatchManifest(
                    campaign_id="ak-cumulative-materialization",
                    proposal_id=f"akp-cumulative-{index}",
                    candidate_id=f"akc-cumulative-{index}",
                    source_tree="llama.cpp",
                    production_base_commit=production,
                    instrument_commit=instrument,
                    change_class="arithmetic", declared_files=(PATH,),
                    declared_symbols={PATH: (symbol,)},
                    mechanism_id=f"lever-{index}",
                    patch_sha256=hashlib.sha256(patch).hexdigest(),
                    patch_bytes=patch)
                manifests.append(manifest)
                proposals.append({
                    "proposal_id": manifest.proposal_id,
                    "change_class": "arithmetic",
                    "change": {
                        "files_and_symbols": [f"{PATH}:{symbol}"],
                        "estimated_diff_size": 2,
                    },
                })
            repo = worktree.GitRepo(str(repo_root))
            destination = worktree.SandboxPath.create(
                str(root / "actor"), sandbox_root=str(root),
                production_trees=())
            actor = repo.add_worktree(
                destination, instrument,
                branch=worktree.SafeBranch.for_campaign(
                    "ak-cumulative-materialization", "stack"))
            applied = source_candidate.apply_source_composition(
                tuple(zip(manifests, proposals)), actor=actor,
                composition_id="ordered-two-lever-stack")
            self.assertEqual(applied.manifests, tuple(manifests))
            self.assertEqual(applied.actual_files, (PATH,))
            self.assertEqual(applied.actual_symbols, (
                f"{PATH}:lever_one", f"{PATH}:lever_two"))
            final = Path(actor.path.path, PATH).read_text()
            self.assertIn("return 11", final)
            self.assertIn("return 22", final)
            self.assertEqual(actor.head_commit(), applied.candidate_commit)
            self.assertTrue(actor.is_clean())
            self.assertEqual(
                applied.mutation_receipt["ordered_manifest_sha256s"],
                [manifest.patch_bundle_sha256 for manifest in manifests])
            worktree.teardown_worktree(actor)


if __name__ == "__main__":
    unittest.main()
