"""Independent, hardware-free final launch gate for static GPU discovery.

This file deliberately treats the public deployment bundle and its immutable
integration commit as external artifacts.  It may read and validate them, but
it never invokes an actor, builds source, profiles, or acquires a real device.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

from .. import schemas


COMBINED_SHA = "164b8751f18008d69597489ffd6f12039f60c821"
GRAPH_SHA = "9421cf2f3d6166ffe4022203f5497a27ac65f4d5c734c1bca158608169495a25"
PRODUCTION_SHA = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
INSTRUMENT_SHA = "81bf32f11b4a421880e8f25faec3e4ba872363f0"
INSTRUMENT_DIFF_SHA = "3cf9178fcc00e8c1d3dfc0bfd6086edbff6a6eb6ac528aa4d88b23843b5599c2"
INSTRUMENT_BRANCH = "codex/autokernel-ready-continue-instrument-20260814"
PUBLIC_ROOT = Path(
    "/mnt/raid0/llm/autokernel/deployments/gpu-discovery-combined-final-v1"
)
COMBINED_ROOT = Path(
    "/mnt/raid0/llm/autokernel/worktrees/autokernel-static-batched-integration-20260814"
)
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
INSTRUMENT_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
EXPECTED_MODULES = frozenset({
    "cpu_region_claim", "deployment_factory", "device_claim", "device_sampler",
    "discovery_controller", "discovery_deployment", "discovery_static_registry",
    "evaluator_integrity", "gpu_discovery_beliefs", "gpu_discovery_runner",
    "gpu_load_admission", "gpu_residency_sampler", "gpu_source_adapter",
    "gpu_source_evidence", "gpu_source_proofs", "inference_window",
    "instrument_integrity", "source_candidate", "split_runtime_verifier", "worktree",
})


def _run(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv, cwd=cwd, check=True, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tree_snapshot(root: Path) -> tuple[bytes, bytes]:
    return (
        _run("git", "-C", str(root), "rev-parse", "HEAD").stdout,
        _run(
            "git", "-C", str(root), "status", "--porcelain=v1",
            "--untracked-files=no",
        ).stdout,
    )


class ImmutablePublicBundleGate(unittest.TestCase):
    maxDiff = None

    def _graph(self) -> dict:
        return json.loads((PUBLIC_ROOT / "state/deployment-graph.json").read_text())

    def test_exact_graph_and_every_execution_module_are_immediately_bound(self) -> None:
        graph = self._graph()
        self.assertEqual(graph["graph_sha256"], GRAPH_SHA)
        unhashed = dict(graph)
        unhashed.pop("graph_sha256")
        self.assertEqual(schemas.content_hash(unhashed), GRAPH_SHA)
        self.assertFalse(graph["inference_executed"])
        self.assertFalse(graph["promotion_claim"])

        self.assertEqual(set(graph["execution_modules"]), EXPECTED_MODULES)
        for name, binding in graph["execution_modules"].items():
            path = Path(binding["path"])
            metadata = path.lstat()
            self.assertTrue(stat.S_ISREG(metadata.st_mode), name)
            self.assertFalse(path.is_symlink(), name)
            self.assertEqual(_sha(path.read_bytes()), binding["sha256"], name)
            self.assertTrue(path.is_relative_to(COMBINED_ROOT), name)

        launcher = graph["actor_argv_authority"]
        launcher_path = Path(launcher["module"])
        self.assertEqual(_sha(launcher_path.read_bytes()), launcher["module_sha256"])
        self.assertEqual(
            _run("git", "-C", str(COMBINED_ROOT), "rev-parse", "HEAD").stdout.decode().strip(),
            COMBINED_SHA,
        )

    def test_instrument_authority_is_exact_diff_path_blob_and_branch_bound(self) -> None:
        graph = self._graph()
        authority = graph["source_authority"]
        self.assertEqual(authority["production_base_commit"], PRODUCTION_SHA)
        self.assertEqual(authority["instrument_commit"], INSTRUMENT_SHA)
        self.assertEqual(authority["instrument_branch"], INSTRUMENT_BRANCH)
        self.assertEqual(authority["instrument_diff_sha256"], INSTRUMENT_DIFF_SHA)

        branch = _run(
            "git", "-C", str(INSTRUMENT_ROOT), "rev-parse",
            f"refs/heads/{INSTRUMENT_BRANCH}",
        ).stdout.decode().strip()
        self.assertEqual(branch, INSTRUMENT_SHA)
        diff = _run(
            "git", "-C", str(INSTRUMENT_ROOT), "diff", "--binary",
            f"{PRODUCTION_SHA}..{INSTRUMENT_SHA}",
        ).stdout
        self.assertEqual(_sha(diff), INSTRUMENT_DIFF_SHA)

        review_path = Path(graph["instrument_review"]["path"])
        review_raw = review_path.read_bytes()
        self.assertEqual(_sha(review_raw), graph["instrument_review"]["sha256"])
        review = json.loads(review_raw)
        names = _run(
            "git", "-C", str(INSTRUMENT_ROOT), "diff", "--name-only", "-z",
            f"{PRODUCTION_SHA}..{INSTRUMENT_SHA}",
        ).stdout
        observed = sorted(part.decode() for part in names.split(b"\0") if part)
        self.assertEqual(observed, review["reviewed_diff_paths"])
        for relative, expected in review["reviewed_blobs"].items():
            blob = _run(
                "git", "-C", str(INSTRUMENT_ROOT), "show",
                f"{INSTRUMENT_SHA}:{relative}",
            ).stdout
            self.assertEqual(_sha(blob), expected, relative)

        receipt = dict(review)
        expected_receipt = receipt.pop("receipt_sha256")
        self.assertEqual(schemas.content_hash(receipt), expected_receipt)
        self.assertEqual(
            review["ready_continue_capability"]["source_sha256"],
            authority["ready_continue_contract_source_sha256"],
        )

    def test_public_initialize_and_validate_are_config_only_and_non_mutating(self) -> None:
        before = (_tree_snapshot(PRODUCTION_ROOT), _tree_snapshot(INSTRUMENT_ROOT))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(COMBINED_ROOT)
        module = "scripts.kernel_rnd.autokernel.controller.discovery_deployment_factory"
        with tempfile.TemporaryDirectory(prefix="autokernel-final-gate-") as temporary:
            fresh = Path(temporary) / "bundle"
            initialized = subprocess.run(
                (sys.executable, "-m", module, "--initialize-bundle", str(fresh)),
                cwd=COMBINED_ROOT, env=environment, check=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            init_result = json.loads(initialized.stdout)
            self.assertEqual(init_result["status"], "initialized")
            self.assertFalse(init_result["inference_executed"])
            validated = subprocess.run(
                (sys.executable, "-m", module, "--deployment",
                 str(fresh / "config/deployment.json"), "--validate-only"),
                cwd=COMBINED_ROOT, env=environment, check=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            result = json.loads(validated.stdout)
            self.assertEqual(result["status"], "validated")
            self.assertFalse(result["inference_executed"])
            self.assertTrue((fresh / "state/deployment-graph.json").is_file())
            self.assertFalse((fresh / "operations/claims/device.jsonl").exists())
            self.assertFalse((fresh / "evidence").exists())
        self.assertEqual(
            (_tree_snapshot(PRODUCTION_ROOT), _tree_snapshot(INSTRUMENT_ROOT)), before
        )

    def test_batching_is_one_process_nine_samples_with_serialized_fallback(self) -> None:
        batched = self._graph()["batched_runner"]
        self.assertEqual(batched["processes_per_arm"], 1)
        self.assertEqual(batched["calls_per_arm"], 9)
        self.assertFalse(batched["early_unlock_enabled"])
        self.assertEqual(batched["safe_fallback"], "full_process_cold_serialized_lock")
        self.assertEqual(batched["trust_limit"], "cooperative_same_uid_not_launch_authority")


if __name__ == "__main__":
    unittest.main()
