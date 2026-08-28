"""The anchor does not depend on the candidate, so it must not be rebuilt per candidate.

`_build_key_contract` hashed `patch_bundle_sha256`, `patch_sha256` and
`proposal_sha256` into ONE key covering BOTH plans, so the anchor -- built from the
instrument commit, with nothing about the patch reaching it -- got a fresh cache key
for every candidate. Measured: 44 of 51 anchor builds recompiled a byte-identical
tree.

Reuse is VERIFIED, never assumed, and every verification failure falls through to a
fresh build rather than raising: a stale cache is a performance problem, and turning
it into a campaign failure would be worse than the recompile it saves.
"""
import json
from pathlib import Path
import tempfile
import unittest

from autokernel.controller import discovery_static_registry as registry


class _Path:
    def __init__(self, path): self.path = str(path)


class _Snapshot:
    """Stands in for a source snapshot; only `.path.path` is read."""
    def __init__(self, path): self.path = _Path(path)


class _Result:
    def __init__(self, targets=("llama-bench", "test-backend-ops")):
        self._body = {"facts": {"built_targets": list(targets)}, "log_sha256": "a" * 64}

    def to_dict(self): return dict(self._body)


DEFINES = (("GGML_HIP", "ON"), ("AMDGPU_TARGETS", "gfx90a"))


def _tree(root: Path) -> _Snapshot:
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "kernel.cu").write_text("__global__ void k() {}\n", encoding="utf-8")
    return _Snapshot(src)


def _built(build_dir: Path) -> None:
    (build_dir / "bin").mkdir(parents=True, exist_ok=True)
    for target in registry._REQUIRED_TARGETS:
        (build_dir / "bin" / target).write_text("#!/bin/sh\n", encoding="utf-8")


class AnchorKey(unittest.TestCase):

    def test_the_candidate_fields_are_exactly_what_is_dropped(self):
        self.assertEqual(
            registry._ANCHOR_IRRELEVANT_KEYS,
            frozenset({"build_key", "patch_bundle_sha256", "patch_sha256",
                       "proposal_sha256", "selected_gpu_base_blobs"}))

    def test_the_instrument_and_defines_are_NOT_dropped(self):
        """Dropping either would let one anchor build serve two different anchors."""
        for load_bearing in ("instrument_authority", "cmake_defines", "toolchain",
                             "build_environment", "parallelism",
                             "production_base_authority"):
            self.assertNotIn(load_bearing, registry._ANCHOR_IRRELEVANT_KEYS)


class Reuse(unittest.TestCase):

    def _publish(self, root: Path, snapshot, key="k" * 64, defines=DEFINES,
                 result=None):
        build_dir = root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        _built(build_dir)
        registry._publish_anchor_build(build_dir, snapshot=snapshot,
                                       anchor_build_key=key, defines=defines,
                                       result=result or _Result())
        return build_dir

    def test_a_published_anchor_is_reused_and_replays_the_real_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            build_dir = self._publish(root, snapshot)
            reused = registry._reuse_anchor_build(
                build_dir, snapshot=snapshot, anchor_build_key="k" * 64,
                defines=DEFINES)
            self.assertIsNotNone(reused)
            self.assertTrue(reused.succeeded)
            # The materialization record downstream hashes this, so it must be the
            # real build's dict, not a synthesised one.
            self.assertEqual(reused.to_dict()["log_sha256"], "a" * 64)
            self.assertIn("llama-bench", reused.facts.built_targets)

    def test_a_changed_source_tree_is_not_reused(self):
        """The whole safety argument: identical DIGEST, not merely identical path."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            build_dir = self._publish(root, snapshot)
            (Path(snapshot.path.path) / "kernel.cu").write_text(
                "__global__ void k() { /* changed */ }\n", encoding="utf-8")
            self.assertIsNone(registry._reuse_anchor_build(
                build_dir, snapshot=snapshot, anchor_build_key="k" * 64,
                defines=DEFINES))

    def test_changed_defines_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            build_dir = self._publish(root, snapshot)
            self.assertIsNone(registry._reuse_anchor_build(
                build_dir, snapshot=snapshot, anchor_build_key="k" * 64,
                defines=(("GGML_HIP", "OFF"),)))

    def test_a_different_anchor_key_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            build_dir = self._publish(root, snapshot)
            self.assertIsNone(registry._reuse_anchor_build(
                build_dir, snapshot=snapshot, anchor_build_key="j" * 64,
                defines=DEFINES))

    def test_a_missing_artifact_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            build_dir = self._publish(root, snapshot)
            (build_dir / "bin" / "llama-bench").unlink()
            self.assertIsNone(registry._reuse_anchor_build(
                build_dir, snapshot=snapshot, anchor_build_key="k" * 64,
                defines=DEFINES))

    def test_a_result_missing_a_required_target_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            build_dir = self._publish(root, snapshot,
                                      result=_Result(targets=("llama-bench",)))
            self.assertIsNone(registry._reuse_anchor_build(
                build_dir, snapshot=snapshot, anchor_build_key="k" * 64,
                defines=DEFINES))

    def test_no_receipt_means_build_fresh_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            build_dir = root / "empty"
            build_dir.mkdir()
            self.assertIsNone(registry._reuse_anchor_build(
                build_dir, snapshot=snapshot, anchor_build_key="k" * 64,
                defines=DEFINES))

    def test_a_corrupt_receipt_means_build_fresh_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            build_dir = self._publish(root, snapshot)
            (build_dir / registry._ANCHOR_RECEIPT).write_text("{ not json",
                                                              encoding="utf-8")
            self.assertIsNone(registry._reuse_anchor_build(
                build_dir, snapshot=snapshot, anchor_build_key="k" * 64,
                defines=DEFINES))

    def test_publishing_never_raises_on_an_unwritable_target(self):
        """Publishing is an optimisation; failing it must not fail the campaign."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _tree(root)
            missing = root / "does" / "not" / "exist"
            registry._publish_anchor_build(missing, snapshot=snapshot,
                                           anchor_build_key="k" * 64,
                                           defines=DEFINES, result=_Result())


if __name__ == "__main__":
    unittest.main()
