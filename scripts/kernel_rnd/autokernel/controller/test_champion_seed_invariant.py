"""CH-2: "an aggregate champion always exists" is an invariant, not an event.

Hermetic — builds a fake production tree and a deterministic `ldd` on PATH, so the
measurement path is exercised without touching the frozen production tree.

The behaviours worth pinning are the ones that make the invariant trustworthy rather
than merely present: it must actually WRITE a champion, it must not re-write one on
resume (which would displace a champion composition has since advanced), it must be
skipped rather than fatal when no production tree is configured, and it must REFUSE an
unratified build instead of silently anchoring on it.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
import unittest
import unittest.mock

from . import champion_seed as CS
from . import discovery_controller as D
from .. import journal


def _fake_tree(root: Path, *, cpu=b"cpu-binary", gpu=b"gpu-binary") -> Path:
    for sub, payload in (("build", cpu), ("build-hip", gpu)):
        binroot = root / sub / "bin"
        binroot.mkdir(parents=True)
        (binroot / "llama-server").write_bytes(payload)
    return root


def _fake_ldd(bindir: Path) -> None:
    script = bindir / "ldd"
    script.write_text("#!/bin/bash\ncat <<'EOF'\n"
                      "\tlibggml.so.0 => /x/libggml.so.0 (0x00007f00)\nEOF\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


class _Store:
    """Minimal DurableState stand-in: a real journal plus a save recorder."""

    def __init__(self, root: Path) -> None:
        self.book = journal.Journal(str(root / "journal"))
        self.book.initialize()
        self.saved: list[str] = []

    def save(self, state, phase):  # noqa: ANN001 - matches DurableState.save
        self.saved.append(phase)


class ChampionSeedInvariantTests(unittest.TestCase):

    def _env(self, tmp: Path):
        fake = tmp / "fakebin"
        fake.mkdir(exist_ok=True)
        _fake_ldd(fake)
        return {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}

    def _config(self, tree: Path | None, digests=None, *, dry_run=False) -> D.ControllerConfig:
        # A LIVE config: ControllerConfig requires the seven sealed authority fields to
        # be all-present or all-absent, so a partial one would only ever exercise the
        # dry-run path and could not represent a real campaign.
        return D.ControllerConfig(
            output_root=Path("/nonexistent-output-root"),
            production_tree_path=tree,
            production_binary_sha256=digests,
            dry_run=dry_run,
            # planner_context and its digest are validated as a pair.
            planner_context={},
            planner_context_sha256="b" * 64,
            production_base_commit="0" * 40,
            instrument_commit="a" * 40,
            experiment_template_registry_sha256="c" * 64,
            admission_corpus_sha256="d" * 64,
            admission_corpus_version="v1",
            deployment_identity_sha256="e" * 64)

    def test_seeds_a_champion_at_campaign_start(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            _fake_tree(tmp / "tree")
            store, state = _Store(tmp), {}
            with unittest.mock.patch.dict(os.environ, self._env(tmp), clear=True):
                D._seed_champion_if_absent(self._config(tmp / "tree"), store, state)
            self.assertIsNotNone(state.get("champion_seeded_at"),
                                 "the invariant must record that a champion now exists")
            self.assertEqual(state["champion_seed_anchor_commit"], "0" * 40)
            self.assertEqual(store.saved, ["champion_seeded"])

    def test_resume_does_not_reseed(self):
        """A second pass must not displace a champion composition has advanced."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            _fake_tree(tmp / "tree")
            store = _Store(tmp)
            state = {"champion_seeded_at": "2026-08-28T00:00:00Z"}
            with unittest.mock.patch.dict(os.environ, self._env(tmp), clear=True):
                D._seed_champion_if_absent(self._config(tmp / "tree"), store, state)
            self.assertEqual(store.saved, [], "seeding must be idempotent across resume")

    def test_absent_production_tree_is_skipped_not_fatal(self):
        """Tests, dry runs and the CLI construct a config with no production tree."""
        with tempfile.TemporaryDirectory() as t:
            store, state = _Store(Path(t)), {}
            D._seed_champion_if_absent(self._config(None), store, state)
            self.assertNotIn("champion_seeded_at", state)
            self.assertEqual(store.saved, [])

    def test_dry_run_writes_nothing(self):
        """A dry run promises no actor and no hardware; a journal write breaks that."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            _fake_tree(tmp / "tree")
            store, state = _Store(tmp), {}
            with unittest.mock.patch.dict(os.environ, self._env(tmp), clear=True):
                D._seed_champion_if_absent(
                    self._config(tmp / "tree", dry_run=True), store, state)
            self.assertNotIn("champion_seeded_at", state)
            self.assertEqual(store.saved, [])

    def test_unratified_build_is_refused(self):
        """Anchoring on an unratified build would re-anchor every later comparison."""
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            _fake_tree(tmp / "tree")
            store, state = _Store(tmp), {}
            wrong = {"llama_cpu": "0" * 64, "llama_gpu": "1" * 64}
            with unittest.mock.patch.dict(os.environ, self._env(tmp), clear=True):
                with self.assertRaisesRegex(CS.AnchorMeasurementError, "refusing to seed"):
                    D._seed_champion_if_absent(
                        self._config(tmp / "tree", digests=wrong), store, state)
            self.assertNotIn("champion_seeded_at", state,
                             "a refused seed must not mark the invariant satisfied")


if __name__ == "__main__":
    unittest.main()
