#!/usr/bin/env python3
"""Unit tests for autokernel/storage.py — durability classes, quota, tombstoned expiry.

NO inference, NO server, NO model, NO GPU, NO sqlite. The suite touches the
filesystem only inside a temporary directory it creates and removes, and it
issues exactly one read-only `git ls-files` (in `GitTrackedIndexTest`) against
this repository.

Temporary directories live **beside this test file**, not in `/tmp`: `/tmp` is a
scratch root in `EPHEMERAL_ROOTS`, so a fixture built there would be refused by
the very guard most of these cases exist to exercise, and every "normal path"
test would pass for the wrong reason.

The suite is organised around the failures the storage plane exists to prevent
(handoff `autokernel-research-loop.md` §5.8, §3.7; `MEASUREMENT.md:146-156`,
`:173-176`, `:223-229`):

  * a citation resolving to a scratch root is an ERROR — including through a
    symlink — because a `tmp` sweep leaves a ratified claim with nothing behind
    it and no event to notice;
  * a MISSING artifact is never classified `hash_and_provenance_only`: that
    would relabel a loss as an intended design decision, which is the exact
    distinction §3.7's durability classes were introduced to preserve;
  * expiry refuses every retention class but `expirable`, refuses outside the
    caller's declared owned roots, and is dry-run until forced;
  * the tombstone reaches the journal BEFORE the bytes go, so the primary record
    survives the artifact;
  * COULD_NOT_CHECK is a third outcome — asserted to be neither PASS nor FAIL,
    and never truthy.

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/test_storage.py
    python3 -W error::ResourceWarning -m unittest scripts/kernel_rnd/autokernel/test_storage.py
    python3 scripts/kernel_rnd/autokernel/test_storage.py
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import shutil
import stat
import sys
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schemas as S  # noqa: E402
import storage as ST  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


class RecordingJournal:
    """The journal contract, instrumented.

    Records each appended record AND whether `watch_path` still existed at the
    moment of the append, which is how the ordering guarantee (tombstone before
    bytes) is asserted rather than assumed.
    """

    # `None` is one of the bad return values under test, so the "use the default"
    # signal cannot be None.
    AUTO = object()

    def __init__(self, watch_path: str | None = None, event_id=AUTO):
        self.records: list[dict] = []
        self.existed_at_append: list[bool] = []
        self.watch_path = watch_path
        self._event_id = event_id

    def append(self, record):
        self.records.append(copy.deepcopy(dict(record)))
        self.existed_at_append.append(
            os.path.lexists(self.watch_path) if self.watch_path else False
        )
        if self._event_id is not RecordingJournal.AUTO:
            return self._event_id
        return f"ake-{len(self.records):04d}"


class _TmpTest(unittest.TestCase):
    """Base fixture: a scratch-free temporary tree beside this file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="_ak_storage_test_", dir=HERE)
        self.addCleanup(self._cleanup)
        self.tmp = os.path.realpath(self._tmp.name)
        self._restore_modes: list[tuple[str, int]] = []

    def _cleanup(self):
        # Restore any mode we broke first, or the tree cannot be removed.
        for path, mode in reversed(self._restore_modes):
            if os.path.lexists(path):
                os.chmod(path, mode)
        self._tmp.cleanup()

    def chmod_temporarily(self, path: str, mode: int):
        self._restore_modes.append((path, stat.S_IMODE(os.stat(path).st_mode)))
        os.chmod(path, mode)

    def write(self, rel: str, content: bytes | str = b"x") -> str:
        path = os.path.join(self.tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
        with open(path, mode, **kwargs) as fh:
            fh.write(content)
        return path


# =============================================================================
# Scratch roots — a citation of record may never resolve into one
# =============================================================================

class ScratchPathTest(_TmpTest):

    def test_every_ephemeral_root_is_rejected(self):
        for root in ST.EPHEMERAL_ROOTS:
            with self.subTest(root=root):
                self.assertTrue(ST.is_scratch_path(root + "/campaign/summary.json"))
                with self.assertRaises(ST.ScratchCitationError):
                    ST.assert_not_scratch(root + "/campaign/summary.json")

    def test_the_root_itself_is_scratch_not_merely_its_children(self):
        self.assertTrue(ST.is_scratch_path("/mnt/raid0/llm/tmp"))

    def test_a_lookalike_sibling_is_not_scratch(self):
        # `/tmpfoo` merely starts with the same characters; a prefix test that
        # forgot the separator would reject a legitimate root.
        self.assertFalse(ST.is_scratch_path("/tmpfoo/data/x.json"))
        self.assertFalse(ST.is_scratch_path("/mnt/raid0/llm/tmpdata/x.json"))

    def test_scratch_reached_through_a_symlink_is_still_scratch(self):
        link = os.path.join(self.tmp, "evidence")
        os.symlink("/mnt/raid0/llm/tmp/some-campaign", link)
        self.assertTrue(ST.is_scratch_path(link))
        with self.assertRaises(ST.ScratchCitationError):
            ST.assert_not_scratch(link)

    def test_scratch_reached_through_dotdot_is_still_scratch(self):
        self.assertTrue(ST.is_scratch_path("/mnt/raid0/llm/models/../tmp/x.json"))

    def test_classify_refuses_a_scratch_path(self):
        index = ST.StaticTrackedIndex(self.tmp, [])
        with self.assertRaises(ST.ScratchCitationError):
            ST.classify("/tmp/whatever.json", tracked_index=index)

    def test_verify_durability_fails_a_scratch_citation_for_every_class(self):
        for klass in sorted(S.DURABILITY_CLASSES):
            with self.subTest(klass=klass):
                verdicts = ST.verify_durability([{
                    "path": "/mnt/raid0/llm/tmp/claude-artifacts/np_context.html",
                    "durability_class": klass,
                    "sha256": _sha("np"), "provenance": "measured 2026-07-23",
                }])
                self.assertEqual(verdicts[0].outcome, ST.FAIL)
                self.assertIn("scratch", verdicts[0].check.reasons[0])

    def test_ephemeral_roots_agree_with_the_ratified_enforcer(self):
        """Two copies of a security boundary is how one of them loses an entry.

        `MEASUREMENT.md:146-156` names `check_evidence_durability.py` as the
        enforcer; this module carries its own copy of the scratch-root tuple, so
        pin them together.
        """
        enforcer_path = os.path.join(
            ST.REPO_ROOT, "scripts", "validate", "check_evidence_durability.py")
        self.assertTrue(os.path.exists(enforcer_path), enforcer_path)
        spec = importlib.util.spec_from_file_location(
            "_ak_test_check_evidence_durability", enforcer_path)
        module = importlib.util.module_from_spec(spec)
        # @dataclass resolves annotations through sys.modules, so the module must
        # be registered before exec_module rather than after.
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        self.assertEqual(set(module.EPHEMERAL_ROOTS), set(ST.EPHEMERAL_ROOTS))


# =============================================================================
# classify() — durability class assignment
# =============================================================================

class ClassifyTest(_TmpTest):

    def test_tracked_file_is_carried_in_git(self):
        path = self.write("data/camp/summary.json", b"{}")
        index = ST.StaticTrackedIndex(self.tmp, ["data/camp/summary.json"])
        result = ST.classify(path, tracked_index=index)
        self.assertEqual(result.durability_class, "carried_in_git")
        self.assertTrue(result.in_repo)
        self.assertIs(result.tracked, True)

    def test_untracked_small_file_is_durable_untracked(self):
        path = self.write("docs/design/protocol.md", b"# p")
        index = ST.StaticTrackedIndex(self.tmp, ["data/camp/summary.json"])
        self.assertEqual(
            ST.classify(path, tracked_index=index).durability_class,
            "durable_untracked")

    def test_untracked_oversized_file_is_hash_and_provenance_only(self):
        path = self.write("build/libggml.so", b"0" * 4096)
        index = ST.StaticTrackedIndex(self.tmp, [])
        self.assertEqual(
            ST.classify(path, tracked_index=index,
                        carry_threshold_bytes=16).durability_class,
            "hash_and_provenance_only")

    def test_path_outside_the_working_tree_is_hash_and_provenance_only(self):
        outside = self.write("outside/binary.bin", b"0" * 32)
        inner = os.path.join(self.tmp, "repo")
        os.makedirs(inner, exist_ok=True)
        index = ST.StaticTrackedIndex(inner, [])
        result = ST.classify(outside, tracked_index=index)
        self.assertEqual(result.durability_class, "hash_and_provenance_only")
        self.assertFalse(result.in_repo)

    def test_missing_path_is_refused_not_relabelled_as_expected_absence(self):
        """The single most important negative in this module.

        Inferring `hash_and_provenance_only` from absence would turn every lost
        artifact into a documented design decision, which is precisely the
        distinction §3.7 introduced the classes to keep.
        """
        index = ST.StaticTrackedIndex(self.tmp, [])
        with self.assertRaises(ST.UnclassifiablePath) as ctx:
            ST.classify(os.path.join(self.tmp, "gone.json"), tracked_index=index)
        self.assertIn("absence is not a durability class", str(ctx.exception))

    def test_missing_tracked_index_is_refused_not_guessed(self):
        path = self.write("data/camp/summary.json", b"{}")
        with self.assertRaises(ST.UnclassifiablePath):
            ST.classify(path)

    def test_every_class_returned_is_in_the_schemas_vocabulary(self):
        path = self.write("data/camp/a.json", b"{}")
        index = ST.StaticTrackedIndex(self.tmp, ["data/camp/a.json"])
        self.assertIn(ST.classify(path, tracked_index=index).durability_class,
                      S.DURABILITY_CLASSES)

    def test_directory_counts_as_tracked_when_git_carries_anything_beneath(self):
        os.makedirs(os.path.join(self.tmp, "data", "camp", "sub"), exist_ok=True)
        self.write("data/camp/sub/x.json", b"{}")
        index = ST.StaticTrackedIndex(self.tmp, ["data/camp/sub/x.json"])
        self.assertTrue(index.is_tracked(os.path.join(self.tmp, "data", "camp")))
        self.assertEqual(
            ST.classify(os.path.join(self.tmp, "data", "camp"),
                        tracked_index=index).durability_class,
            "carried_in_git")

    def test_directory_prefix_match_requires_a_separator(self):
        index = ST.StaticTrackedIndex(self.tmp, ["data/campaign-two/x.json"])
        os.makedirs(os.path.join(self.tmp, "data", "campaign"), exist_ok=True)
        self.assertFalse(index.is_tracked(os.path.join(self.tmp, "data", "campaign")))

    def test_static_index_refuses_a_path_outside_its_tree(self):
        index = ST.StaticTrackedIndex(os.path.join(self.tmp, "repo"), [])
        with self.assertRaises(ST.UnclassifiablePath):
            index.is_tracked(os.path.join(self.tmp, "elsewhere", "x"))


class GitTrackedIndexTest(_TmpTest):
    """One read-only `git ls-files` against this repository."""

    def test_real_repository_tracked_and_untracked(self):
        index = ST.GitTrackedIndex(ST.REPO_ROOT)
        tracked = os.path.join(ST.REPO_ROOT, "scripts", "kernel_rnd", "kernel_store.py")
        self.assertTrue(os.path.exists(tracked))
        self.assertTrue(index.is_tracked(tracked))
        # The fixture directory this test just created cannot be tracked.
        self.assertFalse(index.is_tracked(self.tmp))

    def test_non_repository_raises_rather_than_reporting_nothing_tracked(self):
        outside = tempfile.mkdtemp(prefix="_ak_not_a_repo_", dir="/tmp")
        self.addCleanup(shutil.rmtree, outside, True)
        with self.assertRaises(ST.UnclassifiablePath):
            ST.GitTrackedIndex(outside)

    def test_missing_directory_raises(self):
        with self.assertRaises(ST.UnclassifiablePath):
            ST.GitTrackedIndex(os.path.join(self.tmp, "no-such-dir"))


# =============================================================================
# Campaign evidence root — created, not assumed
# =============================================================================

class EvidenceRootTest(_TmpTest):

    def test_creates_the_mandated_layout(self):
        root = ST.ensure_campaign_evidence_root("ak-llama_gpu-decode-20260803",
                                                repo_root=self.tmp)
        self.assertTrue(root.created)
        self.assertEqual(
            root.path,
            os.path.join(self.tmp, "data", "ak-llama_gpu-decode-20260803"))
        self.assertTrue(os.path.isfile(root.sha256sums_path))
        self.assertTrue(os.path.isfile(root.readme_path))

    def test_created_root_is_not_yet_compliant(self):
        """Creating the SHAPE is not satisfying the CONTENT.

        `MEASUREMENT.md:146-156` requires the README to state what was measured,
        when, and which claim it backs. A checker that blessed the stub would
        certify an empty promise.
        """
        root = ST.ensure_campaign_evidence_root("ak-camp", repo_root=self.tmp)
        self.assertEqual(root.layout.outcome, ST.FAIL)
        self.assertIn("stub", " ".join(root.layout.reasons))

    def test_filled_readme_passes_layout(self):
        root = ST.ensure_campaign_evidence_root(
            "ak-camp", repo_root=self.tmp,
            claim="P-AK-SEARCH-1 decode candidate vs v8 anchor",
            what_was_measured="llama-bench decode t/s, 5 reps",
            measured_at="2026-08-03T12:00:00Z")
        self.assertEqual(root.layout.outcome, ST.PASS)
        with open(root.readme_path, encoding="utf-8") as fh:
            self.assertNotIn(ST.README_STUB_MARKER, fh.read())

    def test_idempotent_and_non_clobbering(self):
        first = ST.ensure_campaign_evidence_root("ak-camp", repo_root=self.tmp)
        with open(first.readme_path, "w", encoding="utf-8") as fh:
            fh.write("# hand written\n")
        with open(first.sha256sums_path, "w", encoding="utf-8") as fh:
            fh.write(f"{_sha('a')}  data/ak-camp/a.json\n")
        second = ST.ensure_campaign_evidence_root("ak-camp", repo_root=self.tmp)
        self.assertFalse(second.created)
        with open(second.readme_path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "# hand written\n")
        with open(second.sha256sums_path, encoding="utf-8") as fh:
            self.assertIn("a.json", fh.read())

    def test_unsafe_campaign_ids_are_refused(self):
        for bad in ("../escape", "a/b", "", ".hidden", "/abs", "x" * 200, None, 7):
            with self.subTest(bad=bad):
                with self.assertRaises(ST.EvidenceRootError):
                    ST.campaign_evidence_root(bad, repo_root=self.tmp)

    def test_layout_fails_when_sha256sums_is_empty_beside_artifacts(self):
        root = ST.ensure_campaign_evidence_root(
            "ak-camp", repo_root=self.tmp, claim="c", what_was_measured="w",
            measured_at="2026-08-03T00:00:00Z")
        with open(os.path.join(root.path, "summary.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        check = ST.check_evidence_root_layout(root.path)
        self.assertEqual(check.outcome, ST.FAIL)
        self.assertIn("empty", " ".join(check.reasons))

    def test_layout_fails_when_root_absent_and_could_not_check_when_unreadable(self):
        missing = os.path.join(self.tmp, "data", "nope")
        self.assertEqual(ST.check_evidence_root_layout(missing).outcome, ST.FAIL)

        root = ST.ensure_campaign_evidence_root("ak-camp", repo_root=self.tmp)
        self.chmod_temporarily(root.path, 0o000)
        check = ST.check_evidence_root_layout(root.path)
        self.assertEqual(check.outcome, ST.COULD_NOT_CHECK)

    def test_layout_fails_when_required_files_are_missing(self):
        root = os.path.join(self.tmp, "data", "bare")
        os.makedirs(root)
        check = ST.check_evidence_root_layout(root)
        self.assertEqual(check.outcome, ST.FAIL)
        joined = " ".join(check.reasons)
        self.assertIn(ST.README_NAME, joined)
        self.assertIn(ST.SHA256SUMS_NAME, joined)


# =============================================================================
# Usage, quota, DISK_PRESSURE
# =============================================================================

class UsageTest(_TmpTest):

    def test_counts_files_and_bytes(self):
        self.write("tree/a.bin", b"0" * 5000)
        self.write("tree/sub/b.bin", b"0" * 5000)
        usage = ST.measure_usage(os.path.join(self.tmp, "tree"))
        self.assertEqual(usage.file_count, 2)
        self.assertEqual(usage.bytes_apparent, 10000)
        self.assertGreaterEqual(usage.bytes_on_disk, 10000)

    def test_hardlinks_are_counted_once(self):
        target = self.write("tree/a.bin", b"0" * 8192)
        os.link(target, os.path.join(self.tmp, "tree", "b.bin"))
        usage = ST.measure_usage(os.path.join(self.tmp, "tree"))
        self.assertEqual(usage.file_count, 2)
        self.assertEqual(usage.hardlink_duplicates, 1)
        self.assertEqual(usage.bytes_apparent, 8192)

    def test_single_file_root(self):
        path = self.write("solo.bin", b"0" * 100)
        usage = ST.measure_usage(path)
        self.assertEqual(usage.file_count, 1)
        self.assertEqual(usage.bytes_apparent, 100)

    def test_unreadable_subtree_raises_instead_of_under_reporting(self):
        """A partial walk silently under-reports, and under-reporting is how a
        campaign blows a budget while every component reports healthy."""
        self.write("tree/open/a.bin", b"0" * 100)
        closed = os.path.join(self.tmp, "tree", "closed")
        os.makedirs(closed)
        self.write("tree/closed/b.bin", b"0" * 100)
        self.chmod_temporarily(closed, 0o000)
        with self.assertRaises(OSError):
            ST.measure_usage(os.path.join(self.tmp, "tree"))


class PolicyTest(unittest.TestCase):

    def test_rejects_negative_and_non_finite_budgets(self):
        for kwargs in ({"campaign_quota_gb": -1.0},
                       {"campaign_quota_gb": float("inf")},
                       {"campaign_quota_gb": float("nan")},
                       {"campaign_quota_gb": 10.0, "headroom_floor_gb": -5.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ST.StoragePolicy(**kwargs)

    def test_rejects_non_numeric_budget(self):
        with self.assertRaises(TypeError):
            ST.StoragePolicy(campaign_quota_gb="20")

    def test_rejects_a_floor_smaller_than_one_allocation_step(self):
        with self.assertRaises(ValueError):
            ST.StoragePolicy(campaign_quota_gb=10.0, allocation_safety_factor=0.5)

    def test_effective_floor_is_stepped_up_to_cover_one_allocation(self):
        policy = ST.StoragePolicy(campaign_quota_gb=10.0, headroom_floor_gb=5.0,
                                  largest_single_allocation_gb=15.0,
                                  allocation_safety_factor=2.0)
        # A 5 GiB floor is meaningless when the next legal action writes 15 GiB.
        self.assertEqual(policy.effective_floor_bytes, int(30.0 * 1024 ** 3))

    def test_declared_floor_wins_when_it_is_the_larger(self):
        policy = ST.StoragePolicy(campaign_quota_gb=10.0, headroom_floor_gb=100.0,
                                  largest_single_allocation_gb=15.0,
                                  allocation_safety_factor=2.0)
        self.assertEqual(policy.effective_floor_bytes, int(100.0 * 1024 ** 3))

    def test_owned_roots_default_to_empty(self):
        self.assertEqual(ST.StoragePolicy(campaign_quota_gb=1.0).owned_roots, ())


class DiskPressureTest(_TmpTest):

    def test_impossible_floor_reports_disk_pressure(self):
        policy = ST.StoragePolicy(campaign_quota_gb=1.0,
                                  headroom_floor_gb=1_000_000.0,
                                  largest_single_allocation_gb=0.0,
                                  allocation_safety_factor=1.0)
        state = ST.disk_pressure(self.tmp, policy)
        self.assertEqual(state.state, ST.DISK_PRESSURE)
        self.assertTrue(state.pressured)
        self.assertTrue(state.reasons)

    def test_zero_floor_reports_ok(self):
        policy = ST.StoragePolicy(campaign_quota_gb=1.0, headroom_floor_gb=0.0,
                                  largest_single_allocation_gb=0.0,
                                  allocation_safety_factor=1.0)
        state = ST.disk_pressure(self.tmp, policy)
        self.assertEqual(state.state, ST.STORAGE_OK)
        self.assertFalse(state.pressured)
        self.assertGreater(state.total_bytes, 0)

    def test_floor_boundary_is_strict_less_than(self):
        policy = ST.StoragePolicy(campaign_quota_gb=1.0, headroom_floor_gb=0.0,
                                  largest_single_allocation_gb=0.0,
                                  allocation_safety_factor=1.0)
        free = ST.disk_pressure(self.tmp, policy).free_bytes
        exact = ST.StoragePolicy(campaign_quota_gb=1.0,
                                 headroom_floor_gb=free / (1024 ** 3),
                                 largest_single_allocation_gb=0.0,
                                 allocation_safety_factor=1.0)
        self.assertEqual(ST.disk_pressure(self.tmp, exact).state, ST.STORAGE_OK)

    def test_unreadable_filesystem_raises_rather_than_assuming_healthy(self):
        with self.assertRaises(OSError):
            ST.disk_pressure(os.path.join(self.tmp, "no-such-path"),
                             ST.StoragePolicy(campaign_quota_gb=1.0))


class QuotaTest(_TmpTest):

    def _usage(self, nbytes: int) -> ST.Usage:
        return ST.Usage(root=self.tmp, bytes_on_disk=nbytes, bytes_apparent=nbytes,
                        file_count=1, dir_count=1, hardlink_duplicates=0)

    def test_ok_warn_exhausted_thresholds(self):
        policy = ST.StoragePolicy(campaign_quota_gb=10.0, quota_warn_fraction=0.8)
        gib = 1024 ** 3
        self.assertEqual(ST.campaign_quota_state(self._usage(1 * gib), policy).state,
                         ST.QUOTA_OK)
        self.assertEqual(ST.campaign_quota_state(self._usage(8 * gib), policy).state,
                         ST.QUOTA_WARN)
        self.assertEqual(ST.campaign_quota_state(self._usage(10 * gib), policy).state,
                         ST.QUOTA_EXHAUSTED)
        self.assertTrue(
            ST.campaign_quota_state(self._usage(11 * gib), policy).exhausted)

    def test_zero_quota_is_bounded_not_unlimited(self):
        policy = ST.StoragePolicy(campaign_quota_gb=0.0)
        self.assertEqual(ST.campaign_quota_state(self._usage(0), policy).state,
                         ST.QUOTA_OK)
        self.assertEqual(ST.campaign_quota_state(self._usage(1), policy).state,
                         ST.QUOTA_EXHAUSTED)

    def test_quota_exhaustion_is_independent_of_disk_pressure(self):
        """§8.11 keeps them apart: one is the campaign overspending, the other is
        the host running out. Conflating them either halts the loop for a budget
        an operator could raise, or lets it allocate on a host that cannot afford it."""
        policy = ST.StoragePolicy(campaign_quota_gb=0.0, headroom_floor_gb=0.0,
                                  largest_single_allocation_gb=0.0,
                                  allocation_safety_factor=1.0)
        quota = ST.campaign_quota_state(self._usage(4096), policy)
        disk = ST.disk_pressure(self.tmp, policy)
        self.assertEqual(quota.state, ST.QUOTA_EXHAUSTED)
        self.assertEqual(disk.state, ST.STORAGE_OK)
        self.assertNotIn(quota.state, (ST.STORAGE_OK, ST.DISK_PRESSURE))

    def test_measured_usage_feeds_the_quota(self):
        self.write("camp/a.bin", b"0" * 4096)
        usage = ST.measure_usage(os.path.join(self.tmp, "camp"))
        policy = ST.StoragePolicy(campaign_quota_gb=1.0)
        self.assertEqual(ST.campaign_quota_state(usage, policy).state, ST.QUOTA_OK)

    def test_rejects_a_non_usage_argument(self):
        with self.assertRaises(TypeError):
            ST.campaign_quota_state({"bytes_on_disk": 1},
                                    ST.StoragePolicy(campaign_quota_gb=1.0))


# =============================================================================
# Hashing helpers
# =============================================================================

class HashTest(_TmpTest):

    def test_hash_file_matches_hashlib(self):
        path = self.write("a.bin", b"hello")
        self.assertEqual(ST.hash_file(path), hashlib.sha256(b"hello").hexdigest())

    def test_tree_manifest_is_stable_and_content_sensitive(self):
        self.write("tree/a.bin", b"a")
        self.write("tree/sub/b.bin", b"b")
        root = os.path.join(self.tmp, "tree")
        first = ST.hash_tree_manifest(root)
        self.assertEqual(first, ST.hash_tree_manifest(root))
        self.write("tree/sub/b.bin", b"c")
        self.assertNotEqual(first, ST.hash_tree_manifest(root))

    def test_tree_manifest_records_symlinks_by_target(self):
        self.write("tree/a.bin", b"a")
        os.symlink("/mnt/raid0/llm/models/x.gguf",
                   os.path.join(self.tmp, "tree", "weights"))
        # Following it would hash bytes that live elsewhere and are not being
        # reclaimed; the manifest must still be computable.
        self.assertTrue(ST.hash_tree_manifest(os.path.join(self.tmp, "tree")))


# =============================================================================
# Tombstone record
# =============================================================================

def _tombstone(**overrides) -> dict:
    record = {
        "schema": ST.SCHEMA_ARTIFACT_TOMBSTONE,
        "tombstone_id": "akt-" + _sha("t")[:32],
        "campaign_id": "ak-llama_gpu-decode-20260803",
        "artifact_path": "/mnt/raid0/llm/llama.cpp-ak-x/build",
        "artifact_sha256": _sha("artifact"),
        "durability_class": "hash_and_provenance_only",
        "retention_class": "expirable",
        "expirable_kind": "rejected_candidate_build_tree",
        "rule_id": "AK0-retention-rule/v1",
        "reason": "candidate rejected on correctness; build tree no longer needed",
        "actor": "autokernel-storage-plane",
        "size_bytes": 13_000_000_000,
        "file_count": 41_000,
        "preconditions": {"candidate_id": "akc-1", "candidate_status": "rejected"},
        "reclaimed_at": NOW.isoformat(),
        "reclamation_state": "intent",
    }
    record.update(overrides)
    return record


class TombstoneRecordTest(unittest.TestCase):

    def test_minimal_valid_record(self):
        self.assertEqual(ST.validate_artifact_tombstone(_tombstone()), [])

    def test_every_required_field_is_required(self):
        base = _tombstone()
        for key in sorted(base):
            with self.subTest(key=key):
                record = _tombstone()
                del record[key]
                violations = ST.validate_artifact_tombstone(record)
                self.assertTrue(violations, f"deleting {key} produced no violation")
                self.assertTrue(any(key in v for v in violations), violations)

    def test_carried_in_git_is_not_a_reclaimable_class(self):
        violations = ST.validate_artifact_tombstone(
            _tombstone(durability_class="carried_in_git"))
        self.assertTrue(any("carried_in_git" in v for v in violations))

    def test_retention_class_must_be_expirable(self):
        for klass in ("permanent_in_repo", "permanent_large", "never_stored"):
            with self.subTest(klass=klass):
                self.assertTrue(
                    ST.validate_artifact_tombstone(_tombstone(retention_class=klass)))

    def test_naive_timestamp_refused(self):
        violations = ST.validate_artifact_tombstone(
            _tombstone(reclaimed_at="2026-08-03T12:00:00"))
        self.assertTrue(any("timezone" in v for v in violations))

    def test_bad_hash_refused(self):
        self.assertTrue(ST.validate_artifact_tombstone(
            _tombstone(artifact_sha256="deadbeef")))

    def test_relative_artifact_path_refused(self):
        self.assertTrue(ST.validate_artifact_tombstone(
            _tombstone(artifact_path="build/x")))

    def test_failed_state_requires_an_error(self):
        self.assertTrue(ST.validate_artifact_tombstone(
            _tombstone(reclamation_state="failed")))
        self.assertEqual(ST.validate_artifact_tombstone(
            _tombstone(reclamation_state="failed", error="PermissionError: x")), [])

    def test_authority_flavoured_key_refused(self):
        violations = ST.validate_artifact_tombstone(_tombstone(auto_freeze=True))
        self.assertTrue(any("authority" in v for v in violations))

    def test_validator_never_raises_on_garbage(self):
        for junk in (None, [], 7, "x", {"schema": 1}):
            with self.subTest(junk=junk):
                self.assertTrue(ST.validate_artifact_tombstone(junk))

    def test_tombstone_id_is_deterministic_and_timestamp_free(self):
        args = ("ak-c", "/a/b", _sha("x"), "stale_profiler_trace", "AK0-rule/v1")
        self.assertEqual(ST.tombstone_id(*args), ST.tombstone_id(*args))
        self.assertTrue(ST.tombstone_id(*args).startswith(ST.TOMBSTONE_ID_PREFIX))
        other = ST.tombstone_id("ak-c", "/a/c", _sha("x"), "stale_profiler_trace",
                                "AK0-rule/v1")
        self.assertNotEqual(ST.tombstone_id(*args), other)


# =============================================================================
# Expiry
# =============================================================================

class _ExpiryTest(_TmpTest):

    def setUp(self):
        super().setUp()
        self.owned = os.path.join(self.tmp, "worktrees")
        os.makedirs(self.owned, exist_ok=True)
        self.policy = ST.StoragePolicy(campaign_quota_gb=100.0,
                                       owned_roots=(self.owned,))

    def make_tree(self, name="llama.cpp-ak-1/build") -> str:
        path = os.path.join(self.owned, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "libggml.so"), "wb") as fh:
            fh.write(b"0" * 2048)
        return path

    def artifact(self, path=None, **overrides) -> ST.ExpirableArtifact:
        path = path or self.make_tree()
        kwargs = dict(
            path=path,
            campaign_id="ak-llama_gpu-decode-20260803",
            sha256=_sha("tree"),
            durability_class="hash_and_provenance_only",
            expirable_kind="rejected_candidate_build_tree",
            reason="candidate rejected on correctness",
            rule_id="AK0-retention-rule/v1",
            actor="autokernel-storage-plane",
            retention_class="expirable",
            preconditions={
                "candidate_id": "akc-0001",
                "candidate_status": "rejected",
                "champion_status": "none",
                "evaluation_events_journaled": True,
            },
        )
        kwargs.update(overrides)
        return ST.ExpirableArtifact(**kwargs)


class ExpiryRefusalTest(_ExpiryTest):

    def test_refuses_every_non_expirable_retention_class(self):
        for klass in ("permanent_in_repo", "permanent_large", "never_stored"):
            with self.subTest(klass=klass):
                with self.assertRaises(ST.ExpiryRefused) as ctx:
                    ST.plan_expiry(self.artifact(retention_class=klass), self.policy)
                self.assertIn("not reclaimable", str(ctx.exception))

    def test_refuses_an_unknown_retention_class(self):
        with self.assertRaises(ST.ExpiryRefused):
            ST.plan_expiry(self.artifact(retention_class="probably_fine"),
                           self.policy)

    def test_refuses_an_expirable_kind_outside_the_three_named_in_5_8(self):
        with self.assertRaises(ST.ExpiryRefused):
            ST.plan_expiry(self.artifact(expirable_kind="old_logs"), self.policy)

    def test_refuses_carried_in_git(self):
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(durability_class="carried_in_git"),
                           self.policy)
        self.assertIn("reclaims nothing", str(ctx.exception))

    def test_refuses_when_no_owned_root_is_declared(self):
        policy = ST.StoragePolicy(campaign_quota_gb=100.0)
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(), policy)
        self.assertIn("owned_roots is empty", str(ctx.exception))

    def test_refuses_outside_the_declared_owned_roots(self):
        outside = os.path.join(self.tmp, "elsewhere", "build")
        os.makedirs(outside)
        with self.assertRaises(ST.ExpiryRefused):
            ST.plan_expiry(self.artifact(path=outside), self.policy)

    def test_refuses_the_owned_root_itself(self):
        with self.assertRaises(ST.ExpiryRefused):
            ST.plan_expiry(self.artifact(path=self.owned), self.policy)

    def test_denies_production_trees_even_when_declared_owned(self):
        """Invariant 3. The denial must beat the caller's own declaration, or a
        mis-declared root becomes authority over a frozen kernel."""
        for tree in ST.PRODUCTION_TREES:
            with self.subTest(tree=tree):
                policy = ST.StoragePolicy(campaign_quota_gb=1.0,
                                          owned_roots=("/mnt/raid0/llm",))
                with self.assertRaises(ST.ExpiryRefused) as ctx:
                    ST.plan_expiry(self.artifact(path=tree + "/build"), policy)
                self.assertIn("FROZEN production tree", str(ctx.exception))

    def test_denies_a_git_directory(self):
        gitdir = os.path.join(self.owned, "repo", ".git", "objects")
        os.makedirs(gitdir)
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(path=gitdir), self.policy)
        self.assertIn(".git", str(ctx.exception))

    def test_refuses_a_scratch_path(self):
        policy = ST.StoragePolicy(campaign_quota_gb=1.0,
                                  owned_roots=("/mnt/raid0/llm/tmp",))
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(path="/mnt/raid0/llm/tmp/build"), policy)
        self.assertIn("scratch", str(ctx.exception))

    def test_refuses_a_symlink(self):
        real = self.make_tree("real/build")
        link = os.path.join(self.owned, "link")
        os.symlink(real, link)
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(path=link), self.policy)
        self.assertIn("symlink", str(ctx.exception))

    def test_refuses_a_missing_artifact_as_an_unrecorded_loss(self):
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(path=os.path.join(self.owned, "gone")),
                           self.policy)
        self.assertIn("UNRECORDED loss", str(ctx.exception))

    def test_refuses_a_malformed_hash(self):
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(sha256="deadbeef"), self.policy)
        self.assertIn("not WHAT", str(ctx.exception))

    def test_refuses_a_size_that_no_longer_matches(self):
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(declared_size_bytes=7), self.policy)
        self.assertIn("changed since it was recorded", str(ctx.exception))

    def test_refuses_empty_identity_fields(self):
        for name in ("campaign_id", "reason", "rule_id", "actor"):
            with self.subTest(name=name):
                with self.assertRaises(ST.ExpiryRefused):
                    ST.plan_expiry(self.artifact(**{name: "  "}), self.policy)

    def test_rejects_a_non_artifact_argument(self):
        with self.assertRaises(TypeError):
            ST.plan_expiry({"path": self.make_tree()}, self.policy)
        with self.assertRaises(TypeError):
            ST.plan_expiry(self.artifact(), policy={"quota": 1})


class ExpiryPreconditionTest(_ExpiryTest):

    def test_missing_fact_is_refused_not_assumed(self):
        for fact in ST.EXPIRY_RULES["rejected_candidate_build_tree"].required_facts:
            with self.subTest(fact=fact):
                facts = dict(self.artifact().preconditions)
                del facts[fact]
                with self.assertRaises(ST.ExpiryRefused) as ctx:
                    ST.plan_expiry(self.artifact(preconditions=facts), self.policy)
                self.assertIn(fact, str(ctx.exception))
                self.assertIn("not a fact that is true", str(ctx.exception))

    def test_candidate_still_in_the_running_keeps_its_build_tree(self):
        for status in ("banked", "evaluating", "built"):
            with self.subTest(status=status):
                facts = dict(self.artifact().preconditions, candidate_status=status)
                with self.assertRaises(ST.ExpiryRefused):
                    ST.plan_expiry(self.artifact(preconditions=facts), self.policy)

    def test_champion_bearing_candidate_keeps_its_build_tree(self):
        facts = dict(self.artifact().preconditions, champion_status="frontier")
        with self.assertRaises(ST.ExpiryRefused):
            ST.plan_expiry(self.artifact(preconditions=facts), self.policy)

    def test_outcomes_must_be_durable_before_the_bytes_go(self):
        facts = dict(self.artifact().preconditions,
                     evaluation_events_journaled=False)
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(preconditions=facts), self.policy)
        self.assertIn("invariant 7", str(ctx.exception))

    def test_truthy_is_not_true_for_a_journalled_flag(self):
        facts = dict(self.artifact().preconditions,
                     evaluation_events_journaled="yes")
        with self.assertRaises(ST.ExpiryRefused):
            ST.plan_expiry(self.artifact(preconditions=facts), self.policy)

    def test_retired_campaign_worktree_rules(self):
        good = {"campaign_status": "closed", "champion_artifacts_preserved": True,
                "evaluation_events_journaled": True}
        plan = ST.plan_expiry(
            self.artifact(expirable_kind="retired_campaign_worktree",
                          preconditions=good), self.policy, now=NOW)
        self.assertEqual(plan.state, "DRY_RUN")
        for bad in ({"campaign_status": "running"},
                    {"champion_artifacts_preserved": False},
                    {"evaluation_events_journaled": False}):
            with self.subTest(bad=bad):
                with self.assertRaises(ST.ExpiryRefused):
                    ST.plan_expiry(
                        self.artifact(expirable_kind="retired_campaign_worktree",
                                      preconditions=dict(good, **bad)), self.policy)

    def test_profiler_trace_needs_both_age_and_a_closed_lineage(self):
        old_enough = self.policy.min_profiler_trace_age_days + 1
        ok = {"informed_lineage_id": "ak-lineage-1", "lineage_closed": True,
              "trace_age_days": old_enough}
        plan = ST.plan_expiry(
            self.artifact(expirable_kind="stale_profiler_trace", preconditions=ok),
            self.policy, now=NOW)
        self.assertEqual(plan.state, "DRY_RUN")

        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(
                self.artifact(expirable_kind="stale_profiler_trace",
                              preconditions=dict(ok, lineage_closed=False)),
                self.policy)
        self.assertIn("not closed", str(ctx.exception))

        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(
                self.artifact(expirable_kind="stale_profiler_trace",
                              preconditions=dict(ok, trace_age_days=1)),
                self.policy)
        self.assertIn("minimum", str(ctx.exception))


class ExpiryDryRunTest(_ExpiryTest):

    def test_dry_run_is_the_default_and_writes_nothing(self):
        tree = self.make_tree()
        journal = RecordingJournal(watch_path=tree)
        outcome = ST.expire_artifact(self.artifact(path=tree), self.policy,
                                     journal=journal, now=NOW)
        self.assertEqual(outcome.state, "DRY_RUN")
        self.assertFalse(outcome.deleted)
        self.assertEqual(journal.records, [], "a dry run that journals is not a dry run")
        self.assertEqual(outcome.journal_event_ids, ())
        self.assertTrue(os.path.isdir(tree))

    def test_dry_run_still_produces_the_exact_tombstone_it_would_write(self):
        tree = self.make_tree()
        outcome = ST.expire_artifact(self.artifact(path=tree), self.policy, now=NOW)
        tombstone = outcome.tombstone
        self.assertEqual(ST.validate_artifact_tombstone(tombstone), [])
        self.assertEqual(tombstone["artifact_sha256"], _sha("tree"))
        self.assertEqual(tombstone["durability_class"], "hash_and_provenance_only")
        self.assertEqual(tombstone["size_bytes"], outcome.measured_size_bytes)
        self.assertIn("rejected", tombstone["reason"])
        self.assertEqual(tombstone["reclamation_state"], "intent")
        self.assertTrue(os.path.isdir(tree))

    def test_plan_measures_size_rather_than_trusting_the_caller(self):
        tree = self.make_tree()
        plan = ST.plan_expiry(self.artifact(path=tree), self.policy, now=NOW)
        self.assertGreaterEqual(plan.measured_size_bytes, 2048)
        self.assertEqual(plan.measured_file_count, 1)

    def test_naive_now_is_refused(self):
        with self.assertRaises(ValueError):
            ST.plan_expiry(self.artifact(), self.policy,
                           now=datetime(2026, 8, 3, 12, 0, 0))


class ExpiryForceTest(_ExpiryTest):

    def test_tombstone_reaches_the_journal_before_the_bytes_go(self):
        tree = self.make_tree()
        journal = RecordingJournal(watch_path=tree)
        outcome = ST.expire_artifact(self.artifact(path=tree), self.policy,
                                     journal=journal, force=True, now=NOW)
        self.assertEqual(outcome.state, "RECLAIMED")
        self.assertTrue(outcome.deleted)
        self.assertFalse(os.path.lexists(tree))

        self.assertEqual(len(journal.records), 2)
        self.assertEqual(journal.records[0]["reclamation_state"], "intent")
        self.assertEqual(journal.records[1]["reclamation_state"], "reclaimed")
        # The ordering guarantee, observed rather than assumed: the artifact was
        # still on disk when the intent was journalled, and gone by the second.
        self.assertTrue(journal.existed_at_append[0])
        self.assertFalse(journal.existed_at_append[1])

    def test_the_record_survives_the_artifact(self):
        """`MEASUREMENT.md:173-176` — never destroy primary records."""
        tree = self.make_tree()
        journal = RecordingJournal(watch_path=tree)
        ST.expire_artifact(self.artifact(path=tree), self.policy, journal=journal,
                           force=True, now=NOW)
        record = journal.records[-1]
        self.assertFalse(os.path.lexists(record["artifact_path"]))
        self.assertEqual(ST.validate_artifact_tombstone(record), [])
        self.assertEqual(record["artifact_sha256"], _sha("tree"))
        self.assertGreater(record["size_bytes"], 0)
        self.assertEqual(record["reason"], "candidate rejected on correctness")
        self.assertEqual(record["durability_class"], "hash_and_provenance_only")

    def test_both_phases_share_one_tombstone_id(self):
        tree = self.make_tree()
        journal = RecordingJournal(watch_path=tree)
        ST.expire_artifact(self.artifact(path=tree), self.policy, journal=journal,
                           force=True, now=NOW)
        self.assertEqual(journal.records[0]["tombstone_id"],
                         journal.records[1]["tombstone_id"])

    def test_force_without_a_journal_refuses_and_keeps_the_bytes(self):
        tree = self.make_tree()
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.expire_artifact(self.artifact(path=tree), self.policy, force=True,
                               now=NOW)
        self.assertIn("requires a journal", str(ctx.exception))
        self.assertTrue(os.path.isdir(tree))

    def test_a_journal_without_append_is_a_type_error(self):
        tree = self.make_tree()
        with self.assertRaises(TypeError):
            ST.expire_artifact(self.artifact(path=tree), self.policy,
                               journal=object(), force=True, now=NOW)
        self.assertTrue(os.path.isdir(tree))

    def test_a_journal_that_returns_no_event_id_keeps_the_bytes(self):
        for bad in (None, "", "   ", 7):
            with self.subTest(bad=bad):
                tree = self.make_tree(f"cand-{bad!r}/build")
                journal = RecordingJournal(watch_path=tree, event_id=bad)
                with self.assertRaises(ST.ExpiryRefused) as ctx:
                    ST.expire_artifact(self.artifact(path=tree), self.policy,
                                       journal=journal, force=True, now=NOW)
                self.assertIn("not demonstrably durable", str(ctx.exception))
                self.assertTrue(os.path.isdir(tree))

    def test_a_single_file_artifact_is_unlinked(self):
        path = os.path.join(self.owned, "trace.rocprof")
        with open(path, "wb") as fh:
            fh.write(b"0" * 512)
        journal = RecordingJournal(watch_path=path)
        artifact = self.artifact(
            path=path, expirable_kind="stale_profiler_trace",
            preconditions={"informed_lineage_id": "ak-lin-1", "lineage_closed": True,
                           "trace_age_days": 90})
        ST.expire_artifact(artifact, self.policy, journal=journal, force=True, now=NOW)
        self.assertFalse(os.path.lexists(path))

    def test_deletion_failure_journals_failed_and_re_raises(self):
        """The journal must never claim bytes are gone while they are on disk."""
        holder = os.path.join(self.owned, "readonly")
        os.makedirs(holder)
        path = os.path.join(holder, "trace.rocprof")
        with open(path, "wb") as fh:
            fh.write(b"0" * 512)
        self.chmod_temporarily(holder, 0o555)

        journal = RecordingJournal(watch_path=path)
        artifact = self.artifact(
            path=path, expirable_kind="stale_profiler_trace",
            preconditions={"informed_lineage_id": "ak-lin-1", "lineage_closed": True,
                           "trace_age_days": 90})
        with self.assertRaises(OSError):
            ST.expire_artifact(artifact, self.policy, journal=journal, force=True,
                               now=NOW)
        self.assertTrue(os.path.lexists(path))
        self.assertEqual([r["reclamation_state"] for r in journal.records],
                         ["intent", "failed"])
        self.assertTrue(journal.records[-1]["error"])
        self.assertEqual(ST.validate_artifact_tombstone(journal.records[-1]), [])

    def test_refusal_before_the_journal_is_touched(self):
        tree = self.make_tree()
        journal = RecordingJournal(watch_path=tree)
        with self.assertRaises(ST.ExpiryRefused):
            ST.expire_artifact(self.artifact(path=tree, retention_class="permanent_large"),
                               self.policy, journal=journal, force=True, now=NOW)
        self.assertEqual(journal.records, [])
        self.assertTrue(os.path.isdir(tree))


class JournalSinkTest(_ExpiryTest):
    """The storage plane and the AK1 journal must actually compose."""

    class FakeJournal:
        """Mimics `journal.Journal.append(kind, payload, *, campaign_id=...)`."""

        def __init__(self):
            self.calls = []

        def append(self, kind, payload, *, campaign_id=None, record_id=None):
            self.calls.append((kind, dict(payload), campaign_id))
            return type("Entry", (), {"event_id": f"ake-{len(self.calls)}"})()

    def test_adapter_renames_retention_class_to_the_journals_storage_class(self):
        tree = self.make_tree()
        fake = self.FakeJournal()
        sink = ST.JournalTombstoneSink(fake)
        ST.expire_artifact(self.artifact(path=tree), self.policy, journal=sink,
                           force=True, now=NOW)
        self.assertEqual([c[0] for c in fake.calls], ["TOMBSTONE", "TOMBSTONE"])
        for _, payload, campaign in fake.calls:
            self.assertEqual(payload["storage_class"], "expirable")
            self.assertEqual(payload["path"], payload["artifact_path"])
            self.assertEqual(campaign, "ak-llama_gpu-decode-20260803")
        self.assertFalse(os.path.lexists(tree))

    def test_adapter_rejects_an_object_without_append(self):
        with self.assertRaises(TypeError):
            ST.JournalTombstoneSink(object())

    def test_adapter_refuses_a_record_this_modules_validator_rejects(self):
        """The journal's native TOMBSTONE check is deliberately WEAKER.

        It knows nothing of `tombstone_id`, `durability_class`, `actor` or
        `reclamation_state`, so a record carrying this module's schema string
        could reach the primary journal without this module's validator ever
        having seen it. `plan_expiry` validates only the `intent` record; the
        `reclaimed` and `failed` variants are built afterwards by mutating a
        copy, and they went straight through.
        """
        tree = self.make_tree()
        fake = self.FakeJournal()
        sink = ST.JournalTombstoneSink(fake)
        ST.expire_artifact(self.artifact(path=tree), self.policy, journal=sink,
                           force=True, now=NOW)
        good = dict(fake.calls[0][1])
        for key in ("storage_class", "path"):
            good.pop(key, None)

        # Fields the journal's own validator does not inspect at all.
        for mutation in ({"reclamation_state": "vanished"},
                         {"tombstone_id": "not-an-akt-id"},
                         {"actor": ""},
                         {"durability_class": "carried_in_git"}):
            with self.subTest(mutation=mutation):
                broken = {**good, **mutation}
                self.assertNotEqual(ST.validate_artifact_tombstone(broken), [])
                with self.assertRaises(ST.StorageError):
                    sink.append(broken)
        # And the untouched record still goes through, so the guard did not
        # simply refuse everything.
        before = len(fake.calls)
        sink.append(good)
        self.assertEqual(len(fake.calls), before + 1)

    def test_payload_satisfies_the_journals_native_tombstone_contract(self):
        """Pinned against the real journal module rather than a remembered shape."""
        import journal as J  # noqa: PLC0415 — imported here, not at module load

        tree = self.make_tree()
        fake = self.FakeJournal()
        ST.expire_artifact(self.artifact(path=tree), self.policy,
                           journal=ST.JournalTombstoneSink(fake), force=True, now=NOW)
        payload = fake.calls[0][1]
        self.assertEqual(J._validate_native_payload(J.KIND_TOMBSTONE, payload), [])
        self.assertIn(payload["storage_class"], J.TOMBSTONABLE_STORAGE_CLASSES)
        # The §5.8 retention vocabulary is shared, not forked.
        self.assertEqual(J.STORAGE_CLASSES, ST.RETENTION_CLASSES)
        self.assertEqual(ST.JournalTombstoneSink.KIND, J.KIND_TOMBSTONE)

    def test_end_to_end_through_a_real_on_disk_journal(self):
        import journal as J  # noqa: PLC0415

        tree = self.make_tree()
        root = os.path.join(self.tmp, "journal")
        real = J.Journal(root, campaign_id="ak-llama_gpu-decode-20260803")
        real.initialize()
        outcome = ST.expire_artifact(self.artifact(path=tree), self.policy,
                                     journal=ST.JournalTombstoneSink(real),
                                     force=True, now=NOW)
        self.assertTrue(outcome.deleted)
        self.assertFalse(os.path.lexists(tree))
        # The artifact is gone; the record that says why is not.
        events = [e for e in real.read_all() if e.kind == J.KIND_TOMBSTONE]
        self.assertEqual(len(events), 2)
        self.assertEqual([e.payload["reclamation_state"] for e in events],
                         ["intent", "reclaimed"])
        self.assertEqual(events[0].payload["artifact_sha256"], _sha("tree"))
        self.assertEqual(outcome.journal_event_ids,
                         (events[0].event_id, events[1].event_id))


# =============================================================================
# verify_durability — PASS / FAIL / COULD_NOT_CHECK
# =============================================================================

class VerifyDurabilityTest(_TmpTest):

    def setUp(self):
        super().setUp()
        self.tracked = self.write("data/camp/summary.json", b"{}")
        self.untracked = self.write("docs/design/protocol.md", b"# p")
        self.index = ST.StaticTrackedIndex(self.tmp, ["data/camp/summary.json"])

    def verdict(self, citation, **kwargs):
        return ST.verify_durability([citation], **kwargs)[0]

    def test_pass_for_a_tracked_carried_in_git_citation(self):
        v = self.verdict({"path": self.tracked, "durability_class": "carried_in_git"},
                         tracked_index=self.index)
        self.assertEqual(v.outcome, ST.PASS)
        self.assertTrue(v.check.passed)

    def test_fail_when_git_does_not_carry_a_carried_in_git_citation(self):
        v = self.verdict({"path": self.untracked, "durability_class": "carried_in_git"},
                         tracked_index=self.index)
        self.assertEqual(v.outcome, ST.FAIL)

    def test_pass_for_an_existing_durable_untracked_citation(self):
        v = self.verdict({"path": self.untracked,
                          "durability_class": "durable_untracked"},
                         tracked_index=self.index)
        self.assertEqual(v.outcome, ST.PASS)

    def test_fail_for_a_missing_durable_untracked_citation(self):
        v = self.verdict({"path": os.path.join(self.tmp, "docs/design/gone.md"),
                          "durability_class": "durable_untracked"},
                         tracked_index=self.index)
        self.assertEqual(v.outcome, ST.FAIL)

    def test_fail_for_durable_untracked_outside_the_working_tree(self):
        outside = self.write("outside/x.json", b"{}")
        index = ST.StaticTrackedIndex(os.path.join(self.tmp, "repo"), [])
        os.makedirs(os.path.join(self.tmp, "repo"), exist_ok=True)
        v = self.verdict({"path": outside, "durability_class": "durable_untracked"},
                         tracked_index=index)
        self.assertEqual(v.outcome, ST.FAIL)

    def test_hash_and_provenance_only_must_actually_carry_both(self):
        full = {"path": "/mnt/raid0/llm/llama.cpp-ak-1/build",
                "durability_class": "hash_and_provenance_only",
                "sha256": _sha("b"), "provenance": "built 2026-08-03 from ak/…"}
        self.assertEqual(self.verdict(full).outcome, ST.PASS)
        for missing in ("sha256", "provenance"):
            with self.subTest(missing=missing):
                citation = dict(full)
                del citation[missing]
                v = self.verdict(citation)
                self.assertEqual(v.outcome, ST.FAIL)
                self.assertTrue(any(missing in r for r in v.check.reasons))

    def test_unknown_class_fails(self):
        v = self.verdict({"path": self.tracked, "durability_class": "probably_fine"},
                         tracked_index=self.index)
        self.assertEqual(v.outcome, ST.FAIL)

    def test_missing_class_fails(self):
        self.assertEqual(self.verdict({"path": self.tracked}).outcome, ST.FAIL)

    def test_missing_path_fails(self):
        self.assertEqual(
            self.verdict({"durability_class": "carried_in_git"}).outcome, ST.FAIL)

    def test_non_mapping_citation_fails(self):
        self.assertEqual(self.verdict("data/camp/summary.json").outcome, ST.FAIL)

    def test_could_not_check_without_a_tracked_index(self):
        v = self.verdict({"path": self.tracked, "durability_class": "carried_in_git"})
        self.assertEqual(v.outcome, ST.COULD_NOT_CHECK)

    def test_could_not_check_for_durable_untracked_without_an_index(self):
        v = self.verdict({"path": self.untracked,
                          "durability_class": "durable_untracked"})
        self.assertEqual(v.outcome, ST.COULD_NOT_CHECK)

    def test_could_not_check_when_the_path_cannot_be_stat_ed(self):
        """Permission denied is an inability to evaluate, not an absence."""
        closed = os.path.join(self.tmp, "closed")
        os.makedirs(closed)
        hidden = os.path.join(closed, "summary.json")
        with open(hidden, "wb") as fh:
            fh.write(b"{}")
        self.chmod_temporarily(closed, 0o000)
        v = self.verdict({"path": hidden, "durability_class": "durable_untracked"},
                         tracked_index=self.index)
        self.assertEqual(v.outcome, ST.COULD_NOT_CHECK)

    def test_could_not_check_when_tracked_but_absent_from_the_worktree(self):
        os.unlink(self.tracked)
        v = self.verdict({"path": self.tracked, "durability_class": "carried_in_git"},
                         tracked_index=self.index)
        self.assertEqual(v.outcome, ST.COULD_NOT_CHECK)

    def test_could_not_check_is_neither_pass_nor_fail_and_never_truthy(self):
        v = self.verdict({"path": self.tracked, "durability_class": "carried_in_git"})
        self.assertEqual(v.outcome, ST.COULD_NOT_CHECK)
        self.assertNotEqual(v.outcome, ST.PASS)
        self.assertNotEqual(v.outcome, ST.FAIL)
        self.assertFalse(v.check.passed)
        self.assertTrue(v.check.reasons)

    def test_verdicts_are_indexed_and_ordered(self):
        verdicts = ST.verify_durability([
            {"path": self.tracked, "durability_class": "carried_in_git"},
            {"path": "/tmp/x.json", "durability_class": "durable_untracked"},
            {"path": self.untracked, "durability_class": "durable_untracked"},
        ], tracked_index=self.index)
        self.assertEqual([v.index for v in verdicts], [0, 1, 2])
        self.assertEqual([v.outcome for v in verdicts],
                         [ST.PASS, ST.FAIL, ST.PASS])

    def test_empty_input_is_empty_output_not_a_pass(self):
        self.assertEqual(ST.verify_durability([]), ())

    def test_rejects_a_bare_mapping_or_string(self):
        for bad in ({"path": "x"}, "data/x.json"):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    ST.verify_durability(bad)

    def test_check_type_is_the_schemas_check(self):
        v = self.verdict({"path": self.tracked, "durability_class": "carried_in_git"},
                         tracked_index=self.index)
        self.assertIsInstance(v.check, S.Check)


# =============================================================================
# The measured host facts the thresholds were calibrated against
# =============================================================================

class CalibrationTest(unittest.TestCase):
    """These are not arbitrary constants; they encode measurements taken on this
    host on 2026-08-03. The tests pin the RELATIONSHIPS, not the measurements, so
    a future re-measure changes numbers without silently inverting a policy."""

    def test_carry_threshold_sits_between_real_evidence_and_large_artifacts(self):
        largest_tracked_data_file = 37_091_750       # measured: a summary.json
        smallest_permanent_large = 13 * 1024 ** 3    # measured: llama.cpp-experimental
        self.assertGreater(ST.DEFAULT_CARRY_THRESHOLD_BYTES, largest_tracked_data_file)
        self.assertLess(ST.DEFAULT_CARRY_THRESHOLD_BYTES, smallest_permanent_large)

    def test_default_floor_covers_more_than_one_build_worktree(self):
        policy = ST.StoragePolicy(campaign_quota_gb=10.0)
        self.assertGreaterEqual(
            policy.effective_floor_bytes,
            int(ST.DEFAULT_LARGEST_SINGLE_ALLOCATION_GB * 1024 ** 3) * 2)

    def test_default_floor_is_below_currently_measured_free_space(self):
        # 157.7 GiB free measured 2026-08-03. A floor above it would put the loop
        # in DISK_PRESSURE from its first BOOTSTRAP, which is a config bug rather
        # than a host condition.
        self.assertLess(ST.DEFAULT_HEADROOM_FLOOR_GB, 157.0)

    def test_profiler_trace_minimum_age_is_positive(self):
        self.assertGreater(ST.DEFAULT_MIN_PROFILER_TRACE_AGE_DAYS, 0)
        self.assertLess(
            timedelta(days=ST.DEFAULT_MIN_PROFILER_TRACE_AGE_DAYS),
            timedelta(days=365))

    def test_vocabularies_match_the_5_8_table(self):
        self.assertEqual(ST.RETENTION_CLASSES, {
            "permanent_in_repo", "permanent_large", "expirable", "never_stored"})
        self.assertEqual(set(ST.EXPIRY_RULES), set(ST.EXPIRABLE_KINDS))
        self.assertEqual(ST.EXPIRABLE_KINDS, {
            "rejected_candidate_build_tree", "retired_campaign_worktree",
            "stale_profiler_trace"})

    def test_durability_vocabulary_is_owned_by_schemas(self):
        self.assertEqual(S.DURABILITY_CLASSES, {
            "carried_in_git", "durable_untracked", "hash_and_provenance_only"})


# =============================================================================
# Red-team regressions — one case per defect found by adversarial review
# =============================================================================
#
# Every test below FAILED against the module as originally written. They are
# grouped by the axis that found them, and each names the wrong behaviour rather
# than restating the fix, so a regression reads as the loss it is.


class RedTeamProductionContainmentTest(_ExpiryTest):
    """The frozen-tree denial tested only one of the two containment directions.

    `_under(resolved, tree)` asks "is the target INSIDE production". `rmtree` is
    recursive, so a target that CONTAINS production destroys it just as
    thoroughly — and for that case the inside-test never fires, because
    production is the descendant. Measured against the unfixed module:
    `owned_roots=('/mnt/raid0',)` with `path='/mnt/raid0/llm'` returned an
    APPROVED plan for 3,389,981,933,568 bytes spanning all three frozen kernel
    trees. Invariant 3 does not care which way the prefix runs.
    """

    def _sandbox_production(self, *trees):
        original = ST.PRODUCTION_TREES
        ST.PRODUCTION_TREES = tuple(trees)
        self.addCleanup(setattr, ST, "PRODUCTION_TREES", original)

    def test_refuses_an_ancestor_of_a_frozen_production_tree(self):
        parent = os.path.join(self.owned, "llm")
        frozen = os.path.join(parent, "llama.cpp")
        os.makedirs(frozen)
        self._sandbox_production(frozen)
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(path=parent,
                                         expirable_kind="retired_campaign_worktree",
                                         preconditions=_RETIRED_PRECONDITIONS),
                           self.policy)
        self.assertIn("CONTAINS the FROZEN production tree", str(ctx.exception))

    def test_the_ancestor_denial_beats_the_callers_declaration(self):
        """`owned_roots` is the caller's own claim of authority; it must not be
        able to buy a production tree by declaring a root above it."""
        parent = os.path.join(self.owned, "llm")
        frozen = os.path.join(parent, "whisper.cpp")
        os.makedirs(frozen)
        self._sandbox_production(frozen)
        greedy = ST.StoragePolicy(campaign_quota_gb=1.0, owned_roots=(self.tmp,))
        with self.assertRaises(ST.ExpiryRefused):
            ST.plan_expiry(self.artifact(path=parent,
                                         expirable_kind="retired_campaign_worktree",
                                         preconditions=_RETIRED_PRECONDITIONS),
                           greedy)

    def test_the_real_production_trees_are_unreclaimable_from_above(self):
        """No sandboxing: the constants as shipped, against the real host paths."""
        for tree in ST.PRODUCTION_TREES:
            parent = os.path.dirname(tree)
            with self.subTest(parent=parent):
                policy = ST.StoragePolicy(campaign_quota_gb=1.0,
                                          owned_roots=(os.path.dirname(parent),))
                with self.assertRaises(ST.ExpiryRefused) as ctx:
                    ST.plan_expiry(
                        self.artifact(path=parent,
                                      expirable_kind="retired_campaign_worktree",
                                      preconditions=_RETIRED_PRECONDITIONS),
                        policy)
                self.assertIn("FROZEN production tree", str(ctx.exception))

    def test_guard_roots_are_matched_through_symlinks_too(self):
        """Every path under test has been realpath'd; the guard ROOTS had not.

        None of them is a symlink on this host today, so this is latent — but
        this repository's working-tree identity rule makes `/workspace/repos/<n>`
        a symlink onto `/mnt/raid0/llm/<n>` (CLAUDE.md), and a literal prefix
        test against a linked root fails OPEN rather than closed.
        """
        real = os.path.join(self.tmp, "real-production")
        os.makedirs(os.path.join(real, "build"))
        link = os.path.join(self.tmp, "linked-production")
        os.symlink(real, link)
        self._sandbox_production(link)  # declared via the LINK
        policy = ST.StoragePolicy(campaign_quota_gb=1.0, owned_roots=(self.tmp,))
        with self.assertRaises(ST.ExpiryRefused) as ctx:
            ST.plan_expiry(self.artifact(path=os.path.join(real, "build")), policy)
        self.assertIn("FROZEN production tree", str(ctx.exception))

    def test_root_match_forms_carries_both_the_link_and_its_target(self):
        real = os.path.join(self.tmp, "target")
        os.makedirs(real)
        link = os.path.join(self.tmp, "alias")
        os.symlink(real, link)
        forms = ST._root_match_forms((link,))
        self.assertIn(link, forms)
        self.assertIn(real, forms)
        # A root that is not a link contributes exactly one form, so matching
        # cannot get quietly broader on the ordinary case.
        self.assertEqual(ST._root_match_forms((real,)), (real,))

    def test_scratch_roots_are_matched_in_both_forms(self):
        self.assertEqual(
            set(ST._EPHEMERAL_ROOT_FORMS) & set(ST.EPHEMERAL_ROOTS),
            set(ST.EPHEMERAL_ROOTS),
            "every declared scratch root must still be matched literally")


_RETIRED_PRECONDITIONS = {
    "campaign_status": "closed",
    "champion_artifacts_preserved": True,
    "evaluation_events_journaled": True,
}


class RedTeamLayoutCheckCannotBePassedByDeletionTest(_TmpTest):
    """This project's standing screen: *can I pass this check by deleting the
    thing it inspects?*

    For `check_evidence_root_layout` the answer was yes, three ways. The
    compliance test was `README_STUB_MARKER in text` — presence of one HTML
    comment the module had written itself. Delete that single line and a README
    whose three answer cells still read "TODO — fill in" scored PASS; truncate
    the file to zero bytes and it scored PASS; point it at /dev/null and it
    scored PASS. `MEASUREMENT.md:146-156` requires the README to state what was
    measured, when, and which claim it backs, and none of those three states
    states anything.
    """

    def _root(self):
        return ST.ensure_campaign_evidence_root("ak-camp", repo_root=self.tmp)

    def test_deleting_only_the_stub_marker_does_not_confer_compliance(self):
        root = self._root()
        self.assertEqual(root.layout.outcome, ST.FAIL)
        with open(root.readme_path, encoding="utf-8") as fh:
            text = fh.read()
        with open(root.readme_path, "w", encoding="utf-8") as fh:
            fh.write(text.replace(ST.README_STUB_MARKER, ""))
        check = ST.check_evidence_root_layout(root.path)
        self.assertEqual(check.outcome, ST.FAIL)
        self.assertIn(ST.README_PLACEHOLDER, " ".join(check.reasons))
        # The unanswered questions are still verbatim in the file.
        with open(root.readme_path, encoding="utf-8") as fh:
            self.assertIn(ST.README_PLACEHOLDER, fh.read())

    def test_an_empty_readme_is_not_a_compliant_readme(self):
        root = self._root()
        with open(root.readme_path, "w", encoding="utf-8"):
            pass
        check = ST.check_evidence_root_layout(root.path)
        self.assertEqual(check.outcome, ST.FAIL)
        self.assertIn("empty", " ".join(check.reasons))

    def test_a_readme_symlinked_to_dev_null_is_not_a_compliant_readme(self):
        root = self._root()
        os.unlink(root.readme_path)
        os.symlink("/dev/null", root.readme_path)
        self.assertEqual(ST.check_evidence_root_layout(root.path).outcome, ST.FAIL)

    def test_a_hand_written_readme_still_passes(self):
        """The screen must not become a format lock: a human README that answers
        the three questions in prose is compliant."""
        root = self._root()
        with open(root.readme_path, "w", encoding="utf-8") as fh:
            fh.write("# ak-camp\n\nllama-bench decode t/s, 5 reps, 2026-08-03T12:00Z,"
                     " backing P-AK-SEARCH-1 against the v8 anchor.\n")
        self.assertEqual(ST.check_evidence_root_layout(root.path).outcome, ST.PASS)

    def test_a_non_empty_but_hashless_manifest_is_not_checkable(self):
        """Size was the whole manifest test, so one space made a root compliant.

        `sha256sum -c` is what makes the artifacts beside it verifiable, and it
        has nothing to consume in a file with no `<sha256>  <name>` line.
        """
        root = ST.ensure_campaign_evidence_root(
            "ak-camp2", repo_root=self.tmp, claim="c", what_was_measured="w",
            measured_at="2026-08-03T00:00:00Z")
        with open(os.path.join(root.path, "summary.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        for content in (" \n", "# generated later\n", "summary.json\n"):
            with self.subTest(content=content):
                with open(root.sha256sums_path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                check = ST.check_evidence_root_layout(root.path)
                self.assertEqual(check.outcome, ST.FAIL)
                self.assertIn("checkable", " ".join(check.reasons))
        with open(root.sha256sums_path, "w", encoding="utf-8") as fh:
            fh.write(f"{_sha('summary')}  summary.json\n")
        self.assertEqual(ST.check_evidence_root_layout(root.path).outcome, ST.PASS)


class RedTeamNaNBoundsTest(unittest.TestCase):
    """A NaN does not violate a bound — it deletes it.

    `allocation_safety_factor` sat outside the module's own finite screen. NaN
    passes `< 1` (every NaN comparison is False) and `max(floor, step * nan)`
    returns the declared floor, so `effective_floor_bytes` silently reverted to
    the very number the stepped floor exists to override: a 5 GiB floor on a host
    whose next legal action writes a 15 GiB worktree. No exception, no warning,
    no reason string — the guarantee simply stopped existing.
    """

    def test_nan_allocation_safety_factor_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ST.StoragePolicy(campaign_quota_gb=1.0, headroom_floor_gb=5.0,
                             largest_single_allocation_gb=15.0,
                             allocation_safety_factor=float("nan"))
        self.assertIn("NaN", str(ctx.exception))

    def test_nan_would_have_silently_restored_the_bare_declared_floor(self):
        """Pins the harm, not just the rejection: a valid factor MUST lift the
        floor above the declared one, so a bypass is observable as a number."""
        policy = ST.StoragePolicy(campaign_quota_gb=1.0, headroom_floor_gb=5.0,
                                  largest_single_allocation_gb=15.0,
                                  allocation_safety_factor=2.0)
        self.assertEqual(policy.effective_floor_bytes, int(30 * 1024 ** 3))
        self.assertGreater(policy.effective_floor_bytes, int(5 * 1024 ** 3))

    def test_infinite_factor_is_rejected_at_construction_not_at_use(self):
        """It used to survive `__post_init__` and raise OverflowError inside
        `effective_floor_bytes`, far from the caller that supplied it."""
        with self.assertRaises(ValueError):
            ST.StoragePolicy(campaign_quota_gb=1.0,
                             allocation_safety_factor=float("inf"))

    def test_nan_quota_warn_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            ST.StoragePolicy(campaign_quota_gb=1.0, quota_warn_fraction=float("nan"))

    def test_bool_min_trace_age_is_rejected(self):
        """`True` is an int in Python: it would have become a 1-day minimum on a
        policy field whose default is 30 days."""
        with self.assertRaises(TypeError):
            ST.StoragePolicy(campaign_quota_gb=1.0, min_profiler_trace_age_days=True)


class RedTeamExpiryWritePathTest(_ExpiryTest):
    """Two asymmetries on the force path, both success-shaped."""

    def test_a_completion_record_with_no_event_id_is_not_a_clean_reclamation(self):
        """The INTENT id was validated and the completion id was not.

        A journal returning None therefore produced `state="RECLAIMED"`,
        `deleted=True`, `journal_event_ids=("ake-0001", None)`: a success-shaped
        result asserting a durable completion record that was never
        demonstrated. The bytes are already gone so this cannot be undone, but it
        must not be REPORTED as clean — the intent record is the recovery anchor.
        """
        tree = self.make_tree()

        class OneIdJournal:
            def __init__(self):
                self.records = []

            def append(self, record):
                self.records.append(dict(record))
                return "ake-0001" if len(self.records) == 1 else None

        journal = OneIdJournal()
        with self.assertRaises(ST.StorageError) as ctx:
            ST.expire_artifact(self.artifact(path=tree), self.policy,
                               journal=journal, force=True, now=NOW)
        self.assertIn("not demonstrably durable", str(ctx.exception))
        self.assertIn("ake-0001", str(ctx.exception))
        self.assertEqual([r["reclamation_state"] for r in journal.records],
                         ["intent", "reclaimed"])
        self.assertFalse(os.path.lexists(tree))

    def test_a_failing_journal_does_not_mask_the_deletion_error(self):
        """On a deletion failure the module appends a `failed` record and
        re-raises. When that append itself raised, the journal's exception
        REPLACED the OSError — so an `except OSError` caller, the only one who
        would know the bytes are still on disk, saw a RuntimeError instead and
        the real cause survived only in `__context__`.
        """
        tree = self.make_tree()

        class BrokenJournal:
            def __init__(self):
                self.n = 0

            def append(self, record):
                self.n += 1
                if self.n == 1:
                    return "ake-0001"
                raise RuntimeError("journal shard is full")

        with unittest.mock.patch("shutil.rmtree",
                                 side_effect=OSError("EBUSY: device or resource busy")):
            with self.assertRaises(OSError) as ctx:
                ST.expire_artifact(self.artifact(path=tree), self.policy,
                                   journal=BrokenJournal(), force=True, now=NOW)
        self.assertIn("EBUSY", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, RuntimeError)
        # The bytes the caller must be told about are still there.
        self.assertTrue(os.path.isdir(tree))


class RedTeamVerifyDurabilityTest(_TmpTest):
    """`verify_durability` is the function that runs over a registry, and it was
    the one with the laundering door open."""

    def setUp(self):
        super().setUp()
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(os.path.join(self.repo, "data", "camp"), exist_ok=True)
        self.gone = os.path.join(self.repo, "data", "camp", "lost.json")
        self.index = ST.StaticTrackedIndex(self.repo, ["data/camp/lost.json"])

    def test_a_lost_tracked_artifact_cannot_be_laundered_by_relabelling(self):
        """`classify()` REFUSES to derive `hash_and_provenance_only` from
        absence, on the stated grounds that it "would relabel every loss as an
        intended design decision". The verifier then accepted exactly the claim
        classify refuses to produce: the SAME missing, git-tracked file scored
        COULD_NOT_CHECK as `carried_in_git`, FAIL as `durable_untracked`, and
        PASS as `hash_and_provenance_only`. Relabelling was the key to the door
        §3.7 exists to lock.
        """
        as_carried = ST.verify_durability(
            [{"path": self.gone, "durability_class": "carried_in_git"}],
            tracked_index=self.index)[0]
        as_untracked = ST.verify_durability(
            [{"path": self.gone, "durability_class": "durable_untracked"}],
            tracked_index=self.index)[0]
        as_hash_only = ST.verify_durability(
            [{"path": self.gone, "durability_class": "hash_and_provenance_only",
              "sha256": _sha("lost"), "provenance": "measured 2026-08-01"}],
            tracked_index=self.index)[0]
        self.assertEqual(as_carried.outcome, ST.COULD_NOT_CHECK)
        self.assertEqual(as_untracked.outcome, ST.FAIL)
        self.assertNotEqual(as_hash_only.outcome, ST.PASS)
        self.assertIn("git", " ".join(as_hash_only.check.reasons).lower())

    def test_hash_only_for_an_absent_in_repo_path_is_undetermined_not_passing(self):
        """The class is a claim about SIZE — too large to carry. An absent
        artifact cannot substantiate it, and whether that absence is expected or
        a loss is precisely the question §3.7 was invented to answer."""
        untracked_index = ST.StaticTrackedIndex(self.repo, ["data/camp/other.json"])
        verdict = ST.verify_durability(
            [{"path": self.gone, "durability_class": "hash_and_provenance_only",
              "sha256": _sha("lost"), "provenance": "p"}],
            tracked_index=untracked_index)[0]
        self.assertEqual(verdict.outcome, ST.COULD_NOT_CHECK)
        self.assertFalse(verdict.check.passed)

    def test_hash_only_for_a_small_present_in_repo_file_is_a_misdeclaration(self):
        small = os.path.join(self.repo, "data", "camp", "small.json")
        with open(small, "w", encoding="utf-8") as fh:
            fh.write("{}")
        index = ST.StaticTrackedIndex(self.repo, ["data/camp/other.json"])
        verdict = ST.verify_durability(
            [{"path": small, "durability_class": "hash_and_provenance_only",
              "sha256": _sha("small"), "provenance": "p"}],
            tracked_index=index)[0]
        self.assertEqual(verdict.outcome, ST.FAIL)
        self.assertIn("durable_untracked", " ".join(verdict.check.reasons))

    def test_hash_only_outside_every_working_tree_still_passes(self):
        """The class is CORRECT there — nothing versions those bytes. The fix
        must not turn a legitimate declaration into a failure."""
        outside = os.path.join(self.tmp, "elsewhere", "build")
        os.makedirs(outside, exist_ok=True)
        verdict = ST.verify_durability(
            [{"path": outside, "durability_class": "hash_and_provenance_only",
              "sha256": _sha("build"), "provenance": "built from ak/…"}],
            tracked_index=self.index)[0]
        self.assertEqual(verdict.outcome, ST.PASS)

    def test_hash_only_without_an_index_is_unchanged(self):
        """With nothing to contradict the claim, the field check is all there
        is; inventing a verdict from no evidence is the opposite of the fix."""
        verdict = ST.verify_durability(
            [{"path": "/mnt/raid0/llm/llama.cpp-ak-1/build",
              "durability_class": "hash_and_provenance_only",
              "sha256": _sha("b"), "provenance": "built 2026-08-03"}])[0]
        self.assertEqual(verdict.outcome, ST.PASS)

    def test_one_unanswerable_citation_does_not_destroy_the_batch(self):
        """A `TrackedIndex` RAISES when it cannot answer — that is the documented
        reason it never guesses. The `durable_untracked` branch called it
        unguarded while the `carried_in_git` branch caught it, so a single
        disclaimed citation aborted the whole call and took every other verdict
        with it, including the FAILs. A per-citation verifier owes a verdict per
        citation.
        """
        class DisclaimingIndex(ST.TrackedIndex):
            def contains_repo(self, path):
                raise ST.UnclassifiablePath("this index cannot answer for that path")

            def is_tracked(self, path):
                raise ST.UnclassifiablePath("this index cannot answer for that path")

        present = os.path.join(self.repo, "data", "camp", "here.json")
        with open(present, "w", encoding="utf-8") as fh:
            fh.write("{}")
        verdicts = ST.verify_durability([
            {"path": present, "durability_class": "durable_untracked"},
            {"path": self.gone, "durability_class": "durable_untracked"},
            {"path": present, "durability_class": "carried_in_git"},
        ], tracked_index=DisclaimingIndex())
        self.assertEqual(len(verdicts), 3)
        self.assertEqual(verdicts[0].outcome, ST.COULD_NOT_CHECK)
        self.assertEqual(verdicts[1].outcome, ST.FAIL)   # absent: no index needed
        self.assertEqual(verdicts[2].outcome, ST.COULD_NOT_CHECK)
        for verdict in verdicts:
            self.assertFalse(verdict.check.passed)

    def test_a_module_bug_still_escapes_rather_than_becoming_could_not_check(self):
        """The per-citation guard must stay narrow. Swallowing every exception
        into COULD_NOT_CHECK is the fail-open shape this whole module exists to
        avoid — it would convert a bug in this file into a soft verdict."""
        class ExplodingIndex(ST.TrackedIndex):
            def contains_repo(self, path):
                raise TypeError("programming error, not an unanswerable question")

            def is_tracked(self, path):
                raise TypeError("programming error")

        present = os.path.join(self.repo, "data", "camp", "here.json")
        with open(present, "w", encoding="utf-8") as fh:
            fh.write("{}")
        with self.assertRaises(TypeError):
            ST.verify_durability(
                [{"path": present, "durability_class": "durable_untracked"}],
                tracked_index=ExplodingIndex())


class RedTeamIndexAndManifestTest(_TmpTest):

    def test_the_working_tree_root_is_tracked_when_git_carries_anything(self):
        """`os.path.relpath(root, root)` is ".", and `git ls-files` never emits a
        leading "./", so the prefix test could not match and a repository
        tracking thousands of files reported its OWN ROOT untracked —
        `classify()` then handed the whole tree `durable_untracked`, and a
        `carried_in_git` citation of the root FAILed."""
        os.makedirs(os.path.join(self.tmp, "repo", "data"), exist_ok=True)
        repo = os.path.join(self.tmp, "repo")
        index = ST.StaticTrackedIndex(repo, ["data/a.json", "README.md"])
        self.assertTrue(index.is_tracked(repo))
        empty = ST.StaticTrackedIndex(repo, [])
        self.assertFalse(empty.is_tracked(repo))

    def test_root_classifies_as_carried_in_git(self):
        repo = os.path.join(self.tmp, "repo2")
        os.makedirs(repo, exist_ok=True)
        with open(os.path.join(repo, "a.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        index = ST.StaticTrackedIndex(repo, ["a.json"])
        self.assertEqual(ST.classify(repo, tracked_index=index).durability_class,
                         "carried_in_git")

    def test_an_empty_directory_changes_the_tree_manifest(self):
        """The manifest was `{relpath: sha}` over files and symlinks only, so a
        build tree carrying an extra EMPTY directory hashed byte-identically to
        one without it. That hash IS the identity a tombstone records for a tree
        that is about to stop existing, and two materially different trees shared
        it."""
        for name in ("t1", "t2"):
            os.makedirs(os.path.join(self.tmp, name), exist_ok=True)
            with open(os.path.join(self.tmp, name, "a.bin"), "wb") as fh:
                fh.write(b"a")
        os.makedirs(os.path.join(self.tmp, "t2", "empty-artifacts"))
        self.assertNotEqual(ST.hash_tree_manifest(os.path.join(self.tmp, "t1")),
                            ST.hash_tree_manifest(os.path.join(self.tmp, "t2")))

    def test_the_manifest_is_still_stable_and_content_sensitive(self):
        self.write("t3/a.bin", b"a")
        self.write("t3/sub/b.bin", b"b")
        root = os.path.join(self.tmp, "t3")
        first = ST.hash_tree_manifest(root)
        self.assertEqual(first, ST.hash_tree_manifest(root))
        self.write("t3/sub/b.bin", b"c")
        self.assertNotEqual(first, ST.hash_tree_manifest(root))


class RedTeamEvidenceRootRaceTest(_TmpTest):
    """`open(..., "x")` is the right primitive — the racing process, not the
    `lexists()` above it, decides who wins — but the `FileExistsError` it raises
    was uncaught, so two sessions opening the same campaign made a function
    documented as *idempotent* raise a bare OSError. Losing the race IS the
    idempotent outcome: the file the caller asked for exists."""

    def test_losing_the_creation_race_is_not_an_error(self):
        real_lexists = os.path.lexists
        raced = {"done": False}

        def racy_lexists(path):
            # Report "absent" once, then let the competitor create it, exactly as
            # a second process interleaving between the check and the open would.
            if path.endswith(ST.SHA256SUMS_NAME) and not raced["done"]:
                raced["done"] = True
                with open(path, "w", encoding="utf-8"):
                    pass
                return False
            return real_lexists(path)

        os.makedirs(os.path.join(self.tmp, "data", "ak-camp"), exist_ok=True)
        with unittest.mock.patch("os.path.lexists", racy_lexists):
            root = ST.ensure_campaign_evidence_root("ak-camp", repo_root=self.tmp)
        self.assertTrue(raced["done"])
        self.assertTrue(os.path.isfile(root.sha256sums_path))

    def test_the_competitors_content_is_never_clobbered(self):
        first = ST.ensure_campaign_evidence_root("ak-camp2", repo_root=self.tmp)
        with open(first.readme_path, "w", encoding="utf-8") as fh:
            fh.write("# written by the other session\n")
        second = ST.ensure_campaign_evidence_root("ak-camp2", repo_root=self.tmp)
        with open(second.readme_path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "# written by the other session\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
