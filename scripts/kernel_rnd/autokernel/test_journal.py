"""Unit tests for autokernel/journal.py — the append-only sharded primary record.

Local filesystem only. NO inference, NO server, NO model, NO GPU, NO process is
started or signalled — every case writes a handful of JSON lines into a
`tempfile.mkdtemp()` tree and deletes it again, which is why this suite is safe
to run on the shared host at any time.

The suite is organised around the failures the journal exists to prevent
(handoff `autokernel-research-loop.md` §5.5, invariants 7/8/19/20):

  * **The three historical shard-reading bugs**, each reproduced against a naive
    reader written inline in the test and then shown to be impossible for
    `Journal.shards()`: a base-only read that froze on pre-rotation data, a
    lexicographic sort that ordered `_10` before `_2`, and a while-loop discovery
    that stopped at the first missing index.
  * **A crash mid-write.** Acknowledged (fsynced) events survive; the torn tail is
    discarded loudly, with a `TORN_APPEND_DISCARDED` event, before the next
    append can be concatenated onto it.
  * **Retrieval-scope supersession.** Hidden from `retrieve()`, present in
    `read_all()` — the asymmetry is the feature, so both directions are asserted.
  * **Narrative exclusion.** Stripped by default at every depth; admitted only for
    an explicitly cited event id; a cited-but-withdrawn belief RAISES.
  * **The consistency assertion.** A journal holding candidates whose rebuilt view
    is empty raises; a deliberate rebase with a stated reason does not.

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/test_journal.py
    python3 -W error::ResourceWarning -m unittest scripts/kernel_rnd/autokernel/test_journal.py
    python3 scripts/kernel_rnd/autokernel/test_journal.py
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import journal as J  # noqa: E402
import schemas as S  # noqa: E402


V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
CAMPAIGN = "ak-llama_gpu-decode-20260803"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


# =============================================================================
# Fixtures — minimal records that pass schemas.py, so `append()` accepts them
# =============================================================================

def _campaign() -> dict:
    return {
        "schema": S.SCHEMA_CAMPAIGN,
        "campaign_id": CAMPAIGN,
        "backend": "llama_gpu",
        "source_tree": "llama.cpp",
        "production_anchor": {
            "repo": "/mnt/raid0/llm/llama.cpp",
            "branch": "production-consolidated-v8",
            "commit": V8_COMMIT,
        },
        "objective": {
            "rule": "per_phase_non_inferiority_plus_improvement",
            "phases": ["prefill", "decode"],
            "protocol_by_phase": {"prefill": "P-BENCH-PREFILL-1", "decode": "P-BENCH-1"},
            "recipe_class": "production_optimal",
            "phase_trade_exception": None,
            "target_regimes": [],
        },
        "scope": {
            "affected_ops": [],
            "affected_arch_classes": [],
            "derived_role_manifest_sha256": _sha("role-manifest"),
        },
        "policy_ref": {
            "search_protocol": "P-AK-SEARCH-1/v1",
            "release_protocol": "P-KERNEL-FREEZE-1/v1",
            "policy_bundle_sha256": _sha("policy-bundle"),
        },
        "budgets": {
            "max_wall_hours": 48.0,
            "max_gpu_hours": 12.0,
            "max_cpu_region_hours": 0.0,
            "max_candidates": 40,
            "max_controller_tokens": 4_000_000,
            "max_storage_gb": 60.0,
        },
        "readiness_reporting": {"reference_point_gain": 0.25, "reference_lcb_gain": 0.20},
        "stop_policy": {
            "plateau_rounds": 6,
            "max_consecutive_integrity_failures": 2,
            "max_consecutive_build_failures": 3,
            "max_command_retries": 3,
        },
    }


def _candidate(suffix: str = "0001", status: str = "built", **extra) -> dict:
    record = {
        "schema": S.SCHEMA_CANDIDATE,
        "candidate_id": f"akc-20260803-{suffix}",
        "campaign_id": CAMPAIGN,
        "proposal_id": "akp-20260803-0001",
        "parent_candidate_id": None,
        "worktree": {
            "path": "/mnt/raid0/llm/llama.cpp-ak-llama_gpu-decode-20260803",
            "branch": f"ak/{CAMPAIGN}/akp-{suffix}",
            "source_commit": V7_COMMIT,
            "clean": True,
        },
        "source_snapshot": {
            "snapshot_sha256": _sha(f"snapshot-{suffix}"),
            "patch_bundle_sha256": _sha(f"patch-{suffix}"),
        },
        "ancestry": {
            "production_base_commit": V8_COMMIT,
            "is_descendant_of_production_base": True,
            "proof": "git merge-base --is-ancestor 67a433bf.. HEAD -> 0",
        },
        "build": {
            "toolchain": "rocm-6.2",
            "compiler": "hipcc 6.2.0",
            "command": "cmake --build build -j 96",
            "build_dir": f"/mnt/raid0/llm/tmp/ak-build/akc-{suffix}",
            "log_path": f"data/{CAMPAIGN}/build/akc-{suffix}.log",
            "log_sha256": _sha(f"build-log-{suffix}"),
        },
        "artifacts": {
            "binary_sha256": _sha(f"binary-{suffix}"),
            "linkage_sha256": _sha(f"linkage-{suffix}"),
            "library_sha256s": {"libggml.so": _sha("libggml")},
        },
        "dispatch": {"feature_flags": ["GGML_AK_WIDE_TILE"], "dispatch_predicate": "K >= 4096"},
        "affected_surface": {
            "derived_sha256": _sha("derived-surface"),
            "traced_sha256": None,
            "reconciled": False,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": _sha("evaluator-bundle")},
        "receipts": {
            "host_receipt": "rcpt-host-20260803T101500Z",
            "resource_claim_receipt": "rcpt-gpu-claim-0042",
        },
        "storage": {"footprint_gb": 3.4, "durability_class": "hash_and_provenance_only"},
        "evaluation_event_ids": [],
        "derived_verdicts": {},
        "controller": {
            "provider": "local",
            "model_id": "architect-a4",
            "effort": "high",
            "prompt_bundle_sha256": _sha("prompt-bundle"),
        },
        "champion_status": "none",
        "status": status,
        "supersession_reason": None,
        "created_at": "2026-08-03T10:15:00+00:00",
    }
    record.update(extra)
    return record


def _event(suffix: str = "0001") -> dict:
    return {
        "schema": S.SCHEMA_EVALUATION_EVENT,
        "event_id": f"ake-20260803-{suffix}",
        "campaign_id": CAMPAIGN,
        "candidate_id": "akc-20260803-0001",
        "tier": "T1",
        "change_class": "parameter",
        "anchor_tier": "T1",
        "transfer_ratio_to": [],
        "backend": "llama_gpu",
        "device_state": {
            "device_id": "mi210_0", "source": "fixture/rocm-smi",
            "nominal_sclk_mhz": 1700.0, "min_sclk_ratio": 0.9,
            "samples": [{"sclk_mhz": 1700.0, "mclk_mhz": 1600.0,
                         "power_w": 180.0, "temperature_c": 55.0,
                         "under_measurement_load": True}],
            "throttle_observed": False,
            "receipt_ref": "fixture://device-state/journal",
        },
        "claim_grammar": {
            "category": "CANDIDATE",
            "protocol_id": "P-AK-SEARCH-1/v1",
            "metric": "decode_tokens_per_s",
            "metric_direction": "higher_better",
            "reps": 5,
            "attestation_ref": "rcpt-host-20260803T101500Z",
        },
        "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": _sha("evaluator-bundle")},
        "artifact": {
            "source_sha256": _sha("snapshot-0001"),
            "binary_sha256": _sha("binary-0001"),
            "linkage_sha256": _sha("linkage-0001"),
        },
        "anchor": {
            "source_commit": V8_COMMIT,
            "binary_sha256": _sha("anchor-binary"),
            "linkage_sha256": _sha("anchor-linkage"),
            "measurement_event_ids": ["ake-20260801-0009"],
        },
        "scope_manifest_sha256": _sha("scope-manifest"),
        "host_receipt": "rcpt-host-20260803T101500Z",
        "resource_claim_receipt": "rcpt-gpu-claim-0042",
        "co_residency": "single",
        "correctness": {"test_backend_ops": "pass"},
        "quality": {},
        "stability": {},
        "scope_denominator": {
            "machine_subset": "partial",
            "numa_nodes": [0],
            "devices": ["gfx90a:0"],
            "cores": 8,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "performance": {
            "raw_samples": [51.2, 51.4, 51.1],
            "paired_blocks": 3,
            "estimate": 51.23,
            "uncertainty": {"e_process_value": 12.4},
        },
        "mechanism": {},
        "integrity_flags": [],
        "status": "pass",
        "supersedes": [],
        "created_at": "2026-08-03T10:45:00+00:00",
    }


def _champion() -> dict:
    return {
        "schema": S.SCHEMA_CHAMPION,
        # A champion states the build it was compiled with, not only its tree.
        "build_recipe": {
            "schema": "epyc.autokernel.gpu_build_recipe.v1",
            "name": "gfx90a-house-v1",
            "production_reference_is_verifiable": False,
            "flags": [{"name": "GGML_HIP", "value": "ON",
                       "production_value": "ON", "diverges": False,
                       "reason": None}],
            "divergences": [],
            "notes": None,
        },
        "source_tree": "llama.cpp",
        "anchor_commit": V8_COMMIT,
        "branch": "ak/champion/llama-20260802",
        "member_candidates": ["akc-20260803-0001"],
        "combined_candidate_id": "akc-20260803-0009",
        "last_t0": {"event_id": "ake-20260803-0002", "status": "pass"},
        "last_t1": {"event_id": "ake-20260803-0001", "status": "pass"},
        "last_t2": None,
        "readiness": {
            "by_backend": {"llama_gpu": {"prefill": {}, "decode": {}}},
            "reference_signal": "point +2.1% / LCB +0.8% versus anchor on 6 cells",
        },
        "affected_surface_union_sha256": _sha("surface-union"),
        "storage_gb": 12.0,
        "blocking_conditions": [],
    }


def _skip(reason: str = "filtered by the pre-run critic") -> dict:
    return {"proposal_ref": "akp-20260803-0002", "reason": reason}


# =============================================================================
# Naive readers — the historically-shipped bugs, written out so the test can
# show each one losing data on a journal this module reads correctly.
# =============================================================================

def _base_only_read(root: str) -> list:
    """Bug 1: read `events.jsonl` and call it the history."""
    path = os.path.join(root, J.BASE_SHARD_NAME)
    with open(path, "rb") as fh:
        data = fh.read()
    return [json.loads(line) for line in data.split(b"\n") if line.strip()]


def _lexicographic_shard_names(root: str) -> list:
    """Bug 2: order shard FILENAMES as strings."""
    return sorted(n for n in os.listdir(root) if J._SHARD_RE.match(n))


def _while_loop_shard_paths(root: str) -> list:
    """Bug 3: probe `_1, _2, _3 …` and stop at the first missing index."""
    paths = [os.path.join(root, J.BASE_SHARD_NAME)]
    index = 1
    while os.path.exists(os.path.join(root, f"events_{index}.jsonl")):
        paths.append(os.path.join(root, f"events_{index}.jsonl"))
        index += 1
    return paths


# =============================================================================
# Base fixture
# =============================================================================

class _JournalTest(unittest.TestCase):
    max_shard_bytes = J.DEFAULT_MAX_SHARD_BYTES

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ak_journal_test_")
        self.root = os.path.join(self.tmp, "journal")
        self.j = J.Journal(self.root, campaign_id=CAMPAIGN,
                           max_shard_bytes=self.max_shard_bytes)
        self.j.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _new_journal(self, **kwargs):
        """A SECOND Journal object over the same root — a restart, in effect."""
        kwargs.setdefault("campaign_id", CAMPAIGN)
        kwargs.setdefault("max_shard_bytes", self.max_shard_bytes)
        return J.Journal(self.root, **kwargs)


# =============================================================================
# Sharding: the three bugs that cost this project data
# =============================================================================

class TestShardReading(_JournalTest):
    # One event per shard, so rotation is easy to force deterministically.
    max_shard_bytes = 1

    def _append_n(self, n: int) -> list:
        return [self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip(f"reason {i}"))
                for i in range(n)]

    def test_bug1_base_only_read_freezes_at_pre_rotation_state(self):
        """A base-only reader sees ONE event; the journal holds five.

        This is the `feedback_autopilot_journal_rotation_read_all_shards` scar:
        the base file froze at trial 999 while the live run wrote 1073 into `_1`,
        and every panel reading the base showed five-day-old data while reporting
        itself healthy.
        """
        self._append_n(5)
        self.assertEqual(len(_base_only_read(self.root)), 1)
        self.assertEqual(len(self.j.read_all()), 5)
        self.assertEqual([e.seq for e in self.j.read_all()], [1, 2, 3, 4, 5])

    def test_bug2_lexicographic_order_puts_10_before_2(self):
        """`_10` sorts before `_2` as a string and after it as an integer."""
        self._append_n(11)
        names = _lexicographic_shard_names(self.root)
        self.assertLess(names.index("events_10.jsonl"), names.index("events_2.jsonl"),
                        "precondition: filename sort really does misorder _10")

        indices = [s.index for s in self.j.shards()]
        self.assertEqual(indices, list(range(11)))
        self.assertEqual([e.seq for e in self.j.read_all()], list(range(1, 12)))
        # And the payloads follow the same order — not just the counters.
        self.assertEqual(
            [e.payload["reason"] for e in self.j.read_all()],
            [f"reason {i}" for i in range(11)],
        )

    def test_bug3_while_loop_discovery_stops_at_the_first_hole(self):
        """A hole must RAISE, not terminate discovery.

        The naive probe returns shards 0..1 and silently drops shard 3's events;
        `shards()` names the missing index instead.
        """
        self._append_n(4)                                   # shards 0,1,2,3
        os.unlink(os.path.join(self.root, "events_2.jsonl"))

        naive = _while_loop_shard_paths(self.root)
        self.assertEqual(len(naive), 2, "precondition: the probe stops at the hole")
        self.assertNotIn(os.path.join(self.root, "events_3.jsonl"), naive)

        with self.assertRaises(J.ShardGapError) as ctx:
            self.j.shards()
        self.assertIn("[2]", str(ctx.exception))
        with self.assertRaises(J.ShardGapError):
            self.j.read_all()

    def test_missing_base_shard_is_a_gap_not_a_new_journal(self):
        self._append_n(3)
        os.unlink(os.path.join(self.root, J.BASE_SHARD_NAME))
        with self.assertRaises(J.ShardGapError):
            self.j.shards()

    def test_non_canonical_shard_name_refuses_rather_than_being_ignored(self):
        """`events_007.jsonl` would collide with `events_7.jsonl` in the order."""
        self._append_n(2)
        with open(os.path.join(self.root, "events_007.jsonl"), "w") as fh:
            fh.write("")
        with self.assertRaises(J.JournalCorruption) as ctx:
            self.j.shards()
        self.assertIn("non-canonical", str(ctx.exception))

    def test_same_index_live_and_archived_refuses(self):
        self._append_n(2)
        shutil.copy(os.path.join(self.root, "events_1.jsonl"),
                    os.path.join(self.root, J.ARCHIVE_DIRNAME, "events_1.jsonl"))
        with self.assertRaises(J.JournalCorruption) as ctx:
            self.j.shards()
        self.assertIn("both live and archived", str(ctx.exception))

    def test_reader_enumerates_archive_too(self):
        """Archived shards remain part of the RECORD (invariant 7)."""
        entries = self._append_n(4)
        self.j.register_reader("planner")
        self.j.commit_cursor("planner", entries[-1].seq)
        archived = self.j.archive_retired_shards()
        self.assertTrue(archived)
        self.assertEqual([e.seq for e in self.j.read_all()], [1, 2, 3, 4])
        # ...and a base-only reader now sees an ARCHIVED file, not the history.
        self.assertTrue(any(s.archived for s in self.j.shards()))


# =============================================================================
# Durability: fsync-per-event and a crash mid-write
# =============================================================================

class TestDurability(_JournalTest):
    def test_every_append_fsyncs_the_shard(self):
        real_fsync = os.fsync
        calls = []

        def counting_fsync(fd):
            calls.append(fd)
            return real_fsync(fd)

        with mock.patch("os.fsync", side_effect=counting_fsync):
            self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip())
        self.assertGreaterEqual(len(calls), 1)

    def test_crash_mid_write_loses_only_the_unacknowledged_event(self):
        """A torn write costs the in-flight event and nothing else.

        `os.write` is made to write half the line and the append is driven to
        failure — the on-disk state a crash between write and fsync produces. The
        three acknowledged events must read back intact, and the partial line
        must be visible as a torn tail rather than as history.
        """
        acked = [self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip(f"acked {i}"))
                 for i in range(3)]

        real_write = os.write

        def half_write(fd, data):
            return real_write(fd, data[: len(data) // 2])

        with mock.patch("os.write", side_effect=half_write):
            with self.assertRaises(J.JournalCorruption):
                self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("never acknowledged"))

        restarted = self._new_journal()
        survivors = restarted.read_all()
        self.assertEqual([e.event_id for e in survivors], [e.event_id for e in acked])
        self.assertEqual([e.payload["reason"] for e in survivors],
                         ["acked 0", "acked 1", "acked 2"])

        torn = restarted.torn_tail()
        self.assertIsNotNone(torn)
        self.assertGreater(torn.byte_count, 0)

    def test_next_append_repairs_the_torn_tail_and_records_the_loss(self):
        """Without repair, the next event is concatenated onto the fragment and
        BOTH are lost. The repair is durable evidence, not a silent truncate."""
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("acked"))
        shard = os.path.join(self.root, J.BASE_SHARD_NAME)
        with open(shard, "ab") as fh:
            fh.write(b'{"journal_schema":"epyc.autokernel.journal_entry.v1","seq":2')

        restarted = self._new_journal()
        after = restarted.append(J.KIND_PROPOSAL_SKIPPED, _skip("after the crash"))

        entries = restarted.read_all()
        kinds = [e.kind for e in entries]
        self.assertEqual(kinds, [J.KIND_PROPOSAL_SKIPPED,
                                 J.KIND_TORN_APPEND_DISCARDED,
                                 J.KIND_PROPOSAL_SKIPPED])
        tombstone = entries[1].payload
        self.assertEqual(tombstone["discarded_byte_count"], 60)
        self.assertEqual(len(tombstone["discarded_sha256"]), 64)
        self.assertEqual(entries[-1].event_id, after.event_id)
        self.assertIsNone(restarted.torn_tail())

    def test_torn_line_in_a_non_final_shard_is_corruption(self):
        j = J.Journal(self.root, campaign_id=CAMPAIGN, max_shard_bytes=1)
        j.append(J.KIND_PROPOSAL_SKIPPED, _skip("a"))
        j.append(J.KIND_PROPOSAL_SKIPPED, _skip("b"))
        with open(os.path.join(self.root, J.BASE_SHARD_NAME), "ab") as fh:
            fh.write(b'{"partial":')
        with self.assertRaises(J.JournalCorruption):
            j.read_all()

    def test_unparseable_line_is_a_reported_defect_not_a_skip(self):
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("good"))
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("also good"))
        with open(os.path.join(self.root, J.BASE_SHARD_NAME), "ab") as fh:
            fh.write(b"{ not json }\n")

        report = self.j.scan()
        self.assertEqual(len(report.entries), 2)
        self.assertEqual(len(report.defects), 1)
        self.assertIn("not valid JSON", report.defects[0].reason)
        with self.assertRaises(J.JournalCorruption):
            self.j.read_all()

    def test_blank_trailing_line_refuses_rather_than_reusing_a_seq(self):
        """A blank last line would read as "this shard is empty", and the next
        append would reuse a sequence number."""
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("first"))
        with open(os.path.join(self.root, J.BASE_SHARD_NAME), "ab") as fh:
            fh.write(b"\n")
        with self.assertRaises(J.JournalCorruption):
            self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("second"))

    def test_unicode_line_separator_does_not_split_a_record(self):
        """U+2028 is a line break to `str.splitlines()` and not to `b"\\n".split`.

        Canonical JSON is written with ensure_ascii=False, so a payload may carry
        one; a splitlines-based reader would read one record as two.
        """
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("before after"))
        with open(os.path.join(self.root, J.BASE_SHARD_NAME), "rb") as fh:
            raw = fh.read()
        self.assertEqual(len(raw.decode("utf-8").splitlines()), 2,
                         "precondition: splitlines really does break on U+2028")
        entries = self.j.read_all()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].payload["reason"], "before after")


# =============================================================================
# Append is a gate, not a funnel
# =============================================================================

class TestAppendRefusals(_JournalTest):
    def test_unknown_kind_refused(self):
        with self.assertRaises(ValueError):
            self.j.append("SOMETHING_NEW", {"x": 1})
        self.assertEqual(self.j.read_all(), [])

    def test_invalid_record_never_enters_the_journal(self):
        broken = _candidate()
        del broken["ancestry"]
        with self.assertRaises(ValueError) as ctx:
            self.j.append(J.KIND_CANDIDATE_RECORDED, broken)
        self.assertIn("ancestry", str(ctx.exception))
        self.assertEqual(self.j.read_all(), [])

    def test_kind_and_payload_schema_must_agree(self):
        with self.assertRaises(ValueError):
            self.j.append(J.KIND_CANDIDATE_RECORDED, _event())

    def test_campaign_id_contradiction_refused(self):
        other = J.Journal(self.root, campaign_id="ak-other-20260803")
        with self.assertRaises(ValueError) as ctx:
            other.append(J.KIND_CANDIDATE_RECORDED, _candidate())
        self.assertIn("contradicts", str(ctx.exception))

    def test_record_id_is_extracted_from_the_declared_identity_key(self):
        entry = self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate())
        self.assertEqual(entry.record_id, "akc-20260803-0001")
        champion = self.j.append(J.KIND_CHAMPION_UPDATED, _champion())
        self.assertEqual(champion.record_id, "ak/champion/llama-20260802")

    def test_native_payload_may_not_smuggle_narrative(self):
        with self.assertRaises(ValueError) as ctx:
            self.j.append(J.KIND_STOP_STATE,
                          {"state": "PLATEAU_STOP", "narrative": "prose"})
        self.assertIn("narrative", str(ctx.exception))

    def test_completed_microbench_run_binds_its_key_and_content_identity(self):
        raw = {
            "candidate_id": "akc-20260803-0001", "attempt": 0,
            "segment": "base", "extension_round": None, "complete": True,
            "ended_at": "2026-08-05T08:00:00+00:00", "blocks": [],
        }
        payload = {
            "campaign_id": CAMPAIGN, "candidate_id": raw["candidate_id"],
            "attempt": 0, "segment": "base", "extension_round": None,
            "run_id": S.content_hash(raw), "completed_at": raw["ended_at"],
            "complete": True, "raw_vector": raw,
        }
        self.j.append(J.KIND_MICROBENCH_RUN_COMPLETED, payload,
                      record_id=payload["run_id"])
        broken = copy.deepcopy(payload)
        broken["attempt"] = 2
        with self.assertRaisesRegex(ValueError, "raw_vector.attempt"):
            self.j.append(J.KIND_MICROBENCH_RUN_COMPLETED, broken,
                          record_id=broken["run_id"])
        broken = copy.deepcopy(payload)
        broken["run_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash"):
            self.j.append(J.KIND_MICROBENCH_RUN_COMPLETED, broken,
                          record_id=broken["run_id"])

    def test_supersession_target_must_exist(self):
        with self.assertRaises(J.SupersessionError):
            self.j.append_superseded("akj-000000000042-deadbeefcafe", "stale")

    def test_supersession_needs_a_reason_and_retrieval_supersession_a_receipt(self):
        entry = self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate())
        with self.assertRaises(ValueError):
            self.j.append(J.KIND_SUPERSEDED,
                          {"target_event_id": entry.event_id, "reason": "  "})
        with self.assertRaises(ValueError):
            self.j.append(J.KIND_RETRIEVAL_SUPERSEDED,
                          {"target_event_id": entry.event_id, "reason": "wrong",
                           "receipt": ""})

    def test_event_ids_are_unique_across_shards(self):
        j = J.Journal(self.root, campaign_id=CAMPAIGN, max_shard_bytes=1)
        ids = {j.append(J.KIND_PROPOSAL_SKIPPED, _skip(f"r{i}")).event_id
               for i in range(6)}
        self.assertEqual(len(ids), 6)

    def test_seq_continues_across_a_restart(self):
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("a"))
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("b"))
        restarted = self._new_journal()
        entry = restarted.append(J.KIND_PROPOSAL_SKIPPED, _skip("c"))
        self.assertEqual(entry.seq, 3)


class TestVoidedRunsAreJournalable(_JournalTest):
    """P-AK-SEARCH-1 "What voids a run": *"A voided run is journaled as `INVALID`
    with its reason, and is **never silently discarded**."*

    The ANCHOR-MISSING void was the one case where that sentence was
    unsatisfiable through the primary record: `evaluation_event.v2` required
    `anchor.binary_sha256`, `append()` validates before it writes, and the whole
    point of the void is that there is no digest to record. The evaluator was
    correct to refuse to invent one, so the record simply could not be written.
    These tests exercise the REAL `Journal.append`, not a validator call — the
    write path is where the defect actually bit.
    """

    def _voided_anchorless_event(self, suffix: str = "0001") -> dict:
        record = _event(suffix)
        record["schema"] = S.SCHEMA_EVALUATION_EVENT
        del record["anchor"]
        record["status"] = "invalid"
        record["integrity_flags"] = ["VOID:ANCHOR_MISSING_OR_MUTATED:FAIL"]
        record["performance"] = {"raw_samples": [], "paired_blocks": 0,
                                 "estimate": None, "uncertainty": None}
        return record

    def test_a_voided_anchorless_run_round_trips_through_append(self):
        record = self._voided_anchorless_event()
        entry = self.j.append(J.KIND_EVALUATION_EVENT, record)
        self.assertEqual(entry.record_id, record["event_id"])

        report = self._new_journal().scan(validate_payloads=True)
        self.assertEqual(report.defects, ())
        [stored] = [e for e in report.entries if e.kind == J.KIND_EVALUATION_EVENT]
        self.assertEqual(stored.payload["status"], "invalid")
        self.assertNotIn("anchor", stored.payload)
        self.assertIn("VOID:ANCHOR_MISSING_OR_MUTATED:FAIL",
                      stored.payload["integrity_flags"])
        self.assertEqual(S.validate_record(stored.payload), [])
        # It survived byte-for-byte, so the reason is durable evidence.
        self.assertEqual(S.content_hash(stored.payload), S.content_hash(record))

    def test_a_pass_record_with_no_anchor_is_still_refused_by_append(self):
        record = self._voided_anchorless_event()
        record["status"] = "pass"
        record["integrity_flags"] = []
        with self.assertRaises(ValueError) as ctx:
            self.j.append(J.KIND_EVALUATION_EVENT, record)
        self.assertIn("anchor", str(ctx.exception))
        self.assertEqual(self.j.read_all(), [])

    def test_a_placeholder_anchor_is_refused_by_append(self):
        record = self._voided_anchorless_event()
        record["anchor"] = {"source_commit": "0" * 40, "binary_sha256": "0" * 64,
                            "linkage_sha256": "0" * 64,
                            "measurement_event_ids": ["ake-20260801-0009"]}
        with self.assertRaises(ValueError) as ctx:
            self.j.append(J.KIND_EVALUATION_EVENT, record)
        self.assertIn("placeholder digest", str(ctx.exception))
        self.assertEqual(self.j.read_all(), [])

    def test_the_kind_accepts_every_live_evaluation_event_version(self):
        self.assertEqual(
            J.ACCEPTED_SCHEMAS_BY_KIND[J.KIND_EVALUATION_EVENT],
            frozenset({S.SCHEMA_EVALUATION_EVENT_V2, S.SCHEMA_EVALUATION_EVENT_V3,
                       S.SCHEMA_EVALUATION_EVENT_V4, S.SCHEMA_EVALUATION_EVENT_V5}))
        records = []
        for index, schema in enumerate((S.SCHEMA_EVALUATION_EVENT_V2,
                                        S.SCHEMA_EVALUATION_EVENT_V3,
                                        S.SCHEMA_EVALUATION_EVENT_V4,
                                        S.SCHEMA_EVALUATION_EVENT_V5), start=2):
            record = _event(f"000{index}")
            record["schema"] = schema
            if schema == S.SCHEMA_EVALUATION_EVENT_V2:
                del record["anchor"]["source_commit"]
            if schema in (S.SCHEMA_EVALUATION_EVENT_V2,
                          S.SCHEMA_EVALUATION_EVENT_V3):
                for key in ("change_class", "anchor_tier", "transfer_ratio_to"):
                    record.pop(key)
            if schema != S.SCHEMA_EVALUATION_EVENT_V5:
                record.pop("backend")
                record.pop("device_state")
            self.j.append(J.KIND_EVALUATION_EVENT, record)
            records.append(schema)
        report = self._new_journal().scan(validate_payloads=True)
        self.assertEqual(report.defects, ())
        self.assertEqual(
            [e.payload["schema"] for e in report.entries
             if e.kind == J.KIND_EVALUATION_EVENT],
            records)

    def test_a_kind_still_refuses_a_schema_from_another_record_family(self):
        record = _event("0004")
        record["schema"] = S.SCHEMA_CHAMPION
        with self.assertRaises(ValueError) as ctx:
            self.j.append(J.KIND_EVALUATION_EVENT, record)
        self.assertIn("declares schema", str(ctx.exception))


class TestTombstones(_JournalTest):
    def test_expirable_artifact_can_be_tombstoned(self):
        entry = self.j.append_tombstone(
            artifact_sha256=_sha("rejected-build-tree"),
            storage_class="expirable",
            size_bytes=13_000_000_000,
            reason="rejected candidate build tree, campaign retired",
            path="/mnt/raid0/llm/tmp/ak-build/akc-20260803-0007",
        )
        self.assertEqual(entry.kind, J.KIND_TOMBSTONE)
        views = J.rebuild_views(self.j.read_all())
        # Keyed by (hash, path): the hash alone merged two reclamations of
        # byte-identical trees into one receipt.
        self.assertIn(
            J.tombstone_view_key(entry.payload), views.tombstones
        )
        self.assertEqual(
            views.tombstones[J.tombstone_view_key(entry.payload)]["artifact_sha256"],
            _sha("rejected-build-tree"),
        )

    def test_permanent_artifact_may_not_be_tombstoned(self):
        """Invariant 7: evidence is never evicted. Only `expirable` expires."""
        for storage_class in ("permanent_in_repo", "permanent_large"):
            with self.subTest(storage_class=storage_class):
                with self.assertRaises(ValueError) as ctx:
                    self.j.append_tombstone(
                        artifact_sha256=_sha("events"),
                        storage_class=storage_class,
                        size_bytes=1,
                        reason="disk pressure",
                        path="/mnt/raid0/llm/epyc-inference-research/data/ak/events",
                    )
                self.assertIn("may not be tombstoned", str(ctx.exception))

    def test_tombstone_requires_hash_class_size_reason_and_path(self):
        for missing in ("artifact_sha256", "storage_class", "size_bytes", "reason",
                        "path"):
            with self.subTest(missing=missing):
                payload = {
                    "artifact_sha256": _sha("a"),
                    "storage_class": "expirable",
                    "size_bytes": 10,
                    "reason": "expired",
                    "path": "/mnt/raid0/llm/tmp/ak-build/akc-0001",
                }
                del payload[missing]
                with self.assertRaises(ValueError) as ctx:
                    self.j.append(J.KIND_TOMBSTONE, payload)
                self.assertIn(missing, str(ctx.exception))

    def test_two_reclamations_of_identical_bytes_are_two_receipts(self):
        """The seam defect this key change closes.

        `storage.tombstone_id` is derived from (campaign, path, sha256, kind,
        rule), so byte-identical build trees at two paths are two reclamations
        there. Keyed by the hash alone they were ONE slot here, and
        `check_view_consistency` agreed because it recounted by the same key —
        one deleted artifact with no visible receipt, reported PASS.
        """
        shared_hash = _sha("byte-identical-build-tree")
        for suffix in ("0007", "0008"):
            self.j.append_tombstone(
                artifact_sha256=shared_hash,
                storage_class="expirable",
                size_bytes=4_000_000_000,
                reason="rejected candidate, lineage retired",
                path=f"/mnt/raid0/llm/tmp/ak-build/akc-{suffix}",
                tombstone_id=f"akt-{suffix}",
            )
        events = self.j.read_all()
        views = J.rebuild_views(events)
        self.assertEqual(len(views.tombstones), 2)
        self.assertEqual(
            sorted(p["path"] for p in views.tombstones.values()),
            ["/mnt/raid0/llm/tmp/ak-build/akc-0007",
             "/mnt/raid0/llm/tmp/ak-build/akc-0008"],
        )
        self.assertEqual(J.check_view_consistency(events, views).outcome, S.PASS)

    def test_collapsing_two_reclamations_into_one_slot_is_caught(self):
        """The independent `tombstone_id` recount, proved non-vacuous."""
        shared_hash = _sha("byte-identical-build-tree")
        for suffix in ("0007", "0008"):
            self.j.append_tombstone(
                artifact_sha256=shared_hash, storage_class="expirable",
                size_bytes=1, reason="retired",
                path=f"/mnt/raid0/llm/tmp/ak-build/akc-{suffix}",
                tombstone_id=f"akt-{suffix}",
            )
        events = self.j.read_all()
        views = J.rebuild_views(events)
        collapsed = dataclasses.replace(
            views,
            tombstones={shared_hash: dict(list(views.tombstones.values())[0])},
        )
        check = J.check_view_consistency(events, collapsed)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(
            any("distinct reclamation id" in r for r in check.reasons),
            check.reasons,
        )


# =============================================================================
# Record vs retrieval (§5.5 items 6 and 7, invariant 20)
# =============================================================================

class TestRecordVersusRetrieval(_JournalTest):
    def setUp(self):
        super().setUp()
        self.belief = self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate("0001"))
        self.other = self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate("0002"))

    def test_retrieval_supersession_hides_from_retrieval_not_from_the_record(self):
        self.j.append_retrieval_superseded(
            self.belief.event_id,
            reason="the mechanism claim was refuted by the T1c receipt",
            receipt="ggml-cuda/mmq.cu@67a433bf:412",
        )

        record_ids = [e.event_id for e in self.j.read_all()]
        self.assertIn(self.belief.event_id, record_ids)

        retrieved_ids = [r["event_id"] for r in self.j.retrieve()]
        self.assertNotIn(self.belief.event_id, retrieved_ids)
        self.assertIn(self.other.event_id, retrieved_ids)

        # The withdrawal itself stays retrievable — otherwise the planner cannot
        # know a belief was withdrawn, only that it vanished.
        kinds = {r["kind"] for r in self.j.retrieve()}
        self.assertIn(J.KIND_RETRIEVAL_SUPERSEDED, kinds)

    def test_retrieval_supersession_leaves_the_record_in_derived_views(self):
        """Derived views are RECORD-level; only retrieval withholds the belief."""
        self.j.append_retrieval_superseded(
            self.belief.event_id, reason="refuted", receipt="src@sha:1")
        views = J.rebuild_views(self.j.read_all())
        self.assertIn("akc-20260803-0001", views.candidates)
        self.assertIn(self.belief.event_id, views.retrieval_superseded_event_ids)

    def test_record_supersession_leaves_the_record_but_drops_the_view_slot(self):
        self.j.append_superseded(self.belief.event_id, "rebuilt against a fresh base",
                                 superseded_by=self.other.event_id)
        record_ids = [e.event_id for e in self.j.read_all()]
        self.assertIn(self.belief.event_id, record_ids)
        views = J.rebuild_views(self.j.read_all())
        self.assertNotIn("akc-20260803-0001", views.candidates)
        self.assertIn("akc-20260803-0002", views.candidates)

    def test_narrative_excluded_by_default_at_every_depth(self):
        record = _candidate(
            "0003",
            narrative="the planner's story about why this will work",
            narrative_retrievable=False,
            derived_verdicts={"t1": {"narrative": "nested prose", "verdict": "pass"}},
        )
        entry = self.j.append(J.KIND_CANDIDATE_RECORDED, record)

        from_record = [e for e in self.j.read_all() if e.event_id == entry.event_id][0]
        self.assertIn("narrative", from_record.payload)
        self.assertIn("narrative", from_record.payload["derived_verdicts"]["t1"])

        row = [r for r in self.j.retrieve() if r["event_id"] == entry.event_id][0]
        self.assertNotIn("narrative", row["payload"])
        self.assertNotIn("narrative", row["payload"]["derived_verdicts"]["t1"])
        self.assertEqual(row["payload"]["derived_verdicts"]["t1"]["verdict"], "pass")

    def test_narrative_admitted_only_for_the_cited_event(self):
        cited = self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate(
            "0004", narrative="cited prose", narrative_retrievable=False))
        uncited = self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate(
            "0005", narrative="uncited prose", narrative_retrievable=False))

        rows = {r["event_id"]: r for r in
                self.j.retrieve(cite_event_ids=[cited.event_id])}
        self.assertEqual(rows[cited.event_id]["payload"]["narrative"], "cited prose")
        self.assertNotIn("narrative", rows[uncited.event_id]["payload"])

    def test_citing_a_withdrawn_belief_raises(self):
        """The one path by which a withdrawn belief could walk back in."""
        self.j.append_retrieval_superseded(
            self.belief.event_id, reason="refuted", receipt="src@sha:1")
        with self.assertRaises(J.RetrievalCitationError) as ctx:
            self.j.retrieve(cite_event_ids=[self.belief.event_id])
        self.assertIn("may not be cited back in", str(ctx.exception))

    def test_citing_an_unknown_event_raises(self):
        with self.assertRaises(J.RetrievalCitationError):
            self.j.retrieve(cite_event_ids=["akj-000000000999-abcabcabcabc"])

    def test_retrieve_kind_filter_rejects_an_unknown_kind(self):
        with self.assertRaises(ValueError):
            self.j.retrieve(kinds=["CANDIDATE_RECORDED", "NOT_A_KIND"])

    def test_strip_narrative_handles_lists(self):
        value = {"a": [{"narrative": "x", "keep": 1}, {"keep": 2}]}
        self.assertEqual(J.strip_narrative(value),
                         {"a": [{"keep": 1}, {"keep": 2}]})


# =============================================================================
# Derived views and the BOOTSTRAP consistency assertion (§8.2 step 10)
# =============================================================================

class TestViewsAndConsistency(_JournalTest):
    def setUp(self):
        super().setUp()
        self.j.append(J.KIND_CAMPAIGN_OPENED, _campaign())
        self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate("0001", status="banked"))
        self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate("0002", status="rejected"))
        self.j.append(J.KIND_EVALUATION_EVENT, _event("0001"))
        self.j.append(J.KIND_CHAMPION_UPDATED, _champion())

    def test_views_rebuild_from_events(self):
        views = J.rebuild_views(self.j.read_all())
        self.assertEqual(set(views.candidates), {"akc-20260803-0001", "akc-20260803-0002"})
        self.assertEqual(views.frontier, ("akc-20260803-0001",))
        self.assertEqual(set(views.evaluations), {"ake-20260803-0001"})
        self.assertEqual(set(views.champions), {"llama.cpp"})
        self.assertEqual(set(views.campaigns), {CAMPAIGN})

    def test_latest_record_wins_per_identity(self):
        self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate("0002", status="banked"))
        views = J.rebuild_views(self.j.read_all())
        self.assertEqual(views.candidates["akc-20260803-0002"]["status"], "banked")
        self.assertEqual(views.frontier, ("akc-20260803-0001", "akc-20260803-0002"))

    def test_healthy_journal_passes_the_assertion(self):
        events = self.j.read_all()
        views = J.rebuild_views(events)
        check = J.assert_views_consistent(events, views)
        self.assertEqual(check.outcome, S.PASS)
        self.assertTrue(check.passed)

    def test_empty_view_with_a_nonempty_journal_raises(self):
        """The AutoPilot loss, exactly: 232 trials and ~16 days of compute went
        to a restart that came up with an empty frontier and nothing objected."""
        events = self.j.read_all()
        views = dataclasses.replace(J.rebuild_views(events), candidates={}, frontier=())
        with self.assertRaises(J.ViewConsistencyError) as ctx:
            J.assert_views_consistent(events, views)
        self.assertIn("EMPTY", str(ctx.exception))

    def test_cardinality_disagreement_raises(self):
        events = self.j.read_all()
        full = J.rebuild_views(events)
        half = dict(full.candidates)
        half.pop("akc-20260803-0002")
        views = dataclasses.replace(full, candidates=half)
        with self.assertRaises(J.ViewConsistencyError) as ctx:
            J.assert_views_consistent(events, views)
        self.assertIn("akc-20260803-0002", str(ctx.exception))

    def test_empty_champion_view_raises(self):
        events = self.j.read_all()
        views = dataclasses.replace(J.rebuild_views(events), champions={})
        with self.assertRaises(J.ViewConsistencyError):
            J.assert_views_consistent(events, views)

    def test_deliberate_rebase_is_the_only_escape_and_needs_a_reason(self):
        events = self.j.read_all()
        views = dataclasses.replace(J.rebuild_views(events), candidates={}, frontier=())

        with self.assertRaises(ValueError):
            J.assert_views_consistent(events, views, deliberate_rebase=True)
        with self.assertRaises(ValueError):
            J.assert_views_consistent(events, views, deliberate_rebase=True,
                                      rebase_reason="   ")

        check = J.assert_views_consistent(
            events, views, deliberate_rebase=True,
            rebase_reason="operator-approved rebase after the v9 anchor move")
        # The escape returns the FAILING check rather than a PASS: the rebase is
        # permitted, not pretended-away.
        self.assertEqual(check.outcome, S.FAIL)
        self.assertFalse(check.passed)

    def test_rebase_does_not_cover_could_not_check(self):
        """"I meant to empty the views" is not an answer to "I cannot tell
        whether these views belong to these events"."""
        events = self.j.read_all()
        views = J.rebuild_views(events[:2])          # built from a different read
        check = J.check_view_consistency(events, views)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        with self.assertRaises(J.ViewConsistencyError) as ctx:
            J.assert_views_consistent(events, views, deliberate_rebase=True,
                                      rebase_reason="deliberate")
        self.assertIn("COULD NOT BE CHECKED", str(ctx.exception))

    def test_base_only_rebuild_is_could_not_check_not_pass(self):
        """A base-only read checked against an all-shard read must not PASS."""
        j = J.Journal(self.root, campaign_id=CAMPAIGN, max_shard_bytes=1)
        j.append(J.KIND_CANDIDATE_RECORDED, _candidate("0003"))
        j.append(J.KIND_CANDIDATE_RECORDED, _candidate("0004"))
        all_events = j.read_all()
        base_only = [e for e in all_events if e.shard_index == 0]
        self.assertLess(len(base_only), len(all_events))
        check = J.check_view_consistency(all_events, J.rebuild_views(base_only))
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_dangling_supersession_target_fails(self):
        events = self.j.read_all()
        forged = J.JournalEntry(
            event_id="akj-000000000099-ffffffffffff", seq=99,
            kind=J.KIND_SUPERSEDED, campaign_id=CAMPAIGN, record_id=None,
            written_at="2026-08-03T12:00:00+00:00",
            payload={"target_event_id": "akj-000000000042-aaaaaaaaaaaa",
                     "reason": "forged", "superseded_by": None},
        )
        events = list(events) + [forged]
        check = J.check_view_consistency(events, J.rebuild_views(events))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("resolve to no event" in r for r in check.reasons))

    def test_check_is_could_not_check_on_a_non_views_object(self):
        events = self.j.read_all()
        self.assertEqual(J.check_view_consistency(events, {"candidates": {}}).outcome,
                         S.COULD_NOT_CHECK)

    def test_bootstrap_views_passes_on_a_healthy_journal(self):
        views = self.j.bootstrap_views()
        self.assertEqual(set(views.candidates),
                         {"akc-20260803-0001", "akc-20260803-0002"})
        self.assertNotIn(J.KIND_VIEW_REBASED, [e.kind for e in self.j.read_all()])

    def test_bootstrap_rebase_journals_the_decision(self):
        """A rebase must be visible in the record afterwards, not inferred."""
        events = self.j.read_all()
        broken = dataclasses.replace(J.rebuild_views(events), candidates={}, frontier=())
        with mock.patch.object(J, "rebuild_views", return_value=broken):
            self.j.bootstrap_views(
                deliberate_rebase=True,
                rebase_reason="operator-approved rebase after the v9 anchor move")
        rebases = [e for e in self.j.read_all() if e.kind == J.KIND_VIEW_REBASED]
        self.assertEqual(len(rebases), 1)
        self.assertIn("operator-approved", rebases[0].payload["rebase_reason"])
        self.assertTrue(rebases[0].payload["suppressed_reasons"])

    def test_rebuild_views_refuses_a_non_entry(self):
        with self.assertRaises(TypeError):
            J.rebuild_views([{"event_id": "x"}])


# =============================================================================
# Cursors and rotation-past-all-cursors
# =============================================================================

class TestCursorsAndArchiving(_JournalTest):
    max_shard_bytes = 1

    def setUp(self):
        super().setUp()
        self.entries = [self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip(f"r{i}"))
                        for i in range(5)]

    def test_archiving_refuses_when_no_reader_is_registered(self):
        """"No cursors" is not "all cursors have passed"."""
        with self.assertRaises(J.CursorError) as ctx:
            self.j.archive_retired_shards()
        self.assertIn("no readers are registered", str(ctx.exception))

    def test_archiving_stops_at_the_slowest_cursor(self):
        self.j.register_reader("planner")
        self.j.register_reader("critic")
        self.j.commit_cursor("planner", self.entries[4].seq)
        self.j.commit_cursor("critic", self.entries[1].seq)   # slow reader

        archived = self.j.archive_retired_shards()
        self.assertEqual(archived, [0, 1])
        live = [s.index for s in self.j.shards() if not s.archived]
        self.assertEqual(live, [2, 3, 4])
        # The slow reader still gets everything it has not consumed.
        self.assertEqual([e.seq for e in self.j.read_since("critic")], [3, 4, 5])

    def test_archived_events_stay_in_the_record(self):
        self.j.register_reader("planner")
        self.j.commit_cursor("planner", self.entries[-1].seq)
        self.j.archive_retired_shards()
        self.assertEqual([e.seq for e in self.j.read_all()], [1, 2, 3, 4, 5])

    def test_appends_continue_after_archiving(self):
        self.j.register_reader("planner")
        self.j.commit_cursor("planner", self.entries[-1].seq)
        self.j.archive_retired_shards()
        entry = self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("after archive"))
        self.assertEqual(entry.seq, 6)
        self.assertEqual(len(self.j.read_all()), 6)

    def test_read_since_crosses_every_shard(self):
        self.j.register_reader("planner")
        self.assertEqual([e.seq for e in self.j.read_since("planner")], [1, 2, 3, 4, 5])
        self.j.commit_cursor("planner", 3)
        self.assertEqual([e.seq for e in self.j.read_since("planner")], [4, 5])

    def test_cursor_will_not_rewind_by_accident(self):
        self.j.register_reader("planner")
        self.j.commit_cursor("planner", 4)
        with self.assertRaises(J.CursorError):
            self.j.commit_cursor("planner", 2)
        self.assertEqual(self.j.cursor("planner").last_seq, 4)
        self.j.commit_cursor("planner", 2, allow_rewind=True)
        self.assertEqual(self.j.cursor("planner").last_seq, 2)

    def test_unregistered_reader_cannot_commit_or_read_since(self):
        with self.assertRaises(J.CursorError):
            self.j.commit_cursor("ghost", 1)
        with self.assertRaises(J.CursorError):
            self.j.read_since("ghost")

    def test_reader_id_may_not_escape_the_cursor_directory(self):
        for bad in ("../evil", "a/b", "", ".hidden"):
            with self.subTest(reader_id=bad):
                with self.assertRaises(J.CursorError):
                    self.j.register_reader(bad)

    def test_unreadable_cursor_raises_rather_than_defaulting(self):
        """A fail-open cursor default silently corrupts the archive decision."""
        self.j.register_reader("planner")
        with open(os.path.join(self.root, J.CURSOR_DIRNAME, "planner.json"), "w") as fh:
            fh.write("{ truncated")
        with self.assertRaises(J.CursorError):
            self.j.cursor("planner")


# =============================================================================
# Locking and control acknowledgement (invariant 19)
# =============================================================================

class TestLockAndControlAck(_JournalTest):
    def test_write_lock_is_reentrant_for_the_same_journal(self):
        """The control plane re-reads its latch under this lock and then acks;
        a non-reentrant lock would deadlock on the ack."""
        with self.j.write_lock():
            entry = self.j.append_control_ack(
                control="pause", control_id="ctl-0001",
                received_at="2026-08-03T12:00:00+00:00", disposition="latched")
        self.assertEqual(entry.kind, J.KIND_OPERATOR_CONTROL_ACK)
        self.assertEqual(self.j.read_all()[0].payload["control"], "pause")

    def test_lock_is_released_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with self.j.write_lock():
                raise RuntimeError("boom")
        # If the lock leaked, this append would block forever rather than return.
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("after the exception"))
        self.assertEqual(len(self.j.read_all()), 1)

    def test_control_ack_requires_every_field(self):
        for missing in ("control", "control_id", "received_at", "disposition"):
            with self.subTest(missing=missing):
                payload = {"control": "pause", "control_id": "c1",
                           "received_at": "2026-08-03T12:00:00+00:00",
                           "disposition": "latched"}
                del payload[missing]
                with self.assertRaises(ValueError):
                    self.j.append(J.KIND_OPERATOR_CONTROL_ACK, payload)


# =============================================================================
# Uninitialized / misconfigured journals fail loudly
# =============================================================================

class TestConstruction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ak_journal_ctor_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reading_an_uninitialized_journal_raises(self):
        j = J.Journal(os.path.join(self.tmp, "nope"))
        with self.assertRaises(J.JournalCorruption):
            j.read_all()

    def test_initialize_is_idempotent(self):
        root = os.path.join(self.tmp, "j")
        first = J.Journal(root)
        first.initialize()
        first.append(J.KIND_PROPOSAL_SKIPPED, _skip("kept"))
        second = J.Journal(root)
        second.initialize()
        self.assertEqual(len(second.read_all()), 1)

    def test_bad_max_shard_bytes_refused(self):
        with self.assertRaises(ValueError):
            J.Journal(os.path.join(self.tmp, "j"), max_shard_bytes=0)
        with self.assertRaises(TypeError):
            J.Journal(os.path.join(self.tmp, "j"), max_shard_bytes="64MB")

    def test_empty_root_refused(self):
        with self.assertRaises(ValueError):
            J.Journal("")


# =============================================================================
# Adversarial red-team regressions (2026-08-03)
#
# Every case below FAILED against the module as first written. They are grouped
# by the axis that found them rather than by API surface, because the point of
# each one is the class of mistake, not the method.
# =============================================================================

def _waiver(waiver_id: str = "akw-20260803-0001") -> dict:
    """A minimal operator waiver that passes schemas.validate_record()."""
    return {
        "schema": S.SCHEMA_OPERATOR_WAIVER,
        "waiver_id": waiver_id,
        "campaign_id": CAMPAIGN,
        "decision": "release with a named forfeit",
        "protocol": "P-KERNEL-FREEZE-1/v1",
        "protocol_changed": False,
        "candidate_head": V7_COMMIT,
        "production_head": V8_COMMIT,
        "scope": {
            "excluded_models": ["qwen3-30b-a3b-q8"],
            "excluded_pairs": [],
            "remaining_matched_pairs": 11,
        },
        "reason": "Q8 cell not measurable within the freeze window",
        "consequences": ["this release makes no Q8 non-regression claim"],
        "authorized_by": "operator",
        "expiry": {"expires_at": None, "reopen_predicate": "next freeze"},
        "created_at": "2026-08-03T11:00:00+00:00",
    }


class TestInitializeDoesNotForgeAShard(_JournalTest):
    """`initialize()` is what every process calls at startup. Both directions of
    getting its precondition wrong were live, and both were silent."""

    max_shard_bytes = 1                       # one event per shard

    def test_initialize_after_archiving_does_not_brick_the_journal(self):
        """Retire index 0 into `archive/`, restart, and the journal must live.

        `initialize()` tested `os.path.exists(events.jsonl)` in the LIVE root
        only, so once `archive_retired_shards()` had moved the base shard, the
        next process re-created it — putting index 0 in two places. `shards()`
        then refuses the journal permanently and EVERY read and EVERY append
        raises. The second process to start destroyed the record.
        """
        self.j.register_reader("planner")
        for i in range(4):
            self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip(f"reason {i}"))
        events = self.j.read_all()
        self.j.commit_cursor("planner", events[-1].seq)
        self.assertIn(0, self.j.archive_retired_shards())
        self.assertFalse(os.path.exists(os.path.join(self.root, J.BASE_SHARD_NAME)))

        restarted = self._new_journal()
        restarted.initialize()                # the ordinary startup call
        self.assertEqual(len(restarted.read_all()), len(events))
        restarted.append(J.KIND_PROPOSAL_SKIPPED, _skip("after restart"))
        self.assertEqual(len(restarted.read_all()), len(events) + 1)

    def test_initialize_does_not_fabricate_a_lost_base_shard(self):
        """Delete `events.jsonl` and the gap detector must keep firing.

        This is the project's standing screen — "can I pass this check by
        deleting what it inspects?" — and the answer was yes. `ShardGapError`
        exists precisely to report a missing base shard; `initialize()` created
        an empty replacement, after which `read_all()` returned only the
        post-hole shards and reported success. Half the journal disappeared with
        no exception and no defect.
        """
        for i in range(4):
            self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip(f"reason {i}"))
        before = len(self.j.read_all())
        os.remove(os.path.join(self.root, J.BASE_SHARD_NAME))
        with self.assertRaises(J.ShardGapError):
            self._new_journal().read_all()

        recovered = self._new_journal()
        recovered.initialize()
        with self.assertRaises(J.ShardGapError):
            recovered.read_all()              # still refuses; the loss is still visible
        self.assertGreater(before, 1)


class TestRotationOnATornJournal(_JournalTest):
    def test_rotate_repairs_the_torn_tail_instead_of_entombing_it(self):
        """A crash plus one `rotate()` must not destroy the journal.

        `rotate()` created the next shard without repairing an unacknowledged
        fragment first, which left the fragment in a shard that was no longer
        final — and `torn_tail()` correctly calls that corruption. `torn_tail()`,
        `scan()`, `read_all()` and `append()` then all raised forever, with no
        recovery path. "Rotation is always safe" was true only for a journal that
        had never crashed.
        """
        kept = self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("acknowledged"))
        with open(os.path.join(self.root, J.BASE_SHARD_NAME), "ab") as fh:
            fh.write(b'{"journal_schema":"epyc.autokernel.journal_entry.v1","seq":2')
        self.assertIsNotNone(self.j.torn_tail())

        restarted = self._new_journal()
        restarted.rotate()

        self.assertIsNone(restarted.torn_tail())
        kinds = [e.kind for e in restarted.read_all()]
        self.assertIn(J.KIND_TORN_APPEND_DISCARDED, kinds)
        self.assertIn(kept.event_id, [e.event_id for e in restarted.read_all()])
        restarted.append(J.KIND_PROPOSAL_SKIPPED, _skip("after rotate"))

    def test_a_torn_fragment_larger_than_a_megabyte_is_still_repairable(self):
        """There is no payload size cap, so a >1 MiB torn fragment is legal.

        `_trailing_fragment()` read exactly one 1 MiB window and, finding no
        newline in it, raised "the file is not line-delimited JSON" — the wrong
        diagnosis for a crash partway through a large event. That raise bricked
        `torn_tail()`, `read_all()` and `append()` at the precise moment the
        torn-tail repair exists to rescue the journal.
        """
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("acknowledged"))
        with open(os.path.join(self.root, J.BASE_SHARD_NAME), "ab") as fh:
            fh.write(b"Z" * (2 << 20))

        restarted = self._new_journal()
        torn = restarted.torn_tail()
        self.assertIsNotNone(torn)
        self.assertEqual(torn.byte_count, 2 << 20)
        self.assertEqual(len(restarted.read_all()), 1)
        restarted.append(J.KIND_PROPOSAL_SKIPPED, _skip("recovered"))
        self.assertIsNone(restarted.torn_tail())
        self.assertEqual(
            [e.kind for e in restarted.read_all()],
            [J.KIND_PROPOSAL_SKIPPED, J.KIND_TORN_APPEND_DISCARDED,
             J.KIND_PROPOSAL_SKIPPED],
        )


class TestReadRacingATornTailRepair(_JournalTest):
    def test_a_lock_free_read_cannot_silently_drop_an_acknowledged_event(self):
        """Reads take no lock, so a writer may repair the tail mid-read.

        `scan()` measured the torn tail in one pass and then subtracted that byte
        COUNT from bytes read in a later pass. A writer that repaired the
        fragment and appended in between left the reader chopping bytes off a
        real, fsynced, acknowledged record — and when the stale count happened to
        equal a whole line, the reader dropped that entire event and returned
        with no defect and no exception. A successful-looking read that is
        missing an acknowledged event is the exact fail-open shape this project
        has paid for before.

        The interleaving is made deterministic by running the writer inside the
        reader's own `torn_tail()` call; everything after that point is the
        real, unmodified `scan()`.
        """
        self.j.append(J.KIND_PROPOSAL_SKIPPED, _skip("acknowledged one"))
        base = os.path.join(self.root, J.BASE_SHARD_NAME)
        line_bytes = os.path.getsize(base)          # exactly one line + newline

        root, factory = self.root, self._new_journal

        class RacingReader(J.Journal):
            raced = False

            def torn_tail(self):
                torn = super().torn_tail()
                if torn is not None and not RacingReader.raced:
                    RacingReader.raced = True
                    # Another process repairs the tail and appends a real event.
                    factory().append(J.KIND_PROPOSAL_SKIPPED, _skip("acknowledged two"))
                return torn

        with open(base, "ab") as fh:                # torn tail the size of a line
            fh.write(b"Z" * line_bytes)

        reader = RacingReader(root, campaign_id=CAMPAIGN,
                              max_shard_bytes=self.max_shard_bytes)
        got = reader.read_all()
        self.assertTrue(RacingReader.raced)

        with open(base, "rb") as fh:
            on_disk = [line for line in fh.read().split(b"\n") if line.strip()]
        self.assertEqual(len(got), len(on_disk))
        self.assertIn("acknowledged two",
                      [e.payload.get("reason") for e in got])


class TestConsistencyCheckerInspectsEveryFold(_JournalTest):
    """The standing screen, applied to `check_view_consistency()` itself: delete
    a fold and see whether the checker still says PASS."""

    def test_deleting_the_tombstone_fold_is_caught(self):
        """Tombstones are the §5.8 receipts for deleted expirable artifacts — the
        one view whose disappearance means evidence went missing unrecorded. The
        checker inspected five families, `champions`, the supersession sets, the
        frontier and the entry count, and never looked at tombstones at all, so
        emptying that fold returned PASS."""
        self.j.append_tombstone(
            artifact_sha256=_sha("rejected-build-tree"), storage_class="expirable",
            size_bytes=4_000_000_000, reason="rejected candidate, lineage retired",
            path="/mnt/raid0/llm/tmp/ak-build/akc-20260803-0011",
        )
        events = self.j.read_all()
        views = J.rebuild_views(events)
        self.assertEqual(len(views.tombstones), 1)

        gutted = dataclasses.replace(views, tombstones={})
        self.assertEqual(J.check_view_consistency(events, gutted).outcome, S.FAIL)
        with self.assertRaises(J.ViewConsistencyError):
            J.assert_views_consistent(events, gutted)

    def test_deleting_the_stop_state_fold_is_caught(self):
        self.j.append(J.KIND_STOP_STATE, {"state": "DISK_PRESSURE"})
        events = self.j.read_all()
        views = J.rebuild_views(events)
        self.assertEqual(len(views.stop_states), 1)

        gutted = dataclasses.replace(views, stop_states=())
        self.assertEqual(J.check_view_consistency(events, gutted).outcome, S.FAIL)

    def test_an_operator_waiver_reaches_a_view_and_is_checked(self):
        """A waiver is a §7 record with a declared identity key and a write-time
        validator, and it was folded into nothing: it entered the journal, got a
        `record_id`, appeared in NO view, and `check_view_consistency()` reported
        PASS — because it only inspected the families that happened to have
        slots. §5.6 makes waivers a first-class T3 input; a release view blind to
        the active waivers is the wrong view."""
        entry = self.j.append(J.KIND_OPERATOR_WAIVER_RECORDED, _waiver())
        self.assertEqual(entry.record_id, "akw-20260803-0001")

        events = self.j.read_all()
        views = J.rebuild_views(events)
        self.assertIn("akw-20260803-0001", views.waivers)
        self.assertEqual(views.cardinalities()["waivers"], 1)
        self.assertEqual(J.check_view_consistency(events, views).outcome, S.PASS)

        gutted = dataclasses.replace(views, waivers={})
        self.assertEqual(J.check_view_consistency(events, gutted).outcome, S.FAIL)
        with self.assertRaises(J.ViewConsistencyError):
            J.assert_views_consistent(events, gutted)


class TestRetrievalFilterNeedsTheWholeJournal(_JournalTest):
    def setUp(self):
        super().setUp()
        self.j.append(J.KIND_CAMPAIGN_OPENED, _campaign())
        self.withdrawn = self.j.append(J.KIND_CANDIDATE_RECORDED, _candidate("0001"))
        self.j.append_retrieval_superseded(
            self.withdrawn.event_id, "mechanism falsified at T1",
            "ake-20260803-0002",
        )

    def test_filtering_a_subset_may_not_silently_readmit_a_withdrawn_belief(self):
        """Withholding was derived from the list handed in, so narrowing the list
        first — an obvious thing to write — dropped the RETRIEVAL_SUPERSEDED
        event out of the basis the withholding was computed from, and the
        withdrawn belief came back with the same confident shape as a live one.
        That is the contamination path §5.5 items 6/7 and invariant 20 exist to
        close. The basis is now a required argument."""
        everything = self.j.read_all()
        subset = [e for e in everything if e.kind == J.KIND_CANDIDATE_RECORDED]
        self.assertIn(self.withdrawn.event_id, [e.event_id for e in subset])

        with self.assertRaises(TypeError):
            J.retrieval_filter(subset)            # no basis: refuses outright

        rows = J.retrieval_filter(subset, supersession_basis=everything)
        self.assertEqual(rows, [])

    def test_the_sanctioned_paths_still_withhold(self):
        for rows in (self.j.retrieve(),
                     self.j.retrieve(kinds=[J.KIND_CANDIDATE_RECORDED])):
            self.assertNotIn(self.withdrawn.event_id, [r["event_id"] for r in rows])

    def test_citation_resolves_against_the_basis_not_the_slice(self):
        """A citation naming an event that exists but is outside the slice must
        report the truth about it, not "does not exist in this journal"."""
        everything = self.j.read_all()
        campaign = [e for e in everything if e.kind == J.KIND_CAMPAIGN_OPENED]
        with self.assertRaises(J.RetrievalCitationError) as ctx:
            J.retrieval_filter(campaign, supersession_basis=everything,
                               cite_event_ids=[self.withdrawn.event_id])
        self.assertIn("superseded out of retrieval", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
