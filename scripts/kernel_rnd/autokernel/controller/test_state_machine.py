#!/usr/bin/env python3
"""test_state_machine.py — the regression barrier for AK4's controller machine.

WHY THIS FILE EXISTS
--------------------
Each property below replaces a documented failure, and each one was visible in
the code that shipped with it — none of them was ASSERTED anywhere:

  * **journal-then-act.** AutoPilot could not tell a completed action from an
    action it was about to take. Here a recorder that raises leaves the machine
    exactly where it was, and the recorder is called while the machine still
    reports the FROM state — asserted, not assumed.
  * **control is verified, not requested** (§4 invariant 19). Pause was a silent
    no-op for months. Here a latch survives a simulated restart, an ack without
    its latch is a hard failure, a latch without its ack is a hard failure, and
    the machine holds no latch to write back over.
  * **BOOTSTRAP refuses an empty view over a non-empty journal** (§8.2 step 10).
    232 trials and ~16 days of compute vanished to a restart that came up empty
    with nothing objecting; the deliberate-rebase escape is tested separately so
    the two cannot be confused.
  * **stop states are reachable, evidenced and terminal** (§8.10), including the
    closure enumeration that "exhausted"/"all paths" may not stand in for.
  * **T3 is refused on the EDGE**, not only on the polite call path (AK5 owns it).

NO inference, NO benchmark, NO build, NO model call, NO process. Every file this
suite writes lives under a per-test temporary directory.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_state_machine.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_state_machine.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE so `state_machine.schemas` is the same module object
# the journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel.controller import state_machine as SM  # noqa: E402
from autokernel.evaluator import api as EV  # noqa: E402

V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
CAMPAIGN = "ak-llama_gpu-decode-20260803"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _anchor(*, commit: str = V8_COMMIT, binary: str = "bin-v8",
            linkage: str = "link-v8", backends=("llama_gpu",)) -> SM.AnchorIdentity:
    return SM.AnchorIdentity(
        source_tree="llama.cpp",
        branch="production-consolidated-v8",
        commit=commit,
        binary_sha256={b: _sha(f"{binary}-{b}") for b in backends},
        linkage_sha256={b: _sha(f"{linkage}-{b}") for b in backends},
    )


def _candidate(suffix: str = "0001", status: str = "banked") -> dict:
    """A minimal §7.3 candidate record that `Journal.append()` accepts.

    Shape mirrors `test_journal.py`'s fixture; it exists here so BOOTSTRAP's
    consistency assertion has a real non-empty journal to disagree with.
    """
    return {
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
        "dispatch": {
            "feature_flags": ["GGML_AK_WIDE_TILE"],
            "dispatch_predicate": "K >= 4096",
        },
        "affected_surface": {
            "derived_sha256": _sha("derived-surface"),
            "traced_sha256": None,
            "reconciled": False,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "evaluator": {
            "id": "P-AK-SEARCH-1/v1",
            "bundle_sha256": _sha("evaluator-bundle"),
        },
        "receipts": {
            "host_receipt": "rcpt-host-20260803T101500Z",
            "resource_claim_receipt": "rcpt-gpu-claim-0042",
        },
        "storage": {
            "footprint_gb": 3.4,
            "durability_class": "hash_and_provenance_only",
        },
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


#: Valid `detail` for every stop state, so the evidence gate is satisfied when
#: the test is about something else.
STOP_DETAIL = {
    SM.RELEASE_PACKAGE_READY: {"package_id": "akr-20260803-0001"},
    SM.PLATEAU_STOP: {
        "closed": [{"sub_scope": "gpu/decode/gemv", "gates_met": ["T0", "T1"]}],
        "deferred": [{"sub_scope": "gpu/prefill", "gates_unrun": ["T1", "T2"]}],
        "planner_health": {
            "proposal_skipped_count": 2,
            "repeated_fingerprint_count": 0,
            "degraded_ruled_out": True,
        },
    },
    SM.PLANNER_DEGRADED: {
        "signal": "repeated_fingerprint",
        "receipt": "akj-evt-0042",
    },
    SM.OPERATOR_STOP_REQUESTED: {"control": "pause", "control_id": "ctl-1"},
    SM.BUDGET_STOP: {"budget": "gpu_hours", "limit": 12.0, "consumed": 12.0},
    SM.DISK_PRESSURE: {
        "path": "/mnt/raid0", "free_bytes": 12, "floor_bytes": 1024,
    },
    SM.EXHAUSTED_SURFACE: {
        "closed": [{"sub_scope": "gpu/decode/gemv", "gates_met": ["T0", "T1", "T2"]}],
        "deferred": [],
    },
    SM.EVALUATOR_COVERAGE_GAP: {
        "missing_coverage_class": "iq1_dequant_oracle",
        "blocked_lineage": "llama.cpp/ak-decode",
        "owner": "operator",
        "deadline": "2026-08-17",
    },
    SM.RESOURCE_UNAVAILABLE: {"resource": "gfx90a:0", "claim_kind": "gpu_device"},
    SM.HOST_REBOOT_REQUIRED: {"uptime_seconds": 700000, "ceiling_seconds": 604800},
    SM.INTEGRITY_STOP: {
        "signal": "dispatch_trace_mismatch", "occurrences": 2, "receipt": "akj-evt-0099",
    },
    SM.OPERATOR_INPUT_REQUIRED: {
        "context": "two mechanisms explain the same 6% and both fit the budget",
        "options": [
            {"label": "A", "entails": "run the discriminating T1a", "cost": "20 min"},
            {"label": "B", "entails": "bank the cheaper one now", "cost": "0"},
        ],
        "recommendation": "A — it closes the mechanism question with a receipt",
        "default": "blocked until answered",
    },
    SM.ANCHOR_MOVED: {
        "recorded_anchor": {"commit": V8_COMMIT},
        "observed_anchor": {"commit": V7_COMMIT},
        "affected_backends": ["llama_gpu"],
    },
}


class _AppendRefusingJournal(J.Journal):
    """A journal whose append always fails. Used to prove the transition does
    not happen when its record cannot be made durable."""

    def append(self, *args, **kwargs):  # noqa: D102 - deliberate refusal
        raise OSError("simulated fsync failure")


class _RaisingRecorder:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def record(self, transition):
        self.calls += 1
        raise self.exc


class _WrongTransitionRecorder:
    """Returns a transition that is not the one it was asked to record."""

    def record(self, transition):
        return dataclasses.replace(transition, to_state=SM.PROPOSE)


class _OrderSpy:
    """Wraps a real recorder and captures the machine's state AT RECORD TIME.

    This is the whole journal-then-act assertion: if the machine has already
    moved when the record is written, the observed state is the destination.
    """

    def __init__(self, inner, machine_box) -> None:
        self.inner = inner
        self.box = machine_box
        self.observed: list = []

    def record(self, transition):
        self.observed.append(self.box[0].state)
        return self.inner.record(transition)


class _FakeReleaseGate:
    """AK5's seam, faked. Runs nothing; returns a sentinel."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_release(self, request):
        self.calls += 1
        return {"release_verdict": "FAKE", "request": request}


class ControllerTestCase(unittest.TestCase):
    """Shared fixture: an initialized journal plus a controller root."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal_root = os.path.join(self.tmp.name, "journal")
        self.controller_root = os.path.join(self.tmp.name, "controller")
        self.journal = J.Journal(self.journal_root, campaign_id=CAMPAIGN)
        self.journal.initialize()

    def machine(self, **kwargs) -> SM.ControllerStateMachine:
        kwargs.setdefault("journal_", self.journal)
        kwargs.setdefault("root", self.controller_root)
        kwargs.setdefault("campaign_id", CAMPAIGN)
        return SM.ControllerStateMachine(**kwargs)

    def at(self, machine, state: str) -> SM.ControllerStateMachine:
        """Walk `machine` to `state` along declared edges only."""
        path = {
            SM.BOOTSTRAP: (),
            SM.DISCOVER: (SM.DISCOVER,),
            SM.SELECT_TARGET: (SM.DISCOVER, SM.SELECT_TARGET),
            SM.PROPOSE: (SM.DISCOVER, SM.SELECT_TARGET, SM.PROPOSE),
            SM.PRE_RUN_CRITIC: (
                SM.DISCOVER, SM.SELECT_TARGET, SM.PROPOSE, SM.PRE_RUN_CRITIC,
            ),
            SM.CHAMPION_GUARD: (
                SM.DISCOVER, SM.SELECT_TARGET, SM.PROPOSE, SM.PRE_RUN_CRITIC,
                SM.MUTATE, SM.BUILD, SM.T0_GATE, SM.T1_SEARCH_EVAL,
                SM.POST_RUN_CRITIC, SM.BANK_EVENT, SM.UPDATE_SEARCH_STATE,
                SM.CHAMPION_GUARD,
            ),
        }[state]
        for step in path:
            machine.transition(step, trigger="test", reason="walk to fixture state")
        self.assertEqual(machine.state, state)
        return machine


# =============================================================================
# The declared graph (§8.1, §8.10)
# =============================================================================

class TestDeclaredGraph(ControllerTestCase):

    def test_every_design_state_is_declared(self):
        for name in (
            "BOOTSTRAP", "DISCOVER", "SELECT_TARGET", "PROPOSE", "PRE_RUN_CRITIC",
            "MUTATE", "BUILD", "T0_GATE", "T1_SEARCH_EVAL", "POST_RUN_CRITIC",
            "BANK_EVENT", "UPDATE_SEARCH_STATE", "CHAMPION_GUARD",
            "SEAL", "T3_RELEASE_GATE", "PACKAGE",
        ):
            self.assertIn(getattr(SM, name), SM.STATES, name)

    def test_every_8_10_stop_state_is_declared(self):
        expected = {
            "RELEASE_PACKAGE_READY", "PLATEAU_STOP", "PLANNER_DEGRADED",
            "OPERATOR_STOP_REQUESTED", "BUDGET_STOP", "DISK_PRESSURE",
            "EXHAUSTED_SURFACE", "EVALUATOR_COVERAGE_GAP", "RESOURCE_UNAVAILABLE",
            "HOST_REBOOT_REQUIRED", "INTEGRITY_STOP", "OPERATOR_INPUT_REQUIRED",
            "ANCHOR_MOVED",
        }
        self.assertEqual(set(SM.STOP_STATES), expected)

    def test_edges_are_total_and_closed(self):
        self.assertEqual(set(SM.EDGES), set(SM.STATES))
        for state, targets in SM.EDGES.items():
            for target in targets:
                self.assertIn(target, SM.STATES, f"{state} -> {target}")
            self.assertEqual(len(set(targets)), len(targets), state)

    def test_every_stop_state_is_reachable_in_the_graph(self):
        reachable = {t for targets in SM.EDGES.values() for t in targets}
        for stop in SM.STOP_STATES:
            self.assertIn(stop, reachable, stop)

    def test_every_stop_state_is_terminal_in_the_graph(self):
        for stop in SM.STOP_STATES:
            self.assertEqual(SM.EDGES[stop], (), stop)

    def test_every_stop_state_has_a_recovery_class_and_evidence_row(self):
        self.assertEqual(set(SM.STOP_RECOVERY), set(SM.STOP_STATES))
        self.assertEqual(set(SM.STOP_EVIDENCE_REQUIREMENTS), set(SM.STOP_STATES))
        self.assertEqual(set(SM.REOPEN_EDGES), set(SM.STOP_STATES))

    def test_search_closure_stops_are_not_reachable_from_arbitrary_states(self):
        # Declaring the surface closed from inside BUILD would be a claim about
        # evidence the machine is not holding there.
        for stop in SM.SEARCH_CLOSURE_STOPS:
            self.assertNotIn(stop, SM.EDGES[SM.BUILD], stop)
            self.assertIn(stop, SM.EDGES[SM.DISCOVER], stop)


class TestIllegalTransitions(ControllerTestCase):

    def test_undeclared_edge_raises(self):
        machine = self.machine()
        with self.assertRaises(SM.IllegalTransition):
            machine.transition(SM.BUILD, trigger="t", reason="skip the middle")
        self.assertEqual(machine.state, SM.BOOTSTRAP)

    def test_unknown_state_raises(self):
        machine = self.machine()
        with self.assertRaises(SM.IllegalTransition):
            machine.transition("PROFIT", trigger="t", reason="not a state")

    def test_stop_state_is_terminal_at_runtime(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        machine.stop(
            SM.BUDGET_STOP, reason="gpu hours exhausted",
            detail=STOP_DETAIL[SM.BUDGET_STOP],
        )
        self.assertTrue(machine.is_stopped())
        for target in (SM.DISCOVER, SM.BOOTSTRAP, SM.SELECT_TARGET):
            with self.assertRaises(SM.IllegalTransition):
                machine.transition(target, trigger="t", reason="escape a stop")
        self.assertEqual(machine.state, SM.BUDGET_STOP)

    def test_transition_is_not_decided_by_a_supplied_label(self):
        # The machine consults EDGES and nothing else; a trigger/reason string is
        # recorded, never consulted.
        machine = self.machine()
        with self.assertRaises(SM.IllegalTransition):
            machine.transition(
                SM.PACKAGE, trigger="the planner said so",
                reason="the model is confident this is fine",
            )


# =============================================================================
# Journal-then-act
# =============================================================================

class TestJournalThenAct(ControllerTestCase):

    def test_record_observes_the_pre_transition_state(self):
        box: list = []
        ledger = SM.TransitionLedger(
            os.path.join(self.controller_root, SM.LEDGER_FILENAME)
        )
        inner = SM.JournalTransitionRecorder(self.journal, ledger, campaign_id=CAMPAIGN)
        spy = _OrderSpy(inner, box)
        machine = self.machine(recorder=spy)
        box.append(machine)

        machine.transition(SM.DISCOVER, trigger="t", reason="first step")
        machine.transition(SM.SELECT_TARGET, trigger="t", reason="second step")
        self.assertEqual(spy.observed, [SM.BOOTSTRAP, SM.DISCOVER])

    def test_failing_recorder_blocks_the_transition(self):
        recorder = _RaisingRecorder(OSError("no space left on device"))
        machine = self.machine(recorder=recorder)
        with self.assertRaises(SM.TransitionNotRecorded):
            machine.transition(SM.DISCOVER, trigger="t", reason="should not happen")
        self.assertEqual(machine.state, SM.BOOTSTRAP)
        self.assertEqual(machine.seq, 0)
        self.assertEqual(recorder.calls, 1)
        self.assertEqual(machine.ledger.read().transitions, ())

    def test_failing_journal_append_blocks_a_stop_transition(self):
        refusing = _AppendRefusingJournal(self.journal_root, campaign_id=CAMPAIGN)
        machine = self.at(
            SM.ControllerStateMachine(
                journal_=refusing, root=self.controller_root, campaign_id=CAMPAIGN
            ),
            SM.DISCOVER,
        )
        with self.assertRaises(SM.TransitionNotRecorded):
            machine.stop(
                SM.BUDGET_STOP, reason="gpu hours exhausted",
                detail=STOP_DETAIL[SM.BUDGET_STOP],
            )
        self.assertEqual(machine.state, SM.DISCOVER)
        # Nothing landed anywhere: not in the journal's stop view, not in the
        # ledger. A transition that failed to journal did not happen.
        views = J.rebuild_views(self.journal.read_all())
        self.assertEqual(views.stop_states, ())
        self.assertEqual([t.to_state for t in machine.ledger.read().transitions],
                         [SM.DISCOVER])

    def test_failing_ledger_write_blocks_the_transition(self):
        machine = self.machine()
        # Replace the ledger file with a directory: O_WRONLY on it raises, on
        # every platform this project runs on, and as any uid.
        os.unlink(machine.ledger.path)
        os.mkdir(machine.ledger.path)
        with self.assertRaises(SM.TransitionNotRecorded):
            machine.transition(SM.DISCOVER, trigger="t", reason="should not happen")
        self.assertEqual(machine.state, SM.BOOTSTRAP)
        self.assertEqual(machine.seq, 0)

    def test_recorder_returning_a_different_transition_is_refused(self):
        machine = self.machine(recorder=_WrongTransitionRecorder())
        with self.assertRaises(SM.TransitionNotRecorded):
            machine.transition(SM.DISCOVER, trigger="t", reason="mismatched record")
        self.assertEqual(machine.state, SM.BOOTSTRAP)

    def test_successful_transition_is_durable_before_it_is_visible(self):
        machine = self.machine()
        recorded = machine.transition(SM.DISCOVER, trigger="t", reason="first step")
        self.assertEqual(machine.state, SM.DISCOVER)
        self.assertEqual(machine.seq, 1)
        on_disk = machine.ledger.read().transitions
        self.assertEqual(len(on_disk), 1)
        self.assertEqual(on_disk[0].to_state, SM.DISCOVER)
        self.assertEqual(on_disk[0].receipt, recorded.receipt)

    def test_stop_transition_lands_in_the_journal_stop_view(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        machine.stop(
            SM.HOST_REBOOT_REQUIRED, reason="uptime ceiling reached",
            detail=STOP_DETAIL[SM.HOST_REBOOT_REQUIRED],
        )
        views = J.rebuild_views(self.journal.read_all())
        self.assertEqual(len(views.stop_states), 1)
        self.assertEqual(views.stop_states[0]["state"], SM.HOST_REBOOT_REQUIRED)
        self.assertEqual(
            J.check_view_consistency(self.journal.read_all(), views).outcome, S.PASS
        )

    def test_non_stop_transitions_do_not_pollute_the_stop_view(self):
        machine = self.at(self.machine(), SM.PROPOSE)
        views = J.rebuild_views(self.journal.read_all())
        self.assertEqual(views.stop_states, ())
        self.assertEqual(len(machine.ledger.read().transitions), 3)


# =============================================================================
# The transition ledger
# =============================================================================

class TestTransitionLedger(ControllerTestCase):

    def test_restore_replays_the_ledger(self):
        first = self.at(self.machine(), SM.SELECT_TARGET)
        self.assertEqual(first.seq, 2)
        second = self.machine()
        self.assertEqual(second.state, SM.SELECT_TARGET)
        self.assertEqual(second.seq, 2)
        self.assertEqual(second.restore_report.transition_count, 2)
        self.assertEqual(second.restore_report.discarded_tail_bytes, 0)

    def test_torn_tail_is_discarded_and_reported(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        with open(machine.ledger.path, "ab") as handle:
            handle.write(b'{"seq": 2, "from_state": "DISCOVER"')  # no newline
        restored = self.machine()
        self.assertEqual(restored.state, SM.DISCOVER)
        self.assertEqual(restored.seq, 1)
        self.assertGreater(restored.restore_report.discarded_tail_bytes, 0)

    def test_unparseable_complete_line_raises(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        with open(machine.ledger.path, "ab") as handle:
            handle.write(b"{not json}\n")
        with self.assertRaises(SM.LedgerCorruption):
            self.machine()

    def test_ledger_that_does_not_describe_one_machine_raises(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        orphan = SM.Transition(
            seq=2, from_state=SM.PACKAGE, to_state=SM.RELEASE_PACKAGE_READY,
            trigger="t", reason="spliced in", at="2026-08-03T00:00:00Z",
        )
        with open(machine.ledger.path, "ab") as handle:
            handle.write((S.canonical_json(orphan.to_dict()) + "\n").encode("utf-8"))
        with self.assertRaises(SM.LedgerCorruption):
            self.machine()

    def test_transition_rejects_an_unserializable_detail(self):
        with self.assertRaises((TypeError, ValueError)):
            SM.Transition(
                seq=1, from_state=SM.BOOTSTRAP, to_state=SM.DISCOVER, trigger="t",
                reason="r", at="2026-08-03T00:00:00Z", detail={"fn": object()},
            )


# =============================================================================
# Stop states (§8.10)
# =============================================================================

class TestStopStates(ControllerTestCase):

    def test_every_stop_state_is_reachable_at_runtime_and_terminal(self):
        for stop in SM.STOP_STATES:
            with self.subTest(stop=stop):
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                jrn = J.Journal(os.path.join(tmp.name, "j"), campaign_id=CAMPAIGN)
                jrn.initialize()
                gate = _FakeReleaseGate()
                machine = SM.ControllerStateMachine(
                    journal_=jrn, root=os.path.join(tmp.name, "c"),
                    campaign_id=CAMPAIGN, release_gate=gate,
                )
                if stop == SM.RELEASE_PACKAGE_READY:
                    for step in (
                        SM.DISCOVER, SM.SELECT_TARGET, SM.PROPOSE, SM.PRE_RUN_CRITIC,
                        SM.MUTATE, SM.BUILD, SM.T0_GATE, SM.T1_SEARCH_EVAL,
                        SM.POST_RUN_CRITIC, SM.BANK_EVENT, SM.UPDATE_SEARCH_STATE,
                        SM.CHAMPION_GUARD, SM.SEAL, SM.T3_RELEASE_GATE, SM.PACKAGE,
                    ):
                        machine.transition(step, trigger="test", reason="walk")
                else:
                    machine.transition(SM.DISCOVER, trigger="test", reason="walk")
                machine.stop(stop, reason=f"reached {stop}", detail=STOP_DETAIL[stop])
                self.assertEqual(machine.state, stop)
                self.assertTrue(machine.is_stopped())
                with self.assertRaises(SM.IllegalTransition):
                    machine.transition(SM.DISCOVER, trigger="t", reason="escape")

    def test_stop_without_its_required_evidence_is_refused(self):
        for stop in SM.STOP_STATES:
            with self.subTest(stop=stop):
                machine = self.at(self.machine(), SM.DISCOVER)
                with self.assertRaises(SM.StopEvidenceMissing):
                    machine.stop(stop, reason="because", detail={})
                self.assertEqual(machine.state, SM.DISCOVER)
                # Nothing was journaled either.
                self.assertEqual(
                    J.rebuild_views(self.journal.read_all()).stop_states, ()
                )
                self.setUp()

    def test_detail_that_cannot_be_read_is_could_not_check_not_a_pass(self):
        check = SM.check_stop_evidence(SM.BUDGET_STOP, "out of budget", None)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(check.passed)

    def test_closure_enumeration_rejects_reserved_phrases(self):
        detail = dict(STOP_DETAIL[SM.EXHAUSTED_SURFACE])
        check = SM.check_closure_enumeration("we tried all paths", detail)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("all paths" in r for r in check.reasons))

    def test_closure_enumeration_requires_the_deferred_list(self):
        detail = {"closed": [{"sub_scope": "x", "gates_met": ["T0"]}]}
        check = SM.check_closure_enumeration("closed sub-scope x", detail)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("deferred" in r for r in check.reasons))
        # An EMPTY deferred list is a legitimate answer; an absent one is not.
        detail["deferred"] = []
        self.assertEqual(
            SM.check_closure_enumeration("closed sub-scope x", detail).outcome, S.PASS
        )

    def test_plateau_requires_planner_degraded_to_be_ruled_out(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        detail = dict(STOP_DETAIL[SM.PLATEAU_STOP])
        detail["planner_health"] = {
            "proposal_skipped_count": 31,
            "repeated_fingerprint_count": 12,
            "degraded_ruled_out": False,
        }
        with self.assertRaises(SM.StopEvidenceMissing):
            machine.stop(SM.PLATEAU_STOP, reason="no readiness movement", detail=detail)
        self.assertEqual(machine.state, SM.DISCOVER)

    def test_operator_input_required_needs_a_real_decision_package(self):
        detail = dict(STOP_DETAIL[SM.OPERATOR_INPUT_REQUIRED])
        detail["options"] = [detail["options"][0]]
        check = SM.check_stop_evidence(
            SM.OPERATOR_INPUT_REQUIRED, "how should I proceed?", detail
        )
        self.assertEqual(check.outcome, S.FAIL)

    def test_stop_request_origin_buys_nothing(self):
        # §8.4.0 / AK-D38: authorship is not evidence. The operator's request and
        # the planner's face the identical gate.
        for origin in ("operator", "planner"):
            with self.subTest(origin=origin):
                machine = self.at(self.machine(), SM.DISCOVER)
                request = SM.StopRequest(
                    state=SM.EXHAUSTED_SURFACE,
                    reason="we have exhausted all paths",
                    detail=STOP_DETAIL[SM.EXHAUSTED_SURFACE],
                    origin=origin,
                )
                with self.assertRaises(SM.StopEvidenceMissing):
                    machine.dispose_stop_request(request)
                self.assertEqual(machine.state, SM.DISCOVER)
                self.setUp()

    def test_dispose_stop_request_records_its_origin(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        transition = machine.dispose_stop_request(SM.StopRequest(
            state=SM.EXHAUSTED_SURFACE,
            reason="closed the gemv sub-scope; prefill deferred",
            detail=STOP_DETAIL[SM.EXHAUSTED_SURFACE],
            origin="planner",
        ))
        self.assertEqual(transition.trigger, "stop_request:planner")
        self.assertEqual(machine.state, SM.EXHAUSTED_SURFACE)


# =============================================================================
# Operator controls (§4 invariant 19)
# =============================================================================

class TestOperatorControls(ControllerTestCase):

    def _pause(self, machine, control_id="ctl-0001"):
        return machine.submit_control(
            SM.CONTROL_PAUSE, control_id=control_id, requested_by="operator",
            reason="dinner",
        )

    def test_clean_iteration_proceeds(self):
        machine = self.machine()
        decision = machine.begin_iteration()
        self.assertTrue(decision.proceed)
        self.assertEqual(decision.state, SM.BOOTSTRAP)

    def test_control_is_acked_in_the_journal_and_latched_on_disk(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        latch = self._pause(machine)
        acks = [
            e for e in self.journal.read_all()
            if e.kind == J.KIND_OPERATOR_CONTROL_ACK
        ]
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0].payload["control"], SM.CONTROL_PAUSE)
        self.assertEqual(latch.acked_event_id, acks[0].event_id)
        self.assertTrue(os.path.exists(machine.latch_store.path))

    def test_latched_control_halts_the_loop_at_the_next_iteration(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        self._pause(machine)
        decision = machine.begin_iteration()
        self.assertFalse(decision.proceed)
        self.assertEqual(machine.state, SM.OPERATOR_STOP_REQUESTED)
        self.assertEqual(decision.control, SM.CONTROL_PAUSE)

    def test_a_latched_halt_survives_a_simulated_restart(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        self._pause(machine)
        machine.begin_iteration()
        self.assertEqual(machine.state, SM.OPERATOR_STOP_REQUESTED)
        del machine  # the process dies here

        restarted = self.machine()
        self.assertEqual(restarted.state, SM.OPERATOR_STOP_REQUESTED)
        self.assertTrue(restarted.restore_report.latch_present)
        decision = restarted.begin_iteration()
        self.assertFalse(decision.proceed)
        self.assertEqual(decision.control_id, "ctl-0001")
        # And it cannot be shrugged off by reopening.
        with self.assertRaises(SM.ControlLatchError):
            restarted.reopen(reason="carry on", authorized_by="the loop")

    def test_only_a_resume_clears_the_halt(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        self._pause(machine)
        machine.begin_iteration()
        machine.resume_control(
            "ctl-0001", requested_by="operator", reason="back at the desk"
        )
        self.assertIsNone(machine.latch_store.read())
        dispositions = [
            e.payload["disposition"] for e in self.journal.read_all()
            if e.kind == J.KIND_OPERATOR_CONTROL_ACK
        ]
        self.assertEqual(dispositions, [SM.DISPOSITION_LATCHED, SM.DISPOSITION_RELEASED])
        transition = machine.reopen(reason="resumed", authorized_by="operator")
        self.assertEqual(transition.to_state, SM.BOOTSTRAP)
        self.assertEqual(machine.state, SM.BOOTSTRAP)
        self.assertTrue(machine.begin_iteration().proceed)

    def test_a_latch_whose_ack_does_not_resolve_is_a_hard_failure(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        self._pause(machine)
        raw = json.loads(Path(machine.latch_store.path).read_text(encoding="utf-8"))
        raw["acked_event_id"] = "akj-evt-does-not-exist"
        Path(machine.latch_store.path).write_text(
            json.dumps(raw), encoding="utf-8"
        )
        with self.assertRaises(SM.UnackedControlError):
            machine.begin_iteration()

    def test_an_ack_whose_latch_is_missing_is_a_hard_failure(self):
        # The crash-between-ack-and-latch window. Treating it as "no control
        # pending" is exactly how a pause becomes a silent no-op.
        machine = self.at(self.machine(), SM.DISCOVER)
        self._pause(machine)
        os.unlink(machine.latch_store.path)
        with self.assertRaises(SM.UnackedControlError):
            machine.begin_iteration()
        with self.assertRaises(SM.UnackedControlError):
            self.machine().begin_iteration()

    def test_a_latch_citing_another_controls_ack_is_a_hard_failure(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        self._pause(machine, control_id="ctl-0001")
        machine.resume_control("ctl-0001", requested_by="operator", reason="ok")
        first_ack = [
            e for e in self.journal.read_all()
            if e.kind == J.KIND_OPERATOR_CONTROL_ACK
        ][0]
        self._pause(machine, control_id="ctl-0002")
        raw = json.loads(Path(machine.latch_store.path).read_text(encoding="utf-8"))
        raw["acked_event_id"] = first_ack.event_id
        Path(machine.latch_store.path).write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(SM.UnackedControlError):
            machine.begin_iteration()

    def test_a_second_control_cannot_overwrite_a_latched_one(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        self._pause(machine, control_id="ctl-0001")
        with self.assertRaises(SM.ControlLatchError):
            machine.submit_control(
                SM.CONTROL_ABORT, control_id="ctl-0002", requested_by="loop",
                reason="overwrite the operator",
            )
        self.assertEqual(machine.latch_store.read().control_id, "ctl-0001")

    def test_resubmitting_the_same_control_is_idempotent(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        first = self._pause(machine)
        again = self._pause(machine)
        self.assertEqual(first.to_dict(), again.to_dict())
        acks = [
            e for e in self.journal.read_all()
            if e.kind == J.KIND_OPERATOR_CONTROL_ACK
        ]
        self.assertEqual(len(acks), 1)

    def test_resume_is_not_a_submittable_control(self):
        machine = self.machine()
        with self.assertRaises(ValueError):
            machine.submit_control(
                SM.CONTROL_RESUME, control_id="ctl-1", requested_by="loop", reason="r"
            )

    def test_resuming_an_unlatched_control_is_refused(self):
        machine = self.machine()
        with self.assertRaises(SM.ControlLatchError):
            machine.resume_control("ctl-9", requested_by="operator", reason="r")

    def test_empty_latch_file_is_not_an_absent_latch(self):
        machine = self.machine()
        os.makedirs(os.path.dirname(machine.latch_store.path), exist_ok=True)
        Path(machine.latch_store.path).write_text("   ", encoding="utf-8")
        with self.assertRaises(SM.ControlLatchError):
            machine.latch_store.read()

    def test_the_machine_caches_no_control_state(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        self._pause(machine)
        machine.begin_iteration()
        self.assertEqual(SM.audit_no_cached_control_state(machine).outcome, S.PASS)

        class _Cacher:
            def __init__(self, latch):
                self.control_latch = latch

        cached = _Cacher(machine.latch_store.read())
        self.assertEqual(SM.audit_no_cached_control_state(cached).outcome, S.FAIL)
        self.assertEqual(
            SM.audit_no_cached_control_state(object()).outcome, S.COULD_NOT_CHECK
        )

    def test_the_latch_store_has_no_write_back_method(self):
        # The original defect was a snapshot written back over the operator's
        # change. There must be no API that can express it.
        for forbidden in ("save", "flush", "sync", "write", "update", "set"):
            self.assertFalse(
                hasattr(SM.ControlLatchStore, forbidden),
                f"ControlLatchStore.{forbidden} would allow a cached write-back",
            )


# =============================================================================
# BOOTSTRAP (§8.2 step 10)
# =============================================================================

class TestBootstrap(ControllerTestCase):

    def test_clean_bootstrap_reaches_discover_and_records_the_anchor(self):
        machine = self.machine()
        report = machine.bootstrap(anchor=_anchor())
        self.assertEqual(report.view_check.outcome, S.PASS)
        self.assertEqual(machine.state, SM.DISCOVER)
        self.assertEqual(machine.anchor_store.read().commit, V8_COMMIT)

    def test_refuses_on_an_empty_view_over_a_non_empty_journal(self):
        self.journal.append(J.KIND_CANDIDATE_RECORDED, _candidate())
        events = self.journal.read_all()
        good = J.rebuild_views(events)
        self.assertTrue(good.candidates)
        self.assertTrue(good.frontier)
        # The AutoPilot shape: a derived store that came up empty while the
        # journal was full.
        empty = dataclasses.replace(good, candidates={}, frontier=())

        machine = self.machine()
        with self.assertRaises(SM.BootstrapRefused) as caught:
            machine.bootstrap(anchor=_anchor(), views=empty)
        self.assertIn("EMPTY", str(caught.exception))
        self.assertEqual(machine.state, SM.BOOTSTRAP)
        self.assertIsNone(machine.anchor_store.read())

    def test_the_rebase_escape_lets_it_through_and_lands_on_the_record(self):
        self.journal.append(J.KIND_CANDIDATE_RECORDED, _candidate())
        events = self.journal.read_all()
        empty = dataclasses.replace(
            J.rebuild_views(events), candidates={}, frontier=()
        )
        machine = self.machine()
        report = machine.bootstrap(
            anchor=_anchor(), views=empty, deliberate_rebase=True,
            rebase_reason="operator rebased the campaign onto the new production tip",
        )
        self.assertEqual(report.view_check.outcome, S.FAIL)
        self.assertTrue(report.deliberate_rebase)
        self.assertEqual(machine.state, SM.DISCOVER)
        rebases = [
            e for e in self.journal.read_all() if e.kind == J.KIND_VIEW_REBASED
        ]
        self.assertEqual(len(rebases), 1)
        self.assertIn("operator rebased", rebases[0].payload["rebase_reason"])

    def test_a_rebase_without_a_reason_is_refused(self):
        self.journal.append(J.KIND_CANDIDATE_RECORDED, _candidate())
        empty = dataclasses.replace(
            J.rebuild_views(self.journal.read_all()), candidates={}, frontier=()
        )
        machine = self.machine()
        with self.assertRaises(ValueError):
            machine.bootstrap(
                anchor=_anchor(), views=empty, deliberate_rebase=True,
                rebase_reason="   ",
            )
        self.assertEqual(machine.state, SM.BOOTSTRAP)

    def test_could_not_check_is_not_covered_by_the_rebase_escape(self):
        self.journal.append(J.KIND_CANDIDATE_RECORDED, _candidate())
        stale = J.rebuild_views(self.journal.read_all())
        self.journal.append(J.KIND_CANDIDATE_RECORDED, _candidate("0002"))
        machine = self.machine()
        # "I meant to empty the views" is not an answer to "I cannot tell whether
        # these views belong to these events".
        with self.assertRaises(SM.BootstrapRefused):
            machine.bootstrap(
                anchor=_anchor(), views=stale, deliberate_rebase=True,
                rebase_reason="deliberate",
            )

    def test_bootstrap_refuses_while_a_halt_is_latched(self):
        machine = self.machine()
        machine.submit_control(
            SM.CONTROL_DRAIN, control_id="ctl-7", requested_by="operator",
            reason="host maintenance",
        )
        with self.assertRaises(SM.ControlLatchError):
            machine.bootstrap(anchor=_anchor())
        self.assertEqual(machine.state, SM.BOOTSTRAP)

    def test_bootstrap_only_runs_in_bootstrap(self):
        machine = self.machine()
        machine.bootstrap(anchor=_anchor())
        with self.assertRaises(SM.IllegalTransition):
            machine.bootstrap(anchor=_anchor())


# =============================================================================
# Anchor identity (§8.9, AK-D22)
# =============================================================================

class TestAnchorIdentity(ControllerTestCase):

    def test_identical_anchor_passes(self):
        self.assertEqual(
            SM.check_anchor_identity(_anchor(), _anchor()).outcome, S.PASS
        )

    def test_moved_commit_fails(self):
        check = SM.check_anchor_identity(_anchor(), _anchor(commit=V7_COMMIT))
        self.assertEqual(check.outcome, S.FAIL)

    def test_rebuilt_binary_fails(self):
        check = SM.check_anchor_identity(_anchor(), _anchor(binary="rebuilt"))
        self.assertEqual(check.outcome, S.FAIL)

    def test_an_unobserved_backend_is_could_not_check_not_a_pass(self):
        recorded = _anchor(backends=("llama_gpu", "llama_cpu"))
        observed = _anchor(backends=("llama_gpu",))
        check = SM.check_anchor_identity(recorded, observed)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(check.passed)

    def test_a_real_mismatch_outranks_an_incomplete_observation(self):
        recorded = _anchor(backends=("llama_gpu", "llama_cpu"))
        observed = _anchor(commit=V7_COMMIT, backends=("llama_gpu",))
        self.assertEqual(SM.check_anchor_identity(recorded, observed).outcome, S.FAIL)

    def test_no_recorded_anchor_is_could_not_check(self):
        self.assertEqual(
            SM.check_anchor_identity(None, _anchor()).outcome, S.COULD_NOT_CHECK
        )

    def test_campaign_boundary_stops_on_a_moved_anchor(self):
        machine = self.machine()
        machine.bootstrap(anchor=_anchor())
        check = machine.campaign_boundary(observed_anchor=_anchor(commit=V7_COMMIT))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertEqual(machine.state, SM.ANCHOR_MOVED)
        stop = J.rebuild_views(self.journal.read_all()).stop_states[-1]
        self.assertEqual(stop["state"], SM.ANCHOR_MOVED)
        self.assertEqual(
            stop["detail"]["supersession_marker"], "superseded_by_anchor_move"
        )

    def test_campaign_boundary_passes_on_an_unmoved_anchor(self):
        machine = self.machine()
        machine.bootstrap(anchor=_anchor())
        self.assertEqual(
            machine.campaign_boundary(observed_anchor=_anchor()).outcome, S.PASS
        )
        self.assertEqual(machine.state, SM.DISCOVER)

    def test_an_uncheckable_anchor_refuses_rather_than_continuing(self):
        machine = self.machine()
        machine.bootstrap(anchor=_anchor(backends=("llama_gpu", "llama_cpu")))
        with self.assertRaises(SM.AnchorUncheckable):
            machine.campaign_boundary(observed_anchor=_anchor(backends=("llama_gpu",)))
        self.assertEqual(machine.state, SM.DISCOVER)

    def test_anchor_identity_rejects_a_half_identity(self):
        with self.assertRaises(ValueError):
            SM.AnchorIdentity(
                source_tree="llama.cpp", branch="production-consolidated-v8",
                commit=V8_COMMIT,
                binary_sha256={"llama_gpu": _sha("b"), "llama_cpu": _sha("c")},
                linkage_sha256={"llama_gpu": _sha("l")},
            )

    def test_anchor_identity_rejects_an_undeclared_backend(self):
        with self.assertRaises(ValueError):
            SM.AnchorIdentity(
                source_tree="llama.cpp", branch="production-consolidated-v8",
                commit=V8_COMMIT,
                binary_sha256={"vllm": _sha("b")},
                linkage_sha256={"vllm": _sha("l")},
            )


# =============================================================================
# Recovery
# =============================================================================

class TestRecovery(ControllerTestCase):

    def test_reopen_re_enters_at_bootstrap(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        machine.stop(
            SM.RESOURCE_UNAVAILABLE, reason="no gpu device claim available",
            detail=STOP_DETAIL[SM.RESOURCE_UNAVAILABLE],
        )
        machine.reopen(reason="device claim released", authorized_by="operator")
        self.assertEqual(machine.state, SM.BOOTSTRAP)

    def test_reopen_requires_authorization(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        machine.stop(
            SM.INTEGRITY_STOP, reason="repeated dispatch trace mismatch",
            detail=STOP_DETAIL[SM.INTEGRITY_STOP],
        )
        with self.assertRaises(ValueError):
            machine.reopen(reason="looks fine now", authorized_by="")

    def test_release_package_ready_is_not_reopenable(self):
        gate = _FakeReleaseGate()
        machine = self.machine(release_gate=gate)
        for step in (
            SM.DISCOVER, SM.SELECT_TARGET, SM.PROPOSE, SM.PRE_RUN_CRITIC, SM.MUTATE,
            SM.BUILD, SM.T0_GATE, SM.T1_SEARCH_EVAL, SM.POST_RUN_CRITIC,
            SM.BANK_EVENT, SM.UPDATE_SEARCH_STATE, SM.CHAMPION_GUARD, SM.SEAL,
            SM.T3_RELEASE_GATE, SM.PACKAGE,
        ):
            machine.transition(step, trigger="test", reason="walk")
        machine.stop(
            SM.RELEASE_PACKAGE_READY, reason="package assembled",
            detail=STOP_DETAIL[SM.RELEASE_PACKAGE_READY],
        )
        with self.assertRaises(SM.IllegalTransition):
            machine.reopen(reason="one more round", authorized_by="operator")

    def test_reopen_is_not_available_from_a_live_state(self):
        machine = self.at(self.machine(), SM.DISCOVER)
        with self.assertRaises(SM.IllegalTransition):
            machine.reopen(reason="no", authorized_by="operator")


# =============================================================================
# The AK5 release seam
# =============================================================================

class TestReleaseSeam(ControllerTestCase):

    def test_t3_is_refused_on_the_edge_when_no_runner_is_wired(self):
        machine = self.at(self.machine(), SM.CHAMPION_GUARD)
        machine.request_freeze(
            requested_by="operator", reason="quarterly freeze window",
        )
        self.assertEqual(machine.state, SM.SEAL)
        with self.assertRaises(EV.TierNotOwned):
            machine.transition(SM.T3_RELEASE_GATE, trigger="t", reason="sneak past")
        with self.assertRaises(EV.TierNotOwned):
            machine.run_release_gate({"sealed": True})
        self.assertEqual(machine.state, SM.SEAL)

    def test_the_refusal_names_ak5(self):
        machine = self.at(self.machine(), SM.CHAMPION_GUARD)
        machine.request_freeze(requested_by="operator", reason="freeze window")
        with self.assertRaises(EV.TierNotOwned) as caught:
            machine.run_release_gate({})
        self.assertIn(EV.RELEASE_TIER_OWNER, str(caught.exception))

    def test_a_wired_runner_is_dispatched(self):
        gate = _FakeReleaseGate()
        machine = self.at(self.machine(release_gate=gate), SM.CHAMPION_GUARD)
        machine.request_freeze(requested_by="operator", reason="freeze window")
        transition, outcome = machine.run_release_gate({"sealed": True})
        self.assertEqual(transition.to_state, SM.T3_RELEASE_GATE)
        self.assertEqual(machine.state, SM.T3_RELEASE_GATE)
        self.assertEqual(gate.calls, 1)
        self.assertEqual(outcome["release_verdict"], "FAKE")

    def test_the_loop_cannot_seal_itself(self):
        machine = self.at(self.machine(), SM.CHAMPION_GUARD)
        with self.assertRaises(ValueError):
            machine.request_freeze(requested_by="", reason="i decided")
        self.assertEqual(machine.state, SM.CHAMPION_GUARD)

    def test_a_failed_release_gate_returns_to_research(self):
        gate = _FakeReleaseGate()
        machine = self.at(self.machine(release_gate=gate), SM.CHAMPION_GUARD)
        machine.request_freeze(requested_by="operator", reason="freeze window")
        machine.run_release_gate({})
        machine.transition(
            SM.CHAMPION_GUARD, trigger="release_gate_fail",
            reason="T3 returned FAIL; research continues",
        )
        self.assertEqual(machine.state, SM.CHAMPION_GUARD)


# =============================================================================
# Construction preconditions
# =============================================================================

class TestConstruction(ControllerTestCase):

    def test_uninitialized_journal_is_refused(self):
        empty = J.Journal(os.path.join(self.tmp.name, "never-initialized"))
        with self.assertRaises(SM.ControllerError):
            SM.ControllerStateMachine(
                journal_=empty, root=os.path.join(self.tmp.name, "c2")
            )

    def test_missing_root_raises(self):
        with self.assertRaises(ValueError):
            SM.ControllerStateMachine(journal_=self.journal, root="")

    def test_a_non_journal_is_refused(self):
        with self.assertRaises(TypeError):
            SM.ControllerStateMachine(journal_=object(), root=self.controller_root)

    def test_machine_has_no_slot_for_a_latch(self):
        slots = set(SM.ControllerStateMachine.__slots__)
        self.assertNotIn("_latch", slots)
        self.assertIn("_latch_store", slots)
        machine = self.machine()
        with self.assertRaises(AttributeError):
            machine._latch = machine.latch_store.read()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
