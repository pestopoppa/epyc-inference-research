#!/usr/bin/env python3
"""test_guards.py — the regression barrier for AK4's deterministic stop guards.

WHY THIS FILE EXISTS
--------------------
Each property below replaces a documented failure, and none of them is asserted
anywhere else in the package:

  * **closure inflation** (§8.10). The bare word "exhausted" is rejected as a
    WORD, in the reason and in every enumerated sub-scope and gate string — the
    phrase list in `state_machine` matches none of them, and "the surface is
    exhausted" is the sentence the rule is actually about.
  * **plateau/degraded conflation** (§8.10, §8.4.1). A closure decision cannot be
    built without a CONTINUE from `guard_planner_degraded` computed over the SAME
    health snapshot, proven by content hash; a stale clean verdict is refused.
  * **a dead gate reading as a closed surface** (§12, AK-D27). Both closure
    guards return COULD_NOT_EVALUATE when the accept-side historical-win replay
    is unavailable, failed to promote, or is outside its cadence.
  * **a spend breaker that halts the loop** (§2.5 row 4). The breaker REFUSES
    metered drafting; it can never return STOP, and the test proves that from the
    source, not from one example.
  * **a permanent silent block** (§8.10). A coverage gap without an owner and a
    deadline cannot be constructed; past its deadline, across two freeze cycles,
    or with no covered surface left, it STOPS with a four-part package.
  * **busy-waiting** (§8.10). No directive can name a wait, and the module's own
    AST is audited for `sleep`/`wait`/`poll` and for any clock at all.
  * **narration deciding a transition** (§8.10 last line, AK-D4, AK-D38). A stop
    REQUEST is honoured only when a guard independently reached the same state;
    origin is recorded and changes nothing.

Every STOP a guard can emit is additionally re-validated against
`state_machine.check_stop_evidence` AND driven through a real
`ControllerStateMachine.stop()` over a real journal, so "the guard's output is
admissible" is a tested fact rather than a shared assumption between two files.

NO inference, NO benchmark, NO build, NO model call, NO process. The only files
this suite writes live under a per-test temporary directory.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_guards.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_guards.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE so `guards.schemas` is the same module object the
# journal and the state machine validate with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel import storage as ST  # noqa: E402
from autokernel.controller import guards as G  # noqa: E402
from autokernel.evaluator import api as evaluator_api  # noqa: E402
from autokernel.controller import state_machine as SM  # noqa: E402

V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
NOW = "2026-08-03T12:00:00Z"
LATER = "2026-08-10T12:00:00Z"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


# =============================================================================
# Fixtures — every one is the SMALLEST thing its guard accepts
# =============================================================================

def _package(**overrides) -> G.DecisionPackage:
    kwargs = dict(
        context="a real decision with real options",
        options=(
            G.DecisionOption(
                option_id="act",
                summary="do the thing",
                tradeoffs=("costs a boundary",),
                consequence_if_chosen="the thing is done",
                reversible=True,
            ),
            G.DecisionOption(
                option_id="hold",
                summary="hold and change nothing",
                tradeoffs=("nothing is measured",),
                consequence_if_chosen="state is preserved",
                reversible=True,
            ),
        ),
        recommendation="act",
        default="hold",
        default_rationale="the safe branch measures nothing",
        owner="operator",
        deadline=LATER,
    )
    kwargs.update(overrides)
    return G.DecisionPackage(**kwargs)


def _closure(**overrides) -> G.ClosureLedger:
    kwargs = dict(
        closed=(
            G.ClosedSubScope(
                sub_scope="iqk Q4_K prefill",
                gates_met=("T0", "T1", "mechanism"),
                receipt="ev-closed-1",
            ),
        ),
        deferred=(
            G.DeferredSubScope(
                sub_scope="IQ1 quants",
                gates_unrun=("T0", "T1"),
                reason="IQ1 remains stubbed in the kernel tree",
            ),
        ),
        hierarchy_layers_considered=("L1 parameter", "L2 dispatch", "L3 layout"),
    )
    kwargs.update(overrides)
    return G.ClosureLedger(**kwargs)


CLOSURE_REASON = (
    "closed for sub-scope iqk Q4_K prefill (gates T0, T1, mechanism met); sub-scope "
    "IQ1 quants deferred (gates T0, T1 un-run)"
)


def _health(**overrides) -> G.PlannerHealth:
    kwargs = dict(
        rounds_observed=12,
        consecutive_noop_rounds=1,
        proposal_skipped_count=3,
        repeated_fingerprint_count=1,
        invalid_dispatch_count=0,
        contradicted_narrative_count=0,
        unavailable_dependency_rounds=0,
        consecutive_build_failures=0,
        repair_cap_exceedances=0,
        banked_count=4,
        receipts={},
    )
    kwargs.update(overrides)
    return G.PlannerHealth(**kwargs)


def _health_policy(**overrides) -> G.PlannerHealthPolicy:
    kwargs = dict(
        window_rounds=10,
        max_consecutive_noop_rounds=5,
        max_repeated_fingerprints=4,
        max_invalid_dispatches=2,
        max_contradicted_narratives=1,
        max_unavailable_dependency_rounds=3,
        max_consecutive_build_failures=2,
        max_repair_cap_exceedances=2,
        policy_receipt="ev-planner-policy",
    )
    kwargs.update(overrides)
    return G.PlannerHealthPolicy(**kwargs)


def _accept(**overrides) -> G.AcceptSideControlReceipt:
    kwargs = dict(
        status=G.ACCEPT_CONTROL_PROMOTED,
        event_id="ev-control-5",
        observed_at=NOW,
        cadence=S.Check(S.PASS),
        win_id="iqk-port",
        # A receipt must NAME its control. The field's default is the UNAVAILABLE
        # status sentinel, which every fixture here used to inherit silently.
        control_id="control-5-historical-win-replay",
    )
    kwargs.update(overrides)
    return G.AcceptSideControlReceipt(**kwargs)


def _series(values) -> tuple:
    return tuple(
        G.ReadinessObservation(
            round_index=index,
            readiness=value,
            at=NOW,
            source_event_id=f"ev-readiness-{index}",
        )
        for index, value in enumerate(values)
    )


def _parity_round(index: int, **overrides) -> G.ParityObservation:
    """A round that measured everything and resolved nothing.

    `reference_gain` is left at `None` by default on purpose: the fixture that
    says nothing about what the campaign is looking for must not be the one that
    unlocks the branch able to STOP on an all-parity window. Tests that want that
    branch declare the target, and the declaration is visible in the test.
    """
    kwargs = dict(
        round_index=index, protected_cells=12, cells_at_parity=12, mde=0.018,
        noise_floor=0.01, sensitivity_bound=0.018, at=NOW,
        source_event_id=f"ev-parity-{index}",
    )
    kwargs.update(overrides)
    return G.ParityObservation(**kwargs)


def _mixed_series(entries, **parity_overrides) -> tuple:
    """A series whose rounds are ORDERABLE floats or the marker `PARITY`."""
    built: list = []
    for index, entry in enumerate(entries):
        if entry is PARITY:
            built.append(_parity_round(index, **parity_overrides))
        else:
            built.append(G.ReadinessObservation(
                round_index=index, readiness=entry, at=NOW,
                source_event_id=f"ev-readiness-{index}"))
    return tuple(built)


#: Marker for "this round produced no orderable readiness" in `_mixed_series`.
PARITY = object()

#: What the campaign is looking for. Coarser than `_parity_round`'s +/-0.018
#: bound, so a round at parity really could have seen it.
CAMPAIGN_TARGET = 0.25


def _plateau_policy(**overrides) -> G.PlateauPolicy:
    kwargs = dict(window_rounds=5, improvement_floor=0.01, floor_receipt="ev-calibration")
    kwargs.update(overrides)
    return G.PlateauPolicy(**kwargs)


def _budget(**consumed) -> G.BudgetLedger:
    return G.BudgetLedger(tuple(
        G.BudgetDimension(
            name=name, limit=100.0, consumed=float(consumed.get(name, 10.0)),
            receipt=f"ev-budget-{name}",
        )
        for name in G.BUDGET_DIMENSIONS
    ))


def _storage(*, free: int, floor: int, backlog: int) -> G.StorageObservation:
    state = ST.StorageState(
        state=ST.DISK_PRESSURE if free < floor else ST.STORAGE_OK,
        free_bytes=free, total_bytes=10 ** 12, floor_bytes=floor,
        reasons=("free space below the floor",) if free < floor else (),
    )
    return G.StorageObservation(
        path="/mnt/raid0/llm", state=state, expirable_backlog_bytes=backlog,
        receipt="ev-storage",
    )


def _anchor(*, commit: str = V8_COMMIT, binary: str = "bin-v8") -> SM.AnchorIdentity:
    return SM.AnchorIdentity(
        source_tree="llama.cpp",
        branch="production-consolidated-v8",
        commit=commit,
        binary_sha256={"llama_gpu": _sha(binary)},
        linkage_sha256={"llama_gpu": _sha("link-v8")},
    )


def _clean_planner_decision(health: G.PlannerHealth) -> G.GuardDecision:
    decision = G.guard_planner_degraded(health, _health_policy())
    assert decision.outcome == G.CONTINUE, decision.reason
    return decision


# =============================================================================
# Vocabulary and structural audits
# =============================================================================

class VocabularyTests(unittest.TestCase):

    def test_every_declared_stop_is_owned(self):
        """§8.10's enumeration is covered, with no stop decided in two places."""
        self.assertEqual(G.audit_stop_coverage_totality().outcome, S.PASS)
        self.assertEqual(
            set(G.STOP_PRECEDENCE) | set(G.NON_GUARD_STOPS), set(SM.STOP_STATES)
        )
        self.assertFalse(set(G.STOP_PRECEDENCE) & set(G.NON_GUARD_STOPS))

    def test_operator_stop_and_release_ready_are_deliberately_not_guarded(self):
        """The latch and the packager are the evidence; a second opinion is a cache."""
        self.assertIn(SM.OPERATOR_STOP_REQUESTED, G.NON_GUARD_STOPS)
        self.assertIn(SM.RELEASE_PACKAGE_READY, G.NON_GUARD_STOPS)
        self.assertIn("begin_iteration", G.NON_GUARD_STOPS[SM.OPERATOR_STOP_REQUESTED])
        self.assertIn("AK6", G.NON_GUARD_STOPS[SM.RELEASE_PACKAGE_READY])

    def test_a_drifted_vocabulary_raises_at_import_time(self):
        """The totality check is a raise, not a comment somebody may not read."""
        original = G.NON_GUARD_STOPS
        try:
            G.NON_GUARD_STOPS = {}
            with self.assertRaises(G.GuardVocabularyError):
                G._assert_vocabulary_total()
        finally:
            G.NON_GUARD_STOPS = original
        G._assert_vocabulary_total()  # restored

    def test_closure_stops_are_adjudicated_last(self):
        """A closure claim is only meaningful over evidence nothing invalidated."""
        order = list(G.STOP_PRECEDENCE)
        self.assertEqual(order[-2:], [SM.EXHAUSTED_SURFACE, SM.PLATEAU_STOP])
        self.assertEqual(order[0], SM.INTEGRITY_STOP)
        self.assertLess(order.index(SM.PLANNER_DEGRADED), order.index(SM.PLATEAU_STOP))

    def test_no_directive_can_express_a_busy_wait(self):
        """§8.10: persist and drain, NEVER busy-wait — enforced by absence."""
        self.assertEqual(G.audit_directive_vocabulary().outcome, S.PASS)
        for directive in G.DIRECTIVES:
            for token in G.FORBIDDEN_DIRECTIVE_TOKENS:
                self.assertNotIn(token, directive)

    def test_audit_catches_a_wait_directive_if_one_is_added(self):
        """The guard must fail on the compliant-looking addition, not only in prose."""
        original = G.DIRECTIVES
        try:
            G.DIRECTIVES = frozenset(set(original) | {"WAIT_FOR_DEVICE"})
            result = G.audit_directive_vocabulary()
            self.assertEqual(result.outcome, S.FAIL)
            self.assertTrue(any("WAIT_FOR_DEVICE" in r for r in result.reasons))
        finally:
            G.DIRECTIVES = original
        self.assertEqual(G.audit_directive_vocabulary().outcome, S.PASS)

    def test_module_cannot_write_signal_wait_or_read_a_clock(self):
        self.assertEqual(G.audit_no_write_process_or_wait_paths().outcome, S.PASS)

    def test_ast_audit_actually_detects_each_forbidden_shape(self):
        """A guard that passes everything is a guard that checks nothing."""
        for snippet in (
            "import time\n",
            "import os\n",
            "import random\n",
            "def f(x):\n    return x.sleep()\n",
            "def f(x):\n    return x.now()\n",
            "def f(p):\n    return open(p)\n",
            "def f(p, d):\n    return p.write(d)\n",
        ):
            with self.subTest(snippet=snippet):
                self.assertEqual(
                    G.audit_no_write_process_or_wait_paths(snippet).outcome, S.FAIL
                )

    def test_ast_audit_reports_could_not_check_on_unparseable_source(self):
        self.assertEqual(
            G.audit_no_write_process_or_wait_paths("def (:").outcome, S.COULD_NOT_CHECK
        )


# =============================================================================
# The decision package (§18 item 7)
# =============================================================================

class DecisionPackageTests(unittest.TestCase):

    def test_four_parts_are_all_required(self):
        package = _package()
        detail = package.to_detail()
        for key in ("context", "options", "recommendation", "default"):
            self.assertIn(key, detail)
        rendered = package.render()
        for heading in ("CONTEXT", "OPTIONS", "RECOMMENDATION", "DEFAULT"):
            self.assertIn(heading, rendered)

    def test_one_option_is_not_a_decision(self):
        with self.assertRaises(G.GuardInputError):
            _package(options=(
                G.DecisionOption("only", "s", ("t",), "c", True),
            ))

    def test_more_than_four_options_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            _package(options=tuple(
                G.DecisionOption(f"o{i}", "s", ("t",), "c", True) for i in range(5)
            ))

    def test_recommendation_and_default_must_name_declared_options(self):
        for field in ("recommendation", "default"):
            with self.subTest(field=field):
                with self.assertRaises(G.GuardInputError):
                    _package(**{field: "not-an-option"})

    def test_an_irreversible_default_is_refused(self):
        """The default is the branch silence selects; silence may not be irreversible."""
        with self.assertRaises(G.GuardInputError) as ctx:
            _package(options=(
                G.DecisionOption("act", "s", ("t",), "c", True),
                G.DecisionOption("hold", "s", ("t",), "c", False),
            ))
        self.assertIn("silence", str(ctx.exception))

    def test_options_need_tradeoffs(self):
        with self.assertRaises(G.GuardInputError):
            G.DecisionOption("a", "s", (), "c", True)

    def test_deadline_must_be_timezone_aware(self):
        with self.assertRaises(G.GuardInputError):
            _package(deadline="2026-08-10T12:00:00")

    def test_render_is_deterministic(self):
        self.assertEqual(_package().render(), _package().render())


# =============================================================================
# GuardDecision — the type that refuses an inadmissible stop at construction
# =============================================================================

class GuardDecisionTests(unittest.TestCase):

    def test_a_stop_the_state_machine_would_refuse_cannot_be_constructed(self):
        with self.assertRaises(G.GuardEvidenceError):
            G.GuardDecision(
                guard_id=G.GUARD_BUDGET, outcome=G.STOP, stop_state=SM.BUDGET_STOP,
                reason="out of budget", detail={"budget": "max_wall_hours"},
                evidence=("ev-1",),
            )

    def test_a_stop_with_no_receipt_is_narration(self):
        with self.assertRaises(G.GuardEvidenceError) as ctx:
            G.GuardDecision(
                guard_id=G.GUARD_BUDGET, outcome=G.STOP, stop_state=SM.BUDGET_STOP,
                reason="out of budget",
                detail={"budget": "max_wall_hours", "limit": 1.0, "consumed": 2.0},
                evidence=(),
            )
        self.assertIn("narration", str(ctx.exception))

    def test_a_guard_may_not_emit_another_guards_stop(self):
        with self.assertRaises(G.GuardInputError) as ctx:
            G.GuardDecision(
                guard_id=G.GUARD_PLATEAU, outcome=G.STOP, stop_state=SM.BUDGET_STOP,
                reason="r", detail={"budget": "x", "limit": 1, "consumed": 2},
                evidence=("ev",),
            )
        self.assertIn("one condition gets one spelling", str(ctx.exception))

    def test_an_escalating_stop_without_a_package_is_refused(self):
        with self.assertRaises(G.GuardEvidenceError) as ctx:
            G.GuardDecision(
                guard_id=G.GUARD_HOST_UPTIME, outcome=G.STOP,
                stop_state=SM.HOST_REBOOT_REQUIRED,
                reason="uptime ceiling", evidence=("ev",),
                detail={"uptime_seconds": 1, "ceiling_seconds": 2},
            )
        self.assertIn("§18 item 7", str(ctx.exception))

    def test_escalating_directive_requires_a_package_even_without_a_stop(self):
        with self.assertRaises(G.GuardEvidenceError):
            G.GuardDecision(
                guard_id=G.GUARD_COVERAGE, outcome=G.REFUSE, reason="r",
                directives=(G.DIRECTIVE_ESCALATE_TO_OPERATOR,),
            )

    def test_refuse_must_say_what_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            G.GuardDecision(guard_id=G.GUARD_BUDGET, outcome=G.REFUSE, reason="no")

    def test_stop_state_on_a_non_stop_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            G.GuardDecision(
                guard_id=G.GUARD_BUDGET, outcome=G.CONTINUE, reason="fine",
                stop_state=SM.BUDGET_STOP,
            )

    def test_undeclared_directive_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            G.GuardDecision(
                guard_id=G.GUARD_BUDGET, outcome=G.REFUSE, reason="r",
                directives=("SPIN_UNTIL_FREE",),
            )

    def test_detail_must_be_journalable(self):
        """A tuple round-trips to a list, so a record built with one is not itself."""
        with self.assertRaises(TypeError):
            G.GuardDecision(
                guard_id=G.GUARD_BUDGET, outcome=G.CONTINUE, reason="r",
                detail={"x": ("a", "b")},
            )

    def test_could_not_evaluate_never_clears(self):
        decision = G.GuardDecision(
            guard_id=G.GUARD_BUDGET, outcome=G.COULD_NOT_EVALUATE, reason="unknown"
        )
        self.assertFalse(decision.clears)
        self.assertEqual(decision.to_check().outcome, S.COULD_NOT_CHECK)

    def test_only_continue_clears(self):
        for outcome in G.OUTCOMES:
            decision = G.GuardDecision(
                guard_id=G.GUARD_BUDGET,
                outcome=outcome,
                reason="r",
                stop_state=SM.BUDGET_STOP if outcome == G.STOP else None,
                detail=(
                    {"budget": "max_wall_hours", "limit": 1.0, "consumed": 2.0}
                    if outcome == G.STOP else {}
                ),
                directives=(G.DIRECTIVE_ROOT_CAUSE_ANALYSIS,) if outcome == G.REFUSE else (),
                evidence=("ev",) if outcome == G.STOP else (),
            )
            self.assertEqual(decision.clears, outcome == G.CONTINUE)


# =============================================================================
# Closure language (§8.10 reserved words)
# =============================================================================

class ClosureLanguageTests(unittest.TestCase):

    def test_the_bare_reserved_words_are_rejected_in_the_reason(self):
        for reason in (
            "the surface is exhausted",
            "we tried all paths",
            "Exhausted.",
            "closure: ALL PATHS covered",
        ):
            with self.subTest(reason=reason):
                result = G.check_closure_language(reason, _closure())
                self.assertEqual(result.outcome, S.FAIL)

    def test_the_state_machine_and_this_checker_now_agree_on_the_bare_word(self):
        """REWRITTEN by the AK4 integration pass.

        This used to assert that `SM.check_closure_enumeration` returned PASS on
        *"the surface is exhausted"* while this checker returned FAIL — the gap
        documented rather than closed. But `state_machine.stop()` and
        `dispose_stop_request()` are both public, so a stop that never went
        through a guard reached the record on the reserved word §8.10 names
        first. The disposer owns the vocabulary now and this checker unions it,
        so the two agree in BOTH directions; the assertions below fail if either
        side drifts.
        """
        for reason in ("the surface is exhausted", "we tried all paths",
                       "nothing left to try", "no more options remain"):
            with self.subTest(reason=reason):
                machine = SM.check_closure_enumeration(reason, _closure().to_detail())
                guard = G.check_closure_language(reason, _closure())
                self.assertEqual(machine.outcome, S.FAIL)
                self.assertEqual(guard.outcome, S.FAIL)

    def test_a_clean_enumeration_still_passes_both(self):
        """The compliant path, so neither scan is passing by refusing everything."""
        reason = "closed for the mmvq dispatch sub-scope; the fusion sub-scope is deferred"
        self.assertEqual(
            SM.check_closure_enumeration(reason, _closure().to_detail()).outcome, S.PASS)
        self.assertEqual(G.check_closure_language(reason, _closure()).outcome, S.PASS)

    def test_reserved_words_hiding_in_a_sub_scope_are_rejected(self):
        ledger = _closure(deferred=(
            G.DeferredSubScope("all paths through iqk", ("T0",), "not run"),
        ))
        result = G.check_closure_language(CLOSURE_REASON, ledger)
        self.assertEqual(result.outcome, S.FAIL)
        self.assertTrue(any("deferred[0].sub_scope" in r for r in result.reasons))

    def test_reserved_words_hiding_in_a_gate_name_are_rejected(self):
        ledger = _closure(closed=(
            G.ClosedSubScope("iqk", ("T0", "exhausted sweep"), "ev-1"),
        ))
        self.assertEqual(
            G.check_closure_language(CLOSURE_REASON, ledger).outcome, S.FAIL
        )

    def test_words_that_merely_contain_a_reserved_word_are_not_rejected(self):
        """A checker that fires on 'exhaustively' would be routed around, not fixed."""
        result = G.check_closure_language(
            "closed for sub-scope X (gates A, B met); sub-scope Y deferred (gate C "
            "un-run). The dispatch table was enumerated exhaustively.",
            _closure(),
        )
        self.assertEqual(result.outcome, S.PASS)

    def test_a_proper_enumeration_passes(self):
        self.assertEqual(
            G.check_closure_language(CLOSURE_REASON, _closure()).outcome, S.PASS
        )

    def test_unreadable_input_is_could_not_check_not_pass(self):
        self.assertEqual(
            G.check_closure_language("", _closure()).outcome, S.COULD_NOT_CHECK
        )
        self.assertEqual(
            G.check_closure_language(CLOSURE_REASON, None).outcome, S.COULD_NOT_CHECK
        )

    def test_a_ledger_may_not_defer_and_close_the_same_sub_scope(self):
        with self.assertRaises(G.GuardInputError):
            _closure(deferred=(
                G.DeferredSubScope("iqk Q4_K prefill", ("T1",), "unsure"),
            ))

    def test_empty_deferred_is_an_answer_but_absent_closed_is_not(self):
        _closure(deferred=())  # legitimate
        with self.assertRaises(G.GuardInputError):
            _closure(closed=())


# =============================================================================
# EXHAUSTED_SURFACE and PLATEAU_STOP
# =============================================================================

class ClosureGuardTests(unittest.TestCase):

    def _exhausted(self, **overrides):
        health = overrides.pop("health", _health())
        if "planner_decision" not in overrides:
            overrides["planner_decision"] = _clean_planner_decision(health)
        kwargs = dict(
            reason=CLOSURE_REASON,
            closure=_closure(),
            accept_control=_accept(),
            health=health,
            eligible_layers_remaining=0,
        )
        kwargs.update(overrides)
        return G.guard_exhausted_surface(**kwargs)

    def _plateau(self, **overrides):
        health = overrides.pop("health", _health())
        if "planner_decision" not in overrides:
            overrides["planner_decision"] = _clean_planner_decision(health)
        kwargs = dict(
            reason=CLOSURE_REASON,
            series=_series([0.30, 0.301, 0.3005, 0.3009, 0.3008, 0.3007]),
            policy=_plateau_policy(),
            closure=_closure(),
            accept_control=_accept(),
            health=health,
        )
        kwargs.update(overrides)
        return G.guard_plateau(**kwargs)

    def test_exhausted_surface_stops_with_the_enumeration(self):
        decision = self._exhausted()
        self.assertEqual(decision.outcome, G.STOP)
        self.assertEqual(decision.stop_state, SM.EXHAUSTED_SURFACE)
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )
        self.assertIn("closed", decision.detail)
        self.assertIn("deferred", decision.detail)

    def test_remaining_layers_means_the_surface_is_simply_not_closed(self):
        decision = self._exhausted(eligible_layers_remaining=2)
        self.assertEqual(decision.outcome, G.CONTINUE)

    def test_plateau_stops_and_records_degraded_ruled_out(self):
        decision = self._plateau()
        self.assertEqual(decision.outcome, G.STOP)
        self.assertEqual(decision.stop_state, SM.PLATEAU_STOP)
        self.assertIs(decision.detail["planner_health"]["degraded_ruled_out"], True)
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )

    def test_plateau_requires_the_same_enumeration_exhausted_surface_does(self):
        self.assertIn("closed", self._plateau().detail)
        self.assertIn("deferred", self._plateau().detail)

    def test_real_improvement_continues(self):
        decision = self._plateau(series=_series([0.30, 0.31, 0.32, 0.33, 0.34, 0.35]))
        self.assertEqual(decision.outcome, G.CONTINUE)

    def test_a_declining_series_plateaus_rather_than_improving(self):
        decision = self._plateau(series=_series([0.40, 0.39, 0.38, 0.37, 0.36, 0.35]))
        self.assertEqual(decision.outcome, G.STOP)
        self.assertEqual(decision.detail["improvement"], 0.0)

    def test_a_partial_window_cannot_conclude(self):
        decision = self._plateau(series=_series([0.30, 0.301]))
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)

    def test_selection_stratum_readiness_is_refused_outright(self):
        """P-AK-SEARCH-1: readiness is computed ONLY from confirmation evidence."""
        with self.assertRaises(G.GuardInputError) as ctx:
            G.ReadinessObservation(0, 0.3, NOW, "ev-1", stratum="selection")
        self.assertIn("confirmation", str(ctx.exception))

    def test_readiness_without_a_source_event_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            G.ReadinessObservation(0, 0.3, NOW, "")

    def test_an_unordered_series_is_refused(self):
        series = _series([0.30, 0.31, 0.32, 0.33, 0.34, 0.35])
        shuffled = (series[3],) + series[1:]
        with self.assertRaises(G.GuardInputError):
            self._plateau(series=shuffled)

    def test_a_broken_searcher_blocks_both_closure_stops(self):
        """§8.10's conflation, made unrepresentable rather than merely documented."""
        health = _health(
            consecutive_noop_rounds=9,
            receipts={"repeated_no_ops": "ev-noop"},
        )
        degraded = G.guard_planner_degraded(health, _health_policy())
        self.assertEqual(degraded.outcome, G.STOP)
        for decision in (
            self._exhausted(health=health, planner_decision=degraded),
            self._plateau(health=health, planner_decision=degraded),
        ):
            self.assertEqual(decision.outcome, G.REFUSE)
            self.assertIn("searcher", decision.reason)

    # -- rounds that produced no orderable readiness -------------------------
    #
    # The release plane withholds a readiness magnitude when every protected cell
    # of a phase resolved below the campaign's own floor or MDE. Those rounds
    # HAPPENED and must stay in the series — a plateau computed over a
    # subsequence the guard chose for itself is a trend in a sample nobody
    # defined — but there is no number on them, and the arithmetic below is only
    # defined on rounds that have one.

    def test_a_parity_round_has_no_readiness_to_read(self):
        observation = _parity_round(0)
        with self.assertRaises(G.ParityHasNoMagnitude):
            observation.readiness
        # `getattr(..., default)` is the usual way an absent attribute becomes a
        # silent zero; a raising property closes that door too.
        with self.assertRaises(G.ParityHasNoMagnitude):
            getattr(observation, "readiness", 0.0)
        # And the SERIALIZED round carries no `readiness` key — not a null one.
        # `to_dict()` lands in `GuardDecision.detail["window"]` and from there in
        # a journal, where the type is gone; `entry["readiness"] or 0.0` on a null
        # is the same substituted zero the property refuses.
        wire = observation.to_dict()
        self.assertNotIn("readiness", wire)
        self.assertFalse(wire["orderable"])
        self.assertIn("sub-floor does not mean zero", wire["no_magnitude_reason"])

    def test_a_parity_round_cannot_raise_the_window_best(self):
        """If parity were read as a magnitude the window would trend through it."""
        decision = self._plateau(
            series=_mixed_series([0.30, 0.30, PARITY, PARITY, 0.30, 0.30]))
        self.assertEqual(decision.outcome, G.STOP)
        self.assertEqual(decision.detail["improvement"], 0.0)
        self.assertEqual(decision.detail["parity_rounds"], 2)
        self.assertEqual(decision.detail["orderable_rounds"], 3)

    def test_a_parity_round_is_counted_as_a_round_that_happened(self):
        """Dropping it would make a full campaign look like a partial window."""
        decision = self._plateau(
            series=_mixed_series([PARITY, 0.30, PARITY, PARITY, PARITY, 0.30]))
        self.assertEqual(decision.outcome, G.STOP)
        # The declared 5-round window is full: three of its rounds are parity and
        # they are still rounds, so this is a plateau rather than a partial window.
        self.assertEqual(decision.detail["parity_rounds"], 3)
        self.assertEqual(decision.detail["improvement"], 0.0)

    def test_a_window_that_opens_at_parity_cannot_be_evaluated(self):
        """`best - opening` has no opening; zero would manufacture an improvement."""
        decision = self._plateau(
            series=_mixed_series([0.10, PARITY, 0.10, 0.10, 0.90, 0.90]))
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertIn("no orderable readiness", decision.reason)
        self.assertNotIn("improvement", decision.detail)

    def test_an_all_parity_window_does_not_stop_on_an_invented_trend(self):
        """Whatever it answers, it never answers by substituting zeros."""
        decision = self._plateau(
            series=_mixed_series([0.30, PARITY, PARITY, PARITY, PARITY, PARITY]))
        self.assertEqual(decision.detail["orderable_rounds"], 0)
        for invented in ("improvement", "opening_readiness", "best_readiness"):
            self.assertNotIn(invented, decision.detail)

    def test_real_improvement_still_continues_with_parity_rounds_present(self):
        """The control: parity rounds must not turn every window into a stop."""
        decision = self._plateau(
            series=_mixed_series([0.30, 0.31, PARITY, 0.33, 0.34, 0.35]))
        self.assertEqual(decision.outcome, G.CONTINUE)
        self.assertEqual(decision.detail["parity_rounds"], 1)

    # -- a converged campaign must terminate, and not on invented numbers ----
    #
    # Under a NON-INFERIORITY objective parity is the most common HEALTHY
    # outcome, and a converged campaign goes all-parity and STAYS there. So the
    # two misreadings of an all-parity window point opposite ways and both are
    # live: reading parity as `0.0` stops on a trend nobody measured, and
    # refusing to read the window at all never stops — the second is the one the
    # first version of this guard chose, and on a converged campaign it is
    # permanent.

    def _all_parity(self, rounds: int = 5, **parity_overrides):
        return self._plateau(
            series=_mixed_series([PARITY] * rounds, **parity_overrides))

    def test_a_converged_campaign_stops_rather_than_spending_forever(self):
        decision = self._all_parity(reference_gain=CAMPAIGN_TARGET)
        self.assertEqual(decision.outcome, G.STOP)
        self.assertEqual(decision.stop_state, SM.PLATEAU_STOP)
        self.assertEqual(decision.detail["plateau_basis"],
                         G.PLATEAU_BASIS_NO_DETECTABLE_EFFECT)
        self.assertEqual(decision.detail["reference_gain"], CAMPAIGN_TARGET)
        self.assertEqual(decision.detail["coarsest_sensitivity_bound"], 0.018)
        self.assertEqual(decision.detail["parity_rounds"], 5)

    def test_the_converged_stop_is_not_a_plateau_of_zeros(self):
        """The stop must not report an improvement it did not measure.

        A `0.0` here is not a harmless placeholder: it is indistinguishable in
        the journal from a window that opened at a magnitude, reached the same
        magnitude and genuinely did not move — and the two are different results.
        """
        decision = self._all_parity(reference_gain=CAMPAIGN_TARGET)
        for invented in ("improvement", "opening_readiness", "best_readiness"):
            self.assertNotIn(invented, decision.detail)
        self.assertIn("absence of a detectable effect",
                      decision.detail["no_improvement_magnitude_reason"])
        # And the two bases are not the same word: an auditor can tell which
        # question the guard answered.
        self.assertNotEqual(G.PLATEAU_BASIS_NO_DETECTABLE_EFFECT,
                            G.PLATEAU_BASIS_MEASURED_IMPROVEMENT)

    def test_a_measured_plateau_still_names_itself_a_measured_one(self):
        """The control on the basis: the subtraction branch keeps its numbers."""
        decision = self._plateau(series=_mixed_series([0.30, 0.30, PARITY, 0.30,
                                                       0.30, 0.30]))
        self.assertEqual(decision.outcome, G.STOP)
        self.assertEqual(decision.detail["plateau_basis"],
                         G.PLATEAU_BASIS_MEASURED_IMPROVEMENT)
        self.assertEqual(decision.detail["improvement"], 0.0)
        self.assertEqual(decision.detail["opening_readiness"], 0.30)

    def test_a_window_too_blind_to_see_the_target_has_observed_nothing(self):
        """Parity at +/-0.4 rules out nothing a campaign hunting 0.25 cares about."""
        decision = self._all_parity(reference_gain=CAMPAIGN_TARGET,
                                    mde=0.4, sensitivity_bound=0.4)
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertIn("too blind to see the effect it is hunting", decision.reason)
        self.assertEqual(decision.detail["blind_round_indices"], [0, 1, 2, 3, 4])
        self.assertNotIn("improvement", decision.detail)

    def test_a_window_with_no_declared_target_has_nothing_to_rule_out(self):
        """`None` is the fixture default: an undeclared target cannot unlock a stop."""
        decision = self._all_parity()
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertIn("does not carry one campaign target", decision.reason)
        self.assertEqual(decision.detail["reference_gains_declared"], [])

    def test_a_window_that_disagrees_about_the_target_is_not_resolved_by_picking_one(self):
        series = (_parity_round(0, reference_gain=CAMPAIGN_TARGET),
                  _parity_round(1, reference_gain=CAMPAIGN_TARGET),
                  _parity_round(2, reference_gain=0.10),
                  _parity_round(3, reference_gain=CAMPAIGN_TARGET),
                  _parity_round(4, reference_gain=CAMPAIGN_TARGET))
        decision = self._plateau(series=series)
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertEqual(decision.detail["reference_gains_declared"],
                         [0.10, CAMPAIGN_TARGET])

    def test_a_blind_parity_round_cannot_be_read_as_evidence_of_non_improvement(self):
        """A round nobody could see into is not a round that showed nothing.

        The window opens at 0.30 and the floor is 0.01, so a round would have to
        reach 0.31 to contradict the stop. A parity round resolving no finer than
        +/-0.5 could have been there and gone unseen.
        """
        decision = self._plateau(series=_mixed_series(
            [0.30, 0.30, PARITY, 0.30, 0.30, 0.30], mde=0.5, sensitivity_bound=0.5))
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertEqual(decision.detail["blind_round_indices"], [2])
        self.assertAlmostEqual(
            decision.detail["magnitude_that_would_contradict"], 0.31)

    def test_a_sighted_parity_round_still_lets_the_window_stop(self):
        """The control on that check: it must not veto every stop with a parity round."""
        decision = self._plateau(series=_mixed_series([0.30, 0.30, PARITY, 0.30,
                                                       0.30, 0.30]))
        self.assertEqual(decision.outcome, G.STOP)
        self.assertNotIn("blind_round_indices", decision.detail)

    def test_a_blind_round_cannot_argue_with_a_measured_improvement(self):
        """Blindness only ever threatens the STOP side.

        A window that MEASURED an improvement above the floor continues on that
        measurement; a round with no magnitude has nothing to set against it, and
        letting it veto the CONTINUE would stall a campaign that is working.
        """
        decision = self._plateau(series=_mixed_series(
            [0.30, 0.31, PARITY, 0.33, 0.34, 0.35], mde=0.5, sensitivity_bound=0.5))
        self.assertEqual(decision.outcome, G.CONTINUE)

    def test_a_parity_round_refuses_a_bound_sharper_than_its_own_numbers(self):
        """An understated bound reads an underpowered round as a clean result."""
        with self.assertRaises(G.GuardInputError) as ctx:
            _parity_round(0, mde=0.018, noise_floor=0.30, sensitivity_bound=0.018)
        self.assertIn("sharper than", str(ctx.exception))
        # A COARSER bound is admitted: it can only make the guard more reluctant.
        self.assertEqual(_parity_round(0, sensitivity_bound=0.5).sensitivity_bound, 0.5)

    def test_the_series_still_refuses_something_that_is_not_an_observation(self):
        with self.assertRaises(G.GuardInputError):
            self._plateau(series=(0.30, 0.31, 0.32, 0.33, 0.34, 0.35))

    def test_a_parity_round_still_carries_its_stratum_and_its_event(self):
        with self.assertRaises(G.GuardInputError):
            _parity_round(0, stratum="selection")
        with self.assertRaises(G.GuardInputError):
            _parity_round(0, source_event_id="")
        with self.assertRaises(G.GuardInputError):
            _parity_round(0, cells_at_parity=13)

    def test_a_stale_clean_verdict_cannot_be_paired_with_a_fresh_plateau(self):
        """The clean verdict must have been computed over THIS health snapshot."""
        old_health = _health(rounds_observed=11)
        stale = _clean_planner_decision(old_health)
        new_health = _health(rounds_observed=12, banked_count=5)
        decision = self._plateau(health=new_health, planner_decision=stale)
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertIn("stale clearance", decision.reason)

    def test_a_planner_decision_from_another_guard_is_refused(self):
        wrong = G.guard_budget(_budget())
        with self.assertRaises(G.GuardInputError):
            self._plateau(planner_decision=wrong)

    def test_an_unavailable_accept_control_makes_closure_undecidable(self):
        """§12: a dead gate is indistinguishable from an exhausted surface."""
        unavailable = _accept(
            status=G.ACCEPT_CONTROL_UNAVAILABLE, win_id=None
        )
        for decision in (
            self._exhausted(accept_control=unavailable),
            self._plateau(accept_control=unavailable),
        ):
            self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
            self.assertIn("promote", decision.reason)

    def test_a_gate_that_failed_to_promote_makes_closure_undecidable(self):
        failed = _accept(status=G.ACCEPT_CONTROL_FAILED_TO_PROMOTE, win_id=None)
        self.assertEqual(self._exhausted(accept_control=failed).outcome,
                         G.COULD_NOT_EVALUATE)

    def test_an_out_of_cadence_accept_control_makes_closure_undecidable(self):
        stale = _accept(cadence=S.Check(S.FAIL, ("A/A is 40 rounds overdue",)))
        self.assertEqual(self._plateau(accept_control=stale).outcome,
                         G.COULD_NOT_EVALUATE)

    def test_reserved_language_in_the_reason_refuses_the_stop(self):
        decision = self._exhausted(reason="the surface is exhausted")
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertIn("reserved language", decision.reason)

    def test_a_plateau_floor_with_no_calibration_receipt_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            _plateau_policy(floor_receipt="")

    def test_a_zero_plateau_floor_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            _plateau_policy(improvement_floor=0.0)

    def test_a_one_round_plateau_window_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            _plateau_policy(window_rounds=1)


# =============================================================================
# PLANNER_DEGRADED
# =============================================================================

class PlannerDegradedTests(unittest.TestCase):

    def test_each_documented_family_can_trip_the_stop(self):
        families = {
            "repeated_no_ops": {"consecutive_noop_rounds": 9},
            "repeated_fingerprints": {"repeated_fingerprint_count": 9},
            "invalid_dispatches": {"invalid_dispatch_count": 9},
            "contradicted_narrative": {"contradicted_narrative_count": 9},
            "unavailable_dependency_loop": {"unavailable_dependency_rounds": 9},
            "consecutive_build_failures": {"consecutive_build_failures": 9},
            "repair_cap_exceeded": {"repair_cap_exceedances": 9},
        }
        for signal, overrides in families.items():
            with self.subTest(signal=signal):
                health = _health(receipts={signal: f"ev-{signal}"}, **overrides)
                decision = G.guard_planner_degraded(health, _health_policy())
                self.assertEqual(decision.outcome, G.STOP)
                self.assertEqual(decision.stop_state, SM.PLANNER_DEGRADED)
                self.assertEqual(decision.detail["signal"], signal)
                self.assertEqual(
                    SM.check_stop_evidence(
                        decision.stop_state, decision.reason, decision.detail
                    ).outcome,
                    S.PASS,
                )

    def test_a_crossed_signal_with_no_receipt_cannot_stop(self):
        health = _health(consecutive_noop_rounds=9, receipts={})
        decision = G.guard_planner_degraded(health, _health_policy())
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertIn("receipt", decision.reason)

    def test_a_partial_window_is_not_health(self):
        health = _health(rounds_observed=3, consecutive_noop_rounds=1)
        decision = G.guard_planner_degraded(health, _health_policy())
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertFalse(decision.clears)

    def test_a_healthy_planner_continues_and_carries_its_digest(self):
        health = _health()
        decision = G.guard_planner_degraded(health, _health_policy())
        self.assertEqual(decision.outcome, G.CONTINUE)
        self.assertEqual(decision.detail["health_digest"], health.digest)

    def test_the_digest_changes_with_the_snapshot(self):
        self.assertNotEqual(_health().digest, _health(banked_count=99).digest)

    def test_every_threshold_must_be_declared(self):
        with self.assertRaises(G.GuardInputError):
            _health_policy(window_rounds=0)
        with self.assertRaises(G.GuardInputError):
            _health_policy(policy_receipt="")

    def test_zero_build_failure_tolerance_is_legitimate(self):
        """§7.1's template declares 0 — zero tolerance is a choice, not an omission."""
        policy = _health_policy(max_consecutive_build_failures=0)
        health = _health(
            consecutive_build_failures=1,
            receipts={"consecutive_build_failures": "ev-build"},
        )
        self.assertEqual(
            G.guard_planner_degraded(health, policy).stop_state, SM.PLANNER_DEGRADED
        )


# =============================================================================
# INTEGRITY_STOP
# =============================================================================

class IntegrityTests(unittest.TestCase):

    def test_zero_tolerance_stops_on_the_first_signal(self):
        ledger = G.IntegrityLedger(
            signals=(G.IntegritySignal("cached_output_as_oracle", NOW, "ev-int-1"),),
            consecutive_failures=1,
            max_consecutive_integrity_failures=0,
        )
        decision = G.guard_integrity(ledger)
        self.assertEqual(decision.stop_state, SM.INTEGRITY_STOP)
        self.assertEqual(decision.detail["signal"], "cached_output_as_oracle")
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )

    def test_within_tolerance_continues(self):
        ledger = G.IntegrityLedger(
            signals=(G.IntegritySignal("s", NOW, "ev-1"),),
            consecutive_failures=1,
            max_consecutive_integrity_failures=2,
        )
        self.assertEqual(G.guard_integrity(ledger).outcome, G.CONTINUE)

    def test_a_failure_count_without_receipts_is_refused(self):
        with self.assertRaises(G.GuardInputError) as ctx:
            G.IntegrityLedger(
                signals=(), consecutive_failures=3,
                max_consecutive_integrity_failures=0,
            )
        self.assertIn("narration", str(ctx.exception))


# =============================================================================
# ANCHOR_MOVED
# =============================================================================

class AnchorTests(unittest.TestCase):

    def test_an_unchanged_anchor_continues(self):
        decision = G.guard_anchor_moved(
            recorded=_anchor(), observed=_anchor(), receipt="ev-anchor"
        )
        self.assertEqual(decision.outcome, G.CONTINUE)

    def test_a_moved_commit_stops_and_names_both_identities(self):
        decision = G.guard_anchor_moved(
            recorded=_anchor(), observed=_anchor(commit=V7_COMMIT), receipt="ev-anchor"
        )
        self.assertEqual(decision.stop_state, SM.ANCHOR_MOVED)
        self.assertEqual(decision.detail["recorded_anchor"]["commit"], V8_COMMIT)
        self.assertEqual(decision.detail["observed_anchor"]["commit"], V7_COMMIT)
        self.assertEqual(decision.detail["affected_backends"], ["llama_gpu"])
        self.assertIn(G.DIRECTIVE_REANCHOR, decision.directives)
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )

    def test_a_rebuilt_binary_is_a_different_anchor(self):
        decision = G.guard_anchor_moved(
            recorded=_anchor(), observed=_anchor(binary="bin-rebuilt"),
            receipt="ev-anchor",
        )
        self.assertEqual(decision.stop_state, SM.ANCHOR_MOVED)

    def test_an_unobservable_anchor_is_not_an_unmoved_one(self):
        decision = G.guard_anchor_moved(
            recorded=_anchor(), observed=None, receipt="ev-anchor"
        )
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertFalse(decision.clears)

    def test_the_comparison_is_the_state_machines_and_not_a_second_one(self):
        recorded, observed = _anchor(), _anchor(commit=V7_COMMIT)
        machine = SM.check_anchor_identity(recorded, observed)
        decision = G.guard_anchor_moved(
            recorded=recorded, observed=observed, receipt="ev"
        )
        self.assertEqual(decision.detail["mismatches"], list(machine.reasons))


# =============================================================================
# HOST_REBOOT_REQUIRED (§10.7)
# =============================================================================

class HostUptimeTests(unittest.TestCase):

    def _guard(self, host):
        return G.guard_host_uptime(
            host, owner="operator", escalation_deadline=LATER, now=NOW
        )

    def test_under_the_ceiling_continues(self):
        host = G.HostHealth(uptime_seconds=3 * 24 * 3600, observed_at=NOW, receipt="ev-h")
        self.assertEqual(self._guard(host).outcome, G.CONTINUE)

    def test_at_one_week_the_loop_requests_a_reboot_it_may_not_perform(self):
        host = G.HostHealth(
            uptime_seconds=G.HOST_UPTIME_CEILING_SECONDS, observed_at=NOW, receipt="ev-h"
        )
        decision = self._guard(host)
        self.assertEqual(decision.stop_state, SM.HOST_REBOOT_REQUIRED)
        self.assertIsNotNone(decision.decision_package)
        self.assertIn(G.DIRECTIVE_PERSIST_AND_DRAIN, decision.directives)
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )

    def test_the_reboot_default_measures_nothing(self):
        host = G.HostHealth(
            uptime_seconds=G.HOST_UPTIME_CEILING_SECONDS + 1, observed_at=NOW,
            receipt="ev-h",
        )
        package = self._guard(host).decision_package
        default = next(o for o in package.options if o.option_id == package.default)
        self.assertTrue(default.reversible)
        self.assertIn("no compute is spent", " ".join(default.tradeoffs))
        self.assertIn("measures nothing", package.default_rationale)

    def test_a_campaign_may_tighten_the_ceiling(self):
        host = G.HostHealth(
            uptime_seconds=4 * 24 * 3600, observed_at=NOW, receipt="ev-h",
            ceiling_seconds=3 * 24 * 3600,
        )
        self.assertEqual(self._guard(host).stop_state, SM.HOST_REBOOT_REQUIRED)

    def test_a_campaign_may_not_loosen_the_ceiling(self):
        with self.assertRaises(G.GuardInputError) as ctx:
            G.HostHealth(
                uptime_seconds=1, observed_at=NOW, receipt="ev-h",
                ceiling_seconds=G.HOST_UPTIME_CEILING_SECONDS + 1,
            )
        self.assertIn("discarded", str(ctx.exception))

    def test_an_unobservable_host_is_not_a_healthy_one(self):
        host = G.HostHealth(
            uptime_seconds=0, observed_at=NOW, receipt="ev-h", observable=False
        )
        decision = self._guard(host)
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)

    def test_a_deadline_already_past_is_refused(self):
        host = G.HostHealth(uptime_seconds=1, observed_at=NOW, receipt="ev-h")
        with self.assertRaises(G.GuardInputError):
            G.guard_host_uptime(
                host, owner="operator", escalation_deadline="2026-08-01T00:00:00Z",
                now=NOW,
            )


# =============================================================================
# RESOURCE_UNAVAILABLE
# =============================================================================

class ResourceTests(unittest.TestCase):

    def test_a_held_claim_continues(self):
        observation = G.ResourceClaimObservation(
            resource="gpu:0", claim_kind="device", acquired=True, observed_at=NOW,
            receipt="ev-claim",
        )
        self.assertEqual(G.guard_resource_available(observation).outcome, G.CONTINUE)

    def test_an_unavailable_claim_persists_and_drains(self):
        observation = G.ResourceClaimObservation(
            resource="gpu:0", claim_kind="device", acquired=False, observed_at=NOW,
            held_by="operator-session", unavailable_reason="held by another holder",
        )
        decision = G.guard_resource_available(observation)
        self.assertEqual(decision.stop_state, SM.RESOURCE_UNAVAILABLE)
        self.assertEqual(decision.directives, (G.DIRECTIVE_PERSIST_AND_DRAIN,))
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )

    def test_no_guard_can_emit_a_waiting_directive(self):
        observation = G.ResourceClaimObservation(
            resource="cpu_region:0-95", claim_kind="cpu_region", acquired=False,
            observed_at=NOW, unavailable_reason="region claim refused",
        )
        for directive in G.guard_resource_available(observation).directives:
            for token in G.FORBIDDEN_DIRECTIVE_TOKENS:
                self.assertNotIn(token, directive)

    def test_an_unavailable_claim_must_say_why(self):
        with self.assertRaises(G.GuardInputError):
            G.ResourceClaimObservation(
                resource="gpu:0", claim_kind="device", acquired=False, observed_at=NOW,
            )

    def test_an_acquired_claim_must_carry_its_receipt(self):
        with self.assertRaises(G.GuardInputError):
            G.ResourceClaimObservation(
                resource="gpu:0", claim_kind="device", acquired=True, observed_at=NOW,
            )


# =============================================================================
# DISK_PRESSURE
# =============================================================================

class StorageTests(unittest.TestCase):

    def test_headroom_continues(self):
        decision = G.guard_storage_headroom(
            _storage(free=1000, floor=500, backlog=0)
        )
        self.assertEqual(decision.outcome, G.CONTINUE)

    def test_a_clearing_expiry_backlog_reclaims_instead_of_stopping(self):
        """P-AK-SEARCH-1 precondition 7's branch most implementations skip."""
        decision = G.guard_storage_headroom(_storage(free=100, floor=500, backlog=900))
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertEqual(decision.directives, (G.DIRECTIVE_RECLAIM_EXPIRABLE_FIRST,))

    def test_a_backlog_that_does_not_clear_the_floor_stops(self):
        decision = G.guard_storage_headroom(_storage(free=100, floor=500, backlog=50))
        self.assertEqual(decision.stop_state, SM.DISK_PRESSURE)
        self.assertEqual(decision.detail["shortfall_bytes"], 350)
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )

    def test_the_storage_reading_must_come_from_the_storage_plane(self):
        """This plane does not read the filesystem; it consumes what did."""
        with self.assertRaises(G.GuardInputError):
            G.StorageObservation(
                path="/mnt/raid0", state={"free_bytes": 1}, expirable_backlog_bytes=0,
                receipt="ev",
            )


# =============================================================================
# BUDGET_STOP and the spend breaker
# =============================================================================

class BudgetTests(unittest.TestCase):

    def test_headroom_on_every_dimension_continues(self):
        self.assertEqual(G.guard_budget(_budget()).outcome, G.CONTINUE)

    def test_each_dimension_can_exhaust_the_campaign(self):
        for name in G.BUDGET_DIMENSIONS:
            with self.subTest(dimension=name):
                decision = G.guard_budget(_budget(**{name: 100.0}))
                self.assertEqual(decision.stop_state, SM.BUDGET_STOP)
                self.assertEqual(decision.detail["budget"], name)
                self.assertEqual(
                    SM.check_stop_evidence(
                        decision.stop_state, decision.reason, decision.detail
                    ).outcome,
                    S.PASS,
                )

    def test_the_governing_dimension_is_the_declared_order_not_the_supplied_one(self):
        ledger = _budget(max_candidates=100.0, max_wall_hours=100.0)
        self.assertEqual(G.guard_budget(ledger).detail["budget"], "max_wall_hours")
        self.assertEqual(len(G.guard_budget(ledger).detail["exhausted"]), 2)

    def test_a_partial_ledger_is_an_unbounded_budget_and_is_refused(self):
        dims = tuple(
            G.BudgetDimension(name, 100.0, 1.0, f"ev-{name}")
            for name in G.BUDGET_DIMENSIONS[:-1]
        )
        with self.assertRaises(G.GuardInputError) as ctx:
            G.BudgetLedger(dims)
        self.assertIn("unbounded", str(ctx.exception))

    def test_a_zero_budget_cannot_be_declared(self):
        with self.assertRaises(G.GuardInputError) as ctx:
            G.BudgetDimension("max_wall_hours", 0.0, 0.0, "ev")
        self.assertIn("precondition 8", str(ctx.exception))

    def test_an_infinite_budget_cannot_be_declared(self):
        with self.assertRaises(G.GuardInputError):
            G.BudgetDimension("max_wall_hours", float("inf"), 0.0, "ev")

    def test_a_nan_budget_cannot_be_declared(self):
        with self.assertRaises(G.GuardInputError):
            G.BudgetDimension("max_wall_hours", float("nan"), 0.0, "ev")

    def test_the_spend_breaker_forces_local_planning_and_never_halts(self):
        """§2.5 row 4: the naive breaker STOPPED the loop. This one cannot."""
        ledger = _budget(max_controller_tokens=90.0)
        decision = G.guard_controller_spend(ledger, G.SpendBreakerPolicy(0.8, "ev-pol"))
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertEqual(decision.directives, (G.DIRECTIVE_LOCAL_PLANNING_ONLY,))
        self.assertIsNone(decision.stop_state)

    def test_the_breaker_never_returns_stop_at_any_spend(self):
        policy = G.SpendBreakerPolicy(0.5, "ev-pol")
        for consumed in (0.0, 49.0, 50.0, 99.0, 100.0, 1000.0):
            with self.subTest(consumed=consumed):
                decision = G.guard_controller_spend(
                    _budget(max_controller_tokens=consumed), policy
                )
                self.assertNotEqual(decision.outcome, G.STOP)

    def test_the_ceiling_stop_belongs_to_the_budget_guard(self):
        ledger = _budget(max_controller_tokens=100.0)
        self.assertEqual(G.guard_budget(ledger).stop_state, SM.BUDGET_STOP)
        self.assertEqual(
            G.guard_controller_spend(ledger, G.SpendBreakerPolicy(0.8, "ev")).outcome,
            G.REFUSE,
        )

    def test_a_breaker_fraction_of_one_is_refused(self):
        for fraction in (0.0, 1.0, 1.5, -0.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(G.GuardInputError):
                    G.SpendBreakerPolicy(fraction, "ev")


# =============================================================================
# EVALUATOR_COVERAGE_GAP
# =============================================================================

class CoverageGapTests(unittest.TestCase):

    def _gap(self, **overrides):
        kwargs = dict(
            gap_id="gap-1",
            missing_coverage_class="fp8_dispatch_shapes",
            blocked_lineage="ak/llama_gpu/champion-1",
            owner="operator",
            deadline=LATER,
            opened_at="2026-08-01T00:00:00Z",
            receipt="ev-gap",
        )
        kwargs.update(overrides)
        return G.CoverageGap(**kwargs)

    def _guard(self, gaps, **overrides):
        kwargs = dict(
            now=NOW, covered_surfaces_remaining=3, escalation_owner="operator",
            escalation_deadline=LATER,
        )
        kwargs.update(overrides)
        return G.guard_evaluator_coverage(gaps, **kwargs)

    def test_no_gap_continues(self):
        self.assertEqual(self._guard([]).outcome, G.CONTINUE)

    def test_an_open_gap_blocks_release_and_lets_research_continue(self):
        decision = self._guard([self._gap()])
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertIn(G.DIRECTIVE_RELEASE_BLOCKED, decision.directives)
        self.assertIsNone(decision.stop_state)

    def test_a_gap_without_an_owner_or_deadline_cannot_exist(self):
        for field in ("owner", "deadline"):
            with self.subTest(field=field):
                with self.assertRaises(G.GuardInputError):
                    self._gap(**{field: ""})

    def test_a_gap_still_open_at_a_campaign_boundary_escalates(self):
        decision = self._guard([self._gap(boundaries_open=1)])
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertIn(G.DIRECTIVE_ESCALATE_TO_OPERATOR, decision.directives)
        self.assertIsNotNone(decision.decision_package)

    def test_an_overdue_gap_stops(self):
        decision = self._guard([self._gap(deadline="2026-08-02T00:00:00Z")])
        self.assertEqual(decision.stop_state, SM.EVALUATOR_COVERAGE_GAP)
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )

    def test_two_freeze_cycles_open_is_a_program_level_defect(self):
        decision = self._guard([self._gap(freeze_cycles_open=2)])
        self.assertEqual(decision.stop_state, SM.EVALUATOR_COVERAGE_GAP)
        self.assertIn("program-level defect", decision.reason)

    def test_no_covered_surface_left_stops(self):
        decision = self._guard([self._gap()], covered_surfaces_remaining=0)
        self.assertEqual(decision.stop_state, SM.EVALUATOR_COVERAGE_GAP)

    def test_every_stopping_branch_carries_the_four_part_package(self):
        for gap in (
            self._gap(deadline="2026-08-02T00:00:00Z"),
            self._gap(freeze_cycles_open=2),
        ):
            decision = self._guard([gap])
            self.assertIsNotNone(decision.decision_package)
            for key in ("context", "options", "recommendation", "default"):
                self.assertIn(key, decision.detail)

    def test_the_package_default_never_applies_an_evaluator_amendment(self):
        """P-AK-SEARCH-1: evaluator changes are human-only; silence may not apply one."""
        decision = self._guard([self._gap(boundaries_open=1)])
        self.assertNotEqual(decision.decision_package.default, "amend_evaluator")

    def test_a_deadline_before_the_open_date_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            self._gap(deadline="2026-07-01T00:00:00Z")


# =============================================================================
# OPERATOR_INPUT_REQUIRED
# =============================================================================

class OperatorInputTests(unittest.TestCase):

    def _question(self, **overrides):
        kwargs = dict(
            question_id="q-1", package=_package(), raised_at=NOW, receipt="ev-q",
        )
        kwargs.update(overrides)
        return G.OperatorQuestion(**kwargs)

    def test_no_open_question_continues(self):
        self.assertEqual(G.guard_operator_input([], now=NOW).outcome, G.CONTINUE)

    def test_an_open_blocking_question_stops_with_its_package(self):
        decision = G.guard_operator_input([self._question()], now=NOW)
        self.assertEqual(decision.stop_state, SM.OPERATOR_INPUT_REQUIRED)
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )
        self.assertEqual(len(decision.detail["options"]), 2)

    def test_an_answered_question_clears(self):
        question = self._question(answered=True, answered_event_id="ev-answer")
        self.assertEqual(G.guard_operator_input([question], now=NOW).outcome, G.CONTINUE)

    def test_an_answer_with_no_record_cannot_be_constructed(self):
        """Clearing a block on an unrecorded answer is the fail-open shape."""
        with self.assertRaises(G.GuardInputError):
            self._question(answered=True)
        with self.assertRaises(G.GuardInputError):
            self._question(answered=True, answered_event_id="")

    def test_a_question_marked_answered_by_a_later_edit_cannot_clear_a_block(self):
        """`dataclasses.replace` re-runs validation, so there is no back door."""
        question = self._question()
        with self.assertRaises(G.GuardInputError):
            dataclasses.replace(question, answered=True, answered_event_id=None)

    def test_a_non_blocking_question_does_not_stop(self):
        question = self._question(blocking=False)
        self.assertEqual(G.guard_operator_input([question], now=NOW).outcome, G.CONTINUE)


# =============================================================================
# Retry and repair caps
# =============================================================================

class RetryAndRepairTests(unittest.TestCase):

    def test_retries_are_bounded_at_three_then_root_cause_analysis(self):
        for attempts, expected in ((1, G.CONTINUE), (3, G.CONTINUE), (4, G.REFUSE)):
            with self.subTest(attempts=attempts):
                ledger = G.CommandRetryLedger(
                    command_id="cmake --build", attempts=attempts,
                    last_error="ninja: build stopped", receipt="ev-cmd",
                )
                decision = G.guard_command_retries(ledger)
                self.assertEqual(decision.outcome, expected)
                if expected == G.REFUSE:
                    self.assertEqual(
                        decision.directives, (G.DIRECTIVE_ROOT_CAUSE_ANALYSIS,)
                    )

    def test_a_campaign_may_not_raise_the_retry_cap(self):
        with self.assertRaises(G.GuardInputError):
            G.CommandRetryLedger(
                command_id="c", attempts=1, last_error="e", receipt="ev",
                max_retries=G.MAX_COMMAND_RETRIES + 1,
            )

    def test_a_ledger_for_a_command_that_never_ran_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            G.CommandRetryLedger(
                command_id="c", attempts=0, last_error="e", receipt="ev"
            )

    def test_the_retry_guard_never_stops_the_campaign(self):
        ledger = G.CommandRetryLedger(
            command_id="c", attempts=99, last_error="e", receipt="ev"
        )
        self.assertIsNone(G.guard_command_retries(ledger).stop_state)

    def test_a_repair_cap_refuses_the_repair_and_emits_a_degraded_signal(self):
        ledger = G.RepairLedger(
            proposal_id="akp-1", repairs_attempted=2, max_repairs=2,
            consecutive_build_failures=0, max_consecutive_build_failures=2,
            receipt="ev-repair",
        )
        decision = G.guard_repair_cap(ledger)
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertIn(G.DIRECTIVE_REPAIR_FORBIDDEN, decision.directives)
        self.assertIn("repair_cap_exceeded", decision.detail["planner_degraded_signals"])
        self.assertIsNone(decision.stop_state)

    def test_consecutive_build_failures_over_the_cap_refuse_the_repair(self):
        ledger = G.RepairLedger(
            proposal_id="akp-1", repairs_attempted=0, max_repairs=2,
            consecutive_build_failures=3, max_consecutive_build_failures=2,
            receipt="ev-repair",
        )
        decision = G.guard_repair_cap(ledger)
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertIn(
            "consecutive_build_failures", decision.detail["planner_degraded_signals"]
        )

    def test_within_caps_continues(self):
        ledger = G.RepairLedger(
            proposal_id="akp-1", repairs_attempted=1, max_repairs=2,
            consecutive_build_failures=1, max_consecutive_build_failures=2,
            receipt="ev-repair",
        )
        self.assertEqual(G.guard_repair_cap(ledger).outcome, G.CONTINUE)


# =============================================================================
# Disposition
# =============================================================================

class DispositionTests(unittest.TestCase):

    def _stop(self, state):
        builders = {
            SM.INTEGRITY_STOP: lambda: G.guard_integrity(G.IntegrityLedger(
                signals=(G.IntegritySignal("s", NOW, "ev-i"),),
                consecutive_failures=1, max_consecutive_integrity_failures=0,
            )),
            SM.ANCHOR_MOVED: lambda: G.guard_anchor_moved(
                recorded=_anchor(), observed=_anchor(commit=V7_COMMIT), receipt="ev-a",
            ),
            SM.HOST_REBOOT_REQUIRED: lambda: G.guard_host_uptime(
                G.HostHealth(
                    uptime_seconds=G.HOST_UPTIME_CEILING_SECONDS, observed_at=NOW,
                    receipt="ev-h",
                ),
                owner="operator", escalation_deadline=LATER, now=NOW,
            ),
            SM.RESOURCE_UNAVAILABLE: lambda: G.guard_resource_available(
                G.ResourceClaimObservation(
                    resource="gpu:0", claim_kind="device", acquired=False,
                    observed_at=NOW, unavailable_reason="held elsewhere",
                )
            ),
            SM.DISK_PRESSURE: lambda: G.guard_storage_headroom(
                _storage(free=1, floor=500, backlog=0)
            ),
            SM.BUDGET_STOP: lambda: G.guard_budget(_budget(max_wall_hours=100.0)),
            SM.EVALUATOR_COVERAGE_GAP: lambda: G.guard_evaluator_coverage(
                [G.CoverageGap(
                    gap_id="g", missing_coverage_class="c", blocked_lineage="L",
                    owner="op", deadline="2026-08-02T00:00:00Z",
                    opened_at="2026-08-01T00:00:00Z", receipt="ev-g",
                )],
                now=NOW, covered_surfaces_remaining=1, escalation_owner="operator",
                escalation_deadline=LATER,
            ),
            SM.OPERATOR_INPUT_REQUIRED: lambda: G.guard_operator_input(
                [G.OperatorQuestion(
                    question_id="q", package=_package(), raised_at=NOW, receipt="ev-q"
                )],
                now=NOW,
            ),
            SM.PLANNER_DEGRADED: lambda: G.guard_planner_degraded(
                _health(consecutive_noop_rounds=9, receipts={"repeated_no_ops": "ev-n"}),
                _health_policy(),
            ),
            SM.EXHAUSTED_SURFACE: lambda: G.guard_exhausted_surface(
                reason=CLOSURE_REASON, closure=_closure(), accept_control=_accept(),
                planner_decision=_clean_planner_decision(_health()), health=_health(),
                eligible_layers_remaining=0,
            ),
            SM.PLATEAU_STOP: lambda: G.guard_plateau(
                reason=CLOSURE_REASON,
                series=_series([0.30, 0.3001, 0.3002, 0.3001, 0.3, 0.3]),
                policy=_plateau_policy(), closure=_closure(), accept_control=_accept(),
                planner_decision=_clean_planner_decision(_health()), health=_health(),
            ),
        }
        decision = builders[state]()
        assert decision.outcome == G.STOP, (state, decision.outcome, decision.reason)
        return decision

    def test_every_guard_decidable_stop_is_reachable(self):
        for state in G.STOP_PRECEDENCE:
            with self.subTest(state=state):
                self.assertEqual(self._stop(state).stop_state, state)

    def test_precedence_is_respected_regardless_of_supplied_order(self):
        pairs = [
            (SM.INTEGRITY_STOP, SM.PLATEAU_STOP),
            (SM.ANCHOR_MOVED, SM.BUDGET_STOP),
            (SM.HOST_REBOOT_REQUIRED, SM.EXHAUSTED_SURFACE),
            (SM.PLANNER_DEGRADED, SM.PLATEAU_STOP),
            (SM.BUDGET_STOP, SM.EVALUATOR_COVERAGE_GAP),
        ]
        for higher, lower in pairs:
            with self.subTest(higher=higher, lower=lower):
                a, b = self._stop(higher), self._stop(lower)
                self.assertEqual(G.dispose([a, b]).stop_state, higher)
                self.assertEqual(G.dispose([b, a]).stop_state, higher)

    def test_could_not_evaluate_outranks_a_refusal_and_never_clears(self):
        unevaluable = G.GuardDecision(
            guard_id=G.GUARD_ANCHOR, outcome=G.COULD_NOT_EVALUATE, reason="unreadable"
        )
        refusal = G.guard_command_retries(G.CommandRetryLedger(
            command_id="c", attempts=9, last_error="e", receipt="ev"
        ))
        disposition = G.dispose([refusal, unevaluable])
        self.assertEqual(disposition.outcome, G.COULD_NOT_EVALUATE)
        self.assertFalse(disposition.clears)

    def test_a_refusal_governs_over_a_continue(self):
        disposition = G.dispose([
            G.guard_budget(_budget()),
            G.guard_controller_spend(_budget(max_controller_tokens=95.0),
                                     G.SpendBreakerPolicy(0.8, "ev")),
        ])
        self.assertEqual(disposition.outcome, G.REFUSE)
        self.assertIn(G.DIRECTIVE_LOCAL_PLANNING_ONLY, disposition.directives)

    def test_all_clear_continues(self):
        disposition = G.dispose([
            G.guard_budget(_budget()),
            G.guard_anchor_moved(recorded=_anchor(), observed=_anchor(), receipt="ev"),
        ])
        self.assertEqual(disposition.outcome, G.CONTINUE)
        self.assertTrue(disposition.clears)

    def test_an_empty_round_cannot_be_evaluated_and_never_clears(self):
        """RED-TEAM: an unchecked round must not reduce to the same verdict as a
        clean one. `dispose([])` returning CONTINUE let the whole guard plane be
        passed by supplying nothing to it."""
        disposition = G.dispose([])
        self.assertEqual(disposition.outcome, G.COULD_NOT_EVALUATE)
        self.assertFalse(disposition.clears)
        self.assertEqual(disposition.stops, ())
        self.assertIsNone(disposition.governing)

    def test_disposition_is_deterministic(self):
        decisions = [self._stop(SM.BUDGET_STOP), self._stop(SM.PLATEAU_STOP)]
        self.assertEqual(
            G.dispose(decisions).governing.reason,
            G.dispose(list(decisions)).governing.reason,
        )

    def test_dispose_refuses_a_non_decision(self):
        with self.assertRaises(G.GuardInputError):
            G.dispose(["PLATEAU_STOP"])

    def test_packages_are_surfaced_for_every_escalating_decision(self):
        disposition = G.dispose([self._stop(SM.HOST_REBOOT_REQUIRED)])
        self.assertEqual(len(disposition.decision_packages), 1)


# =============================================================================
# The LLM requests, the controller disposes (§8.10 last line)
# =============================================================================

class StopRequestDisposalTests(unittest.TestCase):

    def _plateau_stop(self):
        return G.guard_plateau(
            reason=CLOSURE_REASON,
            series=_series([0.30, 0.3001, 0.3002, 0.3001, 0.3, 0.3]),
            policy=_plateau_policy(), closure=_closure(), accept_control=_accept(),
            planner_decision=_clean_planner_decision(_health()), health=_health(),
        )

    def test_a_request_no_guard_reached_is_denied(self):
        request = SM.StopRequest(
            state=SM.EXHAUSTED_SURFACE,
            reason="I have tried everything I can think of",
            detail={"closed": [], "deferred": []},
            origin="planner",
        )
        decision = G.dispose_requested_stop(request, [])
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertEqual(decision.directives, (G.DIRECTIVE_REQUEST_DENIED,))
        self.assertIsNone(decision.stop_state)

    def test_an_operator_origin_buys_nothing(self):
        """AK-D38: authorship is not evidence, and the trusted author is the risk."""
        outcomes = set()
        for origin in sorted(SM.STOP_REQUEST_ORIGINS):
            request = SM.StopRequest(
                state=SM.PLATEAU_STOP, reason="looks done", detail={}, origin=origin
            )
            outcomes.add(G.dispose_requested_stop(request, []).outcome)
        self.assertEqual(outcomes, {G.REFUSE})

    def test_a_request_a_guard_reached_returns_the_guards_decision(self):
        guard_decision = self._plateau_stop()
        request = SM.StopRequest(
            state=SM.PLATEAU_STOP, reason="narrative reason", detail={"vibes": "flat"},
            origin="planner",
        )
        disposed = G.dispose_requested_stop(request, [guard_decision])
        self.assertIs(disposed, guard_decision)
        self.assertNotIn("vibes", disposed.detail)
        self.assertEqual(disposed.reason, CLOSURE_REASON)

    def test_the_denial_records_the_origin_and_uses_it_for_nothing(self):
        request = SM.StopRequest(
            state=SM.PLATEAU_STOP, reason="r", detail={}, origin="critic"
        )
        decision = G.dispose_requested_stop(request, [])
        self.assertEqual(decision.detail["request_origin"], "critic")
        self.assertTrue(decision.detail["origin_is_not_evidence"])

    def test_a_request_for_a_different_state_than_the_guard_reached_is_denied(self):
        guard_decision = self._plateau_stop()
        request = SM.StopRequest(
            state=SM.EXHAUSTED_SURFACE, reason="r", detail={}, origin="planner"
        )
        decision = G.dispose_requested_stop(request, [guard_decision])
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertIn(SM.PLATEAU_STOP, decision.detail["stops_reached_by_guards"])

    def test_dispose_requested_stop_refuses_a_non_request(self):
        with self.assertRaises(G.GuardInputError):
            G.dispose_requested_stop("PLATEAU_STOP", [])


# =============================================================================
# End to end: every guard STOP is admissible to the real state machine
# =============================================================================

class StateMachineAdmissibilityTests(unittest.TestCase):
    """The seam test. Two files agreeing in prose is how a gate goes dead."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.journal = J.Journal(root=str(root / "journal"))
        self.journal.initialize()
        self.controller_root = str(root / "controller")

    def _machine(self):
        machine = SM.ControllerStateMachine(
            journal_=self.journal, root=self.controller_root,
        )
        machine.bootstrap(anchor=_anchor())
        return machine

    def test_every_guard_stop_is_accepted_by_the_machine(self):
        builder = DispositionTests()
        for state in G.STOP_PRECEDENCE:
            with self.subTest(state=state):
                decision = builder._stop(state)
                # A fresh machine per state: stops are terminal by construction.
                sub = tempfile.TemporaryDirectory()
                self.addCleanup(sub.cleanup)
                journal_ = J.Journal(root=str(Path(sub.name) / "journal"))
                journal_.initialize()
                machine = SM.ControllerStateMachine(
                    journal_=journal_, root=str(Path(sub.name) / "controller")
                )
                machine.bootstrap(anchor=_anchor())
                transition = machine.stop(
                    decision.stop_state,
                    reason=decision.reason,
                    detail=decision.detail,
                    trigger=f"guard:{decision.guard_id}",
                )
                self.assertEqual(transition.to_state, state)
                self.assertTrue(machine.is_stopped())

    def test_a_denied_stop_request_never_reaches_the_machine(self):
        machine = self._machine()
        request = SM.StopRequest(
            state=SM.PLATEAU_STOP, reason="the search feels finished", detail={},
            origin="planner",
        )
        disposed = G.dispose_requested_stop(request, [])
        self.assertEqual(disposed.outcome, G.REFUSE)
        self.assertIsNone(disposed.stop_state)
        self.assertEqual(machine.state, SM.DISCOVER)

    def test_the_machine_would_also_refuse_a_bare_exhausted_claim(self):
        """Belt and braces: both planes refuse, and neither relies on the other."""
        machine = self._machine()
        with self.assertRaises(SM.StopEvidenceMissing):
            machine.stop(
                SM.EXHAUSTED_SURFACE,
                reason="we have exhausted all paths",
                detail={"closed": [], "deferred": []},
            )
        self.assertEqual(machine.state, SM.DISCOVER)


# =============================================================================
# Red-team regressions — each one reproduces a defect this module actually had
# =============================================================================

class RedTeamRegressionTests(unittest.TestCase):
    """One test per defect found by adversarial review on 2026-08-03.

    The organising question was the one that finds fail-open checks: *can I make
    this check pass by DELETING the thing it inspects?* Four of the six below
    answered yes.
    """

    # -- delete the subject, keep the PASS ------------------------------------

    def test_ast_audit_cannot_pass_on_a_source_with_nothing_in_it(self):
        """The audit read `__file__`, parsed it, walked it, found no forbidden
        node, and returned PASS — which is also exactly what it did for an empty
        string. A truncated module is not an audited module."""
        for source in ("", "# nothing to see here\n", "\n\n\n", "X = 1\n"):
            with self.subTest(source=source):
                self.assertEqual(
                    G.audit_no_write_process_or_wait_paths(source).outcome,
                    S.COULD_NOT_CHECK,
                )
        # ...and the emptiness excuse must not rescue a forbidden shape.
        self.assertEqual(
            G.audit_no_write_process_or_wait_paths("import os\n").outcome, S.FAIL
        )
        self.assertEqual(G.audit_no_write_process_or_wait_paths().outcome, S.PASS)

    def test_directive_audit_cannot_pass_on_an_emptied_vocabulary(self):
        """Deleting every directive made `audit_directive_vocabulary()` PASS: no
        directive contains 'WAIT' if no directive exists."""
        original = G.DIRECTIVES
        try:
            G.DIRECTIVES = frozenset()
            self.assertEqual(
                G.audit_directive_vocabulary().outcome, S.COULD_NOT_CHECK
            )
        finally:
            G.DIRECTIVES = original
        original_tokens = G.FORBIDDEN_DIRECTIVE_TOKENS
        try:
            G.FORBIDDEN_DIRECTIVE_TOKENS = ()
            self.assertEqual(
                G.audit_directive_vocabulary().outcome, S.COULD_NOT_CHECK
            )
        finally:
            G.FORBIDDEN_DIRECTIVE_TOKENS = original_tokens
        self.assertEqual(G.audit_directive_vocabulary().outcome, S.PASS)

    def test_an_unchecked_round_does_not_reduce_to_a_clean_one(self):
        """`dispose([])` returned CONTINUE with `clears is True`. Supplying no
        guard verdict must not be indistinguishable from every guard clearing."""
        empty = G.dispose([])
        self.assertEqual(empty.outcome, G.COULD_NOT_EVALUATE)
        self.assertFalse(empty.clears)
        clean = G.dispose([G.guard_budget(_budget())])
        self.assertEqual(clean.outcome, G.CONTINUE)
        self.assertNotEqual(empty.outcome, clean.outcome)

    def test_reserved_closure_words_are_caught_in_the_layer_list_too(self):
        """The scan covered `reason`, sub-scopes, gates and deferral reasons but
        not `hierarchy_layers_considered` — a free-text field of the same
        enumeration, in the same object, read by the same operator."""
        ledger = _closure(
            hierarchy_layers_considered=("L1 parameter", "all paths at L2 measured"),
        )
        result = G.check_closure_language(CLOSURE_REASON, ledger)
        self.assertEqual(result.outcome, S.FAIL)
        self.assertTrue(
            any("hierarchy_layers_considered" in reason for reason in result.reasons),
            result.reasons,
        )
        # And the guard that consumes it refuses rather than closing the surface.
        health = _health()
        decision = G.guard_exhausted_surface(
            reason=CLOSURE_REASON,
            closure=ledger,
            accept_control=_accept(),
            planner_decision=_clean_planner_decision(health),
            health=health,
            eligible_layers_remaining=0,
        )
        self.assertEqual(decision.outcome, G.REFUSE)
        self.assertIsNone(decision.stop_state)
        # The clean layer list still passes: the fix must not forbid the
        # compliant idiom it exists to permit.
        self.assertEqual(
            G.check_closure_language(CLOSURE_REASON, _closure()).outcome, S.PASS
        )

    # -- validated once, mutable forever --------------------------------------

    def test_a_validated_stop_cannot_be_mutated_out_from_under_its_validation(self):
        """`detail` was stored as the CALLER's mapping. The construction-time
        `check_stop_evidence` therefore expired the moment the caller wrote to
        that dict, leaving an object that reports as admissible while carrying
        evidence the machine refuses."""
        detail = {"resource": "gpu0", "claim_kind": "device_exclusive"}
        decision = G.GuardDecision(
            guard_id=G.GUARD_RESOURCE,
            outcome=G.STOP,
            stop_state=SM.RESOURCE_UNAVAILABLE,
            reason="the device claim was not acquired",
            detail=detail,
            directives=(G.DIRECTIVE_PERSIST_AND_DRAIN,),
            evidence=("ev-claim",),
        )
        detail.clear()
        detail["vibes"] = "the planner felt strongly about it"
        self.assertEqual(decision.detail["resource"], "gpu0")
        self.assertNotIn("vibes", decision.detail)
        self.assertEqual(
            SM.check_stop_evidence(
                decision.stop_state, decision.reason, decision.detail
            ).outcome,
            S.PASS,
        )
        # `to_dict()` must not hand back a writable view of nested evidence either.
        health = _health()
        plateau = G.guard_plateau(
            reason=CLOSURE_REASON,
            series=_series([0.30, 0.301, 0.3005, 0.3009, 0.3008, 0.3007]),
            policy=_plateau_policy(),
            closure=_closure(),
            accept_control=_accept(),
            planner_decision=_clean_planner_decision(health),
            health=health,
        )
        self.assertEqual(plateau.outcome, G.STOP)
        rendered = plateau.to_dict()
        rendered["detail"]["planner_health"]["degraded_ruled_out"] = "maybe"
        self.assertIs(plateau.detail["planner_health"]["degraded_ruled_out"], True)

    # -- a label believed over the numbers it labels ---------------------------

    def test_storage_headroom_is_not_read_off_a_contradicted_status_string(self):
        """`StorageState.pressured` is `state == DISK_PRESSURE`. A state whose
        string says anything else read as healthy no matter what its numbers
        said — §2.5 row 4's "the budget was only a status string", on the disk."""
        observation = G.StorageObservation(
            path="/mnt/raid0/llm",
            state=ST.StorageState(
                state="LOOKS_FINE_TO_ME",
                free_bytes=1, total_bytes=10 ** 12, floor_bytes=10 ** 11,
            ),
            expirable_backlog_bytes=0,
            receipt="ev-storage",
        )
        decision = G.guard_storage_headroom(observation)
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertFalse(decision.clears)
        self.assertIsNone(decision.stop_state)
        # The reverse disagreement is refused too: a DISK_PRESSURE label over
        # numbers that clear the floor must not stop a healthy campaign either.
        inverted = G.StorageObservation(
            path="/mnt/raid0/llm",
            state=ST.StorageState(
                state=ST.DISK_PRESSURE,
                free_bytes=10 ** 12, total_bytes=10 ** 12, floor_bytes=1,
            ),
            expirable_backlog_bytes=0,
            receipt="ev-storage",
        )
        self.assertEqual(
            G.guard_storage_headroom(inverted).outcome, G.COULD_NOT_EVALUATE
        )
        # Agreeing observations are unaffected in both directions.
        self.assertEqual(
            G.guard_storage_headroom(_storage(free=900, floor=100, backlog=0)).outcome,
            G.CONTINUE,
        )
        self.assertEqual(
            G.guard_storage_headroom(_storage(free=50, floor=500, backlog=0)).outcome,
            G.STOP,
        )

    # -- a status sentinel wearing an identity's field --------------------------

    def test_an_accept_side_receipt_must_name_its_control(self):
        """`control_id`'s default was the UNAVAILABLE *status* sentinel, so every
        receipt built without naming its control journalled
        `accept_side_control.control_id == "HISTORICAL_REPLAY_UNAVAILABLE"` inside
        the evidence of a closure stop — a live gate's record reading as the
        record of a gate nobody could observe."""
        with self.assertRaises(G.GuardInputError):
            G.AcceptSideControlReceipt(
                status=G.ACCEPT_CONTROL_PROMOTED,
                event_id="ev-control-5",
                observed_at=NOW,
                cadence=S.Check(S.PASS),
                win_id="iqk-port",
            )
        with self.assertRaises(G.GuardInputError):
            G.AcceptSideControlReceipt(
                status=G.ACCEPT_CONTROL_FAILED_TO_PROMOTE,
                event_id="ev-control-5",
                observed_at=NOW,
                cadence=S.Check(S.PASS),
            )
        # An UNAVAILABLE receipt may carry the sentinel: there it is the truth.
        unavailable = G.AcceptSideControlReceipt(
            status=G.ACCEPT_CONTROL_UNAVAILABLE,
            event_id="ev-control-5",
            observed_at=NOW,
            cadence=S.Check(S.PASS),
        )
        self.assertFalse(unavailable.promoted)
        self.assertEqual(
            _accept().to_dict()["control_id"], "control-5-historical-win-replay"
        )

    # -- state that grows with campaign length ---------------------------------

    def test_a_clean_integrity_verdict_does_not_carry_the_whole_campaign(self):
        """The CONTINUE branch emitted one evidence entry per integrity signal
        EVER recorded, every round, for the life of the campaign."""
        signals = tuple(
            G.IntegritySignal(signal=f"probe-{index}", at=NOW, receipt=f"ev-int-{index}")
            for index in range(500)
        )
        decision = G.guard_integrity(G.IntegrityLedger(
            signals=signals,
            consecutive_failures=0,
            max_consecutive_integrity_failures=0,
        ))
        self.assertEqual(decision.outcome, G.CONTINUE)
        self.assertEqual(decision.evidence, ())
        # Bounded, but not silently discarded: the history is still counted.
        self.assertEqual(decision.detail["signals_recorded"], 500)
        # A tolerated run still names exactly its own receipts, not the history.
        tolerated = G.guard_integrity(G.IntegrityLedger(
            signals=signals,
            consecutive_failures=2,
            max_consecutive_integrity_failures=3,
        ))
        self.assertEqual(tolerated.outcome, G.CONTINUE)
        self.assertEqual(tolerated.evidence, ("ev-int-498", "ev-int-499"))
        # And the STOP branch is unchanged: it names the run that crossed.
        stopping = G.guard_integrity(G.IntegrityLedger(
            signals=signals,
            consecutive_failures=2,
            max_consecutive_integrity_failures=1,
        ))
        self.assertEqual(stopping.outcome, G.STOP)
        self.assertEqual(stopping.evidence, ("ev-int-498", "ev-int-499"))


class TheReleasePlaneSeamCannotHandOverAPhantomMagnitudeTest(unittest.TestCase):
    """The AK5 -> AK4 seam, exercised across it rather than assumed on both sides.

    `release.readiness` returns a MAPPING (AK5 does not import AK4), and the
    guarantee this file is responsible for is that the parity mapping cannot
    become a number in the plateau series. Asserting that inside AK5 alone would
    be asserting it against a consumer that was never run.
    """

    def _fields(self, cells):
        from ..release import readiness as R  # local: AK4 does not depend on AK5
        from ..release import test_readiness as T
        signal = T.green_signal(cells=cells)
        return R, T, signal.figure_for("decode").observation_fields()

    def test_a_parity_mapping_cannot_construct_a_readiness_observation(self):
        from ..release import test_readiness as T
        _R, _T, fields = self._fields(T.parity_cells())
        with self.assertRaises(TypeError):
            G.ReadinessObservation(round_index=0, at=NOW, **fields)

    def test_the_seam_builds_a_parity_round_for_a_parity_phase(self):
        from ..release import test_readiness as T
        _R, _T, fields = self._fields(T.parity_cells())
        entry = G.observation_from_fields(round_index=3, at=NOW, fields=fields)
        self.assertIsInstance(entry, G.ParityObservation)
        self.assertNotIsInstance(entry, G.ReadinessObservation)
        with self.assertRaises(G.ParityHasNoMagnitude):
            entry.readiness

    def test_the_seam_builds_a_readiness_round_for_an_orderable_phase(self):
        """The control."""
        from ..release import test_readiness as T
        _R, _T, fields = self._fields(T.green_cells())
        entry = G.observation_from_fields(round_index=3, at=NOW, fields=fields)
        self.assertIsInstance(entry, G.ReadinessObservation)
        self.assertEqual(entry.readiness, fields["readiness"])

    def test_both_states_land_in_one_series_the_plateau_rule_accepts(self):
        from ..release import test_readiness as T
        _R, _T, parity_fields = self._fields(T.parity_cells())
        _R2, _T2, orderable_fields = self._fields(T.green_cells())
        series = (
            G.observation_from_fields(round_index=0, at=NOW, fields=orderable_fields),
            G.observation_from_fields(round_index=1, at=NOW, fields=parity_fields),
        )
        self.assertEqual([type(entry).__name__ for entry in series],
                         ["ReadinessObservation", "ParityObservation"])

    def test_a_mapping_claiming_both_or_neither_is_refused(self):
        with self.assertRaises(G.GuardInputError):
            G.observation_from_fields(
                round_index=0, at=NOW,
                fields={"readiness": 0.3, "cells_at_parity": 4, "protected_cells": 4,
                        "mde": 0.02, "noise_floor": 0.01, "source_event_id": "ev-1"})
        with self.assertRaises(G.GuardInputError):
            G.observation_from_fields(round_index=0, at=NOW,
                                      fields={"source_event_id": "ev-1"})

    def test_the_seam_takes_a_mapping_not_whatever_it_is_handed(self):
        with self.assertRaises(G.GuardInputError):
            G.observation_from_fields(round_index=0, at=NOW, fields=[("readiness", 0.3)])

    def test_the_seam_carries_the_sensitivity_the_producer_published(self):
        """Not one this side re-derived from the numbers beside it.

        A consumer that recomputed `max(mde, noise_floor)` would be a second copy
        of a rule that lives in `ParityFigure`, free to drift the moment the
        producer's bound binds on something else.
        """
        from ..release import test_readiness as T
        _R, _T, fields = self._fields(T.parity_cells())
        entry = G.observation_from_fields(round_index=0, at=NOW, fields=fields)
        figure = T.green_signal(cells=T.parity_cells()).figure_for("decode")
        self.assertEqual(entry.sensitivity_bound, figure.sensitivity_bound)
        self.assertEqual(entry.reference_gain, figure.comparable_reference_gain)

    def test_the_two_planes_answer_the_power_question_identically(self):
        """The anti-drift mechanism for the one predicate that exists on both sides.

        `ParityFigure.could_have_detected` renders the operator's "UNDERPOWERED
        FOR THIS CAMPAIGN" clause and `ParityObservation.could_have_detected`
        decides whether a window may be read as a plateau. They cannot import
        each other, so what keeps them honest is running both over one figure.
        """
        from ..release import test_readiness as T
        figure = T.green_signal(cells=T.parity_cells()).figure_for("decode")
        entry = G.observation_from_fields(
            round_index=0, at=NOW, fields=figure.observation_fields())
        for magnitude in (0.0, figure.sensitivity_bound, 0.02, 0.25, -0.5):
            self.assertEqual(figure.could_have_detected(magnitude),
                             entry.could_have_detected(magnitude), magnitude)


class AConvergingCampaignReachesAnAnswerEndToEndTest(unittest.TestCase):
    """A whole campaign, driven from real release-plane figures into the stop rule.

    THE SHAPE THIS EXISTS FOR: an early round finds a real improvement, and the
    rounds after it measure every protected cell and resolve nothing. That is
    what convergence looks like under a non-inferiority objective — parity is the
    HEALTHY outcome — and it is the case the stop rule exists to recognise.

    Two misreadings sit either side of it and they point opposite ways. Reading a
    parity round as `0.0` invents a trend and stops (or continues) on a quantity
    nobody measured; refusing to read the window at all never stops, and since a
    converged campaign goes all-parity and STAYS all-parity, "never" is literal.
    Both are checked here across the seam rather than on either side of it,
    because each plane on its own can be green while the pair is wrong.
    """

    TARGET = 0.25
    WINDOW = 5

    def _plane(self):
        from ..release import readiness as R  # local: AK4 does not depend on AK5
        from ..release import test_readiness as T
        return R, T

    def _policy(self):
        R, _T = self._plane()
        return R.ReferencePolicy(reference_point_gain=self.TARGET,
                                 reference_lcb_gain=0.20)

    def _improving_fields(self, value: float) -> dict:
        """A round whose weakest protected prefill cell measured a real gain."""
        _R, T = self._plane()
        cells = T.green_cells()
        prefill = T.cell(
            "cell-prefill-a", phase="prefill", protocol_id=T.PREFILL_PROTOCOL,
            non_inferiority=T.non_inferior_evidence(
                value=value, effect_per_block=value, metric="prefill_tokens_per_s",
                raw_ref="ak-raw://champion/prefill/blocks.jsonl"))
        signal = T.green_signal(cells=cells[:2] + (prefill,) + cells[3:],
                                reference=self._policy())
        return signal.figure_for("prefill").observation_fields()

    def _parity_fields(self) -> dict:
        """A round that measured its protected prefill cell and resolved nothing."""
        _R, T = self._plane()
        signal = T.green_signal(cells=T.parity_cells(), reference=self._policy())
        return signal.figure_for("prefill").observation_fields()

    def _series(self, fields_by_round):
        return tuple(G.observation_from_fields(round_index=index, at=NOW, fields=fields)
                     for index, fields in enumerate(fields_by_round))

    def _decide(self, series):
        health = _health()
        return G.guard_plateau(
            reason="the campaign converged: no round in the window resolved an effect",
            series=series, policy=_plateau_policy(window_rounds=self.WINDOW),
            closure=_closure(), accept_control=_accept(), health=health,
            planner_decision=_clean_planner_decision(health))

    # -- fixture honesty: the campaign really has the shape claimed -----------

    def test_the_campaign_really_is_one_gain_followed_by_parity(self):
        """Without this the two tests below could both be passing on nothing."""
        opening = self._series([self._improving_fields(0.06)])[0]
        self.assertIsInstance(opening, G.ReadinessObservation)
        self.assertAlmostEqual(opening.readiness, 0.06)
        later = self._series([self._parity_fields()])[0]
        self.assertIsInstance(later, G.ParityObservation)
        # And the parity rounds could have SEEN the campaign's target, so the
        # tests below are about the candidate and not about the instrument.
        self.assertTrue(later.could_have_detected(self.TARGET))
        self.assertLess(later.sensitivity_bound, self.TARGET)

    # -- the two window positions --------------------------------------------

    def test_convergence_is_a_plateau_while_the_winning_round_is_still_in_view(self):
        """Window = [gain, parity x4]. The subtraction is real and comes out flat."""
        series = self._series([self._improving_fields(0.06)]
                              + [self._parity_fields()] * 4)
        decision = self._decide(series)
        self.assertEqual(decision.outcome, G.STOP)
        self.assertEqual(decision.stop_state, SM.PLATEAU_STOP)
        self.assertEqual(decision.detail["plateau_basis"],
                         G.PLATEAU_BASIS_MEASURED_IMPROVEMENT)
        self.assertAlmostEqual(decision.detail["improvement"], 0.0)
        # The zero is a SUBTRACTION of two measured magnitudes, not a substituted
        # one: both ends of it are the round that actually measured something.
        self.assertAlmostEqual(decision.detail["opening_readiness"], 0.06)
        self.assertAlmostEqual(decision.detail["best_readiness"], 0.06)
        self.assertEqual(decision.detail["parity_rounds"], 4)
        for entry in decision.detail["window"][1:]:
            self.assertNotIn("readiness", entry)

    def test_the_answer_does_not_become_a_stall_when_that_round_slides_out(self):
        """Window = [parity x5]. THE case the guard used to refuse forever.

        The campaign is more converged than it was one round earlier, not less,
        so an answer that flips from STOP to "cannot evaluate" the moment the
        last orderable round leaves the window is the guard reporting its own
        arithmetic rather than the campaign.
        """
        series = self._series([self._improving_fields(0.06)]
                              + [self._parity_fields()] * 5)
        decision = self._decide(series)
        self.assertEqual(decision.outcome, G.STOP)
        self.assertEqual(decision.stop_state, SM.PLATEAU_STOP)
        self.assertEqual(decision.detail["plateau_basis"],
                         G.PLATEAU_BASIS_NO_DETECTABLE_EFFECT)
        self.assertEqual(decision.detail["orderable_rounds"], 0)
        self.assertEqual(decision.detail["reference_gain"], self.TARGET)
        # ... and NOT as a plateau of zeros. Nothing was subtracted, so nothing
        # is reported as having been subtracted.
        for invented in ("improvement", "opening_readiness", "best_readiness"):
            self.assertNotIn(invented, decision.detail)
        for entry in decision.detail["window"]:
            self.assertIs(entry["orderable"], False)
            self.assertNotIn("readiness", entry)

    def test_it_keeps_stopping_as_the_converged_campaign_runs_on(self):
        """"Never stops" is not a state a longer run gets out of. Nor should the fix be."""
        rounds = [self._improving_fields(0.06)] + [self._parity_fields()] * 8
        for length in range(self.WINDOW + 1, len(rounds) + 1):
            decision = self._decide(self._series(rounds[:length]))
            self.assertEqual(decision.outcome, G.STOP, length)
            self.assertEqual(decision.detail["plateau_basis"],
                             G.PLATEAU_BASIS_NO_DETECTABLE_EFFECT, length)

    # -- the controls ---------------------------------------------------------

    def test_a_campaign_still_finding_gains_is_not_stopped(self):
        """The compliant path, with a parity round sitting in the middle of it."""
        series = self._series([self._improving_fields(0.06),
                               self._improving_fields(0.10),
                               self._parity_fields(),
                               self._improving_fields(0.20),
                               self._improving_fields(0.30)])
        decision = self._decide(series)
        self.assertEqual(decision.outcome, G.CONTINUE)
        self.assertAlmostEqual(decision.detail["improvement"], 0.24)
        self.assertEqual(decision.detail["parity_rounds"], 1)

    def test_a_converged_campaign_that_never_declared_a_target_is_not_stopped(self):
        """No reference policy: the release plane publishes `None` and the guard says so.

        This is the one shape that still cannot be concluded — and unlike the
        blanket refusal it replaces, it names something the campaign can fix.
        """
        _R, T = self._plane()
        fields = T.green_signal(cells=T.parity_cells()).figure_for(
            "prefill").observation_fields()
        self.assertIsNone(fields["reference_gain"])
        decision = self._decide(self._series([fields] * self.WINDOW))
        self.assertEqual(decision.outcome, G.COULD_NOT_EVALUATE)
        self.assertIn("does not carry one campaign target", decision.reason)


class TheTwoKindsOfRoundStaySeparateOnTheWireTest(unittest.TestCase):
    """`guard_plateau` serialises its whole window into the stop detail.

    Whatever reads that detail reads `to_dict()`, so the type split has to survive
    the trip. `ParityObservation` publishes `orderable: false`, which is the key a
    reader will branch on — and a discriminator carried only on the negative side
    makes `entry.get("orderable", False)` answer "no round is orderable", silently
    emptying the window rather than failing.
    """

    def test_the_discriminator_is_carried_on_both_kinds_of_round(self):
        self.assertIs(
            G.ReadinessObservation(round_index=0, readiness=0.3, at=NOW,
                                   source_event_id="ev-0").to_dict()["orderable"], True)
        self.assertIs(_parity_round(1).to_dict()["orderable"], False)

    def test_a_defaulting_reader_cannot_lose_the_orderable_rounds(self):
        health = _health()
        decision = G.guard_plateau(
            reason=CLOSURE_REASON,
            series=_mixed_series([0.30, 0.31, PARITY, 0.33, 0.34, 0.35]),
            policy=_plateau_policy(), closure=_closure(), accept_control=_accept(),
            health=health, planner_decision=_clean_planner_decision(health))
        window = decision.detail["window"]
        orderable = [entry for entry in window if entry.get("orderable", False)]
        self.assertEqual(len(orderable), len(window) - decision.detail["parity_rounds"])
        self.assertGreater(len(orderable), 0)

    def test_the_two_kinds_are_distinguishable_after_a_json_round_trip(self):
        pair = [G.ReadinessObservation(round_index=0, readiness=0.3, at=NOW,
                                       source_event_id="ev-0").to_dict(),
                _parity_round(1).to_dict()]
        revived = [json.loads(json.dumps(entry, sort_keys=True)) for entry in pair]
        self.assertIs(revived[0]["orderable"], True)
        self.assertIs(revived[1]["orderable"], False)
        # Two independent refusals on the wire, and they must not collapse into
        # one: the parity payload carries no magnitude at all, AND it says which
        # kind of round it is. A reader that only checks for a `readiness` key
        # gets a KeyError; a reader that branches on `orderable` gets an answer.
        self.assertEqual(revived[0]["readiness"], 0.3)
        self.assertNotIn("readiness", revived[1])


class TheSeamDefaultsNothingTest(unittest.TestCase):
    """A key the producer did not send may not become a value it never sent.

    `observation_from_fields` branches on which keys EXIST, and it used to fill
    the rest in with `.get(key, fallback)`. On the parity side those fallbacks
    were `mde=0.0` and `noise_floor=0.0` — not a missing sensitivity but the
    SHARPEST one expressible, so a mapping that lost a key produced the strongest
    parity claim there is: "we resolved to zero and nothing moved". `stratum`
    defaulted to `confirmation`, which enforces P-AK-SEARCH-1 only against
    producers that volunteer the field, and that is not enforcement.
    """

    def _parity_fields(self, **over):
        fields = dict(protected_cells=12, cells_at_parity=12, mde=0.018,
                      noise_floor=0.01, sensitivity_bound=0.018,
                      reference_gain=CAMPAIGN_TARGET, source_event_id="ev-p",
                      stratum=evaluator_api.STRATUM_CONFIRMATION)
        fields.update(over)
        return fields

    def _build(self, fields):
        return G.observation_from_fields(round_index=0, at=NOW, fields=fields)

    def test_a_missing_sensitivity_is_refused_not_read_as_perfect_resolution(self):
        for key in ("mde", "noise_floor", "sensitivity_bound"):
            fields = self._parity_fields()
            del fields[key]
            with self.assertRaises(G.GuardInputError, msg=key) as ctx:
                self._build(fields)
            self.assertIn(key, str(ctx.exception))
            # And refused BY THE SEAM, for being absent — not caught downstream
            # by an invariant that happens to dislike the substituted value. A
            # default that a later constructor rejects is still a default, and
            # the next producer to publish a genuine zero would sail through it.
            self.assertIn("is missing", str(ctx.exception), key)

    def test_a_missing_protected_cell_count_is_refused(self):
        fields = self._parity_fields()
        del fields["protected_cells"]
        with self.assertRaises(G.GuardInputError):
            self._build(fields)

    def test_a_missing_campaign_target_is_refused_though_none_is_a_legal_value(self):
        """The one key whose absence and whose `None` mean different things.

        `reference_gain=None` says "this campaign declared no target". A DROPPED
        key would render as the same thing under `.get()` — and that reading
        disables the only branch that can conclude anything from an all-parity
        window, so a producer bug would present as a campaign that never stops.
        """
        fields = self._parity_fields()
        del fields["reference_gain"]
        with self.assertRaises(G.GuardInputError) as ctx:
            self._build(fields)
        self.assertIn("reference_gain", str(ctx.exception))
        # And the explicit None is accepted, because it is an answer.
        self.assertIsNone(self._build(self._parity_fields(reference_gain=None))
                          .reference_gain)

    def test_a_missing_stratum_is_not_promoted_to_confirmation(self):
        """On BOTH branches: the only stratum a series admits is not a fallback."""
        for fields in (self._parity_fields(),
                       dict(readiness=0.3, source_event_id="ev-r",
                            stratum=evaluator_api.STRATUM_CONFIRMATION)):
            del fields["stratum"]
            with self.assertRaises(G.GuardInputError) as ctx:
                self._build(fields)
            self.assertIn("stratum", str(ctx.exception))

    def test_a_missing_source_event_is_refused_rather_than_blanked(self):
        for fields in (self._parity_fields(),
                       dict(readiness=0.3, stratum=evaluator_api.STRATUM_CONFIRMATION,
                            source_event_id="ev-r")):
            del fields["source_event_id"]
            with self.assertRaises(G.GuardInputError):
                self._build(fields)

    def test_the_complete_mappings_still_build_both_kinds(self):
        """The control. A whole mapping from either figure must still pass."""
        self.assertIsInstance(self._build(self._parity_fields()), G.ParityObservation)
        orderable = self._build(dict(readiness=0.3, source_event_id="ev-r",
                                     stratum=evaluator_api.STRATUM_CONFIRMATION))
        self.assertIsInstance(orderable, G.ReadinessObservation)
        self.assertEqual(orderable.readiness, 0.3)

    def test_a_defaulting_reader_cannot_lose_the_orderable_rounds(self):
        health = _health()
        decision = G.guard_plateau(
            reason=CLOSURE_REASON,
            series=_mixed_series([0.30, 0.31, PARITY, 0.33, 0.34, 0.35]),
            policy=_plateau_policy(), closure=_closure(), accept_control=_accept(),
            health=health, planner_decision=_clean_planner_decision(health))
        window = decision.detail["window"]
        orderable = [entry for entry in window if entry.get("orderable", False)]
        self.assertEqual(len(orderable), len(window) - decision.detail["parity_rounds"])
        self.assertGreater(len(orderable), 0)

    def test_the_two_kinds_are_distinguishable_after_a_json_round_trip(self):
        pair = [G.ReadinessObservation(round_index=0, readiness=0.3, at=NOW,
                                       source_event_id="ev-0").to_dict(),
                _parity_round(1).to_dict()]
        revived = [json.loads(json.dumps(entry, sort_keys=True)) for entry in pair]
        self.assertIs(revived[0]["orderable"], True)
        self.assertIs(revived[1]["orderable"], False)
        self.assertIn("readiness", revived[0])
        self.assertNotIn("readiness", revived[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
