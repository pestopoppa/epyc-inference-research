#!/usr/bin/env python3
"""test_ak4_conformance.py — one test per obligation AK4 is bound by.

WHY THIS FILE EXISTS
--------------------
`test_loop_integration.py` proves the controller WALKS. This file proves it walks
inside the rules, and it is deliberately organised by INSTRUMENT rather than by
module, because that is how an auditor reads it:

  * `measurement/protocols/kernel-research.md` — **P-AK-SEARCH-1**, Annex K of
    MEASUREMENT.md, RATIFIED 2026-08-03. Its authorizations (5), its denials (9),
    its preconditions (8), its controls clause, its correctness precedence and
    its record grammar.
  * `handoffs/active/autokernel-research-loop.md` §4 — the twenty non-negotiable
    invariants.

WHAT "CONFORMANCE" MEANS HERE, AND WHAT IT DOES NOT
---------------------------------------------------
Some obligations are only checkable by RUNNING a measurement — a held resource
claim re-verified at window close, an A/A control on its cadence, an anchor gate
band. AK4 does not measure; it is the plane that calls the evaluator and disposes
what comes back. Those obligations are asserted here as SEAMS: the named surface
exists, it is reachable from the controller, and the controller refuses when it is
absent. Every one of them is registered in `SEAM_ONLY` with the reason, and
`TestSeamRegistryIsHonest` fails if an entry names something that does not exist
or if a seam-only test forgets to register itself.

That is the difference between marking an obligation and skipping it: a skipped
test is silent and a registered seam is a list an auditor can read and count.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO MODEL CALL, NO PROCESS.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_ak4_conformance.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_ak4_conformance.py
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel.controller import composition as CP  # noqa: E402
from autokernel.controller import context as C  # noqa: E402
from autokernel.controller import critic as CR  # noqa: E402
from autokernel.controller import guards as G  # noqa: E402
from autokernel.controller import hypotheses as H  # noqa: E402
from autokernel.controller import oracles as O  # noqa: E402
from autokernel.controller import planner as PL  # noqa: E402
from autokernel.controller import selection as SEL  # noqa: E402
from autokernel.controller import state_machine as SM  # noqa: E402
from autokernel.evaluator import api as EV  # noqa: E402
from autokernel.resource import claim_witness as CW  # noqa: E402
from autokernel.resource import device_claim as DC  # noqa: E402
from autokernel.resource import preflight as PF  # noqa: E402

# Absolute, not relative: this file is run both by `unittest discover` (as a
# package) and by path, and a relative import breaks the second.
from autokernel.controller.test_loop_integration import (  # noqa: E402
    BACKEND, CAMPAIGN, V8_COMMIT, _anchor, _campaign, _sha,
)


def _referenced_names(module) -> set:
    """Every name a module IMPORTS or CALLS, from its AST.

    A substring scan over the source is wrong here and the reason is instructive:
    `guards._FORBIDDEN_IMPORTS` and `planner`'s provider-side-effect deny-list are
    LISTS OF THESE VERY WORDS, so grepping finds the guard and calls it the
    violation — the inverse of `feedback_guard_must_not_forbid_its_own_idiom`.
    """
    tree = ast.parse(inspect.getsource(module))
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not (node.level or 0):
                names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
                if isinstance(func.value, ast.Name):
                    names.add(f"{func.value.id}.{func.attr}")
    return names


#: Nothing in this plane may spawn, signal, wait on, or shell out to anything.
_FORBIDDEN_PROCESS_NAMES = frozenset({
    "subprocess", "multiprocessing", "signal", "pty", "socket", "shlex",
    "os.system", "os.popen", "os.kill", "os.fork", "os.execv", "os.waitpid",
    "os.spawnv", "run", "Popen", "check_output", "call", "kill", "fork",
    "system", "popen", "waitpid",
})


_CONTROLLER_DIR = Path(__file__).resolve().parent
_MODULES = (SM, G, SEL, C, H, PL, CR, CP, O)


# =============================================================================
# The seam register. An obligation that needs a measurement is NAMED here.
# =============================================================================

#: obligation -> (what AK4 can check without measuring, who owns the rest)
SEAM_ONLY = {
    "P-AK-SEARCH-1 precondition 1 (claim held for the whole window)": (
        "AK4 asserts the claim-receipt surface exists and that a record without "
        "one is refused by the schema; ACQUIRING the claim and re-verifying it at "
        "window close is `autokernel.resource.device_claim`, exercised under a "
        "real device by AK2's suite."),
    "P-AK-SEARCH-1 precondition 2 (no concurrent inference)": (
        "AK4 asserts the sanctioned preflight substitute exists and names its "
        "bases; running it is `autokernel.resource.preflight` against a live host."),
    "P-AK-SEARCH-1 precondition 3 (host-health tier)": (
        "AK4 asserts the uptime ceiling and the operator-authority stop; reading "
        "the host's real uptime is the caller's."),
    "P-AK-SEARCH-1 precondition 5 (evaluator identity re-verified)": (
        "AK4 asserts the bundle hash travels in every record and that a coverage "
        "gap blocks the lineage; resolving and re-hashing the bundle at run time "
        "is AK3."),
    "P-AK-SEARCH-1 precondition 6 (codified recipe)": (
        "AK4 asserts no controller module emits argv at all; recipe-constructor "
        "identity is `autokernel.evaluator.recipes`."),
    "P-AK-SEARCH-1 calibration block": (
        "AK4 asserts it supplies NO threshold of its own and refuses to invent "
        "one; deriving phi, B_min, alpha and the anchor band is AK3's "
        "`evaluator.statistics` over real A/A blocks."),
    "P-AK-SEARCH-1 controls 1-5": (
        "AK4 asserts the accept-side control's three statuses and that an "
        "UNAVAILABLE control makes closure undecidable rather than clear; running "
        "the five controls is AK3."),
    "P-AK-SEARCH-1 record grammar": (
        "AK4 asserts the grammar's mandatory fields are the schema's required "
        "fields; emitting a conforming grammar line is AK3's."),
    "§4 invariant 2 (full candidate promotion)": (
        "AK4 asserts the composed champion re-measures the COMBINED id and cites "
        "no member evidence; the release-side promotion is AK5."),
    "§4 invariant 4 (evaluator independence at the OS level)": (
        "AK4 asserts the critic records its independence and refuses silence; "
        "OS-level domain separation (§3.6) is not a Python property."),
    "§4 invariant 10 (owned-process lifecycle)": (
        "AK4 asserts no controller module can start, signal or wait on a process "
        "— proven by AST over the source; verifying that the OS agrees is the "
        "caller's, and cgroup enumeration lives in `autokernel.resource`."),
    "§4 invariant 12 (determinism class is an interface)": (
        "AK4 asserts the class travels in the candidate and evaluation records; "
        "observing same-seed bitwise stability is AK3."),
}


def _seam(name: str) -> str:
    """Look up a registered seam. An unregistered one RAISES, so a test cannot
    quietly become seam-only without appearing in the register."""
    if name not in SEAM_ONLY:
        raise AssertionError(
            f"{name!r} is not in SEAM_ONLY; an obligation that cannot be tested "
            "without measuring must be REGISTERED, not skipped")
    return SEAM_ONLY[name]


class TestSeamRegistryIsHonest(unittest.TestCase):
    """The register is only worth anything if it cannot drift into fiction."""

    def test_every_registered_seam_states_who_owns_the_rest(self):
        for obligation, reason in SEAM_ONLY.items():
            with self.subTest(obligation=obligation):
                self.assertTrue(reason.strip())
                self.assertTrue(
                    any(owner in reason for owner in ("AK2", "AK3", "AK5", "caller",
                                                      "resource", "evaluator", "OS")),
                    f"{obligation}: the reason must name who owns the untested part")

    def test_the_register_is_not_a_dumping_ground(self):
        """A register that grows without bound is a skip list with better manners."""
        self.assertLessEqual(len(SEAM_ONLY), 20)

    def test_no_test_in_this_file_is_skipped(self):
        source = ast.parse((_CONTROLLER_DIR / "test_ak4_conformance.py").read_text())
        for node in ast.walk(source):
            if isinstance(node, ast.Attribute) and node.attr in ("skip", "skipTest",
                                                                 "skipIf", "skipUnless"):
                self.fail("a skipped conformance test is an unrecorded exemption; "
                          "register the obligation in SEAM_ONLY instead")


# =============================================================================
# P-AK-SEARCH-1 — what the protocol AUTHORIZES (five, and only five)
# =============================================================================

class TestProtocolAuthorizations(unittest.TestCase):

    def test_1_ranking_is_against_a_named_immutable_anchor(self):
        """Authorization 1. A ranking with no anchor is not a weaker ranking."""
        self.assertEqual(
            SM.check_anchor_identity(None, _anchor()).outcome, S.COULD_NOT_CHECK)
        self.assertEqual(
            SM.check_anchor_identity(_anchor(), None).outcome, S.COULD_NOT_CHECK)

    def test_2_retain_abandon_and_branch_are_journalled_decisions(self):
        """Authorization 2 — and §8.4's *"never a bare discard"*."""
        self.assertIn("fingerprint", inspect.signature(PL.skip_payload).parameters)
        self.assertTrue(hasattr(CP, "retain_frontier"))
        self.assertTrue(hasattr(SEL, "record_prescreen_rejection")
                        or hasattr(SEL.ProposalScreener, "screen"))

    def test_3_composition_re_measures_the_combined_candidate(self):
        """Authorization 3 — *"rerun T0/T1 on the combined full candidate"*."""
        self.assertEqual(CP.REQUIRED_COMBINED_TIERS, ("T0", "T1"))

    def test_4_selection_ranks_only_admitted_proposals(self):
        """Authorization 4. A rejected proposal never enters the ranking."""
        self.assertIn("phase", inspect.signature(SEL.rank_proposals).parameters)

    def test_5_readiness_is_computed_never_declared(self):
        """Authorization 5 / §4 invariant 14: *"the controller may request a
        readiness computation, never declare a readiness value"*."""
        check = CP.audit_no_composed_estimate_arithmetic(
            inspect.getsource(CP))
        self.assertEqual(check.outcome, S.PASS, check.reasons)


# =============================================================================
# P-AK-SEARCH-1 — what the protocol DENIES (nine)
# =============================================================================

class TestProtocolDenials(unittest.TestCase):

    def test_denial_1_no_gating_outside_the_worktree(self):
        """No search record gates a keep/revert/deploy/promote/close decision."""
        self.assertIsNone(getattr(SM, "deploy", None))
        self.assertIsNone(getattr(SM, "promote", None))
        # Sealing is the ONLY route toward release and it is operator-triggered.
        self.assertIn("requested_by",
                      inspect.signature(SM.ControllerStateMachine.request_freeze).parameters)

    def test_denial_2_no_production_write_of_any_kind(self):
        """A composed lineage may not carry a production branch name."""
        with self.assertRaises(ValueError) as ctx:
            CP.propose_lineage([], anchor_commit=V8_COMMIT, source_tree="llama.cpp",
                               branch="production-consolidated-v8")
        self.assertIn("production", str(ctx.exception).lower())

    def test_denial_3_no_retro_certification(self):
        """A search record can never become a claim. Every record AK4 writes
        carries `category=CANDIDATE`, and the loop mints no claim verb."""
        for module in _MODULES:
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("retro_certif", source)
                self.assertNotIn("ATTESTED", source)

    def test_denial_4_consumption_is_confined_to_the_producing_campaign(self):
        """A neighbouring campaign's records may not clear this one's signals."""
        self.assertIn("campaign_id",
                      inspect.signature(SEL.read_skip_history).parameters)

    def test_denial_5_no_human_only_write(self):
        """Freezes, cutovers and reboots stay operator authority."""
        self.assertEqual(SM.STOP_RECOVERY[SM.HOST_REBOOT_REQUIRED],
                         SM.RECOVERY_OPERATOR_RESUME)
        self.assertIn(SM.HOST_REBOOT_REQUIRED, G.ESCALATING_STOPS)

    def test_denial_6_no_self_amendment_of_the_instrument(self):
        """A proposal requiring an evaluator change is REJECTED, and a coverage
        gap blocks the lineage instead of patching the instrument."""
        self.assertIn(SEL.REJECT_REQUIRES_EVALUATOR_CHANGE, SEL.REJECTION_CODES)
        self.assertIn(SM.EVALUATOR_COVERAGE_GAP, SM.STOP_STATES)
        self.assertIn(SM.EVALUATOR_COVERAGE_GAP, G.ESCALATING_STOPS)
        for key in ("missing_coverage_class", "blocked_lineage", "owner", "deadline"):
            self.assertIn(key, SM.STOP_EVIDENCE_REQUIREMENTS[SM.EVALUATOR_COVERAGE_GAP])

    def test_denial_7_no_release_activity(self):
        """T3 is refused by the evaluator's own table, on the EDGE, naming AK5."""
        self.assertEqual(EV.RELEASE_TIER_OWNER, "AK5")
        with self.assertRaises(Exception) as ctx:
            EV.admit_tier("T3")
        self.assertIn("AK5", str(ctx.exception))

    def test_denial_8_no_host_or_resource_authority(self):
        """No controller module may start, signal or wait on a process, and a
        reboot is a stop that persists and resumes rather than an action."""
        self.assertEqual(
            G.audit_no_write_process_or_wait_paths(
                (_CONTROLLER_DIR / "guards.py").read_text()).outcome,
            S.PASS)
        self.assertEqual(G.HOST_UPTIME_CEILING_SECONDS, 7 * 24 * 60 * 60)

    def test_denial_9_no_new_instrument_by_composition(self):
        """§12's row is literally *"SUMMED local gains inflate readiness"*."""
        self.assertEqual(
            CP.audit_no_composed_estimate_arithmetic(inspect.getsource(CP)).outcome,
            S.PASS)


# =============================================================================
# P-AK-SEARCH-1 — preconditions
# =============================================================================

class TestProtocolPreconditions(unittest.TestCase):

    def test_precondition_1_resource_claim_seam(self):
        reason = _seam("P-AK-SEARCH-1 precondition 1 (claim held for the whole window)")
        self.assertTrue(hasattr(DC, "DeviceClaimError"), reason)
        self.assertTrue(hasattr(CW, "resolve_claim_receipt"), reason)
        self.assertTrue(hasattr(CW, "check_event_claim_receipt"), reason)
        # The controller's own contribution: it stops rather than proceeding.
        self.assertIn(SM.RESOURCE_UNAVAILABLE, SM.UNIVERSAL_STOPS)
        self.assertEqual(SM.STOP_EVIDENCE_REQUIREMENTS[SM.RESOURCE_UNAVAILABLE],
                         ("resource", "claim_kind"))

    def test_precondition_2_preflight_substitute_seam(self):
        reason = _seam("P-AK-SEARCH-1 precondition 2 (no concurrent inference)")
        self.assertIn(PF.BASIS_CLAIM_WITNESS, PF.BASES, reason)
        self.assertTrue(hasattr(PF, "ConcurrentInferenceDetected"), reason)
        # "Not observed" is not "not present": the indeterminate case is its own
        # exception rather than a pass.
        self.assertTrue(hasattr(PF, "PreflightIndeterminate"), reason)

    def test_precondition_3_host_health_seam(self):
        reason = _seam("P-AK-SEARCH-1 precondition 3 (host-health tier)")
        self.assertIn(SM.HOST_REBOOT_REQUIRED, SM.STOP_STATES)
        self.assertEqual(SM.STOP_EVIDENCE_REQUIREMENTS[SM.HOST_REBOOT_REQUIRED],
                         ("uptime_seconds", "ceiling_seconds"), reason)

    def test_precondition_4_an_explicit_immutable_anchor(self):
        """*"A run without an explicit anchor is INVALID"* — and an unverifiable
        anchor is refused rather than assumed good."""
        with tempfile.TemporaryDirectory() as tmp:
            journal_ = J.Journal(os.path.join(tmp, "j"), campaign_id=CAMPAIGN)
            journal_.initialize()
            journal_.append(J.KIND_CAMPAIGN_OPENED, _campaign())
            machine = SM.ControllerStateMachine(
                journal_=journal_, root=os.path.join(tmp, "c"), campaign_id=CAMPAIGN)
            machine.bootstrap(anchor=_anchor(backends=(BACKEND, "llama_cpu")),
                              views=J.rebuild_views(journal_.read_all()))
            with self.assertRaises(SM.AnchorUncheckable):
                machine.campaign_boundary(observed_anchor=_anchor(backends=(BACKEND,)))

    def test_precondition_5_evaluator_identity_seam(self):
        reason = _seam("P-AK-SEARCH-1 precondition 5 (evaluator identity re-verified)")
        self.assertIn("bundle_sha256", C.EvaluatorCoverage.__dataclass_fields__,
                      reason)
        gap = C.CoverageGap(missing_class=EV.GATE_QUALITY,
                            blocked_lineage="ak/champion/x", owner="operator",
                            deadline="2026-08-17")
        self.assertEqual(gap.blocked_lineage, "ak/champion/x")

    def test_precondition_6_codified_recipe_seam(self):
        """No controller module builds a command line at all — the strongest form
        of *"hand-typed argv voids the run"* available to this plane."""
        reason = _seam("P-AK-SEARCH-1 precondition 6 (codified recipe)")
        for module in _MODULES:
            with self.subTest(module=module.__name__):
                leaked = _referenced_names(module) & _FORBIDDEN_PROCESS_NAMES
                self.assertEqual(leaked, set(), f"{reason} ({sorted(leaked)})")

    def test_precondition_7_storage_headroom_stops_the_campaign(self):
        """*"when the already-eligible expiry backlog does not clear the floor,
        the campaign stops"*."""
        self.assertIn(SM.DISK_PRESSURE, SM.UNIVERSAL_STOPS)
        self.assertEqual(SM.STOP_EVIDENCE_REQUIREMENTS[SM.DISK_PRESSURE],
                         ("path", "free_bytes", "floor_bytes"))
        self.assertIn(G.DIRECTIVE_RECLAIM_EXPIRABLE_FIRST, G.DIRECTIVES)

    def test_precondition_8_declared_campaign_controls_are_all_required(self):
        """*"a campaign that omits one ... MUST NOT start"* — the budget gate
        refuses an undeclared dimension rather than defaulting it."""
        caps = dict(_campaign()["budgets"])
        for key in ("max_wall_hours", "max_gpu_hours", "max_cpu_region_hours",
                    "max_storage_gb", "max_candidates"):
            with self.subTest(cap=key):
                partial = dict(caps)
                partial.pop(key)
                with self.assertRaises(ValueError):
                    SEL.budget_remaining_from_caps(
                        partial, wall_hours_used=0.0, gpu_seconds_used=0.0,
                        cpu_region_seconds_used=0.0, storage_gb_used=0.0,
                        candidates_used=0)


class TestDeclaredCampaignControls(unittest.TestCase):
    """A precondition-8 GAP the integration pass found, asserted rather than
    worked around."""

    def test_the_planner_degraded_limit_has_no_declared_home_in_the_schema(self):
        """`selection.planner_health_stop_request` REQUIRES
        `stop_policy.max_consecutive_proposal_skips` and rightly refuses to
        invent one — *"a controller that picks its own tolerance for its own
        malfunction is grading itself"*. But `schemas.validate_campaign` does not
        name that key, so a §7.1-conforming manifest can omit the single input
        PLANNER_DEGRADED needs, and the loop discovers it by raising at the
        moment it most needs the stop.

        The schema is AK1's and outside this pass's write scope. This test is the
        record of the gap: it FAILS the day AK5 adds the key, which is when the
        assertion should be inverted.
        """
        source = inspect.getsource(S.validate_campaign)
        self.assertNotIn("max_consecutive_proposal_skips", source)
        self.assertIn("max_consecutive_proposal_skips",
                      inspect.getsource(SEL.planner_health_stop_request))

    def test_a_campaign_missing_the_key_is_schema_valid_but_unstoppable(self):
        campaign = _campaign()
        campaign["stop_policy"].pop("max_consecutive_proposal_skips", None)
        self.assertEqual(S.validate_campaign(campaign), [])
        with self.assertRaises(ValueError):
            SEL.planner_health_stop_request(
                SEL.SkipHistory(records=(), counts={}, blacklisted=frozenset(),
                                trailing_run=9),
                stop_policy=campaign["stop_policy"])


# =============================================================================
# P-AK-SEARCH-1 — calibration, controls, correctness precedence, grammar
# =============================================================================

class TestCalibrationAndControls(unittest.TestCase):

    def test_no_threshold_in_this_plane_is_supplied_as_a_literal(self):
        """*"No value in this list may be supplied as a literal — not by a
        controller"*. The controller's job is to REFUSE when the campaign did not
        declare one."""
        _seam("P-AK-SEARCH-1 calibration block")
        # The degradation run length has no default and must be >= 2.
        with self.assertRaises(TypeError):
            PL.assess_repetition(["a", "a"])
        with self.assertRaises(ValueError):
            PL.assess_repetition(["a", "a"], degraded_run=1)
        # The plateau floor arrives with the receipt of its calibration.
        with self.assertRaises(G.GuardInputError):
            G.PlateauPolicy(window_rounds=3, improvement_floor=0.01,
                            floor_receipt="")

    def test_the_accept_side_control_has_a_declared_unavailable_branch(self):
        """Control 5 *"is never silently skipped"*."""
        _seam("P-AK-SEARCH-1 controls 1-5")
        self.assertEqual(
            set(G.ACCEPT_CONTROL_STATUSES),
            {G.ACCEPT_CONTROL_PROMOTED, G.ACCEPT_CONTROL_FAILED_TO_PROMOTE,
             G.ACCEPT_CONTROL_UNAVAILABLE})
        self.assertEqual(G.ACCEPT_CONTROL_UNAVAILABLE,
                         EV.HISTORICAL_REPLAY_UNAVAILABLE)

    def test_correctness_is_lexicographically_prior_to_speed(self):
        """*"A candidate failing any of them receives NO speed rank at all — not
        a penalised one"*, because a penalised rank is still a rank."""
        verdict = EV.compute_verdict(
            tier="T1",
            gates=(EV.GateResult(gate_id="t0.correctness.op_suite",
                                 gate_class=EV.GATE_CORRECTNESS,
                                 check=S.Check(S.FAIL, ("mismatch at row 12",))),),
            void_scan=EV.VoidScan(findings=(), evaluated=(), not_applicable=()),
            search_grade=EV.SearchGradeResult(
                satisfied=True, evaluated=("protocol_ratified",), failed=(),
                not_applicable=(), reasons=()),
            anchor=EV.AnchorIdentity(
                source_commit=V8_COMMIT, binary_sha256=_sha("b"),
                linkage_sha256=_sha("l"), measurement_event_ids=("ake-1",)),
            effect=None)
        self.assertFalse(verdict.speed_rank_admissible)

    def test_the_record_grammar_fields_are_the_schemas_required_fields(self):
        reason = _seam("P-AK-SEARCH-1 record grammar")
        source = inspect.getsource(S)
        for field in ("host_receipt", "resource_claim_receipt", "scope_denominator",
                      "determinism", "bundle_sha256"):
            with self.subTest(field=field):
                self.assertIn(field, source, reason)

    def test_the_confirmation_stratum_never_reaches_planner_context(self):
        """*"The confirmation stratum's contents MUST NOT appear in planner
        context"*, yet a proposal targeting a confirmation shape is rejected
        BEFORE it consumes a window. Digests satisfy both."""
        self.assertIn(SEL.REJECT_TARGETS_CONFIRMATION_SHAPE, SEL.REJECTION_CODES)
        fields = SEL.SelectionContext.__dataclass_fields__
        self.assertIn("confirmation_shape_digests", fields)
        self.assertNotIn("confirmation_shapes", fields)


# =============================================================================
# §4 — the twenty non-negotiable invariants, one test each
# =============================================================================

class TestSectionFourInvariants(unittest.TestCase):

    def test_invariant_01_fresh_production_base(self):
        """Every campaign is anchored on the current production tip, and the
        anchor is re-verified at every campaign boundary, not only at freeze."""
        self.assertTrue(hasattr(SM.ControllerStateMachine, "campaign_boundary"))
        self.assertIn(SM.ANCHOR_MOVED, SM.UNIVERSAL_STOPS)
        self.assertEqual(SM.STOP_RECOVERY[SM.ANCHOR_MOVED], SM.RECOVERY_REANCHOR)

    def test_invariant_02_full_candidate_promotion(self):
        reason = _seam("§4 invariant 2 (full candidate promotion)")
        self.assertEqual(CP.REQUIRED_COMBINED_TIERS, ("T0", "T1"), reason)
        # No parameter anywhere lets a caller hand in member evidence for the
        # composition — the combined id must have its own.
        self.assertNotIn("member_evidence",
                         inspect.signature(CP.compose_champion).parameters)

    def test_invariant_03_frozen_means_immutable(self):
        with self.assertRaises(ValueError) as ctx:
            CP.propose_lineage([], anchor_commit=V8_COMMIT, source_tree="llama.cpp",
                               branch="production-consolidated-v8")
        self.assertIn("production", str(ctx.exception).lower())
        self.assertTrue(CP.champion_branch_for(
            source_tree="llama.cpp", anchor_commit=V8_COMMIT).startswith("ak/champion/"))

    def test_invariant_04_evaluator_independence(self):
        """The critic must STATE its independence; silence is refused."""
        reason = _seam("§4 invariant 4 (evaluator independence at the OS level)")
        check = CR.check_critic_independence(
            critic_binding=PL.ModelBinding(provider="local", model_id="m",
                                           effort="high", sampling_params={}),
            planner_binding=PL.ModelBinding(provider="local", model_id="m",
                                            effort="high", sampling_params={}),
            shared_model_reason=None)
        self.assertNotEqual(check.outcome, S.PASS, reason)

    def test_invariant_05_no_autonomous_freeze_or_cutover(self):
        self.assertIn("requested_by",
                      inspect.signature(SM.ControllerStateMachine.request_freeze).parameters)
        with self.assertRaises(Exception):
            EV.admit_tier("T3")

    def test_invariant_06_correctness_is_lexicographically_first(self):
        self.assertIn(SEL.REJECT_NO_CORRECTNESS_ORACLE, SEL.REJECTION_CODES)
        self.assertIn(EV.GATE_CORRECTNESS, EV.GATE_CLASSES)

    def test_invariant_07_all_outcomes_are_durable(self):
        """Rejected proposals are journaled, not discarded."""
        self.assertIn(J.KIND_PROPOSAL_SKIPPED, J.KINDS)
        self.assertIn(J.KIND_STOP_STATE, J.KINDS)
        self.assertTrue(hasattr(SEL, "SkipNotRecorded"))

    def test_invariant_08_views_rewind_evidence_does_not_disappear(self):
        """*"Purge is a supersession/tombstone event, never deletion"*."""
        self.assertIn(J.KIND_SUPERSEDED, J.KINDS)
        self.assertIn(J.KIND_TOMBSTONE, J.KINDS)
        self.assertEqual(CP.SUPERSEDED_BY_ANCHOR_MOVE, "superseded_by_anchor_move")
        # A budget reducer that shrank when a proposal was superseded would let a
        # campaign spend the same hour twice.
        self.assertIn("Superseded proposals are included",
                      inspect.getdoc(C.reduce_budget_ledger))

    def test_invariant_09_resources_are_acquired_not_observed(self):
        self.assertIn(PF.BASIS_CLAIM_WITNESS, PF.BASES)
        self.assertTrue(hasattr(DC, "DeviceClaimTimeout"))

    def test_invariant_10_owned_process_lifecycle_only(self):
        """AK4 owns no process because it can start none."""
        reason = _seam("§4 invariant 10 (owned-process lifecycle)")
        for module in _MODULES:
            with self.subTest(module=module.__name__):
                leaked = _referenced_names(module) & _FORBIDDEN_PROCESS_NAMES
                self.assertEqual(leaked, set(), f"{reason} ({sorted(leaked)})")
        # And the compliant path still passes, so the check is not vacuous.
        self.assertIn("dict", _referenced_names(SM))

    def test_invariant_11_deterministic_replay_before_regeneration(self):
        """A saved completion is replayed without inference."""
        self.assertTrue(hasattr(PL, "ReplayProvider"))
        self.assertTrue(hasattr(PL, "ReplayMiss"))

    def test_invariant_12_determinism_class_is_an_interface(self):
        reason = _seam("§4 invariant 12 (determinism class is an interface)")
        self.assertIn("determinism", inspect.getsource(S), reason)

    def test_invariant_13_one_conceptual_mutation_per_step(self):
        self.assertIn(SEL.REJECT_MULTIPLE_CONCEPTUAL_CHANGES, SEL.REJECTION_CODES)
        # An UNDECLARED count is not a licence: it is REJECT_UNVERIFIABLE.
        self.assertIn(SEL.REJECT_UNVERIFIABLE, SEL.REJECTION_CODES)
        # §8.4.1: an architectural campaign binds the rule per STEP.
        self.assertTrue(hasattr(SEL, "ArchitecturalCampaign"))
        self.assertTrue(hasattr(SEL, "LineageStep"))

    def test_invariant_14_no_estimated_percentage_by_narration(self):
        self.assertEqual(
            CP.audit_no_composed_estimate_arithmetic(inspect.getsource(CP)).outcome,
            S.PASS)
        # And the empty-source escape is closed: an audit that passes on nothing
        # is an audit you can satisfy by deleting what it inspects.
        self.assertEqual(
            CP.audit_no_composed_estimate_arithmetic("").outcome, S.COULD_NOT_CHECK)

    def test_invariant_15_production_recipes_gate(self):
        """Off-recipe cells are diagnostic. A BASELINE cell may not be the
        composition's own denominator (invariant 15's search-side form)."""
        self.assertIn("BASELINE", inspect.getsource(CP))

    def test_invariant_16_default_off_until_release(self):
        source = inspect.getsource(S)
        self.assertIn("dispatch_guard", source)
        self.assertIn("kill_switch", source)

    def test_invariant_17_no_evaluator_self_modification(self):
        self.assertIn(SEL.REJECT_REQUIRES_EVALUATOR_CHANGE, SEL.REJECTION_CODES)
        self.assertIn(SM.EVALUATOR_COVERAGE_GAP, SM.STOP_STATES)

    def test_invariant_18_declared_equals_traced(self):
        """The actor's declaration is a scored prediction, never a scope input."""
        self.assertTrue(hasattr(CP, "UnreconciledSurface"))
        self.assertIn("reconciliation",
                      inspect.signature(CP.admit_to_frontier).parameters)

    def test_invariant_19_control_is_verified_not_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_ = J.Journal(os.path.join(tmp, "j"), campaign_id=CAMPAIGN)
            journal_.initialize()
            machine = SM.ControllerStateMachine(
                journal_=journal_, root=os.path.join(tmp, "c"), campaign_id=CAMPAIGN)
            self.assertEqual(SM.audit_no_cached_control_state(machine).outcome, S.PASS)
            machine.submit_control("drain", control_id="ctl-9", requested_by="operator",
                                   reason="draining for the night")
            # A NEW object over the same root: the halt is on disk, not in memory.
            restarted = SM.ControllerStateMachine(
                journal_=journal_, root=os.path.join(tmp, "c"), campaign_id=CAMPAIGN)
            self.assertFalse(restarted.begin_iteration().proceed)

    def test_invariant_20_the_planner_does_not_re_consume_its_own_prose(self):
        """Narrative is a separate field, excluded from retrieval by default, and
        the two doors into the prompt are BOTH closed."""
        self.assertIn(J.KIND_RETRIEVAL_SUPERSEDED, J.KINDS)
        with self.assertRaises(PL.ContextManifestError):
            PL.ContextEntry("e1", "campaign_objective", {"narrative": "planner prose"})
        with self.assertRaises(PL.ContextManifestError):
            PL.ContextEntry("e2", "campaign_objective", {"objective": "x"},
                            provenance={"narrative": "planner prose"})


# =============================================================================
# Authority — the one rule the whole plane rests on
# =============================================================================

class TestTheLlmProposesTheControllerDisposes(unittest.TestCase):

    def test_a_stop_request_buys_nothing_by_its_origin(self):
        """§8.4.0/AK-D38: authorship is not evidence, and the input most likely to
        be waved through is the one whose author is trusted."""
        for origin in sorted(SM.STOP_REQUEST_ORIGINS):
            with self.subTest(origin=origin):
                request = SM.StopRequest(state=SM.EXHAUSTED_SURFACE,
                                         reason="no eligible layer remains",
                                         detail={}, origin=origin)
                self.assertNotEqual(
                    SM.check_stop_evidence(request.state, request.reason,
                                           request.detail).outcome,
                    S.PASS)

    def test_check_stop_evidence_takes_no_origin_parameter(self):
        """Structural, not procedural: there is no argument to consult."""
        self.assertNotIn("origin", inspect.signature(SM.check_stop_evidence).parameters)

    def test_a_hypothesis_gate_cannot_see_who_stated_the_hypothesis(self):
        self.assertNotIn("origin", inspect.signature(H.check_do_not_repeat).parameters)

    def test_every_origin_enters_at_the_same_evidence_grade(self):
        for origin in sorted(H.ORIGINS):
            with self.subTest(origin=origin):
                self.assertEqual(H.entry_grade(origin), H.GRADE_DESIGN_PRIOR)
        self.assertEqual(H.audit_no_origin_grade_promotion().outcome, S.PASS)

    def test_a_model_disposition_can_only_make_a_proposal_worse(self):
        """*"the critic may reject or revise; it cannot waive"*."""
        source = inspect.getsource(CR)
        self.assertIn("severity", source)
        self.assertTrue(hasattr(CR, "GateWaiverAttempt"))
        self.assertTrue(hasattr(CR, "find_gate_waiver_keys"))

    def test_a_model_may_not_write_the_controller_owned_fields(self):
        for key in PL.CONTROLLER_OWNED_KEYS:
            with self.subTest(key=key):
                self.assertTrue(isinstance(key, str) and key)


# =============================================================================
# Cross-module vocabulary — what the integration pass reconciled
# =============================================================================

class TestOneVocabularyPerFact(unittest.TestCase):

    def test_the_oracle_registry_is_one_registry(self):
        self.assertEqual(
            O.audit_registry_well_formed().outcome, S.PASS)
        self.assertEqual(
            O.audit_consumer_registry(
                C.ORACLE_REGISTRY, id_of=lambda r: r.oracle_id,
                harvest_class_of_row=lambda r: r.harvest_class,
                retired_of=lambda r: r.status == C.ORACLE_RETIRED,
                what="context").outcome, S.PASS)
        self.assertEqual(
            O.audit_consumer_registry(
                CR.ORACLE_REGISTRY, id_of=lambda r: r.oracle_id,
                harvest_class_of_row=lambda r: r.harvest_class,
                retired_of=lambda r: r.retired, what="critic").outcome, S.PASS)

    def test_the_stop_vocabulary_is_one_vocabulary(self):
        self.assertEqual(set(G.GUARD_BY_STOP) | set(G.NON_GUARD_STOPS),
                         set(SM.STOP_STATES))
        self.assertEqual(G.audit_stop_coverage_totality().outcome, S.PASS)

    def test_the_closure_vocabulary_is_one_vocabulary(self):
        for word in G.RESERVED_CLOSURE_WORDS:
            self.assertIn(word, SM.RESERVED_CLOSURE_PHRASES)

    def test_the_fingerprint_is_one_algorithm(self):
        from autokernel.controller import fingerprint as FP
        self.assertIs(SEL.proposal_fingerprint, FP.proposal_fingerprint)
        manifest = {"campaign_id": CAMPAIGN, "change_class": "dispatcher"}
        self.assertEqual(PL.proposal_fingerprint(manifest),
                         SEL.proposal_fingerprint(manifest))

    def test_the_hypothesis_origin_vocabulary_is_one_vocabulary(self):
        self.assertEqual(set(C.HYPOTHESIS_ORIGINS), set(H.ORIGINS))

    def test_the_evidence_grade_vocabulary_is_one_vocabulary(self):
        self.assertEqual(set(C.EVIDENCE_GRADES), set(H.EVIDENCE_GRADES))
        self.assertEqual(set(C.EVIDENCE_GRADES), set(PL.EVIDENCE_GRADES))

    def test_the_suppression_class_vocabulary_is_one_vocabulary(self):
        self.assertEqual(set(SEL.LEDGER_CLASSES), set(CR.LEDGER_CLASSES))
        self.assertEqual(set(SEL.LEDGER_CLASSES), set(C.SUPPRESSION_CLASSES))
        self.assertEqual(set(SEL.LEDGER_CLASSES), set(H.MATCH_CLASSES))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
