#!/usr/bin/env python3
"""test_api.py — the regression barrier for the AK3 evaluator's structural claims.

WHY THIS FILE EXISTS
--------------------
`scripts/kernel_rnd/kernel_eval.sh` was reviewed, validated against a real
kernel, and shipped — and it still emitted `"status":"OK"` unconditionally and
recorded `COH="coherent"` for any non-empty generation. Every one of its defects
was visible in the source; none of them was ASSERTED anywhere. This suite asserts
the properties that replace them, so a future edit that reintroduces one fails
here instead of contaminating `kernel_store.py`'s correct-only Pareto view again.

The properties under test, each traceable to a clause of
`measurement/protocols/kernel-research.md` (P-AK-SEARCH-1, RATIFIED 2026-08-03):

  * a `Verdict` cannot be constructed without its gate results, and cannot carry
    a status its own evidence does not imply;
  * an anchor-less comparison is `INVALID`, and a coherence gate that answers
    PASS without an anchor is demoted to `COULD_NOT_CHECK`;
  * a correctness failure makes a speed rank UNOBTAINABLE, not penalised;
  * each of the twelve void conditions produces `INVALID` carrying its reason;
  * emitted events validate against `schemas.validate_evaluation_event`;
  * T3 is refused, at wiring time and at dispatch time.

NO inference, NO benchmark, NO build, NO process, NO file is written. The suite
also asserts that last property of the module under test by running its own AST
self-audit.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_api.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/evaluator/test_api.py
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

# Import through the PACKAGE so `api.schemas` is the same module object the
# journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel.evaluator import api, devices  # noqa: E402
from autokernel.resource import claim_witness as CW  # noqa: E402
from autokernel.resource import device_claim as DC  # noqa: E402

PASS = S.Check(S.PASS)
NOW = "2026-08-03T12:00:00+00:00"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"


#: Distinguishes "no effect estimate" from "use the fixture default".
_DEFAULT = object()


def sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def fail(*reasons: str) -> S.Check:
    return S.Check(S.FAIL, tuple(reasons) or ("failed",))


def cnc(*reasons: str) -> S.Check:
    return S.Check(S.COULD_NOT_CHECK, tuple(reasons) or ("unreadable",))


# ---------------------------------------------------------------------------
# Fixtures. Deliberately built field-by-field: `api` has no all_clear() helper,
# because a fixture that fabricates PASS is the fixture that removes the signal
# under test.
# ---------------------------------------------------------------------------

def anchor() -> api.AnchorIdentity:
    return api.AnchorIdentity(
        source_commit=V8_COMMIT,
        binary_sha256=sha("anchor-binary"),
        linkage_sha256=sha("anchor-linkage"),
        measurement_event_ids=("ake-anchor-0001",),
    )


def controls(**overrides) -> api.ControlPanel:
    kwargs = dict(positive=PASS, neutral=PASS, degraded_negative=PASS, aa=PASS,
                  historical_replay=PASS)
    kwargs.update(overrides)
    return api.ControlPanel(**kwargs)


def recipe() -> api.RecipeReceipt:
    return api.RecipeReceipt(
        constructor_id="ak.microbench.llama_gpu.decode/v1",
        constructor_sha256=sha("recipe-constructor"),
        argv_sha256=sha("argv"),
    )


def campaign_controls(**overrides) -> api.CampaignControls:
    kwargs = dict(calibration_block_count=30, contribution_floor=0.02, max_candidates=100,
                  confirmation_admission_count=5, max_blocks_per_candidate=40,
                  storage_floor_bytes_free=200 * 1024 ** 3)
    kwargs.update(overrides)
    return api.CampaignControls(**kwargs)


def calibration(**overrides) -> api.CalibrationOutputs:
    kwargs = dict(
        backend="llama_gpu", phase="decode", cell_class="instrument_tokens_per_s",
        noise_floor_phi=0.009, b_min_blocks=10, alpha_sel=0.01, alpha_conf=0.002,
        anchor_gate_band=(0.97, 1.03), accepted=True,
        solve_order_recorded=api.CALIBRATION_SOLVE_ORDER,
        samples_ref="data/ak-gpu-1/calibration/aa-blocks.jsonl",
        e_process_construction_id="sign_martingale_predictable_lambda/v1",
    )
    kwargs.update(overrides)
    return api.CalibrationOutputs(**kwargs)


def window(**overrides) -> api.WindowAttestations:
    kwargs = dict(
        resource_claim_receipt="gpu_device.mi210_0:claim-20260803T1200Z-8801",
        resource_claim_open=PASS,
        resource_claim_close=PASS,
        resource_claim_same_holder=PASS,
        no_concurrent_inference=PASS,
        preflight_attestation_ref="ake-preflight-0007",
        host_receipt="host-health-20260803T1159Z",
        host_health=PASS,
        anchor_at_open=anchor(),
        anchor_at_close=anchor(),
        anchor_gate=PASS,
        evaluator_bundle=PASS,
        runtime_source_label=PASS,
        recipe=recipe(),
        storage_open=PASS,
        storage_close=PASS,
        strata=PASS,
        stopping_rule_id="ak.stopping.bounded_extension/v1",
        rule_immutability=PASS,
        order_randomized=PASS,
        order_seed="campaign-seed-4711",
        aa_cadence=PASS,
        controls=controls(),
        calibration=PASS,
        control_definitions_immutable=PASS,
        raw_evidence_ref="data/ak-gpu-1/raw/akc-0001/",
    )
    kwargs.update(overrides)
    return api.WindowAttestations(**kwargs)


def request(**overrides) -> api.EvaluationRequest:
    kwargs = dict(
        event_id="ake-0001",
        campaign_id="ak-llama_gpu-decode-20260803",
        candidate_id="akc-0001",
        tier="T1",
        backend="llama_gpu",
        phase="decode",
        cell_class="instrument_tokens_per_s",
        protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(
            source_sha256=sha("cand-source"),
            binary_sha256=sha("cand-binary"),
            linkage_sha256=sha("cand-linkage"),
        ),
        anchor=anchor(),
        evaluator=api.EvaluatorIdentity(
            id="P-AK-SEARCH-1/v1",
            bundle_sha256=sha("evaluator-bundle"),
            runtime_source_label_ref="ake-srclabel-0003",
        ),
        scope_denominator=api.ScopeDenominator(
            machine_subset="partial", numa_nodes=(), devices=("mi210_0",), cores=8),
        scope_manifest_sha256=sha("scope-manifest"),
        co_residency="single",
        determinism=api.DeterminismReport(
            determinism_class="bitwise_stable", same_seed_repeat_runs=3),
        metric="decode_tokens_per_s",
        metric_direction="higher_better",
        reps=10,
        change_class="parameter", anchor_tier="T1", transfer_ratio_to=(),
        created_at=NOW,
        campaign_controls=campaign_controls(),
        calibration=calibration(),
        device_state=devices.DeviceState(
            device_id="ROCm0", source="rocm-smi/v6.2",
            nominal_sclk_mhz=1700, min_sclk_ratio=0.9,
            samples=(devices.DeviceStateSample(
                sclk_mhz=1700, mclk_mhz=1600, power_w=180,
                temperature_c=61, under_measurement_load=True),),
            receipt_ref="akraw://state/healthy"),
    )
    kwargs.update(overrides)
    return api.EvaluationRequest(**kwargs)


def effect(**overrides) -> api.EffectEstimate:
    kwargs = dict(
        metric="decode_tokens_per_s",
        metric_direction="higher_better",
        value=0.061,
        e_value=140.0,
        threshold=100.0,          # 1/alpha_sel with alpha_sel = 0.01
        mde=0.021,
        noise_floor=0.009,
        paired_blocks=12,
        stratum=api.STRATUM_SELECTION,
        raw_samples=(41.1, 43.6, 41.4, 43.8),
        raw_samples_ref="data/ak-gpu-1/raw/akc-0001/blocks.jsonl",
        lcb_descriptive=0.031,
    )
    kwargs.update(overrides)
    return api.EffectEstimate(**kwargs)


class TransferAndNoisePresentationTest(unittest.TestCase):
    def test_effect_prints_its_noise_floor_adjacent_to_the_delta(self):
        row = effect(value=0.004, noise_floor=0.009)
        self.assertEqual(row.to_dict()["inside_noise_floor"], True)
        self.assertIn("delta=", row.to_dict()["delta_display"])
        self.assertIn("noise_floor=", row.to_dict()["delta_display"])
        self.assertIn("INSIDE_NOISE_FLOOR", row.to_dict()["delta_display"])


class DeviceStateGateTest(unittest.TestCase):
    def test_loaded_gpu_clock_drop_voids_the_window_and_is_serialized(self):
        throttled = devices.DeviceState(
            device_id="ROCm0", source="rocm-smi/v6.2",
            nominal_sclk_mhz=1700, min_sclk_ratio=0.9,
            samples=(devices.DeviceStateSample(
                sclk_mhz=800, mclk_mhz=1600, power_w=180,
                temperature_c=61, under_measurement_load=True),),
            receipt_ref="akraw://state/throttled")
        outcome = run(req=request(device_state=throttled), eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertIn(api.VOID_HOST_HEALTH_TIER_VIOLATION,
                      outcome.void_scan.reasons())
        self.assertTrue(outcome.event["device_state"]["throttle_observed"])

    def test_gpu_text_blob_or_absence_cannot_read_as_healthy(self):
        scan = api.check_preconditions(request(device_state=None), window())
        self.assertEqual(dict(scan.checks)["host_health_tier"].outcome,
                         S.COULD_NOT_CHECK)

    def test_event_carries_a_recomputable_change_class_keyed_transfer(self):
        transfer = api.TransferRatio(
            event_id="ake-ground-truth-0001", tier="T2",
            source_effect=0.02, target_effect=0.025)
        outcome = run(req=request(change_class="layout", anchor_tier="T2",
                                  transfer_ratio_to=(transfer,)), eff=effect())
        self.assertEqual(outcome.event["change_class"], "layout")
        self.assertEqual(outcome.event["anchor_tier"], "T2")
        self.assertAlmostEqual(outcome.event["transfer_ratio_to"][0]["ratio"], 0.8)
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])


def gates_ok() -> tuple:
    return (
        api.GateResult("mul_mat_exact_shapes", api.GATE_CORRECTNESS, PASS,
                       requires_anchor=True, evidence_ref="raw/tbo.json"),
        api.GateResult("output_coherence_vs_anchor", api.GATE_CORRECTNESS, PASS,
                       requires_anchor=True, evidence_ref="raw/coh.json"),
        api.GateResult("no_fallback_dispatch_trace", api.GATE_INTEGRITY, PASS),
        api.GateResult("ppl_margin", api.GATE_QUALITY, PASS),
        api.GateResult("state_rollback", api.GATE_STABILITY, PASS),
        api.GateResult("bitwise_same_seed", api.GATE_DETERMINISM, PASS,
                       requires_anchor=True),
    )


class _Runner:
    """A tier runner stub. It runs nothing; it returns the gate results it was given."""

    def __init__(self, tier: str, gates):
        self.tier = tier
        self._gates = tuple(gates)
        self.calls = 0

    def run_gates(self, req):
        self.calls += 1
        return self._gates


def dispatcher(tier="T1", gates=None) -> api.TierDispatcher:
    return api.TierDispatcher(gate_runners={tier: _Runner(tier, gates or gates_ok())})


def run(*, req=None, win=None, gates=None, eff=None, tier="T1"):
    req = req if req is not None else request(tier=tier)
    win = win if win is not None else window()
    return dispatcher(tier=req.tier, gates=gates).dispatch(req, win, effect=eff)


# ===========================================================================
# 1. A verdict cannot be constructed without its gate results
# ===========================================================================

class VerdictIsComputedTest(unittest.TestCase):

    def test_direct_construction_is_refused(self):
        """`Verdict(...)` is not a public constructor. `compute_verdict()` is."""
        sg = api.SearchGradeResult(True, (), (), (), ())
        with self.assertRaises(api.VerdictTampering) as ctx:
            api.Verdict(
                tier="T1", status=api.STATUS_PASS, gates=(), void_findings=(),
                search_grade=sg, anchor=anchor(), effect=None,
                effect_resolution=api.EFFECT_NOT_MEASURED, speed_rank_admissible=False,
                integrity_flags=(), derivation=(),
            )
        self.assertIn("derives from the gate results", str(ctx.exception))

    def test_holding_the_mint_token_still_cannot_stamp_a_status(self):
        """The second lock: the verdict re-derives its status from its own evidence.

        This is the test that matters. Reaching in and taking the module-private
        mint token buys nothing — the only status the object accepts is the one
        its gate results imply. `kernel_eval.sh` printed `"status":"OK"`; there is
        no analogue of that line reachable here.
        """
        failing = (api.GateResult("mul_mat_exact_shapes", api.GATE_CORRECTNESS,
                                  fail("2 of 4096 shapes diverged")),)
        sg = api.SearchGradeResult(True, (), (), (), ())
        derived = api._derive(gates=failing, void_findings=(), search_grade=sg,
                              anchor=anchor(), effect=None)
        self.assertEqual(derived.status, api.STATUS_FAIL)

        with self.assertRaises(api.VerdictTampering) as ctx:
            api.Verdict(
                tier="T1", status=api.STATUS_PASS,      # <- the lie
                gates=failing, void_findings=(), search_grade=sg, anchor=anchor(),
                effect=None, effect_resolution=api.EFFECT_NOT_MEASURED,
                speed_rank_admissible=False, integrity_flags=(), derivation=(),
                mint=api._MINT_TOKEN,
            )
        self.assertIn("does not follow from this verdict's own evidence", str(ctx.exception))

    def test_faked_integrity_flags_are_also_refused(self):
        sg = api.SearchGradeResult(True, (), (), (), ())
        derived = api._derive(gates=gates_ok(), void_findings=(), search_grade=sg,
                              anchor=anchor(), effect=None)
        with self.assertRaises(api.VerdictTampering):
            api.Verdict(
                tier="T1", status=derived.status, gates=gates_ok(), void_findings=(),
                search_grade=sg, anchor=anchor(), effect=None,
                effect_resolution=derived.effect_resolution,
                speed_rank_admissible=derived.speed_rank_admissible,
                integrity_flags=("CORRECTNESS:something:FAIL",),  # <- not derivable
                derivation=derived.derivation, mint=api._MINT_TOKEN,
            )

    def test_dataclasses_replace_cannot_launder_a_status(self):
        import dataclasses
        outcome = run(eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)
        with self.assertRaises(api.VerdictTampering):
            dataclasses.replace(outcome.verdict, status=api.STATUS_FAIL)

    def test_compute_verdict_is_the_aggregator(self):
        verdict = api.compute_verdict(
            tier="T1", gates=gates_ok(),
            void_scan=api.VoidScan((), api.VOID_REASONS, ()),
            search_grade=api.SearchGradeResult(True, (), (), (), ()),
            anchor=anchor(), effect=effect())
        self.assertEqual(verdict.status, api.STATUS_PASS)
        self.assertTrue(verdict.speed_rank_admissible)
        self.assertTrue(verdict.derivation)

    def test_no_gates_at_all_is_not_silently_a_pass_at_dispatch(self):
        """An unwired tier raises rather than deriving PASS from an empty gate list."""
        disp = api.TierDispatcher(gate_runners={"T1": _Runner("T1", gates_ok())})
        with self.assertRaises(api.EvaluatorNotWired):
            disp.dispatch(request(tier="T2"), window(), effect=effect())


# ===========================================================================
# 2. An anchor-less comparison is INVALID — never "correct", never "coherent"
# ===========================================================================

class AnchorRequiredTest(unittest.TestCase):

    def test_no_anchor_yields_invalid_with_the_void_reason(self):
        outcome = run(req=request(anchor=None),
                      win=window(anchor_at_open=None, anchor_at_close=None),
                      eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertIn(api.VOID_ANCHOR_MISSING_OR_MUTATED, outcome.void_scan.reasons())

    def test_no_anchor_demotes_a_coherence_PASS_to_COULD_NOT_CHECK(self):
        """The exact `kernel_eval.sh` defect: `COH="coherent"` with no baseline."""
        outcome = run(req=request(anchor=None),
                      win=window(anchor_at_open=None, anchor_at_close=None),
                      eff=effect())
        coh = [g for g in outcome.verdict.gates
               if g.gate_id == "output_coherence_vs_anchor"][0]
        self.assertEqual(coh.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("absence of a comparison is not evidence of equivalence",
                      " ".join(coh.check.reasons))
        self.assertIn("PASS demoted to COULD_NOT_CHECK: no anchor bound", coh.notes)

    def test_non_anchor_gates_are_not_demoted(self):
        outcome = run(req=request(anchor=None),
                      win=window(anchor_at_open=None, anchor_at_close=None),
                      eff=effect())
        trace = [g for g in outcome.verdict.gates
                 if g.gate_id == "no_fallback_dispatch_trace"][0]
        self.assertEqual(trace.check.outcome, S.PASS)

    def test_no_anchor_emits_a_record_with_no_anchor_block_and_no_fabricated_digest(self):
        """v3: the void case is JOURNALABLE, and still names no anchor.

        The old behaviour refused to emit at all, which satisfied "never
        fabricate a digest" by giving up the other half of the same protocol
        sentence — "A voided run is journaled as INVALID with its reason".
        """
        outcome = run(req=request(anchor=None),
                      win=window(anchor_at_open=None, anchor_at_close=None),
                      eff=effect())
        self.assertTrue(outcome.emitted)
        self.assertIsNone(outcome.event_blocked_reason)
        self.assertEqual(outcome.event_violations, ())
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])
        self.assertEqual(outcome.event["schema"], S.SCHEMA_EVALUATION_EVENT_V5)
        self.assertEqual(outcome.event["status"], api.STATUS_INVALID)
        # Structurally absent: not null, not a placeholder, not any key at all.
        self.assertNotIn("anchor", outcome.event)
        self.assertIn(f"VOID:{api.VOID_ANCHOR_MISSING_OR_MUTATED}:{S.FAIL}",
                      outcome.event["integrity_flags"])
        # And the anchor checker still reads it for what it is.
        self.assertEqual(S.check_anchor_binding(outcome.event).outcome, S.FAIL)

    def _emit(self, req, win, *, void_scan=None, eff=None):
        """Build the event directly, so the refusals are asserted at their source."""
        preconditions = api.check_preconditions(req, win)
        scan = void_scan if void_scan is not None else api.check_void_conditions(
            req, win, rate_comparison=eff is not None)
        grammar = api.check_record_grammar_complete(request=req, window=win, effect=eff)
        verdict = api.compute_verdict(
            tier=req.tier, gates=gates_ok(), void_scan=scan,
            search_grade=api.evaluate_search_grade(
                request=req, window=win, preconditions=preconditions, effect=eff,
                grammar_complete=grammar),
            anchor=req.anchor, effect=eff)
        return api.build_evaluation_event(
            request=req, window=win, verdict=verdict, effect=eff,
            preconditions=preconditions)

    def test_a_claimed_but_malformed_anchor_still_refuses_to_emit(self):
        """The exemption is for a run with NO anchor, never for a broken one."""
        req = request()
        object.__setattr__(req.anchor, "binary_sha256", "not-a-sha256")
        with self.assertRaises(api.AnchorMissing) as caught:
            self._emit(req, window(), eff=effect())
        self.assertIn("CLAIMED but is malformed", str(caught.exception))

    def test_event_binds_the_t0_suite_seed_into_search_discipline(self):
        event = self._emit(request(suite_seed=4711), window(), eff=effect())
        self.assertEqual(event["performance"]["search_discipline"]["suite_seed"],
                         4711)

    def test_gate_vector_preserves_structured_property_measurements(self):
        measurement = {
            "schema": "epyc.autokernel.property_measurement.v1",
            "shape_id": "SOFT_MAX(type=f32,ne=[83,2,1,1])#0",
            "op": "SOFT_MAX", "backend": "CPU",
            "metric_id": "softmax_invariants/v1", "residual": 2.5e-08,
            "tolerance": 1e-4, "suite_seed": 4711, "passed": True,
        }
        gate = api.GateResult(
            gate_id="t0.backend_op_units", gate_class=api.GATE_CORRECTNESS,
            check=S.Check(S.PASS, ()), measurements=(measurement,))
        vector = api._vector((gate,), api.GATE_CORRECTNESS)
        self.assertEqual(
            vector["t0.backend_op_units"]["measurements"], [measurement])

    def test_a_placeholder_anchor_digest_is_refused_at_emission(self):
        """`0`*64 is not "no anchor recorded"; it is a claim that one WAS."""
        for field, filler in (("binary_sha256", "0" * 64),
                              ("linkage_sha256", "f" * 64),
                              ("source_commit", "0" * 40)):
            with self.subTest(field=field):
                req = request()
                object.__setattr__(req.anchor, field, filler)
                with self.assertRaises(api.AnchorMissing) as caught:
                    self._emit(req, window(), eff=effect())
                self.assertIn("placeholder digests", str(caught.exception))

    def test_an_anchorless_run_that_declares_no_anchor_void_is_refused(self):
        """The omission is admitted by the VOID REASON, never by the absence.

        A verdict that drops the anchor while voiding for something else would
        otherwise emit a record the schema refuses. The emitter fails here, on
        the same fact the validator checks, rather than handing the journal a
        line it cannot append.
        """
        req = request(anchor=None)
        win = window(anchor_at_open=None, anchor_at_close=None)
        elsewhere = api.VoidFinding(
            reason=api.VOID_HOST_HEALTH_TIER_VIOLATION,
            protocol_phrase=api.VOID_REASON_PHRASES[api.VOID_HOST_HEALTH_TIER_VIOLATION],
            outcome=S.FAIL, detail=("uptime 9 days",))
        scan = api.VoidScan(findings=(elsewhere,),
                            evaluated=(api.VOID_HOST_HEALTH_TIER_VIOLATION,),
                            not_applicable=())
        with self.assertRaises(api.AnchorMissing) as caught:
            self._emit(req, win, void_scan=scan)
        self.assertIn("no digest to fabricate", str(caught.exception))

    def test_a_voided_run_is_still_durable(self):
        """'A voided run is journaled as INVALID with its reason, and is never
        silently discarded.'"""
        outcome = run(req=request(anchor=None),
                      win=window(anchor_at_open=None, anchor_at_close=None),
                      eff=effect())
        payload = outcome.durable_payload
        self.assertEqual(payload["verdict"]["status"], api.STATUS_INVALID)
        self.assertEqual(
            [f["reason"] for f in payload["verdict"]["void_findings"]
             if f["reason"] == api.VOID_ANCHOR_MISSING_OR_MUTATED],
            [api.VOID_ANCHOR_MISSING_OR_MUTATED])
        self.assertIn("missing or mutated anchor", payload["void_scan"]["findings"][0]
                      ["protocol_phrase"] if payload["void_scan"]["findings"] else "")
        # It is canonically serialisable, so a caller can hash and journal it.
        self.assertTrue(S.canonical_json(payload))

    def test_a_rebuilt_anchor_is_a_different_anchor(self):
        moved = api.AnchorIdentity(
            source_commit=V8_COMMIT,
            binary_sha256=sha("anchor-binary-REBUILT"),
            linkage_sha256=sha("anchor-linkage"),
        )
        outcome = run(win=window(anchor_at_close=moved), eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertIn(api.VOID_ANCHOR_MISSING_OR_MUTATED, outcome.void_scan.reasons())

    def test_anchor_short_form_is_the_grammar_field(self):
        self.assertEqual(anchor().short(),
                         f"{V8_COMMIT[:12]}/{sha('anchor-binary')[:12]}/"
                         f"{sha('anchor-linkage')[:12]}")

    def test_malformed_anchor_parses_to_reasons_not_a_traceback(self):
        parsed, reasons = api.AnchorIdentity.parse({"source_commit": "deadbeef",
                                                    "binary_sha256": sha("b"),
                                                    "linkage_sha256": sha("l")})
        self.assertIsNone(parsed)
        self.assertTrue(reasons)

    def test_no_anchor_requiring_gate_can_report_PASS_without_an_anchor(self):
        """The class-level guard behind the demotion: not one anchor-requiring gate
        in an anchor-less run may end up PASS, whatever the runner returned."""
        outcome = run(req=request(anchor=None),
                      win=window(anchor_at_open=None, anchor_at_close=None),
                      eff=effect())
        anchored = [g for g in outcome.verdict.gates if g.requires_anchor]
        self.assertEqual(len(anchored), 3)
        for gate in anchored:
            self.assertNotEqual(gate.check.outcome, S.PASS)


# ===========================================================================
# 3. Correctness precedence — a failing candidate gets NO rank, not a bad one
# ===========================================================================

class CorrectnessPrecedenceTest(unittest.TestCase):

    def _failing(self, gate_class=api.GATE_CORRECTNESS, eff=None):
        gates = list(gates_ok())
        gates[0] = api.GateResult("mul_mat_exact_shapes", gate_class,
                                  fail("2 of 4096 shapes diverged"), requires_anchor=True)
        return run(gates=tuple(gates), eff=eff if eff is not None else effect())

    def test_correctness_failure_makes_the_rank_unobtainable(self):
        outcome = self._failing()
        self.assertEqual(outcome.verdict.status, api.STATUS_FAIL)
        self.assertFalse(outcome.verdict.speed_rank_admissible)
        with self.assertRaises(api.SpeedRankUnavailable) as ctx:
            outcome.verdict.rank_key()
        self.assertIn("no speed rank at all — not a penalised one", str(ctx.exception))

    def test_a_fast_but_wrong_candidate_gets_no_rank_however_large_the_effect(self):
        """Control 3, 'degraded-negative': MUST receive no speed rank at all."""
        huge = effect(value=9.99, e_value=100000.0)
        outcome = self._failing(eff=huge)
        self.assertFalse(outcome.verdict.speed_rank_admissible)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()

    def test_every_lexicographically_prior_class_blocks_the_rank(self):
        for gate_class in api.LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES:
            with self.subTest(gate_class=gate_class):
                outcome = self._failing(gate_class=gate_class)
                self.assertEqual(outcome.verdict.status, api.STATUS_FAIL)
                self.assertFalse(outcome.verdict.speed_rank_admissible)

    def test_could_not_check_is_neither_pass_nor_fail(self):
        gates = list(gates_ok())
        gates[3] = api.GateResult("ppl_margin", api.GATE_QUALITY,
                                  cnc("the PPL harness produced no score"))
        outcome = run(gates=tuple(gates), eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INCONCLUSIVE)
        self.assertFalse(outcome.verdict.speed_rank_admissible)
        self.assertIn("QUALITY:ppl_margin:COULD_NOT_CHECK", outcome.verdict.integrity_flags)

    def test_a_failed_mechanism_prediction_is_inconclusive_not_fail(self):
        """Design §9.4: withholds the bonus and makes the result inconclusive."""
        gates = gates_ok() + (
            api.GateResult("memunitstalled_delta", api.GATE_MECHANISM,
                           fail("predicted -30%, measured -1%")),)
        outcome = run(gates=gates, eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INCONCLUSIVE)
        self.assertFalse(outcome.verdict.speed_rank_admissible)

    def test_rank_candidates_reports_what_it_excluded(self):
        good = run(eff=effect()).verdict
        bad = self._failing().verdict
        ranked, unrankable = api.rank_candidates([good, bad])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(len(unrankable), 1)
        self.assertIs(unrankable[0][0], bad)
        self.assertIn("lexicographically prior to speed", unrankable[0][1])

    def test_rank_candidates_orders_best_first(self):
        small = run(eff=effect(value=0.03)).verdict
        large = run(eff=effect(value=0.30)).verdict
        ranked, _ = api.rank_candidates([small, large])
        self.assertEqual([v.effect.value for v in ranked], [0.30, 0.03])

    def test_below_noise_floor_is_not_ranked(self):
        outcome = run(eff=effect(value=0.005))     # phi = 0.009
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)
        self.assertEqual(outcome.verdict.effect_resolution, api.EFFECT_BELOW_NOISE_FLOOR)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()

    def test_below_mde_is_no_detectable_difference_and_still_a_result(self):
        outcome = run(eff=effect(value=0.015))     # phi 0.009 < 0.015 < mde 0.021
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)
        self.assertEqual(outcome.verdict.effect_resolution,
                         api.EFFECT_NO_DETECTABLE_DIFFERENCE)
        self.assertFalse(outcome.verdict.speed_rank_admissible)
        self.assertIsNotNone(outcome.event)
        self.assertEqual(outcome.event_violations, ())

    def test_e_value_below_threshold_is_not_ranked(self):
        outcome = run(eff=effect(e_value=3.0))
        self.assertEqual(outcome.verdict.effect_resolution,
                         api.EFFECT_EVIDENCE_BELOW_THRESHOLD)
        self.assertFalse(outcome.verdict.speed_rank_admissible)

    def test_lcb_is_carried_but_never_decides(self):
        outcome = run(eff=effect(lcb_descriptive=99.0, value=0.005))
        self.assertEqual(outcome.verdict.effect_resolution, api.EFFECT_BELOW_NOISE_FLOOR)
        unc = outcome.event["performance"]["uncertainty"]
        self.assertEqual(unc["lcb_label"], "descriptive")


# ===========================================================================
# 4. Every void condition produces INVALID with its reason
# ===========================================================================

class VoidConditionTest(unittest.TestCase):

    #: One window mutation per protocol void condition. Both FAIL and
    #: COULD_NOT_CHECK void the window; the finding records which it was.
    CASES = {
        api.VOID_CLAIM_NOT_HELD: dict(resource_claim_close=fail("holder pid 8801 -> 9002")),
        api.VOID_HOST_HEALTH_TIER_VIOLATION: dict(host_health=fail("uptime 9 days")),
        api.VOID_ANCHOR_GATE_FAILED: dict(anchor_gate=fail("anchor cell outside band")),
        api.VOID_AA_CONTROL_FAILED: dict(controls=controls(aa=fail("A/A crossed"))),
        api.VOID_EVALUATOR_BUNDLE_UNVERIFIED: dict(
            runtime_source_label=cnc("no source-label attestation was captured")),
        api.VOID_ANCHOR_MISSING_OR_MUTATED: dict(anchor_at_close=None),
        api.VOID_HAND_TYPED_ARGV: dict(recipe=None),
        api.VOID_CONCURRENT_INFERENCE: dict(
            no_concurrent_inference=fail("a foreign llama-server holds the device")),
        api.VOID_STORAGE_EXHAUSTED: dict(storage_close=fail("below storage_floor_bytes_free")),
        api.VOID_STRATA_VIOLATION: dict(strata=fail("block 7 served both strata")),
        api.VOID_POST_HOC_RULE_CHANGE: dict(
            rule_immutability=fail("stopping rule hash moved mid-campaign")),
        api.VOID_INCOMPLETE_CALIBRATION: dict(calibration=fail("solve did not converge")),
    }

    def test_every_void_reason_has_a_case(self):
        self.assertEqual(sorted(self.CASES), sorted(api.VOID_REASONS))

    def test_each_void_condition_yields_invalid_with_its_reason(self):
        for reason, mutation in self.CASES.items():
            with self.subTest(void=reason):
                outcome = run(win=window(**mutation), eff=effect())
                self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
                self.assertIn(reason, outcome.void_scan.reasons())
                finding = [f for f in outcome.void_scan.findings if f.reason == reason][0]
                self.assertEqual(finding.protocol_phrase, api.VOID_REASON_PHRASES[reason])
                self.assertIn(f"VOID:{reason}:{finding.outcome}",
                              outcome.verdict.integrity_flags)

    def test_a_voided_window_is_not_recorded_as_a_candidate_failure(self):
        """'A VOID window is journaled as INVALID; it MUST NOT be recorded as a
        candidate failure, because a drifted anchor says nothing whatever about
        the candidate.'"""
        gates = list(gates_ok())
        gates[0] = api.GateResult("mul_mat_exact_shapes", api.GATE_CORRECTNESS,
                                  fail("diverged"), requires_anchor=True)
        outcome = run(gates=tuple(gates),
                      win=window(anchor_gate=fail("anchor cell outside band")),
                      eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertNotEqual(outcome.verdict.status, api.STATUS_FAIL)
        # The gate failure is not lost, it is simply not the status.
        self.assertIn("CORRECTNESS:mul_mat_exact_shapes:FAIL",
                      outcome.verdict.integrity_flags)

    def test_could_not_check_voids_but_stays_distinguishable_from_fail(self):
        outcome = run(win=window(resource_claim_close=cnc("/proc/locks unreadable")),
                      eff=effect())
        finding = [f for f in outcome.void_scan.findings
                   if f.reason == api.VOID_CLAIM_NOT_HELD][0]
        self.assertEqual(finding.outcome, S.COULD_NOT_CHECK)
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)

    def test_rate_only_voids_are_reported_not_applicable_for_a_non_rate_record(self):
        outcome = run(req=request(tier="T0"), tier="T0", eff=None)
        self.assertEqual(
            sorted(outcome.void_scan.not_applicable),
            sorted([api.VOID_ANCHOR_GATE_FAILED, api.VOID_AA_CONTROL_FAILED,
                    api.VOID_STRATA_VIOLATION, api.VOID_INCOMPLETE_CALIBRATION]))

    def test_a_t0_record_that_reports_a_number_gets_the_full_scan(self):
        """The rate-only exemption keys on the RECORD, not on the tier label."""
        outcome = run(req=request(tier="T0"), tier="T0", eff=effect())
        self.assertEqual(outcome.void_scan.not_applicable, ())


# ===========================================================================
# 5. Preconditions and the search-grade conjunction
# ===========================================================================

class SearchGradeTest(unittest.TestCase):

    def test_healthy_run_is_search_grade(self):
        outcome = run(eff=effect())
        self.assertTrue(outcome.verdict.search_grade.satisfied)
        self.assertEqual(outcome.verdict.search_grade.failed, ())
        self.assertEqual(outcome.verdict.search_grade.not_applicable, ())

    def test_conjuncts_are_the_protocols_fourteen(self):
        self.assertEqual(len(api.SEARCH_GRADE_CONJUNCTS), 14)
        self.assertEqual(
            [c.id for c in api.SEARCH_GRADE_CONJUNCTS][:3],
            ["ratified_protocol", "preconditions", "calibration_block_accepted"])

    def _search_grade(self, *, req=None, win=None, eff=_DEFAULT):
        req = req or request()
        win = win or window()
        pre = api.check_preconditions(req, win)
        eff = effect() if eff is _DEFAULT else eff
        grammar = api.check_record_grammar_complete(request=req, window=win, effect=eff)
        return api.evaluate_search_grade(request=req, window=win, preconditions=pre,
                                         effect=eff, grammar_complete=grammar)

    def test_wrong_protocol_id_fails_conjunct_one_by_name(self):
        sg = self._search_grade(req=request(protocol_id="P-BENCH-1/v1"))
        self.assertIn("ratified_protocol", sg.failed)
        self.assertTrue(sg.reason_for("ratified_protocol"))

    def test_too_few_blocks_fails_the_b_min_conjunct(self):
        sg = self._search_grade(eff=effect(paired_blocks=4))   # B_min = 10
        self.assertIn("b_min_paired_blocks_order_randomized", sg.failed)
        self.assertIn("below the calibrated B_min",
                      " ".join(sg.reason_for("b_min_paired_blocks_order_randomized")))

    def test_blocked_design_fails_the_order_control(self):
        sg = self._search_grade(win=window(order_randomized=fail("candidate x n, then anchor x n")))
        self.assertIn("b_min_paired_blocks_order_randomized", sg.failed)

    def test_uncalibrated_threshold_fails_the_e_value_conjunct(self):
        sg = self._search_grade(eff=effect(threshold=20.0))    # 1/alpha_sel is 100
        self.assertIn("e_value_against_calibrated_threshold", sg.failed)

    def test_confirmation_stratum_uses_the_tighter_threshold(self):
        ok = self._search_grade(eff=effect(stratum=api.STRATUM_CONFIRMATION,
                                           threshold=500.0))  # 1/alpha_conf = 1/0.002
        self.assertNotIn("e_value_against_calibrated_threshold", ok.failed)
        bad = self._search_grade(eff=effect(stratum=api.STRATUM_CONFIRMATION,
                                            threshold=100.0))
        self.assertIn("e_value_against_calibrated_threshold", bad.failed)

    def test_calibration_for_another_cell_is_refused(self):
        sg = self._search_grade(req=request(calibration=calibration(phase="prefill")))
        self.assertIn("calibration_block_accepted", sg.failed)
        self.assertIn("MUST NOT be reused",
                      " ".join(sg.reason_for("calibration_block_accepted")))

    def test_missing_conjunct_makes_the_record_invalid(self):
        outcome = run(eff=effect(paired_blocks=4))
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertIn("SEARCH_GRADE_MISSING:b_min_paired_blocks_order_randomized",
                      outcome.verdict.integrity_flags)

    def test_non_rate_record_marks_the_statistical_conjuncts_not_applicable(self):
        sg = self._search_grade(eff=None)
        self.assertEqual(len(sg.not_applicable), 10)
        self.assertIn("calibration_block_accepted", sg.not_applicable)
        self.assertIn("complete_record_grammar", sg.evaluated)

    def test_preconditions_are_the_protocols_eight(self):
        pre = api.check_preconditions(request(), window())
        self.assertEqual(tuple(pid for pid, _ in pre.checks), api.PRECONDITION_IDS)
        self.assertTrue(pre.satisfied)

    def test_undeclared_campaign_controls_fail_precondition_eight(self):
        pre = api.check_preconditions(request(campaign_controls=None), window())
        self.assertIn("declared_campaign_controls", pre.unsatisfied)
        self.assertFalse(pre.satisfied)

    def test_campaign_controls_refuse_a_zero_or_unbounded_quantity(self):
        with self.assertRaises(ValueError):
            campaign_controls(max_candidates=0)
        with self.assertRaises(ValueError):
            campaign_controls(contribution_floor=float("inf"))
        parsed, reasons = api.CampaignControls.parse({"max_candidates": 10})
        self.assertIsNone(parsed)
        self.assertTrue(reasons)

    def test_alpha_sel_may_not_exceed_one_over_max_candidates(self):
        cal = calibration(alpha_sel=0.5, alpha_conf=0.05)
        chk = cal.check_against_controls(campaign_controls())
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertIn("false selections", " ".join(chk.reasons))

    def test_calibration_refuses_a_looser_confirmation_budget(self):
        with self.assertRaises(ValueError):
            calibration(alpha_sel=0.01, alpha_conf=0.05)

    def test_calibration_must_record_the_normative_solve_order(self):
        with self.assertRaises(ValueError) as ctx:
            calibration(solve_order_recorded=("phi_estimated_from_aa_control",))
        self.assertIn("normative solve order", str(ctx.exception))

    def test_b_min_above_the_declared_ceiling_fails_calibration(self):
        chk = calibration(b_min_blocks=99).check_against_controls(campaign_controls())
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertIn("the campaign does not start", " ".join(chk.reasons))

    def test_calibration_without_controls_is_could_not_check(self):
        self.assertEqual(calibration().check_against_controls(None).outcome,
                         S.COULD_NOT_CHECK)


# ===========================================================================
# 6. Controls — four mandatory, plus the accept-side control's declared contract
# ===========================================================================

class ControlPanelTest(unittest.TestCase):

    def test_unavailable_control_five_needs_a_reason_and_an_escalation(self):
        with self.assertRaises(ValueError) as ctx:
            api.ControlPanel(positive=PASS, neutral=PASS, degraded_negative=PASS,
                             aa=PASS, historical_replay=None)
        self.assertIn("HISTORICAL_REPLAY_UNAVAILABLE", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            api.ControlPanel(positive=PASS, neutral=PASS, degraded_negative=PASS,
                             aa=PASS, historical_replay=None,
                             historical_replay_unavailable_reason="no durable win on llama_gpu")
        self.assertIn("operator", str(ctx.exception))

    def test_declared_unavailable_control_five_is_search_grade_and_marked(self):
        panel = api.ControlPanel(
            positive=PASS, neutral=PASS, degraded_negative=PASS, aa=PASS,
            historical_replay=None,
            historical_replay_unavailable_reason="no qualifying durable win for llama_gpu",
            operator_escalation_ref="ake-escalation-0002")
        self.assertEqual(panel.marker(), "4/5 (HISTORICAL_REPLAY_UNAVAILABLE)")
        self.assertEqual(panel.check_5().outcome, S.PASS)
        outcome = run(win=window(controls=panel), eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)
        self.assertIn("controls=4/5 (HISTORICAL_REPLAY_UNAVAILABLE)", outcome.grammar_line)

    def test_a_failing_historical_replay_is_a_gate_defect(self):
        panel = controls(historical_replay=fail("the iqk port did not promote"))
        chk = panel.check_5()
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertIn("GATE DEFECT", " ".join(chk.reasons))
        outcome = run(win=window(controls=panel), eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertIn("SEARCH_GRADE_MISSING:control_5_passing_or_recorded_unavailable",
                      outcome.verdict.integrity_flags)

    def test_a_failing_control_1_to_4_blocks_ranking(self):
        outcome = run(win=window(controls=controls(positive=fail("known win did not rank"))),
                      eff=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertFalse(outcome.verdict.speed_rank_admissible)


# ===========================================================================
# 7. Emitted records validate against schemas.py
# ===========================================================================

class EventEmissionTest(unittest.TestCase):

    def test_a_passing_t1_event_validates(self):
        outcome = run(eff=effect())
        self.assertIsNotNone(outcome.event)
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])
        self.assertEqual(outcome.event_violations, ())
        self.assertEqual(outcome.event["status"], "pass")
        self.assertEqual(outcome.event["claim_grammar"]["category"], "CANDIDATE")
        self.assertEqual(outcome.event["integrity_flags"], [])

    def test_a_failing_event_validates_and_carries_its_flags(self):
        gates = list(gates_ok())
        gates[0] = api.GateResult("mul_mat_exact_shapes", api.GATE_CORRECTNESS,
                                  fail("diverged"), requires_anchor=True)
        outcome = run(gates=tuple(gates), eff=effect())
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])
        self.assertEqual(outcome.event["status"], "fail")
        self.assertTrue(outcome.event["integrity_flags"])

    def test_an_invalid_event_validates(self):
        outcome = run(win=window(host_health=fail("uptime 9 days")), eff=effect())
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])
        self.assertEqual(outcome.event["status"], "invalid")

    def test_a_t0_event_validates_without_anchor_measurement_ids(self):
        req = request(tier="T0",
                      anchor=api.AnchorIdentity(source_commit=V8_COMMIT,
                                                binary_sha256=sha("anchor-binary"),
                                                linkage_sha256=sha("anchor-linkage")))
        outcome = run(req=req, tier="T0", eff=None)
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])
        self.assertEqual(outcome.event["anchor"]["measurement_event_ids"], [])
        self.assertIsNone(outcome.event["performance"]["estimate"])

    def test_the_record_carries_the_attestation_triple_not_an_attest_field(self):
        outcome = run(eff=effect())
        ref = outcome.event["claim_grammar"]["attestation_ref"]
        self.assertTrue(ref.startswith("res="))
        self.assertIn(";host=", ref)
        self.assertIn(";srclabel=", ref)
        self.assertNotIn("attest ", outcome.grammar_line)

    def test_the_record_carries_the_resource_claim_receipt(self):
        outcome = run(eff=effect())
        self.assertEqual(outcome.event["resource_claim_receipt"],
                         "gpu_device.mi210_0:claim-20260803T1200Z-8801")

    def test_the_record_carries_the_scope_denominator(self):
        outcome = run(eff=effect())
        self.assertEqual(outcome.event["scope_denominator"],
                         {"machine_subset": "partial", "numa_nodes": [],
                          "devices": ["mi210_0"], "cores": 8})
        chk = S.check_scope_denominator_admits_gate(
            outcome.event, {"machine_subset": "full", "cores": 96})
        self.assertEqual(chk.outcome, S.FAIL)

    def test_anchor_binding_check_passes_on_the_emitted_record(self):
        outcome = run(eff=effect())
        anchor_binary = outcome.event["anchor"]["binary_sha256"]
        resolved = {"artifact": {"binary_sha256": anchor_binary}}
        chk = S.check_anchor_binding(outcome.event, lambda _eid: resolved)
        self.assertEqual(chk.outcome, S.PASS)

    def test_metric_commensurability_holds_for_the_backend(self):
        outcome = run(eff=effect())
        chk = S.check_metric_commensurability("llama_gpu", outcome.event["claim_grammar"])
        self.assertEqual(chk.outcome, S.PASS)

    def test_the_record_carries_no_authority_flavoured_key(self):
        outcome = run(eff=effect())
        self.assertEqual(S.find_authority_flavoured_keys(outcome.event), [])
        self.assertEqual(S.find_authority_flavoured_keys(outcome.durable_payload), [])

    def test_the_estimate_is_reproducible_from_raw_samples(self):
        outcome = run(eff=effect())
        self.assertTrue(outcome.event["performance"]["raw_samples"])
        self.assertTrue(outcome.event["performance"]["raw_samples_ref"])
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])

    def test_an_estimate_without_raw_samples_cannot_be_constructed(self):
        with self.assertRaises(ValueError) as ctx:
            effect(raw_samples=())
        self.assertIn("self-reported score", str(ctx.exception))

    def test_the_content_hash_is_stable(self):
        a = run(eff=effect())
        b = run(eff=effect())
        self.assertEqual(a.record_content_hash, b.record_content_hash)
        self.assertEqual(a.record_content_hash, S.content_hash(a.event))

    def test_correctness_is_a_per_case_vector_not_a_rolled_up_verdict(self):
        outcome = run(eff=effect())
        self.assertIn("mul_mat_exact_shapes", outcome.event["correctness"])
        self.assertIn("output_coherence_vs_anchor", outcome.event["correctness"])
        self.assertEqual(outcome.event["correctness"]["mul_mat_exact_shapes"]["outcome"],
                         "PASS")


class ResourceClaimReceiptSeamTest(unittest.TestCase):
    """The receipt in the record must resolve back to a real device claim.

    `schemas.validate_evaluation_event` can only check that
    `resource_claim_receipt` is a non-empty string, and a non-empty string is
    exactly what an INVENTED receipt also is. This asserts the seam end to end:
    the id the evaluator writes is the id `claim_witness.resolve_claim_receipt`
    looks up in `device_claim`'s claim journal.
    """

    CLAIM_ID = "gpu_device.mi210_0:claim-20260803T1200Z-8801"

    def _stub_journal(self, campaign_id):
        receipt = DC.ClaimReceipt(
            claim_id=self.CLAIM_ID,
            device_id="mi210_0",
            lock_path="/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock",
            state=DC.STATE_HELD,
            holder_pid=8801,
            holder_start_ticks=123456,
            holder_boot_id="boot-abcdef",
            host="epyc",
            purpose="ak3-t1-search",
            campaign_id=campaign_id,
            acquired_at=NOW,
            holder_label="autokernel-evaluator",
        )

        class _Journal:
            def read_all(self):
                return [{"kind": DC.KIND_ACQUIRED,
                         "detail": {"claim_id": receipt.claim_id,
                                    "receipt": receipt.to_dict()}}]

        return _Journal()

    def test_the_emitted_receipt_resolves_to_the_claim_that_produced_it(self):
        outcome = run(eff=effect())
        chk = CW.check_event_claim_receipt(
            outcome.event, self._stub_journal(outcome.event["campaign_id"]))
        self.assertEqual(chk.outcome, S.PASS)

    def test_an_invented_receipt_is_FAIL_not_PASS(self):
        outcome = run(win=window(resource_claim_receipt="made-up"), eff=effect())
        chk = CW.check_event_claim_receipt(
            outcome.event, self._stub_journal(outcome.event["campaign_id"]))
        self.assertEqual(chk.outcome, S.FAIL)

    def test_a_receipt_from_another_campaign_is_FAIL(self):
        outcome = run(eff=effect())
        chk = CW.check_event_claim_receipt(outcome.event,
                                           self._stub_journal("ak-some-other-campaign"))
        self.assertEqual(chk.outcome, S.FAIL)


# ===========================================================================
# 8. Record grammar
# ===========================================================================

class RecordGrammarTest(unittest.TestCase):

    def test_the_line_matches_the_protocol_template(self):
        outcome = run(eff=effect())
        line = outcome.grammar_line
        self.assertTrue(line.startswith("decode_tokens_per_s 0.061 higher-better, tier T1, "
                                        "vs anchor "))
        self.assertIn("— SEARCH RECORD, NOT A CLAIM [P-AK-SEARCH-1, category=CANDIDATE,", line)
        for token in ("blocks=12", "e=140", "thr=100", "MDE=0.021", "floor=0.009",
                      "stratum=selection", "det=bitwise_stable", "scope=partial/devmi210_0/8c",
                      "controls=5/5", "campaign=ak-llama_gpu-decode-20260803",
                      "eval=", "srclabel=ake-srclabel-0003",
                      "recipe=ak.microbench.llama_gpu.decode/v1@",
                      "res=gpu_device.mi210_0:", "host=host-health-", "raw=", "2026-08-03]"):
            self.assertIn(token, line)

    def test_the_line_never_calls_itself_a_claim(self):
        self.assertIn(api.RECORD_CLASS, run(eff=effect()).grammar_line)
        self.assertEqual(api.RECORD_CLASS, "SEARCH RECORD, NOT A CLAIM")

    def test_a_non_rate_record_states_n_a_rather_than_dropping_fields(self):
        outcome = run(req=request(tier="T0"), tier="T0", eff=None)
        for token in ("blocks=n/a", "e=n/a", "thr=n/a", "MDE=n/a", "floor=n/a",
                      "stratum=n/a"):
            self.assertIn(token, outcome.grammar_line)

    def test_grammar_completeness_fails_and_names_the_missing_field(self):
        chk = api.check_record_grammar_complete(
            request=request(), window=window(recipe=None), effect=effect())
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertIn("recipe: no recipe-constructor identity was recorded",
                      " ".join(chk.reasons))

    def test_grammar_completeness_states_the_field_set_it_applied(self):
        chk = api.check_record_grammar_complete(
            request=request(), window=window(), effect=effect())
        self.assertEqual(chk.outcome, S.PASS)
        self.assertIn("grammar fields required for this record", chk.reasons[0])

    def test_an_anchorless_line_says_NO_ANCHOR(self):
        outcome = run(req=request(anchor=None),
                      win=window(anchor_at_open=None, anchor_at_close=None),
                      eff=effect())
        self.assertIn("vs anchor NO-ANCHOR", outcome.grammar_line)


# ===========================================================================
# 9. Tier dispatch and the T3 refusal
# ===========================================================================

class TierDispatchTest(unittest.TestCase):

    def test_t3_is_refused_by_admit_tier(self):
        for tier in api.RELEASE_TIERS:
            with self.subTest(tier=tier):
                with self.assertRaises(api.TierNotOwned) as ctx:
                    api.admit_tier(tier)
                self.assertIn("release", str(ctx.exception))
                self.assertIn(api.RELEASE_TIER_OWNER, str(ctx.exception))

    def test_t3_is_refused_at_wiring_time(self):
        with self.assertRaises(api.TierNotOwned):
            api.TierDispatcher(gate_runners={"T3": _Runner("T3", gates_ok())})

    def test_t3_is_refused_at_dispatch_time(self):
        disp = dispatcher()
        with self.assertRaises(api.TierNotOwned):
            disp.dispatch(request(tier="T3"), window(), effect=effect())

    def test_an_unknown_tier_is_refused(self):
        with self.assertRaises(api.TierNotOwned):
            api.admit_tier("T9")

    def test_the_release_seam_exists_and_is_never_called_from_here(self):
        self.assertTrue(hasattr(api.ReleaseTierEvaluator, "evaluate_release"))
        source = Path(api.__file__).read_text(encoding="utf-8")
        self.assertNotIn("evaluate_release(", source.split("class ReleaseTierEvaluator")[0])

    def test_every_search_tier_dispatches(self):
        for tier in api.SEARCH_TIERS:
            with self.subTest(tier=tier):
                outcome = run(req=request(tier=tier), tier=tier, eff=effect())
                self.assertEqual(outcome.verdict.tier, tier)

    def test_the_state_trail_is_recorded_in_order(self):
        outcome = run(eff=effect())
        self.assertEqual(outcome.states,
                         ("CREATED", "TIER_ADMITTED", "WINDOW_OPENED",
                          "PRECONDITIONS_CHECKED", "ANCHOR_BOUND", "GATES_RUN",
                          "WINDOW_CLOSED", "VERDICT_COMPUTED", "EMITTED"))
        self.assertEqual(outcome.durable_payload["dispatch_states"], list(outcome.states))

    def test_a_voided_run_still_walks_the_whole_state_machine(self):
        outcome = run(win=window(host_health=fail("uptime 9 days")), eff=effect())
        self.assertEqual(outcome.states[-1], "EMITTED")
        self.assertNotIn("VOID", outcome.states)

    def test_an_illegal_transition_raises(self):
        states = ["CREATED"]
        with self.assertRaises(api.StateMachineViolation):
            api.TierDispatcher._advance(states, "EMITTED")

    def test_the_runner_is_called_exactly_once(self):
        runner = _Runner("T1", gates_ok())
        disp = api.TierDispatcher(gate_runners={"T1": runner})
        disp.dispatch(request(), window(), effect=effect())
        self.assertEqual(runner.calls, 1)

    def test_a_runner_returning_the_wrong_type_is_a_wiring_defect(self):
        class Bad:
            tier = "T1"

            def run_gates(self, req):
                return "everything is fine"

        disp = api.TierDispatcher(gate_runners={"T1": Bad()})
        with self.assertRaises(TypeError):
            disp.dispatch(request(), window(), effect=effect())

    def test_a_runner_without_run_gates_is_refused_at_wiring_time(self):
        with self.assertRaises(TypeError):
            api.TierDispatcher(gate_runners={"T1": object()})


# ===========================================================================
# 10. The evaluator never modifies candidate source or production state
# ===========================================================================

class NoWritePathTest(unittest.TestCase):

    def test_the_module_cannot_write_or_signal(self):
        """Design §5.4: the trusted runner 'has no authority to modify candidate
        source or production state'. Proved from the module's own AST."""
        chk = api.audit_no_write_or_process_paths()
        self.assertEqual(chk.outcome, S.PASS, f"write/process paths found: {chk.reasons}")

    def test_the_audit_actually_detects_a_write_path(self):
        """A fixture that cannot fail proves nothing. This one fails on purpose."""
        chk = api.audit_no_write_or_process_paths(
            "import os\n\n\ndef go(p):\n    os.remove(p)\n")
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertEqual(len(chk.reasons), 2)   # the import AND the call

    def test_the_audit_detects_a_process_launch(self):
        chk = api.audit_no_write_or_process_paths(
            "import subprocess\n\n\ndef go():\n    subprocess.check_output(['ls'])\n")
        self.assertEqual(chk.outcome, S.FAIL)

    def test_the_audit_detects_an_open_for_write(self):
        chk = api.audit_no_write_or_process_paths("def go(p):\n    open(p, 'w')\n")
        self.assertEqual(chk.outcome, S.FAIL)

    def test_unparsable_source_is_could_not_check_not_pass(self):
        chk = api.audit_no_write_or_process_paths("def (:\n")
        self.assertEqual(chk.outcome, S.COULD_NOT_CHECK)

    # -- the audit is bound to something, in both of its two readings ---------

    def test_source_with_nothing_in_it_is_not_a_clean_bill_of_health(self):
        """RED TEAM: `audit_no_write_or_process_paths("")` returned PASS.

        The audit is a SEARCH for forbidden constructs, so over source containing
        no constructs it found none and certified it. That is the guarantee
        obtained by deleting the thing under inspection, and `controls.py`,
        `readiness.py` and both speech adapters each hand this engine a string —
        any of them could have recorded a PASS about nothing.

        Break it by deleting the `_is_an_audited_module` branch.
        """
        for label, src in (("empty", ""), ("whitespace", "   \n\n"),
                           ("comment only", "# nothing here\n"),
                           ("docstring only", '"""a module that was deleted"""\n'),
                           ("assignment only", "MODULE_ID = 'whatever'\n")):
            with self.subTest(source=label):
                chk = api.audit_no_write_or_process_paths(src)
                self.assertEqual(chk.outcome, S.COULD_NOT_CHECK, label)
                self.assertIn("nothing to audit", chk.reasons[0])

    def test_COMPLIANT_a_real_module_is_still_audited_and_passes(self):
        """The engine must not forbid its own idiom: reuse over supplied source
        is why `controls.py` and `correctness.py` do not copy the denylists."""
        from autokernel.evaluator import devices as D
        text = Path(D.__file__).read_text(encoding="utf-8")
        self.assertEqual(api.audit_no_write_or_process_paths(
            text, module_id=D.MODULE_ID).outcome, S.PASS)

    def test_supplied_clean_source_requires_and_checks_foreign_module_identity(self):
        from autokernel.evaluator import devices as D
        text = Path(D.__file__).read_text(encoding="utf-8")
        self.assertEqual(api.audit_no_write_or_process_paths(text).outcome,
                         S.COULD_NOT_CHECK)
        self.assertEqual(api.audit_no_write_or_process_paths(
            text, module_id=api.MODULE_ID).outcome, S.COULD_NOT_CHECK)

    def test_a_finding_is_still_returned_over_a_one_line_snippet(self):
        """A FAIL is a finding about the TEXT and is returned unbound, so the
        module-shape test must sit after it, not before."""
        chk = api.audit_no_write_or_process_paths("import subprocess\n")
        self.assertEqual(chk.outcome, S.FAIL)

    def test_the_self_audit_proves_it_read_this_module(self):
        """RED TEAM: the no-argument call trusted `Path(__file__).read_text()`.

        `_defines_this_module` checks it instead. Break it by deleting the
        `own and not _defines_this_module(tree)` branch, then point `__file__` at
        another file: the audit would report PASS about a module it never read.
        """
        from autokernel.evaluator import devices as D
        self.assertEqual(api.audit_no_write_or_process_paths().outcome, S.PASS)
        # `devices.py` is clean, so before the binding this reported PASS — about
        # a module the audit had not read.
        with mock.patch.object(api, "__file__", D.__file__):
            chk = api.audit_no_write_or_process_paths()
            self.assertEqual(chk.outcome, S.COULD_NOT_CHECK)
            self.assertIn(api.MODULE_ID, chk.reasons[0])

    def test_dispatch_leaves_the_inputs_untouched(self):
        req, win = request(), window()
        before = (S.canonical_json(req.artifact.to_dict()),
                  S.canonical_json(req.anchor.to_dict()),
                  win.resource_claim_receipt)
        run(req=req, win=win, eff=effect())
        after = (S.canonical_json(req.artifact.to_dict()),
                 S.canonical_json(req.anchor.to_dict()),
                 win.resource_claim_receipt)
        self.assertEqual(before, after)

    def test_the_typed_inputs_are_immutable(self):
        req = request()
        for target, attr in ((req, "tier"), (req.artifact, "binary_sha256"),
                             (window(), "host_receipt"), (effect(), "value")):
            with self.subTest(attr=attr):
                with self.assertRaises(Exception):
                    setattr(target, attr, "mutated")


# ===========================================================================
# 11. Input contracts refuse rather than degrade
# ===========================================================================

class InputContractTest(unittest.TestCase):

    def test_a_bool_cannot_stand_in_for_a_check(self):
        with self.assertRaises(TypeError) as ctx:
            api.GateResult("g", api.GATE_CORRECTNESS, True)
        self.assertIn("third outcome", str(ctx.exception))

    def test_an_unknown_gate_class_is_refused(self):
        with self.assertRaises(ValueError):
            api.GateResult("g", "vibes", PASS)

    def test_an_unknown_void_reason_is_refused(self):
        with self.assertRaises(ValueError):
            api.VoidFinding("SOMETHING_ELSE", "phrase", S.FAIL)

    def test_a_void_finding_cannot_be_a_pass(self):
        with self.assertRaises(ValueError):
            api.VoidFinding(api.VOID_HAND_TYPED_ARGV, "phrase", S.PASS)

    def test_a_mutable_evaluator_id_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            api.EvaluatorIdentity(id="P-AK-SEARCH-1", bundle_sha256=sha("b"),
                                  runtime_source_label_ref="r")
        self.assertIn("/vN", str(ctx.exception))

    def test_a_missing_runtime_source_label_is_refused(self):
        with self.assertRaises(ValueError):
            api.EvaluatorIdentity(id="P-AK-SEARCH-1/v1", bundle_sha256=sha("b"),
                                  runtime_source_label_ref="")

    def test_a_partial_scope_must_name_what_it_measured(self):
        with self.assertRaises(ValueError):
            api.ScopeDenominator(machine_subset="partial", numa_nodes=(), devices=(),
                                 cores=48)

    def test_a_determinism_class_needs_repeats(self):
        with self.assertRaises(ValueError):
            api.DeterminismReport(determinism_class="bitwise_stable",
                                  same_seed_repeat_runs=0)
        self.assertEqual(
            api.DeterminismReport("not_measured", 0).to_dict()["class"], "not_measured")

    def test_zero_reps_is_not_a_measurement(self):
        with self.assertRaises(ValueError):
            request(reps=0)

    def test_a_precondition_scan_must_cover_all_eight(self):
        with self.assertRaises(ValueError):
            api.PreconditionScan(checks=(("host_health_tier", PASS),))

    def test_the_verdict_status_vocabulary_is_a_subset_of_the_record_vocabulary(self):
        for status in api.VERDICT_STATUSES:
            self.assertIn(status, S.EVENT_STATUSES)


# ===========================================================================
# 12. The anchor triple names ONE tool
#
# `binary_sha256` is single-valued and one anchor build ships several binaries:
# T0 hashes the anchor `llama-cli`, `microbench` compares the plan's digest
# against the anchor `llama-bench` it is about to spawn. The rule this section
# holds down is "`binary_sha256` is the digest of the tool the record's `metric`
# was measured with, and `tool` names it" — ENFORCED, which means the case that
# used to be invisible (two tools, one capture, digests therefore equal) is a
# FAIL and not a PASS.
# ===========================================================================

def tool_anchor(tool=None, **overrides) -> api.AnchorIdentity:
    """One fixed triple, optionally naming a tool. The DIGESTS never vary here.

    Deliberate: every test below that varies the tool holds the three digests
    constant, because a differing digest would make the comparison fail for the
    old reason and prove nothing about the new field.
    """
    fields = dict(source_commit=V8_COMMIT, binary_sha256=sha("anchor-binary"),
                  linkage_sha256=sha("anchor-linkage"), tool=tool)
    fields.update(overrides)
    return api.AnchorIdentity(**fields)


class AnchorNamesOneToolTest(unittest.TestCase):

    def test_one_capture_two_tools_is_a_FAIL_not_a_silent_match(self):
        """The defect. Same bytes cannot be two different binaries."""
        check = tool_anchor("llama-cli").identity_matches(tool_anchor("llama-bench"))
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        joined = " ".join(check.reasons)
        self.assertIn("anchor.tool differs", joined)
        self.assertIn("llama-cli", joined)
        self.assertIn("llama-bench", joined)

    def test_the_same_tool_still_matches(self):
        """Compliant-path control: the rule must not forbid its own idiom."""
        self.assertEqual(
            tool_anchor("llama-bench").identity_matches(tool_anchor("llama-bench")).outcome,
            S.PASS)

    def test_two_unnamed_anchors_compare_exactly_as_before(self):
        """Backward compatibility is a control too: records predating the field."""
        self.assertEqual(tool_anchor().identity_matches(tool_anchor()).outcome, S.PASS)

    def test_named_against_unnamed_is_could_not_check_never_pass(self):
        """Not naming a tool is not evidence that it is the same tool."""
        for mine, theirs in (("llama-bench", None), (None, "llama-bench")):
            with self.subTest(mine=mine, theirs=theirs):
                check = tool_anchor(mine).identity_matches(tool_anchor(theirs))
                self.assertEqual(check.outcome, S.COULD_NOT_CHECK, check.reasons)
                self.assertIn("one side names its tool", " ".join(check.reasons))

    def test_a_digest_difference_outranks_an_unobserved_tool(self):
        """`state_machine.check_anchor_identity`'s rule: a fact beats an absence."""
        check = tool_anchor("llama-bench").identity_matches(
            tool_anchor(None, binary_sha256=sha("some other binary")))
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        joined = " ".join(check.reasons)
        self.assertIn("anchor.binary_sha256 moved", joined)
        self.assertIn("one side names its tool", joined)

    def test_a_triple_cannot_be_relabelled_as_another_tool(self):
        cli = tool_anchor("llama-cli")
        with self.assertRaises(ValueError) as ctx:
            cli.for_tool("llama-bench")
        self.assertIn("already bound", str(ctx.exception))
        self.assertIs(cli.for_tool("llama-cli"), cli)
        self.assertEqual(tool_anchor().for_tool("llama-bench").tool, "llama-bench")
        self.assertEqual(tool_anchor().for_tool(" llama-bench ").tool, "llama-bench")

    def test_a_tool_is_a_name_and_not_a_path_or_a_blank(self):
        for bad in ("/mnt/raid0/llm/llama.cpp/build/bin/llama-bench", "llama bench",
                    "llama-bench\n", "", "   "):
            with self.subTest(tool=bad):
                with self.assertRaises(ValueError):
                    tool_anchor(bad)

    def test_the_tool_survives_a_round_trip_and_absence_has_one_spelling(self):
        named = tool_anchor("llama-bench")
        parsed, reasons = api.AnchorIdentity.parse(named.to_dict())
        self.assertEqual(reasons, ())
        self.assertEqual(parsed, named)
        self.assertEqual(named.to_dict()["tool"], "llama-bench")
        # Unnamed omits the key entirely: a block written before the field existed
        # and one written by a caller that named no tool are the same bytes.
        self.assertNotIn("tool", tool_anchor().to_dict())
        self.assertEqual(api.AnchorIdentity.parse(tool_anchor().to_dict())[0], tool_anchor())

    def test_the_record_grammar_says_which_binary_the_denominator_came_from(self):
        self.assertEqual(tool_anchor("llama-bench").short(),
                         "llama-bench:" + tool_anchor().short())
        outcome = run(req=request(anchor=tool_anchor("llama-bench",
                                                     measurement_event_ids=("ake-a-1",))),
                      eff=effect())
        self.assertIn("vs anchor llama-bench:", outcome.grammar_line)

    def test_the_emitted_record_carries_the_tool_and_still_validates(self):
        outcome = run(req=request(anchor=tool_anchor("llama-bench",
                                                     measurement_event_ids=("ake-a-1",))),
                      eff=effect())
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])
        self.assertEqual(outcome.event["anchor"]["tool"], "llama-bench")

    def test_the_rule_reaches_the_verdict_through_the_anchor_precondition(self):
        """End to end: a record measured with one tool, attested with another.

        The window's open/close re-verification is where a campaign that bound
        T0's `llama-cli` capture into a T1 record gets caught, and this is the
        pair that used to be indistinguishable from a clean re-verification —
        every digest agrees, because both sides came from the same capture.
        """
        bench, cli = tool_anchor("llama-bench"), tool_anchor("llama-cli")
        mismatched = api.check_preconditions(
            request(anchor=bench), window(anchor_at_open=cli, anchor_at_close=cli))
        self.assertFalse(mismatched.satisfied)
        self.assertIn("anchor.tool differs", " ".join(
            r for _, chk in mismatched.checks for r in chk.reasons))
        # Compliant-path control: the same window, attested with the tool the
        # record's metric was measured with, is satisfied.
        self.assertTrue(api.check_preconditions(
            request(anchor=bench), window(anchor_at_open=bench, anchor_at_close=bench)
        ).satisfied)


class CombineIsTheOneLatticeTest(unittest.TestCase):
    """`api._combine` delegates to `schemas.Check.worst_of` — including the empty case.

    This module's own `check_gate_derivation_is_locked` names the defect in
    prose — "an empty gate list derives to PASS and that is a fail-open verdict"
    — while `_combine()` did exactly that.
    """

    def test_combining_nothing_is_could_not_check_and_never_pass(self):
        combined = api._combine()
        self.assertEqual(combined.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(combined.passed)

    def test_every_non_pass_reason_is_prefixed_with_the_outcome_that_raised_it(self):
        combined = api._combine(S.Check(S.COULD_NOT_CHECK, ("claim file unreadable",)),
                                S.Check(S.FAIL, ("holder changed mid-window",)))
        self.assertEqual(combined.outcome, S.FAIL)
        self.assertEqual(combined.reasons,
                         ("[COULD_NOT_CHECK] claim file unreadable",
                          "[FAIL] holder changed mid-window"))

    def test_a_non_check_argument_raises_rather_than_being_reduced_around(self):
        with self.assertRaises(TypeError):
            api._combine(S.Check(S.PASS), "PASS")

    def test_the_delegation_is_real_and_not_a_reimplementation(self):
        """Same inputs, same answer as the classmethod, for all three outcomes."""
        for vector in ([], [S.Check(S.PASS)],
                       [S.Check(S.PASS), S.Check(S.COULD_NOT_CHECK, ("x",))],
                       [S.Check(S.FAIL, ("y",)), S.Check(S.COULD_NOT_CHECK, ("x",))]):
            with self.subTest(vector=[c.outcome for c in vector]):
                self.assertEqual(api._combine(*vector), S.Check.worst_of(vector))


if __name__ == "__main__":
    unittest.main()
