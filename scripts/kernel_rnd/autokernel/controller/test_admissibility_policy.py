#!/usr/bin/env python3
"""AK-ADM-1: typed pre-model admissibility enforcement."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest

from scripts.kernel_rnd.autokernel import hypothesis_portfolio, schemas
from scripts.kernel_rnd.autokernel.controller import admissibility_policy as A
from scripts.kernel_rnd.autokernel.controller import discovery_controller as C


def _evidence() -> dict[str, str]:
    portfolio = hypothesis_portfolio.load(hypothesis_portfolio.DEFAULT_PORTFOLIO)
    return {row["evidence_id"]: row["sha256"] for row in portfolio.body["evidence"]}


class ManifestTests(unittest.TestCase):
    def test_checked_in_manifest_binds_sealed_portfolio_evidence(self):
        manifest = A.load(portfolio_evidence=_evidence())
        self.assertEqual(A.POLICY_IDS, {row["policy_id"] for row in manifest.policies})
        self.assertFalse(manifest.iq_serving_resident)

    def test_tampered_manifest_is_refused(self):
        body = json.loads(A.DEFAULT_POLICY_PATH.read_text())
        bad = copy.deepcopy(body); bad["iq_serving_resident"] = True
        with self.assertRaisesRegex(A.AdmissibilityError, "self-hash"):
            A.validate(bad)
        bad = copy.deepcopy(body); bad["admissibility_policies"].pop()
        bad["manifest_sha256"] = schemas.content_hash(
            {k: v for k, v in bad.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(A.AdmissibilityError, "six"):
            A.validate(bad)
        with self.assertRaisesRegex(A.AdmissibilityError, "sealed portfolio"):
            A.validate(body, portfolio_evidence={})


class EnforcementTests(unittest.TestCase):
    manifest = A.load()

    def ids(self, *, statement="tighten loop", facts=None, symbol="kernel",
            files=("ggml/src/ggml-cuda/x.cu",), patch="", hypothesis="akh-v2-x"):
        proposal = {} if facts is None else {"admissibility_facts": facts}
        return {r.policy_id for r in A.enforce(
            self.manifest, hypothesis_id=hypothesis, statement=statement, proposal=proposal,
            target_symbol=symbol, declared_files=files, patch_text=patch)}

    def test_neutral_candidate_is_admitted(self):
        self.assertEqual(set(), self.ids())

    def test_exact_instantiation(self):
        self.assertEqual({A.EXACT_INSTANTIATION}, self.ids(statement="cut VGPR to 63"))
        self.assertEqual({A.EXACT_INSTANTIATION}, self.ids(facts={"occupancy_evidence": {
            "instantiation": "iq2_xxs<false,false> synthetic",
            "production_instantiation": "iq2_xxs<true,false>"}}))
        self.assertEqual(set(), self.ids(statement="cut VGPR", facts={"occupancy_evidence": {
            "instantiation": "q5<true,false>", "production_instantiation": "q5<true,false>"}}))

    def test_command_phase(self):
        self.assertEqual({A.COMMAND_PHASE}, self.ids(facts={"phase_claim": {
            "phase": "decode", "command_argv": ["llama-bench", "-n", "0"]}}))
        self.assertEqual(set(), self.ids(facts={"phase_claim": {
            "phase": "decode", "command_argv": ["llama-bench", "-n", "128"]}}))

    def test_static_tool_silence(self):
        self.assertEqual({A.STATIC_TOOL}, self.ids(facts={"static_scans": [{
            "tool": "llvm-nm", "tool_resolved": False, "argv": ["llvm-nm", "x.so"],
            "exit_code": 127, "stderr_suppressed": True, "negative": True}]}))
        self.assertEqual(set(), self.ids(facts={"static_scans": [{
            "tool": "llvm-readelf", "tool_resolved": True, "argv": ["llvm-readelf", "-s", "x.so"],
            "exit_code": 0, "stderr_suppressed": False, "negative": True}]}))

    def test_q4k_mmq_baseline(self):
        kwargs = dict(symbol="mul_mat_q", files=("ggml/src/ggml-cuda/mmq.cuh",),
                      hypothesis="akh-v2-q4k-mmq-dequant-gemv")
        self.assertEqual({A.Q4K_BASELINE}, self.ids(**kwargs))
        self.assertEqual({A.Q4K_BASELINE}, self.ids(facts={"q4k_mmq_baseline": {
            "forced_mmq_pass": 18, "forced_mmq_total": 43}}, **kwargs))
        self.assertEqual(set(), self.ids(facts={"q4k_mmq_baseline": {
            "forced_mmq_pass": 43, "forced_mmq_total": 43}}, **kwargs))

    def test_q8_native_premise(self):
        self.assertEqual({A.Q8_PREMISE}, self.ids(symbol="vec_dot_q8_0_q8_1"))
        self.assertEqual(set(), self.ids(symbol="vec_dot_q8_0_q8_1",
                                         facts={"q8_premise": "integer_native_acknowledged"}))

    def test_iq_residency_trigger(self):
        self.assertEqual({A.IQ_RESIDENCY}, self.ids(symbol="vec_dot_iq2_xxs_q8_1"))
        self.assertEqual({A.IQ_RESIDENCY}, self.ids(symbol="vec_dot_iq2_xxs_q8_1",
                                                    facts={"iq_claim": "live_payoff"}))
        self.assertEqual(set(), self.ids(symbol="vec_dot_iq2_xxs_q8_1",
                                         facts={"iq_claim": "latent_diagnostic"}))


class ControllerSeamTests(unittest.TestCase):
    def candidate(self, *, symbol: str, facts=None):
        proposal = {} if facts is None else {"admissibility_facts": facts}
        return SimpleNamespace(
            hypothesis_id="akh-v2-q8-quantizer-new-mechanism", statement="reuse q8_1",
            proposal=proposal, experiment_intent=SimpleNamespace(target_symbol=symbol),
            source_manifest=SimpleNamespace(declared_files=("ggml/src/ggml-cuda/vecdotq.cuh",),
                                            patch_text=""))

    def context(self):
        return {"planner_context": A.load().context_projection()}

    def test_refuses_before_critic(self):
        with self.assertRaises(C.AdmissibilityRefused) as caught:
            C.enforce_admissibility_before_model_judgment(
                self.candidate(symbol="vec_dot_q8_0_q8_1"), context=self.context(),
                require_manifest=True)
        self.assertEqual([A.Q8_PREMISE], [r.policy_id for r in caught.exception.refusals])

    def test_admits_and_requires_bound_manifest(self):
        ok = self.candidate(symbol="vec_dot_q8_0_q8_1",
                            facts={"q8_premise": "integer_native_acknowledged"})
        self.assertIsNotNone(C.enforce_admissibility_before_model_judgment(
            ok, context=self.context(), require_manifest=True))
        with self.assertRaisesRegex(C.DiscoveryControllerError, "lacks the bound"):
            C.enforce_admissibility_before_model_judgment(
                ok, context={"planner_context": {}}, require_manifest=True)
        unbound = self.context(); unbound["planner_context"]["admissibility_policy_sha256"] = "0" * 64
        with self.assertRaisesRegex(C.DiscoveryControllerError, "does not bind"):
            C.enforce_admissibility_before_model_judgment(
                ok, context=unbound, require_manifest=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
