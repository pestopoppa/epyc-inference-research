"""No-inference tests for the prospective AK-WM-2 campaign capture."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from . import least_commitment_capture as C
from . import least_commitment_heldout as HOUT
from . import journal as J
from .controller import hypotheses as H
from .test_journal import _candidate, _event
from .test_schemas import _proposal


def proposal() -> dict:
    value = _proposal()
    value.update({
        "proposal_id": "akp-20260812-1001",
        "campaign_id": "ak-iqk-intervention-20260812",
        "campaign_kind": "config", "change_class": "parameter",
    })
    value["target"]["regimes"] = ["prefill"]
    value["provider_reference"]["target_backend"] = "llama_cpu"
    value["change"]["parameter_surface"] = {
        "candidate": {"ggml_iqk": "1"}, "anchor": {"ggml_iqk": "0"}}
    return value


def diagnostic_source(value: dict, *, candidate_frame_id: str) -> dict:
    is_control = value["change"]["parameter_surface"]["candidate"]["ggml_iqk"] \
        == value["change"]["parameter_surface"]["anchor"]["ggml_iqk"]
    cell = {
        "cell_id": "prefill/mul_mat", "demand_weight": 1.0,
        "supported": True, "compatible": True,
        "report_mass": ({"aa_equal": 0.5, "aa_alternate": 0.5}
                        if is_control else {"iqk_on": 0.8, "iqk_off": 0.2}),
        "regret_margin": 0.0 if is_control else 0.5,
    }
    fixtures = value["representation_contract"][
        "semantics_preserving_recoding_fixture_ids"]
    return {
        "schema": C.SOURCE_SCHEMA, "authority": "prospective_observe_only",
        "receipt_id": f"aklc-source-{value['proposal_id']}",
        "proposal_sha256": C.schemas.content_hash(value),
        "representation_frame_sha256": value["representation_contract"][
            "frame_sha256"],
        "candidate_frame_id": candidate_frame_id,
        "do_not_repeat_match_ids": [],
        "quotients": {"canonical": [cell], **{
            fixture_id: [copy.deepcopy(cell)] for fixture_id in fixtures}},
    }


class _NoPriorMatches:
    def matches_for(self, regime, statement):
        return ()


def _frame_factors(value: dict) -> dict:
    return {
        "candidate_ref": "registered:ggml_iqk",
        "backend": "llama_cpu",
        "model_sha256": "a" * 64,
        "cpu_list": "0-95",
        "devices": [],
        "device_names": [],
        "device_index": 0,
        "n_gpu_layers": 99,
        "production_commit": "b" * 40,
        "measurement_commit": "c" * 40,
        "provider_reference": copy.deepcopy(value["provider_reference"]),
        "ggml_iqk": value["change"]["parameter_surface"]["candidate"]["ggml_iqk"],
        "threads": 96,
    }


def _heldout_receipt(value: dict, *, factors: dict, effect: float) -> dict:
    ordinal = value["proposal_id"].split("-")[-1]
    root = Path(tempfile.mkdtemp(prefix="ak-heldout-journal-"))
    campaign_id = f"ak-heldout-decode-{ordinal}"
    proposal_id = f"akp-heldout-{ordinal}"
    candidate_id = f"akc-heldout-{ordinal}"
    evaluation_id = f"ake-heldout-{ordinal}"
    measured_proposal = copy.deepcopy(value)
    measured_proposal.update({
        "proposal_id": proposal_id,
        "campaign_id": campaign_id,
    })
    measured_proposal["target"]["regimes"] = ["decode"]
    measured_proposal["target"]["ops"] = ["mul_mat"]
    book = J.Journal(str(root), campaign_id=campaign_id)
    book.initialize()
    book.append(J.KIND_PROPOSAL_RECORDED, measured_proposal)
    hypothesis_id = f"akh-heldout-{ordinal}"
    tracker = H.HypothesisTracker(
        journal_=book, root=str(root), campaign_id=campaign_id)
    tracker.open_hypothesis(H.Hypothesis(
        hypothesis_id=hypothesis_id,
        statement=measured_proposal["hypothesis"],
        falsifier="The paired decode observation does not resolve the declared effect.",
        origin=H.ORIGIN_CONTROLLER,
        author="least-commitment-heldout-test",
        regime={"recipe_id": "t1b.llama_cpu.llama_bench_decode.v1"},
    ))
    authorization = tracker.authorize_claim(
        hypothesis_id,
        purpose="exercise prospective held-out journal projection",
        authorized_by="least-commitment-heldout-test",
        ledger=_NoPriorMatches(),
    )
    evaluation = _event(f"heldout-{ordinal}")
    evaluation.update({
        "event_id": evaluation_id,
        "campaign_id": campaign_id,
        "candidate_id": candidate_id,
    })
    evaluation["device_state"]["source"] = "rocm-smi"
    evaluation["device_state"]["receipt_ref"] = "heldout-device-state-receipt"
    candidate = _candidate(f"heldout-{ordinal}", status="banked")
    candidate.update({
        "candidate_id": candidate_id,
        "campaign_id": campaign_id,
        "proposal_id": proposal_id,
        "evaluation_event_ids": [evaluation_id],
    })
    book.append(J.KIND_EVALUATION_EVENT, evaluation)
    book.append(J.KIND_CANDIDATE_RECORDED, candidate)
    terminal = book.append(J.KIND_STOP_STATE, {
        "state": "decided",
        "result": {
            "state": "decided", "campaign_id": campaign_id,
            "candidate_id": candidate_id, "executed": True, "ok": True,
            "spec": {
                "recipe_id": "t1b.llama_cpu.llama_bench_decode.v1",
                "hypothesis": {
                    "bound": True, "hypothesis_id": hypothesis_id,
                    "authorization": authorization.to_dict(),
                },
                "proposal": {
                    "schema": measured_proposal["schema"],
                    "proposal_id": proposal_id,
                    "representation_frame_sha256": measured_proposal[
                        "representation_contract"]["frame_sha256"],
                },
                **{key: factors[key] for key in (
                    "candidate_ref", "backend", "model_sha256", "cpu_list",
                    "devices", "device_names", "device_index", "n_gpu_layers",
                    "production_commit", "measurement_commit")},
            },
            "decision": {"keep": effect > 0.03, "median_relative": effect},
            "production_unchanged": {"outcome": C.schemas.PASS},
            "releases": [{"claim": "cpu", "released": True}],
            "pairs": [{"block_index": 0, "candidate": 1.0 + effect,
                       "anchor": 1.0}],
        },
    })
    return HOUT.project(
        receipt_id=f"aklc-heldout-{value['proposal_id']}",
        target_proposal=value,
        measurement={
            "journal_root": str(root), "campaign_id": campaign_id,
            "proposal_id": proposal_id, "completion_event_id": terminal.event_id,
        },
    )


def plan(value: dict, *, role: str = "intervention",
         matched_control: str | None = "akp-20260812-1000",
         factors_override: dict | None = None) -> dict:
    factors = copy.deepcopy(factors_override or _frame_factors(value))
    heldout = _heldout_receipt(
        value, factors=factors,
        effect=0.02 if role == "intervention" else 0.0)
    candidate_frame_id = heldout["candidate_frame_id"]
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    with handle:
        json.dump(diagnostic_source(
            value, candidate_frame_id=candidate_frame_id), handle)
    binding = C.source_binding(Path(handle.name))
    source = json.loads(Path(handle.name).read_text(encoding="utf-8"))
    diagnostics, recodings = C.derive_diagnostics(
        source, proposal=value, candidate_frame_id=candidate_frame_id)
    receipts = {name: dict(binding) for name in C.DIAGNOSTICS}
    heldout_handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False)
    with heldout_handle:
        json.dump(heldout, heldout_handle)
    raw = {
        "schema": C.SCHEMA, "capture_id": "aklc-20260812-1001",
        "campaign_id": value["campaign_id"], "candidate_id": "akc-20260812-1001",
        "matched_experiment_id": "akm-iqk-20260812-0001",
        "proposal_id": value["proposal_id"], "role": role,
        "matched_control_proposal_id": matched_control,
        "candidate_frame_id": candidate_frame_id, "regime": "prefill",
        "surface": "mul_mat", "intervention_id": "ggml-iqk-1",
        "changed_factor": "ggml_iqk",
        "factors": factors,
        "diagnostics": diagnostics,
        "recodings": recodings,
        "diagnostic_source_receipts": receipts,
        "heldout_outcome_receipt": C.source_binding(Path(heldout_handle.name)),
        "outcome_reducers": dict(C.OUTCOME_REDUCERS),
        "capture_mode": "measured",
    }
    raw["plan_sha256"] = C.plan_sha256(raw)
    return raw


class CapturePlanTest(unittest.TestCase):
    def test_measured_decision_produces_every_projector_field(self):
        proposal_record = proposal()
        capture = C.from_mapping(
            plan(proposal_record), proposal=proposal_record,
            campaign_id=proposal_record["campaign_id"],
            candidate_id="akc-20260812-1001")
        block = C.materialize(
            capture,
            decision=SimpleNamespace(
                keep=True, median_relative=0.06, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
        self.assertEqual(block["schema"], C.BLOCK_SCHEMA)
        self.assertEqual(set(block["diagnostics"]), set(C.DIAGNOSTICS))
        self.assertEqual(block["outcome"], {
            "heldout_regime_transfer": 0.02,
            "falsifier_resolution": 0.03,
            "noise_floor": 0.01,
        })
        self.assertFalse(block["falsifier"]["triggered"])

    def test_missing_heldout_input_is_refused(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        raw.pop("heldout_outcome_receipt")
        raw["plan_sha256"] = C.plan_sha256(raw)
        with self.assertRaisesRegex(C.CapturePlanError, "fields must be exactly"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_target_regime_cannot_masquerade_as_heldout(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        binding = raw["heldout_outcome_receipt"]
        path = Path(binding["path"])
        source = json.loads(path.read_text(encoding="utf-8"))
        source["regime"] = "prefill"
        path.write_text(json.dumps(source), encoding="utf-8")
        raw["heldout_outcome_receipt"] = C.source_binding(path)
        raw["plan_sha256"] = C.plan_sha256(raw)
        with self.assertRaisesRegex(C.CapturePlanError, "fresh projection"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_hand_entered_heldout_effect_is_refused(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        path = Path(raw["heldout_outcome_receipt"]["path"])
        source = json.loads(path.read_text(encoding="utf-8"))
        source["relative_effect"] += 0.25
        path.write_text(json.dumps(source), encoding="utf-8")
        raw["heldout_outcome_receipt"] = C.source_binding(path)
        raw["plan_sha256"] = C.plan_sha256(raw)
        with self.assertRaisesRegex(C.CapturePlanError, "fresh projection"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_mutated_completed_measurement_breaks_heldout_projection(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        receipt_path = Path(raw["heldout_outcome_receipt"]["path"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        journal_path = Path(receipt["measurement_record"]["journal_root"]) / \
            "events.jsonl"
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write('{"torn":')
        with self.assertRaisesRegex(C.CapturePlanError, "completed held-out campaign"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_heldout_cli_projects_only_from_completed_journal(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        receipt = json.loads(Path(
            raw["heldout_outcome_receipt"]["path"]).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            proposal_path = root / "proposal-v4.json"
            proposal_path.write_text(json.dumps(proposal_record), encoding="utf-8")
            manifest = root / "heldout-projection.json"
            manifest.write_text(json.dumps({
                "receipt_id": receipt["receipt_id"],
                "target_proposal": str(proposal_path),
                "measurement": {key: receipt["measurement_record"][key]
                                for key in ("journal_root", "campaign_id",
                                            "proposal_id", "completion_event_id")},
            }), encoding="utf-8")
            output = root / "heldout.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(HOUT.main([
                    str(manifest), "--output", str(output)]), 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), receipt)
            with self.assertRaisesRegex(
                    HOUT.HeldoutProjectionError, "new absolute"):
                HOUT.main([str(manifest), "--output", "relative-heldout.json"])

    def test_control_falsifier_is_keep_or_effect_above_noise(self):
        control = C.make_iqk_control_proposal(
            proposal(), campaign_id="ak-iqk-control-20260812",
            proposal_id="akp-20260812-1000")
        raw = plan(control, role="control", matched_control=None)
        capture = C.from_mapping(
            raw, proposal=control, campaign_id=control["campaign_id"],
            candidate_id=raw["candidate_id"])
        kept = C.materialize(
            capture,
            decision=SimpleNamespace(
                keep=True, median_relative=0.001, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
        self.assertTrue(kept["falsifier"]["triggered"])
        self.assertTrue(kept["falsifier"]["predicates"]["decision_triggered"])
        noisy = C.materialize(
            capture,
            decision=SimpleNamespace(
                keep=False, median_relative=-0.02, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
        self.assertTrue(noisy["falsifier"]["triggered"])
        self.assertTrue(noisy["falsifier"]["predicates"]["noise_exceeded"])
        clean = C.materialize(
            capture,
            decision=SimpleNamespace(
                keep=False, median_relative=0.005, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
        self.assertFalse(clean["falsifier"]["triggered"])
        self.assertEqual(clean["outcome"]["falsifier_resolution"], 0.005)

    def test_post_result_diagnostic_mutation_breaks_plan_hash(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        raw["diagnostics"]["k_rho"] += 1.0
        with self.assertRaisesRegex(C.CapturePlanError, "plan_sha256"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_source_bytes_are_resolved_and_mechanically_reduced(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        source_path = Path(next(iter(
            raw["diagnostic_source_receipts"].values()))["path"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["quotients"]["canonical"][0]["compatible"] = False
        source_path.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(C.CapturePlanError, "source SHA-256 differs"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_information_gain_cannot_disagree_with_proposal(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        raw["diagnostics"]["information_gain"] = 0.9
        raw["plan_sha256"] = C.plan_sha256(raw)
        with self.assertRaisesRegex(C.CapturePlanError, "expected_information_gain"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_control_proposal_is_exact_a_a_and_schema_valid(self):
        intervention = proposal()
        control = C.make_iqk_control_proposal(
            intervention, campaign_id="ak-iqk-control-20260812",
            proposal_id="akp-20260812-1000")
        self.assertEqual(control["change"]["parameter_surface"], {
            "candidate": {"ggml_iqk": "0"}, "anchor": {"ggml_iqk": "0"}})
        control_plan = plan(control, role="control", matched_control=None)
        control_plan.update({
            "campaign_id": control["campaign_id"], "proposal_id": control["proposal_id"],
        })
        control_plan["plan_sha256"] = C.plan_sha256(control_plan)
        parsed = C.from_mapping(
            control_plan, proposal=control, campaign_id=control["campaign_id"],
            candidate_id="akc-20260812-1001")
        self.assertEqual(parsed.role, "control")


if __name__ == "__main__":
    unittest.main()
