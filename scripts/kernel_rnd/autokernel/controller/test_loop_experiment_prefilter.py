from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest

from . import authoring_contract as A
from . import loop_experiment_prefilter as P
from . import loop_experiment_runner as R
from . import loop_experiments as L


def sha(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def role(name: str, seconds: int) -> L.RoleBudget:
    instruction = f"Run fixed {name}."
    return L.RoleBudget(name, seconds, instruction, sha(instruction))


def context() -> A.PricedContext:
    return A.price_context(
        round_id="ak-le-prefilter-round",
        budget=A.ContextBudget(max_total_tokens=64, max_item_tokens=64, max_items=1),
        items=(A.ContextItem(
            source_ref="profile://fixed", purpose="fixed retrieval",
            content="fixed counter context"),))


def hypothesis(mechanism: str) -> dict[str, str]:
    return {
        "mechanism": mechanism,
        "target_surface": "Q4_K decode",
        "falsifiable_counter": "VALU per wave falls",
        "predicted_direction": "lower",
    }


def fingerprint(mechanism: str) -> str:
    return L.HypothesisObservation(
        **hypothesis(mechanism), survived_prefilter=False).fingerprint


def experiment(ctx: A.PricedContext, pin: L.ArtifactPin,
               prior: tuple[str, ...]) -> L.ExperimentContract:
    propose = "PROPOSE one falsifiable hypothesis."
    selected = "Implement the independently selected hypothesis."
    planners = tuple(L.PlannerArm(
        f"plan-model-{effort}-{mode}", "model-a", "native", effort,
        L.TARGET_ABSENT if mode == "control" else L.TARGET_RENDERED)
        for effort in ("high", "xhigh") for mode in ("control", "target"))
    scaffolds = tuple(arm for model in ("model-a", "model-b") for arm in (
        L.ScaffoldArm(
            f"scaffold-{model}-direct", model, "native", "high",
            L.SCAFFOLD_DIRECT, (role("implement", 20),)),
        L.ScaffoldArm(
            f"scaffold-{model}-split", model, "native", "high",
            L.SCAFFOLD_SPLIT, (role("implement", 10), role("exploit", 10))),
    ))
    return L.ExperimentContract(
        "ak-le-prefilter-fixture-v1",
        L.FixedPromptFrame(
            L.ArtifactPin("champion://fixed", sha("champion")),
            L.context_sha256(ctx), propose, sha(propose),
            L.SelectedTaskArtifact("hypothesis://fixed", selected, sha(selected))),
        planners,
        (L.DirectionPrediction(
            "model-a", "native", "higher_effort_increases_search_persistence",
            "Predeclared fixture direction."),),
        scaffolds, prior, pin)


class FixtureRunner:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses

    def __call__(self, argv, cwd, environment, prompt, timeout, result_path):
        result = json.dumps({
            "schema": R.RAW_OBSERVATION_SCHEMA,
            "termination": "budget_exhausted",
            "hypotheses": self.hypotheses,
        })
        if result_path is not None:
            result_path.write_text(result, encoding="utf-8")
        return R.ProcessCapture(
            tuple(argv), 0, result if result_path is None else "event", "", result,
            False, "2026-08-12T00:00:00+00:00",
            "2026-08-12T00:00:01+00:00", 1.0)


class PrefilterFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin = self.root / "planner"
        self.bin.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        self.bin.chmod(self.bin.stat().st_mode | stat.S_IXUSR)
        self.ctx = context()
        self.prior = (fingerprint("prior hypothesis"),)
        self.filter = P.compile_prefilter_contract(
            prior_hypothesis_sha256=self.prior)
        self.filter_path = self.root / "prefilter.json"
        P._atomic_canonical(self.filter_path, self.filter)
        pin = L.ArtifactPin(str(self.filter_path), sha(self.filter_path.read_bytes()))
        self.spec = experiment(self.ctx, pin, self.prior)
        executable_sha = sha(self.bin.read_bytes())
        self.pins = tuple(R.ModelCellPin(
            "codex", arm.model_id, arm.quant_id, arm.effort,
            str(self.bin), executable_sha) for arm in self.spec.planner_arms[::2])
        targets = {
            arm.cell_id: "PROXIMATE AUTHORING TARGET: fixed target"
            for arm in self.spec.planner_arms
            if arm.target_context_mode == L.TARGET_RENDERED}
        self.targets = targets
        self.manifest = P.compile_bound_planner_manifest(
            self.spec, prefilter_contract_path=self.filter_path,
            context=self.ctx, target_lines=targets,
            model_pins=self.pins, timeout_seconds=10)
        self.panel_root = self.root / "panel"
        self.panel = R.run_planner_manifest(
            self.manifest, output_root=self.panel_root,
            runner=FixtureRunner((
                hypothesis("novel hypothesis"),
                hypothesis(" Novel   Hypothesis "),
                hypothesis("prior hypothesis"),
            )))

    def tearDown(self):
        self.temp.cleanup()


class StructuralPrefilterTest(PrefilterFixture):
    def test_duplicate_prior_and_novel_decisions_are_deterministic(self):
        reduced = P.reduce_planner_panel(
            manifest=self.manifest, panel=self.panel,
            prefilter_contract=self.filter)
        first = reduced["prefilter_evidence"][0]
        self.assertEqual([row["decision"] for row in first["decisions"]], [
            "survived_structural_prefilter",
            "rejected_duplicate_in_cell",
            "rejected_exact_prior",
        ])
        receipt = reduced["planner_receipt"]
        self.assertEqual(receipt["schema"], L.PLANNER_RECEIPT_SCHEMA)
        self.assertEqual(receipt["scope"], "ak-le-1-2-planner-only")
        self.assertEqual(receipt["scaffold_throughput_observations"],
                         "absent_not_fabricated")
        self.assertEqual(receipt["search_persistence_observations"][0][
            "prefilter_survival_count"], 1)
        self.assertFalse(receipt["constraints"]["ranking_authority"])

    def test_panel_or_sealed_observation_tamper_refuses(self):
        tampered = json.loads(json.dumps(self.panel))
        tampered["observations"][0]["observation"]["termination"] = "search_exhausted"
        with self.assertRaisesRegex(P.PrefilterError, "panel SHA"):
            P.reduce_planner_panel(
                manifest=self.manifest, panel=tampered,
                prefilter_contract=self.filter)
        sealed = next(self.panel_root.glob("0001-*/observation.json"))
        sealed.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(P.PrefilterError, "SHA-256 drifted"):
            P.verify_panel_evidence_files(self.panel, self.panel_root / "panel.json")

    def test_duplicate_cell_or_wrong_prior_refuses(self):
        duplicate = json.loads(json.dumps(self.panel))
        duplicate["observations"][1]["cell_id"] = duplicate["observations"][0]["cell_id"]
        body = {key: value for key, value in duplicate.items() if key != "panel_sha256"}
        duplicate["panel_sha256"] = P._digest(body)
        with self.assertRaisesRegex(P.PrefilterError, "cell order"):
            P.reduce_planner_panel(
                manifest=self.manifest, panel=duplicate,
                prefilter_contract=self.filter)
        wrong = P.compile_prefilter_contract(prior_hypothesis_sha256=())
        with self.assertRaisesRegex(P.PrefilterError, "prior set differs"):
            P.reduce_planner_panel(
                manifest=self.manifest, panel=self.panel,
                prefilter_contract=wrong)

    def test_source_only_or_unpinned_prefilter_refuses(self):
        source_pin = replace(
            self.spec,
            prefilter=L.ArtifactPin(
                "git://repo/controller/do_not_repeat.py", sha("source only")))
        manifest = R.compile_planner_manifest(
            source_pin, context=self.ctx,
            target_lines={
                arm.cell_id: "PROXIMATE AUTHORING TARGET: fixed target"
                for arm in source_pin.planner_arms
                if arm.target_context_mode == L.TARGET_RENDERED},
            model_pins=self.pins, timeout_seconds=10)
        panel = R.run_planner_manifest(
            manifest, output_root=self.root / "source-only",
            runner=FixtureRunner((hypothesis("novel hypothesis"),)))
        with self.assertRaisesRegex(P.PrefilterError, "does not pin"):
            P.reduce_planner_panel(
                manifest=manifest, panel=panel, prefilter_contract=self.filter)

        refusal = P.refuse_under_specified_panel(manifest=manifest, panel=panel)
        self.assertEqual(refusal["schema"], P.REFUSAL_SCHEMA)
        self.assertEqual(refusal["status"], "refused")
        self.assertEqual(refusal["raw_hypothesis_count"], 4)
        self.assertEqual({row["code"] for row in refusal["reasons"]}, {
            "runnable_prefilter_contract_not_pinned",
            "prefilter_algorithm_and_inputs_not_bound",
            "pinned_source_not_invocable_from_raw_observation",
        })
        self.assertFalse(refusal["constraints"]["planner_receipt_emitted"])
        self.assertIn("does not invoke, weaken, or replace",
                      refusal["next_run_requirement"])

    def test_historical_contract_validates_but_current_execution_refuses_drift(self):
        historical = json.loads(json.dumps(self.filter))
        historical["producer"]["sha256"] = sha("historical producer")
        body = {key: value for key, value in historical.items()
                if key != "contract_sha256"}
        historical["contract_sha256"] = P._digest(body)
        self.assertEqual(P.validate_prefilter_contract(
            historical, expected_prior_hypothesis_sha256=self.prior), historical)
        with self.assertRaisesRegex(P.PrefilterError, "current reducer source"):
            P.reduce_planner_panel(
                manifest=self.manifest, panel=self.panel,
                prefilter_contract=historical)

    def test_bound_compiler_requires_independently_persisted_exact_bytes(self):
        wrong = replace(
            self.spec, prefilter=L.ArtifactPin(str(self.filter_path), sha("wrong")))
        with self.assertRaisesRegex(P.PrefilterError, "does not pin"):
            P.compile_bound_planner_manifest(
                wrong, prefilter_contract_path=self.filter_path,
                context=self.ctx, target_lines=self.targets,
                model_pins=self.pins, timeout_seconds=10)

    def test_contract_and_reduction_outputs_are_atomic_and_new(self):
        contract_path = self.root / "future-contract.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(P.main([
                "compile", "--output", str(contract_path),
                "--prior-hypothesis-sha256", self.prior[0],
            ]), 0)
        self.assertEqual(sha(contract_path.read_bytes()),
                         P._digest(json.loads(contract_path.read_text())))
        with self.assertRaisesRegex(P.PrefilterError, "new absolute"):
            P.main(["compile", "--output", str(contract_path)])

        output = self.root / "reduction.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(P.main([
                "reduce", "--manifest", str(self.panel_root / "manifest.json"),
                "--panel", str(self.panel_root / "panel.json"),
                "--prefilter-contract", str(self.filter_path),
                "--output", str(output),
            ]), 0)
        self.assertEqual(json.loads(output.read_text())["schema"], P.REDUCTION_SCHEMA)
        with self.assertRaisesRegex(P.PrefilterError, "new absolute"):
            P.main([
                "reduce", "--manifest", str(self.panel_root / "manifest.json"),
                "--panel", str(self.panel_root / "panel.json"),
                "--prefilter-contract", str(self.filter_path),
                "--output", str(output),
            ])


if __name__ == "__main__":
    unittest.main()
