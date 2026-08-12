from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import unittest

from . import authoring_contract as A
from . import loop_experiments as L


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def priced_context(content: str = "profile says unpack control dominates") -> A.PricedContext:
    return A.price_context(
        round_id="ak-le-fixed-round",
        budget=A.ContextBudget(max_total_tokens=128, max_item_tokens=128, max_items=2),
        items=(A.ContextItem(
            source_ref="profile://fixed", purpose="fixed retrieval", content=content),),
    )


def role(role_name: str, seconds: int, instruction: str) -> L.RoleBudget:
    return L.RoleBudget(role_name, seconds, instruction, sha(instruction))


def contract(context: A.PricedContext | None = None) -> L.ExperimentContract:
    context = context or priced_context()
    prompt = "PROPOSE one falsifiable kernel hypothesis; do not implement it yet."
    planners = tuple(
        L.PlannerArm(
            f"plan-{model}-{effort}-{target}", model, quant, effort,
            L.TARGET_ABSENT if target == "control" else L.TARGET_RENDERED)
        for model, quant in (("model-a", "q8"), ("model-b", "provider-native"))
        for effort in ("medium", "high")
        for target in ("control", "target")
    )
    predictions = tuple(
        L.DirectionPrediction(
            model, quant, "higher_effort_increases_search_persistence",
            "The predeclared prediction is that extra effort delays premature search stop.")
        for model, quant in (("model-a", "q8"), ("model-b", "provider-native"))
    )
    scaffolds = []
    for model, quant in (("model-a", "q8"), ("model-b", "provider-native")):
        scaffolds.extend((
            L.ScaffoldArm(
                f"scaffold-{model}-direct", model, quant, "high", L.SCAFFOLD_DIRECT,
                (role("implement", 120, "Implement the selected hypothesis."),)),
            L.ScaffoldArm(
                f"scaffold-{model}-split", model, quant, "high", L.SCAFFOLD_SPLIT,
                (role("implement", 60, "Implement the selected hypothesis."),
                 role("exploit", 60, "Exploit remaining measured headroom."))),
        ))
    return L.ExperimentContract(
        experiment_id="ak-le-fixture-v1",
        fixed=L.FixedPromptFrame(
            champion=L.ArtifactPin("champion://fixed", sha("champion")),
            retrieval_context_sha256=L.context_sha256(context),
            propose_prompt=prompt, propose_prompt_sha256=sha(prompt)),
        planner_arms=planners,
        predictions=predictions,
        scaffold_arms=tuple(scaffolds),
        prior_hypothesis_sha256=(),
        prefilter=L.ArtifactPin("prefilter://v1", sha("prefilter")),
    )


def hypothesis(*, survived: bool = True, mechanism: str = "branchless unpack") \
        -> L.HypothesisObservation:
    return L.HypothesisObservation(
        mechanism=mechanism,
        target_surface="Q4_K decode",
        falsifiable_counter="VALU instructions per wave decreases",
        predicted_direction="lower",
        survived_prefilter=survived,
    )


def planner_observations(spec: L.ExperimentContract):
    return tuple(
        L.PlannerObservation(
            arm.cell_id, "already_optimized" if index == 0 else "search_exhausted",
            (hypothesis(),), 10.0, sha(f"planner-{arm.cell_id}"))
        for index, arm in enumerate(spec.planner_arms)
    )


def scaffold_observations(spec: L.ExperimentContract):
    rows = []
    for arm in spec.scaffold_arms:
        observations = tuple(
            L.RoleObservation(
                stage.role, stage.wall_seconds / 2, 2, 1,
                sha(f"{arm.cell_id}-{stage.role}"))
            for stage in arm.roles)
        rows.append(L.ScaffoldObservation(arm.cell_id, observations))
    return tuple(rows)


def keys(value):
    if isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys(child)


class TestImmutablePredeclaration(unittest.TestCase):
    def test_contract_is_factorial_hash_bound_and_authority_free(self):
        spec = contract()
        manifest = spec.to_manifest()
        self.assertEqual(manifest["authority"], L.AUTHORITY)
        self.assertEqual(len(manifest["planner_arms"]), 8)
        self.assertEqual(len(manifest["scaffold_arms"]), 4)
        self.assertFalse(manifest["constraints"]["campaign_1_authority"])
        self.assertFalse(manifest["constraints"]["ranking_authority"])
        self.assertNotIn("proximate_target", set(keys(manifest)))
        self.assertEqual(manifest["contract_sha256"], spec.to_manifest()["contract_sha256"])
        with self.assertRaises(FrozenInstanceError):
            spec.experiment_id = "rewrite"

    def test_effort_target_and_scaffold_factors_cannot_be_confounded(self):
        good = contract()
        with self.assertRaisesRegex(L.LoopExperimentError, "two effort"):
            replace(good, planner_arms=tuple(
                arm for arm in good.planner_arms if arm.effort == "high"))
        with self.assertRaisesRegex(L.LoopExperimentError, "absent/rendered"):
            replace(good, planner_arms=tuple(
                arm for arm in good.planner_arms
                if arm.target_context_mode == L.TARGET_ABSENT))
        with self.assertRaisesRegex(L.LoopExperimentError, "independently estimable"):
            replace(good, scaffold_arms=tuple(
                arm for arm in good.scaffold_arms if arm.model_id == "model-a"))
        same_model = tuple(replace(
            arm, cell_id=arm.cell_id.replace("model-b", "model-a-q2"),
            model_id="model-a", quant_id="q4")
            if arm.model_id == "model-b" else arm for arm in good.scaffold_arms)
        with self.assertRaisesRegex(L.LoopExperimentError, "at least two models"):
            replace(good, scaffold_arms=same_model)
        with self.assertRaisesRegex(L.LoopExperimentError, "same wall-time"):
            bad = replace(
                good.scaffold_arms[0],
                roles=(role("implement", 121, "Implement the selected hypothesis."),))
            replace(good, scaffold_arms=(bad, *good.scaffold_arms[1:]))

    def test_every_model_quant_sweep_requires_a_direction_prediction(self):
        good = contract()
        with self.assertRaisesRegex(L.LoopExperimentError, "exactly one"):
            replace(good, predictions=good.predictions[:1])


class TestRenderedPromptBoundary(unittest.TestCase):
    def test_target_exists_only_in_the_rendered_planner_context(self):
        context = priced_context()
        spec = contract(context)
        control = next(
            arm for arm in spec.planner_arms
            if arm.target_context_mode == L.TARGET_ABSENT)
        target = next(
            arm for arm in spec.planner_arms
            if (arm.model_quant, arm.effort) == (control.model_quant, control.effort)
            and arm.target_context_mode == L.TARGET_RENDERED)
        control_prompt = L.render_planner_prompt(spec, control.cell_id, context=context)
        target_prompt = L.render_planner_prompt(
            spec, target.cell_id, context=context,
            target_line="PROXIMATE AUTHORING TARGET: recover the next bounded rung")
        self.assertNotIn("PROXIMATE AUTHORING TARGET", control_prompt)
        self.assertIn("PROXIMATE AUTHORING TARGET", target_prompt)
        self.assertNotIn("recover the next bounded rung", str(spec.to_manifest()))

    def test_target_and_context_refusals_are_fail_closed(self):
        context = priced_context()
        spec = contract(context)
        control = next(arm for arm in spec.planner_arms
                       if arm.target_context_mode == L.TARGET_ABSENT)
        target = next(arm for arm in spec.planner_arms
                      if arm.target_context_mode == L.TARGET_RENDERED)
        with self.assertRaisesRegex(L.LoopExperimentError, "control cell"):
            L.render_planner_prompt(
                spec, control.cell_id, context=context,
                target_line="PROXIMATE AUTHORING TARGET: forbidden")
        with self.assertRaisesRegex(L.LoopExperimentError, "one rendered"):
            L.render_planner_prompt(
                spec, target.cell_id, context=context,
                target_line="recover 3 percent")
        with self.assertRaisesRegex(L.LoopExperimentError, "retrieval context"):
            L.render_planner_prompt(
                spec, target.cell_id, context=priced_context("drifted"),
                target_line="PROXIMATE AUTHORING TARGET: bounded")

    def test_implement_then_exploit_uses_reviewed_authoring_seam(self):
        context = priced_context()
        spec = contract(context)
        split = next(arm for arm in spec.scaffold_arms
                     if arm.scaffold == L.SCAFFOLD_SPLIT)
        implement = L.render_scaffold_prompt(
            spec, split.cell_id, "implement", context=context)
        exploit = L.render_scaffold_prompt(
            spec, split.cell_id, "exploit", context=context)
        self.assertIn("AUTOKERNEL AUTHORING ROLE: implement", implement)
        self.assertIn("AUTOKERNEL AUTHORING ROLE: exploit", exploit)
        with self.assertRaisesRegex(L.LoopExperimentError, "not predeclared"):
            direct = next(arm for arm in spec.scaffold_arms
                          if arm.scaffold == L.SCAFFOLD_DIRECT)
            L.render_scaffold_prompt(spec, direct.cell_id, "exploit", context=context)


class TestDeterministicReducers(unittest.TestCase):
    def test_receipt_counts_search_and_wall_time_throughput_without_authority(self):
        spec = contract()
        receipt = L.reduce_receipt(
            spec, planner_observations=planner_observations(spec),
            scaffold_observations=scaffold_observations(spec), capture_mode="fixture")
        first = receipt["search_persistence_observations"][0]
        self.assertEqual(first["novel_nonduplicate_count"], 1)
        self.assertEqual(first["already_optimized_termination_count"], 1)
        self.assertEqual(first["prefilter_survival_count"], 1)
        self.assertGreater(
            receipt["scaffold_throughput_observations"][0][
                "survivors_per_wall_hour"], 0)
        self.assertFalse(receipt["constraints"]["empirical_claim"])
        self.assertFalse(receipt["constraints"]["controller_ab_authority"])
        self.assertEqual(receipt["receipt_sha256"], L.reduce_receipt(
            spec, planner_observations=planner_observations(spec),
            scaffold_observations=scaffold_observations(spec),
            capture_mode="fixture")["receipt_sha256"])

    def test_semantics_preserving_recoding_is_a_duplicate(self):
        spec = contract()
        observations = list(planner_observations(spec))
        observations[0] = replace(
            observations[0], hypotheses=(
                hypothesis(mechanism=" Branchless   Unpack "),
                hypothesis(mechanism="branchless unpack"),
            ))
        receipt = L.reduce_receipt(
            spec, planner_observations=observations,
            scaffold_observations=scaffold_observations(spec), capture_mode="fixture")
        first = receipt["search_persistence_observations"][0]
        self.assertEqual(first["hypotheses_total"], 2)
        self.assertEqual(first["hypotheses_unique"], 1)
        self.assertEqual(first["duplicate_count"], 1)
        self.assertEqual(first["novel_nonduplicate_count"], 1)

    def test_incomplete_overbudget_or_wrong_role_evidence_refuses(self):
        spec = contract()
        with self.assertRaisesRegex(L.LoopExperimentError, "every cell"):
            L.reduce_receipt(
                spec, planner_observations=planner_observations(spec)[:-1],
                scaffold_observations=scaffold_observations(spec),
                capture_mode="fixture")
        rows = list(scaffold_observations(spec))
        first = rows[0]
        rows[0] = replace(first, roles=(replace(
            first.roles[0], elapsed_wall_seconds=121),))
        with self.assertRaisesRegex(L.LoopExperimentError, "exceeded"):
            L.reduce_receipt(
                spec, planner_observations=planner_observations(spec),
                scaffold_observations=rows, capture_mode="fixture")
        split_index = next(index for index, arm in enumerate(spec.scaffold_arms)
                           if arm.scaffold == L.SCAFFOLD_SPLIT)
        rows = list(scaffold_observations(spec))
        rows[split_index] = replace(
            rows[split_index], roles=tuple(reversed(rows[split_index].roles)))
        with self.assertRaisesRegex(L.LoopExperimentError, "role order"):
            L.reduce_receipt(
                spec, planner_observations=planner_observations(spec),
                scaffold_observations=rows, capture_mode="fixture")


if __name__ == "__main__":
    unittest.main()
