from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from . import cumulative_composition as C
from . import source_candidate
from .controller import gpu_source_proofs


BASE = "1" * 40
INSTRUMENT = "2" * 40
CAMPAIGN = "ak-composition-test"


def _h(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _manifest(index: int, *, line: int | None = None,
              symbol: str | None = None,
              path: str = "ggml/src/ggml-cuda/composition.cu") \
        -> source_candidate.SourcePatchManifest:
    line = line if line is not None else 10 + index * 10
    symbol = symbol or f"lever_{index}"
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -{line},1 +{line},1 @@ static int {symbol}()\n"
        f"-return {index};\n"
        f"+return {index + 1};\n"
    ).encode()
    return source_candidate.SourcePatchManifest(
        campaign_id=CAMPAIGN, proposal_id=f"akp-composition-{index}",
        candidate_id=f"akc-composition-{index}", source_tree="llama.cpp",
        production_base_commit=BASE, instrument_commit=INSTRUMENT,
        change_class="arithmetic", declared_files=(path,),
        declared_symbols={path: (symbol,)}, mechanism_id=f"mechanism-{index}",
        patch_sha256=hashlib.sha256(patch).hexdigest(), patch_bytes=patch,
    )


def _lever(index: int, **manifest_kwargs) -> C.ReplicatedPositiveLever:
    series = _h(("series", index))
    replications = tuple(
        C.IsolatedReplication(
            result_sha256=_h(("result", index, repetition)),
            series_key=series,
            build_identity_sha256=_h(("build", index)),
            correctness_receipt_sha256=_h(("correct", index, repetition)),
            attribution_receipt_sha256=_h(("attrib", index, repetition)),
            graphs_off_receipt_sha256=_h(("graphs-off", index, repetition)),
            graphs_on_receipt_sha256=_h(("graphs", index, repetition)),
            effect_fraction=.01 + index / 10000,
        )
        for repetition in (1, 2)
    )
    return C.ReplicatedPositiveLever(
        hypothesis_id=f"akh-composition-{index}",
        cross_campaign_candidate_sha256=_h(("cross", index)),
        manifest=_manifest(index, **manifest_kwargs),
        replications=replications,
    )


def _authority(*levers: C.ReplicatedPositiveLever) -> C.CompositionAuthority:
    return C.CompositionAuthority(CAMPAIGN, BASE, INSTRUMENT, tuple(levers))


def _plan(anchor: C.CompositionAuthority, lever: C.ReplicatedPositiveLever,
          attempt: int = 1) -> C.CompositionPlan:
    candidate = anchor.append(lever)
    dnr = C.DnrAuthority.pass_for(
        anchor=anchor, candidate=candidate, registry_sha256=_h("registry"),
        checked_cross_campaign_candidate_sha256s=[
            row.cross_campaign_candidate_sha256 for row in anchor.accepted
        ],
    )
    return C.CompositionPlan.create(
        anchor=anchor, lever=lever, dnr=dnr,
        attempt_id=_h(("attempt", attempt)),
    )


def _identity(label: object) -> gpu_source_proofs.BuildIdentity:
    return gpu_source_proofs.BuildIdentity(
        source_commit=_h(("commit", label))[:40],
        source_sha256=_h(("source", label)),
        binary_sha256=_h(("binary", label)),
        hip_library_sha256=_h(("hip", label)),
        config_sha256=_h(("config", label)),
        linkage_sha256=_h(("linkage", label)),
    )


def _build_pair(plan: C.CompositionPlan) -> C.CumulativeBuildPair:
    return C.CumulativeBuildPair.create(
        plan,
        anchor=C.BuildBinding.create(
            plan.anchor.ordered_patch_set_sha256, _identity((plan.attempt_id, "a")),
            source_materialization_receipt_sha256=
                _h(("materialized", plan.attempt_id, "a"))),
        candidate=C.BuildBinding.create(
            plan.candidate.ordered_patch_set_sha256,
            _identity((plan.attempt_id, "c")),
            source_materialization_receipt_sha256=
                _h(("materialized", plan.attempt_id, "c"))),
    )


def _correctness(pair: C.CumulativeBuildPair, passed: bool = True) \
        -> C.FullCorrectness:
    return C.FullCorrectness.create(
        pair, suite_id="backend-ops-plus-target-runtime-full-v1",
        cases_sha256=_h(("cases", pair.operation_key)),
        receipt_sha256=_h(("correctness-receipt", pair.operation_key, passed)),
        passed=passed,
    )


def _comparison(pair: C.CumulativeBuildPair, correctness: C.FullCorrectness,
                route: float = .01, graphs: float = .01) \
        -> C.IncrementalComparison:
    return C.IncrementalComparison.create(
        pair, correctness,
        exact_route_receipt_sha256=_h(("route", pair.operation_key)),
        graphs_off_receipt_sha256=_h(("graphs-off", pair.operation_key)),
        expected_route_set_sha256=_h(("route-set", pair.operation_key)),
        graphs_on_receipt_sha256=_h(("graphs", pair.operation_key)),
        target_runtime_frame_sha256=_h(("frame", pair.operation_key)),
        exact_route_effect_fraction=route,
        graphs_off_effect_fraction=route,
        graphs_on_effect_fraction=graphs,
    )


class CumulativeCompositionTests(unittest.TestCase):
    def test_ordered_authority_round_trip_and_order_is_identity(self):
        first, second = _lever(1), _lever(2)
        authority = _authority(first, second)
        self.assertEqual(
            C.CompositionAuthority.from_dict(authority.to_dict()), authority)
        self.assertNotEqual(
            authority.ordered_patch_set_sha256,
            _authority(second, first).ordered_patch_set_sha256)
        self.assertEqual(
            [row.manifest_sha256 for row in authority.accepted],
            [first.manifest_sha256, second.manifest_sha256])

    def test_conflicts_and_duplicate_semantics_fail_before_a_plan(self):
        first = _lever(1, line=20, symbol="shared")
        with self.assertRaisesRegex(C.CompositionError, "declared scope"):
            _authority(first).append(_lever(2, line=30, symbol="shared"))
        with self.assertRaisesRegex(C.CompositionError, "coordinates"):
            _authority(first).append(_lever(3, line=20, symbol="other"))
        with self.assertRaisesRegex(C.CompositionError, "already considered"):
            _authority(first).append(first)
        duplicate_cross = replace(
            _lever(4),
            cross_campaign_candidate_sha256=
                first.cross_campaign_candidate_sha256,
        )
        with self.assertRaisesRegex(C.CompositionError, "cross-campaign"):
            _authority(first).append(duplicate_cross)

    def test_replicated_lever_requires_independent_positive_exact_series(self):
        good = _lever(1)
        with self.assertRaisesRegex(C.CompositionError, "at least two"):
            replace(good, replications=good.replications[:1])
        with self.assertRaisesRegex(C.CompositionError, "exact series"):
            replace(good, replications=(
                good.replications[0],
                replace(good.replications[1], series_key=_h("other")),
            ))
        with self.assertRaisesRegex(C.CompositionError, "result was reused"):
            replace(good, replications=(
                good.replications[0],
                replace(good.replications[1],
                        result_sha256=good.replications[0].result_sha256),
            ))
        with self.assertRaisesRegex(C.CompositionError, "changed build identity"):
            replace(good, replications=(
                good.replications[0],
                replace(good.replications[1],
                        build_identity_sha256=_h("different-build")),
            ))
        with self.assertRaisesRegex(C.CompositionError, "must be positive"):
            replace(good.replications[0], effect_fraction=0.0)

    def test_dnr_binds_anchor_candidate_and_cross_campaign_registry(self):
        first, second = _lever(1), _lever(2)
        anchor = _authority(first)
        plan = _plan(anchor, second)
        self.assertEqual(C.CompositionPlan.from_dict(plan.to_dict()), plan)
        wrong = C.DnrAuthority.pass_for(
            anchor=_authority(), candidate=_authority().append(second),
            registry_sha256=_h("registry"),
            checked_cross_campaign_candidate_sha256s=(),
        )
        with self.assertRaisesRegex(C.CompositionError, "does not bind"):
            C.CompositionPlan.create(
                anchor=anchor, lever=second, dnr=wrong,
                attempt_id=_h("wrong-dnr"))
        with self.assertRaisesRegex(C.CompositionError, "DNR authority"):
            C.DnrAuthority.pass_for(
                anchor=anchor, candidate=anchor.append(second),
                registry_sha256=_h("registry"),
                checked_cross_campaign_candidate_sha256s=(
                    first.cross_campaign_candidate_sha256,
                    second.cross_campaign_candidate_sha256,
                ))

    def test_dnr_history_cannot_omit_a_prior_rolled_back_candidate(self):
        first, second = _lever(1), _lever(2)
        initial = _authority()
        first_plan = _plan(initial, first)
        first_pair = _build_pair(first_plan)
        first_correctness = _correctness(first_pair)
        first_comparison = _comparison(
            first_pair, first_correctness, route=-.01, graphs=-.01)
        with tempfile.TemporaryDirectory() as directory:
            ledger = C.CompositionLedger(Path(directory) / "state.json")
            ledger.create(initial)
            ledger.begin(first_plan)
            ledger.record_build_pair(first_pair)
            ledger.record_correctness(first_correctness)
            ledger.record_comparison(first_comparison)
            ledger.finalize(first_plan.operation_key)
            omitted = C.DnrAuthority.pass_for(
                anchor=initial, candidate=initial.append(second),
                registry_sha256=_h("omitted-registry"),
                checked_cross_campaign_candidate_sha256s=())
            second_plan = C.CompositionPlan.create(
                anchor=initial, lever=second, dnr=omitted,
                attempt_id=_h("omitted-attempt"))
            with self.assertRaisesRegex(
                    C.CompositionError, "omits or invents"):
                ledger.begin(second_plan)

    def test_build_pair_binds_exact_anchor_and_candidate_source_stacks(self):
        plan = _plan(_authority(_lever(1)), _lever(2))
        pair = _build_pair(plan)
        pair.bind_plan(plan)
        self.assertEqual(C.CumulativeBuildPair.from_dict(pair.to_dict()), pair)
        wrong = C.CumulativeBuildPair.create(
            plan,
            anchor=C.BuildBinding.create(
                plan.candidate.ordered_patch_set_sha256, _identity("wrong-a"),
                source_materialization_receipt_sha256=_h("wrong-material-a")),
            candidate=C.BuildBinding.create(
                plan.candidate.ordered_patch_set_sha256, _identity("wrong-c"),
                source_materialization_receipt_sha256=_h("wrong-material-c")),
        )
        with self.assertRaisesRegex(C.CompositionError, "source stacks"):
            wrong.bind_plan(plan)

    def test_current_full_correctness_is_mandatory_before_both_comparisons(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        passed = _correctness(pair)
        comparison = _comparison(pair, passed)
        with tempfile.TemporaryDirectory() as directory:
            ledger = C.CompositionLedger(Path(directory) / "state.json")
            ledger.create(plan.anchor)
            ledger.begin(plan)
            with self.assertRaisesRegex(C.CompositionError, "skip correctness"):
                ledger.record_comparison(comparison)
            ledger.record_build_pair(pair)
            with self.assertRaisesRegex(C.CompositionError, "not ready"):
                ledger.finalize(plan.operation_key)
            ledger.record_correctness(passed)
            ledger.record_comparison(comparison)
            final = ledger.finalize(plan.operation_key)
            self.assertEqual(final["terminals"][0]["disposition"], "admitted")

    def test_positive_admission_is_atomic_restart_safe_and_idempotent(self):
        first, second = _lever(1), _lever(2)
        plan = _plan(_authority(first), second)
        pair = _build_pair(plan)
        correct = _correctness(pair)
        comparison = _comparison(pair, correct)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            ledger = C.CompositionLedger(path)
            ledger.create(plan.anchor)
            ledger.begin(plan)
            self.assertEqual(
                C.CompositionLedger(path).load()["pending"]["stage"], "planned")
            ledger.record_build_pair(pair)
            self.assertEqual(
                C.CompositionLedger(path).load()["pending"]["stage"], "built")
            ledger.record_correctness(correct)
            self.assertEqual(
                C.CompositionLedger(path).load()["pending"]["stage"],
                "correctness_passed")
            ledger.record_comparison(comparison)
            final = C.CompositionLedger(path).finalize(plan.operation_key)
            self.assertIsNone(final["pending"])
            self.assertEqual(final["scientific_attempts"], 1)
            self.assertEqual(
                C.CompositionAuthority.from_dict(final["authority"]),
                plan.candidate)
            self.assertEqual(
                C.CompositionLedger(path).create(plan.anchor), final)
            self.assertEqual(
                C.CompositionLedger(path).finalize(plan.operation_key), final)
            terminal = final["terminals"][0]
            self.assertEqual(
                terminal["isolated_result_sha256s"],
                [row.result_sha256 for row in second.replications])
            with self.assertRaisesRegex(C.CompositionError, "already considered"):
                _plan(plan.candidate, second, attempt=2)

    def test_correctness_failure_rolls_back_but_preserves_isolated_science(self):
        first, second = _lever(1), _lever(2)
        plan = _plan(_authority(first), second)
        pair = _build_pair(plan)
        with tempfile.TemporaryDirectory() as directory:
            ledger = C.CompositionLedger(Path(directory) / "state.json")
            ledger.create(plan.anchor)
            ledger.begin(plan)
            ledger.record_build_pair(pair)
            failed = _correctness(pair, passed=False)
            final = ledger.record_correctness(failed)
            self.assertEqual(final["scientific_attempts"], 1)
            self.assertEqual(final["terminals"][0]["disposition"],
                             "correctness_rollback")
            self.assertEqual(
                C.CompositionAuthority.from_dict(final["authority"]), plan.anchor)
            self.assertEqual(
                final["terminals"][0]["isolated_result_sha256s"],
                [row.result_sha256 for row in second.replications])
            self.assertEqual(ledger.record_correctness(failed), final)

    def test_attribution_refusal_is_scientific_and_preserves_anchor(self):
        plan = _plan(_authority(_lever(1)), _lever(2))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        receipt = _h("attribution-refusal")
        with tempfile.TemporaryDirectory() as directory:
            ledger = C.CompositionLedger(Path(directory) / "state.json")
            ledger.create(plan.anchor)
            ledger.begin(plan)
            ledger.record_build_pair(pair)
            ledger.record_correctness(correct)
            final = ledger.rollback_attribution(
                plan.operation_key, receipt_sha256=receipt)
            self.assertEqual(final["scientific_attempts"], 1)
            terminal = final["terminals"][0]
            self.assertEqual(terminal["disposition"], "attribution_rollback")
            self.assertEqual(terminal["attribution_receipt_sha256"], receipt)
            self.assertEqual(
                C.CompositionAuthority.from_dict(final["authority"]),
                plan.anchor)
            self.assertEqual(ledger.rollback_attribution(
                plan.operation_key, receipt_sha256=receipt), final)

    def test_nonpositive_incremental_result_rolls_back_without_losing_anchor(self):
        plan = _plan(_authority(_lever(1)), _lever(2))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        for route, graphs, expected in (
                (-.01, -.01, "screened_out"),
                (.01, -.01, "inconclusive")):
            with self.subTest(route=route, graphs=graphs), \
                    tempfile.TemporaryDirectory() as directory:
                ledger = C.CompositionLedger(Path(directory) / "state.json")
                ledger.create(plan.anchor)
                ledger.begin(plan)
                ledger.record_build_pair(pair)
                ledger.record_correctness(correct)
                comparison = _comparison(pair, correct, route, graphs)
                self.assertEqual(comparison.classification, expected)
                ledger.record_comparison(comparison)
                final = ledger.finalize(plan.operation_key)
                self.assertEqual(final["terminals"][0]["disposition"],
                                 "incremental_rollback")
                self.assertEqual(
                    C.CompositionAuthority.from_dict(final["authority"]),
                    plan.anchor)

    def test_infrastructure_rollback_is_non_scientific_and_fresh_retry_is_allowed(self):
        lever = _lever(1)
        anchor = _authority()
        first = _plan(anchor, lever, attempt=1)
        with tempfile.TemporaryDirectory() as directory:
            ledger = C.CompositionLedger(Path(directory) / "state.json")
            ledger.create(anchor)
            ledger.begin(first)
            rolled = ledger.rollback_infrastructure(
                first.operation_key, reason_code="builder_interrupted",
                receipt_sha256=_h("infra"))
            self.assertEqual(rolled["scientific_attempts"], 0)
            self.assertEqual(rolled["terminals"][0]["disposition"],
                             "infrastructure_rollback")
            self.assertEqual(ledger.rollback_infrastructure(
                first.operation_key, reason_code="builder_interrupted",
                receipt_sha256=_h("infra")), rolled)
            retry = _plan(anchor, lever, attempt=2)
            resumed = ledger.begin(retry)
            self.assertEqual(resumed["pending"]["plan"]["operation_key"],
                             retry.operation_key)

    def test_exact_ten_scientific_terminals_and_no_eleventh(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = C.CompositionLedger(Path(directory) / "state.json")
            authority = _authority()
            ledger.create(authority, max_scientific_attempts=10)
            for index in range(1, 11):
                lever = _lever(index)
                plan = _plan(authority, lever, attempt=index)
                pair = _build_pair(plan)
                correct = _correctness(pair)
                ledger.begin(plan)
                ledger.record_build_pair(pair)
                ledger.record_correctness(correct)
                ledger.record_comparison(_comparison(pair, correct))
                state = ledger.finalize(plan.operation_key)
                authority = plan.candidate
                self.assertEqual(state["scientific_attempts"], index)
            self.assertEqual(len(state["terminals"]), 10)
            with self.assertRaisesRegex(C.CompositionError, "budget is exhausted"):
                ledger.begin(_plan(authority, _lever(11), attempt=11))

    def test_tampered_state_plan_evidence_and_nonfinite_json_fail_closed(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            ledger = C.CompositionLedger(path)
            ledger.create(plan.anchor)
            ledger.begin(plan)
            ledger.record_build_pair(pair)
            original = json.loads(path.read_text())
            mutations = []
            changed_plan = copy.deepcopy(original)
            changed_plan["pending"]["plan"]["candidate_patch_set_sha256"] = "0" * 64
            mutations.append(changed_plan)
            changed_build = copy.deepcopy(original)
            changed_build["pending"]["build_pair"]["candidate"][
                "build_identity_sha256"] = "0" * 64
            mutations.append(changed_build)
            extra = copy.deepcopy(original)
            extra["unexpected"] = True
            mutations.append(extra)
            for mutation in mutations:
                with self.subTest(kind=mutations.index(mutation)):
                    mutation["state_sha256"] = C._sha({
                        key: value for key, value in mutation.items()
                        if key != "state_sha256"})
                    path.write_text(json.dumps(mutation))
                    with self.assertRaises(C.CompositionError):
                        ledger.load()
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(C.CompositionError, "strict JSON"):
                ledger.load()

    def test_resealed_terminal_chain_and_state_aliases_fail_closed(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            ledger = C.CompositionLedger(path)
            ledger.create(plan.anchor)
            ledger.begin(plan)
            ledger.record_build_pair(pair)
            ledger.record_correctness(correct)
            ledger.record_comparison(_comparison(pair, correct))
            ledger.finalize(plan.operation_key)
            original = json.loads(path.read_text())

            mutations = []
            lost_isolated = copy.deepcopy(original)
            lost_isolated["terminals"][0]["isolated_result_sha256s"][0] = _h(
                "invented-isolated-result")
            mutations.append(lost_isolated)
            wrong_authority = copy.deepcopy(original)
            wrong_authority["authority"] = plan.anchor.to_dict()
            mutations.append(wrong_authority)
            wrong_disposition = copy.deepcopy(original)
            wrong_disposition["terminals"][0]["disposition"] = \
                "infrastructure_rollback"
            mutations.append(wrong_disposition)

            for mutation in mutations:
                terminal = mutation["terminals"][0]
                terminal["terminal_sha256"] = C._sha({
                    key: value for key, value in terminal.items()
                    if key != "terminal_sha256"})
                mutation["state_sha256"] = C._sha({
                    key: value for key, value in mutation.items()
                    if key != "state_sha256"})
                path.write_text(json.dumps(mutation))
                os.chmod(path, 0o600)
                with self.assertRaises(C.CompositionError):
                    ledger.load()

            path.write_text(json.dumps(original))
            os.chmod(path, 0o600)
            alias = Path(directory) / "state-hardlink.json"
            os.link(path, alias)
            with self.assertRaisesRegex(C.CompositionError, "identity is unsafe"):
                ledger.load()
            alias.unlink()
            target = Path(directory) / "target.json"
            path.rename(target)
            path.symlink_to(target)
            with self.assertRaises(C.CompositionError):
                ledger.load()


if __name__ == "__main__":
    unittest.main()
