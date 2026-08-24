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


BASE = C.FROZEN_PRODUCTION_COMMIT
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


def _write_receipt(path: Path, body: dict, native_key: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body.pop(native_key, None)
    body[native_key] = C._sha(body)
    raw = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_runner_plan(
        root: Path, pair: C.CumulativeBuildPair,
        correctness: C.FullCorrectness, *, exact_sha256: str,
        route_sha256: str, target_sha256: str,
        workload_sha256: str, runtime_config_sha256: str) -> None:
    exact_path = root / "proof/attribution-pair.json"
    exact_body = json.loads(exact_path.read_text(encoding="utf-8"))
    bundle = gpu_source_proofs.GpuSourceProofBundle.from_validated_paths(
        manifest_sha256=_h((pair.operation_key, "manifest")),
        candidate=pair.candidate.build_identity,
        anchor=pair.anchor.build_identity,
        workload_sha256=_h((pair.operation_key, "proof-workload")),
        correctness={
            "path": str(root / "proof/correctness/receipt.json"),
            "file_sha256": correctness.receipt_sha256,
            "native_sha256": correctness.result_sha256,
            "body": correctness.to_dict(),
        },
        attribution={
            "path": str(exact_path), "file_sha256": exact_sha256,
            "native_sha256": exact_body["receipt_sha256"],
            "body": exact_body,
        })
    _write_receipt(root / "proof/proof-bundle.json", {
        "schema": "epyc.autokernel.gpu_source_evidence_bundle.v1",
        "authority": "nonpromotable_candidate_only_discovery",
        "promotion_claim": False, "bundle": bundle.to_dict(),
    }, "receipt_sha256")
    body = {
        "schema": "epyc.autokernel.gpu_source_runner_plan.v2",
        "authority": "nonpromotable_candidate_only_discovery",
        "promotion_claim": False, "operation_key": pair.operation_key,
        "composition_plan_sha256": pair.plan_sha256,
        "composition_build_pair": pair.to_dict(),
        "composition_correctness": correctness.to_dict(),
        "composition_production_authority": _planned_production(
            workload_sha256=workload_sha256,
            runtime_config_sha256=runtime_config_sha256).to_dict(),
        "composition_exact_route_receipt_sha256": exact_sha256,
        "composition_expected_route_set_sha256": route_sha256,
        "composition_target_runtime_frame_sha256": target_sha256,
        "measurement_graphs_off_output_dir": str(
            root / "runner/s1/measurement-graphs-off"),
        "target_runtime_graphs_on_output_dir": str(
            root / "runner/s1/target-runtime-graphs-on"),
    }
    _write_receipt(root / "runner-plan.json", body, "receipt_sha256")


def _exact_carrier(
        effect: float, pair: C.CumulativeBuildPair,
        *, model_sha256: str = _h("model"),
        workload_sha256: str = _h("workload"),
        runtime_config_sha256: str = _h("runtime-config"),
) -> dict:
    anchor_total = 1_000_000
    candidate_total = round(anchor_total * (1.0 - effect))
    derived = (anchor_total - candidate_total) / anchor_total
    comparison = {
        "candidate_routes": {
            "candidate": {"total_duration_ns": candidate_total, "calls": 9}},
        "anchor_routes": {
            "anchor": {"total_duration_ns": anchor_total, "calls": 9}},
        "candidate_total_duration_ns": candidate_total,
        "anchor_total_duration_ns": anchor_total,
        "relative_improvement_fraction": derived,
        "direction": ("improved" if derived > 0 else
                      "regressed" if derived < 0 else "neutral"),
        "all_candidate_routes_present": True,
        "all_anchor_routes_present": True,
        "statistic": "sum_exact_route_total_duration_ns",
    }
    return {
        "schema": "epyc.autokernel.gpu_kernel_attribution_pair.v2",
        "authority": "nonpromotable_candidate_only_discovery",
        "non_promotable": True, "promotion_claim": False,
        "candidate_build_identity": vars(pair.candidate.build_identity),
        "anchor_build_identity": vars(pair.anchor.build_identity),
        "model_sha256": model_sha256,
        "workload_sha256": workload_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "exact_duration_comparison": comparison,
    }


def _comparison(pair: C.CumulativeBuildPair, correctness: C.FullCorrectness,
                route: float = .01, graphs: float = .01, *,
                workload_sha256: str = _h("workload"),
                runtime_config_sha256: str = _h("runtime-config")) \
        -> C.IncrementalComparison:
    root = Path(tempfile.mkdtemp(prefix="autokernel-comparison-")) / \
        pair.operation_key
    exact_path = root / "proof/attribution-pair.json"
    off_path = root / "runner/s1/measurement-graphs-off/result.json"
    on_path = root / "runner/s1/target-runtime-graphs-on/result.json"
    exact = _exact_carrier(
        route, pair, workload_sha256=workload_sha256,
        runtime_config_sha256=runtime_config_sha256)
    off = _measurement(
        pair=pair, anchor=pair.anchor.build_identity, graph_mode="off",
        factor="source_patch", effect=route)
    on = _measurement(
        pair=pair, anchor=pair.anchor.build_identity, graph_mode="on",
        factor="source_patch", effect=graphs)
    exact_sha = _write_receipt(exact_path, exact, "receipt_sha256")
    off_sha = _write_receipt(off_path, off, "result_sha256")
    on_sha = _write_receipt(on_path, on, "result_sha256")
    route_sha = C._sha(
        exact["exact_duration_comparison"]["candidate_routes"])
    target_sha = C._target_runtime_frame_sha256(on)
    _write_runner_plan(
        root, pair, correctness, exact_sha256=exact_sha,
        route_sha256=route_sha, target_sha256=target_sha,
        workload_sha256=workload_sha256,
        runtime_config_sha256=runtime_config_sha256)
    authority, payload = C._runner_measurement_authority_uncommitted(root)
    C._append_authority_event(
        root, kind="pre_run", operation_key=authority["operation_key"],
        payload=payload)
    return C.IncrementalComparison.create(
        pair, correctness,
        exact_route_receipt_sha256=exact_sha,
        exact_route_receipt_path=exact_path,
        graphs_off_receipt_sha256=off_sha,
        graphs_off_receipt_path=off_path,
        expected_route_set_sha256=route_sha,
        graphs_on_receipt_sha256=on_sha,
        graphs_on_receipt_path=on_path,
        target_runtime_frame_sha256=target_sha,
        exact_route_effect_fraction=
            exact["exact_duration_comparison"][
                "relative_improvement_fraction"],
        graphs_off_effect_fraction=off["median_relative"],
        graphs_on_effect_fraction=on["median_relative"],
    )


def _production(*, frame_sha256: str = _h("matched-on-frame"),
                protocol_sha256: str = _h("protocol-frame"),
                measurement_receipt_sha256: str = _h(("production", "on")),
                model_sha256: str = _h("model"),
                workload_sha256: str = _h("workload"),
                runtime_config_sha256: str = _h("runtime-config"),
                observed_workload_sha256: str = _h("observed-workload"),
                observed_runtime_config_sha256: str =
                    _h("observed-runtime-config")) \
        -> C.FrozenProductionAuthority:
    identity = replace(_identity("production"), source_commit=BASE)
    return C.FrozenProductionAuthority.create(
        production_commit=BASE, build_identity=identity,
        runtime_snapshot_sha256=_h("production-runtime-snapshot"),
        comparator_receipt_sha256=_h("production-comparator-receipt"),
        graphs_mode="graphs_on", frame_sha256=frame_sha256,
        measurement_protocol_sha256=protocol_sha256,
        measurement_receipt_sha256=measurement_receipt_sha256,
        model_sha256=model_sha256, workload_sha256=workload_sha256,
        runtime_config_sha256=runtime_config_sha256,
        observed_workload_sha256=observed_workload_sha256,
        observed_runtime_config_sha256=observed_runtime_config_sha256,
        metric="tokens_per_second", direction="higher_is_better")


def _planned_production(
        *, workload_sha256: str = _h("workload"),
        runtime_config_sha256: str = _h("runtime-config"),
) -> C.FrozenProductionAuthority:
    identity = _production().build_identity
    protocol = C.frozen_production_protocol_binding(
        model_sha256=_h("model"), build_identity=identity)
    return _production(
        frame_sha256=protocol["frame_sha256"],
        protocol_sha256=protocol["measurement_protocol_sha256"],
        workload_sha256=workload_sha256,
        runtime_config_sha256=runtime_config_sha256,
        observed_workload_sha256=protocol["observed_workload_sha256"],
        observed_runtime_config_sha256=
            protocol["observed_runtime_config_sha256"])


def _performance(
        directory: str | Path, plan: C.CompositionPlan,
        pair: C.CumulativeBuildPair, correctness: C.FullCorrectness,
        comparison: C.IncrementalComparison, *,
        cumulative_on: float = .02,
) -> tuple[C.CumulativePerformance, C.CumulativePerformanceRef]:
    production_path = (comparison.operation_root / "runner" /
                       comparison.repetition /
                       "cumulative-vs-production-graphs-on/result.json")
    production_body = _measurement(
        pair=pair, anchor=_production().build_identity, graph_mode="on",
        factor="cumulative_production", effect=cumulative_on)
    production_sha = _write_receipt(
        production_path, production_body, "result_sha256")
    off = C._runner_projection(
        comparison.graphs_off_receipt_ref.load(), graph_mode="off",
        factor_name="source_patch")
    on = C._runner_projection(
        comparison.graphs_on_receipt_ref.load(), graph_mode="on",
        factor_name="source_patch")
    production = C._runner_projection(
        production_body, graph_mode="on",
        factor_name="cumulative_production")
    frozen = _planned_production()
    performance = C.CumulativePerformance.create(
        plan, pair, correctness, comparison,
        frozen_production=frozen,
        model_sha256=_h("model"), workload_sha256=_h("workload"),
        runtime_config_sha256=_h("runtime-config"),
        protocol_frame_sha256=on["protocol_frame_sha256"],
        metric="decode_tokens_per_s", metric_direction="higher_better",
        cumulative_graphs_on_effect_fraction=
            production_body["median_relative"],
        production_graphs_on_receipt_sha256=production_sha,
        production_graphs_on_receipt_path=production_path,
        incremental_graphs_off_frame_sha256=
            off["candidate_frame_sha256"],
        incremental_graphs_on_frame_sha256=
            on["candidate_frame_sha256"],
        production_graphs_on_frame_sha256=
            production["anchor_frame_sha256"])
    reference = C.seal_cumulative_performance(
        comparison.operation_root / "cumulative-performance.json",
        performance)
    return performance, reference


def _record_performance(
        ledger: C.CompositionLedger, directory: str | Path,
        plan: C.CompositionPlan, pair: C.CumulativeBuildPair,
        correctness: C.FullCorrectness,
        comparison: C.IncrementalComparison, **kwargs,
) -> C.CumulativePerformance:
    performance, reference = _performance(
        directory, plan, pair, correctness, comparison, **kwargs)
    ledger.record_cumulative_performance(performance, reference)
    return performance


def _measurement(
        *, pair: C.CumulativeBuildPair,
        anchor: gpu_source_proofs.BuildIdentity, graph_mode: str,
        factor: str, effect: float,
) -> dict:
    metric_contract = {
        "schema": "epyc.autokernel.matched-test-metric.v1",
        "graph_mode": graph_mode,
    }
    raw_row = {
        "n_threads": 8, "n_batch": 512, "n_ubatch": 512,
        "use_mmap": True, "no_op_offload": 0,
        "split_mode": "layer", "no_kv_offload": False, "poll": 50,
        "n_prompt": 0, "n_gen": 128, "flash_attn": 1,
    }
    baseline_center = 1_000_000.0
    anchor_samples = [baseline_center] * 9
    candidate_samples = [baseline_center * (1.0 + effect)] * 9
    relative_effects = [
        (value - baseline_center) / baseline_center
        for value in candidate_samples]
    body = {
        "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
        "authority": "nonpromotable_candidate_only_discovery",
        "promotion_claim": False, "non_promotable": True,
        "hip_residency_proved": True, "runtime_graphs": graph_mode,
        "baseline_center": baseline_center,
        "candidate_samples": candidate_samples,
        "relative_effects": relative_effects,
        "median_relative": relative_effects[0],
        "baseline_sha256": _h("measurement-baseline"),
        "factor": factor, "technical_workload": {"tokens": 128},
        "frame": {
            "backend": "llama_gpu", "recipe": "tg128-ngl99",
            "metric": "decode_tokens_per_s",
            "metric_direction": "higher_better",
            "metric_contract": metric_contract,
            "n_prompt": 0, "n_gen": 128, "model": "/models/test.gguf",
            "model_sha256": _h("model"),
            "source_commit": pair.candidate.build_identity.source_commit,
            "cpu_list": "184-191", "device": "AMD Instinct MI210",
            "architecture": "gfx90a",
        },
        "sole_factor": {"name": factor},
        "anchor_identity": vars(anchor),
        "candidate_identity": vars(pair.candidate.build_identity),
        "anchor_samples": anchor_samples,
        "anchor_runs": [_producer_run(
            anchor_samples, raw_row, metric_contract, "anchor", anchor)],
        "candidate_runs": [_producer_run(
            candidate_samples, raw_row, metric_contract, "candidate",
            pair.candidate.build_identity)],
        "anchor_invocations": 9, "candidate_invocations": 9,
        "anchor_processes": 1, "candidate_processes": 1,
    }
    return body


def _producer_run(samples: list[float], raw_row: dict,
                  metric_contract: dict, label: str,
                  identity: gpu_source_proofs.BuildIdentity) -> dict:
    diagnostic = {"schema": "epyc.test.native_metric.v1", "arm": label}
    diagnostic["receipt_sha256"] = C._sha(diagnostic)
    return {
        "metric": sum(samples) / len(samples), "samples": samples,
        "metric_contract": metric_contract, "sample_count": len(samples),
        "raw_row": raw_row,
        "reward_binary_sha256": identity.binary_sha256,
        "hip_library_sha256": identity.hip_library_sha256,
        "native_metric_diagnostic": diagnostic,
        "supervisor": {
            "stdout_sha256": _h((label, "stdout")),
            "stderr_sha256": _h((label, "stderr")),
        },
    }


class AuthorityJournalTests(unittest.TestCase):
    """Append-only pre-run/result authority journal mechanics."""

    def _root(self, directory):
        root = Path(directory) / "operation"
        root.mkdir(parents=True)
        return root

    def test_journal_appends_chain_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            first = C._append_authority_event(
                root, kind="pre_run", operation_key=_h("op"),
                payload={"runner_plan_file_sha256": _h("plan")})
            second = C._append_authority_event(
                root, kind="result", operation_key=_h("op"),
                payload={"cumulative_performance_file_sha256": _h("perf")})
            rows = C._read_authority_journal(root)
            self.assertEqual([row["kind"] for row in rows],
                             ["pre_run", "result"])
            self.assertEqual([row["sequence"] for row in rows], [1, 2])
            self.assertEqual(rows[0]["previous_event_sha256"], "0" * 64)
            self.assertEqual(rows[1]["previous_event_sha256"],
                             rows[0]["event_sha256"])
            self.assertEqual(C._append_authority_event(
                root, kind="pre_run", operation_key=_h("op"),
                payload={"runner_plan_file_sha256": _h("plan")}), first)
            self.assertEqual(C._append_authority_event(
                root, kind="result", operation_key=_h("op"),
                payload={"cumulative_performance_file_sha256": _h("perf")}),
                second)
            self.assertEqual(len(C._read_authority_journal(root)), 2)

    def test_journal_refuses_result_before_pre_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            with self.assertRaisesRegex(
                    C.CompositionError, "lacks one pre-run commitment"):
                C._append_authority_event(
                    root, kind="result", operation_key=_h("op"),
                    payload={"cumulative_performance_file_sha256": _h("p")})

    def test_journal_refuses_changed_payload_on_reappend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            C._append_authority_event(
                root, kind="pre_run", operation_key=_h("op"),
                payload={"runner_plan_file_sha256": _h("plan")})
            with self.assertRaisesRegex(
                    C.CompositionError, "changed after commitment"):
                C._append_authority_event(
                    root, kind="pre_run", operation_key=_h("op"),
                    payload={"runner_plan_file_sha256": _h("other")})

    def test_journal_tamper_breaks_hash_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            C._append_authority_event(
                root, kind="pre_run", operation_key=_h("op"),
                payload={"runner_plan_file_sha256": _h("plan")})
            C._append_authority_event(
                root, kind="result", operation_key=_h("op"),
                payload={"cumulative_performance_file_sha256": _h("perf")})
            path = root / C._AUTHORITY_JOURNAL
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b"pre_run", b"pre_rxn"))
            with self.assertRaises(C.CompositionError):
                C._read_authority_journal(root)

    def test_journal_strict_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            path = root / C._AUTHORITY_JOURNAL
            path.write_bytes(
                b'{"schema":"epyc.autokernel.cumulative_authority_journal_event.v1",'
                b'"sequence":1,"previous_event_sha256":"' + b"0" * 64 +
                b'","kind":"pre_run","operation_key":"' + _h("op").encode() +
                b'","payload":{"runner_plan_file_sha256":"' + _h("plan").encode() +
                b'"},\n')
            with self.assertRaises(C.CompositionError):
                C._read_authority_journal(root)

    def test_load_requires_pre_run_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "operation"
            plan = _plan(_authority(_lever(1)), _lever(2))
            pair = _build_pair(plan)
            correctness = _correctness(pair)
            exact_path = root / "proof/attribution-pair.json"
            exact = _exact_carrier(.01, pair)
            exact_sha = _write_receipt(exact_path, exact, "receipt_sha256")
            off_path = root / "runner/s1/measurement-graphs-off/result.json"
            on_path = root / "runner/s1/target-runtime-graphs-on/result.json"
            off = _measurement(
                pair=pair, anchor=pair.anchor.build_identity,
                graph_mode="off", factor="source_patch", effect=.01)
            on = _measurement(
                pair=pair, anchor=pair.anchor.build_identity,
                graph_mode="on", factor="source_patch", effect=.01)
            _write_receipt(off_path, off, "result_sha256")
            _write_receipt(on_path, on, "result_sha256")
            route_sha = C._sha(
                exact["exact_duration_comparison"]["candidate_routes"])
            target_sha = C._target_runtime_frame_sha256(on)
            _write_runner_plan(
                root, pair, correctness, exact_sha256=exact_sha,
                route_sha256=route_sha, target_sha256=target_sha,
                workload_sha256=_h("workload"),
                runtime_config_sha256=_h("runtime-config"))
            with self.assertRaisesRegex(
                    C.CompositionError, "pre-run authority differs"):
                C._load_runner_measurement_authority(root)


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
            _record_performance(
                ledger, directory, first_plan, first_pair,
                first_correctness, first_comparison)
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
            _record_performance(
                ledger, directory, plan, pair, passed, comparison)
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
            self.assertEqual(
                C.CompositionLedger(path).load()["pending"]["stage"],
                "incremental_measured")
            performance = _record_performance(
                ledger, directory, plan, pair, correct, comparison)
            self.assertEqual(
                C.CompositionLedger(path).load()["pending"]["stage"],
                "measured")
            final = C.CompositionLedger(path).finalize(plan.operation_key)
            self.assertIsNone(final["pending"])
            self.assertEqual(final["scientific_attempts"], 1)
            self.assertEqual(
                C.CompositionAuthority.from_dict(final["authority"]),
                plan.candidate)
            self.assertTrue(final["terminals"][0]["promotion_eligible"])
            self.assertEqual(
                C.CumulativePerformance.from_dict(
                    final["terminals"][0]["cumulative_performance"]),
                performance)
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

    def test_effect_summaries_are_rederived_and_alias_carriers_refuse(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        comparison = _comparison(pair, correct)

        changed = comparison.to_dict()
        changed["graphs_on_effect_fraction"] += .01
        changed["result_sha256"] = C._sha({
            key: value for key, value in changed.items()
            if key != "result_sha256"})
        with self.assertRaisesRegex(
                C.CompositionError, "incremental comparison identity"):
            C.IncrementalComparison.from_dict(changed)

        original_path = Path(comparison.graphs_on_receipt_ref.path)
        alias_path = (comparison.operation_root / "copied" / "runner/s1" /
                      "target-runtime-graphs-on/result.json")
        alias_path.parent.mkdir(parents=True)
        alias_path.write_bytes(original_path.read_bytes())
        aliased = comparison.to_dict()
        aliased["graphs_on_receipt_ref"]["path"] = str(alias_path)
        aliased["result_sha256"] = C._sha({
            key: value for key, value in aliased.items()
            if key != "result_sha256"})
        with self.assertRaisesRegex(
                C.CompositionError, "locations or hashes"):
            C.IncrementalComparison.from_dict(aliased)

        carrier = comparison.graphs_on_receipt_ref.load()
        carrier["median_relative"] += .01
        changed_sha = _write_receipt(
            original_path, carrier, "result_sha256")
        laundered = comparison.to_dict()
        laundered["graphs_on_receipt_sha256"] = changed_sha
        laundered["graphs_on_receipt_ref"]["sha256"] = changed_sha
        laundered["graphs_on_effect_fraction"] = carrier["median_relative"]
        laundered["result_sha256"] = C._sha({
            key: value for key, value in laundered.items()
            if key != "result_sha256"})
        with self.assertRaisesRegex(
                C.CompositionError, "not derived from samples"):
            C.IncrementalComparison.from_dict(laundered)

    def test_every_duplicated_derived_field_refuses_coherent_reseal(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        comparison = _comparison(pair, correct)
        comparison_mutations = {
            "operation_key": _h("other-operation"),
            "build_pair_sha256": _h("other-pair"),
            "correctness_result_sha256": _h("other-correctness"),
            "expected_route_set_sha256": _h("other-routes"),
            "target_runtime_frame_sha256": _h("other-target-frame"),
            "exact_route_effect_fraction": .06,
            "graphs_off_effect_fraction": .06,
            "graphs_on_effect_fraction": .06,
            "classification": "screened_out",
        }
        for field, replacement in comparison_mutations.items():
            with self.subTest(carrier="comparison", field=field):
                changed = comparison.to_dict()
                changed[field] = replacement
                changed["result_sha256"] = C._sha({
                    key: value for key, value in changed.items()
                    if key != "result_sha256"})
                with self.assertRaises(C.CompositionError):
                    C.IncrementalComparison.from_dict(changed)

        with tempfile.TemporaryDirectory() as directory:
            performance, _reference = _performance(
                directory, plan, pair, correct, comparison)
            performance_mutations = {
                "operation_key": _h("other-operation"),
                "plan_sha256": _h("other-plan"),
                "accepted_authority_sha256": _h("other-authority"),
                "accepted_patch_set_sha256": _h("other-patch-set"),
                "build_pair_sha256": _h("other-pair"),
                "correctness_result_sha256": _h("other-correctness"),
                "incremental_comparison_result_sha256":
                    _h("other-comparison"),
                "model_sha256": _h("other-model"),
                "workload_sha256": _h("other-workload"),
                "runtime_config_sha256": _h("other-runtime"),
                "protocol_frame_sha256": _h("other-protocol"),
                "metric": "prefill_tokens_per_s",
                "metric_direction": "lower_better",
                "incremental_exact_route_effect_fraction": .06,
                "incremental_graphs_off_effect_fraction": .06,
                "incremental_graphs_on_effect_fraction": .06,
                "cumulative_graphs_on_effect_fraction": .06,
                "incremental_graphs_off_frame_sha256": _h("other-off-frame"),
                "incremental_graphs_on_frame_sha256": _h("other-on-frame"),
                "production_graphs_on_frame_sha256":
                    _h("other-production-frame"),
                "production_graphs_mode": "off",
                "cumulative_classification": "screened_out",
                "promotion_eligible": False,
                "promotion_reason": "cumulative_screened_out",
                "composition_terminal_sha256": _h("other-terminal"),
            }
            for field, replacement in performance_mutations.items():
                with self.subTest(carrier="performance", field=field):
                    changed = performance.to_dict()
                    changed[field] = replacement
                    changed["result_sha256"] = C._sha({
                        key: value for key, value in changed.items()
                        if key != "result_sha256"})
                    with self.assertRaises(C.CompositionError):
                        parsed = C.CumulativePerformance.from_dict(changed)
                        parsed.bind(plan, pair, correct, comparison)

            frozen = performance.frozen_production
            changed_frozen = C.FrozenProductionAuthority.create(
                production_commit=frozen.production_commit,
                build_identity=frozen.build_identity,
                runtime_snapshot_sha256=_h("substituted-runtime-snapshot"),
                comparator_receipt_sha256=frozen.comparator_receipt_sha256,
                graphs_mode=frozen.graphs_mode,
                frame_sha256=frozen.frame_sha256,
                measurement_protocol_sha256=
                    frozen.measurement_protocol_sha256,
                measurement_receipt_sha256=
                    frozen.measurement_receipt_sha256,
                model_sha256=frozen.model_sha256,
                workload_sha256=frozen.workload_sha256,
                runtime_config_sha256=frozen.runtime_config_sha256,
                observed_workload_sha256=
                    frozen.observed_workload_sha256,
                observed_runtime_config_sha256=
                    frozen.observed_runtime_config_sha256,
                metric=frozen.metric, direction=frozen.direction)
            changed = performance.to_dict()
            changed["frozen_production"] = changed_frozen.to_dict()
            changed["result_sha256"] = C._sha({
                key: value for key, value in changed.items()
                if key != "result_sha256"})
            with self.assertRaisesRegex(
                    C.CompositionError, "measurement carriers"):
                C.CumulativePerformance.from_dict(changed)

    def test_raw_run_and_route_reseals_cannot_override_pre_run_authority(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        correct = _correctness(pair)

        comparison = _comparison(pair, correct)
        raw = comparison.graphs_on_receipt_ref.load()
        raw["candidate_samples"] = [2_000_000.0] * 9
        raw["relative_effects"] = [1.0] * 9
        raw["median_relative"] = 1.0
        path = Path(comparison.graphs_on_receipt_ref.path)
        receipt_sha = _write_receipt(path, raw, "result_sha256")
        changed = comparison.to_dict()
        changed["graphs_on_receipt_sha256"] = receipt_sha
        changed["graphs_on_receipt_ref"]["sha256"] = receipt_sha
        changed["graphs_on_effect_fraction"] = 1.0
        changed["result_sha256"] = C._sha({
            key: value for key, value in changed.items()
            if key != "result_sha256"})
        with self.assertRaisesRegex(C.CompositionError, "flattened samples"):
            C.IncrementalComparison.from_dict(changed)

        comparison = _comparison(pair, correct)
        raw = comparison.graphs_on_receipt_ref.load()
        raw["candidate_runs"][0]["samples"] = [2_000_000.0] * 9
        raw["candidate_runs"][0]["metric"] = 2_000_000.0
        path = Path(comparison.graphs_on_receipt_ref.path)
        receipt_sha = _write_receipt(path, raw, "result_sha256")
        changed = comparison.to_dict()
        changed["graphs_on_receipt_sha256"] = receipt_sha
        changed["graphs_on_receipt_ref"]["sha256"] = receipt_sha
        changed["result_sha256"] = C._sha({
            key: value for key, value in changed.items()
            if key != "result_sha256"})
        with self.assertRaisesRegex(C.CompositionError, "flattened samples"):
            C.IncrementalComparison.from_dict(changed)

        comparison = _comparison(pair, correct)
        exact = comparison.exact_route_receipt_ref.load()
        routes = exact["exact_duration_comparison"]
        routes["candidate_routes"] = {
            "substituted": {"total_duration_ns": 100_000, "calls": 9}}
        routes["candidate_total_duration_ns"] = 100_000
        routes["relative_improvement_fraction"] = .9
        routes["direction"] = "improved"
        path = Path(comparison.exact_route_receipt_ref.path)
        receipt_sha = _write_receipt(path, exact, "receipt_sha256")
        changed = comparison.to_dict()
        changed["exact_route_receipt_sha256"] = receipt_sha
        changed["exact_route_receipt_ref"]["sha256"] = receipt_sha
        changed["expected_route_set_sha256"] = C._sha(
            routes["candidate_routes"])
        changed["exact_route_effect_fraction"] = .9
        changed["result_sha256"] = C._sha({
            key: value for key, value in changed.items()
            if key != "result_sha256"})
        with self.assertRaisesRegex(C.CompositionError, "proof bundle"):
            C.IncrementalComparison.from_dict(changed)

    def test_preflight_and_result_derive_one_target_runtime_authority(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        result = _measurement(
            pair=pair, anchor=pair.anchor.build_identity, graph_mode="on",
            factor="source_patch", effect=.01)
        raw = result["candidate_runs"][0]["raw_row"]
        sealed = {
            "frame": result["frame"]["recipe"],
            "metric": result["frame"]["metric"],
            "prompt_tokens": result["frame"]["n_prompt"],
            "generation_tokens": result["frame"]["n_gen"],
            "model_sha256": result["frame"]["model_sha256"],
            "runtime_graphs": result["runtime_graphs"],
            "sole_factor": result["sole_factor"],
            "candidate_threads": raw["n_threads"],
            "candidate_batch": raw["n_batch"],
            "candidate_ubatch": raw["n_ubatch"],
            "candidate_mmap": raw["use_mmap"],
            "candidate_no_op_offload": raw["no_op_offload"],
            "candidate_split_mode": raw["split_mode"],
            "candidate_no_kv_offload": raw["no_kv_offload"],
            "candidate_poll": raw["poll"],
            "candidate_flash_attention": raw["flash_attn"],
        }
        self.assertEqual(
            C.planned_target_runtime_frame_sha256(
                sealed, candidate_identity=pair.candidate.build_identity),
            C._target_runtime_frame_sha256(result))

    def test_restart_rederives_pending_and_terminal_cumulative_effects(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        comparison = _comparison(pair, correct)
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            ledger = C.CompositionLedger(state_path)
            ledger.create(plan.anchor)
            ledger.begin(plan)
            ledger.record_build_pair(pair)
            ledger.record_correctness(correct)
            ledger.record_comparison(comparison)
            performance = _record_performance(
                ledger, directory, plan, pair, correct, comparison)
            self.assertEqual(
                C.CompositionLedger(state_path).load()["pending"]["stage"],
                "measured")

            changed = performance.to_dict()
            changed["cumulative_graphs_on_effect_fraction"] += .01
            changed["result_sha256"] = C._sha({
                key: value for key, value in changed.items()
                if key != "result_sha256"})
            with self.assertRaisesRegex(
                    C.CompositionError,
                    "measurement carriers changed"):
                C.CumulativePerformance.from_dict(changed)

            changed = performance.to_dict()
            changed["incremental_graphs_on_effect_fraction"] += .01
            changed["result_sha256"] = C._sha({
                key: value for key, value in changed.items()
                if key != "result_sha256"})
            with self.assertRaisesRegex(
                    C.CompositionError,
                    "measurement carriers changed"):
                C.CumulativePerformance.from_dict(changed)

            ledger.finalize(plan.operation_key)
            self.assertIsNone(C.CompositionLedger(state_path).load()["pending"])
            production_path = Path(
                performance.production_graphs_on_receipt_ref.path)
            production = performance.production_graphs_on_receipt_ref.load()
            production["candidate_samples"][0] *= 1.01
            _write_receipt(production_path, production, "result_sha256")
            with self.assertRaisesRegex(
                    C.CompositionError, "measurement receipt bytes changed"):
                C.CompositionLedger(state_path).load()

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
                performance = _record_performance(
                    ledger, directory, plan, pair, correct, comparison)
                final = ledger.finalize(plan.operation_key)
                self.assertEqual(final["terminals"][0]["disposition"],
                                 "incremental_rollback")
                self.assertEqual(
                    C.CompositionAuthority.from_dict(final["authority"]),
                    plan.anchor)
                self.assertFalse(performance.promotion_eligible)

    def test_positive_incremental_nonpositive_cumulative_is_admitted_but_nonpromotable(self):
        plan = _plan(_authority(_lever(1)), _lever(2))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        comparison = _comparison(pair, correct)
        with tempfile.TemporaryDirectory() as directory:
            ledger = C.CompositionLedger(Path(directory) / "state.json")
            ledger.create(plan.anchor)
            ledger.begin(plan)
            ledger.record_build_pair(pair)
            ledger.record_correctness(correct)
            ledger.record_comparison(comparison)
            performance = _record_performance(
                ledger, directory, plan, pair, correct, comparison,
                cumulative_on=-.02)
            final = ledger.finalize(plan.operation_key)
            terminal = final["terminals"][0]
            self.assertEqual(terminal["disposition"], "admitted")
            self.assertFalse(terminal["promotion_eligible"])
            self.assertEqual(terminal["promotion_reason"],
                             "cumulative_screened_out")
            self.assertFalse(performance.promotion_eligible)
            self.assertEqual(
                C.CompositionAuthority.from_dict(final["authority"]),
                plan.candidate)

    def test_positive_cumulative_with_negative_latest_rolls_back(self):
        plan = _plan(_authority(_lever(1)), _lever(2))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        comparison = _comparison(pair, correct, route=-.01, graphs=-.02)
        with tempfile.TemporaryDirectory() as directory:
            ledger = C.CompositionLedger(Path(directory) / "state.json")
            ledger.create(plan.anchor)
            ledger.begin(plan)
            ledger.record_build_pair(pair)
            ledger.record_correctness(correct)
            ledger.record_comparison(comparison)
            performance = _record_performance(
                ledger, directory, plan, pair, correct, comparison,
                cumulative_on=.03)
            final = ledger.finalize(plan.operation_key)
            terminal = final["terminals"][0]
            self.assertEqual(terminal["disposition"],
                             "incremental_rollback")
            self.assertFalse(performance.promotion_eligible)
            self.assertEqual(performance.promotion_reason,
                             "incremental_screened_out")
            self.assertEqual(
                C.CompositionAuthority.from_dict(final["authority"]),
                plan.anchor)

    def test_three_result_protocol_match_creates_promotion_authority(self):
        plan = _plan(_authority(_lever(1)), _lever(2))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        comparison = _comparison(
            pair, correct,
            workload_sha256=_h("deployment-workload-file"),
            runtime_config_sha256=_h("deployment-runtime-file"))
        production_identity = replace(
            _identity("production"), source_commit=BASE)
        incremental_off = comparison.graphs_off_receipt_ref.load()
        incremental_on = comparison.graphs_on_receipt_ref.load()
        production_on = _measurement(
            pair=pair, anchor=production_identity, graph_mode="on",
            factor="cumulative_production", effect=.03)
        production_path = (
            comparison.operation_root / "runner" / comparison.repetition /
            "cumulative-vs-production-graphs-on/result.json")
        production_sha = _write_receipt(
            production_path, production_on, "result_sha256")
        production_descriptor = C._measurement_descriptor(
            production_on, graph_mode="on", candidate=pair.candidate,
            anchor_identity=production_identity,
            factor_name="cumulative_production")
        production = _planned_production(
            workload_sha256=_h("deployment-workload-file"),
            runtime_config_sha256=_h("deployment-runtime-file"))
        performance = C.performance_from_measurements(
            plan, pair, correct, comparison,
            frozen_production=production,
            incremental_graphs_off=incremental_off,
            incremental_graphs_on=incremental_on,
            production_graphs_on=production_on,
            production_graphs_on_receipt_sha256=production_sha,
            production_graphs_on_receipt_path=production_path)
        self.assertTrue(performance.promotion_eligible)
        self.assertEqual(performance.workload_sha256,
                         _h("deployment-workload-file"))
        self.assertEqual(performance.runtime_config_sha256,
                         _h("deployment-runtime-file"))
        self.assertNotEqual(performance.runtime_config_sha256,
                            production_descriptor["runtime_config_sha256"])
        self.assertEqual(
            C.CumulativePerformance.from_dict(performance.to_dict()),
            performance)

        mixed = copy.deepcopy(production_on)
        mixed["runtime_graphs"] = "off"
        with self.assertRaisesRegex(C.CompositionError, "authority changed"):
            C.performance_from_measurements(
                plan, pair, correct, comparison,
                frozen_production=production,
                incremental_graphs_off=incremental_off,
                incremental_graphs_on=incremental_on,
                production_graphs_on=mixed,
                production_graphs_on_receipt_sha256=production_sha,
                production_graphs_on_receipt_path=production_path)

        mismatched = copy.deepcopy(production_on)
        mismatched["frame"]["model_sha256"] = _h("different-model")
        with self.assertRaisesRegex(C.CompositionError, "protocol"):
            C.performance_from_measurements(
                plan, pair, correct, comparison,
                frozen_production=production,
                incremental_graphs_off=incremental_off,
                incremental_graphs_on=incremental_on,
                production_graphs_on=mismatched,
                production_graphs_on_receipt_sha256=production_sha,
                production_graphs_on_receipt_path=production_path)

        protocol_changed = copy.deepcopy(production_on)
        protocol_changed["frame"]["cpu_list"] = "176-183"
        with self.assertRaisesRegex(C.CompositionError, "protocol"):
            C.performance_from_measurements(
                plan, pair, correct, comparison,
                frozen_production=production,
                incremental_graphs_off=incremental_off,
                incremental_graphs_on=incremental_on,
                production_graphs_on=protocol_changed,
                production_graphs_on_receipt_sha256=production_sha,
                production_graphs_on_receipt_path=production_path)

        observed_runtime_changed = C.FrozenProductionAuthority.create(
            production_commit=production.production_commit,
            build_identity=production.build_identity,
            runtime_snapshot_sha256=
                production.runtime_snapshot_sha256,
            comparator_receipt_sha256=
                production.comparator_receipt_sha256,
            graphs_mode=production.graphs_mode,
            frame_sha256=production.frame_sha256,
            measurement_protocol_sha256=
                production.measurement_protocol_sha256,
            measurement_receipt_sha256=
                production.measurement_receipt_sha256,
            model_sha256=production.model_sha256,
            workload_sha256=production.workload_sha256,
            runtime_config_sha256=
                production.runtime_config_sha256,
            observed_workload_sha256=
                production.observed_workload_sha256,
            observed_runtime_config_sha256=
                _h("foreign-observed-runtime"),
            metric=production.metric, direction=production.direction)
        with self.assertRaisesRegex(
                C.CompositionError, "sealed protocol"):
            C.performance_from_measurements(
                plan, pair, correct, comparison,
                frozen_production=observed_runtime_changed,
                incremental_graphs_off=incremental_off,
                incremental_graphs_on=incremental_on,
                production_graphs_on=production_on,
                production_graphs_on_receipt_sha256=production_sha,
                production_graphs_on_receipt_path=production_path)

        candidate_changed = copy.deepcopy(production_on)
        replacement_identity = _identity("replacement-candidate")
        candidate_changed["candidate_identity"] = \
            replacement_identity.__dict__
        candidate_changed["frame"]["source_commit"] = \
            replacement_identity.source_commit
        candidate_changed["candidate_runs"][0]["reward_binary_sha256"] = \
            replacement_identity.binary_sha256
        candidate_changed["candidate_runs"][0]["hip_library_sha256"] = \
            replacement_identity.hip_library_sha256
        replacement_binding = C.BuildBinding.create(
            pair.candidate.patch_set_sha256, replacement_identity,
            source_materialization_receipt_sha256=
                pair.candidate.source_materialization_receipt_sha256)
        replacement_descriptor = C._measurement_descriptor(
            candidate_changed, graph_mode="on",
            candidate=replacement_binding,
            anchor_identity=production_identity,
            factor_name="cumulative_production")
        self.assertEqual(
            replacement_descriptor["protocol_frame_sha256"],
            production_descriptor["protocol_frame_sha256"])
        self.assertNotEqual(
            replacement_descriptor["frame_sha256"],
            production_descriptor["frame_sha256"])
        with self.assertRaisesRegex(
                C.CompositionError, "candidate build identity changed"):
            C.performance_from_measurements(
                plan, pair, correct, comparison,
                frozen_production=production,
                incremental_graphs_off=incremental_off,
                incremental_graphs_on=incremental_on,
                production_graphs_on=candidate_changed,
                production_graphs_on_receipt_sha256=production_sha,
                production_graphs_on_receipt_path=production_path)

        with self.assertRaisesRegex(
                C.CompositionError, "three distinct runs"):
            C.performance_from_measurements(
                plan, pair, correct, comparison,
                frozen_production=production,
                incremental_graphs_off=incremental_off,
                incremental_graphs_on=incremental_on,
                production_graphs_on=production_on,
                production_graphs_on_receipt_sha256=
                    comparison.graphs_on_receipt_sha256,
                production_graphs_on_receipt_path=production_path)

    def test_static_frozen_comparator_exact_schema_and_tamper_refusal(self):
        identity = replace(_identity("production-static"), source_commit=BASE)
        comparator = C.FrozenProductionComparator.create(
            build_identity=identity, build_receipt_sha256=_h("build"),
            linkage_receipt_sha256=_h("linkage"),
            runtime_receipt_sha256=_h("runtime"),
            runtime_snapshot_sha256=_h("snapshot"),
            measurement_receipt_sha256=_h("measurement"),
            model_sha256=_h("model"), workload_sha256=_h("workload"),
            runtime_config_sha256=_h("runtime-config"),
            observed_workload_sha256=_h("observed-workload"),
            observed_runtime_config_sha256=_h("observed-runtime-config"),
            frame_sha256=_h("production-frame"),
            measurement_protocol_sha256=_h("protocol"))
        reopened = C.FrozenProductionComparator.from_dict(
            comparator.to_dict())
        self.assertEqual(reopened, comparator)
        self.assertEqual(reopened.authority().build_identity, identity)
        for mutation in (
                lambda value: value.update(graphs_mode="graphs_off"),
                lambda value: value.update(commit="f" * 40),
                lambda value: value.update(runtime_snapshot_sha256="0" * 64),
                lambda value: value.update(extra="unexpected")):
            damaged = copy.deepcopy(comparator.to_dict())
            mutation(damaged)
            with self.assertRaises(C.CompositionError):
                C.FrozenProductionComparator.from_dict(damaged)

    def test_performance_receipt_same_fd_reopen_and_tamper_refusal(self):
        plan = _plan(_authority(), _lever(1))
        pair = _build_pair(plan)
        correct = _correctness(pair)
        comparison = _comparison(pair, correct)
        with tempfile.TemporaryDirectory() as directory:
            comparison = _comparison(pair, correct)
            performance, reference = _performance(
                directory, plan, pair, correct, comparison)
            reopened, digest = C.load_cumulative_performance(
                Path(reference.path), expected_file_sha256=reference.sha256)
            self.assertEqual((reopened, digest),
                             (performance, reference.sha256))
            path = Path(reference.path)
            raw = json.loads(path.read_text())
            raw["promotion_eligible"] = False
            raw["result_sha256"] = C._sha({
                key: value for key, value in raw.items()
                if key != "result_sha256"})
            path.write_text(json.dumps(raw))
            os.chmod(path, 0o600)
            with self.assertRaises(C.CompositionError):
                C.load_cumulative_performance(
                    path, expected_file_sha256=reference.sha256)

        with tempfile.TemporaryDirectory() as directory:
            comparison = _comparison(pair, correct)
            performance, reference = _performance(
                directory, plan, pair, correct, comparison)
            path = Path(reference.path)
            alias = path.with_name("performance-hardlink.json")
            os.link(path, alias)
            with self.assertRaisesRegex(C.CompositionError, "identity is unsafe"):
                C.load_cumulative_performance(path)

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
                comparison = _comparison(pair, correct)
                ledger.record_comparison(comparison)
                _record_performance(
                    ledger, directory, plan, pair, correct, comparison)
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
            comparison = _comparison(pair, correct)
            ledger.record_comparison(comparison)
            _record_performance(
                ledger, directory, plan, pair, correct, comparison)
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
