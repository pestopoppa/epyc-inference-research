"""Hardware-free controller integration for cumulative source composition."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from .. import cumulative_composition as composition
from .. import schemas
from .. import source_candidate
from . import discovery_controller as controller
from . import gpu_source_adapter
from . import gpu_source_evidence
from . import gpu_source_proofs


BASE = composition.FROZEN_PRODUCTION_COMMIT
INSTRUMENT = "2" * 40
RUNTIME = {
    "kind": "docker_workspace_bind_only", "docker_path": "/docker",
    "docker_sha256": "a" * 64, "image_id": "image",
    "codex_native_sha256": "a" * 64,
    "code_mode_host_sha256": "a" * 64,
    "ca_certificate_sha256": "a" * 64,
    "writable_host_binds": ["/workspace"],
    "host_network_mode": "docker_bridge",
}
CLAUDE_RUNTIME = {
    "kind": "claude_cli_structured_critic", "provider": "claude",
    "model": "claude-fable-5", "effort": "high",
    "wrapper_path": "/sealed/claude", "wrapper_sha256": "a" * 64,
    "argv_policy_sha256": "a" * 64,
    "auth_staging_policy":
        "ephemeral_0600_copy_atomic_oauth_rotation_sync_no_secret_receipt",
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _candidate() -> controller.PlannedCandidate:
    path = "ggml/src/ggml-cuda/composition-test.cu"
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n"
        "@@ -10,1 +10,1 @@ static int fresh_lever()\n"
        "-return 1;\n+return 2;\n"
    ).encode()
    manifest = source_candidate.SourcePatchManifest(
        campaign_id="ak-cumulative-controller-test",
        proposal_id="akp-cumulative-controller-test",
        candidate_id="akc-cumulative-controller-test",
        source_tree="llama.cpp", production_base_commit=BASE,
        instrument_commit=INSTRUMENT, change_class="arithmetic",
        declared_files=(path,), declared_symbols={path: ("fresh_lever",)},
        mechanism_id="fresh-lever",
        patch_sha256=hashlib.sha256(patch).hexdigest(), patch_bytes=patch)
    return controller.PlannedCandidate(
        hypothesis_id="akh-cumulative-controller-test",
        statement="the fresh lever reduces target runtime",
        falsifier="two isolated repetitions do not remain positive",
        regime={"backend": "gpu", "phase": "decode"},
        proposal={
            "proposal_id": manifest.proposal_id,
            "change_class": manifest.change_class,
            "change": {
                "files_and_symbols": [f"{path}:fresh_lever"],
                "estimated_diff_size": 2,
            },
        }, source_manifest=manifest,
        source_manifest_sha256=manifest.patch_bundle_sha256)


def _second_candidate() -> controller.PlannedCandidate:
    path = "ggml/src/ggml-cuda/composition-test.cu"
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n"
        "@@ -30,1 +30,1 @@ static int second_lever()\n"
        "-return 3;\n+return 4;\n"
    ).encode()
    manifest = source_candidate.SourcePatchManifest(
        campaign_id="ak-cumulative-controller-test",
        proposal_id="akp-cumulative-controller-second",
        candidate_id="akc-cumulative-controller-second",
        source_tree="llama.cpp", production_base_commit=BASE,
        instrument_commit=INSTRUMENT, change_class="arithmetic",
        declared_files=(path,), declared_symbols={path: ("second_lever",)},
        mechanism_id="second-lever",
        patch_sha256=hashlib.sha256(patch).hexdigest(), patch_bytes=patch)
    return controller.PlannedCandidate(
        hypothesis_id="akh-cumulative-controller-second",
        statement="the second lever reduces target runtime",
        falsifier="two isolated repetitions do not remain positive",
        regime={"backend": "gpu", "phase": "decode"},
        proposal={
            "proposal_id": manifest.proposal_id,
            "change_class": manifest.change_class,
            "change": {
                "files_and_symbols": [f"{path}:second_lever"],
                "estimated_diff_size": 2,
            },
        }, source_manifest=manifest,
        source_manifest_sha256=manifest.patch_bundle_sha256)


def _screen(repetition: int, classification: str) -> controller.SealedScreen:
    return controller.SealedScreen(
        receipt_path=f"/sealed/s{repetition}/result.json",
        result_sha256=_sha(f"result-{repetition}"),
        effect_fraction=.01 + repetition / 1000,
        classification=classification,
        baseline_sha256=_sha("baseline"),
        source_proof_sha256=_sha(f"source-{repetition}"),
        dispatch_proof_sha256=_sha(f"dispatch-{repetition}"),
        exact_attribution_effect_fraction=.02,
        target_runtime_effect_fraction=.01 + repetition / 1000,
        series_key=_sha("series"),
        series_effect_fraction=.0115,
        build_identity_sha256=_sha("same-build"),
        correctness_receipt_sha256=_sha(f"correctness-{repetition}"),
        attribution_receipt_sha256=_sha(f"attribution-{repetition}"),
        graphs_off_receipt_sha256=_sha(f"graphs-off-{repetition}"),
        graphs_on_receipt_sha256=_sha(f"graphs-on-{repetition}"))


def _second_screen(
        repetition: int, classification: str) -> controller.SealedScreen:
    return replace(
        _screen(repetition, classification),
        receipt_path=f"/sealed/second/s{repetition}/result.json",
        result_sha256=_sha(f"second-result-{repetition}"),
        source_proof_sha256=_sha(f"second-source-{repetition}"),
        dispatch_proof_sha256=_sha(f"second-dispatch-{repetition}"),
        series_key=_sha("second-series"),
        build_identity_sha256=_sha("second-same-build"),
        correctness_receipt_sha256=
            _sha(f"second-correctness-{repetition}"),
        attribution_receipt_sha256=
            _sha(f"second-attribution-{repetition}"),
        graphs_off_receipt_sha256=
            _sha(f"second-graphs-off-{repetition}"),
        graphs_on_receipt_sha256=
            _sha(f"second-graphs-on-{repetition}"))


def _identity(label: str) -> gpu_source_proofs.BuildIdentity:
    return gpu_source_proofs.BuildIdentity(
        source_commit=_sha(f"commit-{label}")[:40],
        source_sha256=_sha(f"source-{label}"),
        binary_sha256=_sha(f"binary-{label}"),
        hip_library_sha256=_sha(f"hip-{label}"),
        config_sha256=_sha(f"config-{label}"),
        linkage_sha256=_sha(f"linkage-{label}"))


def _production(*, frame_sha256: str = _sha("matched-on"),
                protocol_sha256: str = _sha("protocol"),
                measurement_receipt_sha256: str = _sha("production-on"),
                model_sha256: str = _sha("model"),
                workload_sha256: str = _sha("workload"),
                runtime_config_sha256: str = _sha("runtime"),
                observed_workload_sha256: str = _sha("observed-workload"),
                observed_runtime_config_sha256: str =
                    _sha("observed-runtime")) \
        -> composition.FrozenProductionAuthority:
    identity = replace(_identity("production"), source_commit=BASE)
    return composition.FrozenProductionAuthority.create(
        production_commit=BASE, build_identity=identity,
        runtime_snapshot_sha256=_sha("production-runtime-snapshot"),
        comparator_receipt_sha256=_sha("production-comparator-receipt"),
        graphs_mode="graphs_on", frame_sha256=frame_sha256,
        measurement_protocol_sha256=protocol_sha256,
        measurement_receipt_sha256=measurement_receipt_sha256,
        model_sha256=model_sha256, workload_sha256=workload_sha256,
        runtime_config_sha256=runtime_config_sha256,
        observed_workload_sha256=observed_workload_sha256,
        observed_runtime_config_sha256=observed_runtime_config_sha256,
        metric="tokens_per_second", direction="higher_is_better")


def _planned_production(
        *, workload_sha256: str = _sha("workload"),
        runtime_config_sha256: str = _sha("runtime"),
) -> composition.FrozenProductionAuthority:
    identity = _production().build_identity
    protocol = composition.frozen_production_protocol_binding(
        model_sha256=_sha("model"), build_identity=identity)
    return _production(
        frame_sha256=protocol["frame_sha256"],
        protocol_sha256=protocol["measurement_protocol_sha256"],
        workload_sha256=workload_sha256,
        runtime_config_sha256=runtime_config_sha256,
        observed_workload_sha256=protocol["observed_workload_sha256"],
        observed_runtime_config_sha256=
            protocol["observed_runtime_config_sha256"])


def _composition_performance(
        root: Path, plan: composition.CompositionPlan,
        pair: composition.CumulativeBuildPair,
        correctness: composition.FullCorrectness,
        comparison: composition.IncrementalComparison, *,
        cumulative_on: float = .02,
) -> tuple[composition.CumulativePerformance,
           composition.CumulativePerformanceRef]:
    production_path = (
        comparison.operation_root / "runner" / comparison.repetition /
        "cumulative-vs-production-graphs-on/result.json")
    production_body = _measurement(
        pair=pair, anchor=_production().build_identity, graph_mode="on",
        factor="cumulative_production", effect=cumulative_on)
    production_sha = _write_receipt(
        production_path, production_body, "result_sha256")
    off = composition._runner_projection(
        comparison.graphs_off_receipt_ref.load(), graph_mode="off",
        factor_name="source_patch")
    on = composition._runner_projection(
        comparison.graphs_on_receipt_ref.load(), graph_mode="on",
        factor_name="source_patch")
    production = composition._runner_projection(
        production_body, graph_mode="on",
        factor_name="cumulative_production")
    frozen = _planned_production()
    performance = composition.CumulativePerformance.create(
        plan, pair, correctness, comparison,
        frozen_production=frozen,
        model_sha256=_sha("model"), workload_sha256=_sha("workload"),
        runtime_config_sha256=_sha("runtime"),
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
    reference = composition.seal_cumulative_performance(
        comparison.operation_root / "cumulative-performance.json",
        performance)
    return performance, reference


def _write_receipt(path: Path, body: dict, native_key: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body.pop(native_key, None)
    body[native_key] = schemas.content_hash(body)
    raw = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_runner_plan(
        root: Path, pair: composition.CumulativeBuildPair,
        correctness: composition.FullCorrectness, *, exact_sha256: str,
        route_sha256: str, target_sha256: str) -> None:
    exact_path = root / "proof/attribution-pair.json"
    exact_body = json.loads(exact_path.read_text(encoding="utf-8"))
    bundle = gpu_source_proofs.GpuSourceProofBundle.from_validated_paths(
        manifest_sha256=_sha(pair.operation_key + "-manifest"),
        candidate=pair.candidate.build_identity,
        anchor=pair.anchor.build_identity,
        workload_sha256=_sha(pair.operation_key + "-proof-workload"),
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
        "composition_production_authority": _planned_production().to_dict(),
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
        effect: float, pair: composition.CumulativeBuildPair,
        *, workload_sha256: str = _sha("workload"),
        runtime_config_sha256: str = _sha("runtime"),
) -> dict:
    anchor_total = 1_000_000
    candidate_total = round(anchor_total * (1.0 - effect))
    derived = (anchor_total - candidate_total) / anchor_total
    return {
        "schema": "epyc.autokernel.gpu_kernel_attribution_pair.v2",
        "authority": "nonpromotable_candidate_only_discovery",
        "non_promotable": True, "promotion_claim": False,
        "candidate_build_identity": vars(pair.candidate.build_identity),
        "anchor_build_identity": vars(pair.anchor.build_identity),
        "model_sha256": _sha("model"),
        "workload_sha256": workload_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "exact_duration_comparison": {
            "candidate_routes": {"candidate": {
                "total_duration_ns": candidate_total, "calls": 9}},
            "anchor_routes": {"anchor": {
                "total_duration_ns": anchor_total, "calls": 9}},
            "candidate_total_duration_ns": candidate_total,
            "anchor_total_duration_ns": anchor_total,
            "relative_improvement_fraction": derived,
            "direction": ("improved" if derived > 0 else
                          "regressed" if derived < 0 else "neutral"),
            "all_candidate_routes_present": True,
            "all_anchor_routes_present": True,
            "statistic": "sum_exact_route_total_duration_ns",
        },
    }


def _composition_comparison(
        root: Path, pair: composition.CumulativeBuildPair,
        correctness: composition.FullCorrectness, *, effect: float,
) -> composition.IncrementalComparison:
    operation = root / pair.operation_key
    exact_path = operation / "proof/attribution-pair.json"
    off_path = operation / "runner/s1/measurement-graphs-off/result.json"
    on_path = operation / "runner/s1/target-runtime-graphs-on/result.json"
    exact = _exact_carrier(effect, pair)
    off = _measurement(
        pair=pair, anchor=pair.anchor.build_identity, graph_mode="off",
        factor="source_patch", effect=effect)
    on = _measurement(
        pair=pair, anchor=pair.anchor.build_identity, graph_mode="on",
        factor="source_patch", effect=effect)
    exact_sha = _write_receipt(exact_path, exact, "receipt_sha256")
    off_sha = _write_receipt(off_path, off, "result_sha256")
    on_sha = _write_receipt(on_path, on, "result_sha256")
    route_sha = schemas.content_hash(
        exact["exact_duration_comparison"]["candidate_routes"])
    target_sha = composition._target_runtime_frame_sha256(on)
    _write_runner_plan(
        operation, pair, correctness, exact_sha256=exact_sha,
        route_sha256=route_sha, target_sha256=target_sha)
    authority, payload = composition._runner_measurement_authority_uncommitted(
        operation)
    composition._append_authority_event(
        operation, kind="pre_run", operation_key=authority["operation_key"],
        payload=payload)
    return composition.IncrementalComparison.create(
        pair, correctness,
        exact_route_receipt_sha256=exact_sha,
        exact_route_receipt_path=exact_path,
        expected_route_set_sha256=route_sha,
        graphs_off_receipt_sha256=off_sha,
        graphs_off_receipt_path=off_path,
        graphs_on_receipt_sha256=on_sha,
        graphs_on_receipt_path=on_path,
        target_runtime_frame_sha256=target_sha,
        exact_route_effect_fraction=
            exact["exact_duration_comparison"][
                "relative_improvement_fraction"],
        graphs_off_effect_fraction=off["median_relative"],
        graphs_on_effect_fraction=on["median_relative"])


def _measurement(
        *, pair: composition.CumulativeBuildPair,
        anchor: gpu_source_proofs.BuildIdentity, graph_mode: str,
        factor: str, effect: float) -> dict:
    metric_contract = {
        "schema": "epyc.autokernel.matched-test-metric.v1",
        "graph_mode": graph_mode,
    }
    baseline_center = 1_000_000.0
    anchor_samples = [baseline_center] * 9
    candidate_samples = [baseline_center * (1.0 + effect)] * 9
    relative_effects = [
        (value - baseline_center) / baseline_center
        for value in candidate_samples]
    raw_row = {
        "n_threads": 8, "n_batch": 512, "n_ubatch": 512,
        "use_mmap": True, "no_op_offload": 0,
        "split_mode": "layer", "no_kv_offload": False, "poll": 50,
        "n_prompt": 0, "n_gen": 128, "flash_attn": 1,
    }
    body = {
        "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
        "authority": "nonpromotable_candidate_only_discovery",
        "non_promotable": True, "promotion_claim": False,
        "hip_residency_proved": True, "runtime_graphs": graph_mode,
        "baseline_center": baseline_center,
        "candidate_samples": candidate_samples,
        "relative_effects": relative_effects,
        "median_relative": relative_effects[0],
        "baseline_sha256": _sha("recovery-baseline"),
        "factor": factor, "technical_workload": {"tokens": 32},
        "frame": {
            "backend": "llama_gpu", "recipe": "tg128-ngl99",
            "metric": "decode_tokens_per_s",
            "metric_direction": "higher_better",
            "metric_contract": metric_contract,
            "n_prompt": 0, "n_gen": 128,
            "model": "/models/test.gguf", "model_sha256": _sha("model"),
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
    body["result_sha256"] = schemas.content_hash(body)
    return body


def _producer_run(samples: list[float], raw_row: dict,
                  metric_contract: dict, label: str,
                  identity: gpu_source_proofs.BuildIdentity) -> dict:
    diagnostic = {"schema": "epyc.test.native_metric.v1", "arm": label}
    diagnostic["receipt_sha256"] = schemas.content_hash(diagnostic)
    return {
        "metric": sum(samples) / len(samples), "samples": samples,
        "metric_contract": metric_contract, "sample_count": len(samples),
        "raw_row": raw_row,
        "reward_binary_sha256": identity.binary_sha256,
        "hip_library_sha256": identity.hip_library_sha256,
        "native_metric_diagnostic": diagnostic,
        "supervisor": {
            "stdout_sha256": _sha(label + "-stdout"),
            "stderr_sha256": _sha(label + "-stderr"),
        },
    }


class CumulativeControllerIntegrationTests(unittest.TestCase):
    def _scheduled(self, root: Path):
        item = _candidate()
        first = _screen(1, "candidate")
        second = _screen(2, "top_k_replicated_candidate")
        config = controller.ControllerConfig(
            root, max_iterations=10, dry_run=True,
            production_base_commit=BASE, instrument_commit=INSTRUMENT)
        common = {
            "hypothesis_id": item.hypothesis_id,
            "statement": item.statement, "falsifier": item.falsifier,
            "regime": dict(item.regime),
            "proposal_sha256": controller._sha(item.proposal),
            "source_manifest_sha256": item.source_manifest_sha256,
            "candidate_semantic_sha256":
                controller._candidate_semantic_identity(item),
        }
        state = {
            "iterations": [
                {**controller._screen_iteration_fields(first, repetition=1),
                 **common},
                {**controller._screen_iteration_fields(second, repetition=2),
                 **common},
            ],
            "next": 3, "scientific_attempts": 2,
            "attempted_candidate_identities": {},
        }
        controller._schedule_cumulative_composition(
            state, config=config, item=item, result=second)
        return config, state, controller._restore_pending(
            state["pending"], config)

    def _completed_adapter_recovery(self, root: Path):
        config, state, item = self._scheduled(root)
        plan = item.composition_plan
        assert plan is not None
        pair = composition.CumulativeBuildPair.create(
            plan,
            anchor=composition.BuildBinding.create(
                plan.anchor.ordered_patch_set_sha256,
                _identity("recovery-anchor"),
                source_materialization_receipt_sha256=
                    _sha("recovery-anchor-source")),
            candidate=composition.BuildBinding.create(
                plan.candidate.ordered_patch_set_sha256,
                _identity("recovery-candidate"),
                source_materialization_receipt_sha256=
                    _sha("recovery-candidate-source")))
        correctness_body = {"status": "PASS", "cases": 1139}
        correctness_ref = {
            "path": str(root / "proof/correctness.json"),
            "file_sha256": _sha("recovery-correctness-file"),
            "native_sha256": _sha("recovery-correctness-native"),
            "body": correctness_body,
        }
        attribution_body = _exact_carrier(
            .03, pair,
            workload_sha256=_sha("deployment-workload-file"),
            runtime_config_sha256=_sha("deployment-runtime-file"))
        correctness = composition.FullCorrectness.create(
            pair, suite_id="current-gpu-source-full-correctness-v1",
            cases_sha256=schemas.content_hash(correctness_body),
            receipt_sha256=correctness_ref["file_sha256"], passed=True)
        adapter_operation = plan.operation_key
        operations = root / "operations"
        operation = operations / adapter_operation
        operation.mkdir(parents=True)
        attribution_path = operation / "proof/attribution-pair.json"
        attribution_sha = _write_receipt(
            attribution_path, attribution_body, "receipt_sha256")
        attribution_ref = {
            "path": str(attribution_path),
            "file_sha256": attribution_sha,
            "native_sha256": attribution_body["receipt_sha256"],
            "body": attribution_body,
        }
        bundle = gpu_source_proofs.GpuSourceProofBundle.from_validated_paths(
            manifest_sha256=item.source_manifest_sha256,
            candidate=pair.candidate.build_identity,
            anchor=pair.anchor.build_identity,
            workload_sha256=_sha("recovery-workload"),
            correctness=correctness_ref, attribution=attribution_ref)
        gpu_source_evidence._seal(
            operation / "proof/proof-bundle.json", {
                "schema": gpu_source_evidence.SEALED_BUNDLE_SCHEMA,
                "authority": gpu_source_adapter.AUTHORITY,
                "promotion_claim": False, "bundle": bundle.to_dict(),
            })
        authorization = {"claim": "sealed-composition"}
        lease = {"operation_key": adapter_operation,
                 "mode": "hardware-free", "repetition": 1}
        gpu_source_evidence._seal(
            operation / "intent.json", gpu_source_adapter._intent_body(
                operation_key=adapter_operation, candidate=item,
                authorization=authorization, lease=lease))
        production_identity = replace(
            _identity("production"), source_commit=BASE)
        stages = (
            ("measurement-graphs-off", "off", "source_patch", .02,
             pair.anchor.build_identity),
            ("target-runtime-graphs-on", "on", "source_patch", .01,
             pair.anchor.build_identity),
            ("cumulative-vs-production-graphs-on", "on",
             "cumulative_production", .03,
             production_identity),
        )
        outputs = []
        bodies = []
        for name, graph_mode, factor, effect, anchor in stages:
            output = operation / "runner/s1" / name
            output.mkdir(parents=True)
            body = _measurement(
                pair=pair, anchor=anchor, graph_mode=graph_mode,
                factor=factor, effect=effect)
            (output / "result.json").write_text(
                json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
            outputs.append(output)
            bodies.append(body)
        production_descriptor = composition._measurement_descriptor(
            bodies[2], graph_mode="on", candidate=pair.candidate,
            anchor_identity=production_identity,
            factor_name="cumulative_production")
        production = _production(
            frame_sha256=production_descriptor["anchor_frame_sha256"],
            protocol_sha256=
                production_descriptor["protocol_frame_sha256"],
            measurement_receipt_sha256=hashlib.sha256(
                (outputs[2] / "result.json").read_bytes()).hexdigest(),
            model_sha256=production_descriptor["model_sha256"],
            workload_sha256=_sha("deployment-workload-file"),
            runtime_config_sha256=_sha("deployment-runtime-file"),
            observed_workload_sha256=
                production_descriptor["workload_sha256"],
            observed_runtime_config_sha256=
                production_descriptor["runtime_config_sha256"])
        runner_body = {
            "schema": gpu_source_adapter.RUNNER_PLAN_SCHEMA,
            "authority": gpu_source_adapter.AUTHORITY,
            "promotion_claim": False,
            "operation_key": adapter_operation,
            "composition_plan_sha256": plan.plan_sha256,
            "composition_build_pair": pair.to_dict(),
            "composition_correctness": correctness.to_dict(),
            "composition_production_authority": production.to_dict(),
            "composition_exact_route_receipt_sha256": attribution_sha,
            "composition_expected_route_set_sha256": schemas.content_hash(
                attribution_body["exact_duration_comparison"][
                    "candidate_routes"]),
            "composition_target_runtime_frame_sha256":
                composition._target_runtime_frame_sha256(bodies[1]),
            "measurement_graphs_off_output_dir": str(outputs[0]),
            "target_runtime_graphs_on_output_dir": str(outputs[1]),
            "production_graphs_on_output_dir": str(outputs[2]),
            "cumulative_performance_path": str(
                operation / "cumulative-performance.json"),
        }
        gpu_source_evidence._seal(
            operation / "runner-plan.json", runner_body)
        authority, payload = composition._runner_measurement_authority_uncommitted(
            operation)
        composition._append_authority_event(
            operation, kind="pre_run",
            operation_key=authority["operation_key"], payload=payload)
        adapter = object.__new__(
            gpu_source_adapter.GovernedGpuSourceAdapter)
        adapter.operations_root = operations
        adapter.runner_attest = lambda: None
        adapter.reservation_manager = None
        inflight = {
            "operation_key": adapter_operation,
            "candidate": {"candidate": {
                "source_manifest_sha256": item.source_manifest_sha256,
                "composition_plan": plan.to_dict()}},
            "authorization": authorization, "lease": lease,
        }
        return (config, state, item, pair, correctness, bundle, adapter,
                inflight, operation)

    def test_replicated_positive_queues_exact_anchor_plus_one_without_actor(self):
        item = _candidate()
        first = _screen(1, "candidate")
        second = _screen(2, "top_k_replicated_candidate")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = controller.ControllerConfig(
                root, max_iterations=10, dry_run=True,
                production_base_commit=BASE,
                instrument_commit=INSTRUMENT)
            state = {
                "iterations": [
                    {**controller._screen_iteration_fields(first, repetition=1),
                     "hypothesis_id": item.hypothesis_id,
                     "statement": item.statement, "falsifier": item.falsifier,
                     "regime": dict(item.regime),
                     "proposal_sha256": controller._sha(item.proposal),
                     "source_manifest_sha256": item.source_manifest_sha256,
                     "candidate_semantic_sha256":
                         controller._candidate_semantic_identity(item)},
                    {**controller._screen_iteration_fields(second, repetition=2),
                     "hypothesis_id": item.hypothesis_id,
                     "statement": item.statement, "falsifier": item.falsifier,
                     "regime": dict(item.regime),
                     "proposal_sha256": controller._sha(item.proposal),
                     "source_manifest_sha256": item.source_manifest_sha256,
                     "candidate_semantic_sha256":
                         controller._candidate_semantic_identity(item)},
                ],
                "next": 3, "scientific_attempts": 2,
                "attempted_candidate_identities": {},
            }
            controller._schedule_cumulative_composition(
                state, config=config, item=item, result=second)
            pending = state["pending"]
            self.assertEqual(pending["phase"], "cumulative_ready")
            restored = controller._restore_pending(pending, config)
            self.assertIsNotNone(restored.composition_plan)
            plan = restored.composition_plan
            assert plan is not None
            self.assertEqual(plan.anchor.accepted, ())
            self.assertEqual(len(plan.candidate.accepted), 1)
            self.assertEqual(
                plan.candidate.accepted[0].manifest, item.source_manifest)
            self.assertEqual(
                plan.candidate.accepted[0].replications[0].result_sha256,
                first.result_sha256)
            self.assertEqual(
                plan.candidate.accepted[0].replications[1].result_sha256,
                second.result_sha256)
            ledger = composition.CompositionLedger(
                root / "cumulative-composition.json").load()
            self.assertIsNone(ledger["pending"])
            self.assertEqual(
                state["pending"]["candidate"]["composition_plan"],
                plan.to_dict())
            self.assertEqual(ledger["scientific_attempts"], 0)

    def test_current_correctness_failure_rolls_back_once_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            plan = item.composition_plan
            assert plan is not None
            pair = composition.CumulativeBuildPair.create(
                plan,
                anchor=composition.BuildBinding.create(
                    plan.anchor.ordered_patch_set_sha256,
                    _identity("failed-anchor"),
                    source_materialization_receipt_sha256=
                        _sha("failed-anchor-source")),
                candidate=composition.BuildBinding.create(
                    plan.candidate.ordered_patch_set_sha256,
                    _identity("failed-candidate"),
                    source_materialization_receipt_sha256=
                        _sha("failed-candidate-source")))
            correctness = composition.FullCorrectness.create(
                pair, suite_id="full-current-suite-v1",
                cases_sha256=_sha("failed-cases"),
                receipt_sha256=_sha("failed-correctness"), passed=False)
            exc = controller.CumulativeCorrectnessRefusal(
                "current stack correctness failed",
                receipt_path="/sealed/correctness-failed.json",
                receipt_sha256=correctness.receipt_sha256,
                build_pair=pair, correctness=correctness)
            terminal = controller._terminalize_cumulative_refusal(
                config, item, exc)
            row = dict(state["pending"]["row"])
            row["operation_key"] = _sha("controller-correctness-operation")
            state["inflight"] = {"operation_key": row["operation_key"]}
            controller._bind_cumulative_terminal_row(state, row, terminal)
            controller._record_governed_stage_refusal(state, row, exc)
            self.assertEqual(state["scientific_attempts"], 3)
            self.assertEqual(state["iterations"][-1]["status"],
                             "correctness_falsified")
            self.assertEqual(state["iterations"][-1][
                "composition_disposition"], "correctness_rollback")
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json")
            self.assertEqual(ledger.load()["scientific_attempts"], 1)
            self.assertEqual(
                controller._terminalize_cumulative_refusal(
                    config, item, exc), terminal)

    def test_positive_stack_admission_becomes_next_anchor_plus_one(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            first_plan = item.composition_plan
            assert first_plan is not None
            pair = composition.CumulativeBuildPair.create(
                first_plan,
                anchor=composition.BuildBinding.create(
                    first_plan.anchor.ordered_patch_set_sha256,
                    _identity("first-stack-anchor"),
                    source_materialization_receipt_sha256=
                        _sha("first-stack-anchor-source")),
                candidate=composition.BuildBinding.create(
                    first_plan.candidate.ordered_patch_set_sha256,
                    _identity("first-stack-candidate"),
                    source_materialization_receipt_sha256=
                        _sha("first-stack-candidate-source")))
            correctness = composition.FullCorrectness.create(
                pair, suite_id="full-current-suite-v1",
                cases_sha256=_sha("first-stack-cases"),
                receipt_sha256=_sha("first-stack-correctness"), passed=True)
            comparison = _composition_comparison(
                Path(directory).resolve(), pair, correctness, effect=.01)
            performance, performance_ref = _composition_performance(
                Path(directory).resolve(), first_plan, pair,
                correctness, comparison)
            measured = replace(
                _screen(3, "candidate"),
                composition_build_pair=pair,
                composition_correctness=correctness,
                composition_comparison=comparison,
                cumulative_performance=performance,
                cumulative_performance_ref=performance_ref)
            terminal = controller._finalize_cumulative_screen(
                config, item, measured)
            self.assertEqual(terminal["disposition"], "admitted")
            admitted_row = dict(state["pending"]["row"])
            controller._bind_cumulative_terminal_row(
                state, admitted_row, terminal)
            state["iterations"].append(admitted_row)
            state.pop("pending")
            state["scientific_attempts"] = 3
            state["next"] = 4

            second = _second_candidate()
            first_replication = _second_screen(1, "candidate")
            second_replication = _second_screen(
                2, "top_k_replicated_candidate")
            common = {
                "hypothesis_id": second.hypothesis_id,
                "statement": second.statement,
                "falsifier": second.falsifier,
                "regime": dict(second.regime),
                "proposal_sha256": controller._sha(second.proposal),
                "source_manifest_sha256": second.source_manifest_sha256,
                "candidate_semantic_sha256":
                    controller._candidate_semantic_identity(second),
            }
            state["iterations"].extend((
                {**controller._screen_iteration_fields(
                    first_replication, repetition=1), **common},
                {**controller._screen_iteration_fields(
                    second_replication, repetition=2), **common},
            ))
            state["scientific_attempts"] = 5
            state["next"] = 6
            controller._schedule_cumulative_composition(
                state, config=config, item=second,
                result=second_replication)
            next_item = controller._restore_pending(state["pending"], config)
            next_plan = next_item.composition_plan
            assert next_plan is not None
            self.assertEqual(next_plan.anchor, first_plan.candidate)
            self.assertEqual(len(next_plan.anchor.accepted), 1)
            self.assertEqual(len(next_plan.candidate.accepted), 2)
            self.assertEqual(
                next_plan.candidate.accepted[-1].manifest,
                second.source_manifest)
            self.assertEqual(
                next_plan.anchor.ordered_patch_set_sha256,
                first_plan.candidate.ordered_patch_set_sha256)

    def test_precompute_failure_rolls_back_without_spending_science(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            exc = controller.SourceApplyRefusal(
                "stack application interrupted",
                receipt_path="/sealed/source-apply-failed.json",
                receipt_sha256=_sha("source-apply-failed"))
            terminal = controller._terminalize_cumulative_refusal(
                config, item, exc)
            row = dict(state["pending"]["row"])
            row["operation_key"] = _sha("controller-source-operation")
            controller._bind_cumulative_terminal_row(state, row, terminal)
            controller._record_governed_stage_refusal(state, row, exc)
            self.assertEqual(controller._derived_scientific_attempts(state), 2)
            self.assertFalse(terminal["scientific_budget_spent"])
            self.assertEqual(terminal["disposition"],
                             "infrastructure_rollback")
            self.assertEqual(
                composition.CompositionLedger(
                    config.output_root /
                    "cumulative-composition.json").load()[
                        "scientific_attempts"], 0)

    def test_attribution_refusal_rolls_back_scientifically_once(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            plan = item.composition_plan
            assert plan is not None
            pair = composition.CumulativeBuildPair.create(
                plan,
                anchor=composition.BuildBinding.create(
                    plan.anchor.ordered_patch_set_sha256,
                    _identity("attribution-anchor"),
                    source_materialization_receipt_sha256=
                        _sha("attribution-anchor-source")),
                candidate=composition.BuildBinding.create(
                    plan.candidate.ordered_patch_set_sha256,
                    _identity("attribution-candidate"),
                    source_materialization_receipt_sha256=
                        _sha("attribution-candidate-source")))
            correctness = composition.FullCorrectness.create(
                pair, suite_id="full-current-suite-v1",
                cases_sha256=_sha("attribution-cases"),
                receipt_sha256=_sha("attribution-correctness"), passed=True)
            receipt = _sha("attribution-refusal")
            exc = controller.CumulativeAttributionRefusal(
                "exact route authority failed",
                receipt_path="/sealed/attribution-failed.json",
                receipt_sha256=receipt,
                build_pair=pair, correctness=correctness)
            terminal = controller._terminalize_cumulative_refusal(
                config, item, exc)
            row = dict(state["pending"]["row"])
            row["operation_key"] = _sha("controller-attribution-operation")
            state["inflight"] = {"operation_key": row["operation_key"]}
            controller._bind_cumulative_terminal_row(state, row, terminal)
            controller._record_governed_stage_refusal(state, row, exc)
            self.assertEqual(state["scientific_attempts"], 3)
            self.assertEqual(state["iterations"][-1]["status"],
                             "attribution_route_falsified")
            self.assertEqual(terminal["disposition"],
                             "attribution_rollback")
            self.assertEqual(terminal["attribution_receipt_sha256"], receipt)
            self.assertEqual(
                controller._terminalize_cumulative_refusal(
                    config, item, exc), terminal)

    def test_infrastructure_ambiguity_rolls_back_and_requeues_fresh_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            prior_plan = item.composition_plan
            assert prior_plan is not None
            controller_operation = _sha("ambiguous-controller-operation")
            row = dict(state["pending"]["row"])
            state["inflight"] = {
                "operation_key": controller_operation,
                "row": row, "candidate": state["pending"]["candidate"],
                "authorization": {"fixture": "authorization"},
                "confirmation": False, "parent_authorization": None,
                "infrastructure_retry_epoch": 0,
            }
            state.pop("pending")
            exc = controller.ScreenInfrastructureAmbiguity(
                "within-arm output instability",
                receipt_path="/sealed/infrastructure.json",
                receipt_sha256=_sha("infrastructure-receipt"),
                operation_key=controller_operation)
            controller._record_cumulative_infrastructure_ambiguity(
                state, config=config, item=item, row=row, exc=exc)
            self.assertEqual(controller._derived_scientific_attempts(state), 2)
            retry = controller._restore_pending(state["pending"], config)
            self.assertIsNotNone(retry.composition_plan)
            self.assertNotEqual(retry.composition_plan.operation_key,
                                prior_plan.operation_key)
            self.assertEqual(retry.composition_plan.anchor,
                             prior_plan.anchor)
            self.assertEqual(retry.composition_plan.candidate,
                             prior_plan.candidate)
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json").load()
            self.assertEqual(ledger["scientific_attempts"], 0)
            self.assertEqual(ledger["terminals"][0]["disposition"],
                             "infrastructure_rollback")
            self.assertIsNone(ledger["pending"])

    def test_controller_ledger_join_allows_only_owned_crash_window(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            plan = item.composition_plan
            assert plan is not None
            # A controller checkpoint may exist before lazy ledger begin.
            controller._validate_cumulative_composition_state(config, state)
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json")
            ledger.begin(plan)
            # The exact pending plan is owned by the controller checkpoint.
            controller._validate_cumulative_composition_state(config, state)
            pair = composition.CumulativeBuildPair.create(
                plan,
                anchor=composition.BuildBinding.create(
                    plan.anchor.ordered_patch_set_sha256,
                    _identity("join-anchor"),
                    source_materialization_receipt_sha256=
                        _sha("join-anchor-source")),
                candidate=composition.BuildBinding.create(
                    plan.candidate.ordered_patch_set_sha256,
                    _identity("join-candidate"),
                    source_materialization_receipt_sha256=
                        _sha("join-candidate-source")))
            failed = composition.FullCorrectness.create(
                pair, suite_id="full-current-suite-v1",
                cases_sha256=_sha("join-cases"),
                receipt_sha256=_sha("join-correctness"), passed=False)
            ledger.record_build_pair(pair)
            terminal_state = ledger.record_correctness(failed)
            terminal = terminal_state["terminals"][0]
            # The ledger may be one terminal ahead while the exact controller
            # plan is still pending/inflight after a crash.
            controller._validate_cumulative_composition_state(config, state)
            row = dict(state["pending"]["row"])
            controller._bind_cumulative_terminal_row(state, row, terminal)
            state["iterations"].append(row)
            state.pop("pending")
            controller._validate_cumulative_composition_state(config, state)
            state["iterations"][-1]["composition_terminal_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                    controller.DiscoveryControllerError,
                    "differs from its ledger"):
                controller._validate_cumulative_composition_state(config, state)

    def test_unowned_ledger_pending_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config, state, item = self._scheduled(Path(directory).resolve())
            plan = item.composition_plan
            assert plan is not None
            composition.CompositionLedger(
                config.output_root / "cumulative-composition.json").begin(plan)
            state.pop("pending")
            with self.assertRaisesRegex(
                    controller.DiscoveryControllerError,
                    "lacks controller ownership"):
                controller._validate_cumulative_composition_state(config, state)

    def test_adapter_operation_identity_cryptographically_binds_stack_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            _config, state, item = self._scheduled(Path(directory).resolve())
            operation_key = _sha("adapter-composition-operation")
            authorization = {"claim": "sealed"}
            lease = {"operation_key": operation_key, "repetition": 1,
                     "mode": "hardware-free"}
            intent = gpu_source_adapter._intent_body(
                operation_key=operation_key, candidate=item,
                authorization=authorization, lease=lease)
            inflight = {
                "operation_key": operation_key,
                "candidate": state["pending"]["candidate"],
                "authorization": authorization, "lease": lease,
            }
            identity = gpu_source_adapter._inflight_identity(inflight)
            self.assertEqual(
                intent["composition_plan_sha256"],
                item.composition_plan.plan_sha256)
            self.assertEqual(
                identity["composition_plan_sha256"],
                intent["composition_plan_sha256"])
            altered = dict(inflight)
            altered["candidate"] = dict(inflight["candidate"])
            altered["candidate"]["composition_plan"] = dict(
                altered["candidate"]["composition_plan"])
            altered["candidate"]["composition_plan"]["plan_sha256"] = "0" * 64
            with self.assertRaises(gpu_source_adapter.GpuSourceAdapterError):
                gpu_source_adapter._inflight_identity(altered)

    def test_missing_wrapper_reconstructs_typed_screen_and_terminal_once(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._completed_adapter_recovery(
                Path(directory).resolve())
            (config, _state, item, pair, correctness, bundle, adapter,
             inflight, operation) = values
            with mock.patch.object(
                    gpu_source_adapter.evidence,
                    "load_gpu_source_evidence_bundle",
                    return_value=bundle), mock.patch.object(
                    gpu_source_adapter, "_source_frame",
                    return_value=(_sha("recovery-series"), bundle)), \
                    mock.patch.object(
                        gpu_source_adapter.autokernel_progression,
                        "_gpu_screen", return_value={"stage": "candidate"}):
                first = adapter.reconcile(inflight)
                second = adapter.reconcile(inflight)
            self.assertEqual(first.status, "sealed_result")
            self.assertEqual(second, first)
            self.assertTrue((operation / "screen-result.json").is_file())
            journal = operation / composition._AUTHORITY_JOURNAL
            self.assertTrue(journal.is_file())
            rows = composition._read_authority_journal(operation)
            self.assertEqual([row["kind"] for row in rows],
                             ["pre_run", "result"])
            result = first.result
            self.assertEqual(result.composition_build_pair, pair)
            self.assertEqual(result.composition_correctness, correctness)
            self.assertIsNotNone(result.composition_comparison)
            terminal = controller._finalize_cumulative_screen(
                config, item, result)
            repeated = controller._finalize_cumulative_screen(
                config, item, result)
            self.assertEqual(repeated, terminal)
            self.assertEqual(terminal["scientific_budget_spent"], True)
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json").load()
            self.assertEqual(ledger["scientific_attempts"], 1)
            self.assertEqual(len(ledger["terminals"]), 1)

    def test_result_tamper_refuses_against_committed_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._completed_adapter_recovery(
                Path(directory).resolve())
            (config, _state, item, pair, correctness, bundle, adapter,
             inflight, operation) = values
            with mock.patch.object(
                    gpu_source_adapter.evidence,
                    "load_gpu_source_evidence_bundle",
                    return_value=bundle), mock.patch.object(
                    gpu_source_adapter, "_source_frame",
                    return_value=(_sha("recovery-series"), bundle)), \
                    mock.patch.object(
                        gpu_source_adapter.autokernel_progression,
                        "_gpu_screen", return_value={"stage": "candidate"}):
                first = adapter.reconcile(inflight)
            self.assertEqual(first.status, "sealed_result")
            off_result = operation / "runner/s1" / \
                "measurement-graphs-off" / "result.json"
            off_result.write_bytes(off_result.read_bytes() + b" ")
            with mock.patch.object(
                    gpu_source_adapter.evidence,
                    "load_gpu_source_evidence_bundle",
                    return_value=bundle), mock.patch.object(
                    gpu_source_adapter, "_source_frame",
                    return_value=(_sha("recovery-series"), bundle)), \
                    mock.patch.object(
                        gpu_source_adapter.autokernel_progression,
                        "_gpu_screen", return_value={"stage": "candidate"}):
                self.assertEqual(
                    adapter.reconcile(inflight).status, "ambiguous")

    def test_recovery_refuses_swapped_builds_and_partial_result_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._completed_adapter_recovery(
                Path(directory).resolve())
            (_config, _state, item, pair, _correctness, bundle, adapter,
             inflight, operation) = values
            plan = item.composition_plan
            assert plan is not None
            runner = operation / "runner-plan.json"
            runner.unlink()
            swapped = composition.CumulativeBuildPair.create(
                plan,
                anchor=composition.BuildBinding.create(
                    plan.anchor.ordered_patch_set_sha256,
                    pair.candidate.build_identity,
                    source_materialization_receipt_sha256=
                        pair.candidate.source_materialization_receipt_sha256),
                candidate=composition.BuildBinding.create(
                    plan.candidate.ordered_patch_set_sha256,
                    pair.anchor.build_identity,
                    source_materialization_receipt_sha256=
                        pair.anchor.source_materialization_receipt_sha256))
            swapped_correctness = composition.FullCorrectness.create(
                swapped,
                suite_id="current-gpu-source-full-correctness-v1",
                cases_sha256=schemas.content_hash(
                    bundle.correctness["body"]),
                receipt_sha256=bundle.correctness["file_sha256"],
                passed=True)
            gpu_source_evidence._seal(runner, {
                "schema": gpu_source_adapter.RUNNER_PLAN_SCHEMA,
                "authority": gpu_source_adapter.AUTHORITY,
                "promotion_claim": False,
                "operation_key": inflight["operation_key"],
                "composition_plan_sha256": plan.plan_sha256,
                "composition_build_pair": swapped.to_dict(),
                "composition_correctness": swapped_correctness.to_dict(),
                "measurement_graphs_off_output_dir":
                    str(operation / "runner/off"),
                "target_runtime_graphs_on_output_dir":
                    str(operation / "runner/on"),
            })
            with mock.patch.object(
                    gpu_source_adapter.evidence,
                    "load_gpu_source_evidence_bundle",
                    return_value=bundle), mock.patch.object(
                    gpu_source_adapter, "_source_frame",
                    return_value=(_sha("recovery-series"), bundle)), \
                    mock.patch.object(
                        gpu_source_adapter.autokernel_progression,
                        "_gpu_screen", return_value={"stage": "candidate"}):
                self.assertEqual(
                    adapter.reconcile(inflight).status, "ambiguous")

        with tempfile.TemporaryDirectory() as directory:
            values = self._completed_adapter_recovery(
                Path(directory).resolve())
            (_config, _state, _item, _pair, _correctness, bundle, adapter,
             inflight, operation) = values
            with mock.patch.object(
                    gpu_source_adapter.evidence,
                    "load_gpu_source_evidence_bundle",
                    return_value=bundle), mock.patch.object(
                    gpu_source_adapter, "_source_frame",
                    return_value=(_sha("recovery-series"), bundle)), \
                    mock.patch.object(
                        gpu_source_adapter.autokernel_progression,
                        "_gpu_screen", return_value={"stage": "candidate"}):
                recovered = adapter.reconcile(inflight)
                self.assertEqual(recovered.status, "sealed_result")
                wrapper = operation / "screen-result.json"
                raw = json.loads(wrapper.read_text(encoding="utf-8"))
                raw.pop("receipt_sha256")
                for holder in (raw["screen"], raw["receipt_series"][-1]):
                    holder["composition_build_pair"] = None
                    holder["composition_correctness"] = None
                    holder["composition_comparison"] = None
                wrapper.unlink()
                gpu_source_evidence._seal(wrapper, raw)
                self.assertEqual(
                    adapter.reconcile(inflight).status, "ambiguous")

    def test_runner_plan_partial_composition_carrier_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._completed_adapter_recovery(
                Path(directory).resolve())
            (_config, _state, item, _pair, _correctness, _bundle, _adapter,
             inflight, operation) = values
            runner = operation / "runner-plan.json"
            raw = json.loads(runner.read_text(encoding="utf-8"))
            raw.pop("receipt_sha256")
            raw["composition_correctness"] = None
            runner.unlink()
            gpu_source_evidence._seal(runner, raw)
            identity = gpu_source_adapter._inflight_identity(inflight)
            with self.assertRaisesRegex(
                    gpu_source_adapter.GpuSourceAdapterError,
                    "cumulative recovery evidence"):
                gpu_source_adapter._validated_runner_plan(
                    runner, identity,
                    composition_plan=item.composition_plan)

    def test_full_loop_executes_cumulative_third_science_and_rolls_back(self):
        item = _candidate()

        class Planner:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return {**controller.SOL, "runtime": RUNTIME}

            def plan(self, **_kwargs):
                self.calls += 1
                return item

        class Critic:
            def __init__(self):
                self.calls = 0

            def attest(self):
                return {**controller.FABLE5_CRITIC,
                        "runtime": CLAUDE_RUNTIME}

            def review(self, *_args, **_kwargs):
                self.calls += 1
                return controller.Critique("accept", "bounded test acceptance")

        class Lease:
            def admit(self, _item, *, operation_key):
                return {"admitted": True, "operation_key": operation_key,
                        "mode": "hardware-free-fixture"}

        class Screener:
            def __init__(self):
                self.calls = 0

            def reconcile(self, _inflight):
                return controller.Recovery("safe_to_start")

            def screen(self, candidate, _authorization, _permit):
                self.calls += 1
                if self.calls <= 2:
                    return _screen(self.calls, "candidate")
                plan = candidate.composition_plan
                self_outer.assertIsNotNone(plan)
                pair = composition.CumulativeBuildPair.create(
                    plan,
                    anchor=composition.BuildBinding.create(
                        plan.anchor.ordered_patch_set_sha256,
                        _identity("anchor"),
                        source_materialization_receipt_sha256=
                            _sha("anchor-materialization")),
                    candidate=composition.BuildBinding.create(
                        plan.candidate.ordered_patch_set_sha256,
                        _identity("candidate"),
                        source_materialization_receipt_sha256=
                            _sha("candidate-materialization")))
                correctness = composition.FullCorrectness.create(
                    pair, suite_id="full-current-suite-v1",
                    cases_sha256=_sha("cases"),
                    receipt_sha256=_sha("correctness"), passed=True)
                comparison = _composition_comparison(
                    Path(directory).resolve(), pair, correctness,
                    effect=-.01)
                performance, performance_ref = _composition_performance(
                    Path(directory).resolve(), plan, pair, correctness,
                    comparison, cumulative_on=.03)
                return controller.SealedScreen(
                    receipt_path="/sealed/composition/result.json",
                    result_sha256=comparison.result_sha256,
                    effect_fraction=-.01, classification="screened_out",
                    baseline_sha256=_sha("composition-baseline"),
                    source_proof_sha256=_sha("composition-source"),
                    dispatch_proof_sha256=_sha("composition-dispatch"),
                    exact_attribution_effect_fraction=-.01,
                    target_runtime_effect_fraction=-.01,
                    series_key=plan.candidate.ordered_patch_set_sha256,
                    build_identity_sha256=
                        pair.candidate.build_identity_sha256,
                    correctness_receipt_sha256=
                        correctness.receipt_sha256,
                    attribution_receipt_sha256=_sha("exact-route"),
                    graphs_off_receipt_sha256=_sha("graphs-off"),
                    graphs_on_receipt_sha256=_sha("graphs-on"),
                    composition_build_pair=pair,
                    composition_correctness=correctness,
                    composition_comparison=comparison,
                    cumulative_performance=performance,
                    cumulative_performance_ref=performance_ref)

        self_outer = self
        planner, critic, screener = Planner(), Critic(), Screener()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                controller, "_write_projection"):
            config = controller.ControllerConfig(
                Path(directory).resolve(), max_iterations=3)
            state = controller.run_controller(
                config, planner=planner, critic=critic,
                screener=screener, lease=Lease())
            self.assertTrue(state["complete"])
            self.assertEqual(state["scientific_attempts"], 3)
            self.assertEqual(planner.calls, 1)
            self.assertEqual(critic.calls, 1)
            self.assertEqual(screener.calls, 3)
            self.assertEqual(
                [row["status"] for row in state["iterations"]],
                ["candidate", "top_k_replicated_candidate", "screened_out"])
            cumulative = state["iterations"][-1]
            self.assertEqual(
                cumulative["composition_disposition"],
                "incremental_rollback")
            self.assertTrue(
                cumulative["composition_scientific_budget_spent"])
            ledger = composition.CompositionLedger(
                config.output_root / "cumulative-composition.json").load()
            self.assertEqual(ledger["scientific_attempts"], 1)
            self.assertEqual(ledger["authority"]["accepted"], [])
            again = controller.run_controller(
                config, planner=planner, critic=critic,
                screener=screener, lease=Lease())
            self.assertEqual(again, state)
            self.assertEqual((planner.calls, critic.calls, screener.calls),
                             (1, 1, 3))


if __name__ == "__main__":
    unittest.main()
