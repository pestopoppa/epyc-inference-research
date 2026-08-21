from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import os
import shutil
import stat
import subprocess
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import tempfile

from .. import hypothesis_portfolio, preauthored_continuation, source_candidate
from . import discovery_deployment_factory as F
from . import discovery_controller as C


def frozen_production_comparator(
        root: Path, *, production_path: Path | None = None,
        model_path: Path | None = None) -> Path:
    production_path = (
        F.deployment.FROZEN_PRODUCTION_PATH
        if production_path is None else production_path)
    model_path = F._SITE_MODEL if model_path is None else model_path
    production_build = production_path / "build-hip"
    _files, runtime_semantics = F._production_runtime_snapshot(production_path)
    identity = F.gpu_source_proofs.BuildIdentity(
        F.deployment.FROZEN_PRODUCTION_HEAD,
        F.cumulative_composition.FROZEN_PRODUCTION_SOURCE_SHA256,
        F._digest_regular(
            production_build / "bin/llama-bench", "fixture production binary"),
        F._digest_regular(
            (production_build / "bin/libggml-hip.so").resolve(strict=True),
            "fixture production HIP library"),
        F.cumulative_composition.FROZEN_BUILD_RECEIPT_SHA256,
        F.discovery_static_registry._linkage_sha(production_build))
    model_sha = F._digest_regular(model_path, "fixture model")
    protocol = F.cumulative_composition.frozen_production_protocol_binding(
        model_sha256=model_sha,
        build_identity=identity)
    comparator = F.cumulative_composition.FrozenProductionComparator.create(
        build_identity=identity,
        build_receipt_sha256=
            F.cumulative_composition.FROZEN_BUILD_RECEIPT_SHA256,
        linkage_receipt_sha256=
            F.cumulative_composition.FROZEN_LINKAGE_RECEIPT_SHA256,
        runtime_receipt_sha256=
            F.cumulative_composition.FROZEN_RUNTIME_RECEIPT_SHA256,
        runtime_snapshot_sha256=F.schemas.content_hash(runtime_semantics),
        measurement_receipt_sha256=
            F.cumulative_composition.FROZEN_MEASUREMENT_RECEIPT_SHA256,
        model_sha256=model_sha,
        workload_sha256=F._pretty_json_sha256(F._deployment_workload_body()),
        runtime_config_sha256=F._pretty_json_sha256(F._deployment_runtime_body()),
        observed_workload_sha256=protocol["observed_workload_sha256"],
        observed_runtime_config_sha256=
            protocol["observed_runtime_config_sha256"],
        frame_sha256=protocol["frame_sha256"],
        measurement_protocol_sha256=protocol["measurement_protocol_sha256"])
    path = root / "frozen-production-comparator.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(comparator.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return path.resolve()


def disposable_v9_clone(root: Path) -> Path:
    production = root / "production"
    subprocess.run([
        "git", "clone", "--shared", "--no-checkout",
        str(F.deployment.FROZEN_PRODUCTION_PATH), str(production)],
        check=True, capture_output=True)
    for flavor in ("build", "build-hip"):
        subprocess.run([
            "cp", "-a", "--reflink=auto",
            str(F.deployment.FROZEN_PRODUCTION_PATH / flavor),
            str(production / flavor)], check=True, capture_output=True)
    return production


def template(path="ggml/src/ggml-cuda/fattn.cu", symbol="fattn_kernel"):
    return F.ExperimentTemplate("fattn-v1", "gpu_decode", symbol, "backend-fattn",
                                "fattn-dispatch", mock.Mock(), frozenset({path}),
                                {path: frozenset({symbol})}, {"kind": "fattn"})


def planned_source_candidate() -> C.PlannedCandidate:
    relative = "ggml/src/ggml-cuda/fattn.cu"
    symbol = "fattn_kernel"
    patch_bytes = (
        f"diff --git a/{relative} b/{relative}\n"
        f"--- a/{relative}\n+++ b/{relative}\n"
        f"@@ -1 +1 @@ {symbol}\n-old\n+new\n"
    ).encode()
    manifest = source_candidate.SourcePatchManifest(
        campaign_id="ak-test", proposal_id="akp-test", candidate_id="akc-test",
        source_tree="llama.cpp", production_base_commit="0" * 40,
        instrument_commit="1" * 40, change_class="arithmetic",
        declared_files=(relative,), declared_symbols={relative: (symbol,)},
        mechanism_id="manifest-carrier-regression",
        patch_sha256=hashlib.sha256(patch_bytes).hexdigest(), patch_bytes=patch_bytes)
    proposal = {
        "proposal_id": manifest.proposal_id, "change_class": manifest.change_class,
        "change": {"files_and_symbols": [f"{relative}:{symbol}"],
                   "estimated_diff_size": 2}}
    return C.PlannedCandidate(
        "akh-manifest-carrier", "canonical manifest carrier",
        "carrier identity mismatch", {"backend": "gpu"}, proposal, manifest,
        manifest.patch_bundle_sha256)


def template_symbol_authority(registry):
    return {
        template_id: {
            path: sorted(symbols)
            for path, symbols in reviewed.allowed_symbols.items()}
        for template_id, reviewed in sorted(registry.templates.items())}


def mocked_v25_carry_forward():
    erratum = C._q5_lds0_attribution_erratum()
    body = {
        "schema": "epyc.autokernel.discovery_carry_forward.v2",
        "predecessor_state_file_sha256": F._V25_STATE_FILE_SHA256,
        "predecessor_journal_file_sha256": F._V25_JOURNAL_FILE_SHA256,
        "predecessor_state_semantic_sha256": F._V25_STATE_SEMANTIC_SHA256,
        "portfolio_outcomes": {
            "akh-v2-q5-type-specific-dequant": "nominated",
            "akh-v2-q8-quantizer-new-mechanism": "retire",
            "akh-v2-fa-gqa7-pair-tail": "bounded_authoring_skip",
            "akh-v2-rms-direct-load-reduction": "bounded_authoring_skip",
        },
        "candidate_semantic_sha256": sorted({
            *F._V25_CANDIDATE_SEMANTICS,
            erratum["candidate_semantic_sha256"]}),
        "candidate_patch_sha256": sorted({
            "00787755e680f56e82af4f5f2a8ebc7e58f8cc3a84cc806732d62b2019f9916e",
            "7529a82f6210a4a5afe25a7903354d5ed4e32d82185d0a8355956138ae32768f",
            "b40ed0a83b9b2891283736f870c7c07e1d9153eec28bd1f79f51ff3e49581d02",
            "bee615524c8ee6544bcccd4a8ff8b9b337b7fcad047b313bc01701200d7423c0",
            "c38ef7bdab57be586092cb568fac733ca05edb76992551c680cabffc5f0a6bdf",
            "f7ced7defb40d08224b3f904586b732f41f68b23b00241622cea02fe404bdba4",
            "ffc7046ce2758fe4d72aac0fae11d612c48711bed49809c22b44bfd185255942",
            erratum["candidate_patch_sha256"],
        }),
        "cross_campaign_candidate_sha256": sorted({
            "15757d6c7a5466f62f75fcc520d0f5a11f9edd842463438889cc189c8fd141f0",
            "583a174d6dbd04061277ec3802ddfb9cd522143fd29817246760a256732ba51f",
            "93d7eb1790fb57e47db15db811574575515a90b7b59020902a063f99c80063a6",
            "a1055efc62aa516c0aa1feaa31e86e0e4e2c23ed7f925d27815f340450c8a0e3",
            "c64dbe5f27171e45b79301cd8d3702671e9567275c746b9534319dc4c47a37d9",
            "d639dfe011775c87359b1b93afbd08a3f3dd194adbf48fc861ae25099d128e67",
            "dc299de155757172d160f9ac59ccd36e83dacf1d003da73ca3a91a6fe8c364db",
            erratum["cross_campaign_candidate_sha256"],
        }),
        "attribution_expectation_erratum": erratum,
    }
    body["carry_forward_sha256"] = F.schemas.content_hash(body)
    return body


class DeploymentFactoryTests(unittest.TestCase):
    def test_live_shape_manifest_and_policy_share_real_operation_namespace(self):
        candidate = planned_source_candidate()
        with tempfile.TemporaryDirectory() as directory:
            operations = Path(directory).resolve() / "operations"
            operation_key = "a" * 64
            operation_root = operations / operation_key
            operation_root.mkdir(mode=0o700, parents=True)
            config = SimpleNamespace(operations_root=operations)

            synchronized_modes = []
            original_fsync = F.os.fsync
            def recording_fsync(descriptor):
                synchronized_modes.append(os.fstat(descriptor).st_mode)
                return original_fsync(descriptor)
            with mock.patch.object(F.os, "fsync", side_effect=recording_fsync):
                manifest_file = F._manifest_file_for_operation(
                    config, candidate, operation_key)
            expected = source_candidate.source_patch_manifest_bytes(
                candidate.source_manifest)
            self.assertEqual(manifest_file.path.parent, operation_root)
            self.assertEqual(manifest_file.path.read_bytes(), expected)
            self.assertEqual(hashlib.sha256(expected).hexdigest(),
                             candidate.source_manifest_sha256)
            self.assertEqual(json.loads(expected)["schema"],
                             source_candidate.SCHEMA_SOURCE_PATCH)
            self.assertTrue(any(stat.S_ISREG(mode) for mode in synchronized_modes))
            self.assertTrue(any(stat.S_ISDIR(mode) for mode in synchronized_modes))

            policy = b'{"schema":"test-policy"}'
            policy_path = F._write_operation_carrier(
                config, operation_key, "evidence-policy.json", policy,
                "sealed evidence policy")
            self.assertEqual(policy_path.parent, operation_root)
            self.assertEqual(policy_path.read_bytes(), policy)
            build_key = "b" * 64
            reopened = F._manifest_file(
                config, candidate, SimpleNamespace(
                    operation_key=operation_key, build_key=build_key))
            self.assertEqual(reopened.path, manifest_file.path)
            self.assertEqual(reopened.path.read_bytes(), expected)
            self.assertFalse((operations / build_key).exists())
            self.assertFalse((operations / "materialization").exists())

            for index, link in enumerate(("hardlink", "symlink"), start=1):
                unsafe_key = str(index) * 64
                unsafe_root = operations / unsafe_key
                unsafe_root.mkdir(mode=0o700)
                target = Path(directory).resolve() / f"{link}-target.json"
                target.write_bytes(expected)
                target.chmod(0o600)
                carrier = unsafe_root / "source-manifest.json"
                if link == "hardlink":
                    os.link(target, carrier)
                else:
                    carrier.symlink_to(target)
                with self.subTest(link=link), self.assertRaisesRegex(
                        F.DeploymentFactoryError,
                        "(single-link regular file|reopened safely)"):
                    F._manifest_file_for_operation(config, candidate, unsafe_key)

    def test_manifest_carrier_refuses_missing_or_escaped_operation_namespace(self):
        candidate = planned_source_candidate()
        with tempfile.TemporaryDirectory() as directory:
            operations = Path(directory).resolve() / "operations"
            operations.mkdir()
            config = SimpleNamespace(operations_root=operations)
            with self.assertRaisesRegex(F.DeploymentFactoryError,
                                        "operation carrier root"):
                F._manifest_file_for_operation(config, candidate, "a" * 64)
            insecure = operations / ("c" * 64)
            insecure.mkdir(mode=0o700)
            insecure.chmod(0o755)
            with self.assertRaisesRegex(F.DeploymentFactoryError,
                                        "private owner directory"):
                F._manifest_file_for_operation(config, candidate, "c" * 64)
            outside = Path(directory).resolve() / "outside"
            outside.mkdir()
            (operations / ("b" * 64)).symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(F.DeploymentFactoryError,
                                        "operation carrier root"):
                F._manifest_file_for_operation(config, candidate, "b" * 64)

    def test_manifest_carrier_pins_leaf_and_parent_namespace_against_races(self):
        candidate = planned_source_candidate()
        expected = source_candidate.source_patch_manifest_bytes(
            candidate.source_manifest)

        for attack in ("inode-replacement", "escaping-symlink", "late-hardlink"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                operations = root / "operations"
                operation_key = "d" * 64
                operation = operations / operation_key
                operation.mkdir(mode=0o700, parents=True)
                config = SimpleNamespace(operations_root=operations)
                carrier = operation / "source-manifest.json"
                original_reader = F._read_operation_carrier
                attacked = False

                def mutate_after_fstat(descriptor, label):
                    nonlocal attacked
                    result = original_reader(descriptor, label)
                    if not attacked:
                        attacked = True
                        if attack == "inode-replacement":
                            carrier.unlink()
                            carrier.write_bytes(expected)
                            carrier.chmod(0o600)
                        elif attack == "escaping-symlink":
                            outside = root / "outside-manifest.json"
                            outside.write_bytes(expected)
                            carrier.unlink()
                            carrier.symlink_to(outside)
                        else:
                            os.link(carrier, root / "late-hardlink.json")
                    return result

                with mock.patch.object(
                        F, "_read_operation_carrier",
                        side_effect=mutate_after_fstat), self.assertRaisesRegex(
                            F.DeploymentFactoryError, "directory entry"):
                    F._manifest_file_for_operation(config, candidate, operation_key)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            operations = root / "operations"
            operation_key = "e" * 64
            operation = operations / operation_key
            operation.mkdir(mode=0o700, parents=True)
            outside = root / "outside"
            outside.mkdir(mode=0o700)
            parked = root / "parked-operation"
            config = SimpleNamespace(operations_root=operations)
            original_open = F.os.open
            attacked = False

            def replace_directory_before_leaf(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal attacked
                if path == "source-manifest.json" and dir_fd is not None and not attacked:
                    attacked = True
                    operation.rename(parked)
                    operation.symlink_to(outside, target_is_directory=True)
                return original_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(F.os, "open",
                                   side_effect=replace_directory_before_leaf), \
                    self.assertRaisesRegex(F.DeploymentFactoryError,
                                           "parent chain changed"):
                F._manifest_file_for_operation(config, candidate, operation_key)
            self.assertFalse((outside / "source-manifest.json").exists())

    def test_v3_templates_are_extracted_from_exact_sealed_profile(self):
        self.assertEqual(
            hashlib.sha256(F._PROFILE_TRACE_RECEIPT.read_bytes()).hexdigest(),
            F._PROFILE_TRACE_RECEIPT_SHA256)
        self.assertEqual(
            hashlib.sha256(F._PROFILE_TRACE_CSV.read_bytes()).hexdigest(),
            F._PROFILE_TRACE_CSV_SHA256)
        self.assertEqual(
            hashlib.sha256(F._PROFILE_V3_TRACE_CSV.read_bytes()).hexdigest(),
            F._PROFILE_V3_TRACE_CSV_SHA256)
        rows = F.evidence._load_dispatches(
            F._PROFILE_V3_TRACE_CSV,
            profiler_trace_schema_id=F.evidence.ROCPROF_V3_TRACE_ID,
            expected_rows=59_925)
        self.assertEqual(len(rows), 59_925)
        registry = F._template_registry()
        self.assertEqual(registry.version, "gpu-source-templates-v4")
        self.assertEqual(len(registry.templates), 15)
        for template_id, reviewed in registry.templates.items():
            with self.subTest(template=template_id):
                reduced = F.evidence._reduce_arm(
                    rows, exact=reviewed.dispatch.anchor_exact,
                    forbidden=reviewed.dispatch.anchor_forbidden,
                    invariants=reviewed.dispatch.invariants)
                self.assertEqual(set(reduced["exact"]),
                                 {item.signature for item in reviewed.dispatch.anchor_exact})

    def test_quantize_and_vecdot_templates_have_exact_source_and_correctness_authority(self):
        registry = F._template_registry()
        quantize = registry.templates["cuda-quantize-q8-v1"]
        vecdot = registry.templates["cuda-vecdotq-v1"]
        self.assertEqual(quantize.allowed_files,
                         frozenset({"ggml/src/ggml-cuda/quantize.cu"}))
        self.assertEqual(quantize.semantics["correctness_op"], "MUL_MAT")
        self.assertEqual(quantize.semantics["expected_correctness_cases"], 1139)
        self.assertEqual(len(quantize.dispatch.anchor_exact), 2)
        self.assertEqual(vecdot.allowed_files,
                         frozenset({"ggml/src/ggml-cuda/vecdotq.cuh"}))
        self.assertEqual(vecdot.semantics["correctness_op"], "MUL_MAT")
        self.assertEqual(vecdot.semantics["expected_correctness_cases"], 1139)
        self.assertTrue(any("vec_dot_q5_0_q8_1" in symbol
                            for symbol in vecdot.allowed_symbols[next(iter(vecdot.allowed_files))]))
        self.assertEqual(vecdot.semantics["planner_target_exclusions"], [{
            "kernel_pattern": vecdot.dispatch.anchor_exact[3].kernel_pattern,
            "calls": 129, "grid": 57344, "workgroup": 128,
            "lds_bytes": 512,
            "reason": "Q5 false/true tail is not the reviewed true/true dequant route"}])
        q5_true = vecdot.dispatch.anchor_exact[:3]
        self.assertEqual(sum(row.calls for row in q5_true), 13_803)

    def test_v26_portfolio_dispatch_authority_round_trips_real_trace(self):
        portfolio = hypothesis_portfolio.load(hypothesis_portfolio.DEFAULT_PORTFOLIO)
        registry = F._template_registry()
        surfaces = F._normalized_template_surfaces(registry, portfolio)
        authority = F._portfolio_dispatch_authority(registry, portfolio)
        expected_authority = {
            "akh-v2-q5-onewave-preauthored": [
                (6063, 57344), (4644, 8192), (3096, 311296)],
            "akh-v26-q4k-branchless-sixbit-scale": [(1548, 114688)],
            "akh-v26-rms-scale-broadcast": [(6321, 256)],
            "akh-v26-rope-neox-index-strength-reduction": [
                (3096, 512), (3096, 3584)],
            "akh-v26-fa-combine-wave-normalization": [(3096, 896)],
            "akh-v26-q6k-packed-decode": [(1548, 114688)],
            "akh-v26-fa-gqa7-common-map": [(3096, 7168), (3096, 896)],
        }
        self.assertEqual({
            hypothesis_id: [(row["calls"], row["grid"]) for row in rows]
            for hypothesis_id, rows in authority.items()
        }, expected_authority)
        self.assertEqual(surfaces["cuda-vecdotq-v1"]["excluded_signatures"],
                         [{"route_id": "cuda-vecdotq-v1.anchor.3",
                           "calls": 129, "grid": 57344,
                           "workgroup": 128, "lds_bytes": 512}])
        rows = F.evidence._load_dispatches(
            F._PROFILE_V3_TRACE_CSV,
            profiler_trace_schema_id=F.evidence.ROCPROF_V3_TRACE_ID,
            expected_rows=59_925)
        bindings = (
            ("akh-v26-q4k-branchless-sixbit-scale", "cuda-vecdotq-q4k-v1"),
            ("akh-v26-rms-scale-broadcast", "cuda-norm-v2"),
            ("akh-v26-rope-neox-index-strength-reduction", "cuda-rope-v2"),
            ("akh-v26-fa-combine-wave-normalization", "cuda-fattn-combine-v1"),
            ("akh-v26-q6k-packed-decode", "cuda-vecdotq-q6k-v1"),
            ("akh-v26-fa-gqa7-common-map", "cuda-fattn-gqa7-common-v1"),
        )
        for hypothesis_id, template_id in bindings:
            template = registry.templates[template_id]
            expected = tuple(C.BoundedDispatchExpectation(**row)
                             for row in authority[hypothesis_id])
            intent = C.GpuSourceExperimentIntent(
                template.template_id, template.target_surface,
                template.target_symbol, template.correctness_id,
                template.dispatch_id, expected)
            anchor = F.evidence._reduce_arm(
                rows, exact=template.dispatch.anchor_exact,
                forbidden=template.dispatch.anchor_forbidden,
                invariants=template.dispatch.invariants)
            self.assertEqual(len(anchor["exact"]), len(expected))
            bound = template.bind_dispatch(intent)
            if template_id == "cuda-fattn-gqa7-common-v1":
                self.assertEqual(
                    [(row.calls, row.grid) for row in bound.candidate_exact],
                    [(3096, 3072), (3096, 1024), (3096, 896)])
                continue
            reduced = F.evidence._reduce_arm(
                rows, exact=bound.candidate_exact,
                forbidden=bound.candidate_forbidden,
                invariants=bound.invariants)
            self.assertEqual(len(reduced["exact"]), len(expected))

    def test_rocprofv3_policy_and_per_arm_cardinality_cover_all_six_strategies(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                operations_root=Path(directory).resolve() / "operations")
            policy = F._rocprof_v3_policy(config)
        roles = {item.role for item in policy}
        self.assertTrue(F.evidence.PROFILER_MAPPED_ROLES.issubset(roles))
        self.assertTrue({"executable", "profiler_wrapper", "profiler_package",
                         "profiler_runtime_manifest",
                         "profiler_aqlprofile_manifest",
                         "profiler_libpci_manifest"}.issubset(roles))
        portfolio = hypothesis_portfolio.load(hypothesis_portfolio.DEFAULT_PORTFOLIO)
        registry = F._template_registry()
        authority = F._portfolio_dispatch_authority(registry, portfolio)
        observed = {}
        for record in portfolio.eligible_hypotheses():
            template_id = record["current_bundle_eligibility"]["template_ids"][0]
            reviewed = registry.templates[template_id]
            intent = C.GpuSourceExperimentIntent(
                reviewed.template_id, reviewed.target_surface,
                reviewed.target_symbol, reviewed.correctness_id,
                reviewed.dispatch_id,
                tuple(C.BoundedDispatchExpectation(**row)
                      for row in authority[record["hypothesis_id"]]))
            observed[template_id] = F._expected_rocprofv3_rows(
                reviewed.bind_dispatch(intent))
        self.assertEqual(observed, {
            "cuda-mmvq-q5-onewave-continuation-v1": (59_925, 59_925),
            "cuda-vecdotq-q4k-v1": (59_925, 59_925),
            "cuda-norm-v2": (59_925, 59_925),
            "cuda-rope-v2": (59_925, 59_925),
            "cuda-fattn-combine-v1": (59_925, 59_925),
            "cuda-vecdotq-q6k-v1": (59_925, 59_925),
            "cuda-fattn-gqa7-common-v1": (63_021, 59_925),
        })

    def test_balanced_arm_schedule_is_seeded_and_s2_exactly_reverses_s1(self):
        first_seed, first = F._arm_order_schedule(
            deployment_config_sha256="a" * 64,
            source_manifest_sha256="b" * 64, repetition=1)
        second_seed, second = F._arm_order_schedule(
            deployment_config_sha256="a" * 64,
            source_manifest_sha256="b" * 64, repetition=2)
        self.assertEqual(first_seed, second_seed)
        self.assertEqual(second.split(","), list(reversed(first.split(","))))
        self.assertEqual(len(first_seed), 64)

    def test_all_four_portfolio_strategies_materialize_exact_runner_cli(self):
        """Every live strategy must parse and preflight both runner stages."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model.gguf"; model.write_bytes(b"model")
            def make_build(name, hip_bytes):
                build_root = root / name
                bindir = build_root / "bin"; bindir.mkdir(parents=True)
                cache = build_root / "CMakeCache.txt"
                cache.write_text(
                    "GGML_HIP_GRAPHS:BOOL=ON\n"
                    "GGML_HIP_ROCWMMA_FATTN:BOOL=ON\n"
                    "GGML_HIP_MMQ_MFMA:BOOL=OFF\n", encoding="utf-8")
                binary = bindir / "llama-bench"
                binary.write_bytes(b"shared-binary"); binary.chmod(0o755)
                versioned = bindir / "libggml-hip.so.0.16.0"
                versioned.write_bytes(hip_bytes)
                (bindir / "libggml-hip.so.0").symlink_to(versioned.name)
                (bindir / "libggml-hip.so").symlink_to("libggml-hip.so.0")
                return build_root
            anchor_build = make_build("anchor-build", b"anchor-hip")
            candidate_build = make_build("candidate-build", b"candidate-hip")
            production_build = make_build(
                "production/build-hip", b"production-hip")
            policy_path = root / "admission-policy.json"
            policy_path.write_text("{}\n", encoding="utf-8")
            corpus = SimpleNamespace(
                version="test-v1", policy_sha256="a" * 64,
                file_sha256=hashlib.sha256(
                    policy_path.read_bytes()).hexdigest())
            config = SimpleNamespace(
                operations_root=root / "operations",
                config_sha256="b" * 64,
                admission_policy=SimpleNamespace(
                    corpus=corpus,
                    input=SimpleNamespace(
                        path=policy_path,
                        sha256=corpus.file_sha256)),
                planner_context=SimpleNamespace(
                    value={"context_sha256": "c" * 64}),
                model=SimpleNamespace(path=model),
                production_path=root / "production",
                inference_window_lock=root / "model-call.lock",
                device_id="mi210_0")
            def identity(build_root, commit):
                return C.gpu_source_proofs.BuildIdentity(
                    source_commit=commit,
                    source_sha256=hashlib.sha256(
                        (commit + "-source").encode()).hexdigest(),
                    binary_sha256=hashlib.sha256(b"shared-binary").hexdigest(),
                    hip_library_sha256=hashlib.sha256(
                        (build_root / "bin/libggml-hip.so").resolve().read_bytes()
                    ).hexdigest(),
                    config_sha256=hashlib.sha256(
                        (build_root / "CMakeCache.txt").read_bytes()).hexdigest(),
                    linkage_sha256=F.discovery_static_registry._linkage_sha(
                        build_root))
            build = SimpleNamespace(
                operation_key="d" * 64,
                anchor_build=anchor_build,
                candidate_build=candidate_build,
                measurement_binary=anchor_build / "bin/llama-bench",
                common_loader_dir=anchor_build / "bin",
                anchor_loader_dir=anchor_build / "bin",
                candidate_loader_dir=candidate_build / "bin",
                anchor_identity=identity(
                    anchor_build, F.controller.gpu_discovery.READY_CONTINUE_INSTRUMENT_COMMIT),
                candidate_identity=identity(candidate_build, "6" * 40))
            production_identity = identity(
                production_build, F.deployment.FROZEN_PRODUCTION_HEAD)
            production_authority = \
                F.cumulative_composition.FrozenProductionAuthority.create(
                    production_commit=F.deployment.FROZEN_PRODUCTION_HEAD,
                    build_identity=production_identity,
                    runtime_snapshot_sha256="7" * 64,
                    comparator_receipt_sha256="8" * 64,
                    graphs_mode="graphs_on", frame_sha256="9" * 64,
                    measurement_protocol_sha256="a" * 64,
                    measurement_receipt_sha256="b" * 64,
                    model_sha256="c" * 64, workload_sha256="d" * 64,
                    runtime_config_sha256="e" * 64,
                    observed_workload_sha256="f" * 64,
                    observed_runtime_config_sha256="0" * 64,
                    metric="tokens_per_second",
                    direction="higher_is_better")
            effective = F.schemas.content_hash({
                "planner_context_sha256": "c" * 64,
                "admission_policy_sha256": corpus.policy_sha256,
                "admission_policy_version": corpus.version})
            decision = {
                "effective_context_sha256": effective,
                "mode": "cold_serialized",
                "request": {
                    "model_path": str(model),
                    "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    "model_bytes": model.stat().st_size,
                    "workload": "decode_tg128", "calls_per_arm": 9,
                    "device_id": "mi210_0"}}
            binding = F._runner_binding(config)
            for index, (hypothesis_id, _template, _op, _cases) in enumerate((
                    ("akh-v2-q5-type-specific-dequant", "cuda-vecdotq-v1", "MUL_MAT", 1139),
                    ("akh-v2-q8-quantizer-new-mechanism", "cuda-quantize-q8-v1", "MUL_MAT", 1139),
                    ("akh-v2-fa-gqa7-pair-tail", "cuda-fattn-tile-v1", "FLASH_ATTN_EXT", 2868),
                    ("akh-v2-rms-direct-load-reduction", "cuda-norm-v2", "RMS_NORM", 21),
            ), start=1):
                candidate = SimpleNamespace(
                    hypothesis_id=hypothesis_id,
                    source_manifest_sha256=f"{index}" * 64,
                    composition_plan=(object() if index == 4 else None))
                current_build = (
                    SimpleNamespace(
                        **vars(build),
                        composition_production_authority=
                        production_authority)
                    if candidate.composition_plan is not None else build)
                with mock.patch.object(
                        F.gpu_load_admission, "validate_decision_receipt"):
                    args = binding.build(candidate, current_build, {
                        "operation_key": build.operation_key,
                        "repetition": 1,
                        "load_admission": decision})
                args = F._bind_runner_runtime_authority(
                    config, current_build,
                    {"load_admission": decision, "repetition": 1}, args)
                target = args._target_runtime_args
                self.assertEqual(args._operation_key, build.operation_key)
                self.assertEqual(target._operation_key, build.operation_key)
                self.assertEqual(args._operations_root,
                                 str(config.operations_root))
                self.assertEqual(target._operation_repetition, 1)
                self.assertEqual((args.runtime_graphs, target.runtime_graphs),
                                 ("off", "on"))
                self.assertEqual(args.load_admission_decision_path,
                                 str(root / "operations" / build.operation_key /
                                     "runner/s1/load-admission-decision.json"))
                self.assertEqual(args.load_admission_policy_path,
                                 str(policy_path))
                self.assertEqual(args.load_admission_policy_file_sha256,
                                 corpus.file_sha256)
                self.assertEqual(args.load_admission_effective_context_sha256,
                                 decision["effective_context_sha256"])
                self.assertEqual(
                    args._sealed_candidate_source_build_identity,
                    build.candidate_identity.__dict__)
                self.assertEqual(
                    target._sealed_anchor_source_build_identity,
                    build.anchor_identity.__dict__)
                with mock.patch.object(
                        F.controller.gpu_discovery.gpu_load_admission,
                        "validate_decision_receipt"):
                    off = F.controller.gpu_discovery.preflight(args)
                    on = F.controller.gpu_discovery.preflight(target)
                self.assertEqual((off["runtime_graphs"], on["runtime_graphs"]),
                                 ("off", "on"))
                self.assertEqual(off["operation_key"], build.operation_key)
                self.assertEqual(on["operation_key"], build.operation_key)
                self.assertNotEqual(
                    off["anchor_identity"]["source_commit"],
                    off["candidate_identity"]["source_commit"])
                production = getattr(args, "_production_graphs_on_args", None)
                if candidate.composition_plan is None:
                    self.assertIsNone(production)
                else:
                    self.assertEqual(production.factor,
                                     "cumulative_production")
                    self.assertEqual(production.runtime_graphs, "on")
                    self.assertIsNone(production.measurement_binary)
                    self.assertEqual(
                        production._frozen_production_authority,
                        production_authority.to_dict())
                    self.assertEqual(
                        production._sealed_anchor_source_build_identity,
                        production_identity.__dict__)
                    with mock.patch.object(
                            F.controller.gpu_discovery.gpu_load_admission,
                            "validate_decision_receipt"):
                        production_preflight = \
                            F.controller.gpu_discovery.preflight(production)
                    self.assertEqual(
                        production_preflight["sole_factor"]["name"],
                        "cumulative_production")
                    self.assertIsNone(
                        production_preflight["runtime_arms"])
                    production.factor = "source_patch"
                    with self.assertRaisesRegex(
                            F.DeploymentFactoryError,
                            "comparator authority"):
                        F._bind_runner_runtime_authority(
                            config, current_build,
                            {"load_admission": decision,
                             "repetition": 1}, args)

    def test_argparse_nonzero_is_an_ordinary_resumable_operation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = SimpleNamespace(
                operations_root=root / "operations", config_sha256="a" * 64,
                admission_policy=SimpleNamespace(
                    corpus=SimpleNamespace(
                        version="test-v1", policy_sha256="b" * 64,
                        file_sha256="c" * 64),
                    input=SimpleNamespace(
                        path=root / "policy.json", sha256="c" * 64)),
                planner_context=SimpleNamespace(
                    value={"context_sha256": "d" * 64}),
                model=SimpleNamespace(path=root / "model"),
                inference_window_lock=root / "lock", device_id="mi210_0")
            binding = F._runner_binding(config)
            parser = mock.Mock()
            parser.parse_args.side_effect = SystemExit(2)
            with mock.patch.object(F.gpu_load_admission,
                                   "validate_decision_receipt"), \
                    mock.patch.object(F.controller.gpu_discovery, "parser",
                                      return_value=parser), self.assertRaisesRegex(
                        C.ResumableScreenInterruption,
                        "parser refused with exit 2"):
                binding.build(
                    SimpleNamespace(source_manifest_sha256="e" * 64),
                    SimpleNamespace(
                        operation_key="f" * 64, anchor_build=root / "anchor",
                        candidate_build=root / "candidate",
                        measurement_binary=root / "bench",
                        common_loader_dir=root / "common",
                        anchor_loader_dir=root / "anchor-lib",
                        candidate_loader_dir=root / "candidate-lib"),
                    {"operation_key": "f" * 64, "repetition": 1,
                     "load_admission": {"effective_context_sha256": "0" * 64}})

    def static_config(self, root: Path):
        production = F.deployment.FROZEN_PRODUCTION_PATH
        package = root / "codex-package"
        wrapper = package / "bin/codex.js"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/sh\nexit 77\n", encoding="utf-8")
        wrapper.chmod(0o700)
        critic_wrapper = root / "claude-fable5"
        critic_wrapper.write_text("#!/bin/sh\nexit 77\n", encoding="utf-8")
        critic_wrapper.chmod(0o700)
        native = package / F.codex_container_actor.CODEX_NATIVE_RELATIVE
        native.parent.mkdir(parents=True)
        native.write_bytes(b"native")
        native.chmod(0o700)
        host = native.with_name(F.codex_container_actor.CODE_MODE_HOST_NAME)
        host.write_bytes(b"host")
        host.chmod(0o700)
        docker = root / "docker"
        docker.write_bytes(b"docker")
        docker.chmod(0o700)
        ca = root / "ca.pem"
        ca.write_bytes(b"certificate")
        model = root / "model.gguf"
        model.write_bytes(b"small model")
        workload = root / "workload.json"
        workload.write_text(
            json.dumps(F._deployment_workload_body(), sort_keys=True, indent=2)
            + "\n", encoding="utf-8")
        runtime = root / "runtime.json"
        runtime.write_text(
            json.dumps(F._deployment_runtime_body(), sort_keys=True, indent=2)
            + "\n", encoding="utf-8")
        policy = root / "policy.json"
        policy.write_text("{}", encoding="utf-8")
        planner = root / "planner.json"
        planner.write_text("{}", encoding="utf-8")
        state, evidence, operations, builds, locks = (root / name for name in
                                               ("state", "evidence", "operations", "builds", "locks"))
        for path in (state.parent, locks):
            path.mkdir(parents=True, exist_ok=True)
        immutable = lambda path: SimpleNamespace(
            path=path.resolve(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        registry_sha = F.static_template_registry_sha256()
        portfolio = hypothesis_portfolio.load(hypothesis_portfolio.DEFAULT_PORTFOLIO)
        templates = F._template_registry()
        surfaces = F._normalized_template_surfaces(templates, portfolio)
        source_body = {"schema": "epyc.autokernel.reviewed_source_package.v1",
                       "instrument_commit": F._INSTRUMENT_COMMIT,
                       "files": [{"relative_path": path,
                                  "sha256": F._TARGET_SOURCE_SHA256[path],
                                  "workspace_path": f"reviewed-source/{path}"}
                                 for path in sorted(F._TARGET_SOURCE_SHA256)]}
        portfolio_input = immutable(hypothesis_portfolio.DEFAULT_PORTFOLIO)
        continuation = preauthored_continuation.load(
            preauthored_continuation.DEFAULT_CARRIER)
        erratum_path = root / "q5-lds0-attribution-erratum-v1.json"
        shutil.copyfile(C._Q5_LDS0_ERRATUM_CARRIER, erratum_path)
        erratum_path.chmod(0o600)
        carry_forward_root = root / "v25-carry-forward"
        carry_forward_root.mkdir(mode=0o700)
        portfolio_evidence = {
            row["evidence_id"]: row for row in portfolio.body["evidence"]}
        carry_forward_evidence = {}
        carry_forward_ids = {
            "ev-v25-terminal-state", "ev-v25-terminal-journal",
            *(f"ev-v25-source-manifest-turn{turn:02d}"
              for turn in (2, 4, 6, 8, 12, 13, 14)),
        }
        for receipt in continuation.historical_receipts:
            carry_forward_ids.update({
                receipt[f"{kind}_evidence_id"]
                for kind in ("receipt", "stdout", "stderr", "binary")})
        for evidence_id in sorted(carry_forward_ids):
            source = portfolio_evidence[evidence_id]
            target = carry_forward_root / f"{evidence_id}.json"
            shutil.copyfile(source["path"], target)
            target.chmod(0o600)
            carry_forward_evidence[evidence_id] = {
                "path": str(target.resolve()), "sha256": source["sha256"]}
        evidence_manifest = {
            "manifest_sha256": "f" * 64,
            "evidence": carry_forward_evidence,
        }
        historical_evidence = F._preauthored_historical_evidence(
            continuation, carry_forward_evidence)
        carry_forward = mocked_v25_carry_forward()
        carry_forward_path = root / "discovery-carry-forward-v2.json"
        carry_forward_path.write_text(
            json.dumps(carry_forward, sort_keys=True,
                       separators=(",", ":")) + "\n", encoding="utf-8")
        carry_forward_path.chmod(0o600)
        carry_forward_input = immutable(carry_forward_path)
        comparator_input = immutable(
            frozen_production_comparator(
                root / "production-authority", production_path=production,
                model_path=model))
        config = SimpleNamespace(
            config_sha256="c" * 64, production_path=production.resolve(),
            production_branch=F.deployment.FROZEN_PRODUCTION_BRANCH,
            production_head=F.deployment.FROZEN_PRODUCTION_HEAD,
            instrument_path=F._INSTRUMENT_PATH,
            instrument_commit=F._INSTRUMENT_COMMIT,
            instrument_branch=F._INSTRUMENT_BRANCH,
            state_root=state.resolve(),
            evidence_root=evidence.resolve(), operations_root=operations.resolve(),
            build_root=builds.resolve(),
            max_iterations=2, nomination_threshold=.03,
            actor_wrapper=immutable(wrapper), critic_wrapper=immutable(critic_wrapper),
            environment_profile_id="sealed-codex",
            device_id="mi210_0", claim_timeout_s=0.0,
            inference_window_lock=(locks / "window.lock").resolve(),
            model=immutable(model), workload=immutable(workload), runtime_config=immutable(runtime),
            frozen_production_comparator=comparator_input,
            policy=immutable(policy),
            admission_policy=SimpleNamespace(value={"policy_sha256": "a" * 64},
                corpus=SimpleNamespace(profiles=(SimpleNamespace(
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    model_path=str(model.resolve()), model_bytes=model.stat().st_size,
                    workload="decode_tg128", calls_per_arm=9, device_id="mi210_0"),))),
            planner_context=SimpleNamespace(value={
                "context_sha256": "b" * 64,
                "reviewed_source_package_sha256": F.schemas.content_hash(source_body),
                "template_registry_sha256": registry_sha,
                "template_symbol_authority": template_symbol_authority(templates),
                "template_surfaces": surfaces,
                "template_surfaces_sha256": F.schemas.content_hash(surfaces),
                "portfolio_dispatch_authority": F._portfolio_dispatch_authority(
                    templates, portfolio),
                "preauthored_continuation_sha256": continuation.sha256,
                "preauthored_source_backed_diff_sha256":
                    continuation.source_backed_diff_sha256,
                "preauthored_historical_evidence_sha256":
                    historical_evidence["receipt_sha256"]}),
            hypothesis_portfolio=SimpleNamespace(value=portfolio, input=portfolio_input),
            preauthored_continuation=SimpleNamespace(
                value=continuation,
                input=immutable(preauthored_continuation.DEFAULT_CARRIER)),
            q5_lds0_attribution_erratum=immutable(
                erratum_path),
            carry_forward=SimpleNamespace(
                value=carry_forward, input=carry_forward_input,
                self_sha256=carry_forward["carry_forward_sha256"],
                semantic_sha256=carry_forward["carry_forward_sha256"]),
            hypothesis_evidence_manifest=SimpleNamespace(value=evidence_manifest),
            hypothesis_portfolio_contract=SimpleNamespace(
                sha256=F._PORTFOLIO_CONTRACT_SHA256),
            source_builder_id="gpu-source-v1",
            evidence_plan_id="reviewed-gpu-source-evidence-v1",
            runner_args_id="qwen05b-tg128",
            experiment_template_registry_id="gpu-source-templates-v3",
            experiment_template_registry_sha256=registry_sha,
            inference_window_lease_id="mi210-window-v1",
            production_snapshot_id="llama-v9-artifacts", revalidate=mock.Mock())
        site = SimpleNamespace(model_sha256=config.model.sha256,
                               model_path=str(config.model.path),
                               model_bytes=config.model.path.stat().st_size,
                               device_id="mi210_0")
        return config, site, docker, ca

    def test_public_launcher_signature_has_no_injection_authority(self):
        signature = inspect.signature(F.deployment_main)
        self.assertEqual(tuple(signature.parameters), ("argv",))
        self.assertIsNone(signature.parameters["argv"].default)
        self.assertNotIn("registry", str(signature))
        self.assertNotIn("executor", str(signature))
        self.assertNotIn("journal", str(signature))

    def test_public_initializer_vendors_and_revalidates_all_portfolio_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "portfolio-v26-bundle"
            path = F.initialize_static_deployment_bundle(
                root, frozen_production_comparator=
                frozen_production_comparator(Path(temporary) / "authority"))
            loaded = F.deployment.load_deployment_config(path)
            controller_config = F.controller_config(loaded, dry_run=True)
            json.dumps(controller_config.planner_context)
            portfolio = loaded.hypothesis_portfolio.value
            self.assertEqual(portfolio.sha256,
                             "7ba7dd1c3c246fb22a247d6e24facb5fbe0eaebec8b2eb21635fde20043e8303")
            context = loaded.planner_context.value
            self.assertEqual(
                {row["hypothesis_id"] for row in context["eligible_hypotheses"]},
                {"akh-v2-q5-onewave-preauthored",
                 "akh-v26-q4k-branchless-sixbit-scale",
                 "akh-v26-rms-scale-broadcast",
                 "akh-v26-rope-neox-index-strength-reduction",
                 "akh-v26-fa-combine-wave-normalization",
                 "akh-v26-q6k-packed-decode",
                 "akh-v26-fa-gqa7-common-map"},
            )
            self.assertEqual(len(portfolio.hypotheses), 27)
            self.assertEqual(len(context["eligible_hypotheses"]), 7)
            self.assertEqual(len(context["ineligible_hypotheses"]), 20)
            self.assertEqual(
                context["template_symbol_authority"],
                template_symbol_authority(F._template_registry()))
            self.assertEqual(
                controller_config.carry_forward["carry_forward_sha256"],
                controller_config.carry_forward_sha256)

            self.assertEqual(
                tuple(len(controller_config.carry_forward[key]) for key in (
                    "candidate_semantic_sha256", "candidate_patch_sha256",
                    "cross_campaign_candidate_sha256")),
                (13, 8, 8))
            self.assertEqual(
                controller_config.carry_forward[
                    "attribution_expectation_erratum"],
                C._q5_lds0_attribution_erratum())
            self.assertEqual(
                loaded.q5_lds0_attribution_erratum.sha256,
                C._Q5_LDS0_ERRATUM_FILE_SHA256)
            self.assertEqual(
                loaded.q5_lds0_attribution_erratum.path,
                (root / "config" /
                 "q5-lds0-attribution-erratum-v1.json").resolve())
            self.assertEqual(
                loaded.carry_forward.input.path,
                (root / "config" /
                 "discovery-carry-forward-v2.json").resolve())
            self.assertEqual(
                loaded.carry_forward.self_sha256,
                loaded.carry_forward.semantic_sha256)
            ineligible = {
                row["hypothesis_id"]: row for row in context["ineligible_hypotheses"]
            }
            for hypothesis_id in (
                    "akh-v2-lowbit-type-specialized-mmvq",
                    "akh-v2-quant-ladder-batched-wave-slot-residual",
                    "akh-v2-iq1s-occupancy-discriminator",
                    "akh-v2-batching-closes-all-lowbit-gaps"):
                self.assertIn(hypothesis_id, ineligible)
            self.assertEqual(
                ineligible["akh-v2-batching-closes-all-lowbit-gaps"]["status"],
                "retired")
            self.assertNotIn(
                "akh-v2-batching-closes-all-lowbit-gaps",
                {row["dnr_id"] for row in context["do_not_repeat"]})
            rows = loaded.hypothesis_evidence_manifest.value["evidence"]
            self.assertEqual(len(rows), 43)
            self.assertEqual(set(rows), {row["evidence_id"]
                                         for row in portfolio.body["evidence"]})
            self.assertTrue(all(str(root / "portfolio-evidence") in row["path"]
                                for row in rows.values()))
            self.assertFalse(any(
                original["path"] == rows[original["evidence_id"]]["path"]
                for original in portfolio.body["evidence"]))
            carrier = Path(next(iter(rows.values()))["path"])
            carrier.chmod(0o600)
            carrier.write_bytes(b"tampered")
            carrier.chmod(0o400)
            with self.assertRaises(F.deployment.DeploymentConfigError):
                F.deployment.load_deployment_config(path)

    def test_initializer_refuses_coherently_resealed_runtime_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparator = frozen_production_comparator(root / "authority")
            body = json.loads(comparator.read_text(encoding="utf-8"))
            body["runtime_snapshot_sha256"] = "0" * 64
            body["receipt_sha256"] = F.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "receipt_sha256"})
            comparator.write_text(
                json.dumps(body, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "runtime snapshot changed"):
                F.initialize_static_deployment_bundle(
                    root / "bundle",
                    frozen_production_comparator=comparator)

    def test_canonical_frozen_v9_comparator_sealer_and_validate_only(self):
        from . import seal_frozen_production_comparator as sealer

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "frozen-v9-comparator.json"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(sealer.main(["--output", str(output)]), 0)
            result = json.loads(stream.getvalue())
            self.assertEqual(result["status"], "sealed")
            self.assertFalse(result["inference_executed"])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o400)
            self.assertEqual(output.stat().st_nlink, 1)
            comparator = F._load_frozen_production_comparator(output)
            self.assertEqual(
                comparator.build_identity.source_sha256,
                F.cumulative_composition.FROZEN_PRODUCTION_SOURCE_SHA256)
            self.assertEqual(
                comparator.build_identity.config_sha256,
                comparator.build_receipt_sha256)
            self.assertEqual(
                (comparator.build_receipt_sha256,
                 comparator.linkage_receipt_sha256,
                 comparator.runtime_receipt_sha256,
                 comparator.measurement_receipt_sha256),
                (F.cumulative_composition.FROZEN_BUILD_RECEIPT_SHA256,
                 F.cumulative_composition.FROZEN_LINKAGE_RECEIPT_SHA256,
                 F.cumulative_composition.FROZEN_RUNTIME_RECEIPT_SHA256,
                 F.cumulative_composition.FROZEN_MEASUREMENT_RECEIPT_SHA256))
            self.assertNotEqual(comparator.workload_sha256,
                                comparator.observed_workload_sha256)
            self.assertNotEqual(comparator.runtime_config_sha256,
                                comparator.observed_runtime_config_sha256)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(sealer.main([
                    "--output", str(output), "--validate-only"]), 0)
            self.assertFalse(json.loads(stream.getvalue())["inference_executed"])

    def test_comparator_sealer_refuses_tamper_placeholder_and_aliases(self):
        from . import seal_frozen_production_comparator as sealer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "frozen-v9-comparator.json"
            with contextlib.redirect_stdout(io.StringIO()):
                sealer.main(["--output", str(output)])
            output.chmod(0o600)
            body = json.loads(output.read_text(encoding="utf-8"))
            body["measurement_protocol_sha256"] = "0" * 64
            body["receipt_sha256"] = F.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "receipt_sha256"})
            output.write_text(
                json.dumps(body, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
            output.chmod(0o400)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "differs from current"):
                sealer.main(["--output", str(output), "--validate-only"])

            canonical = root / "canonical.json"
            with contextlib.redirect_stdout(io.StringIO()):
                sealer.main(["--output", str(canonical)])
            alias = root / "alias.json"
            alias.symlink_to(canonical)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "alias|stable carrier"):
                sealer.main(["--output", str(alias)])
            hardlink = root / "hardlink.json"
            os.link(canonical, hardlink)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "stable carrier"):
                sealer.main(["--output", str(hardlink)])
            alias_parent = root / "alias-parent"
            actual_parent = root / "actual-parent"
            actual_parent.mkdir()
            alias_parent.symlink_to(actual_parent, target_is_directory=True)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "alias"):
                sealer.main(["--output", str(alias_parent / "new.json")])
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "git metadata"):
                F._validate_comparator_output_path(
                    root / ".git" / "comparator.json")
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "frozen production"):
                F._validate_comparator_output_path(
                    F.deployment.FROZEN_PRODUCTION_PATH /
                    "build-hip/comparator.json")

    def test_comparator_loader_refuses_noncanonical_carrier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            canonical = frozen_production_comparator(root / "authority")
            compact = root / "compact.json"
            compact.write_text(
                json.dumps(json.loads(canonical.read_text(encoding="utf-8")),
                           sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8")
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "canonical JSON"):
                F._load_frozen_production_comparator(compact)
            real_stat = os.stat

            def swapped_stat(path, *args, **kwargs):
                observed = real_stat(path, *args, **kwargs)
                if Path(path) != canonical:
                    return observed
                values = {
                    key: getattr(observed, key) for key in (
                        "st_dev", "st_ino", "st_uid", "st_nlink", "st_mode",
                        "st_size", "st_mtime_ns", "st_ctime_ns")}
                values["st_ino"] += 1
                return SimpleNamespace(**values)
            with mock.patch.object(F.os, "stat", side_effect=swapped_stat), \
                    self.assertRaisesRegex(
                        F.DeploymentFactoryError, "changed while read"):
                F._load_frozen_production_comparator(canonical)

    def test_historical_provenance_tamper_refuses_exact_digest(self):
        manifest = F._frozen_v9_closure_manifest()
        role = "measurement"
        authority = manifest["provenance"][role]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            carrier = root / authority["path"]
            carrier.parent.mkdir(parents=True)
            carrier.write_bytes(
                (F._SITE_GOVERNANCE_ROOT /
                 authority["path"]).read_bytes() + b"\n")
            carrier.chmod(0o400)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "changed while read"):
                F._stable_public_bytes(
                    carrier, authority["file_sha256"],
                    f"frozen production {role} provenance")

    def test_closure_manifest_refuses_alias_and_coherent_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            exact = root / "exact.json"
            shutil.copyfile(F._FROZEN_CLOSURE_MANIFEST, exact)
            exact.chmod(0o400)
            self.assertEqual(
                F._frozen_v9_closure_manifest(exact)["manifest_sha256"],
                F._FROZEN_CLOSURE_MANIFEST_SHA256)
            alias = root / "alias.json"
            alias.symlink_to(exact)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "stable carrier"):
                F._frozen_v9_closure_manifest(alias)
            exact.chmod(0o600)
            body = json.loads(exact.read_text(encoding="utf-8"))
            body["runtime"]["llama_bench_sha256"] = "0" * 64
            body["manifest_sha256"] = F.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "manifest_sha256"})
            exact.write_text(
                json.dumps(body, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
            exact.chmod(0o400)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "changed while read"):
                F._frozen_v9_closure_manifest(exact)

    def test_closure_manifest_requires_ratified_receipt_semantics(self):
        manifest = F._frozen_v9_closure_manifest()
        originals = {
            role: (F._SITE_GOVERNANCE_ROOT / row["path"]).read_bytes()
            for role, row in manifest["provenance"].items()}
        ratification = json.loads(originals["measurement"])
        ratification["production_binary_sha256"]["hip"] = "0" * 64
        changed = json.dumps(ratification, sort_keys=True).encode()

        def carrier(path, _expected, _label):
            for role, authority in manifest["provenance"].items():
                if Path(path) == F._SITE_GOVERNANCE_ROOT / authority["path"]:
                    return changed if role == "measurement" else originals[role]
            raise AssertionError(path)

        with mock.patch.object(
                F, "_stable_public_bytes", side_effect=carrier), \
                self.assertRaisesRegex(
                    F.DeploymentFactoryError, "semantics changed"):
            F._verify_frozen_v9_provenance(
                manifest, F._SITE_GOVERNANCE_ROOT)

    def test_runtime_inventory_fixture_matches_before_approved_readelf(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = Path("/usr/bin/true").resolve(strict=True)
            relative_paths = (
                "build/bin/llama-bench",
                "build/bin/llama-server",
                "build-hip/bin/llama-bench",
                "build-hip/bin/llama-server",
            )
            rows = []
            for relative in relative_paths:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                rows.append({
                    "relative_path": relative,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "symlink_target": None,
                })
            hip_real = root / "build-hip/bin/libggml-hip.so.0.16.0"
            shutil.copyfile(source, hip_real)
            hip_alias = root / "build-hip/bin/libggml-hip.so.0"
            hip_alias.symlink_to(hip_real.name)
            rows.append({
                "relative_path": "build-hip/bin/libggml-hip.so.0",
                "sha256": hashlib.sha256(hip_real.read_bytes()).hexdigest(),
                "symlink_target": hip_real.name,
            })
            rows.sort(key=lambda row: row["relative_path"])
            inventory = {
                "schema": "epyc.autokernel.runtime_inventory.v1",
                "readelf": {
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "version": "fixture readelf 1",
                },
                "objects": rows,
            }
            commands = []

            def inspect(argv, **_kwargs):
                commands.append(tuple(argv))
                return SimpleNamespace(
                    returncode=0,
                    stdout=("fixture readelf 1\n" if argv[1] == "--version"
                            else ""), stderr="")

            files, semantics = F._production_runtime_snapshot(
                root, closure_manifest={"runtime_inventory": inventory},
                runner=inspect)
            self.assertEqual(len(files), 5)
            self.assertEqual(commands[0], (str(source), "--version"))
            self.assertEqual(
                set(semantics["closures"]), {"build", "build-hip"})
            self.assertEqual(
                set(semantics["closures"]["build-hip"]["topology"]),
                {"libggml-hip.so.0", "llama-bench", "llama-server"})

    def test_runtime_inventory_fixture_refuses_before_readelf(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "build/bin/llama-bench"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture-runtime-object")
            inventory = {
                "schema": "epyc.autokernel.runtime_inventory.v1",
                "readelf": {
                    "path": str(Path("/usr/bin/true").resolve(strict=True)),
                    "sha256": hashlib.sha256(
                        Path("/usr/bin/true").resolve(strict=True).read_bytes()
                    ).hexdigest(),
                    "version": "fixture readelf 1",
                },
                "objects": [{
                    "relative_path": "build/bin/llama-bench",
                    "sha256": "0" * 64,
                    "symlink_target": None,
                }],
            }
            runner = mock.Mock()
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "runtime inventory differs from manifest"):
                F._production_runtime_snapshot(
                    root,
                    closure_manifest={"runtime_inventory": inventory},
                    runner=runner)
            runner.assert_not_called()

            inventory["objects"][0]["sha256"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
            inventory["objects"][0]["symlink_target"] = "unexpected-target"
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "runtime inventory differs from manifest"):
                F._production_runtime_snapshot(
                    root,
                    closure_manifest={"runtime_inventory": inventory},
                    runner=runner)
            runner.assert_not_called()

    def test_approved_readelf_requires_exact_bytes_and_version(self):
        inventory = json.loads(json.dumps(
            F._frozen_v9_closure_manifest()["runtime_inventory"]))
        inventory["readelf"]["sha256"] = "0" * 64
        runner = mock.Mock()
        with self.assertRaisesRegex(
                F.DeploymentFactoryError, "approved readelf is unavailable"):
            F._approved_readelf(inventory, runner=runner)
        runner.assert_not_called()

        inventory = F._frozen_v9_closure_manifest()["runtime_inventory"]
        runner.return_value = SimpleNamespace(
            returncode=0, stdout="foreign readelf\n", stderr="")
        with self.assertRaisesRegex(
                F.DeploymentFactoryError, "readelf version differs"):
            F._approved_readelf(inventory, runner=runner)

    def test_disposable_v9_runtime_mutations_refuse_frozen_closure(self):
        mutations = {
            "binary": lambda production: (
                production / "build-hip/bin/llama-bench"),
            "server": lambda production: (
                production / "build-hip/bin/llama-server"),
            "hip": lambda production: (
                production / "build-hip/bin/libggml-hip.so.0.16.0"),
            "runtime": lambda production: (
                production / "build/bin/libggml-base.so.0.16.0"),
        }
        with tempfile.TemporaryDirectory(
                dir="/mnt/raid0/llm/tmp") as temporary:
            production = disposable_v9_clone(
                Path(temporary).resolve())
            self.assertEqual(
                F.derive_frozen_production_comparator(
                    production_path=production).receipt_sha256,
                "79143fbfe62305dc1c9fce29248a8b3d4302eb27d3f55a60020eeb683c9de6bb")
        for name, target in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                    dir="/mnt/raid0/llm/tmp") as temporary:
                production = disposable_v9_clone(
                    Path(temporary).resolve())
                path = target(production)
                path.chmod(0o700)
                with path.open("ab") as stream:
                    stream.write(b"coherent-current-disk-substitution")
                with self.assertRaisesRegex(
                        F.DeploymentFactoryError,
                        "runtime (?:inventory|closure) differs from manifest"):
                    F.derive_frozen_production_comparator(
                        production_path=production)
        with tempfile.TemporaryDirectory(
                dir="/mnt/raid0/llm/tmp") as temporary:
            production = disposable_v9_clone(
                Path(temporary).resolve())
            alias = production / "build-hip/bin/libggml-hip.so"
            alias.unlink()
            alias.symlink_to("libggml-hip.so.0.16.0")
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "runtime (?:inventory|closure) differs from manifest"):
                F.derive_frozen_production_comparator(
                    production_path=production)

    def test_disposable_v9_clone_refuses_foreign_source_head(self):
        with tempfile.TemporaryDirectory(
                dir="/mnt/raid0/llm/tmp") as temporary:
            production = disposable_v9_clone(
                Path(temporary).resolve())
            subprocess.run([
                "git", "-C", str(production), "checkout", "--detach",
                f"{F.deployment.FROZEN_PRODUCTION_HEAD}^"],
                check=True, capture_output=True)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "source identity (observation is unavailable|differs from manifest)"):
                F.derive_frozen_production_comparator(
                    production_path=production)

    def test_substituted_server_is_not_executed_before_closure_authentication(
            self):
        with tempfile.TemporaryDirectory(
                dir="/mnt/raid0/llm/tmp") as temporary:
            root = Path(temporary).resolve()
            production = disposable_v9_clone(root)
            marker = root / "substituted-server-executed"
            server = production / "build-hip/bin/llama-server"
            source = root / "substituted-server.c"
            source.write_text(
                "#include <stdio.h>\n"
                "int main(void) {\n"
                f'  FILE *marker = fopen("{marker}", "w");\n'
                "  if (marker != NULL) { fputs(\"executed\", marker); "
                "fclose(marker); }\n"
                '  puts("version: 10125 (0db32c06e)");\n'
                "  return 0;\n"
                "}\n",
                encoding="utf-8")
            subprocess.run(
                ("/usr/bin/cc", str(source), "-o", str(server)),
                check=True, stdin=subprocess.DEVNULL, capture_output=True)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "runtime (?:inventory|closure) differs from manifest"):
                F.derive_frozen_production_comparator(
                    production_path=production)
            self.assertFalse(marker.exists())

    def test_post_hash_server_path_replacement_is_never_executed(self):
        with tempfile.TemporaryDirectory(
                dir="/mnt/raid0/llm/tmp") as temporary:
            root = Path(temporary).resolve()
            production = disposable_v9_clone(root)
            runtime_snapshot = F._production_runtime_snapshot(production)
            marker = root / "post-hash-replacement-executed"
            replacement = root / "replacement-llama-server"
            source = root / "replacement-llama-server.c"
            source.write_text(
                "#include <stdio.h>\n"
                "int main(void) {\n"
                f'  FILE *marker = fopen("{marker}", "w");\n'
                "  if (marker != NULL) { fputs(\"executed\", marker); "
                "fclose(marker); }\n"
                '  puts("version: 10125 (0db32c06e)");\n'
                "  return 0;\n"
                "}\n",
                encoding="utf-8")
            subprocess.run(
                ("/usr/bin/cc", str(source), "-o", str(replacement)),
                check=True, stdin=subprocess.DEVNULL, capture_output=True)
            server = production / "build-hip/bin/llama-server"
            original_verify = F._verify_frozen_v9_runtime_closure
            original_run = subprocess.run
            authenticated = []
            commands = []

            def replace_after_auth(*args, **kwargs):
                result = original_verify(*args, **kwargs)
                authenticated.append(True)
                if len(authenticated) == 1:
                    os.replace(replacement, server)
                return result

            def record_command(argv, *args, **kwargs):
                commands.append(tuple(argv))
                return original_run(argv, *args, **kwargs)

            with mock.patch.object(
                    F, "_production_runtime_snapshot",
                    return_value=runtime_snapshot), \
                    mock.patch.object(
                        F, "_verify_frozen_v9_runtime_closure",
                        side_effect=replace_after_auth), \
                    mock.patch.object(
                        F.subprocess, "run", side_effect=record_command), \
                    self.assertRaisesRegex(
                        F.DeploymentFactoryError,
                        "runtime (?:inventory|closure) differs from manifest"):
                F.derive_frozen_production_comparator(
                    production_path=production)
            self.assertEqual(authenticated, [True])
            self.assertFalse(marker.exists())
            self.assertTrue(commands)
            self.assertTrue(all(
                len(argv) >= 4 and argv[0] == "git" and argv[1] == "-C"
                and argv[3] in {"rev-parse", "symbolic-ref", "archive"}
                for argv in commands), commands)

    def test_initializer_refuses_coherently_resealed_build_config_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparator = frozen_production_comparator(root / "authority")
            body = json.loads(comparator.read_text(encoding="utf-8"))
            body["build_identity"]["config_sha256"] = "0" * 64
            body["receipt_sha256"] = F.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "receipt_sha256"})
            comparator.write_text(
                json.dumps(body, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "build/linkage identity changed"):
                F.initialize_static_deployment_bundle(
                    root / "bundle", frozen_production_comparator=comparator)

    def test_initializer_refuses_foreign_deployment_runtime_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparator = frozen_production_comparator(root / "authority")
            body = json.loads(comparator.read_text(encoding="utf-8"))
            self.assertNotEqual(
                body["runtime_config_sha256"],
                body["observed_runtime_config_sha256"])
            body["runtime_config_sha256"] = "0" * 64
            body["receipt_sha256"] = F.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "receipt_sha256"})
            comparator.write_text(
                json.dumps(body, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "another immutable frame"):
                F.initialize_static_deployment_bundle(
                    root / "bundle", frozen_production_comparator=comparator)

    def test_static_registry_refuses_coherently_resealed_linkage_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config, _site, _docker, _ca = self.static_config(root)
            comparator = config.frozen_production_comparator.path
            body = json.loads(comparator.read_text(encoding="utf-8"))
            body["build_identity"]["linkage_sha256"] = "0" * 64
            body["receipt_sha256"] = F.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "receipt_sha256"})
            comparator.write_text(
                json.dumps(body, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
            config.frozen_production_comparator = SimpleNamespace(
                path=comparator,
                sha256=hashlib.sha256(comparator.read_bytes()).hexdigest())
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "build/linkage identity changed"):
                F._static_registry(config, F._template_registry())

    def test_static_registry_refuses_coherently_resealed_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config, _site, _docker, _ca = self.static_config(root)
            comparator = config.frozen_production_comparator.path
            body = json.loads(comparator.read_text(encoding="utf-8"))
            body["build_identity"]["source_sha256"] = "0" * 64
            body["receipt_sha256"] = F.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "receipt_sha256"})
            comparator.write_text(
                json.dumps(body, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
            config.frozen_production_comparator = SimpleNamespace(
                path=comparator,
                sha256=hashlib.sha256(comparator.read_bytes()).hexdigest())
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "build/linkage identity changed"):
                F._static_registry(config, F._template_registry())

    def test_q5_erratum_vendored_carrier_refuses_coherent_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _site, _docker, _ca = self.static_config(Path(temporary))
            carrier = config.q5_lds0_attribution_erratum.path
            body = json.loads(carrier.read_text(encoding="utf-8"))
            body["operation_key"] = "0" * 64
            body["erratum_sha256"] = F.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "erratum_sha256"})
            carrier.write_text(
                json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8")
            config.q5_lds0_attribution_erratum = SimpleNamespace(
                path=carrier,
                sha256=hashlib.sha256(carrier.read_bytes()).hexdigest())
            with self.assertRaisesRegex(
                    C.DiscoveryControllerError,
                    "erratum file identity changed"):
                F._v25_carry_forward(config)

    def test_carry_forward_vendored_carrier_refuses_coherent_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _site, _docker, _ca = self.static_config(Path(temporary))
            changed = dict(config.carry_forward.value)
            changed["predecessor_state_semantic_sha256"] = "0" * 64
            changed["carry_forward_sha256"] = F.schemas.content_hash({
                key: value for key, value in changed.items()
                if key != "carry_forward_sha256"})
            carrier = config.carry_forward.input.path
            carrier.write_text(
                json.dumps(changed, sort_keys=True, separators=(",", ":"))
                + "\n", encoding="utf-8")
            changed_file = SimpleNamespace(
                path=carrier,
                sha256=hashlib.sha256(carrier.read_bytes()).hexdigest())
            config.carry_forward = SimpleNamespace(
                value=changed, input=changed_file,
                self_sha256=changed["carry_forward_sha256"],
                semantic_sha256=changed["carry_forward_sha256"])
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "carrier differs from derived authority"):
                F._v25_carry_forward(config)

    def test_v25_manifest_evidence_must_join_its_exact_controller_turn(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _site, _docker, _ca = self.static_config(Path(temporary))
            rows = config.hypothesis_evidence_manifest.value["evidence"]
            rows["ev-v25-source-manifest-turn04"] = dict(
                rows["ev-v25-source-manifest-turn06"])
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError,
                    "source manifest does not join its controller row"):
                F._v25_carry_forward(config)

    def test_execution_module_attestor_refuses_any_live_byte_drift(self):
        sealed = {"runner": {"logical_path": "scripts/runner.py", "sha256": "a" * 64}}
        attest = F._module_attestor(sealed)
        with mock.patch.object(F, "_execution_module_identity", return_value=sealed):
            attest()
        changed = {"runner": {"logical_path": "scripts/runner.py", "sha256": "b" * 64}}
        with mock.patch.object(F, "_execution_module_identity", return_value=changed), \
             self.assertRaisesRegex(F.DeploymentFactoryError, "module bytes changed"):
            attest()

    def test_t0_capability_contract_is_in_exact_graph_and_tamper_attested(self):
        sealed = F._execution_module_identity()
        self.assertEqual(
            sealed["t0_provider"], {
                "logical_path": "scripts/kernel_rnd/autokernel/execution/t0_provider.py",
                "sha256": F._digest_regular(
                    Path(F.t0_provider.__file__).resolve(strict=True),
                    "t0_provider"),
            })
        changed = json.loads(json.dumps(sealed))
        changed["t0_provider"]["sha256"] = "0" * 64
        attest = F._module_attestor(sealed)
        with mock.patch.object(F, "_execution_module_identity", return_value=changed), \
                self.assertRaisesRegex(
                    F.DeploymentFactoryError, "module bytes changed"):
            attest()

    def test_module_runtime_provenance_refuses_alias_symlink_and_hardlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runner.py"
            source.write_bytes(b"trusted module\n")
            runtime = F._runtime_module_file(
                "scripts/runner.py", source, "runner")
            semantic = {"runner": {
                "logical_path": runtime["logical_path"],
                "sha256": runtime["sha256"]}}
            attest = F._module_attestor(semantic, {"runner": runtime})
            alias = root / "same-bytes.py"
            alias.write_bytes(source.read_bytes())
            alias_runtime = F._runtime_module_file(
                "scripts/runner.py", alias, "runner")
            with mock.patch.object(F, "_execution_module_identity",
                                   return_value=semantic), \
                    mock.patch.object(F, "_execution_module_runtime_provenance",
                                      return_value={"runner": alias_runtime}), \
                    self.assertRaisesRegex(
                        F.DeploymentFactoryError, "runtime provenance changed"):
                attest()
            symlink = root / "symlink.py"
            symlink.symlink_to(source)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "single-link regular non-symlink"):
                F._runtime_module_file("scripts/runner.py", symlink, "runner")
            hardlink = root / "hardlink.py"
            os.link(source, hardlink)
            with self.assertRaisesRegex(
                    F.DeploymentFactoryError, "single-link regular non-symlink"):
                F._runtime_module_file("scripts/runner.py", hardlink, "runner")

    def test_validate_only_materializes_static_graph_without_actor_or_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, site, docker, ca = self.static_config(root)
            deployment_path = root / "deployment.json"
            deployment_path.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            forbidden = AssertionError("validate-only crossed the execution boundary")
            with mock.patch.object(F.deployment, "load_deployment_config", return_value=config), \
                    mock.patch.dict(F.controller.gpu_discovery.SITE_LOAD_PROFILES,
                                    {F._LOAD_PROFILE_ID: site}, clear=True), \
                    mock.patch.object(F.codex_container_actor, "DOCKER_EXECUTABLE", str(docker)), \
                    mock.patch.object(F.codex_container_actor, "CA_CERTIFICATE_PATH", ca), \
                    mock.patch.object(F, "_target_source_equality_receipt",
                                      return_value=(root / "equality.json", "e" * 64)), \
                    mock.patch.object(F, "_instrument_review_receipt",
                                      return_value=(root / "instrument.json", "i" * 64)), \
                    mock.patch.object(F.codex_container_actor, "run_actor", side_effect=forbidden), \
                    mock.patch.object(F.claude_fable5_critic_actor, "run_critic", side_effect=forbidden), \
                    mock.patch.object(F.controller.gpu_discovery, "run", side_effect=forbidden), \
                    mock.patch.object(F.controller, "run_controller", side_effect=forbidden), \
                    mock.patch.object(F.evidence.SubprocessCommandExecutor, "__call__",
                                      side_effect=forbidden), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(F.deployment_main(
                    ["--deployment", str(deployment_path), "--validate-only"]), 0)
                graph_path = config.state_root / "deployment-graph.json"
                sealed_graph = graph_path.read_bytes()
                for legacy_schema in (
                        "epyc.autokernel.static_discovery_graph.v5",
                        "epyc.autokernel.static_discovery_graph.v6",
                        "epyc.autokernel.static_discovery_graph.v7",
                        "epyc.autokernel.static_discovery_graph.v8"):
                    with self.subTest(legacy_schema=legacy_schema):
                        graph_path.write_text(json.dumps({
                            "schema": legacy_schema,
                            "graph_sha256": "0" * 64,
                        }), encoding="utf-8")
                        with self.assertRaisesRegex(
                                F.DeploymentFactoryError,
                                "legacy deployment graph cannot authorize successor"):
                            F.build_static_deployment_graph(config)
                graph_path.write_bytes(sealed_graph)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "validated")
            self.assertFalse(payload["inference_executed"])
            receipt = json.loads(Path(payload["graph_receipt"]).read_text(encoding="utf-8"))
            self.assertFalse(receipt["inference_executed"])
            self.assertEqual(receipt["registry_ids"], dict(F._STATIC_IDS))
            profiler = receipt["profiler_runtime_authority"]
            self.assertEqual(profiler["trace_schema_id"],
                             F.evidence.ROCPROF_V3_TRACE_ID)
            package = next(row for row in profiler["inputs"]
                           if row["role"] == "profiler_package")
            self.assertEqual(package, {
                "role": "profiler_package",
                "path": str(F._ROCPROF_V3_PACKAGE.resolve()),
                "sha256": F._ROCPROF_V3_PACKAGE_SHA256,
            })
            self.assertEqual(receipt["actor_wrappers"]["planner"]["sha256"],
                             config.actor_wrapper.sha256)
            self.assertEqual(receipt["actor_wrappers"]["critic"]["sha256"],
                             config.critic_wrapper.sha256)
            self.assertEqual(receipt["actor_cells"],
                             [dict(C.SOL), dict(C.FABLE5_CRITIC)])
            timed_oracle = receipt["batched_runner"]["timed_output_oracle"]
            self.assertTrue(timed_oracle["enabled_for_source_patch"])
            self.assertTrue(timed_oracle["independent_of_early_unlock"])
            self.assertEqual(timed_oracle["instrument_commit"], F._INSTRUMENT_COMMIT)
            self.assertEqual(timed_oracle["environment"], {
                "AMD_SERIALIZE_KERNEL": "3",
                "AMD_SERIALIZE_COPY": "3",
                "GGML_CUDA_DISABLE_GRAPHS": "1",
            })
            self.assertEqual(timed_oracle["scope"], "integrity_discovery_only")
            self.assertFalse(timed_oracle["production_throughput_authority"])
            self.assertIn("discovery_telemetry", receipt["execution_modules"])
            self.assertIn("discovery_supervisor", receipt["execution_modules"])
            self.assertIn("hypotheses", receipt["execution_modules"])
            self.assertIn("do_not_repeat", receipt["execution_modules"])
            self.assertIn("t0_provider", receipt["execution_modules"])
            self.assertIn("codex_container_actor", receipt["execution_modules"])
            self.assertEqual(receipt["schema"],
                             "epyc.autokernel.static_discovery_graph.v9")
            self.assertEqual(receipt["carry_forward"], {
                "schema": F.deployment.CARRY_FORWARD_SCHEMA,
                "file_sha256": config.carry_forward.input.sha256,
                "self_sha256": config.carry_forward.self_sha256,
                "semantic_sha256": config.carry_forward.semantic_sha256,
            })
            self.assertEqual(receipt["attribution_expectation_erratum"], {
                "schema":
                    "epyc.autokernel.attribution_expectation_erratum_source.v1",
                "erratum_schema":
                    "epyc.autokernel.attribution_expectation_erratum.v1",
                "erratum_sha256":
                    C._q5_lds0_attribution_erratum()["erratum_sha256"],
                "file_sha256": C._Q5_LDS0_ERRATUM_FILE_SHA256,
                "operation_key":
                    C._q5_lds0_attribution_erratum()["operation_key"],
                "attribution_refusal_file_sha256":
                    C._q5_lds0_attribution_erratum()[
                        "attribution_refusal_file_sha256"],
                "candidate_semantic_sha256":
                    C._q5_lds0_attribution_erratum()[
                        "candidate_semantic_sha256"],
            })
            self.assertTrue(all(
                set(row) == {"logical_path", "sha256"}
                and row["logical_path"].startswith("scripts/")
                and not Path(row["logical_path"]).is_absolute()
                for row in receipt["execution_modules"].values()))
            self.assertEqual(
                receipt["actor_argv_authority"]["planner"]["module_id"],
                "codex_container_actor")
            self.assertEqual(
                receipt["actor_argv_authority"]["critic"]["module_id"],
                "claude_fable5_critic_actor")
            self.assertEqual(
                receipt["execution_modules"]["discovery_telemetry"]["sha256"],
                F._digest_regular(Path(F.discovery_telemetry.__file__).resolve(),
                                  "discovery_telemetry"))
            self.assertEqual(
                receipt["execution_modules"]["hypotheses"]["sha256"],
                F._digest_regular(Path(C.hypotheses.__file__).resolve(),
                                  "hypotheses"))
            self.assertEqual(
                receipt["execution_modules"]["do_not_repeat"]["sha256"],
                F._digest_regular(Path(C.do_not_repeat.__file__).resolve(),
                                  "do_not_repeat"))
            self.assertEqual(
                receipt["execution_modules"]["t0_provider"]["sha256"],
                F._digest_regular(Path(F.t0_provider.__file__).resolve(),
                                  "t0_provider"))
            self.assertTrue(receipt["critic_auth_source"]["validated"])
            self.assertFalse(receipt["critic_auth_source"]["secret_digest_persisted"])
            self.assertNotIn("sha256", receipt["critic_auth_source"])
            for profile in receipt["environment_profiles"].values():
                self.assertNotIn("LD_LIBRARY_PATH", profile)
                self.assertNotIn("PYTHONPATH", profile)
            self.assertNotIn("HOME", receipt["environment_profiles"]["critic"])
            self.assertNotIn("CODEX_HOME", receipt["environment_profiles"]["critic"])

    def test_static_graph_refuses_unknown_constructor_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, site, docker, ca = self.static_config(Path(temporary))
            config.runner_args_id = "caller-injected"
            with mock.patch.dict(F.controller.gpu_discovery.SITE_LOAD_PROFILES,
                                 {F._LOAD_PROFILE_ID: site}, clear=True), \
                    mock.patch.object(F.codex_container_actor, "DOCKER_EXECUTABLE", str(docker)), \
                    mock.patch.object(F.codex_container_actor, "CA_CERTIFICATE_PATH", ca), \
                    self.assertRaises(F.DeploymentFactoryError):
                F.build_static_deployment_graph(config)

    def test_materialized_actor_refuses_runtime_identity_drift_before_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, site, docker, ca = self.static_config(Path(temporary))
            with mock.patch.dict(F.controller.gpu_discovery.SITE_LOAD_PROFILES,
                                 {F._LOAD_PROFILE_ID: site}, clear=True), \
                    mock.patch.object(F.codex_container_actor, "DOCKER_EXECUTABLE", str(docker)), \
                    mock.patch.object(F.codex_container_actor, "CA_CERTIFICATE_PATH", ca), \
                    mock.patch.object(F, "_target_source_equality_receipt",
                                      return_value=(Path(temporary) / "equality.json", "e" * 64)), \
                    mock.patch.object(F, "_instrument_review_receipt",
                                      return_value=(Path(temporary) / "instrument.json", "i" * 64)):
                graph = F.build_static_deployment_graph(config)
                config.critic_wrapper.path.write_bytes(b"mutated Claude CLI")
                with self.assertRaises(C.DiscoveryControllerError):
                    graph.adapters["critic"].attest()
                native, _host = F.codex_container_actor._codex_native_assets(
                    config.actor_wrapper.path)
                native.write_bytes(b"mutated native")
                with self.assertRaises(C.DiscoveryControllerError):
                    graph.adapters["planner"].attest()

    def test_environment_rejects_loader_injection(self):
        for key in ("LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "DYLD_INSERT_LIBRARIES"):
            with self.subTest(key=key):
                with self.assertRaises(F.DeploymentFactoryError):
                    F.EnvironmentProfile({"PATH": "/usr/bin", key: "bad"})

    def test_validate_only_auth_check_refuses_without_persisting_secret_identity(self):
        with mock.patch.object(
                F.claude_fable5_critic_actor, "_credentials",
                side_effect=F.claude_fable5_critic_actor.ClaudeFable5CriticError(
                    "unsafe credential")), self.assertRaisesRegex(
                        F.claude_fable5_critic_actor.ClaudeFable5CriticError,
                        "unsafe credential"):
            F._validate_critic_auth_source()
        receipt = F._validate_critic_auth_source()
        self.assertFalse(receipt["secret_digest_persisted"])
        self.assertFalse(any("sha" in key for key in receipt))

    def test_source_scope_refuses_reward_and_toolchain_mutations(self):
        class Manifest:
            source_tree = "llama.cpp"
            def __init__(self, paths):
                self.declared_files = paths
                self.declared_symbols = {path: ("fattn_kernel",) for path in paths}
        for path in ("tools/llama-bench/llama-bench.cpp", "CMakeLists.txt",
                     "cmake/toolchain.cmake", "scripts/parse.py", "tests/test.cpp",
                     "ggml/src/ggml.c"):
            candidate = mock.Mock(source_manifest=Manifest((path,)))
            with self.subTest(path=path), self.assertRaises(F.DeploymentFactoryError):
                F._validate_source_scope(candidate, template())
        F._validate_source_scope(mock.Mock(
            source_manifest=Manifest(("ggml/src/ggml-cuda/fattn.cu",))), template())

    def test_controller_config_has_no_cli_override_authority(self):
        continuation = preauthored_continuation.load(
            preauthored_continuation.DEFAULT_CARRIER)
        context = {
            "context_sha256": "a" * 64,
            "preauthored_continuation_sha256": continuation.sha256,
            "preauthored_source_backed_diff_sha256":
                continuation.source_backed_diff_sha256,
        }
        config = mock.Mock(state_root=Path("/state"), evidence_root=Path("/evidence"),
                           max_iterations=2, nomination_threshold=.03,
                           planner_context=mock.Mock(value=context), production_branch="production-consolidated-v9",
                           production_head="b" * 40,
                           instrument_branch="measurement-instrument",
                           instrument_path=Path("/instrument"),
                           instrument_commit=continuation.compatibility_bridge[
                               "current_instrument_commit"],
                           config_sha256="c" * 64,
                           experiment_template_registry_sha256="d" * 64)
        portfolio = hypothesis_portfolio.load(hypothesis_portfolio.DEFAULT_PORTFOLIO)
        config.hypothesis_portfolio = SimpleNamespace(value=portfolio)
        config.preauthored_continuation = SimpleNamespace(value=continuation)
        config.admission_policy = SimpleNamespace(
            value={"policy_sha256": "e" * 64, "examples": [], "profiles": []},
            corpus=SimpleNamespace(policy_sha256="e" * 64, version="test-v2"))
        config.revalidate = mock.Mock()
        carry_forward = mocked_v25_carry_forward()
        with mock.patch.object(
                F, "_v25_carry_forward", return_value=carry_forward) as derive, \
                mock.patch.object(
                    F.preauthored_continuation, "verify_git_authority",
                    return_value=continuation.source_backed_diff) as verify:
            result = F.controller_config(config, dry_run=True)
        self.assertEqual((result.output_root, result.evidence_root,
                          result.max_iterations, result.nomination_threshold,
                          result.dry_run, result.planner_context_sha256, result.production_base_commit),
                         (Path("/state"), Path("/evidence"), 2, .03, True,
                          F.schemas.content_hash({"planner_context_sha256": "a" * 64,
                                                  "admission_policy_sha256": "e" * 64,
                                                  "admission_policy_version": "test-v2",
                                                  "deployment_identity_sha256": "c" * 64}), "b" * 40))
        self.assertEqual(result.deployment_identity_sha256, "c" * 64)
        self.assertEqual(result.carry_forward, carry_forward)
        self.assertEqual(result.carry_forward_sha256,
                         carry_forward["carry_forward_sha256"])
        derive.assert_called_once_with(config)
        verify.assert_called_once_with(
            continuation, config.instrument_path, config.instrument_commit)
        config.revalidate.assert_called_once()

    def test_window_lease_uses_sealed_arbiter_and_never_probes_cpu_lock(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"; model.write_bytes(b"model")
            kfd_root = Path(directory) / "kfd"; kfd_root.mkdir()
            profile = SimpleNamespace(model_path=str(model), model_sha256="a" * 64,
                                      device_id="mi210_0", workload="tg128", calls_per_arm=9,
                                      cold_load_host_bytes=4, worst_case_loads_per_interval=18)
            config = mock.Mock(inference_window_lock="/lock", device_id="mi210_0",
                               model=SimpleNamespace(path=model, sha256="a" * 64),
                               admission_policy=SimpleNamespace(corpus=SimpleNamespace(
                                   profiles=(profile,), policy_sha256="b" * 64, version="test-v2")),
                               planner_context=SimpleNamespace(value={"context_sha256": "c" * 64}))
            config.revalidate = mock.Mock()
            decision = SimpleNamespace(mode="cold_serialized", to_dict=lambda: {"decision_sha256": "d" * 64})
            opened = F.device_claim.ClaimReceipt(
                claim_id="akd-test", device_id="mi210_0", lock_path="/claim",
                state="held", holder_pid=1, holder_start_ticks=1,
                holder_boot_id="boot", host="host", purpose="probe",
                campaign_id="ak-test", acquired_at="now")
            class Claim:
                held = True
                def receipt(self): return opened
                def release(self):
                    self.held = False
                    return F.replace(opened, released_at="done")
            with mock.patch.object(F.gpu_load_admission, "arbitrate", return_value=decision), \
                 mock.patch.object(F.inference_window.InferenceCallWindow, "acquire",
                                   side_effect=AssertionError("lease must not invent a CPU lock probe")):
                admitted = F.GpuDiscoveryLease(
                    config=config, mode="allowed_discovery_noise", claim_journal=mock.Mock(),
                    claim_acquirer=lambda *_args, **_kwargs: Claim(),
                    claim_verifier=lambda _receipt: F.schemas.Check(F.schemas.PASS),
                    kfd_root=kfd_root).admit(
                        mock.Mock(source_manifest=mock.Mock(campaign_id="ak-test")),
                        operation_key="e" * 64)
        self.assertTrue(admitted["admitted"])
        self.assertEqual(admitted["mode"], "cold_serialized")
        self.assertEqual(admitted["load_admission"], {"decision_sha256": "d" * 64})

    def test_prebuild_kfd_inventory_refuses_foreign_unreadable_and_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root / "model"; model.write_bytes(b"model")
            kfd_root = root / "kfd"; kfd_root.mkdir()
            profile = SimpleNamespace(
                model_path=str(model), model_sha256="a" * 64, device_id="mi210_0",
                workload="tg128", calls_per_arm=9, cold_load_host_bytes=4,
                worst_case_loads_per_interval=18)
            config = mock.Mock(
                inference_window_lock="/lock", device_id="mi210_0",
                model=SimpleNamespace(path=model, sha256="a" * 64),
                admission_policy=SimpleNamespace(corpus=SimpleNamespace(
                    profiles=(profile,), policy_sha256="b" * 64, version="test-v2")),
                planner_context=SimpleNamespace(value={"context_sha256": "c" * 64}))
            config.revalidate = mock.Mock()
            opened = F.device_claim.ClaimReceipt(
                claim_id="akd-kfd", device_id="mi210_0", lock_path="/claim",
                state="held", holder_pid=1, holder_start_ticks=1,
                holder_boot_id="boot", host="host", purpose="probe",
                campaign_id="ak-test", acquired_at="now")
            class Claim:
                held = True
                def receipt(self): return opened
                def release(self): self.held=False; return F.replace(opened,released_at="done")
            acquire = mock.Mock(side_effect=lambda *_args, **_kwargs: Claim())
            decision = SimpleNamespace(
                mode="cold_serialized", to_dict=lambda:{"decision_sha256":"d"*64})
            candidate = mock.Mock(source_manifest=mock.Mock(campaign_id="ak-test"))
            lease = F.GpuDiscoveryLease(
                config=config, mode="allowed_discovery_noise", claim_journal=mock.Mock(),
                claim_acquirer=acquire, claim_verifier=lambda _receipt: True,
                kfd_root=kfd_root)
            for pid in (42, 7): (kfd_root / str(pid)).mkdir()
            with mock.patch.object(F.gpu_load_admission,"arbitrate",return_value=decision):
                busy=lease.admit(candidate,operation_key="e"*64)
            self.assertEqual((busy["admitted"],busy["reason"],busy["foreign_kfd_pids"]),
                             (False,"foreign_kfd_busy",[7,42]))
            acquire.assert_not_called()
            for pid in (42, 7): (kfd_root / str(pid)).rmdir()
            with mock.patch.object(F.gpu_load_admission,"arbitrate",return_value=decision):
                resumed=lease.resume(candidate,busy)
            self.assertTrue(resumed["admitted"]); acquire.assert_called_once()
            (kfd_root / "99").mkdir()
            with self.assertRaises(C.ResourceWait) as blocked:
                lease.reserve("e"*64)
            self.assertEqual((blocked.exception.receipt["phase"],
                              blocked.exception.receipt["reason"],
                              blocked.exception.receipt["foreign_kfd_pids"]),
                             ("pre_executor_reservation","foreign_kfd_busy",[99]))
            # Only the released prebuild probe was acquired; no outer claim ran.
            acquire.assert_called_once(); (kfd_root / "99").rmdir()

            malformed=kfd_root / "not-a-pid"; malformed.mkdir(); acquire.reset_mock()
            with mock.patch.object(F.gpu_load_admission,"arbitrate",return_value=decision):
                invalid=lease.admit(candidate,operation_key="f"*64)
            self.assertEqual((invalid["admitted"],invalid["reason"]),
                             (False,"foreign_kfd_inventory_invalid"))
            acquire.assert_not_called(); malformed.rmdir()
            with mock.patch.object(Path,"iterdir",side_effect=PermissionError("denied")), \
                 mock.patch.object(F.gpu_load_admission,"arbitrate",return_value=decision):
                unreadable=lease.admit(candidate,operation_key="1"*64)
            self.assertEqual((unreadable["admitted"],unreadable["reason"]),
                             (False,"foreign_kfd_inventory_unreadable"))
            acquire.assert_not_called()

    def test_reservation_cleanup_owns_malformed_verifier_and_retry_failures(self):
        kfd_directory = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree,kfd_directory)
        kfd_root = Path(kfd_directory)
        opened = F.device_claim.ClaimReceipt(
            claim_id="akd-test", device_id="mi210_0", lock_path="/claim",
            state="held", holder_pid=1, holder_start_ticks=1,
            holder_boot_id="boot", host="host", purpose="outer",
            campaign_id="ak-test", acquired_at="now")
        class Claim:
            def __init__(self, *, malformed=False, fail_release_once=False):
                self.malformed = malformed; self.fail_release_once = fail_release_once
                self.release_calls = 0; self.held = True
            def receipt(self): return {"bad": True} if self.malformed else opened
            def release(self):
                self.release_calls += 1
                self.held = False
                if self.fail_release_once and self.release_calls == 1:
                    raise RuntimeError("journal unavailable once")
                return F.replace(opened, released_at="done")
        config = mock.Mock(device_id="mi210_0")
        operation_key = "e" * 64
        for label, claim, verifier in (
                ("malformed", Claim(malformed=True), lambda _receipt: True),
                ("verifier", Claim(), lambda _receipt: (_ for _ in ()).throw(
                    RuntimeError("verifier failed")))):
            with self.subTest(label=label):
                lease = F.GpuDiscoveryLease(
                    config=config, mode="allowed_discovery_noise",
                    claim_journal=mock.Mock(),
                    claim_acquirer=lambda *_args, claim=claim, **_kwargs: claim,
                    claim_verifier=verifier, kfd_root=kfd_root)
                lease._campaigns[operation_key] = "ak-test"
                with self.assertRaises(Exception):
                    lease.reserve(operation_key)
                self.assertFalse(claim.held)
                self.assertEqual(claim.release_calls, 1)
                self.assertNotIn(operation_key, lease._active)
        claim = Claim(fail_release_once=True)
        lease = F.GpuDiscoveryLease(
            config=config, mode="allowed_discovery_noise", claim_journal=mock.Mock(),
            claim_acquirer=lambda *_args, **_kwargs: claim,
            claim_verifier=lambda _receipt: True, kfd_root=kfd_root)
        lease._campaigns[operation_key] = "ak-test"
        lease.reserve(operation_key)
        released = lease.release(operation_key)
        self.assertEqual(released["claim_id"], opened.claim_id)
        self.assertEqual(claim.release_calls, 2)
        self.assertNotIn(operation_key, lease._active)

    def test_probe_validation_errors_always_release_the_acquired_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"; model.write_bytes(b"model")
            kfd_root = Path(directory) / "kfd"; kfd_root.mkdir()
            profile = SimpleNamespace(
                model_path=str(model), model_sha256="a" * 64, device_id="mi210_0",
                workload="tg128", calls_per_arm=9, cold_load_host_bytes=4,
                worst_case_loads_per_interval=18)
            config = mock.Mock(
                inference_window_lock="/lock", device_id="mi210_0",
                model=SimpleNamespace(path=model, sha256="a" * 64),
                admission_policy=SimpleNamespace(corpus=SimpleNamespace(
                    profiles=(profile,), policy_sha256="b" * 64, version="test-v2")),
                planner_context=SimpleNamespace(value={"context_sha256": "c" * 64}))
            config.revalidate = mock.Mock()
            base = F.device_claim.ClaimReceipt(
                claim_id="akd-probe", device_id="mi210_0", lock_path="/claim",
                state="held", holder_pid=1, holder_start_ticks=1,
                holder_boot_id="boot", host="host", purpose="probe",
                campaign_id="ak-test", acquired_at="now")
            class Probe:
                def __init__(self, malformed=False): self.malformed=malformed; self.held=True; self.release_calls=0
                def receipt(self): return {"bad":True} if self.malformed else base
                def release(self): self.release_calls+=1; self.held=False; return F.replace(base,released_at="done")
            decision = SimpleNamespace(mode="cold_serialized",to_dict=lambda:{"decision_sha256":"d"*64})
            for label, probe, verifier in (
                    ("malformed",Probe(True),lambda _receipt:True),
                    ("verifier",Probe(),lambda _receipt:(_ for _ in ()).throw(RuntimeError("verify")))):
                lease=F.GpuDiscoveryLease(config=config,mode="allowed_discovery_noise",claim_journal=mock.Mock(),claim_acquirer=lambda *_args,probe=probe,**_kwargs:probe,claim_verifier=verifier,kfd_root=kfd_root)
                with self.subTest(label=label), mock.patch.object(F.gpu_load_admission,"arbitrate",return_value=decision), self.assertRaises(Exception):
                    lease.admit(mock.Mock(source_manifest=mock.Mock(campaign_id="ak-test")),operation_key="f"*64)
                self.assertFalse(probe.held); self.assertEqual(probe.release_calls,1)

    def test_materialized_builder_preserves_each_operation_and_binds_deployment_authority(self):
        """The build cache key must never replace the controller operation key."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            protected = root / "protected"; protected.mkdir()
            artifact = protected / "artifact"; artifact.write_bytes(b"frozen")
            bound = F.evidence.BoundInputFile(
                "production_artifact", artifact,
                hashlib.sha256(artifact.read_bytes()).hexdigest())
            operations = root / "operations"
            operations.mkdir()
            calls = []
            def source_build(_candidate, _authorization, permit):
                carrier = operations / permit["operation_key"] / "source-manifest.json"
                self.assertTrue(carrier.is_file())
                self.assertEqual(
                    carrier.read_bytes(),
                    source_candidate.source_patch_manifest_bytes(
                        _candidate.source_manifest))
                calls.append(dict(permit))
                return dict(permit)
            source = F.SourceBuilderBinding(source_build)
            templates = mock.Mock(spec=F.ExperimentTemplateRegistry)
            templates.registry_sha256 = "e" * 64
            templates.templates = {}
            templates.resolve.return_value = mock.sentinel.template
            resolved = SimpleNamespace(
                environment_profile=F.EnvironmentProfile({"PATH": "/usr/bin"}),
                source_builder=source,
                evidence_plan=F.EvidencePlanBinding(mock.Mock()),
                runner_args=F.RunnerArgsBinding(mock.Mock()),
                experiment_template_registry=templates,
                inference_window_lease=F.InferenceWindowLeaseBinding(),
                production_snapshot=F.ProductionSnapshotBinding(
                    protected, (bound,), {}, F.schemas.content_hash({})))
            config = mock.Mock(
                config_sha256="d" * 64, experiment_template_registry_sha256="e" * 64,
                actor_wrapper=SimpleNamespace(path=Path("/sealed/codex-wrapper"),
                                              sha256="a" * 64),
                critic_wrapper=SimpleNamespace(path=Path("/sealed/claude-fable5"),
                                               sha256="b" * 64),
                production_path=protected, instrument_path=protected,
                operations_root=operations, claim_timeout_s=0.0,
                production_head="0" * 40, instrument_commit="1" * 40,
                instrument_branch="measurement-instrument")
            config.revalidate = mock.Mock()
            candidate = planned_source_candidate()
            restored = C._restore_pending({"candidate": C._pending_item(candidate)})
            adapters = {}
            def adapter_factory(**kwargs):
                adapters.update(kwargs)
                return mock.sentinel.screener
            supervised_authority = {
                "schema": "epyc.autokernel.supervised_build_authority.v2",
                "deployment_config_canonical_sha256": "c" * 64,
                "deployment_config_semantic_sha256": config.config_sha256}
            with mock.patch.object(F.deployment, "resolve_registry", return_value=resolved), \
                 mock.patch.object(F, "_SUPERVISED_BUILD_AUTHORITY",
                                   supervised_authority), \
                 mock.patch.object(F, "_validate_source_scope"), \
                 mock.patch.object(F.gpu_source_adapter, "build_governed_gpu_source_adapter",
                                   side_effect=adapter_factory), \
                 mock.patch.object(F, "_production_runtime_snapshot",
                                   return_value=((bound,), {})), \
                 mock.patch.object(F, "_reviewed_source_package", return_value=None), \
                 mock.patch.object(F.controller, "build_controller_adapters",
                                   side_effect=lambda **kwargs: kwargs), \
                 mock.patch.object(F.codex_container_actor, "runtime_identity",
                                   return_value={}), \
                 mock.patch.object(F.claude_fable5_critic_actor, "runtime_identity",
                                   return_value={}), \
                 mock.patch.object(F, "_instrument_review_receipt",
                                   return_value=(root / "instrument-review.json", "f" * 64)):
                F.materialize(config, {}, correctness_executor=mock.Mock(),
                              rocprof_executor=mock.Mock(), claim_journal=mock.Mock())
                build = adapters["build_source"]
                (operations / ("1" * 64)).mkdir(mode=0o700)
                first = build(candidate, object(), {"operation_key": "1" * 64})
                (operations / ("2" * 64)).mkdir(mode=0o700)
                second = build(restored, object(), {"operation_key": "2" * 64})
                (operations / ("3" * 64)).mkdir(mode=0o700)
                with mock.patch.object(
                        source_candidate, "SCHEMA_SOURCE_PATCH",
                        "epyc.autokernel.source_patch.v1"), self.assertRaisesRegex(
                            F.DeploymentFactoryError,
                            "canonical carrier hash mismatch"):
                    build(restored, object(), {"operation_key": "3" * 64})
                (operations / ("4" * 64)).mkdir(mode=0o700)
                with mock.patch.object(
                        F, "_instrument_review_receipt",
                        side_effect=F.DeploymentFactoryError(
                            "instrument capability changed")), self.assertRaisesRegex(
                                F.DeploymentFactoryError,
                                "instrument capability changed"):
                    build(restored, object(), {"operation_key": "4" * 64})
            self.assertEqual(first["operation_key"], "1" * 64)
            self.assertEqual(second["operation_key"], "2" * 64)
            self.assertFalse((operations / ("3" * 64) / "source-manifest.json").exists())
            self.assertEqual(len(calls), 2)
            self.assertEqual([row["deployment_config_semantic_sha256"] for row in calls],
                             [config.config_sha256, config.config_sha256])
            self.assertEqual([row["deployment_config_canonical_sha256"] for row in calls],
                             ["c" * 64, "c" * 64])
            self.assertEqual([row["supervised_build_authority"] for row in calls],
                             [supervised_authority, supervised_authority])
            self.assertEqual([row["instrument_branch"] for row in calls],
                             [config.instrument_branch, config.instrument_branch])
            with self.assertRaisesRegex(F.DeploymentFactoryError, "operation identity"):
                adapters["args_factory"](
                    candidate, mock.Mock(operation_key="1" * 64),
                    {"operation_key": "2" * 64})

    def test_generated_bundle_materializes_nonoverlapping_builder_contract(self):
        """The public bundle must reach the real static contract without a build."""
        with tempfile.TemporaryDirectory() as directory:
            bundle_root = Path(directory).resolve()
            deployment_path = F.initialize_static_deployment_bundle(
                bundle_root, frozen_production_comparator=
                frozen_production_comparator(
                    Path(directory) / "authority"))
            config = F.deployment.load_deployment_config(deployment_path)
            registry = F._static_registry(config, F._template_registry())
            binding = registry["source_builder"][F._STATIC_IDS["source_builder"]]
            self.assertIsInstance(binding, F.SourceBuilderBinding)
            builder = binding.build.__self__
            self.assertIs(binding.build.__self__, builder)
            self.assertIsInstance(
                builder.composition_production_authority,
                F.cumulative_composition.FrozenProductionAuthority)
            manifest = SimpleNamespace(
                production_base_commit=config.production_head,
                instrument_commit=config.instrument_commit,
                declared_files=("ggml/src/ggml-cuda/fattn.cu",),
                patch_bundle_sha256="a" * 64,
                patch_sha256="b" * 64)
            candidate = SimpleNamespace(
                source_manifest=manifest,
                proposal={"proposal_id": "akp-static-build-root",
                          "change_class": "dispatch"})
            validated_authority = {
                "schema": "epyc.autokernel.supervised_build_authority.v2",
                "launch_spec": {}, "death_ledger": {},
                "spec_sha256": "1" * 64,
                "deployment_config_canonical_sha256": "c" * 64,
                "deployment_config_semantic_sha256": config.config_sha256}
            with mock.patch.object(
                    F.discovery_static_registry,
                    "_validate_supervised_build_authority",
                    return_value=validated_authority):
                contract, _environment = builder._contract(candidate, {
                    "instrument_branch": config.instrument_branch,
                    "deployment_config_canonical_sha256": "c" * 64,
                    "deployment_config_semantic_sha256": config.config_sha256,
                })
            self.assertEqual(Path(contract["operations_root"]), config.operations_root)
            self.assertEqual(contract["deployment_config_canonical_sha256"], "c" * 64)
            self.assertEqual(
                contract["deployment_config_semantic_sha256"], config.config_sha256)
            self.assertEqual(Path(contract["build_root"]), config.build_root)
            self.assertEqual(config.build_root, bundle_root / "builds")
            self.assertNotEqual(config.build_root, config.operations_root)
            self.assertFalse(config.build_root.is_relative_to(config.operations_root))
            self.assertFalse(config.operations_root.is_relative_to(config.build_root))


if __name__ == "__main__":
    unittest.main()
