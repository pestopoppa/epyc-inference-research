from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import os
import shutil
import stat
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import tempfile

from .. import hypothesis_portfolio, source_candidate
from . import discovery_deployment_factory as F
from . import discovery_controller as C


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

    def test_v2_templates_are_extracted_from_exact_sealed_profile(self):
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
        self.assertEqual(registry.version, "gpu-source-templates-v2")
        self.assertEqual(len(registry.templates), 10)
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

    def test_portfolio_dispatch_authority_round_trips_real_q5_and_q8_trace(self):
        portfolio = hypothesis_portfolio.load(hypothesis_portfolio.DEFAULT_PORTFOLIO)
        registry = F._template_registry()
        surfaces = F._normalized_template_surfaces(registry, portfolio)
        authority = F._portfolio_dispatch_authority(registry, portfolio)
        self.assertEqual([(row["calls"], row["grid"]) for row in
                          authority["akh-v2-q5-type-specific-dequant"]],
                         [(6063, 57344), (4644, 8192), (3096, 311296)])
        self.assertEqual([(row["calls"], row["grid"]) for row in
                          authority["akh-v2-q8-quantizer-new-mechanism"]],
                         [(15609, 1024), (3096, 5120)])
        self.assertEqual(surfaces["cuda-vecdotq-v1"]["excluded_signatures"],
                         [{"route_id": "cuda-vecdotq-v1.anchor.3",
                           "calls": 129, "grid": 57344,
                           "workgroup": 128, "lds_bytes": 512}])
        rows = F.evidence._load_dispatches(
            F._PROFILE_V3_TRACE_CSV,
            profiler_trace_schema_id=F.evidence.ROCPROF_V3_TRACE_ID,
            expected_rows=59_925)
        for hypothesis_id, template_id in (
                ("akh-v2-q5-type-specific-dequant", "cuda-vecdotq-v1"),
                ("akh-v2-q8-quantizer-new-mechanism", "cuda-quantize-q8-v1")):
            template = registry.templates[template_id]
            expected = tuple(C.BoundedDispatchExpectation(**row)
                             for row in authority[hypothesis_id])
            intent = C.GpuSourceExperimentIntent(
                template.template_id, template.target_surface,
                template.target_symbol, template.correctness_id,
                template.dispatch_id, expected)
            bound = template.bind_dispatch(intent)
            reduced = F.evidence._reduce_arm(
                rows, exact=bound.candidate_exact,
                forbidden=bound.candidate_forbidden,
                invariants=bound.invariants)
            self.assertEqual(len(reduced["exact"]), len(expected))

    def test_rocprofv3_policy_and_per_arm_cardinality_cover_all_four_strategies(self):
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
            "cuda-vecdotq-v1": (59_925, 59_925),
            "cuda-quantize-q8-v1": (59_925, 59_925),
            "cuda-fattn-tile-v1": (63_021, 59_925),
            "cuda-norm-v2": (59_925, 59_925),
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
                    linkage_sha256=hashlib.sha256(
                        (commit + "-linkage").encode()).hexdigest())
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
                    source_manifest_sha256=f"{index}" * 64)
                with mock.patch.object(
                        F.gpu_load_admission, "validate_decision_receipt"):
                    args = binding.build(candidate, build, {
                        "operation_key": build.operation_key,
                        "repetition": 1,
                        "load_admission": decision})
                args = F._bind_runner_runtime_authority(
                    config, build, {"load_admission": decision}, args)
                target = args._target_runtime_args
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
                self.assertNotEqual(
                    off["anchor_identity"]["source_commit"],
                    off["candidate_identity"]["source_commit"])

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
        production = root / "production"
        (production / "ggml/src/ggml-cuda").mkdir(parents=True)
        for relative in ("CMakeLists.txt", "ggml/src/ggml-cuda/unary.cu",
                         "ggml/src/ggml-cuda/mmvq.cu"):
            path = production / relative
            path.write_text(f"sealed {relative}\n", encoding="utf-8")
        for flavor in ("build", "build-hip"):
            binary_dir = production / flavor / "bin"
            binary_dir.mkdir(parents=True)
            for name in ("llama-server", "llama-bench"):
                shutil.copyfile("/bin/true", binary_dir / name)
            if flavor == "build-hip":
                shutil.copyfile("/bin/true", binary_dir / "libggml-hip.so.0")
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
        workload.write_text('{"workload":"decode_tg128"}', encoding="utf-8")
        runtime = root / "runtime.json"
        runtime.write_text("{}", encoding="utf-8")
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
        evidence_manifest = {"manifest_sha256": "f" * 64, "evidence": {}}
        config = SimpleNamespace(
            config_sha256="c" * 64, production_path=production.resolve(),
            production_branch=F.deployment.FROZEN_PRODUCTION_BRANCH,
            production_head="0" * 40,
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
                "template_surfaces": surfaces,
                "template_surfaces_sha256": F.schemas.content_hash(surfaces),
                "portfolio_dispatch_authority": F._portfolio_dispatch_authority(
                    templates, portfolio)}),
            hypothesis_portfolio=SimpleNamespace(value=portfolio, input=portfolio_input),
            hypothesis_evidence_manifest=SimpleNamespace(value=evidence_manifest),
            hypothesis_portfolio_contract=SimpleNamespace(
                sha256=F._PORTFOLIO_CONTRACT_SHA256),
            source_builder_id="gpu-source-v1",
            evidence_plan_id="reviewed-gpu-source-evidence-v1",
            runner_args_id="qwen05b-tg128",
            experiment_template_registry_id="gpu-source-templates-v2",
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
            root = Path(temporary) / "portfolio-v2-bundle"
            path = F.initialize_static_deployment_bundle(root)
            loaded = F.deployment.load_deployment_config(path)
            controller_config = F.controller_config(loaded, dry_run=True)
            json.dumps(controller_config.planner_context)
            portfolio = loaded.hypothesis_portfolio.value
            self.assertEqual(portfolio.sha256,
                             "c894690f56041ae355a50fffe23688abed1fa3eea9df4b7201faee2e565b4e78")
            context = loaded.planner_context.value
            self.assertEqual(
                {row["hypothesis_id"] for row in context["eligible_hypotheses"]},
                {"akh-v2-q5-type-specific-dequant",
                 "akh-v2-q8-quantizer-new-mechanism",
                 "akh-v2-fa-gqa7-pair-tail",
                 "akh-v2-rms-direct-load-reduction"},
            )
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

    def test_execution_module_attestor_refuses_any_live_byte_drift(self):
        sealed = {"runner": {"path": "/sealed/runner.py", "sha256": "a" * 64}}
        attest = F._module_attestor(sealed)
        with mock.patch.object(F, "_execution_module_identity", return_value=sealed):
            attest()
        changed = {"runner": {"path": "/sealed/runner.py", "sha256": "b" * 64}}
        with mock.patch.object(F, "_execution_module_identity", return_value=changed), \
             self.assertRaisesRegex(F.DeploymentFactoryError, "module bytes changed"):
            attest()

    def test_t0_capability_contract_is_in_exact_graph_and_tamper_attested(self):
        sealed = F._execution_module_identity()
        self.assertEqual(
            sealed["t0_provider"], {
                "path": str(Path(F.t0_provider.__file__).resolve(strict=True)),
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
        context = {"context_sha256": "a" * 64}
        config = mock.Mock(state_root=Path("/state"), evidence_root=Path("/evidence"),
                           max_iterations=2, nomination_threshold=.03,
                           planner_context=mock.Mock(value=context), production_branch="production-consolidated-v9",
                           production_head="b" * 40,
                           instrument_branch="measurement-instrument",
                           instrument_commit="c" * 40,
                           config_sha256="c" * 64,
                           experiment_template_registry_sha256="d" * 64)
        portfolio = hypothesis_portfolio.load(hypothesis_portfolio.DEFAULT_PORTFOLIO)
        config.hypothesis_portfolio = SimpleNamespace(value=portfolio)
        config.admission_policy = SimpleNamespace(
            value={"policy_sha256": "e" * 64, "examples": [], "profiles": []},
            corpus=SimpleNamespace(policy_sha256="e" * 64, version="test-v2"))
        config.revalidate = mock.Mock()
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
            with mock.patch.object(F.deployment, "resolve_registry", return_value=resolved), \
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
            self.assertEqual([row["deployment_config_sha256"] for row in calls],
                             [config.config_sha256, config.config_sha256])
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
            deployment_path = F.initialize_static_deployment_bundle(bundle_root)
            config = F.deployment.load_deployment_config(deployment_path)
            registry = F._static_registry(config, F._template_registry())
            binding = registry["source_builder"][F._STATIC_IDS["source_builder"]]
            self.assertIsInstance(binding, F.SourceBuilderBinding)
            builder = binding.build.__self__
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
            contract, _environment = builder._contract(candidate, {
                "instrument_branch": config.instrument_branch,
                "deployment_config_sha256": config.config_sha256,
            })
            self.assertEqual(Path(contract["operations_root"]), config.operations_root)
            self.assertEqual(Path(contract["build_root"]), config.build_root)
            self.assertEqual(config.build_root, bundle_root / "builds")
            self.assertNotEqual(config.build_root, config.operations_root)
            self.assertFalse(config.build_root.is_relative_to(config.operations_root))
            self.assertFalse(config.operations_root.is_relative_to(config.build_root))


if __name__ == "__main__":
    unittest.main()
