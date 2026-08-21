from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import unittest
from unittest import mock

from scripts.benchmark import run_autokernel_gpu_discovery as gpu
from scripts.kernel_rnd.autokernel import schemas


def _admission(model: Path, *, workload: str = "prefill_pp512", calls: int = 3,
               mode: str = "cold_serialized") -> dict:
    """A self-hashed lease decision; tests never use a runner-side profile."""
    request = {"effective_context_sha256": "c" * 64, "model_path": str(model.resolve()),
               "model_sha256": gpu.sha256_file(model), "model_bytes": model.stat().st_size,
               "workload": workload, "calls_per_arm": calls, "device_id": gpu.DEVICE_ID,
               "cold_load_host_bytes": model.stat().st_size,
               "worst_case_loads_per_interval": calls * 2, "telemetry_observed": False,
               "telemetry_age_ms": None, "observed_headroom_bytes_per_s": None,
               "telemetry_receipt_sha256": None, "foreign_kfd_pids": [],
               "runtime_manifest_sha256": None, "runtime_arm": None,
               "hot_residency_identity_sha256": None, "expected_hot_identity_sha256": None,
               "hot_revalidation_identity_sha256": None}
    body = {"schema": "epyc.autokernel.gpu_load_admission_decision.v1",
            "policy_version": "test-v1", "policy_sha256": "a" * 64,
            "policy_file_sha256": "b" * 64, "effective_context_sha256": "c" * 64,
            "request": request, "profile": None, "actor_recommendation": None,
            "mode": mode, "reason": "test sealed decision", "disqualifiers": [],
            "promotion_claim": False}
    return {**body, "decision_sha256": schemas.content_hash(body)}


def _bind_admission(args: argparse.Namespace, *, mode: str = "cold_serialized") -> argparse.Namespace:
    decision = _admission(Path(args.model), workload=args.workload, calls=args.calls, mode=mode)
    args.load_admission_decision = decision
    args.load_admission_policy_version = "test-v1"
    args.load_admission_policy_sha256 = "a" * 64
    args.load_admission_policy_file_sha256 = "b" * 64
    args.load_admission_effective_context_sha256 = "c" * 64
    return args


def _readiness(model: Path, *, arm: str = "anchor") -> gpu.LoadReadinessPolicy:
    body = {
        "schema": "epyc.autokernel.gpu_load_readiness_policy.v1",
        "runtime_root": "/sealed/runtime",
        "runtime_manifest_sha256": "d" * 64,
        "runtime_arm": arm,
        "model_path": str(model.resolve()),
        "model_sha256": gpu.sha256_file(model),
        "device_id": gpu.DEVICE_ID,
    }
    return gpu.LoadReadinessPolicy(
        schema=body["schema"], runtime_root=Path(body["runtime_root"]),
        runtime_manifest_sha256=body["runtime_manifest_sha256"],
        runtime_arm=arm, model_path=model.resolve(),
        model_sha256=body["model_sha256"], device_id=gpu.DEVICE_ID,
        policy_sha256=schemas.content_hash(body))


def _build(root: Path, *, rocwmma: str, mfma: str, graphs: str = "ON") -> Path:
    build = root / f"build-{rocwmma}-{mfma}"
    bindir = build / "bin"
    bindir.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text(
        f"GGML_HIP_ROCWMMA_FATTN:BOOL={rocwmma}\n"
        f"GGML_HIP_MMQ_MFMA:BOOL={mfma}\n"
        f"GGML_HIP_GRAPHS:BOOL={graphs}\n", encoding="utf-8")
    binary = bindir / "llama-bench"
    binary.write_bytes(b"sealed-binary")
    binary.chmod(0o755)
    (bindir / "libggml-hip.so").write_bytes(b"sealed-hip-dso")
    return build


def _source_identity(build: Path, commit: str) -> dict[str, str]:
    return {
        "source_commit": commit,
        "source_sha256": hashlib.sha256((commit + "-source").encode()).hexdigest(),
        "binary_sha256": gpu.sha256_file(build / "bin/llama-bench"),
        "hip_library_sha256": gpu.sha256_file(
            (build / "bin/libggml-hip.so").resolve(strict=True)),
        "config_sha256": gpu.sha256_file(build / "CMakeCache.txt"),
        "linkage_sha256": hashlib.sha256((commit + "-linkage").encode()).hexdigest(),
    }


class TestGpuDiscoveryBuildIdentity(unittest.TestCase):
    def test_cli_hydrates_exact_sealed_load_admission_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model.gguf"; model.write_bytes(b"model")
            profile = {
                "profile_id": "mi210-test-v1", "model_path": str(model),
                "model_sha256": gpu.sha256_file(model),
                "model_bytes": model.stat().st_size,
                "workload": "decode_tg128", "calls_per_arm": 9,
                "device_id": gpu.DEVICE_ID,
                "cold_load_host_bytes": model.stat().st_size,
                "worst_case_loads_per_interval": 18,
                "minimum_headroom_bytes_per_s": 1,
                "telemetry_max_age_ms": 1000,
                "evidence_sha256": "1" * 64,
            }
            examples = []
            for identifier, polarity in (("positive", "positive"),
                                         ("negative", "negative")):
                examples.append({
                    "id": identifier, "polarity": polarity,
                    "facts": {"profile_id": profile["profile_id"],
                              "state": identifier},
                    "missing": [] if polarity == "positive" else ["telemetry"],
                    "mode": ("cold_overlap" if polarity == "positive"
                             else "cold_serialized"),
                    "rationale": identifier,
                    "disqualifiers": ([] if polarity == "positive"
                                       else ["telemetry_missing"]),
                    "counterfactual": "use exact facts",
                    "evidence": ["sha256:" + ("2" if polarity == "positive"
                                                else "3") * 64],
                })
            policy_body = {
                "schema": gpu.gpu_load_admission.POLICY_SCHEMA,
                "version": "test-v1", "profiles": [profile],
                "examples": examples}
            policy_body["policy_sha256"] = schemas.content_hash(policy_body)
            policy = root / "policy.json"
            policy.write_text(json.dumps(policy_body, sort_keys=True),
                              encoding="utf-8")
            policy_file_sha = gpu.sha256_file(policy)
            corpus = gpu.gpu_load_admission.load_policy_corpus(
                policy, expected_file_sha256=policy_file_sha)
            effective = "4" * 64
            request = gpu.gpu_load_admission.AdmissionRequest(
                effective_context_sha256=effective,
                model_path=str(model), model_sha256=gpu.sha256_file(model),
                model_bytes=model.stat().st_size, workload="decode_tg128",
                calls_per_arm=9, device_id=gpu.DEVICE_ID,
                cold_load_host_bytes=model.stat().st_size,
                worst_case_loads_per_interval=18,
                telemetry_observed=False, telemetry_age_ms=None,
                observed_headroom_bytes_per_s=None,
                telemetry_receipt_sha256=None)
            decision = gpu.gpu_load_admission.arbitrate(corpus, request).to_dict()
            decision_path = root / "decision.json"
            decision_path.write_text(json.dumps(decision, sort_keys=True),
                                     encoding="utf-8")
            argv = [
                "--anchor-build", "/anchor", "--candidate-build", "/candidate",
                "--model", str(model), "--output-dir", str(root / "output"),
                "--campaign-id", "ak-cli-frame", "--workload", "decode_tg128",
                "--calls", "9", "--load-admission-decision", str(decision_path),
                "--load-admission-policy", str(policy),
                "--load-admission-policy-sha256", policy_file_sha,
                "--effective-context-sha256", effective]
            hydrated = gpu._hydrate_cli_load_admission(
                gpu.parser().parse_args(argv))
            self.assertEqual(hydrated.load_admission_decision, decision)
            self.assertEqual(hydrated.load_admission_policy_version,
                             corpus.version)
            self.assertEqual(hydrated.load_admission_policy_sha256,
                             corpus.policy_sha256)
            with self.assertRaisesRegex(RuntimeError, "CLI frame refused"):
                gpu._hydrate_cli_load_admission(gpu.parser().parse_args([
                    *argv[:-3], "0" * 64, *argv[-2:]]))

    def test_parser_and_preflight_preserve_sealed_arm_schedule(self) -> None:
        parsed = gpu.parser().parse_args([
            "--anchor-build", "/anchor", "--candidate-build", "/candidate",
            "--model", "/model", "--output-dir", "/output",
            "--campaign-id", "ak-order", "--arm-order-schedule", "candidate,anchor",
            "--arm-order-seed-sha256", "d" * 64])
        self.assertEqual(parsed.arm_order_schedule, "candidate,anchor")
        self.assertEqual(parsed.arm_order_seed_sha256, "d" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root / "build", rocwmma="ON", mfma="OFF")
            model = root / "model"; model.write_bytes(b"model")
            args = _bind_admission(argparse.Namespace(
                model=str(model), anchor_build=str(build), candidate_build=str(build),
                factor="flash_attention", campaign_id="ak-order", calls=3,
                workload="prefill_pp512", device_id=gpu.DEVICE_ID,
                inference_window_lock=None, arm_order_schedule="candidate,anchor",
                arm_order_seed_sha256="d" * 64))
            sealed = gpu.preflight(args)
        self.assertEqual(sealed["arm_order_schedule"], ["candidate", "anchor"])
        self.assertEqual(sealed["arm_order_seed_sha256"], "d" * 64)

    def test_source_patch_accepts_shared_reward_binary_and_distinct_hip_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="ON", mfma="OFF")
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF")
            # Candidate source identity differs, but one anchor-built benchmark
            # executable is used for both arms; only HIP loading may differ.
            def git_commit(argv, **_kwargs):
                return mock.Mock(
                    stdout=(gpu.READY_CONTINUE_INSTRUMENT_COMMIT
                            if str(anchor) in argv else "b" * 40), returncode=0)
            with mock.patch.object(gpu.subprocess, "run", side_effect=git_commit):
                model = root / "model.gguf"; model.write_bytes(b"model")
                for build, payload in ((anchor, b"anchor-hip"), (candidate, b"candidate-hip")):
                    directory = build / "bin"
                    (directory / "libggml-hip.so").unlink()
                    versioned = directory / "libggml-hip.so.0.16.0"; versioned.write_bytes(payload)
                    (directory / "libggml-hip.so.0").symlink_to(versioned.name)
                    (directory / "libggml-hip.so").symlink_to("libggml-hip.so.0")
                args = _bind_admission(argparse.Namespace(model=str(model), anchor_build=str(anchor), candidate_build=str(candidate),
                    factor="source_patch", campaign_id="gpu-source", calls=3, workload="prefill_pp512",
                    measurement_binary=str(anchor / "bin" / "llama-bench"),
                    common_loader_dir=str(anchor / "bin"), anchor_loader_dir=str(anchor / "bin"), candidate_loader_dir=str(candidate / "bin"),
                    device_id=gpu.DEVICE_ID,
                    inference_window_lock=None), mode="cold_overlap")
                args._sealed_anchor_source_build_identity = _source_identity(
                    anchor, gpu.READY_CONTINUE_INSTRUMENT_COMMIT)
                args._sealed_candidate_source_build_identity = _source_identity(
                    candidate, "b" * 40)
                args._operation_key = "9" * 64
                sealed = gpu.preflight(args)
            self.assertEqual(sealed["runtime_arms"]["measurement_binary_sha256"],
                             gpu.sha256_file(anchor / "bin" / "llama-bench"))
            self.assertNotEqual(sealed["runtime_arms"]["anchor_hip_sha256"],
                                sealed["runtime_arms"]["candidate_hip_sha256"])
            self.assertTrue(sealed["timed_output_oracle"]["enabled"])

    def test_source_patch_refuses_cli_identity_invention_and_live_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="ON", mfma="OFF")
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            for build, payload in ((anchor, b"anchor-hip"),
                                   (candidate, b"candidate-hip")):
                bindir = build / "bin"
                (bindir / "libggml-hip.so").unlink()
                versioned = bindir / "libggml-hip.so.0.16.0"
                versioned.write_bytes(payload)
                (bindir / "libggml-hip.so.0").symlink_to(versioned.name)
                (bindir / "libggml-hip.so").symlink_to("libggml-hip.so.0")
            args = _bind_admission(argparse.Namespace(
                model=str(model), anchor_build=str(anchor),
                candidate_build=str(candidate), factor="source_patch",
                campaign_id="gpu-source-refusal", calls=3,
                workload="prefill_pp512",
                measurement_binary=str(anchor / "bin/llama-bench"),
                common_loader_dir=str(anchor / "bin"),
                anchor_loader_dir=str(anchor / "bin"),
                candidate_loader_dir=str(candidate / "bin"),
                device_id=gpu.DEVICE_ID, inference_window_lock=None),
                mode="cold_overlap")
            args._operation_key = "9" * 64
            with self.assertRaisesRegex(RuntimeError, "sealed builder identity"):
                gpu.preflight(args)
            args._sealed_anchor_source_build_identity = _source_identity(
                anchor, gpu.READY_CONTINUE_INSTRUMENT_COMMIT)
            args._sealed_candidate_source_build_identity = _source_identity(
                candidate, "b" * 40)
            (candidate / "CMakeCache.txt").write_text(
                "GGML_HIP_ROCWMMA_FATTN:BOOL=ON\n"
                "GGML_HIP_MMQ_MFMA:BOOL=OFF\n"
                "GGML_HIP_GRAPHS:BOOL=ON\n# drift\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "live artifact differs"):
                gpu.preflight(args)
            parser_options = {
                option for action in gpu.parser()._actions
                for option in action.option_strings}
            self.assertFalse(any("source-build-identity" in option
                                 for option in parser_options))

    def test_preflight_records_cold_serialization_for_over_budget_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build = _build(root, rocwmma="ON", mfma="OFF"); model = root / "m"; model.write_bytes(b"x")
            sealed = gpu.preflight(_bind_admission(argparse.Namespace(model=str(model), anchor_build=str(build), candidate_build=str(build),
                    factor="flash_attention", campaign_id="gpu", calls=3, workload="prefill_pp512",
                    device_id=gpu.DEVICE_ID, inference_window_lock=None)))
            self.assertEqual(sealed["host_transfer"]["mode"], "cold_serialized")

    def test_preflight_refuses_unsealed_ready_continue_instrument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "m"; model.write_bytes(b"x")
            args = _bind_admission(argparse.Namespace(
                model=str(model), anchor_build=str(build), candidate_build=str(build),
                factor="flash_attention", campaign_id="gpu", calls=3,
                workload="prefill_pp512", device_id=gpu.DEVICE_ID,
                inference_window_lock=None, instrument_ready_continue_v1=True,
                instrument_ready_continue_commit="wrong",
                instrument_ready_continue_contract_sha256="wrong"))
            with self.assertRaisesRegex(RuntimeError, "sealed 81bf32f11"):
                gpu.preflight(args)
    def test_decode_preflight_seals_tg128_shape_and_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="ON", mfma="OFF",
                            graphs="ON")
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF",
                               graphs="OFF")
            model = root / "model.gguf"
            model.write_bytes(b"model")
            sealed = gpu.preflight(_bind_admission(argparse.Namespace(
                model=str(model), anchor_build=str(anchor),
                candidate_build=str(candidate), factor="hip_graphs",
                campaign_id="gpu-decode", calls=9, workload="decode_tg128",
                device_id=gpu.DEVICE_ID, inference_window_lock=None)))
        self.assertEqual(sealed["frame"], "tg128-ngl99")
        self.assertEqual(sealed["metric"], "decode_tokens_per_s")
        self.assertEqual((sealed["prompt_tokens"], sealed["generation_tokens"]),
                         (0, 128))
        self.assertEqual(sealed["cpu_overlap_policy"], "cold_serialized_load_window")
        self.assertFalse(sealed["promotion_claim"])
        self.assertIn("host_transfer", sealed)

    def test_seals_factor_binary_and_dsos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            identity = gpu.build_identity(
                _build(Path(directory), rocwmma="ON", mfma="OFF"))
        self.assertEqual(identity["source_commit"], gpu.SOURCE_COMMIT)
        self.assertTrue(identity["rocwmma_fattn"])
        self.assertFalse(identity["mmq_mfma"])
        self.assertEqual(len(identity["artifacts"]["binary_sha256"]), 64)
        self.assertEqual(set(identity["artifacts"]["libraries"]), {"libggml-hip.so"})

    def test_refuses_an_unsealed_factor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF")
            (build / "CMakeCache.txt").write_text(
                "GGML_HIP_ROCWMMA_FATTN:BOOL=ON\n"
                "GGML_HIP_GRAPHS:BOOL=ON\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "GGML_HIP_MMQ_MFMA"):
                gpu.build_identity(build)

    def test_flash_attention_factor_requires_one_r1m0_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF").resolve()
            identity = gpu.build_identity(build)
            factor = gpu.factor_spec(
                factor="flash_attention", anchor_build=build, candidate_build=build,
                anchor_identity=identity, candidate_identity=identity)
        self.assertEqual(factor["name"], "flash_attention")
        self.assertFalse(factor["anchor_flash_attention"])
        self.assertTrue(factor["candidate_flash_attention"])

    def test_flash_attention_factor_refuses_distinct_builds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="ON", mfma="OFF").resolve()
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF").resolve()
            with self.assertRaisesRegex(RuntimeError, "one identical build path"):
                gpu.factor_spec(
                    factor="flash_attention", anchor_build=anchor,
                    candidate_build=candidate,
                    anchor_identity=gpu.build_identity(anchor),
                    candidate_identity=gpu.build_identity(candidate))

    def test_rocwmma_factor_keeps_mmq_off_and_flash_on(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="OFF", mfma="OFF").resolve()
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF").resolve()
            factor = gpu.factor_spec(
                factor="rocwmma_fattn", anchor_build=anchor,
                candidate_build=candidate,
                anchor_identity=gpu.build_identity(anchor),
                candidate_identity=gpu.build_identity(candidate))
        self.assertEqual(factor["name"], "GGML_HIP_ROCWMMA_FATTN")
        self.assertEqual((factor["anchor"], factor["candidate"]), ("OFF", "ON"))
        self.assertTrue(factor["anchor_flash_attention"])
        self.assertTrue(factor["candidate_flash_attention"])

    def test_rocwmma_factor_refuses_mmq_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="OFF", mfma="ON").resolve()
            candidate = _build(root / "candidate", rocwmma="ON", mfma="ON").resolve()
            with self.assertRaisesRegex(RuntimeError, "MMQ_MFMA=OFF"):
                gpu.factor_spec(
                    factor="rocwmma_fattn", anchor_build=anchor,
                    candidate_build=candidate,
                    anchor_identity=gpu.build_identity(anchor),
                    candidate_identity=gpu.build_identity(candidate))

    def test_hip_graphs_factor_keeps_other_compile_factors_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="ON", mfma="OFF",
                            graphs="ON").resolve()
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF",
                               graphs="OFF").resolve()
            factor = gpu.factor_spec(
                factor="hip_graphs", anchor_build=anchor,
                candidate_build=candidate,
                anchor_identity=gpu.build_identity(anchor),
                candidate_identity=gpu.build_identity(candidate))
        self.assertEqual(factor["name"], "GGML_HIP_GRAPHS")
        self.assertEqual((factor["anchor"], factor["candidate"]), ("ON", "OFF"))
        self.assertTrue(factor["anchor_flash_attention"])
        self.assertTrue(factor["candidate_flash_attention"])

    def test_hip_graphs_factor_refuses_other_compile_factor_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="ON", mfma="OFF",
                            graphs="ON").resolve()
            candidate = _build(root / "candidate", rocwmma="OFF", mfma="OFF",
                               graphs="OFF").resolve()
            with self.assertRaisesRegex(RuntimeError, "ROCWMMA_FATTN identical"):
                gpu.factor_spec(
                    factor="hip_graphs", anchor_build=anchor,
                    candidate_build=candidate,
                    anchor_identity=gpu.build_identity(anchor),
                    candidate_identity=gpu.build_identity(candidate))

    def test_source_patch_requires_matching_compile_frame_and_distinct_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="OFF", mfma="OFF").resolve()
            candidate = _build(root / "candidate", rocwmma="OFF", mfma="OFF").resolve()
            (candidate / "bin" / "llama-bench").write_bytes(b"source-candidate")
            anchor_identity = gpu.build_identity(anchor)
            candidate_identity = gpu.build_identity(candidate)
            candidate_identity["source_commit"] = "1" * 40
            factor = gpu.factor_spec(
                factor="source_patch", anchor_build=anchor,
                candidate_build=candidate, anchor_identity=anchor_identity,
                candidate_identity=candidate_identity)
        self.assertEqual(factor["name"], "source_patch")
        self.assertEqual(factor["candidate"], "1" * 12)
        self.assertTrue(factor["anchor_flash_attention"])
        self.assertTrue(factor["candidate_flash_attention"])


class TestGpuDiscoveryInferenceWindow(unittest.TestCase):
    def test_three_mode_load_policy_fails_closed_to_serialized_or_refusal(self) -> None:
        transfer = {"admitted": True}
        self.assertEqual(gpu.decide_load_mode(hot_resident=True, residency_identity_matches=True,
            host_observation_available=False, transfer=transfer, dedicated_window_available=False)["mode"], "hot_resident")
        self.assertEqual(gpu.decide_load_mode(hot_resident=False, residency_identity_matches=False,
            host_observation_available=True, transfer=transfer, dedicated_window_available=False)["mode"], "cold_overlap")
        self.assertEqual(gpu.decide_load_mode(hot_resident=False, residency_identity_matches=False,
            host_observation_available=False, transfer=transfer, dedicated_window_available=True)["mode"], "cold_serialized")
        with self.assertRaisesRegex(RuntimeError, "cannot be admitted"):
            gpu.decide_load_mode(hot_resident=False, residency_identity_matches=False,
                host_observation_available=False, transfer={"admitted": False}, dedicated_window_available=False)
    def test_parser_exposes_distinct_decode_workload(self) -> None:
        parsed = gpu.parser().parse_args([
            "--anchor-build", "/anchor", "--candidate-build", "/candidate",
            "--model", "/model", "--output-dir", "/out",
            "--campaign-id", "gpu-decode", "--workload", "decode_tg128",
        ])
        self.assertEqual(parsed.workload, "decode_tg128")

    def test_small_model_overlap_records_claims_without_cpu_exclusivity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "small.gguf"
            model.write_bytes(b"model")
            claims = {"regions": {"q0": [{"held": True, "role": "autokernel",
                "holder_pids": [123], "attribution": {"campaign_id": "cpu-r4"}}]}}
            with mock.patch.object(gpu.cpu_region_claim, "inspect_region_claims",
                                   return_value=claims), \
                 mock.patch.object(gpu, "_invoke_locked", return_value={"metric": 1.0}), \
                 mock.patch.object(gpu.cpu_region_claim, "acquire_cpu_region_claim") as acquire:
                result = gpu.invoke(
                    build=Path("/build"), model=model, seed=1, baseline_vram=0,
                    flash_attention=True, campaign_id="gpu-s2",
                    cpu_journal=mock.Mock(), sealed_load_decision=_admission(model, mode="cold_overlap"))
        acquire.assert_not_called()
        self.assertIsNone(result["inference_call_window"])
        coverage = result["cpu_coverage"]
        self.assertEqual(coverage["cpu_overlap_policy"], "allowed_discovery_noise")
        self.assertFalse(coverage["cpu_exclusivity"])
        self.assertFalse(coverage["promotion_claim"])
        self.assertEqual(coverage["concurrent_claims"][0]["attribution"]["campaign_id"],
                         "cpu-r4")

    def test_overlap_without_sealed_profile_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "large.gguf"
            model.write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "sealed cold load"):
                gpu.invoke(build=Path("/build"), model=model, seed=1,
                           baseline_vram=0, flash_attention=True,
                           campaign_id="gpu-s2", cpu_journal=mock.Mock())

    def test_transfer_budget_is_diagnostic_not_live_cli_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "small.gguf"; model.write_bytes(b"x" * 100)
            with mock.patch.object(gpu.cpu_region_claim, "inspect_region_claims", return_value={"regions": {}}), \
                 mock.patch.object(gpu, "_invoke_locked", return_value={"metric": 1.0}):
                decision = gpu.host_transfer_admission(bytes_per_cold_load=100, cold_loads=1,
                    interval_s=1.0, host_bandwidth_bytes_s=1_000, conservative_fraction=.2)
            self.assertEqual(decision["mode"], "cold_overlap")

    def test_owned_cpu_coverage_exists_only_inside_model_call(self) -> None:
        events = []

        class Lease:
            path = Path("/tmp/window")
            waited_s = 0.125
            held = True

            def release(self):
                if self.held:
                    events.append("window-close")
                    self.held = False

        class CallWindow:
            def acquire(self):
                events.append("window-open")
                return Lease()

        class Receipt:
            def to_dict(self):
                return {"claim_id": "cpu-owned"}

        class Claim:
            def receipt(self):
                return Receipt()

            def release(self):
                events.append("cpu-release")

        def measured(**kwargs):
            events.append("model-call")
            return {"metric": 1.0}

        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(gpu, "MODEL_CALL_WINDOW", CallWindow()), \
             mock.patch.object(gpu.cpu_region_claim, "acquire_cpu_region_claim",
                               return_value=Claim()), \
             mock.patch.object(gpu, "_invoke_locked", side_effect=measured):
            model = Path(directory) / "model"; model.write_bytes(b"model")
            result = gpu.invoke(
                build=Path("/build"), model=model, seed=1,
                baseline_vram=0, flash_attention=True,
                campaign_id="ak-gpu-test", cpu_journal=mock.Mock(),
                sealed_load_decision=_admission(model), runtime_arm="anchor",
                load_readiness_policy=_readiness(model))
        self.assertEqual(events, ["window-open", "model-call", "cpu-release",
                                  "window-close"])
        self.assertFalse(result["cpu_coverage"]["borrowed"])

    def test_contended_cpu_claim_borrows_only_inside_model_window(self) -> None:
        events = []

        class Lease:
            path = Path("/tmp/window")
            waited_s = 0.0
            held = True

            def release(self):
                if self.held:
                    events.append("window-close")
                    self.held = False

        class CallWindow:
            def acquire(self):
                events.append("window-open")
                return Lease()

        class Borrowed:
            borrowed = True

            def validate(self):
                events.append("borrow-validate")

            def to_dict(self):
                return {"borrowed": True, "claim_id": "cpu-live-controls"}

        def acquire(*_args, **_kwargs):
            raise gpu.cpu_region_claim.CpuRegionClaimTimeout("held")

        def borrow(_cpu_list):
            events.append("borrow-open")
            return Borrowed()

        def measured(**kwargs):
            events.append("model-call")
            return {"metric": 1.0}

        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(gpu, "MODEL_CALL_WINDOW", CallWindow()), \
             mock.patch.object(gpu.cpu_region_claim, "acquire_cpu_region_claim",
                               side_effect=acquire), \
             mock.patch.object(gpu.inference_window, "borrow_windowed_cpu_coverage",
                               side_effect=borrow), \
             mock.patch.object(gpu, "_invoke_locked", side_effect=measured):
            model = Path(directory) / "model"; model.write_bytes(b"model")
            result = gpu.invoke(
                build=Path("/build"), model=model, seed=1,
                baseline_vram=0, flash_attention=True,
                campaign_id="ak-gpu-test", cpu_journal=mock.Mock(),
                sealed_load_decision=_admission(model), runtime_arm="anchor",
                load_readiness_policy=_readiness(model))
        self.assertEqual(events, ["window-open", "borrow-open", "model-call",
                                  "borrow-validate", "window-close"])
        self.assertTrue(result["cpu_coverage"]["borrowed"])


class TestGpuDiscoveryBatchedSubprocess(unittest.TestCase):
    class _Process:
        pid = 991

        def __init__(self, stdout: str, *, running_polls: int = 1):
            self._stdout = stdout
            self._running_polls = running_polls
            self._polls = 0
            self.returncode = 0
            self.terminated = False
            self.killed = False
            self.waited = False

        def poll(self):
            self._polls += 1
            return None if self._polls <= self._running_polls else self.returncode

        def communicate(self, timeout):
            self.returncode = 0
            return self._stdout, ""

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout):
            self.waited = True
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    @staticmethod
    def _row(samples: list[float]) -> str:
        count = len(samples)
        samples_ns = [max(1, round(512e9 / value)) for value in samples]
        input_hashes = [f"{index + 1:016x}" for index in range(count)]
        output_hashes = [f"{index + 101:016x}" for index in range(count)]
        payload = __import__("json").dumps({
            "backends": "ROCm", "gpu_info": "AMD Instinct MI210",
            "build_commit": "0db32c0", "n_prompt": 512, "n_gen": 0,
            "flash_attn": 1, "n_threads": 8, "n_batch": 512, "n_ubatch": 512,
            "use_mmap": True, "no_op_offload": 0, "split_mode": "layer",
            "no_kv_offload": False, "poll": 50, "avg_ns": sum(samples_ns) // count,
            "samples_ns": samples_ns, "avg_ts": "__AVG_TS__", "samples_ts": samples,
            "autokernel_hardened": True,
            "autokernel_output_invariant": True,
            "autokernel_hybrid_ab_complete": True,
            "autokernel_thread_set_stable": True,
            "autokernel_escape_checks_complete": True,
            "autokernel_input_working_set_bytes": 4096 * count,
            "autokernel_input_hashes": ",".join(input_hashes),
            "autokernel_input_addresses": ",".join(
                f"0x{2 * index + 1:x}/0x{2 * index + 2:x}" for index in range(count)),
            "autokernel_context_addresses": ",".join(
                f"0x{2 * count + 2 * index + 1:x}/0x{2 * count + 2 * index + 2:x}"
                for index in range(count)),
            "autokernel_output_hashes": ",".join(
                f"{value}/{value}" for value in output_hashes),
            "autokernel_unsynchronized_samples_ns": ",".join(
                str(max(1, round(512e9 / value))) for value in samples),
            "autokernel_thread_set_hashes": ",".join(
                ["00000000000000aa/00000000000000aa/"
                 "00000000000000aa/00000000000000aa"] * count),
            "autokernel_device_sync_mode": "hip_full_device",
        })
        return payload.replace('"__AVG_TS__"', f"{sum(samples) / count:.6f}") + "\n"

    def test_one_process_collects_exactly_nine_native_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF")
            model = Path(directory) / "model.gguf"; model.write_bytes(b"model")
            process = self._Process(self._row([100.0 + i for i in range(9)]))
            seen = []
            result = gpu._invoke_locked(
                build=build, model=model, seed=8613, baseline_vram=0,
                flash_attention=True, expected_source_commit=gpu.SOURCE_COMMIT,
                repetitions=9,
                process_factory=lambda argv, **kwargs: (seen.append((argv, kwargs)) or process),
                kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                pgid_provider=lambda pid: process.pid, sleep=lambda _: None)
        self.assertEqual(seen[0][0][seen[0][0].index("-r") + 1], "9")
        for actual, reported in zip(result["samples"],
                                    [100.0 + i for i in range(9)]):
            self.assertAlmostEqual(actual, reported, places=6)
        self.assertEqual(result["sample_count"], 9)
        self.assertAlmostEqual(result["metric"], 104.0, places=6)
        self.assertEqual(result["native_metric_diagnostic"]["samples_ns"],
                         result["raw_row"]["samples_ns"])
        self.assertNotIn("AMD_SERIALIZE_KERNEL", seen[0][1]["env"])
        self.assertNotIn("AMD_SERIALIZE_COPY", seen[0][1]["env"])
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", seen[0][1]["env"])
        self.assertEqual(result["metric_contract"], {
            "schema": "epyc.autokernel.native_llama_bench_metric.v1",
            "scope": "legacy_nonpromotable_discovery",
            "production_throughput_authority": False,
        })

    def test_native_decimal_contract_rederives_tg128_from_integer_nanoseconds(self) -> None:
        row = {
            "avg_ns": 380474161, "samples_ns": [380474161],
            "avg_ts": "__AVG__", "samples_ts": [336.422],
        }
        payload = json.dumps(row).replace(
            '"__AVG__"', "336.422320").encode() + b"\n"
        _parsed, diagnostic = gpu._parse_native_measurement(
            payload, repetitions=1, tokens_per_repetition=128)
        self.assertEqual(diagnostic["reported_avg_ts_decimal"], "336.422320")
        self.assertEqual(diagnostic["reported_samples_ts_decimal"], ["336.422"])
        self.assertAlmostEqual(diagnostic["rederived_avg_ts"], 336.4223199,
                               places=6)
        self.assertEqual(diagnostic["integer_timing_authority"], "samples_ns")
        _row, integer_sample = gpu._parse_native_measurement(
            b'{"avg_ns":5120000000,"samples_ns":[5120000000],'
            b'"avg_ts":100.000000,"samples_ts":[100]}\n',
            repetitions=1, tokens_per_repetition=512)
        self.assertEqual(integer_sample["reported_samples_ts_decimal"], ["100"])

    def test_native_decimal_contract_refuses_rounding_nan_duplicate_and_truncation(self) -> None:
        valid = (b'{"avg_ns":380474161,"samples_ns":[380474161],'
                 b'"avg_ts":336.422320,"samples_ts":[336.422]}\n')
        mutations = (
            valid.replace(b"336.422320", b"336.422999"),
            valid.replace(b"336.422]", b"336.499]"),
            valid.replace(b'"avg_ns":380474161,',
                          b'"avg_ns":380474161,"avg_ns":380474161,'),
            valid.replace(b"336.422320", b"NaN"),
            valid[:-1],
            valid + b"\n",
        )
        for payload in mutations:
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                gpu._parse_native_measurement(
                    payload, repetitions=1, tokens_per_repetition=128)

    def test_native_decimal_contract_covers_pp512_and_precision_boundaries(self) -> None:
        for tokens, samples_ns in (
                (512, [204893812, 204800000, 205010101]),
                (128, [128001, 128000, 127999]),
                (128, [10_368_071_631_799_082])):
            with self.subTest(tokens=tokens, samples_ns=samples_ns):
                exact = [1e9 * tokens / value for value in samples_ns]
                payload = {
                    "avg_ns": sum(samples_ns) // len(samples_ns),
                    "samples_ns": samples_ns,
                    "avg_ts": "__AVG__",
                    "samples_ts": [
                        float(format(value, ".6g")) for value in exact],
                }
                raw = json.dumps(payload).replace(
                    '"__AVG__"', f"{sum(exact) / len(exact):.6f}").encode() + b"\n"
                _row, diagnostic = gpu._parse_native_measurement(
                    raw, repetitions=len(samples_ns),
                    tokens_per_repetition=tokens)
                self.assertEqual(diagnostic["tokens_per_repetition"], tokens)
                self.assertEqual(diagnostic["samples_ns"], samples_ns)

                changed = dict(payload)
                changed["samples_ns"] = [
                    samples_ns[0] + max(1000, samples_ns[0] // 100),
                    *samples_ns[1:]]
                changed["avg_ns"] = sum(changed["samples_ns"]) // len(samples_ns)
                bad = json.dumps(changed).replace(
                    '"__AVG__"', f"{sum(exact) / len(exact):.6f}").encode() + b"\n"
                with self.assertRaises(RuntimeError):
                    gpu._parse_native_measurement(
                        bad, repetitions=len(samples_ns),
                        tokens_per_repetition=tokens)
                with self.assertRaises(RuntimeError):
                    gpu._parse_native_measurement(
                        raw, repetitions=len(samples_ns),
                        tokens_per_repetition=tokens + 1)

    def test_completed_process_checkpoint_resumes_without_process_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            receipt_root = root / "process-anchor"
            process = self._Process(self._row([100.0] * 3))
            calls = []

            def factory(*_args, **_kwargs):
                calls.append("process")
                return process

            with self.assertRaisesRegex(RuntimeError, "crash after process"):
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True,
                    expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=3, process_factory=factory,
                    kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                    pgid_provider=lambda _pid: process.pid, sleep=lambda _: None,
                    process_receipt_root=receipt_root,
                    process_context={"operation": "test", "arm": "anchor"},
                    process_resource_context={"claim_id": "akd-test"},
                    after_process_checkpoint=lambda _path: (_ for _ in ()).throw(
                        RuntimeError("crash after process")))
            self.assertTrue((receipt_root / "stdout.bin").is_file())
            result = gpu._invoke_locked(
                build=build, model=model, seed=8613, baseline_vram=0,
                flash_attention=True,
                expected_source_commit=gpu.SOURCE_COMMIT,
                repetitions=3,
                process_factory=lambda *_args, **_kwargs: self.fail(
                    "completed process was replayed"),
                kfd_pid_provider=lambda: (), vram_reader=lambda: 0,
                pgid_provider=lambda _pid: process.pid, sleep=lambda _: None,
                process_receipt_root=receipt_root,
                process_context={"operation": "test", "arm": "anchor"},
                process_resource_context={"claim_id": "akd-new"})
        self.assertEqual(calls, ["process"])
        self.assertAlmostEqual(result["metric"], 100.0)

    def test_malformed_completed_output_refusal_rederives_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            receipt_root = root / "process-anchor"
            malformed = self._row([100.0] * 3).replace(
                '"avg_ts": 100.000000', '"avg_ts": 101.000000')
            process = self._Process(malformed)
            with self.assertRaises(gpu.MeasurementOutputRefusal) as first:
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True,
                    expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=3,
                    process_factory=lambda *_args, **_kwargs: process,
                    kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                    pgid_provider=lambda _pid: process.pid, sleep=lambda _: None,
                    process_receipt_root=receipt_root,
                    process_context={"operation": "test", "arm": "anchor"})
            with self.assertRaises(gpu.MeasurementOutputRefusal) as reopened:
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True,
                    expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=3,
                    process_factory=lambda *_args, **_kwargs: self.fail(
                        "refused process was replayed"),
                    kfd_pid_provider=lambda: (), vram_reader=lambda: 0,
                    pgid_provider=lambda _pid: process.pid, sleep=lambda _: None,
                    process_receipt_root=receipt_root,
                    process_context={"operation": "test", "arm": "anchor"})
        self.assertEqual(first.exception.receipt_sha256,
                         reopened.exception.receipt_sha256)

    def test_output_refusal_projects_secret_free_native_timing_diagnostic(self) -> None:
        class WithPrivateStderr(self._Process):
            def communicate(self, timeout):
                self.returncode = 0
                return self._stdout, "private compiler or runtime detail"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            malformed = self._row([100.0] * 3).replace(
                '"avg_ts": 100.000000', '"avg_ts": 101.000000')
            process = WithPrivateStderr(malformed)
            with self.assertRaises(gpu.MeasurementOutputRefusal) as refused:
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True,
                    expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=3,
                    process_factory=lambda *_args, **_kwargs: process,
                    kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                    pgid_provider=lambda _pid: process.pid,
                    sleep=lambda _: None,
                    process_receipt_root=root / "process-candidate",
                    process_context={
                        "arm": "candidate", "workload": "pp512",
                        "metric": "decode_tokens_per_s",
                        "runtime_graphs": "off", "prompt_tokens": 512,
                        "generation_tokens": 0,
                        "tokens_per_repetition": 512,
                        "preflight_sha256": "1" * 64,
                    })
            receipt_bytes = Path(refused.exception.receipt_path).read_bytes()
            refusal = json.loads(receipt_bytes)
        diagnostic = refusal["diagnostic"]
        self.assertTrue(diagnostic["diagnostic_available"])
        self.assertEqual(diagnostic["measurement_identity"]["arm"],
                         "candidate")
        self.assertEqual(diagnostic["measurement_identity"]["workload"],
                         "pp512")
        self.assertEqual(diagnostic["native_fields"]["avg_ts_decimal"],
                         "101.000000")
        self.assertEqual(len(diagnostic["native_fields"]["samples_ns"]), 3)
        self.assertAlmostEqual(diagnostic["rederived"]["avg_ts"], 100.0)
        self.assertRegex(diagnostic["stdout"]["observed_sha256"],
                         r"^[0-9a-f]{64}$")
        self.assertRegex(diagnostic["stderr"]["observed_sha256"],
                         r"^[0-9a-f]{64}$")
        self.assertNotIn(b"private compiler or runtime detail", receipt_bytes)

    def test_two_arm_resume_preserves_first_checkpoint_and_reversed_order(self) -> None:
        for order in (("anchor", "candidate"), ("candidate", "anchor")):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                build = _build(root, rocwmma="ON", mfma="OFF")
                model = root / "model.gguf"; model.write_bytes(b"model")
                process_calls = []

                def invoke_arm(arm, *, malformed=False, crash=False):
                    row = self._row([100.0] * 3)
                    if malformed:
                        row = row.replace(
                            '"avg_ts": 100.000000',
                            '"avg_ts": 101.000000')

                    def factory(*_args, **_kwargs):
                        process_calls.append(arm)
                        return self._Process(row)

                    return gpu._invoke_locked(
                        build=build, model=model, seed=8613,
                        baseline_vram=0, flash_attention=True,
                        expected_source_commit=gpu.SOURCE_COMMIT,
                        repetitions=3, process_factory=factory,
                        kfd_pid_provider=lambda: (123,),
                        vram_reader=lambda: 64,
                        pgid_provider=lambda _pid: 991, sleep=lambda _: None,
                        process_receipt_root=root / f"process-{arm}",
                        process_context={"operation": "two-arm", "arm": arm},
                        after_process_checkpoint=(
                            (lambda _path: (_ for _ in ()).throw(
                                RuntimeError("crash after arm checkpoint")))
                            if crash else None))

                with self.assertRaisesRegex(RuntimeError,
                                            "crash after arm checkpoint"):
                    invoke_arm(order[0], crash=True)
                # The completed first arm is reused.  The second process runs
                # once, then its malformed output becomes a durable refusal.
                invoke_arm(order[0])
                with self.assertRaises(gpu.MeasurementOutputRefusal) as first:
                    invoke_arm(order[1], malformed=True)
                with self.assertRaises(gpu.MeasurementOutputRefusal) as reopened:
                    invoke_arm(order[1], malformed=True)
                self.assertEqual(process_calls, list(order))
                self.assertEqual(first.exception.receipt_sha256,
                                 reopened.exception.receipt_sha256)

    def test_rehashed_output_refusal_reason_does_not_override_rederivation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            receipt_root = root / "process-anchor"
            malformed = self._row([100.0] * 3).replace(
                '"avg_ts": 100.000000', '"avg_ts": 101.000000')
            process = self._Process(malformed)
            kwargs = dict(
                build=build, model=model, seed=8613, baseline_vram=0,
                flash_attention=True,
                expected_source_commit=gpu.SOURCE_COMMIT,
                repetitions=3,
                kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                pgid_provider=lambda _pid: process.pid, sleep=lambda _: None,
                process_receipt_root=receipt_root,
                process_context={"operation": "test", "arm": "anchor"})
            with self.assertRaises(gpu.MeasurementOutputRefusal):
                gpu._invoke_locked(
                    **kwargs,
                    process_factory=lambda *_args, **_kwargs: process)
            refusal_path = root / "process-anchor-refusal.json"
            refusal = json.loads(refusal_path.read_text())
            refusal["reason_code"] = "samples_ts_rounding"
            unsigned = {key: value for key, value in refusal.items()
                        if key != "receipt_sha256"}
            refusal["receipt_sha256"] = schemas.content_hash(unsigned)
            refusal_path.write_text(json.dumps(refusal, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RuntimeError,
                                        "refusal changed on reopen"):
                gpu._invoke_locked(
                    **kwargs,
                    process_factory=lambda *_args, **_kwargs: self.fail(
                        "tampered refusal replayed process"))

    def test_nonzero_process_is_durable_typed_and_not_replayed(self) -> None:
        class Nonzero(self._Process):
            def communicate(self, timeout):
                self.returncode = 2
                return "", "argparse refused"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            process = Nonzero("", running_polls=0)
            kwargs = dict(
                build=build, model=model, seed=8613, baseline_vram=0,
                flash_attention=True,
                expected_source_commit=gpu.SOURCE_COMMIT,
                repetitions=3, kfd_pid_provider=lambda: (123,),
                vram_reader=lambda: 64,
                pgid_provider=lambda _pid: process.pid, sleep=lambda _: None,
                process_receipt_root=root / "process-anchor",
                process_context={"operation": "nonzero", "arm": "anchor"})
            with self.assertRaises(gpu.MeasurementOutputRefusal) as first:
                gpu._invoke_locked(
                    **kwargs,
                    process_factory=lambda *_args, **_kwargs: process)
            self.assertIn("exited 2", str(first.exception))
            with self.assertRaises(gpu.MeasurementOutputRefusal) as reopened:
                gpu._invoke_locked(
                    **kwargs,
                    process_factory=lambda *_args, **_kwargs: self.fail(
                        "nonzero completed process replayed"))
            self.assertEqual(first.exception.receipt_sha256,
                             reopened.exception.receipt_sha256)

    def test_timed_output_oracle_requires_exact_serialization_environment(self) -> None:
        row = json.loads(self._row([100.0] * 9))
        exact = {"AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                 "GGML_CUDA_DISABLE_GRAPHS": "1"}
        invalid = [
            {key: value for key, value in exact.items() if key != missing}
            for missing in exact]
        invalid.extend(({**exact, "GGML_CUDA_DISABLE_GRAPHS": "0"},
                        {**exact, "AMD_SERIALIZE_KERNEL": "2"}))
        for incomplete in invalid:
            with self.subTest(incomplete=incomplete), \
                    self.assertRaisesRegex(RuntimeError, "exact serialized graphs-off"):
                gpu._validate_timed_output_semantics(
                    row, repetitions=9, seed=8613, tokens_per_repetition=512,
                    serialization_env=incomplete)

    def test_timed_output_oracle_is_independent_of_ready_continue_unlock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF")
            model = Path(directory) / "model.gguf"; model.write_bytes(b"model")
            process = self._Process(self._row([100.0] * 3))
            seen = []
            result = gpu._invoke_locked(
                build=build, model=model, seed=8613, baseline_vram=0,
                flash_attention=True, expected_source_commit=gpu.SOURCE_COMMIT,
                repetitions=3, timed_output_oracle=True,
                process_factory=lambda argv, **kwargs: (seen.append((argv, kwargs)) or process),
                kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                pgid_provider=lambda _pid: process.pid, sleep=lambda _: None)
        self.assertEqual(seen[0][1]["env"]["AMD_SERIALIZE_KERNEL"], "3")
        self.assertEqual(seen[0][1]["env"]["AMD_SERIALIZE_COPY"], "3")
        self.assertEqual(seen[0][1]["env"]["GGML_CUDA_DISABLE_GRAPHS"], "1")
        self.assertIn("timed_output_semantics", result)
        self.assertIsNone(result["load_readiness_witness"])
        self.assertTrue(result["timed_output_semantics"]["reward_admissible"])

    def test_pair_max_charges_slower_member_under_cache_asymmetry(self) -> None:
        row = json.loads(self._row([1_000_000_000.0]))
        row["autokernel_unsynchronized_samples_ns"] = "10000"
        receipt = gpu._validate_timed_output_semantics(
            row, repetitions=1, seed=8613, tokens_per_repetition=512,
            serialization_env={"AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                               "GGML_CUDA_DISABLE_GRAPHS": "1"})
        self.assertEqual(receipt["second_samples_ns"], [512])
        self.assertEqual(receipt["protected_samples_ns"], [10000])
        self.assertEqual(receipt["protected_samples_ts"], [51_200_000.0])
        self.assertEqual(receipt["anti_shift_witness"], "hip_serialized_pair_max")
        self.assertTrue(receipt["reward_admissible"])

    def test_timed_output_oracle_refuses_wrong_or_replayed_hashes(self) -> None:
        wrong = json.loads(self._row([100.0] * 3))
        wrong["autokernel_output_hashes"] = (
            "a000000000000065/a000000000000066,"
            "a000000000000066/a000000000000066,"
            "a000000000000067/a000000000000067")
        with self.assertRaisesRegex(RuntimeError, "not bitwise invariant"):
            gpu._validate_timed_output_semantics(
                wrong, repetitions=3, seed=8613, tokens_per_repetition=512,
                serialization_env={"AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                                   "GGML_CUDA_DISABLE_GRAPHS": "1"})
        replayed = json.loads(self._row([100.0] * 3))
        replayed["autokernel_input_hashes"] = ",".join(["a000000000000001"] * 3)
        with self.assertRaisesRegex(RuntimeError, "malformed or reused"):
            gpu._validate_timed_output_semantics(
                replayed, repetitions=3, seed=8613, tokens_per_repetition=512,
                serialization_env={"AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                                   "GGML_CUDA_DISABLE_GRAPHS": "1"})

    def test_within_arm_instability_is_operation_bound_infrastructure_ambiguity(self) -> None:
        """A bad hardening pair cannot consume science as output refusal."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root / "build", rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            wrong = self._row([100.0] * 3).replace(
                '"autokernel_hardened": true',
                '"autokernel_hardened": false')
            process = self._Process(wrong)
            operation_key = "9" * 64
            receipt_root = root / "process-anchor"
            kwargs = dict(
                build=build, model=model, seed=8613, baseline_vram=0,
                flash_attention=True,
                expected_source_commit=gpu.SOURCE_COMMIT,
                repetitions=3, timed_output_oracle=True,
                runtime_graphs="off",
                process_factory=lambda *_args, **_kwargs: process,
                kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                pgid_provider=lambda _pid: process.pid, sleep=lambda _: None,
                process_receipt_root=receipt_root,
                process_context={"operation_key": operation_key,
                                 "arm": "anchor"})
            with self.assertRaises(
                    gpu.TimedOutputInfrastructureAmbiguity) as first:
                gpu._invoke_locked(**kwargs)
            self.assertEqual(first.exception.operation_key, operation_key)
            ambiguity = json.loads(
                Path(first.exception.receipt_path).read_text(encoding="utf-8"))
            self.assertEqual(
                ambiguity["schema"],
                gpu.SCHEMA_TIMED_OUTPUT_INFRASTRUCTURE)
            self.assertFalse(ambiguity["scientific_budget_spent"])
            self.assertFalse(ambiguity["candidate_disposition"])
            self.assertTrue(ambiguity["requires_fresh_operation"])
            self.assertFalse((root / "process-anchor-refusal.json").exists())
            kwargs["process_factory"] = lambda *_args, **_kwargs: self.fail(
                "ambiguous completed process was replayed")
            with self.assertRaises(
                    gpu.TimedOutputInfrastructureAmbiguity) as reopened:
                gpu._invoke_locked(**kwargs)
            self.assertEqual(first.exception.receipt_sha256,
                             reopened.exception.receipt_sha256)

    def test_cross_arm_oracle_refuses_changed_outputs(self) -> None:
        row = json.loads(self._row([100.0] * 3))
        kwargs = {
            "repetitions": 3, "seed": 8613, "tokens_per_repetition": 512,
            "serialization_env": {
                "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                "GGML_CUDA_DISABLE_GRAPHS": "1"},
        }
        anchor = {"timed_output_semantics": gpu._validate_timed_output_semantics(row, **kwargs)}
        changed = dict(row)
        changed["autokernel_output_hashes"] = (
            "f0000000000000ff/f0000000000000ff,"
            "a000000000000066/a000000000000066,"
            "a000000000000067/a000000000000067")
        candidate = {
            "timed_output_semantics": gpu._validate_timed_output_semantics(changed, **kwargs)}
        with self.assertRaisesRegex(RuntimeError, "differ bitwise"):
            gpu._validate_cross_arm_timed_outputs(anchor, candidate)
        tampered = dict(anchor["timed_output_semantics"])
        tampered["serialization_env"] = {"AMD_SERIALIZE_KERNEL": "3"}
        with self.assertRaisesRegex(RuntimeError, "semantic receipt is invalid"):
            gpu._validate_cross_arm_timed_outputs(
                {"timed_output_semantics": tampered}, candidate)
        wrong_inputs = dict(candidate["timed_output_semantics"])
        wrong_inputs["input_hashes"] = [
            "e000000000000001", *wrong_inputs["input_hashes"][1:]]
        wrong_inputs["receipt_sha256"] = schemas.content_hash({
            key: value for key, value in wrong_inputs.items()
            if key != "receipt_sha256"})
        with self.assertRaisesRegex(
                RuntimeError, "same hidden input bank") as infrastructure:
            gpu._validate_cross_arm_timed_outputs(
                anchor, {"timed_output_semantics": wrong_inputs})
        self.assertNotIsInstance(
            infrastructure.exception, gpu._CrossArmOutputDivergence)

    def test_v24_nine_of_nine_divergence_seals_reusable_scientific_terminal(self) -> None:
        """Replay v24's same-input, internally stable, all-output mismatch."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [f"{index + 1:016x}" for index in range(9)]
            anchor_outputs = [f"a{index:015x}" for index in range(9)]
            candidate_outputs = [f"b{index:015x}" for index in range(9)]
            exact_env = {
                "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                "GGML_CUDA_DISABLE_GRAPHS": "1"}
            operation_key = "9" * 64
            output_root = (root / operation_key / "runner" / "s1" /
                           "measurement-graphs-off")
            output_root.mkdir(parents=True)

            def semantics(outputs):
                body = {
                    "schema": "epyc.autokernel.timed_output_semantics.v1",
                    "instrument_commit": gpu.READY_CONTINUE_INSTRUMENT_COMMIT,
                    "seed": 8613, "repetitions": 9,
                    "tokens_per_repetition": 128,
                    "input_hashes": inputs, "output_hashes": outputs,
                    "within_pair_bitwise_equal": True,
                    "ranked_member_device_sync": "hip_full_device",
                    "serialization_env": exact_env,
                    "first_samples_ns": [1_000_000] * 9,
                    "second_samples_ns": [1_000_000] * 9,
                    "protected_samples_ns": [1_000_000] * 9,
                    "protected_samples_ts": [128_000.0] * 9,
                    "anti_shift_witness": "hip_serialized_pair_max",
                    "reward_admissible": True,
                }
                return {**body, "receipt_sha256": schemas.content_hash(body)}

            runs = {}
            for arm, outputs in (("anchor", anchor_outputs),
                                 ("candidate", candidate_outputs)):
                receipt = output_root / f"process-{arm}" / "receipt.json"
                receipt.parent.mkdir()
                receipt.write_text(json.dumps({
                    "identity": {"process_context": {
                        "campaign_id": "ak-v24-replay",
                        "operation_key": operation_key,
                        "preflight_sha256": "f" * 64,
                        "arm": arm, "runtime_graphs": "off"}}}),
                    encoding="utf-8")
                receipt.chmod(0o600)
                runs[arm] = {
                    "timed_output_semantics": semantics(outputs),
                    "supervisor": {
                        "process_receipt_path": str(receipt.resolve()),
                        "process_receipt_file_sha256": gpu.sha256_file(receipt),
                    },
                }

            with self.assertRaises(gpu._CrossArmOutputDivergence):
                gpu._validate_cross_arm_timed_outputs(
                    runs["anchor"], runs["candidate"])
            with self.assertRaises(gpu.CandidateCorrectnessDivergence) as first:
                raise gpu._seal_candidate_correctness_divergence(
                    output_root, anchor=runs["anchor"], candidate=runs["candidate"],
                    runtime_graphs="off", campaign_id="ak-v24-replay",
                    operation_key=operation_key,
                    anchor_identity={"source_commit": "a" * 40},
                    candidate_identity={"source_commit": "b" * 40})
            with self.assertRaises(gpu.CandidateCorrectnessDivergence) as reopened:
                raise gpu._seal_candidate_correctness_divergence(
                    output_root, anchor=runs["anchor"], candidate=runs["candidate"],
                    runtime_graphs="off", campaign_id="ak-v24-replay",
                    operation_key=operation_key,
                    anchor_identity={"source_commit": "a" * 40},
                    candidate_identity={"source_commit": "b" * 40})
            self.assertEqual(first.exception.receipt_sha256,
                             reopened.exception.receipt_sha256)
            receipt_path = Path(first.exception.receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "correctness_falsified")
            self.assertEqual(receipt["classification"], "screened_out")
            self.assertTrue(receipt["scientific_budget_spent"])
            self.assertEqual(receipt["differing_repetitions"], 9)
            self.assertEqual(receipt["operation_key"], operation_key)
            self.assertFalse(receipt["target_runtime_executed"])
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
            rendered = receipt_path.read_text(encoding="utf-8")
            self.assertFalse(any(value in rendered for value in
                                 anchor_outputs + candidate_outputs))
            escaped = dict(runs["candidate"])
            escaped["supervisor"] = dict(escaped["supervisor"])
            outside = root / "outside-receipt.json"
            outside.write_bytes(Path(
                escaped["supervisor"]["process_receipt_path"]).read_bytes())
            outside.chmod(0o600)
            escaped["supervisor"]["process_receipt_path"] = str(
                outside.resolve())
            escaped["supervisor"]["process_receipt_file_sha256"] = (
                gpu.sha256_file(outside))
            with self.assertRaisesRegex(RuntimeError,
                                        "escaped its operation namespace"):
                gpu._seal_candidate_correctness_divergence(
                    output_root, anchor=runs["anchor"], candidate=escaped,
                    runtime_graphs="off", campaign_id="ak-v24-replay",
                    operation_key=operation_key,
                    anchor_identity={"source_commit": "a" * 40},
                    candidate_identity={"source_commit": "b" * 40})

    def test_serialized_readiness_does_not_unlock_on_maps_without_instrument_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF")
            model = Path(directory) / "model.gguf"; model.write_bytes(b"model")
            policy = _readiness(model)
            process = self._Process(self._row([1.0] * 9), running_polls=100)
            identity = {
                "schema": "epyc.autokernel.split_reward_runtime_maps.v1",
                "runtime_manifest_sha256": "d" * 64, "arm": "anchor",
                "model_path": str(model.resolve()), "model_sha256": gpu.sha256_file(model),
                "device_id": gpu.DEVICE_ID, "identity_sha256": "e" * 64,
            }
            with mock.patch.object(gpu, "_runtime_maps_identity", return_value=identity):
                result = gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True, expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=9, runtime_arm="anchor", readiness_policy=policy,
                    common_loader_dir=build / "bin", hip_library_dir=build / "bin",
                    process_factory=lambda *_args, **_kwargs: process,
                    kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                    pgid_provider=lambda _pid: process.pid, sleep=lambda _: None)
        self.assertEqual(result["runtime_maps_identity"], identity)

    def test_serialized_invoke_holds_claim_and_lock_through_batched_process_without_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF")
            model = Path(directory) / "model.gguf"; model.write_bytes(b"model")
            policy = _readiness(model)
            process = self._Process(self._row([1.0] * 9), running_polls=100)
            events = []

            class Lease:
                path = Path("/tmp/autokernel-test-window")
                waited_s = 0.01
                held = True

                def release(self):
                    if self.held:
                        events.append("lock-release")
                        self.held = False

            class Window:
                def acquire(self):
                    events.append("lock-acquire")
                    return Lease()

            class Claim:
                def receipt(self):
                    return mock.Mock(to_dict=lambda: {"claim_id": "cpu-test"})

                def release(self):
                    events.append("claim-release")

            identity = {
                "schema": "epyc.autokernel.split_reward_runtime_maps.v1",
                "runtime_manifest_sha256": "d" * 64, "arm": "anchor",
                "model_path": str(model.resolve()), "model_sha256": gpu.sha256_file(model),
                "device_id": gpu.DEVICE_ID, "identity_sha256": "e" * 64,
            }
            def factory(*_args, **_kwargs):
                events.append("spawn")
                return process
            def maps(**_kwargs):
                events.append("maps-witness")
                return identity
            with mock.patch.object(gpu, "MODEL_CALL_WINDOW", Window()), \
                 mock.patch.object(gpu.cpu_region_claim, "acquire_cpu_region_claim",
                                   return_value=Claim()), \
                 mock.patch.object(gpu, "_runtime_maps_identity", side_effect=maps):
                result = gpu.invoke(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True, campaign_id="ak-gpu-test", cpu_journal=mock.Mock(),
                    sealed_load_decision=_admission(model), repetitions=9,
                    runtime_arm="anchor", load_readiness_policy=policy,
                    reward_binary=build / "bin" / "llama-bench",
                    hip_library_dir=build / "bin", common_loader_dir=build / "bin",
                    process_factory=factory,
                    kfd_pid_provider=lambda: () if process.waited else (123,),
                    vram_reader=lambda: 64, pgid_provider=lambda _pid: process.pid,
                    sleep=lambda _: None)
        self.assertEqual(events, ["lock-acquire", "spawn", "maps-witness",
                                  "claim-release", "lock-release"])
        self.assertEqual(result["load_readiness_transition"]["status"],
                         "instrument_barrier_unavailable_held_through_process")
        self.assertTrue(result["inference_call_window"]["released"])

    def test_bad_native_sample_vector_refuses_without_leaking_live_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF")
            model = Path(directory) / "model.gguf"; model.write_bytes(b"model")
            process = self._Process(self._row([1.0] * 8))
            with self.assertRaisesRegex(RuntimeError, "exactly 9 positive integer samples_ns"):
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True, expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=9, process_factory=lambda *_args, **_kwargs: process,
                    kfd_pid_provider=lambda: (123,), vram_reader=lambda: 64,
                    pgid_provider=lambda _pid: process.pid, sleep=lambda _: None)
        self.assertFalse(process.terminated)
        self.assertEqual(process.returncode, 0)

    def test_governed_ready_continue_binds_pid_seed_reps_and_releases_before_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            policy = _readiness(model)
            decision = _admission(model)
            handshake = gpu.ReadyContinueHandshake.create(
                root=root / "barrier", decision=decision, policy=policy,
                arm="anchor", seed=8613, repetitions=9)
            row = json.loads(self._row([1.0] * 9))
            row["autokernel_unsynchronized_samples_ns"] = ",".join(
                [str(512_000_000_000)] * 8 + [str(1_024_000_000_000)])
            rendered = json.dumps(row).replace(
                '"avg_ts": 1.0', '"avg_ts": 1.000000') + "\n"
            process = self._Process(rendered)
            identity = {
                "schema": "epyc.autokernel.split_reward_runtime_maps.v1",
                "runtime_manifest_sha256": "d" * 64, "arm": "anchor",
                "model_path": str(model.resolve()), "model_sha256": gpu.sha256_file(model),
                "device_id": gpu.DEVICE_ID, "identity_sha256": "e" * 64,
            }
            events = []
            def factory(argv, **kwargs):
                self.assertIn("--autokernel-ready-file", argv)
                self.assertIn("--autokernel-continue-file", argv)
                self.assertIn("--autokernel-ready-token", argv)
                self.assertEqual(kwargs["env"]["AMD_SERIALIZE_KERNEL"], "3")
                self.assertEqual(kwargs["env"]["AMD_SERIALIZE_COPY"], "3")
                self.assertEqual(kwargs["env"]["GGML_CUDA_DISABLE_GRAPHS"], "1")
                handshake.ready_path.write_text(
                    f"{handshake.schema} {process.pid} 8613 9 {handshake.token}\n",
                    encoding="ascii")
                handshake.ready_path.chmod(0o600)
                return process
            def release(witness):
                self.assertFalse(handshake.continue_path.exists())
                self.assertEqual(witness["ready"]["pid"], process.pid)
                self.assertEqual(witness["ready"]["seed"], 8613)
                self.assertEqual(witness["ready"]["repetitions"], 9)
                events.append("lock-released")
            with mock.patch.object(gpu, "_runtime_maps_identity", return_value=identity):
                result = gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True, expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=9, runtime_arm="anchor", readiness_policy=policy,
                    timed_output_oracle=True,
                    ready_continue_handshake=handshake, on_load_ready=release,
                    common_loader_dir=build / "bin", hip_library_dir=build / "bin",
                    process_factory=factory,
                    kfd_pid_provider=lambda: () if process.waited else (123,),
                    vram_reader=lambda: 64, pgid_provider=lambda _pid: process.pid,
                    sleep=lambda _: None)
            self.assertEqual(events, ["lock-released"])
            self.assertEqual(handshake.continue_path.read_text(encoding="ascii"),
                             handshake.token + "\n")
            self.assertEqual(result["load_readiness_witness"]["ready"]["token"], handshake.token)
            self.assertAlmostEqual(result["metric"], 0.9)
            self.assertEqual(result["samples"], [1.0] * 8 + [0.5])
            self.assertEqual(result["metric_contract"], {
                "schema": "epyc.autokernel.serialized_pair_max_metric.v1",
                "scope": "integrity_discovery_only",
                "production_throughput_authority": False,
                "graph_mode": "disabled_for_integrity",
                "scored_sample": "min(first_tokens_per_s,second_tokens_per_s)",
                "serialization_env": {
                    "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                    "GGML_CUDA_DISABLE_GRAPHS": "1"},
            })
            self.assertEqual(handshake.cleanup(), {"ready_removed": True, "continue_removed": True})

    def test_tampered_ready_receipt_terminates_child_before_any_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            policy = _readiness(model)
            handshake = gpu.ReadyContinueHandshake.create(
                root=root / "barrier", decision=_admission(model), policy=policy,
                arm="anchor", seed=8613, repetitions=9)
            process = self._Process(self._row([1.0] * 9), running_polls=100)
            identity = {
                "schema": "epyc.autokernel.split_reward_runtime_maps.v1",
                "runtime_manifest_sha256": "d" * 64, "arm": "anchor",
                "model_path": str(model.resolve()), "model_sha256": gpu.sha256_file(model),
                "device_id": gpu.DEVICE_ID, "identity_sha256": "e" * 64,
            }
            def factory(*_args, **_kwargs):
                handshake.ready_path.write_text(
                    f"{handshake.schema} 0 8613 9 {handshake.token}\n", encoding="ascii")
                handshake.ready_path.chmod(0o600)
                return process
            with mock.patch.object(gpu, "_runtime_maps_identity", return_value=identity), \
                 self.assertRaisesRegex(RuntimeError, "PID/seed/repetitions/token"):
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True, expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=9, runtime_arm="anchor", readiness_policy=policy,
                    ready_continue_handshake=handshake, on_load_ready=lambda _w: None,
                    common_loader_dir=build / "bin", hip_library_dir=build / "bin",
                    process_factory=factory,
                    kfd_pid_provider=lambda: () if process.waited else (123,),
                    vram_reader=lambda: 64, pgid_provider=lambda _pid: process.pid,
                    sleep=lambda _: None)
            self.assertIsNotNone(process.returncode)
            self.assertFalse(handshake.continue_path.exists())
            handshake.cleanup()

    def test_foreign_kfd_during_handshake_terminates_child_without_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            policy = _readiness(model)
            handshake = gpu.ReadyContinueHandshake.create(
                root=root / "barrier", decision=_admission(model), policy=policy,
                arm="anchor", seed=8613, repetitions=9)
            process = self._Process(self._row([1.0] * 9), running_polls=100)
            with self.assertRaisesRegex(RuntimeError, "foreign KFD inference"):
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True, expected_source_commit=gpu.SOURCE_COMMIT,
                    repetitions=9, runtime_arm="anchor", readiness_policy=policy,
                    ready_continue_handshake=handshake, on_load_ready=lambda _w: None,
                    common_loader_dir=build / "bin", hip_library_dir=build / "bin",
                    process_factory=lambda *_args, **_kwargs: process,
                    kfd_pid_provider=lambda: (() if process.waited else (123, 456)),
                    vram_reader=lambda: 64,
                    pgid_provider=lambda pid: process.pid if pid == 123 else 777,
                    sleep=lambda _: None)
            self.assertTrue(process.terminated)
            self.assertTrue(process.waited)
            self.assertFalse(handshake.continue_path.exists())
            handshake.cleanup()

    def test_supervisor_deadline_terminates_and_proves_child_dead(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(Path(directory), rocwmma="ON", mfma="OFF")
            model = Path(directory) / "model.gguf"; model.write_bytes(b"model")
            process = self._Process(self._row([1.0] * 9), running_polls=100)
            with mock.patch.object(gpu.time, "monotonic", side_effect=(0.0, 2.0)), \
                 self.assertRaisesRegex(RuntimeError, "supervisor deadline exceeded"):
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True, repetitions=9, max_runtime_s=1,
                    process_factory=lambda *_args, **_kwargs: process,
                    kfd_pid_provider=lambda: (), vram_reader=lambda: 0,
                    pgid_provider=lambda _pid: process.pid, sleep=lambda _: None)
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)
        self.assertIsNotNone(process.returncode)

    def test_popen_failure_cleans_private_output_carriers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = _build(root, rocwmma="ON", mfma="OFF")
            model = root / "model.gguf"; model.write_bytes(b"model")
            supervisor = root / "supervisor"
            with self.assertRaisesRegex(OSError, "spawn refused"):
                gpu._invoke_locked(
                    build=build, model=model, seed=8613, baseline_vram=0,
                    flash_attention=True, repetitions=9,
                    supervisor_root=supervisor,
                    process_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        OSError("spawn refused")))
            self.assertEqual(list(supervisor.iterdir()), [])


class TestGpuDiscoveryRunCleanup(unittest.TestCase):
    def test_cross_arm_output_oracles_use_one_seed_in_both_arm_orders(self) -> None:
        for arm_order in (("anchor", "candidate"), ("candidate", "anchor")):
            with self.subTest(arm_order=arm_order, oracle="graphs-on"):
                seeds = {
                    arm: gpu._invocation_seed(
                        base_seed=8613, repetitions=9, arm=arm,
                        timed_output_oracle_enabled=False,
                        runtime_graphs="on")
                    for arm in arm_order
                }
                self.assertEqual(seeds, {"anchor": 8613, "candidate": 8613})
            with self.subTest(arm_order=arm_order, oracle="timed-output"):
                seeds = {
                    arm: gpu._invocation_seed(
                        base_seed=8613, repetitions=9, arm=arm,
                        timed_output_oracle_enabled=True,
                        runtime_graphs="off")
                    for arm in arm_order
                }
                self.assertEqual(seeds, {"anchor": 8613, "candidate": 8613})

        self.assertEqual(gpu._invocation_seed(
            base_seed=8613, repetitions=9, arm="candidate",
            timed_output_oracle_enabled=False, runtime_graphs="off"), 8622)

    def test_sampler_stop_failure_never_prevents_device_claim_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"; model.write_bytes(b"model")
            vram = root / "vram"; vram.write_text("0", encoding="ascii")
            out = root / "out"
            build = root / "build"; build.mkdir()
            anchor_identity = {"source_commit": "a" * 40}
            candidate_identity = {"source_commit": "b" * 40}
            sealed = {
                "model": str(model), "model_sha256": gpu.sha256_file(model),
                "model_size_bytes": model.stat().st_size,
                "anchor_build": str(build), "candidate_build": str(build),
                "anchor_identity": anchor_identity, "candidate_identity": candidate_identity,
                "sole_factor": {"name": "source_patch", "anchor": "a", "candidate": "b"},
                "serialized_readiness": {"ready_continue": {"enabled": False}},
                "host_transfer": {"mode": "cold_serialized"},
                "cpu_overlap_policy": "cold_serialized_load_window",
                "runtime_arms": None, "inference_window_lock": str(root / "lock"),
                "anchor_flash_attention": True, "candidate_flash_attention": True,
                "prompt_tokens": 0, "generation_tokens": 128,
                "anchor_threads": 8, "candidate_threads": 8,
                "anchor_ubatch": 512, "candidate_ubatch": 512,
                "anchor_batch": 512, "candidate_batch": 512,
                "anchor_mmap": True, "candidate_mmap": True,
                "anchor_no_op_offload": False, "candidate_no_op_offload": False,
                "anchor_split_mode": "layer", "candidate_split_mode": "layer",
                "anchor_no_kv_offload": False, "candidate_no_kv_offload": False,
                "anchor_poll": 50, "candidate_poll": 50,
                "frame": "tg128-ngl99", "metric": "decode_tokens_per_s",
                "arm_order_schedule": ["candidate", "anchor"],
                "arm_order_seed_sha256": "d" * 64,
                "operation_key": "9" * 64,
            }
            claim = mock.Mock()
            claim.borrowed_outer_reservation = True
            claim.receipt.return_value.to_dict.return_value = {"claim": "open"}
            claim.release.return_value = {
                "schema": "epyc.autokernel.borrowed_device_claim_phase.v1",
                "mode": "borrowed_outer_reservation", "outer_claim_id": "akd-outer",
                "device_id": "mi210_0", "campaign_id": "cleanup-test",
                "phase_ended_at": "done", "physical_release": False}
            borrow_calls = []
            def borrowed_acquirer(*args, **kwargs):
                borrow_calls.append((args, kwargs)); return claim
            sampler = mock.Mock()
            sampler.start.return_value = sampler
            sampler.stop.side_effect = RuntimeError("sampler stop failed")
            args = argparse.Namespace(
                output_dir=str(out), cpu_claim_journal=str(root / "cpu.jsonl"),
                device_claim_journal=str(root / "gpu.jsonl"),
                campaign_id="cleanup-test", seed=8613, calls=9,
                _device_claim_acquirer=borrowed_acquirer)
            call_order = []
            def ordered_invocation(**kwargs):
                call_order.append(kwargs["expected_source_commit"])
                return invocation
            with mock.patch.object(gpu, "preflight", return_value=sealed), \
                 mock.patch.object(gpu.storage, "assert_not_scratch", return_value=out), \
                 mock.patch.object(gpu, "_readiness_policy_for_arm", return_value=None), \
                 mock.patch.object(gpu, "_kfd_pids", return_value=()), \
                 mock.patch.object(gpu, "VRAM_USED", vram), \
                 mock.patch.object(gpu.device_claim, "acquire_device_claim",
                                   side_effect=AssertionError("nested physical claim")), \
                 mock.patch.object(gpu.device_sampler, "RocmSmiSampler", return_value=sampler), \
                 mock.patch.object(gpu, "invoke", side_effect=RuntimeError("primary failure")), \
                 self.assertRaisesRegex(RuntimeError, "primary failure"):
                gpu.run(args)
            claim.release.assert_called_once_with()
            self.assertEqual(len(borrow_calls), 1)
            sampler.stop.assert_called_once_with()

            # Success seals the logical throughput phase while the outer
            # reservation remains physical release authority.
            claim.reset_mock()
            claim.borrowed_outer_reservation = True
            claim.receipt.return_value.to_dict.return_value = gpu.device_claim.ClaimReceipt(
                claim_id="akd-outer", device_id="mi210_0", lock_path="/claim",
                state="held", holder_pid=1, holder_start_ticks=1,
                holder_boot_id="boot", host="host", holder_label="test",
                purpose="outer", campaign_id="cleanup-test", acquired_at="now").to_dict()
            claim.release.return_value = {
                "schema": "epyc.autokernel.borrowed_device_claim_phase.v1",
                "mode": "borrowed_outer_reservation", "outer_claim_id": "akd-outer",
                "device_id": "mi210_0", "campaign_id": "cleanup-test",
                "phase_ended_at": "done", "physical_release": False}
            sampler.reset_mock(); sampler.start.return_value = sampler
            sampler.stop.side_effect = None
            sampler.stop.return_value.to_dict.return_value = {"samples": []}
            args.output_dir = str(root / "success")
            invocation = {"samples": [100.0] * 9, "sample_count": 9,
                          "hip_residency_proved": True,
                          "cpu_coverage": {"covered": True}}
            with mock.patch.object(gpu, "preflight", return_value=sealed), \
                 mock.patch.object(gpu.storage, "assert_not_scratch",
                                   return_value=root / "success"), \
                 mock.patch.object(gpu, "_readiness_policy_for_arm", return_value=None), \
                 mock.patch.object(gpu, "_kfd_pids", return_value=()), \
                 mock.patch.object(gpu, "VRAM_USED", vram), \
                 mock.patch.object(gpu.device_claim, "acquire_device_claim",
                                   side_effect=AssertionError("nested physical claim")), \
                 mock.patch.object(gpu.device_sampler, "RocmSmiSampler", return_value=sampler), \
                 mock.patch.object(gpu, "invoke", side_effect=ordered_invocation), \
                 mock.patch.object(gpu.autokernel_progression, "export_progression"):
                result = gpu.run(args)
            self.assertEqual(call_order, ["b" * 40, "a" * 40])
            self.assertEqual(result["arm_order_schedule"], ["candidate", "anchor"])
            self.assertEqual(result["arm_order_seed_sha256"], "d" * 64)
            bank = json.loads((root / "success/baseline-bank.json").read_text())
            self.assertEqual(bank["arm_order_schedule"], ["candidate", "anchor"])
            self.assertEqual(bank["arm_order_seed_sha256"], "d" * 64)
            claim.release.assert_called_once_with()
            self.assertEqual(result["device_claim_open"]["claim_id"], "akd-outer")
            self.assertEqual(result["device_claim_borrowed_phase_end"]["outer_claim_id"],
                             "akd-outer")
            self.assertFalse(result["device_claim_borrowed_phase_end"]["physical_release"])
            self.assertNotIn("device_claim_released", result)
            governance = json.loads((root / "success/live-governance.json").read_text())
            self.assertEqual(governance["status"], "borrowed_phase_ended")
            self.assertNotIn("device_claim_released", governance)

            # v24: both hardened arms complete on the same nine inputs, but
            # every candidate output differs.  This must seal a scientific
            # correctness terminal and still end the borrowed claim exactly
            # once; no result may make the mismatched throughput admissible.
            sealed["runtime_graphs"] = "off"
            sealed["timed_output_oracle"] = {"enabled": True}
            args.output_dir = str(root / "v24-divergence")
            claim.reset_mock()
            claim.borrowed_outer_reservation = True
            claim.receipt.return_value.to_dict.return_value = gpu.device_claim.ClaimReceipt(
                claim_id="akd-outer", device_id="mi210_0", lock_path="/claim",
                state="held", holder_pid=1, holder_start_ticks=1,
                holder_boot_id="boot", host="host", holder_label="test",
                purpose="outer", campaign_id="cleanup-test", acquired_at="now").to_dict()
            claim.release.return_value = {
                "schema": "epyc.autokernel.borrowed_device_claim_phase.v1",
                "mode": "borrowed_outer_reservation", "outer_claim_id": "akd-outer",
                "device_id": "mi210_0", "campaign_id": "cleanup-test",
                "phase_ended_at": "done", "physical_release": False}
            sampler.reset_mock(); sampler.start.return_value = sampler
            sampler.stop.side_effect = None
            sampler.stop.return_value.to_dict.return_value = {"samples": []}
            inputs = [f"{index + 1:016x}" for index in range(9)]

            def divergent_invocation(**kwargs):
                arm = ("anchor" if kwargs["expected_source_commit"] == "a" * 40
                       else "candidate")
                outputs = [
                    f"{'a' if arm == 'anchor' else 'b'}{index:015x}"
                    for index in range(9)]
                body = {
                    "schema": "epyc.autokernel.timed_output_semantics.v1",
                    "instrument_commit": gpu.READY_CONTINUE_INSTRUMENT_COMMIT,
                    "seed": kwargs["seed"], "repetitions": 9,
                    "tokens_per_repetition": 128,
                    "input_hashes": inputs, "output_hashes": outputs,
                    "within_pair_bitwise_equal": True,
                    "ranked_member_device_sync": "hip_full_device",
                    "serialization_env": {
                        "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                        "GGML_CUDA_DISABLE_GRAPHS": "1"},
                    "first_samples_ns": [1_000_000] * 9,
                    "second_samples_ns": [1_000_000] * 9,
                    "protected_samples_ns": [1_000_000] * 9,
                    "protected_samples_ts": [128_000.0] * 9,
                    "anti_shift_witness": "hip_serialized_pair_max",
                    "reward_admissible": True,
                }
                process_root = kwargs["process_receipt_root"]
                process_root.mkdir(mode=0o700)
                receipt_path = process_root / "receipt.json"
                receipt_path.write_text(json.dumps({
                    "identity": {"process_context": kwargs["process_context"]}}),
                    encoding="utf-8")
                receipt_path.chmod(0o600)
                return {
                    **invocation,
                    "timed_output_semantics": {
                        **body, "receipt_sha256": schemas.content_hash(body)},
                    "supervisor": {
                        "process_receipt_path": str(receipt_path.resolve()),
                        "process_receipt_file_sha256": gpu.sha256_file(receipt_path),
                    },
                }

            with mock.patch.object(gpu, "preflight", return_value=sealed), \
                 mock.patch.object(gpu.storage, "assert_not_scratch",
                                   return_value=root / "v24-divergence"), \
                 mock.patch.object(gpu, "_readiness_policy_for_arm", return_value=None), \
                 mock.patch.object(gpu, "_kfd_pids", return_value=()), \
                 mock.patch.object(gpu, "VRAM_USED", vram), \
                 mock.patch.object(gpu.device_claim, "acquire_device_claim",
                                   side_effect=AssertionError("nested physical claim")), \
                 mock.patch.object(gpu.device_sampler, "RocmSmiSampler",
                                   return_value=sampler), \
                 mock.patch.object(gpu, "invoke", side_effect=divergent_invocation), \
                 self.assertRaises(gpu.CandidateCorrectnessDivergence):
                gpu.run(args)
            claim.release.assert_called_once_with()
            sampler.stop.assert_called_once_with()
            self.assertTrue((root / "v24-divergence/correctness-divergence.json").is_file())
            self.assertFalse((root / "v24-divergence/result.json").exists())

            # The graphs-on run owns a separate cross-arm output oracle.  It
            # must pass the identical hidden seed to both arms regardless of
            # the counterbalanced execution order; otherwise the reducer sees
            # different input/output content banks, as live v18 did.
            sealed["runtime_graphs"] = "on"
            sealed["timed_output_oracle"] = {"enabled": False}
            args.output_dir = str(root / "graphs-on-success")
            claim.reset_mock()
            claim.borrowed_outer_reservation = True
            claim.receipt.return_value.to_dict.return_value = gpu.device_claim.ClaimReceipt(
                claim_id="akd-outer", device_id="mi210_0", lock_path="/claim",
                state="held", holder_pid=1, holder_start_ticks=1,
                holder_boot_id="boot", host="host", holder_label="test",
                purpose="outer", campaign_id="cleanup-test", acquired_at="now").to_dict()
            claim.release.return_value = {
                "schema": "epyc.autokernel.borrowed_device_claim_phase.v1",
                "mode": "borrowed_outer_reservation", "outer_claim_id": "akd-outer",
                "device_id": "mi210_0", "campaign_id": "cleanup-test",
                "phase_ended_at": "done", "physical_release": False}
            sampler.reset_mock(); sampler.start.return_value = sampler
            sampler.stop.side_effect = None
            sampler.stop.return_value.to_dict.return_value = {"samples": []}
            graph_seeds = []
            hashes = [f"{index:064x}" for index in range(9)]

            def graphs_on_invocation(**kwargs):
                arm = ("anchor" if kwargs["expected_source_commit"] == "a" * 40
                       else "candidate")
                graph_seeds.append((arm, kwargs["seed"]))
                body = {
                    "schema": "epyc.autokernel.graphs_on_output_integrity.v1",
                    "instrument_commit": gpu.READY_CONTINUE_INSTRUMENT_COMMIT,
                    "seed": kwargs["seed"], "repetitions": 9,
                    "input_hashes": hashes, "output_hashes": hashes,
                    "graph_environment": {"GGML_CUDA_DISABLE_GRAPHS": None},
                    "reward_admissible": True,
                }
                return {**invocation, "graphs_on_output_integrity": {
                    **body, "receipt_sha256": gpu.schemas.content_hash(body)}}

            with mock.patch.object(gpu, "preflight", return_value=sealed), \
                 mock.patch.object(gpu.storage, "assert_not_scratch",
                                   return_value=root / "graphs-on-success"), \
                 mock.patch.object(gpu, "_readiness_policy_for_arm", return_value=None), \
                 mock.patch.object(gpu, "_kfd_pids", return_value=()), \
                 mock.patch.object(gpu, "VRAM_USED", vram), \
                 mock.patch.object(gpu.device_claim, "acquire_device_claim",
                                   side_effect=AssertionError("nested physical claim")), \
                 mock.patch.object(gpu.device_sampler, "RocmSmiSampler", return_value=sampler), \
                 mock.patch.object(gpu, "invoke", side_effect=graphs_on_invocation), \
                 mock.patch.object(gpu.autokernel_progression, "export_progression"):
                graph_result = gpu.run(args)
            self.assertEqual(graph_seeds, [("candidate", 8613), ("anchor", 8613)])
            self.assertEqual(graph_result["graphs_on_output_oracle"]["seed"], 8613)
            self.assertTrue(
                graph_result["graphs_on_output_oracle"]["cross_arm_bitwise_equal"])
