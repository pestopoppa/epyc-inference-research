from pathlib import Path
import argparse
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


class TestGpuDiscoveryBuildIdentity(unittest.TestCase):
    def test_source_patch_accepts_shared_reward_binary_and_distinct_hip_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = _build(root / "anchor", rocwmma="ON", mfma="OFF")
            candidate = _build(root / "candidate", rocwmma="ON", mfma="OFF")
            # Candidate source identity differs, but one anchor-built benchmark
            # executable is used for both arms; only HIP loading may differ.
            def git_commit(argv, **_kwargs):
                return mock.Mock(stdout=("a" * 40 if str(anchor) in argv else "b" * 40), returncode=0)
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
                sealed = gpu.preflight(args)
            self.assertEqual(sealed["runtime_arms"]["measurement_binary_sha256"],
                             gpu.sha256_file(anchor / "bin" / "llama-bench"))
            self.assertNotEqual(sealed["runtime_arms"]["anchor_hip_sha256"],
                                sealed["runtime_arms"]["candidate_hip_sha256"])

    def test_preflight_records_cold_serialization_for_over_budget_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build = _build(root, rocwmma="ON", mfma="OFF"); model = root / "m"; model.write_bytes(b"x")
            sealed = gpu.preflight(_bind_admission(argparse.Namespace(model=str(model), anchor_build=str(build), candidate_build=str(build),
                    factor="flash_attention", campaign_id="gpu", calls=3, workload="prefill_pp512",
                    device_id=gpu.DEVICE_ID, inference_window_lock=None)))
            self.assertEqual(sealed["host_transfer"]["mode"], "cold_serialized")
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

        class Held:
            def __enter__(self):
                events.append("window-open")
                return Lease()

            def __exit__(self, *_):
                events.append("window-close")

        class CallWindow:
            def hold(self):
                return Held()

        class Receipt:
            def to_dict(self):
                return {"claim_id": "cpu-owned"}

        class Claim:
            def receipt(self):
                return Receipt()

            def release(self):
                events.append("cpu-release")

        def measured(**_kwargs):
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
                sealed_load_decision=_admission(model))
        self.assertEqual(events, ["window-open", "model-call", "cpu-release",
                                  "window-close"])
        self.assertFalse(result["cpu_coverage"]["borrowed"])

    def test_contended_cpu_claim_borrows_only_inside_model_window(self) -> None:
        events = []

        class Lease:
            path = Path("/tmp/window")
            waited_s = 0.0

        class Held:
            def __enter__(self):
                events.append("window-open")
                return Lease()

            def __exit__(self, *_):
                events.append("window-close")

        class CallWindow:
            def hold(self):
                return Held()

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

        def measured(**_kwargs):
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
                sealed_load_decision=_admission(model))
        self.assertEqual(events, ["window-open", "borrow-open", "model-call",
                                  "borrow-validate", "window-close"])
        self.assertTrue(result["cpu_coverage"]["borrowed"])
