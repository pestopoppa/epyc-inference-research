from pathlib import Path
import argparse
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
        return __import__("json").dumps({
            "backends": "ROCm", "gpu_info": "AMD Instinct MI210",
            "build_commit": "0db32c0", "n_prompt": 512, "n_gen": 0,
            "flash_attn": 1, "n_threads": 8, "n_batch": 512, "n_ubatch": 512,
            "use_mmap": True, "no_op_offload": 0, "split_mode": "layer",
            "no_kv_offload": False, "poll": 50, "avg_ts": sum(samples) / len(samples),
            "samples_ts": samples,
        }) + "\n"

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
        self.assertEqual(result["samples"], [100.0 + i for i in range(9)])
        self.assertEqual(result["sample_count"], 9)
        self.assertEqual(result["metric"], 104.0)
        self.assertEqual(result["raw_row"]["samples_ts"], result["samples"])

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
            with self.assertRaisesRegex(RuntimeError, "exactly 9 finite raw samples"):
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
            process = self._Process(self._row([1.0] * 9))
            identity = {
                "schema": "epyc.autokernel.split_reward_runtime_maps.v1",
                "runtime_manifest_sha256": "d" * 64, "arm": "anchor",
                "model_path": str(model.resolve()), "model_sha256": gpu.sha256_file(model),
                "device_id": gpu.DEVICE_ID, "identity_sha256": "e" * 64,
            }
            events = []
            def factory(argv, **_kwargs):
                self.assertIn("--autokernel-ready-file", argv)
                self.assertIn("--autokernel-continue-file", argv)
                self.assertIn("--autokernel-ready-token", argv)
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
    def test_sampler_stop_failure_never_prevents_device_claim_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"; model.write_bytes(b"model")
            vram = root / "vram"; vram.write_text("0", encoding="ascii")
            out = root / "out"
            build = root / "build"; build.mkdir()
            identity = {"source_commit": "a" * 40}
            sealed = {
                "model": str(model), "model_sha256": gpu.sha256_file(model),
                "model_size_bytes": model.stat().st_size,
                "anchor_build": str(build), "candidate_build": str(build),
                "anchor_identity": identity, "candidate_identity": identity,
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
                 mock.patch.object(gpu, "invoke", return_value=invocation), \
                 mock.patch.object(gpu.autokernel_progression, "export_progression"):
                result = gpu.run(args)
            claim.release.assert_called_once_with()
            self.assertEqual(result["device_claim_open"]["claim_id"], "akd-outer")
            self.assertEqual(result["device_claim_borrowed_phase_end"]["outer_claim_id"],
                             "akd-outer")
            self.assertFalse(result["device_claim_borrowed_phase_end"]["physical_release"])
            self.assertNotIn("device_claim_released", result)
            governance = json.loads((root / "success/live-governance.json").read_text())
            self.assertEqual(governance["status"], "borrowed_phase_ended")
            self.assertNotIn("device_claim_released", governance)
