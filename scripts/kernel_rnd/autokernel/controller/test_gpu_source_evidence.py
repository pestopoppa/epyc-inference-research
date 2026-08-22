import hashlib
import json
from dataclasses import replace
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest import mock

from . import gpu_source_evidence as E
from . import gpu_source_proofs as P


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def identity(letter: str) -> P.BuildIdentity:
    return P.BuildIdentity(
        source_commit=f"commit-{letter}", source_sha256=digest(letter),
        binary_sha256=digest(f"{letter}-binary"),
        hip_library_sha256=digest(f"{letter}-hip"),
        config_sha256=digest(f"{letter}-config"),
        linkage_sha256=digest(f"{letter}-linkage"))


def write_bound(root: Path, name: str, content: bytes, role: str) -> E.BoundInputFile:
    path = (root / name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return E.BoundInputFile(role, path, hashlib.sha256(content).hexdigest())


def build_files(root: Path, label: str) -> tuple[P.BuildIdentity, E.BuildIdentityFiles]:
    entries = [["100644", digest(f"{label}-tree-entry"), "ggml/src/ggml-cuda/fattn.cu"]]
    tree_sha = hashlib.sha256(
        "".join(f"{mode}\t{entry_sha}\t{path}\n" for mode, entry_sha, path in entries)
        .encode("utf-8")).hexdigest()
    source_body = {
        "schema": E.SOURCE_TREE_SCHEMA,
        "source_commit": f"commit-{label}",
        "root_provenance": str((root / "snapshot").resolve()),
        "exclusions": [".git"],
        "tree": {"sha256": tree_sha, "file_count": len(entries), "total_bytes": 1,
                 "entries": entries, "listing_is_complete": True},
    }
    source_body["receipt_sha256"] = E.schemas.content_hash(source_body)
    source = write_bound(
        root, f"{label}-source.json",
        json.dumps(source_body, sort_keys=True).encode(),
        "source_identity")
    binary = write_bound(root, f"{label}-binary", f"binary-{label}".encode(), "binary")
    hip = write_bound(root, f"{label}-hip.so", f"hip-{label}".encode(), "hip_library")
    config = write_bound(root, f"{label}-config.json", f"config-{label}".encode(), "config")
    linkage = write_bound(root, f"{label}-linkage.json", f"linkage-{label}".encode(), "linkage")
    return (P.BuildIdentity(
        source_commit=f"commit-{label}", source_sha256=tree_sha,
        binary_sha256=binary.sha256, hip_library_sha256=hip.sha256,
        config_sha256=config.sha256, linkage_sha256=linkage.sha256),
        E.BuildIdentityFiles(source, binary, hip, config, linkage))


def plan(root: Path, *, shared_reward: bool = False) -> E.GpuSourceEvidencePlan:
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    candidate, candidate_files = build_files(root / "candidate", "candidate")
    anchor, anchor_files = build_files(root / "anchor", "anchor")
    manifest = write_bound(root, "manifest.json", b"sealed manifest", "manifest")
    model = write_bound(root, "model.gguf", b"model bytes", "model")
    workload = write_bound(root, "workload.json", b"workload bytes", "workload")
    runtime = write_bound(root, "runtime.json", b"runtime bytes", "runtime_config")
    materialization_body = {
        "schema": "epyc.autokernel.gpu_source_materialization.v1",
        "manifest_sha256": manifest.sha256,
        "candidate_source_commit": candidate.source_commit,
        "candidate_source_sha256": candidate.source_sha256,
        "patch_applied": True,
        "production_tree": False,
    }
    materialization_body["receipt_sha256"] = E.schemas.content_hash(materialization_body)
    materialization = write_bound(
        root, "materialization.json",
        json.dumps(materialization_body, sort_keys=True).encode(), "materialization")
    correctness_tool = write_bound(root, "test-backend-ops", b"correctness tool", "executable")
    profiler = write_bound(root, "rocprof", b"profiler tool", "executable")
    correctness_tool.path.chmod(0o700)
    profiler.path.chmod(0o700)
    timestamp_input = write_bound(root, "timestamps.xml", b"timestamp policy", "timestamp_input")
    placeholder_policy = E.BoundInputFile("execution_policy", (root / "policy.json").resolve(), "0" * 64)
    shared = None
    if shared_reward:
        common = (root / "shared-runtime" / "common").resolve()
        anchor_overlay = (root / "shared-runtime" / "anchor-hip").resolve()
        candidate_overlay = (root / "shared-runtime" / "candidate-hip").resolve()
        for directory in (common, anchor_overlay, candidate_overlay):
            directory.mkdir(parents=True, exist_ok=True)
        reward = write_bound(common, "llama-bench", b"shared reward", "reward_binary")
        reward.path.chmod(0o700)
        anchor_hip = write_bound(anchor_overlay, "libggml-hip.so.0", b"hip-anchor", "runtime_hip")
        candidate_hip = write_bound(candidate_overlay, "libggml-hip.so.0", b"hip-candidate", "runtime_hip")
        # The diagnostic correctness binary remains in the complete candidate
        # build closure.  Its original HIP DSO has the same bytes as its arm
        # overlay; the overlay is only for the shared reward executable.
        assert candidate.hip_library_sha256 == candidate_hip.sha256
        assert anchor.hip_library_sha256 == anchor_hip.sha256
        runtime_body = {
            "schema": "epyc.autokernel.shared_reward_runtime.v1",
            "authority": E.AUTHORITY, "promotion_claim": False,
            "measurement_binary_sha256": reward.sha256,
            "anchor_hip_sha256": anchor_hip.sha256,
            "candidate_hip_sha256": candidate_hip.sha256,
            "split_runtime_manifest": {
                "schema": "epyc.autokernel.split_reward_runtime.v1",
                "root": str((root / "shared-runtime").resolve()),
                "manifest_sha256": digest("shared-runtime-manifest"),
            },
        }
        runtime_body["receipt_sha256"] = E.schemas.content_hash(runtime_body)
        runtime_receipt = write_bound(root / "shared-runtime", "reward-runtime.json",
                                      json.dumps(runtime_body, sort_keys=True).encode(), "runtime_receipt")
        shared = E.SharedRewardRuntimeFiles(
            measurement_binary=reward, runtime_receipt=runtime_receipt,
            anchor_hip_library=anchor_hip, candidate_hip_library=candidate_hip)
    c6_tool = write_bound(root, "c6-harness", b"c6 tool", "executable")
    c6_tool.path.chmod(0o700)
    c6_source = write_bound(root, "c6-harness.cpp", b"source", "c6_source")
    c6_capability = write_bound(root, "c6-capability.json", b"capability", "c6_capability")
    c6 = E.C6CorrectnessPlan(
        argv=(str(c6_tool.path), "--mode", "oracle", "--operation", "MUL_MAT",
              "--backend", "ROCm0", "--type-a", "q4_K",
              "--m", "32", "--n", "1", "--k", "256", "--seed", "42",
              "--sidecar", E.C6_SIDECAR_OUTPUT),
        inputs=(c6_tool, c6_source, c6_capability),
        precision_contract={
            "required_output_dtype": "f32", "required_accumulator_dtype": "f32",
            "atol": .01, "rtol": .01, "required_matched_ratio": .95,
            "lowbit": True},
        precision_equivalence_policy={
            "operator_id": "MUL_MAT", "template_id": "fixture-q4k",
            "input_dtype": "float32", "required_output_dtype": "f32",
            "required_accumulator_dtype": "f32", "reduce_dimension": 256,
            "structural_evidence_sha256": digest("c6-structural"),
            "bound_multiplier": 1.0},
        structural_precision_evidence={
            "output_dtype": "f32", "accumulator_dtype": "f32",
            "evidence_sha256": digest("c6-structural")},
        semantic_judge_verdicts={})
    result = E.GpuSourceEvidencePlan(
        campaign_id="ak-gpu-source-evidence-test",
        device_id="mi210_0",
        manifest_sha256=manifest.sha256, model_sha256=model.sha256,
        workload_sha256=workload.sha256, runtime_config_sha256=runtime.sha256,
        candidate=candidate, anchor=anchor,
        correctness_argv=(str(correctness_tool.path), "--op", "MUL_MAT",
                          "--binary", str(candidate_files.binary.path),
                          "--model", str(model.path), "--workload", str(workload.path),
                          "--config", str(runtime.path)),
        correctness_backend="ROCm0",
        correctness_op="MUL_MAT",
        expected_correctness_cases=3,
        candidate_rocprof_argv=(str(profiler.path), "-i", str(timestamp_input.path),
                                "--candidate", str(shared.measurement_binary.path if shared else candidate_files.binary.path),
                                "--model", str(model.path), "--workload", str(workload.path),
                                "--config", str(runtime.path)),
        anchor_rocprof_argv=(str(profiler.path), "-i", str(timestamp_input.path),
                             "--candidate", str(shared.measurement_binary.path if shared else anchor_files.binary.path),
                             "--model", str(model.path), "--workload", str(workload.path),
                             "--config", str(runtime.path)),
        dispatch=E.DispatchContract(
            candidate_exact=(E.ExactDispatch(
                "new_kernel", r"^new_kernel$", 2, 128, 64, 0, 2),),
            anchor_exact=(E.ExactDispatch(
                "old_kernel", r"^old_kernel$", 2, 256, 64, 512, 4),),
            candidate_forbidden=(E.ForbiddenDispatch(
                "candidate_has_no_old", r"^old_kernel$"),),
            anchor_forbidden=(E.ForbiddenDispatch(
                "anchor_has_no_new", r"^new_kernel$"),),
            invariants=(E.InvariantDispatch("hot_invariant", r"^hot_kernel$"),),
        ),
        identity_files=E.EvidenceIdentityFiles(
            candidate_files, anchor_files, manifest, model, workload, runtime,
            materialization, shared),
        policy=placeholder_policy,
        correctness_inputs=(correctness_tool, candidate_files.binary),
        candidate_rocprof_inputs=(profiler, timestamp_input,
                                  shared.measurement_binary if shared else candidate_files.binary),
        anchor_rocprof_inputs=(profiler, timestamp_input,
                               shared.measurement_binary if shared else anchor_files.binary),
        required_correctness_argv_paths=(candidate_files.binary.path, model.path,
                                         workload.path, runtime.path),
        required_candidate_rocprof_argv_paths=((shared.measurement_binary.path if shared else candidate_files.binary.path), model.path,
                                               workload.path, runtime.path),
        required_anchor_rocprof_argv_paths=((shared.measurement_binary.path if shared else anchor_files.binary.path), model.path,
                                            workload.path, runtime.path),
        execution_cwd=root,
        correctness_environment=(("HIP_VISIBLE_DEVICES", "0"),
                                 ("LD_LIBRARY_PATH", str(candidate_files.hip_library.path.parent))),
        candidate_rocprof_environment=(("HIP_VISIBLE_DEVICES", "0"),
                                       ("LD_LIBRARY_PATH", (str(shared.candidate_hip_library.path.parent) + ":" + str(shared.measurement_binary.path.parent)) if shared else str(candidate_files.hip_library.path.parent))),
        anchor_rocprof_environment=(("HIP_VISIBLE_DEVICES", "0"),
                                    ("LD_LIBRARY_PATH", (str(shared.anchor_hip_library.path.parent) + ":" + str(shared.measurement_binary.path.parent)) if shared else str(anchor_files.hip_library.path.parent))),
        shared_runtime=shared, c6_correctness=c6)
    policy_path = placeholder_policy.path
    policy_path.write_text(json.dumps(E._policy_payload(result), sort_keys=True))
    policy = E.BoundInputFile(
        "execution_policy", policy_path,
        hashlib.sha256(policy_path.read_bytes()).hexdigest())
    return replace(result, policy=policy)


class FakeClaim:
    def __init__(self, number, device, kwargs, *, missing_release=False):
        self.number = number
        self.missing_release = missing_release
        self.released = False
        self.value = {
            "schema": "epyc.autokernel.device_claim_receipt.v1",
            "claim_id": f"akd-test-{number}", "device_id": device,
            "lock_path": f"/tmp/gpu_device.{device}.lock", "state": "held",
            "holder_pid": 9000 + number, "holder_start_ticks": 100 + number,
            "holder_boot_id": "boot-test", "host": "test-host",
            "holder_label": kwargs["holder_label"], "purpose": kwargs["purpose"],
            "campaign_id": kwargs["campaign_id"],
            "acquired_at": "2026-08-13T00:00:00Z", "expires_at": None,
            "released_at": None, "reclaimed_from": None,
        }

    def receipt(self):
        return dict(self.value)

    def release(self):
        self.released = True
        value = dict(self.value)
        if not self.missing_release:
            value["released_at"] = "2026-08-13T00:01:00Z"
        return value


class ClaimFactory:
    def __init__(self, *, missing_release_at=None):
        self.claims = []
        self.missing_release_at = missing_release_at

    def __call__(self, device, **kwargs):
        number = len(self.claims) + 1
        claim = FakeClaim(
            number, device, kwargs,
            missing_release=number == self.missing_release_at)
        self.claims.append(claim)
        return claim


def csv_text(arm: str, *, inverse_bad=False, invariant_changed=False,
             forbidden=False) -> str:
    header = "Index,KernelName,grd,wgr,lds,BeginNs,EndNs\n"
    rows = []
    kernel = "new_kernel" if arm == "candidate" or inverse_bad else "old_kernel"
    grid, lds = ((128, 0) if kernel == "new_kernel" else (256, 512))
    rows.extend([
        f"0,{kernel},{grid},64,{lds},10,20",
        f"1,{kernel},{grid},64,{lds},21,30",
    ])
    invariant_grid = 128 if invariant_changed and arm == "anchor" else 64
    rows.append(f"2,hot_kernel,{invariant_grid},64,0,31,40")
    if forbidden:
        rows.append("3,old_kernel,256,64,512,41,50")
    return header + "\n".join(rows) + "\n"


def correctness_console(summary: str = "3/3 tests passed", *,
                        backend: str = "ROCm0",
                        op: str = "MUL_MAT") -> str:
    match = re.fullmatch(r"(\d+)/(\d+) tests passed", summary)
    if match is None:
        return summary + "\n"
    passed, total = (int(value) for value in match.groups())
    cases = [
        f"  {op}(case={index}): {'OK' if index < passed else 'FAIL'}"
        for index in range(total)
    ]
    backend_status = "OK" if passed == total else "FAIL"
    backends_passed = 2 if backend_status == "OK" else 1
    overall = "OK" if backend_status == "OK" else "FAIL"
    return "\n".join((
        "Testing 2 devices", "", f"Backend 1/2: {backend}", *cases,
        f"  {passed}/{total} tests passed", f"  Backend {backend}: {backend_status}",
        "Backend 2/2: CPU", "  Skipping",
        f"{backends_passed}/2 backends passed", overall, ""))


class FakeExecutors:
    def __init__(self, *, correctness_exit=0, correctness_summary="3/3 tests passed",
                 non_overlap=False, inverse_bad=False, invariant_changed=False,
                 forbidden=False, rocprof_exit=0, runtime_maps=None,
                 c6_nondeterministic=False, c6_partial=False,
                 c6_wrong_seed=False, c6_stall_ready=False,
                 c6_candidate_exit=0):
        self.calls = []
        self.correctness_exit = correctness_exit
        self.correctness_summary = correctness_summary
        self.non_overlap = non_overlap
        self.inverse_bad = inverse_bad
        self.invariant_changed = invariant_changed
        self.forbidden = forbidden
        self.rocprof_exit = rocprof_exit
        self.runtime_maps = runtime_maps or {}
        self.c6_nondeterministic = c6_nondeterministic
        self.c6_partial = c6_partial
        self.c6_wrong_seed = c6_wrong_seed
        self.c6_stall_ready = c6_stall_ready
        self.c6_candidate_exit = c6_candidate_exit
        self.c6_receipt_seen_before_target = False
        self.c6_operation = "MUL_MAT"
        self.c6_leg_calls = 0

    def _c6_witness(self, option, m, n, k):
        output_elements = (m*n*14 if self.c6_operation == "FLASH_ATTN_EXT"
                           else m*n)
        output = struct.pack(
            f"<{output_elements}f",
            *[float(index) / max(1, output_elements)
              for index in range(output_elements)])
        output_f64 = struct.pack(
            f"<{output_elements}d",
            *[float(index) / max(1, output_elements)
              for index in range(output_elements)])
        if self.c6_operation == "MUL_MAT":
            witness = {"weights_hex": "00" * 128,
                       "activations_f32le_hex": "00" * (k*n*4)}
        elif self.c6_operation == "RMS_NORM":
            witness = {"activations_f32le_hex": "00" * (m*n*4),
                       "scale_f32le_hex": "00" * (m*4)}
        else:
            witness = {"query_f32le_hex": "00" * (m*n*14*4),
                       "key_f16le_hex": "00" * (m*k*2*2),
                       "value_f16le_hex": "00" * (m*k*2*2)}
        return output_elements, output, output_f64, witness

    def correctness(self, invocation):
        self.calls.append((invocation.kind, invocation.arm, invocation.argv,
                           invocation.environment))
        if "--mode" in invocation.argv:
            option = {invocation.argv[index]: invocation.argv[index + 1]
                      for index in range(1, len(invocation.argv), 2)}
            mode = option["--mode"]
            self.c6_operation = option["--operation"]
            m, n, k = (int(option[name]) for name in ("--m", "--n", "--k"))
            output_elements, output, output_f64, witness = self._c6_witness(
                option, m, n, k)
            if mode == "oracle":
                sidecar = Path(option["--sidecar"])
                sidecar.write_text(json.dumps({
                    "schema": E.C6_ORACLE_SIDECAR_SCHEMA,
                    "backend": option["--backend"],
                    "operation": self.c6_operation,
                    "type_a": option["--type-a"], "m": m, "n": n, "k": k,
                    "seed": (int(option["--seed"]) - 1 if self.c6_wrong_seed
                             else int(option["--seed"])),
                    "output_elements": output_elements,
                    "input_witness": witness,
                    "reference_output_f32le_hex": output.hex(),
                    "reference_output_f64le_hex": output_f64.hex(),
                }, sort_keys=True))
                invocation.stdout_path.write_text("native C6 oracle complete\n")
            else:
                self.c6_leg_calls += 1
                leg = self.c6_leg_calls
                if not self.c6_stall_ready:
                    Path(option["--ready-file"]).write_bytes(b"R")
                chosen = output
                if self.c6_nondeterministic and leg == 3:
                    chosen = output[:-1] + bytes([output[-1] ^ 1])
                Path(option["--output"]).write_bytes(chosen)
                invocation.stdout_path.write_text(
                    f"native C6 candidate leg {leg} complete\n")
                exit_code = self.c6_candidate_exit
                if self.c6_partial and leg == 3:
                    exit_code = 1
                invocation.stderr_path.write_text("")
                return self._capture(invocation, exit_code)
            invocation.stderr_path.write_text("")
            return self._capture(invocation, 0)
        if "--sidecar" in invocation.argv:
            sidecar = Path(invocation.argv[invocation.argv.index("--sidecar") + 1])
            option = {invocation.argv[index]: invocation.argv[index + 1]
                      for index in range(1, len(invocation.argv), 2)}
            self.c6_operation = option["--operation"]
            m, n, k = (int(option[name]) for name in ("--m", "--n", "--k"))
            output_elements, output, output_f64, witness = self._c6_witness(
                option, m, n, k)
            sidecar.write_text(json.dumps({
                "schema": E.C6_SIDECAR_SCHEMA,
                "sequence": ["reference", "candidate-1", "candidate-2", "candidate-3"],
                "backend": option["--backend"], "operation": self.c6_operation,
                "type_a": option["--type-a"], "type_b": "f32",
                "output_dtype": "f32", "m": m, "n": n, "k": k,
                "seed": (int(option["--seed"]) - 1 if self.c6_wrong_seed
                         else int(option["--seed"])),
                "output_elements": output_elements,
                "input_witness": witness,
                "reference_output_f32le_hex": output.hex(),
                "reference_output_f64le_hex": output_f64.hex(),
                "candidate_outputs_f32le_hex": (
                    [output.hex()] * 2 if self.c6_partial else
                    [output.hex(), output.hex(),
                     (output[:-1] + bytes([output[-1] ^ 1])).hex()]
                    if self.c6_nondeterministic else [output.hex()] * 3),
                "candidate_clone_ids": (
                    ["candidate-1", "candidate-2"] if self.c6_partial else
                    ["candidate-1", "candidate-2", "candidate-3"]),
            }, sort_keys=True))
            invocation.stdout_path.write_text("native C6 complete\n")
        else:
            c6_receipt = invocation.stdout_path.parent / "c6-receipt.json"
            if not c6_receipt.is_file():
                c6_receipt = invocation.stdout_path.parent.parent / "c6-receipt.json"
            self.c6_receipt_seen_before_target = c6_receipt.is_file()
            if self.c6_receipt_seen_before_target:
                E.proofs.load_receipt(
                    c6_receipt, schema=E.C6_CORRECTNESS_SCHEMA)
            invocation.stdout_path.write_text(correctness_console(
                self.correctness_summary, op=self.c6_operation))
        invocation.stderr_path.write_text("")
        return self._capture(
            invocation, 0 if "--sidecar" in invocation.argv
            else self.correctness_exit)

    def rocprof(self, invocation):
        self.calls.append((invocation.kind, invocation.arm, invocation.argv,
                           invocation.environment))
        invocation.stdout_path.write_text(f"{invocation.arm} profile complete\n")
        invocation.stderr_path.write_text("")
        invocation.timestamp_csv_path.write_text(csv_text(
            invocation.arm, inverse_bad=self.inverse_bad,
            invariant_changed=self.invariant_changed,
            forbidden=self.forbidden and invocation.arm == "candidate"))
        return self._capture(invocation, self.rocprof_exit)

    def _capture(self, invocation, exit_code):
        timestamp = 99 if self.non_overlap else 150
        return E.ExecutionCapture(
            argv=invocation.argv, exit_code=exit_code, child_pid=1234,
            started_at="2026-08-13T00:00:00Z",
            ended_at="2026-08-13T00:00:01Z",
            started_monotonic_ns=100, ended_monotonic_ns=200,
            samples=(E.GpuResidencySample(
                observed_monotonic_ns=timestamp, device_id="mi210_0",
                kfd_pids=(1234,), vram_bytes=4096),),
            runtime_maps_identity=self.runtime_maps.get(invocation.arm))


def runtime_maps_for(current: E.GpuSourceEvidencePlan, arm: str) -> dict:
    assert current.shared_runtime is not None
    runtime_body = json.loads(current.shared_runtime.runtime_receipt.path.read_text())
    hip = (current.shared_runtime.candidate_hip_library if arm == "candidate"
           else current.shared_runtime.anchor_hip_library)
    body = {
        "runtime_manifest_sha256": runtime_body["split_runtime_manifest"]["manifest_sha256"],
        "arm": arm,
        "reward_binary_sha256": current.shared_runtime.measurement_binary.sha256,
        "hip_library_sha256": hip.sha256,
        "model_path": str(current.identity_files.model.path),
        "model_sha256": current.model_sha256,
        "device_id": current.device_id, "kfd_pid": 1234,
        "boot_id": "boot-test", "process_start_ticks": 99,
        "mapped_local_sha256": {
            str(current.shared_runtime.measurement_binary.path): current.shared_runtime.measurement_binary.sha256,
            str(hip.path): hip.sha256,
        },
    }
    body["identity_sha256"] = E.split_runtime_verifier._content_hash(
        {"schema": E.split_runtime_verifier.MAPS_SCHEMA, **body})
    return {"schema": E.split_runtime_verifier.RESIDENCY_SCHEMA, **body}


class GpuSourceEvidenceTests(unittest.TestCase):
    def test_claim_checker_uses_schema_check_outcome(self):
        self.assertTrue(E._check_result_passed(E.schemas.Check(E.schemas.PASS)))
        self.assertFalse(E._check_result_passed(E.schemas.Check(E.schemas.FAIL)))
        self.assertFalse(E._check_result_passed(
            E.schemas.Check(E.schemas.COULD_NOT_CHECK)))

    @staticmethod
    def _dispatch(kernel: str, grid: int, workgroup: int, lds: int) -> dict:
        return {"kernel": kernel, "grid": grid, "workgroup": workgroup,
                "lds": lds, "blocks_per_call": grid // workgroup,
                "begin_ns": 1, "end_ns": 2}

    def test_exact_reducer_accepts_only_all_reviewed_geometries(self):
        pattern = r"^void quantize_q8_1"
        exact = (
            E.ExactDispatch("q8.small", pattern, 2, 1024, 256, 0, 4),
            E.ExactDispatch("q8.large", pattern, 1, 5120, 256, 0, 20),
        )
        rows = [self._dispatch("void quantize_q8_1<float>()", 1024, 256, 0)
                for _ in range(2)]
        rows.append(self._dispatch("void quantize_q8_1<float>()", 5120, 256, 0))
        reduced = E._reduce_arm(rows, exact=exact, forbidden=(), invariants=())
        self.assertEqual(reduced["exact"]["q8.small"]["calls"], 2)
        self.assertEqual(reduced["exact"]["q8.large"]["calls"], 1)

        with self.assertRaisesRegex(E.EvidenceProducerError, "count/geometry"):
            E._reduce_arm(rows[:-1], exact=exact, forbidden=(), invariants=())
        rows.append(self._dispatch("void quantize_q8_1<float>()", 2048, 256, 0))
        with self.assertRaisesRegex(E.EvidenceProducerError, "unreviewed geometry"):
            E._reduce_arm(rows, exact=exact, forbidden=(), invariants=())

    def produce(self, directory, *, executors=None, claims=None, plan_=None,
                verifier=lambda _receipt: True, c6_ready_timeout_s=120.0):
        executors = executors or FakeExecutors()
        claims = claims or ClaimFactory()
        bundle = E.produce_gpu_source_evidence(
            output_root=Path(directory) / "evidence",
            plan=plan_ or plan(Path(directory) / "inputs"),
            correctness_executor=executors.correctness,
            rocprof_executor=executors.rocprof,
            claim_journal=object(), claim_acquirer=claims,
            claim_verifier=verifier, claim_timeout_s=0,
            c6_ready_timeout_s=c6_ready_timeout_s)
        return bundle, executors, claims

    def test_happy_path_binds_exact_commands_claims_files_and_inverse(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, executors, claims = self.produce(directory)
            self.assertEqual(
                [row[:2] for row in executors.calls],
                [("correctness", "candidate")] * 5 +
                [("rocprof", "candidate"),
                 ("rocprof", "anchor")])
            self.assertTrue(all(claim.released for claim in claims.claims))
            self.assertEqual(len(claims.claims), 3)
            self.assertTrue(executors.c6_receipt_seen_before_target)
            self.assertNotEqual(executors.calls[5][3], executors.calls[6][3])
            self.assertEqual(
                Path(dict(executors.calls[5][3])["LD_LIBRARY_PATH"]).name,
                "candidate")
            self.assertEqual(
                Path(dict(executors.calls[6][3])["LD_LIBRARY_PATH"]).name,
                "anchor")
            root = Path(directory) / "evidence"
            loaded = E.load_gpu_source_evidence_bundle(root / "proof-bundle.json")
            self.assertEqual(loaded, bundle)
            pair = json.loads((root / "attribution-pair.json").read_text())
            self.assertTrue(pair["inverse_attribution_proved"])
            self.assertEqual(
                pair["invariant_signatures"]["hot_invariant"]["calls"], 1)
            candidate = json.loads(
                (root / "attribution-candidate/receipt.json").read_text())
            self.assertIn("timestamp_reduction_sha256", candidate)
            self.assertTrue(any(
                row["role"] == "timestamp_input"
                for row in candidate["command_input_files"]))
            for relative in (
                "correctness/stdout.txt", "correctness/stderr.txt",
                "attribution-candidate/stdout.txt",
                "attribution-candidate/stderr.txt",
                "attribution-candidate/timestamps.csv",
                "attribution-anchor/stdout.txt", "attribution-anchor/stderr.txt",
                "attribution-anchor/timestamps.csv"):
                self.assertTrue((root / relative).is_file())

    def test_rms_and_flash_native_sidecars_run_before_targeted_correctness(self):
        for operation, type_a, dimensions in (
                ("RMS_NORM", "f32", (256, 1, 1)),
                ("FLASH_ATTN_EXT", "f16", (64, 1, 128))):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                current = plan(Path(directory) / "inputs")
                c6 = replace(
                    current.c6_correctness,
                    argv=(current.c6_correctness.argv[0], "--mode", "oracle",
                          "--operation", operation,
                          "--backend", "ROCm0", "--type-a", type_a,
                          "--m", str(dimensions[0]), "--n", str(dimensions[1]),
                          "--k", str(dimensions[2]), "--seed", "42",
                          "--sidecar", E.C6_SIDECAR_OUTPUT),
                    precision_contract={
                        "required_output_dtype": "f32",
                        "required_accumulator_dtype": "f32",
                        "atol": .001, "rtol": .001,
                        "required_matched_ratio": 1.0, "lowbit": False})
                current = replace(
                    current, correctness_op=operation, c6_correctness=c6)
                current.policy.path.write_text(json.dumps(
                    E._policy_payload(current), sort_keys=True))
                current = replace(current, policy=E.BoundInputFile(
                    "execution_policy", current.policy.path,
                    hashlib.sha256(current.policy.path.read_bytes()).hexdigest()))
                bundle, executors, _ = self.produce(
                    directory, plan_=current)
                self.assertTrue(executors.c6_receipt_seen_before_target)
                self.assertEqual(
                    bundle.correctness["body"]["c6_correctness"]["body"]
                    ["seeded_case_identity"]["operation"], operation)

    def test_aggregate_correctness_seals_c6_before_first_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            contracts = tuple({
                "invocation_id": f"case-{index}",
                "argv": list(current.correctness_argv),
                "backend": current.correctness_backend,
                "op": current.correctness_op,
                "case_set": f"set-{index}",
                "expected_cases": current.expected_correctness_cases,
                "required_cases": [],
            } for index in (1, 2))
            current = replace(current, correctness_invocations=contracts)
            current.policy.path.write_text(json.dumps(
                E._policy_payload(current), sort_keys=True))
            current = replace(current, policy=E.BoundInputFile(
                "execution_policy", current.policy.path,
                hashlib.sha256(current.policy.path.read_bytes()).hexdigest()))
            bundle, executors, claims = self.produce(
                directory, plan_=current)
            self.assertEqual(
                [row[:2] for row in executors.calls[:6]],
                [("correctness", "candidate")] * 6)
            self.assertTrue(executors.c6_receipt_seen_before_target)
            self.assertEqual(len(claims.claims), 4)
            self.assertEqual(
                bundle.correctness["body"]["invocations"][0]["body"]
                ["c6_claim_join"], "same_held_claim")
            self.assertEqual(
                bundle.correctness["body"]["invocations"][1]["body"]
                ["c6_claim_join"], "sealed_c6_restart")

    def test_native_c6_refuses_partial_or_nondeterministic_three_run_witness(self):
        for label, executor, calls in (
                ("partial", FakeExecutors(c6_partial=True), 4),
                ("nondeterministic", FakeExecutors(c6_nondeterministic=True), 4),
                ("wrong-seed", FakeExecutors(c6_wrong_seed=True), 1)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(E.EvidenceProducerError):
                    self.produce(directory, executors=executor)
                self.assertEqual(len(executor.calls), calls)
                self.assertFalse((Path(directory) / "evidence" /
                                  "correctness" / "receipt.json").exists())

    def test_sealed_c6_crash_boundary_reuses_exact_receipt_without_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            first = FakeExecutors()
            original = E._validate_c6_correctness_receipt
            crashed = False

            def crash_after_reopen(loaded, plan_):
                nonlocal crashed
                original(loaded, plan_)
                if not crashed:
                    crashed = True
                    raise RuntimeError("crash after sealed C6 reopen")

            with mock.patch.object(
                    E, "_validate_c6_correctness_receipt",
                    side_effect=crash_after_reopen), self.assertRaisesRegex(
                        E.EvidenceProducerError, "crash after sealed C6"):
                self.produce(directory, executors=first, plan_=current)
            self.assertEqual(len(first.calls), 4)
            c6_receipt = (Path(directory) / "evidence" / "correctness" /
                          "c6-receipt.json")
            self.assertTrue(c6_receipt.is_file())

            resumed = FakeExecutors()
            bundle, resumed, claims = self.produce(
                directory, executors=resumed, plan_=current)
            self.assertIsInstance(bundle, E.proofs.GpuSourceProofBundle)
            self.assertEqual([row[:2] for row in resumed.calls], [
                ("correctness", "candidate"),
                ("rocprof", "candidate"), ("rocprof", "anchor")])
            self.assertTrue(resumed.c6_receipt_seen_before_target)
            correctness = json.loads((Path(directory) / "evidence" /
                                      "correctness" / "receipt.json").read_text())
            self.assertEqual(correctness["c6_claim_join"], "sealed_c6_restart")
            self.assertEqual(len(claims.claims), 3)

    def test_tampered_sealed_c6_restart_fails_before_any_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            original = E._validate_c6_correctness_receipt

            def crash_after_reopen(loaded, plan_):
                original(loaded, plan_)
                raise RuntimeError("crash after sealed C6 reopen")

            with mock.patch.object(
                    E, "_validate_c6_correctness_receipt",
                    side_effect=crash_after_reopen), self.assertRaises(
                        E.EvidenceProducerError):
                self.produce(directory, executors=FakeExecutors(), plan_=current)
            sidecar = (Path(directory) / "evidence" / "correctness" /
                       "c6-oracle-sidecar.json")
            sidecar.write_text(sidecar.read_text() + " ")
            resumed = FakeExecutors()
            with self.assertRaisesRegex(E.EvidenceProducerError, "sidecar changed"):
                self.produce(directory, executors=resumed, plan_=current)
            self.assertEqual(resumed.calls, [])

    def test_native_c6_sidecar_tamper_refuses_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            self.produce(directory)
            sidecar = (Path(directory) / "evidence" / "correctness" /
                       "c6-oracle-sidecar.json")
            sidecar.write_text(sidecar.read_text() + " ")
            with self.assertRaisesRegex(E.EvidenceProducerError, "sidecar changed"):
                E.load_gpu_source_evidence_bundle(
                    Path(directory) / "evidence" / "proof-bundle.json")

    def test_production_evidence_refuses_missing_native_c6_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            current = replace(plan(Path(directory) / "inputs"),
                              c6_correctness=None)
            policy_path = current.policy.path
            policy_path.write_text(json.dumps(E._policy_payload(current), sort_keys=True))
            current = replace(current, policy=E.BoundInputFile(
                "execution_policy", policy_path,
                hashlib.sha256(policy_path.read_bytes()).hexdigest()))
            with self.assertRaisesRegex(E.EvidenceProducerError, "lacks native C6"):
                self.produce(directory, plan_=current)

    def test_split_c6_binds_oracle_candidate_legs_and_handshake(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, executors, claims = self.produce(directory)
            self.assertEqual(
                [row[:2] for row in executors.calls],
                [("correctness", "candidate")] * 5 +
                [("rocprof", "candidate"), ("rocprof", "anchor")])
            self.assertTrue(all(claim.released for claim in claims.claims))
            self.assertTrue(executors.c6_receipt_seen_before_target)
            correctness = Path(directory) / "evidence" / "correctness"
            c6_receipt = json.loads(
                (correctness / "c6-receipt.json").read_text())
            body = c6_receipt
            self.assertEqual(body["c6_process_mode"], "oracle_candidate_split")
            self.assertTrue((correctness / "c6-oracle-sidecar.json").is_file())
            self.assertTrue((correctness / "c6-input-binding.json").is_file())
            inputs = sorted(path.name for path in
                            (correctness / "c6-inputs").iterdir())
            self.assertEqual(inputs, ["activations_f32le.bin", "weights.bin"])
            for path in (correctness / "c6-inputs").iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            legs = body["per_leg_bindings"]
            self.assertEqual(len(legs), 3)
            for index, leg in enumerate(legs, 1):
                self.assertEqual(leg["leg_index"], index)
                self.assertEqual(leg["exit_code"], 0)
                self.assertGreaterEqual(
                    leg["event_stream"]["ready_observed_monotonic_ns"],
                    leg["event_stream"]["launched_monotonic_ns"])
                self.assertGreaterEqual(
                    leg["event_stream"]["continue_written_monotonic_ns"],
                    leg["event_stream"]["ready_observed_monotonic_ns"])
                self.assertGreaterEqual(
                    leg["event_stream"]["completed_monotonic_ns"],
                    leg["event_stream"]["continue_written_monotonic_ns"])
                for suffix in ("ready", "continue"):
                    path = Path(leg[f"{suffix}_path"])
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.stat().st_nlink, 1)
            self.assertEqual(
                body["input_identity_sha256"],
                body["input_binding"]["input_identity_sha256"])
            reopened = E.load_gpu_source_evidence_bundle(
                Path(directory) / "evidence" / "proof-bundle.json")
            self.assertEqual(reopened, bundle)

    def test_combined_c6_mode_still_seals_without_oracle_split(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            c6 = replace(current.c6_correctness, argv=(
                current.c6_correctness.argv[0],
                "--operation", "MUL_MAT", "--backend", "ROCm0",
                "--type-a", "q4_K", "--m", "32", "--n", "1", "--k", "256",
                "--seed", "42", "--sidecar", E.C6_SIDECAR_OUTPUT))
            current = replace(current, c6_correctness=c6)
            current.policy.path.write_text(json.dumps(
                E._policy_payload(current), sort_keys=True))
            current = replace(current, policy=E.BoundInputFile(
                "execution_policy", current.policy.path,
                hashlib.sha256(current.policy.path.read_bytes()).hexdigest()))
            bundle, executors, _ = self.produce(directory, plan_=current)
            self.assertEqual(
                [row[:2] for row in executors.calls],
                [("correctness", "candidate")] * 2 +
                [("rocprof", "candidate"), ("rocprof", "anchor")])
            body = json.loads((Path(directory) / "evidence" / "correctness" /
                               "c6-receipt.json").read_text())
            self.assertNotIn("c6_process_mode", body)
            self.assertNotIn("per_leg_bindings", body)

    def test_split_c6_refuses_missing_ready_token(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    E.EvidenceProducerError,
                    "ready token was not observed"):
                self.produce(directory, executors=FakeExecutors(
                    c6_stall_ready=True), c6_ready_timeout_s=0.3)
            self.assertFalse((Path(directory) / "evidence" /
                              "correctness" / "receipt.json").exists())

    def test_split_c6_refuses_prearmed_continue_path(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            ready = Path(directory) / "leg-ready"
            continue_path = Path(directory) / "leg-continue"
            continue_path.write_bytes(b"C")
            invocation = E.CommandInvocation(
                kind="correctness", arm="candidate",
                argv=E._c6_candidate_argv_from_argv(
                    current.c6_correctness.argv,
                    input_dir=Path(directory) / "inputs",
                    output=Path(directory) / "leg-out.bin",
                    ready_file=ready, continue_file=continue_path),
                stdout_path=(Path(directory) / "leg-stdout.txt").resolve(),
                stderr_path=(Path(directory) / "leg-stderr.txt").resolve(),
                working_directory=current.execution_cwd,
                environment=current.correctness_environment)
            with self.assertRaisesRegex(
                    E.EvidenceProducerError, "handshake paths are not fresh"):
                E._paced_candidate(
                    FakeExecutors().correctness, invocation, ready,
                    continue_path)

    def test_split_c6_refuses_candidate_leg_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            executors = FakeExecutors(c6_candidate_exit=1)
            with self.assertRaisesRegex(
                    E.EvidenceProducerError, "exited nonzero"):
                self.produce(directory, executors=executors)
            self.assertEqual(len(executors.calls), 2)
            self.assertFalse((Path(directory) / "evidence" /
                              "correctness" / "receipt.json").exists())

    def test_split_c6_tampered_candidate_output_refuses_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            self.produce(directory)
            output = (Path(directory) / "evidence" / "correctness" /
                      "c6-candidate-2-output.bin")
            output.write_bytes(output.read_bytes()[:-1] + b"\x00")
            with self.assertRaisesRegex(E.EvidenceProducerError, "changed"):
                E.load_gpu_source_evidence_bundle(
                    Path(directory) / "evidence" / "proof-bundle.json")

    def test_split_c6_tampered_input_binding_refuses_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            self.produce(directory)
            weights = (Path(directory) / "evidence" / "correctness" /
                       "c6-inputs" / "weights.bin")
            weights.write_bytes(weights.read_bytes() + b"\x00")
            with self.assertRaisesRegex(E.EvidenceProducerError, "changed"):
                E.load_gpu_source_evidence_bundle(
                    Path(directory) / "evidence" / "proof-bundle.json")

    def test_split_c6_tampered_leg_argv_refuses_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            bundle, _, _ = self.produce(directory, plan_=current)
            receipt = Path(directory) / "evidence" / "correctness" / \
                "c6-receipt.json"
            loaded = json.loads(receipt.read_text())
            loaded["per_leg_bindings"][0]["argv"][0] = "/tmp/other"
            loaded["receipt_sha256"] = E.schemas.content_hash({
                key: value for key, value in loaded.items()
                if key != "receipt_sha256"})
            receipt.write_text(json.dumps(loaded, sort_keys=True) + "\n")
            with self.assertRaisesRegex(
                    E.EvidenceProducerError, "derivation"):
                E._validate_c6_correctness_receipt(
                    E.proofs.load_receipt(receipt, schema=E.C6_CORRECTNESS_SCHEMA),
                    current)

    def test_split_c6_tampered_leg_event_stream_refuses_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            self.produce(directory, plan_=current)
            receipt = Path(directory) / "evidence" / "correctness" / \
                "c6-receipt.json"
            loaded = json.loads(receipt.read_text())
            loaded["per_leg_bindings"][0]["event_stream"][
                "continue_written_monotonic_ns"] = 1
            loaded["receipt_sha256"] = E.schemas.content_hash({
                key: value for key, value in loaded.items()
                if key != "receipt_sha256"})
            receipt.write_text(json.dumps(loaded, sort_keys=True) + "\n")
            with self.assertRaisesRegex(
                    E.EvidenceProducerError, "not monotonic"):
                E._validate_c6_correctness_receipt(
                    E.proofs.load_receipt(receipt, schema=E.C6_CORRECTNESS_SCHEMA),
                    current)

    def test_shared_reward_rocprof_uses_one_binary_with_separate_hashed_hip_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs", shared_reward=True)
            runtime_body = json.loads(current.shared_runtime.runtime_receipt.path.read_text())
            split = runtime_body["split_runtime_manifest"]
            class VerifiedRuntime:
                reward_binary = current.shared_runtime.measurement_binary.path
                anchor_hip_dir = current.shared_runtime.anchor_hip_library.path.parent
                candidate_hip_dir = current.shared_runtime.candidate_hip_library.path.parent
                def to_dict(self): return split
            with mock.patch.object(E.split_runtime_verifier, "verify_split_runtime",
                                   return_value=VerifiedRuntime()):
                bundle, executors, _claims = self.produce(
                    directory, plan_=current,
                    executors=FakeExecutors(runtime_maps={
                        "candidate": runtime_maps_for(current, "candidate"),
                        "anchor": runtime_maps_for(current, "anchor")}))
            self.assertEqual(executors.calls[5][2], executors.calls[6][2])
            self.assertIn(str(current.shared_runtime.measurement_binary.path), executors.calls[5][2])
            self.assertNotEqual(dict(executors.calls[5][3])["LD_LIBRARY_PATH"],
                                dict(executors.calls[6][3])["LD_LIBRARY_PATH"])
            pair = json.loads((Path(directory) / "evidence" / "attribution-pair.json").read_text())
            self.assertEqual(pair["shared_runtime"], E._shared_runtime_reference(current.shared_runtime))
            self.assertEqual(bundle.candidate.hip_library_sha256,
                             current.shared_runtime.candidate_hip_library.sha256)
            self.assertEqual(pair["candidate_runtime_maps_identity"]["arm"], "candidate")
            self.assertEqual(pair["anchor_runtime_maps_identity"]["arm"], "anchor")

    def test_shared_reward_injects_only_producer_owned_distinct_rocprof_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs", shared_reward=True)
            argv = current.candidate_rocprof_argv + ("-o", E.ROCPROF_TIMESTAMP_OUTPUT)
            current = replace(current, candidate_rocprof_argv=argv, anchor_rocprof_argv=argv)
            current.policy.path.write_text(json.dumps(E._policy_payload(current), sort_keys=True))
            current = replace(current, policy=replace(
                current.policy, sha256=hashlib.sha256(current.policy.path.read_bytes()).hexdigest()))
            runtime_body = json.loads(current.shared_runtime.runtime_receipt.path.read_text())
            split = runtime_body["split_runtime_manifest"]
            class VerifiedRuntime:
                reward_binary = current.shared_runtime.measurement_binary.path
                anchor_hip_dir = current.shared_runtime.anchor_hip_library.path.parent
                candidate_hip_dir = current.shared_runtime.candidate_hip_library.path.parent
                def to_dict(self): return split
            with mock.patch.object(E.split_runtime_verifier, "verify_split_runtime",
                                   return_value=VerifiedRuntime()):
                _bundle, executors, _claims = self.produce(
                    directory, plan_=current,
                    executors=FakeExecutors(runtime_maps={
                        "candidate": runtime_maps_for(current, "candidate"),
                        "anchor": runtime_maps_for(current, "anchor")}))
            candidate_argv, anchor_argv = executors.calls[5][2], executors.calls[6][2]
            self.assertNotEqual(candidate_argv, anchor_argv)
            self.assertEqual(
                E._receipt_rocprof_template({"command_argv": candidate_argv,
                                              "timestamp_csv_path": str(Path(directory) / "evidence" / "attribution-candidate" / "timestamps.csv")}),
                E._receipt_rocprof_template({"command_argv": anchor_argv,
                                              "timestamp_csv_path": str(Path(directory) / "evidence" / "attribution-anchor" / "timestamps.csv")}))
            self.assertIn(str(Path(directory) / "evidence" / "attribution-candidate" / "timestamps.csv"), candidate_argv)
            self.assertIn(str(Path(directory) / "evidence" / "attribution-anchor" / "timestamps.csv"), anchor_argv)

    def test_tampered_file_is_rejected_by_recursive_bundle_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            self.produce(directory)
            root = Path(directory) / "evidence"
            (root / "correctness/stdout.txt").write_text("3/3 tests passed\ntamper\n")
            with self.assertRaisesRegex(E.EvidenceProducerError, "bytes changed"):
                E.load_gpu_source_evidence_bundle(root / "proof-bundle.json")

    def test_wrong_identity_and_runtime_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.produce(directory)
            root = Path(directory) / "evidence"
            candidate = P.load_receipt(
                root / "attribution-candidate/receipt.json",
                schema=E.ATTRIBUTION_SCHEMA)["body"]
            wrong_identity = plan(Path(directory) / "wrong-identity")
            object.__setattr__(wrong_identity, "candidate", identity("q"))
            with self.assertRaisesRegex(E.EvidenceProducerError, "identity/config"):
                E._validate_attribution_body(candidate, plan=wrong_identity,
                                             arm="candidate")
            wrong_config = plan(Path(directory) / "wrong-config")
            object.__setattr__(wrong_config, "runtime_config_sha256", digest("z"))
            with self.assertRaisesRegex(E.EvidenceProducerError, "identity/config"):
                E._validate_attribution_body(candidate, plan=wrong_config,
                                             arm="candidate")

    def test_nonoverlap_refuses_but_releases_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            claims = ClaimFactory()
            with self.assertRaisesRegex(E.EvidenceProducerError, "no KFD"):
                self.produce(directory, executors=FakeExecutors(non_overlap=True),
                             claims=claims)
            self.assertEqual(len(claims.claims), 1)
            self.assertTrue(claims.claims[0].released)
            self.assertFalse((Path(directory) / "evidence/proof-bundle.json").exists())

    def test_failed_exit_and_failed_correctness_never_mint_bundle(self):
        for executor, message in (
            (FakeExecutors(correctness_exit=7), "exited nonzero"),
            (FakeExecutors(correctness_summary="2/2 tests passed"),
             "exact expected case count"),
            (FakeExecutors(rocprof_exit=9), "rocprof command exited nonzero"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                claims = ClaimFactory()
                with self.assertRaisesRegex(E.EvidenceProducerError, message):
                    self.produce(directory, executors=executor, claims=claims)
                self.assertTrue(all(claim.released for claim in claims.claims))
                self.assertFalse((Path(directory) / "evidence/proof-bundle.json").exists())

    def test_completed_correctness_survives_a_later_attribution_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            with self.assertRaisesRegex(
                    E.EvidenceProducerError, "rocprof command exited nonzero"):
                self.produce(
                    directory, plan_=current,
                    executors=FakeExecutors(rocprof_exit=9))
            receipt = Path(directory) / "evidence/correctness/receipt.json"
            self.assertTrue(receipt.is_file())
            loaded = E.load_gpu_source_correctness_receipt(receipt, current)
            self.assertEqual(loaded["body"]["result"], "PASS")
            self.assertEqual(loaded["body"]["passed_cases"], 3)

    def test_typed_parse_refusal_is_durable_and_releases_the_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            claims = ClaimFactory()
            with self.assertRaisesRegex(
                    E.CorrectnessParseRefusal, "no test-backend-ops console frame"):
                self.produce(
                    directory, plan_=current, claims=claims,
                    executors=FakeExecutors(correctness_summary="unparseable"))
            self.assertEqual(len(claims.claims), 1)
            self.assertTrue(claims.claims[0].released)
            refusal = Path(directory) / "evidence/correctness/refusal.json"
            self.assertTrue(refusal.is_file())
            loaded = E.load_gpu_source_correctness_refusal(refusal, current)
            self.assertEqual(
                loaded["body"]["classification"], "output_parse_refusal")
            self.assertFalse(
                (Path(directory) / "evidence/correctness/receipt.json").exists())

    def test_missing_release_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(E.EvidenceProducerError, "release is missing"):
                self.produce(directory, claims=ClaimFactory(missing_release_at=1))

    def test_inverse_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            claims = ClaimFactory()
            with self.assertRaisesRegex(E.EvidenceProducerError, "old_kernel"):
                self.produce(directory, executors=FakeExecutors(inverse_bad=True),
                             claims=claims)
            self.assertEqual(len(claims.claims), 3)
            self.assertTrue(all(claim.released for claim in claims.claims))

    def test_forbidden_dispatch_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(E.EvidenceProducerError, "forbidden dispatch"):
                self.produce(directory, executors=FakeExecutors(forbidden=True))

    def test_changed_invariant_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(E.EvidenceProducerError, "changed an invariant"):
                self.produce(directory,
                             executors=FakeExecutors(invariant_changed=True))

    def test_unverified_claim_refuses_before_executor_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            executors, claims = FakeExecutors(), ClaimFactory()
            with self.assertRaisesRegex(E.EvidenceProducerError, "before execution"):
                self.produce(directory, executors=executors, claims=claims,
                             verifier=lambda _receipt: False)
            self.assertEqual(executors.calls, [])
            self.assertTrue(claims.claims[0].released)

    def test_file_backed_identity_policy_and_profiler_input_tamper_refuse(self):
        for target in ("candidate_binary", "policy", "timestamp_input"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                current = plan(Path(directory) / "inputs")
                if target == "candidate_binary":
                    path = current.identity_files.candidate.binary.path
                elif target == "policy":
                    path = current.policy.path
                else:
                    path = next(x.path for x in current.candidate_rocprof_inputs
                                if x.role == "timestamp_input")
                path.write_bytes(path.read_bytes() + b"tamper")
                executors = FakeExecutors()
                with self.assertRaisesRegex(E.EvidenceProducerError, "bytes changed"):
                    self.produce(directory, executors=executors, plan_=current)
                self.assertEqual(executors.calls, [])
                self.assertFalse((Path(directory) / "evidence").exists())

    def test_source_tree_receipt_requires_complete_self_hashed_tree_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            source = current.identity_files.candidate.source_identity.path
            body = json.loads(source.read_text())
            body["tree"]["entries"][0][2] = "../measurement-reward-hack"
            body["receipt_sha256"] = E.schemas.content_hash(
                {key: value for key, value in body.items() if key != "receipt_sha256"})
            source.write_text(json.dumps(body, sort_keys=True))
            carrier = replace(current.identity_files.candidate.source_identity,
                              sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            current = replace(current, identity_files=replace(
                current.identity_files,
                candidate=replace(current.identity_files.candidate,
                                  source_identity=carrier)))
            policy = json.loads(current.policy.path.read_text())
            policy["identity_files"] = E._identity_files_reference(current.identity_files)
            current.policy.path.write_text(json.dumps(policy, sort_keys=True))
            current = replace(current, policy=replace(
                current.policy,
                sha256=hashlib.sha256(current.policy.path.read_bytes()).hexdigest()))
            with self.assertRaisesRegex(E.EvidenceProducerError, "source tree entry"):
                self.produce(directory, plan_=current)

    def test_unapplied_manifest_wrong_environment_and_cwd_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            materialization = current.identity_files.materialization.path
            raw = json.loads(materialization.read_text())
            raw["patch_applied"] = False
            raw["receipt_sha256"] = E.schemas.content_hash(
                {key: value for key, value in raw.items() if key != "receipt_sha256"})
            materialization.write_text(json.dumps(raw, sort_keys=True))
            changed = replace(
                current.identity_files.materialization,
                sha256=hashlib.sha256(materialization.read_bytes()).hexdigest())
            current = replace(current, identity_files=replace(
                current.identity_files, materialization=changed))
            policy = json.loads(current.policy.path.read_text())
            policy["correctness_environment"] = [list(x) for x in current.correctness_environment]
            policy["candidate_rocprof_environment"] = [list(x) for x in current.candidate_rocprof_environment]
            policy["anchor_rocprof_environment"] = [list(x) for x in current.anchor_rocprof_environment]
            current.policy.path.write_text(json.dumps(policy, sort_keys=True))
            current = replace(current, policy=replace(
                current.policy,
                sha256=hashlib.sha256(current.policy.path.read_bytes()).hexdigest()))
            with self.assertRaisesRegex(E.EvidenceProducerError, "manifest-applied"):
                self.produce(directory, plan_=current)
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            with self.assertRaisesRegex(E.EvidenceProducerError, "LD_LIBRARY_PATH"):
                replace(current, correctness_environment=(
                    ("LD_LIBRARY_PATH", "/wrong/generation"),))
            bad_cwd = Path(directory) / "missing"
            with self.assertRaisesRegex(E.EvidenceProducerError, "working directory"):
                replace(current, execution_cwd=bad_cwd)

    def test_shell_metacharacters_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            with self.assertRaisesRegex(E.EvidenceProducerError, "shell metacharacters"):
                replace(current, correctness_argv=(
                    str(current.correctness_inputs[0].path), ";", "evil"))

    def test_direct_executor_spawns_exact_argv_without_shell_and_samples_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation = E.CommandInvocation(
                kind="correctness", arm="candidate", argv=("/bin/true",),
                stdout_path=(root / "stdout").resolve(),
                stderr_path=(root / "stderr").resolve(),
                working_directory=root.resolve(),
                environment=(("LD_LIBRARY_PATH", "/test/lib"),))

            class Child:
                pid = 9191
                polls = 0
                def poll(self):
                    self.polls += 1
                    return None if self.polls == 1 else 0
                def wait(self): return 0
                def terminate(self): raise AssertionError("unexpected terminate")

            child = Child()
            popen = mock.Mock(return_value=child)
            sampler = lambda pid: E.GpuResidencySample(
                observed_monotonic_ns=__import__("time").monotonic_ns(),
                device_id="mi210_0", kfd_pids=(pid,), vram_bytes=4096)
            executor = E.SubprocessCommandExecutor(
                residency_sampler=sampler,
                sample_interval_s=.00001, popen=popen)
            capture = executor(invocation)
            self.assertEqual(capture.argv, ("/bin/true",))
            self.assertEqual(capture.child_pid, 9191)
            self.assertTrue(capture.samples)
            self.assertNotIn("shell", popen.call_args.kwargs)
            self.assertEqual(popen.call_args.args[0], ["/bin/true"])
            self.assertEqual(popen.call_args.kwargs["cwd"], str(root.resolve()))
            self.assertEqual(popen.call_args.kwargs["env"]["LD_LIBRARY_PATH"],
                             "/test/lib")

    def test_direct_executor_captures_maps_only_during_owned_resident_rocprof_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation = E.CommandInvocation(
                kind="rocprof", arm="candidate", argv=("/bin/true",),
                stdout_path=(root / "stdout").resolve(), stderr_path=(root / "stderr").resolve(),
                timestamp_csv_path=(root / "timestamps.csv").resolve(),
                working_directory=root.resolve(), environment=(("LD_LIBRARY_PATH", "/test/lib"),),
                runtime_maps_required=True, runtime_maps_context={"sealed": "context"})
            class Child:
                pid = 9393; polls = 0
                def poll(self): self.polls += 1; return None if self.polls == 1 else 0
                def wait(self): return 0
            seen = []
            def maps(call, child_pid, sample):
                seen.append((call.arm, child_pid, sample.kfd_pids, call.runtime_maps_context))
                return {"maps": "captured"}
            executor = E.SubprocessCommandExecutor(
                residency_sampler=lambda pid: E.GpuResidencySample(
                    observed_monotonic_ns=__import__("time").monotonic_ns(), device_id="mi210_0",
                    kfd_pids=(pid,), vram_bytes=4096), runtime_maps_sampler=maps,
                sample_interval_s=.00001, popen=mock.Mock(return_value=Child()))
            capture = executor(invocation)
            self.assertEqual(capture.runtime_maps_identity, {"maps": "captured"})
            self.assertEqual(seen, [("candidate", 9393, (9393,), {"sealed": "context"})])

    def test_direct_executor_retries_only_typed_runtime_maps_startup_race(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation = E.CommandInvocation(
                kind="rocprof", arm="candidate", argv=("/bin/true",),
                stdout_path=(root / "stdout").resolve(), stderr_path=(root / "stderr").resolve(),
                timestamp_csv_path=(root / "timestamps.csv").resolve(),
                working_directory=root.resolve(), environment=(("LD_LIBRARY_PATH", "/test/lib"),),
                runtime_maps_required=True, runtime_maps_context={"sealed": "context"})

            class Child:
                pid = 9394; polls = 0
                def poll(self):
                    self.polls += 1
                    return None if self.polls <= 2 else 0
                def wait(self): return 0

            attempts = []
            def maps(_call, _child_pid, _sample):
                attempts.append(len(attempts) + 1)
                if len(attempts) == 1:
                    raise E.RuntimeMapsNotReady("model mapping is not complete")
                return {"maps": "captured-after-startup"}

            executor = E.SubprocessCommandExecutor(
                residency_sampler=lambda pid: E.GpuResidencySample(
                    observed_monotonic_ns=__import__("time").monotonic_ns(), device_id="mi210_0",
                    kfd_pids=(pid,), vram_bytes=4096), runtime_maps_sampler=maps,
                sample_interval_s=.00001, popen=mock.Mock(return_value=Child()))
            capture = executor(invocation)
            self.assertEqual(attempts, [1, 2])
            self.assertEqual(capture.runtime_maps_identity,
                             {"maps": "captured-after-startup"})

    def test_direct_executor_refuses_if_runtime_maps_never_become_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation = E.CommandInvocation(
                kind="rocprof", arm="candidate", argv=("/bin/true",),
                stdout_path=(root / "stdout").resolve(), stderr_path=(root / "stderr").resolve(),
                timestamp_csv_path=(root / "timestamps.csv").resolve(),
                working_directory=root.resolve(), environment=(("LD_LIBRARY_PATH", "/test/lib"),),
                runtime_maps_required=True, runtime_maps_context={"sealed": "context"})

            class Child:
                pid = 9395; polls = 0
                def poll(self): self.polls += 1; return None if self.polls == 1 else 0
                def wait(self): return 0

            executor = E.SubprocessCommandExecutor(
                residency_sampler=lambda pid: E.GpuResidencySample(
                    observed_monotonic_ns=__import__("time").monotonic_ns(), device_id="mi210_0",
                    kfd_pids=(pid,), vram_bytes=4096),
                runtime_maps_sampler=lambda *_args: (_ for _ in ()).throw(
                    E.RuntimeMapsNotReady("model mapping is not complete")),
                sample_interval_s=.00001, popen=mock.Mock(return_value=Child()))
            with self.assertRaisesRegex(E.EvidenceProducerError,
                                        "did not prove the sealed arm"):
                executor(invocation)

    def test_direct_executor_does_not_retry_untyped_runtime_maps_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation = E.CommandInvocation(
                kind="rocprof", arm="candidate", argv=("/bin/true",),
                stdout_path=(root / "stdout").resolve(), stderr_path=(root / "stderr").resolve(),
                timestamp_csv_path=(root / "timestamps.csv").resolve(),
                working_directory=root.resolve(), environment=(("LD_LIBRARY_PATH", "/test/lib"),),
                runtime_maps_required=True, runtime_maps_context={"sealed": "context"})

            class Child:
                pid = 9396; alive = True; terminated = False
                def poll(self): return None if self.alive else -15
                def terminate(self): self.terminated = True; self.alive = False
                def wait(self, timeout=None): return -15

            child = Child()
            executor = E.SubprocessCommandExecutor(
                residency_sampler=lambda pid: E.GpuResidencySample(
                    observed_monotonic_ns=__import__("time").monotonic_ns(), device_id="mi210_0",
                    kfd_pids=(pid,), vram_bytes=4096),
                runtime_maps_sampler=lambda *_args: (_ for _ in ()).throw(
                    E.EvidenceProducerError("runtime identity is ambiguous")),
                sample_interval_s=.00001, popen=mock.Mock(return_value=child))
            with self.assertRaisesRegex(E.EvidenceProducerError,
                                        "runtime identity is ambiguous"):
                executor(invocation)
            self.assertTrue(child.terminated)

    def test_shared_rocprof_without_production_maps_callback_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation = E.CommandInvocation(
                kind="rocprof", arm="candidate", argv=("/bin/true",),
                stdout_path=(root / "stdout").resolve(), stderr_path=(root / "stderr").resolve(),
                timestamp_csv_path=(root / "timestamps.csv").resolve(),
                working_directory=root.resolve(), environment=(("LD_LIBRARY_PATH", "/test/lib"),),
                runtime_maps_required=True, runtime_maps_context={"sealed": "context"})
            class Child:
                pid = 9494; polls = 0
                def poll(self): self.polls += 1; return None if self.polls == 1 else 0
                def wait(self, timeout=None): return 0
                def terminate(self): return None
            executor = E.SubprocessCommandExecutor(
                residency_sampler=lambda pid: E.GpuResidencySample(
                    observed_monotonic_ns=__import__("time").monotonic_ns(), device_id="mi210_0",
                    kfd_pids=(pid,), vram_bytes=4096), sample_interval_s=.00001,
                popen=mock.Mock(return_value=Child()))
            with self.assertRaisesRegex(E.EvidenceProducerError, "maps sampler"):
                executor(invocation)

    def test_sampler_exception_terminates_then_kills_exact_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation = E.CommandInvocation(
                kind="correctness", arm="candidate", argv=("/bin/true",),
                stdout_path=(root / "stdout").resolve(),
                stderr_path=(root / "stderr").resolve(),
                working_directory=root.resolve(),
                environment=(("LD_LIBRARY_PATH", "/test/lib"),))

            class Child:
                pid = 9292
                alive = True
                terminated = False
                killed = False
                def poll(self): return None if self.alive else -9
                def terminate(self): self.terminated = True
                def kill(self): self.killed = True; self.alive = False
                def wait(self, timeout=None):
                    if self.alive and timeout is not None:
                        raise __import__("subprocess").TimeoutExpired("fake", timeout)
                    return -9

            child = Child()
            executor = E.SubprocessCommandExecutor(
                residency_sampler=lambda _pid: (_ for _ in ()).throw(
                    RuntimeError("sample failed")),
                sample_interval_s=.00001,
                popen=mock.Mock(return_value=child))
            with self.assertRaisesRegex(RuntimeError, "sample failed"):
                executor(invocation)
            self.assertTrue(child.terminated)
            self.assertTrue(child.killed)
            self.assertIsNotNone(child.poll())

    def test_borrowed_phase_end_cannot_masquerade_as_physical_release(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_plan = plan(Path(directory) / "inputs")
            opened = FakeClaim(1, evidence_plan.device_id, {
                "holder_label": "outer", "purpose": "outer",
                "campaign_id": evidence_plan.campaign_id}).receipt()
            phase_end = {
                "schema": E.BORROWED_PHASE_SCHEMA,
                "mode": "borrowed_outer_reservation",
                "outer_claim_id": opened["claim_id"],
                "device_id": evidence_plan.device_id,
                "campaign_id": evidence_plan.campaign_id,
                "phase_ended_at": "2026-08-14T00:00:01Z",
                "physical_release": False,
            }
            body = {"device_claim_open": opened,
                    "device_claim_mode": "borrowed_outer_reservation",
                    "device_claim_borrowed_phase_end": phase_end}
            E._validate_claim_boundary(body, plan=evidence_plan)
            forged = {**body, "device_claim_released": {
                **opened, "released_at": "2026-08-14T00:00:01Z"}}
            with self.assertRaisesRegex(E.EvidenceProducerError, "physical"):
                E._validate_claim_boundary(forged, plan=evidence_plan)


if __name__ == "__main__":
    unittest.main()
