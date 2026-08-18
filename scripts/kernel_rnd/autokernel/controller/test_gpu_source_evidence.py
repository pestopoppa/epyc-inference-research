import hashlib
import json
from dataclasses import replace
from pathlib import Path
import re
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
    result = E.GpuSourceEvidencePlan(
        campaign_id="ak-gpu-source-evidence-test",
        device_id="mi210_0",
        manifest_sha256=manifest.sha256, model_sha256=model.sha256,
        workload_sha256=workload.sha256, runtime_config_sha256=runtime.sha256,
        candidate=candidate, anchor=anchor,
        correctness_argv=(str(correctness_tool.path), "--op", "MUL_MAT_ID",
                          "--binary", str(candidate_files.binary.path),
                          "--model", str(model.path), "--workload", str(workload.path),
                          "--config", str(runtime.path)),
        correctness_backend="ROCm0",
        correctness_op="MUL_MAT_ID",
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
        shared_runtime=shared)
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
                        op: str = "MUL_MAT_ID") -> str:
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
                 forbidden=False, rocprof_exit=0, runtime_maps=None):
        self.calls = []
        self.correctness_exit = correctness_exit
        self.correctness_summary = correctness_summary
        self.non_overlap = non_overlap
        self.inverse_bad = inverse_bad
        self.invariant_changed = invariant_changed
        self.forbidden = forbidden
        self.rocprof_exit = rocprof_exit
        self.runtime_maps = runtime_maps or {}

    def correctness(self, invocation):
        self.calls.append((invocation.kind, invocation.arm, invocation.argv,
                           invocation.environment))
        invocation.stdout_path.write_text(correctness_console(self.correctness_summary))
        invocation.stderr_path.write_text("")
        return self._capture(invocation, self.correctness_exit)

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
                verifier=lambda _receipt: True):
        executors = executors or FakeExecutors()
        claims = claims or ClaimFactory()
        bundle = E.produce_gpu_source_evidence(
            output_root=Path(directory) / "evidence",
            plan=plan_ or plan(Path(directory) / "inputs"),
            correctness_executor=executors.correctness,
            rocprof_executor=executors.rocprof,
            claim_journal=object(), claim_acquirer=claims,
            claim_verifier=verifier, claim_timeout_s=0)
        return bundle, executors, claims

    def test_happy_path_binds_exact_commands_claims_files_and_inverse(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, executors, claims = self.produce(directory)
            self.assertEqual(
                [row[:2] for row in executors.calls],
                [("correctness", "candidate"), ("rocprof", "candidate"),
                 ("rocprof", "anchor")])
            self.assertTrue(all(claim.released for claim in claims.claims))
            self.assertEqual(len(claims.claims), 3)
            self.assertNotEqual(executors.calls[1][3], executors.calls[2][3])
            self.assertEqual(
                Path(dict(executors.calls[1][3])["LD_LIBRARY_PATH"]).name,
                "candidate")
            self.assertEqual(
                Path(dict(executors.calls[2][3])["LD_LIBRARY_PATH"]).name,
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
            self.assertEqual(executors.calls[1][2], executors.calls[2][2])
            self.assertIn(str(current.shared_runtime.measurement_binary.path), executors.calls[1][2])
            self.assertNotEqual(dict(executors.calls[1][3])["LD_LIBRARY_PATH"],
                                dict(executors.calls[2][3])["LD_LIBRARY_PATH"])
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
            candidate_argv, anchor_argv = executors.calls[1][2], executors.calls[2][2]
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
