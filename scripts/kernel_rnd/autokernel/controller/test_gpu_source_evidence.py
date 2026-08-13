import hashlib
import json
from dataclasses import replace
from pathlib import Path
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
    source = write_bound(
        root, f"{label}-source.json",
        json.dumps({"source_commit": f"commit-{label}"}, sort_keys=True).encode(),
        "source_identity")
    binary = write_bound(root, f"{label}-binary", f"binary-{label}".encode(), "binary")
    hip = write_bound(root, f"{label}-hip.so", f"hip-{label}".encode(), "hip_library")
    config = write_bound(root, f"{label}-config.json", f"config-{label}".encode(), "config")
    linkage = write_bound(root, f"{label}-linkage.json", f"linkage-{label}".encode(), "linkage")
    return (P.BuildIdentity(
        source_commit=f"commit-{label}", source_sha256=source.sha256,
        binary_sha256=binary.sha256, hip_library_sha256=hip.sha256,
        config_sha256=config.sha256, linkage_sha256=linkage.sha256),
        E.BuildIdentityFiles(source, binary, hip, config, linkage))


def plan(root: Path) -> E.GpuSourceEvidencePlan:
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    candidate, candidate_files = build_files(root, "candidate")
    anchor, anchor_files = build_files(root, "anchor")
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
    materialization = write_bound(
        root, "materialization.json",
        json.dumps(materialization_body, sort_keys=True).encode(), "materialization")
    correctness_tool = write_bound(root, "test-backend-ops", b"correctness tool", "executable")
    profiler = write_bound(root, "rocprof", b"profiler tool", "executable")
    correctness_tool.path.chmod(0o700)
    profiler.path.chmod(0o700)
    timestamp_input = write_bound(root, "timestamps.xml", b"timestamp policy", "timestamp_input")
    placeholder_policy = E.BoundInputFile("execution_policy", (root / "policy.json").resolve(), "0" * 64)
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
        correctness_summary_pattern=r"(?P<passed>\d+)/(?P<total>\d+) tests passed",
        expected_correctness_cases=3,
        candidate_rocprof_argv=(str(profiler.path), "-i", str(timestamp_input.path),
                                "--candidate", str(candidate_files.binary.path),
                                "--model", str(model.path), "--workload", str(workload.path),
                                "--config", str(runtime.path)),
        anchor_rocprof_argv=(str(profiler.path), "-i", str(timestamp_input.path),
                             "--anchor", str(anchor_files.binary.path),
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
            materialization),
        policy=placeholder_policy,
        correctness_inputs=(correctness_tool, candidate_files.binary),
        candidate_rocprof_inputs=(profiler, timestamp_input, candidate_files.binary),
        anchor_rocprof_inputs=(profiler, timestamp_input, anchor_files.binary),
        required_correctness_argv_paths=(candidate_files.binary.path, model.path,
                                         workload.path, runtime.path),
        required_candidate_rocprof_argv_paths=(candidate_files.binary.path, model.path,
                                               workload.path, runtime.path),
        required_anchor_rocprof_argv_paths=(anchor_files.binary.path, model.path,
                                            workload.path, runtime.path),
        execution_cwd=root,
        execution_environment=(("HIP_VISIBLE_DEVICES", "0"),
                               ("LD_LIBRARY_PATH", str(candidate_files.hip_library.path.parent))))
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


class FakeExecutors:
    def __init__(self, *, correctness_exit=0, correctness_summary="3/3 tests passed",
                 non_overlap=False, inverse_bad=False, invariant_changed=False,
                 forbidden=False, rocprof_exit=0):
        self.calls = []
        self.correctness_exit = correctness_exit
        self.correctness_summary = correctness_summary
        self.non_overlap = non_overlap
        self.inverse_bad = inverse_bad
        self.invariant_changed = invariant_changed
        self.forbidden = forbidden
        self.rocprof_exit = rocprof_exit

    def correctness(self, invocation):
        self.calls.append((invocation.kind, invocation.arm, invocation.argv))
        invocation.stdout_path.write_text(self.correctness_summary + "\n")
        invocation.stderr_path.write_text("")
        return self._capture(invocation, self.correctness_exit)

    def rocprof(self, invocation):
        self.calls.append((invocation.kind, invocation.arm, invocation.argv))
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
                kfd_pids=(1234,), vram_bytes=4096),))


class GpuSourceEvidenceTests(unittest.TestCase):
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
            (FakeExecutors(correctness_summary="2/3 tests passed"),
             "exact expected case count"),
            (FakeExecutors(rocprof_exit=9), "rocprof command exited nonzero"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                claims = ClaimFactory()
                with self.assertRaisesRegex(E.EvidenceProducerError, message):
                    self.produce(directory, executors=executor, claims=claims)
                self.assertTrue(all(claim.released for claim in claims.claims))
                self.assertFalse((Path(directory) / "evidence/proof-bundle.json").exists())

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

    def test_unapplied_manifest_wrong_environment_and_cwd_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            materialization = current.identity_files.materialization.path
            raw = json.loads(materialization.read_text())
            raw["patch_applied"] = False
            materialization.write_text(json.dumps(raw, sort_keys=True))
            changed = replace(
                current.identity_files.materialization,
                sha256=hashlib.sha256(materialization.read_bytes()).hexdigest())
            current = replace(current, identity_files=replace(
                current.identity_files, materialization=changed))
            policy = json.loads(current.policy.path.read_text())
            policy["execution_environment"] = [list(x) for x in current.execution_environment]
            current.policy.path.write_text(json.dumps(policy, sort_keys=True))
            current = replace(current, policy=replace(
                current.policy,
                sha256=hashlib.sha256(current.policy.path.read_bytes()).hexdigest()))
            with self.assertRaisesRegex(E.EvidenceProducerError, "manifest-applied"):
                self.produce(directory, plan_=current)
        with tempfile.TemporaryDirectory() as directory:
            current = plan(Path(directory) / "inputs")
            with self.assertRaisesRegex(E.EvidenceProducerError, "LD_LIBRARY_PATH"):
                replace(current, execution_environment=(
                    ("LD_LIBRARY_PATH", "/wrong/generation"),))
            bad_cwd = Path(directory) / "missing"
            with self.assertRaisesRegex(E.EvidenceProducerError, "cwd"):
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


if __name__ == "__main__":
    unittest.main()
