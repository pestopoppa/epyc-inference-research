"""Acceptance gate for the v8 deterministic-correctness capability failure.

This is intentionally a test-only, red gate.  It consumes the completed v8
build as immutable evidence when that deployment is present, but never runs a
backend test: ``--autokernel-property-self-test`` returns before
``ggml_backend_load_all()`` in the reviewed instrument.
"""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from ..execution import t0_provider
from . import discovery_controller as C
from . import discovery_deployment_factory as F
from . import discovery_static_registry as R
from . import gpu_source_proofs


DEFAULT_V8_BUNDLE = Path(
    "/mnt/raid0/llm/autokernel/deployments/"
    "gpu-discovery-quant-ladder-occupancy-v8"
)
V8_BUNDLE = Path(os.environ.get(
    "AUTOKERNEL_V8_ACCEPTANCE_BUNDLE", str(DEFAULT_V8_BUNDLE))).resolve()
SELF_TEST_SUFFIX = (
    t0_provider.backend_ops_property_self_test_argv(
        "test-backend-ops", R._CORRECTNESS_CAPABILITY_SEED)[1:]
)
SELF_TEST_MARKER = (
    "AUTOKERNEL_PROPERTY_SELF_TEST "
    f"suite_seed={R._CORRECTNESS_CAPABILITY_SEED} "
    "sensitivity=1.000 specificity=1.000 planted=5 clean=5"
)
REAL_SUBPROCESS_RUN = subprocess.run


def _self_hash_holds(body: dict[str, object]) -> bool:
    expected = body.get("receipt_sha256")
    payload = {key: value for key, value in body.items()
               if key != "receipt_sha256"}
    return expected == F.schemas.content_hash(payload)


def _load_v8() -> tuple[object, C.PlannedCandidate, C.GpuSourceBuild, dict[str, object]]:
    config = F.deployment.load_deployment_config(V8_BUNDLE / "config/deployment.json")
    state = json.loads((V8_BUNDLE / "state/state.json").read_text(encoding="utf-8"))
    candidate = C._restore_pending(state["inflight"])
    terminals = tuple((V8_BUNDLE / "operations/build-cache/entries").glob(
        "*/terminal.json"))
    if len(terminals) != 1:
        raise AssertionError(f"v8 must have one completed build terminal, found {len(terminals)}")
    terminal = json.loads(terminals[0].read_text(encoding="utf-8"))
    raw = terminal["build"]
    build = C.GpuSourceBuild(
        anchor_build=Path(raw["anchor_build"]),
        candidate_build=Path(raw["candidate_build"]),
        candidate_identity=gpu_source_proofs.BuildIdentity(**raw["candidate_identity"]),
        anchor_identity=gpu_source_proofs.BuildIdentity(**raw["anchor_identity"]),
        measurement_binary=Path(raw["measurement_binary"]),
        common_loader_dir=Path(raw["common_loader_dir"]),
        anchor_loader_dir=Path(raw["anchor_loader_dir"]),
        candidate_loader_dir=Path(raw["candidate_loader_dir"]),
        reward_runtime_sha256=raw["reward_runtime_sha256"],
        operation_key=state["inflight"]["operation_key"],
        build_key=raw["build_key"],
        materialization_receipt=Path(raw["materialization_receipt"]),
        materialization_sha256=raw["materialization_sha256"],
        anchor_source_tree_receipt=Path(raw["anchor_source_tree_receipt"]),
        anchor_source_tree_sha256=raw["anchor_source_tree_sha256"],
        candidate_source_tree_receipt=Path(raw["candidate_source_tree_receipt"]),
        candidate_source_tree_sha256=raw["candidate_source_tree_sha256"],
        teardown_receipt=Path(raw["teardown_receipt"]),
        teardown_sha256=raw["teardown_sha256"],
    )
    return config, candidate, build, terminal


@unittest.skipUnless(V8_BUNDLE.is_dir(),
                     "requires the immutable completed AutoKernel v8 bundle")
class V8DeterministicCorrectnessCapabilityAcceptance(unittest.TestCase):
    """The real v8 artifacts must pass the capability boundary without a GPU."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.candidate, cls.build, cls.terminal = _load_v8()

    def test_completed_receipt_binds_both_required_tools(self):
        self.assertEqual(self.terminal["schema"],
                         "epyc.autokernel.gpu_source_build_terminal.v1")
        self.assertEqual(self.terminal["state"], "complete")
        self.assertTrue(_self_hash_holds(self.terminal))
        materialization = json.loads(
            self.build.materialization_receipt.read_text(encoding="utf-8"))
        self.assertTrue(_self_hash_holds(materialization))
        self.assertEqual(materialization["build_contract"]["required_targets"],
                         ["llama-bench", "test-backend-ops"])
        for arm in (self.build.anchor_build, self.build.candidate_build):
            tool = arm / "bin/test-backend-ops"
            self.assertTrue(tool.is_file())
            self.assertFalse(tool.is_symlink())

    def test_v8_terminal_gap_is_exact_and_not_reinterpreted_as_absence(self):
        materialization = json.loads(
            self.build.materialization_receipt.read_text(encoding="utf-8"))
        terminal_raw = json.dumps(self.terminal, sort_keys=True)
        materialization_raw = json.dumps(materialization, sort_keys=True)
        for arm in (self.build.anchor_build, self.build.candidate_build):
            digest = hashlib.sha256(
                (arm / "bin/test-backend-ops").read_bytes()).hexdigest()
            self.assertNotIn(digest, terminal_raw)
            self.assertNotIn(digest, materialization_raw)
        self.assertNotIn("correctness_capability_receipt", self.terminal["build"])
        self.assertNotIn("correctness_capability_sha256", self.terminal["build"])

    def test_real_v8_tools_have_a_hardware_free_behavioral_capability_probe(self):
        for arm in (self.build.anchor_build, self.build.candidate_build):
            tool = arm / "bin/test-backend-ops"
            environment = {
                "HIP_VISIBLE_DEVICES": "-1",
                "LD_LIBRARY_PATH": f"{tool.parent}:/opt/rocm/lib",
                "PATH": "/opt/rocm/bin:/usr/bin:/bin",
                "ROCM_PATH": "/opt/rocm",
            }
            with self.subTest(arm=arm.name):
                result = REAL_SUBPROCESS_RUN(
                    (str(tool), *SELF_TEST_SUFFIX), check=False,
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    env=environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr.strip(), SELF_TEST_MARKER)

                # This is the v8 false-negative trigger: the reviewed binary
                # has no --help option.  Unknown arguments intentionally emit
                # usage and return 1 even though the capability is present.
                help_result = REAL_SUBPROCESS_RUN(
                    (str(tool), "--help"), check=False,
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    env=environment)
                self.assertEqual(help_result.returncode, 1)
                self.assertIn("--suite-seed <u64>", help_result.stdout)

    def _attest(self, arm: str, build: Path, *, runner=REAL_SUBPROCESS_RUN):
        calls = []

        def no_device_runner(argv, **kwargs):
            calls.append((tuple(argv), dict(kwargs)))
            safe = dict(kwargs)
            safe["env"] = {**safe["env"], "HIP_VISIBLE_DEVICES": "-1"}
            return runner(argv, **safe)

        record = R._attest_correctness_capability(
            build, arm=arm, runner=no_device_runner)
        return record, calls

    def test_static_builder_attestor_binds_real_v8_tool_identity_and_result(self):
        """The builder must run and bind this before sealing a complete terminal."""
        for label, arm in (("anchor", self.build.anchor_build),
                           ("candidate", self.build.candidate_build)):
            tool = arm / "bin/test-backend-ops"
            with self.subTest(arm=label):
                record, calls = self._attest(label, arm)
                self.assertEqual(record["schema"],
                                 R._CORRECTNESS_CAPABILITY_SCHEMA)
                self.assertEqual(record["arm"], label)
                self.assertEqual(record["binary"]["path"],
                                 str(tool.resolve(strict=True)))
                self.assertEqual(record["binary"]["sha256"],
                                 hashlib.sha256(tool.read_bytes()).hexdigest())
                self.assertEqual(tuple(record["argv"]),
                                 (str(tool.resolve(strict=True)), *SELF_TEST_SUFFIX))
                self.assertEqual(record["exit_code"], 0)
                self.assertEqual(record["stdout_sha256"], hashlib.sha256(b"").hexdigest())
                self.assertEqual(record["stderr"], SELF_TEST_MARKER + "\n")
                self.assertEqual(record["result"], {
                    "suite_seed": R._CORRECTNESS_CAPABILITY_SEED,
                    "sensitivity": 1.0, "specificity": 1.0,
                    "planted": 5, "clean": 5})
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], tuple(record["argv"]))

    def test_usage_rc1_wrong_seed_and_malformed_output_cannot_grant_capability(self):
        usage = b"Usage: test-backend-ops [--suite-seed <u64>]\n"
        cases = {
            "rc1-usage": subprocess.CompletedProcess(
                args=(), returncode=1, stdout=usage, stderr=b""),
            "rc0-usage": subprocess.CompletedProcess(
                args=(), returncode=0, stdout=usage, stderr=b""),
            "wrong-seed": subprocess.CompletedProcess(
                args=(), returncode=0, stdout=b"",
                stderr=(b"AUTOKERNEL_PROPERTY_SELF_TEST suite_seed=99 "
                        b"sensitivity=1.000 specificity=1.000 planted=5 clean=5\n")),
            "malformed": subprocess.CompletedProcess(
                args=(), returncode=0, stdout=b"",
                stderr=(SELF_TEST_MARKER + " trailing\n").encode()),
        }
        for label, completed in cases.items():
            with self.subTest(case=label), self.assertRaisesRegex(
                    R.StaticRegistryError, "correctness capability"):
                R._attest_correctness_capability(
                    self.build.candidate_build, arm="candidate",
                    runner=lambda *_args, completed=completed, **_kwargs: completed)

    def _capability_bound_build(self, root: Path) -> C.GpuSourceBuild:
        values = {}
        for arm, build in (("anchor", self.build.anchor_build),
                           ("candidate", self.build.candidate_build)):
            capability, _calls = self._attest(arm, build)
            receipt, receipt_sha = R._sealed_write(
                root / f"{arm}-correctness-capability.json", capability)
            binary = build / "bin/test-backend-ops"
            values.update({
                f"{arm}_correctness_binary": binary,
                f"{arm}_correctness_binary_sha256": hashlib.sha256(
                    binary.read_bytes()).hexdigest(),
                f"{arm}_correctness_capability_receipt": receipt,
                f"{arm}_correctness_capability_sha256": receipt_sha,
            })
        return replace(self.build, **values)

    def test_build_projection_requires_preterminal_capability_receipt(self):
        required = {
            f"{arm}_{suffix}"
            for arm in ("anchor", "candidate")
            for suffix in ("correctness_binary", "correctness_binary_sha256",
                           "correctness_capability_receipt",
                           "correctness_capability_sha256")}
        self.assertTrue(required.issubset(
            {field.name for field in fields(C.GpuSourceBuild)}))
        with self.assertRaisesRegex(
                R.StaticRegistryError, "projection is incomplete"):
            R.StaticGpuSourceBuilder._build_projection(self.build)
        with tempfile.TemporaryDirectory() as temporary:
            bound = self._capability_bound_build(Path(temporary).resolve())
            projection = R.StaticGpuSourceBuilder._build_projection(bound)
            for name in required:
                value = getattr(bound, name)
                self.assertEqual(projection[name],
                                 str(value) if isinstance(value, Path) else value)

    def test_evidence_reopens_exact_candidate_binary_and_capability_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            bound = self._capability_bound_build(Path(temporary).resolve())
            binary, receipt = R.correctness_capability_files_for_build(
                bound, arm="candidate")
        self.assertEqual(binary.role, "executable")
        self.assertEqual(binary.path, bound.candidate_correctness_binary)
        self.assertEqual(binary.sha256,
                         bound.candidate_correctness_binary_sha256)
        self.assertEqual(receipt.role, "instrument_capability")
        self.assertEqual(receipt.sha256,
                         bound.candidate_correctness_capability_sha256)

    def test_validate_only_receipt_preloads_exact_suite_contract_before_build(self):
        # The instrument source hash is available during validate-only, long
        # before a candidate can trigger two 17-minute builds.  Receipt it as
        # the prebuild authority; the behavioral binary self-test above is the
        # postbuild confirmation that the target actually compiled it.
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(self.config,
                             state_root=(Path(temporary).resolve() / "state"))
            path, digest = F._instrument_review_receipt(config)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            receipt = json.loads(path.read_text(encoding="utf-8"))
        capability = receipt["backend_ops_property_capability"]
        self.assertEqual(capability["source"], "tests/test-backend-ops.cpp")
        self.assertEqual(capability["source_sha256"],
                         F._INSTRUMENT_TEST_SOURCE_SHA256)
        self.assertEqual(capability["suite_seed"], R._CORRECTNESS_CAPABILITY_SEED)
        self.assertEqual(tuple(capability["argv_suffix"]), SELF_TEST_SUFFIX)
        self.assertEqual(capability["expected_stderr"], SELF_TEST_MARKER + "\n")


if __name__ == "__main__":
    unittest.main()
