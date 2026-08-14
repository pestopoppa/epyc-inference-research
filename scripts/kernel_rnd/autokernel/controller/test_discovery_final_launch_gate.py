"""Independent, hardware-free final launch gate for static GPU discovery.

This file deliberately treats the public deployment bundle and its immutable
integration commit as external artifacts.  It may read and validate them, but
it never invokes an actor, builds source, profiles, or acquires a real device.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from .. import schemas
from . import discovery_controller as controller
from . import discovery_deployment_factory as factory
from . import test_discovery_controller_blackbox as blackbox


COMBINED_SHA = "eac3ae7c0fab14160520f10d20b4330fe5c573e6"
GRAPH_SHA = "6592b478314572d4fc508c6397a6568286b5e5efe336f9b7c265196970752740"
PRODUCTION_SHA = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
INSTRUMENT_SHA = "81bf32f11b4a421880e8f25faec3e4ba872363f0"
INSTRUMENT_DIFF_SHA = "3cf9178fcc00e8c1d3dfc0bfd6086edbff6a6eb6ac528aa4d88b23843b5599c2"
INSTRUMENT_BRANCH = "codex/autokernel-ready-continue-instrument-20260814"
PUBLIC_ROOT = Path(
    "/mnt/raid0/llm/autokernel/deployments/gpu-discovery-final-eac3ae7c-v1"
)
COMBINED_ROOT = Path(
    "/mnt/raid0/llm/autokernel/worktrees/autokernel-final-integration-20260814"
)
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
INSTRUMENT_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
EXPECTED_MODULES = frozenset({
    "cpu_region_claim", "deployment_factory", "device_claim", "device_sampler",
    "discovery_controller", "discovery_deployment", "discovery_static_registry",
    "evaluator_integrity", "gpu_discovery_beliefs", "gpu_discovery_runner",
    "gpu_load_admission", "gpu_residency_sampler", "gpu_source_adapter",
    "gpu_source_evidence", "gpu_source_proofs", "inference_window",
    "instrument_integrity", "source_candidate", "split_runtime_verifier", "worktree",
})


def _run(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv, cwd=cwd, check=True, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tree_snapshot(root: Path) -> tuple[bytes, bytes]:
    return (
        _run("git", "-C", str(root), "rev-parse", "HEAD").stdout,
        _run(
            "git", "-C", str(root), "status", "--porcelain=v1",
            "--untracked-files=no",
        ).stdout,
    )


class ImmutablePublicBundleGate(unittest.TestCase):
    maxDiff = None

    def _graph(self) -> dict:
        return json.loads((PUBLIC_ROOT / "state/deployment-graph.json").read_text())

    def test_exact_graph_and_every_execution_module_are_immediately_bound(self) -> None:
        graph = self._graph()
        self.assertEqual(graph["graph_sha256"], GRAPH_SHA)
        unhashed = dict(graph)
        unhashed.pop("graph_sha256")
        self.assertEqual(schemas.content_hash(unhashed), GRAPH_SHA)
        self.assertFalse(graph["inference_executed"])
        self.assertFalse(graph["promotion_claim"])

        self.assertEqual(set(graph["execution_modules"]), EXPECTED_MODULES)
        for name, binding in graph["execution_modules"].items():
            path = Path(binding["path"])
            metadata = path.lstat()
            self.assertTrue(stat.S_ISREG(metadata.st_mode), name)
            self.assertFalse(path.is_symlink(), name)
            self.assertEqual(_sha(path.read_bytes()), binding["sha256"], name)
            self.assertTrue(path.is_relative_to(COMBINED_ROOT), name)

        launcher = graph["actor_argv_authority"]
        launcher_path = Path(launcher["module"])
        self.assertEqual(_sha(launcher_path.read_bytes()), launcher["module_sha256"])
        self.assertEqual(
            _run("git", "-C", str(COMBINED_ROOT), "rev-parse", "HEAD").stdout.decode().strip(),
            COMBINED_SHA,
        )

    def test_instrument_authority_is_exact_diff_path_blob_and_branch_bound(self) -> None:
        graph = self._graph()
        authority = graph["source_authority"]
        self.assertEqual(authority["production_base_commit"], PRODUCTION_SHA)
        self.assertEqual(authority["instrument_commit"], INSTRUMENT_SHA)
        self.assertEqual(authority["instrument_branch"], INSTRUMENT_BRANCH)
        self.assertEqual(authority["instrument_diff_sha256"], INSTRUMENT_DIFF_SHA)

        branch = _run(
            "git", "-C", str(INSTRUMENT_ROOT), "rev-parse",
            f"refs/heads/{INSTRUMENT_BRANCH}",
        ).stdout.decode().strip()
        self.assertEqual(branch, INSTRUMENT_SHA)
        diff = _run(
            "git", "-C", str(INSTRUMENT_ROOT), "diff", "--binary",
            f"{PRODUCTION_SHA}..{INSTRUMENT_SHA}",
        ).stdout
        self.assertEqual(_sha(diff), INSTRUMENT_DIFF_SHA)

        review_path = Path(graph["instrument_review"]["path"])
        review_raw = review_path.read_bytes()
        self.assertEqual(_sha(review_raw), graph["instrument_review"]["sha256"])
        review = json.loads(review_raw)
        names = _run(
            "git", "-C", str(INSTRUMENT_ROOT), "diff", "--name-only", "-z",
            f"{PRODUCTION_SHA}..{INSTRUMENT_SHA}",
        ).stdout
        observed = sorted(part.decode() for part in names.split(b"\0") if part)
        self.assertEqual(observed, review["reviewed_diff_paths"])
        for relative, expected in review["reviewed_blobs"].items():
            blob = _run(
                "git", "-C", str(INSTRUMENT_ROOT), "show",
                f"{INSTRUMENT_SHA}:{relative}",
            ).stdout
            self.assertEqual(_sha(blob), expected, relative)

        receipt = dict(review)
        expected_receipt = receipt.pop("receipt_sha256")
        self.assertEqual(schemas.content_hash(receipt), expected_receipt)
        self.assertEqual(
            review["ready_continue_capability"]["source_sha256"],
            authority["ready_continue_contract_source_sha256"],
        )

    def test_public_initialize_and_validate_are_config_only_and_non_mutating(self) -> None:
        before = (_tree_snapshot(PRODUCTION_ROOT), _tree_snapshot(INSTRUMENT_ROOT))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(COMBINED_ROOT)
        module = "scripts.kernel_rnd.autokernel.controller.discovery_deployment_factory"
        with tempfile.TemporaryDirectory(prefix="autokernel-final-gate-") as temporary:
            fresh = Path(temporary) / "bundle"
            initialized = subprocess.run(
                (sys.executable, "-m", module, "--initialize-bundle", str(fresh)),
                cwd=COMBINED_ROOT, env=environment, check=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            init_result = json.loads(initialized.stdout)
            self.assertEqual(init_result["status"], "initialized")
            self.assertFalse(init_result["inference_executed"])
            validated = subprocess.run(
                (sys.executable, "-m", module, "--deployment",
                 str(fresh / "config/deployment.json"), "--validate-only"),
                cwd=COMBINED_ROOT, env=environment, check=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            result = json.loads(validated.stdout)
            self.assertEqual(result["status"], "validated")
            self.assertFalse(result["inference_executed"])
            self.assertTrue((fresh / "state/deployment-graph.json").is_file())
            self.assertFalse((fresh / "operations/claims/device.jsonl").exists())
            self.assertFalse((fresh / "evidence").exists())
        self.assertEqual(
            (_tree_snapshot(PRODUCTION_ROOT), _tree_snapshot(INSTRUMENT_ROOT)), before
        )

    def test_batching_is_one_process_nine_samples_with_serialized_fallback(self) -> None:
        graph = self._graph()
        batched = graph["batched_runner"]
        self.assertEqual(batched["processes_per_arm"], 1)
        self.assertEqual(batched["calls_per_arm"], 9)
        self.assertFalse(batched["early_unlock_enabled"])
        self.assertEqual(batched["safe_fallback"], "full_process_cold_serialized_lock")
        self.assertEqual(batched["trust_limit"], "cooperative_same_uid_not_launch_authority")

        reservation = graph["device_reservation"]
        self.assertEqual(
            reservation["busy_disposition"],
            "durable_pending_exact_candidate_no_iteration",
        )
        self.assertEqual(
            reservation["inner_claim_mode"],
            "borrowed_logical_phase_no_physical_release",
        )
        self.assertFalse(reservation["nested_physical_acquisitions"])
        self.assertEqual(
            reservation["physical_release_authority"],
            "adapter_terminal_reservation_release",
        )


class DeviceContentionGate(unittest.TestCase):
    def _lease_config(self, root: Path) -> SimpleNamespace:
        model = root / "model.gguf"
        model.write_bytes(b"hardware-free model identity")
        model_sha = _sha(model.read_bytes())
        profile = SimpleNamespace(
            model_path=str(model.resolve()), model_sha256=model_sha,
            device_id="mi210_0", workload="decode_tg128", calls_per_arm=9,
            cold_load_host_bytes=model.stat().st_size,
            worst_case_loads_per_interval=2,
        )
        corpus = SimpleNamespace(
            profiles=(profile,), examples=(), policy_sha256="a" * 64,
            version="gate-v1",
        )
        config = SimpleNamespace(
            device_id="mi210_0", model=SimpleNamespace(path=model.resolve(), sha256=model_sha),
            admission_policy=SimpleNamespace(corpus=corpus),
            planner_context=SimpleNamespace(value={"context_sha256": "b" * 64}),
            operations_root=(root / "operations").resolve(), claim_timeout_s=0.0,
            inference_window_lock=(root / "window.lock").resolve(),
            revalidate=mock.Mock(),
        )
        return config

    def test_busy_real_device_claim_is_wait_not_admitted_or_ambiguous(self) -> None:
        """Ordinary contention must be classified before any build/evidence intent."""
        with tempfile.TemporaryDirectory(prefix="autokernel-contention-gate-") as temporary:
            config = self._lease_config(Path(temporary))
            decision = SimpleNamespace(
                mode="cold_serialized", to_dict=lambda: {"decision_sha256": "c" * 64},
            )
            busy = factory.device_claim.DeviceClaimTimeout("held by another session")
            acquire = mock.Mock(side_effect=busy)
            with mock.patch.object(factory.gpu_load_admission, "arbitrate", return_value=decision):
                permit = factory.GpuDiscoveryLease(
                    config=config, mode="allowed_discovery_noise",
                    claim_journal=mock.Mock(), claim_acquirer=acquire,
                    claim_verifier=lambda _receipt: True,
                ).admit(
                    SimpleNamespace(
                        experiment_intent=None,
                        source_manifest=SimpleNamespace(campaign_id="ak-final-gate"),
                    ),
                    operation_key="d" * 64,
                )
        acquire.assert_called_once()
        self.assertFalse(permit["admitted"])
        self.assertEqual(permit["device_id"], "mi210_0")
        self.assertEqual(permit["reason"], "device_busy")

    def test_busy_pending_retry_spends_no_more_planning_build_or_iteration(self) -> None:
        """A durable WAIT resumes the byte-identical candidate once admission succeeds."""
        with tempfile.TemporaryDirectory(prefix="autokernel-pending-gate-") as temporary, \
                mock.patch.object(controller.source_candidate, "SourcePatchManifest",
                                  blackbox.Manifest), \
                mock.patch.object(controller, "_write_projection"):
            root = Path(temporary)
            planner, critic, screener = (
                blackbox.Planner(), blackbox.Critic(), blackbox.Screen()
            )
            lease = blackbox.Lease((False, False, True))
            config = controller.ControllerConfig(root / "out", max_iterations=1)
            first = controller.run_controller(
                config, planner=planner, critic=critic, screener=screener, lease=lease
            )
            second = controller.run_controller(
                config, planner=planner, critic=critic, screener=screener, lease=lease
            )
            self.assertEqual(first["pending"]["row"]["status"], "waiting_resource")
            self.assertEqual(second["pending"]["candidate"], first["pending"]["candidate"])
            self.assertEqual(first["next"], 1)
            self.assertEqual(second["next"], 1)
            self.assertEqual(first["iterations"], [])
            self.assertEqual(second["iterations"], [])
            self.assertEqual((planner.calls, critic.calls, screener.calls), (1, 1, 0))
            final = controller.run_controller(
                config, planner=planner, critic=critic, screener=screener, lease=lease
            )
            self.assertTrue(final["complete"])
            self.assertEqual((planner.calls, critic.calls, screener.calls), (1, 1, 1))
            self.assertEqual(
                screener.items[0].source_manifest.patch_bytes,
                blackbox.Manifest().patch_bytes,
            )

    def test_crash_recovery_reacquires_or_waits_without_replanning(self) -> None:
        """Lost in-memory reservations demote to WAIT and keep the exact candidate."""
        with tempfile.TemporaryDirectory(prefix="autokernel-crash-wait-gate-") as temporary, \
                mock.patch.object(controller.source_candidate, "SourcePatchManifest",
                                  blackbox.Manifest), \
                mock.patch.object(controller, "_write_projection"):
            root = Path(temporary)
            planner, critic = blackbox.Planner(), blackbox.Critic()
            screener = blackbox.CrashBeforeRunner()
            lease = blackbox.Lease((True, False, True))
            config = controller.ControllerConfig(root / "out", max_iterations=1)
            with self.assertRaises(blackbox.ProcessCrash):
                controller.run_controller(
                    config, planner=planner, critic=critic,
                    screener=screener, lease=lease,
                )
            waiting = controller.run_controller(
                config, planner=planner, critic=critic,
                screener=screener, lease=lease,
            )
            self.assertEqual(waiting["pending"]["row"]["status"], "waiting_resource")
            self.assertEqual(waiting["iterations"], [])
            self.assertEqual((planner.calls, critic.calls, screener.entries), (1, 1, 1))
            final = controller.run_controller(
                config, planner=planner, critic=critic,
                screener=screener, lease=lease,
            )
            self.assertTrue(final["complete"])
            self.assertEqual((planner.calls, critic.calls, screener.entries), (1, 1, 2))
            self.assertEqual(
                screener.items[0].source_manifest.patch_bytes,
                blackbox.Manifest().patch_bytes,
            )

    def test_probe_outer_and_borrowed_phases_have_one_physical_release_each(self) -> None:
        """Logical proof/runner phases never create or release another physical claim."""
        with tempfile.TemporaryDirectory(prefix="autokernel-claim-gate-") as temporary:
            config = self._lease_config(Path(temporary))
            events: list[str] = []
            claims = []

            class PhysicalClaim:
                def __init__(self, label: str, purpose: str, campaign_id: str) -> None:
                    self.label = label
                    self.held = True
                    self.release_calls = 0
                    self.opened = factory.device_claim.ClaimReceipt(
                        claim_id=f"akd-{label}", device_id="mi210_0",
                        lock_path="/hardware-free-claim", state="held",
                        holder_pid=1, holder_start_ticks=1,
                        holder_boot_id="boot", host="host", holder_label="gate",
                        purpose=purpose, campaign_id=campaign_id, acquired_at="now",
                    )

                def receipt(self):
                    return self.opened

                def release(self):
                    self.release_calls += 1
                    self.held = False
                    events.append(f"{self.label}_release")
                    return factory.replace(self.opened, released_at="done")

            def acquire(_device, *, purpose, campaign_id, **_kwargs):
                label = "probe" if "probe" in purpose else "outer"
                events.append(f"{label}_acquire")
                claim = PhysicalClaim(label, purpose, campaign_id)
                claims.append(claim)
                return claim

            lease = factory.GpuDiscoveryLease(
                config=config, mode="allowed_discovery_noise",
                claim_journal=mock.Mock(), claim_acquirer=acquire,
                claim_verifier=lambda _receipt: True,
            )
            candidate = SimpleNamespace(
                experiment_intent=None,
                source_manifest=SimpleNamespace(campaign_id="ak-final-gate"),
            )
            decision = SimpleNamespace(
                mode="cold_serialized", to_dict=lambda: {"decision_sha256": "c" * 64},
            )
            operation_key = "e" * 64
            with mock.patch.object(factory.gpu_load_admission, "arbitrate", return_value=decision):
                permit = lease.admit(candidate, operation_key=operation_key)
            self.assertTrue(permit["admitted"])
            outer = lease.reserve(operation_key)
            self.assertEqual(outer["claim_id"], "akd-outer")
            for phase in ("proof", "runner"):
                borrowed = lease.borrower(operation_key)(
                    "mi210_0", campaign_id="ak-final-gate", purpose=phase,
                )
                phase_end = borrowed.release()
                self.assertEqual(phase_end["outer_claim_id"], "akd-outer")
                self.assertFalse(phase_end["physical_release"])
                self.assertNotIn("released_at", phase_end)
                self.assertTrue(claims[1].held)
            released = lease.release(operation_key)
            self.assertEqual(released["claim_id"], "akd-outer")
            self.assertEqual(events, [
                "probe_acquire", "probe_release", "outer_acquire", "outer_release",
            ])
            self.assertEqual([claim.release_calls for claim in claims], [1, 1])


if __name__ == "__main__":
    unittest.main()
