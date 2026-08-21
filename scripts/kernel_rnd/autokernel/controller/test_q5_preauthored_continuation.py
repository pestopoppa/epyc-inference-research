"""Hardware-free acceptance tests for the sealed Q5 one-wave continuation."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from .. import hypothesis_portfolio
from .. import preauthored_continuation as P
from .. import source_candidate as S
from ..execution import worktree as W
from . import discovery_controller as D
from . import discovery_deployment_factory as F
from . import gpu_source_evidence as E
from .test_gpu_source_evidence import ClaimFactory, FakeExecutors, plan


Q5_HYPOTHESIS = "akh-v2-q5-onewave-preauthored"
Q5_COMMIT = "eb26918fa82f8aef3ab72f1e3263bd8fecde62e7"
Q5_PARENT = "e1cbca9fcbc0ed81164c5532b94cd106a83d7368"
Q5_TREE = "a723f77d3666987318f017228a993d610f2b44b1"
Q5_PARENT_TREE = "1a3c38a26f4d569b48679d16633078bd36900be5"
Q5_PATCH = "f4cc49cd11cdfd93a2d5d2e00e653f503b6a16ce675bfb12c034fbbfae3e7a77"
Q5_BINARY = "e6540dc80ae41f28cd2791e13e65d12aa9ebba83f63aeba8b48adcc540aec378"
Q5_CARRIER = "819d16c0903d71649c4674080d2718159d12ea1769e1f4f943d04dc7e2974889"
Q5_SYMBOLS = tuple(sorted((
    "mmvq_parameter_table_id", "get_device_table_id",
    "calc_nwarps", "calc_rows_per_block",
)))
Q5_EXPECTED_ROUTES = (
    ("cuda-mmvq-q5-onewave-continuation-v1.anchor.0", 6063, 57344, 128, 1024),
    ("cuda-mmvq-q5-onewave-continuation-v1.anchor.1", 4644, 8192, 128, 1024),
    ("cuda-mmvq-q5-onewave-continuation-v1.anchor.2", 3096, 311296, 128, 1024),
)
Q5_EXCLUDED_ROUTE = (
    "cuda-mmvq-q5-onewave-continuation-v1.anchor.3", 129, 57344, 128, 512)
TARGETED_RECEIPT = "94f6dad89c7f8cdf8beec43c73783c57f5dcae1a9c4c38f91006dcfd6e385747"
FULL_RECEIPT = "8b7dc07643f79255193978417d3d2705506ab0793b2b89473259e5912bcadccb"


class NeverPlanner:
    def __init__(self):
        self.calls = 0

    def attest(self):
        return {**D.SOL, "runtime": {
            "kind": "docker_workspace_bind_only",
            "docker_path": "/sealed/docker", "docker_sha256": "1" * 64,
            "image_id": "sealed-image", "codex_native_sha256": "2" * 64,
            "code_mode_host_sha256": "3" * 64,
            "ca_certificate_sha256": "4" * 64,
            "writable_host_binds": ["/workspace"],
            "host_network_mode": "docker_bridge",
        }}

    def plan(self, **_kwargs):
        self.calls += 1
        raise AssertionError("preauthored continuation invoked planner")


class NeverCritic:
    def __init__(self):
        self.calls = 0

    def attest(self):
        return {**D.FABLE5_CRITIC, "runtime": {
            "kind": "claude_cli_structured_critic",
            "provider": "claude", "model": "claude-fable-5",
            "effort": "high", "wrapper_path": "/sealed/critic",
            "wrapper_sha256": "5" * 64,
            "argv_policy_sha256": "6" * 64,
            "auth_staging_policy":
                D.claude_fable5_critic_actor.AUTH_STAGING_POLICY,
        }}

    def review(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("preauthored continuation invoked critic")


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _write_carrier(root: Path, value: dict) -> Path:
    path = root / "carrier.json"
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    path.chmod(0o600)
    return path


class Q5PreauthoredContinuationTests(unittest.TestCase):
    @staticmethod
    def _scope(patch: str, *, source_backed: bool = True) -> tuple[str, ...]:
        return tuple(row[2] for row in S._hunk_rows(
            patch, source_backed_declarations=source_backed))

    @staticmethod
    def _preauthored_config(
            root: Path, *, dry_run: bool = True) -> D.ControllerConfig:
        carrier = P.load(P.DEFAULT_CARRIER)
        portfolio = hypothesis_portfolio.load(
            hypothesis_portfolio.DEFAULT_PORTFOLIO)
        templates = F._template_registry()
        context = {
            "preauthored_continuation_sha256": carrier.sha256,
            "preauthored_source_backed_diff_sha256":
                carrier.source_backed_diff_sha256,
            "template_symbol_authority": {
                template_id: {
                    path: sorted(symbols)
                    for path, symbols in template.allowed_symbols.items()}
                for template_id, template in sorted(
                    templates.templates.items())},
            "portfolio_dispatch_authority":
                F._portfolio_dispatch_authority(templates, portfolio),
        }
        carry_forward = None
        carry_forward_sha256 = None
        if not dry_run:
            digest = lambda label: hashlib.sha256(label.encode()).hexdigest()
            erratum = D._q5_lds0_attribution_erratum()
            carry_forward = {
                "schema": "epyc.autokernel.discovery_carry_forward.v2",
                "predecessor_state_file_sha256": digest("predecessor-state"),
                "predecessor_journal_file_sha256": digest("predecessor-journal"),
                "predecessor_state_semantic_sha256": digest("predecessor-semantic"),
                "portfolio_outcomes": {
                    "akh-v2-q5-type-specific-dequant": "nominated",
                    "akh-v2-q8-quantizer-new-mechanism": "retire",
                    "akh-v2-fa-gqa7-pair-tail": "bounded_authoring_skip",
                    "akh-v2-rms-direct-load-reduction": "bounded_authoring_skip",
                },
                "candidate_semantic_sha256": sorted({
                    *(digest(f"semantic-{index}") for index in range(12)),
                    erratum["candidate_semantic_sha256"]}),
                "candidate_patch_sha256": sorted({
                    *(digest(f"patch-{index}") for index in range(7)),
                    erratum["candidate_patch_sha256"]}),
                "cross_campaign_candidate_sha256": sorted({
                    *(digest(f"cross-{index}") for index in range(7)),
                    erratum["cross_campaign_candidate_sha256"]}),
                "attribution_expectation_erratum": erratum,
            }
            carry_forward_sha256 = _canonical_hash(carry_forward)
            carry_forward["carry_forward_sha256"] = carry_forward_sha256
        return D.ControllerConfig(
            output_root=root, max_iterations=10, dry_run=dry_run,
            planner_context=context,
            planner_context_sha256=_canonical_hash(context),
            production_base_commit=F.deployment.FROZEN_PRODUCTION_HEAD,
            instrument_commit=carrier.compatibility_bridge[
                "current_instrument_commit"],
            experiment_template_registry_sha256=(
                templates.registry_sha256 if not dry_run else None),
            admission_corpus_sha256=("a" * 64 if not dry_run else None),
            admission_corpus_version=("q5-test-v1" if not dry_run else None),
            deployment_identity_sha256=("b" * 64 if not dry_run else None),
            hypothesis_portfolio=portfolio,
            hypothesis_portfolio_sha256=portfolio.sha256,
            carry_forward=carry_forward,
            carry_forward_sha256=carry_forward_sha256,
            preauthored_continuations={carrier.hypothesis_id: carrier},
        )

    @staticmethod
    def _q5_binding(config: D.ControllerConfig) -> dict:
        record = next(
            row for row in config.hypothesis_portfolio.hypotheses
            if row["hypothesis_id"] == Q5_HYPOTHESIS)
        return D._portfolio_binding(config, record)

    def test_default_carrier_binds_exact_historical_candidate_and_routes(self):
        carrier = P.load(P.DEFAULT_CARRIER)
        self.assertEqual(carrier.hypothesis_id, Q5_HYPOTHESIS)
        self.assertEqual(carrier.source_tree, "llama.cpp")
        self.assertEqual(carrier.source_file, "ggml/src/ggml-cuda/mmvq.cu")
        self.assertEqual(carrier.historical_commit, Q5_COMMIT)
        self.assertEqual(carrier.historical_parent_commit, Q5_PARENT)
        self.assertEqual(carrier.historical_tree, Q5_TREE)
        self.assertEqual(carrier.historical_parent_tree, Q5_PARENT_TREE)
        self.assertEqual(carrier.patch_sha256, Q5_PATCH)
        self.assertEqual(hashlib.sha256(carrier.patch_bytes).hexdigest(), Q5_PATCH)
        self.assertEqual(
            carrier.source_backed_diff_sha256,
            "2adf93c7af423debf39307a3e4d6fa675d5061c565f36682d0b22295df4339c9")
        self.assertEqual(
            hashlib.sha256(carrier.source_backed_diff.encode()).hexdigest(),
            carrier.source_backed_diff_sha256)
        self.assertEqual(tuple(carrier.declared_symbols), Q5_SYMBOLS)
        self.assertEqual(carrier.template_id,
                         "cuda-mmvq-q5-onewave-continuation-v1")
        self.assertEqual(carrier.correctness_id, "backend-ops-hip-v1")
        self.assertEqual(
            tuple((row["route_id"], row["calls"], row["grid"],
                   row["workgroup"], row["lds_bytes"])
                  for row in carrier.expected_dispatch), Q5_EXPECTED_ROUTES)
        self.assertEqual(
            tuple((row["route_id"], row["calls"], row["grid"],
                   row["workgroup"], row["lds_bytes"])
                  for row in carrier.excluded_dispatch), (Q5_EXCLUDED_ROUTE,))
        self.assertEqual(
            {row["file_sha256"] for row in carrier.historical_receipts},
            {TARGETED_RECEIPT, FULL_RECEIPT})
        self.assertEqual(
            {row["binary_sha256"] for row in carrier.historical_receipts},
            {Q5_BINARY})
        self.assertEqual(carrier.correctness_policy, {
            "historical_receipts_authority": "provenance_only",
            "modern_governed_correctness":
                "required_after_current_instrument_build",
            "bridge_waives_current_correctness": False,
            "scientific_boundary": "dispatch_attribution",
        })

    def test_q5_lds0_erratum_carrier_is_exact_and_file_authoritative(self):
        erratum = D._q5_lds0_attribution_erratum()
        self.assertEqual(
            hashlib.sha256(D._Q5_LDS0_ERRATUM_CARRIER.read_bytes()).hexdigest(),
            D._Q5_LDS0_ERRATUM_FILE_SHA256)
        self.assertEqual(erratum["erratum_sha256"],
                         "a0eab4fee2cb7450a590f161b359d479ecbab49bf3ee7686bb205b67bffb2ebd")
        self.assertEqual(set(erratum["corrected_candidate_lds_bytes"].values()),
                         {0})
        self.assertEqual(
            set(erratum["stale_candidate_lds_bytes"].values()), {256, 512})
        self.assertFalse(erratum["scientific_budget_spent"])
        self.assertFalse(erratum["do_not_repeat"])
        self.assertTrue(erratum["replay_authorized"])

    def test_q5_lds0_erratum_transport_and_coherent_rewrites_refuse(self):
        canonical = D._Q5_LDS0_ERRATUM_CARRIER.read_bytes()
        original = json.loads(canonical)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mutations: list[tuple[str, bytes]] = []
            for key in ("operation_key", "correctness_receipt_sha256",
                        "candidate_semantic_sha256", "compiler_metadata_proof"):
                body = copy.deepcopy(original)
                if key == "compiler_metadata_proof":
                    body[key]["rows"][0][
                        "candidate_group_segment_fixed_size"] = 512
                else:
                    body[key] = "0" * 64
                body["erratum_sha256"] = _canonical_hash({
                    item: value for item, value in body.items()
                    if item != "erratum_sha256"})
                mutations.append((key, (json.dumps(
                    body, sort_keys=True, separators=(",", ":")) + "\n").encode()))
            mutations.extend((
                ("duplicate", canonical.replace(
                    b'{"anchor_hip_library_sha256":',
                    b'{"anchor_hip_library_sha256":"0",'
                    b'"anchor_hip_library_sha256":', 1)),
                ("noncanonical", json.dumps(original, indent=2).encode() + b"\n"),
                ("nonfinite", canonical.replace(
                    b'"candidate_group_segment_fixed_size":0',
                    b'"candidate_group_segment_fixed_size":NaN', 1)),
            ))
            for label, raw in mutations:
                with self.subTest(label=label):
                    path = root / f"{label}.json"
                    path.write_bytes(raw)
                    path.chmod(0o600)
                    with self.assertRaises(D.DiscoveryControllerError):
                        D._q5_lds0_attribution_erratum(path)

            target = root / "target.json"
            target.write_bytes(canonical)
            target.chmod(0o600)
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(D.DiscoveryControllerError):
                D._q5_lds0_attribution_erratum(link)
            hardlink = root / "hardlink.json"
            os.link(target, hardlink)
            with self.assertRaises(D.DiscoveryControllerError):
                D._q5_lds0_attribution_erratum(target)
            hardlink.unlink()
            target.chmod(0o622)
            with self.assertRaises(D.DiscoveryControllerError):
                D._q5_lds0_attribution_erratum(target)

    def test_controller_reconstructs_without_actor_and_resume_rederives_exact_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._preauthored_config(Path(directory) / "state")
            binding = self._q5_binding(config)
            with mock.patch.object(
                    D.source_candidate,
                    "source_backed_source_patch_manifest",
                    wraps=D.source_candidate.source_backed_source_patch_manifest) as build:
                item = D._preauthored_candidate(config, binding, 1)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(item.hypothesis_id, Q5_HYPOTHESIS)
            self.assertEqual(item.source_manifest.patch_sha256, Q5_PATCH)
            self.assertEqual(
                tuple(item.source_manifest.declared_symbols[
                    P.load(P.DEFAULT_CARRIER).source_file]), Q5_SYMBOLS)
            self.assertEqual(item.proposal["preauthored_continuation_sha256"],
                             Q5_CARRIER)

            pending = {
                "phase": "preauthored_ready",
                "row": {"turn": 1, "authoring_turn": 1,
                        "portfolio_binding": binding},
                "candidate": D._pending_item(item),
                "preauthored_continuation":
                    D._preauthored_checkpoint_authority(config, item),
            }
            restored = D._restore_pending(pending, config)
            self.assertEqual(D._pending_item(restored), pending["candidate"])
            self.assertEqual(
                D._candidate_semantic_identity(restored),
                pending["preauthored_continuation"][
                    "candidate_semantic_sha256"])

            mutations = []
            changed_manifest = copy.deepcopy(pending)
            changed_manifest["candidate"]["manifest"]["patch_base64"] = (
                base64.b64encode(b"different\n").decode("ascii"))
            mutations.append(changed_manifest)
            changed_authority = copy.deepcopy(pending)
            changed_authority["preauthored_continuation"][
                "source_manifest_sha256"] = "0" * 64
            mutations.append(changed_authority)
            changed_binding = copy.deepcopy(pending)
            changed_binding["row"]["portfolio_binding"]["mechanism_id"] = (
                "0" * 64)
            mutations.append(changed_binding)
            for mutated in mutations:
                with self.subTest(mutation=mutations.index(mutated)), \
                        self.assertRaises(D.DiscoveryControllerError):
                    D._restore_pending(mutated, config)

    def test_controller_checkpoints_and_resumes_q5_without_planner_critic_or_source_actor(self):
        class StopAfterCheckpoint(RuntimeError):
            pass

        class NeverScreen:
            def screen(self, *_args):
                raise AssertionError("dry-run continuation invoked source/GPU screen")

        class NeverLease:
            def admit(self, *_args, **_kwargs):
                raise AssertionError("dry-run continuation acquired resource")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state"
            config = self._preauthored_config(root)
            planner, critic = NeverPlanner(), NeverCritic()
            native_save = D.DurableState.save

            def stop_after_checkpoint(store, state, phase):
                native_save(store, state, phase)
                if phase == "preauthored_checkpointed":
                    raise StopAfterCheckpoint

            with mock.patch.object(
                    D.DurableState, "save", new=stop_after_checkpoint), \
                    self.assertRaises(StopAfterCheckpoint):
                D.run_controller(
                    config, planner=planner, critic=critic,
                    screener=NeverScreen(), lease=NeverLease())
            checkpoint = D.DurableState(root).load()
            self.assertEqual(checkpoint["pending"]["phase"],
                             "preauthored_ready")
            self.assertNotIn("planning", checkpoint)
            self.assertEqual((planner.calls, critic.calls), (0, 0))
            self.assertEqual(checkpoint["scientific_attempts"], 0)

            def stop_after_dry_run(store, state, phase):
                native_save(store, state, phase)
                if phase == "dry_run_authorized":
                    raise StopAfterCheckpoint

            with mock.patch.object(
                    D.DurableState, "save", new=stop_after_dry_run), \
                    self.assertRaises(StopAfterCheckpoint):
                D.run_controller(
                    config, planner=planner, critic=critic,
                    screener=NeverScreen(), lease=NeverLease())
            completed = D.DurableState(root).load()
            self.assertEqual((planner.calls, critic.calls), (0, 0))
            self.assertEqual(completed["iterations"][0]["status"],
                             "dry_run_authorized")
            self.assertEqual(completed["scientific_attempts"], 0)

    def test_q5_replication_reuses_original_authoring_turn_and_exact_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._preauthored_config(Path(directory) / "state")
            binding = self._q5_binding(config)
            item = D._preauthored_candidate(config, binding, 1)
            authority = D._preauthored_checkpoint_authority(config, item)
            row = {
                "turn": 1, "authoring_turn": 1,
                "portfolio_binding": binding,
                "preauthored_continuation": authority,
                "candidate_semantic_sha256":
                    D._candidate_semantic_identity(item),
            }
            state = {"iterations": [], "next": 2}
            result = D.SealedScreen(
                "receipt", "1" * 64, .01, "candidate", "2" * 64,
                "3" * 64, "4" * 64,
                exact_attribution_effect_fraction=.01,
                target_runtime_effect_fraction=.01,
                stages=("materialized", "built", "correctness", "attribution",
                        "measurement_graphs_off_screen",
                        "target_runtime_graphs_on_screen"))
            authorization = SimpleNamespace(
                to_dict=lambda: {"sealed": "parent-authorization"})
            D._schedule_replication(
                state, item=item, authorization=authorization, row=row,
                result=result, max_iterations=10)
            pending = state["pending"]
            self.assertEqual(pending["row"]["turn"], 2)
            self.assertEqual(
                pending["preauthored_continuation"]["authoring_turn"], 1)
            restored = D._restore_pending(pending, config)
            self.assertEqual(D._pending_item(restored), pending["candidate"])
            self.assertEqual(restored.source_manifest.proposal_id,
                             "akp-discovery-1")

    def test_q5_wait_and_ambiguity_preserve_imported_provenance_without_actors(self):
        class WaitingLease:
            def __init__(self):
                self.calls = 0
            def admit(self, _item, *, operation_key):
                self.calls += 1
                return {"admitted": False, "reason": "device busy",
                        "operation_key": operation_key}

        class FreshLease:
            def __init__(self):
                self.calls = 0
            def admit(self, _item, *, operation_key):
                self.calls += 1
                return {"admitted": True, "mode": "fresh_test_claim",
                        "operation_key": operation_key}

        class AmbiguousScreen:
            def __init__(self):
                self.calls = 0
            def screen(self, _item, _authorization, permit):
                self.calls += 1
                raise D.ScreenInfrastructureAmbiguity(
                    "sealed runner epoch interrupted",
                    receipt_path="sealed-ambiguity.json",
                    receipt_sha256="7" * 64,
                    operation_key=permit["operation_key"])

        for mode in ("wait", "ambiguity"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "state"
                config = self._preauthored_config(root, dry_run=False)
                planner, critic = NeverPlanner(), NeverCritic()
                if mode == "wait":
                    lease, screen = WaitingLease(), SimpleNamespace(
                        screen=lambda *_args: self.fail(
                            "waiting lease reached screen"))
                else:
                    lease, screen = FreshLease(), AmbiguousScreen()
                state = D.run_controller(
                    config, planner=planner, critic=critic,
                    screener=screen, lease=lease)
                self.assertEqual((planner.calls, critic.calls), (0, 0))
                self.assertEqual(lease.calls, 1)
                self.assertEqual(state["scientific_attempts"], 0)
                pending = state["pending"]
                self.assertEqual(
                    pending["preauthored_continuation"]["carrier_sha256"],
                    Q5_CARRIER)
                restored = D._restore_pending(pending, config)
                self.assertEqual(restored.hypothesis_id, Q5_HYPOTHESIS)
                if mode == "wait":
                    self.assertEqual(pending["row"]["status"],
                                     "waiting_resource")
                else:
                    self.assertEqual(pending["infrastructure_retry_epoch"], 1)
                    self.assertEqual(screen.calls, 1)
                question = D._tracker(D.DurableState(root)).get(
                    Q5_HYPOTHESIS).hypothesis
                self.assertEqual(question.origin, D.hypotheses.ORIGIN_IMPORT)
                self.assertEqual(question.author,
                                 "reviewed-eb26918-continuation")
                self.assertEqual(
                    question.source["preauthored_continuation_sha256"],
                    Q5_CARRIER)

    def test_q5_attribution_boundary_counts_science_and_gates_whole_model(self):
        class StopAfterScreen(RuntimeError):
            pass

        class FreshLease:
            def __init__(self):
                self.calls = []
            def admit(self, item, *, operation_key):
                self.calls.append((item, operation_key))
                return {"admitted": True, "mode": "fresh_test_claim",
                        "operation_key": operation_key}

        class BoundaryScreen:
            def __init__(self, positive):
                self.positive = positive
                self.calls = []
            def screen(self, item, authorization, permit):
                self.calls.append((item, authorization, dict(permit)))
                effect = .01 if self.positive else -.01
                result_sha256 = f"{len(self.calls):064x}"
                return D.SealedScreen(
                    "sealed-result.json", result_sha256, effect,
                    "candidate" if self.positive else "screened_out",
                    "9" * 64, "a" * 64, "b" * 64,
                    exact_attribution_effect_fraction=effect,
                    target_runtime_effect_fraction=(effect if self.positive
                                                    else None),
                    stages=(
                        ("materialized", "built", "correctness", "attribution",
                         "measurement_graphs_off_screen",
                         "target_runtime_graphs_on_screen")
                        if self.positive else
                        ("materialized", "built", "correctness", "attribution")))

        for positive in (False, True):
            with self.subTest(positive=positive), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "state"
                config = self._preauthored_config(root, dry_run=False)
                planner, critic = NeverPlanner(), NeverCritic()
                lease, screen = FreshLease(), BoundaryScreen(positive)
                native_save = D.DurableState.save

                def stop_after_screen(store, state, phase):
                    native_save(store, state, phase)
                    if phase == "screened":
                        raise StopAfterScreen

                with mock.patch.object(
                        D.DurableState, "save", new=stop_after_screen), \
                        self.assertRaises(StopAfterScreen):
                    D.run_controller(
                        config, planner=planner, critic=critic,
                        screener=screen, lease=lease)
                state = D.DurableState(root).load()
                self.assertEqual((planner.calls, critic.calls), (0, 0))
                self.assertEqual((len(lease.calls), len(screen.calls)), (1, 1))
                self.assertEqual(state["scientific_attempts"], 1)
                row = state["iterations"][0]
                self.assertEqual(row["scientific_budget_spent"], True)
                self.assertEqual(row["exact_attribution_effect_fraction"],
                                 .01 if positive else -.01)
                self.assertEqual(
                    row["target_runtime_executed"], positive)
                self.assertEqual(
                    tuple(row["stages"]),
                    ("materialized", "built", "correctness", "attribution",
                     "measurement_graphs_off_screen",
                     "target_runtime_graphs_on_screen")
                    if positive else
                    ("materialized", "built", "correctness", "attribution"))
                if positive:
                    self.assertTrue(state["pending"]["confirmation"])
                    self.assertEqual(
                        state["pending"]["preauthored_continuation"][
                            "authoring_turn"], 1)
                    with mock.patch.object(
                            D.DurableState, "save", new=stop_after_screen), \
                            self.assertRaises(StopAfterScreen):
                        D.run_controller(
                            config, planner=planner, critic=critic,
                            screener=screen, lease=lease)
                    replicated = D.DurableState(root).load()
                    self.assertEqual((planner.calls, critic.calls), (0, 0))
                    self.assertEqual(
                        (len(lease.calls), len(screen.calls)), (2, 2))
                    self.assertEqual(replicated["scientific_attempts"], 2)
                    rows = replicated["iterations"]
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(
                        {row["source_manifest_sha256"] for row in rows},
                        {rows[0]["source_manifest_sha256"]})
                    self.assertEqual(
                        {row["candidate_semantic_sha256"] for row in rows},
                        {rows[0]["candidate_semantic_sha256"]})
                    self.assertEqual(len({row["operation_key"] for row in rows}),
                                     2)
                    self.assertNotEqual(
                        screen.calls[0][1].to_dict(),
                        screen.calls[1][1].to_dict())
                    self.assertEqual(
                        {row["series_key"] for row in rows},
                        {rows[0]["series_key"]})
                    self.assertEqual(
                        replicated["portfolio_terminals"][Q5_HYPOTHESIS]
                                  ["disposition"],
                        "nominated")
                    self.assertNotIn("pending", replicated)
                    self.assertNotEqual(
                        D._select_portfolio_binding(replicated, config)
                         ["hypothesis_id"],
                        Q5_HYPOTHESIS)
                else:
                    self.assertNotIn("pending", state)
                    successor = D._select_portfolio_binding(state, config)
                    self.assertIsNotNone(successor)
                    self.assertNotEqual(successor["hypothesis_id"],
                                        Q5_HYPOTHESIS)

    def test_q5_dispatch_attribution_refusal_is_operation_bound_science_and_not_replayed(self):
        class StopAtBoundary(RuntimeError):
            pass

        class FreshLease:
            def __init__(self):
                self.operation_keys = []
            def admit(self, _item, *, operation_key):
                self.operation_keys.append(operation_key)
                return {"admitted": True, "mode": "fresh_test_claim",
                        "operation_key": operation_key}

        class RefusingScreen:
            def __init__(self):
                self.calls = 0
            def screen(self, _item, _authorization, _permit):
                self.calls += 1
                raise D.DispatchAttributionRefusal(
                    "selected route has no positive duration",
                    receipt_path="exact-attribution-refusal.json",
                    receipt_sha256="c" * 64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state"
            config = self._preauthored_config(root, dry_run=False)
            planner, critic = NeverPlanner(), NeverCritic()
            lease, screen = FreshLease(), RefusingScreen()
            native_save = D.DurableState.save

            def stop_after_refusal(store, state, phase):
                native_save(store, state, phase)
                if phase == "attribution_route_falsified":
                    raise StopAtBoundary

            with mock.patch.object(
                    D.DurableState, "save", new=stop_after_refusal), \
                    self.assertRaises(StopAtBoundary):
                D.run_controller(
                    config, planner=planner, critic=critic,
                    screener=screen, lease=lease)
            state = D.DurableState(root).load()
            self.assertEqual((planner.calls, critic.calls, screen.calls),
                             (0, 0, 1))
            self.assertEqual(state["scientific_attempts"], 1)
            row = state["iterations"][0]
            self.assertEqual(row["status"], "attribution_route_falsified")
            self.assertEqual(row["result_sha256"], "c" * 64)
            self.assertEqual(row["evidence"], {
                "dispatch_attribution": "c" * 64})
            self.assertEqual(row["operation_key"], lease.operation_keys[0])
            semantic = row["candidate_semantic_sha256"]
            self.assertEqual(
                state["attempted_candidate_identities"][semantic]["attempts"][0]
                     ["operation_key"],
                lease.operation_keys[0])
            self.assertNotIn("pending", state)

            def stop_at_next_intent(store, resumed, phase):
                native_save(store, resumed, phase)
                if phase == "planner_intent":
                    raise StopAtBoundary

            with mock.patch.object(
                    D.DurableState, "save", new=stop_at_next_intent), \
                    self.assertRaises(StopAtBoundary):
                D.run_controller(
                    config, planner=planner, critic=critic,
                    screener=screen, lease=lease)
            resumed = D.DurableState(root).load()
            self.assertEqual(screen.calls, 1)
            self.assertEqual(resumed["scientific_attempts"], 1)
            self.assertNotEqual(
                resumed["planning"]["portfolio_binding"]["hypothesis_id"],
                Q5_HYPOTHESIS)

    def test_carrier_refuses_every_missing_extra_or_unhashed_top_level_mutation(self):
        original = json.loads(P.DEFAULT_CARRIER.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key in tuple(original):
                with self.subTest(missing=key):
                    mutated = copy.deepcopy(original)
                    del mutated[key]
                    with self.assertRaises(P.PreauthoredContinuationError):
                        P.load(_write_carrier(root, mutated))
            mutated = copy.deepcopy(original)
            mutated["unexpected"] = True
            with self.assertRaises(P.PreauthoredContinuationError):
                P.load(_write_carrier(root, mutated))
            for section, key in (
                    ("source", "source_file"),
                    ("historical_candidate", "commit"),
                    ("patch", "sha256"),
                    ("experiment_intent", "template_id"),
                    ("correctness_policy", "modern_governed_correctness")):
                with self.subTest(section=section, key=key):
                    mutated = copy.deepcopy(original)
                    mutated[section][key] = "tampered"
                    with self.assertRaises(P.PreauthoredContinuationError):
                        P.load(_write_carrier(root, mutated))

    def test_resealed_join_mutations_cannot_mix_candidate_receipt_or_tail_route(self):
        original = json.loads(P.DEFAULT_CARRIER.read_text())
        mutations = []
        wrong_commit = copy.deepcopy(original)
        wrong_commit["historical_candidate"]["commit"] = "0" * 40
        mutations.append(("foreign candidate", wrong_commit))
        wrong_receipt = copy.deepcopy(original)
        wrong_receipt["historical_receipts"][0]["binary_sha256"] = "0" * 64
        mutations.append(("foreign correctness binary", wrong_receipt))
        tail_as_expected = copy.deepcopy(original)
        tail_as_expected["experiment_intent"]["expected_dispatch"].append(
            copy.deepcopy(tail_as_expected["experiment_intent"]["excluded_dispatch"][0]))
        mutations.append(("excluded tail promoted", tail_as_expected))
        wrong_source_context = copy.deepcopy(original)
        source_context = base64.b64decode(
            wrong_source_context["patch"]["source_backed_base64"],
            validate=True) + b"\n"
        wrong_source_context["patch"]["source_backed_base64"] = (
            base64.b64encode(source_context).decode("ascii"))
        wrong_source_context["patch"]["source_backed_sha256"] = (
            hashlib.sha256(source_context).hexdigest())
        mutations.append(("foreign source-backed scope", wrong_source_context))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, mutated in mutations:
                with self.subTest(label=label):
                    mutated["carrier_sha256"] = _canonical_hash({
                        key: value for key, value in mutated.items()
                        if key != "carrier_sha256"})
                    with self.assertRaises(P.PreauthoredContinuationError):
                        P.load(_write_carrier(root, mutated))

    def test_carrier_file_and_json_transport_fail_closed(self):
        original = json.loads(P.DEFAULT_CARRIER.read_text())
        canonical = (json.dumps(original, sort_keys=True, separators=(",", ":"))
                     + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_bytes(canonical)
            target.chmod(0o600)
            symlink = root / "symlink.json"
            symlink.symlink_to(target)
            with self.assertRaises(P.PreauthoredContinuationError):
                P.load(symlink)
            hardlink = root / "hardlink.json"
            os.link(target, hardlink)
            with self.assertRaises(P.PreauthoredContinuationError):
                P.load(target)
            hardlink.unlink()

            duplicate = root / "duplicate.json"
            duplicate.write_bytes(canonical.replace(
                b'{"carrier_sha256":',
                b'{"carrier_sha256":"0", "carrier_sha256":', 1))
            duplicate.chmod(0o600)
            with self.assertRaises(P.PreauthoredContinuationError):
                P.load(duplicate)

            noncanonical = root / "noncanonical.json"
            noncanonical.write_text(json.dumps(original, indent=2) + "\n")
            noncanonical.chmod(0o600)
            with self.assertRaises(P.PreauthoredContinuationError):
                P.load(noncanonical)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_bytes(canonical.replace(b'"calls":129',
                                                    b'"calls":NaN', 1))
            nonfinite.chmod(0o600)
            with self.assertRaises(P.PreauthoredContinuationError):
                P.load(nonfinite)

            unsafe_mode = root / "unsafe-mode.json"
            unsafe_mode.write_bytes(canonical)
            unsafe_mode.chmod(0o622)
            with self.assertRaises(P.PreauthoredContinuationError):
                P.load(unsafe_mode)

            raced = root / "raced.json"
            raced.write_bytes(canonical)
            raced.chmod(0o600)
            native_fstat = P.os.fstat
            calls = 0
            def changed_epoch(descriptor):
                nonlocal calls
                calls += 1
                current = native_fstat(descriptor)
                if calls == 1:
                    return current
                return SimpleNamespace(**{
                    name: getattr(current, name)
                    for name in ("st_mode", "st_nlink", "st_dev", "st_ino",
                                 "st_size", "st_mtime_ns")},
                    st_ctime_ns=current.st_ctime_ns + 1)
            with mock.patch.object(P.os, "fstat", side_effect=changed_epoch), \
                    self.assertRaises(P.PreauthoredContinuationError):
                P.load(raced)

    def test_generic_mmvq_template_does_not_authorize_preauthored_geometry_symbols(self):
        registry = F._template_registry()
        template = registry.templates["cuda-mmvq-v2"]
        allowed = template.allowed_symbols["ggml/src/ggml-cuda/mmvq.cu"]
        self.assertTrue(set(Q5_SYMBOLS).isdisjoint(allowed))

    def test_dedicated_template_selects_only_routes_zero_through_two_and_forbids_tail(self):
        carrier = P.load(P.DEFAULT_CARRIER)
        registry = F._template_registry()
        template = registry.templates[carrier.template_id]
        intent = D.GpuSourceExperimentIntent(
            carrier.template_id, "gpu_decode", "calc_nwarps",
            carrier.correctness_id, carrier.dispatch_id,
            tuple(D.BoundedDispatchExpectation(**row)
                  for row in carrier.expected_dispatch))
        contract = template.bind_dispatch(intent)
        self.assertEqual(
            tuple(row.signature for row in contract.anchor_exact),
            tuple(row[0] for row in Q5_EXPECTED_ROUTES))
        self.assertEqual(len(contract.candidate_exact), 3)
        self.assertEqual({row.workgroup for row in contract.candidate_exact}, {64})
        self.assertEqual(
            {row.signature for row in contract.candidate_forbidden},
            {Q5_EXCLUDED_ROUTE[0] + ".candidate-stale-tail-forbidden"})
        self.assertEqual(
            tuple((row.calls, row.grid, row.workgroup, row.lds_bytes,
                   row.blocks_per_call)
                  for row in contract.anchor_structural_exact),
            ((129, 57344, 128, 512, 448),))
        self.assertEqual(
            tuple((row.calls, row.grid, row.workgroup, row.lds_bytes,
                   row.blocks_per_call)
                  for row in contract.candidate_structural_exact),
            ((129, 57344, 64, 0, 896),))
        self.assertEqual({row.lds_bytes for row in contract.candidate_exact}, {0})
        tail = D.BoundedDispatchExpectation(**carrier.excluded_dispatch[0])
        with self.assertRaises(F.DeploymentFactoryError):
            template.bind_dispatch(D.GpuSourceExperimentIntent(
                carrier.template_id, "gpu_decode", "calc_nwarps",
                carrier.correctness_id, carrier.dispatch_id,
                (*intent.expected_dispatch, tail)))

    def test_sealed_v26_lds_erratum_authorizes_only_the_exact_q5_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._preauthored_config(
                Path(directory) / "state", dry_run=False)
            erratum = D._q5_lds0_attribution_erratum()
            carry = config.carry_forward
            self.assertEqual(carry["schema"],
                             "epyc.autokernel.discovery_carry_forward.v2")
            self.assertEqual(carry["attribution_expectation_erratum"], erratum)
            self.assertIn(erratum["candidate_semantic_sha256"],
                          carry["candidate_semantic_sha256"])
            self.assertIn(erratum["candidate_patch_sha256"],
                          carry["candidate_patch_sha256"])
            self.assertIn(erratum["cross_campaign_candidate_sha256"],
                          carry["cross_campaign_candidate_sha256"])
            self.assertFalse(erratum["scientific_budget_spent"])
            self.assertFalse(erratum["do_not_repeat"])
            self.assertTrue(erratum["replay_authorized"])
            self.assertEqual(erratum["replacement_disposition"],
                             "attribution_expectation_invalid")
            self.assertEqual(
                erratum["invalidated_predecessor_projection"], {
                    "turn": 1,
                    "result_file_sha256":
                        "40707008b6fceae9749dfca56253836e07ce51b19eb7fb003377c3340503eb86",
                    "removed_effects": [
                        "scientific_attempt", "attempted_candidate_identity",
                        "portfolio_skip", "cross_campaign_do_not_repeat"],
                    "history_retained": True,
                })
            metadata = erratum["compiler_metadata_proof"]
            self.assertEqual(
                {(row["candidate_group_segment_fixed_size"],
                  row["anchor_group_segment_fixed_size"])
                 for row in metadata["rows"]},
                {(0, 1024), (0, 512)})
            self.assertEqual(metadata["candidate_code_object_sha256"],
                             "d40bbb57a78c4474904518a9267370b78f0ae05bfe1dc76c79a86ab589eb2cff")
            self.assertEqual(metadata["anchor_code_object_sha256"],
                             "7a1390f93dda7e5624f0621b00632b7af67fe832d61e9fab16dc64369cf28c0b")
            self.assertEqual(config.max_iterations, 10)
            fresh = D.DurableState(config.output_root).load()
            self.assertEqual(
                (fresh["iterations"], fresh["next"],
                 fresh["scientific_attempts"], fresh["complete"]),
                ([], 1, 0, False))

            binding = self._q5_binding(config)
            candidate = D._preauthored_candidate(config, binding, 1)
            # The exact semantic/patch/cross-campaign triple is in the imported
            # replay sets, but this one receipt-bound erratum authorizes a fresh
            # operation.  The normal portfolio/source checks still run.
            D._validate_portfolio_candidate(
                candidate, binding, config.hypothesis_portfolio, carry)

            for path, replacement in (
                    (("operation_key",), "0" * 64),
                    (("attribution_refusal_file_sha256",), "0" * 64),
                    (("attribution_refusal_receipt_sha256",), "0" * 64),
                    (("correctness_receipt_file_sha256",), "0" * 64),
                    (("profiler_trace_sha256",), "0" * 64),
                    (("scientific_budget_spent",), True),
                    (("stale_candidate_lds_bytes",
                      "cuda-mmvq-q5-onewave-continuation-v1.anchor.0."
                      "candidate-onewave"), 0)):
                with self.subTest(path=path):
                    changed = copy.deepcopy(carry)
                    changed_erratum = changed[
                        "attribution_expectation_erratum"]
                    target = changed_erratum
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = replacement
                    changed_erratum["erratum_sha256"] = _canonical_hash({
                        key: value for key, value in changed_erratum.items()
                        if key != "erratum_sha256"})
                    changed["carry_forward_sha256"] = _canonical_hash({
                        key: value for key, value in changed.items()
                        if key != "carry_forward_sha256"})
                    with self.assertRaises(D.DiscoveryControllerError):
                        replace(
                            config, carry_forward=changed,
                            carry_forward_sha256=changed[
                                "carry_forward_sha256"])

    def test_structural_tail_is_exact_but_never_enters_reward_duration(self):
        carrier = P.load(P.DEFAULT_CARRIER)
        template = F._template_registry().templates[carrier.template_id]
        intent = D.GpuSourceExperimentIntent(
            carrier.template_id, "gpu_decode", "calc_nwarps",
            carrier.correctness_id, carrier.dispatch_id,
            tuple(D.BoundedDispatchExpectation(**row)
                  for row in carrier.expected_dispatch))
        contract = template.bind_dispatch(intent)

        candidate_kernel = (
            "void mul_mat_vec_q<(ggml_type)6, 1, true, false>("
            "void const*, void const*, int const*, "
            "ggml_cuda_mm_fusion_args_device, float*, unsigned int, "
            "HIP_vector_type<unsigned int, 3u>, unsigned int, unsigned int, "
            "unsigned int, HIP_vector_type<unsigned int, 3u>, unsigned int, "
            "unsigned int, unsigned int, HIP_vector_type<unsigned int, 3u>, "
            "unsigned int, unsigned int, unsigned int, unsigned int)")

        def rows(expectations, kernel, duration):
            return [
                {"kernel": kernel,
                 "grid": expectation.grid,
                 "workgroup": expectation.workgroup,
                 "lds": expectation.lds_bytes,
                 "blocks_per_call": expectation.blocks_per_call,
                 "begin_ns": 1,
                 "end_ns": 1 + duration}
                for expectation in expectations
                for _ in range(expectation.calls)]

        candidate_reward = rows(
            contract.candidate_exact, candidate_kernel, 1)
        anchor_reward = rows(
            contract.anchor_exact,
            "void mul_mat_vec_q<(ggml_type)6, 1, true, true>(args)", 2)
        candidate_tail = rows(
            contract.candidate_structural_exact,
            "void mul_mat_vec_q<(ggml_type)6, 1, false, false>(args)",
            10**12)
        anchor_tail = rows(
            contract.anchor_structural_exact,
            "void mul_mat_vec_q<(ggml_type)6, 1, false, true>(args)", 3)

        candidate = E._reduce_arm(
            [*candidate_reward, *candidate_tail],
            exact=contract.candidate_exact,
            structural_exact=contract.candidate_structural_exact,
            forbidden=contract.candidate_forbidden,
            invariants=contract.invariants)
        anchor = E._reduce_arm(
            [*anchor_reward, *anchor_tail], exact=contract.anchor_exact,
            structural_exact=contract.anchor_structural_exact,
            forbidden=contract.anchor_forbidden,
            invariants=contract.invariants)
        comparison = E._exact_duration_comparison(
            {"exact_dispatch_signatures": candidate["exact"],
             "structural_dispatch_signatures": candidate["structural_exact"]},
            {"exact_dispatch_signatures": anchor["exact"],
             "structural_dispatch_signatures": anchor["structural_exact"]})
        self.assertEqual(comparison["direction"], "improved")
        self.assertEqual(
            comparison["candidate_total_duration_ns"],
            sum(row.calls for row in contract.candidate_exact))
        self.assertEqual(
            comparison["anchor_total_duration_ns"],
            2 * sum(row.calls for row in contract.anchor_exact))
        self.assertEqual(
            next(iter(candidate["structural_exact"].values()))[
                "total_duration_ns"],
            10**12 * contract.candidate_structural_exact[0].calls)

        with self.assertRaisesRegex(E.EvidenceProducerError, "count/geometry"):
            E._reduce_arm(
                candidate_reward, exact=contract.candidate_exact,
                structural_exact=contract.candidate_structural_exact,
                forbidden=contract.candidate_forbidden,
                invariants=contract.invariants)
        wrong_tail = [dict(row, workgroup=128) for row in candidate_tail]
        with self.assertRaisesRegex(E.EvidenceProducerError, "count/geometry"):
            E._reduce_arm(
                [*candidate_reward, *wrong_tail],
                exact=contract.candidate_exact,
                structural_exact=contract.candidate_structural_exact,
                forbidden=contract.candidate_forbidden,
                invariants=contract.invariants)
        stale_reward_lds = [dict(row, lds=512) for row in candidate_reward]
        with self.assertRaisesRegex(E.EvidenceProducerError, "count/geometry"):
            E._reduce_arm(
                [*stale_reward_lds, *candidate_tail],
                exact=contract.candidate_exact,
                structural_exact=contract.candidate_structural_exact,
                forbidden=contract.candidate_forbidden,
                invariants=contract.invariants)
        stale_tail_lds = [dict(row, lds=256) for row in candidate_tail]
        with self.assertRaisesRegex(E.EvidenceProducerError, "count/geometry"):
            E._reduce_arm(
                [*candidate_reward, *stale_tail_lds],
                exact=contract.candidate_exact,
                structural_exact=contract.candidate_structural_exact,
                forbidden=contract.candidate_forbidden,
                invariants=contract.invariants)
        stale_tail = rows(
            contract.anchor_structural_exact,
            "void mul_mat_vec_q<(ggml_type)6, 1, false, true>(args)", 3)
        with self.assertRaisesRegex(E.EvidenceProducerError,
                                    "forbidden dispatch"):
            E._reduce_arm(
                [*candidate_reward, *candidate_tail, *stale_tail],
                exact=contract.candidate_exact,
                structural_exact=contract.candidate_structural_exact,
                forbidden=contract.candidate_forbidden,
                invariants=contract.invariants)

    def test_historical_v1_receipts_are_not_current_correctness_authority(self):
        carrier = P.load(P.DEFAULT_CARRIER)
        self.assertTrue(carrier.historical_receipts)
        self.assertTrue(all(row["schema"] ==
                            "epyc.autokernel.targeted_correctness_receipt.v1"
                            for row in carrier.historical_receipts))
        self.assertEqual(E.CORRECTNESS_SCHEMA,
                         "epyc.autokernel.targeted_correctness_receipt.v3")
        self.assertFalse(carrier.correctness_policy[
            "bridge_waives_current_correctness"])
        self.assertEqual(carrier.sha256, Q5_CARRIER)
        self.assertEqual(carrier.correctness_policy[
            "modern_governed_correctness"],
            "required_after_current_instrument_build")

    def test_current_correctness_resumes_only_for_exact_modern_build_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = plan(root / "inputs")
            evidence_root = root / "evidence"
            first_executor, first_claims = FakeExecutors(), ClaimFactory()
            E._produce_correctness(
                evidence_root, current, first_executor.correctness,
                claim_acquirer=first_claims, claim_verifier=lambda _row: True,
                claim_journal=object(), claim_timeout_s=0)
            self.assertEqual([row[0] for row in first_executor.calls],
                             ["correctness"])
            self.assertEqual(len(first_claims.claims), 1)

            resumed_executor, resumed_claims = FakeExecutors(), ClaimFactory()
            E.produce_gpu_source_evidence(
                output_root=evidence_root, plan=current,
                correctness_executor=resumed_executor.correctness,
                rocprof_executor=resumed_executor.rocprof,
                claim_journal=object(), claim_acquirer=resumed_claims,
                claim_verifier=lambda _row: True, claim_timeout_s=0)
            self.assertEqual([row[:2] for row in resumed_executor.calls],
                             [("rocprof", "candidate"),
                              ("rocprof", "anchor")])
            self.assertEqual(len(resumed_claims.claims), 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = plan(root / "inputs")
            evidence_root = root / "evidence"
            E._produce_correctness(
                evidence_root, current, FakeExecutors().correctness,
                claim_acquirer=ClaimFactory(), claim_verifier=lambda _row: True,
                claim_journal=object(), claim_timeout_s=0)
            receipt = evidence_root / "correctness/receipt.json"
            body = json.loads(receipt.read_text())
            body["candidate_build_identity"]["binary_sha256"] = "0" * 64
            body["receipt_sha256"] = E.schemas.content_hash({
                key: value for key, value in body.items()
                if key != "receipt_sha256"})
            receipt.write_text(json.dumps(body, sort_keys=True) + "\n")
            refused_executor, refused_claims = FakeExecutors(), ClaimFactory()
            with self.assertRaises(E.EvidenceProducerError):
                E.produce_gpu_source_evidence(
                    output_root=evidence_root, plan=current,
                    correctness_executor=refused_executor.correctness,
                    rocprof_executor=refused_executor.rocprof,
                    claim_journal=object(), claim_acquirer=refused_claims,
                    claim_verifier=lambda _row: True, claim_timeout_s=0)
            self.assertEqual(refused_executor.calls, [])
            self.assertEqual(refused_claims.claims, [])

    def test_enum_scope_is_source_backed_and_spoof_resistant(self):
        prefix = "diff --git a/x.cu b/x.cu\n--- a/x.cu\n+++ b/x.cu\n"
        exact = prefix + (
            "@@ -1,3 +1,4 @@ stale\n"
            " enum mmvq_parameter_table_id {\n"
            "     GENERIC = 0,\n"
            "+    CDNA2,\n"
            " };\n")
        self.assertEqual(self._scope(exact), ("mmvq_parameter_table_id",))
        self.assertEqual(
            S.source_backed_symbol_map(exact),
            {"x.cu": ("mmvq_parameter_table_id",)})

        spoofs = {
            "comment": " // enum mmvq_parameter_table_id {\n-old\n+new\n",
            "call": " use_enum(mmvq_parameter_table_id,\n-old\n+new\n",
            "macro": " #define mmvq_parameter_table_id enum_value\n-old\n+new\n",
            "typedef": " typedef enum { VALUE } mmvq_parameter_table_id;\n-old\n+new\n",
            "anonymous": " enum { mmvq_parameter_table_id = 1 };\n-old\n+new\n",
            "adjacent-function": (
                " enum mmvq_parameter_table_id { VALUE };\n"
                " static int adjacent_spoof() {\n-old\n+new\n }\n"),
        }
        for label, body in spoofs.items():
            with self.subTest(label=label):
                derived = self._scope(prefix + "@@ -1,3 +1,3 @@ stale\n" + body)
                self.assertNotIn("mmvq_parameter_table_id", derived)

        actor_header = prefix + (
            "@@ -1,1 +1,1 @@ enum mmvq_parameter_table_id {\n-old\n+new\n")
        self.assertEqual(S.hunk_identities(actor_header)[1], (S.FILE_SCOPE,))

    def test_full_q5_patch_materializes_exact_four_symbols_on_current_preimage(self):
        carrier = P.load(P.DEFAULT_CARRIER)
        source_repository = Path("/mnt/raid0/llm/llama.cpp")
        P.verify_git_authority(
            carrier, source_repository,
            carrier.compatibility_bridge["current_instrument_commit"])
        preimage = subprocess.run(
            ["/usr/bin/git", "-C", str(source_repository), "show",
             f"{carrier.compatibility_bridge['current_instrument_commit']}:"
             f"{carrier.source_file}"], check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE).stdout
        self.assertEqual(hashlib.sha256(preimage).hexdigest(),
                         carrier.compatibility_bridge[
                             "current_instrument_file_sha256"])
        source_backed_diff = carrier.source_backed_diff

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "test", repository],
                           check=True)
            for key, value in (("user.name", "AutoKernel Test"),
                               ("user.email", "ak@test.invalid")):
                subprocess.run(["git", "-C", repository, "config", key, value],
                               check=True)
            target = repository / carrier.source_file
            target.parent.mkdir(parents=True)
            target.write_bytes(preimage)
            subprocess.run(["git", "-C", repository, "add", "--",
                            carrier.source_file], check=True)
            subprocess.run(["git", "-C", repository, "commit", "-qm", "base"],
                           check=True)
            production = subprocess.run(
                ["git", "-C", repository, "rev-parse", "HEAD"], check=True,
                text=True, stdout=subprocess.PIPE).stdout.strip()
            subprocess.run(["git", "-C", repository, "commit", "--allow-empty",
                            "-qm", "instrument"], check=True)
            instrument = subprocess.run(
                ["git", "-C", repository, "rev-parse", "HEAD"], check=True,
                text=True, stdout=subprocess.PIPE).stdout.strip()
            manifest_kwargs = dict(
                campaign_id="ak-q5-continuation-test",
                proposal_id="akp-q5-continuation-test",
                candidate_id="akc-q5-continuation-test",
                source_tree="llama.cpp", production_base_commit=production,
                instrument_commit=instrument, change_class=carrier.change_class,
                declared_files=(carrier.source_file,),
                declared_symbols={carrier.source_file: carrier.declared_symbols},
                mechanism_id=carrier.mechanism_id,
                patch_sha256=carrier.patch_sha256,
                patch_bytes=carrier.patch_bytes)
            with self.assertRaisesRegex(S.SourceCandidateError, "undeclared"):
                S.SourcePatchManifest(**manifest_kwargs)
            manifest = S.source_backed_source_patch_manifest(
                **manifest_kwargs, source_backed_diff=source_backed_diff)
            proposal = {
                "proposal_id": manifest.proposal_id,
                "change_class": manifest.change_class,
                "change": {"files_and_symbols": [
                    f"{carrier.source_file}:{symbol}"
                    for symbol in carrier.declared_symbols],
                    "estimated_diff_size": 15},
            }
            repo = W.GitRepo(repository)
            destination = W.SandboxPath.create(
                root / "actor", sandbox_root=root, production_trees=())
            actor = repo.add_worktree(
                destination, instrument,
                branch=W.SafeBranch.for_campaign(
                    "ak-q5-continuation-test", "source"))
            try:
                applied = S.apply_source_candidate(
                    manifest, proposal=proposal, actor=actor)
                self.assertEqual(
                    applied.actual_symbols,
                    tuple(f"{carrier.source_file}:{symbol}"
                          for symbol in carrier.declared_symbols))
            finally:
                if destination.path in repo.worktree_paths():
                    repo.remove_worktree(destination, force=True)


if __name__ == "__main__":
    unittest.main()
