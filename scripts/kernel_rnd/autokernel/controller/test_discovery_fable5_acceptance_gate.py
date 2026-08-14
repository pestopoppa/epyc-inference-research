"""Independent, hardware-free acceptance gate for the Fable 5 critic cutover.

The tests deliberately invoke neither Codex nor Claude.  They exercise pure
parsers, captured prompts, declarative graph construction, and the controller
state machine with explosive compute/resource fakes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from .. import schemas
from . import discovery_controller as controller
from . import discovery_deployment as deployment
from . import discovery_deployment_factory as factory

try:
    from . import claude_fable5_critic_actor as fable_actor
except ImportError:  # The gate is intentionally red on the pre-cutover base.
    fable_actor = None


HASH = "a" * 64
SOL_RUNTIME = {
    "kind": "docker_workspace_bind_only", "docker_path": "/docker",
    "docker_sha256": HASH, "image_id": "image", "codex_native_sha256": HASH,
    "code_mode_host_sha256": HASH, "ca_certificate_sha256": HASH,
    "writable_host_binds": ["/workspace"], "host_network_mode": "docker_bridge",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fable_identity() -> dict[str, str]:
    return {
        "provider": "claude", "model": "claude-fable-5",
        "effort": "high", "role": "critic",
    }


def _claude_runtime(wrapper: Path) -> dict[str, object]:
    assert fable_actor is not None
    return fable_actor.runtime_identity(wrapper)


def _wrapper(root: Path, name: str = "claude-fable5") -> Path:
    path = root / name
    path.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _bindings() -> dict[str, str]:
    assert fable_actor is not None
    return {key: chr(97 + index) * 64
            for index, key in enumerate(fable_actor.BINDING_KEYS)}


class _Manifest:
    campaign_id = "ak-fable-gate"
    proposal_id = "akp-fable-gate"
    candidate_id = "akc-fable-gate"
    production_base_commit = "0" * 40
    instrument_commit = "1" * 40
    declared_files = ("ggml/src/ggml-cuda/fattn.cu",)
    declared_symbols = {"ggml/src/ggml-cuda/fattn.cu": ("fattn_kernel",)}

    def __init__(self, patch_text: str) -> None:
        self.patch_text = patch_text
        self.patch_sha256 = _sha(patch_text.encode("utf-8"))


def _candidate(patch_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        hypothesis_id="akh-fable-gate", statement="bounded mechanism",
        falsifier="no improvement", proposal={"proposal_id": "akp-fable-gate"},
        experiment_intent=None, source_manifest=_Manifest(patch_text),
        source_manifest_sha256="f" * 64,
    )


class ExactRosterAndConfigGate(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(fable_actor, "Fable critic launcher module is absent")
        self.assertTrue(hasattr(controller, "FABLE5_CRITIC"),
                        "controller lacks the sealed Fable critic identity")

    def test_roster_is_exactly_one_sol_and_one_fable_with_zero_terra(self) -> None:
        self.assertEqual(controller.FABLE5_CRITIC, _fable_identity())
        self.assertEqual(controller.sealed_roster(), {
            "schema": "epyc.autokernel.discovery_roster.v3",
            "members": [controller.SOL, _fable_identity()],
            "claude_members": 1, "member_count": 2,
        })
        critics = [row for row in controller.sealed_roster()["members"]
                   if row["role"] == "critic"]
        self.assertEqual(critics, [_fable_identity()])
        self.assertNotIn("gpt-5.6-terra", {row["model"] for row in critics})

    def test_fresh_config_and_graph_reseal_every_heterogeneous_actor_authority(self) -> None:
        forbidden = AssertionError("config-only graph construction invoked a model")
        with tempfile.TemporaryDirectory(prefix="autokernel-fable-config-gate-") as temporary, \
                mock.patch.object(factory.codex_container_actor, "run_actor",
                                  side_effect=forbidden), \
                mock.patch.object(fable_actor, "run_critic", side_effect=forbidden), \
                mock.patch.object(controller, "run_controller", side_effect=forbidden):
            root = Path(temporary) / "bundle"
            config_path = factory.initialize_static_deployment_bundle(root)
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema"], "epyc.autokernel.discovery_deployment.v2")
            self.assertEqual(raw["config_sha256"], schemas.content_hash(
                {key: value for key, value in raw.items() if key != "config_sha256"}))
            self.assertEqual(set(raw["actors"]), {
                "wrapper_path", "wrapper_sha256", "critic_path", "critic_sha256",
                "environment_profile_id",
            })
            critic_path = Path(raw["actors"]["critic_path"])
            self.assertTrue(critic_path.is_absolute())
            self.assertFalse(critic_path.is_symlink())
            self.assertEqual(_sha(critic_path.read_bytes()), raw["actors"]["critic_sha256"])

            parsed = deployment.load_deployment_config(config_path)
            graph = factory.build_static_deployment_graph(parsed)
            receipt = json.loads(graph.graph_receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], "epyc.autokernel.static_discovery_graph.v2")
            self.assertEqual(receipt["actor_cells"], [controller.SOL, _fable_identity()])
            self.assertEqual(set(receipt["actor_wrappers"]), {"planner", "critic"})
            self.assertEqual(set(receipt["actor_runtimes"]), {"planner", "critic"})
            self.assertEqual(set(receipt["actor_argv_authority"]), {"planner", "critic"})
            self.assertEqual(
                receipt["actor_runtimes"]["critic"],
                fable_actor.runtime_identity(parsed.critic_wrapper.path),
            )
            critic_authority = receipt["actor_argv_authority"]["critic"]
            launcher = Path(critic_authority["module"])
            self.assertEqual(_sha(launcher.read_bytes()),
                             critic_authority["module_sha256"])
            self.assertIn("claude", critic_authority["constructor"])
            self.assertIn("claude_fable5_critic_actor", receipt["execution_modules"])
            self.assertFalse(receipt["inference_executed"])
            self.assertIsInstance(graph.adapters["critic"], controller.ClaudeCritic)

            registry = factory._static_registry(parsed, factory._template_registry())
            generic = factory.materialize(
                parsed, registry, correctness_executor=mock.Mock(),
                rocprof_executor=mock.Mock(), claim_journal=mock.Mock(),
            )
            self.assertIsInstance(generic["critic"], controller.ClaudeCritic)
            self.assertNotEqual(type(generic["critic"]).__name__, "CodexCritic")

    def test_actor_fields_are_config_hash_authority_and_exact_parser_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autokernel-fable-hash-gate-") as temporary:
            config_path = factory.initialize_static_deployment_bundle(Path(temporary) / "bundle")
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            original = raw["config_sha256"]
            for field in ("wrapper_path", "wrapper_sha256", "critic_path",
                          "critic_sha256", "environment_profile_id"):
                with self.subTest(field=field):
                    changed = json.loads(json.dumps(raw))
                    changed["actors"][field] = changed["actors"][field] + "x"
                    changed_hash = schemas.content_hash({
                        key: value for key, value in changed.items()
                        if key != "config_sha256"})
                    self.assertNotEqual(changed_hash, original)


class StrictClaudeBoundaryGate(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(fable_actor, "Fable critic launcher module is absent")

    def test_argv_is_exact_fable_high_no_tools_no_settings_no_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autokernel-fable-argv-gate-") as temporary:
            root = Path(temporary)
            wrapper = _wrapper(root)
            config = root / "config"
            config.mkdir(mode=0o700)
            mcp = config / "empty-mcp.json"
            mcp.write_bytes(fable_actor._EMPTY_MCP)
            mcp.chmod(0o600)
            argv = fable_actor.build_argv(
                wrapper=wrapper, config_dir=config, bindings=_bindings())
            self.assertEqual(argv[argv.index("--model") + 1], "claude-fable-5")
            self.assertEqual(argv[argv.index("--effort") + 1], "high")
            self.assertEqual(argv[argv.index("--permission-mode") + 1], "plan")
            self.assertEqual(argv[argv.index("--tools") + 1], "")
            self.assertIn("--strict-mcp-config", argv)
            self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
            self.assertIn("--no-session-persistence", argv)
            schema = json.loads(argv[argv.index("--json-schema") + 1])
            for key, value in _bindings().items():
                self.assertEqual(schema["properties"][key], {"const": value})

    def test_strict_structured_output_accepts_exact_bindings_and_refuses_replay(self) -> None:
        bindings = _bindings()
        payload = {"decision": "reject", "reason": "unsafe patch", **bindings}
        wrapped = json.dumps({"type": "result", "structured_output": payload})
        self.assertEqual(fable_actor._parse_result(wrapped, bindings), payload)

        for key in fable_actor.BINDING_KEYS:
            with self.subTest(replayed_binding=key):
                replay = dict(payload)
                replay[key] = "9" * 64
                with self.assertRaises(fable_actor.ClaudeFable5CriticError):
                    fable_actor._parse_result(json.dumps({
                        "type": "result", "structured_output": replay}), bindings)
        for malformed in (
            json.dumps(payload),
            json.dumps({"result": json.dumps(payload)}),
            json.dumps({"result": "```json\n" + json.dumps(payload) + "\n```"}),
            json.dumps({"structured_output": {**payload, "extra": True}}),
        ):
            with self.subTest(malformed=malformed[:40]), \
                    self.assertRaises(fable_actor.ClaudeFable5CriticError):
                fable_actor._parse_result(malformed, bindings)

    def test_full_prompt_is_exact_and_65536_byte_patch_boundary_is_not_truncated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autokernel-fable-prompt-gate-") as temporary:
            root = Path(temporary)
            wrapper = _wrapper(root)
            runtime = _claude_runtime(wrapper)
            launcher_sha = _sha(Path(fable_actor.__file__).resolve().read_bytes())
            catalog = {"template": {
                "allowed_files": ["ggml/src/ggml-cuda/fattn.cu"],
                "allowed_symbols": {"ggml/src/ggml-cuda/fattn.cu": ["fattn_kernel"]},
                "semantics": {"dispatch": {"calls": 9}},
            }}
            context = {"planner_context": {"hotspots": ["fattn_kernel"]},
                       "do_not_repeat": {"mechanisms": ["retired"]}}
            critic = controller.ClaudeCritic(
                wrapper=wrapper, environment={"HOME": str(root)},
                template_catalog=catalog, wrapper_sha256=_sha(wrapper.read_bytes()),
                runtime_identity=runtime, actor_launcher_sha256=launcher_sha,
            )
            accepted = SimpleNamespace(decision="accept", reason="bounded")
            with mock.patch.object(fable_actor, "run_critic", return_value=accepted) as run:
                result = critic.review(
                    _candidate("x" * 65536), context=context, workspace=root)
            self.assertEqual(result.decision, "accept")
            prompt = json.loads(run.call_args.kwargs["prompt"])
            self.assertEqual(prompt["role"], _fable_identity())
            self.assertEqual(prompt["context"], context)
            self.assertEqual(prompt["experiment_template_catalog"], catalog)
            self.assertEqual(prompt["candidate"]["proposal"],
                             {"proposal_id": "akp-fable-gate"})
            self.assertEqual(prompt["candidate"]["manifest"]["patch_text"],
                             "x" * 65536)
            self.assertEqual(prompt["required_output_bindings"],
                             run.call_args.kwargs["bindings"])
            self.assertEqual(run.call_args.kwargs["expected_wrapper_sha256"],
                             _sha(wrapper.read_bytes()))
            self.assertEqual(run.call_args.kwargs["expected_runtime_identity"], runtime)
            self.assertEqual(run.call_args.kwargs["expected_launcher_sha256"],
                             launcher_sha)

            with mock.patch.object(fable_actor, "run_critic") as forbidden, \
                    self.assertRaises(controller.DiscoveryControllerError):
                critic.review(_candidate("x" * 65537), context=context, workspace=root)
            forbidden.assert_not_called()

    def test_binary_runtime_and_launcher_drift_refuse_before_model_call(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autokernel-fable-drift-gate-") as temporary:
            root = Path(temporary)
            wrapper = _wrapper(root)
            digest = _sha(wrapper.read_bytes())
            runtime = _claude_runtime(wrapper)
            launcher_sha = _sha(Path(fable_actor.__file__).resolve().read_bytes())
            critic = controller.ClaudeCritic(
                wrapper=wrapper, environment={"HOME": str(root)},
                wrapper_sha256=digest, runtime_identity=runtime,
                actor_launcher_sha256=launcher_sha,
            )
            with mock.patch.object(fable_actor, "run_critic",
                                   side_effect=AssertionError("model invoked")):
                self.assertEqual(critic.attest()["runtime"], runtime)
                wrapper.write_text("#!/bin/sh\nexit 92\n", encoding="utf-8")
                with self.assertRaises(controller.DiscoveryControllerError):
                    critic.attest()

            fresh = _wrapper(root, "fresh-wrapper")
            with mock.patch.object(fable_actor, "run_critic",
                                   side_effect=AssertionError("model invoked")):
                wrong_runtime = dict(_claude_runtime(fresh))
                wrong_runtime["argv_policy_sha256"] = "0" * 64
                with self.assertRaises(controller.DiscoveryControllerError):
                    controller.ClaudeCritic(
                        wrapper=fresh, environment={"HOME": str(root)},
                        wrapper_sha256=_sha(fresh.read_bytes()),
                        runtime_identity=wrong_runtime,
                    ).attest()
                with self.assertRaises(controller.DiscoveryControllerError):
                    controller.ClaudeCritic(
                        wrapper=fresh, environment={"HOME": str(root)},
                        wrapper_sha256=_sha(fresh.read_bytes()),
                        runtime_identity=_claude_runtime(fresh),
                        actor_launcher_sha256="0" * 64,
                    ).attest()

    def test_launcher_drift_during_staging_refuses_before_model_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autokernel-fable-module-race-gate-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            auth = root / "auth"
            workspace.mkdir(mode=0o700)
            auth.mkdir(mode=0o700)
            credential = auth / ".credentials.json"
            credential.write_text(
                json.dumps({"claudeAiOauth": {"accessToken": "test-only"}}),
                encoding="utf-8",
            )
            credential.chmod(0o600)
            wrapper = _wrapper(root)
            runtime = _claude_runtime(wrapper)
            launcher = Path(fable_actor.__file__).resolve()
            launcher_bytes = launcher.read_bytes()
            real_read = fable_actor._read_regular
            launcher_reads = 0

            def drifted_read(path, **kwargs):
                nonlocal launcher_reads
                if Path(path).resolve() == launcher:
                    launcher_reads += 1
                    if launcher_reads > 1:
                        return b"simulated-launcher-drift"
                return real_read(path, **kwargs)

            with mock.patch.object(fable_actor, "_read_regular",
                                   side_effect=drifted_read), \
                    mock.patch.object(
                        fable_actor, "_run_process",
                        side_effect=AssertionError("drift reached model boundary"),
                    ) as model_boundary, \
                    self.assertRaisesRegex(
                        fable_actor.ClaudeFable5CriticError, "launcher"):
                fable_actor.run_critic(
                    wrapper=wrapper, workspace=workspace, prompt="review",
                    bindings=_bindings(),
                    environment={"HOME": str(root)}, auth_root=auth,
                    expected_wrapper_sha256=runtime["wrapper_sha256"],
                    expected_runtime_identity=runtime,
                    expected_launcher_sha256=_sha(launcher_bytes),
                )
            model_boundary.assert_not_called()
            self.assertGreaterEqual(launcher_reads, 2)


class VetoAndResumeGate(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(fable_actor, "Fable critic launcher module is absent")

    @staticmethod
    def _manifest() -> SimpleNamespace:
        return SimpleNamespace(
            campaign_id="ak-veto", proposal_id="akp-veto", candidate_id="akc-veto",
            source_tree="llama.cpp", production_base_commit="0" * 40,
            instrument_commit="1" * 40, change_class="source",
            declared_files=("ggml/src/ggml-cuda/fattn.cu",),
            declared_symbols={"ggml/src/ggml-cuda/fattn.cu": ("fattn_kernel",)},
            mechanism_id="veto", patch_sha256="0" * 64,
            patch_bundle_sha256=HASH, patch_bytes=b"patch",
        )

    def _planner(self):
        manifest = self._manifest()

        class Planner:
            calls = 0
            def attest(inner_self):
                return {**controller.SOL, "runtime": SOL_RUNTIME}
            def plan(inner_self, *, context, workspace):
                inner_self.calls += 1
                return controller.PlannedCandidate(
                    "akh-veto", "bounded", "no effect", {"backend": "gpu"},
                    {"proposal_id": "akp-veto"}, manifest, HASH,
                )
        return Planner()

    @staticmethod
    def _critic(decision: str):
        runtime = {
            "kind": "claude_cli_structured_critic", "provider": "claude",
            "model": "claude-fable-5", "effort": "high",
            "wrapper_path": "/sealed/claude-fable5", "wrapper_sha256": HASH,
            "argv_policy_sha256": HASH,
            "auth_staging_policy": "ephemeral_0600_copy_no_secret_receipt",
        }
        class Critic:
            calls = 0
            def attest(inner_self):
                return {**controller.FABLE5_CRITIC, "runtime": runtime}
            def review(inner_self, candidate, *, context, workspace):
                inner_self.calls += 1
                return controller.Critique(decision, "binding veto")
        return Critic()

    def test_reject_and_revise_veto_before_authorization_lease_build_claim_or_screen(self) -> None:
        class ExplosiveLease:
            def admit(self, *_args, **_kwargs):
                raise AssertionError("veto reached resource admission/claim")
            def resume(self, *_args, **_kwargs):
                raise AssertionError("veto reached resource resume")
        class ExplosiveScreen:
            def screen(self, *_args, **_kwargs):
                raise AssertionError("veto reached build/evidence/runner")
            def reconcile(self, *_args, **_kwargs):
                raise AssertionError("veto reached reconciliation")

        for decision in ("reject", "revise"):
            with self.subTest(decision=decision), \
                    tempfile.TemporaryDirectory(prefix="autokernel-fable-veto-gate-") as temporary, \
                    mock.patch.object(controller, "_write_projection"), \
                    mock.patch.object(
                        controller, "_ensure_question",
                        side_effect=AssertionError("veto reached authorization"),
                    ) as authorization:
                planner = self._planner()
                critic = self._critic(decision)
                result = controller.run_controller(
                    controller.ControllerConfig(Path(temporary) / "out", 1),
                    planner=planner, critic=critic, screener=ExplosiveScreen(),
                    lease=ExplosiveLease(),
                )
                self.assertEqual(planner.calls, 1)
                self.assertEqual(critic.calls, 1)
                self.assertEqual(result["iterations"][0]["status"],
                                 f"critic_{decision}")
                authorization.assert_not_called()

    def test_old_terra_state_refuses_before_plan_critic_or_compute(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autokernel-fable-resume-gate-") as temporary:
            output = Path(temporary) / "out"
            output.mkdir(parents=True)
            old = {
                "schema": "epyc.autokernel.discovery_controller.v2",
                "authority": controller.AUTHORITY,
                "roster": {
                    "schema": "epyc.autokernel.discovery_roster.v2",
                    "members": [controller.SOL, {
                        "provider": "codex", "model": "gpt-5.6-terra",
                        "effort": "high", "role": "critic"}],
                    "claude_members": 0, "member_count": 2,
                },
                "iterations": [], "next": 1, "complete": False,
            }
            old["state_sha256"] = controller._sha(old)
            (output / "state.json").write_text(
                json.dumps(old, sort_keys=True), encoding="utf-8")

            class ExplosivePlanner:
                def attest(inner_self):
                    return {**controller.SOL, "runtime": SOL_RUNTIME}
                def plan(inner_self, **_kwargs):
                    raise AssertionError("old Terra state reached planner")
            critic = self._critic("accept")
            critic.review = mock.Mock(side_effect=AssertionError(
                "old Terra state reached critic"))
            class Explosive:
                def __getattr__(inner_self, _name):
                    raise AssertionError("old Terra state reached compute")

            with self.assertRaisesRegex(controller.DiscoveryControllerError, "roster"):
                controller.run_controller(
                    controller.ControllerConfig(output, 1),
                    planner=ExplosivePlanner(), critic=critic,
                    screener=Explosive(), lease=Explosive(),
                )
            critic.review.assert_not_called()


if __name__ == "__main__":
    unittest.main()
