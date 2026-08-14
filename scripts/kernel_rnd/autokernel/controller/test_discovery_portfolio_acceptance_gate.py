"""Independent, hardware-free acceptance gate for the discovery portfolio.

The gate calls no model, profiler, compiler, or GPU API.  It validates generated
configuration, immutable source/evidence carriers, pure dispatch reduction over
the already-sealed Qwen trace, and fail-closed controller/runner boundaries.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import discovery_controller as controller
from . import discovery_deployment as deployment
from . import discovery_deployment_factory as factory
from . import gpu_source_evidence as evidence


FINAL_PRODUCT_SHA = "2153ccacb7f333c34abbc625b384644231f0595c"
PRODUCTION_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
INSTRUMENT_COMMIT = "81bf32f11b4a421880e8f25faec3e4ba872363f0"
MODEL_SHA256 = "f175ecace8c24336cbf9e22bd71ea032a16492bd264a3caab6dfa4cafe80ddd3"
WORKLOAD_SHA256 = "d20a7a5530b9701170dce8aafc3ed96156c708785b9f695c386ea247717de0c9"
RUNTIME_SHA256 = "d298c1db7601ec6874eb45cd6f595c267fbc38604531996a61f140c051c28246"
HASH64 = "a" * 64
PLANNER_RUNTIME = {
    "kind": "docker_workspace_bind_only", "docker_path": "/docker",
    "docker_sha256": HASH64, "image_id": "image",
    "codex_native_sha256": HASH64, "code_mode_host_sha256": HASH64,
    "ca_certificate_sha256": HASH64, "writable_host_binds": ["/workspace"],
    "host_network_mode": "docker_bridge",
}
CRITIC_RUNTIME = {
    "kind": "claude_cli_structured_critic", "provider": "claude",
    "model": "claude-fable-5", "effort": "high",
    "wrapper_path": "/sealed/claude", "wrapper_sha256": HASH64,
    "argv_policy_sha256": HASH64,
    "auth_staging_policy": "ephemeral_0600_copy_no_secret_receipt",
}
TRACE_RECEIPT = Path(
    "/mnt/raid0/llm/autokernel/screens/"
    "ak-gpu-qwen05b-tg128-rocprof-attribution-20260813/receipt.json"
)
OLD_BUNDLE = Path(
    "/mnt/raid0/llm/autokernel/deployments/"
    "gpu-discovery-fable5-critic-v1/config/deployment.json"
)
HIGH_VALUE_SOURCES = {
    "ggml/src/ggml-cuda/quantize.cu",
    "ggml/src/ggml-cuda/fattn-tile.cuh",
    "ggml/src/ggml-cuda/fattn-common.cuh",
    "ggml/src/ggml-cuda/set-rows.cu",
}
Q5_ONEWAVE_COMMIT = "eb26918fa82f8aef3ab72f1e3263bd8fecde62e7"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _content_hash(value: object) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode())


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _refs(value: object) -> list[dict]:
    found: list[dict] = []
    if isinstance(value, dict):
        if (set(("path", "sha256")).issubset(value)
                and isinstance(value.get("path"), str)
                and isinstance(value.get("sha256"), str)):
            found.append(value)
        for child in value.values():
            found.extend(_refs(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_refs(child))
    return found


def _rows(portfolio: dict, key: str) -> list[dict]:
    value = portfolio.get(key)
    if not isinstance(value, list):
        raise AssertionError(f"portfolio {key!r} must be a list")
    if not all(isinstance(row, dict) for row in value):
        raise AssertionError(f"portfolio {key!r} contains a non-object")
    return value


def _hypotheses(portfolio: dict) -> list[dict]:
    return _rows(portfolio, "hypotheses")


def _eligible(portfolio: dict) -> list[dict]:
    return [row for row in _hypotheses(portfolio)
            if row.get("current_bundle_eligibility", {}).get("eligible") is True]


def _planner_partition(planner: dict, key: str) -> list[dict]:
    """Find one named portfolio partition without prescribing wrapper nesting."""
    if isinstance(planner.get(key), list):
        return planner[key]
    for value in planner.values():
        if isinstance(value, dict):
            found = _planner_partition(value, key)
            if found:
                return found
    return []


def _geometry_records(value: object, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict]]:
    """Return every exact dispatch-shaped row together with its semantic path."""
    found: list[tuple[tuple[str, ...], dict]] = []
    if isinstance(value, dict):
        if {"calls", "grid", "workgroup"}.issubset(value) and (
                "lds_bytes" in value or "lds" in value):
            found.append((path, value))
        for key, child in value.items():
            found.extend(_geometry_records(child, (*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_geometry_records(child, (*path, str(index))))
    return found


def _geometry(row: dict) -> tuple[int, int, int, int]:
    return (row["calls"], row["grid"], row["workgroup"],
            row.get("lds_bytes", row.get("lds")))


def _routed_geometry(row: dict) -> tuple[str, int, int, int, int]:
    route_id = row.get("route_id", row.get("signature"))
    if not isinstance(route_id, str) or not route_id:
        raise AssertionError("dispatch row lost its reviewed route identity")
    return (route_id, *_geometry(row))


def _matches(row: dict, groups: tuple[tuple[str, ...], ...]) -> bool:
    text = json.dumps(row, sort_keys=True).lower()
    return all(any(token in text for token in group) for group in groups)


def _assert_eligible_projection(
        case: unittest.TestCase, projected: list[dict], source_rows: list[dict]) -> None:
    """Prove the bounded projection preserves every spend-authority field."""
    source_by_id = {row["hypothesis_id"]: row for row in source_rows}
    case.assertEqual({row["hypothesis_id"] for row in projected}, set(source_by_id))
    exact_fields = (
        "hypothesis_id", "record_version", "statement", "falsifiers",
        "primary_falsifier", "mechanism", "regime", "target", "dispatch_anchors",
        "decision_policy", "evidence_refs", "epistemic", "provenance", "stop_rule",
    )
    for row in projected:
        source = source_by_id[row["hypothesis_id"]]
        for key in exact_fields:
            case.assertEqual(row[key], source[key],
                             f"eligible projection changed {row['hypothesis_id']}:{key}")
        case.assertEqual(row["template_ids"],
                         source["current_bundle_eligibility"]["template_ids"])
        lifecycle = source["lifecycle"]
        for key in ("maturity", "next_action", "candidate_identity",
                    "diagnostic_identity"):
            case.assertEqual(row[key], lifecycle.get(key))
        case.assertEqual({frame["frame_id"] for frame in row["frames"]},
                         set(source["target"]["frame_ids"]))


class _BundleMixin:
    def make_bundle(self, temporary: str) -> tuple[Path, Path, dict, dict, dict]:
        root = Path(temporary) / "bundle"
        config_path = factory.initialize_static_deployment_bundle(root)
        config = _json(config_path)
        immutable = config.get("immutable_inputs")
        self.assertIsInstance(immutable, dict)
        self.assertIn("hypothesis_portfolio", immutable,
                      "deployment does not bind a hypothesis portfolio")
        portfolio_path = Path(immutable["hypothesis_portfolio"]["path"])
        planner_path = Path(config["planner_context"]["path"])
        self.assertTrue(portfolio_path.is_relative_to(root))
        return root, config_path, config, _json(portfolio_path), _json(planner_path)


class PortfolioAuthorityGate(_BundleMixin, unittest.TestCase):
    def test_v2_portfolio_projection_and_graph_bind_immutable_authority(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-v2-gate-") as temporary:
            _root, config_path, config, portfolio, planner = self.make_bundle(temporary)
            parsed = deployment.load_deployment_config(config_path)
            graph = factory.build_static_deployment_graph(parsed)
            receipt = _json(graph.graph_receipt)
            portfolio_ref = config["immutable_inputs"]["hypothesis_portfolio"]
            portfolio_bytes = Path(portfolio_ref["path"]).read_bytes()

        self.assertEqual(portfolio.get("schema"),
                         "epyc.autokernel.discovery_hypothesis_portfolio.v2")
        self.assertEqual(config.get("schema"),
                         "epyc.autokernel.discovery_deployment.v4")
        declared = portfolio_ref["sha256"]
        self.assertEqual(declared, _sha(portfolio_bytes))
        self.assertEqual(planner.get("schema"),
                         "epyc.autokernel.discovery_planner_context.v2")
        semantic = receipt["hypothesis_portfolio"]["semantic_sha256"]
        self.assertEqual(planner.get("hypothesis_portfolio_sha256"), semantic)
        self.assertEqual(receipt["hypothesis_portfolio"]["file_sha256"], declared)
        self.assertEqual(receipt["hypothesis_portfolio"]["semantic_sha256"], semantic)
        self.assertEqual(receipt.get("config_sha256"), config["config_sha256"])
        self.assertFalse(receipt["inference_executed"])
        self.assertEqual(planner["template_surfaces_sha256"],
                         _content_hash(planner["template_surfaces"]))
        self.assertEqual(planner["template_registry_sha256"],
                         config["source_plan"]["experiment_template_registry_sha256"])
        self.assertEqual(receipt["template_surfaces"], planner["template_surfaces"])
        self.assertEqual(receipt["template_surfaces_sha256"],
                         planner["template_surfaces_sha256"])
        self.assertEqual(receipt["portfolio_dispatch_authority"],
                         planner["portfolio_dispatch_authority"])
        self.assertEqual(receipt["portfolio_dispatch_authority_sha256"],
                         _content_hash(planner["portfolio_dispatch_authority"]))

        eligible = _eligible(portfolio)
        self.assertTrue(eligible, "v2 corpus has no current-frame eligible hypothesis")
        _assert_eligible_projection(
            self, _planner_partition(planner, "eligible_hypotheses"), eligible)
        projected_ids = {row["hypothesis_id"] for row in
                         _planner_partition(planner, "eligible_hypotheses")}
        self.assertTrue(all(row["hypothesis_id"] not in projected_ids
                            for row in _hypotheses(portfolio)
                            if row not in eligible),
                        "planner eligible projection leaks an ineligible hypothesis")

    def test_vendored_evidence_tamper_refuses_public_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-evidence-gate-") as temporary:
            root, config_path, _config, _portfolio, planner = self.make_bundle(temporary)
            refs = [row for row in planner["hypothesis_evidence"].values()
                    if Path(row["path"]).is_relative_to(root)]
            self.assertTrue(refs, "portfolio evidence is not vendored into the bundle")
            victim = Path(refs[0]["path"])
            victim.chmod(0o600)
            victim.write_bytes(victim.read_bytes() + b"\n")
            with self.assertRaises((deployment.DeploymentConfigError,
                                    factory.DeploymentFactoryError)):
                parsed = deployment.load_deployment_config(config_path)
                factory.build_static_deployment_graph(parsed)

    def test_only_current_exact_frame_is_eligible(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-frame-gate-") as temporary:
            _root, _path, _config, portfolio, planner = self.make_bundle(temporary)
        current = portfolio.get("current_bundle")
        self.assertIsInstance(current, dict)
        frame_id = current.get("frame_id")
        frames = _rows(portfolio, "frames")
        matching = [row for row in frames if row.get("frame_id") == frame_id]
        self.assertEqual(len(matching), 1)
        frame = matching[0]
        self.assertIsInstance(frame, dict)
        expected = {
            "production_commit": PRODUCTION_COMMIT,
            "architecture": "gfx90a",
            "phase": "decode",
            "batch": 1,
            "generated_tokens": 128,
            "measurement_graphs": False,
            "target_runtime_graphs": True,
            "flash_attention": True,
        }
        for key, value in expected.items():
            self.assertEqual(frame.get(key), value, f"portfolio frame differs at {key}")
        eligible = _eligible(portfolio)
        self.assertTrue(eligible)
        for row in eligible:
            self.assertEqual(row.get("status"), "queued")
            self.assertIn(frame_id, row["target"]["frame_ids"])
            self.assertIn(frame_id, row["portability"]["source_frames"])
            self.assertTrue(row.get("falsifiers"))
            self.assertTrue(row.get("mechanism"))

    def test_complete_dnr_incumbent_and_hold_partitions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-memory-gate-") as temporary:
            _root, _path, _config, portfolio, _planner = self.make_bundle(temporary)
        dnr = _rows(portfolio, "do_not_repeat")
        hypotheses = _hypotheses(portfolio)
        incumbents = [row for row in hypotheses
                      if str(row.get("status", "")).endswith("incumbent")]
        ineligible = [row for row in hypotheses
                      if not row.get("current_bundle_eligibility", {}).get("eligible")]
        expected_dnr = (
            (("fattn", "flash"), ("singlecol", "single-column", "single column")),
            (("q5",), ("four-wave", "four wave", "nwarps4")),
            (("q8",), ("vec4", "vectorized")),
            (("rms",), ("two-wave", "two wave")),
            (("q6",), ("onewave", "one-wave", "one wave")),
            (("rope",), ("q4",), ("stack", "compose", "combined")),
            (("fattn", "flash"), ("vec", "vector"), ("gqa7", "gqa 7")),
        )
        for signature in expected_dnr:
            self.assertEqual(sum(_matches(row, signature) for row in dnr), 1,
                             f"DNR missing or duplicates {signature}")
        self.assertTrue(any(_matches(row, (("rope64", "rope 64", "rope"),))
                            for row in incumbents))
        self.assertTrue(any(_matches(row, (("q4",), ("onewave", "one-wave", "one wave")))
                            for row in incumbents))
        self.assertTrue(any(_matches(row, (("q8",), ("128",))) for row in dnr))
        self.assertTrue(ineligible, "portfolio omits ineligible/held hypotheses")
        for row in (*dnr, *incumbents):
            self.assertTrue(row.get("evidence_refs"),
                            "portfolio memory row lacks immutable evidence references")

    def test_q5_onewave_and_q4k_vecdotq_are_explicit_current_v2_work(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-mapping-gate-") as temporary:
            _root, _path, _config, portfolio, planner = self.make_bundle(temporary)
        hypotheses = _hypotheses(portfolio)
        q5 = [row for row in hypotheses
              if "q5" in row.get("hypothesis_id", "")
              and "onewave" in row.get("hypothesis_id", "")]
        self.assertEqual(len(q5), 1, "Q5 one-wave must be one unambiguous hypothesis")
        q5 = q5[0]
        self.assertEqual(q5["status"], "queued")
        lifecycle = q5.get("lifecycle", q5)
        self.assertEqual(lifecycle.get("maturity"), "correctness_validated")
        q5_text = json.dumps(q5, sort_keys=True).lower()
        self.assertIn(Q5_ONEWAVE_COMMIT, q5_text)
        self.assertTrue({"ev-q5-onewave-correctness-targeted",
                         "ev-q5-onewave-correctness-full"}.issubset(
                            set(q5.get("evidence_refs", ()))))
        self.assertIn("attribution", str(lifecycle.get("next_action", "")).lower())
        self.assertIn("screen", str(lifecycle.get("next_action", "")).lower())
        eligibility = q5["current_bundle_eligibility"]
        blockers = " ".join(eligibility["blocking_conditions"]).lower()
        if eligibility["eligible"]:
            self.assertIn("patch", q5_text)
            self.assertIn("manifest", q5_text)
            self.assertIn("continuation", q5_text)
            self.assertTrue(eligibility["template_ids"])
        else:
            self.assertTrue("carrier" in blockers or "continuation" in blockers)
            refusal_text = blockers + " " + eligibility.get("reason", "").lower()
            self.assertTrue("re-author" in refusal_text or "replan" in refusal_text)
            self.assertIn("correctness", refusal_text,
                          "ineligibility must forbid repeating completed correctness")

        branchless = [row for row in hypotheses
                      if row.get("hypothesis_id") == "akh-v2-q4k-branchless-scale-min"]
        self.assertEqual(len(branchless), 1)
        target = branchless[0]["target"]
        self.assertEqual(target["source_files"], ["ggml/src/ggml-cuda/vecdotq.cuh"])
        self.assertIn("vec_dot_q4_K_q8_1", target["source_symbols"])
        self.assertEqual(target["template_intent"], "cuda-vecdotq-v1")
        self.assertEqual(branchless[0]["priority"]["rank"], 1,
                         "Q4 branchless must retain its global portfolio rank")
        current_frame = next(row for row in portfolio["frames"]
                             if row["frame_id"] == portfolio["current_bundle"]["frame_id"])
        q4_hotspot = next(row for row in current_frame["hotspots"]
                          if "q4" in row["family"].lower())
        self.assertAlmostEqual(q4_hotspot["device_time_share_pct"], 3.8994, places=4)
        diagnostic_delta = 10.554
        current_ceiling = q4_hotspot["device_time_share_pct"] * diagnostic_delta / 100
        self.assertAlmostEqual(current_ceiling, 0.4115, places=3)
        self.assertLess(current_ceiling, 0.5)
        branchless_eligibility = branchless[0]["current_bundle_eligibility"]
        if branchless_eligibility["eligible"]:
            evidence_by_id = {row["evidence_id"]: row
                              for row in portfolio["evidence"]}
            reward_refs = [evidence_by_id[ref]
                           for ref in branchless[0]["evidence_refs"]
                           if ref in evidence_by_id
                           and all(token in json.dumps(
                               evidence_by_id[ref], sort_keys=True).lower()
                                   for token in ("35b", "q4", "reward"))]
            self.assertTrue(reward_refs,
                            "vecdotq template alone cannot authorize a proxy-only reward frame")
            self.assertTrue(all(_refs(row) for row in reward_refs))
        else:
            blockers = " ".join(branchless_eligibility["blocking_conditions"]).lower()
            self.assertEqual(branchless[0]["decision_policy"]["frame_id"],
                             "frame-v9-qwen35b-q4km-tg128")
            self.assertTrue("sealed base" in blockers or "clean replay" in blockers)
            self.assertTrue("dirty" in blockers or "diagnostic" in blockers)
            self.assertNotIn(branchless[0], _planner_partition(
                planner, "eligible_hypotheses"))

        bundle = portfolio["current_bundle"]
        self.assertEqual(bundle["template_catalog_version"], "gpu-source-templates-v2")
        current_ids = set(bundle["template_ids"])
        registry = factory._template_registry()
        self.assertEqual(registry.version, "gpu-source-templates-v2")
        self.assertEqual(current_ids, set(registry.templates))
        self.assertIn("cuda-vecdotq-v1", current_ids)
        _assert_eligible_projection(
            self, _planner_partition(planner, "eligible_hypotheses"),
            _eligible(portfolio))
        self.assertEqual(_planner_partition(planner, "do_not_repeat"),
                         [{key: value for key, value in row.items() if key != "title"}
                          for row in portfolio["do_not_repeat"]])
        self.assertEqual(_planner_partition(planner, "incumbents"),
                         [row for row in hypotheses
                          if str(row["status"]).endswith("incumbent")])
        self.assertEqual(_planner_partition(planner, "ineligible_hypotheses"),
                         [row for row in hypotheses
                          if not row["current_bundle_eligibility"]["eligible"]])

    def test_each_eligible_record_carries_its_own_floors_spread_and_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-policy-gate-") as temporary:
            _root, config_path, _config, portfolio, planner = self.make_bundle(temporary)
            config = factory.controller_config(
                deployment.load_deployment_config(config_path), dry_run=True)
            eligible = _eligible(portfolio)
            projected = {row["hypothesis_id"]: row for row in
                         _planner_partition(planner, "eligible_hypotheses")}
            self.assertEqual(set(projected), {row["hypothesis_id"] for row in eligible})
            for row in eligible:
                policy = row["decision_policy"]
                self.assertEqual(projected[row["hypothesis_id"]]["decision_policy"], policy)
                self.assertEqual(controller._portfolio_binding(
                    config, row)["decision_policy"], policy)
                continuation = controller._decision_floor(
                    policy, "continuation_floor_pct", 99.0)
                nomination = controller._decision_floor(
                    policy, "nomination_floor_pct", 99.0)
                minimum = controller._decision_floor(
                    policy, "min_replication_effect_pct", 99.0)
                spread = controller._decision_floor(
                    policy, "max_replication_spread_pct", 99.0)
                self.assertLessEqual(minimum, continuation)
                self.assertLessEqual(continuation, nomination)
                self.assertGreater(spread, 0.0)
                self.assertEqual(controller._required_replications(policy), 2)
                self.assertIn(policy["max_distinct_candidates"], range(1, 9))
                strong = max(nomination, continuation, minimum) + min(spread / 4, .0001)
                self.assertEqual(controller.classify_screen_series(
                    [strong, strong], continuation_floor=continuation,
                    nomination_floor=nomination,
                    min_replication_effect=minimum,
                    max_replication_spread=spread,
                    required_replications=2), "top_k_replicated_candidate")
                self.assertEqual(controller.classify_screen_series(
                    [minimum + spread * 2, minimum],
                    continuation_floor=continuation,
                    nomination_floor=nomination,
                    min_replication_effect=minimum,
                    max_replication_spread=spread,
                    required_replications=2), "inconclusive")
                state = {"iterations": [
                    {"portfolio_hypothesis_id": row["hypothesis_id"],
                     "source_manifest_sha256": f"{index:064x}"}
                    for index in range(policy["max_distinct_candidates"])],
                    "portfolio_terminals": {}}
                outcome = {"portfolio_hypothesis_id": row["hypothesis_id"],
                           "portfolio_decision_policy": policy,
                           "status": "screened_out"}
                controller._apply_portfolio_outcome(state, outcome)
                self.assertEqual(
                    state["portfolio_terminals"][row["hypothesis_id"]]["disposition"],
                    policy["terminal_rule"])


class PlannerCorpusGate(_BundleMixin, unittest.TestCase):
    class BoundaryReached(RuntimeError):
        pass

    def _planner_inputs(self, temporary: str):
        _root, config_path, _config, portfolio, planner_projection = self.make_bundle(temporary)
        parsed = deployment.load_deployment_config(config_path)
        graph = factory.build_static_deployment_graph(parsed)
        assignment = controller.AuthoringAssignment(
            campaign_id="ak-portfolio-gate", proposal_id="akp-portfolio-gate",
            candidate_id="akc-portfolio-gate",
            production_base_commit=PRODUCTION_COMMIT,
            instrument_commit=INSTRUMENT_COMMIT,
        )
        context = {
            "authority": controller.AUTHORITY, "turn": 1,
            "roster": controller.sealed_roster(),
            "planner_context": planner_projection,
            "authoring_assignment": assignment.to_dict(),
            "prior_results": [], "do_not_repeat": {},
        }
        return graph.adapters["planner"], context, portfolio

    @staticmethod
    def _mounted_source(prompt: dict, workspace: Path, relative: str) -> tuple[Path, dict]:
        package = prompt.get("reviewed_source_package", {})
        matches = [row for row in package.get("files", [])
                   if row.get("relative_path") == relative]
        if len(matches) != 1:
            raise AssertionError(
                f"reviewed-source manifest has {len(matches)} entries for {relative}")
        ref = matches[0]
        raw = Path(ref["workspace_path"])
        path = raw if raw.is_absolute() else workspace / raw
        try:
            path.resolve(strict=True).relative_to(workspace.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise AssertionError(f"reviewed source escaped mounted workspace: {relative}") from exc
        return path, ref

    def test_sol_prompt_binds_full_portfolio_and_exact_mounted_source_package(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-prompt-gate-") as temporary:
            planner, context, portfolio = self._planner_inputs(temporary)
            captured: dict = {}

            def stop(**kwargs):
                captured.update(kwargs)
                prompt = json.loads(kwargs["prompt"])
                workspace = Path(kwargs["workspace"])
                production = Path("/mnt/raid0/llm/llama.cpp")
                for relative in HIGH_VALUE_SOURCES:
                    path, ref = self._mounted_source(prompt, workspace, relative)
                    expected = (production / relative).read_bytes()
                    self.assertEqual(path.stat().st_mode & 0o777, 0o400)
                    self.assertEqual(path.read_bytes(), expected)
                    self.assertEqual(ref["sha256"], _sha(expected))
                raise self.BoundaryReached("model boundary captured")

            with mock.patch.object(factory.codex_container_actor, "run_actor",
                                   side_effect=stop), \
                    self.assertRaises(self.BoundaryReached):
                planner.plan(context=context, workspace=Path(temporary))

        prompt = json.loads(captured["prompt"])
        projection = prompt["context"]["planner_context"]
        projected = _planner_partition(projection, "eligible_hypotheses")
        _assert_eligible_projection(self, projected, _eligible(portfolio))
        for key in ("do_not_repeat", "incumbents", "ineligible_hypotheses"):
            self.assertTrue(_planner_partition(projection, key),
                            f"planner lacks complete {key} memory")
        self.assertEqual(projection.get("hypothesis_portfolio_sha256"),
                         context["planner_context"]["hypothesis_portfolio_sha256"])

    def test_mounted_source_tamper_refuses_after_actor_before_plan_load(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-source-tamper-gate-") as temporary:
            planner, context, _portfolio = self._planner_inputs(temporary)
            victim: dict[str, Path] = {}

            def tamper(**kwargs):
                prompt = json.loads(kwargs["prompt"])
                workspace = Path(kwargs["workspace"])
                path, _ref = self._mounted_source(
                    prompt, workspace, sorted(HIGH_VALUE_SOURCES)[0])
                victim["path"] = path
                path.chmod(0o600)
                path.write_bytes(path.read_bytes() + b"\n")
                return SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(factory.codex_container_actor, "run_actor",
                                   side_effect=tamper), \
                    mock.patch.object(controller, "_load_plan",
                                      side_effect=self.BoundaryReached) as load_plan, \
                    self.assertRaises(controller.DiscoveryControllerError):
                planner.plan(context=context, workspace=Path(temporary))
            self.assertIn("path", victim)
            load_plan.assert_not_called()

    def test_blind_patch_context_refuses_before_model_call(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-blind-patch-gate-") as temporary:
            planner, context, _portfolio = self._planner_inputs(temporary)
            blind = copy.deepcopy(context)
            blind["planner_context"]["reviewed_source_package_sha256"] = "b" * 64
            with mock.patch.object(
                    factory.codex_container_actor, "run_actor",
                    side_effect=AssertionError("blind patch reached Sol")) as actor, \
                    self.assertRaises(controller.DiscoveryControllerError):
                planner.plan(context=blind, workspace=Path(temporary))
            actor.assert_not_called()


class PortfolioSpendAuthorityGate(_BundleMixin, unittest.TestCase):
    """The portfolio authorizes questions; Sol still authors each concrete patch."""

    @staticmethod
    def _candidate(context: dict, row: dict, *, hypothesis_id: str,
                   mechanism_id: str | None = None,
                   regime: dict | None = None) -> controller.PlannedCandidate:
        assignment = context["authoring_assignment"]
        target = row.get("target", {})
        relative = next(iter(target.get("source_files", ())),
                        "ggml/src/ggml-cuda/mmvq.cu")
        symbols = tuple(target.get("source_symbols", (factory.source_candidate.FILE_SCOPE,)))
        patch = (f"diff --git a/{relative} b/{relative}\n"
                 f"--- a/{relative}\n+++ b/{relative}\n"
                 f"@@ -1 +1 @@ {symbols[0]}()\n-x\n+y\n").encode()
        change_class = row.get("mechanism", {}).get("facets", {}).get(
            "change_class", "arithmetic")
        manifest = factory.source_candidate.SourcePatchManifest(
            campaign_id=assignment["campaign_id"],
            proposal_id=assignment["proposal_id"],
            candidate_id=assignment["candidate_id"], source_tree="llama.cpp",
            production_base_commit=assignment["production_base_commit"],
            instrument_commit=assignment["instrument_commit"],
            change_class=change_class, declared_files=(relative,),
            declared_symbols={relative: symbols},
            mechanism_id=mechanism_id or row.get("mechanism", {}).get(
                "fingerprint_sha256", "arbitrary-new-mechanism"),
            patch_sha256=_sha(patch), patch_bytes=patch)
        falsifiers = row.get("falsifiers", ["no measured improvement"])
        return controller.PlannedCandidate(
            hypothesis_id=hypothesis_id,
            statement=row.get("statement", "arbitrary new discovery question"),
            falsifier=falsifiers[0], regime=regime or {
                "architecture": "gfx90a", "batch": 1, "phase": "decode"},
            proposal={
                "proposal_id": assignment["proposal_id"],
                "change_class": change_class,
                "change": {"files_and_symbols": {
                    relative: list(symbols)},
                    "estimated_diff_size": 2}},
            source_manifest=manifest,
            source_manifest_sha256=manifest.patch_bundle_sha256)

    class Planner:
        def __init__(self, build):
            self.build = build
        def attest(self):
            return {**controller.SOL, "runtime": PLANNER_RUNTIME}
        def plan(self, *, context, workspace):
            return self.build(context)

    class Critic:
        calls = 0
        def attest(self):
            return {**controller.FABLE5_CRITIC, "runtime": CRITIC_RUNTIME}
        def review(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("portfolio refusal reached critic")

    class Compute:
        calls = 0
        def __getattr__(self, _name):
            def called(*_args, **_kwargs):
                self.calls += 1
                raise AssertionError("portfolio refusal reached compute")
            return called

    def _assert_precritic_refusal(self, config, planner) -> None:
        critic, lease, screen = self.Critic(), self.Compute(), self.Compute()
        try:
            result = controller.run_controller(
                config, planner=planner, critic=critic, screener=screen, lease=lease)
        except controller.DiscoveryControllerError:
            result = None
        except AssertionError as exc:
            self.fail(str(exc))
        self.assertEqual(critic.calls, 0)
        self.assertEqual(lease.calls, 0)
        self.assertEqual(screen.calls, 0)
        if result is not None:
            self.assertTrue(result["iterations"])
            self.assertIn(result["iterations"][0]["status"], {
                "portfolio_refused", "authorization_refused"})

    def test_arbitrary_new_hypothesis_refuses_before_critic_or_compute(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-spend-gate-") as temporary:
            _root, config_path, _config, portfolio, _planner = self.make_bundle(temporary)
            eligible = _eligible(portfolio)
            self.assertTrue(eligible)
            config = factory.controller_config(
                deployment.load_deployment_config(config_path), dry_run=True)
            planner = self.Planner(lambda context: self._candidate(
                context, eligible[0], hypothesis_id="akh-arbitrary-unloaded-question"))
            self._assert_precritic_refusal(config, planner)

    def test_renamed_exact_dnr_mechanism_and_regime_refuses_precompute(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ak-portfolio-dnr-gate-") as temporary:
            _root, config_path, _config, portfolio, _planner = self.make_bundle(temporary)
            dnr = _rows(portfolio, "do_not_repeat")[0]
            eligible = _eligible(portfolio)[0]
            config = factory.controller_config(
                deployment.load_deployment_config(config_path), dry_run=True)
            mechanism = dnr["mechanism"]["fingerprint_sha256"]
            planner = self.Planner(lambda context: self._candidate(
                context, eligible, hypothesis_id="akh-renamed-retired-mechanism",
                mechanism_id=mechanism, regime=dict(dnr["regime"])))
            self._assert_precritic_refusal(config, planner)


class DispatchAndTemplateGate(_BundleMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        receipt = _json(TRACE_RECEIPT)
        cls.trace_path = Path(receipt["timestamp_csv"])
        if _sha(cls.trace_path.read_bytes()) != receipt["timestamp_csv_sha256"]:
            raise AssertionError("sealed Qwen trace bytes changed")
        cls.rows = evidence._load_dispatches(cls.trace_path)
        if len(cls.rows) != 59_925:
            raise AssertionError(f"sealed Qwen trace row count changed: {len(cls.rows)}")

    def test_every_current_frame_template_accepts_real_qwen_anchor_dispatch(self) -> None:
        failures = []
        for name, template in factory._template_registry().templates.items():
            try:
                evidence._reduce_arm(
                    self.rows, exact=template.dispatch.anchor_exact,
                    forbidden=template.dispatch.anchor_forbidden,
                    invariants=template.dispatch.invariants,
                )
            except evidence.EvidenceProducerError as exc:
                failures.append(f"{name}: {exc}")
        self.assertEqual(failures, [], " ; ".join(failures))

    def test_anchor_contract_handles_large_counts_and_raw_rocprof_names(self) -> None:
        contracts = [template.dispatch.anchor_exact
                     for template in factory._template_registry().templates.values()]
        expectations = [item for group in contracts for item in group]
        matched_counts = set()
        matched_void = False
        for item in expectations:
            hits = evidence._matching(self.rows, item.kernel_pattern)
            if hits:
                matched_counts.add(len(hits))
                matched_void |= any(str(row["kernel"]).startswith("void ") for row in hits)
        self.assertTrue({13803, 18705, 6321}.issubset(matched_counts), matched_counts)
        self.assertTrue(matched_void, "dispatch policy does not normalize/match raw 'void ' names")
        self.assertGreater(max(item.calls for item in expectations), 4096)

    def test_portfolio_preserves_multirow_q5_q8_truth_and_registry_cross_binding(self) -> None:
        q5_trace = [row for row in self.rows
                    if "mul_mat_vec_q<(ggml_type)6," in row["kernel"]]
        q8_trace = [row for row in self.rows if "quantize_q8_1(" in row["kernel"]]
        q5_actual: dict[tuple[int, int, int, str], int] = {}
        for row in q5_trace:
            key = (row["grid"], row["workgroup"], row["lds"],
                   "false,true" if ", false, true>" in row["kernel"] else "true,true")
            q5_actual[key] = q5_actual.get(key, 0) + 1
        self.assertEqual(q5_actual, {
            (57344, 128, 1024, "true,true"): 6063,
            (8192, 128, 1024, "true,true"): 4644,
            (311296, 128, 1024, "true,true"): 3096,
            (57344, 128, 512, "false,true"): 129,
        })
        q8_actual: dict[tuple[int, int, int], int] = {}
        for row in q8_trace:
            key = (row["grid"], row["workgroup"], row["lds"])
            q8_actual[key] = q8_actual.get(key, 0) + 1
        self.assertEqual(q8_actual, {
            (1024, 256, 0): 15609,
            (5120, 256, 0): 3096,
        })

        with tempfile.TemporaryDirectory(prefix="ak-multirow-dispatch-gate-") as temporary:
            _root, _path, _config, portfolio, planner = self.make_bundle(temporary)
        q5 = next(row for row in _eligible(portfolio)
                  if "q5" in row.get("hypothesis_id", ""))
        q8 = next(row for row in _eligible(portfolio)
                  if "q8" in row.get("hypothesis_id", ""))

        q5_records = _geometry_records(q5)
        q5_expected = {_geometry(row) for path, row in q5_records
                       if "exclude" not in "/".join(path).lower()
                       and "forbid" not in "/".join(path).lower()
                       and "tail" not in "/".join(path).lower()
                       and "false,true" not in json.dumps(row).replace(" ", "").lower()}
        self.assertEqual(q5_expected, {
            (6063, 57344, 128, 1024),
            (4644, 8192, 128, 1024),
            (3096, 311296, 128, 1024),
        })
        self.assertEqual(sum(row[0] for row in q5_expected), 13803)
        q5_excluded = [(path, row) for path, row in q5_records
                       if any(token in "/".join(path).lower()
                              for token in ("exclude", "forbid", "tail"))]
        self.assertIn((129, 57344, 128, 512),
                      {_geometry(row) for _path, row in q5_excluded})
        self.assertTrue(any("false" in json.dumps(row).lower()
                            for _path, row in q5_excluded))
        q5_routed = {_routed_geometry(row) for path, row in q5_records
                     if "signatures" in path and "excluded_signatures" not in path}

        q8_expected = {_geometry(row) for path, row in _geometry_records(q8)
                       if not any(token in "/".join(path).lower()
                                  for token in ("exclude", "forbid", "tail"))}
        self.assertEqual(q8_expected, {
            (15609, 1024, 256, 0),
            (3096, 5120, 256, 0),
        })
        self.assertEqual(sum(row[0] for row in q8_expected), 18705)
        q8_routed = {_routed_geometry(row) for path, row in _geometry_records(q8)
                     if "signatures" in path and "excluded_signatures" not in path}

        registry = factory._template_registry()
        q5_template = registry.templates[
            q5["current_bundle_eligibility"]["template_ids"][0]]
        q8_template = registry.templates[
            q8["current_bundle_eligibility"]["template_ids"][0]]
        q5_registry = {_geometry(vars(item))
                       for item in q5_template.dispatch.anchor_exact}
        q8_registry = {_geometry(vars(item))
                       for item in q8_template.dispatch.anchor_exact}
        self.assertTrue(q5_expected.issubset(q5_registry),
                        "portfolio Q5 rows are absent from deployed registry anchors")
        self.assertEqual(q8_registry, q8_expected,
                         "portfolio Q8 rows drift from deployed registry anchors")
        excluded_registry = {
            _geometry(row) for row in
            q5_template.semantics.get("planner_target_exclusions", [])}
        self.assertEqual(excluded_registry, {(129, 57344, 128, 512)})
        registry_routes = {
            _routed_geometry(vars(item))
            for item in (*q5_template.dispatch.anchor_exact,
                         *q8_template.dispatch.anchor_exact)}
        self.assertTrue(q5_routed.issubset(registry_routes))
        self.assertTrue(q8_routed.issubset(registry_routes))

        dispatch_authority = planner.get("portfolio_dispatch_authority")
        self.assertIsInstance(dispatch_authority, dict,
                              "planner context lacks deployment-derived dispatch authority")
        self.assertEqual({_routed_geometry(row) for row in
                          dispatch_authority[q5["hypothesis_id"]]}, q5_routed)
        self.assertEqual({_routed_geometry(row) for row in
                          dispatch_authority[q8["hypothesis_id"]]}, q8_routed)
        surfaces = planner.get("template_surfaces")
        self.assertIsInstance(surfaces, dict)
        self.assertEqual({_routed_geometry(row) for row in
                          surfaces[q5_template.template_id]["dispatch_signatures"]},
                         q5_routed)
        self.assertEqual({_routed_geometry(row) for row in
                          surfaces[q5_template.template_id]["excluded_signatures"]},
                         {_routed_geometry(row) for _path, row in q5_excluded})

        vecdot = registry.templates["cuda-vecdotq-v1"]
        aliased_geometry = [row for row in vecdot.dispatch.anchor_exact
                            if _geometry(vars(row)) == (1548, 114688, 128, 512)]
        self.assertEqual(len(aliased_geometry), 2)
        self.assertEqual(len({row.signature for row in aliased_geometry}), 2)
        self.assertEqual(len({row.kernel_pattern for row in aliased_geometry}), 2)
        self.assertTrue(any("type\\)12" in row.kernel_pattern
                            or "type\\)12" in row.kernel_pattern.replace("ggml_", "")
                            for row in aliased_geometry))
        self.assertTrue(any("type\\)14" in row.kernel_pattern
                            or "type\\)14" in row.kernel_pattern.replace("ggml_", "")
                            for row in aliased_geometry))
        registry_text = json.dumps({name: {
            "anchor_exact": [vars(row) for row in template.dispatch.anchor_exact],
            "anchor_forbidden": [vars(row) for row in template.dispatch.anchor_forbidden],
            "invariants": [vars(row) for row in template.dispatch.invariants],
            "semantics": dict(template.semantics),
        } for name, template in registry.templates.items()}, sort_keys=True).lower()
        for token in ("129", "57344", "512", "false", "true"):
            self.assertIn(token, registry_text,
                          "registry does not explicitly bind the excluded Q5 tail")

    def test_high_value_source_surfaces_have_reviewed_templates(self) -> None:
        templates = factory._template_registry().templates
        covered = {path for template in templates.values() for path in template.allowed_files}
        self.assertTrue(HIGH_VALUE_SOURCES.issubset(covered),
                        f"missing high-value templates: {sorted(HIGH_VALUE_SOURCES-covered)}")
        for relative in HIGH_VALUE_SOURCES:
            owners = [template for template in templates.values()
                      if relative in template.allowed_files]
            self.assertTrue(owners)
            self.assertTrue(all(template.semantics.get("workload") == "decode_tg128"
                                for template in owners))

    def test_candidate_dispatch_binding_preserves_every_literal_geometry_field(self) -> None:
        registry = factory._template_registry()
        for template in registry.templates.values():
            anchor = template.dispatch.anchor_exact[0]
            matches = evidence._matching(self.rows, anchor.kernel_pattern)
            self.assertTrue(matches, f"{template.template_id} has no real trace anchor")
            literal = matches[0]["kernel"]
            expectation = controller.BoundedDispatchExpectation(
                route_id=anchor.signature, kernel_name=literal,
                calls=anchor.calls, grid=anchor.grid,
                workgroup=anchor.workgroup, lds_bytes=anchor.lds_bytes)
            intent = controller.GpuSourceExperimentIntent(
                template_id=template.template_id,
                target_surface=template.target_surface,
                target_symbol=template.target_symbol,
                correctness_id=template.correctness_id,
                dispatch_id=template.dispatch_id,
                expected_dispatch=(expectation,))
            bound = template.bind_dispatch(intent).candidate_exact
            self.assertEqual(len(bound), 1)
            candidate = bound[0]
            self.assertEqual((candidate.calls, candidate.grid, candidate.workgroup,
                              candidate.lds_bytes, candidate.blocks_per_call),
                             (expectation.calls, expectation.grid, expectation.workgroup,
                              expectation.lds_bytes,
                              expectation.grid // expectation.workgroup))


class BalancedOrderAndLegacyGate(unittest.TestCase):
    def test_discovery_arm_order_is_deterministic_randomized_and_replication_balanced(self) -> None:
        starts = set()
        deployment_sha = "a" * 64
        for index in range(8):
            manifest_sha = f"{index + 1:064x}"
            seed1, schedule1 = factory._arm_order_schedule(
                deployment_config_sha256=deployment_sha,
                source_manifest_sha256=manifest_sha, repetition=1)
            seed2, schedule2 = factory._arm_order_schedule(
                deployment_config_sha256=deployment_sha,
                source_manifest_sha256=manifest_sha, repetition=2)
            s1, s2 = tuple(schedule1.split(",")), tuple(schedule2.split(","))
            self.assertEqual(seed1, seed2)
            self.assertEqual((seed1, schedule1), factory._arm_order_schedule(
                deployment_config_sha256=deployment_sha,
                source_manifest_sha256=manifest_sha, repetition=1))
            self.assertEqual(set(s1), {"anchor", "candidate"})
            self.assertEqual(s2, tuple(reversed(s1)), "S2 must oppose S1")
            starts.add(s1[0])
        self.assertEqual(starts, {"anchor", "candidate"},
                         "seed never randomizes the first arm")

    def test_old_bundle_and_v3_state_refuse_before_actor_or_compute(self) -> None:
        with self.assertRaises(deployment.DeploymentConfigError):
            deployment.load_deployment_config(OLD_BUNDLE)

        planner_runtime = {
            "kind": "docker_workspace_bind_only", "docker_path": "/docker",
            "docker_sha256": "a" * 64, "image_id": "image",
            "codex_native_sha256": "a" * 64, "code_mode_host_sha256": "a" * 64,
            "ca_certificate_sha256": "a" * 64, "writable_host_binds": ["/workspace"],
            "host_network_mode": "docker_bridge",
        }
        critic_runtime = {
            "kind": "claude_cli_structured_critic", "provider": "claude",
            "model": "claude-fable-5", "effort": "high",
            "wrapper_path": "/sealed/claude", "wrapper_sha256": "a" * 64,
            "argv_policy_sha256": "a" * 64,
            "auth_staging_policy": "ephemeral_0600_copy_no_secret_receipt",
        }

        class Planner:
            def attest(self):
                return {**controller.SOL, "runtime": planner_runtime}
            def plan(self, **_kwargs):
                raise AssertionError("old state reached planner")

        class Critic:
            def attest(self):
                return {**controller.FABLE5_CRITIC, "runtime": critic_runtime}
            def review(self, *_args, **_kwargs):
                raise AssertionError("old state reached critic")

        class Compute:
            def __getattr__(self, _name):
                raise AssertionError("old state reached compute")

        with tempfile.TemporaryDirectory(prefix="ak-old-portfolio-state-gate-") as temporary:
            output = Path(temporary) / "out"
            output.mkdir()
            for schema in ("epyc.autokernel.discovery_controller.v3",
                           "epyc.autokernel.discovery_controller.v4"):
                old = {
                    "schema": schema, "authority": controller.AUTHORITY,
                    "roster": controller.sealed_roster(), "iterations": [],
                    "next": 1, "complete": False,
                }
                old["state_sha256"] = controller._sha(old)
                (output / "state.json").write_text(
                    json.dumps(old, sort_keys=True), encoding="utf-8")
                try:
                    controller.run_controller(
                        controller.ControllerConfig(output, 1), planner=Planner(),
                        critic=Critic(), screener=Compute(), lease=Compute())
                except controller.DiscoveryControllerError:
                    pass
                except AssertionError as exc:
                    self.fail(str(exc))
                else:
                    self.fail(f"old controller state was accepted: {schema}")


if __name__ == "__main__":
    unittest.main()
