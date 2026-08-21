from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import cpu_inference_controller as C


class FakeRunner:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run(self, args):
        self.calls.append(tuple(args))
        value = self.results.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class NeverRunner:
    def run(self, _args):
        raise AssertionError("sealed or ambiguous operation replayed")


class CpuInferenceControllerTests(unittest.TestCase):
    def setUp(self):
        data_root = Path(__file__).resolve().parents[4] / "data"
        data_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            dir=data_root, prefix="ak-cpu-controller-test-")
        self.root = Path(self.temp.name)
        self.output = self.root / "durable-output"
        self.controller_id = "ak-cpu-controller-test"

    def tearDown(self):
        self.temp.cleanup()

    def _material(self, prefix):
        files = {}
        for name in ("proposal.json", "source-patch.json", "prerequisite.json",
                     "envelope.json", "hypotheses.json", "model.gguf"):
            path = self.root / f"{prefix}-{name}"
            path.write_bytes(f"{prefix}:{name}\n".encode())
            files[name] = path
        calibration = self.root / f"{prefix}-calibration"
        calibration.mkdir(exist_ok=True)
        (calibration / "campaign_declaration.json").write_text(
            json.dumps({"candidate": prefix}))
        return files, calibration

    def _candidate(self, index):
        candidate_id = f"akc-cpu-{index}"
        hypothesis_id = f"akh-cpu-{index}"
        files, calibration = self._material(str(index))
        args = [
            "--campaign-id", self.controller_id,
            "--candidate-id", candidate_id,
            "--candidate", f"reviewed-cpu-patch-{index}",
            "--proposal-manifest", str(files["proposal.json"]),
            "--source-patch-manifest", str(files["source-patch.json"]),
            "--source-prerequisite-package", str(files["prerequisite.json"]),
            "--calibration-bundle", str(calibration),
            "--physical-envelope", str(files["envelope.json"]),
            "--backend", "llama_cpu",
            "--blocks", "2",
            "--recipe", "t1b.llama_cpu.llama_bench_prefill.v1",
            "--model", str(files["model.gguf"]),
            "--reps", "5",
            "--nominal-khz", "3000000",
            "--journal-root", str(self.output / "campaign-journal"),
            "--hypothesis", hypothesis_id,
            "--hypothesis-store", str(files["hypotheses.json"]),
        ]
        artifacts = []
        for path in sorted(
                [files["proposal.json"], files["source-patch.json"],
                 files["prerequisite.json"], calibration, files["envelope.json"],
                 files["model.gguf"], files["hypotheses.json"]], key=str):
            kind = "tree" if path == calibration else "file"
            artifacts.append({"path": str(path), "kind": kind,
                              "sha256": C._hash_path(path, kind)})
        reviewed_spec = {
            "campaign_id": self.controller_id, "candidate_id": candidate_id,
            "candidate_ref": f"reviewed-cpu-patch-{index}",
            "backend": "llama_cpu",
            "model_sha256": C._hash_path(files["model.gguf"], "file"),
            "production_commit": C.campaign.PRODUCTION_COMMIT,
            "measurement_commit": C.campaign.MEASUREMENT_COMMIT,
            "blocks_precommitted": 2,
            "recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1",
            "metric": "prefill_tokens_per_s", "drift_bound": 0.1,
            "drift_bound_evidence": "fixture-drift", "calibration": {
                "contribution_floor": 0.03, "b_min_blocks": 2,
                "max_blocks": 10, "noise_floor_phi": 0.01, "mde": 0.03,
                "evidence_ref": "fixture", "anchor_motion_bound": 0.1,
                "anchor_motion_window_blocks": 2,
                "anchor_motion_evidence_ref": "fixture-drift",
            },
            "model": str(files["model.gguf"]), "reps": 5,
            "n_gen": 128, "n_prompt": 512, "fresh_pairs_per_block": 1,
            "suite_seed": 1, "schedule_seed": 2, "holdout_selection_seed": 3,
            "matched_experiment_id": None,
            "t0_ops": ["MUL_MAT", "MUL_MAT_ID"], "devices": [],
            "device_names": [], "device_index": 0, "n_gpu_layers": 99,
            "cpu_list": "0-95", "worktree": str(self.root / f"worktree-{index}"),
            "build_dir": str(self.root / f"build-{index}"),
            "journal_root": str(self.output / "campaign-journal"),
            "created_at": C._RUNTIME_CREATED_AT,
            "proposal": {"proposal_id": f"proposal-{index}",
                         "schema": "fixture", "representation_frame_sha256": "0" * 64},
            "source_patch_bundle_sha256": "1" * 64,
            "source_prerequisite_package_sha256": "2" * 64,
            "fresh_source_prerequisite_plan_sha256": None,
            "least_commitment_capture_plan_sha256": None,
            "physical_envelope": {"fixture": "bound"}, "ranked_units": [],
            "hypothesis": {"bound": True, "hypothesis_id": hypothesis_id,
                           "falsifier": "fixture", "authorization": {
                               "hypothesis_id": hypothesis_id,
                               "falsifier": "fixture",
                               "falsifier_source": "stated",
                               "origin": "operator",
                               "purpose": f"fixture claim {hypothesis_id}",
                               "authorized_by": "campaign.py",
                               "authorized_at": C._RUNTIME_AUTHORIZED_AT,
                               "ledger_seq": C._RUNTIME_LEDGER_SEQ,
                               "do_not_repeat_outcome": "PASS",
                               "do_not_repeat_reasons": [],
                               "campaign_id": self.controller_id,
                               "entry_evidence_grade": "design_prior",
                           }},
            "claim_purpose": f"fixture claim {hypothesis_id}",
            "anchor": {"repo": C.campaign.PRODUCTION_REPO,
                       "branch": C.campaign.PRODUCTION_BRANCH,
                       "expected_commit": C.campaign.PRODUCTION_COMMIT},
            "measurement_instrument": {
                "repo": C.campaign.MEASUREMENT_REPO,
                "branch": C.campaign.MEASUREMENT_BRANCH,
                "expected_commit": C.campaign.MEASUREMENT_COMMIT,
                "parent_production_commit": C.campaign.PRODUCTION_COMMIT,
                "build_root": C.campaign.MEASUREMENT_BUILD_ROOT,
            },
        }
        body = {"schema": C.CANDIDATE_SCHEMA, "candidate_id": candidate_id,
                "hypothesis_id": hypothesis_id, "campaign_args": args,
                "artifacts": artifacts, "reviewed_result_spec": reviewed_spec}
        return {**body, "candidate_sha256": C._sha(body)}

    def _manifest(self, count=2, budget=10):
        body = {"schema": C.SCHEMA, "controller_id": self.controller_id,
                "output_root": str(self.output),
                "max_scientific_attempts": budget,
                "candidates": [self._candidate(index + 1)
                               for index in range(count)]}
        value = {**body, "manifest_sha256": C._sha(body)}
        path = self.root / "controller.json"
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        return path, value

    def _write_manifest(self, path, value):
        for candidate in value["candidates"]:
            body = {key: item for key, item in candidate.items()
                    if key != "candidate_sha256"}
            candidate["candidate_sha256"] = C._sha(body)
        body = {key: item for key, item in value.items()
                if key != "manifest_sha256"}
        value["manifest_sha256"] = C._sha(body)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def _replace_output(self, path, value, output):
        old = value["output_root"]
        value["output_root"] = str(output)
        for candidate in value["candidates"]:
            args = candidate["campaign_args"]
            index = args.index("--journal-root") + 1
            self.assertEqual(args[index], str(Path(old) / "campaign-journal"))
            args[index] = str(output / "campaign-journal")
            candidate["reviewed_result_spec"]["journal_root"] = args[index]
        self._write_manifest(path, value)

    def _rewrite_completed_receipt(self, config, mutate):
        state_path = config.output_root / "state.json"
        state = json.loads(state_path.read_text())
        receipt_path = Path(state["iterations"][0]["receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        mutate(receipt["campaign_result"])
        receipt["campaign_result_sha256"] = C._sha(receipt["campaign_result"])
        receipt["receipt_sha256"] = C._sha({
            key: item for key, item in receipt.items()
            if key != "receipt_sha256"})
        encoded = json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n"
        receipt_path.write_bytes(encoded)
        state["iterations"][0]["receipt_file_sha256"] = C.hashlib.sha256(
            encoded).hexdigest()
        state["iterations"][0]["result_sha256"] = receipt[
            "campaign_result_sha256"]
        state["state_sha256"] = C._sha({
            key: item for key, item in state.items() if key != "state_sha256"})
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n")

    def _result(self, candidate, state, *, keep=None, ok=True):
        scientific = state in {C.campaign.STATE_DECIDED,
                               C.campaign.STATE_T0_FAILED}
        releases = ([{"name": "cpu_region_claim", "released": True,
                      "detail": "released"},
                     {"name": "campaign_worktree", "released": True,
                      "detail": "released"}] if scientific else [])
        t0 = None
        decision = None
        pairs = []
        if state == C.campaign.STATE_DECIDED:
            t0 = {"all_pass": True, "report_ref": "t0.json",
                  "gates": [["output", C.campaign.schemas.PASS, []]]}
            candidate_value = 11.0 if keep else 9.0
            pair_objects = (
                C.campaign.Pair(0, 10.0, candidate_value, "anchor_first"),
                C.campaign.Pair(1, 10.0, candidate_value, "candidate_first"),
            )
            pairs = [row.to_dict() for row in pair_objects]
            derived = C.campaign.decide(
                pair_objects, t0=C.campaign.T0Outcome(all_pass=True),
                blocks_precommitted=2, drift_bound=0.1,
                contribution_floor=0.03,
                calibration_evidence_ref="fixture")
            self.assertIs(derived.keep, keep)
            decision = derived.to_dict()
        elif state == C.campaign.STATE_T0_FAILED:
            t0 = {"all_pass": False, "report_ref": "t0.json",
                  "gates": [["output", C.campaign.schemas.FAIL,
                             ["output mismatch"]]]}
        result_spec = json.loads(json.dumps(candidate["reviewed_result_spec"]))
        result_spec["created_at"] = "2026-08-21T00:00:00Z"
        result_spec["hypothesis"]["authorization"]["authorized_at"] = (
            "2026-08-21T00:00:00Z")
        result_spec["hypothesis"]["authorization"]["ledger_seq"] = 7
        return {
            "schema": "epyc.autokernel.campaign_result.v1",
            "campaign_id": self.controller_id,
            "candidate_id": candidate["candidate_id"],
            "spec": result_spec,
            "state": state,
            "steps": [],
            "t0": t0,
            "decision": decision,
            "pairs": pairs,
            "preflight": ({"outcome": C.campaign.schemas.FAIL,
                           "reasons": ["fixture preflight refusal"]}
                          if state == C.campaign.STATE_PREFLIGHT_REFUSED else
                          {"outcome": "PASS", "reasons": []}),
            "releases": releases,
            "production_unchanged": {"outcome": "PASS", "reasons": []},
            "executed": True, "screening_only": False,
            "non_promotable": False, "journal_error": None,
            "screening_report": None,
            "ok": (False if state in {C.campaign.STATE_PREFLIGHT_REFUSED,
                                       C.campaign.STATE_ERROR} else ok),
            "error": ("fixture preflight refusal"
                      if state == C.campaign.STATE_PREFLIGHT_REFUSED else
                      "RuntimeError: fixture" if state == C.campaign.STATE_ERROR else None),
            "grammar": "SEARCH RECORD, NOT A CLAIM",
        }

    def test_strict_manifest_binds_artifacts_and_cpu_full_campaign_surface(self):
        path, raw = self._manifest()
        config = C.ControllerManifest.load(path)
        self.assertEqual(len(config.candidates), 2)
        self.assertEqual(config.candidates[0].parsed_args["--backend"], "llama_cpu")
        self.assertIn("--source-prerequisite-package",
                      config.candidates[0].parsed_args)
        bad = json.loads(json.dumps(raw))
        bad["candidates"][0]["campaign_args"][
            bad["candidates"][0]["campaign_args"].index("llama_cpu")] = "llama_gpu"
        body = {key: value for key, value in bad["candidates"][0].items()
                if key != "candidate_sha256"}
        bad["candidates"][0]["candidate_sha256"] = C._sha(body)
        outer = {key: value for key, value in bad.items()
                 if key != "manifest_sha256"}
        bad["manifest_sha256"] = C._sha(outer)
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "backend llama_cpu"):
            C.ControllerManifest.load(path)

    def test_back_to_back_keep_revert_t0_and_infrastructure_are_distinct(self):
        path, raw = self._manifest(count=4, budget=10)
        config = C.ControllerManifest.load(path)
        results = [
            (0, self._result(raw["candidates"][0], C.campaign.STATE_DECIDED,
                             keep=True)),
            (0, self._result(raw["candidates"][1], C.campaign.STATE_DECIDED,
                             keep=False)),
            (0, self._result(raw["candidates"][2], C.campaign.STATE_T0_FAILED,
                             keep=None)),
            (1, self._result(raw["candidates"][3],
                             C.campaign.STATE_PREFLIGHT_REFUSED, keep=None)),
        ]
        runner = FakeRunner(results)
        state = C.run_controller(config, runner=runner)
        self.assertTrue(state["complete"])
        self.assertEqual(state["terminal_reason"], "candidate_portfolio_exhausted")
        self.assertEqual(state["scientific_attempts"], 3)
        self.assertEqual([row["classification"] for row in state["iterations"]],
                         ["candidate", "screened_out", "correctness_falsified",
                          "infrastructure_ambiguous"])
        self.assertEqual([row["keep"] for row in state["iterations"]],
                         [True, False, False, None])
        self.assertTrue(all(call[-3:] == (
            "--execute", "--i-hold-the-host", "--json")
                            for call in runner.calls))
        self.assertTrue(all("--screening-only" not in call
                            for call in runner.calls))
        before = (self.output / "state.json").read_bytes()
        reopened = C.run_controller(config, runner=NeverRunner())
        self.assertEqual(reopened, state)
        self.assertEqual((self.output / "state.json").read_bytes(), before)

    def test_science_budget_stops_before_later_candidate(self):
        path, raw = self._manifest(count=3, budget=2)
        config = C.ControllerManifest.load(path)
        runner = FakeRunner([
            (0, self._result(raw["candidates"][0], C.campaign.STATE_DECIDED,
                             keep=False)),
            (0, self._result(raw["candidates"][1], C.campaign.STATE_T0_FAILED)),
        ])
        state = C.run_controller(config, runner=runner)
        self.assertEqual(state["scientific_attempts"], 2)
        self.assertEqual(state["next_index"], 2)
        self.assertEqual(state["terminal_reason"], "scientific_budget_exhausted")
        self.assertEqual(len(runner.calls), 2)

    def test_restart_reconciles_campaign_terminal_without_replay(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        store = C.StateStore(config)
        state = store.load()
        candidate = config.candidates[0]
        state["inflight"] = {"candidate_index": 0,
                             "candidate_id": candidate.candidate_id,
                             "candidate_sha256": candidate.candidate_sha256,
                             "operation_key": candidate.operation_key(
                                 self.controller_id),
                             "started_at": C._now()}
        store.save(state, "candidate_started")
        result = self._result(raw["candidates"][0], C.campaign.STATE_DECIDED,
                              keep=True)
        book = C.journal.Journal(str(self.output / "campaign-journal"),
                                 campaign_id=self.controller_id)
        book.initialize()
        book.append(C.journal.KIND_STOP_STATE, {
            "state": "decided", "campaign_id": self.controller_id,
            "result": result})
        recovered = C.run_controller(config, runner=NeverRunner())
        self.assertTrue(recovered["complete"])
        self.assertEqual(recovered["iterations"][0]["classification"], "candidate")
        self.assertTrue(Path(recovered["iterations"][0]["receipt_path"]).is_file())

    def test_restart_without_terminal_is_ambiguity_and_never_replays_key(self):
        path, _raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        store = C.StateStore(config)
        state = store.load()
        candidate = config.candidates[0]
        state["inflight"] = {"candidate_index": 0,
                             "candidate_id": candidate.candidate_id,
                             "candidate_sha256": candidate.candidate_sha256,
                             "operation_key": candidate.operation_key(
                                 self.controller_id),
                             "started_at": C._now()}
        store.save(state, "candidate_started")
        recovered = C.run_controller(config, runner=NeverRunner())
        row = recovered["iterations"][0]
        self.assertEqual(row["classification"], "infrastructure_ambiguous")
        self.assertFalse(row["scientific_budget_spent"])
        self.assertEqual(row["reason_code"],
                         "inflight_operation_has_no_sealed_terminal")

    def test_artifact_tamper_refuses_before_runner(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        model = Path(C._parse_args(tuple(
            raw["candidates"][0]["campaign_args"]))["--model"])
        model.write_bytes(b"tampered")
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "artifact identity changed"):
            C.run_controller(config, runner=NeverRunner())
        self.assertFalse((self.output / "state.json").exists())

    def test_unreleased_claim_or_nonfinite_result_refuses(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        result = self._result(raw["candidates"][0], C.campaign.STATE_DECIDED,
                              keep=True)
        result["releases"][0]["released"] = False
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "release of every acquired"):
            C.run_controller(config, runner=FakeRunner([(0, result)]))
        # The candidate stays inflight; no false scientific disposition exists.
        stored = C.StateStore(config).load()
        self.assertIsNotNone(stored["inflight"])
        self.assertEqual(stored["scientific_attempts"], 0)

    def test_recomputed_state_hash_cannot_authorize_malformed_science(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        C.run_controller(config, runner=FakeRunner([(
            0, self._result(raw["candidates"][0],
                            C.campaign.STATE_DECIDED, keep=True))]))
        state_path = self.output / "state.json"
        state = json.loads(state_path.read_text())
        state["iterations"][0]["scientific_budget_spent"] = False
        state["scientific_attempts"] = 0
        state["state_sha256"] = C._sha({
            key: value for key, value in state.items()
            if key != "state_sha256"})
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "scientific CPU iteration"):
            C.run_controller(config, runner=NeverRunner())

    def test_complete_restart_revalidates_private_result_receipt(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        state = C.run_controller(config, runner=FakeRunner([(
            0, self._result(raw["candidates"][0],
                            C.campaign.STATE_DECIDED, keep=False))]))
        receipt = Path(state["iterations"][0]["receipt_path"])
        receipt.write_bytes(receipt.read_bytes() + b" ")
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "receipt file hash changed"):
            C.run_controller(config, runner=NeverRunner())

    def test_nonfinite_campaign_result_is_never_sealed(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        result = self._result(raw["candidates"][0],
                              C.campaign.STATE_DECIDED, keep=True)
        result["decision"]["median_relative"] = float("nan")
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "canonical finite JSON"):
            C.run_controller(config, runner=FakeRunner([(0, result)]))

    def test_inprocess_runner_injects_only_reviewed_campaign_entrypoint(self):
        seen = {}
        expected = {"schema": "epyc.autokernel.campaign_result.v1"}
        def campaign_main(args, *, out):
            seen["args"] = tuple(args)
            out.write(json.dumps(expected))
            return 1
        with mock.patch.object(C.campaign, "main", side_effect=campaign_main):
            code, value = C.InProcessCampaignRunner().run(
                ("--backend", "llama_cpu", "--execute", "--json"))
        self.assertEqual(code, 1)
        self.assertEqual(value, expected)
        self.assertEqual(seen["args"],
                         ("--backend", "llama_cpu", "--execute", "--json"))

    def test_validate_only_is_default_and_creates_no_controller_state(self):
        path, _raw = self._manifest(count=2)
        output = io.StringIO()
        with redirect_stdout(output):
            code = C.main(("--manifest", str(path)))
        self.assertEqual(code, 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "validated")
        self.assertFalse(value["inference_executed"])
        self.assertFalse((self.output / "state.json").exists())

    def test_validate_only_parses_numeric_argv_and_strict_ids(self):
        for mutation, message in (
                (("--nominal-khz", "not-an-int"), "do not parse"),
                (("--blocks", "1"), "at least 2"),
                (("--reps", "0"), "positive"),
                (("--candidate-id", "akc-BAD"), "identity is malformed"),
                (("--hypothesis", "akh-bad/escape"), "identity is malformed")):
            with self.subTest(mutation=mutation):
                path, raw = self._manifest(count=1)
                args = raw["candidates"][0]["campaign_args"]
                flag, value = mutation
                if flag not in args:
                    args.extend((flag, value))
                else:
                    args[args.index(flag) + 1] = value
                if flag == "--candidate-id":
                    raw["candidates"][0]["candidate_id"] = value
                if flag == "--hypothesis":
                    raw["candidates"][0]["hypothesis_id"] = value
                self._write_manifest(path, raw)
                with self.assertRaisesRegex(C.CpuInferenceControllerError, message):
                    C.ControllerManifest.load(path)

    def test_controller_owned_paths_refuse_traversal_symlinks_git_and_production(self):
        def refuses(output, message, frozen_paths=()):
            path, raw = self._manifest(count=1)
            self._replace_output(path, raw, output)
            with mock.patch.object(C.campaign.worktree, "frozen_tree_paths",
                                   return_value=frozen_paths), self.assertRaisesRegex(
                                       C.CpuInferenceControllerError, message):
                C.ControllerManifest.load(path)

        refuses(Path(str(self.root / "owned") + "/../escape"),
                "canonical absolute")
        real = self.root / "real"
        real.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        refuses(alias / "owned", "symlink ancestry")

        repository = self.root / "repository"
        repository.mkdir()
        (repository / ".git").mkdir()
        refuses(repository, "Git metadata")

        frozen = self.root / "frozen"
        frozen.mkdir()
        refuses(frozen / "evidence", "frozen production", (frozen,))
        container = self.root / "frozen-container"
        nested_frozen = container / "production"
        nested_frozen.mkdir(parents=True)
        refuses(container, "frozen production", (nested_frozen,))

    def test_result_candidate_operation_t0_and_decision_mutations_refuse(self):
        mutations = (
            ("candidate_ref", lambda result: result["spec"].__setitem__(
                "candidate_ref", "foreign"), "selected full CPU operation"),
            ("empty_t0", lambda result: result["t0"].__setitem__("gates", []),
             "nonempty gate"),
            ("t0_false_pass", lambda result: result["t0"].__setitem__(
                "all_pass", False), "all_pass disagrees"),
            ("pair_delta", lambda result: result["pairs"][0].__setitem__(
                "delta", 99.0), "derived values disagree"),
            ("pair_index", lambda result: result["pairs"][1].__setitem__(
                "block_index", 0), "geometry"),
            ("decision_effect", lambda result: result["decision"].__setitem__(
                "median_relative", 0.5), "decision/effect differs"),
            ("decision_error", lambda result: result.__setitem__(
                "error", "keep_or_revert: failed"), "exact decision"),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                self.output = self.root / f"durable-{name}"
                path, raw = self._manifest(count=1)
                config = C.ControllerManifest.load(path)
                result = self._result(raw["candidates"][0],
                                      C.campaign.STATE_DECIDED, keep=True)
                mutate(result)
                with self.assertRaisesRegex(C.CpuInferenceControllerError, message):
                    C.run_controller(config, runner=FakeRunner([(0, result)]))

    def test_t0_could_not_check_is_infrastructure_and_fail_is_science(self):
        path, raw = self._manifest(count=2)
        config = C.ControllerManifest.load(path)
        uncertain = self._result(raw["candidates"][0], C.campaign.STATE_T0_FAILED)
        uncertain["t0"]["gates"][0][1] = C.campaign.schemas.COULD_NOT_CHECK
        failed = self._result(raw["candidates"][1], C.campaign.STATE_T0_FAILED)
        state = C.run_controller(config, runner=FakeRunner([(0, uncertain), (0, failed)]))
        self.assertEqual(state["scientific_attempts"], 1)
        self.assertEqual(
            [row["classification"] for row in state["iterations"]],
            ["infrastructure_ambiguous", "correctness_falsified"])

    def test_completed_restart_rederives_release_and_production_immutability(self):
        for name, mutate, message in (
                ("release", lambda result: result["releases"][0].__setitem__(
                    "released", False), "release of every acquired"),
                ("immutability", lambda result: result["production_unchanged"].__setitem__(
                    "outcome", C.campaign.schemas.COULD_NOT_CHECK),
                 "production immutability PASS")):
            with self.subTest(name=name):
                self.output = self.root / f"durable-restart-{name}"
                path, raw = self._manifest(count=1)
                config = C.ControllerManifest.load(path)
                C.run_controller(config, runner=FakeRunner([(
                    0, self._result(raw["candidates"][0],
                                    C.campaign.STATE_DECIDED, keep=True))]))
                self._rewrite_completed_receipt(config, mutate)
                with self.assertRaisesRegex(C.CpuInferenceControllerError, message):
                    C.run_controller(config, runner=NeverRunner())

    def test_recomputed_state_cannot_exceed_budget_or_swap_operation(self):
        path, raw = self._manifest(count=1, budget=1)
        config = C.ControllerManifest.load(path)
        C.run_controller(config, runner=FakeRunner([(
            0, self._result(raw["candidates"][0],
                            C.campaign.STATE_DECIDED, keep=True))]))
        state_path = self.output / "state.json"
        state = json.loads(state_path.read_text())
        state["scientific_attempts"] = 2
        state["state_sha256"] = C._sha({
            key: item for key, item in state.items() if key != "state_sha256"})
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(C.CpuInferenceControllerError, "counters"):
            C.run_controller(config, runner=NeverRunner())

        self.output = self.root / "durable-operation-swap"
        path, raw = self._manifest(count=1, budget=1)
        config = C.ControllerManifest.load(path)
        C.run_controller(config, runner=FakeRunner([(
            0, self._result(raw["candidates"][0],
                            C.campaign.STATE_DECIDED, keep=True))]))
        state_path = self.output / "state.json"
        state = json.loads(state_path.read_text())
        state["iterations"][0]["operation_key"] = "0" * 64
        state["state_sha256"] = C._sha({
            key: item for key, item in state.items() if key != "state_sha256"})
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "iteration identity"):
            C.run_controller(config, runner=NeverRunner())

    def test_journal_recovery_requires_current_candidate_operation_result(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        candidate = config.candidates[0]
        store = C.StateStore(config)
        state = store.load()
        state["inflight"] = {
            "candidate_index": 0, "candidate_id": candidate.candidate_id,
            "candidate_sha256": candidate.candidate_sha256,
            "operation_key": candidate.operation_key(self.controller_id),
            "started_at": C._now(),
        }
        store.save(state, "candidate_started")
        result = self._result(raw["candidates"][0],
                              C.campaign.STATE_DECIDED, keep=True)
        result["spec"]["candidate_ref"] = "foreign-operation"
        book = C.journal.Journal(str(self.output / "campaign-journal"),
                                 campaign_id=self.controller_id)
        book.initialize()
        book.append(C.journal.KIND_STOP_STATE, {
            "state": "decided", "campaign_id": self.controller_id,
            "result": result})
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "selected full CPU operation"):
            C.run_controller(config, runner=NeverRunner())

    def test_foreign_recipe_or_full_spec_shape_refuses_initial_and_restart(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        result = self._result(raw["candidates"][0],
                              C.campaign.STATE_DECIDED, keep=True)
        result["spec"]["recipe_id"] = "t1b.llama_cpu.llama_bench_decode.v1"
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "selected full CPU operation"):
            C.run_controller(config, runner=FakeRunner([(0, result)]))

        self.output = self.root / "durable-foreign-recipe-restart"
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        C.run_controller(config, runner=FakeRunner([(
            0, self._result(raw["candidates"][0],
                            C.campaign.STATE_DECIDED, keep=True))]))
        self._rewrite_completed_receipt(
            config, lambda value: value["spec"].__setitem__(
                "recipe_id", "t1b.llama_cpu.llama_bench_decode.v1"))
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "selected full CPU operation"):
            C.run_controller(config, runner=NeverRunner())

    def test_t0_could_not_check_requires_normalized_nonempty_reasons(self):
        for reason in ("", " ", " padded "):
            with self.subTest(reason=reason):
                self.output = self.root / f"durable-ctc-{len(reason)}-{reason.count(' ')}"
                path, raw = self._manifest(count=1)
                config = C.ControllerManifest.load(path)
                result = self._result(raw["candidates"][0],
                                      C.campaign.STATE_T0_FAILED)
                result["t0"]["gates"][0][1] = C.campaign.schemas.COULD_NOT_CHECK
                result["t0"]["gates"][0][2] = [reason]
                with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                            "normalized reasons"):
                    C.run_controller(config, runner=FakeRunner([(0, result)]))

    def test_preflight_refusal_is_exactly_preclaim_and_unsuccessful(self):
        mutations = (
            ("ok", lambda value: value.__setitem__("ok", True)),
            ("outcome", lambda value: value["preflight"].__setitem__(
                "outcome", C.campaign.schemas.PASS)),
            ("error", lambda value: value.__setitem__("error", "other")),
            ("step", lambda value: value["steps"].append({
                "index": 1, "name": "acquire_claim", "what": "claimed",
                "detail": {}})),
            ("release", lambda value: value["releases"].append({
                "name": "cpu_region_claim", "released": True, "detail": "released"})),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                self.output = self.root / f"durable-preflight-{name}"
                path, raw = self._manifest(count=1)
                config = C.ControllerManifest.load(path)
                result = self._result(raw["candidates"][0],
                                      C.campaign.STATE_PREFLIGHT_REFUSED)
                mutate(result)
                with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                            "preflight-refused"):
                    C.run_controller(config, runner=FakeRunner([(1, result)]))

    def test_state_error_is_typed_and_releases_every_step_acquisition(self):
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        untyped = self._result(raw["candidates"][0], C.campaign.STATE_ERROR,
                               ok=False)
        untyped["error"] = None
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "exact typed error"):
            C.run_controller(config, runner=FakeRunner([(1, untyped)]))

        self.output = self.root / "durable-state-error-claim-missing"
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        result = self._result(raw["candidates"][0], C.campaign.STATE_ERROR,
                              ok=False)
        result["steps"] = [{"index": 1, "name": "acquire_claim",
                            "what": "claimed", "detail": {}}]
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "matching release evidence"):
            C.run_controller(config, runner=FakeRunner([(1, result)]))

        self.output = self.root / "durable-state-error-worktree-missing"
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        result = self._result(raw["candidates"][0], C.campaign.STATE_ERROR,
                              ok=False)
        result["steps"] = [
            {"index": 1, "name": "acquire_claim", "what": "claimed", "detail": {}},
            {"index": 2, "name": "create_worktree", "what": "created", "detail": {}},
            {"index": 3, "name": "build", "what": "built", "detail": {}},
        ]
        result["releases"] = [{"name": "cpu_region_claim", "released": True,
                               "detail": "released"}]
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "matching release evidence"):
            C.run_controller(config, runner=FakeRunner([(1, result)]))

        self.output = self.root / "durable-state-error-valid"
        path, raw = self._manifest(count=1)
        config = C.ControllerManifest.load(path)
        result = self._result(raw["candidates"][0], C.campaign.STATE_ERROR,
                              ok=False)
        result["steps"] = [{"index": 1, "name": "acquire_claim",
                            "what": "claimed", "detail": {}}]
        result["releases"] = [{"name": "cpu_region_claim", "released": True,
                               "detail": "released"}]
        state = C.run_controller(config, runner=FakeRunner([(1, result)]))
        self.assertFalse(state["iterations"][0]["scientific_budget_spent"])

    def test_bool_is_never_accepted_for_a_numeric_result_field(self):
        mutations = (
            lambda value: value["spec"].__setitem__("blocks_precommitted", True),
            lambda value: value["pairs"][0].__setitem__("block_index", False),
            lambda value: value["decision"].__setitem__("blocks", True),
            lambda value: value["decision"]["deltas"].__setitem__(0, True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.output = self.root / f"durable-bool-{index}"
                path, raw = self._manifest(count=1)
                config = C.ControllerManifest.load(path)
                result = self._result(raw["candidates"][0],
                                      C.campaign.STATE_DECIDED, keep=True)
                mutate(result)
                with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                            "bool for a numeric field"):
                    C.run_controller(config, runner=FakeRunner([(0, result)]))

    def test_state_refuses_any_row_after_scientific_budget_is_reached(self):
        path, raw = self._manifest(count=2, budget=1)
        config = C.ControllerManifest.load(path)
        store = C.StateStore(config)
        state = store.load()
        for index, campaign_state in enumerate((
                C.campaign.STATE_DECIDED, C.campaign.STATE_PREFLIGHT_REFUSED)):
            candidate = config.candidates[index]
            operation_key = candidate.operation_key(self.controller_id)
            result = self._result(raw["candidates"][index], campaign_state,
                                  keep=True if index == 0 else None)
            disposition = C._result_disposition(
                candidate, result, config.output_root, operation_key)
            receipt, file_sha, result_sha = C._write_receipt(
                config.output_root, candidate, operation_key, result)
            C._append_iteration(state, C._iteration(
                candidate, disposition, operation_key=operation_key,
                receipt_path=receipt, receipt_file_sha256=file_sha,
                result_sha256=result_sha))
        state["complete"] = True
        state["terminal_reason"] = "scientific_budget_exhausted"
        store.save(state, "forged_post_budget_row")
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "after its scientific budget"):
            store.load()

        self.output = self.root / "durable-final-derived-counter"
        path, raw = self._manifest(count=1, budget=2)
        config = C.ControllerManifest.load(path)
        C.run_controller(config, runner=FakeRunner([(
            0, self._result(raw["candidates"][0],
                            C.campaign.STATE_DECIDED, keep=True))]))
        state_path = self.output / "state.json"
        state = json.loads(state_path.read_text())
        state["scientific_attempts"] = 0
        state["state_sha256"] = C._sha({
            key: item for key, item in state.items() if key != "state_sha256"})
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(C.CpuInferenceControllerError,
                                    "final science counter"):
            C.StateStore(config).load()


if __name__ == "__main__":
    unittest.main()
