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
        calibration.mkdir()
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
            "--model", str(files["model.gguf"]),
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
        body = {"schema": C.CANDIDATE_SCHEMA, "candidate_id": candidate_id,
                "hypothesis_id": hypothesis_id, "campaign_args": args,
                "artifacts": artifacts}
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

    def _result(self, candidate, state, *, keep=None, ok=True):
        scientific = state in {C.campaign.STATE_DECIDED,
                               C.campaign.STATE_T0_FAILED}
        releases = ([{"name": "cpu_region_claim", "released": True,
                      "detail": "released"},
                     {"name": "campaign_worktree", "released": True,
                      "detail": "released"}] if scientific else [])
        return {
            "schema": "epyc.autokernel.campaign_result.v1",
            "campaign_id": self.controller_id,
            "candidate_id": candidate["candidate_id"],
            "spec": {"campaign_id": self.controller_id,
                     "candidate_id": candidate["candidate_id"],
                     "backend": "llama_cpu",
                     "journal_root": str(self.output / "campaign-journal")},
            "state": state,
            "steps": [],
            "t0": ({"all_pass": state == C.campaign.STATE_DECIDED,
                    "report_ref": "t0.json", "gates": []}
                   if scientific else None),
            "decision": ({"keep": keep, "reason": "fixture",
                          "blocks": 2, "min_delta": 1.0,
                          "median_relative": 0.1,
                          "contribution_floor": 0.03,
                          "calibration_evidence_ref": "fixture",
                          "drift_bound": 0.1, "anchor_drift": 0.0,
                          "deltas": [1.0, 1.0],
                          "relatives": [0.1, 0.1],
                          "anchors": [10.0, 10.0],
                          "orders": ["anchor_first", "candidate_first"]}
                         if state == C.campaign.STATE_DECIDED else None),
            "pairs": ([{"block_index": 0, "anchor": 10.0,
                        "candidate": 11.0, "order": "anchor_first",
                        "delta": 1.0, "relative": 0.1}]
                      if state == C.campaign.STATE_DECIDED else []),
            "preflight": {"outcome": "PASS", "reasons": []},
            "releases": releases,
            "production_unchanged": {"outcome": "PASS", "reasons": []},
            "executed": True, "screening_only": False,
            "non_promotable": False, "journal_error": None,
            "screening_report": None, "ok": ok,
            "error": None, "grammar": "SEARCH RECORD, NOT A CLAIM",
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
            (0, self._result(raw["candidates"][3],
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


if __name__ == "__main__":
    unittest.main()
