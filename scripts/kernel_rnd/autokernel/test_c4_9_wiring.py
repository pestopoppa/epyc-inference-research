#!/usr/bin/env python3
"""RVP-C4-9 wiring: eval-backend run -> per-rule telemetry. Offline, no GPU."""
import json
import unittest
from pathlib import Path
from unittest import mock

from . import c4_rule_telemetry as M
from .controller import arena_cell_runner as ACR
from .test_profile_report import ProfileReportTest

OK = {"pass_compilation": True, "pass_correctness": True, "average_speedup": 1.7}


class EvaluationSpeedup(unittest.TestCase):
    def test_valid_run_gives_speedup(self):
        self.assertEqual(M.evaluation_speedup(OK), 1.7)

    def test_failed_or_degenerate_runs_are_null_not_zero(self):
        for bad in ({**OK, "pass_correctness": False}, {**OK, "pass_compilation": False},
                    {**OK, "average_speedup": 0.0}, {**OK, "average_speedup": float("nan")},
                    {**OK, "average_speedup": True}, {"pass_compilation": True,
                                                       "pass_correctness": True}):
            self.assertIsNone(M.evaluation_speedup(bad))


class LogEvalRun(ProfileReportTest):
    def config(self, tmp, values=(1.2, 1.6)):
        d = self.manifest_dict()
        for pattern, value in zip(d["patterns"], values):
            pattern["estimated_speedup"] = value
        manifest = Path(tmp) / "manifest.json"
        manifest.write_text(json.dumps(d), encoding="utf-8")
        return {"manifest": str(manifest), "mapping_profile": str(self.mapping_path),
                "formal_profile": str(self.formal_path), "out": str(Path(tmp) / "t.jsonl")}

    def test_logs_one_record_per_rule_and_folds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            summary = M.log_eval_run(cfg, run_id="c:a", task_id="t1", evaluation=OK)
            M.log_eval_run(cfg, run_id="c:a", task_id="t2",
                           evaluation={**OK, "pass_correctness": False})
            recs = M.load_records([Path(cfg["out"])])
            self.assertEqual(summary["status"], "logged")
            self.assertEqual(summary["records"], len(recs) // 2)
            self.assertEqual({r.task_id for r in recs}, {"t1", "t2"})
            self.assertEqual([r.downstream_speedup for r in recs if r.task_id == "t1"],
                             [1.7] * summary["records"])
            self.assertTrue(all(r.downstream_speedup is None for r in recs if r.task_id == "t2"))
            self.assertEqual(M.fold(recs)["tasks"], 2)

    def test_missing_config_key_refused(self):
        with self.assertRaises(M.TelemetryError):
            M.log_eval_run({"manifest": "x"}, run_id="r", task_id="t", evaluation=OK)


class CellRunnerHook(unittest.TestCase):
    def call(self, config, **kw):
        args = dict(baseline=False, campaign_id="c", arm_id="a", task_id="t",
                    evaluation=OK)
        args.update(kw)
        return ACR._log_c4_rule_telemetry(config, **args)

    def test_off_by_default_and_for_baseline(self):
        self.assertIsNone(self.call(None))
        self.assertIsNone(self.call({"out": "x"}, baseline=True))

    def test_delegates_with_run_identity(self):
        with mock.patch.object(M, "log_eval_run", return_value={"status": "logged"}) as m:
            self.assertEqual(self.call({"out": "x"}), {"status": "logged"})
        m.assert_called_once_with({"out": "x"}, run_id="c:a", task_id="t", evaluation=OK)

    def test_failure_is_recorded_and_never_raises(self):
        out = self.call({"manifest": "x"})
        self.assertEqual(out["status"], "error")
        self.assertIn("TelemetryError", out["error"])
        self.assertEqual(self.call("not-an-object")["status"], "error")


del ProfileReportTest  # noqa

if __name__ == "__main__":
    unittest.main()
