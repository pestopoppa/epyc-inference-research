"""VB-AK-SEAT write side: every actor call appends an `epyc.autokernel.actor_call.v1` record.

The contract (closed field set, validators, reference writer) lives in the ROOT repo at
`scripts/vidya/adapters/autokernel_actor_seat_capture.py`; the producer builds its line
with that module so writer and reader cannot drift. These tests validate what
`_run_agent` actually wrote with the same module's validator.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actors

ROOT = Path(os.environ.get(actors.ROOT_REPO_ENV, "/workspace"))
HAVE_CONTRACT = (ROOT / actors.SEAT_CAPTURE_REL).is_file()

HYP = json.dumps({"mechanism_id": "akm-x", "statement": "s", "falsifier": "f",
                  "target_surface": "a.cpp", "target_symbol": "f"})


class _Lane:

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = self.root / "lane"
        self.ws.mkdir()
        actors._V1_CACHE.pop(f"capture:{(ROOT / actors.SEAT_CAPTURE_REL).resolve()}", None)

    def tearDown(self):
        self._tmp.cleanup()

    def _rows(self):
        log = self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
        return [json.loads(line) for line in log.read_text().splitlines()]

    def _call(self, backend, *, schema, env=None, rc=0, stdout=HYP):
        done = subprocess.CompletedProcess([], rc, stdout=stdout, stderr="chrome")
        with mock.patch.object(actors.subprocess, "run", return_value=done):
            try:
                actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                  schema=schema, env=env)
            except actors.ProviderTransient:
                pass
        return self._rows()[-1]


@unittest.skipUnless(HAVE_CONTRACT, f"VB-AK-SEAT contract not in ROOT checkout {ROOT}")
class CallRecordV1(_Lane, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.capture = actors._seat_capture()

    def test_a_plain_opencode_planner_call_writes_a_valid_v1_record(self):
        row = self._call(actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
                         schema=actors.HYPOTHESIS_SCHEMA)
        self.assertEqual(self.capture.validate_call_record(row), [])
        self.assertEqual(row["schema"], self.capture.CALL_SCHEMA)
        self.assertEqual((row["role"], row["seat"]["arm"], row["seat"]["bounded"]),
                         ("planner", "plain", False))
        self.assertEqual(row["prompt"]["chars"], len("the prompt"))
        self.assertRegex(row["producer"]["commit"], r"^[0-9a-f]{40}$")
        # The reply files the record names are the bytes actually kept.
        replies = self.ws.parent / actors.ACTOR_REPLY_DIR
        out = row["reply"]["stdout"]
        self.assertEqual((replies / out["path"]).read_text(), HYP)
        self.assertEqual(out["bytes"], len(HYP.encode()))

    def test_a_bounded_seat_records_its_config_instructions_and_step_cap(self):
        instructions = self.root / "actor-opencode-planner.instructions.md"
        instructions.write_text("tool discipline")
        config = self.root / "actor-opencode-planner.json"
        config.write_text(json.dumps({"instructions": [str(instructions)]}))
        env = {"OPENCODE_CONFIG": str(config), actors.SEAT_ENV_ARM: "bounded",
               actors.SEAT_ENV_FAN_OUT: "1", actors.SEAT_ENV_STEPS: "60"}
        row = self._call(actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
                         schema=actors.PATHS_SCHEMA, env=env,
                         stdout='{"paths": ["a.cpp"]}')
        self.assertEqual(self.capture.validate_call_record(row), [])
        seat = row["seat"]
        self.assertEqual((row["role"], seat["arm"], seat["bounded"], seat["fan_out"],
                          seat["steps"]), ("author", "bounded", True, True, 60))
        self.assertEqual(seat["config"]["path"], str(config))
        self.assertEqual(seat["instructions"]["bytes"], len("tool discipline"))

    def test_a_hosted_critic_is_a_plain_seat_with_no_opencode_fields(self):
        row = self._call(actors.CRITIC_DEFAULT, schema=actors.REVIEW_SCHEMA,
                         stdout='{"accepted": true}')
        self.assertEqual(self.capture.validate_call_record(row), [])
        self.assertEqual(row["role"], "critic")
        self.assertEqual(row["server"]["endpoint"], f"hosted:{actors.CRITIC_DEFAULT.kind}")
        self.assertIsNone(row["seat"]["opencode_version"])

    def test_a_failed_call_is_still_one_valid_record(self):
        row = self._call(actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
                         schema=actors.HYPOTHESIS_SCHEMA, rc=1, stdout="no json")
        self.assertEqual(self.capture.validate_call_record(row), [])
        self.assertEqual(row["returncode"], 1)


class CallRecordFallsBackVisibly(_Lane, unittest.TestCase):

    def test_without_the_contract_the_line_is_pre_hook_and_says_why(self):
        with mock.patch.dict(os.environ, {actors.ROOT_REPO_ENV: str(self.root / "no-root")}):
            row = self._call(actors.backend_for("qwen-gpu/qwen3.8-27b", "high"),
                             schema=actors.HYPOTHESIS_SCHEMA)
        self.assertNotIn("schema", row)            # pre-hook shape: projects no claim
        self.assertIn("VB-AK-SEAT contract missing", row["v1_refused"])
        self.assertEqual(row["prompt_chars"], len("the prompt"))

    def test_an_unknown_role_is_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            actors._role_of({"required": ["x"]})


class GitHeadReader(unittest.TestCase):

    def test_it_reads_this_checkout_without_a_subprocess(self):
        with mock.patch.object(actors.subprocess, "run",
                               side_effect=AssertionError("no subprocess")):
            head = actors._git_head(Path(actors.__file__).resolve().parent)
        self.assertRegex(head, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
