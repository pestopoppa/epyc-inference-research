import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from autokernel.journal import Journal
from autokernel.lesson_native import _read_capture, produce, verify_artifacts


class NativeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.j = Journal(str(self.root / 'journal'))
        self.j.initialize()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        (self.repo / 'source.txt').write_text('initial')
        subprocess.run(['git', '-C', str(self.repo), 'add', 'source.txt'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Test', '-c',
                        'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        self.commit = subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip()
        self.exe = self.root / 'task.py'
        self.exe.write_text('#!/usr/bin/python3\nimport json,sys\np=sys.stdin.read()\nprint(json.dumps({"baseline_ns":1000,"candidate_ns":400 if p.startswith("LESSON (") else 500,"correctness":True,"protocol_id":"fixture/v1"}))\n')
        self.exe.chmod(0o755)
        self.args = {'artifact_root': self.root/'run', 'task_id': 't1', 'task_prompt': 'solve task',
            'candidate_id': 'c1', 'commit': self.commit, 'lesson': 'try this', 'reopen_when': 'changed',
            'upper_cap': 10.0, 'argv': [str(self.exe)], 'timeout': 5,
            'evaluator_sha256': hashlib.sha256(self.exe.read_bytes()).hexdigest(), 'protocol_id': 'fixture/v1',
            'source_repo': str(self.repo), 'code_artifacts': {}}

    def test_real_first_turn_execution_and_persisted_receipt(self):
        e = produce(self.j, **self.args)
        self.assertEqual(e.payload['decision'], 'ADMITTED')
        self.assertTrue(verify_artifacts(e.payload))
        self.assertEqual(e.payload['r1'], 2.0)
        self.assertEqual(e.payload['verification']['r_verify'], 2.5)
        self.assertEqual((self.root/'run/baseline/prompt.txt').read_text(), 'solve task')
        self.assertTrue((self.root/'run/verification/prompt.txt').read_text().startswith('LESSON ('))
        self.assertEqual(self.j.retrieve(), [])
        self.assertEqual(len(self.j.scan(validate_payloads=True).entries), 1)

    def test_missing_cap_never_executes(self):
        e = produce(self.j, **dict(self.args, upper_cap=None))
        self.assertEqual(e.payload['reason'], 'cap_missing')
        self.assertFalse((self.root/'run').exists())

    def test_capture_tamper_rejected(self):
        e = produce(self.j, **self.args)
        folder = self.root/'run/verification'
        meta = json.loads((folder/'capture.json').read_text())
        (folder/'stdout.txt').write_text('{"first_turn_speedup": 999}')
        with self.assertRaises(ValueError):
            _read_capture(folder, (folder/'prompt.txt').read_text(), [str(self.exe)], meta['executable_sha256'], 'fixture/v1', str(self.repo), self.commit, {})
        with self.assertRaises(ValueError):
            self.j.append('LESSON_ADMISSION_EVALUATED', e.payload)

    def test_native_failure_is_journaled_unknown(self):
        self.exe.write_text("#!/usr/bin/python3\nraise SystemExit(4)\n")
        self.args["evaluator_sha256"] = hashlib.sha256(self.exe.read_bytes()).hexdigest()
        e = produce(self.j, **self.args)
        self.assertEqual((e.payload["decision"], e.payload["reason"]), ("UNKNOWN", "verification_unavailable"))
        self.assertTrue((self.root / "run/baseline/capture.json").exists())
        self.assertTrue(e.payload["attempt_ref"].startswith(str(self.root / "run/attempt.json") + "#"))
        self.assertIn("baseline/capture.json", json.loads((self.root / "run/attempt.json").read_text()))

    def test_invalid_raw_measurement_is_unknown(self):
        self.exe.write_text("#!/usr/bin/python3\nprint('{\"baseline_ns\": 10, \"candidate_ns\": 0, \"correctness\": true}')\n")
        self.args["evaluator_sha256"] = hashlib.sha256(self.exe.read_bytes()).hexdigest()
        e = produce(self.j, **self.args)
        self.assertEqual(e.payload["decision"], "UNKNOWN")

    def test_cli_runs_the_native_producer_and_journals(self):
        config = dict(self.args, artifact_root=str(self.root / "cli-run"), argv=[str(self.exe)])
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        p = subprocess.run([sys.executable, "-m", "autokernel.lesson_native", "--config", str(path), "--journal", str(self.root / "journal")], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(p.stdout)["decision"], "ADMITTED")
        self.assertEqual(len(self.j.read_all()), 1)

    def test_wrong_evaluator_digest_refused_before_execution(self):
        e = produce(self.j, **dict(self.args, evaluator_sha256="0" * 64))
        self.assertEqual(e.payload['decision'], 'UNKNOWN')
        self.assertIn('attempt.json#', e.payload['attempt_ref'])
        self.assertFalse((self.root / "run/baseline").exists())

    def test_wrong_protocol_fails_closed_with_attempt_receipt(self):
        e = produce(self.j, **dict(self.args, protocol_id="other/v1"))
        self.assertEqual(e.payload["decision"], "UNKNOWN")
        self.assertIn("attempt.json#", e.payload["attempt_ref"])

    def test_replay_uses_captured_source_proof_not_current_tree(self):
        e = produce(self.j, **self.args)
        (self.repo / "source.txt").write_text("later change")
        self.assertTrue(verify_artifacts(e.payload))
        self.assertEqual(len(self.j.scan(validate_payloads=True).entries), 1)

    def test_wrong_source_commit_is_unknown_and_retains_proof(self):
        e = produce(self.j, **dict(self.args, commit='f'*40))
        self.assertEqual(e.payload['decision'], 'UNKNOWN')
        self.assertTrue((self.root/'run/baseline/source-failure.txt').exists())
        self.assertIn('baseline/source-failure.txt', json.loads((self.root/'run/attempt.json').read_text()))

    def test_dirty_tracked_source_is_unknown(self):
        (self.repo/'source.txt').write_text('modified')
        e = produce(self.j, **self.args)
        self.assertEqual(e.payload['decision'], 'UNKNOWN')

    def test_source_changed_during_verification_is_unknown(self):
        self.exe.write_text('#!/usr/bin/python3\nimport json,sys\nfrom pathlib import Path\np=sys.stdin.read()\nif p.startswith("LESSON ("): Path('+repr(str(self.repo / "source.txt"))+').write_text("changed")\nprint(json.dumps({"baseline_ns":1000,"candidate_ns":400,"correctness":True,"protocol_id":"fixture/v1"}))\n')
        self.exe.chmod(0o755)
        self.args['evaluator_sha256'] = hashlib.sha256(self.exe.read_bytes()).hexdigest()
        e = produce(self.j, **self.args)
        self.assertEqual(e.payload['decision'], 'UNKNOWN')
        inventory = json.loads((self.root/'run/attempt.json').read_text())
        self.assertIn('verification/source-failure.txt', inventory)
        self.assertIn('verification/stdout.txt', inventory)

    def test_interpreter_requires_pinned_entry(self):
        args = dict(self.args, argv=[sys.executable, str(self.exe)],
                    evaluator_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            produce(self.j, **args)
        args['code_artifacts'] = {str(self.exe): hashlib.sha256(self.exe.read_bytes()).hexdigest()}
        self.assertEqual(produce(self.j, **args).payload['decision'], 'ADMITTED')

    def test_code_changed_during_execution_is_unknown(self):
        self.exe.write_text('#!/usr/bin/python3\nfrom pathlib import Path\nimport json\nPath(__file__).write_text("changed")\nprint(json.dumps({"baseline_ns":1000,"candidate_ns":500,"correctness":True,"protocol_id":"fixture/v1"}))\n')
        self.exe.chmod(0o755)
        self.args['argv'] = [sys.executable, str(self.exe)]
        self.args['evaluator_sha256'] = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
        self.args['code_artifacts'] = {str(self.exe): hashlib.sha256(self.exe.read_bytes()).hexdigest()}
        e = produce(self.j, **self.args)
        self.assertEqual(e.payload['decision'], 'UNKNOWN')
        self.assertTrue((self.root/'run/baseline/source-failure.txt').exists())

    def test_bad_commit_refused_before_execution(self):
        with self.assertRaises(ValueError):
            produce(self.j, **dict(self.args, commit="invalid"))
        self.assertFalse((self.root / "run").exists())

    def test_no_overwrite_or_second_run(self):
        produce(self.j, **self.args)
        with self.assertRaises(FileExistsError):
            produce(self.j, **self.args)

if __name__ == '__main__': unittest.main()
