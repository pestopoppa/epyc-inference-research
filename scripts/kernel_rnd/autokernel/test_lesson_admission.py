import tempfile
import unittest
from unittest.mock import patch

from autokernel.journal import Journal, RetrievalCitationError
from autokernel.lesson_admission import KIND, _evaluate_and_record


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal = Journal(self.tmp.name)
        self.journal.initialize()
        verifier = patch("autokernel.lesson_native.verify_artifacts", return_value=True)
        verifier.start()
        self.addCleanup(verifier.stop)
        self.args = {"task_id": 'task1', "candidate_id": 'candidate1', "commit": 'a'*40,
                         "lesson": 'apply X', "reopen_when": 'source changes', "r1": 2.0, "upper_cap": 10.0}
        self.calls = []

    def run_verify(self, result=2.4, **overrides):
        def rerun(**kw):
            self.calls.append(kw)
            return {"task_id": overrides.get('task_id', kw['task_id']),
                        "commit": overrides.get('commit', kw['commit']),
                        "r_verify": result, "run_ref": '/native#manifest='+'b'*64+';baseline='+'c'*64+';verification='+'d'*64}
        return rerun

    def evaluate(self, runner, **overrides):
        args = dict(self.args, **overrides)
        return _evaluate_and_record(self.journal, rerun=runner, **args)

    def test_real_append_and_replay(self):
        e = self.evaluate(self.run_verify())
        self.assertEqual(e.payload['decision'], 'ADMITTED')
        self.assertEqual(self.calls, [{"task_id": 'task1', "commit": 'a'*40, "lesson": 'apply X'}])
        self.assertEqual(self.journal.scan(validate_payloads=True).entries[0].payload, e.payload)

    def test_threshold_is_relative_to_first_run(self):
        self.assertEqual(self.evaluate(self.run_verify(2.39)).payload['reason'], 'verification_below_threshold')

    def test_missing_cap_never_runs(self):
        e = self.evaluate(self.run_verify(), upper_cap=None)
        self.assertEqual((e.payload['decision'], e.payload['reason']), ('UNKNOWN', 'cap_missing'))
        self.assertFalse(self.calls)

    def test_cap_mismatch_and_bad_result_fail_closed(self):
        self.assertEqual(self.evaluate(self.run_verify(11)).payload['reason'], 'upper_cap_exceeded')
        self.assertEqual(self.evaluate(self.run_verify(task_id='different')).payload['decision'], 'UNKNOWN')
        self.assertEqual(self.evaluate(self.run_verify(float('nan'))).payload['decision'], 'UNKNOWN')

    def test_direct_tampering_is_refused(self):
        e = self.evaluate(self.run_verify(1.5))
        p = dict(e.payload, decision='ADMITTED', reason='execution_verified')
        with self.assertRaises(ValueError):
            self.journal.append(KIND, p)
        for invalid in (float('inf'), float('nan'), True, 10**400):
            with self.assertRaises(ValueError):
                self.evaluate(self.run_verify(), upper_cap=invalid)

    def test_admission_remains_out_of_retrieval_even_if_cited(self):
        e = self.evaluate(self.run_verify())
        self.assertEqual(self.journal.retrieve(), [])
        with self.assertRaises(RetrievalCitationError):
            self.journal.retrieve(cite_event_ids=[e.event_id])

    def test_runner_failure_is_durable_unknown(self):
        def fail(**kw):
            raise RuntimeError('native failed')
        self.assertEqual(self.evaluate(fail).payload['decision'], 'UNKNOWN')

if __name__ == '__main__':
    unittest.main()
