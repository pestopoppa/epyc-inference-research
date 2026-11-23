"""AK-HARNESS-CRASH-VS-VACUOUS: judge() separates CRASH (a gfx90a kernel ran and faulted) from VACUOUS (no kernel ran)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import job

U = dict(unit='u', family='fmoe', script='s', groups=[], load_re=r'/gfx90a/fmoe', msg_re=None, timeout=100, patch=None, tol=None, idle_s=10)
LOAD = '==RUN==\nLoadKernel: k hsaco: hsa/gfx90a/fmoe/a.co\n'
OK = {'rc': 0}


class CrashVsVacuous(unittest.TestCase):
    def test_memory_fault_after_load_is_crash(self):
        # the T1 r3 id1024 case: fault right after the R9 spill kernels loaded, rc -6, no runner result
        s, why, _ = job.judge(U, LOAD + 'HSA_STATUS_ERROR_MEMORY_FAULT\n', None, -6, None)
        self.assertEqual(s, 'crash'); self.assertIn('crash', why)

    def test_signal_rc_after_load_is_crash(self):
        self.assertEqual(job.judge(U, LOAD, OK, -11, None)[0], 'crash')

    def test_nonzero_rc_no_runner_result_after_load_is_crash(self):
        self.assertEqual(job.judge(U, LOAD, None, 1, None)[0], 'crash')

    def test_no_kernel_ran_is_vacuous(self):
        s, why, _ = job.judge(U, '==RUN==\nnothing loaded\n', OK, 0, None)
        self.assertEqual(s, 'vacuous'); self.assertIn('no kernel ran', why)

    def test_fault_text_without_load_is_not_crash(self):
        # an import-time abort never reached a gfx90a kernel: not a kernel crash
        self.assertNotEqual(job.judge(U, '==RUN==\nAborted\n', None, -6, None)[0], 'crash')

    def test_loaded_clean_exit_no_check_is_not_vacuous(self):
        s, _, _ = job.judge(U, LOAD, OK, 0, None)
        self.assertEqual(s, 'error')

    def test_pass_unchanged(self):
        t = LOAD + 'ASMT1_CHECK {"msg": "m", "ret": 0.0, "max_abs": 1.0}\n'
        self.assertEqual(job.judge(U, t, OK, 0, None)[0], 'pass')


if __name__ == '__main__': unittest.main()
