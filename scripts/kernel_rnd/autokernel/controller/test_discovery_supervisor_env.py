"""Pure-Python checks for the detached-launch environment allowlist."""
import unittest

from scripts.kernel_rnd.autokernel.controller import discovery_supervisor as S


class ForwardedLauncherEnvTest(unittest.TestCase):
    def test_forwards_only_allowlisted_present_names(self):
        env = {"ORCHESTRATOR_TMP_DIR": "/x/nest", "FILLER_ID": "row-1", "HOME": "/h",
               "PYTHONPATH": "/evil"}
        self.assertEqual(S._forwarded_launcher_env(env),
                         ("ORCHESTRATOR_TMP_DIR=/x/nest", "FILLER_ID=row-1"))

    def test_empty_when_absent(self):
        self.assertEqual(S._forwarded_launcher_env({}), ())

    def test_refuses_newline(self):
        with self.assertRaises(S.SupervisorError):
            S._forwarded_launcher_env({"FILLER_ID": "a\nb"})


if __name__ == "__main__":
    unittest.main()
