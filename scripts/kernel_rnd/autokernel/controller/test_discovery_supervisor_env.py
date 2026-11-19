"""Pure-Python checks for the detached-launch environment allowlist."""
import os
import shutil
import subprocess
import tempfile
import types
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


@unittest.skipUnless(shutil.which("tmux"), "tmux not installed")
class PreexistingTmuxServerTest(unittest.TestCase):
    """A tmux server already running on the socket must not change the child env."""

    def test_launcher_env_survives_a_preexisting_server(self):
        sock = f"epyc-test-{os.getpid()}"
        tmp = tempfile.mkdtemp(prefix="supenv-")
        out = os.path.join(tmp, "environ")

        def tmux(*a):
            return subprocess.run(("tmux", "-L", sock, *a), text=True, capture_output=True, check=False)

        try:
            # server first, with a conflicting global env for the same names
            self.assertEqual(tmux("-f", "/dev/null", "new-session", "-d", "-s", "keep", "sleep 60").returncode, 0)
            tmux("set-environment", "-g", "FILLER_ID", "SERVER-VALUE")
            tmux("set-environment", "-g", "ORCHESTRATOR_TMP_DIR", "/server/value")
            spec = types.SimpleNamespace(
                body={"execution_closure": {"path": tmp}, "python": "/bin/sh"},
                runtime_root=tmp,
            )
            env = {"ORCHESTRATOR_TMP_DIR": "/launcher/nest", "FILLER_ID": "row-9"}
            argv = S._detached_command(spec, env)
            # same prefix start_detached uses; swap the python tail for an env dump
            prefix = argv[: argv.index("/bin/sh")]
            r = tmux("new-session", "-d", "-s", "probe", "--", *prefix, "/bin/sh", "-c", f"tr '\\0' '\\n' < /proc/self/environ > {out}")
            self.assertEqual(r.returncode, 0, r.stderr)
            for _ in range(100):
                if os.path.exists(out) and os.path.getsize(out):
                    break
                subprocess.run(("sleep", "0.05"))
            got = dict(l.split("=", 1) for l in open(out).read().splitlines() if "=" in l)
            self.assertEqual(got.get("FILLER_ID"), "row-9")
            self.assertEqual(got.get("ORCHESTRATOR_TMP_DIR"), "/launcher/nest")
        finally:
            tmux("kill-server")  # only the private socket this test created
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
