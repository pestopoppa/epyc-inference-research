import ast
import re
from pathlib import Path
import unittest
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from yarn_proof import check_proof

def gpu_constants(name):
    tree = ast.parse((HERE / 'yarn_needle.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)

class ProofTests(unittest.TestCase):
    def test_actual_a1_last_launch(self):
        log = (Path(__file__).parent / 'fixtures/A1.server.log').read_text()
        result = check_proof(log, gpu_constants('PROOF')['A1'], gpu_constants('FORBID'))
        self.assertTrue(result['ok'], result)
    def test_split_and_inline(self):
        pattern = r"Using metadata override \(\s*str\) 'a' = yarn"
        for sep in ['0.00.597.049 I ', '\n0.00.597.049 I ', '\n']:
            self.assertTrue(check_proof("Using metadata override ( str) 'a' = " + sep + 'yarn', [pattern], [])['ok'])
    def test_wrong_and_missing_values(self):
        pattern = r"Using metadata override \(\s*str\) 'a' = yarn"
        for log in ["Using metadata override ( str) 'a' = linear", "Using metadata override ( str) 'a' =\nunrelated\nyarn"]:
            self.assertFalse(check_proof(log, [pattern], [])['ok'])
    def test_last_launch_only(self):
        self.assertFalse(check_proof('=== launch old\nMATCH\n=== launch new\nmissing', ['MATCH'], [])['ok'])
    def test_no_marker_keeps_full_log(self):
        self.assertTrue(check_proof('MATCH', ['MATCH'], [])['ok'])
    def test_forbidden_retained(self):
        self.assertFalse(check_proof('MATCH\nHIP error', ['MATCH'], ['HIP error'])['ok'])
    def test_actual_preflight_with_only_live_inspection_stubs(self):
        import yarn_needle as yn
        real_read_bytes, real_read_text = Path.read_bytes, Path.read_text
        def read_bytes(path):
            if str(path) == '/proc/424242/environ':
                return b'LD_LIBRARY_PATH=/store\0'
            if str(path) == '/proc/424242/cmdline':
                return b'llama-server\0--rope-scaling\0yarn\0'
            return real_read_bytes(path)
        def read_text(path, *a, **kw):
            if str(path) == '/proc/424242/maps':
                return '/store/gpu-20260921-ffc1bac82/libggml-hip.so\n'
            return real_read_text(path, *a, **kw)
        def http(url, **kwargs):
            return (200, [{'n_ctx': 524288}]) if url.endswith('/slots') else (200, {'n_ctx': 524288})
        args = SimpleNamespace(port=18083, arm='A1', server_log=str(Path(__file__).parent / 'fixtures/A1.server.log'))
        self.assertFalse(hasattr(args, 'force_preflight'))
        with patch.object(yn, 'port_pid', side_effect=lambda port: None if port == 8083 else 424242), patch.object(yn, 'http_json', side_effect=http), patch.object(yn, 'kfd_gib', return_value=30.0), patch.object(Path, 'read_bytes', read_bytes), patch.object(Path, 'read_text', read_text):
            result = yn.preflight(args, 'http://fixture')
            def legacy(log, patterns, forbidden):
                return {'proof': {p: bool(re.search(p, log, re.M)) for p in patterns}, 'forbidden': {p: len(re.findall(p, log, re.I | re.M)) for p in forbidden}}
            with patch.object(yn, 'check_proof', side_effect=legacy):
                before = yn.preflight(args, 'http://fixture')
            self.assertEqual(len([p for p in before['problems'] if p.startswith('proof line missing')]), 2)
        self.assertEqual(result['problems'], [])
        self.assertTrue(all(result['proof'].values()))

    def test_real_cpu_and_gpu_import_same_helper(self):
        import yarn_needle as yn
        import yarn_proof
        sys.path.insert(0, str(HERE / 'cpu_leg'))
        spec = importlib.util.spec_from_file_location('private_yarn_cpu', HERE / 'cpu_leg/yarn_cpu_leg.py')
        cpu = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cpu
        spec.loader.exec_module(cpu)
        self.assertIs(yn.check_proof, yarn_proof.check_proof)
        self.assertIs(cpu.check_proof, yarn_proof.check_proof)
        # Exercise the real CPU proof method with a log lacking a launch marker.
        server = object.__new__(cpu.Server)
        server.external = False
        server.arm = 'CN'
        server.log = SimpleNamespace(read_text=lambda **kw: 'MATCH')
        with patch.object(cpu, 'proof_patterns', return_value=['MATCH']), patch.object(cpu, 'FORBID', []):
            self.assertTrue(server.proof()['ok'])

    def test_launch_flags_shared(self):
        self.assertIn("*PROOF_LOG_FLAGS", (HERE / "cpu_leg/yarn_cpu_leg.py").read_text())
        self.assertIn('"$D/yarn_proof.py" --log-flags', (HERE / "launch_arm.sh").read_text())

    def test_both_consumers_use_shared_helper(self):
        for rel in ['yarn_needle.py', 'cpu_leg/yarn_cpu_leg.py']:
            s = (HERE / rel).read_text()
            compile(s, rel, 'exec')
            self.assertIn('check_proof(log,', s)

if __name__ == '__main__':
    unittest.main()
