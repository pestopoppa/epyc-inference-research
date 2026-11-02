import json
import os
import socket
import sys
import time
import server_lifecycle
from pathlib import Path
import tempfile
import unittest
import producer
import sampler

class ProducerTests(unittest.TestCase):
    def test_only_named_isolation_substitutions(self):
        original = ['server', '--port', '8083', '--no-cache-idle-slots', '--slot-save-path', '/live']
        changed = producer.replace_single(producer.replace_single(original, '--port', 18197), '--slot-save-path', '/private')
        self.assertEqual(changed, ['server', '--port', '18197', '--no-cache-idle-slots', '--slot-save-path', '/private'])
        self.assertEqual(original[2], '8083')
        with self.assertRaises(ValueError): producer.replace_single(original + ['--port', '42'], '--port', 1)

    def test_effective_env_keeps_runtime_knobs_without_secrets(self):
        self.assertEqual(producer.selected_env({'GGML_IQK':'1', 'LD_LIBRARY_PATH':'/x', 'API_KEY':'secret'}),
                         {'GGML_IQK':'1', 'LD_LIBRARY_PATH':'/x'})

    def test_fake_runtime_startup_unknown_then_healthy_and_failed_sample(self):
        with tempfile.TemporaryDirectory() as d:
            phase = ['startup']; calls = []
            def reader(pid, birth, library, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1: return {'own_vram_b':None, 'card_used_b':0, 'hip_resident':False}
                if len(calls) == 2:
                    phase[0] = 'healthy'
                    return {'own_vram_b':1024, 'card_used_b':1024, 'hip_resident':True}
                raise OSError('lost KFD sample')
            m = sampler.Sampler(1, 2, '/fake/libggml-hip.so', Path(d)/'samples', interval=0.001,
                                reader=reader, startup=True, phase=lambda: phase[0]).start()
            self.assertTrue(m.abort.wait(2)); self.assertTrue(m.stop())
            records = [json.loads(x) for x in (Path(d)/'samples').read_text().splitlines()]
            self.assertIsNone(records[0]['own_vram_b'])
            self.assertEqual(records[0]['phase'], 'startup')
            self.assertEqual(records[1]['phase'], 'healthy')
            self.assertEqual(records[2]['status'], 'missing')
            self.assertIn('lost KFD sample', records[2]['error'])
            self.assertTrue(calls[0]['allow_uninitialized'])
            self.assertNotIn('allow_uninitialized', calls[2])

    def test_exclusive_write_preserves_original(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'record.json'; producer.write_json(path, {'value':1})
            with self.assertRaises(FileExistsError): producer.write_json(path, {'value':2})
            self.assertEqual(json.loads(path.read_text()), {'value':1})

    def test_owned_fake_http_server_lifecycle(self):
        with tempfile.TemporaryDirectory() as d:
            sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]; sock.close()
            server = server_lifecycle.Server([sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'],
                                              dict(os.environ), Path(d)/'server.log', Path(d)/'pid')
            try:
                pid, birth = server.start()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        conn = socket.create_connection(('127.0.0.1', port), timeout=0.1); conn.close(); break
                    except OSError:
                        time.sleep(0.02)
                # The stock fake returns 404 for /health, but we prove it actually serves HTTP.
                with socket.create_connection(('127.0.0.1', port), timeout=1): pass
                self.assertEqual(sampler.start_ticks(Path('/proc')/str(pid)), birth)
            finally:
                result = server.close(term_grace=2, kill_grace=2)
            self.assertTrue(result['verified_dead'])
            self.assertIsNotNone(server.proc.poll())

    def test_keeper_dry_run_is_fake_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]; sock.close()
            result = Path(d) / 'dry.json'
            self.assertEqual(producer.fake_dry_run(port, Path(d) / 'fake', result, Path(d) / 'pid'), 0)
            receipt = json.loads(result.read_text())
            self.assertTrue(receipt['fake'] and receipt['passed'] and receipt['cleanup']['verified_dead'])
            self.assertNotIn('sampled_max_own_vram_b', receipt)

    def test_process_identity_rejects_argv_mismatch(self):
        birth = sampler.start_ticks(Path('/proc') / str(os.getpid()))
        plan = {'argv':['/deliberately-wrong'], 'effective_env':{}, 'identities':{}}
        with self.assertRaisesRegex(RuntimeError, 'argv differs'):
            producer.proc_identity(os.getpid(), birth, plan)
        with self.assertRaisesRegex(RuntimeError, 'birth changed'):
            producer.proc_identity(os.getpid(), birth + 1, plan)

if __name__ == '__main__': unittest.main()
