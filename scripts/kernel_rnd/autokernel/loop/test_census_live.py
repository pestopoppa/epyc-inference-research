import subprocess
import tempfile
import unittest
from pathlib import Path

from . import census as CE
from . import census_live as CL
from . import census_store as CS


def fake_gguf(path: Path):
    import struct
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 2) + struct.pack("<Q", 1)
    body += s("general.architecture") + struct.pack("<I", 8) + s("llama")
    for name, t in (("a", 0), ("b", 8)):
        body += s(name) + struct.pack("<I", 1) + struct.pack("<Q", 64) + struct.pack("<I", t) + struct.pack("<Q", 0)
    path.write_bytes(body)


class Live(unittest.TestCase):
    def test_produce_pins_cpus_stamps_the_hook_identity_and_observes(self):
        calls = []
        lines = "\n".join(f"node #{i:4d} ( MUL_MAT):  x [ CPU ]" for i in range(1500))

        def run(argv, **kw):
            assert kw.get("errors") == "replace"
            calls.append((argv, kw["env"]))
            return subprocess.CompletedProcess(argv, 0, stdout=lines, stderr="")
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "m.gguf"
            fake_gguf(model)
            binary = Path(td) / "bin" / "llama-bench"
            binary.parent.mkdir()
            binary.write_text("")
            doc = CL.produce(binary=binary, model=model, workload_id="W3", signature="sig",
                             champion_commit="c" * 40, probe_name="p", cpus="24-71", threads=48,
                             shapes=[CE.Shape("prefill", 8), CE.Shape("decode", 1)],
                             runner=CL.taskset_runner("24-71", run=run))
        self.assertEqual(doc["validity"]["state"], CE.OBSERVED)
        self.assertEqual(doc["validity"]["freshness_key"], CS.expected_keys(
            [{"target_ids": ["W3"], "workload_signature": "sig"}],
            champion_commit="c" * 40, probe="p")["W3"])
        self.assertEqual({"F32", "Q8_0"}, set(doc["weights"]["tensor_types"]))
        argv, env = calls[0]
        self.assertEqual(argv[:3], ["/usr/bin/taskset", "-c", "24-71"])
        self.assertIn("-t", argv)
        self.assertNotIn("-np", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "8")
        dec = [a for a, _ in calls if "0" == a[a.index("-p") + 1]]
        self.assertTrue(dec and dec[0][dec[0].index("-n") + 1] == "1")
        self.assertEqual(env["GGML_SCHED_DEBUG"], "2")
        self.assertEqual(env["LD_LIBRARY_PATH"], str(binary.parent.resolve()))


    def test_bench_argv_refuses_parallel_sequences_and_never_emits_np(self):
        with self.assertRaises(ValueError):
            CL.bench_argv("b", "m", CE.Shape("decode", 1, 4))
        v = CL.bench_argv("b", "m", CE.Shape("verify", 5))
        self.assertEqual((v[v.index("-p") + 1], v[v.index("-n") + 1]), ("5", "0"))
        self.assertNotIn("-np", v)
