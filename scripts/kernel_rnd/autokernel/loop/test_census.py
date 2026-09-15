"""Zero-hardware acceptance fixtures for the per-workload census."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from autokernel.controller.workload_contract import WorkloadCensus
from autokernel.loop import census


def recorded_27b_log() -> str:
    """Compact construction of the recorded G4 multiset (27,516 scheduler nodes)."""
    banner = "Device 0: AMD Instinct MI210, gfx90a:sramecc+:xnack-\n"
    cpu = "node #  0 (  GET_ROWS): embed [  CPU         ]\n" * 12
    gpu = "node # 44 (GATED_DELT): state [ROCm0         ]\n" * (27516 - 12)
    return banner + cpu + gpu


class SchedulerFixture(unittest.TestCase):
    def test_recorded_27b_counts_are_exact(self):
        graph = census.parse_scheduler_graph(recorded_27b_log())
        self.assertEqual(graph["nodes_total"], 27516)
        self.assertEqual(graph["op_backend"].get("SSM_SCAN", {}), {})
        self.assertEqual(graph["op_backend"]["GET_ROWS"]["CPU"], 12)

    def test_ansi_mangled_vacuous_log_is_unknown(self):
        def runner(*_args, **_kwargs):
            return mock.Mock(returncode=0, stdout="\033[32mDevice 0: MI210\033[0m\n", stderr="")
        row = census.run_dispatch_probe(Path("/bin/false"), Path("m.gguf"),
                                        census.Shape("decode", 16), runner=runner)
        self.assertEqual(row["nodes_total"], 0)
        self.assertEqual(row["state"], census.UNKNOWN)

    def test_missing_workload_expected_op_is_unknown(self):
        def runner(*_args, **_kwargs):
            text = ("Device 0: AMD Instinct MI210, gfx90a\n" +
                    "node #  1 (   MUL_MAT): x [ROCm0         ]\n" * 1002)
            return mock.Mock(returncode=0, stdout=text, stderr="")
        row = census.run_dispatch_probe(
            Path("/bin/false"), Path("m.gguf"), census.Shape("decode", 16),
            expected_ops={"GATED_DELT"}, require_device=True, runner=runner)
        self.assertFalse(row["vacuous_guards"]["expected_ops_present"])
        self.assertEqual(row["state"], census.UNKNOWN)


class CensusValidity(unittest.TestCase):
    def _record(self, nodes=27516):
        identity = {"kernel": {"commit": "abc"}, "recipe": {"recipe_sha256": "def"}}
        key = census.freshness_key(identity)
        return ({"schema": census.SCHEMA,
                 "identity": {"workload_id": "w1", **identity},
                 "graph": {"shape_envelope": [{"phase": "decode", "n_tokens": 16,
                                                  "n_seq": 1}],
                           "per_shape": [{"state": census.OBSERVED if nodes else census.UNKNOWN,
                                          "nodes_total": nodes}]},
                 "validity": {"freshness_key": key}}, key)

    def test_stale_key_is_unknown(self):
        record, _ = self._record()
        self.assertEqual(census.census_state(record, "not-the-current-key"), census.UNKNOWN)

    def test_zero_nodes_is_unknown(self):
        record, key = self._record(nodes=0)
        self.assertEqual(census.census_state(record, key), census.UNKNOWN)

    def test_fresh_observed_record_is_observed(self):
        record, key = self._record()
        self.assertEqual(census.census_state(record, key), census.OBSERVED)


class Producer(unittest.TestCase):
    def test_emits_schema_weights_envelope_and_stores_under_requested_census_dir(self):
        weights = WorkloadCensus(path="m.gguf", architecture="qwen35", n_embd=5120,
                                 tensor_types={"Q8_0": 999, "F32": 4})

        def probe(_binary, _model, shape, **_kwargs):
            return {"shape": shape.to_dict(), "nodes_total": 27516,
                    "op_backend": {"GET_ROWS": {"CPU": 12}}, "device_seen": True,
                    "rc": 0, "state": census.OBSERVED,
                    "vacuous_guards": {"nodes_total_gt_min": True}}

        with mock.patch.object(census.workload_contract, "read_census", return_value=weights):
            record = census.produce_census(
                workload_id="w1", workload_signature="sig", model=Path("m.gguf"),
                identity={"kernel": {"commit": "abc"}, "recipe": {"recipe_sha256": "def"}},
                shapes=[census.Shape("prefill", 512), census.Shape("decode", 16)],
                binary=Path("llama-bench"), probe=probe)
        self.assertEqual(record["schema"], "epyc.autokernel.workload_census.v1")
        self.assertEqual(record["weights"]["dominant_quant"], "Q8_0")
        self.assertEqual(len(record["graph"]["shape_envelope"]), 2)
        self.assertEqual(record["validity"]["state"], census.OBSERVED)
        with tempfile.TemporaryDirectory() as tmp:
            path = census.store_census(record, Path(tmp) / "loop-memory" / "census")
            self.assertEqual(path.parent.name, "census")
            self.assertEqual(json.loads(path.read_text())["identity"]["workload_id"], "w1")


class ShapeEnvelope(unittest.TestCase):
    def test_recipe_resolves_prefill_decode_and_draft_verify(self):
        shapes = census.shape_envelope({"ubatch": 512, "np": 2, "draft_max": 8})
        self.assertEqual([shape.to_dict() for shape in shapes], [
            {"phase": "prefill", "n_tokens": 512, "n_seq": 2},
            {"phase": "decode", "n_tokens": 1, "n_seq": 2},
            {"phase": "verify", "n_tokens": 8, "n_seq": 2},
        ])

    def test_prefill_decode_and_verify_widths_are_encoded(self):
        prefill = census.shape_argv(Path("bench"), Path("m"), census.Shape("prefill", 512))
        decode = census.shape_argv(Path("bench"), Path("m"), census.Shape("decode", 16, 1))
        verify = census.shape_argv(Path("bench"), Path("m"), census.Shape("verify", 8, 4))
        self.assertEqual(prefill[prefill.index("-p") + 1], "512")
        self.assertEqual(decode[decode.index("-np") + 1], "1")
        self.assertEqual(verify[verify.index("-np") + 1], "4")
        self.assertIn("-v", verify)


if __name__ == "__main__":
    unittest.main()
