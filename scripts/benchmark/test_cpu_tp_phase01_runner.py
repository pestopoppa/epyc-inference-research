#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import cpu_tp_phase01_runner as runner


def canonical_receipt(body: dict) -> bytes:
    value = runner.self_hashed(body)
    return runner.canonical_bytes(value) + b"\n"


def gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def tiny_gguf(path: Path, metadata: dict[str, object]) -> None:
    raw = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata)))
    for key, value in metadata.items():
        raw += gguf_string(key)
        if isinstance(value, str):
            raw += struct.pack("<I", 8) + gguf_string(value)
        elif isinstance(value, int):
            raw += struct.pack("<I", 4) + struct.pack("<I", value)
        else:
            raise TypeError(value)
    path.write_bytes(raw)


class ParserTests(unittest.TestCase):
    def test_perf_parser_requires_numeric_complete_unmultiplexed_panel(self) -> None:
        text = "\n".join([
            "100;;cycles;200000000;100.00",
            "250;;instructions;200000000;95.00",
        ])
        parsed = runner.parse_perf_stat(text, ("cycles", "instructions"))
        self.assertEqual(parsed["events"]["cycles"]["value"], 100.0)
        self.assertEqual(parsed["events"]["instructions"]["time_running_ratio"], 0.95)
        for bad in (
            "<not counted>;;cycles;1;100\n2;;instructions;1;100",
            "1;;cycles;1;89\n2;;instructions;1;100",
            "1;;cycles;1;100",
            "1;;cycles;1;100\n2;;cycles;1;100\n3;;instructions;1;100",
        ):
            with self.assertRaises(runner.CpuTpError):
                runner.parse_perf_stat(bad, ("cycles", "instructions"))

    def test_uprof_parser_requires_exact_long_form_and_positive_duration(self) -> None:
        header = "metric_group,metric,scope,scope_id,value,unit,duration_seconds\n"
        text = header + (
            "memory,total_read,package,0,123.5,GB/s,0.2\n"
            "ipc,ipc,system,all,1.25,ratio,0.2\n"
        )
        parsed = runner.parse_uprof_pcm(text, ("memory", "ipc"))
        self.assertEqual(parsed["metric_groups"]["memory"][0]["value"], 123.5)
        with self.assertRaisesRegex(runner.CpuTpError, "duration"):
            runner.parse_uprof_pcm(header + "memory,total,system,all,1,GB/s,0\n", ("memory",))
        with self.assertRaisesRegex(runner.CpuTpError, "header"):
            runner.parse_uprof_pcm("metric,value\nmemory,1\n", ("memory",))
        with self.assertRaisesRegex(runner.CpuTpError, "extra CSV"):
            runner.parse_uprof_pcm(
                header + "memory,total,system,all,1,GB/s,0.2,rogue\n", ("memory",))

    def test_region_lock_parser_is_exact(self) -> None:
        value = runner.parse_region_lock_status("q0 free\nq1 free\nq2 free\nq3 free\n")
        self.assertTrue(value["all_free"])
        with self.assertRaises(runner.CpuTpError):
            runner.parse_region_lock_status("q0 free\nq1 free\nq2 free\n")


class ScheduleTests(unittest.TestCase):
    def test_fixed_schedules_are_reproducible_and_have_no_extension(self) -> None:
        first = runner.stopping_rules()
        second = runner.stopping_rules()
        self.assertEqual(first, second)
        self.assertEqual(len(first["phase0"]["schedule"]), 10)
        self.assertEqual(len(first["phase0"]["panel_schedule"]), 30)
        self.assertEqual(len(first["phase1"]["latin_square_schedule"]), 30)
        counts = {name: 0 for name in (
            "central-reduce-broadcast", "binary-tree", "reduce-scatter-all-gather")}
        for row in first["phase1"]["latin_square_schedule"]:
            self.assertEqual(set(row["algorithms"]), set(counts))
            for name in row["algorithms"]:
                counts[name] += 1
        self.assertEqual(set(counts.values()), {30})
        self.assertTrue(first["phase1"]["no_extension"])
        self.assertEqual(first["phase1"]["allreduce_calls_per_sample"], 128)


class ModelAndTopologyTests(unittest.TestCase):
    def test_streaming_gguf_metadata_reader_extracts_tp_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            tiny_gguf(path, runner.EXPECTED_MODEL_METADATA)
            result = runner.read_model_metadata(path)
            self.assertEqual(result["selected"], runner.EXPECTED_MODEL_METADATA)

    def test_topology_derives_physical_rank_masks_from_live_rows(self) -> None:
        summary = {"lscpu": [
            {"field": "Vendor ID:", "data": "AuthenticAMD"},
            {"field": "Model name:", "data": "AMD EPYC 9655 96-Core Processor"},
            {"field": "CPU family:", "data": "26"},
            {"field": "Model:", "data": "2"},
            {"field": "Socket(s):", "data": "1"},
        ]}
        cpus = []
        for cpu in range(192):
            core = cpu % 96
            cpus.append({"cpu": cpu, "node": core // 24, "socket": 0,
                         "core": core, "online": True})
        body, reasons = runner.topology_attestation(
            json.dumps(summary), json.dumps({"cpus": cpus}),
            "available: 4 nodes (0-3)\n",
        )
        self.assertEqual(reasons, [])
        self.assertEqual(body["physical_cpus_by_node"]["3"], list(range(72, 96)))


class ReceiptAndGateTests(unittest.TestCase):
    def test_all_shipped_json_schemas_parse_and_have_unique_ids(self) -> None:
        schemas = [json.loads(path.read_text()) for path in sorted(runner.SCHEMA_DIR.glob("*.json"))]
        self.assertEqual(len(schemas), 5)
        self.assertEqual(len({schema["$id"] for schema in schemas}), 5)
        self.assertTrue(all(schema["additionalProperties"] is False for schema in schemas))

    def test_receipt_is_canonical_self_hashed_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            written = runner.write_receipt(path, {"schema": "test.v1", "value": 1})
            self.assertEqual(written["receipt_sha256"], runner.content_sha256(
                {"schema": "test.v1", "value": 1}))
            self.assertEqual(path.read_bytes(), runner.canonical_bytes(written) + b"\n")
            with self.assertRaises(FileExistsError):
                runner.write_receipt(path, {"schema": "test.v1", "value": 2})

    def test_ratification_requires_external_file_hash_and_exact_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol.md"
            protocol.write_text("protocol\n")
            runfile = root / "runner.py"
            runfile.write_text("runner\n")
            protocol_row = runner.file_identity(protocol)
            runner_row = runner.file_identity(runfile)
            schemas = {"schema_manifest_sha256": "a" * 64}
            receipt = root / "ratification.json"
            receipt.write_bytes(canonical_receipt({
                "schema": runner.RATIFICATION_SCHEMA, "status": "ratified",
                "protocol_id": runner.PROTOCOL_ID,
                "protocol_sha256": protocol_row["sha256"],
                "runner_sha256": runner_row["sha256"],
                "schema_manifest_sha256": "a" * 64,
                "ratified_at": "2026-08-20T00:00:00+00:00", "ratified_by": "operator",
            }))
            file_hash = runner.sha256_file(receipt)
            verified = runner.verify_ratification(
                receipt, file_hash, protocol=protocol_row, runner=runner_row,
                schemas=schemas,
            )
            self.assertEqual(verified["status"], "ratified")
            with self.assertRaisesRegex(runner.CpuTpError, "file SHA"):
                runner.verify_ratification(
                    receipt, "0" * 64, protocol=protocol_row, runner=runner_row,
                    schemas=schemas,
                )

    def test_execute_has_no_unratified_or_live_benchmark_path(self) -> None:
        blocked = {"ratification": {"status": "absent"}}
        with mock.patch.object(runner, "collect_preflight", return_value=blocked):
            self.assertEqual(runner.main([
                "execute", "--protocol-file", "/tmp/protocol"]), 2)
        ratified = {"ratification": {"status": "valid"}}
        with mock.patch.object(runner, "collect_preflight", return_value=ratified):
            self.assertEqual(runner.main([
                "execute", "--protocol-file", "/tmp/protocol"]), 2)


if __name__ == "__main__":
    unittest.main()
