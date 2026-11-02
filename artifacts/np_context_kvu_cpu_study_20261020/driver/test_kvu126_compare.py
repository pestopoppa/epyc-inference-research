"""Synthetic producer-shaped records only; no native or model work."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import kvu126_compare as compare
import kvu_cpu_matrix as matrix


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"schema": "epyc.kvu126.comparison_plan.v1", "max_occupancy_fraction_gap": 0.02,
                     "min_occupancy_samples": 2, "min_waves_per_arm": 1, "arms": {},
                     "comparisons": [{"left": "np8_48k", "right": "np4_96k"}]}
        self.row("np8_48k", 8, 49152, 390000)
        self.row("np4_96k", 4, 98304, 391000)

    def write_json(self, name, data):
        raw = json.dumps(data).encode()
        (self.root / name).write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    def row(self, name, np, length, occupancy):
        row = {"schema": matrix.CAPTURE_SCHEMA, "row_sha256": None, "wave": 1,
               "binary_sha256": "a" * 64, "binary_realpath": "/fake/llama-server", "n_ctx_slot": length,
               "c": 524288, "np": np, "L": length, "prompt_mode": "depth", "ts": "2026-11-02T00:00:00Z",
               "cpu_list": "24-47", "threads": 24, "loadavg_start": [0, 0, 0], "result": "OK",
               "kv_unified_from_log": "true", "model": "fake-model", "mtp": True, "draft_depth": 4,
               "cache_idle_slots_argv": "off", "prompt_n": [400] * np, "cache_n": [length - 1000] * np,
               "decode_tps_per_stream": [2.0] * np, "decode_tps_sum": 2.0 * np}
        row["row_sha256"] = matrix.row_hash(row)
        wave = self.write_json(name + ".jsonl", row)
        native = {"schema": "epyc.kvu126.native_unique_cells.v1", "row_sha256": row["row_sha256"],
                  "decode_start_monotonic_s": 10.0, "decode_end_monotonic_s": 20.0,
                  "samples": [{"monotonic_s": 11.0, "unique_cells": occupancy},
                              {"monotonic_s": 19.0, "unique_cells": occupancy + 100}]}
        source = self.write_json(name + "-raw-native.json", native)
        native.update(native_source=source["path"], native_source_sha256=source["sha256"])
        obs = {"wave_file": wave, "wave": 1, "occupancy_file": self.write_json(name + "-native.json", native)}
        self.plan["arms"][name] = {"np": np, "L": length, "c": 524288, "observations": [obs]}

    def update_native(self, name, func):
        item = self.plan["arms"][name]["observations"][0]
        file = item["occupancy_file"]["path"]
        value = json.loads((self.root / file).read_bytes())
        func(value)
        source = self.write_json(value["native_source"], {k: value[k] for k in
                                 ("row_sha256", "decode_start_monotonic_s", "decode_end_monotonic_s", "samples")})
        value["native_source_sha256"] = source["sha256"]
        item["occupancy_file"] = self.write_json(file, value)

    def test_matched_comparison_reports_raw_resume_and_samples(self):
        result = compare.analyze(self.plan, self.root)
        self.assertTrue(result["comparisons"][0]["occupancy_matched"])
        self.assertEqual(result["arms"]["np8_48k"][0]["resume_cache_n"], [48152] * 8)
        self.assertIn("no performance or causal verdict", result["comparisons"][0]["interpretation"])

    def test_unmatched_is_descriptive_not_a_pass(self):
        self.update_native("np4_96k", lambda x: x["samples"][0].update(unique_cells=100))
        self.update_native("np4_96k", lambda x: x["samples"][1].update(unique_cells=200))
        self.assertFalse(compare.analyze(self.plan, self.root)["comparisons"][0]["occupancy_matched"])

    def test_native_source_resolves_relative_to_occupancy_receipt(self):
        item = self.plan["arms"]["np4_96k"]["observations"][0]
        receipt = item["occupancy_file"]
        content = json.loads((self.root / receipt["path"]).read_bytes())
        nested = self.root / "nested"
        nested.mkdir()
        (nested / content["native_source"]).write_bytes((self.root / content["native_source"]).read_bytes())
        (nested / Path(receipt["path"]).name).write_bytes((self.root / receipt["path"]).read_bytes())
        receipt["path"] = "nested/" + Path(receipt["path"]).name
        self.assertTrue(compare.analyze(self.plan, self.root)["comparisons"][0]["occupancy_matched"])

    def test_missing_and_post_window_occupancy_refuse(self):
        for bad in ([], [{"monotonic_s": 11, "unique_cells": 100}, {"monotonic_s": 21, "unique_cells": 100}]):
            with self.subTest(bad=bad):
                self.update_native("np4_96k", lambda x: x.update(samples=bad))
                with self.assertRaises(ValueError):
                    compare.analyze(self.plan, self.root)

    def test_wrong_binding_and_mutated_pinned_file_refuse(self):
        self.update_native("np4_96k", lambda x: x.update(row_sha256="c" * 64))
        with self.assertRaisesRegex(ValueError, "not bound"):
            compare.analyze(self.plan, self.root)
        item = self.plan["arms"]["np4_96k"]["observations"][0]["wave_file"]
        (self.root / item["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            compare.analyze(self.plan, self.root)

    def test_null_stream_and_errors_refuse(self):
        arm = self.plan["arms"]["np8_48k"]["observations"][0]
        row = json.loads((self.root / arm["wave_file"]["path"]).read_bytes())
        row["cache_n"][0] = None
        row["row_sha256"] = matrix.row_hash(row)
        arm["wave_file"] = self.write_json(arm["wave_file"]["path"], row)
        with self.assertRaisesRegex(ValueError, "cache_n"):
            compare.analyze(self.plan, self.root)

    def prospective(self):
        self.row("np8_32k", 8, 32768, 260000)
        self.row("np8_64k", 8, 65536, 510000)
        identity = {"binary_sha256": "a" * 64, "model_sha256": "d" * 64, "source_commit": "e" * 40,
                    "model": "fake-model", "cpu_list": "24-47", "threads": 24, "prompt_mode": "depth",
                    "mtp": True, "draft_depth": 4, "cache_idle_slots_argv": "off"}
        dsos = {"libllama.so": "f" * 64}
        build = {"schema": "epyc.kvu126.build_identity.v1", "source_commit": identity["source_commit"],
                 "binary_sha256": identity["binary_sha256"], "source_tree_clean": True, "dso_sha256": dsos}
        records = {"schema": "epyc.kvu126.evidence_index.v1", "build_receipt": self.write_json("build.json", build), "arms": {}}
        arms = {}
        for name, value in self.plan["arms"].items():
            arms[name] = {k: value[k] for k in ("np", "L", "c")}
            entries = value["observations"]
            for entry in entries:
                row = json.loads((self.root / entry["wave_file"]["path"]).read_bytes())
                receipt = {"schema": "epyc.kvu126.runtime_identity.v1", "binary_sha256": identity["binary_sha256"],
                           "model_sha256": identity["model_sha256"], "mapped_dso_sha256": dsos,
                           "pid_birth_identity": "1:123", "row_sha256": row["row_sha256"]}
                entry["runtime_receipt"] = self.write_json(name + "-runtime.json", receipt)
            records["arms"][name] = entries
        plan = {"schema": "epyc.kvu126.prospective_plan.v1", "expected_identity": identity, "arms": arms,
                "comparisons": [{"left": "np8_32k", "right": "np8_64k"},
                                {"left": "np8_48k", "right": "np4_96k"}],
                "noise_rule": "synthetic test only", "max_occupancy_fraction_gap": 0.02,
                "min_occupancy_samples": 2, "min_waves_per_arm": 1}
        return plan, records

    def test_prospective_declaration_is_separate_from_evidence(self):
        plan, records = self.prospective()
        self.assertNotIn("observations", plan["arms"]["np8_48k"])
        result = compare.analyze_prospective(plan, records, self.root)
        self.assertEqual(len(result["comparisons"]), 2)
        self.assertFalse(result["comparisons"][0]["occupancy_matched"])
        self.assertTrue(result["comparisons"][1]["occupancy_matched"])

    def test_prospective_rejects_wrong_pair_and_unbound_runtime(self):
        plan, records = self.prospective()
        plan["comparisons"].pop()
        with self.assertRaisesRegex(ValueError, "comparisons"):
            compare.analyze_prospective(plan, records, self.root)
        plan["comparisons"].append({"left": "np8_48k", "right": "np4_96k"})
        ref = records["arms"]["np8_48k"][0]["runtime_receipt"]
        receipt = json.loads((self.root / ref["path"]).read_bytes())
        receipt["row_sha256"] = "0" * 64
        records["arms"]["np8_48k"][0]["runtime_receipt"] = self.write_json(ref["path"], receipt)
        with self.assertRaisesRegex(ValueError, "not bound to wave"):
            compare.analyze_prospective(plan, records, self.root)


if __name__ == "__main__":
    unittest.main()
