#!/usr/bin/env python3
"""Focused policy and scanner regressions; no model or production process."""

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
import sandbox3
import run3

HERE = pathlib.Path(__file__).resolve().parent


class A6Tests(unittest.TestCase):
    def test_source_import_and_help_without_historical_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            missing = pathlib.Path(td) / "missing-c95"
            env = {**os.environ, "DS41_C95_BASE": str(missing)}
            for script in ("run3.py", "sandbox3.py"):
                result = subprocess.run(
                    [sys.executable, str(HERE / script), "--help"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            source = subprocess.run(
                [sys.executable, "-c", "import run3; run3._pinned_aklib_source('actor_context')"],
                env={**env, "PYTHONPATH": str(HERE)},
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertNotEqual(source.returncode, 0)
            self.assertIn("SOURCE_MISSING: pinned C95 aklib snapshot required", source.stderr)

    def test_repeated_output_claim_preserves_provenance_and_wire(self):
        with tempfile.TemporaryDirectory() as td:
            parent = pathlib.Path(td)
            old = sandbox3.claim_fresh_output(parent, "normal")
            provenance = old / "preexposure-provenance.json"
            wire = old / "wire/wire.jsonl"
            wire.parent.mkdir()
            provenance.write_bytes(b'{"run_id":"old"}\n')
            wire.write_bytes(b'{"old":true}\n')
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (provenance, wire)}
            with self.assertRaises(FileExistsError):
                sandbox3.claim_fresh_output(parent, "normal")
            with self.assertRaises(FileExistsError):
                run3.require_fresh_call_slot(old)
            for p, (content, mtime) in before.items():
                self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), (content, mtime))

    def test_preexposure_exists_before_first_request_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            directory = pathlib.Path(td)
            first_request = Mock(
                side_effect=lambda: self.assertTrue(
                    (directory / "preexposure-provenance.json").is_file()
                )
            )
            ref = sandbox3.emit_preexposure(
                directory, {"run_id": "test", "verified_backend_identity": None}
            )
            first_request()
            first_request.assert_called_once_with()
            original = (directory / "preexposure-provenance.json").read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), ref["sha256"])
            with self.assertRaises(FileExistsError):
                sandbox3.emit_preexposure(directory, {"run_id": "rewritten"})
            self.assertEqual((directory / "preexposure-provenance.json").read_bytes(), original)

    def test_context_anchor_allow_is_exact(self):
        original = sandbox3.BASE
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for context, anchor in [("C2", "anchor-gen-006"), ("C4", "anchor-gen-007")]:
                d = root / "contexts" / context
                d.mkdir(parents=True)
                (d / "meta.json").write_text(json.dumps({"anchor_build": anchor}))
            sandbox3.BASE = root
            try:
                for context, anchor in [("C2", "anchor-gen-006"), ("C4", "anchor-gen-007")]:
                    p = sandbox3.c95_policy(
                        cutoff=1,
                        ws_root=root / "w",
                        call_dir=root / "r",
                        scratch_clone=root / "lane",
                        context_id=context,
                    )
                    granted = [x["path"] for x in p["allow"] if "/ctx-campaign/store/" in x["path"]]
                    self.assertEqual(granted, [str(root / "ctx-campaign/store" / anchor)])
                    self.assertNotIn(str(root / "ctx-campaign"), [x["path"] for x in p["allow"]])
            finally:
                sandbox3.BASE = original

    def test_raw_hits_retained_and_denial_correlated(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "events.jsonl"
            p.write_text(
                json.dumps(
                    {
                        "item": {
                            "command": "cat /bad/053c3bd82",
                            "aggregated_output": "cat: /bad/053c3bd82: Permission denied",
                            "exit_code": 1,
                        }
                    }
                )
                + "\n"
            )
            raw = run3.scan_contamination([p], "0c8b896ee")
            checked = run3.crosscheck_contamination([p], "0c8b896ee")
            self.assertEqual(raw["053c3bd82"], 2)
            self.assertEqual(checked["053c3bd82"]["denied_attempts"], 2)
            self.assertEqual(checked["053c3bd82"]["output_without_denial"], 0)

    def test_historical_successful_read_fixture(self):
        p = HERE / "fixtures/a3warm-event22.raw.jsonl"
        self.assertEqual(
            hashlib.sha256(p.read_bytes()).hexdigest(),
            "3097f03b548859b84dcf2bd14d93df4b0215241e789a1e0860a9932509c010f6",
        )
        item = json.loads(p.read_text())["item"]
        self.assertEqual(item["exit_code"], 0)
        self.assertIn("anchor-gen-007/provenance.json", item["command"])
        self.assertIn("053c3bd82", item["aggregated_output"])
        checked = run3.crosscheck_contamination([p], "0c8b896ee")
        self.assertGreater(checked["053c3bd82"]["output_without_denial"], 0)


if __name__ == "__main__":
    unittest.main()
