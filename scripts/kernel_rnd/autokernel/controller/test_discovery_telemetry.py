from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from . import discovery_telemetry as T


class DiscoveryTelemetryTest(unittest.TestCase):
    def test_planner_event_is_dual_written_without_unbounded_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            telemetry.emit(
                "planner", "planner_completed", campaign_id="ak-discovery-deadbeef",
                hypothesis_id="akh-q4k", provider="codex", model="gpt-5.6-sol",
                effort="high", result={"returncode": 0,
                    "stdout_sha256": "a" * 64, "stderr_sha256": "b" * 64})
            all_row = json.loads((Path(td) / "live/autokernel.jsonl").read_text())
            planner_row = json.loads((Path(td) / "live/planner.jsonl").read_text())
            self.assertEqual(all_row, planner_row)
            self.assertEqual(all_row["schema"], T.SCHEMA)
            self.assertNotIn("prompt", json.dumps(all_row))
            self.assertNotIn("stdout", all_row["result"])

    def test_non_allowlisted_payload_is_refused_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            with self.assertRaisesRegex(T.TelemetryError, "non-allowlisted"):
                telemetry.emit(
                    "planner", "planner_completed", campaign_id="ak-discovery-x",
                    hypothesis_id="akh-x", provider="codex", model="gpt-5.6-sol",
                    effort="high", result={"prompt": "secret"})
            self.assertFalse((Path(td) / "live").exists())


if __name__ == "__main__":
    unittest.main()
