from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import discovery_telemetry as T


class DiscoveryTelemetryTest(unittest.TestCase):
    def test_planner_event_is_dual_written_without_unbounded_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            telemetry.emit(
                "planner", "planner_completed", campaign_id="ak-discovery-deadbeef",
                hypothesis_id="akh-q4k", provider="codex", model="gpt-5.6-sol",
                effort="high", operation_key="0" * 64, result={"returncode": 0,
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
                    effort="high", operation_key="1" * 64,
                    result={"prompt": "secret"})
            self.assertFalse((Path(td) / "live").exists())

    def test_planner_refusal_is_typed_and_secret_free(self) -> None:
        reason = "planner estimated_diff_size is smaller (14 < 15)"
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            telemetry.emit(
                "planner", "planner_refused",
                campaign_id="ak-discovery-v16",
                hypothesis_id="akh-v2-q5-type-specific-dequant",
                provider="codex", model="gpt-5.6-sol", effort="high",
                operation_key="2" * 64,
                result={
                    "returncode": 0,
                    "stdout_sha256": "a" * 64,
                    "stderr_sha256": "b" * 64,
                    "refusal_type": "planner_output_refusal",
                    "refusal_reason_sha256": hashlib.sha256(
                        reason.encode()).hexdigest(),
                })
            row = json.loads((Path(td) / "live/planner.jsonl").read_text())
            self.assertEqual(row["event"], "planner_refused")
            self.assertEqual(row["result"]["refusal_type"],
                             "planner_output_refusal")
            self.assertNotIn(reason, json.dumps(row))
            self.assertNotIn("refusal_reason", row["result"])

    def test_raw_planner_refusal_reason_is_never_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            with self.assertRaisesRegex(T.TelemetryError, "non-allowlisted"):
                telemetry.emit(
                    "planner", "planner_refused",
                    campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                    provider="codex", model="gpt-5.6-sol", effort="high",
                    operation_key="3" * 64,
                    result={"refusal_reason": "may contain actor text"})
            self.assertFalse((Path(td) / "live").exists())

    def test_operation_event_is_idempotent_across_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            kwargs = dict(
                campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                provider="codex", model="gpt-5.6-sol", effort="high",
                operation_key="a" * 64,
                result={"returncode": 0, "stdout_sha256": "b" * 64,
                        "stderr_sha256": "c" * 64})
            telemetry.emit("planner", "planner_completed", **kwargs)
            telemetry.emit("planner", "planner_completed", **kwargs)
            all_lines = (Path(td) / "live/autokernel.jsonl").read_text().splitlines()
            planner_lines = (Path(td) / "live/planner.jsonl").read_text().splitlines()
            self.assertEqual(len(all_lines), 1)
            self.assertEqual(all_lines, planner_lines)
            self.assertRegex(json.loads(all_lines[0])["event_id"], r"^ake-[0-9a-f]{64}$")
            drifted = {**kwargs, "result": {**kwargs["result"],
                                             "stdout_sha256": "d" * 64}}
            with self.assertRaisesRegex(T.TelemetryError, "identity collision"):
                telemetry.emit("planner", "planner_completed", **drifted)
            self.assertEqual(len((Path(td) / "live/autokernel.jsonl").read_text().splitlines()), 1)

    def test_second_stream_failure_rolls_back_both_streams(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            original = T.DiscoveryTelemetry._write_event
            calls = 0

            def fail_second(fd, encoded):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected planner projection failure")
                return original(fd, encoded)

            with patch.object(T.DiscoveryTelemetry, "_write_event",
                              side_effect=fail_second), \
                    self.assertRaisesRegex(OSError, "projection failure"):
                telemetry.emit(
                    "planner", "planner_completed",
                    campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                    provider="codex", model="gpt-5.6-sol", effort="high",
                    operation_key="d" * 64,
                    result={"returncode": 0, "stdout_sha256": "e" * 64,
                            "stderr_sha256": "f" * 64})
            self.assertEqual((Path(td) / "live/autokernel.jsonl").read_bytes(), b"")
            self.assertEqual((Path(td) / "live/planner.jsonl").read_bytes(), b"")

    def test_retry_repairs_a_crash_partial_without_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            kwargs = dict(
                campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                provider="codex", model="gpt-5.6-sol", effort="high",
                operation_key="1" * 64,
                result={"returncode": 0, "stdout_sha256": "2" * 64,
                        "stderr_sha256": "3" * 64})
            telemetry.emit("planner", "planner_completed", **kwargs)
            authoritative = (Path(td) / "live/autokernel.jsonl").read_bytes()
            (Path(td) / "live/planner.jsonl").write_bytes(b"")
            telemetry.emit("planner", "planner_completed", **kwargs)
            self.assertEqual((Path(td) / "live/autokernel.jsonl").read_bytes(),
                             authoritative)
            self.assertEqual((Path(td) / "live/planner.jsonl").read_bytes(),
                             authoritative)

    def test_mirror_timestamp_drift_is_corruption_not_a_deduplicated_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "live"
            telemetry = T.DiscoveryTelemetry(root)
            kwargs = dict(
                campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                provider="codex", model="gpt-5.6-sol", effort="high",
                operation_key="4" * 64,
                result={"returncode": 0, "stdout_sha256": "5" * 64,
                        "stderr_sha256": "6" * 64})
            telemetry.emit("planner", "planner_completed", **kwargs)
            authoritative = root.joinpath("autokernel.jsonl").read_bytes()
            row = json.loads(root.joinpath("planner.jsonl").read_text())
            row["ts"] = "2026-08-19T12:34:56Z"
            root.joinpath("planner.jsonl").write_text(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            with self.assertRaisesRegex(T.TelemetryError, "mirror .* disagree"):
                telemetry.emit("planner", "planner_completed", **kwargs)
            self.assertEqual(root.joinpath("autokernel.jsonl").read_bytes(),
                             authoritative)
            self.assertNotEqual(root.joinpath("planner.jsonl").read_bytes(),
                                authoritative)

    def test_every_existing_row_is_validated_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "live"
            root.mkdir()
            garbage = b'{"arbitrary":"valid-json"}\n'
            root.joinpath("autokernel.jsonl").write_bytes(garbage)
            telemetry = T.DiscoveryTelemetry(root)
            with self.assertRaisesRegex(T.TelemetryError, "row schema"):
                telemetry.emit(
                    "autokernel", "critic_started",
                    campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                    provider="claude", model="claude-fable-5", effort="high",
                    operation_key="7" * 64)
            self.assertEqual(root.joinpath("autokernel.jsonl").read_bytes(),
                             garbage)

    def test_duplicate_or_malformed_v2_event_identity_blocks_append(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "live"
            telemetry = T.DiscoveryTelemetry(root)
            kwargs = dict(
                campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                provider="codex", model="gpt-5.6-sol", effort="high",
                operation_key="8" * 64)
            telemetry.emit("planner", "planner_started", **kwargs)
            original = root.joinpath("autokernel.jsonl").read_bytes()
            root.joinpath("autokernel.jsonl").write_bytes(original + original)
            with self.assertRaisesRegex(T.TelemetryError, "duplicate telemetry"):
                telemetry.emit("planner", "planner_started", **kwargs)

            root.joinpath("autokernel.jsonl").write_bytes(original)
            row = json.loads(root.joinpath("planner.jsonl").read_text())
            row["event_id"] = "ake-not-a-digest"
            root.joinpath("planner.jsonl").write_text(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            with self.assertRaisesRegex(T.TelemetryError, "event identity"):
                telemetry.emit("planner", "planner_started", **kwargs)

    def test_impossible_timestamp_and_unrelated_mirror_gap_block_append(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "live"
            telemetry = T.DiscoveryTelemetry(root)
            common = dict(
                campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                provider="codex", model="gpt-5.6-sol", effort="high")
            telemetry.emit("planner", "planner_started",
                           operation_key="a" * 64, **common)
            authoritative = root.joinpath("autokernel.jsonl").read_bytes()
            row = json.loads(root.joinpath("planner.jsonl").read_text())
            row["ts"] = "2026-99-99T99:99:99Z"
            root.joinpath("planner.jsonl").write_text(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            with self.assertRaisesRegex(T.TelemetryError, "timestamp"):
                telemetry.emit("planner", "planner_started",
                               operation_key="a" * 64, **common)

            root.joinpath("planner.jsonl").write_bytes(b"")
            with self.assertRaisesRegex(T.TelemetryError, "mirror sequences disagree"):
                telemetry.emit("planner", "planner_started",
                               operation_key="b" * 64, **common)
            self.assertEqual(root.joinpath("autokernel.jsonl").read_bytes(),
                             authoritative)

    def test_exact_legacy_v1_rows_remain_readable_during_migration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "live"
            root.mkdir()
            legacy = {
                "schema": T.LEGACY_SCHEMA,
                "ts": "2026-08-18T23:17:38.588034Z",
                "channel": "planner", "event": "planner_started",
                "campaign_id": "ak-discovery-v16", "hypothesis_id": "akh-q5",
                "provider": "codex", "model": "gpt-5.6-sol", "effort": "high",
            }
            encoded = (json.dumps(legacy, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode()
            root.joinpath("autokernel.jsonl").write_bytes(encoded)
            root.joinpath("planner.jsonl").write_bytes(encoded)
            T.DiscoveryTelemetry(root).emit(
                "planner", "planner_started",
                campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                provider="codex", model="gpt-5.6-sol", effort="high",
                operation_key="9" * 64)
            for name in ("autokernel.jsonl", "planner.jsonl"):
                rows = [json.loads(line) for line in root.joinpath(name).read_text().splitlines()]
                self.assertEqual([row["schema"] for row in rows],
                                 [T.LEGACY_SCHEMA, T.SCHEMA])

    def test_legacy_partial_or_v2_order_drift_blocks_migration_append(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "live"
            root.mkdir()
            legacy = {
                "schema": T.LEGACY_SCHEMA,
                "ts": "2026-08-18T23:17:38.588034Z",
                "channel": "planner", "event": "planner_started",
                "campaign_id": "ak-discovery-v16", "hypothesis_id": "akh-q5",
                "provider": "codex", "model": "gpt-5.6-sol", "effort": "high",
            }
            encoded = (json.dumps(legacy, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode()
            root.joinpath("autokernel.jsonl").write_bytes(encoded)
            root.joinpath("planner.jsonl").write_bytes(b"")
            telemetry = T.DiscoveryTelemetry(root)
            with self.assertRaisesRegex(T.TelemetryError,
                                        "mirror sequences disagree"):
                telemetry.emit(
                    "planner", "planner_started",
                    campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                    provider="codex", model="gpt-5.6-sol", effort="high",
                    operation_key="b" * 64)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "live"
            telemetry = T.DiscoveryTelemetry(root)
            common = dict(
                campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                provider="codex", model="gpt-5.6-sol", effort="high")
            telemetry.emit("planner", "planner_started",
                           operation_key="c" * 64, **common)
            telemetry.emit("planner", "planner_started",
                           operation_key="d" * 64, **common)
            rows = root.joinpath("planner.jsonl").read_text().splitlines()
            root.joinpath("planner.jsonl").write_text(
                "\n".join(reversed(rows)) + "\n")
            with self.assertRaisesRegex(T.TelemetryError,
                                        "mirror sequences disagree"):
                telemetry.emit("planner", "planner_started",
                               operation_key="e" * 64, **common)

    def test_event_family_and_refusal_exit_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry = T.DiscoveryTelemetry(Path(td) / "live")
            common = dict(
                campaign_id="ak-discovery-v16", hypothesis_id="akh-q5",
                provider="codex", model="gpt-5.6-sol", effort="high",
                operation_key="a" * 64)
            with self.assertRaisesRegex(T.TelemetryError, "wrong channel"):
                telemetry.emit("autokernel", "planner_started", **common)
            with self.assertRaisesRegex(T.TelemetryError, "successful actor exit"):
                telemetry.emit(
                    "planner", "planner_refused", **common,
                    result={
                        "returncode": 1, "stdout_sha256": "b" * 64,
                        "stderr_sha256": "c" * 64,
                        "refusal_type": "planner_output_refusal",
                        "refusal_reason_sha256": "d" * 64,
                    })
            self.assertFalse((Path(td) / "live").exists())


if __name__ == "__main__":
    unittest.main()
