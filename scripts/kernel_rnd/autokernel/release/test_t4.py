#!/usr/bin/env python3
"""Unit and red-team tests for the read-only T4 release instrument."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path

from .. import schemas
from . import packager, t4

CUTOVER = "2026-08-12T00:00:00Z"
NOW_OPEN = "2026-08-14T00:00:00Z"
NOW_CLOSEABLE = "2026-08-20T00:00:00Z"
ROLES = ("frontdoor", "worker_general")


def digest(label: str) -> str:
    return schemas.content_hash({"t4_fixture": label})


def bands() -> tuple:
    edges = {
        packager.SIGNAL_THROUGHPUT: ("tokens_per_s", 40.0, None),
        packager.SIGNAL_LATENCY: ("requests_per_s_inverse", 10.0, None),
        packager.SIGNAL_ERROR_RATES: ("fraction", None, 0.01),
        packager.SIGNAL_MEMORY: ("gib_headroom", 8.0, None),
        packager.SIGNAL_QUALITY: ("quality_score", 0.8, None),
        packager.SIGNAL_SUPERVISOR: ("events", None, 0.0),
    }
    return tuple(packager.WatchSignalBand(
        signal_id=signal, unit=unit,
        basis_ref=f"evidence://E9/{signal}",
        noise_reference_ref="measurement://standing-noise-reference",
        lower=lower, upper=upper, roles=ROLES)
        for signal, (unit, lower, upper) in edges.items())


def window(**overrides) -> packager.WatchWindow:
    fields = {
        "window_id": "ak-t4-v10", "package_id": "akr-v10-001",
        "owner": "daniele", "incumbent_era": "E9", "candidate_era": "E10",
        "affected_roles": ROLES, "min_duration_days": 7,
        "min_volume_by_role": {role: 100 for role in ROLES}, "bands": bands(),
        "bands_fixed_at": "2026-08-11T20:00:00Z", "opens_at": CUTOVER,
        "close_step": packager.WatchWindowCloseStep(owner="daniele"),
        "rollback_anchor_ref": "archive://production-consolidated-v9",
        "activation_manifest_ref": "evidence://release/v10/activation-manifest",
        "activation_manifest_sha256": t4.activation_manifest_sha256(
            package_id="akr-v10-001", candidate_era="E10",
            expected_roles=expectations()),
    }
    fields.update(overrides)
    return packager.WatchWindow(**fields)


def expectations() -> tuple:
    return tuple(t4.LiveRoleExpectation(
        role=role, backend="llama_gpu" if role == "frontdoor" else "llama_cpu",
        binary_path=f"/mnt/raid0/llm/kernels/v10/{role}/llama-server",
        binary_sha256=digest(f"binary:{role}"),
        linkage_root=f"/mnt/raid0/llm/kernels/v10/{role}",
        linkage_sha256=digest(f"linkage:{role}")) for role in ROLES)


def live_roles(**by_role) -> tuple:
    rows = []
    for index, expected in enumerate(expectations(), start=1):
        fields = {
            **expected.to_dict(), "pid": 4000 + index,
            "enumerated_role_pids": (4000 + index,),
            "process_start_ticks": 100000 + index,
            "boot_id": "b3d927ef-4834-4f17-ae22-541c1373f0bf",
            "process_started_at": "2026-08-12T00:01:00Z",
            "captured_at": "2026-08-12T00:05:00Z",
            "linkage_verifier": (
                "/mnt/raid0/llm/epyc-inference-research/"
                "scripts/utils/verify_ggml_linkage.sh"),
            "linkage_exit_code": 0,
            "evidence_ref": f"evidence://live/{expected.role}",
            "evidence_sha256": digest(f"live-receipt:{expected.role}"),
        }
        fields.update(by_role.get(expected.role, {}))
        rows.append(t4.LiveRoleReceipt(**fields))
    return tuple(rows)


def probes(**overrides) -> tuple:
    rows = [t4.ProbeReceipt(
        probe_kind=t4.PROBE_ROLE_CANARY, role=role,
        observed_at="2026-08-12T00:06:00Z", exit_code=0, status_code=None,
        semantic_success=True, evidence_ref=f"evidence://canary/{role}",
        evidence_sha256=digest(f"canary:{role}"))
        for role in ROLES]
    for kind in t4.GLOBAL_PROBE_KINDS:
        fields = {
            "probe_kind": kind, "role": None,
            "observed_at": "2026-08-12T00:07:00Z", "exit_code": 0,
            "status_code": 200 if "health" in kind else None,
            "semantic_success": True, "evidence_ref": f"evidence://probe/{kind}",
            "evidence_sha256": digest(f"probe:{kind}"),
        }
        fields.update(overrides.get(kind, {}))
        rows.append(t4.ProbeReceipt(**fields))
    return tuple(rows)


def observations(*, era: str = "E10", values=None) -> tuple:
    actual = {
        packager.SIGNAL_THROUGHPUT: 50.0,
        packager.SIGNAL_LATENCY: 12.0,
        packager.SIGNAL_ERROR_RATES: 0.0,
        packager.SIGNAL_MEMORY: 16.0,
        packager.SIGNAL_QUALITY: 0.9,
        packager.SIGNAL_SUPERVISOR: 0.0,
    }
    actual.update(values or {})
    return tuple(packager.WatchObservation(
        signal_id=signal, value=value, observed_at="2026-08-13T00:00:00Z",
        era_label=era, samples_ref=f"evidence://watch/{signal}")
        for signal, value in actual.items())


def progress(*, closeable: bool = False, era: str = "E10", values=None,
             window_value: packager.WatchWindow | None = None) -> packager.WatchWindowProgress:
    declared = window_value or window()
    return packager.WatchWindowProgress(
        now=NOW_CLOSEABLE if closeable else NOW_OPEN,
        volume_by_role={role: 150 for role in ROLES},
        bands_sha256=declared.bands_sha256(), observations=observations(era=era, values=values))


def rollback(**overrides) -> t4.RollbackAnchorReceipt:
    fields = {
        "anchor_ref": "archive://production-consolidated-v9",
        "verified_at": "2026-08-12T00:08:00Z", "available": True,
        "immutable": True, "evidence_ref": "evidence://archive/v9",
        "evidence_sha256": digest("archive-v9-verification"),
    }
    fields.update(overrides)
    return t4.RollbackAnchorReceipt(**fields)


def request(**overrides) -> t4.T4Request:
    declared = overrides.pop("watch_window", window())
    fields = {
        "request_id": "akt4-v10-r1", "cutover_at": CUTOVER,
        "watch_window": declared, "expected_roles": expectations(),
        "live_roles": live_roles(), "probes": probes(), "rollback_anchor": rollback(),
        "progress": progress(window_value=declared),
    }
    fields.update(overrides)
    return t4.T4Request(**fields)


class TestT4HappyPath(unittest.TestCase):

    def test_healthy_open_window_continues(self):
        result = t4.evaluate_t4(request())
        self.assertEqual(result.activation_check.outcome, schemas.PASS)
        self.assertEqual(result.recommendation, t4.RECOMMEND_CONTINUE)
        self.assertEqual(result.watch.state, packager.WATCH_STATE_OPEN)

    def test_clean_closeable_window_recommends_keep_and_human_close(self):
        result = t4.evaluate_t4(request(progress=progress(closeable=True)))
        self.assertEqual(result.recommendation, t4.RECOMMEND_KEEP)
        self.assertEqual(result.watch.state, packager.WATCH_STATE_CLOSEABLE)
        self.assertIn("NOT A CLAIM", result.to_dict()["record_class"])

    def test_runner_implements_release_tier_seam(self):
        runner = t4.T4Runner()
        self.assertEqual(runner.tier, "T4")
        self.assertIsInstance(runner.evaluate_release(request()), t4.T4Result)
        with self.assertRaises(t4.T4InputError):
            runner.evaluate_release({"tier": "T4"})

    def test_request_round_trips_strictly(self):
        original = request()
        decoded = t4.T4Request.from_dict(original.to_dict())
        self.assertEqual(decoded.to_dict(), original.to_dict())
        self.assertEqual(decoded.fingerprint(), original.fingerprint())


class TestActivationFailsClosed(unittest.TestCase):

    def test_stale_process_raises_decision_package(self):
        stale = live_roles(frontdoor={"process_started_at": "2026-08-11T23:00:00Z"})
        result = t4.evaluate_t4(request(live_roles=stale))
        self.assertEqual(result.recommendation, t4.RECOMMEND_DECISION)
        self.assertIn("stale", " ".join(result.role_checks["frontdoor"].reasons))

    def test_a_stale_peer_process_cannot_hide_behind_the_new_pid(self):
        result = t4.evaluate_t4(request(live_roles=live_roles(
            frontdoor={"enumerated_role_pids": (4001, 3999)})))
        self.assertEqual(result.recommendation, t4.RECOMMEND_DECISION)
        self.assertIn("enumeration", " ".join(result.role_checks["frontdoor"].reasons))

    def test_binary_path_hash_backend_and_linkage_root_are_all_bound(self):
        mutations = {
            "binary_path": "/tmp/some-other-binary",
            "binary_sha256": digest("wrong-binary"),
            "backend": "llama_cpu",
            "linkage_root": "/tmp/wrong-libraries",
            "linkage_sha256": digest("wrong-linkage"),
        }
        for name, value in mutations.items():
            with self.subTest(field=name):
                result = t4.evaluate_t4(request(
                    live_roles=live_roles(frontdoor={name: value})))
                self.assertEqual(result.recommendation, t4.RECOMMEND_DECISION)
                self.assertIn(name, " ".join(result.role_checks["frontdoor"].reasons))

    def test_wrong_or_failed_linkage_verifier_fails(self):
        for change in ({"linkage_verifier": "/tmp/fake-verifier"},
                       {"linkage_exit_code": 1}):
            with self.subTest(change=change):
                result = t4.evaluate_t4(request(
                    live_roles=live_roles(worker_general=change)))
                self.assertEqual(result.recommendation, t4.RECOMMEND_DECISION)

    def test_failed_probe_raises_decision_package(self):
        result = t4.evaluate_t4(request(probes=probes(
            **{t4.PROBE_API_HEALTH: {"status_code": 503,
                                    "semantic_success": False}})))
        self.assertEqual(result.recommendation, t4.RECOMMEND_DECISION)
        self.assertEqual(result.probe_checks[t4.PROBE_API_HEALTH].outcome, schemas.FAIL)

    def test_probe_without_semantic_result_is_incomplete(self):
        result = t4.evaluate_t4(request(probes=probes(
            **{t4.PROBE_SPEECH_SMOKE: {"semantic_success": None}})))
        self.assertEqual(result.recommendation, t4.RECOMMEND_INCOMPLETE)

    def test_http_health_without_status_is_incomplete(self):
        result = t4.evaluate_t4(request(probes=probes(
            **{t4.PROBE_API_HEALTH: {"status_code": None}})))
        self.assertEqual(result.recommendation, t4.RECOMMEND_INCOMPLETE)

    def test_unavailable_or_mutable_rollback_anchor_fails(self):
        for field in ("available", "immutable"):
            with self.subTest(field=field):
                result = t4.evaluate_t4(request(
                    rollback_anchor=rollback(**{field: False})))
                self.assertEqual(result.recommendation, t4.RECOMMEND_DECISION)

    def test_watch_alarm_raises_decision_package(self):
        changed = {packager.SIGNAL_THROUGHPUT: 30.0}
        result = t4.evaluate_t4(request(progress=progress(values=changed)))
        self.assertEqual(result.recommendation, t4.RECOMMEND_DECISION)
        self.assertIn(packager.SIGNAL_THROUGHPUT, result.watch.alarms)

    def test_wrong_era_watch_evidence_is_incomplete(self):
        result = t4.evaluate_t4(request(progress=progress(era="E9")))
        self.assertEqual(result.recommendation, t4.RECOMMEND_INCOMPLETE)
        self.assertEqual(set(result.watch.unevaluable), set(packager.REQUIRED_WATCH_SIGNALS))


class TestCoverageAndTimeRefusals(unittest.TestCase):

    def test_missing_affected_role_receipt_is_refused(self):
        with self.assertRaisesRegex(t4.T4InputError, "cover affected roles exactly"):
            request(live_roles=live_roles()[:-1])

    def test_missing_role_canary_is_refused(self):
        without = tuple(p for p in probes()
                        if not (p.probe_kind == t4.PROBE_ROLE_CANARY
                                and p.role == "frontdoor"))
        with self.assertRaisesRegex(t4.T4InputError, "role canaries"):
            request(probes=without)

    def test_missing_global_probe_is_refused(self):
        without = tuple(p for p in probes() if p.probe_kind != t4.PROBE_SPEECH_SMOKE)
        with self.assertRaisesRegex(t4.T4InputError, "global probe"):
            request(probes=without)

    def test_cutover_and_watch_open_must_be_one_boundary(self):
        with self.assertRaisesRegex(t4.T4InputError, "must equal"):
            request(cutover_at="2026-08-12T00:00:01Z")

    def test_post_progress_probe_is_refused(self):
        shifted = list(probes())
        shifted[0] = dataclasses.replace(shifted[0], observed_at="2026-08-15T00:00:00Z")
        with self.assertRaisesRegex(t4.T4InputError, "outside"):
            request(probes=tuple(shifted))

    def test_pre_cutover_watch_sample_is_refused(self):
        base = progress()
        changed = (dataclasses.replace(base.observations[0],
                                       observed_at="2026-08-11T00:00:00Z"),
                   *base.observations[1:])
        with self.assertRaisesRegex(t4.T4InputError, "outside"):
            request(progress=dataclasses.replace(base, observations=changed))


class TestStrictJsonAndCli(unittest.TestCase):

    def test_unknown_request_field_is_refused(self):
        document = request().to_dict()
        document["execute_rollback"] = True
        with self.assertRaisesRegex(t4.T4InputError, "unknown"):
            t4.T4Request.from_dict(document)

    def test_expectations_not_bound_by_release_manifest_are_refused(self):
        changed = list(expectations())
        changed[0] = dataclasses.replace(changed[0], binary_sha256=digest("after-cutover"))
        with self.assertRaisesRegex(t4.T4InputError, "manifest fixed"):
            request(expected_roles=tuple(changed))

    def test_tampered_derived_window_field_is_refused(self):
        document = request().to_dict()
        document["watch_window"]["bands_sha256"] = digest("different-bands")
        with self.assertRaisesRegex(t4.T4InputError, "derived fields"):
            t4.T4Request.from_dict(document)

    def test_moved_bands_progress_is_refused(self):
        base = progress()
        with self.assertRaises(packager.BandsNotFixedBeforeData):
            t4.evaluate_t4(request(progress=dataclasses.replace(
                base, bands_sha256=digest("moved-bands"))))

    def test_cli_emits_json_and_status_codes(self):
        cases = (
            (request(), t4.EXIT_OK, t4.RECOMMEND_CONTINUE),
            (request(progress=progress(values={packager.SIGNAL_THROUGHPUT: 1.0})),
             t4.EXIT_DECISION, t4.RECOMMEND_DECISION),
            (request(progress=progress(era="E9")),
             t4.EXIT_INCOMPLETE, t4.RECOMMEND_INCOMPLETE),
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "request.json"
            for item, expected_exit, expected_recommendation in cases:
                with self.subTest(recommendation=expected_recommendation):
                    path.write_text(json.dumps(item.to_dict()), encoding="utf-8")
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        exit_code = t4.main(("--request", str(path)))
                    self.assertEqual(exit_code, expected_exit)
                    self.assertEqual(json.loads(stdout.getvalue())["recommendation"],
                                     expected_recommendation)

    def test_cli_refuses_malformed_json(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "request.json"
            path.write_text("{not json", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(t4.main(("--request", str(path))), t4.EXIT_INPUT)
            self.assertIn(t4.RESULT_SCHEMA, stderr.getvalue())

    def test_cli_refuses_wrong_nested_types_without_traceback(self):
        document = request().to_dict()
        document["watch_window"]["affected_roles"] = 7
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "request.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(t4.main(("--request", str(path))), t4.EXIT_INPUT)
            error = json.loads(stderr.getvalue())
            self.assertIn("required array", error["error"])


class TestStructuralBoundary(unittest.TestCase):

    def test_module_has_no_live_or_mutating_capability(self):
        self.assertEqual(t4.audit_no_live_or_mutating_capability().outcome, schemas.PASS)

    def test_capability_audit_bites(self):
        source = "import subprocess\nsubprocess.run(['rollback'])\n"
        check = t4.audit_no_live_or_mutating_capability(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("denied module", " ".join(check.reasons))

    def test_result_contains_no_machine_authority(self):
        self.assertEqual(schemas.find_authority_flavoured_keys(
            t4.evaluate_t4(request()).to_dict()), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
