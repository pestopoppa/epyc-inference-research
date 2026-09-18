"""Hermetic acceptance for the standalone client against the real service."""
from __future__ import annotations

import configparser
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import time
from urllib.parse import urlsplit

import pytest

from . import campaign_control as control
from . import campaign_command_v2
from . import campaign_cli
from . import campaign_service
from . import standalone_cli as cli
from .test_campaign_control import _resolved


class RunningService:
    def __init__(self, controller, token="fixture-secret"):
        self.controller = controller
        self.token = token
        self.service = campaign_service.CampaignHTTPService(
            controller, "127.0.0.1", 0, token)
        self.service.start()
        self.endpoint = f"http://{self.service.address[0]}:{self.service.address[1]}"

    def close(self):
        self.service.close()


def _pins(controller):
    return [
        "--expect-campaign-id", controller.resolved.campaign_id,
        "--expect-config-generation", str(controller.config_generation),
        "--expect-config-digest", controller.config_digest,
        "--expect-supervisor-incarnation", str(controller.supervisor_incarnation),
    ]


def _invoke(capsys, monkeypatch, running, action, *extra):
    monkeypatch.setenv(cli.TOKEN_ENV, running.token)
    result = cli.main([action, "--endpoint", running.endpoint,
                       *_pins(running.controller), *extra])
    captured = capsys.readouterr()
    body = json.loads(captured.out) if captured.out else None
    return result, body, captured.err


@pytest.fixture
def running(tmp_path):
    controller = control.CampaignController(
        _resolved(), tmp_path / "service", snapshot_version=2)
    controller.__enter__()
    server = RunningService(controller)
    try:
        yield server
    finally:
        server.close()
        controller.close()


def test_real_service_inspect_preflight_and_management_only_status(
        running, capsys, monkeypatch, tmp_path):
    resolved_enrollment = tmp_path / "resolved-enrollment.json"
    resolved_enrollment.write_text(json.dumps(campaign_cli.build_output(
        running.controller.resolved, verify_artifacts=False)), encoding="utf-8")
    before = set(tmp_path.rglob("*"))
    before_bytes = {path: path.read_bytes() for path in before if path.is_file()}
    assert cli.main(["inspect", "--endpoint", running.endpoint]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["semantic_liveness"] == "not_inspected"
    assert cli.main(["preflight", "--endpoint", running.endpoint,
                     "--resolved-campaign", str(resolved_enrollment)]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["disposition"] == "not_ready"
    assert preflight["mutating"] is False
    assert preflight["token_accessed"] is False
    assert preflight["claims_obtained"] is False
    assert preflight["resolved_enrollment"]["loader"] == "campaign_service.load_resolved"
    assert preflight["resolved_enrollment"]["config_digest"] == running.controller.config_digest
    assert set(tmp_path.rglob("*")) == before
    assert {path: path.read_bytes() for path in before if path.is_file()} == before_bytes

    code, status, _ = _invoke(capsys, monkeypatch, running, "status")
    assert code == 0
    assert status["semantic_liveness"] == "management_only"
    snapshot = status["campaign_snapshot"]
    assert snapshot["desired_state"] == snapshot["observed_state"] == "paused"
    assert snapshot["execution_capability_available"] is False


def test_real_service_controls_distinguish_requested_applied_and_refused(
        running, capsys, monkeypatch):
    code, paused, _ = _invoke(
        capsys, monkeypatch, running, "pause", "--request-id", "pause-1",
        "--expected-revision", "0")
    assert code == 0 and paused["disposition"] == "applied"
    assert paused["command_result"]["completed"] is True

    code, resumed, _ = _invoke(
        capsys, monkeypatch, running, "resume", "--request-id", "resume-1",
        "--expected-revision", "1")
    assert code == 0 and resumed["disposition"] == "requested"
    assert resumed["command_result"]["observed_state"] == "waiting_prerequisite"

    code, refused, _ = _invoke(
        capsys, monkeypatch, running, "drain", "--request-id", "stale-drain",
        "--expected-revision", "0")
    assert code == 3 and refused["disposition"] == "refused"
    assert refused["http_status"] == 400


def test_lost_ack_exact_retry_keeps_request_digest_and_revision(
        running, capsys, monkeypatch):
    controller = running.controller
    row = {
        "schema": campaign_command_v2.COMMAND_SCHEMA,
        "campaign_id": controller.resolved.campaign_id,
        "config_generation": controller.config_generation,
        "config_digest": controller.config_digest,
        "supervisor_incarnation": controller.supervisor_incarnation,
        "request_id": "lost-ack",
        "operation": "pause",
        "payload": {},
        "expected_control_revision": 0,
    }
    row["payload_digest"] = campaign_command_v2.command_digest(
        operation="pause", payload={}, campaign_id=row["campaign_id"],
        config_generation=row["config_generation"],
        config_digest=row["config_digest"],
        supervisor_incarnation=row["supervisor_incarnation"],
        request_id=row["request_id"],
        expected_control_revision=row["expected_control_revision"])
    raw = json.dumps(row).encode()
    parsed = urlsplit(running.endpoint)
    client = socket.create_connection((parsed.hostname, parsed.port), timeout=2)
    client.sendall(
        b"POST /commands HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        + f"Authorization: Bearer {running.token}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
    client.close()
    for _ in range(100):
        if controller.control_revision == 1:
            break
        import time
        time.sleep(0.01)
    assert controller.control_revision == 1

    code, result, _ = _invoke(
        capsys, monkeypatch, running, "pause", "--request-id", "lost-ack",
        "--expected-revision", "0", "--retries", "2")
    assert code == 0 and result["disposition"] == "applied"
    assert result["command"]["payload_digest"] == row["payload_digest"]
    assert result["command"]["expected_control_revision"] == 0


def test_shutdown_latch_keeps_status_and_exact_retry_but_refuses_new_http_resume(
        running, capsys, monkeypatch):
    result = running.controller.request_shutdown_drain()
    request_id = result["request_id"]
    accepted = running.controller._command_requests[request_id]
    code, retried = cli._request(
        running.endpoint, "/commands", token=running.token, body=accepted, timeout=1)
    assert code == 200 and retried == result

    code, body, _error = _invoke(
        capsys, monkeypatch, running, "resume", "--request-id", "late-resume",
        "--expected-revision", str(running.controller.control_revision))
    assert code == 3 and body["disposition"] == "refused"
    code, status, _error = _invoke(capsys, monkeypatch, running, "status")
    assert code == 0
    assert status["campaign_snapshot"]["desired_state"] == "drained"


def test_auth_and_identity_refusals_do_not_disclose_secret(
        running, capsys, monkeypatch):
    monkeypatch.setenv(cli.TOKEN_ENV, "wrong-fixture-secret")
    assert cli.main(["status", "--endpoint", running.endpoint,
                     *_pins(running.controller)]) == 2
    captured = capsys.readouterr()
    assert "wrong-fixture-secret" not in captured.err
    monkeypatch.setenv(cli.TOKEN_ENV, running.token)
    pins = _pins(running.controller)
    pins[1] = "another-campaign"
    assert cli.main(["status", "--endpoint", running.endpoint, *pins]) == 2
    assert "differs" in capsys.readouterr().err


@pytest.mark.parametrize("endpoint", [
    "http://0.0.0.0:80", "http://localhost:80", "https://127.0.0.1:80",
    "http://user:secret@127.0.0.1:80", "http://127.0.0.1:80/path",
])
def test_endpoint_refuses_public_names_credentials_tls_and_paths(endpoint):
    with pytest.raises(cli.CLIRefused):
        cli._endpoint(endpoint)


def test_token_file_requires_owner_only_regular_single_link(tmp_path):
    token = tmp_path / "token"
    token.write_text("fixture-secret\n", encoding="utf-8")
    token.chmod(0o644)
    args = type("Args", (), {"token_file": token, "token_env": cli.TOKEN_ENV})()
    with pytest.raises(cli.CLIRefused, match="0600"):
        cli._token_from(args)

    token.chmod(0o600)
    assert cli._token_from(args) == "fixture-secret"
    token.write_bytes(b"x" * (cli.MAX_TOKEN_BYTES + 1))
    args.token_file = token
    with pytest.raises(cli.CLIRefused, match="size bound"):
        cli._token_from(args)
    link = tmp_path / "link"
    link.symlink_to(token)
    args.token_file = link
    with pytest.raises(cli.CLIRefused, match="protected token"):
        cli._token_from(args)


def test_fifo_token_path_refuses_before_read_without_blocking(tmp_path):
    fifo = tmp_path / "token-fifo"
    os.mkfifo(fifo, 0o600)
    args = type("Args", (), {"token_file": fifo, "token_env": cli.TOKEN_ENV})()
    started = time.monotonic()
    with pytest.raises(cli.CLIRefused, match="regular"):
        cli._token_from(args)
    assert time.monotonic() - started < 0.2


def test_environment_token_bound_and_control_bytes_never_disclose_secret(
        monkeypatch, capsys):
    args = type("Args", (), {"token_file": None, "token_env": "ADVERSARIAL_TOKEN"})()
    monkeypatch.setenv("ADVERSARIAL_TOKEN", "x" * (cli.MAX_TOKEN_BYTES + 1))
    with pytest.raises(cli.CLIRefused, match="size bound"):
        cli._token_from(args)
    secret = "do-not-disclose\x01still-secret"
    monkeypatch.setenv("ADVERSARIAL_TOKEN", secret)
    with pytest.raises(cli.CLIRefused, match="unsafe"):
        cli._token_from(args)
    assert secret not in capsys.readouterr().err
    with pytest.raises(cli.CLIRefused, match="unsafe"):
        cli._request("http://127.0.0.1:1", "/snapshot", token=secret)
    captured = capsys.readouterr()
    assert secret not in captured.out and secret not in captured.err


def test_newer_or_malformed_snapshot_is_rejected_by_actual_validator(
        running, capsys, monkeypatch):
    original = running.controller.publish_snapshot

    def newer():
        row = original()
        row["schema"] = "epyc.autokernel.campaign_snapshot.v999"
        return row

    running.controller.publish_snapshot = newer
    code, body, error = _invoke(capsys, monkeypatch, running, "status")
    assert code == 2 and body is None
    assert "schema is unsupported" in error and "v999" not in error


def test_malformed_published_v3_is_refused_by_local_closed_validator(
        running, capsys, monkeypatch):
    original = running.controller.publish_snapshot

    def v3_shape_cannot_fall_through_v2():
        row = original()
        row["schema"] = "epyc.autokernel.campaign_snapshot.v3"
        row["producer_schema"] = "epyc.autokernel.campaign_snapshot.v3"
        row["unified_projection"] = {}
        return row

    running.controller.publish_snapshot = v3_shape_cannot_fall_through_v2
    code, body, error = _invoke(capsys, monkeypatch, running, "status")
    assert code == 2 and body is None
    assert "snapshot refused by the v3 typed validator" in error


def test_snapshot_dispatches_v3_only_to_an_installed_closed_validator(
        running, monkeypatch):
    v2 = running.controller.publish_snapshot()
    v3 = dict(v2, schema="epyc.autokernel.campaign_snapshot.v3",
              producer_schema="epyc.autokernel.campaign_snapshot.v3", unified={})
    _code, health = running.service._transport_health()
    calls = []

    def request(_endpoint, path, **_kwargs):
        return (200, health) if path == "/health" else (200, v3)

    def validate(value):
        calls.append(value)
        return v2

    monkeypatch.setattr(cli, "_request", request)
    monkeypatch.setattr(control, "validate_snapshot_v3", validate, raising=False)
    args = type("Args", (), {
        "endpoint": running.endpoint, "timeout": 1.0,
        "expect_campaign_id": running.controller.resolved.campaign_id,
        "expect_config_generation": running.controller.config_generation,
        "expect_config_digest": running.controller.config_digest,
        "expect_supervisor_incarnation": running.controller.supervisor_incarnation,
    })()
    _health, snapshot = cli._snapshot(args, running.token)
    assert calls == [v3]
    assert snapshot == v2


def test_actual_v3_controller_snapshot_roundtrips_over_real_loopback(
        tmp_path, capsys, monkeypatch):
    from . import scheduling
    from .test_unified_driver import runtime_driver
    from .test_unified_planner import scheduler

    _instance, _engine, enrolled, _target, _digest = runtime_driver()
    base_config, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base_config.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    controller = control.CampaignController(
        enrolled, tmp_path / "v3-service", snapshot_version=3,
        scheduler_engine=engine)
    controller.__enter__()
    running = RunningService(controller)
    try:
        code, body, error = _invoke(capsys, monkeypatch, running, "status")
        assert code == 0 and not error
        assert body["campaign_snapshot"]["schema"] == control.SNAPSHOT_SCHEMA_V3
        assert body["campaign_snapshot"]["unified"]["schema"] \
            == control.UNIFIED_PROJECTION_SCHEMA
    finally:
        running.close()
        controller.close()


def test_paused_state_survives_real_service_and_controller_restart(
        tmp_path, capsys, monkeypatch):
    store = tmp_path / "service"
    resolved = _resolved()
    first = control.CampaignController(resolved, store, snapshot_version=2)
    first.__enter__()
    initial = RunningService(first)
    try:
        code, body, _ = _invoke(capsys, monkeypatch, initial, "status")
        assert code == 0 and body["campaign_snapshot"]["desired_state"] == "paused"
    finally:
        initial.close()
        first.close()
    second = control.CampaignController(resolved, store, snapshot_version=2)
    second.__enter__()
    restarted = RunningService(second)
    try:
        code, body, _ = _invoke(capsys, monkeypatch, restarted, "status")
        assert code == 0
        assert body["campaign_snapshot"]["desired_state"] == "paused"
        assert second.supervisor_incarnation == first.supervisor_incarnation + 1
    finally:
        restarted.close()
        second.close()


def test_exact_accepted_control_retry_survives_newer_supervisor(tmp_path, capsys,
                                                               monkeypatch):
    resolved = _resolved()
    store = tmp_path / "accepted-retry"
    first = control.CampaignController(resolved, store, snapshot_version=2)
    first.__enter__()
    service = RunningService(first)
    monkeypatch.setenv(cli.TOKEN_ENV, service.token)
    original = ["pause", *_pins(first), "--request-id", "retained-command",
                "--expected-revision", "0"]
    try:
        assert cli.main([*original, "--endpoint", service.endpoint]) == 0
        accepted = json.loads(capsys.readouterr().out)
    finally:
        service.close()
        first.close()
    second = control.CampaignController(resolved, store, snapshot_version=2)
    second.__enter__()
    service = RunningService(second)
    try:
        assert cli.main([*original, "--endpoint", service.endpoint]) == 0
        retried = json.loads(capsys.readouterr().out)
        assert retried["command_result"] == accepted["command_result"]
        assert retried["command"] == accepted["command"]
        assert second.control_revision == 1
    finally:
        service.close()
        second.close()


def test_old_incarnation_control_without_durable_command_is_server_refused(
        tmp_path, capsys, monkeypatch):
    resolved = _resolved()
    store = tmp_path / "absent-old-command"
    first = control.CampaignController(resolved, store, snapshot_version=2)
    first.__enter__()
    old_pins = _pins(first)
    first.close()
    second = control.CampaignController(resolved, store, snapshot_version=2)
    second.__enter__()
    service = RunningService(second)
    try:
        code, body, error = _invoke(
            capsys, monkeypatch, service, "pause", *old_pins,
            "--request-id", "never-accepted", "--expected-revision", "0")
        assert code == 3 and not error
        assert body["disposition"] == "refused" and body["http_status"] == 400
        assert second.control_revision == 0
    finally:
        service.close()
        second.close()


def test_timeout_retry_and_response_bounds_are_finite(monkeypatch):
    args = type("Args", (), {"token_file": None, "token_env": "MISSING_TOKEN"})()
    monkeypatch.delenv("MISSING_TOKEN", raising=False)
    with pytest.raises(cli.CLIRefused, match="required"):
        cli._token_from(args)
    with pytest.raises(cli.CLIRefused, match="timeout"):
        cli._request("http://127.0.0.1:1", "/health", timeout=31)
    with pytest.raises(cli.CLIRefused, match="retries"):
        cli._request("http://127.0.0.1:1", "/health", retries=3)


def test_response_body_bound_is_enforced_by_the_client():
    class Oversize(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def do_GET(self):
            raw = b"{" + b" " * cli.MAX_RESPONSE_BYTES + b"}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Oversize)
    import threading
    thread = threading.Thread(target=server.handle_request, name="fixture-oversize-response")
    thread.start()
    try:
        with pytest.raises(cli.CLIRefused, match="exceeds the bound"):
            cli._request(f"http://127.0.0.1:{server.server_port}", "/health")
    finally:
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_absolute_attempt_deadline_stops_a_trickling_response_and_joins_fixture():
    class Trickle(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(b"{")
                self.wfile.flush()
                for _ in range(100):
                    time.sleep(0.02)
                    self.wfile.write(b" ")
                    self.wfile.flush()
            except OSError:
                pass

    server = HTTPServer(("127.0.0.1", 0), Trickle)
    import threading
    thread = threading.Thread(target=server.handle_request, name="fixture-trickle-response")
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(cli.CLIRefused, match="absolute request deadline"):
            cli._request(f"http://127.0.0.1:{server.server_port}", "/health", timeout=0.12)
    finally:
        server.server_close()
        thread.join(timeout=1)
    assert time.monotonic() - started < 0.6
    assert not thread.is_alive()


def test_late_successful_decode_cannot_cross_absolute_deadline(monkeypatch):
    class Response:
        status = 200

        @staticmethod
        def getheader(_name):
            return "2"

        @staticmethod
        def read(_size):
            time.sleep(0.06)
            return b"{}"

    class Connection:
        sock = None

        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(cli.http.client, "HTTPConnection", Connection)
    with pytest.raises(cli.CLIRefused, match="absolute request deadline"):
        cli._request("http://127.0.0.1:1", "/health", timeout=0.02)


def test_mocked_remote_refusal_echo_is_omitted_through_main(
        running, capsys, monkeypatch):
    original = cli._request
    secret = running.token

    def echoing_request(endpoint, path, **kwargs):
        if path == "/commands":
            return 400, {"error": f"refused bearer {secret}"}
        return original(endpoint, path, **kwargs)

    monkeypatch.setattr(cli, "_request", echoing_request)
    code, body, error = _invoke(
        capsys, monkeypatch, running, "pause", "--request-id", "echo-refusal",
        "--expected-revision", "0")
    assert code == 3 and body["remote_refusal"].endswith("body omitted")
    rendered = json.dumps(body) + error
    assert secret not in rendered and "refused bearer" not in rendered


def test_same_port_new_supervisor_incarnation_refuses_old_identity(
        tmp_path, capsys, monkeypatch):
    store = tmp_path / "service"
    resolved = _resolved()
    first = control.CampaignController(resolved, store, snapshot_version=2)
    first.__enter__()
    original = RunningService(first)
    port = original.service.address[1]
    old_pins = _pins(first)
    original.close()
    first.close()

    second = control.CampaignController(resolved, store, snapshot_version=2)
    second.__enter__()
    replacement_service = campaign_service.CampaignHTTPService(
        second, "127.0.0.1", port, original.token)
    replacement_service.start()
    endpoint = f"http://127.0.0.1:{port}"
    monkeypatch.setenv(cli.TOKEN_ENV, original.token)
    try:
        assert cli.main(["status", "--endpoint", endpoint, *old_pins]) == 2
        error = capsys.readouterr().err
        assert "supervisor_incarnation differs" in error
        assert cli.main(["status", "--endpoint", endpoint, *_pins(second)]) == 0
        assert json.loads(capsys.readouterr().out)["disposition"] == "observed"
    finally:
        replacement_service.close()
        second.close()


def test_cli_source_never_accepts_token_value_on_argv():
    parser_actions = cli._parser()._actions
    assert "--token" not in {option for action in parser_actions
                             for option in action.option_strings}
    assert "socket" not in cli.__all__


def test_packaging_is_parseable_pinned_and_deliberately_not_installable():
    root = Path(__file__).parents[4] / "deploy" / "autokernel"
    unit = (root / "autokernel-campaign.service.in").read_text(encoding="utf-8")
    environment = (root / "autokernel-campaign.env.example").read_text(encoding="utf-8")
    assert "[Unit]" in unit and "[Service]" in unit and "[Install]" not in unit
    assert "ConditionPathExists=@STATE_DIRECTORY@/launch-bridge-approved" in unit
    assert "ExecStart=@PYTHON@ -m scripts.kernel_rnd.autokernel.loop.campaign_service" in unit
    assert "--snapshot-version @SNAPSHOT_VERSION@" in unit
    assert "EnvironmentFile=@SECRET_ENVIRONMENT_FILE@" in unit
    assert "SendSIGKILL=no" in unit and "KillMode=process" in unit
    assert "AUTOKERNEL_CONTROL_TOKEN=" in environment
    assert "AUTOKERNEL_CONTROL_TOKEN=" not in unit
    assert "$" not in unit and ";" not in unit and "`" not in unit
    parsed = configparser.RawConfigParser(interpolation=None, strict=True)
    parsed.optionxform = str
    parsed.read_string(unit)
    assert parsed.sections() == ["Unit", "Service"]
    assert parsed["Service"]["WorkingDirectory"] == "@PINNED_CHECKOUT@"
    assert parsed["Service"]["ExecStart"].split()[0] == "@PYTHON@"
