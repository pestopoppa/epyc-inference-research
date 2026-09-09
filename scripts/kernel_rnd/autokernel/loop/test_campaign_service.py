"""Loopback HTTP and offline CLI tests for the campaign service."""
from __future__ import annotations

import json
import inspect
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from . import campaign, campaign_control as control, campaign_service as service
from .test_campaign import _manifest, _registry, _target


def _resolved():
    raw = _manifest(production=[_target("prod")])
    return campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                     registry_snapshot=_registry())


def _resolved_file(tmp_path):
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(_resolved().to_dict()), encoding="utf-8")
    return path


def _request(base, path, *, token=None, body=None, declared_length=None, origin=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    if origin is not None:
        headers["Origin"] = origin
    if declared_length is not None:
        headers["Content-Length"] = str(declared_length)
    request = Request(base + path, data=data, headers=headers,
                      method="POST" if body is not None else "GET")
    try:
        response = urlopen(request, timeout=3)
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())
    return response.status, json.loads(response.read())


def _raw_status(base, raw):
    parsed = urlsplit(base)
    connection = socket.create_connection((parsed.hostname, parsed.port), timeout=2)
    try:
        connection.sendall(raw)
        return connection.recv(256).split(b"\r\n", 1)[0]
    finally:
        connection.close()


@pytest.fixture
def running_service(tmp_path):
    controller = control.CampaignController(_resolved(), tmp_path / "service")
    controller.__enter__()
    server = service.CampaignHTTPService(
        controller, "127.0.0.1", 0, "secret-token",
        allowed_origin="http://hub.test")
    server.start()
    host, port = server.address
    try:
        yield controller, f"http://{host}:{port}"
    finally:
        server.close()
        controller.close()


def test_health_is_public_but_snapshot_requires_valid_bearer(running_service):
    controller, base = running_service
    initial_sequence = controller.sequence
    assert initial_sequence == 1
    code, first = _request(base, "/health")
    assert code == 200 and first["ok"] is True
    assert first["schema"] == service.HEALTH_SCHEMA
    assert first["campaign_id"] == controller.resolved.campaign_id
    assert first["config_digest"] == controller.config_digest
    assert first["supervisor_incarnation"] == controller.supervisor_incarnation
    assert first["allowed_origin"] == "http://hub.test"
    assert first["service_build"]["module"] == service.__name__
    assert first["service_build"]["scope"] == (
        "campaign_service_loaded_transport_bytecode_and_constants")
    assert len(first["service_build"]["sha256"]) == 64
    assert _request(base, "/health")[1]["service_build"] == first["service_build"]
    assert controller.sequence == initial_sequence
    assert (controller.store / control.SNAPSHOT_FILE).exists()
    assert _request(base, "/snapshot")[0] == 401
    assert _request(base, "/snapshot", token="wrong")[0] == 401
    code, snapshot = _request(base, "/snapshot", token="secret-token")
    assert code == 200
    assert snapshot["sequence"] == initial_sequence + 1
    assert snapshot["execution_authorized"] is False
    assert snapshot["last_scientific_result_at"] is None
    assert set(snapshot) == {
        "schema", "producer_build", "producer_schema", "campaign_id",
        "config_generation", "config_digest", "requested_manifest_digest",
        "supervisor_incarnation", "stream_epoch", "sequence", "journal_cursor",
        "control_revision", "generated_at", "desired_state", "observed_state",
        "command_results", "active_worker", "producer_heartbeat_at",
        "last_scientific_result_at", "worker_activity_at", "execution_authorized",
        "prerequisite_reason",
    }


def test_loaded_service_identity_is_stable_after_warm_execution_and_reconstruction(tmp_path):
    before = service._loaded_service_build_identity()
    with control.CampaignController(_resolved(), tmp_path / "one") as first:
        one = service.CampaignHTTPService(first, "127.0.0.1", 0, "token")
        one.start()
        assert _request(f"http://{one.address[0]}:{one.address[1]}", "/health")[0] == 200
        one.close()
    for _ in range(100):
        service.validate_origin("http://hub.test")
    with control.CampaignController(_resolved(), tmp_path / "two") as second:
        two = service.CampaignHTTPService(second, "127.0.0.1", 0, "token")
        try:
            assert two._health_identity["service_build"] == before
        finally:
            two.close()
    assert service._loaded_service_build_identity() == before


def test_authenticated_command_and_retry_use_same_contract(running_service):
    controller, base = running_service
    row = {"schema": control.COMMAND_SCHEMA,
           "campaign_id": controller.resolved.campaign_id,
           "config_generation": 1, "request_id": "pause-http",
           "operation": "pause", "payload": {}, "expected_control_revision": 0}
    row["payload_digest"] = control.command_digest(
        operation="pause", payload={}, campaign_id=row["campaign_id"],
        config_generation=1)
    assert _request(base, "/commands", body=row)[0] == 401
    code, result = _request(base, "/commands", token="secret-token", body=row)
    assert code == 200 and result["accepted"] is True
    row["expected_control_revision"] = 999
    assert _request(base, "/commands", token="secret-token", body=row)[1] == result


def test_disconnect_after_command_body_preserves_durable_retry_identity(running_service):
    controller, base = running_service
    row = {"schema": control.COMMAND_SCHEMA,
           "campaign_id": controller.resolved.campaign_id,
           "config_generation": 1, "request_id": "lost-ack",
           "operation": "pause", "payload": {}, "expected_control_revision": 0}
    row["payload_digest"] = control.command_digest(
        operation="pause", payload={}, campaign_id=row["campaign_id"],
        config_generation=1)
    raw = json.dumps(row).encode()
    parsed = urlsplit(base)
    client = socket.create_connection((parsed.hostname, parsed.port), timeout=2)
    client.sendall(
        b"POST /commands HTTP/1.1\r\nHost: localhost\r\n"
        b"Authorization: Bearer secret-token\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
    client.close()
    deadline = time.monotonic() + 1
    while controller.control_revision == 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    row["expected_control_revision"] = 999
    code, retried = _request(base, "/commands", token="secret-token", body=row)
    assert code == 200 and retried["request_id"] == "lost-ack"
    assert retried["control_revision"] == 1


def test_oversize_and_unknown_command_fields_refuse_without_token_disclosure(
        running_service):
    _controller, base = running_service
    code, body = _request(base, "/commands", token="secret-token", body={},
                          declared_length=service.MAX_BODY + 1)
    assert code == 413 and "secret-token" not in json.dumps(body)
    code, body = _request(base, "/commands", token="secret-token",
                          body={"unexpected": True})
    assert code == 400 and "unknown" in body["error"]
    assert "secret-token" not in body["error"]


def test_commands_require_json_and_malformed_operation_is_typed_refusal(running_service):
    controller, base = running_service
    raw = b"{}"
    request = Request(base + "/commands", data=raw,
                      headers={"Authorization": "Bearer secret-token",
                               "Content-Type": "text/plain"}, method="POST")
    with pytest.raises(HTTPError) as caught:
        urlopen(request, timeout=3)
    assert caught.value.code == 415
    row = {"schema": control.COMMAND_SCHEMA,
           "campaign_id": controller.resolved.campaign_id,
           "config_generation": 1, "request_id": "bad-operation",
           "operation": [], "payload": {}, "payload_digest": "0" * 64,
           "expected_control_revision": 0}
    code, body = _request(base, "/commands", token="secret-token", body=row)
    assert code == 400 and "operation" in body["error"]


def test_transfer_encoding_duplicate_lengths_and_large_headers_refuse(running_service):
    _controller, base = running_service
    common = (b"POST /commands HTTP/1.1\r\nHost: localhost\r\n"
              b"Authorization: Bearer secret-token\r\n"
              b"Content-Type: application/json\r\n")
    assert b" 400 " in _raw_status(
        base, common + b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
    assert b" 400 " in _raw_status(
        base, common + b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}")
    assert b" 431 " in _raw_status(
        base, b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Large: "
        + b"x" * (service.MAX_HEADERS + 1) + b"\r\n\r\n")


def test_cors_is_exact_origin_finite_and_never_wildcard(running_service):
    _controller, base = running_service
    request = Request(base + "/health", headers={"Origin": "http://hub.test"})
    response = urlopen(request, timeout=3)
    assert response.headers["Access-Control-Allow-Origin"] == "http://hub.test"
    assert response.headers["Vary"] == "Origin"
    assert "*" not in response.headers.get("Access-Control-Allow-Origin", "")
    assert _request(base, "/health", origin="http://evil.test")[0] == 403

    preflight = Request(
        base + "/commands", method="OPTIONS",
        headers={"Origin": "http://hub.test",
                 "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "Authorization, Content-Type"})
    response = urlopen(preflight, timeout=3)
    assert response.status == 204
    assert response.headers["Access-Control-Allow-Methods"] == "GET, POST, OPTIONS"
    assert response.headers["Access-Control-Allow-Headers"] == "Authorization, Content-Type"
    assert response.headers.get("Access-Control-Allow-Credentials") is None


def test_missing_origin_allowance_leaves_browser_access_unavailable(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        server = service.CampaignHTTPService(controller, "127.0.0.1", 0, "token")
        server.start()
        try:
            base = f"http://{server.address[0]}:{server.address[1]}"
            assert _request(base, "/health", origin="http://hub.test")[0] == 403
        finally:
            server.close()


def test_listener_requires_token_and_numeric_loopback(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        with pytest.raises(control.ControlRefused, match="TOKEN"):
            service.CampaignHTTPService(controller, "127.0.0.1", 0, "")
        with pytest.raises(control.ControlRefused, match="loopback"):
            service.CampaignHTTPService(controller, "0.0.0.0", 0, "token")
        with pytest.raises(control.ControlRefused, match="numeric"):
            service.CampaignHTTPService(controller, "localhost", 0, "token")


def test_service_close_before_start_and_twice_never_hangs(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        server = service.CampaignHTTPService(controller, "127.0.0.1", 0, "token")
        server.close()
        server.close()
        with pytest.raises(control.ControlRefused, match="started or closed"):
            server.start()


def test_stalled_declared_body_times_out_and_service_stops_bounded(tmp_path):
    controller = control.CampaignController(_resolved(), tmp_path / "service")
    controller.__enter__()
    server = service.CampaignHTTPService(controller, "127.0.0.1", 0, "token",
                                         request_timeout=0.1)
    server.start()
    client = socket.create_connection(server.address, timeout=1)
    client.sendall(
        b"POST /commands HTTP/1.1\r\nHost: localhost\r\n"
        b"Authorization: Bearer token\r\nContent-Type: application/json\r\n"
        b"Content-Length: 100\r\n\r\n{")
    started = time.monotonic()
    try:
        server.close()
        server.close()
    finally:
        client.close()
        controller.close()
    assert time.monotonic() - started < service.STOP_TIMEOUT_S
    assert server.thread is not None and not server.thread.is_alive()


def test_slow_trickle_cannot_extend_service_close_deadline(tmp_path):
    controller = control.CampaignController(_resolved(), tmp_path / "service")
    controller.__enter__()
    server = service.CampaignHTTPService(
        controller, "127.0.0.1", 0, "token", request_timeout=0.08)
    server.start()
    client = socket.create_connection(server.address, timeout=1)
    client.sendall(
        b"POST /commands HTTP/1.1\r\nHost: localhost\r\n"
        b"Authorization: Bearer token\r\nContent-Type: application/json\r\n"
        b"Content-Length: 1000\r\n\r\n")
    sender_stopped = threading.Event()

    def trickle():
        try:
            while True:
                client.sendall(b" ")
                time.sleep(0.02)
        except OSError:
            pass
        finally:
            sender_stopped.set()

    sender = threading.Thread(target=trickle, name="test-slow-client")
    sender.start()
    time.sleep(0.12)
    started = time.monotonic()
    try:
        server.close()
    finally:
        client.close()
        sender.join(timeout=1)
        controller.close()
    assert time.monotonic() - started < service.STOP_TIMEOUT_S
    assert sender_stopped.is_set() and not sender.is_alive()
    assert server.thread is not None and not server.thread.is_alive()
    assert server.publisher_thread is not None and not server.publisher_thread.is_alive()


def test_total_request_deadline_returns_listener_after_continuous_trickle(tmp_path):
    controller = control.CampaignController(_resolved(), tmp_path / "service")
    controller.__enter__()
    server = service.CampaignHTTPService(
        controller, "127.0.0.1", 0, "token", request_timeout=0.08,
        total_request_deadline=0.18)
    server.start()
    client = socket.create_connection(server.address, timeout=1)
    client.sendall(
        b"POST /commands HTTP/1.1\r\nHost: localhost\r\n"
        b"Authorization: Bearer token\r\nContent-Type: application/json\r\n"
        b"Content-Length: 1000\r\n\r\n")
    stopped = threading.Event()

    def trickle():
        try:
            while True:
                client.sendall(b" ")
                time.sleep(0.02)
        except OSError:
            stopped.set()

    sender = threading.Thread(target=trickle, name="test-total-deadline-trickle")
    sender.start()
    try:
        time.sleep(0.3)
        base = f"http://{server.address[0]}:{server.address[1]}"
        assert _request(base, "/health")[0] == 200
        sender.join(timeout=1)
        assert stopped.is_set() and not sender.is_alive()
    finally:
        client.close()
        server.close()
        controller.close()


def test_periodic_publisher_advances_only_health_projection(tmp_path):
    ticks = iter(f"2026-09-09T00:00:{second:02d}Z" for second in range(20))
    with control.CampaignController(
            _resolved(), tmp_path / "service", clock=lambda: next(ticks)) as controller:
        server = service.CampaignHTTPService(
            controller, "127.0.0.1", 0, "token", refresh_interval=0.02)
        server.start()
        try:
            initial = json.loads(
                (controller.store / control.SNAPSHOT_FILE).read_text(encoding="utf-8"))
            deadline = time.monotonic() + 1
            latest = initial
            while latest["sequence"] < initial["sequence"] + 2:
                assert time.monotonic() < deadline
                time.sleep(0.02)
                latest = json.loads(
                    (controller.store / control.SNAPSHOT_FILE).read_text(encoding="utf-8"))
            assert latest["journal_cursor"] == initial["journal_cursor"]
            assert latest["control_revision"] == initial["control_revision"]
            assert latest["last_scientific_result_at"] is None
            assert latest["producer_heartbeat_at"] != initial["producer_heartbeat_at"]
        finally:
            server.close()


def test_periodic_publisher_fault_is_retained_and_health_is_unhealthy(
        tmp_path, monkeypatch):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        original = control.status.write_json
        calls = 0

        def fail_after_initial(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise OSError("injected publisher write fault")
            return original(*args, **kwargs)

        monkeypatch.setattr(control.status, "write_json", fail_after_initial)
        server = service.CampaignHTTPService(
            controller, "127.0.0.1", 0, "token", refresh_interval=0.02)
        server.start()
        try:
            deadline = time.monotonic() + 1
            while server.publisher_error is None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            host, port = server.address
            code, body = _request(f"http://{host}:{port}", "/health")
            assert code == 503 and body["producer"] == "failed"
            assert "injected publisher write fault" in body["error"]
            assert "token" not in json.dumps(body)
        finally:
            server.close()


def test_initial_publisher_fault_allows_owned_listener_cleanup(tmp_path, monkeypatch):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        monkeypatch.setattr(
            control.status, "write_json",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("initial fault")))
        server = service.CampaignHTTPService(controller, "127.0.0.1", 0, "token")
        with pytest.raises(OSError, match="initial fault"):
            server.start()
        server.close()
        assert server.server.fileno() == -1
        assert server.thread is None and server.publisher_thread is None


@pytest.mark.parametrize("kwargs, match", [
    ({"port": True}, "port"),
    ({"port": "1"}, "port"),
    ({"token": 5}, "TOKEN"),
    ({"request_timeout": float("nan")}, "timeout"),
    ({"request_timeout": float("inf")}, "timeout"),
    ({"refresh_interval": float("nan")}, "refresh"),
    ({"total_request_deadline": None}, "total request deadline"),
    ({"total_request_deadline": True}, "total request deadline"),
    ({"total_request_deadline": float("nan")}, "total request deadline"),
    ({"allowed_origin": "*"}, "origin"),
    ({"allowed_origin": "http://hub.test/path"}, "origin"),
    ({"host": "::1"}, "IPv4"),
])
def test_constructor_refuses_malformed_transport_values_before_binding(
        tmp_path, kwargs, match):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        values = {"host": "127.0.0.1", "port": 0, "token": "token"}
        values.update(kwargs)
        with pytest.raises(control.ControlRefused, match=match):
            service.CampaignHTTPService(controller, **values)


def test_cli_dry_inspection_does_not_create_store_or_offer_execution(tmp_path, capsys):
    source = _resolved_file(tmp_path)
    store = tmp_path / "absent-store"
    assert service.main(["--resolved-campaign", str(source), "--store", str(store)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["execution_authorized"] is False
    assert not store.exists()


def test_cli_once_materializes_one_snapshot_without_launching(tmp_path, capsys):
    source = _resolved_file(tmp_path)
    store = tmp_path / "service"
    assert service.main(["--resolved-campaign", str(source), "--store", str(store),
                         "--once"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["active_worker"] is None
    assert body["execution_authorized"] is False
    assert (store / control.SNAPSHOT_FILE).is_file()
    source = inspect.getsource(service)
    assert "subprocess" not in source
    assert "os.system" not in source


@pytest.mark.parametrize("extra", [
    ["--config-generation", "0"],
    ["--listen", "localhost:1234"],
    ["--listen", "127.0.0.1:not-a-port"],
    ["--refresh-interval", "nan", "--listen", "127.0.0.1:1234"],
])
def test_bad_generation_or_listener_refuses_before_store_mutation(tmp_path, extra):
    source = _resolved_file(tmp_path)
    store = tmp_path / "must-stay-absent"
    with pytest.raises(control.ControlRefused):
        service.main(["--resolved-campaign", str(source), "--store", str(store), *extra])
    assert not store.exists()
