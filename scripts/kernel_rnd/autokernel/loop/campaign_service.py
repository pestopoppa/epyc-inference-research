#!/usr/bin/env python3
"""Offline authenticated HTTP/CLI surface for campaign controls."""
from __future__ import annotations

import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping

from .campaign import ResolvedCampaign
from .campaign_control import (
    CampaignController,
    ControlRefused,
    _stable_code_projection,
    resolved_config_digest,
)

MAX_BODY = 16 * 1024
MAX_HEADERS = 8 * 1024
REQUEST_TIMEOUT_S = 0.25
TOTAL_REQUEST_DEADLINE_S = 2.0
STOP_TIMEOUT_S = 2.0
DEFAULT_REFRESH_INTERVAL_S = 30.0
HEALTH_SCHEMA = "epyc.autokernel.campaign_transport_health.v1"
TRUSTED_ORIGIN_ENV = "AUTOKERNEL_TRUSTED_HUB_ORIGIN"


def _loaded_service_build_identity() -> dict[str, Any]:
    """Identify the loaded transport implementation, not an actor-supplied label."""
    digest = hashlib.sha256()
    included: list[str] = []
    callables = [
        ("make_handler", make_handler),
        ("_OwnedHTTPServer.get_request", _OwnedHTTPServer.get_request),
        ("_OwnedHTTPServer._deadline_watchdog", _OwnedHTTPServer._deadline_watchdog),
        ("CampaignHTTPService.__init__", CampaignHTTPService.__init__),
        ("CampaignHTTPService._transport_health", CampaignHTTPService._transport_health),
        ("CampaignHTTPService.start", CampaignHTTPService.start),
        ("CampaignHTTPService.close", CampaignHTTPService.close),
    ]
    for label, function in callables:
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(json.dumps(
            _stable_code_projection(function.__code__), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode())
        included.append(label)
    for name in ("HEALTH_SCHEMA", "MAX_BODY", "MAX_HEADERS", "REQUEST_TIMEOUT_S",
                 "TOTAL_REQUEST_DEADLINE_S"):
        digest.update(f"{name}={globals()[name]!r}".encode())
        included.append(f"constant:{name}")
    return {
        "schema": "epyc.autokernel.loaded_producer_build.v1",
        "scope": "campaign_service_loaded_transport_bytecode_and_constants",
        "module": __name__,
        "identity_basis": "loaded_callable_bytecode_and_constants_sha256",
        "included_symbols": included,
        "excluded_scope": ["campaign_controller", "journal", "network_peer_identity"],
        "sha256": digest.hexdigest(),
    }


class _OwnedHTTPServer(HTTPServer):
    """HTTPServer that records only sockets accepted by this listener."""

    def __init__(self, *args, total_request_deadline: float, **kwargs):
        self._connections: set[socket.socket] = set()
        self._deadlines: dict[socket.socket, float] = {}
        self._connections_lock = threading.Lock()
        self._closing = False
        self._watchdog_stop = threading.Event()
        self._total_request_deadline = total_request_deadline
        super().__init__(*args, **kwargs)
        self.watchdog_thread = threading.Thread(
            target=self._deadline_watchdog, name="campaign-request-deadline", daemon=True)
        self.watchdog_thread.start()

    def get_request(self):
        connection, address = super().get_request()
        with self._connections_lock:
            if self._closing:
                connection.close()
                raise OSError("HTTP listener is closing")
            self._connections.add(connection)
            self._deadlines[connection] = time.monotonic() + self._total_request_deadline
        return connection, address

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            with self._connections_lock:
                self._connections.discard(request)
                self._deadlines.pop(request, None)

    def _deadline_watchdog(self) -> None:
        interval = min(0.05, self._total_request_deadline / 4)
        while not self._watchdog_stop.wait(interval):
            now = time.monotonic()
            with self._connections_lock:
                expired = tuple(connection for connection, deadline in self._deadlines.items()
                                if deadline <= now)
            for connection in expired:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    connection.close()
                except OSError:
                    pass

    def close_owned_connections(self) -> None:
        with self._connections_lock:
            self._closing = True
            self._watchdog_stop.set()
            owned = tuple(self._connections)
        for connection in owned:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def handle_error(self, _request, _client_address):
        # Errors are returned by the handler when the socket is still writable;
        # never dump request headers (and therefore bearer tokens) to stderr.
        return

    def join_watchdog(self, timeout: float) -> None:
        self._watchdog_stop.set()
        self.watchdog_thread.join(timeout=timeout)


def load_resolved(path: Path) -> ResolvedCampaign:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlRefused(f"cannot load resolved campaign: {exc}") from exc
    if isinstance(row, Mapping) and "resolved_campaign" in row:
        row = row["resolved_campaign"]
    try:
        return ResolvedCampaign.from_dict(row)
    except (TypeError, ValueError) as exc:
        raise ControlRefused(f"invalid resolved campaign: {exc}") from exc


def _authorized(header: str | None, token: str) -> bool:
    prefix = "Bearer "
    supplied = header[len(prefix):] if isinstance(header, str) and header.startswith(prefix) else ""
    return hmac.compare_digest(supplied.encode(), token.encode())


def validate_origin(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ControlRefused("trusted hub origin must be non-empty or absent")
    from urllib.parse import urlsplit
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ControlRefused("trusted hub origin must be one exact http(s) origin")
    return f"{parsed.scheme}://{parsed.netloc}"


def make_handler(controller: CampaignController, token: str,
                 request_timeout: float = REQUEST_TIMEOUT_S,
                 transport_health=lambda: (200, {"ok": True}),
                 allowed_origin: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AutoKernelCampaign/1"

        def setup(self):
            super().setup()
            self.connection.settimeout(request_timeout)

        def log_message(self, _format, *_args):
            return

        def _json(self, code: int, body: Mapping[str, Any]) -> None:
            raw = json.dumps(body, sort_keys=True).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Vary", "Origin")
            origin = self.headers.get("Origin")
            if allowed_origin is not None and origin == allowed_origin:
                self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.end_headers()
            self.wfile.write(raw)

        def _origin_allowed(self) -> bool:
            origins = self.headers.get_all("Origin", [])
            if not origins:
                return True
            if len(origins) != 1 or allowed_origin is None or origins[0] != allowed_origin:
                self._json(403, {"error": "request origin is not allowlisted"})
                return False
            return True

        def _require_auth(self) -> bool:
            values = self.headers.get_all("Authorization", [])
            if len(values) != 1 or not _authorized(values[0], token):
                self._json(401, {"error": "authentication required"})
                return False
            return True

        def _headers_bounded(self) -> bool:
            size = sum(len(key) + len(value) + 4 for key, value in self.headers.items())
            if size > MAX_HEADERS:
                self._json(431, {"error": "request headers too large"})
                return False
            return True

        def do_GET(self):
            try:
                if not self._headers_bounded():
                    return
                if not self._origin_allowed():
                    return
                if self.path == "/health":
                    code, body = transport_health()
                    self._json(code, body)
                elif self.path == "/snapshot" and self._require_auth():
                    self._json(200, controller.publish_snapshot())
                elif self.path != "/snapshot":
                    self._json(404, {"error": "not found"})
            except Exception as exc:
                self._json(503, {"error": type(exc).__name__})

        def do_POST(self):
            if not self._headers_bounded():
                return
            if not self._origin_allowed():
                return
            if self.path != "/commands":
                self._json(404, {"error": "not found"})
                return
            if not self._require_auth():
                return
            try:
                if self.headers.get_all("Transfer-Encoding", []):
                    self._json(400, {"error": "Transfer-Encoding is unsupported"})
                    return
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isdigit():
                    self._json(400, {"error": "one decimal Content-Length is required"})
                    return
                length = int(lengths[0])
                if length > MAX_BODY:
                    self._json(413, {"error": "invalid request size"})
                    return
                content_types = self.headers.get_all("Content-Type", [])
                if (len(content_types) != 1
                        or content_types[0].split(";", 1)[0].strip().lower()
                        != "application/json"):
                    self._json(415, {"error": "application/json is required"})
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    self._json(400, {"error": "incomplete request body"})
                    return
                body = json.loads(raw)
                self._json(200, controller.apply_command(body))
            except socket.timeout:
                self._json(408, {"error": "request body timed out"})
            except (ValueError, ControlRefused) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(503, {"error": type(exc).__name__})

        def do_OPTIONS(self):
            if not self._headers_bounded() or not self._origin_allowed():
                return
            if allowed_origin is None or self.headers.get("Origin") != allowed_origin:
                self._json(403, {"error": "CORS origin is not configured"})
                return
            if self.path not in {"/snapshot", "/commands", "/health"}:
                self._json(404, {"error": "not found"})
                return
            self.send_response(204)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Max-Age", "300")
            self.end_headers()

    return Handler


def parse_listen(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not port_text.isdigit():
        raise ControlRefused("--listen must be numeric HOST:PORT")
    port = int(port_text)
    if not 0 <= port <= 65535:
        raise ControlRefused("listener port is out of range")
    try:
        address = ipaddress.ip_address(host)
        if address.version != 4 or not address.is_loopback:
            raise ControlRefused("v1 listener must bind to numeric IPv4 loopback")
    except ValueError as exc:
        raise ControlRefused("listener host must be a numeric loopback address") from exc
    return host, port


class CampaignHTTPService:
    def __init__(self, controller: CampaignController, host: str, port: int, token: str,
                 *, request_timeout: float = REQUEST_TIMEOUT_S,
                 total_request_deadline: float = TOTAL_REQUEST_DEADLINE_S,
                 refresh_interval: float = DEFAULT_REFRESH_INTERVAL_S,
                 allowed_origin: str | None = None):
        if not isinstance(host, str):
            raise ControlRefused("listener host must be a string")
        if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
            raise ControlRefused("listener port must be an integer from 0 through 65535")
        try:
            address = ipaddress.ip_address(host)
            if address.version != 4 or not address.is_loopback:
                raise ControlRefused("v1 listener must bind to numeric IPv4 loopback")
        except ValueError as exc:
            raise ControlRefused("listener host must be a numeric loopback address") from exc
        if not isinstance(token, str) or not token:
            raise ControlRefused("AUTOKERNEL_CONTROL_TOKEN is required")
        if (not isinstance(request_timeout, (int, float)) or isinstance(request_timeout, bool)
                or not math.isfinite(float(request_timeout)) or request_timeout <= 0):
            raise ControlRefused("request timeout must be positive")
        if (not isinstance(refresh_interval, (int, float))
                or isinstance(refresh_interval, bool)
                or not math.isfinite(float(refresh_interval)) or refresh_interval <= 0):
            raise ControlRefused("producer refresh interval must be positive and finite")
        if (not isinstance(total_request_deadline, (int, float))
                or isinstance(total_request_deadline, bool)
                or not math.isfinite(float(total_request_deadline))
                or total_request_deadline <= 0):
            raise ControlRefused("total request deadline must be positive and finite")
        self.allowed_origin = validate_origin(allowed_origin)
        self.controller = controller
        self._lock = threading.Lock()
        self._closed = False
        self._started = False
        self._stop = threading.Event()
        self._publisher_error: str | None = None
        self.refresh_interval = float(refresh_interval)
        self._health_identity = {
            "schema": HEALTH_SCHEMA, "transport": "campaign-control-http",
            "service_build": _loaded_service_build_identity(),
            "campaign_id": controller.resolved.campaign_id,
            "config_generation": controller.config_generation,
            "config_digest": controller.config_digest,
            "supervisor_incarnation": controller.supervisor_incarnation,
            "stream_epoch": controller.stream_epoch,
            "allowed_origin": self.allowed_origin,
        }
        self.server = _OwnedHTTPServer(
            (host, port), make_handler(
                controller, token, float(request_timeout), self._transport_health,
                self.allowed_origin),
            total_request_deadline=float(total_request_deadline))
        self.thread: threading.Thread | None = None
        self.publisher_thread: threading.Thread | None = None

    def _transport_health(self):
        with self._lock:
            error = self._publisher_error
        if error is not None:
            return 503, self._health_identity | {
                "ok": False, "producer": "failed", "error": error}
        return 200, self._health_identity | {
            "ok": True, "producer": "running" if self._started else "starting",
            "error": None}

    def _publish_loop(self) -> None:
        while not self._stop.wait(self.refresh_interval):
            try:
                self.controller.publish_snapshot()
            except Exception as exc:
                with self._lock:
                    self._publisher_error = f"{type(exc).__name__}: {exc}"
                self._stop.set()
                return

    @property
    def address(self):
        return self.server.server_address

    def start(self) -> None:
        with self._lock:
            if self._closed or self._started:
                raise ControlRefused("HTTP service is already started or closed")
            self._started = True
        try:
            self.controller.publish_snapshot()
        except Exception:
            with self._lock:
                self._started = False
            raise
        with self._lock:
            self.thread = threading.Thread(
                target=lambda: self.server.serve_forever(poll_interval=0.05),
                name="campaign-control-http", daemon=True)
            self.publisher_thread = threading.Thread(
                target=self._publish_loop, name="campaign-snapshot-publisher", daemon=True)
            self.thread.start()
            self.publisher_thread.start()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            started, thread, publisher = self._started, self.thread, self.publisher_thread
        deadline = time.monotonic() + STOP_TIMEOUT_S
        self._stop.set()
        self.server.close_owned_connections()
        shutdown_thread = None
        if started:
            shutdown_thread = threading.Thread(target=self.server.shutdown,
                                               name="campaign-http-shutdown", daemon=True)
            shutdown_thread.start()
        for owned in (publisher, shutdown_thread, thread):
            if owned is not None:
                owned.join(timeout=max(0.0, deadline - time.monotonic()))
        self.server.join_watchdog(max(0.0, deadline - time.monotonic()))
        alive = [owned.name for owned in (publisher, shutdown_thread, thread)
                 if owned is not None and owned.is_alive()]
        if self.server.watchdog_thread.is_alive():
            alive.append(self.server.watchdog_thread.name)
        if alive:
            raise ControlRefused(
                f"service stop deadline expired; ownership retained by {alive}")
        self.server.server_close()
        with self._lock:
            self._closed = True
            self._started = False

    @property
    def publisher_error(self) -> str | None:
        with self._lock:
            return self._publisher_error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-campaign", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--config-generation", type=int, default=1)
    parser.add_argument("--snapshot-version", type=int, choices=(1, 2), default=1,
                        help="explicit projection contract; v2 does not create grant authority")
    parser.add_argument("--refresh-interval", type=float,
                        default=DEFAULT_REFRESH_INTERVAL_S)
    parser.add_argument("--request-deadline", type=float,
                        default=TOTAL_REQUEST_DEADLINE_S)
    parser.add_argument("--trusted-origin")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--once", action="store_true")
    modes.add_argument("--listen", metavar="HOST:PORT")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolved = load_resolved(args.resolved_campaign)
    if args.config_generation < 1:
        raise ControlRefused("config generation must be positive")
    if not math.isfinite(args.refresh_interval) or args.refresh_interval <= 0:
        raise ControlRefused("producer refresh interval must be positive and finite")
    if not math.isfinite(args.request_deadline) or args.request_deadline <= 0:
        raise ControlRefused("total request deadline must be positive and finite")
    listen = parse_listen(args.listen) if args.listen else None
    token = os.environ.get("AUTOKERNEL_CONTROL_TOKEN", "") if listen else None
    if listen and not token:
        raise ControlRefused("AUTOKERNEL_CONTROL_TOKEN is required")
    if not args.once and not args.listen:
        print(json.dumps({"campaign_id": resolved.campaign_id,
                          "config_generation": args.config_generation,
                          "config_digest": resolved_config_digest(resolved),
                          "requested_manifest_digest": resolved.manifest_digest,
                          "execution_authorized": False}, sort_keys=True))
        return 0
    controller = CampaignController(
        resolved, args.store, config_generation=args.config_generation,
        snapshot_version=args.snapshot_version)
    controller.__enter__()
    if args.once:
        try:
            print(json.dumps(controller.publish_snapshot(), sort_keys=True))
            return 0
        finally:
            controller.close()
    assert listen is not None and token is not None
    try:
        service = CampaignHTTPService(controller, *listen, token,
                                      refresh_interval=args.refresh_interval,
                                      total_request_deadline=args.request_deadline,
                                      allowed_origin=(args.trusted_origin
                                                      or os.environ.get(TRUSTED_ORIGIN_ENV)))
    except BaseException:
        controller.close()
        raise
    try:
        try:
            service.start()
            while service.thread is not None and service.thread.is_alive():
                service.thread.join(timeout=0.25)
                if service.publisher_error is not None:
                    raise ControlRefused(
                        f"snapshot publisher failed: {service.publisher_error}")
        finally:
            service.close()
    except BaseException:
        # If bounded service close refuses, its threads may still own controller
        # calls. Retain the lease until process teardown instead of claiming a
        # clean handoff to another supervisor incarnation.
        raise
    controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CampaignHTTPService", "DEFAULT_REFRESH_INTERVAL_S", "HEALTH_SCHEMA",
           "MAX_BODY", "MAX_HEADERS", "TOTAL_REQUEST_DEADLINE_S",
           "TRUSTED_ORIGIN_ENV", "load_resolved", "main", "make_handler",
           "parse_listen", "validate_origin"]
