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
import select
import signal
import socket
import sys
import threading
import time
from typing import Any, Callable, Mapping

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
        ("CampaignHTTPService.report_runtime_unresolved",
         CampaignHTTPService.report_runtime_unresolved),
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

    def report_runtime_unresolved(self) -> None:
        """Expose a fixed non-authoritative failure without accepting runtime work."""
        with self._lock:
            self._publisher_error = "runtime owner stopped unresolved"


RuntimeFactory = Callable[
    [ResolvedCampaign, argparse.Namespace],
    tuple[CampaignController, Any],
]


class _RuntimeExecution:
    """Own the sole non-HTTP runtime thread and retain its terminal state."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.result: Any = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="campaign-runtime-owner", daemon=False)

    def _run(self) -> None:
        from . import standalone_runtime as runtime_module

        try:
            result = self.runtime.run(self.stop_event)
            if not isinstance(result, runtime_module.RuntimeTickResult):
                raise ControlRefused("runtime worker returned an untyped result")
            with self._lock:
                self.result = result
        except BaseException as exc:
            with self._lock:
                self.error = exc

    def start(self) -> None:
        self.thread.start()

    def request_stop(self) -> None:
        self.runtime.request_stop()
        self.stop_event.set()

    def finished(self) -> bool:
        return not self.thread.is_alive()

    def join(self, deadline: float) -> bool:
        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not self.thread.is_alive()


def _close_runtime(runtime: Any, deadline: float):
    from . import standalone_runtime as runtime_module

    result = runtime.close(deadline=deadline)
    if not isinstance(result, runtime_module.RuntimeShutdownResult):
        raise ControlRefused("runtime close returned an untyped result")
    return result


def _runtime_owner(runtime_factory: RuntimeFactory, resolved: ResolvedCampaign,
                   args: argparse.Namespace
                   ) -> tuple[CampaignController, Any]:
    from . import standalone_runtime as runtime_module

    if not callable(runtime_factory):
        raise ControlRefused("runtime_factory must be callable or absent")
    value = runtime_factory(resolved, args)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ControlRefused("runtime_factory must return (controller, runtime)")
    controller, runtime = value
    if not isinstance(controller, CampaignController) \
            or not isinstance(runtime, runtime_module.StandaloneRuntime):
        raise ControlRefused(
            "runtime_factory returned an untyped owner; factory retains cleanup authority")
    if runtime.controller is not controller:
        raise ControlRefused(
            "runtime and service controllers differ; typed owners remain retained")
    try:
        if controller.snapshot_version != 3:
            raise ControlRefused("runtime service requires snapshot v3")
        if (controller.resolved.to_dict() != resolved.to_dict()
                or controller.config_generation != args.config_generation
                or controller.store != Path(args.store).absolute()):
            raise ControlRefused("runtime owner identity differs from service configuration")
        controller.snapshot()  # public active-owner check after factory-controlled replay
        return controller, runtime
    except BaseException as exc:
        runtime.request_stop()
        closed = _close_runtime(runtime, time.monotonic() + args.shutdown_deadline)
        if closed.status == "closed":
            controller.close()
            raise
        raise ControlRefused(
            f"runtime owner validation failed and cleanup remains unresolved: "
            f"{closed.reason}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-campaign", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--config-generation", type=int, default=1)
    parser.add_argument("--snapshot-version", type=int, choices=(1, 2, 3), default=1,
                        help="v3 execution requires an installed typed runtime_factory")
    parser.add_argument("--refresh-interval", type=float,
                        default=DEFAULT_REFRESH_INTERVAL_S)
    parser.add_argument("--request-deadline", type=float,
                        default=TOTAL_REQUEST_DEADLINE_S)
    parser.add_argument("--shutdown-deadline", type=float, default=30.0)
    parser.add_argument("--trusted-origin")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--once", action="store_true")
    modes.add_argument("--listen", metavar="HOST:PORT")
    return parser


def main(argv: list[str] | None = None, *, runtime_factory: RuntimeFactory | None = None) -> int:
    args = _parser().parse_args(argv)
    resolved = load_resolved(args.resolved_campaign)
    if args.config_generation < 1:
        raise ControlRefused("config generation must be positive")
    if not math.isfinite(args.refresh_interval) or args.refresh_interval <= 0:
        raise ControlRefused("producer refresh interval must be positive and finite")
    if not math.isfinite(args.request_deadline) or args.request_deadline <= 0:
        raise ControlRefused("total request deadline must be positive and finite")
    if not math.isfinite(args.shutdown_deadline) or args.shutdown_deadline <= 0:
        raise ControlRefused("shutdown deadline must be positive and finite")
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
    if runtime_factory is not None and (args.once or listen is None):
        raise ControlRefused("runtime_factory requires the listening service mode")
    if runtime_factory is None and args.snapshot_version == 3:
        raise ControlRefused("snapshot v3 service requires a connected runtime_factory")
    runtime = None
    if runtime_factory is None:
        controller = CampaignController(
            resolved, args.store, config_generation=args.config_generation,
            snapshot_version=args.snapshot_version)
        controller.__enter__()
    else:
        controller, runtime = _runtime_owner(runtime_factory, resolved, args)
    if args.once:
        try:
            print(json.dumps(controller.publish_snapshot(), sort_keys=True))
            return 0
        finally:
            controller.close()
    assert listen is not None and token is not None
    execution = None
    runtime_closed = runtime is None
    service = None
    read_fd = None
    write_fd = None
    previous_sigterm = None
    handler_attempted = False
    try:
        read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def notify_shutdown(_signum, _frame):
            try:
                os.write(write_fd, b"x")
            except BlockingIOError:
                pass

        handler_attempted = True
        signal.signal(signal.SIGTERM, notify_shutdown)
        if runtime is not None:
            recovered = runtime.recover()
            from . import standalone_runtime as runtime_module
            if not isinstance(recovered, runtime_module.RuntimeTickResult):
                raise ControlRefused("runtime recovery returned an untyped result")
            if recovered.status not in {"recovered", "settled"}:
                raise ControlRefused(f"runtime recovery unavailable: {recovered.reason}")
            pending_signal, _, _ = select.select([read_fd], [], [], 0)
            if pending_signal:
                os.read(read_fd, 4096)
                deadline = time.monotonic() + args.shutdown_deadline
                controller.request_shutdown_drain()
                runtime.request_stop()
                closed = _close_runtime(runtime, deadline)
                if closed.status != "closed":
                    raise ControlRefused(
                        f"runtime cleanup remains unresolved: {closed.reason}")
                runtime_closed = True
                controller.await_shutdown_drain(deadline)
                controller.close()
                return 0
        service = CampaignHTTPService(
            controller, *listen, token, refresh_interval=args.refresh_interval,
            total_request_deadline=args.request_deadline,
            allowed_origin=(args.trusted_origin or os.environ.get(TRUSTED_ORIGIN_ENV)))
        service.start()
        if runtime is not None:
            execution = _RuntimeExecution(runtime)
            execution.start()
        shutdown_requested = False
        shutdown_deadline = None
        deadline_reported = False
        runtime_unresolved = False
        while True:
            readable, _, _ = select.select([read_fd], [], [], 0.25)
            worker_finished = execution is not None and execution.finished()
            if worker_finished and execution is not None \
                    and execution.result is not None \
                    and execution.result.status == "recovery_required":
                runtime_unresolved = True
                service.report_runtime_unresolved()
            publisher_failed = service.publisher_error is not None
            transport_stopped = service.thread is None or not service.thread.is_alive()
            if readable:
                try:
                    os.read(read_fd, 4096)
                except BlockingIOError:
                    pass
            if (readable or worker_finished or publisher_failed or transport_stopped) \
                    and not shutdown_requested:
                controller.request_shutdown_drain()
                shutdown_requested = True
                shutdown_deadline = time.monotonic() + args.shutdown_deadline
                if execution is not None:
                    execution.request_stop()
            if shutdown_requested:
                try:
                    controller.reconcile_shutdown_ownership()
                except ControlRefused as exc:
                    if not deadline_reported:
                        print(f"shutdown ownership unresolved: {exc}",
                              file=sys.stderr, flush=True)
                try:
                    assert shutdown_deadline is not None
                    controller.await_shutdown_drain(shutdown_deadline)
                except ControlRefused as exc:
                    if not deadline_reported:
                        print(f"shutdown remains unresolved: {exc}",
                              file=sys.stderr, flush=True)
                        deadline_reported = True
                    continue
                if runtime_unresolved:
                    # A typed unresolved terminal cannot become absence proof merely
                    # because the runtime thread returned and the drain is quiescent.
                    continue
                if execution is not None and not execution.join(shutdown_deadline):
                    if not deadline_reported:
                        print("shutdown runtime thread remains owned",
                              file=sys.stderr, flush=True)
                        deadline_reported = True
                    continue
                if runtime is not None:
                    closed = _close_runtime(runtime, shutdown_deadline)
                    if closed.status != "closed":
                        if not deadline_reported:
                            print(f"shutdown runtime unresolved: {closed.reason}",
                                  file=sys.stderr, flush=True)
                            deadline_reported = True
                        continue
                    runtime_closed = True
                break
        if execution is not None and execution.error is not None:
            raise ControlRefused(
                f"runtime worker failed: {type(execution.error).__name__}")
        if execution is not None and execution.result is not None \
                and execution.result.status == "recovery_required":
            raise ControlRefused(
                f"runtime worker stopped unresolved: {execution.result.reason}")
        service.close()
        service = None
        controller.close()
        return 0
    except BaseException:
        if execution is not None and execution.thread.is_alive():
            # The non-daemon runtime thread and entered controller retain ownership.
            raise
        if runtime is not None and not runtime_closed:
            closed = _close_runtime(runtime, time.monotonic() + args.shutdown_deadline)
            runtime_closed = closed.status == "closed"
            if not runtime_closed:
                raise ControlRefused(
                    f"runtime cleanup remains unresolved: {closed.reason}")
        if service is not None:
            service.close()
        controller.close()
        raise
    finally:
        restored = not handler_attempted
        if handler_attempted and previous_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, previous_sigterm)
                restored = True
            except BaseException:
                # Keep the self-pipe valid if its handler could still be installed.
                restored = False
        if restored:
            descriptor_error = None
            for descriptor in (read_fd, write_fd):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError as exc:
                        descriptor_error = descriptor_error or exc
            if descriptor_error is not None:
                raise descriptor_error
        else:
            raise ControlRefused(
                "SIGTERM handler restoration failed; self-pipe ownership retained")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CampaignHTTPService", "DEFAULT_REFRESH_INTERVAL_S", "HEALTH_SCHEMA",
           "MAX_BODY", "MAX_HEADERS", "TOTAL_REQUEST_DEADLINE_S",
           "RuntimeFactory", "TRUSTED_ORIGIN_ENV", "load_resolved", "main", "make_handler",
           "parse_listen", "validate_origin"]
