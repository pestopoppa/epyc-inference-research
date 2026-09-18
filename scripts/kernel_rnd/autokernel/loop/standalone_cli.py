#!/usr/bin/env python3
"""Bounded client for the existing authenticated campaign control service.

This module is a client only.  It creates no controller, service, token, claim,
worker, or campaign store.
"""
from __future__ import annotations

import argparse
import hmac
import http.client
import json
import os
from pathlib import Path
import socket
import stat
import sys
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from . import campaign_control as control
from . import campaign_command_v2
from . import campaign_service as service
from . import worker_lifecycle


OUTPUT_SCHEMA = "epyc.autokernel.standalone_cli_result.v1"
MAX_RESPONSE_BYTES = 128 * 1024
DEFAULT_TIMEOUT_S = 2.0
MAX_TIMEOUT_S = 30.0
MAX_RETRIES = 2
TOKEN_ENV = "AUTOKERNEL_CONTROL_TOKEN"
MAX_TOKEN_BYTES = 4096


class CLIRefused(RuntimeError):
    """A local or remote standalone operation failed closed."""


def _endpoint(value: str) -> tuple[str, int]:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise CLIRefused("endpoint must be an exact http numeric-loopback origin")
    try:
        host = parsed.hostname
        if host is None:
            raise ValueError
        import ipaddress
        address = ipaddress.ip_address(host)
        if address.version != 4 or not address.is_loopback:
            raise ValueError
        port = parsed.port
    except ValueError as exc:
        raise CLIRefused("endpoint must use numeric IPv4 loopback and an explicit port") from exc
    if port is None:
        raise CLIRefused("endpoint must include an explicit port")
    return host, port


def _token_from(args: argparse.Namespace) -> str:
    if args.token_file is not None:
        path = Path(args.token_file)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as exc:
            raise CLIRefused(f"cannot read protected token file: {exc}") from exc
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise CLIRefused("token file must be owner-owned regular mode 0600 with one link")
            raw = os.read(fd, MAX_TOKEN_BYTES + 1)
            if len(raw) > MAX_TOKEN_BYTES or os.read(fd, 1):
                raise CLIRefused("token file exceeds the size bound")
            try:
                token = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise CLIRefused("token file is not UTF-8") from exc
        finally:
            os.close(fd)
    else:
        token = os.environ.get(args.token_env, "")
    return _authorization_token(token)


def _authorization_token(token: Any) -> str:
    """Return an HTTP-header-safe bounded token without echoing rejected input."""
    if not isinstance(token, str) or not token:
        raise CLIRefused("a non-empty control token is required")
    try:
        encoded = token.encode("utf-8")
    except UnicodeError as exc:
        raise CLIRefused("control token encoding is invalid") from exc
    if len(encoded) > MAX_TOKEN_BYTES:
        raise CLIRefused("control token exceeds the size bound")
    # Bearer credentials are constrained to visible ASCII.  This rejects CR/LF,
    # every other HTTP control byte, whitespace ambiguity, and latin-1 surprises.
    if any(byte < 0x21 or byte > 0x7e for byte in encoded):
        raise CLIRefused("control token is unsafe for an HTTP Authorization header")
    return token


def _decode_response(response: http.client.HTTPResponse) -> dict[str, Any]:
    length = response.getheader("Content-Length")
    if length is not None:
        if not length.isdigit() or int(length) > MAX_RESPONSE_BYTES:
            raise CLIRefused("response Content-Length is invalid or exceeds the bound")
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise CLIRefused("response body exceeds the bound")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CLIRefused("response is not one JSON document") from exc
    if not isinstance(value, Mapping):
        raise CLIRefused("response JSON must be an object")
    return dict(value)


def _request(endpoint: str, path: str, *, token: str | None = None,
             body: Mapping[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT_S,
             retries: int = 0) -> tuple[int, dict[str, Any]]:
    host, port = _endpoint(endpoint)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) \
            or not 0 < float(timeout) <= MAX_TIMEOUT_S:
        raise CLIRefused(f"timeout must be positive and at most {MAX_TIMEOUT_S:g} seconds")
    if not isinstance(retries, int) or isinstance(retries, bool) \
            or not 0 <= retries <= MAX_RETRIES:
        raise CLIRefused(f"retries must be from 0 through {MAX_RETRIES}")
    raw = None if body is None else json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    headers = {"Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {_authorization_token(token)}"
    if raw is not None:
        headers["Content-Type"] = "application/json"
    last: BaseException | None = None
    for attempt in range(retries + 1):
        connection = http.client.HTTPConnection(host, port, timeout=float(timeout))
        expired = threading.Event()
        socket_holder: list[socket.socket | None] = [None]
        attempt_deadline = time.monotonic() + float(timeout)

        def expire_owned_socket() -> None:
            expired.set()
            owned = socket_holder[0] or connection.sock
            if owned is not None:
                try:
                    owned.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        # Socket timeouts are per operation; this watchdog bounds connect,
        # response headers, and the complete body as one attempt.
        watchdog = threading.Timer(float(timeout), expire_owned_socket)
        watchdog.name = "standalone-cli-request-deadline"
        watchdog.daemon = True
        watchdog.start()
        try:
            connection.request("POST" if raw is not None else "GET", path,
                               body=raw, headers=headers)
            socket_holder[0] = connection.sock
            response = connection.getresponse()
            decoded = _decode_response(response)
            if expired.is_set() or time.monotonic() >= attempt_deadline:
                raise CLIRefused("absolute request deadline expired")
            return response.status, decoded
        except CLIRefused as exc:
            if not expired.is_set():
                raise
            last = exc
            if attempt == retries:
                break
        except (OSError, http.client.HTTPException, ValueError) as exc:
            last = exc
            if attempt == retries:
                break
        finally:
            watchdog.cancel()
            connection.close()
            watchdog.join(timeout=float(timeout))
    reason = "absolute request deadline expired" if expired.is_set() else "transport refused"
    raise CLIRefused(f"{reason} after {retries + 1} attempt(s)") from last


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise CLIRefused(f"{label} must be lowercase SHA-256")
    return value


def validate_health(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "transport", "service_build", "campaign_id",
              "config_generation", "config_digest", "supervisor_incarnation",
              "stream_epoch", "allowed_origin", "ok", "producer", "error"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CLIRefused("transport health has missing/unknown fields")
    row = dict(value)
    if row["schema"] != service.HEALTH_SCHEMA or row["transport"] != "campaign-control-http":
        raise CLIRefused("transport health schema/transport is unsupported")
    if not isinstance(row["campaign_id"], str) or not row["campaign_id"]:
        raise CLIRefused("transport campaign identity is invalid")
    _sha(row["config_digest"], "transport config_digest")
    for name in ("config_generation", "supervisor_incarnation", "stream_epoch"):
        if not isinstance(row[name], int) or isinstance(row[name], bool) or row[name] < 1:
            raise CLIRefused(f"transport {name} is invalid")
    if type(row["ok"]) is not bool or row["producer"] not in {"starting", "running", "failed"}:
        raise CLIRefused("transport producer state is invalid")
    if row["error"] is not None and not isinstance(row["error"], str):
        raise CLIRefused("transport error is invalid")
    build = row["service_build"]
    build_fields = {"schema", "scope", "module", "identity_basis", "included_symbols",
                    "excluded_scope", "sha256"}
    if not isinstance(build, Mapping) or set(build) != build_fields:
        raise CLIRefused("transport loaded-build identity is invalid")
    _sha(build["sha256"], "transport loaded-build sha256")
    return row


def _check_identity(row: Mapping[str, Any], args: argparse.Namespace, *,
                    allow_newer_supervisor: bool = False) -> None:
    expected = {
        "campaign_id": args.expect_campaign_id,
        "config_generation": args.expect_config_generation,
        "config_digest": args.expect_config_digest,
        "supervisor_incarnation": args.expect_supervisor_incarnation,
    }
    for key, value in expected.items():
        if key == "supervisor_incarnation" and allow_newer_supervisor \
                and value is not None:
            actual = row.get(key)
            if not isinstance(actual, int) or isinstance(actual, bool) or actual < value:
                raise CLIRefused(
                    "endpoint supervisor_incarnation predates or cannot satisfy "
                    "the historical command pin")
            continue
        if value is not None and not hmac.compare_digest(str(row.get(key)), str(value)):
            raise CLIRefused(f"endpoint {key} differs from the pinned identity")


def _snapshot(args: argparse.Namespace, token: str, *,
              allow_newer_supervisor: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    code, health_raw = _request(args.endpoint, "/health", timeout=args.timeout)
    if code != 200:
        raise CLIRefused(f"transport health refused with HTTP {code}")
    health = validate_health(health_raw)
    _check_identity(health, args, allow_newer_supervisor=allow_newer_supervisor)
    code, snapshot_raw = _request(args.endpoint, "/snapshot", token=token,
                                  timeout=args.timeout)
    if code != 200:
        raise CLIRefused(f"authenticated snapshot refused with HTTP {code}")
    schema = snapshot_raw.get("schema") if isinstance(snapshot_raw, Mapping) else None
    if schema == getattr(control, "SNAPSHOT_SCHEMA_V2", None):
        validator = control.validate_snapshot_v2
        version = "v2"
    elif schema == "epyc.autokernel.campaign_snapshot.v3":
        validator = getattr(control, "validate_snapshot_v3", None)
        if validator is None:
            raise CLIRefused(
                "snapshot v3 is unavailable: the installed campaign_control lacks "
                "validate_snapshot_v3")
        version = "v3"
    else:
        raise CLIRefused("snapshot schema is unsupported")
    try:
        snapshot = validator(snapshot_raw)
    except (control.ControlRefused, worker_lifecycle.LifecycleRefused) as exc:
        raise CLIRefused(f"snapshot refused by the {version} typed validator: {exc}") from exc
    _check_identity(snapshot, args, allow_newer_supervisor=allow_newer_supervisor)
    for key in ("campaign_id", "config_generation", "config_digest",
                "supervisor_incarnation", "stream_epoch"):
        if snapshot[key] != health[key]:
            raise CLIRefused(f"health/snapshot {key} differs")
    return health, snapshot


def _output(action: str, disposition: str, **values: Any) -> dict[str, Any]:
    return {"schema": OUTPUT_SCHEMA, "action": action, "disposition": disposition, **values}


def _redact(value: Any, secret: str) -> Any:
    """Remove the credential from every authenticated report leaf."""
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, Mapping):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "inspect":
        code, raw = _request(args.endpoint, "/health", timeout=args.timeout)
        health = validate_health(raw)
        return _output("inspect", "observed" if code == 200 else "refused",
                       transport_health=health, semantic_liveness="not_inspected")
    if args.action == "preflight":
        code, raw = _request(args.endpoint, "/health", timeout=args.timeout)
        health = validate_health(raw)
        enrollment = None
        if args.resolved_campaign is not None:
            try:
                resolved = service.load_resolved(args.resolved_campaign)
            except control.ControlRefused as exc:
                raise CLIRefused(f"resolved enrollment is invalid: {exc}") from exc
            enrollment = {
                "loader": "campaign_service.load_resolved",
                "campaign_id": resolved.campaign_id,
                "requested_manifest_digest": resolved.manifest_digest,
                "config_digest": control.resolved_config_digest(resolved),
            }
        return _output(
            "preflight", "not_ready", transport_http_status=code,
            transport_health=health, mutating=False, token_accessed=False,
            claims_obtained=False, resolved_enrollment=enrollment,
            prerequisites={
                "authenticated_snapshot": "not_checked_by_nonmutating_preflight",
                "runtime_launch_bridge": "unavailable",
                "provider_authority": "unavailable",
                "sigterm_drain_bridge": "unavailable",
            })
    token = _token_from(args)
    health, snapshot = _snapshot(
        args, token, allow_newer_supervisor=args.action != "status")
    if args.action == "status":
        return _redact(_output(
            "status", "observed", transport_health=health,
            campaign_snapshot=snapshot,
            semantic_liveness=("management_only" if not snapshot[
                "execution_capability_available"] else "producer_reported_capability")), token)
    operation = args.action
    if args.request_id is None or args.expected_revision is None:
        raise CLIRefused("control requires --request-id and --expected-revision")
    command = {
        "schema": campaign_command_v2.COMMAND_SCHEMA,
        "campaign_id": args.expect_campaign_id,
        "config_generation": args.expect_config_generation,
        "config_digest": args.expect_config_digest,
        "supervisor_incarnation": args.expect_supervisor_incarnation,
        "request_id": args.request_id,
        "operation": operation,
        "payload": {},
        "expected_control_revision": args.expected_revision,
    }
    command["payload_digest"] = campaign_command_v2.command_digest(
        operation=operation, payload={}, campaign_id=args.expect_campaign_id,
        config_generation=args.expect_config_generation,
        config_digest=args.expect_config_digest,
        supervisor_incarnation=args.expect_supervisor_incarnation,
        request_id=args.request_id,
        expected_control_revision=args.expected_revision)
    campaign_command_v2.validate_command(command)
    code, raw = _request(args.endpoint, "/commands", token=token, body=command,
                         timeout=args.timeout, retries=args.retries)
    if code != 200:
        return _redact(_output(
            operation, "refused", http_status=code,
            request_id=args.request_id, payload_digest=command["payload_digest"],
            expected_control_revision=args.expected_revision,
            remote_refusal="remote request refused; body omitted"), token)
    try:
        result = worker_lifecycle.validate_command_result_v2(raw)
    except worker_lifecycle.LifecycleRefused as exc:
        raise CLIRefused(f"command result refused by the v2 typed validator: {exc}") from exc
    for key in ("request_id", "operation", "payload_digest"):
        if result[key] != command[key]:
            raise CLIRefused(f"command result {key} differs from the request")
    disposition = ("applied" if result["completed"]
                   and result["desired_state"] == result["observed_state"]
                   else "requested")
    return _redact(_output(operation, disposition, http_status=code, command=command,
                           command_result=result), token)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "preflight", "status", "pause",
                                           "resume", "drain"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8077")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--resolved-campaign", type=Path,
                        help="campaign_cli resolution/enrollment envelope to inspect in preflight")
    parser.add_argument("--token-env", default=TOKEN_ENV,
                        help="environment variable name; token values are never accepted on argv")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--expect-campaign-id")
    parser.add_argument("--expect-config-generation", type=int)
    parser.add_argument("--expect-config-digest")
    parser.add_argument("--expect-supervisor-incarnation", type=int)
    parser.add_argument("--request-id")
    parser.add_argument("--expected-revision", type=int)
    parser.add_argument("--retries", type=int, default=0,
                        help="transport retries reuse the exact command body")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action in {"status", "pause", "resume", "drain"}:
        required = (args.expect_campaign_id, args.expect_config_generation,
                    args.expect_config_digest, args.expect_supervisor_incarnation)
        if any(value is None for value in required):
            print("standalone CLI refused: authenticated operations require all --expect-* pins",
                  file=sys.stderr)
            return 2
    try:
        result = execute(args)
    except (CLIRefused, control.ControlRefused, ValueError) as exc:
        print(f"standalone CLI refused: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["disposition"] not in {"refused"} else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLIRefused", "MAX_RESPONSE_BYTES", "MAX_RETRIES", "MAX_TIMEOUT_S",
           "MAX_TOKEN_BYTES", "OUTPUT_SCHEMA", "execute", "main", "validate_health"]
