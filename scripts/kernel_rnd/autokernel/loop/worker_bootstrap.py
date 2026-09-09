#!/usr/bin/env python3
"""Pipe-gated bootstrap for one exact lifecycle worker command.

This is deliberately not a command server.  It accepts one bounded contract on an
already-owned inherited file descriptor, waits for one byte of execution authority,
executes the exact argv without a shell, and returns one closed outcome on another
inherited pipe.  Its parent must attach this bootstrap to an owned container before
releasing the gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA = "epyc.autokernel.worker_bootstrap_contract.v1"
OUTCOME_SCHEMA = "epyc.autokernel.worker_bootstrap_outcome.v1"
MAX_CONTRACT_BYTES = 64 * 1024
MAX_OUTCOME_BYTES = 16 * 1024
MAX_LOG_BYTES_PER_STREAM = 64 * 1024
_CONTRACT_FIELDS = frozenset({"schema", "nonce", "contract_digest", "argv", "env", "cwd"})


class BootstrapRefused(RuntimeError):
    """The inherited launch contract or gate is malformed."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def contract_digest(*, nonce: str, argv: Sequence[str], env: Mapping[str, str],
                    cwd: str) -> str:
    body = {"schema": CONTRACT_SCHEMA, "nonce": nonce, "argv": list(argv),
            "env": dict(env), "cwd": cwd}
    return hashlib.sha256(_canonical(body)).hexdigest()


def make_contract(*, nonce: str, argv: Sequence[str], env: Mapping[str, str],
                  cwd: str) -> dict[str, Any]:
    row = {"schema": CONTRACT_SCHEMA, "nonce": nonce, "argv": list(argv),
           "env": dict(env), "cwd": cwd}
    row["contract_digest"] = contract_digest(
        nonce=nonce, argv=argv, env=env, cwd=cwd)
    return validate_contract(row)


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONTRACT_FIELDS:
        raise BootstrapRefused("bootstrap contract has missing/unknown fields")
    row = dict(value)
    if row["schema"] != CONTRACT_SCHEMA:
        raise BootstrapRefused("unsupported bootstrap contract schema")
    if not isinstance(row["nonce"], str) or len(row["nonce"]) < 16:
        raise BootstrapRefused("bootstrap nonce is invalid")
    argv = row["argv"]
    if (not isinstance(argv, list) or not argv or len(argv) > 256
            or any(not isinstance(item, str) or not item or "\0" in item
                   or len(item) > 16 * 1024 for item in argv)):
        raise BootstrapRefused("bootstrap argv is invalid")
    env = row["env"]
    if (not isinstance(env, Mapping) or len(env) > 256
            or any(not isinstance(key, str) or not key or "=" in key or "\0" in key
                   or not isinstance(item, str) or "\0" in item
                   or len(key) > 1024 or len(item) > 16 * 1024
                   for key, item in env.items())):
        raise BootstrapRefused("bootstrap env is invalid")
    if (not isinstance(row["cwd"], str) or not os.path.isabs(row["cwd"])
            or "\0" in row["cwd"]):
        raise BootstrapRefused("bootstrap cwd must be an absolute path")
    expected = contract_digest(nonce=row["nonce"], argv=argv, env=env,
                               cwd=row["cwd"])
    if row["contract_digest"] != expected:
        raise BootstrapRefused("bootstrap contract digest differs")
    row["env"] = dict(env)
    return row


def _read_bounded(fd: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(fd, min(65536, limit + 1 - size))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            raise BootstrapRefused("bootstrap contract exceeds size limit")


def _write_outcome(fd: int, value: Mapping[str, Any]) -> None:
    raw = _canonical(value) + b"\n"
    if len(raw) > MAX_OUTCOME_BYTES:
        raise BootstrapRefused("bootstrap outcome exceeds size limit")
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BootstrapRefused("bootstrap outcome pipe made no progress")
        view = view[written:]


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BootstrapRefused("worker log made no progress")
        view = view[written:]


def _drain_child(child: subprocess.Popen[bytes], stdout_fd: int,
                 stderr_fd: int) -> dict[str, Any]:
    selector = selectors.DefaultSelector()
    streams = {child.stdout: ("stdout", stdout_fd), child.stderr: ("stderr", stderr_fd)}
    totals = {"stdout": 0, "stderr": 0}
    retained = {"stdout": 0, "stderr": 0}
    digests = {"stdout": hashlib.sha256(), "stderr": hashlib.sha256()}
    for stream in streams:
        if stream is None:
            raise BootstrapRefused("worker output pipe is unavailable")
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ)
    try:
        while selector.get_map():
            for key, _mask in selector.select(timeout=0.05):
                stream = key.fileobj
                name, target_fd = streams[stream]
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                totals[name] += len(chunk)
                digests[name].update(chunk)
                available = MAX_LOG_BYTES_PER_STREAM - retained[name]
                if available > 0:
                    saved = chunk[:available]
                    _write_all(target_fd, saved)
                    retained[name] += len(saved)
        child.wait()
    finally:
        selector.close()
    return {
        "stdout_bytes": totals["stdout"], "stderr_bytes": totals["stderr"],
        "stdout_sha256": digests["stdout"].hexdigest(),
        "stderr_sha256": digests["stderr"].hexdigest(),
        "stdout_truncated": totals["stdout"] > retained["stdout"],
        "stderr_truncated": totals["stderr"] > retained["stderr"],
    }


def run(*, gate_fd: int, contract_fd: int, outcome_fd: int,
        stdout_fd: int, stderr_fd: int) -> int:
    raw = _read_bounded(contract_fd, MAX_CONTRACT_BYTES)
    os.close(contract_fd)
    try:
        contract = validate_contract(json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise BootstrapRefused("bootstrap contract is not canonical JSON") from exc
    token = os.read(gate_fd, 2)
    os.close(gate_fd)
    if token != b"G":
        raise BootstrapRefused("bootstrap execution gate was not released")

    child: subprocess.Popen[bytes] | None = None
    requested_signal: int | None = None

    def forward(signum: int, _frame: Any) -> None:
        nonlocal requested_signal
        requested_signal = requested_signal or signum
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    previous = {item: signal.signal(item, forward)
                for item in (signal.SIGTERM, signal.SIGINT)}
    try:
        child = subprocess.Popen(
            contract["argv"], cwd=contract["cwd"], env=contract["env"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True)
        output = _drain_child(child, stdout_fd, stderr_fd)
        return_code = child.returncode
        outcome = {
            "schema": OUTCOME_SCHEMA,
            "nonce": contract["nonce"],
            "contract_digest": contract["contract_digest"],
            "child_pid": child.pid,
            "return_code": return_code,
            "forwarded_signal": requested_signal,
            **output,
        }
        _write_outcome(outcome_fd, outcome)
        return 0
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)
        for fd in (stdout_fd, stderr_fd, outcome_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-fd", type=int, required=True)
    parser.add_argument("--contract-fd", type=int, required=True)
    parser.add_argument("--outcome-fd", type=int, required=True)
    parser.add_argument("--stdout-fd", type=int, required=True)
    parser.add_argument("--stderr-fd", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run(gate_fd=args.gate_fd, contract_fd=args.contract_fd,
                   outcome_fd=args.outcome_fd, stdout_fd=args.stdout_fd,
                   stderr_fd=args.stderr_fd)
    except BootstrapRefused as exc:
        print(f"worker bootstrap refused: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BootstrapRefused", "CONTRACT_SCHEMA", "MAX_CONTRACT_BYTES",
           "MAX_LOG_BYTES_PER_STREAM", "MAX_OUTCOME_BYTES", "OUTCOME_SCHEMA",
           "contract_digest", "main",
           "make_contract", "run", "validate_contract"]
