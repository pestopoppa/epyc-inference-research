"""Bounded, synthetic grader-reference isolation probe for the Docker scorer.

The host creates and reads the reference; only its path is sent to the test
candidate. The candidate runs fixed diagnostic code, not arbitrary model code.
This is a probe of one container boundary, not isolation for /chat or OpenCode,
and it cannot establish secrecy of tests embedded in a scored program.
"""
from __future__ import annotations

import argparse
import base64
import json
import secrets
import socket
import tempfile
from pathlib import Path

from scripts.benchmark.code_exec_scorer import SandboxUnavailable, _run_once


def _candidate(reference: str, port: int) -> str:
    # Only the pathname is passed; the secret reference bytes stay on the host.
    return f'''import base64, json, os, socket, stat
reference = {reference!r}
port = {port}
def read_file(path):
    try:
        with open(path, "rb") as stream:
            data = stream.read(4097)
            return {{"opened": True, "data": base64.b64encode(data[:4096]).decode(),
                    "truncated": len(data) > 4096}}
    except OSError as exc:
        return {{"opened": False, "errno": exc.errno}}
fds = []
for root in ("/proc/self/fd", "/proc/1/fd"):
    try:
        names = os.listdir(root)
    except OSError:
        continue
    for name in names:
        path = root + "/" + name
        try:
            target = os.readlink(path)
            item = {{"target": target}}
            if stat.S_ISREG(os.stat(path).st_mode):
                item["read"] = read_file(path)
            fds.append(item)
        except OSError:
            pass
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(("127.0.0.1", port))
    s.sendall(b"evl46-probe")
    sent = True
except OSError:
    sent = False
finally:
    s.close()
print(json.dumps({{"direct": read_file(reference),
                  "proc_root": read_file("/proc/1/root" + reference),
                  "environ_self": read_file("/proc/self/environ"),
                  "environ_init": read_file("/proc/1/environ"),
                  "fds": fds, "sent": sent,
                  "affinity": sorted(os.sched_getaffinity(0))}}, sort_keys=True))
'''


def probe(image: str, cpuset: str, scratch_root: str | None = None) -> dict:
    """Run one synthetic probe; return only validated, non-secret check results."""
    from scripts.benchmark.code_exec_scorer import _parse_cpuset

    secret = secrets.token_hex(32).encode()
    with tempfile.TemporaryDirectory(prefix="grader-ref-", dir=scratch_root) as directory:
        reference = Path(directory) / "reference"
        reference.write_bytes(secret)
        host_read = reference.read_bytes() == secret
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(0.2)
            ok, output = _run_once(_candidate(str(reference), listener.getsockname()[1]),
                                   "", 20, 10, 256, isolation="docker",
                                   docker_image=image, docker_cpuset=cpuset)
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                network_received = False
            else:
                with connection:
                    network_received = bool(connection.recv(1024))
        if not ok or len(output.encode()) > 65536 or secret.decode() in output:
            raise SandboxUnavailable("untrusted probe export failed validation")
        try:
            record = json.loads(output)
            expected = {"direct", "proc_root", "environ_self", "environ_init", "fds", "sent", "affinity"}
            if set(record) != expected or not isinstance(record["fds"], list):
                raise ValueError("unexpected probe export")
            blob = json.dumps(record, sort_keys=True).encode()
            complete = bool(record["fds"]) and all(
                record[name].get("opened") is True for name in ("environ_self", "environ_init")
            )
            for name in ("environ_self", "environ_init"):
                if record[name].get("opened"):
                    blob += base64.b64decode(record[name]["data"], validate=True)
                    complete = complete and record[name].get("truncated") is False
            for fd in record["fds"]:
                if fd.get("read", {}).get("opened"):
                    blob += base64.b64decode(fd["read"]["data"], validate=True)
                    complete = complete and fd["read"].get("truncated") is False
            checks = {
                "host_reference_readable": host_read,
                "direct_reference_denied": record["direct"]["opened"] is False,
                "proc_root_reference_denied": record["proc_root"]["opened"] is False,
                "reference_absent_from_environment_and_fds": (
                    complete and secret not in blob and str(reference).encode() not in blob
                ),
                "host_listener_received_no_bytes": not network_received,
                "candidate_reported_no_send": record["sent"] is False,
                "affinity_matches": record["affinity"] == sorted(_parse_cpuset(cpuset)),
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise SandboxUnavailable(f"invalid probe export: {exc}") from exc
    return {"checks": checks, "passed": all(checks.values()), "image": image,
            "cpuset": cpuset,
            "scope": "synthetic Docker scorer probe; no claim about chat, OpenCode or embedded tests"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="trusted local sha256 image ID")
    parser.add_argument("--cpuset", required=True, help="CPUs already claimed and in caller affinity")
    parser.add_argument("--scratch-root", default=None)
    args = parser.parse_args()
    result = probe(args.image, args.cpuset, args.scratch_root)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
