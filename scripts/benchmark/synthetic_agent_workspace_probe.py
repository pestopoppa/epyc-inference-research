"""Synthetic file-agent workspace and host-only oracle boundary probe.

Only a fixed, benign inventory agent is supported. Its public input is carried
over stdin and reconstructed inside a fresh no-network Docker sandbox. This is
not a live REPL/OpenCode transport or an authorization receipt for one.
"""
from __future__ import annotations

import hashlib
import json
import base64
import secrets
import socket
import tempfile
from pathlib import Path

from scripts.benchmark.code_exec_scorer import SandboxUnavailable, _parse_cpuset, _run_once

AGENT_SOURCE = r'''import base64, csv, json, os, pathlib, socket, stat, sys
request = json.load(sys.stdin)
workspace = pathlib.Path('/tmp/workspace')
workspace.mkdir(mode=0o700)
target = workspace / 'inventory.csv'
target.write_text(request['inventory'], encoding='utf-8')
try:
    pathlib.Path(request['reference_probe_path']).read_bytes()
    direct_denied = False
except OSError:
    direct_denied = True
try:
    pathlib.Path('/proc/1/root' + request['reference_probe_path']).read_bytes()
    proc_denied = False
except OSError:
    proc_denied = True
try:
    pathlib.Path(request['reference_probe_path']).write_text('modified')
    write_denied = False
except OSError:
    write_denied = True
def read_sample(path):
    try:
        with open(path, 'rb') as f:
            data = f.read(4097)
        return {'opened': True, 'truncated': len(data) > 4096,
                'data': base64.b64encode(data[:4096]).decode()}
    except OSError:
        return {'opened': False}
fds = []
for root in ('/proc/self/fd', '/proc/1/fd'):
    for name in os.listdir(root):
        path = root + '/' + name
        try:
            item = {'target': os.readlink(path)}
            if stat.S_ISREG(os.stat(path).st_mode):
                item['sample'] = read_sample(path)
            fds.append(item)
        except OSError:
            pass
conn = socket.socket()
conn.settimeout(0.5)
try:
    conn.connect(('127.0.0.1', request['probe_port']))
    conn.sendall(b'evl46-probe')
    sent = True
except OSError:
    sent = False
finally:
    conn.close()
with target.open(newline='', encoding='utf-8') as source:
    total = sum(int(row['quantity']) for row in csv.DictReader(source))
answer = json.dumps({'total': total}, sort_keys=True, separators=(',', ':'))
print(json.dumps({'answer': answer, 'direct_denied': direct_denied,
                  'proc_denied': proc_denied, 'write_denied': write_denied,
                  'environ_self': read_sample('/proc/self/environ'),
                  'environ_init': read_sample('/proc/1/environ'), 'fds': fds, 'sent': sent,
                  'affinity': sorted(os.sched_getaffinity(0)), 'workspace': str(workspace)},
                  sort_keys=True, separators=(',', ':')))
'''


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_public_inventory(inventory: str) -> bytes:
    if not isinstance(inventory, str):
        raise ValueError("inventory must be UTF-8 text")
    data = inventory.encode("utf-8")
    if not data or len(data) > 8192:
        raise ValueError("inventory must be nonempty and at most 8192 bytes")
    return data


def run_probe(image: str, cpuset: str, scratch_root: str | None = None) -> dict:
    """Run a source-bound synthetic trial; host oracle never enters candidate stdin."""
    inventory = "item,quantity\nalpha,4\nbeta,7\n"
    fixture = _validate_public_inventory(inventory)
    expected = '{"total":11}'
    canary = secrets.token_hex(32)
    with tempfile.TemporaryDirectory(prefix="agent-oracle-", dir=scratch_root) as directory:
        reference = Path(directory) / "reference.json"
        hidden = json.dumps({"expected": expected, "canary": canary}, sort_keys=True)
        reference.write_text(hidden, encoding="utf-8")
        # Negative probe deliberately gets the path, never the file bytes.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(0.2)
            request = json.dumps({"inventory": inventory, "reference_probe_path": str(reference),
                                  "probe_port": listener.getsockname()[1]},
                                 sort_keys=True, separators=(',', ':'))
            if canary in request or canary in AGENT_SOURCE:
                raise SandboxUnavailable("oracle material entered the candidate request")
            ok, raw = _run_once(AGENT_SOURCE, request, 20, 10, 256,
                                isolation="docker", docker_image=image, docker_cpuset=cpuset)
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                received = False
            else:
                with connection:
                    received = bool(connection.recv(1024))
        if not ok or len(raw.encode()) > 16384 or canary in raw:
            raise SandboxUnavailable("agent result failed controlled-export validation")
        try:
            export = json.loads(raw)
            if set(export) != {"answer", "direct_denied", "proc_denied", "write_denied",
                               "environ_self", "environ_init", "fds", "sent", "affinity", "workspace"}:
                raise ValueError("unexpected result schema")
            inspected = json.dumps(export, sort_keys=True).encode()
            complete = isinstance(export["fds"], list) and bool(export["fds"])
            for entry in [export["environ_self"], export["environ_init"]]:
                if entry.get("opened") is not True or entry.get("truncated") is not False:
                    complete = False
                else:
                    inspected += base64.b64decode(entry["data"], validate=True)
            for entry in export["fds"]:
                sample = entry.get("sample", {})
                if sample.get("opened"):
                    if sample.get("truncated") is not False:
                        complete = False
                    else:
                        inspected += base64.b64decode(sample["data"], validate=True)
            answer = export["answer"]
            if not isinstance(answer, str) or len(answer.encode()) > 1024:
                raise ValueError("invalid candidate answer")
            parsed = json.loads(answer)
            if (not isinstance(parsed, dict) or set(parsed) != {"total"}
                    or type(parsed["total"]) is not int or parsed["total"] < 0
                    or answer != json.dumps(parsed, sort_keys=True, separators=(',', ':'))):
                raise ValueError("candidate answer is not canonical inventory JSON")
            observed = reference.read_text(encoding="utf-8")
            if observed != hidden:
                raise ValueError("host oracle changed during execution")
            checks = {"reference_denied": export["direct_denied"] is True,
                      "proc_root_denied": export["proc_denied"] is True,
                      "reference_write_denied": export["write_denied"] is True,
                      "env_fds_no_canary": complete and canary.encode() not in inspected,
                      "host_network_write_denied": export["sent"] is False and not received,
                      "affinity_matches": export["affinity"] == sorted(_parse_cpuset(cpuset)),
                      "workspace_confined": export["workspace"] == "/tmp/workspace",
                      "reference_intact": observed == hidden,
                      "answer_matches": answer == expected}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise SandboxUnavailable(f"agent result has invalid schema: {exc}") from exc
    return {"schema": "synthetic-file-agent-isolation.v1", "passed": all(checks.values()),
            "checks": checks, "image": image, "cpuset": cpuset,
            "public_input_sha256": _hash(fixture), "agent_source_sha256": _hash(AGENT_SOURCE.encode()),
            "candidate_sha256": _hash(answer.encode()),
            "scope": "fixed synthetic agent only; not a /chat or OpenCode isolation receipt"}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--cpuset', required=True)
    parser.add_argument('--scratch-root', default=None)
    args = parser.parse_args()
    result = run_probe(args.image, args.cpuset, args.scratch_root)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
