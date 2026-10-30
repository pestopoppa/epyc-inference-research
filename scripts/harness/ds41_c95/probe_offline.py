#!/usr/bin/env python3
import hashlib
import http.server
import json
import pathlib
import socketserver
import subprocess
import sys
import threading

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sandbox3

BASE = sandbox3.BASE
meta = json.loads((BASE / "contexts/C2/meta.json").read_text())
import os

root = pathlib.Path(os.environ["A6_OUTPUT_ROOT"]).resolve() / "probe-layout"
ws = root / "workspace"
result = root / "result"
for d in (ws, result):
    d.mkdir(parents=True, exist_ok=True)
policy = sandbox3.c95_policy(
    cutoff=sandbox3.iso_epoch(meta["asran_started"]),
    ws_root=ws,
    call_dir=result,
    scratch_clone=BASE / "lanes/0c8b896ee",
    context_id="C2",
    prompt_text=(BASE / "contexts/C2/prompt.inline.txt").read_text(),
)
pf = root / "policy.json"
pf.write_text(json.dumps(policy, indent=2) + "\n")


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"WIRE_OK")

    def log_message(self, *_):
        pass


server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
probes = [
    (
        "future provenance blocked",
        False,
        "cat /mnt/raid0/llm/tmp/ds41-c95/ctx-campaign/store/anchor-gen-007/provenance.json",
    ),
    (
        "live campaign blocked",
        False,
        "ls /mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store/anchor-gen-007",
    ),
    (
        "own anchor reachable",
        True,
        "test -r /mnt/raid0/llm/tmp/ds41-c95/ctx-campaign/store/anchor-gen-006/provenance.json",
    ),
    ("own workspace writable", True, f"touch {ws}/probe && test -f {ws}/probe"),
    ("own result writable", True, f"touch {result}/probe && test -f {result}/probe"),
    (
        "wire port reachable",
        True,
        f"python3 -c \"import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:{server.server_address[1]}/').read() == b'WIRE_OK'\"",
    ),
]
rows = []
try:
    for label, expected, command in probes:
        control = subprocess.run(
            ["bash", "-lc", command], capture_output=True, text=True, timeout=30
        )
        guarded = subprocess.run(
            [
                sys.executable,
                str(HERE / "sandbox3.py"),
                "exec",
                "--policy",
                str(pf),
                "--",
                "bash",
                "-lc",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        rows.append(
            {
                "probe": label,
                "expected_allowed": expected,
                "control_rc": control.returncode,
                "sandbox_rc": guarded.returncode,
                "sandbox_stderr": guarded.stderr.strip()[:220],
                "pass": control.returncode == 0 and (guarded.returncode == 0) == expected,
            }
        )
finally:
    server.shutdown()
    server.server_close()
report = {
    "abi": sandbox3.landlock_abi(),
    "policy_sha256": hashlib.sha256(pf.read_bytes()).hexdigest(),
    "rows": rows,
    "all_pass": all(r["pass"] for r in rows),
}
(root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["all_pass"] else 1)
