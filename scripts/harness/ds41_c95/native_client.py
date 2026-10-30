#!/usr/bin/env python3
"""Runs inside Landlock: filesystem controls and one native model request via proxy3."""

import hashlib
import json
import os
import pathlib
import time
import urllib.request

work = pathlib.Path(os.environ["A6_WORK"])
result = pathlib.Path(os.environ["A6_RESULT"])
future = pathlib.Path(
    "/mnt/raid0/llm/tmp/ds41-c95/ctx-campaign/store/anchor-gen-007/provenance.json"
)
live = pathlib.Path(
    "/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store/anchor-gen-007/provenance.json"
)
checks = {}
for label, path in [("future", future), ("live", live)]:
    try:
        path.read_bytes()
        checks[label + "_denied"] = False
    except PermissionError:
        checks[label + "_denied"] = True
    except OSError as exc:
        checks[label + "_denied"] = False
        checks[label + "_error"] = str(exc)
anchor = pathlib.Path(
    "/mnt/raid0/llm/tmp/ds41-c95/ctx-campaign/store/anchor-gen-006/provenance.json"
)
checks["anchor_read"] = anchor.is_file() and len(anchor.read_bytes()) > 0
for label, directory in [("workspace", work), ("result", result)]:
    marker = directory / ("a6-" + label + ".marker")
    marker.write_text("A6_OK\n")
    checks[label + "_write"] = marker.read_text() == "A6_OK\n"
mode = os.environ.get("A6_MODE", "normal")
if mode not in ("normal", "cutoff"):
    raise SystemExit("invalid A6 mode")


def durable_new(path, value):
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


if mode == "cutoff":
    durable_new(result / "filesystem-checks.json", checks)
body = {
    "model": os.environ["A6_MODEL_NAME"],
    "messages": [{"role": "user", "content": (
        "Count upward from 1, one integer per line. Continue until 1000."
        if mode == "cutoff" else "Reply with the single word READY."
    )}],
    "temperature": 0,
    "max_tokens": 1024 if mode == "cutoff" else 12,
    "stream": mode == "cutoff",
}
req = urllib.request.Request(
    os.environ["A6_PROXY_URL"] + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
if mode == "cutoff":
    durable_new(result / "request-start.json", {
        "body_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "monotonic_ns": time.monotonic_ns(), "pid": os.getpid(), "stream": True,
    })
    stop = pathlib.Path(os.environ["A6_CUTOFF_STOP"])
    progress = 0
    request_id = None
    with urllib.request.urlopen(req, timeout=90) as response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                raise SystemExit("stream finished before owned cutoff")
            if not data:
                continue
            obj = json.loads(data)
            if request_id is None:
                request_id = obj.get("id")
            if not request_id or obj.get("id") != request_id:
                raise SystemExit("stream request ID missing or changed")
            for choice in obj.get("choices", []):
                if choice.get("finish_reason") is not None:
                    raise SystemExit("stream finished before owned cutoff")
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    progress += 1
                    if progress == 2:
                        durable_new(result / "stream-progress.json", {
                            "monotonic_ns": time.monotonic_ns(), "request_id": request_id,
                            "nonterminal_content_events": progress, "pid": os.getpid(),
                            "finish_observed": False, "done_observed": False,
                        })
            if stop.exists() and progress >= 2:
                durable_new(result / "cutoff-ack.json", {
                    "monotonic_ns": time.monotonic_ns(), "request_id": request_id,
                    "nonterminal_content_events": progress, "pid": os.getpid(),
                    "finish_observed": False, "done_observed": False,
                })
                raise SystemExit(75)
    raise SystemExit("stream ended before owned cutoff")
with urllib.request.urlopen(req, timeout=90) as response:
    obj = json.load(response)
checks["model_http"] = bool(obj.get("choices")) and bool(obj.get("id"))
checks["model_text"] = str(obj.get("choices", [{}])[0].get("message", {}).get("content", ""))[:120]
checks["wire_proxy_reached"] = True
(result / "client-proof.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks))
raise SystemExit(
    0
    if all(
        checks.get(k)
        for k in [
            "future_denied",
            "live_denied",
            "anchor_read",
            "workspace_write",
            "result_write",
            "model_http",
            "wire_proxy_reached",
        ]
    )
    else 1
)
