#!/usr/bin/env python3
"""Runs inside Landlock: filesystem controls and one native model request via proxy3."""

import json
import os
import pathlib
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
body = {
    "model": os.environ["A6_MODEL_NAME"],
    "messages": [{"role": "user", "content": "Reply with the single word READY."}],
    "temperature": 0,
    "max_tokens": 12,
    "stream": False,
}
req = urllib.request.Request(
    os.environ["A6_PROXY_URL"] + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
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
