import hashlib
import json
import sys
from pathlib import Path

role, port, response_path, meta_path, out_path = sys.argv[1:]
response_text = Path(response_path).read_text(encoding="utf-8")
meta_text = Path(meta_path).read_text(encoding="utf-8")
try:
    response = json.loads(response_text)
except json.JSONDecodeError as exc:
    response = None
    content = ""
    parse_error = str(exc)
else:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or choice.get("text") or ""
    parse_error = None
try:
    curl_meta = json.loads(meta_text)
except json.JSONDecodeError:
    curl_meta = dict(raw=meta_text.strip())
ok = content.strip() == "OP2_READY"
out = dict(
    role=role,
    port=int(port),
    ok=ok,
    expected="OP2_READY",
    content=content,
    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    response_parse_error=parse_error,
    usage=(response or {}).get("usage"),
    timings=(response or {}).get("timings"),
    curl=curl_meta,
)
Path(out_path).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(out, sort_keys=True))
