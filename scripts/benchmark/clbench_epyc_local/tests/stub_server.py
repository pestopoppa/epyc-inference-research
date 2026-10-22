"""Fake OpenAI-compatible /v1/chat/completions server that records every request."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class StubServer:
    def __init__(self, replies):
        self.requests: list[dict] = []
        self._replies = list(replies)
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                text = outer._replies.pop(0) if outer._replies else '{"action": "fold"}'
                extra = {}
                if isinstance(text, dict):  # {"content", "reasoning_content", "finish_reason", "completion_tokens"}
                    extra, text = text, text.get("content", "")
                n = len(outer.requests)
                out = json.dumps({
                    "id": f"stub-{n}",
                    "choices": [{"message": {"role": "assistant", "content": text,
                                             **({"reasoning_content": extra["reasoning_content"]} if "reasoning_content" in extra else {})},
                                 "finish_reason": extra.get("finish_reason", "stop")}],
                    "usage": {"prompt_tokens": 10 * n, "completion_tokens": extra.get("completion_tokens", 5),
                              "total_tokens": 10 * n + 5,
                              "prompt_tokens_details": {"cached_tokens": 4}},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self._srv.server_port}/v1"
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._srv.shutdown()
        self._srv.server_close()
