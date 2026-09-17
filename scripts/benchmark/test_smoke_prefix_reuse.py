#!/usr/bin/env python3
"""M-12 B5: the prefix-reuse smoke decides PASS/FAIL from the server's own accounting.

A fake llama-server runs in-process on a stdlib HTTP thread, so no real server is
started and nothing is inferred.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import smoke_prefix_reuse as smoke


class FakeLlamaServer:
    """Emulates checkpoint reuse: the longest common prefix with any slot is 'cached',
    minus one checkpoint stride, unless ``reuse`` is off."""

    def __init__(self, *, reuse=True, n_slots=4, n_ctx=196608, model="Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
                 stride=2052):
        self.reuse, self.n_slots, self.n_ctx, self.model, self.stride = (
            reuse, n_slots, n_ctx, model, stride)
        self.slots: list[str] = []
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body):
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/health":
                    self._send({"status": "ok"})
                elif self.path == "/props":
                    self._send({"model_path": f"/mnt/raid0/llm/models/{outer.model}",
                                "build_info": "fake", "total_slots": outer.n_slots})
                elif self.path == "/slots":
                    self._send([{"id": i, "n_ctx": outer.n_ctx, "speculative": True}
                                for i in range(outer.n_slots)])
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                self._send(outer.complete(body["messages"][0]["content"]))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def complete(self, prompt: str) -> dict:
        tokens = len(prompt)  # one "token" per character is enough for the fake
        best, best_slot = 0, None
        for i, held in enumerate(self.slots):
            lcp = 0
            for x, y in zip(held, prompt):
                if x != y:
                    break
                lcp += 1
            if lcp > best:
                best, best_slot = lcp, i
        cached = max(0, best - self.stride) if (self.reuse and best) else 0
        if best_slot is None:
            if len(self.slots) < self.n_slots:
                self.slots.append(prompt)
            else:
                self.slots[0] = prompt
        else:
            self.slots[best_slot] = prompt
        processed = tokens - cached
        return {
            "choices": [{"message": {"content": "- A"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": tokens, "completion_tokens": 2,
                      "prompt_tokens_details": {"cached_tokens": cached}},
            "timings": {"cache_n": cached, "prompt_n": processed,
                        "prompt_ms": processed * 0.01, "predicted_n": 2},
        }

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


BOOK = "B" * 100_000
TRANSCRIPT = "T" * 180_000


@pytest.fixture(autouse=True)
def no_production_ports(monkeypatch, tmp_path):
    """Hermetic: the production port declarations are a fixture naming no test port."""
    manifest = tmp_path / "launch_manifest.yaml"
    manifest.write_text("port_map:\n  frontdoor: 2\n")
    monkeypatch.setattr(smoke, "PRODUCTION_PORT_SOURCES", (manifest,))


@pytest.fixture(autouse=True)
def fake_prompts(monkeypatch):
    monkeypatch.setattr(smoke, "tulving_prompts", lambda chapters=200: (
        "200ch", [BOOK + f"\n\nQ{i}" for i in range(3)]))
    monkeypatch.setattr(smoke, "beam_prompts", lambda: (
        "conversation 11", [TRANSCRIPT + f"\n\nU{i}" for i in range(2)]))
    monkeypatch.setattr(smoke, "_vram_card", lambda: None)


def _run(monkeypatch, tmp_path, server, *extra):
    out = tmp_path / "receipt.json"
    monkeypatch.setattr("sys.argv", ["smoke_prefix_reuse.py", "--port", str(server.port),
                                     "--out", str(out), *extra])
    code = smoke.main()
    return code, json.loads(out.read_text())


def test_reuse_passes(monkeypatch, tmp_path, capsys):
    with FakeLlamaServer() as server:
        code, receipt = _run(monkeypatch, tmp_path, server, "--legs", "tulving,beam",
                             "--expect-model", "Qwen3.6-35B-A3B-MTP-Q8_0")
    assert code == 0 and receipt["verdict"] == "PASS", receipt["fails"]
    assert capsys.readouterr().out.strip().splitlines()[-1] == "SMOKE_PREFIX_REUSE: PASS"
    legs = receipt["legs"]
    assert legs["tulving"]["B"]["cache_n"] > 0.95 * legs["tulving"]["B"]["prompt_tokens"]
    assert legs["beam"]["F"]["cache_n"] > 0.95 * legs["beam"]["F"]["prompt_tokens"]
    # The run parameters are the M-12 ones.
    body = server.requests[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["cache_prompt"] is True and body["temperature"] == 0.0
    # No prompt text is written into the receipt.
    assert "BBBB" not in json.dumps(receipt)


def test_no_reuse_fails(monkeypatch, tmp_path, capsys):
    with FakeLlamaServer(reuse=False) as server:
        code, receipt = _run(monkeypatch, tmp_path, server)
    assert code == 1 and receipt["verdict"] == "FAIL"
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("SMOKE_PREFIX_REUSE: FAIL (") and "checkpoint NOT restored" in last


def test_one_slot_evicts_the_book_and_fails_the_affinity_check(monkeypatch, tmp_path):
    with FakeLlamaServer(n_slots=1) as server:
        code, receipt = _run(monkeypatch, tmp_path, server, "--legs", "tulving,beam")
    assert code == 1
    assert any("book after transcript" in f for f in receipt["fails"])
    assert any("relies on --cache-ram" in w for w in receipt["warns"])


def test_short_slot_context_fails_before_any_request(monkeypatch, tmp_path):
    with FakeLlamaServer(n_ctx=16384) as server:
        code, receipt = _run(monkeypatch, tmp_path, server)
        assert server.requests == []
    assert code == 1 and "slot n_ctx" in receipt["fails"][0]


def test_wrong_model_fails(monkeypatch, tmp_path):
    with FakeLlamaServer(model="gemma-4-26B-A4B-it-ORIG-Q8_0.gguf") as server:
        code, receipt = _run(monkeypatch, tmp_path, server, "--expect-model", "Qwen3.8-27B")
    assert code == 1 and "does not contain" in receipt["fails"][0]


def test_unreachable_server_fails(monkeypatch, tmp_path):
    out = tmp_path / "r.json"
    monkeypatch.setattr("sys.argv", ["smoke_prefix_reuse.py", "--port", "1", "--out", str(out)])
    assert smoke.main() == 1
    assert "not reachable" in json.loads(out.read_text())["fails"][0]


def test_real_tulving_prompts_share_the_book_prefix():
    """Against the staged dataset (skipped if absent): three prompts, one shared book."""
    if not Path("/mnt/raid0/llm/data/eval/tulving_episodic/Udefault_Sdefault_seed0").exists():
        pytest.skip("Tulving dataset not staged")
    import importlib

    real = importlib.reload(smoke)
    tag, prompts = real.tulving_prompts(200)
    assert "196 chapters" in tag and len(prompts) == 3
    head = prompts[0][: len(prompts[0]) - 200]
    assert all(p.startswith(head) for p in prompts) and len(head) > 400_000
