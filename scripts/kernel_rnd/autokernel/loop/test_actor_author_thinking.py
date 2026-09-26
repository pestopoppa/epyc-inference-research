"""OAB-24: the author's reasoning switch (`--actor-author-thinking {default,off}`).

Operator, 2026-09-25: the local 27B AUTHOR runs with thinking OFF, author calls only;
the planner keeps reasoning on. DS41 run 10c's author decoded 74,288 tokens in 2,700 s
re-deriving the block_q8_2_x4 layout inside <think> and made zero edits.

What must hold:

* config: "off" puts `chat_template_kwargs: {"enable_thinking": false}` in the model
  `options` of the AUTHOR's per-call OPENCODE_CONFIG (plain and bounded seat), merged
  with `limit` on the same model entry; planner and critic configs never carry it, and
  "default" (the library default) is the historical config byte for byte;
* record: the arm label gains `+think-off` and the metrics row's `budgets` block names
  the switch;
* wire: opencode 1.18.31 really sends it. `OpencodeMockBody` runs the INSTALLED opencode
  against a local recording HTTP server (no model server anywhere: the provider's
  baseURL is rewritten to the mock in a scratch copy of the global config, under a
  scratch HOME/XDG so the real global config -- and :8083 -- are unreachable) and
  asserts the kwarg in the captured POST body of the author call, and its absence in the
  planner's.
"""
from __future__ import annotations

import argparse
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unittest

from autokernel.loop import actor_opencode_config as aoc, actors

MODEL = "qwen-gpu/qwen3.8-27b"
KWARG = {"enable_thinking": False}
SCHEMA = "https://opencode.ai/config.json"
REAL_GLOBAL = Path.home() / ".config" / "opencode" / "opencode.jsonc"
OPENCODE = shutil.which("opencode")
#: Scratch space for the wire test (the host blocks /tmp for some sessions).
SCRATCH_BASE = Path("/mnt/raid0/llm/tmp")


def _model_entry(config: dict) -> dict:
    return config.get("provider", {}).get("qwen-gpu", {}).get("models", {}).get(
        "qwen3.8-27b", {})


class AuthorThinkingConfig(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()
        self.backend = actors.backend_for(MODEL, "high")

    def tearDown(self):
        self._tmp.cleanup()

    def test_model_thinking_block(self):
        self.assertEqual(aoc.model_thinking(MODEL, "default"), {})
        self.assertEqual(aoc.model_thinking(MODEL, "off"),
                         {"provider": {"qwen-gpu": {"models": {"qwen3.8-27b": {
                             "options": {"chat_template_kwargs": KWARG}}}}}})
        with self.assertRaises(ValueError):
            aoc.model_thinking(MODEL, "on")
        with self.assertRaises(ValueError):
            aoc.model_thinking("no-provider", "off")
        # A shared constant is never aliased into a config.
        block = aoc.model_thinking(MODEL, "off")
        _model_entry(block)["options"]["chat_template_kwargs"]["enable_thinking"] = True
        self.assertIs(aoc.THINKING_OFF_OPTIONS["chat_template_kwargs"]["enable_thinking"],
                      False)

    def test_limits_and_thinking_share_the_model_entry(self):
        for config in (
                aoc.build_plain_config(role="author", lane=self.ws, model=MODEL,
                                       context_limit=131072, output_limit=32768,
                                       thinking="off"),
                aoc.build_actor_config(role="author", lane=self.ws, model=MODEL,
                                       instructions_path=self.ws / "i.md",
                                       context_limit=131072, output_limit=32768,
                                       thinking="off")):
            self.assertEqual(config["provider"], {"qwen-gpu": {"models": {"qwen3.8-27b": {
                "limit": {"context": 131072, "output": 32768},
                "options": {"chat_template_kwargs": KWARG}}}}})

    def test_default_is_byte_identical(self):
        for kw in ({}, {"context_limit": 131072, "output_limit": 8192}):
            with self.subTest(**kw):
                self.assertEqual(
                    json.dumps(aoc.build_plain_config(role="author", lane=self.ws,
                                                      model=MODEL, lane_guard=True, **kw)),
                    json.dumps(aoc.build_plain_config(role="author", lane=self.ws,
                                                      model=MODEL, lane_guard=True,
                                                      thinking="default", **kw)))
                self.assertEqual(
                    json.dumps(aoc.build_actor_config(role="author", lane=self.ws,
                                                      model=MODEL,
                                                      instructions_path=self.ws / "i.md",
                                                      **kw)),
                    json.dumps(aoc.build_actor_config(role="author", lane=self.ws,
                                                      model=MODEL,
                                                      instructions_path=self.ws / "i.md",
                                                      thinking="default", **kw)))
        self.assertEqual(aoc.seat_label("plain"), "plain")
        self.assertEqual(aoc.seat_label("plain", thinking_off=False), "plain")

    def test_label(self):
        self.assertEqual(aoc.seat_label("bounded", context_limit=131072, output_limit=32768,
                                        thinking_off=True, concise=True),
                         "bounded+ctx128k+out32k+think-off+concise")

    # -- seat wiring: what `_seat_call` / `_seated` write (no call is run) -------------

    def _plain(self, role: str, seat: actors.ActorSeat) -> tuple[dict, dict]:
        env = actors._seat_call(seat, self.backend, role, self.ws, {})
        return env, json.loads(Path(env["OPENCODE_CONFIG"]).read_text())

    def test_plain_author_carries_it_planner_and_critic_do_not(self):
        seat = actors.ActorSeat(bounded=False, author_thinking="off", context_limit=131072,
                                output_limit=8192, author_output_limit=32768)
        env, config = self._plain("author", seat)
        self.assertEqual(_model_entry(config)["options"], {"chat_template_kwargs": KWARG})
        self.assertEqual(env[actors.SEAT_ENV_ARM], "plain+ctx128k+out32k+think-off")
        self.assertEqual(json.loads(env[actors.SEAT_ENV_BUDGETS])["thinking"], "off")
        for role in ("planner", "critic"):
            with self.subTest(role=role):
                env, config = self._plain(role, seat)
                self.assertNotIn("options", _model_entry(config))
                self.assertNotIn("think", env[actors.SEAT_ENV_ARM])
                self.assertNotIn("thinking", json.loads(env[actors.SEAT_ENV_BUDGETS]))

    def test_bounded_author_carries_it_planner_does_not(self):
        planner = actors.AgentPlanner(
            workspace=self.ws, backend=self.backend,
            seat=actors.ActorSeat(bounded=True, author_thinking="off"))
        _backend, env = planner._seated("author", {})
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        self.assertEqual(_model_entry(config), {"options": {"chat_template_kwargs": KWARG}})
        self.assertEqual(env[actors.SEAT_ENV_ARM], "bounded+think-off")
        _backend, env = planner._seated("planner", {})
        self.assertNotIn("provider", json.loads(Path(env["OPENCODE_CONFIG"]).read_text()))
        self.assertEqual(env[actors.SEAT_ENV_ARM], "bounded")

    def test_knobs_off_author_is_the_historical_call(self):
        env, config = self._plain("author", actors.ActorSeat(bounded=False))
        self.assertEqual(config, {"$schema": SCHEMA, "snapshot": False})
        self.assertNotIn(actors.SEAT_ENV_ARM, env)
        self.assertNotIn(actors.SEAT_ENV_BUDGETS, env)

    def test_thinking_alone_reaches_the_metrics_row(self):
        seat = actors.ActorSeat(bounded=False, author_thinking="off")
        env, _config = self._plain("author", seat)
        self.assertEqual(json.loads(env[actors.SEAT_ENV_BUDGETS]),
                         {"concise": False, "context_limit": 0, "output_limit": 0,
                          "thinking": "off"})
        block = actors._budgets_of(env, None, False, None)
        self.assertEqual(block["thinking"], "off")
        limits_only = {actors.SEAT_ENV_BUDGETS: json.dumps({"context_limit": 131072})}
        self.assertEqual(actors._budgets_of(limits_only, None, False, None)["thinking"],
                         "default")

    def test_cli_knob(self):
        from autokernel.loop import run
        # Operator 2026-09-26: the default moved to "medium" (test_actor_author_medium).
        self.assertEqual(aoc.DEFAULT_AUTHOR_THINKING, "medium")
        self.assertEqual(run._actor_thinking(argparse.Namespace(actor_author_thinking="off")),
                         {"author_thinking": "off", "author_action_rule": False})
        source = Path(run.__file__).read_text(encoding="utf-8")
        block = source[source.index('"--actor-author-thinking"'):][:300]
        self.assertIn("DEFAULT_AUTHOR_THINKING", block)
        self.assertIn("THINKING_CHOICES", block)
        # The planner/author seat takes it; the critic's seat construction does not.
        critic = source[source.index("make_critic=lambda worker"):][:600]
        self.assertNotIn("_actor_thinking", critic)
        planner = source[source.index("def make_planner(worker)"):][:1400]
        self.assertIn("**_actor_thinking(args)", planner)


# -------------------------------------------------------------------------------------
# The wire: installed opencode -> recording mock server.
# -------------------------------------------------------------------------------------

class _Recorder(BaseHTTPRequestHandler):
    """Records every request; answers chat completions with a minimal SSE stream (or a
    JSON completion when stream is false). Serves nothing else."""
    bodies: list = []

    def log_message(self, *_a):   # quiet
        pass

    def _send(self, code: int, ctype: str, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        type(self).bodies.append({"method": "GET", "path": self.path})
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, "application/json", json.dumps(
                {"object": "list", "data": [{"id": "qwen3.8-27b", "object": "model"}]}).encode())
        else:
            self._send(404, "application/json", b"{}")

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"_raw": raw.decode("utf-8", "replace")}
        type(self).bodies.append({"method": "POST", "path": self.path, "body": body})
        if not self.path.endswith("/chat/completions"):
            self._send(404, "application/json", b"{}")
            return
        base = {"id": "chatcmpl-mock", "created": int(time.time()), "model": "qwen3.8-27b"}
        usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        if not body.get("stream"):
            self._send(200, "application/json", json.dumps({
                **base, "object": "chat.completion", "usage": usage,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}]}).encode())
            return
        chunks = [
            {**base, "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"},
                          "finish_reason": None}]},
            {**base, "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
        ]
        payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self._send(200, "text/event-stream", payload.encode())


def _jsonc_text_rebased(text: str, url: str) -> str:
    """The global config with qwen-gpu's baseURL pointed at the mock; refuses unless the
    :8083 URL occurs exactly once and no 8083 remains."""
    real = "http://127.0.0.1:8083/v1"
    if text.count(real) != 1:
        raise AssertionError(f"expected exactly one {real} in the global config")
    out = text.replace(real, url)
    live = "\n".join(line for line in out.splitlines() if not line.strip().startswith("//"))
    if ":8083" in live:   # comments may mention the port; no config value may
        raise AssertionError(":8083 still present after rebasing the global config")
    return out


def run_opencode_against_mock(role: str, seat: actors.ActorSeat, *, timeout_s: int = 120
                              ) -> tuple[list, dict, str]:
    """Run the installed opencode once for `role`'s seat against a recording mock.

    Returns (recorded requests, the per-call config, opencode stderr tail). No model
    server is involved: HOME and every XDG dir are scratch, the scratch global config is
    the host's with qwen-gpu's baseURL rewritten to the mock, and OPENCODE_CONFIG is the
    per-call config the seat wrote -- exactly what an actor call would load."""
    base = SCRATCH_BASE if SCRATCH_BASE.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="ak-authorthink-mock-", dir=base) as tmp:
        tmp = Path(tmp)
        _Recorder.bodies = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
        port = server.server_address[1]
        if port == 8083:
            raise AssertionError("mock bound :8083")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            home = tmp / "home"
            (home / ".config" / "opencode").mkdir(parents=True)
            (home / ".config" / "opencode" / "opencode.jsonc").write_text(
                _jsonc_text_rebased(REAL_GLOBAL.read_text(encoding="utf-8"),
                                    f"http://127.0.0.1:{port}/v1"), encoding="utf-8")
            lane = tmp / "work" / "lane"
            lane.mkdir(parents=True)
            subprocess.run(["git", "init", "-q", str(lane)], check=True, timeout=30)
            backend = dataclasses.replace(actors.backend_for(MODEL, "high"), binary=OPENCODE)
            if seat.bounded:
                planner = actors.AgentPlanner(workspace=lane, backend=backend, seat=seat)
                backend, extra = planner._seated(role, {})
            else:
                extra = actors._seat_call(seat, backend, role, lane, {})
            config = json.loads(Path(extra["OPENCODE_CONFIG"]).read_text())
            if "8083" in json.dumps(config):
                raise AssertionError("per-call config names :8083")
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("OPENCODE_") and not k.startswith("XDG_")}
            env.update({"HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
                        "XDG_DATA_HOME": str(home / ".local" / "share"),
                        "XDG_STATE_HOME": str(home / ".local" / "state"),
                        "XDG_CACHE_HOME": str(home / ".cache"),
                        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
                        **extra})
            proc = subprocess.Popen(backend.argv("", lane, read_only=(role != "author")),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, cwd=str(lane),
                                    start_new_session=True, text=True)
            try:
                _out, err = proc.communicate("Reply with the single word ok.",
                                             timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    _out, err = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    _out, err = proc.communicate()
            finally:
                try:   # nothing of our own session may outlive the call (MCP child etc.)
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return list(_Recorder.bodies), config, (err or "")[-2000:]
        finally:
            server.shutdown()
            server.server_close()


def _chat_bodies(records: list) -> list[dict]:
    return [r["body"] for r in records
            if r.get("method") == "POST" and r["path"].endswith("/chat/completions")]


@unittest.skipUnless(OPENCODE and REAL_GLOBAL.is_file()
                     and "http://127.0.0.1:8083/v1" in REAL_GLOBAL.read_text(errors="replace"),
                     "needs the installed opencode and the host's qwen-gpu global config")
class OpencodeMockBody(unittest.TestCase):

    def _assert_bodies(self, role: str, seat: actors.ActorSeat, expect: bool):
        records, _config, err = run_opencode_against_mock(role, seat)
        bodies = _chat_bodies(records)
        self.assertTrue(bodies, f"opencode sent no chat completion; stderr: {err}")
        for body in bodies:
            self.assertEqual(body.get("model"), "qwen3.8-27b")
            if expect:
                self.assertEqual(body.get("chat_template_kwargs"), KWARG)
            else:
                self.assertNotIn("chat_template_kwargs", body)
        return bodies

    def test_plain_author_body_carries_the_kwarg(self):
        bodies = self._assert_bodies("author", actors.ActorSeat(
            bounded=False, author_thinking="off", context_limit=131072,
            author_output_limit=32768, output_limit=8192), True)
        # The limit rides the same model entry; above opencode's 32000 ceiling the call's
        # env raises it (OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX), so 32768 reaches the wire.
        self.assertTrue(any(b.get("max_tokens") == 32768 for b in bodies))

    def test_plain_planner_body_does_not(self):
        self._assert_bodies("planner", actors.ActorSeat(
            bounded=False, author_thinking="off", context_limit=131072,
            planner_output_limit=16384, output_limit=8192), False)

    def test_bounded_author_body_carries_the_kwarg(self):
        self._assert_bodies("author", actors.ActorSeat(
            bounded=True, fan_out=False, author_thinking="off"), True)


if __name__ == "__main__":   # evidence dump: python3 -m autokernel.loop.test_actor_author_thinking
    import sys
    role = sys.argv[1] if len(sys.argv) > 1 else "author"
    records, config, err = run_opencode_against_mock(role, actors.ActorSeat(
        bounded=False, author_thinking="off", context_limit=131072,
        planner_output_limit=16384, author_output_limit=32768, output_limit=8192))
    print("PER-CALL CONFIG:", json.dumps(config, indent=2))
    for body in _chat_bodies(records):
        slim = {k: v for k, v in body.items() if k not in ("messages", "tools")}
        slim["messages"] = f"<{len(body.get('messages') or [])} messages>"
        slim["tools"] = f"<{len(body.get('tools') or [])} tools>"
        print("CAPTURED POST BODY:", json.dumps(slim, indent=2))
    print("STDERR TAIL:", err[-600:])
