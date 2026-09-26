"""The read-only critic can never lose its opencode session to a permission decision.

DS41 run 10i (2026-09-26): the critic (opencode, run WITHOUT `--auto` because it is
read-only) ran `cd <campaign root> && rg ... .`; opencode asked external_directory,
auto-rejected it, and the rejection ENDED the session with no final text -- an empty
reply, critic pass 1 lost (run 10h lost one the same way on `<store>/experiments.md`).

Measured against the installed opencode 1.18.31 with a scripted fake model (this file's
`OpencodeCriticWire`): without `--auto` an `ask` is auto-rejected and ends the session;
a configured `deny` fails only that tool call and the model's next request -- and its
final text -- still arrive (with or without `--auto`). So the critic's per-call config
leaves nothing to ask (`actor_opencode_config.CRITIC_NEVER_ASK`): an external_directory
catch-all deny ahead of its allows, `.env` reads denied, doom_loop and question denied,
and its read-only fence stated as denies (edit, mutating bash).

What must hold:

* config: the critic block, knobs off and on, has no `ask` anywhere, puts the
  external_directory catch-all FIRST and the anchor-build / read-root allows after it,
  and denies edit and every READ_ONLY_DENY verb; planner and author blocks are the
  historical ones;
* wire (installed opencode, fake model, scratch HOME, no model server): a critic tool
  call to an outside path, a bash `cd <outside> && grep`, a `.env` read and a doom loop
  each leave the session alive (the final text arrives); a read under a read root is
  served; a write, an edit-by-bash and an out-of-lane `touch` are refused and change
  nothing; and the control (the pre-fix config, no catch-all) really does end the
  session, so the harness can tell the two apart.
"""
from __future__ import annotations

import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import unittest

from autokernel.loop import actor_opencode_config as aoc, actors
from autokernel.loop.test_actor_author_thinking import (
    OPENCODE, REAL_GLOBAL, SCRATCH_BASE, _jsonc_text_rebased)

MODEL = "qwen-gpu/qwen3.8-27b"
FINAL = "CRITIC-FINAL-TEXT"
GUARDED = dict(bounded=False, trim_instructions=True, trim_tools=True, lane_guard=True)


def _values(node) -> list:
    if isinstance(node, dict):
        return [v for value in node.values() for v in _values(value)]
    return [node]


class CriticConfig(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="ak-critic-perm-")
        tmp = Path(self._tmp.name)
        self.lane = tmp / "lane"
        self.lane.mkdir()
        self.anchor = tmp / "anchor"
        (self.anchor / ".git").mkdir(parents=True)
        (self.anchor / "build").mkdir()
        (self.anchor / "src").mkdir()
        self.store = tmp / "campaign" / "store"

    def tearDown(self):
        self._tmp.cleanup()

    def _critic(self, **kw) -> dict:
        return aoc.seat_permission("critic", lane=self.lane, **kw)

    def test_knobs_off_critic_never_asks(self):
        perm = self._critic()
        self.assertNotIn("ask", _values(perm))
        self.assertEqual(perm["external_directory"], {"*": "deny"})
        self.assertEqual(perm["edit"], "deny")
        self.assertEqual((perm["doom_loop"], perm["question"]), ("deny", "deny"))
        self.assertEqual(perm["read"], aoc.CRITIC_READ_RULES)
        for pattern in aoc.READ_ONLY_DENY:
            self.assertEqual(perm["bash"][pattern], "deny", pattern)

    def test_guarded_critic_catch_all_first_allows_after(self):
        perm = self._critic(build_dir=self.anchor / "build", lane_guard=True,
                            trim_tools=True, read_roots=(self.store, self.store.parent))
        self.assertNotIn("ask", _values(perm))
        external = list(perm["external_directory"].items())
        self.assertEqual(external[0], ("*", "deny"))
        self.assertEqual(external[1], (f"{self.anchor}/*", "deny"))
        self.assertIn((f"{self.anchor / 'build'}/*", "allow"), external)
        self.assertEqual(external[-2:], [(f"{self.store}/*", "allow"),
                                         (f"{self.store.parent}/*", "allow")])
        self.assertEqual(perm["edit"], "deny")
        for pattern in (*aoc.READ_ONLY_DENY, *aoc.BUILD_DENY):
            self.assertEqual(perm["bash"][pattern], "deny", pattern)

    def test_planner_and_author_blocks_unchanged(self):
        for role in ("planner", "author"):
            self.assertEqual(aoc.seat_permission(role, lane=self.lane), {}, role)
            perm = aoc.seat_permission(role, lane=self.lane, build_dir=self.anchor / "build",
                                       lane_guard=True, trim_tools=True)
            for key in ("read", "doom_loop"):
                self.assertNotIn(key, perm, role)
            self.assertNotIn("*", perm["external_directory"], role)
            self.assertEqual(next(iter(perm["external_directory"])), f"{self.anchor}/*")
        self.assertEqual(aoc.seat_permission("author", lane=self.lane, lane_guard=True,
                                             build_dir=self.anchor / "build")["edit"],
                         aoc.AUTHOR_EDIT_GUARD)

    def test_seat_call_writes_it_for_the_critic_only(self):
        backend = actors.backend_for(MODEL, "high")
        context = {"actor_read_roots": [str(self.store)]}
        for seat in (None, actors.ActorSeat(**GUARDED)):
            critic = json.loads(Path(actors._seat_call(
                seat, backend, "critic", self.lane, context)["OPENCODE_CONFIG"]).read_text())
            self.assertEqual(next(iter(critic["permission"]["external_directory"])), "*")
            self.assertIn(f"{self.store}/*", critic["permission"]["external_directory"])
            planner = json.loads(Path(actors._seat_call(
                seat, backend, "planner", self.lane, context)["OPENCODE_CONFIG"]).read_text())
            self.assertNotIn("*", planner.get("permission", {}).get("external_directory", {}))


# -------------------------------------------------------------------------------------
# The wire: the installed opencode, run as the critic is run, against a scripted model.
# -------------------------------------------------------------------------------------

class _Scripted(BaseHTTPRequestHandler):
    """A fake OpenAI-compatible model: the n-th agent request (n = tool results so far)
    answers `script[n]` as a tool call, and once the script is spent the final text."""
    script: list = []
    posts: list = []

    def log_message(self, *_a):
        pass

    def _send(self, code: int, ctype: str, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._send(200, "application/json", json.dumps(
            {"object": "list", "data": [{"id": "qwen3.8-27b", "object": "model"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if not self.path.endswith("/chat/completions"):
            self._send(404, "application/json", b"{}")
            return
        type(self).posts.append(body)
        done = sum(1 for m in body.get("messages", []) if m.get("role") == "tool")
        base = {"id": "chatcmpl-scripted", "created": int(time.time()),
                "model": "qwen3.8-27b", "object": "chat.completion.chunk"}
        usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        if body.get("tools") and done < len(type(self).script):
            name, args = type(self).script[done]
            delta = {"role": "assistant", "tool_calls": [{
                "index": 0, "id": f"call_{done}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        else:
            delta, finish = {"role": "assistant", "content": FINAL}, "stop"
        chunks = [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                  {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                   "usage": usage}]
        payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self._send(200, "text/event-stream", payload.encode())


@dataclasses.dataclass
class Outcome:
    stdout: str
    stderr: str
    tool_results: list
    lane: Path
    outside: Path
    config: dict

    @property
    def alive(self) -> bool:
        return FINAL in self.stdout


def run_critic(script, *, seat: actors.ActorSeat | None, permission_override=None,
               timeout_s: int = 120) -> Outcome:
    """Run the installed opencode once exactly as `AgentCritic` does (plain seat config
    from `_seat_call`, `backend.argv(read_only=True)`: no `--auto`) against `_Scripted`.
    `@LANE@`, `@OUT@`, `@STORE@` in the script are the scratch lane, an outside dir and
    the context's read root. `permission_override` replaces the written config's
    permission block (the pre-fix control)."""
    base = SCRATCH_BASE if SCRATCH_BASE.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="ak-critic-wire-", dir=base) as tmp:
        tmp = Path(tmp)
        lane = tmp / "work" / "lane"
        lane.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(lane)], check=True, timeout=30)
        (lane / "kernel.c").write_text("int k(void) { return 1; }\n")
        (lane / ".env").write_text("TOKEN=lane-secret\n")
        outside = tmp / "outside"
        outside.mkdir()
        (outside / "notes.txt").write_text("outside-content\n")
        store = tmp / "campaign" / "store"
        store.mkdir(parents=True)
        (store / "experiments.md").write_text("store-content\n")
        subs = {"@LANE@": str(lane), "@OUT@": str(outside), "@STORE@": str(store)}

        def sub(value):
            text = json.dumps(value)
            for key, repl in subs.items():
                text = text.replace(key, repl)
            return json.loads(text)

        _Scripted.script = [(name, sub(args)) for name, args in script]
        _Scripted.posts = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Scripted)
        port = server.server_address[1]
        if port == 8083:
            raise AssertionError("mock bound :8083")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            home = tmp / "home"
            (home / ".config" / "opencode").mkdir(parents=True)
            (home / ".config" / "opencode" / "opencode.jsonc").write_text(
                _jsonc_text_rebased(REAL_GLOBAL.read_text(encoding="utf-8"),
                                    f"http://127.0.0.1:{port}/v1"), encoding="utf-8")
            backend = dataclasses.replace(actors.backend_for(MODEL, "high"), binary=OPENCODE)
            context = {"actor_read_roots": [str(store), str(store.parent)]}
            extra = actors._seat_call(seat, backend, "critic", lane, context)
            config_path = Path(extra["OPENCODE_CONFIG"])
            config = json.loads(config_path.read_text())
            if permission_override is not None:
                config["permission"] = sub(permission_override)
                config_path.write_text(json.dumps(config))
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
            argv = backend.argv("", lane, read_only=True)
            if "--auto" in argv:
                raise AssertionError("the critic argv carries --auto")
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, cwd=str(lane),
                                    start_new_session=True, text=True)
            try:
                out, err = proc.communicate("Review the hypothesis.", timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    out, err = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    out, err = proc.communicate()
            finally:
                try:   # nothing of our own session may outlive the call
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            last = max((p for p in _Scripted.posts if p.get("tools")),
                       key=lambda p: len(p.get("messages", [])), default={})
            results = [str(m.get("content")) for m in last.get("messages", [])
                       if m.get("role") == "tool"]
            outcome = Outcome(out or "", (err or "")[-4000:], results, lane, outside, config)
            # Snapshot what the call left behind before the scratch dir goes.
            outcome.lane_files = sorted(p.name for p in lane.iterdir())
            outcome.outside_files = sorted(p.name for p in outside.iterdir())
            outcome.kernel_text = (lane / "kernel.c").read_text()
            return outcome
        finally:
            server.shutdown()
            server.server_close()


@unittest.skipUnless(OPENCODE and REAL_GLOBAL.is_file()
                     and "http://127.0.0.1:8083/v1" in REAL_GLOBAL.read_text(errors="replace"),
                     "needs the installed opencode and the host's qwen-gpu global config")
class OpencodeCriticWire(unittest.TestCase):
    """No model server anywhere: HOME/XDG are scratch, the provider points at the mock."""

    OUTSIDE_READ = ("read", {"filePath": "@OUT@/notes.txt"})
    OUTSIDE_CD = ("bash", {"command": "cd @OUT@ && grep -rn content .",
                           "description": "grep the campaign"})

    def _alive(self, outcome: Outcome, n_results: int):
        self.assertTrue(outcome.alive, f"session ended without the final text; stderr: "
                                       f"{outcome.stderr}")
        self.assertEqual(len(outcome.tool_results), n_results, outcome.tool_results)
        self.assertNotIn("rejected permission", outcome.stderr)

    def test_control_pre_fix_config_dies_on_the_ask(self):
        # The pre-fix guarded critic: external_directory allows, no catch-all.
        outcome = run_critic([self.OUTSIDE_CD], seat=actors.ActorSeat(**GUARDED),
                             permission_override={"external_directory": {
                                 "@STORE@/*": "allow"}})
        self.assertFalse(outcome.alive)
        self.assertIn("rejected permission", outcome.stderr)

    def test_outside_paths_are_denied_and_the_session_continues(self):
        for seat in (actors.ActorSeat(**GUARDED), None):
            with self.subTest(seat="guarded" if seat else "knobs-off"):
                outcome = run_critic([self.OUTSIDE_READ, self.OUTSIDE_CD], seat=seat)
                self._alive(outcome, 2)
                for result in outcome.tool_results:
                    self.assertIn("prevents you from using this specific tool call", result)
                    self.assertNotIn("outside-content", result)

    def test_read_roots_are_served(self):
        outcome = run_critic([("read", {"filePath": "@STORE@/experiments.md"}),
                              ("bash", {"command": "cd @STORE@ && cat experiments.md",
                                        "description": "read the store"})],
                             seat=actors.ActorSeat(**GUARDED))
        self._alive(outcome, 2)
        for result in outcome.tool_results:
            self.assertIn("store-content", result)

    def test_env_read_and_doom_loop_do_not_end_the_session(self):
        same = ("read", {"filePath": "@LANE@/kernel.c"})
        outcome = run_critic([("read", {"filePath": "@LANE@/.env"}), same, same, same],
                             seat=actors.ActorSeat(**GUARDED))
        self._alive(outcome, 4)
        self.assertNotIn("lane-secret", outcome.tool_results[0])

    def test_critic_still_cannot_write(self):
        for seat in (actors.ActorSeat(**GUARDED), None):
            with self.subTest(seat="guarded" if seat else "knobs-off"):
                outcome = run_critic([
                    ("write", {"filePath": "@LANE@/new.c", "content": "int x;\n"}),
                    ("bash", {"command": "touch @LANE@/touched.c", "description": "touch"}),
                    ("bash", {"command": "sed -i s/1/2/ @LANE@/kernel.c", "description": "sed"}),
                    ("bash", {"command": "rm @OUT@/notes.txt", "description": "rm"}),
                ], seat=seat)
                self.assertTrue(outcome.alive, outcome.stderr)
                self.assertEqual(outcome.lane_files, [".env", ".git", "kernel.c"])
                self.assertEqual(outcome.outside_files, ["notes.txt"])
                self.assertEqual(outcome.kernel_text, "int k(void) { return 1; }\n")
                self.assertEqual(outcome.config["permission"]["edit"], "deny")


if __name__ == "__main__":
    unittest.main()
