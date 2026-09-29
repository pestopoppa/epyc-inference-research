"""The PLANNER's reasoning history (`--actor-planner-reasoning-history {keep,drop}`).

2026-09-29: DS41 planner sessions decode 24-182k chars of reasoning, and opencode 1.18.31
re-sends every earlier step's reasoning (`reasoning_content` on each assistant history
message) in every later request of the call; the served Qwen template renders it inside
the tool chain whatever `preserve_thinking` says. "drop" moves it to a message key
llama-server ignores via the model's `interleaved` field (see
`actor_opencode_config.model_reasoning_history`).

What must hold:

* config: "keep" (the default everywhere) is the historical config byte for byte; "drop"
  adds `interleaved: {field: REASONING_DROP_FIELD}` to the call's model entry, merged
  with `limit`; llama-server providers only;
* seat: only the PLANNER's call carries it (never the author or the critic), the arm label
  gains `+reason-drop`, the budgets block names it, and an overridden lane
  (`--lane-actor-models`, DeepSeek) keeps its history;
* run.py: the flag defaults to keep, reaches the proposal planner's seat only, and refuses
  a planner that is not on a llama-server provider;
* wire: the INSTALLED opencode, run against a scripted mock model whose every step streams
  reasoning then a tool call, sends the earlier steps' reasoning as `reasoning_content`
  under keep, and under drop sends none of it as `reasoning_content` (the text rides
  REASONING_DROP_FIELD instead), with the rest of each history message unchanged.
"""
from __future__ import annotations

import argparse
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
from types import SimpleNamespace
import unittest

from autokernel.loop import actor_opencode_config as aoc, actors, lane_actors
from autokernel.loop.test_actor_author_thinking import (MODEL, OPENCODE, REAL_GLOBAL,
                                                        SCRATCH_BASE, _jsonc_text_rebased)

DEEPSEEK = "deepseek/deepseek-flash"
DROP_ENTRY = {"interleaved": {"field": aoc.REASONING_DROP_FIELD}}


def _model_entry(config: dict) -> dict:
    return config.get("provider", {}).get("qwen-gpu", {}).get("models", {}).get(
        "qwen3.8-27b", {})


class ReasoningHistoryConfig(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "lane"
        self.ws.mkdir()
        self.backend = actors.backend_for(MODEL, "high")

    def tearDown(self):
        self._tmp.cleanup()

    def test_block(self):
        self.assertEqual(aoc.DEFAULT_PLANNER_REASONING_HISTORY, "keep")
        self.assertEqual(aoc.model_reasoning_history(MODEL, "keep"), {})
        self.assertEqual(aoc.model_reasoning_history(None, "keep"), {})
        self.assertEqual(aoc.model_reasoning_history(MODEL, "drop"),
                         {"provider": {"qwen-gpu": {"models": {"qwen3.8-27b": DROP_ENTRY}}}})
        self.assertEqual(aoc.model_reasoning_history("qwen-local/qwen3.8-flash-next", "drop")
                         ["provider"]["qwen-local"]["models"]["qwen3.8-flash-next"], DROP_ENTRY)
        for bad in (("drop", DEEPSEEK), ("drop", "no-provider"), ("drop", None),
                    ("off", MODEL)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                aoc.model_reasoning_history(bad[1], bad[0])
        # Never a key llama-server reads from an input message.
        self.assertNotIn(aoc.REASONING_DROP_FIELD, ("role", "content", "tool_calls",
                                                    "reasoning_content", "name",
                                                    "tool_call_id"))

    def test_merges_with_limits_and_thinking_on_the_model_entry(self):
        for config in (
                aoc.build_plain_config(role="planner", lane=self.ws, model=MODEL,
                                       context_limit=90112, output_limit=32000,
                                       reasoning_history="drop"),
                aoc.build_actor_config(role="planner", lane=self.ws, model=MODEL,
                                       instructions_path=self.ws / "i.md",
                                       context_limit=90112, output_limit=32000,
                                       reasoning_history="drop")):
            self.assertEqual(config["provider"], {"qwen-gpu": {"models": {"qwen3.8-27b": {
                "limit": {"context": 90112, "output": 32000}, **DROP_ENTRY}}}})
        both = aoc.model_block(MODEL, thinking="medium", reasoning_history="drop")
        self.assertEqual(set(_model_entry(both)), {"options", "interleaved"})

    def test_keep_is_byte_identical(self):
        for kw in ({}, {"context_limit": 180224, "output_limit": 32000},
                   {"thinking": "medium", "lane_guard": True}):
            with self.subTest(**kw):
                for role in ("planner", "critic", "author"):
                    self.assertEqual(
                        json.dumps(aoc.build_plain_config(role=role, lane=self.ws,
                                                          model=MODEL, **kw)),
                        json.dumps(aoc.build_plain_config(role=role, lane=self.ws,
                                                          model=MODEL,
                                                          reasoning_history="keep", **kw)))
                bounded = {k: v for k, v in kw.items()}
                self.assertEqual(
                    json.dumps(aoc.build_actor_config(role="planner", lane=self.ws,
                                                      model=MODEL,
                                                      instructions_path=self.ws / "i.md",
                                                      **bounded)),
                    json.dumps(aoc.build_actor_config(role="planner", lane=self.ws,
                                                      model=MODEL,
                                                      instructions_path=self.ws / "i.md",
                                                      reasoning_history="keep", **bounded)))
        self.assertEqual(aoc.seat_label("plain", reasoning_history="keep"), "plain")
        self.assertEqual(aoc.seat_label("plain", reasoning_history=""), "plain")
        self.assertEqual(aoc.seat_label("plain", context_limit=90112, output_limit=32000,
                                        reasoning_history="drop", budget_s=4500),
                         "plain+ctx88k+out32000+reason-drop+budget4500s")

    # -- seat wiring: what `_seat_call` / `_seated` write (no call is run) -------------

    def _plain(self, role: str, seat: actors.ActorSeat) -> tuple[dict, dict]:
        env = actors._seat_call(seat, self.backend, role, self.ws, {})
        return env, json.loads(Path(env["OPENCODE_CONFIG"]).read_text())

    def test_plain_planner_carries_it_author_and_critic_do_not(self):
        seat = actors.ActorSeat(bounded=False, planner_reasoning_history="drop",
                                context_limit=90112, planner_output_limit=32000,
                                author_output_limit=40960, author_thinking="medium")
        env, config = self._plain("planner", seat)
        self.assertEqual(_model_entry(config), {"limit": {"context": 90112, "output": 32000},
                                                **DROP_ENTRY})
        self.assertEqual(env[actors.SEAT_ENV_ARM], "plain+ctx88k+out32000+reason-drop")
        self.assertEqual(json.loads(env[actors.SEAT_ENV_BUDGETS])["reasoning_history"], "drop")
        block = actors._budgets_of(env, None, False, None)
        self.assertEqual(block["reasoning_history"], "drop")
        for role in ("author", "critic"):
            with self.subTest(role=role):
                env, config = self._plain(role, seat)
                self.assertNotIn("interleaved", _model_entry(config))
                self.assertNotIn("reason", env[actors.SEAT_ENV_ARM])
                self.assertNotIn("reasoning_history", json.loads(env[actors.SEAT_ENV_BUDGETS]))

    def test_bounded_planner_carries_it(self):
        planner = actors.AgentPlanner(
            workspace=self.ws, backend=self.backend,
            seat=actors.ActorSeat(bounded=True, planner_reasoning_history="drop"))
        _backend, env = planner._seated("planner", {})
        self.assertEqual(_model_entry(json.loads(Path(env["OPENCODE_CONFIG"]).read_text())),
                         DROP_ENTRY)
        self.assertEqual(env[actors.SEAT_ENV_ARM], "bounded+reason-drop")
        _backend, env = planner._seated("author", {})
        self.assertNotIn("provider", json.loads(Path(env["OPENCODE_CONFIG"]).read_text()))

    def test_keep_seat_is_the_historical_call(self):
        for seat in (actors.ActorSeat(bounded=False),
                     actors.ActorSeat(bounded=False, planner_reasoning_history="keep")):
            env, config = self._plain("planner", seat)
            self.assertEqual(config, {"$schema": "https://opencode.ai/config.json",
                                      "snapshot": False})
            self.assertNotIn(actors.SEAT_ENV_ARM, env)
            self.assertNotIn(actors.SEAT_ENV_BUDGETS, env)
        # Rows of calls that did not apply it carry no key (historical rows unchanged).
        limits_only = {actors.SEAT_ENV_BUDGETS: json.dumps({"context_limit": 180224})}
        self.assertNotIn("reasoning_history", actors._budgets_of(limits_only, None, False, None))

    def test_overridden_lane_keeps_its_history(self):
        seat = actors.ActorSeat(planner_reasoning_history="drop", author_thinking="medium",
                                context_limit=90112)
        self.assertIs(lane_actors.seat_for(None, seat), seat)
        moved = lane_actors.seat_for(lane_actors.LaneActor(1, DEEPSEEK), seat)
        self.assertEqual(moved.reasoning_history_for("planner"), "keep")
        self.assertEqual(moved.limits_for("planner"), seat.limits_for("planner"))
        config = aoc.build_plain_config(role="planner", lane=self.ws, model=DEEPSEEK,
                                        reasoning_history=moved.reasoning_history_for("planner"),
                                        **moved.limits_for("planner"))
        self.assertNotIn("interleaved", json.dumps(config))


class RunFlag(unittest.TestCase):

    def test_flag_default_and_seat_fields(self):
        from autokernel.loop import run
        self.assertEqual(run._actor_reasoning_history(argparse.Namespace()),
                         {"planner_reasoning_history": "keep"})
        self.assertEqual(run._actor_reasoning_history(argparse.Namespace(
            actor_planner_reasoning_history="drop")), {"planner_reasoning_history": "drop"})
        source = Path(run.__file__).read_text(encoding="utf-8")
        block = source[source.index('"--actor-planner-reasoning-history"'):][:300]
        self.assertIn("DEFAULT_PLANNER_REASONING_HISTORY", block)
        self.assertIn("REASONING_HISTORY_CHOICES", block)
        # The proposal planner's seat takes it; the critic and panel members do not.
        planner = source[source.index("def make_planner(worker)"):][:1600]
        self.assertIn("**_actor_reasoning_history(args)", planner)
        critic = source[source.index("make_critic=lambda worker"):][:600]
        self.assertNotIn("_actor_reasoning_history", critic)
        panel = source[source.index("def make_author(spec, workspace, member_stop)"):][:1800]
        self.assertNotIn("_actor_reasoning_history", panel)

    def test_refuses_a_planner_off_llama_server(self):
        from autokernel.loop import run
        ok = SimpleNamespace(actor_planner_reasoning_history="drop", planner_model=MODEL)
        self.assertIsNone(run._reasoning_history_error(ok))
        self.assertIsNone(run._reasoning_history_error(SimpleNamespace(
            actor_planner_reasoning_history="keep", planner_model=DEEPSEEK)))
        for model in (DEEPSEEK, "gpt-5.6-sol", "orch:auto", "claude-fable-5"):
            with self.subTest(model=model):
                error = run._reasoning_history_error(SimpleNamespace(
                    actor_planner_reasoning_history="drop", planner_model=model))
                self.assertIn("llama-server", error)

    def test_actor_config_records_it_only_when_dropping(self):
        from autokernel.loop import run
        keep = run._actor_config(argparse.Namespace(planner_model=MODEL,
                                                    actor_planner_reasoning_history="keep"))
        self.assertNotIn("planner_reasoning_history", keep)
        drop = run._actor_config(argparse.Namespace(planner_model=MODEL,
                                                    actor_planner_reasoning_history="drop"))
        self.assertEqual(drop["planner_reasoning_history"], "drop")
        self.assertEqual({k: v for k, v in drop.items()
                          if k != "planner_reasoning_history"}, keep)


#: DS41 lane-0 arm B/C deltas over the run's common args (the planner context arms).
ARM_C = ["--actor-context-limit", "90112", "--actor-planner-reasoning-history", "drop",
         "--actor-authors", "single", "--actor-context-mode", "variable"]


def test_arm_flags_are_not_continuation_identity(tmp_path):
    """An arm switch at a batch boundary carries the source lineage (POOL_ACTOR_FLAGS)."""
    from autokernel.loop import serial_run
    launch = tmp_path / "launch.json"
    launch.write_text("{}")
    base = ["--worktree", "/w", "--cpu-serving-launch", str(launch), "--workers", "2",
            "--planner-model", MODEL]
    assert serial_run.resume_binding([*base, *ARM_C]) == serial_run.resume_binding(base)
    assert serial_run.resume_binding(
        [*base, "--actor-context-limit=90112"]) == serial_run.resume_binding(base)
    # Other stable inputs still bind.
    assert serial_run.resume_binding([*base, "--pairs", "9"]) != serial_run.resume_binding(base)
    assert serial_run.resume_binding([*base, "--actor-output-limit", "4096"]) != \
        serial_run.resume_binding(base)


def test_serial_common_args_admit_the_arm_flags(tmp_path):
    from autokernel.loop import serial_roster, serial_run
    from autokernel.loop.test_serial_roster import _inputs
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    common = tmp_path / "common.json"
    common.write_text(json.dumps(["--workers", "2", "--lane-actor-models", f"1={DEEPSEEK}",
                                  *ARM_C]))
    targets, _skipped, _cpus = serial_roster.build_targets(
        Path(serial_run.option(argv, "--resolved-campaign")),
        Path(serial_run.option(argv, "--owned-targets")),
        target_root=tmp_path / "targets", common_path=common)
    for flag, value in zip(ARM_C[::2], ARM_C[1::2]):
        assert serial_run.option(targets[0], flag) == value


# -------------------------------------------------------------------------------------
# The wire: installed opencode -> scripted mock model with a reasoning tool chain.
# -------------------------------------------------------------------------------------

#: Tool-calling steps before the final answer; each step streams reasoning first.
CHAIN_STEPS = 3
FINAL = "FINAL-REASONING-HISTORY-OK"


def _reasoning(step: int) -> str:
    return f"REASON-STEP-{step}: the kernel at step {step} needs another look."


class _ReasoningChain(BaseHTTPRequestHandler):
    """A fake OpenAI-compatible model: the n-th agent request (n = tool results so far)
    streams `reasoning_content` then a `read` tool call; after CHAIN_STEPS, reasoning then
    the final text. Records every chat-completion POST body."""
    posts: list = []
    target: str = ""

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
        base = {"id": "chatcmpl-chain", "created": int(time.time()), "model": "qwen3.8-27b",
                "object": "chat.completion.chunk"}
        chunks = [{**base, "choices": [{"index": 0, "delta": {
            "role": "assistant", "reasoning_content": _reasoning(done)},
            "finish_reason": None}]}]
        if body.get("tools") and done < CHAIN_STEPS:
            chunks.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": f"call_{done}", "type": "function",
                "function": {"name": "read", "arguments": json.dumps(
                    {"filePath": type(self).target})}}]}, "finish_reason": None}]})
            finish = "tool_calls"
        else:
            chunks.append({**base, "choices": [{"index": 0, "delta": {"content": FINAL},
                                                "finish_reason": None}]})
            finish = "stop"
        chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 1,
                                 "total_tokens": 11}})
        payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self._send(200, "text/event-stream", payload.encode())


def run_planner_chain(seat: actors.ActorSeat, *, timeout_s: int = 120
                      ) -> tuple[list, dict, str, str]:
    """Run the installed opencode once as the PLAIN planner seat is run, against
    `_ReasoningChain`. Returns (tool-bearing POST bodies with the scratch root as
    `@TMP@`, the per-call config, stdout, stderr tail). No model server is involved: HOME and every XDG dir are scratch, and
    the scratch global config is the host's with qwen-gpu's baseURL rewritten to the mock."""
    base = SCRATCH_BASE if SCRATCH_BASE.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="ak-reasonhist-mock-", dir=base) as tmp:
        tmp = Path(tmp)
        lane = tmp / "work" / "lane"
        lane.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(lane)], check=True, timeout=30)
        (lane / "kernel.c").write_text("int k(void) { return 1; }\n")
        _ReasoningChain.posts = []
        _ReasoningChain.target = str(lane / "kernel.c")
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ReasoningChain)
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
            extra = actors._seat_call(seat, backend, "planner", lane, {})
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
            proc = subprocess.Popen(backend.argv("", lane, read_only=True),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, cwd=str(lane),
                                    start_new_session=True, text=True)
            try:
                out, err = proc.communicate("Investigate the kernel.", timeout=timeout_s)
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
            # The scratch root differs per run: name it @TMP@ so two runs compare.
            bodies = [json.loads(json.dumps(p).replace(str(tmp), "@TMP@"))
                      for p in _ReasoningChain.posts if p.get("tools")]
            return bodies, config, out or "", (err or "")[-2000:]
        finally:
            server.shutdown()
            server.server_close()


def _assistant_history(body: dict) -> list[dict]:
    return [m for m in body.get("messages", []) if m.get("role") == "assistant"]


@unittest.skipUnless(OPENCODE and REAL_GLOBAL.is_file()
                     and "http://127.0.0.1:8083/v1" in REAL_GLOBAL.read_text(errors="replace"),
                     "needs the installed opencode and the host's qwen-gpu global config")
class OpencodeReasoningHistoryWire(unittest.TestCase):

    SEAT = dict(bounded=False, context_limit=90112, planner_output_limit=32000)

    def _chain(self, history: str) -> list[dict]:
        bodies, config, out, err = run_planner_chain(actors.ActorSeat(
            **self.SEAT, planner_reasoning_history=history))
        self.assertIn(FINAL, out, f"chain did not finish; stderr: {err}")
        # One request per step: CHAIN_STEPS tool steps plus the final answer.
        self.assertEqual(len(bodies), CHAIN_STEPS + 1, err)
        self.assertEqual([len(_assistant_history(b)) for b in bodies],
                         list(range(CHAIN_STEPS + 1)))
        self.assertEqual("interleaved" in _model_entry(config), history == "drop")
        return bodies

    def test_keep_resends_every_earlier_steps_reasoning(self):
        """The evidence: request N carries steps 0..N-1's reasoning as reasoning_content."""
        for body in self._chain("keep"):
            history = _assistant_history(body)
            self.assertEqual([m.get("reasoning_content") for m in history],
                             [_reasoning(step) for step in range(len(history))])
            self.assertFalse(any(aoc.REASONING_DROP_FIELD in m for m in history))

    def test_drop_sends_no_earlier_reasoning_as_reasoning_content(self):
        keep = self._chain("keep")
        drop = self._chain("drop")
        for kept, dropped in zip(keep, drop):
            history = _assistant_history(dropped)
            self.assertFalse(any("reasoning_content" in m for m in history))
            self.assertEqual([m.get(aoc.REASONING_DROP_FIELD) for m in history],
                             [_reasoning(step) for step in range(len(history))])
            # Everything else in each history message is what keep sends.
            strip = lambda m: {k: v for k, v in m.items()  # noqa: E731
                               if k not in ("reasoning_content", aoc.REASONING_DROP_FIELD)}
            self.assertEqual([strip(m) for m in history],
                             [strip(m) for m in _assistant_history(kept)])
            self.assertEqual({k: v for k, v in dropped.items() if k != "messages"},
                             {k: v for k, v in kept.items() if k != "messages"})
            others = lambda b: [m for m in b["messages"]  # noqa: E731
                                if m.get("role") != "assistant"]
            self.assertEqual(others(dropped), others(kept))


if __name__ == "__main__":   # evidence dump: python3 -m autokernel.loop.test_actor_reasoning_history
    import sys
    history = sys.argv[1] if len(sys.argv) > 1 else "keep"
    bodies, config, out, err = run_planner_chain(actors.ActorSeat(
        bounded=False, context_limit=90112, planner_output_limit=32000,
        planner_reasoning_history=history))
    print("PER-CALL CONFIG:", json.dumps(config, indent=2))
    for index, body in enumerate(bodies):
        print(f"REQUEST {index}:", json.dumps(
            [{k: v for k, v in m.items() if k != "tool_calls"}
             for m in _assistant_history(body)]))
    print("STDOUT:", out[-200:])
