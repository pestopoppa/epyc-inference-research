"""The per-run opencode config for the autokernel actor seat, tested offline.

What must hold: the planner cannot edit and the author can; fan-out is a real A/B knob
(prompt paragraph AND task permission move together); profiles reach the MCP command;
the file lands atomically as valid JSON; and nothing here can re-allow a globally
denied bash verb.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actor_opencode_config as aoc

LANE = Path("/mnt/raid0/llm/tmp/lane-x")


def _agent(cfg, role):
    return cfg["agent"][aoc.AGENT_NAMES[role]]


def build(**kw):
    """v1 path (guidance in the agent prompt): what the prompt-content tests read."""
    kw.setdefault("replace_system_prompt", True)
    return aoc.build_actor_config(**kw)


class Roles(unittest.TestCase):

    def test_the_planner_only_proposes(self):
        cfg = build(role="planner", lane=LANE)
        self.assertEqual(_agent(cfg, "planner")["permission"]["edit"], "deny")
        self.assertIn("cannot edit", _agent(cfg, "planner")["prompt"])

    def test_the_author_may_edit(self):
        cfg = build(role="author", lane=LANE)
        self.assertEqual(_agent(cfg, "author")["permission"]["edit"], "allow")
        self.assertNotIn(aoc.AGENT_NAMES["planner"], cfg["agent"])

    def test_an_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            build(role="critic", lane=LANE)

    def test_no_agent_rule_can_reopen_the_global_bash_denylist(self):
        """Agent permission is merged after the global rules, last match wins."""
        for role in aoc.AGENT_NAMES:
            cfg = build(role=role, lane=LANE)
            self.assertNotIn("permission", cfg)
            self.assertNotIn("plugin", cfg)
            for agent in cfg["agent"].values():
                self.assertNotIn("bash", agent["permission"])
                self.assertNotIn("tools", agent)   # deprecated key, use permission

    def test_the_seat_is_primary_and_step_capped(self):
        cfg = build(role="planner", lane=LANE, steps=33)
        agent = _agent(cfg, "planner")
        self.assertEqual((agent["mode"], agent["steps"]), ("primary", 33))
        self.assertNotIn("maxSteps", agent)

    def test_nonpositive_caps_are_refused(self):
        for kw in ({"steps": 0}, {"tool_output_max_lines": -1},
                   {"tool_output_max_bytes": True}):
            with self.assertRaises(ValueError):
                build(role="planner", lane=LANE, **kw)


class Discipline(unittest.TestCase):

    def test_the_prompt_steers_to_bounded_tools_and_a_final_json(self):
        prompt = _agent(build(role="planner", lane=LANE), "planner")["prompt"]
        for tool in ("outline", "read_range", "profile_top", "symbol_annotate", "grep"):
            self.assertIn(f"{aoc.MCP_SERVER}_{tool}", prompt)
        self.assertIn("200 lines", prompt)
        self.assertIn("JSON object as the LAST thing", prompt)

    def test_tool_output_is_capped_globally(self):
        cfg = build(role="author", lane=LANE, tool_output_max_lines=111,
                                     tool_output_max_bytes=2222)
        self.assertEqual(cfg["tool_output"], {"max_lines": 111, "max_bytes": 2222})


class FanOut(unittest.TestCase):

    def test_fan_out_on_enables_a_bounded_scout(self):
        cfg = build(role="planner", lane=LANE, fan_out=True)
        agent = _agent(cfg, "planner")
        self.assertEqual(agent["permission"]["task"], {"*": "deny", aoc.SCOUT_AGENT: "allow"})
        self.assertIn("`task` tool", agent["prompt"])
        self.assertIn(aoc.SCOUT_AGENT, agent["prompt"])
        self.assertIn(f"never more than {aoc.MAX_CONCURRENT_SUBAGENTS}", agent["prompt"])
        scout = cfg["agent"][aoc.SCOUT_AGENT]
        self.assertEqual(scout["mode"], "subagent")
        self.assertEqual(scout["permission"]["edit"], "deny")
        self.assertEqual(scout["permission"]["task"], "deny")
        self.assertNotIn("JSON object as the LAST thing", scout["prompt"])

    def test_fan_out_off_is_the_control_arm(self):
        on = build(role="author", lane=LANE, fan_out=True)
        off = build(role="author", lane=LANE, fan_out=False)
        agent = _agent(off, "author")
        self.assertEqual(agent["permission"]["task"], "deny")
        self.assertNotIn("`task` tool", agent["prompt"])
        self.assertNotIn(aoc.SCOUT_AGENT, off["agent"])
        # the discipline itself is identical across arms: only fan-out differs
        self.assertTrue(_agent(on, "author")["prompt"].startswith(agent["prompt"]))


class Mcp(unittest.TestCase):

    def test_profiles_thread_into_the_command(self):
        profiles = [Path("/p/a.perf.data"), Path("/p/b.perf.data")]
        cfg = build(role="planner", lane=LANE, profiles=profiles,
                                     python="/venv/bin/python",
                                     research_root=Path("/repo"))
        server = cfg["mcp"][aoc.MCP_SERVER]
        self.assertEqual(server["command"], [
            "/venv/bin/python", "-m", "scripts.kernel_rnd.autokernel.loop.actor_tools_mcp",
            "--root", str(LANE),
            "--profiles", "/p/a.perf.data", "--profiles", "/p/b.perf.data"])
        self.assertEqual((server["type"], server["enabled"]), ("local", True))
        self.assertEqual(server["cwd"], "/repo")
        self.assertEqual(server["environment"]["PYTHONPATH"].split(os.pathsep)[0], "/repo")

    def test_no_profiles_means_no_profile_flags(self):
        cfg = build(role="planner", lane=LANE)
        self.assertNotIn("--profiles", cfg["mcp"][aoc.MCP_SERVER]["command"])

    def test_the_default_root_is_this_repository(self):
        root = build(role="planner", lane=LANE)["mcp"][aoc.MCP_SERVER]["cwd"]
        module = Path(root, *aoc.MCP_MODULE.split(".")[:-1])
        self.assertTrue(module.joinpath("actor_opencode_config.py").is_file())


class Write(unittest.TestCase):

    def test_it_writes_valid_json_and_returns_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "opencode.json"
            got = aoc.write_actor_config(target, role="author", lane=LANE,
                                         profiles=[Path("/p/x")])
            self.assertEqual(got, target)
            loaded = json.loads(target.read_text())
            instructions = target.parent / "opencode.instructions.md"
            self.assertEqual(loaded, aoc.build_actor_config(role="author", lane=LANE,
                                                            profiles=[Path("/p/x")],
                                                            instructions_path=instructions))
            self.assertEqual(sorted(os.listdir(target.parent)),
                             ["opencode.instructions.md", "opencode.json"])

    def test_a_failed_write_leaves_the_old_file_and_no_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "opencode.json"
            target.write_text('{"old": true}')
            with mock.patch.object(aoc.json, "dump", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    aoc.write_actor_config(target, role="planner", lane=LANE,
                                           replace_system_prompt=True)
            self.assertEqual(json.loads(target.read_text()), {"old": True})
            self.assertEqual(os.listdir(tmp), ["opencode.json"])

    def test_an_unknown_role_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "opencode.json"
            with self.assertRaises(ValueError):
                aoc.write_actor_config(target, role="nope", lane=LANE)
            self.assertFalse(target.exists())
            self.assertEqual(os.listdir(tmp), [], "no stray instructions file either")


class SnapshotOff(unittest.TestCase):
    """Operator 2026-09-25: every per-call config the loop writes sets the top-level,
    SINGULAR `snapshot: false` (opencode 1.18.31 `ConfigV1.Info.snapshot`; `snapshots`
    is only the v2 internal name). opencode's per-step lane snapshots are what bloat
    opencode.db, and the reaper VACUUM that rewrites it killed DS41 run 9c's author."""

    KNOB_SETS = [dict(trim_instructions=a, trim_tools=b, lane_guard=c)
                 for a in (False, True) for b in (False, True) for c in (False, True)]

    def test_every_bounded_config_turns_snapshots_off(self):
        for role in aoc.AGENT_NAMES:
            for knobs in self.KNOB_SETS:
                for fan_out in (True, False):
                    for replace in (True, False):
                        cfg = aoc.build_actor_config(
                            role=role, lane=LANE, fan_out=fan_out, replace_system_prompt=replace,
                            instructions_path=None if replace else LANE.parent / "i.md", **knobs)
                        self.assertIs(cfg["snapshot"], False, (role, knobs, fan_out, replace))
                        self.assertNotIn("snapshots", cfg)

    def test_every_plain_config_turns_snapshots_off_and_is_never_none(self):
        for role in aoc.PLAIN_ROLES:
            for knobs in self.KNOB_SETS:
                cfg = aoc.build_plain_config(role=role, lane=LANE,
                                             author_note_path=LANE.parent / "n.md", **knobs)
                self.assertIsNotNone(cfg, (role, knobs))
                self.assertIs(cfg["snapshot"], False, (role, knobs))
                self.assertNotIn("snapshots", cfg)

    def test_knobs_off_plain_config_is_snapshot_off_and_nothing_else(self):
        for role in aoc.PLAIN_ROLES:
            # The critic (no `--auto`) always carries its never-ask block as well.
            extra = ({"permission": aoc.seat_permission("critic")} if role == "critic" else {})
            self.assertEqual(aoc.build_plain_config(role=role, lane=LANE),
                             {"$schema": "https://opencode.ai/config.json", "snapshot": False,
                              **extra})

    def test_write_plain_config_always_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for role in aoc.PLAIN_ROLES:
                path = aoc.write_plain_config(Path(tmp) / f"plain-{role}.json", role=role,
                                              lane=Path(tmp) / "lane")
                self.assertIsNotNone(path)
                self.assertIs(json.loads(path.read_text())["snapshot"], False)

    def test_write_actor_config_writes_snapshot_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = aoc.write_actor_config(Path(tmp) / "c.json", role="author", lane=LANE)
            self.assertIs(json.loads(path.read_text())["snapshot"], False)


if __name__ == "__main__":
    unittest.main()


class AppendNotReplace(unittest.TestCase):
    """v2 default: the seat's guidance is ADDED to opencode's system prompt via
    `instructions`; no agent `prompt` replaces it (seat A/B 2026-09-24: replacing it
    made the 27B decode ~6x more per step)."""

    def test_default_uses_instructions_and_sets_no_agent_prompt(self):
        cfg = aoc.build_actor_config(role="planner", lane=LANE, instructions_path=Path("/x/i.md"))
        self.assertEqual(cfg["instructions"], ["/x/i.md"])
        for agent in cfg["agent"].values():
            self.assertNotIn("prompt", agent)

    def test_default_without_an_instructions_path_refuses(self):
        with self.assertRaises(ValueError):
            aoc.build_actor_config(role="planner", lane=LANE)

    def test_v1_replace_keeps_the_agent_prompt_and_no_instructions(self):
        cfg = aoc.build_actor_config(role="planner", lane=LANE, replace_system_prompt=True)
        self.assertNotIn("instructions", cfg)
        self.assertIn("cannot edit", _agent(cfg, "planner")["prompt"])

    def test_the_instructions_file_carries_discipline_fan_out_and_a_scout_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "actor-opencode-planner.json"
            aoc.write_actor_config(target, role="planner", lane=LANE, fan_out=True)
            text = (Path(tmp) / "actor-opencode-planner.instructions.md").read_text()
            self.assertIn("cannot edit", text)
            self.assertIn("`task` tool", text)
            self.assertIn(f"If you are a {aoc.SCOUT_AGENT} subagent", text)
            cfg = json.loads(target.read_text())
            self.assertEqual(cfg["instructions"], [str(Path(tmp) / "actor-opencode-planner.instructions.md")])

    def test_fan_out_off_drops_the_fan_out_text(self):
        text = aoc.actor_instructions("author", fan_out=False)
        self.assertNotIn("`task` tool", text)
        self.assertNotIn(aoc.SCOUT_AGENT, text)
